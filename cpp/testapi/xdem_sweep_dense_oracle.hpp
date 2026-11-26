// xdem_sweep_dense_oracle.hpp — TEST ONLY (LM-3): the backward sweep of the xdem writer as it was at 2e89d05 (dense
// wires x columns bit planes, the X plane kept whole as x_start and cut to the readout words), with the two builders
// on it (build_fault_columns, build_tau_columns), kept verbatim as the oracle the production sweep (sparse per-wire
// column lists with a density switch to these planes, xdem_writer.cpp) is compared against: the noise slots' column
// lists, x_start and the tau columns (tests/test_lm3_sparse_sweep.py through the test API, cpp/tests/test_xdem_lean.cpp).
// Header-only so it lives in the test API and the C++ tests, never in the shipped _xtim.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "qeccore/xdem_writer.hpp"

namespace qeccore::xdem::test_oracle {

using SweepAtDense = std::vector<std::vector<std::array<std::vector<int32_t>, 2>>>;

inline SweepAtDense backward_sweep_dense(const Skeleton& sk, const Circuit& def, int C, const std::vector<Pauli>* end,
                                         const std::vector<std::vector<int>>* read_cols, int loc_base,
                                         std::vector<uint64_t>* x_start = nullptr, size_t* cw_out = nullptr) {
    if (sk.op_of_stream.size() != def.stream.size() + 1)
        throw WriterError("xdem writer: internal pipeline mismatch: the skeleton is not of this deferred stream");
    const size_t CW = (size_t)std::max(1, (C + 63) / 64);
    const int N = sk.N;
    std::vector<uint64_t> X((size_t)N * CW, 0), Z((size_t)N * CW, 0);
    auto xr = [&](int w) { return X.data() + (size_t)w * CW; };
    auto zr = [&](int w) { return Z.data() + (size_t)w * CW; };
    if (end)
        for (size_t c = 0; c < end->size(); ++c) {
            const Pauli& Q = (*end)[c];
            const uint64_t bit = 1ull << (c & 63);
            for (int w = 0; w < N; ++w) {
                if (Q.xbit(w)) xr(w)[c >> 6] |= bit;
                if (Q.zbit(w)) zr(w)[c >> 6] |= bit;
            }
        }
    std::vector<int> lq_off;                          // group -> its first location column - loc_base
    if (loc_base >= 0) {
        lq_off.assign(sk.group_qubits.size() + 1, 0);
        for (size_t g = 0; g < sk.group_qubits.size(); ++g)
            lq_off[g + 1] = lq_off[g] + (int)sk.group_qubits[g].size();
    }
    std::vector<int> noise_ord(def.stream.size(), -1), read_ord(def.stream.size(), -1);
    int nn = 0, nr = 0;
    for (size_t i = 0; i < def.stream.size(); ++i) {
        if (def.stream[i].kind == Instr::Kind::Noise) noise_ord[i] = nn++;
        else if (def.stream[i].kind == Instr::Kind::Measure) read_ord[i] = nr++;
    }
    if (nr != (int)sk.reads.size())
        throw WriterError("xdem writer: internal pipeline mismatch: Measure count != terminal reads");
    SweepAtDense at((size_t)nn);
    auto bwd_op = [&](const Skeleton::Op& o) {
        switch (o.kind) {
            case Skeleton::H: {
                uint64_t* x = xr(o.a); uint64_t* z = zr(o.a);
                for (size_t w = 0; w < CW; ++w) std::swap(x[w], z[w]);
                break;
            }
            case Skeleton::S:
            case Skeleton::SDG: {
                const uint64_t* x = xr(o.a); uint64_t* z = zr(o.a);
                for (size_t w = 0; w < CW; ++w) z[w] ^= x[w];
                break;
            }
            case Skeleton::CX: {
                const uint64_t* xa = xr(o.a); uint64_t* xb = xr(o.b);
                uint64_t* za = zr(o.a); const uint64_t* zb = zr(o.b);
                for (size_t w = 0; w < CW; ++w) { xb[w] ^= xa[w]; za[w] ^= zb[w]; }
                break;
            }
            case Skeleton::CZ: {
                const uint64_t* xa = xr(o.a); const uint64_t* xb = xr(o.b);
                uint64_t* za = zr(o.a); uint64_t* zb = zr(o.b);
                for (size_t w = 0; w < CW; ++w) { za[w] ^= xb[w]; zb[w] ^= xa[w]; }
                break;
            }
            case Skeleton::LOC: {
                if (loc_base < 0) break;
                const int g = o.a;
                const auto& qs = sk.group_qubits[(size_t)g];
                for (int i = 0; i < (int)qs.size(); ++i) {
                    const int c = loc_base + lq_off[(size_t)g] + i;
                    zr(qs[(size_t)i])[(size_t)c >> 6] ^= 1ull << (c & 63);
                }
                break;
            }
            default: break;
        }
    };
    auto sparse = [&](const uint64_t* row) {
        std::vector<int32_t> v;
        for (size_t w = 0; w < CW; ++w)
            for (uint64_t b = row[w]; b; b &= b - 1) v.push_back((int32_t)(w * 64 + (size_t)__builtin_ctzll(b)));
        return v;
    };
    for (size_t i = def.stream.size(); i-- > 0;) {
        for (int oi = sk.op_of_stream[i + 1]; oi-- > sk.op_of_stream[i];) bwd_op(sk.ops[(size_t)oi]);
        const Instr& ins = def.stream[i];
        if (read_ord[i] >= 0) {
            if (!read_cols) continue;
            const auto& [b, w] = sk.reads[(size_t)read_ord[i]];
            for (int c : (*read_cols)[(size_t)read_ord[i]]) {
                const uint64_t bit = 1ull << (c & 63);
                if (b == 0 || b == 1) xr(w)[(size_t)c >> 6] ^= bit;
                if (b == 2 || b == 1) zr(w)[(size_t)c >> 6] ^= bit;
            }
        } else if (noise_ord[i] >= 0) {
            auto& slot = at[(size_t)noise_ord[i]];
            slot.resize(ins.qubits.size());
            for (size_t qi = 0; qi < ins.qubits.size(); ++qi) {
                const int q = ins.qubits[qi];
                slot[qi][0] = sparse(zr(q));
                slot[qi][1] = sparse(xr(q));
            }
        }
    }
    if (x_start) *x_start = std::move(X);
    if (cw_out) *cw_out = CW;
    return at;
}

// build_fault_columns at 2e89d05 (the readout words of x_start kept, wire-major, cw words per wire).
struct FaultColumnsDense {
    int ncols = 0;
    std::vector<std::pair<int, int>> xid;
    SweepAtDense at;
    std::vector<uint64_t> x_start; size_t cw = 0;
};

inline FaultColumnsDense build_fault_columns_dense(const Skeleton& sk, const Circuit& def,
                                                   const std::vector<std::vector<int>>& col_recs) {
    FaultColumnsDense fc;
    fc.ncols = (int)col_recs.size();
    int nloc = 0;
    for (int g = 0; g < (int)sk.group_qubits.size(); ++g)
        for (int i = 0; i < (int)sk.group_qubits[(size_t)g].size(); ++i, ++nloc) fc.xid.push_back({g, i});
    std::vector<std::vector<int>> read_cols(sk.reads.size());
    for (int c = 0; c < fc.ncols; ++c) {
        std::vector<int32_t> js(col_recs[(size_t)c].begin(), col_recs[(size_t)c].end());
        parity_reduce_sorted(js);
        for (int j : js) {
            if (j < 0 || j >= (int)sk.reads.size())
                throw WriterError("xdem writer: a column references an out-of-range measurement");
            read_cols[(size_t)j].push_back(c);
        }
    }
    fc.at = backward_sweep_dense(sk, def, fc.ncols + nloc, nullptr, &read_cols, fc.ncols, &fc.x_start, &fc.cw);
    const size_t kw = (size_t)std::max(1, (fc.ncols + 63) / 64);
    if (kw < fc.cw) {
        std::vector<uint64_t> xs((size_t)sk.N * kw);
        for (size_t w = 0; w < (size_t)sk.N; ++w)
            std::copy_n(fc.x_start.begin() + (std::ptrdiff_t)(w * fc.cw), kw, xs.begin() + (std::ptrdiff_t)(w * kw));
        fc.x_start = std::move(xs);
        fc.cw = kw;
    }
    return fc;
}

inline SweepAtDense build_tau_columns_dense(const Skeleton& sk, const Circuit& def) {
    return backward_sweep_dense(sk, def, (int)sk.axis_q.size(), &sk.axis_q, nullptr, -1);
}

}  // namespace qeccore::xdem::test_oracle
