// record_api.hpp — J-b1 P11b: the record API's export builders (design §4: record_report schema, frames() /
// plan_structure() dicts, signs() bulk array, record_layout entries, layout-3 accessors). Shared by the production
// binding (bound at F) and the test API (P11b, cpp/testapi/family_py.cpp). Pure builders: the binding extracts the
// engine data (plan views, layout-3 rows, record operators, classes, counters) and these assemble the exported
// objects, so both modules export byte-identical structures from the same data.
//
// Field contract (design §4):
//   * per row: prefix_words (P_rec, X words then Z words), plan_index (EXPORT-LOCAL: an index into this export's
//     per-plan arrays, first-appearance order; never the internal dense plan id), frame_folded, y (the raw record
//     values the state is projected on), outcomes (u then z, nuz bits);
//   * per plan (CSR: flat arrays plus int64 [P+1] offsets for every ragged field; J-b1 P12, fixplan-p12-frames (A)):
//     plan_key_digest uint8 [P, 16] (the layout-3 content handle's), kind int8 [P], term_index / term_offsets (the
//     plan's C0 factors as indices into the export's SHARED axis_table — a MULTISET: kind-0 plans repeat −Z_q),
//     rho / kappa_rec int32 [P], born_xz (x, z uint8 [B, n]) / born_phase int8 [B] / born_offsets (the Born
//     directions Λ_j as physical Paulis), record_layout;
//   * axis_table = (x uint8 [A, n], z uint8 [A, n], phase int8 [A]): the distinct axes this export uses, first-seen
//     order, deduplicated by (x, z, phase) — +Z_q and −Z_q are different rotations;
//   * a uint8 [P, n] and cz_pairs int32 [E, 2] / cz_offsets only when EVERY plan of the export is kind 0 (otherwise
//     absent: KeyError); σ-space fields (r, kappa, kernel_xz, kernel_phase, fallback) and the P11b names `terms` /
//     `term_axes` (their meaning changed) are never exported;
//   * the record operators M_c (record_xz / record_phase) the state is projected on.
// The state of row i is P_rec · C0 |B⟩ projected onto the records at y, C0 = Π_{t ∈ term_index[k]} exp(iπ/4 A_t).
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "qeccore/pauli.hpp"
#include "qeccore/twirl_sampler.hpp"

namespace xtim_api {
namespace py = pybind11;

// J-b1 P12: the Born-measured decision operators of a sampler as PURE DATA (the b9c2e92 BarrierBuffer.born_dec format:
// {nborn, n, dec_index int32[nborn], xz uint8[nborn, 2n], phase uint8[nborn], inv uint8[nborn]}). The production
// binding's born_dec_dict (xtim_py.cpp) is this format; it moves here at F (one owner).
inline py::dict born_dec_dict(const std::vector<qeccore::TwirlRecordSampler::BornDecOp>& ops) {
    const int nb = (int)ops.size();
    const int N = nb ? ops[0].op.n : 0;
    py::array_t<int32_t> idx((py::ssize_t)nb);
    py::array_t<uint8_t> xz({(py::ssize_t)nb, (py::ssize_t)(2 * N)});
    py::array_t<uint8_t> ph((py::ssize_t)nb);
    py::array_t<uint8_t> inv((py::ssize_t)nb);
    for (int j = 0; j < nb; ++j) {
        const auto& b = ops[(size_t)j];
        idx.mutable_data()[j] = (int32_t)b.dec_index;
        uint8_t* row = xz.mutable_data() + (size_t)j * (size_t)(2 * N);
        for (int q = 0; q < N; ++q) { row[q] = b.op.xbit(q) ? 1 : 0; row[N + q] = b.op.zbit(q) ? 1 : 0; }
        ph.mutable_data()[j] = (uint8_t)(((b.op.phase % 4) + 4) % 4);
        inv.mutable_data()[j] = b.inv;
    }
    py::dict res;
    res["nborn"] = nb;
    res["n"] = N;
    res["dec_index"] = idx;
    res["xz"] = xz;
    res["phase"] = ph;
    res["inv"] = inv;
    return res;
}

constexpr int kRecordKeyLayout = 3;

// The shared axis table of one export: distinct axes in first-seen order, keyed by their sparse form (qubit, X / Z /
// Y code) plus the phase. Built without heap Paulis for kind 0 (single −Z_q, +Z_qZ_r).
struct AxisTable {
    int n = 0;
    std::vector<std::vector<std::pair<int, uint8_t>>> sparse;   // (qubit, code: 1 X, 2 Z, 3 Y)
    std::vector<int8_t> phase;
    std::unordered_map<std::string, int32_t> ix;
    explicit AxisTable(int n_) : n(n_) {}
    int32_t add(std::vector<std::pair<int, uint8_t>> sp, int ph) {
        std::string key;
#ifndef XTIM_MUTATE_AXTAB_NOPHASE
        key.push_back((char)(ph & 3));
#endif
        for (const auto& e : sp) { key.append(reinterpret_cast<const char*>(&e.first), sizeof(int)); key.push_back((char)e.second); }
        auto it = ix.find(key);
        if (it != ix.end()) return it->second;
        const int32_t id = (int32_t)sparse.size();
        sparse.push_back(std::move(sp));
        phase.push_back((int8_t)(ph & 3));
        ix.emplace(std::move(key), id);
        return id;
    }
    int32_t z1(int q, int ph) { return add({{q, (uint8_t)2}}, ph); }
    int32_t zz(int q, int r, int ph) {
        if (q > r) std::swap(q, r);
        return add({{q, (uint8_t)2}, {r, (uint8_t)2}}, ph);
    }
    int32_t pauli(const qeccore::Pauli& P) {
        std::vector<std::pair<int, uint8_t>> sp;
        for (size_t w = 0; w < P.x.size(); ++w)
            for (uint64_t m = P.x[w] | P.z[w]; m; m &= m - 1) {
                const int b = __builtin_ctzll(m);
                const int q = (int)(w * 64 + (size_t)b);
                sp.push_back({q, (uint8_t)(((P.x[w] >> b) & 1) | (((P.z[w] >> b) & 1) << 1))});
            }
        return add(std::move(sp), P.phase);
    }
    py::tuple arrays() const {
        const py::ssize_t A = (py::ssize_t)sparse.size();
        py::array_t<uint8_t> x({A, (py::ssize_t)n}), z({A, (py::ssize_t)n});
        py::array_t<int8_t> ph(A);
        if (A && n) { std::memset(x.mutable_data(), 0, (size_t)A * n); std::memset(z.mutable_data(), 0, (size_t)A * n); }
        for (py::ssize_t a = 0; a < A; ++a) {
            for (const auto& e : sparse[(size_t)a]) {
                if (e.second & 1) x.mutable_data()[(size_t)a * n + e.first] = 1;
                if (e.second & 2) z.mutable_data()[(size_t)a * n + e.first] = 1;
            }
            ph.mutable_data()[a] = phase[(size_t)a];
        }
        return py::make_tuple(x, z, ph);
    }
};

// One plan (a record content) as exported.
struct PlanExport {
    int kind = 0;
    std::string content;
    uint64_t digest_hi = 0, digest_lo = 0;
    std::vector<int32_t> term_index;                 // indices into the export's AxisTable (a multiset)
    int rho = 0, kappa = 0;
    std::vector<qeccore::Pauli> born;
    std::vector<uint8_t> a;                          // kind 0 only
    std::vector<std::pair<int, int>> cz;             // kind 0 only
    long class_id = 0;                               // plancache-design: the plan's class (FamilyPlanView::class_id)
};

// The kind-0 C0 = S^{a&1}·CZ as axis indices, diag_plan_axes' order (−Z_q per a_q; per CZ (f, s): −Z_f, −Z_s,
// +Z_f Z_s). MUT-TERMS-SET (in-plan dedup) is the mutation the multiset gate catches.
inline void kind0_term_index(AxisTable& T, const std::vector<uint8_t>& a, const std::vector<std::pair<int, int>>& cz,
                             std::vector<int32_t>& out) {
    out.clear();
    auto push = [&](int32_t id) {
#ifdef XTIM_MUTATE_TERMS_SET
        if (std::find(out.begin(), out.end(), id) != out.end()) return;   // mutation build only: a set, not a multiset
#endif
        out.push_back(id);
    };
    for (int q = 0; q < (int)a.size(); ++q) if (a[(size_t)q] & 1) push(T.z1(q, 2));
    for (const auto& pr : cz) { push(T.z1(pr.first, 2)); push(T.z1(pr.second, 2)); push(T.zz(pr.first, pr.second, 0)); }
}

// Layout-3 rows (borrowed; the binding owns the storage). plan[i] is the export-local plan index of row i.
struct RecordRows {
    long rows = 0;
    int n = 0, m = 0, PNW = 0, YW = 0, UZW = 0;
    const int32_t* plan = nullptr;
    const uint64_t* prec = nullptr;
    const uint64_t* y = nullptr;
    const uint8_t* uz = nullptr;
    const int32_t* nuz = nullptr;
    const uint8_t* ff = nullptr;
};

inline py::bytes digest_bytes(uint64_t hi, uint64_t lo) {
    char b[16];
    std::memcpy(b, &hi, 8);
    std::memcpy(b + 8, &lo, 8);
    return py::bytes(b, 16);
}

// (x uint8[k, n], z uint8[k, n], phase int8[k]) of a Pauli list.
inline py::tuple pauli_arrays(const std::vector<qeccore::Pauli>& ps, int n) {
    const py::ssize_t k = (py::ssize_t)ps.size();
    py::array_t<uint8_t> x({k, (py::ssize_t)n}), z({k, (py::ssize_t)n});
    py::array_t<int8_t> ph(k);
    if (k && n) {
        std::memset(x.mutable_data(), 0, (size_t)k * (size_t)n);
        std::memset(z.mutable_data(), 0, (size_t)k * (size_t)n);
    }
    for (py::ssize_t i = 0; i < k; ++i) {
        const qeccore::Pauli& P = ps[(size_t)i];
        for (size_t w = 0; w < P.x.size(); ++w)
            for (uint64_t m = P.x[w] | P.z[w]; m; m &= m - 1) {
                const int b = __builtin_ctzll(m);
                const size_t q = w * 64 + (size_t)b;
                if (q >= (size_t)n) break;
                x.mutable_data()[(size_t)i * n + q] = (uint8_t)((P.x[w] >> b) & 1);
                z.mutable_data()[(size_t)i * n + q] = (uint8_t)((P.z[w] >> b) & 1);
            }
        ph.mutable_data()[i] = (int8_t)(P.phase & 3);
    }
    return py::make_tuple(x, z, ph);
}

// The layout-3 plan content: [kind u8][len u32 LE] content (the key's leading bytes).
inline py::bytes plan_content_bytes(int kind, const std::string& content) {
    std::string k;
    k.push_back((char)kind);
    const uint32_t L = (uint32_t)content.size();
    for (int b = 0; b < 4; ++b) k.push_back((char)((L >> (8 * b)) & 0xFF));
    k += content;
    return py::bytes(k);
}

// record_layout(): the layout-3 key and the outcome order.
inline py::dict record_layout_dict(int n, int m, int PNW, int YW) {
    py::dict d;
    d["layout"] = kRecordKeyLayout;
    d["key"] = py::make_tuple("kind u8", "content length u32 LE", "content", "P_rec words: X then Z, u64 LE",
                              "y words, u64 LE");
    d["n"] = n;
    d["records"] = m;
    d["prefix_words"] = 2 * PNW;
    d["y_words"] = YW;
    d["outcomes"] = py::make_tuple("u: rho coin bits (pivot order)", "z: kappa_rec Born bits (direction order)");
    return d;
}

// The per-plan fields of an export (CSR), shared by frames() and plan_structure().
inline void plans_into(py::dict& d, const std::vector<PlanExport>& plans, const AxisTable& T, int n) {
    const py::ssize_t P = (py::ssize_t)plans.size();
    py::array_t<uint8_t> dig({P, (py::ssize_t)16});
    py::array_t<int8_t> kind(P);
    py::array_t<int32_t> rho(P), kap(P);
    py::array_t<int64_t> toff(P + 1), boff(P + 1), czoff(P + 1);
    size_t nt = 0, nb = 0, ncz = 0;
    bool all_k0 = true;
    for (const PlanExport& p : plans) { nt += p.term_index.size(); nb += p.born.size(); ncz += p.cz.size(); all_k0 = all_k0 && p.kind == 0; }
    py::array_t<int32_t> tix((py::ssize_t)nt);
    std::vector<qeccore::Pauli> born;
    born.reserve(nb);
    py::array_t<int32_t> czp({(py::ssize_t)ncz, (py::ssize_t)2});
    size_t t = 0, c = 0, bb = 0;
    toff.mutable_data()[0] = boff.mutable_data()[0] = czoff.mutable_data()[0] = 0;
    for (py::ssize_t k = 0; k < P; ++k) {
        const PlanExport& p = plans[(size_t)k];
        std::memcpy(dig.mutable_data() + (size_t)k * 16, &p.digest_hi, 8);
        std::memcpy(dig.mutable_data() + (size_t)k * 16 + 8, &p.digest_lo, 8);
        kind.mutable_data()[k] = (int8_t)p.kind;
        rho.mutable_data()[k] = p.rho;
        kap.mutable_data()[k] = p.kappa;
        for (int32_t v : p.term_index) tix.mutable_data()[t++] = v;
        toff.mutable_data()[k + 1] = (int64_t)t;
        for (const auto& L : p.born) born.push_back(L);
        bb += p.born.size();
        boff.mutable_data()[k + 1] = (int64_t)bb;
        for (const auto& pr : p.cz) { czp.mutable_data()[2 * c] = pr.first; czp.mutable_data()[2 * c + 1] = pr.second; ++c; }
        czoff.mutable_data()[k + 1] = (int64_t)c;
    }
    d["plan_key_digest"] = dig;
    d["kind"] = kind;
    d["term_index"] = tix;
    d["term_offsets"] = toff;
    d["axis_table"] = T.arrays();
    d["rho"] = rho;
    d["kappa_rec"] = kap;
    const py::tuple b = pauli_arrays(born, n);
    d["born_xz"] = py::make_tuple(b[0], b[1]);
    d["born_phase"] = b[2];
    d["born_offsets"] = boff;
    if (all_k0) {                                  // the S^a·CZ view only when every plan has one
        py::array_t<uint8_t> a({P, (py::ssize_t)n});
        if (P && n) std::memset(a.mutable_data(), 0, (size_t)P * n);
        for (py::ssize_t k = 0; k < P; ++k)
            for (int q = 0; q < n && q < (int)plans[(size_t)k].a.size(); ++q) a.mutable_data()[(size_t)k * n + q] = plans[(size_t)k].a[(size_t)q];
        d["a"] = a;
        d["cz_pairs"] = czp;
        d["cz_offsets"] = czoff;
    }
}

// plan_structure(): one plan's fields (the same CSR fields with P = 1, over its own axis table).
inline py::dict plan_structure_dict(const PlanExport& p, const AxisTable& T, int n) {
    py::dict d;
    plans_into(d, std::vector<PlanExport>{p}, T, n);
    d["plan_content"] = plan_content_bytes(p.kind, p.content);
    d["class_id"] = p.class_id;                      // plancache-design: kind 0 its class (0: unclassed), else −arena id
    py::dict lay;
    lay["u"] = py::make_tuple(0, p.rho);
    lay["z"] = py::make_tuple(p.rho, p.rho + p.kappa);
    d["record_layout"] = lay;
    return d;
}

// FFS: plan_structure()'s inputs (one plan export, its axis table, n) as one compact byte string — the port's
// plan_structure memo holds this, ~16x smaller than the dict (cult_d5: ~0.8 KB vs ~13 KB): the Born directions as
// Pauli words (bit-packed rows), the axes sparse — and plan_structure_unpack rebuilds THE dict with plan_structure_dict itself, so every key, dtype,
// shape and value is the engine's by construction. Native byte order; read only by this extension (bounds-checked).
namespace ps_blob_detail {
constexpr uint32_t kMagic = 0x31535046u;            // "FPS1"
template <class T> void put(std::string& s, T v) { s.append(reinterpret_cast<const char*>(&v), sizeof(T)); }
struct Reader {
    const std::string& s;
    size_t pos = 0;
    template <class T> T get() {
        if (s.size() - pos < sizeof(T)) throw std::invalid_argument("plan_structure blob: truncated");
        T v;
        std::memcpy(&v, s.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    uint32_t count(size_t unit) {                   // a length prefix whose payload must fit in what is left
        const uint32_t k = get<uint32_t>();
        if (unit && (size_t)k > (s.size() - pos) / unit) throw std::invalid_argument("plan_structure blob: truncated");
        return k;
    }
};
}  // namespace ps_blob_detail

inline std::string plan_structure_blob(const PlanExport& p, const AxisTable& T, int n) {
    using ps_blob_detail::put;
    std::string s;
    put<uint32_t>(s, ps_blob_detail::kMagic);
    put<int32_t>(s, n); put<int32_t>(s, p.kind); put<int32_t>(s, p.rho); put<int32_t>(s, p.kappa);
    put<int64_t>(s, (int64_t)p.class_id); put<uint64_t>(s, p.digest_hi); put<uint64_t>(s, p.digest_lo);
    put<uint32_t>(s, (uint32_t)p.content.size()); s += p.content;
    put<uint32_t>(s, (uint32_t)p.term_index.size());
    for (int32_t v : p.term_index) put<int32_t>(s, v);
    put<uint32_t>(s, (uint32_t)p.born.size());
    for (const qeccore::Pauli& P : p.born) {
        put<int32_t>(s, P.n); put<int32_t>(s, P.phase); put<uint32_t>(s, (uint32_t)P.x.size());
        for (uint64_t w : P.x) put<uint64_t>(s, w);
        for (uint64_t w : P.z) put<uint64_t>(s, w);
    }
    put<uint32_t>(s, (uint32_t)p.a.size()); s.append(reinterpret_cast<const char*>(p.a.data()), p.a.size());
    put<uint32_t>(s, (uint32_t)p.cz.size());
    for (const auto& pr : p.cz) { put<int32_t>(s, pr.first); put<int32_t>(s, pr.second); }
    put<int32_t>(s, T.n);
    put<uint32_t>(s, (uint32_t)T.sparse.size());
    for (size_t a = 0; a < T.sparse.size(); ++a) {
        put<int8_t>(s, T.phase[a]);
        put<uint32_t>(s, (uint32_t)T.sparse[a].size());
        for (const auto& e : T.sparse[a]) { put<int32_t>(s, e.first); put<uint8_t>(s, e.second); }
    }
    return s;
}

inline py::dict plan_structure_unpack(const std::string& s) {
    ps_blob_detail::Reader r{s};
    if (r.get<uint32_t>() != ps_blob_detail::kMagic) throw std::invalid_argument("plan_structure blob: bad magic");
    PlanExport p;
    const int n = r.get<int32_t>();
    p.kind = r.get<int32_t>(); p.rho = r.get<int32_t>(); p.kappa = r.get<int32_t>();
    p.class_id = (long)r.get<int64_t>(); p.digest_hi = r.get<uint64_t>(); p.digest_lo = r.get<uint64_t>();
    {
        const uint32_t L = r.count(1);
        p.content.assign(s.data() + r.pos, L);
        r.pos += L;
    }
    p.term_index.resize(r.count(4));
    for (auto& v : p.term_index) v = r.get<int32_t>();
    p.born.resize(r.count(12));
    for (qeccore::Pauli& P : p.born) {
        P.n = r.get<int32_t>(); P.phase = r.get<int32_t>();
        const uint32_t W = r.count(16);
        P.x.resize(W); P.z.resize(W);
        for (auto& w : P.x) w = r.get<uint64_t>();
        for (auto& w : P.z) w = r.get<uint64_t>();
    }
    {
        const uint32_t L = r.count(1);
        p.a.assign(s.data() + r.pos, s.data() + r.pos + L);
        r.pos += L;
    }
    p.cz.resize(r.count(8));
    for (auto& pr : p.cz) { pr.first = r.get<int32_t>(); pr.second = r.get<int32_t>(); }
    AxisTable T(r.get<int32_t>());
    const uint32_t A = r.count(5);
    T.sparse.resize(A);
    T.phase.resize(A);
    for (uint32_t a = 0; a < A; ++a) {
        T.phase[a] = r.get<int8_t>();
        T.sparse[a].resize(r.count(5));
        for (auto& e : T.sparse[a]) {
            e.first = r.get<int32_t>();
            e.second = r.get<uint8_t>();
            if (e.first < 0 || e.first >= T.n) throw std::invalid_argument("plan_structure blob: axis qubit out of range");
        }
    }
    if (r.pos != s.size()) throw std::invalid_argument("plan_structure blob: trailing bytes");
    if (n < 0 || T.n != n) throw std::invalid_argument("plan_structure blob: qubit count mismatch");
    if (p.a.size() > (size_t)n) throw std::invalid_argument("plan_structure blob: a longer than n");
    return plan_structure_dict(p, T, n);
}

// frames(): the rows and the per-plan arrays.
inline py::dict frames_dict(const RecordRows& R, const std::vector<PlanExport>& plans, const AxisTable& T,
                            const std::vector<qeccore::Pauli>& record_ops) {
    const py::ssize_t rows = (py::ssize_t)R.rows, W = 2 * (py::ssize_t)R.PNW;
    py::array_t<uint64_t> words({rows, W});
    py::array_t<int32_t> pidx(rows);
    py::array_t<uint8_t> ff(rows), y({rows, (py::ssize_t)R.m}), out({rows, (py::ssize_t)R.UZW});
    py::array_t<int32_t> nuz(rows);
    if (rows) {
        std::memcpy(words.mutable_data(), R.prec, (size_t)rows * (size_t)W * sizeof(uint64_t));
        std::memcpy(pidx.mutable_data(), R.plan, (size_t)rows * sizeof(int32_t));
        std::memcpy(out.mutable_data(), R.uz, (size_t)rows * (size_t)R.UZW);
        std::memcpy(nuz.mutable_data(), R.nuz, (size_t)rows * sizeof(int32_t));
        if (R.ff) std::memcpy(ff.mutable_data(), R.ff, (size_t)rows);
        else std::memset(ff.mutable_data(), 0, (size_t)rows);
        if (R.m) std::memset(y.mutable_data(), 0, (size_t)rows * R.m);
    }
    for (py::ssize_t i = 0; i < rows; ++i) {
        const uint64_t* yr = R.y + (size_t)i * (size_t)R.YW;
        uint8_t* yo = y.mutable_data() + (size_t)i * R.m;
        for (int w = 0; w < R.YW; ++w)
            for (uint64_t m = yr[w]; m; m &= m - 1) {
                const int c = w * 64 + __builtin_ctzll(m);
                if (c < R.m) yo[c] = 1;
            }
    }
    py::dict d;
    d["prefix_words"] = words;
    d["plan_index"] = pidx;
    d["frame_folded"] = ff;
    d["y"] = y;
    d["outcomes"] = out;
    d["n_outcomes"] = nuz;
    plans_into(d, plans, T, R.n);
    const py::tuple ro = pauli_arrays(record_ops, R.n);
    d["record_xz"] = py::make_tuple(ro[0], ro[1]);
    d["record_phase"] = ro[2];
    d["record_layout"] = record_layout_dict(R.n, R.m, R.PNW, R.YW);
    return d;
}

// record_report(): classes (the R = I classification) per detector / observable / decision, the nondeterministic
// lists, route counts and the plan / memo / miss counters. cls[c] ∈ {0 deterministic, 1 coin, 2 born}; order[c] =
// (kind << 24) | index with kind 0 detector, 1 observable, 2 decision.
inline py::dict record_report_dict(const std::vector<int>& cls, const std::vector<int>& order,
                                   const std::map<std::string, long>& route_counts,
                                   const std::map<std::string, long>& counters) {
    static const char* names[] = {"deterministic", "coin", "born"};
    py::dict dets, obs, dec;
    std::vector<int> nd_det, nd_obs;
    for (size_t c = 0; c < cls.size() && c < order.size(); ++c) {
        const int kind = order[c] >> 24, idx = order[c] & ((1 << 24) - 1);
        py::dict& d = kind == 0 ? dets : kind == 1 ? obs : dec;
        d[py::int_(idx)] = names[cls[c] < 0 || cls[c] > 2 ? 0 : cls[c]];
        if (cls[c] != 0 && kind == 0) nd_det.push_back(idx);
        if (cls[c] != 0 && kind == 1) nd_obs.push_back(idx);
    }
    std::sort(nd_det.begin(), nd_det.end());
    std::sort(nd_obs.begin(), nd_obs.end());
    py::dict r, classes;
    classes["detectors"] = dets;
    classes["observables"] = obs;
    classes["decisions"] = dec;
    r["classes"] = classes;
    r["nondeterministic_detectors"] = nd_det;
    r["nondeterministic_observables"] = nd_obs;
    py::dict rc, ct;
    for (const auto& kv : route_counts) rc[py::str(kv.first)] = kv.second;
    for (const auto& kv : counters) ct[py::str(kv.first)] = kv.second;
    r["route_counts"] = rc;
    r["counters"] = ct;
    return r;
}

// signs(): the bulk int8[rows, k] array a signs core fills (+1 / -1 definite, 0 indefinite or not certified).
inline py::array_t<int8_t> signs_array(long rows, size_t k) {
    py::array_t<int8_t> a({(py::ssize_t)rows, (py::ssize_t)k});
    if (rows && k) std::memset(a.mutable_data(), 0, (size_t)rows * k);
    return a;
}

}  // namespace xtim_api
