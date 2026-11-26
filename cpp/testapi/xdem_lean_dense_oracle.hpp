// xdem_lean_dense_oracle.hpp — TEST ONLY (LM-2): the dense form of xdem::nondeterministic_columns as it was at 9fb442f
// (a bit gather over every column x every wire, every axis per column), kept verbatim as the oracle the sparse
// production form is compared against (tests/test_lm2_lean_nondet.py through the test API, cpp/tests/test_xdem_lean.cpp).
// Header-only so it lives in the test API and the C++ tests, never in the shipped _xtim.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qeccore/xdem_writer.hpp"

namespace qeccore::xdem::test_oracle {

inline std::vector<int> nondeterministic_columns_dense(const WriterModel& wm, const FaultColumns& fc) {
    const Skeleton& sk = *wm.sk;
    const Header& H = wm.header;
    const int A = (int)H.axes.size(), N = sk.N, r = H.r, T = H.D + H.O;
    if ((int)sk.axis_p0.size() != A || fc.x_start.size() != (size_t)N * fc.cw || fc.ncols < T)
        throw std::logic_error("xdem nondeterministic_columns: writer model / fault columns mismatch (engine bug)");
    auto getb = [](const std::vector<uint64_t>& w, size_t i) { return ((w[i >> 6] >> (i & 63)) & 1ull) != 0; };
    std::vector<int> n((size_t)A, 0);                          // n_a mod 8
    for (const Group& gr : H.groups)
        for (size_t vi = 0; vi < gr.v_axis.size(); ++vi) {
            const auto [a, s] = gr.v_axis[vi];
            if (a < 0) continue;                               // an identity axis: a global phase
            n[(size_t)a] = ((n[(size_t)a] + s * gate_rotation_count(gr.gate, (uint32_t)(vi + 1))) % 8 + 8) % 8;
        }
    // Fully reduced echelon of X(U^dag Q_a U) (N bits), pivot = lowest set bit, with combinations (A bits).
    const size_t NW = (size_t)(N + 63) / 64, AW = (size_t)A / 64 + 1, RW = std::max<size_t>(1, words_for(r));
    std::vector<std::vector<uint64_t>> rows, comb;
    std::vector<int> piv;
    auto reduce = [&](std::vector<uint64_t>& v, std::vector<uint64_t>& c) {
        for (size_t i = 0; i < rows.size(); ++i)
            if (getb(v, (size_t)piv[i])) {
                for (size_t w = 0; w < NW; ++w) v[w] ^= rows[i][w];
                for (size_t w = 0; w < AW; ++w) c[w] ^= comb[i][w];
            }
    };
    for (int a = 0; a < A; ++a) {
        std::vector<uint64_t> v = sk.axis_p0[(size_t)a].x, c(AW, 0);
        v.resize(NW, 0);
        c[(size_t)a >> 6] |= 1ull << (a & 63);
        reduce(v, c);
        int h = -1;
        for (size_t w = 0; w < NW && h < 0; ++w) if (v[w]) h = (int)(w * 64 + (size_t)__builtin_ctzll(v[w]));
        if (h < 0) continue;
        for (size_t i = 0; i < rows.size(); ++i)
            if (getb(rows[i], (size_t)h)) {
                for (size_t w = 0; w < NW; ++w) rows[i][w] ^= v[w];
                for (size_t w = 0; w < AW; ++w) comb[i][w] ^= c[w];
            }
        rows.push_back(std::move(v)); comb.push_back(std::move(c)); piv.push_back(h);
    }
    std::vector<int> bad;
    std::vector<uint64_t> x(NW), c(AW), kB(RW);
    std::vector<uint8_t> alpha((size_t)r);
    std::vector<std::vector<uint64_t>> beta((size_t)r, std::vector<uint64_t>(RW));   // zero between columns
    std::vector<int> brows;                                    // the rows of beta this column touched
    for (int t = 0; t < T; ++t) {
        std::fill(x.begin(), x.end(), 0); std::fill(c.begin(), c.end(), 0);
        for (int w = 0; w < N; ++w)
            if ((fc.x_start[(size_t)w * fc.cw + ((size_t)t >> 6)] >> (t & 63)) & 1ull) x[(size_t)w >> 6] |= 1ull << (w & 63);
        reduce(x, c);
        bool nz = false;
        for (uint64_t w : x) nz |= w != 0;
        if (nz) { bad.push_back(t); continue; }                // X(U^dag P U) not in the axes' span: <P> = 0
        std::fill(kB.begin(), kB.end(), 0);
        for (int a = 0; a < A; ++a)
            if (getb(c, (size_t)a)) for (size_t w = 0; w < RW && w < H.axes[(size_t)a].k.size(); ++w) kB[w] ^= H.axes[(size_t)a].k[w];
        for (int i = 0; i < r; ++i) alpha[(size_t)i] = getb(kB, (size_t)i) ? 2 : 0;
        for (int a = 0; a < A; ++a) {
            if (!getb(H.axes[(size_t)a].img.w, (size_t)t)) continue;   // a not in A_P
            const int wa = ((n[(size_t)a] * (H.axes[(size_t)a].b ? -1 : 1)) % 4 + 4) % 4;
            if (!wa) continue;
            const std::vector<uint64_t>& ka = H.axes[(size_t)a].k;
            for (int i = 0; i < r; ++i) {
                if (!getb(ka, (size_t)i)) continue;
                alpha[(size_t)i] = (uint8_t)((alpha[(size_t)i] + wa) & 3);
                if (wa & 1) {
                    if (std::all_of(beta[(size_t)i].begin(), beta[(size_t)i].end(), [](uint64_t w) { return w == 0; }))
                        brows.push_back(i);                    // (re-listing a row that cancelled to 0 is harmless)
                    for (size_t w = 0; w < ka.size() && w < RW; ++w) beta[(size_t)i][w] ^= ka[w];
                }
            }
        }
        bool det = true;
        for (int i = 0; i < r; ++i) if (alpha[(size_t)i] & 3) det = false;
        for (int i : brows) {                                  // only the touched rows can be nonzero
            std::vector<uint64_t>& row = beta[(size_t)i];
            row[(size_t)i >> 6] &= ~(1ull << (i & 63));        // the diagonal is not a pair coefficient
            for (uint64_t w : row) if (w) det = false;
            std::fill(row.begin(), row.end(), 0);              // back to zero for the next column
        }
        brows.clear();
        if (!det) bad.push_back(t);
    }
    return bad;
}

}  // namespace qeccore::xdem::test_oracle
