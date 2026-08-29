#pragma once
// Ray-stabiliser bit machinery (shared by sampler.cpp and factored_stab.cpp).
//
// A CanonicalStabSum with chi<=2 has a well-defined RAY-STABILISER GROUP: the set of
// Hermitian Paulis that fix every branch ray |phi_i> up to a common sign. Membership of a
// Pauli in this group is exactly the "single-qubit Pauli is a stabiliser" test the A/B
// factorization needs (a qubit q is in the A-group iff X_q / Y_q / Z_q reduces to 0 against
// this basis). BitRref is the RREF over (x|z) bit vectors; ray_stabiliser_bits builds the
// basis for chi<=2 (empty for chi>2). Promoted verbatim from sampler.cpp so both call sites
// share one definition (sampler.cpp behavior must stay byte-identical).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

struct BitRref {                               // RREF basis over (x|z) bit vectors
    int W2 = 0;
    std::vector<std::vector<uint64_t>> rows;   // reduced basis (unique for the subspace)
    std::vector<int> pivots;                   // pivot bit index per row
    static int lowbit(const std::vector<uint64_t>& v) {
        for (size_t w = 0; w < v.size(); ++w)
            if (v[w]) return (int)(w * 64) + __builtin_ctzll(v[w]);
        return -1;
    }
    void reduce(std::vector<uint64_t>& v) const {
        for (size_t r = 0; r < rows.size(); ++r)
            if ((v[pivots[r] >> 6] >> (pivots[r] & 63)) & 1)
                for (int w = 0; w < W2; ++w) v[w] ^= rows[r][w];
    }
    void add(std::vector<uint64_t> v) {
        reduce(v);
        const int p = lowbit(v);
        if (p < 0) return;
        for (size_t r = 0; r < rows.size(); ++r)            // back-eliminate: keep RREF
            if ((rows[r][p >> 6] >> (p & 63)) & 1)
                for (int w = 0; w < W2; ++w) rows[r][w] ^= v[w];
        rows.push_back(std::move(v));
        pivots.push_back(p);
    }
};

inline std::vector<uint64_t> pauli_bits(const Pauli& P, int W) {
    std::vector<uint64_t> v((size_t)2 * W, 0);
    for (int w = 0; w < W; ++w) { v[w] = P.x[w]; v[W + w] = P.z[w]; }
    return v;
}

inline Pauli hermitian_from_bits(int n, const std::vector<uint64_t>& v, int W) {
    Pauli P(n);
    int ycount = 0;
    for (int w = 0; w < W; ++w) {
        P.x[w] = v[w];
        P.z[w] = v[W + w];
        ycount += __builtin_popcountll(v[w] & v[W + w]);
    }
    P.phase = ycount & 3;                       // i^{#Y}·X^x·Z^z is Hermitian
    return P;
}

// RREF basis of the post-state's ray-stabiliser bit group; empty basis when chi > 2.
inline BitRref ray_stabiliser_bits(const CanonicalStabSum& post) {
    const int n = post.n();
    const int W = (n + 63) / 64;
    BitRref basis;
    basis.W2 = 2 * W;
    const int chi = post.chi();
    if (chi > 2) return basis;
    if (chi == 1) {
        for (int a = 0; a < n; ++a) basis.add(pauli_bits(post.U.Zrow[a], W));
        return basis;
    }
    // chi == 2: branch translate D (bits only; phases are irrelevant to ray-fixing tests)
    Pauli D(n);
    const auto& s0 = post.branches[0].sigma;
    const auto& s1 = post.branches[1].sigma;
    for (size_t d = 0; d < post.free.size(); ++d) {
        const uint8_t b0 = d < s0.size() ? s0[d] : 0, b1 = d < s1.size() ? s1[d] : 0;
        if (b0 ^ b1) D = Pauli::multiply(D, post.U.Xrow[post.free[d]]);
    }
    int m1 = -1;                                // first generator anticommuting with D
    for (int a = 0; a < n; ++a) {
        if (Pauli::anticommute_bit(post.U.Zrow[a], D)) {
            if (m1 < 0) { m1 = a; continue; }
            basis.add(pauli_bits(Pauli::multiply(post.U.Zrow[a], post.U.Zrow[m1]), W));
        } else {
            basis.add(pauli_bits(post.U.Zrow[a], W));
        }
    }
    // swap part: one Born test per K-coset of D·S_bits (at most one coset contributes)
    auto swap_ok = [&](const Pauli& cand) {
        auto e = post.born_probabilities(cand);
        return std::abs(std::abs(e.first - e.second) - 1.0) < 1e-9;
    };
    Pauli c1 = hermitian_from_bits(n, pauli_bits(D, W), W);
    if (swap_ok(c1)) {
        basis.add(pauli_bits(c1, W));
    } else if (m1 >= 0) {
        Pauli c2 = hermitian_from_bits(
            n, pauli_bits(Pauli::multiply(D, post.U.Zrow[m1]), W), W);
        if (swap_ok(c2)) basis.add(pauli_bits(c2, W));
    }
    return basis;
}

}  // namespace qeccore
