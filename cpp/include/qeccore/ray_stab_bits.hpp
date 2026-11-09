#pragma once
// Ray-stabiliser bit machinery (factored_stab.cpp's A/B factorization): BitRref is the RREF over
// (x|z) bit vectors; pauli_bits / hermitian_from_bits convert Paulis to and from them.
// (bloat-cut B1: ray_stabiliser_bits(CanonicalStabSum), whose only caller was the deleted affine
// factorize, is deleted.)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
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

}  // namespace qeccore
