#pragma once
// Residue-projection primitives for the magic-state reducer: GF(2) Row bitvectors + the
// rank-r Projector that maps original-coordinate parities into a compact residue space, the
// gadget multiset merge, and the dense Z8 / Moebius / Clifford-grade utilities. Moved verbatim
// out of the apps/ phase_poly_front engines into the src/ library so the reducer (FastTODD) and
// the --solve certifier can call them directly.
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "qeccore/compiler_builtins.hpp"   // __builtin_ctzll shim on MSVC (Projector::extend)

namespace qeccore {

using Row = std::vector<uint64_t>;                 // GF(2) row, 64 bits / word

int popcount_row(const Row& r);
bool row_zero(const Row& r);
Row row_xor(const Row& a, const Row& b);

// Projection onto the span of the odd parity rows (all-inline; used to map original-coordinate
// parities into a compact rank-r residue space).
struct Projector {
    int words = 1;
    std::vector<Row> basis;          // RREF rows in original coordinates
    std::vector<int> pivot;          // unique pivot bit per basis row
    int r() const { return (int)basis.size(); }
    bool extend(Row p) {             // grow basis; false if p already in span
        for (size_t i = 0; i < basis.size(); ++i)
            if (p[pivot[i] / 64] >> (pivot[i] % 64) & 1)
                for (int w = 0; w < words; ++w) p[w] ^= basis[i][w];
        if (row_zero(p)) return false;
        int pb = -1;
        for (int w = 0; w < words && pb < 0; ++w)
            if (p[w]) pb = 64 * w + __builtin_ctzll(p[w]);
        for (size_t i = 0; i < basis.size(); ++i)   // keep RREF: clear pb elsewhere
            if (basis[i][pb / 64] >> (pb % 64) & 1)
                for (int w = 0; w < words; ++w) basis[i][w] ^= p[w];
        basis.push_back(std::move(p));
        pivot.push_back(pb);
        return true;
    }
    uint32_t project(Row p) const {  // aborts if p outside span (would be a bug)
        uint32_t m = 0;
        for (size_t i = 0; i < basis.size(); ++i)
            if (p[pivot[i] / 64] >> (pivot[i] % 64) & 1) {
                for (int w = 0; w < words; ++w) p[w] ^= basis[i][w];
                m |= (uint32_t)1 << i;
            }
        if (!row_zero(p)) { std::fprintf(stderr, "todd: row outside projection span\n"); std::abort(); }
        return m;
    }
    Row reconstruct(uint32_t m) const {
        Row p(words, 0);
        for (size_t i = 0; i < basis.size(); ++i)
            if (m >> i & 1)
                for (int w = 0; w < words; ++w) p[w] ^= basis[i][w];
        return p;
    }
};

// In-place multilinear Z8 expansion by finite differences (Moebius transform).
void moebius(std::vector<uint8_t>& h, int r);

}  // namespace qeccore
