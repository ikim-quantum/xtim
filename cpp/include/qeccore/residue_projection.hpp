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
    // r-bit residue coordinates as a word vector (any r; bit i = basis row i).  The r <= 32
    // `project` below is its one-word special case.  Aborts if p is outside the span (a bug).
    Row project_wide(Row p) const {
        Row m((basis.size() + 63) / 64 + (basis.empty() ? 1 : 0), 0);
        for (size_t i = 0; i < basis.size(); ++i)
            if (p[pivot[i] / 64] >> (pivot[i] % 64) & 1) {
                for (int w = 0; w < words; ++w) p[w] ^= basis[i][w];
                m[i >> 6] |= 1ull << (i & 63);
            }
        if (!row_zero(p)) { std::fprintf(stderr, "todd: row outside projection span\n"); std::abort(); }
        return m;
    }
    // Original-coordinate row of the residue combination listed by its set bit indices.
    Row reconstruct_bits(const std::vector<int>& on) const {
        Row p(words, 0);
        for (int i : on)
            for (int w = 0; w < words; ++w) p[w] ^= basis[(size_t)i][w];
        return p;
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

// ── Even (Clifford) residue of a T-count reduction ───────────────────────────────────────────
// Over the rank-r residue, the true magic phase is f(x) = Σ_in c_in·p_{m_in}(x) (mod 8) and the
// reduced one is f'(x) = Σ_{m∈red} p_m(x), with p_m(x) = popcount(m & x) & 1.  The remainder
// g = f − f' must be EVEN-GRADED (deg-1 coeffs even, deg-2 ≡ 0 mod 4, nothing at deg >= 3).
// A column is given by its residue support (sorted set-bit indices) and its ζ8 coefficient c (mod 8).
// Result: the nonzero monomials of g, in ASCENDING numeric order of their residue mask (the order a
// dense 2^r Möbius table is scanned in), or ok = false + the first offending monomial.
struct ResidueColumn { std::vector<int> on; int c8; };
struct EvenMonomial { int hi, lo; int a; };          // {hi} (lo = -1) or {lo < hi}; coeff a mod 8
struct EvenResidue {
    bool ok = true;
    std::vector<EvenMonomial> mono;                  // ascending residue-mask order
    int bad_deg = 0, bad_coeff = 0;                  // first violation when !ok
    std::vector<int> bad_on;
};
// Production path: symbolic, polynomial (O(Σ w³) for column weights w).  Uses the exact mod-8
// identity p_m(x) = Σ_{∅≠S⊆m} (−2)^{|S|−1} Π_{i∈S} x_i  (|S| >= 4 terms vanish mod 8).
EvenResidue even_residue_symbolic(int r, const std::vector<ResidueColumn>& f_cols,
                                  const std::vector<ResidueColumn>& fp_cols);
// ORACLE (tests only, r <= ~24): dense 2^r tables + Möbius transform — the pre-3.1.8 production code.
EvenResidue even_residue_dense(int r, const std::vector<ResidueColumn>& f_cols,
                               const std::vector<ResidueColumn>& fp_cols);

}  // namespace qeccore
