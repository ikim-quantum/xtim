#pragma once
// Shared GF(2) Gaussian elimination for the exact-overlap inner products.
//
// AffineState::inner_product solves a GF(2) system A·x = rhs (A is m×N) to match
// two ket families, then reads off a particular solution + nullspace basis to
// build the Z4 quadratic form whose Gauss sum is the overlap. This routine
// lives here once.
//
// Performance: the dominant cost in the COMMON overlap case is this solve —
// two random stabilizer states are usually orthogonal, so the system is
// inconsistent and we early-exit (return false) right here. The hot loop is the
// row elimination `M[i][j] ^= M[prow][j]` over all columns j of all rows i. We
// bit-pack each augmented row [A_row | rhs_bit] into uint64 words so that row
// XOR is word-wide (`row_i[w] ^= row_prow[w]`) instead of per-byte scalar.
//
// Interface choice (documented):
//   INPUT  — byte matrix A (m×N) + byte rhs (length m). We pack once into
//            uint64 word-rows. Packing is O(m·N), dominated by the O(m·N^2)
//            elimination, so it does not defeat the win. (Packed-assembly from
//            the two reps was deemed not worth the entangling; the elimination
//            word-XOR is where the time goes.)
//   OUTPUT — `part` (length-N byte particular solution, free vars = 0) and
//            `ker` (nullspace basis: one length-N byte vector per free column).
//            Byte output is what the downstream form-extraction (Efn / ch_E)
//            consumes today; a later optimization can word-pack that consumer.
//
// Semantics are bit-identical to the old routine:
//   * pivot columns scanned ascending col = 0..N-1;
//   * particular solution sets free vars = 0, pivot vars = rhs of their row;
//   * one nullspace basis vector per free column fc, with g[fc]=1 and
//     g[pivot_col(r)] ^= M[r][fc] for each pivot row r;
//   * inconsistency = any row that is all-zero in cols 0..N-1 but 1 in col N.

#include <cstdint>
#include <cstddef>
#include <vector>

#include "qeccore/exact_phase.hpp"

namespace qeccore {

// ---------------------------------------------------------------------------
// Allocation-free, fully bit-packed variant of gf2_solve used by the amortized
// PreparedAffine::overlap hot path. Caller supplies:
//   * Apk   — the system matrix A, packed by ROW (m rows × kw words, kw =
//             ceil(N/64)); A(i,j) lives in word Apk[i*kw + (j>>6)] bit (j&63).
//   * rhspk — rhs packed as one bit per row: rhspk[i>>6] bit (i&63).
//   * Mwork, pivot_col, col_pivot — reusable scratch buffers (resize-on-grow;
//     never shrunk). Mwork holds the W-word augmented working matrix.
//   * part_pk — packed particular solution (kw words).
//   * ker_pk  — packed nullspace basis, ker_rows rows × kw words (resize-on-grow,
//               only ker_rows are meaningful on return).
// Returns true + fills part_pk/ker_pk/ker_rows; false on inconsistency.
//
// The ALGORITHM and RESULT are bit-identical to the byte gf2_solve below: same
// ascending pivot-column scan, same particular solution (free vars 0, pivot var
// = its row's rhs bit), same one-basis-vector-per-free-column nullspace, same
// inconsistency test. Only the I/O representation differs (packed in, packed out)
// and the working buffers are caller-owned (no per-call heap allocation).
inline bool gf2_solve_packed(const std::vector<uint64_t>& Apk, int m, int N,
                             const std::vector<uint64_t>& rhspk,
                             std::vector<uint64_t>& Mwork,
                             std::vector<int>& pivot_col,
                             std::vector<int>& col_pivot,
                             std::vector<uint64_t>& part_pk,
                             std::vector<uint64_t>& ker_pk,
                             int& ker_rows) {
    const int kw = (N + 63) >> 6;          // words per A-row (input packing)
    const int Wcols = N + 1;
    const int W = (Wcols + 63) >> 6;       // words per augmented working row
    if ((int)Mwork.size() < m * W) Mwork.assign((size_t)m * W, 0ull);
    auto row = [&](int i) -> uint64_t* { return Mwork.data() + (size_t)i * W; };

    // Build the augmented matrix [A | rhs] into the reusable working buffer.
    // Copy A's kw words verbatim (cols 0..N-1), clear any padding words up to W,
    // then set the rhs bit at column N.
    for (int i = 0; i < m; ++i) {
        uint64_t* r = row(i);
        const uint64_t* a = Apk.data() + (size_t)i * kw;
        for (int w = 0; w < kw; ++w) r[w] = a[w];
        for (int w = kw; w < W; ++w) r[w] = 0ull;
        // If N is a multiple of 64, the rhs bit starts a fresh word (already
        // cleared above); otherwise it shares the last A-word, which may have the
        // rhs bit position as a high bit — A only fills bits 0..N-1 so it's clear.
        if ((rhspk[(size_t)i >> 6] >> (i & 63)) & 1ull)
            r[N >> 6] |= (1ull << (N & 63));
    }

    if ((int)pivot_col.size() < m) pivot_col.assign(m, -1);
    else for (int i = 0; i < m; ++i) pivot_col[i] = -1;
    if ((int)col_pivot.size() < N) col_pivot.assign(N, -1);
    else for (int i = 0; i < N; ++i) col_pivot[i] = -1;

    int prow = 0;
    for (int col = 0; col < N && prow < m; ++col) {
        const int cw = col >> 6;
        const uint64_t cbit = 1ull << (col & 63);
        int sel = -1;
        for (int i = prow; i < m; ++i)
            if (row(i)[cw] & cbit) { sel = i; break; }
        if (sel < 0) continue;
        if (sel != prow) {
            uint64_t* a = row(prow);
            uint64_t* b = row(sel);
            for (int w = 0; w < W; ++w) std::swap(a[w], b[w]);
        }
        const uint64_t* pr = row(prow);
        for (int i = 0; i < m; ++i) {
            if (i == prow) continue;
            uint64_t* ri = row(i);
            if (ri[cw] & cbit)
                for (int w = 0; w < W; ++w) ri[w] ^= pr[w];
        }
        pivot_col[prow] = col;
        col_pivot[col] = prow;
        ++prow;
    }

    const int Nw = N >> 6;
    const uint64_t Nbit = 1ull << (N & 63);
    for (int i = 0; i < m; ++i) {
        const uint64_t* r = row(i);
        bool allzero = true;
        for (int w = 0; w < W; ++w) {
            uint64_t word = r[w];
            if (w == Nw) word &= ~Nbit;
            if (word) { allzero = false; break; }
        }
        if (allzero && (r[Nw] & Nbit)) return false;
    }

    // Particular solution (packed): free vars = 0, pivot var = its row's rhs bit.
    if ((int)part_pk.size() < kw) part_pk.assign(kw, 0ull);
    for (int w = 0; w < kw; ++w) part_pk[w] = 0ull;
    for (int r = 0; r < prow; ++r) {
        int c = pivot_col[r];
        if (c >= 0 && (row(r)[Nw] & Nbit)) part_pk[c >> 6] |= (1ull << (c & 63));
    }

    // Nullspace basis (packed): one vector per free column, in ascending fc order
    // (same order as the byte routine), packed into kw words per row.
    ker_rows = 0;
    for (int fc = 0; fc < N; ++fc) {
        if (col_pivot[fc] != -1) continue;
        const int fcw = fc >> 6;
        const uint64_t fcbit = 1ull << (fc & 63);
        if ((int)ker_pk.size() < (ker_rows + 1) * kw)
            ker_pk.resize((size_t)(ker_rows + 1) * kw, 0ull);
        uint64_t* g = ker_pk.data() + (size_t)ker_rows * kw;
        for (int w = 0; w < kw; ++w) g[w] = 0ull;
        g[fcw] |= fcbit;
        for (int r = 0; r < prow; ++r) {
            int c = pivot_col[r];
            if (c >= 0 && (row(r)[fcw] & fcbit)) g[c >> 6] ^= (1ull << (c & 63));
        }
        ++ker_rows;
    }
    return true;
}

// Single-word Gaussian solve for N <= 63 (formerly file-local in stab_affine.cpp; shared so
// PreparedAffine::overlap's small-k fast path can use it too): each augmented row [A | rhs] of the
// m×N GF(2) system is one uint64 (bit j = column j, bit N = rhs). Identical algorithm and output
// to gf2_solve / gf2_solve_packed — ascending pivot-column scan, first-hit pivot row brought up by
// swap, full elimination of every other row, particular solution = rhs bits on pivot columns (free
// vars 0), one nullspace vector per free column in ascending order — only the row layout differs,
// and the per-row conditional XOR is branchless (XOR with an all-zero mask when the column bit is
// clear, a no-op). Mw is consumed in place; part and the ker rows are single packed words.
// Differentially verified vs the byte gf2_solve over ~840k live calls (IP_W1_XCHECK).
inline bool gf2_solve_w1(uint64_t* Mw, int m, int N,
                         uint64_t& part, uint64_t* ker, int& ker_rows) {
    int pivot_col[64];
    int col_pivot[64];
    for (int c = 0; c < N; ++c) col_pivot[c] = -1;
    int prow = 0;
    for (int col = 0; col < N && prow < m; ++col) {
        const uint64_t cbit = 1ull << col;
        int sel = -1;
        for (int i = prow; i < m; ++i)
            if (Mw[i] & cbit) { sel = i; break; }
        if (sel < 0) continue;
        const uint64_t pr = Mw[sel];
        Mw[sel] = Mw[prow];
        Mw[prow] = pr;
        for (int i = 0; i < prow; ++i) Mw[i] ^= pr & (0ull - ((Mw[i] >> col) & 1ull));
        for (int i = prow + 1; i < m; ++i) Mw[i] ^= pr & (0ull - ((Mw[i] >> col) & 1ull));
        pivot_col[prow] = col;
        col_pivot[col] = prow;
        ++prow;
    }
    const uint64_t rhsbit = 1ull << N;
    for (int i = 0; i < m; ++i)
        if (!(Mw[i] & (rhsbit - 1)) && (Mw[i] & rhsbit)) return false;
    part = 0;
    for (int rr = 0; rr < prow; ++rr)
        if (Mw[rr] & rhsbit) part |= 1ull << pivot_col[rr];
    ker_rows = 0;
    for (int fc = 0; fc < N; ++fc) {
        if (col_pivot[fc] != -1) continue;
        const uint64_t fcbit = 1ull << fc;
        uint64_t g = fcbit;
        for (int rr = 0; rr < prow; ++rr)
            if (Mw[rr] & fcbit) g ^= 1ull << pivot_col[rr];
        ker[ker_rows++] = g;
    }
    return true;
}

// LSB-first 64x64 in-place bit transpose (formerly file-local in canonical_stab_sum.cpp;
// Hacker's-Delight shape with the sub-block roles flipped for bit-0-first columns):
// A[i] bit j -> A[j] bit i. Verified against the per-bit definition.
inline void transpose64(uint64_t A[64]) {
    uint64_t m = 0x00000000FFFFFFFFULL;
    for (int j = 32; j != 0; j >>= 1, m ^= m << j) {
        for (int k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            const uint64_t t = ((A[k] >> j) ^ A[k | j]) & m;
            A[k] ^= t << j;
            A[k | j] ^= t;
        }
    }
}

// Blockwise transpose of an R x C bit matrix (rows of ceil(C/64) words, LSB-first) into a C x R
// matrix (rows of ceil(R/64) words), zero-padded edge blocks. Verified over 2000 random sizes up
// to 140 (multi-word blocks) against the per-bit definition.
inline void transpose_bits(const uint64_t* M, int R, int C, uint64_t* MT) {
    const int Wc = (C + 63) >> 6, Wr = (R + 63) >> 6;
    for (size_t i = 0; i < (size_t)C * Wr; ++i) MT[i] = 0;
    uint64_t blk[64];
    for (int bi = 0; bi < Wr; ++bi) {
        for (int bj = 0; bj < Wc; ++bj) {
            for (int k = 0; k < 64; ++k) {
                const int r = (bi << 6) + k;
                blk[k] = (r < R) ? M[(size_t)r * Wc + bj] : 0ull;
            }
            transpose64(blk);
            for (int k = 0; k < 64; ++k) {
                const int c = (bj << 6) + k;
                if (c < C) MT[(size_t)c * Wr + bi] = blk[k];
            }
        }
    }
}

// Solve A·x = rhs over GF(2). On success returns true and fills `part` (one
// particular solution, length N) and `ker` (nullspace basis, each length N).
// On inconsistency returns false (leaving part/ker unspecified).
inline bool gf2_solve(const std::vector<std::vector<uint8_t>>& A,  // m rows × N cols
                      const std::vector<uint8_t>& rhs,             // length m
                      int N,
                      std::vector<uint8_t>& part,
                      std::vector<std::vector<uint8_t>>& ker) {
    const int m = (int)A.size();
    // Augmented matrix has N+1 columns (cols 0..N-1 = A, col N = rhs), packed
    // into W = ceil((N+1)/64) uint64 words per row. Bit c lives in
    // word (c>>6), bit (c&63).
    const int Wcols = N + 1;
    const int W = (Wcols + 63) >> 6;
    std::vector<uint64_t> M((size_t)m * W, 0ull);
    auto row = [&](int i) -> uint64_t* { return M.data() + (size_t)i * W; };

    // Pack [A | rhs] once.
    for (int i = 0; i < m; ++i) {
        uint64_t* r = row(i);
        const uint8_t* a = A[i].data();
        for (int j = 0; j < N; ++j)
            if (a[j] & 1) r[j >> 6] |= (1ull << (j & 63));
        if (rhs[i] & 1) r[N >> 6] |= (1ull << (N & 63));
    }

    std::vector<int> pivot_col(m, -1);     // pivot column for each used row
    std::vector<int> col_pivot_row(N, -1); // pivot row for each column (-1 = free)
    int prow = 0;
    for (int col = 0; col < N && prow < m; ++col) {
        const int cw = col >> 6;
        const uint64_t cbit = 1ull << (col & 63);
        // Find a row at >= prow with a 1 in this column.
        int sel = -1;
        for (int i = prow; i < m; ++i)
            if (row(i)[cw] & cbit) { sel = i; break; }
        if (sel < 0) continue;
        // Bring pivot to row `prow` (swap whole packed rows, word by word).
        if (sel != prow) {
            uint64_t* a = row(prow);
            uint64_t* b = row(sel);
            for (int w = 0; w < W; ++w) std::swap(a[w], b[w]);
        }
        // Eliminate this column from every other row that has a 1 in it.
        // Word-wide XOR of the whole augmented row (full reduction, matching the
        // old `for j in [col,N]` — XORing the already-zero low words is a no-op,
        // so doing all W words is equivalent and lets us skip per-bit work).
        const uint64_t* pr = row(prow);
        for (int i = 0; i < m; ++i) {
            if (i == prow) continue;
            uint64_t* ri = row(i);
            if (ri[cw] & cbit)
                for (int w = 0; w < W; ++w) ri[w] ^= pr[w];
        }
        pivot_col[prow] = col;
        col_pivot_row[col] = prow;
        ++prow;
    }

    // Inconsistency: a row that is all-zero in cols 0..N-1 but 1 in col N.
    // The augmented matrix only occupies columns 0..N (= N+1 bits), so the only
    // bit to mask out when checking "all-zero in the A part" is the rhs bit at
    // column N; no bits beyond it exist.
    const int Nw = N >> 6;
    const uint64_t Nbit = 1ull << (N & 63);
    for (int i = 0; i < m; ++i) {
        const uint64_t* r = row(i);
        bool allzero = true;
        for (int w = 0; w < W; ++w) {
            uint64_t word = r[w];
            if (w == Nw) word &= ~Nbit;     // ignore the rhs bit for the A-part test
            if (word) { allzero = false; break; }
        }
        if (allzero && (r[Nw] & Nbit)) return false;
    }

    // Particular solution: free vars = 0, pivot vars = rhs bit of their row.
    part.assign(N, 0);
    for (int r = 0; r < prow; ++r) {
        int c = pivot_col[r];
        if (c >= 0 && (row(r)[Nw] & Nbit)) part[c] = 1;
    }

    // Nullspace basis: one vector per free column.
    ker.clear();
    for (int fc = 0; fc < N; ++fc) {
        if (col_pivot_row[fc] != -1) continue;   // not a free column
        const int fcw = fc >> 6;
        const uint64_t fcbit = 1ull << (fc & 63);
        std::vector<uint8_t> g(N, 0);
        g[fc] = 1;
        for (int r = 0; r < prow; ++r) {
            int c = pivot_col[r];
            if (c >= 0 && (row(r)[fcw] & fcbit)) g[c] ^= 1;
        }
        ker.push_back(std::move(g));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Z4 Gauss sum  G = Σ_{z∈𝔽₂^r} i^{E(z)} for a ℤ4 quadratic form E given by a
// constant c0, a linear part L (length r, mod 4), and quadratic bits K (r×r
// symmetric, zero diagonal). Returns the exact value as an ExactPhase.
//
// Used by AffineState::inner_product (companion to gf2_solve above); it lives
// here once.
//
// Algorithm (recursive variable elimination). Each step picks the first active
// var v, finds its coupled active neighbours (K[v][j]=1), and folds the i^{E}
// sum over z_v:
//   * L[v] odd  → 1 + i^{Lv+2t} = √2·ζ8^{e0}; the linear term toggles a phase on
//                 each coupled var (L[j]+=dL) and the pair-product term flips
//                 K[a][b] for every pair {a,b} of coupled vars; v is removed.
//   * L[v] even, no coupling → factor 2 (Lv=0) or 0 (Lv=2); v removed.
//   * L[v] even, coupled     → factor 2, a constraint z_pp = constraintBit ⊕
//                 (⊕ other coupled), eliminating BOTH v and the pivot pp; the
//                 reduced ℤ4 form on the surviving vars is rebuilt by evaluating
//                 E on the basis vectors (curE/buildFull) and re-reading c/L/K.
//
// Representation: K is a byte matrix (std::vector<std::vector<uint8_t>>, the
// form emitted by the inner_product form-extraction). An earlier experiment
// word-packed K internally, but profiling showed no speedup and a slight
// regression at the small r typical of overlaps (the recursion is too small a
// fraction of the work after the other opts, so the pack overhead dominates),
// so the byte representation is kept. L is taken by value (mutated internally);
// K likewise by value (the working form is mutated in place).
inline ExactPhase gauss_sum(int r, int c0, std::vector<int> L,
                            std::vector<std::vector<uint8_t>> K) {
    ExactPhase acc = ExactPhase::zeta8(2 * c0);   // i^{c0}
    std::vector<char> active(r, 1);
    int remaining = r;
    while (remaining > 0) {
        int v = -1;
        for (int i = 0; i < r; ++i) if (active[i]) { v = i; break; }
        std::vector<int> coupled;
        for (int j = 0; j < r; ++j) if (j != v && active[j] && K[v][j]) coupled.push_back(j);
        int Lv = ((L[v] % 4) + 4) % 4;
        if (Lv & 1) {
            int e0 = 2 - (Lv & 3);
            int e1 = 2 - ((Lv + 2) & 3);
            acc = acc.mul({false, +1, ((e0 % 8) + 8) % 8});  // √2·ζ8^{e0}
            int de = e1 - e0;
            int dL = (((de / 2) % 4) + 4) % 4;
            for (int j : coupled) L[j] = (((L[j] + dL) % 4) + 4) % 4;
            for (size_t ai = 0; ai < coupled.size(); ++ai)
                for (size_t bi = ai + 1; bi < coupled.size(); ++bi) {
                    int a = coupled[ai], bc = coupled[bi];
                    K[a][bc] ^= 1; K[bc][a] ^= 1;
                }
            active[v] = 0; --remaining;
        } else {
            if (coupled.empty()) {
                if (Lv == 0) acc = acc.mul({false, +2, 0});
                else return ExactPhase::zero();
                active[v] = 0; --remaining;
            } else {
                int constraintBit = (Lv == 2) ? 1 : 0;
                acc = acc.mul({false, +2, 0});
                int pp = coupled[0];
                std::vector<int> rest(coupled.begin() + 1, coupled.end());
                std::vector<int> rem;
                for (int i = 0; i < r; ++i)
                    if (active[i] && i != v && i != pp) rem.push_back(i);
                int mm = (int)rem.size();
                auto curE = [&](const std::vector<uint8_t>& full) -> int {
                    int q = 0;
                    for (int i = 0; i < r; ++i) if (active[i] && full[i]) q += L[i];
                    for (int i = 0; i < r; ++i)
                        for (int j = i + 1; j < r; ++j)
                            if (active[i] && active[j] && full[i] && full[j] && K[i][j])
                                q += 2;
                    return ((q % 4) + 4) % 4;
                };
                auto buildFull = [&](const std::vector<uint8_t>& zr) {
                    std::vector<uint8_t> full(r, 0);
                    for (int idx = 0; idx < mm; ++idx) full[rem[idx]] = zr[idx];
                    full[v] = 0;
                    uint8_t par = (uint8_t)constraintBit;
                    for (int sx : rest) par ^= full[sx];
                    full[pp] = par;
                    return full;
                };
                int cst = curE(buildFull(std::vector<uint8_t>(mm, 0)));
                std::vector<int> Ln(mm, 0);
                for (int t = 0; t < mm; ++t) {
                    std::vector<uint8_t> zr(mm, 0); zr[t] = 1;
                    Ln[t] = ((curE(buildFull(zr)) - cst) % 4 + 4) % 4;
                }
                std::vector<std::vector<uint8_t>> Kn(mm, std::vector<uint8_t>(mm, 0));
                for (int i = 0; i < mm; ++i)
                    for (int j = i + 1; j < mm; ++j) {
                        std::vector<uint8_t> zr(mm, 0); zr[i] = 1; zr[j] = 1;
                        int val = ((curE(buildFull(zr)) - cst - Ln[i] - Ln[j]) % 4 + 4) % 4;
                        Kn[i][j] = Kn[j][i] = (uint8_t)((val / 2) & 1);
                    }
                acc = acc.mul(ExactPhase::zeta8(2 * cst));
                for (int idx = 0; idx < mm; ++idx) L[rem[idx]] = Ln[idx];
                for (int i = 0; i < mm; ++i)
                    for (int j = 0; j < mm; ++j)
                        K[rem[i]][rem[j]] = Kn[i][j];
                active[v] = 0; active[pp] = 0; remaining -= 2;
            }
        }
    }
    return acc;
}

}  // namespace qeccore
