#include "qeccore/stab_overlap.hpp"
#include <stdexcept>
#include <cstddef>
#include <string>
#include <utility>

#include <vector>

namespace qeccore {

// ----------------------------------------------------------------------------
// prepare(θ): one bit-packed RREF over R_θ's k columns that tracks each row's row-op combination in
// SLOT space (k bits) instead of an n-wide identity block. Row i of the working matrix is
// [R-part (kw words) | slot part (kw words)] plus orig[i], the qubit it started as; its row-op
// combination is e_{orig[i]} ⊕ Σ_s (slot bit s)·e_{piv[s]}. When a row is brought up as column j's
// pivot, its own e_{orig} becomes slot j (no earlier slot names it) — so the combinations are those
// of the dense RREF of [R_θ | I_n] with the pivot coordinates renamed. Pivot selection (first row
// >= prow with the column bit), swaps and elimination are that RREF's, which reads only the R-part:
// same pivots, same row order, same G_θ/H_θ (cpp/tests/test_prepared_affine.cpp keeps the dense
// RREF as the oracle). Cost O(n·k + Σ eliminations·(kw+1)) words, memory n×2kw words.
// ----------------------------------------------------------------------------

PreparedAffine PreparedAffine::prepare(const AffineState& theta) {
    PreparedAffine pa;
    const int n = theta.n_;
    const int k = theta.k_;
    pa.n_ = n;
    pa.kth_ = k;
    pa.m_ = n - k;
    const int mw = (n + 63) / 64;
    const int kw = (k + 63) / 64;
    pa.mw_ = mw;
    pa.kw_ = kw;

    const int W = 2 * kw;
    std::vector<uint64_t> A((size_t)n * W, 0);
    auto Arow = [&](int i) -> uint64_t* { return A.data() + (size_t)i * W; };
    std::vector<int> orig(n);
    for (int i = 0; i < n; ++i) {
        // R-part: R_θ row i's words (trailing bits beyond k are zero by the PackedMat invariant).
        uint64_t* r = Arow(i);
        const uint64_t* rr = theta.R.row(i);
        for (int t = 0; t < kw && t < theta.R.words_; ++t) r[t] = rr[t];
        orig[i] = i;
    }

    // RREF over the k columns. Pivot rows go to 0..k-1. Because R_θ has full column rank k, every
    // one of the k columns is a pivot (enforced below).
    pa.piv.assign(k, -1);
    int prow = 0;
    for (int col = 0; col < k && prow < n; ++col) {
        const int cw = col >> 6;
        const uint64_t cbit = 1ULL << (col & 63);
        int sel = -1;
        for (int i = prow; i < n; ++i)
            if (Arow(i)[cw] & cbit) { sel = i; break; }
        if (sel < 0) continue;          // (cannot happen: full column rank)
        if (sel != prow) {
            uint64_t* a = Arow(prow);
            uint64_t* b = Arow(sel);
            for (int w = 0; w < W; ++w) std::swap(a[w], b[w]);
            std::swap(orig[prow], orig[sel]);
        }
        uint64_t* pr = Arow(prow);
        pr[kw + cw] |= cbit;            // the pivot's own e_{orig} is slot `col`
        pa.piv[col] = orig[prow];
        // The pivot row's R-part is zero below word cw (every earlier pivot column was eliminated
        // from it) and its slot part is zero above word cw (slots <= col): XOR only those words —
        // the skipped words are XORs with zero.
        for (int i = 0; i < n; ++i) {
            if (i == prow) continue;
            uint64_t* ri = Arow(i);
            if (ri[cw] & cbit) {
                for (int w = cw; w < kw; ++w) ri[w] ^= pr[w];
                for (int w = 0; w <= cw; ++w) ri[kw + w] ^= pr[kw + w];
            }
        }
        ++prow;
    }
    // ENFORCE the full-column-rank precondition instead of assuming it. Everything
    // downstream (G·R = I_k, im(R) = ker(H), and stabilizer_generators' prepared
    // phase pin, which takes the G-solve as THE unique solution) is silently wrong
    // on a rank-deficient R. Every production path feeds canonicalized states, so
    // this never fires today — but a future non-canonical construction path must
    // fail loudly here, not emit wrong stabilizer phases.
    if (prow != k)
        throw std::logic_error(
            "PreparedAffine::prepare: R is rank-deficient (rank " + std::to_string(prow) +
            " < k = " + std::to_string(k) + ") — state is not in canonical form");

    // After RREF the first k rows are the pivot rows (row j = column j's pivot): their slot parts
    // are G_θ. Rows k..n-1 are H_θ (qubit orig[k+i] plus their slot parts).
    pa.slot_of.assign(n, -1);
    for (int j = 0; j < k; ++j) pa.slot_of[pa.piv[j]] = j;
    pa.Gs.assign((size_t)k * kw, 0);
    pa.Hs.assign((size_t)pa.m_ * kw, 0);
    pa.hq.assign(pa.m_, 0);
    for (int j = 0; j < k; ++j)
        for (int w = 0; w < kw; ++w) pa.Gs[(size_t)j * kw + w] = Arow(j)[kw + w];
    for (int i = 0; i < pa.m_; ++i) {
        pa.hq[i] = orig[k + i];
        for (int w = 0; w < kw; ++w) pa.Hs[(size_t)i * kw + w] = Arow(k + i)[kw + w];
    }

    // c_θ = H_θ b_θ  (packed): bit i = b_θ[hq[i]] ⊕ parity(Hs row i AND b_θ in slot space).
    std::vector<uint64_t> bslot(kw, 0);
    for (int j = 0; j < k; ++j) if (theta.b[pa.piv[j]]) bslot[j >> 6] |= (1ULL << (j & 63));
    pa.cth.assign(mw, 0);
    for (int i = 0; i < pa.m_; ++i) {
        const uint64_t* hr = pa.Hrow(i);
        int par = theta.b[pa.hq[i]] ? 1 : 0;
        for (int w = 0; w < kw; ++w) par ^= __builtin_popcountll(hr[w] & bslot[w]);
        if (par & 1) pa.cth[i >> 6] |= (1ULL << (i & 63));
    }
    return pa;
}

}  // namespace qeccore
