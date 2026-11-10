#include "qeccore/stab_overlap.hpp"
#include <stdexcept>
#include <cstddef>

#include <vector>

#include "qeccore/gf2_gauss.hpp"   // shared gf2_solve + gauss_sum

namespace qeccore {

// ----------------------------------------------------------------------------
// prepare(θ): one bit-packed RREF of [R_θ | I_n] to extract G_θ (top k rows of
// the row-op matrix P) and H_θ (bottom n−k rows of P).  P R_θ = [I_k ; 0].
// ----------------------------------------------------------------------------

PreparedAffine PreparedAffine::prepare(const AffineState& theta) {
    PreparedAffine pa;
    const int n = theta.n_;
    const int k = theta.k_;
    pa.n_ = n;
    pa.kth_ = k;
    pa.m_ = n - k;
    pa.bth = theta.b;
    pa.Dth = theta.D;
    pa.Jth = theta.J;
    pa.omth = theta.omega;

    const int mw = (n + 63) / 64;
    pa.mw_ = mw;

    // Augmented matrix Aaug = [R_θ | I_n]: n rows, columns 0..k-1 = R_θ
    // (Aaug bit j = R_θ(i,j)), columns k..k+n-1 = I_n.  Words per row = W.
    const int ncols = k + n;
    const int W = (ncols + 63) / 64;
    std::vector<uint64_t> A((size_t)n * W, 0);
    auto Arow = [&](int i) -> uint64_t* { return A.data() + (size_t)i * W; };
    const int kw = (k + 63) / 64;                      // words holding the R_θ block
    for (int i = 0; i < n; ++i) {
        uint64_t* r = Arow(i);
        // R_θ block: columns [0,k) are exactly R_θ row i's words (trailing bits beyond
        // k are zero by the PackedMat invariant) — verbatim word copy, no per-bit scan.
        const uint64_t* rr = theta.R.row(i);
        for (int t = 0; t < kw && t < theta.R.words_; ++t) r[t] = rr[t];
        // I_n block: column k+i.
        int c = k + i;
        r[c >> 6] |= (1ULL << (c & 63));
    }

    // RREF over the first k columns (the R_θ block).  Pivot rows go to 0..k-1.
    // Because R_θ has full column rank k, every one of the k columns is a pivot.
    std::vector<int> pivot_row_of_col(k, -1);
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
        }
        const uint64_t* pr = Arow(prow);
        for (int i = 0; i < n; ++i) {
            if (i == prow) continue;
            uint64_t* ri = Arow(i);
            if (ri[cw] & cbit)
                for (int w = 0; w < W; ++w) ri[w] ^= pr[w];
        }
        pivot_row_of_col[col] = prow;
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

    // After RREF the first k rows are the pivot rows (P R_θ = I_k in the R block);
    // their I_n block (columns k..k+n-1) is G_θ.  Rows k..n-1 are H_θ.
    // Extract them, packed n-bit per row (mw words).
    pa.Gth.assign((size_t)k * mw, 0);
    pa.Hth.assign((size_t)pa.m_ * mw, 0);
    // The I-block (columns k..k+n-1) of a row is its words funnel-shifted down by k —
    // bits beyond column k+n-1 are zero in A, so no trailing mask is needed. Replaces
    // the per-bit O(n) column scan (O(n²) over all rows — the dominant prepare cost).
    auto copy_Iblock = [&](int srcrow, uint64_t* dst) {
        const uint64_t* r = Arow(srcrow);
        const int off = k >> 6, sh = k & 63;
        for (int w = 0; w < mw; ++w) {
            const uint64_t lo = r[off + w] >> sh;
            const uint64_t hi = (sh && off + w + 1 < W) ? (r[off + w + 1] << (64 - sh)) : 0ull;
            dst[w] = lo | hi;
        }
    };
    for (int i = 0; i < k; ++i) copy_Iblock(i, pa.Gth.data() + (size_t)i * mw);
    for (int i = 0; i < pa.m_; ++i) copy_Iblock(k + i, pa.Hth.data() + (size_t)i * mw);

    // c_θ = H_θ b_θ  (packed): bit i = parity of (Hrow_i AND b_θ).
    // Pack b_θ once.
    std::vector<uint64_t> bpk(mw, 0);
    for (int i = 0; i < n; ++i) if (theta.b[i]) bpk[i >> 6] |= (1ULL << (i & 63));
    pa.cth.assign(mw, 0);
    for (int i = 0; i < pa.m_; ++i) {
        const uint64_t* hr = pa.Hrow(i);
        int par = 0;
        for (int w = 0; w < mw; ++w) par ^= __builtin_popcountll(hr[w] & bpk[w]);
        if (par & 1) pa.cth[i >> 6] |= (1ULL << (i & 63));
    }
    return pa;
}

}  // namespace qeccore
