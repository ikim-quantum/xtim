#include "qeccore/stab_overlap.hpp"
#include <stdexcept>
#include <cstddef>

#include <vector>

#include "qeccore/gf2_gauss.hpp"   // shared gf2_solve + gauss_sum

namespace qeccore {

namespace {
using qeccore::gf2_solve;
using qeccore::gf2_solve_packed;
using qeccore::gauss_sum;

// Q(x) mod 4 from a bit-packed x (kw words) against (D, J).  Identical formula
// to AffineState::inner_product's eval_Q_packed (verified there vs the byte
// eval_Q): linear Σ_i D_i x_i over set bits, quadratic J symmetric zero-diagonal
// so Σ_{i:x_i} popcount(Jrow_i & x) double-counts each set pair → halve, ×2.
inline int eval_Q_packed(int kw, const std::vector<int>& D,
                         const SymPackedMat& J,
                         const uint64_t* x) {
    long q = 0;
    long pairs = 0;
    for (int w = 0; w < kw; ++w) {
        uint64_t bits = x[w];
        while (bits) {
            int b = __builtin_ctzll(bits);
            bits &= bits - 1;
            int i = (w << 6) + b;
            q += D[i];
            const uint64_t* jr = J.row(i);
            for (int ww = 0; ww < kw; ++ww)
                pairs += __builtin_popcountll(jr[ww] & x[ww]);
        }
    }
    return (int)(((q + (pairs / 2) * 2) % 4 + 4) % 4);
}

}  // namespace

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

    // Column-major copy H_θ^T (Opt A): row c (coordinate) holds the m_-bit set of
    // check rows i with H_θ(i,c)=1.  Built once; θ fixed.  Blockwise bit transpose
    // (shared transpose_bits) — identical layout to the old per-set-bit scatter.
    pa.mwc_ = (pa.m_ + 63) / 64;
    pa.HthT.assign((size_t)n * pa.mwc_, 0);
    transpose_bits(pa.Hth.data(), pa.m_, n, pa.HthT.data());

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

// ----------------------------------------------------------------------------
// overlap(ψ): solve only ψ's side, then build E(z) and Gauss-sum.
// ----------------------------------------------------------------------------
ExactPhase PreparedAffine::overlap(const AffineState& psi) const {
    const int n = n_;
    const int kpsi = psi.k_;
    const int m = m_;
    const int mw = mw_;
    const int kth_words = (kth_ + 63) / 64;
    const int kpsi_words = (kpsi + 63) / 64;   // words to hold a kpsi-bit w-vector

    // ---- Reusable per-evaluator scratch (grow-then-reuse, never shrink) -------
    // After the first overlap() these resizes are all no-ops, so the hot path is
    // allocation-free.  (Single-threaded-per-evaluator: ws_ is mutated here.)
    Workspace& ws = ws_;
    auto grow = [](std::vector<uint64_t>& v, size_t need) {
        if (v.size() < need) v.resize(need);
    };
    const int Mkw = kpsi_words > 0 ? kpsi_words : 1;   // words per packed M row
    const int mwc = mwc_;                              // words per m-bit vector
    grow(ws.bpsi, mw);
    grow(ws.bth_pk, mw);
    grow(ws.Mpk, (size_t)m * Mkw);
    grow(ws.rhs, (size_t)((m + 63) / 64));
    grow(ws.Hbpsi, mwc);

    uint64_t* bpsi = ws.bpsi.data();
    uint64_t* bth_pk = ws.bth_pk.data();

    for (int t = 0; t < mw; ++t) bpsi[t] = 0ull;
    for (int i = 0; i < n; ++i) if (psi.b[i]) bpsi[i >> 6] |= (1ULL << (i & 63));

    // ---- Opt A: assemble M = H_θ R_ψ by OUTER-PRODUCT over H_θ^T, no R_ψ ------
    // transpose.  M is row-major (m rows × kpsi bits = Mpk).  For each coordinate c
    // (0..n-1): hcol = H_θ^T row c (m bits = check rows with a 1 at c), rrow = R_ψ
    // row c (kpsi bits, native row-major).  For each set bit i of hcol, M_row_i ^=
    // rrow.  rhs uses H_θ b_ψ accumulated the same way: Hb ^= H_θ^T row c for each
    // set coordinate c of b_ψ; then rhs(i) = c_θ(i) ⊕ Hb(i).  Bit-identical to the
    // popcount-parity form (XOR of selected columns == parity of AND, per bit).
    if (kpsi <= 63) {
        // ---- FUSED single-word build + solve (k_ψ <= 63, the common case) -----
        // Augmented row i = (⊕_{c: H_θ(i,c)=1} R_ψ row c) | rhs_i << k_ψ, with
        // rhs_i = parity(Hrow_i & b_ψ) ⊕ c_θ(i): a row-major GATHER over H rows
        // keeping the k_ψ-bit row accumulator in a register, replacing the Mpk
        // scatter + the Hb pass + gf2_solve_packed's internal Mwork copy. Same
        // (i,c) pair set and XOR is commutative ⇒ identical M rows; bit i of
        // H_θ b_ψ == parity(Hrow_i & b_ψ) ⇒ identical rhs. Solved by the shared
        // gf2_solve_w1 (algorithm and output bit-identical to gf2_solve_packed)
        // into the same ws.part / ws.ker layout (Mkw = 1), so everything
        // downstream is unchanged. The section profile put the scatter + packed
        // solve at ~6.1k of the 6.2k-cycle call on the chi=16 n=64 workload.
        grow(ws.gf_M, (size_t)m);
        uint64_t* Mw = ws.gf_M.data();
        for (int i = 0; i < m; ++i) {
            const uint64_t* hr = Hrow(i);
            uint64_t acc = 0;
            int par = 0;
            for (int t = 0; t < mw; ++t) {
                uint64_t w = hr[t];
                par ^= __builtin_popcountll(w & bpsi[t]);
                if (kpsi > 0) {
                    const int base = t << 6;
                    while (w) { acc ^= psi.R.row(base + __builtin_ctzll(w))[0]; w &= w - 1; }
                }
            }
            par = (par & 1) ^ (int)((cth[i >> 6] >> (i & 63)) & 1ull);
            Mw[i] = acc | ((uint64_t)par << kpsi);
        }
        grow(ws.part, 1);
        grow(ws.ker, (size_t)(kpsi > 0 ? kpsi : 1));
        if (!gf2_solve_w1(Mw, m, kpsi, ws.part[0], ws.ker.data(), ws.ker_rows))
            return ExactPhase::zero();       // orthogonal supports (prefilter)
    } else {
    uint64_t* Mpk = ws.Mpk.data();
    uint64_t* rhs = ws.rhs.data();
    uint64_t* Hb = ws.Hbpsi.data();
    for (int i = 0; i < m; ++i) {
        uint64_t* mrow = Mpk + (size_t)i * Mkw;
        for (int w = 0; w < Mkw; ++w) mrow[w] = 0ull;
    }
    for (int w = 0; w < mwc; ++w) Hb[w] = 0ull;
    const int rww = psi.R.words_;                     // words per R_ψ row (≥ Mkw)
    // M += outer(hcol, rrow): for each coordinate c, for each set check-row i of
    // H_θ^T row c, M_row_i ^= R_ψ row c (a sparse GF(2) row-SCATTER).
    //
    // NOTE on SIMD: a masked-lane AVX-512 form (8 check-rows/__m512i, mask_xor) was
    // evaluated and DROPPED.  It does O(n·(m/8)) lane-XORs REGARDLESS of sparsity,
    // but H_θ (an RREF of a magic state's R block) measures ~0.6–1% DENSE, so the
    // set-bit scalar scatter touches only ~m·n/100 words.  Benchmarked: the
    // masked-lane version REGRESSED (~4.7µs→~6.6µs at n=256,k_ψ=2).  Scalar kept.
    for (int c = 0; c < n; ++c) {
        const uint64_t* hcol = HTrow(c);              // m bits (mwc words)
        if (kpsi > 0) {
            const uint64_t* rr = psi.R.row(c);        // kpsi bits (native row-major)
            for (int wc = 0; wc < mwc; ++wc) {
                uint64_t hb = hcol[wc];
                if (!hb) continue;
                int base = wc << 6;
                while (hb) {
                    int i = base + __builtin_ctzll(hb);
                    hb &= hb - 1;
                    uint64_t* mrow = Mpk + (size_t)i * Mkw;
                    int t = 0;
                    for (; t < Mkw && t < rww; ++t) mrow[t] ^= rr[t];
                }
            }
        }
        // Hb += hcol if b_ψ has a 1 at coordinate c.
        if ((bpsi[c >> 6] >> (c & 63)) & 1ull)
            for (int wc = 0; wc < mwc; ++wc) Hb[wc] ^= hcol[wc];
    }
    // rhs(i) = c_θ(i) ⊕ (H_θ b_ψ)(i), one bit per row.
    for (int w = 0; w < (m + 63) / 64; ++w)
        rhs[w] = (w < mwc ? Hb[w] : 0ull) ^ cth[w];

    // ---- Allocation-free GF(2) solve (caller-owned scratch, packed I/O) -------
    if (!gf2_solve_packed(ws.Mpk, m, kpsi, ws.rhs, ws.gf_M, ws.gf_pivot_col,
                          ws.gf_col_pivot, ws.part, ws.ker, ws.ker_rows))
        return ExactPhase::zero();           // orthogonal supports (prefilter)
    }

    const int r = ws.ker_rows;
    const uint64_t* w0 = ws.part.data();                       // kpsi_words words
    auto kerv = [&](int l) -> const uint64_t* {                // kpsi_words words
        return ws.ker.data() + (size_t)l * Mkw;
    };

    for (int t = 0; t < mw; ++t) bth_pk[t] = 0ull;
    for (int i = 0; i < n; ++i) if (bth[i]) bth_pk[i >> 6] |= (1ULL << (i & 63));

    // ---- Pack ψ's R into COLUMN-major (only on the nonzero path) -------------
    // Needed by xfromw (x = b_ψ ⊕ R_ψ w).  Deferred here so the common DISJOINT
    // (orthogonal) case never pays for the column transpose: it returned above.
    grow(ws.Rcol, (size_t)kpsi * mw);
    uint64_t* Rcol = ws.Rcol.data();   // column j lives at Rcol + j*mw (mw words)
    auto rcol = [&](int j) -> uint64_t* { return Rcol + (size_t)j * mw; };
    for (int j = 0; j < kpsi; ++j)
        for (int t = 0; t < mw; ++t) rcol(j)[t] = 0ull;
    for (int i = 0; i < n; ++i) {
        const uint64_t* rr = psi.R.row(i);
        const int iw = i >> 6;
        const uint64_t imask = 1ULL << (i & 63);
        for (int t = 0; t < psi.R.words_; ++t) {
            uint64_t w = rr[t];
            int base = t << 6;
            while (w) {
                int j = base + __builtin_ctzll(w);
                w &= w - 1;
                rcol(j)[iw] |= imask;       // set bit i of column j
            }
        }
    }

    const int Kth = kth_words > 0 ? kth_words : 1;   // ≥1 word for kth-side buffers
    grow(ws.xtmp, mw);
    grow(ws.xd, mw);
    grow(ws.wpsi0, Mkw);
    grow(ws.u0, Kth);
    grow(ws.wpsi, Mkw);
    grow(ws.uscratch, Kth);
    grow(ws.wpsiL, (size_t)r * Mkw);
    grow(ws.duL, (size_t)r * Kth);
    uint64_t* xtmp = ws.xtmp.data();
    uint64_t* xd = ws.xd.data();

    // x = b_ψ ⊕ R_ψ w  (packed, mw words), w given packed (kpsi bits).
    auto xfromw = [&](const uint64_t* wv, uint64_t* x) {
        for (int t = 0; t < mw; ++t) x[t] = bpsi[t];
        for (int j = 0; j < kpsi; ++j)
            if ((wv[j >> 6] >> (j & 63)) & 1ull)
                for (int t = 0; t < mw; ++t) x[t] ^= rcol(j)[t];
    };
    // u = G_θ (x ⊕ b_θ)  (k_θ bits, packed into kth_words).  Per G-row: parity of
    // popcount(Grow & xd) over mw words.  The lambda captures mw (loop-invariant)
    // so the compiler can vectorize the inner scalar popcount loop cleanly; for
    // n ≤ 512 (mw ≤ 8) the scalar XOR-of-popcounts is the fastest available path.
    auto ufromx = [&](const uint64_t* x, uint64_t* u) {
        for (int t = 0; t < mw; ++t) xd[t] = x[t] ^ bth_pk[t];
        for (int t = 0; t < kth_words; ++t) u[t] = 0;
        for (int i = 0; i < kth_; ++i) {
            const uint64_t* gr = Grow(i);
            int par = 0;
            for (int t = 0; t < mw; ++t) par ^= __builtin_popcountll(gr[t] & xd[t]);
            if (par & 1) u[i >> 6] |= (1ULL << (i & 63));
        }
    };

    // ---- Precompute packed (wψ, u) for w0 and each kernel basis vector ----
    // The maps w ↦ wψ (identity, wψ is just w itself, already packed) and
    // w ↦ u = G_θ(b_ψ⊕R_ψw ⊕ b_θ) are AFFINE over GF(2).  So w(z)=w0⊕Σ z_l g_l ⇒
    // wψ(z)=w0 ⊕ Σ z_l g_l (g_l already packed = wψ_l) and u(z)=u0 ⊕ Σ z_l du_l,
    // where du_l = u(w0⊕g_l) ⊕ u(w0).  Each O(r²) probe is then word-XORs of
    // these short buffers + two Q evals.
    uint64_t* wpsi0 = ws.wpsi0.data();
    uint64_t* u0 = ws.u0.data();
    for (int t = 0; t < kpsi_words; ++t) wpsi0[t] = w0[t];   // wψ0 = w0 (no repack)
    xfromw(w0, xtmp);
    ufromx(xtmp, u0);

    {
        // du_l = u(w0 ⊕ g_l) ⊕ u(w0); wψ_l = g_l (already packed in ws.ker).
        uint64_t* wtmp = ws.wpsi.data();   // reuse wpsi as a scratch w-vector here
        uint64_t* utmp = ws.uscratch.data();
        for (int l = 0; l < r; ++l) {
            const uint64_t* gl = kerv(l);
            uint64_t* wpsiL = ws.wpsiL.data() + (size_t)l * Mkw;
            for (int t = 0; t < kpsi_words; ++t) wpsiL[t] = gl[t];
            for (int t = 0; t < kpsi_words; ++t) wtmp[t] = w0[t] ^ gl[t];
            xfromw(wtmp, xtmp);
            ufromx(xtmp, utmp);
            uint64_t* duL = ws.duL.data() + (size_t)l * kth_words;
            for (int t = 0; t < kth_words; ++t) duL[t] = utmp[t] ^ u0[t];
        }
    }

    uint64_t* wpsi = ws.wpsi.data();
    uint64_t* uscratch = ws.uscratch.data();
    auto Efn = [&](const std::vector<uint8_t>& z) -> int {
        for (int t = 0; t < kpsi_words; ++t) wpsi[t] = wpsi0[t];
        for (int t = 0; t < kth_words; ++t) uscratch[t] = u0[t];
        for (int l = 0; l < r; ++l) if (z[l]) {
            const uint64_t* wp = ws.wpsiL.data() + (size_t)l * Mkw;
            for (int t = 0; t < kpsi_words; ++t) wpsi[t] ^= wp[t];
            const uint64_t* du = ws.duL.data() + (size_t)l * kth_words;
            for (int t = 0; t < kth_words; ++t) uscratch[t] ^= du[t];
        }
        int qpsi = eval_Q_packed(kpsi_words, psi.D, psi.J, wpsi);
        int qth  = eval_Q_packed(kth_words, Dth, Jth, uscratch);
        return ((qpsi - qth) % 4 + 4) % 4;
    };

    // Extract E as a ℤ4 quadratic form in z (length-r GF(2) vars).  Same probing
    // pattern as inner_product's form-extraction.
    if ((int)ws.ebuf.size() < r) ws.ebuf.assign(r, 0);
    else for (int i = 0; i < r; ++i) ws.ebuf[i] = 0;
    std::vector<uint8_t>& ebuf = ws.ebuf;
    int c0 = Efn(ebuf);
    if ((int)ws.L.size() < r) ws.L.assign(r, 0);
    std::vector<int>& L = ws.L;
    for (int l = 0; l < r; ++l) {
        ebuf[l] = 1;
        L[l] = ((Efn(ebuf) - c0) % 4 + 4) % 4;
        ebuf[l] = 0;
    }
    // K bilinear coefficients in CLOSED FORM — the port of inner_product's RᵀJR form (commit
    // b609fbc; same proof, verified there over 3·10^5 random forms): for E(z) whose arguments are
    // AFFINE in z through a ℤ4 form (D, J) with difference vectors d_l, the z_a z_b bilinear is
    //   2·[ parity(d_a · (J · d_b))  ⊕  parity(d_a & d_b & (D & 1)) ].
    // Here the ψ-side difference vectors are the kernel basis g_l (= wpsiL rows) and the θ-side
    // are du_l; E = Q_ψ − Q_θ, and −2K ≡ 2K (mod 4), so the two sides XOR. Replaces the old
    // O(r²) two-bit Efn sampling (two eval_Q per probe — the dominant cost at medium k). Verified
    // bit-identical end-to-end by the overlap-vs-inner_product exact fuzz (ovfuzz, 20k pairs).
    static thread_local std::vector<uint64_t> doddpsi, doddth, Jbpsi, Jbth;
    auto pack_Dodd = [](const std::vector<int>& D, int kw, std::vector<uint64_t>& mv) {
        mv.assign(kw > 0 ? kw : 1, 0);
        for (int i = 0; i < (int)D.size(); ++i) if (D[i] & 1) mv[i >> 6] |= (1ULL << (i & 63));
    };
    pack_Dodd(psi.D, kpsi_words, doddpsi);
    pack_Dodd(Dth, kth_words, doddth);
    auto Jmatvec = [](const SymPackedMat& J, int k, int kw, const uint64_t* gv, uint64_t* out) {
        for (int w = 0; w < kw; ++w) out[w] = 0;
        for (int i = 0; i < k; ++i) {
            const uint64_t* jr = J.row(i);
            int s = 0;
            for (int w = 0; w < kw; ++w) s += __builtin_popcountll(jr[w] & gv[w]);
            if (s & 1) out[i >> 6] |= (1ULL << (i & 63));
        }
    };
    auto par_and = [](const uint64_t* a, const uint64_t* b, int nw) {
        int s = 0; for (int w = 0; w < nw; ++w) s += __builtin_popcountll(a[w] & b[w]); return s & 1;
    };
    auto par_and3 = [](const uint64_t* a, const uint64_t* b, const uint64_t* c, int nw) {
        int s = 0; for (int w = 0; w < nw; ++w) s += __builtin_popcountll(a[w] & b[w] & c[w]); return s & 1;
    };
    const int Jpw = kpsi_words > 0 ? kpsi_words : 1;     // safe strides for the kpsi=0 / kth=0 edges
    const int Jtw = kth_words > 0 ? kth_words : 1;
    Jbpsi.resize((size_t)r * Jpw);
    Jbth.resize((size_t)r * Jtw);
    for (int b = 0; b < r; ++b) {
        Jmatvec(psi.J, kpsi, kpsi_words, ws.wpsiL.data() + (size_t)b * Mkw,
                Jbpsi.data() + (size_t)b * Jpw);
        Jmatvec(Jth, kth_, kth_words, ws.duL.data() + (size_t)b * kth_words,
                Jbth.data() + (size_t)b * Jtw);
    }
    std::vector<std::vector<uint8_t>> K(r, std::vector<uint8_t>(r, 0));
    for (int a = 0; a < r; ++a)
        for (int bb = a + 1; bb < r; ++bb) {
            const uint64_t* ga = ws.wpsiL.data() + (size_t)a * Mkw;
            const uint64_t* gb = ws.wpsiL.data() + (size_t)bb * Mkw;
            const uint64_t* da = ws.duL.data() + (size_t)a * kth_words;
            const uint64_t* db = ws.duL.data() + (size_t)bb * kth_words;
            int kbit = par_and(ga, Jbpsi.data() + (size_t)bb * Jpw, kpsi_words)
                     ^ par_and(da, Jbth.data() + (size_t)bb * Jtw, kth_words)
                     ^ par_and3(ga, gb, doddpsi.data(), kpsi_words)
                     ^ par_and3(da, db, doddth.data(), kth_words);
            K[a][bb] = K[bb][a] = (uint8_t)kbit;
        }

    ExactPhase acc = gauss_sum(r, c0, std::vector<int>(L.begin(), L.begin() + r),
                               std::move(K));

    // Combine: conj(ω_θ)·ω_ψ·G · 2^{-(k_θ+k_ψ)/2}.
    ExactPhase result = omth.conj().mul(psi.omega).mul(acc);
    if (result.is_zero) return ExactPhase::zero();
    result.scale -= (kth_ + kpsi);
    return result;
}

}  // namespace qeccore
