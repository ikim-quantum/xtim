#include "qeccore/stab_affine.hpp"
#include <cstddef>
#include "qeccore/gf2_gauss.hpp"   // shared word-packed gf2_solve
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
namespace qeccore {

// Audit counter (declared in stab_state.hpp): bumped on every pairwise state overlap.
unsigned long long g_inner_product_calls = 0;

// i^Q for Q in {0,1,2,3}.
static const std::complex<double> kImagPow[4] = {{1,0},{0,1},{-1,0},{0,-1}};

std::vector<std::complex<double>> AffineState::to_statevector() const {
    assert(n_ < 31 && "to_statevector: 2^n statevector infeasible for n>=31 (and 1<<n is UB for n>=64)");
    std::vector<std::complex<double>> v((size_t)1 << n_, {0.0, 0.0});
    std::complex<double> base = omega.to_complex() * std::pow(2.0, -(double)k_ / 2.0);
    for (uint64_t y = 0; y < (1ull << k_); ++y) {
        std::vector<uint8_t> x = b;                 // x = b ⊕ R y
        for (int j = 0; j < k_; ++j)
            if ((y >> j) & 1)
                for (int i = 0; i < n_; ++i) x[i] ^= R.get(i, j);
        uint64_t idx = 0;
        for (int i = 0; i < n_; ++i) idx |= (uint64_t)x[i] << i;
        int Q = 0;                                  // Q(y) mod 4
        for (int j = 0; j < k_; ++j) if ((y >> j) & 1) Q += D[j];
        for (int a = 0; a < k_; ++a)
            for (int c = a + 1; c < k_; ++c)
                if (((y >> a) & 1) && ((y >> c) & 1)) Q += 2 * J.get(a, c);
        v[idx] += base * kImagPow[Q & 3];
    }
    return v;
}

// --- Task 4: single-qubit gates ---

void AffineState::apply_x(int q) {
    b[q] ^= 1;
}

// Collect the set-bit column indices of R-row q (the support of qubit q's generators).
static inline void row_support(const PackedMat& R, int q, std::vector<int>& out) {
    out.clear();
    const uint64_t* r = R.row(q);
    for (int t = 0; t < R.words_; ++t) {
        uint64_t w = r[t];
        int base = t << 6;
        while (w) {
            int b = __builtin_ctzll(w);
            out.push_back(base + b);
            w &= w - 1;
        }
    }
}

// Allocation-free support primitives for the diagonal gate hot path. They bit-scan
// R-row words directly instead of materializing a std::vector<int> support per call.
// D[]+= and J.flip_sym are commutative, so any enumeration order reproduces the exact
// same D and J as the old "row_support + O(|sup|^2) double loop" (byte-identical).

// D[j] += add (mod 4) for each set column j of the packed row `r` (W words).
static inline void add_row_support_D(std::vector<int>& D, const uint64_t* r, int W, int add) {
    for (int t = 0; t < W; ++t) {
        uint64_t w = r[t];
        int base = t << 6;
        while (w) { int j = base + __builtin_ctzll(w); D[j] = (D[j] + add) & 3; w &= w - 1; }
    }
}

// Flip J over every unordered pair {a,c} of set columns of the packed row `r` (W words).
// `a` is always the lower index (low word/low bit first), so each pair is emitted once.
static inline void flip_row_support_pairs(SymPackedMat& J, const uint64_t* r, int W) {
    for (int t = 0; t < W; ++t) {
        uint64_t wt = r[t];
        int baset = t << 6;
        for (uint64_t wu = wt; wu; wu &= wu - 1) {
            int a = baset + __builtin_ctzll(wu);
            for (uint64_t wd = wu & (wu - 1); wd; wd &= wd - 1)   // partners: higher bits, same word
                J.flip_sym(a, baset + __builtin_ctzll(wd));
            for (int u = t + 1; u < W; ++u) {                     // partners: all bits in higher words
                uint64_t wv = r[u];
                int baseu = u << 6;
                for (uint64_t wd = wv; wd; wd &= wd - 1)
                    J.flip_sym(a, baseu + __builtin_ctzll(wd));
            }
        }
    }
}

void AffineState::apply_z(int q) {
    uint8_t bq = b[q];
    omega.add_z8(4 * bq);                 // (-1)^{bq}
    const uint64_t* r = R.row(q);
    for (int t = 0; t < R.words_; ++t) {  // (-1)^{c_j y_j}: D[j] += 2 for each set c_j
        uint64_t w = r[t];
        int base = t << 6;
        while (w) {
            int j = base + __builtin_ctzll(w);
            D[j] = (D[j] + 2) & 3;
            w &= w - 1;
        }
    }
}

void AffineState::apply_s(int q) {
    uint8_t bq = b[q];
    omega.add_z8(2 * bq);                 // i^{bq}
    const uint64_t* r = R.row(q);
    add_row_support_D(D, r, R.words_, bq ? 3 : 1);   // cj + 2*bq*cj for cj=1
    flip_row_support_pairs(J, r, R.words_);
}

void AffineState::apply_sdg(int q) {
    uint8_t bq = b[q];
    omega.add_z8(-2 * bq);                // (-i)^{bq}
    const uint64_t* r = R.row(q);
    add_row_support_D(D, r, R.words_, bq ? 1 : 3);   // (-1 + 2*bq) mod 4 for cj=1
    flip_row_support_pairs(J, r, R.words_);
}

void AffineState::apply_y(int q) {
    apply_z(q);
    apply_x(q);
    omega.add_z8(2);                      // Y = i·X·Z
}

void AffineState::apply_h(int q) {
    uint8_t bq = b[q];
    const int cidx = k_;
    const int W = R.words_;

    // Snapshot old R-row q's support words BEFORE zeroing/append (the column append may
    // reallocate R, invalidating the row pointer). Its set bits are the j with which the
    // new generator column cidx couples in J. Reused per-thread scratch (no per-call alloc).
    thread_local std::vector<uint64_t> oldrow;
    oldrow.assign(W, 0);
    {
        const uint64_t* r = R.row(q);
        for (int t = 0; t < W; ++t) oldrow[t] = r[t];
    }

    // Zero qubit q's dependence on existing generators / constant part.
    {
        uint64_t* r = R.row(q);
        for (int t = 0; t < W; ++t) r[t] = 0;
    }
    b[q] = 0;

    // Append the new generator column cidx = e_q. The column region beyond cols_ is
    // invariant-zero (trailing bits kept zero), so appending an all-zero-except-bit-q
    // column is: ensure capacity (same reserve policy as append_col), bump cols_, set
    // the single bit q. O(1) + occasional grow, vs append_col's O(n) per-row writes and
    // an n-byte temporary vector.
    {
        const int j = R.cols_;
        if (((R.cols_ + 64) + 63) / 64 > R.words_) R.reserve_words((R.cols_ + 64) / 64 + 1);
        if ((j >> 6) >= R.words_) R.reserve_words((j >> 6) + 1);
        R.cols_ += 1;
        R.set(q, j, 1);
    }
    D.push_back(0);
    // Grow J to (k_+1)x(k_+1) with a zero new row/col (packed symmetric).
    J.append_zero();

    // Phase (-1)^{c·x_q} = i^{2c·bq + 2c(rq·y)}: D[cidx] += 2·bq, and J[cidx][j] ^= 1
    // for each j in old support of row q (bit-scanned straight from the snapshot).
    D[cidx] = (D[cidx] + 2 * bq) & 3;
    for (int t = 0; t < W; ++t) {
        uint64_t w = oldrow[t];
        int base = t << 6;
        while (w) { J.flip_sym(cidx, base + __builtin_ctzll(w)); w &= w - 1; }
    }

    k_ += 1;
    canonicalize();
}

// ---- canonicalize() helpers (file-local) ----
namespace {

// Quadratic part of Q from a packed symmetric zero-diagonal J, given the packed
// bit-vector y (word-array yw, length J.words()). J symmetric with zero diagonal,
// Drop generator column l from (R, D, J); decrements k_.
void drop_col(AffineState& s, int l) {
    s.R.drop_col(l);
    s.D.erase(s.D.begin() + l);
    s.J.drop(l);
    s.k_ -= 1;
}

// Form-only substitution: in Q(y) replace variable `p` by (y_p XOR y_c), c != p.
// Updates (D,J) in closed form, O(k). Does NOT touch R/b.
//
// Derivation, using y_p⊕y_c = y_p + y_c - 2 y_p y_c over the integers, Q mod 4
// (J-entries are the coeff of 2·y_a y_b, so only their parity matters):
//   * D_p y_p  →  D_p(y_p + y_c - 2 y_p y_c):  D_c += D_p;  J_pc ^= (D_p & 1).
//   * 2 J_pj y_p y_j (j≠p) → +2 J_pj y_c y_j (the -4(...) term dies mod 4):
//       j≠c: J_cj ^= J_pj;   j=c: D_c += 2 J_pc  (y_c²=y_c).
// Order matters: read original D_p and J_pc first.
inline void subst_form_var(int /*k*/, std::vector<int>& D,
                           SymPackedMat& J,
                           int p, int c) {
    const int Dp = D[p];
    const uint8_t Jpc = J.get(p, c);

    // Snapshot row p BEFORE any mutation (the symmetric column update below flips
    // bits in arbitrary rows, including p and c, so the loop must read a copy of
    // the original Jp). words = J.words(); only bits [0,k) are meaningful.
    const int words = J.words();
    std::vector<uint64_t> Jp(words);
    {
        const uint64_t* src = J.row(p);
        for (int t = 0; t < words; ++t) Jp[t] = src[t];
    }

    // Mask out bits p and c from the row-p snapshot: the original loop skips j==p
    // and j==c. (Diagonal bit p is always 0; bit c is the Jpc term handled via D.)
    Jp[p >> 6] &= ~(1ULL << (p & 63));
    Jp[c >> 6] &= ~(1ULL << (c & 63));

    // row_c ^= (masked row_p): this is the j≠p,c part of  Jc[j] ^= Jp[j].
    {
        uint64_t* rc = J.row(c);
        for (int t = 0; t < words; ++t) rc[t] ^= Jp[t];
    }
    // Symmetric column-c update: for each set bit j of (masked) row_p, flip bit c
    // of row j (i.e. J[j][c] ^= 1). Strided, O(popcount). This also re-flips the
    // J[c][c]/diagonal? No: j ranges over set bits of Jp which excludes p,c, and
    // c∉{j}, so flip_sym(j,c) toggles (j,c) and (c,j) — keeping symmetry. The
    // (c,j) flips were already applied by the word-XOR above into row c; flip_sym
    // would double-flip them, so flip ONLY the (j,c) half here.
    for (int t = 0; t < words; ++t) {
        uint64_t w = Jp[t];
        int base = t << 6;
        while (w) {
            int j = base + __builtin_ctzll(w);
            w &= w - 1;
            // flip bit c of row j (the symmetric counterpart of the row-c XOR).
            J.row(j)[c >> 6] ^= (1ULL << (c & 63));
        }
    }

    D[c] = (D[c] + Dp + 2 * (int)Jpc) & 3;
    if (Dp & 1) J.flip_sym(c, p);              // J_pc ^= (D_p & 1)
}

// Invertible change of generator basis that XORs column `src` into column `ctrl`
// of R (i.e. new R-column ctrl = old ctrl XOR src), leaving the state unchanged.
// Equivalently the variable substitution: old y_src = w_src XOR w_ctrl, i.e. in
// Q(y) replace variable `src` by (y_src XOR y_ctrl). R gets the column op; (D,J)
// the closed-form form substitution. O(k) — replaces the old O(k^2) eval-Q rebuild.
void add_var(AffineState& s, int src, int ctrl) {
    // Column op: R-column ctrl ^= R-column src (per-row bit XOR).
    const int sw = src >> 6, sb = src & 63, cw = ctrl >> 6, cb = ctrl & 63;
    for (int i = 0; i < s.n_; ++i) {
        uint64_t* r = s.R.row(i);
        if ((r[sw] >> sb) & 1ULL) r[cw] ^= (1ULL << cb);
    }
    subst_form_var(s.k_, s.D, s.J, src, ctrl);
}

// Eliminate variable p via the affine relation y_p = a0 XOR (XOR_{s in S} y_s).
// Updates R, b (constant fold), D, J, omega, and k_.
//
// Closed-form: apply the substitution y_p -> y_p XOR y_s for each s in S (each a
// form-only subst_form_var, net p -> p XOR (XOR_{s in S} y_s)), then substitute
// the constant y_p -> a0 and drop variable p. Constant substitution into the form
// at this point: term D_p y_p -> D_p a0 (constant -> omega); 2 J_pj y_p y_j ->
// 2 a0 J_pj y_j (linear: D_j += 2 a0 J_pj). O(k + |S|*k) instead of the old
// O(k^2) eval-Q rebuild.
void eliminate(AffineState& s, int p, uint8_t a0, const std::vector<int>& S) {
    const int k = s.k_;
    // 1. Variable substitutions p -> p XOR y_s (form only).
    for (int sx : S) subst_form_var(k, s.D, s.J, p, sx);

    // 2. Constant substitution y_p -> a0, into the (now updated) form.
    int cst = (int)((s.D[p] * (int)a0) & 3);      // D_p a0  -> omega
    if (a0) {
        const uint64_t* Jp = s.J.row(p);
        const int words = s.J.words();
        for (int t = 0; t < words; ++t) {
            uint64_t w = Jp[t];
            int base = t << 6;
            while (w) {
                int j = base + __builtin_ctzll(w);
                w &= w - 1;
                if (j != p) s.D[j] = (s.D[j] + 2) & 3;        // 2 a0 J_pj y_j
            }
        }
    }

    // 3. Apply the relation to R / b: column p folds into a0 (->b) and S columns.
    const int pw = p >> 6, pb = p & 63;
    for (int i = 0; i < s.n_; ++i) {
        uint64_t* r = s.R.row(i);
        if ((r[pw] >> pb) & 1ULL) {        // cp == 1
            s.b[i] ^= a0;
            for (int sx : S) r[sx >> 6] ^= (1ULL << (sx & 63));
        }
    }

    // 4. Drop variable/column p from R and the form.
    drop_col(s, p);
    s.omega.add_z8(2 * cst);                   // i^{cst} = zeta8^{2 cst}
}

// Project an affine state onto Z_q = (-1)^{x_q} eigenspace with x_q = t (0/1).
// x_q = b[q] ⊕ (R-row q · y). Returns α and mutates s to the collapsed state.
// static: file-local (no header decl); avoids ODR collisions across TUs.
static ExactPhase project_z(AffineState& s, int q, int t) {
    thread_local std::vector<int> rq;                  // reused per-thread (sampler measure path)
    row_support(s.R, q, rq);                            // row_support clears `out` first
    if (rq.empty()) {                                  // x_q deterministic = b[q]
        return (s.b[q] == (uint8_t)t) ? ExactPhase::one() : ExactPhase::zero();
    }
    int p = rq[0];
    thread_local std::vector<int> S;
    S.assign(rq.begin() + 1, rq.end());
    uint8_t a0 = (uint8_t)(t ^ s.b[q]);                // y_p = (t⊕b_q) ⊕ ⊕_{j∈S} y_j
    eliminate(s, p, a0, S);
    return ExactPhase::one().mul_inv_sqrt2();          // 2^(−1/2)
}

// Find AT MOST ONE right-null-space vector of R (n×k, columns = generators),
// exploiting the H-invariant that the post-H nullity is ≤ 1. Returns true and
// fills `v` (length k) with the single kernel vector if R is column-rank-
// deficient; returns false (R full column rank, kernel trivial) otherwise.
//
// Bit-exactness: the 1-dimensional kernel of a rank-(k-1) GF(2) matrix contains
// exactly ONE nonzero vector, so the result is canonical and independent of the
// elimination order — it is byte-identical to kernel(R) row 0. For the (H never
// produces it, but the canonicalize loop is robust to it) nullity≥2 case we
// reproduce kernel()'s convention exactly: same augmented matrix [R^T | I_k],
// same ascending-column Gauss–Jordan, returning the FIRST non-pivot row's I_k
// block — identical to the old kernel(R).get(0,·).
//
// Cost: one O(n·k / 64) bit-packed Gauss–Jordan pass that early-exits the moment
// all k columns become pivots (the common full-rank no-op H), versus kernel()'s
// from-scratch full-basis elimination plus its GF2Mat allocations.
static bool find_single_kernel_vec(const PackedMat& R, int n, int k,
                                   std::vector<uint8_t>& v) {
    if (k == 0) return false;

    // Fast rank test. Column rank == row rank, so if the n rows span all k dimensions, R
    // has full column rank and the kernel is trivial — return false WITHOUT building the
    // O(n·k) transpose A below. This is the common no-op H (post-H nullity 0). Only the
    // rare rank-deficient case falls through to the exact kernel-vector extraction (which
    // must stay byte-identical to kernel(), so its code path is left untouched). A standard
    // leading-bit row-space sieve with early-exit the moment rank reaches k.
    //
    // The result is a pure rank predicate (full vs deficient) — it never affects the kernel
    // vector returned in the deficient case — so any GF(2) reduction order is correct.
    if (k <= 64) {
        const uint64_t kmask = (k == 64) ? ~0ULL : ((1ULL << k) - 1);
        uint64_t pivots[64] = {0};   // pivots[c] = reduced row with leading set bit c
        int rank = 0;
        for (int i = 0; i < n; ++i) {
            uint64_t row = R.row(i)[0] & kmask;
            while (row) {
                int c = __builtin_ctzll(row);
                if (!pivots[c]) { pivots[c] = row; if (++rank == k) return false; break; }
                row ^= pivots[c];
            }
        }
        // rank < k here ⇒ R is column-rank-deficient ⇒ fall through to extract the vector.
    } else {
        // k > 64: same sieve over word-packed rows (each R-row is R.words_ words; only the
        // low k columns matter). One pivot row per leading column, kept in a flat
        // (k × kw) thread-local scratch so the common full-rank case early-exits at
        // O(n·k/64) instead of building+reducing the O(n+k)-wide augmented matrix.
        const int kw = (k + 63) >> 6;
        const uint64_t topmask = (k & 63) ? ((1ULL << (k & 63)) - 1) : ~0ULL;
        thread_local std::vector<uint64_t> piv;     // pivots[c]·kw words; piv_set[c] occupancy
        thread_local std::vector<uint8_t> piv_set;
        piv.assign((size_t)k * kw, 0);
        piv_set.assign(k, 0);
        thread_local std::vector<uint64_t> row;
        row.resize(kw);
        int rank = 0;
        for (int i = 0; i < n; ++i) {
            const uint64_t* src = R.row(i);
            for (int w = 0; w < kw; ++w) row[w] = src[w];
            row[kw - 1] &= topmask;                 // mask columns ≥ k (R.words_ may exceed kw)
            for (;;) {
                int c = -1;
                for (int w = 0; w < kw; ++w) if (row[w]) { c = (w << 6) + __builtin_ctzll(row[w]); break; }
                if (c < 0) break;                    // row reduced to zero
                if (!piv_set[c]) {
                    uint64_t* dst = piv.data() + (size_t)c * kw;
                    for (int w = 0; w < kw; ++w) dst[w] = row[w];
                    piv_set[c] = 1;
                    if (++rank == k) return false;   // full column rank ⇒ kernel trivial
                    break;
                }
                const uint64_t* pr = piv.data() + (size_t)c * kw;
                for (int w = 0; w < kw; ++w) row[w] ^= pr[w];
            }
        }
        // rank < k here ⇒ R is column-rank-deficient ⇒ fall through to extract the vector.
    }

    // Augmented matrix A: k rows × (n + k) cols, row j = (R-column j | e_j).
    // Columns 0..n-1 hold R^T (A[j] bit i = R(i,j)); columns n..n+k-1 hold I_k.
    const int ncols = n + k;
    const int words = (ncols + 63) / 64;
    // Reusable per-thread scratch: this runs once per H (canonicalize) on the hot
    // replay path; a fresh heap vector per call was pure churn. thread_local keeps it
    // race-free under the parallel apply-to-many usage. Zero only the words we use.
    thread_local std::vector<uint64_t> A;
    A.assign((size_t)k * words, 0);
    auto Arow = [&](int j) -> uint64_t* { return A.data() + (size_t)j * words; };
    // Fill R^T block: for each R-row i, scatter its set columns j into A[j] bit i.
    for (int i = 0; i < n; ++i) {
        const uint64_t* r = R.row(i);
        const int iw = i >> 6, ibit = i & 63;
        const uint64_t imask = 1ULL << ibit;
        for (int t = 0; t < R.words_; ++t) {
            uint64_t w = r[t];
            int base = t << 6;
            while (w) {
                int j = base + __builtin_ctzll(w);
                Arow(j)[iw] |= imask;
                w &= w - 1;
            }
        }
    }
    for (int j = 0; j < k; ++j) Arow(j)[(n + j) >> 6] |= (1ULL << ((n + j) & 63));

    // Gauss–Jordan over the first n columns (the R^T block), ascending — exactly
    // kernel()'s pivot order. Track rank r; early-exit once r == k (full rank).
    int r = 0;
    for (int c = 0; c < n && r < k; ++c) {
        const int cw = c >> 6;
        const uint64_t cmask = 1ULL << (c & 63);
        int piv = -1;
        for (int i = r; i < k; ++i)
            if (Arow(i)[cw] & cmask) { piv = i; break; }
        if (piv < 0) continue;
        if (piv != r) {
            uint64_t* a = Arow(r);
            uint64_t* bp = Arow(piv);
            for (int w = 0; w < words; ++w) std::swap(a[w], bp[w]);
        }
        const uint64_t* pr = Arow(r);
        for (int i = 0; i < k; ++i)
            if (i != r && (Arow(i)[cw] & cmask)) {
                uint64_t* x = Arow(i);
                for (int w = 0; w < words; ++w) x[w] ^= pr[w];
            }
        ++r;
    }
    if (r == k) return false;  // full column rank: kernel trivial.

    // First non-pivot row is row r; its I_k block (cols n..n+k-1) is kernel vec 0.
    v.assign(k, 0);
    const uint64_t* kr = Arow(r);
    for (int j = 0; j < k; ++j) {
        int col = n + j;
        if (kr[col >> 6] & (1ULL << (col & 63))) v[j] = 1;
    }
    return true;
}

// Exact diagonal ⟨φ|Z_q|φ⟩ ∈ {+1,−1,0}: x_q = b[q] ⊕ (R-row q · y). If R-row q is
// zero, x_q is pinned to b[q] ⇒ ⟨Z_q⟩ = (−1)^{b[q]}; else x_q is balanced ⇒ 0.
static int diag_z(const AffineState& s, int q) {
    thread_local std::vector<int> rq;                  // reused per-thread (sampler measure path)
    row_support(s.R, q, rq);
    if (!rq.empty()) return 0;
    return (s.b[q] == 0) ? +1 : -1;
}

}  // namespace

void AffineState::canonicalize() {
    thread_local std::vector<uint8_t> v;   // reused single-kernel-vector buffer (per-thread)
    for (;;) {
        if (k_ == 0) break;
        // Targeted single-dependency check: find AT MOST ONE kernel vector of R
        // (nullity ≤ 1 after H), replacing the full from-scratch kernel() basis.
        if (!find_single_kernel_vec(R, n_, k_, v)) break;  // full column rank

        // One kernel vector: v with R-column l == XOR of {R-column j : v_j=1, j!=l}.
        int l = -1;
        for (int j = 0; j < k_; ++j) if (v[j]) { l = j; break; }
        thread_local std::vector<int> T;
        T.clear();
        for (int j = 0; j < k_; ++j) if (v[j] && j != l) T.push_back(j);

        // Zero R-column l by absorbing the other support columns into it.
        for (int j : T) add_var(*this, j, l);

        // Now column l is zero; integrate out the free variable y_l. Q is at most
        // linear in y_l: L(y_-l) = D[l] + 2*sum_{j in Scpl} y_j, sum_{y_l} i^Q = i^{Q0}(1+i^L).
        int Dl = D[l];
        thread_local std::vector<int> Scpl;    // support of J-row l, excluding l (per-thread)
        Scpl.clear();
        {
            const uint64_t* Jl = J.row(l);
            const int words = J.words();
            for (int t = 0; t < words; ++t) {
                uint64_t w = Jl[t];
                int base = t << 6;
                while (w) {
                    int j = base + __builtin_ctzll(w);
                    w &= w - 1;
                    if (j != l) Scpl.push_back(j);
                }
            }
        }

        if (Dl & 1) {
            // 1+i^{Dl+2t} = sqrt2 * zeta8^{e(t)}, e(t) = 2 - ((Dl+2t) mod 4), with
            // t = XOR of the coupled survivor vars (Scpl). The constant part e(0)
            // goes to omega (zeta8 units); the t-dependent part stays in the form.
            // D[] tracks i-exponents (period 4); zeta8 has period 8, so a zeta8
            // step de = e(1)-e(0) (always even) maps to an i-power step of de/2.
            //
            // For |Scpl| >= 2: writing XOR(y_j...) in terms of sums requires a
            // quadratic correction: zeta8^{de*(XOR)} = zeta8^{de*(sum)} * zeta8^{-2*de*(sum of pairs)}.
            // Since -2*de = ∓4, zeta8^{∓4} = i^{∓2} and each J[a][b] flip contributes
            // i^2 per pair when y_a=y_b=1. So flip J for all pairs in Scpl.
            int e0 = 2 - (Dl & 3);
            int de = (Dl & 3) - ((Dl + 2) & 3);   // e(1)-e(0), always ±2 (a zeta8 step)
            omega.add_z8(e0);
            int dD = (((de / 2) % 4) + 4) % 4;     // zeta8 step -> i-power step
            for (int j : Scpl) D[j] = (D[j] + dD) & 3;
            // Cross-term correction for |Scpl|>=2: flip J for every pair.
            for (int ai = 0; ai < (int)Scpl.size(); ++ai)
                for (int bi = ai + 1; bi < (int)Scpl.size(); ++bi)
                    J.flip_sym(Scpl[ai], Scpl[bi]);
            drop_col(*this, l);                     // sqrt2 cancels base; no scale change
        } else {
            uint8_t a0 = (uint8_t)(Dl / 2);         // need t == a0 (projector)
            if (Scpl.empty()) {
                if (a0 == 1) { omega = ExactPhase::zero(); return; }  // 1+i^2 = 0
                omega.scale += 1;                   // factor 2, then drop -> net sqrt2
                drop_col(*this, l);
            } else {
                int p = Scpl[0];
                thread_local std::vector<int> rest;
                rest.assign(Scpl.begin() + 1, Scpl.end());
                eliminate(*this, p, a0, rest);      // enforce t == a0
                if (l > p) l -= 1;                  // p removed, reindex l
                drop_col(*this, l);                 // fold factor 2 absorbs the var drops
            }
        }
        // loop (H creates at most one dependency, but be robust to more).
    }
}

// --- Task 5: 2-qubit gates ---

void AffineState::apply_cx(int c, int t) {
    // CX(c,t): |x> -> |x XOR x_c e_t>, i.e. bit t gets XORed with bit c.
    // Output bit t: new x_t = x_t XOR x_c = (bt XOR bc) XOR ((Rt XOR Rc) . y)
    // So: b[t] ^= b[c], R[t][j] ^= R[c][j] for all j. D, J, omega unchanged.
    // No canonicalize() needed: a row op preserves the column rank of R.
    b[t] ^= b[c];
    R.xor_row(t, c);                       // word-wide row XOR
}

void AffineState::apply_cz(int c, int t) {
    // CZ(c,t): amplitude x(-1)^{x_c x_t}. x_c = bc XOR (Rc.y), x_t = bt XOR (Rt.y).
    // Expand x_c x_t = bc*bt XOR bc*(Rt.y) XOR bt*(Rc.y) XOR (Rc.y)(Rt.y)  (mod 2)
    // CZ does not modify R, b -- only D, J, omega.
    uint8_t bc = b[c], bt = b[t];

    // Global phase: (-1)^{bc*bt} = zeta8^{4*bc*bt}
    omega.add_z8(4 * (int)(bc * bt));

    const uint64_t* rc = R.row(c);
    const uint64_t* rt = R.row(t);
    const int W = R.words_;

    // Diagonal update: each y_j picks up (-1)^{bc*Rt_j + bt*Rc_j + Rc_j*Rt_j}
    // = i^{2*(bc*Rt_j + bt*Rc_j + Rc_j*Rt_j)}. Only j in supp(Rc) ∪ supp(Rt) matter.
    if (bt) add_row_support_D(D, rc, W, 2);             // bt*Rc_j
    if (bc) add_row_support_D(D, rt, W, 2);             // bc*Rt_j
    for (int wi = 0; wi < W; ++wi) {                    // Rc_j*Rt_j (j in both)
        uint64_t w = rc[wi] & rt[wi];
        int base = wi << 6;
        while (w) { int j = base + __builtin_ctzll(w); D[j] = (D[j] + 2) & 3; w &= w - 1; }
    }

    // Off-diagonal update: J[a][b] ^= (Rc_a*Rt_b XOR Rc_b*Rt_a). Visit ordered
    // pairs (x in supp Rc, y in supp Rt), x!=y; each ordered visit toggles the
    // symmetric entry once, giving the correct parity for {a,b}.
    for (int wi = 0; wi < W; ++wi) {
        uint64_t wc = rc[wi];
        int basec = wi << 6;
        while (wc) {
            int x = basec + __builtin_ctzll(wc);
            wc &= wc - 1;
            for (int wj = 0; wj < W; ++wj) {
                uint64_t wy = rt[wj];
                int baset = wj << 6;
                while (wy) {
                    int y = baset + __builtin_ctzll(wy);
                    wy &= wy - 1;
                    if (x != y) J.flip_sym(x, y);
                }
            }
        }
    }
}

// ---- Task 6: exact inner product ⟨φ|ψ⟩ via GF(2) Gauss sum ----
// gf2_solve and gauss_sum now both live in qeccore/gf2_gauss.hpp (shared).
// gf2_solve is
// word-packed; gauss_sum keeps the byte-K recursion (word-packing it gave no
// speedup at the small r typical of overlaps). They are pulled into this
// translation unit's helper namespace below so the call sites resolve unchanged.
namespace {
using qeccore::gf2_solve;
using qeccore::gf2_solve_w1;   // shared single-word solver (was file-local here)
using qeccore::gauss_sum;

}  // namespace

ExactPhase AffineState::inner_product(const StabState& other_base) const {
    ++g_inner_product_calls;
    const AffineState& phi = *this;
    const AffineState& psi = dynamic_cast<const AffineState&>(other_base);

    const int kphi = phi.k_, kpsi = psi.k_, n = phi.n_;
    const int N = kphi + kpsi;
    const int kw = (N + 63) >> 6;            // words per packed system row
    const int kphi_words = (kphi + 63) / 64;
    const int kpsi_words = (kpsi + 63) / 64;

    // Build the packed system A = [R_phi | R_psi] (n × N) and rhs = b_phi ⊕ b_psi
    // DIRECTLY in word-packed form from the row-packed R words — the per-bit byte
    // matrix this replaces (plus gf2_solve's internal re-packing of it) was ~2/3 of
    // the whole call at small k. R rows keep trailing bits beyond k_ zero (PackedMat
    // invariant), so phi's kphi_words copy verbatim and psi's words funnel-shift into
    // bit positions [kphi, N). Solved with the shared allocation-free gf2_solve_packed,
    // whose algorithm and result are bit-identical to the byte gf2_solve (same pivot
    // scan, same particular solution, same nullspace basis order). Per-thread reusable
    // scratch throughout — this is called O(χ²) times in pairwise-overlap measurements.
    static thread_local std::vector<uint64_t> Mwork, part_pk, ker_pk;
    static thread_local std::vector<int> pivot_col, col_pivot;
    int r = 0;
    const int off = kphi >> 6, sh = kphi & 63;
    if (N <= 63) {
        // Single-word fast path: the whole augmented row [A | rhs] fits one uint64
        // (phi bits [0,kphi), psi bits [kphi,N), rhs at bit N), built straight from
        // R's first row words — both R's are single-word here, and bits beyond k_
        // are zero by the PackedMat invariant. Solved in place by gf2_solve_w1 into
        // the same part_pk/ker_pk layout the general path fills (kw = 1), so all
        // downstream code is shared.
        if ((int)Mwork.size() < n) Mwork.resize(n);
        const uint64_t rhsbit = 1ull << N;
        for (int i = 0; i < n; ++i) {
            uint64_t a = kphi ? phi.R.row(i)[0] : 0ull;
            if (kpsi) a |= psi.R.row(i)[0] << kphi;
            if ((phi.b[i] ^ psi.b[i]) & 1) a |= rhsbit;
            Mwork[i] = a;
        }
        if (part_pk.empty()) part_pk.resize(1);
        if ((int)ker_pk.size() < N) ker_pk.resize(N);
        const bool ok = gf2_solve_w1(Mwork.data(), n, N, part_pk[0], ker_pk.data(), r);
#ifdef IP_W1_XCHECK
        {   // TEMP differential check vs the original byte gf2_solve (build-flag only)
            std::vector<std::vector<uint8_t>> A2(n, std::vector<uint8_t>(N, 0));
            std::vector<uint8_t> rhs2(n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < kphi; ++j) A2[i][j] = phi.R.get(i, j);
                for (int j = 0; j < kpsi; ++j) A2[i][kphi + j] = psi.R.get(i, j);
                rhs2[i] = phi.b[i] ^ psi.b[i];
            }
            std::vector<uint8_t> p02;
            std::vector<std::vector<uint8_t>> ker2;
            const bool ok2 = gf2_solve(A2, rhs2, N, p02, ker2);
            if (ok != ok2) { fprintf(stderr, "w1 XCHECK: ok mismatch\n"); abort(); }
            if (ok) {
                if ((int)ker2.size() != r) { fprintf(stderr, "w1 XCHECK: r mismatch\n"); abort(); }
                uint64_t pw = 0;
                for (int j = 0; j < N; ++j) if (p02[j]) pw |= 1ull << j;
                if (pw != part_pk[0]) { fprintf(stderr, "w1 XCHECK: part mismatch\n"); abort(); }
                for (int l = 0; l < r; ++l) {
                    uint64_t gw = 0;
                    for (int j = 0; j < N; ++j) if (ker2[l][j]) gw |= 1ull << j;
                    if (gw != ker_pk[l]) { fprintf(stderr, "w1 XCHECK: ker mismatch\n"); abort(); }
                }
            }
        }
#endif
        if (!ok) return ExactPhase::zero();  // disjoint supports
    } else {
        static thread_local std::vector<uint64_t> Apk, rhspk;
        Apk.assign((size_t)n * kw, 0ull);
        rhspk.assign(((size_t)n + 63) >> 6, 0ull);
        for (int i = 0; i < n; ++i) {
            uint64_t* arow = Apk.data() + (size_t)i * kw;
            const uint64_t* pr = phi.R.row(i);
            for (int w = 0; w < kphi_words; ++w) arow[w] = pr[w];
            const uint64_t* sr = psi.R.row(i);
            if (sh == 0) {
                for (int w = 0; w < kpsi_words; ++w) arow[off + w] |= sr[w];
            } else {
                for (int w = 0; w < kpsi_words; ++w) {
                    const uint64_t v = sr[w];
                    arow[off + w] |= v << sh;
                    // High spill: any nonzero bit lands at global position < N, so the
                    // target word exists whenever the spill is nonzero (bounds-guard only).
                    if (off + w + 1 < kw) arow[off + w + 1] |= v >> (64 - sh);
                }
            }
            if ((phi.b[i] ^ psi.b[i]) & 1) rhspk[(size_t)i >> 6] |= 1ull << (i & 63);
        }
        if (!gf2_solve_packed(Apk, n, N, rhspk, Mwork, pivot_col, col_pivot, part_pk, ker_pk, r))
            return ExactPhase::zero();       // disjoint supports
    }

    // ---- Word-packed form-extraction scratch (Opt B+C) ----
    // E(z) = (Q_psi(w(z)) − Q_phi(u(z))) mod 4, with x(z) = p0 ⊕ Σ z_l g_l.
    // The phi-variables are bits [0,kphi) of x; psi-variables are [kphi,N).
    // We give each its own aligned uint64 buffer so neither J-eval nor the
    // kernel-XOR has to cross sub-range word boundaries. Built ONCE here and
    // reused across the O(r²) Efn calls below.

    // Split a packed length-N solution vector into its phi-part (bits [0,kphi)) and
    // psi-part (bits [kphi,N), shifted down to start at 0). Pure word ops; identical
    // output to the old per-bit byte pack_split. The psi top-word mask is belt-and-
    // braces (the solver leaves bits ≥ N zero); the phi mask is REQUIRED — psi bits
    // share the boundary word when kphi isn't word-aligned.
    auto split_pk = [&](const uint64_t* v, uint64_t* outphi, uint64_t* outpsi) {
        for (int w = 0; w < kphi_words; ++w) outphi[w] = v[w];
        if (kphi & 63) outphi[kphi_words - 1] &= (1ull << (kphi & 63)) - 1;
        for (int w = 0; w < kpsi_words; ++w) {
            const uint64_t lo = v[off + w] >> sh;
            const uint64_t hi = (sh && off + w + 1 < kw) ? (v[off + w + 1] << (64 - sh)) : 0ull;
            outpsi[w] = lo | hi;
        }
        if (kpsi & 63) outpsi[kpsi_words - 1] &= (1ull << (kpsi & 63)) - 1;
    };
    // Flat per-thread scratch (every word is overwritten before use): the p0 split,
    // the r kernel splits at strides kphi_words/kpsi_words, and the Efn x-accumulators.
    // Replaces fresh per-call vectors and a vector-of-vectors per side.
    static thread_local std::vector<uint64_t> p0phi, p0psi, xphi, xpsi, kerphi, kerpsi;
    p0phi.resize(kphi_words); p0psi.resize(kpsi_words);
    xphi.resize(kphi_words);  xpsi.resize(kpsi_words);
    kerphi.resize((size_t)r * kphi_words);
    kerpsi.resize((size_t)r * kpsi_words);
    split_pk(part_pk.data(), p0phi.data(), p0psi.data());
    for (int l = 0; l < r; ++l)
        split_pk(ker_pk.data() + (size_t)l * kw,
                 kerphi.data() + (size_t)l * kphi_words,
                 kerpsi.data() + (size_t)l * kpsi_words);

    // The stored J is now ALREADY bit-packed (SymPackedMat, full symmetric storage,
    // zero diagonal) — no per-call byte→pack copy needed. eval_Q_packed reads its
    // word-rows directly. Note J.words() may exceed kw (capacity grows in blocks),
    // but we only AND/popcount the first kw words against the x buffer (width kw);
    // bits ≥ k are zero in both, so this is exact.

    // Packed Q(x) mod 4 for the form (D, J) of dimension k.
    // Linear: Σ_i D_i x_i over set bits (ctz). Quadratic: J symmetric
    // with zero diagonal, so Σ_{i:x_i} popcount(Jrow_i & x) counts each set
    // pair {i,j} twice; halve to get Σ_{i<j} J_ij x_i x_j, then ×2 mod 4.
    auto eval_Q_packed = [](int kw, const std::vector<int>& D,
                            const SymPackedMat& J,
                            const uint64_t* x) -> int {
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
    };

    // Efn accumulates into the reused thread_local xphi/xpsi declared with the splits.
    auto Efn = [&](const std::vector<uint8_t>& z) -> int {
        for (int w = 0; w < kphi_words; ++w) xphi[w] = p0phi[w];
        for (int w = 0; w < kpsi_words; ++w) xpsi[w] = p0psi[w];
        for (int l = 0; l < r; ++l) if (z[l]) {
            const uint64_t* kp = kerphi.data() + (size_t)l * kphi_words;
            for (int w = 0; w < kphi_words; ++w) xphi[w] ^= kp[w];
            const uint64_t* ks = kerpsi.data() + (size_t)l * kpsi_words;
            for (int w = 0; w < kpsi_words; ++w) xpsi[w] ^= ks[w];
        }
        int qpsi = eval_Q_packed(kpsi_words, psi.D, psi.J, xpsi.data());
        int qphi = eval_Q_packed(kphi_words, phi.D, phi.J, xphi.data());
        return ((qpsi - qphi) % 4 + 4) % 4;
    };

    // Extract E as a ℤ₄ quadratic form in z (length-r GF(2) vars).
    std::vector<uint8_t> ebuf(r, 0);   // reused scratch z-vector
    int c0 = Efn(ebuf);                // ebuf all-zero here = z0
    std::vector<int> L(r, 0);
    for (int l = 0; l < r; ++l) {
        ebuf[l] = 1;
        L[l] = ((Efn(ebuf) - c0) % 4 + 4) % 4;
        ebuf[l] = 0;
    }
    // K bilinear coefficients in CLOSED FORM — replaces the old O(r²) E-sampling for K. The z_a z_b
    // bilinear of E = Q_psi − Q_phi is a GF(2) form (the leading 2 reduces the quadratic to mod 2):
    //   K_ab = parity(ker_a·(J·ker_b))            [J layer, via RᵀJR]
    //        ⊕ parity(ker_a & ker_b & (D&1))      [odd-D linear layer: the XOR-parity inclusion-
    //                                               exclusion contributes 2 mod 4 per shared support]
    // summed over the psi and phi sides. Per side: r packed J-matvecs + r² parity dots, i.e.
    // O(r·k²/64 + r²·k/64) vs the sampling's O(r²·k²/64) — a ~k× win on the K extraction when the
    // overlap kernel r and dim k are large. Verified bit-identical to the sampler over 3·10^5 random
    // (D,J,p0,kernel) trials, and end-to-end via the inner_product statevector fuzz. Per-thread scratch.
    static thread_local std::vector<uint64_t> doddphi, doddpsi;
    auto pack_Dodd = [](const std::vector<int>& D, int kw, std::vector<uint64_t>& m) {
        m.assign(kw, 0);
        for (int i = 0; i < (int)D.size(); ++i) if (D[i] & 1) m[i >> 6] |= (1ULL << (i & 63));
    };
    pack_Dodd(phi.D, kphi_words, doddphi);
    pack_Dodd(psi.D, kpsi_words, doddpsi);
    auto Jmatvec = [](const SymPackedMat& J, int k, int kw, const uint64_t* g,
                      uint64_t* out) {
        for (int w = 0; w < kw; ++w) out[w] = 0;
        for (int i = 0; i < k; ++i) {
            const uint64_t* jr = J.row(i);
            int s = 0;
            for (int w = 0; w < kw; ++w) s += __builtin_popcountll(jr[w] & g[w]);
            if (s & 1) out[i >> 6] |= (1ULL << (i & 63));
        }
    };
    auto par_and = [](const uint64_t* a, const uint64_t* b, int nw) {
        int s = 0; for (int w = 0; w < nw; ++w) s += __builtin_popcountll(a[w] & b[w]); return s & 1;
    };
    auto par_and3 = [](const uint64_t* a, const uint64_t* b, const uint64_t* c, int nw) {
        int s = 0; for (int w = 0; w < nw; ++w) s += __builtin_popcountll(a[w] & b[w] & c[w]); return s & 1;
    };
    static thread_local std::vector<uint64_t> Jbphi, Jbpsi;   // J·ker_b per side, flat
    Jbphi.resize((size_t)r * kphi_words);
    Jbpsi.resize((size_t)r * kpsi_words);
    for (int b = 0; b < r; ++b) {
        Jmatvec(phi.J, kphi, kphi_words, kerphi.data() + (size_t)b * kphi_words,
                Jbphi.data() + (size_t)b * kphi_words);
        Jmatvec(psi.J, kpsi, kpsi_words, kerpsi.data() + (size_t)b * kpsi_words,
                Jbpsi.data() + (size_t)b * kpsi_words);
    }
    std::vector<std::vector<uint8_t>> K(r, std::vector<uint8_t>(r, 0));
    for (int a = 0; a < r; ++a)
        for (int bb = a + 1; bb < r; ++bb) {
            const uint64_t* kpa = kerphi.data() + (size_t)a * kphi_words;
            const uint64_t* kpb = kerphi.data() + (size_t)bb * kphi_words;
            const uint64_t* ksa = kerpsi.data() + (size_t)a * kpsi_words;
            const uint64_t* ksb = kerpsi.data() + (size_t)bb * kpsi_words;
            int k = par_and(ksa, Jbpsi.data() + (size_t)bb * kpsi_words, kpsi_words)
                  ^ par_and(kpa, Jbphi.data() + (size_t)bb * kphi_words, kphi_words)
                  ^ par_and3(ksa, ksb, doddpsi.data(), kpsi_words)
                  ^ par_and3(kpa, kpb, doddphi.data(), kphi_words);
            K[a][bb] = K[bb][a] = (uint8_t)k;
        }

    // ---- Gauss sum  G = Σ_{z∈𝔽₂^r} i^{E(z)} ----
    // Shared byte-K implementation in qeccore/gf2_gauss.hpp (formerly an inlined
    // copy here). L and K are dead after this; move them in so the by-value
    // params don't copy.
    ExactPhase acc = gauss_sum(r, c0, std::move(L), std::move(K));

    // Combine: conj(ω_φ)·ω_ψ·G · 2^{-(kφ+kψ)/2}.
    ExactPhase result = phi.omega.conj().mul(psi.omega).mul(acc);
    if (result.is_zero) return ExactPhase::zero();
    result.scale -= (kphi + kpsi);
    return result;
}

// Exact amplitude ⟨bits|ψ⟩ at one computational-basis index. |ψ⟩ = ω·2^{−k/2} Σ_y i^{Q(y)}|b⊕Ry⟩,
// so ⟨bits|ψ⟩ = ω·2^{−k/2}·Σ_{y: Ry = bits⊕b} i^{Q(y)}. Solve the GF(2) system R y = bits⊕b for a
// particular y0 + kernel; the sum over the coset is a ℤ₄ Gauss sum (same machinery as
// inner_product, but single-state). Off-support ⇒ no solution ⇒ amplitude 0.
ExactPhase AffineState::amplitude_at_bits(const std::vector<uint8_t>& bits) const {
    // Build A = R (n×k), rhs = bits ⊕ b.
    std::vector<std::vector<uint8_t>> A(n_, std::vector<uint8_t>(k_, 0));
    std::vector<uint8_t> rhs(n_, 0);
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < k_; ++j) A[i][j] = R.get(i, j);
        rhs[i] = (uint8_t)((bits[i] & 1) ^ b[i]);
    }
    std::vector<uint8_t> p0;
    std::vector<std::vector<uint8_t>> ker;
    if (!gf2_solve(A, rhs, k_, p0, ker)) return ExactPhase::zero();   // off support
    const int r = (int)ker.size();

    // Q(y) mod 4 for y = p0 ⊕ Σ z_l ker_l: extract the ℤ₄ quadratic form E(z).
    auto Qof = [&](const std::vector<uint8_t>& y) -> int {
        int q = 0;
        for (int j = 0; j < k_; ++j) if (y[j]) q += D[j];
        for (int a = 0; a < k_; ++a) if (y[a])
            for (int c = a + 1; c < k_; ++c) if (y[c]) q += 2 * J.get(a, c);
        return (q % 4 + 4) % 4;
    };
    std::vector<uint8_t> y = p0;
    int c0 = Qof(y);
    std::vector<int> L(r, 0);
    for (int l = 0; l < r; ++l) {
        for (int j = 0; j < k_; ++j) y[j] ^= ker[l][j];
        L[l] = ((Qof(y) - c0) % 4 + 4) % 4;
        for (int j = 0; j < k_; ++j) y[j] ^= ker[l][j];
    }
    std::vector<std::vector<uint8_t>> K(r, std::vector<uint8_t>(r, 0));
    for (int a = 0; a < r; ++a)
        for (int bb = a + 1; bb < r; ++bb) {
            for (int j = 0; j < k_; ++j) { y[j] ^= ker[a][j]; y[j] ^= ker[bb][j]; }
            int val = ((Qof(y) - c0 - L[a] - L[bb]) % 4 + 4) % 4;
            for (int j = 0; j < k_; ++j) { y[j] ^= ker[a][j]; y[j] ^= ker[bb][j]; }
            K[a][bb] = K[bb][a] = (uint8_t)((val / 2) & 1);
        }
    ExactPhase acc = gauss_sum(r, c0, std::move(L), std::move(K));
    ExactPhase result = omega.mul(acc);
    if (result.is_zero) return ExactPhase::zero();
    result.scale -= k_;     // ·2^{−k/2}
    return result;
}

// Measure single-qubit Pauli `pauli` (0=X,1=Y,2=Z), forced outcome. Rotate X/Y→Z by
// existing Clifford gates (αs unaffected by conjugation), Z-project, rotate back.
ExactPhase AffineState::pauli_project(int pauli, int q, int outcome) {
    if (outcome != +1 && outcome != -1)
        throw std::logic_error("pauli_project: outcome must be +1 or -1");
    const int t = (1 - outcome) / 2;   // +1→0, −1→1
    ExactPhase a;
    if (pauli == 2) {                  // Z
        a = project_z(*this, q, t);
    } else if (pauli == 0) {           // X = H Z H
        apply_h(q);
        a = project_z(*this, q, t);
        apply_h(q);
    } else if (pauli == 1) {           // Y = (H S†) Z (S H), since H S† Y S H = Z
        apply_sdg(q); apply_h(q);
        a = project_z(*this, q, t);
        apply_h(q); apply_s(q);
    } else {
        throw std::logic_error("pauli_project: pauli must be 0(X),1(Y),2(Z)");
    }
    return a;
}

int AffineState::single_pauli_expectation(int pauli, int q) const {
    if (pauli == 2) return diag_z(*this, q);              // Z: read directly, no mutation
    // X/Y: conjugate a CLONE to put P into Z form (X = H Z H, Y = S H Z H S†), so
    // *this is untouched. ⟨φ|X|φ⟩ = ⟨Hφ|Z|Hφ⟩; ⟨φ|Y|φ⟩ = ⟨HS†φ|Z|HS†φ⟩.
    std::unique_ptr<StabState> c = clone();
    AffineState& s = static_cast<AffineState&>(*c);
    if (pauli == 0) { s.apply_h(q); return diag_z(s, q); }
    if (pauli == 1) { s.apply_sdg(q); s.apply_h(q); return diag_z(s, q); }
    throw std::logic_error("single_pauli_expectation: pauli must be 0(X),1(Y),2(Z)");
}

int AffineState::z_expectation(int q) const {
    return diag_z(*this, q);
}

}  // namespace qeccore
