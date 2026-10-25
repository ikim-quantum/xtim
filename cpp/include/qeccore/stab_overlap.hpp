#pragma once
// Amortized fixed-θ exact overlap for the AFFINE representation.
//
// `AffineState::inner_product(θ, ψ)` solves the joint GF(2) system
// [R_θ|R_ψ]·(u;w) = b_θ⊕b_ψ each call, re-reducing θ's R block every time.
// When ONE state θ is fixed and reused for many overlaps ⟨θ|ψ⟩, that θ-side
// reduction can be done ONCE.  `PreparedAffine::prepare(θ)` precomputes:
//
//   * H_θ  ((n−k_θ)×n) — parity checks spanning the left null-space of R_θ
//                        (H_θ R_θ = 0), and c_θ = H_θ b_θ.  x ∈ supp(θ) ⟺
//                        H_θ x = c_θ.  Used as the per-ψ orthogonality prefilter.
//   * G_θ  (k_θ×n)     — a left inverse, G_θ R_θ = I_{k_θ}, so for x ∈ supp(θ),
//                        u = G_θ(x⊕b_θ) is θ's quadratic-form parameter.
//   * θ's stored (D_θ, J_θ, ω_θ, b_θ).
//
// Per ψ, `overlap(ψ)` solves only ψ's side:  (H_θ R_ψ) w = c_θ ⊕ H_θ b_ψ.  An
// INCONSISTENT system ⟺ orthogonal supports ⟺ ⟨θ|ψ⟩ = 0 (the common case,
// dispatched cheaply).  Otherwise it builds the same ℤ4 form E(z) =
// Q_ψ(w(z)) − Q_θ(u(z)) over the r free vars (u(z) = G_θ(x(z)⊕b_θ)) and feeds
// the shared `gauss_sum`.  The result is BIT-IDENTICAL to
// `AffineState::inner_product(ψ)` (same ExactPhase: is_zero/scale/z8) for ALL ψ.
//
// Both H_θ and G_θ come from a single bit-packed RREF of [R_θ | I_n]: the
// invertible row-op matrix P satisfies P R_θ = [I_{k}; 0], so its top k rows are
// G_θ and its bottom n−k rows are H_θ.

#include <cstdint>
#include <cstddef>
#include <vector>

#include "qeccore/exact_phase.hpp"
#include "qeccore/stab_affine.hpp"

namespace qeccore {

struct PreparedAffine {
    int n_ = 0;
    int kth_ = 0;                 // k_θ
    int m_ = 0;                   // n − k_θ  (number of parity checks)

    // θ-side stored data (copied from the source AffineState).
    std::vector<uint8_t> bth;     // b_θ, length n
    std::vector<int>     Dth;     // D_θ, length k_θ
    SymPackedMat         Jth;     // J_θ, k_θ×k_θ
    ExactPhase           omth = ExactPhase::one();   // ω_θ

    // Parity-check form, bit-packed by ROW (each row = n-bit vector, mw words).
    int mw_ = 0;                  // words per H_θ / b-vector row = ceil(n/64)
    std::vector<uint64_t> Hth;    // m_ rows × mw_ words   (H_θ)
    std::vector<uint64_t> cth;    // length mw_            (c_θ = H_θ b_θ, packed)

    // Column-major copy of H_θ (== H_θ^T): n rows, each m_-bit (mwc_ words) giving
    // "which check rows have a 1 in this coordinate c". Lets overlap() build
    // M = H_θ R_ψ by outer-product accumulation over R_ψ's NATIVE row-major layout
    // (no per-ψ R_ψ transpose). Precomputed once in prepare (θ fixed). (Opt A.)
    int mwc_ = 0;                 // words per H_θ^T row = ceil(m_/64)
    std::vector<uint64_t> HthT;   // n rows × mwc_ words   (H_θ^T)

    // θ-coordinate map G_θ, bit-packed by ROW (k_θ rows × mw_ words).
    std::vector<uint64_t> Gth;    // k_θ rows × mw_ words

    const uint64_t* Hrow(int i) const { return Hth.data() + (size_t)i * mw_; }
    const uint64_t* HTrow(int c) const { return HthT.data() + (size_t)c * mwc_; }
    const uint64_t* Grow(int i) const { return Gth.data() + (size_t)i * mw_; }

    // ---- Reusable per-evaluator scratch (single-threaded-per-evaluator) ----
    // One PreparedAffine (fixed θ) is reused across many ψ SEQUENTIALLY, so the
    // buffers that overlap() used to heap-allocate on every call live here and are
    // grow-then-reuse (never shrink): after the first overlap() the hot path makes
    // NO heap allocations. NOT thread-safe to share one PreparedAffine across
    // threads (the scratch is mutated); give each thread its own prepared copy.
    // `mutable` so overlap() can stay a const method while reusing the scratch.
    struct Workspace {
        // ψ-side packing (n-bit packed by mw_ words).
        std::vector<uint64_t> Rcol;      // kpsi columns × mw_ words (col-major blocks)
        std::vector<uint64_t> bpsi;      // mw_ words
        std::vector<uint64_t> bth_pk;    // mw_ words (θ's b, packed once per overlap)
        // Packed system M = H_θ R_ψ : m rows × kw_psi words (each row = kpsi bits).
        std::vector<uint64_t> Mpk;       // m_ × kw_psi
        std::vector<uint64_t> rhs;       // m_ words (1 bit/row: rhs[i>>6] bit i)
        // Column-major (H_θ^T) outer-product scratch (Opt A): H_θ b_ψ as m bits.
        std::vector<uint64_t> Hbpsi;     // mwc_ words (H_θ b_ψ packed as an m-bit vector)
        // gf2_solve scratch (packed augmented working matrix + pivot bookkeeping).
        std::vector<uint64_t> gf_M;      // m_ × W words
        std::vector<int>      gf_pivot_col;   // length m_
        std::vector<int>      gf_col_pivot;   // length N (=kpsi)
        // Packed solve outputs: particular solution + nullspace basis, kpsi bits each.
        std::vector<uint64_t> part;      // kw_psi words
        std::vector<uint64_t> ker;       // r × kw_psi words
        int ker_rows = 0;                // valid kernel-basis rows in `ker`
        // form-extraction buffers
        std::vector<uint64_t> xtmp;      // mw_ words
        std::vector<uint64_t> xd;        // mw_ words (x ⊕ bθ)
        std::vector<uint64_t> wpsi0;     // kw_psi words
        std::vector<uint64_t> u0;        // kth_words words
        std::vector<uint64_t> wpsiL;     // r × kw_psi words
        std::vector<uint64_t> duL;       // r × kth_words words
        std::vector<uint64_t> wpsi;      // kw_psi words
        std::vector<uint64_t> uscratch;  // kth_words words
        std::vector<uint8_t>  ebuf;      // r
        std::vector<int>      L;         // r
        // NOTE: the ℤ4 quadratic-form K (r×r) and the recursive gauss_sum buffers
        // are NOT scratch-backed: gauss_sum (shared with inner_product / CH) takes
        // its form BY VALUE and mutates it internally, so K would have to be copied
        // in regardless. K/gauss_sum are only reached on the r>0 (nonzero-overlap)
        // path; the LOW-k_ψ DISJOINT regime targeted here early-exits in
        // gf2_solve_packed before ever building K, so that hot path is allocation-
        // free. The r>0 allocation is O(r²)-amortized and left as-is.
    };

    static PreparedAffine prepare(const AffineState& theta);
    ExactPhase overlap(const AffineState& psi) const;

private:
    mutable Workspace ws_;
};

}  // namespace qeccore
