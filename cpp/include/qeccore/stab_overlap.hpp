#pragma once
// Fixed-θ GF(2) factoring for the AFFINE representation. `PreparedAffine::prepare(θ)` precomputes,
// from a single bit-packed RREF of [R_θ | I_n] (the invertible row-op matrix P satisfies
// P R_θ = [I_k; 0]: its top k rows are G_θ, its bottom n−k rows are H_θ):
//
//   * H_θ  ((n−k_θ)×n) — parity checks spanning the left null-space of R_θ
//                        (H_θ R_θ = 0), and c_θ = H_θ b_θ.  x ∈ supp(θ) ⟺ H_θ x = c_θ.
//   * G_θ  (k_θ×n)     — a left inverse, G_θ R_θ = I_{k_θ}, so for x ∈ supp(θ),
//                        u = G_θ(x⊕b_θ) is θ's quadratic-form parameter.
//   * θ's stored (D_θ, J_θ, ω_θ, b_θ).
//
// Consumer: stab_generators.cpp (the bare-state generator pinning). (bloat-cut B3: the per-ψ
// `overlap(ψ)` evaluator and its scratch had no caller and are deleted.)

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

    // θ-coordinate map G_θ, bit-packed by ROW (k_θ rows × mw_ words).
    std::vector<uint64_t> Gth;    // k_θ rows × mw_ words

    const uint64_t* Hrow(int i) const { return Hth.data() + (size_t)i * mw_; }
    const uint64_t* Grow(int i) const { return Gth.data() + (size_t)i * mw_; }

    static PreparedAffine prepare(const AffineState& theta);
};

}  // namespace qeccore
