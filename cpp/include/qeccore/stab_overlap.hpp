#pragma once
// Fixed-θ GF(2) factoring for the AFFINE representation. `PreparedAffine::prepare(θ)` precomputes,
// from one bit-packed RREF of R_θ (the invertible row-op matrix P satisfies P R_θ = [I_k; 0]: its
// top k rows are G_θ, its bottom n−k rows are H_θ):
//
//   * H_θ  ((n−k_θ)×n) — parity checks spanning the left null-space of R_θ
//                        (H_θ R_θ = 0), and c_θ = H_θ b_θ.  x ∈ supp(θ) ⟺ H_θ x = c_θ.
//   * G_θ  (k_θ×n)     — a left inverse, G_θ R_θ = I_{k_θ}, so for x ∈ supp(θ),
//                        u = G_θ(x⊕b_θ) is θ's quadratic-form parameter.
//
// Sparse form (k-wide, never n-wide). The RREF brings a row of R up as the pivot of column j
// ("slot j", qubit piv[j]); a row's row-op combination only ever picks up pivot rows, so every G_θ
// row is supported on the k pivot qubits and every H_θ row is e_{hq[i]} ⊕ (a combination of pivot
// qubits). Both are stored as k-bit slot vectors (bit s ↔ qubit piv[s]): G_θ is k×k bits and H_θ
// (n−k)×k bits plus hq, instead of the dense n-wide rows of an RREF over [R_θ | I_n]. Same pivot
// selection, swaps and row order as that dense RREF (kept as the test oracle in
// cpp/tests/test_prepared_affine.cpp), so the expanded G_θ, H_θ, c_θ are byte-identical.
//
// Consumer: stab_generators.cpp (the bare-state generator pinning). (bloat-cut B3: the per-ψ
// `overlap(ψ)` evaluator and its scratch had no caller and are deleted.)

#include <cstdint>
#include <cstddef>
#include <vector>

#include "qeccore/stab_affine.hpp"

namespace qeccore {

struct PreparedAffine {
    int n_ = 0;
    int kth_ = 0;                 // k_θ
    int m_ = 0;                   // n − k_θ  (number of parity checks)
    int mw_ = 0;                  // words of an n-bit vector = ceil(n/64)
    int kw_ = 0;                  // words of a k-bit slot vector = ceil(k/64)

    std::vector<int> piv;         // k: slot j ↔ qubit piv[j] (the row brought up as column j's pivot)
    std::vector<int> slot_of;     // n: slot of qubit q, or −1 (q is not a pivot qubit)
    std::vector<int> hq;          // m: H_θ row i = e_{hq[i]} ⊕ Σ_s Hs(i,s)·e_{piv[s]}
    std::vector<uint64_t> Hs;     // m rows × kw_ words (slot part of H_θ)
    std::vector<uint64_t> Gs;     // k rows × kw_ words: G_θ row j = Σ_s Gs(j,s)·e_{piv[s]}
    std::vector<uint64_t> cth;    // length mw_ (c_θ = H_θ b_θ, packed: bit i = row i)

    const uint64_t* Hrow(int i) const { return Hs.data() + (size_t)i * kw_; }
    const uint64_t* Grow(int j) const { return Gs.data() + (size_t)j * kw_; }

    static PreparedAffine prepare(const AffineState& theta);
};

}  // namespace qeccore
