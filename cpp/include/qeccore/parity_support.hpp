#pragma once
// =============================================================================
// GF(2) parity restriction of a diagonal magic layer onto a stabilizer support.
//
// This REPLACES the old substitute_onto_support / GradedPoly machinery with a
// purely GF(2)-linear "restriction" — no polynomial substitution, no monomials.
//
// Input:
//   * cprime_dagger : the gate list C'† (cprime_dagger[0] applied FIRST, back()
//                     LAST), i.e. the diagonalizing Clifford's INVERSE. Applying
//                     it to |0…0⟩ builds  |ψ⟩ = C'†|0⟩.
//   * tz            : the diagonal magic as pure-Z rotations exp(i(π/8)coeff·D)
//                     (DiagResult::ZTerm, x-bits all zero).
//
// Math (all GF(2)). Build |ψ⟩ = C'†|0⟩ as an AffineState with support
//     y = b ⊕ R·t,   t ∈ {0,1}^d,   b ∈ 𝔽₂ⁿ,  R an n×d GF(2) matrix.
// A Z-string D with qubit-mask m acts on the support point y as the diagonal
// phase
//     (-1)^{m·y} = (-1)^{(m·b) ⊕ (m·R)·t}.
// So per Z-term:
//   * parity column  = m·R  (a d-bit row over the free vars t),
//   * constant bit   = m·b.
//   If m·R == 0 (D is CONSTANT on the support): it contributes only the global
//   phase exp(i(π/8)·coeff·(-1)^{m·b}) — fold coeff·(±1) into global_phase_mult8
//   (mod 16). NO magic column.
//   Else it is genuine magic: append column (m·R) and coeff.
//
// rot(c,D) eigenvalue on a point y is ζ16^{c·(-1)^{m·y}} where ζ16 = e^{iπ/8};
// the whole diagonal magic operator on |ψ⟩ is therefore
//     diag(T_z)|ψ⟩(y) = ζ16^{global_phase_mult8}
//                         · ∏_{magic k} ζ16^{coeff_k·(-1)^{column_k·t}} · ⟨y|ψ⟩.
//
// Part of the parity-native bare-state pipeline driven by `build_bare_state_parity` (normal_form.cpp).
// Reuses AffineState, Pauli, CliffGate, and DiagResult.
// =============================================================================
#include <cstdint>
#include <vector>

#include "qeccore/pauli_diagonalize.hpp"   // DiagResult::ZTerm
#include "qeccore/pauli_rotation_form.hpp" // CliffGate
#include "qeccore/stab_affine.hpp"

namespace qeccore {

struct ParitySupport {
    AffineState support;  // |ψ⟩ = C'†|0⟩, support b ⊕ R·t over d free vars
    // one parity column per MAGIC Z-string, over the d free vars, bit-packed
    // (column[w] holds free-var bits 64*w .. 64*w+63; bit j of free var j).
    std::vector<std::vector<uint64_t>> columns;
    std::vector<int> coeffs;          // coeff per magic column (parallel to columns)
    int global_phase_mult8 = 0;       // constant phase from non-magic Z-strings, units of π/8 (mod 16)

    explicit ParitySupport(int n) : support(n) {}
};

// Build |ψ⟩ = C'†|0⟩, then split every Z-term into a magic parity column (m·R)
// or a constant global-phase contribution (m·R == 0). n = qubit count.
ParitySupport restrict_to_support(const std::vector<CliffGate>& cprime_dagger,
                                  const std::vector<DiagResult::ZTerm>& tz, int n);

}  // namespace qeccore
