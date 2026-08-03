#pragma once
// =============================================================================
// Parity-direct CanonicalStabSum assembler (REPLACES assemble_from_support; NO
// GradedPoly).
//
// Builds  U|0> = (output Clifford C·C') · diag(magic) · |ψ>,  where diag(magic)|ψ>
// is given by a ParitySupport (Task 2): on support point t∈{0,1}^d (basis state
// b ⊕ R·t) the amplitude ⟨b⊕R·t|ψ⟩ is multiplied by
//     ζ16^{global_phase_mult8} · ∏_k ζ16^{coeff_k·(-1)^{column_k·t}},   ζ16=e^{iπ/8}.
//
// The whole diagonal magic is a stabilizer superposition of rank χ = 2^r, where r
// is the GF(2) rank of the *genuinely magic* (odd-coefficient) parity columns.  We
//   (1) combine equal columns (coeffs add mod 16); drop coeff≡0; fold coeff≡8 (a
//       constant −1 on every coset) into the global phase;
//   (2) split the survivors into ODD (true magic) and EVEN (Clifford) columns;
//   (3) FOLD every even column i^{(coeff/2)·(m·y)} into the support |ψ> as a REAL
//       Clifford (CX fan-in to one qubit of the Z-string m, S^{coeff/2}, CX
//       uncompute) — exact, no GradedPoly, no ancilla;
//   (4) branch over an independent GF(2) basis of the odd columns: for each sign
//       pattern σ∈{0,1}^r the ray is |ψ> restricted to the coset {t : basis·t = σ}
//       (a Z-string projection per basis column) with the constant per-coset magic
//       coefficient ζ16^{gp}·∏_{odd k} ζ16^{coeff_k·(-1)^{(col_k in basis)·σ}};
//   (5) assemble those χ rays via CanonicalStabSum::from_rays;
//   (6) apply the output Clifford (C·C') through the deferred-Clifford path.
//
// So χ is the TRUE magic rank.  Part of the parity-native bare-state pipeline driven by `build_bare_state_parity` (normal_form.cpp).  Reuses
// AffineState (its native, D/J-correct project/Clifford), CanonicalStabSum,
// Pauli, parity_support, pauli_rotation_form's CliffGate.
// =============================================================================
#include <vector>

#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/parity_support.hpp"
#include "qeccore/pauli_rotation_form.hpp"  // CliffGate

namespace qeccore {

struct ParityBareState {
    CanonicalStabSum state;
    int chi = 0;
    explicit ParityBareState(int n) : state(n) {}
};

// Build U|0> = (output_clifford) · diag(magic) · |ψ>.  output_clifford is the C·C'
// gate list applied AFTER the magic (output_clifford[0] FIRST, back() LAST).
// Builds at the TRUE magic rank χ = 2^r; there is NO chi cap (per the design decision that the
// propagation-class check is the sole rejection — see build_pauli_rotation_form).
ParityBareState assemble_from_parities(const ParitySupport& ps,
                                       const std::vector<CliffGate>& output_clifford,
                                       int n);

}  // namespace qeccore
