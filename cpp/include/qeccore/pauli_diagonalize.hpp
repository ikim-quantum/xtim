#pragma once
// =============================================================================
// Simultaneous Clifford diagonalization of the commuting magic layer.
//
// Given the (mutually commuting) magic terms of a PauliRotationForm —
//     T = Π_k exp(i (π/8) c_k P_k)        (the P_k pairwise commute)
// produce a Clifford C' and a set of pure-Z Paulis D_k such that
//     C'† P_k C' = D_k           for every k,
// hence
//     C' · ( Π_k exp(i (π/8) c_k D_k) ) · C'†  ==  T.
//
// This is "option (a)": ANY valid symplectic diagonalization is returned. The
// sign that Clifford conjugation may attach to a Pauli (C'† P_k C' = ± Z-string)
// is folded into the term's coeff (mod 16), so every emitted ZTerm.zstring is a
// SIGNLESS (phase ∈ {0,2}) pure-Z Pauli and the rotation angle carries the sign.
//
// Part of the parity-native bare-state pipeline driven by `build_bare_state_parity` (normal_form.cpp). Reuses qeccore::Pauli,
// CliffordTableau, and the pauli_rotation_form CliffGate type.
//
// Materialization contract (the cprime gate list):
//   cprime[0] is applied FIRST, cprime.back() LAST, so as a unitary
//       C' = cprime.back() · … · cprime[1] · cprime[0].
//   Materialize  C' · (Π_k rot(coeff_k, D_k)) · C'†  to reproduce T.
// =============================================================================
#include <vector>

#include "qeccore/pauli.hpp"
#include "qeccore/pauli_rotation_form.hpp"  // CliffGate, PauliRotationForm::Entry

namespace qeccore {

struct DiagResult {
    // C' as an ordered Clifford gate list (same CliffGate as pauli_rotation_form:
    // {GateKind kind; std::vector<int> targets}). cprime[0] applied FIRST.
    std::vector<CliffGate> cprime;

    struct ZTerm {
        Pauli zstring;  // pure Z: NO x-bits; phase ∈ {0,2} (signless), sign folded into coeff
        int coeff;      // rotation coeff (mod 16); carries the conjugation sign
    };
    std::vector<ZTerm> tz;
};

// Diagonalize a mutually-commuting magic layer (the PauliRotationForm terms) into
// pure-Z rotations. Asserts the input Paulis pairwise commute. n = qubit count.
DiagResult diagonalize_commuting_layer(
    const std::vector<PauliRotationForm::Entry>& terms, int n);

}  // namespace qeccore
