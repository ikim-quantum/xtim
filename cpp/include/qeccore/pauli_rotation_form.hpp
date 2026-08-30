#pragma once
// =============================================================================
// Pauli-rotation normal form  U = C · T   (T applied FIRST / input, C SECOND /
// output).  Part of the parity-native bare-state pipeline driven by `build_bare_state_parity` (normal_form.cpp).
//
//        U = global_phase · C · ( Π_{i=0..|T|-1} rot(coeff_i, P_i) )
//
//   where  rot(c, P) = cos(πc/8) I + i sin(πc/8) P = exp(i (π/8) c P),
//   T[0] is applied FIRST (acts on |0>) and T.back() LAST (just before C).
//
//   * T : an ORDERED list of (signed Hermitian Pauli P, int coeff) entries in
//         the |0> frame.  Each entry denotes exp(i (π/8) coeff · P).  A `Y` site
//         carries the i of the Hermitian representative (phase ∈ {0,1}); the sign
//         of a conjugated generator is folded into `coeff` (mod 16).  Entries
//         with ODD coeff are genuine magic; EVEN-coeff entries are Clifford and
//         are kept in T purely for materialization (they do NOT participate in
//         the accept/reject decision).
//   * C : an ordered list of Clifford gates (X,Y,Z,S,SDG,H,CX,CZ).  cliffords[0]
//         is applied FIRST among the Cliffords, cliffords.back() LAST.  C as a
//         whole is applied AFTER all of T.
//   * global_phase : a tracked std::complex<double> for exactness.
//
// Build algorithm (CANONICAL "fold-to-completion, then test"): process gates in
// FORWARD order.  A Clifford g is appended to C (g·(C·T) = (g... no: it acts
// last, so C := C·g).  A non-Clifford diagonal gate is expanded to π/8 Z-string
// generators; each is conjugated into the |0> frame (P = C⁻¹ · Zstring · C) and
// FOLDED into the ORDERED T-list from the output (back) end toward the input
// (front) end — combining coeffs on an equal Pauli, sliding past commuting
// entries, and STOPPING (inserting) at the first anticommuting entry.  No
// rejection happens during folding.
//
// The SINGLE accept/reject decision is made at the END: collect the surviving
// ODD-coeff entries; if they mutually commute -> ACCEPT, else -> reject.
// =============================================================================
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

// CliffGate (the single canonical definition: {kind; targets}) lives in circuit_ir.hpp,
// included above. In the C-layer it is applied-after-T; targets sized to arity.

// The U = C·T normal form of a circuit's unitary.
struct PauliRotationForm {
    int n = 0;

    // T-layer: ORDERED list. T[0] applied FIRST (on |0>), T.back() LAST.
    // The Pauli stored is the signed Hermitian representative (phase ∈ {0,1}
    // for the Hermitian i^{#Y}; the +/- sign of the rotation is folded into
    // `coeff`).  coeff is reduced mod 16 and never 0 (zero entries are erased).
    struct Entry {
        Pauli pauli;  // Hermitian representative (phase ∈ {0,1})
        int coeff;    // in (0,16)
    };
    std::vector<Entry> terms;

    // C-layer: ordered Clifford gates. cliffords[0] applied FIRST, back() LAST.
    std::vector<CliffGate> cliffords;

    std::complex<double> global_phase{1.0, 0.0};

    // rejection bookkeeping. Set only by the END-of-circuit commute test.
    bool rejected = false;
    int reject_gate_index = -1;     // >=0: out-of-class gate at this deferred index on rejection; -1 otherwise
    std::string reject_reason;
};

// Canonical key for a Pauli IGNORING phase: x words || z words, serialized.
std::string pauli_key(const Pauli& p);

// Build the U = C·T form from a circuit. Processes Gate instructions in order;
// non-Gate instructions are ignored (unitary-only form). Folds magic generators
// without rejecting; the single accept/reject test runs at the end.
PauliRotationForm build_pauli_rotation_form(const Circuit& circuit);

}  // namespace qeccore
