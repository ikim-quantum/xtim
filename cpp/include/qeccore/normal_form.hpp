#pragma once
#include <string>
#include <vector>
#include "qeccore/circuit_ir.hpp"
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/framed_superposition.hpp"

namespace qeccore {

// The parity-native (C-4) bare-state build result.
struct BareState {
    CanonicalStabSum state;         // valid iff !rejected
    bool rejected = true;           // default-constructed = invalid; build_bare_state clears it
    int  reject_gate_index = -1;    // >=0: out-of-class gate at this deferred index; -1: non-gate
                                    //      reject (non-commuting magic) — see reject_reason
    std::string reject_reason;      // human-readable cause (empty when !rejected)
    int  chi = 0;
    BareState() : state(0) {}
};

// PARITY-NATIVE (C-4) bare-state build — the GradedPoly-free pipeline (and the SOLE build path):
//   build_pauli_rotation_form (U = C·T, single commuting magic layer; rejects non-commuting magic)
//   -> diagonalize_commuting_layer (C', Z-strings)
//   -> restrict_to_support (GF(2) parity columns over d free vars + constant phase)
//   -> FastTODD residue-project / reduce / map-back on the magic columns (T-count minimization)
//   -> assemble_from_parities (CanonicalStabSum at χ = 2^rank, output Clifford C·C').
// Accepts the H-on-magic class (a body-Hadamard on a magic wire).  Returns a rejected BareState
// (reject_gate_index = the rotation-form reject index) on non-commuting magic.
BareState build_bare_state_parity(const Circuit& deferred);

// Public bare-state entry point.  A thin wrapper over build_bare_state_parity (the legacy GradedPoly
// path + its QEC_LEGACY_BARE_STATE opt-out were deleted in C4-Int 6).
BareState build_bare_state(const Circuit& circ);

// Lean bare-state result: mirrors BareState but holds a FramedSuperposition (the affine anchor dropped).
struct FramedBareState {
    FramedSuperposition state{0};             // valid iff !rejected
    bool rejected = true;
    int  reject_gate_index = -1;
    std::string reject_reason;
    int  chi = 0;
};

// Lean bare-state sourcing: call build_bare_state, and on success project the CanonicalStabSum to a
// FramedSuperposition via FramedSuperposition::from_css (the conversion point). BYTE-IDENTICAL to
// FramedSuperposition::from_css(build_bare_state(circ).state), with rejected/reject_*/chi propagated. Does
// NOT modify build_bare_state itself (dem_export / ref_compile still consume its CSS return).
FramedBareState build_bare_state_framed(const Circuit& circ);

// TEST-ONLY instrumentation: incremented by fast_todd_reduce_columns each time the recovered
// even (Clifford) FastTODD residue g is NON-TRIVIAL (i.e. at least one even parity column was
// emitted).  Lets the parity-build tests assert that the cubic-magic stress cases actually
// exercised the phase-faithful recovery path.  Not load-bearing for any production behaviour.
extern long g_even_residue_recovered_count;

// TEST-ONLY instrumentation: incremented by fast_todd_reduce_columns each time the recovered
// even residue g carries a DEG-2 (CZ) monomial — i.e. the deg-2 even-residue reconstruction
// branch (4·t_j·t_k split into three even parity columns) fired.  Lets the parity-build tests
// assert that the deg-2 CZ-residue reconstruction is actually exercised at χ≥2.
extern long g_even_residue_deg2_count;

}  // namespace qeccore
