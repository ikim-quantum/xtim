#pragma once
#include <vector>
#include "qeccore/circuit_ir.hpp"
namespace qeccore {
// Single-qubit Clifford mapping the +1/-1 eigenstates of Pauli `b` to |0>/|1> (Z basis), and inverse:
//   Z -> {} ;  X -> {H} ;  Y -> {SDG,H} (forward) / {H,S} (inverse).
std::vector<Instr> rotate_to_z(PauliBasis b, int q);
std::vector<Instr> rotate_from_z(PauliBasis b, int q);

// Targeted, oracle-validated, SOUND: push single-qubit H's to boundaries and absorb into
// prep/measure basis tags. Any H that cannot be pushed to a boundary is left VERBATIM (the
// downstream class-exit reject then fires unchanged). Physics-preserving. No-op early-out
// when no bulk H is present.
//
// Operates on a DEFERRED circuit (terminal measurements at the end). Two passes:
//   1. push_frames: a per-wire single-qubit Clifford frame is pushed left->right through
//      the two-qubit Clifford bulk (CZ/CX), rewriting gates via exact symplectic
//      factorization (H·CZ·H=CX, etc.) so leading H's reach a boundary. Frames are
//      ABSORBED into terminal measurement bases (e.g. H;M(Z) -> M(X), with a record-bit
//      flip for a sign), and FLUSHED verbatim at any boundary they cannot pass (reset,
//      non-Clifford gate, feedback). A frame that cannot factor through a two-qubit gate
//      is flushed and the gate emitted verbatim (the H survives -> downstream reject).
//   2. apply_boundary_folds: the residual Task-2 boundary folds (R;H -> RX, H;M -> MX)
//      as a final idempotent cleanup.
// SOUND: every rewrite is an exact Clifford identity verified against the dense oracle in
// the tests; a flush is always physics-preserving.
Circuit eliminate_hadamards(const Circuit& deferred);
}  // namespace qeccore
