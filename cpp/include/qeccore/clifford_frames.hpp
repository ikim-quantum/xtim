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
//      flip for a sign), COMMUTED THROUGH every Pauli noise channel (the channel is
//      conjugated exactly — DEPOLARIZE1/2 invariant, X/Y/Z_ERROR and PAULI_CHANNEL_1/2
//      relabelled per qubit, E/ELSE chains relabelled per term and factor and kept one
//      instruction; provenance kept in Instr::frame_prov), and
//      FLUSHED verbatim at any boundary they cannot pass (reset, non-Clifford gate,
//      feedback, an unknown noise kind). A frame that cannot factor through a two-qubit gate
//      is flushed and the gate emitted verbatim (the H survives -> downstream reject).
//   2. apply_boundary_folds: the residual Task-2 boundary folds (R;H -> RX, H;M -> MX)
//      as a final idempotent cleanup.
// SOUND: every rewrite is an exact Clifford identity verified against the dense oracle in
// the tests; a flush is always physics-preserving.
// `zero_noise_transparent`: a Noise instruction whose probabilities are ALL zero is the
// identity channel, so a pending frame passes through it with the instruction emitted
// VERBATIM (not relabelled). Law-identical to the general commutation; the twirl/DEM presets
// opt in so their p=0 streams stay byte-identical. The forced-fault compiler keeps the default
// (relabelled + provenance), since it addresses a p=0 site by its user Pauli letter.
Circuit eliminate_hadamards(const Circuit& deferred, bool zero_noise_transparent = false);
}  // namespace qeccore
