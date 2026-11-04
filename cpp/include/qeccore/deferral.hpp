#pragma once
#include <utility>
#include <vector>
#include "qeccore/circuit_ir.hpp"

namespace qeccore {

// Wire placement for SUPPLIED reference states (design log §4.10): final_wire[w] = the
// physical deferred-space qubit holding user wire w at end-of-circuit; terminal_reads =
// the (qubit, basis) list in record order (== the trailing Measure instructions).
// Where each DEFERRED qubit comes from (metadata for messages; read by no sampling path).
// `wire` is the input-circuit wire it carries; `kind` says how the deferral created it:
//   Original    — the wire's first incarnation: deferred qubit d == input wire d (d < in.n), or
//                 the fresh qubit of a Reset that hits the wire before anything else touched it
//                 (the usual `R q` / `RX q` prologue).
//   MeasureCopy — the fresh ancilla that copies `wire`'s eigenvalue at its live-wire measurement
//                 (measurement record `record`), read terminally in place of the wire.
//   Retired     — the fresh dead wire that replaces `wire` after its abandoned-wire measurement
//                 (record `record`); later noise on the wire lands here.
//   Reset       — the fresh |0> that replaces `wire` at a Reset.
struct WireOrigin {
    enum class Kind { Original, MeasureCopy, Retired, Reset } kind = Kind::Original;
    int wire = -1;
    int record = -1;        // MeasureCopy / Retired: the measurement's record index; else -1
};

struct DeferralMap {
    std::vector<int> final_wire;
    std::vector<std::pair<int, PauliBasis>> terminal_reads;
    std::vector<WireOrigin> wire_origin;   // indexed by deferred qubit (size == deferred n)
};

// Normalize a circuit with mid-circuit Measure/Reset into canonical form: all unitary gates
// first, all measurements trailing, on an expanded qubit set. Order-preserving (deferred
// measurement #i corresponds to original measurement #i). A Measure on a wire still used
// afterward (Gate/Measure/Observable use before its next Reset or stream end) becomes a
// CNOT-copy onto a fresh ancilla, end-measured in Z; a Measure on an abandoned wire becomes
// a basis-tagged terminal measurement of the wire itself, which is immediately retired (so
// later Noise on that wire is redirected to a dead fresh wire); each Reset gives its wire a
// fresh |0>. v1: no classical feedforward.
Circuit defer_measurements(const Circuit& in);

// As above, additionally filling `map` (when non-null) with the end-of-circuit wire
// placement and terminal-read list — exactly the function's own w2q / end_meas.
Circuit defer_measurements(const Circuit& in, DeferralMap* map);

// As above, with `diagonal_copy` selecting the deferral of a LIVE-wire X/Y measurement:
//   false (the default above, every sampler/reference path): rotate q to Z, CNOT(q->a), rotate
//         back — a basis change ON q, end-measured on a in Z.
//   true  (DEM export's fallback only): a fresh ancilla a is put in |+> (H a), a controlled-X/Y
//         (a -> q) copies q's X/Y eigenvalue onto X_a, end-measured on a in X. The same
//         measurement channel (an exact dilation of it), but q never sees a Hadamard, so a
//         Pauli-times-diagonal-Clifford error residual on q stays in that class through the
//         copy. Z measurements and abandoned-wire measurements are unchanged.
Circuit defer_measurements(const Circuit& in, DeferralMap* map, bool diagonal_copy);

}  // namespace qeccore
