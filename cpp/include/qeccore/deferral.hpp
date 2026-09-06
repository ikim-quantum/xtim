#pragma once
#include <utility>
#include <vector>
#include "qeccore/circuit_ir.hpp"

namespace qeccore {

// Wire placement for SUPPLIED reference states (design log §4.10): final_wire[w] = the
// physical deferred-space qubit holding user wire w at end-of-circuit; terminal_reads =
// the (qubit, basis) list in record order (== the trailing Measure instructions).
struct DeferralMap {
    std::vector<int> final_wire;
    std::vector<std::pair<int, PauliBasis>> terminal_reads;
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

}  // namespace qeccore
