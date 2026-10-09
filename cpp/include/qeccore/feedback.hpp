#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "qeccore/circuit_ir.hpp"

namespace qeccore {

// ---------------------------------------------------------------------------------------------
// Classically-controlled Pauli feedback (Stim's `CX/CY/CZ rec[-k] q`), handled EXACTLY as a
// post-sampling triangular record/expectation relabel — never on the state-evolution hot loop.
//
// Why exact: a Pauli applied before a measurement only flips that measurement's outcome (iff it
// anticommutes) / an expectation's sign — never a Born probability. So we sample the records
// feedback-FREE (the existing pipeline, full speed) and XOR in the implied flips. Pure Pauli
// algebra, state-independent: exact even with magic in the circuit. This is the standard frame
// argument and exactly what Stim does for `CZ rec[-1] 0`.
//
// VALIDITY LIMIT: the frame argument holds only while the byproduct stays a PAULI. A controlled-Pauli
// that crosses a NON-Clifford gate (e.g. a transversal T between the control and the end) propagates
// to a non-Pauli Clifford — an odd S-power / CZ arm — which ROTATES an expectation (<X̄>↔<Ȳ>) rather
// than flipping its sign. The sign-flip model is then invalid, so build_feedback_plan REFUSES loudly
// (ok=false) rather than silently approximate. (Same honest-gate stance as the DEM exporter's
// ±1-expressibility gate.)
//
// Each controlled-Pauli is propagated (as if an error atom seeded at its position) through the
// SUBSEQUENT circuit to the end, giving its end-of-circuit terminal-record-flip mask `mflips`
// and per-expectation anticommute mask `esign`. Its `rec[-k]` control resolves to an ABSOLUTE
// measurement-record index. Causality ⇒ the control record precedes every flipped record ⇒ the
// relabel is TRIANGULAR (one forward pass, controlled-Paulis in record/stream order).
// ---------------------------------------------------------------------------------------------

// One resolved controlled-Pauli, ready for the per-shot relabel.
struct FeedbackOp {
    int control_record = -1;             // absolute packed-record index of the rec[-k] control
    std::vector<uint64_t> mflips;        // MW words: terminal records this op flips when fired
    uint64_t esign = 0;                  // bit r: this op anticommutes with observable r's Pauli
};

// The precomputed feedback relabel plan. `ops` is in circuit (== record) order — the order the
// triangular forward pass consumes them. `has_feedback` is false for every feedback-free
// circuit (the relabel pass is then dead code: byte-identical streams, same speed).
struct FeedbackPlan {
    bool has_feedback = false;
    bool ok = true;                      // false iff the feedback could not be resolved
    int  reject_gate_index = -1;         // set when !ok (a controlled-Pauli left the prop class)
    std::string error;                   // human-readable reason when !ok
    std::vector<FeedbackOp> ops;
    int MW = 0;                          // flip-mask words (= ceil(num_records / 64))
};

// The bare circuit = the user circuit with every ControlledPauli REMOVED. Fed to the existing
// pipeline unchanged (deferral / propagation / bare state / cascade — all feedback-free).
Circuit strip_feedback(const Circuit& user);

// Build the feedback relabel plan for `user`. Internally: a PROBE circuit (each controlled-Pauli
// replaced by a deterministic single-qubit Pauli noise location) is deferred + propagated with
// the existing machinery; each probe location's end-of-circuit atom yields `mflips`/`esign`. The
// probe and bare deferrals share an identical wire layout + terminal-read order (a ControlledPauli
// is a "use"-free instruction in both), so the resolved absolute record indices line up 1:1 with
// the bare pipeline's packed records. Returns has_feedback=false (and ok=true) when `user` has no
// feedback. Rejects (ok=false) if a controlled-Pauli's byproduct leaves the Pauli class (crosses a
// non-Clifford gate: odd S-power or CZ arm) or if the feedback is acausal.
FeedbackPlan build_feedback_plan(const Circuit& user, int num_records, int num_observables);

}  // namespace qeccore
