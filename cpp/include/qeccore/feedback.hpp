#pragma once
#include <cstdint>
#include <string>
#include <utility>
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
    std::vector<uint64_t> esign;         // EW words (EW = ceil(#PAULI_EXPECTATION / 64)): bit r = this op
                                         // anticommutes with observable r's Pauli
    // The op's end-of-circuit Pauli P_j on the deferred wires (the probe deferral's layout, which is the
    // stripped consumer's): its X-support (sorted) and, per declared OUTPUT_QUBITS qubit i (flattened
    // declaration order), its frame hits as (i, 0 = X-support / 1 = Z-support). DEM export (option OC) reads
    // them: the X-support decides the per-fault correction Q_j, the frame hits the frame columns.
    std::vector<int> xwires;
    std::vector<int> zwires;             // its Z-support (sorted), same frame
    std::vector<std::pair<int, int>> frame;
};

// The precomputed feedback relabel plan. `ops` is in circuit (== record) order — the order the
// triangular forward pass consumes them. `has_feedback` is false for every feedback-free
// circuit (the relabel pass is then dead code: byte-identical streams, same speed).
struct FeedbackPlan {
    bool has_feedback = false;
    bool ok = true;                      // false iff the feedback could not be resolved
    int  reject_gate_index = -1;         // set when !ok (a controlled-Pauli left the prop class)
    int  reject_op = -1;                 // when !ok: the refused controlled-Pauli's ordinal (stream order)
    char reject_kind = 0;                // when !ok: 'C' its byproduct left the Pauli class, 'A' acausal,
                                         // 'E' > 64 PAULI_EXPECTATION columns (sampler preset only)
    std::string error;                   // human-readable reason when !ok
    std::vector<FeedbackOp> ops;
    int MW = 0;                          // flip-mask words (= ceil(num_records / 64))
    int EW = 0;                          // expectation-sign words (= ceil(#PAULI_EXPECTATION / 64))
    // The probe deferral's terminal reads, record order: (basis 0 X / 1 Y / 2 Z, wire). A consumer whose
    // eliminate_hadamards folded a read wire's last single-qubit Clifford into the read basis sees that wire's
    // end-of-circuit frame rotated by it (xwires / zwires are in the probe's frame).
    std::vector<std::pair<int, int>> reads;
};

// Options of build_feedback_plan. The defaults are the sampler preset (cpp/src/sampler.cpp, detector_determinism).
struct FeedbackPlanOptions {
    // The sampler's per-shot relabel keeps a 64-column expectation-sign limit (refused past it, unchanged);
    // DEM export folds the signs into record functionals and has no limit (false).
    bool limit_64_expectations = true;
    // Defer the probe with the diagonal copy (DEM export's class fallback deferred the consumer that way:
    // the probe must share its wire layout).
    bool diagonal_copy = false;
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
FeedbackPlan build_feedback_plan(const Circuit& user, int num_records, int num_observables,
                                 const FeedbackPlanOptions& opt = FeedbackPlanOptions{});

// The feedback relabel as GF(2) record functionals (demfb T1). The reported records y' are a unitriangular
// image of the raw (feedback-stripped) records y: ops in record order apply y'[mflips_j] ^= y'[c_j]. Row j is
// reported record j as a parity of raw records; ctrl[i] is op i's control value at its own time (later ops
// never flip an earlier control: causality). A reported record set S reads the raw set XOR_{j in S} row j
// (the transpose of the sampler's forward pass); expectation r's feedback sign is the raw functional
// XOR_{i: esign_i bit r} ctrl[i]; OUTPUT_QUBITS frame column (i, kind) gains XOR_{i': (i,kind) in frame_i'}
// ctrl[i'].
struct FeedbackFold {
    int M = 0, MW = 0;
    std::vector<uint64_t> rows;                // M x MW
    std::vector<std::vector<uint64_t>> ctrl;   // per op, MW words
    // Build from `plan` (ops in record order).
    void build(const FeedbackPlan& plan, int num_records);
    // Reported record set -> raw record set (sorted, duplicates cancel). Out-of-range indices: `recs` is
    // returned unchanged (the caller's own range check then refuses on the original set).
    std::vector<int> fold(const std::vector<int>& recs) const;
    // XOR of ctrl[i] over the ops selected by `sel(i)` -> sorted raw record set.
    template <class Sel> std::vector<int> ctrl_sum(size_t n_ops, Sel sel) const {
        std::vector<uint64_t> m((size_t)MW, 0);
        for (size_t i = 0; i < n_ops; ++i)
            if (sel(i)) for (int w = 0; w < MW; ++w) m[(size_t)w] ^= ctrl[i][(size_t)w];
        return bits(m);
    }
    std::vector<int> bits(const std::vector<uint64_t>& m) const;
};

}  // namespace qeccore
