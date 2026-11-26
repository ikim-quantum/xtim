#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "qeccore/circuit_ir.hpp"

namespace qeccore {

struct ParseError { int line; std::string message; };

// Task 6: one entry in the IF-guard stack accumulated at detector-declaration time.
// Records the condition that must hold for the detector to be active.
// cond_port: index in circuit.inputs (always a Bits port).
// cond_bit:  bit within that port.
// cond_value: true if the condition is `IF name[bit]`, false for `IF !name[bit]`.
struct IfGuard { int cond_port; int cond_bit; bool cond_value; };

struct ParsedStim {
    Circuit circuit;                              // flat (unrolled, desugared)
    std::vector<std::vector<int>> detectors;      // absolute measurement indices per DETECTOR
    std::map<int, std::vector<int>> observables;  // observable index -> absolute meas indices
    // DECISION(k) rec[-j] ... : declared Born output bits (nondeterministic record parities).
    // Each entry is (decision_index_k, absolute_meas_indices), one per DECISION line.
    // Mirrors observables' record handling (absolute index resolution, branch-remap treatment).
    std::vector<std::pair<int, std::vector<int>>> decisions;
    std::vector<int> expectation_labels;          // PAULI_EXPECTATION labels, declaration order
    // M-B coordinate preservation (ADDITIVE — annotations, not operations; never touch the
    // sampling stream). SHIFT_COORDS is resolved at parse time into these absolute values, so
    // they reproduce Stim's get_final_qubit_coordinates() / get_detector_coordinates() exactly.
    std::map<int, std::vector<double>> qubit_coords;   // qubit -> FINAL coords (last QUBIT_COORDS
                                                       // wins, shift-resolved); == Stim's
                                                       // get_final_qubit_coordinates()
    std::vector<std::vector<double>> detector_coords;  // per-detector coords (parallel to
                                                       // `detectors`, declaration order; empty
                                                       // vector for a coordinate-free DETECTOR)
    // Task 6: per-detector IF guard stack (parallel to `detectors`).
    // Element i is the list of IF conditions that must ALL be true (outer-to-inner order) for
    // detector i to be active.  Empty = detector is outside all IF blocks = always active.
    // Used by emit_resolved_text to skip detectors from dropped branches and by the Python
    // partition layer to build the superset live-mask.
    std::vector<std::vector<IfGuard>> detector_if_guards;
    // 1 + the largest qubit index the TEXT references: every qubit at or above it is a desugar
    // ancilla (MPP/SPP/MPAD/MZZ... gadgets). Messages only (user-level qubit naming).
    int num_user_qubits = 0;
    // Source provenance, filled only by parse_stim_circuit(text, /*track_source=*/true) (empty otherwise; metadata,
    // read only by the xdem writer's canonical-form check and t_c). Parallel to circuit.stream: the 1-based source
    // line of the user instruction that emitted each stream instruction, and that instruction's dispatch ordinal
    // (REPEAT iterations count separately, so two stream instructions come from the same user instruction iff equal).
    std::vector<int> stream_line;
    std::vector<int> stream_step;
    std::vector<ParseError> errors;               // non-empty => parse failed
    bool ok() const { return errors.empty(); }
};

// True iff the ParsedStim contains at least one DECISION declaration.
// Gates the Born-measurement path so decision-free circuits are unaffected.
inline bool circuit_has_decisions(const ParsedStim& ps) {
    return !ps.decisions.empty();
}

// Parse a strict-Stim-superset circuit (the parser's internal design note,
// 2026-06-09, §1). Errors are COLLECTED (all lines diagnosed),
// 1-based line numbers; any error fails the parse (ok() == false; circuit unusable).
ParsedStim parse_stim_circuit(const std::string& text);
// As above, also filling ParsedStim::stream_line / stream_step when track_source (the stream is identical).
ParsedStim parse_stim_circuit(const std::string& text, bool track_source);

}  // namespace qeccore
