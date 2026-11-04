#pragma once
#include <string>
#include <vector>
#include "qeccore/circuit_ir.hpp"

namespace qeccore {

// One bool per ControlledPauli in `user.stream` (stream order). True iff the feedback's Pauli
// byproduct can reach a non-Clifford gate (T/CS/CCZ/CH) in its forward lightcone — i.e. the
// post-sampling frame trick is INVALID for it and it must be coherentized. Conservative
// (over-approximate) by design: a `true` only ever costs an unnecessary coherentization, never
// a wrong answer.
std::vector<bool> feedback_crosses_nonclifford(const Circuit& user);

// Rewrite each ControlledPauli whose byproduct crosses a non-Clifford gate into a coherent
// Clifford entangler (a controlled-Pauli with the control in the source measurement's basis),
// leaving every other ControlledPauli untouched for the existing frame trick. A non-idle
// control wire is driven from a fresh copy ancilla; a probabilistic readout flip is folded
// into an anticommuting pre-measurement error. That fold also flips the control's projected
// STATE, which Stim's `M(p)` does not, so when the control's state is observed again before a
// `Reset` the flip instance is held on a fresh ancilla and applied twice (before and after the
// measurement) to cancel exactly — the record keeps the flip, the state does not, and the
// feedback is driven from a copy ancilla that captured the flipped value. A record that
// cannot be resolved at all (acausal / out-of-range `rec[-k]`) is malformed input and throws.
// A feedback-free circuit is returned unchanged.
//
// `coherentize_all` (default false) drops the "crosses a non-Clifford gate" restriction and
// coherentizes EVERY ControlledPauli — each one is rewritten or the call throws, so the result
// never contains a ControlledPauli. It exists for consumers that run no downstream sign-relabel
// (`build_feedback_plan`) and so have no exact home for a surviving ControlledPauli: the pybind
// TwirlSampler family (see `normalize_for_twirl` in cpp/bindings/xtim_py.cpp). Consumers that
// DO run the relabel (cpp/src/sampler.cpp, dem_export) leave this false and are byte-unaffected.
Circuit coherentize_classically_controlled_feedback(const Circuit& user,
                                                    bool coherentize_all = false);

// As above, with `hoist` (requires `coherentize_all`): every coherentized feedback that the
// deferred-measurement theorem allows (the block above `deferred_feedback_pauli` in the
// implementation: conjugate the feedback Pauli P back through its window W, instruction by
// instruction; moved when the conjugate stays a Pauli at EVERY step and commutes with every
// measurement / reset / expectation it passes and with the source's measured basis on the
// control — a SUFFICIENT condition, not a necessary one: e.g. `T 1; T_DAG 1` before
// `CX rec 1` is refused although W^dag X W = X) is emitted just BEFORE its source
// measurement (after its readout fold) as the b-controlled-(W^dag P W), each factor in the
// target-rotated form (emit_coherent_feedback_hoisted). Exact: the gate is diagonal in the
// control's measured basis. The control wire then ends at its own measurement (a basis-tagged
// terminal read after deferral) and never carries a Hadamard, so a fault's
// Pauli-times-diagonal residual on it stays in the diagonal class. Decided per feedback.
// `*n_hoisted` (if non-null) = number of feedbacks hoisted; with 0 the result is identical to
// the 2-argument call.
Circuit coherentize_classically_controlled_feedback(const Circuit& user, bool coherentize_all,
                                                    bool hoist, int* n_hoisted);

// Resolution of one ControlledPauli's control record to its source measurement. `resolved` is
// false if the record index is out of range. `idle` is true iff the control qubit is untouched
// (by any Gate/Reset/Measure/ControlledPauli) strictly between the source measurement and the
// feedback — the condition under which an in-place coherent rewrite is exact.
struct FeedbackSource {
    bool resolved = false;
    int  measure_stream_index = -1;
    int  control_qubit = -1;
    PauliBasis basis = PauliBasis::Z;
    bool invert = false;
    double readout_flip_p = 0.0;
    bool idle = false;
};

FeedbackSource resolve_feedback_source(const Circuit& user, int controlled_pauli_stream_index);

// Per-feedback verdict of the deferred-measurement hoist (refusal attribution; off every hot
// path). One entry per ControlledPauli in stream order.
struct FeedbackHoistInfo {
    int cp_index = -1;            // stream index of the ControlledPauli
    int ordinal = 0;              // 0-based index among the circuit's feedbacks
    int target = -1;              // its target qubit
    PauliBasis pauli = PauliBasis::Z;          // its Pauli (CX->X, CY->Y, CZ->Z)
    int control = -1;             // the source measurement's qubit
    PauliBasis source_basis = PauliBasis::Z;   // the source measurement's basis
    int source_record = -1;       // the source's absolute measurement-record index (Stim order)
    bool hoistable = false;       // the theorem allows the move
    std::string why;              // !hoistable: the blocking instruction and the rule
    // hoistable, and the moved Pauli W^dag P W has an X/Y factor while the source read is X/Y:
    // the hoisted gate is then a P'-controlled-B with P' X/Y-type (outside the retained engine's
    // Pauli x diagonal-Clifford representation when it meets a fault residual on the control).
    bool xy_from_xy_read = false;
    std::vector<int> moved_support;   // hoistable: the qubits W^dag P W acts on (control dropped)
};
// `user_qubits` (ParsedStim::num_user_qubits, -1 unknown): wires at or above it are named as
// desugar ancillas in `why`.
std::vector<FeedbackHoistInfo> feedback_hoist_report(const Circuit& user, int user_qubits = -1);

// Coherent gate list replacing one ControlledPauli: control on `a` in basis `b`, feedback Pauli
// `p` on `q`. `invert` (Stim `!` on the source measurement) adds an unconditional `p` on `q`.
std::vector<Instr> emit_coherent_feedback(int a, PauliBasis b, int q, PauliBasis p, bool invert);

// The same unitary as emit_coherent_feedback, written for an X/Y control `b` as a `p`-controlled-`b`
// with the basis rotation on the TARGET `q`: the control wire `a` sees only CX (b=X) or SDG;CX;S
// (b=Y). A Z control is already rotation-free on both wires and is emitted as emit_coherent_feedback.
std::vector<Instr> emit_coherent_feedback_hoisted(int a, PauliBasis b, int q, PauliBasis p,
                                                  bool invert);

}  // namespace qeccore
