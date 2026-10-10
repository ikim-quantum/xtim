#pragma once
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

// Coherent gate list replacing one ControlledPauli: control on `a` in basis `b`, feedback Pauli
// `p` on `q`. `invert` (Stim `!` on the source measurement) adds an unconditional `p` on `q`.
std::vector<Instr> emit_coherent_feedback(int a, PauliBasis b, int q, PauliBasis p, bool invert);

}  // namespace qeccore
