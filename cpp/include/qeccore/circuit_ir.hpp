#pragma once
#include <string>
#include <vector>

namespace qeccore {

// The gate alphabet of the circuit IR. Defined HERE (the foundational IR) rather than in the
// higher-level clifford_conjugate module, which used to own it and force this header to pull in
// the whole clifford_op stack just for an enum (a layering inversion).
enum class GateKind { X, Y, Z, S, SDG, CZ, CX, H, T, CS, CCZ, CH };

// One Clifford gate (X,Y,Z,S,SDG,H,CX,CZ) in an ordered Clifford layer. THE single canonical
// CliffGate for the whole tree (formerly two incompatible definitions in normal_form.hpp and
// pauli_rotation_form.hpp — an ODR hazard for any TU including both). Layout: {kind; targets},
// the general form carrying 1- or 2-qubit gates. The g()/a()/b() accessors serve the older
// {g,a,b}-style call sites (1q: targets={a}; 2q: targets={a,b}; b()==-1 for 1q). Stays an
// AGGREGATE (no user-declared constructors) so `{GateKind::H, {a}}` / `{GateKind::CX, {a,b}}`
// brace-init still works.
struct CliffGate {
    GateKind kind;             // one of X,Y,Z,S,SDG,H,CX,CZ
    std::vector<int> targets;  // 1 qubit (single) or 2 qubits (CX/CZ: control, target)

    GateKind g() const { return kind; }
    int a() const { return targets.empty() ? -1 : targets[0]; }
    int b() const { return targets.size() > 1 ? targets[1] : -1; }
};

enum class PauliBasis { X, Y, Z };

enum class NoiseChannel {
    X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1, DEPOLARIZE2, PAULI_CHANNEL_1, PAULI_CHANNEL_2
};

struct PauliTerm { int qubit; PauliBasis p; };

// One instruction in the ordered circuit stream.
// Canonical-IR precondition (Phase 1): a Gate instruction holds exactly ONE gate, with
// `targets` sized to the gate's arity (1, 2, or 3). Stim-style grouping is expanded upstream.
struct Instr {
    // ControlledPauli is a classically-controlled Pauli feedback `CX/CY/CZ rec[-k] q`
    // (Stim convention: rec[-k] = control, q = target; CX->X, CY->Y, CZ->Z applied to q
    // iff measurement record rec[-k] is 1). It is NOT a state-evolution operation — the
    // feedback is handled as a post-sampling record/expectation relabel via build_feedback_plan
    // (see the classically-controlled-Pauli design).
    enum class Kind { Gate, Noise, Measure, Reset, Observable, ControlledPauli, IfBegin, IfEnd } kind;
    GateKind gate{GateKind::X};            // Gate
    std::vector<int> targets;              // Gate: arity-sized qubit list
    NoiseChannel channel{NoiseChannel::DEPOLARIZE1};  // Noise channel
    std::vector<int> qubits;               // Noise / Measure / Reset / ControlledPauli: affected qubits
    std::vector<double> probs;             // Noise channel parameters
    PauliBasis basis{PauliBasis::Z};       // Measure basis / ControlledPauli Pauli (X/Y/Z on the target)
    std::vector<PauliTerm> obs;            // Observable Pauli terms
    // Kind::Observable only: declared byproduct-frame records (absolute measurement indices)
    // whose parity fixes the sign of <P>. Empty = no frame declared.
    std::vector<int> obs_frame;
    // ControlledPauli only: the record offset k of the controlling rec[-k] (k>=1). Absolute
    // record index is resolved downstream (= meas_count_at_this_point - k). -1 when unused.
    int control_rec_offset{-1};
    // Measure only: a post-projection record-bit modification. The measurement still
    // projects the state to the TRUE outcome; only the recorded classical bit changes —
    // Stim's `!q` (deterministic) and `M(p)` (probabilistic readout flip) semantics.
    bool invert = false;          // record bit ^= 1  (Stim `!` target prefix)
    double readout_flip_p = 0.0;  // record bit flipped with this probability  (Stim M(p))
    // IfBegin only: run the span [IfBegin..matching IfEnd] for a shot iff
    // decision[cond_port][cond_bit] == cond_value. -1 port when unused.
    int cond_port{-1};
    int cond_bit{0};
    bool cond_value{true};
};

// A named classical input the engine receives from the previous stage's decoder.
// Qubits: an input frame applied to the barrier state at t=0 (qubits = its support).
// Bits:   `width` decision bits consumed by IfBegin conditions.
struct InputPort {
    enum class Kind { Qubits, Bits } kind;
    std::string name;
    int width = 0;                 // Bits: number of decision bits; Qubits: unused (0)
    std::vector<int> qubits;       // Qubits: support; Bits: empty
};

// A named qubit output port — the surviving (unmeasured) qubits carried forward.
// ENGINE INVARIANT (verified at parse time): every qubit in `qubits` must NOT appear
// in any Measure instruction in the circuit.
struct OutputPort {
    std::string name;
    std::vector<int> qubits;
};

struct Circuit {
    int n = 0;                             // qubit count
    std::vector<Instr> stream;             // ordered instructions
    std::vector<InputPort> inputs;         // named classical inputs from previous decoder stage
    std::vector<OutputPort> outputs;       // named qubit output ports (declared OUTPUT_QUBITS)
};

// True iff the circuit contains classically-controlled Pauli feedback (a ControlledPauli
// instruction). The sampler uses this flag to GATE the feedback path (the post-sampling
// record/expectation relabel built by build_feedback_plan). For every feedback-FREE circuit
// this returns false, so the existing hot path is untouched.
inline bool circuit_has_feedback(const Circuit& c) {
    for (const Instr& i : c.stream)
        if (i.kind == Instr::Kind::ControlledPauli) return true;
    return false;
}

// True iff any Measure carries a record-bit flip (invert or readout_flip_p>0). The packed
// sampler uses this to GATE the record-flip pass so flip-free circuits are byte-identical.
inline bool circuit_has_record_flips(const Circuit& c) {
    for (const Instr& i : c.stream)
        if (i.kind == Instr::Kind::Measure && (i.invert || i.readout_flip_p > 0.0))
            return true;
    return false;
}

// True iff the circuit carries at least one Qubits input port (an input frame to apply at t=0).
// Gates the input-frame path so frame-free circuits hit the existing hot path byte-identically.
inline bool circuit_has_input_frame(const Circuit& c) {
    for (const InputPort& p : c.inputs)
        if (p.kind == InputPort::Kind::Qubits) return true;
    return false;
}

// True iff the circuit contains at least one IfBegin instruction (conditional branch).
// Gates the per-shot branch-skip path so branch-free circuits are unaffected.
inline bool circuit_has_branches(const Circuit& c) {
    for (const Instr& i : c.stream)
        if (i.kind == Instr::Kind::IfBegin) return true;
    return false;
}

}  // namespace qeccore
