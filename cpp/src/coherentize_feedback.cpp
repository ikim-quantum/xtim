#include "qeccore/coherentize_feedback.hpp"
#include <cstddef>
#include "qeccore/clifford_frames.hpp"

#include <map>
#include <set>
#include <stdexcept>

namespace qeccore {
namespace {

bool is_nonclifford_gate(GateKind g) {
    return g == GateKind::T || g == GateKind::CS || g == GateKind::CCZ || g == GateKind::CH;
}

// Does a Pauli byproduct seeded on `seed_qubit` at stream position `start` reach a non-Clifford
// gate before the end? Forward lightcone: a multi-qubit Clifford gate touching the cone spreads
// it to all the gate's qubits; a non-Clifford gate touching the cone is a hit.
bool byproduct_reaches_nonclifford(const Circuit& c, int start, int seed_qubit) {
    std::set<int> cone{seed_qubit};
    for (size_t i = start + 1; i < c.stream.size(); ++i) {
        const Instr& ins = c.stream[i];
        if (ins.kind != Instr::Kind::Gate) continue;
        bool touches = false;
        for (int q : ins.targets)
            if (cone.count(q)) { touches = true; break; }
        if (!touches) continue;
        if (is_nonclifford_gate(ins.gate)) return true;
        for (int q : ins.targets) cone.insert(q);  // Clifford gate spreads the cone
    }
    return false;
}

// Is the control's post-measurement STATE observed after its source measurement at `mi`? A
// `Reset` on the wire retires it (nothing past a Reset can see the old state); until then any
// Gate / Measure / ControlledPauli on the wire, a `PAULI_EXPECTATION` term or an `OUTPUT_QUBITS`
// port naming it observes the state. Noise does not: a stochastic Pauli neither reads nor
// reports the wire (an observation that follows it is still found by the scan). Conservative
// otherwise — a Pauli gate on the wire counts as an observation — since a `true` only ever costs
// an extra ancilla, never a wrong answer.
//
// This decides how the noisy-readout fold is emitted. `M(p) c` is a READOUT error (Stim projects
// to the true outcome and flips only the reported bit), but the fold stands a physical
// anticommuting error before the measurement, which flips the projected STATE too — exact only
// while nothing looks at that state again.
bool control_state_observed_after(const Circuit& c, int mi, int qubit) {
    auto hits = [&](const std::vector<int>& qs) {
        for (int q : qs)
            if (q == qubit) return true;
        return false;
    };
    for (size_t i = (size_t)mi + 1; i < c.stream.size(); ++i) {
        const Instr& ins = c.stream[i];
        if (ins.kind == Instr::Kind::Observable) {
            for (const PauliTerm& t : ins.obs)
                if (t.qubit == qubit) return true;
            continue;
        }
        const std::vector<int>* qs = nullptr;
        if (ins.kind == Instr::Kind::Gate) qs = &ins.targets;
        else if (ins.kind == Instr::Kind::Measure || ins.kind == Instr::Kind::Reset ||
                 ins.kind == Instr::Kind::ControlledPauli)
            qs = &ins.qubits;
        if (!qs || !hits(*qs)) continue;
        return ins.kind != Instr::Kind::Reset;   // a Reset retires the wire; anything else observes it
    }
    for (const OutputPort& p : c.outputs)
        if (hits(p.qubits)) return true;
    return false;
}

}  // namespace

std::vector<bool> feedback_crosses_nonclifford(const Circuit& user) {
    std::vector<bool> out;
    for (size_t i = 0; i < user.stream.size(); ++i) {
        const Instr& ins = user.stream[i];
        if (ins.kind != Instr::Kind::ControlledPauli) continue;
        out.push_back(byproduct_reaches_nonclifford(user, (int)i, ins.qubits[0]));
    }
    return out;
}

FeedbackSource resolve_feedback_source(const Circuit& user, int cp_index) {
    FeedbackSource s;
    const Instr& cp = user.stream[cp_index];
    // Count measurements strictly before the feedback; the controlling record index.
    int meas_before = 0;
    for (int i = 0; i < cp_index; ++i)
        if (user.stream[i].kind == Instr::Kind::Measure) ++meas_before;
    const int record_index = meas_before - cp.control_rec_offset;
    if (record_index < 0) return s;  // unresolved (acausal/out-of-range)
    // Find the record_index-th Measure (0-based) in stream order.
    int seen = 0, mi = -1;
    for (int i = 0; i < (int)user.stream.size(); ++i) {
        if (user.stream[i].kind != Instr::Kind::Measure) continue;
        if (seen == record_index) { mi = i; break; }
        ++seen;
    }
    if (mi < 0 || mi >= cp_index) return s;  // unresolved or not before the feedback
    const Instr& m = user.stream[mi];
    s.resolved = true;
    s.measure_stream_index = mi;
    s.control_qubit = m.qubits[0];
    s.basis = m.basis;
    s.invert = m.invert;
    s.readout_flip_p = m.readout_flip_p;
    // Idle: nothing touches the control qubit strictly between mi and cp_index.
    s.idle = true;
    for (int i = mi + 1; i < cp_index; ++i) {
        const Instr& ins = user.stream[i];
        const std::vector<int>* qs = nullptr;
        if (ins.kind == Instr::Kind::Gate) qs = &ins.targets;
        else if (ins.kind == Instr::Kind::Measure || ins.kind == Instr::Kind::Reset ||
                 ins.kind == Instr::Kind::Noise || ins.kind == Instr::Kind::ControlledPauli)
            qs = &ins.qubits;
        if (!qs) continue;
        for (int q : *qs)
            if (q == s.control_qubit) { s.idle = false; break; }
        if (!s.idle) break;
    }
    return s;
}

namespace {
Instr g1(GateKind g, int q) { Instr i; i.kind = Instr::Kind::Gate; i.gate = g; i.targets = {q}; return i; }
Instr g2(GateKind g, int a, int b) { Instr i; i.kind = Instr::Kind::Gate; i.gate = g; i.targets = {a, b}; return i; }

// The single-qubit noise channel whose error ANTICOMMUTES with a `basis`-measurement, so it
// flips the measured eigenvalue with probability p: X-basis -> Z_ERROR, Y/Z-basis -> X_ERROR.
NoiseChannel readout_flip_channel(PauliBasis basis) {
    return basis == PauliBasis::X ? NoiseChannel::Z_ERROR : NoiseChannel::X_ERROR;
}

Instr noise1(NoiseChannel ch, int qubit, double p) {
    Instr i;
    i.kind = Instr::Kind::Noise;
    i.channel = ch;
    i.qubits = {qubit};
    i.probs = {p};
    return i;
}
}  // namespace

std::vector<Instr> emit_coherent_feedback(int a, PauliBasis b, int q, PauliBasis p, bool invert) {
    // A controlled-Pauli from a wire onto itself is degenerate (`CX q q` is not a unitary), and
    // nothing downstream (deferral, eliminate_hadamards, the frame machinery) rejects it — it
    // would leave here as a silently wrong circuit. The planner routes a self-targeting feedback
    // through the copy gadget, so reaching this is an engine bug, not an input error.
    if (a == q)
        throw std::logic_error(
            "emit_coherent_feedback: control wire equals target wire (degenerate entangler); "
            "a self-targeting feedback must drive from a copy ancilla");
    std::vector<Instr> out;
    // Control-basis rotation R on `a` mapping the b-basis to the Z-basis (b=+1 -> |0>, -1 -> |1>):
    //   Z: I ; X: H ; Y: SDG then H  (so |+i> -> |0>, |-i> -> |1>).  R^-1 is the reverse.
    // Shared frame primitives (clifford_frames): rotate_to_z = forward, rotate_from_z = inverse.
    auto rot = [&](bool forward) {
        std::vector<Instr> r = forward ? rotate_to_z(b, a) : rotate_from_z(b, a);
        out.insert(out.end(), r.begin(), r.end());
    };
    rot(true);
    // Z-control controlled-`p` from a to q.
    if (p == PauliBasis::X) out.push_back(g2(GateKind::CX, a, q));
    else if (p == PauliBasis::Z) out.push_back(g2(GateKind::CZ, a, q));
    else {  // Y target: SDG q ; CX a q ; S q  (since S X S^dag = Y)
        out.push_back(g1(GateKind::SDG, q));
        out.push_back(g2(GateKind::CX, a, q));
        out.push_back(g1(GateKind::S, q));
    }
    rot(false);
    if (invert) {  // record bit inverted -> apply p on the +1 branch instead: add an unconditional p
        GateKind pg = p == PauliBasis::X ? GateKind::X : p == PauliBasis::Z ? GateKind::Z : GateKind::Y;
        out.push_back(g1(pg, q));
    }
    return out;
}

Circuit coherentize_classically_controlled_feedback(const Circuit& user, bool coherentize_all) {
    // Early-out: a feedback-free circuit has nothing to coherentize. Return it unchanged
    // (structurally identical) without the rescan + stream rebuild. Independent of
    // `coherentize_all` — there is nothing to coherentize either way, which is what keeps every
    // feedback-free workload byte- AND cost-identical under the new flag.
    if (!circuit_has_feedback(user)) return user;

    // A feedback is coherentized ONLY if its byproduct crosses a non-Clifford gate — exactly the
    // case where the downstream sign-relabel (feedback.cpp) would be a non-Pauli byproduct and so
    // return a WRONG expectation. A non-crossing feedback (Pauli byproduct) is left as a
    // ControlledPauli and handled exactly by the relabel. `feedback_crosses_nonclifford` returns
    // one flag per ControlledPauli in stream order; map it to the CP's stream index.
    //
    // UNLESS `coherentize_all`: a consumer that runs no relabel has no exact home for a surviving
    // ControlledPauli, so every feedback is coherentized and the crossing scan is skipped
    // entirely (it is a per-feedback forward lightcone walk — O(feedbacks x stream) — and its
    // answer is not consulted in that mode).
    std::vector<bool> crosses_v;
    std::map<int, bool> crosses;        // cp stream index -> crosses a non-Clifford
    if (!coherentize_all) {
        crosses_v = feedback_crosses_nonclifford(user);
        int ord = 0;
        for (int i = 0; i < (int)user.stream.size(); ++i)
            if (user.stream[i].kind == Instr::Kind::ControlledPauli)
                crosses.emplace(i, crosses_v[ord++]);
    }
    // The single decision point: coherentize this ControlledPauli, or leave it for the relabel.
    auto wants_coherent = [&](int cp_stream_index) {
        return coherentize_all || crosses.at(cp_stream_index);
    };

    // ----- Pass 1: resolve every ControlledPauli we coherentize and plan its rewrite.
    // For each feedback we need: its resolved source (control wire, basis, invert), whether the
    // control is REUSED between its measurement and the feedback (then we cannot drive from the
    // control wire), and whether the source measurement is NOISY (then we fold the readout flip
    // into a pre-measurement anticommuting error). A reused-control source measurement gets a
    // FRESH ancilla onto which its eigenvalue is copied; two feedbacks sharing one such source
    // share the same ancilla (keyed by the measurement's stream index).
    struct CPPlan {
        int cp_index;
        FeedbackSource src;
        bool needs_copy;        // drive from a fresh copy ancilla (not the control wire)
        int copy_anc;           // ancilla holding the copied eigenvalue (-1 if driven from control)
    };
    std::vector<CPPlan> plans;
    std::map<int, int> mi_anc;          // source-measure stream index -> assigned copy ancilla
    std::set<int> noisy_meas;           // source-measure stream indices needing a noise-fold
    std::set<int> undo_meas;            // ... whose state-flip must be undone after the measurement
    int next_anc = user.n;              // fresh ancillas allocated above the current max qubit

    // A noisy-readout fold whose control state is observed again has its state-flip UNDONE right
    // after the measurement (pass 2, step 4), so the control wire no longer carries the flipped
    // value when the feedback fires: such a feedback is driven from the copy ancilla, which
    // captured the value before the undo. Memoized per source measurement.
    std::map<int, bool> undo_memo;
    auto fold_is_undone = [&](const FeedbackSource& s) {
        if (s.readout_flip_p <= 0.0) return false;
        auto it = undo_memo.find(s.measure_stream_index);
        if (it != undo_memo.end()) return it->second;
        const bool u = control_state_observed_after(user, s.measure_stream_index, s.control_qubit);
        undo_memo.emplace(s.measure_stream_index, u);
        return u;
    };

    for (int i = 0; i < (int)user.stream.size(); ++i) {
        if (user.stream[i].kind != Instr::Kind::ControlledPauli) continue;
        if (!wants_coherent(i)) continue;  // left as a ControlledPauli for the consumer's relabel
        FeedbackSource s = resolve_feedback_source(user, i);
        if (!s.resolved) {
            // Genuinely unresolved feedback (acausal / out-of-range record): malformed input, not a
            // class decision. Refuse loudly rather than silently drop or emit a wrong gate.
            throw std::runtime_error(
                "coherentize_classically_controlled_feedback: unresolved feedback control record "
                "(acausal or out-of-range rec[-k]); malformed circuit");
        }
        // Drive from a fresh copy ancilla (rather than the control wire) when the control is REUSED
        // (its wire value is overwritten before the feedback, so it cannot be read in place). An
        // IDLE control drives directly from its wire: for X/Y the in-place emit brackets the
        // feedback gate with the basis rotation (H ... H), which eliminate_hadamards absorbs since
        // nothing intervenes between the measurement and the feedback. (A REUSED non-Z control would
        // strand that rotation, hence the copy gadget there.)
        //
        // A feedback onto its OWN control wire (`M q; CX rec[-1] q`, measure-and-conditionally-
        // reset) also takes the copy: driven in place it would be `CX q q`, which is not a unitary
        // on the wire at all (both |0> and |1> go to |0>) — exactly what a classical record makes
        // implementable and a coherent entangler from the wire itself cannot. The parser refuses a
        // duplicate qubit in user text; the coherentizer must not manufacture one.
        const bool self_target = user.stream[i].qubits[0] == s.control_qubit;
        const bool needs_copy = !s.idle || fold_is_undone(s) || self_target;
        int copy_anc = -1;
        if (needs_copy) {
            auto it = mi_anc.find(s.measure_stream_index);
            if (it == mi_anc.end()) {
                copy_anc = next_anc++;
                mi_anc.emplace(s.measure_stream_index, copy_anc);
            } else {
                copy_anc = it->second;
            }
        }
        if (s.readout_flip_p > 0.0) {
            noisy_meas.insert(s.measure_stream_index);
            if (fold_is_undone(s)) undo_meas.insert(s.measure_stream_index);
        }
        plans.push_back({i, s, needs_copy, copy_anc});
    }

    // ----- Pass 1b: one flip-INSTANCE ancilla per noisy fold whose state-flip must be undone.
    // The fold stands an anticommuting error before the measurement so the recorded bit and the
    // driving value flip together — exact for the RECORD, but it also flips the control's
    // post-measurement STATE, which `M(p)` does not. When that state is observed again before a
    // `Reset` the flip must be undone right after the measurement, and a second independent
    // `X_ERROR(p)` cannot do it (it would fire independently): the flip INSTANCE is held on a
    // fresh ancilla f (|0> -> |1> with probability p) and applied to the control as a coherent
    // controlled-Pauli before AND after the measurement. Shot by shot: X^f · M · X^f flips the
    // outcome and leaves the projected state where M alone would have — Stim's `M(p)` exactly.
    // The post-measurement edge on the control costs nothing new: it is emitted only where the
    // user already touches the control after its measurement. Folds whose control is never
    // observed again keep the single-error emission, byte for byte.
    //
    // Allocated AFTER every copy ancilla so the copy numbering — and hence the emitted stream of
    // every circuit that does not take this branch — is unchanged.
    std::map<int, int> mi_flip_anc;     // source-measure stream index -> flip-instance ancilla
    for (int mi : undo_meas) mi_flip_anc.emplace(mi, next_anc++);

    // Map cp_index -> its plan for the rebuild pass.
    std::map<int, const CPPlan*> by_cp;
    for (const CPPlan& pl : plans) by_cp.emplace(pl.cp_index, &pl);

    // ----- Pass 2: rebuild the stream, rewriting each planned ControlledPauli into coherent gates
    // and leaving the others verbatim (for the consumer's downstream sign-relabel).
    Circuit out;
    out.n = next_anc;                   // grown by any fresh copy ancillas
    out.stream.reserve(user.stream.size());
    for (int i = 0; i < (int)user.stream.size(); ++i) {
        const Instr& ins = user.stream[i];

        if (ins.kind == Instr::Kind::Measure) {
            const bool noisy = noisy_meas.count(i) != 0;
            auto anc_it = mi_anc.find(i);
            const bool has_copy = anc_it != mi_anc.end();
            int ctrl = ins.qubits[0];

            // 1. Noisy-readout fold: an anticommuting error BEFORE the measurement (and before the
            //    copy) so the flip is shared by BOTH the recorded bit and the copied/driving value.
            //    When the control's state is observed after the measurement (pass 1b) the error is
            //    instead HELD on the flip ancilla f (|0> -> |1> with probability p) and applied to
            //    the control as a coherent controlled-Pauli, so the SAME instance is applied again
            //    in step 4. Anticommuting with a `basis`-measurement: X-basis -> Z on the control
            //    (CZ f ctrl), Y/Z-basis -> X on the control (CX f ctrl).
            auto flip_it = mi_flip_anc.find(i);
            const bool undo = flip_it != mi_flip_anc.end();
            const int flip_anc = undo ? flip_it->second : -1;
            auto emit_held_flip = [&]() {
                out.stream.push_back(g2(ins.basis == PauliBasis::X ? GateKind::CZ : GateKind::CX,
                                        flip_anc, ctrl));
            };
            if (noisy && undo) {
                out.stream.push_back(noise1(NoiseChannel::X_ERROR, flip_anc, ins.readout_flip_p));
                emit_held_flip();
            } else if (noisy) {
                out.stream.push_back(noise1(readout_flip_channel(ins.basis), ctrl,
                                            ins.readout_flip_p));
            }

            // 2. Copy the eigenvalue onto a fresh ancilla BEFORE the measurement. Two requirements:
            //      (a) PRE-measurement (the control's last touch is then its own measurement) keeps
            //          the control's deferred-teleport clean — a POST-measurement coherent edge on
            //          the control would strand the teleport's basis-rotation H in the bulk and push
            //          the deferred circuit out of the diagonal-Clifford class.
            //      (b) the copy must NOT disturb the control's `basis`-eigenvalue (its measurement
            //          comes next, and later feedbacks may read that record).
            //    Z and X bases use a DIRECT `basis`-CNOT that keeps the control as the control in
            //    its own basis (so it is undisturbed) AND needs no rotation on the control wire:
            //      Z: anc=|0>,  CX(control, anc)   (Z-CNOT: control is Z-control, undisturbed in Z)
            //      X: anc=|+>,  CX(anc, control)   (X-CNOT control->anc: control undisturbed in X)
            //    A direct Y-CNOT(control->anc) would need an H on the control (Y-control), which
            //    strands in the deferral; instead Y uses a frame-BRACKETED Z-copy that restores the
            //    control's frame: rotate_to_z(Y); CX(control, anc); rotate_from_z(Y). The two
            //    rotations bracket the copy CX (eliminate_hadamards absorbs them) and leave the
            //    control's Y-eigenvalue undisturbed for its measurement. The copy stores the
            //    eigenvalue in anc's Z (X copies into anc's X, but anc=|+> then; we instead copy
            //    Y into anc's Z via the bracket). The feedback reads anc in Z for Y/Z and X for X.
            if (has_copy) {
                int anc = anc_it->second;
                if (ins.basis == PauliBasis::Z) {
                    out.stream.push_back(g2(GateKind::CX, ctrl, anc));   // anc=|0>, copies ctrl's Z
                } else if (ins.basis == PauliBasis::X) {
                    out.stream.push_back(g1(GateKind::H, anc));          // anc |0> -> |+>
                    out.stream.push_back(g2(GateKind::CX, anc, ctrl));   // X-CNOT ctrl->anc; ctrl X undisturbed
                } else {  // Y: frame-bracketed Z-copy, control restored (Y-eigenvalue undisturbed)
                    std::vector<Instr> rz = rotate_to_z(ins.basis, ctrl);
                    for (Instr& r : rz) out.stream.push_back(std::move(r));
                    out.stream.push_back(g2(GateKind::CX, ctrl, anc));
                    std::vector<Instr> rzi = rotate_from_z(ins.basis, ctrl);
                    for (Instr& r : rzi) out.stream.push_back(std::move(r));
                }
            }

            // 3. The measurement (now readout-noiseless after any fold). Nothing we emit follows it
            //    on the control wire except the step-4 undo, which exists only where the user's own
            //    circuit already touches the control after it — so deferral teleports the
            //    measurement as cleanly as the user's circuit allows.
            if (noisy) { Instr m = ins; m.readout_flip_p = 0.0; out.stream.push_back(std::move(m)); }
            else out.stream.push_back(ins);

            // 4. Undo the state half of the held fold (the record keeps the flip): same ancilla,
            //    same shot, so the two applications cancel exactly wherever the flip fired.
            if (undo) emit_held_flip();
            continue;
        }

        if (ins.kind != Instr::Kind::ControlledPauli) { out.stream.push_back(ins); continue; }

        // A ControlledPauli we do not coherentize (default mode: a Pauli byproduct that crosses no
        // non-Clifford gate) is emitted VERBATIM for the consumer's exact downstream sign-relabel.
        // In default mode this also keeps A6-class magic-control feedback (an X-read of a T-loaded
        // wire) off the coherent path: coherentizing it would strand an H on the magic wire and
        // leave the diagonal-Clifford+Pauli class, whereas the relabel consumes the X-measurement
        // classically. A `coherentize_all` consumer has no relabel and coherentizes that class
        // too; its gauge/anti detectors on such circuits classify identically either way.
        if (!wants_coherent(i)) { out.stream.push_back(ins); continue; }

        // Coherentize. Drive from the copy ancilla when we copied (reused control, or a held noisy
        // fold), else directly from the control wire in its measured basis. The copy stores the
        // eigenvalue in anc's X for an X-basis control (anc=|+>, X-CNOT) and in anc's Z for a Y/Z
        // control (Z-copy / frame-bracketed Z-copy), so the feedback reads the ancilla in that basis.
        const CPPlan* pl = by_cp.at(i);
        const int target = ins.qubits[0];
        const PauliBasis anc_read =
            pl->src.basis == PauliBasis::X ? PauliBasis::X : PauliBasis::Z;
        std::vector<Instr> gates =
            pl->needs_copy
                ? emit_coherent_feedback(pl->copy_anc, anc_read, target, ins.basis, pl->src.invert)
                : emit_coherent_feedback(pl->src.control_qubit, pl->src.basis, target, ins.basis,
                                         pl->src.invert);
        for (Instr& g : gates) out.stream.push_back(std::move(g));
    }
    return out;
}

}  // namespace qeccore
