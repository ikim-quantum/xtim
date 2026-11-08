#include "qeccore/coherentize_feedback.hpp"
#include <cstddef>
#include "qeccore/clifford_frames.hpp"

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

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

std::vector<Instr> emit_coherent_feedback_hoisted(int a, PauliBasis b, int q, PauliBasis p,
                                                  bool invert) {
    if (a == q)
        throw std::logic_error(
            "emit_coherent_feedback_hoisted: control wire equals target wire (degenerate "
            "entangler); a self-targeting feedback is never hoisted");
    // The SAME unitary as emit_coherent_feedback(a, b, q, p, invert) — the b-controlled-p is
    // symmetric, (1+B_a)/2 + (1-B_a)/2 P_q == (1+P_q)/2 + (1-P_q)/2 B_a, i.e. a P_q-controlled-B_a
    // — written with the basis rotation on the TARGET q instead of on the control a. On the
    // control wire a only a Z-controlled X (b=X) or its S-conjugate (b=Y: SDG a; CX; S a, since
    // S X S^dag = Y) acts: no Hadamard. A Pauli-times-diagonal-Clifford fault
    // residual on `a` therefore stays diagonal through this gate (a diagonal Clifford conjugated
    // by a controlled-X on its TARGET stays diagonal: X S X = i S^dag), which the H-bracketed
    // control form does not (H S H is not diagonal).
    // A Z-read control needs no rotation on EITHER wire in the control form (CX/CZ, or
    // SDG;CX;S on the target for Y): keep it — the target form would add a target rotation.
    if (b == PauliBasis::Z) return emit_coherent_feedback(a, b, q, p, invert);
    std::vector<Instr> out;
    {
        std::vector<Instr> r = rotate_to_z(p, q);
        out.insert(out.end(), r.begin(), r.end());
    }
    if (b == PauliBasis::X) {
        out.push_back(g2(GateKind::CX, q, a));
    } else {  // Y on the control: S (Z-controlled X) S^dag
        out.push_back(g1(GateKind::SDG, a));
        out.push_back(g2(GateKind::CX, q, a));
        out.push_back(g1(GateKind::S, a));
    }
    {
        std::vector<Instr> r = rotate_from_z(p, q);
        out.insert(out.end(), r.begin(), r.end());
    }
    if (invert) {
        GateKind pg = p == PauliBasis::X ? GateKind::X : p == PauliBasis::Z ? GateKind::Z : GateKind::Y;
        out.push_back(g1(pg, q));
    }
    return out;
}

// ===== Hoisting a feedback to its source measurement: the deferred-measurement theorem =========
//
// Setting. A feedback F = "apply P_t iff record r = 1" sits at stream position j; r is the
// REPORTED outcome of the measurement M_b(c) at position i < j (basis b, control wire c, readout
// flip folded as an anticommuting error just before M, `!` as an unconditional P_t). W is the
// ordered list of instructions strictly between i and j (the window).
//
// Theorem. Start from Q = P_t, or Q = I when nothing observes t after the feedback (rule (0): its
// next touch is a reset or none and it is no output, so P_t acts on a discarded subsystem).
// Conjugate Q BACKWARD through the window, Q <- O^dag Q O for O = O_{j-1}, ..., O_{i+1},
// keeping Q as a Pauli up to sign, with these rules per instruction O:
//   (1) Clifford gate: Q <- O^dag Q O (a Pauli).
//   (2) diagonal non-Clifford gate (T, CS, CCZ): transparent if Q is Z-type (I/Z) on its qubits
//       (Q then commutes with it); otherwise O^dag Q O is not a Pauli -> NOT hoistable. CH: Q must
//       avoid both its qubits (conservative: a Z on the control would commute; see the code).
//   (3) Pauli noise channel: transparent. A Pauli channel is a mixture of Paulis E, and
//       Q E = +-E Q, so per realization moving Q past E costs only a sign.
//   (4) another classically-controlled Pauli: transparent for the same reason (per record branch it
//       is a fixed Pauli).
//   (5) single-qubit measurement of q in basis b_q (any readout flip only changes its record):
//       transparent iff Q_q in {I, B_q} (Q commutes with the projectors); otherwise NOT hoistable.
//   (6) reset of q (to |0>): Q_q = I passes; Q_q = Z is dropped (Z|0> = |0>); X/Y -> NOT hoistable.
//   (7) PAULI_EXPECTATION O_P: transparent iff Q commutes with O_P (then every conditional
//       expectation is unchanged); otherwise NOT hoistable.
//   (8) an IF block boundary: NOT hoistable (the window would not be a fixed operation).
// Finally, at the measurement: Q_c in {I, B} is dropped (B on a B-eigenstate is a phase);
// Q_c anticommuting with B -> NOT hoistable. If every step passed, then
//     M_b(c) ; W ; P_t^r   ==   (b-controlled Q from c) ; M_b(c) ; W
// exactly, as a channel on every record and every retained state. The check is per instruction,
// so it is SUFFICIENT, not necessary: a window whose non-Clifford gates cancel (`T 1; T_DAG 1`)
// is refused although W^dag P W is a Pauli; so is an anticommuting measurement whose qubit and
// record are both discarded afterwards. Such over-refusals are loud, never wrong.
//
// Proof sketch. Steps (1)-(7) give W ; P_t^r == (+-Q)^r ; W branch by branch: for a unitary
// Clifford O, O^dag P O is the Pauli moved past it; for a Pauli mixture or a commuting projector
// the move costs a sign; for a reset the dropped factor stabilizes the reset state. A sign is a
// phase on the record-r branch, i.e. a global phase within that branch: invisible. So the
// classical Q^r can sit immediately after M_b(c), on the post-measurement state where c is the
// B-eigenstate whose eigenvalue is r (the readout fold makes the state's eigenvalue the reported
// one). There "apply Q iff the B-eigenvalue of c is -1" is the b-controlled-Q, which is diagonal
// in c's measured basis and so commutes with M_b(c): it may be applied just BEFORE M_b(c),
// reading the wire coherently (the deferred-measurement principle read backwards). The c factor
// was removed in the last step, so the controlled gate never targets its own control. Since each
// hoisted gate equals its classical Pauli right after its own measurement, hoists of different
// feedbacks compose in any order (Paulis commute up to a per-branch sign) and are decided per
// feedback, from that feedback's window alone.
//
// Consequences (the review's I-1 list): MR/MRX/MRY sources (the reset comes after the read, so
// it is in no window), Pauli noise on t or c in the window, and Paulis, measurements and
// PAULI_EXPECTATIONs that commute with the moved Pauli all hoist. What does NOT is exactly a
// non-Clifford gate, an anticommuting measurement / expectation / reset, or an IF boundary on the
// moved Pauli's light cone. Operations on c after its measurement matter only through rule (1):
// they can carry Q onto c, where the last step decides.
//
// Implementation notes. Q is held sign-free as a sparse map qubit -> (x, z) bits; signs are
// immaterial by the proof. The hoisted gate is emitted factor by factor
// (emit_coherent_feedback_hoisted per qubit of Q: the b-controlled product is the product of the
// b-controlled factors), after the readout fold, before any copy-out and the measurement.
namespace {

const char* gate_name(GateKind g) {
    switch (g) {
        case GateKind::X: return "X";   case GateKind::Y: return "Y";   case GateKind::Z: return "Z";
        case GateKind::S: return "S";   case GateKind::SDG: return "S_DAG";
        case GateKind::CZ: return "CZ"; case GateKind::CX: return "CX"; case GateKind::H: return "H";
        case GateKind::T: return "T";   case GateKind::CS: return "CS"; case GateKind::CCZ: return "CCZ";
        case GateKind::CH: return "CH";
    }
    return "?";
}
const char* basis_letter(PauliBasis b) {
    return b == PauliBasis::X ? "X" : b == PauliBasis::Y ? "Y" : "Z";
}
uint8_t basis_bits(PauliBasis b) { return b == PauliBasis::X ? 1 : b == PauliBasis::Z ? 2 : 3; }
PauliBasis bits_basis(uint8_t v) {
    return v == 1 ? PauliBasis::X : v == 2 ? PauliBasis::Z : PauliBasis::Y;
}

struct HoistResult {
    bool ok = false;
    std::vector<std::pair<int, PauliBasis>> q;   // sign-free W^dag P W, control factor dropped
    std::string why;                             // !ok: the blocking instruction and the rule
};

// The backward conjugation of the theorem above, for the feedback at `cp_index` whose source
// measurement is at `mi`. Pure function of the user stream.
HoistResult deferred_feedback_pauli(const Circuit& user, int mi, int cp_index,
                                    int user_qubits = -1) {
    HoistResult r;
    const Instr& m = user.stream[mi];
    const int c = m.qubits[0];
    const Instr& cp = user.stream[cp_index];
    std::map<int, uint8_t> Q;
    // Rule (0): a Pauli on a wire that nothing observes after the feedback (its next touch is a
    // reset, or there is none, and it is no output) acts on a discarded subsystem: the feedback is
    // the identity channel and moves trivially.
    if (control_state_observed_after(user, cp_index, cp.qubits[0]))
        Q.emplace(cp.qubits[0], basis_bits(cp.basis));
    auto get = [&](int q) -> uint8_t {
        auto it = Q.find(q);
        return it == Q.end() ? 0 : it->second;
    };
    auto put = [&](int q, uint8_t v) {
        if (v) Q[q] = v;
        else Q.erase(q);
    };
    auto fail = [&](std::string why) {
        r.ok = false;
        r.why = std::move(why);
        return r;
    };
    // User-level naming: wires at/above `user_qubits` are parser desugar ancillas (-1: unknown).
    auto qs = [&](int q) {
        return (user_qubits >= 0 && q >= user_qubits)
                   ? std::string("an MPP/MPAD desugar ancilla wire")
                   : "qubit " + std::to_string(q);
    };
    for (int j = cp_index - 1; j > mi; --j) {
        const Instr& ins = user.stream[j];
        switch (ins.kind) {
            case Instr::Kind::Gate: {
                const std::vector<int>& tg = ins.targets;
                switch (ins.gate) {
                    case GateKind::X: case GateKind::Y: case GateKind::Z: break;
                    case GateKind::S: case GateKind::SDG: {          // X <-> Y
                        const uint8_t v = get(tg[0]);
                        if (v & 1) put(tg[0], v ^ 2);
                        break;
                    }
                    case GateKind::H: {                              // X <-> Z
                        const uint8_t v = get(tg[0]);
                        put(tg[0], (uint8_t)(((v & 1) << 1) | ((v >> 1) & 1)));
                        break;
                    }
                    case GateKind::CX: {                             // X_a -> X_a X_b, Z_b -> Z_a Z_b
                        uint8_t a = get(tg[0]), b = get(tg[1]);
                        b ^= (a & 1);
                        a ^= (b & 2);
                        put(tg[0], a);
                        put(tg[1], b);
                        break;
                    }
                    case GateKind::CZ: {                             // X_a -> X_a Z_b, X_b -> Z_a X_b
                        uint8_t a = get(tg[0]), b = get(tg[1]);
                        const uint8_t xa = a & 1, xb = b & 1;
                        a ^= (uint8_t)(xb << 1);
                        b ^= (uint8_t)(xa << 1);
                        put(tg[0], a);
                        put(tg[1], b);
                        break;
                    }
                    case GateKind::T: case GateKind::CS: case GateKind::CCZ:
                        for (int q : tg)
                            if (get(q) & 1)
                                return fail(std::string("a non-Clifford ") + gate_name(ins.gate) +
                                            " on " + qs(q) +
                                            " where the moved Pauli has an X/Y component (the "
                                            "instruction-by-instruction conjugation leaves the "
                                            "Pauli group there)");
                        break;
                    case GateKind::CH:
                        // Conservative on purpose: Q must avoid BOTH qubits, although a Z on the
                        // CH control would commute. Relaxing it changes no accepted output on any
                        // route (the retained path refuses every CH circuit; the PPR route never
                        // adopts a hoisted form through a CH) and could not be oracle-verified —
                        // the same standard as the unimplemented PAULI_EXPECTATION obs_frame move.
                        for (int q : tg)
                            if (get(q))
                                return fail("a non-Clifford CH on " + qs(q) +
                                            " on the moved Pauli's support (the "
                                            "instruction-by-instruction conjugation leaves the "
                                            "Pauli group there)");
                        break;
                }
                break;
            }
            case Instr::Kind::Noise:            // rule (3): a Pauli mixture
            case Instr::Kind::ControlledPauli:  // rule (4): a Pauli per record branch
                break;
            case Instr::Kind::Measure:
                for (int q : ins.qubits) {
                    const uint8_t v = get(q);
                    if (v && v != basis_bits(ins.basis))
                        return fail(std::string("a ") + basis_letter(ins.basis) +
                                    "-measurement of " + qs(q) +
                                    " that anticommutes with the moved Pauli");
                }
                break;
            case Instr::Kind::Reset:
                for (int q : ins.qubits) {
                    const uint8_t v = get(q);
                    if (v == 2) put(q, 0);   // Z fixes |0>: a phase
                    else if (v)
                        return fail("a reset of " + qs(q) +
                                    " where the moved Pauli is X/Y-type (it does not fix |0>)");
                }
                break;
            case Instr::Kind::Observable: {
                int anti = 0;
                for (const PauliTerm& term : ins.obs) {
                    const uint8_t v = get(term.qubit), w = basis_bits(term.p);
                    if (v && v != w) anti ^= 1;
                }
                if (anti)
                    return fail("a PAULI_EXPECTATION that anticommutes with the moved Pauli");
                break;
            }
            case Instr::Kind::IfBegin:
            case Instr::Kind::IfEnd:
                return fail("an IF block boundary");
        }
    }
    const uint8_t vc = get(c);
    if (vc) {
        if (vc != basis_bits(m.basis))
            return fail(std::string("the moved Pauli anticommutes with the source ") +
                        basis_letter(m.basis) + "-measurement on its own control wire (" + qs(c) + ")");
        put(c, 0);   // B on the measured B-eigenstate: a phase
    }
    r.ok = true;
    for (const auto& kv : Q) r.q.emplace_back(kv.first, bits_basis(kv.second));
    return r;
}

}  // namespace

std::vector<FeedbackHoistInfo> feedback_hoist_report(const Circuit& user, int user_qubits) {
    std::vector<FeedbackHoistInfo> out;
    int ord = 0;
    for (int i = 0; i < (int)user.stream.size(); ++i) {
        const Instr& ins = user.stream[i];
        if (ins.kind != Instr::Kind::ControlledPauli) continue;
        FeedbackHoistInfo info;
        info.cp_index = i;
        info.ordinal = ord++;
        info.target = ins.qubits[0];
        info.pauli = ins.basis;
        FeedbackSource s = resolve_feedback_source(user, i);
        if (!s.resolved) {
            info.why = "unresolved control record";
            out.push_back(std::move(info));
            continue;
        }
        info.source_basis = s.basis;
        info.control = s.control_qubit;
        info.source_record = 0;
        for (int k = 0; k < s.measure_stream_index; ++k)
            if (user.stream[k].kind == Instr::Kind::Measure) ++info.source_record;
        HoistResult r = deferred_feedback_pauli(user, s.measure_stream_index, i, user_qubits);
        info.hoistable = r.ok;
        info.why = r.why;
        if (r.ok)
            for (const auto& f : r.q) info.moved_support.push_back(f.first);
        if (r.ok && s.basis != PauliBasis::Z)
            for (const auto& f : r.q)
                if (f.second != PauliBasis::Z) info.xy_from_xy_read = true;
        out.push_back(std::move(info));
    }
    return out;
}

Circuit coherentize_classically_controlled_feedback(const Circuit& user, bool coherentize_all) {
    return coherentize_classically_controlled_feedback(user, coherentize_all, false, nullptr);
}

Circuit coherentize_classically_controlled_feedback(const Circuit& user, bool coherentize_all,
                                                    bool hoist, int* n_hoisted) {
    if (n_hoisted) *n_hoisted = 0;
    if (hoist && !coherentize_all)
        throw std::logic_error(
            "coherentize_classically_controlled_feedback: hoist requires coherentize_all (a "
            "hoisted feedback must not be reordered past a ControlledPauli left for a relabel)");
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
    // share the same ancilla (keyed by the measurement's stream index). Copy ancillas are
    // allocated after the hoist decision (Pass 1c) — a hoisted feedback reads its control AT the
    // measurement and needs none — in plan order, so with hoisting off the numbering is the
    // historical one.
    struct CPPlan {
        int cp_index;
        FeedbackSource src;
        bool needs_copy;        // drive from a fresh copy ancilla (not the control wire)
        int copy_anc;           // ancilla holding the copied eigenvalue (-1 if driven from control)
        bool hoisted = false;   // Pass 1c: emitted at its source measurement as b-controlled-`hq`
        std::vector<std::pair<int, PauliBasis>> hq;   // hoisted: W^dag P W (sign-free, control dropped)
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
        if (s.readout_flip_p > 0.0) {
            noisy_meas.insert(s.measure_stream_index);
            if (fold_is_undone(s)) undo_meas.insert(s.measure_stream_index);
        }
        plans.push_back({i, s, needs_copy, -1});
    }

    // ----- Pass 1c (hoist mode only): emit a feedback AT its source measurement, by the
    // deferred-measurement theorem (see the block above `deferred_feedback_pauli`): the classical
    // P_t^r after the window W equals the b-controlled-(W^dag P_t W) just before the measurement
    // M_b(c), whenever the backward conjugation succeeds. Decided PER FEEDBACK from its own window
    // alone — another classically-controlled Pauli in the window is transparent (a Pauli
    // mixture, like noise), and every hoisted gate is equivalent to its classical Pauli right
    // after its own measurement, so hoists compose and their mutual order is immaterial.
    if (hoist) {
        for (CPPlan& pl : plans) {
            HoistResult r = deferred_feedback_pauli(user, pl.src.measure_stream_index, pl.cp_index);
            if (!r.ok) continue;
            pl.hoisted = true;
            pl.hq = std::move(r.q);
        }
        // Composition with the NOT-moved feedbacks. A feedback n left in place (driven from its
        // control wire c2 after its measurement m2) relies on ITS OWN deferred-measurement
        // precondition: c2 is untouched between m2 and its read at cp_n, so the in-place coherent
        // read equals the record (FeedbackSource::idle, computed on the USER stream). A moved gate
        // is a new operation at its source measurement mh; if m2 < mh < cp_n and its support
        // meets c2, that precondition no longer holds in the EMITTED stream, and n must read a
        // copy of c2 taken at m2 (the existing gadget for a control disturbed between its
        // measurement and its feedback). Load-bearing: without it such a circuit is accepted and
        // wrong (tests/test_feedback_after_t.py::test_moved_gate_on_an_in_place_control_forces_
        // its_copy). Conservative: it fires even when the moved factor on c2 commutes with n's
        // read basis (costs one ancilla, never a wrong answer).
        for (CPPlan& n : plans) {
            if (n.hoisted || n.needs_copy) continue;
            const int c2 = n.src.control_qubit, m2 = n.src.measure_stream_index;
            for (const CPPlan& h : plans) {
                if (!h.hoisted) continue;
                const int mh = h.src.measure_stream_index;
                if (!(m2 < mh && mh < n.cp_index)) continue;
                for (const auto& f : h.hq)
                    if (f.first == c2) { n.needs_copy = true; break; }
                if (n.needs_copy) break;
            }
        }
        if (n_hoisted)
            for (const CPPlan& pl : plans)
                if (pl.hoisted) ++*n_hoisted;
    }
    for (CPPlan& pl : plans) {
        if (!pl.needs_copy || pl.hoisted) continue;
        auto it = mi_anc.find(pl.src.measure_stream_index);
        if (it == mi_anc.end()) {
            pl.copy_anc = next_anc++;
            mi_anc.emplace(pl.src.measure_stream_index, pl.copy_anc);
        } else {
            pl.copy_anc = it->second;
        }
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
    // observed again keep the single-error emission, byte for byte. A hoisted feedback sits
    // between the two applications (after the fold, before the measurement), so it reads the
    // REPORTED bit, and the undo restores the state for the reuse (review M-3).
    //
    // Allocated AFTER every copy ancilla so the copy numbering — and hence the emitted stream of
    // every circuit that does not take this branch — is unchanged.
    std::map<int, int> mi_flip_anc;     // source-measure stream index -> flip-instance ancilla
    for (int mi : undo_meas) mi_flip_anc.emplace(mi, next_anc++);

    // Map cp_index -> its plan for the rebuild pass.
    std::map<int, const CPPlan*> by_cp;
    for (const CPPlan& pl : plans) by_cp.emplace(pl.cp_index, &pl);

    // Source mi -> its hoisted feedback plans in stream order (plans is in stream order).
    std::map<int, std::vector<const CPPlan*>> hoist_at;
    for (const CPPlan& pl : plans)
        if (pl.hoisted) hoist_at[pl.src.measure_stream_index].push_back(&pl);

    // ----- Pass 2: rebuild the stream, rewriting each planned ControlledPauli into coherent gates
    // and leaving the others verbatim (for the consumer's downstream sign-relabel).
    Circuit out;
    out.n = next_anc;                   // grown by any fresh copy ancillas
    out.stream.reserve(user.stream.size());
    int record = -1;                    // user measurement record of the current Measure
    for (int i = 0; i < (int)user.stream.size(); ++i) {
        const Instr& ins = user.stream[i];

        if (ins.kind == Instr::Kind::Measure) {
            ++record;                   // one record per Measure, stream order (resolve_feedback_source)
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
            // Either way the fold is ONE noise instruction, tagged with its provenance (metadata
            // only): firing its Pauli IS the readout flip of `record` — record and driving value
            // flip together, and the held form's step-4 undo restores the state.
            if (noisy && undo) {
                out.stream.push_back(noise1(NoiseChannel::X_ERROR, flip_anc, ins.readout_flip_p));
                out.stream.back().origin = Instr::Origin::ReadoutFold;
                out.stream.back().origin_index = record;
                emit_held_flip();
            } else if (noisy) {
                out.stream.push_back(noise1(readout_flip_channel(ins.basis), ctrl,
                                            ins.readout_flip_p));
                out.stream.back().origin = Instr::Origin::ReadoutFold;
                out.stream.back().origin_index = record;
            }

            // 1b. Hoisted feedbacks of this source (Pass 1c): after the fold (which they must see,
            //     exactly as the in-place drive would) and before the measurement they commute with.
            {
                auto hit = hoist_at.find(i);
                if (hit != hoist_at.end())
                    for (const CPPlan* pl : hit->second)
                        for (const auto& f : pl->hq) {   // b-controlled-(W^dag P W), factor by factor
                            std::vector<Instr> gates = emit_coherent_feedback_hoisted(
                                pl->src.control_qubit, pl->src.basis, f.first, f.second,
                                pl->src.invert);
                            for (Instr& g : gates) out.stream.push_back(std::move(g));
                        }
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
        if (pl->hoisted) continue;   // emitted at its source measurement (Pass 1c)
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
