// Unit test for NormalizePolicy::coherentize_all / the `coherentize_all` argument of
// coherentize_classically_controlled_feedback.
//
// Contract under test:
//   1. DEFAULT (false) is byte-unchanged: a feedback whose byproduct does NOT cross a
//      non-Clifford gate stays a ControlledPauli, exactly as through 3.1.0. This is what keeps
//      cpp/src/sampler.cpp and dem_export (which relabel via build_feedback_plan) unaffected.
//   2. coherentize_all=true removes EVERY ControlledPauli, rewriting the non-crossing one into
//      coherent gates too — so a consumer that runs no relabel (the pybind TwirlSampler family)
//      has nothing left to strip and nothing left to drop silently.
//   3. A feedback-FREE circuit is returned structurally unchanged under BOTH settings (the
//      early-out), which is the mechanism behind the "no cost on feedback-free workloads" claim.
//   4. Crossing feedback is coherentized IDENTICALLY under both settings — the byte-identity
//      argument for the shipped magic examples (cultivation_d5, distillation_15_1_3).
//   5. Through NormalizePolicy: coherentize_all + Feedback::Reject leaves no residue to reject,
//      while Strip on the same circuit (the 3.1.0 binding policy) silently deletes it.
//   6. The noisy-readout fold (`M(p) c` driving a feedback) is emitted as a single pre-measurement
//      error iff the control's post-measurement state is never observed before a Reset; otherwise
//      the flip INSTANCE is held on a fresh ancilla and applied before AND after the measurement
//      (record flipped, state restored — Stim's M(p)), and the feedback drives from a copy ancilla.
//      Noise on the wire is not an observation; the held form is taken in default mode too.
//   7. A feedback onto its OWN control wire (`M q; CX rec[-1] q`) takes the copy gadget even
//      with an idle control: the in-place emit would be the degenerate `CX q q` (not a unitary on
//      the wire), which the coherentizer must never emit — emit_coherent_feedback throws on it.

#include "check.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

Circuit parse(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    return ps.circuit;
}

int count_feedback(const Circuit& c) {
    int n = 0;
    for (const Instr& ins : c.stream)
        if (ins.kind == Instr::Kind::ControlledPauli) ++n;
    return n;
}

// Full structural equality of two circuits: every Instr field the coherentizer can touch
// (kind/gate/targets/qubits/basis/invert AND channel/probs/readout_flip_p/control_rec_offset/
// observable terms), plus the qubit count.
bool same_stream(const Circuit& a, const Circuit& b) {
    if (a.n != b.n) return false;
    if (a.stream.size() != b.stream.size()) return false;
    for (size_t i = 0; i < a.stream.size(); ++i) {
        const Instr& x = a.stream[i];
        const Instr& y = b.stream[i];
        if (x.kind != y.kind) return false;
        if (x.kind == Instr::Kind::Gate && x.gate != y.gate) return false;
        if (x.kind == Instr::Kind::Noise && x.channel != y.channel) return false;
        if (x.targets != y.targets) return false;
        if (x.qubits != y.qubits) return false;
        if (x.probs != y.probs) return false;
        if (x.basis != y.basis) return false;
        if (x.invert != y.invert) return false;
        if (x.readout_flip_p != y.readout_flip_p) return false;
        if (x.control_rec_offset != y.control_rec_offset) return false;
        if (x.obs.size() != y.obs.size()) return false;
        for (size_t t = 0; t < x.obs.size(); ++t)
            if (x.obs[t].qubit != y.obs[t].qubit || x.obs[t].p != y.obs[t].p) return false;
        if (x.obs_frame != y.obs_frame) return false;
    }
    return true;
}

int count_kind(const Circuit& c, Instr::Kind k) {
    int n = 0;
    for (const Instr& ins : c.stream)
        if (ins.kind == k) ++n;
    return n;
}

// Stream positions of every two-qubit gate `g a b`.
std::vector<int> positions_of(const Circuit& c, GateKind g, int a, int b) {
    std::vector<int> out;
    for (size_t i = 0; i < c.stream.size(); ++i) {
        const Instr& ins = c.stream[i];
        if (ins.kind == Instr::Kind::Gate && ins.gate == g && ins.targets == std::vector<int>{a, b})
            out.push_back((int)i);
    }
    return out;
}

// Stream position of the k-th (0-based) Measure on `qubit`; -1 if absent.
int measure_pos(const Circuit& c, int qubit, int k = 0) {
    for (size_t i = 0; i < c.stream.size(); ++i) {
        const Instr& ins = c.stream[i];
        if (ins.kind == Instr::Kind::Measure && ins.qubits == std::vector<int>{qubit} && k-- == 0)
            return (int)i;
    }
    return -1;
}

// A NON-crossing feedback: pure Clifford region, no T/CS/CCZ/CH anywhere.
const char* kNonCrossing =
    "R 0 1\n"
    "H 1\n"
    "M 1\n"
    "CX rec[-1] 0\n"
    "M 0\n";

// A CROSSING feedback: the X byproduct on qubit 0 reaches a T.
const char* kCrossing =
    "R 0 1\n"
    "H 1\n"
    "M 1\n"
    "CX rec[-1] 0\n"
    "T 0\n"
    "M 0\n";

const char* kFeedbackFree =
    "R 0 1\n"
    "H 1\n"
    "CX 1 0\n"
    "M 0 1\n";

// 1 + 2: the flag is exactly what decides a non-crossing feedback's fate.
void test_noncrossing_kept_by_default_coherentized_by_flag() {
    Circuit user = parse(kNonCrossing);
    CHECK_EQ(count_feedback(user), 1);

    Circuit def = coherentize_classically_controlled_feedback(user);
    CHECK_EQ(count_feedback(def), 1);          // left for the downstream relabel (3.1.0 behaviour)
    CHECK(same_stream(def, user));             // and left byte-for-byte alone

    Circuit all = coherentize_classically_controlled_feedback(user, /*coherentize_all=*/true);
    CHECK_EQ(count_feedback(all), 0);          // rewritten into coherent gates
    // An idle Z-basis control with an X byproduct rewrites 1:1 into `CX control target`,
    // so check the GATE, not the stream length.
    int cx_1_0 = 0;
    for (const Instr& ins : all.stream)
        if (ins.kind == Instr::Kind::Gate && ins.gate == GateKind::CX &&
            ins.targets == std::vector<int>{1, 0})
            ++cx_1_0;
    CHECK_EQ(cx_1_0, 1);                       // the coherent entangler is really there
    CHECK_EQ((int)all.stream.size(), (int)user.stream.size());
}

// 3: feedback-free circuits early-out under BOTH settings — no scan, no rebuild, no cost.
void test_feedback_free_unchanged_under_both() {
    Circuit user = parse(kFeedbackFree);
    CHECK_EQ(count_feedback(user), 0);
    CHECK(same_stream(coherentize_classically_controlled_feedback(user, false), user));
    CHECK(same_stream(coherentize_classically_controlled_feedback(user, true), user));
}

// 4: crossing feedback is coherentized identically either way — the byte-identity argument
//    for the shipped magic benchmarks.
void test_crossing_identical_under_both() {
    Circuit user = parse(kCrossing);
    CHECK_EQ(count_feedback(user), 1);
    Circuit def = coherentize_classically_controlled_feedback(user, false);
    Circuit all = coherentize_classically_controlled_feedback(user, true);
    CHECK_EQ(count_feedback(def), 0);
    CHECK_EQ(count_feedback(all), 0);
    CHECK(same_stream(def, all));
}

// 5: through NormalizePolicy — the binding's new policy pair vs the 3.1.0 one.
void test_policy_reject_has_no_residue_where_strip_deleted_it() {
    Circuit user = parse(kNonCrossing);

    // 3.1.0 binding policy: coherentize (crossing only) + Strip. The feedback SURVIVES
    // coherentize and is then deleted from the consumer circuit — the defect.
    NormalizePolicy old_pol;
    old_pol.coherentize = true;
    old_pol.defer = true;
    old_pol.want_map = true;
    old_pol.feedback = NormalizePolicy::Feedback::Strip;
    NormalizeResult old_nr = normalize(user, old_pol);
    CHECK_EQ(count_feedback(old_nr.coherent), 1);      // survived coherentize ...
    CHECK_EQ(count_feedback(old_nr.normalized), 0);    // ... and was stripped away
    CHECK(!old_nr.feedback_rejected);

    // New binding policy: coherentize_all + Reject. Nothing survives, so nothing is rejected
    // and nothing is silently deleted.
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    NormalizeResult nr = normalize(user, pol);
    CHECK(!nr.feedback_rejected);
    CHECK_EQ(count_feedback(nr.coherent), 0);
    CHECK_EQ(count_feedback(nr.normalized), 0);
    // The coherentized entangler really is in the consumer circuit (not merely absent):
    // the deferred stream is strictly longer than the stripped 3.1.0 one.
    CHECK(nr.normalized.stream.size() > old_nr.normalized.stream.size());

    // And a feedback-free circuit is byte-unaffected by the new policy pair.
    Circuit ff = parse(kFeedbackFree);
    NormalizeResult a = normalize(ff, old_pol);
    NormalizeResult b = normalize(ff, pol);
    CHECK(!b.feedback_rejected);
    CHECK(same_stream(a.normalized, b.normalized));
}

// 6a: the control is never observed after its noisy measurement (Reset retires it; Noise on the
//     wire in between is NOT an observation) -> the single pre-measurement error, no new ancilla.
void test_noisy_fold_single_error_when_control_retired() {
    for (const char* text : {
             "R 0 1\nM(0.5) 1\nCX rec[-1] 0\nR 1\nM 0 1\n",
             "R 0 1\nM(0.5) 1\nCX rec[-1] 0\nX_ERROR(0.1) 1\nR 1\nM 0 1\n",   // noise, then Reset
             "R 0 1\nM(0.5) 1\nCX rec[-1] 0\nM 0\n",                          // never touched again
         }) {
        Circuit user = parse(text);
        Circuit all = coherentize_classically_controlled_feedback(user, /*coherentize_all=*/true);
        CHECK_EQ(count_feedback(all), 0);
        CHECK_EQ(all.n, user.n);                                   // no copy / flip ancilla
        // Exactly one folded error: X_ERROR(0.5) on the control, immediately before its Measure,
        // and that Measure is now readout-noiseless.
        int folded = 0;
        for (size_t i = 0; i < all.stream.size(); ++i) {
            const Instr& ins = all.stream[i];
            if (ins.kind == Instr::Kind::Noise && ins.channel == NoiseChannel::X_ERROR &&
                ins.qubits == std::vector<int>{1} && ins.probs == std::vector<double>{0.5}) {
                ++folded;
                CHECK(i + 1 < all.stream.size());
                CHECK(all.stream[i + 1].kind == Instr::Kind::Measure);
                CHECK_EQ(all.stream[i + 1].qubits[0], 1);
                CHECK_EQ(all.stream[i + 1].readout_flip_p, 0.0);
            }
        }
        CHECK_EQ(folded, 1);
        CHECK_EQ(positions_of(all, GateKind::CX, 1, 0).size(), (size_t)1);   // driven from the control
    }
}

// 6b: the control's state IS observed after its noisy measurement (re-measured) -> the flip
//     instance is held on a fresh ancilla f, applied to the control before and after the
//     measurement, and the feedback drives from a copy ancilla taken after the first flip.
void test_noisy_fold_held_when_control_observed() {
    Circuit user = parse("R 0 1\nM(0.5) 1\nCX rec[-1] 0\nM 1\nM 0\n");
    Circuit all = coherentize_classically_controlled_feedback(user, /*coherentize_all=*/true);
    CHECK_EQ(count_feedback(all), 0);
    CHECK_EQ(all.n, 4);                                            // copy ancilla 2, flip ancilla 3
    // The held flip: X_ERROR(0.5) on f=3 (and NO error on the control itself).
    CHECK_EQ(count_kind(all, Instr::Kind::Noise), 1);
    for (const Instr& ins : all.stream)
        if (ins.kind == Instr::Kind::Noise) {
            CHECK(ins.channel == NoiseChannel::X_ERROR);
            CHECK(ins.qubits == std::vector<int>{3});
            CHECK(ins.probs == std::vector<double>{0.5});
        }
    // CX f ctrl twice, bracketing the (now readout-noiseless) source measurement.
    std::vector<int> flips = positions_of(all, GateKind::CX, 3, 1);
    CHECK_EQ(flips.size(), (size_t)2);
    const int m_src = measure_pos(all, 1, 0);
    CHECK(m_src >= 0);
    CHECK(flips[0] < m_src && m_src < flips[1]);
    CHECK_EQ(all.stream[m_src].readout_flip_p, 0.0);
    // The copy CX ctrl->anc sits AFTER the first flip and BEFORE the measurement, so the copy
    // carries the FLIPPED value; the feedback drives from the copy, not the control.
    std::vector<int> copy = positions_of(all, GateKind::CX, 1, 2);
    CHECK_EQ(copy.size(), (size_t)1);
    CHECK(flips[0] < copy[0] && copy[0] < m_src);
    CHECK_EQ(positions_of(all, GateKind::CX, 2, 0).size(), (size_t)1);
    CHECK_EQ(positions_of(all, GateKind::CX, 1, 0).size(), (size_t)0);
    // The user's second measurement of the control follows the undo.
    CHECK(measure_pos(all, 1, 1) > flips[1]);

    // X-basis source: the anticommuting held flip is CZ f ctrl (Z on the control).
    Circuit ux = parse("R 0 1\nMX(0.5) 1\nCX rec[-1] 0\nM 1\nM 0\n");
    Circuit ax = coherentize_classically_controlled_feedback(ux, true);
    CHECK_EQ(positions_of(ax, GateKind::CZ, 3, 1).size(), (size_t)2);
    CHECK_EQ(positions_of(ax, GateKind::CX, 3, 1).size(), (size_t)0);

    // Default mode (crossing feedback, the exact/CLI consumers) takes the SAME held form: the fold
    // is a property of the coherentizer, not of `coherentize_all`.
    Circuit uc = parse("R 0 1\nM(0.5) 1\nCX rec[-1] 0\nT 0\nM 1\nM 0\n");
    Circuit dc = coherentize_classically_controlled_feedback(uc, false);
    CHECK_EQ(count_feedback(dc), 0);
    CHECK_EQ(dc.n, 4);
    CHECK_EQ(positions_of(dc, GateKind::CX, 3, 1).size(), (size_t)2);
}

// Any Gate whose target list repeats a qubit — `CX q q` and friends. The parser rejects these
// in user text; nothing the coherentizer emits may contain one either.
bool has_degenerate_gate(const Circuit& c) {
    for (const Instr& ins : c.stream) {
        if (ins.kind != Instr::Kind::Gate) continue;
        for (size_t i = 0; i < ins.targets.size(); ++i)
            for (size_t j = i + 1; j < ins.targets.size(); ++j)
                if (ins.targets[i] == ins.targets[j]) return true;
    }
    return false;
}

// 7a: self-target feedback with an IDLE control -> copy gadget, never `CX q q`. Z-basis source:
//     copy `CX q anc` before the measurement, entangler `CX anc q` after it; X/Y sources likewise
//     rewrite without a degenerate gate. Default mode (crossing) takes the same gadget.
void test_self_target_feedback_takes_copy_gadget() {
    Circuit user = parse("R 0 1\nH 0\nM 0\nCX rec[-1] 0\nM 0\nM 1\n");
    Circuit all = coherentize_classically_controlled_feedback(user, /*coherentize_all=*/true);
    CHECK_EQ(count_feedback(all), 0);
    CHECK(!has_degenerate_gate(all));
    CHECK_EQ(all.n, 3);                                            // one copy ancilla (2), no flip
    const int m_src = measure_pos(all, 0, 0);
    CHECK(m_src >= 0);
    std::vector<int> copy = positions_of(all, GateKind::CX, 0, 2);
    std::vector<int> drive = positions_of(all, GateKind::CX, 2, 0);
    CHECK_EQ(copy.size(), (size_t)1);
    CHECK_EQ(drive.size(), (size_t)1);
    CHECK(copy[0] < m_src && m_src < drive[0]);                    // copy before M, drive after
    CHECK(drive[0] < measure_pos(all, 0, 1));                      // ... and before the re-read

    for (const char* text : {
             "R 0 1\nMX 0\nCZ rec[-1] 0\nMX 0\nM 1\n",              // X source, Z byproduct
             "R 0 1\nMY 0\nCX rec[-1] 0\nMY 0\nM 1\n",              // Y source, X byproduct
             "R 0 1\nH 0\nM !0\nCY rec[-1] 0\nM 0\nM 1\n",          // inverted source, Y byproduct
             "R 0 1\nH 0\nM 0\nCX rec[-1] 0\nCX rec[-1] 1\nM 0\nM 1\n",  // shared source, one self
         }) {
        Circuit u = parse(text);
        Circuit a = coherentize_classically_controlled_feedback(u, true);
        CHECK_EQ(count_feedback(a), 0);
        CHECK(!has_degenerate_gate(a));
        CHECK_EQ(a.n, u.n + 1);                                    // exactly one copy ancilla
    }

    // Default mode, crossing self-target (`T` after the reset): the same copy gadget.
    Circuit uc = parse("R 0 1\nH 0\nM 0\nCX rec[-1] 0\nT 0\nM 0\nM 1\n");
    Circuit dc = coherentize_classically_controlled_feedback(uc, false);
    CHECK_EQ(count_feedback(dc), 0);
    CHECK(!has_degenerate_gate(dc));
    CHECK_EQ(dc.n, 3);

    // A feedback onto a DIFFERENT idle wire still drives in place (no ancilla) — the self-target
    // rule must not widen the copy class.
    Circuit uo = parse(kNonCrossing);
    Circuit ao = coherentize_classically_controlled_feedback(uo, true);
    CHECK_EQ(ao.n, uo.n);
}

// 7b: the guard — emit_coherent_feedback refuses a control that is its own target, so a
//     degenerate entangler can never leave the coherentizer silently again.
void test_emit_coherent_feedback_rejects_self_target() {
    bool threw = false;
    try {
        emit_coherent_feedback(/*a=*/0, PauliBasis::Z, /*q=*/0, PauliBasis::X, false);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    // Distinct wires: the plain 1:1 rewrite, exactly one gate.
    std::vector<Instr> ok = emit_coherent_feedback(1, PauliBasis::Z, 0, PauliBasis::X, false);
    CHECK_EQ(ok.size(), (size_t)1);
    CHECK((ok[0].targets == std::vector<int>{1, 0}));
}

}  // namespace

int main() {
    RUN(test_noncrossing_kept_by_default_coherentized_by_flag);
    RUN(test_feedback_free_unchanged_under_both);
    RUN(test_crossing_identical_under_both);
    RUN(test_policy_reject_has_no_residue_where_strip_deleted_it);
    RUN(test_noisy_fold_single_error_when_control_retired);
    RUN(test_noisy_fold_held_when_control_observed);
    RUN(test_self_target_feedback_takes_copy_gadget);
    RUN(test_emit_coherent_feedback_rejects_self_target);
    REPORT();
}
