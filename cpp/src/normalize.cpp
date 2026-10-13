#include "qeccore/normalize.hpp"

#include "qeccore/clifford_frames.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/propagation_table.hpp"

#include <map>
#include <set>
#include <string>

namespace qeccore {

namespace {

// Steps 2-4b of the pipeline on an already-coherentized circuit (`res.coherent` set).
void finish_normalize(NormalizeResult& res, const NormalizePolicy& policy);

// Can any fault atom leave the diagonal class at all? Only a fault that later crosses a
// non-Clifford gate carries a non-Pauli residual (a Pauli through Clifford gates stays a
// Pauli), so a circuit with no Noise instruction before some non-Clifford gate is in class
// by construction — no table needed. A linear scan of the coherent stream (readout folds
// included).
bool fault_can_meet_nonclifford(const Circuit& c) {
    bool noise_seen = false;
    for (const Instr& ins : c.stream) {
        if (ins.kind == Instr::Kind::Noise) noise_seen = true;
        else if (noise_seen && ins.kind == Instr::Kind::Gate &&
                 (ins.gate == GateKind::T || ins.gate == GateKind::CS ||
                  ins.gate == GateKind::CCZ || ins.gate == GateKind::CH))
            return true;
    }
    return false;
}

}  // namespace

NormalizeResult normalize(const Circuit& user, const NormalizePolicy& policy) {
    NormalizeResult res;

    // 1. Coherentize classically-controlled feedback (default). The samplers + reference builder
    //    consume `coherent` (pre-strip) and MUST coherentize so their deferred circuits agree. The
    //    T*-certify path sets policy.coherentize=false: it certifies the bare magic STATE, so the
    //    trailing classical corrections are stripped raw rather than entangled into the state.
    res.coherent = policy.coherentize
                       ? coherentize_classically_controlled_feedback(user, policy.coherentize_all)
                       : user;
    finish_normalize(res, policy);

    // 5. Class-exit fallback (policy.hoist_feedback_on_class_exit; the TwirlSampler family).
    //    Cheapest tests first (feedback-free circuits never get here): can any fault reach a
    //    non-Clifford gate at all (a linear scan), and is there anything to hoist (one
    //    coherentize pass)? Only then the deciding test — does the default form leave the
    //    diagonal propagation class? — the same table build every consumer runs, without the
    //    PPR retry. The hoisted form replaces it only if IT stays in class; anything else keeps
    //    the default form byte for byte (and its PPR routing, as before).
    if (policy.hoist_feedback_on_class_exit && policy.coherentize && policy.coherentize_all &&
        policy.defer && !res.feedback_rejected && circuit_has_feedback(user) &&
        fault_can_meet_nonclifford(res.coherent)) {
        int n_hoisted = 0;
        Circuit hoisted =
            coherentize_classically_controlled_feedback(user, true, /*hoist=*/true, &n_hoisted);
        if (n_hoisted > 0 && !build_propagation_table(res.normalized).all_in_class) {
            NormalizeResult alt;
            alt.coherent = std::move(hoisted);
            finish_normalize(alt, policy);
            if (!alt.feedback_rejected && build_propagation_table(alt.normalized).all_in_class) {
                alt.feedback_hoisted = n_hoisted;
                return alt;
            }
        }
    }
    return res;
}

std::string explain_feedback_class_exit(const Circuit& user, const NormalizePolicy& policy,
                                       int user_qubits) {
    if (!circuit_has_feedback(user) || !policy.defer) return "";
    NormalizeResult with = normalize(user, policy);
    if (with.feedback_rejected || build_propagation_table(with.normalized).all_in_class) return "";
    NormalizePolicy bare = policy;
    bare.coherentize = false;                  // feedback stripped raw: the feedback-free circuit
    bare.hoist_feedback_on_class_exit = false;
    bare.feedback = NormalizePolicy::Feedback::Strip;
    NormalizeResult without = normalize(user, bare);
    if (!build_propagation_table(without.normalized).all_in_class) return "";

    // Attribution BY CLASS MEMBERSHIP, through the very pipeline that refused: feedback k is a
    // CAUSE iff the circuit with every OTHER feedback removed (k kept, normalized with the same
    // policy, move fallback included) still leaves the class. A feedback that is harmless on its
    // own is never blamed. How it fails is read off the same per-feedback theorem verdict
    // (feedback_hoist_report == coherentize's Pass 1c).
    const std::vector<FeedbackHoistInfo> report = feedback_hoist_report(user, user_qubits);
    // The circuit with exactly the feedbacks at the given stream indices kept.
    auto keep = [&](const std::set<int>& kept) {
        Circuit c = user;
        c.stream.clear();
        for (int i = 0; i < (int)user.stream.size(); ++i)
            if (user.stream[i].kind != Instr::Kind::ControlledPauli || kept.count(i))
                c.stream.push_back(user.stream[i]);
        return c;
    };
    auto exits = [&](const std::set<int>& kept) {
        NormalizeResult r = normalize(keep(kept), policy);
        return !r.feedback_rejected && !build_propagation_table(r.normalized).all_in_class;
    };
    auto only = [&](int keep_cp) { return keep(std::set<int>{keep_cp}); };
    std::vector<const FeedbackHoistInfo*> causes;
    for (const FeedbackHoistInfo& f : report) {
        NormalizeResult r = normalize(only(f.cp_index), policy);
        if (!r.feedback_rejected && !build_propagation_table(r.normalized).all_in_class)
            causes.push_back(&f);
    }

    // User-level naming: the target is always a user qubit; the source is named by its record
    // (absolute, Stim order) and, if it is a plain measurement, by basis and qubit.
    auto name = [&](const FeedbackHoistInfo& f) {
        const char* p = f.pauli == PauliBasis::X ? "CX" : f.pauli == PauliBasis::Y ? "CY" : "CZ";
        const char* m = f.source_basis == PauliBasis::X ? "MX" : f.source_basis == PauliBasis::Y ? "MY" : "M";
        std::string src = "record #" + std::to_string(f.source_record);
        if (user_qubits >= 0 && f.control >= user_qubits)
            src += " (an MPP/MPAD product measurement)";
        else
            src += " (" + std::string(m) + " of qubit " + std::to_string(f.control) + ")";
        return "feedback #" + std::to_string(f.ordinal) + " (" + p + " rec onto qubit " +
               std::to_string(f.target) + ", reading " + src + ")";
    };
    auto how = [&](const FeedbackHoistInfo& f) -> std::string {
        if (!f.hoistable)
            return name(f) + " is NOT MOVABLE to its source measurement by the deferred-"
                   "measurement rule: its window contains " + f.why + ". Left after its "
                   "measurement, its coherent form takes a fault residual out of the diagonal class";
        if (f.xy_from_xy_read)
            return name(f) + " was moved to its source measurement, but its moved Pauli is "
                   "X/Y-type and driven by an X/Y read: with the control's residual S it becomes an "
                   "(I + iX)-type rotation on a kept qubit, outside the retained engine's Pauli x "
                   "diagonal-Clifford representation";
        return name(f) + " was moved to its source measurement, but the moved gate couples a fault "
               "residual onto its target, where a later non-diagonal Clifford (in the window or "
               "after it) makes it an (I + iX)-type rotation on a kept qubit, outside the retained "
               "engine's Pauli x diagonal-Clifford representation";
    };
    std::string msg =
        "classically-controlled Pauli feedback (CX/CY/CZ rec[-k] q) puts this circuit out of the "
        "diagonal class on the state-retaining path, although the same circuit without the "
        "feedback is in it. The retained state carries each feedback COHERENTLY, and a fault "
        "before a non-Clifford gate leaves a Pauli-times-diagonal residual (T X T^dag ~ X S^dag) "
        "on that gate's wires.";
    bool representation = false;
    if (!causes.empty()) {
        msg += causes.size() == 1 ? " Cause (it leaves the class on its own): "
                                  : " Causes (each leaves the class on its own): ";
        for (size_t i = 0; i < causes.size(); ++i) {
            msg += (i ? "; " : "") + how(*causes[i]);
            representation |= causes[i]->hoistable;
        }
        msg += ".";
    } else {
        // No single cause: shrink to a jointly-fatal set by the same membership
        // test (drop a feedback whenever the rest still exits; one normalize per feedback),
        // then name it and the wires its members' moved gates share.
        std::set<int> set;
        for (const FeedbackHoistInfo& f : report) set.insert(f.cp_index);
        for (const FeedbackHoistInfo& f : report) {
            std::set<int> trial = set;
            trial.erase(f.cp_index);
            if (!trial.empty() && exits(trial)) set = std::move(trial);
        }
        std::map<int, int> wire_use;   // qubit -> number of moved members acting on it
        bool all_moved = true;
        std::string members;
        for (const FeedbackHoistInfo& f : report) {
            if (!set.count(f.cp_index)) continue;
            members += (members.empty() ? "" : "; ") + name(f) +
                       (f.hoistable ? std::string(f.xy_from_xy_read ? " (moved; X/Y-type from an "
                                                                      "X/Y read)"
                                                                    : " (moved)")
                                    : " (not movable: " + f.why + ")");
            if (!f.hoistable) { all_moved = false; continue; }
            for (int q : f.moved_support) ++wire_use[q];
        }
        std::string shared;
        for (const auto& kv : wire_use)
            if (kv.second > 1) shared += (shared.empty() ? "qubit " : ", qubit ") + std::to_string(kv.first);
        msg += " No single feedback leaves the class on its own; these do together (each "
               "stays in class alone; found by dropping feedbacks while the rest still leaves "
               "the class): " +
               members + ".";
        if (all_moved && !shared.empty()) {
            msg += " Their moved gates act on the same wire(s) (" + shared + "): each couples a "
                   "fault residual onto that wire, and the composition leaves the retained "
                   "engine's Pauli x diagonal-Clifford representation (an (I + iX)-type rotation "
                   "on a kept qubit).";
            representation = true;
        } else if (all_moved) {
            msg += " All are moved; their moved gates share no wire, so the exit comes from how "
                   "their couplings meet later gates.";
            representation = true;
        }
    }
    if (representation)
        msg += " A moved feedback's exit is a limit of the retained-state representation (it "
               "would need per-shot collapse on the record), not of the circuit.";
    msg += " Refused, never approximated; sample() without random-record DECISIONs is "
           "unaffected (PPR route). For retained states apply that Pauli as a frame on the "
           "recorded bit";
    return msg;
}

namespace {

void finish_normalize(NormalizeResult& res, const NormalizePolicy& policy) {

    // 2. Feedback policy. NOTE: the deferred/normalized circuit must be feedback-FREE because
    //    defer_measurements cannot represent a ControlledPauli (it throws). So:
    //      KeepCoherent: the sampler keeps `res.coherent` (pre-strip) for its feedback PLAN, but
    //                    the consumer circuit is stripped so it can defer (mirrors sampler.cpp:
    //                    `src = has_fb ? strip_feedback(coherent) : coherent`).
    //      Strip:        strip feedback from the consumer circuit.
    //      Reject:       if any ControlledPauli SURVIVED coherentize, set feedback_rejected and
    //                    return early — the consumer (dem_export) bails before deferral, exactly
    //                    as it does today (it checks circuit_has_feedback and refuses).
    Circuit work;
    switch (policy.feedback) {
        case NormalizePolicy::Feedback::KeepCoherent:
        case NormalizePolicy::Feedback::Strip:
            work = strip_feedback(res.coherent);
            break;
        case NormalizePolicy::Feedback::Reject:
            if (circuit_has_feedback(res.coherent)) {
                res.feedback_rejected = true;
                return;  // consumer bails on the flag; do not defer (would throw).
            }
            work = res.coherent;
            break;
    }

    // 3. Defer measurements (terminal reads at the end), optionally filling the map.
    if (policy.defer) {
        if (policy.diagonal_copy)
            work = defer_measurements(work, policy.want_map ? &res.map : nullptr, true);
        else if (policy.want_map)
            work = defer_measurements(work, &res.map);
        else
            work = defer_measurements(work);
    }

    // 4. Eliminate Hadamards: ALWAYS, LAST. No-op early-out when already Hadamard-free.
    res.normalized = eliminate_hadamards(work, policy.zero_noise_transparent);

    // 4b. Reconcile the terminal-read map with the ACTUAL post-elimination read bases.
    //     defer_measurements (step 3) fills map.terminal_reads at DEFER time, before
    //     eliminate_hadamards folds a pending single-qubit Clifford frame into the read
    //     basis (H q; M q (Z) -> MX q). That fold leaves map.terminal_reads[i].second stale
    //     (still Z) while the consumer's record operator W_k is built from this basis — so a
    //     folded read (e.g. a Bell-partner X measurement written as `H 1; M 1`) would be
    //     mis-measured as Z. The i-th trailing single-qubit Measure in `normalized`
    //     corresponds to terminal_reads[i] (defer makes all reads terminal and
    //     order-preserving; eliminate_hadamards preserves Measure count/order, only rewriting
    //     the basis/invert). Overwrite the map basis (and qubit, which is unchanged in
    //     practice) from the normalized stream. A NO-OP whenever no fold occurred (basis
    //     already matches) — non-fold circuits are byte-identical.
    if (policy.defer && policy.want_map) {
        size_t i = 0;
        for (const Instr& ins : res.normalized.stream) {
            if (ins.kind != Instr::Kind::Measure || ins.qubits.size() != 1) continue;
            if (i >= res.map.terminal_reads.size()) break;
            res.map.terminal_reads[i].first = ins.qubits[0];
            res.map.terminal_reads[i].second = ins.basis;
            ++i;
        }
    }
}

}  // namespace

}  // namespace qeccore
