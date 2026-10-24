#include "qeccore/normalize.hpp"

#include "qeccore/clifford_frames.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/propagation_table.hpp"

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
