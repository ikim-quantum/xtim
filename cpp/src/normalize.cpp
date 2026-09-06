#include "qeccore/normalize.hpp"

#include "qeccore/clifford_frames.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/feedback.hpp"

namespace qeccore {

NormalizeResult normalize(const Circuit& user, const NormalizePolicy& policy) {
    NormalizeResult res;

    // 1. Coherentize classically-controlled feedback (default). The samplers + reference builder
    //    consume `coherent` (pre-strip) and MUST coherentize so their deferred circuits agree. The
    //    T*-certify path sets policy.coherentize=false: it certifies the bare magic STATE, so the
    //    trailing classical corrections are stripped raw rather than entangled into the state.
    res.coherent = policy.coherentize
                       ? coherentize_classically_controlled_feedback(user)
                       : user;

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
                return res;  // consumer bails on the flag; do not defer (would throw).
            }
            work = res.coherent;
            break;
    }

    // 3. Defer measurements (terminal reads at the end), optionally filling the map.
    if (policy.defer) {
        if (policy.want_map)
            work = defer_measurements(work, &res.map);
        else
            work = defer_measurements(work);
    }

    // 4. Eliminate Hadamards: ALWAYS, LAST. No-op early-out when already Hadamard-free.
    res.normalized = eliminate_hadamards(work);

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

    return res;
}

}  // namespace qeccore
