#pragma once
#include "qeccore/circuit_ir.hpp"
#include "qeccore/deferral.hpp"

namespace qeccore {

// Shared circuit-normalization entry point. Consolidates the scattered passes
// (coherentize feedback -> feedback policy -> defer measurements ->
// eliminate Hadamards) behind ONE call, routed by a per-consumer policy. The canonical
// output is Hadamard-free: `eliminate_hadamards` runs for EVERY consumer (one canonical
// form), unconditionally and LAST (it no-ops if already Hadamard-free).
struct NormalizePolicy {
    // Feedback (classically-controlled Pauli) handling AFTER coherentize. The deferred consumer
    // circuit (`normalized`) is always feedback-FREE because defer_measurements cannot represent
    // a ControlledPauli. The three policies differ in intent, not in `normalized`'s feedback-ness:
    //   KeepCoherent: the sampler consumes `coherent` (pre-strip) for its feedback PLAN; the
    //                 consumer circuit is stripped so it can defer (mirrors sampler.cpp).
    //   Strip:        strip every ControlledPauli from the consumer circuit (reference path).
    //   Reject:       if any ControlledPauli SURVIVES coherentize, set `feedback_rejected` and
    //                 return early (no defer/eliminate); the consumer bails on the flag (DEM path).
    enum class Feedback { KeepCoherent, Strip, Reject } feedback = Feedback::Strip;
    // Coherentize classically-controlled feedback into coherent gates BEFORE the feedback policy
    // runs (the default — and MANDATORY for the samplers and the reference builder, whose deferred
    // circuits must agree). The T*-certification (`--solve`/extract) path sets this FALSE: it
    // certifies the prepared magic STATE, for which the trailing classical syndrome corrections are
    // a post-state relabel, not part of the state's non-Clifford content — coherentizing them into
    // the certify would entangle about-to-be-measured syndrome ancillas into the magic block and
    // corrupt the T* count. With coherentize=false the feedback is simply stripped (raw) for that
    // analysis-only consumer.
    bool coherentize     = true;
    bool defer            = true;    // defer_measurements (terminal reads at the end)
    bool want_map         = false;   // fill `map` from the deferral (requires defer)
};

struct NormalizeResult {
    Circuit coherent;               // post-coherentize, pre-strip (the sampler's feedback plan input)
    Circuit normalized;             // consumer-ready (deferred iff policy.defer, then eliminate_hadamards)
    DeferralMap map;                // valid iff policy.want_map (and policy.defer)
    bool feedback_rejected = false; // Reject policy hit residual feedback after coherentize
};

// Pipeline, IN ORDER:
//   coherent   = coherentize_classically_controlled_feedback(user)
//   feedback policy: KeepCoherent/Strip -> strip_feedback for the consumer circuit;
//                    Reject -> set feedback_rejected + return early if residual feedback
//   defer_measurements (with map iff want_map) (if policy.defer)
//   eliminate_hadamards  (ALWAYS, LAST)
NormalizeResult normalize(const Circuit& user, const NormalizePolicy& policy);

}  // namespace qeccore
