#pragma once
#include <string>
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
    // Coherentize EVERY classically-controlled Pauli, not only those whose byproduct crosses a
    // non-Clifford gate. Requires `coherentize`. Set by consumers that run NO downstream
    // sign-relabel (`build_feedback_plan`) and therefore have no exact home for a surviving
    // ControlledPauli — the pybind TwirlSampler family (`normalize_for_twirl`). The coherentizer
    // rewrites each feedback or throws, so `coherent` then never carries a ControlledPauli and the
    // `feedback` policy has nothing to act on. Consumers that DO relabel (cpp/src/sampler.cpp,
    // dem_export) leave this false: their bytes, and every feedback-free circuit's bytes, are
    // unaffected.
    bool coherentize_all = false;
    bool defer            = true;    // defer_measurements (terminal reads at the end)
    bool want_map         = false;   // fill `map` from the deferral (requires defer)
    // Defer live-wire X/Y measurements by the diagonal copy (defer_measurements' diagonal_copy;
    // see deferral.hpp). Set ONLY by DEM export's fallback when the default deferral leaves the
    // diagonal propagation class; false everywhere else (byte-identical default).
    bool diagonal_copy    = false;
    // Class-exit fallback for coherentized feedback (requires `coherentize_all` + `defer`): when
    // the deferred circuit leaves the diagonal propagation class (some fault atom rejects without
    // the PPR retry) AND hoisting feedbacks to their source measurements
    // (coherentize_classically_controlled_feedback's `hoist`) yields a circuit that stays in it,
    // the hoisted form is used instead. Same channel, different (diagonal-class) representation.
    // Every circuit whose default form is already in class, or that has nothing to hoist, is
    // returned byte-identical. Set only by the TwirlSampler family (`normalize_for_twirl`),
    // whose state-retaining sampler serves the diagonal class only.
    bool hoist_feedback_on_class_exit = false;
    // eliminate_hadamards' `zero_noise_transparent`: a Noise instruction whose probabilities are
    // all zero (the identity channel) lets a pending single-qubit Clifford frame pass instead of
    // flushing it, so `H q; Z_ERROR(0) q; H q` cancels. Set ONLY by the twirl record samplers
    // (xtim_py normalize_for_twirl, the family whose state retention needs the diagonal residual
    // class); false everywhere else (byte-identical default) -- in particular the forced-fault
    // compiler, which fires a user Pauli letter at the site's FLUSHED position.
    bool zero_noise_transparent = false;
};

struct NormalizeResult {
    Circuit coherent;               // post-coherentize, pre-strip (the sampler's feedback plan input)
    Circuit normalized;             // consumer-ready (deferred iff policy.defer, then eliminate_hadamards)
    DeferralMap map;                // valid iff policy.want_map (and policy.defer)
    bool feedback_rejected = false; // Reject policy hit residual feedback after coherentize
    int  feedback_hoisted = 0;      // feedbacks hoisted by the class-exit fallback (0 = default form)
};

// Pipeline, IN ORDER:
//   coherent   = coherentize_classically_controlled_feedback(user)
//   feedback policy: KeepCoherent/Strip -> strip_feedback for the consumer circuit;
//                    Reject -> set feedback_rejected + return early if residual feedback
//   defer_measurements (with map iff want_map) (if policy.defer)
//   eliminate_hadamards  (ALWAYS, LAST)
NormalizeResult normalize(const Circuit& user, const NormalizePolicy& policy);

}  // namespace qeccore
