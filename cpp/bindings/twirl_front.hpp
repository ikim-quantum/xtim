// twirl_front.hpp — J-b1 F: the twirl front end shared by xtim._xtim and the test API's legacy sampler: the
// normalization presets, the port-verdict dict and the bare-state refusal text. Moved verbatim from xtim_py.cpp.
#pragma once

#include "common_py.hpp"
#include <set>
#include "qeccore/normalize.hpp"
#include "qeccore/port_contract.hpp"
#include "refusal_text.hpp"

namespace {

// Task 2: convert a PortVerdict to a Python dict with keys:
//   product (bool), k_port (int), stabilized (list of (q,axis,sign) tuples),
//   port_wires (list of int), witness (str).
inline py::dict port_verdict_to_dict(const PortVerdict& v) {
    py::dict d;
    d["product"]    = v.product;
    d["k_port"]     = v.k_port;
    py::list stab;
    for (const ProdQ& pq : v.stabilized)
        stab.append(py::make_tuple((int)pq.q, (int)pq.axis, (int)pq.sign));
    d["stabilized"] = stab;
    py::list pw;
    for (int w : v.port_wires) pw.append(w);
    d["port_wires"] = pw;
    d["witness"]    = v.witness;
    return d;
}

// ── The normalization preset shared by every twirl-path entry point ─────────────────────────
// The TwirlSampler family builds NO FeedbackPlan (unlike cpp/src/sampler.cpp, which normalizes
// KeepCoherent and then relabels), so a ControlledPauli that survived coherentization would have
// no exact home here — `Strip`, the policy through 3.1.0, silently DELETED it from the retained
// state and the record algebra. `coherentize_all` rewrites EVERY feedback into coherent gates (or
// throws on a malformed record), so nothing can survive to be stripped or rejected; `Reject`
// states the intent — a survivor would be an engine bug, never a feature to strip.
inline NormalizeResult normalize_for_twirl(const Circuit& user) {
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    // A coherentized feedback driven by an X/Y read of a wire that carries a fault's
    // Pauli-times-diagonal residual (a fault before a T on that wire) puts the residual through
    // the copy-out Hadamard and out of the diagonal class — refused by the state-retaining
    // sampler. The fallback hoists such feedbacks to their source measurements (exact: they
    // commute) only when that keeps the circuit in class; every other circuit is unchanged.
    pol.hoist_feedback_on_class_exit = true;
    return normalize(user, pol);
}

// The SAMPLER preset: the preset above plus `zero_noise_transparent`. The twirl family builds its
// atoms with AtomSet::ChannelSupport (propagation_table.hpp): a factor of probability zero is never
// fired, so its atom is ABSENT — a p=0 noise instruction is nothing to this consumer, and the one
// place it was not nothing was push_frames, where any Noise flushed a pending Clifford frame
// (`H 1; Z_ERROR(0) 1; H 1` after a magic gate left an explicit H pair -> CH/PPR routing, state
// retention refused). Under this preset the frame passes it unconditionally (the instruction stays
// in place; noise sites and RNG draws unchanged). Used by the three TwirlSampler constructors and by
// the bare/input-state providers they compose with (_bare_state_of / _input_state_of), so provider
// and consumer see one deferred frame. NOT used by deferred_signature (the reference cache key of
// ref_compile, whose consumer — the exact sampler — normalizes flushed), the bare-state oracle
// diffs, or output_wires_of (final_wire is frame-independent). The forced-fault compiler
// (AtomSet::AnyPauli: ANY Pauli may be forced at a p=0 site, so the site is a real position) keeps
// the flushed frame by contract.
static NormalizeResult normalize_for_twirl_sampler(const Circuit& user) {
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;   // as normalize_for_twirl (merge of fix/feedback-after-t + fix/xtim-small-items)
    pol.zero_noise_transparent = true;
    return normalize(user, pol);
}

// "bare state rejected: ..." with, for a non-commuting-magic rejection, the two anticommuting
// terms rendered on USER qubits (the engine's text names deferred-space wires, e.g. "X15 and
// X14Z15": fresh wires the coherentizer and the deferral allocated above the user's qubits).
// Each deferred wire is traced through DeferralMap::wire_origin (measurement copy / retired /
// reset wire of an input wire) and, for a coherentizer copy ancilla, to the control it copies;
// then the non-Clifford gates of the parsed circuit on those user qubits are listed (flat
// instruction index, REPEAT unrolled — not a source line). Messages only; no stream reads it.
// Generic over BareState (CanonicalStabSum) and FramedBareState (memoised framed build): both carry
// the reject_* fields this message reads (merge of fix/xtim-small-items + perf/dem-share-bare).
template <class BS>
static std::string bare_reject_message(const BS& bs, const NormalizeResult& nr, const ParsedStim& ps) {
    // J-b1 F (F item 3): one owner of the user-qubit rendering (refusal_text.hpp's user_terms_text: the two
    // anticommuting terms in user qubits, then the parsed circuit's non-Clifford gates on them).
    return "bare state rejected: " + (bs.reject_reason.empty() ? std::string("unknown cause") : bs.reject_reason) +
           user_terms_text(bs.reject_terms, nr, ps);
}

}  // namespace
