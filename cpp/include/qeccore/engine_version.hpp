#pragma once
// ENGINE_VERSION — the engine identity of every sampled byte stream. Bump it whenever an engine
// change could alter the constructed reference or a byte stream. It participates in the Python
// refcache key (xtim/reference.py) and is stamped into every TWPL plan-cache header (a file
// written by another engine version loads nothing and is rebuilt).
namespace qeccore {

constexpr const char* kEngineVersion = "xtim-engine-13";
// engine-13 (twirl consolidation J-b1, re-pin manifest docs/repin/J-b1-manifest.md, PRE-SIGNED conditional,
//   OD-2026-10-19-jb1-switchover-presigned): the record-FAMILY law is the twirl sampler. Every record class is a
//   row of the plan's record table — gauge (LOGICAL) detectors exact Born records, ANTI detectors and second
//   LOGICAL / ANTI observables served, CH / PPR plans through the term route ((G) / kind 2) instead of the PPR family
//   and the exact engine, input frames folded into P_rec; no exact per-shot route remains. Layout-3 records (sigma
//   retired); INPUT_QUBITS / ideal-state columns reference-relative; non-commuting magic refused by the owner's
//   static rule; TWPL v6. engine="auto" removed (default "twirl", explicit "exact").
// engine-12 (twirl consolidation S2b, re-pin manifest docs/repin/S2b-manifest.md, owner signature
//   pending): every diagonal record (IN_GROUP channels, the Born observable, Born LOGICAL / ANTI
//   decisions) is a row of the plan's RECORD TABLE in the record-ordered canonical basis — coin k =
//   COIN bit k is the value of pivot record k, Born word j drives canonical direction j (literal
//   products of the σ / extra directions), and the record trie walks only the directions a record
//   reads. The Born observable / decisions no longer take an exact route or the OBS / DEC streams on
//   diagonal plans; the barrier's σ is drawn by σ-4 (conditioned on the channels and the Born
//   observable: COIN bits ρ.., BORN words κ_rec..); the retained record is (plan, P, u, z) and
//   materialize is a forced projection. sample()/sample_barrier() byte streams move; the sample()
//   and barrier (records, σ) laws are unchanged.
// engine-11 (twirl consolidation S0(b), the owner-signed re-pin D11): ONE keyed counter-based
//   generator (qeccore/keyed_rng.hpp) for every twirl stream — noise skips (keyed per channel),
//   fair coins, κ Born steps / PPR folds, the Born observable, Born decisions, readout flips,
//   gauge coins and the exact-fallback engine — and ONE coin convention on every route (fair coin
//   k = bit k of the shot's COIN words; a Born ±1 outcome is +1 iff u < p(+1)). Bytes become
//   independent of the route, the plan-cache state and the call split (D10). Every twirl
//   sample()/sample_barrier() byte stream moves; laws are unchanged.
// engine-10 (one bump, two fixes landed together on dev/next):
//  * fix/coin-stream-and-offsets: seed=None coin stream continues (2026-10-14).
//  * fix/input-frame-mirror: input_pauli frames that a shot's residual DRESSES (a clean anchor image anticommuting with a fault atom's odd-rotation axis, e.g. a T…T_DAG mirror the frame crosses twice) are relabeled by ⟨P'·Q(shot), M_c⟩ instead of the static ⟨P', M_c⟩ — sample() and sample_barrier() bytes move on dressed-frame shots only; frame-free and undressed streams are byte-identical.

}  // namespace qeccore
