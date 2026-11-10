#pragma once
// ENGINE_VERSION — the engine identity of every sampled byte stream. Bump it whenever an engine
// change could alter the constructed reference or a byte stream. It participates in the Python
// refcache key (xtim/reference.py).
namespace qeccore {

constexpr const char* kEngineVersion = "xtim-engine-17";
// engine-17 (lean twirl arc N1, OWNER-SIGNED 2026-11-07 "Signing b", supersedes the S-c signature): the NOISE stream is a
//   mechanism table (level 0: alternatives whose record is all zero pruned, identical-record alternatives of a channel
//   merged, probabilities added; no cross-channel convolution; groups of equal P in ascending P bits, channel order
//   inside a group) walked by ONE geometric-skip chain per (seed, aligned 4096-shot block, group) over (mechanisms x
//   shots), mechanism-major; a fired cell's outcome is an alias draw on the next word (qeccore/noise_chain.hpp). Recording
//   runs and every run that reads P_f (framed / term-recorded) draw the FULL channel table (option b: exact P_rec), so a
//   recording run's bytes differ from sample()'s for the same seed. Every gate-noise circuit's sampled bytes, postselection
//   counts and route / plan counters move (Clifford and T circuits); circuits without gate noise, DEM / exact / reference
//   legs are byte-identical. Evidence: lean/N1/equivalence.md.
//   + lean N2 (OWNER-SIGNED 2026-11-07, engine-17 unreleased, no bump): each (seed, group, block) chain owns a disjoint
//   counter range of its tag's keyed sequence (draw c = mix(key + (b·R + r0_g + c + 1)·γ), R_g = 2·M_g·4096 + 1), so no two
//   chains share a word (N1's hashed per-chain bases could share a stretch of the one γ-cycle); every NOISE / NOISE_FULL
//   byte moves (same law); a block past (b + 1)·R ≤ 2^64 is refused by name. COIN / BORN / RF, DEM / exact / reference
//   unchanged. Evidence: lean/N2/REPORT.md.
// engine-16 (lean twirl arc D1-S3, OWNER-SIGNED 2026-11-06 "Sign CH/PPR move, then delete", the rotation loop not (G)):
//   the rotation picture covers any pairwise-commuting term axes, so CH / PPR circuits and every shot with a PPR-retry
//   alternative are served by the ROTATION LOOP (keys = s, the pivot gauge g, the Born direction gauge δ: the same laws).
//   CH / PPR circuits' sample() / sample_barrier() bytes, postselection counts, plan counters and `_ctor_digest` move;
//   Clifford circuits and the D1-S2 rotation-loop circuits are byte-identical. Evidence: lean/D1/S3 (equiv/).
// engine-15 (lean twirl arc D1-S2, OWNER-SIGNED 2026-11-05 "S1+S2, deletions later": the stream change, its re-pins and
//   this bump): the ROTATION SHOT LOOP serves every unframed shot of a diagonal-class circuit with π/8 rotations — a shot
//   is keyed by its rotation word s = ⊕ S_f and served by a sparse t-bit plan; the coin columns carry the pivot gauge
//   y' = y ⊕ A·g(s) and the Born directions the sign gauge δ (the same keyed COIN / BORN words: the same laws); records are
//   kind-2 contents (2, s). T circuits' sample() / sample_barrier() bytes, postselection `early` counts and plan
//   counters move; Clifford circuits, PPR shots and CH / PPR circuits are byte-identical; `_ctor_digest` hashes the
//   rotation model / records and the extras' rows (alt_off widened to size_t). Evidence: lean/D1/S2/equivalence.md.
// engine-14 (lean twirl arc L5, owner-signed 2026-11-04 "L5 signed." + bump 2026-11-05 "Sure bump it"): readout flips are
//   skip-sampled — equal-p noisy-measurement slots form one chain of inverse-CDF geometric gaps on the shot's RF words
//   (exact Bernoulli(p) per slot). The RF stream moves on every circuit with noisy measurements; circuits without them
//   are byte-identical; laws unchanged. Evidence: equivalence report of the L5 stage.
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
