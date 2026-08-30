# Changelog

All notable changes to `xtim` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.1] - 2026-08-30

Maintenance release. No behavioural change to sampling, the port, or any
numerical result: the public decision/detector streams are byte-identical to
3.0.0 (verified against the downstream consumer's seven pinned streams).

### Fixed

- **Public type annotations now resolve.** `xtim.fidelity_helpers` annotated its
  public signatures — `extract_frame`, `fidelity_from_logicals` and the helpers
  they delegate to — with `"xtim.Circuit"` and `"PostselectedLER"`, names the
  module never imported. Harmless under postponed annotations, but
  `typing.get_type_hints()` raised, and IDEs and documentation tooling could not
  resolve the types on a package meant to be read. All 13 callables in that
  module now resolve.

- **Release verification no longer skips its heaviest legs.** Two tests
  resolved their example circuits relative to their own file rather than the
  installed package, so `RELEASING.md`'s "run the suite against the installed
  package" step silently skipped them — including the `cultivation_d5`
  port-groups workload. Against the 3.0.1 wheel: 153 passed / 11 skipped →
  156 passed / 8 skipped.

### Changed (internal)

- Dead imports removed (`xtim.diagnose`, `xtim.collect`) and a dead local in the
  twirl record-scatter loop that survived only in a comment. `ruff --select F`
  (pyflakes rules) is clean across `xtim/`, `tests/` and `examples/`.
- The three private `_bare_state_of` / `_input_state_of` /
  `_project_bare_onto_syndrome` re-exports in `xtim/__init__.py` are retained
  deliberately (pre-2.x spellings, kept working) and marked as such rather than
  removed — dropping a name from a published package is a breaking change.

## [3.0.0] - 2026-08-29

**The declared-consumption port ships.** `2.7.0` said "v3 will simplify the API
(a declared-consumption segment surface currently in development replaces some
of today's lower-level sampler plumbing)" — this is that release. `xtim.port`
becomes public API: you declare once what you will read, the port compiles once,
routes to the cheapest engine surface that serves the declaration, and computes
nothing undeclared. Everything else in the `2.x` API is unchanged, streams
included: `ENGINE_VERSION` stays `xtim-engine-9`, `sample()` / `sample_barrier()`
byte streams are unmoved, and no consumer `.ref` cache invalidates.

### BREAKING

Two `xtim.port` parameters are **removed**. Neither was ever in a published
release (the port module itself was held back from `2.7.0`), so this breaks only
consumers built against the unpublished `2.8.0`-class development line:

- **`Segment.run(..., frame_in=…)`** and **`Result.frame_out`** — removed.
  `run(shots, seed)` is now the entire signature and `Result` carries only the
  declared record outputs. WHY: the seam contract was ratified in its strongest
  form — **only classical data ever crosses a segment seam, and the engine holds
  no state between calls**. The Pauli-frame crossing is per-shot classical data
  held and applied *by the consumer* as record post-processing (frame-correcting
  records by anticommutation, the boundary partial-syndrome channel for non-CSS
  seams, the charge-matched frame advance, the terminal syndrome XOR). There is
  therefore nothing an engine-side injection surface could do, and a live
  `frame_in` would invite exactly the engine-held-state coupling the contract
  forbids. `frame_in` had been a loud `NotImplementedError` and `frame_out` an
  always-`None` attribute; both are now simply gone, so `Segment.run` is a
  stateless draw *by construction*. Callers who genuinely need an engine-side
  input Pauli use the raw sampler's `input_pauli` on `xtim.twirl`, outside the
  port. The seam contract is stated positively in `xtim/port.py`'s docstring.
- **`Segment.run(..., input_bits=…)`** — removed, with its per-shot row-gather
  pattern-enumeration path (per-pattern full-shots draws, the distinct-pattern
  cap, the draw memo, and the merged pattern-major group surface). It had no
  consumers. Consequence: a circuit carrying `IF` blocks driven by external
  `INPUT_BITS` now **refuses at `port.compile`** rather than silently serving the
  all-zeros resolved branch; the refusal names both the circuit property and the
  escape. Resolve the branches yourself (`xtim._xtim.resolve_branches_text`) and
  compile each resolved text into its own `Segment`, or use the unchanged raw
  `xtim.twirl.PartitionedTwirlSampler.sample(decision_bits=…)` surface.

### Added

- **`xtim.port` — the declared-consumption segment surface** (public debut; also
  exported top-level as `port`, `Consume`, `compile_segment`, `Segment`,
  `Result`). Two nouns, two verbs:

  ```python
  from xtim.port import Consume, compile as compile_segment
  seg = compile_segment(circuit_text, Consume(dets=True, decisions=True))
  res = seg.run(shots=200_000, seed=42)
  ```

  - **Declaration-driven routing.** `dets`/`decisions` alone route to the bare
    `TwirlSampler.sample()` path; `groups`/`residual_law` route to the record
    surface. Declaring nothing is a loud `ValueError`. Nothing undeclared is
    computed, stored, serialized or surfaced — a groups-only Segment never
    performs a σ-law fetch (spy-gated in the tests).
  - **Compile once, run many.** `compile()` memoizes per
    `(text, consume, plan_cache)` and returns the identical Segment object on a
    hit; `run(shots, seed)` reseeds the engine **in place** (no reconstruction),
    so a seeded draw costs only the sampling.
  - **Group-indexed results.** `Consume(groups=True)` adds `group_id` per shot
    plus per-GROUP arrays — `group_keys`, `group_first_shot`, `group_sigmas`,
    `sig_bits`, representative `group_dets`/`group_decisions`, and lazily parsed
    `group_coins`/`group_plans` — one row per distinct residual class instead of
    per shot. Arrays are handed out as views on the engine's own buffers where
    the layout allows (no per-shot Python objects, no `.tolist()`).
  - **σ-law read model.** `Consume(residual_law=True)` adds `Segment.sigma_law`,
    `born_dec`, `output_wires`, `sig_bits`/`gw` and the memoized
    `plan_structure(plan_key)` — pure functions of the compiled circuit,
    computed once at compile time.
  - **Compact record store (internal).** The record route samples with the
    engine's compact sink: per shot it stores an interned dense plan id plus
    residual-prefix support words, σ words and coins instead of serializing
    ~600 B of plan bytes, and materializes the legacy padded key bytes once per
    GROUP at read time. Contract, oracle-gated: the group partition, the
    first-occurrence order and every legacy key byte are identical to the
    historical `record_groups()` output. Old callers (`compact=False`, the
    default) are byte- and cost-identical.
  - **Plan disk cache**, as a compile option: `compile(..., plan_cache=True)`
    for the automatic path (`$XTIM_TWIRL_CACHE_DIR` or `~/.cache/xtim/twirl/`,
    one `.twpl` file per structural circuit signature) or an explicit path.
    Location/size/eviction/byte-stability policy is documented in the module
    docstring — notably, a pre-warmed cache does not move record-route bytes,
    and stale or corrupt files load nothing and are silently rebuilt.
  - **Refusals name both sides.** Every refusal states the circuit property AND
    the colliding declaration: CH/PPR (non-diagonal) circuits refuse
    `groups`/`residual_law` (state retention is diagonal-only — physically there
    is no Pauli residual to read) but still serve `dets`/`decisions`; a circuit
    with zero record channels refuses with the concrete fix (declare the
    feedback MPPs as `DECISION(k)`); IF-driven circuits refuse as above.
  - **Immutability.** `Consume` is frozen; `Segment` and `Result` are
    `__slots__` objects that reject mutation. Undeclared fields raise
    `AttributeError` naming the declaration that would provide them.

- **THE SEAM CONTRACT** — stated in `xtim/port.py`'s module docstring: what
  crosses a segment seam (classical data only, consumer-held) and why the port
  therefore has no frame surface.

### Performance

- **Born-decision fast path.** Born-decision circuits now run the fast abelian
  σ-path (`need_state` decoupled from `has_born_dec_`) and feed the existing
  Born-probability memo instead of taking a per-shot canonicalize/collapse
  detour. Noisy H+T: **5.82 → 5.27 µs/shot** at 40k shots/call and
  **2.33 → 2.08 µs/shot** at 200k. The fast-σ == structural-σ self-check stays
  active; non-Born and exact-tier streams are byte-identical, and for a
  `kappa=0` diagonal plan the Born DECISION stream itself is byte-identical to
  the pre-change structural path (collapsed amplitudes are invariant to the fair
  coins) — verified over 40k H+T + 5-Born-qubit shots at 5σ with exact `m_out`
  weights.
- **Record surface** (cultivation_d5, 100k shots/call, plan-warm):
  `sample_barrier` **5.42 → 4.54 µs/shot**, `record_groups`
  **3.84 → 1.11 µs/shot**, end-to-end **9.26 → 5.65 µs/shot** — the compact
  record store plus the matrix (never list-of-lists) key surface.
- **In-place `set_seed` carries forward** (shipped in 2.6.1, unchanged here) —
  the single largest win on the seeded-sampling path and the reason the port's
  compile-once / `run(shots, seed)` loop is viable at all: a seeded call never
  reconstructs the sampler, so a seeded zero-shot call costs ~0.04 ms instead of
  the ~180 ms full rebuild that preceded it.

### Performance (release verification)

Re-measured for this release on the release tree (linux x86_64, CPython 3.12,
48 cores, **host load ~2.1–2.5** — a shared box, not idle; a quiet host runs
lower). Warm median-of-5, one warmup call per leg, circuit `cultivation_d5`
(42 qubits, 107 detectors), `selfcheck=0` (the fast path — with the
once-per-sampler window a default sampler is in the same class from its second
call on). A second pass at load ~3.6 reproduced every leg within 5%.

| leg | result |
|---|---|
| record `sample()` 200k shots/call, same-seed warm | 0.59 µs/shot |
| record `sample()` 2k shots/call, same-seed warm | 0.57 µs/shot |
| record `sample()` 200k shots/call, fresh-seed warm (plan-law tax) | 3.7 µs/shot |
| `sample_barrier()` 200k shots/call, same-seed warm | 6.0 µs/shot |
| seeded zero-shot `sample(0, seed=…)` call | 0.031 ms |
| **port** `Segment.run(Consume(dets, decisions))` 200k/call, same-seed warm | 0.61 µs/shot |
| **port** `Segment.run(Consume(dets, decisions))` 2k/call, same-seed warm | 0.58 µs/shot |
| **port** `Segment.run(… groups=True)` 100k/call, same-seed warm | 5.9 µs/shot |

Every leg is at or better than the 2.7.0 release table measured under heavier
load, and the port costs ~0.02 µs/shot over the raw sampler it wraps — the
declaration is resolved at compile time, not per call. The 0.031 ms seeded
zero-shot call is the standing confirmation that the ~180 ms per-call rebuild
tax is gone.

### Compatibility

- `ENGINE_VERSION` unchanged (`xtim-engine-9`); `REF_FORMAT_VERSION` unchanged
  (4). No `.ref` cache invalidates.
- `sample()` / `sample_barrier()` byte streams are unmoved from 2.7.0 for every
  non-Born-decision circuit; the Born-decision fast path is byte-identical on
  `kappa=0` diagonal plans and law-identical (5σ-verified) elsewhere — it was
  re-pinned once when it landed.
- The selfcheck oracle window still runs once per sampler lifetime (2.7.0
  behavior), never re-arming on `set_seed`, and remains stream-neutral.
- No removals outside `xtim.port` (see BREAKING).

## [2.8.0] - 2026-08-11

*Local tag only — never published. Its contents ship in 3.0.0 above, minus
the two dead port parameters 3.0.0 removes.*

**The frame-forwarded (FF) arc's engine-side surface.** `xtim.port` grows from a v3
skeleton into THE SEAM CONTRACT the adaptq FF tier is built against, plus a
Born-decision fast-path speedup, selfcheck-once, and a `PAULI_EXPECTATION`
round-trip fix. `sample_barrier` / `sample` streams are byte-identical throughout
(self-check-gated where applicable); `ENGINE_VERSION` is unchanged
(`xtim-engine-9`) — no pinned decision-stream hash moves with this release.
adaptq's frame-forwarded segmented-simulation arc (`feat/frame-forwarded-sim`,
T1-T8) is the consumer of everything in this release.

### Added
- **`xtim.port` v3** (`Consume`, `compile()`, `Segment`, `Result`) — declared-
  consumption segment surface: dets/decisions-only routes to the bare
  `TwirlSampler.sample()` (in-place reseeding, no reconstruction);
  `groups`/`residual_law` route to the record surface (per-shot `group_id` +
  per-group arrays: keys/first_shot/sigmas/dets/decisions/coins/plans), backed
  by a compact internal record representation (interned plan id + prefix/σ
  words + coins in place of ~600 B serialized plan-key bytes) that is byte-
  identical to the legacy serialized-key groups it replaces (port-v3 T1/T2).
- **THE SEAM CONTRACT** (`xtim/port.py` module docstring, `Segment.run`) —
  ratified statement that only classical data ever crosses an adaptq segment
  seam (the per-shot Pauli frame, the boundary partial-syndrome channel for
  non-CSS seams, the charge-matched T_phys frame advance, Born/driven-bit
  relabels); the engine holds no state between calls. `frame_in` remains a
  loud `NotImplementedError` and `frame_out` remains always `None` by design —
  under the ratified contract no engine-side frame-injection surface exists to
  implement (FF T5, docstring-only; no code-path change).
- **`Segment.run` driven `input_bits`** — per-shot driven bits for IF-driven
  circuits (row-gather semantics; stream-neutral on IF-free circuits) (port-v3
  T5).
- **Selfcheck-once** — the oracle selfcheck window (default 2000 shots: the
  first N fired shots re-derived through the full σ-path and matched exactly)
  now runs once per sampler lifetime instead of re-arming on every seeded
  `set_seed()` call. Stream-neutral: a spent sampler is exactly a
  `selfcheck=0` sampler, byte-identical to the window-on path. Two-call
  cultivation-d5 timing: call1 ~80 ms (window + cold plan laws) / call2
  ~1.7 ms (was ~53 ms pre-fix).

### Performance
- **Born-decision fast-path**: Born-decision circuits now run the fast abelian
  σ-path (`need_state` decoupled from `has_born_dec_`) and feed the existing
  Born-probability memo instead of a per-shot canonicalize/collapse detour.
  Noisy H+T: 5.82 → 5.27 µs/shot at 40k shots/call, 2.33 → 2.08 µs/shot at
  200k. Byte-identity maintained: the fast-σ == structural-σ self-check stays
  active; non-Born / exact-tier streams are byte-identical, and for a
  kappa=0 diagonal plan the Born DECISION stream itself is byte-identical to
  the pre-change structural path (collapsed amplitudes are invariant to the
  fair coins) — verified 40k H+T + 5-Born-qubit shots, 5σ, m_out weights
  exact.
- Port groups/record-surface (port-v3 T2, cultivation_d5, 100k shots/call,
  plan-warm): `sample_barrier` 5.42 → 4.54 µs/shot, `record_groups`
  3.84 → 1.11 µs/shot, end-to-end 9.26 → 5.65 µs/shot.

### Fixed
- **`PAULI_EXPECTATION` serializer round-trip** — `resolve_branches_text`
  emitted the space form (`PAULI_EXPECTATION 0 X0`) but the parser requires
  the paren form (`PAULI_EXPECTATION(0) X0`); any payload-carrying circuit
  using `resolve_branches_text` (adaptq's exact tier) failed to re-parse its
  own resolved output.

### Notes
- `ENGINE_VERSION` (`xtim-engine-9`) is unchanged across this entire release —
  no consumer `.ref` cache invalidates.
- This is the engine state the adaptq frame-forwarded (FF) segmented-
  simulation arc is built and byte-gated against.

## [2.7.0] - 2026-08-03

**Public engine cut** — consolidates the 2.6.x line for release: the latest
engine, correctness- and speed-verified, with no orchestration layer. `v2.x`
will be maintained until `v3.0`; **v3 will simplify the API** (a declared-
consumption segment surface currently in development replaces some of today's
lower-level sampler plumbing). If you are starting fresh, the `v2.x` API below
is stable and supported.

### Highlights

- **In-place `set_seed`** — seeded `sample()` / `sample_barrier()` calls no
  longer reconstruct the sampler; the ~180 ms per-call rebuild tax is removed.
  Streams after `set_seed(s)` are byte-identical to a freshly constructed
  sampler at seed `s` (zero tolerance; Python + C++ seed-parity oracles).
  Cultivation-d5 at 2000 shots/call: ~90 µs/shot → ~9–11 µs/shot warm.
- **Exact-residual read accessors** (all additive; from 2.6.0):
  `BarrierBuffer.sigmas()` / `sig_bits` (bulk certified-generator syndrome
  rows), `FramedSuperposition.port_sigma_law` (combination-tracked port
  decomposition), `certified_symplectic`, `BarrierBuffer.plan_structure`,
  `born_dec`. Residual signs are resolvable from exact engine reads; state
  materialization is demoted to a test oracle.
- **Serializer round-trip fix** — `PAULI_EXPECTATION` instructions survive
  `resolve_branches` text round-trips (emitted as `PAULI_EXPECTATION(label)`,
  matching the parser).
- **Selfcheck window runs once per sampler** (default-behavior change) — the
  `selfcheck` oracle window (default 2000: the first N fired shots also run
  the full sigma path and must match exactly) now runs ONCE per sampler
  lifetime, on the first `sample()`/`sample_barrier()` call after
  compilation, and never re-arms on later calls. Previously the in-place
  `set_seed` reset the window's gating counters, so EVERY seeded call
  silently re-paid the slow oracle window. The window is a stream-neutral
  verification overlay: sampled bytes are byte-identical with the window on
  or off (oracle: `tests/test_selfcheck_once.py`; zero tolerance).
  Recompiling re-arms it. Default-settings two-call behavior on cultivation
  d5, 2000 shots/call, same seed: first call ~80 ms (oracle window + cold
  plan laws), second call ~1.7 ms (~0.9 µs/shot — the fast path); pre-fix
  the second call re-paid the window at ~53 ms (~26 µs/shot).

### Removed

- The in-flight v3 `xtim.port` module (and its top-level `Consume` /
  `compile_segment` / `Segment` / `Result` exports) is not part of this
  release; it is mid-flight API and ships at v3.0.

### Internal / experimental

- The compact-record C++ machinery (plan interning + per-shot prefix words in
  the record sink) remains in the engine. It is engine-internal, default-off
  (`compact=False`), and inert without the v3 port surface; it is a bijective
  re-encoding of the legacy plan bytes with byte-identical group keys.

### Compatibility

- `sample_barrier` streams are byte-identical to 2.5.0; `ENGINE_VERSION` is
  unchanged. No API removals other than the unreleased v3 port surface.

### Performance (release verification)

Measured on the installed 2.7.0 wheel in a clean env (linux x86_64, python
3.12, warm median-of-5, host load ~3.0 — a LOADED host; quiet-host numbers
run lower). Circuit: `cultivation_d5` (42 qubits, 107 detectors).
`selfcheck=0` legs (the fast path). With the once-per-sampler selfcheck
window (above), a DEFAULT sampler is in the same speed class from its second
call on — only the first call after compilation pays the oracle window:

| leg | result |
|---|---|
| record `sample()` 200k shots/call, same-seed warm | 0.72 µs/shot |
| record `sample()` 2k shots/call, same-seed warm | 0.68 µs/shot (1.35 ms/call) ¹ |
| record `sample()` 200k shots/call, fresh-seed warm (plan-law tax) | 4.2 µs/shot |
| `sample_barrier()` 200k shots/call, same-seed warm | 6.1 µs/shot |
| seeded zero-shot `sample(0, seed=…)` call | 0.04 ms |
| bare detector `sample()` 200k shots/call warm | 0.64 µs/shot |
| DEFAULT sampler (`selfcheck=2000`), 2k shots/call, same seed — call 1 / call 2 | ~80 ms / ~1.7 ms (~0.9 µs/shot) ² |

¹ Quiet-host measurement at `selfcheck=0` — the engine's best case, not the
typical loaded-host figure (the other rows were taken at host load ~3.0).

² Once-per-sampler window (this release): call 1 pays the oracle window plus
cold plan laws; call 2 and every later call run the fast path. Pre-fix, every
seeded call re-paid the window: call 2 measured ~53 ms (~26 µs/shot).

The seeded zero-shot call at 0.04 ms confirms the ~180 ms per-call rebuild
tax is gone in the shipped wheel (pre-2.6.1 this call cost a full sampler
reconstruction).

## [2.6.1] - 2026-07-31

**In-place `set_seed` (speed-kill T1/T2)** — seeded `sample()` / `sample_barrier()`
calls no longer reconstruct the `TwirlRecordSampler`.  No API or byte-stream change;
`ENGINE_VERSION` unchanged.

### Added
- **`TwirlRecordSampler::set_seed(uint64_t)`** — reseed the twirl record sampler
  IN PLACE, resetting exactly the seed-dependent state a fresh construction
  initializes (noise-event stream via `DiagErrorSampler::reseed`, per-shot counter-
  RNG base, dedicated obs/fallback/decision streams, rf shot counter, cumulative run
  counters) without reconstructing anything.  Previously every seeded `sample()` /
  `sample_barrier()` call tore down and reconstructed the entire
  `TwirlRecordSampler` — a ~176–184 ms circuit-compile-scale fixed cost on
  cultivation d5 that appeared as a "~90 µs/shot floor" at 2000 shots/call.
  Returns `false` without mutating anything in the one case where in-place reseeding
  is physically incorrect: synthetic-channel mode (unreachable from the Python
  bindings, which always compile circuit channels).  Byte contract: streams emitted
  after `set_seed(s)` are byte-identical to a freshly constructed sampler at `s`
  (zero tolerance; oracle: `tests/test_set_seed_equivalence.py`).
- **`PyTwirlSampler::reseed_in_place(seed)`** — wrapper that routes both seeded
  call sites (`sample`, `sample_barrier`) through `set_seed` + gauge-RNG re-tie
  (identical seed-tie as `rebuild()`), falling back to the retained `rebuild()` with
  a loud `stderr` note on the (unexpected) synthetic-channel path.
- **C++ seed-parity test** (`cpp/tests/test_set_seed_parity.cpp`, REPORT() harness,
  wired via `cpp/CMakeLists.txt`) — asserts `ctor(seed=s)` streams equal
  `ctor(1)+set_seed(s)` streams for `s ∈ {2, 7, 5_000_000_007}` at the C++ API
  level, pinning seed-value-dependent parity that the Python suite structurally
  cannot (Python ctors fix seed=1).
- **`cpp/CMakeLists.txt`** — standalone CMake build for the xtim C++ engine
  (mirrors `qec_library/cpp/CMakeLists.txt` pattern): static `qeccore` library
  linking all `cpp/src/*.cpp`, `qeccore_test` helper macro, `enable_testing()`.

### Performance
- Cultivation d5 warm throughput at 2000 shots/call: was ~97.5–115.8 µs/shot
  (dominated by the ~180 ms per-call rebuild); now **~9–11 µs/shot** (~10–12×
  speedup) — the rebuild tax is eliminated.
- At 200k shots/call the bare `sample()` record is load-sensitive: measured 4.03
  µs/shot (median, load ~3) in the T4 closeout window; 0.70 µs/shot was a T1-era
  minimum that did not reproduce under normal load.  Quiet-host re-measurement
  pending (quiet_repin flow).

## [2.6.0] - 2026-07-31

The **exact-residual engine surface** (the exact-residual-retirement arc,
E1/E2/E3 + T5 step 0). Everything is ADDITIVE: no existing method or signature
changed, `sample_barrier` streams are byte-identical to 2.5.0, and
`ENGINE_VERSION` is unchanged — no pinned decision-stream hash moves with this
release. The new surface is what lets a consumer resolve every residual sign
from exact engine reads (bulk σ + combination vectors) and classify
indefiniteness per plan, with state materialization demoted to a test oracle.

### Added
- **`BarrierBuffer.sigmas()`** — bulk per-shot σ words, `uint64 (shots,
  ceil(sig_bits/64))`, a COPY of the internal word store; bit b of shot i =
  `(row[b>>6] >> (b&63)) & 1`; tail bits above `sig_bits` are zero by
  construction. Companion read-only property **`BarrierBuffer.sig_bits`**
  (= certified-generator count).
- **`FramedSuperposition.port_sigma_law(wires)`** — combination-tracked port
  decomposition beside `port_signature` (existing bytes untouched): per
  surviving port check, in `port_signature`'s exact sorted order, its
  combination vector over the certified generators (packed in the `sigmas()`
  word layout), the reference sign bit, and the carried logical pair rows
  (M·P law, identical to `port_orbit_operators`). Exact law:
  `sign_j(shot) = ref_signs[j] XOR parity(comb[j] AND sigma_shot)`.
- **`FramedSuperposition.certified_symplectic()`** — the certified stabilizer
  generators as full n-qubit symplectic rows + exact phases; row b == σ bit b
  == `port_sigma_law` comb bit b.
- **`BarrierBuffer.plan_structure(plan_key_bytes)`** — a plan key parsed into
  the residual normal form (prefix / a-mask / cz list) plus the plan's
  `ShotLaw` dump (det signs, coin masks, kernel masks/reps); pure exposure of
  the structures `materialize_shot` already computes. The plan-key byte format
  is now documented as a standalone spec (T3 report §3).
- **`BarrierBuffer.born_dec()`** (`TwirlRecordSampler::born_dec_ops`) —
  pure-data exposure of the Born-measured decision operators (LOGICAL/ANTI
  class, declaration order): operator supports/phases, decision indices, and
  inversion bits; raw outcome of op j is `coins(i)[r+kappa+j]`.

### Changed
- **Plan-key parser hardening**: `plan_structure` accepts arbitrary caller
  bytes (unlike `materialize_shot`, which replays sampler-serialized keys
  only), so an out-of-range cz wire index now raises `std::invalid_argument`
  before law construction.

### Notes
- Byte gates held throughout: the consumer's full 6-leg decision-stream byte
  gate (noiseless-Z/Y + noisy-Z, instrument + exact tiers) passes against this
  build with pins unmoved, and the exact-tier streams are unchanged.
- New test files: `tests/test_barrier_sigmas.py`, `tests/test_port_sigma_law.py`
  (incl. corruption probes + self-certifying out-of-span gap accounting), and
  `tests/test_plan_structure_classifier.py` (classifier prototype + the born5
  `born_dec` contract tests); vendored workload data under `tests/data/` with
  provenance headers.

## [2.5.0] - 2026-07-30

Engine-generation sync: the standalone package now ships the **record / barrier /
decision sampler** engine (the decoder-feedback arc), replacing the 2.4.0
detector/observable fast sampler. This is the engine surface consumed by adaptq
(M2b/M3 validation suite); the package is a wholesale sync of exactly that
validated engine, with the standalone packaging (setup.py/pyproject/MANIFEST/
smoke_test/examples) preserved.

### Added
- **Record-sampler engine API** on `xtim._xtim.TwirlSampler`:
  `sample_barrier(shots, seed)` (returns a `BarrierBuffer`), `output_wires()`,
  and the `num_decisions` property — the DECISION-carrying record path absent in
  2.4.0. `sample(shots, seed, input_pauli)` accepts a per-call input Pauli.
- **`BarrierBuffer`** methods: `dets()`, `decisions()`,
  `record_groups() -> (gids, first_occ, keys)`, and `state(i)` (per-shot
  re-materialized `FramedSuperposition`, for reproducible twirl-split copies).
- **Port-contract / branch-resolution engine functions** (new C++ TUs
  `port_contract.cpp`, `resolve_branches.cpp`): `input_qubits(...)`,
  `bits_total_width(...)`, `resolve_branches_text(...)`, and
  `carried_symplectic_action(...)`.
- **State-object methods** on `FramedSuperposition`: `port_orbit_operators(wires)`,
  `pauli_expectation_xz(...)`, `apply_clifford(...)`, `measure_pauli(basis, q, u)`
  (Born sign-sector projection) — the port-projection surface for twirl-split.
- **Full `xtim.twirl` engine API** (was a 4.3 KB stub in 2.4.0, now the complete
  decision-carrying compiler): `compile_twirl_sampler` (with disk cache +
  `skip_refused_observables`), `compile_twirl_sampler_from_state`,
  `PartitionedTwirlSampler` (IF-branch blocks), and the state constructors
  `bare_state_of` / `input_state_of` (+ the underscore-prefixed variants and
  `_project_bare_onto_syndrome`). These are re-exported from the top-level
  `xtim` package.

### Changed
- The twirl sampler generation changed 2.4.0 → 2.5.0 (record/barrier/decision
  path). The engine identity string is unchanged (`ENGINE_VERSION =
  "xtim-engine-9"`), so the decision-stream bytes are **identical** to the engine
  adaptq pinned against: adaptq's full byte gate (6 legs: noiseless-Z/Y and
  noisy-Z, each instrument + exact) passes against this build, sha256-for-sha256.
- Native vs portable wheels: the `-march=native` (source/dev) and
  `QECCORE_PORTABLE` (manylinux wheel) builds produce the **same** decision-stream
  hashes — the byte gate passes identically against both. No wheels-vs-native
  stream divergence on the validated paths; adaptq's pins hold for either
  install stream.

### Notes
- `ENGINE_VERSION` is unchanged, so this bump does not by itself invalidate any
  pinned byte stream; it does refresh consumer disk-extraction caches that fold
  the package version into their key (a one-time, desired cold rebuild).
- Two reserved binding names (`dem_column_layout`, `if_driving_global_offsets`)
  remain exported (zero consumers); kept to hold the binding byte-identical to
  the validated engine rather than hand-edit the surface.

_Engine surface consumed by adaptq (M2b/M3 validation suite)._

## [2.4.0] - 2026-07-27

### Changed
- Default engine changed from `"exact"` to `"auto"` (auto = twirl when eligible, exact
  otherwise; `return_expectations`/`return_measurements` transparently served by exact under
  auto; `QEC_NO_TWIRL` kill switch unchanged). Explicit `engine="twirl"` still refuses those
  sample-time arguments loudly.
- Twirl-engine sampling streams changed (per-shot derived coins — routing-independent: repeated
  same-seed `sample()` calls are now byte-identical between cold and warm runs; distributions
  5-sigma-gated vs exact and unchanged in law).

### Performance
- Plan caches (`TwirlPlanCache`/`PprPlanCache`) and fast-route index (`SharedFastIndex`) persist
  across `sample()` calls — warm cultivation_d5 ~8.5 µs/shot (cold) → ~1.85 µs/shot (warm,
  4.6× speedup measured).
- Devirtualized detector-pack sink (`DetsPackSink`) replaces `std::function` shot sink — word-
  level XOR copy for identity-contiguous circuits.
- Transposed readout-flip application: column-form ctz XOR loop (O(fired × CW) vs O(nchan × RFW)).
- Fixed-point per-shot rf coins (one integer compare per slot vs double draw+branch).
- Selfcheck oracle off by default; set `QEC_TW_SELFCHECK=<n>` to enable the oracle window.

### Added
- `XtimPerformanceWarning` — emitted (once per reason per process) when `engine="auto"` falls
  back to exact. Importable from `xtim`.
- `channel_report()["sink_kind"]` — `"dets_pack"`, `"barrier"`, or `"null"`.
- `channel_one_counts()` on the compiled twirl sampler — per-channel fired counts (all-zero on
  the default `selfcheck=0` fast path; active with `QEC_TW_COUNT=1`).

### Removed
- Dead `std::function` `ShotSink` machinery and `set_shot_sink()` (superseded by `DetsPackSink`).
- Dead `coll_gen`/`cu` RNG members (hot path uses per-shot `ShotRng`).

### Fixed
- **`engine="twirl"` / `engine="auto"` produced incorrect detector statistics on circuits
  with Y-basis measurements** (e.g. `color_code:memory_xyz`) — present in the published
  2.3.0 opt-in twirl engine (measured at its exact source commit), latent there only
  because the default engine was `exact`. Root cause: the noiseless reference bits
  (`chan_ref`) were derived from the record-product's Pauli phase, which accumulates a
  spurious −1 for even products of Y-measurements
  whose record-product contains an even number of MY measurements (Y = iXZ → phase 2 = −1,
  giving `ref=1`). Every shot in a noiseless circuit then fired those detectors. Fix:
  `chan_ref` is zeroed for all detector/observable channels after the readout-flip layer build,
  because for self-contained circuits the bare state IS the noiseless reference. MX/MZ circuits
  were unaffected (X and Z Paulis carry phase 0 or 2 but always cancel correctly in products).
- `XTIM_QUIET=1` now suppresses all three `[twirl_records]` informational banners
  (circuit-channels, disk-LOADED, disk-SAVED). The disk-LOADED banner was previously
  unguarded.
- `QEC_TW_PROF=1` now emits a complete segment ledger: `coins/rf/post/sink/draw/prolog/nrf/probe/wall`
  alongside `key/find/emit/pair` — the accumulator fast path probes were missing from the
  ported build, leaving ~0.55 µs/shot unattributed on the warm cultivation_d5 pass.
- `__version__` fallback string updated to `"2.4.0+dev"` (was `"2.3.0+dev"`).

## [2.3.0] - 2026-07-16

A fast detector/observable sampling engine — the stabilizer-twirl record sampler —
selectable per compile, **opt-in this release** (the default engine is unchanged).
No breaking changes; the exact engine, `.ref` format, and all existing sampling
streams are byte-identical to 2.2.0.

### Added
- **`Circuit.compile_detector_sampler(..., engine="auto"|"exact"|"twirl")`** — a new
  fast engine that samples the SAME deterministic detector and observable statistics
  as the exact engine, **9–28× faster** on the bundled protocols (measured, quiet
  machine): cultivation_d5 18×, code_switching 25×, distillation 28×, the CH-class
  circuits 9×. It compiles each noise pattern's end-of-circuit residual into a cached
  "plan" and emits record bits from closed-form channel algebra instead of evolving
  the state.
  - `engine="exact"` (the default) is the legacy engine, unchanged.
  - `engine="twirl"` selects the fast engine explicitly.
  - `engine="auto"` uses the fast engine when the circuit is eligible and silently
    falls back to exact otherwise — never an error, never a silent change of result.
  - The returned sampler exposes `engine_report()` (which engine, and why).
  - `QEC_NO_TWIRL=1` forces the exact engine everywhere.
- **Exactness contract** (5σ-gated against the exact engine on every bundled circuit,
  and against Stim): every deterministic detector's frequency — jointly, so
  **post-selected and decoded logical error rates are exact** — and logical-observable
  statistics including their joint with the detectors. Noisy measurements (`M(p)`),
  deterministic record inverts, and CH-class (controlled-H) circuits are all handled.
- **`xtim.load_example("...")`** now includes the twirl fast-sampling walkthrough
  (`examples/twirl_fast_sampling.py`).

### Notes
- **Declared semantics.** Gauge detectors (rare) are declared fair coins under
  `engine="twirl"`; a circuit containing them is routed to the exact engine under
  `engine="auto"` rather than sampled with declared semantics silently. Observables
  whose record operator anticommutes with the code (e.g. some byproduct-frame
  readouts) have no exact fast-engine channel and are routed to exact under `auto`.
- **Byte streams.** The fast engine's raw per-shot bytes differ from the exact engine
  for the same seed (the distributions are what match, and are gated); use
  `engine="exact"` when you need byte-for-byte reproducibility against older runs.
- Optional on-disk plan cache (`disk_cache=`) makes a `p`-sweep or rerun compile the
  plans once.

## [2.2.0] - 2026-07-10

Large-circuit compile speed and a hardened reference cache. No API change; all
sampling streams are byte-identical to 2.1.0; `.ref` format v4 and the engine
identity are unchanged.

### Changed (performance — no API, behavior, or sampling-stream change)
- **Large-Clifford-circuit compile is 15–25× faster.** The reference-state compile
  and error-propagation table paid dense-n² costs per noise alternative and per
  stabilizer generator on wide deferred circuits; a stock d=11, r=7 surface-code
  memory now compiles in ~9 s (was ~4 min), d=7 in ~1 s (was ~11 s). All sampling
  streams are byte-identical; protocol-circuit compile and per-shot speed unchanged.
- **Noise-value-blind reference cache.** The `.xtim_cache` key now masks noise
  probability values (which the reference provably cannot depend on), so a
  `Task(p=...)`/`scale_noise` sweep compiles the reference once and every
  p-point is a cache hit. Noise structure/placement stays in the key. (Existing
  cache entries under the old raw-text key are simply orphaned — delete the
  cache dir to reclaim the space.)

### Fixed (hardening from a full-engine correctness audit; none released)
- **Stack corruption compiling circuits wider than 4096 qubits**: the CX error-
  conjugation sized a stack snapshot buffer from the (new) lazy CZ-layer's word
  count and masked two entries unconditionally — an out-of-bounds write for
  targets ≥ qubit 4096. Caught by a new wide-circuit regression test; the
  snapshot is now skipped entirely for the lazy (couplings-free) form.
- **Reference-cache frame guard**: every cache entry now stores a signature of
  the normalized+deferred circuit structure (noise values excluded) and every
  hit re-derives it from the caller's text — a cached reference compiled for a
  circuit that normalizes to a different terminal frame is rejected and falls
  back, closing the frame-divergence class for detector/expectation-bearing
  circuits where the cheap-verify invariants alone are weak.
- **Rank guard in `PreparedAffine::prepare`**: the full-column-rank precondition
  the fast phase pin relies on is now enforced (loud throw) instead of assumed.
- **`collect` beta memo** keys on the deferred-frame signature (was: noise-
  stripped text), so tasks differing in noise *placement* each get their own
  frame-adequacy check; p-sweep sharing unchanged.
- **Noise-line recognition** in `_strip_noise`/the cache key now matches the
  parser's tokenizer (lowercase names, `[tag]` suffixes, tab separators).
- New test batteries: packed-matrix differential fuzz (word-level compaction vs
  per-bit reference; lazy vs materialized CZ-layer equivalence; trailing-bit
  invariants), and an xtim-vs-Stim distribution-parity harness on surface-code
  memory circuits (marginals, pairwise correlations, random parity masks, 5.5σ).

### Changed (docs only)
- **README rewritten as a lean landing page** (~8× shorter): pitch, install,
  a sixty-second worked example with real output, a measured 15-to-1
  distillation curve (new `docs/assets/distillation_curve.png`), an
  "xtim or Stim?" table, and a links table. Nothing was deleted — the
  displaced material moved to two new pages, `docs/xtim_scoring.md` (the full
  `diagnose()` walkthrough + the four fidelity-arithmetic pitfalls) and
  `docs/xtim_practical_notes.md` (CLI, caching, seeding, programmatic
  building, scope in practice), and the protocol throughput table moved to
  `examples/README.md`.

## [2.1.0] - 2026-07-07

Controlled-Hadamard support, end to end. Existing circuits are unaffected — every
diagonal-class (T/CS/CCZ) sampling stream is byte-identical to 2.0.x, `.ref` format
stays at v4, engine identity stays `xtim-engine-9`.

### Added
- **Controlled-Hadamard (`CH`/`CX_H`) is now in the simulable class** — compile,
  error propagation, and sampling, including sign-exact `PAULI_EXPECTATION` on
  error-carrying shots. Internally each error is propagated past non-diagonal
  third-level gates as a Pauli times a list of commuting π/4 rotations
  (Pauli-product-rotation form) and recompiled into a Clifford tableau for the
  shot loop. The acceptance criterion is unified and unchanged in spirit: a
  circuit is accepted iff its folded π/8 axes mutually commute (generalized
  T-depth one); a non-commuting fold rejects loudly, never silently.
- **H-eigenstate cultivation ships as a bundled example**:
  `xtim.load_example("ch_cultivation")` (3 rounds of logical-Hadamard
  measurement via CH, χ=2), plus a runnable walkthrough
  `examples/h_cultivation_walkthrough.py` (asserts deterministic detectors and
  fidelity, and shows the alignment rules — it is covered by the packaged tests).
- **`diagnose()` and expectation frames now serve CH circuits** (DEM export
  still intentionally rejects them — post-selection/diagnose only, see
  `docs/xtim_dem_reject.md`).
- **Docs**: a reader-journey index (`docs/README.md`) and a precise statement of
  the simulable class (`docs/xtim_simulable_class.md`); README gains CH in the
  dialect table and a corrected class-reject example. The class-reject error
  hint now explains the alignment condition (CH itself is supported; a fold
  only rejects when π/8 axes fail to commute).

### Fixed
- **Wide-circuit compile crash**: circuits deferring to ≳2100 effective qubits
  (e.g. a stock d=11, r≥7 surface-code memory) crashed with a spurious
  "wrong X/Z support" error while pinning stabilizer-generator phases — the
  amplitude ratio was materialized as a double, whose 2^(−k/2) modulus
  underflows. The ratio is now taken exactly in the engine's exact-phase
  representation; such circuits compile and sample (χ=1) with all existing
  streams byte-identical.
- **README speed guidance corrected by measurement**: the "~10²–10³× slower
  than Stim" figure conflated text-record I/O with engine cost. Measured
  same-circuit, same-core, packed records: Stim is ~1.3× faster on the
  measurement channel and ~2.4–2.9× on the detector channel at d=7–11; the
  real asymmetry is compile time (Stim: milliseconds; xtim's reference-state
  compile: seconds at d=7 to minutes at d=11).

### Changed (build hygiene only, byte-identical)
- Removed dead engine code (`gadgetize_level3` and orphaned residue helpers)
  and stopped compiling dev/research-only sources (`symplectic.*`,
  `stim_io.*`) into the wheel. Leaner build, no functional impact.

## [2.0.1] - 2026-07-05

A packaging fix so `xtim` installs alongside any recent numpy. No API, behavior,
`.ref`-format, or engine change — the Python API and sampling output are identical
to 2.0.0.

### Fixed
- **Installs with numpy ≥ 2.5 again (the runtime `numpy` upper bound is removed).**
  2.0.0 shipped with `numpy<2.5` because the old manylinux2014 build image (GCC
  10.2) could not install/build numpy ≥ 2.5. The Linux wheels now build on
  `manylinux_2_28` (GCC 12), where numpy ships prebuilt wheels, so the dependency
  is back to `numpy>=1.26` (unpinned upper bound). Windows/macOS wheels are
  unchanged.
- **Engine headers are now clean under GCC 12** (explicit `#include <cstddef>` for
  `size_t`, which older GCC/Clang/MSVC pulled in transitively). Byte-neutral —
  codegen and sampling streams are unchanged; verified with a local gcc-12.4
  syntax pass over every translation unit and the full test suite.

### Changed (packaging only)
- **Linux wheels: `manylinux2014` → `manylinux_2_28`.** The glibc floor for the
  Linux wheels rises from 2.17 to 2.28 (RHEL 8 / Ubuntu 18.10+ / any 2019-or-newer
  distro). Wheel performance is unchanged — same portable `-O3` build flags, only
  the base image (and its glibc/GCC) differs.

## [2.0.0] - 2026-07-05

Declared byproduct frames, exact determinism, and a fidelity-scoring toolkit.
`PAULI_EXPECTATION` expectation values are now sign-exact and shot-independent;
`Circuit.postselected_logical_error_rate` scores a post-selected `1 − F` (or an
observable LER) with automatic detector post-selection; and two opt-in helpers
(`extract_frame`, `fidelity_from_logicals`) remove the tedious parts of scoring a
magic state's fidelity. Breaking changes are confined to the per-shot expectation
sign streams and one bundled circuit's detector layout — `.mean()`/LER values are
unaffected.

### Added
- **`PAULI_EXPECTATION(i) <Pauli> rec[-k]…` — declared byproduct Pauli frame.**
  A trailing `rec[-k]` target list (the same `rec[-k]` grammar as
  `OBSERVABLE_INCLUDE`) can now be appended to any `PAULI_EXPECTATION` declaration
  to name the measurement records whose parity fixes the sign of ⟨P⟩. With a
  declared frame the engine folds the record-parity into the per-shot sign, so
  `exps[:, i]` is **sign-constant** in the noiseless case: `|⟨P⟩|` is the
  frame-independent physical magnitude; the SIGN is the convention in the frame
  where the declared records all read 0; and the target ⟨P⟩ is the mean of the
  per-shot signed values.
- **`Circuit.postselected_logical_error_rate(*, p, shots, seed, target_k, p0)`**
  and the `PostselectedLER` result type. Scores a post-selected logical error rate
  — `1 − F` in fidelity mode (summing the declared `PAULI_EXPECTATION` columns) or
  the flipped-observable fraction in observable mode (`OBSERVABLE_INCLUDE`) — with
  the automatic post-selection everyone forgets: it rejects every shot in which a
  deterministic detector fired. The result carries `value`, `sem`, `acceptance`,
  `kept`, `mode`, `upper_bound` (rule-of-three 3/kept when zero errors are
  observed), and `warnings` (partial-support-proxy, no-noise-floor, low-acceptance
  advisories). The name makes the post-selection explicit.
- **`extract_frame(circuit, operator, *, minimal=True)` — byproduct-frame helper.**
  Given a logical Pauli operator string, returns the `rec[-k]` offsets whose parity
  pins its sign, ready to paste into a `PAULI_EXPECTATION` line (`minimal=False`
  gives a spanning full-readout representative). Raises `XtimError` when the sign is
  not record-pinnable; returns `[]` when the sign is already fixed by the bare state.
- **`fidelity_from_logicals(circuit, logicals, *, p, shots, seed, p0)` — true
  full-support fidelity.** Given the k logical generators
  `{0: {"X": "…", "Z": "…"}, …}`, it validates the algebra, generates all 4^k
  logical Pauli products, extracts each product's frame, and scores the true
  full-support `1 − F` (auto post-selected). Unlike scoring only the columns a
  circuit happens to declare, this cannot silently miss a Pauli with nonzero ideal
  expectation. Opt-in; the default path remains `postselected_logical_error_rate`.

### Changed
- **engine-9** (`xtim-engine-9`, bumped from engine-8). The engine now computes the
  EXACT set of sign-controlling records (bare-state stabilizer anticommutation
  analysis) and refuses (`XtimRejectError`) if a declared frame is missing a
  required record or includes a spurious one. A column with no solvable fixed frame
  (a genuine free/logical read, e.g. a destructive `MPP` readout) is not refused —
  it reports `signed_value = nan`, `sign_constant = False`, `frame_hint = None`.
- **`_discover_frame` retired — `xtim.diagnose` determinism is now EXACT.** The
  statistical sign-inference pass that inferred the byproduct frame from a sample of
  noiseless shots is removed; detector determinism is decided exactly (the
  parity-Pauli lies in the bare state's stabilizer, via `detector_determinism`)
  rather than by noiseless sampling, which mislabeled measurement-noise-fed
  detectors. `signed_value` is now exact and shot-independent (computed from the
  declared frame, verified analytically by the engine); `frame_hint` now echoes the
  DECLARED `rec[-k]` targets, not a statistically-discovered parity.

### Fixed
- **`code_switching_faithful` post-selection is now complete.** The final noiseless
  syndrome-extraction round declares its six Steane stabilizers as detectors, so a
  pure post-selection heralds every single-fault logical error. Its
  post-selected `1 − F` drops from ~5×10⁻³ (a spurious O(p) floor) to ~1.5×10⁻⁶ at
  p=10⁻³ — the fault-distance-limited rate. (See BREAKING: this changes the
  circuit's detector count.)

### BREAKING
- **Expectation streams for `code_switching_faithful` and `miniature_oracle` are
  now frame-fixed.** `code_switching_faithful` (records {16…22}) and
  `miniature_oracle` (record {2}) now declare their byproduct frames in-circuit.
  Their per-shot `exps` columns are sign-constant in the noiseless case; previously
  a noiseless run could flip sign across shots (the old `_discover_frame` inference).
  Downstream code that applied its own sign-correction based on the raw per-shot
  sign must be updated. **`exps.mean()` calls are unaffected** — the mean was
  correctly computed before and remains correct after.
- **`code_switching_faithful` detector count changed (20 → 26).** The added
  end-of-circuit noiseless SE round (see Fixed) declares six more detectors, so the
  circuit's detector sample shape (`sample()` / `.dets` columns) changed. Code that
  indexes this circuit's detector array by absolute position must be updated.
  **Post-selected LER / fidelity values are unaffected** (they improve, per Fixed).

---

## [1.0.2] - 2026-07-03

A performance fix. No API change, no `.ref` format change.

### Fixed
- **The R=0 record-sampling path no longer thrashes on high-pattern-diversity circuits.**
  1.0.1 fixed the expectation-channel (R>0) memo; the memory-mode (R=0) path had the same
  pathology. `code_switching_faithful` R=0 sampling went from ~13 µs/shot (default, warmup-
  bound) to **~0.85 µs/shot** — now ~3× faster than a comparable near-Clifford simulator.
  The R=0 path now uses the same linear-Pauli-correction-chain construction as R>0 (one
  unified builder for both modes).

### Reproducibility (note)
- R=0 same-seed byte streams changed for the affected (high-pattern-diversity) circuits —
  statistically identical, validated; the committed pins were regenerated. Other circuits'
  streams are unchanged.

## [1.0.1] - 2026-07-03

A performance fix for 1.0.0. No API change, no `.ref` format change.

### Fixed
- **Expectation-channel sampling no longer thrashes on high-pattern-diversity circuits.**
  In 1.0.0 the per-shot expectation-channel tree was built path-unique (an exponential
  forced-collapse recursion) and cap-saturated builds were wrongly marked unusable, so a
  circuit like `code_switching_faithful` fell off the fast path: **~21 µs/shot**. The two
  forced-collapse branches at each biased coin are related by a Pauli read off the shared
  frame, so the channel tree now collapses into a linear correction chain (one linear-time
  construction, no per-path recursion) with zero build refusals. `code_switching_faithful`
  is now **~0.61 µs/shot** (≈1.6 M shots/s); the per-shot memo is a net win again. A ctest
  tripwire guards this class of regression.

### Reproducibility (note)
- R=0 (memory-mode) byte streams are **identical** to 1.0.0. R>0 (expectation-mode)
  same-seed byte streams changed (the expectation re-routing) — statistically identical,
  dense-oracle + `stim_parity` validated; pins/goldens regenerated.

## [1.0.0] - 2026-07-03

**One engine.** The framed circuit-level engine now carries ALL modes — memory (R=0)
and expectation (R>0) — at every stabilizer rank χ. The legacy shot loop and the
separate lean state are deleted. No public Python API change; the breaking changes are
in the sampling byte streams, the removed env levers, and the on-disk `.ref` format.

### BREAKING
- **Env levers removed.** `QEC_FRAMED`, `QEC_AB_REDUCE`, and `QEC_AB_OBS` are gone (they
  are now silently ignored). In particular the `QEC_FRAMED=0` "reproduce the old engine's
  streams" contract no longer exists — **there is no env combination that reproduces a
  ≤0.8.0 byte stream.** The one retained cross-check lever is `QEC_FORCE_FALLBACK=1`
  (compile-time): it routes every shot through the engine's full-state general path — an
  independent route through the same physics whose distribution must match the default.
  New performance-only tuning levers (results unchanged): `QEC_TREEPLAN_CAP`,
  `QEC_FRAMED_MEMO`, `QEC_PARTNER_HASH_MIN`.
- **Same-seed byte streams changed.** All expectation-mode (R>0) circuits now run on the
  framed engine, and R=0 streams changed for `cultivation_d5`-class circuits. The changes
  are statistically identical — validated against a dense-statevector oracle and
  `stim_parity --full`; the committed goldens/pins were regenerated. Within-a-build
  same-seed determinism still holds exactly.
- **Expectation semantics: consistent outcome-conditioning.** Observables read on a
  fresh-coin measurement wire are now outcome-conditioned at *every* χ (per-shot values
  are conditioned on the recorded outcomes). Previously some χ>2 paths inconsistently
  reported unconditional marginals. **Means are unaffected** (dense-statevector-validated);
  only the per-shot values on those wires change.
- **`.ref` on-disk format v4** (`REF_FORMAT_VERSION` 3 → 4; new refs drop the anchor
  block). **A v4 `.ref` cannot be read by xtim ≤ 0.8.0.** This release reads v2/v3/v4, so
  upgrading is safe — your existing `.ref` files still load. The reference cache keys on
  the format version, so a stale-format cache entry misses cleanly and recompiles.

### Performance
- Expectation-mode (R>0) circuits now compile framed expectation channels: large-χ R>0
  workloads speed up by orders of magnitude (a χ=16 sweep: 24.3 → 0.11 µs/shot);
  `cultivation_d3_faithful` ~0.64 → ~0.50 µs/shot, `distillation_15_1_3` ~unchanged.
- Memory-mode `cultivation_d5` marginal shot cost ~12 → ~8.7 µs/shot; χ up to 1024 now
  samples at the sub-µs tier-0 rate.

### Changed
- Internal: a single state type (`FramedSuperposition`); the χ=3 capability cliff is gone
  (χ>2 is fully supported); a unified `TreePlan` plans at all χ.
- `Reference` now exposes `.version` (the `.ref` format version of the stored artifact).

## [0.8.0] - 2026-07-01

A sampler-speed release. A new framed circuit-level engine is now the production
default. No public API change, no `.ref` format change (0.7.1 references stay valid).

### Performance
- **Memory-mode sampling is ~1.2× faster** via the new framed circuit-level engine
  (`QEC_FRAMED`, default on; set the env var `QEC_FRAMED=0` for the previous engine).
- **Expectation-mode worst-path shots are 2–10× faster** — pattern-miss shots now take
  an A/B block path with tensor-split expectations instead of a full-state fallback
  (`QEC_AB_OBS`, default on). Output is unchanged, only faster.

### Reproducibility (note)
- **Memory-mode (R=0, no `PAULI_EXPECTATION`) same-seed byte streams differ from 0.7.1.**
  Statistically identical, Stim-validated to 5σ — but the *specific* shots at a fixed seed
  differ. **`QEC_FRAMED=0` reproduces the 0.7.1 streams exactly.** (Same class as the
  0.7.0 lean-gauge note.) Within-a-build same-seed determinism still holds exactly, and
  expectation-mode (R>0) streams are unchanged.

### Fixed
- `XtimDemError` no longer prints "not DEM-expressible" twice.
- Docs: refreshed the `cultivation_d3_faithful` throughput figure (~1 M shots/s after the
  expectation-mode speedup) and corrected a syndrome-rate comment in the DEM guide.

## [0.7.1] - 2026-06-30

An internal cleanup release — **byte-identical sampler output**, leaner source. No
public API change, no `.ref` format change, no behavior change.

### Changed
- **~6.3k net lines of C++ removed**: legacy stabilizer representations, the `extsim`
  subsystem, and ~12 dead bench/oracle apps are gone; research-only code (distance,
  code I/O) split out of the magic engine into a separate library the xtim wheel does
  not link. The magic engine is now ~18k C++. Same wheel, smaller source.
- `chi_max` is now fully retired internally (it was already gone from the public API in
  0.6.0; this removes the last dead internal threading, including the CLI positional arg).
- `REF_FORMAT_VERSION` now correctly reports **3** (was a stale `1`).

### Fixed
- `Circuit.__eq__`/`__hash__` normalize trailing whitespace, so a circuit equals its
  `from_file(to_file(...))` round-trip and its `append`-built twin (`to_file`'s
  documented round-trip now holds).
- `xtim.Reference(circuit)` (a `Circuit` instead of stored `.ref` text) now raises a
  typed `XtimReferenceError` pointing you to `c.compile_reference()`, instead of a raw
  pybind `TypeError`.
- Docs: removed an internal handoff section that was shipping in `xtim_dem_reject.md`;
  fixed the `cube_ccz` row in the examples table; clarified that `xtim.cache_dir`
  defaults to `None` (meaning `.xtim_cache/`) and that `scale_noise` on a noiseless
  circuit is a no-op, not an error.

## [0.7.0] - 2026-06-29

An internal engine refactor: the sampler's execution path is now fully tableau/lean —
the affine `CanonicalStabSum` is gone from sampling (replaced by a lean A/B
factorization). **No public API change and no `.ref` format change** — your 0.6.0
`.ref` files and all code keep working unchanged.

### Changed
- **The sampler is affine-free** (tableau/lean throughout). Internal; verified by an
  exact state-overlap oracle, a stim distribution-equivalence campaign, and the
  whole-branch review.

### Reproducibility (note)
- **Memory-mode (R=0, no `PAULI_EXPECTATION`) same-seed sampling streams differ from
  0.6.0.** The lean factorization changed the B-block gauge, which reorders the
  per-shot RNG byte stream for memory/syndrome circuits — **statistically identical**
  and stim-validated (max |z| = 1.03), but the *specific* shots at a fixed seed are not
  the same as 0.6.0. Expectation-mode (R>0) streams are unchanged. If you pinned exact
  bytes for a memory-mode circuit at 0.6.0, regenerate them; results are equivalent.

### Fixed
- Docs: the post-selected `distillation_15_1_3` infidelity at p=1e-2 is ≈ 3.5e-5 (the
  35·p³ floor), and the un-post-selected baseline is ≈ n·p ≈ 15p (the `Z_ERROR` model)
  — earlier text had ≈5e-5 / ≈10p.
- `diagnose()`'s byproduct-frame HINT now labels the record index as 0-based
  (`parity(0-based records [k])`), so it isn't misread as 1-based (which scores F≈0.5).

## [0.6.0] - 2026-06-29

A new engine (engine-7) with a **breaking** simplification: the `chi_max` cap is
gone. Stabilizer rank χ is now purely a cost knob, never a rejection criterion — the
diagonal-Clifford propagation-class check is the **sole** thing that rejects a
circuit. Plus a compile cache (repeated `sample()` no longer recompiles), a leaner
per-shot path, and two new flagship examples.

### Removed (BREAKING)
- **`chi_max` is removed everywhere.** It was a memory backstop that the engine no
  longer needs, so the argument is gone from `Circuit.compile_sampler(...)`,
  `Circuit.compile_detector_sampler(...)`, `Circuit.detector_error_model_with_reject(...)`,
  the `CompiledSampler`/`CompiledDetectorSampler` constructors, and the `python -m xtim`
  CLI (`--chi_max`). Any output whose stabilizer rank χ is large is still simulated
  **exactly** — it just costs more per shot. The old `XtimRejectError(kind="chi_cap")`
  reject no longer exists; the only rejection is `kind="class"` (a basis change on a
  magic-carrying wire leaves the simulable class). **Migration:** delete any `chi_max=`
  argument — there is nothing to replace it with.

### Changed
- **engine-7** (C-4 parity-native reference construction; FastTODD-only magic
  compiler). Internal — sampling distributions are unchanged (verified by an exact
  state-overlap oracle + a distribution-equivalence gate). The on-disk `.ref` cache is
  internal and self-invalidates on the upgrade.

### Performance
- **Compile cache.** A compiled sampler now holds its compiled program, so repeated
  `sample()` calls on the same object no longer recompile the reference (e.g. a d5
  cultivation went from ~1.5 s per `sample()` call to ~ms after the first).
- **Leaner per-shot path** (the affine amplitude anchor is off the sampling path),
  plus a precomputed active set — measurable per-shot wins on the d5-class circuits.

### Added
- **`distillation_15_1_3`** — a genuine post-selected 15-to-1 magic distillation on the
  [[15,1,3]] quantum Reed–Muller code: transversal `T` with physical T-gate phase noise
  (`Z_ERROR`, baked p0=1e-3) and the 4 X-stabilizers measured as detectors. Post-select
  a trivial syndrome (`keep=lambda dets, meas: ~dets.any(axis=1)`) and the distilled
  infidelity drops to the weight-3 floor ~35·p³ — far below the physical rate. Declares
  ⟨X̄⟩ and ⟨Ȳ⟩, so `Task(target_k=1)` gives a true 1−F vs p curve.
- **`cultivation_d5`** — a d=5 magic-state cultivation T-count / syndrome fixture (no
  `PAULI_EXPECTATION`; syndrome sampling only).
- Both are bundled in the wheel — `xtim.load_example("distillation_15_1_3")` etc.
- A **Highlights** section in the README featuring the d3/d5 cultivation and the 15-to-1
  distillation with measured amortized throughput.

### Fixed
- `scale_noise(circuit)` (a `Circuit` instead of its text) now raises a clear
  `TypeError` pointing you to `.text`, instead of a leaked `AttributeError`.
- The DEM-refusal hint for a high-weight (`>16`) observable now describes the real
  cause (a too-wide twirl-coin-read expansion) instead of misattributing it to a
  `PAULI_EXPECTATION` column.
- An out-of-class circuit no longer prints a misleading "sampling stays exact" cache
  warning before its `XtimRejectError`.
- Docs: honest hardware-dependent throughput figures; corrected the reject example
  (`CH`, not a χ cap); removed a stale `to_stim_text` claim (`Circuit.text` preserves
  the original tokens) and clarified the `sample()` return contract.

## [0.5.12] - 2026-06-24

An engine performance + cache release. **No public API, Python-surface, or
sampling-result change** — pure internal speedups and a larger reference cache.

### Performance
- Reference compilation is dramatically faster for large magic-state /
  distillation references (direct structure-preserving construction): a
  [[15,1,3]]×k distillation reference at k=8 went ~272 s → ~5 s. And disjoint
  magic blocks now reduce independently (stabilizer rank `2^k` instead of `8^k`),
  so multi-block distillation references compile at all (k≥4 was infeasible before).
- Faster warm-cache loading: the on-disk reference format loads without re-deriving
  the frame structure, so cache hits on large references start faster.

### Changed
- Reference cache cap raised **1024 → 16384**: magic states with stabilizer rank up
  to 16384 now memoize (the internal `.ref` cache) instead of recompiling every run.
- Internal engine bump (engine-6, on-disk reference format v3). The reference cache is
  internal and self-invalidates on the upgrade — the first run after upgrading
  recompiles references, then they cache. Nothing is user-written.
- Sampling **distributions are unchanged** (verified by an exact state-overlap oracle
  + a distribution-equivalence gate). For very large references the exact per-shot
  byte sequence at a fixed seed may differ — never a supported contract; statistical
  results are identical.

## [0.5.11] - 2026-06-22

A small fidelity-feature hardening plus a worked multi-magic example.

### Added
- `xtim.collect` also warns when a row's **noiseless fidelity ceiling**
  `(1 + Σβ²) / 2**target_k` is below 1 — the symmetric partner to the existing
  `F > 1` warning. It flags an INCOMPLETE declared `PAULI_EXPECTATION` support (or
  a `target_k` that is too large); the value is returned unchanged. The two
  warnings now bracket a correct `target_k` / column set from both sides.

### Docs
- New worked example `examples/cube_832_ccz.py`: the [[8,3,2]] cube code's
  transversal-CCZ magic state, prepared with stabilizer-measurement + feedback,
  scored to a TRUE 3-qubit fidelity over the complete logical Pauli support
  (noiseless `F = 1`), with a noiseless syndrome round + post-selection that
  suppresses `1−F` from `O(p)` to `O(p²)` (only the physical T gates are noisy).

## [0.5.10] - 2026-06-22

Documentation and examples for the v0.5.9 fidelity feature. **No code change** —
the engine, wheels, and results are identical to 0.5.9.

### Docs
- New worked example `examples/fidelity_curve.py`: a `1−F ± sem` vs `p` curve
  (with a matplotlib plot) built from the `Task(target_k=...)` feature.
- README gains a *"shortcut: a fidelity curve from `collect`"* subsection, the
  examples index lists the new example, and the tutorial notebook shows
  `target_k` scoring (the post-selection-buys-fidelity / acceptance trade-off).

## [0.5.9] - 2026-06-22

A figure-of-merit release: `xtim.collect` now turns declared `PAULI_EXPECTATION`
targets into a publishable fidelity curve directly. Backward-compatible — opt-in
via `Task(target_k=...)`; without it, rows are unchanged.

### Added
- `xtim.collect` computes **true-state fidelity** when a task sets `target_k`
  (logical-qubit count of the target magic state). Rows gain `fidelity`,
  `infidelity`, `infidelity_sem`, `target_k`, `target_beta` — a `p`-sweep drops
  straight into a `1−F ± sem` vs `p` curve. The error bar is `std/√N` of the
  per-shot fidelity scalar (correlation-aware). Completeness of the declared
  `PAULI_EXPECTATION` set is the user's responsibility; gauge columns are refused.
- A `UserWarning` is raised when a row's `fidelity` exceeds 1 beyond sampling
  noise — the loud signal that `target_k` is too small or the declared columns
  are over-complete. The returned value is left unclamped.

### Performance
- `target_beta` (the noiseless target) is computed ONCE per unique noiseless
  circuit per `collect` call instead of once per task, so a `p`-sweep over one
  base circuit pays the `diagnose` cost a single time (~3× faster real-circuit
  sweeps).
- `substitute_var` ~2.8× faster transversal-T reference compile (deferred from
  the v0.5.8 trailing perf note).

### Docs
- The tour shows the `target_k` one-liner for a 1−F vs p curve; `help(xtim.collect)`
  is self-contained; `infidelity` is documented as unclamped (guard
  `max(0.0, infidelity)` before log plots); ndarray row fields note `.tolist()`
  for JSON.

## [0.5.8] - 2026-06-21

A foundation consolidation plus internal performance wins. **No public API,
on-disk format, or sampling-result change** — the consolidation is byte-exact
(default engine behavior, results, and `.ref` format unchanged; verified across a
broad statevector cross-check and the bundled demos to the digit).

### Performance

- **`build_bare_state` ~12 s → sub-second for transversal-`T` states** — much faster
  reference compile for high-distance transversal-`T` circuits.
- **`CHState` overlap ~k× faster** (k = active magic dimension).

Both are compile-time / internal; sampling per-shot speed and results are unchanged.

### Fixed

- **Pathological `cache_dir` no longer leaks a raw error.** A `xtim.cache_dir` the
  filesystem rejects (e.g. an embedded NUL byte) raised a bare `ValueError` from the
  cache store; it now follows the documented contract — `XtimCacheWarning` + the exact
  deduced bare state.
- **`scale_noise(..., p0=0)` raises a clear `ValueError`** instead of a raw
  `ZeroDivisionError`.

### Internal

- Foundation cleanup (−269 LOC): `clifford_frame → clifford_tableau` rename, a unified
  reference-χ ceiling header (`ref_config.hpp`), and dead-code removal — byte-exact.
- The packaging build now lists the TU-`#include`d engine sources as `depends`, so an
  incremental rebuild of the exported tree can't serve a stale `.so`.
- Docs: noted that `PAULI_EXPECTATION`'s parenthesized argument is an integer label,
  not the target value.

## [0.5.7] - 2026-06-21

A **caching-only** improvement (no sampling-result or public-API change): higher-χ
references now memoize instead of recomputing every run.

### Fixed

- **References with 64 < χ ≤ 1024 now cache.** The reference-compile verification
  battery sampled the just-built reference at a hardcoded χ≤64, while construction
  built it at χ≤1024 — so a reference whose stabilizer rank fell in (64, 1024] (e.g.
  ≳7 independent magic qubits) passed construction but **failed the verification
  gates**, was rejected, and fell back to the exact *deduced* bare state, recompiling
  on every run (with an `XtimCacheWarning` each time). The chi ceiling is now unified to
  1024 across construction, verification, and the compile-time oracle floor, so these
  references compile, verify, and **cache once** (`reference_info()['chi']` is reported,
  no warning). A χ=8 reference (e.g. `cube_ccz`) is unaffected — `chi_max` is a cap, not
  the actual rank, so low-χ references sample identically. (χ > 1024 still falls back to
  the exact deduced state — the synthetic genuinely-irreducible worst case.) Supersedes
  the [0.5.6] note that said 64 < χ ≤ 1024 falls back.

## [0.5.6] - 2026-06-21

A performance fix: the multi-magic reference compile no longer falls off a cliff.
**Compile-time only** — no public API, on-disk format, or sampling change; cached
`.ref` bytes and stream pins are byte-identical, results unchanged.

### Fixed

- **Multi-magic reference compile no longer hangs / cliffs.** Compiling the reference
  for a state with many independent magic qubits (stabilizer rank χ growing as `2^m`)
  previously blew up super-exponentially and effectively hung past ~6 magic qubits.
  Three independent hot spots are fixed: the conditioning pass's
  `max_clifford_subspace` search (an exponential Clifford-grade family →
  apriori-enumerated, bounded, overflow-safe), and both the standalone and
  `build_frontier`-inline `reduce_to_chi2` stabilizer finders (`O(χ³)` brute
  enumeration → `O(χ²)` phase-solve). The compile **no longer hangs** at any m — the
  whole χ=2^m sweep up to m≈13 finishes in seconds. The protocols xtim actually ships —
  reducible / code-structured states like the [[8,3,2]] transversal CCZ (`cube_ccz`) —
  compile **and cache** in milliseconds. *(A genuinely-irreducible χ=2^m state with
  m ≳ 7 still falls back to the exact **deduced** bare state rather than caching: the
  compile attempt is now bounded — ~1–3 s at χ≈2k–4k, no hang (`build_frontier`'s
  residual `O(rays²)`) — sampling is exact regardless, but the verification battery
  samples at χ≤64 so these don't memoize. The synthetic worst case; real protocols are
  unaffected.)*

## [0.5.5] - 2026-06-20

An engine consolidation (`xtim-engine-4`) that removes several subtle internal
errors and extends the reference compiler. No public API change; sampling results
are unchanged (raw measurement-record *bytes* may shift for circuits using `MPP`
of an all-Z product — see below — but every distribution, detector, observable,
and expectation is identical).

### Fixed

- **Multi-magic on a code state now compiles & caches a reference.** Reference
  compile previously fell back to the (exact) deduced bare state for a transversal
  non-Clifford layer acting on a measure+feedback-prepared code state — e.g. the
  [[8,3,2]] transversal CCZ — leaving `reference_info()['chi']` as `None` and emitting
  an `XtimCacheWarning` every run. These now compile and cache like any other circuit
  (`cube_ccz` reports χ=8, no warning). Most multi-magic circuits now cache.
- **Stale references can no longer be served, automatically.** The invisible reference
  cache now keys on `sha256(circuit text + ref-format version + sha256(the compiled
  engine))`, so any engine rebuild changes the key and old `.ref` entries drop
  themselves — no manual `ENGINE_VERSION` bump can be forgotten (the label is now purely
  cosmetic). This closes a bug where a stale reference could be silently served after the
  engine changed. For end users this is invisible: a released wheel ships one engine
  binary, so its hash is stable and caches reuse fully across runs; only a rebuild (a
  different binary) invalidates — which is correct.

### Changed

- **`MPP` of an all-Z product desugars to a Z-basis ancilla** (no Hadamard). The same
  operator is measured (verified against Stim) and all distributions are unchanged; only
  the internal representation differs, so raw record *bytes* may shift.
- **The sampler accepts more circuits** (e.g. a Hadamard on a magic-carrying wire that
  used to reject now samples).

## [0.5.4] - 2026-06-20

An **onboarding** release: a `pip install`-the-wheel user can now go from
"installed" to "loaded and scored a real protocol" with no repo clone. No engine
or on-disk-format change; `0.5.3` caches and DEMs are unaffected.

### Added

- **`xtim.load_example(name)` — bundled demo circuits, no clone needed.** The six
  demo `.stim` circuits now ship *inside* the wheel, so `xtim.load_example("cube_ccz")`
  loads any of them straight from the installed package; `xtim.list_examples()` lists
  them and `xtim.example_path(name)` gives the file path. Names accept the `.stim`
  suffix; an unknown or path-like name raises a clear error. (Previously a wheel user
  hit “No such file” on the README’s very first example, because the circuits lived
  only in the repo.)
- **`examples/xtim_tutorial.ipynb` — a hands-on Jupyter tutorial** walking the whole
  arc end to end: hello magic state → `diagnose()` → scoring (with the frame-correction
  footgun) → decoder-in-the-loop → noise sweep → the [[8,3,2]] multi-magic CCZ. Loads
  its circuits via `load_example`, so it runs from any directory.
- A **`[tutorial]`** install extra (`stim` + `pymatching` + `matplotlib`) for the
  notebook (`pip install "xtim[tutorial]"`).

### Fixed

- **Source distribution completeness.** `MANIFEST.in` shipped only `examples/*.stim`
  and `*.md`, so the runnable example scripts (`onboarding_new_protocol.py`,
  `dem_reject_region.py`) — which the docs tell you to run — were missing from the
  sdist; the tutorial notebook would have been excluded too. Now ships `*.py` and
  `*.ipynb`.
- **Wheel-vs-repo clarity.** The README and `examples/README.md` now state plainly that
  the wheel bundles the demo *circuits* (`load_example`) while the example *scripts*,
  the notebook, and `docs/` live in the repo / sdist — so a wheel user knows what needs
  a clone before hitting a missing file.

## [0.5.3] - 2026-06-20

An engine **correctness fix** for classically-controlled feedback that crosses a
non-Clifford gate, plus the first multi-magic example. No public API or on-disk
format change.

### Fixed

- **Feedback (`CX`/`CY`/`CZ rec[-k] q`) crossing a non-Clifford gate is now
  coherentized — it was previously silently wrong (or refused).** Such feedback is
  rewritten into a coherent basis-matched controlled-Pauli before deferral, so it
  propagates correctly through `T`/`CS`/`CCZ`. Concretely, a logical magic-state prep
  that measures a syndrome and feedback-corrects *before* the transversal non-Clifford
  gate (e.g. the [[8,3,2]] transversal CCZ) now returns the exact expectation on every
  shot, instead of a wrong value on the feedback-corrected shots. Feedback that does
  **not** cross magic is unchanged (same fast frame trick, byte-identical, no χ growth).
  The only outcomes now are correct-coherentized, correct-frame-trick, or a loud
  `XtimRejectError` — never a silent wrong answer. `detector_error_model_with_reject()`
  coherentizes too, so feedback-before-magic circuits export a DEM cleanly.

### Added

- **`examples/cube_ccz.stim` — the first multi-magic demo.** The [[8,3,2]] cube color
  code with a transversal CCZ: `k=3` logical qubits, output `CCZ|+++⟩_L` (a genuine
  `χ>2` state, beyond the single-magic `χ≤2` sweet spot the other demos use). A
  deterministic feedback prep gives a clean `diagnose()` — sign-constant, well-defined,
  `⟨X̄₁⟩=⟨X̄₂⟩=⟨X̄₃⟩=+0.5`, 100% acceptance, no post-selection. (Reference compile falls
  back to the exact deduced-bare-state path here — the normal multi-magic case; the
  `XtimCacheWarning` is benign.)

## [0.5.2] - 2026-06-19

Five rounds of persona-driven UX hardening (fresh-eyes "new user" testing). All
documentation and error-message polish — **no engine, API, or on-disk-format
change**; `0.5.0`/`0.5.1` caches and DEMs are unaffected and existing code keeps
working.

### Fixed

- **`diagnose()` no longer reads as a failure on a healthy magic-prep run.** A
  whole-circuit DEM refusal previously tagged every expectation value line
  (`[circuit DEM refused …]`) and printed a mid-report `reason:` wall of engine
  jargon ("negative probability", "refusing to approximate"). It now surfaces
  **once**, as a demoted bottom note that *leads* with reassurance ("expected for
  magic-state prep … sampling stays exact … nothing is wrong"), with the engine
  reason trimmed to one line. The cold-cache `χ` line reads `not yet cached
  (deduced on the fly — fine)` instead of a bare `?`.
- **README fidelity guidance is now honest and complete.** The 30-second example
  contracts the full target Bloch vector against all expectation columns (prints
  `F ≈ 1.0` for the *post-selected* — not "noiseless" — state), states the
  column-coverage precondition, and carries a prominent ⚠️ caveat that the
  raw-mean shortcut is valid only for `sign constant` columns — a `byproduct
  frame` protocol must fold in the record-parity `HINT` first or `F` collapses to
  ~0.5. Added a 3-term vocabulary block (χ / gauge detector / byproduct frame)
  and glossed "DEM" at first use. The shown `diagnose()` block now matches real
  output verbatim.
- **Clean, typed errors on bad input everywhere.** `Circuit.from_file()`,
  `Reference.load()`, and the CLI now raise a typed `XtimError`/`XtimReferenceError`
  (or print a one-line CLI message + non-zero exit) for a missing **or
  binary/corrupt** file and for invalid `--shots`/`--chi-max`, instead of leaking
  a raw `pathlib`/codecs/`ValueError` traceback.
- **Doc/example accuracy.** `examples/dem_reject_region.py` resolves its circuit
  from any directory and reports the nontrivial-correction count honestly
  (cultivation's reject region is saturated → decode is a no-op on survivors; the
  guide now demonstrates real partial-decode on the `cultivation_d3_rate` twin).
  Corrected the `miniature_oracle` table cell, the `code_switching` refusal
  reason (non-deterministic observable), the decode-and-budget characterization
  (a coherent-error *budget/estimate*, not a "lower bound"), and a stale test
  filename; dropped a `sinter` name-drop that implied a bridge that does not
  exist.

## [0.5.1] - 2026-06-19

Documentation and example-UX patch from new-user feedback. No engine, API, or
on-disk-format change — `0.5.0` cache files and DEMs are unaffected.

### Fixed

- **README 30-second fidelity example was misleading.** The headline snippet combined
  only a single expectation column into `F`, printing `F ≈ 0.75` for a state whose true
  fidelity with the target is `1.0`. It now contracts the **full** signed target Bloch
  vector against **all** expectation columns
  (`F = (1 + (target * exps[keep].mean(0)).sum()) / 2`), printing `F ≈ 1.0` as expected
  for the noiseless target.
- **`examples/dem_reject_region.py` only ran from the `examples/` directory.** It now
  resolves its circuit via the same beside-file-then-`benchmarks/` fallback as
  `onboarding_new_protocol.py`, so it runs from anywhere (and from the monorepo).
- **Reference-compile fallback warning read as an alarm.** When a circuit has no
  gate-battery reference (expected for some circuits), the warning now leads with the
  reassurance that *sampling stays exact — no action needed*, instead of surfacing the
  internal `oracle=FAIL` detail as if something were wrong.

## [0.5.0] - 2026-06-19

First release since 0.3.1: ships both the engine-2 reference-compile speedup (see
[0.4.0] below — staged but never tagged on its own) and the new decoder-DEM-with-reject
feature.

### Added

- **Decoder DEMs that flag the not-Pauli-correctable faults, for magic-state-prep.** New
  `Circuit.detector_error_model_with_reject(...) -> xtim.DemWithReject` plus the public
  `xtim.PostselectFault` dataclass. The `DemWithReject` carries `{dem, reject_detectors,
  postselect_faults}` + `.keep_mask(dets)`: `dem` is a clean Stim `DetectorErrorModel` of the
  Pauli-correctable faults, and **`postselect_faults`** is the primary output — each a
  `PostselectFault(detectors, probability)` flagging a fault that is detectable but not
  Pauli-correctable (a coherent logical `S` no Pauli frame recovers), for *you* to either
  **post-select** (sound) or **decode-and-budget** (their summed probability is the coherent-error
  floor). `reject_detectors` is the derived convenience union (and `.keep_mask` the blunt
  conservative post-select); xtim does not force a policy. Faults are classified by whether a
  logical magnitude `|⟨P̄_r⟩|` changes (sign flips stay correctable). An odd-`S` on a measured
  detector read twirls to a probability-½ Pauli edge (no refusal). A **completeness gate** refuses
  loudly (`XtimDemError`) when a single fault changes a magnitude but fires no detector (the circuit
  under-declares its post-selection syndrome). See `docs/xtim_dem_reject.md` and
  `examples/dem_reject_region.py`. Additive: `detector_error_model()` is unchanged and
  byte-identical for non-magic circuits.

## [0.4.0]

A large reference-compile speedup (engine version bump) plus the `diagnose()`
robustness/signed-target work from the QEC-user feedback round.

### Changed

- **Reference compile is dramatically faster.** It now prefers the deduced
  conditioning path (`build_bare_state`) over the per-T-gate frontier split, falling
  back to the frontier only when the deduced path rejects. Transversal-T-on-code (a
  single logical magic state) now compiles **n=127 in ~0.10 s (was ~8.75 s) and
  n=1023 in ~12 s (was ~50 min)**. The physics is identical (overlap = 1; every
  compile-battery gate still passes); per-shot sampling is unchanged.
- **`chi_max` is a memory backstop, not a rejection criterion — and the default is
  raised 4096 → 16384.** xtim does not reject on stabilizer rank in normal use: a χ>2
  output (the sweet spot is χ≤2, a single logical magic state) is still simulated
  exactly via the deduced bare state. `chi_max` only guards against a genuinely
  enormous output; it now allows substantially larger χ before that backstop fires,
  and you can raise it further. (The docs previously framed χ>2 as "overflow and
  reject," which was inaccurate — χ>2 samples fine.)

### ⚠ Cache invalidation (engine version bump)

- The engine version is bumped **`xtim-engine-1` → `xtim-engine-2`**. The faster
  compile yields a different — but physically identical — serialized reference
  (`.ref`) gauge, so **every engine-1 `.ref` cache is invalidated; xtim recompiles
  automatically on next use.** No action required: caches under `.xtim_cache/` are
  safe to delete, and old caches are simply ignored.

### Fixed

- **`diagnose()` no longer crashes on a feedback (or otherwise DEM-refusing)
  circuit.** Its internal DEM-export probe only caught `XtimDemError`, so a circuit
  with classically-controlled Pauli feedback — whose DEM export raises
  `XtimParseError` ("feedback not yet handled by DEM export") — propagated out and
  aborted the whole report. It now also catches `XtimParseError`/`XtimRejectError`
  and records the refusal as a fact (with a clean reason, stripped of the generic
  parse hint that misleads in a DEM context). The report renders for every circuit.

### Added

- **Quality-of-life from the user-feedback rounds:** `Circuit.without_noise()` (Stim
  parity); `Diagnosis._repr_html_` (the formatted report renders in Jupyter);
  `collect(progress=...)` callback and per-row `p0` / `expectation_paulis` fields (so a
  flattened results table stays self-describing); and a `py.typed` marker so downstream
  type-checkers see xtim's inline annotations.
- **Signed target in `diagnose()`.** Each `ExpectationInfo` now carries a
  `signed_value` — the byproduct-frame-corrected `βᵢ` (the target `⟨Pᵢ⟩` of the ideal
  output in the canonical zero-frame), so the report shows `value=+0.707…` rather than
  only `|value|=0.707…`, and a closing `target (signed beta): [+0.707, -0.707]` line
  gives the full target Bloch vector in one place. `signed_value` is `nan` when the
  sign is genuinely indeterminate (varies with no single record-parity explaining it);
  `|signed_value| == abs_value` otherwise. This is a DERIVED, reported fact — diagnose
  still applies nothing to the raw shot data.

## [0.3.1]

Documentation + diagnostics **honesty pass** from a round of QEC-user feedback — no
engine or API change. The simulable scope and the decoder/DEM story are now stated
accurately, and several misleading error hints were corrected.

### Fixed / clarified

- **README:** added an upfront **"Scope & limits"** section (xtim is a
  magic-state-prep *co-processor* — single χ≤2 magic output, ≲10 T gates, Clifford
  bulk belongs in Stim); documented that classically-controlled Pauli feedback and
  `decompose_errors` are *shipped* (the old "deferred" list was stale and denied the
  v0.3 headline feature); honest DEM scope (faithful magic protocols are
  post-selection-only, not decoder-in-the-loop); made the install snippet
  version-agnostic (`pip install xtim-*.whl`) so it can't go stale; documented the
  run-from-a-clone (`ModuleNotFoundError: xtim._xtim`) and
  benign `XtimCacheWarning` (χ>2) traps.
- **examples/README:** corrected the table — `miniature_oracle` is n=5; `faithful`
  cultivation's *reference* χ is 2 (the ~256 figure is a transient one-time-compile
  rank, not per-shot; it samples at ~0.04 ms/shot); only `cultivation_d3_rate` is
  Clifford / DEM-exportable (`code_switching_rate` keeps the transversal `T̄`, so it
  is still non-Clifford).
- **`docs/STIM_GAP_AUDIT.md` is now shipped** (it was referenced but absent).
- **dialect doc:** `CH` is marked parse-only / run-time-rejecting (it embeds a
  Hadamard that leaves the simulable class).
- **Error hints:** the DEM-refusal hint now routes by cause — graphlike-decompose
  failures advise `ignore_decomposition_failures` (not the unhelpful
  `include_expectations=False`); the class-reject hint names `CH`; the chi-cap hint no
  longer leaks a literal `chi=K` on the deduced-reference path; `HERALDED_*` reports
  as a recognized-but-unsupported Stim instruction (not "unknown instruction"); the
  `rec`-as-gate-target message notes that only `CX`/`CY`/`CZ` take feedback.

## [0.3.0]

Stim-compatibility foundation — most of Stim's circuit surface now loads, plus
classically-controlled Pauli feedback (a Stim feature earlier xtim lacked). The
non-Clifford engine and the sampling hot path are unchanged: feedback-free
circuits stay byte-identical (stream pins 6/6).

### Added

- **Classically-controlled Pauli feedback `CX` / `CY` / `CZ rec[-k] q`.** A `CX`/`CY`/
  `CZ` whose **first target is a measurement record** `rec[-k]` applies `X`/`Y`/`Z` to
  qubit `q` iff that record bit is 1 (Stim's measurement-record-controlled Pauli
  feed-forward). Handled as an **exact post-sampling triangular GF(2) record-relabel**:
  the controlled-Paulis are pulled out of the bare circuit (so the bare state / records /
  expectations stay feedback-free), each is propagated once at setup to a
  `(control record, record-flip mask, expectation-sign mask)` descriptor, and per shot a
  triangular pass XORs the flips into later records / flips the selected
  `PAULI_EXPECTATION` signs. A Pauli before a measurement only flips that outcome (or an
  expectation sign), never a Born probability — so this is exact even with magic in the
  circuit. The relabel is **off the state-evolution hot loop** and entered only when the
  circuit contains feedback, so feedback-free circuits are byte-identical and same-speed
  (stream pins 6/6). The record distribution matches real Stim (which runs feedback
  natively). Out of scope: classically-controlled **Cliffords**, `sweep[k]` controls, and
  DEM-with-feedback (all rejected with a clear message). See
  [`docs/xtim_dialect.md`](docs/xtim_dialect.md) §1b.
- **Measurement-gap parity with Stim:** two-qubit Pauli measurements `MXX` / `MYY` /
  `MZZ`, inverted measurement targets (`!q`), and readout-flip probabilities `M(p)` /
  `MR(p)` / `MX(p)` (the record takes the wrong value with probability *p*). Record
  distributions byte-match real Stim.
- **Full Stim Clifford gate set, verified tableau-equal to Stim.** Every Stim unitary
  Clifford now loads and is pinned against Stim — `I`, the SWAP family, `SQRT_*`,
  basis-cycling `C_*`, the `XCX … ZCZ` two-qubit set, aliases, and the `[tag]` suffix —
  plus `MPAD`, `I_ERROR`, and `II_ERROR`. Locked by a 50-gate Stim-equality suite
  (which self-tests that it would catch a wrong desugaring).
- **DEM `decompose_errors=True`** — graphlike error decomposition for matching decoders
  (PyMatching/MWPM); `detector_error_model(decompose_errors=,
  ignore_decomposition_failures=)` mirrors Stim's kwargs. Surface and repetition codes
  are decode-equivalent to Stim's DEM; color-code d3 carries a small documented residual
  (xtim's coarser canonical-channel merge — see `STIM_GAP_AUDIT.md`); non-graphlike
  cases require `ignore_decomposition_failures`, as in Stim.

## [0.2.0]

### Added

- **Pauli-product measurement `MPP` and rotation gates `SPP` / `SPP_DAG`.** Now
  supported via **parse-time desugaring** — no engine/sampler change. `MPP` measures
  one or more `*`-joined Pauli products, each producing one measurement record in
  Stim's record order (desugared to a fresh-ancilla cat-check gadget: `H a`; one
  controlled-Pauli per factor; a `Z a` for a leading `!`; terminal `MX a` = the
  record). `SPP P` = `exp(-iπ/4·P)` / `SPP_DAG P` = `exp(+iπ/4·P)` desugar to a
  CX/CY/CZ basis-change ladder onto a pivot + `S`/`S_DAG`, then uncompute. The record
  distribution byte-matches real Stim (incl. the `i^{#Y}` and `!`-inversion sign
  conventions); circuits that use neither are byte-identical and same-speed as before.
  See [`docs/xtim_dialect.md`](docs/xtim_dialect.md) §2b/§2c.
- **`Circuit.diagnose()`** — an onboarding report from ONE noiseless run. It
  surfaces the facts you would otherwise reverse-engineer by hand: the reference
  χ and cache status, which detectors are deterministic (post-selectable) vs
  gauge, each `PAULI_EXPECTATION` column's `|β|` magnitude, its byproduct-frame
  sign HINT (a record parity), and whether it is DEM-expressible — plus an
  advisory post-selection recipe. It *reports*; it never applies a correction or
  filters shots.
- **`python -m xtim diagnose <circuit.stim> [--shots N] [--seed S]`** — the CLI
  surface for the report above.
- **Actionable `.hint` on reject exceptions.** `XtimParseError`,
  `XtimRejectError`, and `XtimDemError` now each carry a `.hint` (a student-facing
  fix) and a `.location`, and their `__str__` prints cause + location + hint. The
  2/3/4 CLI exit-code contract is unchanged.
- **Gates `T_DAG` and `CS_DAG`** — the inverse π/4 phase and controlled-S†
  (parse-time conveniences that desugar to existing gates).
- **Coordinate preservation.** `QUBIT_COORDS` / `DETECTOR(coords)` are stored and
  surfaced via `Circuit.get_final_qubit_coordinates()` and
  `Circuit.get_detector_coordinates(only=)` (Stim's method names), with
  `SHIFT_COORDS` resolved cumulatively. The exported DEM carries detector
  coordinates as `detector(x, …) D#` lines. Coordinates are annotations: they never
  affect any record stream.
- **Output formats `dets` and `r8`.** `python -m xtim sample`/`detect` now accept
  `--out_format dets|r8` (alongside `01`/`b8`), byte-identical to
  `stim … --out_format dets`/`r8`. xtim keeps detection events, observable flips and
  measurements in SEPARATE channels (`--out`/`--obs_out`/`--meas_out`), so each `dets`
  channel is homogeneous and labelled by its Stim type hint: `D#` detectors, `L#`
  observables, `M#` measurements. `r8` is the run-length byte encoding (gap-to-each-1
  plus a trailing-run byte, 0xFF continuation) per channel. The `--exp_out` float64
  channel has no bit meaning and stays `%.17g` text (or raw f64 under `b8`).
- **Programmatic circuit building.** `xtim.Circuit` now mirrors Stim's
  construction surface: `append(name, targets, arg)`, `+`/`+=` (concatenate),
  `*`/`*=` (a `REPEAT` block), `to_file`, `copy()`, `len()`, `str()`, and the
  complete `num_qubits`/`num_measurements`/`num_detectors`/`num_observables` set
  (all matching real Stim's counts). Targets accept ints, raw target strings
  (`rec[-1]`), or `stim.GateTarget`s. Every mutation re-parses, so an invalid
  build raises `XtimParseError` and leaves the circuit untouched.
- **Onboarding example + dialect doc.** `examples/onboarding_new_protocol.py`
  walks a fresh protocol from circuit text to a scored fidelity via
  `Circuit.diagnose()`; `docs/xtim_dialect.md` specifies the dialect delta.
- **`xtim.__version__`** — resolved from the installed package metadata.

## [0.1.0]

Initial release: Stim-shaped sampling of logical magic-state-preparation
protocols (an extended-Stim dialect with `T`/`CS`/`CH`/`CCZ` and
`PAULI_EXPECTATION`), a real `stim.DetectorErrorModel` export with expectation
L-columns, the `collect()` batch runner, and the `python -m xtim` CLI
(`sample` / `detect` / `analyze_errors` / `state`).
