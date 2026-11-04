# The twirl record sampler (engine contract)

`xtim.compile_twirl_sampler(text, p_factor=1.0, selfcheck=2000)` compiles a circuit into xtim's
sampling engine (`_xtim.TwirlSampler`, C++ `qeccore::TwirlRecordSampler`); `Circuit.compile_detector_sampler()`
wraps the same engine behind Stim's interface. Per shot, the sparse noise draw is composed exactly into one Clifford
residual, reduced to a content-keyed plan, and every DETECTOR / OBSERVABLE / DECISION record is a row of that
plan's record table. User-level API: [usage.md](usage.md).

## Contract

1. **One joint record law.** Every record (deterministic, gauge or anticommuting) is sampled from the exact joint
   law after the shot's residual. Gauge detectors are exact Born records; nondeterministic records are reported in
   `record_report()`; no observable is refused.
2. **Reference-relative bits**, Stim's `reference_sample` convention, including `INPUT_QUBITS` / `ideal_state`
   port columns.
3. **No fallback.** Every shot is served by the law. What the engine cannot serve is refused by name at compile
   with `xtim.XtimTwirlRefusal` (`twirl_class`: `OUTSIDE_T_DEPTH_1`, `NONCOMMUTING_MAGIC`,
   `LAW_LIMITATION_PROPAGATION`, `LAW_LIMITATION_CARRIED_AXIS`); `sample(return_expectations=True)` and
   `sample(return_measurements=True)` are refused as `EXPECTATIONS_NOT_SERVED` / `RAW_RECORDS_NOT_SERVED`.
   Engine invariant failures raise `RuntimeError`; parse errors raise before any engine runs.
4. **Non-Clifford scope rule (static fixed-axis merge).** Push every non-Clifford term and every fault site with
   p > 0 to the end of the circuit. Same-axis terms merge only when nothing between them can anticommute with
   their axis; a merged pair that sums to an S-type Clifford folds into the Clifford frame; iterate to a fixpoint.
   The circuit is accepted iff the surviving axes commute pairwise for every fault configuration.
5. **Noise-blind plan cache.** Plans are keyed on noise content, not probabilities. The cache is in memory and
   owned by its sampler; nothing is persisted to disk. One header per distinct residual key, and one record table +
   emission block + Born trie per CLASS of keys with the same record law (the key's S bits on wires no record reads
   and the CZ pairs in the kernel `K2'` — the pair sets whose pullback of every record is a +1 certified element,
   with every pair of records still commuting — are quotiented out; `record_report()["plan_memory"]` reports
   `classes`, `class_builds`, `class_kernel_dim` and `class_cap`). Every reader of a plan's table reads its class's.
   Payloads are flat blobs in a slab arena owned by the cache (`arena_chunk_bytes`, `arena_free_bytes`); the FIFO
   eviction under the payload budget gives a class served since the last pass one more lap (`second_chances`).

## Sampling

- `sample(shots, seed=None)` returns `(detectors, observables)` as Stim b8 `uint8[shots, ceil(n / 8)]` (bit `i` at
  byte `i >> 3`, bit `i & 7`), plus decisions when the circuit declares `DECISION`s.
- Every draw of shot `i` is a pure function of `(seed, i, stream, draw)`: the same seed gives the same bytes,
  independent of the plan-cache state, of `sample` vs `sample_barrier`, and of how shots are split across calls
  (same binary and CPU feature set).
- The first `selfcheck` fired shots of a sampler's first run are re-checked by in-engine oracles (stream-neutral,
  announced once by `xtim.XtimVerificationWarning`; a trip raises `RuntimeError`). `selfcheck=0` turns it off.
- Feedback `CX/CY/CZ rec[-k] q` is rewritten into coherent gates before deferral; when needed it is moved to its
  source measurement (the theorem and its per-instruction rules are in `cpp/src/coherentize_feedback.cpp`).
- `sample_barrier(shots, seed)` also retains each shot's state (plan, branch, frame Pauli, outcomes) for
  `xtim.RecordFidelity`. One call at a time per sampler: a concurrent call is refused (`XT_SAMPLER_BUSY`).
- `threads=` (`sample`, `sample_barrier`; an int or `"auto"`) runs one call's shots on up to that many threads with
  byte-identical output for every count (each shot keeps its global index and row). The lanes share the sampler's
  construction state; during a super-block they read the plan cache and build what it lacks into their own overlays,
  which the calling thread adopts at the barrier; the term route's shots (CH / PPR faults, input frames, carried
  branches) run on the calling thread in shot order. The selfcheck window and `oracle_all_shots` run serially, and the
  lane count is capped to the state budget: `record_report()["threads"]` = `{requested, effective, limit,
  deferred_shots}`. The next super-block's noise is generated while the lanes run the current one (tasks over channel
  ranges; the same stream). `RecordFidelity.evaluate(threads=)` builds the contents new to the evaluator in parallel
  (installed in first-row order) and splits a content of many rows into chunks on several lanes; it runs without the
  GIL at every thread count.

## `record_report()`

A dict: `classes` (per record: `deterministic`, `coin` or `born`), `nondeterministic_detectors` /
`_observables`, `route_counts`, `term_route`, plan / memory / trie / branch counters, `semantic_oracle_shots`,
`feedback_hoisted`, `threads` (the last call's), and `patterns` (IF-partitioned samplers only).
Thread-count-dependent (diagnostics, not output): the route split between the fast and the structural paths, plan
misses / builds, Born-trie node evaluations and root-memo hits, plan ids and record content handle numbering; and
RecordFidelity's `projection_skips` and `rn_words` (its value, shot, miss, memo-hit, byte and sector counters are
invariant).

Tests: `tests/test_twirl.py`, `tests/test_ja_*`, `tests/test_jb1_*`.
