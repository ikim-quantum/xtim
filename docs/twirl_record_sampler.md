# The twirl record sampler (engine contract)

`xtim.compile_twirl_sampler(text, p_factor=1.0, disk_cache=None, selfcheck=2000)` compiles a circuit into xtim's
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
5. **Noise-blind plan cache.** Plans are keyed on noise content, not probabilities, so one `disk_cache` file
   (a path, or `True` for `$XTIM_TWIRL_CACHE_DIR`, default `~/.cache/xtim/twirl/`) serves a whole p-sweep. Stale,
   corrupt or older-format files load nothing and rebuild.

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
  `xtim.RecordFidelity`. Not thread-safe: do not sample one sampler from two threads.

## `record_report()`

A dict: `classes` (per record: `deterministic`, `coin` or `born`), `nondeterministic_detectors` /
`_observables`, `route_counts`, `term_route`, plan / memory / trie / branch counters, `semantic_oracle_shots`,
`feedback_hoisted`, `disk_path`, and `patterns` (IF-partitioned samplers only).

Tests: `tests/test_twirl.py`, `tests/test_ja_*`, `tests/test_jb1_*`.
