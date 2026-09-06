# Twirl record sampler — Python API + semantics contract

```python
import xtim

s = xtim.compile_twirl_sampler(circuit_text, p_factor=1.0, disk_cache=None,
                               selfcheck=2000, skip_refused_observables=False)
dets, obs = s.sample(shots, seed=1)   # Stim-b8-packed numpy uint8 arrays
rep = s.channel_report()              # dict, keys below
```

Engine: `qeccore::TwirlRecordSampler` (`cpp/include/qeccore/twirl_sampler.hpp`), the
end-to-end twirl record sampler of the V2 twirl-kernel arc. Per shot: production-grade
sparse noise draw → exact composition of the fired atoms → diagonal/PPR normal form →
canonicalization mod the certified stabilizer group → content-keyed plan memo → record
emission directly in channel space (baseline ⊕ coin rows ⊕ prefix qubit rows ⊕ kernel
folds). The binding (`_xtim.TwirlSampler`) owns the bare `FramedSuperposition` the engine
references, and replicates the bench driver's setup exactly (parse → normalize
{coherentize, defer, want_map, feedback=Strip} → `build_bare_state` → `from_css` →
terminal reads), with `circuit_channels=true`.

## The semantics contract

1. **Deterministic detector/observable channels are EXACT.** A DETECTOR (or OBSERVABLE)
   whose combined record operator lies IN the certified stabilizer group is compiled to a
   σ-mask channel; its per-shot bit is `parity(σ & mask) ⊕ ref` (the exact reduction sign).
   These channels' laws equal the production sampler's — **5σ-gated vs the production
   sampler on all benchmarks, including the CH-class ones** (`framed_bench twirl_records
   --circuit-channels --gate`, V2-T2/T3).

2. **Gauge-record marginals/correlations are TWIRLED.** A detector classified LOGICAL
   (gauge) has no deterministic channel in the V1 record product. Its column in
   `sample()`'s output is a **DECLARED independent fair coin** (Bernoulli(1/2), drawn
   from the sampler's seed-tied gauge RNG). This is the spec's declared semantics for the
   record product — gauge columns are *not* claimed to reproduce the physical gauge
   record's correlations with anything else.

3. **Anti detectors refuse at compile.** A record operator that anticommutes with the
   certified group cannot be a well-formed detector; `compile_twirl_sampler` raises
   `ValueError` naming it (cannot happen for well-formed circuits).

4. **One LOGICAL observable is emitted as a Born-weighted channel (V3).** On diagonal
   tables, the FIRST observable classified LOGICAL becomes a σ-correlated biased coin
   (`⟨W⟩_σ = ±(−1)^{c_W·σ}·⟨Λ_W⟩`, executable spec `scripts/twirl_obs_reference.py`):
   the (detectors, observable) JOINT is exact — post-selected and decoded statistics
   match the true channel (5σ-gated incl. post-selected and rejected-shot means).
   Remaining observables with no exact or Born channel are REFUSED: `sample()` raises
   `RuntimeError` naming them — unless compiled with `skip_refused_observables=True`
   (all-zero columns, marked in `channel_report()["refused_observables"]`).
   Deterministic (IN_GROUP) observables are emitted exactly like detector channels.

5. **No shot is ever dropped (V3-T3 R1).** Shots that trip a twirl guard (κ≥2 plan
   geometry, out-of-class PPR composition, born κ>0 / classification guards) are
   computed **per shot by the exact engine**: the residual is applied to a fresh bare
   copy (as gates for diagonal residuals; by generator pullback through the fired alts'
   composed error tableaus for PPR), the certified generators are measured sequentially
   (a commuting family — the sequential joint IS the true channel law), and in born mode
   the observable is measured on the SAME collapsed state. The exact rows are true
   samples of the same law and are delivered/accumulated like any other shot;
   `channel_report()["exact_shots"]` counts them (diagnostic only; 0 in distribution on
   all benchmarks). The exact path draws from a dedicated RNG, so all other shots'
   byte streams are unchanged. Forced-fallback gates: `tests/test_twirl_fallback_exact.py`
   on `benchmarks/fallback_{kappa2,ppr_kf2,born_k1}.stim` (each class exercised with
   fb>0 AND 5σ-gated vs production).

6. **The plan-cache disk path serves p-sweeps.** Plans are keyed on noise *content*, not
   probability values (noise-blind): one `disk_cache` file warms every p-point of a sweep
   and every rerun. `disk_cache` accepts (V3-T3 R2):
   * a **path string** — used verbatim;
   * **`True`** — automatic lifecycle: the file lives under `$XTIM_TWIRL_CACHE_DIR`
     (default `~/.cache/xtim/twirl/`) and is keyed
     `<fnv64(deferred-signature)>-<group-token>.twpl`. The deferred signature is
     structural (noise values excluded), the group token is the certified group's
     identity token; the TWPL framing stores + re-verifies the token on load, so
     stale/corrupt/cross-group files load nothing and rebuild silently. The resolved
     path is reported in `channel_report()["disk_path"]`.

## `sample(shots, seed=None) -> (detectors, observables)`

* `detectors`: `uint8[shots, ceil(num_detectors/8)]`, Stim b8 layout (bit `i` of shot `s`
  at byte `i>>3`, bit `i&7`). Deterministic detector columns exact (point 1); gauge
  columns fair coins (point 2); all other bits zero.
* `observables`: `uint8[shots, ceil(num_observables/8)]`. Deterministic observables
  exact; skipped refused observables all-zero (point 4).
* `seed=None` continues the sampler's current stream (the stream before any explicit
  reseed runs at the default seed 1). An explicit `seed` deterministically **rebuilds**
  the engine sampler.
* **Byte-determinism contract:** same seed → byte-identical output **at fixed plan-cache
  state**. Without `disk_cache`, repeated `sample(N, seed=s)` on one object is byte-stable
  (each rebuild starts from an empty in-memory memo). With `disk_cache` set, every run
  enriches the cache file and a later rebuild pre-warms plans whose shots previously took
  the slow path; the collapse RNG is consumed differently on the fast path (64-bit coin
  reservoir) than on the slow path (one draw per coin), so BYTES may differ across calls
  while every reported DISTRIBUTION is identical. Statistics never depend on cache
  warmth; raw byte reproducibility does.
* Gauge coins are drawn for **every** row so the gauge coin stream stays row-aligned.
* Not thread-safe: the GIL is released during sampling; do not call `sample()` /
  `channel_report()` concurrently from another thread.
* Exact-fallback rows (semantics point 5) are REAL samples written like any other row;
  `channel_report()["exact_shots"]` counts them (`fallback_shots` is a legacy alias of
  the same diagnostic). `used` always equals `shots`.
* A tripped engine oracle (the first `selfcheck` fired shots re-run the full σ path and
  must match EXACTLY) raises `RuntimeError` — loud, never silent-wrong.

## Engine selection: `Circuit.compile_detector_sampler(..., engine=)`  (V3-T3 R3)

* `engine="exact"` (default for this release) — the legacy engine, unchanged.
* `engine="twirl"` — this sampler behind a `TwirlDetectorSampler` adapter whose
  `sample()` matches `CompiledDetectorSampler.sample()` for the record path
  (`separate_observables` / `append_observables` / `bit_packed`; every call replays the
  same stream at the compile seed). Documented delta: `return_expectations` /
  `return_measurements` raise `ValueError` (not part of the twirl record product).
* `engine="auto"` — the twirl channel compilation runs once; the twirl engine is used
  iff EVERY detector classifies deterministic (gauge/anti content disqualifies), no
  observable remains refused, and setup succeeded — else the exact engine, silently.
  `engine_report()` on the returned sampler says which engine won and why.
* Kill switch: `QEC_NO_TWIRL=1` forces exact everywhere (`auto` routes to exact;
  explicit `engine="twirl"` raises `RuntimeError` naming the switch).

## `channel_report() -> dict`

| key | value |
| --- | --- |
| `deterministic_detectors` | detector indices with exact channels |
| `gauge_detectors` | LOGICAL-classified detector indices (fair-coin columns) |
| `anti_detectors` | ANTI-classified detector indices (empty — compile refuses) |
| `refused_observables` | observable indices with no exact channel |
| `deterministic_observables` | observable indices emitted as exact channels |
| `num_channels` | total compiled record channels (detectors + observables) |
| `plans` | diagonal plan-cache size |
| `ppr_plans` | PPR plan-cache size |
| `exact_shots` | shots computed by the per-shot exact engine (INCLUDED in rows/counts) |
| `fallback_shots` | legacy alias of `exact_shots` |
| `used` | shots entering the accumulation (== shots sampled) |
| `disk_path` | resolved plan-cache file ("" = disk layer off) |

## Test

`tests/test_twirl_python_api.py` — on `benchmarks/cultivation_d3_faithful.stim`:
compile + 200k-shot sample, each deterministic detector's mean vs the production
sampler (`xtim.Circuit(...).compile_detector_sampler().sample(...)`) at 5σ;
channel-report sanity; the refused-observable raise + skip path; determinism
(same seed → same bytes).

Run:

```
PYTHONPATH=$PWD /home/user/anaconda3/bin/conda run -n qec \
    python -m pytest tests/test_twirl_python_api.py -x -q
```
