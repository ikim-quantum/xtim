# Practical notes — CLI, caching, seeding, and scope in practice

Working knowledge that doesn't fit the tutorial or the reference pages: the
command line, how the reference cache behaves, reproducibility guarantees, and
what xtim's scope means for day-to-day use.

## Scope in practice

xtim is a **magic-state-preparation co-processor, not a Stim replacement.** Its
sweet spot is protocols whose *output* is a single low-rank logical magic state
(cultivation, magic-state injection, code switching) — there it exactly computes
the magic-state expectation that Stim cannot, and fast.

- **Stabilizer rank — a sweet spot, never a reject.** The reference (bare) state
  is built by an exact stabilizer-rank decomposition; cost grows with χ, so the
  **speed/memory sweet spot is χ ≤ 2** (a single logical magic state). Higher-χ
  outputs (multi-magic — a transversal-gate logical magic state like the [[8,3,2]]
  CCZ, a 15-to-1 distillation, CCZ factories) are **still simulated exactly** and
  compile to a cached reference too — they just cost more per shot. There is
  **no χ cap** — xtim never rejects a circuit for its stabilizer rank. The *only*
  thing it rejects is a circuit outside the simulable class — its non-Clifford
  gates must fold into **one mutually-commuting layer of π/8 rotations**
  ("generalized T-depth 1"; see
  [`xtim_simulable_class.md`](xtim_simulable_class.md)); χ itself is purely a
  cost knob, not a gate.
- **Decoder-in-the-loop — including magic protocols.** Magic-state-prep protocols
  (cultivation, oracle prep) export via `c.detector_error_model_with_reject()` → a
  `DemWithReject`: feed `.dem` to `pymatching` to decode the Pauli-correctable
  faults, and handle `.postselect_faults` — the faults a Pauli decoder *can't*
  correct, each **flagged** (detector signature + probability) for *you* to either
  post-select (`.keep_mask(dets)` / the `.reject_detectors` union are the blunt
  conservative policy) or decode-and-budget (their summed probability is the
  coherent-error floor). **Default to the blunt `keep_mask` post-select** (always
  sound) unless you need every shot. Plain Clifford circuits export an ordinary
  `detector_error_model()`. The exception is `code_switching`, which genuinely
  refuses (post-select only): it raises `XtimDemError: non-deterministic
  observable L0 (... not ±1)` — its logical observable isn't deterministic
  noiselessly, so there is no exact Pauli DEM. See
  [`xtim_dem_reject.md`](xtim_dem_reject.md).
- **Clifford bulk: keep using Stim.** Per-shot the gap is modest (measured on stock
  surface-code memories, same core, packed records: Stim is ~1.3× faster on the
  measurement channel and ~2.4–2.9× on the detector channel at d = 7–11), but xtim's
  reference-state compile takes ~1 s (d=7) to ~9 s (d=11) — paid once per circuit
  shape, since the cache is noise-value-blind — where Stim's is milliseconds. Sweeps
  over many circuit *shapes* belong to Stim; pair xtim (magic) with Stim (Clifford).

## Command line

```bash
python -m xtim diagnose protocol.stim --shots 1024 --seed 0          # the onboarding report
python -m xtim sample   --in protocol.stim --shots 1000000 --seed 7 --out meas.b8 --out_format b8
python -m xtim detect   --in protocol.stim --shots 1000000 --seed 7 --out dets.b8 --obs_out obs.b8 --exp_out exps.txt --meas_out meas.b8
python -m xtim analyze_errors --in protocol.stim [--no-expectations]  # the detector error model
python -m xtim state compile  protocol.stim out.ref                  # precompute the reference state
python -m xtim state verify   protocol.stim out.ref                  # check a saved reference matches
```

(`sample` / `detect` / `analyze_errors` take the circuit via `--in`; `diagnose`
and `state` take it as a positional argument.)

## Building circuits programmatically

`xtim.Circuit` mirrors Stim's construction surface, so a Stim user's habits
transfer — `append(name, targets, arg)`, `+`/`+=` (concatenate), `*`/`*=` (a
`REPEAT` block), `to_file`/`from_file`, `copy()`, `len()`, and the full
`num_qubits`/`num_measurements`/`num_detectors`/`num_observables` set:

```python
c = xtim.Circuit("")
c.append("H", [0, 1])
c.append("CX", [0, 1])
c.append("X_ERROR", 0, 1e-3)        # parens arg = noise probability
c.append("M", [0, 1])
c.append("DETECTOR", ["rec[-1]"])   # rec[-k] resolves against c's own measurements
program = c + xtim.Circuit("X 0") * 3   # `+` concatenates; `* 3` makes a REPEAT block

# The magic readout: PAULI_EXPECTATION needs a mandatory integer (label) and a
# `*`-joined Pauli product — write it as text (`PAULI_EXPECTATION X0` with no label
# is rejected).
magic = xtim.Circuit("RX 0\nRX 1\nT 0\nT 1\nPAULI_EXPECTATION(0) X0*X1")
```

(A `rec[-k]` target is resolved against the circuit it is appended to, so build
`DETECTOR`/`OBSERVABLE_INCLUDE` with `append` onto a circuit that already has the
measurements — not as a standalone `xtim.Circuit("DETECTOR rec[-1]")` text fragment,
which has no measurements to point back at.)

Every mutation re-parses, so an invalid build (unknown gate, wrong arity) raises
`XtimParseError` immediately and leaves the circuit untouched — you can never
assemble something that won't load.

## Good to know

- **Invisible reference state.** The first `compile_*` deduces and caches the
  circuit's reference state under `.xtim_cache/` (relative to your working
  directory; `xtim.cache_dir` defaults to `None`, meaning that `.xtim_cache/` — set
  it to a path to override). Later runs load it (a corrupted or
  unreadable entry is ignored — it falls back to recomputing, exact either way). The
  cache is safe to delete. The cache key masks **noise values** (the reference
  provably cannot depend on the probabilities), so a `Task(p=...)` / `scale_noise`
  sweep compiles the reference once and every p-point hits the same entry. Noise
  *structure* stays in the key — adding, removing, or moving a noise line is a
  different circuit as far as the cache is concerned.
- **Persisting a reference yourself.** `ref = c.compile_reference()` runs the compile
  now and returns a `Reference` (`.n`, `.chi`, `.source`, `.text`); `ref.save(path)`
  writes a portable `.ref`, `xtim.Reference.load(path)` reads one back, and the CLI
  mirrors this with `python -m xtim state compile`/`verify`. An **explicitly supplied**
  `.ref` whose qubit count doesn't match the circuit is rejected (`XtimRejectError`,
  `kind="ref_mismatch"`); a *cached* entry that mismatches is instead handled by the
  silent-correct fallback (a benign `XtimCacheWarning` + the deduced bare state), since
  the cache is an optimization, not a contract.
- **`diagnose()` is also programmatic.** Beyond the printed report, the returned
  `Diagnosis` exposes `chi`, `deterministic_detectors`, `gauge_detectors`,
  `suggested_postselect`, and `expectations` — a list of `ExpectationInfo` with
  `abs_value` (|β|), `signed_value` (the frame-corrected target βᵢ), `sign_constant`,
  `frame_hint`/`frame_const` (the byproduct record-parity), and `dem_expressible`.
- **Honest DEM — and the reject region.** `c.detector_error_model()` returns a real
  `stim.DetectorErrorModel`, or refuses loudly (`XtimDemError`) rather than silently
  approximating. For **magic-state-prep** protocols, use
  `c.detector_error_model_with_reject()` → a `DemWithReject(dem, reject_detectors,
  postselect_faults)`: the clean Pauli decoder DEM plus `postselect_faults` — the
  not-Pauli-correctable faults (those that change a logical magnitude), each flagged as
  a detector signature + probability for you to post-select or budget. `reject_detectors`
  is the derived convenience union (and `.keep_mask(dets)` the blunt conservative
  post-select); both are empty for the Clifford `rate` twin. Only `code_switching`
  (non-independent detector channels) genuinely refuses and is post-selection-only
  (`diagnose()` + NumPy). `include_expectations=False` (the default)
  keeps the magic value out of the DEM — it's the payload, read from the
  expectation channel, **never decoded as a DEM observable**. Full guide:
  [`xtim_dem_reject.md`](xtim_dem_reject.md).
- **Running from a clone?** Run your scripts from a *different* directory (or rely on
  the installed wheel). Importing `xtim` from the repo root picks up the uncompiled
  source package and — unless the `_xtim` extension has already been built in-tree —
  fails with `ModuleNotFoundError: xtim._xtim`.
- **Reproducible seeding.** A compiled sampler re-seeds deterministically from its
  fixed `seed` on *every* `sample()` call (it does not advance the stream as repeated
  Stim sampling would), so two calls on the same object return identical results; the
  streams are also identical across processes. (Same-seed *byte* streams are **not**
  guaranteed stable across xtim versions — an engine change can reorder the per-shot RNG
  while keeping the distribution identical; the CHANGELOG notes each such change. The
  statistics you compute are stable; exact byte pins are not.)
  Bit-packed output (`bit_packed=True` / the `b8` format) is little-endian within each
  byte — unpack with `np.unpackbits(..., bitorder="little")`, matching Stim.
- **`XtimCacheWarning` is benign (and rare).** If xtim ever can't build a cached
  reference for a circuit — an unwritable cache dir, or a shape the reference compiler
  doesn't yet handle — it falls back to the engine's deduced bare state and emits this
  warning. **Sampling is still exact**, identical to the cached path; only the one-time
  caching is skipped (so `reference_info()['chi']` is then `None`). Most circuits,
  including multi-magic ones, now compile and cache.

## When something rejects

The typed exceptions (`XtimParseError`, `XtimRejectError`, `XtimDemError`) each
carry an actionable **`.hint`** — read it rather than guessing. There is exactly ONE
class-rejection condition: the circuit's non-Clifford gates must fold into a single
mutually-commuting layer of π/8 rotations (["Which circuits can xtim
simulate?"](xtim_simulable_class.md)). For example, a T-state fed straight into a
same-wire controlled-Hadamard check is *non-commuting magic* and rejects:

```python
try:
    # T-magic on the CH TARGET wire anticommutes with the CH's own magic -> rejects.
    # (T on the CONTROL wire commutes and is accepted; so is the Clifford-ALIGNED
    #  H-eigenstate protocol — see xtim_simulable_class.md.)
    xtim.Circuit("RX 0\nRX 1\nT 1\nCH 0 1\nM 0 1").compile_detector_sampler().sample(100)
except xtim.XtimRejectError as e:
    print(e.kind)   # 'class'
    print(e.hint)   # '...must fold into ONE mutually-commuting layer of π/8 rotations...
                    #  Clifford-align the state's magic axis with the measured check...'
```

(Writing `H` then `M` to read in the X basis does **not** reject — xtim soundly
rewrites that terminal pair to a native `MX`. And `CH` itself is supported — see
[`xtim_dialect.md`](xtim_dialect.md) — it rejects only in genuinely non-commuting
placements.)
