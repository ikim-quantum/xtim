# xtim tour — the lean decoder-in-the-loop workflow

The executable worked example for the `xtim` package (spec:
`docs/superpowers/specs/2026-06-12-xtim-python-ux-design.md`, M2). Every
`python` code block below is executed verbatim by
`tests/test_tour.py::test_tour_document_executes` from the repository
root, so this document cannot rot.

xtim's workflow is ordinary Stim's, with two extra ingredients made easy: the
reference/final state is resolved invisibly, and `PAULI_EXPECTATION`
declarations come back as **raw** per-shot expectation values. The engine emits
raw physics only — every interpretive step below (decoding, post-selection, the
β-combination) is plain user numpy.

## Setup

The committed benchmark `benchmarks/cultivation_d3_rate.stim` is the d=3
cultivation circuit's Clifford rate twin: it ends with two declarations,
`PAULI_EXPECTATION(0) X0*X3*...` and `PAULI_EXPECTATION(1) Y0*Y3*...` — the
logical equator axes of the output state. It is DEM-expressible end-to-end,
expectation columns included.

```python
import numpy as np
import pymatching
import xtim

N = 20_000
c = xtim.Circuit.from_file("benchmarks/cultivation_d3_rate.stim")

# The circuit has 14 noisy in-circuit detectors + 6 noiseless end-SE syndrome
# detectors (declared at coordinate (0,0,4), 2026-07-04). For MWPM we build
# the DEM without the end-SE DETECTOR lines — those checks are for post-selection
# (accepted shots have all end-SE syndromes == 0), not for decoding.
text_for_dem = "\n".join(l for l in c.text.splitlines()
                          if not l.startswith("DETECTOR(0, 0, 4)"))
c_dem = xtim.Circuit(text_for_dem)   # 14 in-circuit detectors only

# user inputs to the workflow below:
my_detector_subset = np.array([0, 1, 2])  # post-select these; DECODE the rest
beta = np.array([0.0, 1.0])  # the target's Bloch decomposition: a Y-eigenstate,
                             # so <X-product> carries beta 0 and <Y-product> beta 1
```

The first run compiles and caches the circuit's reference state (the full
`ref_compile` gate battery) under `.xtim_cache/`; later runs load and
cheap-verify it. Nothing about that is visible here — that is the point.

## The workflow

This is the spec's eight-line workflow. One engine feature (`PAULI_EXPECTATION`
columns live in the DEM after the observables — `c.expectation_columns` is the
index list) and one convention make any DEM-compatible decoder work unmodified:
a Pauli correction acts on a declared Pauli product by anticommutation only,
⟨C†P̄C⟩ = ±⟨P̄⟩, so applying a decoder's predicted flip to an expectation is
exactly a sign — no re-simulation.

```python
dets, obs, exps = c.compile_detector_sampler(seed=7).sample(
    N, separate_observables=True, return_expectations=True)
m = pymatching.Matching.from_detector_error_model(c_dem.detector_error_model())
n_decode = c_dem.num_detectors                          # = 14 in-circuit detectors
flips = m.decode_batch(dets[:, :n_decode])              # standard decoder
corrected = np.where(flips[:, c.expectation_columns], -exps, exps)
keep  = ~dets[:, my_detector_subset].any(axis=1)        # post-selection = discard
value = corrected[keep].mean(axis=0)                    # a-bar per declared Pauli
F = (1 + beta @ value) / 2                              # user's own target arithmetic
```

- `dets` is `bool[N, 20]` detection events (14 noisy + 6 noiseless end-SE), `obs` is `bool[N, 1]` observable
  flips, `exps` is `float64[N, 2]` **raw** expectation values — one engine run,
  one RNG stream.
- `c.detector_error_model()` is a real `stim.DetectorErrorModel`;
  `pymatching` consumes it directly. Its L-columns are the observables first,
  then one column per `PAULI_EXPECTATION`, both in declaration order, so
  `c.expectation_columns == [1, 2]` here.
- Label-ordering note: stim itself calls every L-column an "observable", so the
  parsed DEM reports `dem.num_observables == c.num_observables +
  c.num_expectations` — the LAST `c.num_expectations` of those labels are the
  expectation columns, which is exactly what `c.expectation_columns` indexes.
- Post-selection modifies nothing; it discards. Everything not post-selected is
  *decoded*: the matched correction's anticommutation with each declared Pauli
  arrives as the flip bits in `flips[:, c.expectation_columns]`.

## The final β-combination is user arithmetic

For a pure k=1 target, |φ⟩⟨φ| = (1/2)(I + Σᵢ βᵢ P̄ᵢ) gives
F = (1 + Σᵢ βᵢ·āᵢ)/2 — the last line above. With this benchmark and seed:

```python
assert keep.sum() > 0.8 * N
assert value[1] > 0.9              # decoded a-bar of the Y-product channel
assert 0.9 < F < 1.0
print(f"kept {keep.sum()}/{N} shots, a-bar = {value}, F = {F:.4f}")
```

At the committed noise strength (p = 1e-3 baked in) this prints
`F ≈ 0.96` — and the decoder is doing real work: replacing `corrected` with
the raw `exps` in the `value` line drops F by about 2 percentage points, the
sign damage that matching predicted and repaired.

```python
raw_value = exps[keep].mean(axis=0)
assert beta @ raw_value < beta @ value   # decoding strictly helps here
```

## Sweeping over p: `xtim.collect`

Batch studies are the same lean workflow, run per task by `xtim.collect`. A
`Task` bundles a circuit with its decoding policy — `keep` (post-selection
mask) and `decoder` (flip bits for the expectation columns), both plain
callables of `(dets, meas)` — plus `p`, which rescales every noise argument of
the circuit text by `p / p0` (`p0` = the baked-in strength, default the
committed benchmarks' 1e-3; `xtim.scale_noise` is the helper, and it fails
loudly on any partial or line-count mismatch, so a botched rewrite never
silently mis-scales — though a circuit with *no* noise lines rescales to itself,
so a `p`-sweep on a noiseless circuit is a no-op, not an error). Each row that
comes back is plot-ready: `p`, `shots`, `kept`, `acceptance_rate`, `value`
(mean corrected expectation per declared Pauli), per-entry `sem`, `seconds`,
and your `metadata` — e.g. `pandas.DataFrame(rows)`.

```python
def my_keep(dets, meas):
    return ~dets[:, my_detector_subset].any(axis=1)

def my_decoder(dets, meas):
    return m.decode_batch(dets[:, :n_decode])[:, c.expectation_columns].astype(bool)
    # (decoding every p with the p-baked DEM's matching — deliberate here;
    #  build a per-p matching from xtim.Circuit(xtim.scale_noise(c.text, p))
    #  when the weights should track p; [:, :n_decode] drops the 6 end-SE columns)

tasks = [xtim.Task(c, p=p, shots=2_000, seed=7, keep=my_keep,
                   decoder=my_decoder, metadata={"p": p})
         for p in (3e-4, 1e-3, 3e-3)]
rows = xtim.collect(tasks)
for row in rows:
    print(f"p={row['p']:g}  acc={row['acceptance_rate']:.3f}  "
          f"F={(1 + beta @ row['value']) / 2:.4f}")
assert rows[0]["acceptance_rate"] >= rows[-1]["acceptance_rate"]
```

To get fidelity directly, set `target_k` (the logical-qubit count of the target
magic state) on each Task: each row then carries `fidelity`, `infidelity`
(=1−F), and `infidelity_sem` — a publishable 1−F ± sem vs p curve with no
manual β arithmetic. Declare every Pauli with nonzero ideal weight (completeness
is yours to ensure; an undeclared column is silently dropped, biasing F).

```python
tasks = [xtim.Task(c, p=p, shots=20_000, seed=7, target_k=1) for p in (3e-4, 1e-3, 3e-3)]
rows = xtim.collect(tasks)
for row in rows:
    print(f"p={row['p']:g}  1-F={row['infidelity']:.2e} ± {row['infidelity_sem']:.1e}  acc={row['acceptance_rate']:.3f}")
```

`collect(tasks, num_workers=N)` fans the tasks over a fork-start
multiprocessing pool: results are bit-identical to a serial run (each task is
one self-seeded engine stream, returned in task order), and because workers are
forked, closures and lambdas work fine as `keep`/`decoder` — nothing
user-supplied is pickled on the way in (`metadata` does come back pickled, so
keep it plain data). Where fork is unavailable (non-POSIX), collect warns and
runs serially.

## Building a circuit programmatically

A circuit doesn't have to come from a file. `xtim.Circuit` mirrors Stim's
construction surface — `append`, `+`/`+=`, `*` (a `REPEAT` block), `to_file`,
`copy`, `len()`, and the full `num_*` set — so a Stim user's muscle memory
transfers. Every mutation re-parses, so you can never assemble an unparseable
circuit (a bad `append` raises `XtimParseError` and leaves the circuit
untouched).

```python
prep = xtim.Circuit("")
prep.append("H", [0, 1])
prep.append("CX", [0, 1])
prep.append("X_ERROR", 0, 1e-3)        # parens arg = noise probability
prep.append("M", [0, 1])
prep.append("DETECTOR", "rec[-1]")     # raw target text or stim.target_rec(-1)

round_ = xtim.Circuit("X_ERROR(1e-3) 0\nM 0\nDETECTOR rec[-1]")
program = prep + round_ * 3            # concatenate; '* 3' is a REPEAT 3 block

assert len(program) == 6              # 5 from prep + the one REPEAT block
assert program.num_measurements == prep.num_measurements + 3
print(program)                        # str(circuit) is the Stim text
```

## Where the pieces live

- `c.expectation_columns` — the ordering contract, computed from the parse.
- `c.detector_error_model(include_expectations=False)` — the documented
  opt-out: the detector/observable-only DEM (the classic Stim question) for
  circuits whose expectations are not DEM-expressible. The default refuses
  loudly (`XtimDemError`, CLI exit 4) naming the column and the mechanism,
  never approximating.
- Acceptance/correction rules for the committed benchmark studies (frame
  parities, end-syndrome lookups) are *decoding policy*: they live with the
  benchmarks as private tooling configs (`benchmarks/targets/*.json`), not in
  the xtim API — `tests/test_tour.py` reproduces the committed cs/d3
  LER results through exactly the lean path above.
