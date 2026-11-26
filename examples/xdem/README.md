# xdem examples

Pairs of an xtim circuit (`<name>.stim`, self-contained) and the file
`xtim.Circuit(text).detector_error_model_text(format="xdem")` writes for it with the default options
(`<name>.xdem`). The format is described in [docs/xdem.md](../../docs/xdem.md). Every file uses the `.xdem` suffix,
also when nothing in the circuit rotates and the output is a conventional DEM (then it is byte for byte the stim-format
DEM, `detector_error_model_text()`).

Columns: axis lines; rotating error lines (a `rot` or `late` field); error lines with a `late` field; `tc` against the
number of error lines (`tc` below the count means some lines are after `t_c`); lossy readout (the genie decoder under
`correction="optimal"`, `counters()["lines_lossy"]`, 2e4 shots); file lines; bytes. A conventional file has `-` in
every column from axes to lossy; an extended file with no observable has `-` for lossy.

| name | circuit | xdem | axes | rotating | late | tc / lines | lossy | lines | bytes |
|---|---|---|---|---|---|---|---|---|---|
| `cultivation_d3_faithful` | d=3 magic-state cultivation (T), bundled example | extended | 15 | 357 | 0 | 471 / 471 | no | 487 | 35820 |
| `cultivation_d5` | d=5 magic-state cultivation, bundled example | extended | 72 | 2593 | 0 | 3243 / 3243 | yes, 2.3 % of shots | 3316 | 455647 |
| `color5_cultivation` | SOFT's d=5 colour-code cultivation (Apache-2.0), bundled example | extended | 72 | 2593 | 2593 | 3243 / 3243 | - | 3316 | 511182 |
| `miniature_oracle` | a T state teleported into a 3-qubit repetition code, bundled example | extended | 3 | 16 | 0 | 29 / 29 | - | 33 | 1406 |
| `ch_cultivation` | H-eigenstate cultivation with CH (two π/8 Y-rotations around a CZ in the xdem), bundled example | extended | 4 | 9 | 0 | 9 / 9 | - | 14 | 492 |
| `distillation_15_1_3` | 15-to-1 distillation, Z noise only, bundled example | conventional: byte for byte the stim-format DEM | - | - | - | - | - | 15 | 291 |
| `cultivation_d3_rate` | the Clifford twin (S) of `cultivation_d3_faithful`, bundled example | conventional: byte for byte the stim-format DEM | - | - | - | - | - | 366 | 17198 |
| `distillation_15_1_3_gatenoise` | `distillation_15_1_3` with `DEPOLARIZE1(0.001)` after the H layer and before each of its 14 MPPs (variant) | extended | 15 | 89 | 0 | 151 / 151 | - | 167 | 9803 |
| `code_switching_faithful_noobs` | qRM to Steane code switching, observables removed (variant) | extended | 15 | 378 | 0 | 670 / 670 | - | 686 | 39933 |
| `code_switching_rate_noobs` | its count-matched twin, observables removed (variant) | extended | 15 | 170 | 0 | 302 / 302 | - | 318 | 17507 |
| `ex422_obs` | [[4,2,2]] code, X, Z, Y faults on one data qubit, both observables (the worked file of docs/xdem.md) | extended | 8 | 2 | 0 | 3 / 3 | no | 12 | 496 |
| `toy2obs` | two rep-3 blocks, transversal T, late Z faults, two observables | extended | 6 | 12 | 12 | 20 / 22 | no | 29 | 845 |
| `rm15rep` | a 15-qubit RM block and a rep-3 block, transversal T, late Z faults, two observables | extended | 33 | 21 | 21 | 40 / 41 | no | 75 | 3243 |
| `ccz_toy` | CCZ twice with X noise around it, all three qubits read as detectors | extended | 7 | 3 | 0 | 6 / 6 | - | 14 | 475 |
| `ccz_toy_obs` | `ccz_toy` with the last readout an observable instead of a detector | extended | 7 | 3 | 0 | 6 / 6 | no | 14 | 475 |
| `ccz_toy_obs_p5` | `ccz_toy_obs` at p = 0.05 | extended | 7 | 3 | 0 | 6 / 6 | no | 14 | 571 |
| `tdag_cycle` | one qubit through T, T_DAG, T, T_DAG with X noise between, read out in X | extended | 1 | 2 | 0 | 4 / 4 | - | 6 | 208 |

The bundled examples are copies of `examples/<name>.stim`, the third-party ones with their headers (see
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)). The variants are written out in full, with a comment line naming the change (the
`tests/_xdem_circuits.py` variants): `_gatenoise` adds `DEPOLARIZE1(0.001)` after the H layer and before each of the 14
MPPs, and nothing else (T and T_DAG get no noise, and the only two-qubit gates are classically controlled `CX rec[-k] q`,
which it skips); `_noobs` removes the `OBSERVABLE_INCLUDE` lines (with them the
export is refused, below). The small circuits are the xdem test circuits of `tests/data/xdem/circuits/` and
`tests/test_xdem_export.py`.

## Refused

These bundled examples have no xdem; the export raises `xtim.XtimDemError` with a message starting:

- `code_switching_faithful`: `not DEM-expressible: non-deterministic observable L0`
- `code_switching_rate`: `not DEM-expressible: non-deterministic observable L0`

`cube_ccz` is noiseless: its export is the empty string.

## Regenerate

```
python examples/xdem/regenerate.py
```

rewrites every `.xdem` from its `.stim`. `tests/test_xdem_examples.py` checks the committed files are the current
export.

## Use a pair

```python
import xtim
s = xtim.XdemSampler(open("examples/xdem/cultivation_d3_faithful.xdem").read())
dets, status = s.sample(1000, seed=1)                                                # detector bits + per-shot status
r = s.infidelity(100_000, lambda d: [[] for _ in range(len(d))], seed=2, lines=True)  # the zero decoder
print(r.mean, r.stderr)
```
