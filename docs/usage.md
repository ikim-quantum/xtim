# Using xtim

The API in one page. Every block below runs on its own.

## Sampling

```python
import numpy as np
import xtim

sampler = xtim.compile_twirl_sampler(xtim.load_example("cultivation_d3_faithful").text, p_factor=1.0)
dets_b8, obs_b8 = sampler.sample(100_000, seed=1)       # Stim b8 bit-packed uint8[shots, ceil(n / 8)]
dets = np.unpackbits(dets_b8, axis=1, bitorder="little")[:, :sampler.num_detectors]
obs = np.unpackbits(obs_b8, axis=1, bitorder="little")[:, :sampler.num_observables]
```

- `p_factor` scales every noise probability in the circuit (0.0 is noiseless). `xtim.scale_noise(text, p)` rewrites
  a circuit written at p = 10⁻³ to p.
- Bits are relative to the noiseless reference, as in Stim. A circuit with `DECISION` lines returns a third array.
- Same seed, same bytes, however the shots are split across calls.
- `xtim.Circuit(text)` is the Stim-shaped interface (`compile_detector_sampler()`, `detector_error_model()`, ...).

## Post-selection and output fidelity

```python
import numpy as np
import xtim

sampler = xtim.compile_twirl_sampler(xtim.load_example("cultivation_d3_faithful").text)
buf = sampler.sample_barrier(100_000, seed=1)           # same detectors, plus each shot's output state
dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
accepted = ~dets.any(axis=1)
F = xtim.RecordFidelity(sampler).evaluate(buf)          # float64[shots]
print(f"accepted {accepted.mean():.1%}; 1-F accepted {1 - F[accepted].mean():.1e}")
```

- `evaluate(buf, shot_index=None, correction=None)`: one float per shot, per listed shot, or a float for one index.
- Each value is exact, so a post-selected 1-F is resolved far below 1/shots. A mean 1-F of exactly 0 means no
  scored shot had an output error; with N scored shots the error rate is below about 3/N (95% confidence).
- `RecordFidelity(sampler, max_bytes=...)` caps its caches; `clear()` drops them.
- For a circuit scored by `OBSERVABLE_INCLUDE`s rather than an output register,
  `xtim.postselected_logical_error_rate(circuit, p=..., shots=...)` returns the post-selected logical error rate
  ([details](xtim_postselected_logical_error_rate.md)).
- To score an output state, declare it with `OUTPUT_QUBITS` and use `RecordFidelity` — do not declare its Pauli
  expansion as `PAULI_EXPECTATION` columns (hundreds of columns make the DEM's per-shot column analysis slow, and
  the fidelity is what `RecordFidelity` computes exactly per shot).

## Closing a frame

`RF_OUTPUT_NOT_STRICT` names the output directions left undetermined and the measurements (0-based, in circuit order)
whose outcomes fix them. For each, add the feedback that undoes the byproduct its outcome 1 leaves: an X byproduct
on qubit `q` is undone by `CX rec[-k] q`, a Z byproduct by `CZ rec[-k] q`. With M measurements before the feedback
line, measurement m is `rec[-(M - m)]`. A Bell-measurement teleport leaves X from the second outcome and Z from the
first; a one-bit teleport (`CX` data to a fresh qubit, then `MX` the data) leaves Z. The message lists the output
Paulis involved, which need not be the byproduct itself. Check the fix with `p_factor=0.0`: F must be 1 on every
shot. See [`examples/08_bring_your_own.py`](../examples/08_bring_your_own.py).

## Decoded fidelity

A decoder's predicted observable flips become a Pauli correction on the output. Export the detector error model with
`xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True)` (a `stim.DetectorErrorModel`; the
circuit's observables are its first `num_observables` columns), decode, and give per observable the output Pauli
that undoes its flip (`duals`: `uint8[num_observables, 2, n_out]`, X and Z masks in `OUTPUT_QUBITS` order).
`xtim.correction_from_observables(sampler, predictions, duals)` returns `uint8[shots, 2, n_out]` for
`evaluate(buf, correction=...)`. The observables must be deterministic without noise; a magic output has none, so
the magic-state examples are scored by post-selection. See
[`examples/09_decoded_fidelity.py`](../examples/09_decoded_fidelity.py).

Pauli feedback (`CX`/`CY`/`CZ rec[-k] q`) exports too: feedback whose Pauli reaches a T/CS/CCZ/CH gate is made
coherent (as in the sampler), and the rest relabels the record sets of every column (detectors, observables,
decisions, expectation frames, `OUTPUT_QUBITS` frame columns), exactly; one relabel serves every fault (a fault's
Clifford residual commutes with a later-applied residual feedback Pauli up to sign: an invariant the export checks).
A PAULI_EXPECTATION declared before a feedback still carries that feedback's sign (the sampler's convention: the
feedback flips the reported sign whenever it fires and anticommutes with the declared Pauli). An `OUTPUT_QUBITS`
frame whose feedback controls are noiselessly random outcomes is refused when a fault randomizes one of them.

Known limit: the test that decides which feedback is made coherent looks only at each feedback's own Pauli. It does
not follow the Pauli through other feedback, so the choice is not transitive. Suppose a feedback whose own Pauli meets
no non-Clifford gate (so it is left as a record relabel) lands on the control qubit of a feedback that is made
coherent. That coherent gate carries the Pauli into a non-Clifford gate, and the Pauli stops being a Pauli correction.
The export then refuses by name (`` `CY rec[-1] q` is not coherentized ... reaches the control of a feedback that IS
coherentized ``), as `detector_determinism` does. Such a circuit could be served by making the first feedback
coherent too. The export does not do that yet.

A circuit with T/CCZ gates exports each Clifford fault as graphlike lines by default (`clifford_cutoff=2`: its
patterns on <= 2 detectors), which a matching decoder can use directly:

```python
import math, pymatching
dem = circuit.detector_error_model()                       # default clifford_cutoff=2
matcher = pymatching.Matching.from_detector_error_model(dem)
exact = circuit.detector_error_model(clifford_cutoff=math.inf)   # the exact expansion (hyperedges)
```

The cutoff drops patterns and can lower the suppression order (a fault whose fixed part already flips 3 or more
detectors vanishes); the dropped mass is reported (next section). `clifford_cutoff=math.inf` (or `None`) keeps every
pattern.

### Clifford faults in the DEM (`clifford_cutoff`)

A fault before a T or CCZ gate leaves a Clifford error (an S or CZ residue) rather than a Pauli. Twirled over the
circuit's stabilizer, its effect on the DEM targets is uniform on an affine space `c + W` of dimension `v` (an
unheralded erasure): with probability `p` it flips `v` independent fair coins on fixed signature vectors, plus a
fixed offset when `c` is not in `W`. Each such fault is its own record in the DEM text, never merged with another
fault. For an X fault fanned out by CX onto two T-rotated reads, with its own wire also read (`v = 2` plus a
deterministic offset on `D2`):

```
R 0
X_ERROR(0.02) 0
RX 1 2
T 1 2
CX 0 1
CX 0 2
T_DAG 1 2
MX 1 2
DETECTOR rec[-2]
DETECTOR rec[-1]
M 0
DETECTOR rec[-1]
```

exports at the default `clifford_cutoff=2` (the Pauli mechanisms of other faults, as stim would print them, come
first; the exact export, `clifford_cutoff=math.inf`, also keeps `D0 D1 D2`, with `cutoff=inf dropped_mass=0`):

```
# clifford faults=1 cutoff=2 dropped_mass=0.0050000000000000001
# clifford f0: p=0.02 v=2 offset=1 qW=-5.2075197289924888e-05 support=D0,D1,D2 loc=X_ERROR@0/site0 alt=X3 kept=3 dropped=0.0050000000000000001
error[clifford:f0](0.0050767996160234466) D0 D2
error[clifford:f0](0.0050767996160234466) D1 D2
error[clifford:f0](0.0050767996160234466) D2
```

The header names the fault (`loc` = channel@deferred stream index/user noise site, `alt` = the Pauli alternative on
its deferred qubits; `|` joins identical laws of one channel instance, merged), its probability `p`, dimension `v`,
whether it has an offset (`qW` = the omitted correction), every target it can flip, and the kept count and dropped
mass. Every line of a record is an independent mechanism at the same closed-form probability. The **weight** of a pattern
is the number of detectors it flips (observable, expectation, frame and decision columns never count).
`clifford_cutoff=t` keeps the patterns of weight `<= t` (observable-only patterns always):

- `2` (default): graphlike lines a matching decoder can use directly (at the cost above);
- `math.inf` (or `None`) = every pattern (`2^v - 1` lines, or `2^v` with an offset): the exact
  independent-mechanism split, capped at `v <= 12` (a larger fault is refused by name; the refusal names the exact
  request as the cause and a finite `clifford_cutoff` as the way out);
- an int `t >= 0`: the cap on `v` is lifted for these laws; a fault is refused only past `2^12` candidate patterns.

The offset's detectors count toward the weight (`D0 D1 D2` above has weight 3 and is dropped although `v = 2`), so
the exact split needs `t` at least the largest pattern weight, which can exceed `v`. Kept patterns are first-order
exact: a fault with an offset omits a negative `O(p^2 / 2^(v+1))` correction. Each
detector's marginal from one fault is below `p/2`. A dropped pattern of weight `w` is still produced by about
`ceil(w/t)` kept mechanisms, so a cutoff can lower the suppression order of the model. **Where the dropped mass is
reported:** the `# clifford ... dropped_mass=` summary line that opens the DEM text,
`detector_error_model_with_reject(...).clifford_dropped_mass` (with `.clifford_faults`), and the binding dict keys
`clifford_dropped_mass` / `clifford_faults`. `detector_error_model()` returns a `stim.DetectorErrorModel`, which keeps
the `[clifford:f<id>]` tags but drops `#` comments, so the report is not on that object. No warning is raised.
**How a Clifford fault's law is computed (twirl-routes).** One routine serves the DEM and the samplers: the fault's
record law on the targets it randomizes is the sampler's own record table, `f = c + sum_k u_k W_k + sum_j z_j B_j`
(`u` fair coins, `z` the Born bits of its `kappa <= log2(chi)` kernel-logical directions). The coin part is GF(2) only (the
stabilizer twirl over the certified group: no Born sum, no size guard — the former "Fourier support of dimension > 20"
and "> 64 targets" refusals are gone); Born sums are needed only on the `kappa` directions (`2^kappa` leaves). With
`kappa = 0`, or a uniform Born quotient, the law is a uniform coset (the record above); otherwise a Born mixture, solved
by classes of its patterns modulo the coin span. PAULI_EXPECTATION columns' per-shot laws use the same table (their
coin parities are GF(2) rows, never branched on); their Born walk has no work cap (`pe_max_work` was removed in this
release). **The Born law's cost (explaw).** A law of `d` commuting directions on the bare free register (support `s`,
`k` coordinates, `chi <= 2^k`) is evaluated by ONE routine that picks, per law and upfront from `(s, d, d_X, the coset
census)`, the cheaper of two exact evaluators with identical leaf semantics: the sequential projection walk
(`Theta(2^d x s)` on a dense hull; always for `d <= 2`) or the coset-of-X-span law — the support splits into cosets
of the directions' X-span `A` (`d_X = dim A`), the pure-Z directions are constants per coset, the mixed ones are
diagonalised per coset by an `S`/`CZ` layer and a Walsh–Hadamard transform, and the column (if any) is diagonal or a
coset swap in that basis: `O(s d + sum_C min(m_C^2, d_X 2^d_X) + #classes d_X 2^d_X + d 2^d)`. Nothing is exponential
beyond the `2^d` output table (`d <= k`). The k = 13 / 15 / 17 T-ladders export in 0.01 / 0.04 / 0.17 s (were 6.6 s,
109 s, ~40 min); the per-coset sums are accumulated in double-double, so the table's rounding never creates a leaf the
projection walk would not visit (same leaf sets; DEM texts of `d >= 3` laws byte-identical on the pinned circuits).
A Clifford fault whose law is not a uniform coset (an explicit-support law, header `law=support`) is solved exactly on
its own; when it has no independent split it is emitted as its disjoint law (each pattern at its probability,
first-order exact, header `first_order=1`). The Pauli alternatives of a channel instance that also has Clifford
alternatives are solved alone (independent instead of disjoint: an `O(p^2)` change), and emitted as their disjoint law
when they do not factor. `approximate_disjoint_errors` applies to all-Pauli channels (stim's rule); it does not change
a Clifford record (only one whose closed form over-mixes, `p >= 1/2`). The CLI takes `--clifford-cutoff T|inf`.

## Refusals

A refusal means the request is outside xtim's model; nothing is approximated silently. Every refusal is an
`xtim.XtimError` and a `ValueError`. [`examples/07_refusals.py`](../examples/07_refusals.py) triggers each.

| name | raised by | meaning / fix |
|---|---|---|
| `RF_NO_OUTPUT` | `RecordFidelity` | no `OUTPUT_QUBITS`: declare the output register |
| `RF_OUTPUT_NOT_STRICT` | `RecordFidelity` | the noiseless output is not one fixed pure state: close the frame (above) |
| `RF_OUTPUT_MEASURED` | `compile_twirl_sampler` | an output qubit is measured: move it to a fresh wire (`R` + `SWAP`) |
| `RF_RECORD_ON_OUTPUT` | `RecordFidelity` | a measurement acts on an output qubit: as above |
| `RF_CORRECTION_OFF_OUTPUT` | `evaluate` | a correction acts off the output: pass `uint8[shots, 2, n_out]` |
| `RF_SECTOR_RANK` | `evaluate` | a shot's magic reaches more unrecorded directions than the evaluator's cap (checked only on shots that can have F > 0; a shot orthogonal to the reference scores 0) |
| `OUTSIDE_T_DEPTH_1` | `compile_twirl_sampler` | the non-Clifford gates do not fold into one commuting π/8 layer ([class](xtim_simulable_class.md)) |
| `NONCOMMUTING_MAGIC`, `LAW_LIMITATION_*` | `compile_twirl_sampler` | magic axes that commute only fault-dependently |
| `EXPECTATIONS_NOT_SERVED`, `RAW_RECORDS_NOT_SERVED` | `sample(...)` options | not a sampler output: score the output with `RecordFidelity` |
| `XtimDemError` | `detector_error_model` | no Pauli decoder model (e.g. a non-deterministic observable): post-select instead |
| `XtimCapacityError` | compile | stabilizer rank above the engine limit (2^30) |

`RF_*` refusals are `xtim.XtimFidelityRefusal` (`.code` is the name); the compile-time classes are
`xtim.XtimTwirlRefusal` (`.twirl_class`).

## Circuit language

Stim's language (every Clifford gate, noise channels including `E`/`ELSE_CORRELATED_ERROR` chains, `M`/`MX`/`MY`/`MR`,
`M(p)`, `!` inverted targets, `MPP`, `MXX`/`MYY`/`MZZ`, `SPP`, `DETECTOR`, `OBSERVABLE_INCLUDE`, `REPEAT`,
coordinates), with Stim's semantics, plus:

| instruction | meaning |
|---|---|
| `T q`, `T_DAG q` | π/8 phase gate and its inverse |
| `CS a b`, `CS_DAG a b` | controlled-S (symmetric) |
| `CCZ a b c` | controlled-controlled-Z |
| `CH c t` | controlled-Hadamard (control `c`, target `t`) |
| `CX`/`CY`/`CZ rec[-k] q` | Pauli feedback: apply the Pauli to `q` iff the recorded bit is 1 (Stim's semantics) |
| `OUTPUT_QUBITS name q0 q1 ...` | the output register `RecordFidelity` scores (score an output state this way, not by declaring its Pauli expansion as `PAULI_EXPECTATION` columns) |
| `DECISION(k) rec[-j]` | marks a record as decision bit `k`, returned per shot by `sample` and `buf.decisions()` |

Not supported: record-controlled Cliffords and `sweep[k]` controls (refused by name). A circuit's non-Clifford
gates must fold into one commuting layer of π/8 rotations; see [the simulable class](xtim_simulable_class.md).

## Internals

For contributors: [twirl_record_sampler.md](twirl_record_sampler.md) (the sampling engine's contract),
[branch_frames.md](branch_frames.md), [frame_cosets_cpp.md](frame_cosets_cpp.md),
[dem_export_readout_noise.md](dem_export_readout_noise.md), [xtim_forced_faults.md](xtim_forced_faults.md).
