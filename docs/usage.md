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
| `RF_SECTOR_RANK` | `evaluate` | a shot's magic reaches more unrecorded directions than the evaluator's cap |
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
| `OUTPUT_QUBITS name q0 q1 ...` | the output register `RecordFidelity` scores |
| `DECISION(k) rec[-j]` | marks a record as decision bit `k`, returned per shot by `sample` and `buf.decisions()` |

Not supported: record-controlled Cliffords and `sweep[k]` controls (refused by name). A circuit's non-Clifford
gates must fold into one commuting layer of π/8 rotations; see [the simulable class](xtim_simulable_class.md).

## Internals

For contributors: [twirl_record_sampler.md](twirl_record_sampler.md) (the sampling engine's contract),
[branch_frames.md](branch_frames.md), [frame_cosets_cpp.md](frame_cosets_cpp.md),
[dem_export_readout_noise.md](dem_export_readout_noise.md), [xtim_forced_faults.md](xtim_forced_faults.md).
