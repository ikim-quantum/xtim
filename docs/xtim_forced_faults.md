# Forced faults, exact branches (3.1.11)

"What exactly happens if *these* faults fire?" — for any explicit fault set, every
measurement-outcome branch of nonzero probability, with its exact probability, records,
detectors, observables, decisions and `PAULI_EXPECTATION` values. No sampling.

```python
import xtim

sim = xtim.Circuit(text).compile_exact_branches()   # compile ONCE
sim.noise_sites()        # [{'channel': 'DEPOLARIZE1', 'qubits': [0, 1], 'probs': [0.001]}, ...]

br = sim.exact_branches([(3, 0, "X"),            # noise site 3, its target 0, Pauli X
                         (7, 1, "XZ"),           # a DEPOLARIZE2 site: pair 1, X on its first qubit, Z on its second
                         {"record": 5}])         # flip the RECORDED bit of measurement record 5
br.probabilities          # float64[B], sums to 1 (exact, 1e-12)
br.measurements           # bool[B, M]  recorded bits (`!`, forced flips, feedback applied)
br.detectors              # bool[B, D]  reference-relative (Stim detector-sampler semantics)
br.observables            # bool[B, O]  reference-relative
br.decisions              # bool[B, K]  raw recorded parity of each DECISION line
br.expectations           # float64[B, R]  each PAULI_EXPECTATION, conditioned on the branch
```

Many queries per compile are cheap: nothing is recompiled; each query composes the faults'
propagated errors and resolves the reads once.

## Addressing: noise sites

A fault names a **noise site**: one noise instruction of the circuit (`X_ERROR`, `Y_ERROR`,
`Z_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`), numbered
0, 1, … in order of appearance with `REPEAT` blocks unrolled (`sim.noise_sites()` lists them).

| form | meaning |
|---|---|
| `(site, target, "X")` or `{"noise": site, "target": t, "pauli": "X"}` | 1-qubit channel: Pauli X/Y/Z on the site's `target`-th qubit |
| `(site, target, "XZ")` | 2-qubit channel: the `target`-th qubit PAIR; first letter on its first qubit, second on its second; letters over IXYZ, not `II` |
| `{"record": j}` | flip the recorded bit of absolute measurement record `j` (the `M(p)` readout flip; the state is NOT flipped); any measurement may be named |

Every other noise channel and every readout coin is forced **not** to fire. Any Pauli may be
forced at a site: the site only fixes the *position* (to address a position in a noiseless
circuit, put a noise instruction there — the probability is irrelevant). One fault per
(site, target) and per record; a repeat is an error.

**Why noise sites (and not free Pauli insertions).** A site's faults are exactly the events
the sampler fires there: the same propagation-table atoms, the same composition order. So
`exact_branches(F)` is *by construction* the circuit's sampled law conditioned on the fired
set `F` (the tests check exactly that against the sampler), with no second propagation
machinery to keep consistent. Free insertion points would need new propagation entries for
arbitrary positions; a noise instruction at the position gives the same thing for free.

## Engine and exactness

The compiled program is the exact sampler's (`compile_detector_sampler(engine="exact")`):
normalize/defer (every measurement becomes a terminal read), the propagation table, the bare
state. A query

1. composes the faults' end-of-circuit errors E (diagonal S/CZ/X atoms by the canonical
   `diag_conjugate` rules; controlled-Hadamard-class atoms by their general tableau),
2. conjugates every terminal read and declared expectation to E†·P·E,
3. resolves the reads in record order on a clone of the bare state, **branching wherever both
   outcomes have conditional probability > 1e-12** (the engine's own coin threshold), +1
   branch first (depth-first order),
4. per leaf: the expectations on the post-read state, then the sampler's record tail made
   deterministic (`!` inverts, the forced flips, feedback relabel, declared expectation
   frames, detector/observable channels vs the noiseless reference).

This covers every circuit the exact sampler accepts: T/CS/CCZ/CH anywhere (not only at the
end of a qubit's life), mid-circuit measurements after magic, Pauli feedback. The branch
probabilities sum to 1 (checked; a violation raises `RuntimeError`).

## Refusals (loud, `ValueError`)

- A malformed address, an out-of-range site/target/record, a repeated location.
- Noise-site faults on a circuit where **coherentized feedback** (a `CX rec[-k] q` whose
  byproduct crosses a non-Clifford gate) reads an `M(p)` measurement: normalization inserts
  readout-flip noise there, so the sites cannot be matched 1:1 (`sim.noise_refusal` says so;
  e.g. the bundled `cultivation_d5`).
- Record flips on a circuit with coherentized feedback (its coherent control would read the
  true outcome, not the flipped record; `sim.record_refusal`).
- More than `max_branches` (default 2^20) branches.

## Read-only

A query consumes no RNG and mutates nothing: every sampled stream (exact, twirl, DEM export)
is byte-identical to 3.1.10.

## Verification

- `tests/test_forced_branches.py`: vs an independent dense statevector oracle
  (`tests/_forced_oracle.py`, own numpy) on random circuits with T/CS/CCZ, mid-circuit
  measurements, MR resets, `!`/M(p), feedback, faults before/after the magic: branch sets,
  probabilities (1e-12), decisions, expectations and detector/observable parities.
- `tests/test_forced_vs_sampler.py`: vs the existing exact sampler on the bundled examples with
  the faults written in as Pauli gates / `M(1)` (every sampled shot is an enumerated branch with
  identical channels; frequencies match the exact probabilities), and vs Stim on Clifford
  memories (branch set and uniform probabilities, detectors vs the unfaulted reference).
