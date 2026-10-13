# Forced faults, exact law (3.1.11; factored law: Unreleased)

"What exactly happens if *these* faults fire?" — for any explicit fault set, the exact law of
the circuit's detectors, observables and decisions, with the conditional `PAULI_EXPECTATION`
values. No sampling, and no read-by-read enumeration.

```python
import xtim

sim = xtim.Circuit(text).compile_exact_branches()   # compile ONCE
sim.noise_sites()        # [{'channel': 'DEPOLARIZE1', 'qubits': [0, 1], 'probs': [0.001]}, ...]

law = sim.exact_branches([(3, 0, "X"),            # noise site 3, its target 0, Pauli X
                          (7, 1, "XZ"),           # a DEPOLARIZE2 site: pair 1, X on its first qubit, Z on its second
                          {"record": 5}])         # flip the RECORDED bit of measurement record 5
# factored (always available)
law.rows_probabilities       # float64[B], sums to 1 (exact)
law.rows_detectors           # bool[B, D]  reference-relative (Stim detector-sampler semantics)
law.rows_observables         # bool[B, O]  reference-relative
law.rows_decisions           # bool[B, K]  raw recorded parity of each DECISION line
law.rows_expectations        # float64[B, R]  conditional <P> given the row's targets
law.uniform_detectors        # bool[u, D]  the u independent fair-coin directions ...
law.uniform_observables      # bool[u, O]
law.uniform_decisions        # bool[u, K]
law.uniform_expectation_signs  # bool[u, R]  ... and the columns each one flips the sign of
# expanded (built lazily on access; ExpandedLawTooLarge beyond max_rows)
law.probabilities, law.detectors, law.observables, law.decisions, law.expectations
len(law)                     # B * 2^u
```

**The law.** Pick a row with its probability; XOR any subset of the `u` directions into its
targets, every subset equally likely (each direction is an independent fair coin); a direction
flips the sign of the expectation columns it marks. The expanded view lists every
(row, subset) pair as its own row: `B * 2^u` rows, all with distinct targets, probability =
row probability / 2^u. `exact_branches(faults, expectations=None, max_rows=1<<20,
max_work=4e8)`: more than `max_rows` structured rows is refused by name (`ValueError`); the
expanded view (built only when one of its fields is accessed) beyond `max_rows` raises
`ExpandedLawTooLarge` (a `ValueError`) on access, pointing at the factored fields; `max_work`
caps the deterministic work of the exact walk over the magic-sector parities (refused by name
beyond, naming the magic rank — see below). `expectations=[]` evaluates no column.

**Expectations are conditional on the targets.** A row reports, per column, the expectation
of the declared Pauli (frames and feedback sign flips applied) CONDITIONED on the row's
(detectors, observables, decisions): the probability-weighted average of the per-shot values
over every measurement outcome with those targets — 0 for a column that anticommutes with a
terminal read. This is what a consumer applying a LINEAR functional of the expectations per
outcome of the targets needs, and it is what makes the row count independent of how many reads
are random. It is NOT the sampler's per-shot value (conditioned on every read): it is
conditioned ONLY on (detectors, observables, decisions); every other read — declared frame
records included — is averaged over. Example: `RX 0 1`, `M 0`, `PAULI_EXPECTATION(0) X1 rec[-1]`
(no detector): the sampler reports +1 or -1 per shot (the frame record's sign), while the law
has one row with expectation 0 — the average over the frame record, which is not a target.
Declare the frame record as a detector, observable or `DECISION` to condition on it.

**No raw records.** Per-branch measurement records are not returned (there is no
`measurements` field; the factored law never enumerates them).

Many queries per compile are cheap: nothing is recompiled; the record tail's linear forms and
the bare-state analyzer are built once per compiled program.

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
circuit, put a noise instruction there — the probability is irrelevant), with ONE exception: a
Pauli the site's channel gives probability ZERO whose propagated error leaves the simulable
class is refused by name (`ValueError`: "noise site s target t Pauli 'X' has probability zero
in its channel and its propagated error leaves the simulable class ..."). Such an atom is never
fired by the sampler, so it does not reject the circuit; only forcing it is refused. Example:
`Z_ERROR(0) 0; CS_DAG 1 0; SQRT_X_DAG 0; T_DAG 0; X 0; T 0` — forcing `(0, 0, "Z")` is answered,
`"X"` / `"Y"` are refused. One fault per (site, target) and per record; a repeat is an error.

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
   `diag_conjugate` rules; controlled-Hadamard-class atoms by their general tableau) and
   conjugates every terminal read and declared expectation to A = E†·P·E (the reads commute);
2. writes every target as an affine GF(2) form `f = c + S m` of the raw read bits `m`, and each
   column's sign as `(-1)^(e + u.m)`: `!` inverts, forced record flips and the feedback relabel
   (a control record, itself linear in the reads, XORs a constant flip mask) compose into the
   linear parts once per program; a query only shifts the constants;
3. **the stabilizer twirl.** In the bare state's frame `|chi>_free (x) |eps>_rest` a product of
   reads A_S has a nonzero expectation only when its X-part outside the magic (`free`) sector
   vanishes. So the target directions `S X` (X = the reads' non-free X-parts) are exactly the
   independent fair coins — the reduced column echelon of `S X` gives the `u` directions, each
   1 at its pivot target and 0 at the others; rows are the coset representatives zero on every
   pivot. For each remaining target the representative's read set has no non-free X-part: its
   parity is either a deterministic ±1 on the bare state or a function of at most `k` magic-
   sector parities (`k` = magic generators; the parities commute, so their free-sector parts
   are isotropic), whose exact joint law is walked on its SUPPORT only
   (`PeShotAnalyzer::joint_law`, the engine's `born_p1` / `measure_pauli` and coin threshold
   1e-12), reading each column's conditional value `<X A_w0>` on every support leaf;
4. a column: zero if it anticommutes with a read or its non-free X-part is outside the reads'
   span; otherwise the pivot-row solution `alpha` of that span fixes `w0` and the sign
   constant, and direction `j` flips the column iff `alpha_j = 1`.

Nothing is sampled or approximated; every bound is a count, never a clock — beyond any the
query refuses by name: structured rows <= `max_rows`, magic-sector parities <= 63, and the
magic-parity walk's WORK <= `max_work` (each Born probability / collapse / leaf read is charged
max((chi+1)·n·(k+1), (chi+1)^2) units; the default 4e8 is about one second on the reference
host — on k |T> states each read in X as a `DECISION`, k <= 8 answers and k >= 9 is refused,
naming the magic rank). The row
probabilities sum to 1 (checked; a violation raises `RuntimeError`). This covers every circuit
the exact sampler accepts: T/CS/CCZ/CH anywhere, mid-circuit measurements after magic, Pauli
feedback, `!` and `M(p)` records, nondeterministic detectors.

**The retired branch walk.** Until this release the reads were resolved in record order on a
clone of the bare state, branching on every random read: 2^(#random reads) leaves even when no
target depends on them. Before its removal the factored law was checked against it, aggregated
by (detectors, observables, decisions) with probability-weighted conditional expectations, on
25,087 queries (the dense random suite, the bundled examples, every single fault of
`cultivation_d5` including the folded record flips, the held-fold circuit, Stim memories,
ftqcp's qrm15 magic-t-prep Choi program): max |dp| 1.8e-15, max |dexp| 5.2e-15.

| query (quiet host) | factored law | retired walk |
|---|---|---|
| `cultivation_d3_faithful`, weight 1-3 | 36 us | 149 us |
| `cultivation_d5`, weight 1 | 152 us | 16.0 ms |
| dense random suite (mean) | 14 us | 21 us |
| d=3 memory + 40 `MX` ancillas (DECISION each) | 0.15 ms, 1 row, u = 40 | refused (> 2^16 branches) |

## Coherentized feedback: provenance addressing (Unreleased)

A feedback `CX/CY/CZ rec[-k] q` whose byproduct crosses a non-Clifford gate is coherentized:
it becomes a controlled-Pauli from the measured wire, and a noisy source measurement `M(p)` has
its readout flip FOLDED into one extra noise instruction (an anticommuting `X_ERROR`/`Z_ERROR`
on the control before the measurement, or — when the control is observed again — an `X_ERROR`
on a flip-instance ancilla applied to the control before AND after the measurement, so the
state is restored). Sites are therefore matched by **provenance**, not by order: the compiler
tags every user noise instruction with its site index and the coherentizer tags each fold
with its measurement record (`Instr::origin`, metadata only — no sampling/DEM/emit path reads
it). Site `i` is the normalized instruction tagged `i`; a record flip of a folded measurement
(`sim.folded_records`) fires the fold's Pauli, which flips the recorded bit and the coherent
control together (and is undone on the state where the control is observed again) — exactly
the fold's model of `M(p)`. Any other record flip flips the recorded bit in the record tail,
as before. The bundled `cultivation_d5` (7 folds) is fully addressable.

## Refusals (loud, `ValueError`)

- A malformed address, an out-of-range site/target/record, a repeated location.
- A record flip of a measurement that is read by coherentized feedback but is
  readout-NOISELESS (`M`, not `M(p)`): there is no fold channel to force, and a record-tail
  flip would not reach the coherent control (`sim.refused_records` names each such record).
- `sim.noise_refusal` / `sim.record_refusal` remain for a provenance mismatch (a normalized
  noise instruction of unrecognised origin, a site without exactly one image, an unresolved
  coherentized source); none is known to occur.
- More than `max_rows` (default 2^20) structured rows, more than `max_work` (default 4e8)
  work units in the magic-parity walk, or more than 63 magic-sector parities.

## Read-only

A query consumes no RNG and mutates nothing: every sampled stream (exact, twirl, DEM export)
is byte-identical to 3.1.10.

## Verification

- `tests/test_forced_branches.py`: the expanded law vs an independent dense statevector oracle
  (`tests/_forced_oracle.py`, own numpy) aggregated by (detectors, observables, decisions), on
  random circuits with T/CS/CCZ, mid-circuit measurements, MR resets, `!`/M(p), feedback,
  faults before/after the magic: probabilities and conditional expectations (1e-12), one
  constant detector/observable reference per circuit.
- `tests/test_forced_law.py`: the factored structure — random reads cost nothing (rows stay 1,
  directions = the random targets), exact biased magic rows and the `max_rows` refusal, the
  expanded view, and a circuit exercising directions that flip expectation signs together
  with magic rows vs the dense oracle on every single fault.
- `tests/test_forced_vs_sampler.py`: vs the existing exact sampler on the bundled examples with
  the faults written in as Pauli gates / `M(1)` (every sampled outcome is a row of the law,
  frequencies match the exact probabilities, sampled expectations averaged per row match the
  conditional expectations), and vs Stim on Clifford memories.
- `tests/test_forced_provenance.py`: coherentized feedback by provenance vs the inserted-circuit
  law (and the dense oracle on the held-fold circuit).
- `tests/test_forced_golden.py`: circuits too large for the dense oracle (`cultivation_d5`:
  50 single faults over the sites, the 7 folded flips, 10 weight-2/3 sets; ftqcp's qrm15
  magic-t-prep Choi program: 20 queries) vs a golden record of the retired walk
  (`tests/data/forced_walk_golden.json`, generated at 57e728a; generator source inside), 1e-12.
