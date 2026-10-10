# Which circuits can xtim simulate?

The first question everyone asks. The answer has one sentence, one condition, and one
cost knob.

**The sentence:** xtim exactly simulates any circuit that is Clifford-equivalent to a
*single mutually-commuting layer of π/8 Pauli-product rotations* — "generalized T-depth 1".

**The condition:** take every non-Clifford gate in your circuit, expand it into its π/8
rotations, and commute each one back through the Cliffords that precede it (its rotation
axis, a Pauli product, gets conjugated along the way). If all the folded axes **mutually
commute**, xtim accepts the circuit — compile, noise, sampling, everything. If any two
anticommute, it rejects, loudly. That is the *only* rejection.

**The cost knob:** χ (stabilizer rank) = 2^rank of that folded layer (after T-count
reduction). One magic state ⇒ χ ≤ 2 and millions of shots per second; multi-magic
outputs cost more per shot but are still exact. Rank — not gate count, not circuit
depth — is what you pay.

---

## 1. What this means in practice

- **You do not have to write the circuit as one layer.** The engine folds for you. A
  cultivation protocol with a magic prep followed by k check rounds has non-Clifford
  gates at k+1 different depths — it is depth-1 *after folding*, and accepted.
- **Diagonal magic (`T`, `CS`, `CCZ`) usually commutes**, because their axes are Z-type;
  interleaved `CX`/`CZ` keep them Z-type. Circuits like the 15-to-1 distillation or the
  [[8,3,2]] CCZ compile at small χ.
- **Cliffords between magic gates can rotate axes into conflict.** `T; H; T` on one wire
  rejects: the H turns the second T's axis into X against the first's Z. `T; H; T; T`
  accepts (the trailing pair merges into a Clifford). The dense rule is the folded-axes
  test, not a syntactic pattern.
- **Non-diagonal magic is supported since the controlled-Hadamard (`CH`) landed**: CH
  contributes Y-type axes, and it is accepted exactly when those commute with the rest
  of the circuit's magic. Errors crossing a CH are handled exactly too (they pick up
  Clifford — not just Pauli — content; the engine propagates it and conjugates the
  measurements through it, sign-exactly).

## 2. The flagship example: measuring the logical Hadamard (H-eigenstate cultivation)

The Hadamard test — ancilla |+⟩, `CH` onto the data, `MX` the ancilla — measures the
Hadamard operator on the data wire. Used repeatedly, it *cultivates* the H-eigenstate
magic state |H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩.

One physical subtlety, and it is the class condition in disguise: **the raw magic state
must be Clifford-aligned with the check that measures it.** The raw physical magic is
necessarily T-form (T is the only single-qubit non-Clifford in the gate set); rotate it
onto the Hadamard axis before the checks:

```
# |H> = (H S H) T H |0>          — the +1 eigenstate of H
H 0
T 0
H 0
S 0
H 0
# one check round (repeat as desired)
H 1
CH 1 0        # control = ancilla, target = data
MX 1
DETECTOR rec[-1]
PAULI_EXPECTATION(0) X0
PAULI_EXPECTATION(1) Z0
```

Aligned, the prep's folded magic axis and the CH's folded axes *coincide* — commuting
magic, χ = 2 **independent of the number of rounds** (measuring H on an H-eigenstate
never entangles the ancillas), every check detector deterministic (post-selectable), and
`diagnose()` hands you the full recipe:

```
xtim diagnosis (noiseless run, 1024 shots)
====================================================
reference chi : not yet cached (deduced on the fly — fine; run any compile_* to cache it)
detectors     : 3 total — 3 deterministic, 0 gauge
expectations  : 2 declared PAULI_EXPECTATION column(s)
  [0] value=+0.707106781187  sign constant
  [1] value=+0.707106781187  sign constant
  target (signed beta): [+0.707107, +0.707107]
suggested post-selection (advisory; not applied):
  keep = ~dets[:, [0, 1, 2]].any(axis=1)
```

**Unaligned, it rejects — and should.** Feed a raw T-state straight into a same-wire CH
and the T's Z-axis anticommutes with the CH's Y-axis: physically, "apply T then
coherently rotate toward the H basis" has no single commuting-rotation description.
The class boundary *is* the protocol's alignment requirement. Corollary: don't mix
old-style controlled-XS check rounds and CH rounds on the same data wire.

The runnable protocol ships as the `ch_cultivation` example circuit.

## 3. When xtim rejects — what the message means, what to do

All rejections are `XtimRejectError` with `kind == "class"` and an actionable `.hint`.
The cause is always the same condition; the common shapes:

| you see | what happened | what to do |
|---|---|---|
| reject pointing at an `H` (or the hint mentions `H` then `M`) | you read a magic-carrying wire in the X/Y basis via an explicit mid-circuit basis change | write `MX`/`MY` directly (a *terminal* `H;M` pair is rewritten soundly and does not reject) |
| reject at a `CH` / "non-commuting magic" | a magic state's axis anticommutes with the check measuring it (e.g. raw T-state into a same-wire CH) | Clifford-align the state to the check's axis first (§2) |
| reject with no single gate index | the folded π/8 layer is non-commuting in a way that isn't localized to one gate | the circuit has genuine T-depth ≥ 2 after folding; not in the class (see §4) |
| `XtimDemError` from `detector_error_model()` on an accepted circuit | not a class reject: the circuit *samples* exactly, but its faults have no exact Pauli decoder model | use `diagnose()` + post-selection, or `detector_error_model_with_reject()` — see docs/xtim_dem_reject.md. CH-circuit faults carry Clifford (not Pauli) actions, so DEM export always refuses them; sampling and post-selection are unaffected |

## 4. Outside the class

A circuit whose folded π/8 axes genuinely don't commute — true T-depth ≥ 2, e.g. a
random `T`/`H` sequence, or magic teleportation chains that interleave non-commuting
rotations — is not simulable by the current engine at any χ. (The mathematics of an
*ordered* rotation sequence with χ growing per non-commuting layer is a natural
extension of the same machinery; it is future work, not a small flag.)

For those circuits: Stim (if you can Pauli-twirl the magic), a statevector/tensor-network
simulator at small n, or restructure the protocol so the magic commutes — which, as §2
shows, is often what the *physics* wants anyway.

## 5. Cost model, concretely (measured)

| circuit | χ | throughput |
|---|---|---|
| `ch_cultivation` (3 H-check rounds, p=10⁻³) | 2 | ≈10 M shots/s |
| `cultivation_d3_faithful` | 2 | ≈1 M shots/s |
| `distillation_15_1_3` (15-to-1) | 2 | ≈5 M shots/s |
| `cultivation_d5` | 2 | ≈50 K shots/s |

Error-carrying shots in a CH-class circuit take a slower exact path (~10 µs/shot at toy
sizes); at realistic noise rates they are a ~1% minority and invisible in the amortized
rate. Compile is milliseconds at protocol scale (~0.6 s for d5-class references), cached
across runs.

---

*Internals (how the folding, the residual algebra, and the byte-exactness guarantees
work): the engine-design record `docs/ch_ppr_design.md` in the development repository.*
