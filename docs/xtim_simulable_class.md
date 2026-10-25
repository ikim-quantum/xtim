# Which circuits can xtim simulate?

The answer has one sentence, one condition, and one cost knob.

**The sentence:** xtim simulates any circuit that is Clifford-equivalent to a *single mutually-commuting layer of
π/8 Pauli-product rotations* ("generalized T-depth 1"), under Pauli noise.

**The condition:** take every non-Clifford gate in your circuit (`T`, `CS`, `CCZ`, `CH` and their daggers), expand
it into its π/8 rotations, and commute each one back through the Cliffords that precede it (its rotation axis, a
Pauli product, gets conjugated along the way). If all the folded axes **mutually commute**, xtim accepts the
circuit. If any two anticommute, it refuses by name (`xtim.XtimTwirlRefusal`, `twirl_class == OUTSIDE_T_DEPTH_1`).

**The cost knob:** χ (stabilizer rank) = 2^rank of that folded layer. One magic state gives χ ≤ 2; multi-magic
outputs (the [[8,3,2]] CCZ state has χ = 8) cost more per shot. Rank, not gate count or depth, is what you pay.

---

## 1. What this means in practice

- **You do not have to write the circuit as one layer.** The engine folds for you. A cultivation protocol with a
  magic prep followed by k check rounds has non-Clifford gates at k+1 depths; it is depth 1 *after folding*.
- **Diagonal magic (`T`, `CS`, `CCZ`) usually commutes**, because the axes are Z-type and `CX` / `CZ` keep them
  Z-type. The 15-to-1 distillation and the [[8,3,2]] CCZ compile at small χ.
- **Cliffords between magic gates can rotate axes into conflict.** `T; H; T` on one wire is refused: the H turns the
  second T's axis into X against the first's Z. `T; H; T; T` is accepted (the trailing pair merges into a Clifford).
- **Non-diagonal magic: the controlled-Hadamard (`CH`)** contributes Y-type axes and is accepted when those commute
  with the rest of the circuit's magic. Errors crossing a CH pick up Clifford (not just Pauli) content; the engine
  propagates it exactly.

## 2. Example: measuring the Hadamard (H-eigenstate cultivation)

The Hadamard test (ancilla |+⟩, `CH` onto the data, `MX` the ancilla) measures the Hadamard operator on the data.
Repeated, it *cultivates* the H-eigenstate |H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩.

The class condition shows up as a physical requirement: **the raw magic state must be Clifford-aligned with the
check that measures it.** The raw magic is T-form; rotate it onto the Hadamard axis before the checks:

```
# |H> = (H S H) T H |0>          the +1 eigenstate of H
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
OUTPUT_QUBITS out 0
```

Aligned, the prep's folded axis and the CH's axes coincide: commuting magic, χ = 2 for any number of rounds, and
every check detector is deterministic (post-selectable). Unaligned (a raw T state straight into a same-wire CH), the
T's Z axis anticommutes with the CH's Y axis and the circuit is refused: the class boundary is the protocol's
alignment requirement. The runnable protocol ships as the `ch_cultivation` example
(`examples/04_h_cultivation.py`).

## 3. When xtim refuses

The sampler refuses at compile time with `xtim.XtimTwirlRefusal` (a `ValueError`); `twirl_class` names the reason.
Nothing is approximated and there is no fallback.

| `twirl_class` | what it means |
|---|---|
| `OUTSIDE_T_DEPTH_1` | the folded π/8 axes anticommute: the circuit is outside the class (the condition above). The message names the two axes in your qubits and the non-Clifford gates on them |
| `NONCOMMUTING_MAGIC` | in the class, but two same-axis magic terms anticommute through a *fault* between them, so the surviving axis depends on the faults (e.g. two `CH` on one wire with noise between them). A law limitation, not physics |
| `LAW_LIMITATION_PROPAGATION` | in the class, but a fault meets a same-axis non-Clifford pair before the pair merges. A law limitation |
| `LAW_LIMITATION_CARRIED_AXIS` | a carried producer state (`compile_twirl_sampler_from_ideal_state`) whose law the engine does not build yet |
| `EXPECTATIONS_NOT_SERVED` / `RAW_RECORDS_NOT_SERVED` | not a circuit property: `sample(return_expectations=True)` / `sample(return_measurements=True)`. Score the end state with `OUTPUT_QUBITS` + `RecordFidelity` |

Common shapes of `OUTSIDE_T_DEPTH_1`:

| you see | what happened | what to do |
|---|---|---|
| the two axes sit on a wire read in the X/Y basis after an explicit `H` | a magic-carrying wire read via a mid-circuit basis change | write `MX` / `MY` directly |
| the axes meet at a `CH` | a magic state's axis anticommutes with the check measuring it | Clifford-align the state to the check's axis first (§2) |
| the axes are spread over many gates | the folded layer genuinely does not commute | the circuit has T-depth ≥ 2 after folding: outside the class (§4) |

`detector_error_model()` can refuse a circuit the sampler accepts (`xtim.XtimDemError`): its faults have no
Pauli decoder model (e.g. faults crossing a `CH`, or a non-deterministic detector). Sampling and post-selection are
unaffected.

## 4. Outside the class

A circuit whose folded π/8 axes genuinely do not commute (true T-depth ≥ 2, e.g. a random `T`/`H` sequence) is not
simulable by xtim at any χ. Options: restructure the protocol so the magic commutes (as §2 shows, this is often what
the physics wants), or use a statevector / tensor-network simulator at small size.
