# Opt-in logical-fidelity helpers — `extract_frame` and `fidelity_from_logicals`

> **These are opt-in helpers, not the default path.**  The standard way to
> measure the post-selected logical error rate of a magic-state-prep circuit
> is `Circuit.postselected_logical_error_rate`, which scores the
> `PAULI_EXPECTATION` columns the circuit already declares.  Use these helpers
> when you need to (a) audit or construct a byproduct frame for a logical
> operator you choose, or (b) compute the TRUE full-support infidelity rather
> than only the subset of Paulis a particular circuit happens to declare.
>
> See also: [xtim_postselected_logical_error_rate.md](xtim_postselected_logical_error_rate.md)

---

## `xtim.extract_frame` — byproduct-frame lookup for any logical Pauli

```python
offsets = xtim.extract_frame(circuit, operator, *, minimal=True)
# equivalently:
offsets = circuit.extract_frame(operator, minimal=True)
```

`operator` is a Pauli product in `PAULI_EXPECTATION` grammar, e.g.
`"X15*X16*X17*X18*X19*X20*X21"`.  The return value is a `list[int]` of
positive `k` values such that `rec[-k]` are the records whose XOR fixes the
sign of `<operator>` in the circuit's **bare state** (the noiseless ideal
output state of the protocol, in which every deterministic observable has a
fixed value).

### What it does

Magic-state-prep circuits use **byproduct frames**: intermediate measurement
outcomes that determine a sign correction.  Without the right frame in the
`PAULI_EXPECTATION` line the engine reads the raw, uncorrected expectation —
which is ±1 depending on random byproduct outcomes and gives a nonsensical
average near 0.  `extract_frame` queries the C++ `expectation_frames` kernel
(the same kernel used internally when the circuit compiles its own columns) for
any operator you supply.

### Empty return `[]` — sign already constant

An empty list `[]` is a **valid** result.  It means the operator's sign is
already determined by the bare state without any measurement conditioning.  Two
cases produce this:

1. The operator is in the stabilizer group (ideal expectation ±1 regardless of
   any measurement outcomes).
2. The operator's ideal expectation is exactly 0 — a "magic direction" such as
   ⟨Z̄⟩=0 for a T-state.

The resulting `PAULI_EXPECTATION` line simply carries no `rec[-k]` records.  To
see the numeric expectation value, call `xtim.diagnose`.

> **Note:** an empty frame is distinct from a non-pinnable operator.  A
> non-pinnable operator raises `XtimError`; an operator with a constant or
> zero-magnitude sign returns `[]`.

### `minimal=True` (default) — the canonical minimal frame

For a **new circuit you are building**, `minimal=True` is always the right
choice: it returns the smallest set of records whose XOR pins the sign, which
is the most readable and robust form.

```python
import xtim

c = xtim.load_example("code_switching_faithful")
XBAR = "X15*X16*X17*X18*X19*X20*X21"
offsets = xtim.extract_frame(c, XBAR)
# offsets = [21, 20, 19, 18, 17, 16, 15]
# absolute record indices: {16, 17, 18, 19, 20, 21, 22}  (= num_meas - k)

frame = " ".join(f"rec[-{k}]" for k in offsets)
print(f"PAULI_EXPECTATION(0) {XBAR} {frame}")
# PAULI_EXPECTATION(0) X15*...*X21 rec[-21] rec[-20] ... rec[-15]
```

> **Unique labels:** the integer label in `PAULI_EXPECTATION(N)` must be unique
> within a circuit.  When appending your extracted line to a circuit that already
> declares other columns, pick a fresh index — naively reusing `0` when a
> `PAULI_EXPECTATION(0)` already exists raises a "duplicate label" error.

### `minimal=False` — full-readout representative

`minimal=False` is for **auditing or comparing against a declaration already in
the circuit**: it returns a spanning superset that mirrors an existing
`PAULI_EXPECTATION` declaration in the original circuit, if one exists;
otherwise falls back to the minimal set silently.  Both forms give the identical
corrected expectation value — the extra records cancel.

```python
offsets_full = xtim.extract_frame(c, XBAR, minimal=False)
# [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7]
# absolute record indices: {16, ..., 30}
```

### When it raises `XtimError`

`extract_frame` raises `xtim.XtimError` when the sign of `operator` is NOT
pinnable by any measurement-record frame — i.e., when the operator has a free
or logical component whose sign is not fixed by the circuit's bare-state
stabilizer frame.  This is a named, actionable refusal: you never get a
silently wrong number.

---

## `xtim.fidelity_from_logicals` — full-support 1−F from k generator pairs

```python
result = xtim.fidelity_from_logicals(circuit, logicals, *, p=None, shots=1_000_000, seed=0)
# equivalently:
result = circuit.fidelity_from_logicals(logicals, p=..., shots=..., seed=...)
```

`logicals` maps each logical qubit index to its `{"X": ..., "Z": ...}` generator
strings.  The return value is a fidelity-mode `PostselectedLER` with `result.value`
equal to `1 − F`.  `result.target_k` is the inferred number of logical qubits `k`
(used as the `2^k` fidelity denominator).

### What it does

`postselected_logical_error_rate` scores only the `PAULI_EXPECTATION` columns
a circuit happens to declare.  If a circuit declares an *incomplete* support —
some logical Paulis with nonzero ideal expectation are missing — the returned
`1 − F` is a **partial-support proxy** that has a noiseless floor which
**inflates / over-states** the apparent infidelity.  It is NOT a lower bound.

**Concrete example:** for `cultivation_d3_faithful`, declaring only X̄ (missing
Ȳ) gives a noiseless floor of `1−F = 0.25` even at `p=0`.  Declaring the full
support (X̄ + Ȳ) gives `1−F = 0.0` noiselessly (Parseval-exact).  The partial
proxy over-states the infidelity by 0.25, not under-states it.

`fidelity_from_logicals` always scores the **full support**: it generates all
`4^k − 1` non-identity logical Pauli products from the supplied generators,
pins each product's byproduct frame via `extract_frame`, and scores them all
together.  This gives the true infidelity

```
F = (1 / 2^k)  Σ_P  β_P · <P>
```

where `P` ranges over all `4^k` logical Pauli products and `β_P` is the
noiseless ideal expectation.

### The noiseless-F=1 validator trick

Before running a noisy estimate, always verify your generators at `p=0.0`:

```python
import xtim

c = xtim.load_example("cultivation_d3_faithful")
XBAR = "X0*X3*X7*X9*X10*X12*X13"
ZBAR = "Z0*Z3*Z7*Z9*Z10*Z12*Z13"

r = c.fidelity_from_logicals({0: {"X": XBAR, "Z": ZBAR}}, p=0.0, shots=2000)
# PostselectedLER(value=0, ..., upper_bound=0.0015)
```

A correct full-support algebra for a pure magic state satisfies Parseval's
identity (`Σ β²_P = 2^k − 1`), which forces `F = 1` noiselessly.  **The direct
check is `r.value`: if `r.value > 0` at `p=0.0`, at least one generator is
wrong** — fix them before running noisy estimates.  (As a belt-and-suspenders
signal, a non-zero noiseless `1 − F` also attaches a `"noiseless 1-F = …"`
advisory to `r.warnings`; but note that most wrong generators are rejected
earlier — a non-anticommuting pair raises immediately, and a generator whose
sign is not record-pinnable raises from `extract_frame` — so in practice
`r.value > 0` is the signal you will actually see for the residual cases,
such as a genuinely mixed-state prep.)

> **Note:** `fidelity_from_logicals` ignores any `PAULI_EXPECTATION` lines
> already in the circuit — it strips them and rebuilds the full `4^k` column
> set from your generators.  The result reflects your declared logicals, not
> whatever the circuit happened to declare.

### Noisy estimate and the partial-support divergence

```python
r = c.fidelity_from_logicals({0: {"X": XBAR, "Z": ZBAR}}, p=1e-3, shots=1_000_000)
# PostselectedLER(value≈1.09e-05, sem≈8.2e-06, mode=fidelity, ...)

r_ler = c.postselected_logical_error_rate(p=1e-3, shots=1_000_000)
# PostselectedLER(value≈1.09e-05, ...)
```

They agree for `cultivation_d3_faithful` because that circuit already declares
X̄ and Ȳ, and the T-magic state has `⟨Z̄⟩ = 0` — the missing Z̄ column
contributes nothing to F.

**When they diverge (partial-support proxy inflates):**

Strip cultivation's declared `PAULI_EXPECTATION` lines and declare only X̄.
Running `postselected_logical_error_rate(target_k=1)` at `p=0` returns
`1−F ≈ 0.25` (fires the "PARTIAL-support proxy" warning); the full-support
`fidelity_from_logicals` returns `1−F = 0.0`.  The partial proxy **inflates**
the apparent infidelity by 0.25 because the missing Ȳ column accounts for half
the Parseval sum.  A runnable demonstration is in
`examples/logical_fidelity_helpers.py` Section 4.

**General principle:** when a circuit's declared columns are incomplete, the
partial-support proxy has a noiseless floor — it is NOT a lower bound.  In
those cases reach for `fidelity_from_logicals`, which always scores the full
`4^k` support and has a noiseless floor of exactly `1−F = 0.0`.

### Algebra validation

The helper validates the logical Pauli algebra before any sampling:

* `X̄_i` and `Z̄_i` must **anticommute** (they are conjugate operators of one
  qubit);
* every cross-qubit pair (`X̄_i` / `X̄_j`, `X̄_i` / `Z̄_j`, etc.) must
  **commute**.

A violation raises `xtim.XtimError` naming the offending pair — no silent
wrong number.  The same applies if a generator string is malformed or a
product's sign is not pinnable by the circuit's measurement frame.

### Caveats

**Full-support / pure-logical-state assumption.** `fidelity_from_logicals`
assumes the output of your protocol is a *pure* logical state in the code
space.  If the prep is genuinely mixed (e.g., a probabilistic ensemble of
different logical states), `1 − F` from this helper is not the average-case
infidelity; it measures departure from a single target pure state.

**k grows exponentially.** The helper evaluates `4^k − 1` Pauli products.  For
`k ≥ 5` (≥ 1023 columns) it emits an advisory; at `k ≈ 8` the construction
time starts to dominate.  For large `k`, consider measuring a representative
subset and using `extract_frame` to build a custom `PAULI_EXPECTATION` set.

**`XtimError` on an unpinnable product.** `fidelity_from_logicals` calls
`extract_frame` for every one of the `4^k − 1` products.  If a product's sign
is not pinnable (a genuine failure: the product has no fixed byproduct frame in
this circuit), the call raises `XtimError` with the offending product named.

---

## Parameters

### `extract_frame`

| parameter | default | meaning |
|---|---|---|
| `circuit` | — | `xtim.Circuit` containing the protocol |
| `operator` | — | Pauli product string, e.g. `"X15*X16*X17"` |
| `minimal` | `True` | `True` → canonical minimal frame (use for new circuits); `False` → full-readout superset (for auditing existing declarations; falls back to minimal if no matching declaration found) |

Returns `list[int]` of positive offsets `k` where `rec[-k]` are the
sign-controlling records.  To recover the 0-indexed absolute position:
`abs_index = circuit.num_measurements - k`.  An empty list `[]` is a valid
result indicating the sign is constant (stabilizer or zero-magnitude direction).

### `fidelity_from_logicals`

| parameter | default | meaning |
|---|---|---|
| `circuit` | — | `xtim.Circuit` containing the protocol |
| `logicals` | — | `{qubit_index: {"X": ..., "Z": ...}, ...}` |
| `p` | `None` | physical error rate to scale noise to (`None` = run as-is, unlike the sibling's default `p=1e-3`) |
| `shots` | `1_000_000` | number of shots (matches `postselected_logical_error_rate`) |
| `seed` | `0` | engine seed |
| `p0` | `1e-3` | the circuit's baked-in noise strength; pass this if your circuit was built with a different noise rate |

Returns a fidelity-mode `PostselectedLER`; `.value` is `1 − F`.  `.target_k`
is the inferred number of logical qubits `k` (fidelity denominator `2^k`).

---

## When to use which helper

| goal | tool |
|---|---|
| Measure the LER of an existing circuit as written | `postselected_logical_error_rate` (default) |
| Audit / build a `PAULI_EXPECTATION` line for a chosen logical Pauli | `extract_frame` |
| Check whether your X̄/Z̄ generators are correct (noiseless F=1 test) | `fidelity_from_logicals(p=0.0)` |
| Measure TRUE full-support 1−F (circuit may declare incomplete columns) | `fidelity_from_logicals` |

---

## Note on `true_fidelity` (dropped)

An automatic zero-declaration variant (`true_fidelity`) was designed and
deliberately dropped: the sampler frame does not carry the output-code logical
basis, so there is no sound way to derive the output-state logicals from the
circuit text alone — you must supply them explicitly.

---

## Runnable example

```
conda activate qec
python examples/logical_fidelity_helpers.py
```

See `examples/logical_fidelity_helpers.py` for a complete, runnable demo of
both helpers with printed output and explanations.  Section 4 demonstrates the
partial-support divergence explicitly.
