# One-call post-selected logical error rate — `Circuit.postselected_logical_error_rate`

> **This reports the post-selected logical error rate: every shot in which ANY
> deterministic detector fired is DISCARDED (heralded). This is the
> pure-post-selection / maximal-heralding number — NOT a decoded logical error rate.
> A decoded / partial-post-selection variant is future work.**

Estimating the post-selected logical error rate (LER) of a magic-state-prep protocol
used to require you to know an entire hidden workflow.
`Circuit.postselected_logical_error_rate` collapses it into a single call.

```python
import xtim

c = xtim.load_example("code_switching_faithful")
r = c.postselected_logical_error_rate(p=1e-3, shots=10**6)

r.value        # the LER: 1-F (fidelity mode) | post-selected observable LER
r.sem          # standard error of r.value
r.upper_bound  # rule-of-three 3/kept (95% CL) when 0 errors observed, else None
r.acceptance   # post-selection acceptance rate (kept / shots)
r.kept         # number of accepted shots
r.mode         # "fidelity" | "observable"
r.target_k     # k used for the fidelity denominator 2**k
r.warnings     # list[str] of advisories (also re-emitted via `warnings.warn`)
```

`xtim.postselected_logical_error_rate(circuit, ...)` is an identical module-level alias.

## Why one call — the O(p) vs O(p^d) pitfall

The value of this API is a single automatic step you would otherwise do by hand
and, crucially, *forget*: **post-selecting on the deterministic detectors**.

A magic-state-prep circuit's fault tolerance comes from *post-selection*: you keep
only the shots in which no deterministic detector fired (the noiseless output-code
syndrome — including the end-of-circuit syndrome-extraction round — must be clean),
and throw the rest away. Skip that filter and you are measuring the *raw* error
rate, which scales as `O(p)`. Apply it and you measure the *suppressed* rate, which
scales as `O(p^d)`.

Concretely, for `code_switching_faithful` at `p = 1e-3`:

| workflow | LER |
|---|---|
| **no post-selection** (the pitfall) | `~7e-2` — unsuppressed, `O(p)` |
| **auto post-selection** (this API) | `~5e-7` — suppressed, `~O(p^3)` |

That is a **five-orders-of-magnitude** difference produced by one line of numpy the
user routinely omits. `postselected_logical_error_rate` builds that keep mask for you,
from `xtim.diagnose(circuit).deterministic_detectors`, *every time*:

```python
keep = ~dets[:, deterministic_detectors].any(axis=1)   # applied automatically
```

The recent end-SE detector fixes mean this mask now correctly captures the
noiseless output-code syndrome, so the suppression is real and not an artifact of a
missing detector.

## The two modes (auto-detected)

### Fidelity mode — `PAULI_EXPECTATION`

If the circuit declares at least one *scorable* `PAULI_EXPECTATION` column — one
whose noiseless target `beta` is both well-defined (non-nan) and nonzero — the value
is the infidelity `1 - F` of the prepared magic state, computed by the per-shot
fidelity path in `xtim.collect` (`Task(target_k=...)`):

```
f_s   = (1 + <corrected expectations on shot s> · beta) / 2**target_k
1 - F = 1 - mean(f_s over accepted shots)
```

`beta` is the noiseless target `<P_i>` from `diagnose`. `target_k` defaults to 1;
set it to the number of logical qubits in the target magic state.

Worked numbers (auto post-selected):

| example | mode | `p` | LER |
|---|---|---|---|
| `cultivation_d3_faithful` | fidelity | 1e-3 | `~p^3` (≈0 at 1e6 shots) |
| `distillation_15_1_3` | fidelity | 1e-3 | `~p^3` (15-to-1 distillation) |
| `code_switching_faithful` | fidelity | 1e-3 | `~5e-7` (suppressed, `~p^3`) |

### Observable mode — `OBSERVABLE_INCLUDE`

If the circuit declares no scorable `PAULI_EXPECTATION` but does declare an
`OBSERVABLE_INCLUDE`, the value is the **post-selected logical error rate**: the
fraction of *accepted* shots in which any declared logical observable flipped.

```python
c = xtim.load_example("cultivation_d5")     # ref chi=2, two OBSERVABLE_INCLUDE, no PAULI_EXPECTATION
r = c.postselected_logical_error_rate(p=1e-3, shots=10**6)
r.mode        # "observable"
r.value       # post-selected logical-flip fraction (≈0 — d=5 suppresses it hard)
```

A shot counts as a logical error if *any* declared observable is flipped.

### Nothing to score

A circuit with neither a scorable `PAULI_EXPECTATION` nor an `OBSERVABLE_INCLUDE`
raises `xtim.XtimError` with an actionable message — declare a logical observable or
a magic expectation so there is something to measure against.

## Warnings — when the value is a proxy, not a clean LER

`r.warnings` is a list of strings; the value is always returned (except in the
nothing-to-score case, which raises — see below). Each advisory is ALSO re-emitted
through `warnings.warn`, so a script that reads only `r.value` still sees it on
stderr; `r.warnings` keeps them for programmatic access.

### No noise → the value is a floor, not an LER

If the circuit declares **no noise instruction** at all, scaling `p` rescales
nothing and `value` is just the noiseless fidelity floor — never an error rate. This
fires for `cube_ccz` (a static, noiseless magic-state definition):

```python
r = xtim.load_example("cube_ccz").postselected_logical_error_rate(p=1e-3, shots=10**4)
# r.value == 0.125 at EVERY p (no noise to scale) — NOT a 12.5% LER.
# r.warnings[0]: "circuit declares NO noise instructions: 'value' is the noiseless
#                 fidelity floor, NOT an error rate ..."
```

Add a noise channel (`DEPOLARIZE1/2`, `X_ERROR`, …) to measure a real LER.

### Incomplete Pauli support

If the declared `PAULI_EXPECTATION` set cannot reach fidelity 1 even with no noise
— `sum(beta**2) < 2**target_k - 1`, i.e. a noiseless ceiling below 1 — the returned
`1 - F` is a **partial-support proxy**, not the true infidelity. This also fires for
`cube_ccz` (a `[[8,3,2]]` transversal-CCZ state whose 3 declared columns are an
incomplete support of the 3-logical-qubit state at the default `target_k=1`, so the
`0.125` above is doubly flagged — no noise AND partial support):

```python
r.warnings
# [..., "declared PAULI_EXPECTATION set cannot reach fidelity 1 (noiseless ceiling=0.875 < 1): ...",
#       "value=0.125 is a PARTIAL-support proxy, NOT a true infidelity/LER: ... pass target_k=<k> ..."]
```

To get a true fidelity, declare every Pauli with a nonzero ideal expectation and set
`target_k` to the number of logical qubits in the target (for `cube_ccz`, `target_k=3`).

### Free / logical or magnitude-0 read (no fidelity anchor)

A `PAULI_EXPECTATION` column carries a fidelity anchor only if its noiseless target
`beta` is both **well-defined (non-nan)** and **nonzero**. Two kinds of column are
therefore *non-scorable* and excluded from the fidelity score:

* a **free/logical read** whose sign is not governed by a fixed byproduct frame — a
  `nan` target (e.g. a destructive terminal MPP whose logical read is a fair Born
  coin);
* a **magnitude-0 read** whose target `beta` is exactly `0` — a zero anchor carries
  no fidelity information (scoring it would give a spurious `F = (1 + 0)/2 = 0.5`).

`mpp_magic` is the illustrative case: its two columns are `X0*X1*X2` (a free/logical
read, `nan`) and `Y0*X1*X2` (`beta = 0`). **Both** are non-scorable and there is no
`OBSERVABLE_INCLUDE`, so there is nothing to score — the call **raises `XtimError`**
rather than report a plausible-looking `0.5`:

```python
xtim.Circuit.from_file("benchmarks/mpp_magic.stim").postselected_logical_error_rate(p=1e-3, shots=10**4)
# XtimError: ... this circuit declares no scorable PAULI_EXPECTATION and no
#            OBSERVABLE_INCLUDE ... (all columns free/logical or magnitude-0) ...
```

If *some* columns are scorable, the non-scorable ones are simply excluded (with a
warning) and fidelity is scored on the rest. If *every* `PAULI_EXPECTATION` column is
non-scorable but an `OBSERVABLE_INCLUDE` exists, the call falls back to observable
mode (with a warning); if there is no observable either, it raises `XtimError`.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `p` | `1e-3` | physical error rate the noise is scaled to |
| `shots` | `10**6` | number of shots |
| `seed` | `0` | engine seed (one run, one RNG stream) |
| `target_k` | `1` | logical-qubit count of the target (fidelity denominator `2**k`) |
| `p0` | `1e-3` | the circuit's baked-in noise strength |

`p` / `shots` / `seed` rescale the noise and control statistics exactly as an
`xtim.Task` does — the call reuses `diagnose`, `collect`, and `Task` verbatim and
re-implements no sampling.

## Input validation

`postselected_logical_error_rate` validates its inputs **before** any sampling: `p`
must be `None` or a float in `[0, 1]` (else `ValueError("p must be a float in [0,1]
or None, ...")`), `shots` must be `>= 1`, and `circuit` must be an `xtim.Circuit` (a
`stim.Circuit` raises a `TypeError` that names the type and suggests
`xtim.Circuit(str(the_stim_circuit))`).

## Notes

* `value` is **unclamped**: in fidelity mode `1 - F` can be a tiny negative near
  `F = 1` (floating-point, or within the error bar at low `p`). `repr` clamps a
  tiny-negative to `0.0` for display, but `r.value` keeps the raw number — guard
  `max(0.0, r.value)` before a log-scale plot.
* **Zero-error regime — read `upper_bound`, not `sem`.** When 0 logical errors
  survive post-selection, the sample `sem` collapses to ~0, which reads as a
  misleading "LER = 0 ± 0". In that case `value` rounds to 0 and the call sets
  `r.upper_bound = 3 / kept` — the rule-of-three 95%-confidence upper bound — and
  warns `0 errors in K accepted shots → LER < 3/K`. Report `LER < r.upper_bound`
  (or raise `shots` / combine seeds by inverse-variance to resolve a nonzero value).
* When `0 < kept < 100`, the estimate is statistically weak regardless of `sem`; a
  warning says so. Very low acceptance (`< 1e-3`) warns too.
* See `examples/postselected_logical_error_rate.py` for a runnable p-sweep across
  the bundled circuits (it prints every advisory at every `p`).
* **See also:** [xtim_logical_fidelity_helpers.md](xtim_logical_fidelity_helpers.md)
  — opt-in `extract_frame` and `fidelity_from_logicals` helpers for when you need
  to supply the logical operators explicitly and score the full 4^k support.
