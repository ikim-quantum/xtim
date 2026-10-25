# One-call post-selected logical error rate — `Circuit.postselected_logical_error_rate`

> **This reports the post-selected logical error rate of the circuit's `OBSERVABLE_INCLUDE`
> observable(s): every shot in which ANY deterministic detector fired is DISCARDED (heralded).
> This is the pure-post-selection / maximal-heralding number — NOT a decoded logical error rate.**

```python
import xtim

c = xtim.load_example("cultivation_d5")
r = c.postselected_logical_error_rate(p=1e-3, shots=10**6)

r.value        # fraction of accepted shots with a logical observable flipped
r.sem          # binomial standard error of r.value
r.upper_bound  # rule-of-three 3/kept (95% CL) when 0 errors observed, else None
r.acceptance   # post-selection acceptance rate (kept / shots)
r.kept         # number of accepted shots
r.warnings     # list[str] of advisories (also re-emitted via `warnings.warn`)
```

`xtim.postselected_logical_error_rate(circuit, ...)` is an identical module-level alias.

## Why one call — the O(p) vs O(p^d) pitfall

A magic-state-prep circuit's fault tolerance comes from *post-selection*: you keep only the
shots in which no deterministic detector fired, and throw the rest away. Skip that filter and
you measure the *raw* error rate, `O(p)`; apply it and you measure the *suppressed* rate,
`O(p^d)`. The call builds that keep mask for you, every time, from the EXACT noiseless
determinism partition (the parity-Pauli eigenvalue of each detector on the bare state — no
sampling, no statistical estimate):

```python
keep = ~dets[:, deterministic_detectors].any(axis=1)   # applied automatically
```

The shots come from the default twirl detector sampler; a circuit the twirl engine refuses
raises `xtim.XtimTwirlRefusal` by name (there is no fallback engine).

## What it refuses

* **A circuit with an output register to score** is not this function's job: use `xtim.RecordFidelity`
  ([usage](usage.md)). A circuit that declares `PAULI_EXPECTATION` lines is refused by name (`xtim.XtimError`).
* **A circuit with no `OBSERVABLE_INCLUDE`** raises `xtim.XtimError`: there is nothing to score.

## Warnings

* *no noise* — the circuit declares no noise instruction, so `value` is the noiseless logical
  error rate (varying `p` rescales nothing).
* *zero-error upper bound* — when 0 logical errors survive, `sem` is 0 and not meaningful;
  `upper_bound = 3 / kept` (rule of three, 95% CL) is set and `value` is 0.
* *low acceptance* — few shots survive post-selection; raise `shots`.

## Parameters

`p` (default `1e-3`; `None` runs the circuit as-is) rescales every noise argument by `p / p0`
(`xtim.scale_noise`; `p0` = the baked-in strength, default `1e-3`); `shots`, `seed` control the
statistics.
