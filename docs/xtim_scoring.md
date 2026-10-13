# Scoring a magic state — `diagnose()`, the fidelity arithmetic, and its pitfalls

This page is the complete version of the README's sixty-second example: what
`diagnose()` reports, how to turn raw per-shot expectations into a true-state
fidelity, and the four ways that arithmetic silently goes wrong. Read it once
before scoring your first protocol; after that the
[`collect` shortcut](#the-shortcut-a-fidelity-curve-from-collect) does the
arithmetic for you.

**Three terms used throughout** (all also printed by `diagnose()`):

- **χ (chi)** — stabilizer rank = how many stabilizer states your logical output
  superposes (one magic state ⇒ χ ≤ 2).
- **gauge detector** — a detector that flips randomly even noiselessly (leave it;
  don't post-select on it).
- **byproduct frame** — an expectation column whose sign depends on measurement
  outcomes, so it must be sign-corrected with the record-parity `HINT` before you
  average it (contrast: `sign constant`).

## Start with `diagnose()`

Point it at a protocol and it does one noiseless run and *reports the facts you
would otherwise reverse-engineer by hand* — the reference χ, which detectors to
post-select, each magic channel's `|β|` magnitude and its byproduct-frame-corrected
**signed** target βᵢ (the bottom `target (signed beta)` line is the full target
Bloch vector), and its byproduct-frame sign rule:

```python
import xtim

c = xtim.load_example("cultivation_d3_faithful")   # a bundled demo — no clone needed
print(c.diagnose())
```

> `xtim.load_example(name)` loads any of the bundled demos straight from the installed
> package (`xtim.list_examples()` lists them; `xtim.example_path(name)` gives the file).
> For your own circuit use `xtim.Circuit.from_file("your_protocol.stim")`. (Declare the
> observables you want with `PAULI_EXPECTATION(<label>) <Pauli>` — the parenthesized arg
> is an **integer label/index**, not the target value, e.g. `PAULI_EXPECTATION(0) X0*X1`.)

```
xtim diagnosis (noiseless run, 1024 shots)
====================================================
reference chi : 2 (cached)
detectors     : 20 total — 20 deterministic, 0 gauge
  deterministic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  gauge        : (none)
expectations  : 2 declared PAULI_EXPECTATION column(s)
  [0] value=+0.707106781187  sign constant
  [1] value=+0.707106781187  sign constant
  target (signed beta): [+0.707107, +0.707107]
suggested post-selection (advisory; not applied):
  keep = ~dets[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].any(axis=1)
  (this recipe post-selects ALL deterministic detectors — the conservative policy. For a decoder DEM that rejects only the
   not-Pauli-correctable faults and decodes the rest, see Circuit.detector_error_model_with_reject() — docs/xtim_dem_reject.md.)
legend: deterministic detector = always 0 noiselessly (post-select on it); gauge detector = flips randomly (leave it);
        signed beta = the target expectation <P_i> of your magic state. To SCORE: flip exps by the HINT's record-parity only;
        the trailing '+1' is the canonical sign, ALREADY folded into signed beta — do not re-apply it per shot.
```

> On a **cold cache** the first run prints `reference chi : not yet cached (deduced on
> the fly — fine; run any compile_* to cache it)` instead of the `2 (cached)` shown
> above — same facts, just not memoized yet. Any `compile_*` call (or
> `compile_reference()`) warms it.

`diagnose()` *reports*; it never applies a correction or filters shots. With those
facts in hand the rest is ordinary NumPy — sample, post-select, combine the raw
expectations into a fidelity:

```python
import numpy as np

dets, obs, exps = c.compile_detector_sampler(seed=7).sample(
    20_000, separate_observables=True, return_expectations=True)
keep = ~dets.any(axis=1)               # post-select the deterministic detectors
target = np.array([0.70710678, 0.70710678])    # COPY the report's `target (signed beta)`
F = (1 + (target * exps[keep].mean(axis=0)).sum()) / 2   # sum over ALL expectation columns
print(keep.sum(), "accepted;  F ≈", F)         # ≈ 1.0: the post-selected state IS the target
```

`F ≈ 1.0` is the fidelity of the **post-selected** state (here ~69% of shots are
kept) — not of every shot; the circuit is noisy, and post-selecting the
deterministic detectors heralds the errors away.

## The four pitfalls

### 1. Copy the target — don't hand-derive the signs

🎯 **Use the `target (signed beta)` line from `diagnose()` verbatim.** `diagnose()`
prints the exact target your circuit prepares; in the example above it happens to
be sign-symmetric `[+0.707, +0.707]`, but if you guess a sign wrong (e.g.
`[+0.5, −0.5]` when the report says `[+0.5, +0.5]`) `F` silently lands at ~0.5.
Copy the line; that's what it's for.

### 2. A raw mean is only valid for `sign constant` columns

⚠️ **The raw-mean shortcut above is valid only because the report says `sign
constant` for every column.** Averaging `exps` directly works when the sign is
fixed. If your report instead says `byproduct frame` (the *expected* case for a
fresh protocol — `code_switching`, `miniature_oracle`, …), the sign flips shot to
shot and a raw mean collapses to `F ≈ 0.5`. You must first fold in the
record-parity `HINT` `diagnose()` prints, per column, before averaging —
[`../examples/onboarding_new_protocol.py`](../examples/onboarding_new_protocol.py)
does exactly this and asserts it matters (`raw_F ≈ 0.50` vs frame-corrected `0.99`).

### 3. Completeness: declare every nonzero Bloch component

The Bloch-vector formula `F = (1 + target·⟨P⟩)/2` is exact **only when every
declared `PAULI_EXPECTATION` column covers the target's nonzero Bloch components**
(the T-state target above has `⟨Z̄⟩ = 0`, so the two declared columns suffice). If
your target has a nonzero component you did not declare, add a column for it —
otherwise `F` silently over-counts.

### 4. `F > 1` means your target is unphysical — fix the target, not the statistics

**`target` must be a physical state — `|target| ≤ 1`.** Because xtim's per-shot
expectations are *exact* (not ±1 samples), the post-selected mean is the true Pauli
vector of a genuine density matrix, so with a correct target `F` is an average of
per-shot overlaps `|⟨φ|ψₛ⟩|²` and **cannot exceed 1 except by floating-point
rounding**, for any finite shot count. So `F > 1` (beyond ~1e-12) is never noise —
it's a reliable signal that your `target` is unphysical (`|target| > 1`, or weight on
an undeclared component). A single
`assert abs(np.linalg.norm(target)) <= 1` is the whole guardrail.

## The shortcut: a fidelity curve from `collect`

The Bloch arithmetic above is worth understanding once — but for a **p-sweep you
usually just want the curve**. Set `target_k` (the number of logical qubits in the
target magic state) on each `Task` and `xtim.collect` scores it for you:

```python
tasks = [xtim.Task(c, p=p, shots=50_000, seed=7, target_k=1)
         for p in (3e-4, 1e-3, 3e-3, 1e-2)]
for row in xtim.collect(tasks):
    print(f"p={row['p']:g}  1-F = {row['infidelity']:.2e} ± {row['infidelity_sem']:.1e}")
```

Each row gains `fidelity`, `infidelity` (= 1 − F), `infidelity_sem` (the
correlation-aware error bar — *not* a per-column quadrature), `target_k`, and
`target_beta` (the noiseless target, derived for you — no copying the signed-beta
line by hand). Same true-state fidelity, now with an error bar and a plot-ready
curve. `xtim` warns if a row's `F` exceeds 1 — the loud sign that `target_k` is
wrong or a column is double-counted. You still own **completeness**: declare every
Pauli with a nonzero ideal expectation. A full worked curve with a matplotlib plot
is [`../examples/fidelity_curve.py`](../examples/fidelity_curve.py).

## See also

- [`xtim_tour.md`](xtim_tour.md) — the same arithmetic inside the full
  decoder-in-the-loop workflow (executed verbatim by the test suite).
- [`xtim_postselected_logical_error_rate.md`](xtim_postselected_logical_error_rate.md)
  — the one-call post-selected 1−F scorer.
- [`xtim_logical_fidelity_helpers.md`](xtim_logical_fidelity_helpers.md) — opt-in
  helpers (`extract_frame`, `fidelity_from_logicals`) for doing the frame
  arithmetic by hand.
