# xtim: Decoder DEMs with a Reject Region (magic-state-prep)

This guide covers `Circuit.detector_error_model_with_reject()` — how xtim exports a **sound,
complete** detector error model (DEM) for circuits with magic (`PAULI_EXPECTATION` columns), where
some single faults are detectable but **not** correctable by a Pauli decoder and must instead be
**post-selected**.

> **Per-shot judgement (3.1.13).** A fault is classified by what it does to each column **per shot**
> — the value the sampler reports, conditioned on every measurement outcome — never by the
> read-averaged expectation. Until 3.1.12 the averaged rule flagged 177 faults of the bundled
> `cultivation_d3_faithful` (a "saturated reject region" over all 20 detectors); judged per shot every
> one of them keeps every column's magnitude (a detector-correlated *sign* law), so they are ordinary
> mechanisms and cultivation has **no** reject region. Its exact export needs
> `approximate_disjoint_errors=True` (those sign laws are not products of independent mechanisms),
> like `cultivation_d5`. The worked example below uses a circuit with a genuine flagged fault.
>
> **Columns with no DEM concept.** A `PAULI_EXPECTATION` whose *noiseless* per-shot value varies
> shot by shot (it depends on measurement records not declared as its frame) is not a DEM column: it
> is dropped (no `L` line, mechanism, flag or refusal; its `L` index stays reserved), reported in
> `DemWithReject.dropped_expectation_columns` as `(declaration index, reason)` and named once in an
> `xtim.XtimDemWarning`. Declare the named records as the column's frame to make it a DEM column.

If you only have an ordinary stabilizer memory/QEC circuit (no `PAULI_EXPECTATION` declarations),
you do not need any of this — use `detector_error_model()` as usual; the reject region is empty and
the DEM is byte-identical to before.

> **Controlled-Hadamard circuits:** a circuit containing `CH` samples exactly, but its faults act
> as general *Cliffords* (not Paulis) on the state, so there is no linear DEM to export — both DEM
> entry points refuse with `XtimDemError`. This is the honest answer, not a bug: score CH-class
> protocols with `diagnose()` + post-selection on the deterministic detectors (which `diagnose`
> lists), exactly as its report suggests. See
> [`xtim_simulable_class.md`](xtim_simulable_class.md).

---

## TL;DR

```python
import xtim, pymatching

c = xtim.Circuit(open("protocol.stim").read())

res = c.detector_error_model_with_reject()      # -> xtim.DemWithReject
dem            = res.dem                          # a stim.DetectorErrorModel (the decoder's model;
                                                  #   its L columns AFTER the ordinary observables are
                                                  #   the expectation columns -- ignore their predictions)
flagged        = res.postselect_faults            # PRIMARY: the not-Pauli-correctable faults to flag
reject_dets    = res.reject_detectors             # derived convenience union (the blunt policy below)

matcher = pymatching.Matching.from_detector_error_model(dem)

# Per shot: discard if any reject detector fired; otherwise decode.
def accept_and_decode(det_bits):                  # det_bits: np.bool_ array over all detectors
    if det_bits[reject_dets].any():
        return None                               # rejected (post-selected out)
    return matcher.decode(det_bits)               # Pauli correction for the survivors
```

The magic value itself (⟨X̄⟩, ⟨Ȳ⟩, …) is **not** in the DEM — it is the *payload*, read from the
expectation channel (`compile_detector_sampler(..., return_expectations=True)`), evaluated on the
accepted, decoded shots.

---

## Why a reject region exists

A Stim DEM is a **Pauli/stochastic** noise model: every mechanism is `(probability, detector flips,
logical-observable flips)`. A decoder reads it and outputs a **Pauli** correction.

In a magic-state-prep circuit, a single circuit-level fault — after deferral (pushing the fault's
Paulis to the end of the circuit), the protocol's syndrome measurements, and measurement-basis
twirling (the measurement collapses the coherent cross-terms, turning a residual coherent error into a
random Pauli flip — see the "Twirled randomization" note at the end) — can leave a residual that is
a **coherent logical Clifford** (e.g. a logical `S`) on the magic qubit. A coherent `S` is not a Pauli channel and
**cannot** be written as a DEM mechanism; a Pauli decoder cannot recover it. Such a fault is
**detectable but not Pauli-correctable**: the syndrome flags it, but the only sound response is to
*reject* the shot, not correct it.

So a complete magic-state-prep DEM has three classes of single fault:

| class | what it is | what xtim does |
|---|---|---|
| **Pauli-correctable** | every logical magnitude `\|⟨P̄_r⟩\|` is preserved (the residual at most flips signs) | ordinary DEM edge, with a logical-observable flip on each sign-changed column |
| **detectable, not Pauli-correctable** | some `\|⟨P̄_r⟩\|` changes (decoherence / a different magnitude) and ≥1 detector fires | emitted **nowhere** in the DEM; **flagged** in `postselect_faults` (its detector signature + probability) — *you* choose to post-select it or budget it |
| **undetectable, not Pauli-correctable** | some `\|⟨P̄_r⟩\|` changes but **no** detector fires | **fatal**: `XtimDemError` (the circuit under-declares its post-selection — see the gate below) |

### The magnitude criterion (why `|⟨P̄⟩|`, not the sign)

A Pauli logical error only **sign-flips** the equatorial magic point, e.g. logical-`Z` sends
`(⟨X̄⟩,⟨Ȳ⟩) → (-⟨X̄⟩,-⟨Ȳ⟩)`; the decoder's Pauli frame undoes that. The general reason a sign flip is
recoverable: a **known sign flip is just bookkeeping** — the decoder tracks the Pauli frame and the
readout applies it — whereas a **magnitude change is information lost from the state**, which no frame
can restore. Importantly, a logical-`S` on a **symmetric** magic state (`|⟨X̄⟩|=|⟨Ȳ⟩|`) is *also* only
a sign flip there: `S` rotates the equator by 90°, and at the symmetric point that 90° rotation lands
exactly where the π-rotation `Ȳ` does, so `S|m⟩ = Ȳ|m⟩` — a Pauli, hence correctable; only *away* from
the symmetric point does `S` change a magnitude. The thing that is genuinely uncorrectable is a
change to a **magnitude** (the state decohered, or rotated to a different length). So xtim classifies
a fault by whether any `|⟨P̄_r⟩|` changes — checked exactly against the noiseless prepared state, so a
state-stabilizing "error" (which does nothing) is never flagged.

---

## API

### `Circuit.detector_error_model_with_reject(...) -> DemWithReject`

```python
def detector_error_model_with_reject(
    self, *,
    include_expectations: bool = True,          # the reject region needs the columns; see below
    decompose_errors: bool = False,
    ignore_decomposition_failures: bool = False,
    drop_gauge_observables: bool = False,
    approximate_disjoint_errors: bool | float = False,
    pe_max_work: float = 1.5e7,                 # exact per-shot column analysis: work units per walk
) -> "xtim.DemWithReject"
```

- Returns `DemWithReject(dem: stim.DetectorErrorModel, reject_detectors: list[int],
  postselect_faults: list[PostselectFault], dropped_expectation_columns: list[tuple[int, str]])` —
  see the dataclass listing below.
- **The reject region exists only when the columns are requested**, so `include_expectations`
  defaults to **`True`** here, matching `detector_error_model()` (2026-10-12 owner decision; until
  then it defaulted to `False`). The flags, the completeness gate and the accepted-branch check are
  statements about the `PAULI_EXPECTATION` columns. A caller who wants the plain
  detector/observable DEM with empty flags passes `include_expectations=False`: no column is
  requested and the DEM carries **every** fault on its detector/observable signature — nothing is
  refused, flagged or dropped on column grounds (`postselect_faults` and `reject_detectors` are
  empty, `dem` is byte-identical to the twin circuit without the declarations for every class-admitted circuit), and what the column
  analysis would have refused is *reported* in `dropped_expectation_columns` (see the gate section).
- With the default the `PAULI_EXPECTATION` channels are extra DEM observables (L-columns after the
  ordinary ones). Decode `dem` as usual and ignore the decoder's predictions for those columns: the
  magic value is the *payload*, read from the expectation channel
  (`compile_detector_sampler(..., return_expectations=True)`), never decoded.
- Raises `xtim.XtimDemError` if the completeness gate fires (an undetectable logical error — see
  below) with `include_expectations=True`; with `False` the finding is reported instead. Raises
  `xtim.XtimRejectError` if the circuit is outside the simulable class.
- `pe_max_work` caps the exact per-shot analysis of a `PAULI_EXPECTATION` column (the branch walk
  of `dem_pe_per_shot.hpp`) in work units per walk: each Born / collapse step is charged
  `(chi+1)·n·(k+1)`, and the default `1.5e7` is about 0.5 s. It is a deterministic count, never
  wall-clock, so a verdict never depends on the host. Past it the export refuses by name with
  `include_expectations=True` — the message names the argument — and reports the column
  (`"...; not judged"`) with `include_expectations=False`. The cost is exponential in the number of
  magic reads a column is correlated with (up to `2^k` branches, each at `chi ~ 2^k`): on the
  k-T-controls-on-one-target ladder it grows about 7x per T — k=9 exports in ~1 s at
  `pe_max_work=1e9`, k=11 in ~50 s, k=13 needs about `1e11` work units (about an hour). The branch
  cap (4096 support leaves per walk) is fixed: it is not the binding limit on these cases (a column
  over `d` magic-correlated reads has at most `2^d` leaves and the work cap binds first).

### `xtim.DemWithReject` and `xtim.PostselectFault`

```python
@dataclass(frozen=True)
class PostselectFault:
    detectors: list[int]    # the syndrome this fault flips/randomizes (no Pauli correction exists)
    probability: float      # this fault's rate (sum over faults = the O(p) coherent-error floor)

@dataclass(frozen=True)
class DemWithReject:
    dem: "stim.DetectorErrorModel"        # decode this (all the Pauli-correctable faults)
    reject_detectors: list[int]           # derived convenience: the union of the flagged signatures
    postselect_faults: list[PostselectFault]   # the PRIMARY output — the faults to flag
    dropped_expectation_columns: list[tuple[int, str]]   # columns with no DEM concept + reason
    # keep_mask(dets) -> bool[shots]: the blunt conservative keep-mask (no reject_detector fired)
```

**`postselect_faults` is the primary output, and xtim does not impose a policy** — because the
per-fault flags let *you* choose any policy (post-select, a finer per-signature rule, or
decode-and-budget), whereas `reject_detectors` bakes in just one of them (the blunt union). The faults
are the general object; the union is one view of them. Each entry is a not-Pauli-correctable fault:
the detector signature it flips/randomizes (where no Pauli correction exists) plus its probability.
You decide what to do with them:

- **Post-select** (sound): discard any shot whose syndrome is consistent with a flagged fault. The
  blunt, always-sound version is `keep_mask(dets)` / `reject_detectors` (reject if *any* flagged
  detector fires) — simplest, but it over-rejects when a flagged detector is shared with an ordinary
  Pauli error. A finer per-signature policy (reject only syndromes matching a flagged signature)
  accepts more shots, **but is only sound when a flagged signature can be distinguished from the
  Pauli-correctable edges that share its detectors** — and xtim ships no helper for it. ⚠ When
  `reject_detectors` saturates the syndrome (every detector), the blunt policy is already minimal and
  a naive signature-match is **unsound** — it accepts shots that still carry the coherent error (the
  magic magnitude collapses). Verify any finer policy against the noiseless magic value before
  trusting it.
- **Decode-and-budget**: decode everything with `dem` and carry `sum(f.probability for f in
  postselect_faults)` as a coherent-error **budget** — the total `O(p)` probability of the
  not-Pauli-correctable faults you are accepting instead of post-selecting. It is an *estimate* of
  their infidelity contribution, not a strict bound: the realized infidelity also depends on whether
  each such fault actually flips the logical and on higher-order/decode-failure terms. Note the budget
  is a *probability*, not the drop in `|⟨P̄⟩|` directly — to compare against a magic-value target,
  sample the no-post-select magic mean (see the example). Use decode-and-budget when that budget is
  acceptable for your target.

**Which to use:** default to the blunt `keep_mask` — it is always sound, and on a saturated-reject
protocol it is provably minimal (the Intuition below explains why). Reach for
decode-and-budget only when you would rather keep *every* shot and accept the known `O(p)` floor.

`reject_detectors` is the **sorted, de-duplicated union** of the flagged signatures — a *convenience*
for the blunt conservative policy, not a forced one. Both `reject_detectors` and `postselect_faults`
are empty for a Pauli/Clifford circuit (then `dem` is byte-identical to `detector_error_model()`),
and always empty with `include_expectations=False` (no column requested: every fault is in `dem`).

### Intuition: why post-select at all, and when a finer policy can (and can't) keep more shots

It is tempting to think the blunt `keep_mask` throws away too much, and that a cleverer rule could
keep more shots while staying correct. Sometimes yes; for a saturated reject region, **no** — and
here is why, in plain terms.

Think of detectors as **alarm bells**. An error makes some bells ring, and from the pattern you try
to infer what happened. Two kinds of error ring bells:

- **fixable** (ordinary Pauli) — the decoder reads the pattern and corrects it; the shot is fine;
- **unfixable** (the coherent logical-`S` that corrupts the magic) — it *also* rings bells, but no
  Pauli correction undoes it; the magic value is silently wrong.

The trouble is that **a fixable error and an unfixable error can ring the same bells.** If a shot
rings a bell that *some* unfixable fault could also ring, you cannot tell which one happened from the
syndrome alone — so the only safe move is to **discard that shot** (post-select it out). Keeping it
would sometimes mean keeping a corrupted magic value without knowing it. That is the entire reason a
reject region exists.

A "finer" policy tries to keep the shots that are *clearly* fixable and discard only the ones that
are *clearly* unfixable. Two things make that hard:

1. **Errors add up (XOR), so fingerprints smudge.** A flagged fault's catalogued signature is not the
   syndrome you actually observe — another stray error in the same shot can cancel part of it. So a
   naive "does the syndrome match a flagged signature?" test *misses* contaminated shots (it sees a
   partial fingerprint) and wrongly keeps them. This is the unsound trap: it can keep ~96% of shots
   while the magic value is no better than doing nothing. **Always verify any finer policy against the
   noiseless magic value** before trusting it.
2. **Some protocols have no "clearly fixable" pattern left.** When `reject_detectors` is the *entire*
   detector set (a *saturated* reject region), every non-trivial bell pattern the
   decoder would happily fix is *also* a pattern some unfixable fault could have produced. There is no
   safe non-trivial syndrome to rescue, so the only keepable shots are the ones where **no bell rang
   at all** — which is exactly what the blunt `keep_mask` already keeps. A cleverer rule cannot keep a
   single extra shot.

This is not a defect: a fully post-selected protocol is *meant* to discard anything imperfect, and
the exporter faithfully reports that. A finer policy only buys you
something on a *non-saturated* circuit — one with bells that *only* fixable errors ring (e.g. a real
QEC bulk you decode plus a strict-subset magic check you post-select). xtim ships no finer helper
because no such protocol is in hand yet; when one is, the sound rule is "reject a shot if some flagged
fault, combined with correctable errors, could have produced its syndrome" (the conservative
consistency test) — strictly safe, and provably never worse than blunt.

### The completeness gate (fatal, on purpose)

If a single fault changes a logical magnitude but **fires no detector**, the circuit cannot
post-select it away — it is an undetectable logical error, i.e. a fault-tolerance defect in the
circuit (usually a post-selection syndrome that was applied "by accounting" but never declared as
`DETECTOR`s). xtim refuses loudly rather than ship a DEM that silently drops it:

```
XtimDemError: non-fault-tolerant circuit (<loc>): a single fault changes a logical expectation
(its magnitude, or its sign on the accepted branch) yet is undetectable on the accepted branch (all
post-selection reads trivial); declare the missing post-selection syndrome as detectors.
```

When the fault fires no detector deterministically but randomizes some (an S-type residual on a
detector read), the gate decides "undetectable on the accepted branch" exactly (3.1.10): with
A_d = s_d·E†D_dE for each randomized detector (s_d its noiseless sign) and Q = E†P'E for each
column, the accepted-branch value of the column is Σ_S <Q·A_S> / Σ_S <A_S> over the subsets S of
those detectors, all Pauli expectations on the noiseless state. The fault is flagged when that value
equals the noiseless one (magnitude and sign) and refused otherwise; more than 16 randomized
detectors refuse without a verdict. (3.1.9 instead collapsed each randomized read's qubit in the Z
basis and re-classified, which also collapsed declared frame reads.) For a column with a declared
frame, P' is the frame-folded operator P·Π R_j (see `dem_export.hpp`).

The fix is in the **circuit**: declare every post-selection condition (the cultivation check rounds,
*and* the final output-code stabilizer round) as `DETECTOR`s. Once they are declared, the fault flips
one of them and moves from "undetectable" to "post-selectable."

**The gate refuses only when a column is requested.** With `include_expectations=False` (the
opt-out of both `detector_error_model` and `detector_error_model_with_reject`) no expectation
column is in the output, and the detector/observable DEM carries every fault on its
detector/observable signature — a refusal or an omission would name nothing the requested output
gets wrong. So the export returns that DEM (for every class-admitted circuit byte-identical to the twin circuit without the
declarations; nothing flagged, nothing dropped) and **reports** what the column analysis would have
refused in `dropped_expectation_columns`, one entry per (column, fault location), with the location
and column the refusal names:

```
(0, "non-fault-tolerant circuit (<loc>): a single fault changes the magnitude of logical expectation
     column L0 but fires no detector; declare the missing post-selection syndrome as detectors
     (reported, not refused: include_expectations=False)")
```

The two "accepted branch was not verified" refusals (more than 16 randomized detectors; the
joint law over the work cap) are reported the same way, in their own words.
`detector_error_model(include_expectations=False)` / `detector_error_model_text(...)` return no
report field and raise an `xtim.XtimDemWarning` with the same text instead. The finding is still a
fault-tolerance defect of the circuit; it is simply not a defect of the detector/observable model
you asked for. With `include_expectations=True` nothing changed: the refusal above.

---

## Worked example 1 — a genuine flagged fault (decode + flag)

A Bell pair carries a magic rotation (`T` then `T_DAG`, the identity when noiseless) with a
`DEPOLARIZE1` fault between them and a repeated `Z0Z1` parity check (runnable as
`examples/dem_reject_region.py`):

```python
import xtim
CIRCUIT = """RX 0
R 1
CX 0 1
T 0
DEPOLARIZE1(0.03) 0
T_DAG 0
MPP(0.01) Z0*Z1
MPP(0.01) Z0*Z1
DETECTOR rec[-2]
DETECTOR rec[-2] rec[-1]
MX(0.005) 1
PAULI_EXPECTATION(0) X0 rec[-1]
"""
c = xtim.Circuit(CIRCUIT)
res = c.detector_error_model_with_reject(include_expectations=True)
print(res.dem)                  # error(0.01) D0 D1 / error(0.01) D1 / error(0.0149) L0
print(res.postselect_faults)    # [PostselectFault(detectors=[0], probability=0.02)]
print(res.reject_detectors)     # [0]
```

- The `Z` alternative commutes with `T`: a Pauli frame flip of the column (the `L0` edge, together
  with the `MX(0.005)` readout flip of its frame record).
- The `X`/`Y` alternatives become `X·S`-type residuals. **Per shot** they take the frame-corrected
  `<X0>` from 1 to 0 (`c.compile_exact_branches().exact_branches([(0, 0, "X")])` shows it on every
  branch), which is a genuine magnitude change, and they fire `D0`. So they are flagged (total
  probability 0.02), not decoded.
- Post-selecting with `res.keep_mask(dets)` keeps about 97% of shots and restores `<X0>` to about
  0.97. The residual is the undetectable `Z` / readout-flip mass, which the decoder's `L0` correction
  removes.

## Worked example 2 — partial decode (a circuit with an empty reject region)

The Clifford twin `cultivation_d3_rate.stim` has an **empty** reject region (nothing flagged), so
every shot is kept and the decoder corrects genuine syndromes. (`cultivation_d3_faithful` has no
reject region either since 3.1.13; its exact DEM needs `approximate_disjoint_errors=True`.)

```python
import numpy as np, pymatching
c2 = xtim.Circuit.from_file("examples/cultivation_d3_rate.stim")
res = c2.detector_error_model_with_reject()        # empty reject region: 0 flagged faults
matcher = pymatching.Matching.from_detector_error_model(res.dem)
dets, _obs, exps = c2.compile_detector_sampler(seed=0).sample(
    100_000, separate_observables=True, return_expectations=True)

keep = res.keep_mask(dets)                 # keeps everything here (no flagged detectors)
corr = matcher.decode_batch(dets[keep])    # REAL corrections on nonzero syndromes (~27% nontrivial at p0=1e-3)
# read the magic value (exps) on the decoded shots. When the reject region is NON-empty (the general
# case), keep_mask drops the flagged shots and you carry their total probability as a budget instead.
```

## Worked example 3 — Pauli-only circuit (no change vs. plain export)

```python
REP_CODE = ("R 0 1 2\nX_ERROR(0.05) 0 1 2\nM 0 1 2\n"
            "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\n")
res = xtim.Circuit(REP_CODE).detector_error_model_with_reject()
assert res.reject_detectors == []
assert str(res.dem) == str(xtim.Circuit(REP_CODE).detector_error_model())   # byte-identical
```

---

## Notes & limitations

- **Twirled randomization.** An odd-`S` on a *measured detector* read is a probability-½ Pauli flip
  (the measurement twirl), emitted as an ordinary `error(p/2) D#` edge — not a refusal. When the same
  fault also flips a column's per-shot sign together with that detector, the joint law is exported
  (e.g. `error(p/2) D0 L#`). (`miniature_oracle` exports cleanly for this reason; since 3.1.13 its two
  formerly flagged faults are such sign laws.)
- **Genuinely non-DEM circuits.** Two distinct properties force a refusal, both unrelated to the
  reject region. (i) A **non-deterministic logical observable** (its noiseless parity isn't ±1) —
  this is what `code_switching` hits first (`XtimDemError: non-deterministic observable L0 ...`).
  (ii) **Detector mechanisms that are not a product of independent record-flips** (an exact DEM would
  need a negative probability) — the deeper failure mode some configurations also have. Either has no
  exact Pauli DEM and refuses.
- The magic value is exact via `PAULI_EXPECTATION`; never decode it as a DEM observable.
- **The reject region is a property of the protocol, not of the noise.** `reject_detectors`
  (and the DEM's edge set) are noise-independent — export them once and reuse across an
  entire noise sweep; only the per-shot detector samples change with `p`.
- **Sweeping at scale.** `DemWithReject.keep_mask(dets)` gives the boolean post-select mask
  directly (no `~`/`axis=1` to get wrong). For a `p`-sweep, `xtim.Task(keep=...)` +
  `xtim.collect(...)` runs the post-select-and-score loop for you and returns
  `acceptance_rate` / `value` / `sem` per task, e.g.
  `keep = lambda dets, meas: res.keep_mask(dets)`.
