# The xtim circuit dialect

`xtim` reads a **strict superset of the Stim circuit language**. Everything Stim
accepts — qubit coordinates, the **full Stim Clifford gate set** (every 1- and
2-qubit unitary Clifford, plus the identity `I` and gate `[tag]` suffixes), the
noise channels (`X_ERROR`, `DEPOLARIZE1/2`, `PAULI_CHANNEL_*`, …),
`M`/`MR`/`R`/`MX`/`RX`/…, `DETECTOR`, `OBSERVABLE_INCLUDE`, `rec[-k]`, `REPEAT`,
`TICK`, `SHIFT_COORDS` — parses and behaves **identically** in xtim. This page
documents only the *delta*. (One small strictness difference: xtim requires every
gate to carry at least one target, whereas Stim accepts a zero-target gate line as a
no-op; an empty-target line is rejected with a typed error rather than ignored.)

### The full Stim Clifford gate set (level-2, accepted exactly)

Every Stim unitary Clifford gate loads. The engine keeps a small native kernel set
(`H`, `S`, `S_DAG`, `X`, `Y`, `Z`, `CX`, `CZ`); every *other* Stim Clifford is
**desugared at parse time** into a word over that set that equals the gate **up to
a global phase** (unobservable in sampling). This is purely additive to the parser —
no new IR gate kind, no engine change — and each desugaring is verified two ways
against real Stim: a dense up-to-global-phase unitary match, and
record-distribution equality on random circuits (`tests/test_stim_parse_stim.py`).

- **Identity** `I q…` — a no-op (emits nothing) but still counts its qubits, so
  `I 5` widens the circuit to 6 qubits exactly as in Stim.
- **Aliases** resolve to their canonical handler with no desugar: `H_XZ`→`H`,
  `SQRT_Z`→`S`, `SQRT_Z_DAG`→`S_DAG`, `CNOT`/`ZCX`→`CX`, `ZCZ`→`CZ`, `ZCY`→`CY`,
  `SWAPCZ`→`CZSWAP`.
- **1-qubit Cliffords** desugared to `H`/`S`/Pauli words: `SQRT_X(_DAG)`,
  `SQRT_Y(_DAG)`, `H_XY`, `H_YZ`, `H_NXY`, `H_NXZ`, `H_NYZ`, `C_XYZ`, `C_ZYX`,
  `C_NXYZ`, `C_XNYZ`, `C_XYNZ`, `C_NZYX`, `C_ZNYX`, `C_ZYNX`.
- **2-qubit Cliffords** desugared to `CX`/`CZ`/`H`/`S` words: `SWAP`, `ISWAP(_DAG)`,
  `CXSWAP`, `SWAPCX`, `CZSWAP`, `CY`, `XCZ`, `YCZ`, `XCX`, `XCY`, `YCX`, `YCY`,
  `SQRT_XX(_DAG)`, `SQRT_YY(_DAG)`, `SQRT_ZZ(_DAG)`, and the 2-qubit identity `II`
  (e.g. `SWAP a b` → `CX a b ; CX b a ; CX a b`; `XCZ a b` → `CX b a`).
- **Gate `[tag]` suffix** (Stim ≥1.14, e.g. `H[mytag] 0`, `CX[t] 0 1`) is accepted
  and ignored — the bracketed tag carries no sampling semantics, so tagged circuits
  load unchanged.

Desugared 1q/2q gates inherit their analogue's broadcast, arity, distinct-qubit and
no-`()`-argument validation. The desugaring is **internal-only**: `Circuit.text`
round-trips the *original* gate tokens (`SWAP 0 1` stays `SWAP 0 1`), not the
expanded word — you never see the desugared form unless you inspect the engine IR.

A Clifford-only xtim file is a valid Stim file. The extensions follow Stim's own
syntactic rules, so they read naturally — but a file using them is not runnable by
the stock `stim` binary (it rejects non-Clifford gates). That's expected for a
superset; the exported DEM, by contrast, *is* a real `stim.DetectorErrorModel`.

> This is the reference page. To see the dialect *used* end to end:
> [`xtim_tour.md`](xtim_tour.md) is the decoder-in-the-loop workflow, and
> [`../examples/onboarding_new_protocol.py`](../examples/onboarding_new_protocol.py)
> walks a fresh protocol from circuit text to a scored fidelity via
> `Circuit.diagnose()`.

## 0. Coordinates (`QUBIT_COORDS` / `DETECTOR(coords)` / `SHIFT_COORDS`)

Coordinates are **preserved**, not just parsed: `QUBIT_COORDS(x,…) q` and
`DETECTOR(x,…) rec[…]` coordinates are stored and surfaced exactly as in Stim, with
`SHIFT_COORDS(dx,…)` resolved at parse time (cumulative, component-wise — a shorter
shift vector only offsets leading components — and accumulated once per `REPEAT`
iteration, matching Stim).

```python
c = xtim.Circuit(text)
c.get_final_qubit_coordinates()        # {qubit: [x, y, …]}  (last QUBIT_COORDS wins)
c.get_detector_coordinates()           # {det_index: [x, y, …]}  ([] if coordinate-free)
c.get_detector_coordinates(only=[0,2]) # restrict to those detectors
```

Both methods mirror `stim.Circuit`'s names/signatures and return identical values.
The exported DEM carries the geometry too: a coordinate-carrying detector emits a
`detector(x, y, …) D#` line (so pymatching/visualization keying off detector coords
get them), while coordinate-free detectors are unchanged. Coordinates are
annotations, never operations — they do not affect any record/measurement stream.

## 1. Added gates (level-3 / non-Clifford)

| Gate | Arity | Notes |
|------|-------|-------|
| `T`      | 1 | the π/4 phase, `diag(1, e^{iπ/4})` |
| `T_DAG`  | 1 | the inverse π/4 phase, `diag(1, e^{-iπ/4})` = `T†` |
| `CS`     | 2 | controlled-S, **symmetric** in its two targets |
| `CS_DAG` | 2 | controlled-S†, **symmetric** in its two targets |
| `CH`     | 2 | controlled-H, `ctrl tgt` (NOT symmetric) — the first **non-diagonal** level-3 gate. Supported end-to-end (compile, noise, sampling) whenever its magic commutes with the rest of the circuit's (the general class condition — see [`xtim_simulable_class.md`](xtim_simulable_class.md)); a non-commuting placement rejects with `kind="class"`. DEM export refuses CH circuits (their faults act as Cliffords, not Paulis) — post-select via `diagnose()` instead |
| `CCZ`    | 3 | controlled-controlled-Z, **symmetric** |

**`T_DAG` and `CS_DAG` are parse-time conveniences** that *desugar* into existing
gates — they are not distinct kinds in the IR:

- `T_DAG q` → `T q ; S_DAG q` (since `T·S† = diag(1, ζ₈⁷) = T†`),
- `CS_DAG a b` → `CS a b ; CZ a b` (since `CS·CZ = diag(1,1,1,ζ₈⁶) = CS†`).

They broadcast and validate exactly like their non-dagger analogues (`T_DAG` is
1-qubit, `CS_DAG` is 2-qubit symmetric with the same canonical qubit sort). The
desugaring is internal: `Circuit.text` round-trips the *original* token (`T_DAG 0`
stays `T_DAG 0`); the expanded pair is never surfaced.

**Grammar is Stim's, unchanged:** a gate name followed by a flat target list,
consumed in groups of the gate's arity and broadcast. No new syntax.

```
T 0 1 2                 # T on qubits 0, 1, 2
CS 3 4                  # one controlled-S on (3,4)
CCZ 0 1 2 3 4 5         # CCZ(0,1,2) and CCZ(3,4,5)
```

Parse-time validation (Stim-strict): the target count must be a multiple of the
arity, qubits within a group must be distinct (`CCZ 0 0 1` and `CH 0 0` are
rejected), and indices must be in bounds.

## 1b. Classically-controlled Pauli feedback (`CX/CY/CZ rec[-k] q`)

A `CX`/`CY`/`CZ` whose **first target is a measurement record** `rec[-k]` is Stim's
*classically-controlled Pauli feedback*: the Pauli is applied to the qubit target iff
the controlling record bit is 1. The control is the **first** target and the qubit the
**second**, matching Stim exactly:

| Syntax            | Meaning                                         |
|-------------------|-------------------------------------------------|
| `CX rec[-k] q`    | apply `X` to qubit `q` iff record `rec[-k]` = 1 |
| `CY rec[-k] q`    | apply `Y` to qubit `q` iff record `rec[-k]` = 1 |
| `CZ rec[-k] q`    | apply `Z` to qubit `q` iff record `rec[-k]` = 1 |

```
R 0
H 1
CX 1 0
M 0
CX rec[-1] 1            # X-correct qubit 1 conditioned on the just-measured rec[-1]
```

**Multi-pair broadcast (matches Stim).** Like Stim, `CX`/`CY`/`CZ` consume targets in
`(control, target)` **pairs**, broadcast over a single line, and each pair is independently
either a feedback (`rec[-k]` control) or an ordinary two-qubit gate (qubit control). A line
may even **mix** the two. All of Stim's documented idioms are accepted and produce the same
record distribution:

```
CX rec[-3] 2 rec[-1] 2                            # TWO X-feedbacks onto q2 (q2 ^= rec[-3] ^ rec[-1])
CY rec[-4] 0 rec[-3] 0 rec[-3] 3 rec[-2] 3 rec[-1] 0   # five Y-feedbacks
CZ rec[-4] 1 rec[-1] 1                            # two Z-feedbacks onto q1
CX 0 1 rec[-1] 2                                  # MIXED: normal CX(0,1) + X-feedback on q2
```

Each feedback pair becomes its own controlled-Pauli IR instruction (so *N* independent
feedbacks compose through the same triangular relabel); each `(qubit, qubit)` pair stays
byte-identical to the ordinary gate path. A `rec[]` may only be a **control** (the first of a
pair) — `rec[]` as a gate *target* is rejected (it is never a runnable gate; Stim rejects the
samplable cases as "measurement record editing"), and an odd target count is rejected
("requires an even number of targets") exactly as Stim does.

`rec[-k]` is relative (`k≥1`, the k-th most recent measurement), validated in range at parse
time (a control before any measurement, or a non-negative `rec[5]`, is rejected). The
in-scope class is exactly the **Paulis** `X/Y/Z`; this is faithful because a Pauli before a
measurement only flips that measurement's outcome (and contributes a conditional sign to a
`PAULI_EXPECTATION` channel) — never a Born probability — so it is an exact, triangular,
post-sampling record/expectation relabel rather than a state-evolution operation.

**One semantics, two mechanisms.** Every consumer implements the table above — the
feedback reads the *recorded* bit (`!` invert and `M(p)` flip included) and applies the
Pauli — but by two different exact mechanisms, chosen per consumer:

*The exact engine and the `run_stim_main` CLI: a post-sampling relabel (off the hot loop).*
(DEM export is not a third consumer: it refuses every feedback circuit, see "Out of scope".)
A feedback whose Pauli byproduct would cross a non-Clifford gate (`T`/`CS`/`CCZ`/`CH`)
is first **coherentized** — rewritten into a coherent controlled-Pauli with the control in
the source measurement's basis (see below) — because a byproduct crossing a `T` propagates to
a non-Pauli Clifford that rotates an expectation rather than flipping a sign. Every remaining
controlled-Pauli is pulled OUT of the bare circuit (so the bare state / records / expectations
are feedback-free and unchanged), and each is propagated once at setup through the existing
propagation table to a descriptor `(control record, end-of-circuit record-flip mask,
expectation-sign mask)`. Per shot, AFTER the normal sampling and AFTER the record-flip pass
(`!` / `M(p)`), a triangular relabel pass runs in record order: if a controlled-Pauli's
control record is 1, its record-flip mask is XOR'd into the (later) records and its
expectation-sign bits flip the selected `PAULI_EXPECTATION` channels. The noiseless reference
that detector/observable channels are taken relative to goes through the same two passes in
the same order. Causality makes the control record precede every record it flips ⇒ the
relabel is triangular ⇒ resolvable in one forward pass; an acausal/cyclic feedback (a
controlled-Pauli that flips a record at or before its own control) is rejected. The pass is
entered **only** for circuits that contain feedback, so feedback-free circuits keep a
byte-identical hot loop and stream.

*The twirl engine (the `TwirlSampler` family behind `engine="twirl"`, `compile_twirl_sampler`,
`sample_barrier` / `materialize`): coherentize everything.* This path runs no relabel, so
**every** controlled-Pauli is rewritten into real gates before deferral
(`NormalizePolicy::coherentize_all`), whether or not it crosses a non-Clifford gate:

- an idle control whose target is **another** wire drives the entangler directly from its
  wire, in its measured basis (`H`-bracketed for an `X` read, `S†;H`-bracketed for a `Y` read;
  `!` adds an unconditional Pauli on the target);
- a control **wire reused** between its measurement and the feedback has its eigenvalue copied
  onto a fresh ancilla just before the measurement, and the feedback drives from the copy;
- a feedback onto its **own** control wire (`M q; CX rec[-1] q`, measure-and-conditionally-
  reset) takes the same copy gadget even with an idle control: driven in place it would be the
  degenerate `CX q q`, which is not a unitary on the wire (the coherentizer refuses to emit a
  controlled-Pauli whose control is its target);
- a noisy source `M(p) c` is folded into a physical anticommuting error placed just before the
  measurement, so the recorded bit and the driving value flip **together**. That error also
  flips the control's projected state, which Stim's `M(p)` does not; so when the control's
  state is observed again before a `R` (re-measured, gated, named in a `PAULI_EXPECTATION` or
  an `OUTPUT_QUBITS` port), the flip instance is held on a fresh ancilla and applied to the
  control before **and** after the measurement — record flipped, state restored, shot for shot
  — with the feedback driven from a copy ancilla.

Consequently a feedback's control wire (and any copy/flip ancilla) stays **live across the
deferral** on the twirl path, and a circuit's qubit count as seen by the engine can exceed
the text's; the flip ancilla of a held fold is never measured or reset (a classical mixture
controlling two cancelling CXs — harmless, one extra live wire). The cost is compile/setup
only — the per-shot hot path is unchanged — and it scales with the live wire count: nil
beyond the wires the circuit already had for the in-place class, and about 3× the compile
time at N=40 chained noisy-observed sources for the held form (2N extra wires), with no
measurable per-shot change. Feedback-free circuits take an early-out and are unaffected.
Feedback that crosses a non-Clifford gate is coherentized identically on both routes.

**Out of scope:** classically-controlled **Cliffords** (e.g. a record-controlled `S`/`H`)
— a Clifford conditioned on a bit changes frame *structure*, not just a sign, so it is not a
record relabel. `sweep[k]`-bit controls (`CX sweep[0] 0`) are also out of scope; both are
rejected with a clear message. (Stim itself accepts `sweep[k]`; xtim does not.) DEM export
also rejects feedback (a DEM has no record-control concept).

> Status: **supported** on both engines since 3.1.1 (parser + IR, the propagate + per-shot
> relabel, the full A1–A6 analytical battery, the random-Clifford Stim record-equality
> campaign, the magic-teleportation dense-oracle check and the speed/stream-pin proof were
> established on the exact route; through 3.1.0 the twirl route silently dropped non-crossing
> feedback and answered wrong on feedback onto its own control wire, and the exact route's
> detector columns were wrong whenever an `H` was present — see CHANGELOG [3.1.1], which
> lists the exact classes that moved). A feedback circuit samples through the Python API and the
> `run_stim_main` CLI, and its record distribution matches real Stim (which runs
> `rec[-k]`-controlled Paulis natively). The cross-engine / Stim oracle suite is
> `tests/test_twirl_feedback_semantics.py`; the coherentizer's structural contract is
> `cpp/tests/test_coherentize_all.cpp`.

## 2. The `PAULI_EXPECTATION` declaration

The one output channel Stim has no analog for: **report the raw expectation value
of a Pauli operator** (a magic-state expectation is not a measurement parity, so it
can't be an `OBSERVABLE_INCLUDE`).

```
PAULI_EXPECTATION(0) X0*X1*X2*X3*X4*X5*X6 rec[-3] rec[-1]  # report <X̄>; frame: records -3, -1
PAULI_EXPECTATION(1) Y0*Y3*Y5                                # report <Ȳ>; sign already constant
```

- One `*`-joined Pauli product per labeled instruction, using Stim's existing MPP
  token syntax (`*` is the Pauli-product combiner; not new punctuation). Each term
  is a Pauli letter `X`/`Y`/`Z` followed by a qubit index.
- The `(index)` is an integer **label**; labels must be distinct. Several
  designated Paulis → several `PAULI_EXPECTATION` lines.
- **Optional declared byproduct frame: `rec[-k]` targets.** A trailing `rec[-k]`
  target list (the same grammar as `OBSERVABLE_INCLUDE`) declares the **byproduct
  Pauli frame** — the measurement records whose parity fixes the SIGN of ⟨P⟩. The
  semantics parallel `OBSERVABLE_INCLUDE`, except the noiseless value is not
  required to be ±1 (it can be any `|⟨P⟩|`, e.g. 1/√2 for a magic-state
  expectation). With a declared frame the engine folds the record-parity into the
  per-shot sign, so `exps[:, i]` is **sign-constant** in the noiseless case:
  `|⟨P⟩|` is the frame-independent physical magnitude; the SIGN is the convention
  in the frame where the declared records all read 0; and the target ⟨P⟩ is the
  mean of the per-shot signed values. A column whose sign is already constant
  (no record governs it) needs no frame declaration.
- **Strict frame verification — refuse on wrong frame.** The engine computes the
  EXACT set of records that govern the sign (bare-state stabilizer anticommutation
  analysis) and REFUSES (`XtimRejectError`) if the declared `rec[-k]` set is
  missing a sign-controlling record OR includes a spurious one — mirroring Stim's
  refusal of a non-deterministic observable. A column with no solvable fixed frame
  (a genuine free/logical read — e.g. a destructive `MPP` readout like `mpp_magic`,
  or a magnitude-0 equator channel) is NOT refused: it reports
  `signed_value = nan`, `sign_constant = False`, `frame_hint = None`
  ("sign varies; free/logical read; no fixed byproduct frame").
- **Per-shot output.** The engine emits the frame-corrected expectation value as a
  raw `float64`, exact up to floating-point rounding (≈1 ulp) — for example `⟨X⟩`
  on a noiseless `T|+⟩` with a declared frame reads `+1/√2` on every shot
  (sign-constant), and a true-zero column (e.g. `⟨Z⟩` on `T|+⟩`) reads `0` to
  within rounding (~1e-16, not bit-zero). Any further interpretive step
  (post-selection, β-combination into a fidelity) is user-side analysis.
- **`_discover_frame` retired.** The earlier statistical sign-inference pass is
  removed. It silently produced a shot-count-dependent wrong sign at low shot
  counts. `signed_value` is now exact and shot-independent; `frame_hint` echoes
  the DECLARED `rec[-k]` targets, not a statistically-discovered parity.

## 2b. `MPP` — Pauli-product measurement (parse-time desugar)

`MPP` measures one or more `*`-joined Pauli products, each producing **one
measurement record** in Stim's record order. It is supported by **parse-time
desugaring into an ancilla gadget** — no engine/sampler change — so circuits that
do not use `MPP` are byte-identical and same-speed as before.

```
MPP X0*X1                 # one record: the ±1 measurement of X0 X1
MPP X0*X1 Z2*Z3           # two records, in order (one ancilla each)
MPP !Y0*Y1*Z4             # ! inverts the recorded bit; Y factors allowed
```

Each product `P = p_0(q_0)·p_1(q_1)·…` desugars to:

```
H a                       # fresh |0> ancilla a (allocated ABOVE every user qubit) -> |+>
C{p_i} a q_i              # one controlled-Pauli per factor: X->CX, Y->CY, Z->CZ (control=a)
Z a                       # ONLY if the product carries a leading `!` (flips the X-basis read)
MX a                      # terminal X-measure of a = the MPP record
```

This is structurally a cat-check; the abandoned ancilla flows through the existing
terminal-Pauli deferral. The gadget, including the **sign conventions**, is pinned
byte-for-byte against real Stim (Stim runs `MPP` natively):

- the `i^{#Y}` normalisation of a Y-containing product is absorbed by Stim's own `CY`
  definition (no extra correction);
- a leading `!` records the **complement** bit — emitted as a `Z` on the ancilla
  before `MX` (a `Z` anticommutes with the X-basis read; an `X` would *not* flip it).

Validation is Stim-strict: each factor is `X`/`Y`/`Z`<qubit>, a product may not repeat
a qubit (`MPP X0*X0` rejects), and a bad Pauli letter (`MPP Q0`) rejects with a clear
message. Note the desugar ancillas raise `circuit.num_qubits` above Stim's count (Stim
hides MPP ancillas); measurement/detector/observable counts and the record distribution
match Stim exactly.

## 2b-ii. `MXX` / `MYY` / `MZZ` — two-qubit Pauli-parity measurements (parse-time desugar)

`MXX`, `MYY`, and `MZZ` measure the two-qubit parity of a fixed Pauli axis over **qubit
pairs**, each pair producing **one measurement record**. The target list is broadcast in
pairs: `MZZ 0 1` produces one record; `MZZ 0 1 2 3` produces two records in order.

```
MZZ 0 1               # one record: the ±1 measurement of Z0·Z1
MXX 0 1 2 3           # two records: XX on (0,1), then XX on (2,3)
MYY 0 1               # one record: Y0·Y1
MZZ !0 1              # ! on EITHER qubit of a pair inverts that pair's record
MZZ 0 !1              # equivalent inversion (same record complement)
```

These are supported by **parse-time desugaring into `MPP`**: each pair `(q0, q1)` becomes
one `MPP P<q0>*P<q1>` product, and the existing MPP gadget handles the rest. Stim's
product-sign convention is inherited for free. A `!` prefix on either qubit of a pair
inverts that pair's record (the two `!` prefixes XOR: one inverts, two cancel):

```
MZZ 0 1 2 3   →   MPP Z0*Z1 Z2*Z3
MXX !0 1      →   MPP !X0*X1
MYY 0 !1      →   MPP !Y0*Y1
```

The target count must be even and positive; an odd count is rejected. Record distribution
matches real Stim (which runs `MXX`/`MYY`/`MZZ` natively), verified by affine GF(2)
equality in `tests/test_measurement_gaps_stim.py`.

## 2b-iii. `!q` — inverted measurement targets

Any qubit target in a measurement instruction may carry a **`!` prefix** to invert the
recorded bit without disturbing the post-measurement quantum state:

```
M !0          # measure qubit 0, record ~outcome  (record-only complement)
M 0 !1        # measure 0 normally, invert qubit 1's record
MX !3         # X-basis measurement with inverted record
MRY !5        # Y-basis measure-and-reset with inverted record
MPP !X0*Y1    # product measurement with the product-record inverted
MZZ !0 1      # parity measurement with its record inverted  (! on EITHER qubit)
```

The `!` prefix is accepted on all measurement gates: `M`/`MX`/`MY`/`MZ`, `MR`/`MRX`/`MRY`,
on individual factors of `MPP` products, and on either qubit of an `MXX`/`MYY`/`MZZ` pair.
On parity pairs the two prefixes XOR: one `!` inverts, two cancel. The semantics match Stim
exactly: the collapsed state is the **true** eigenstate (un-inverted), only the reported bit
in the record is flipped. Verified vs real Stim by affine GF(2) equality in
`tests/test_measurement_gaps_stim.py`.

## 2b-iv. `M(p)` — readout-flip probability (record-only Bernoulli noise)

Any measurement gate may carry an **optional probability argument** `(p)` to model
classical readout error:

```
M(0.01) 0        # measure qubit 0; record is flipped w.p. 0.01
MX(0.05) 3       # X-basis measurement with 5% readout-flip noise
MR(0.02) 1       # measure-and-reset; record flipped w.p. 0.02, state reset cleanly
MZZ(0.1) 0 1     # parity measurement with 10% record-flip noise
```

The `(p)` argument is a **record-only** Bernoulli flip: the engine samples the true
measurement outcome, then independently flips the **recorded** bit with probability p.
The post-measurement quantum state collapses to the **true** (un-flipped) outcome in all
cases — this is Stim's exact semantics. In particular, `M(p)` differs from
`X_ERROR(p); M` (which would perturb the state) for mid-circuit measurements; only for
terminal measurements or `MR(p)` are the two rewrites equivalent.

Accepted on all measurement gates: `M`/`MX`/`MY`/`MZ`, `MR`/`MRX`/`MRY`,
`MXX`/`MYY`/`MZZ` (forwarded through the MPP desugar). Probability must be in [0, 1];
`p = 0` is accepted and is a no-flip no-op. Verified vs real Stim by z-score sampling
tests in `tests/test_measurement_gaps_stim.py`.

## 2c. `SPP` / `SPP_DAG` — Pauli-product rotation gate (parse-time desugar)

`SPP P` = `exp(-iπ/4·P)` and `SPP_DAG P` = `exp(+iπ/4·P)` are **Clifford gates** (no
measurement, no record). Both are supported by **parse-time desugaring into a
basis-change ladder** — no engine/sampler change, no ancilla, no new gate kind — so
circuits that do not use `SPP` are byte-identical and same-speed as before.

```
SPP X0                    # exp(-iπ/4 X) = √X  (the single-factor anchor)
SPP X0*Y1*Z2              # rotation by the Pauli product
SPP_DAG Z0                # exp(+iπ/4 Z) = S†
SPP !X0*Y1                # ! negates the product -> flips the rotation sense
```

For a product `P = p_0(q_0)·p_1(q_1)·…` with **pivot** = the first factor's qubit, the
desugar is (over `{H, S, S_DAG, CX}` only):

```
# (1) rotate each factor's local Pauli to +Z      X->H ; Y->S_DAG,H ; Z->(nothing)
# (2) fan every NON-pivot factor onto the pivot    CX q_i pivot
# (3) pivot rotation                               S pivot   (SPP)   /  S_DAG pivot (SPP_DAG)
# (4) uncompute the fan (reverse CX) then the pre  Y->H,S ; X->H ; Z->(nothing)
```

The ladder collapses `P` onto `Z_pivot`, where the `S`/`S_DAG` performs the `exp(∓iπ/4 Z)`
rotation, then uncomputes. The **`!` inversion** (Stim allows it on any factor; each
negates the product) is the **XOR-parity** of the `!` prefixes: an odd parity sends
`P → -P`, flipping the rotation sense — implemented by **toggling `S`↔`S_DAG`** on the
pivot. The desugar word is pinned against real Stim (which runs `SPP`/`SPP_DAG`
natively): the **dense unitary** equals Stim's gate up to a global phase, and the
**record distribution** equals Stim's on random Clifford circuits. Validation is
Stim-strict (`X`/`Y`/`Z`<qubit> factors, no repeated qubit, clear reject on a bad term).

## 3. Output channels

`compile_detector_sampler(...).sample(shots)` returns just `dets` (detection events)
by default — matching Stim's convention. Pass `separate_observables=True` to also get
`obs` (observable flips, reference-relative — Stim's exact detection-event /
logical-flip semantics) and `return_expectations=True` to also get `exps` (the raw
`float64[shots, R]` expectation values); with both, one **single** engine run (one RNG
stream) returns the `(dets, obs, exps)` tuple. `compile_sampler` returns raw
measurement bits. **The engine emits raw physics only.**

## 4. The detector error model

`detector_error_model()` returns a real `stim.DetectorErrorModel`. The
`PAULI_EXPECTATION` declarations appear as **extra L-columns after the
observables**, one per declaration in declaration order — `circuit.expectation_columns`
gives their indices, so any DEM-compatible decoder (e.g. pymatching) consumes the
DEM unmodified:

> Note: this is about the **DEM**. `sinter` is a *sampling* harness that drives a
> `stim.Circuit`; it cannot host the `PAULI_EXPECTATION` payload, so use xtim's own
> `xtim.collect` / `xtim.Task` for the magic-state sampling loop — the DEM here is
> still consumable by any DEM-level decoder.

```python
flips = matching.decode_batch(dets)
corrected = np.where(flips[:, c.expectation_columns], -exps, exps)
```

**Honest gate:** if a declared Pauli's action under some error mechanism is not a
clean ±1 sign, the export refuses loudly with `XtimDemError`, naming the column and
mechanism — it never approximates. Pass `include_expectations=False` for the
classic detector/observable-only DEM.

## 5. What is NOT supported — the v1 class boundary

The level-3 gates above keep the state in a **diagonal-Clifford-simulable** class.
The boundary that buys the efficient simulation: a Hadamard (or any gate that is a
basis change) **on a wire that carries magic** leaves that class and is **rejected**
— `XtimRejectError`, naming the offending gate index. The common cause is a
measurement written the long way: an `H` followed by an `M` to read in the X basis.
Write the native basis measurement instead — `MX` / `MY` / `MZ` (and `RX`/`RY` for
resets) stay in class. The reject diagnostic spells the cause and this fix out at
the raise site; read it rather than guessing the gate index.

## 6. Rejects, stabilizer rank, and the reference cache

- **Out-of-class reject.** If a circuit's non-Clifford content genuinely leaves the
  efficiently simulable class (see §5), compilation raises `XtimRejectError` with the
  offending gate index and an actionable hint. This is the **sole** rejection
  criterion. (Parse errors raise `XtimParseError` with line numbers.)
- **Stabilizer rank χ is a cost knob, not a gate — there is no χ cap.** xtim samples
  any output **exactly**, whatever its stabilizer rank χ. χ≤2 (a single logical magic
  state) is just the fast, cached sweet spot; higher χ (multi-magic outputs — a
  transversal-gate magic state, a 15-to-1 distillation, CCZ factories) is equally
  exact, it only costs more per shot. xtim never rejects a circuit for its χ.
- **Invisible reference + cache.** The first `compile_*` deduces the circuit's
  reference/bare state and caches it under `.xtim_cache/` (relative to the working
  directory; override with `xtim.cache_dir`). Later runs load it; the cache is safe
  to delete. `c.reference_info()` reports cache status and χ on request.
