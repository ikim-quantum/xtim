# xtim vs Stim — feature-gap audit

**Reference:** Stim **1.16.0** (introspected, not from memory), conda env `qec`.
**xtim under audit:** the `error-propagation` worktree — `xtim/` Python package +
`cpp/src/stim_parse.cpp` (the parser), `cpp/bindings/xtim_py.cpp`,
`cpp/apps/run_stim_main.cpp`, `cpp/src/dem_export.cpp`.
**Audience weighting:** researchers/students testing their **own** magic-state-prep
protocols — they write extended-Stim circuits, sample, get detection events +
observable flips + Pauli-expectations, build DEMs, decode (pymatching), compute
LER/fidelity under post-selection. Weighted HIGH: circuit IO / sampling / noise /
detectors+observables / DEM / decoding interface / formats / CLI / diagrams.
Weighted LOW: stabilizer-algebra tools (Tableau / PauliString / FlipSimulator
arithmetic). **By xtim's design** the engine emits RAW physics only — corrections,
β-combination, post-selection are user-owned numpy; there is deliberately no
Target/account/k API. Those are NOT counted as gaps.

Classification: **Supported** / **Partial** / **Missing** / **N/A-by-design**.

---

## HEADLINE — existing Stim circuits xtim will REJECT

The xtim parser (`cpp/src/stim_parse.cpp`) is a hand-written allow-list. It accepts
**31** of Stim's ~81 instruction names (+ a few aliases). A realistic Stim circuit
written by a QEC researcher routinely uses constructs xtim rejects. Empirically
tested by loading a one-line circuit per construct via `xtim.Circuit(...)`:

### Accepted (34)
`H S S_DAG X Y Z CX(=CNOT) CZ T T_DAG CS CS_DAG CH CCZ`,
`M(=MZ) MX MY R(=RZ) RX RY MR(=MRZ) MRX MRY`,
`MPP SPP SPP_DAG` (RESOLVED — parse-time desugar, see below),
`X_ERROR Y_ERROR Z_ERROR DEPOLARIZE1 DEPOLARIZE2 PAULI_CHANNEL_1 PAULI_CHANNEL_2`,
`DETECTOR OBSERVABLE_INCLUDE QUBIT_COORDS SHIFT_COORDS TICK REPEAT`.
(Plus the xtim-only `PAULI_EXPECTATION`.)

### Rejected — and how much it hurts this audience

> **⚠ Superseded by v0.3 — read this first.** The `REJECT unknown` rows below for the
> **full Stim Clifford gate set** are out of date: `I`, the `SWAP`/`ISWAP`/`CXSWAP`/
> `CZSWAP`/`SWAPCX` family, `SQRT_X/Y/XX/YY/ZZ(+_DAG)`, the basis-cycling
> `H_XY`/`H_YZ`/`C_XYZ`/`C_*`, the `XCX`…`ZCZ`/`CY` controlled-Paulis, all canonical
> **aliases** (`ZCX`, `H_XZ`, `SQRT_Z`, …), and gate `[tag]` suffixes are **now
> SUPPORTED** (parse-time desugar + alias table; CHANGELOG `[0.3.0]`, verified by
> `tests/test_gate_coverage_stim.py`). The constructs xtim **still rejects**:
> `CORRELATED_ERROR`/`E`/`ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`/
> `HERALDED_PAULI_CHANNEL_1`, `sweep[k]` feed-forward, classically-controlled
> **Cliffords**, and DEM-with-feedback.

| Stim construct | xtim result | Relevance | Note |
|---|---|---|---|
| **`I`** (identity) | REJECT `unknown instruction 'I'` | **HIGH** | Ubiquitous as a timing/noise placeholder. Trivial to add (no-op like TICK). Most surprising gap. |
| `SWAP ISWAP ISWAP_DAG CXSWAP CZSWAP SWAPCX` | REJECT unknown | **HIGH** | SWAP-routing is standard in surface/qLDPC layouts; many published circuits use them. |
| `SQRT_X SQRT_X_DAG SQRT_Y SQRT_Y_DAG` | REJECT unknown | **HIGH** | Common single-qubit Cliffords (e.g. √X in many SE schedules). |
| `SQRT_XX/YY/ZZ (+_DAG)` | REJECT unknown | MED | Native 2-qubit Cliffords in some hardware-dialect circuits. |
| `H_XY H_YZ H_NXZ …`, `C_XYZ C_ZYX C_*` | REJECT unknown | MED | Basis-cycling Cliffords; appear in fault-tolerant SE / Pauli-frame circuits. |
| `XCX XCY XCZ YCX YCY YCZ ZCX ZCY ZCZ`, `CY` | REJECT unknown | MED | Generalized controlled-Paulis. `CY` and `ZCX/ZCZ` (aliases of CX/CZ) especially common. |
| **`MPP SPP SPP_DAG`** | **RESOLVED** (parse-time desugar) | **HIGH** | Pauli-product measurement (`MPP`) and Pauli-product rotation gates (`SPP`/`SPP_DAG`), heavily used in LS / lattice-surgery and many SE schedules. **Now supported** by parse-time desugaring (`MPP`→ancilla cat-check gadget per product → one record each; `SPP`/`SPP_DAG`→basis-change CX-ladder + `S`/`S_DAG` on a pivot). No engine/sampler/kernel change; record distribution byte-matches real Stim incl. `i^{#Y}` / `!`-inversion sign conventions; verified vs Stim (Clifford) + the dense oracle (magic-carrying). See §4 + the speed note below. |
| **`MXX MYY MZZ`** | **RESOLVED** (parse-time desugar to `MPP`) | MED | Two-qubit-parity convenience names. **Now supported**: each pair `(q0,q1)` desugars to one `MPP P<q0>*P<q1>` product (reusing the existing MPP gadget); `!` on either qubit of a pair inverts that pair's record (two `!` cancel); the `(p)` readout-flip arg is forwarded to emit_mpp. Record distribution matches real Stim, verified by affine GF(2) equality in `tests/test_measurement_gaps_stim.py`. |
| **`MPAD`** | **RESOLVED** (fresh-ancilla + invert) | LOW | Padding measurement. **Now supported**: each space-separated target is a padding VALUE (`0`/`1`, not a qubit) that appends one measurement record carrying that fixed bit. Desugars each value to a fresh `\|0>` ancilla measured in Z (records 0) with the record-only `invert` flag set iff the value is `1` — reusing the existing MPP fresh-ancilla allocator + invert mechanism (no new IR). Record distribution matches real Stim by affine GF(2) equality in `tests/test_measurement_gaps_stim.py`. |
| `CORRELATED_ERROR(=E) ELSE_CORRELATED_ERROR` | REJECT "use PAULI_CHANNEL_*" | MED | Correlated-error injection is common in DEM/error-budget studies. |
| `HERALDED_ERASE HERALDED_PAULI_CHANNEL_1` | REJECT (recognized, unsupported) | MED | Erasure/heralded noise — a growing modeling need. v0.3 gives a clear "recognized Stim op, not supported" message (not "unknown instruction"). |
| **`I_ERROR II_ERROR II`** | **RESOLVED** | LOW | Newer Stim identity/error ops. **Now supported**: `II` is the 2q identity (no-op desugar word, already in the table). `I_ERROR(p…) q…` / `II_ERROR(p…) q…` are identity-ERROR placeholders — pure NO-OPs for sampling regardless of the probability args (verified vs Stim 1.16.0: `X_ERROR(0.5); I_ERROR(0.5); M` leaves the X_ERROR rate unchanged). They emit nothing into the stream (0 measurements, 0 state effect) but still `touch()` each qubit target so `circuit.n` matches Stim; any `()` prob args are validated finite in `[0,1]`; `II_ERROR` rejects an odd target count. Verified vs real Stim in `tests/test_measurement_gaps_stim.py`. |
| **Aliases `ZCX ZCY ZCZ H_XZ SQRT_Z SQRT_Z_DAG SWAPCZ`** | REJECT unknown | MED | Canonical forms (`CX/CZ/H/S/…`) are accepted but the Stim-valid **aliases are not** — a file using `H_XZ` or `ZCX` (both 100% standard Stim) breaks. Cheap to fix (alias table). |
| **gate tags `H[tag] 0`** (Stim ≥1.14) | REJECT `bad qubit target '[tag]'` | MED | Tagged instructions appear in modern Stim output; parser treats `[tag]` as a qubit token. |
| `X_ERROR[tag](p)` tagged noise | REJECT (arg-count) | LOW | Same tag issue on noise. |
| **sweep bits** `sweep[k]` | REJECT "not supported in v1" | MED | Data-driven feed-forward (sweep-bit-controlled Paulis). Still **deferred** — `rec[-k]`-controlled Paulis are now resolved (below), but a `sweep[k]` control is an *externally-supplied* bit, not a measurement record; it has no triangular-relabel form (no producing measurement to read), so it stays out of scope with a clear reject. |
| **classically-controlled PAULI feedback** `CX/CY/CZ rec[-k] q` | **RESOLVED** (exact post-sampling relabel) | MED | Stim's measurement-record-controlled Pauli feed-forward. **Now supported**: a `CX/CY/CZ` whose first target is `rec[-k]` is an `X`/`Y`/`Z` on qubit `q` gated by record `rec[-k]`. Handled as an **exact post-sampling triangular GF(2) record-relabel** (a Pauli before a measurement only flips that outcome / an expectation sign — never a Born probability), computed once at setup by propagating each controlled-Pauli through the existing propagation table; per shot it's a cheap forward pass over records, entered ONLY for circuits that contain feedback. **OFF the hot loop** — feedback-free circuits are byte-identical and same-speed (stream pins 6/6, see the feedback speed note below). Verified vs hand-derived A1–A6 + real Stim (record-distribution equality, 48-circuit campaign) + a dense oracle (magic teleportation). Boundary: classically-controlled **Cliffords** (`controlled-S/H/…`) are NOT a record relabel (a Clifford conditioned on a bit changes frame *structure*) — **deferred**; `sweep[k]` controls deferred (above); DEM-with-feedback deferred (the DEM export cleanly rejects feedback — a DEM has no record-control concept). |
| **inverted targets** `MX !0`, `M !0` | **RESOLVED** (parse-time negation flag) | LOW | Negated measurement results. **Now supported**: a `!` prefix on a qubit target flips the recorded bit without disturbing the post-measurement state (record-only complement). Accepted on `M`/`MX`/`MY`/`MZ`/`MR`/`MRX`/`MRY`, on individual `MPP` factors (`MPP !X0*Y1`), and on `MXX`/`MYY`/`MZZ` qubit pairs. Verified vs real Stim by affine GF(2) equality in `tests/test_measurement_gaps_stim.py`. |
| **readout-flip `M(p)` / `MR(p)` / `MX(p)` etc.** | **RESOLVED** (record-only Bernoulli flip) | MED | Stim's `M(p)` flips only the **reported record** with probability p; the post-measurement state collapses to the **true** outcome. **Now supported**: the `(p)` argument on any measurement gate (`M`/`MX`/`MY`/`MZ`/`MR`/`MRX`/`MRY`, `MXX`/`MYY`/`MZZ`) is a readout-flip probability — the sampler adds an independent Bernoulli(p) bit to the recorded outcome per shot, without perturbing the state. The key constraint (`X_ERROR(p); M q` rewrite is unsound for mid-circuit measurements) is handled by applying the flip at record-emit time, not as a pre-measurement error. Verified vs real Stim by z-score sampling tests in `tests/test_measurement_gaps_stim.py`. Note: Pauli-`OBSERVABLE_INCLUDE` targets remain deferred. |

**Bottom line:** any Stim circuit containing `I`, a `SWAP`/`SQRT_X`-family gate, a
Stim alias like `H_XZ`/`ZCX`, or a `[tag]` will
**fail to load**. (`MPP`/`SPP`/`SPP_DAG`, `MXX`/`MYY`/`MZZ`, `!q` inverted targets, and
`M(p)` readout-flip are now RESOLVED via parse-time desugar or record-emit Bernoulli flip.)
For the MSP audience the most painful remaining walls are `I`, the SWAP family,
`SQRT_X/Y`, and the aliases — all common in real circuits, several trivial to add
(identity, aliases, SWAP, √X-family are all Clifford and in-class).

> **UPDATE — Clifford gate coverage is now COMPLETE and locked in with full-tableau probes.**
> Every one of the gates listed in the "Rejected" table above (the SWAP family,
> `SQRT_X/Y/XX/YY/ZZ(+_DAG)`, the `H_*`/`C_*` basis-cyclers, the generalized
> controlled-Paulis `XCX…ZCZ`/`CY`, the aliases, `[tag]` suffixes, `I`, `II`) is now
> accepted AND verified against real Stim 1.16.0.
>
> `tests/test_gate_coverage_stim.py` (Tier-1) sweeps the FULL accepted Clifford set —
> **23 single-qubit + 27 two-qubit = 50 gates** — with probe circuits that pin each gate's
> **full single- or two-qubit tableau** (each gate applied exactly once per block, no
> double-application cancellation). The 1q probe uses 9 blocks (3 Pauli input preps × 3
> Pauli output bases); the 2q probe uses 81 blocks (9 product-state preps × 9 basis-pair
> measurements). A wrong desugar within any former collision class (e.g. `CX→CZ`,
> `I→Y`, `CY→II`, `SQRT_X→SQRT_X_DAG`) now makes the affine record spaces differ and
> the suite fails. This is verified in `test_distinguishability`: the cross-circuit check
> xtim(G) vs stim(wrong_G) returns `_affine_equal = False` for all four historically-
> colliding pairs. The only indistinguishable pairs are genuine tableau aliases by Stim's
> definition (`H==H_XZ`, `ZCX==CX==CNOT`, `CZSWAP==SWAPCZ`, `ZCY==CY`, `ZCZ==CZ`).
> Each gate also parses with a `[tag]` suffix. Note: Stim has no `X_DAG/Y_DAG/Z_DAG`
> (X/Y/Z are self-inverse), so those are absent by design. The three remaining unsupported
> instructions `MPAD`, `I_ERROR`, `II_ERROR` are now **RESOLVED** too (see the rows below).

---

## MPP / SPP speed-preservation note (the user's explicit requirement)

`MPP`/`SPP`/`SPP_DAG` were added as a **parse-time desugar only** — an additive
branch in `cpp/src/stim_parse.cpp` that fires solely on those tokens. No sampler,
kernel, deferral or reference-state code was touched, so a circuit that uses none of
them is unchanged at every level. This is verified two ways:

**(1) Byte-identical streams (the no-leak proof).** None of the speed-critical
benchmark circuits (`code_switching_faithful`, `cultivation_d3_faithful`, and the
rate twins `code_switching_rate`, `cultivation_d3_rate`) contain `MPP`/`SPP` — only
the new test fixture `benchmarks/mpp_magic.stim` does. `tests/test_stream_pins.py`
re-hashes every channel buffer (`meas`/`dets`/`obs`/`exp`) for the 5 pinned configs
with SHA-256 and they remain **6/6 byte-identical** to the committed pins post-MPP.
A byte-identical output stream is a strictly stronger guarantee than equal speed: the
non-MPP code path produces the exact same bytes it did before the MPP work.

**(2) Measured µs/shot, unchanged.** Marginal cost (20k→200k shots, seed 2026,
`cpp/build_rel` Release binary, `run_stim_main … --b8`), best-of-2 on this machine:

| circuit | µs/shot (post-MPP) | prior reference (WORKLOAD_PROFILE R-round3, same machine class) |
|---|---|---|
| `cs_faithful` (`--ref`, χ=2) | **0.94** | 1.13 |
| `d3_faithful` (`--ref`, χ=2) | **0.68** | 0.85 |
| `code_switching_rate` | **0.17** | ~0.22 |
| `cultivation_d3_rate` | **0.22** | ~0.22 (d3 rate "unchanged") |

All four are within run-to-run noise of the pre-MPP reference numbers (the CascadePlan
fast path holds at χ≤2; the rate twins go through the general/cascade branches exactly
as before). Conclusion: **adding MPP/SPP did not change the speed of the magic-state-prep
circuits** — guaranteed by construction (byte-identical pins) and confirmed by
measurement.

---

## Classically-controlled Pauli feedback speed-preservation note (the binding requirement)

`CX/CY/CZ rec[-k] q` feedback was added as an **exact post-sampling triangular
record-relabel** that is computed once at setup (propagate each controlled-Pauli through
the existing propagation table → a `(control_record, record-flip mask, expectation-sign
mask)` descriptor) and applied per shot as a cheap forward pass over records — entered
**only** for circuits that contain feedback. The state-evolution hot loop
(`compose_fired` / `apply_diag_pauli` / the CascadePlan cascade) is **untouched**; the
bare circuit fed to the existing pipeline excludes the controlled-Paulis, so its bare
state / propagation / records are feedback-free and identical. This is verified two ways:

**(1) Byte-identical streams (the no-leak proof).** None of the five committed
speed-critical benchmark configs (`code_switching_faithful --ref`, `cultivation_d3_faithful
--ref`, `miniature_oracle --ref`, `code_switching_rate`, `cultivation_d3_rate`) contain
feedback. `tests/test_stream_pins.py` re-hashes every channel buffer (`meas`/`dets`/`obs`/
`exp`) with SHA-256 and they remain **6/6 byte-identical** to the committed pins post-feedback
— the strictly stronger guarantee than equal speed (same RNG consumption order + same
per-shot layout = the non-feedback code path is bit-for-bit unchanged).

**(2) Measured µs/shot, unchanged.** Marginal cost (20k→200k shots, seed 2026, `cpp/build_rel`
Release binary, `--b8`), best-of-4 on this machine:

| circuit | µs/shot (post-feedback) | pre-feedback reference (MPP note, same machine class) |
|---|---|---|
| `cs_faithful` (`--ref`, χ=2) | **1.00** | 0.94 |
| `d3_faithful` (`--ref`, χ=2) | **0.78** | 0.68 |
| `code_switching_rate` | **0.22** | 0.17 |
| `cultivation_d3_rate` | **0.22** | 0.22 |

All within run-to-run noise of the reference (these are tiny circuits whose marginal-cost
measurement carries ±0.1 µs jitter on this machine). The binding guarantee is the
**byte-identical stream pins** (which fix the exact code path); the µs/shot table only
confirms it. Conclusion: **classically-controlled Pauli feedback did not change the speed of
the magic-state-prep circuits** — the relabel is dead code when no controlled-Paulis are
present.

---

## Category tables

### 1. Circuit IO & construction

| Stim feature | xtim | Class | Rel | Cost | Note |
|---|---|---|---|---|---|
| `Circuit(text)` parse | yes | **Supported** | high | – | line-numbered `XtimParseError`. |
| `from_file` | yes | **Supported** | high | – | `xtim.Circuit.from_file`. |
| `to_file` / round-trip serialize | yes | **Supported** (M-D) | med | small | `to_file(path-or-file)` round-trips with `from_file`; `str(circuit)` / `.text` give the Stim text. |
| `append` / `+` / `*` programmatic build | yes | **Supported** (M-D) | med | med | `append(name, targets, arg)`, `+`/`+=` (concatenate), `*`/`*=` (REPEAT block), `copy()`, `len()` — every mutation re-parses, so an invalid build raises `XtimParseError` and leaves the circuit untouched. `insert`/`pop` deferred (low value for the build-then-sample workflow). |
| `num_qubits/measurements/detectors/observables` | yes | **Supported** | high | – | all present (+ `num_expectations`). |
| `num_ticks` / `num_sweep_bits` | no | **Missing** | low | small | minor introspection. |
| `flattened` / `without_noise` / `inverse` / `decomposed` | no | **Missing** | low | med | analysis transforms; low value for sampling workflow. |
| `to_qasm` / `to_quirk_url` / `to_crumble_url` | no | **Missing** | low | large | interop/visualization exports. |
| `to_tableau` / `flow_generators` / `has_flow` | no | **N/A** | low | – | stabilizer-algebra; out of scope + non-Clifford circuits anyway. |

### 2. Sampling

| Stim feature | xtim | Class | Rel | Cost | Note |
|---|---|---|---|---|---|
| `compile_sampler().sample()` (measurements) | yes | **Supported** | high | – | `bit_packed` supported. |
| `compile_detector_sampler().sample()` (dets+obs) | yes | **Supported** | high | – | `separate_observables`, `bit_packed`; **plus** raw `exps` and `return_measurements` from one run. |
| Per-shot **Pauli expectations** | yes (xtim-only) | **Supported+** | high | – | `PAULI_EXPECTATION` → `exps float64`. Stim has no analog. A genuine xtim advantage. |
| seeded determinism | yes | **Supported** | high | – | `seed=` pins the run; same stream every `sample()`. |
| `sample_write` to file | via CLI | **Partial** | med | small | Python samplers return arrays only; file-writing is in the `python -m xtim` CLI, not on the compiled-sampler objects. |
| `CompiledDemSampler` (`dem.compile_sampler`) | no | **Missing** | med | med | sampling detection events directly from a DEM (Stim's `sample_dem`); useful for decoder testing without re-simulating. |
| `reference_sample()` | internal only | **Partial** | low | – | xtim resolves a reference internally (invisible cache); not exposed as a Stim-style API. |

### 3. Noise channels

| Stim channel | xtim | Class | Rel | Note |
|---|---|---|---|---|
| `X/Y/Z_ERROR`, `DEPOLARIZE1/2`, `PAULI_CHANNEL_1/2` | yes | **Supported** | high | full validation (prob∈[0,1], sums≤1). |
| `CORRELATED_ERROR`(`E`)/`ELSE_CORRELATED_ERROR` | no | **Missing** | med | explicitly rejected → "use PAULI_CHANNEL_*". Correlated multi-qubit error injection has no substitute. |
| `HERALDED_ERASE`, `HERALDED_PAULI_CHANNEL_1` | no | **Missing** | med | erasure noise; rising importance. |
| `I_ERROR`/`II_ERROR` | **yes** | **Supported** | low | RESOLVED: identity-error placeholders = pure no-ops for sampling (any prob args ignored); targets `touch()`ed for qubit-count consistency; `II_ERROR` requires an even target count. Verified vs Stim 1.16.0 in `tests/test_measurement_gaps_stim.py`. |
| `MPAD` (padding records) | **yes** | **Supported** | low | RESOLVED: each `0`/`1` target appends one fixed-bit measurement record via a fresh `\|0>` ancilla + record-invert flag. Verified vs Stim in `tests/test_measurement_gaps_stim.py`. |

### 4. Detectors / observables / annotations

| Stim feature | xtim | Class | Rel | Note |
|---|---|---|---|---|
| `DETECTOR rec[-k]` | yes | **Supported** | high | – |
| **`DETECTOR(coords)` coordinates** | **yes (M-B)** | **Supported** | high | RESOLVED (M-B): coords STORED on `ParsedStim`, surfaced via `Circuit.get_detector_coordinates(only=)`, and emitted into the DEM as `detector(coords) D#`; matches real Stim. |
| `OBSERVABLE_INCLUDE(i) rec[-k]` | yes | **Supported** | high | – |
| `QUBIT_COORDS` / `SHIFT_COORDS` | **yes (M-B)** | **Supported** | high | RESOLVED (M-B): `QUBIT_COORDS` stored (last write wins) and exposed via `Circuit.get_final_qubit_coordinates()`; `SHIFT_COORDS` resolved cumulatively (component-wise, per-REPEAT-iteration) into both qubit and detector coords, matching Stim. |
| `TICK` / `REPEAT` | yes | **Supported** | high | REPEAT unrolled with hostile-input caps. |
| `MPP` measurement-record products | **yes (M-MPP)** | **Supported** | high | RESOLVED: parse-time ancilla cat-check gadget per `*`-product → one record each, in Stim's record order; `i^{#Y}` / `!`-inversion signs byte-match real Stim. No engine change. |
| `SPP` / `SPP_DAG` Pauli-product rotation | **yes (M-MPP)** | **Supported** | high | RESOLVED: parse-time CX-ladder basis change + `S`/`S_DAG` on a pivot, uncomputed; dense unitary = Stim's up to global phase, record distribution matches Stim. |
| sweep bits | no | **Missing** | med | `sweep[k]` controls deferred (see headline). |
| `rec[]` feedback | **yes (Tasks 1-3)** | **Supported** | med | `CX/CY/CZ rec[-k] q` fully supported via triangular record relabel (see §1b in `docs/xtim_dialect.md`). |
| `!` inverted measurement targets | **yes (Task 4+5)** | **Supported** | low | `M !0`, `MX !q`, `MPP !X0*Y1`, `MZZ !0 1` — all RESOLVED (see headline). |
| `M(p)` / `MR(p)` readout-flip | **yes (Task 4+5)** | **Supported** | med | Record-only Bernoulli flip on any measurement gate — RESOLVED (see headline). |
| gate/instruction **tags** `[...]` | no | **Missing** | med | (see headline) — modern Stim emits these. |

### 5. Detector Error Model (DEM)

| Stim feature | xtim | Class | Rel | Note |
|---|---|---|---|---|
| `circuit.detector_error_model()` → `stim.DetectorErrorModel` | yes | **Supported** | high | xtim returns a **real** `stim.DetectorErrorModel` (parses its own DEM text). Excellent for decoders. |
| DEM **with PAULI_EXPECTATION L-columns** | yes (xtim-only) | **Supported+** | high | extra L-columns after observables; `expectation_columns` maps them. |
| DEM ctor options: `decompose_errors` + `ignore_decomposition_failures` | **yes (Task 4)** | **Supported** | **HIGH** | RESOLVED: **`decompose_errors=True`** (with `ignore_decomposition_failures`) emits `^`-separated graphlike (≤2-detector) components — exactly what pymatching needs. **Decode-equivalent** graphlike decomposition (greedy peel against the model's own edges, choosing the all-real-edge split that maximizes observable-probability mass on its components — reconstructs Stim's per-edge weight profile that xtim's canonical-channel merge would otherwise blur). Verified by **PyMatching LER-equivalence vs Stim** on the generated surface/repetition/color library (`tests/test_xtim_standard_circuits.py`, z<5 vs Stim+PyMatching). **Surface + repetition are genuinely decode-equivalent** (shared-matcher per-shot disagreement ~0.03%, symmetric — Monte-Carlo noise, no √N growth). **Color d3 carries a small documented residual**: xtim's canonical-channel merge yields a coarser decomposition (≈81 vs Stim's ≈130 DEM lines) → a sign-consistent ~2.6%-relative-LER bias (Stim's matcher slightly better) that only resolves to z>5 past ~500k shots; decode-equivalent at practical shot counts but NOT a perfect match. Color codes are a non-standard MWPM target anyway — color d5 is not graphlike at all (Stim itself requires `ignore_decomposition_failures` there, which xtim mirrors). The `decompose_errors=False` path stays **byte-identical** to Stim's undecomposed grouping. Out of scope: byte-matching Stim's *exact* component grouping (and closing the color-d3 residual), and the niche flags `block_decomposition_from_introducing_remnant_edges`, `approximate_disjoint_errors`, `flatten_loops`, `allow_gauge_detectors`. |
| `explain_detector_error_model_errors` | no | **Missing** | med | error-provenance debugging. |
| `shortest_graphlike_error` / `search_for_undetectable_logical_errors` | no | **Missing** | med | circuit-distance tools; very commonly used to sanity-check a code+circuit. (Could be done on the exported DEM via stim itself.) |
| `likeliest_error_sat_problem` / `shortest_error_sat_problem` | no | **Missing** | low | SAT distance. |
| `compile_m2d_converter` (measurements→detection events) | no | **Missing** | med | converting recorded measurements into detector events offline (Stim's `m2d`). xtim only samples detectors directly. |

### 6. CLI (`python -m xtim` vs `stim`)

| Stim subcommand | xtim verb | Class | Rel | Note |
|---|---|---|---|---|
| `stim sample` | `xtim sample` | **Supported** | high | `--in --shots --seed --out --out_format 01\|b8\|dets\|r8` (M-C added dets/r8). |
| `stim detect` | `xtim detect` | **Supported+** | high | dets + `--obs_out --exp_out --meas_out` (4 channels, one run); `--out_format 01\|b8\|dets\|r8` (M-C). |
| `stim analyze_errors` | `xtim analyze_errors` | **Partial** | high | exports DEM but **no `--decompose_errors`** and no other DEM options. |
| `stim m2d` | – | **Missing** | med | measurement→detector conversion. |
| `stim sample_dem` | – | **Missing** | med | sample from a DEM. |
| `stim explain_errors` | – | **Missing** | low | – |
| `stim gen` | – | **N/A** | low | canned example circuits; not the MSP workflow. |
| `stim diagram` | – | **Missing** | med | (see §7). |
| `stim convert` | – | **Missing** | med | result-format conversion (01↔b8↔dets…). |
| `stim repl` | – | **Missing** | low | – |
| (xtim-only) `diagnose`, `state compile/verify` | n/a | **Supported+** | high | onboarding/reference surface Stim has no analog for. |

### 7. Output formats

| Stim `--out_format` | xtim | Class | Rel | Note |
|---|---|---|---|---|
| `01` | yes | **Supported** | high | text; exps as `%.17g` floats. |
| `b8` | yes | **Supported** | high | byte-identical to `stim … --out_format b8`. |
| `dets` | **yes (M-C)** | **Supported** | **HIGH** | RESOLVED (M-C): `shot D… L… M…` text on `sample`/`detect`. xtim keeps detectors/observables/measurements in SEPARATE channels (`--out`/`--obs_out`/`--meas_out`), so each is homogeneous — byte-identical to `stim … --out_format dets` with observables split via `--obs_out` (D# on `detect --out`, L# on `--obs_out`, M# on `sample`/`--meas_out`). |
| `r8` | **yes (M-C)** | **Supported** | med | RESOLVED (M-C): run-length bytes (gap-to-each-1 + trailing-run byte, 0xFF continuation) per channel — byte-identical to `stim … --out_format r8`. |
| `hits` | no | **Missing** | med | sparse hit-index format. |
| `ptb64` | no | **Missing** | low | bit-transposed 64-shot blocks. |
| `stim.read_shot_data_file` / `write_shot_data_file` | no | **Missing** | low | helper IO. |

### 8. Diagrams

| Stim `circuit.diagram(...)` | xtim | Class | Rel | Note |
|---|---|---|---|---|
| `timeline-svg` / `timeline-text` | no | **Missing** | **HIGH** | Stim users lean on these constantly to eyeball a circuit. None in xtim. |
| `detector-slice-svg` / `time-slice` | no | **Missing** | med | detector-geometry views. |
| `match-graph-svg` / `match-graph-3d` | no | **Missing** | med | DEM matching-graph views. Could be obtained via stim on the exported DEM. |
| `timeslice+detector-slice` etc. | no | **Missing** | low | – |

---

## Top gaps to consider for v0.3+

Ranked by (impact on the MSP audience) × (how often a real Stim circuit hits it),
with cost. The first cluster is **cheap and removes "won't load" walls**:

1. **Accept `I` (identity).** No-op like `TICK`. One line in the dispatcher.
   Highest surprise-to-cost ratio — `I` is everywhere in Stim circuits. *(small)*
2. **Accept Stim aliases** `ZCX/ZCY/ZCZ`, `H_XZ`, `SQRT_Z`/`SQRT_Z_DAG`, `SWAPCZ`,
   `MZ/RZ/MRZ` (last three already in). A static alias→canonical table. A file
   using `H_XZ` or `ZCX` is 100% valid Stim and currently breaks. *(small)*
3. **Add the in-class Cliffords** `SWAP`, `ISWAP`/`ISWAP_DAG`, `CXSWAP/CZSWAP/SWAPCX`,
   `SQRT_X(_DAG)`, `SQRT_Y(_DAG)`, `CY`, `XCZ/YCZ/…`, `H_XY/H_YZ`, `C_XYZ/C_ZYX`.
   All are Clifford and stay in the diagonal-Clifford-simulable class, so they are
   engine-compatible; the wall is purely the parser allow-list. *(med — per-gate
   engine wiring, but no class change)*
4. ~~**`decompose_errors=True` on DEM export**~~ **DONE (Task 4):** decode-equivalent
   graphlike decomposition (greedy peel against the model's edges, observable-mass
   maximizing all-real-edge split) + `ignore_decomposition_failures`. Verified by
   PyMatching LER-equivalence vs Stim on the generated surface/repetition/color
   library; `decompose_errors=False` stays byte-identical. *(med)*
5. ~~**`dets` output format** (and ideally `r8`/`hits`).~~ **DONE (M-C):** `dets` and
   `r8` added to `--out_format` on `sample`/`detect`, byte-identical to Stim per
   channel (D#/L#/M#). `hits`/`ptb64` remain open (lower priority).
6. ~~**Preserve `DETECTOR`/`QUBIT_COORDS` coordinates** and expose
   `get_detector_coordinates()` / `get_final_qubit_coordinates()`.~~ **DONE (M-B):**
   coords stored on `ParsedStim` (SHIFT_COORDS resolved), exposed via
   `Circuit.get_detector_coordinates(only=)` / `get_final_qubit_coordinates()`, and
   emitted into the DEM as `detector(coords) D#` — all verified against real Stim.
7. ~~**Pauli-product measurements** `MPP`/`MZZ`/`MXX`/`MYY` (+ `SPP`/`SPP_DAG`).~~
   **DONE (M-MPP):** `MPP` and `SPP`/`SPP_DAG` are RESOLVED by **parse-time
   desugaring** — `MPP` → a fresh-ancilla cat-check gadget per `*`-product (`H a`;
   one controlled-Pauli per factor; optional `Z a` for `!`; terminal `MX a` = the
   record), `SPP`/`SPP_DAG` → a CX/CY/CZ basis-change ladder onto a pivot + `S`/`S_DAG`
   then uncompute — so NO engine/sampler/kernel change was needed; the v1 simulable
   class is untouched and non-`MPP` circuits stay byte-identical and same-speed (see
   the speed note below). Verified vs real Stim (Clifford record-distribution equality,
   incl. `i^{#Y}` and `!`-inversion signs) and vs the dense oracle (magic-carrying
   `MPP`). `MXX`/`MYY`/`MZZ` are expressible as `MPP X_a*X_b` etc. The native engine
   `measure(Pauli)` port stays **deferred** as an *optional* optimization (it would
   drop the per-product ancilla and the gadget's record bookkeeping); it is not needed
   for correctness and was kept out to leave the single-qubit hot path untouched. *(done)*
8. **`stim diagram` equivalents** — at minimum `timeline-text`/`timeline-svg`.
   Heavily used; high relevance. Match-graph diagrams can piggyback on the exported
   DEM + stim. *(med–large)*
9. **`CORRELATED_ERROR`/`E` and heralded noise.** Correlated and erasure error
   injection have no substitute today. *(med)*
10. **Gate tags `[...]`** — accept-and-ignore so modern Stim output loads. *(small)*
11. ~~**Programmatic circuit building** (`append`/`+`/`*`/`to_file`).~~ **DONE (M-D):**
    `Circuit.append(name, targets, arg)`, `+`/`+=` (concatenate), `*`/`*=` (REPEAT
    block), `to_file`, `copy`, `len()` — Stim-named/-semantic, text-backed, every
    mutation re-parses (invalid build → `XtimParseError`, circuit untouched).
    `insert`/`pop` remain deferred (low value for build-then-sample). *(small)*

### Honest non-gaps (N/A by xtim's design — do NOT "fix")
- No `Target`/`account`/`k` correction API, no built-in post-selection or
  β-combination: deliberate. The engine emits raw physics; interpretation is
  user-owned numpy (`xtim.collect` shows the pattern). Correct call.
- No `Tableau`/`PauliString`/`FlipSimulator` arithmetic surface: out of scope for
  the sampling/analysis workflow, and the circuits are non-Clifford anyway.
- `rec[]` feedback / sweep-bit *corrections* are conceptually user-side — but note
  the **circuit syntax** still fails to load, which is a real ingestion gap even if
  the semantics are deferred.

---

## Surprising findings

- **`I` is rejected.** The single most common "filler" instruction in Stim circuits
  doesn't parse. Trivial fix, outsized impact.
- **Stim aliases break** even when the canonical gate is accepted: `H_XZ` (≡`H`),
  `ZCX` (≡`CX`), `SQRT_Z` (≡`S`) all fail. A user who copied an alias-using circuit
  out of Stim's own output gets a parse error for a gate xtim *does* support.
- ~~**DETECTOR/QUBIT_COORDS coordinates are silently discarded**, not rejected — so a
  circuit loads fine but coordinate-aware tooling downstream gets nothing, with no
  warning.~~ **RESOLVED (M-B):** coordinates are now stored (SHIFT_COORDS resolved),
  surfaced via the Stim-named methods, and carried into the DEM.
- ~~**No `decompose_errors` knob on DEM export**~~ **RESOLVED (Task 4):** the DEM
  option pymatching most needs is now supported (decode-equivalent graphlike
  decomposition + `ignore_decomposition_failures`), verified by PyMatching
  LER-equivalence vs Stim on the generated library.
- ~~**`dets` output format is missing**~~ **RESOLVED (M-C):** `dets` and `r8` are now
  offered on `sample`/`detect` and byte-match Stim per channel, so xtim CLI output
  drops into the Stim ecosystem (sinter, decoders); `hits`/`ptb64` still absent.
- On the plus side, xtim genuinely **exceeds** Stim in two places that matter for
  this audience: per-shot `PAULI_EXPECTATION` values, and a DEM that carries those
  expectations as extra L-columns — neither has a Stim analog.
