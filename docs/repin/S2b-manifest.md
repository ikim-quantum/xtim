# S2b re-pin manifest — ENGINE_VERSION `xtim-engine-11` → `xtim-engine-12`

Status: **SIGNED** by the owner 2026-10-18 ("Approved"), OD-2026-10-18-s2b-signoff (this manifest plus the ENGINE_VERSION bump).
Nothing is merged. Branch `stage/s2b`.

## What moved, and why

S2b emits every diagonal record — the IN_GROUP channels, the Born observable and the Born
LOGICAL / ANTI decisions — as a row of the plan's **record table** in the record-ordered
canonical basis (design §5.1 S2b, Theorem 1 / Lemma 4). The **byte streams** move:

* COIN word assignments. Canonical coin k is COIN bit k, and it is the value of pivot record k.
  Previously coin k was σ-law coin k.
* BORN word assignments. BORN word j drives canonical direction j. A canonical direction is a
  literal ordered product of the σ / extra directions. The record trie walks only the directions
  a record reads.
* The Born observable and Born decisions on diagonal plans no longer draw the OBS / DEC streams.
  On the exact fallback they draw EXACT words ng and ng + 1 + j. DEC is retired; OBS stays for
  PPR plans (Q2).
* The barrier's σ is drawn by σ-4 (owner 2026-10-16), conditioned on the channel records and the Born
  observable, never on the decisions (wording SIGNED by the owner 2026-10-17): COIN bits from ρ_rec on,
  and the σ-aux trie on BORN words from κ_rec on.
* The retained outcome record is now (u, z) (`BarrierBuffer.outcomes(i)`). For an exact shot it is
  the raw y. The record keys and groups follow.
* `ref_info()["engine_version"]` and every stamp move with ENGINE_VERSION.

Provenance line on every re-pinned literal:

> RE-PINNED 2026-10-17 (S2b, ENGINE_VERSION xtim-engine-12): canonical record-order basis + extras
> as rows; the sample() and barrier (records, σ) laws are unchanged (σ-4, owner 2026-10-16); byte
> streams moved.

## The four preserved laws and the gate proving each

| # | law (unchanged) | gate |
|---|---|---|
| 1 | the `sample()` joint law of detectors, observables and decisions | L-1 `tests/test_s2b_record_law.py` (exact (X) / dense law, 5σ cells, 0 impossible, on fixtures + 3 families + the σ fixtures, sample and barrier); L-2 `tests/test_s2b_mechanisms.py` (every deterministic relation, 0 violations); the S0(b) OLD-vs-NEW law battery on the corpus (below) |
| 2 | the `sample_barrier()` joint law of (detectors, observables, decisions, σ) | `tests/s2b/test_sigma_joint_law.py`, P7 vs B1, every cell / marginal / pair at 5σ, 0 impossible cells, 25 fixtures (40k dev captures green; the 200k stage run is listed below) |
| 3 | the retained state of every shot as a function of its record | S-1 `tests/test_s2b_state_equals_bits.py`. For every record-route shot and every record column, materialize(i)⟨M_c⟩ = (−1)^{y_c} to 1e-12, and the emitted bits are y XOR the constant offsets. S-4 (factory, 240k shots): RecordFidelity reads the record's y and equals materialize to 2.2e-16 on 3254 shots. |
| 4 | every exact-engine, DEM and forced-fault law | the must-not-move list below: byte-identical |

## Law-preservation evidence (run 2026-10-17)

**Multiplicity rule (pre-declared before any hit; departures review §4).**
- 5σ is the threshold.
- A SINGLE exceedance with 5 < |z| < 6 on one statistic triggers ONE rerun of that circuit or fixture at 2N with fresh seeds.
- A gate FAILS if the hit recurs, if two circuits exceed, or if any |z| ≥ 6.
- Every gate reports its test count next to its max |z|. No hit occurred in the runs below.

| gate | tests | max \|z\| | other |
|---|---|---|---|
| S0(b) law battery, B1 vs P7, 264 circuits (`s2b/tools/law_dump_s2b.py`; its OLD-side exact-shot plan id 0xFFFFFFFF is now parsed from the exact record, MINOR-5) | 6,667,332 statistics (strongly dependent) | tw_vs_exact 4.159, br_vs_exact 4.81, tw_vs_old 2.888, br_vs_old 3.575, plan_freq 1.617 | 0 fails; 0 deterministic-relation violations |
| σ joint law, B4 tip vs P7, 200k shots per build, 26 fixtures (incl. anti_aux, a4_comb) | 2,548 (cells + marginals + pairs) | 3.70 | 0 impossible cells |
| m2: sample() ≡ sample_barrier() records, and refusal status + message of sample() vs sample_barrier().obs() (E1) | 52 compiled core circuits | — | 0 unequal at P7, B1 and the B4 tip |
| S-4, CCZ factory f3, 240k shots | 3254 shots, F vs materialize | — | infidelity 1.0542e-3 ± 1.05e-4 vs 9.7996e-4 (0.5σ); acceptance 74,819 vs 75,035 (1σ); \|ΔF\| ≤ 2.2e-16 |

## Re-pinned literals (old → new)

| file:line | literal | old | new |
|---|---|---|---|
| tests/test_frame_cosets_cpp.py `_STREAM_HASHES` | born5 | `8f321b1e848f355f…` | `71d015373ea81c85…` |
| tests/test_frame_cosets_cpp.py `_STREAM_HASHES` | qrm_magic | `262063258028cfae…` | `d76a6cc0fdf6dcb8…` |
| tests/test_frame_cosets_cpp.py `_STREAM_HASHES` | steane_h | unchanged `f316c8f0…` (no coin, no Born content: asserted identical) | |
| tests/test_born_fastpath.py `_REF_DEC_H` | born5 decisions | `157a00f65218d6d4…` | `4ff09ada7a50d7b3…` |
| tests/test_exact_fallback_retained_state.py `_G` | non-exact fingerprint | `{exact 502, streams 95d39fce…, plan_keys 71432d73…, frames b29733ec…}` | `{exact 697, streams cc00cfb2…, plan_keys e3b0c442…, frames e7f86c64…}` (the module now runs under QEC_TW_FORCE_RECORD_FALLBACK=1) |
| tests/test_exact_fallback_retained_state.py | REPRO exact-shot count | 2021 | 2750 (forced lever: every diagonal retained shot) |
| tests/test_keyed_streams.py `test_known_answers` | ENGINE_VERSION | `xtim-engine-11` | `xtim-engine-12` |

Tests that are rewritten rather than re-pinned (semantics changed):

* test_record_layout.py (version 2);
* test_coin_layout.py (deleted with `coin_layout()`);
* test_keyed_streams.py (the Born observable reads BORN word 0; the coins are pivot coins);
* test_born_trie.py (κ_rec; plans by content; the C-3 unread-direction test);
* test_record_report.py and test_twirl_diagnostics.py (route attribution; the deleted routes);
* test_record_fidelity.py and test_exact_fallback_retained_state.py (the exact class runs under the
  lever via tests/test_exact_fallback_forced.py);
* test_state_budget.py (the forced-projection materialize peaks at one state, not two);
* test_plan_structure_classifier.py (record widths ρ + κ_rec).

## Byte movement on the 165-circuit corpus (P7 vs B1, `s2b/tools/bytes_s2b.py`)

Per-surface counts of (circuit, store) entries that differ: moved / compared. `docs/repin/S2b-manifest.json`
holds every moved entry, old → new.

| surface | moved / compared | reason |
|---|---|---|
| bare_frames, chi, n, sigbits | 0 / all | must not move |
| dem, dem_approx | 0 / 69 | must not move |
| det_exact, det_exact_exp, raw | 0 / 65 | must not move (the exact engine) |
| ref_text | 0 / 158 | must not move |
| ref_info | 158 / 158 | only `engine_version` (verified field by field) |
| port, py_state_api, barrier refusals | 0 | unchanged |
| det_twirl / twirl_sample / det_auto | 69 / 165, 164 / 320, 12 / 65 | streams with coins / Born content / extras |
| barrier dets / obs / dec | 130 / 38 / 8 of 302 | the same |
| sig | 144 / 302 | σ byte streams (σ-4; the law is gated) |
| coins (outcomes) | 48 / 302 | the record is now (u, z) |
| keys / rkeys / gid / rgid / fo | 150 / 150 / 16 / 16 / 16 | record keys follow the outcomes |
| pk, F_*, exact, kappa_pos_plans | 8 / 302 | the Born-observable κ>0 shots are record-route shots now (exact → 0) |
| plan_structs | 302 / 302 | export keys: coin_* removed, rec_* added |
| rf_ref / rf_target | 3 / 165 | fidelities of moved streams |

**No coin-free, Born-free stream moved.** Every moved non-PPR circuit has a plan with a coin, a Born
direction or an extra (checked from the OLD barrier plan data). The 8 PPR circuits that moved in
`sample()` only (E10, E2 t0/t1, E7 AA/SD6) have a Born observable whose plans now carry κ_rec ≥ 1.
For example, e10_AA: OLD `slow_kappa0` 871 shots, NEW `record_kappa_pos` 871 shots.

## Must not move (asserted identical OLD vs NEW)

* DEM / forced-fault surfaces: dem, dem_approx (corpus); tests/test_dem_factored_law_golden.py and
  tests/test_forced_golden.py green unchanged.
* The exact engine: det_exact, det_exact_exp, raw.
* `bare_state_of` and the branch_frames of bare states: bare_frames.
* The TWPL key set: payloads rebuild on the version bump; the keys are unchanged
  (tests/test_plan_payload_budget.py green unchanged).
* test_state_budget: no Amplitudes or layout change. One assertion changed: the CHARGED materialize
  peak (the state-budget accounting the test pins, not measured RSS) went from 2 states to 1. This is
  the forced-projection mechanism (plan §1.4); no byte pin moved. Measured at both boundaries
  (`s2b/r6/budget_boundary.py`): P7 refuses at L+2m−1 and succeeds at L+2m; the S2b tip refuses at
  L+m−1 and succeeds at L+m (m = 4096 × 24 B). The test asserts both boundaries. The moved pin should
  have been a STOP under plan §4 step 3 and was not (recorded in the stage report).
  SIGNED by the owner 2026-10-17 ("All as recommended"), on the basis that it is the charged peak, not RSS.

## Open at S2b

- **Q7.** The Born-decision refusal of `xtim.RecordFidelity` STAYS at S2b (coordinator 2026-10-17). It can be lifted only
  when S-4, extended to the FX-born Born-decision fixtures, shows record_fidelity == materialize to 1e-12.
- **allow_born_obs / allow_born_dec.** These are kept. `frames()` now exports every record value (y) and the record
  operators, so the contract state can be composed from the export alone (G-frames green). Retiring the two flags is a loud
  API removal and is left to the owner's iff rule.

## Downstream

* adaptq byte pins are not re-pinned (D10/D11).
* adaptq readers of `coins` / `coin_layout` / `record_layout` v1 / `group_coins` / `born_obs_raw`
  break loudly by design (Q8: report-only inventory).
* ftqcp `tools.xtim` stamps change with ENGINE_VERSION.

## r6 / r7 / r8 evidence (final tip 92f7c46)
- **Byte identity chain** (165 circuits, `s2b/tools/bytes_s2b2.py`):
  - B1 ≡ B3 ≡ 14fbd66 ≡ fc65d4e ≡ f4836a7 ≡ c8d1ff5 ≡ the final tip on every array.
  - Differences appear only on declared surfaces: record_report keys removed at B2/B3 and r6.5, `plan_memory` reader /
    header / payload byte counters, `frames()["y"]` and `record_outcomes` added at B3, and the `ref_info` reference-cache
    flag (environment).
  - The summary's `plan_structs` hash hashed object-array pointers (a tool artifact, now fixed). A field-level probe shows
    plan_structure identical.
- **Law battery** (264 circuits, fc65d4e): 0 relation violations.
  - One single hit: bt27 br_vs_exact |z| 5.46. Under the declared multiplicity rule it was rerun once at 2N with fresh
    seeds: max |z| 3.12, so it passes.
- **σ joint law, 200k** (selfcheck ≥ shots, fc65d4e): 2734 tests, 0 FAIL, max |z| 3.10. The σ-fixture bytes are identical
  at every r7 / r8 commit.
- **S-4 rerun at the final tip** (after r7.1 / r8): infidelity 1.0542e-3 ± 1.05e-4, acceptance 74,819 vs the spike's
  75,035, max |ΔF| 2.2e-16 on 3254 shots.
- **Sampling speed, B5** (`--ir-only` vs parent_s2k_a55c1c0_lb2_sysfix, cumulative vs PRE-S0): 0 FAIL, G2z clean. Timing:
  see the stage report.

## API surface changes (S2b)
- **Removed:** `record_report()["sigma_aux"]["channel_violations"]` (r6.5: a named throw in the selfcheck window
  instead).
- **Added:**
  - `record_report()["plan_memory"]`: reader_lookups, reader_tables, reader_bytes, reader_peak_bytes,
    reader_overshoot_bytes, reader_evictions;
  - `plan_structure()["rec_folds"]`;
  - `BarrierBuffer.record_tables_held()`.
- Earlier S2b changes are listed above.

## Signed items

| item | signed | basis |
|---|---|---|
| test_state_budget.py `2*m-1` → `m-1` (r6.6), plus the two-sided boundary assert | owner, 2026-10-17 ("All as recommended") | the charged peak (state-budget accounting), not RSS; measured 2m → m at both boundaries |
| σ-4 wording: σ is conditioned on the channel records and the Born observable, never on the decisions | owner, 2026-10-17 ("All as recommended") | the law the σ gate checks (r2.1 / r5) |
| Buffer-held reader tables sit outside the plan payload budget (the cache stays bounded; a buffer is user data) | coordinator ruling, 2026-10-17 (against r6.4) | r7.1 / r8; run()'s payload builds pinned at the no-reader control |
| t_build/{d5,fac_b,fac}/k0 overage (+33/+62/+74% vs PRE-S0), optimisation deferred | owner, 2026-10-17 ("Yes put it for later stage"), OD-2026-10-17-s2b-build-k0 | levels pinned in `benchmarks/ledger/baseline/accepted_overages.json` |
| ENGINE_VERSION `xtim-engine-12` and this manifest | owner, 2026-10-18 ("Approved"), OD-2026-10-18-s2b-signoff | the S2b stage report (r9 timing in band) |
| G5 t_pprbuild snprintf 1 → 0 | pinned waiver (`baseline/waivers_s2b_g5.txt`) | P6 b522a0c: the born_node seconds are formatted in Python |

## Signature

Owner decision id: OD-2026-10-18-s2b-signoff  ENGINE_VERSION `xtim-engine-12` and this manifest — signed by the owner 2026-10-18 ("Approved")
