# J-b1 re-pin manifest — ENGINE_VERSION `xtim-engine-12` → `xtim-engine-13`

Status: **PRE-SIGNED, conditional** (OD-2026-10-19-jb1-switchover-presigned); effective after J-b1's gates and the
adversarial review. Branch `stage/jb1`, F commit (the commit adding this file; its sha is recorded in jb1/STATUS.md) (the production flip), docs merge (the stage/jb1-docs merge on stage/jb1).
Nothing is pushed or merged into dev/next. The machine-readable twin is `docs/repin/J-b1-manifest.json`.

## 1. What moved, and why

F switches production — the law, the record and the API together — onto the **family (FAM) record law**:
every production constructor (`compile_twirl_sampler`, `compile_detector_sampler`, the Partitioned inner samplers,
ports, `from_ideal_state`) builds the family sampler with ALL record classes; the legacy sampler (`run_w<*, false>`,
σ, the PPR family, the exact per-shot routes) moves to the test API (`xtim._xtim_testapi.compile_legacy_sampler`) and
is deleted at J-b2. `RECORD_KEY_LAYOUT` 2 → 3, TWPL v5 → v6, `engine="auto"` removed (default `"twirl"`).

Provenance line on every re-pinned literal:

> RE-PINNED 2026-10-20 (J-b1, ENGINE_VERSION xtim-engine-13): the family record law, layout 3 and the J-b1 API;
> <mechanism ids>

Mechanism ids (plan §3.2; every moved byte names at least one; predicates are computed from OLD-side data and the
circuit text, never from the diff — `classify_jb1.py`):

| id | mechanism |
|---|---|
| M-VER | ENGINE_VERSION `xtim-engine-13` (every stamp, `ref_info()`) |
| M-MEMCTR | counters renamed / re-keyed (`record_report()` replaces `channel_report()`) |
| M-GAUGE | gauge detector columns are exact Born records (were independent fair coins; the GAUGE stream is not drawn) |
| M-ANTI | ANTI detectors / refused observables are served (were refused by name) |
| M-NOCH | circuits with no deterministic detector channel are served (were refused) |
| M-PPR | PPR-class shots take the kind-2 term route (was the PPR family + exact fallback) |
| M-EXACT | the legacy exact per-shot routes are retired; those shots are law-served (shot-selected replay below) |
| M-DECPPR | Born decisions / retention on CH/PPR circuits are served (were refused) |
| M-PORTREF | INPUT_QUBITS / ideal-state DETECTOR / OBSERVABLE columns are reference-relative (were absolute) |
| M-SPLICE | the prep-splice / carried composition reroute (XQ9 (A)) |
| M-NC | fault-dependent anticommuting terms (static fixed-axis rule) are refused by name (NONCOMMUTING_MAGIC) |
| M-ENGINE | `engine="auto"` removed; the default is `"twirl"` |
| M-RFL3 | layout-3 record fidelity projects onto EVERY record operator (Born observables and decisions included) |
| M-EXPREF | `return_expectations=True` on the twirl engine raises `XtimTwirlRefusal(EXPECTATIONS_NOT_SERVED)` |
| M-TYPED | refused-on-both-sides surfaces: the refusal TEXT changed (typed `XtimTwirlRefusal`) |
| M-API | removed / renamed fields and methods (plan §4.1; CHANGELOG) |
| M-KEY | layout-3 record keys (no σ ‖ coins ‖ plan bytes) |
| M-MATROUTE | (r2) the value readers (`materialize`, `state`, `branch_frames`, `pauli_expectations_all`) take the gate-level route on branch-0 rows: equal to C0̂ as a STATE (1e-12); `pauli_expectations_all` values may move within 1e-12 |

## 2. The preserved laws and the gate proving each

| # | law (unchanged) | gate |
|---|---|---|
| 1 | the `sample()` joint law of detectors, observables and decisions (where both engines serve) | law battery OLD vs NEW, 264 circuits (`<S>/jb1/law/`): tw / br vs exact max \|z\| 3.99 / 4.81, 0 violations of the exact relation basis; J-a dual-run gates (`tests/test_ja_dual_run.py`) |
| 2 | the barrier records' joint law | the same battery, barrier legs; `tests/test_jb1_*` record-law gates |
| 3 | the retained state of every shot as a function of its record | `tests/test_s2b_state_equals_bits.py` (legacy) and `tests/test_ja_retention.py` G-8a (family): ⟨M_c⟩ on materialize(i) = (−1)^{y_c} to 1e-12; `tests/test_jb1_c0hat.py` (C0̂ == tableau route as a state, 1e-12); `tests/test_jb1_materialize_speed.py` (cultivation_d5 shots 0-9: F's public materialize == the legacy law's, 1e-12) |
| 4 | every exact-engine, DEM and forced-fault law | F-B3 must-not-move (below): byte-identical |

Shot-selected exact-route replay (the shots the legacy sampler sent to an exact route, replayed on the family law):
3 M-EXACT circuits (ja_seed23: 185 legacy exact shots; E2/t0_SD6: 11; E7/e7_SD6: 24), **0 mismatches**
(`<S>/jb1/F/exact_replay.json`).

## 3. Byte movement on the corpus (OLD = b9c2e92, NEW = F, FAMP = a13133e test-API FAM, LEG = F's legacy)

Captured with ONE tool revision (`<S>/jb1/tools/bytes_jb1f.py`; r2 additions: the handle-aware `n` read, the
`JB1_FAMP_NO_PE` skip, the `JB1_ONLY` filter) and classified by `classify_jb1.py` (`<S>/jb1/F/classify_r2.json`):
171 circuits in each of OLD / NEW / FAMP / LEG (core corpus 12 shards + big corpus 6 shards; NEW recaptured in full
on the r2 tree).

| check | result |
|---|---|
| **BUG** | **0** |
| F-B1a NEW == FAMP | 0 differences; 25 `c_pe` (pauli_expectations_all) value moves, max 2.4e-14 (M-MATROUTE); 2 tool-only rows (FAMP builds its Target through the pre-F constructor, which refuses first: both sides refused) |
| F-B1a where FAMP cannot capture `c_pe` (a13133e's C0̂ value route: the defect itself; core_2 / core_6 rerun with `JB1_FAMP_NO_PE`) | NEW `c_pe` == OLD == LEG **exactly** (max 0.0) on bh6, blowup cult_d5, born/m3_obs_2dec, code_switching_rate, examples/cultivation_d5, E1/e1_ro_AA, E5/e5_AA; the three PPR circuits E2/t0_AA, E2/t1_AA, E7/e7_AA (shots differ from OLD by M-PPR) recaptured on FAMP with `c_pe` (`JB1_ONLY`): == NEW within 1e-12 |
| F-B1b LEG == OLD | 0 differences |
| F-B3 must-not-move | 0 differences |
| OLD vs NEW twirl surfaces | 307 moved (circuit, surface) pairs, every one explained: ref_info 163, det_default_exp 68, rf_ref 21, det_twirl 18, twirl_sample_c 14, det_default 13, rf_target 6, c_dets 2, c_obs 1, state_sample 1. Mechanism hits: M-VER 163, M-EXPREF 68, M-PPR 53, M-DECPPR 53, M-EXACT 23, M-ANTI 19, M-RFL3 19, M-PORTREF 11, M-NOCH 8, M-ENGINE 7, M-TYPED 3, M-SPLICE 1 |
| reverse check (predicate true, no surface moved) | 42: M-RFL3 36 (Born-trie circuits whose fidelity legs did not move at the captured shots), M-ENGINE 6 (auto had routed to twirl already) |

## 4. Re-pinned literals and rewritten / retired tests

### 4.1 Literals whose VALUE moved
| file:line | provenance (mechanism) |
|---|---|
| tests/test_barrier_refusal.py:1 | E1 (fix/barrier-refuse), RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness. |
| tests/test_diagnose_state_route.py:46 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness: a circuit with no detector / decision |
| tests/test_feedback_after_t.py:243 | theorem) it cannot move; the first still can. RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> |
| tests/test_feedback_after_t.py:264 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3): the X/Y-type feedback from a |
| tests/test_feedback_after_t.py:621 | the legacy engine refused naming the representation limit. RE-PINNED 2026-10-20 (J-b1), refusal -> exactness. |
| tests/test_feedback_after_t.py:634 | only feedback #0). RE-PINNED 2026-10-20 (J-b1), refusal -> exactness: served, dense-exact per record class. |
| tests/test_feedback_after_t.py:641 | RE-PINNED 2026-10-20 (J-b1), refusal -> exactness. |
| tests/test_feedback_after_t.py:678 | name on the legacy engine, naming the shared wire). RE-PINNED 2026-10-20 (J-b1), refusal -> exactness. |
| tests/test_ja_frames.py:226 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3 test_ja_frames.py:228-238; |
| tests/test_keyed_streams.py:16 | the GAUGE stream is not drawn), RE-PINNED 2026-10-20 (J-b1, ENGINE_VERSION xtim-engine-13): M-GAUGE; |
| tests/test_keyed_streams.py:57 | assert _x.ENGINE_VERSION == "xtim-engine-13" # RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-VER |
| tests/test_plan_payload_budget.py:170 | # RE-PINNED 2026-10-20 (J-b1 F): resident plans are reused by the readers (P12 F3), so the unbounded sampler's |
| tests/test_plan_payload_budget.py:220 | # RE-PINNED 2026-10-20 (J-b1 F, layout 3): the readers resolve a kind-0 content to the sampler's RESIDENT plan |
| tests/test_port_basic.py:131 | PORT-03, RE-PINNED 2026-10-20 (J-b1, xtim-engine-13; M-DECPPR), refusal -> exactness: ch_cultivation with |
| tests/test_port_basic.py:402 | PORT-16, RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness: the raw cube_ccz.stim has ZERO |
| tests/test_port_groups.py:240 | PG-08, RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3 test_port_groups.py:319-338; |
| tests/test_record_fidelity.py:535 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3; M-FRAME): an input_pauli frame |
| tests/test_record_fidelity.py:570 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3; M-DECPPR): a Born-class DECISION |
| tests/test_record_report.py:15 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-MEMCTR / M-API — the family sampler has no exact route (no |
| tests/test_refusal_design_inputs.py:44 | The repro is in scope: records sample, and the shots really take the PPR (kind-2) route. RE-PINNED 2026-10-20 |
| tests/test_refusal_design_inputs.py:76 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-ANTI — the ANTI detector is served (it was refused by name) |
| tests/test_twirl_audit_sampling.py:241 | # RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-EXACT / M-PPR — the KF2 shots the legacy sampler sent to |
| tests/test_twirl_decision_detector_sampler.py:309 | RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-DECPPR — a Born-class DECISION on a CH/PPR circuit is SERVED |
| tests/test_twirl_factory_kappa.py:57 | Every shot is a plan-path shot: kappa>=2 is not a fallback reason any more. RE-PINNED 2026-10-20 (J-b1, |

Each row's file carries the provenance line next to the moved value; the JSON twin lists them.

### 4.2 Plan §5.2 rows confirmed at F
| file | literal | outcome at F |
|---|---|---|
| tests/test_keyed_streams.py | `ENGINE_VERSION == "xtim-engine-12"` | → `"xtim-engine-13"` (M-VER) |
| tests/test_keyed_streams.py | gauge detectors == GAUGE bits | rewritten: gauge columns are Born rows, the GAUGE stream is not drawn (M-GAUGE) |
| tests/test_frame_cosets_cpp.py `_STREAM_HASHES` | frames() + record_keys() hashes | UNCHANGED values; the gate re-pointed to the legacy sampler (its own coverage until J-b2); production's frames are gated by tests/test_ja_frames.py / tests/test_jb1_record_api.py |
| tests/test_born_fastpath.py `_REF_DEC_H` | born5 decisions | UNCHANGED (diagonal; decisions are rows since S2b) |
| tests/test_exact_fallback_retained_state.py, tests/test_exact_fallback_forced.py | forced-lever fingerprints | RETIRED with the lever and the exact class (files deleted) |
| tests/test_frames_born_dec_refusal.py pins | per-field frames() hashes | UNCHANGED values on the legacy sampler (re-pointed) |
| tests/test_barrier_refusal.py | refused-observable rf / dressing hashes | rewritten: refusal → exactness (M-ANTI) |
| tests/test_record_report.py | `(fast, clean, slow_kappa0)` counts | UNCHANGED as counts (diagonal); `_EXACT_TOTAL` / `exact_shots` removed (M-MEMCTR / M-API) |
| tests/test_refusal_design_inputs.py | `ppr_plans == 1 and exact_shots == 0` | → kind-2 plan count (M-PPR / M-API); the two strict xfails flip to passes |
| tests/test_twirl_factory_kappa.py, test_input_frame_mirror.py, test_noise_frame_commutation.py, test_wide_channel_rows.py | `exact_shots == 0` | → "no exact key" (M-API) |
| tests/test_small_items.py | `ppr_plans == 0` | → `record_report()["term_route"]["plans_kind2"]` (rename only) |
| tests/test_ja_term_key.py G-TK4 | (16, 4, 4, 3) | UNCHANGED (FAM pin); NONCOMM fixtures flip to the M-NC refusal |
| tests/test_record_layout.py | `version == 2` | production: `layout == 3`; the v2 legacy gate re-pointed |
| tests/test_twpl_cache_guards.py | `_TWPL_VERSION = 5` | v5 loader guards re-pointed to the legacy sampler; production writes v6 (tests/test_jb1_k2_arena.py pins its skew / token guards) |
| README.md factory numbers (test_readme_snippets) | `accepted 86.0%; distilled 1-F = 2.7e-05`, `4.5e-05` | confirmed by the docs merge (tests/test_readme_snippets.py green at (the stage/jb1-docs merge on stage/jb1)) |
| benchmarks/ledger/baseline/workload_manifest.json `keys.*` | sha256 of plan keys | UNCHANGED: kind-0 content == the old key's content bytes (the driver converts; 1,746 distinct fac plans equal) |
| tests/data/frames_contract_golden.json | exact-law golden | values UNCHANGED; the consumer drops `exact` (M-API) |
| tests/test_state_budget.py r6.6 (materialize charged peak = ONE state) | — | **UNCHANGED** at F (r2): the pre-r2 re-pin to 6.875 states (OD-2026-10-20-materialize-peak) is WITHDRAWN, see §8 |

### 4.3 Tests re-pointed to the legacy sampler (literals unchanged; the legacy engine's own coverage until J-b2)
test_barrier_sigmas.py, test_born_trie.py, test_branch_frames.py, test_frame_cosets_cpp.py, test_frames_born_dec_refusal.py, test_frames_contract.py, test_injection_prebake.py, test_input_frame_mirror.py, test_jb1_fidelity_vs_production.py, test_jb1_k2_arena.py, test_jb1_materialize_speed.py, test_keyed_streams.py, test_overlap_binding.py, test_plan_structure_classifier.py, test_record_fidelity.py, test_record_layout.py, test_s2b_mechanisms.py, test_s2b_r7.py, test_s2b_readers.py, test_s2b_record_law.py, test_s2b_record_table.py, test_s2b_removed_names.py, test_s2b_route_identity.py, test_s2b_state_equals_bits.py, test_s5a_fast_retention.py, test_twirl_diagnostics.py, test_twirl_feedback_semantics.py, test_twirl_kappa_dual.py.

### 4.4 Tests retired (deleted)
- tests/test_exact_fallback_forced.py, tests/test_exact_fallback_retained_state.py: the forced-exact lever and the
  exact class are retired (plan §5.3); their law content is covered by the family retention gates and the
  shot-selected exact-route replay.
- tests/test_port_sigma_law.py: σ retired (D1); `FramedSuperposition.port_sigma_law` is test-API only.
- tests/test_frame_port_chain.py: `sample_barrier(frame_dressing=)` / `Result.frame_dressing` removed (plan §4.1, P12):
  the frames class folds a supplied frame into the record's P_rec (`frame_folded(i)`); the two-stage frame chain is
  covered by tests/test_ja_frames.py and tests/test_jb1_branch_*.

### 4.5 Refusal → exactness rewrites
Plan §5.3 list, each test now asserting the served value (dense / exact oracle) where it asserted a refusal:
test_barrier_refusal.py (whole file), test_refusal_design_inputs.py (two strict xfails pass), test_feedback_after_t.py,
test_port_basic.py (PORT-03, PORT-16), test_port_groups.py (PG-08), test_ja_frames.py, test_twirl_retained_records.py,
test_s2b_route_identity.py, test_frames_contract.py, test_input_frame_mirror.py, test_s2b_record_table.py,
test_record_fidelity.py (M-FRAME, M-DECPPR), test_twirl_decision_detector_sampler.py (M-DECPPR),
test_diagnose_state_route.py; route counters: test_record_report.py, test_twirl_diagnostics.py,
test_twirl_audit_sampling.py (KF2: the shots are law-served), test_twirl_decision_detector_sampler.py.

## 5. Behaviour changes (checked against the pre-signature's scope)

### 5.1 Refusals ADDED (declared; by name, `xtim.XtimTwirlRefusal`)
| class / sub | what | prevalence (measured) | served when |
|---|---|---|---|
| NONCOMMUTING_MAGIC (M-NC) | fault-dependent anticommuting terms (static fixed-axis rule): the 4 ch_pair fixtures (`ch_pair`, `t_h_ch`, `t_h_ch_decision`, `NONCOMM_PORT`) plus the CS witness; the T;X_ERROR;T;H;T precedence (TS:119 refuses it first, typed NONCOMMUTING_MAGIC) | 0 real workloads | owner's static rule (J-c0 confirms with nc_oracle) |
| LAW_LIMITATION_PROPAGATION | 29 circuits (27 fuzz, 2 xtim test texts) | 0 real workloads | — |
| LAW_LIMITATION_CARRIED_AXIS / branch_frame | a producer branch whose materialize frame differs from the ideal post-barrier frame: CE1-chain dressed branch, toy-factory teleport-out producers; branch_handles on a consumer's branch row (3-stage re-export) | **0** on adaptq (366 producer stages, 2,197 branches), ftqcp (0 stages), cult_d3/d5, both CCZ factories, f3 | J-c ((b′) frame alignment) |
| LAW_LIMITATION_CARRIED_AXIS / carried_free_x | consumer axis combination with its X-part on the ideal's free rows only: xstage F2 / F4, H·T·H on a carried \|T> | **0** on adaptq (36 INPUT_QUBITS texts), ftqcp, core corpus | J-c ((B) register Clifford) |
| LAW_LIMITATION_CARRIED_AXIS / carried_branch_x | same on a branch's E_min rows: the plan's per-E fixture (consumer T 0 / MX 0, **T-depth 1**); xstage F1 (NOT T-depth 1: the fault-dependent class, as ch_pair) | E_min ≠ ∅ not seen among enumerated first-order faults | J-c (narrowed (B)) |
| carried-branch rows (`input_branch != 0`), fixplan-F-branch-readers | `RecordFidelity.evaluate`, `frames`, `signs` (F-1), `record_keys` / `record_group_ids` / `record_groups` / port RecordGroups (F-2, option (b)): refused by name; `sample_records` refuses E_min ≠ ∅ rows. Branch-0 rows unchanged; materialize serves branch rows | — | J-c |
| EXPECTATIONS_NOT_SERVED / RAW_RECORDS_NOT_SERVED | `sample(return_expectations=True / return_measurements=True)` on the twirl engine (end-of-circuit `<P>` under auto included; J-b2/S7 candidate) | — | J-b2 |
| CORRECTION_NOT_SERVED (final review C1) | `RecordFidelity(reference=True).evaluate(..., correction=)` | a reader argument, not a circuit | — (pass the target as atoms) |
| OUTSIDE_T_DEPTH_1 / "bare", "composed" on ideal-state consumers (final review C3) | a consumer whose own bare, fresh block or composed build is outside the class (were untyped ValueError) | adaptq 0 of 36 INPUT_QUBITS texts | — |
| LAW_LIMITATION_CARRIED_AXIS / branch_row (final review m2) | the carried-branch-row reader refusals below, now typed | — | J-c |
| **C0̂ capacity (r2 item 5; ruling on fixplan-F-c0hat-capacity: (B))** `xtim.XtimCapacityError` | the frame-preserving C0̂ amplitude map (branch export: `ideal_fs` / `add_branch` / `branch_handles`; carried-branch-row reads) refuses, BEFORE any work, a plan whose produced state cannot fit: support bound c·2^d entries (d = GF(2) rank of the term images' X-parts on F₀ ∪ E_S, c ≤ χ the cosets of their span met by the bare's keys) at the result's 24 B per entry above the state budget (`XTIM_MAX_STATE_BYTES`, default 2 GiB); the transient maps stay uncharged, as before r2. Never a hang | cultivation_d5 seed 7 shots 5, 6 (bound 2·2^30) refused in ~4 ms (test-API C0̂ route); branch-export census on the F tree: adaptq 2,197 / 2,197 distinct branches accepted; factory producers served as before r2: census at the F tip (time-capped as the P9b census) factory_out 4 / 4, private factory a_out 4 / 4, private factory b_port 16 / 16 accepted, 0 refused (factory a distinct branch #4, bound 512·2^16 = 805 MB at 24 B: served in 275 s); under the pre-ruling (A) charge, 14 / 64, 14 / 64 and 6 / 37 were refused | J-c (C0̂ replacement for the branch export: the factory exports are served but slow, 259 s / +2.2 GB transient for one factory-a branch — a J-c priority) |

### 5.2 Refusals LIFTED (served)
Gauge detectors, refused observables, ANTI detectors / observables, no-channel circuits, Born decisions and retention
on CH/PPR circuits, poisoned input frames, opaque-PPR dressing, RecordFidelity on Born decisions and input_pauli
buffers, Y frames through CH; the ladder-gap circuits (T;T;H;T, 2 of the 4 test_forced_branches fixtures, 209 of
the 236 fuzz); `check_no_magic_on_carried` lifted (XQ9 (A): the M-SPLICE census refuses 0 of 114, 102 via the
composed route, = exact at z < 5, composed bare = exact spliced state at 1e-12).
F-B5 on the corpus: OLD accepts / NEW refuses: **0** circuits; OLD refuses / NEW serves: 9 surfaces (bis_noCZrec sampler + det_twirl, adaptq_born5_dec_enum_producer, adaptq_steane_h_producer, code_switching_faithful, cube_ccz sampler + det_twirl, E10/e10_AA, E10/e10_SD6), each with its mechanism (M-ANTI / M-NOCH / M-DECPPR).

### 5.3 Convention and routing
- INPUT_QUBITS / ideal-state DETECTOR / OBSERVABLE columns are reference-relative (M-PORTREF); exposure: 0 port
  samplers with a nonzero reference in adaptq gates / battery and ftqcp (P1 census).
- `engine="auto"` removed (ValueError naming `engine='exact'`); the default is `"twirl"`; no routing, no silent
  fallback (owner OD-B). `xtim.ler` observable mode stays on twirl (R3-2).
- The exact engine is an oracle only: the session-level exact-construction counter stays 0 over the full suite
  (tests/conftest.py; MUT-AUTO-FALLBACK RED).

### 5.4 The value readers (r2, fixplan-F-materialize-c0hat)
- `BarrierBuffer.materialize(i)` / `.state(i)` return the extension's `FramedSuperposition` handle, as b9c2e92;
  `.branch_frames(i)` is `materialize(i).branch_frames()`. The planned M-API change of the frame arrays is
  WITHDRAWN (r2 item 1).
- Value readers (materialize, state, expectations, `pauli_expectations_all`) use the gate-level (tableau) route on
  branch-0 rows; C0̂ stays only where the frame is the contract (the branch export) and on carried-branch rows.
- Kind-0 reads never arm the term view (the reader's former one-time first-call cost: 3.7 ms on cultivation_d5,
  62 ms on the factory — TermResidual's merge + ker Φ); kind-2 content arms it once.
- Per-sampler invariants are computed once: the state-model key (lazy-once; a process-unique sampler serial skips the
  bit-exact key comparison for the sampler's own records), and `arm_` is idempotent.

## 6. Must not move (F-B3) and the identity checks
- F-B3 (OLD vs NEW, byte-identical): dem, dem_approx, det_exact, det_exact_exp, raw, bare_frames, chi, ref_text,
  port, py_state_api; ref_info moves by M-VER only. **0** differences on the 171-circuit corpus.
- F-B1a (NEW == FAMP on every shared surface, the pre-flip test-API family sampler): **0** differences (25 value moves within 2.4e-14 on `c_pe`, M-MATROUTE; see §3).
- F-B1b (LEG == OLD: the legacy sampler moved verbatim): **0** differences.

## 7. API surface and downstream
- The full list is plan §4 (r3.4 additions) and the CHANGELOG entry of the docs merge. Removed names raise
  (AttributeError / TypeError / KeyError; `tests/test_jb1_removed_names.py`).
- adaptq breaks loudly (design §4.1 table); its pins are not re-pinned (decision 10). ftqcp stamps move with
  ENGINE_VERSION. The owner's study scripts (σ, 3.1.13 asserts) break loudly; migration recipe:
  tests/private/ccz_factory_a_out_target.json + tests/test_jb1_fidelity_vs_production.py (correction = flips · duals).

## 8. Speed

### 8.1 Timing (the ONE locked run at the final tip 94c21c1 vs b9c2e92; `ledger.py timing`, 3 rounds, all ACCEPTED, host load ~10 from a foreign job, foreign share on the pinned core 0.0004; `<S>/jb1/ledger/timing-final-F.json`)
Gate: <= +5% (HARD for cold/*, cold_plain/*, api/materialize_*, api/pe_all_*; t_fid 5% vs first green; frames <= b9c2e92).
| component | unit | base b9c2e92 | F tip 94c21c1 | tip/base (95% CI) | verdict |
|---|---|---|---|---|---|
| cold/color5_det | us_per_shot | 8.86 | 7.54 | 0.854 (0.839-0.869) | PASS |
| cold_plain/color5_det | us_per_shot | 9.12 | 7.79 | 0.859 (0.852-0.868) | PASS |
| cold/rp2_det | us_per_shot | 8.63 | 9.01 | 1.045 (1.041-1.049) | PASS |
| cold_plain/rp2_det | us_per_shot | 9.19 | 9.50 | 1.031 (1.023-1.040) | PASS |
| cold/color5_fid | us_per_shot | 25.30 | 9.89 | 0.389 (0.383-0.395) | PASS |
| cold_plain/color5_fid | us_per_shot | 25.64 | 10.19 | 0.395 (0.390-0.400) | PASS |
| cold/rp2_fid | us_per_shot | 31.03 | 12.57 | 0.403 (0.400-0.407) | PASS |
| cold_plain/rp2_fid | us_per_shot | 31.55 | 13.01 | 0.405 (0.394-0.417) | PASS |
| cold/fac | us_per_shot | 48.45 | 42.97 | 0.891 (0.882-0.900) | PASS |
| cold_plain/fac | us_per_shot | 49.53 | 43.80 | 0.892 (0.873-0.912) | PASS |
| cold/ccz_f3 | us_per_shot | 89.80 | 73.74 | 0.823 (0.819-0.826) | PASS |
| cold_plain/ccz_f3 | us_per_shot | 92.61 | 74.96 | 0.811 (0.807-0.815) | PASS |
| t_fid/color5 | us_per_shot | 25.36 | 9.55 | 0.376 (0.370-0.383) | PASS |
| t_fid/rp2 | us_per_shot | 31.00 | 11.95 | 0.387 (0.385-0.388) | PASS |
| t_fid/d5b | us_per_shot | 34.61 | 14.50 | 0.418 (0.414-0.421) | PASS |
| t_fid/fac | us_per_shot | 67.97 | 17.15 | 0.251 (0.249-0.254) | PASS |
| api/frames_d3/cold | us_per_row | 0.24 | 0.07 | 0.304 (0.304-0.305) | PASS |
| api/frames_d3/warm | us_per_row | 0.06 | 0.00 | 0.003 (0.003-0.004) | PASS |
| api/frames_d5/cold | us_per_row | 17.31 | 6.21 | 0.359 (0.359-0.360) | PASS |
| api/frames_d5/warm | us_per_row | 0.87 | 0.00 | 0.000 (0.000-0.000) | PASS |
| api/frames_d15/cold | us_per_row | 0.02 | 0.01 | 0.515 (0.508-0.523) | PASS |
| api/frames_d15/warm | us_per_row | 0.02 | 0.00 | 0.005 (0.005-0.005) | PASS |
| api/frames_fac/cold | us_per_row | 38.28 | 4.27 | 0.112 (0.111-0.114) | PASS |
| api/frames_fac/warm | us_per_row | 0.91 | 0.00 | 0.000 (0.000-0.000) | PASS |
| api/materialize_d5 | us_per_call | 161.86 | 162.00 | 1.002 (1.000-1.003) | PASS |
| api/pe_all_d5 | us_per_call | 163.24 | 163.01 | 1.000 (0.996-1.004) | PASS |
| api/materialize_fac | us_per_call | 1271.66 | 1279.22 | 1.020 (0.985-1.056) | PASS |
| api/pe_all_fac | us_per_call | 956.83 | 946.99 | 0.989 (0.988-0.990) | PASS |
| ns_draw/uniform | ns_per_draw | 1.08 | 1.08 | 1.000 (1.000-1.000) | PASS |
| ns_draw/word | ns_per_draw | 0.41 | 0.41 | 1.000 (1.000-1.000) | PASS |
| ns_draw/noise | ns_per_draw | 1.20 | 1.20 | 1.000 (1.000-1.000) | PASS |
| ns_draw/position | ns_per_draw | 0.20 | 0.20 | 1.000 (1.000-1.000) | PASS |

Re-time after the final-review fixes (7dee8b1, locked, 5 rounds ACCEPTED): api/materialize_fac 0.996 (CI
0.993-0.999), api/pe_all_fac 0.993 (CI 0.991-0.995).

Every component passes at the point estimate. Thin margins: cold/rp2_det 1.045 (CI upper 1.049; the numpy-heap
placement effect of the rp2_det ruling — plain process 1.031) and api/materialize_fac 1.020 (CI upper 1.056; a
noisy factory leg, its pe_all twin 0.989). The value-reader route is at parity (r2).

### 8.2 Ledger (Ir per shot, valgrind, engine of ad5d6f0 — 94c21c1 changes only the constructor, apply_input_frame
and the exact sampler's pointer forms, none on these components' collected paths; `<S>/jb1/ledger/measure_tipF.log`, t_fid/fac in `measure_tipF_fac.log`). Column `pre-flip FAM`: the J-a fam_fast twin for t_fast (sample mode on every batch, so the barrier rows d3b / d5b / bt81b are NOT like for like), the p12s5 / p10 FAM measurement otherwise.
    component                                  base b9c2e92          F tip   F/base  pre-flip FAM    F/fam malloc/shot
    cold/ccz_f3                                    978918.6       632903.0   0.6465       632747.1   1.0002   161.78
    cold/color5_det                                145183.6       112075.1   0.7720       112062.3   1.0001    77.45
    cold/color5_fid                                530373.8       127622.7   0.2406       127607.8   1.0001    76.78
    cold/fac                                       314111.3       214924.4   0.6842       214931.9   1.0000    58.68
    cold/rp2_det                                   115661.9       116921.4   1.0109       116945.1   0.9998    97.48
    cold/rp2_fid                                   583347.0       139577.4   0.2393       139594.9   0.9999   100.56
    t_fast/bt81b                                    12602.4         2102.7   0.1669          681.5   3.0856     0.00
    t_fast/d3b                                       1966.1         1225.1   0.6231          474.5   2.5820     0.00
    t_fast/d5b                                      30608.3        12012.6   0.3925         4416.2   2.7201     0.00
    t_fast/d5s                                       4435.7         4443.5   1.0018         4450.3   0.9985     0.00
    t_fast/e7                                       18873.4         2896.9   0.1535         2896.9   1.0000     0.00
    t_fast/fac_b                                       8948.0         8984.7   1.0041         8984.7   1.0000     0.00
    t_fast/fac                                       2209.3         2225.7   1.0074         2225.7   1.0000     0.00
    t_fast/t0                                        9516.0         2730.1   0.2869         2730.1   1.0000     0.00
    t_fast/t1                                       13919.2         2801.7   0.2013         2801.7   1.0000     0.00
    t_fid/color5                                          -        90056.0        -        90030.9   1.0003    52.08
    t_fid/rp2                                             -       105460.5        -       105460.9   1.0000    74.00
    t_fid/fac                                     1946522.0       401034.5   0.2060       385312.0   1.0408   103.47
    t_pprbuild/e7                               152885195.0    166814361.0   1.0911    152309278.0   1.0952 276011.00
    t_pprbuild/t0                                97451724.0    144980609.0   1.4877     97628035.0   1.4850 247691.00
    t_pprbuild/t1                               123295387.0    164241412.0   1.3321    122592136.0   1.3397 274014.00
    t_retention/d3b/sample_same_batch                 464.2          474.5   1.0222          464.2   1.0222     0.00
    t_retention/d5b/sample_same_batch                4402.2         4409.5   1.0017         4402.2   1.0017     0.00
    t_row/d3b                                          32.7           80.0   2.4472           32.7   2.4472     0.00
    t_row/d5s                                         167.8          167.8   1.0000          167.8   1.0000     0.00
    t_row/fac_b                                         322.4          322.4   1.0000          322.4   1.0000     0.00
    t_row/fac                                         169.6          169.6   1.0000          169.6   1.0000     0.00

- G1 (<= 5% cumulative vs PRE-S0): t_fast/d5s +2.36%, t_retention/d5b +2.35%, t_retention/d3b +2.02%; every other
  sampling component equal or lower. G2z: 0 allocations per shot on every t_fast / t_retention component.
- t_fid/fac Ir 401,035 per shot (b9c2e92 1,946,522; pre-flip FAM p12s5 385,312: +4.1% vs first green, within 5%).
  t_fid/d5b Ir does not finish under callgrind (as at P12; its wall timing stands, 0.418x).
- **t_row/d3b 2.45x (32.7 -> 80.0 Ir per shot): a call-count mechanism, not a cost change.** record_rows_<false,true>
  costs 80.0 Ir per call on both trees; the layout-3 recording run calls it on EVERY barrier shot (24,000 calls per
  3x8,000-shot pass) where b9c2e92's barrier run called it on 9,807 (the rest took the σ / coin store). The whole
  barrier run (t_fast/d3b) is -38%. ACCEPTED (coordinator ruling at 05cdebf) as a measurement-definition shift:
  t_row's reference is re-taken at F as the FIRST pin of the redefined component (record_rows_ on every barrier
  shot).
- t_pprbuild/{e7,t0,t1} (plan build, report-only): +9% / +49% / +33% vs b9c2e92 (the J-a §4.1 forecast +25..50%):
  needs its accepted_overages.json entry with the owner-approved id (plan §6.5) — PENDING WITH THE OWNER (proposed:
  accept for J-b1 as an S8 target; sampling on those circuits is 61-81% faster).
- cold/* Ir: every workload <= b9c2e92 except rp2_det +1.1% (Ir; wall +4.5%, see 8.1); equal to the pre-flip FAM
  (p10) within 0.02%.

## 8b. Refusal census at the F tip (public `compile_twirl_sampler` + `sample(20)`, 12,293 texts)
adaptq 811 / 811 and ftqcp 438 / 438 served. Against the P1 census (pre-F FAM(all)): 12,062 served on both, 219
refused on both (OUTSIDE_T_DEPTH_1 bare 201, chi capacity 6, NONCOMMUTING_MAGIC rule refusals, 2
LAW_LIMITATION_PROPAGATION xtim test texts), 6 newly served (rule-ACCEPT ladder-gap circuits: 3 formerly
NONCOMMUTING-by-alt, 3 formerly bare-rejected), and 6 harness-only rows (IF circuits: the census called
`sample()` without `decision_bits`, which `PartitionedTwirlSampler` requires — not a refusal). OLD accepts and F
refuses: exactly the 4 declared M-NC fixtures (ch_pair class) in the xtim test texts. (`<S>/jb1/r2/refcensus/`)

## 8c. Post-F commits (each byte-identical to F on the core corpus)
| commit | what | gates |
|---|---|---|
| ad5d6f0 | merge of stage/jb1-docs (Q1 docs; F's code, the docs' docstrings; TODO(F) / OPEN-1 filled) | Python AST without docstrings identical to F; check_docs_postF.sh PASS; pytest 5559 passed |
| 6c22a48 | test: the public entry points refuse by name with no exact construction | MUT-AUTO-FALLBACK was GREEN on the suite (the refusal tests used the pre-flip twin); RED now (5 failed + the session counter fixture) |
| 0826a2f, 94c21c1 | fixplan-F-ubsan-empty-row (+ addendum r2, cleared): operator[] / memcpy on empty vectors when a circuit has no channel words (reachable since F serves zero-channel circuits): the CW-strided row arrays allocated with >= 1 element, `data() + offset` at the indexing sites (twirl_sampler.cpp, twirl_sampler_fam.cpp compose_frame_, framed_sampler.cpp next_shot), the measurement-row memcpy guarded | UBSan (recover mode) 0 reports on 20 files / 464 tests; core bytes vs F DIFFS = 0 (73 circuits, 2,774 arrays); linked run_w<*, true> unchanged (codegen_so); pytest 5564 passed; ctest 28 / 28 |

## 8d. Final-review fixes (final-review.md MERGE-READY-WITH-FIXES; fixplan-final-review, cleared 2026-10-21)
| commit | item | change | gates |
|---|---|---|---|
| 186a895 | C1 / MAJOR-1 | `RecordFidelity(reference=True).evaluate(..., correction=)` silently ignored the correction: now refused before any work, `XtimTwirlRefusal(CORRECTION_NOT_SERVED)` (new TwirlClass member, ruling) | test (reference refused; a target takes the correction: \|+> under Z -> 0); MUT-RF-REF-CORR RED |
| da1f1f4 | C2 / MAJOR-2 | refusal texts no longer advertise `engine='exact'` where it also rejects: LAW_LIMITATION_PROPAGATION and NONCOMMUTING_MAGIC-after-the-propagation-table say "outside xtim's fast law and also rejected by the exact engine (the same gate-by-gate propagation table)"; the static-rule NONCOMMUTING_MAGIC path makes no fixed claim (the exact engine SERVES the ch_pair / t_h_ch / CS-witness fixtures but rejects e.g. the review's p18 circuit, and both reach that path — the ruling's "also rejected" wording would be false there); CHANGELOG / docs rows corrected | agreement test (29 class-14 texts + NC fixtures: every claim == the exact engine's verdict); MUT-EXACT-CLAUSE RED |
| 153ce9b | C3 / MAJOR-3 | an ideal-state consumer's fresh magic block takes the P5 ladder-gap settle (build-then-settle, as the standalone family build): T-depth-1 ladder-gap consumers are served; consumer rejections typed `OUTSIDE_T_DEPTH_1` ("bare" / "composed") | the review's p14 consumers (producer and input_state_of) == the independent dense oracle on the joint circuit (5 sigma, 0 impossible); T-depth-2 consumer typed; MUT-INJ-NOSETTLE RED |
| 766be10 | C4 / MAJOR-4 | the exact-construction counter covers the test API (own engine copy: bound counter + scope; `xtim._exact_scope` sums / enters both) | full suite 0 unaccounted; carried-refusal tests on the public API and the test API (carried_free_x, carried_branch_x, branch_frame, T-depth-2 consumer): typed, no exact construction; MUT-INJ-FALLBACK RED (8 + session) |
| 7dee8b1 | C5 / m1 m2 m3 m6 | production reads only its own BarrierBuffer (foreign module_local records refused by type; serials salted per extension); branch-row reader refusals typed (LAW_LIMITATION_CARRIED_AXIS / "branch_row"); CHANGELOG env-lever claim corrected (QEC_TW_PROF, QEC_TW_NULLSPACE_XCHECK, QEC_PARTNER_HASH_MIN, XTIM_BARE_MEMO* still read; S7 C5b); Born-trie oracles moved to the test API (ledger t_node driver follows); stale docstrings | MUT-RECORDS-FOREIGN RED; pytest 5566 passed; ctest 28 / 28; core bytes vs 05cdebf DIFFS = 0 (73 circuits) |

Declared behaviour changes from the fixes: refusal TEXT on LAW_LIMITATION_PROPAGATION / NONCOMMUTING_MAGIC circuits
(M-TYPED; no core-corpus circuit moved); refusal TYPE on the branch-row readers and the consumer refusals (were
ValueError, now XtimTwirlRefusal, still ValueError subclasses); ideal-state ladder-gap consumers newly SERVED; a
`reference=True` + `correction=` call newly REFUSED (it returned a wrong value). J-b2 notes (jb2/plan.md): the tableau
materialize route is production (E13 KEEP); the five legacy-re-pointed files need production equivalents first.

## 9. Owner decisions and signed items
| id | item | state |
|---|---|---|
| OD-2026-10-19-jb1-switchover-presigned | the J-b1 switchover, conditional on its gates and the adversarial review | pre-signed |
| OD-2026-10-20-materialize-peak | (pre-r2) re-pin test_state_budget's materialize peak to 6.875 states | **WITHDRAWN at r2, not needed**: the r2 value route (bare copy, the Clifford on its frame, the records forced in place) restores the r6.6 one-state charged peak; the original pin passes unchanged. The decision row in benchmarks/ledger/README.md is marked withdrawn |
| rulings 2026-10-20 (rp2_det cold harness option (2); fixplan-F-branch-readers CLEARED; fixplan-F-materialize-c0hat r2) | as recorded in jb1/rulings.md | applied |

Signature: ________________ (owner), date ________
