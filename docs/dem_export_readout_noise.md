# DEM export: measurement readout noise (`M(p)`) — defect, fix, oracles

**Status:** FIXED 2026-10-06 on branch `fix/dem-export-measurement-noise` (commit `08fc4ad`);
RELEASED in **3.1.4** (2026-10-07, owner-approved). Every release up to and including 3.1.3 is
affected — see CHANGELOG [3.1.4]. The sections below are the fix report as written before the
cut; "Installed environment" describes the state at the time of the fix.

## The defect

`detector_error_model()` / `detector_error_model_text()` / `detector_error_model_with_reject()`
/ `_xtim.export_dem_text` DROPPED every measurement readout-flip probability written on the
measurement itself: `M(p) q`, `MX(p)`, `MY(p)`, `MR(p)`, `MRX(p)`, `MRY(p)`, `MPP(p)`,
`MXX(p)`/`MYY(p)`/`MZZ(p)`. Stim's `detector_error_model()` emits one
`error(p) <every detector / observable that reads the record>` per such measurement; xtim's
export emitted nothing. The sampler was never affected — it always applied the flip
(`sampler.cpp` record-flip pass; `twirl_sampler.cpp` `rec_flip`) — so the *samples* carried
readout errors that the *decoder's model* did not know about: a decoder fed xtim's DEM never
corrected a readout flip.

Found in adaptq (2026-10-06): adaptq's run-time noise policy (`adaptq/noise.py`) spells readout
noise exactly in this form (`MX(p_meas)` etc.), so for every policy-noised, decoded protocol
the decoder had no readout mechanisms. A Steane 1-round memory at SD6 p = 1e-3 decoded at 6.6 %
(= the raw readout-flip rate) instead of ~1.2 %; rewriting authored `X_ERROR(p) q; M q` as
`M(p) q` reproduced it (4e-5 -> 1.19 %).

## Root cause (one line)

`export_dem` (`cpp/src/dem_export.cpp`) enumerated error mechanisms ONLY from
`Instr::Kind::Noise` instructions of the deferred stream and never read
`Instr::readout_flip_p` — which deferral drops from the rebuilt terminal Measures, so even a
walk over the deferred Measures would have seen 0; the sampler reads the field off the
pre-deferral coherent circuit (`sampler.cpp`, "record-flip metadata"), the exporter did not.

## The fix

After the noise-channel walk and before emission, `export_dem` walks the Measures of
`nr.coherent` (post-coherentize, pre-deferral — the same source and the same order contract
the sampler relies on: coherentize copies Measures verbatim and in order, deferral never
reorders Measures among themselves, so coherent's Measure order == terminal-read order) and,
for each record `j` with `readout_flip_p > 0`, folds ONE independent mechanism of probability
`p` whose signature is every target reading record `j` by parity:

* detectors / observables — `tmask` rows (a dropped gauge observable's row is all-zero);
* `PAULI_EXPECTATION` columns — the declared byproduct frame (`obs_frame`): the sampler applies
  `rec_flip` BEFORE the `emask` fold, so a flipped frame record flips the *reported* sign of
  that column (only when `include_expectations`, i.e. `R > 0`);
* `DECISION` logical-flip columns — `dmask` rows.

The mechanism goes through the same `fold` (`acc[sig] = a + p - 2ap`) as every channel
mechanism, into the same `std::map` keyed by sorted signature, so emission order is unchanged
and a readout-free circuit's text is byte-identical. A readout flip is a classical record
flip: nothing propagates, no logical magnitude can change (never a `postselect_faults` entry),
and no `OUTPUT_QUBITS` frame column is touched. An unread record folds into the empty
signature and is dropped, like any undetectable error.

Where it sits in FP order: readout mechanisms fold LAST (after all channels). Two-way
composition `a + p - 2ap` is commutative in IEEE arithmetic, so the only possible ulp-level
difference from Stim's reverse-walk order is on signatures hit by >= 3 mechanisms — the same
pre-existing property xtim's forward fold already has for gate noise (measured below:
identical bit-exact counts on the authored twins).

Files: `cpp/src/dem_export.cpp` (+62 lines, the readout pass + `<cassert>`),
`tests/test_dem_measurement_noise.py` (new, 32 tests), `docs/xtim_dialect.md` §2b-iv (DEM
semantics paragraph), `CHANGELOG.md` [Unreleased].

## Oracles and results

**RED first.** `tests/test_dem_measurement_noise.py` on the 3.1.3 build: 26 failed / 6 passed
(the passes were `M(0)`, the readout-free pin, and 4 Stim-generated circuits whose readout
noise Stim authors as `X_ERROR(p); M` — those were then rewritten to `M(p)` via `readoutify`;
final test-of-test on the backed-up 3.1.3 `.so`: 30 failed / 2 passed, the 2 being the
build-independent `M(0)` and readout-free pin). On the fixed build: 32 passed.

**Stim oracle (Clifford circuits), canonicalised** — both DEMs parsed by `stim`, flattened,
each mechanism keyed by its sorted target set, probabilities compared to 1e-12 relative,
detector / observable id sets compared: 28 circuits (every M-family spelling incl. `MRX`/`MRY`/
`MPP`/`MXX`/`MZZ`, `!`-inverted, p = 0.5, p = 0 (no mechanism), det+obs on one record, unread
record dropped, multi-target lines, distinct p per record, same-signature composition with
`X_ERROR`, the `X_ERROR;M` vs `M(p)` authored-rewrite equivalence, mid-circuit `MR(p)` with
qubit reuse, MPP rounds, a Steane 1-round memory at p_meas = 0.0066, Stim's generated
repetition / rotated-surface memories (d = 3 and d = 5, 2–3 rounds, with depolarizing +
reset noise) readoutified): mechanism / detector / observable SETS equal in 28/28; 1080
mechanisms, 1080 within 1e-12 (worst 3.3e-13), 423 bit-exact. The non-bit-exact remainder is
pre-existing: on the authored (`X_ERROR;M`, byte-identical to 3.1.3) twins the bit-exact
counts are the same number for number (e.g. surface d = 5: 326/861 authored, 326/861
readoutified); every single-record case is bit-exact.

**Magic-bearing circuit** (`cultivation_d3_rate`, T gates + post-selection; no Stim oracle):
with its authored `X_ERROR(p) q; M q` readout noise readoutified to `M(p)`, the DEM equals (i)
an independent Python oracle — the readout-free xtim DEM composed with the record-flip
mechanisms derived from the DETECTOR / OBSERVABLE_INCLUDE / PAULI_EXPECTATION-frame
declarations by `p ⊕ q` — set-equal and within 1e-12, and (ii) its authored twin's DEM;
`postselect_faults` identical to the readout-free export. (`cultivation_d5`, which ships with
`M(0.001)` lines, refuses its DEM for an unrelated reason: until 3.1.9 "19 twirled fair-coin
reads > 16 on one alternative"; since 3.1.10, whose exact correlated-read law replaces the
fair-coin expansion, because a channel has no exact conversion into independent mechanisms —
`approximate_disjoint_errors=True` exports it.)

**Byte-identity corpus (readout-free exports):** 18 circuits — the 9 bundled examples, 6
Stim-generated memories (repetition, rotated/unrotated surface, color), 4 hand circuits
(Pauli channels, DEPOLARIZE2 + MPP, coords, a magic frame) — including the refusal texts of
the 5 that refuse (`ch_cultivation` reject, `code_switching_*` gauge observable,
`miniature_oracle` non-product channel, one gauge-observable hand case): SHA256 of the export
IDENTICAL before/after, 18/18. The 8 readoutified twins changed (expected), the 4 that refuse
for independent reasons did not. (Corpus script and both hash lists are in the session
scratchpad; the corpus is reproducible from `tests/test_dem_measurement_noise.py` helpers.)

**Suite:** `pytest tests/` 287 passed, 5 skipped (22 s). `ctest` (cpp/build: seed_parity,
coherentize_all, frame_cosets) 3/3 passed.

**Speed:** the sampler hot path is untouched (no sampler source changed). The exporter has no
speed gate; measured `export_dem_text` warm median-of-7, 3.1.3 `.so` vs fixed:
cultivation_d3_rate 10.1 -> 9.9 ms (readoutified 10.0 -> 9.9), surface d = 5 r = 3
53.3 -> 53.8 ms (readoutified 52.6 -> 53.2) — unchanged within noise. The readout pass is
O(#flipped records × (T + R + NDEC)) bit tests.

## Installed environment

The env `adaptq-eng` holds an EDITABLE install of `/home/user/codes/xtim` (`pip show -f xtim`:
`__editable__.xtim-3.1.3.pth`), and the extension it imports is the in-tree
`xtim/_xtim.cpython-312-x86_64-linux-gnu.so`. That `.so` was REBUILT in place with the fix
(`python setup.py build_ext --inplace`, the same route that produced the 3.1.3 build on
2026-09-22), so the env now runs the fixed exporter whenever the checkout is on this branch
(the `.py` files are branch-tracked; the `.so` is gitignored and does NOT switch with
`git checkout`). To revert to the 3.1.3 extension: copy back
`build/_xtim.cpython-312-x86_64-linux-gnu.so.3.1.3-backup` (sha256
`0e864c7a386c3ea0729ea99515392c70b843ec4e2536484dfb7597d31eb026ec`) over
`xtim/_xtim.cpython-312-x86_64-linux-gnu.so`, or `git checkout main && python setup.py
build_ext --inplace`. The `cpp/build` cmake tree was also rebuilt for ctest (gitignored).
Nothing was published, tagged or pushed.

## adaptq blast radius (read-only survey; nothing re-run)

Writer of `M(p)`-style noise: `adaptq/noise.py:263` (`gate_noise`, meas branch:
`f"{head}({_fmt(model.p_meas)})"`), documented at `noise.py:19-20`;
`NoiseModel.uniform(p)` sets `p_meas = 5p` (`noise.py:91`). Authored gate bodies with a
measurement probability argument: none (`gates_builtin.py`, `gates_generic.py`,
`gates_distillation.py`, `codes_builtin.py`, `composites.py`, `_fragments/`: 0 hits;
`scripts/gates/byte_gate.py:137` states no pinned FF leg has an `M(p)` at all — `ny_z`
authors X/Y_ERROR, `nl_*` run at p = 0).

Export path: `adaptq/engine.py:233` `export_dem_text` -> `adaptq/runner_ff.py:1189`
`_stage_dem_text`, `:1210` `_stage_dem_text_ported`, consumed at `runner_ff.py:2224`
(window model) and `:2663` (`_adapter(model.dem_text, c.decoder)`). So the affected set is:
**FF-tier runs with a decoded stage or window (decoder not None / "postselect") on a
protocol whose stage text carries `M(p)`** — in practice every `with_policy` / `run(...,
noise=NoiseModel(p_meas > 0))` protocol that decodes. Numbers that may move once the fix
lands (file:line):

* `tests/oracle/test_windows_over_calls.py:463` (`se_memory_window(rounds=2, window="bp_osd")`
  under `NoiseModel.uniform(1e-3)`, 2000 shots, FF) and `:730` (`se_rounds/memory2(window=
  "bp_osd")` top under the policy); `:744` is a refusal test (unaffected).
* `tests/oracle/test_acceptance_sentinels.py:598` and `:673` (`with_policy(..., uniform
  (1e-3))` on a protocol with `steane_readout_z_decoded_noisy` + `decoder="frontier"`,
  `:645`).
* `tests/test_usability_fixes.py:61` (`run(with_policy(memory_chain(window=None, noisy=
  False), uniform(1e-3)), 300, 42, tier="ff")` — moves iff `memory_chain`'s readout stage
  decodes); `:82` (`canonical_t_protocol("Z")`, undecoded: unaffected).
* The finding itself: the Steane 1-round memory at SD6 p = 1e-3 in the qec_library
  `decoder-feedback` worktree (6.6 % -> expected ~1.2 %).

Policy-noised but UNDECODED (no stage DEM built) — expected unmoved: pinned legs
`po_z__exact` (`scripts/gates/byte_gate.py:141`), `po_z__ff` (`:142`), `po_y__ff` (`:162`)
(`scripts/gates/baselines/bytes.json:129-138` notes only `ny_z_decoded__ff` depends on a
stage DEM, and that leg (`byte_gate.py:181`) authors no `M(p)`); the Choi-harness
whole-policy canonical (`docs/memos/choi-harness-report.md:60,116,556`); the measuring-gates
`po_z__ff` / `po_y__ff` rows (`docs/memos/measuring-gates-report.md:206-223`);
`tests/test_runner_ff_harness_selection.py:72`; `tests/oracle/test_defect_b.py:75,135,155,
170,542,563,612` (instrument / decision columns, no decoder); `tests/oracle/test_measuring_
acceptance.py:135-139,497`; `tests/test_engine_frame_rows.py:74-77,885-888`;
`tests/oracle/test_rz_word_windows.py:261` (`NoiseModel(0,0,0,0,1e-3)`: p_meas = 0).
The 1M Shor / surface and 200k Steane recheck legs
(`docs/memos/readout-baseline-recheck/recheck.py:100-125`) use authored `*_noisy` gates and
`readout_decoder`, no `NoiseModel` — unaffected.
