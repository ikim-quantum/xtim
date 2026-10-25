# Changelog

All notable changes to `xtim` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.2] - 2026-10-25

- **Opt-in `postselect=` on `sample` / `sample_barrier`:** rejected shots are dropped early, as soon as a flagged
  detector deterministically fires; the accepted shots are identical to sampling then filtering. No change without the mask.

### Added: in-sampler post-selection with early reject (opt-in; no change without a mask)

- **Added: `postselect=` on `TwirlSampler.sample` and `TwirlSampler.sample_barrier`** (keyword; a `dtype=bool` mask
  of length `num_detectors`, or a sequence of distinct detector indices). Any non-bool array is read as detector
  indices, and duplicate or out-of-range indices are refused by name (`ValueError`), so an integer 0/1 mask (e.g. a
  uint8 `np.unpackbits` row) is refused rather than misread; pass masks as bool. An empty argument of any form (`[]`,
  a length-0 array) flags no detector: every shot passes and the counts are still returned. A call returns only the shots whose flagged detectors are
  all 0, in shot order. `sample` appends the counts `{attempts, passed, discarded, early}` to its tuple (a DECISION
  circuit's wrapper returns `(batch, counts)`); `sample_barrier` returns a buffer of the accepted shots with the counts
  in `BarrierBuffer.postselection` (None without a mask). The accepted shots are byte-identical to an unmasked call at
  the same seed filtered afterwards (detectors, observables, decisions, layout-3 records, `RecordFidelity` values):
  every shot keeps its cumulative shot index, so the counter-based streams are unchanged. Internal record-content
  handles are numbered over the contents of the accepted shots only.
- **Early reject.** Per shot, before any plan lookup, plan build or Born draw, the shot's Clifford part C (the
  diagonal routes' canonical plan key; the term route's reduced term set) and Pauli part P are read: a flagged
  deterministic detector whose operator shares no qubit with C has the fixed value <P, M_d> plus its readout flips,
  and a shot where one fires is dropped at once (`early` counts them). Every emitted shot then passes the exact filter
  on its detector row, so gauge / ANTI detectors and C-touched detectors are filtered exactly as well. On heavily
  post-selected circuits most rejected shots never reach the plan cache.
- No mask, or an all-false mask: the sampler runs exactly as before (sampled bytes unchanged; `ENGINE_VERSION`
  unchanged). With a mask, `record_report()` counters: early-rejected shots are not counted in `used` or the
  plan / trie counters, while `used` does count emitted shots that the exact filter then drops (so it is neither
  `passed` nor `attempts`); `term_route.shots_framed` is counted before the early test and includes early-rejected
  framed shots.

## [3.2.1] - 2026-10-25

A speed release on top of 3.2.0; sampled bytes and per-shot F are unchanged. In short:

- **Faster construction.** Compiling a twirl sampler for a factory-scale circuit is several times faster and needs far
  less memory (no dense PPR tableaux for the sampler table, an early-stop class check, in-class table reuse).
- **Faster `RecordFidelity`.** Orthogonal rows are decided first and score F = 0 without building any value structure.
  The declared behaviour changes (counters, refusals on orthogonal rows, memory per content) are listed below.

### RecordFidelity decides orthogonality first (fid-orth; no byte-stream or `ENGINE_VERSION` change)

- **Changed: `RecordFidelity` (route 3b) decides orthogonality first.** Per content the constraint system is
  row-reduced once; a row whose right-hand side lies outside its column space is orthogonal and scores F = 0 exactly,
  with no value structure built and no memo entry. The value structure (including the record split and its dressed
  record images, for kind-0 and factor-list contents alike) is built on the content's first consistent row. Per-shot
  F is byte-identical to 3.2.0; evaluators of buffers with many orthogonal rows run faster and hold less.
- **Changed (counters):** `stats()` adds `orthogonal_rows` (rows scored 0 by the consistency test alone),
  `projection_skips` (rows reusing the content's previous projection) and `value_builds` (contents whose value
  structure was built). `misses` and `memo_hits` now exclude orthogonal rows; `dker_hist` counts value builds only.
  `kernel_compiles`, `max_r`, `max_kappa_prime`, `r_hist` and `kappa_prime_hist` keep their meaning.
- **Changed (refusals):** the impossible-record checks and the content-level caps `RF_SECTOR_RANK` /
  `RF_LOGICAL_RANK` run only on rows that can have F > 0. An orthogonal row now scores 0 where 3.2.0 refused it
  (an impossible record on such a row, or a content past a cap whose rows are all orthogonal); a consistent row is
  refused exactly as before. Sampler buffers carry no impossible records.
- **Changed (memory):** a route-3b content whose value structure is built holds about 280 B more than in 3.2.0:
  184 B of per-row projection state (the right-hand-side constant and the previous row's b and R b, read on every
  row) and 96 B of structure headers. On buffers whose rows are all consistent this is +2-3% held bytes (20k shots:
  rp2/p1e-3/dec 64.13 -> 65.99 MB, color5/p1e-3/dec 65.33 -> 66.76 MB), so a `max_bytes` in that window that 3.2.0
  served is now refused ("the evaluator holds ... above the byte budget"). Buffers with orthogonal rows hold less.
- **Changed (C++ API):** route 3b computes a factor-list content's record pull-backs from its factor list
  (C0 = F_V ⋯ F_1), not from the `pull` callback of `RecordFidelity::intern_content`; `pull` serves route 3a.

## [3.2.0] - 2026-10-24

A large release since 3.1.13. In short:

- **Twirl consolidation.** The twirl sampler runs on one record law (the family record law) and is the default
  engine; the legacy twirl paths are deleted. There is no automatic engine choice and no silent fallback: what the
  twirl engine cannot serve is refused by name, and the exact engine runs only when asked for (`ENGINE_VERSION`
  xtim-engine-13).
- **Exact per-shot fidelity under a strict output contract.** `xtim.RecordFidelity` scores the qubits named by
  `OUTPUT_QUBITS`; the output must be one pure, fixed state with its frame closed by Pauli feedback, and output
  qubits are never measured. Violations raise `xtim.XtimFidelityRefusal` (`RF_OUTPUT_NOT_STRICT`,
  `RF_OUTPUT_MEASURED`, ...) naming the directions and records involved. Every bundled example follows the contract.
- **Removed:** the Target API and the `XTIM_QUIET` environment variable (use `warnings` filters and `logging`
  levels instead), plus several internal debug levers (see Removed below).
- **Faster term route** for circuits such as CCZ factories, and **default 512 MiB cache budgets** on every twirl
  sampler and on `RecordFidelity` (`None` = the default, `math.inf` = unbounded); output bytes are unchanged by
  the budget.
- **New README and docs:** a short front page, one API page (`docs/usage.md`), nine numbered runnable examples, and
  a new bundled example `color5_cultivation`.

The detailed entries follow, grouped by the development stage that produced them.

### README / examples rehaul (no byte-stream or `ENGINE_VERSION` change)

- **Bundled examples meet the strict output contract**: every `xtim.load_example(...)` circuit declares
  `OUTPUT_QUBITS` with its frames closed (Pauli feedback, or a noiseless move of the output onto fresh wires), and no
  longer carries `PAULI_EXPECTATION`; `RecordFidelity` scores each directly (noiseless F = 1 on every shot). The
  previous files stay in the test suite (`tests/data/`) unchanged. New bundled example `color5_cultivation` (SOFT's
  d = 5 colour-code cultivation, Apache-2.0).
- **Added: `xtim.XtimFidelityRefusal`** (a `ValueError`): every `RecordFidelity` RF_* refusal raises it, with the name
  in `.code`. `RF_OUTPUT_NOT_STRICT` names the undetermined output directions on the user's qubits (was: internal
  deferred wires) and the measurements whose outcomes fix them.
- Changed: a measured `OUTPUT_QUBITS` qubit is refused by `compile_twirl_sampler` as `XtimFidelityRefusal`
  `RF_OUTPUT_MEASURED` with a plain message (was: a `ValueError` parse error prefixed "line 0:").
- Changed: refusal texts no longer point at `engine="exact"`; a refused circuit or request is "outside xtim's model".
- `[decode]` extra: adds `ldpc` (BP-OSD for colour codes).
- Private CCZ-factory fixtures left the repository; the tests that use them skip when the files are absent.
- Docs: README cut to a short front page; one API page (`docs/usage.md`); the tour, dialect, practical-notes,
  DEM-reject and Stim-gap pages are removed; `docs/twirl_record_sampler.md` is the short engine contract.

### termroute-a: faster term route, and one default cache budget (output bytes unchanged)

- **Changed: every twirl sampler now has a 512 MiB cache budget by default** (it was unbounded at 3.1.13). It bounds
  the resident plan payloads, the readers' tables and, new, the term route's dispatch entries (each with its kind-2
  plan); the oldest are dropped first and rebuilt on demand. Output bytes are identical for every budget.
- **Changed: `compile_twirl_sampler(plan_payload_budget=None)` (the default) now means the 512 MiB default**, where
  it meant unbounded at 3.1.13. **`None` means the default everywhere** (`plan_payload_budget=`,
  `TwirlSampler.set_plan_payload_budget(None)`, `RecordFidelity(cache_budget=None)`). **`math.inf` is the one way
  to ask for unbounded**, on all three. A positive int is a byte count; `0`, negative values and other floats are
  refused by name (`ValueError`). `set_plan_payload_budget(None)` meant unbounded before; it now restores the
  default.
- **Changed: `record_report()["plan_memory"]["budget"]`** reports `536870912` (512 << 20) by default, and `math.inf`
  when unbounded. It reported `None` before.
- **Added: `RecordFidelity(cache_budget=)`** (default 512 MiB). An `evaluate()` call that finds more than this held
  (memo plus interned contents and kernels) drops it all first. Values are pure functions of their keys, so the
  output is unchanged. `stats()` gains `cache_budget` and `cache_flushes` (the calls that dropped). The register's
  construction-time structure is never dropped and stays counted in `stats()["held_bytes"]` and the `max_bytes`
  check, also after `clear()`.
- **Added counters:** `record_report()["term_route"]` gains `key_bytes` (the bytes the dispatch entries hold) and
  `key_evictions` (the entries dropped under the budget).
- Performance: the term route's reductions and kernel folds are word-parallel, and its per-column constants are
  memoised per sampler. On CCZ-factory circuits that take the term route, warm sampling is about 14x faster.
  The warm `RecordFidelity.evaluate` at the default budget is slower on long runs (about +35% on a large CCZ factory),
  because the held contents are dropped and recompiled about every 200k shots.

### S7: cleanup (no API, byte-stream or `ENGINE_VERSION` change)

- Removed debug levers (no environment levers; no dense routes): `FSYND_XCHECK` and `OVERLAP_FORCE_DENSE` are no
  longer read; the `CB_BATCH_XCHECK` and `IP_W1_XCHECK` build-flag differential checks are deleted.
- Internal: the uncalled `explain_feedback_class_exit` is deleted.
- **Removed: `XTIM_QUIET`** (no deprecation shim; setting it no longer does anything). Silence xtim with `warnings`
  filters (`xtim.XtimVerificationWarning` etc.) and `logging` levels (`xtim.twirl`).
- Changed: the twirl sampler's per-compile `[twirl_records] circuit channels: ...` line is a `logging` INFO record on
  `xtim.twirl` (same text), no longer a stderr line; it is silent unless the application configures logging.
- Removed stderr lines: the exact engine's `QEC_TREEPLAN_CAP` / `QEC_FORCE_FALLBACK` / `QEC_FRAMED_MEMO` lever notices
  (the levers themselves are unchanged), and the stderr copies of twirl refusals whose text already rides in the
  raised exception ("table not all-in-class", family refusals).
- Benchmarks: the ledger's fast-path replay reads `record_report()["counters"]["plan_misses"]` (`channel_report()` was
  removed at J-b1), so `ledger.py prepare` runs again.
- Changed: a malformed `XTIM_BARE_MEMO` / `XTIM_BARE_MEMO_ENTRIES` / `XTIM_BARE_MEMO_MB` / `XTIM_BARE_MEMO_ENTRY_MB`
  value is refused by name (`ValueError` naming the variable and the value) instead of a stderr line and a silent
  default. **Removed:** `XTIM_BARE_MEMO=trace` (and its stderr lines); `trace` is now refused like any unknown value.
- **Removed: `XTIM_FRAME_COSETS_BACKEND`** (and `xtim.frames.FRAME_COSETS_BACKEND`). The sector-table backend is the
  test-only `backend=` argument of `xtim.frames.plan_frame_cosets_fast` (default `"auto"` = the compiled table).
  `XTIM_TWIRL_CACHE_DIR` is unchanged.
- Changed: the twirl ctor's one-time alt-blob verification failure (an engine bug) is raised with its text
  ("twirl_records: BLOB VERIFY FAIL ...") instead of a stderr line and a generic "setup failed (code 5)".
- Changed: the internal-invariant `abort()`s in the FastTODD even-residue recovery (`normal_form`, `carried_compose`)
  and the residue projector (`residue_projection`) throw `std::logic_error` (a Python exception naming the "(engine bug)")
  instead of killing the process.

### J-b2: legacy twirl paths deleted (internal)

- Internal: the legacy twirl paths are deleted (legacy sampler, PPR / exact routes, σ dual basis, class switches);
  no API, byte-stream or `ENGINE_VERSION` change. Two refusal texts now say "served after J-c" /
  "in a later stage" instead of "after J-b2", and `QEC_TW_NULLSPACE_XCHECK` is no longer read (its kernel section is gone).

### EXP-fix: the exponential costs chi does not justify, and the beta-extraction fidelity route

#### Changed — DEM export (owner design 2026-10-21, docs/twirl_pauli_condition.tex "Noise model")

- Each fault of a channel instance is classified: a Pauli signature; a **uniform coset** `c' + W` (the stabilizer
  twirl of a Clifford fault — found by the +-1 character test on the Fourier basis with `wdim` Born sums, no 2^wdim
  walk, or re-detected after the walk when every |mu| is 0 or 1); or an explicit support (the general law; the
  PAULI_EXPECTATION per-shot random-sign list law). A single coset fault takes the **closed form**
  `(1-2q)^(2^(v-1)) = 1-p` on W\{0} (c' in W), or `(1-2q_c)^(2^v) = 1-2p` on c'+W and
  `(1-2q_W)^(2^v) = (1-p)^2/(1-2p)` on W\{0} (c' not in W; q_W <= 0: skipped, or the approximate_disjoint_errors
  refusal), O(2^(v+1)). A joint instance is solved by ONE fast Walsh-Hadamard transform over the 2^m group its
  supports generate (m 2^m); the O(4^m) loops and the per-fault 2^(wdim+vdim) pattern expansion are gone.
  All-Pauli instances of rank <= 4 keep the old solve, byte-identical.
- **DEM probabilities move within 1e-6 relative** (owner ruling 2026-10-21; measured on the 171-circuit corpus x 6
  legs: worst 3.5e-13). The summation order changes: decompositions may pick another split among exact score
  ties (5 corpus mechanisms reorder their components). Detector / observable / coordinate lines and every refusal
  of a circuit that refused before are unchanged except as listed below.
- The CCZ-factory class no longer hangs: `ccz_factory_a` / `ccz_factory_a_out` with
  approximate_disjoint_errors (> 300 s before) refuse by name in 2-4 s at 0.42 GB.

#### Added — named DEM refusals (owner cap)

- **Support cap**: a fault whose law spans a support of dimension v > 12 is refused (owner 2026-10-22; was 10 in the plan), before any 2^v work:
  "`<CHANNEL> at deferred stream index k (alternative X on deferred qubit q; user noise site i): the Clifford error
  this fault leaves triggers too many detectors: it randomizes nd detector(s) (and no other DEM target(s)) over a
  support of dimension v = <v> > 12; its exact DEM needs 2^v - 1 = <N> hyperedges (refusing)`". It replaces the
  "support of dimension > 16" text for 12 < v <= 16, and on the PAULI_EXPECTATION list path it fires before the
  joint walk ("a single fault correlates more than 63 parities" no longer appears for these faults).
- **Joint rank** (exact mode only): "`... the alternatives' joint signature group has rank m = <m> > 20 (component
  supports v = [..]); its exact conversion needs 2^m - 1 mechanisms (refusing; approximate_disjoint_errors emits the
  disjoint law)`". With approximate_disjoint_errors the disjoint law is emitted, as before.

#### Changed — frames

- `plan_frame_cosets_fast` (the compiled sector table) enumerates the 2^rank cosets directly and sums each in closed
  form (an exact Z4 Gauss sum) instead of walking 2^|supp a| masks: plans with |supp a| > 30 (refused before) are
  served. `weight` matches `plan_frame_cosets_numpy` to 1e-12 relative (was bit-for-bit; owner Q4); every
  structural field is exact. `FRAME_COSETS_ABI` 1 -> 2. New named refusals: |supp a| > 64, and the state budget
  (the A table + representatives + Gauss memo). The two Python oracles refuse past |supp a| = 24 / 1 GiB.
- `frames.projected_expectation` refuses by name above 16 sector operators (2^m Pauli products).

#### Removed (loud) — the beta-extraction fidelity route (owner 2026-10-21)

- `xtim.diagnose`, `Circuit.diagnose`, `Diagnosis`, `ExpectationInfo`, `OutputStateInfo`, the CLI `diagnose` verb;
  `Task(target_k=...)` and collect's `fidelity` / `infidelity` / `target_beta` row fields;
  `postselected_logical_error_rate(target_k=...)`, its fidelity mode and `PostselectedLER.target_k`;
  `xtim.fidelity_from_logicals` / `Circuit.fidelity_from_logicals`; docs/xtim_scoring.md,
  docs/xtim_logical_fidelity_helpers.md. Score an output state by direct fidelity estimation: `OUTPUT_QUBITS` +
  `xtim.RecordFidelity` (docs/twirl_record_sampler.md, "Scoring the bundled examples").
- `postselected_logical_error_rate` now scores OBSERVABLE_INCLUDE observables only and **refuses by name every circuit
  that declares PAULI_EXPECTATION** (including PE + OBSERVABLE circuits whose PE columns the old mode found
  unscorable and silently scored on the observable); strip the PAULI_EXPECTATION lines to score the observable.
- Kept: PAULI_EXPECTATION and all its engine consumers (`return_expectations` on the exact engine, CLI
  `detect --exp_out`, DEM L-columns), `extract_frame`, the collect/Task batcher, `scale_noise`.

### RF-ref: reference-mode RecordFidelity by the Gauss-sum route

- **No dense reference register.** `RecordFidelity(reference=True)` evaluates A = <B| Pi_{R_n} P C0 |B> on the
  k + u' virtual qubits of the register: C0's factors are pi/4 rotations composed into a stabilizer tableau, the
  Choi state of P.C0.Pi is read in closed affine form (a quadratic Gauss sum) and contracted against the bare
  amplitudes (direct Gray code over 2^{r'} terms, r' <= 2k, or a coset-wise Walsh-Hadamard form in O(k 2^k)). The
  reduction is compiled once per plan content with symbolic signs. Every u' is served (the 2^{k+u'} register and
  its refusal are gone); per-shot values are bit-identical to the dense route on the pinned cases. A pure-Pauli
  content (no rotation, no Born record) takes the dense route's own 2^k pass (C1).
- **Declared value change (commit 2, owner D-RF1 2026-10-21): the reference is conditioned on the RECORDED Born
  outcomes.** F = |<R|psi>|^2 / (<R|R><psi|psi>) with R = Pi_{R_n}(y)|B> the bare state projected onto every Born
  record at the row's recorded value, i.e. the former value divided by p_B(y_{R_n}). Values move only on circuits
  with Born-sampled records (kappa_rec > 0): `xtim.diagnose`'s state route on teleport-type circuits; in the byte
  corpus the born/* fixtures. A harmless stabilizer fault on a Born-read qubit now scores 1. `stats()["rn_words"]`
  counts the distinct recorded words whose norm was computed.
- A C0 factor that is not a pi/4 rotation (S, CZ, rot(B)) is refused by name: `RF_FACTOR_NOT_CLIFFORD`.
- **RF-ref strict (owner 2026-10-22): `RecordFidelity(sampler)` is the REGISTER fidelity on OUTPUT_QUBITS**, the only
  route: F = <phi| P_c Tr_R(rho_shot) P_c |phi> (ancillas and read wires traced out; `correction=` is the decoder's
  Pauli P_c on the output). The Target API (`xtim.Target`, `flips_from_paulis`), `reference=` and the conditioned-
  reference semantics are removed; the Clifford / poly target code is deleted (no dense route). STRICT output contract
  checked at construction (Theorem 1 on the bare stabilizer group, records off the output), refused by name:
  `RF_OUTPUT_NOT_STRICT` (names the undetermined output directions, (a, b, c, e) and the measurements that fix them),
  `RF_NO_OUTPUT`, `RF_RECORD_ON_OUTPUT`; `RF_CORRECTION_OFF_OUTPUT`, `RF_SECTOR_RANK` (> 2^20 rest sectors).
  Routes: a stabilizer output -> one rotation-free projector expectation; else the rest-sector sum of the Gauss-sum
  kernel, the sectors solved per shot. `xtim.correction_from_observables(sampler, pred, duals)` maps a decoder's
  predicted observable flips to the output correction with the user's duals. Circuits without OUTPUT_QUBITS, or
  whose output depends on unfed-back outcomes (e.g. the CCZ factory: e = 72, c = 9), are refused.
- **Lifecycle (C2).** Interned plan contents and their reference kernels persist across `evaluate()` calls
  (a warm evaluator compiles nothing); `evaluate(keep_memo=False)` drops the memo only; `clear()` drops
  everything. `stats()` gains `held_bytes` (everything held, the bare register included), `max_virtual_qubits`,
  `kernel_compiles`, `fast_path`, `contraction_direct` / `contraction_wht`, `rn_misses`, `rotation_misses`,
  `random_projection_misses`, `max_choi_rank`, `rprime_hist`. `memo_bytes` is the memo alone.
  `max_dim_bits` is the largest dense register (reference mode: the bare's k).
- **`max_bytes` (Q2)** also budgets what the evaluator holds (register + memo + contents + kernels + caches): an
  evaluation that would exceed it is refused by name ("... above the byte budget ...").

### J-b1: the production flip onto the family record law

One flip moves the twirl sampler onto ONE record law (the family record law) and makes it
the default engine. There is no automatic engine choice and no silent fallback anywhere:
what the twirl engine cannot serve is refused by name, and the exact engine runs only when
it is asked for.

#### Versions

- `xtim.ENGINE_VERSION`: `xtim-engine-12` → `xtim-engine-13` (every twirl byte stream is
  re-pinned in the J-b1 manifest; ftqcp stamps move with it).
- `_xtim.RECORD_KEY_LAYOUT`: 2 → 3 (`xtim.port` checks it). A retained record is (plan
  content handle, branch handle, frame-folded Pauli, outcomes). `record_layout()` stays
  version 2 (class-agnostic) with new entries.
- TWPL plan-cache format: v5 → v6 (plan keys of kind 0 and kind 2, with a term-set token).
  Older files load nothing and rebuild silently.

#### Removed (loud)

- **`engine="auto"`** raises `ValueError("engine='auto' was removed at xtim-engine-13: the
  twirl engine refuses by name what it cannot serve; pass engine='exact' to run the exact
  oracle")`. The default engine of `Circuit.compile_detector_sampler` is now `"twirl"`. The
  private `_exact_thunk` / `_exact` / `_AUTO_EXACT_WARNED` are gone.
- **Environment levers no longer read by the twirl sampler** (env vars cannot raise, so this is documented
  only): `QEC_NO_TWIRL` (the kill switch), `QEC_TW_SELFCHECK` (pass `selfcheck=`),
  `QEC_TW_FORCE_RECORD_FALLBACK`, `QEC_TW_COUNT`, `QEC_TW_OBSDBG`. **Still read** (diagnostic levers on the
  plan-build and bare-state paths, scheduled for S7): `QEC_TW_PROF` and `QEC_TW_NULLSPACE_XCHECK`
  (twirl_law.cpp), `QEC_PARTNER_HASH_MIN`, `XTIM_BARE_MEMO*`. `XTIM_QUIET` is unchanged.
- `diagnose()`: `OutputStateInfo.route == "expectation"` is removed; a refused circuit
  reports `route == "refused"` with `refusal=<TwirlClass name>` and no number from any engine.
- σ retired: `BarrierBuffer.sigmas()`, `.sig_bits`, `TwirlSampler.sig_bits`; port
  `Result.sig_bits`, `Result.group_sigmas`, `Segment.sigma_law`, `Segment.sig_bits`;
  `FramedSuperposition.port_sigma_law`; `_project_bare_onto_syndrome(circuit_text, sigmas)`
  (`AttributeError`).
- The exact-fallback class: `BarrierBuffer.exact_flags()`; port `group_exact`,
  `group_plans_or_none`; `record_layout()` kind `exact_y` and the per-shot `exact` key;
  `channel_report()["exact_shots"]`, `RecordFidelity.stats()["exact_shots"]` and the
  `record_report()["route_counts"]` `exact_*` keys (`AttributeError` / `KeyError`).
- `channel_report()` (`TwirlSampler`, `PartitionedTwirlSampler`): use `record_report()`
  (`AttributeError`).
- `frames(allow_kappa=, allow_exact=, allow_born_obs=, allow_born_dec=)`: no flags
  (`TypeError`); `frames()` fields `r`, `kappa`, `fallback`, `plan_id` (`KeyError`).
- `plan_structure()` keys `r`, `kappa`, `fallback`, `coin_masks`, `kernel_masks`,
  `kernel_pre`, `kernel_base`, `kernel_foldable`, `kernel_xz`, `kernel_phase` (`KeyError`);
  `a` / `cz` only when every plan of the export is kind 0.
- `skip_refused_observables=` (every constructor): nothing is refused any more (`TypeError`).
- `inject=`: every carried state is branch 0 (`TypeError`); the positional
  `_xtim.TwirlSampler(state, text, …)` / `(carried, src_wires, text, …)` overloads
  (`TypeError`; use `_xtim.TwirlSampler.from_ideal_state(*, ideal_state, …)`).
- `xtim._xtim_testapi.compile_family_partitioned`, `PartitionedTwirlSampler(_make_inner=)`.

#### Renamed (loud)

- `BarrierBuffer.plan_key(i)` → `plan_content(i)`; port `group_plans` → `group_plan_content`
  (kind 0 and kind 2).
- `xtim.frames.require_kappa_zero` → `require_kappa_rec_zero`.
- `prefix_xz` → `prefix_words` (`plan_structure()` / `frames()`).
- `compile_twirl_sampler_from_state` → `compile_twirl_sampler_from_ideal_state(…, ideal_state=…)`.
- `RecordFidelity.fidelities(buf, flips=, shots=, keep_memo=)` →
  `evaluate(buffer, shot_index=None, *, correction=None, keep_memo=False)`: the evaluator reads
  each shot's own record and takes no caller-supplied bits (`AttributeError`; `flips=` is a
  `TypeError`). A decoder's per-shot correction enters as a Pauli on the OUTPUT qubits,
  `correction=uint8[shots, 2, n_out]` (X mask row, Z mask row, `OUTPUT_QUBITS` declaration order),
  applied before the fidelity; it never touches records (owner ruling). The migration recipe for
  the old `flips=` (correction = flips · duals) is `tests/test_jb1_fidelity_vs_production.py`.
  `RecordFidelity(reference=True)` takes no correction: it raises `XtimTwirlRefusal(CORRECTION_NOT_SERVED)`
  (pass the target as atoms).

#### Changed (behaviour)

- `BarrierBuffer.materialize(i)` / `state(i)` still return a `FramedSuperposition` (as before);
  on layout 3 the state is rebuilt on demand from the shot's own record (gate-level route), and
  `branch_frames(i)` is `materialize(i).branch_frames()`. Speed is at parity with xtim-engine-12
  (ledger `api/materialize_*`, `api/pe_all_*`). The frame-preserving branch export refuses by name
  (`XtimCapacityError`, before any work) a plan whose produced state cannot fit the state budget.
- **`TwirlDetectorSampler.sample(return_expectations=True | return_measurements=True)`
  raises `XtimTwirlRefusal`** (`EXPECTATIONS_NOT_SERVED` / `RAW_RECORDS_NOT_SERVED`). It used
  to be served silently by a lazily built exact sampler under `auto`. The message names the
  escapes: `engine="exact"`; for an end-of-circuit `<P>`, the record route `sample_barrier` +
  `Target.pauli` + `RecordFidelity`; for raw records, `compile_sampler()`.
- **Explicit-exact surfaces (the documented oracle route):** `xtim.collect`,
  `python -m xtim detect`, the quickstart and `diagnose()`'s noiseless reference call
  `engine="exact"` explicitly, until expectations are served by the law (owner decision 9).
- **`xtim.ler` observable mode** (`postselected_logical_error_rate`) stays on the twirl
  engine and raises `XtimTwirlRefusal` by name on refused circuits (it used to go
  `auto → exact`). Fidelity mode runs through `collect` (explicit exact).
- **Refusals moved earlier:** bare-state rejects and class-14 circuits raise
  `XtimTwirlRefusal` at compile with a named class (they used to go `auto → exact` and raise
  `XtimRejectError` at `sample()`; the exact engine rejects the same circuits, so nothing
  samplable is lost).
- `engine_report()["reason"]` ∈ {`"default"`, `"requested"`}.
- **Gauge detector columns are exact Born records** (they were declared independent fair
  coins; the GAUGE stream is not drawn on the production path).
- **DETECTOR / OBSERVABLE columns on `INPUT_QUBITS` circuits and `ideal_state` compiles are
  reference-relative** (they were absolute). Census: 0 port samplers with a nonzero
  reference in adaptq and ftqcp.
- **Served now (formerly refused or routed to exact):** gauge detectors, refused and ANTI
  observables, ANTI detectors, no-channel (noiseless) circuits, Born decisions and retention
  on CH / PPR circuits, poisoned input frames, opaque-PPR dressing, `RecordFidelity` on Born
  decisions and `input_pauli` buffers, Y frames through CH. The per-shot exact-fallback routes
  are gone: their shots are served by the law (shot-selected replay against the exact engine
  in the manifest).

#### New refusals (by name, at compile; `xtim.XtimTwirlRefusal`, a `ValueError`)

The non-Clifford scope rule is the **static fixed-axis merge**: same-axis terms merge only
when nothing between them (gate or fault site with p > 0) can anticommute with the axis;
S remainders fold into the Clifford frame; iterated to a fixpoint.

| `TwirlClass` | circuits | census (corpus + xtim tests + adaptq + ftqcp) |
|---|---|---|
| `NONCOMMUTING_MAGIC` | same-axis terms that anticommute only through a fault (fault-dependent surviving axis / Clifford frame): a law limitation (the circuits are T-depth ≤ 1 per shot). Whether `engine="exact"` samples one is decided by the exact engine's own gate-by-gate propagation table: it does for the five texts listed here, and REJECTS others (e.g. `T;X_ERROR;T;H;T`, which the propagation table refuses first, and random fuzz members); the refusal text states the exact engine's verdict only where it is fixed | 5 synthetic texts: the four `test_ja_term_key` NONCOMM fixtures (`ch_pair`, `t_h_ch`, `t_h_ch_decision`, `NONCOMM_PORT`) and one CS/T witness (≈ 1 in 12,000 random CS/T circuits). **Real workloads affected: 0** (corpus 0, adaptq 0, ftqcp 0) |
| `LAW_LIMITATION_PROPAGATION` | the scope rule accepts, but a fault meets a same-axis non-Clifford pair before it merges (class 14); served after J-c (merged-op-list table). Outside the fast law and also rejected by the exact engine (the same propagation table); the refusal text says so | 29 texts: 27 fuzz, 2 xtim test texts. **Real workloads affected: 0** |
| `OUTSIDE_T_DEPTH_1` | the bare state's pushed magic axes anticommute (sub_reason `"bare"`); not new, now typed and at compile | — (existing refusal) |
| `EXPECTATIONS_NOT_SERVED`, `RAW_RECORDS_NOT_SERVED` | `return_expectations` / `return_measurements` on the twirl engine | callers rewritten to explicit `engine="exact"`: xtim tests and docs; external, loud at F: ftqcp `tests/test_certificates.py`, `tools/check_certificates.py`; 7 adaptq memo / probe / script uses (none in the adaptq package) |

Engine invariants raise `RuntimeError` (an engine bug), not a refusal. Parse errors raise
`XtimParseError` before any engine. The circuits the old bare builder refused but the rule
accepts (`T;T;H;T` and the even-Clifford ladder-gap fixtures) are now served.

#### Added

- `xtim.XtimTwirlRefusal(ValueError)` with `.twirl_class` (`xtim.TwirlClass`:
  `OUTSIDE_T_DEPTH_1`, `NONCOMMUTING_MAGIC`, `LAW_LIMITATION_PROPAGATION`,
  `LAW_LIMITATION_CARRIED_AXIS`, `EXPECTATIONS_NOT_SERVED`, `RAW_RECORDS_NOT_SERVED`,
  `CORRECTION_NOT_SERVED`) and `.sub_reason`.
- `BarrierBuffer.signs(paulis) -> int8[shots, m]` and port `Result.signs(paulis)`. The
  contract is CONSERVATIVE: 0 means indefinite or not certified; a reported sign is never
  wrong, and the evaluator may under-report in the magic state's own-symmetry case.
- `BarrierBuffer.plan_content(i)`; port `group_plan_content`.
- `record_report()`: `classes`, `nondeterministic_detectors`, `nondeterministic_observables`,
  branch counts, `branch_table_digest`; on `PartitionedTwirlSampler` too (static schema of
  the reference sampler, counters summed over the inner samplers, plus `patterns`).
- `plan_structure()` / `frames()`: `prefix_words`, `plan_index`, `plan_key_digest`, `kind`,
  `terms`, `term_axes`, `rho`, `kappa_rec`, `born_xz`, `born_phase`, `record_layout`, the
  record operators and y, `frame_folded`.
- `record_layout()` entries for gauge / ANTI / PPR-plan records and the branch handle.
- Branch API: `compile_twirl_sampler_from_ideal_state`, `_xtim.TwirlSampler.from_ideal_state`,
  `add_branch(h, vector=None)`, `input_branch` (per-shot handle, default branch 0),
  `branches(shots)`, the content handle in layout 3 and `frames()`.
- `RecordFidelity.evaluate(buffer, shot_index)` on every record kind, with frames and Born
  decisions.

#### Speed (F's final tip 94c21c1; ledger = engine Ir per shot, valgrind; timing = wall, 3 accepted rounds, vs b9c2e92)

| component | before (b9c2e92) | after (F) | gate |
|---|---|---|---|
| t_fast/d5s (cultivation d5, sample), Ir/shot | 4,435.7 | 4,443.5 (+0.18%; +2.36% cumulative vs PRE-S0 4,341.0) | ≤ +5% cumulative vs PRE-S0 |
| t_fast/{fac, ccz_b}, Ir/shot | 2,209.3 / 8,948.0 | 2,225.7 / 8,984.7 (+0.74% / +0.41%) | ≤ +5% cumulative |
| t_fast/{e7, t0, t1} (formerly PPR family + exact routes), Ir/shot | 18,873 / 9,516 / 13,919 | 2,897 / 2,730 / 2,802 (−85% / −71% / −80%) | ≤ +5% cumulative |
| t_retention/{d5b, d3b}/sample_same_batch, Ir/shot | 4,402.2 / 464.2 | 4,409.5 / 474.5 (+2.35% / +2.02% cumulative vs PRE-S0) | ≤ +5% cumulative |
| barrier recording run, layout 3 (t_fast d5b / d3b / bt81b), Ir/shot | 30,608 / 1,966 / 12,602 | 12,013 / 1,225 / 2,103 (−61% / −38% / −83%) | ≤ +5% cumulative |
| t_fid/* fidelity route (`sample_barrier` + `OUTPUT_QUBITS` + `RecordFidelity`), µs/shot: Color d=5, RP2 d=5, d5b, CCZ factory | 25.36 / 31.00 / 34.61 / 67.97 | 9.55 / 11.95 / 14.50 / 17.15 (0.38× / 0.39× / 0.42× / 0.25×) | 5% vs first green |
| cold start (fresh sampler, every plan built), µs/shot: Color d=5 det, RP2 d=5 det, CCZ factory, private CCZ factory b | 8.86 / 8.63 / 48.45 / 89.80 | 7.54 / 9.01 / 42.97 / 73.74 (0.85× / 1.045× / 0.89× / 0.82×; plain-process variant RP2 1.031×) | ≤ +5% |
| `BarrierBuffer.materialize` / `pauli_expectations_all`, µs/call: cultivation d5, CCZ factory | 161.9 / 163.2; 1,271.7 / 956.8 | 162.0 / 163.0; 1,279.2 / 947.0 (1.00× / 1.00×; 1.02× / 0.99×) | ≤ +5% |
| `frames()` µs/row, cold: d5 / factory | 17.31 / 38.28 | 6.21 / 4.27 | ≤ b9c2e92 |
| t_pprbuild/{e7, t0, t1} (plan build, report-only), Ir | 152.9M / 97.5M / 123.3M | 166.8M / 145.0M / 164.2M (+9% / +49% / +33%) | report |
| allocations per shot (sampling components t_fast / t_retention) | 0 | 0 | 0 (hard) |

σ is absent from the fidelity path after F: a native profile of the factory fidelity route (6,581 samples) has no
σ frame (the family sampler builds core-mode payloads, no σ law).

### Added

- **S2b record readers and accounting (r6–r8).** `record_report()["plan_memory"]` gains `reader_lookups`,
  `reader_tables`, `reader_bytes`, `reader_peak_bytes`, `reader_overshoot_bytes` and `reader_evictions` (the readers'
  record-table memo, bounded and charged to the plan payload budget; reader tables are evicted before any sampler
  payload, and readers never evict a sampler payload). `BarrierBuffer.record_tables_held()` reports the tables a compact
  buffer holds for its readers (one per distinct plan, resolved on first read; user data outside the payload budget).
  `plan_structure()["rec_folds"]` exports the introducers built with the fold bit β′ = 1. RecordFidelity.evaluate and
  frames() derive y word-parallel with no per-shot allocation.

- **Plan memory: permanent plan headers, evictable payloads, a caller payload budget (twirl
  consolidation S0(c), D6).** A twirl plan is now a permanent HEADER — its key stored ONCE as a
  bitset inline in the header (a-support ‖ CZ pairs over the circuit's pair universe), a dense
  content id, the r/κ/fallback flags — plus an evictable PAYLOAD (law vectors, channel rows,
  observable data; the same split for PPR plans). Headers are never freed, so a plan's id is its
  content for the sampler's lifetime: the fast-route index is the cache's own flat (hash, header)
  table (the duplicated key copies and the separate shared fast index are gone), and the Born
  memo key and the compact barrier record intern on the id. `TwirlSampler.set_plan_payload_budget
  (nbytes)` / `compile_twirl_sampler(..., plan_payload_budget=)` bound the resident payload bytes
  (`None` = unbounded, the default; `0` is refused by name): payloads are evicted only between
  shots (oldest first) and rebuilt on demand, so every output byte is identical for every budget;
  a shot's own payloads are exempt and any transient overshoot is reported in
  `record_report()["plan_memory"]` (budget, resident / peak / overshoot bytes, header bytes,
  payload builds, evictions). The closed-form canonical key is verified against the canonical
  normal form when a plan header is created and on every self-check-window shot (a mismatch is a
  loud engine-bug refusal); every index probe compares the full stored key (collision safety). TWPL format v5 stores plan KEYS only (payloads rebuilt on first use; v4 files rebuild
  silently). No sampled byte changes (no ENGINE_VERSION bump).
- **`record_layout()` version 1: typed per-class offsets of the outcome record (twirl
  consolidation S0(d)).** `BarrierBuffer.record_layout()` gives, per shot and per class in record
  order — `fair` (the plan's r fair coins), `kernel` (its κ kernel-chain outcomes), `born_obs`
  (the Born observable's raw outcome), `born_dec` (the Born decisions' raw outcomes) — a
  `{at, len}` pair, plus `prefix` / `born_obs_at` / `born_dec_at` / `len` (= `coin_layout()`);
  `TwirlSampler.record_layout()` / `port.Segment.record_layout()` give the static part
  (`version`, `classes`, `has_born_obs`, `nborn`, `born_dec_index`). It coexists with
  `coin_layout()` until J-b1, whose layout is version 2 with disjoint keys — check `version`.
  On the adv-C p2/p9 fixture (Born observable + decision) 0 of 4000 shots mismatch (the old
  `r + kappa + j` reader: 965). Read-only: no sampled byte changes.
- **The twirl sampler has no channel cap (twirl consolidation S0(f)).** The per-shot channel rows
  were fixed `uint64_t[8]` arrays, so a circuit with more than 512 record channels (detectors +
  in-group observables + in-group decisions) was refused at compile ("no deterministic detector
  channels"). The rows are now run-width (`ceil(nchan/64)` words, sized once per sampler).
  Byte-identical for every circuit with <= 512 channels; a 600-detector circuit is gated against
  the exact engine (`tests/test_wide_channel_rows.py`).
- **`TwirlSampler.record_report()`: exact route + memo/miss counters (twirl consolidation S0(a)).**
  `route_counts` gives every shot's route (the two hot paths `fast` / `retain_kappa0` are
  attributed per run, never counted per shot; the `exact_*` keys are the per-shot exact-engine
  fallbacks, one key per trigger); `plan_misses` / `ppr_plan_misses` split law builds by the built
  plan's class (`kappa0` / `kappa_pos` / `fallback`); `memo_nodes` counts the Born-decision memo
  entries and `born_trie` the κ>0 Born-trie walks (see the S2a entry below). Values are exact at a
  fixed seed. Every exact-fallback trigger is reported once per run in which it fired, including
  the formerly SILENT non-commuting PPR route, as an `XtimPerformanceWarning` (see the S0d entry).
  `channel_report()` is unchanged (it stays until J-b1). Sampled bytes are unchanged.
- **Twirl-engine diagnostics are Python warnings / logging, not stderr lines (twirl consolidation
  S0d, diagnostics policy).** The engine queues its events and the bindings raise them in Python
  after the call returns (never from inside the shot loop): an exact-fallback trigger is an
  `XtimPerformanceWarning` (one per trigger per run), the self-check oracle window an
  `XtimVerificationWarning` (once per process), an oracle failure an `XtimCorrectnessWarning`
  followed by the call's oracle-abort `RuntimeError`, and a plan-cache disk load an INFO record on
  the `xtim.twirl` logger (NullHandler by default). The texts are the former stderr lines.
  `XTIM_QUIET` still suppresses the self-check-window and disk-load events (the variable is
  retired in S7). Sampled bytes are unchanged.
- **The per-plan Born trie replaces the κ chain (twirl consolidation S2a).** A κ>0 shot's Born
  outcomes are a walk of its plan's trie (nodes `{p0, kid[2]}`, keyed by the z-prefix, allocated on
  visit, evicted and rebuilt with the plan payload); a missing node is evaluated on the bare free
  register only, never on an n-qubit copy of the bare state. Every node equals its from-scratch
  definition bit for bit, so bytes are identical to the chain's (no re-pin, no ENGINE_VERSION
  bump). `record_report()["born_trie"]` (walks, steps = Σκ, nodes, node_evals, passes, chi_j,
  max_support, max_depth) replaces the removed `kappa_chain` key. A call that evaluates a node
  logs ONE INFO record (`born_node`) on `xtim.twirl` with χ, κ and that node's evaluation time,
  emitted when the call returns (owner decision 7: no threshold, no cap).
- **The state route is the default way to score a protocol's output.** `xtim.RecordFidelity`,
  `xtim.Target` and `xtim.flips_from_paulis` are exported at the top level (the module path
  `xtim.record_fidelity` keeps working). The README's front-page loop now scores the output STATE
  (`OUTPUT_QUBITS` → `sample_barrier` → decoder Pauli corrections → exact per-shot fidelity); the
  `PAULI_EXPECTATION` route stays documented as the alternative, with the cost model for choosing
  (state route `O(chi)` per distinct record for any target; expectation route one column per
  declared Pauli, up to `4^k - 1` for a full k-qubit target). The README loop is executed by
  `tests/test_readme_snippets.py` and must print what the README says it prints.
- **`diagnose()` scores a declared output on the state route.** For a circuit that declares
  `OUTPUT_QUBITS`, the new `Diagnosis.output_state` (`xtim.OutputStateInfo`) holds the exact
  per-shot fidelity vs the noiseless reference (`RecordFidelity(sampler, reference=True)`) on ONE
  sampled run of the circuit as given (`shots`/`seed` of the call; no decoding): the mean over all
  shots and 1 − F over the post-selected shots (no deterministic detector fired) — reported
  numbers only, nothing applied to the caller's data. When the state route refuses (e.g. the twirl
  sampler refuses the circuit), `output_state.route == "expectation"` and `.refusal` names why;
  the report says so and reads the `PAULI_EXPECTATION` columns. No existing `Diagnosis` field
  changed meaning; a circuit without `OUTPUT_QUBITS` gets `output_state=None` and the same report
  text as before (the header gains "output state: one sampled run of the circuit as given" only
  when the state route ran). `diagnose(..., output_state=False)` skips the run; the internal
  callers (`collect`, the post-selected LER helper, the fidelity helpers) pass it, so they pay no
  twirl compile / `sample_barrier` they would not use (call-counted in the test; their outputs
  are byte-identical to before). Tests: `tests/test_diagnose_state_route.py`.
- **`xtim.record_fidelity`: exact per-record fidelity of a retained output state with a target,
  computed in the bare state's shared stabilizer frame (no `materialize`).** Per shot the retained
  state is `Pi_rec(sigma) · P · S^a · CZ |B>` (exactly `materialize(i)`: the record-channel
  projection, no separate kernel-fold projectors). Every operator is pulled back through the
  Clifford `W = P·S^a·CZ` to a Pauli; the dressed record channels and the Pauli target generators
  are reduced by one GF(2) elimination of their fixed-row parts (the "kernel lemma"), so the only
  exponential left is `2^{u'}` from polynomial (Clifford / Pauli-sum) generators whose atoms leave
  the stabilizer sector in unmeasured directions (`u' <= 2` on the private CCZ-factory fixture, 0 on the other
  validation fixtures; random hypergraph targets reach larger `u'`).
  Per shot: `O(#atoms · n/64)` atom-sign parities + a `(plan index, sign key)` memo (the
  `reference=True` target also takes the prefix's frame image, `O(|P|·n/64) <= O(n²/64)`); per
  distinct record: `O((#projectors + #monomials) · 2^{k+u'})`. The memo is a pure speed device
  scoped to one `fidelities()` call by default (`keep_memo=True` keeps it; `clear()`;
  `stats()["memo_bytes"]`). Targets: Pauli stabilizer generators (with signed
  record frames), Clifford stabilizers `A·prod CZ(A', A'')` (|CCZ>/hypergraph), Pauli sums such as
  `(X_L + Y_L)/sqrt2`, and the full-state overlap with the noiseless bare (`reference=True`).
  Decoder corrections are per-shot atom sign flips (`flips_from_paulis`). Refused by name: Born-class
  DECISION circuits, buffers sampled with `input_pauli`, buffers from a circuit
  whose state model (bare state, certified generators, record channels, the Born-sampled
  observable — compared bit for bit)
  differs from the evaluator's, record frames on `M(p)` records, generators that are not commuting
  Hermitian involutions, and a record whose exact register need `16 B x (2·2^{k+u'} + 2^k)` exceeds
  the caller's `max_bytes` budget (default: no budget; a failed allocation is refused by name).
  Decoder flips must come from deterministic detectors; gauge-dependent corrections belong in the
  atoms' record frames (gauge DETECTOR bits are independent coins w.r.t. the retained state). Validated per shot: the private
  CCZ-factory fixture (240,000 shots, 75,035 accepted, exact-MLE decode) reproduces the prototype's
  1 − F = 9.80e-4 ± 1.02e-4 with every shot within 2.2e-16 (3,163 shots also against an independent
  `materialize(i)` route); the merlin circuits (undo `|+>^m`, cultivation `(X+Y)/sqrt2`, no-undo
  reference overlap) against `materialize(i)` on every record group; 288 random dense-oracle
  circuits (T/CS/CCZ, forced faults, MPP, feedback, gauge outcomes, record frames, Pauli
  corrections) within 5.6e-16. New read-only accessors: `TwirlSampler.record_reads()`,
  `TwirlSampler.output_ports()`, `TwirlSampler.record_channels()`, `BarrierBuffer.exact_flags()` (the barrier sink now records the
  exact-fallback flag it already received); `TwirlRecordSampler::bare_state()`,
  `certified_generators()`, `record_channel_masks()`. Additive: every existing stream, record and
  DEM text is byte-identical to the pre-merge build on the bundled examples.
  **Exact-fallback shots are served, not refused** (with the C1 fix their exact state is retained):
  the buffer's exact record is the shot's canonical residual plan bytes followed by the
  observable's raw byte, and the exact engine's state on it is the evaluator's own formula, so one
  algorithm serves both shot classes (`stats()["exact_shots"]`). **A Born-sampled observable is
  part of the scored state** (owner decision 2026-10-13): its projector is carried like a record
  channel in numerator and denominator; the reference overlap gains the `(-1)^raw <B|W PC|B>` term.

- **Stim's correlated-error chains — `E` / `CORRELATED_ERROR` + `ELSE_CORRELATED_ERROR` — on every
  engine (owner-approved 2026-10-13).** A chain is parsed as ONE noise site
  (`NoiseChannel::CORRELATED`): its support (the distinct qubits of all terms, first-appearance
  order), the sequential probabilities verbatim, and each term's Pauli product as factor codes
  over the support (`Instr::corr`, the any-arity generalization of `PAULI_CHANNEL_2`'s slot map).
  Stim's sequential-conditional semantics make it a categorical Pauli channel — term t fires with
  `q_t = p_t Π_{j<t}(1-p_j)` — computed in ONE place (`corr_outcome_probs`, circuit_ir.hpp) and
  consumed by every route: the propagation table's support, `sample_noise`, the framed/exact
  engine and the twirl record sampler (one categorical channel per chain, alternatives = the
  terms that can fire, any count — the `DiagErrorSampler` 255-alternatives guard was not
  structural and is gone), forced faults (`compile_exact_branches`: a chain site has one target,
  its whole support; `noise_sites()` lists `qubits`/`probs`/`paulis`; any non-identity word over
  the support may be forced), DEM export (Stim's rules exactly: a single-term chain is one exact
  mechanism; a chain with an `ELSE` requires `approximate_disjoint_errors`, every categorical
  component is threshold-checked, and every term is its own independent mechanism at `q_t` —
  same-signature terms XOR-compose, `error(0.172) D0` for `E(0.1) X0 / ELSE(0.1) Y0` as in stim),
  the resolved-text round trip (re-emitted as `E` + `ELSE_CORRELATED_ERROR` lines), `diagnose()`
  / `without_noise()` (chains stripped), `scale_noise` (rescales the categorical law exactly and
  writes the chain back in sequential form), and the deferred-signature / TWPL plan-cache keys
  (a chain's term supports are structural and enter the key; existing circuits' keys are
  unchanged, no cache version bump). Validation mirrors Stim (one probability in [0,1], Pauli
  targets only, `!` prefix accepted and ignored, an empty product is an identity term) plus
  Stim's error-analysis adjacency rule enforced at parse time: an `ELSE_CORRELATED_ERROR` must
  directly follow its chain (any other line, `TICK`/annotations included, or a `REPEAT`/`IF`
  boundary ends it — refused by name), and a qubit repeated within one term is refused by name.
  Oracles (`tests/test_correlated_error.py`, 55 tests): stim DEMs on 24 random chain-bearing
  memory circuits (mechanism sets and values at 1e-12, refusals and threshold identical,
  categorical values bit-exact), stim detector statistics at 1e6 shots, the hand-written
  categorical law on every sampling path (records / exact / twirl / port) and deterministic
  chain components with scrambled factor order, a 15-term chain encoding `DEPOLARIZE2(p)` =
  the native law on every path (incl. a T-type magic example on exact and twirl), the dense
  oracles on non-Clifford circuits (chains before/after T and CCZ, X-type outcomes crossing a
  CCZ: sampled law + every forced single outcome), forced-address refusals, `scale_noise`,
  `diagnose`. ctest: `test_channel_support_atoms` covers chain supports (dead and identity
  terms). Byte identity vs 3.1.13 on the 9 bundled examples + the 69-circuit corpus (DEM text,
  sampler streams, barrier buffers): 3,095 fields, 0 differing — chains are additive. Docs:
  `docs/xtim_dialect.md` §2b-v, `docs/xtim_forced_faults.md`, `docs/STIM_GAP_AUDIT.md`.
- **Correlated chains commute with pending single-qubit Clifford frames (dev/next merge of
  fix/noncommuting-magic-readout + feat/correlated-error).** `conjugate_noise_through_frames`
  relabels every chain term factor by factor (`corr'[t][i] = preimage_i(corr[t][i])`, sequential
  probabilities untouched) and keeps the chain ONE instruction; the forced-fault program maps a
  user word through the same per-qubit letter provenance. Previously (the H-frame branch's
  extension point) a chain flushed the frames. Law tests: dense oracle (exact forced law +
  sampled law, 24 random frame triples on magic inputs), stim DEM equality on a Clifford probe;
  mutation (no relabel) turns both red.

### Changed

- **Engine kernels without `thread_local` scratch (stage TLC).** The framed measurement / Born /
  canonicalise / batch-measurement / active-block / TreePlan kernels keep their working buffers in
  a caller-owned `FramedScratch` (the exact engine owns one, released at the end of every compile
  and run; cold callers own one per call) instead of per-thread storage; `dual_image`,
  `CertifiedGroupPlanes::reduce` / `reduce_cheap` and the `FramedSuperposition` expectation /
  measurement pivot search no longer need scratch at all. Every output byte is identical (no
  ENGINE_VERSION bump). The exact engine's `__tls_get_addr` calls drop from ~451 per shot to ~1
  per collapse (cultivation d5, engine="exact": -14% total Ir per shot). State-budget accounting
  changed (declared, docs/repin/TLC-budget-manifest.md): the chi-scaled kernel scratch is charged
  to its owner for one compile / run / call and is no longer retained per thread between calls,
  so `_thread_scratch_bytes()` covers only the CanonicalStabSum oracle's remaining slots; no
  corpus circuit's peak rises.
- **Every diagonal record is a row of the plan's record table; ENGINE_VERSION `xtim-engine-12`
  (twirl consolidation S2b; re-pin manifest `docs/repin/S2b-manifest.md`, signed by the owner
  2026-10-18, OD-2026-10-18-s2b-signoff).** A diagonal plan now carries its RECORD TABLE in the record-ordered canonical basis
  (design Theorem 1 / Lemma 4, `cpp/include/qeccore/record_table.hpp`): every record column — the
  IN_GROUP channels, the Born observable and the Born LOGICAL / ANTI decisions — is
  `A·u ⊕ B·z ⊕ β ⊕ t(P)`, where the coins u are the values of the pivot records (coin k = COIN bit
  k of the shot) and z are the Born outcomes of the canonical directions (BORN word j = direction
  j; each a literal ordered product of the σ / extra directions, its exact phase kept). The Born
  observable and the decisions are added to the channel table by ONE `add_extra` each (one plane,
  one pivot reduction, one `G.reduce`, one free-register echelon; counted in
  `record_report()["rec"]`). Consequences:
  * one emission for every diagonal route (clean shots are the identity plan's rows; the fast
    accumulator route serves kappa_rec > 0 plans too); the Born observable and Born decisions never
    take an exact route on in-class plans (the routes `exact_born_obs_kappa_pos`,
    `exact_born_dec_kappa_pos` and `exact_obs_guard` are deleted, with their `route_counts` keys) and
    never read an OBS / DEC stream there (the DEC stream is retired; OBS remains for PPR plans only,
    the exact fallback draws EXACT words ng / ng + 1 + j: `record_report()["rec"]["obs_draws"]`);
  * the record trie walks only the directions a record reads (`born_trie.steps` = Σ kappa_rec);
  * the barrier's σ is drawn by **σ-4** (owner 2026-10-16): conditioned on the channel records
    and the Born observable — never on the decisions — through the plan's σ map (the certified
    generators as record columns after C); its fair coins read COIN bits from rho_rec on and its
    directions a σ-aux trie on BORN words from kappa_rec on, so `sample()` and `sample_barrier()`
    records stay byte-identical. `record_report()["sigma_aux"]` counts its walks; the relation
    channel bit = mask·σ is checked inside the selfcheck window by a named throw (no production
    counter);
  * the retained record is (plan, P, u, z): `BarrierBuffer.outcomes(i)` holds the outcomes (u, z)
    (an exact-fallback shot: its raw record values y), `record_outcomes(i)` derives y, and
    `materialize(i)` is a forced projection of the pivot / direction records at y (a zero-weight
    anchor is a named engine-bug error);
  * `record_layout()` is version 2 (per-shot segments `coin` / `born` / `exact_y`); version 1,
    `coin_layout()` and `plan_structure()["coin_*"]` are removed (see Removed).
  Every twirl `sample()` / `sample_barrier()` byte stream with a coin, a Born direction or an extra
  moves; the `sample()` joint law of detectors / observables / decisions, the barrier joint law of
  (records, σ), every retained state as a function of its record, and every exact-engine / DEM /
  forced-fault law are unchanged (gates: tests/test_s2b_record_law.py, tests/s2b/
  test_sigma_joint_law.py, tests/test_s2b_state_equals_bits.py, the must-not-move list of the
  manifest).
- **State-budget accounting of amplitude entries changed; outputs byte-identical (twirl
  consolidation S2k).** A FramedSuperposition / DenseAmplitudes entry now stores its free-register
  sign pattern as ONE packed 64-bit key (bit d = sigma_d) instead of a heap byte vector, and
  `inner` / `axpy` find keys through a transient O(chi) index (direct addressing when 2^k <=
  next_pow2(2 chi), else open addressing) instead of a linear scan: a Born-trie node costs
  O(chi (k+1) (j+1)) instead of O(chi^2). The list order (the summation order), first-match
  semantics, the stable prune and the ascending-bit diagonal product are unchanged, so every
  sampled / exported byte is identical on every engine (no ENGINE_VERSION bump). What changes is
  the budget accounting: an entry is charged sizeof(entry) = 24 B (was sizeof + 48 B = 88 B for
  the key's heap block), the bare-build peak estimate counts it the same way (estimate == charge
  is kept), and the lookup index's heap slots are charged (4 B each) by an RAII ticket — a state's
  reads and projections use a per-call index on the stack (64 inline slots; heap beyond them
  charged for the call only; a state retains no index and its copies are unchanged), the Born trie
  owns one per sampler (charged while it lives). Consequently
  `live_state_bytes()` values and `XtimCapacityError` byte counts are lower, and a circuit near
  the `XTIM_MAX_STATE_BYTES` boundary can move from refused to served (list:
  `benchmarks/ledger/s2k/budget-manifest.md`, owner-signed 2026-10-17). A free register wider than 64 coordinates (states assembled from
  rows or tensor products; `add_index`) is refused by name (`XtimCapacityError`); a sigma byte
  other than 0/1 in `from_rows` is refused (`ValueError`).

- **One keyed counter-based generator for every twirl stream, one coin convention on every route
  (twirl consolidation S0(b); the owner-signed re-pin D11).** Every random draw of the twirl
  sampler is now `mix(key(seed, tag) + (shot·2^20 + ctr + 1)·γ)` — the SplitMix64 finalizer of a
  keyed counter, the stream tag hashed into the key (`cpp/include/qeccore/keyed_rng.hpp`) — with
  `shot` the cumulative shot index: fair coins (bit k of the shot's COIN words, on the fast AND the
  structural/retention route, which used one `next_double() < 0.5` draw per coin), kappa Born steps
  and PPR folds, the Born observable, Born decisions, readout flips (the tag no longer XORs into the
  seed), gauge coins (formerly a sequential `mt19937_64` in the binding) and the per-shot
  exact-fallback engine (formerly sequential `mt19937_64` streams). The noise-event stream keeps its
  geometric skips but is keyed per noise channel (it was one xoshiro stream drawn channel-major per
  4096-shot batch). A Born ±1 outcome is +1 iff `u < p(+1)` on every route. Consequences (D10): the
  bytes no longer depend on the plan-cache state (cold / warm / TWPL-warm), on the route
  (`sample()` and `sample_barrier()` agree unless one surface sends a shot to the exact fallback),
  or on how shots are split across calls (ANY split, not only multiples of 4096). Scope (D13): the
  same binary on the same CPU feature set. The TWPL plan-cache header carries `ENGINE_VERSION`
  (format v4); a file another engine wrote loads nothing. A shot index ≥ 2^44 per seed, or ≥ 2^20
  draws of one stream in one shot, is refused by name (the layout's injectivity range). Every twirl
  `sample()` / `sample_barrier()` byte stream moves; laws are unchanged (S0(b) law battery: 5 sigma
  vs the pre-change build and `engine="exact"`, noiseless-relation counters, an exact-enumeration
  check of the structural-route coins). `ENGINE_VERSION` `xtim-engine-10` -> `xtim-engine-11`.
  The engine="exact" (framed) sampler's streams are unchanged. Tests:
  `tests/test_keyed_streams.py`, ctest `keyed_rng`; generator micro-gate:
  `benchmarks/rng_draw_bench.cpp`.
- **docs/branch_frames.md §7 / `BarrierBuffer.frames()` docstring: the retained-state contract now
  states what `materialize(i)` does.** The state is `P·S^a·CZ·|bare>` projected onto the shot's
  RECORD CHANNELS `G_c` at `(-1)^{c·sigma(i)}` — not onto every certified generator at
  `sigma(i)[b]`, and with no kernel-chain collapse at kappa > 0. The 3.1.13 text was false whenever a
  plan dresses a generator no record channel reads (counterexample in §7: an X fault on the |0>
  control of a CCZ whose target is measured in X without a detector). The pinning test now projects
  onto the record channels (`test_frames_state_is_projected_onto_the_record_channels`) and pins the
  counterexample (`test_unread_dressed_generator_is_not_projected`). Docs/tests only; no engine change.

### Fixed

- **`BarrierBuffer.frames()` refuses Born-class decisions by default (`allow_born_dec`) —
  BEHAVIOUR CHANGE (fix/frames-born-dec).** On a circuit with a Born-class DECISION
  (LOGICAL/ANTI; `born_dec()`), `frames()` with default arguments now raises `ValueError`
  naming the decision indices, where it used to export silently. Every exported field was and is
  exact residual data, but the shot's STATE is additionally projected onto each decision operator
  at its recorded raw outcome, which the frame does not carry — the old docstring and
  docs/branch_frames.md §7 said the frame (projected onto the record channels) IS the state, which
  was wrong on every such shot (200/200 on the review's probes). Pass `allow_born_dec=True` to get
  the export unchanged (no new field at that commit); the raw outcomes are (S2b) the decisions'
  columns of `frames()["y"]` and the operators `born_dec()`. The docstring and
  §7 now state the full contract, `|ψ_i⟩ ∝ M_dec(i)·Π_W(i)·Π_c(…)·P·S^a·CZ|bare⟩` with every outcome
  RAW, and that its factors commute (each is a record product over one read basis per wire).
  Pinned by tests/test_frames_born_dec_refusal.py and the G-frames gate tests/test_frames_contract.py
  (commutation assertion from exported data; composed state vs a golden exact-engine law pinned
  from 4c0c26b). Binding-layer only: no sampled byte changes (no ENGINE_VERSION bump).
- **`BarrierBuffer.obs()` refuses refused observables by name — BEHAVIOUR CHANGE
  (fix/barrier-refuse, E1).** On a circuit with observables the twirl sampler refuses (operator
  not IN_GROUP: no exact record channel), `sample()` raised by name but
  `sample_barrier().obs()` silently returned all-zero columns with the default
  `skip_refused_observables=False` (e.g. an exact law of ~0.32 reported as 0). `obs()` now raises
  `sample()`'s exact `RuntimeError` through the same single check; with
  `skip_refused_observables=True` it returns the documented zero columns as before.
  `sample_barrier()` itself, `dets()`, `decisions()`, the states, `frames()` and `RecordFidelity`
  are exact and unchanged, so state/record consumers (port, diagnose's state route,
  record fidelity) are unaffected. The same holds for `frame_dressing()`: under the default its
  whole `"obs"` array (the observable flips, written as 0 for a refused observable whose true flip
  can be 1) is omitted and `"obs_refused"` (indices) / `"obs_refused_reason"` (sample()'s exact
  message) are added; with the opt-in, and on circuits without refused observables, the dict is
  unchanged. Binding-layer only: no sampled byte changes (no ENGINE_VERSION bump). Pinned by
  tests/test_barrier_refusal.py.

- **`seed=None` now continues the twirl coin stream across calls.** The per-shot coin generator
  of `TwirlSampler.sample()` / `sample_barrier()` was indexed by the shot's position within the
  call, so two consecutive unseeded calls replayed the same twirl/collapse coins shot for shot
  (a coin-driven detector agreed on 100% of shots across calls instead of ~50%; the noise stream
  did continue). The index is now cumulative across calls, as the readout-flip counter already
  was; an explicit seed (and `set_seed`) resets it to 0, so every explicitly seeded call is
  byte-identical to before. `sample(N, seed=s)` then `sample(M)` now equals `sample(N + M,
  seed=s)` byte for byte when N is a multiple of the 4096-shot noise batch (the noise sampler
  draws channel-major within a batch, so its bytes depend on the batch split; its law does not).
  Tests: `tests/test_coin_stream_continuation.py`.
  `ENGINE_VERSION` is bumped `xtim-engine-9` -> `xtim-engine-10` (unseeded byte streams changed;
  downstream caches and provenance stamps keyed on it invalidate/restamp).
- **Coin-record layout is exported; the `born_dec` docstring was wrong with a Born observable.**
  A shot's coin record is `[r fair | kappa kernel | Born-observable raw outcome (iff the circuit
  has one) | Born-decision raw outcomes]`, so with a Born observable decision j is at
  `r + kappa + 1 + j`; the `born_dec` docstrings (and `port.Segment.born_dec`) said
  `r + kappa + j`, which reads the observable bit instead. The engine now exports the offsets
  from one definition (`TwirlRecordSampler::CoinLayout`): `BarrierBuffer.coin_layout()` (per
  shot, exact-fallback shots included), `TwirlSampler.coin_layout(n_coins)` /
  `port.Segment.coin_layout(n_coins)` (for a `Result.group_coins` row), and the additive
  `plan_structure()` keys `coin_born_obs_at`, `coin_born_dec_at`, `coin_len`. `frames()` reads
  `born_obs_raw` through it. Sampled streams are unchanged. Tests: `tests/test_coin_layout.py`.

- **`OUTPUT_QUBITS 0 1 2` (a qubit index where the port NAME goes) parsed silently as a port named
  "0" holding qubits 1 and 2** — qubit 0 dropped from the port without a word. The grammar is
  `OUTPUT_QUBITS <name> q0 q1 ...`; an all-digit first target is now refused by name with the
  corrected line. `INPUT_QUBITS` / `INPUT_BITS` had the same silent mis-parse (`INPUT_QUBITS 0 1
  2` -> port "0" holding qubits 1 and 2) and get the same refusal. Tests:
  `tests/test_output_qubits_grammar.py` (red on the previous build).
- **Tests: `test_zz_summary` / `test_zz_coverage_summary` failed under `pytest -n 16`** (they read
  module-level summaries filled by tests xdist had sent to other workers). Each summary now runs
  the same oracle, with the same assertions, for any declared workload its own process did not run
  (memoised per process) — order- and worker-independent, nothing weakened.
  `tests/test_ref_battery_sizing.py` reads the bundled examples through `xtim.load_example` (works
  from an installed wheel). Tests only.

- **CRITICAL silent-wrong (review C1): exact-fallback shots of `sample_barrier` retained the
  identity-plan state instead of their own (every release since 2.5.0).** A shot routed to the
  per-shot exact engine under state retention (3.1.13: a Born-class OBSERVABLE on a κ > 0 or
  classify-guarded plan, or a record-path guard trip; ≤ 3.1.12 also every shot on a κ ≥ 2 plan
  whose legacy fold fell back) was stored with plan id 0 / empty plan bytes, so
  `BarrierBuffer.materialize(i)` / `state(i)` / `branch_frames(i)` / `pauli_expectations_all`
  replayed the IDENTITY-plan state for it (noiseless-like; on the review repro 650/4,000 shots
  contradicted their own fired detector), `frames()` exported an identity frame, `plan_key(i)`
  returned `b""`, and `record_groups()` merged such shots with clean shots (a group's
  representative could hand the wrong state to every member). Fix: the sampler hands the barrier
  sink the shot's EXACT record (canonical residual plan bytes ‖ the born observable's raw value);
  `materialize(i)` replays it through `TwirlRecordSampler::materialize_exact_shot` — the same
  operations, in the same order, as the exact shot itself (`apply_residual_gates_` +
  `exact_hand_on_project_` are now shared by both), then the born decisions with their coins.
  New `BarrierBuffer.exact_flags()`. Record keys carry the exact record in the plan component
  under an explicit kind tag (record-key layout 2, below), so exact shots never share a group
  with non-exact shots. Refusals by name, never a placeholder: `plan_key(i)` on an exact shot;
  `frames()` when any shot is exact unless `allow_exact=True` (then exact rows have
  `plan_id = 0xFFFFFFFF`, a zero prefix and `exact[i] = 1`; the `exact` field is always
  present); `xtim.frames.require_kappa_zero` on an export with exact rows; port
  `Result.group_plans` when any group is exact (new `Result.group_exact`).
  **Byte identity:** sampled streams (dets/obs/decisions/σ/coins), DEM text and every output of
  NON-exact shots are unchanged (69-circuit corpus incl. the bundled examples, both stores:
  3,095 fields, 0 differences). Changed outputs, exact shots only: their materialized states,
  their record-key rows / group ids (and hence the group numbering and key width of buffers
  that contain exact shots), `plan_key` (now refused) and `frames()` (refused by default).
  Fix round 1: `port.Result.group_exact` is taken from the buffer (`exact_flags()` at each
  group's first shot), never from the key bytes (a non-exact plan of >= 32,768 bytes no longer
  reads as exact or truncates); `Result.group_plans` still refuses when any group is exact, and
  the new per-group `Result.group_plan(g)` (refuses only an exact group) and opt-in
  `Result.group_plans_or_none` (None for exact groups) keep non-exact plans readable. New oracle
  lever `QEC_TW_FORCE_RECORD_FALLBACK=1` (read once per process; retention path only) routes every
  retained diagonal shot through the record-path exact fallback so that class is testable.
  Tests: `tests/test_exact_fallback_retained_state.py` (repro vs own detector, independent dense
  oracle incl. born decisions, group purity, refusals, port, non-exact goldens from 3.1.13);
  mutation builds `XTIM_MUTATE_C1_EXACT_PLACEHOLDER` (9 tests red) and exact identity dropped
  from the keys (2 red).
- **Record-key layout 2 (OWNER-SIGNED re-pin 2026-10-13).** `record_keys()` / `record_groups()`
  key rows (both stores) now read σ bytes ‖ u16-LE len_coins ‖ coins (padded) ‖ **kind u8 (0 =
  residual plan, 1 = exact record) ‖ u32-LE length** ‖ plan / exact record (padded). Layout 1
  (≤ 3.1.13) had a u16 plan length — it wrapped SILENTLY at 65,536 bytes — and no kind tag (the
  interim exact marker was bit 15 of that field with a 32,767-byte refusal; both are gone). A
  coins record of ≥ 65,536 bytes is now refused by name (it wrapped silently before). New
  `_xtim.RECORD_KEY_LAYOUT = 2`; `xtim.port` checks it and cross-checks every group's kind tag
  against the buffer's `exact_flags()`. Only the raw key BYTES change: group partition,
  first_occ, streams, states and frames are byte-identical to a575d4c on the 69-circuit corpus.
  Re-pinned (and only these): the three `_STREAM_HASHES` of
  `tests/test_frame_cosets_cpp.py::test_sampling_stream_unchanged_since_3_1_2` — with the
  record_keys() contribution removed they agree between a575d4c and this build, and this build's
  keys re-encoded to layout 1 reproduce the old pins. Tests: plans over 65,536 bytes with and
  without exact shots (no refusal, no truncation, round trip through `Result.group_plan(g)`);
  mutation `XTIM_MUTATE_C1_DROP_KIND_TAG` (exact and plan records alias) turns 6 tests red.
- **A Born-sampled observable conditions every shot's retained state (owner decision
  2026-10-13).** Before, ordinary (non-exact) shots of a circuit with a Born-class OBSERVABLE and
  no Born DECISION materialized UNconditioned on the observable's outcome, while exact shots and
  every circuit with a decision were conditioned. Now, under retention, the observable's raw
  outcome always enters the coin record right after r + κ (the slot the decision path already
  used) and `materialize_shot` projects W onto it through the existing `obs_raw_from_coins_` /
  `force_project_` machinery — no new path. `materialize` / `state` / `branch_frames` /
  `pauli_expectations_all` and the record keys reflect it (shots differing only in the outcome no
  longer share a group). `frames()` cannot carry the projection in P·S^a·CZ: on such a circuit it
  refuses by name unless `allow_born_obs=True`, which exports `born_obs_raw` int8[shots] (−1 =
  none), `born_obs_xz` and `born_obs_phase` (W); `xtim.frames.require_kappa_zero` refuses exports
  with an outcome. New `TwirlRecordSampler::born_obs_op()`. sample() and circuits without a Born
  observable are untouched. Changed outputs (corpus): only the Born-observable circuits without a
  decision — their coin records, record keys/groups, non-exact states and frames refusal.
  Mutation `XTIM_MUTATE_OBS_NO_PROJECTION` (the old split) turns 4 tests red.

### Removed

- `record_report()["sigma_aux"]["channel_violations"]` (S2b r6.5): the σ-4 channel relation is checked inside the
  selfcheck window by a named throw, not counted in production.

- **`coin_layout()` (BarrierBuffer, TwirlSampler, port Segment), `record_layout()` version 1 and
  `plan_structure()["coin_born_obs_at" / "coin_born_dec_at" / "coin_len"]` (twirl consolidation
  S2b).** The record no longer has a fair / kernel / Born-tail coin layout: read the version-2
  segments (`record_layout()["entries"]`, `rho` / `kappa_rec` per shot) and the raw record values
  (`BarrierBuffer.record_outcomes(i)`, column order `TwirlSampler.record_operators()`).
- **Renamed (twirl consolidation S2b): `BarrierBuffer.coins(i)` → `outcomes(i)`, port
  `Result.group_coins` → `group_outcomes`; `frames()["born_obs_raw"]` → the record values
  `frames()["y"]` (with `born_obs_col`, `record_xz`, `record_phase`).** The old names raise
  `AttributeError` / `KeyError`. `record_report()` loses `memo_nodes` (the Born memos are deleted)
  and the route keys `exact_born_obs_kappa_pos`, `exact_born_dec_kappa_pos`, `exact_obs_guard`,
  `exact_chain_guard` (unreachable). `xtim.RecordFidelity` reads every record channel and the Born
  observable from the record's y (`record_outcomes`), never from σ.

## [3.1.13] - 2026-10-13

### Changed

- **Twirl record sampler: κ ≥ 2 kernel directions are folded by a canonical dual basis — the
  κ≥2 `law.fallback` to the exact per-shot engine is gone (owner-signed derivation 2026-10-12).**
  `build_shot_law` now builds, for EVERY law with κ ≥ 1, the fold masks δ_1..δ_κ as the dual basis
  of the kernel preimages c_j on the admissible fold space W = C_det^⊥ (c_j·δ_k = δ_jk, δ_k ∈ W,
  each reduced modulo the coin RREF — one κ×(r+κ) GF(2) solve, no per-direction pick, no cap, no
  tunable). The dual basis always exists (the preimages restricted to W are independent because
  the kernel directions are independent modulo D), so the former κ≥2 trips — "kernel preimage/fold
  pairing != delta_jk" and "coin+kernel masks rank-deficient" — were artifacts of the fold choice
  δ_j := c_j, not representability limits; they and the two provably unreachable branches ("kernel
  preimage not ⊥ coins", "no kernel fold direction") are removed, and the fold conditions (every
  fold in C_det^⊥, pairing = I, coins ∪ folds of rank r+κ) are engine-bug asserts
  (`std::logic_error`; mutation-checked: δ := c_j trips the pairing assert on the factory and the
  C_det^⊥ assert on the [[4,2,2]] fixtures). The exact per-shot engine remains only for the
  pre-existing spec guards (PPR, born-mode κ>0, mid-chain guard). On the private CCZ-factory fixture
  (`ccz_factory_a.stim`, 729 qubits, detectors only) `channel_report()["exact_shots"]` drops
  from 365/25,000 (1,416/100,000) to 0 and the sampler goes from 49.6 to 3,710 shots/s
  (interleaved, quiet host); the law-build cost per distinct plan falls 1,104 → 820 µs there
  (the legacy "[qubit rows ‖ C_det] probe" solve is gone) and is unchanged on cultivation_d5
  (26.8 → 26.5 µs; the per-build rank RREF no longer runs at κ=0). Law identity against the
  3.1.12 engine on the factory at 1e5 shots: 312 detector marginals max|z| = 0.29, 351 pairwise
  joints of the 27 κ-touched detectors max|z| = 0.82; κ=0 shots byte-identical (22,745/22,745).
  **Declared byte change (law-preserving):** κ=1 laws now use the plain qubit-row preimage and the
  canonical fold instead of the legacy "[qubit ‖ C_det] probe" pick — the same law (the fold is
  unique modulo the coin gauge; the base bit is representative-independent), a different coset
  representative in a minority of κ=1 plans (29 of 1,016 κ=1 shots on the factory at 25k shots;
  every κ≥2 shot's bytes move since its fold changed). κ=0 laws never enter the kernel section and
  are byte-identical: the 9 bundled examples (twirl + auto streams, DEM text) and the 358-circuit
  pinned/captured corpus (4,000 shots each) are byte-identical, and a per-plan count over that
  corpus finds no κ≥1 plan (1,408,000 shots, 45,862 plans). `ShotLaw`/`plan_structure` gain
  `kernel_pre` (the κ preimages, additive). The TWPL disk plan cache version is bumped 2 → 3 (v2
  files store `fallback=1` for the tripped κ≥2 plans and legacy κ≥1 folds; version skew rebuilds
  them).

- **The `[twirl_records] circuit channels: ...` stderr banner prints by default again on every
  twirl sampler compile** (owner decision 2026-10-13: 3.1.12's opt-in is reverted). `XTIM_QUIET=1`
  silences it, as before. `XTIM_TWIRL_VERBOSE` (3.1.12's opt-in switch) is no longer read; setting
  it has no effect. stderr only: no sampled stream or DEM byte changes.

- **`BarrierBuffer.frames()` contract corrected (documentation; the export is unchanged).** The
  docstring and `docs/branch_frames.md` §7 said each retained shot's state is
  `P·S^a·CZ·|collapsed⟩`. It is that expression PROJECTED onto the shot's recorded syndrome
  `sigma(i)` (every certified generator `g_b` at eigenvalue `(-1)^sigma(i)[b]`), which is exactly
  what `materialize(i)` returns. The projection is the identity unless the plan's `S^a·CZ` moves
  a certified generator, which needs `a` or a `cz` pair on a wire where a generator has X support,
  e.g. CZ on measurement-record wires. In that case the unprojected expression leaves those
  detector parities undetermined. The private CCZ-factory fixture study found 4–15 indefinite detectors
  per such shot; projecting onto `sigma(i)` reproduced `materialize(i)` on 312 detectors × 298
  shots, κ > 0 included. A consumer of such a shot projects onto `buf.sigma(i)` or calls
  `materialize(i)`. What `frames()` returns is byte-identical.

- **Version labels in code comments, docs and the 3.1.12 CHANGELOG entries corrected** (text only).
  The per-shot `PAULI_EXPECTATION` judgement, its owner drop rule and the `pe_max_work` cap shipped
  in 3.1.12 but were labelled "3.1.13" (the dev-branch label). They now say 3.1.12, and the
  pre-change behaviour they contrast with is labelled 3.1.11. The places fixed are `xtim/circuit.py`,
  `cpp/src/dem_export.cpp`, `cpp/src/dem_pe_per_shot.cpp`, `cpp/include/qeccore/dem_pe_per_shot.hpp`,
  `cpp/include/qeccore/dem_export.hpp`, `docs/xtim_dem_reject.md`, `examples/README.md`,
  `examples/dem_reject_region.py`, `examples/xtim_tutorial.ipynb` and the test comments. The
  per-shot entry's "byte-identical to 3.1.12" now names the build that change was made on, and
  the factored-law entry's "3.1.12 walk" / "identical to 3.1.12" now name the pre-factoring build
  fd4f31c. Outputs are byte-identical.

### Added

- **`xtim.XtimVerificationWarning`: a reduced reference self-check says so** (owner decision: a
  warning only). Sometimes the work-sized noiseless-invariants battery runs fewer shots than its
  ceiling: 500 at a reference compile (`compile_reference`, a cache miss of `compile_*`) or 200 at
  a cache-hit verify. When that happens, the Python compile path emits ONE warning per compile.
  The warning gives the shots run vs the ceiling, `W_compile`, `w1`, the sized count, and the miss
  bound: "a wobble confined to outcome paths of probability q is missed with probability
  (1-q)^N". The battery is silent when it runs in full (w1 = 0), which is every bundled example
  and every corpus circuit. Neither `UserWarning` subclass already in use fits:
  `XtimCacheWarning` means caching failed, and `XtimPerformanceWarning` means a slow diagnostic is
  active. `ref_compile_text` / `ref_verify_cheap` gain `invariants_ceiling` (additive).
  On a cache miss the warning comes AFTER the reference is stored, so under
  `warnings.simplefilter("error", XtimVerificationWarning)` the exception reaches the caller but
  the entry is cached (the next compile is a hit, whose warning propagates the same way). CLI:
  `python -m xtim state compile` goes through `compile_reference` and prints the warning on
  stderr; `python -m xtim state verify` calls the binding directly and does NOT emit it (its
  `VERIFY PASS` line does not report the shot count).

### Tests

- `tests/test_ref_battery_sizing.py`: the cap-2 kMagicPE circuit (w1 = 42, W_compile = 214,
  sized 5) warns once at the compile and once at the cache-hit verify, with those numbers and
  `(1-q)^N`. A second compile of the same `Circuit` stays silent, and every bundled example is
  silent through both paths. `tests/test_branch_frames.py::test_frames_state_is_projected_onto_the_recorded_syndrome`
  is a 4-qubit toy with the factory's mechanism: an X fault before a CCZ leaves CZ on the two MX
  record wires. On it the unprojected `P·S^a·CZ·|bare⟩` is not `materialize(i)` (two generators
  indefinite). Projecting onto `sigma(i)` with engine Clifford gates and a forced `measure_pauli`
  gives `materialize(i)`, and a flipped σ bit does not. `materialize(i)` has every certified
  generator at `(-1)^sigma(i)[b]`. `tests/test_twirl_decision_detector_sampler.py`: the banner
  prints by default, and `XTIM_QUIET=1` wins, also over the retired `XTIM_TWIRL_VERBOSE=1`.

- `cpp/tests/test_kernel_dual_basis.cpp` (ctest `kernel_dual_basis`): the solver on random GF(2)
  systems (ng ≤ 128, κ ≤ 4): folds in W, pairing = I, rank r+κ, coin-reduced, invariant under the
  preimage class and the C_det basis, coin gauge; dependent preimages throw; the note's Prop. F
  instance pinned. `tests/test_twirl_kappa_dual.py`: dense-oracle law identity (independent
  statevector of the fixture circuit; the engine's deferred ancilla wires separated through their
  certified single-wire stabilizers) at κ = 2 and 3 on six [[4,2,2]]-block fixtures handed to
  `plan_structure` as hand-built residual keys, each of which the ported pre-change fold choice
  trips on (the 3.1.12 binary reports `fallback=True` on all six); mutations m1–m3 fail the
  identity, m4 (δ ⊕ coin) passes (the coin gauge, recorded). `tests/test_twirl_factory_kappa.py`
  (vendored `tests/private/ccz_factory_a.stim`): `exact_shots == 0` and 5σ agreement of
  the detector marginals and κ-touched pairwise joints with the pinned 3.1.12 counts
  (`tests/private/ccz_factory_a_golden.json`, 1e5 shots). Evidence labelling: the factory gate
  checks the law identity and `exact_shots == 0` only — it is structurally blind to the pairing
  condition and to the base-sign bits (one diagonal magic frame ⇒ every kernel-logical product
  has stabilizer expectation 0/±1 there, |⟨L⟩| ≤ 4e-16 on all 3,661 products of its κ≥1 plans);
  the pairing / base-sign claims rest on the dense fixtures and the solver ctest.
### Performance

- **Exact-engine compile: the noiseless-invariants battery is WORK-SIZED ("verification never
  costs more than compilation"), and a failed TreePlan extraction is never repeated.** On
  `ccz_factory_a` (chi=512 block, n=1809) with its `PAULI_EXPECTATION` column an
  `engine="exact"` compile took 1124 s cold (1032 s reference battery + 92 s program compile) and
  620 s on a warm cache (fresh process); now 116 s cold (104 s + 12 s) and 115 s warm. Without the
  column: 557 s -> 118 s cold (single-threaded, host load ~2.5).
  Neither change touches the reference state: the `.ref` text, every sampled stream, DEM text and
  exact expectation is byte-identical on the bundled examples, tests/data, benchmarks,
  blowup/circ, the 302-circuit adaptq capture and the 400-circuit random corpus.
  - `noiseless_invariants` (compile: 500 shots; cache hit `ref_verify_cheap`: 200) runs
    `clamp(W_compile / w1, 1, ceiling)` shots, the ceiling being the former fixed count.
    `W_compile` = the collapse work of compiling the checked sampler program (TreePlan extraction
    attempt + reference shot), `w1` = noiseless shot 0's, both in the engine's deterministic unit
    `dual_image_scan_work()` (the scan kernel's row-word visits; no wall clock, no tuned
    constant). Every circuit whose noiseless shots replay a TreePlan (w1 = 0: every corpus circuit
    above) runs the full 500 / 200 — same shots, same verdict; the factory runs 28 with its
    expectation column (W_compile = 5.55e10, w1 = 1.95e9) and 60 without (w1 = 9.2e8). The shots are ONE sequence (new `sample_program_streamed`, a prefix
    of the ceiling-shot run): while a detector has read 1 on every shot so far the same sequence
    continues (to the ceiling at most), so the detector verdict is exactly the fixed battery's.
    The |expectation| wobble check sees only the shots run: a wobble confined to outcome paths of
    total probability q is missed with probability (1-q)^N (q = 1/64: 0.64 at N = 28, 4e-4 at
    500); the cache's DEFSIG fingerprint does not cover that. `ref_compile_text` /
    `ref_verify_cheap` report `invariants_shots`, `invariants_sized_shots`,
    `invariants_shot_work`, `invariants_compile_work`. A `shots=0` verify now runs no shot (a
    vacuous pass) instead of reading shot 0 of an empty batch.
  - A failed TreePlan extraction is a pure, mode-independent function of (block, reads, cap). The
    R>0 channel rebuild of a block whose zero-error extraction failed is now SKIPPED explicitly,
    and failures are recorded in a process-level memo (exact input serialisation as the key;
    separate slots for the zero-error plan and the noisy-shot pattern plans; a hit credits the
    recorded work to the counter, so `W_compile` is the same fresh or warm) — 4 cap-bound
    extractions per exact compile -> 1 per process. Diagnostics: `_xtim._tree_fail_memo_stats()` /
    `_tree_fail_memo_clear()`.
  - **Declared behaviour change (refusal -> samples).** The skipped channel rebuild used to run
    and could refuse with `XtimCapacityError` under a tight `XTIM_MAX_STATE_BYTES` while capturing
    leaf states of a tree that was then discarded (e.g. a 4-qubit T circuit with a
    `PAULI_EXPECTATION` at `QEC_TREEPLAN_CAP=2` compiled while an earlier program of ~11.7 KB is still
    alive, `XTIM_MAX_STATE_BYTES=12000`). Such a circuit now samples,
    exactly — the samples equal those at a budget large enough for the old path. The outcome is
    named: the exact engine's `engine_report()` (and `CompiledProgram.plan_report()`) gains
    `zero_error_plan` / `expectation_channel`, the latter `"skipped: zero-error plan failed (live
    block)"` in this case.

## [3.1.12] - 2026-10-12

### Changed

- **DEM export with `include_expectations=False` carries EVERY fault on its detector/observable
  signature: nothing is refused, flagged or dropped on column grounds (owner decision 2026-10-12).**
  The completeness gate ("non-fault-tolerant circuit (<loc>): ..."), the post-select flags and the
  accepted-branch checks are statements about the `PAULI_EXPECTATION` columns; with no column
  requested a refusal or an omission named nothing the requested output got wrong. Now
  `detector_error_model_with_reject(include_expectations=False)` (the default is now `True`, see below) returns the
  detector/observable DEM — byte-identical to the twin circuit without the declarations — with
  `postselect_faults` / `reject_detectors` empty, and what the column analysis would have refused
  (the gate in both forms, and the two "accepted branch was not verified" refusals: `> 16
  detectors`, joint law over the work cap) is listed in `dropped_expectation_columns`, one
  `(column, reason)` per (column, fault location), the reason naming the location and column as the
  refusal does and suffixed `(reported, not refused: include_expectations=False)`;
  `detector_error_model(include_expectations=False)` / `detector_error_model_text(...)` raise an
  `XtimDemWarning` with the same text (they have no report field). **The reject region therefore
  exists only with `include_expectations=True`** (the docs' TL;DR and `xtim_practical_notes.md`
  now say so; `examples/dem_reject_region.py` already used it). With `include_expectations=True`
  the refusals, flags and DEM text are byte-identical to before. Only no-expectation exports
  change, and only on circuits the column analysis refused or flagged: on the 1,711-circuit DEM
  byte sweep (4 legs each) that is 18 random Clifford+T circuits / 35 exports — 24 formerly
  refused (12 circuits) and 11 formerly exported with the flagged faults left out (6 circuits) —
  none of them bundled examples, `tests/data`, blowup or vendored circuits; each is now
  byte-identical to its twin's DEM, and the no-expectation export equals the twin's on EVERY
  class-admitted corpus circuit with a declaration (708/708 export legs, 0 exceptions). `test_magic_t_variant_refuses_loudly`, `test_accepted_branch_refusal_message`,
  `test_silent_wrong_repros_now_refused_per_their_exact_law` and the `test_api` flagged circuit
  pinned the old behaviour and now pin the new contract;
  `test_non_ft_finding_is_reported_not_refused_without_expectations` checks the D/O DEM against the
  twin's and the reported text (restoring the refusal turns them red).

- **`detector_error_model_with_reject` now defaults to `include_expectations=True`**, matching
  `detector_error_model` (owner decision 2026-10-12). The method exists for the reject region, which
  exists only with the expectation columns; after the same-day ruling its old default `False` would
  have returned empty flags silently. A caller who wants the plain detector/observable DEM with
  empty flags passes `include_expectations=False` explicitly. `with_reject(include_expectations=True)`
  output is unchanged vs 3.1.11/16eb1ef (same code path, byte-identical); default calls change by
  design: `dem` carries the expectation L-columns after the ordinary observables (ignore the
  decoder's predictions for them) and the flags/gate apply. Tests moved to the explicit `False`
  where they test the no-expectation mode (`test_api` `gen0`,
  `test_non_ft_finding_is_reported_not_refused_without_expectations`,
  `test_accepted_branch_refusal_message`, the `t_bell_z0_emit` check); the tutorial's cultivation
  cell passes `approximate_disjoint_errors=True` (as `test_api` and the docs already did).

- **Forced faults: `exact_branches` returns the exact FACTORED law; the 2^r branch walk is
  removed.** The walk resolved every terminal read in record order and branched on each random
  one, so a query cost 2^(#random reads) leaves even when no target depended on them (a d=3
  surface memory plus 40 `MX` ancillas: past any cap; `cultivation_d5` weight-1: 16 ms). Now
  every target — detector / observable (reference-relative), decision (raw recorded parity) —
  is composed once per program into an affine GF(2) form of the raw reads (`!` inverts, forced
  flips and the feedback relabel included), and the stabilizer twirl of the bare state splits
  the law: target directions in the span of the reads' X-parts outside the magic sector are
  independent FAIR COINS (`uniform_detectors/observables/decisions`, plus
  `uniform_expectation_signs` for the columns each flips), never enumerated; the rest is
  deterministic or a function of d <= k magic-sector parities whose joint law is walked on its
  support only (`PeShotAnalyzer::joint_law`; deterministic caps: `max_rows` rows and `max_work`
  work units, default 4e8 ~ 1 s: k |T> states read in X answer up to k = 8, k >= 9 is refused
  naming the magic rank). Result
  (`ExactBranches`): `rows_probabilities/detectors/observables/decisions/expectations` (every
  coin 0) and the EXPANDED view `probabilities/detectors/observables/decisions/expectations`
  (rows x 2^u coin subsets, all distinct; built lazily on access; `ExpandedLawTooLarge` beyond
  `max_rows`). Expectations are now the CONDITIONAL expectation given the row's (detectors,
  observables, decisions) only — the probability-weighted average of the old per-branch values
  over branches with those targets; every other read, declared frame records included, is
  averaged over. **Breaking:** raw per-branch records are no longer returned (no
  `measurements`), rows are merged by targets, `max_branches` is gone (`max_rows` instead), and
  the C++ `sample_program_exact_branches` / `ExactBranchResult` / `qeccore::exact_branches` and
  the binding's `ExactBranchProgram.exact_branches` are removed (`sample_program_exact_law`,
  `qeccore::exact_law`, `ExactBranchProgram.exact_law`). Before removal the law was checked
  against the walk aggregated by targets on 25,087 queries (dense random suite, the bundled
  examples, every single fault of `cultivation_d5` incl. folded flips, the held-fold circuit,
  Stim memories, ftqcp's qrm15 magic-t-prep Choi program): max |dp| 1.8e-15, max |dexp|
  5.2e-15. Per query (quiet host, expanded view read): `cultivation_d3_faithful` weight 1-3 36 us
  (walk 149 us), `cultivation_d5` weight 1 152 us (walk 16.0 ms), dense random suite 14 us
  (walk 21 us).
  Tests now use the independent dense statevector oracle (aggregated by targets; it gained
  RX/RY/MRY and every other stim Clifford gate), the exact sampler, and for `cultivation_d5`
  and the qrm15 Choi program a golden record of the walk (`tests/data/forced_walk_golden.json`).

- **DEM disjoint-channel refusal now uses Stim's wording.** When `detector_error_model()`
  refuses a channel that is not a product of independent record-flip mechanisms (the exact
  conversion needs a negative probability), the `XtimDemError` message now opens like Stim
  1.16's: "Encountered the operation <NAME> during error analysis, but this operation
  requires the `approximate_disjoint_errors` option to be enabled." plus the python hint
  (`approximate_disjoint_errors=True`), followed by xtim's detail (deferred stream index;
  "exact DEM conversion has a negative probability"). Exception class unchanged.

### Fixed

- **`compile_detector_sampler(engine="twirl"/"auto")` (the default) crashed on every circuit that
  declares a `DECISION`** (`ValueError: too many values to unpack (expected 2)`), and — the root —
  compiled the circuit's DECISIONs into its twirl engine although it never emits them: a
  Born-class decision forced the per-shot Born path (1.2-3.6x slower than the same circuit
  without the lines) and, on CS/CH/PPR circuits, a `RuntimeError` at `sample()` ("Born decisions
  unsupported for CH/PPR circuits") AFTER `engine_report()` had said twirl. The detector sampler
  now compiles its engine with the new `compile_twirl_sampler(..., decisions=False)` /
  `_xtim.TwirlSampler(..., decisions=False)` flag: no DECISION output channel exists, so the
  engine is exactly the engine of the same circuit with its DECISION lines removed — det/obs
  streams byte-identical to it at the same seed, at its speed, with none of the Born-decision
  path's refusals — and returns exactly what `engine="exact"` returns (same shapes/dtypes for
  `separate_observables`, `append_observables`, `bit_packed`; the exact detector sampler has no
  decision output either). Lossless because a DECISION is never consumed inside the engine (`IF`
  conditions read `INPUT_BITS` port bits supplied by the caller); decision bits remain on
  `xtim.port` / `sample_barrier` / `compile_twirl_sampler` (default `decisions=True`, unchanged).
  Circuits without DECISION are byte-identical.

- **IF blocks without bit values now REFUSE BY NAME on every `xtim.Circuit` surface**
  (`compile_sampler`, `compile_detector_sampler` for exact/twirl/auto, `detector_error_model_text`
  / `_with_reject`). An `IF name[k] { ... }` body is a per-shot conditional on the caller's
  `INPUT_BITS` (circuit_ir.hpp), but no engine path under `xtim.Circuit` takes those bits: the
  exact sampler and the DEM export ignore the IfBegin/IfEnd markers and ran EVERY body
  unconditionally (undocumented, silently wrong — `IF drv[0] {X 0}` measured 1 with probability
  1); the twirl partition sampler already refused at `sample()`. The refusal names the IF
  condition(s) and the escape (`xtim._xtim.resolve_branches_text(text, bits)` per bit pattern,
  or `PartitionedTwirlSampler.sample(decision_bits=...)`); behaviour with bits supplied is
  unchanged, and `INPUT_BITS` that drive no IF are not refused. (`sample_barrier` on the
  partition sampler, `compile_exact_branches`, `compile_reference` and the `state` CLI verbs
  joined the refusal in the entries below.)

- **`PartitionedTwirlSampler.sample_barrier` (and any other sampling delegate) now refuses by
  name instead of silently sampling the all-zeros driving pattern.** `__getattr__` forwarded it
  to the reference (all-zeros) inner sampler, so every IF body counted as "not taken" — reachable
  from adaptq's `choi.harness_port_ref` on a protocol with a `Conditional` fix-up (the port
  reference of the S-conditional T gate was pattern-0-only, no warning). The IR has no pattern that
  is correct on its own: an `IF` condition reads `decision[cond_port][cond_bit]` of an
  `INPUT_BITS` port (circuit_ir.hpp; the parser refuses any other name), never one of the
  circuit's own records, so the wrapper — which exists only when an IF drives a bit — cannot pick a
  pattern without the caller. The refusal names the IF conditions, the driving-bit count and the
  escapes; new `sampler_for_pattern(bits)` returns the (cached) inner `_xtim.TwirlSampler` of one
  full decision-bit vector, whose `sample_barrier` is exact for shots with those bits (bytes
  identical to compiling `resolve_branches_text(text, bits)` directly; read pattern-dependent
  metadata such as `output_wires()` from THAT sampler). Metadata delegation is an allow-list
  (`num_*`, `output_wires`, `channel_report`, `channel_one_counts`, `plan_structure`, `born_dec`,
  `sig_bits` — the all-zeros pattern's compiled circuit); any other inner attribute is denied by name.

- **`PartitionedTwirlSampler.sample(decision_bits=...)` on a detector-free IF circuit** (one whose
  only channels are DECISIONs / observables) raised `ValueError: shape mismatch (n,0) -> (n,1)`:
  the non-mask scatter buffer was padded to one byte. It is now exactly `ceil(num_detectors/8)`
  wide, so the returned `dets` is the engine's `(n, 0)` and the group-wise obs/decision bytes equal
  the per-pattern engine's.

- **`compile_exact_branches`, `compile_reference` and `python -m xtim state compile|verify`
  refuse an unresolved IF by name** (same message family as the sampling surfaces). They ran
  every IF body unconditionally: the exact-branch compiler and the reference compiler ignore the
  IfBegin/IfEnd markers. Bits-supplied use (resolve the branches, then compile the concrete text)
  is unchanged; the `python -m xtim sample|detect|analyze_errors|diagnose` verbs already refused
  through `xtim.Circuit`. `compile_twirl_sampler_from_state(state, text)` refuses too (it compiled
  the raw text through the engine: every IF body taken). Still unguarded, raw-binding surfaces:
  `_xtim.ExactBranchProgram(text)` / `ExactBranchSimulator(text)` from `xtim.forced` and
  `_xtim.ref_compile_text(text)` (exact-branch bindings and `xtim/forced.py` are being edited on
  another branch).

- **A zero-probability noise instruction is transparent to a pending Clifford frame for every
  ChannelSupport consumer (the twirl record samplers incl. `sample_barrier` / `xtim.port`, and the
  DEM export).** `H 1; Z_ERROR(0) 1; H 1` after a magic gate flushed an explicit H pair (a Noise
  instruction was an unconditional frame boundary in `eliminate_hadamards`), which turned a p>0
  atom's residual off-diagonal mid-stream: CH/PPR routing, `sample_barrier` refused ("state
  retention requires a diagonal-class circuit"; barrier-fuzz seed 366 of the z-channel-atoms work)
  and the DEM refused "outside the simulable class at deferred-stream gate index 4". Derived from the
  atom contract (`propagation_table.hpp` AtomSet::ChannelSupport: a factor of probability zero is
  never fired, so its atom is ABSENT): the instruction is nothing to these consumers, so
  `eliminate_hadamards(deferred, zero_noise_transparent=true)` (new `NormalizePolicy` flag, default
  false) lets the frame pass it; the instruction stays in place (noise sites and RNG draws
  unchanged). Taken unconditionally by the TwirlSampler constructors, `_bare_state_of` /
  `_input_state_of` (xtim_py.cpp `normalize_for_twirl_sampler`) and by the DEM's normalization entry
  (`dem_export.cpp` `dem_normalize`, both of its `normalize` calls). NOT taken, by contract or by
  consistency: the forced-fault compiler (AtomSet::AnyPauli — ANY Pauli may be forced at a p=0 site,
  so the site is a real position; a forced letter there stays at the user's position, test), the
  exact sampler with its reference compile and `deferred_signature` cache key (one flushed trio),
  and the DEM's `detector_determinism` / `expectation_frame_check` diagnostics (they mirror the
  exact sampler's frame). DECLARED OUTPUT MOVEMENTS (ChannelSupport consumers only; every moved
  stream's law equals the deleted-line twin's and the dense oracle): (i) `sample_barrier` and the
  DEM on p=0-under-frame CH/PPR circuits: refusal -> exact result; (ii) `channel_report()["plans"
  / "ppr_plans"]` on circuits the flushed frame routed to PPR and the transparent one routes
  diagonal (their sample streams were equal on every corpus circuit, but that is a measurement, not
  a stated invariant); (iii) Born-decision / barrier BYTES of a base-accepted circuit whose pending
  frame folds across a p=0 line into a measurement (`RX 0; DEPOLARIZE1(0) 0; ...; MY 0; DECISION`):
  the deferred read form changes (`H; N0; MY` -> a folded read), the law does not (1 of 400 fuzz
  circuits, barfuzz/197). Twins with the p=0 line deleted are byte-identical to the new output.

- **The "bare state rejected: non-commuting magic: X15 and X14Z15 anticommute" refusal names the
  user's qubits.** The engine text names deferred-space wires (fresh wires the coherentizer and the
  deferral allocate above the user's qubits). The two anticommuting terms are now carried
  structurally (`PauliRotationForm::reject_terms` / `BareState::reject_terms`), every deferred
  qubit's provenance is recorded by the deferral (`DeferralMap::wire_origin`: the wire's first
  incarnation, a measurement-copy ancilla with its record index, a retired wire, a reset), and the
  message appends `; in user qubits: X on q3 (deferred wire 15) and X on q2 (deferred wire 14) *
  Z on q3 (deferred wire 15); the non-Clifford gates on these qubits, as parsed (...): T 3 (#15),
  T 3 (#18)` — a coherentizer copy ancilla is named "copy of qN (coherentized-feedback ancilla)", a
  held readout-flip instance "readout-flip ancilla of qN's measurement #k (coherentized M(p))", and
  an MPP/SPP desugar wire "desugar ancilla wire N (MPP/SPP gadget)" (new
  `ParsedStim::num_user_qubits` = 1 + the largest qubit index the text references). The original
  text is kept as the prefix.

- `compile_twirl_sampler` (`decisions=True`) on a CH/PPR circuit with a Born-class DECISION now
  refuses AT COMPILE with a message naming the limit (`ValueError: ... Born-class DECISION on a
  CH/PPR circuit ... unsupported by the twirl engine`, setup code 6) instead of
  `RuntimeError: twirl sampler oracle abort (details on stderr)` at the first `sample()`.

- `compile_twirl_sampler_from_state` (no-port path) now wraps a DECISION circuit's sampler like
  `compile_twirl_sampler` (a `TwirlBatch` with `.decisions`, not a raw 3-tuple).
- The `[twirl_records] circuit channels: ...` stderr diagnostic is now OPT-IN
  (`XTIM_TWIRL_VERBOSE=1`, documented in the README); it was printed on every twirl/auto compile
  unless `XTIM_QUIET` was set.

- **Pure-Z / pure-X noise no longer rejects a circuit through an atom it never fires.** The
  propagation table built BOTH the X and the Z atom of every noise location, so a `Z_ERROR`,
  `PAULI_CHANNEL_1(0,0,p)` or Z-only `PAULI_CHANNEL_2` before a T/CS/CCZ also propagated its
  probability-zero X atom. When that atom left the diagonal class, the circuit was rejected:
  `RX 0; Z_ERROR(0.1) 0; T 0; H 0; OUTPUT_QUBITS out 0` was refused by `sample_barrier`
  ("state retention requires a diagonal-class circuit"), while the same channel AFTER the T was
  accepted. DEM export and the feedback plan (neither has a PPR retry) refused for the same
  reason. Now a location's atom set is the channel's support. The X atom of a qubit is built iff
  some nonzero-probability Pauli has an X or Y factor there; the Z atom iff one has a Z or Y
  factor. This covers X/Y/Z_ERROR, DEPOLARIZE1/2, and PAULI_CHANNEL_1/2 with explicit zero entries
  (per pair slot, first/second qubit, unioned over a qubit repeated in one instruction). A
  zero-probability channel builds no atom. Every atom that IS built is unchanged.
  Every consumer reads atoms through `LocationEntry::atom()`, which refuses an absent atom by
  name, never by indexing it. The consumers are `compose_fired`, the framed sampler, the twirl
  sampler, the feedback plan and `exact_branches`. The framed and twirl samplers skip a
  zero-probability `X/Y/Z_ERROR`, as they already skipped zero DEPOLARIZE / PAULI_CHANNEL
  alternatives (no RNG draw either way). `dem_export.cpp` is untouched: it reads atoms only for
  alternatives with nonzero probability.
  - **Forced faults keep their contract** (any Pauli at any site; the probability is
    irrelevant). The forced-fault program builds its table with `AtomSet::AnyPauli`: every atom
    is built as before. A zero-probability atom that leaves the class no longer rejects the
    circuit; it is marked absent, and forcing it is refused by name at the user's (site, target,
    Pauli) address. Example: `Z_ERROR(0) 0; CS_DAG 1 0; SQRT_X_DAG 0; T_DAG 0; X 0; T 0`, where
    forcing X or Y is refused and forcing Z is answered. This exception is documented in
    docs/xtim_forced_faults.md and xtim/forced.py. Every forced law on a
    circuit accepted at a80dd33 is byte-identical.
  - **Changed outputs, all on circuits whose only out-of-diagonal atoms had probability zero.**
    (a) Newly accepted: `sample_barrier`, DEM export, `dem_column_layout`, and feedback-plan
    programs. (b) Re-routed: exact-sampler streams, when such an atom had been accepted as a
    general PPR residual. The engine now takes the diagonal path, so the streams differ by seed
    but have the same law.
  - **Verification.** Byte identity against a80dd33 (DEM default/approximate × with/without
    expectations, layout, exact expectations, det exact/auto/twirl, raw records, forced laws):
    9944/9944 fields on the bundled examples, the speed-audit/blow-up corpus and the rule
    examples. On 50 random noisy Clifford+T circuits and 30 review repros, 2794 fields are
    identical and every diff is in class (a) or (b), or is a still-refused circuit whose refusal
    now names its next genuine reason. Dense oracles: 400 random circuits (exact/auto/twirl and raw record laws,
    every single-channel DEM exact to 1e-12, and 13,509 forced single faults vs
    `_forced_oracle`). 400 `sample_barrier` circuits: 44 newly accepted, all exact; 2,996 forced
    laws exact. Tests: `tests/test_channel_support_atoms.py`, ctest `channel_support_atoms`.

- **Retained states accept a Z-type rec-feedback driven by an X/Y read of a fault-dressed
  wire; the X/Y-type remainder is refused by name.** The TwirlSampler family coherentizes every
  `CX/CY/CZ rec[-k] q`. A feedback driven by an X/Y read was emitted AFTER its source
  measurement, so deferral copied the control out through an `H` that then sat in the bulk of
  the control wire; a fault before a T/CS on that wire leaves a Pauli-times-diagonal residual
  (`T X T^dag ~ X S^dag`) which `H` takes out of the diagonal class. `sample()` survived on the
  PPR route, but `sample_barrier` (`xtim.port` groups / residual_law, adaptq's segment harness)
  refused with `state retention / Born decisions unsupported for CH/PPR circuits` — e.g.
  `RX 0; DEPOLARIZE1(p) 0; T 0; RX 1; CZ 0 1; MX 0; CZ rec[-1] 1` — and so did `sample()` as soon
  as the circuit carried a random-record `DECISION`. New `NormalizePolicy::
  hoist_feedback_on_class_exit` (set by `normalize_for_twirl` only): when the default deferred
  form leaves the diagonal class, feedbacks driven in place are emitted just BEFORE their source
  measurement (exact: the controlled-Pauli is diagonal in the measured basis, so it commutes
  with the measurement; which feedbacks may move is the deferred-measurement theorem below),
  written with the basis rotation on the TARGET
  (`CX t c` for `CZ rec` from an X read). The control then ends at its own measurement, as in
  the feedback-free circuit. The hoisted form is adopted only if it is in class. Circuits whose
  default form is in class, that have nothing to hoist, or in which no fault precedes a
  non-Clifford gate are byte- and cost-identical (no extra table build). An X/Y-type feedback
  (`CX/CY rec`) from an X/Y read of such a wire onto a RETAINED target stays out of class
  whatever its decomposition (the gate conjugates the residual's `S` into a rotation about
  `Z_c*P_t`); `sample_barrier`/`sample` now append the reason to the refusal
  (`explain_feedback_class_exit`). `channel_report()["feedback_hoisted"]` counts hoisted
  feedbacks. Oracles: exact dense record law (`tests/_record_oracle.py`) on 10 random circuits
  with feedback after T/T_DAG/CS/CCZ, noise and noisy readout, plus the deferred-measurement
  rewrite through the exact engine; a per-shot dense state conditioned on the shot's records for
  every retained output Pauli (`tests/test_feedback_after_t.py`). Changed outputs: only circuits
  that were refused (or, for `sample()` without Born decisions, routed through PPR) before.

- **Feedback hoist: the window rule is the deferred-measurement theorem (review I-1, I-2,
  M-1, M-4).** The move of a feedback to its source measurement was allowed only when nothing
  in between touched the target or the control (and never for a reused / copy-driven
  control). It is now decided per feedback by conjugating the feedback Pauli P back through
  its window W: the feedback moves as the b-controlled-(W^dag P W) iff that stays a Pauli
  that commutes with every measurement, reset and PAULI_EXPECTATION it passes and with the
  source basis on the control (Cliffords conjugate; T/CS/CCZ pass iff it is Z-type there;
  Pauli noise and other `rec` feedback are transparent; a feedback onto a wire nothing
  observes again is the identity). Theorem + proof sketch in `coherentize_feedback.cpp` and
  `docs/twirl_record_sampler.md`. Newly accepted on `sample_barrier` (all exact against a
  dense branch oracle, `tests/_branch_oracle.py`): MR/MRX/MRY sources (p = 0 and p > 0),
  Z_ERROR / DEPOLARIZE2 on the target and X/Z_ERROR on the control in the window, an MPP or a
  Clifford acting on the target in the window (by conjugation), commuting
  PAULI_EXPECTATIONs, feedbacks onto measured wires. A noisy-readout control observed again
  keeps its held-flip undo around the moved gate. Refusals now name the true cause: the
  blocking window instruction ("NOT MOVABLE ... its window contains a non-Clifford T on qubit
  2 ..."), or, with every feedback moved, the representation limit ("an (I + iX)-type rotation
  on a kept qubit, outside the retained engine's Pauli x diagonal-Clifford representation"),
  no longer "whatever its gate decomposition". Changed outputs: only circuits refused before;
  base-accepted circuits are byte-identical. Re-review fixes: the refusal now blames only
  feedbacks that leave the class ON THEIR OWN (class membership with every other feedback
  removed), named at user level (target qubit, source record; MPP sources as "an MPP/MPAD
  product measurement", never an internal wire — `ParsedStim::num_user_qubits`); the copy
  guard for a not-moved in-place feedback whose control a moved gate touches is documented as
  that feedback's deferred-measurement precondition and covered by oracle tests (disabled ->
  accepted and wrong); the rule is stated as sufficient, not "iff"; a refusal with no single
  cause names a minimal jointly-fatal set and the wires its moved gates share.

- **DEM export judges PAULI_EXPECTATION columns PER SHOT** (silent-wrong exports -> refusals;
  false refusals / post-select flags -> exact exports). The sampler reports a column per shot
  (<P> conditioned on every measurement outcome; declared frame records only fix the sign), but
  DEM export classified each fault's action on a column (sign flip -> L-column mechanism,
  magnitude change -> refuse / flag) from the READ-AVERAGED <D^dag P' D> on the bare state. With
  a read that is neither a detector nor a frame record correlated with the column the two
  disagreed (adversarial DEM audit 2026-10-11): `RX 0; RX 1; T 1; Y_ERROR(0.1) 1; CS 1 0; MX 1;
  PAULI_EXPECTATION(0) X0` (per shot 0.5 -> +-0.7071, no detector) exported with no mechanism;
  a zero column becoming +-1 per shot exported as `error(0.1) L0`; zero columns that stay 0,
  columns whose per-shot sign is a fair coin, and detector-correlated sign laws were refused or
  flagged. Now every alternative that acts non-Pauli on a column or randomizes a read is judged
  from the column's EXACT per-shot law (new `dem_pe_per_shot.{hpp,cpp}`): in frame coordinates
  v(m) = (-1)^{m.S0} h(y), with the coin parity S0 and the magic-correlated parities y obtained
  from ONE elimination of the noiseless reads plus a |J|-dimensional update for the fault's
  changed reads (no Born sum: a column whose value cannot depend on any read is decided by this
  GF(2) structure alone), and h read on the SUPPORT of y by the engine's own exact branch walk
  (`born_p1` / `measure_pauli`, early exit as soon as the verdict is known, memoized joint laws).
  The walk is budgeted (4096 branches / a deterministic work bound, ~0.5 s): past it the export
  refuses by name, fast (`include_expectations=False` reports the column in
  `dropped_expectation_columns` instead of refusing). Magnitude change on some branch -> the existing flag / non-FT path (its accepted-branch
  test now per shot too); a per-shot random sign with |value| kept -> emitted through the exact
  joint law with the randomized targets (Stim's `approximate_disjoint_errors` rule when that
  law is not a product); otherwise the deterministic per-shot flip. Pauli-only alternatives
  with no randomized read take the unchanged path.
  **Columns with no DEM concept (owner rule):** a PAULI_EXPECTATION whose NOISELESS per-shot
  value varies shot by shot (it depends on records not declared as its frame) is not a DEM
  column. It is dropped — no `L` line, no mechanism, no flag and no refusal on its account (its
  `L` index stays reserved, so `dem_column_layout` is unchanged) — and reported as
  `DemWithReject.dropped_expectation_columns` / the export dict's `dropped_expectation_columns`
  (`(declaration index, reason)`; when one exists the reason names a SUFFICIENT frame — the full
  record set to declare, replacing the current one — verified on the 468 such columns of the audit
  campaign), plus a
  once-per-circuit `xtim.XtimDemWarning`; the rest of the DEM exports normally.
  **`cultivation_d3_faithful`'s former reject region (177 flagged faults over all 20 detectors)
  was an artifact of the read-averaged classification:** judged per shot, every one of those
  faults keeps every column's magnitude (a detector-correlated sign law) and is an ordinary
  mechanism. `examples/dem_reject_region.py` and `docs/xtim_dem_reject.md` now use a circuit with
  a genuine per-shot magnitude-changing, detected fault. DEM text, `postselect_faults` and
  `dem_column_layout` are byte-identical to the build this change was made on (3.1.11 + the
  DEM-refusal-message, forced-provenance and diagonal-fallback entries of this release) on every
  bundled example and the blow-up /
  rule corpora except `cultivation_d3_faithful` and `miniature_oracle`, whose 177 / 2 flagged
  faults are detector-correlated per-shot sign laws, now exported (`cultivation_d3_faithful`
  therefore needs `approximate_disjoint_errors=True`, like `cultivation_d5`). Sampling is
  unchanged. Regression tests: `tests/test_dem_pe_per_shot.py` (verdicts derived from
  `compile_exact_branches` in the test).

- **Wrong hint on DEM refusals.** The "dresses a declared PAULI_EXPECTATION column beyond a
  sign ... `include_expectations=False`" hint was the catch-all for any unrecognized refusal,
  so it was shown for the disjoint-channel refusal of circuits with no PAULI_EXPECTATION
  lines (bundled `cultivation_d5`; merlin bt27 / cult_d3 / cult_d5), where
  `include_expectations=False` does not help. It is now emitted only for the engine refusal
  of a fault acting non-Pauli on a declared expectation column; the disjoint-channel refusal
  gets the `approximate_disjoint_errors` hint; any other unrecognized refusal gets no hint.
  DEM text is unchanged for every circuit that exported before.

- **Forced faults through coherentized feedback: provenance addressing** (refusal -> exact
  answer). `compile_exact_branches()` matched user noise sites to the normalized circuit's noise
  instructions 1:1 BY ORDER, so a feedback `CX rec[-k] q` coherentized across a non-Clifford gate
  whose source is `M(p)` (its readout flip folded in as an extra noise instruction) made EVERY
  noise site refused (`noise_refusal`, e.g. bundled `cultivation_d5`: 187 vs 180), and every
  record flip was refused whenever any feedback was coherentized (`record_refusal`). Now every
  noise instruction carries a metadata-only provenance tag (`Instr::origin`: user site i, or the
  readout-flip fold of record m, set by the coherentizer) through normalization; sites map by tag,
  and a record flip of a folded measurement fires the fold's Pauli (recorded bit AND coherent
  control flip together; the held fold's undo restores the control state — exactly the fold's
  model of `M(p)`). New `ExactBranchSimulator.folded_records` / `refused_records`. Remaining
  named refusal, per record: a readout-NOISELESS measurement read by coherentized feedback (no
  fold channel to force). Tags are read by no sampling/DEM/emit path: exact and auto sampled
  streams (fixed seed) and DEM text byte-identical to 3.1.11 on all 9 bundled examples and 19
  blow-up corpus circuits. Verified (tests/test_forced_provenance.py) against an inserted-circuit
  oracle — C with noise removed, the faults written in as Pauli gates, forced record flips as
  the `!` reported-bit inversion, evaluated by xtim's noiseless exact law: cultivation_d5 200
  single + 100 weight-2/3 fault sets (93 flipping folded records), max |diff| 0; a hand-built
  held-fold circuit (noisy M -> feedback across T, control re-measured) for every single fault,
  every record pair and 150 random sets, also vs the dense statevector oracle; mutation legs
  (record flipped without the control; fold plus record double flip; undo dropped) go red. The
  dense random suite now accepts coherentized-feedback circuits (2 x 3000 circuits, 152 folds).

- **DEM export rejected circuits the sampler accepts** ("outside the simulable class at
  deferred-stream gate index N" from `detector_error_model()` while `compile_sampler()` /
  `compile_detector_sampler()` accept and sample them). Both paths normalize identically; the
  difference is the propagation class. The sampler retries a rejected fault atom as a general
  PPR residual; DEM export deliberately does not (a general Clifford residual has no linear-DEM
  form) and so requires every fault to stay Pauli × diagonal Clifford. A fault before a T layer
  (residual X·S-type) left that class only because of how the deferred stream was WRITTEN:
  (a) a live-wire `MX`/`MY` was deferred as `H q; CX q a; H q`, putting a Hadamard on the
  magic-dressed data wire (the qRM15 → Steane code-switch readout with a coherent `CS_DAG`
  fix-up); (b) a user `H q; DEPOLARIZE1 q; M q` X-check readout, whose noise blocked the
  terminal `H;M -> MX` fold. DEM export (and `dem_column_layout`) now retry, ONLY when the
  default normalization rejects, with two exact rewrites: the diagonal-copy deferral of a
  live-wire X/Y read (fresh `|+>` ancilla, controlled-X/Y onto the wire, terminal X read —
  `defer_measurements(..., diagonal_copy=true)`, `NormalizePolicy::diagonal_copy`) and the
  noisy `H;M` fold (single-qubit Pauli noise conjugated by H, read in X). A circuit still
  outside the class is rejected with the default path's index, as before. Exact-or-refuse is
  unchanged: a fault whose law is not a product of independent mechanisms still refuses unless
  `approximate_disjoint_errors` is set. Verified per single fault against
  `compile_exact_branches` (the sampler's own exact law, PPR residuals included) with every other
  noise instruction kept in place at zero probability, on the E2/E7/E10 code-switch circuits and
  four new minimal cases (`tests/test_dem_diagonal_fallback.py`, also vs the `exact_sv`
  statevector); mutation-tested. Every other DEM (bundled examples, benchmark and blowup corpus,
  default and approximate, with and without expectations) and every sampled stream is
  byte-identical to 3.1.11.

### Added

- **The per-shot expectation-column analysis's deterministic work cap is an argument:
  `detector_error_model(..., pe_max_work=xtim.circuit.PE_MAX_WORK_DEFAULT)`** (also
  `detector_error_model_text`, `detector_error_model_with_reject`, `_xtim.export_dem_text`, C++
  `export_dem`). The single source of the default is C++ `PeShotAnalyzer::kWorkBudget` (1.5e7, the
  3.1.12 constant; `_xtim.PE_MAX_WORK_DEFAULT`), so every output is byte-identical by default; a
  refusal past it now names the argument (`...; raise pe_max_work on detector_error_model() to
  continue the exact analysis`; `inf` is printed for a lifted bound, never cast). It is a count
  (work units per branch walk), never wall-clock. The branch cap (`kLeafBudget`, 4096 support
  leaves per walk) stays fixed: on the reviewer's T-ladder it is never the binding limit (k=13 has
  d=12 magic-correlated parities, at most 4096 leaves; the work cap binds first at every cap tried
  up to 3e9). The ladder's cost grows ~7x per T: k=9 (refused by name since 3.1.12) exports at
  `pe_max_work=1e9` in ~1 s with a DEM equal to the exact per-shot law (dense oracle;
  `test_t_ladder_budget_names_argument_and_a_raised_budget_exports`), k=11 in ~50 s, k=13 needs
  about `1e11` work units (about an hour; the opt-in `XTIM_SLOW_TESTS=1` test). Documented in
  `docs/xtim_dem_reject.md`.

### Tests

- **Test-hygiene: `tests/test_forced_vs_sampler.py` no longer requires scipy.** It imported
  `from scipy.stats import binom` to compute a two-sided binomial-tail Bonferroni check; in a
  wheel env without scipy (3.1.11's release, where the wheel-env `pytest tests/` run had 5
  FAILED and was not gated — see below) 5 tests failed. Replaced with `binom_cdf`/`binom_sf`:
  an in-test log-pmf (`math.lgamma` for the binomial coefficient, logsumexp'd) — no new
  dependency. (First attempt used `math.comb` for an exact integer coefficient; that is a bigint
  whose cost grows with n, and summing O(n) of them for this file's n=3000-4000 shots hung for
  10+ minutes — switched to `lgamma`, O(1) per term, accurate well past the tolerance needed.)
  Proved equal to `scipy.stats.binom.cdf`/`.sf` on a grid of n/p/k (incl. p=0, p=1) in a new
  `test_binom_cdf_sf_match_scipy` (scipy optional there via `pytest.importorskip`; every other
  test in the file is now scipy-free). Audited the rest of `tests/` for module-level optional
  third-party imports (scipy, networkx, sympy, pandas, matplotlib, sklearn, torch, ...): none
  found — `stim`/`pymatching` are already behind `pytest.importorskip`, and the lazy `import
  stim` calls in `tests/_dense_oracle.py` are function-local, reached only from stim-gated tests.
  `RELEASING.md` step 6 (wheel-env `pytest`) now reads as an explicit GATE (must exit 0) instead
  of a plain instruction, since that's what let the 5 scipy failures ship unnoticed in 3.1.11.

### Performance

- **DEM export: the exact correlated-read law's Born sums are factored through the free
  register (`cultivation_d5` approximate DEM 2.2 s -> 0.6 s; DEM text byte-identical).** The
  law of a fault that randomizes reads (3.1.10) needs the coset weights nu = WHT(mu),
  mu[c] = <A_{S_c}> over the Gray-code products of the Fourier-support basis — up to 2^20 Born
  sums per distinct law, 65% of `cultivation_d5`'s DEM time. In frame coordinates the bare
  state is |chi>_free (x) |eps>_rest, and the Born kernel reads each A_S only through its X/Z
  bits on the `free` generators plus the eps-sign of its non-free Z part: `<A_S>` equals the
  SAME kernel on that free-register projection, bit for bit. `correlated_law_coset_weights`
  (`dem_export.hpp`) now evaluates one Born sum per DISTINCT projection, memoised across every
  law of the export (<= 8 * 4^{r_free} Born sums; one such sum per subset before), keeping the
  Gray walk's exact phase bookkeeping and the WHT. Every printed probability is reproduced to
  the byte (oracle: `cpp/tests/test_dem_factored_law.cpp`, the pre-factoring walk (fd4f31c) verbatim on hand-built
  and circuit-built states, r_free 0..4, Fourier dimension up to 12; DEM text of the bundled
  examples, `tests/data`, the blowup corpus, 302 adaptq-captured port circuits, 1200 random
  noisy Clifford+T circuits (r_free 0) and 150 code-block circuits with magic left in the bare
  state — [[8,3,2]] cube CCZ and [[15,1,3]] QRM T blocks, r_free 1..6, chi up to 64 — identical
  to the pre-factoring build fd4f31c; 27 of the latter are vendored as `tests/data/dem_factored_law/*.stim` and pinned by
  `tests/test_dem_factored_law_golden.py` against a golden produced by the fd4f31c build). The
  refusals (Fourier dimension > 20, support dimension > 16) and the per-law cache are unchanged.

- **The framed bare state is memoised process-wide (`cpp/src/bare_state_memo.cpp`): a DEM export
  after a compile, a second DEM, a compile after a DEM, `detector_determinism` /
  `expectation_frames` on the same circuit reuse one build instead of re-running FastTODD + the
  coset-family assembly + per-H canonicalisation (70 % of a bt81 DEM, 60 % of its compile).**
  Every consumer (exact sampler compile, twirl compile, DEM export, determinism, frame check,
  reference reconstruction) now calls `build_bare_state_framed`, which is the old builder
  (`build_bare_state_framed_uncached`) keyed on the build's EXACT read set — `n` plus the Gate
  instructions in stream order; the build reads nothing else (non-Gate instructions are skipped
  at `build_pauli_rotation_form`), pinned by mutation in `test_bare_state_memo` (noise channels /
  probabilities, Measure bases / inverts / flips, Resets, Observables, ports perturbed: key and
  build unchanged; any gate perturbation re-keys). A hit re-materialises the captured
  `FramedSuperposition` exactly as `from_css` does, so every stream and DEM text is byte-identical
  with the memo on or off (`XTIM_BARE_MEMO=verify` rebuilds and compares field-for-field, bit-level
  coefficients, on every hit and raises on a difference; `tests/test_bare_state_memo.py` runs the
  bundled examples through every path in that mode). Bounded LRU by count AND bytes
  (`XTIM_BARE_MEMO_ENTRIES`, default 8; `XTIM_BARE_MEMO_MB`, default 256; `XTIM_BARE_MEMO_ENTRY_MB`,
  default 16: a state over it — chi = 2^20 BH, ~97 MB — is never captured, so a large-chi cold
  DEM costs exactly the build), mutex-guarded, and held OUTSIDE the state budget: `live_state_bytes()` and every
  capacity refusal are exactly what they were. `XTIM_BARE_MEMO=0` disables it; `=trace` logs
  each lookup. `_xtim.bare_state_memo_stats() / _clear() / _reset_stats() / _set_caps()` expose it.
  The twirl compile previously built the CSS state and converted it itself; it now takes the same
  framed builder (the documented byte-identical equivalent).

- **Faster bare-state build (compile and DEM export), byte-identical.** The two top hotspots
  on large circuits (bt81 / merlin Switch-5) are now word-parallel:
  `restrict_to_support` XORs R-rows over the set bits of each Z-mask instead of a scalar
  O(n·d) bit loop; `AffineState::apply_h` skips `canonicalize()` when the old R-row is zero,
  and the nullity-1 kernel vector is read off the rank sieve's echelon rows by
  back-substitution instead of an augmented Gauss–Jordan. Sampler streams, DEM text, exact
  expectations and bare states are unchanged on the bundled examples and the speed-audit
  circuits. bt81 compile 1.9-2.1 s -> 1.3 s and approximate DEM
  1.9-2.0 s -> 1.3 s (median of 5, interleaved A/B, two runs); smaller circuits unchanged
  within noise.

## [3.1.11] - 2026-10-10

### Added

- **Forced faults, exact branches** (`Circuit.compile_exact_branches()` ->
  `ExactBranchSimulator.exact_branches(faults, expectations=None)`, C++
  `sample_program_exact_branches` / `compile_exact_branch_program`, binding
  `_xtim.ExactBranchProgram`). Compile once; per query, every measurement-outcome branch of
  nonzero probability under exactly the given faults, with its exact probability, recorded
  bits, reference-relative detectors/observables, raw DECISION parities and the
  PAULI_EXPECTATION values on that branch. Faults address the circuit's noise sites
  `(site, target, pauli)` and record flips `{"record": j}` — the sampler's own fired events, so a
  query is the sampled law conditioned on that fired set. Covers every circuit the exact sampler
  accepts (magic anywhere, mid-circuit measurements after T, Pauli feedback, CH-class atoms);
  refuses by name noise sites behind coherentized readout-flip noise and record flips feeding
  coherentized feedback. Read-only: no RNG, no change to any sampled stream (byte table and DEM
  corpus identical to 3.1.10). See docs/xtim_forced_faults.md.
  Verified vs an independent dense statevector (4000 random circuits with T/CS/CCZ,
  mid-circuit measurements, MR, `!`/M(p), feedback; 16,000 queries, ~72.7k branches:
  probabilities to 1e-12, max expectation error 1.1e-15), vs the exact sampler with the faults
  written in, and vs Stim on Clifford memories; mutation-tested (address off-by-one,
  alternative swaps, dropped fault, probability factor / normalization, dropped sign, dropped
  record flip: all red). About 0.24 ms per query on cultivation_d3_faithful (weight 1-3,
  loaded host).

## [3.1.10] - 2026-10-10

### Fixed

#### DEM export

- **DEM export dropped gate faults that act on a `PAULI_EXPECTATION` column through its declared
  frame records** (`dem_export.cpp`). The column the sampler reports is (-1)^{frame records} <P>,
  but the gate-noise path applied the sign / magnitude / completeness rules to the declared Pauli P
  alone. `R 0 1 / H 0 / CX 0 1 / Z_ERROR(0.2) 0 / MX 0 / PAULI_EXPECTATION(0) X1 rec[-1]` exported
  `logical_observable L0` while the sampler flips the column in 20% of shots (its readout twin
  `MX(0.2) 0` was already exported, 3.1.4). In deferred space a frame record is the terminal read
  R_j of its own wire, so every rule (and the noiseless value beta0) now uses the frame-folded
  operator P' = P . prod_{j in frame} R_j. A non-Pauli (T-residual) fault on a frame read is
  classified like one on P: the Bell-pair T variant (`X_ERROR` before `T 0`, `MX 0`, frame
  `rec[-1]`), whose fault takes |<X1>| from 1 to 0 with no detector, was neither emitted nor
  refused and now refuses ("non-fault-tolerant circuit"). OBSERVABLE_INCLUDE and DECISION columns
  already folded gate faults through their records; DEM export refuses residual feedback (tested).
- **Meaning of an expectation L-flip, made explicit and tested**: the mechanism multiplies
  P' = P . (frame-record Paulis) by -1, i.e. multiplies the reported <P> by -1 (records relabelled by
  the fault's own record flips) — defined for any noiseless value (0, 1/sqrt2, branch-dependent);
  such columns are exported, not refused. A Clifford-residual fault (action on P' not +-1) on a
  ZERO-valued column was emitted as "no flip" (the classifier read the sign of a 0 average); it is
  now flagged (fires a detector) or refused (fires none). Columns the residual does not touch take
  the operator rule. Oracles: stim Pauli-frame propagation + per-shot exact-engine values
  (v = (-1)^s v0(records XOR f)) on Bell+T columns valued 1/sqrt2, 1, 0 and a teleported T state
  with and without frames. No DEM in the corpus changes.
- **Faults that randomize terminal X/Y reads: exact correlated law (silent DEM errors fixed)**
  (`dem_export.cpp`). Until 3.1.9 each read randomized by a fault's S/CZ residual was an
  INDEPENDENT fair coin. The flips of one fault are correlated in general (two residuals can
  cancel on the state, a CZ fixes a parity), so the exported law was wrong: refused as "negative
  probability", or — when the factoring happened to succeed — SILENTLY WRONG. Found by an exact
  per-fault sweep (exact engine, every Pauli alternative): `cultivation_d3_faithful` (a bundled
  example) carried 15 wrong channels / 212 spurious mechanisms (e.g. a DEPOLARIZE2 X(x)X whose
  residuals cancel, never firing a detector, spread over 32 patterns): its DEM drops from 319 to
  107 edges, all matching the exact engine; random magic circuits (Bell/GHZ blocks with T(x)T^dag)
  exported spurious `error(p/2)` mechanisms and wrong correlations. The targets fed by randomized
  reads now get their exact joint law from the bare state (Fourier transform <prod A_t>, A_t =
  s_t E^dag D_t E; the stabilizer-twirled Pauli channel of the residual), expanded over its
  support (dimension <= #randomized reads).
- **Exact per-fault oracle** (`tests/exact_sv.py`, `tests/test_dem_exact_oracle.py`): an independent
  branching statevector simulator (live qubits only, every outcome branch with its exact probability
  and PAULI_EXPECTATION values) checks every Pauli alternative and channel of an export to 1e-12.
  Clean on the bundled examples it can hold (cultivation_d3_faithful/rate, miniature_oracle,
  distillation_15_1_3, cube_ccz), the Clifford corpus, the stim Clifford set and 1,200 random
  Bell/GHZ + T(x)T^dag circuits; it rejects 3.1.9 on 6 of 240 random circuits.
- **DEM export with correlated reads is faster** (text-identical): products in frame coordinates
  (one Born sum per subset, no conjugation), the noiseless target values and the per-residual law
  memoised per export, and `conjugate_through` walks the CZ rows of the Pauli's X-support instead of
  all pairs. `approximate_disjoint_errors=True` export of cultivation_d5: 31 s -> 3.5 s (loaded host).
- **`approximate_disjoint_errors`** (`detector_error_model*`, `export_dem_text`), stim's
  semantics: a channel with no exact conversion into independent mechanisms refuses by default;
  `True` approximates it (each merged signature an independent mechanism with its disjoint
  probability, exact to first order), a float threshold approximates only if every probability
  argument is <= it. `code_switching_faithful`, `cultivation_d5`, merlin cult_d3/d5 and bt27/bt81
  export with it (they refused in 3.1.9). **xtim converts exactly in every case stim does, and
  additionally whenever the channel's distribution over DEM signatures (not over Paulis) factors into
  independent mechanisms** — e.g. PAULI_CHANNEL_1(0.1,0.1,0) before a Z read (X and Y share a
  signature: one mechanism error(0.2)) and generic PAULI_CHANNEL_2 channels, which stim refuses
  without (or approximates with) approximate_disjoint_errors; each such conversion reproduces the
  disjoint signature distribution exactly (tested). xtim refuses only where no exact independent
  factoring exists, unless approximate_disjoint_errors is set. Where stim exports, xtim's DEM is
  semantically equal (tested on 165 Clifford case x option combinations: 108 equal, 18 both refuse,
  39 xtim-exact-superset, 0 where xtim refuses and stim exports).
- **The completeness gate's accepted-branch check is exact** (`dem_export.cpp`). For a fault that
  randomizes detectors without flipping any deterministically, 3.1.9 collapsed every randomized
  read's qubit in the Z basis (randomized reads are X/Y reads) and re-classified; with frame-folded
  columns that also collapsed frame reads and reported harmless faults as non-fault-tolerant. The
  accepted-branch value is now computed from the Pauli algebra, Sum_S <Q A_S> / Sum_S <A_S> over the
  subsets S of the randomized detectors (A_d = s_d E^dag D_d E), and the fault is refused when the
  column differs there in magnitude OR sign (an undetectable logical flip on the accepted branch);
  more than 16 randomized detectors refuse without a verdict.
- DEM text changes (DEM corpus, 30 circuits): only `example_miniature_oracle` (3.1.9 refused it,
  "negative probability", after mis-modelling its teleport frame read as an independent coin and
  missing the frame-carried flips on five channels; it now exports, every Pauli alternative of
  every channel verified against the exact engine). Every sampled stream is byte-identical.

#### Sampling: front end and reference convention (audit A, E, F/F2, K)

- **Swapped measurement records when an `H` folded into a later measurement (audit A).**
  `eliminate_hadamards` rewrote `H q; ...; M q` into `MX q` but emitted the Measure at the H's
  position, so a Measure on another wire lying between them was overtaken and the two records
  swapped — on the exact engine, `compile_sampler`, barrier DECISIONs and `xtim.port`, and in every
  detector / observable / expectation pairing built on them. The H only reached that fold when the
  frame pass had been flushed, which every `PAULI_EXPECTATION` does (~98% of the audit's flags).
  The Measure now keeps its own position. (Also fixes audit E: the twirl classifier saw the
  swapped records and called a random detector deterministic.)
- **Inverted targets with MPP/MZZ/MXX/MYY/MPAD (audit F/F2).** A qubit referenced only as `!q`
  (or as a `!`-prefixed product factor) did not raise the desugar-ancilla base, so the ancilla
  aliased that qubit: wrong MPP records on every engine, or a spurious "outside the simulable
  class" rejection of a pure Clifford circuit. Any future user-qubit/ancilla collision is now a
  loud parse error.
- **One reference for random channels on every engine (audit K).** Detector/observable channels
  are reported relative to Stim's `reference_sample()` convention (noise off, every collapse biased
  toward +1 in record order, feedback applied), via one shared primitive
  (`framed_reference_records`). The exact engine's u=0 batch reference disagreed with Stim on
  random observables after an `eliminate_hadamards` sign stamp (`RX 0; MY 0`) or a magic-coupled
  out-of-order collapse; the twirl engine's Born observable reported the ABSOLUTE parity, so
  `engine="auto"` changed the observable bit's meaning with the engine. Deterministic channels are
  unchanged. Known remaining divergence from Stim: the hidden collapse of a plain `R`/`RX`/`RY` on
  an entangled qubit (Stim biases it; deferral marginalizes it).

#### Sampling: exact engine (audit B, C, D, G, H, L1)

- **PPR rotation sense for sign-carrying axes (audit B, H, C)** (`ppr_residual.cpp`,
  `PprResidual::clifford_tableau`). An axis is stored as `i^phase X^x Z^z`, phase in {0,1}; with
  #Y = 2 or 3 (mod 4) the stored axis is the NEGATIVE of its letter string (phase 0 on X0Z0X1Z1
  is -Y0Y1), but the diagonalizer applied S^dag^e regardless of that sign, so those axes rotated
  the wrong way. The error tableau of a fired off-diagonal noise atom (CH, or a conjugated CS)
  then carried an extra factor A, flipping every read anticommuting with A (impossible records
  under forced noise; B's X-atom tableau had all four rows negated). The exponent is now negated
  when `i^(phase-#Y) = -1`. Checked against dense exp(i pi/4 e A) for every n=3 axis and exponent.
- **Case-B collapse labels relative to the new reference (audit G, D)**
  (`framed_superposition.cpp`, `framed_measure_anticommuting_general`, used by `batch_measure`,
  the TreePlan and the reference read). The post-state is built from the projected sigma=0
  state R0, but the new frame reference is survivor 0's syndrome; when survivor 0 was not the
  sigma=0 branch, branch labels and signs were off by a relative Pauli frame D_{x0} (G: 224 of
  415 nonzero expectations with the wrong sign after an X1 collapse). Labels are now x_g XOR x0
  and the sign parity is taken against eps XOR x0; x0 = 0 runs the old instructions unchanged
  (no benchmark circuit hits x0 != 0, so no stream changes).
- **Odd-Y conjugated Paulis are Hermitian (audit L1)** (`twirl_sampler.cpp` `exact_ppr_shot`,
  and the exact engine's Debug-only PE assert in `framed_sampler.cpp`). The guard tested
  `phase & 1`; a Hermitian `i^phase X^x Z^z` has phase = #Y (mod 2). Every pulled-back generator
  with an odd number of Y factors threw, so `engine="auto"` raised on an accepted circuit.

#### Sampling: twirl sampler / auto / barrier / port (audit I, J, M, SEG; campaign seeds 8022, 133, 103021, 911643/911649)

- **The Born observable dropped its deterministic record invert on every noisy shot**
  (`cpp/src/twirl_sampler.cpp`, audit I/J). The invert of the observable's records (Stim `!`,
  `MPAD 1`, the H-elimination's folded sign) was folded into the clean-shot weight only; the
  diagonal and PPR channel laws (`plan.obs_m`) and the exact-path measurement ignored it. Any
  shot with a fired noise atom therefore reported the observable FLIPPED: the
  (detector, observable) joint was wrong (`X_ERROR(0.2) 0; M 0; …; MX 0; MY 1` gave
  (1,0)/(1,1) = 0.030/0.173 against 0.171/0.029), and noise anywhere — even after the last
  measurement — leaked into an observable containing an `MPAD 1` (0.780 against 0.854). The
  invert is now XORed in at every non-clean emission site. **Who is affected:** `engine="twirl"`
  / `"auto"` observables and `compile_twirl_sampler` born observables whose records carry an
  odd number of inverts, on every release with the born observable (2.3.0 through 3.1.9).
- **A Born DECISION (and the retained state) was collapsed by certified generators no record
  reads** (audit M). On a plan with kernel directions (κ > 0) the born-decision memo replayed the
  record path's kernel chain, which measures EVERY kernel logical on the bare state; the exact
  per-shot path measured every certified generator. Neither is a physical event when no
  DETECTOR / OBSERVABLE / DECISION reads that generator, and a decision anticommuting with it was
  randomised (`X_ERROR(p) 2; CS_DAG 4 3; CCZ 4 3 2; MX 4; DECISION(0) rec[-1]`: P(0) = 0.500
  against 0.677 at p = 0.5). The state handed to Born decisions and to `sample_barrier`'s
  retained-state consumers is now R|bare> projected onto the shot's record channels at their
  recorded signs (`project_onto_records_`; a definite channel that disagrees with its record
  throws). Record streams are unchanged (same draws); κ = 0 shots are untouched.
- **Several Born decisions on an exact per-shot shot were sampled as independent marginals**
  (audit M, campaign seed 103021). The exact path (κ > 0 / fallback shots; `sample()` and
  `xtim.port` route every such shot there) read each decision's p_j with `born_p1` but never
  collapsed the state, so p_{j+1} ignored b_j and the retained state was not collapsed by the
  decisions at all. Each decision now collapses the exact state with its own coin
  (`measure_pauli`, whose p₊ is bit-identical to `born_p1`; a disagreement throws).
- **A Born observable and Born DECISIONs were sampled as independent laws** (twirl-targeted
  campaign, seeds 911643/911649, found by the new selfcheck semantics oracle; present on 3.1.9).
  The observable is a σ-conditioned biased coin and the decisions a memoized Born chain on a
  state that never saw the observable's outcome, so an observable and a decision reading the SAME
  record disagreed on 25% of shots (`MY 0; OBSERVABLE_INCLUDE(0) rec[-1]; DECISION(0) rec[-1]`).
  When both exist, the observable's raw outcome now enters the coin record right after the
  r + κ entries and the decision state (memo and `BarrierBuffer.materialize`) is projected onto
  W_obs at that outcome before any decision; the exact path already conditions on it. Circuits
  with only one of the two are unchanged.
- **`sample_barrier` never drew the Born observable on a noisy diagonal shot** (campaign seed
  8022). The retention path re-emitted the PREVIOUS shot's observable bit, decorrelating it from
  the shot's own detectors (invisible when they are independent; wrong joint otherwise, 3.1.9
  included). It now draws it exactly like `sample()` (σ-conditioned weight; κ > 0 / classify
  guard ⇒ the whole shot exact). The decision memo's injective key also carries the observable's
  prefix sign ⟨W_obs, P⟩ (it conditions every p_j once the observable couples to the decisions).
- **PPR exact-fallback shots reported the Born observable as 0** (front-end audit lead). When a
  CH-class shot's PPR composition falls out of class (non-commuting axes, or the κ_fold ≥ 2
  scope guard), `exact_ppr_shot` measured the observable on the shot's exact state but the
  emission passed a constant 0 and skipped the observable statistics. On the two-block CH
  benchmark with a Born observable on a magic control, P(obs) came out 0.209 against 0.270.
  The measured bit is now emitted and accounted like the obs-guard exact branch.
- **Segfault / stale record sector in the Born-decision memo** (audit SEG). Since 3.1.7 the
  memo's `materialize_shot` conditions on the shot's σ, but the abelian fast path never writes
  `o_rec.sigma`: empty on a fast-path first shot (null dereference — `compile_twirl_sampler(t,
  selfcheck=0).sample(…)` and `xtim.port` crashed for ~25% of seeds on the audit repro) and
  otherwise STALE (zeros from the last clean shot), conditioning the decision on the wrong
  record sector — silently wrong with `selfcheck=0`, i.e. on every `xtim.port` run. The memo now
  re-derives σ from the replayed (plan, prefix, coin) record — bit-for-bit the σ the record path
  wrote, and a pure function of the memo key.

#### State budget

- **A state-budget refusal inside the exact sampler's setup is a capacity error**: the setup's
  out-of-domain guard (`sampler.cpp`) caught every exception and reported a budget refusal as an
  out-of-class rejection; it now re-raises `XtimCapacityError` (likewise the reference compile's
  serialization / overlap guards and the reference loader's overlap check).

### Added

- **Selfcheck SEMANTICS ORACLE** (`semantic_oracle_`, twirl selfcheck window). On circuits with a
  Born observable or Born DECISIONs the first `min(selfcheck, 64)` shots of a sampler's first
  run recompute each shot's physics independently — `compose_fired`'s residual applied as gates
  to the bare state, projected onto the emitted raw channel bits — and require the emitted Born
  observable weight (invert included) and every decision's p_j to match it exactly (1e-9); an
  impossible channel record also trips. It would have caught I, J and M (mutation-verified:
  each fix reverted ⇒ `compile_twirl_sampler(selfcheck=2000)` raises); it cannot see a
  normalization defect (E = A, shared by every in-engine reconstruction) or post-window /
  `selfcheck=0` defects (SEG) — those are gated by the exact-oracle tests. Stream-neutral; zero
  cost once the window is spent or when `selfcheck=0`. `channel_report()["semantic_oracle_shots"]`
  reports the verified shots.

### Tests

- Regression oracles: `tests/test_twirl_audit_sampling.py` against `tests/_record_oracle.py`
  (exact record laws by dense branch enumeration); `tests/test_exact_engine_audit_fixes.py`
  against `tests/_dense_oracle.py` (independent density-matrix oracle) and the ctest
  `exact_engine_audit`; `tests/test_audit_frontend.py` (A/F/F2/K, incl. Stim `reference_sample`
  per shot); `tests/test_dem_expectation_frames.py`, `tests/test_dem_correlated_reads.py`,
  `tests/test_dem_exact_oracle.py` against `tests/exact_sv.py` (exact per-fault statevector);
  `tests/test_state_budget.py` (budget gaps).

### Changed

- **The state-memory budget covers what 3.1.9 left out** (`state_budget.hpp`):
  - *Thread scratch.* The chi-scaled per-thread scratch of the measurement, Born / expectation and
    canonicalise kernels (packed per-branch sign rows, coefficient copies, per-branch Pauli images,
    partner hash tables, the batched dual-image buffer, the transposed echelon matrices) is charged
    through one `ScratchMark` per kernel: a per-branch byte estimate at the call's chi, kept as a
    high-water mark and charged BEFORE the buffers grow (and before the measurement mutates the
    state). Scratch that does not scale with chi (O(n) words, O(n^2) bit matrices, O(#reads)
    tables, the twirl-law build scratch) is not charged.
  - *Python exports.* `branch_frames()` (state and `BarrierBuffer`) returns its chi-scaled arrays
    (`sigma`, `coeff`, `branch_x/z/phase`) backed by a charged buffer that is released when the last
    reference to the array (or a view) goes; the whole export is pre-checked. Not counted (they
    cannot carry a release hook): Python `str` / `list` returns — the `.ref` text of
    `ref_compile_text` (charged while it is built, then a plain Python object) and
    `plan_frame_cosets`' sector list.
  - *Back to baseline automatically.* Live Python-held samplers, states, barrier buffers and
    compiled programs are counted; when the last one is dropped, or a module-level call returns with
    none left, the calling thread's scratch is freed, so `live_state_bytes()` returns to its
    baseline without `_xtim._release_thread_state_scratch()` (kept, now releasing every slot).
    Scratch of other threads is freed when those threads next reach that point, or exit.
  - *A refusal in the middle of an operation.* A sampler (`CompiledProgram`, `TwirlSampler`) or a
    `FramedSuperposition` whose mutating call is refused mid-operation marks itself invalid: the
    refusal says so, and every later call raises `XtimCapacityError` ("is no longer usable")
    instead of computing from a half-updated object. Read-only calls (`materialize`,
    `branch_frames`, expectations) that are refused leave their object valid.
  - *sigma heap block.* A branch's sign pattern (r <= 30 bytes) is one 48-byte glibc chunk for
    r = 25..30; 3.1.9 charged 32 bytes per branch (exact for r <= 24) and so under-counted
    r = 25..30 by 16 B/branch. It is now charged as 48 B for every r (an over-count of 16 B/branch
    for r <= 24, never an under-count).
  - Diagnostics: `_xtim._thread_scratch_bytes()`, `_xtim._scratch_owner_count()`.


## [3.1.9] - 2026-10-09

### Added

- **Process-wide budget on live chi-scaled state storage** (`state_budget.hpp`). Every container
  whose size scales with the branch count chi = 2^r (bare-state branch lists, FramedSuperposition
  amplitude entries, parsed reference branches) now allocates through one accounting allocator:
  one shared atomic counter is charged on allocation and credited EXACTLY on release, so copies
  (`.copy()`, copy constructors), `BarrierBuffer.materialize` results, `framed_from_rows`, tensor
  products, grows/swaps and Python-side GC are covered by construction. Transient chi-scaled build
  buffers (the assembly's coefficient tables, the reference compile's text) are charged by RAII
  tickets for their lifetime, and the bare-state build pre-checks its whole peak before allocating
  anything. An allocation that would push the live total over the budget raises
  `xtim.XtimCapacityError` (a `MemoryError`) BEFORE it happens — no OOM kill — naming the requested
  bytes, the live total, the budget, chi / r / n when a build is in progress, and the override.
  Default budget 2 GiB; `XTIM_MAX_STATE_BYTES=<bytes>` (K/M/G/T suffixes = powers of 1024) raises
  or lowers it, read once per process. `xtim.live_state_bytes()` and `xtim.state_budget_bytes()`
  expose the counter and the budget. The accounted peak is conservative against measured peak RSS
  (bh14-bh18: 1.5x for the twirl/bare build, 1.1-1.2x for the exact engine's reference compile);
  every benchmark (bh up to 18, cultivation, 15-to-1, bt27, bt81) stays far below 2 GiB (bh18
  build peak 52 MB, exact compile 159 MB). The existing r > 30 representability refusal is kept.
  Hot path: one relaxed atomic add/subtract per allocation; in-place buffer reuse costs nothing.
  Not charged: per-measurement thread-local scratch (reused across calls) and Python-owned
  exports such as `branch_frames()` arrays (both charged since 3.1.10).

### Performance

- **Compile of large-n magic circuits** (rotation form, bare-state assembly). Compile,
  user-facing `compile_detector_sampler(seed)` (engine=auto -> twirl; loaded host, min of reps,
  3.1.8 -> 3.1.9): bt27 1.44 s -> 0.118 s, bt81 51.5 s -> 1.62 s, bh18 0.104 s -> 0.047 s,
  cultivation-d5 0.331 s -> 0.232 s. Every sampled stream, bare state and DEM is byte-identical.
  - `build_pauli_rotation_form` keeps the inverse Clifford frame incrementally
    (`pauli_rotation_form.cpp`). Each magic generator (T: 1, CS: 3, CCZ: 7, CH: 2) was conjugated
    into the |0> frame by walking back through EVERY Clifford seen so far (terms x gates x n, with
    fresh n-qubit allocations per gate) — 49 s of bt81's 59 s compile. The images of X_q / Z_q
    under C^-1 . C are now updated per appended gate on its 1-2 qubits, and a generator's image is
    the signed product of the rows on its support. The Pauli representation is unique, so the
    output equals the walk's bits AND phase; the walk is kept as the test oracle
    (`_xtim._rotation_form_oracle_diff`). `fold_term` compares Paulis in place.
  - The coset-family bare-state construction reads the unit-ray syndromes and the weight <= 2
    gauges off the Pauli algebra instead of materialised rays (`CanonicalStabSum::
    from_coset_family`). The branches are cosets of r commuting Z-strings on one stabilizer state
    psi, ray_g = M_g ray_0 with M_g = prod s_i^{g_i} and s_i in Stab(psi) anticommuting with the
    i-th Z-string alone: a unit ray's syndrome is the commutation of the frame generators with s_i,
    and alpha(g) = <ray_0|M_g D_sigma(g)|ray_0> is read off the anchor frame. 3.1.8 disentangled
    1 + r + C(r,2) rays through an n^3-ish H-circuit each (~8 s of bt81's ~10 s). The 3.1.8
    construction is kept as the test oracle (`AssemblePath::coset_rays`,
    `_xtim._ray_syndrome_oracle_diff`: every unit-ray syndrome, gauge exponent, b_ij and the state's
    bytes); the per-build spot check still projects and inner-products five probe branches.
  - `solve_row_combination` eliminates R once per distinct R (content-keyed) instead of once per
    even/basis column (1.8 s of the remaining 3.8 s at bt81).

## [3.1.8] - 2026-10-09

### Performance

- **Bare-state construction no longer materialises the chi branches** (`parity_assemble.cpp`,
  `CanonicalStabSum::from_coset_family`). 3.1.7 projected, disentangled and inner-producted all
  chi = 2^r rays (≈ 330–600 µs and ≈ 5 KB per branch at n ≈ 100–130: 90–99% of compile time
  from chi = 2^10 up). The branches are the cosets of r commuting Z-strings on one stabilizer
  state, so each frame generator's sign is LINEAR in the branch index and the per-branch gauge is
  a QUADRATIC character: both are read off the rays of weight ≤ 2 (r + 1 + C(r,2) rays), every
  other branch is closed form. The result is byte-identical to the 3.1.7 construction, which is
  kept as the test oracle (`AssemblePath::ray_oracle`, `_xtim._bare_state_oracle_diff`); a
  deterministic spot check re-derives five branches the old way on every build and raises
  `XtimInvariantError` on disagreement. Compile, user-facing `compile_detector_sampler`
  (loaded host): bh16 35.0 s → 0.10 s, bh18 158 s → 0.23 s (peak RSS 1.38 GB → 79 MB),
  bt27 12.6 s → 1.6 s. Every sampled stream is unchanged.
- **The remaining per-branch constants** (same commit series): the per-branch gauge exponent is
  tabulated in O(1) per branch; the final `canonicalise()` is skipped exactly when it is provably
  the identity (all r branch-sign columns independent); the coefficient product chain shares its
  prefixes across branches through one out-of-line complex multiply (bit-identical; an inlined
  `w *= z` may contract differently per call site); `FramedSuperposition::from_css` gains a
  moving overload used at the bare-state build sites. bh18: 0.23 s → 0.10 s; per-branch cost of
  the whole χ-linear part ≈ 90 ns.

### Fixed

- **A bare state with magic branch rank r > 30 was silently WRONG** (`parity_assemble.cpp`).
  FastTODD was skipped above r = 30 and `chi = 1 << r` was formed unchecked (UB; wraps mod 32
  on x86). Switch-3 (bt27, r = 33) built a truncated chi = 2 state of norm² 2^-32 that
  `engine="auto"` sampled on the exact engine ("sampling stays exact"): at p = 0, 24 detectors
  fired 50% of the time. Switch-5 (bt81, r = 87) tried 2^23 rays and ran out of memory.
  Now: r > 30 raises `xtim.XtimCapacityError` (a `MemoryError`, not rerouted by `auto`) naming
  2^r and the 2^30 limit; an allocation failure at a representable chi raises the same error;
  every parity-native build checks Σ|c|² = 1 and raises `xtim.XtimInvariantError` otherwise;
  a reference compile that reports a non-normalised state no longer falls back with the
  "sampling stays exact" message. Also guarded: `from_recipe` m > 30, and feedback plans with
  more than 64 `PAULI_EXPECTATION` records (64-bit sign mask). Streams unchanged.
- **FastTODD now runs at any magic rank** (`fast_todd.cpp`, `normal_form.cpp`): the algorithm is
  written once over the column type — `uint32_t` for r <= 30 (the old path, output identical) and
  word vectors above. The dense 2^r tables + Möbius transform that recovered the even (Clifford)
  residue are replaced by the exact symbolic expansion p_m(x) = Σ_S (−2)^{|S|−1} x^S (mod 8,
  |S| <= 3), polynomial in r; the dense path is kept as the test oracle (`even_residue_dense`).
  Switch-3 (bt27) now reduces from raw rank 33 to chi = 2^9 and samples correctly (no detector
  fires at p = 0, twirl and exact); Switch-5 (bt81, raw rank 87) also reduces to chi = 2^9.
  Streams of every circuit that compiled before are byte-identical.

## [3.1.7] - 2026-10-09

Two twirl-engine correctness fixes. No engine-version (`xtim-engine-9`) or
`.ref`-format change. Sampling streams: only twirl-engine streams of circuits
containing `DEPOLARIZE2` move (a permutation of its 15 equiprobable alternatives;
same law). The exact engine's streams and the DEM export are byte-identical to
3.1.6 (30-circuit DEM corpus; exact-engine detector hashes on 6 workloads).

### Performance (release verification)

Twirl and exact engines within about 1% of 3.1.6 per shot (interleaved, pinned,
1M shots), except `distillation_15_1_3` on the twirl engine: +6% (about 1 ns per
shot), most likely code placement: neither fix adds work to the per-shot record
loop, and that workload's stream is unchanged. A 64-byte entry alignment on the
twirl hot functions was tried. It did not recover that, and it cost 1-2% on the
`PAULI_CHANNEL_2` workloads, so it was not adopted. No `-falign-functions` build flag.

### Fixed

- **The twirl sampler applied `PAULI_CHANNEL_2` probabilities to the wrong Paulis**
  (`cpp/src/twirl_sampler.cpp`, channel setup). Its slot table was
  `XI YI ZI IX IY IZ XX XY XZ YX YY YZ ZX ZY ZZ`, not Stim's argument order
  `IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ` (first letter = first target of the
  pair): `PAULI_CHANNEL_2(p_IX=…)` fired `X` on the FIRST qubit instead of the second,
  and so on for 11 of the 15 slots — only `YY`, `ZX`, `ZY`, `ZZ` landed right.
  **Affected:** any authored non-uniform `PAULI_CHANNEL_2` sampled through the twirl
  record engine — `compile_twirl_sampler`, `compile_twirl_sampler_from_state`,
  `compile_detector_sampler(engine="twirl")`, `engine="auto"` (the default since 2.4.0)
  whenever it selects the twirl engine, and `xtim.port` — in every release since 2.3.0
  (2.3.0 – 3.1.6). Detector/observable statistics were those of the mis-assigned
  channel. **Unaffected:** `DEPOLARIZE2` and a `ZZ`-only `PAULI_CHANNEL_2` (the law is
  unchanged), every single-qubit channel, the exact engine (`compile_sampler`,
  `compile_detector_sampler(engine="exact")`), and the DEM export — those already used
  Stim's order.
- The slot order is now ONE definition shared by every noise consumer (sampler, framed
  sampler, twirl sampler, DEM export): `pc1_slot_basis` / `pc2_first_code` /
  `pc2_second_code` in `cpp/include/qeccore/circuit_ir.hpp` (closed form: the slot-`t`
  factors are the base-4 digits of `t+1`), checked at compile time by a `static_assert`
  against Stim's order written out literally. The four per-file tables are gone.
  `tests/test_pauli_channel_order.py` checks every `PAULI_CHANNEL_1`/`PAULI_CHANNEL_2`
  slot on every sampling path (exact equality with Stim at probability 1; Stim's DEM
  per slot; an analytic mixed-channel DEM; z < 5 against Stim sampling for a
  15-distinct-probability channel); swapping two slots turns it red on every path.
- **Sampling-stream note:** the twirl engine's `DEPOLARIZE2` alternatives are now in Stim's
  slot order, a permutation of the 15 equiprobable alternatives: the same law, but a
  different seeded byte stream for circuits with `DEPOLARIZE2` sampled through the twirl
  engine. Two pins re-pinned with this justification (the `adaptq_qrm_magic_producer`
  stream hash in `tests/test_frame_cosets_cpp.py`, its T2 gap total 34 → 33 in
  `tests/test_plan_structure_classifier.py`); a build that keeps the old order for
  `DEPOLARIZE2` alone reproduced both old pins and the full suite.
- **The retained barrier state was not conditioned on its own records after a
  T-crossing fault** (`cpp/src/twirl_sampler.cpp`, `materialize_shot` →
  `condition_on_records_`; found by adaptq's harness3 investigation, §D2). A fault
  that crosses a `T` leaves an S-tail residual that mixes the sector of a split
  check as a fair coin. `twirl_collapse` deliberately leaves such coins
  uncollapsed — right for a check nobody reads, wrong for one whose sign reaches a
  record. The record path drew the coin into σ and from σ into the DETECTOR /
  in-group DECISION / in-group OBSERVABLE bits, but `BarrierBuffer.materialize(i)`
  (and `state`, `pauli_expectations_all`, `branch_frames`) returned the
  unprojected state. On such a shot the read check had ⟨g⟩ = 0 while its record
  said ±1. Now `materialize_shot` takes the shot's recorded σ and projects the
  state onto G_c = Π_{i∈c} g_i = (−1)^{c·σ} for a maximal set of record channels
  c whose coin rows are independent. Checks nobody reads stay coherent. Oracle:
  `tests/test_twirl_retained_records.py` compares, shot by shot, every
  output-wire Pauli with a dense state vector projected onto the shot's records.
  It covers the 4-qubit §D2 repro, two Bell pairs with read and unread split
  checks, and records carried by a DECISION or an OBSERVABLE. It is red on 3.1.6
  and on a mutation build that defines `XTIM_MUTATE_D2_NO_RECORD_CONDITIONING`.
  **Who is affected:** consumers of `sample_barrier` retained states (`materialize`
  / `state` / `pauli_expectations_all` / `branch_frames`) on circuits where a
  `T`/`T_DAG`-crossing fault splits a check that a record reads. In the bundled
  examples these are `cultivation_d3_faithful`, `cultivation_d5`,
  `code_switching_faithful` and `miniature_oracle`; the `*_rate` variants and
  adaptq's qRM/Steane producers are unaffected. Affected versions: every release
  with `sample_barrier`, 2.5.0 through 3.1.6. **Unchanged:** every record stream
  (dets / obs / decisions / σ / coins / plan keys, `sample()` and
  `sample_barrier()`) is byte-identical on the bundled corpus, and so is every
  retained state of a shot without a record-bearing coin (all clean shots).
  `C++ API`: `TwirlRecordSampler::materialize_shot` gains a `sigma` argument.

## [3.1.6] - 2026-10-07

One additive `BarrierBuffer` binding for adaptq's segmented Choi harness and
the xtim-side regression test for the 3.1.5 injection pre-bake fix. No engine
kernel, engine-version (`xtim-engine-9`), `.ref`-format or sampling-stream
change.

### Added

- `BarrierBuffer.pauli_expectations_all(reads)` — materialize every shot and read
  a list of literal Paulis on each, `(shots, len(reads))` complex, one call (the
  segmented Choi harness's matching and final overlaps).
- `tests/test_injection_prebake.py` — pins the 3.1.5 injection pre-bake rule (two
  syndrome rounds in one injected stage compile and agree with the monolithic circuit).

## [3.1.5] - 2026-10-07

Additive `FramedSuperposition` bindings for adaptq's segmented Choi harness
(every shot kept; spec in adaptq
`docs/superpowers/specs/2026-10-07-segmented-harness-design.md`), plus one
state-level-injection fix. No engine kernel, engine-version (`xtim-engine-9`),
`.ref`-format or sampling-stream change.

### Fixed

- **State-level injection pre-baked every fresh-only gate into the fresh block**
  (`cpp/bindings/xtim_py.cpp`, `prebake_mask`). The injection built the fresh
  block from EVERY gate that touched only fresh wires and applied only the
  port-touching gates in stream order. That is unsound once a fresh ancilla has
  interacted with the port — e.g. a second syndrome round inside one injected
  stage, whose deferred `H` on the ancilla follows its `CX` to the port — and the
  exactness guard refused such stages. Now only the fresh-only gates that precede
  their wire's FIRST port contact are pre-baked; every later gate is applied in
  stream order. Stages that were accepted before classify identically
  (byte-identical).

### Added

- `FramedSuperposition.copy()` — an independent deep copy.
- `FramedSuperposition.grow(k)` — append `k` fresh |0> qubits (the state-level
  injection's own grow).
- `FramedSuperposition.permute(perm)` — relabel qubits (`perm[q]` = new index of
  old qubit `q`); `Π·U` is a valid frame of `Π|ψ>`, so only the row columns move.
- `FramedSuperposition.pauli_expectation(xs, zs, phase)` — the exact complex
  `<i^phase X^xs Z^zs>` (the `certified_symplectic` literal convention), no copy;
  `pauli_expectations(reads)` the batched form.
- `framed_from_rows(n, zx, zz, zph, xx, xz, xph, eps, free, entries)` — a
  `FramedSuperposition` from explicit frame rows, `eps`, free indices and
  amplitude entries; validates a symplectic frame, Hermitian rows and unit norm.
  adaptq rebuilds a carried state's PORT RESTRICTION with it (the state on the
  port wires only), so a segment sampler's deferred space stays small.

## [3.1.4] - 2026-10-07

Correctness patch; no API, engine-version (`xtim-engine-9`), `.ref`-format or
sampling-stream change. Only the exported detector error model changes, and only
for circuits that carry a readout-flip probability on a measurement instruction.

### Fixed

- **DEM export dropped measurement readout noise** (`cpp/src/dem_export.cpp`,
  `docs/dem_export_readout_noise.md`). **Disclosure — every release that accepts the
  `M(p)` spelling (0.3.0 through 3.1.3) is affected:** a DEM exported by xtim ≤ 3.1.3 from a circuit with a
  readout-flip probability on any of `M(p)`, `MX(p)`, `MY(p)`, `MR(p)`, `MRX(p)`,
  `MRY(p)`, `MPP(p)`, `MXX(p)`, `MYY(p)`, `MZZ(p)` omitted those mechanisms entirely, so a
  decoder driven by that DEM never corrected a readout flip; logical error rates obtained
  by decoding xtim samples against such a DEM are too HIGH (the sampler always applied the
  flip) and should be re-run with 3.1.4. Circuits whose readout noise is authored as a
  separate channel (`X_ERROR(p) q` then `M q`) were never affected, and the samples
  themselves were never affected. A readout-flip probability written on the
  measurement itself — `M(p) q`, `MX(p)`, `MY(p)`, `MR(p)`, `MRX(p)`, `MRY(p)`, `MPP(p)`,
  `MXX(p)`/`MYY(p)`/`MZZ(p)` — was silently ABSENT from `detector_error_model()` /
  `detector_error_model_text()` / `detector_error_model_with_reject()` / `_xtim.export_dem_text`:
  the exporter walked only the Noise instructions of the deferred stream and never read
  `Instr::readout_flip_p` (which deferral drops from the terminal Measures; the sampler reads
  it off the pre-deferral circuit, the exporter did not). Stim's `detector_error_model()`
  emits `error(p) <every detector / observable reading that record>` for each such
  measurement, so a decoder fed xtim's DEM had NO readout mechanisms and never corrected a
  readout flip (found 2026-10-06 in adaptq: a Steane 1-round memory under a run-time noise
  policy that spells readout noise as `M(p)` decoded at the raw readout-flip rate, 6.6 % at
  p = 1e-3, instead of ~1.2 %). The exporter now folds, after the noise-channel walk, one
  independent mechanism per record with `readout_flip_p > 0`, read off `nr.coherent` in
  record order exactly as the sampler does: signature = the detectors / observables whose
  record set reads it (parity), the `PAULI_EXPECTATION` columns whose declared byproduct
  frame folds it (the sampler applies the flip BEFORE the frame fold), and the `DECISION`
  logical-flip columns; same cross-mechanism XOR composition, same emission order. A
  readout flip is a classical record flip — no propagation, never a post-select flag, no
  `OUTPUT_QUBITS` frame column. Oracle (`tests/test_dem_measurement_noise.py`, 32 tests):
  `stim.Circuit(text).detector_error_model()` after canonicalisation on every M-family
  spelling, `!`-inverted and p = 0.5 records, multi-target lines, same-signature
  composition with `X_ERROR`, mid-circuit `MR(p)` with qubit reuse, MPP rounds, a Steane
  1-round memory, and Stim's generated repetition / surface memories with their
  `X_ERROR(p); M` readout noise rewritten to `M(p)` — 28 circuits, 1080 mechanisms, every
  set equal, every probability within 1e-12 (423 bit-exact; the remainder is the
  pre-existing forward-fold vs Stim back-walk FP ordering, identical counts on the authored
  twins). A magic-bearing protocol (`cultivation_d3_rate`, T gates + post-selection) is
  pinned to an independent Python composition oracle and to its authored twin, with
  identical `postselect_faults`. Every readout-free export is byte-identical to 3.1.3
  (18-circuit corpus: bundled examples, generated memories, hand circuits, including the
  refusal texts). Export time unchanged (cultivation_d3_rate 9.9 vs 10.1 ms; surface d=5
  r=3 53.8 vs 53.3 ms, warm median of 7). The sampler was never affected (it always applied
  the flip); only the exported model was wrong.

## [3.1.3] - 2026-09-22

### Added

- **`_xtim.plan_frame_cosets` — the Born SECTOR table of an S-layer plan, compiled**
  (`cpp/include/qeccore/frame_cosets.hpp`, `docs/frame_cosets_cpp.md`).
  `xtim.frames.plan_frame_cosets_fast(ref, a_mask, *, backend=None)` now dispatches to it:
  `backend="auto"` (the default) MEANS the compiled table; a loaded extension that lacks
  the symbol or exposes another `_xtim.FRAME_COSETS_ABI` than the package's
  `xtim.frames.FRAME_COSETS_ABI` (= 1) is a stale build and is REFUSED with
  `xtim.frames.StaleExtensionError` (a `RuntimeError` naming the `.so` and the rebuild
  command) — there is no silent fallback to the 20× slower pure-Python path;
  `backend="numpy"` / `XTIM_FRAME_COSETS_BACKEND=numpy` select that path explicitly, and
  an unknown env value is refused at import.  The 3.1.2 body is kept as
  `xtim.frames.plan_frame_cosets_numpy` and is the ORACLE the compiled table is pinned to
  BIT FOR BIT — same sectors, same first-encounter order, same fields and dtypes, the same
  `weight` double — on every plan of the multi-sector fixtures, the synthetic masks to
  |supp a| = 9, the 287 REAL a-masks of adaptq's u2 unit
  (`tests/data/u2_qrm_magic_amasks.npy`) and a seeded random sweep
  (`tests/test_frame_cosets_cpp.py`); both stay pinned to the reference
  `plan_frame_cosets` (bit-for-bit on the weight as well, plus the 3.1.2 tolerance leg).
  Same refusals, same exception types (`ValueError` for a null-space read that is not ±1,
  `AssertionError` for an export that is not a full frame); the compiled path's one extra
  refusal is `|supp a| > 30` (`ValueError`).  The sampling hot path is untouched: the
  record stream of the three data fixtures hashes to the 3.1.2 values (test).  C++ unit
  test `cpp/tests/test_frame_cosets.cpp`.
- **`weight` arithmetic made PORTABLE in all three sector-table bodies** (reference,
  numpy, compiled): per branch `re·re + im·im`, accumulated left to right, normalised by an
  explicit left-to-right sum — IEEE basic operations in a fixed order, so the bit pin holds
  on every interpreter / numpy / CPU dispatch xtim declares (python ≥ 3.11, numpy ≥ 1.26)
  instead of reproducing numpy's FMA-dispatched complex `absolute`, its pairwise sum and
  CPython 3.12's compensated `sum()`.  Versus 3.1.2 the numpy body's `weight` moves by at
  most ~1 ulp on some plans; no consumer reads its bits (adaptq consumes
  `frame_copy`/`fx`/`fz`; every test that reads `weight` pins it with a tolerance) and
  adaptq's byte gate is unmoved (re-run: PASS).
  Cost: the u2 unit's real plans go from 0.44 ms median / 0.79 ms mean per plan (pure
  Python, 45 % `ref_expectation` interpreter overhead, 30 % numpy setup, 25 % the
  per-sector Python loop) to the compiled figures recorded in `docs/frame_cosets_cpp.md` §5
  and the arc report; adaptq's `engine._plan_frame_cosets` needs no change.
- `setup.py`: `cpp/include/qeccore/frame_cosets.hpp` listed in the extension's `depends`
  (the header is included by the bindings TU only; without it an edit would not rebuild
  `_xtim.so`).

## [3.1.2] - 2026-09-09

### Added

- **`FramedSuperposition.branch_frames()` / `BarrierBuffer.branch_frames(i)`** — export a
  retained state as `|psi> = sum_i c_i D_i |ref>`: the n SIGNED generators of `|ref>` (frame
  rows with eps folded, the `certified_stabilizers()` convention extended to the free rows),
  the destabilizers, the free indices, and per branch its sigma, coefficient and Pauli
  `D_i = prod_{d asc: sigma_i[d]} d_{free[d]}` (engine phase algebra, the `oW[i]` of the
  anticommuting kernel), all in the state's wire indices. Read-only; O(chi*n) bit copies on
  the materialized state; the sampling path is byte-identical (suite unchanged).
  `xtim.frames` ships the consumer algebra (`ref_expectation`, `expectation` on any branch
  subset, `branch_sign`, `dense_state` for oracles), tested bit-for-bit against
  `pauli_expectation` incl. Y-type Paulis (which `pauli_expectation_xz` cannot read — it
  drops the i-phase). Purpose: the exact per-BRANCH logical charge of a twirl-split
  (recipe-coherent) magic-port state, which no group-level expectation can separate
  (`docs/branch_frames.md`). `xtim.frames.projected_expectation` reads a Born SECTOR
  `(<Pi P Pi>/<Pi>, <Pi>)` of the split checks (a recipe-coherent retained state is chi = 2
  in a frame where the split checks are not stabilizers; the consumer's branches are those
  sectors). `branch_frames` + `xtim.frames` are the ORACLE surface.
- **`BarrierBuffer.frames(allow_kappa=False)`** — "frames at the port": the per-shot residual
  PREFIX `P` (the shot's exact Pauli frame relative to the noiseless reference, already stored
  per retained shot) as `prefix_words uint64[shots, 2*pnw]`, the per-shot `plan_id`, and per
  distinct plan its S-layer mask `a`, `cz`, `r`, `kappa` — straight copies of the record store
  (no state replay; ns per shot). With `xtim.frames.unpack_prefix / plan_frame_cosets /
  sector_coset / frame_charge` a consumer computes every Born sector's recipe charge as
  `<P,O> ⊕ <F_c,O> ⊕ ref` (F_c the sector's frame from the per-plan table) by bit algebra — exact for kappa = 0 plans, which the
  export verifies against `materialize(i)` shot by shot; kappa > 0 / fallback plans are REFUSED
  loudly (never silently materialized; the `fallback` field is exported so the consumer twin
  `require_kappa_zero` mirrors both halves). Per-buffer memo of `plan_structure` per plan
  suffix, so repeated `frames()` calls are pure copies. `frames.plan_frame_cosets_cached`
  (per-plan memo of the only expensive step), `sector_coset` refuses non-frame-copy sectors,
  `frame_charge` documents its two `ref` conventions. In-suite multi-sector oracles (GHZ+T with
  an X fault before T: 2 sectors; two magic qubits: 4 sectors, m = 2) pin sector weights to the
  Born weights and `frame_charge` to the sign of `projected_expectation` on `materialize(i)`.
  `docs/branch_frames.md` §7.
- **`xtim.frames.plan_frame_cosets_fast` + `lambda_functional`** — the per-plan sector table
  as the PRODUCTION path (the wrapper's consumer must not carry the algebra): `λ_{v,c} =
  <r|Z^{v⊕v_c}|r>` is a GF(2) linear functional (a homomorphism K -> {±1}; proof in the
  docstring and §7), so it is fixed by `|supp a|` reads on a null-space basis instead of
  `2^{|supp a|}`, and the remaining sums are vectorized; `plan_frame_cosets` stays as the
  oracle (`plan_frame_cosets_cached(..., reference=True)`), pinned equal field by field on the
  multi-sector fixtures and on synthetic masks up to |supp a| = 9. Same sectors, same order,
  same fields; milliseconds -> tens of microseconds per plan (numbers in §7 / the report).

### Known work (not in 3.1.2)

- **One shot tail, not two.** The exact engine's noiseless reference is relabeled
  by a second copy of the shot tail in `compile_sampler_program` (`!` invert XOR,
  then the feedback relabel) rather than by the shared `pack_shot_records`; the
  order of those two passes was wrong once (3.1.1 class B below) and is pinned
  only behaviourally. Planned: derive the reference by running `pack_shot_records`
  on the reference word with the `M(p)` coin masked to 0, so there is one
  mechanism and no order to get wrong. Deferred because it touches the sampler
  setup path and deserves its own byte-identity run.

## [3.1.1] - 2026-09-07

Correctness release for classically-controlled Pauli feedback (`CX/CY/CZ rec[-k] q`).
Feedback-free circuits, and feedback that crosses a non-Clifford gate (the shipped
magic examples), are byte-identical to 3.1.0: the three pinned engine streams —
`cultivation_d3_rate`, `cultivation_d5`, `distillation_15_1_3` — and the downstream
consumer's eight byte-gate legs were re-verified against this engine. The streams
that move are the four classes A–D below, each verified against real Stim (a hand
corpus of the idioms plus 234 random Clifford+feedback circuits with and without
noise, both engines: 0 mismatches, 0 refusals of a Stim-expressible circuit).

### What moves, and where

| class | route | 3.1.0 | 3.1.1 |
|---|---|---|---|
| **A.** feedback inside a pure-Clifford region (byproduct crosses no `T`/`CS`/`CCZ`/`CH`): any control basis, reused control wires, chained and broadcast feedback | twirl (`engine="twirl"`, `compile_twirl_sampler`, `sample_barrier` / `materialize`) | refused ("no deterministic detector channels" / anti-commuting) or a wrong absolute state, with detector columns masked by a reference built from the same feedback-stripped circuit | right |
| **B.** non-crossing feedback whose control record carries a `!` — user-written, or stamped by `eliminate_hadamards` absorbing `X c; M c`, i.e. whenever the circuit contains any `H` | exact engine, `run_stim_main` CLI | detector/observable columns inverted (raw records right) | right |
| **C.** `M(p)` feedback source whose control is observed again before a `R` (re-measured, gated, fed back, named in a `PAULI_EXPECTATION` or an `OUTPUT_QUBITS` port) | both engines | control's post-measurement state flipped | right |
| **D.** feedback onto its **own** control wire (`M q; CX rec[-1] q`, measure-and-conditionally-reset), idle control | twirl | fair coin, or refused | right |
| **E.** feedback-free circuits; crossing feedback (`cultivation_d5`, `distillation_15_1_3`) | all | — | byte-identical |

`engine="auto"` was protected on class A/D circuits (the twirl classifier demoted
the affected channel to gauge and auto fell back to exact). DEM export
(`detector_error_model_text`) refuses **every** feedback circuit, before and after
this release — a DEM has no record-control concept — so it neither moves nor is
"unaffected": feedback is out of its scope.

### Fixed

- **The twirl engine now honours feedback inside a pure-Clifford region.** Through
  3.1.0 every pybind `TwirlSampler` entry point normalized with `Feedback::Strip`
  and built no `FeedbackPlan`, so a feedback whose Pauli byproduct did **not** cross
  a non-Clifford gate — any correction in a Clifford region — was silently deleted
  from both the retained/materialized state and the record algebra. Symptoms were
  wrong absolute readouts (`PAULI_EXPECTATION`, `sample_materialized` states) and
  spurious "no deterministic detector channels" refusals of well-formed
  teleportation-style circuits; detector/observable values looked self-consistent
  because the reference was computed from the same feedback-stripped circuit.
  Feedback that crosses a `T`/`CS`/`CCZ`/`CH` was never affected (it was already
  coherentized into real gates), so `examples/cultivation_d5.stim` and
  `examples/distillation_15_1_3.stim` were always correct and are byte-identical.
  Internally: `NormalizePolicy::coherentize_all` — every feedback is rewritten into
  coherent gates (copy ancillas for reused control wires and self-targeting
  feedback, `M(p)` folded into a physical pre-measurement error) — behind one
  `normalize_for_twirl` preset. The exact engine and the `run_stim_main` CLI keep
  their post-sampling relabel for non-crossing feedback (same mechanism as 3.1.0,
  with the one reference-order fix below). (`docs/xtim_dialect.md` §1b documents
  both mechanisms.)

- **The exact engine's detector/observable columns were wrong for non-crossing
  feedback whenever the circuit contained a Hadamard** (raw measurements were
  right). The noiseless reference is relabeled by the same triangular feedback pass
  as every shot, but ran it *before* XOR-ing in the deterministic `!` invert, while
  the shot path applies the invert first — and a feedback reads the *recorded* bit,
  invert included. `eliminate_hadamards` only rebuilds a circuit that has an H to
  eliminate, and when it does it absorbs `X c; M c` into `M c` with a stamped
  invert, which is why any Hadamard anywhere (even `H 0; H 0`) inverted every
  detector the feedback fed: 200/200 vs Stim 0/200. The reference now applies
  invert-then-feedback exactly as `pack_shot_records` does
  (`cpp/src/sampler.cpp`). Pinned against Stim in
  `tests/test_twirl_feedback_semantics.py`.

- **A noisy-readout feedback source (`M(p) c`) no longer disturbs the control's
  post-measurement state.** The coherentizer folds `M(p)` into an anticommuting
  error *before* the measurement so the recorded bit and the driving value flip
  together; that also flips the projected state, which Stim's `M(p)` does not — so
  a control re-measured (or otherwise observed) before a `R` read the flipped
  value (200/200 at p=1). The flip instance is now held on a fresh ancilla and
  applied to the control before *and* after the measurement (record flipped,
  state restored, shot for shot), with the feedback driven from a copy ancilla.
  Taken only when the control's state is observed again before a `Reset`; the
  previous single-error emission is kept, byte for byte, otherwise. This is the
  shared coherentizer, so it also fixes the exact engine on *crossing* noisy
  feedback with a reused control wire.

- **A feedback onto its own control wire is no longer coherentized into `CX q q`.**
  `M q; CX rec[-1] q` (measure-and-conditionally-reset; likewise `CZ`/`CY`, a `!`
  source, `MX`/`MY` sources) has an idle control, and the coherentizer drove an
  idle control "in place" from its own wire — for a self-targeting feedback that
  is a controlled-Pauli from `q` onto `q`, which is not a unitary on the wire at
  all (both `|0>` and `|1>` go to `|0>`; it is exactly what a classical record
  makes implementable and a coherent entangler from the wire itself cannot). The
  parser rejects a duplicate qubit in user text, but the coherentizer bypasses
  the parser and nothing downstream objected: the twirl path answered a fair
  coin on the re-read of `q` (or refused as anti-commuting), and this class was
  the only one moving in the wrong direction in the 3.1.1 candidate (loud
  refusal → silent wrong answer). A self-targeting feedback now takes the same
  copy-ancilla gadget as a reused control, and `emit_coherent_feedback` throws
  on `control == target` so a degenerate entangler can never leave the
  coherentizer silently again. The exact/CLI relabel route always handled this
  class (a self-targeting Pauli is a trivial record relabel) and is unchanged.
  Pinned against Stim in Z/X/Y, inverted, noisy-source and wire-reused forms
  (`tests/test_twirl_feedback_semantics.py`; structural contract in
  `cpp/tests/test_coherentize_all.cpp`).

### Note (setup cost on the twirl path)

- Circuits with feedback inside a Clifford region now compile on the twirl path
  where they previously refused. Every classically-controlled Pauli keeps its
  control wire live across the deferral, and three cases add ancillas: a control
  wire **reused** before its feedback, a **self-targeting** feedback, and an
  `M(p)` source whose control is **observed again** (one copy ancilla each; the
  noisy held form adds one flip ancilla too, which stays live to the end of the
  circuit — a classical mixture controlling two cancelling CXs, harmless). The
  per-shot hot path is untouched (no per-shot source changed; the pinned
  streams are byte-identical), so the cost is compile/setup only and scales
  with the live wire count: identical to the same circuit with its feedback
  removed for the in-place class, and about 3× the compile time at N=40 chained
  noisy-observed sources for the held form (2N extra live wires), with no
  measurable per-shot change. Ratios from interleaved A/B runs on a loaded host;
  no absolute timing is quoted.

## [3.1.0] - 2026-09-05

Feature release. No behavioural change to sampling, the port, or any numerical
result: decision/detector streams are byte-identical to 3.0.1 (the downstream
consumer's seven pinned streams and its full battery were re-verified against
this engine before tagging).

### Added

- **`FramedSuperposition.overlap(other) -> float`** — the exact overlap VALUE
  `|<self|other>|^2` (global-phase insensitive) between two retained states, the
  quantity `approx_equal` computes internally (`exact_sum_overlap`: framed ray
  materialisation + the stabilizer inner-product kernel) and then thresholds.
  Both states must have the same qubit count; a mismatch raises `ValueError`.
  Motivation: per-shot Choi-state (entanglement) fidelity estimation of logical
  protocols — one stabilizer-superposition inner product per shot instead of a
  sum over `4^{2k}` Pauli expectations. Oracle: `tests/test_overlap_binding.py`
  (closed-form overlaps of Bell-pair mutations; consistency with `approx_equal`).

## [3.0.1] - 2026-08-30

Maintenance release. No behavioural change to sampling, the port, or any
numerical result: the public decision/detector streams are byte-identical to
3.0.0 (verified against the downstream consumer's seven pinned streams).

### Fixed

- **Public type annotations now resolve.** `xtim.fidelity_helpers` annotated its
  public signatures — `extract_frame`, `fidelity_from_logicals` and the helpers
  they delegate to — with `"xtim.Circuit"` and `"PostselectedLER"`, names the
  module never imported. Harmless under postponed annotations, but
  `typing.get_type_hints()` raised, and IDEs and documentation tooling could not
  resolve the types on a package meant to be read. All 13 callables in that
  module now resolve.

- **Release verification no longer skips its heaviest legs.** Two tests
  resolved their example circuits relative to their own file rather than the
  installed package, so `RELEASING.md`'s "run the suite against the installed
  package" step silently skipped them — including the `cultivation_d5`
  port-groups workload. Against the 3.0.1 wheel: 153 passed / 11 skipped →
  156 passed / 8 skipped.

### Changed (internal)

- Dead imports removed (`xtim.diagnose`, `xtim.collect`) and a dead local in the
  twirl record-scatter loop that survived only in a comment. `ruff --select F`
  (pyflakes rules) is clean across `xtim/`, `tests/` and `examples/`.
- The three private `_bare_state_of` / `_input_state_of` /
  `_project_bare_onto_syndrome` re-exports in `xtim/__init__.py` are retained
  deliberately (pre-2.x spellings, kept working) and marked as such rather than
  removed — dropping a name from a published package is a breaking change.

## [3.0.0] - 2026-08-29

**The declared-consumption port ships.** `2.7.0` said "v3 will simplify the API
(a declared-consumption segment surface currently in development replaces some
of today's lower-level sampler plumbing)" — this is that release. `xtim.port`
becomes public API: you declare once what you will read, the port compiles once,
routes to the cheapest engine surface that serves the declaration, and computes
nothing undeclared. Everything else in the `2.x` API is unchanged, streams
included: `ENGINE_VERSION` stays `xtim-engine-9`, `sample()` / `sample_barrier()`
byte streams are unmoved, and no consumer `.ref` cache invalidates.

### BREAKING

Two `xtim.port` parameters are **removed**. Neither was ever in a published
release (the port module itself was held back from `2.7.0`), so this breaks only
consumers built against the unpublished `2.8.0`-class development line:

- **`Segment.run(..., frame_in=…)`** and **`Result.frame_out`** — removed.
  `run(shots, seed)` is now the entire signature and `Result` carries only the
  declared record outputs. WHY: the seam contract was ratified in its strongest
  form — **only classical data ever crosses a segment seam, and the engine holds
  no state between calls**. The Pauli-frame crossing is per-shot classical data
  held and applied *by the consumer* as record post-processing (frame-correcting
  records by anticommutation, the boundary partial-syndrome channel for non-CSS
  seams, the charge-matched frame advance, the terminal syndrome XOR). There is
  therefore nothing an engine-side injection surface could do, and a live
  `frame_in` would invite exactly the engine-held-state coupling the contract
  forbids. `frame_in` had been a loud `NotImplementedError` and `frame_out` an
  always-`None` attribute; both are now simply gone, so `Segment.run` is a
  stateless draw *by construction*. Callers who genuinely need an engine-side
  input Pauli use the raw sampler's `input_pauli` on `xtim.twirl`, outside the
  port. The seam contract is stated positively in `xtim/port.py`'s docstring.
- **`Segment.run(..., input_bits=…)`** — removed, with its per-shot row-gather
  pattern-enumeration path (per-pattern full-shots draws, the distinct-pattern
  cap, the draw memo, and the merged pattern-major group surface). It had no
  consumers. Consequence: a circuit carrying `IF` blocks driven by external
  `INPUT_BITS` now **refuses at `port.compile`** rather than silently serving the
  all-zeros resolved branch; the refusal names both the circuit property and the
  escape. Resolve the branches yourself (`xtim._xtim.resolve_branches_text`) and
  compile each resolved text into its own `Segment`, or use the unchanged raw
  `xtim.twirl.PartitionedTwirlSampler.sample(decision_bits=…)` surface.

### Added

- **`xtim.port` — the declared-consumption segment surface** (public debut; also
  exported top-level as `port`, `Consume`, `compile_segment`, `Segment`,
  `Result`). Two nouns, two verbs:

  ```python
  from xtim.port import Consume, compile as compile_segment
  seg = compile_segment(circuit_text, Consume(dets=True, decisions=True))
  res = seg.run(shots=200_000, seed=42)
  ```

  - **Declaration-driven routing.** `dets`/`decisions` alone route to the bare
    `TwirlSampler.sample()` path; `groups`/`residual_law` route to the record
    surface. Declaring nothing is a loud `ValueError`. Nothing undeclared is
    computed, stored, serialized or surfaced — a groups-only Segment never
    performs a σ-law fetch (spy-gated in the tests).
  - **Compile once, run many.** `compile()` memoizes per
    `(text, consume, plan_cache)` and returns the identical Segment object on a
    hit; `run(shots, seed)` reseeds the engine **in place** (no reconstruction),
    so a seeded draw costs only the sampling.
  - **Group-indexed results.** `Consume(groups=True)` adds `group_id` per shot
    plus per-GROUP arrays — `group_keys`, `group_first_shot`, `group_sigmas`,
    `sig_bits`, representative `group_dets`/`group_decisions`, and lazily parsed
    `group_coins`/`group_plans` — one row per distinct residual class instead of
    per shot. Arrays are handed out as views on the engine's own buffers where
    the layout allows (no per-shot Python objects, no `.tolist()`).
  - **σ-law read model.** `Consume(residual_law=True)` adds `Segment.sigma_law`,
    `born_dec`, `output_wires`, `sig_bits`/`gw` and the memoized
    `plan_structure(plan_key)` — pure functions of the compiled circuit,
    computed once at compile time.
  - **Compact record store (internal).** The record route samples with the
    engine's compact sink: per shot it stores an interned dense plan id plus
    residual-prefix support words, σ words and coins instead of serializing
    ~600 B of plan bytes, and materializes the legacy padded key bytes once per
    GROUP at read time. Contract, oracle-gated: the group partition, the
    first-occurrence order and every legacy key byte are identical to the
    historical `record_groups()` output. Old callers (`compact=False`, the
    default) are byte- and cost-identical.
  - **Plan disk cache**, as a compile option: `compile(..., plan_cache=True)`
    for the automatic path (`$XTIM_TWIRL_CACHE_DIR` or `~/.cache/xtim/twirl/`,
    one `.twpl` file per structural circuit signature) or an explicit path.
    Location/size/eviction/byte-stability policy is documented in the module
    docstring — notably, a pre-warmed cache does not move record-route bytes,
    and stale or corrupt files load nothing and are silently rebuilt.
  - **Refusals name both sides.** Every refusal states the circuit property AND
    the colliding declaration: CH/PPR (non-diagonal) circuits refuse
    `groups`/`residual_law` (state retention is diagonal-only — physically there
    is no Pauli residual to read) but still serve `dets`/`decisions`; a circuit
    with zero record channels refuses with the concrete fix (declare the
    feedback MPPs as `DECISION(k)`); IF-driven circuits refuse as above.
  - **Immutability.** `Consume` is frozen; `Segment` and `Result` are
    `__slots__` objects that reject mutation. Undeclared fields raise
    `AttributeError` naming the declaration that would provide them.

- **THE SEAM CONTRACT** — stated in `xtim/port.py`'s module docstring: what
  crosses a segment seam (classical data only, consumer-held) and why the port
  therefore has no frame surface.

### Performance

- **Born-decision fast path.** Born-decision circuits now run the fast abelian
  σ-path (`need_state` decoupled from `has_born_dec_`) and feed the existing
  Born-probability memo instead of taking a per-shot canonicalize/collapse
  detour. Noisy H+T: **5.82 → 5.27 µs/shot** at 40k shots/call and
  **2.33 → 2.08 µs/shot** at 200k. The fast-σ == structural-σ self-check stays
  active; non-Born and exact-tier streams are byte-identical, and for a
  `kappa=0` diagonal plan the Born DECISION stream itself is byte-identical to
  the pre-change structural path (collapsed amplitudes are invariant to the fair
  coins) — verified over 40k H+T + 5-Born-qubit shots at 5σ with exact `m_out`
  weights.
- **Record surface** (cultivation_d5, 100k shots/call, plan-warm):
  `sample_barrier` **5.42 → 4.54 µs/shot**, `record_groups`
  **3.84 → 1.11 µs/shot**, end-to-end **9.26 → 5.65 µs/shot** — the compact
  record store plus the matrix (never list-of-lists) key surface.
- **In-place `set_seed` carries forward** (shipped in 2.6.1, unchanged here) —
  the single largest win on the seeded-sampling path and the reason the port's
  compile-once / `run(shots, seed)` loop is viable at all: a seeded call never
  reconstructs the sampler, so a seeded zero-shot call costs ~0.04 ms instead of
  the ~180 ms full rebuild that preceded it.

### Performance (release verification)

Re-measured for this release on the release tree (linux x86_64, CPython 3.12,
48 cores, **host load ~2.1–2.5** — a shared box, not idle; a quiet host runs
lower). Warm median-of-5, one warmup call per leg, circuit `cultivation_d5`
(42 qubits, 107 detectors), `selfcheck=0` (the fast path — with the
once-per-sampler window a default sampler is in the same class from its second
call on). A second pass at load ~3.6 reproduced every leg within 5%.

| leg | result |
|---|---|
| record `sample()` 200k shots/call, same-seed warm | 0.59 µs/shot |
| record `sample()` 2k shots/call, same-seed warm | 0.57 µs/shot |
| record `sample()` 200k shots/call, fresh-seed warm (plan-law tax) | 3.7 µs/shot |
| `sample_barrier()` 200k shots/call, same-seed warm | 6.0 µs/shot |
| seeded zero-shot `sample(0, seed=…)` call | 0.031 ms |
| **port** `Segment.run(Consume(dets, decisions))` 200k/call, same-seed warm | 0.61 µs/shot |
| **port** `Segment.run(Consume(dets, decisions))` 2k/call, same-seed warm | 0.58 µs/shot |
| **port** `Segment.run(… groups=True)` 100k/call, same-seed warm | 5.9 µs/shot |

Every leg is at or better than the 2.7.0 release table measured under heavier
load, and the port costs ~0.02 µs/shot over the raw sampler it wraps — the
declaration is resolved at compile time, not per call. The 0.031 ms seeded
zero-shot call is the standing confirmation that the ~180 ms per-call rebuild
tax is gone.

### Compatibility

- `ENGINE_VERSION` unchanged (`xtim-engine-9`); `REF_FORMAT_VERSION` unchanged
  (4). No `.ref` cache invalidates.
- `sample()` / `sample_barrier()` byte streams are unmoved from 2.7.0 for every
  non-Born-decision circuit; the Born-decision fast path is byte-identical on
  `kappa=0` diagonal plans and law-identical (5σ-verified) elsewhere — it was
  re-pinned once when it landed.
- The selfcheck oracle window still runs once per sampler lifetime (2.7.0
  behavior), never re-arming on `set_seed`, and remains stream-neutral.
- No removals outside `xtim.port` (see BREAKING).

## [2.8.0] - 2026-08-11

*Local tag only — never published. Its contents ship in 3.0.0 above, minus
the two dead port parameters 3.0.0 removes.*

**The frame-forwarded (FF) arc's engine-side surface.** `xtim.port` grows from a v3
skeleton into THE SEAM CONTRACT the adaptq FF tier is built against, plus a
Born-decision fast-path speedup, selfcheck-once, and a `PAULI_EXPECTATION`
round-trip fix. `sample_barrier` / `sample` streams are byte-identical throughout
(self-check-gated where applicable); `ENGINE_VERSION` is unchanged
(`xtim-engine-9`) — no pinned decision-stream hash moves with this release.
adaptq's frame-forwarded segmented-simulation arc (`feat/frame-forwarded-sim`,
T1-T8) is the consumer of everything in this release.

### Added
- **`xtim.port` v3** (`Consume`, `compile()`, `Segment`, `Result`) — declared-
  consumption segment surface: dets/decisions-only routes to the bare
  `TwirlSampler.sample()` (in-place reseeding, no reconstruction);
  `groups`/`residual_law` route to the record surface (per-shot `group_id` +
  per-group arrays: keys/first_shot/sigmas/dets/decisions/coins/plans), backed
  by a compact internal record representation (interned plan id + prefix/σ
  words + coins in place of ~600 B serialized plan-key bytes) that is byte-
  identical to the legacy serialized-key groups it replaces (port-v3 T1/T2).
- **THE SEAM CONTRACT** (`xtim/port.py` module docstring, `Segment.run`) —
  ratified statement that only classical data ever crosses an adaptq segment
  seam (the per-shot Pauli frame, the boundary partial-syndrome channel for
  non-CSS seams, the charge-matched T_phys frame advance, Born/driven-bit
  relabels); the engine holds no state between calls. `frame_in` remains a
  loud `NotImplementedError` and `frame_out` remains always `None` by design —
  under the ratified contract no engine-side frame-injection surface exists to
  implement (FF T5, docstring-only; no code-path change).
- **`Segment.run` driven `input_bits`** — per-shot driven bits for IF-driven
  circuits (row-gather semantics; stream-neutral on IF-free circuits) (port-v3
  T5).
- **Selfcheck-once** — the oracle selfcheck window (default 2000 shots: the
  first N fired shots re-derived through the full σ-path and matched exactly)
  now runs once per sampler lifetime instead of re-arming on every seeded
  `set_seed()` call. Stream-neutral: a spent sampler is exactly a
  `selfcheck=0` sampler, byte-identical to the window-on path. Two-call
  cultivation-d5 timing: call1 ~80 ms (window + cold plan laws) / call2
  ~1.7 ms (was ~53 ms pre-fix).

### Performance
- **Born-decision fast-path**: Born-decision circuits now run the fast abelian
  σ-path (`need_state` decoupled from `has_born_dec_`) and feed the existing
  Born-probability memo instead of a per-shot canonicalize/collapse detour.
  Noisy H+T: 5.82 → 5.27 µs/shot at 40k shots/call, 2.33 → 2.08 µs/shot at
  200k. Byte-identity maintained: the fast-σ == structural-σ self-check stays
  active; non-Born / exact-tier streams are byte-identical, and for a
  kappa=0 diagonal plan the Born DECISION stream itself is byte-identical to
  the pre-change structural path (collapsed amplitudes are invariant to the
  fair coins) — verified 40k H+T + 5-Born-qubit shots, 5σ, m_out weights
  exact.
- Port groups/record-surface (port-v3 T2, cultivation_d5, 100k shots/call,
  plan-warm): `sample_barrier` 5.42 → 4.54 µs/shot, `record_groups`
  3.84 → 1.11 µs/shot, end-to-end 9.26 → 5.65 µs/shot.

### Fixed
- **`PAULI_EXPECTATION` serializer round-trip** — `resolve_branches_text`
  emitted the space form (`PAULI_EXPECTATION 0 X0`) but the parser requires
  the paren form (`PAULI_EXPECTATION(0) X0`); any payload-carrying circuit
  using `resolve_branches_text` (adaptq's exact tier) failed to re-parse its
  own resolved output.

### Notes
- `ENGINE_VERSION` (`xtim-engine-9`) is unchanged across this entire release —
  no consumer `.ref` cache invalidates.
- This is the engine state the adaptq frame-forwarded (FF) segmented-
  simulation arc is built and byte-gated against.

## [2.7.0] - 2026-08-03

**Public engine cut** — consolidates the 2.6.x line for release: the latest
engine, correctness- and speed-verified, with no orchestration layer. `v2.x`
will be maintained until `v3.0`; **v3 will simplify the API** (a declared-
consumption segment surface currently in development replaces some of today's
lower-level sampler plumbing). If you are starting fresh, the `v2.x` API below
is stable and supported.

### Highlights

- **In-place `set_seed`** — seeded `sample()` / `sample_barrier()` calls no
  longer reconstruct the sampler; the ~180 ms per-call rebuild tax is removed.
  Streams after `set_seed(s)` are byte-identical to a freshly constructed
  sampler at seed `s` (zero tolerance; Python + C++ seed-parity oracles).
  Cultivation-d5 at 2000 shots/call: ~90 µs/shot → ~9–11 µs/shot warm.
- **Exact-residual read accessors** (all additive; from 2.6.0):
  `BarrierBuffer.sigmas()` / `sig_bits` (bulk certified-generator syndrome
  rows), `FramedSuperposition.port_sigma_law` (combination-tracked port
  decomposition), `certified_symplectic`, `BarrierBuffer.plan_structure`,
  `born_dec`. Residual signs are resolvable from exact engine reads; state
  materialization is demoted to a test oracle.
- **Serializer round-trip fix** — `PAULI_EXPECTATION` instructions survive
  `resolve_branches` text round-trips (emitted as `PAULI_EXPECTATION(label)`,
  matching the parser).
- **Selfcheck window runs once per sampler** (default-behavior change) — the
  `selfcheck` oracle window (default 2000: the first N fired shots also run
  the full sigma path and must match exactly) now runs ONCE per sampler
  lifetime, on the first `sample()`/`sample_barrier()` call after
  compilation, and never re-arms on later calls. Previously the in-place
  `set_seed` reset the window's gating counters, so EVERY seeded call
  silently re-paid the slow oracle window. The window is a stream-neutral
  verification overlay: sampled bytes are byte-identical with the window on
  or off (oracle: `tests/test_selfcheck_once.py`; zero tolerance).
  Recompiling re-arms it. Default-settings two-call behavior on cultivation
  d5, 2000 shots/call, same seed: first call ~80 ms (oracle window + cold
  plan laws), second call ~1.7 ms (~0.9 µs/shot — the fast path); pre-fix
  the second call re-paid the window at ~53 ms (~26 µs/shot).

### Removed

- The in-flight v3 `xtim.port` module (and its top-level `Consume` /
  `compile_segment` / `Segment` / `Result` exports) is not part of this
  release; it is mid-flight API and ships at v3.0.

### Internal / experimental

- The compact-record C++ machinery (plan interning + per-shot prefix words in
  the record sink) remains in the engine. It is engine-internal, default-off
  (`compact=False`), and inert without the v3 port surface; it is a bijective
  re-encoding of the legacy plan bytes with byte-identical group keys.

### Compatibility

- `sample_barrier` streams are byte-identical to 2.5.0; `ENGINE_VERSION` is
  unchanged. No API removals other than the unreleased v3 port surface.

### Performance (release verification)

Measured on the installed 2.7.0 wheel in a clean env (linux x86_64, python
3.12, warm median-of-5, host load ~3.0 — a LOADED host; quiet-host numbers
run lower). Circuit: `cultivation_d5` (42 qubits, 107 detectors).
`selfcheck=0` legs (the fast path). With the once-per-sampler selfcheck
window (above), a DEFAULT sampler is in the same speed class from its second
call on — only the first call after compilation pays the oracle window:

| leg | result |
|---|---|
| record `sample()` 200k shots/call, same-seed warm | 0.72 µs/shot |
| record `sample()` 2k shots/call, same-seed warm | 0.68 µs/shot (1.35 ms/call) ¹ |
| record `sample()` 200k shots/call, fresh-seed warm (plan-law tax) | 4.2 µs/shot |
| `sample_barrier()` 200k shots/call, same-seed warm | 6.1 µs/shot |
| seeded zero-shot `sample(0, seed=…)` call | 0.04 ms |
| bare detector `sample()` 200k shots/call warm | 0.64 µs/shot |
| DEFAULT sampler (`selfcheck=2000`), 2k shots/call, same seed — call 1 / call 2 | ~80 ms / ~1.7 ms (~0.9 µs/shot) ² |

¹ Quiet-host measurement at `selfcheck=0` — the engine's best case, not the
typical loaded-host figure (the other rows were taken at host load ~3.0).

² Once-per-sampler window (this release): call 1 pays the oracle window plus
cold plan laws; call 2 and every later call run the fast path. Pre-fix, every
seeded call re-paid the window: call 2 measured ~53 ms (~26 µs/shot).

The seeded zero-shot call at 0.04 ms confirms the ~180 ms per-call rebuild
tax is gone in the shipped wheel (pre-2.6.1 this call cost a full sampler
reconstruction).

## [2.6.1] - 2026-07-31

**In-place `set_seed` (speed-kill T1/T2)** — seeded `sample()` / `sample_barrier()`
calls no longer reconstruct the `TwirlRecordSampler`.  No API or byte-stream change;
`ENGINE_VERSION` unchanged.

### Added
- **`TwirlRecordSampler::set_seed(uint64_t)`** — reseed the twirl record sampler
  IN PLACE, resetting exactly the seed-dependent state a fresh construction
  initializes (noise-event stream via `DiagErrorSampler::reseed`, per-shot counter-
  RNG base, dedicated obs/fallback/decision streams, rf shot counter, cumulative run
  counters) without reconstructing anything.  Previously every seeded `sample()` /
  `sample_barrier()` call tore down and reconstructed the entire
  `TwirlRecordSampler` — a ~176–184 ms circuit-compile-scale fixed cost on
  cultivation d5 that appeared as a "~90 µs/shot floor" at 2000 shots/call.
  Returns `false` without mutating anything in the one case where in-place reseeding
  is physically incorrect: synthetic-channel mode (unreachable from the Python
  bindings, which always compile circuit channels).  Byte contract: streams emitted
  after `set_seed(s)` are byte-identical to a freshly constructed sampler at `s`
  (zero tolerance; oracle: `tests/test_set_seed_equivalence.py`).
- **`PyTwirlSampler::reseed_in_place(seed)`** — wrapper that routes both seeded
  call sites (`sample`, `sample_barrier`) through `set_seed` + gauge-RNG re-tie
  (identical seed-tie as `rebuild()`), falling back to the retained `rebuild()` with
  a loud `stderr` note on the (unexpected) synthetic-channel path.
- **C++ seed-parity test** (`cpp/tests/test_set_seed_parity.cpp`, REPORT() harness,
  wired via `cpp/CMakeLists.txt`) — asserts `ctor(seed=s)` streams equal
  `ctor(1)+set_seed(s)` streams for `s ∈ {2, 7, 5_000_000_007}` at the C++ API
  level, pinning seed-value-dependent parity that the Python suite structurally
  cannot (Python ctors fix seed=1).
- **`cpp/CMakeLists.txt`** — standalone CMake build for the xtim C++ engine
  (mirrors `qec_library/cpp/CMakeLists.txt` pattern): static `qeccore` library
  linking all `cpp/src/*.cpp`, `qeccore_test` helper macro, `enable_testing()`.

### Performance
- Cultivation d5 warm throughput at 2000 shots/call: was ~97.5–115.8 µs/shot
  (dominated by the ~180 ms per-call rebuild); now **~9–11 µs/shot** (~10–12×
  speedup) — the rebuild tax is eliminated.
- At 200k shots/call the bare `sample()` record is load-sensitive: measured 4.03
  µs/shot (median, load ~3) in the T4 closeout window; 0.70 µs/shot was a T1-era
  minimum that did not reproduce under normal load.  Quiet-host re-measurement
  pending (quiet_repin flow).

## [2.6.0] - 2026-07-31

The **exact-residual engine surface** (the exact-residual-retirement arc,
E1/E2/E3 + T5 step 0). Everything is ADDITIVE: no existing method or signature
changed, `sample_barrier` streams are byte-identical to 2.5.0, and
`ENGINE_VERSION` is unchanged — no pinned decision-stream hash moves with this
release. The new surface is what lets a consumer resolve every residual sign
from exact engine reads (bulk σ + combination vectors) and classify
indefiniteness per plan, with state materialization demoted to a test oracle.

### Added
- **`BarrierBuffer.sigmas()`** — bulk per-shot σ words, `uint64 (shots,
  ceil(sig_bits/64))`, a COPY of the internal word store; bit b of shot i =
  `(row[b>>6] >> (b&63)) & 1`; tail bits above `sig_bits` are zero by
  construction. Companion read-only property **`BarrierBuffer.sig_bits`**
  (= certified-generator count).
- **`FramedSuperposition.port_sigma_law(wires)`** — combination-tracked port
  decomposition beside `port_signature` (existing bytes untouched): per
  surviving port check, in `port_signature`'s exact sorted order, its
  combination vector over the certified generators (packed in the `sigmas()`
  word layout), the reference sign bit, and the carried logical pair rows
  (M·P law, identical to `port_orbit_operators`). Exact law:
  `sign_j(shot) = ref_signs[j] XOR parity(comb[j] AND sigma_shot)`.
- **`FramedSuperposition.certified_symplectic()`** — the certified stabilizer
  generators as full n-qubit symplectic rows + exact phases; row b == σ bit b
  == `port_sigma_law` comb bit b.
- **`BarrierBuffer.plan_structure(plan_key_bytes)`** — a plan key parsed into
  the residual normal form (prefix / a-mask / cz list) plus the plan's
  `ShotLaw` dump (det signs, coin masks, kernel masks/reps); pure exposure of
  the structures `materialize_shot` already computes. The plan-key byte format
  is now documented as a standalone spec (T3 report §3).
- **`BarrierBuffer.born_dec()`** (`TwirlRecordSampler::born_dec_ops`) —
  pure-data exposure of the Born-measured decision operators (LOGICAL/ANTI
  class, declaration order): operator supports/phases, decision indices, and
  inversion bits; raw outcome of op j is `coins(i)[r+kappa+j]`.

### Changed
- **Plan-key parser hardening**: `plan_structure` accepts arbitrary caller
  bytes (unlike `materialize_shot`, which replays sampler-serialized keys
  only), so an out-of-range cz wire index now raises `std::invalid_argument`
  before law construction.

### Notes
- Byte gates held throughout: the consumer's full 6-leg decision-stream byte
  gate (noiseless-Z/Y + noisy-Z, instrument + exact tiers) passes against this
  build with pins unmoved, and the exact-tier streams are unchanged.
- New test files: `tests/test_barrier_sigmas.py`, `tests/test_port_sigma_law.py`
  (incl. corruption probes + self-certifying out-of-span gap accounting), and
  `tests/test_plan_structure_classifier.py` (classifier prototype + the born5
  `born_dec` contract tests); vendored workload data under `tests/data/` with
  provenance headers.

## [2.5.0] - 2026-07-30

Engine-generation sync: the standalone package now ships the **record / barrier /
decision sampler** engine (the decoder-feedback arc), replacing the 2.4.0
detector/observable fast sampler. This is the engine surface consumed by adaptq
(M2b/M3 validation suite); the package is a wholesale sync of exactly that
validated engine, with the standalone packaging (setup.py/pyproject/MANIFEST/
smoke_test/examples) preserved.

### Added
- **Record-sampler engine API** on `xtim._xtim.TwirlSampler`:
  `sample_barrier(shots, seed)` (returns a `BarrierBuffer`), `output_wires()`,
  and the `num_decisions` property — the DECISION-carrying record path absent in
  2.4.0. `sample(shots, seed, input_pauli)` accepts a per-call input Pauli.
- **`BarrierBuffer`** methods: `dets()`, `decisions()`,
  `record_groups() -> (gids, first_occ, keys)`, and `state(i)` (per-shot
  re-materialized `FramedSuperposition`, for reproducible twirl-split copies).
- **Port-contract / branch-resolution engine functions** (new C++ TUs
  `port_contract.cpp`, `resolve_branches.cpp`): `input_qubits(...)`,
  `bits_total_width(...)`, `resolve_branches_text(...)`, and
  `carried_symplectic_action(...)`.
- **State-object methods** on `FramedSuperposition`: `port_orbit_operators(wires)`,
  `pauli_expectation_xz(...)`, `apply_clifford(...)`, `measure_pauli(basis, q, u)`
  (Born sign-sector projection) — the port-projection surface for twirl-split.
- **Full `xtim.twirl` engine API** (was a 4.3 KB stub in 2.4.0, now the complete
  decision-carrying compiler): `compile_twirl_sampler` (with disk cache +
  `skip_refused_observables`), `compile_twirl_sampler_from_state`,
  `PartitionedTwirlSampler` (IF-branch blocks), and the state constructors
  `bare_state_of` / `input_state_of` (+ the underscore-prefixed variants and
  `_project_bare_onto_syndrome`). These are re-exported from the top-level
  `xtim` package.

### Changed
- The twirl sampler generation changed 2.4.0 → 2.5.0 (record/barrier/decision
  path). The engine identity string is unchanged (`ENGINE_VERSION =
  "xtim-engine-9"`), so the decision-stream bytes are **identical** to the engine
  adaptq pinned against: adaptq's full byte gate (6 legs: noiseless-Z/Y and
  noisy-Z, each instrument + exact) passes against this build, sha256-for-sha256.
- Native vs portable wheels: the `-march=native` (source/dev) and
  `QECCORE_PORTABLE` (manylinux wheel) builds produce the **same** decision-stream
  hashes — the byte gate passes identically against both. No wheels-vs-native
  stream divergence on the validated paths; adaptq's pins hold for either
  install stream.

### Notes
- `ENGINE_VERSION` is unchanged, so this bump does not by itself invalidate any
  pinned byte stream; it does refresh consumer disk-extraction caches that fold
  the package version into their key (a one-time, desired cold rebuild).
- Two reserved binding names (`dem_column_layout`, `if_driving_global_offsets`)
  remain exported (zero consumers); kept to hold the binding byte-identical to
  the validated engine rather than hand-edit the surface.

_Engine surface consumed by adaptq (M2b/M3 validation suite)._

## [2.4.0] - 2026-07-27

### Changed
- Default engine changed from `"exact"` to `"auto"` (auto = twirl when eligible, exact
  otherwise; `return_expectations`/`return_measurements` transparently served by exact under
  auto; `QEC_NO_TWIRL` kill switch unchanged). Explicit `engine="twirl"` still refuses those
  sample-time arguments loudly.
- Twirl-engine sampling streams changed (per-shot derived coins — routing-independent: repeated
  same-seed `sample()` calls are now byte-identical between cold and warm runs; distributions
  5-sigma-gated vs exact and unchanged in law).

### Performance
- Plan caches (`TwirlPlanCache`/`PprPlanCache`) and fast-route index (`SharedFastIndex`) persist
  across `sample()` calls — warm cultivation_d5 ~8.5 µs/shot (cold) → ~1.85 µs/shot (warm,
  4.6× speedup measured).
- Devirtualized detector-pack sink (`DetsPackSink`) replaces `std::function` shot sink — word-
  level XOR copy for identity-contiguous circuits.
- Transposed readout-flip application: column-form ctz XOR loop (O(fired × CW) vs O(nchan × RFW)).
- Fixed-point per-shot rf coins (one integer compare per slot vs double draw+branch).
- Selfcheck oracle off by default; set `QEC_TW_SELFCHECK=<n>` to enable the oracle window.

### Added
- `XtimPerformanceWarning` — emitted (once per reason per process) when `engine="auto"` falls
  back to exact. Importable from `xtim`.
- `channel_report()["sink_kind"]` — `"dets_pack"`, `"barrier"`, or `"null"`.
- `channel_one_counts()` on the compiled twirl sampler — per-channel fired counts (all-zero on
  the default `selfcheck=0` fast path; active with `QEC_TW_COUNT=1`).

### Removed
- Dead `std::function` `ShotSink` machinery and `set_shot_sink()` (superseded by `DetsPackSink`).
- Dead `coll_gen`/`cu` RNG members (hot path uses per-shot `ShotRng`).

### Fixed
- **`engine="twirl"` / `engine="auto"` produced incorrect detector statistics on circuits
  with Y-basis measurements** (e.g. `color_code:memory_xyz`) — present in the published
  2.3.0 opt-in twirl engine (measured at its exact source commit), latent there only
  because the default engine was `exact`. Root cause: the noiseless reference bits
  (`chan_ref`) were derived from the record-product's Pauli phase, which accumulates a
  spurious −1 for even products of Y-measurements
  whose record-product contains an even number of MY measurements (Y = iXZ → phase 2 = −1,
  giving `ref=1`). Every shot in a noiseless circuit then fired those detectors. Fix:
  `chan_ref` is zeroed for all detector/observable channels after the readout-flip layer build,
  because for self-contained circuits the bare state IS the noiseless reference. MX/MZ circuits
  were unaffected (X and Z Paulis carry phase 0 or 2 but always cancel correctly in products).
- `XTIM_QUIET=1` now suppresses all three `[twirl_records]` informational banners
  (circuit-channels, disk-LOADED, disk-SAVED). The disk-LOADED banner was previously
  unguarded.
- `QEC_TW_PROF=1` now emits a complete segment ledger: `coins/rf/post/sink/draw/prolog/nrf/probe/wall`
  alongside `key/find/emit/pair` — the accumulator fast path probes were missing from the
  ported build, leaving ~0.55 µs/shot unattributed on the warm cultivation_d5 pass.
- `__version__` fallback string updated to `"2.4.0+dev"` (was `"2.3.0+dev"`).

## [2.3.0] - 2026-07-16

A fast detector/observable sampling engine — the stabilizer-twirl record sampler —
selectable per compile, **opt-in this release** (the default engine is unchanged).
No breaking changes; the exact engine, `.ref` format, and all existing sampling
streams are byte-identical to 2.2.0.

### Added
- **`Circuit.compile_detector_sampler(..., engine="auto"|"exact"|"twirl")`** — a new
  fast engine that samples the SAME deterministic detector and observable statistics
  as the exact engine, **9–28× faster** on the bundled protocols (measured, quiet
  machine): cultivation_d5 18×, code_switching 25×, distillation 28×, the CH-class
  circuits 9×. It compiles each noise pattern's end-of-circuit residual into a cached
  "plan" and emits record bits from closed-form channel algebra instead of evolving
  the state.
  - `engine="exact"` (the default) is the legacy engine, unchanged.
  - `engine="twirl"` selects the fast engine explicitly.
  - `engine="auto"` uses the fast engine when the circuit is eligible and silently
    falls back to exact otherwise — never an error, never a silent change of result.
  - The returned sampler exposes `engine_report()` (which engine, and why).
  - `QEC_NO_TWIRL=1` forces the exact engine everywhere.
- **Exactness contract** (5σ-gated against the exact engine on every bundled circuit,
  and against Stim): every deterministic detector's frequency — jointly, so
  **post-selected and decoded logical error rates are exact** — and logical-observable
  statistics including their joint with the detectors. Noisy measurements (`M(p)`),
  deterministic record inverts, and CH-class (controlled-H) circuits are all handled.
- **`xtim.load_example("...")`** now includes the twirl fast-sampling walkthrough
  (`examples/twirl_fast_sampling.py`).

### Notes
- **Declared semantics.** Gauge detectors (rare) are declared fair coins under
  `engine="twirl"`; a circuit containing them is routed to the exact engine under
  `engine="auto"` rather than sampled with declared semantics silently. Observables
  whose record operator anticommutes with the code (e.g. some byproduct-frame
  readouts) have no exact fast-engine channel and are routed to exact under `auto`.
- **Byte streams.** The fast engine's raw per-shot bytes differ from the exact engine
  for the same seed (the distributions are what match, and are gated); use
  `engine="exact"` when you need byte-for-byte reproducibility against older runs.
- Optional on-disk plan cache (`disk_cache=`) makes a `p`-sweep or rerun compile the
  plans once.

## [2.2.0] - 2026-07-10

Large-circuit compile speed and a hardened reference cache. No API change; all
sampling streams are byte-identical to 2.1.0; `.ref` format v4 and the engine
identity are unchanged.

### Changed (performance — no API, behavior, or sampling-stream change)
- **Large-Clifford-circuit compile is 15–25× faster.** The reference-state compile
  and error-propagation table paid dense-n² costs per noise alternative and per
  stabilizer generator on wide deferred circuits; a stock d=11, r=7 surface-code
  memory now compiles in ~9 s (was ~4 min), d=7 in ~1 s (was ~11 s). All sampling
  streams are byte-identical; protocol-circuit compile and per-shot speed unchanged.
- **Noise-value-blind reference cache.** The `.xtim_cache` key now masks noise
  probability values (which the reference provably cannot depend on), so a
  `Task(p=...)`/`scale_noise` sweep compiles the reference once and every
  p-point is a cache hit. Noise structure/placement stays in the key. (Existing
  cache entries under the old raw-text key are simply orphaned — delete the
  cache dir to reclaim the space.)

### Fixed (hardening from a full-engine correctness audit; none released)
- **Stack corruption compiling circuits wider than 4096 qubits**: the CX error-
  conjugation sized a stack snapshot buffer from the (new) lazy CZ-layer's word
  count and masked two entries unconditionally — an out-of-bounds write for
  targets ≥ qubit 4096. Caught by a new wide-circuit regression test; the
  snapshot is now skipped entirely for the lazy (couplings-free) form.
- **Reference-cache frame guard**: every cache entry now stores a signature of
  the normalized+deferred circuit structure (noise values excluded) and every
  hit re-derives it from the caller's text — a cached reference compiled for a
  circuit that normalizes to a different terminal frame is rejected and falls
  back, closing the frame-divergence class for detector/expectation-bearing
  circuits where the cheap-verify invariants alone are weak.
- **Rank guard in `PreparedAffine::prepare`**: the full-column-rank precondition
  the fast phase pin relies on is now enforced (loud throw) instead of assumed.
- **`collect` beta memo** keys on the deferred-frame signature (was: noise-
  stripped text), so tasks differing in noise *placement* each get their own
  frame-adequacy check; p-sweep sharing unchanged.
- **Noise-line recognition** in `_strip_noise`/the cache key now matches the
  parser's tokenizer (lowercase names, `[tag]` suffixes, tab separators).
- New test batteries: packed-matrix differential fuzz (word-level compaction vs
  per-bit reference; lazy vs materialized CZ-layer equivalence; trailing-bit
  invariants), and an xtim-vs-Stim distribution-parity harness on surface-code
  memory circuits (marginals, pairwise correlations, random parity masks, 5.5σ).

### Changed (docs only)
- **README rewritten as a lean landing page** (~8× shorter): pitch, install,
  a sixty-second worked example with real output, a measured 15-to-1
  distillation curve (new `docs/assets/distillation_curve.png`), an
  "xtim or Stim?" table, and a links table. Nothing was deleted — the
  displaced material moved to two new pages, `docs/xtim_scoring.md` (the full
  `diagnose()` walkthrough + the four fidelity-arithmetic pitfalls) and
  `docs/xtim_practical_notes.md` (CLI, caching, seeding, programmatic
  building, scope in practice), and the protocol throughput table moved to
  `examples/README.md`.

## [2.1.0] - 2026-07-07

Controlled-Hadamard support, end to end. Existing circuits are unaffected — every
diagonal-class (T/CS/CCZ) sampling stream is byte-identical to 2.0.x, `.ref` format
stays at v4, engine identity stays `xtim-engine-9`.

### Added
- **Controlled-Hadamard (`CH`/`CX_H`) is now in the simulable class** — compile,
  error propagation, and sampling, including sign-exact `PAULI_EXPECTATION` on
  error-carrying shots. Internally each error is propagated past non-diagonal
  third-level gates as a Pauli times a list of commuting π/4 rotations
  (Pauli-product-rotation form) and recompiled into a Clifford tableau for the
  shot loop. The acceptance criterion is unified and unchanged in spirit: a
  circuit is accepted iff its folded π/8 axes mutually commute (generalized
  T-depth one); a non-commuting fold rejects loudly, never silently.
- **H-eigenstate cultivation ships as a bundled example**:
  `xtim.load_example("ch_cultivation")` (3 rounds of logical-Hadamard
  measurement via CH, χ=2), plus a runnable walkthrough
  `examples/h_cultivation_walkthrough.py` (asserts deterministic detectors and
  fidelity, and shows the alignment rules — it is covered by the packaged tests).
- **`diagnose()` and expectation frames now serve CH circuits** (DEM export
  still intentionally rejects them — post-selection/diagnose only, see
  `docs/xtim_dem_reject.md`).
- **Docs**: a reader-journey index (`docs/README.md`) and a precise statement of
  the simulable class (`docs/xtim_simulable_class.md`); README gains CH in the
  dialect table and a corrected class-reject example. The class-reject error
  hint now explains the alignment condition (CH itself is supported; a fold
  only rejects when π/8 axes fail to commute).

### Fixed
- **Wide-circuit compile crash**: circuits deferring to ≳2100 effective qubits
  (e.g. a stock d=11, r≥7 surface-code memory) crashed with a spurious
  "wrong X/Z support" error while pinning stabilizer-generator phases — the
  amplitude ratio was materialized as a double, whose 2^(−k/2) modulus
  underflows. The ratio is now taken exactly in the engine's exact-phase
  representation; such circuits compile and sample (χ=1) with all existing
  streams byte-identical.
- **README speed guidance corrected by measurement**: the "~10²–10³× slower
  than Stim" figure conflated text-record I/O with engine cost. Measured
  same-circuit, same-core, packed records: Stim is ~1.3× faster on the
  measurement channel and ~2.4–2.9× on the detector channel at d=7–11; the
  real asymmetry is compile time (Stim: milliseconds; xtim's reference-state
  compile: seconds at d=7 to minutes at d=11).

### Changed (build hygiene only, byte-identical)
- Removed dead engine code (`gadgetize_level3` and orphaned residue helpers)
  and stopped compiling dev/research-only sources (`symplectic.*`,
  `stim_io.*`) into the wheel. Leaner build, no functional impact.

## [2.0.1] - 2026-07-05

A packaging fix so `xtim` installs alongside any recent numpy. No API, behavior,
`.ref`-format, or engine change — the Python API and sampling output are identical
to 2.0.0.

### Fixed
- **Installs with numpy ≥ 2.5 again (the runtime `numpy` upper bound is removed).**
  2.0.0 shipped with `numpy<2.5` because the old manylinux2014 build image (GCC
  10.2) could not install/build numpy ≥ 2.5. The Linux wheels now build on
  `manylinux_2_28` (GCC 12), where numpy ships prebuilt wheels, so the dependency
  is back to `numpy>=1.26` (unpinned upper bound). Windows/macOS wheels are
  unchanged.
- **Engine headers are now clean under GCC 12** (explicit `#include <cstddef>` for
  `size_t`, which older GCC/Clang/MSVC pulled in transitively). Byte-neutral —
  codegen and sampling streams are unchanged; verified with a local gcc-12.4
  syntax pass over every translation unit and the full test suite.

### Changed (packaging only)
- **Linux wheels: `manylinux2014` → `manylinux_2_28`.** The glibc floor for the
  Linux wheels rises from 2.17 to 2.28 (RHEL 8 / Ubuntu 18.10+ / any 2019-or-newer
  distro). Wheel performance is unchanged — same portable `-O3` build flags, only
  the base image (and its glibc/GCC) differs.

## [2.0.0] - 2026-07-05

Declared byproduct frames, exact determinism, and a fidelity-scoring toolkit.
`PAULI_EXPECTATION` expectation values are now sign-exact and shot-independent;
`Circuit.postselected_logical_error_rate` scores a post-selected `1 − F` (or an
observable LER) with automatic detector post-selection; and two opt-in helpers
(`extract_frame`, `fidelity_from_logicals`) remove the tedious parts of scoring a
magic state's fidelity. Breaking changes are confined to the per-shot expectation
sign streams and one bundled circuit's detector layout — `.mean()`/LER values are
unaffected.

### Added
- **`PAULI_EXPECTATION(i) <Pauli> rec[-k]…` — declared byproduct Pauli frame.**
  A trailing `rec[-k]` target list (the same `rec[-k]` grammar as
  `OBSERVABLE_INCLUDE`) can now be appended to any `PAULI_EXPECTATION` declaration
  to name the measurement records whose parity fixes the sign of ⟨P⟩. With a
  declared frame the engine folds the record-parity into the per-shot sign, so
  `exps[:, i]` is **sign-constant** in the noiseless case: `|⟨P⟩|` is the
  frame-independent physical magnitude; the SIGN is the convention in the frame
  where the declared records all read 0; and the target ⟨P⟩ is the mean of the
  per-shot signed values.
- **`Circuit.postselected_logical_error_rate(*, p, shots, seed, target_k, p0)`**
  and the `PostselectedLER` result type. Scores a post-selected logical error rate
  — `1 − F` in fidelity mode (summing the declared `PAULI_EXPECTATION` columns) or
  the flipped-observable fraction in observable mode (`OBSERVABLE_INCLUDE`) — with
  the automatic post-selection everyone forgets: it rejects every shot in which a
  deterministic detector fired. The result carries `value`, `sem`, `acceptance`,
  `kept`, `mode`, `upper_bound` (rule-of-three 3/kept when zero errors are
  observed), and `warnings` (partial-support-proxy, no-noise-floor, low-acceptance
  advisories). The name makes the post-selection explicit.
- **`extract_frame(circuit, operator, *, minimal=True)` — byproduct-frame helper.**
  Given a logical Pauli operator string, returns the `rec[-k]` offsets whose parity
  pins its sign, ready to paste into a `PAULI_EXPECTATION` line (`minimal=False`
  gives a spanning full-readout representative). Raises `XtimError` when the sign is
  not record-pinnable; returns `[]` when the sign is already fixed by the bare state.
- **`fidelity_from_logicals(circuit, logicals, *, p, shots, seed, p0)` — true
  full-support fidelity.** Given the k logical generators
  `{0: {"X": "…", "Z": "…"}, …}`, it validates the algebra, generates all 4^k
  logical Pauli products, extracts each product's frame, and scores the true
  full-support `1 − F` (auto post-selected). Unlike scoring only the columns a
  circuit happens to declare, this cannot silently miss a Pauli with nonzero ideal
  expectation. Opt-in; the default path remains `postselected_logical_error_rate`.

### Changed
- **engine-9** (`xtim-engine-9`, bumped from engine-8). The engine now computes the
  EXACT set of sign-controlling records (bare-state stabilizer anticommutation
  analysis) and refuses (`XtimRejectError`) if a declared frame is missing a
  required record or includes a spurious one. A column with no solvable fixed frame
  (a genuine free/logical read, e.g. a destructive `MPP` readout) is not refused —
  it reports `signed_value = nan`, `sign_constant = False`, `frame_hint = None`.
- **`_discover_frame` retired — `xtim.diagnose` determinism is now EXACT.** The
  statistical sign-inference pass that inferred the byproduct frame from a sample of
  noiseless shots is removed; detector determinism is decided exactly (the
  parity-Pauli lies in the bare state's stabilizer, via `detector_determinism`)
  rather than by noiseless sampling, which mislabeled measurement-noise-fed
  detectors. `signed_value` is now exact and shot-independent (computed from the
  declared frame, verified analytically by the engine); `frame_hint` now echoes the
  DECLARED `rec[-k]` targets, not a statistically-discovered parity.

### Fixed
- **`code_switching_faithful` post-selection is now complete.** The final noiseless
  syndrome-extraction round declares its six Steane stabilizers as detectors, so a
  pure post-selection heralds every single-fault logical error. Its
  post-selected `1 − F` drops from ~5×10⁻³ (a spurious O(p) floor) to ~1.5×10⁻⁶ at
  p=10⁻³ — the fault-distance-limited rate. (See BREAKING: this changes the
  circuit's detector count.)

### BREAKING
- **Expectation streams for `code_switching_faithful` and `miniature_oracle` are
  now frame-fixed.** `code_switching_faithful` (records {16…22}) and
  `miniature_oracle` (record {2}) now declare their byproduct frames in-circuit.
  Their per-shot `exps` columns are sign-constant in the noiseless case; previously
  a noiseless run could flip sign across shots (the old `_discover_frame` inference).
  Downstream code that applied its own sign-correction based on the raw per-shot
  sign must be updated. **`exps.mean()` calls are unaffected** — the mean was
  correctly computed before and remains correct after.
- **`code_switching_faithful` detector count changed (20 → 26).** The added
  end-of-circuit noiseless SE round (see Fixed) declares six more detectors, so the
  circuit's detector sample shape (`sample()` / `.dets` columns) changed. Code that
  indexes this circuit's detector array by absolute position must be updated.
  **Post-selected LER / fidelity values are unaffected** (they improve, per Fixed).

---

## [1.0.2] - 2026-07-03

A performance fix. No API change, no `.ref` format change.

### Fixed
- **The R=0 record-sampling path no longer thrashes on high-pattern-diversity circuits.**
  1.0.1 fixed the expectation-channel (R>0) memo; the memory-mode (R=0) path had the same
  pathology. `code_switching_faithful` R=0 sampling went from ~13 µs/shot (default, warmup-
  bound) to **~0.85 µs/shot** — now ~3× faster than a comparable near-Clifford simulator.
  The R=0 path now uses the same linear-Pauli-correction-chain construction as R>0 (one
  unified builder for both modes).

### Reproducibility (note)
- R=0 same-seed byte streams changed for the affected (high-pattern-diversity) circuits —
  statistically identical, validated; the committed pins were regenerated. Other circuits'
  streams are unchanged.

## [1.0.1] - 2026-07-03

A performance fix for 1.0.0. No API change, no `.ref` format change.

### Fixed
- **Expectation-channel sampling no longer thrashes on high-pattern-diversity circuits.**
  In 1.0.0 the per-shot expectation-channel tree was built path-unique (an exponential
  forced-collapse recursion) and cap-saturated builds were wrongly marked unusable, so a
  circuit like `code_switching_faithful` fell off the fast path: **~21 µs/shot**. The two
  forced-collapse branches at each biased coin are related by a Pauli read off the shared
  frame, so the channel tree now collapses into a linear correction chain (one linear-time
  construction, no per-path recursion) with zero build refusals. `code_switching_faithful`
  is now **~0.61 µs/shot** (≈1.6 M shots/s); the per-shot memo is a net win again. A ctest
  tripwire guards this class of regression.

### Reproducibility (note)
- R=0 (memory-mode) byte streams are **identical** to 1.0.0. R>0 (expectation-mode)
  same-seed byte streams changed (the expectation re-routing) — statistically identical,
  dense-oracle + `stim_parity` validated; pins/goldens regenerated.

## [1.0.0] - 2026-07-03

**One engine.** The framed circuit-level engine now carries ALL modes — memory (R=0)
and expectation (R>0) — at every stabilizer rank χ. The legacy shot loop and the
separate lean state are deleted. No public Python API change; the breaking changes are
in the sampling byte streams, the removed env levers, and the on-disk `.ref` format.

### BREAKING
- **Env levers removed.** `QEC_FRAMED`, `QEC_AB_REDUCE`, and `QEC_AB_OBS` are gone (they
  are now silently ignored). In particular the `QEC_FRAMED=0` "reproduce the old engine's
  streams" contract no longer exists — **there is no env combination that reproduces a
  ≤0.8.0 byte stream.** The one retained cross-check lever is `QEC_FORCE_FALLBACK=1`
  (compile-time): it routes every shot through the engine's full-state general path — an
  independent route through the same physics whose distribution must match the default.
  New performance-only tuning levers (results unchanged): `QEC_TREEPLAN_CAP`,
  `QEC_FRAMED_MEMO`, `QEC_PARTNER_HASH_MIN`.
- **Same-seed byte streams changed.** All expectation-mode (R>0) circuits now run on the
  framed engine, and R=0 streams changed for `cultivation_d5`-class circuits. The changes
  are statistically identical — validated against a dense-statevector oracle and
  `stim_parity --full`; the committed goldens/pins were regenerated. Within-a-build
  same-seed determinism still holds exactly.
- **Expectation semantics: consistent outcome-conditioning.** Observables read on a
  fresh-coin measurement wire are now outcome-conditioned at *every* χ (per-shot values
  are conditioned on the recorded outcomes). Previously some χ>2 paths inconsistently
  reported unconditional marginals. **Means are unaffected** (dense-statevector-validated);
  only the per-shot values on those wires change.
- **`.ref` on-disk format v4** (`REF_FORMAT_VERSION` 3 → 4; new refs drop the anchor
  block). **A v4 `.ref` cannot be read by xtim ≤ 0.8.0.** This release reads v2/v3/v4, so
  upgrading is safe — your existing `.ref` files still load. The reference cache keys on
  the format version, so a stale-format cache entry misses cleanly and recompiles.

### Performance
- Expectation-mode (R>0) circuits now compile framed expectation channels: large-χ R>0
  workloads speed up by orders of magnitude (a χ=16 sweep: 24.3 → 0.11 µs/shot);
  `cultivation_d3_faithful` ~0.64 → ~0.50 µs/shot, `distillation_15_1_3` ~unchanged.
- Memory-mode `cultivation_d5` marginal shot cost ~12 → ~8.7 µs/shot; χ up to 1024 now
  samples at the sub-µs tier-0 rate.

### Changed
- Internal: a single state type (`FramedSuperposition`); the χ=3 capability cliff is gone
  (χ>2 is fully supported); a unified `TreePlan` plans at all χ.
- `Reference` now exposes `.version` (the `.ref` format version of the stored artifact).

## [0.8.0] - 2026-07-01

A sampler-speed release. A new framed circuit-level engine is now the production
default. No public API change, no `.ref` format change (0.7.1 references stay valid).

### Performance
- **Memory-mode sampling is ~1.2× faster** via the new framed circuit-level engine
  (`QEC_FRAMED`, default on; set the env var `QEC_FRAMED=0` for the previous engine).
- **Expectation-mode worst-path shots are 2–10× faster** — pattern-miss shots now take
  an A/B block path with tensor-split expectations instead of a full-state fallback
  (`QEC_AB_OBS`, default on). Output is unchanged, only faster.

### Reproducibility (note)
- **Memory-mode (R=0, no `PAULI_EXPECTATION`) same-seed byte streams differ from 0.7.1.**
  Statistically identical, Stim-validated to 5σ — but the *specific* shots at a fixed seed
  differ. **`QEC_FRAMED=0` reproduces the 0.7.1 streams exactly.** (Same class as the
  0.7.0 lean-gauge note.) Within-a-build same-seed determinism still holds exactly, and
  expectation-mode (R>0) streams are unchanged.

### Fixed
- `XtimDemError` no longer prints "not DEM-expressible" twice.
- Docs: refreshed the `cultivation_d3_faithful` throughput figure (~1 M shots/s after the
  expectation-mode speedup) and corrected a syndrome-rate comment in the DEM guide.

## [0.7.1] - 2026-06-30

An internal cleanup release — **byte-identical sampler output**, leaner source. No
public API change, no `.ref` format change, no behavior change.

### Changed
- **~6.3k net lines of C++ removed**: legacy stabilizer representations, the `extsim`
  subsystem, and ~12 dead bench/oracle apps are gone; research-only code (distance,
  code I/O) split out of the magic engine into a separate library the xtim wheel does
  not link. The magic engine is now ~18k C++. Same wheel, smaller source.
- `chi_max` is now fully retired internally (it was already gone from the public API in
  0.6.0; this removes the last dead internal threading, including the CLI positional arg).
- `REF_FORMAT_VERSION` now correctly reports **3** (was a stale `1`).

### Fixed
- `Circuit.__eq__`/`__hash__` normalize trailing whitespace, so a circuit equals its
  `from_file(to_file(...))` round-trip and its `append`-built twin (`to_file`'s
  documented round-trip now holds).
- `xtim.Reference(circuit)` (a `Circuit` instead of stored `.ref` text) now raises a
  typed `XtimReferenceError` pointing you to `c.compile_reference()`, instead of a raw
  pybind `TypeError`.
- Docs: removed an internal handoff section that was shipping in `xtim_dem_reject.md`;
  fixed the `cube_ccz` row in the examples table; clarified that `xtim.cache_dir`
  defaults to `None` (meaning `.xtim_cache/`) and that `scale_noise` on a noiseless
  circuit is a no-op, not an error.

## [0.7.0] - 2026-06-29

An internal engine refactor: the sampler's execution path is now fully tableau/lean —
the affine `CanonicalStabSum` is gone from sampling (replaced by a lean A/B
factorization). **No public API change and no `.ref` format change** — your 0.6.0
`.ref` files and all code keep working unchanged.

### Changed
- **The sampler is affine-free** (tableau/lean throughout). Internal; verified by an
  exact state-overlap oracle, a stim distribution-equivalence campaign, and the
  whole-branch review.

### Reproducibility (note)
- **Memory-mode (R=0, no `PAULI_EXPECTATION`) same-seed sampling streams differ from
  0.6.0.** The lean factorization changed the B-block gauge, which reorders the
  per-shot RNG byte stream for memory/syndrome circuits — **statistically identical**
  and stim-validated (max |z| = 1.03), but the *specific* shots at a fixed seed are not
  the same as 0.6.0. Expectation-mode (R>0) streams are unchanged. If you pinned exact
  bytes for a memory-mode circuit at 0.6.0, regenerate them; results are equivalent.

### Fixed
- Docs: the post-selected `distillation_15_1_3` infidelity at p=1e-2 is ≈ 3.5e-5 (the
  35·p³ floor), and the un-post-selected baseline is ≈ n·p ≈ 15p (the `Z_ERROR` model)
  — earlier text had ≈5e-5 / ≈10p.
- `diagnose()`'s byproduct-frame HINT now labels the record index as 0-based
  (`parity(0-based records [k])`), so it isn't misread as 1-based (which scores F≈0.5).

## [0.6.0] - 2026-06-29

A new engine (engine-7) with a **breaking** simplification: the `chi_max` cap is
gone. Stabilizer rank χ is now purely a cost knob, never a rejection criterion — the
diagonal-Clifford propagation-class check is the **sole** thing that rejects a
circuit. Plus a compile cache (repeated `sample()` no longer recompiles), a leaner
per-shot path, and two new flagship examples.

### Removed (BREAKING)
- **`chi_max` is removed everywhere.** It was a memory backstop that the engine no
  longer needs, so the argument is gone from `Circuit.compile_sampler(...)`,
  `Circuit.compile_detector_sampler(...)`, `Circuit.detector_error_model_with_reject(...)`,
  the `CompiledSampler`/`CompiledDetectorSampler` constructors, and the `python -m xtim`
  CLI (`--chi_max`). Any output whose stabilizer rank χ is large is still simulated
  **exactly** — it just costs more per shot. The old `XtimRejectError(kind="chi_cap")`
  reject no longer exists; the only rejection is `kind="class"` (a basis change on a
  magic-carrying wire leaves the simulable class). **Migration:** delete any `chi_max=`
  argument — there is nothing to replace it with.

### Changed
- **engine-7** (C-4 parity-native reference construction; FastTODD-only magic
  compiler). Internal — sampling distributions are unchanged (verified by an exact
  state-overlap oracle + a distribution-equivalence gate). The on-disk `.ref` cache is
  internal and self-invalidates on the upgrade.

### Performance
- **Compile cache.** A compiled sampler now holds its compiled program, so repeated
  `sample()` calls on the same object no longer recompile the reference (e.g. a d5
  cultivation went from ~1.5 s per `sample()` call to ~ms after the first).
- **Leaner per-shot path** (the affine amplitude anchor is off the sampling path),
  plus a precomputed active set — measurable per-shot wins on the d5-class circuits.

### Added
- **`distillation_15_1_3`** — a genuine post-selected 15-to-1 magic distillation on the
  [[15,1,3]] quantum Reed–Muller code: transversal `T` with physical T-gate phase noise
  (`Z_ERROR`, baked p0=1e-3) and the 4 X-stabilizers measured as detectors. Post-select
  a trivial syndrome (`keep=lambda dets, meas: ~dets.any(axis=1)`) and the distilled
  infidelity drops to the weight-3 floor ~35·p³ — far below the physical rate. Declares
  ⟨X̄⟩ and ⟨Ȳ⟩, so `Task(target_k=1)` gives a true 1−F vs p curve.
- **`cultivation_d5`** — a d=5 magic-state cultivation T-count / syndrome fixture (no
  `PAULI_EXPECTATION`; syndrome sampling only).
- Both are bundled in the wheel — `xtim.load_example("distillation_15_1_3")` etc.
- A **Highlights** section in the README featuring the d3/d5 cultivation and the 15-to-1
  distillation with measured amortized throughput.

### Fixed
- `scale_noise(circuit)` (a `Circuit` instead of its text) now raises a clear
  `TypeError` pointing you to `.text`, instead of a leaked `AttributeError`.
- The DEM-refusal hint for a high-weight (`>16`) observable now describes the real
  cause (a too-wide twirl-coin-read expansion) instead of misattributing it to a
  `PAULI_EXPECTATION` column.
- An out-of-class circuit no longer prints a misleading "sampling stays exact" cache
  warning before its `XtimRejectError`.
- Docs: honest hardware-dependent throughput figures; corrected the reject example
  (`CH`, not a χ cap); removed a stale `to_stim_text` claim (`Circuit.text` preserves
  the original tokens) and clarified the `sample()` return contract.

## [0.5.12] - 2026-06-24

An engine performance + cache release. **No public API, Python-surface, or
sampling-result change** — pure internal speedups and a larger reference cache.

### Performance
- Reference compilation is dramatically faster for large magic-state /
  distillation references (direct structure-preserving construction): a
  [[15,1,3]]×k distillation reference at k=8 went ~272 s → ~5 s. And disjoint
  magic blocks now reduce independently (stabilizer rank `2^k` instead of `8^k`),
  so multi-block distillation references compile at all (k≥4 was infeasible before).
- Faster warm-cache loading: the on-disk reference format loads without re-deriving
  the frame structure, so cache hits on large references start faster.

### Changed
- Reference cache cap raised **1024 → 16384**: magic states with stabilizer rank up
  to 16384 now memoize (the internal `.ref` cache) instead of recompiling every run.
- Internal engine bump (engine-6, on-disk reference format v3). The reference cache is
  internal and self-invalidates on the upgrade — the first run after upgrading
  recompiles references, then they cache. Nothing is user-written.
- Sampling **distributions are unchanged** (verified by an exact state-overlap oracle
  + a distribution-equivalence gate). For very large references the exact per-shot
  byte sequence at a fixed seed may differ — never a supported contract; statistical
  results are identical.

## [0.5.11] - 2026-06-22

A small fidelity-feature hardening plus a worked multi-magic example.

### Added
- `xtim.collect` also warns when a row's **noiseless fidelity ceiling**
  `(1 + Σβ²) / 2**target_k` is below 1 — the symmetric partner to the existing
  `F > 1` warning. It flags an INCOMPLETE declared `PAULI_EXPECTATION` support (or
  a `target_k` that is too large); the value is returned unchanged. The two
  warnings now bracket a correct `target_k` / column set from both sides.

### Docs
- New worked example `examples/cube_832_ccz.py`: the [[8,3,2]] cube code's
  transversal-CCZ magic state, prepared with stabilizer-measurement + feedback,
  scored to a TRUE 3-qubit fidelity over the complete logical Pauli support
  (noiseless `F = 1`), with a noiseless syndrome round + post-selection that
  suppresses `1−F` from `O(p)` to `O(p²)` (only the physical T gates are noisy).

## [0.5.10] - 2026-06-22

Documentation and examples for the v0.5.9 fidelity feature. **No code change** —
the engine, wheels, and results are identical to 0.5.9.

### Docs
- New worked example `examples/fidelity_curve.py`: a `1−F ± sem` vs `p` curve
  (with a matplotlib plot) built from the `Task(target_k=...)` feature.
- README gains a *"shortcut: a fidelity curve from `collect`"* subsection, the
  examples index lists the new example, and the tutorial notebook shows
  `target_k` scoring (the post-selection-buys-fidelity / acceptance trade-off).

## [0.5.9] - 2026-06-22

A figure-of-merit release: `xtim.collect` now turns declared `PAULI_EXPECTATION`
targets into a publishable fidelity curve directly. Backward-compatible — opt-in
via `Task(target_k=...)`; without it, rows are unchanged.

### Added
- `xtim.collect` computes **true-state fidelity** when a task sets `target_k`
  (logical-qubit count of the target magic state). Rows gain `fidelity`,
  `infidelity`, `infidelity_sem`, `target_k`, `target_beta` — a `p`-sweep drops
  straight into a `1−F ± sem` vs `p` curve. The error bar is `std/√N` of the
  per-shot fidelity scalar (correlation-aware). Completeness of the declared
  `PAULI_EXPECTATION` set is the user's responsibility; gauge columns are refused.
- A `UserWarning` is raised when a row's `fidelity` exceeds 1 beyond sampling
  noise — the loud signal that `target_k` is too small or the declared columns
  are over-complete. The returned value is left unclamped.

### Performance
- `target_beta` (the noiseless target) is computed ONCE per unique noiseless
  circuit per `collect` call instead of once per task, so a `p`-sweep over one
  base circuit pays the `diagnose` cost a single time (~3× faster real-circuit
  sweeps).
- `substitute_var` ~2.8× faster transversal-T reference compile (deferred from
  the v0.5.8 trailing perf note).

### Docs
- The tour shows the `target_k` one-liner for a 1−F vs p curve; `help(xtim.collect)`
  is self-contained; `infidelity` is documented as unclamped (guard
  `max(0.0, infidelity)` before log plots); ndarray row fields note `.tolist()`
  for JSON.

## [0.5.8] - 2026-06-21

A foundation consolidation plus internal performance wins. **No public API,
on-disk format, or sampling-result change** — the consolidation is byte-exact
(default engine behavior, results, and `.ref` format unchanged; verified across a
broad statevector cross-check and the bundled demos to the digit).

### Performance

- **`build_bare_state` ~12 s → sub-second for transversal-`T` states** — much faster
  reference compile for high-distance transversal-`T` circuits.
- **`CHState` overlap ~k× faster** (k = active magic dimension).

Both are compile-time / internal; sampling per-shot speed and results are unchanged.

### Fixed

- **Pathological `cache_dir` no longer leaks a raw error.** A `xtim.cache_dir` the
  filesystem rejects (e.g. an embedded NUL byte) raised a bare `ValueError` from the
  cache store; it now follows the documented contract — `XtimCacheWarning` + the exact
  deduced bare state.
- **`scale_noise(..., p0=0)` raises a clear `ValueError`** instead of a raw
  `ZeroDivisionError`.

### Internal

- Foundation cleanup (−269 LOC): `clifford_frame → clifford_tableau` rename, a unified
  reference-χ ceiling header (`ref_config.hpp`), and dead-code removal — byte-exact.
- The packaging build now lists the TU-`#include`d engine sources as `depends`, so an
  incremental rebuild of the exported tree can't serve a stale `.so`.
- Docs: noted that `PAULI_EXPECTATION`'s parenthesized argument is an integer label,
  not the target value.

## [0.5.7] - 2026-06-21

A **caching-only** improvement (no sampling-result or public-API change): higher-χ
references now memoize instead of recomputing every run.

### Fixed

- **References with 64 < χ ≤ 1024 now cache.** The reference-compile verification
  battery sampled the just-built reference at a hardcoded χ≤64, while construction
  built it at χ≤1024 — so a reference whose stabilizer rank fell in (64, 1024] (e.g.
  ≳7 independent magic qubits) passed construction but **failed the verification
  gates**, was rejected, and fell back to the exact *deduced* bare state, recompiling
  on every run (with an `XtimCacheWarning` each time). The chi ceiling is now unified to
  1024 across construction, verification, and the compile-time oracle floor, so these
  references compile, verify, and **cache once** (`reference_info()['chi']` is reported,
  no warning). A χ=8 reference (e.g. `cube_ccz`) is unaffected — `chi_max` is a cap, not
  the actual rank, so low-χ references sample identically. (χ > 1024 still falls back to
  the exact deduced state — the synthetic genuinely-irreducible worst case.) Supersedes
  the [0.5.6] note that said 64 < χ ≤ 1024 falls back.

## [0.5.6] - 2026-06-21

A performance fix: the multi-magic reference compile no longer falls off a cliff.
**Compile-time only** — no public API, on-disk format, or sampling change; cached
`.ref` bytes and stream pins are byte-identical, results unchanged.

### Fixed

- **Multi-magic reference compile no longer hangs / cliffs.** Compiling the reference
  for a state with many independent magic qubits (stabilizer rank χ growing as `2^m`)
  previously blew up super-exponentially and effectively hung past ~6 magic qubits.
  Three independent hot spots are fixed: the conditioning pass's
  `max_clifford_subspace` search (an exponential Clifford-grade family →
  apriori-enumerated, bounded, overflow-safe), and both the standalone and
  `build_frontier`-inline `reduce_to_chi2` stabilizer finders (`O(χ³)` brute
  enumeration → `O(χ²)` phase-solve). The compile **no longer hangs** at any m — the
  whole χ=2^m sweep up to m≈13 finishes in seconds. The protocols xtim actually ships —
  reducible / code-structured states like the [[8,3,2]] transversal CCZ (`cube_ccz`) —
  compile **and cache** in milliseconds. *(A genuinely-irreducible χ=2^m state with
  m ≳ 7 still falls back to the exact **deduced** bare state rather than caching: the
  compile attempt is now bounded — ~1–3 s at χ≈2k–4k, no hang (`build_frontier`'s
  residual `O(rays²)`) — sampling is exact regardless, but the verification battery
  samples at χ≤64 so these don't memoize. The synthetic worst case; real protocols are
  unaffected.)*

## [0.5.5] - 2026-06-20

An engine consolidation (`xtim-engine-4`) that removes several subtle internal
errors and extends the reference compiler. No public API change; sampling results
are unchanged (raw measurement-record *bytes* may shift for circuits using `MPP`
of an all-Z product — see below — but every distribution, detector, observable,
and expectation is identical).

### Fixed

- **Multi-magic on a code state now compiles & caches a reference.** Reference
  compile previously fell back to the (exact) deduced bare state for a transversal
  non-Clifford layer acting on a measure+feedback-prepared code state — e.g. the
  [[8,3,2]] transversal CCZ — leaving `reference_info()['chi']` as `None` and emitting
  an `XtimCacheWarning` every run. These now compile and cache like any other circuit
  (`cube_ccz` reports χ=8, no warning). Most multi-magic circuits now cache.
- **Stale references can no longer be served, automatically.** The invisible reference
  cache now keys on `sha256(circuit text + ref-format version + sha256(the compiled
  engine))`, so any engine rebuild changes the key and old `.ref` entries drop
  themselves — no manual `ENGINE_VERSION` bump can be forgotten (the label is now purely
  cosmetic). This closes a bug where a stale reference could be silently served after the
  engine changed. For end users this is invisible: a released wheel ships one engine
  binary, so its hash is stable and caches reuse fully across runs; only a rebuild (a
  different binary) invalidates — which is correct.

### Changed

- **`MPP` of an all-Z product desugars to a Z-basis ancilla** (no Hadamard). The same
  operator is measured (verified against Stim) and all distributions are unchanged; only
  the internal representation differs, so raw record *bytes* may shift.
- **The sampler accepts more circuits** (e.g. a Hadamard on a magic-carrying wire that
  used to reject now samples).

## [0.5.4] - 2026-06-20

An **onboarding** release: a `pip install`-the-wheel user can now go from
"installed" to "loaded and scored a real protocol" with no repo clone. No engine
or on-disk-format change; `0.5.3` caches and DEMs are unaffected.

### Added

- **`xtim.load_example(name)` — bundled demo circuits, no clone needed.** The six
  demo `.stim` circuits now ship *inside* the wheel, so `xtim.load_example("cube_ccz")`
  loads any of them straight from the installed package; `xtim.list_examples()` lists
  them and `xtim.example_path(name)` gives the file path. Names accept the `.stim`
  suffix; an unknown or path-like name raises a clear error. (Previously a wheel user
  hit “No such file” on the README’s very first example, because the circuits lived
  only in the repo.)
- **`examples/xtim_tutorial.ipynb` — a hands-on Jupyter tutorial** walking the whole
  arc end to end: hello magic state → `diagnose()` → scoring (with the frame-correction
  footgun) → decoder-in-the-loop → noise sweep → the [[8,3,2]] multi-magic CCZ. Loads
  its circuits via `load_example`, so it runs from any directory.
- A **`[tutorial]`** install extra (`stim` + `pymatching` + `matplotlib`) for the
  notebook (`pip install "xtim[tutorial]"`).

### Fixed

- **Source distribution completeness.** `MANIFEST.in` shipped only `examples/*.stim`
  and `*.md`, so the runnable example scripts (`onboarding_new_protocol.py`,
  `dem_reject_region.py`) — which the docs tell you to run — were missing from the
  sdist; the tutorial notebook would have been excluded too. Now ships `*.py` and
  `*.ipynb`.
- **Wheel-vs-repo clarity.** The README and `examples/README.md` now state plainly that
  the wheel bundles the demo *circuits* (`load_example`) while the example *scripts*,
  the notebook, and `docs/` live in the repo / sdist — so a wheel user knows what needs
  a clone before hitting a missing file.

## [0.5.3] - 2026-06-20

An engine **correctness fix** for classically-controlled feedback that crosses a
non-Clifford gate, plus the first multi-magic example. No public API or on-disk
format change.

### Fixed

- **Feedback (`CX`/`CY`/`CZ rec[-k] q`) crossing a non-Clifford gate is now
  coherentized — it was previously silently wrong (or refused).** Such feedback is
  rewritten into a coherent basis-matched controlled-Pauli before deferral, so it
  propagates correctly through `T`/`CS`/`CCZ`. Concretely, a logical magic-state prep
  that measures a syndrome and feedback-corrects *before* the transversal non-Clifford
  gate (e.g. the [[8,3,2]] transversal CCZ) now returns the exact expectation on every
  shot, instead of a wrong value on the feedback-corrected shots. Feedback that does
  **not** cross magic is unchanged (same fast frame trick, byte-identical, no χ growth).
  The only outcomes now are correct-coherentized, correct-frame-trick, or a loud
  `XtimRejectError` — never a silent wrong answer. `detector_error_model_with_reject()`
  coherentizes too, so feedback-before-magic circuits export a DEM cleanly.

### Added

- **`examples/cube_ccz.stim` — the first multi-magic demo.** The [[8,3,2]] cube color
  code with a transversal CCZ: `k=3` logical qubits, output `CCZ|+++⟩_L` (a genuine
  `χ>2` state, beyond the single-magic `χ≤2` sweet spot the other demos use). A
  deterministic feedback prep gives a clean `diagnose()` — sign-constant, well-defined,
  `⟨X̄₁⟩=⟨X̄₂⟩=⟨X̄₃⟩=+0.5`, 100% acceptance, no post-selection. (Reference compile falls
  back to the exact deduced-bare-state path here — the normal multi-magic case; the
  `XtimCacheWarning` is benign.)

## [0.5.2] - 2026-06-19

Five rounds of persona-driven UX hardening (fresh-eyes "new user" testing). All
documentation and error-message polish — **no engine, API, or on-disk-format
change**; `0.5.0`/`0.5.1` caches and DEMs are unaffected and existing code keeps
working.

### Fixed

- **`diagnose()` no longer reads as a failure on a healthy magic-prep run.** A
  whole-circuit DEM refusal previously tagged every expectation value line
  (`[circuit DEM refused …]`) and printed a mid-report `reason:` wall of engine
  jargon ("negative probability", "refusing to approximate"). It now surfaces
  **once**, as a demoted bottom note that *leads* with reassurance ("expected for
  magic-state prep … sampling stays exact … nothing is wrong"), with the engine
  reason trimmed to one line. The cold-cache `χ` line reads `not yet cached
  (deduced on the fly — fine)` instead of a bare `?`.
- **README fidelity guidance is now honest and complete.** The 30-second example
  contracts the full target Bloch vector against all expectation columns (prints
  `F ≈ 1.0` for the *post-selected* — not "noiseless" — state), states the
  column-coverage precondition, and carries a prominent ⚠️ caveat that the
  raw-mean shortcut is valid only for `sign constant` columns — a `byproduct
  frame` protocol must fold in the record-parity `HINT` first or `F` collapses to
  ~0.5. Added a 3-term vocabulary block (χ / gauge detector / byproduct frame)
  and glossed "DEM" at first use. The shown `diagnose()` block now matches real
  output verbatim.
- **Clean, typed errors on bad input everywhere.** `Circuit.from_file()`,
  `Reference.load()`, and the CLI now raise a typed `XtimError`/`XtimReferenceError`
  (or print a one-line CLI message + non-zero exit) for a missing **or
  binary/corrupt** file and for invalid `--shots`/`--chi-max`, instead of leaking
  a raw `pathlib`/codecs/`ValueError` traceback.
- **Doc/example accuracy.** `examples/dem_reject_region.py` resolves its circuit
  from any directory and reports the nontrivial-correction count honestly
  (cultivation's reject region is saturated → decode is a no-op on survivors; the
  guide now demonstrates real partial-decode on the `cultivation_d3_rate` twin).
  Corrected the `miniature_oracle` table cell, the `code_switching` refusal
  reason (non-deterministic observable), the decode-and-budget characterization
  (a coherent-error *budget/estimate*, not a "lower bound"), and a stale test
  filename; dropped a `sinter` name-drop that implied a bridge that does not
  exist.

## [0.5.1] - 2026-06-19

Documentation and example-UX patch from new-user feedback. No engine, API, or
on-disk-format change — `0.5.0` cache files and DEMs are unaffected.

### Fixed

- **README 30-second fidelity example was misleading.** The headline snippet combined
  only a single expectation column into `F`, printing `F ≈ 0.75` for a state whose true
  fidelity with the target is `1.0`. It now contracts the **full** signed target Bloch
  vector against **all** expectation columns
  (`F = (1 + (target * exps[keep].mean(0)).sum()) / 2`), printing `F ≈ 1.0` as expected
  for the noiseless target.
- **`examples/dem_reject_region.py` only ran from the `examples/` directory.** It now
  resolves its circuit via the same beside-file-then-`benchmarks/` fallback as
  `onboarding_new_protocol.py`, so it runs from anywhere (and from the monorepo).
- **Reference-compile fallback warning read as an alarm.** When a circuit has no
  gate-battery reference (expected for some circuits), the warning now leads with the
  reassurance that *sampling stays exact — no action needed*, instead of surfacing the
  internal `oracle=FAIL` detail as if something were wrong.

## [0.5.0] - 2026-06-19

First release since 0.3.1: ships both the engine-2 reference-compile speedup (see
[0.4.0] below — staged but never tagged on its own) and the new decoder-DEM-with-reject
feature.

### Added

- **Decoder DEMs that flag the not-Pauli-correctable faults, for magic-state-prep.** New
  `Circuit.detector_error_model_with_reject(...) -> xtim.DemWithReject` plus the public
  `xtim.PostselectFault` dataclass. The `DemWithReject` carries `{dem, reject_detectors,
  postselect_faults}` + `.keep_mask(dets)`: `dem` is a clean Stim `DetectorErrorModel` of the
  Pauli-correctable faults, and **`postselect_faults`** is the primary output — each a
  `PostselectFault(detectors, probability)` flagging a fault that is detectable but not
  Pauli-correctable (a coherent logical `S` no Pauli frame recovers), for *you* to either
  **post-select** (sound) or **decode-and-budget** (their summed probability is the coherent-error
  floor). `reject_detectors` is the derived convenience union (and `.keep_mask` the blunt
  conservative post-select); xtim does not force a policy. Faults are classified by whether a
  logical magnitude `|⟨P̄_r⟩|` changes (sign flips stay correctable). An odd-`S` on a measured
  detector read twirls to a probability-½ Pauli edge (no refusal). A **completeness gate** refuses
  loudly (`XtimDemError`) when a single fault changes a magnitude but fires no detector (the circuit
  under-declares its post-selection syndrome). See `docs/xtim_dem_reject.md` and
  `examples/dem_reject_region.py`. Additive: `detector_error_model()` is unchanged and
  byte-identical for non-magic circuits.

## [0.4.0]

A large reference-compile speedup (engine version bump) plus the `diagnose()`
robustness/signed-target work from the QEC-user feedback round.

### Changed

- **Reference compile is dramatically faster.** It now prefers the deduced
  conditioning path (`build_bare_state`) over the per-T-gate frontier split, falling
  back to the frontier only when the deduced path rejects. Transversal-T-on-code (a
  single logical magic state) now compiles **n=127 in ~0.10 s (was ~8.75 s) and
  n=1023 in ~12 s (was ~50 min)**. The physics is identical (overlap = 1; every
  compile-battery gate still passes); per-shot sampling is unchanged.
- **`chi_max` is a memory backstop, not a rejection criterion — and the default is
  raised 4096 → 16384.** xtim does not reject on stabilizer rank in normal use: a χ>2
  output (the sweet spot is χ≤2, a single logical magic state) is still simulated
  exactly via the deduced bare state. `chi_max` only guards against a genuinely
  enormous output; it now allows substantially larger χ before that backstop fires,
  and you can raise it further. (The docs previously framed χ>2 as "overflow and
  reject," which was inaccurate — χ>2 samples fine.)

### ⚠ Cache invalidation (engine version bump)

- The engine version is bumped **`xtim-engine-1` → `xtim-engine-2`**. The faster
  compile yields a different — but physically identical — serialized reference
  (`.ref`) gauge, so **every engine-1 `.ref` cache is invalidated; xtim recompiles
  automatically on next use.** No action required: caches under `.xtim_cache/` are
  safe to delete, and old caches are simply ignored.

### Fixed

- **`diagnose()` no longer crashes on a feedback (or otherwise DEM-refusing)
  circuit.** Its internal DEM-export probe only caught `XtimDemError`, so a circuit
  with classically-controlled Pauli feedback — whose DEM export raises
  `XtimParseError` ("feedback not yet handled by DEM export") — propagated out and
  aborted the whole report. It now also catches `XtimParseError`/`XtimRejectError`
  and records the refusal as a fact (with a clean reason, stripped of the generic
  parse hint that misleads in a DEM context). The report renders for every circuit.

### Added

- **Quality-of-life from the user-feedback rounds:** `Circuit.without_noise()` (Stim
  parity); `Diagnosis._repr_html_` (the formatted report renders in Jupyter);
  `collect(progress=...)` callback and per-row `p0` / `expectation_paulis` fields (so a
  flattened results table stays self-describing); and a `py.typed` marker so downstream
  type-checkers see xtim's inline annotations.
- **Signed target in `diagnose()`.** Each `ExpectationInfo` now carries a
  `signed_value` — the byproduct-frame-corrected `βᵢ` (the target `⟨Pᵢ⟩` of the ideal
  output in the canonical zero-frame), so the report shows `value=+0.707…` rather than
  only `|value|=0.707…`, and a closing `target (signed beta): [+0.707, -0.707]` line
  gives the full target Bloch vector in one place. `signed_value` is `nan` when the
  sign is genuinely indeterminate (varies with no single record-parity explaining it);
  `|signed_value| == abs_value` otherwise. This is a DERIVED, reported fact — diagnose
  still applies nothing to the raw shot data.

## [0.3.1]

Documentation + diagnostics **honesty pass** from a round of QEC-user feedback — no
engine or API change. The simulable scope and the decoder/DEM story are now stated
accurately, and several misleading error hints were corrected.

### Fixed / clarified

- **README:** added an upfront **"Scope & limits"** section (xtim is a
  magic-state-prep *co-processor* — single χ≤2 magic output, ≲10 T gates, Clifford
  bulk belongs in Stim); documented that classically-controlled Pauli feedback and
  `decompose_errors` are *shipped* (the old "deferred" list was stale and denied the
  v0.3 headline feature); honest DEM scope (faithful magic protocols are
  post-selection-only, not decoder-in-the-loop); made the install snippet
  version-agnostic (`pip install xtim-*.whl`) so it can't go stale; documented the
  run-from-a-clone (`ModuleNotFoundError: xtim._xtim`) and
  benign `XtimCacheWarning` (χ>2) traps.
- **examples/README:** corrected the table — `miniature_oracle` is n=5; `faithful`
  cultivation's *reference* χ is 2 (the ~256 figure is a transient one-time-compile
  rank, not per-shot; it samples at ~0.04 ms/shot); only `cultivation_d3_rate` is
  Clifford / DEM-exportable (`code_switching_rate` keeps the transversal `T̄`, so it
  is still non-Clifford).
- **`docs/STIM_GAP_AUDIT.md` is now shipped** (it was referenced but absent).
- **dialect doc:** `CH` is marked parse-only / run-time-rejecting (it embeds a
  Hadamard that leaves the simulable class).
- **Error hints:** the DEM-refusal hint now routes by cause — graphlike-decompose
  failures advise `ignore_decomposition_failures` (not the unhelpful
  `include_expectations=False`); the class-reject hint names `CH`; the chi-cap hint no
  longer leaks a literal `chi=K` on the deduced-reference path; `HERALDED_*` reports
  as a recognized-but-unsupported Stim instruction (not "unknown instruction"); the
  `rec`-as-gate-target message notes that only `CX`/`CY`/`CZ` take feedback.

## [0.3.0]

Stim-compatibility foundation — most of Stim's circuit surface now loads, plus
classically-controlled Pauli feedback (a Stim feature earlier xtim lacked). The
non-Clifford engine and the sampling hot path are unchanged: feedback-free
circuits stay byte-identical (stream pins 6/6).

### Added

- **Classically-controlled Pauli feedback `CX` / `CY` / `CZ rec[-k] q`.** A `CX`/`CY`/
  `CZ` whose **first target is a measurement record** `rec[-k]` applies `X`/`Y`/`Z` to
  qubit `q` iff that record bit is 1 (Stim's measurement-record-controlled Pauli
  feed-forward). Handled as an **exact post-sampling triangular GF(2) record-relabel**:
  the controlled-Paulis are pulled out of the bare circuit (so the bare state / records /
  expectations stay feedback-free), each is propagated once at setup to a
  `(control record, record-flip mask, expectation-sign mask)` descriptor, and per shot a
  triangular pass XORs the flips into later records / flips the selected
  `PAULI_EXPECTATION` signs. A Pauli before a measurement only flips that outcome (or an
  expectation sign), never a Born probability — so this is exact even with magic in the
  circuit. The relabel is **off the state-evolution hot loop** and entered only when the
  circuit contains feedback, so feedback-free circuits are byte-identical and same-speed
  (stream pins 6/6). The record distribution matches real Stim (which runs feedback
  natively). Out of scope: classically-controlled **Cliffords**, `sweep[k]` controls, and
  DEM-with-feedback (all rejected with a clear message). See
  [`docs/xtim_dialect.md`](docs/xtim_dialect.md) §1b.
- **Measurement-gap parity with Stim:** two-qubit Pauli measurements `MXX` / `MYY` /
  `MZZ`, inverted measurement targets (`!q`), and readout-flip probabilities `M(p)` /
  `MR(p)` / `MX(p)` (the record takes the wrong value with probability *p*). Record
  distributions byte-match real Stim.
- **Full Stim Clifford gate set, verified tableau-equal to Stim.** Every Stim unitary
  Clifford now loads and is pinned against Stim — `I`, the SWAP family, `SQRT_*`,
  basis-cycling `C_*`, the `XCX … ZCZ` two-qubit set, aliases, and the `[tag]` suffix —
  plus `MPAD`, `I_ERROR`, and `II_ERROR`. Locked by a 50-gate Stim-equality suite
  (which self-tests that it would catch a wrong desugaring).
- **DEM `decompose_errors=True`** — graphlike error decomposition for matching decoders
  (PyMatching/MWPM); `detector_error_model(decompose_errors=,
  ignore_decomposition_failures=)` mirrors Stim's kwargs. Surface and repetition codes
  are decode-equivalent to Stim's DEM; color-code d3 carries a small documented residual
  (xtim's coarser canonical-channel merge — see `STIM_GAP_AUDIT.md`); non-graphlike
  cases require `ignore_decomposition_failures`, as in Stim.

## [0.2.0]

### Added

- **Pauli-product measurement `MPP` and rotation gates `SPP` / `SPP_DAG`.** Now
  supported via **parse-time desugaring** — no engine/sampler change. `MPP` measures
  one or more `*`-joined Pauli products, each producing one measurement record in
  Stim's record order (desugared to a fresh-ancilla cat-check gadget: `H a`; one
  controlled-Pauli per factor; a `Z a` for a leading `!`; terminal `MX a` = the
  record). `SPP P` = `exp(-iπ/4·P)` / `SPP_DAG P` = `exp(+iπ/4·P)` desugar to a
  CX/CY/CZ basis-change ladder onto a pivot + `S`/`S_DAG`, then uncompute. The record
  distribution byte-matches real Stim (incl. the `i^{#Y}` and `!`-inversion sign
  conventions); circuits that use neither are byte-identical and same-speed as before.
  See [`docs/xtim_dialect.md`](docs/xtim_dialect.md) §2b/§2c.
- **`Circuit.diagnose()`** — an onboarding report from ONE noiseless run. It
  surfaces the facts you would otherwise reverse-engineer by hand: the reference
  χ and cache status, which detectors are deterministic (post-selectable) vs
  gauge, each `PAULI_EXPECTATION` column's `|β|` magnitude, its byproduct-frame
  sign HINT (a record parity), and whether it is DEM-expressible — plus an
  advisory post-selection recipe. It *reports*; it never applies a correction or
  filters shots.
- **`python -m xtim diagnose <circuit.stim> [--shots N] [--seed S]`** — the CLI
  surface for the report above.
- **Actionable `.hint` on reject exceptions.** `XtimParseError`,
  `XtimRejectError`, and `XtimDemError` now each carry a `.hint` (a student-facing
  fix) and a `.location`, and their `__str__` prints cause + location + hint. The
  2/3/4 CLI exit-code contract is unchanged.
- **Gates `T_DAG` and `CS_DAG`** — the inverse π/4 phase and controlled-S†
  (parse-time conveniences that desugar to existing gates).
- **Coordinate preservation.** `QUBIT_COORDS` / `DETECTOR(coords)` are stored and
  surfaced via `Circuit.get_final_qubit_coordinates()` and
  `Circuit.get_detector_coordinates(only=)` (Stim's method names), with
  `SHIFT_COORDS` resolved cumulatively. The exported DEM carries detector
  coordinates as `detector(x, …) D#` lines. Coordinates are annotations: they never
  affect any record stream.
- **Output formats `dets` and `r8`.** `python -m xtim sample`/`detect` now accept
  `--out_format dets|r8` (alongside `01`/`b8`), byte-identical to
  `stim … --out_format dets`/`r8`. xtim keeps detection events, observable flips and
  measurements in SEPARATE channels (`--out`/`--obs_out`/`--meas_out`), so each `dets`
  channel is homogeneous and labelled by its Stim type hint: `D#` detectors, `L#`
  observables, `M#` measurements. `r8` is the run-length byte encoding (gap-to-each-1
  plus a trailing-run byte, 0xFF continuation) per channel. The `--exp_out` float64
  channel has no bit meaning and stays `%.17g` text (or raw f64 under `b8`).
- **Programmatic circuit building.** `xtim.Circuit` now mirrors Stim's
  construction surface: `append(name, targets, arg)`, `+`/`+=` (concatenate),
  `*`/`*=` (a `REPEAT` block), `to_file`, `copy()`, `len()`, `str()`, and the
  complete `num_qubits`/`num_measurements`/`num_detectors`/`num_observables` set
  (all matching real Stim's counts). Targets accept ints, raw target strings
  (`rec[-1]`), or `stim.GateTarget`s. Every mutation re-parses, so an invalid
  build raises `XtimParseError` and leaves the circuit untouched.
- **Onboarding example + dialect doc.** `examples/onboarding_new_protocol.py`
  walks a fresh protocol from circuit text to a scored fidelity via
  `Circuit.diagnose()`; `docs/xtim_dialect.md` specifies the dialect delta.
- **`xtim.__version__`** — resolved from the installed package metadata.

## [0.1.0]

Initial release: Stim-shaped sampling of logical magic-state-preparation
protocols (an extended-Stim dialect with `T`/`CS`/`CH`/`CCZ` and
`PAULI_EXPECTATION`), a real `stim.DetectorErrorModel` export with expectation
L-columns, the `collect()` batch runner, and the `python -m xtim` CLI
(`sample` / `detect` / `analyze_errors` / `state`).
