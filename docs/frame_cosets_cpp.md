# The compiled sector table (`_xtim.plan_frame_cosets`, 3.1.3)

Design note for the C++ port of `xtim.frames.plan_frame_cosets_fast` — the Born
SECTOR table of an S-layer plan on a `branch_frames()` reference (`docs/branch_frames.md`
§2 for the export, `plan_frame_cosets`'s docstring for the algebra).

## 1. Why (the grounding, 2026-09-14)

The pure-Python table was the largest remaining COLD cost on adaptq's magic-producer row:
the speed gate's `cold_rows` leg reports it as `sector_table_us_per_shot` (signed 0.6913
hard-cold / 0.8477 fresh-steady, report-only with a ×1.25 advisory), larger than the row
build it feeds.  Profiled on the u2 unit's 287 distinct real a-masks (78 wires, χ = 2,
k = 1; |supp a| from 0 to 9, 2^9 = 512 terms at most):

| where the 0.79 ms/plan (mean; median 0.44) goes | share |
|---|---|
| `lambda_functional` → `ref_expectation` → `_anticommute` (125 k Python calls per 287 plans: one per generator per basis read) | ~45 % |
| the per-coset Python loop (frame-copy search over 2^{2k} = 4 codes, `np.zeros`/`count_nonzero`/`nonzero`/`any`/`max` per sector) | ~25 % |
| numpy setup of the vectorized block (`unique`, `argsort`, packbits, matmuls on (2^m, 77) int64) | ~30 % |

Nothing in the table is arithmetic-bound: the whole plan is 2^m × n bit-parities (≤ 40 k
word operations), an RREF of an n × m GF(2) matrix, and ≤ 2^{2k} · χ complex adds per
sector.  The cost is interpreter and array-object overhead, which a compiled loop over
bit-packed rows removes wholesale.  Estimated compiled cost 5–20 µs/plan including the
pybind marshalling of the export's arrays; measured in §5.  Expected effect on the
signed per-shot number: ≥ 20× on the table build (the leg's other cost, the p=0 reference
compile, is KEPT across the cold windows and not part of the number).

## 2. Contract

`_xtim.plan_frame_cosets(stab_x, stab_z, stab_phase, destab_x, destab_z, free, sigma,
coeff, branch_x, branch_z, a_mask) -> list[dict]` — the arrays of a `branch_frames()`
export (unpacked `uint8` bit rows, the numpy layout; `free` int32, `sigma` uint8 (χ, k),
`coeff` complex128) and the plan's a-mask (uint8, n).  The output is EXACTLY the numpy
path's: a list, in first-encounter order of the sectors (ascending v), of dicts with the
keys, in order, `vz`, `fx`, `fz` (uint8 arrays of shape (n,)), `logical_flip` (a tuple of
two lists of Python ints over the free rows, or `None`), `weight` (float), `frame_copy`
(bool).  Sectors of weight < 1e-12 are dropped before normalisation, as in the numpy path.

`xtim.frames.plan_frame_cosets_fast(ref, a_mask, *, backend=None)` is the production
entry; adaptq's `engine._plan_frame_cosets` forwards to it unchanged.  `backend`
resolves as `None` → `FRAME_COSETS_BACKEND` (module default, from
`XTIM_FRAME_COSETS_BACKEND` at import, default `"auto"`; an unknown value is refused at
import) → `"auto"` MEANS `"cpp"`.  A loaded extension that lacks the symbol or exposes a
`FRAME_COSETS_ABI` other than the package's `xtim.frames.FRAME_COSETS_ABI` is a STALE
build: `"auto"`/`"cpp"` raise `xtim.frames.StaleExtensionError` (a `RuntimeError` naming
the `.so` and the rebuild command) — there is no silent fallback to the 20× slower
pure-Python path (review MAJOR-1: a speed regression no gate can see is not a null
option).  `"numpy"` selects `plan_frame_cosets_numpy` EXPLICITLY — the 3.1.2 body with
the portable weight arithmetic of §3, the ORACLE the compiled path is pinned against
(bit-for-bit); `plan_frame_cosets` remains the reference oracle both are pinned against
(bit-for-bit on the weight too, since all three bodies execute the same arithmetic, plus
the 1e-12 tolerance leg of 3.1.2).

The compiled path's one extra refusal: `|supp a| > 30` raises `ValueError` (the numpy body
has no guard and would exhaust memory enumerating 2^{|supp a|} terms; the u2 unit's
maximum is 9).

Refusals are the numpy path's, same exception type: `ValueError` from `lambda_functional`
when a null-space basis read is not ±1; `AssertionError` from `ref_expectation` when a
Pauli commutes with every generator but is not their product.  The κ > 0 / fallback / CZ
refusals and the non-frame-copy refusal are the CONSUMER's (`require_kappa_zero`,
adaptq's `_frames_sector_table`) and are untouched: a non-frame-copy sector is returned
with `frame_copy=False`, `fx=0`, `fz=vz`, `logical_flip=None`, exactly as before.

## 3. Bit-for-bit: why the float `weight` can be pinned

Every term of A_{c,i} is c_i times a unit in {±1, ±i} times ±1 twice — products with 0
and ±1 are exact in IEEE arithmetic, so each term's components are ± the components of
c_i (or 0) whatever the multiplication order or FMA contraction, and A is a sum of exact
values in a FIXED order (ascending v; numpy's `add.at` is sequential).  The C++ adds in
that order.  The weight is then PORTABLE arithmetic, the same statements in all three
bodies (`xtim.frames._sector_weight` / `_normalise_weights` in the reference and numpy
bodies, `detail::sector_weight` and the normalisation loop in the header): per branch
`re·re + im·im` on doubles (two rounded products, one rounded add), accumulated left to
right from 0.0; the normalisation an explicit left-to-right sum from 0.0 and one division.
CPython never contracts, and the header is compiled with `fp-contract=off` (an MSVC
`fp_contract(off)` pragma beside it), so the bit pin is a claim about IEEE basic
operations in a fixed order — NOT about numpy's complex-`absolute` kernel (FMA-fused only
on some dispatch targets), its pairwise sum, or CPython 3.12's compensated `sum()`, none
of which xtim's declared support matrix (python ≥ 3.11, numpy ≥ 1.26) guarantees (review
MAJOR-2; the first cut reproduced those kernels and its pin would have failed on a
supported interpreter).  The only non-exact operations — the complex division for μ
(numpy's Smith method, reproduced) and the residual `B − μ·c'` — feed a TOLERANCE test
(1e-9) that decides `frame_copy`, with `std::hypot` as the modulus; they are compiled
without contraction so the test is the same arithmetic on every build, and their inputs
are far from the threshold on every fixture (the residual is ~1e-16 on a frame copy and
O(1) otherwise).  Signed zeros can differ (numpy's `(-1j)**p` produces −0 components);
they are invisible to `re·re` and to every comparison.

Versus 3.1.2 the numpy body's `weight` moves by at most an ulp or so on some plans (the
kernels above differ from the sequential formula by that much); no consumer reads the
bits — adaptq consumes `frame_copy`/`fx`/`fz` and every xtim/adaptq test that reads
`weight` pins it with a tolerance — and adaptq's byte gate is the evidence (re-run after
the change: PASS, no leg moved).

## 4. Verification (tests/test_frame_cosets_cpp.py, cpp/tests/test_frame_cosets.cpp)

Oracle-first: `_assert_bitwise` checks structure, dtypes, key order, the flip's Python
types, and the weight's 8 bytes, on (i) every plan of the multi-sector fixtures, (ii) the
synthetic masks up to |supp a| = 9 on the magic and stabilizer 10-qubit references, (iii)
the 287 real u2 a-masks (`tests/data/u2_qrm_magic_amasks.npy`) on the qrm producer
fixture's bare state — verified equal to the reference adaptq builds for that unit, (iv) a
seeded random sweep on the three data fixtures; each also against `plan_frame_cosets`.
Refusals: both paths raise the same type on a corrupted copy of a real export (a phase
flipped → `ValueError`; a destabilizer dropped → `AssertionError`).  Determinism: two
compiled calls byte-identical.  Backend selection: a stale extension (symbol missing, or
another `FRAME_COSETS_ABI`) is refused on `auto`/`cpp` with `StaleExtensionError` and
served only on an explicit `numpy`; `XTIM_FRAME_COSETS_BACKEND` exercised in a fresh
subprocess (`numpy` → zero compiled calls, `cpp`/unset → compiled, an unknown value →
refused at import).
Hot path untouched: the record stream (prefix words, plan ids, plan tables, record keys,
group ids, branch frames) of the three data fixtures hashes to the values captured on the
3.1.2 build before the change.  The C++ unit test covers the core without Python: the
stabilizer reference (each table one frame), the |T⟩ reference where S|T⟩ is Y|T⟩ (a
definite plan WITH a logical flip), a two-sector split, determinism, both refusals, and
the portable weight sequence on non-exact inputs against literals computed by the Python
statement.

## 5. Measured (2026-09-14, host load ~1.0–1.5, median of 5 passes per plan)

| the 287 real u2 plans | numpy path | compiled | ratio |
|---|---|---|---|
| µs/plan median | 512.8 (437 on a quiet pass) | 28.8 | 18× |
| µs/plan mean | 964.7 (792 quiet) | 30.6 | 32× |
| µs/plan max (\|supp a\| = 9) | 4521 | 71.9 | 63× |
| the empty plan (the marshalling floor) | 162.6 | 18.6 | — |

The compiled cost is flat in |supp a| (21 µs at 0–1, 50 µs at 9): two thirds of it is
the fixed pybind cost of converting eleven arrays and packing the 78 × 78 export per call
plus building the output dicts, not the enumeration.  Caching the packed `FrameRef` per
reference would remove most of the floor; not done — the number below is already far
inside the noise of the row it feeds.

adaptq speed gate (`scripts/gates/speed_gate.py check --replicates 3`, main 258541f,
unchanged), the magic row's REPORT-ONLY `sector_table_us_per_shot`:

| leg | signed | before (3.1.2 .so) | after (compiled) |
|---|---|---|---|
| noisy hard_cold  | 0.6913 | 0.7477 | 0.0349 |
| noisy fresh_steady | 0.8477 | 0.8433 | 0.0396 |

The fenced builder number (`cold`, exclusive of the table) is unchanged within its fence
(0.5958 → 0.6191 hard-cold on a host 0.4 load higher, fence 0.6669; 0.3466 → 0.3535
fresh-steady, fence 0.3865); SPEED GATE PASS both runs, BYTE GATE PASS after, adaptq's
`tests/oracle/test_prep_coherent_charge.py` + `tests/test_instrument_branch_index.py`
17/17 on the rebuilt extension.  adaptq needs NO change; an owner re-sign of the
`cold_rows` leg's report-only `sector_table_us_per_shot` (0.69/0.85 → ~0.035/0.040) is
the coordinator's call — the ×1.25 advisory only speaks upward, so the stale signed
value cannot fail anything.

## 6. Null option

Keep the numpy table and cache harder.  Rejected: the cost is per NEW (text, a-mask) and
the cold-rows leg measures exactly first touch; a persistent on-disk table cache would
trade a 0.8 ms compute for an I/O + validation path and a new invalidation surface (the
reference state's identity), for a number the compiled loop makes ~20 µs.  The numpy path
is kept, selectable, and is the oracle — nothing is lost if the compiled path is ever
withdrawn.
