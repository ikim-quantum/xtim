# `branch_frames()` — exporting a retained state as (frame, stabilizers, branch Paulis, amplitudes)

Status: DESIGN + built on `feat/branch-frames` (2026-09-08, for 3.1.2). Read-only surface; the
sampling path is untouched (byte identity asserted by the existing stream tests).

## 1. Why

A downstream consumer (adaptq's fast tier) needs, per retained shot, the exact logical
charge of every Born branch of a coherent (twirl-split) state on a magic port. Until now the
only surface was `materialize(i)` + `pauli_expectation_*` reads, which (a) cannot separate a
coherent state's branches at all (the group-level `<X_S>` is exactly 0), (b) cost a Y-site
rotation per read for Y-type operators (~1.4 µs/site), and (c) return `0.0` from
`pauli_expectation_xz` for a Y-type Pauli because it takes the real part of `i·Y_S`.
The engine already holds the state in exactly the form the consumer wants — a shared
signed stabilizer frame plus a list of (branch Pauli, amplitude) entries — so it exports it.

## 2. The state model (the contract)

A `FramedSuperposition` is

    |ψ⟩ = Σ_i  c_i · D_i |ref⟩

* `|ref⟩` is the unique (up to global phase) joint +1 eigenstate of the n SIGNED generators
  `G_a = i^{stab_phase[a]} X^{stab_x[a]} Z^{stab_z[a]}`, a = 0..n−1, where `G_a` is the frame
  row `g_a = U Z_a U†` with the reference sign `(−1)^{eps[a]}` folded into the phase (exactly
  `certified_stabilizers()`'s convention, extended to the free rows).
* `D_i = Π_{d ascending, σ_i[d]=1}  d_{free[d]}` with `d_a = U X_a U†` (the destabilizer
  rows), multiplied LEFT-TO-RIGHT in ascending `d` with the engine's exact Pauli phase
  algebra (`pmul_into` ≡ `Pauli::multiply(acc, b)`: phase += b.phase + 2·|acc.z ∧ b.x|).
  This is the same product `framed_measure_anticommuting_general` forms (`oW[i]`), so the
  exported `D_i` reproduce the engine's own branch states, phases included.
* `c_i` are the stored complex coefficients (`entries()[i].second`), in the container's
  iteration order — the engine's canonical order, deterministic for a given (plan, σ, coins)
  record, hence identical across re-materializations of the same shot.
* Branches are mutually orthogonal (`D_i† D_j` flips the free generators in `σ_i ⊕ σ_j ≠ 0`
  and so anticommutes with a stabilizer of `|ref⟩`), so `⟨ψ|ψ⟩ = Σ|c_i|²`.
* Index space: every Pauli is over the STATE's n wires — the same indices
  `pauli_expectation_x(q)` addresses and `TwirlSampler.output_wires()` returns for the
  circuit's `OUTPUT_QUBITS`.

Everything a consumer needs follows by Pauli algebra with no engine call:

    ⟨ref| P |ref⟩ = i^m · Π_a (−1)^{... }   if P ∈ ⟨G⟩ up to phase, else 0
       (decompose: b_a = anticommute(P, d_a); R = Π_{a:b_a} G_a; P = i^m R)
    ⟨ψ_S| P |ψ_S⟩ = Σ_{i,j∈S} conj(c_i) c_j ⟨ref| D_i† P D_j |ref⟩  /  Σ_{i∈S}|c_i|²

for any branch subset S — the RESTRICTED expectation a consumer uses to read one Born
sector's charge. `xtim/frames.py` ships this reference implementation, tested bit-for-bit
against `pauli_expectation` on the full set.

## 3. API

    FramedSuperposition.branch_frames() -> dict
    BarrierBuffer.branch_frames(i)      -> dict      # == materialize(i).branch_frames()

Keys: `n` (int), `free` (int32, k), `eps` (uint8, n), `stab_x`/`stab_z` (uint8, n×n),
`stab_phase` (int8, n — eps folded), `destab_x`/`destab_z` (uint8, n×n), `destab_phase`
(int8, n), `sigma` (uint8, χ×k), `coeff` (complex128, χ), `branch_x`/`branch_z` (uint8,
χ×n), `branch_phase` (int8, χ). Row a of the `stab_*` arrays is `G_a`; row i of `branch_*`
is `D_i`.

## 3b. Measured (2026-09-08, u2 unit of the downstream consumer: 3 995 record groups, load ≈ 1)

* Definite groups: charges from the frames == expectation reads, 0/2 424 mismatches.
* Coherent groups: 0/5 810 sector rows unmatched; sector weights == Born weights (1e-9);
  every sector definite (|⟨X_S⟩| = |⟨Y_S⟩| = 1/√2); vs the engine's projected read on 454
  single-check rows: 0 mismatches. NOTE: a recipe-coherent retained state is χ = 2 in a
  frame where the anti checks are NOT stabilizers (E·Π_q|T̄⟩) — the consumer's branches are
  the Born SECTORS (I±g)/2 of those checks, read with `projected_expectation`, not xtim
  branches.
* Cost per group: materialize 62.6 µs; branch_frames 85.7 µs (export ≈ 23 µs); today's
  state + 2 reads 68.0 µs.

## 4. What it does NOT do

* It does not avoid `materialize(i)`: the branch list exists only in the reconstructed state
  (the buffer stores compact records, not states — decoder-feedback perf). The export itself
  is O(χ·n) bit copies on top of the replay. The replay cost is reported, not changed here.
* It is not a new kernel: nothing in the sampling path reads it.

## 5. Alternatives considered

* Export only per-branch CHARGES given logical operators: needs the consumer's operators
  inside the engine and hides the convention; rejected — the frames are the general object
  and the charge is two symplectic products away.
* A Hermitian-Y `pauli_expectation` (phase-correct `pauli_expectation_xz`): cheap and worth
  having (follow-up), but still a group-level read — it cannot separate branches.
* Null option (keep the expectation surface): leaves the coherent-branch defect unfixable
  from outside the engine. Rejected by the owner.

## 6. Tests (tests/test_branch_frames.py)

1. Dense oracle: ≤6-qubit circuits with T and injected coherent Pauli faults; the state
   rebuilt from `branch_frames()` equals the dense statevector up to global phase.
2. Convention: `frames.expectation(...)` equals `pauli_expectation` bit-for-bit for random
   Paulis, incl. Y-type (phase-correct) — and the RESTRICTED expectation on one branch of a
   two-branch superposition is ±1 where the full one is 0.
3. Determinism: identical arrays across two materializations and two identical batches.
4. Byte identity of the sampling path: the existing stream tests (unchanged engine).

## 7. Frames at the port — `BarrierBuffer.frames()` (3.1.2)

**Why.** The replay behind `materialize(i)` (~57 µs at n = 78: ≈ 5–7 µs law rebuild, ≈ 50 µs
bare-state copy + gate replay) exists only because the buffer stores RECORDS, not states. But
the record already contains the shot's exact Pauli frame: `residual_normal_form` splits the
composed end-of-circuit residual into a Pauli PREFIX `P` and a canonical diagonal tail
`C = S^a · CZ` (`twirl_kernel.hpp` §"Task 6"), the retained state is `P · C · |collapsed⟩`
(`twirl_kernel_sampler.cpp` Step 4: S^a, then CZ, then the prefix's X/Z), and the compact
store keeps `P` per shot (`prefix_words_`) and `(a, cz)` per distinct plan. `frames()` exports
exactly that — straight copies, no replay.

**API.**

    BarrierBuffer.frames(allow_kappa=False) -> dict
      n, pnw, shots, n_plans, compact
      prefix_words : uint64[shots, 2*pnw]   word w bit b <-> wire q = 64*w + b; words [0, pnw) are
                                             the X part, [pnw, 2*pnw) the Z part (the sampler's own
                                             LE word layout). All-zero row = identity prefix.
      plan_id      : uint32[shots]           0 = identity plan; k >= 1 indexes the per-plan arrays
      a            : uint8[n_plans+1, n]     S-layer mask per plan (row 0 = zeros)
      cz           : list[int32[m_k, 2]]     CZ pairs per plan
      r, kappa     : int32[n_plans+1]        fair-coin count / kernel-chain length per plan
      fallback     : int32[n_plans+1]        1 = fallback law (refused exactly like kappa > 0)

`xtim.frames.unpack_prefix(words, n)` gives `(prefix_x, prefix_z)` as `uint8[shots, n]` in
one vectorized numpy shift (the consumer's row format); the packed words are what the export
copies because they ARE the stored layout (32 B/shot at n = 78) — the unpack is a per-batch
numpy op, not a per-shot call.

**Contract — what P is.** `P` is the exact Pauli part of the shot's residual relative to the
NOISELESS REFERENCE run: the noisy final state equals `P · S^a · CZ · |collapsed⟩` where
`|collapsed⟩` is the bare reference state after the κ-step kernel chain (κ = 0: the bare
reference itself). Its global phase is dropped (unobservable). Indices are the STATE's wires
(`output_wires()` / `pauli_expectation_x(q)`); `a` and `cz` are wire-indexed.

**The sector-charge formula (κ = 0 plans).** Write the reference as the frame export gives
it, `|T⟩ = Σ_i c_i D_i |r⟩` (`|r⟩` the frame's stabilizer state, `D_i` the branch Paulis over
the free rows; χ = 1 for a stabilizer reference, χ = 2 for a magic one). With
`S_q = e^{iπ/4}(I − i Z_q)/√2`,

    S^a |T⟩ ∝ Σ_{v ⊆ supp(a)} (−i)^{|v|} Z^v |T⟩ ,     Z^v D_i |r⟩ = (−1)^{⟨Z^v, D_i⟩} λ_{v,c} D_i Z^{v_c} |r⟩

where `v_c` represents v's COSET modulo the frame's stabilizers (same anticommutation pattern
with the n generators) and `λ_{v,c} = ⟨r| Z^{v ⊕ v_c} |r⟩ = ±1`. Collecting the coefficient of
`D_i Z^{v_c}|r⟩` gives `A_{c,i} = c_i Σ_{v∈c} (−i)^{|v|} (−1)^{⟨Z^v,D_i⟩} λ_{v,c}`: the Born
SECTOR `c` has weight `Σ_i |A_{c,i}|²` (these are exactly the classifier's Born weights —
measured equal to 1e-9 on 900 rows) and is a FRAME COPY of `|T⟩` iff its amplitude vector is
a LOGICAL Pauli image of `c`: `B_c ∝ (D^x G^z)·c` for some product of free destabilizers
`D^x` (branch swap) and free signed generators `G^z` (branch sign) — the T-state orbit under
these is the Pauli orbit (e.g. `S̄|T̄⟩ ∝ Ȳ|T̄⟩`), which is why every sector measured is one
(0 non-copies on 1 200 definite groups and 900 sector rows). The sector's frame is then

    F_c = Z^{v_c} · D^x · G^z          (`frames.plan_frame_cosets`: per plan, 2^{|a|}·χ terms)

**The table is cheap (`frames.plan_frame_cosets_fast`, the production path).** The only
expensive term of the reference implementation is `λ_{v,c} = ⟨r| Z^{v ⊕ v_c} |r⟩`, one
`ref_expectation` per `v` (`2^{|a|}` reads). It is a GF(2) LINEAR functional, not a table:
Z-type Paulis multiply without phase, so on the frame's stabilizer state `|r⟩` the map
`w ↦ ⟨r|Z^w|r⟩ = ±1` on the subgroup `K = {w : Z^w commutes with every generator}` satisfies
`⟨r|Z^{w1 ⊕ w2}|r⟩ = ⟨r|Z^{w1}|r⟩ ⟨r|Z^{w2}|r⟩` — a homomorphism `K → {±1}`, i.e. linear over
GF(2), hence fixed by its values on a basis of `K` (the null space of `stab_x[:, supp a]`,
read off an RREF): `|supp a|` reads at most (`frames.lambda_functional`). Every `v ⊕ v_c`
lies in `K` (same coset ⇒ same anticommutation pattern). With that, the coset keys, the
`(−i)^{|v|}` amplitudes, the branch signs and the sums `A_{c,i}` are four vectorized
expressions over all `v` at once; the frame-copy search is the reference's, verbatim. The
suite pins `plan_frame_cosets_fast == plan_frame_cosets` field by field on the multi-sector
fixtures and on synthetic masks up to `|supp a| = 9`; `plan_frame_cosets_cached` serves the
fast table (the slow one on `reference=True`).

and on a shot with prefix `P` the split check `g_j` (a certified generator that is not a
stabilizer of the state) takes eigenvalue `ref_j · (−1)^{⟨P·F_c, g_j⟩}` on sector `c`, which
is how a sector sign vector `s` selects its coset (`frames.sector_coset`: the unique `c` with
`⟨P·F_c, g_j⟩ = [s_j ≠ ref_j]` — one coset matched every one of 900 rows). The recipe charge
of a logical operator `O` (commuting with every stabilizer and every `g_j`) on that sector is

    charge(shot, O, s) = ⟨P, O⟩ ⊕ ⟨F_{c(s)}, O⟩ ⊕ ref(O)

with `⟨·,·⟩` the symplectic (anticommutation) bit and `ref(O)` the sign bit of `⟨O⟩` on `|T⟩`
(`frames.frame_charge`). TWO CONVENTIONS for the last term, never both: with `ref(O)`
included the charge is the ABSOLUTE sign bit of `⟨O⟩` on the sector; with it omitted
(`frame_charge(..., ref_bit=0)`) the charge is the FLIP relative to the reference, i.e. 1 iff
the sector's `⟨O⟩` has the opposite sign to the reference's — adaptq's pin convention
(`b = 1 iff eX·eX0 < 0`). They differ by exactly the reference's own bit; a consumer that
applies `ref` on top of a flip double-counts it. Pure bit algebra on the export: no state, no replay. A plan may be
DEFINITE (one sector) and still carry a non-identity frame `F_c` (an S-layer whose `Z`'s
are stabilizers up to a logical): 618 of 1 200 pin-definite groups here have several
sectors that all agree on the recipe charge, and every one of the 1 200 matched today's
expectation read. This is the same structure the engine uses for its LOGICAL observable
channel (`ObsChannel`: per-plan constants plus one prefix parity `⟨P, W⟩` per shot).

**The κ > 0 rule.** When a plan has `kappa > 0` (or a fallback law) the collapsed sector
depends on the recorded kernel-chain outcome bits and is NOT a single frame coset, so the
formula above does not apply. `frames()` REFUSES loudly (`ValueError` naming the plan ids)
unless `allow_kappa=True` (the exported `fallback` field lets `frames.require_kappa_zero`
mirror both halves of the rule); it never materializes on the caller's behalf. Consumers must
`materialize(i)` those shots explicitly (or refuse). Measured on the downstream u2 unit at
p = 1e-3: κ = 0 and cz = ∅ on all 3 995 record groups.

**Oracles (tests/test_branch_frames.py + the u2 report).** (1) For every retained shot,
`P · S^a · CZ · |bare⟩` built with `apply_clifford` on the bare reference equals
`materialize(i)` up to global phase (`approx_equal`, 1e-9) — dense fixtures and the u2 unit;
(2) the sector table's weights equal the classifier's Born weights, every sector is a frame
copy, and the frame-algebra charges equal the sign of `projected_expectation` on the
materialized export for coherent sectors and today's expectation read for definite groups
(u2: 0 mismatches / 1 200 definite groups, 0 / 900 sector rows); (3) determinism
across batches; (4) the κ > 0 refusal; (5) cost: see the report (ns per shot).
