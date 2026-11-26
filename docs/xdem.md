# The xdem route: file-only sampling with decoder feedback

`Circuit.detector_error_model_text(format="xdem")` writes an **xdem** file: a detector error model whose lines may
carry, besides their Pauli flips, a diagonal-Clifford part. `xtim.XdemSampler(text)` reads it (or any conventional stim
DEM) and samples the detectors and observables from the file alone, with the same outcome law as
`compile_detector_sampler` on the circuit. A decoder's hypothesis (a list of the file's line ids per shot) is turned
into a correction and scored exactly, per shot.

```python
import xtim
c = xtim.load_example("cultivation_d3_faithful").text
text = xtim.Circuit(c).detector_error_model_text(format="xdem")
s = xtim.XdemSampler(text)
dets, status = s.sample(100_000, seed=7, fired=True)
lines = [status.fired[1][status.fired[0][k]:status.fired[0][k + 1]] for k in range(len(dets))]   # the genie decoder
print(1 - s.score(status, lines=lines).mean())            # the default R2b with Ê = E: 0.0 (no lossy readout)
r = s.infidelity(100_000, lambda d: [[] for _ in range(len(d))], seed=8, lines=True)   # the zero decoder
print(r.mean, r.stderr)                                    # 0.05278 0.000707...
```

Example circuit and xdem file pairs: [examples/xdem/](../examples/xdem/).

The genie decoder returns, for each shot, the lines that actually fired (`status.fired`, a CSR of line ids). On this
circuit every pass is then 1, because it has no late axis (`tc` = the number of lines) and no lossy readout. That is not
true in general. The default `"r2b"` is a truncated correction (rule R2b, §4.4), and under it the genie's infidelity can
be above 0: about 1e-5 to 1e-2 on in-repo circuits with a late axis or a lossy readout (rep3T_pre, toy2obs, rm15rep,
cultivation_d5). Under `correction="optimal"` the genie gives pass 1 on every shot except the lossy readouts, where
each lossy observable is a fair coin (cultivation_d5: 2.4 % of shots lossy, genie infidelity 1.2e-2; measured on 2e5
shots). So the genie score does not check R2b,
and it is not a bound. The zero decoder applies nothing; its infidelity is the logical failure rate with no correction
at all.

## 1. What the route computes

> **Definition (extended detector error model).** Take `n` *readout* qubits, `r` *gauge* qubits and `m`
> *hidden-syndrome* qubits, prepared in `|+>^{n+r} ⊗ |0>^m`. Fix `A` *axes*, each a signed Z-string on these qubits,
> `Ẑ_a = (−1)^{b̂_a} Z[Φ(a)]`, `Φ(a) = (img_a | κ_a | σ_a)` with `img_a` on the readout, `κ_a` on the gauge and `σ_a`
> on the hidden-syndrome qubits. On the noiseless state `Ẑ_a` reads `b̂_a` XOR the X-readout values of its gauge
> qubits (each uniformly random); the readout and hidden-syndrome qubits contribute 0. An *error mechanism* is a Pauli
> times a diagonal Clifford built from the axes,
>
>     M = Z[f] X[χ] ∏_a e^{iπ/4 E_a Ẑ_a},      f on the readout qubits, χ on the gauge and hidden-syndrome qubits, E_a ∈ Z_4,
>
> the rotation acting first. The model is a time-ordered list of mechanisms `(p_i, M_i)`; mechanism `i` fires
> independently with probability `p_i`; the fired mechanisms act in list order; then X is measured on the readout
> qubits. The hidden-syndrome qubits are never read. The outcomes are the detectors and observables, relative to
> their noiseless values. With `r = m = 0` and no rotations this is the conventional detector error model.

The state before readout is a stabilizer state, so the outcomes are uniform on an affine subspace (§3).

**Theorem (exactness).** Take a circuit of Clifford gates and the non-Clifford gates xtim's parser accepts (the
diagonal T, T_DAG, CS, CS_DAG, CCZ, and CH, which the writer rewrites exactly as two π/8 rotations about Y on its
target around a CZ: `CH(c,t) = (W T W†) CZ(c,t) (W T† W†)` on `t`, `W = S H`, with no global phase (the T and T†
phases cancel), i.e. two non-Clifford locations of the circuit, a `T_DAG` then a `T` on `t` in time order; the file
names no gate), whose axes commute and whose detectors and observables are deterministic on the noiseless *real*
circuit (not on its Clifford skeleton). Build the model with readout qubits = the detectors and observables, gauge qubits = the random
axis directions, hidden-syndrome qubits = the stabilizers of the noiseless output that are products of axes (modulo
±I), `b̂` = the stabilizer signs, and per fault `f` = the bits it flips, `χ` = an X-string anticommuting with exactly
the axes its propagated Pauli anticommutes with, `E` = its residues. Then the model reproduces the joint law of the
detectors and observables for every fault set.

The hidden-syndrome qubits: the noiseless output has stabilizers that are products of axes. They commute with every
detector, so they are never read, but a fault can flip one, and a later fault's rotations about the axes whose product
it is then act differently. The model keeps one `|0>` qubit per independent such stabilizer; a mechanism's X on it
records the flip, and a Z through it gives the later rotation its sign. With them each line is a well-defined
operator, which the correction (§4) needs.

Conventions:

| convention | choice |
|---|---|
| rotation sign | a line applies `∏_a e^{iπ/4 E_a Ẑ_a}`, `E_a ∈ {0, 1, 2, 3}` |
| order within a line | the rotation first, then `Z[f]`, then `X[χ]` |
| order of lines | lines `i_1 < … < i_k` in file order act in that order; the end operator is `M_{i_k} ⋯ M_{i_1}` |
| reference value | every readout bit is XORed with its noiseless value (stim's convention) |
| even exponents | a file carries only `±1`; the writer folds every `E_a = 2` into the line's flips (`e^{iπ/2 Ẑ_a}` is a Pauli: its readout part is the flip `img_a`, the rest is invisible) |
| `τ` for `χ` | the file stores `τ_i`, the axes the line's X part anticommutes with; `χ_i` is any X-string on the gauge and hidden-syndrome qubits with `χ_i · Φ(a) = τ_i(a)` |

Scope and premises: the axes must commute and the detectors and observables must be deterministic on the real
circuit; deferred classical feedback (`CX rec[-1] q`) is covered; a conditional non-Clifford gate and `DECISION`
columns are refused (§8).

## 2. File format (xdem, version 1)

An xdem is a conventional stim DEM text with, when some fault rotates, three plain extensions: a header line, axis
lines, and three optional fields at the end of an error line. There is no tag, no marker and no checksum. Nothing in
the file names a gate, a group or an x-pattern: every line is a Pauli plus diagonal-Clifford (π/4 Z-rotation) operator
on `|+>^{n+r} ⊗ |0>^m`, read out in X. File extension `.xdem`.

```
xdem 1 detectors <D> observables <O> gauge <r> hidden <m> tc <t>      (header: first line of an extended file)
axis Q<a> sign - img <D.. L..> gauge <g..> hidden <h..>              (one per axis, a = 0..A-1, before the first error line)
error(<p>) <targets> x <Q..> rot <±Q..> late <±Q..>                  (an error line; the three fields are optional)
error(<p>) <targets>                                                  (a conventional error line)
```

| field | meaning | spelling (one spelling per model) |
|---|---|---|
| `xdem 1` | format name and version | first token of the first non-comment line; its presence makes the file extended |
| `detectors observables gauge hidden` | `D O r m` | decimal, no leading zeros; an observable no line touches still exists |
| `tc <t>` | lines `0..t-1` are before `t_c`, lines `t..` after (error lines in file order) | `0 ≤ t ≤ #error lines` |
| `Q<a>` | axis id | `0..A-1`, each declared exactly once, in order |
| `sign -` | `b̂_a = 1`: `Ẑ_a = −Z[…]` | written only when negative |
| `img` | the readout qubits of the string: `D<j>`, `L<k>` | ascending, D's before L's; omitted iff empty |
| `gauge` | the gauge qubits of the string (`κ_a`): `g<t>`, `t < r` | ascending; omitted iff empty |
| `hidden` | the hidden-syndrome qubits of the string (`σ_a`): `h<t>`, `t < m` | ascending; omitted iff empty |
| `targets` | stim error targets (`D<j>`, `L<k>`, `^`): `f_i` | the writer emits ascending D's then L's, no duplicates, no `^` |
| `x` | `τ_i`: the axes the line's X part anticommutes with | ascending; omitted iff empty |
| `rot` | `E_i^<`, the rotations generated before `t_c`: `+Q<a>` for `e^{+iπ/4 Ẑ_a}`, `-Q<a>` for `e^{−iπ/4 Ẑ_a}` | ascending; omitted iff empty |
| `late` | `E_i^>`, the rotations generated after `t_c`, same syntax | omitted iff empty; an axis may appear in both `rot` and `late` |

A line *fires with probability `p`, flips `targets`, negates every later rotation about the axes in `x`, and rotates
by `e^{±iπ/4 Ẑ_a}` for each `±Q_a` in `rot` and `late`*. The law uses `E = E^< + E^>` only; the split is what the R2b
correction reads.

Conventional lines keep stim's syntax and meaning inside an extended file: `detector(…) D<j>` and
`logical_observable L<k>` are accepted (an index `≥ D` or `≥ O` is an error), `shift_detectors(…) <n>` shifts the
`D` indices of every later line (axis lines included), `repeat N { … }` is unrolled in place (its error lines count
toward `tc`), and `^` is a decomposition hint (the line flips the XOR of its targets). The writer emits none of these.
A file without the `xdem` header is a conventional DEM and is read with stim's semantics throughout.

**A circuit with no rotating fault gets the conventional DEM.** For a Clifford circuit, and for a non-Clifford circuit
none of whose faults carries a residue (the bundled 15-to-1 distillation with Z noise only),
`detector_error_model_text(format="xdem")` returns exactly `detector_error_model_text()` with the same options, byte
for byte. Such files parse in stim and in every DEM tool as what they are.

**What the writer emits** (the canonical form): readout qubits = the `D` detectors then the `O` observables, in stim
order; `r = rank κ` gauge qubits, the `t`-th pivot axis (its `κ` not in the span of the earlier axes') owning `g_t`
alone and a non-pivot axis listing the pivots its `κ` depends on; `m = dim K − dim K_I` hidden-syndrome qubits
(`K` = the axis products that are stabilizers up to sign, `K_I` = those equal to ±I); `b̂ = 0` on pivot axes and the
stabilizer sign on non-pivot ones; per line the flips, the strict `τ` (every axis the fault's propagated end Pauli
anticommutes with), and `E` split at `t_c`, each field folded. Lines are listed in the time order of their earliest
faults. Two faults share a line iff they have the same `(f, τ, E^<, E^>)`, lie on the same side of `t_c`, and every
line created or joined between them has an invisible swap with the fault (spec erratum E1: the readout image of the
swap set `(τ_j ∧ odd E) ⊕ (τ ∧ odd E_j)` is zero); probabilities combine as `p₁(1−p₂) + p₂(1−p₁)`.

### Worked file: the [[4,2,2]] example with both observables

The circuit is `tests/data/xdem/circuits/ex422_obs.stim`: three faults X, Z, Y on one data qubit, three detectors and
the observables `L0 = X1 X2`, `L1 = Z1 Z2`. Its xdem (`tests/data/xdem/ex422_obs.xdem`):

```
xdem 1 detectors 3 observables 2 gauge 3 hidden 2 tc 3
axis Q0 img D0 D1 D2 gauge g0
axis Q1 img D1 D2 gauge g1
axis Q2 img D0 D1 D2 L0 L1 gauge g2
axis Q3 img D1 D2 L0 L1 gauge g0 g1 g2
axis Q4 sign - img D0 D1 D2 L0 L1 gauge g2 hidden h0
axis Q5 sign - img D1 D2 L0 L1 gauge g0 g1 g2 hidden h0
axis Q6 sign - img D0 D1 D2 gauge g0 hidden h1
axis Q7 sign - img D1 D2 gauge g1 hidden h1
error(0.001) D0 D1 L1 x Q2 Q3 rot -Q2 +Q3
error(0.001) D2 L0 x Q2 Q3 rot -Q2 +Q3
error(0.001) D0 D1 D2 L0 L1
```

Reading it. The pivot axes `Q0 Q1 Q2` carry the gauge qubits `g0 g1 g2` alone. `Q3 = Q0·Q1·Q2` up to the identity:
no hidden-syndrome qubit, and its noiseless value is the XOR of the X-readout values of `g0 g1 g2`. `Q4` closes
`Q2·Q4`, a stabilizer of the noiseless output with value −1: the hidden-syndrome qubit `h0` and `sign -`. `Q5` equals
`Q4` modulo the identity relations, so it shares `h0`; `Q6` closes `Q0·Q6` (`h1`), and `Q7` equals `Q6`. The first
line flips D0 D1 L1, anticommutes with `Q2, Q3` (and not with `Q4`, although `Q2·Q4` is a stabilizer: that is what
`h0` records, `χ = X[g2 h0]`), and rotates by `e^{−iπ/4 Ẑ_2} e^{+iπ/4 Ẑ_3}`. Its law (§3): `Od = {Q2, Q3}`, the coin
span is `span{D0, D1 D2 L0 L1}`, and the deterministic parities are `D1⊕D2 = 1`, `D1⊕L0 = 1`, `D1⊕L1 = 0`. The first
two lines composed give `η = 0` on both axes and the all-flip, which is the third line, an ordinary DEM line. `tc 3`:
all lines are before `t_c`, there is no late axis, and R2b never applies.

## 3. The law

**The composite of a set of lines.** For lines `i_1 < … < i_k` (file order), with `τ_{<j} = ⊕_{l<j} τ_{i_l}`:

    f = ⊕_j f_{i_j},     τ = ⊕_j τ_{i_j},     η_a = Σ_j (−1)^{τ_{<j}(a)} · E_{i_j,a}  (mod 4),

and `M_{i_k} ⋯ M_{i_1} = Z[f] X[χ] ∏_a e^{iπ/4 η_a Ẑ_a}` up to a global phase. Each later line's rotation about `a` is
negated once for every earlier line whose X part anticommutes with `Ẑ_a`. Firing a line twice is the identity, so
repeated ids in a hypothesis cancel, and a hypothesis may list its ids in any order (they are composed in file order).
The sampler's noise walk may reach a shot's fired lines out of file order; it sorts them before composing.

**The law of a composite.** Let `Od = {a : η_a odd}` and, for a parity `J` of readout bits, `A_J = {a : img_a·J = 1}`
and `B_J = A_J ∩ Od`. Then

- `J` is deterministic iff `κ_{B_J} = 0`;
- its value is `f·J ⊕ (Σ_{a∈A_J} η_a)/2 ⊕ b̂_{B_J}` (mod 2);
- the outcome vector is uniform on `offset + coins`, `coins = rowspace(Σ_{a∈Od} κ_a img_aᵀ)`.

Every file the reader accepts defines a law. The hidden-syndrome bits never enter the law of the lines alone; they
enter through `τ` and through the correction. Per shot, `status` keeps the space `C` of observable values that the
gauge qubits leave undetermined given the detectors, so `score(status, flips)` = `2^-dim C` if `(v XOR flips) ∈ C`,
else 0.

## 4. The decoder loop

    |init>  →  fired lines (time order)  →  X readout of the detectors: s  →  decoder: Ê (line ids, or give up)
           →  ONE correction line K(s, Ê)  →  X readout of the observables: o;     pass = Pr(o = c | s, E, K).

The optimal `K` acts after every line; R2b's acts at `t_c`, between lines `tc − 1` and `tc`. Either way `K` commutes
with the detector readout, and the score is the X-readout law of the extended list `[lines, K]`: one stabilizer law
(§3) of the extended composite.

**Decoder interface.** In: the detectors as `bool[n, D]` (reference-relative). Out: per shot a list of the file's
line ids (file order of the error lines; repeats cancel; empty = apply nothing), optionally with a give-up flag.
Line-naming decoders include tesseract with `merge_errors=False` (its ids refer to the Pauli surrogate, which has the
same error lines in the same order once the extended fields are stripped) and MLE tables over the file's lines.
Flips-only decoders such as pymatching use `score(status, flips)`.

**The corrections** (`correction=`):

- `"r2b"`, **the default**. The hypothesis' late part as one Pauli at `t_c`. Available iff every line with index
  `≥ tc` has no detector target and no `rot`/`late` field (it is a classical readout flip) and no late axis has a
  detector in its `img`; `s.r2b_available` gives `(ok, reason)`, and `correction="r2b"` on a file without it raises
  `XtimXdemError` naming the reason (pass `"optimal"` or `"none"`). Per shot: compose `Ê_< = {i ∈ Ê : i < tc}` with
  the cocycle restricted to the late parts, giving `η^>` and `τ_K` on the late axes;
  `K_R2b = X[χ_K] ∏_a e^{iπ/4 η^>_a Ẑ_a}` is inserted at `tc`; it is applied iff `η^> ≠ 0` and it does not raise the
  observables' undetermined dimension given the detectors (the guard); `c` is the chosen law's definite observable
  value at `s` when there is one, else the hypothesis' observable flips.
- `"optimal"`, **opt-in**: the reference, best-possible correction, and slower. Compose `Ê`; build from the detector
  parts of the odd axes `Ω = Σ_{a∈Od} ı_a ı_aᵀ` and its greedy pivots `P` (detector order); per observable `k` solve
  for the pivots `P_k`. Without a solution the readout is **lossy**. Otherwise `K` is the product of the S and CZ gates
  that undo the hypothesis' rotation of the readout `X_{L_k} X_{P_k}`, with a Pauli frame from `s·P_k`, and the
  prediction is `c = 0` on the non-lossy observables (lossy ones: below). `K`'s strings are scored with
  `κ' = [κ | δ]` and `b̂ = 0` (spec errata E2 and E3). With `Ê = E` and no lossy readout every pass is 1.
- `"none"`: law-aware flips: `c` = the hypothesis law's definite observable value given `s`, else its observable
  flips.

**Lossy readouts.** Lossy readouts are common on real circuits under the true hypothesis, not rare (spec erratum E2).
On the bundled `cultivation_d5` example, 24 189 of 10^6 shots (2.42 %; `sample(10**6, seed=7, fired=True)` scored with
`lines=status.fired, correction="optimal"`) have a lossy readout. `counters()["lines_lossy"]` counts them. The rule
for a lossy observable (spec erratum E4): `K` is built for the non-lossy observables only, with no string on the lossy
ones; their frame bits are a point `c` of the support of `law([Ê, K])` restricted to the lossy observables, given `s`.
The point is fixed by a deterministic rule: the support point whose difference from `Ê`'s observable flips on the
lossy observables is lexicographically least, observable 0 first. So `c` equals `Ê`'s flips there whenever they are
in the support (the prediction of `"none"` and of R2b step 4); when `s` is outside `Ê`'s detector support there is no
point and `c = 0` (spec §7 Q1). The score is the one law of `[lines, K]` at that `c`; the joint law of several lossy observables is computed,
not assumed to be a product. Independent lossy observables score ½ each. A lossy parity that the hypothesis
determines (two lossy observables whose XOR is fixed given `s`) is predicted, not discarded. With the true hypothesis
`"optimal"` never scores below `"none"`: such a parity commutes with every non-lossy rotated readout, so `K`'s CZ edges
cannot add a coin, and `c` is a point of the support.

**Scoring.** `pass ∈ {0} ∪ {2^{−k}}` is `Pr(o = c | s, E, K)`; the infidelity is `1 − mean(pass)` over the accepted
(post-selected) shots. A give-up shot is scored with what the decoder returned and counted in `giveup_fraction`.
Counters: `lines_shots`, `lines_applied` / `lines_withheld` (R2b), `lines_coin_after` (an undetermined observable
after the correction), `lines_lossy`, `lines_outside` (shots whose `s` has probability 0 under `Ê`'s own law; `K` is
applied anyway, with no fallback).

`s.correction(status, lines=L, correction=...)` exports the correction per shot as an `XdemCorrection`, for a physical
implementation. For `"optimal"`: `axes` (the axis ids `S_k`), `y`, `frame`, `pivots` (`P_k`) and `lossy`, meaning
*measure, in place of observable `k`, the Hermitian Pauli `(−1)^{frame} · ι · X_{L_k} ∏_{a∈S_k} Q_a` (`ι = i` iff
`y`), whose noiseless value is +1*. For `"r2b"`: `late` (`η^>` per late axis), `late_tau` (`τ_K`), `applied` and
`c`: *a Pauli at `t_c` anticommuting with exactly the late axes in `late_tau`*. Mapping axis ids to physical Paulis
needs the circuit.

## 5. Python API

```
text = xtim.Circuit(c).detector_error_model_text(format="xdem")    # the conventional DEM when nothing rotates
dem  = xtim.Circuit(c).detector_error_model(format="xdem")         # stim.DetectorErrorModel; ValueError if extended
xtim.xdem.canonical_report(circuit)                                # (True, None) or (False, reason): §7

s = xtim.XdemSampler(text, *, cache_cap=65536, device="cpu")       # an xdem, or any conventional stim DEM;
                                                                   # device="gpu" | "metal" | "cuda": §11
s.device                                                           # "cpu", or the GPU in use
s.extended                                                         # False for a conventional DEM
s.num_detectors, s.num_observables, s.num_gauge, s.num_hidden, s.num_axes, s.num_lines, s.num_rotating
s.tc                                                               # first late line (num_lines for a conventional DEM)
s.late_axes                                                        # axes with a nonzero late exponent on some line
s.r2b_available                                                    # (bool, reason)
s.invisible_residues                                               # rank img[A1] - rank img[A1]|_D (a property)

dets, status = s.sample(shots, *, seed, threads=1, postselect=None, fired=False)
dets, status, counts = s.sample(shots, *, seed, threads=1, postselect=mask_or_indices, fired=False)
passes = s.score(status, flips, *, bit_packed=False)
passes = s.score(status, *, lines=L, correction=None)              # None = "r2b"; "r2b" | "optimal" | "none"
r = s.infidelity(shots, decoder, *, seed, threads=1, batch=100_000, postselect=None, lines=False, correction=None)
r = s.infidelity(shots, None, *, seed, ...)                        # decoder=None: the zero decoder (no lines=True)
corr = s.correction(status, *, lines=L, correction="r2b")          # "r2b" | "optimal"
s.counters()
xtim.score_plain(obs, flips, *, bit_packed=False)                  # the plain XOR score for any bit sampler
```

- `lines=` takes a list of per-shot integer sequences or arrays, or a CSR pair `(offsets, ids)` as a tuple; it needs
  a status from `sample(..., fired=True)`. `correction=` applies to `lines=` only: with flips it must stay None.
- `infidelity(..., lines=True)`: the decoder returns line-id lists (or `(lines, giveup)`); otherwise flips (or
  `(flips, giveup)`).
- Stream: the shots of `sample(shots, seed=s)` are cut into blocks of 1024; block `b` depends only on `(s, b)`, never
  on `threads`. `infidelity(shots, decoder, seed=s)` consumes exactly the shots of `sample(shots, seed=s)`, in order,
  whatever `batch` is. The same seed gives the same shots: use a new seed for new shots.
- Postselection (`postselect=`: a bool mask of length D, or detector indices) runs in C++; `shots` are attempted
  shots, and the survivors are distributed exactly as `sample()` followed by the filter. A shot whose detectors on the
  mask cannot be zeroed by any axis combination is rejected before its gauge-qubit work (exact, counted in `early`).
  With `postselect=`, `sample()` returns the 3-tuple `(dets, status, counts)`: `dets` and `status` hold the survivors
  only, and `counts` is a dict with the keys `attempts` (= `shots`), `passed` (the survivors, = `len(dets)`),
  `discarded` (`attempts - passed`) and `early` (the discarded shots that the span test rejected before their
  gauge-qubit work). A non-bool `postselect` is read as detector indices. `sample()` refuses a repeated index (pass a
  0/1 mask as `dtype=bool`); `infidelity()` accepts repeats (it builds the mask itself, so a repeat changes nothing).
  The spec does not fix this rule; this is the current behaviour. An empty selection rejects nothing.
- Refusals of a file raise `xtim.XtimXdemError` (a subclass of `xtim.XtimError`, not of `ValueError`) naming the cause
  and the text line. On a GPU sampler (`device=` other than `"cpu"`, §11) a missing or failing device, an
  unsupported file and the calls the GPU does not offer raise `xtim.XtimGpuError` (also a subclass of
  `xtim.XtimError`).

## 6. Speed notes

- Sampling (`fired=False`) keeps the conventional noise walk: lines with equal `p` form chains, walked with geometric
  skipping. A conventional DEM samples byte-identically to the previous file-only route's plain-DEM mode.
- Per shot, the fired extended lines are gathered, sorted into file order and composed with a sign register over the
  axes, at cost `O(Σ_fired (|τ_i| + |E_i|))`; there is no residue lookup at sampling time.
- Every file samples on the law's fast path: coordinates over a basis of the axis images, no cache; one word per
  vector when the images span at most 64 dimensions and `r ≤ 64`, else `ceil(max(r, rho) / 64)` words. A general
  path with a per-thread law cache keyed by the odd-axis set (bounded by `cache_cap`; the cache never changes an
  output byte) remains as the test oracle only. Scoring with lines, the correction exports and `infidelity` read the
  same fast laws at every size (the optimal correction's law of the corrected shot and its detector support of the
  hypothesis' law too). Files beyond one word draw
  their coin bits exactly as the general path does, so both paths give the same bytes there.
- The fast law is computed in graph-state form: the gauge basis is re-expressed once at load in the axes' image
  coordinates, so the per-shot law matrix is the symmetric adjacency `G = Σ_{a odd} t_a t_aᵀ` of the odd axes; the
  phases on its kernel come from the mod-4 quadratic form of `G` (no second pass over the odd axes), and the kernel
  from an elimination that visits only the pivots a column holds, in the same column order. Same kernel basis, same
  coin words, same bytes as before. The `"optimal"` correction's law of the corrected shot (whose gauge coordinates
  carry the observables too) is computed the same way, its basis change made once per sampler, on the first
  `"optimal"` call (tens of µs on bt81-sized files, a few ms at 1500 gauge qubits).
- `"r2b"` costs at most three law evaluations per shot without a coin draw. `"optimal"` is the reference path: it is
  slower (scoring took 1.2 to 5.3 times the time of `sample(fired=True)` on the development benchmarks), and its speed
  is reported, not gated. `K`'s diagonal part is memoised per distinct hypothesis.
- Both corrections run in C++ under the sampler's lock; Python only converts the decoder's lists into a CSR.

## 7. Canonical form

A circuit is in **canonical single-stage form** when the protocol (all detectors) comes first and the logical readout
after it, with no noise inside the readout's non-Clifford dressing. Let `t_c` be the point right after the instruction
that measures the last detector record (the start of the circuit if no detector reads a record). The circuit is
canonical iff

- (C1') every observable reads at least one record measured after `t_c`. A record in the same instruction as the last
  detector record (a final `M`, or one `MPP` with several products) is not after it. Records measured before `t_c`
  are allowed: they are a classical frame, XORed in. An observable with no record (all cancelled) violates C1'.
  Why: the observable's readout must happen after the detectors are complete; an earlier record is already a
  classical bit and changes nothing about that.
- (C2') no noise acts between `t_c` and the last non-Clifford gate after `t_c`: no noise channel with a nonzero
  probability and no noisy measurement (`MX(p)`, `MPP(p)`, ...) there. Noise after that gate, or anywhere after `t_c`
  if no non-Clifford gate follows `t_c`, is allowed. Heralded noise is refused at parse time anyway.
  Why: after the last non-Clifford gate, a Pauli fault only flips readout bits (a classical flip, e.g. `Z_ERROR(p)`
  before an `MX`); before it, a fault is dressed by the non-Clifford readout and is not one Pauli at `t_c`.

Gates of any kind are allowed after `t_c`, non-Clifford readout dressing included, and so are feedback and noiseless
measurements. Record sets are the declared ones (`DETECTOR` / `OBSERVABLE_INCLUDE` targets, a record listed twice
cancels), and instructions are counted with REPEAT blocks unrolled. The check runs on the parsed circuit, not on the
deferred one the writer builds, because deferral moves every measurement to the end.

Why the form: the decoder's input (the detectors) is complete at `t_c`, and the readout after it is a fixed map of the
state at `t_c` whose only noise is classical readout flips, so a correction can be stated as one Pauli at `t_c`. The
file itself states what R2b needs (`r2b_available`, §4); the circuit conditions stay in `canonical_report`.

`detector_error_model_text(format="xdem", require_canonical=True)` (and `detector_error_model`) refuses a
non-canonical circuit with `xtim.XtimDemError` naming the first violation (observable and record, or noise channel,
with source lines) and the number of others. The check runs when the output is extended; the option is never
forwarded to a conventional export, and `require_canonical=True` with `format="stim"` is a `ValueError`.
`xtim.xdem.canonical_report(circuit)` (an `xtim.Circuit` or its text) runs the same check without exporting and
returns `(True, None)` or `(False, reason)`.

## 8. Refusals

**The reader** (`XdemSampler(text)`) raises `xtim.XtimXdemError` naming the cause and the text line for:

- structure: an unknown version; a header field unknown, missing, repeated or out of range; an axis line after the
  first error line, out of order, duplicated or missing; an undeclared axis or coordinate; a non-canonical spelling
  (a duplicate, unsorted or leading-zero token, a double or trailing space, a field out of order, `sign` with
  anything but `-`, an empty field written out, a `rot`/`late` token without `+`/`-`, an axis repeated inside one
  field); an axis with no qubit; an `x`/`rot`/`late` field in a file with no xdem header; more than `2^31 − 1`
  readout qubits;
- header consistency: `rank[img | κ] ≠ rank κ` (some axis product with `κ_B = 0` has `img_B ≠ 0`);
- a line that is not an operator: `τ_i` not orthogonal to `ker(κ | σ)` (no `χ` exists);
- a dc-DEM v1 file (§9).

There is no check on `b̂`, no checksum, and no W-phase refusal: a hand-edited file is a different, valid model. A
file cut short parses as a smaller model when the cut leaves `tc` in range; a cut below `tc` is refused by name.

**The writer** (`detector_error_model_text(format="xdem")`). On an extended output a set option is a `ValueError`
naming it: `decompose_errors=True` (an xdem line is a fault, not a graphlike edge), `ignore_decomposition_failures`,
`erasure=True`, `drop_gauge_observables=True`, an explicit `clifford_cutoff` (nothing is cut). `XtimDemError`, naming
the cause: `include_expectations=True` with a `PAULI_EXPECTATION` on an extended output; and, with non-Clifford gates
whatever the output, a `DECISION` column, a residue that does not stay diagonal (a
location's Z carried to a later location with X or Y on its qubits; both locations are named, a CH's as
`T_DAG (CH, rotation 1/2)` / `T (CH, rotation 2/2)`: for example a T-state fed into a CH without aligning its axis,
which xtim's class check may also reject first, `XtimRejectError`),
CH-class fault residuals together with classical feedback (the DEM export's own refusal), IF blocks, a
non-deterministic detector or observable, a disjoint noise channel with no exact conversion (unless
`approximate_disjoint_errors` is given, as in stim), and, with `require_canonical=True`, a non-canonical circuit (§7).
`detector_error_model(format="xdem")` raises `ValueError` on an extended text (stim cannot hold it), and
`detector_error_model_with_reject` refuses `format="xdem"` (the xdem law has no post-selection region).
Most extended exports build no exact-engine bare state (determinism is decided algebraically from the writer model),
so they have no stabilizer-rank (2^30) or state-budget limit. The others go through the exact exporter, which keeps
that limit and can still raise `XtimCapacityError`: a circuit with a `PAULI_EXPECTATION` declaration, a circuit with
an H-type gate (H, SQRT_X, SQRT_Y, H_YZ, C_XYZ, …: a single-qubit Clifford that maps Z to X or Y) acting on a qubit where the Z of an earlier non-Clifford location has support (even when it exports an
extended file; every circuit with a CH is one, since its rewrite puts an H on the wire of its first rotation), and a circuit the export refuses for another reason. A non-deterministic detector or observable
is refused on the algebraic route itself, at any size, with the exact exporter's message (the first one and its
noiseless expectation); `XtimDemError.nondeterministic` and the hint list every one and name the fix (add the Pauli
feedback that corrects the measured byproducts, or drop them).

## 9. Migration

The format this replaces was **dc-DEM** (diagonal-Clifford DEM, development builds only): stim DEM text with `dc:` tags
on error lines, header data in tagged `shift_detectors` no-ops and a numeric end marker. Its export format, reader and
sampler are removed; it never shipped in a release.

| case | behaviour |
|---|---|
| a dc-DEM v1 file (`shift_detectors(6463, 1, …)`, `dc:` tags) | **dc-DEM v1 files are refused: re-export.** The xdem reader names the cause ("dc-DEM v1 file: re-export the circuit with `format="xdem"`"). There is no converter: a v1 file lacks the strict `τ` and the hidden-syndrome qubits (circuit data), so a faithful xdem cannot be built from it. |
| an xdem on stim | stim fails at the first extended token (`xdem`, `axis`, or the `x` field) rather than misreading it; a Clifford-only xdem is a plain DEM and parses |
| a decoder that names lines | the `score(lines=)` call is the same on `XdemSampler`; `correction="r2b"` reproduces the previous route's numbers shot by shot (checked before the removal: 16 runs, max \|Δ\| = 0). Line ids change (the strict merge splits some lines): line-keyed tables are rebuilt from the new file |
| Clifford-only circuits | the conventional DEM, byte-identical to the ordinary export |
| `canonical_report`, `require_canonical=True` | under `xtim.xdem`, unchanged on the circuit side; the export no longer writes a `can` field |

The default export (`format="stim"`) and `compile_detector_sampler` are unchanged, byte for byte.

## 10. The two-route check

The xdem route and `compile_detector_sampler` compute the same outcome law by different means: the xdem route from the
file alone, the existing route from the circuit. The test suite compares them:

- per fault configuration (`tests/test_xdem_oracle_configs.py`): the file's law of the lines carrying the faults
  (offset modulo the coin span, and the span) against xtim's exact engine (a test-only oracle), on every in-repo
  circuit of the registry with an extended export: singles, pairs, random sets, faults sharing an axis, and two
  faults on one line; exact over GF(2);
- per fault subset (`tests/test_xdem_export.py::test_law_per_fault_subset_against_the_exact_engine`), including
  hand circuits for the merge rule;
- on random small circuits (`tests/test_xdem_fuzz.py`): the file's exact outcome law against xtim's exact law, total
  variation ≤ 1e-12; and the corrections against the engine's own forms, exactly;
- by sampling, route against route (`tests/test_xdem_cross_route.py`): the xdem route (export → `XdemSampler.sample` →
  `score` with the zero decoder) against `compile_detector_sampler` (a new seed per chunk on both sides) →
  `score_plain` with the zero decoder. Statistics: the pass mean, the acceptance rate, every detector's marginal and,
  with an observable, every detector's joint with failure `E[D_j (1 − pass)]` (the file's pass is the conditional
  expectation of the XOR pass given the detector bits and the fired lines). Fail if any `|z| > 5`; 310 z-tests over the
  in-repo cases, `P(any |z| > 5) ≤ 1.8e-4` under agreement (union bound). The long legs run with `--runslow`.

## 11. GPU sampling

`XdemSampler(text, device="gpu")` samples the same files on the machine's GPU. It is for the large runs: millions of
shots of an extended file, where the one number you want is an infidelity or an acceptance. The default,
`device="cpu"`, imports nothing GPU-related and is unchanged byte for byte.

**Status of the two backends: Metal tested (Apple M3 Max); CUDA experimental, validated on one NVIDIA card.** The
Metal backend (Apple silicon) is tested on an Apple M3 Max: its kernels give byte-identical output to a CPU reference
build of the same kernels, and its samples agree statistically with the CPU sampler's. The CUDA backend (NVIDIA) is
experimental and has been validated on **one NVIDIA GPU**, a Quadro RTX 6000 (compute capability 7.5, driver
580.196.07, CUDA 13.0, CuPy 14.2.0): on that card it is byte-identical to the C++ reference build on every kernel case
(148) and every entry-point case (96), and the CUDA tests pass (42). That is still one card, one driver and one CuPy
version: run `python tools/gpu_validate.py --device cuda` on your machine before you rely on it
([tools/GPU_VALIDATE.md](../tools/GPU_VALIDATE.md) is the runbook; it writes a JSON report and exits non-zero on a
failure). The CUDA build needs block shared memory, `__syncthreads`, 32-bit `atomicAdd` / `atomicXor` and 64-bit
`atomicXor` (compute capability 3.5 or later); only compute capability 7.5 has run so far. The reports are
[assets/gpu/metal-m3max.json](assets/gpu/metal-m3max.json) and
[assets/gpu/cuda-rtx6000.json](assets/gpu/cuda-rtx6000.json), and the speed measurements on both machines, with the
caveats, are in [performance.md](performance.md#how-fast-is-the-gpu-sampler).

### Install and devices

- Apple silicon: `pip install "xtim[metal]"` (the pyobjc Metal bindings).
- NVIDIA: install the [CuPy](https://cupy.dev) wheel that matches your CUDA version (`nvidia-smi` shows it), for example
  `pip install cupy-cuda12x`. The `xtim[cuda]` extra is empty on purpose, because CuPy ships one wheel per CUDA major
  version.

No GPU code is compiled when you install xtim: the kernels are compiled at first use.

`device` is `"cpu"` (the default), `"gpu"`, `"metal"` or `"cuda"`. `"gpu"` takes what `xtim.gpu.available()` returns:
it tries Metal (an arm64 Mac with pyobjc's Metal), then CUDA (CuPy and an NVIDIA device), and returns the first that
opens, or `None` when none does (then `device="gpu"` raises `xtim.XtimGpuError`). `XdemSampler.device` is the device
in use (`"gpu"` resolves to its name), and `repr(s)` shows it. `import xtim.gpu` first: `xtim.gpu` is not imported
by `import xtim`.

The kernels are compiled and the file's tables uploaded when the sampler is built, so a device that cannot run them
fails there with `xtim.XtimGpuError`. A device failure later (an allocation, a launch, a lost device) is also
`xtim.XtimGpuError`, carrying the device's own message; `device="cpu"` is the remedy. A GPU sampler may be shared
between threads: its calls are serialised (one at a time on the device), and each returns what it would alone.
`s.counters()` on a GPU sampler counts the shots drawn on the device (`shots`) and its submissions (`gpu_batches`,
`gpu_relaunches`); the CPU-only counters `ext_shots` and `coin_shots` are `None` there.

```python
import xtim
import xtim.gpu
c = xtim.load_example("cultivation_d3_faithful").text
text = xtim.Circuit(c).detector_error_model_text(format="xdem")
print(xtim.gpu.supports(text))                          # (True, None)
device = "gpu" if xtim.gpu.available() else "cpu"       # the CPU is the fallback on a machine without a GPU
s = xtim.XdemSampler(text, device=device)
dets, status = s.sample(100_000, seed=7)                # the same return types as on the CPU
flips = [[0] * s.num_observables for _ in range(len(dets))]
print(s.device, s.score(status, flips).mean())          # the zero decoder's mean pass value
```

### What runs on the device

On a GPU sampler these work as on the CPU (same arguments, same return types):

- `sample(shots, seed=...)`, with `postselect=` too: the survivors come back with the `counts` dict. The device
  writes each returned row in its final layout (detector bytes, observable bytes, C id) and, with `postselect=`,
  compacts the survivors itself, so the host copies each batch once. The C ids come from a table of the C the sampler
  has already met (up to 4096 of them), kept on the device; a C new to it is looked up on the host, in shot order, so
  the ids are those of the CPU table either way.
- `sample(shots, seed=..., status=False)` (with or without `postselect=`): only the detectors and the observables,
  `(dets, obs)` or `(dets, obs, counts)`, with `obs` the bytes `status.obs` would hold. The device then builds no C
  key and the host looks up no C id, while the rows stay the default call's, byte for byte. Use it when you decode
  the observables yourself and need no `score`: on the Bravyi-Haah files most shots carry a C the device table does
  not hold, and looking those up on the host is most of the default call's time (Apple M3 Max, 2^22 shots, min ns
  per shot, default -> `status=False`: `merlin_bh36` 1334 -> 6.4, `merlin_bh50` 3251 -> 8.2, `merlin_bh70` 8441 ->
  19.5, about the device estimate's time; with post-selection on every detector, and on files whose C ids the
  device table holds, such as `cultivation_d5`, the two are within about 20%). In the committed report (generated
  Bravyi-Haah-like files, 36 and 70 observables) the `status=False` call runs at 166 and 119 times the default call's
  rate. On the CPU, `status=False` computes the status and drops it (the same rows, the default's cost).
  `fired=True` needs the status, so `status=False` refuses it.
- `score(status, flips)`.
- `infidelity(shots, decoder, ...)` with a decoder that returns flips (`lines=False`). The shots are sampled on the
  device and handed to your decoder on the host, so a Python decoder, not the GPU, sets its speed.
- `infidelity(shots, None, ...)`, the zero decoder (flips all 0), with or without `postselect=`. On a GPU this runs
  **entirely on the device**: sampling, post-selection and scoring, and only per-block counters come back. It is the
  fastest GPU path: on the M3 Max 1.72 to 6.71 times the best CPU route (the faster of `infidelity(shots, None)` and
  `sample` + `score` with zero flips, at all 14 threads) on the bundled cultivation files and two generated
  Bravyi-Haah-like files (see [performance.md](performance.md#how-fast-is-the-gpu-sampler), from a 4.2.0 development
  build). The estimate needs only dim C and whether the observable row lies in C, so the device does not build C's
  key there.
  `mean` and `stderr` are computed exactly from the counts. The CPU accepts `decoder=None` too, and gives the same
  result, bit for bit, as a decoder that returns zeros (it skips unpacking the detectors such a decoder ignores).

```python
# (continuing the block above)
import numpy as np
mask = np.zeros(s.num_detectors, dtype=bool)
mask[:10] = True                                        # post-select: the first ten detectors must not fire
r = s.infidelity(1_000_000, None, seed=1, postselect=mask)
print(r.accepted, r.mean, r.stderr)                     # accepted shots, 1 - mean pass, its standard error
```

`threads` is accepted on a GPU sampler and ignored. About `batch` in `infidelity`: with a decoder it is the number of
shots sampled, decoded and scored at a time, rounded up to a multiple of 1024, as on the CPU. With the zero decoder on
a GPU it plays no part, because the device sizes its own batches: at most 2^24 shots each, at most half of the
device's free memory, and at most about a second of expected work on the M3 Max (estimated from the file's mean
fired and extended lines per shot). Every bundled example runs at 2^24 shots per batch; a DEM firing 4000 lines per
shot gets batches of 192 000 shots, each 0.36 to 0.70 s on the M3 Max (the validation report). No output depends on the batch size. A
wait on the device cannot be interrupted, so Ctrl-C takes effect at the end of the current batch.

### Which files

The GPU takes every file the CPU samples on its fast path, which is every file `XdemSampler` reads: one coordinate
word (at most 64 gauge qubits and a span of at most 64 dimensions, or no extended lines) and several (more gauge
qubits or a larger span: the multi-word law, with no limit on either; its working vectors live in device memory, so
memory sets the batch size, not a cap). Two limits remain: at most 4096 axes (the kernels' per-thread memory), and
dim C at most 253 (the status keeps dim C in a byte). The number of extended lines a shot fires has no limit either:
the per-block lists live in device memory and grow as needed (tested at capacities past 2^17 records per block).
`xtim.gpu.supports(text)` returns `(True, None)` or `(False, reason)` without needing a GPU. A file over a limit
raises `xtim.XtimGpuError` when you build the sampler, with the numbers in the message, for example `5000 axes: the
GPU sampler handles at most 4096 axes (its per-thread memory); use device='cpu'`. Every file in
[examples/xdem/](../examples/xdem/) is supported. On a multi-word file the GPU draws the law's coins differently from
the CPU (a uniform vector of gauge coins rather than coins over a basis of the span): the law is the same.

**The structured law.** Some one-word files have a law that factors: a small core of gauge qubits and detector
coordinates, many "leaves" (one gauge qubit and one observable coordinate each) and axes that touch the core and one
leaf, plus a few that touch every leaf the same way. The Bravyi-Haah distillation files (`merlin_bh*`) are the
example. For such a file the device computes the law from one word per kind of axis instead of an elimination over
the span. `xtim.gpu.structure` finds the structure, checks that the file's T is linear in its gauge part (always so
for a valid xdem file), and renumbers the axes for the kernel. The result is the same C on every shot (dimension and
key), the same zero-decoder verdict and the same distribution. The correction may differ from the generic law's by an
element of the coin span, which is the same law with the coins relabelled. So on these files a seed gives other rows
than the generic law gave, but not other statistics. The GPU uses this law by default when it accepts a file whose
span has more than 32 dimensions; at 32 or fewer, the generic law is as fast (`GpuBackend(structured=True)` forces it
wherever the file is accepted). None of the bundled examples takes it (their spans have at most 15 dimensions and
the detector refuses the cultivation files); the validation tool and the tests use generated Bravyi-Haah-like files
(`bh_like_text` of `tests/_xdem_synth.py`) and, in the tests, the external `merlin_bh*` circuits when present
([not in this repository](performance.md#benchmark-files-not-in-this-repository)). On the M3 Max the device estimate went
from about 86 to 11.6 ns per shot on `merlin_bh50` and from 47 to 8.0 on `merlin_bh36` (about 9.5 and 7.0 with the
later changes to the walk: a one-pass sort of its lists and, on these files, blocks of 32 shots per walk thread).
`sample` there is limited by the host, not by the device: most shots meet a C the device's table of 4096 C does not
hold (81% on `merlin_bh50`, which meets about 0.6 new C per shot), and interning those keys in the CPU's table of C
takes over 99% of the call. A file that is not accepted keeps the generic law.

The same law runs on multi-word files (a span over 64 dimensions: `merlin_bh70`, `merlin_bh100`), where the generic
law is an elimination over several words per vector in device memory. There each vector of the law is a few 64-bit
words in thread-local memory (one word per 64 coordinates of the span), at most 4 words, so spans of up to 256
dimensions; a wider file, or one that is not accepted, keeps the generic multi-word law. The guarantees are the
same: the same C on every shot, the same zero-decoder verdict, the same distribution, rows differing from the generic
law's by a coin shift. This law runs only in the kernel that finishes the shots queued by the main resolve kernel
(`GpuBackend(defer=False)` is ignored there): compiled into the main kernel as well, it gave wrong results on Metal
that neither the second kernel nor the C++ reference build of the same source showed. On the M3 Max the device
estimate went from about 800 to 23 ns per shot on `merlin_bh70` and from 1590 to 31 on `merlin_bh100` (with
post-selection on every detector, 9.4 to 6.6 and 17.3 to 10.4). Of `merlin_bh100`'s 32 ns, the law takes about 21
and the walk 10. `sample` on these files is limited by the host's lookup of C ids, about 8 and 12.5 microseconds per
shot. The backend (`xtim.gpu.backend.GpuBackend(..., structured=None | True | False)`) reports the law it runs as
`law_path` (`"structured"`, `"generic"`, `"multi-word structured"` or `"multi-word"`) with `law_path_reason`, and
`tools/gpu_compare.py` prints it per circuit.

### The walk

The first kernel draws which lines fire. For each chain of lines with the same probability p, a block of B shots
has M x B slots (M lines), and the kernel walks them with geometric skips drawn from exact integer thresholds: each
slot fires independently with probability exactly its threshold / 2^64, as on the CPU. The host cuts each chain's
slots into segments of about 8 expected fires, each with the thresholds of its own length, and hands them out as
items, the most expensive first. One group of threads walks a block: each thread takes items in turn, applies the
fires to the block's rows (atomic XORs), and appends the extended records; the group then sorts each shot's records
into file order, so nothing depends on which thread walked which item or on the order of the atomics. Each item
draws from its own counter stream, keyed by the seed, the block and the item: a seed's shots do not depend on the
batch, the device or its number of threads. Defaults on the M3 Max: blocks of 512 shots, 64 threads per group, rows
in device memory; on a file firing at least 3 lines per shot the post-selection span test also runs in the walk, so
a rejected shot never reaches the law.

Measured on the M3 Max (`tools/gpu_compare.py`, 2^22 shots, min of 5 calls, old and new runs alternated; ns per
attempted shot, before and after this walk):

| circuit | estimate | estimate, post-selected | sample, post-selected |
|---|---|---|---|
| `merlin_bh12` | 2.10 -> 1.78 | 1.11 -> 0.79 | 1.52 -> 1.21 |
| `merlin_bh36` | 6.90 -> 5.89 | 3.70 -> 1.44 | 3.72 -> 2.20 |
| `merlin_bh50` | 9.11 -> 7.79 | 3.13 -> 1.78 | 3.80 -> 2.38 |
| `merlin_bh70` | 20.5 -> 18.7 | 5.06 -> 2.66 | 5.65 -> 3.13 |
| `cultivation_d5` | 2.64 -> 2.07 | 1.58 -> 1.03 | 1.89 -> 1.32 |
| `cultivation_d3_faithful` | 0.74 -> 0.66 | 0.65 -> 0.57 | 0.97 -> 0.90 |
| `magna_rp2-d5` | 4.67 -> 3.94 | 2.23 -> 1.50 | 2.52 -> 1.79 |

The walk's own GPU time went from 2.76 to 1.51 ns per shot on `merlin_bh50` (post-selected estimate, the span test
included) and from 1.09 to 0.56 on `cultivation_d5`; it draws 14.8 instead of 20.5 64-bit numbers per shot on the
first and 5.1 instead of 7.1 on the second. `sample` without post-selection is unchanged where the host's lookup of
C ids dominates (the `merlin` files). Tried and dropped: the rows in group memory (fewer groups fit on a core; slower
on every file), deciding the first tie of each word without a branch (0.02 to 0.16 ns per shot slower), and a
32-bit-half generator for the words (no consistent gain, and the 64-bit SplitMix finalizer's quality is the one the
CPU sampler already relies on).

### Seeds

The GPU draws its random numbers differently from the CPU, by design: **the same seed gives different shots on the CPU
and on a GPU**, with the same distribution. Within a GPU sampler a seed is reproducible, and the shots do not depend
on the batch size; `sample(n, seed=s)` is a prefix of `sample(m, seed=s)` for `n <= m`. Metal and CUDA are designed to
give identical bytes for the same seed; that has been checked against the reference build for Metal and, on one NVIDIA GPU (a Quadro RTX 6000), for CUDA.
A GPU estimate and a CPU estimate agree within their standard errors, not digit for digit.

### What is refused

A GPU sampler raises `xtim.XtimGpuError` (a subclass of `xtim.XtimError`) for the calls that need the fired lines,
which the GPU does not return yet. `device="cpu"` serves all of them:

- `sample(..., fired=True)`;
- `score(status, lines=...)`;
- `correction(...)`;
- `infidelity(..., lines=True)`.

The same error is raised at construction when no device opens (no Metal device or pyobjc, or no CuPy and CUDA device),
and for an unsupported file (above).
