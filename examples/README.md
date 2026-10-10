# xtim examples — magic-state-preparation protocols

> **New here? Start with [`xtim_tutorial.ipynb`](xtim_tutorial.ipynb)** — a hands-on
> notebook that walks the whole arc end to end (hello magic state → `diagnose()` →
> scoring → decoder-in-the-loop → noise sweep → multi-magic). Install everything it
> needs with `pip install "xtim[tutorial]"` (both optional cells degrade gracefully
> without it).

> **Wheel users:** the prebuilt wheel bundles the demo *circuits* — load any with
> `xtim.load_example("cube_ccz")` (`xtim.list_examples()` lists them), no clone needed.
> The example **scripts** (`*.py`), the **notebook**, and `docs/` live in the repo and
> the sdist tarball, not the wheel — clone the repo (or `pip download --no-binary :all:
> xtim` for the sdist) to run them.

Runnable circuits for the two protocol families xtim is built to study. Each comes
in two variants:

- **`faithful`** — the true protocol (keeps the real non-Clifford content).
- **`rate`** — a count-matched twin. For *cultivation* the rate twin is fully
  **Clifford** (magic removed), so it exports a plain DEM. For *code switching* the
  rate twin keeps the transversal `T̄` layer, so it is **still non-Clifford** (it
  Cliffordizes only the verification structure, not the logical magic). The *faithful*
  cultivation/oracle protocols now also export — a decoder DEM + reject region via
  `detector_error_model_with_reject()` (see the table below); only `code_switching`
  genuinely refuses.

`ref χ` is the reference/output stabilizer rank (what `reference_info()['chi']`
reports). The magic-state circuits now export a **decoder DEM + reject region** via
`detector_error_model_with_reject()` (see [`../docs/xtim_dem_reject.md`](../docs/xtim_dem_reject.md)
and [`dem_reject_region.py`](dem_reject_region.py)); `code_switching` is the exception —
its logical observable isn't deterministic noiselessly (`XtimDemError: non-deterministic
observable L0 ... not ±1`), so it has no exact Pauli DEM and genuinely refuses
(post-selection only).

| Circuit | Protocol | n | ref χ | DEM | expectations |
|---|---|---|---|---|---|
| `code_switching_faithful.stim` | qRM ↔ Steane code switching (arXiv:2410.07327) | 44 | 2 | no — post-select only (non-deterministic observable) | `X̄`, `Ȳ` |
| `code_switching_rate.stim`     | …count-matched twin (keeps `T̄`; non-Clifford)  | 44 | 2 | no — post-select only | 3 cols |
| `cultivation_d3_faithful.stim` | d=3 magic-state cultivation (Gidney-style)     | 21 | 2 | **yes** — decodable DEM (319 edges) + 177 flagged not-correctable faults (you post-select or budget) | `X̄`, `Ȳ` |
| `cultivation_d3_rate.stim`     | …**Clifford** twin (magic removed)             | 21 | 1 | **yes** — plain DEM | `X̄`, `Ȳ` |
| `miniature_oracle.stim`        | teleported `T\|+⟩` in a small rep code          | 5  | 2 | detector DEM only (empty reject); the magic columns are **byproduct-frame** (not DEM L-columns) → post-select + frame hint | `X̄`, `Ȳ` |
| `cube_ccz.stim`                | [[8,3,2]] cube code **transversal CCZ** — 3 logical qubits, **multi-magic** (output `CCZ\|+++⟩_L`) | 12 (8 data + 4 MPP ancillas) | 8 | none — deterministic feedback prep, 100% acceptance, no detectors | `X̄₁`,`X̄₂`,`X̄₃` (all `+0.5`) |
| `distillation_15_1_3.stim`     | [[15,1,3]] punctured-Reed–Muller transversal-`T`, **15-to-1 magic distillation** (Z_ERROR T-gate noise, p0=1e-3) | 29 (15 data + 14 MPP ancillas) | 2 | **4 X-stabilizer detectors** — post-select a trivial syndrome to distill (1−F → ~35 p³) | `X̄`, `Ȳ` (both `+0.70711`) |
| `ch_cultivation.stim`         | **H-eigenstate cultivation** — the raw T-form magic Clifford-aligned onto the Hadamard axis, then 3 logical-H checks via **controlled-Hadamard** (Hadamard test, ancilla `MX` + `DETECTOR` each) | 4 | 2 | no — CH faults act as Cliffords, not Paulis (post-select the 3 deterministic detectors; `diagnose()` prints the recipe) | `X̄`, `Z̄` (both `+0.70711`) |
| `cultivation_d5.stim`          | d=5 magic-state cultivation (**T-count / syndrome fixture**) | 61 | 2 | none — `detector_error_model()` refuses (a 19-qubit `Y` observable exceeds the DEM's weight-16 twirl-coin-read limit, `19 > 16`); **sample / post-select only** | none — **no** `PAULI_EXPECTATION`; syndrome sampling only |

## Throughput

Representative amortized rates for four of the bundled protocols, spanning the
regimes xtim targets — from a single low-rank magic output to a multi-magic
distillation:

| `load_example(...)` | protocol | amortized throughput | per-shot payload |
|---|---|---|---|
| `"distillation_15_1_3"` | [[15,1,3]] punctured-Reed–Muller transversal-`T`, 15-to-1 magic distillation | **≈ 5 M shots/s** | exact ⟨X̄⟩, ⟨Ȳ⟩ of the distilled T state |
| `"cultivation_d3_faithful"` | d=3 magic-state cultivation (χ = 2) | **≈ 1 M shots/s** | exact ⟨X̄⟩, ⟨Ȳ⟩ of the cultivated T state |
| `"cultivation_d5"` | d=5 magic-state cultivation (T-count / syndrome fixture) | **≈ 50 K shots/s** | syndrome sampling only — **no** `PAULI_EXPECTATION` |
| `"ch_cultivation"` | H-eigenstate cultivation by **measuring the logical Hadamard** (controlled-`H` checks, χ = 2 at any round count) | **≈ 10 M shots/s** | exact ⟨X̄⟩, ⟨Z̄⟩ of the cultivated H state ([walkthrough](h_cultivation_walkthrough.py)) |

> Throughput is the **amortized** per-shot rate over a warm compile cache: the first
> `sample()` on a sampler compiles the reference once (a few ms for the two small
> circuits, ~0.6 s for d5) and **every later `sample()` on the same object reuses
> it** — no recompile. The figures are **hardware-dependent** (best-of-5 on one
> modern x86 core; expect ±30 % across machines) — treat them as orders of magnitude,
> not benchmarks. `cultivation_d5` declares no `PAULI_EXPECTATION`, so its number is
> pure syndrome sampling (the d3 and 15-qubit circuits also compute the exact magic
> expectation each shot).
>
> `distillation_15_1_3` is a genuine distillation: physical T-gate phase noise
> (`Z_ERROR`, baked p0=1e-3) plus the 4 X-stabilizers measured as detectors.
> **Post-select** on a trivial syndrome (`keep=lambda dets, meas: ~dets.any(axis=1)`)
> and the distilled infidelity drops to the weight-3 floor **~35 p³** — far below the
> physical rate (e.g. 1−F ≈ 3.5e-5 at p=1e-2, vs ~0.13 un-post-selected), at a falling
> acceptance rate. The throughput above samples that noise per shot; the noiseless
> reference still compiles to χ=2.

## Onboarding: scoring your OWN protocol

`onboarding_new_protocol.py` is the walkthrough for *bringing a new protocol to
xtim*. It treats `miniature_oracle.stim` as a circuit you just wrote and shows
how `Circuit.diagnose()` hands you everything you'd otherwise reverse-engineer —
which detectors to post-select, the |β| magnitudes, the byproduct-frame sign
rule — so you can score it with no sidecar and no internal lore:

```bash
python examples/onboarding_new_protocol.py
```

That example is the post-selection-only case (its expectations are not
DEM-expressible, which `diagnose()` reports). The full decoder-in-the-loop
workflow — pymatching + the DEM — is the tour below.

`h_cultivation_walkthrough.py` is the **controlled-Hadamard** walkthrough:
cultivate the Hadamard-eigenstate magic state by *measuring* the logical Hadamard
(Hadamard-test rounds via `CH`), post-select the deterministic checks, and score
the state — then watch the *unaligned* T-state get rejected by the simulable-class
condition, which is the protocol's alignment requirement in disguise
([`../docs/xtim_simulable_class.md`](../docs/xtim_simulable_class.md)):

```bash
python examples/h_cultivation_walkthrough.py
```

`dem_reject_region.py` is the **decoder DEM + reject region** walkthrough on
`cultivation_d3_faithful.stim`: it exports `detector_error_model_with_reject()`, post-selects on
`reject_detectors`, and reads the magic value on the accepted shots — the sound, complete
magic-state-prep DEM path. See [`../docs/xtim_dem_reject.md`](../docs/xtim_dem_reject.md).

```bash
python examples/dem_reject_region.py
```

## Fidelity figure-of-merit: a 1−F vs p curve

`fidelity_curve.py` is the **fidelity figure-of-merit** walkthrough (xtim ≥ 0.5.9).
Set `target_k` on each `Task`, sweep `p`, and `xtim.collect` returns `fidelity`,
`infidelity` (= 1−F), and a correlation-aware `infidelity_sem` per row — the
publishable `1−F ± sem` vs `p` curve, no manual Bloch arithmetic. The script
sweeps a noisy logical `|T⟩` and (with matplotlib) writes `fidelity_curve.png`:

```bash
python examples/fidelity_curve.py
```

On a post-selected protocol the only addition is a `keep` mask (the tour shows
it); the fidelity part is identical. See the README's *“The shortcut: a fidelity
curve from `collect`”* for the one-liner.

> **Which examples work with `collect(target_k=...)`?** `cultivation_d3_faithful`,
> `distillation_15_1_3`, and the standalone `|T⟩` in `fidelity_curve.py` — each carries
> noise and declares the full single-qubit Pauli support (⟨X̄⟩ and ⟨Ȳ⟩), so
> `Task(target_k=1)` gives a true 1−F vs p curve. `distillation_15_1_3` needs the
> post-select mask `keep=lambda dets, meas: ~dets.any(axis=1)` to show distillation gain
> (1−F → ~35 p³); without it you get the un-corrected ~n·p ≈ 15 p baseline. Only
> `cultivation_d5` does **not**: it declares no `PAULI_EXPECTATION` at all (so `target_k`
> raises a clear error) — use it for syndrome sampling / throughput, not fidelity.

## Post-selected LER in one call (`postselected_logical_error_rate`)

`postselected_logical_error_rate.py` shows the one-call path to a **post-selected
logical error rate**: `Circuit.postselected_logical_error_rate(p=…, shots=…)`
sums the declared `PAULI_EXPECTATION` columns (fidelity mode) or the
`OBSERVABLE_INCLUDE` (observable mode) and applies the post-selection everyone
forgets — it rejects every shot in which a deterministic detector fired. The
returned `PostselectedLER` carries `value` (`1 − F`), `sem`, `acceptance`,
`kept`, `upper_bound` (rule-of-three when zero errors are seen), and `warnings`.
See [`../docs/xtim_postselected_logical_error_rate.md`](../docs/xtim_postselected_logical_error_rate.md).

```bash
python examples/postselected_logical_error_rate.py
```

## Logical-fidelity helpers (`extract_frame`, `fidelity_from_logicals`)

`logical_fidelity_helpers.py` demonstrates two **opt-in** helpers that remove the
tedious parts of scoring a magic state's fidelity: `extract_frame` computes the
byproduct-frame `rec[-k]` offsets that pin a logical operator's sign (ready to
paste into a `PAULI_EXPECTATION`), and `fidelity_from_logicals` takes the k
logical generators and scores the **true full-support** `1 − F` (all `4^k`
products, auto post-selected) — so you cannot silently miss a Pauli with nonzero
ideal expectation. The default path stays `postselected_logical_error_rate`.
See [`../docs/xtim_logical_fidelity_helpers.md`](../docs/xtim_logical_fidelity_helpers.md).

```bash
python examples/logical_fidelity_helpers.py
```

## [[8,3,2]] transversal CCZ, scored to a true 3-qubit fidelity

`cube_832_ccz.py` is the worked, end-to-end version of the bundled `cube_ccz.stim`
demo. It prepares logical `|+++⟩` on the [[8,3,2]] cube code by **measuring the
stabilizers + classically-controlled Pauli feedback**, applies the **logical CCZ**
(transversal `T`/`T_dag`, with noise on *only* the physical T gates), then runs a
**noiseless syndrome round and post-selects** the +1 outcomes. Crucially it
declares the **complete logical Pauli support** (the full 63-element group; 28 are
the nonzero `CCZ|+++⟩` terms), so `target_k=3` gives the *true* state fidelity —
`F = 1` noiselessly, where `cube_ccz.stim`'s three-marginal readout maxes out at
0.219 (now flagged by xtim's completeness warning).

```bash
python examples/cube_832_ccz.py
```

The payoff curve: the code has distance 2, so syndrome post-selection heralds away
every single fault and suppresses the infidelity from `O(p)` to `O(p²)` (at a
falling acceptance rate).

## Run one

```python
import xtim

c = xtim.load_example("cultivation_d3_rate")   # bundled; or from_file("your.stim")
dets, obs, exps = c.compile_detector_sampler(seed=0).sample(
    100_000, separate_observables=True, return_expectations=True)
print(exps.mean(axis=0))          # raw per-shot ⟨X̄⟩, ⟨Ȳ⟩ (pre-decoding)
```

or from the shell:

```bash
python -m xtim detect --in examples/cultivation_d3_rate.stim --shots 100000 --seed 0 \
    --out dets.b8 --obs_out obs.b8 --exp_out exps.txt
```

## Reading the output

The `exps` columns are **raw** per-shot expectation values — the engine emits raw
physics only. The *meaningful* magic-state fidelity comes after the user-side
decode + post-selection workflow (feed the DEM to a decoder, discard rejected
shots, combine the corrected expectations). That full workflow — the same eight
lines for any protocol — is worked through end-to-end in
[`../docs/xtim_tour.md`](../docs/xtim_tour.md), using `cultivation_d3_rate` as its
running example (it lands at `F ≈ 0.96`). The circuit dialect itself is specified
in [`../docs/xtim_dialect.md`](../docs/xtim_dialect.md).

> The cultivation circuits use **unitary** injection (a physical `T` on an
> already-encoded qubit). The `faithful` cultivation's *reference* state is **χ=2**
> and it samples fast (≈ 1 M shots/s amortized — see the README *Highlights*); any
> higher rank seen during the one-time reference compile is a transient compile cost,
> not a per-shot cost.
