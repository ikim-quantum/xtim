# xtim

[![arXiv](https://img.shields.io/badge/arXiv-2512.23799-b31b1b.svg)](https://arxiv.org/abs/2512.23799)
[![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](../../releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

xtim simulates **magic-state preparation** circuits (distillation, cultivation, code switching) under circuit-level
noise. It reads [Stim](https://github.com/quantumlib/Stim) circuits plus the non-Clifford gates `T`, `CS`, `CCZ`,
`CH`, samples detectors and observables like Stim, and adds what Stim cannot give: the **exact fidelity of each
shot's output state**. It implements Surti, Daguerre & Kim, *Efficient simulation of logical magic state
preparation protocols* ([arXiv:2512.23799](https://arxiv.org/abs/2512.23799)).

## Install

From source (CPython 3.11-3.13; needs a C++17 compiler, e.g. g++ >= 9 or clang; the build takes a few minutes):

```bash
git clone https://github.com/ikim-quantum/xtim.git
cd xtim
python -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -e ".[decode]"         # builds the extension into the clone; [decode] adds stim, pymatching, ldpc
```

Run the examples and tests from the clone: `python examples/01_quickstart_distillation.py`,
`pip install pytest && pytest tests/`. Install with `-e` (editable) as above: a plain `pip install .` puts the
compiled extension only in site-packages, so Python started inside the clone (including the test suite's
subprocesses) imports the uncompiled source tree and fails with `No module named 'xtim._xtim'`.

## Quickstart

15-to-1 distillation at p = 10⁻²: sample a million shots, keep the shots where no detector fired, and score each
kept shot's output exactly.

```python
import numpy as np
import xtim

circuit = xtim.load_example("distillation_15_1_3")           # bundled; declares OUTPUT_QUBITS
text = xtim.scale_noise(circuit.text, 0.01)                   # physical error p = 1e-2
sampler = xtim.compile_twirl_sampler(text)
buf = sampler.sample_barrier(1_000_000, seed=7)               # keeps each shot's output state

dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
keep = np.flatnonzero(~dets.any(axis=1))                      # post-select: no detector fired

F = xtim.RecordFidelity(sampler).evaluate(buf, shot_index=keep)   # exact fidelity, one float per shot
print(f"accepted {len(keep) / len(dets):.1%}, distilled 1-F = {1 - F.mean():.1e}")
# accepted 86.0%, distilled 1-F = 2.7e-05
```

The first sample call runs a one-time self-check and says so with an `xtim.XtimVerificationWarning`
(`compile_twirl_sampler(text, selfcheck=0)` skips it). `sampler.sample(shots, seed)` returns Stim-style bit-packed
detectors and observables when you do not need fidelities.

To post-select inside the sampler, pass `postselect=` (a `dtype=bool` mask over the detectors, or distinct detector
indices; a non-bool array is read as indices, so duplicate or out-of-range indices are refused and an integer 0/1 mask
must be cast to bool) to `sample` or `sample_barrier`: only the shots whose flagged detectors are all 0 come back, byte-identical to filtering
an unmasked call, and shots that a flagged detector rejects for certain are dropped before any plan work.
`sample(..., postselect=m)` appends the counts `{attempts, passed, discarded, early}`; `sample_barrier` puts them in
`buf.postselection`:

```python
import numpy as np
import xtim

sampler = xtim.compile_twirl_sampler(xtim.load_example("distillation_15_1_3").text, selfcheck=0)
buf = sampler.sample_barrier(100_000, seed=7, postselect=np.ones(sampler.num_detectors, bool))
F = xtim.RecordFidelity(sampler).evaluate(buf)                # the accepted shots only
print(buf.postselection["passed"] == len(F))
# True
```

## Examples

Each script runs standalone (`python examples/<script>`); its docstring explains the protocol and the API it uses.

| script | shows |
|---|---|
| [`01_quickstart_distillation.py`](examples/01_quickstart_distillation.py) | 15-to-1 distillation: post-selected 1-F vs p |
| [`02_cultivation_d3.py`](examples/02_cultivation_d3.py) | d=3 cultivation: acceptance, F of accepted vs all shots |
| [`03_color5_postselected.py`](examples/03_color5_postselected.py) | d=5 colour-code cultivation (SOFT, Apache-2.0): post-selected F |
| [`04_h_cultivation.py`](examples/04_h_cultivation.py) | H-state cultivation with controlled-Hadamard checks |
| [`05_ccz_cube.py`](examples/05_ccz_cube.py) | [[8,3,2]] transversal CCZ, a 3-qubit magic output |
| [`06_code_switching.py`](examples/06_code_switching.py) | qRM to Steane code switching, frame closed by feedback |
| [`07_refusals.py`](examples/07_refusals.py) | every named refusal and its fix |
| [`08_bring_your_own.py`](examples/08_bring_your_own.py) | your own circuit: declare the output, close the frame, score |
| [`09_decoded_fidelity.py`](examples/09_decoded_fidelity.py) | a decoder's correction applied before scoring (needs `[decode]`) |

The bundled circuits are listed by `xtim.list_examples()` and loaded by `xtim.load_example(name)`. They carry noise
at p = 10⁻³ (except the noiseless `cube_ccz`); `xtim.scale_noise(text, p)` rescales it and
`compile_twirl_sampler(text, p_factor=0.0)` turns it off.

## Output contract

`RecordFidelity` scores the qubits named by `OUTPUT_QUBITS name q0 q1 ...`: per shot,
`F = <φ|ρ_out|φ>`, where `ρ_out` is that shot's state of those qubits and `|φ>` is the noiseless output. Two rules:

1. **The output is pure and fixed.** Without noise it must be one pure state, the same for every measurement
   outcome. If an outcome leaves a Pauli byproduct on the output (teleportation, code switching), close the frame
   with Pauli feedback `CX` / `CY` / `CZ rec[-k] q`, as you would to make a Stim detector deterministic.
2. **Output qubits are never measured.** Move a state you want to keep onto a fresh wire (noiseless `R`, then
   `SWAP`) and declare that wire.

If rule 1 fails, `RecordFidelity` raises `xtim.XtimFidelityRefusal` with code `RF_OUTPUT_NOT_STRICT`. The message
names the undetermined output directions on your qubits (e.g. `X2, Z2`) and the measurements whose outcomes fix
them; add the feedback from those records. A measured output qubit is refused as `RF_OUTPUT_MEASURED`. How to pick
each feedback Pauli and check the fix: [docs/usage.md](docs/usage.md#closing-a-frame); a worked fix:
[`08_bring_your_own.py`](examples/08_bring_your_own.py). Every bundled circuit already follows
the contract.

## Documentation

- [docs/usage.md](docs/usage.md): the API in one page (sampling, post-selection, fidelity, closing frames, decoding,
  refusals, circuit language).
- [docs/xtim_simulable_class.md](docs/xtim_simulable_class.md): which circuits xtim can simulate.
- [CHANGELOG.md](CHANGELOG.md), [LICENSE](LICENSE) (Apache-2.0), [CITATION.cff](CITATION.cff).

Bug reports with a minimal reproducing circuit are welcome as GitHub issues. Development: `pip install -e ".[dev]"`
then `pytest tests/`.
