# xtim

[![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://github.com/ikim-quantum/xtim/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ikim-quantum/xtim/blob/main/LICENSE)

A fast, exact sampler for noisy quantum circuits with `T`, `CS`, `CCZ` and `CH` gates. xtim reads
[Stim](https://github.com/quantumlib/Stim) circuits plus these non-Clifford gates and samples detectors and
observables the way Stim does. It is built for magic-state preparation under circuit-level noise: distillation,
cultivation, code switching and state factories.

## Install

```bash
pip install xtim              # wheels for Linux x86_64 and macOS arm64 (11.0+), CPython 3.11-3.13
pip install "xtim[decode]"    # adds stim, pymatching, ldpc
```

From source (a C++17 compiler; the build takes a few minutes; other platforms build the sdist this way too):

```bash
git clone https://github.com/ikim-quantum/xtim.git
cd xtim
pip install -e ".[decode]"    # editable install
```

Platforms: Linux (primary); macOS supported for correctness; Windows not tested.

Optional GPU sampling of xdem files: `pip install "xtim[metal]"` on Apple silicon, or a CuPy wheel on NVIDIA. Metal
tested (Apple M3 Max); CUDA experimental, validated on one NVIDIA card. See
[GPU sampling](https://github.com/ikim-quantum/xtim/blob/main/docs/xdem.md#11-gpu-sampling).

## Quick start

Prepare a T state, add noise, then undo the T gate noiselessly and measure in the X basis. Without noise the
observable is always 0, so every flip is a logical error:

```python
import numpy as np
import xtim

circuit = """
RX 0
T 0
DEPOLARIZE1(0.01) 0
T_DAG 0
MX 0
OBSERVABLE_INCLUDE(0) rec[-1]
"""
sampler = xtim.compile_twirl_sampler(circuit)
dets, obs = sampler.sample(1_000_000, seed=1)     # Stim-style bit-packed uint8 arrays
flips = np.unpackbits(obs, axis=1, bitorder="little")[:, 0]
print(f"logical error rate {flips.mean():.2e}")
# logical error rate 6.78e-03
```

## Documentation

[docs/usage.md](https://github.com/ikim-quantum/xtim/blob/main/docs/usage.md) covers post-selection, threads, exact output-state fidelity, output-state helpers,
decoding, [xdem files with decoder feedback](https://github.com/ikim-quantum/xtim/blob/main/docs/xdem.md), the circuit language and
[which circuits xtim can simulate](https://github.com/ikim-quantum/xtim/blob/main/docs/xtim_simulable_class.md). See also [examples/](https://github.com/ikim-quantum/xtim/tree/main/examples) and
[CHANGELOG.md](https://github.com/ikim-quantum/xtim/blob/main/CHANGELOG.md).

## How fast?

See [docs/performance.md](https://github.com/ikim-quantum/xtim/blob/main/docs/performance.md): comparisons with tsim, Clifft, SymFT, Stim and Magna on the
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench) workloads, on one CPU core.
