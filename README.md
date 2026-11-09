# xtim

[![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](../../releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

A fast, exact sampler for noisy quantum circuits with `T`, `CS`, `CCZ` and `CH` gates. xtim reads
[Stim](https://github.com/quantumlib/Stim) circuits plus these non-Clifford gates and samples detectors and
observables the way Stim does. It is built for magic-state preparation under circuit-level noise: distillation,
cultivation, code switching and state factories.

## Install

From source (CPython 3.11-3.13 and a C++17 compiler; the build takes a few minutes):

```bash
git clone https://github.com/ikim-quantum/xtim.git
cd xtim
pip install -e ".[decode]"    # editable install; [decode] adds stim, pymatching, ldpc
```

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

[docs/usage.md](docs/usage.md) covers post-selection, threads, exact output-state fidelity, decoding, the circuit
language and [which circuits xtim can simulate](docs/xtim_simulable_class.md). See also [examples/](examples/) and
[CHANGELOG.md](CHANGELOG.md).

## How fast?

See [docs/performance.md](docs/performance.md): comparisons with tsim, Clifft, SymFT, Stim and Magna on the
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench) workloads, on one CPU core.
