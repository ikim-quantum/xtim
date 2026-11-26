# Validating the GPU xdem sampler on an NVIDIA GPU

`tools/gpu_validate.py` checks the GPU sampler on whatever GPU the machine has and writes a JSON report. On a Mac it
validates Metal; this page is the runbook for the CUDA path. It has been run once on real NVIDIA hardware and passed: a Lambda
on-demand cloud instance with a Quadro RTX 6000 (Turing, compute capability 7.5, 24 GB), driver 580.196.07 (CUDA 13.0),
CuPy 14.2.0, Ubuntu 22.04, Python 3.13.18, on 2026-11-25; the report is `docs/assets/gpu/cuda-rtx6000.json`. The report's
commit is 92c4627 with a dirty tree: the one difference was the test fix committed afterwards
(`tests/test_gpu_cuda.py`: `CAPS1` lacked `XK_CAP_SHM`, which NVRTC rejects as fatal; the only failure of the first
CUDA test run), plus untracked log files. Other cards, drivers and CuPy versions are unchecked, and a run on one may
still fail to compile or disagree with the reference; that is what the run finds out. The same text is in the tool's
module docstring.

## What you need

A Linux machine with an NVIDIA GPU and its driver (`nvidia-smi` works), Python 3.11 or newer, git and a C++17
compiler. The kernels need of the device: block shared memory (`XK_CAP_SHM` 32-bit words per block),
`__syncthreads`, `atomicAdd` on 32-bit shared and device words (the walk's lists, the law queue, emit's list) and
`atomicXor` on 32-bit shared words and on 64-bit device words (the rows), so compute capability 3.5 or later (64-bit
`atomicXor` on `unsigned long long`). On Ubuntu:

```sh
sudo apt install -y build-essential git python3-venv python3-dev
```

## Commands

```sh
git clone <the repository URL> xtim && cd xtim && git checkout perf/gpu-xdem-speed    # until the branch is merged; then the default branch
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip
pip install -e ".[dev]"            # builds the C++ extension and its test API (a few minutes)

nvidia-smi                         # read "CUDA Version: X.Y" in the top right
pip install cupy-cuda12x           # CUDA 12.x; for CUDA 11.x: pip install cupy-cuda11x
                                   # CUDA 13.x: pip install cupy-cuda13x (worked: CuPy 14.2.0 on a 13.0 driver)
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"     # must print 1 or more

python -m pytest -m cuda tests -q
python tools/gpu_validate.py --device cuda --out report.json
```

If CuPy reports that it cannot find NVRTC (`libnvrtc.so`) when the first kernel compiles, install it with
`pip install "cupy-cuda12x[ctk]"` (if NVRTC is missing; unverified: we have not run this on a machine without it), or
use a machine image that ships the CUDA toolkit (e.g. an `nvidia/cuda:12.x-devel` image), and rerun the last two
commands.

On the validated run (the Lambda "GPU Base 22.04" image) the CUDA toolkit of the image included NVRTC, so nothing
beyond `cupy-cuda13x` was installed. That image ships Python 3.10 and the tool needs 3.11 or newer, so the environment
was made with `uv venv --python 3.13` in place of `python3 -m venv .venv`.

`--quick` cuts the shot counts (about a minute); the full run takes a few minutes, most of it the CPU benchmark.
Pass `--device cuda` explicitly: without it the tool uses whatever device opens, and if none does it only benchmarks
the CPU and reports `device: none` (which is not a CUDA pass).

## What a pass looks like

- pytest ends with `N passed` and no `failed`, `error` or `skipped`.
- The tool prints `device: cuda (name=..., compute_capability=..., driver=..., runtime=..., cupy=..., kernels=...)`
  (`kernels`: each kernel's `local_size_bytes` and `num_regs` where CuPy exposes them), then every section `PASS`
  (`tests`, `kernels`, `entry_points`, `stress`, `benchmark`), the shots/s table, and `RESULT: PASS`. Its exit code
  is 0.

The sections: `tests` runs `pytest -m cuda tests`; `kernels` compares all eight kernels (`walk`, `resolve`,
`resolve_law`, `reduce`, `reduce_groups`, `reduce2`, `scan_blocks`, `emit`: the list of `xtim/gpu/_stubs.py`; the
section fails if one ran in no case) and every buffer with the kernels' C++ reference build on every bundled example,
with and without a postselection mask, including blocks that overflow their pair capacity, with the law queued
(`resolve_law`) and with `sample(status=False)`'s mode (no C key, no C id), and on generated files: Bravyi-Haah-like
files whose law is the structured law (one word and several words) and multi-word files of the generic law;
`entry_points` compares `sample`, `sample_ps`, `sample(status=False)`, `sample_ps(status=False)` and `estimate` byte for
byte with the reference on the bundled files, a surface-code DEM and the generated files (and the `status=False` rows
with the default call's rows); `stress` shares one sampler between threads (every result equal to its
single-threaded one, `sample(status=False)` among the calls), samples a
file whose pair capacity passes 2^17 (byte-equal to the reference) and times every submission of a DEM firing 4000
lines per shot (at most a few seconds each); `benchmark` measures shots/s of `sample`, `sample(status=False)`,
`sample(postselect=all)`, `infidelity(None)` and `infidelity(None, postselect=all)` on three bundled files and two
generated Bravyi-Haah-like files (`bh_like36`, `bh_like70`; the tests' generator, not the external `merlin_bh*` benchmark files), against the CPU at 1 thread and at every
thread (for the infidelity calls also against `sample` + `score(status, zeros)`, the stronger CPU baseline), with the
device's compile time recorded separately.

Optional test legs on circuits that are not in this repository: the `merlin_bh*` tests of
`tests/test_gpu_structure*.py` read `merlin_bh<k>_dep_p1e-3.stim` from the directory named by `XTIM_MERLIN_CIRCUITS`,
and the xdem tests read their external circuits by file name from `XTIM_EXTERNAL_CIRCUITS`. Both have no default;
the legs skip when the variable is unset.

## If something fails, send back

1. `report.json`.
2. The tool's printed output (a failed comparison names the first differing buffer, its index and both values; a
   compile failure carries NVRTC's log). After a kernel fault the CUDA context is unusable for the rest of the run,
   so only the first failing case is meaningful; the later failures follow from it.
3. `python -m pytest -m cuda tests -q -rA 2>&1 | tail -200`.
4. The output of `nvidia-smi` and of `pip list | grep -i -E "cupy|nvidia|numpy"`.
