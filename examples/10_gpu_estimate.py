"""Estimate an infidelity on the GPU: the zero decoder, post-selected, counted on the device.

`XdemSampler(text, device="gpu")` samples an xdem file on the machine's GPU (the first that opens of Metal, on an
Apple silicon Mac, and CUDA, where CuPy finds an NVIDIA device); `infidelity(shots, None, postselect=...)` is the zero decoder: every shot is sampled, post-selected and
scored on the device, and only counters come back. The script falls back to `device="cpu"` when no GPU opens, so it
runs anywhere (the CPU gives the same law with different random bits, so the numbers agree within their standard
errors, not digit for digit).

Install a GPU backend with `pip install "xtim[metal]"` (Apple silicon) or CuPy (NVIDIA); see docs/xdem.md, "GPU
sampling".  Run:  python examples/10_gpu_estimate.py
"""
import pathlib
import time

import xtim
import xtim.gpu

XDEM = pathlib.Path(__file__).resolve().parent / "xdem" / "cultivation_d3_faithful.xdem"

if __name__ == "__main__":
    text = XDEM.read_text()
    ok, reason = xtim.gpu.supports(text)                 # needs no GPU
    print(f"{XDEM.name}: GPU sampler supported = {ok}" + ("" if ok else f" ({reason})"))

    device = "gpu" if xtim.gpu.available() else "cpu"
    sampler = xtim.XdemSampler(text, device=device)
    print(f"device: {sampler.device}")

    shots = 1_000_000
    keep = list(range(10))                               # post-select: the first ten detectors must not fire
    t0 = time.perf_counter()
    r = sampler.infidelity(shots, None, seed=1, postselect=keep)
    dt = time.perf_counter() - t0
    print(f"{r.accepted} of {shots} shots accepted ({r.accepted / shots:.3f}) in {dt:.2f} s")
    print(f"infidelity = {r.mean:.5f} +- {r.stderr:.5f}")
