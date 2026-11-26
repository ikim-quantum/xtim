"""Scaling of compile_detector_sampler, compile + sample(1000) and the DEM export in the circuit size (Stage 2 gates).

Two Clifford families, each size in a fresh subprocess (ru_maxrss is a process-lifetime maximum):
  chain   : N x (R q / X_ERROR(0.01) q / M q / DETECTOR rec[-1]) on N qubits, M = N, N = 1k ... 64k (n' = 3N);
  surface : stim's rotated_memory_z, distance d, d rounds, p = 0.001, d = 11, 15, 21 (M = 1441, 3585, 9681).
For consecutive sizes with measurement ratio m = M2 / M1 (the deferred-wire count n' grows in the same ratio, within a
few per cent on the surface code) the gates assert
  RSS(M2) / RSS(M1) <= m ** log2(RSS_BOUND)   and   time(M2) / time(M1) <= m ** log2(TIME_BOUND),
i.e. at most RSS_BOUND / TIME_BOUND per doubling of n' (the Stage-2 plan's 2.3x: near-linear growth plus slack for
the residual n'/64 word factors). On a chain doubling the bound is exactly 2.3x; surface d = 11 -> 15 (m = 2.49) is
bounded by 2.99x and d = 15 -> 21 (m = 2.70) by 3.30x.

Memory is the peak resident set of the whole worker process (resource.ru_maxrss after the operation): the simplest
robust measure, it needs no allocator hook and counts every allocation the engine makes, C++ and NumPy alike. It
includes about 40-200 MB of interpreter, extension and parsed circuit, growing with the circuit, so on the smallest
sizes it understates the engine's growth; the large sizes (where a quadratic object dominates) are what the gate is
for. RSS is deterministic to ~0.1%, so the memory sweeps start at chain 1k / d = 11.
Time is the worker's CPU time for the operation alone, circuit parsing excluded (time.process_time: user + system, so
page faults count), which other load on the machine does not inflate the way it inflates wall time. Each time cell is
the min of TIME_REPEATS fresh processes, run in rounds over the sizes of a sweep; on a quiet pinned core the five runs
spread by about 1%, so a ratio does not flip from run to run unless it sits within a few per cent of its bound. The
time sweeps start at chain 4k and d = 15 (smaller cells are tens of ms, where cache and memory-bandwidth contention
move CPU time by 10-20%) and are slow-only: run them on a quiet machine, pinned to one core, e.g.
  taskset -c 17 python -m pytest tests/test_compile_scaling.py --runslow
(the workers run with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1, so numpy's BLAS pool does not spin into the CPU time).

Where a ratio still fails on the final Stage-2 code, _PENDING lists the failing steps and who owns the cause (today
no task: the time steps listed there). Pending is strict, per step: a sweep whose failing steps are all listed reports xfail with the measured ratios;
a failing step that is not listed fails (a new regression); a listed step that now meets its bound fails with a
request to delete it from _PENDING. The default suite runs the memory ratios on chain 2k -> 8k and surface
d = 11 -> 15; the full sweeps (memory to chain 64k / d = 21, and every time ratio) are marked slow. The acceptance
table (chain N = 100k, REPEAT N = 100k, surface d = 21 / 25 / 29) is benchmarks/compile_dem_acceptance.py, run under
the heavy lock.
"""
import json
import math
import os
import statistics
import subprocess
import sys

import pytest

TIME_BOUND = 2.3             # per doubling of n', CPU time of the operation
RSS_BOUND = 2.3              # per doubling of n', peak RSS of the worker process
CHAIN_SIZES = (1000, 2000, 4000, 8000, 16000, 32000, 64000)          # memory sweeps
SURFACE_SIZES = (11, 15, 21)
CHAIN_TIME_SIZES = (4000, 8000, 16000, 32000, 64000)                # time sweeps (slow only)
SURFACE_TIME_SIZES = (15, 21)
TIME_REPEATS = 5             # fresh processes per size in a time sweep; the cell's time is their min
OPS = ("compile", "sample", "dem")   # compile_detector_sampler; compile + sample(1000); detector_error_model_text

# (family, op, key) -> (owner, the failing steps (size1, size2)). Final Stage-2 code (f1a010b): every memory step and
# every DEM step on the chain meets the bound (chain DEM time 2.05x .. 2.12x per doubling), and so do the chain compile
# and sample time steps up to 16k (2.1x .. 2.2x). The steps listed below fail reproducibly (min of 5; every sweep and
# slow run so far, the implementer's and the reviewer's, on four cores; spread <= 1%; the per-doubling ranges over
# those runs are in the comments). The growth fits the residual M x n'/64 word work the Stage-2 plan's complexity keeps
# (compile O(L + ... + M n'/64), quadratic, so it overtakes the linear part as M grows); not profiled. No task owns
# it; the owner accepted the miss (2026-11-23) and the bound is not loosened. Delete a step once it passes.
_RESIDUAL = "no task (super-linear residual, likely the plan's M*n'/64 term)"
_PENDING = {
    ("chain", "compile", "work"): (_RESIDUAL, ((16000, 32000), (32000, 64000))),   # 2.39-2.45x, 2.71-2.75x
    ("chain", "sample", "work"): (_RESIDUAL, ((16000, 32000), (32000, 64000))),    # 2.35-2.40x, 2.67-2.72x
    ("surface", "compile", "work"): (_RESIDUAL, ((15, 21),)),                      # 2.58-2.64x
    ("surface", "sample", "work"): (_RESIDUAL, ((15, 21),)),                       # 2.57-2.62x
    ("surface", "dem", "work"): (_RESIDUAL, ((15, 21),)),                          # 2.35-2.38x
}

_WORKER = r"""
import json, os, resource, sys, time
import xtim
family, size, op, want = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
got = os.path.dirname(os.path.realpath(xtim.__file__))
if got != want:
    sys.exit(f"worker imported xtim from {got}, not from {want}")
if family == "chain":
    text = "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(size))
else:
    import stim
    text = str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=size, rounds=size,
               after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
               after_reset_flip_probability=0.001))
c = xtim.Circuit(text)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
t0 = time.process_time()
if op == "compile":
    c.compile_detector_sampler(seed=1)
elif op == "sample":
    c.compile_detector_sampler(seed=1).sample(1000)
else:
    c.detector_error_model_text()
t1 = time.process_time()
print("SCALING " + json.dumps({"M": c.num_measurements, "work": t1 - t0, "base_mb": base,
      "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}))
"""


def _run(family, size, op):
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    # numpy's OpenBLAS starts one worker per core on import, and they busy-wait after a BLAS call: process_time would
    # count that spin (surface d=5 compile + sample: 0.017 s wall, 0.65 s CPU unpinned). One BLAS thread removes it.
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    # cwd=root: `python -c` puts the working directory first on sys.path, so an xtim/ there would shadow this tree's.
    # The worker must import the xtim this test process imported (the source tree's, or the installed package when
    # the suite runs against an install), so it measures the build under test.
    import xtim
    want = os.path.dirname(os.path.realpath(xtim.__file__))
    p = subprocess.run([sys.executable, "-c", _WORKER, family, str(size), op, want], env=env, cwd=root,
                       capture_output=True, text=True)
    assert p.returncode == 0, f"worker {family} {size} {op} failed: {p.stderr.strip()[-2000:]}"
    out = p.stdout
    return json.loads([ln for ln in out.splitlines() if ln.startswith("SCALING ")][-1][len("SCALING "):])


_RUNS = {}                   # (family, size, op) -> the worker runs made so far this session


def _measure(family, size, op):
    """{M, work, base_mb, rss_mb} for one cell from the worker runs made so far (at least one), shared by the memory
    and time gates of a session: work = the min over the runs (the low-noise estimator of CPU time), RSS = the
    median."""
    runs = _RUNS.setdefault((family, size, op), [])
    if not runs:
        runs.append(_run(family, size, op))
    return dict(runs[0], work=min(x["work"] for x in runs),
                **{k: statistics.median(x[k] for x in runs) for k in ("base_mb", "rss_mb")})


def _time_runs(family, sizes, op):
    """TIME_REPEATS fresh processes per size, in rounds over the sizes (one run of every size per round), so a slow
    spell of the machine lands on all sizes of a sweep rather than on one end of it."""
    for _ in range(TIME_REPEATS):
        for s in sizes:
            runs = _RUNS.setdefault((family, s, op), [])
            if len(runs) < TIME_REPEATS:
                runs.append(_run(family, s, op))


def _steps(family, sizes, op, key):
    """[(size1, size2, ok, text)] for consecutive sizes."""
    bound_per_doubling = RSS_BOUND if key == "rss_mb" else TIME_BOUND
    if key == "work":
        _time_runs(family, sizes, op)
    rs = [_measure(family, s, op) for s in sizes]
    out = []
    for (s1, a), (s2, b) in zip(zip(sizes, rs), zip(sizes[1:], rs[1:])):
        m = b["M"] / a["M"]
        ratio = b[key] / a[key]
        out.append((s1, s2, ratio <= m ** math.log2(bound_per_doubling),
                    f"{s1} -> {s2}: {key} {a[key]:.3f} -> {b[key]:.3f} ({ratio:.2f}x = "
                    f"{ratio ** (1.0 / math.log2(m)):.2f}x per doubling, bound {bound_per_doubling}x)"))
    return out


def _gate(family, sizes, op, key):
    steps = _steps(family, sizes, op, key)
    why, listed = _PENDING.get((family, op, key), ("", ()))
    listed = set(listed)
    new_bad = [t for s1, s2, ok, t in steps if not ok and (s1, s2) not in listed]
    fixed = [t for s1, s2, ok, t in steps if ok and (s1, s2) in listed]
    pending_bad = [t for s1, s2, ok, t in steps if not ok and (s1, s2) in listed]
    assert not new_bad, f"{family} {op}: growth above the per-doubling bound: " + "; ".join(new_bad)
    assert not fixed, (f"{family} {op} {key}: pending step(s) now meet the bound, delete them from _PENDING: "
                       + "; ".join(fixed))
    if pending_bad:
        pytest.xfail(f"pending {why}: {family} {op}: " + "; ".join(pending_bad))


def _cases(slow):
    return [pytest.param(op, marks=[pytest.mark.slow] if slow else [], id=op) for op in OPS]


@pytest.mark.parametrize("op", _cases(slow=False))
def test_chain_peak_rss_near_linear(op):
    _gate("chain", CHAIN_SIZES[1:4], op, "rss_mb")


@pytest.mark.parametrize("op", _cases(slow=False))
def test_surface_peak_rss_near_linear(op):
    _gate("surface", SURFACE_SIZES[:2], op, "rss_mb")


@pytest.mark.parametrize("op", _cases(slow=True))
def test_chain_peak_rss_near_linear_to_64k(op):
    _gate("chain", CHAIN_SIZES, op, "rss_mb")


@pytest.mark.parametrize("op", _cases(slow=True))
def test_chain_time_near_linear_to_64k(op):
    _gate("chain", CHAIN_TIME_SIZES, op, "work")


@pytest.mark.parametrize("op", _cases(slow=True))
def test_surface_peak_rss_near_linear_to_d21(op):
    _gate("surface", SURFACE_SIZES, op, "rss_mb")


@pytest.mark.parametrize("op", _cases(slow=True))
def test_surface_time_near_linear_to_d21(op):
    _gate("surface", SURFACE_TIME_SIZES, op, "work")
