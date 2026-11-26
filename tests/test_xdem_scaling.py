"""Scaling of the public xdem DEM export (`detector_error_model_text(format="xdem")`) on large magic circuits (LM-8).

Same method as test_compile_scaling.py: each size in a fresh subprocess, time = process_time of the export alone (circuit
parsing excluded), memory = ru_maxrss of the worker process, min (time) / median (RSS) of 5 fresh runs per cell, each worker pinned to one core. The bound is the Stage-2
plan's 2.3x per doubling (TIME_BOUND, RSS_BOUND), expressed on the size parameter; for the surface family (measurements
grow as d^3) on the measurement ratio, as in test_compile_scaling.py. All sweeps are slow-marked (run with --runslow,
under the heavy lock): the default suite is untouched.

Families (every one has a magic gate, so the lean route or its certified no-rotation fallback serves it):
  surfT / surfTn : rotated_memory_z surface code d x d rounds + one T pair, noiseless T (no rotating fault: the certified
                   no-rotation path) / with an X_ERROR(0.001) before the T;
  chainT / chainTn : N x (R, X_ERROR, M, DETECTOR) + one T pair, noiseless / noisy;
  ghzTzn / ghzT  : N-qubit GHZ, T on wire 0 with a noisy rotating fault (read with M) / H T H;
  repTn          : repetition code distance d, 2 rounds, + two noisy T pairs;
  manyT          : one T per wire (N wires, each with a rotating fault): expected pending (LM-4, dropped).
The stim-format Clifford surface code is measured by the same worker (mode "surf", fmt "stim") as the reference.

_PENDING lists the steps that miss the bound at the measured commit, with the measured ratio; pending is strict per step
as in test_compile_scaling.py (an unlisted failing step fails; a listed memory step that now passes fails, a listed time
step that passes is tolerated: time ratios sit near the bound and flip).
"""
import json
import math
import os
import statistics
import subprocess
import sys

import pytest

TIME_BOUND = 2.3             # per doubling of the size, CPU time of the export
RSS_BOUND = 2.3              # per doubling of the size, peak RSS of the worker process
CEILING = 1.25                # a pending step fails above (its measured ratio per doubling) x CEILING
REPEAT = 5                   # fresh processes per cell: work = their min (low-noise CPU time), RSS = median

_WORKER = r"""
import json, os, resource, sys, time
import xtim
try:   # the core the parent picked: every cell of a session runs on the same core (no migration noise)
    os.sched_setaffinity(0, {int(sys.argv[5])})
except (AttributeError, OSError, IndexError):
    pass
mode, size, fmt, want = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
got = os.path.dirname(os.path.realpath(xtim.__file__))
if got != want:
    sys.exit(f"worker imported xtim from {got}, not from {want}")
noisy = mode.endswith("n")
base = mode[:-1] if noisy else mode


def tpair(a, b, z=False):
    e = f"X_ERROR(0.001) {a}\n" if noisy else ""
    return f"R {a} {b}\nH {a}\n{e}T {a}\nCX {a} {b}\n", (f"M {a}\nM {b}\n" if z else f"MX {a}\nM {b}\n")


def gen(name, **kw):
    import stim
    return stim.Circuit.generated(name, after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                  after_reset_flip_probability=0.001, **kw)


chain = lambda n: "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(n))
if mode == "surf":
    text = str(gen("surface_code:rotated_memory_z", distance=size, rounds=size))
elif mode == "manyT":
    text = "".join(f"R {q}\nX_ERROR(0.01) {q}\nT {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(size))
elif base == "surfT":
    c0 = gen("surface_code:rotated_memory_z", distance=size, rounds=size)
    pre, post = tpair(c0.num_qubits, c0.num_qubits + 1)
    text = pre + str(c0) + "\n" + post
elif base == "chainT":
    pre, post = tpair(size, size + 1)
    text = pre + chain(size) + post
elif base in ("ghzTz", "ghzT"):
    n = size
    L = ["R " + " ".join(map(str, range(n))), "H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)]
    e = "X_ERROR(0.001) 0\n" if noisy else ""
    L.append(e + ("T 0" if base == "ghzTz" else "H 0\nT 0\nH 0"))
    L.append("X_ERROR(0.01) " + " ".join(map(str, range(n))))
    L.append("M " + " ".join(map(str, range(n))))
    L += [f"DETECTOR rec[-{n - i}] rec[-{n - i - 1}]" for i in range(1, n - 1)]
    text = "\n".join(L) + "\n"
elif base == "repT":
    c0 = gen("repetition_code:memory", distance=size, rounds=2)
    nq = c0.num_qubits
    p1, q1 = tpair(nq, nq + 1)
    p2, q2 = tpair(nq + 2, nq + 3)
    text = p1 + p2 + str(c0) + "\n" + q1 + q2
else:
    sys.exit("bad mode")
c = xtim.Circuit(text)
m = c.num_measurements
b0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
t0 = time.process_time()
c.detector_error_model_text(**({"format": "xdem"} if fmt == "xdem" else {}))
t1 = time.process_time()
print("SCALING " + json.dumps({"M": m, "work": t1 - t0, "base_mb": b0,
      "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}))
"""

# family -> (mode, fmt, sizes); the bound is per doubling of the size, except "surf" (measurement ratio).
_FAMILIES = {
    "surface+T": ("surfT", "xdem", (15, 21, 25)),
    "surface+noisyT": ("surfTn", "xdem", (15, 21, 25)),
    "chain+T": ("chainT", "xdem", (16000, 32000, 64000)),
    "chain+noisyT": ("chainTn", "xdem", (16000, 32000, 64000)),
    "ghz+noisyT": ("ghzTzn", "xdem", (16000, 32000, 64000)),
    "ghz H T H": ("ghzT", "xdem", (16000, 32000, 64000)),
    "rep+2noisyT": ("repTn", "xdem", (2100, 4200, 8400)),
    "oneT-per-wire": ("manyT", "xdem", (2000, 4000, 8000)),
    "surface-stim": ("surf", "stim", (15, 21, 25)),
}
_SURFACE = ("surface+T", "surface+noisyT", "surface-stim")

# (family, key) -> (why, {failing step (size1, size2): measured ratio per doubling}); CEILING x ratio is a hard ceiling. Filled from the LM-8 table
# (docs/performance.md, "magic DEM export"); the one T per wire family is LM-4 territory (dropped by the owner).
_PENDING = {
    # surface + T (certified no-rotation path = the stim-format export): time 2.31x then 2.35x per doubling of M, the same
    # as the Clifford stim-format reference (2.36x, 2.43x): the d15 -> 21 step sits near the bound and flips, d21 -> 25 is over.
    ("surface+T", "work"): ("the stim-format export's own growth (same as the Clifford reference)",
                            {(15, 21): 2.31, (21, 25): 2.35}),
    ("surface-stim", "work"): ("the stim-format export's growth (reference, not the lean route)",
                               {(15, 21): 2.36, (21, 25): 2.43}),
    # GHZ with H T H is read by the no-rotation path, not the sparse sweep: 2.82x, 3.2x per doubling at 16k -> 64k.
    ("ghz H T H", "work"): ("no-rotation path on GHZ H T H", {(16000, 32000): 2.82, (32000, 64000): 3.2}),
    # one T per wire: A ~ N rotation axes, the writer model is super-linear (LM-3 profile: build_writer_model); LM-4 dropped.
    ("oneT-per-wire", "work"): ("LM-4 dropped (writer model)", {(2000, 4000): 3.4, (4000, 8000): 3.95}),
    ("oneT-per-wire", "rss_mb"): ("LM-4 dropped (writer model)", {(4000, 8000): 2.93}),
}

_RUNS = {}


_CORE = []


def _core():
    """The least busy allowed core (idle share over 0.5 s of /proc/stat), chosen once per session."""
    if not _CORE:
        try:
            import time

            def snap():
                d = {}
                for ln in open("/proc/stat"):
                    f = ln.split()
                    if f[0].startswith("cpu") and f[0] != "cpu":
                        v = list(map(int, f[1:9]))
                        d[int(f[0][3:])] = (v[3] + v[4], sum(v))
                return d
            a = snap()
            time.sleep(0.5)
            b = snap()
            allowed = sorted(os.sched_getaffinity(0))
            idle = {c: (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in allowed if c in a}
            _CORE.append(max(idle, key=lambda c: (idle[c], c)))
        except Exception:
            _CORE.append(-1)
    return _CORE[0]


def _run(mode, size, fmt):
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    want = os.path.join(os.path.realpath(root), "xtim")
    p = subprocess.run([sys.executable, "-c", _WORKER, mode, str(size), fmt, want, str(_core())], env=env, cwd=root,
                       capture_output=True, text=True)
    assert p.returncode == 0, f"worker {mode} {size} {fmt} failed: {p.stderr.strip()[-2000:]}"
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("SCALING ")][-1]
    return json.loads(line[len("SCALING "):])


def _measure(mode, size, fmt):
    """REPEAT fresh processes: work = min, base_mb and rss_mb = median; M from the first."""
    runs = _RUNS.setdefault((mode, size, fmt), [])
    while len(runs) < REPEAT:
        runs.append(_run(mode, size, fmt))
    return dict(M=runs[0]["M"], work=min(r["work"] for r in runs),
                **{k: statistics.median(r[k] for r in runs) for k in ("base_mb", "rss_mb")})


def _steps(family, key):
    """[(size1, size2, ok, per_doubling, text)] for consecutive sizes."""
    mode, fmt, sizes = _FAMILIES[family]
    rs = [_measure(mode, s, fmt) for s in sizes]
    bound = RSS_BOUND if key == "rss_mb" else TIME_BOUND
    out = []
    for (s1, a), (s2, b) in zip(zip(sizes, rs), zip(sizes[1:], rs[1:])):
        m = b["M"] / a["M"] if family in _SURFACE else s2 / s1
        ratio = b[key] / a[key]
        per_doubling = ratio ** (1.0 / math.log2(m))
        out.append((s1, s2, ratio <= m ** math.log2(bound), per_doubling,
                    f"{s1} -> {s2}: {key} {a[key]:.3f} -> {b[key]:.3f} ({ratio:.2f}x = {per_doubling:.2f}x per "
                    f"doubling, bound {bound}x)"))
    return out


def _gate(family, key):
    steps = _steps(family, key)
    if os.environ.get("XDEM_SCALING_LOG"):     # all measured ratios, for the acceptance table
        with open(os.environ["XDEM_SCALING_LOG"], "a") as f:
            f.writelines(f"{family}\t{key}\t{s1}\t{s2}\t{pd:.3f}\n" for s1, s2, ok, pd, t in steps)
    why, listed = _PENDING.get((family, key), ("", {}))
    new_bad = [t for s1, s2, ok, pd, t in steps if not ok and (s1, s2) not in listed]
    fixed = [t for s1, s2, ok, pd, t in steps if ok and (s1, s2) in listed]
    pending_bad = [t for s1, s2, ok, pd, t in steps if not ok and (s1, s2) in listed]
    # a pending step may not regress: its measured per-doubling ratio times CEILING is a hard ceiling
    over = [f"{t} [ceiling {listed[(s1, s2)] * CEILING:.2f}x]" for s1, s2, ok, pd, t in steps
            if (s1, s2) in listed and pd > listed[(s1, s2)] * CEILING]
    assert not new_bad, f"{family}: growth above the per-doubling bound: " + "; ".join(new_bad)
    assert not over, f"{family}: pending step regressed past its measured ratio x {CEILING}: " + "; ".join(over)
    assert key == "work" or not fixed, (f"{family} {key}: pending step(s) now meet the bound, delete them from "
                                        "_PENDING: " + "; ".join(fixed))
    if pending_bad:
        pytest.xfail(f"pending {why}: {family}: " + "; ".join(pending_bad))


@pytest.mark.slow
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_xdem_export_peak_rss_near_linear(family):
    _gate(family, "rss_mb")


@pytest.mark.slow
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_xdem_export_time_near_linear(family):
    _gate(family, "work")
