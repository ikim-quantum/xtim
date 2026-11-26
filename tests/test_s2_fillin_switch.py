"""S2-FINALFIX F1: the planes' working rows switch to packed words when their elimination fills in.

The family (the Stage 2 final review's hw / hwr): half the qubits |+>, each other qubit the parity of w random
controls, X_ERROR, R rounds of M, round-to-round and parity detectors (random-expander parity checks). The certified
generators are sparse, so the starting-weight rule picks sparse working rows, but the planes' RREF fills in (~25x):
before the fix, compile was 2-8x slower than 453a778 (hwr4_3 n=1200: 2.4 s against 0.38 s with
XTIM_PLANES_LAYOUT=dense). The lever is read once per process, so each layout runs in its own subprocess.
"""
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def family(n, w, rounds=1, seed=29):
    rr = random.Random(seed)
    m = n // 2
    L = ["R " + " ".join(map(str, range(n))), "H " + " ".join(map(str, range(m)))]
    ctl = {}
    for j in range(m, n):
        cs = rr.sample(range(m), w)
        ctl[j] = cs
        L += [f"CX {c} {j}" for c in cs]
    L.append("X_ERROR(0.01) " + " ".join(map(str, range(n))))
    L += ["M " + " ".join(map(str, range(n)))] * rounds
    for r in range(1, rounds):
        L += [f"DETECTOR rec[-{(rounds - r + 1) * n - i}] rec[-{(rounds - r) * n - i}]" for i in range(n)]
    for j in range(m, n):
        L.append("DETECTOR " + " ".join(f"rec[-{n - q}]" for q in [j] + ctl[j]))
    for _ in range(n):                                   # products of two parity checks
        a, b = rr.sample(range(m, n), 2)
        qs = set([a] + ctl[a]) ^ set([b] + ctl[b])
        if qs:
            L.append("DETECTOR " + " ".join(f"rec[-{n - q}]" for q in sorted(qs)))
    L.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{n - q}]" for q in range(m, min(n, m + 5))))
    return "\n".join(L) + "\n"


def _run(code, layout):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("XTIM_PLANES_LAYOUT", None)
    if layout != "auto":
        env["XTIM_PLANES_LAYOUT"] = layout
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200,
                       cwd=str(ROOT))
    assert p.returncode == 0, (layout, p.stdout[-2000:], p.stderr[-4000:])
    return p.stdout.strip().splitlines()[-1]


_SAMPLES = """
import hashlib, sys
sys.path.insert(0, {root!r}); sys.path.insert(0, {here!r})
import numpy as np, xtim
from test_s2_fillin_switch import family
out = []
for n, w, r in ((600, 8, 1), (400, 4, 3)):
    text = family(n, w, r)
    d, o = xtim.Circuit(text).compile_detector_sampler(seed=7).sample(500, separate_observables=True)
    out.append(hashlib.md5(np.ascontiguousarray(d).tobytes() + np.ascontiguousarray(o).tobytes()).hexdigest())
    tw = xtim.compile_twirl_sampler(text, selfcheck=0).sample(200, seed=5)
    out.append(hashlib.md5(b"".join(np.ascontiguousarray(a).tobytes() for a in tw)).hexdigest())
print(" ".join(out))
"""


def test_fillin_family_layouts_identical():
    code = _SAMPLES.format(root=str(ROOT), here=str(Path(__file__).resolve().parent))
    got = {lay: _run(code, lay) for lay in ("auto", "sparse", "dense")}
    assert got["auto"] == got["sparse"] == got["dense"], got


_TIMES = """
import sys, time
sys.path.insert(0, {root!r}); sys.path.insert(0, {here!r})
import xtim
from test_s2_fillin_switch import family
c = xtim.Circuit(family(1200, 4, 3))
best = 1e30
for _ in range(3):
    t0 = time.process_time()
    c.compile_detector_sampler(seed=1)
    best = min(best, time.process_time() - t0)
print(best)
"""


def test_fillin_family_compile_not_slower_than_words():
    """The automatic switch against the forced packed words (XTIM_PLANES_LAYOUT=dense, base 453a778's speed): min of
    3 process-time compiles each, single BLAS thread. Before the fix the ratio was ~6x."""
    code = _TIMES.format(root=str(ROOT), here=str(Path(__file__).resolve().parent))
    t_auto = float(_run(code, "auto"))
    t_dense = float(_run(code, "dense"))
    assert t_auto <= 1.5 * t_dense + 0.1, (t_auto, t_dense)
