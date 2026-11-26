"""xdem Task 6 (SPEC §6 T5(b)): cross-route consistency by sampling. Port of tests/test_dc_cross_route.py (19f2f78)
with `_xtim._XdemSampler` on the `format="xdem"` export.

Route A (file only): `Circuit.detector_error_model_text(format="xdem")` -> `_xtim._XdemSampler` -> `.sample` ->
`.score(obs, cspace, zero flips)` (status-scored pass, zero decoder).
Route B (circuit): `Circuit.compile_detector_sampler(seed)` -> `.sample(separate_observables=True, bit_packed=True)`
-> `xtim.score_plain(obs, zero flips)` (plain XOR score, zero decoder).

Both routes are drawn in chunks with a NEW seed per chunk on each side. Seeds are fixed per case (listed in CASES),
disjoint across cases and sides, so failures reproduce.

Statistics, per case (two-sample z-scores):
  * pass mean (1 - infidelity of the zero decoder): z = (m_A - m_B) / sqrt(var_A / N + var_B / N), sample variances;
    skipped when O = 0;
  * acceptance (all detectors zero) and each detector's marginal: pooled two-proportion z; a detector whose pooled
    rate is exactly 0 or 1 gets z = 0 and is counted as a test;
  * detector x failure, per detector j (only when O >= 1): E[D_j * (1 - pass)], two-sample z with each side's own
    sample variance (valid across the two scorings by Rao-Blackwell: the file's pass is E[XOR pass | d, T]).
Fail if any |z| > 5. Under the null each |z| > 5 has probability <= 5.73e-7, so with N_tests z-tests P(any) <=
N_tests * 5.73e-7 (union bound). The in-repo set runs 310 tests (fast leg: 42, slow legs: 268) -> <= 1.78e-4.
`test_accept_and_l0_given_accept_vs_xtim` is the port of tests/test_dc_sampler.py's (dcfast review: P(L0 | accepted)
on cultivation_d5), on the xdem fixture.
"""
import math
import time

import numpy as np
import pytest

import xtim
from xtim import _xtim

from _xdem_circuits import circuit, xdem_export
from test_xdem_sampler import DATA

Z_MAX = 5.0


def _z_prop(k1, k2, n):
    """Pooled two-proportion z for k1/n vs k2/n; 0 if the pooled rate is 0 or 1 (then k1 == k2)."""
    p = (k1 + k2) / (2 * n)
    if p <= 0 or p >= 1:
        assert k1 == k2
        return 0.0
    return (k1 / n - k2 / n) / math.sqrt(p * (1 - p) * 2 / n)


def _z_mean(s1, ss1, s2, ss2, n):
    """Two-sample z for means from sums / sums of squares (sample variances, equal N)."""
    m1, m2 = s1 / n, s2 / n
    v1 = max(ss1 / n - m1 * m1, 0.0) * n / (n - 1)
    v2 = max(ss2 / n - m2 * m2, 0.0) * n / (n - 1)
    if v1 + v2 == 0:
        return 0.0 if m1 == m2 else math.inf
    return (m1 - m2) / math.sqrt((v1 + v2) / n)


class _Acc:
    """Running sums for one route: pass sum / sum of squares, accepted count, per-detector fire counts, per-detector
    sums of D_j * (1 - pass) and of its square."""

    def __init__(self, D):
        self.s = 0.0
        self.ss = 0.0
        self.acc = 0
        self.det = np.zeros(D, np.int64)
        self.F = np.zeros(D)
        self.FF = np.zeros(D)

    def add(self, dets_packed, passv, D):
        bits = np.unpackbits(dets_packed, axis=1, bitorder="little")[:, :D]
        self.det += bits.sum(axis=0, dtype=np.int64)
        self.acc += int((~bits.any(axis=1)).sum()) if D else len(passv)
        self.s += float(passv.sum())
        self.ss += float((passv * passv).sum())
        f = 1.0 - np.asarray(passv, np.float64)
        bf = bits.astype(np.float64)
        self.F += bf.T @ f
        self.FF += bf.T @ (f * f)   # D_j in {0,1}: (D_j f)^2 = D_j f^2


def cross_compare(circuit_text, shots, seed_x, seed_xt, *, chunk=500_000, threads=4):
    """Run both routes on `shots` shots each (chunks of `chunk`; chunk i uses seed_x + i / seed_xt + i)."""
    circ = xtim.Circuit(circuit_text)
    t0 = time.time()
    text = xdem_export(circuit_text)
    t_export = time.time() - t0
    S = _xtim._XdemSampler(text, 65536)
    D, O = S.num_detectors, S.num_observables
    A, B = _Acc(D), _Acc(D)
    t_x = t_xt = 0.0
    nchunks = -(-shots // chunk)
    for i in range(nchunks):
        n = min(chunk, shots - i * chunk)
        zero = np.zeros((n, O), np.uint8)
        t0 = time.time()
        d, o, cs = S.sample(n, seed_x + i, threads, 0, False)
        A.add(d, S.score(o, cs, zero), D)
        t_x += time.time() - t0
        t0 = time.time()
        X = circ.compile_detector_sampler(seed=seed_xt + i)
        xd, xo = X.sample(n, separate_observables=True, bit_packed=True, threads=threads)
        assert xd.shape == d.shape
        B.add(xd, xtim.score_plain(xo, zero.astype(bool)), D)
        t_xt += time.time() - t0
    z_pass = _z_mean(A.s, A.ss, B.s, B.ss, shots) if O else None
    z_acc = _z_prop(A.acc, B.acc, shots) if D else None
    z_det = [_z_prop(int(a), int(b), shots) for a, b in zip(A.det, B.det)]
    z_dxf = [_z_mean(A.F[j], A.FF[j], B.F[j], B.FF[j], shots) for j in range(D)] if O else []
    zs = ([z_pass] if O else []) + ([z_acc] if D else []) + z_det + z_dxf
    return dict(
        D=D, O=O, shots=shots, num_lines=S.num_lines, num_rotating=S.num_rotating,
        pass_x=A.s / shots, pass_xt=B.s / shots, z_pass=z_pass,
        acc_x=A.acc / shots, acc_xt=B.acc / shots, z_acc=z_acc,
        z_det=z_det, z_dxf=z_dxf, n_const_det=int(sum((a + b) in (0, 2 * shots) for a, b in zip(A.det, B.det))),
        n_tests=len(zs), max_abs_z=max(abs(z) for z in zs) if zs else 0.0,
        argmax=_argmax_name(z_pass, z_acc, z_det, z_dxf),
        seeds=dict(xdem=(seed_x, seed_x + nchunks - 1), xtim=(seed_xt, seed_xt + nchunks - 1), chunk=chunk),
        t_export=t_export, t_x=t_x, t_xt=t_xt)


def _argmax_name(z_pass, z_acc, z_det, z_dxf):
    c = ([("pass", z_pass), ("accept", z_acc)] + [(f"D{j}", z) for j, z in enumerate(z_det)]
         + [(f"D{j}xfail", z) for j, z in enumerate(z_dxf)])
    c = [(k, z) for k, z in c if z is not None]
    return max(c, key=lambda kz: abs(kz[1]))[0] if c else None


def summary(name, r):
    zp = "trivial (O=0)" if r["z_pass"] is None else f"{r['z_pass']:+.2f}"
    return (f"{name}: N={r['shots']}/side D={r['D']} O={r['O']} | 1-pass xdem {1 - r['pass_x']:.6g} xtim "
            f"{1 - r['pass_xt']:.6g} z {zp} | accept xdem {r['acc_x']:.6g} xtim {r['acc_xt']:.6g} z "
            f"{r['z_acc']:+.2f} | det max|z| {max(map(abs, r['z_det'])):.2f} ({r['n_const_det']} constant) | det x fail "
            f"max|z| {max(map(abs, r['z_dxf'])) if r['z_dxf'] else float('nan'):.2f} | "
            f"{r['n_tests']} tests, max|z| {r['max_abs_z']:.2f} at {r['argmax']} | seeds {r['seeds']} | "
            f"export {r['t_export']:.2f}s xdem {r['t_x']:.2f}s xtim {r['t_xt']:.2f}s")


# (name, shots per side, xdem seed base, xtim seed base, slow); chunk i uses base + i. The dc legs' seeds + 10000.
CASES = [
    ("cultivation_d3_faithful", 200_000, 91_000, 92_000, False),
    ("cultivation_d3_faithful", 4_000_000, 91_100, 92_100, True),
    ("cultivation_d5", 4_000_000, 91_200, 92_200, True),
    ("distillation_15_1_3", 4_000_000, 91_300, 92_300, True),
    ("distillation_15_1_3:gatenoise", 4_000_000, 91_400, 92_400, True),
    ("ch_cultivation", 1_000_000, 91_500, 92_500, False),
    ("ch_cultivation", 8_000_000, 91_600, 92_600, True),
]
N_TESTS = {"cultivation_d3_faithful": 42, "cultivation_d5": 216, "distillation_15_1_3": 5,
           "distillation_15_1_3:gatenoise": 5, "ch_cultivation": 4}


def _text(name):
    """the registry's circuit; ch_cultivation (CH in the xdem export) is the example, outside the registry"""
    return xtim.load_example(name).text if name == "ch_cultivation" else circuit(name)


@pytest.mark.parametrize("name,shots,sx,sxt", [
    pytest.param(n, s, a, b, id=f"{n}-{s}", marks=[pytest.mark.slow] if slow else [])
    for n, s, a, b, slow in CASES])
def test_cross_route(name, shots, sx, sxt):
    r = cross_compare(_text(name), shots, sx, sxt)
    print(summary(name, r))
    assert r["n_tests"] == N_TESTS[name]
    if r["O"] == 0:
        assert r["pass_x"] == 1.0 and r["pass_xt"] == 1.0   # no observable: pass is 1 on every shot, both routes
    assert r["max_abs_z"] <= Z_MAX, summary(name, r)


def _acc_l0(dets, obs):
    acc = ~dets.any(axis=1)
    return int(acc.sum()), int((acc & (obs[:, 0] & 1).astype(bool)).sum())


@pytest.mark.slow
def test_accept_and_l0_given_accept_vs_xtim(shots=4_000_000):
    """the xdem route (tests/data/xdem/cultivation_d5.xdem) vs xtim's compile_detector_sampler on
    xtim/_examples/cultivation_d5.stim, fresh seeds on both sides: acceptance rate and P(L0 | accepted), |z| <= 5"""
    S = _xtim._XdemSampler((DATA / "cultivation_d5.xdem").read_text(), 65536)
    d1, o1, _ = S.sample(shots, 9511, 4, 0, False)
    a1, l1 = _acc_l0(np.unpackbits(d1, axis=1, bitorder="little")[:, :S.num_detectors].astype(bool),
                     np.unpackbits(o1, axis=1, bitorder="little"))
    X = xtim.Circuit(circuit("cultivation_d5")).compile_detector_sampler(seed=9611)
    d2, o2 = X.sample(shots, separate_observables=True, threads=4)
    a2, l2 = _acc_l0(d2, o2.astype(np.uint8))
    pa1, pa2 = a1 / shots, a2 / shots
    pa = (a1 + a2) / (2 * shots)
    za = (pa1 - pa2) / np.sqrt(pa * (1 - pa) * 2 / shots)
    q1, q2 = l1 / a1, l2 / a2
    q = (l1 + l2) / (a1 + a2)
    zl = (q1 - q2) / np.sqrt(q * (1 - q) * (1 / a1 + 1 / a2)) if 0 < q < 1 else 0.0
    print(f"cultivation_d5 {shots}: accepted xdem {a1} / xtim {a2} (z {za:.2f}); L0|acc xdem {l1} / xtim {l2} "
          f"(z {zl:.2f})")
    assert abs(za) <= 5 and abs(zl) <= 5
