"""S8 RF1: the logical expectation λ(u, v) = <b| Z^v X^u |b> on the bare's free register (plan r4.1 RF1, M2).

ONE formula, λ(u, v) = Σ_{y in supp b, ascending, b_{y^u} != 0} conj(b_y) b_{y^u} (-1)^{v.y} (a structural +0.0 when no
support pair exists); the dense 4^k table and the sparse exact-key memo are memos of exactly it, so a value never
depends on the path (budget, table fit). Oracle: the bare state's exact pauli_expectation of the lab Pauli
U Z^v X^u U^dag (<= 4 ulp * chi absolute; the structural zeros exactly 0 — sparse-support bares in the C++ dense
oracle, cpp/tests/test_record_fidelity_ref.cpp test_lambda).
"""
import os
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")

# a magic output register with k = 4 free rows (route 3b) beside a stabilizer rest with a Born read
_SMALL = ("RX 0 1 2 3\nT 0\nT_DAG 1\nT 2\nT 3\nCX 0 1\nCZ 2 3\nH 1\nCX 2 0\nS 3\n"
          "R 4\nH 4\nCX 4 5\nM 5\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1 2 3\n")


def _fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def _bits(row):
    return [int(q) for q in np.flatnonzero(row)]


def _pmul(a, b):
    """exact product of (x set, z set, phase) Paulis, i^p X^x Z^z"""
    x1, z1, p1 = a
    x2, z2, p2 = b
    return (x1 ^ x2, z1 ^ z2, (p1 + p2 + 2 * (len(z1 & x2) % 2)) % 4)


def _lab(fr, u, v):
    """(-1)^{v.eps} U Z^v X^u U^dag from the bare frames (the stab rows carry the eps sign: the free register is
    |eps + x>, so Z on a free row reads (-1)^{eps} on it)"""
    free = [int(a) for a in fr["free"]]
    sx, sz, sp = (np.asarray(fr[k]) for k in ("stab_x", "stab_z", "stab_phase"))
    dx, dz, dp = (np.asarray(fr[k]) for k in ("destab_x", "destab_z", "destab_phase"))
    P = (frozenset(), frozenset(), 0)
    for d, a in enumerate(free):
        if (v >> d) & 1:
            P = _pmul(P, (frozenset(_bits(sx[a])), frozenset(_bits(sz[a])), int(sp[a]) % 4))
    for d, a in enumerate(free):
        if (u >> d) & 1:
            P = _pmul(P, (frozenset(_bits(dx[a])), frozenset(_bits(dz[a])), int(dp[a]) % 4))
    return P


def _setup(text):
    s = _fam(text)
    rf = s.record_fidelity()
    fr = TA.bare_frames(s)
    B = xtim.bare_state_of(text)
    return s, rf, fr, B


def _bytes(c):
    return np.array([c.real, c.imag]).tobytes()


@pytest.mark.parametrize("budget", [None, 0])
def test_memo_equals_formula_every_entry_small(budget):
    """k <= 6: every entry through the memo (dense table, or the sparse memo with the table budget forced to 0)
    carries the formula's bytes; repeated reads return the memo."""
    s, rf, fr, _B = _setup(_SMALL)
    k = len(fr["free"])
    assert 2 <= k <= 6, k
    if budget is not None:
        TA._rf_set_lambda_budget(rf, budget)
    for u in range(1 << k):
        for v in range(1 << k):
            d = TA._rf_lambda(rf, u, v, True)
            assert _bytes(TA._rf_lambda(rf, u, v)) == _bytes(d)
            assert _bytes(TA._rf_lambda(rf, u, v)) == _bytes(d)
    assert TA._rf_lambda_mode(rf) == (1 if budget is None else 2)


def test_memo_equals_formula_sampled_color5():
    """k = 9 (color5): 10^4 sampled entries, dense and sparse memo vs the formula, byte for byte."""
    text = open(os.path.join(DATA, "magna_color5_fid.stim")).read()
    s = _fam(text)
    rd, rs = s.record_fidelity(), s.record_fidelity()
    TA._rf_set_lambda_budget(rs, 0)
    k = 9
    rng = np.random.default_rng(5)
    uv = rng.integers(0, 1 << k, (10000, 2))
    for u, v in uv:
        d = _bytes(TA._rf_lambda(rd, int(u), int(v), True))
        assert _bytes(TA._rf_lambda(rd, int(u), int(v))) == d
        assert _bytes(TA._rf_lambda(rs, int(u), int(v))) == d
    assert TA._rf_lambda_mode(rd) == 1 and TA._rf_lambda_mode(rs) == 2


@pytest.mark.parametrize("name", ["small", "color5", "rp2"])
def test_lambda_vs_pauli_expectation(name):
    """λ(u, v) = <B| U Z^v X^u U^dag |B> (the bare state's exact expectation): <= 4 ulp * chi * max|b|^2 absolute;
    structural zeros (no support pair) exactly +0.0."""
    text = _SMALL if name == "small" else open(os.path.join(DATA, f"magna_{name}_fid.stim")).read()
    s, rf, fr, B = _setup(text)
    k = len(fr["free"])
    sig, co = np.asarray(fr["sigma"]), np.asarray(fr["coeff"])
    supp = sorted({int(sum(int(b) << d for d, b in enumerate(row))) for row, c in zip(sig, co) if c != 0})
    chi = len(supp)
    amax = float(np.max(np.abs(co)) ** 2)
    rng = np.random.default_rng(len(name))
    N = (1 << (2 * k)) if k <= 4 else 400
    pairs = [(u, v) for u in range(1 << k) for v in range(1 << k)] if k <= 4 else \
        [tuple(int(t) for t in x) for x in rng.integers(0, 1 << k, (N, 2))]
    sset = set(supp)
    nzero = nval = 0
    for u, v in pairs:
        lam = TA._rf_lambda(rf, u, v)
        x, z, p = _lab(fr, u, v)
        pe = complex(B.pauli_expectation(sorted(x), sorted(z), p))
        if not any((y ^ u) in sset for y in supp):
            assert _bytes(lam) == _bytes(0j), (u, v, lam)
            nzero += 1
        else:
            nval += 1
        assert abs(lam - pe) <= 4 * 2.0 ** -52 * chi * amax, (u, v, lam, pe)
    assert nval > 0      # (the corpus bares have full support; structural zeros: ctest record_fidelity_ref)
