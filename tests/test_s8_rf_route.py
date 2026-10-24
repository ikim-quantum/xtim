"""S8 RF: the twirled Pauli-expansion register fidelity (rfref/fable-decoded-design.md; plan r4.1 RF3-RF6), the
production route 3b.

Per row the route reports a structural zero class (test lever _rf_row_log, memo bypassed): 0 INFEASIBLE (the GF(2)
constraint solve fails: F = 0 exactly), 1 BORN-ZERO (every sector's conditioned norm is zero, ref_ny), 2 GAUSS-ZERO
(the exact Gauss sum vanishes on every surviving sector), 3 VALUE. Oracles: the dense partial-trace register oracle
(tests/_register_oracle.py) on the fixture corpus; the exact Gaussian-integer ENUMERATION of every fiber
(_rf_hform_oracle, fibers <= 12) byte-identical to the Gauss route; the memo byte-identical to the memo-free
evaluation (the key carries everything the value reads). The C++ oracle (cpp/tests/test_record_fidelity_ref.cpp) adds
the statevector on random instances, padding invariance, the two-word fiber and the RF_SECTOR_RANK / RF_FIBER_RANK
boundaries; the per-class comparison to the parent (c066608) is the stage gate (s8/STATUS.md).
"""
import os
import sys

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from xtim.record_fidelity import RecordFidelity
from _register_oracle import bare_rho, register_F

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "ledger"))
try:  # the performance ledger (benchmarks/ledger) is not part of the public tree; its legs skip when absent
    import s8_fiddec as LEGS  # noqa: E402
except ImportError:
    LEGS = None
import test_rfref_route as RF  # noqa: E402


def _eval(s, buf, corr=None, oracle=False, log=True, shots=None, keep_memo=False):
    rf = RecordFidelity(s)
    if log:
        TA._rf_row_log(rf._rf, True)
    if oracle:
        TA._rf_hform_oracle(rf._rf, True)
    F = np.asarray(rf.evaluate(buf, shots, correction=corr, keep_memo=keep_memo))
    cls = np.asarray(TA._rf_take_row_log(rf._rf)) if log else None
    return F, cls, rf.stats()


def _classes_ok(F, cls):
    assert np.all(F[(cls == 0) | (cls == 1) | (cls == 2)] == 0.0)
    return np.bincount(cls[cls != 255], minlength=4).tolist()


def _oracle_bytes(s, buf, corr, F, cls, shots=None):
    Fo, clso, _st = _eval(s, buf, corr, oracle=True, shots=shots)
    fit = clso != 255
    assert np.array_equal(clso[fit], cls[fit])
    assert Fo[fit].tobytes() == F[fit].tobytes()
    return int(fit.sum())


CORPUS = [(name, txt) for name, txt in RF.FIXTURES] + \
         [(f"born{nb}", RF._born_rest(nb, "DEPOLARIZE1(0.3) 0")) for nb in (40, 130, 540)]


@pytest.mark.parametrize("name", [c[0] for c in CORPUS])
def test_fixture_corpus_vs_dense_oracle_and_enumeration(name):
    """Every fixture row (half with a random output correction) == the dense partial-trace oracle (<= 1e-12); the
    structural classes score 0; the enumeration oracle is byte-identical."""
    txt = dict(CORPUS)[name]
    s = RF._fam(txt)
    N = 40
    buf = s.sample_barrier(N, seed=3)
    ow = list(xtim.compile_twirl_sampler(txt, selfcheck=0).output_wires())
    rng = np.random.default_rng(len(name))
    corr = rng.integers(0, 2, (N, 2, len(ow))).astype(np.uint8)
    corr[: N // 2] = 0
    F, cls, st = _eval(s, buf, corr)
    rhoB = bare_rho(txt, ow)
    Fd = np.array([register_F(rhoB, buf, i, ow, corr[i]) for i in range(N)])
    assert np.max(np.abs(F - Fd)) <= 1e-12
    _classes_ok(F, cls)
    assert _oracle_bytes(s, buf, corr, F, cls) == N


def test_2048_born_records():
    s = RF._compile(RF._born_rest(2048, "DEPOLARIZE1(0.3) 0"))
    F, cls, st = _eval(s, s.sample_barrier(12, seed=2))
    _classes_ok(F, cls)
    assert st["route"] == 2 and np.all(np.min(np.abs(F[:, None] - np.array([0.0, 0.5, 1.0])[None, :]), axis=1) <= 1e-12)


@pytest.mark.parametrize("leg", ["color5/p1e-3/zero", "color5/p1e-3/dec", "color5/p1e-3/rand", "rp2/p1e-3/zero",
                                 "rp2/p1e-3/rand", "color5/p1e-4/zero"])
def test_corpus_legs(leg):
    """The decoded legs (20k, seed 7) on their first 1500 rows: classes, the enumeration oracle byte-identical on every
    row whose fibers fit; the memo (keep_memo) byte-identical to the memo-free evaluation."""
    if LEGS is None:
        pytest.skip("benchmarks/ledger (s8_fiddec) not present")
    s, buf, corr = LEGS.inputs(ROOT, leg, testapi=True)
    rows = np.arange(1500, dtype=np.int64)
    F, cls, st = _eval(s, buf, corr, shots=rows)
    c = _classes_ok(F, cls)
    assert c[3] > 0
    assert _oracle_bytes(s, buf, corr, F, cls, shots=rows) > 1000
    Fm, _c, stm = _eval(s, buf, corr, log=False, shots=rows, keep_memo=True)
    assert Fm.tobytes() == F.tobytes() and stm["memo_hits"] > 0
