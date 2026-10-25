"""J-b1 P8: C0̂ materialize — the plan's C0 applied as the amplitude map on the register F₀ ∪ E_S (design §2.10; branch
scaffold term images), replacing the gate-level tableau route (kept as the comparator until J-b2).

Gates:
  * every retained shot: the C0̂ state equals the tableau-route state at 1e-12 (the expectations of every record
    operator, 24 random Paulis; and the dense state vector up to a global phase where
    n <= 12), kinds 0 and 2, framed records, mirror-dressed shots (MIRROR: a T ... T_DAG mirror around a CH);
  * mirror-dressed materialize vs DENSE (tests/test_ja_retention.dense_state of the exported frames vs the DENSE
    record oracle's conditional state is J-a's G-8; here: the C0̂ state's record expectations equal the record values);
  * MUT-C0HAT (the term images' sign, i^phase (-1)^par, dropped) -> red here.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from test_jb1_record_fidelity_l3 import KAPPA, BORN  # noqa: E402
from test_ja_retention import dense_state  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402
from _private import private_fixture, private_text  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-12

MIRROR = ("RX 0 1\nT 0\nH 2\nCH 2 1\nX_ERROR(0.2) 0\nZ_ERROR(0.2) 1\nT_DAG 0\nDEPOLARIZE1(0.1) 2\nMX 0 1 2\n"
          "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")
CORPUS = dict(KAPPA)
CORPUS.update(BORN)
CORPUS["ch_cultivation"] = open(os.path.join(ROOT, "tests", "data", "ch_cultivation.stim")).read()
CORPUS["cultivation_d5"] = open(os.path.join(ROOT, "tests", "data", "cultivation_d5.stim")).read()
CORPUS["factory"] = private_text("ccz_factory_a.stim")   # None when absent: the test skips
CORPUS["mirror"] = MIRROR
# eps = 1 on the moved rows (|-> and |1> preps): the term images' eps parity matters (MUT-C0HAT -> red)
# a CS pair split by an X fault on |1> (x) |+>: the residual's Z0Z1 term image carries eps parity 1 (the |1> row), and
# its E_S row (the |+> row) is superposed — the parity decides the relative phase (MUT-C0HAT -> red here)
CORPUS["eps_cs"] = ("R 0\nX 0\nRX 1\nCS 0 1\nX_ERROR(0.3) 1\nCS_DAG 0 1\nMX 1\nM 0\nDETECTOR rec[-1]\n"
                    "R 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")
CORPUS["mirror_eps"] = MIRROR.replace("RX 0 1\n", "RX 0 1\nZ 0 1\n").replace("H 2\n", "X 2\nH 2\nZ 2\n")


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def probes(s, rng, k=24):
    ops = [tuple(o) for o in s.record_ops()]
    n = len(ops[0][0])
    out = list(ops)
    for _ in range(k):
        x = (rng.random(n) < 0.15).astype(np.uint8)
        z = (rng.random(n) < 0.15).astype(np.uint8)
        out.append((x, z, int(np.dot(x, z) % 2)))           # Hermitian i^{x.z} X^x Z^z
    return out


def compare(s, R, rows, rng):
    P = probes(s, rng)
    worst = 0.0
    for i in rows:
        a = s.state_expectations(R, i, P)
        b = s.state_expectations(R, i, P, tableau=True)
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_c0hat_equals_tableau(name):
    text = CORPUS[name]
    if text is None:
        private_fixture("ccz_factory_a.stim")
    s = None
    try:
        s = fam(text)
        d, o, dc, R = s.sample_records(240, seed=17)
    except (ValueError, RuntimeError):
        s = None
    if s is None:
        pytest.skip("refused by name")
    rng = np.random.default_rng(5)
    step = 6 if s.num_detectors < 60 else 40         # the wide circuits: fewer rows (each state is O(chi n^2))
    assert compare(s, R, range(0, 240, step), rng) < TOL


def test_c0hat_equals_tableau_dense_small():
    s = fam(MIRROR)
    d, o, dc, R = s.sample_records(200, seed=3)
    for i in range(0, 200, 10):
        a = dense_state(s.materialize(R, i))
        b = dense_state(s.materialize(R, i, tableau=True))
        ov = abs(np.vdot(a, b))
        assert abs(ov - 1.0) < TOL, (i, ov)


def test_c0hat_framed_records():
    s = fam(F.PORT)
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 2, size=(200, 2)).astype(np.uint8)
    d, o, dc, R = s.sample_records(200, seed=3, input_pauli=frames)
    assert compare(s, R, range(0, 200, 5), rng) < TOL


def test_c0hat_records_are_definite_at_y():
    """Every record operator has expectation (-1)^y on the C0̂ state (the retained state is projected at y)."""
    s = fam(MIRROR)
    d, o, dc, R = s.sample_records(200, seed=9)
    ops = [tuple(o) for o in s.record_ops()]
    for i in range(0, 200, 7):
        e = s.state_expectations(R, i, ops)
        y = R.y(i)
        assert np.max(np.abs(e - (1 - 2 * np.asarray(y[: len(ops)], float)))) < TOL
