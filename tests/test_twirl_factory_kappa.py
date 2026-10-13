"""The f=3 CCZ factory (Menon et al., arXiv:2610.10731; vendored as
tests/data/menon_factory_f3_realccz.stim, 729 qubits, 312 detectors, 54 physical CCZ, SD6 noise at
p=1e-3) is the one circuit in the corpus whose twirl plans have kappa >= 2 kernel directions
(per-shot kappa histogram at seed 2: 0: 91.0 %, 1: 4.06 %, 2: 4.82 %, 3: 0.12 %, 4: 0.012 %).
Before the canonical dual-basis fold (derivation note 2026-10-12) the legacy fold tripped on
164 of its 383 kappa>=2 plans and 365 of 25,000 shots went to the exact per-shot engine
(~19 s/shot on this circuit); now every plan folds and ``exact_shots == 0``.

Law gate: the detector statistics of the new sampler must agree with the pre-change engine's —
the 3.1.12 twirl sampler, whose law is exact on this circuit (verified laws for the plans it kept
and the exact per-shot path for the ones it tripped on) — at 5 sigma on every detector marginal
and on the pairwise joints of the detectors that kappa>=1 shots touch. The reference counts are
pinned in tests/data/menon_factory_f3_golden.json (1e5 shots, seed 3, 3.1.12 engine); the exact
engine itself is not a feasible reference here (0.05 shots/s: 1e5 shots would take 22 days).

WHAT THIS GATE DOES AND DOES NOT CHECK (evidence labelling): it checks the record LAW identity and
``exact_shots == 0`` only. It is structurally BLIND to the pairing condition (D2) c_j.delta_k =
delta_jk and to the base-sign bits kernel_base: the factory has one diagonal magic frame, so the
Z-dressings commute with the magic and every kernel-logical product has stabilizer expectation
0 or +-1 (measured |<L>| <= 4e-16 on all 3,661 products of its kappa>=1 plans) — a wrong pairing
or base bit leaves its detector law unchanged. The mechanism behind the pairing / base-sign
claims is the dense-oracle fixture test tests/test_twirl_kappa_dual.py (biased Born laws,
mutations m1-m3 fail) together with the solver ctest kernel_dual_basis.
"""
import json
import pathlib

import numpy as np
import pytest

import xtim

_DATA = pathlib.Path(__file__).parent / "data"
_SHOTS = 25000
_SEED = 7


@pytest.fixture(scope="module")
def run():
    text = (_DATA / "menon_factory_f3_realccz.stim").read_text()
    golden = json.load(open(_DATA / "menon_factory_f3_golden.json"))
    s = xtim.compile_twirl_sampler(text, skip_refused_observables=True, selfcheck=0)
    dets, _obs = s.sample(_SHOTS, seed=_SEED)
    ndet = int(golden["ndet"])
    d = np.unpackbits(np.asarray(dets), axis=1, bitorder="little")[:, :ndet].astype(np.uint8)
    return dict(sampler=s, d=d, golden=golden, ndet=ndet)


def _z(c1, n1, c2, n2):
    p1, p2 = c1 / n1, c2 / n2
    pp = (c1 + c2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return np.where(se > 0, (p1 - p2) / se, 0.0)


def test_no_exact_shots_and_no_fallback(run):
    """Every shot is a plan-path shot: kappa>=2 is not a fallback reason any more."""
    rep = run["sampler"].channel_report()
    assert rep["exact_shots"] == 0, rep["exact_shots"]
    assert rep["plans"] > 1000


def test_detector_marginals_vs_golden(run):
    g, d = run["golden"], run["d"]
    c_new = d.sum(0).astype(float)
    c_old = np.asarray(g["marginal_counts"], float)
    z = _z(c_old, g["shots"], c_new, _SHOTS)
    assert np.max(np.abs(z)) < 5.0, (int(np.argmax(np.abs(z))), float(np.max(np.abs(z))))


def test_touched_pairwise_joints_vs_golden(run):
    g, d = run["golden"], run["d"]
    worst = 0.0
    for a, b, c_old in g["pair_counts"]:
        c_new = float(np.sum(d[:, a] & d[:, b]))
        worst = max(worst, abs(float(_z(c_old, g["shots"], c_new, _SHOTS))))
    assert worst < 5.0, worst
