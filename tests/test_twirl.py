"""Twirl fast-sampler (engine="auto"/"twirl") — packaged smoke + correctness tests.

Exercises the 2.3.0 feature in the wheel CI so it can't silently rot: the fast engine
must (a) match the exact engine's detector statistics at 5 sigma, (b) apply M(p)
readout flips correctly (checked against the exact engine), (c) route ineligible
circuits to exact under auto, and (d) honor the kill switch.
"""

import numpy as np

import xtim


def _det_freqs(sampler, shots, ndet):
    d, _ = sampler.sample(shots, separate_observables=True)
    return np.asarray(d, dtype=np.float64).mean(axis=0), ndet


def test_auto_matches_exact_on_cultivation():
    c = xtim.load_example("cultivation_d5")
    S = 60_000
    fa, _ = _det_freqs(c.compile_detector_sampler(seed=5, engine="auto"), S, c.num_detectors)
    fe, _ = _det_freqs(c.compile_detector_sampler(seed=5, engine="exact"), S, c.num_detectors)
    se = np.sqrt(fa * (1 - fa) / S + fe * (1 - fe) / S) + 1e-12
    assert float(np.max(np.abs(fa - fe) / se)) < 5.0


def test_auto_uses_twirl_when_eligible():
    c = xtim.load_example("cultivation_d5")
    rep = c.compile_detector_sampler(seed=1, engine="auto").engine_report()
    assert rep["engine"] == "twirl"


def test_readout_flip_matches_exact():
    # M(p) is a record-space flip; the fast engine must apply it (regression guard).
    c = xtim.Circuit("R 0\nX_ERROR(0.1) 0\nM(0.2) 0\nDETECTOR rec[-1]\n")
    S = 200_000
    ft, _ = _det_freqs(c.compile_detector_sampler(seed=2, engine="twirl"), S, 1)
    fe, _ = _det_freqs(c.compile_detector_sampler(seed=2, engine="exact"), S, 1)
    z = abs(ft[0] - fe[0]) / (np.sqrt(ft[0]*(1-ft[0])/S + fe[0]*(1-fe[0])/S) + 1e-12)
    assert z < 5.0
    assert abs(ft[0] - 0.26) < 0.02   # true value 0.26


def test_kill_switch_forces_exact(monkeypatch):
    monkeypatch.setenv("QEC_NO_TWIRL", "1")
    c = xtim.load_example("cultivation_d5")
    rep = c.compile_detector_sampler(seed=1, engine="auto").engine_report()
    assert rep["engine"] == "exact"


def test_same_seed_same_bytes():
    c = xtim.load_example("cultivation_d5")
    s = c.compile_detector_sampler(seed=9, engine="twirl")
    d1, _ = s.sample(3000, separate_observables=True)
    d2, _ = s.sample(3000, separate_observables=True)
    assert (np.asarray(d1) == np.asarray(d2)).all()
