"""J-b1 P13 (plan r3.2, r3.1-fix R3-5): the runtime exact-construction counter.

Each exact-engine entry class bumps the counter when built OUTSIDE an explicit-exact scope (raw binding calls),
and the public explicit entry points build it inside one (no bump). The counter has no effect on outputs. The
session-level "counter == accounted over the whole suite" fixture is tests/conftest.py (F); the raw-binding bumps
below are accounted there.
"""
import warnings

import pytest

import xtim
from _exact_engine import (compile_exact_branches, exact_detector_sampler, exact_sampler,  # noqa: E402
                           resolve)  # (exact-out T5: the exact engine is test API)
from xtim import _xtim
from xtim import _xtim_testapi as TA
from xtim._exact_scope import exact_constructions, explicit_exact

pytestmark = pytest.mark.filterwarnings("ignore")

TEXT = "R 0\nT 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nR 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\n"


def delta(fn):
    from conftest import account_exact_constructions
    a = exact_constructions()
    fn()
    d = exact_constructions() - a
    account_exact_constructions(d)       # deliberate raw-binding constructions (the session fixture's ledger)
    return d


# each class, built through its raw binding (no scope): the counter moves
RAW = {
    "SamplerProgram + FramedCircuitShotSampler (CompiledProgram)": lambda: TA.CompiledProgram(TEXT, ""),
    "ExactBranchProgram": lambda: TA.ExactBranchProgram(TEXT),
    "DEM export": lambda: _xtim.export_dem_text(TEXT, True, False, False, False, [], [], 0.0),
}
# twirl-routes T3 (coordinator 2026-10-30): detector_determinism is a bare-state Born check, not an exact construction
# (it no longer bumps the counter, scoped or not).


@pytest.mark.parametrize("name", sorted(RAW))
def test_each_exact_class_bumps_outside_a_scope(name):
    assert delta(RAW[name]) >= 1, name


@pytest.mark.parametrize("name", sorted(RAW))
def test_each_exact_class_is_silent_inside_a_scope(name):
    def scoped():
        with explicit_exact():
            RAW[name]()
    assert delta(scoped) == 0, name


# (exact-out T5: the exact sampler / raw sampler / forced branches / reference resolution are test API —
#  tests/_exact_engine.py — and keep their explicit scope there)
PUBLIC = {
    "exact_detector_sampler (test API)": lambda: exact_detector_sampler(xtim.Circuit(TEXT), seed=1).sample(10),
    "exact_sampler (test API)": lambda: exact_sampler(xtim.Circuit(TEXT), seed=1).sample(10),
    "compile_exact_branches (test API)": lambda: compile_exact_branches(xtim.Circuit(TEXT)),
    "detector_error_model": lambda: xtim.Circuit(TEXT).detector_error_model(),
    "compile_twirl_sampler (noiseless reference inside)": lambda: xtim.compile_twirl_sampler(TEXT, selfcheck=0),
    "exact reference resolution (test API)": lambda: resolve(xtim.Circuit(TEXT + "R 2\nM 2\nDETECTOR rec[-1]\n")),
}


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_public_explicit_entry_points_do_not_bump(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert delta(PUBLIC[name]) == 0, name


def test_family_sampler_does_not_bump():
    from xtim import _xtim_testapi as TA
    # the test API is a separate engine copy: its constructions never reach _xtim's counter, and the FAM
    # sampler builds no exact program (F-X: 0 exact leaves); the production counter stays put
    assert delta(lambda: TA.compile_family_sampler(TEXT, selfcheck=0).sample(10, seed=1)) == 0


def test_auto_is_removed_and_nothing_routes_to_exact():
    """J-b1 F: engine="auto" is gone (exact-out T5: the engine= argument itself is removed — TypeError); the
    sampler on a formerly auto -> exact circuit (a gauge detector) is the twirl engine and builds no exact program."""
    gauge = "RX 0\nT 0\nH 0\nX_ERROR(0.1) 0\nMX 0\nDETECTOR rec[-1]\n"
    with pytest.raises(TypeError, match="engine"):
        xtim.Circuit(gauge).compile_detector_sampler(seed=1, engine="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert delta(lambda: xtim.Circuit(gauge).compile_detector_sampler(seed=1).sample(100)) == 0


def test_detector_determinism_is_not_an_exact_construction():
    assert delta(lambda: _xtim.detector_determinism(TEXT)) == 0
