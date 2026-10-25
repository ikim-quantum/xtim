"""J-b1 P13 (plan r3.2, r3.1-fix R3-5): the runtime exact-construction counter.

Each exact-engine entry class bumps the counter when built OUTSIDE an explicit-exact scope (raw binding calls),
and the public explicit entry points build it inside one (no bump). The counter has no effect on outputs. The
session-level "counter == accounted over the whole suite" fixture is tests/conftest.py (F); the raw-binding bumps
below are accounted there.
"""
import warnings

import pytest

import xtim
from xtim import _xtim
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
    "SamplerProgram + FramedCircuitShotSampler (CompiledProgram)": lambda: _xtim.CompiledProgram(TEXT, ""),
    "ExactBranchProgram": lambda: _xtim.ExactBranchProgram(TEXT),
    "DEM export": lambda: _xtim.export_dem_text(TEXT, True, False, False, False, [], [], 0.0, 1e9),
    "detector_determinism": lambda: _xtim.detector_determinism(TEXT),
}


@pytest.mark.parametrize("name", sorted(RAW))
def test_each_exact_class_bumps_outside_a_scope(name):
    assert delta(RAW[name]) >= 1, name


@pytest.mark.parametrize("name", sorted(RAW))
def test_each_exact_class_is_silent_inside_a_scope(name):
    def scoped():
        with explicit_exact():
            RAW[name]()
    assert delta(scoped) == 0, name


PUBLIC = {
    "compile_detector_sampler(engine='exact')":
        lambda: xtim.Circuit(TEXT).compile_detector_sampler(seed=1, engine="exact").sample(10),
    "compile_sampler": lambda: xtim.Circuit(TEXT).compile_sampler(seed=1).sample(10),
    "compile_exact_branches": lambda: xtim.Circuit(TEXT).compile_exact_branches(),
    "detector_error_model": lambda: xtim.Circuit(TEXT).detector_error_model(),
    "compile_twirl_sampler (noiseless reference inside)": lambda: xtim.compile_twirl_sampler(TEXT, selfcheck=0),
    "exact reference resolution": lambda: xtim.Circuit(TEXT + "R 2\nM 2\nDETECTOR rec[-1]\n").compile_sampler(seed=2),
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
    """J-b1 F: engine="auto" raises by name; the default engine on a formerly auto -> exact circuit (a gauge detector)
    is the twirl engine and builds no exact program."""
    gauge = "RX 0\nT 0\nH 0\nX_ERROR(0.1) 0\nMX 0\nDETECTOR rec[-1]\n"
    with pytest.raises(ValueError, match="engine='auto' was removed at xtim-engine-13"):
        xtim.Circuit(gauge).compile_detector_sampler(seed=1, engine="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert delta(lambda: xtim.Circuit(gauge).compile_detector_sampler(seed=1).sample(100)) == 0
