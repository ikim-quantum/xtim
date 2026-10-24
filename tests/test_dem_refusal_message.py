"""DEM refusal wording: the disjoint-channel refusal uses Stim's message shape, and the
PAULI_EXPECTATION hint is emitted only for a fault that acts non-Pauli on a declared
expectation column (never for the negative-probability channel refusal)."""

import pytest

import xtim
from _orig_examples import ORIG_EXAMPLES, load_orig  # noqa: E402,F401
from xtim.errors import XtimDemError

STIM_LEAD = (
    "Encountered the operation X_ERROR during error analysis, but this operation requires "
    "the `approximate_disjoint_errors` option to be enabled.\n"
    "If you're calling from python, using xtim.Circuit.detector_error_model, you need to "
    "add the argument approximate_disjoint_errors=True."
)


def test_cultivation_d5_refusal_is_stim_shaped_without_expectation_hint():
    c = load_orig("cultivation_d5")
    with pytest.raises(XtimDemError) as ei:
        c.detector_error_model()
    text = str(ei.value)
    assert text.startswith(STIM_LEAD)
    # xtim's own detail is kept after the Stim-shaped lead.
    assert "deferred stream index" in text
    assert "exact DEM conversion has a negative probability" in text
    assert "PAULI_EXPECTATION" not in text
    assert "include_expectations" not in text
    assert "approximate_disjoint_errors=True" in ei.value.hint
    # include_expectations=False does not lift it (no expectation column is involved).
    with pytest.raises(XtimDemError) as ei2:
        c.detector_error_model(include_expectations=False)
    assert str(ei2.value).startswith(STIM_LEAD)


def test_cultivation_d5_exports_with_approximate_disjoint_errors():
    dem = load_orig("cultivation_d5").detector_error_model_text(
        approximate_disjoint_errors=True)
    assert "error(" in dem


def test_expectation_dressing_refusal_keeps_pauli_expectation_hint():
    # The engine's one refusal for a fault acting non-Pauli on a declared expectation
    # column (dem_export.cpp, the >16-randomized-detector accepted-branch guard).
    engine = ("X_ERROR at deferred stream index 3: a single fault randomizes 17 > 16 "
              "detectors while acting non-Pauli on a logical expectation; the accepted "
              "branch was not verified (refusing)")
    e = XtimDemError(engine)
    assert "dresses a declared PAULI_EXPECTATION column beyond a sign" in e.hint
    assert "include_expectations=False" in e.hint
    assert str(e).startswith("not DEM-expressible: X_ERROR at deferred stream index 3")


def test_unrecognized_refusal_gets_no_expectation_hint():
    e = XtimDemError("two terminal reads share a wire; the DEM determinism analysis "
                     "does not cover this")
    assert e.hint is None
    assert "PAULI_EXPECTATION" not in str(e)
