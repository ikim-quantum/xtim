"""DEM refusal wording: the disjoint-channel refusal uses Stim's message shape, and the
PAULI_EXPECTATION hint is emitted only for a fault that acts non-Pauli on a declared
expectation column (never for the negative-probability channel refusal)."""

import pytest

import xtim
from _orig_examples import ORIG_EXAMPLES, load_orig  # noqa: E402,F401
from xtim.errors import XtimDemError

STIM_LEAD = (
    "Encountered the operation {op} during error analysis, but this operation requires "
    "the `approximate_disjoint_errors` option to be enabled.\n"
    "If you're calling from python, using xtim.Circuit.detector_error_model, you need to "
    "add the argument approximate_disjoint_errors=True."
)

# A Bell pair with XX / ZZ checks and a PAULI_CHANNEL_1 with X and Z but no Y: the record-flip law puts mass on D0 and
# D1 but none on D0 D1, so a product of independent mechanisms would need a negative probability.
NONFACTORABLE = ("R 0 1\nH 0\nCX 0 1\nPAULI_CHANNEL_1(0.001,0,0.001) 0\nMPP X0*X1 Z0*Z1\n"
                 "DETECTOR rec[-1]\nDETECTOR rec[-2]\n")


def _bundled(name):
    with open(xtim.example_path(name)) as f:
        return xtim.Circuit(f.read())


def test_negative_probability_refusal_is_stim_shaped_without_expectation_hint():
    # dem-erasure (signed re-pin): the original cultivation_d5's refusal (an X_ERROR offset coset, "negative") is
    # lifted — the fault is a first-order-exact Clifford record now — so it exports by default. dem-noframe (owner
    # 2026-11-03): the bundled cultivation_d5's DEPOLARIZE2 refusal was caused by its OUTPUT_QUBITS frame columns (its
    # alternatives differed only there); without them it exports, byte-identical to its twin without OUTPUT_QUBITS.
    # The stim-shaped negative-probability refusal is exercised on a minimal non-factorable channel.
    assert "error" in load_orig("cultivation_d5").detector_error_model_text()
    bundled = _bundled("cultivation_d5")
    stripped = "".join(line + "\n" for line in str(bundled).splitlines() if not line.startswith("OUTPUT_QUBITS"))
    assert bundled.detector_error_model_text() == xtim.Circuit(stripped).detector_error_model_text()
    c = xtim.Circuit(NONFACTORABLE)
    with pytest.raises(XtimDemError) as ei:
        c.detector_error_model()
    text = str(ei.value)
    assert text.startswith(STIM_LEAD.format(op="PAULI_CHANNEL_1"))
    # xtim's own detail is kept after the Stim-shaped lead.
    assert "deferred stream index" in text
    assert "exact DEM conversion has a negative probability" in text
    assert "PAULI_EXPECTATION" not in text
    assert "include_expectations" not in text
    assert "approximate_disjoint_errors=True" in ei.value.hint
    # include_expectations=False does not lift it (no expectation column is involved).
    with pytest.raises(XtimDemError) as ei2:
        c.detector_error_model(include_expectations=False)
    assert str(ei2.value).startswith(STIM_LEAD.format(op="PAULI_CHANNEL_1"))


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
