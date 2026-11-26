"""Canonical single-stage form (docs/xdem.md "Canonical form"): ``xtim.xdem.canonical_report`` and
``detector_error_model_text(format="xdem", require_canonical=True)``. Moved in xdem Task 11 (fix round 1) from the
dc-DEM route's test of the same check; the API refusals and the in-repo agreement leg are in tests/test_xdem_export.py.

t_c = right after the instruction that measures the last detector record. Canonical iff (C1') every observable reads
at least one record measured after t_c and (C2') no noise acts between t_c and the last non-Clifford gate after it.
The default (require_canonical=False) runs no check, and a conventional export (nothing rotates) never checks.
External circuits (clifft-bench originals and our canonical copies, the bt27 scored circuit) are skipped when absent.
"""
import os
from pathlib import Path

import pytest
import stim

import xtim
from xtim import XtimDemError
from xtim.xdem import canonical_report

from _xdem_circuits import external_path

ROOT = Path(__file__).resolve().parents[1]
# External circuits (not in this repository) are looked up by file name in $XTIM_EXTERNAL_CIRCUITS (no default,
# skipped when unset; see _xdem_circuits.external_path). The unchanged clifft-bench originals (third-party, kept out
# of the repo) are looked up in $XTIM_CLIFFT_BENCH_CIRCUITS when it is set, else in $XTIM_EXTERNAL_CIRCUITS.
CB = "XTIM_CLIFFT_BENCH_CIRCUITS" if os.environ.get("XTIM_CLIFFT_BENCH_CIRCUITS") else "XTIM_EXTERNAL_CIRCUITS"
MOD = "XTIM_EXTERNAL_CIRCUITS"


def _report(text):
    return canonical_report(xtim.Circuit(text))


def _need(var, name):
    if not os.environ.get(var):
        pytest.skip(f"external circuits not configured (set {var})")
    path = external_path(name, var)
    if path is None:
        pytest.skip(f"external circuit not present: {name}")
    return Path(path).read_text()


# ── canonical ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_rm15_distillation_is_canonical():
    # in-repo [[15,1,3]] 15-to-1 (feedback CX rec[-k] before the T layer; detectors are the last measurements)
    assert _report((ROOT / "tests/data/distillation_15_1_3.stim").read_text()) == (True, None)


def test_clifft_bench_rm15_is_canonical():
    assert _report(_need(CB, "distillation_15to1_rm15_p1e-3.stim")) == (True, None)


@pytest.mark.parametrize("name", ["bt27_direct_x.mod", "bt81_direct_x.mod", "reed_muller_measured_sampling.mod"])
def test_noiseless_modified_copies_are_canonical(name):
    assert _report(_need(MOD, f"{name}.stim")) == (True, None)


def test_dressed_noiseless_readout_is_canonical_and_bytes_unchanged():
    t = ("RX 0\nR 1\nZ_ERROR(0.1) 0\nX_ERROR(0.1) 1\nT 0\nM 1\nDETECTOR rec[-1]\nT_DAG 0\nMX 0\n"
         "OBSERVABLE_INCLUDE(0) rec[-1]\n")
    c = xtim.Circuit(t)
    assert canonical_report(c) == (True, None)
    assert canonical_report(t) == (True, None)                     # text also accepted
    assert c.detector_error_model_text(format="xdem", require_canonical=True) == \
        c.detector_error_model_text(format="xdem")


def test_bt27_scored_is_canonical():
    """bt27 scored: the 9 observables read the final MX after t_c (C1'); the Z_ERROR(p) right before each MX comes
    after the last non-Clifford gate (the inverse logical CCZs): a classical readout flip, allowed by C2'."""
    assert _report(_need(MOD, "bt27_scored_p1e-3.stim")) == (True, None)


@pytest.mark.parametrize("d", [3, 5])
def test_msc_canonical_copies(d):
    """The copies measure the stabilizers before the dressed Ybar readout: L0's Ybar record is after t_c (its early
    MX records are a classical frame, C1') and the dressing is noiseless (C2')."""
    text = _need(MOD, f"msc_d{d}_inject_cultivate_p1e-3.canon.stim")
    assert _report(text) == (True, None)
    c = xtim.Circuit(text)
    assert c.detector_error_model_text(format="xdem", require_canonical=True) == \
        c.detector_error_model_text(format="xdem")


def test_frame_records_before_t_c_are_allowed():
    assert _report("R 0 1 2\nM 0\nM 1\nDETECTOR rec[-1]\nM 2\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3]\n") == (True, None)


# ── non-canonical, each with its reason ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("d,line_y,line_stab", [(3, 187, 189), (5, 532, 535)])
def test_msc_originals_c1(d, line_y, line_stab):
    ok, why = _report(_need(CB, f"msc_d{d}_inject_cultivate_p1e-3.stim"))
    # L0's latest record is the dressed Ybar readout, measured before the stabilizer MPP's detectors
    assert not ok and why.startswith(f"observable L0 reads no record measured after the last detector record")
    assert f"line {line_stab}): its latest" in why and f"(line {line_y}), is measured before it" in why


def test_all_early_observable_c1():
    ok, why = _report("R 0 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 1\nDETECTOR rec[-1]\n")
    assert not ok and why == ("observable L0 reads no record measured after the last detector record (record #1 of "
                              "D0, line 4): its latest, record #0 (line 2), is measured before it")
    ok, why = _report("R 0 1\nM 1\nDETECTOR rec[-1]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-1]\n")
    assert not ok and why.startswith("observable L0 reads no record (none measured after")


def test_surface_memory_c1_same_instruction():
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=2,
                               after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3)
    ok, why = _report(str(c))
    assert not ok and "is measured by the same instruction" in why


def test_clifft_bench_surface_d7r7_c1():
    ok, why = _report(_need(CB, "pure_surface_d7_r7_p1e-3.stim"))
    assert not ok and "is measured by the same instruction" in why


def test_noise_inside_dressing_c2_and_refusal():
    t = ("R 0 1\nX_ERROR(0.1) 0 1\nM 1\nDETECTOR rec[-1]\nH 0\nT 0\nZ_ERROR(0.01) 0\nT_DAG 0\nMX 0\n"
         "OBSERVABLE_INCLUDE(0) rec[-1]\n")
    ok, why = _report(t)
    assert not ok and why == ("noise Z_ERROR at line 7 acts between the last detector record (record #0 of D0, line 3) "
                              "and the last non-Clifford gate (T_DAG, line 8)")
    # the export refuses it only when its output is extended: a fault between the two T gates rotates the X readout
    x = ("RX 0\nR 1\nM 1\nDETECTOR rec[-1]\nT 0\nX_ERROR(0.1) 0\nT_DAG 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    c = xtim.Circuit(x)
    assert c.detector_error_model_text(format="xdem").startswith("xdem 1")    # default: no check
    with pytest.raises(XtimDemError, match="not in canonical single-stage form: noise X_ERROR at line 6"):
        c.detector_error_model_text(format="xdem", require_canonical=True)
    with pytest.raises(XtimDemError, match="require_canonical"):
        c.detector_error_model(format="xdem", require_canonical=True)
    # the same noise after the dressing is a readout flip: allowed
    t2 = t.replace("Z_ERROR(0.01) 0\nT_DAG 0\n", "T_DAG 0\nZ_ERROR(0.01) 0\n")
    assert _report(t2) == (True, None)


def test_noisy_measurement_c2():
    ok, why = _report("R 0 1 2\nM 1\nDETECTOR rec[-1]\nMX(0.01) 0\nT 2\nM 2\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert not ok and why.startswith("noisy measurement at line 4 (record #1, readout flip p=0.01) acts between")
    assert _report("R 0 1\nM 1\nDETECTOR rec[-1]\nMX(0.01) 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == (True, None)


# ── edge cases ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_no_detectors():
    assert _report("R 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == (True, None)
    assert _report("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == (True, None)
    ok, why = _report("R 0\nX_ERROR(0.1) 0\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert not ok and "t_c = the start of the circuit (no detector reads a record)" in why


def test_no_observables():
    assert _report("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n") == (True, None)
    assert _report("R 0 1\nM 0\nDETECTOR rec[-1]\nX_ERROR(0.1) 1\n") == (True, None)
    ok, why = _report("R 0 1\nM 0\nDETECTOR rec[-1]\nX_ERROR(0.1) 1\nT 1\n")
    assert not ok and why.startswith("noise X_ERROR at line 4 acts between")


def test_repeat_blocks():
    rep = "R 0 1\nREPEAT 3 {\nX_ERROR(0.1) 1\nMR 1\nDETECTOR rec[-1]\n}\n"
    assert _report(rep + "M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == (True, None)
    ok, why = _report(rep + "REPEAT 2 {\nZ_ERROR(0.1) 0\n}\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert not ok and why == ("noise Z_ERROR at line 8 acts between the last detector record (record #2 of D2, line 4) "
                              "and the last non-Clifford gate (T, line 10) (and 1 more violation(s))")
    # an observable read inside an earlier iteration of the detector block
    ok, why = _report("R 0 1\nREPEAT 2 {\nM 1\nDETECTOR rec[-1]\n}\nOBSERVABLE_INCLUDE(0) rec[-2]\n")
    assert not ok and why.endswith("its latest, record #0 (line 3), is measured before it")


def test_feedback():
    # feedback after t_c is not noise; declared (not folded) record sets are checked
    assert _report("R 0 1\nM 1\nDETECTOR rec[-1]\nCX rec[-1] 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == (True, None)
    ok, why = _report("R 0 1\nM 1\nCX rec[-1] 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 0\nDETECTOR rec[-1]\n")
    assert not ok and why.endswith("its latest, record #0 (line 2), is measured before it")


def test_mpp_several_products():
    ok, why = _report("R 0 1 2\nMPP Z0*Z1 Z1*Z2\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert not ok and why.endswith("its latest, record #1 (line 2), is measured by the same instruction")
    assert _report("R 0 1 2\nMPP Z0*Z1\nMPP Z1*Z2\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == (True, None)
    ok, why = _report("R 0 1 2\nMPP Z0*Z1\nMPP(0.01) Z1*Z2 X0*X1*X2\nT 2\nDETECTOR rec[-3]\n"
                      "OBSERVABLE_INCLUDE(0) rec[-2]\n")
    assert not ok and "noisy measurement at line 3" in why


def test_parity_reduced_and_zero_probability():
    assert _report("R 0 1\nM 0\nM 1\nDETECTOR rec[-1]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3] rec[-3]\n") == \
        (True, None)
    assert _report("R 0 1\nM 1\nDETECTOR rec[-1]\nDEPOLARIZE1(0) 0\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == \
        (True, None)
