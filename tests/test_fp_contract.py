"""Exported probabilities are build-flag independent (setup.py _FP_ARGS, -ffp-contract=off): two faults on one
mechanism merge as P + p - 2 P p, and with P=0.049, p=0.0276 the unfused result (0.073895199999999994) differs in the
last ulp from the FMA-fused one (0.073895200000000008, what a -march=native build printed before the flag). Each
pinned text is the portable (no-FMA) build's, so every build must print it."""
import pytest

import xtim
from xtim import _xtim_testapi as ta

STIM = "R 0\nX_ERROR(0.049) 0\nX_ERROR(0.0276) 0\nM 0\nDETECTOR rec[-1]\n"
XDEM = "RX 0\nT 0\nZ_ERROR(0.049) 0\nZ_ERROR(0.0276) 0\nX_ERROR(0.01) 0\nT_DAG 0\nMX 0\nDETECTOR rec[-1]\n"
MERGED = "error(0.073895199999999994) D0\n"


def test_unfused_merge_value():
    assert repr(0.049 + 0.0276 - 2.0 * 0.049 * 0.0276) == "0.0738952"   # CPython never contracts


def test_stim_dem_merge_is_unfused():
    assert xtim.Circuit(STIM).detector_error_model_text() == MERGED


@pytest.mark.parametrize("route", ["oracle", "lean"])
def test_xdem_merge_is_unfused(route):
    r = ta.xdem_export_route(XDEM, route)
    assert r["ok"], r["error"]
    assert r["dem"] == ("xdem 1 detectors 1 observables 0 gauge 1 hidden 0 tc 2\naxis Q0 img D0 gauge g0\n"
                        + MERGED + "error(0.01) x Q0 rot -Q0\n")
