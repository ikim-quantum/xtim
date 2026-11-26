"""Parser edge spellings follow stim 1.16 (hunt 2026-11-20 item D): what stim accepts parses (with stim's meaning),
what stim refuses is a parse error naming the line. Each case is checked against stim itself."""
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim

stim = pytest.importorskip("stim")


def _stim_ok(text):
    try:
        stim.Circuit(text)
        return True
    except ValueError:
        return False


ACCEPTED = [
    "M 0\nDETECTOR\n", "M 0\nDETECTOR(1, 2)\n", "M 0\nOBSERVABLE_INCLUDE(0)\n", "M 0\nOBSERVABLE_INCLUDE(3)\n",
    "M 0\nXCZ 0 rec[-1]\n", "M 0\nYCZ 0 rec[-1]\n", "M 0 1\nXCZ 0 rec[-1] 1 rec[-2]\n", "M 0\nXCZ 2 rec[-1] 0 1\n",
    "X_ERROR[a#b](0.1) 0\n", "H[#x] 0 # a comment\n", "X_ERROR[a)b](0.1) 0\n",
    "M\n", "H\n", "CX\n", "CZ\n", "CY\n", "SWAP\n", "MPP\n", "MPP(0.1)\n", "MXX\n", "MZZ(0.1)\n", "SPP\n", "R\n",
    "RX\n", "MX\n", "MR\n", "MRX(0.1)\n", "X_ERROR(0.1)\n", "DEPOLARIZE2(0.1)\n", "PAULI_CHANNEL_1(0.1, 0, 0)\n", "I\n",
    "X_ERROR(1e-3) 0\n", "X_ERROR(.5) 0\n", "X_ERROR(0.) 0\n", "X_ERROR(+0.1) 0\n", "X_ERROR(-0) 0\n",
    "X_ERROR(1E-3) 0\n", "X_ERROR( 0.1 ) 0\n", "X_ERROR(\t0.1) 0\n", "X_ERROR(1.e-1) 0\n", "X_ERROR(1e+0) 0\n",
    "X_ERROR(00.1) 0\n", "X_ERROR() 0\n", "DETECTOR(1,)\n", "DETECTOR(,1)\n", "DETECTOR(1,,2)\n", "DETECTOR(-.5)\n",
    "DETECTOR(1e-400)\n", "DETECTOR(1e-310)\n", "DETECTOR(1e308)\n", "DETECTOR(" + "1" * 63 + ")\n",
    "MPP X0* Z1\n", "MPP X0 * Z1\n", "MPP X0 *Z1\n", "MPP !X0 * !Z1\n", "SPP X0 * Z1\n", "MPP !X0*!Z1\n",
]

REFUSED = [
    "TICK(1)\n", "TICK()\n", "TICK 0\n", "MPP Z0*\n", "MPP X0 *\n", "SPP Z0*\n", "MPP !!X0\n", "MPP X0*!!Z1\n",
    "SPP !!X0\n", "MPP *Z0\n", "MPP X0**Z1\n",
    "X_ERROR(0.5abc) 0\n", "X_ERROR(0x1p-3) 0\n", "DETECTOR(inf)\n", "DETECTOR(nan)\n", "DETECTOR(INF)\n",
    "DETECTOR(1e)\n", "DETECTOR(e5)\n", "DETECTOR(.)\n", "DETECTOR(-)\n", "X_ERROR(+) 0\n", "DETECTOR(1e999)\n",
    "DETECTOR(2e308)\n", "DETECTOR(1 2)\n", "DETECTOR(--1)\n", "DETECTOR(1e+-1)\n", "DETECTOR(1.2.3)\n",
    "DETECTOR(1_0)\n", "DETECTOR(1D)\n", "DETECTOR(" + "1" * 64 + ")\n", "X_ERROR(0.1)0\n", "H[x]0\n",
    "DETECTOR(1)x\n", "M 0\nOBSERVABLE_INCLUDE\n",
]


@pytest.mark.parametrize("text", ACCEPTED)
def test_accepted_as_in_stim(text):
    assert _stim_ok(text)
    info = _xtim.parse_info(text)
    assert info["ok"], info["errors"]


@pytest.mark.parametrize("text", REFUSED)
def test_refused_as_in_stim_naming_the_line(text):
    assert not _stim_ok(text)
    info = _xtim.parse_info("R 0\n" + text)
    assert not info["ok"]
    assert info["errors"][-1][0] == text.count("\n") + 1          # the offending (last) line
    with pytest.raises(xtim.XtimParseError):
        xtim.Circuit("R 0\n" + text)


@pytest.mark.parametrize("arg", ["1e-3", ".5", "0.", "+0.1", "-0", "1E-3", " 0.25 ", "1.e-1", "1e+0", "00.1",
                                 "", "1e-310", "1e-400", "0.1000000000000000055511151231257827", "-1.5e2"])
def test_numbers_parse_to_stims_double(arg):
    text = f"M 0\nDETECTOR({arg}, 7) rec[-1]\n"                 # parses on the old parser too
    want = list(stim.Circuit(text)[1].gate_args_copy())
    assert _xtim.parse_info(text)["detector_coords"] == [want]


def test_empty_parens_and_commas_are_zero_arguments():
    assert _xtim.parse_info("DETECTOR(1,,2)\n")["detector_coords"] == [[1.0, 0.0, 2.0]]
    assert _xtim.parse_info("DETECTOR()\n")["detector_coords"] == [[0.0]]
    assert not _xtim.parse_info("H() 0\n")["ok"]                    # stim: H given 1 parens argument


def _compare(text, shots=4000):
    c, s = xtim.Circuit(text), stim.Circuit(text)
    assert (c.num_detectors, c.num_observables) == (s.num_detectors, s.num_observables)
    assert stim.DetectorErrorModel(c.detector_error_model_text()).approx_equals(s.detector_error_model(), atol=1e-12)
    a = np.asarray(c.compile_detector_sampler(seed=1).sample(shots, append_observables=True)).mean(axis=0)
    b = s.compile_detector_sampler(seed=1).sample(shots, append_observables=True).mean(axis=0)
    assert np.all(np.abs(a - b) < 0.04), (a, b)


def test_empty_detector_and_observable_match_stim():
    _compare("R 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR\nDETECTOR rec[-1]\nDETECTOR(1, 2)\nOBSERVABLE_INCLUDE(2)\n"
             "OBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert xtim.Circuit("M 0\nDETECTOR\n").detector_error_model_text() == "detector D0\n"


def test_zero_target_instructions_are_no_ops():
    _compare("R 0\nM\nH\nCX\nX_ERROR(0.1)\nMPP(0.1)\nMXX\nSPP\nSWAP\nR\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n")
    assert _xtim.parse_info("R 0\nM\nMR\nMPP\nM 0\n")["num_measurements"] == 1


@pytest.mark.parametrize("text", [
    "R 0 1\nX_ERROR(0.3) 0\nM 0\nXCZ 1 rec[-1]\nM 1\nDETECTOR rec[-1] rec[-2]\n",
    "R 0 1\nX_ERROR(0.3) 0\nM 0\nRY 1\nYCZ 1 rec[-1]\nMY 1\nDETECTOR rec[-1] rec[-2]\n",
    "R 0 1 2\nX_ERROR(0.3) 0\nM 0\nRX 1 2\nXCZ 1 rec[-1] 2 1\nMX 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n",
])
def test_xcz_ycz_record_control_is_feedback_as_in_stim(text):
    _compare(text)


def test_xcz_feedback_is_the_swapped_cx():
    a = xtim.Circuit("R 0 1\nX_ERROR(0.3) 0\nM 0\nXCZ 1 rec[-1]\nM 1\nDETECTOR rec[-1]\n")
    b = xtim.Circuit("R 0 1\nX_ERROR(0.3) 0\nM 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]\n")
    assert a.detector_error_model_text() == b.detector_error_model_text()
    assert np.array_equal(np.asarray(a.compile_detector_sampler(seed=3).sample(500)),
                          np.asarray(b.compile_detector_sampler(seed=3).sample(500)))


def test_spaced_mpp_combiner_is_one_product():
    _compare("R 0 1\nH 0\nCX 0 1\nX_ERROR(0.1) 0\nMPP X0* X1 Z0 * Z1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n")
    assert _xtim.parse_info("MPP X0 * Z1 Y2\n")["num_measurements"] == 2


def test_hash_inside_tag_is_not_a_comment():
    info = _xtim.parse_info("R 0\nX_ERROR[a#b](0.1) 0 # real comment\nM 0\nDETECTOR rec[-1]\n")
    assert info["ok"] and info["num_detectors"] == 1


@pytest.mark.parametrize("seed,exc", [(-1, ValueError), (2**64, ValueError), (1.5, TypeError)])
def test_twirl_sampler_seed_is_checked_like_compile_detector_sampler(seed, exc):
    text = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"
    with pytest.raises(exc) as want:
        xtim.Circuit(text).compile_detector_sampler(seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
    first = s.sample(20, seed=5)
    with pytest.raises(exc) as got:
        s.sample(20, seed=seed)
    assert str(got.value) == str(want.value)
    with pytest.raises(exc):
        s.sample_barrier(20, seed=seed)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(first, s.sample(20, seed=5)))

