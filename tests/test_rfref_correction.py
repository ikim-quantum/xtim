"""RF-ref: decoded scoring with the REGISTER RecordFidelity and the D1 helper (owner rulings 2026-10-22).

``evaluate(buf, correction=c)``: c uint8[shots, 2, n_out] is the decoder's Pauli P_c on OUTPUT_QUBITS; F =
<phi| P_c Tr_R(rho_shot) P_c |phi> — the read wires are traced out, so a fired detector costs nothing by itself (only
the output is compared). ``correction_from_observables(sampler, pred, duals)`` = pred . duals (mod 2) with the user's
per-observable duals on the port (D1). Oracle: tests/_register_oracle.py (dense partial trace, test-only).
"""
import os
import warnings

import numpy as np
import pytest

import xtim
from xtim.record_fidelity import RecordFidelity
from xtim.errors import XtimError
from _register_oracle import bare_rho, register_F

pytestmark = pytest.mark.filterwarnings("ignore")
TOL = 1e-12


def _compile(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(text, selfcheck=0)


# teleportation with its frame CLOSED (CZ / CX feedback), a magic input, an unrelated detector; a repetition code
_TELE = ("RX 0\nR 1 2\nT 0\nH 1\nCX 1 2\nCX 0 1\nH 0\n{err}\nM 0 1\nCZ rec[-2] 2\nCX rec[-1] 2\n"
         "DECISION(0) rec[-2]\nDECISION(1) rec[-1]\nR 3\nX_ERROR({pm}) 3\nM 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 2\n")
_REP = ("R 0 1 2 3 4\nRX 5\nT 5\nCX 5 0\nCX 5 1\nCX 5 2\n{err}\nCX 0 3\nCX 1 3\nCX 1 4\nCX 2 4\nX_ERROR({pm}) 3 4\n"
        "M 3 4\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1 2 5\n")


def _fixtures():
    return [("tele_clean", _TELE.format(err="", pm=0)),
            ("tele_noisy", _TELE.format(err="DEPOLARIZE1(0.2) 0 1 2", pm=0.2)),
            ("rep_meas", _REP.format(err="", pm=0.25)),
            ("rep_data", _REP.format(err="X_ERROR(0.2) 0 1 2", pm=0.1)),
            ("rep_dep", _REP.format(err="DEPOLARIZE1(0.15) 0 1 2 5", pm=0.1))]


@pytest.mark.parametrize("name,txt", _fixtures())
def test_decoded_register_fidelity_vs_dense_oracle(name, txt):
    s = _compile(txt)
    N = 60
    buf = s.sample_barrier(N, seed=5)
    ow = list(s.output_wires())
    rng = np.random.default_rng(1)
    corr = rng.integers(0, 2, (N, 2, len(ow))).astype(np.uint8)
    corr[: N // 3] = 0
    F = np.asarray(RecordFidelity(s).evaluate(buf, correction=corr))
    rhoB = bare_rho(txt, ow)
    worst = max(abs(F[i] - register_F(rhoB, buf, i, ow, corr[i])) for i in range(N))
    assert worst <= TOL, worst
    assert np.any(F > 1e-6)


@pytest.mark.parametrize("name,txt", _fixtures())
def test_zero_correction_equals_no_correction(name, txt):
    s = _compile(txt)
    N = 300
    buf = s.sample_barrier(N, seed=9)
    rf = RecordFidelity(s)
    F0 = np.asarray(rf.evaluate(buf))
    Fz = np.asarray(rf.evaluate(buf, correction=np.zeros((N, 2, len(s.output_wires())), np.uint8)))
    assert np.array_equal(F0, Fz)


def test_measurement_error_costs_nothing():
    """rep_meas: the only faults are readout X on the parity ancillas — every shot scores 1, fired detectors included."""
    s = _compile(_REP.format(err="", pm=0.25))
    buf = s.sample_barrier(200, seed=2)
    assert np.asarray(buf.dets()).any()
    assert np.allclose(RecordFidelity(s).evaluate(buf), 1.0, atol=TOL)


def test_data_error_corrected_by_the_right_pauli():
    """rep_data with a forced X on data qubit 1: 0 uncorrected, 1 with X on qubit 1."""
    s = _compile(_REP.format(err="X_ERROR(1) 1", pm=0))
    buf = s.sample_barrier(4, seed=1)
    c = np.zeros((4, 2, 4), np.uint8)
    rf = RecordFidelity(s)
    assert np.allclose(rf.evaluate(buf, correction=c), 0.0, atol=TOL)
    c[:, 0, 1] = 1
    assert np.allclose(rf.evaluate(buf, correction=c), 1.0, atol=TOL)


def test_correction_shape_refused():
    s = _compile(_REP.format(err="", pm=0.1))
    buf = s.sample_barrier(4, seed=1)
    with pytest.raises(ValueError, match="shape"):
        RecordFidelity(s).evaluate(buf, correction=np.zeros((4, 2, 3), np.uint8))


# ── D1: correction_from_observables(sampler, pred, duals) = pred . duals (mod 2) ──
_REP_OBS = ("R 0 1 2 3 4\nRX 5\nT 5\nCX 5 0\nCX 5 1\nCX 5 2\nX_ERROR(1) 0\nCX 0 3\nCX 1 3\nCX 1 4\nCX 2 4\n"
            "M 3 4\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\nOUTPUT_QUBITS out 0 1 2 5\n")


def test_helper_round_trip_and_decoded_workflow():
    s = _compile(_REP_OBS)
    assert s.num_observables == 1 and len(s.output_wires()) == 4
    N = 8
    buf = s.sample_barrier(N, seed=1)
    pred = np.unpackbits(np.asarray(buf.obs()), axis=1, bitorder="little")[:, :1]   # a perfect decoder
    duals = np.zeros((1, 2, 4), np.uint8)
    duals[0, 0, 0] = 1                                                             # X on data qubit 0
    c = xtim.correction_from_observables(s, pred, duals)
    assert c.shape == (N, 2, 4) and c.dtype == np.uint8
    assert np.array_equal(c, (pred[:, :, None] * duals[None, 0]).astype(np.uint8))
    assert pred.all()
    rf = RecordFidelity(s)
    assert np.allclose(rf.evaluate(buf, correction=c), 1.0, atol=1e-12)
    assert np.allclose(rf.evaluate(buf, correction=np.zeros_like(c)), 0.0)
    rng = np.random.default_rng(4)                                                 # GF(2) linearity
    p2 = rng.integers(0, 2, (50, 1)).astype(bool)
    d2 = rng.integers(0, 2, (1, 2, 4)).astype(np.uint8)
    assert np.array_equal(xtim.correction_from_observables(s, p2, d2), (p2[:, :, None] * d2[None, 0]).astype(np.uint8))


@pytest.mark.parametrize("pred,duals,msg", [
    (np.zeros((3, 2), np.uint8), np.zeros((1, 2, 4), np.uint8), "pred must be"),
    (np.zeros((3, 1), np.uint8), np.zeros((1, 2, 3), np.uint8), "off the port"),
    (np.zeros((3, 1), np.uint8), np.zeros((2, 2, 4), np.uint8), "duals must be"),
    (np.full((3, 1), 2, np.uint8), np.zeros((1, 2, 4), np.uint8), "0 / 1"),
    (np.zeros((3, 1), np.uint8), np.full((1, 2, 4), 3, np.uint8), "0 / 1"),
])
def test_helper_refusals(pred, duals, msg):
    s = _compile(_REP_OBS)
    with pytest.raises(XtimError, match=msg):
        xtim.correction_from_observables(s, pred, duals)
