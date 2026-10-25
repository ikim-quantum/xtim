"""The STATE ROUTE (direct fidelity estimation) on circuits that declare OUTPUT_QUBITS: exact per-shot
fidelity vs the noiseless reference on one sampled run of the circuit as given (RecordFidelity).
Until EXP-fix (2026-10-21) these facts were packaged by diagnose()'s output-state report; diagnose (the
beta-extraction workflow) was removed, and the same checks now go through the public state-route API."""
import numpy as np
import pytest

import xtim
from _orig_examples import ORIG_EXAMPLES, load_orig  # noqa: E402,F401
from xtim._exact_scope import explicit_exact

_OQ15 = "\nOUTPUT_QUBITS out " + " ".join(map(str, range(15))) + "\n"


@pytest.fixture(scope="module")
def dist():
    return load_orig("distillation_15_1_3").text


def _route(text, shots, seed):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s.sample_barrier(shots, seed=seed)
    F = np.asarray(xtim.RecordFidelity(s).evaluate(buf), np.float64)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :s.num_detectors]
    return s, F, dets


def test_state_route_is_exact_and_postselectable(dist):
    text = dist + _OQ15
    s, F, dets = _route(text, 512, 3)
    assert F.shape == (512,) and ((F >= -1e-12) & (F <= 1 + 1e-12)).all()
    with explicit_exact():
        det = list(xtim._xtim.detector_determinism(text)["deterministic_detectors"])
    keep = ~dets[:, det].astype(bool).any(axis=1)
    assert 0 < int(keep.sum()) <= 512
    # deterministic per seed
    _, F2, _ = _route(text, 512, 3)
    assert np.array_equal(F, F2)


def test_noiseless_output_has_unit_fidelity(dist):
    _, F, _ = _route(xtim.Circuit(dist + _OQ15).without_noise().text, 256, 0)
    assert F.mean() == pytest.approx(1.0, abs=1e-12)


def test_zero_channel_circuit_takes_the_state_route():
    """RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness: a circuit with no detector / decision
    channel was REFUSED by the legacy sampler ("no deterministic detector channels") and fell back to the
    PAULI_EXPECTATION route. The FAM sampler relaxes the empty gate (J-a), so the state route serves it; the circuit
    is noiseless, so every shot's fidelity is exactly 1. (EXP-fix: checked through RecordFidelity directly.)"""
    _, F, _ = _route("R 0\nH 0\nT 0\nPAULI_EXPECTATION(0) X0\nOUTPUT_QUBITS out 0\n", 64, 0)
    assert F.mean() == pytest.approx(1.0, abs=1e-12)


def test_state_route_refusal_is_named_by_class():
    """A circuit the twirl engine refuses (NONCOMMUTING_MAGIC; the exact engine accepts it) is refused by its
    TwirlClass name; there is no exact fallback."""
    cs_witness = ("RX 0 1\nT 1\nH 1\nCZ 0 1\nCS_DAG 0 1\nDEPOLARIZE1(0.1) 0\nCX 0 1\nCS_DAG 1 0\nH 1\n"
                  "DEPOLARIZE1(0.1) 1\nS_DAG 1\nMX 0 1\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n"
                  "R 20\nOUTPUT_QUBITS out 20\n")   # tests/test_jb1_engine_selection.py CS_WITNESS + an output wire
    with pytest.raises(xtim.XtimTwirlRefusal) as ei:
        xtim.compile_twirl_sampler(cs_witness)
    assert ei.value.twirl_class.name == "NONCOMMUTING_MAGIC"
