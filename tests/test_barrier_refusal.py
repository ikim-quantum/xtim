"""E1 (fix/barrier-refuse), RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness.

History: an observable whose operator is not IN_GROUP had no exact record channel in the LEGACY twirl product; the
legacy sampler refused it by name in sample() and BarrierBuffer.obs() (E1), and frame_dressing()['obs'] named the
refusal (E1 r3). At the J-b1 flip production IS the FAM sampler: such observables are served as exact records
(record_report classes 'coin' / 'born'), so every former refusal becomes an exactness check against the exact
engine (engine="exact", the explicit oracle). frame_dressing and skip_refused_observables are retired (plan §1.4 C;
tests/test_jb1_removed_names.py). The legacy refusal itself stays pinned on the test API's legacy sampler
(tests/test_s2b_*), the comparison object until J-b2.
"""
import gc

import numpy as np
import pytest

import xtim
from xtim.port import Consume
from xtim.port import compile as port_compile
from xtim.record_fidelity import RecordFidelity

_F1 = "RX 3\nT 3\nRX 4\nT_DAG 4\nX_ERROR(0.5) 2\nCS_DAG 4 3\nCCZ 4 3 2\nR 10\nM 10\nDETECTOR rec[-1]\n"
_F2 = ("RX 3\nT 3\nRX 4\nT_DAG 4\nX_ERROR(0.5) 2\nCS_DAG 4 3\nCCZ 4 3 2\nR 5 6\nH 5\nCX 5 6\nRX 7\nT 7\n"
       "R 8\nX_ERROR(0.5) 8\nCCZ 7 5 8\nR 9\nH 9\nCX 9 5\nCX 9 6\nH 9\nM 9\nDETECTOR rec[-1]\n")
# ndhaz F1x / F2x (<S>/ndhaz): observables the twirl sampler refuses
REFUSED = {
    "F1x": _F1 + "MX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "F2x": (_F2 + "MX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\nMX 7\nOBSERVABLE_INCLUDE(1) rec[-1]\nMX 3\n"
            "OBSERVABLE_INCLUDE(2) rec[-1]\n"),
}
# F1x with OUTPUT_QUBITS (state / record-fidelity consumers)
F1X_OUT = _F1 + "MX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\nOUTPUT_QUBITS out 2 3\n"
# controls: the adv-A ANTI trio (Born/ANTI decisions, no observable) and an IN_GROUP Born observable
CONTROLS = {
    "q3dec_noisyT": ("RX 0\nR 1 2\nX_ERROR(0.2) 0\nT 0\nH 1\nCX 1 2\nCX 0 1\nDEPOLARIZE1(0.05) 1 2\n"
                     "MX 1 2 0\nDETECTOR rec[-3] rec[-2]\nDECISION(0) rec[-3]\nDECISION(1) rec[-1]\n"),
    "bell2_3dec": ("RX 0\nR 1 2 3 4\nX_ERROR(0.25) 0\nT 0\nH 1 3\nCX 1 2 3 4\nCX 0 1\nCZ 0 3\n"
                   "DEPOLARIZE1(0.06) 0 1 2 3 4\nMX 1 2 3 4 0\nDETECTOR rec[-5] rec[-4]\nDECISION(0) rec[-5]\n"
                   "DECISION(1) rec[-3]\nDECISION(2) rec[-1]\n"),
    "teleport_dec": ("RX 0\nR 1 2 3\nX_ERROR(0.2) 0\nT 0\nH 1\nCX 1 2\nCX 0 1\nH 0\nDEPOLARIZE1(0.05) 0 1 2 3\n"
                     "M 0 1\nMX 2\nM 3\nDETECTOR rec[-1]\nDECISION(0) rec[-4]\nDECISION(1) rec[-3]\n"
                     "DECISION(2) rec[-2]\n"),
    "ccz_obs": ("RX 0 1 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nRX 3\nT 3\nCX 3 1\nMX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"),
}
N, SEED = 400, 7
NJ = 40_000


def _dets_obs(t):
    """sample() returns (dets, obs) or a TwirlBatch (with decisions)."""
    return (np.asarray(t.dets), np.asarray(t.obs)) if hasattr(t, "dets") else (np.asarray(t[0]), np.asarray(t[1]))


def _joint(dets, obs, nd, no):
    d = np.unpackbits(np.asarray(dets, np.uint8), axis=1, bitorder="little")[:, :nd]
    o = np.unpackbits(np.asarray(obs, np.uint8), axis=1, bitorder="little")[:, :no]
    bits = np.concatenate([d, o], axis=1).astype(np.int64)
    key = (bits << np.arange(bits.shape[1])).sum(1)
    return np.bincount(key, minlength=1 << bits.shape[1]) / len(key)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_formerly_refused_observables_are_served_exactly(name):
    """The joint (detector, observable) law of the served twirl sampler equals the exact engine's, every cell within
    5 sigma (two independent samples)."""
    text = REFUSED[name]
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    classes = s.record_report()["classes"]["observables"]
    assert len(classes) == s.num_observables and set(classes.values()) <= {"coin", "born", "deterministic"}
    d, o = _dets_obs(s.sample(NJ, seed=SEED))
    ex = xtim.Circuit(text).compile_detector_sampler(seed=SEED, engine="exact")
    ed, eo = ex.sample(NJ, separate_observables=True, bit_packed=True)
    nd, no = s.num_detectors, s.num_observables
    p, q = _joint(d, o, nd, no), _joint(ed, eo, nd, no)
    sig = np.sqrt((p * (1 - p) + q * (1 - q)) / NJ) + 1e-12
    assert (np.abs(p - q) <= 5 * sig + 1e-9).all(), (p, q)


@pytest.mark.parametrize("name", sorted(REFUSED) + sorted(CONTROLS))
def test_barrier_obs_equals_sample_obs(name):
    """BarrierBuffer.obs() (formerly the refusal point) == sample()'s observable columns at the same seed; dets and
    decisions likewise."""
    text = REFUSED.get(name) or CONTROLS[name]
    s = xtim.compile_twirl_sampler(text)
    b = s.sample_barrier(N, seed=SEED)
    dets, obs = _dets_obs(s.sample(N, seed=SEED))
    assert np.array_equal(np.asarray(b.obs()), obs)
    assert np.array_equal(np.asarray(b.dets()), dets)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_lifetime_obs_after_the_python_sampler_reference_is_dropped(name):
    """n2 (lifetime): dropping the caller's Python reference does not free the sampler (keep_alive and the buffer's
    shared_ptr); obs() keeps its bytes on that path."""
    text = REFUSED[name]
    want = np.asarray(xtim.compile_twirl_sampler(text).sample_barrier(N, seed=SEED).obs())
    s = xtim.compile_twirl_sampler(text)
    b = s.sample_barrier(N, seed=SEED)
    del s
    gc.collect()
    assert np.array_equal(np.asarray(b.obs()), want)


def test_record_fidelity_on_formerly_refused_observable_circuit():
    """The F1x circuit with OUTPUT_QUBITS: dets equal sample()'s (the legacy 4c0c26b byte pin is a legacy-sampler pin,
    retired with the flip); its output depends on the unfed-back MX 4 (the observable's measurement), so the register
    fidelity refuses it by name, naming that measurement (strict contract)."""
    s = xtim.compile_twirl_sampler(F1X_OUT)
    b = s.sample_barrier(50, seed=3)
    assert np.array_equal(np.asarray(b.dets()), _dets_obs(s.sample(50, seed=3))[0])
    with pytest.raises(ValueError, match=r"RF_OUTPUT_NOT_STRICT.*measurements \[1\]"):
        RecordFidelity(s)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_port_record_route_on_formerly_refused_observable_circuit(name):
    text = REFUSED[name]
    seg = port_compile(text, Consume(dets=True, decisions=True, groups=True))
    r = seg.run(N, SEED)
    s = xtim.compile_twirl_sampler(text)
    b = s.sample_barrier(N, seed=SEED)
    want = np.unpackbits(np.asarray(b.dets(), np.uint8), axis=1, bitorder="little")[:, :s.num_detectors].astype(bool)
    assert np.array_equal(np.asarray(r.dets), want)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_detector_sampler_twirl_engine_serves(name):
    """Formerly: engine='twirl' refused and engine='auto' routed to exact. engine='auto' is removed
    (test_jb1_removed_names); the default twirl engine now serves these circuits."""
    c = xtim.Circuit(REFUSED[name])
    out = c.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    assert np.asarray(out[1]).shape[0] == 64
