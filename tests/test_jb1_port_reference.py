"""J-b1 P10: Addendum A1 on the FAM sampler (test API) — INPUT_QUBITS and ideal-state compiles are REFERENCE-RELATIVE.

Owner decision O1 / A1 §2: the reference of a port circuit is the noiseless run from the ideal state (the bare),
frame-free; DETECTOR / OBSERVABLE columns are relative to it (as on every other circuit), DECISION columns stay raw.
This inverts J-a N1's ABSOLUTE port pin on FAM; production keeps its absolute contract until F (TS:486, :597).

Gates (plan §2.1 P10):
  * the inverted pin: FAM columns == engine="exact" columns on INPUT_QUBITS circuits, the A1 repro included (its
    reference is nonzero, so the identity is not vacuous); with frames, the FAM law == DENSE's reference-relative
    law with the frame on the port;
  * ideal-state compiles (the producer's ideal_state, and a rows-only ideal state rebuilt from production's
    _input_state_of — the prep-splice reroute: every carried state through injection as branch 0): the carried-|1>
    example reports 0 at p = 0, its law == exact on the closed circuit, and an X frame flips it;
  * a Y input frame through CH (A1 §3): D0 = 0 on every shot, = exact;
  * a Born observable on a port circuit reports relative to its noiseless reference (family_port_born_ref);
  * DECISION columns stay raw;
  * MUT-REF-ABS (the port columns kept ABSOLUTE) -> red here.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from _record_oracle import emitted_columns, emitted_law, reference_records  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
N = 20_000

A1_REPRO = "INPUT_QUBITS in 0\nR 1\nX 1\nM 1\nDETECTOR rec[-1]\n"
# nonzero references on a deterministic detector, a Born observable (the T|+> MX read), a gauge-ish extra and a raw
# DECISION, with noise on the port and off it
BORN_PORT = ("INPUT_QUBITS in 4\nR 1\nX 1\nRX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX 0\nM 4\nX 2\nM 2\n"
             "DETECTOR rec[-4]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\nDETECTOR rec[-2]\n")
DEC_PORT = "INPUT_QUBITS in 3\nR 0\nX 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]\nDECISION(0) rec[-1]\n"
CASES = {"a1_repro": A1_REPRO, "port": F.PORT, "born_port": BORN_PORT, "dec_port": DEC_PORT}

PRODUCER_1 = "R 0\nX 0\nOUTPUT_QUBITS out 0\n"                       # the carried |1> (A1 item 1)
CONSUMER_1 = "INPUT_QUBITS in 0\nR 1\nX_ERROR(0.1) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
CLOSED_1 = "R 0\nX 0\nR 1\nX_ERROR(0.1) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"

Y_CH = "INPUT_QUBITS in 1\nRX 0\nCH 0 1\nCH 0 1\nMX 0\nDETECTOR rec[-1]\n"
Y_CH_CLOSED = "RX 0\nR 1\nY 1\nCH 0 1\nCH 0 1\nMX 0\nDETECTOR rec[-1]\n"


def fam(text, p_factor=1.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0, p_factor=p_factor)


def cols(s, n, seed, frames=None):
    d, o, dc = s.sample(n, seed=seed, input_pauli=frames)[:3]
    return np.concatenate([F.unpack(d, s.num_detectors), F.unpack(o, s.num_observables),
                           F.unpack(dc, s.num_decisions)], 1)


def exact_cols(text, n, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = xtim.Circuit(text)
        d, o = c.compile_detector_sampler(seed=seed, engine="exact").sample(n, separate_observables=True)
    return np.concatenate([np.asarray(d, np.uint8), np.asarray(o, np.uint8)], 1)


def nonzero_reference(text):
    ref = reference_records(text)
    return [sum(ref[i] for i in recs) & 1 for kind, _, recs in emitted_columns(text) if kind != "decision"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_inverted_pin_text_port_columns_equal_exact(name):
    text = CASES[name]
    s = fam(text)
    m = s.num_detectors + s.num_observables
    F.check_identity("pin-" + name, lambda n, sd: cols(s, n, sd)[:, :m], lambda n, sd: exact_cols(text, n, sd), N)
    if name != "dec_port":
        assert any(nonzero_reference(text)), name           # the identity is not vacuous


@pytest.mark.parametrize("name", sorted(CASES))
def test_inverted_pin_noiseless_is_zero(name):
    """At p = 0 every DETECTOR / OBSERVABLE column of a port circuit is 0 (reference-relative); a DECISION column
    is the raw recorded parity."""
    text = CASES[name]
    s = fam(text, p_factor=0.0)
    M = cols(s, 500, 3)
    m = s.num_detectors + s.num_observables
    born = name == "born_port"                              # the Born observable is random noiselessly
    det = M[:, s.channel_report()["deterministic_detectors"]]   # gauge / extras are random noiselessly
    assert det.shape[1] >= 1 and not det.any(), name
    if not born:
        assert not M[:, s.num_detectors:m].any(), name
    if s.num_decisions:
        assert M[:, m:].all(), name                         # DEC_PORT: the raw record (X 0, M 0) is 1


@pytest.mark.parametrize("P", ["X", "Y", "Z"])
def test_inverted_pin_with_frames_equals_dense(P):
    text = F.PORT
    s = fam(text)
    port = 5
    fr = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}[P]
    law = emitted_law(text, frame={port: P})
    nq = 1

    def draw(n, seed):
        rows = np.zeros((n, 2 * nq), np.uint8)
        rows[:, 0], rows[:, nq] = fr
        return cols(s, n, seed, rows)
    F.check_law("pin-frame-" + P, draw, law, N)


def _ideal_states():
    p = fam(PRODUCER_1)
    yield "producer", p.ideal_state()
    st = xtim.twirl._input_state_of(PRODUCER_1)
    yield "rows_only", TA.ideal_state_from_frames(st.state.branch_frames(), list(st.carried_wires))


@pytest.mark.parametrize("which", ["producer", "rows_only"])
def test_ideal_state_carried_one_reports_zero(which):
    ideal = dict(_ideal_states())[which]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c0 = TA.compile_family_sampler_from_ideal_state(ideal, CONSUMER_1, p_factor=0.0)
        c = TA.compile_family_sampler_from_ideal_state(ideal, CONSUMER_1)
    assert not cols(c0, 400, 1).any()                       # the carried |1> read in Z reports 0 (A1 item 1)
    F.check_identity("ideal-" + which, lambda n, sd: cols(c, n, sd), lambda n, sd: exact_cols(CLOSED_1, n, sd), N)
    xr = np.zeros((2000, 2), np.uint8)
    xr[:, 0] = 1                                            # an X frame on the port: the deviation as a frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = TA.compile_family_sampler_from_ideal_state(ideal, CONSUMER_1, p_factor=0.0)
    assert cols(cf, 2000, 5, xr)[:, 0].all()


def test_ideal_state_from_frames_round_trip():
    """The rows-only ideal state is the production state verbatim (frames export -> rows -> frames export)."""
    st = xtim.twirl._input_state_of("RX 0\nT 0\nH 1\nS 1\nOUTPUT_QUBITS out 0 1\n")
    a = st.state.branch_frames()
    i = TA.ideal_state_from_frames(a, list(st.carried_wires))
    assert not i.provenance and i.wires == list(st.carried_wires)
    p = fam("RX 0\nT 0\nH 1\nS 1\nOUTPUT_QUBITS out 0 1\n")
    assert p.ideal_state(provenance=False).token == i.token        # the same rows -> the same rows-only token


def test_y_frame_through_ch():
    """A1 §3: a Y input frame through CH·CH is a Pauli (CH Y1 CH† = Z0 Y1); D0 = 0 on every shot, = exact."""
    s = fam(Y_CH)
    rows = np.zeros((3000, 2), np.uint8)
    rows[:, 0] = rows[:, 1] = 1
    assert not cols(s, 3000, 9, rows).any()
    assert not exact_cols(Y_CH_CLOSED, 3000, 9).any()
