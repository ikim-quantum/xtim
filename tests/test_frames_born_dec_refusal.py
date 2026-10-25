"""fix/frames-born-dec on PRODUCTION (J-b2 J2): Born-class DECISION circuits are served by the record (layout 3).

The legacy sampler refused BarrierBuffer.frames() on these circuits by name unless allow_born_dec=True (the shot's state
is additionally projected onto each Born-class decision operator at its recorded raw outcome, which the sigma frame did
not carry); its refusal and 4c0c26b pins retired with it at J-b2 J2. Production's record carries the decisions: the
gate below checks the retained state against every record column, decisions included.
"""
import numpy as np  # noqa: F401
import pytest

# s2b-adv/probes/adv_frames.py (the review's independent reproduction), verbatim.
CASES = {
    "k0_born_dec": "RX 0\nT 0\nMX 0\nDECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "noisy_born_dec": ("RX 0 1\nT 0\nCX 0 1\nDEPOLARIZE1(0.2) 0 1\nMX 0\nDECISION(0) rec[-1]\nR 9\nM 9\n"
                       "DETECTOR rec[-1]\n"),
    "born_dec_then_gate": ("RX 0 1\nT 0\nT 1\nCZ 0 1\nMX 0\nDECISION(0) rec[-1]\nH 1\nR 9\nM 9\n"
                           "DETECTOR rec[-1]\n"),
    "two_born_dec": ("RX 0 1\nT 0\nT_DAG 1\nMX 0\nMX 1\nDECISION(0) rec[-2]\nDECISION(1) rec[-1]\nR 9\n"
                     "M 9\nDETECTOR rec[-1]\n"),
}

DET_DEC = "RX 0 1\nT 1\nMX 0\nDECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"


# J-b2 J1b / J2: the PRODUCTION gate (plan r2.1 J1b). Production serves frames() on Born-class DECISION circuits (the record
# carries the decisions; no refusal, no allow_born_dec flag): the retained state of every shot is projected onto each
# decision operator at its recorded raw value — <M_c> on materialize(i) == (-1)^{y_c} for EVERY record column
# (decisions included), and emitted XOR y is the constant reference/invert offset — on the same four probe circuits
# and the deterministic-DECISION control. The legacy export's refusal and its 4c0c26b sha pins retire at J2.
@pytest.mark.parametrize("name", sorted(CASES) + ["det_dec"])
def test_born_decision_state_equals_records_public(name):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_s2b_state_equals_bits import check_state_equals_bits_public
    assert check_state_equals_bits_public({**CASES, "det_dec": DET_DEC}[name], shots=200, seed=3) == 200
