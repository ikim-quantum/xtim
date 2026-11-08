"""J-b1 final review MAJOR-3: an ideal-state consumer's FRESH magic block takes the same P5 even-Clifford ladder-gap
settle as the standalone family build (build-then-settle, inject_carried_), so a T-depth-1 consumer that the
standalone / joint circuit serves is served from a carried state too; a genuine T-depth > 1 consumer is refused
TYPED (OUTSIDE_T_DEPTH_1). MUT-INJ-NOSETTLE (the settle removed) -> red here.

Oracle: the consumer's joint detector law from the |+> producer equals the independent DENSE record oracle
(tests/_record_oracle.py, no xtim code) on the JOINT closed circuit (producer prep ; consumer body), every cell at
5 sigma and no sample in a zero-probability cell. (The exact engine's own bare builder rejects these joint circuits:
it has no ladder settle.)"""
import warnings

import numpy as np
import pytest

import os
import sys

import xtim
from xtim.errors import TwirlClass, XtimTwirlRefusal
from xtim.twirl import compile_twirl_sampler, compile_twirl_sampler_from_ideal_state, input_state_of

sys.path.insert(0, os.path.dirname(__file__))
from _record_oracle import emitted_law, max_z  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

PRODUCER = "R 0\nH 0\nOUTPUT_QUBITS out 0\n"
BODIES = {   # the review's p14 consumers (noise added so the law is not trivial)
    "ttht_fresh": "R 1\nX_ERROR(0.1) 1\nH 1\nT 1\nT 1\nH 1\nT 1\nH 1\nX_ERROR(0.1) 1\nZ_ERROR(0.05) 0\nM 1\n"
                  "DETECTOR rec[-1]\nMX 0\nDETECTOR rec[-1]\n",
    "t_tdag_cancel_fresh": "R 1\nZ_ERROR(0.05) 0\nT_DAG 1\nH 1\nT 1\nT_DAG 1\nZ_ERROR(0.1) 1\nMX 1\nDETECTOR rec[-1]\n"
                           "MX 0\nDETECTOR rec[-1]\n",
}
N = 40000



@pytest.mark.parametrize("name", sorted(BODIES))
@pytest.mark.parametrize("src", ["producer", "input_state_of"])
def test_ladder_gap_consumer_is_served_and_exact(name, src):
    body = BODIES[name]
    ideal = (compile_twirl_sampler(PRODUCER, selfcheck=0).ideal_state() if src == "producer"
             else input_state_of(PRODUCER))
    s = compile_twirl_sampler_from_ideal_state("INPUT_QUBITS in 0\n" + body, ideal_state=ideal, selfcheck=0)
    d, _o = s.sample(N, seed=11)
    k = s.num_detectors
    a = np.unpackbits(np.asarray(d, np.uint8), axis=1, bitorder="little")[:, :k]
    z, impossible = max_z(a, emitted_law("RX 0\n" + body))
    assert z < 5.0 and impossible == 0, (name, src, z, impossible)


def test_t_depth_2_consumer_is_refused_typed():
    body = "R 1\nH 1\nT 1\nH 1\nT 1\nMX 1\nDETECTOR rec[-1]\nMX 0\nDETECTOR rec[-1]\n"
    with pytest.raises(XtimTwirlRefusal) as ei:
        compile_twirl_sampler_from_ideal_state("INPUT_QUBITS in 0\n" + body, ideal_state=input_state_of(PRODUCER),
                                               selfcheck=0)
    assert ei.value.twirl_class == TwirlClass.OUTSIDE_T_DEPTH_1 and ei.value.sub_reason in ("bare", "composed")
