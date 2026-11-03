"""Owner-pinned per-shot fidelity numbers of the public state route — the REGISTER RecordFidelity (RF-ref strict).

Re-pinned to the SIGNED table OD-2026-10-22-rfref-register-pins (rfref/rulings.md, owner 14:52):
  * color5 (Magna Color d=5): 7c8fa8cc... -> 8716b444... (mean F 0.198), max_virtual_qubits 36;
  * rp2 (Magna RP2 d=5): f30af08c... -> 73e67f4a... (mean F 0.646), max_virtual_qubits 46;
  * fac_ref (the private CCZ-factory fixture) and the CCZ decoded golden: refused, RF_OUTPUT_NOT_STRICT (e = 72, c = 9);
  * d5b_ref (cultivation_d5) and d15_ref (distillation_15_1_3): refused, RF_NO_OUTPUT.
The per-shot F bytes are SHA-256 of float64 at seed 7 (keep_memo=True), with the stat the evaluator reports.
"""
import hashlib
import os
import warnings

import numpy as np
import pytest

import xtim
from xtim.record_fidelity import RecordFidelity
from _private import private_fixture, private_path

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
FAC = str(private_path("ccz_factory_a_out.stim"))   # private (the tests skip when absent)

SERVED = {   # name: (circuit, shots, (sha256 of the per-shot F, max_r, max_kappa_prime)) — OD-2026-10-23-s8-pins
    # (signed as declared float values: the tiny rows are float residue of exact zeros, 44 color5 rows read 1+4.4e-16)
    "color5": (os.path.join(DATA, "magna_color5_fid.stim"), 20000,
               ("18724cf716dd8a163eeb26f2bff8d9620d09c23d13308e4e025b7d2c9118852a", 34, 20)),
    "rp2": (os.path.join(DATA, "magna_rp2_fid.stim"), 20000,
            ("73e67f4a8408366bb94d2444d5078ce5914c6b459a7215aa78d587c608e3845e", 35, 16)),
}
REFUSED = {  # name: (circuit, the named refusal)
    "fac_ref": (FAC, r"RF_OUTPUT_NOT_STRICT.*c = 9, e = 72"),
    "d5b_ref": (os.path.join(ROOT, "tests", "data", "cultivation_d5.stim"), "RF_NO_OUTPUT"),
    "d15_ref": (os.path.join(ROOT, "tests", "data", "distillation_15_1_3.stim"), "RF_NO_OUTPUT"),
}


def _compile(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(open(path).read(), selfcheck=0)


@pytest.mark.parametrize("name", sorted(SERVED))
def test_per_shot_register_fidelity_pinned(name):
    circ, N, pin = SERVED[name]
    sf = _compile(circ)
    rf = RecordFidelity(sf)
    Ff = np.asarray(rf.evaluate(sf.sample_barrier(N, seed=7), keep_memo=True), np.float64)
    assert np.any(Ff > 0) and np.any(Ff != Ff[0])                       # not vacuous
    st = rf.stats()
    assert (hashlib.sha256(Ff.tobytes()).hexdigest(), st["max_r"], st["max_kappa_prime"]) == pin


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_pinned_refusals(name):
    circ, pat = REFUSED[name]
    if circ == FAC:
        private_fixture("ccz_factory_a_out.stim")
    with pytest.raises(ValueError, match=pat):
        RecordFidelity(_compile(circ))


def test_ccz_decoded_golden_is_refused():
    """The CCZ decoded-fidelity fixture (private ccz_factory_a_out_target.json) scored the factory's output against the
    owner's Clifford target; the register fidelity refuses the circuit by name (its output depends on unrecorded
    reads), so the golden is retired (signed)."""
    with pytest.raises(ValueError, match=r"RF_OUTPUT_NOT_STRICT.*measurements \["):
        RecordFidelity(_compile(str(private_fixture("ccz_factory_a_out.stim"))))
