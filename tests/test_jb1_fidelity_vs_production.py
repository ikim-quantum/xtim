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
    # RE-PINNED 2026-11-06 (lean D1-S2, owner-signed stream change): the barrier's records are the rotation loop's (kind-2
    # contents, the pivot / Born gauges): was color5 (18724cf7…, 34, 20), rp2 (73e67f4a…, 35, 16); the mean F agrees with the
    # pre-S2 engine within 1 SEM on 3 seeds each (lean/D1/S2/REPORT.md)
    # RE-PINNED 2026-11-07 (lean N1, owner-signed NOISE-stream change: the barrier draws the full channel table on the new
    # chains): was color5 (e2d1476b…, 36, 22), rp2 (15450f4b…, 49, 24); mean F vs the f59959d tree: lean/N1/equivalence.md
    # (signed as declared float values: the tiny rows are float residue of exact zeros, 44 color5 rows read 1+4.4e-16)
    # RE-PINNED 2026-11-07 (lean N2, owner-signed: each noise chain owns a disjoint counter range): was color5 (1b4451ae…,
    # 35, 21), rp2 (34f40424…, 49, 23); mean F over seeds 7-9 vs the 40ecedd tree: color5 0.2036 vs 0.2008 (1.2 sigma),
    # rp2 0.6417 vs 0.6398 (0.8 sigma) (lean/N2/REPORT.md)
    # RE-PINNED 2026-11-08 (lean FID fix 1, owner-signed fidelity pins): the per-shot F hashes are UNCHANGED (bit-identical);
    # only the evaluator's structure stats moved — the rotation contents are interned by their canonical closed diagonal form
    # (smaller expansions): was color5 (35, 21), rp2 (48, 23) (lean/FID/REPORT.md)
    "color5": (os.path.join(DATA, "magna_color5_fid.stim"), 20000,
               ("30d80023d070286117e66759ccb0f8196797409f66bbf8c75c0a09520b41a53e", 34, 19)),
    "rp2": (os.path.join(DATA, "magna_rp2_fid.stim"), 20000,
            ("cdc6e797e9c057f84b0578e8cf5570767ed04e4228b8624800aeb0456a16d1a0", 35, 16)),
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
