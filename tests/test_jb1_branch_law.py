"""J-b1 P9b: the carried-branch law (design §2.10; plan §2.1 P9b; fixplan-p9b-xq9 ruling, review C1-C10).

Part 1 (the P9a-1 / P9a-2 fixes, review C5 / C9):
  * ideal_state() is the POST-barrier state (the identity plan's u = 0, z = 0 retained state at the noiseless
    reference), and the frame token hashes it: a producer read collapses the carried wire (Bell-read gate);
  * branches(records) exports the shot's Pauli frame P_rec·W(u) on OUTPUT_QUBITS; the consumer's conditional law
    through the exported frames equals dense (the producer record ↔ the consumer read). MUT-WU-DROP (W(u) = I) and
    MUT-WU-COLS (the output columns permuted) -> red here;
  * E_min is classified against the post-barrier F₀ (re-expressing a dressed branch vector on the ideal frame is OPEN:
    fixplan-p9b-frame).
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

BELL_P = "RX 0\nR 1\nCX 0 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"
BELL_C = "INPUT_QUBITS in 0\nM 0\nDETECTOR rec[-1]\n"
TWO_P = "RX 0 2\nR 1 3\nCX 0 1\nCX 2 3\nM 1\nM 3\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 2\n"
TWO_C = "INPUT_QUBITS in 0 1\nM 0\nDETECTOR rec[-1]\n"


def fam(text, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0, **kw)


def consumer(ideal, text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler_from_ideal_state(ideal, text)


def frames_rows(Fr):
    return np.concatenate([Fr[:, 0, :], Fr[:, 1, :]], 1).astype(np.uint8)


def test_ideal_state_is_post_barrier():
    p = fam(BELL_P)
    c = consumer(p.ideal_state(), BELL_C)
    d = F.unpack(c.sample(2000, seed=1)[0], 1)
    assert not d.any()                                 # the carried wire is |0>: the producer read collapsed it
    rows = p.ideal_state(provenance=False)
    assert rows.token != p.ideal_state().token


@pytest.mark.parametrize("prod,cons,col", [(BELL_P, BELL_C, 0), (TWO_P, TWO_C, 0)])
def test_exported_frames_give_the_conditional_law(prod, cons, col):
    """The producer's record on wire 1 (Bell partner of the carried wire 0) equals the consumer's read of that
    carried wire on every shot, through the exported frames P_rec·W(u) (W(u) = X on the carried wire when the
    coin u = 1: a non-stabilizer frame of the carried |0>)."""
    p = fam(prod)
    d, o, dc, R = p.sample_records(3000, seed=3)
    H, Fr = p.branches(R)
    y = np.array([R.y(i)[col] for i in range(R.num_records)])
    assert 0.4 < y.mean() < 0.6
    nontrivial = int(np.sum(Fr[:, 0, 0]))
    assert nontrivial > 1000                           # the counter: rows whose W(u)|out is not a stabilizer of |0>
    c = consumer(p.ideal_state(), cons)
    got = F.unpack(c.sample(R.num_records, seed=5, input_pauli=frames_rows(Fr))[0], 1)[:, 0]
    assert np.array_equal(got, y)
