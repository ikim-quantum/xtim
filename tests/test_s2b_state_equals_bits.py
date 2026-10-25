"""S2b gate S-1: the retained state equals the emitted bits [C U-1].

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

For every record-route barrier shot: the record's RAW values y (BarrierBuffer.record_outcomes, the
Theorem-1 rows of the shot's record (plan, u, z) plus t(P)) are

  * the emitted bits up to the circuit's constant emission offsets (detector / observable reference,
    record inverts): emitted XOR y is the SAME vector on every shot (no readout-flip coins, no input
    frames on these fixtures), and
  * the retained state's record values: materialize(i).pauli_expectation(M_c) == (-1)^{y_c} to 1e-12
    for EVERY record column c (pivots, direction introducers and every dependent record).
"""
import os
import re
import sys

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from test_s2b_record_law import _family  # noqa: E402
from test_s2b_record_table import FIXTURES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def _unpack(a, k):
    a = np.asarray(a, np.uint8)
    if k == 0:
        return np.zeros((a.shape[0], 0), np.uint8)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :k]


# ── J-b2 J1b: the PRODUCTION leg (the public sampler; plan r2.1 J1b). The same law against the same oracle (the
# retained state's <M_c> == (-1)^{y_c} to 1e-12, emitted XOR y a constant offset), on the fixtures and the fuzz family.
# Production serves every shot by the law (no exact shot), and its barrier serves CH/PPR-class circuits too (the
# kind-2 route), so the production leg checks every shot of every family member.
from _public import public_sampler, rec_order  # noqa: E402


def check_state_equals_bits_public(text, shots=300, seed=5):
    s = public_sampler(text)
    b = s.sample_barrier(shots, seed=seed)
    nd, no, nde = s.num_detectors, s.num_observables, s.num_decisions
    em = [_unpack(b.dets(), nd), _unpack(b.obs(), no)]
    if nde:
        em.append(_unpack(b.decisions(), nde))
    E = np.concatenate(em, 1)
    ops = s.record_operators()
    base = {0: 0, 1: nd, 2: nd + no}
    emitted_index = [base[k] + i for k, i in rec_order(text)]
    off = None
    readout_flips = re.search(r"\bM[XYR]*\(", text) is not None   # rf coins: the offset is per shot
    for i in range(shots):
        y = np.asarray(b.record_outcomes(i))
        assert len(y) == len(ops)
        d = E[i, emitted_index] ^ y
        if off is None:
            off = d
        assert readout_flips or np.array_equal(d, off), (i, d, off)
        st = b.materialize(i)
        for c, (xs, zs, ph) in enumerate(ops):
            v = st.pauli_expectation(list(xs), list(zs), int(ph))
            assert abs(v - (1.0 - 2.0 * y[c])) < 1e-12, (i, c, v, int(y[c]))
    return shots


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_state_equals_bits_public(name):
    assert check_state_equals_bits_public(FIXTURES[name]) > 0


@pytest.mark.parametrize("name,text", _family()[::3], ids=lambda v: v if isinstance(v, str) and len(v) < 8 else "")
def test_family_state_equals_bits_public(name, text):
    assert check_state_equals_bits_public(text, shots=120, seed=int(name[1:]) + 3) > 0
