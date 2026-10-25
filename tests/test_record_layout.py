"""S2b: ``record_layout()`` version 2 — the outcome segments of every shot's record (design Q4/Q5).

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

A record-route shot's outcomes (``coins(i)``) are (u, z) of its plan's record table: kind ``coin``
(the rho pivot records' fair coins, offset 0) then kind ``born`` (the kappa_rec record-trie
outcomes, offset rho). An exact-fallback shot's outcomes are its RAW record values y (kind
``exact_y``, width n_records). Checked per shot, with and without a Born observable / Born decisions:
  * widths: rho / kappa_rec equal the shot's plan table (plan_structure), the segments tile the
    record, exact shots have width n_records;
  * the record determines the emitted bits: record_outcomes(i) (Theorem 1 rows of (u, z) plus t(P),
    or an exact shot's y) gives every decision's raw outcome (emitted bit XOR its invert) and the
    observable's up to one constant (invert XOR reference) — the fixtures carry no readout flip and
    no frame;
  * the version-1 keys and coin_layout() are gone (loud).
"""
import numpy as np
import pytest

import xtim
import xtim._xtim as _x

# adv-C p2/p9 fixture: a detector, a Born observable (MY 0) and a Born decision (MX 2).
P2 = ("RX 0\nT 0\nRX 2\nT 2\nX_ERROR(0.2) 1\nM 1\nMY 0\nMX 2\nDETECTOR rec[-3]\n"
      "OBSERVABLE_INCLUDE(0) rec[-2]\nDECISION(0) rec[-1]\n")
DEC_ONLY = ("RX 0\nT 0\nRX 2\nT 2\nX_ERROR(0.2) 1\nM 1\nMY 0\nMX 2\nDETECTOR rec[-3]\n"
            "DECISION(0) rec[-1]\n")
_M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
       "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
       "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
_DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
M3_OBS_2DEC = (_M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET + "MX 0\nDECISION(0) rec[-1]\n"
               "MY 1\nDECISION(1) rec[-1]\nOUTPUT_QUBITS out 2\n")
M3_2DEC = (_M3 + _DET + "MX 0\nDECISION(0) rec[-1]\nMY 1\nDECISION(1) rec[-1]\nOUTPUT_QUBITS out 2\n")
Q3_DEC = ("R 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\nMY 0\nMPP Z0\nMPP Z0\nH 3\nT 3\nX_ERROR(0.5) 2\n"
          "CCZ 0 3 2\nDECISION(0) rec[-3]\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\nOUTPUT_QUBITS out 2 3\n")


def _bits(a, k):
    return np.unpackbits(np.asarray(a), axis=1, bitorder="little")[:, :k]


# ── J-b2 J1b: the PRODUCTION legs (plan r2.1 J1b): record_layout() layout 3 on the PUBLIC sampler. Per shot: the
# outcomes are (u: rho coins, z: kappa_rec Born bits) of the shot's plan (BarrierBuffer.plan_structure(i)), and the
# record determines the emitted bits — record_outcomes(i) gives every decision's raw outcome (emitted XOR its invert)
# and the observable's up to one constant. Layout 3 has no exact-shot segment (production has no exact shot).
from _public import public_sampler, rec_order  # noqa: E402


@pytest.mark.parametrize("text", [P2, DEC_ONLY, M3_OBS_2DEC, M3_2DEC, Q3_DEC],
                         ids=["p2", "dec_only", "m3_obs_2dec", "m3_2dec", "q3_dec"])
def test_record_layout_segments_match_the_record_public(text):
    N = 3000
    s = public_sampler(text)
    b = s.sample_barrier(N, seed=5)
    L, st = b.record_layout(), s.record_layout()
    assert L["layout"] == st["layout"] == 3 and st["records"] == len(s.record_operators())
    assert tuple(L["outcomes"]) == ("u: rho coin bits (pivot order)", "z: kappa_rec Born bits (direction order)")
    m = st["records"]
    col = rec_order(text)
    nd, no, ndec = s.num_detectors, s.num_observables, s.num_decisions
    bd = b.born_dec()
    dec = _bits(b.decisions(), ndec) if ndec else None
    obs = _bits(b.obs(), no) if no else None
    pi = np.asarray(b.frames()["plan_index"])
    plans, obs_xor, mism, krec_any = {}, set(), 0, False
    for i in range(N):
        p = int(pi[i])
        if p not in plans:
            ps = b.plan_structure(i)
            plans[p] = (int(np.asarray(ps["rho"])[0]), int(np.asarray(ps["kappa_rec"])[0]))
        rho, krec = plans[p]
        krec_any |= krec > 0
        assert len(b.outcomes(i)) == rho + krec
        y = np.asarray(b.record_outcomes(i))
        assert len(y) == m
        for c_, (kind, idx) in enumerate(col):
            if kind == 2 and idx in list(bd["dec_index"]):
                j = list(bd["dec_index"]).index(idx)
                mism += int(y[c_]) != (int(dec[i, idx]) ^ int(bd["inv"][j]))
            if kind == 1:
                obs_xor.add((idx, int(y[c_]) ^ int(obs[i, idx])))
    assert mism == 0
    assert len(obs_xor) == len({k for k, _ in obs_xor})
    if text is M3_OBS_2DEC:
        assert krec_any


def test_record_layout_static_and_v1_keys_gone_public():
    s = public_sampler("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n")
    b = s.sample_barrier(50, seed=1)
    L = b.record_layout()
    for k in ("classes", "has_born_obs", "nborn", "born_dec_index", "fair", "kernel", "born_obs",
              "born_dec", "prefix", "born_obs_at", "born_dec_at", "len", "version", "kinds", "entries"):
        with pytest.raises(KeyError):
            L[k]
    for obj in (b, s):
        assert not hasattr(obj, "coin_layout")
    assert all(len(b.outcomes(i)) == 0 for i in range(50))              # rho = kappa_rec = 0 on every plan
