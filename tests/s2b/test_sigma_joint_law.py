"""S2b gate r2.2 / r3.2 / r4.3 / r5.3: the barrier's joint law of (record columns, sigma bits) is
today's law (sigma-4, owner 2026-10-16), compared OLD (the P7 tip, byte-identical to a55c1c0) vs NEW.

Two builds are captured with this file as a script (each run imports the xtim of the tree given):

    python tests/s2b/test_sigma_joint_law.py capture <tree> <out.npz> [N] [seed]

and the gate compares the two captures (pytest, with S2B_SIGMA_OLD / S2B_SIGMA_NEW set to the
two .npz files; skipped otherwise):

  * every cell of the tuple (every emitted record column, every sigma bit not constant in OLD):
    two-sample binomial |z| < 5;
  * every marginal and every PAIR of those columns: |z| < 5 on each of the four cells;
  * impossible outcomes: a cell with OLD count 0 holding >= 25 NEW shots FAILS.

The fixture set names, per class, the mutation it turns red (plan r3.2 / r4.3 / r5.3 / r5.4).
"""
import os
import sys

import numpy as np
import pytest

_M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
       "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
       "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
_DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
_H = "R 0\nRX 1 2\nT 2\nX_ERROR(0.5) 0\nCCZ 0 1 2\nM 2\n"
_NDH = "RX 3\nT 3\nRX 4\nT_DAG 4\nX_ERROR(0.5) 2\nCS_DAG 4 3\nCCZ 4 3 2\n"
_NDH2 = (_NDH + "R 5 6\nH 5\nCX 5 6\nRX 7\nT 7\nR 8\nX_ERROR(0.5) 8\nCCZ 7 5 8\n"
         "R 9\nH 9\nCX 9 5\nCX 9 6\nH 9\nM 9\nDETECTOR rec[-1]\n")
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")

FIXTURES = {
    # MA-1 (C without W): sigma11 == W on the kappa>0 plan
    "wlam": _H + "OBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    # MS-1 (sigma-2: aux conditioned on the decisions): sigma11 independent of D
    "wlam_dec": _H + "DECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    # MB-1 (introducer-based read span)
    "sig_ovl": _H + "DECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "span_obs_obs": _H + "OBSERVABLE_INCLUDE(0) rec[-1]\nMX 1\nOBSERVABLE_INCLUDE(1) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "span_obs_dec": _H + "OBSERVABLE_INCLUDE(0) rec[-1]\nMX 1\nDECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "span_dec_mix": _H + ("DECISION(0) rec[-1]\nMX 1\nR 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\n"
                          "DECISION(1) rec[-2] rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # MC-1 / MC-2: W out of the kappa_sigma span, and its null control
    "wout": ("R 0\nRX 1 2 3\nT 2\nT 3\nH 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nMPP Z2*Z3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
             "R 9\nM 9\nDETECTOR rec[-1]\n"),
    "wout_null": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nMPP Z2*Z3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                  "R 9\nM 9\nDETECTOR rec[-1]\n"),
    # MD-1 (support rule): a decision reading only a combination of two decision directions
    "sig_comb": ("R 0\nRX 1 2 3 4\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nCCZ 0 4 2\n"
                 "M 2\nDECISION(0) rec[-1]\nM 3\nDECISION(1) rec[-1]\nMX 1\nDECISION(2) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # MP-1 (literal-product rule): an X-type C member on a diagonal-magic register, |b| >= 2
    "sig_comb_x": ("R 0\nRX 1 2 3 4 5\nCCZ 2 3 5\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nCCZ 0 4 2\n"
                   "MX 5\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                   "M 2\nDECISION(0) rec[-1]\nM 3\nDECISION(1) rec[-1]\nMX 1\nDECISION(2) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # MS-3 (aux before the records): the audit-M hazard family
    "ndhaz_F1": _NDH + "R 10\nM 10\nDETECTOR rec[-1]\nMX 4\nDECISION(0) rec[-1]\n",
    "ndhaz_F1o": _NDH + "R 10\nM 10\nDETECTOR rec[-1]\n",
    "ndhaz_F1x": _NDH + "R 10\nM 10\nDETECTOR rec[-1]\nMX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "ndhaz_F2": _NDH2 + "MX 4\nDECISION(0) rec[-1]\nMX 7\nDECISION(1) rec[-1]\nMX 3\nDECISION(2) rec[-1]\n",
    # MS-2 (aux coins at rho_ch): the adv-A ANTI-decision trio
    "anti_q3dec": ("RX 0\nR 1 2\nX_ERROR(0.2) 0\nT 0\nH 1\nCX 1 2\nCX 0 1\nDEPOLARIZE1(0.05) 1 2\nMX 1 2 0\n"
                   "DETECTOR rec[-3] rec[-2]\nDECISION(0) rec[-3]\nDECISION(1) rec[-1]\n"),
    "anti_bell2_3dec": ("RX 0\nR 1 2 3 4\nX_ERROR(0.25) 0\nT 0\nH 1 3\nCX 1 2 3 4\nCX 0 1\nCZ 0 3\n"
                        "DEPOLARIZE1(0.06) 0 1 2 3 4\nMX 1 2 3 4 0\nDETECTOR rec[-5] rec[-4]\n"
                        "DECISION(0) rec[-5]\nDECISION(1) rec[-3]\nDECISION(2) rec[-1]\n"),
    "anti_teleport_dec": ("RX 0\nR 1 2 3\nX_ERROR(0.2) 0\nT 0\nH 1\nCX 1 2\nCX 0 1\nH 0\n"
                          "DEPOLARIZE1(0.05) 0 1 2 3\nM 0 1\nMX 2\nM 3\nDETECTOR rec[-1]\n"
                          "DECISION(0) rec[-4]\nDECISION(1) rec[-3]\nDECISION(2) rec[-2]\n"),
    # MS-2 (aux coins at rho_ch): unread sigma coins (CZ(1,2) dressing |++>) beside an ANTI decision
    # pivot — the one fixture with sigma-aux COINS (record_report()["sigma_aux"]["coins"] > 0)
    "anti_aux": ("R 0\nRX 1 2\nX_ERROR(0.5) 0\nCCZ 0 1 2\nR 9\nM 9\nDETECTOR rec[-1]\nMX 0\n"
                 "DECISION(0) rec[-1]\n"),
    # MP-1 / MP-1b end to end (departures review MAJOR-B): a combined, biased record direction
    "a4_comb": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nMPP X2*X3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                 "MPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # code review MAJOR-1 / MAJOR-2: the fold bit beta' = 1 (M-BETA) and a non-canonical combined
    # literal product (MP-1) reached from a circuit
    "a4_minus_z": ("R 0\nRX 1 2 3\nZ 1\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nMPP X2*X3\n"
                   "OBSERVABLE_INCLUDE(0) rec[-1]\nMPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\n"
                   "R 9\nM 9\nDETECTOR rec[-1]\n"),
    "a4_comb_rev": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 3\nCCZ 0 1 2\nMPP X2*X3\n"
                    "OBSERVABLE_INCLUDE(0) rec[-1]\nMPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\n"
                    "R 9\nM 9\nDETECTOR rec[-1]\n"),
    "a4d_rev": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 3\nCCZ 0 1 2\nMPP X2*X3\n"
                "DECISION(0) rec[-1]\nMPP Y2*Y3\nDECISION(1) rec[-1]\nMX 1\nDECISION(2) rec[-1]\n"
                "R 9\nM 9\nDETECTOR rec[-1]\n"),
    # regression coverage (non-discriminating for A/B)
    "m3_obs": _M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET,
    "m3_obs_x3": _M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 5\nX_ERROR(0.3) 5\nM 5\nDETECTOR rec[-1]\n",
    "m3_det": _M3 + "MX 3\nMX 4\n" + _DET,
    "m3_obs_dec": _M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET + "MX 0\nDECISION(0) rec[-1]\n",
    "m3_obs_2dec": _M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET + "MX 0\nDECISION(0) rec[-1]\nMY 1\nDECISION(1) rec[-1]\n",
    "q3_dec": ("R 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\nMY 0\nMPP Z0\nMPP Z0\nH 3\nT 3\nX_ERROR(0.5) 2\nCCZ 0 3 2\n"
               "DECISION(0) rec[-3]\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\n"),
    "born5": open(os.path.join(_DATA, "adaptq_born5_dec_enum_producer.stim")).read(),
}


def _unpack(a, k):
    a = np.asarray(a, np.uint8)
    if k == 0:
        return np.zeros((a.shape[0], 0), np.uint8)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :k]


def capture(tree, out, N=200_000, seed=11):
    sys.path.insert(0, tree)
    import warnings
    warnings.simplefilter("ignore")
    import xtim
    assert xtim.__file__.startswith(tree), xtim.__file__
    arrs = {}
    for name, text in FIXTURES.items():
        # ndhaz_F1x's second LOGICAL observable is refused: its documented zero column (E1: obs()
        # refuses by name without the opt-in) — the same column on both builds
        # selfcheck >= N (review r6 NIT): every barrier shot is in the selfcheck window, where the
        # σ-4 channel relation (channel bit = mask·σ) is a named throw (r6.5); stream-neutral (Q6)
        s = xtim.compile_twirl_sampler(text, selfcheck=N, skip_refused_observables=True)
        b = s.sample_barrier(N, seed=seed)
        cols = [_unpack(b.dets(), s.num_detectors), _unpack(b.obs(), s.num_observables)]
        if s.num_decisions:
            cols.append(_unpack(b.decisions(), s.num_decisions))
        rec = np.concatenate(cols, 1)
        sw = np.asarray(b.sigmas()).astype("<u8")
        sig = np.unpackbits(sw.view(np.uint8).reshape(N, -1), axis=1, bitorder="little")
        arrs[name + "/rec"] = rec
        arrs[name + "/sig"] = sig
        arrs[name + "/exact"] = np.asarray(b.exact_flags()).astype(np.uint8)
    np.savez_compressed(out, **arrs)


def _z(ca, na, cb, nb):
    p = (ca + cb) / (na + nb)
    if p <= 0 or p >= 1:
        return 0.0
    return (ca / na - cb / nb) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb))


def compare_fixture(old_rec, old_sig, new_rec, new_sig):
    """(max |z| over cells, marginals and pairs; impossible-outcome cells; tuple width)."""
    var = [j for j in range(old_sig.shape[1]) if 0 < old_sig[:, j].mean() < 1]
    A = np.concatenate([old_rec, old_sig[:, var]], 1).astype(np.uint8)
    B = np.concatenate([new_rec, new_sig[:, var]], 1).astype(np.uint8)
    na, nb = len(A), len(B)
    worst, impossible = 0.0, []
    ntests = [0]

    def cells(Xa, Xb):
        nonlocal worst
        ka, ca = np.unique(Xa, axis=0, return_counts=True)
        kb, cb = np.unique(Xb, axis=0, return_counts=True)
        da = {tuple(k): int(c) for k, c in zip(ka, ca)}
        db = {tuple(k): int(c) for k, c in zip(kb, cb)}
        for k in set(da) | set(db):
            a, b = da.get(k, 0), db.get(k, 0)
            if a == 0 and b >= 25:
                impossible.append((k, b))
            ntests[0] += 1
            worst = max(worst, abs(_z(a, na, b, nb)))

    cells(A, B)
    m = A.shape[1]
    for i in range(m):
        cells(A[:, [i]], B[:, [i]])
        for j in range(i + 1, m):
            cells(A[:, [i, j]], B[:, [i, j]])
    compare_fixture.ntests = ntests[0]
    return worst, impossible, m


_OLD, _NEW = os.environ.get("S2B_SIGMA_OLD"), os.environ.get("S2B_SIGMA_NEW")


@pytest.mark.skipif(not (_OLD and _NEW), reason="needs S2B_SIGMA_OLD / S2B_SIGMA_NEW captures")
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_sigma_joint_law_old_equals_new(name):
    a, b = np.load(_OLD), np.load(_NEW)
    worst, imp, m = compare_fixture(a[name + "/rec"], a[name + "/sig"], b[name + "/rec"], b[name + "/sig"])
    assert not imp and worst < 5, (name, worst, imp[:5], m)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:6]))
    elif sys.argv[1] == "compare":
        a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
        bad = 0
        total = 0
        for name in sorted(FIXTURES):
            worst, imp, m = compare_fixture(a[name + "/rec"], a[name + "/sig"], b[name + "/rec"], b[name + "/sig"])
            ok = not imp and worst < 5
            bad += not ok
            total += compare_fixture.ntests
            print("%-18s width=%2d tests=%6d max|z|=%6.2f impossible=%d %s"
                  % (name, m, compare_fixture.ntests, worst, len(imp), "ok" if ok else "FAIL"))
        print("TESTS", total, "FAILS", bad)
