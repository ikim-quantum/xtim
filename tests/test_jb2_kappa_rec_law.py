"""J-b2 J1b: the PRODUCTION carrier of tests/test_twirl_kappa_dual.py's law (plan r2.1 J1b, review MINOR-4).

test_twirl_kappa_dual checked the sigma-space law of kappa >= 2 plans (the dual-basis fold) on the legacy sampler's
exported plans. sigma is not a production surface since J-b1 F; the law it guarded is the RECORD law of plans with
kappa_rec >= 2 Born directions. Here the same fixture family — B blocks of [[4,2,2]] with BOTH logicals magic (HT|+>,
X.HT|+>, T|+>: distinct Born factors), the X ancillas chained so the basis carries NESTED X-type generators — is
sampled on the PUBLIC sampler with faults before every T (the fault's S residual on the logical wires makes the
kappa_rec >= 2 plans) and two Born-read DECISIONs per block. The emitted joint law of every record column equals the
exact law of the circuit text on dense states (tests/_record_oracle.py, no xtim code), every cell at |z| < 5 and no
shot in a zero-probability cell; a coverage assertion pins that the run reaches kappa_rec >= 2 plans
(record_report()["rec"]["kappa_rec_hist"])."""
import functools
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _public import public_sampler  # noqa: E402
from _record_oracle import emitted_columns, emitted_law, max_z  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def circuit_text(B, p=0.3, reads="MX", faulty=None):
    """test_twirl_kappa_dual.circuit_text(B) (copied verbatim, minus OUTPUT_QUBITS) with X_ERROR(p) before every T
    of the `faulty` blocks (default: all) and per block two Born-read DECISIONs (reads on the two logical wires)."""
    faulty = range(B) if faulty is None else faulty
    L = []
    for b in range(B):
        L.append("RX %d" % (4 * b))
    for b in range(B - 1, 0, -1):
        L.append("CX %d %d" % (4 * b, 4 * (b - 1)))
    for b in range(B):
        for d in (4 * b + 1, 4 * b + 2):
            L += ["RX %d" % d] + (["X_ERROR(%g) %d" % (p, d)] if b in faulty else []) + ["T %d" % d] + (["H %d" % d] if b != 2 else []) + \
                 (["X %d" % d] if b == 1 else [])
        L.append("R %d" % (4 * b + 3))
    for b in range(B):
        a0, d1, d2, a3 = 4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3
        L += ["CX %d %d" % (a0, d1), "CX %d %d" % (a0, d2), "CX %d %d" % (d1, a3),
              "CX %d %d" % (d2, a3), "CX %d %d" % (a0, a3)]
    L.append("X_ERROR(0.01) %d" % (4 * B - 1))
    for b in range(B):
        q = [4 * b + i for i in range(4)]
        L.append("MPP " + "*".join("X%d" % i for i in q))
        L.append("DETECTOR rec[-1]")
        L.append("MPP " + "*".join("Z%d" % i for i in q))
        L.append("DETECTOR rec[-1]")
    k = 0
    for b in range(B):
        for q in (4 * b + 1, 4 * b + 2):
            L += ["%s %d" % (reads, q), "DECISION(%d) rec[-1]" % k]
            k += 1
    return "\n".join(L) + "\n"


def _unpack(a, k):
    a = np.asarray(a, np.uint8)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :k] if k else np.zeros((a.shape[0], 0), np.uint8)


N = 40_000
CASES = [(B, r) for B in (2, 3) for r in ("MX", "M", "MY")]
# B = 3 with faults on every block exceeds the dense oracle's branch cap; faults on block 0 reach kappa_rec = 3.
FAULTY = {2: None, 3: (0,)}


@functools.lru_cache(maxsize=None)
def _law(text):
    return emitted_law(text)


@pytest.mark.parametrize("mode", ["sample", "barrier"])
@pytest.mark.parametrize("B,reads", CASES, ids=["B%d_%s" % c for c in CASES])
def test_kappa_rec_ge2_record_law_is_exact(B, reads, mode):
    text = circuit_text(B, reads=reads, faulty=FAULTY[B])
    s = public_sampler(text)
    nd, no, nde = s.num_detectors, s.num_observables, s.num_decisions
    if mode == "sample":
        r = s.sample(N, seed=11)
        d, o, dec = r[0], r[1], r[2]
    else:
        b = s.sample_barrier(N, seed=11)
        d, o, dec = b.dets(), b.obs(), b.decisions()
    M = np.concatenate([_unpack(d, nd), _unpack(o, no), _unpack(dec, nde)], 1)
    assert M.shape[1] == len(emitted_columns(text))
    z, imp = max_z(M, _law(text))
    assert imp == 0 and z < 5, (z, imp)
    hist = s.record_report()["rec"]["kappa_rec_hist"]
    assert sum(hist[2:]) > 0, hist                     # coverage: kappa_rec >= 2 plans are reached


def test_kappa_rec_law_mutation_is_red():
    """Non-vacuity: the dense law of a DIFFERENT circuit (block 0's H on wire 1 dropped: HT|+> -> T|+>, its Z-read
    Born factor 1/sqrt2 -> 0) is rejected by the same check."""
    text = circuit_text(2, reads="M")
    s = public_sampler(text)
    r = s.sample(N, seed=11)
    M = np.concatenate([_unpack(r[0], s.num_detectors), _unpack(r[1], s.num_observables),
                        _unpack(r[2], s.num_decisions)], 1)
    wrong = text.replace("T 1\nH 1\n", "T 1\n", 1)
    assert wrong != text
    z, imp = max_z(M, emitted_law(wrong))
    assert imp > 0 or z >= 5, (z, imp)


def kappa_dual_circuit_text(B):
    """tests/test_twirl_kappa_dual.circuit_text(B) verbatim (that file retired at J-b2 J2: its sigma-space law has no
    production surface; its law is carried by the gates above). Used by test_jb1_record_fidelity_l3."""
    L = []
    for b in range(B):
        L.append("RX %d" % (4 * b))
    for b in range(B - 1, 0, -1):               # nest: X_{4b} -> X_{4b} X_{4(b-1)} ...
        L.append("CX %d %d" % (4 * b, 4 * (b - 1)))
    for b in range(B):
        for d in (4 * b + 1, 4 * b + 2):
            # block b's logicals, one T each (the engine refuses T.H.T on a wire): block 0
            # HT|+> (<Z> = +1/sqrt2), block 1 X.HT|+> (-1/sqrt2), block 2 T|+> (0) — distinct
            # Born factors, so a PERMUTED pairing matrix cannot hide behind a block symmetry
            L += ["RX %d" % d, "T %d" % d] + (["H %d" % d] if b != 2 else []) + (["X %d" % d] if b == 1 else [])
        L.append("R %d" % (4 * b + 3))
    for b in range(B):
        a0, d1, d2, a3 = 4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3
        L += ["CX %d %d" % (a0, d1), "CX %d %d" % (a0, d2), "CX %d %d" % (d1, a3),
              "CX %d %d" % (d2, a3), "CX %d %d" % (a0, a3)]
    L.append("X_ERROR(0.01) %d" % (4 * B - 1))
    for b in range(B):
        q = [4 * b + i for i in range(4)]
        L.append("MPP " + "*".join("X%d" % i for i in q))
        L.append("DETECTOR rec[-1]")
        L.append("MPP " + "*".join("Z%d" % i for i in q))
        L.append("DETECTOR rec[-1]")
    L.append("OUTPUT_QUBITS out " + " ".join(str(q) for q in range(4 * B)))
    return "\n".join(L) + "\n"
