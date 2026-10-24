"""J-b1 P9b: the branch-frame refusal (ruling on fixplan-p9b-frame, path (R); review jb1/fixplan-p9b-frame-review.md).

A producer branch's materialize frame can leave the ideal post-barrier frame when a forced coin read's pivot rule sees
the C0-hat-promoted E_S rows: (i) a skipped pivot (the CE1-chain dressed branch) or (ii) a different pivot (the review's
p2 producer: the frames are row-permuted and the old E_min MISSED the extended row), plus cascades. add_branch detects
it EXACTLY (stabilizer rows on every row, destabilizer rows on every row free in either state) and refuses it by name:
XtimTwirlRefusal(LAW_LIMITATION_CARRIED_AXIS, "branch_frame"); the alignment (b') is J-c.

Gates:
  * CE1-chain and p2 dressed branches are refused by name; the refused branch leaves the table untouched (branches,
    distinct_emin, branch_table_digest), so branch_emin / distinct_emin / the digest only ever cover frame-equal
    branches;
  * CE1 MX0 (frame-equal, E_min != {}) is accepted and its E_min equals the dense oracle;
  * Lemma R1's counter on the force_record_ -> measure_pauli path (forced_pivot_nonpattern_count) is > 0 on CE1 and p2
    and 0 on every accepted branch;
  * the review's random producer family (coin reads included): every branch is accepted iff an INDEPENDENT row compare
    of the exported frames finds the frame equal; every accepted branch's E_min equals the dense oracle's; non-vacuous
    (refusals > 0, accepted extended branches > 0). MUT-FRAME-CHECK-OFF -> red here.
"""
import os
import random
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from xtim import _xtim_testapi as TA  # noqa: E402
from xtim.errors import TwirlClass, XtimTwirlRefusal  # noqa: E402
from xtim.frames import _pauli_apply, dense_state  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CE1_CHAIN = "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"
CE1_MX0 = "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nMX 0\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n"
P2 = ("RX 0 1\nR 2\nH 0\nCX 1 0\nT 1\nDEPOLARIZE1(0.2) 1\nCX 0 1\nCX 1 0\nS 1\nDEPOLARIZE1(0.2) 1\nT 0\nCZ 0 2\n"
      "H 2\nM 2\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1\n")


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def consumer(p):
    no = len(p.ideal_state().wires)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler_from_ideal_state(
            p.ideal_state(), "INPUT_QUBITS in " + " ".join(map(str, range(no))) + "\nI 0\n")


def rows(fr):
    n = int(fr["n"])
    return [(bytes(np.asarray(fr["stab_x"][a], np.uint8)), bytes(np.asarray(fr["stab_z"][a], np.uint8)))
            for a in range(n)]


def ideal_dict(p, R, H):
    """The independent oracle's ideal frame: materialize of an identity-content record at z = 0."""
    for i in range(R.num_records):
        ct = R.content(i)
        if R.kind(i) == 0 and not any(ct[:-1]) and not any(bytes(H[i]["z"])):
            return p.materialize(R, i)
    return None


def divergent(idf, fr):
    n = int(idf["n"])
    if rows(idf) != rows(fr):
        return True
    free = set(map(int, idf["free"])) | set(map(int, fr["free"]))
    for a in free:
        for k in ("destab_x", "destab_z"):
            if not np.array_equal(np.asarray(idf[k][a]), np.asarray(fr[k][a])):
                return True
    return False


def true_emin(idf, v):
    n = int(idf["n"])
    psi = dense_state(v)
    psi = psi / np.linalg.norm(psi)
    F0 = set(map(int, idf["free"]))
    out = []
    for a in range(n):
        if a in F0:
            continue
        e = np.vdot(psi, _pauli_apply(psi, n, idf["stab_x"][a], idf["stab_z"][a], int(idf["stab_phase"][a])))
        if abs(e) < 1 - 1e-9:
            out.append(a)
    return out


def handles(p, shots=300, seed=1):
    R = p.sample_records(shots, seed=seed)[-1]
    H = p.branch_handles(R)
    seen, out = set(), []
    for i, h in enumerate(H):
        k = (bytes(h["digest"]), bytes(h["z"]))
        if k not in seen:
            seen.add(k)
            out.append((i, h))
    return R, H, out


@pytest.mark.parametrize("name,text", [("ce1_chain", CE1_CHAIN), ("p2_different_pivot", P2)])
def test_divergent_branch_refused_by_name(name, text):
    p = fam(text)
    c = consumer(p)
    R, H, D = handles(p)
    refused = 0
    for i, h in D:
        before = (c.branch_counters(), c.branch_table_digest())
        c0 = TA.forced_pivot_nonpattern_count()
        try:
            c.add_branch(h)
        except XtimTwirlRefusal as e:
            assert e.twirl_class is TwirlClass.LAW_LIMITATION_CARRIED_AXIS and e.sub_reason == "branch_frame"
            assert "T-depth" not in str(e)
            assert (c.branch_counters(), c.branch_table_digest()) == before    # the table is untouched
            assert TA.forced_pivot_nonpattern_count() > c0                      # R1 counter sees the pivot
            refused += 1
    assert refused == 1


def test_frame_equal_extended_branch_accepted():
    p = fam(CE1_MX0)
    c = consumer(p)
    R, H, D = handles(p)
    idf = ideal_dict(p, R, H)
    ext = 0
    for i, h in D:
        c0 = TA.forced_pivot_nonpattern_count()
        b = c.add_branch(h)
        assert TA.forced_pivot_nonpattern_count() == c0
        assert sorted(c.branch_emin(b)) == true_emin(idf, c.branch_vector(b))
        ext += bool(c.branch_emin(b))
    assert ext == 1 and c.branch_counters()["distinct_emin"] == 2


def _gen(rng):
    """The review's random producer family (frrev/fuzz.py): 2-3 data qubits, T/T_DAG with Pauli noise, ancilla
    CX/CZ coin reads, an optional MX data read."""
    nq = rng.choice([2, 3]); na = rng.choice([1, 2])
    L = ["RX " + " ".join(str(q) for q in range(nq) if rng.random() < 0.6), "R " + " ".join(str(nq + a) for a in range(na))]
    for _ in range(rng.randint(3, 9)):
        r = rng.random(); q = rng.randrange(nq)
        if r < 0.2: L.append(f"T {q}")
        elif r < 0.3: L.append(f"T_DAG {q}")
        elif r < 0.45: L.append(f"H {q}")
        elif r < 0.55: L.append(f"S {q}")
        elif r < 0.7:
            a, b = rng.sample(range(nq), 2); L.append(f"CX {a} {b}")
        elif r < 0.85: L.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'Y_ERROR'])}(0.2) {q}")
        else: L.append(f"DEPOLARIZE1(0.2) {q}")
    for a in range(na):
        for q in rng.sample(range(nq), rng.randint(1, nq)):
            L.append(f"CX {q} {nq + a}" if rng.random() < 0.6 else f"CZ {q} {nq + a}")
        if rng.random() < 0.3: L.append(f"H {nq + a}")
        L.append(f"M {nq + a}"); L.append("DETECTOR rec[-1]")
    if rng.random() < 0.3:
        q = rng.randrange(nq); L.append(f"MX {q}"); L.append("DETECTOR rec[-1]")
        outs = [x for x in range(nq) if x != q]
    else:
        outs = list(range(nq))
    L.append("OUTPUT_QUBITS out " + " ".join(map(str, outs)))
    return "\n".join(l for l in L if l.strip() not in ("RX", "R", "")) + "\n"


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_random_producers_refused_iff_frame_divergent(seed):
    rng = random.Random(seed)
    st = dict(acc=0, ref=0, ext=0)
    for it in range(60):
        text = _gen(rng)
        try:
            p = fam(text)
            R, H, D = handles(p, seed=it + 1)
            idf = ideal_dict(p, R, H)
            if idf is None:
                continue
            c = consumer(p)
        except (ValueError, RuntimeError):
            continue
        for i, h in D:
            div = divergent(idf, p.materialize(R, i))
            c0 = TA.forced_pivot_nonpattern_count()
            try:
                b = c.add_branch(h)
            except XtimTwirlRefusal as e:
                assert e.sub_reason == "branch_frame" and div, text
                st["ref"] += 1
                continue
            assert not div, text
            assert TA.forced_pivot_nonpattern_count() == c0, text
            e = sorted(c.branch_emin(b))
            assert e == true_emin(idf, c.branch_vector(b)), text
            st["acc"] += 1
            st["ext"] += bool(e)
    assert st["ref"] > 0 and st["acc"] > 0, st
