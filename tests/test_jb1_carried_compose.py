"""J-b1 P9b (XQ9 (A), ruling on fixplan-p9b-xq9, review C1-C10): a consumer's non-Clifford gates on CARRIED wires.

check_no_magic_on_carried is lifted on the FAM path: the consumer's bare is composed on the carried free register
exactly (cpp/src/carried_compose.cpp: the MERGED odd axes, the span criterion with the frame gauge, the F rows as
FastTODD variables, the weighted assembly). The criterion's failure is refused by name:
XtimTwirlRefusal(LAW_LIMITATION_CARRIED_AXIS, "carried_free_x"), with no T-depth claim.

Gates:
  * fixed chains (carried |T>, the review's gauge cases p5 / pA-pB, the merge case p3 (two T's merge into an S), CCZ
    across the seam): the composed bare equals the dense evolution of the producer's ideal state through the
    consumer's deferred stream (|overlap| = 1 at 1e-12); the minimal T-depth-2 instance is refused by name;
  * the random chained family: every ACCEPT equals dense, every REFUSE is confirmed by an INDEPENDENT span oracle
    computed from the exported ideal frame and the consumer's own gates (merged odd axes, ranks over the frame rows);
    the T-depth-1 refusals (joint circuit accepted in-stage) are counted and reported;
  * MUT-F-CONST (the F rows reduced as constants) and MUT-XQ9-KEEP-CHECK (the old refusal) -> red here.
"""
import os
import random
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _carried_oracle as O  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402
from xtim.errors import TwirlClass, XtimTwirlRefusal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
TOL = 1e-12

FIXED = {
    "t_then_t": ("RX 0\nT 0\nOUTPUT_QUBITS out 0\n", "INPUT_QUBITS in 0\nT 0\nMX 0\n"),
    "t_cx_fresh": ("RX 0\nT 0\nOUTPUT_QUBITS out 0\n", "INPUT_QUBITS in 0\nCX 0 1\nT 1\nH 1\nM 1\nM 0\n"),
    "gauge_p5": ("RX 0 1 2\nH 2\nH 2\nS 1\nH 0\nCX 1 0\nT 0\nOUTPUT_QUBITS out 0 1 2\n",
                 "INPUT_QUBITS in 0 1 2\nH 1\nT 1\nT 0\nS 2\nT 0\nT 0\nH 1\nCZ 0 1\nM 0 1 2\n"),
    "gauge_pA": ("RX 0\nR 1\nCX 0 1\nT 1\nCX 0 1\nOUTPUT_QUBITS out 0 1\n",
                 "INPUT_QUBITS in 0 1\nCX 0 1\nH 0\nT 0\nH 0\nM 0 1\n"),
    "gauge_pB": ("RX 0\nR 1\nT 0\nOUTPUT_QUBITS out 0 1\n", "INPUT_QUBITS in 0 1\nCX 0 1\nH 0\nT 0\nH 0\nM 0 1\n"),
    "merge_p3": ("R 0\nRX 1\nT 1\nH 1\nCX 0 1\nCX 1 0\nOUTPUT_QUBITS out 0 1\n",
                 "INPUT_QUBITS in 0 1\nT 0\nCX 0 1\nS 0\nCZ 1 0\nCZ 0 1\nS 1\nT 0\nCZ 1 0\nM 0 1\n"),
    "stab_htH": ("R 0\nOUTPUT_QUBITS out 0\n", "INPUT_QUBITS in 0\nH 0\nT 0\nH 0\nM 0\n"),
    "ccz_seam": ("RX 0\nT 0\nOUTPUT_QUBITS out 0\n", "INPUT_QUBITS in 0\nR 1\nH 1\nCCZ 0 1 2\nM 0 1 2\n"),
}
MINIMAL_TD2 = ("RX 0\nT 0\nOUTPUT_QUBITS out 0\n", "INPUT_QUBITS in 0\nH 0\nT 0\nH 0\nM 0\n")


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def consumer(p, ctext):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler_from_ideal_state(p.ideal_state(), ctext)


def check_dense(ptext, ctext):
    p = fam(ptext)
    R = p.sample_records(50, seed=1)[-1]
    H = p.branch_handles(R)
    v, n_p = O.ideal_dense(p, R, H)
    c = consumer(p, ctext)
    if int(TA.bare_frames(c)["n"]) > 18:
        return None, c                                  # beyond the dense oracle
    w, n = O.consumer_oracle(v, n_p, p.ideal_state().wires, ctext)
    b = O.bare_dense(c)
    return abs(abs(np.vdot(w, b)) - 1.0), c


@pytest.mark.parametrize("name", sorted(FIXED))
def test_fixed_chains_equal_dense(name):
    err, c = check_dense(*FIXED[name])
    assert err < TOL, err
    if name != "stab_htH":
        assert c.carried_compose_report()["route"] == "compose"
    if name.startswith("gauge_p5") or name == "gauge_pB":
        assert c.carried_compose_report()["gauge_moves"] > 0           # the gauge carries these cases


def test_minimal_td2_refused_by_name():
    p = fam(MINIMAL_TD2[0])
    with pytest.raises(XtimTwirlRefusal) as ei:
        consumer(p, MINIMAL_TD2[1])
    assert ei.value.twirl_class is TwirlClass.LAW_LIMITATION_CARRIED_AXIS
    assert ei.value.sub_reason == "carried_free_x"
    assert "T-depth" not in str(ei.value)


# ── the random chained family (C8) ──────────────────────────────────────────────────────────────────────────────
def _gates(rng, nq, ng, tprob):
    out = []
    for _ in range(ng):
        r = rng.random()
        if r < tprob: out.append(("T", rng.randrange(nq)))
        elif r < tprob + 0.25: out.append(("H", rng.randrange(nq)))
        elif r < tprob + 0.4: out.append(("S", rng.randrange(nq)))
        elif nq == 1: out.append(("H", 0))
        else:
            a, b = rng.sample(range(nq), 2)
            out.append((rng.choice(["CX", "CZ"]), a, b))
    return out


def _txt(gs, mp=lambda q: q):
    return "".join(f"{g[0]} {' '.join(str(mp(q)) for q in g[1:])}\n" for g in gs)


def _pull(gs, upto, q, n):
    """Z_q at gate `upto`, pulled back to the consumer's input through the preceding Cliffords (bits only)."""
    x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); z[q] = 1
    for g in reversed(gs[:upto]):
        k = g[0]
        if k == "H": a = g[1]; x[a], z[a] = z[a], x[a]
        elif k == "S": a = g[1]; z[a] ^= x[a]
        elif k == "CX": c, t = g[1], g[2]; x[t] ^= x[c]; z[c] ^= z[t]
        elif k == "CZ": c, t = g[1], g[2]; z[c] ^= x[t]; z[t] ^= x[c]
    return x, z


def _rank(M):
    M = [list(r) for r in M]
    r = 0
    if not M: return 0
    for c in range(len(M[0])):
        p = next((i for i in range(r, len(M)) if M[i][c]), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(len(M)):
            if i != r and M[i][c]: M[i] = [a ^ b for a, b in zip(M[i], M[r])]
        r += 1
    return r


def span_oracle(idf, wires, cg, nc, k):
    """ACCEPT iff no combination of the merged odd consumer axes has its X-part on the ideal state's free rows only
    (rank over all frame rows == rank over the non-free rows); None if the consumer's own axes anticommute."""
    axes = {}
    for i, g in enumerate(cg):
        if g[0] == "T":
            x, z = _pull(cg, i, g[1], nc)
            key = (bytes(x), bytes(z))
            axes[key] = axes.get(key, 0) ^ 1
    odd = [(np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)) for (a, b), v in axes.items() if v]
    for a in odd:
        for b in odd:
            if (int(a[0] @ b[1]) + int(a[1] @ b[0])) % 2:
                return None
    n_p = int(idf["n"])
    sx, sz = np.asarray(idf["stab_x"], np.uint8), np.asarray(idf["stab_z"], np.uint8)
    F = set(map(int, idf["free"]))
    rows_all, rows_c = [], []
    for x, z in odd:
        px = np.zeros(n_p, np.uint8); pz = np.zeros(n_p, np.uint8)
        for j in range(k):
            px[wires[j]] = x[j]; pz[wires[j]] = z[j]
        prod = [int((px @ sz[a] + pz @ sx[a]) % 2) for a in range(n_p)]   # <P, g_a> on the producer rows
        fresh = [int(x[w]) for w in range(k, nc)]                         # <P, Z_w> on the fresh rows
        rows_all.append(prod + fresh)
        rows_c.append([prod[a] for a in range(n_p) if a not in F] + fresh)
    return _rank(rows_all) == _rank(rows_c)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_chained_family(seed):
    rng = random.Random(seed)
    st = dict(acc=0, ref=0, compose=0, td1_refused=0, skipped=0)
    for it in range(150):
        k = rng.choice([1, 2, 2, 3]); nf = rng.choice([0, 1, 1, 2])
        pg = _gates(rng, k, rng.randint(2, 8), 0.3) if k > 1 else [("T", 0), ("H", 0), ("T", 0)][:rng.randint(1, 3)]
        if not any(g[0] == "T" for g in pg): pg.append(("T", 0))
        prep = "".join(f"{'RX' if rng.random() < 0.6 else 'R'} {q}\n" for q in range(k))
        body = _txt(pg)
        if rng.random() < 0.3:                          # a producer coin / check read on an ancilla (post-barrier frame)
            a = k + nf + 2
            body += f"R {a}\n{rng.choice(['CX', 'CZ'])} {rng.randrange(k)} {a}\nM {a}\nDETECTOR rec[-1]\n"
        ptext = prep + body + "OUTPUT_QUBITS out " + " ".join(map(str, range(k))) + "\n"
        nc = k + nf
        cg = _gates(rng, nc, rng.randint(2, 8), 0.3)
        if not any(g[0] == "T" for g in cg):
            continue
        ctext = "INPUT_QUBITS in " + " ".join(map(str, range(k))) + "\n" + _txt(cg) + \
            "M " + " ".join(map(str, range(nc))) + "\n"
        try:
            p = fam(ptext)
            R = p.sample_records(64, seed=1)[-1]
            H = p.branch_handles(R)
            i0 = O.ideal_record(R, H)
            if i0 is None:
                st["skipped"] += 1
                continue
            idf = p.materialize(R, i0)
        except (ValueError, RuntimeError):
            st["skipped"] += 1
            continue
        verdict = span_oracle(idf, p.ideal_state().wires, cg, nc, k)
        if verdict is None:
            st["skipped"] += 1
            continue
        try:
            c = consumer(p, ctext)
        except XtimTwirlRefusal as e:
            assert e.twirl_class is TwirlClass.LAW_LIMITATION_CARRIED_AXIS and e.sub_reason == "carried_free_x"
            assert verdict is False, (ptext, ctext)
            st["ref"] += 1
            joint = prep + body + _txt(cg) + "M " + " ".join(map(str, range(nc))) + "\n"
            try:
                fam(joint)
                st["td1_refused"] += 1
            except (ValueError, RuntimeError):
                pass
            continue
        assert verdict is True, (ptext, ctext)
        err, c = check_dense(ptext, ctext)
        if err is None:
            st["skipped"] += 1
            continue
        assert err < TOL, (err, ptext, ctext)
        st["acc"] += 1
        st["compose"] += c.carried_compose_report()["route"] == "compose"
    print("chained family", seed, st)
    assert st["acc"] > 0 and st["compose"] > 0 and st["ref"] > 0, st


def test_one_info_event_and_counters(caplog):
    import logging
    p = fam(FIXED["t_then_t"][0])
    with caplog.at_level(logging.INFO, logger="xtim.twirl"):
        c = consumer(p, FIXED["t_then_t"][1])
    msgs = [r.getMessage() for r in caplog.records if "carried composition" in r.getMessage()]
    assert len(msgs) == 1
    rep = c.carried_compose_report()
    assert rep["route"] == "compose" and rep["axes"] >= 1
    with caplog.at_level(logging.INFO, logger="xtim.twirl"):
        c2 = consumer(p, "INPUT_QUBITS in 0\nH 0\nM 0\n")             # no magic on carried wires: unchanged route
    assert c2.carried_compose_report()["route"] == "inject"
