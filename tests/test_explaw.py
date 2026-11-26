"""explaw (owner-approved 2026-10-30): the coset-of-X-span Born law behind the ONE Born-law routine.

A Born law of d commuting directions on the bare free register (the DEM's kappa kernel logicals; a PAULI_EXPECTATION
column's magic-correlated read parities) used to cost Theta(2^d x chi) sequential projections (BornTrie::enumerate). The
routine now routes by a measured cost model: enumerate for d <= 2 or when it is cheaper, else the coset law
(cpp/src/born_law_coset.cpp): O(s d + sum_C min(m_C^2, d_X 2^d_X) + #classes d_X 2^d_X + d 2^d) with enumerate's leaf
semantics replayed from the table (same leaf sets, rounding only; cpp/tests/test_born_law_coset.cpp is the oracle).

Gates here (mechanism + pins, owner pre-signed 2026-10-30): the k = 9 T-ladder's column walk routes to the coset law
with work <= c k 2^k (mutation: the route forced to enumerate -> RED); the k = 13 / 15 / 17 ladders export in seconds
(they took 6 s / 109 s / ~40 min); a DEM law with kappa >= 3 routes to the coset law and its DEM text stays the dense
oracle's; d <= 2 laws keep enumerate (born_sums unchanged on the twirl-routes pins).
"""
import hashlib
import time
import warnings

import numpy as np
import pytest

import xtim
from xtim.record_fidelity import RecordFidelity
from _register_oracle import bare_rho, purity, register_F
from xtim import _xtim_testapi as TA
from xtim._exact_scope import explicit_exact
import _exact


def census(text, **kw):
    with explicit_exact():
        return TA.dem_law_census(text, **kw)


def export(text, incl=True):
    return _exact.export_dem_text(text, incl, False, False, False, [], [], 0.0, -1)


def t_ladder(k):
    c = [f"RX {' '.join(map(str, range(k + 1)))}", "X_ERROR(0.01) 0",
         f"T {' '.join(map(str, range(k)))}"] + [f"CZ {i} {k}" for i in range(k)] + \
        [f"MX {' '.join(map(str, range(k)))}", f"PAULI_EXPECTATION(0) X{k}"]
    return "\n".join(c) + "\n"


def test_k9_ladder_column_walk_routes_to_the_coset_law():
    # E3 mechanism gate: the column's d = 8 Born directions on the k = 9 register (chi = 512, a dense hull) take the
    # coset law; its work is linear-log in the register (s d + the 2^d_X blocks), nowhere near enumerate's 2^d chi.
    k = 9
    r = census(t_ladder(k))
    assert r["ok"], r["error"]
    assert r["coset_laws"] >= 1
    assert r["coset_work"] <= 8 * k * 2 ** k, r["coset_work"]
    # the DEM text is the dense-oracle-checked one (test_dem_pe_per_shot.test_t_ladder_exports_exactly)
    assert r["dem"] == "error(0.01) L0\n"


@pytest.mark.parametrize("k,ceiling", [(13, 10.0), (15, 10.0), (17, 10.0)])
def test_t_ladder_exports_in_seconds(k, ceiling):
    # Re-pin (owner pre-signed 2026-10-30): k = 13 had a 600 s ceiling (6 s measured), k = 15 took 109 s, k = 17 was
    # never run (~40 min). The odd ladder's column is identically 0 per shot (k = 9's dense check certifies the
    # family's law), so the DEM is the vacuous sign flip of a zero column.
    t0 = time.time()
    r = export(t_ladder(k))
    dt = time.time() - t0
    assert r["ok"] and r["dem"] == "error(0.01) L0\n", r
    assert dt < ceiling, dt
    c = census(t_ladder(k))
    assert c["coset_laws"] >= 1


# A fault whose record table has kappa = 3 Born directions: three T magic qubits, each CS-coupled to its own |+> ancilla,
# and ONE fault flipping all three ancillas inside the CS / CS_DAG pairs (a correlated X, or an X on a hub fanned out
# by CX): the three MX reads carry one Born direction each on the free register (P(z) = 1/8 uniform: a coset law).
KAPPA3 = ("RX 0 2 4\nT 0 2 4\nRX 1 3 5\nCS 1 0\nCS 3 2\nCS 5 4\nE(0.01) X1 X3 X5\nCS_DAG 1 0\nCS_DAG 3 2\nCS_DAG 5 4\n"
          "MX 1 3 5\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")
KAPPA3_HUB = ("RX 0 2 4\nT 0 2 4\nRX 1 3 5\nR 6\nCS 1 0\nCS 3 2\nCS 5 4\nX_ERROR(0.01) 6\nCX 6 1\nCX 6 3\nCX 6 5\n"
              "CS_DAG 1 0\nCS_DAG 3 2\nCS_DAG 5 4\nMX 1 3 5\nM 6\nDETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\n"
              "DETECTOR rec[-1]\n")


@pytest.mark.parametrize("text,md5,lines", [(KAPPA3, "2c343a5e2619fc5bfbf06934bebf1ec6", 7),
                                            (KAPPA3_HUB, "6fca714b29b4629bfe314bff8db738cb", 8)])
def test_kappa3_dem_law_routes_to_the_coset_law(text, md5, lines):
    # Owner signature item (d >= 3 DEM laws): the texts are BYTE-IDENTICAL to dev/next 8d61f21's (enumerate's), the
    # census Born count moves 15 (2^4 - 1 node + leaf evaluations) -> 16 (8 leaves + 8 cosets; pre-signed re-pin).
    # dem-t2: the pinned texts are the exact split, requested explicitly (the default is t = 2).
    r = census(text, clifford_cutoff=-1)
    assert r["ok"], r["error"]
    comps = [c for i in r["instances"] for c in i["comps"]]
    assert [c["kappa"] for c in comps] == [3] and comps[0]["kind"] == "C", comps
    assert r["coset_laws"] == 1 and r["born_sums"] == 16
    assert hashlib.md5(r["dem"].encode()).hexdigest() == md5
    assert sum(1 for l in r["dem"].splitlines() if l.startswith("error")) == lines


def test_small_laws_keep_enumerate():
    # d <= 2: the routine never builds a table (the twirl-routes born_sums pins hold: 2^(kappa+1) - 1 node + leaf evals)
    tz0 = "RX 0\nT 0\nRX 1\nCS 1 0\nX_ERROR(0.01) 1\nCS_DAG 1 0\nMX 1\nDETECTOR rec[-1]\n"
    r = census(tz0)
    assert r["ok"] and r["coset_laws"] == 0 and r["coset_work"] == 0
    assert 0 < r["born_sums"] <= 3


# ── RF T4(b): ksec >= 1 circuits (strict contract: output pure, b = e = 0) vs the dense register oracle ─────────────
# (i) one output wire |T>; per coin i a |0> control c_i (X fault -> a CZ residual between the output and the UNMEASURED
# |+> wire r_i) and the CCZ(out, c_i, r_i), the identity noiselessly; M c_i observes the Pauli part only: the Z part of
# the residual flips the unobserved rest stabilizer X_{r_i}: one unobserved coin acting on the output. All coins on the
# SAME output wire: ksec = 1 for every j (the restated fact: the rank of the output-rest coupling, not the coin count).
def t4_same_wire(j, p=0.5):
    L = ["RX 0", "T 0"]
    for i in range(j):
        L += [f"R {2*i+1}", f"RX {2*i+2}", f"X_ERROR({p}) {2*i+1}", f"CCZ 0 {2*i+1} {2*i+2}"]
    for i in range(j):
        L += [f"M {2*i+1}", f"DECISION({i}) rec[-1]"]
    L.append("OUTPUT_QUBITS out 0")
    return "\n".join(L) + "\n"


# (ii) an ENCODED output |T-bar> = alpha|0..0> + beta|1..1> (rep-n, a = 1), one unobserved CZ coin per output wire:
# ksec = j sectors but only 3 nonzero logical classes {I, X-bar, Y-bar}. Route by the cost rule (fix-1 M1: class
# 4^a (256 + |supp|) + 4 · 2^k = 1040 units vs sector 128 · 2^ksec): j <= 3 the sector sum, j >= 4 the class route.
def t4_rep(n, j, p=0.5):
    L = ["RX 0", "T 0", "R " + " ".join(str(w) for w in range(1, n))] + [f"CX 0 {w}" for w in range(1, n)]
    for i in range(j):
        c, r = n + 2 * i, n + 1 + 2 * i
        L += [f"R {c}", f"RX {r}", f"X_ERROR({p}) {c}", f"CCZ {i} {c} {r}"]
    for i in range(j):
        L += [f"M {n + 2 * i}", f"DECISION({i}) rec[-1]"]
    L.append("OUTPUT_QUBITS out " + " ".join(str(w) for w in range(n)))
    return "\n".join(L) + "\n"


def t4_encoded(j, p=0.5):
    return t4_rep(3, j, p)


def _rf_vs_oracle(text, shots=64, seed=3, lever=0):
    # lever (test API): 0 the cost rule, 1 the sector route, 2 the class route whenever computable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
        buf = s.sample_barrier(shots, seed=seed)
        ow = list(s.output_wires())
        rf = RecordFidelity(s)
        TA._rf_class_route(rf._rf, lever)
        F = np.asarray(rf.evaluate(buf))
    rhoB = bare_rho(text, ow)
    assert abs(purity(rhoB) - 1.0) <= 1e-9
    worst = max(abs(F[i] - register_F(rhoB, buf, i, ow)) for i in range(shots))
    return F, rf, worst


@pytest.mark.parametrize("j", [1, 2, 3])
def test_t4_same_wire_ksec_is_one_and_takes_the_sector_route(j):
    F, rf, worst = _rf_vs_oracle(t4_same_wire(j))
    st = rf.stats()
    assert st["route"] == 2 and st["out_a"] == 1
    assert worst <= 1e-12, worst
    assert set(np.round(F, 9)) <= {0.5, 1.0}
    cs = [c for c in rf.content_structures() if c["full"]]
    assert cs and max(c["ksec"] for c in cs) == 1, cs
    assert st["max_sectors"] == 2 and st["class_contents"] == 0 and st["class_rows"] == 0
    assert st["classes_nonzero"] == -1                         # the cost rule keeps the sector route: classes never computed


@pytest.mark.parametrize("lever", [0, 2], ids=["rule", "class"])
@pytest.mark.parametrize("j", [1, 2, 3])
def test_t4_encoded_output_ksec_is_j(j, lever):
    F, rf, worst = _rf_vs_oracle(t4_encoded(j), lever=lever)
    st = rf.stats()
    assert st["route"] == 2 and st["out_a"] == 1
    assert worst <= 1e-12, worst
    cs = [c for c in rf.content_structures() if c["full"]]
    assert cs and max(c["ksec"] for c in cs) == j, cs
    if lever == 0:                                               # 128 · 2^j <= 1040: the sector sum (unchanged bytes)
        assert st["class_contents"] == 0 and st["max_sectors"] == 2 ** j and st["classes_nonzero"] == -1
    else:                                                        # forced: the class route, {I, X-bar, Y-bar}
        assert st["classes_nonzero"] == 3 and st["class_contents"] >= 1 and st["class_rows"] >= 1
        assert all(c["class_route"] for c in cs if c["ksec"] == j)
        assert st["max_sectors"] <= 2                            # no 2^j sector loop ran on those contents
    assert set(np.round(F, 9)) <= {0.5, 1.0}


@pytest.mark.parametrize("j", [4, 5])
def test_t4_rep5_output_takes_the_class_route_by_the_rule(j):
    # rep-5 |T-bar>, j coins: 128 · 2^j > 1040 -> the class route by the cost rule
    F, rf, worst = _rf_vs_oracle(t4_rep(5, j))
    st = rf.stats()
    assert worst <= 1e-12, worst
    cs = [c for c in rf.content_structures() if c["full"]]
    assert cs and max(c["ksec"] for c in cs) == j, cs
    assert st["classes_nonzero"] == 3 and st["class_contents"] >= 1 and st["max_sectors"] <= 2 ** 3
    assert all(c["class_route"] for c in cs if c["ksec"] == j)


# review fix-1 M1 (the reviewer's case): a rep-2 encoded |T>^10 (a = 10, k = 10), X_ERROR(1) coins on output wires 0..15
# (ksec = 16). The class route costs ~3 s (4^10 classes), the sector sum ~0.05 s: the rule must keep the sector route.
# Oracle: the blocks are independent; blocks 0..7 carry a coin on both wires, each coin a fair Z on its wire (CZ to an
# unmeasured |+>), whose product is a fair Z-bar on the block: F = (1/2 (1 + <Z>^2))^8 = 2^-8 on |T> (<Z> = 0).
def t4_rep2_blocks(nb, nc, p=1.0):
    L = []
    for i in range(nb):
        L += [f"RX {2*i}", f"T {2*i}", f"R {2*i+1}", f"CX {2*i} {2*i+1}"]
    nxt, meas = 2 * nb, []
    for w in range(nc):
        c, r = nxt, nxt + 1
        nxt += 2
        L += [f"R {c}", f"RX {r}", f"X_ERROR({p}) {c}", f"CCZ {w} {c} {r}"]
        meas.append(c)
    for i, c in enumerate(meas):
        L += [f"M {c}", f"DECISION({i}) rec[-1]"]
    L.append("OUTPUT_QUBITS out " + " ".join(map(str, range(2 * nb))))
    return "\n".join(L) + "\n"


def test_rep2_t10_ksec16_keeps_the_sector_route():
    text = t4_rep2_blocks(10, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
        buf = s.sample_barrier(20, seed=3)
        rf = RecordFidelity(s)
        t0 = time.perf_counter()
        F = np.asarray(rf.evaluate(buf))
        dt = time.perf_counter() - t0
    st = rf.stats()
    cs = [c for c in rf.content_structures() if c["full"]]
    assert st["out_a"] == 10 and cs and max(c["ksec"] for c in cs) == 16, cs
    assert st["class_contents"] == 0 and st["classes_nonzero"] == -1 and st["max_sectors"] == 2 ** 16
    assert np.all(np.abs(F - 2.0 ** -8) <= 1e-12), F
    assert dt < 2.0, dt                                           # (the class route: ~3 s; the sector sum: ~0.05 s)


@pytest.mark.parametrize("lever", [0, 1, 2], ids=["rule", "sector", "class"])
def test_rep2_blocks_small_vs_the_dense_oracle(lever):
    # the same family at a = 3, ksec = 6 (dense register oracle on 6 wires): every route exact, F = 2^-3
    F, rf, worst = _rf_vs_oracle(t4_rep2_blocks(3, 6), shots=8, lever=lever)
    assert worst <= 1e-12, worst
    assert np.all(np.abs(F - 2.0 ** -3) <= 1e-12)
    assert (rf.stats()["class_contents"] >= 1) == (lever == 2)   # the rule: 4^3 (256 + 8) + 32 > 128 · 2^6


# (iii) the encoded |T-bar> (wires 0 1 2); per coin i an unobserved Bell pair (r_i, r_i') on the rest (stabilizers X X,
# Z Z) and a |0> control c_i whose X fault leaves the residual CZ(w_i, r_i) CZ(w_i', r_i') on TWO different output
# wires: its expansion keeps the lone Z_r term (it flips X_r X_r': a sector, ksec = j) while the classes X-bar / Y-bar
# are dressed with Z_r Z_r' = +1 and stay live; on |000> / |111> the residual is the identity, so F = 1 exactly on every
# shot. For j >= 2 the class route sums three live terms (1 + 1/2 + 1/2) / 2 = 1 — a dropped live class gives 3/4 (the
# CLS_DROP mutation's witness). (With ONE control wire the rest part is the single operator W = prod Z_r: a live class
# needs W in the stabilizer group, which leaves no sector — the engine's ksec = 0 there is right.)
def t4_bell_pair(j, p=0.5):
    L = ["RX 0", "T 0", "R 1 2", "CX 0 1", "CX 0 2"]
    for i in range(j):
        c, r, r2 = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
        L += [f"R {c}", f"RX {r}", f"R {r2}", f"CX {r} {r2}", f"X_ERROR({p}) {c}", f"CCZ {i % 3} {c} {r}", f"CCZ {(i + 1) % 3} {c} {r2}"]
    for i in range(j):
        L += [f"M {3 + 3 * i}", f"DECISION({i}) rec[-1]"]
    L.append("OUTPUT_QUBITS out 0 1 2")
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("j", [2, 3])
def test_t4_bell_pair_coins_class_route_with_live_classes(j):
    F, rf, worst = _rf_vs_oracle(t4_bell_pair(j), lever=2)      # (the rule keeps the sector sum at ksec = 2)
    st = rf.stats()
    assert st["route"] == 2 and st["out_a"] == 1
    assert worst <= 1e-12, worst
    assert np.all(np.abs(F - 1.0) <= 1e-12)
    cs = [c for c in rf.content_structures() if c["full"]]
    # ksec is the RANK of the rest coupling, not the coin count: j = 2 gives 2; the third coin (pair (2, 0)) is dependent
    assert cs and max(c["ksec"] for c in cs) == 2, cs
    assert st["classes_nonzero"] == 3 and st["class_contents"] >= 1 and st["class_rows"] >= 1
    assert all(c["class_route"] for c in cs if c["ksec"] >= 2)


def test_rf_corpus_keeps_the_sector_route():
    # ksec = 0 on the RF corpus (every F byte-identical: the class route is never entered)
    import os
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for name in ("magna_rp2_fid.stim",):
        text = open(os.path.join(DATA, name)).read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = xtim.compile_twirl_sampler(text, selfcheck=0)
            buf = s.sample_barrier(64, seed=5)
            rf = RecordFidelity(s)
            rf.evaluate(buf)
        st = rf.stats()
        assert st["max_ksec"] == 0 and st["class_contents"] == 0 and st["classes_nonzero"] == -1


# threads=: the class route on lanes (whole contents built on a lane in parallel, split contents' chunks on any lane):
# values byte-identical to threads=1 and the class counters folded to the serial loop's (one computation per memo key)
@pytest.mark.parametrize("text,lever", [(t4_rep(5, 5), 0), (t4_bell_pair(2), 2)], ids=["rep5_rule", "bellpair2_forced"])
def test_t4_class_route_identical_on_every_lane(text, lever):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
        buf = s.sample_barrier(6000, seed=11)
        ref = None
        for T in (1, 2, 7, 48):
            rf = RecordFidelity(s)
            TA._rf_class_route(rf._rf, lever)
            F = np.asarray(rf.evaluate(buf, threads=T))
            st = rf.stats()
            assert st["class_contents"] >= 1 and st["class_rows"] >= 1 and st["classes_nonzero"] == 3
            key = {k: st[k] for k in ("misses", "memo_hits", "class_rows", "class_contents", "max_ksec",
                                      "classes_nonzero", "max_sectors", "value_builds")}
            if ref is None:
                ref = (F.tobytes(), key)
                continue
            assert F.tobytes() == ref[0], T
            assert key == ref[1], (T, key, ref[1])
