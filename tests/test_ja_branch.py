"""J-a N7 (J-a-b): the carried-branch scaffolding (cpp/src/branch_scaffold.cpp, qeccore/branch_scaffold.hpp), switched
OFF in production; reached through the test API on a FAM sampler (xtim._xtim_testapi: branch_plan, branch_workload,
alpha_payload_check, plan_payload_fields, emin_categories, bare_frames). Plan §1.1 rows 13-14, §3.8, r2.3, r2.6;
design §2.10; R-JA-ES. The C++ halves are the ctest branch_scaffold.

  G-ES     E_S per plan == Lemma R2' {a : exists j in S, <g_a, B_j> = 1}, recomputed HERE from the plan's term bitset
           and the certified rows (kind 0: the axes re-derived from (a, cz) independently; kind 2: the term view and the
           reduced S); the rows moved by Phi(S) are reported and are a subset (R-JA-ES).
  G-W      on every pivot-bearing plan of FX-xstage (CE1/CE1b/CE2/CE2b and the chains' stage 1) and of the small sets:
           DENSE retained states psi_y = Pi_y C0 |B> (the definition, design §2.6: every record projected at its raw
           outcome; |B> from the sampler's own bare-state export): W_p maps every nonzero sector y onto y + (coin row
           p) with the same probability, |<psi_{y'}| W_p |psi_y>| = 1 (psi_u = P W(u) psi_0). Negative control: the
           undressed g_p fails it on the dressed plans. The exact phase convention (g_p left, -iB_j) and MUT-W-SIGN are
           the ctest's (W_p == C0 g_p C0^dag, dense, exact).
  G-ALPHA  the bare amplitudes moved by 1 ulp in one entry: every compared payload field byte-identical on every plan
           of FX-twin, FX-PPR (diagonal and kind 2) and the factory; only Born-trie node values move (counted > 0 over
           the corpus). Exclusion integrity: every field of plan_payload_fields() is compared or excluded, and the
           exclusions are exactly the r2.3 list; MUT-ALPHA-EXCL ('base' excluded) is refused. MUT-ALPHA-READ (build
           flag) -> red.
  G-DIG    payload builds insert every plan into the E_min = {} table, and every plan resolves by FULL key; a forced
           digest collision is refused by name (C++: DigestCollision; MUT-DIG-NOCMP in the ctest).
  G-LAZY   a sampling workload plus the payload builds of every plan it reached computes E_S 0 times; the branches core
           computes it once per plan. MUT-ES-EAGER (build flag) -> red.
  F3-2     emin_categories over EVERY first-order fault (each noise site outcome) of cult_d3, cult_d5, the factory
           fixture and f3 (|S| > 12 included): exact counters per category (Pauli / dressed -> Pauli frame / LOGICAL /
           EXTENDED), on the canonical (reduced mod ker Phi) term set and on the raw one; the linear-algebra categories
           == a brute-force 2^|S| grouping (audit_emin.py's rule) on every |S| <= 12 fault. Any E_min != {} is reported.
"""
import itertools
import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from _record_law_x import oracle  # noqa: E402
from test_ja_record_space_b import FX_S2B, diag_key, fam, fam_any, n_wires, plan_keys  # noqa: E402
from test_ja_term_key import FX_PPR  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402
from xtim import frames  # noqa: E402
from _private import private_fixture, private_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FX-xstage (xstage/proto2.py CE1..CE2b; proto3.py chains' stage 1)
FX_XSTAGE = {
    "CE1": "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nMX 0\nM 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
    "CE1b": "RX 0\nR 1\nT 0\nDEPOLARIZE1(0.3) 0\nT_DAG 0\nCX 0 1\nMX 0\nM 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
    "CE2": "R 0 1\nH 0\nCX 0 1\nX_ERROR(0.3) 0\nT 0\nT_DAG 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "CE2b": ("R 0 1 2\nH 0\nCX 0 1\nDEPOLARIZE1(0.3) 0\nT 0\nT_DAG 1\nH 2\nCX 2 0\nCX 2 1\nH 2\nM 2\nMX 0 1\n"
             "DETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n"),
    "CE1_chain": "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\n",
    "CE1_chain_mx": "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nMX 0\nDETECTOR rec[-1]\n",
    "CE2_chain": ("R 0 1 2\nH 0\nCX 0 1\nDEPOLARIZE1(0.3) 0\nT 0\nT_DAG 1\nH 2\nCX 2 0\nCX 2 1\nH 2\nM 2\n"
                  "DETECTOR rec[-1]\n"),
}


def _read(rel):
    if rel.startswith("private:"):                      # a private fixture: the calling test skips when absent
        return private_fixture(rel[len("private:"):]).read_text()
    return open(os.path.join(ROOT, rel)).read()


def kind2_sets(s, shots=2000, seed=11):
    """The distinct reduced term sets of the kind-2 shots among `shots` shots."""
    out, seen = [], set()
    for e in s.term_events(shots, seed):
        alts = [int(x) for x in e]
        if not alts or s.diag_shot(alts) is not None:
            continue
        t = s.term_shot(alts)
        if t["kind"] != 2:
            continue
        S = tuple(int(j) for j in t["S_red"])
        if S not in seen:
            seen.add(S)
            out.append(list(S))
    return out


# ── Pauli helpers (x, z bit arrays; phase i^p) ─────────────────────────────────────────────────────
def anti(a, b):
    return (int(np.dot(np.asarray(a[0], int), np.asarray(b[1], int)))
            + int(np.dot(np.asarray(a[1], int), np.asarray(b[0], int)))) % 2


def apply_pauli(psi, n, x, z, ph):
    """i^ph X^x Z^z |psi> on n qubits, qubit 0 = the MOST significant bit (xtim.frames.dense_state)."""
    idx = np.arange(1 << n)
    zm = sum(1 << (n - 1 - q) for q in range(n) if z[q])
    xm = sum(1 << (n - 1 - q) for q in range(n) if x[q])
    par = np.array([bin(int(i) & zm).count("1") & 1 for i in idx])
    out = np.zeros_like(psi)
    out[idx ^ xm] = psi * np.where(par, -1.0, 1.0)
    return out * (1j ** (ph % 4))


def apply_c0(psi, n, d):
    """C0 of the plan: kind 0 = S^{a} . CZ (diag_pullback's C0); kind 2 = prod_{j in S} rot(B_j), rot(B) = (1 + iB)/sqrt2."""
    if d["kind"] == 0:
        idx = np.arange(1 << n)
        bitq = lambda q: (idx >> (n - 1 - q)) & 1  # noqa: E731
        ph = np.ones(1 << n, complex)
        for q in d["a"]:
            ph *= np.where(bitq(q), 1j, 1.0)
        for f, s2 in d["cz"]:
            ph *= np.where(bitq(f) & bitq(s2), -1.0, 1.0)
        return psi * ph
    for j in d["S"]:
        x, z, p = d["axes"][j]
        psi = (psi + 1j * apply_pauli(psi, n, x, z, p)) / math.sqrt(2)
    return psi


def sectors(s, d):
    """{y: psi_y (unnormalized)} over every raw record outcome y with nonzero weight: Pi_y C0 |B>."""
    fr = TA.bare_frames(s)
    n = int(fr["n"])
    psi = apply_c0(frames.dense_state(fr), n, d)
    ops = TA.record_operators(s)
    out = {(): psi}
    for (x, z, p) in ops:
        nxt = {}
        for y, v in out.items():
            Mv = apply_pauli(v, n, x, z, p)
            for b in (0, 1):
                w = (v + (1 - 2 * b) * Mv) / 2
                if np.vdot(w, w).real > 1e-20:
                    nxt[y + (b,)] = w
        out = nxt
    return n, out


def w_check(s, d, frame_key="W"):
    """(sector pairs checked, failures): W_p maps the y-sector onto y + coin row p, same weight, overlap 1."""
    n, sec = sectors(s, d)
    A = np.asarray(d["coin_rows"], np.uint8)
    checked = bad = 0
    for p, (x, z, ph) in enumerate(d[frame_key]):
        col = tuple(int(v) for v in A[:, p])
        for y, v in sec.items():
            y2 = tuple(a ^ b for a, b in zip(y, col))
            w = sec.get(y2)
            checked += 1
            if w is None:
                bad += 1
                continue
            nv, nw = np.vdot(v, v).real, np.vdot(w, w).real
            ov = abs(np.vdot(w, apply_pauli(v, n, x, z, ph))) / math.sqrt(nv * nw)
            if abs(nv - nw) > 1e-12 or abs(ov - 1.0) > 1e-9:
                bad += 1
    return checked, bad


# ── G-ES ─────────────────────────────────────────────────────────────────────────────────────────────
def kind0_axes(a, cz, n):
    """diag_plan_axes of (a, cz), re-derived: Z_q per a-wire; Z_f, Z_s, Z_f Z_s per CZ pair (signs irrelevant here)."""
    ax = []
    for q in a:
        z = np.zeros(n, np.uint8); z[q] = 1; ax.append(z)
    for f, s2 in cz:
        for qs in ((f,), (s2,), (f, s2)):
            z = np.zeros(n, np.uint8)
            for q in qs:
                z[q] = 1
            ax.append(z)
    return ax


def gens_arrays(gens):
    return (np.array([np.asarray(g[0], np.int64) for g in gens]), np.array([np.asarray(g[1], np.int64) for g in gens]))


def es_by_definition(G, axes, S):
    """{a : exists j in S, <g_a, B_j> = 1}; G = (gx, gz) arrays, axes = [(x, z)]."""
    S = list(S)
    if not S:
        return set()
    bx = np.array([np.asarray(axes[j][0], np.int64) for j in S])
    bz = np.array([np.asarray(axes[j][1], np.int64) for j in S])
    A = (G[0] @ bz.T + G[1] @ bx.T) % 2                 # ng x |S|
    return set(int(a) for a in np.nonzero(A.any(axis=1))[0])


def apply_c0_dag(psi, n, d):
    """C0^dag: kind 0 = the conjugate phases; kind 2 = prod (1 - iB_j)/sqrt2 (the axes commute)."""
    if d["kind"] == 0:
        idx = np.arange(1 << n)
        bitq = lambda q: (idx >> (n - 1 - q)) & 1  # noqa: E731
        ph = np.ones(1 << n, complex)
        for q in d["a"]:
            ph *= np.where(bitq(q), -1j, 1.0)
        for f, s2 in d["cz"]:
            ph *= np.where(bitq(f) & bitq(s2), -1.0, 1.0)
        return psi * ph
    for j in d["S"]:
        x, z, p = d["axes"][j]
        psi = (psi - 1j * apply_pauli(psi, n, x, z, p)) / math.sqrt(2)
    return psi


def moved_dense(n, d, gens, rng):
    """The rows g with C0^dag g C0 != +-g, by DENSE action on a random vector (independent of E_S's algebra)."""
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    out = set()
    for a, (x, z, p) in enumerate(gens):
        w = apply_c0_dag(apply_pauli(apply_c0(v, n, d), n, x, z, p), n, d)
        g = apply_pauli(v, n, x, z, p)
        if not (np.allclose(w, g, atol=1e-9) or np.allclose(w, -g, atol=1e-9)):
            out.add(a)
    return out


def g_es_circuit(text, kind2=True, shots=2000):
    s = fam_any(text)
    if s is None:
        return None
    n = n_wires(s)
    gl = TA.certified_generators(s)
    G = gens_arrays(gl)
    rng = np.random.default_rng(5)
    dense_ok = n <= 14
    r = {"plans": 0, "es": 0, "moved": 0, "moved_dense": 0, "bad": []}

    def check_moved(d, tag):
        if not dense_ok:
            return
        md = moved_dense(n, d, gl, rng)
        r["moved_dense"] += 1
        if set(d["moved"]) != md:
            r["bad"].append(("moved != dense", tag, sorted(md), d["moved"]))
    for pk in plan_keys(s, shots):
        try:
            d = TA.branch_plan(s, key=pk)
        except ValueError:
            continue                                    # a fallback plan
        ax = kind0_axes(d["a"], d["cz"], n)
        got_axes = [np.asarray(z, np.uint8) for (_, z, _) in d["axes"]]
        if [tuple(a) for a in ax] != [tuple(a) for a in got_axes]:
            r["bad"].append(("axes", pk.hex()[:40]))
        full = [(np.zeros(n, np.uint8), z) for z in ax]
        want = es_by_definition(G, full, range(len(full)))
        r["plans"] += 1
        r["es"] += len(want)
        r["moved"] += len(d["moved"])
        if set(d["es"]) != want:
            r["bad"].append(("E_S", pk.hex()[:40], sorted(want), d["es"]))
        if not set(d["moved"]) <= set(d["es"]):
            r["bad"].append(("moved not in E_S", pk.hex()[:40]))
        check_moved(d, pk.hex()[:40])
    if kind2:
        for S in kind2_sets(s, shots):
            d = TA.branch_plan(s, S=S)
            full = [(np.asarray(x, np.uint8), np.asarray(z, np.uint8)) for (x, z, _) in d["axes"]]
            want = es_by_definition(G, full, d["S"])
            r["plans"] += 1
            r["es"] += len(want)
            r["moved"] += len(d["moved"])
            if set(d["es"]) != want or d["S"] != sorted(S):
                r["bad"].append(("E_S kind 2", S, sorted(want), d["es"]))
            if not set(d["moved"]) <= set(d["es"]):
                r["bad"].append(("moved not in E_S (kind 2)", S))
            check_moved(d, ("S", tuple(S)))
    return r


GES_SETS = {"FX-S2b": FX_S2B, "FX-xstage": FX_XSTAGE, "FX-PPR": FX_PPR}


@pytest.mark.parametrize("setname", sorted(GES_SETS))
def test_ges_definition(setname):
    tot = {"plans": 0, "es": 0, "moved": 0, "moved_dense": 0}
    bad = {}
    for name, text in sorted(GES_SETS[setname].items()):
        r = g_es_circuit(text)
        if r is None:
            continue
        for k in tot:
            tot[k] += r[k]
        if r["bad"]:
            bad[name] = r["bad"][:3]
    print(f"\n[G-ES] {setname}: {tot['plans']} plans, sum |E_S| = {tot['es']}, sum |moved| = {tot['moved']} "
          f"(moved == DENSE C0^dag g C0 != +-g on {tot['moved_dense']} plans), disagreements {len(bad)}")
    assert not bad, bad
    assert tot["plans"] > 0 and tot["es"] > 0 and tot["moved_dense"] > 0


@pytest.mark.parametrize("name", ["fac", "d5"])
def test_ges_speed_set(name):
    text = _read({"fac": "private:ccz_factory_a.stim", "d5": "tests/data/cultivation_d5.stim"}[name])
    r = g_es_circuit(text, kind2=False, shots=400)
    print(f"\n[G-ES] {name}: {r['plans']} plans, sum |E_S| = {r['es']}, sum |moved| = {r['moved']} (reported; the "
          f"dense moved check needs n <= 14), disagreements {len(r['bad'])}")
    assert not r["bad"], r["bad"][:3]
    assert r["es"] >= r["moved"] and r["plans"] > 1


# ── G-W ──────────────────────────────────────────────────────────────────────────────────────────────
GW_SETS = dict(FX_XSTAGE)
GW_SETS.update({k: FX_S2B[k] for k in ("a4_comb", "sig_ovl", "sig_comb", "span_obs_dec", "wlam", "ccz_obs")})


@pytest.mark.parametrize("name", sorted(GW_SETS))
def test_gw_dense(name):
    s = fam_any(GW_SETS[name])
    if s is None:
        pytest.skip("refused by name")
    if n_wires(s) > 16:
        pytest.skip("dense oracle: n > 16")
    plans = piv = checked = 0
    neg_fail = neg_checked = 0
    plans_all = [("k", pk) for pk in plan_keys(s, 2000)] + [("S", S) for S in kind2_sets(s, 2000)]
    for kind, v in plans_all:
        try:
            d = TA.branch_plan(s, key=v) if kind == "k" else TA.branch_plan(s, S=v)
        except ValueError:
            continue
        if not d["W"]:
            continue
        plans += 1
        piv += len(d["W"])
        c, bad = w_check(s, d)
        checked += c
        assert bad == 0, (name, kind, d["S"], bad, c)
        if d["es"]:                                      # dressed: the undressed g_p must fail somewhere
            c2, bad2 = w_check(s, d, "g")
            neg_checked += c2
            neg_fail += bad2
    print(f"\n[G-W] {name}: {plans} pivot-bearing plans, {piv} pivots, {checked} sector maps W_p checked vs dense; "
          f"negative control (undressed g_p) failed {neg_fail}/{neg_checked}")


def test_gw_coverage_and_negative_control():
    """The FX-xstage set reaches dressed pivot-bearing plans, and on them the undressed g_p is NOT the sector map."""
    plans = neg_fail = 0
    for name, text in sorted(FX_XSTAGE.items()):
        s = fam_any(text)
        for pk in plan_keys(s, 2000):
            try:
                d = TA.branch_plan(s, key=pk)
            except ValueError:
                continue
            if d["W"] and d["es"]:
                plans += 1
                neg_fail += w_check(s, d, "g")[1]
    print(f"\n[G-W] FX-xstage dressed pivot-bearing plans {plans}; undressed-g_p failures {neg_fail}")
    assert plans > 0 and neg_fail > 0


# ── G-ALPHA ──────────────────────────────────────────────────────────────────────────────────────────
EXCLUDED = ["rec_trie.p0", "rec_trie.children", "reader_memo.values", "signs.value_cache",
            "ppr.fold_p1"]
LAW_FIELDS_PREFIX = ("header.", "rec.", "law.", "branch.")   # J-b2 J7b: sigma.* -> law.* (det_signs, kernel_logicals)


def check_exclusions(excluded):
    """r2.3 exclusion integrity: every payload field is compared or excluded, never both, and no law field (header,
    RecTable, sigma law, E_S, W_p, term images, table index) may be excluded."""
    f = TA.plan_payload_fields()
    comp, excl_api = list(f["compared"]), list(f["excluded"])
    for x in excluded:
        if x.startswith(LAW_FIELDS_PREFIX) or x in comp:
            raise ValueError(f"G-ALPHA exclusion integrity: '{x}' is a law field and may not be excluded")
    if sorted(excluded) != sorted(excl_api):
        raise ValueError(f"G-ALPHA exclusion integrity: the exclusion list {sorted(excluded)} != the payload "
                         f"accessor's {sorted(excl_api)}")
    return comp


def test_galpha_exclusion_integrity():
    comp = check_exclusions(EXCLUDED)
    assert "rec.base" in comp and "branch.E_S" in comp and "branch.W_p" in comp


def test_galpha_mut_alpha_excl_refused():
    """MUT-ALPHA-EXCL: 'base' added to the exclusion list is refused (it is a law field)."""
    with pytest.raises(ValueError, match="law field"):
        check_exclusions(EXCLUDED + ["rec.base"])


GALPHA_SETS = {"FX-twin": F.twins(), "FX-PPR": FX_PPR}


@pytest.mark.parametrize("setname", sorted(GALPHA_SETS) + ["factory"])
def test_galpha_payload_never_reads_alpha(setname):
    comp = check_exclusions(EXCLUDED)
    texts = ({"fac": _read("private:ccz_factory_a.stim")} if setname == "factory"
             else GALPHA_SETS[setname])
    plans = nodes = moved = 0
    diffs = {f: 0 for f in comp}
    for name, text in sorted(texts.items()):
        s = fam_any(text)
        if s is None:
            continue
        keys = plan_keys(s, 2000 if setname != "factory" else 600)
        r = TA.alpha_payload_check(s, keys, kind2_sets(s, 2000) if setname == "FX-PPR" else [], 1)
        plans += r["plans"]
        nodes += r["trie_nodes"]
        moved += r["trie_moved"]
        for f, v in r["field_diffs"].items():
            diffs[f] += v
    print(f"\n[G-ALPHA] {setname}: {plans} plans, payload field diffs {sum(diffs.values())} over {len(diffs)} fields; "
          f"trie nodes {nodes}, moved {moved}")
    assert set(diffs) == set(comp)
    assert sum(diffs.values()) == 0, {f: v for f, v in diffs.items() if v}
    assert plans > 0


def fam_any(text):
    try:
        return fam(text)
    except ValueError:
        return None


@pytest.mark.parametrize("setname", sorted(GALPHA_SETS) + ["factory"])
def test_galpha_production_builders_twin(setname):
    """The PRODUCTION plan builders (plan_structure = build_shot_law + build_rec_; family_general_table = (G)) of the
    sampler vs a twin sampler whose bare amplitudes moved by 1 ulp: every law / table byte equal on every plan."""
    texts = ({"fac": _read("private:ccz_factory_a.stim")} if setname == "factory"
             else GALPHA_SETS[setname])
    plans = diffs = moved = 0
    for name, text in sorted(texts.items()):
        s = fam_any(text)
        if s is None:
            continue
        keys = plan_keys(s, 2000 if setname != "factory" else 600)
        r = TA.alpha_twin_check(s, text, keys, kind2_sets(s, 2000) if setname == "FX-PPR" else [], 1)
        plans += r["plans"]
        diffs += r["diffs"]
        moved += r["amplitudes_moved"]
    print(f"\n[G-ALPHA twin] {setname}: {plans} plans through the production builders, byte diffs {diffs}; "
          f"amplitude entries moved in the twins {moved}")
    assert diffs == 0 and plans > 0 and moved > 0


def test_galpha_perturbation_is_live():
    """Non-vacuity: the 1-ulp perturbation reaches Born-trie node values somewhere in the corpus."""
    moved = 0
    for text in list(F.twins().values()) + [FX_S2B[k] for k in ("a4_comb", "a4_comb_rev", "dec_first_minus")]:
        s = fam_any(text)
        if s is not None:
            moved += TA.alpha_payload_check(s, plan_keys(s, 400), [], 1)["trie_moved"]
    assert moved > 0


# ── G-DIG / G-LAZY ───────────────────────────────────────────────────────────────────────────────────
LAZY_SET = {"fac": private_text("ccz_factory_a.stim"), "a4_comb": FX_S2B["a4_comb"],
            "CE2b": FX_XSTAGE["CE2b"]}


@pytest.mark.parametrize("name", sorted(LAZY_SET))
def test_glazy_and_gdig(name):
    if LAZY_SET[name] is None:
        private_fixture("ccz_factory_a.stim")
    s = fam(LAZY_SET[name])
    s.sample(500, seed=3)                               # a sampling workload (no materialize, no branches core)
    keys = plan_keys(s, 2000 if name != "fac" else 600)
    r0 = TA.branch_workload(s, keys, False, False)      # payload builds only
    r1 = TA.branch_workload(s, keys, True, False)       # + the branches core
    rc = TA.branch_workload(s, keys, False, True)       # a forced digest collision
    print(f"\n[G-LAZY/G-DIG] {name}: {r0['plans']} plans; E_S computations: payload builds {r0['es_computations']}, "
          f"with the branches core {r1['es_computations']}; resolved {r0['resolved']}; forced collision refused "
          f"{rc['refused']}")
    assert r0["es_computations"] == 0 and r0["es_after_payload_builds"] == 0
    assert r1["es_after_payload_builds"] == 0 and r1["es_computations"] == r1["plans"]
    # every key resolves (by full key) to the plan it was inserted as; keys with one content key share one plan
    assert r0["resolved"] == r0["plans"] and 0 < r0["distinct"] <= r0["plans"]
    assert r0["mismatched"] == 0 and r0["refused"] == 0
    assert r0["tables"] == 1                            # E_min = {} for every plan today
    if rc["plans"] > 1:
        assert rc["refused"] > 0 and rc["mismatched"] == 0


# ── F3-2 ─────────────────────────────────────────────────────────────────────────────────────────────
F32 = {"cult_d3": "tests/data/cultivation_d3_faithful.stim", "cult_d5": "tests/data/cultivation_d5.stim",
       "factory": "private:ccz_factory_a.stim", "f3": "private:ccz_factory_b.stim"}
# the xstage audit's own d3 circuit (cultivation_d3_rate.stim is Clifford-only: no terms), when the ledger tree has it
_CD = os.environ.get("LEDGER_CIRC_DIR")
if _CD and os.path.exists(os.path.join(_CD, "stateroute", "circ", "cult_d3_ts.stim")):
    F32["cult_d3_ts"] = os.path.join(_CD, "stateroute", "circ", "cult_d3_ts.stim")


def brute_category(S, acert, sig):
    """audit_emin.py's rule on 2^|S| components: group by record signature; any group >= 2 -> LOGICAL unless its
    members differ on a certified row (EXTENDED); all groups of size 1 -> Pauli frame."""
    if not S:
        return "pauli"
    groups = {}
    for r in range(len(S) + 1):
        for T in itertools.combinations(S, r):
            x = 0
            sg = 0
            for j in T:
                x ^= acert[j]
                sg ^= sig[j]
            groups.setdefault(sg, []).append(x)
    cat = "pauli_frame"
    for g in groups.values():
        if len(g) < 2:
            continue
        if any(v != g[0] for v in g):
            return "extended"
        cat = "logical"
    return cat


def term_patterns(s):
    """Per term j: its anticommutation pattern over the certified rows and its record signature, as Python ints."""
    gx, gz = gens_arrays(TA.certified_generators(s))
    axes = np.asarray(s.term_axes()[0], np.int64)
    n = axes.shape[1] // 2
    A = (gx @ axes[:, n:].T + gz @ axes[:, :n].T) % 2   # ng x T
    ng = A.shape[0]
    Mk = np.zeros((0, ng), np.int64)
    rows = []
    for m in TA.record_channel_masks(s):
        bits = np.unpackbits(np.asarray(m, np.uint64).view(np.uint8), bitorder="little")[:ng]
        rows.append(np.pad(bits, (0, ng - len(bits))).astype(np.int64))
    if rows:
        Mk = np.array(rows)
    SG = (Mk @ A) % 2                                   # nch x T
    ops = TA.record_operators(s)
    ex = [(np.asarray(x, np.int64), np.asarray(z, np.int64)) for (x, z, _) in ops[len(rows):]]
    if ex:                                              # the extras: their own operators vs each axis
        EX = np.array([np.concatenate([z, x]) for (x, z) in ex])        # <M, B> = M.x.B.z + M.z.B.x
        SG = np.vstack([SG, (EX @ axes.T) % 2])
    to_int = lambda col: int("".join(str(int(b)) for b in col[::-1]) or "0", 2)  # noqa: E731
    return [to_int(A[:, j]) for j in range(A.shape[1])], [to_int(SG[:, j]) for j in range(SG.shape[1])]


@pytest.mark.parametrize("name", sorted(F32))
def test_f32_emin_categories(name):
    text = _read(F32[name])
    s = fam(text)
    o = oracle(text)
    xz, ph, _ = o.terms()
    axes = np.asarray(s.term_axes()[0], np.uint8)
    assert np.array_equal(np.asarray(xz, np.uint8), axes), "XO's terms != the sampler's term view"
    faults, probs = [], []
    for site in o.noise_sites():
        for outs, pr in zip(site["outcomes"], site["probs"]):
            if not any(outs):
                continue
            S, _px, _pz = o.residual([(site["op_pos"], site["qubits"], list(outs))], None)
            faults.append([j for j in range(len(S)) if S[j]])
            probs.append(pr)
    red = TA.emin_categories(s, faults, True)
    raw = TA.emin_categories(s, faults, False)
    tot = sum(probs)
    cnt, mass, big = {}, {}, {"n": 0, "mass": 0.0}
    ext = []
    for S, pr, (cat, ns, nred, dimk, erows, Jred) in zip(faults, probs, red):
        cnt[cat] = cnt.get(cat, 0) + 1
        mass[cat] = mass.get(cat, 0.0) + pr / tot
        if ns > 12:
            big["n"] += 1
            big["mass"] += pr / tot
        if cat == "extended":
            ext.append((S, erows))
    # the brute-force grouping (audit_emin.py's rule) on every raw |S| <= 12 fault equals the linear algebra
    acert, sig = term_patterns(s)
    nb = 0
    rawmass = {}
    for S, pr, (cat, ns, nred, dimk, erows, _J) in zip(faults, probs, raw):
        key = cat if ns <= 12 else "|S| > 12"
        rawmass[key] = rawmass.get(key, 0.0) + pr / tot
        if 0 < len(S) <= 12:
            assert brute_category(S, acert, sig) == cat, (name, S, cat)
            nb += 1
    nbr = 0                                             # the canonical (reduced) path, brute-forced too
    for S, (cat, ns, nred, dimk, erows, Jred) in zip(faults, red):
        if 0 < len(Jred) <= 12:
            assert brute_category(list(Jred), acert, sig) == cat, (name, S, Jred, cat)
            nbr += 1
    print(f"\n[F3-2] {name}: raw-S categories, audit_emin.py's form (|S| > 12 not enumerated there): "
          + "; ".join(f"{k}: {v:.4f}" for k, v in sorted(rawmass.items())))
    print(f"\n[F3-2] {name}: {len(faults)} first-order faults (mass {tot:.4g}); |S| > 12: {big['n']} faults, "
          f"mass fraction {big['mass']:.4f}; categories (canonical S): "
          + "; ".join(f"{k}: {cnt[k]} faults / {mass[k]:.4f}" for k in sorted(cnt))
          + f"; EXTENDED (E_min != {{}}): {len(ext)}; brute force == linear algebra on {nb} raw and {nbr} canonical "
          f"|S| <= 12 faults")
    # plan §3.8: any E_min != {} is reported, STOP-and-report if it changes the XQ5 / Y-3 cost picture
    assert not ext, f"STOP-and-report (F3-2): {len(ext)} first-order faults with E_min != {{}} on {name}: {ext[:10]}"
