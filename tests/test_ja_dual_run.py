"""J-a N3: the record-table builder (G) (cpp/src/record_general.cpp, NULL FORM: add_extra over every record column
in record order from an empty table, term_pullback for the pullback, then canonicalize_record_order) and its
dual-run harness, plan §3.3 (G-9a, G-9b, G-9c) and §3.10 (the null form's operation counts).

(G) is switched OFF in production: it is reached only through the FAM sampler test API
(xtim._xtim_testapi.compile_family_sampler):
  * plan_structure(pk, force_general=True) — (G)'s table of a DIAGONAL plan key (the plan's C0 = S^{a&1} CZ written
    as terms, record_general.hpp diag_plan_axes); plan_structure(pk) is (D)'s table of the same plan;
  * term_plan_structure(S, born_law) — (G)'s table of C0 = rot(S) for a term set S of the sampler's term view (the
    N2 term key's plan: term_shot()['S_red']), with p_z (the directions' Born law by the unchanged BornTrie node
    definition) on request;

Gates:
  G-9a  (D) vs (G) FIELD identity (pivot_rec, born_rec, A, B, beta, Lambda as (a, b, EFFECTIVE phase), plan §5
        Q12 / L-3) on every diagonal plan reached by 2k shots of FX-S2b (+ the sigma-4 set), FX-twin, FX-design,
        FX-fuzz and the J-b1 list; disagreements = 0 (exact counter). Plus (G) vs (X) through compare_tables on
        the S2b fixture set ("field identity on the S2b fixture set through (G)").
  G-9b  (G) vs (X) record_oracle::plan_law, the same fields (FreeDir conversion of record_table.hpp) and the Born
        law p0 (1e-12), for every plan of FX-PPR and FX-fuzz, DIAGONAL AND KIND 2 (S as composed, S reduced, and
        every (X) fault configuration of the small circuits), and the KF fixtures.
  G-9c  (G) vs the PPR family: RETIRED at J-b2 J2 with the PPR family's test-API leg (ppr_law); its KF fixtures moved
        into G-9b (G vs (X)), with a kind-2 kappa_rec >= 1 coverage assertion.
  OPS   the null form's elimination work per plan build: plane = pivot reduction = m, G.reduce = echelon = m - rho
        (and (D)'s extras: plane = pivot reduction = extras, G.reduce = echelon).
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from _record_law_x import fault_configs, random_circuit  # noqa: E402
from test_ja_term_key import FX_PPR, rot_image, xo_axes  # noqa: E402
from test_s2b_record_table import FIXTURES, KIND, dressed_circuit, family_circuit  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = 2000
FIELDS = ("rho", "kappa_rec", "pivot_rec", "born_rec", "rec_A", "rec_B", "rec_beta", "rec_lam", "rec_lam_phase")


def _sigma_set():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "s2b"))
    from test_sigma_joint_law import FIXTURES as SF
    return {k: v for k, v in SF.items() if k not in FIXTURES}


def _example(name):
    return open(os.path.join(ROOT, "tests", "data", name)).read()


# ── fixture sets (plan §3.0) ─────────────────────────────────────────────────────────────────────────
FX_S2B = dict(FIXTURES)
FX_S2B.update({"sigma." + k: v for k, v in _sigma_set().items()})
FX_S2B.update({"F1." + k: v for k, v in F.FX_F1.items()})
FX_TWIN = F.twins()
FX_DESIGN = dict(F.DESIGN)
FX_DESIGN.update({"gate1": F.GATE1, "dup_gauge": F.DUP_GAUGE})
FX_DESIGN.update({"ah11." + k: v for k, v in F.AH11.items()})
FX_DESIGN.update({"conv." + k: v for k, v in F.CONVENTION.items()})
# the J-b1 list (code_switching_faithful, cube_ccz, the correlated-ANTI Bell pair, q3_dec, m3_obs_2dec, the
# Bell / teleport decision fixtures); e10 lives outside the repository (the ledger's @rule tree) and is not used.
JB1 = {"code_switching_faithful": _example("code_switching_faithful.stim"), "cube_ccz": _example("cube_ccz.stim"),
       "bell_anti": F.DESIGN["bell_anti"], "q3_dec": FIXTURES["q3_dec"], "m3_obs_2dec": FIXTURES["m3_obs_2dec"],
       "teleport": F.DESIGN["teleport"], "teleport_t": F.DESIGN["teleport_t"]}


def fx_fuzz():
    out = {}
    for sd in range(40):
        out[f"rc{sd}"] = random_circuit(sd)[0]
    for sd in range(12):
        out[f"family{sd}"] = family_circuit(sd)
        out[f"dressed{sd}"] = dressed_circuit(sd)
    return out


FX_FUZZ = fx_fuzz()
SETS = {"FX-S2b": FX_S2B, "FX-twin": FX_TWIN, "FX-design": FX_DESIGN, "J-b1": JB1, "FX-fuzz": FX_FUZZ,
        "FX-PPR": FX_PPR}


def fam_any(text):
    """The FAM sampler of `text`, else None (refused by name)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TA.compile_family_sampler(text, selfcheck=0)
    except ValueError:
        return None


def n_wires(s):
    return np.asarray(s.term_axes()[0]).shape[1] // 2


def diag_key(n, nf):
    """plan_structure's key bytes (identity prefix) of a diagonal normal form {a, cz}."""
    nw = (n + 63) // 64
    a = set(nf["a"])
    out = b"\0" * (16 * nw) + bytes(1 if q in a else 0 for q in range(n)) + b"\xff"
    for f, s2 in nf["cz"]:
        out += int(f).to_bytes(2, "little") + int(s2).to_bytes(2, "little")
    return out


def shots_of(s, shots=SHOTS, seed=11):
    seen, out = set(), []
    for e in s.term_events(shots, seed):
        t = tuple(int(x) for x in e)
        if t and t not in seen:
            seen.add(t)
            out.append(list(t))
    return out


def table_diffs(a, b):
    return [f for f in FIELDS if not np.array_equal(np.asarray(a[f]), np.asarray(b[f]))]


def check_ops_g(g):
    o, m, rho = g["ops"], g["m"], g["rho"]
    return (o["plane"] == m and o["pivot_reduce"] == m and o["greduce"] == m - rho and o["echelon"] == m - rho
            and o["extras"] == m)


def check_null_form(g):
    """N3-D3 (the shared-routine mutations M-BETA / MP-1b cannot reach (G)): every (G) build is the null form whose
    canonical introducers are single input directions — no folded beta' (folds == 0) and |b| = 1 for every
    introducer (intro_b: per canonical direction, the number of input directions its literal product takes)."""
    return g["folds"] == 0 and [int(b) for b in g["intro_b"]] == [1] * g["kappa_rec"]


def check_ops_d(d):
    o = d["ops"]
    return (o["plane"] == o["extras"] and o["pivot_reduce"] == o["extras"] and o["greduce"] == o["echelon"]
            and o["greduce"] <= o["extras"])


# ── G-9a ─────────────────────────────────────────────────────────────────────────────────────────────
def g9a_circuit(text):
    """(plans compared, (D) fallback plans, [disagreement descriptions]) over the diagonal plans reached by SHOTS
    shots (+ the identity plan)."""
    s = fam_any(text)
    if s is None:
        return None
    n = n_wires(s)
    keys = {b""}
    for alts in shots_of(s):
        d = s.diag_shot(alts)
        if d is None:
            t = s.term_shot(alts)
            if t["kind"] != 0:
                continue
            d = t["nf"]
        keys.add(diag_key(n, d))
    cmp_, fb, bad = 0, 0, []
    for pk in sorted(keys):
        d = s.plan_structure(pk)
        if d["fallback"] or not d["rec_built"]:
            fb += 1
            continue
        g = s.plan_structure(pk, force_general=True)
        cmp_ += 1
        diff = table_diffs(d, g)
        if diff:
            bad.append(("fields", pk.hex(), diff))
        if not check_ops_g(g):
            bad.append(("G ops", pk.hex(), g["ops"], g["m"], g["rho"]))
        if not check_null_form(g):
            bad.append(("G null form", pk.hex(), g["folds"], list(g["intro_b"])))
        if not check_ops_d(d):
            bad.append(("D ops", pk.hex(), d["ops"]))
    return cmp_, fb, bad


@pytest.mark.parametrize("setname", ["FX-S2b", "FX-twin", "FX-design", "J-b1", "FX-fuzz", "FX-PPR"])
def test_g9a_d_vs_g_field_identity(setname):
    plans = refused = fbs = 0
    bad = {}
    for name, text in sorted(SETS[setname].items()):
        r = g9a_circuit(text)
        if r is None:
            refused += 1
            continue
        c, fb, b = r
        plans += c
        fbs += fb
        if b:
            bad[name] = b[:3]
    print(f"\n[G-9a] {setname}: {len(SETS[setname])} circuits ({refused} refused by name), {plans} diagonal plans "
          f"compared, {fbs} (D)-fallback plans, disagreements {len(bad)}")
    assert not bad, bad
    assert plans >= len(SETS[setname]) - refused


@pytest.mark.parametrize("name", sorted(FX_S2B))
def test_g9a_s2b_fixtures_through_g_equal_x(name):
    """"field identity on the S2b fixture set through (G)": every plan of every fault configuration, (G) vs (X)."""
    assert F.compare_tables(FX_S2B[name], force_general=True) > 0


# ── G-9b ─────────────────────────────────────────────────────────────────────────────────────────────
def is_kind0(B, S, n):
    for q in range(n):
        im = rot_image(B, S, (0, 1 << q, 0))
        if im[0] != 0 or im[1] != 1 << q:
            return False
    return True


def g9b_circuit(text, max_configs=4096, stats=None):
    s = fam_any(text)
    if s is None:
        return None
    orc = TA.RecordLawOracle(text)
    n, T = orc.n, orc.num_terms
    xz, ph = s.term_axes()
    oxz, oph, _ = orc.terms()
    assert np.array_equal(np.asarray(xz), np.asarray(oxz)) and [int(p) % 4 for p in ph] == \
        [int(p) % 4 for p in oph], "the sampler's term view differs from (X)'s"
    cols = [(KIND[k], i) for k, i, _ in orc.columns()]
    order = [(o >> 24, o & ((1 << 24) - 1)) for o in s.plan_structure(b"")["rec_order"]]
    if sorted(order) != sorted(cols):
        return "columns"                                # the sampler does not carry every (X) column
    ocol = {kc: i for i, kc in enumerate(cols)}
    st2or = [ocol[o] for o in order]
    sets = {(): None}
    for alts in shots_of(s):
        t = s.term_shot(alts)
        sets[tuple(t["S"])] = None
        sets[tuple(t["S_red"])] = None
    try:
        for _, faults in fault_configs(orc, max_configs):
            S, _, _ = orc.residual(faults, None)
            sets[tuple(int(j) for j in np.flatnonzero(np.asarray(S)))] = None
    except ValueError:
        pass                                            # too many configurations: the sampled plans only
    B = xo_axes(orc)
    k0 = k2 = 0
    bad = []
    for S in sets:
        Sv = np.zeros(T, np.uint8)
        Sv[list(S)] = 1
        g = s.term_plan_structure(list(S), born_law=True)
        L = orc.law(Sv)
        if is_kind0(B, S, n):
            k0 += 1
        else:
            k2 += 1
            if stats is not None and g["kappa_rec"] >= 1:
                stats["k2_kappa_pos"] = stats.get("k2_kappa_pos", 0) + 1
        why = []
        if g["rho"] != L["rho"] or g["kappa_rec"] != L["kappa"]:
            why.append("rho/kappa")
        else:
            if [st2or[c] for c in g["pivot_rec"]] != list(L["pivots"]):
                why.append("pivots")
            if [st2or[c] for c in g["born_rec"]] != list(L["born"]):
                why.append("born")
            A, Bm, be = np.asarray(g["rec_A"]), np.asarray(g["rec_B"]), np.asarray(g["rec_beta"])
            LA, LB, Lb = np.asarray(L["A"]), np.asarray(L["B"]), np.asarray(L["beta"])
            for c in range(len(order)):
                oc = st2or[c]
                if not np.array_equal(A[c], LA[oc]):
                    why.append(("A", c))
                if not np.array_equal(Bm[c], LB[oc]):
                    why.append(("B", c))
                if int(be[c]) != int(Lb[oc]):
                    why.append(("beta", c))
            if L["kappa"]:
                if not np.array_equal(np.asarray(g["rec_lam"]), np.asarray(L["lam"])):
                    why.append("lam")
                if [int(x) for x in g["rec_lam_phase"]] != [int(x) for x in L["lam_phase"]]:
                    why.append("lam phase")
            if not np.allclose(np.asarray(g["p_z"]), np.asarray(L["p0"]), atol=1e-12, rtol=0):
                why.append("p0")
        if not check_ops_g(g):
            why.append(("ops", g["ops"]))
        if not check_null_form(g):
            why.append(("null form", g["folds"], list(g["intro_b"])))
        if why:
            bad.append((S, why[:4]))
    return k0, k2, bad


# PPR plans WITH foldable kernel directions (FX-PPR reaches none): the qec_library fallback_ppr_kf2 benchmark
# (tests/test_twirl_audit_sampling.py KF2_TXT: two sandwiched CH Hadamard tests; firing both X faults composes to
# kappa_fold = 2, the PPR family's scope fallback) and a variant whose two controls are entangled (CX after T).
KF = {"kf2": ("H 1\nT 1\nH 0\nCH 1 0\nX_ERROR(0.35) 0\nCH 1 0\nH 3\nT 3\nH 2\nCH 3 2\nX_ERROR(0.35) 2\nCH 3 2\n"
              "H 4\nX_ERROR(0.05) 4\nMX 0\nMX 2\nMX 4\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"),
      "kf2_cx": ("H 1\nT 1\nH 3\nT 3\nCX 1 3\nH 0\nCH 1 0\nX_ERROR(0.35) 0\nCH 1 0\nH 2\nCH 3 2\nX_ERROR(0.35) 2\n"
                 "CH 3 2\nMX 0\nMX 2\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")}
# J-b2 J2 (review M4.1): G-9b carries the KF fixtures too (the only PPR-class plans with foldable kernel directions;
# G-9c, (G) vs the PPR family, retired with the PPR family's test-API leg).
G9B = dict(FX_PPR)
G9B.update(FX_FUZZ)
G9B.update(KF)


@pytest.mark.parametrize("name", sorted(G9B))
def test_g9b_g_vs_x_every_plan(name):
    r = g9b_circuit(G9B[name])
    if r is None or r == "columns":
        pytest.skip("refused by name" if r is None else "the FAM sampler does not carry every (X) column")
    k0, k2, bad = r
    print(f"\n[G-9b] {name}: kind-0 plans {k0}, kind-2 plans {k2}, disagreements {len(bad)}")
    assert not bad, bad[:3]
    assert k0 + k2 > 0


def test_g9b_kf_reaches_kind2_plans_with_born_directions():
    """J-b2 J2 (review M4.1): non-vacuity of G-9b on KF — kind-2 plans with kappa_rec >= 1 (the fold directions the
    retired G-9c checked against the PPR family) are reached and compared with (X)."""
    tot = 0
    for name in sorted(KF):
        st = {}
        r = g9b_circuit(KF[name], stats=st)
        assert r is not None and r != "columns", name
        assert not r[2], r[2][:3]
        tot += st.get("k2_kappa_pos", 0)
    assert tot > 0, tot


def test_g9b_kind2_coverage():
    """Non-vacuity: G-9b's FX-PPR circuits reach kind-2 plans (the (G)-only case)."""
    tot = 0
    for name in sorted(FX_PPR):
        r = g9b_circuit(FX_PPR[name], max_configs=0)
        assert r is not None and r != "columns", name
        tot += r[1]
    assert tot >= 20, tot


# ── the term-class boundary's other side (fixplan-termcomm r2, MUT-TC-ALWAYS) ────────────────────────────
def test_tc_commuting_term_families_are_accepted():
    """kf2, kf2_cx, ch_cultivation and the FX-PPR set (commuting term families) build a FAM sampler with a term
    view: NO skip on refusal (XTIM_MUTATE_TC_ALWAYS, the check refusing every circuit, is red here)."""
    texts = dict(FX_PPR)
    texts.update(KF)
    assert "ch_cultivation" in texts and "kf2" in texts and "kf2_cx" in texts
    for name, text in sorted(texts.items()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = TA.compile_family_sampler(text, selfcheck=0)
        assert n_wires(s) > 0 and np.asarray(s.term_axes()[0]).shape[0] > 0, name


# ── §3.10: the null form's operation counts on a gauge-heavy workload ─────────────────────────────────
GAUGE_HEAVY = ("RX 0 1 2\nZ_ERROR(0.2) 0 1\nT 0 1 2\nCS 0 1\nX_ERROR(0.1) 3\nM 3\n"
               + "".join("MX %d\nMPP X%d*X%d\n" % (q, q, (q + 1) % 3) for q in range(3)) * 3
               + "DETECTOR rec[-19]\n" + "".join("DETECTOR rec[-%d]\n" % k for k in range(1, 19)))


def test_ops_null_form_gauge_heavy():
    s = fam_any(GAUGE_HEAVY)
    assert s is not None
    assert len(s.channel_report()["family_extras"]) >= 18
    n = n_wires(s)
    keys = {b""}
    for alts in shots_of(s, 500):
        d = s.diag_shot(alts)
        keys.add(diag_key(n, d))
    for pk in keys:
        g = s.plan_structure(pk, force_general=True)
        d = s.plan_structure(pk)
        m, rho = g["m"], g["rho"]
        assert m == 19
        assert g["ops"] == {"plane": m, "pivot_reduce": m, "greduce": m - rho, "echelon": m - rho, "extras": m}
        assert check_ops_d(d) and d["ops"]["extras"] == 18
        assert check_null_form(g), (g["folds"], list(g["intro_b"]))
        assert not table_diffs(d, g)
