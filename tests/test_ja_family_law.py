"""J-a N1 law gates on FAM samplers, DIAGONAL circuits (plan section 3.2; gauge-plan section 4.1 gates 1-6,
10, 11; G-CS, G-F1).

A FAM sampler (xtim._xtim_testapi.compile_family_sampler — switched OFF in production) serves gauge
(LOGICAL-class) detectors, ANTI detectors and extra observables as record-table extras, emitted
reference-relative (y ^ rf ^ inv ^ ref). Oracles: DENSE = tests/_record_oracle.emitted_law on the ORIGINAL
text (no xtim code); (X) = the S1 RecordLawOracle folded by _record_law_x.emitted_law_x; engine="exact" for
the column-identity gate. Every gate sampler runs the semantic oracle on EVERY shot (oracle_all_shots,
G-11) and its per-class check counts are asserted > 0. Statistics: tests/_ja_fixtures (5 sigma, the S2b
multiplicity rule; impossible outcomes are exact counters).
"""
import numpy as np
import pytest

import _ja_fixtures as F
from _ja_fixtures import family, columns, check_law, check_identity, emitted_law, emitted_law_x, oracle

pytestmark = pytest.mark.filterwarnings("ignore")

N = 40_000

_CLASS_OF_KIND = {"gauge_detector": "gauge_detector", "anti_detector": "anti_detector",
                  "extra_observable": "extra_observable"}


def gate_sampler(text, **kw):
    """A gate FAM sampler: the semantic oracle on every shot (G-11)."""
    return family(text, oracle_all_shots=True, **kw)


def assert_oracle_ran(s, shots):
    """G-11: the oracle ran on every shot and checked every record of every class the circuit has: per
    family class exactly (shots x the class's extras), > 0 for the channels and the Born observable."""
    oc = s.oracle_counts()
    assert oc["shots"] == shots, oc
    rep = s.channel_report()
    for kind in set(k for k, *_ in rep["family_extras"]):
        nk = sum(1 for k, *_ in rep["family_extras"] if k == kind)
        assert oc[_CLASS_OF_KIND[kind]] == shots * nk > 0, (kind, nk, shots, oc)
    if rep["deterministic_detectors"] or rep["deterministic_observables"]:
        assert oc["channel"] > 0, oc
    if rep["born_observable"] >= 0:
        assert oc["born_observable"] > 0, oc


def law_gate(name, text, n=N, x=True):
    """The joint law of every emitted column vs DENSE at 5 sigma (and vs (X) at 1e-12: the two exact
    oracles agree), impossible outcomes = 0; the oracle on every shot. Returns (samples, law)."""
    law = emitted_law(text)
    if x:
        lx = emitted_law_x(oracle(text))
        assert max(abs(law.get(k, 0) - lx.get(k, 0)) for k in set(law) | set(lx)) < 1e-12
    s = gate_sampler(text)
    shots = [0]

    def draw(k, seed):
        shots[0] += k
        return columns(s, k, seed)

    M = check_law(name, draw, law, n)
    assert M.shape[1] == len(F.emitted_columns(text))
    assert_oracle_ran(s, shots[0])
    return M, law


# ── G-1: the gate-1 repro ───────────────────────────────────────────────────────────────────────
def test_g1_gauge_last_word_path_order():
    # M1: det_identity on, the gauge record (an extra, GAUGE class) ORed in AFTER the word copy of the channel
    # detectors in the same byte (MUT-FS-ORDER wipes it to 0)
    M, law = law_gate("gauge_last", F.GATE1_LAST)
    p = M[:, 1].mean()
    exp = sum(v for k, v in law.items() if k[1] == 1)
    assert exp > 0.1 and abs(p - exp) < 5 * np.sqrt(exp * (1 - exp) / M.shape[0]), (p, exp)


def test_g1_gate1_repro():
    M, law = law_gate("gate1", F.GATE1)
    p = M[:, 0].mean()
    exp = sum(v for k, v in law.items() if k[0] == 1)
    assert abs(exp - 0.1464466) < 1e-6
    assert abs(p - exp) < 5 * np.sqrt(exp * (1 - exp) / len(M)), p
    assert family(F.GATE1).channel_report()["family_extras"] == [("gauge_detector", 0, 1, 0)]


# ── G-2: duplicated gauge read ──────────────────────────────────────────────────────────────────
def test_g2_dup_gauge_reported_xor_is_zero_raw_xor_is_beta():
    M, _ = law_gate("dup_gauge", F.DUP_GAUGE)
    violations = int((M[:, 1] ^ M[:, 2]).sum())            # REPORTED bits: exact counter, every shot
    assert violations == 0
    s = family(F.DUP_GAUGE)
    fx = {i: (c, off) for _, i, c, off in s.channel_report()["family_extras"]}
    (c1, o1), (c2, o2) = fx[1], fx[2]
    # raw XOR = the plan constant beta: in every plan reached the two columns have equal coin / Born rows
    orc = oracle(F.DUP_GAUGE)
    nplans = 0
    for _, faults in F.fault_configs(orc):
        S, _, _ = orc.residual(faults, None)
        got = F.plan_key_of(orc, S)
        if got is None:
            continue
        ps = s.plan_structure(got[0])
        A, B, be = np.asarray(ps["rec_A"]), np.asarray(ps["rec_B"]), np.asarray(ps["rec_beta"])
        assert np.array_equal(A[c1], A[c2]) and np.array_equal(B[c1], B[c2])
        nplans += 1
    ps0 = s.plan_structure(b"")
    beta = int(np.asarray(ps0["rec_beta"])[c1]) ^ int(np.asarray(ps0["rec_beta"])[c2])
    assert beta == o1 ^ o2                                 # the reference absorbs the raw constant
    assert nplans >= 2


# ── G-3: design values (the diagonal FX-design circuits) ────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(F.DESIGN))
def test_g3_design_joint_law(name):
    M, law = law_gate(name, F.DESIGN[name])
    sd = lambda p: 5 * np.sqrt(max(p * (1 - p), 1e-12) / len(M))  # noqa: E731
    if name == "p4_gauge":
        assert abs((M[:, 1] == 0).mean() - 0.853553) < sd(0.8536)
        assert int((M[:, 1] != M[:, 2]).sum()) == 0                       # P(D1 == D2) = 1, exact
    elif name == "p3_anti":
        assert abs((M[:, 1] == M[:, 2]).mean() - 0.853553) < sd(0.8536)
        assert int((M[:, 1] != M[:, 3]).sum()) == 0                       # Bell: Z1 = Z2, exact
    elif name == "p4_stim":
        assert abs((M[:, 0] ^ M[:, 1]).mean() - 0.146447) < sd(0.1464)
    elif name == "bell_anti":
        assert int((M[:, 1] ^ M[:, 2]).sum()) == 0                        # the correlated-ANTI relation
        assert abs(M[:, 1].mean() - 0.5) < sd(0.5)
    elif name.startswith("teleport"):
        assert int(M[:, 0].sum()) == 0 and abs(M[:, 1].mean() - 0.5) < sd(0.5)


def test_g3_code_switching_faithful_obs0():
    """code_switching_faithful's observable 0 (refused today: ANTI) is a record extra; 0.506 vs exact."""
    text = open("tests/data/code_switching_faithful.stim").read()
    s = gate_sampler(text)
    assert [k for k, *_ in s.channel_report()["family_extras"]] == ["extra_observable"]
    n = 20_000
    a = columns(s, n, 3)
    b = F.exact_columns(text, n, 4)
    assert a.shape == b.shape
    pa, pb = a[:, -1].mean(), b[:, -1].mean()
    assert abs(pa - 0.506) < 0.02 and abs(pb - 0.506) < 0.02, (pa, pb)
    z = abs(pa - pb) / np.sqrt(pb * (1 - pb) * 2 / n)
    assert z < 5, (pa, pb, z)
    assert_oracle_ran(s, n)


def test_g3_cube_ccz_compiles_on_fam():
    """cube_ccz has no record (expectations only): production refuses it ("no channels of any kind");
    the emptiness gate relaxes on FAM samplers."""
    text = open("tests/data/cube_ccz.stim").read()
    s = family(text)
    d, o, dc = s.sample(100, seed=1)
    assert d.shape == (100, 0) and o.shape == (100, 0)


# ── G-4: gauge + Born observable + Born decision, coincident and independent Lambda ──────────────
COINCIDENT = "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\nDECISION(0) rec[-1]\n"
INDEPENDENT = ("RX 0 1 2\nT 0 1 2\nX_ERROR(0.2) 1\nCS 0 1\nMX 0 1 2\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
               "DECISION(0) rec[-1]\n")
SHARED = ("RX 0 1\nT 0 1\nCX 0 1\nZ_ERROR(0.2) 0\nMX 0\nMX 1\nMPP X0*X1\nDETECTOR rec[-3]\n"
          "OBSERVABLE_INCLUDE(0) rec[-2]\nDECISION(0) rec[-1]\n")


def test_g4_coincident_lambda_perfectly_correlated():
    M, _ = law_gate("coincident", COINCIDENT)
    # columns: D0 (gauge), O0 (Born observable), DEC0 (Born decision): one Lambda
    assert int((M[:, 0] != M[:, 1]).sum()) == 0 and int((M[:, 0] != M[:, 2]).sum()) == 0
    assert compare_tables_ok(COINCIDENT)


@pytest.mark.parametrize("name,text", [("independent", INDEPENDENT), ("shared", SHARED)])
def test_g4_independent_and_shared_lambda_joint_born(name, text):
    law_gate(name, text)
    assert compare_tables_ok(text)


def compare_tables_ok(text):
    return F.compare_tables(text) > 0


# ── G-5: noisy dressing on FX-twin: per-plan field identity vs (X), joint law vs DENSE ──────────
TWINS = F.twins()


@pytest.mark.parametrize("name", sorted(TWINS))
def test_g5_twin_tables_equal_x_on_every_plan(name):
    assert F.compare_tables(TWINS[name]) > 0


@pytest.mark.parametrize("name", sorted(TWINS))
def test_g5_twin_joint_law(name):
    law_gate(name, TWINS[name], n=20_000, x=False)


# ── G-6: convention (column identity with engine="exact") ───────────────────────────────────────
@pytest.mark.parametrize("name", sorted(F.CONVENTION))
def test_g6_convention_columns_equal_exact(name):
    text = F.CONVENTION[name]
    s = gate_sampler(text)
    check_identity(name, lambda n, sd: columns(s, n, sd), lambda n, sd: F.exact_columns(text, n, sd), N)
    law_gate(name, text, n=20_000)


def test_g6_port_circuit_reference_relative():
    """A carried-input port circuit: every column of the FAM sampler — the deterministic channel AND the gauge
    extra — is REFERENCE-RELATIVE, column-identical to engine="exact" (J-b1 P10, Addendum A1 §2, owner O1; this
    inverts J-a N1's ABSOLUTE port pin). The circuit's noiseless reference is nonzero on every detector, so the
    identity is not vacuous (MUT-REF-ABS -> red here)."""
    from _record_oracle import reference_records
    text = F.PORT
    ref = reference_records(text)
    dets, _, _ = F.parities(text)
    off = np.array([sum(ref[r] for r in d) & 1 for d in dets], np.uint8)
    assert off.tolist() == [1, 1, 1]
    s = gate_sampler(text)
    check_identity("port", lambda n, sd: columns(s, n, sd), lambda n, sd: F.exact_columns(text, n, sd), N)


@pytest.mark.parametrize("bit", [0, 1])
def test_g6_if_partitioned_per_pattern(bit):
    """IF-partitioned: each driving pattern resolved (PartitionedTwirlSampler's per-pattern circuit), its
    FAM sampler column-identical to exact. (Through PartitionedTwirlSampler itself: tests/test_ja_ppr_records.py, N4a.)"""
    from xtim import _xtim
    text = _xtim.resolve_branches_text(F.IF_TEXT, [bit])
    s = gate_sampler(text)
    check_identity("if%d" % bit, lambda n, sd: columns(s, n, sd), lambda n, sd: F.exact_columns(text, n, sd), N)


# ── G-F1: the F1 gauge records after the chi-drop ───────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(F.FX_F1))
def test_gf1_case_a_records(name):
    text = F.FX_F1[name]
    M, law = law_gate(name, text)
    if name == "F1":
        for v in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            p = sum(q for k, q in law.items() if (k[0], k[1]) == v)
            assert abs(p - 0.25) < 1e-12, (v, p)
    s = family(text)
    columns(s, 4000, 5)
    chk, bad = s.born_trie_check()
    ora, bad2 = s.born_trie_oracle()                      # every node vs measure_pauli's semantics, bitwise
    assert chk > 0 and bad == 0 and ora == chk and bad2 == 0, (chk, bad, ora, bad2)


# ── G-10: every relation parity of the law (deterministic: exact counters; Born: 5 sigma) ─────────
REL = dict(F.AH11)
REL.update({"bell_anti": F.DESIGN["bell_anti"], "p3_anti": F.DESIGN["p3_anti"], "dup_gauge": F.DUP_GAUGE,
            "coincident": COINCIDENT, "F1": F.FX_F1["F1"]})


@pytest.mark.parametrize("name", sorted(REL))
def test_g10_relation_basis(name):
    text = REL[name]
    # the relation counters on a sampler WITHOUT the semantic oracle: gate 10 stands on its own (the
    # joint laws of these fixtures, with the oracle on every shot, are G-2 / G-3 / G-4 / G-F1 / G-11)
    law = emitted_law(text)
    M = columns(family(text), N, 13)
    ndet, viol, worst = F.relation_counters(M, law)
    assert ndet > 0 and viol == 0, (ndet, viol)
    if worst >= 5:                                         # the S2b multiplicity rule
        F._rerun_rule("g10-" + name, worst)
        M2 = columns(family(text), 2 * N, 31)
        ndet, viol, worst = F.relation_counters(M2, law)
        assert viol == 0 and worst < 5, (viol, worst)
    if name.startswith("ah11"):
        # the three-way relation is the only non-trivial one: every marginal and pair is uniform
        assert int((M[:, 0] ^ M[:, 1] ^ M[:, 2]).sum()) == 0


@pytest.mark.parametrize("name", sorted(F.AH11))
def test_g10_ah11_marginals_and_pairs_are_blind(name):
    """The weaker gate (every marginal and every pairwise joint vs DENSE) on the A-H1-1 fixture: green
    here AND under MUT-ML2 (the beta flip on the dependent D3), while test_g10_relation_basis turns red
    there — the relation counter is what makes gate 10 non-vacuous."""
    text = F.AH11[name]
    law = emitted_law(text)
    M = columns(family(text), N, 7)                        # no semantic oracle: the statistic alone
    import itertools
    for cs in [(0,), (1,), (2,)] + list(itertools.combinations(range(3), 2)):
        sub = {}
        for k, p in law.items():
            kk = tuple(k[c] for c in cs)
            sub[kk] = sub.get(kk, 0.0) + p
        z, imp = F.max_z(M[:, list(cs)], sub)
        assert imp == 0 and z < 5, (cs, z, imp)
