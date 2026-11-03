"""J-b1 P5: the owner's non-Clifford rule, the STATIC fixed-axis merge (plan r3.1 + r3.1-fix; rulings 2026-10-19).

Two terms with the same pushed axis A merge iff nothing between them (an unmerged term or a fault site with p > 0)
anticommutes with A; equal signs fold an S-type Clifford at the earlier slot (convention (A): the earlier axes and
alternatives are recomputed by the end-push), opposite signs cancel; passes in time order to a fixpoint; ACCEPT iff
the merged set commutes, else a FAM sampler is refused at construction with the law-limitation text (the M-NC class
"fault-dependent anticommuting terms"). The bare builder settles the even (Clifford) entries of an accepted form on
the family build ("even-Clifford ladder gap": T;T;H;T = S;H;T), so FAM accepts every circuit the rule accepts.

Oracles: tests/_nc_oracle.py (the static rule in independent code: stim end-push of the family-normalized deferred
stream; shares no code with TermResidual or the bare builder); DENSE = tests/_record_oracle.emitted_law on the
original text (the law gates); a dense unitary per fault configuration (the semantic evidence for the refusals).

Mutations (all red here): XTIM_MUTATE_TC_SKIP (NC-SKIP), XTIM_MUTATE_TC_ALWAYS (NC-ALWAYS), XTIM_MUTATE_NC_MERGE,
_NC_ORDER, _NC_NOMERGE, _NC_NOFOLD (TR + the bare settle), _NC_NOREPUSH; _NC_SINGLEPASS: see
test_singlepass_is_empirically_one_pass.
"""
import os
import random
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
import _nc_oracle as O  # noqa: E402
from _ja_fixtures import check_law, columns, emitted_law  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402
from xtim.errors import TwirlClass, XtimTwirlRefusal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

NC_REFUSAL = (r"twirl sampler: law limitation: fault-dependent surviving axis/Clifford frame: terms \d+ \(deferred gate "
              r"\d+\) and \d+ \(deferred gate \d+\) merge only when no fault occurs at deferred noise site \d+, so the "
              r"surviving term \(or the Clifford frame\) depends on the faults; the circuit is T-depth <= 1 per shot but "
              r"outside the fixed-axis term law \(a per-shot fold is not built\)")
BARE_REFUSAL = r"bare state rejected: non-commuting magic: .* anticommute"

# a deterministic detector tail (every FAM sampler then has a channel)
TAIL = "R 15\nX_ERROR(0.1) 15\nM 15\nDETECTOR rec[-1]\n"


def _c(body, meas):
    return body.replace(";", "\n") + "\n" + meas.replace(";", "\n") + "\n" + TAIL


CH_PAIR = "H 5;T 5;CH 5 1;H 0;CH 1 0;{mid}CH 1 0"
CH_PAIR_MEAS = "MX 0 5;DETECTOR rec[-2];DETECTOR rec[-1]"
T_H_CH = "T 0;H 0;H 1;CH 0 1;{mid}CH 0 1{after}"
T_H_CH_MEAS = "MX 1 0;DETECTOR rec[-2];DETECTOR rec[-1]"

# name -> (text, oracle verdict, FAM verdict: "ACCEPT" | "NC" | "BARE")
FIXTURES = {
    # ACCEPT: commuting or mergeable term sets
    "t_t": (_c("RX 0;T 0;T 0", "MX 0;DETECTOR rec[-1]"), "ACCEPT", "ACCEPT"),
    "t_cx_t": (_c("RX 0 1;T 0;CX 0 1;T 0", "MX 0 1;DETECTOR rec[-2];DETECTOR rec[-1]"), "ACCEPT", "ACCEPT"),
    "ch_single": (_c("RX 0;H 1;CH 1 0;DEPOLARIZE1(0.1) 0", "MX 0 1;DETECTOR rec[-2];DETECTOR rec[-1]"),
                  "ACCEPT", "ACCEPT"),
    "ch_ch_disjoint": (_c("RX 0 2;H 1 3;CH 1 0;CH 3 2", "MX 0 2;DETECTOR rec[-2];DETECTOR rec[-1]"),
                       "ACCEPT", "ACCEPT"),
    "ch_ch_same_pair": (_c("RX 0;H 1;CH 1 0;CH 1 0", "MX 0 1;DETECTOR rec[-2];DETECTOR rec[-1]"),
                        "ACCEPT", "ACCEPT"),
    "t_h_ch_noiseless": (_c(T_H_CH.format(mid="", after=""), T_H_CH_MEAS), "ACCEPT", "ACCEPT"),
    "t_h_ch_noise_after": (_c(T_H_CH.format(mid="", after=";X_ERROR(0.35) 1"), T_H_CH_MEAS), "ACCEPT", "ACCEPT"),
    "ch_pair_q7_first": (_c("X_ERROR(0.35) 7;" + CH_PAIR.format(mid=""), CH_PAIR_MEAS + ";M 7;DETECTOR rec[-1]"),
                         "ACCEPT", "ACCEPT"),
    "ch_pair_q7_last": (_c(CH_PAIR.format(mid="") + ";X_ERROR(0.35) 7", CH_PAIR_MEAS + ";M 7;DETECTOR rec[-1]"),
                        "ACCEPT", "ACCEPT"),
    "ch_pair_q7_between": (_c("H 5;T 5;CH 5 1;X_ERROR(0.35) 7;H 0;CH 1 0;CH 1 0",
                              CH_PAIR_MEAS + ";M 7;DETECTOR rec[-1]"), "ACCEPT", "ACCEPT"),
    "ch_pair_x0_first": (_c("X_ERROR(0.35) 0;" + CH_PAIR.format(mid=""), CH_PAIR_MEAS), "ACCEPT", "ACCEPT"),
    "ch_pair_z_control_between": (_c(CH_PAIR.format(mid="Z_ERROR(0.35) 1;"), CH_PAIR_MEAS), "ACCEPT", "ACCEPT"),
    "r1_rx0_ch_ch": (_c("R 1;RX 0;X_ERROR(0.1) 1;T 0;H 0;CH 0 1;CH 0 1", "MX 0;M 1;DETECTOR rec[-2];DETECTOR rec[-1]"),
                     "ACCEPT", "ACCEPT"),
    "t_x_t": (_c("RX 0;T 0;X_ERROR(0.2) 0;T 0", "MX 0;DETECTOR rec[-1]"), "ACCEPT", "ACCEPT"),
    "t_x_tdag": (_c("RX 0;T 0;X_ERROR(0.2) 0;T_DAG 0", "MX 0;DETECTOR rec[-1]"), "ACCEPT", "ACCEPT"),
    "t_t_h_t": (_c("RX 0;T 0;T 0;H 0;T 0", "MY 0;DETECTOR rec[-1]"), "ACCEPT", "ACCEPT"),
    # p = 0 is not a fault site (r3.1-fix R3-4; the p_factor = 0 row is test_p_factor_zero_is_not_a_fault_site)
    "ch_pair_x_error_0": (_c(CH_PAIR.format(mid="X_ERROR(0) 0;"), CH_PAIR_MEAS), "ACCEPT", "ACCEPT"),
    # the measured truth (ruling on fixplan-p5-pc1-row): PAULI_CHANNEL_1 is (px, py, pz); the CH target's term axis
    # is Y-type, so only the Y-only channel leaves the pair mergeable
    "ch_pair_pc1_y_only": (_c(CH_PAIR.format(mid="PAULI_CHANNEL_1(0,0.1,0) 0;"), CH_PAIR_MEAS), "ACCEPT", "ACCEPT"),
    "ch_pair_pc1_x_only": (_c(CH_PAIR.format(mid="PAULI_CHANNEL_1(0.1,0,0) 0;"), CH_PAIR_MEAS), "REFUSE", "NC"),
    "ch_pair_pc1_z_only": (_c(CH_PAIR.format(mid="PAULI_CHANNEL_1(0,0,0.1) 0;"), CH_PAIR_MEAS), "REFUSE", "NC"),
    # the armed fold law fixture (r3.1): a fault BEFORE a folded S pair, a CH elsewhere (the kind-2 route reached
    # by the Z fault before it); gates the re-push of P_f through the fold (MUT-NC-NOREPUSH)
    "fold_armed": ("RX 0 2\nH 1\nZ_ERROR(0.3) 2\nCH 1 2\nDEPOLARIZE1(0.1) 2\nX_ERROR(0.2) 0\nT 0\nT 0\nH 0\nT 0\n"
                   "MX 0 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n" + TAIL, "ACCEPT", "ACCEPT"),
    # REFUSE by the bare builder (noiselessly anticommuting: OUTSIDE_T_DEPTH_1)
    "t_h_t": (_c("RX 0;T 0;H 0;T 0", "MX 0;DETECTOR rec[-1]"), "REFUSE", "BARE"),
    "t_h_t_noisy": (_c("RX 0;T 0;X_ERROR(0.1) 0;H 0;T 0", "MX 0;DETECTOR rec[-1]"), "REFUSE", "BARE"),
    "ch_cx_ch": (_c("RX 0;H 1;CH 1 0;CX 0 1;CH 1 0", "MX 0 1;DETECTOR rec[-2];DETECTOR rec[-1]"), "REFUSE", "BARE"),
    "ch_h_ch": (_c("RX 0;H 1;CH 1 0;H 1;CH 1 0", "MX 0 1;DETECTOR rec[-2];DETECTOR rec[-1]"), "REFUSE", "BARE"),
    # REFUSE: the M-NC class (fault-dependent anticommuting terms)
    "ch_pair": (_c(CH_PAIR.format(mid="X_ERROR(0.35) 0;"), CH_PAIR_MEAS), "REFUSE", "NC"),
    "ch_pair_depolarize": (_c(CH_PAIR.format(mid="DEPOLARIZE1(0.35) 0;"), CH_PAIR_MEAS), "REFUSE", "NC"),
    "t_h_ch": (_c(T_H_CH.format(mid="X_ERROR(0.35) 1;", after=""), T_H_CH_MEAS), "REFUSE", "NC"),
    # the propagation table (TS:119) refuses it first; typed by the static rule's own refusal (rulings on
    # fixplan-p5-c14): NONCOMMUTING_MAGIC, its text naming the table's refusal (the precedence, in the manifest)
    "t_x_t_h_t": (_c("RX 0;T 0;X_ERROR(0.2) 0;T 0;H 0;T 0", "MY 0;DETECTOR rec[-1]"), "REFUSE", "NC"),
    "cs_witness": ("RX 0 1\nT 1\nH 1\nCZ 0 1\nCS_DAG 0 1\nDEPOLARIZE1(0.1) 0\nCX 0 1\nCS_DAG 1 0\nH 1\n"
                   "DEPOLARIZE1(0.1) 1\nS_DAG 1\nMX 0 1\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n", "REFUSE", "NC"),
    # review-r3 static_fuzz: a false ACCEPT without the recompute after a fold (MUT-NC-NOREPUSH, oracle level); its
    # merged terms anticommute noiselessly, so the bare builder refuses it first
    "norepush_witness": (_c("DEPOLARIZE1(0.1) 1;T 0;H 1;CX 1 0;S 0;T_DAG 0;CH 1 0;T 1;T 0;T 0;CZ 0 1;CH 1 0;S 0;"
                            "CX 0 1;CX 0 1;T_DAG 0", "MX 0 1;DETECTOR rec[-2];DETECTOR rec[-1]"), "REFUSE", "BARE"),
}

# a sample of the 236 rule-ACCEPT / production-REFUSE fuzz circuits (census <S>/jb1/wit; ruling on
# fixplan-nc-cs-class): random 2-3-qubit T/T_DAG/H/S/S_DAG/X/Z/CX/CZ/CS/CS_DAG circuits with X/Y/Z/DEPOLARIZE1 noise
FUZZ_ACCEPT = {}  # filled from tests/data_jb1_nc_fuzz.txt below


def _load_fuzz():
    p = os.path.join(os.path.dirname(__file__), "data_jb1_nc_fuzz.txt")
    if not os.path.exists(p):
        return
    for block in open(p).read().split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        name, text = block.split("\n", 1)
        FUZZ_ACCEPT[name.strip("# ")] = text + "\n"


_load_fuzz()


def fam_verdict(text, **kw):
    """The FAM verdict; a refusal is classified by its TYPE (J-b1 P13: XtimTwirlRefusal) and checked by its text."""
    try:
        F.family(text, **kw).sample(20, seed=1)
        return "ACCEPT", ""
    except XtimTwirlRefusal as e:
        m = str(e)
        if e.twirl_class == TwirlClass.NONCOMMUTING_MAGIC:
            assert re.match(NC_REFUSAL, m), m
            return "NC", m
        if e.twirl_class == TwirlClass.LAW_LIMITATION_PROPAGATION:   # TS:119 on a rule-ACCEPT circuit (class 14)
            assert m.startswith("twirl sampler: law limitation (propagation): the static rule accepts"), m
            return "C14", m
        assert e.twirl_class == TwirlClass.OUTSIDE_T_DEPTH_1 and e.sub_reason == "bare", (e.twirl_class, e.sub_reason)
        assert re.match(BARE_REFUSAL, m), m
        return "BARE", m
    except (ValueError, RuntimeError) as e:
        return "OTHER", str(e)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_verdicts(name):
    text, ov, fv = FIXTURES[name]
    assert O.verdict_of_text(text) == ov, name
    got, msg = fam_verdict(text)
    assert got == fv, (name, got, msg)


@pytest.mark.parametrize("name", sorted(n for n, (_, ov, _) in FIXTURES.items() if ov == "ACCEPT"))
def test_accept_fixture_law_vs_dense(name):
    text = FIXTURES[name][0]
    s = F.family(text, oracle_all_shots=True)
    check_law(name, lambda k, sd: columns(s, k, sd), emitted_law(text), 40_000)


def test_fold_armed_reaches_the_term_route():
    s = F.family(FIXTURES["fold_armed"][0])
    s.sample(4000, seed=3)
    st = s.route_stats()
    assert st["shots_kind2"] > 0, st                           # the folded pair's fault reaches the term route


def test_tthT_bytes_equal_sht():
    """T;T;H;T = S;H;T up to a global phase: the settled bare state is the same state, so a FAM sampler emits the
    same bytes."""
    a = _c("RX 0;X_ERROR(0.2) 0;T 0;T 0;H 0;T 0;Z_ERROR(0.1) 0", "MY 0;DETECTOR rec[-1]")
    b = _c("RX 0;X_ERROR(0.2) 0;S 0;H 0;T 0;Z_ERROR(0.1) 0", "MY 0;DETECTOR rec[-1]")
    ra, rb = F.family(a).sample(5000, seed=7), F.family(b).sample(5000, seed=7)
    for x, y in zip(ra, rb):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_production_accepts_the_ladder_gap():
    """J-b1 F (manifest: the ladder-gap circuits are ACCEPTED): production IS the family build — T;T;H;T compiles
    (it refused 'even-Clifford ladder gap' pre-flip) and emits the FAM bytes."""
    import xtim
    text = FIXTURES["t_t_h_t"][0]
    pub = xtim.compile_twirl_sampler(text, selfcheck=0).sample(3000, seed=4)
    fam = F.family(text).sample(3000, seed=4)
    for x, y in zip(pub, fam):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_p_factor_zero_is_not_a_fault_site():
    text = FIXTURES["ch_pair"][0]
    assert O.verdict_of_text(text, 0.0) == "ACCEPT"
    assert fam_verdict(text, p_factor=0.0)[0] == "ACCEPT"
    assert fam_verdict(text)[0] == "NC"


@pytest.mark.parametrize("name", sorted(FUZZ_ACCEPT))
def test_fuzz_rule_accept_is_fam_accept_and_law_correct(name):
    text = FUZZ_ACCEPT[name]
    assert O.verdict_of_text(text) == "ACCEPT"
    got, msg = fam_verdict(text)
    assert got == "ACCEPT", (name, msg)
    s = F.family(text, oracle_all_shots=True)
    check_law(name, lambda k, sd: columns(s, k, sd), emitted_law(text), 20_000)


def test_fuzz_sample_is_present():
    assert len(FUZZ_ACCEPT) >= 20


# the declared class-14 set (rulings on fixplan-p5-c14 (A)): the 29 rule-ACCEPT circuits the propagation table refuses
# (27 of the 236 fuzz circuits, 2 xtim test texts: test_forced_branches), typed LAW_LIMITATION_PROPAGATION.
C14 = {}
for _blk in open(os.path.join(os.path.dirname(__file__), "data_jb1_nc_c14.txt")).read().split("\n\n"):
    _blk = _blk.strip("\n")
    if _blk:
        _nm, _tx = _blk.split("\n", 1)
        assert _nm.startswith("# c14_"), _nm
        C14[_nm[2:]] = _tx + "\n"


def test_c14_declared_set_size():
    assert len(C14) == 29 and sum(k.startswith("c14_xtim_tests_") for k in C14) == 2


@pytest.mark.parametrize("name", sorted(C14))
def test_c14_rule_accept_is_a_declared_propagation_refusal(name):
    assert O.verdict_of_text(C14[name]) == "ACCEPT"
    assert fam_verdict(C14[name])[0] == "C14"


# ── G-NC-CONF: the verdict does not depend on the merge order (32 random orders) ─────────────────────────
def _random_order_verdict(n, stream, rng):
    """The static rule with a RANDOM merge order: repeatedly pick a uniformly random mergeable pair (any i < j with
    the same axis, no anticommuting term and no anticommuting fault between) until none is left."""
    ops = O.ops_of(n, stream)
    items = O.end_push(n, ops, faults=True)
    while True:
        cand = []
        tpos = [k for k, x in enumerate(items) if x[0] == "T"]
        for a_, i in enumerate(tpos):
            for j in tpos[a_ + 1:]:
                A = items[j][1]
                if items[i][1] != A:
                    continue
                mid = items[i + 1:j]
                if any(x[0] == "T" and O._anti(x[1], A) for x in mid):
                    continue
                if any(O._anti(P, A) for x in mid if x[0] == "F" for P in x[1]):
                    continue
                cand.append((i, j))
        if not cand:
            break
        i, j = rng.choice(cand)
        A = items[j][1]
        k = (items[i][2] + items[j][2]) % 8
        if k % 2:
            items[i][2] = k
            del items[j]
            continue
        del items[j]
        del items[i]
        if k:
            for x in items[:i]:
                if x[0] == "T":
                    B = O._conj(x[1], A, k)
                    if B.sign == -1:
                        B, x[2] = -B, -x[2]
                    x[1] = B
                else:
                    x[1] = [O._conj(P, A, k) for P in x[1]]
    T = [x[1] for x in items if x[0] == "T"]
    return "REFUSE" if any(O._anti(a, b) for a_, a in enumerate(T) for b in T[a_ + 1:]) else "ACCEPT"


def test_g_nc_conf_merge_order():
    rng = random.Random(2026)
    texts = [t for t, ov, _ in FIXTURES.values()] + list(FUZZ_ACCEPT.values())
    diffs = []
    for t in texts:
        n, st = TA._deferred_stream(t, 1.0)
        v = O.verdict(n, st)
        for _ in range(32):
            if _random_order_verdict(n, st, rng) != v:
                diffs.append(t)
                break
    assert not diffs, diffs[:3]


def test_singlepass_is_empirically_one_pass():
    """MUT-NC-SINGLEPASS (r3.1-fix R3-3): no fixture or fuzz circuit needs a second merging pass (the fixpoint loop
    stays as specified; review-r3 static_fuzz: 0 / 4,500). This pins the observation: if a circuit ever needs a
    second pass, add it as the SINGLEPASS red fixture."""
    for t in [t for t, _, _ in FIXTURES.values()] + list(FUZZ_ACCEPT.values()):
        v, d = O.verdict_of_text(t, 1.0, True)
        assert d["passes"] <= 2, (t, d)   # pass 2 is the confirming pass that merges nothing


# ── semantic evidence for the refusals (dense, per fault configuration) ──────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_G1 = {0: np.array([[0, 1], [1, 0]], complex), 1: np.array([[0, -1j], [1j, 0]]), 2: np.diag([1, -1]).astype(complex),
       3: np.diag([1, 1j]), 4: np.diag([1, -1j]), 7: np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
       8: np.diag([1, np.exp(1j * np.pi / 4)])}
_PM = {"X": _G1[0], "Y": _G1[1], "Z": _G1[2]}


def _apply(U, n, mat, qs):
    k = len(qs)
    t = U.reshape([2] * n + [-1])
    t = np.moveaxis(t, qs, list(range(k)))
    sh = t.shape
    t = (mat @ t.reshape(2 ** k, -1)).reshape(sh)
    return np.moveaxis(t, list(range(k)), qs).reshape(2 ** n, -1)


def _two(kind):
    if kind == 5:
        return np.diag([1, 1, 1, -1]).astype(complex)
    if kind == 6:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
    if kind == 9:
        return np.diag([1, 1, 1, 1j])
    if kind == 11:
        m = np.eye(4, dtype=complex)
        m[2:, 2:] = _G1[7]
        return m
    raise ValueError(kind)


def _unitary(n, stream, fault=None):
    """The deferred stream's unitary (noise dropped), with `fault` = (stream index, stim.PauliString) applied at its
    noise site. Qubit q is tensor axis q (big-endian)."""
    U = np.eye(2 ** n, dtype=complex)
    for si, d in enumerate(stream):
        if d["kind"] == 0:
            g, t = d["gate"], d["targets"]
            if g in _G1:
                U = _apply(U, n, _G1[g], [t[0]])
            elif g == 10:
                m = np.eye(8, dtype=complex)
                m[7, 7] = -1
                U = _apply(U, n, m, t[:3])
            else:
                U = _apply(U, n, _two(g), t[:2])
        elif d["kind"] == 1 and fault is not None and fault[0] == si:
            P = fault[1]
            for q in range(n):
                c = "_XYZ"[P[q]]
                if c != "_":
                    U = _apply(U, n, _PM[c], [q])
    return U


def _compact(n, stream):
    """Relabel the wires touched by a gate or a noise site to 0..m-1 (the others are identity throughout)."""
    used = sorted({q for d in stream if d["kind"] == 0 for q in d["targets"]} |
                  {q for d in stream if d["kind"] == 1 for q in d["qubits"]})
    m = {q: i for i, q in enumerate(used)}
    out = []
    for d in stream:
        if d["kind"] == 0:
            out.append(dict(d, targets=[m[q] for q in d["targets"]]))
        elif d["kind"] == 1:
            out.append(dict(d, qubits=[m[q] for q in d["qubits"]]))
    return len(used), out


def _is_pauli_times_phase(M):
    N = M.shape[0]
    col0 = M[:, 0]
    a = int(np.argmax(np.abs(col0)))
    if abs(abs(col0[a]) - 1) > 1e-9:
        return False
    ph = col0[a]
    for x in range(N):
        v = M[:, x]
        y = x ^ a
        if abs(abs(v[y]) - 1) > 1e-9 or np.sum(np.abs(v) ** 2) - abs(v[y]) ** 2 > 1e-9:
            return False
        r = v[y] / ph
        if min(abs(r - 1), abs(r + 1)) > 1e-9:
            return False
    sgn = [np.real(M[x ^ a, x] / ph) for x in range(N)]
    nb = N.bit_length() - 1
    b = [sgn[1 << i] for i in range(nb)]
    return all(abs(sgn[x] - np.prod([b[i] for i in range(nb) if x >> i & 1])) < 1e-9 for x in range(N))


@pytest.mark.parametrize("name", sorted(n for n, (_, _, fv) in FIXTURES.items() if fv == "NC"))
def test_refusal_is_fault_dependent_dense(name):
    """For a refused M-NC fixture: the no-fault configuration is accepted by the rule, and some single fault makes
    D = U_f U_0^dag something no fixed-axis law can produce: not a Pauli times a product of the no-fault surviving
    rotations exp(i m pi/4 A_k) (the surviving axis / Clifford frame depends on the fault)."""
    assert _fault_dependence_evidence(name)


@pytest.mark.parametrize("name", ["t_x_t", "ch_pair_q7_between", "ch_pair_z_control_between", "fold_armed",
                                  "t_h_ch_noise_after"])
def test_fault_dependence_evidence_negative_controls(name):
    """The same check on ACCEPTED noisy fixtures finds every single fault explained by the fixed-axis law."""
    assert not _fault_dependence_evidence(name)


def _fault_dependence_evidence(name):
    """True iff some single fault's D = U_f U_0^dag is NOT a Pauli times a product of exp(i m pi/4 A_k) over the
    candidate fixed axes A_k: for a refused fixture the no-fault configuration's merged axes (the rule's own output
    anticommutes, so the merged set is the only commuting candidate the no-fault shot realizes); for an accepted
    one the rule's surviving axes (the law the sampler serves)."""
    text, _, fv = FIXTURES[name]
    n, st = _compact(*TA._deferred_stream(text, 1.0))
    assert n <= 10, (name, n)                                  # dense bound (the touched wires only)
    v, det = O.verdict(n, st, True)
    if fv == "NC":
        assert v == "REFUSE", name
        v, det = O.verdict(n, [d for d in st if d["kind"] != 1], True)
    assert v == "ACCEPT", name
    surv = det["axes"]
    U0 = _unitary(n, st)
    sites = [si for si, d in enumerate(st) if d["kind"] == 1]
    rots = []
    for A in surv:
        Am = np.eye(1, dtype=complex)
        for q in range(n):
            Am = np.kron(Am, {0: _I2, 1: _PM["X"], 2: _PM["Y"], 3: _PM["Z"]}[A[q]])
        Am = Am * A.sign
        rots.append(np.cos(np.pi / 4) * np.eye(2 ** n) + 1j * np.sin(np.pi / 4) * Am)
    import itertools
    evidence = False
    for si in sites:
        for P in O.supports(n, st[si]):
            D = _unitary(n, st, (si, P)) @ U0.conj().T
            explained = False
            for m in itertools.product(range(4), repeat=len(rots)):
                R = np.eye(2 ** n, dtype=complex)
                for r, e in zip(rots, m):
                    R = R @ np.linalg.matrix_power(r, e)
                if _is_pauli_times_phase(D @ R.conj().T):
                    explained = True
                    break
            if not explained:
                evidence = True
                break
        if evidence:
            break
    return evidence
