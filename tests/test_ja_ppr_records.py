"""J-a N4a: records on CH/PPR plans — the kind-2 route of a FAM sampler with the ppr_records class (plan §1.1
rows 5 (r2.4), 8, 16; §2.2 N4; §3.2 G-1..G-6, G-10, G-11, G-12 on PPR circuits; §3.10).

A FAM sampler (xtim._xtim_testapi.compile_family_sampler, switched OFF in production) with the ppr_records class
serves every PPR-class shot (>= 1 fired alternative with a PPR-retry atom) through the TERM-INDEXED residual
(run_w<*>, at the PPR-route decision; twirl_sampler_fam.cpp fam_term_shot_): compose_terms_ -> reduce_key_
-> kind_of_; kind 0 -> the shared normal form -> the diagonal plan cache's kind-0 plan (builder D); kind 2 -> the
(G) table of the reduced term set (a sampler-owned header, its payload charged to and evicted by the plan cache);
then the channel prefix rows of P, record_rows_ (the Born trie walk) and t(P) for the extras. The legacy route
(the V2-T1 PPR plan family) was deleted at J-b2 J5; the term route is the only route of a PPR-class shot.

Oracles: DENSE = tests/_record_oracle.emitted_law on the ORIGINAL text (pure states; small circuits) or its
density-matrix twin tests/_record_oracle_mixed.emitted_law_mixed (same gate semantics, noise as Kraus sums: the
fuzz corpus); (X) = RecordLawOracle where its fault enumeration is feasible; engine="exact" for column identity.
Statistics: 5 sigma with the S2b multiplicity rule (_ja_fixtures); impossible outcomes are exact counters. G-11:
every gate sampler runs the semantic oracle on EVERY shot; a term-routed shot is checked by the TERM-VIEW oracle
(R|bare> built by gates from the composed (P, S): rot(B_j) per term, then P), and every gate asserts that the
oracle verified every term-routed shot (oracle_term_shots == shots_kind0 + shots_kind2 > 0) and per-class counts.
"""
import json
import re
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import _ja_fixtures as F
from _record_oracle import emitted_law, max_z, record_law
from _record_oracle_mixed import emitted_law_mixed, record_law_mixed
from test_ja_term_key import SEED23, fuzz_circuit

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
N = 40_000


# ── fixtures ─────────────────────────────────────────────────────────────────────────────────────
def _no_expect(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("PAULI_EXPECTATION")) + "\n"


CH = _no_expect(open(ROOT / "tests" / "data" / "ch_cultivation.stim").read())
CH_NOISY = CH.replace("0.001", "0.05")                     # kind-0 and kind-2 term shots
CH_ZNOISE = CH_NOISY.replace("X_ERROR(0.05) 0 1\n", "").replace("X_ERROR(0.05) 0 2\n", "").replace(
    "X_ERROR(0.05) 0 3\n", "")                                # Z noise only: every term-routed shot is kind 2
# the cultivated H-eigenstate read out as every record class: an inverted gauge read (!Z0: inv = ref = 1, MUT-REF),
# a Born observable, an ANTI detector, a Born decision, an extra (ANTI) observable
_CH_RECS = ("MPP !Z0\nMPP Z0\nMX 0\nM 0\nDETECTOR rec[-4]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
            "OBSERVABLE_INCLUDE(1) rec[-1]\nDECISION(0) rec[-2]\n")
CH_REC = CH_NOISY + _CH_RECS
CH_REC_Z = CH_ZNOISE + _CH_RECS
# N4a-1: the same records with the H-eigenstate read in X FIRST (Bloch (1/sqrt2, 0, 1/sqrt2); the kind-2 axes rotate
# it by pi/2 about Y). A Z-first read sees only the z component, which the true rotation and the identity
# (MUT-K2-ROUTE) leave alike on X+Z noise; the x component tells the true rotation from both the identity and the
# reversed sign (MUT-G-PULL).
CH_REC_X = CH_NOISY + _CH_RECS.replace("MPP !Z0\nMPP Z0\nMX 0\nM 0", "MPP !X0\nMPP X0\nM 0\nMX 0")


_SWAP_M = {"M": "MX", "MX": "M"}
_SWAP_ERR = {"X_ERROR": "Z_ERROR", "Z_ERROR": "X_ERROR"}


def ppr_twin(text, p=0.1):
    """The PPR twin of a diagonal fixture: on the wire q of its first single-target T line, a Y_ERROR(p) just
    before the T and an H just after it, every later operation on q re-expressed in the H-conjugated frame
    (M <-> MX, MY -> MY with the record inverted, MPP factors X_q <-> Z_q and Y_q -> the product inverted,
    X_ERROR <-> Z_ERROR, X <-> Z) — the same physical records, while the fault's X part crossing T is now a
    non-diagonal (PPR-retry) atom, so the circuit takes the kind-2 route. Fixtures with a later two-qubit gate
    on q are not transformed (ValueError)."""
    lines = text.splitlines()
    out, q = [], None
    for ln in lines:
        tok = ln.split()
        if q is None:
            if tok and tok[0] == "T" and len(tok) == 2:
                q = tok[1]
                out += ["Y_ERROR(%g) %s" % (p, q), ln, "H " + q]
            else:
                out.append(ln)
            continue
        if not tok or tok[0].startswith(("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "OUTPUT_QUBITS", "MPAD")):
            out.append(ln)
            continue
        name = tok[0]
        base = name.split("(")[0]
        args = name[len(base):]
        tg = tok[1:]
        if base == "MPP":
            prods = []
            for t in tg:
                inv = t.startswith("!")
                fs = []
                for f in t.lstrip("!").split("*"):
                    if f[1:] == q:
                        if f[0] == "Y":
                            inv = not inv
                        f = {"X": "Z", "Z": "X", "Y": "Y"}[f[0]] + q
                    fs.append(f)
                prods.append(("!" if inv else "") + "*".join(fs))
            out.append(name + " " + " ".join(prods))
            continue
        if q not in [t.lstrip("!") for t in tg]:
            out.append(ln)
            continue
        if base in ("M", "MX", "MY", "X_ERROR", "Z_ERROR", "Y_ERROR", "X", "Z", "Y", "H"):
            for t in tg:                                  # one instruction per target: record order kept
                if t.lstrip("!") != q:
                    out.append("%s %s" % (name, t))
                elif base in _SWAP_M:
                    out.append("%s%s %s" % (_SWAP_M[base], args, t))
                elif base == "MY":
                    out.append("MY%s %s" % (args, q if t.startswith("!") else "!" + q))
                elif base in _SWAP_ERR:
                    out.append("%s%s %s" % (_SWAP_ERR[base], args, q))
                elif base in ("X", "Z"):
                    out.append("%s %s" % ({"X": "Z", "Z": "X"}[base], q))
                else:                                     # Y_ERROR, Y, H: invariant up to a sign / phase
                    out.append("%s %s" % (name, t))
            continue
        raise ValueError("ppr_twin: %r acts on the magic wire %s" % (ln, q))
    if q is None:
        raise ValueError("ppr_twin: no single-target T line")
    return "\n".join(out) + "\n"


def _twins(src):
    out = {}
    for k, v in src.items():
        try:
            out[k] = ppr_twin(v)
        except ValueError:
            pass
    return out


def fuzz_records(seed):
    """The adv-A fuzz circuit `seed` (fuzz.py build, n = 4) extended with every record class (G-12): a magic
    wire 4 (T|+>) and, before the final reads, an inverted gauge read !Z0*X4 (LOGICAL; inv = ref = 1), a Born
    observable Z1*Z2*X4, an ANTI observable Y4, a Born decision X0*X1 (which also turns the final Z0 / Z1
    detectors ANTI); the final parity detector stays deterministic."""
    L = fuzz_circuit(seed).splitlines()
    i = L.index("M 0 1 2 3")
    head = L[:1] + ["RX 4", "T 4"] + L[1:i]
    mid = ["MPP !Z0*X4", "MPP Z1*Z2*X4", "MPP Y4", "MPP X0*X1"]
    tail = L[i:] + ["DETECTOR rec[-8]", "OBSERVABLE_INCLUDE(0) rec[-7]", "OBSERVABLE_INCLUDE(1) rec[-6]",
                    "DECISION(0) rec[-5]"]
    return "\n".join(head + mid + tail) + "\n"


def biased_fuzz_records(seed, open_magic=False, p=0.02):
    """fuzz_records(seed) with every DEPOLARIZE1 / DEPOLARIZE2 replaced by X_ERROR(p) on the same wires, and (with
    open_magic) the T_DAG layer removed, so the circuit's T layer is left on the data wires.

    Why (N4a-1): depolarizing noise is T-covariant, T D(rho) T^dag = D(T rho T^dag), so the emitted law of a
    depolarized circuit cannot tell a fault's residual P rot(B) from P alone (MUT-K2-ROUTE) or from P rot(-B)
    (MUT-G-PULL): the [A] corpus is blind to the kind-2 rotation. A single-Pauli channel (X_ERROR) is not
    covariant, so the mirror circuit (C1 T C2 C2^dag T^dag C1^dag) sees a DROPPED rotation (the bare state is a
    stabilizer state there, rot(B) turns a deterministic Z read into a coin). A rotation's SIGN is invisible on a
    stabilizer bare state (a single-record pullback i M B is then a coin, and products of anticommuting records
    pull back sign-free), so MUT-G-PULL needs the magic on the read wires: the open variant, whose bare state
    carries the T layer."""
    t = fuzz_records(seed)
    if open_magic:
        t = "\n".join(ln for ln in t.splitlines() if not ln.startswith("T_DAG")) + "\n"
    t = re.sub(r"DEPOLARIZE1\([^)]*\) (\d+)", lambda m: "X_ERROR(%g) %s" % (p, m.group(1)), t)
    return re.sub(r"DEPOLARIZE2\([^)]*\) (\d+) (\d+)", lambda m: "X_ERROR(%g) %s %s" % (p, m.group(1), m.group(2)), t)


K0_PAIR_REFUSAL = "outside the diagonal plan cache's pair universe"   # the N4-D1 refusal (now MUT-K0-REFUSE only)

# N4-D1 (fixplan-k0pairs r2): the circuits whose kind-0 term composites carry a CZ pair outside the diagonal plan
# cache's pair universe (the review's census): served by the kind-2 (G) path, kind 0 kept (shots_kind0_general).
K0G_OPEN = (1, 9, 11, 14, 31, 32, 39, 48, 49, 58, 59, 68, 70, 79, 80, 83, 84)
K0G_MIRROR = (131, 157, 160, 162, 189, 286, 296, 307, 460, 466, 528, 552, 564)


def seed23_records():
    L = SEED23.strip().splitlines()
    i = L.index("M 0 1 2 3")
    head = L[:1] + ["RX 4", "T 4"] + L[1:i]
    mid = ["MPP !Z0*X4", "MPP Z1*Z2*X4", "MPP Y4", "MPP X0*X1"]
    tail = L[i:] + ["DETECTOR rec[-8]", "OBSERVABLE_INCLUDE(0) rec[-7]", "OBSERVABLE_INCLUDE(1) rec[-6]",
                    "DECISION(0) rec[-5]"]
    return "\n".join(head + mid + tail) + "\n"



# FX-design / G-1 / G-2 / G-4 / G-6 PPR twins (DENSE pure + (X)), the FX-PPR records set (DENSE mixed)
# (the twins whose magic wire is read once, right after the H, are not PPR: the normalizer folds that H into the
# read; they are replaced by the ch_cultivation-based fixtures below)
TWINS = _twins({"gate1": F.GATE1, "dup_gauge": F.DUP_GAUGE, "p4_gauge": F.DESIGN["p4_gauge"],
                "p4_stim": F.DESIGN["p4_stim"],
                # N4a-1: gate1 read in Y. T|+> sits at angle pi/4 in the XY plane; the Y fault's residual rotates it
                # by -pi/2 to -pi/4, which an X read (cos) cannot tell from the identity (MUT-K2-ROUTE) but a Y read
                # (sin) can; MUT-G-PULL (to 3pi/4) shows in both
                "gate1_y": F.GATE1.replace("MX 0", "MY 0")})
# ch_cultivation's H-eigenstate (wire 0; <X> = <Z> = 1/sqrt 2, <Y> = 0) as the magic read of the FX-design and
# convention fixtures (every one PPR: the CH Hadamard tests' faults are PPR-retry atoms)
CH_DESIGN = {
    # p3_anti: D1 = Z4, D2 = Z4 X0, D3 = Z5 on a Bell pair (4, 5): P(D1 == D3) = 1
    "p3_anti_ch": CH_NOISY + ("R 4\nR 5\nH 4\nCX 4 5\nX_ERROR(0.05) 6\nM 6\nDETECTOR rec[-1]\nM 4\nMX 0\nM 5\n"
                              "DETECTOR rec[-3]\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\n"),
    # one Lambda read as a gauge detector, a Born observable and a Born decision: perfectly correlated
    "coincident_ch": CH_NOISY + "MX 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\nDECISION(0) rec[-1]\n",
}
CONV = _twins({k: F.CONVENTION[k] for k in ("fb_noisy_gauge", "fb_noiseless_gauge")})
CONV.update({
    "readout_flip_ch": CH_NOISY + "MX(0.1) 0\nDETECTOR rec[-1]\n",
    "invert_ch": CH_NOISY + "MX !0\nDETECTOR rec[-1]\n",
    "mpad_ch": CH_NOISY + "MX 0\nMPAD 1\nX_ERROR(0.1) 4\nM 4\nDETECTOR rec[-1]\nDETECTOR rec[-3] rec[-2]\n",
    "cy_feedback_ch": CH_NOISY + ("X_ERROR(0.2) 4\nM 4\nH 4\nCY rec[-1] 4\nMX 4\nMY 0\n"
                                  "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\n"),
    "fb_noisy_gauge_ch": CH_NOISY + ("RX 4\nR 5\nMX(0.1) 0\nCZ rec[-1] 4\nMX 4\nM 5\nDETECTOR rec[-1]\n"
                                     "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-3] rec[-2]\n"),
})
FX_PPR_REC = {"ch_rec": CH_REC, "seed23": seed23_records()}
FX_PPR_REC.update({f"fuzz{sd}": fuzz_records(sd) for sd in (37, 45, 51, 59, 67, 86)})
# N4a-1: the same seeds under X_ERROR noise (mirror: sees a dropped rotation) and with the magic left open (sees its
# sign); seed 59's open variant is an N4-D1 refusal and is left out
FX_PPR_REC.update({f"fuzz{sd}_x": biased_fuzz_records(sd) for sd in (37, 45, 51, 59, 67, 86)})
FX_PPR_REC.update({f"fuzz{sd}_open_x": biased_fuzz_records(sd, True) for sd in (37, 45, 51, 67, 86)})


# ── samplers, oracles, statistics ────────────────────────────────────────────────────────────────
def gate_sampler(text, **kw):
    return F.family(text, oracle_all_shots=True, **kw)


_CLASS = ("gauge_detector", "anti_detector", "extra_observable")


def assert_oracle_ran(s, shots, term=True):
    """G-11: the oracle ran on every shot, checked every record of every class the circuit has, and verified
    EVERY term-routed shot with the term-view oracle (kind 0 and kind 2)."""
    oc, rs, rep = s.oracle_counts(), s.route_stats(), s.channel_report()
    assert oc["shots"] == shots, oc
    for kind in set(k for k, *_ in rep["family_extras"]):
        nk = sum(1 for k, *_ in rep["family_extras"] if k == kind)
        assert oc[kind] == shots * nk > 0, (kind, nk, shots, oc)
    if rep["deterministic_detectors"] or rep["deterministic_observables"]:
        assert oc["channel"] > 0, oc
    if rep["born_observable"] >= 0:
        assert oc["born_observable"] == shots, oc
    assert rs["oracle_term_shots"] == rs["shots_kind0"] + rs["shots_kind2"], rs
    if term:
        assert rs["shots_kind2"] > 0, ("no kind-2 shot: the fixture does not reach the route", rs)


def max_z_pooled(M, law, min_expected=20):
    """max_z with the cells of expected count < min_expected pooled into ONE cell (the normal approximation of
    a sparse cell is not a 5-sigma statistic); impossible outcomes (law = 0) stay an exact counter."""
    Nn = len(M)
    keys, cnt = np.unique(np.asarray(M, np.uint8), axis=0, return_counts=True)
    emp = {tuple(int(x) for x in k): int(c) for k, c in zip(keys, cnt)}
    worst, imp, rp, rc = 0.0, 0, 0.0, 0
    for k in set(emp) | set(law):
        p, c = law.get(k, 0.0), emp.get(k, 0)
        if p < 1e-15:
            imp += c
        elif Nn * p < min_expected:
            rp, rc = rp + p, rc + c
        elif p < 1:
            worst = max(worst, abs(c - Nn * p) / np.sqrt(Nn * p * (1 - p)))
    if 0 < rp < 1:
        worst = max(worst, abs(rc - Nn * rp) / np.sqrt(Nn * rp * (1 - rp)))
    return worst, imp


def check_law_pooled(name, draw, law, n):
    M = draw(n, 11)
    z, imp = max_z_pooled(M, law)
    assert imp == 0, (name, "impossible outcomes", imp)
    if z >= 5:
        F._rerun_rule(name, z)
        M = draw(2 * n, 23)
        z, imp = max_z_pooled(M, law)
        assert imp == 0 and z < 5, (name, "recurrence", z, imp)
    return M


def law_then_oracle(name, text, law, n, mixed, term=True):
    """The LAW gate first, on a sampler WITHOUT the semantic oracle (so a law error is red through the law statistic,
    never only through an oracle abort; N4a-1), then G-11: a gate sampler with the oracle on every shot replays the
    same draws (same sizes and seeds) and must emit the same bits (the oracle is output-neutral) with every shot
    verified. Returns (samples, the oracle sampler)."""
    s_law = F.family(text)
    draws = []

    def draw(k, seed):
        M = F.columns(s_law, k, seed)
        draws.append((k, seed, M))
        return M

    M = (check_law_pooled if mixed else F.check_law)(name, draw, law, n)
    s = gate_sampler(text)
    for k, seed, Mk in draws:
        assert np.array_equal(F.columns(s, k, seed), Mk), (name, "the oracle sampler's bits differ", k, seed)
    assert_oracle_ran(s, sum(k for k, _, _ in draws), term=term)
    return M, s


def law_gate(name, text, n=N, x=True, mixed=False, term=True):
    """The joint law of every emitted column vs DENSE at 5 sigma (vs (X) at 1e-12 where enumerable), impossible
    outcomes = 0, then the oracle on every shot (term-routed shots by the term-view oracle): law_then_oracle."""
    law = emitted_law_mixed(text) if mixed else emitted_law(text)
    if x:
        lx = F.emitted_law_x(F.oracle(text))
        assert max(abs(law.get(k, 0) - lx.get(k, 0)) for k in set(law) | set(lx)) < 1e-12, name
    M, s = law_then_oracle(name, text, law, n, mixed, term=term)
    assert M.shape[1] == len(F.emitted_columns(text))
    return M, law, s


# ── the mixed-state DENSE oracle equals the pure one where both run ──────────────────────────────
_PURE = dict(TWINS, ch_rec=CH_REC, ch_rec_x=CH_REC_X, **CH_DESIGN, **CONV)


@pytest.mark.parametrize("name", sorted(_PURE))
def test_dense_mixed_equals_pure(name):
    text = _PURE[name]
    a, b = record_law(text, readout=True), record_law_mixed(text, readout=True)
    assert max(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) < 1e-12


# ── G-1 / G-2 / G-3 / G-4 on PPR twins (DENSE pure + (X)) ────────────────────────────────────────
@pytest.mark.parametrize("name", ["gate1", "gate1_y"])
def test_g1_gate1_ppr_twin(name):
    M, law, s = law_gate(name + "_ppr", TWINS[name])
    exp = sum(v for k, v in law.items() if k[0] == 1)
    assert abs(M[:, 0].mean() - exp) < 5 * np.sqrt(exp * (1 - exp) / len(M))
    assert s.channel_report()["family_extras"] == [("gauge_detector", 0, 1, 0)]


def test_g2_dup_gauge_ppr_twin_reported_xor_is_zero():
    M, _, _ = law_gate("dup_gauge_ppr", TWINS["dup_gauge"])
    assert int((M[:, 1] ^ M[:, 2]).sum()) == 0             # REPORTED bits: exact counter, every shot


@pytest.mark.parametrize("name", ["p4_gauge", "p4_stim", "p3_anti_ch"])
def test_g3_design_ppr(name):
    M, law, _ = law_gate(name, CH_DESIGN[name] if name in CH_DESIGN else TWINS[name])
    if name == "p4_gauge":
        assert int((M[:, 1] != M[:, 2]).sum()) == 0                       # P(D1 == D2) = 1, exact
    elif name == "p3_anti_ch":
        assert int((M[:, 4] != M[:, 6]).sum()) == 0                       # Bell: Z4 = Z5, exact
        exp = sum(p for k, p in law.items() if k[4] == k[5])              # (1 + <X0>)/2 on the noisy |H>
        assert 0.7 < exp < 0.86 and abs((M[:, 4] == M[:, 5]).mean() - exp) < 5 * np.sqrt(exp * (1 - exp) / len(M))


@pytest.mark.parametrize("name", ["ch_rec", "ch_rec_x"])
def test_g3_ch_cultivation_records(name):
    """ch_cultivation (the H-eigenstate, CH Hadamard tests) at p = 0.05, read out as every record class (ch_rec_x:
    the magic read in X first, N4a-1)."""
    M, law, s = law_gate(name, CH_REC if name == "ch_rec" else CH_REC_X)
    kinds = sorted(k for k, *_ in s.channel_report()["family_extras"])
    assert kinds == ["anti_detector", "extra_observable", "gauge_detector"], kinds
    assert s.channel_report()["born_observable"] == 0 and s.num_decisions == 1
    rs = s.route_stats()
    assert rs["shots_kind0"] > 0 and rs["shots_kind2"] > 0, rs


def test_g4_coincident_lambda_ppr():
    M, _, _ = law_gate("coincident_ch", CH_DESIGN["coincident_ch"])
    assert int((M[:, 3] != M[:, 4]).sum()) == 0 and int((M[:, 3] != M[:, 5]).sum()) == 0


# ── G-5: FX-PPR with records (fuzz seeds 37 .. 86 and seed23, DENSE by density matrices) ─────────
@pytest.mark.parametrize("name", sorted(k for k in FX_PPR_REC if k != "ch_rec"))
def test_g5_fx_ppr_records_joint_law(name):
    law_gate(name, FX_PPR_REC[name], n=40_000 if name.endswith("_x") else 20_000, x=False, mixed=True)


# ── G-6: convention twins (column identity with engine="exact"), the IF-partitioned wrapper ─────
@pytest.mark.parametrize("name", sorted(CONV))
def test_g6_convention_ppr_twins_equal_exact(name):
    text = CONV[name]
    s = gate_sampler(text)
    F.check_identity(name + "_ppr", lambda n, sd: F.columns(s, n, sd), lambda n, sd: F.exact_columns(text, n, sd), N)
    law_gate(name + "_ppr", text, n=20_000)


IF_PPR = ppr_twin(F.IF_TEXT)


@pytest.mark.parametrize("name,text", [("if", F.IF_TEXT), ("if_ppr", IF_PPR)])
def test_g6_if_partitioned_through_partitioned_sampler(name, text):
    """An IF-partitioned circuit through xtim.twirl.PartitionedTwirlSampler. J-b1 F: the public partition sampler's
    inner samplers ARE family samplers with every class (compile_family_partitioned / _make_inner removed, m5).
    Shots are mixed over both driving patterns; each pattern's columns equal engine="exact" on that pattern's
    resolved circuit at 5 sigma."""
    import xtim
    from xtim import _xtim
    p = xtim.compile_twirl_sampler(text, selfcheck=0)
    W = p._total_width
    nd = p.num_detectors
    for bit in (0, 1):
        resolved = _xtim.resolve_branches_text(text, [bit] + [0] * (W - 1))

        def draw(n, sd, bit=bit):
            db = np.zeros((2 * n, W), np.uint8)
            db[1::2, 0] = 1                                   # interleaved patterns: one call serves both
            out = p.sample(2 * n, seed=sd, decision_bits=db)
            d = F.unpack(out[0], nd)
            return d[bit::2]
        F.check_identity(f"{name}{bit}", draw, lambda n, sd, r=resolved: F.exact_columns(r, n, sd)[:, :nd], N)
    for key, inner in p._sampler_cache.items():
        # the production TwirlSampler IS the family sampler (cpp/bindings/family_sampler.hpp); its record classes are
        # the family law's (the gauge detector is a Born record), read through the public record_report
        assert type(inner).__module__ == "xtim._xtim" and type(inner).__name__ == "TwirlSampler"
        rep = inner.record_report()
        assert "born" in rep["classes"]["detectors"].values()
        if name == "if_ppr":
            assert rep["term_route"]["shots_kind2"] > 0


# ── G-10: deterministic relations = exact zero-violation counters on PPR plans ───────────────────
REL = {"ch_rec": CH_REC, "dup_gauge": TWINS["dup_gauge"], "p3_anti_ch": CH_DESIGN["p3_anti_ch"],
       "coincident_ch": CH_DESIGN["coincident_ch"], "seed23": FX_PPR_REC["seed23"], "fuzz45": FX_PPR_REC["fuzz45"]}


@pytest.mark.parametrize("name", sorted(REL))
def test_g10_relation_basis_ppr(name):
    text = REL[name]
    mixed = name.startswith(("seed", "fuzz"))
    law = emitted_law_mixed(text) if mixed else emitted_law(text)
    s = F.family(text)
    M = F.columns(s, N, 13)
    assert s.route_stats()["shots_kind2"] > 0
    ndet, viol, worst = F.relation_counters(M, law)
    assert ndet > 0 and viol == 0, (ndet, viol)
    if worst >= 5:                                         # Born-valued relations at 5 sigma (the S2b rule)
        F._rerun_rule("g10-ppr-" + name, worst)
        ndet, viol, worst = F.relation_counters(F.columns(F.family(text), 2 * N, 31), law)
        assert viol == 0 and worst < 5, (viol, worst)


# ── G-12: the [A] fuzz corpus (adv-A fuzz.py, 90 circuits) with every record class ───────────────
def test_g12_fuzz_corpus_with_records_vs_dense():
    """DENSE (density matrices, the ORIGINAL text) vs the FAM sample at 5 sigma (sparse cells pooled) and the
    impossible-outcome counter, over the 90 circuits of the [A] corpus extended with a gauge detector, a Born
    observable, an ANTI observable and a Born decision; the oracle on every shot. Non-vacuity: MUT-CLASS and
    MUT-REF turn this red (mutations.md). The corpus's DEPOLARIZE noise is T-covariant, so its law cannot see the
    kind-2 rotation: MUT-G-PULL / MUT-K2-ROUTE non-vacuity is test_g12b's (N4a-1)."""
    tot = {"shots_kind0": 0, "shots_kind2": 0, "ppr_circuits": 0}
    for sd in range(90):
        text = fuzz_records(sd)
        law = emitted_law_mixed(text)
        _, s = law_then_oracle(f"g12-fuzz{sd}", text, law, 20_000, mixed=True, term=False)
        rs = s.route_stats()
        tot["shots_kind0"] += rs["shots_kind0"]
        tot["shots_kind2"] += rs["shots_kind2"]
        tot["ppr_circuits"] += rs["shots_kind2"] > 0
    assert tot["ppr_circuits"] >= 30 and tot["shots_kind0"] > 0 and tot["shots_kind2"] > 0, tot


@pytest.mark.parametrize("open_magic", [False, True], ids=["mirror_x", "open_x"])
def test_g12b_biased_corpus_with_records_vs_dense(open_magic):
    """N4a-1: G-12 on the [A] corpus with X_ERROR noise (biased_fuzz_records: depolarizing noise is T-covariant, so
    the depolarized corpus above cannot see the kind-2 rotation). mirror_x: the T layer undone (sees a dropped
    rotation, MUT-K2-ROUTE); open_x: the T layer left on the read wires (also sees the rotation's sign, MUT-G-PULL).
    DENSE (density matrices) vs the FAM sample at 5 sigma, impossible outcomes = 0, the oracle on every shot. No
    refusal of any kind (N4-D1 r2: the out-of-universe kind-0 entries are served by the (G) path)."""
    tot = {"shots_kind2": 0, "ppr_circuits": 0, "k0general": 0}
    for sd in range(90):
        text = biased_fuzz_records(sd, open_magic)
        law = emitted_law_mixed(text)
        _, s = law_then_oracle(f"g12b-{int(open_magic)}-fuzz{sd}", text, law, 40_000, mixed=True, term=False)
        rs = s.route_stats()
        tot["shots_kind2"] += rs["shots_kind2"]
        tot["ppr_circuits"] += rs["shots_kind2"] > 0
        tot["k0general"] += rs["shots_kind0_general"] > 0
    assert tot["ppr_circuits"] >= 30, tot
    if open_magic:                                         # the 17 circuits that refused before N4-D1 r2
        assert tot["k0general"] >= len(K0G_OPEN), tot


def _k0g_variants():
    def opened(t):
        return "\n".join(ln for ln in t.splitlines() if not ln.startswith("T_DAG")) + "\n"
    return {"mirror_dep": lambda sd: fuzz_records(sd), "open_dep": lambda sd: opened(fuzz_records(sd)),
            "mirror_x": lambda sd: biased_fuzz_records(sd, False), "open_x": lambda sd: biased_fuzz_records(sd, True)}


@pytest.mark.parametrize("variant", ["mirror_dep", "open_dep", "mirror_x", "open_x"])
def test_n4d1_zero_refusal_census(variant):
    """N4-D1 r2 condition 2: the route over the [A] corpus with records, seeds 0-599, in all four variants (T layer
    undone / left open x DEPOLARIZE / X_ERROR): ZERO refusals of any kind, and the out-of-universe kind-0 route is
    reached (the review's census: 13 mirror / 134 open circuits refused before the fix)."""
    make = _k0g_variants()[variant]
    k0g, refused = [], []
    for sd in range(600):
        s = F.family(make(sd))
        try:
            F.columns(s, 20_000, 3)
        except RuntimeError as e:
            refused.append((sd, str(e)[:120]))
            continue
        rs = s.route_stats()
        if rs["shots_kind0_general"] > 0:
            k0g.append(sd)
            assert rs["builds_kind0_general"] == rs["plans_kind0_general"] > 0, (sd, rs)
        assert rs["builds_kind2"] == rs["plans_kind2"], (sd, rs)
    assert refused == [], refused
    assert len(k0g) >= (13 if variant.startswith("mirror") else 100), (variant, len(k0g))
    if variant.startswith("mirror"):
        assert set(K0G_MIRROR) <= set(k0g), (variant, k0g)
    else:
        assert set(K0G_OPEN) <= set(k0g), (variant, k0g)


@pytest.mark.parametrize("sd,open_magic", [(sd, True) for sd in K0G_OPEN] + [(sd, False) for sd in K0G_MIRROR])
def test_n4d1_out_of_universe_law_and_oracle(sd, open_magic):
    """N4-D1 r2 condition 3: the 17 open + 13 mirror circuits (X_ERROR) that refused before the fix: the LAW gate vs
    DENSE (density matrices) at 5 sigma, impossible outcomes = 0, then the oracle on every shot (term-view oracle on
    every term-routed shot); the out-of-universe route counter > 0 (condition 4; MUT-K0-REFUSE turns this red)."""
    text = biased_fuzz_records(sd, open_magic)
    _, s = law_then_oracle(f"n4d1-{int(open_magic)}-fuzz{sd}", text, emitted_law_mixed(text), 40_000, mixed=True,
                           term=False)
    rs = s.route_stats()
    assert rs["shots_kind0_general"] > 0 and rs["plans_kind0_general"] > 0, rs
    assert rs["shots_kind0"] >= rs["shots_kind0_general"], rs


# ── the route: dispatch = N2's term route; plan ids; payload caching / eviction ──────────────────
def test_route_dispatch_equals_term_shot_on_the_same_events():
    """Every PPR-class shot of a run is dispatched exactly as N2's term_shot() computes it on the same fired
    alternatives: the (kind, plan id) histogram of the run equals the replay's (exact counter); kind-0 term
    shots exist and their plan ids are the diagonal route's (one intern table, JA-4)."""
    s = F.family(CH_REC)
    n, seed = 20_000, 7
    F.columns(s, n, seed)
    got = {pid: tuple(v) for pid, v in s.route_stats()["plan_shots"].items()}
    is_ppr = s.term_alt_is_ppr()
    want = {}
    for ev in s.term_events(n, seed):
        alts = [int(a) for a in ev]
        if not any(is_ppr[a] for a in alts):
            continue
        r = s.term_shot(alts)
        k, c = want.get(r["plan_id"], (r["kind"], 0))
        assert k == r["kind"]
        want[r["plan_id"]] = (k, c + 1)
    assert got == want
    k0 = [pid for pid, (k, _) in got.items() if k == 0]
    assert k0, "no kind-0 term shot (MUT-G-KIND: the kind is ignored)"
    rs = s.route_stats()
    assert rs["builds_kind2"] == rs["plans_kind2"] and rs["payload_builds_kind2"] == 0, rs


@pytest.mark.parametrize("name,text", [("fuzz45_k0", None), ("ch_rec", CH_REC), ("fuzz45_x", FX_PPR_REC["fuzz45_x"])])
def test_kind0_normal_form_is_the_key_constant_xor_the_shot_pauli(name, text):
    """N4a-2's identity, by the FULL normal-form path (term_shot: P rot(S) composed, residual_normal_form,
    canonicalize_mod_stabilizers): on every kind-0 term shot of a run, (a, cz) depend on the reduced key S only and
    the prefix XOR the shot's reduced Pauli P is a per-key constant (exact counters). The route serves kind 0 from
    that constant (nf(I, S)) and P, so this is the route's exactness condition."""
    s = F.family(K0_WORKLOAD if text is None else text)
    isp = s.term_alt_is_ppr()
    seen, nk0 = {}, 0
    for ev in s.term_events(20_000, 7):
        alts = [int(a) for a in ev]
        if not any(isp[a] for a in alts):
            continue
        r = s.term_shot(alts)
        if r["kind"] != 0:
            continue
        nk0 += 1
        (px, pz), nf = r["P_red"], r["nf"]
        qx, qz = nf["prefix"][0], nf["prefix"][1]
        val = (tuple(nf["a"]), tuple(map(tuple, nf["cz"])), bytes(np.bitwise_xor(px, qx)), bytes(np.bitwise_xor(pz, qz)))
        assert seen.setdefault(tuple(r["S_red"]), val) == val, (name, r["S_red"])
    assert nk0 > 50 and len(seen) >= 1, (name, nk0, len(seen))


@pytest.mark.parametrize("name,text", [("ch_rec", CH_REC), ("seed23", FX_PPR_REC["seed23"])])
def test_route_payload_eviction_is_output_neutral(name, text):
    """Plan payloads (kind 2 included) are charged to the plan cache and evicted between shots like (D)'s: under
    a 1-byte budget every shot rebuilds what it needs, and the output bytes equal the unbounded sampler's."""
    a, b = F.family(text), F.family(text)
    b.set_plan_payload_budget(1)
    x, y = a.sample(20_000, seed=5), b.sample(20_000, seed=5)
    assert all(np.array_equal(u, v) for u, v in zip(x, y))
    ra, rb = a.route_stats(), b.route_stats()
    assert ra["payload_builds_kind2"] == 0 and rb["payload_builds_kind2"] > 0 and b.plan_payload_stats()["evictions"] > 0


def test_route_refusals_by_name():
    s = F.family(CH_REC)
    with pytest.raises(RuntimeError, match="sigma export is refused on a family sampler"):   # retention: N5
        s.sample(10, retain=True)
    with pytest.raises(ValueError, match="declares no INPUT_QUBITS port"):
        s.sample(10, input_pauli=np.zeros((10, 2), np.uint8))


# ── §3.10: zero allocations per shot on the term-routed per-shot paths (warm plans, fixed replay) ──
K0_WORKLOAD = fuzz_records(45).replace("(0.02)", "(0.05)")
K0G_WORKLOAD = biased_fuzz_records(1, True, p=0.05)              # N4-D1 r2 condition 5 (open_x seed 1)
ALLOC_CHILD = r"""
import ctypes, json, os, sys, warnings
sys.path.insert(0, sys.argv[1])
from xtim import _xtim_testapi as TA
shim = ctypes.CDLL(sys.argv[2])
out = (ctypes.c_long * 2)()
def counted(f):
    shim.za_start(); f(); shim.za_stop(out)
    return [out[0], out[1]]
res = {}
for name, text, n1, n2 in json.loads(sys.argv[3]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = TA.compile_family_sampler(text, selfcheck=0)
    for _ in range(2):                         # cold, then every buffer / cache at its final state for both sizes
        s.sample(n2, seed=5); s.sample(n1, seed=5)
    k0 = s.route_stats()
    a2 = counted(lambda: s.sample(n2, seed=5))
    a1 = counted(lambda: s.sample(n1, seed=5))
    a2b = counted(lambda: s.sample(n2, seed=5))
    k1 = s.route_stats()
    res[name] = {"n1": n1, "n2": n2, "a1": a1, "a2": a2, "a2b": a2b,
                 "kind2": k1["shots_kind2"] - k0["shots_kind2"], "kind0": k1["shots_kind0"] - k0["shots_kind0"],
                 "k0general": k1["shots_kind0_general"] - k0["shots_kind0_general"],
                 "builds": [k0["builds_kind2"] + k0["builds_kind0_general"], k1["builds_kind2"] + k1["builds_kind0_general"]]}
print(json.dumps(res))
"""


@pytest.mark.skipif(platform.system() != "Linux" or platform.libc_ver()[0] != "glibc",
                    reason="needs Linux/glibc (LD_PRELOAD counting allocator)")
def test_zero_allocations_per_shot_kind2_route(tmp_path):
    """A warm FAM sampler call of N2 shots performs exactly as many allocation calls as one of N1 < N2 shots
    (tests/test_zero_alloc_per_shot.py's counting allocator, same seed = a fixed replay of the events), on the
    kind-2 path: compose_terms_, reduce_key_, the dispatch, record_rows_ on (G) tables (Born trie walk), t(P),
    the FamilySink emit. Workloads: ch_cultivation with Z noise (every term-routed shot kind 2), with and
    without the record extras (gauge / ANTI / observables / a Born decision); N4a-2: fuzz seed 45 with every record
    class at p = 0.05, whose term-routed shots are kind 0 (the per-key normal form plus the shot's Pauli: the kind-0
    path) as well as kind 2; N4-D1 r2: open_x seed 1, whose kind-0 entries outside the pair universe are served by
    the (G) path (shots_kind0_general)."""
    from test_zero_alloc_per_shot import SHIM, _cc
    if _cc() is None:
        pytest.skip("no C compiler")
    (tmp_path / "za.c").write_text(SHIM)
    so = tmp_path / "libza.so"
    subprocess.run([_cc(), "-O2", "-fPIC", "-shared", str(tmp_path / "za.c"), "-o", str(so)], check=True)
    import xtim
    pkg_parent = str(Path(xtim.__file__).resolve().parents[1])
    wl = [["ch_z", CH_ZNOISE, 4000, 8000], ["ch_rec_z", CH_REC_Z, 4000, 8000],
          ["fuzz45_k0", K0_WORKLOAD, 4000, 12000],
          ["open1_k0g", K0G_WORKLOAD, 4000, 12000]]
    env = dict(os.environ, LD_PRELOAD=str(so))
    r = subprocess.run([sys.executable, "-I", "-c", ALLOC_CHILD, pkg_parent, str(so), json.dumps(wl)], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    for name, c in res.items():
        assert c["a1"][0] > 0 and c["a2"][0] > 0, (name, c)           # the counter is live
        if name == "fuzz45_k0":                                        # N4a-2: the kind-0 term path is measured
            assert c["kind0"] > 1000 and c["kind2"] > 1000, (name, c)
        elif name == "open1_k0g":                                      # N4-D1 r2: kind 0 outside the pair universe
            assert c["k0general"] > 1000, (name, c)
        else:
            assert c["kind2"] > 1000 and c["kind0"] == 0, (name, c)    # the measured shots ARE the kind-2 path
        assert c["builds"][0] == c["builds"][1], (name, c)             # warm: no plan built in the window
        assert c["a2"][0] == c["a1"][0], (name, "allocations per shot",
                                          (c["a2"][0] - c["a1"][0]) / (c["n2"] - c["n1"]), c)
        assert c["a2b"] == c["a2"], (name, c)


# ── J-a final review m1: the bitwise Born-trie gates walk the FAM route's own plans ───────────────────
# born_trie_check / born_trie_oracle covered the diagonal cache's header table only (production walk); kind-2
# and route-1 plans (FamRoute::k2plans) were never visited (vacuous on the new route). Exact counts: every
# non-NaN node of every resident record trie (a linear scan) is walked, per kind; route 1 witnessed; the
# walk recurses (more nodes than plans). MUT-K2-TRIE-P0 (a perturbed kind-2 root) -> bad > 0 here.
@pytest.mark.parametrize("name", ["ch_rec_x", "open_x14"])
def test_born_trie_walks_cover_the_term_route_plans(name):
    text = CH_REC_X if name == "ch_rec_x" else biased_fuzz_records(14, True)
    s = F.family(text)
    s.sample(20000, seed=3)
    diag, k2, r1 = s.born_trie_nodes()
    st = s.route_stats()
    assert k2 >= st["plans_kind2"] > 0, (k2, st["plans_kind2"])         # ch_rec_x: one root per kind-2 plan
    if name == "open_x14":
        assert k2 > st["plans_kind2"], (k2, st["plans_kind2"])          # deep tries: a root-only walk fails
        assert r1 > 0 and st["plans_kind0_general"] > 0, (r1, st["plans_kind0_general"])   # route 1 witnessed
    assert s.born_trie_walk_family(False) == [k2, r1, 0]
    assert s.born_trie_walk_family(True) == [k2, r1, 0]
    assert tuple(s.born_trie_check()) == (diag + k2 + r1, 0)
    assert tuple(s.born_trie_oracle()) == (diag + k2 + r1, 0)
