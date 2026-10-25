"""J-a N5: retention for the whole family (plan §1.1 rows 10-11, §2.2 N5, §3.7, r2.2 JA-2), switched OFF.

A FAM sampler's recording run (FamilySampler.sample_records, test API only) routes EVERY shot through the term
route and writes its layout-3 record: the content (kind 0 = the canonical S^a.CZ view, kind 2 = the reduced term
bitset), P_rec (the frame folded on the input_pauli route), y (the raw record values, the emission's own
derivation), the emission's outcomes (u, z) and frame_folded. materialize (materialize_terms_ for kind 2) is the
FORCED PROJECTION: bare -> C0 -> P_rec -> every pivot record, then one introducer record per Born direction, at y
(never Lambda_j: A2). signs() is the H-membership rule (design §4); record_classes() the R = I classification.

Gates (state gates, so every mutation is red through a state value, not an oracle abort):
  G-8a  state == emitted bits on every shot of every family circuit: <M_c> on materialize(i) == (-1)^{y_c}
        (1e-12) for EVERY record column, and emitted XOR y is a per-column constant (no readout-flip coins).
  G-8b  the record's (content, P_rec, outcomes) determines y on the (D) and kind-2 routes (exact counter).
  G-8c  materialize == DENSE (the all-records projector of P_rec C0 |bare>, never the anchors) at 1e-10.
  G-SG  signs() == the materialize-based signs: adv-A r5check (2) / (2b), r6check (the t_c term), wlam_dec (the
        decision-conditioned generator), and bulk over the certified generators, the records and random Paulis.
  JA-2  a2_pivot_then_intro + its gauge twin (MUT-ANCH-LAM).
Mutations (mutations.md): MUT-RET-ROT, MUT-ANCH-LAM, M-ANCH (MS_BORN_FLIP), MUT-SG-TC, MUT-SG-UNDRESSED,
MUT-SG-NODEC.
"""
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _ja_fixtures as F
from test_ja_ppr_records import (CH_DESIGN, FX_PPR_REC, K0G_MIRROR, K0G_OPEN, CONV as PPR_CONV,
                                 TWINS as PPR_TWINS, biased_fuzz_records, fuzz_records)
from test_ja_frames import FIXTURES as FR_FIX, FRAMES, frame_rows
from test_s2b_record_table import FIXTURES as S2B

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")

# ── JA-2: the A2 dense counterexample as a circuit ────────────────────────────────────────────────
# adv-A-r2 R2-2 rebuilt as a circuit: the magic wire 0 (T|+>, Bloch (1/sqrt2, 1/sqrt2, 0)) and a fixed wire 1 in |0>.
# Records Q1 = Z0 X1 (a PIVOT: X1 on |0> is a fair coin, projected first) and Q2 = X0 Z1 (a Born direction:
# K = Z1 . X_L, Lambda = X_L), which commute. After the pivot projection Lambda = X0 anticommutes with Q1, so it is
# no longer the record: P(Q2 = 0 | Q1) = (1 + <X>)/2 = 0.854 on the record, 0.5 on Lambda (adv-A's 0.424 vs 0.25
# for its alpha). (The adv-A alpha itself is not Clifford+T-preparable; RX T H T is rejected by the bare builder.)
A2 = ("RX 0\nT 0\nR 1\nX_ERROR(0.1) 2\nM 2\nMPP Z0*X1\nMPP X0*Z1\n"
      "DETECTOR rec[-3]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
A2_GAUGE = A2.replace("OBSERVABLE_INCLUDE(0) rec[-1]", "DETECTOR rec[-1]")   # the introducer as a DETECTOR

# ── fixture sets ──────────────────────────────────────────────────────────────────────────────────
DIAG = {"gate1": F.GATE1, "dup_gauge": F.DUP_GAUGE, "port": F.PORT, "a2": A2, "a2_gauge": A2_GAUGE}
DIAG.update({f"design_{k}": v for k, v in F.DESIGN.items()})
DIAG.update({f"f1_{k}": v for k, v in F.FX_F1.items()})
DIAG.update({f"ah11_{k}": v for k, v in F.AH11.items()})
DIAG.update({f"conv_{k}": v for k, v in F.CONVENTION.items()})
DIAG.update({f"twin_{k}": v for k, v in F.twins().items()})
DIAG.update({f"s2b_{k}": v for k, v in S2B.items()})
PPRS = dict(FX_PPR_REC)
PPRS.update({f"twin_{k}": v for k, v in PPR_TWINS.items()})
PPRS.update(CH_DESIGN)
PPRS.update({f"conv_{k}": v for k, v in PPR_CONV.items()})
PPRS["fuzz59_open_x"] = biased_fuzz_records(59, True)                      # N4-D1: no longer refused
FUZZ = {f"fuzz{sd}": fuzz_records(sd) for sd in range(0, 90, 3)}
FUZZ.update({f"fuzz{sd}_open_x": biased_fuzz_records(sd, True) for sd in range(1, 90, 6)})
K0G = {f"k0g_open{sd}": biased_fuzz_records(sd, True, p=0.05) for sd in K0G_OPEN[:6]}
K0G.update({f"k0g_mirror{sd}": biased_fuzz_records(sd, False, p=0.05) for sd in K0G_MIRROR[:4]})


def run(text, shots, seed, P=None):
    s = F.family(text)
    d, o, dc, R = s.sample_records(shots, seed=seed, input_pauli=None if P is None else frame_rows(shots, P))
    E = np.concatenate([F.unpack(d, s.num_detectors), F.unpack(o, s.num_observables),
                        F.unpack(dc, s.num_decisions)], 1)
    return s, R, E


def emitted_index(s):
    order = s.plan_structure(b"")["rec_order"]
    base = {0: 0, 1: s.num_detectors, 2: s.num_detectors + s.num_observables}
    return [base[o >> 24] + (o & ((1 << 24) - 1)) for o in order]


def state_equals_bits(text, shots=200, seed=5, P=None):
    """G-8a on one run; returns (s, R, per-route counts)."""
    s, R, E = run(text, shots, seed, P)
    ops = TA.record_operators(s)
    idx = emitted_index(s)
    rf = re.search(r"\bM[XYR]*\(", text) is not None or re.search(r"\bMPP\(", text) is not None
    off, routes = None, {}
    for i in range(shots):
        y = R.y(i).astype(int)
        d = E[i, idx] ^ y
        if off is None:
            off = d
        assert rf or np.array_equal(d, off), ("emitted XOR y is not a per-column constant", i, d, off)
        e = s.state_expectations(R, i, ops)
        assert np.max(np.abs(e - (1.0 - 2.0 * y))) < 1e-12, ("state != bits", i, e, y)
        routes[R.route(i)] = routes.get(R.route(i), 0) + 1
    return s, R, routes


# ── G-8a: state == emitted bits ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(DIAG))
def test_g8a_state_equals_bits_diagonal(name):
    _, R, routes = state_equals_bits(DIAG[name])
    assert set(routes) == {0} and sum(routes.values()) == R.num_records


@pytest.mark.parametrize("name", sorted(PPRS))
def test_g8a_state_equals_bits_ppr(name):
    _, _, routes = state_equals_bits(PPRS[name], shots=300)
    assert set(routes) <= {0, 1, 2}


def test_g8a_ppr_routes_reached():
    """The PPR set reaches the (D) kind-0 route, the kind-2 (G) route and (with the N4-D1 circuits) route 1."""
    tot = {}
    for name in ("ch_rec", "fuzz45", "fuzz59_open_x", "twin_gate1_y"):
        _, _, routes = state_equals_bits(PPRS[name], shots=300)
        for k, v in routes.items():
            tot[k] = tot.get(k, 0) + v
    assert tot.get(0, 0) > 0 and tot.get(1, 0) > 0 and tot.get(2, 0) > 0, tot


@pytest.mark.parametrize("name", sorted(FUZZ))
def test_g8a_state_equals_bits_fuzz(name):
    state_equals_bits(FUZZ[name], shots=150, seed=7)


@pytest.mark.parametrize("name", sorted(K0G))
def test_g8a_state_equals_bits_out_of_universe(name):
    """N4-D1's route 1 (kind 0 served by (G)): recorded as kind 0 (its S^a.CZ view, P_rec = nf0.prefix xor P)."""
    s, R, routes = state_equals_bits(K0G[name], shots=300, seed=9)
    assert routes.get(1, 0) > 0, routes
    assert all(R.kind(i) == 0 for i in range(R.num_records) if R.route(i) == 1)


@pytest.mark.parametrize("name,P", [(k, P) for k in sorted(FR_FIX) for P in FRAMES[k]])
def test_g8a_state_equals_bits_frames(name, P):
    """FX-frames on the input_pauli route: the frame folded into P_rec (frame_folded = 1)."""
    _, R, _ = state_equals_bits(FR_FIX[name], shots=200, seed=11, P=P)
    assert all(R.frame_folded(i) == 1 for i in range(R.num_records))


def test_frame_folded_route_rule():
    """A port circuit without a supplied frame: frame-free records (frame_folded = 0); with the frame supplied on a
    frames-class sampler: frame_folded = 1 (J-b2 J6: every sampler serves the frames class)."""
    _, R, _ = state_equals_bits(F.PORT)
    assert all(R.frame_folded(i) == 0 for i in range(R.num_records))
    s = F.family(FR_FIX["l4"])
    _, _, _, R = s.sample_records(10, seed=1, input_pauli=frame_rows(10, "X"))
    assert any(R.frame_folded(i) == 1 for i in range(R.num_records))


# ── G-8b: (content, P_rec, outcomes) determines y on the (D) and kind-2 routes ────────────────────
@pytest.mark.parametrize("name,text", [("ch_rec", PPRS["ch_rec"]), ("fuzz45", PPRS["fuzz45"]),
                                       ("twin_dup_gauge", PPRS["twin_dup_gauge"]),
                                       ("twin_wlam_dec", DIAG["twin_wlam_dec.gdec0"]),
                                       ("s2b_wlam_dec", DIAG["s2b_wlam_dec"])])
def test_g8b_group_constancy_of_y(name, text):
    _, R, _ = run(text, 3000, 13)
    seen, groups, viol = {}, 0, 0
    for i in range(R.num_records):
        if R.route(i) == 1:
            continue
        x, z = R.prec(i)
        k = (R.handle(i), bytes(x), bytes(z), bytes(R.outcomes(i)))
        y = bytes(R.y(i))
        if k in seen:
            viol += seen[k] != y
        else:
            seen[k] = y
            groups += 1
    assert viol == 0 and groups > 1, (viol, groups)
    keys = {R.key(i) for i in range(R.num_records)}            # layout 3: the key carries y (its sufficient stat)
    assert len(keys) >= groups // 4


# ── G-8c: materialize == DENSE (the all-records projector, no anchor in the oracle) ───────────────
_IDX = {}


def _apply_pauli(psi, n, x, z, ph):
    """i^ph X^x Z^z |psi>, qubit 0 = the most significant bit (xtim.frames.dense_state's convention)."""
    idx = _IDX.get(n)
    if idx is None:
        idx = _IDX[n] = np.arange(1 << n, dtype=np.uint64)
    zm = sum(1 << (n - 1 - q) for q in range(n) if z[q])
    xm = sum(1 << (n - 1 - q) for q in range(n) if x[q])
    sign = 1.0 - 2.0 * (np.bitwise_count(idx & np.uint64(zm)) & 1)
    out = np.empty_like(psi)
    out[(idx ^ np.uint64(xm)).astype(np.int64)] = psi * sign
    return out * (1j ** (int(ph) % 4))


def dense_state(fr):
    """xtim.frames.dense_state with a RANDOM seed vector: its affine 'generic' vector is annihilated exactly by two
    X-type -1 projections (seen on F1_noisy); the state is rank one, so any vector with weight on it serves."""
    n = int(fr["n"])
    dim = 1 << n
    rng = np.random.default_rng(12345)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    ops = [(fr["stab_x"][a], fr["stab_z"][a], fr["stab_phase"][a]) for a in range(n)]
    for (x, z, p) in ops:
        v = (v + _apply_pauli(v, n, x, z, p)) / 2.0
    nv = np.linalg.norm(v)
    assert nv > 1e-9
    ref = v / nv
    psi = np.zeros(dim, complex)
    c = np.asarray(fr["coeff"], complex)
    for i in range(c.shape[0]):
        psi += c[i] * _apply_pauli(ref, n, fr["branch_x"][i], fr["branch_z"][i], fr["branch_phase"][i])
    return psi


_BARE = {}


def dense_record_state(s, R, i):
    fr = TA.bare_frames(s)
    n = int(fr["n"])
    if id(s) not in _BARE:                                  # the bare vector once per sampler (kept alive with it)
        _BARE[id(s)] = (s, dense_state(fr))
    psi = _BARE[id(s)][1]
    d = R.plan(i)
    if d["kind"] == 0:
        idx = np.arange(1 << n)
        bit = lambda q: (idx >> (n - 1 - q)) & 1  # noqa: E731
        ph = np.ones(1 << n, complex)
        for q in d["a"]:
            ph *= np.where(bit(q), 1j, 1.0)
        for f, g in d["cz"]:
            ph *= np.where(bit(f) & bit(g), -1.0, 1.0)
        psi = psi * ph
    else:
        xz, phs = s.term_axes()
        xz = np.asarray(xz)
        for j in d["S"]:                                    # rot(B) = (1 + iB)/sqrt 2 (exp(i pi B / 4))
            psi = (psi + 1j * _apply_pauli(psi, n, xz[j, :n], xz[j, n:], phs[j])) / math.sqrt(2)
    x, z = R.prec(i)
    psi = _apply_pauli(psi, n, x, z, 0)
    y = R.y(i)
    for c, (ox, oz, op) in enumerate(TA.record_operators(s)):
        psi = (psi + (1 - 2 * int(y[c])) * _apply_pauli(psi, n, ox, oz, op)) / 2
    nv = np.linalg.norm(psi)
    assert nv > 1e-9, ("the record has zero weight in DENSE", i)
    return n, psi / nv


def materialize_equals_dense(text, shots=60, seed=3, P=None):
    """G-8c on one run (n <= 22: the vector oracle is O(2^n) per Pauli; above n = 16 the first 12 shots)."""
    s, R, _ = run(text, shots, seed, P)
    nq = int(TA.bare_frames(s)["n"])
    assert nq <= 22, ("dense oracle: n > 22", nq)
    order = sorted(range(shots), key=lambda i: (R.handle(i) in {R.handle(j) for j in range(i)}, i))
    pick = order[:12] if nq > 16 else order                 # one shot per distinct plan first
    worst = 0.0
    for i in pick:
        n, psi = dense_record_state(s, R, i)
        got = dense_state(s.materialize(R, i))
        worst = max(worst, abs(1.0 - abs(np.vdot(psi, got))))
    _BARE.pop(id(s), None)
    assert worst < 1e-10, worst
    return s, R, pick


G8C = {k: DIAG[k] for k in ("gate1", "dup_gauge", "a2", "a2_gauge", "design_p4_gauge", "design_p3_anti",
                            "design_bell_anti", "design_obs3", "f1_F1_noisy", "twin_wlam_dec.gdec0", "twin_dph_sign.gdec1", "s2b_wlam_dec",
                            "s2b_sig_comb", "s2b_a4_comb", "s2b_dph_sign")}


@pytest.mark.parametrize("name", sorted(G8C))
def test_g8c_materialize_equals_dense_diagonal(name):
    materialize_equals_dense(G8C[name])


@pytest.mark.parametrize("name", ["fuzz45", "fuzz37_x", "fuzz51_open_x", "fuzz59_open_x", "twin_gate1_y",
                                  "twin_dup_gauge", "seed23"])
def test_g8c_materialize_equals_dense_ppr(name):
    """kind 2 (materialize_terms_: Pi_{j in S} rot(B_j) then P_rec) and kind 0 vs DENSE; the PPR exact route's
    state is replaced by DENSE (an independent derivation: the term axes applied as matrices)."""
    s, R, pick = materialize_equals_dense(PPRS[name], shots=80)
    assert any(R.kind(i) == 2 for i in pick) or name.startswith("twin_dup"), name


@pytest.mark.parametrize("sd", K0G_OPEN[:4])
def test_g8c_materialize_equals_dense_out_of_universe(sd):
    s, R, pick = materialize_equals_dense(biased_fuzz_records(sd, True, p=0.05), shots=120, seed=9)
    assert any(R.route(i) == 1 for i in pick)


@pytest.mark.parametrize("sd", [14, 25, 50, 76])
def test_g8c_kind2_rotation_sign_witnesses(sd):
    """MUT-RET-ROT's witnesses. rot(B, -1) = rot(B).(-iB), so the mutated materialize_terms_ state is P_rec B_S C0|bare>
    forced at y: a Pauli to the right of C0 only relabels the anchor sectors, which the forcing re-fixes, so EVERY
    record expectation is unchanged (G-8a is blind to it by construction) and only the full state can see it — where
    B_S's action survives the record projection. Probed over the [A] corpus (X_ERROR 0.05): open_x seeds 14, 25, 50,
    76 of 0-89 (none of the mirror corpus). DENSE on these must hold on kind-2 shots."""
    s, R, pick = materialize_equals_dense(biased_fuzz_records(sd, True, p=0.05), shots=120, seed=9)
    assert any(R.kind(i) == 2 for i in pick)


@pytest.mark.parametrize("name,P", [("l4", "X"), ("l4", "Y"), ("c6", "Y"), ("strip", "X")])
def test_g8c_materialize_equals_dense_frames(name, P):
    materialize_equals_dense(FR_FIX[name], P=P)


def test_a2_fixture_is_the_a2_case():
    """JA-2: the identity plan projects the PIVOT Z0 X1 (detector 1) before the introducer X0 Z1 (observable 0 /
    the twin's detector 2) whose K involves it, and the record law gives P(Q2 = 0) = (1 + <X>)/2 = 0.854 (the Lambda
    projection after the pivot would give 0.5: MUT-ANCH-LAM)."""
    for text, intro in ((A2, 1 << 24), (A2_GAUGE, 2)):
        s = F.family(text)
        ps = s.plan_structure(b"")
        order = ps["rec_order"]
        assert [order[c] for c in ps["pivot_rec"]] == [1], ps["pivot_rec"]
        assert [order[c] for c in ps["born_rec"]] == [intro], ps["born_rec"]
        s, R, _ = run(text, 4000, 3)
        col = list(order).index(intro)
        p0 = 1.0 - np.mean([int(R.y(i)[col]) for i in range(R.num_records)])
        assert abs(p0 - (1 + 1 / math.sqrt(2)) / 2) < 5 * math.sqrt(0.25 / R.num_records), p0


# ── G-SG: signs() ─────────────────────────────────────────────────────────────────────────────────
def pauli(n, xs=(), zs=(), ys=()):
    x = np.zeros(n, np.uint8)
    z = np.zeros(n, np.uint8)
    for q in xs:
        x[q] = 1
    for q in zs:
        z[q] = 1
    for q in ys:
        x[q] = 1
        z[q] = 1
    return (x, z, len(ys) & 3)


def truth_signs(s, R, paulis):
    out = np.zeros((R.num_records, len(paulis)), np.int8)
    for i in range(R.num_records):
        e = s.state_expectations(R, i, paulis)
        out[i] = np.where(e > 1 - 1e-9, 1, np.where(e < -1 + 1e-9, -1, 0))
    return out


def check_signs(s, R, paulis):
    """signs() == the materialize-based signs: every reported sign is right, nothing definite is reported 0."""
    got = np.asarray(s.signs(R, paulis))
    want = truth_signs(s, R, paulis)
    wrong = int(np.sum((got != 0) & (got != want)))
    under = int(np.sum((got == 0) & (want != 0)))
    return wrong, under, int(np.sum(want != 0))


def _wire(ops_or_gens, k, part):
    """the single wire of operator k's x (part 0) or z (part 1) support (the deferred circuit renumbers wires)."""
    w = np.nonzero(np.asarray(ops_or_gens[k][part]))[0]
    assert len(w) == 1, w
    return int(w[0])


def test_gsg_r5check_undressed_but_indefinite():
    """adv-A r5check (2): |+> with the ANTI record Z0: <X0> = 0 although X0 is undressed (C0 = I)."""
    text = "RX 0\nR 1\nX_ERROR(0.2) 1\nM 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    s, R, _ = run(text, 200, 3)
    n = int(TA.bare_frames(s)["n"])
    w = _wire(TA.record_operators(s), 0, 1)                # the ANTI record Z_w on |+>_w
    got = np.asarray(s.signs(R, [pauli(n, xs=[w])]))
    assert np.all(got == 0)
    wrong, under, ndef = check_signs(s, R, [pauli(n, xs=[w]), pauli(n, zs=[w])])
    assert wrong == 0 and under == 0 and ndef > 0


def test_gsg_r5check_dressed_but_definite():
    """adv-A r5check (2b): a fault between T and T_DAG leaves C0 = S on |+> (dressed): <Y> is definite there."""
    text = "RX 0\nT 0\nX_ERROR(0.5) 0\nT_DAG 0\nR 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\n"
    s, R, _ = run(text, 400, 3)
    n = int(TA.bare_frames(s)["n"])
    gens = list(TA.certified_generators(s))
    w = [_wire(gens, k, 0) for k in range(len(gens)) if np.any(np.asarray(gens[k][0]))]
    assert len(w) == 1, w                                  # the |+> wire: its certified generator is X_w
    w = w[0]
    ps = [pauli(n, ys=[w]), pauli(n, xs=[w])]
    wrong, under, ndef = check_signs(s, R, ps)
    assert wrong == 0 and under == 0
    got = np.asarray(s.signs(R, ps))
    dressed = [i for i in range(R.num_records) if w in R.plan(i)["a"]]
    assert dressed and all(got[i, 0] != 0 and got[i, 1] == 0 for i in dressed), "no dressed-but-definite shot"
    assert all(got[i, 0] == 0 and got[i, 1] != 0 for i in range(R.num_records) if i not in dressed)


def test_gsg_r6check_tc_term():
    """adv-A r6check: |0>, record Z0, an X fault, g = Z0: truth -1 (y = 1, t_c = 1, <P_rec, g> = 1)."""
    text = "R 0\nX_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\n"
    s, R, _ = run(text, 200, 3)
    n = int(TA.bare_frames(s)["n"])
    w = _wire(TA.record_operators(s), 0, 1)
    got = np.asarray(s.signs(R, [pauli(n, zs=[w])]))[:, 0]
    faulted = [i for i in range(R.num_records) if R.prec(i)[0][w]]
    assert faulted and all(got[i] == -1 for i in faulted)
    assert all(got[i] == 1 for i in range(R.num_records) if i not in faulted)
    wrong, under, _ = check_signs(s, R, [pauli(n, zs=[w])])
    assert wrong == 0 and under == 0


def test_gsg_wlam_dec_generator_definite_after_the_decision():
    """MAJOR-A in signs(): on wlam_dec the decision is a record, so a certified generator it fixes is DEFINITE in
    signs() (projected), unlike sigma-4's draw; signs() equals materialize on every certified generator."""
    s, R, _ = run(S2B["wlam_dec"], 400, 3)
    gens = list(TA.certified_generators(s))
    wrong, under, ndef = check_signs(s, R, gens)
    assert wrong == 0 and under == 0 and ndef > 0


def _random_paulis(n, k, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        x = rng.integers(0, 2, n).astype(np.uint8)
        z = rng.integers(0, 2, n).astype(np.uint8)
        out.append((x, z, int(np.sum(x & z)) & 3))
    return out


SG_BULK = {k: DIAG[k] for k in sorted(DIAG) if k.startswith(("twin_", "s2b_", "design_", "a2"))}
SG_BULK.update({k: PPRS[k] for k in ("fuzz45", "fuzz59_open_x", "twin_gate1_y", "seed23")})


def _gf2_rank(A):
    A = np.array(A, np.uint8) % 2
    r = 0
    for c in range(A.shape[1]):
        piv = [i for i in range(r, A.shape[0]) if A[i, c]]
        if not piv:
            continue
        A[[r, piv[0]]] = A[[piv[0], r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return r


def _pull_kind0(p, d, n):
    """the symplectic part of C0^dag P C0 for C0 = S^a . CZ (kind 0)."""
    x = np.asarray(p[0], np.uint8).copy()
    z = np.asarray(p[1], np.uint8).copy()
    for q in d["a"]:
        z[q] ^= x[q]
    for f, g in d["cz"]:
        z[g] ^= x[f]
        z[f] ^= x[g]
    return np.concatenate([x, z])


def outside_span(s, R, i, p):
    """An independent certificate for an under-report (design §4: 'definite => +-1, but 0 may still be definite
    when alpha has an uncertified Pauli stabilizer'): C0^dag g C0 is not even in span(G u {Q_c}) (a superset of H),
    so the H-membership rule is right to report 0. Kind 0 only (the symplectic pullback through S^a . CZ)."""
    d = R.plan(i)
    assert d["kind"] == 0, "an under-report on a kind-2 plan"
    n = int(TA.bare_frames(s)["n"])
    rows = [np.concatenate([np.asarray(a, np.uint8), np.asarray(b, np.uint8)]) for a, b, _ in TA.certified_generators(s)]
    rows += [_pull_kind0(q, d, n) for q in TA.record_operators(s)]
    gp = _pull_kind0(p, d, n)
    return _gf2_rank(rows + [gp]) > _gf2_rank(rows)


@pytest.mark.parametrize("name", sorted(SG_BULK))
def test_gsg_bulk_equals_materialize(name):
    """Every reported sign equals the materialize-based sign (exact counter wrong == 0); every under-report (a sign
    definite on the state but reported 0) is certified outside span(G u Q) — the design's OPEN alpha-symmetry case,
    counted and reported."""
    s, R, _ = run(SG_BULK[name], 150, 21)
    n = int(TA.bare_frames(s)["n"])
    ps = list(TA.certified_generators(s)) + list(TA.record_operators(s)) + _random_paulis(n, 6, 5)
    got = np.asarray(s.signs(R, ps))
    want = truth_signs(s, R, ps)
    wrong = int(np.sum((got != 0) & (got != want)))
    under = np.argwhere((got == 0) & (want != 0))
    assert wrong == 0 and int(np.sum(want != 0)) > 0, (name, wrong)
    for i, k in under:
        assert outside_span(s, R, int(i), ps[int(k)]), (name, "an under-report inside span(G u Q)", int(i), int(k))
    if len(under):
        print(f"[gsg] {name}: {len(under)} under-reports (alpha symmetry, certified outside span(G u Q))")


def test_gsg_refuses_non_hermitian():
    s, R, _ = run(F.GATE1, 10, 1)
    n = int(TA.bare_frames(s)["n"])
    x, z, _ = pauli(n, xs=[0])
    with pytest.raises(ValueError, match="non-Hermitian"):
        s.signs(R, [(x, z, 1)])


# ── record_report() classes (R = I) ───────────────────────────────────────────────────────────────
def _noiseless(text):
    out = []
    for ln in text.splitlines():
        if re.match(r"^(X_ERROR|Z_ERROR|Y_ERROR|DEPOLARIZE1|DEPOLARIZE2|PAULI_CHANNEL_1|PAULI_CHANNEL_2)\(", ln):
            continue
        out.append(re.sub(r"\b(M[XYR]*|MPP)\([^)]*\)", r"\1", ln))
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("name", ["gate1", "dup_gauge", "a2", "design_p4_gauge", "design_p3_anti", "design_bell_anti",
                                  "design_obs3", "f1_F1", "ah11_ah11", "s2b_wlam_dec", "s2b_sig_comb"])
def test_record_classes_match_the_noiseless_law(name):
    """deterministic <=> the NOISELESS DENSE marginal is 0 / 1; coin => marginal exactly 1/2; and the lists agree
    with the classes (channel_report's deterministic channels are deterministic)."""
    from _record_oracle import emitted_columns, emitted_law
    text = DIAG[name]
    s = F.family(text)
    rc = s.record_classes()
    cls = rc["classes"]
    law = emitted_law(_noiseless(text))
    cols = emitted_columns(text)
    key = {"detector": "detectors", "observable": "observables", "decision": "decisions"}
    for k, (kind, idx, _) in enumerate(cols):
        p1 = sum(p for o, p in law.items() if o[k])
        c = cls[key[kind]][idx]
        assert (c == "deterministic") == (abs(p1) < 1e-12 or abs(p1 - 1) < 1e-12), (name, kind, idx, c, p1)
        if c == "coin":
            assert abs(p1 - 0.5) < 1e-12, (name, kind, idx, p1)
    assert rc["nondeterministic_detectors"] == sorted(i for i, c in cls["detectors"].items() if c != "deterministic")
    assert rc["nondeterministic_observables"] == sorted(i for i, c in cls["observables"].items() if c != "deterministic")
    rep = s.channel_report()
    assert all(cls["detectors"][d] == "deterministic" for d in rep["deterministic_detectors"])


# ── §3.10: zero allocations per shot on the recording run (layout-3 write, every route) ───────────
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
    for _ in range(2):
        s.sample_records(n2, seed=5); s.sample_records(n1, seed=5)
    a2 = counted(lambda: s.sample_records(n2, seed=5))
    a1 = counted(lambda: s.sample_records(n1, seed=5))
    a2b = counted(lambda: s.sample_records(n2, seed=5))
    res[name] = {"a1": a1, "a2": a2, "a2b": a2b, "n1": n1, "n2": n2}
print(json.dumps(res))
"""


@pytest.mark.skipif(platform.system() != "Linux" or platform.libc_ver()[0] != "glibc",
                    reason="needs Linux/glibc (LD_PRELOAD counting allocator)")
def test_zero_allocations_per_shot_recording_run(tmp_path):
    from test_zero_alloc_per_shot import SHIM, _cc
    if _cc() is None:
        pytest.skip("no C compiler")
    (tmp_path / "za.c").write_text(SHIM)
    so = tmp_path / "libza.so"
    subprocess.run([_cc(), "-O2", "-fPIC", "-shared", str(tmp_path / "za.c"), "-o", str(so)], check=True)
    import xtim
    pkg_parent = str(Path(xtim.__file__).resolve().parents[1])
    wl = [["dup_gauge", F.DUP_GAUGE, 4000, 12000],
          ["ch_rec", PPRS["ch_rec"], 4000, 12000],
          ["open1_k0g", K0G["k0g_open1"], 4000, 12000]]
    env = dict(os.environ, LD_PRELOAD=str(so))
    r = subprocess.run([sys.executable, "-I", "-c", ALLOC_CHILD, pkg_parent, str(so), json.dumps(wl)], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    for name, c in res.items():
        assert c["a1"][0] > 0, (name, c)
        assert c["a2"][0] == c["a1"][0], (name, "allocations per shot", (c["a2"][0] - c["a1"][0]) / (c["n2"] - c["n1"]), c)
        assert c["a2b"] == c["a2"], (name, c)


def test_retain_still_refuses_sigma_export():
    s = F.family(F.GATE1)
    with pytest.raises(RuntimeError, match="sigma export is refused on a family sampler"):
        s.sample(10, retain=True)


# ── G-8d: the state-model identity tuple (gauge-plan G-3) ─────────────────────────────────────────
@pytest.mark.parametrize("variant", ["invert", "operator"])
def test_g8d_identity_tuple_refuses_a_gauge_only_difference(variant):
    """Records are read (materialize / signs) only by a sampler with the producer's state model: the same circuit
    compiled again is accepted; a circuit differing ONLY in a gauge record (its invert, or its operator) is refused
    by name — the identity tuple includes every record operator with its inv / ref fold."""
    a_text = F.CONVENTION["p10_gauge"]                       # the gauge read MX !0 (inv = ref = 1)
    b_text = a_text.replace("MX !0", "MX 0") if variant == "invert" else a_text.replace("MX !0", "MY !0")
    A, A_again, B = F.family(a_text), F.family(a_text), F.family(b_text)
    assert B.channel_report()["family_extras"], "the variant must differ in a FAMILY (gauge) record"
    _, _, _, R = A.sample_records(20, seed=1)
    n = int(TA.bare_frames(A)["n"])
    A_again.materialize(R, 0)
    A_again.signs(R, [pauli(n, zs=[0])])
    with pytest.raises(ValueError, match="different state model"):
        B.materialize(R, 0)
    with pytest.raises(ValueError, match="different state model"):
        B.signs(R, [pauli(n, zs=[0])])


# ── J-a final review m3: the layout-3 content map is keyed by the TAGGED key ─────────────────────────
def test_content_handles_are_keyed_by_kind_and_bytes():
    """Kind 0 is `a ‖ 0xFF ‖ cz pairs` (n + 1 + 4p bytes), kind 2 the TW-word term bitset: equal bytes under the
    two kinds are two contents (1c88eb1: 'one content key under two kinds (engine bug)' mid-run)."""
    text = "RX 0 1 2 3 4 5 6\nT 0\nMX 0 1 2 3 4 5 6\nDETECTOR rec[-1]\n"   # n = 7, one term: TW * 8 = 8 = n + 1
    s = F.family(text)
    c = b"\x00" * 7 + b"\xff"
    h0 = s._intern_content(0, c)
    h2 = s._intern_content(2, c)
    assert h0 != h2
    assert s._intern_content(0, c) == h0 and s._intern_content(2, c) == h2
