"""J-a N2: the term-indexed residual and key (cpp/src/term_residual.cpp), gates G-TK1..G-TK6 (plan §3.4,
extended by r2.4 / JA-4).

The term route lives on FAM samplers only (xtim._xtim_testapi.compile_family_sampler; switched OFF in
production). Oracles, all independent of term_residual.cpp:
  * (X) = RecordLawOracle (cpp/testapi/record_law_oracle.cpp): `terms()` and `residual()` (one forward pass
    over the deferred circuit) — G-TK1;
  * dense 2^n matrices (numpy) — G-TK2;
  * a pure-Python Pauli algebra over (X)'s term axes: the symplectic action of rot(S) (G-TK3, G-TK4) and
    the diagonal normal form of a composite P·rot(S) read off its image signs (G-TK5);
  * the diagonal route itself (compose_fired -> residual_normal_form -> canonicalize_mod_stabilizers, the
    production self-check chain) — G-TK5.

Fixtures (plan §3.0): FX-PPR = tests/data/ch_cultivation.stim, the adv-A fuzz circuits C1 T C2 noise C2^-1
T^-1 C1^-1 (seed 23 = tests/data/ja_seed23.stim, and 37 / 45 / 51 / 59 / 67 / 86); FX-fuzz =
_record_law_x.random_circuit (diag / ppr / mirror / a4 flavours) and the S2b family_circuit /
dressed_circuit; the diagonal corpus = the S2b FIXTURES, family_circuit, dressed_circuit and the "diag"
random circuits. The plan's e7 / t0 / t1_SD6 circuits live outside the repository (the ledger's @rule
tree) and are not used here.
"""
import os
import random
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _record_law_x import random_circuit  # noqa: E402
from test_s2b_record_table import FIXTURES, dressed_circuit, family_circuit  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED23 = open(os.path.join(ROOT, "tests", "data", "ja_seed23.stim")).read()
CH_CULT = open(os.path.join(ROOT, "tests", "data", "ch_cultivation.stim")).read()


# ── the adv-A fuzz family (unify/adv-A/fuzz.py `build`, verbatim): seed 23 == tests/data/ja_seed23.stim ──
def _rc(n, d, r):
    g = []
    for _ in range(d):
        k = r.randrange(4)
        if k == 0:
            g.append(("H", [r.randrange(n)]))
        elif k == 1:
            g.append(("S", [r.randrange(n)]))
        else:
            a, b = r.sample(range(n), 2)
            g.append(("CX" if k == 2 else "CZ", [a, b]))
    return g


_INV = {"H": "H", "S": "S_DAG", "CX": "CX", "CZ": "CZ", "T": "T_DAG"}


def _emit(gs, p):
    out = []
    for name, q in gs:
        out.append(f"{name} " + " ".join(map(str, q)))
        if p:
            out.append(f"DEPOLARIZE{len(q)}({p}) " + " ".join(map(str, q)))
    return out


def fuzz_circuit(seed, n=4):
    r = random.Random(seed)
    C1 = _rc(n, 8, r)
    C2 = _rc(n, 8, r)
    tq = r.sample(range(n), r.randrange(1, n + 1))
    T = [("T", [q]) for q in tq]

    def dag(gs):
        return [(_INV[a], q) for a, q in reversed(gs)]

    L = ["R " + " ".join(map(str, range(n)))]
    L += _emit(C1, 0.02) + _emit(T, 0) + _emit(C2, 0.02) + _emit(dag(C2), 0.02) + _emit(dag(T), 0) + _emit(dag(C1), 0.02)
    L.append("M " + " ".join(map(str, range(n))))
    for i in range(n):
        L.append(f"DETECTOR rec[-{i + 1}]")
    L.append("DETECTOR " + " ".join(f"rec[-{i + 1}]" for i in range(n)))
    return "\n".join(L) + "\n"


FX_PPR = {"ch_cultivation": CH_CULT, "fuzz23": SEED23}
for _sd in (37, 45, 51, 59, 67, 86):
    FX_PPR[f"fuzz{_sd}"] = fuzz_circuit(_sd)




def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def try_fam(text):
    """The FAM sampler of `text`, or None when it is refused by name."""
    try:
        return fam(text)
    except ValueError:
        return None


# ── an independent Pauli algebra: (x, z, ph) = i^ph X^x Z^z, x / z int bitmasks ─────────────────────
def _pc(v):
    return bin(v).count("1")


def pmul(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1], (a[2] + b[2] + 2 * _pc(a[1] & b[0])) % 4)


def anti(a, b):
    return (_pc(a[0] & b[1]) + _pc(a[1] & b[0])) & 1


def bits_int(row):
    v = 0
    for i, b in enumerate(row):
        if b:
            v |= 1 << i
    return v


def xo_axes(orc):
    xz, ph, _ = orc.terms()
    xz = np.asarray(xz)
    n = orc.n
    return [(bits_int(xz[j, :n]), bits_int(xz[j, n:]), int(ph[j]) % 4) for j in range(xz.shape[0])]


def rot_image(B, S, V):
    """rot(S) V rot(S)^dag, exact phase: each anticommuting rot(B) multiplies by iB on the left."""
    R = V
    for j in S:
        if anti(B[j], V):
            R = pmul((B[j][0], B[j][1], (B[j][2] + 1) % 4), R)
    return R


def action(B, S, n):
    """The symplectic action of rot(S) (images of X_q, Z_q mod sign)."""
    out = []
    for q in range(n):
        for V in ((1 << q, 0, 0), (0, 1 << q, 0)):
            im = rot_image(B, S, V)
            out.append((im[0], im[1]))
    return tuple(out)


def ref_diag_nf(B, n, Px, Pz, S):
    """The diagonal normal form of R = P . rot(S): None when rot(S) is not diagonal (some Z_q is not fixed up
    to sign); else (prefix_x, prefix_z, a, cz) with R = X^x Z^z . S^a . CZ(cz) mod phase, read off R's image
    signs: R Z_q R^dag = (-1)^{x_q} Z_q; R X_q R^dag = (-1)^{z_q + a_q x_q + sum_l cz_ql x_l} D X_q D^dag."""
    P = (Px, Pz, 0)
    px = 0
    for q in range(n):
        im = rot_image(B, S, (0, 1 << q, 0))
        if im[0] != 0 or im[1] != 1 << q:
            return None
        assert im[2] in (0, 2)
        if (im[2] == 2) ^ anti(P, im):
            px |= 1 << q
    pz, a, cz = 0, [], []
    for q in range(n):
        im = rot_image(B, S, (1 << q, 0, 0))
        assert im[0] == 1 << q
        s_q = (im[1] >> q) & 1
        Bq = im[1] & ~(1 << q)
        d = (im[2] - s_q) % 4
        assert d in (0, 2)
        neg = (d == 2) ^ anti(P, im)
        if neg ^ (s_q & (px >> q) & 1) ^ (_pc(Bq & px) & 1):
            pz |= 1 << q
        if s_q:
            a.append(q)
        cz += [(q, l) for l in range(q + 1, n) if (Bq >> l) & 1]
    return px, pz, a, cz


def nf_tuple(nf):
    return (bits_int(nf["prefix"][0]), bits_int(nf["prefix"][1]), list(nf["a"]), [tuple(p) for p in nf["cz"]])


def configs(s, shots, seed, nrand, rseed):
    """Fault configurations: the distinct non-empty shots as the sampler draws them, plus random subsets of
    alternatives (ids in construction = stream order)."""
    seen, out = set(), []
    for e in s.term_events(shots, seed):
        t = tuple(e)
        if t and t not in seen:
            seen.add(t)
            out.append(list(t))
    r = random.Random(rseed)
    A = len(s.term_alt_faults())
    for _ in range(nrand if A else 0):
        out.append(sorted(r.sample(range(A), min(A, r.randrange(1, 5)))))
    return out


def fx_fuzz():
    out = {}
    for sd in range(40):
        out[f"rc{sd}"] = random_circuit(sd)[0]
    for sd in range(12):
        out[f"family{sd}"] = family_circuit(sd)
        out[f"dressed{sd}"] = dressed_circuit(sd)
    return out


# ── G-TK1: compose_terms_ == (X) residual(), exactly ─────────────────────────────────────────────────
def _check_composition(name, text, shots, nrand):
    s = try_fam(text)
    if s is None:
        return None
    orc = TA.RecordLawOracle(text)
    xz, ph = s.term_axes()
    oxz, oph, _ = orc.terms()
    assert np.array_equal(np.asarray(xz), np.asarray(oxz)), name           # the same term DEFINITION
    assert np.array_equal(np.asarray(ph), np.asarray(oph)), name
    B = xo_axes(orc)
    rows = s.term_rows()                                                    # records anticommuting with B_j
    for j, Bj in enumerate(B):
        want = [r for r, (basis, w) in enumerate(orc.reads)
                if anti(Bj, ((1 << w) if basis in (0, 1) else 0, (1 << w) if basis in (1, 2) else 0, 0))]
        assert list(rows[j]) == want, (name, j)
    op_pos = {st["stream_index"]: st["op_pos"] for st in orc.noise_sites()}
    af = s.term_alt_faults()
    bad, n = [], 0
    for alts in configs(s, shots, 11, nrand, 5):
        d = s.term_shot(alts)
        faults = [(op_pos[loc], [q], [c]) for a in alts for (loc, q, c) in af[a]]
        So, Px, Pz = orc.residual(faults)
        n += 1
        if (sorted(d["S"]) != np.nonzero(So)[0].tolist() or not np.array_equal(d["P"][0], Px)
                or not np.array_equal(d["P"][1], Pz)):
            bad.append(alts)
    assert not bad, f"{name}: compose_terms_ != (X) residual on {len(bad)} of {n} configurations, e.g. {bad[:3]}"
    return n


def test_gtk1_composition_matches_x_on_fx_ppr():
    total = 0
    for name, text in FX_PPR.items():
        n = _check_composition(name, text, 20000, 300)
        assert n is not None, f"{name}: refused"
        total += n
    assert total > 5000


def test_gtk1_composition_matches_x_on_fx_fuzz():
    total, circuits, with_terms = 0, 0, 0
    for name, text in fx_fuzz().items():
        n = _check_composition(name, text, 2000, 60)
        if n is None:
            continue
        circuits += 1
        total += n
        with_terms += TA.RecordLawOracle(text).num_terms > 0
    assert circuits >= 50 and with_terms >= 45 and total > 3000, (circuits, with_terms, total)


# ── G-TK2: the sequential kernel fold (adv-A kfold.py) ───────────────────────────────────────────────
def _dense_pauli(n, x, z, ph=0):
    d = 1 << n
    M = np.zeros((d, d), complex)
    for b in range(d):
        M[b ^ x, b] = (1j ** ph) * (-1) ** _pc(z & b)
    return M


def _dense_rot(n, B, S):
    R = np.eye(1 << n, dtype=complex)
    for j in sorted(S):
        R = R @ (np.eye(1 << n) + 1j * _dense_pauli(n, *B[j])) / np.sqrt(2)
    return R


def _eq_phase(A, B):
    i = np.argmax(abs(B))
    if abs(B.flat[i]) < 1e-12:
        return False
    ph = A.flat[i] / B.flat[i]
    return abs(abs(ph) - 1) < 1e-9 and np.allclose(A, ph * B)


def _termset(n, B):
    xz = np.zeros((len(B), 2 * n), np.uint8)
    for j, (x, z, _) in enumerate(B):
        for q in range(n):
            xz[j, q] = (x >> q) & 1
            xz[j, n + q] = (z >> q) & 1
    return TA.TermSet(n, xz, [p for (_, _, p) in B])


def test_gtk2_kfold_two_vector_basis():
    n = 4
    zs = [0b1111] + [(1 << a) | (1 << b) for a in range(4) for b in range(a + 1, 4)] + [0b0011]
    B = [(0, z, 0) for z in zs]                 # index 7 duplicates index 1 (Z1 Z2)
    ts = _termset(n, B)
    ker = ts.kernel()
    assert [(j, k) for (j, k, _) in ker] == [(6, list(range(7))), (7, [1, 7])]   # kfold's k1, k2
    seq_ok = red_ok = 0
    for m in range(256):
        S = [j for j in range(8) if (m >> j) & 1]
        truth = _dense_rot(n, B, S)
        (px, pz), S2 = ts.fold_sequence(S, [0, 1])
        if _eq_phase(truth, _dense_pauli(n, bits_int(px), bits_int(pz)) @ _dense_rot(n, B, S2)):
            seq_ok += 1
        (px, pz), Sr = ts.reduce(S)
        if _eq_phase(truth, _dense_pauli(n, bits_int(px), bits_int(pz)) @ _dense_rot(n, B, Sr)) and not {6, 7} & set(Sr):
            red_ok += 1
    assert (seq_ok, red_ok) == (256, 256)


def test_gtk2_mirror_pair_no_silent_z():
    ts = _termset(1, [(0, 1, 0), (0, 1, 0)])    # B1 = B2 = Z: k = {0, 1}, N_k = iZ
    (j, k, (nx, nz)), = ts.kernel()
    assert (j, k, list(nx), list(nz)) == (1, [0, 1], [0], [1])
    (px, pz), Sr = ts.reduce([1])
    assert Sr == [0] and list(px) == [0] and list(pz) == [0]      # rot(B2) = rot(B1): no Z
    assert _eq_phase(_dense_rot(1, [(0, 1, 0)] * 2, [1]), _dense_rot(1, [(0, 1, 0)] * 2, [0]))


# ── G-TK3: equal Clifford mod Pauli <=> equal key ────────────────────────────────────────────────────
def test_gtk3_key_is_canonical_on_fx_ppr():
    pairs = 0
    collapsed = 0
    for name, text in FX_PPR.items():
        s = fam(text)
        B = xo_axes(TA.RecordLawOracle(text))
        n = TA.RecordLawOracle(text).n
        key_act, act_key, raw = {}, {}, set()
        for alts in configs(s, 20000, 13, 200, 7):
            d = s.term_shot(alts)
            act = action(B, d["S"], n)                     # from the COMPOSED S (independent of the reduction)
            k = d["key"]
            raw.add(tuple(d["S"]))
            assert key_act.setdefault(k, act) == act, f"{name}: one key, two Cliffords mod Pauli"
            assert act_key.setdefault(act, k) == k, f"{name}: one Clifford mod Pauli, two keys"
        pairs += len(key_act)
        collapsed += len(raw) - len(key_act)
    assert pairs > 20 and collapsed > 0, (pairs, collapsed)


# ── G-TK4: exponent collapse on seed 23 ──────────────────────────────────────────────────────────────
def test_gtk4_seed23_plan_collapse():
    s = fam(SEED23)
    orc = TA.RecordLawOracle(SEED23)
    B = xo_axes(orc)
    ppr = s.term_alt_is_ppr()
    raw, keys, acts, ids = set(), set(), set(), set()
    memo = {}
    for e in s.term_events(200000, 5):
        if not e or not any(ppr[a] for a in e):
            continue
        t = tuple(e)
        if t not in memo:
            memo[t] = s.term_shot(e)
        d = memo[t]
        raw.add(tuple(d["S"]))
        keys.add(d["key"])
        acts.add(action(B, d["S"], orc.n))
        ids.add(d["plan_id"])
    # 16 distinct term sets reach 4 Cliffords mod Pauli (rank Phi = 2 of 4 terms: two mirror pairs); the
    # key is exactly that count; the plan ids (the kind-0 shots through the stripped (a, cz) key) are 3.
    assert (len(raw), len(acts), len(keys), len(ids)) == (16, 4, 4, 3)
    assert s.term_rank() == 2 and len(s.term_kernel()) == 2


# ── G-TK5: kind / plan identity (r2.4, JA-4) ─────────────────────────────────────────────────────────
def diag_corpus():
    out = dict(FIXTURES)
    for sd in range(20):
        out[f"family{sd}"] = family_circuit(sd)
    for sd in range(8):
        out[f"dressed{sd}"] = dressed_circuit(sd)
    for sd in range(0, 80, 4):                  # the "diag" flavour
        out[f"rc{sd}"] = random_circuit(sd)[0]
    return out


def _route_equal(s, alts, d):
    """The term route's (key, P', plan id, t(P')) equals the diagonal route's for the same alternatives."""
    g = s.diag_shot(alts)
    assert g is not None
    assert d["kind"] == 0, alts
    assert nf_tuple(d["nf"]) == nf_tuple(g), alts
    assert d["plan_id"] == g["plan_id"], alts
    tx = s.term_relabel(*d["nf"]["prefix"])
    assert np.array_equal(tx, s.term_relabel(*g["prefix"])), alts
    return g


def test_gtk5_term_route_equals_diagonal_route_on_the_diagonal_corpus():
    shots_checked, circuits, nontrivial = 0, 0, 0
    for name, text in diag_corpus().items():
        s = try_fam(text)
        if s is None:
            continue
        orc = TA.RecordLawOracle(text)
        B, n = xo_axes(orc), orc.n
        op_pos = {st["stream_index"]: st["op_pos"] for st in orc.noise_sites()}
        af = s.term_alt_faults()
        if any(s.term_alt_is_ppr()):            # J-b2 J6: a CH/PPR corpus member with a Born decision now builds
            continue                            # (formerly refused without ppr_records): not a diagonal circuit
        circuits += 1
        for alts in configs(s, 3000, 17, 40, 9):
            d = s.term_shot(alts)
            _route_equal(s, alts, d)
            So, Px, Pz = orc.residual([(op_pos[l], [q], [c]) for a in alts for (l, q, c) in af[a]])
            ref = ref_diag_nf(B, n, bits_int(Px), bits_int(Pz), np.nonzero(So)[0].tolist())
            assert ref is not None and nf_tuple(d["nf_raw"]) == (ref[0], ref[1], ref[2], ref[3]), (name, alts)
            shots_checked += 1
            nontrivial += bool(d["nf"]["a"] or d["nf"]["cz"])
    assert circuits >= 60 and shots_checked > 2500 and nontrivial > 1000, (circuits, shots_checked, nontrivial)


def test_gtk5_mixed_shots_share_the_diagonal_plan_on_fx_ppr():
    kind0 = mixed0 = alldiag = natural = constructed = 0
    for name, text in FX_PPR.items():
        s = fam(text)
        orc = TA.RecordLawOracle(text)
        B, n = xo_axes(orc), orc.n
        op_pos = {st["stream_index"]: st["op_pos"] for st in orc.noise_sites()}
        af = s.term_alt_faults()
        ppr = s.term_alt_is_ppr()
        by_act = {}                     # action -> plan id of the all-diagonal shots reaching it
        mixed = []
        for alts in configs(s, 20000, 19, 200, 3):
            d = s.term_shot(alts)
            So, Px, Pz = orc.residual([(op_pos[l], [q], [c]) for a in alts for (l, q, c) in af[a]])
            S = np.nonzero(So)[0].tolist()
            ref = ref_diag_nf(B, n, bits_int(Px), bits_int(Pz), S)
            assert (d["kind"] == 0) == (ref is not None), (name, alts)      # the kind is exact
            if ref is None:
                continue
            kind0 += 1
            assert nf_tuple(d["nf_raw"]) == (ref[0], ref[1], ref[2], ref[3]), (name, alts)
            act = action(B, S, n)
            if any(ppr[a] for a in alts):
                mixed0 += 1
                mixed.append((act, d["plan_id"]))
            else:
                alldiag += 1
                _route_equal(s, alts, d)
                assert by_act.setdefault(act, d["plan_id"]) == d["plan_id"], name
        for act, pid in mixed:          # a mixed shot reaching an all-diagonal shot's C0: one plan id
            if act in by_act:
                natural += 1
                assert pid == by_act[act], name
        # constructed: R_a R_d R_a (a PPR alternative fired twice around a diagonal one) reaches C0(d)
        dg = [a for a in range(len(ppr)) if not ppr[a]][:6]
        for a in [a for a in range(len(ppr)) if ppr[a]][:6]:
            for dd in dg:
                g = s.diag_shot([dd])
                d = s.term_shot([a, dd, a])
                assert d["kind"] == 0 and d["plan_id"] == g["plan_id"], (name, a, dd)
                constructed += 1
    print("G-TK5 FX-PPR counters", kind0, mixed0, alldiag, natural, constructed)
    assert kind0 > 5000 and mixed0 > 500 and alldiag > 1000 and natural > 100 and constructed > 100, (
        kind0, mixed0, alldiag, natural, constructed)


# ── G-TK6: the term-class boundary (TermResidual's ONE commutation check, a typed TermClassRefusal) ──
def test_gtk6_noncommuting_pair_throws_the_named_assert():
    with pytest.raises(RuntimeError, match="TERM CLASS REFUSAL"):
        _termset(1, [(1, 0, 0), (0, 1, 0)])         # X and Z on one wire
    with pytest.raises(RuntimeError, match="TERM CLASS REFUSAL"):
        _termset(2, [(0b01, 0b10, 0), (0b10, 0b00, 0)])   # X0 Z1 and X1


# J-b1 P5 (plan r3.1, ruling on fixplan-nc-cs-class): the owner's non-Clifford rule (the STATIC fixed-axis merge,
# then the commutation check) runs at CONSTRUCTION on every family sampler, once the alternatives exist; a circuit
# whose merged terms still anticommute is refused by name there (setup_error 7) with the law-limitation text, whatever
# the classes. These fixtures are the M-NC class "fault-dependent anticommuting terms": their terms anticommute only
# through a fault between a same-axis pair. (Pre-P5 the check ran only where the term route was armed; unarmed
# samplers built and production served them.) Mutations: XTIM_MUTATE_TC_SKIP (the check skipped) -> red here.
TC_REFUSAL = (r"twirl sampler: law limitation: fault-dependent surviving axis/Clifford frame: terms \d+ \(deferred gate "
              r"\d+\) and \d+ \(deferred gate \d+\) merge only when no fault occurs at deferred noise site \d+, so the "
              r"surviving term \(or the Clifford frame\) depends on the faults; the circuit is T-depth <= 1 per shot but "
              r"outside the fixed-axis term law \(a per-shot fold is not built\)")
NONCOMM = {
    "ch_pair": "H 5\nT 5\nCH 5 1\nH 0\nCH 1 0\nX_ERROR(0.35) 0\nCH 1 0\nMX 0\nDETECTOR rec[-1]\n",
    "t_h_ch": "T 0\nH 0\nH 1\nCH 0 1\nX_ERROR(0.35) 1\nCH 0 1\nMX 1\nDETECTOR rec[-1]\n",
    "t_h_ch_decision": ("T 0\nH 0\nH 1\nCH 0 1\nX_ERROR(0.35) 1\nCH 0 1\nMX 1\nDETECTOR rec[-1]\n"
                        "MX 0\nDECISION(0) rec[-1]\n"),
}
# ch_pair with an input port (a Clifford-only wire): the frames-class arming row
NONCOMM_PORT = "INPUT_QUBITS in 9\n" + NONCOMM["ch_pair"] + "MX 9\n"


def _prod_bytes(text, n, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = xtim.compile_twirl_sampler(text, selfcheck=0).sample(n, seed=seed)
    if hasattr(r, "decisions"):
        return [np.asarray(r.dets), np.asarray(r.obs), np.asarray(r.decisions)]
    return [np.asarray(r[0]), np.asarray(r[1])]


def _assert_prod_bytes(s, text, n=4000, seed=9):
    out = s.sample(n, seed=seed)
    exp = _prod_bytes(text, n, seed)
    for a, b in zip(out, exp):
        assert np.array_equal(np.asarray(a), b)


# ── JA-4 structure gate: one normal-form function ────────────────────────────────────────────────────
def _strip_mutation_blocks(src):
    """Drop `#ifdef XTIM_MUTATE_*` branches (keeping an `#else` branch) and comments."""
    out, stack = [], []
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("#ifdef XTIM_MUTATE") or t.startswith("#ifndef XTIM_MUTATE"):
            stack.append(["mut" if t.startswith("#ifdef") else "keep", t.startswith("#ifdef")])
            continue
        if stack and t.startswith("#else"):
            stack[-1][0] = "keep" if stack[-1][0] == "mut" else "mut"
            continue
        if stack and t.startswith("#endif"):
            stack.pop()
            continue
        if any(st[0] == "mut" for st in stack):
            continue
        out.append(line)
    src = "\n".join(out)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def test_ja4_term_route_has_no_normal_form_of_its_own():
    src = _strip_mutation_blocks(open(os.path.join(ROOT, "cpp", "src", "term_residual.cpp")).read())
    assert "residual_normal_form(" in src and "canonicalize_mod_stabilizers_inplace(" in src
    for pat in (r"\.prefix\b", r"\.cz\b", r"\.diagonal_class\b", r"\bnf\w*\.a\b", r"DiagNormalForm\s+\w+\s*;"):
        assert not re.search(pat, src), f"term_residual.cpp extracts a normal form of its own ({pat})"
    obj = os.path.join(ROOT, "build", "temp.linux-x86_64-cpython-312", "cpp", "src", "term_residual.o")
    if not os.path.exists(obj):
        pytest.skip("no in-place build object for the nm check")
    nm = subprocess.run(["nm", "-C", obj], capture_output=True, text=True, check=True).stdout
    und = [ln for ln in nm.splitlines() if ln.strip().startswith("U ")]
    assert any("qeccore::residual_normal_form(qeccore::DiagPauliClifford const&)" in ln for ln in und)
    assert any("qeccore::canonicalize_mod_stabilizers_inplace(" in ln for ln in und)
    defined = [ln for ln in nm.splitlines() if " T " in ln or " W " in ln]
    assert not any("residual_normal_form" in ln or "canonicalize_mod_stabilizers" in ln for ln in defined)


def test_xo_stays_an_oracle_nothing_under_cpp_src_includes_testapi():
    hits = []
    for d in ("src", "include"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, "cpp", d)):
            for f in files:
                for ln in open(os.path.join(dirpath, f), errors="replace"):
                    if re.match(r"\s*#\s*include\s*[<\"].*(testapi|record_law_oracle)", ln):
                        hits.append((f, ln.strip()))
    assert not hits, hits
