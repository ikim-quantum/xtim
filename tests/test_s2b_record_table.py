"""S2b gate L-3: the engine's per-plan RECORD TABLE equals the S1 oracle's Theorem-1 law, field by field.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

The engine builds, for every diagonal plan, the record table in the record-ordered canonical basis
(cpp/src/record_table.cpp: channels from the σ law, the Born observable and Born decisions by
add_extra, one canonicalization in global record order). The oracle (X)
(`xtim._xtim_testapi.RecordLawOracle`, cpp/testapi/record_law_oracle.cpp) computes the same law
independently, in record space, from the term-indexed residual. For every fault configuration of
each fixture, the oracle's term set S is turned into the engine's diagonal plan (S^a · CZ, from S's
phase polynomial) and BOTH tables are compared exactly:

    pivots (record order), Born introducers, A, B, beta, and every direction Lambda_j
    (free-register a/b masks and EFFECTIVE phase: the literal-product convention).

Only circuits whose every record column is in the engine's family are compared (no gauge / ANTI
detectors, no refused observables): the canonical basis is defined over the whole family.
"""
import itertools
import struct

import numpy as np
import pytest

import xtim._xtim as _x
from _record_law_x import fault_configs, oracle, random_circuit

M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
      "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
      "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
H = "R 0\nRX 1 2\nT 2\nX_ERROR(0.5) 0\nCCZ 0 1 2\nM 2\n"
FIXTURES = {
    "m3_obs": M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET,
    "m3_obs_dec": M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET + "MX 0\nDECISION(0) rec[-1]\n",
    "m3_obs_2dec": M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET + "MX 0\nDECISION(0) rec[-1]\nMY 1\nDECISION(1) rec[-1]\n",
    "m3_det": M3 + "MX 3\nMX 4\n" + DET,
    "ccz_obs": "RX 0 1 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nRX 3\nT 3\nCX 3 1\nMX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET,
    "q3_dec": ("R 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\nMY 0\nMPP Z0\nMPP Z0\nH 3\nT 3\nX_ERROR(0.5) 2\nCCZ 0 3 2\n"
               "DECISION(0) rec[-3]\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\n"),
    "wlam": H + "OBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "wlam_dec": H + "DECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "sig_ovl": H + "DECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "span_obs_dec": H + "OBSERVABLE_INCLUDE(0) rec[-1]\nMX 1\nDECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n",
    "span_dec_mix": H + ("DECISION(0) rec[-1]\nMX 1\nR 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\n"
                         "DECISION(1) rec[-2] rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    "sig_comb": ("R 0\nRX 1 2 3 4\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nCCZ 0 4 2\n"
                 "M 2\nDECISION(0) rec[-1]\nM 3\nDECISION(1) rec[-1]\nMX 1\nDECISION(2) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    "sig_comb_x": ("R 0\nRX 1 2 3 4 5\nCCZ 2 3 5\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nCCZ 0 4 2\n"
                   "MX 5\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                   "M 2\nDECISION(0) rec[-1]\nM 3\nDECISION(1) rec[-1]\nMX 1\nDECISION(2) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # S-2 / M-SIGN: a dependent extra whose operator is MINUS the literal product of the earlier
    # direction (X0X1 read twice around a Z0 flip): add_extra's ordered-product sign s_c = 1
    "dph_sign": ("RX 0 1\nT 0\nT 1\nMPP X0*X1\nZ 0\nMPP X0*X1\nDECISION(0) rec[-2]\nDECISION(1) rec[-1]\n"
                 "R 9\nM 9\nDETECTOR rec[-1]\n"),
    # departures review MAJOR-B (MP-1 end to end): W = X2X3 introduces a direction first, the Born
    # decision Y2Y3 = -(X2X3)(Z2Z3) introduces the COMBINED literal product with the in-group
    # decision's Z2Z3 direction; W ^ d0 ^ d1 = 1 and P(d0 | W = 0) = 1/3, P(d0 | W = 1) = 0 on the
    # faulted plan
    "a4_comb": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nMPP X2*X3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                 "MPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # code review MAJOR-1 (M-BETA end to end): qubit 1 in |-> (certified generator -X1), so the
    # in-group decision MX 1 introduces its channel direction with the fold bit beta' = 1; on the
    # faulted plan W ^ d0 ^ d1 = 0
    "a4_minus_z": ("R 0\nRX 1 2 3\nZ 1\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nMPP X2*X3\n"
                   "OBSERVABLE_INCLUDE(0) rec[-1]\nMPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\n"
                   "R 9\nM 9\nDETECTOR rec[-1]\n"),
    "dec_first_minus": ("R 0\nRX 1 2 3\nS 1\nS 1\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\nMPP Z2*Z3\n"
                        "OBSERVABLE_INCLUDE(0) rec[-1]\nMX 1\nDECISION(0) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
    # code review MAJOR-2 (MP-1 end to end on a COMBINED direction): the CCZ order reversed, so the
    # bare frame maps X2X3 -> X_0 X_1 and the Born decision introduces the combined literal product
    # at phase 0 against canonical phase 2 (a4_comb's combined direction is already canonical);
    # a4d_rev is the same with W a decision
    "a4_comb_rev": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 3\nCCZ 0 1 2\nMPP X2*X3\n"
                    "OBSERVABLE_INCLUDE(0) rec[-1]\nMPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\n"
                    "R 9\nM 9\nDETECTOR rec[-1]\n"),
    "a4d_rev": ("R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 3\nCCZ 0 1 2\nMPP X2*X3\n"
                "DECISION(0) rec[-1]\nMPP Y2*Y3\nDECISION(1) rec[-1]\nMX 1\nDECISION(2) rec[-1]\n"
                "R 9\nM 9\nDETECTOR rec[-1]\n"),
    "a4_sign": ("RX 0 1\nT 0\nT 1\nCX 0 2\nCX 1 3\nX_ERROR(0.5) 4\nMPP Z0*Z1\nDECISION(0) rec[-1]\nMPP X0*X1\n"
                "DECISION(1) rec[-1]\nMPP Y0*Y1\nDECISION(2) rec[-1]\nR 9\nM 9\nDETECTOR rec[-1]\n"),
}
KIND = {"detector": 0, "observable": 1, "decision": 2}


def plan_key_of(orc, S):
    """The engine's diagonal plan bytes (identity prefix) of the oracle's term set S: C0 = prod rot(B_j)
    has phase pi/4 * sum_j s_j (-1)^{T_j.x}; expanding (-1)^{xor x_T} = sum_U (-2)^{|U|} x^U gives the
    S-exponent of each wire (|U| = 1) and the CZ pairs (|U| = 2); |U| >= 3 vanishes mod 2 pi."""
    xz, ph, _ = orc.terms()
    n = orc.n
    c1 = np.zeros(n, np.int64)
    c2 = {}
    for j in np.flatnonzero(np.asarray(S)):
        x = xz[j, :n]
        if x.any():
            return None                                   # a PPR term: not a diagonal plan
        T = list(np.flatnonzero(xz[j, n:]))
        s = 1 if int(ph[j]) % 4 == 0 else -1
        if int(ph[j]) % 2:
            raise AssertionError("non-Hermitian term")
        for q in T:
            c1[q] += s * (-2)
        for a, b in itertools.combinations(T, 2):
            c2[(a, b)] = c2.get((a, b), 0) + s * 4
    for q in range(n):
        assert c1[q] % 2 == 0
    a = [int((c1[q] // 2) % 4) for q in range(n)]
    cz = sorted(p for p, v in c2.items() if v % 4 == 0 and (v // 4) % 2 == 1)
    for p, v in c2.items():
        assert v % 4 == 0, "non-Clifford residual"
    NW = (n + 63) // 64
    out = b"\0" * (16 * NW)                               # identity prefix (P-free table)
    out += bytes(ai & 1 for ai in a) + b"\xff"
    for f, s2 in cz:
        out += struct.pack("<HH", f, s2)
    # The engine keeps S^{a&1} in the plan and moves Z^{a&2} = S^2 into the Pauli prefix; the oracle's
    # C0 keeps it. That Pauli T flips every record by t_c = <T, M_c> (Lemma 2), which the comparison
    # below accounts for exactly.
    zpart = np.array([1 if ai & 2 else 0 for ai in a], np.uint8)
    return out, zpart


def compare(text, max_configs=4096):
    """L-3 on the PRODUCTION table (J-b2: record-space B, _RecBTables): every distinct fault residual's table field
    by field against the dense oracle's law."""
    orc = oracle(text)
    s = _RecBTables(text)
    cols = [(KIND[k], i) for k, i, _ in orc.columns()]
    ps0 = s.plan_structure(b"")
    order = [(o >> 24, o & ((1 << 24) - 1)) for o in ps0["rec_order"]]
    if sorted(order) != sorted(cols):
        pytest.skip("the circuit has columns outside the engine's S2b family")
    ocol = {kc: i for i, kc in enumerate(cols)}
    st2or = [ocol[o] for o in order]                      # storage column -> oracle column
    seen, n_cmp = set(), 0
    for _, faults in fault_configs(orc, max_configs):
        S, _, _ = orc.residual(faults, None)
        key = bytes(S.tobytes())
        if key in seen:
            continue
        seen.add(key)
        got = plan_key_of(orc, S)
        if got is None:
            continue
        pk, zpart = got
        ps = s.plan_structure(pk)
        if ps["fallback"] or not ps["rec_built"]:
            continue
        L = orc.law(S)
        n_cmp += 1
        ctx = (text, faults)
        assert ps["rho"] == L["rho"] and ps["kappa_rec"] == L["kappa"], (ctx, ps["rho"], L["rho"], ps["kappa_rec"], L["kappa"])
        assert [st2or[c] for c in ps["pivot_rec"]] == list(L["pivots"]), ctx
        assert [st2or[c] for c in ps["born_rec"]] == list(L["born"]), ctx
        A, B, be = np.asarray(ps["rec_A"]), np.asarray(ps["rec_B"]), np.asarray(ps["rec_beta"])
        LA, LB, Lb = np.asarray(L["A"]), np.asarray(L["B"]), np.asarray(L["beta"])
        n = orc.n
        t = np.asarray(orc.relabel(np.zeros(n, np.uint8), zpart)).astype(int)   # oracle column order
        tp = np.array([t[st2or[c]] for c in ps["pivot_rec"]], int)
        tau = []
        for j, cj in enumerate(ps["born_rec"]):
            tau.append((t[st2or[cj]] + int(np.dot(A[cj].astype(int), tp))) & 1)
        tau = np.array(tau, int)
        for c in range(len(order)):
            oc = st2or[c]
            assert np.array_equal(A[c], LA[oc]), ("A", ctx, c)
            assert np.array_equal(B[c], LB[oc]), ("B", ctx, c)
            if c in ps["pivot_rec"]:
                exp = int(be[c])
            else:
                exp = (int(be[c]) + t[oc] + int(np.dot(A[c].astype(int), tp)) + int(np.dot(B[c].astype(int), tau))) & 1
            assert exp == int(Lb[oc]), ("beta", ctx, c)
        if L["kappa"]:
            assert np.array_equal(np.asarray(ps["rec_lam"]), np.asarray(L["lam"])), ("lam", ctx)
        assert [(int(x) + 2 * int(tj)) & 3 for x, tj in zip(ps["rec_lam_phase"], tau)] == \
               [int(x) for x in L["lam_phase"]], ("lam phase", ctx)
    return n_cmp


def sign_preps(seed, wires, p=0.35):
    """Negative-sign generator preparations (code review MAJOR-1): each |+> wire becomes |-> (a
    certified generator -X) with probability p, by Z, S S or H X H. A separate stream, so a seed's
    circuit is its pre-review circuit with these lines inserted after the preparations; the fold
    bit beta' = 1 is then reachable from the family (an in-group X read of a |-> wire)."""
    r = np.random.default_rng(777_000 + seed)
    out = []
    for q in wires:
        if r.random() < p:
            out += [["Z %d", "S %d\nS %d", "H %d\nX %d\nH %d"][int(r.integers(0, 3))].replace("%d", str(q))]
    return out


def family_circuit(seed):
    """A random DIAGONAL circuit in the S2b family: |+> / |0> preps, T / T_DAG / CS / CCZ magic, X / Z
    faults before the magic (non-Pauli residuals: S and CZ dressings), terminal X / Y / Z reads, and
    random DETECTOR / OBSERVABLE / DECISION record parities (compiles that refuse a record are skipped
    by the caller)."""
    r = np.random.default_rng(seed)
    n = int(r.integers(3, 6))
    lines = ["RX " + " ".join(map(str, range(n)))]
    lines += sign_preps(seed, range(n))
    if r.random() < 0.5:
        lines.append("R %d" % n)
        n_aux = 1
    else:
        n_aux = 0
    for q in range(n):
        if r.random() < 0.3:
            lines.append("%s_ERROR(0.5) %d" % ("X" if r.random() < 0.7 else "Z", q))
    for _ in range(int(r.integers(1, 4))):
        g = r.choice(["T", "T_DAG", "CS", "CCZ"])
        qs = list(map(int, r.choice(n, {"T": 1, "T_DAG": 1, "CS": 2, "CCZ": 3}[g] if n >= 3 else 1, replace=False)))
        if len(qs) < {"T": 1, "T_DAG": 1, "CS": 2, "CCZ": 3}[g]:
            g, qs = "T", qs[:1]
        lines.append(g + " " + " ".join(map(str, qs)))
    reads = []
    for q in range(n + n_aux):
        b = r.choice(["M", "MX", "MY"]) if q < n else "M"
        lines.append("%s %d" % (b, q))
        reads.append(q)
    nr = len(reads)
    def recs(k):
        idx = sorted(map(int, r.choice(nr, k, replace=False)))
        return " ".join("rec[%d]" % (i - nr) for i in idx)
    for _ in range(int(r.integers(0, 3))):
        lines.append("DETECTOR " + recs(int(r.integers(1, 3))))
    if r.random() < 0.6:
        lines.append("OBSERVABLE_INCLUDE(0) " + recs(int(r.integers(1, 3))))
    for k in range(int(r.integers(0, 4))):
        lines.append("DECISION(%d) " % k + recs(int(r.integers(1, 3))))
    lines.append("R 9\nM 9\nDETECTOR rec[-1]")
    return "\n".join(lines) + "\n"


def dressed_circuit(seed):
    """A random diagonal circuit whose CHANNELS carry Born directions (κ_σ > 0, the σ-law path): blocks
    of (|0> control c, |+> spectator s, magic m = T|+>) with CCZ c s m and a 50% X fault on c — a fault
    leaves CZ(s, m), dressing the in-group X_s into X_s Z_m. Records: DETECTORs on spectator X reads
    (in-group, Born-dressed), and random OBSERVABLE / DECISION parities over every read."""
    r = np.random.default_rng(10_000 + seed)
    nb = int(r.integers(1, 4))
    lines, reads = [], []
    for b in range(nb):
        c, sp, m = 3 * b, 3 * b + 1, 3 * b + 2
        lines += ["R %d" % c, "RX %d %d" % (sp, m), ("T %d" if r.random() < 0.5 else "T_DAG %d") % m,
                  "X_ERROR(0.5) %d" % c]
    lines += sign_preps(seed, [3 * b + 1 for b in range(nb)], p=0.5)
    for b in range(nb):
        lines.append("CCZ %d %d %d" % (3 * b, 3 * b + 1, 3 * b + 2))
    if nb > 1 and r.random() < 0.5:
        lines.append("CZ %d %d" % (2, 5))
    for b in range(nb):
        lines.append("MX %d" % (3 * b + 1)); reads.append(("s", b))
        lines.append(r.choice(["M", "MX", "MY"]) + " %d" % (3 * b + 2)); reads.append(("m", b))
    nr = len(reads)
    def rec_of(i):
        return "rec[%d]" % (i - nr)
    for i, (kind, b) in enumerate(reads):
        if kind == "s" and r.random() < 0.8:
            lines.append("DETECTOR " + rec_of(i))
    if r.random() < 0.7:
        k = int(r.integers(1, 3))
        lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(rec_of(int(i)) for i in sorted(r.choice(nr, k, replace=False))))
    for d in range(int(r.integers(0, 3))):
        k = int(r.integers(1, 3))
        lines.append("DECISION(%d) " % d + " ".join(rec_of(int(i)) for i in sorted(r.choice(nr, k, replace=False))))
    lines.append("R 99\nM 99\nDETECTOR rec[-1]")
    return "\n".join(lines) + "\n"


# ── J-b2 J1b: the PRODUCTION leg (plan r2.1 J1b). The same independent oracle (orc.law: the record law of each
# distinct fault residual on dense states, no engine code) against the table PRODUCTION builds: record-space B from
# scratch (build_record_table_recB, the run path's builder and, from J7a, the readers'), on the family sampler in the
# production configuration (classes "all"). Production refuses no record, so the dressed / random generators run
# UNRESTRICTED (every gauge / ANTI / extra-observable column is in the table and compared) — a superset of the legacy
# leg's restricted family.
class _RecBTables:
    def __init__(self, text):
        import warnings
        from xtim import _xtim_testapi as TA
        self._TA = TA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.f = TA.compile_family_sampler(text, selfcheck=0)

    def plan_structure(self, pk):
        if pk == b"":
            return self.f.plan_structure(pk, False)          # rec_order (the record family of the compile)
        r = self._TA.recb_plan_structure(self.f, pk)
        out = dict(r.get("b", {}))
        out["fallback"] = r["fallback"]
        out.setdefault("rec_built", not r["fallback"])
        return out


def compare_recb(text, max_configs=4096):
    return compare(text, max_configs)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_record_table_equals_oracle_on_fixtures_recb(name):
    assert compare_recb(FIXTURES[name]) > 0


@pytest.mark.parametrize("seed", range(200))
def test_record_table_equals_oracle_on_dressed_channel_circuits_recb(seed):
    assert compare_recb(dressed_circuit(seed)) > 0


@pytest.mark.parametrize("seed", range(300))
def test_record_table_equals_oracle_on_random_family_circuits_recb(seed):
    assert compare_recb(family_circuit(seed)) > 0
