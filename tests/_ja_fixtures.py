"""J-a (stage J-a, commit N1) fixture sets and helpers for the FAM (record-family) sampler gates.

The FAM sampler is built ONLY through the test API `xtim._xtim_testapi.compile_family_sampler`
(switched OFF in production). N1 serves the new record classes on DIAGONAL circuits: gauge
(LOGICAL-class) detectors, ANTI detectors and extra observables become record-table extras, emitted
reference-relative (y XOR rf XOR inv XOR ref, gauge-plan section 1.4).

Fixture sets (plan section 3.0, the diagonal part):
  GATE1        the gauge-plan section 4.1 gate-1 repro (P = 0.146);
  DUP_GAUGE    the same logical read twice, records differing by a stabilizer (raw XOR = 1 = beta);
  DESIGN       FX-design circuits that are diagonal (adv-C p4_gauge / p3_anti twins, adv-A p4.stim,
               the correlated-ANTI Bell pair, the teleport-parity ANTI detector);
  FX_F1        the J-b1 gate circuit of fixplan-s2a F1 (+ a noisy variant);
  AH11         the A-H1-1 three-ANTI fixture (D3 = D1 xor D2; the signed variant);
  CONVENTION   gate-6 convention circuits (M(p), !, MPAD, CY feedback, a feedback-DRIVING noisy record);
  twins()      FX-twin: the S2b FIXTURES with each Born-class observable / decision re-declared as a
               gauge DETECTOR, and an ANTI twin (that record times a fresh |0> wire read in X).
Statistics: 5 sigma with the S2b multiplicity rule (one rerun at 2N for a single 5 < |z| < 6; a
recurrence, a second circuit needing a rerun, or |z| >= 6 fails); impossible outcomes are exact
counters (= 0).
"""
import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _record_law_x import emitted_law_x, fault_configs, oracle  # noqa: E402
from _record_oracle import emitted_columns, emitted_law, max_z, parities  # noqa: E402
from test_s2b_record_table import FIXTURES, KIND, plan_key_of  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402


GATE1 = "RX 0 1\nR 2\nT 0\nMX 0\nCZ rec[-1] 1\nMX 1\nM 2\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
# GATE1 with its two detectors swapped: deterministic detector 0 (channel 0, det_identity on), the gauge
# detector LAST (index 1 >= ndetc, same byte): the FamilySink word path's ordering witness (M1).
GATE1_LAST = "RX 0 1\nR 2\nT 0\nMX 0\nCZ rec[-1] 1\nMX 1\nM 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"

# X0 (read inverted) and X0*X1 on T|+> (x) |->: the records differ by a stabilizer and the invert, so
# their RAW XOR is a plan constant (beta, read from the table) and the REPORTED XOR is 0 on every shot;
# Z noise before the T dresses both reads equally.
DUP_GAUGE = ("RX 0 1\nZ_ERROR(0.2) 0\nT 0\nZ 1\nX_ERROR(0.1) 2\nM 2\nMX !0\nMPP X0*X1\n"
             "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")

DESIGN = {
    # adv-C p4_gauge: D1 = X0 (T|+>), D2 = X0 read again: P(D1 = 0) = 0.8536, P(D1 == D2) = 1
    "p4_gauge": "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX 0\nMX 0\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    # adv-C p3_anti, its ANTI-detector twin: D1 = Z1, D2 = Z1 X0, D3 = Z2 on a Bell pair (1, 2):
    # P(D1 == D2) = 0.8536, P(D1 == D3) = 1
    "p3_anti": ("RX 0\nT 0\nR 1\nR 2\nH 1\nCX 1 2\nX_ERROR(0.05) 3\nM 3\nDETECTOR rec[-1]\n"
                "M 1\nMX 0\nM 2\nDETECTOR rec[-3]\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\n"),
    # adv-A p4.stim: D0 = Z1 and D1 = Z1 X0 are ANTI; P(D0 xor D1 = 1) = 0.146
    "p4_stim": "RX 0 1\nR 2\nT 0\nDEPOLARIZE1(0.001) 2\nM 1\nMPP Z1*X0\nM 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    # the correlated-ANTI Bell fixture: Z0 Z1 = -1 on every shot (ANTI detectors, fair marginals)
    "bell_anti": "R 0 1\nH 0\nCX 0 1\nX 1\nX_ERROR(0.1) 2\nM 2\nM 0 1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    # the teleport-parity ANTI detector (test_refusal_design_inputs TELEPORT; Cor. 1.4)
    "teleport": ("RX 0\nR 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nX_ERROR(0.01) 2\nM 0 1\nR 3\nM 3\n"
                 "DETECTOR rec[-1]\nDETECTOR rec[-3]\n"),
    "teleport_t": ("RX 0\nT 0\nR 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nX_ERROR(0.01) 2\nM 0 1\nR 3\nM 3\n"
                   "DETECTOR rec[-1]\nDETECTOR rec[-3]\nOUTPUT_QUBITS out 2\n"),
    # a second LOGICAL observable (today refused) beside the Born one, and an ANTI observable
    "obs3": ("RX 0 1 3\nT 0 1\nCX 0 1\nX_ERROR(0.1) 2\nM 2\nMX 0\nMX 1\nM 3\nDETECTOR rec[-4]\n"
             "OBSERVABLE_INCLUDE(0) rec[-3]\nOBSERVABLE_INCLUDE(1) rec[-2]\nOBSERVABLE_INCLUDE(2) rec[-1]\n"),
}

# fixplan-s2a F1 / adv-A-s2a section 2: the gauge records D1 (X0 X1), D2 (X0) after the chi-drop
FX_F1 = {
    "F1": "RX 0 1\nR 2\nT 0 1\nCS 0 1\nMPP X0*X1\nMX 0\nM 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "F1_noisy": ("RX 0 1\nR 2\nZ_ERROR(0.2) 0\nX_ERROR(0.1) 1\nT 0 1\nCS 0 1\nX_ERROR(0.1) 2\nMPP X0*X1\nMX 0\nM 2\n"
                 "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"),
}

# A-H1-1: three ANTI detectors on |+>|+>, D3 = D1 xor D2 (signed: an X between the reads makes the RAW
# D3 = D1 xor D2 xor 1, absorbed by the reference); only the 3-way parity distinguishes a beta error
AH11 = {
    "ah11": "RX 0 1\nM 0\nM 1\nMPP Z0*Z1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "ah11_signed": "RX 0 1\nM 0\nM 1\nX 0\nMPP Z0*Z1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
}

# gauge-plan section 4.1 gate 6 (adv-C-g1 g1_conv / p10_conv, adv-A-g1 g1.py), the gauge-column twins
CONVENTION = {
    "readout_flip": "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX(0.1) 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "invert": "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX !0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "mpad": "RX 0\nT 0\nMX 0\nMPAD 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-3] rec[-2]\n",
    "cy_feedback": ("X_ERROR(0.2) 0\nM 0\nRX 1\nT 1\nH 0\nCY rec[-1] 0\nMX 0\nMY 1\n"
                    "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\n"),
    # adv-A-g1 MINOR-2 (dev/next twirl 0.2181 vs exact 0.2170): the noisy MX(0.1) 0 DRIVES CZ rec 1
    "fb_noisy_gauge": ("RX 0 1\nR 2\nT 0\nMX(0.1) 0\nCZ rec[-1] 1\nMX 1\nM 2\nDETECTOR rec[-1]\n"
                       "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-3] rec[-2]\n"),
    "fb_noiseless_gauge": ("RX 0 1\nR 2\nT 0\nMX 0\nCZ rec[-1] 1\nMX 1\nM 2\nDETECTOR rec[-1]\n"
                           "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-3] rec[-2]\n"),
    # p10_conv's `!` observable as a gauge detector: inv = ref = 1 (MUT-REF flips it)
    "p10_gauge": "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX !0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
}

# a carried-input port circuit: its gauge column has noiseless reference 1 (MX after Z)
PORT = ("INPUT_QUBITS in 5\nRX 0\nT 0\nMX 0\nZ 0\nMX 0\nR 9\nX 9\nM 9\n"
        "DETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-3] rec[-2]\n")

# an IF-partitioned circuit (resolved per driving pattern, as PartitionedTwirlSampler does)
IF_TEXT = ("INPUT_BITS drv 1\nRX 0 1\nR 2\nT 0\nIF drv[0] {\n  Z 0\n}\nMX 0\nCZ rec[-1] 1\nMX 1\nM 2\n"
           "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")


# ── samplers and emitted columns ────────────────────────────────────────────────────────────────
def family(text, **kw):
    kw.setdefault("selfcheck", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, **kw)


def unpack(a, k):
    a = np.asarray(a, np.uint8)
    if k == 0:
        return np.zeros((a.shape[0], 0), np.uint8)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :k]


def columns(s, n, seed):
    """The emitted columns of a FAM sampler: detectors, observables, decisions (emitted_columns order)."""
    d, o, dc = s.sample(n, seed=seed)
    return np.concatenate([unpack(d, s.num_detectors), unpack(o, s.num_observables),
                           unpack(dc, s.num_decisions)], 1)


def exact_columns(text, n, seed):
    import xtim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d, o = xtim.Circuit(text).compile_detector_sampler(seed=seed, engine="exact").sample(
            n, separate_observables=True)
    return np.concatenate([np.asarray(d, np.uint8), np.asarray(o, np.uint8)], 1)


# ── statistics (the S2b multiplicity rule) ─────────────────────────────────────────────────────
_RERUNS = []


def _rerun_rule(name, z):
    if z >= 6:
        raise AssertionError("%s: |z| = %.2f >= 6" % (name, z))
    _RERUNS.append(name)
    if len(set(_RERUNS)) > 1:
        raise AssertionError("5 < |z| < 6 on a second circuit: %s" % sorted(set(_RERUNS)))


def check_law(name, draw, law, n):
    """`draw(n, seed)` -> emitted columns; every cell of the joint law at |z| < 5 (one rerun at 2N),
    zero shots in an impossible cell (exact counter). Returns the samples."""
    M = draw(n, 11)
    z, imp = max_z(M, law)
    assert imp == 0, (name, "impossible outcomes", imp)
    if z >= 5:
        _rerun_rule(name, z)
        M = draw(2 * n, 23)
        z, imp = max_z(M, law)
        assert imp == 0 and z < 5, (name, "recurrence", z, imp)
    return M


def two_sample_z(a, b):
    """Largest |z| over every column marginal and every pairwise joint cell of two 0/1 samples."""
    worst = 0.0
    m = a.shape[1]
    cells = [(c,) for c in range(m)] + list(itertools.combinations(range(m), 2))
    for cs in cells:
        for v in itertools.product((0, 1), repeat=len(cs)):
            pa = np.all(a[:, cs] == v, axis=1).mean()
            pb = np.all(b[:, cs] == v, axis=1).mean()
            p = (pa * len(a) + pb * len(b)) / (len(a) + len(b))
            if 0 < p < 1:
                worst = max(worst, abs(pa - pb) / np.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b))))
            elif pa != pb:
                worst = float("inf")
    return worst


def check_identity(name, draw_a, draw_b, n):
    z = two_sample_z(draw_a(n, 11), draw_b(n, 12))
    if z >= 5:
        _rerun_rule(name, z)
        z = two_sample_z(draw_a(2 * n, 21), draw_b(2 * n, 22))
        assert z < 5, (name, "recurrence", z)


def relation_parities(law, m):
    """Every linear relation r (bitmask over the m columns) of the exact law: its P(r.y = 1)."""
    out = {}
    for r in range(1, 1 << m):
        p1 = sum(p for k, p in law.items() if sum(k[c] for c in range(m) if r >> c & 1) & 1)
        out[r] = p1
    return out


def relation_counters(M, law):
    """G-10: the deterministic relations of the law (P(r.y) in {0, 1}, 3-way and higher included) as
    exact violation counters over every shot, and the Born-valued relations' worst |z|."""
    m = M.shape[1]
    viol, worst, ndet = 0, 0.0, 0
    for r, p1 in relation_parities(law, m).items():
        par = (M[:, [c for c in range(m) if r >> c & 1]].sum(1) & 1)
        if p1 < 1e-12 or p1 > 1 - 1e-12:
            ndet += 1
            viol += int((par != (1 if p1 > 0.5 else 0)).sum())
        else:
            worst = max(worst, abs(par.mean() - p1) / np.sqrt(p1 * (1 - p1) / len(M)))
    return ndet, viol, worst


# ── FX-twin ─────────────────────────────────────────────────────────────────────────────────────
def _born_columns(text):
    """(kind, index) of the Born-class (LOGICAL at R = I) observables / decisions of `text`, per (X)."""
    orc = oracle(text)
    S, _, _ = orc.residual([], None)
    L = orc.law(S)
    out = []
    for oc, (k, i, reads) in enumerate(orc.columns()):
        if k != "detector" and len(reads) and int(L["cls"][oc]) == 2:
            out.append((k, i))
    return out


def _drop_and_append(text, kind, idx, extra_lines):
    """`text` without the record line(s) of observable / decision `idx`, decisions renumbered densely,
    then `extra_lines` appended."""
    out = []
    tag = "OBSERVABLE_INCLUDE(" if kind == "observable" else "DECISION("
    for ln in text.splitlines():
        if ln.startswith(tag) and int(ln[len(tag):ln.index(")")]) == idx:
            continue
        out.append(ln)
    dec = sorted({int(ln[9:ln.index(")")]) for ln in out if ln.startswith("DECISION(")})
    ren = {d: i for i, d in enumerate(dec)}
    out = ["DECISION(%d)%s" % (ren[int(ln[9:ln.index(")")])], ln[ln.index(")") + 1:])
           if ln.startswith("DECISION(") else ln for ln in out]
    return "\n".join(out + extra_lines) + "\n"


def twins():
    """{name: text}: per S2b FIXTURE, per Born-class observable / decision, the gauge twin (re-declared
    as a DETECTOR) and the ANTI twin (that record times a fresh |0> wire read in X)."""
    out = {}
    for name in sorted(FIXTURES):
        text = FIXTURES[name]
        dets, obs, decs = parities(text)
        import xtim
        c = xtim.Circuit(text)
        m_total = c.num_measurements
        for kind, idx in _born_columns(text):
            recs = obs[idx] if kind == "observable" else decs[idx]
            rel = " ".join("rec[-%d]" % (m_total - r) for r in recs)
            out["%s.g%s%d" % (name, kind[:3], idx)] = _drop_and_append(text, kind, idx, ["DETECTOR " + rel])
            q = c.num_qubits
            rel2 = " ".join("rec[-%d]" % (m_total + 1 - r) for r in recs)
            out["%s.a%s%d" % (name, kind[:3], idx)] = _drop_and_append(
                text, kind, idx, ["R %d" % q, "MX %d" % q, "DETECTOR rec[-1] " + rel2])
    return out


# ── per-plan field identity against (X) (the S2b compare(), on a FAM sampler) ──────────────────
def compare_tables(text, max_configs=4096, force_general=False):
    """Every diagonal plan reached by a fault configuration of `text`: the FAM sampler's record table
    (pivots, Born records, A, B, beta, Lambda with the EFFECTIVE phase) equals (X)'s plan law, field by
    field (test_s2b_record_table.compare's convention). force_general=True (J-a N3): the table of each
    plan built by (G) instead of (D). Returns the number of plans compared."""
    orc = oracle(text)
    s = family(text)
    cols = [(KIND[k], i) for k, i, _ in orc.columns()]
    ps0 = s.plan_structure(b"")
    order = [(o >> 24, o & ((1 << 24) - 1)) for o in ps0["rec_order"]]
    assert sorted(order) == sorted(cols), ("the FAM sampler does not cover every column", order, cols)
    ocol = {kc: i for i, kc in enumerate(cols)}
    st2or = [ocol[o] for o in order]
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
        ps = s.plan_structure(pk, force_general=force_general)
        if ps["fallback"] or not ps["rec_built"]:
            continue
        if force_general:                     # N3-D3: (G) builds are the null form (no fold, |b| = 1 introducers)
            assert ps["folds"] == 0 and [int(b) for b in ps["intro_b"]] == [1] * ps["kappa_rec"], (text, faults)
        L = orc.law(S)
        n_cmp += 1
        ctx = (text, faults)
        assert ps["rho"] == L["rho"] and ps["kappa_rec"] == L["kappa"], ctx
        assert [st2or[c] for c in ps["pivot_rec"]] == list(L["pivots"]), ctx
        assert [st2or[c] for c in ps["born_rec"]] == list(L["born"]), ctx
        A, B, be = np.asarray(ps["rec_A"]), np.asarray(ps["rec_B"]), np.asarray(ps["rec_beta"])
        LA, LB, Lb = np.asarray(L["A"]), np.asarray(L["B"]), np.asarray(L["beta"])
        t = np.asarray(orc.relabel(np.zeros(orc.n, np.uint8), zpart)).astype(int)
        tp = np.array([t[st2or[c]] for c in ps["pivot_rec"]], int)
        tau = np.array([(t[st2or[cj]] + int(np.dot(A[cj].astype(int), tp))) & 1 for cj in ps["born_rec"]], int)
        for c in range(len(order)):
            oc = st2or[c]
            assert np.array_equal(A[c], LA[oc]), ("A", ctx, c)
            assert np.array_equal(B[c], LB[oc]), ("B", ctx, c)
            exp = int(be[c]) if c in ps["pivot_rec"] else \
                (int(be[c]) + t[oc] + int(np.dot(A[c].astype(int), tp)) + int(np.dot(B[c].astype(int), tau))) & 1
            assert exp == int(Lb[oc]), ("beta", ctx, c)
        if L["kappa"]:
            assert np.array_equal(np.asarray(ps["rec_lam"]), np.asarray(L["lam"])), ("lam", ctx)
        assert [(int(x) + 2 * int(tj)) & 3 for x, tj in zip(ps["rec_lam_phase"], tau)] == \
               [int(x) for x in L["lam_phase"]], ("lam phase", ctx)
    return n_cmp


__all__ = ["GATE1", "DUP_GAUGE", "DESIGN", "FX_F1", "AH11", "CONVENTION", "PORT", "IF_TEXT",
           "family", "unpack", "columns", "exact_columns", "check_law", "check_identity", "two_sample_z",
           "relation_counters", "twins", "compare_tables", "emitted_law", "emitted_law_x", "oracle",
           "emitted_columns"]
