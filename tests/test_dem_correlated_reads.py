"""DEM export: faults whose Clifford residual randomizes terminal X/Y reads (3.1.10).

3.1.9 treated every randomized read as an INDEPENDENT fair coin. The read flips of one fault are
correlated in general — two S-type residuals can cancel on the state, a CZ fixes a parity — so
that law was wrong: refused as "negative probability" or, when the factoring happened to
succeed, SILENTLY WRONG (e.g. a Y fault before T that never fires a detector was exported as
error(p/2) on it; cultivation_d3_faithful carried 212 such spurious mechanisms).

3.1.10 computes the exact joint law of the targets fed by the randomized reads from the bare
state (the stabilizer-twirled Pauli channel of the residual: each stabilizer coset with its
deterministic signature) and converts disjoint -> independent mechanisms with stim's rules:
exactly where possible, otherwise refuse unless `approximate_disjoint_errors` (bool, or a float
threshold on the channel's probability arguments), which approximates each merged signature
as an independent mechanism with its disjoint probability.

Oracles: the exact engine per fault alternative (inserted as a gate; exact per-shot records;
the law is sampled, 5-sigma tolerance) — the primary arbiter; stim on Clifford circuits,
compared SEMANTICALLY (flattened target-set maps, duplicates XOR-composed, |dp| <= 1e-12),
including when each tool refuses (by condition, not message text).
"""
import json
import math
import random

import numpy as np
import pytest

import xtim
from xtim.errors import XtimDemError
import test_dem_expectation_frames as F

stim = pytest.importorskip("stim")


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def sem_map(dem, D=None, drop_L=None):
    """{frozenset(targets): p}: detectors + OBSERVABLE_INCLUDE observables only, coordinates /
    separators ignored, duplicate target sets XOR-composed, p = 0 dropped. `drop_L`: L ids >= it
    (xtim's PAULI_EXPECTATION columns) are removed."""
    out = {}
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        k = []
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                k.append(("D", t.val))
            elif t.is_logical_observable_id() and (drop_L is None or t.val < drop_L):
                k.append(("L", t.val))
        k = frozenset(k)
        if not k:
            continue
        p = ins.args_copy()[0]
        q = out.get(k, 0.0)
        out[k] = q + p - 2 * q * p
    return {k: v for k, v in out.items() if v > 0}


def maps_equal(a, b, tol=1e-12):
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) <= tol * max(1.0, abs(b[k])) for k in b)


def try_dem(text, tool, approx):
    try:
        if tool == "stim":
            return sem_map(stim.Circuit(text).detector_error_model(approximate_disjoint_errors=approx))
        return sem_map(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=approx,
                                                               include_expectations=False))
    except (ValueError, XtimDemError):
        return None


def _base(text):
    import re
    lines = [l.split("#")[0].strip() for l in text.strip().splitlines()]
    lines = [l for l in lines if l]
    noise_idx = [i for i, l in enumerate(lines) if F._split(l)[0] in F.NOISE]
    base_lines = [re.sub(r"^(M|MX|MY|MR|MRX|MRY|MPP)\([^)]*\)", r"\1", l)
                  for i, l in enumerate(lines) if i not in noise_idx]
    return lines, noise_idx, base_lines


def exact_law(base_lines, pos, d, keep, recs, D, O, ref, shots):
    fl = list(base_lines)
    fl[pos:pos] = [f"{P} {q}" for q, P in sorted(d.items())]
    meas, _ = F._run_exact("\n".join(fl) + "\n", shots, 7)
    par = F._parities(recs, meas, D, O)
    law = {}
    for row in par:
        s = frozenset(c for c in keep if row[c] != ref[c])
        law[s] = law.get(s, 0.0) + 1.0 / shots
    return law


def dist_of(dem_text, D):
    return F.dem_distribution(stim.DetectorErrorModel(dem_text), D)


def law_check(text, shots=1500, every=1, approx_ok=True):
    """Every Pauli alternative (one-hot, p = 0.1) and every channel of `text` against the exact
    engine. Exact conversions must reproduce the law; refusals must be laws with no exact
    independent factoring, and the approximate_disjoint_errors export must carry each merged
    signature with exactly its first-order (disjoint) mass. Returns counters."""
    lines, noise_idx, base_lines = _base(text)
    base = "\n".join(base_lines) + "\n"
    recs, _ = F.parse_records(base)
    D = sum(1 for k, *_ in recs if k == "DETECTOR")
    O = 1 + max([int(a) for k, a, _ in recs if k == "OBSERVABLE_INCLUDE"] or [-1])
    meas0, _ = F._run_exact(base, 512, 3)
    par0 = F._parities(recs, meas0, D, O)
    keep = [c for c in range(D + O) if (par0[:, c] == par0[0, c]).all()]
    assert all(c in keep for c in range(D)), "non-deterministic detector"
    ref = par0[0]
    tol1 = 5 * math.sqrt(0.25 / shots) + 1e-12
    cnt = dict(alts=0, random=0, exact=0, approximated=0, flagged=0, channels=0)
    for n_line, li in enumerate(noise_idx):
        if n_line % every:
            continue
        name, arg, tg = F._split(lines[li])
        pos = li - sum(1 for j in noise_idx if j < li)
        for qs, alts in F.channel_alternatives(name, arg, tg):
            cnt["channels"] += 1
            mix, ptot, skipped = {frozenset(): 0.0}, 0.0, False
            for p, d in alts:
                if p <= 0:
                    continue
                law = exact_law(base_lines, pos, d, keep, recs, D, O, ref, shots)
                one = F._one_hot_channel(d, qs, 0.1)
                oh = "\n".join(base_lines[:pos] + [one] + base_lines[pos:]) + "\n"
                r = xtim._xtim.export_dem_text(oh, False, False, False, True, [], [], 0.0)
                cnt["alts"] += 1
                if r["ok"] and r["postselect_faults"]:
                    # logical-Clifford fault (unchanged magnitude path): post-selected on the
                    # detectors it fires; its accepted-branch harmlessness is checked by the
                    # expectation-column oracle (test_dem_expectation_frames)
                    cnt["flagged"] += 1
                    assert any(any(c < D for c in s) for s in law), "flagged fault never detected"
                    skipped = True
                    continue
                ptot += p
                for s, v in law.items():
                    mix[s] = mix.get(s, 0.0) + p * v
                if len(law) > 1:
                    cnt["random"] += 1
                want = {frozenset(): 0.9}
                for s, v in law.items():
                    want[s] = want.get(s, 0.0) + 0.1 * v
                if r["ok"]:
                    got = dist_of(r["dem"], D)
                    for k in set(got) | set(want):
                        assert abs(got.get(k, 0) - want.get(k, 0)) <= 0.1 * tol1, (
                            lines[li], d, sorted(k), got.get(k, 0), want.get(k, 0))
                    cnt["exact"] += 1
                    continue
                # refused: the law must have no exact independent factoring — a uniform law on an
                # affine set s0 + V with s0 not in V (every such mixture needs a negative mechanism)
                assert "negative probability" in r["error"] or "over-mixing" in r["error"], r["error"]
                sup = [s for s, v in law.items() if v > 0.03]
                s0 = sup[0]
                V = {s ^ s0 for s in sup}
                assert len(sup) & (len(sup) - 1) == 0 and all((a ^ b) in V for a in V for b in V), law
                assert frozenset() not in sup or not V, law
                if approx_ok:
                    ra = xtim._xtim.export_dem_text(oh, False, False, False, True, [], [], 1.0)
                    assert ra["ok"], ra["error"]
                    mech = {k: v for k, v in F.canon(stim.DetectorErrorModel(ra["dem"])).items()}
                    mech = {frozenset(c if t == "D" else D + c for t, c in k): v for k, v in mech.items()}
                    for s, v in law.items():
                        if s:
                            assert abs(mech.get(s, 0) - 0.1 * v) <= 0.1 * tol1, (lines[li], d, s, mech.get(s), v)
                    assert set(mech) <= {s for s in law if s}
                cnt["approximated"] += 1
            if skipped:
                continue
            mix[frozenset()] += 1 - ptot
            iso = "\n".join(base_lines[:pos] + [f"{name}({arg}) " + " ".join(map(str, qs))] + base_lines[pos:]) + "\n"
            r = xtim._xtim.export_dem_text(iso, False, False, False, True, [], [], 0.0)
            if r["ok"]:
                got = dist_of(r["dem"], D)
                for k in set(got) | set(mix):
                    assert abs(got.get(k, 0) - mix.get(k, 0)) <= ptot * tol1, (lines[li], qs, sorted(k))
    return cnt


# ---------------------------------------------------------------------------------------
# 1. minimal repros of the 3.1.9 silent errors (random search, exact engine): all exported
#    by 3.1.9 with a spurious error(p/2) on a detector the fault never fires
# ---------------------------------------------------------------------------------------
SILENT_319 = {
    "y_before_t_bell": "R 0 1 2 3 4 5\nH 1\nH 2\nH 3\nH 4\nH 5\nCX 1 0\nCZ 3 2\nCX 5 4\nY_ERROR(0.02) 5\n"
                       "Y_ERROR(0.02) 0\nY_ERROR(0.02) 3\nT_DAG 5\nT 0\nT 3\nCX 0 4\nCX 4 0\nCZ 5 3\nMX 0\n"
                       "MX 1\nMY 2\nMX 3\nMX 4\nMX 5\nDETECTOR rec[-6] rec[-2]\n",
    "y_before_t_cz": "R 0 1 2 3 4 5\nH 0\nH 2\nH 3\nH 4\nCZ 3 5\nCZ 5 4\nCZ 1 3\nCZ 4 5\nCZ 2 0\n"
                     "Y_ERROR(0.02) 4\nZ_ERROR(0.02) 5\nT 4\nT_DAG 5\nCX 3 4\nMX 0\nMY 1\nMX 2\nMX 3\nMX 4\n"
                     "MY 5\nDETECTOR rec[-3] rec[-2]\n",
    "pc1_before_tdag": "R 0 1 2 3 4\nH 1\nH 2\nH 3\nH 4\nCZ 4 1\nCZ 3 4\nCX 4 3\nCX 2 0\n"
                       "PAULI_CHANNEL_1(0.01,0.02,0.015) 1\nDEPOLARIZE1(0.03) 4\nX_ERROR(0.02) 3\nT_DAG 1\n"
                       "T_DAG 4\nT_DAG 3\nCZ 1 4\nCX 0 1\nMX 0\nMX 1\nMX 2\nM 3\nMX 4\n"
                       "DETECTOR rec[-5] rec[-4] rec[-3]\n",
    "y_before_t_cx": "R 0 1 2 3 4\nH 0\nH 1\nH 2\nH 3\nCZ 4 1\nCZ 4 2\nCZ 0 2\nX_ERROR(0.02) 1\n"
                     "Y_ERROR(0.02) 2\nT 1\nT_DAG 2\nCX 3 2\nCZ 2 4\nCX 1 3\nMX 0\nMY 1\nMX 2\nMX 3\nMX 4\n"
                     "DETECTOR rec[-3] rec[-2]\n",
    # two S-type residuals that cancel on a Bell pair (the code_switching / cultivation pattern)
    "cancelling_pair": "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE2(0.03) 0 1\nT 0\nT_DAG 1\nMX 0 1\n"
                       "DETECTOR rec[-1] rec[-2]\n",
    # a correlated pair: 3.1.9 split the X fault over three detector pairs; exactly it flips
    # D0 and D2 together with probability 1/2 (random search, seed 22)
    "correlated_pair": "R 0 1 2\nH 0\nCX 0 1\nCX 0 2\nR 3 4\nH 3\nCX 3 4\nX_ERROR(0.02) 3\nT 0\nT_DAG 2\n"
                       "T 4\nT_DAG 3\nCX 1 2\nCX 0 3\nMX 0\nMX 1\nMX 2\nMX 3\nMX 4\n"
                       "DETECTOR rec[-2] rec[-1]\nDETECTOR rec[-5] rec[-4] rec[-2]\n"
                       "DETECTOR rec[-5] rec[-4] rec[-1]\n",
    # a correlated pair whose detectors have noiseless values -1 (Y0Y1 on a Bell pair) and +1:
    # the law is relative to the noiseless value of each randomized target (its sign s_t)
    "negative_parity_pair": "R 0 1 2 3\nH 0\nCX 0 1\nH 2\nCX 2 3\nX_ERROR(0.05) 0\nT 0\nT_DAG 1\n"
                            "MY 0 1 2 3\nDETECTOR rec[-4] rec[-3]\nDETECTOR rec[-4] rec[-3] rec[-2] rec[-1]\n",
}


@pytest.mark.parametrize("name", sorted(SILENT_319))
def test_correlated_reads_exact_law(name):
    cnt = law_check(SILENT_319[name])
    assert cnt["alts"] >= 1


def test_y_before_t_never_fires_and_is_not_exported():
    # 3.1.9: error(0.01) D0; exact: the Y residual acts trivially on every detector
    c = xtim.Circuit(SILENT_319["y_before_t_cx"].replace("X_ERROR(0.02) 1\n", ""))
    m = sem_map(c.detector_error_model(include_expectations=False))
    assert m == {}


# ---------------------------------------------------------------------------------------
# 2. random circuits engineered to correlate the reads (S-type residuals on entangled X/Y reads,
#    CZ/CX between read wires after the magic gates)
# ---------------------------------------------------------------------------------------
def _random_case(rng):
    """Bell / GHZ blocks with T x T^dag (their X-parity survives the magic), faults before the
    magic, Clifford couplings of the blocks after it, X/Y/Z reads, deterministic parities as
    detectors."""
    import itertools
    k = rng.randint(1, 3)                       # Bell / GHZ blocks whose X-parity survives T x T^dag
    L = []; n = 0; blocks = []
    for _ in range(k):
        m = rng.choice([2, 2, 3])
        qs = list(range(n, n + m)); n += m; blocks.append(qs)
        L.append("R " + " ".join(map(str, qs))); L.append(f"H {qs[0]}")
        for q in qs[1:]: L.append(f"CX {qs[0]} {q}")
    ops = ["X_ERROR(0.02) {a}", "Y_ERROR(0.02) {a}", "DEPOLARIZE1(0.03) {a}", "DEPOLARIZE2(0.03) {a} {b}",
           "PAULI_CHANNEL_1(0.01,0.02,0.015) {a}", "Z_ERROR(0.02) {a}"]
    for _ in range(rng.randint(1, 4)):
        a, b = rng.sample(range(n), 2) if n > 1 else (0, 0)
        L.append(rng.choice(ops).format(a=a, b=b))
    for qs in blocks:                            # magic that preserves the block's X-parity
        a, b = rng.sample(qs, 2)
        L.append(f"T {a}"); L.append(f"T_DAG {b}")
    for _ in range(rng.randint(0, 2)):           # couple blocks (Clifford) after the magic
        a, b = rng.sample(range(n), 2)
        L.append(f"{rng.choice(['CZ', 'CX'])} {a} {b}")
    for q in range(n):
        L.append(f"{rng.choice(['MX', 'MX', 'MY', 'M'])} {q}")
    body = "\n".join(L) + "\n"
    meas, _ = F._run_exact(body, 256, 1)
    dets = []
    for kk in (1, 2, 3, 4):
        for S in itertools.combinations(range(n), kk):
            par = np.bitwise_xor.reduce(meas[:, list(S)], axis=1)
            if (par == par[0]).all() and not any(set(d) <= set(S) for d in dets):
                dets.append(S)
    if not dets:
        return None
    for S in dets[:5]:
        body += "DETECTOR " + " ".join(f"rec[{q - n}]" for q in S) + "\n"
    return body


def test_random_correlated_read_circuits():
    rng = random.Random(22)
    done = rand = 0
    while done < 40:
        body = _random_case(rng)
        if body is None:
            continue
        r = xtim._xtim.export_dem_text(body, False, False, False, True, [], [], 0.0)
        if r["rejected"]:
            continue
        cnt = law_check(body, shots=1000)
        done += 1
        rand += cnt["random"]
    assert rand >= 4, rand


# ---------------------------------------------------------------------------------------
# 3. the bundled examples
# ---------------------------------------------------------------------------------------
def test_code_switching_faithful_needs_approximate_disjoint_errors_and_matches_per_fault():
    text = xtim.load_example("code_switching_faithful").text
    with pytest.raises(XtimDemError, match="approximate_disjoint_errors"):
        xtim.Circuit(text).detector_error_model(include_expectations=False, drop_gauge_observables=True)
    dem = xtim.Circuit(text).detector_error_model(include_expectations=False, drop_gauge_observables=True,
                                                  approximate_disjoint_errors=True)
    assert sum(1 for i in dem.flattened() if i.type == "error") > 100
    cnt = law_check(text, shots=800, every=3)
    assert cnt["approximated"] >= 5 and cnt["random"] >= 5, cnt


def test_cultivation_d3_faithful_per_fault():
    cnt = law_check(xtim.load_example("cultivation_d3_faithful").text, shots=600, every=5)
    assert cnt["alts"] > 50, cnt


# ---------------------------------------------------------------------------------------
# 4. stim equality on Clifford circuits (semantic), including approximate_disjoint_errors and
#    the conditions under which each tool refuses
# ---------------------------------------------------------------------------------------
CLIFF = {
    "bell_zz_xx": "R 0 1\nH 0\nCX 0 1\n{n}\nMPP Z0*Z1 X0*X1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "ghz3": "R 0 1 2\nH 0\nCX 0 1 0 2\n{n}\nMPP Z0*Z1 Z1*Z2 X0*X1*X2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\n",
    "two_reads": "R 0 1\n{n}\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
}
CHANNELS = ["PAULI_CHANNEL_1(0.1,0.2,0.3) 0", "PAULI_CHANNEL_1(0.01,0.02,0.03) 0",
            "PAULI_CHANNEL_1(0.1,0.1,0) 0", "PAULI_CHANNEL_1(0.04,0,0.04) 1",
            "PAULI_CHANNEL_2(" + ",".join(["0.01"] * 15) + ") 0 1",
            "PAULI_CHANNEL_2(" + ",".join(f"{0.001 * (i + 1):.3f}" for i in range(15)) + ") 0 1",
            "PAULI_CHANNEL_2(0,0,0,0.1,0,0,0,0,0,0,0,0,0,0,0) 0 1",
            "DEPOLARIZE1(0.3) 0", "DEPOLARIZE2(0.2) 0 1", "X_ERROR(0.4) 1", "Y_ERROR(0.2) 0"]
APPROX = [False, True, 0.015, 0.05, 0.2]


def _cases():
    for cn, c in sorted(CLIFF.items()):
        for ch in CHANNELS:
            yield cn, ch, c.format(n=ch)


@pytest.mark.parametrize("cn,ch,text", list(_cases()))
def test_stim_semantic_equality(cn, ch, text):
    """Where stim exports, xtim exports the same map (same option). Where xtim refuses, stim
    refuses. xtim converts exactly in signature space, so it may export a channel stim refuses
    without the approximation (a superset): then xtim's map is the exact conversion of the
    disjoint signature distribution — checked against stim's per-Pauli signatures."""
    for approx in APPROX:
        s, x = try_dem(text, "stim", approx), try_dem(text, "xtim", approx)
        if s is not None and x is not None and maps_equal(x, s):
            continue
        if x is None:
            assert s is None, (approx, s)
            continue
        if s is not None and not maps_equal(x, s):
            # both export but differ: only allowed when stim approximated and xtim is exact
            s_exact = try_dem(text, "stim", False)
            assert s_exact is None, (approx, x, s)
        _assert_exact_conversion(text, x)


def _assert_exact_conversion(text, xmap):
    """xmap's independent mechanisms reproduce the channel's disjoint signature distribution
    exactly (signatures per Pauli from stim, one Pauli at a time)."""
    import re
    m = re.search(r"^(PAULI_CHANNEL_[12]|DEPOLARIZE[12]|[XYZ]_ERROR)\(([^)]*)\) (.*)$", text, re.M)
    name, arg, tg = m.group(1), m.group(2), m.group(3).split()
    (qs, alts), = F.channel_alternatives(name, arg, tg)
    dist = {frozenset(): 1.0}
    for p, d in alts:
        if p <= 0:
            continue
        one = text.replace(m.group(0), "\n".join(f"{P}_ERROR(1) {q}" for q, P in sorted(d.items())))
        sig = frozenset()
        for k in sem_map(stim.Circuit(one).detector_error_model()):   # each p = 1 mechanism
            sig = sig ^ k
        dist[frozenset()] -= p
        dist[sig] = dist.get(sig, 0.0) + p
    got = {frozenset(): 1.0}
    for k, p in xmap.items():
        nd = {}
        for s, q in got.items():
            nd[s] = nd.get(s, 0.0) + q * (1 - p)
            nd[s ^ k] = nd.get(s ^ k, 0.0) + q * p
        got = nd
    for k in set(got) | set(dist):
        assert abs(got.get(k, 0.0) - dist.get(k, 0.0)) <= 1e-12, (k, got.get(k), dist.get(k))


def test_approximate_disjoint_errors_threshold_refusal_by_condition():
    """stim's semantics: True = approximate any channel; a float t = approximate only if every
    probability argument of the channel is <= t (code_switching_faithful's arguments are 0.001)."""
    c = xtim.Circuit(xtim.load_example("code_switching_faithful").text)
    kw = dict(include_expectations=False, drop_gauge_observables=True)
    with pytest.raises(XtimDemError, match="approximate_disjoint_errors"):
        c.detector_error_model(**kw)
    a = sem_map(c.detector_error_model(approximate_disjoint_errors=True, **kw))
    b = sem_map(c.detector_error_model(approximate_disjoint_errors=0.001, **kw))
    assert a and maps_equal(a, b)
    with pytest.raises(XtimDemError, match="threshold"):
        c.detector_error_model(approximate_disjoint_errors=0.0005, **kw)
    with pytest.raises(ValueError):
        c.detector_error_model(approximate_disjoint_errors=1.5, **kw)


# ---------------------------------------------------------------------------------------
# 5. a randomized read feeding a DECISION column
# ---------------------------------------------------------------------------------------
BELL_T = "R 0 1\nH 0\nCX 0 1\n{f}T 0\nT_DAG 1\nMX 0 1\nDETECTOR rec[-1] rec[-2]\n{dec}"


def test_randomized_read_feeding_a_random_valued_decision_refuses():
    """MX 0 of a Bell pair is random noiselessly; a DECISION on it has no reference value, so a
    fault that randomizes that read has no defined decision flip: refused by name."""
    text = BELL_T.format(f="X_ERROR(0.1) 0\n", dec="DECISION(0) rec[-2]\n")
    r = xtim._xtim.export_dem_text(text, True, False, False, False, [], [], 0.0)
    assert not r["ok"] and "DECISION column" in r["error"] and "not deterministic" in r["error"], r
    # approximate_disjoint_errors does not lift it (it is not a conversion problem)
    r = xtim._xtim.export_dem_text(text, True, False, False, False, [], [], 1.0)
    assert not r["ok"] and "DECISION column" in r["error"], r


def test_decision_controls_are_exported():
    # a Pauli fault flips the random-valued decision deterministically (record-flip semantics)
    r = xtim._xtim.export_dem_text(BELL_T.format(f="Z_ERROR(0.1) 0\n", dec="DECISION(0) rec[-2]\n"),
                                   True, False, False, False, [], [], 0.0)
    assert r["ok"] and F.canon(stim.DetectorErrorModel(r["dem"])) == {
        (("D", 0), ("L", 0)): pytest.approx(0.1, rel=1e-15)}, r
    # a deterministic decision (the XX parity) fed by the randomized read: part of the exact law,
    # flipping together with the detector with probability 1/2
    r = xtim._xtim.export_dem_text(BELL_T.format(f="X_ERROR(0.1) 0\n", dec="DECISION(0) rec[-1] rec[-2]\n"),
                                   True, False, False, False, [], [], 0.0)
    assert r["ok"] and F.canon(stim.DetectorErrorModel(r["dem"])) == {
        (("D", 0), ("L", 0)): pytest.approx(0.05, rel=1e-12)}, r
