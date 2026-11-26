"""Engine legs of SPEC §6 T3/T6: per (configuration, d), score_lines "optimal" == the engine's sign-free projector
form; "r2b" == the engine's physical Pauli at t_c; rep3T_pre exact MLE LER.

The engine oracles (tests/_xdem_engine.py) run xtim's exact engine on the xdem export's own deferred circuit (test API
`xdem_deferred_debug`) with the actual faults forced as atoms (each xdem line's representative = its highest-p atom).
"Equal" is exact `==` on float64 pass values. Every checked leg asserts a checked-row count > 0 (>= 3 on ex422_obs;
preflight ruling S11)."""
import random

import numpy as np
import pytest

from xtim import _xtim
from xtim import _xtim_testapi as ta

from _xdem_circuits import NAMES_INREPO, circuit, present, xdem_export
from _xdem_engine import (engine_beta, engine_r2b, engine_vdag_proj, rep3_exact, rep3_exact_ler, representative_atoms,
                          vdag_args)
from _xdem_model import XdemModel
from test_xdem_lines import csr, det_int, mle2_table

LOOP_ROWS = {"rep3T_pre": 500, "rep3T_pre_p1e-2": 1000, "rm15_x": 1000, "msc_d3_canon": 1000, "toy2obs": 500,
             "rm15rep": 1000, "ex422_obs": 300, "ccz_toy_obs_p5": 1000}
R2B_NAMES = ["rep3T_pre", "rm15_x", "msc_d3_canon", "toy2obs", "rm15rep"]


def rows(name, hyp):
    """(text, xdem, sampler, dets, fo, fi, T, H): LOOP_ROWS[name] sampled shots (seed 11) and their hypotheses"""
    text = circuit(name)
    x = xdem_export(text)
    s = _xtim._XdemSampler(x, 65536)
    n = LOOP_ROWS[name]
    dets, obs, cs, fo, fi = s.sample(n, 11, 1, 0, True)
    T = [list(fi[fo[k]:fo[k + 1]]) for k in range(n)]
    D = s.num_detectors
    if hyp == "E":
        H = T
    else:
        lines = ta.xdem_parse(x)["lines"]
        tab = mle2_table([l[0] for l in lines], [sum(1 << j for j in l[1] if j < D) for l in lines])
        H = [tab.get(det_int(dets[k], D), (0.0, []))[1] for k in range(n)]
    return text, x, s, dets, fo, fi, T, H


@pytest.mark.parametrize("name", list(LOOP_ROWS))
@pytest.mark.parametrize("hyp", ["E", "mle2"])
def test_engine_rows(name, hyp):
    if not present(name):
        pytest.skip("external circuit absent")
    text, x, s, dets, fo, fi, T, H = rows(name, hyp)
    n, D = len(T), s.num_detectors
    ho, hi = csr(H)
    opt = s.score_lines(dets, fo, fi, ho, hi, "optimal")
    M = XdemModel.from_text(x)
    rep = representative_atoms(text)
    bad = checked = nontrivial = nlossy = nedge = 0
    for k in range(n):
        d = det_int(dets[k], D)
        piv, lossy, edge = vdag_args(M, H[k], d)               # the greedy pivots of Omega etc., from the oracle
        nlossy += bool(lossy)
        nedge += edge                                          # a CZ edge to a lossy observable (checked too)
        e = engine_vdag_proj(text, [rep[i] for i in T[k]], [rep[i] for i in H[k]], d, piv, lossy)
        if e is None:
            continue
        checked += 1
        nontrivial += bool(T[k] or H[k])
        bad += e != opt[k]
    print(name, hyp, f"checked {checked} (with a fault or a hypothesis: {nontrivial}; lossy {nlossy}, with a "
          f"CZ edge to a lossy observable {nedge}) bad {bad}")
    assert bad == 0
    assert checked >= (3 if name == "ex422_obs" else 1) and nontrivial > 0


@pytest.mark.parametrize("name", R2B_NAMES)
@pytest.mark.parametrize("hyp", ["E", "mle2"])
def test_engine_rows_r2b(name, hyp):
    if not present(name):
        pytest.skip("external circuit absent")
    text, x, s, dets, fo, fi, T, H = rows(name, hyp)
    n, D = len(T), s.num_detectors
    ho, hi = csr(H)
    got = s.score_lines(dets, fo, fi, ho, hi, "r2b")
    k = s.correction(dets, ho, hi, "r2b")
    applied, late = k["applied"], np.diff(k["late_off"]) > 0
    rep = representative_atoms(text)
    bad = checked = napplied = nwithheld = 0
    for k in range(n):
        d = det_int(dets[k], D)
        e = engine_r2b(text, [rep[i] for i in T[k]], [rep[i] for i in H[k]], d)
        if e is None:
            continue
        checked += 1
        napplied += bool(applied[k])
        nwithheld += bool(late[k] and not applied[k])
        bad += e != got[k]
    print(name, hyp, f"checked {checked} (R2b applied: {napplied}, withheld: {nwithheld}) bad {bad}")
    assert bad == 0
    assert checked > 0
    if hyp == "E":                                             # the true hypothesis exercises the Pauli at t_c
        assert napplied > 0


BETA_SKIP = {"distillation_15_1_3": "conventional", "cube_ccz": "conventional",
             "code_switching_faithful": "refused", "code_switching_rate": "refused"}


@pytest.mark.parametrize("name", NAMES_INREPO)
def test_beta_on_K_against_the_exact_engine(name):
    """T1(d), engine leg: beta of every K_basis element = the sign of <prod Q_a> on the noiseless real circuit"""
    if name in BETA_SKIP:                                      # test_xdem_oracle_configs.EXPECTED_SKIP's circuits
        pytest.skip(f"{name}: {BETA_SKIP[name]} export")
    text = circuit(name)
    h = ta.xdem_header_debug(text)
    assert len(h["K_basis"]) > 0 or h["m"] == 0
    got = [engine_beta(text, c) for c in h["K_basis"]]
    print(name, "K_basis", len(got), "beta = 1:", sum(got))
    assert got == list(h["beta_K"])


def test_rep3T_pre_exact_mle_ler():
    """T6: rep3T_pre summed LER with the exact MLE, every configuration and every d: R2b 9.9705e-6 (the dc-DEM v1
    value, tests/test_dc_lines.py) within 1e-9 relative; optimal with Ê = E is 0. Every row also equals its engine
    oracle (max |pass - engine| = 0)."""
    want = 9.970535980841078e-06
    ler, worst, n = rep3_exact("r2b", 1e-3)
    print(f"rep3T_pre R2b exact-MLE LER {ler!r} over {n} (configuration, d) rows, max |pass - engine| {worst}")
    assert abs(ler - want) <= 1e-9 * want
    assert worst == 0.0
    ler_o, worst_o, n_o = rep3_exact("optimal", 1e-3, hypothesis="E")
    print(f"rep3T_pre optimal (Ê = E) LER {ler_o!r} over {n_o} rows, max |pass - engine| {worst_o}")
    assert ler_o == 0.0 and worst_o == 0.0
    assert rep3_exact_ler("r2b", 1e-3) == ler


def test_engine_lossy_rows_cultivation_d5():
    """Lossy readouts are common on real circuits (ERRATA E2: cultivation_d5 2.4 % of shots under the true
    hypothesis): on cultivation_d5's lossy rows (Ê = E, and on odd shots Ê = E with one fired line dropped), "optimal"
    == the engine form that reads the lossy observables last, in the fired frame (SPEC §7 Q4: no V^dag fallback)"""
    text = circuit("cultivation_d5")
    x = xdem_export(text)
    s = _xtim._XdemSampler(x, 65536)
    n = 2000
    dets, obs, cs, fo, fi = s.sample(n, 11, 1, 0, True)
    T = [list(fi[fo[k]:fo[k + 1]]) for k in range(n)]
    rng = random.Random(3)
    H = [t if k % 2 == 0 or not t else sorted(rng.sample(t, len(t) - 1)) for k, t in enumerate(T)]
    opt = s.score_lines(dets, fo, fi, *csr(H), "optimal")
    M = XdemModel.from_text(x)
    rep = representative_atoms(text)
    bad = checked = wrong = nedge = 0
    for k in range(n):
        d = det_int(dets[k], s.num_detectors)
        piv, lossy, edge = vdag_args(M, H[k], d)
        if not lossy:
            continue
        nedge += edge
        e = engine_vdag_proj(text, [rep[i] for i in T[k]], [rep[i] for i in H[k]], d, piv, lossy)
        checked += 1
        wrong += H[k] != T[k]
        bad += e != opt[k]
    print(f"cultivation_d5 lossy rows checked {checked} (wrong hypothesis {wrong}; with a CZ edge {nedge}) "
          f"bad {bad}")
    assert bad == 0 and checked > 0 and wrong > 0


R2B_LEVERS = {1: "K from E^< and E^>", 2: "no guard", 3: "c = f_h|_O always", 4: "no cocycle sign in K", 5: "t_c - 1"}


def test_r2b_levers_fail_against_the_engine():
    """T3 mutations via the C++ levers (kXdemMut*, xdem_sampler.hpp): each makes "r2b" disagree with the engine's
    physical Pauli at t_c on some row; unmutated, 0 rows disagree. Hypotheses: the fired lines with 0-2 random lines
    toggled (wrong hypotheses reach the cocycle and t_c levers)."""
    names = [n for n in ("rep3T_pre_p1e-2", "toy2obs", "msc_d3_canon") if present(n)]
    caught = {m: 0 for m in R2B_LEVERS}
    for name in names:
        text = circuit(name)
        x = xdem_export(text)
        s = _xtim._XdemSampler(x, 65536)
        n = 2000
        dets, obs, cs, fo, fi = s.sample(n, 12, 1, 0, True)
        T = [list(fi[fo[k]:fo[k + 1]]) for k in range(n)]
        rng = random.Random(5)
        H = [sorted(set(t) ^ set(rng.sample(range(s.num_lines), rng.randint(0, 2)))) for t in T]
        rep = representative_atoms(text)
        eng = [engine_r2b(text, [rep[i] for i in T[k]], [rep[i] for i in H[k]], det_int(dets[k], s.num_detectors))
               for k in range(n)]
        ho, hi = csr(H)
        for mut in [0] + list(R2B_LEVERS):
            p = ta.xdem_lines_fixed(x, fo, fi, dets, ho, hi, "r2b", mut)[0]
            bad = sum(e is not None and p[k] != e for k, e in enumerate(eng))
            print(name, "lever", mut, R2B_LEVERS.get(mut, "none"), "rows disagreeing with the engine", bad)
            if mut == 0:
                assert bad == 0
            else:
                caught[mut] += bad
    if "msc_d3_canon" not in names:
        # the t_c - 1 lever shows on msc_d3_canon only (1 row of 2000); no in-repo circuit shows it (searched:
        # rep3T_pre, rep3T_pre_p1e-2, toy2obs, rm15rep, ccz_toy_obs_p5, cultivation_d3_faithful, 9000 shots each)
        # the c = f_h|_O lever shows on msc_d3_canon only too (49 rows of 2000; 0 on rep3T_pre_p1e-2 and toy2obs)
        # a print, not warnings.warn: an external circuit's absence must not fail a `-W error` run (Task 10 review)
        print("test_r2b_levers_fail_against_the_engine: msc_d3_canon is absent, so the 'c = f_h|_O always' lever (3) "
              "and the 't_c - 1' lever (5) are NOT checked against the engine in this run")
        caught.pop(3)
        caught.pop(5)
    assert all(v > 0 for v in caught.values()), caught


def test_ex422_rule_x_rows_score_1():
    """SPEC T6 / preflight S12: the ex422_obs configurations where vdag's EngineV closed-form pair signs gave 0
    (loop/out/diag_ex422.json: one fired line, Ê = E, d in {2, 3}) score 1 under "optimal", and so does the engine's
    sign-free projector form; here every single line with every d in its support"""
    text = circuit("ex422_obs")
    x = xdem_export(text)
    s = _xtim._XdemSampler(x, 65536)
    M = XdemModel.from_text(x)
    rep = representative_atoms(text)
    from _xdem_engine import engine_detector_law
    rows = 0
    for i in range(len(M.lines)):
        for d, q in sorted(engine_detector_law(text, [rep[i]]).items()):
            if q <= 0:
                continue
            dets = np.frombuffer(int(d).to_bytes((M.D + 7) // 8, "little"), np.uint8).reshape(1, -1)
            o, ids = csr([[i]])
            p = s.score_lines(dets, o, ids, o, ids, "optimal")[0]
            piv, lossy, edge = vdag_args(M, [i], d)
            e = engine_vdag_proj(text, [rep[i]], [rep[i]], d, piv, lossy)
            print("ex422_obs line", i, "d", d, "pivots", piv, "optimal", p, "engine", e)
            assert p == 1.0 and e == 1.0 and not lossy
            rows += 1
    assert rows >= 3


# a loop-fuzz circuit (tests/test_xdem_fuzz.py, seed 54) whose wrong hypotheses make the observable's rotated readout
# lossy: the engine reads it last, in the fired frame (SPEC §7 Q4), where V^dag would give 1/2 and the optimal K 1
LOSSY_FUZZ = """R 0 1 2 3
S 3
SWAP 1 2
CX 1 3
SWAP 2 3
H 3
S 3
CY 0 2
Z 0
DEPOLARIZE1(0.2) 1
MR 0
CX rec[-1] 0
CCZ 2 3 1
CCZ 3 1 0
PAULI_CHANNEL_1(0.0, 0.2, 0.0) 1
M 0
MRX 0
MYY(0.08) 2 0
DETECTOR rec[-4]
DEPOLARIZE2(0.1) 1 0
CCZ 3 1 0
CCZ 2 3 1
Z 0
CY 0 2
S_DAG 3
H 3
SWAP 2 3
CX 1 3
SWAP 1 2
S_DAG 3
MX !0
M 1
MX 2
MY !3
DETECTOR rec[-8] rec[-7] rec[-5] rec[-3] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-5] rec[-1]
"""


def test_engine_lossy_rows_with_wrong_hypotheses():
    """every (E, Ê) of <= 1 line each, every d with Pr(d | E) > 0: "optimal" == the engine; lossy rows occur, some
    with Ê != E, and there the engine form differs from plain V^dag (read after C_H^dag)"""
    from _xdem_engine import deferred, engine_detector_law
    text = LOSSY_FUZZ
    x = xdem_export(text)
    M = XdemModel.from_text(x)
    rep = representative_atoms(text)
    L = len(M.lines)
    rows = [(E, H, d) for E in [[]] + [[i] for i in range(L)] for H in [[]] + [[i] for i in range(L)]
            for d, q in sorted(engine_detector_law(text, [rep[i] for i in E]).items()) if q > 0]
    fo, fi = csr([r[0] for r in rows])
    ho, hi = csr([r[1] for r in rows])
    dets = np.array([np.frombuffer(int(d).to_bytes((M.D + 7) // 8, "little"), np.uint8) for _, _, d in rows])
    got = ta.xdem_lines_fixed(x, fo, fi, dets, ho, hi, "optimal")[0]
    bad = lossy_wrong = differs_from_vdag = 0
    for k, (E, H, d) in enumerate(rows):
        piv, lossy, _ = vdag_args(M, H, d)
        e = engine_vdag_proj(text, [rep[i] for i in E], [rep[i] for i in H], d, piv, lossy)
        bad += e != got[k]
        if lossy and E != H:
            lossy_wrong += 1
            differs_from_vdag += engine_vdag_proj(text, [rep[i] for i in E], [rep[i] for i in H], d, piv) != e
    print(f"rows {len(rows)} bad {bad}; lossy with Ê != E {lossy_wrong}, of which plain V^dag differs "
          f"{differs_from_vdag}")
    assert bad == 0 and lossy_wrong > 0 and differs_from_vdag > 0
    assert deferred(text).tc_index >= 0


def test_engine_e4_edge_rows():
    """ERRATA E4 on the slow fuzz's 10 CZ-edge rows (Ê = E; L0, L2 lossy with a determined parity): the engine, with
    its own c (engine_lossy_c: the support of the K-law with Ê forced, given d), == the C++ "optimal" == 1/2 on every
    row, including the 6 that f_K = 0 scored 0.0 (V^dag 1/2 on all 6, "none" 1/2 on 3 and 0 on 3); the export's
    lossy frame == the engine's c (asserted inside)"""
    from test_xdem_optimal import E4_FUZZ, E4_ROWS
    text = E4_FUZZ
    x = xdem_export(text)
    M = XdemModel.from_text(x)
    rep = representative_atoms(text)
    fo, fi = csr([E for E, _ in E4_ROWS])
    dets = np.array([np.frombuffer(int(d).to_bytes((M.D + 7) // 8, "little"), np.uint8) for _, d in E4_ROWS])
    got = ta.xdem_lines_fixed(x, fo, fi, dets, fo, fi, "optimal")[0]
    edge = 0
    for k, (E, d) in enumerate(E4_ROWS):
        piv, lossy, e_ = vdag_args(M, E, d)
        assert lossy == [0, 2]
        edge += e_
        e = engine_vdag_proj(text, [rep[i] for i in E], [rep[i] for i in E], d, piv, lossy)
        assert e == got[k] == 0.5, (E, d, e, got[k])
    assert edge == len(E4_ROWS)
