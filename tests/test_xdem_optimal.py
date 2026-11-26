"""score(lines=, "optimal") and correction() (SPEC §4.3, §4.6, §6 T6) against the model oracle, per shot, exactly."""
import random
from pathlib import Path

import numpy as np
import pytest
from xtim.errors import XtimXdemError

from xtim import _xtim

from _xdem_circuits import circuit, present, xdem_export
from _xdem_model import XdemModel, pc
from _xdem_synth import random_text
from test_xdem_lines import csr

DATA = Path(__file__).parent / "data" / "xdem"


def det_ints(dets, D):
    d = np.unpackbits(dets, axis=1, bitorder="little", count=D)
    return [int(sum(int(b) << j for j, b in enumerate(row))) for row in d]


def per_shot(text, shots, seed, decoder):
    """every shot's pass and every (shot, observable) row of correction() equal the oracle's k_optimal / pass_optimal
    (exact ==); returns (passes, non-lossy rows checked, non-lossy rows with S_k nonempty)"""
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, seed, 1, 0, True)
    H = [decoder(list(fi[fo[k]:fo[k + 1]]), k) for k in range(shots)]
    ho, hi = csr(H)
    got = s.score_lines(dets, fo, fi, ho, hi, "optimal")
    corr = s.correction(dets, ho, hi, "optimal")
    checked = with_s = 0
    for k, sv in enumerate(det_ints(dets, M.D)):
        T = list(fi[fo[k]:fo[k + 1]])
        assert got[k] == M.pass_optimal(T, H[k], sv), (k, T, H[k], sv)
        K = M.k_optimal(H[k], sv)
        for o in range(M.O):
            row = k * M.O + o
            assert bool(corr["lossy"][row]) == K["lossy"][o]
            if K["lossy"][o]:                                            # ERRATA E4: the frame is c there
                assert corr["frame"][row] == K["frame"][o], (k, H[k], sv, o)
                continue
            assert list(corr["axes"][corr["axes_off"][row]:corr["axes_off"][row + 1]]) == sorted(K["S"][o])
            assert sum(1 << int(j) for j in corr["piv"][corr["piv_off"][row]:corr["piv_off"][row + 1]]) == K["P"][o]
            assert corr["y"][row] == K["y"][o] and corr["frame"][row] == K["frame"][o]
            checked += 1
            with_s += bool(K["S"][o])
    return got, checked, with_s


def synth_text(seed, attempt):
    """attempt 0: the brief's ranges (D = 1 and A = 1 allowed); later attempts (the S_k-nonempty redraws): D, A >= 2"""
    rng = random.Random(seed * 1000 + attempt)
    lo = 1 if attempt == 0 else 2
    return random_text(rng, D=rng.randint(lo, 4), O=rng.randint(1, 3), r=rng.randint(1, 3), m=rng.randint(0, 2),
                       A=rng.randint(lo, 5), L=6)


def synth_decoder(T, k):
    rng = random.Random(k)
    return T if k % 2 else sorted(rng.sample(range(6), rng.randint(0, 3)))


@pytest.mark.parametrize("seed", range(60))
def test_synthetic_per_shot_equals_the_oracle(seed):
    """Task 1 review: the closed-form eps's rotation-dependent terms (S_k nonempty) are exercised on EVERY seed (N =
    60): a seed's first model has the brief's ranges; it redraws (D >= 2, A >= 2) until some non-lossy row of its 200
    shots has S_k nonempty. Every drawn model is checked in full."""
    for attempt in range(20):
        _, checked, with_s = per_shot(synth_text(seed, attempt), 200, seed, synth_decoder)
        assert checked > 0                                               # preflight S11
        if with_s:
            return
    pytest.fail("no model with S_k nonempty in 20 draws")


@pytest.mark.parametrize("name", ["ex422_obs", "toy2obs", "rm15rep", "ccz_toy_obs_p5", "rep3T_pre", "msc_d3_canon"])
def test_circuits_per_shot_equals_the_oracle(name):
    if not present(name):
        pytest.skip("external circuit absent")
    _, checked, _ = per_shot(xdem_export(circuit(name)), 1000, 17, lambda T, k: T)
    assert checked > 0                                                   # preflight S11


@pytest.mark.parametrize("name", ["toy2obs", "ccz_toy_obs_p5", "rep3T_pre_p1e-2"])
def test_circuits_under_wrong_hypotheses_equal_the_oracle(name):
    """random 0-3 line hypotheses: a wrong Ê gives wrong signs (SPEC §4.3 step 6), lossy readouts and shots outside
    law(Ê) — every row still equals the oracle"""
    text = xdem_export(circuit(name))
    L = len(XdemModel.from_text(text).lines)

    def dec(T, k):
        rng = random.Random(k)
        return sorted(rng.sample(range(L), rng.randint(0, min(3, L))))

    _, checked, _ = per_shot(text, 600, 23, dec)
    assert checked > 0


def test_with_the_true_hypothesis_every_non_lossy_shot_passes():          # optrecovery Thm 2-3
    for name in ("ex422_obs", "rm15rep", "ccz_toy_obs_p5"):
        got, checked, _ = per_shot(xdem_export(circuit(name)), 2000, 5, lambda T, k: T)
        assert checked > 0
        assert (got == 1.0).all()


def test_ex422_configurations_where_the_vdag_sign_rule_gave_0():          # T6 / LOOP §6
    text = xdem_export(circuit("ex422_obs"))
    M = XdemModel.from_text(text)
    checked = 0
    for T, svals in (([0], (3, 2)), ([1], (2, 3))):
        for sv in svals:
            if M.pass_of(M.law(M.circuit(T)), sv, 0) is None:
                continue
            s = _xtim._XdemSampler(text, 65536)
            dets = np.packbits(np.array([[(sv >> j) & 1 for j in range(M.D)]], np.uint8), axis=1, bitorder="little")
            fo, fi = csr([T])
            assert s.score_lines(dets, fo, fi, fo, fi, "optimal")[0] == 1.0
            checked += 1
    assert checked >= 3                                                  # preflight S11


def fired_on_line0(text, shots, seed):
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, seed, 1, 0, True)
    fired = [k for k in range(shots) if 0 in list(fi[fo[k]:fo[k + 1]])]
    return s, dets, fo, fi, fired


def test_lossy_readout_scores_one_half():                                  # T6 synthetic lossy file
    text = (DATA / "synth" / "lossy.xdem").read_text()
    M = XdemModel.from_text(text)
    for sv in (0, 1):                                                    # the plan author's oracle check
        assert M.k_optimal([0], sv)["lossy"] == [True]
        assert M.pass_optimal([0], [0], sv) == 0.5
    s, dets, fo, fi, fired = fired_on_line0(text, 2000, 1)
    got = s.score_lines(dets, fo, fi, fo, fi, "optimal")
    assert fired and all(got[k] == 0.5 for k in fired)
    assert s.counters()["lines_lossy"] >= len(fired)
    corr = s.correction(dets, fo, fi, "optimal")
    assert all(corr["lossy"][k] == 1 and corr["frame"][k] == 0 for k in fired)
    assert all(corr["axes_off"][k] == corr["axes_off"][k + 1] for k in fired)


def test_optimal_beats_none_on_a_hidden_coin():
    """one axis Q0 = Z_{L0} Z_{g0}, one line rotating about it: the readout is a coin of the gauge qubit's that the
    hypothesis knows (t = 0, not lossy); "optimal" undoes it (pass 1), "none" scores the coin (1/2)"""
    text = "xdem 1 detectors 0 observables 1 gauge 1 hidden 0 tc 1\naxis Q0 img L0 gauge g0\nerror(0.3) rot +Q0\n"
    s, dets, fo, fi, fired = fired_on_line0(text, 2000, 2)
    assert fired
    opt = s.score_lines(dets, fo, fi, fo, fi, "optimal")
    none = s.score_lines(dets, fo, fi, fo, fi, "none")
    assert all(opt[k] == 1.0 and none[k] == 0.5 for k in fired)
    assert not s.correction(dets, fo, fi, "optimal")["lossy"].any()


def test_kmemo_is_bounded_and_invisible():                                 # Review Focus (memo growth)
    text = random_text(random.Random(3), D=4, O=2, r=3, m=2, A=5, L=8)
    a, b = _xtim._XdemSampler(text, 4), _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = a.sample(3000, 2, 1, 0, True)
    rng = random.Random(1)
    H = [sorted(rng.sample(range(8), rng.randint(0, 4))) for _ in range(3000)]
    ho, hi = csr(H)
    assert np.array_equal(a.score_lines(dets, fo, fi, ho, hi, "optimal"), b.score_lines(dets, fo, fi, ho, hi, "optimal"))
    ca, cb = a.correction(dets, ho, hi, "optimal"), b.correction(dets, ho, hi, "optimal")
    assert all(np.array_equal(ca[key], cb[key]) for key in ca)
    c = a.counters()
    assert c["kmemo_entries"] <= 4 and c["kmemo_clears"] > 0
    assert "kmemo_law_clears" in c and b.counters()["kmemo_clears"] == 0


def test_lines_counters_equal_the_oracle():
    """lines_shots, lines_lossy (a lossy observable), lines_outside (d outside law(Ê)'s detector support),
    lines_coin_after (the corrected law has observable coins at d) under wrong hypotheses"""
    text = synth_text(7, 0)
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    n = 800
    dets, obs, cs, fo, fi = s.sample(n, 4, 1, 0, True)
    rng = random.Random(9)
    H = [sorted(rng.sample(range(6), rng.randint(0, 3))) for _ in range(n)]
    s.score_lines(dets, fo, fi, *csr(H), "optimal")
    lossy = outside = coin = 0
    for k, sv in enumerate(det_ints(dets, M.D)):
        K = M.k_optimal(H[k], sv)
        lossy += any(K["lossy"])
        outside += not M.given(M.law(M.circuit(H[k])), sv)[3]
        T = list(fi[fo[k]:fo[k + 1]])
        coin += M.given(M.law(M.circuit(T) + K["circuit"]), sv)[0] > 0
    c = s.counters()
    assert (c["lines_shots"], c["lines_lossy"], c["lines_outside"], c["lines_coin_after"]) == (n, lossy, outside, coin)
    assert lossy + outside + coin > 0


@pytest.mark.parametrize("name", ["rep3T_pre_p1e-2", "toy2obs"])
def test_r2b_export_equals_the_oracle(name):
    """correction(..., "r2b"): eta^> per late axis, tau_K, applied, c per shot (SPEC §4.4 steps 1-4, §4.6)"""
    text = xdem_export(circuit(name))
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    n = 800
    dets, obs, cs, fo, fi = s.sample(n, 13, 1, 0, True)
    rng = random.Random(3)
    H = [sorted(rng.sample(range(len(M.lines)), rng.randint(0, 3))) if k % 2 else list(fi[fo[k]:fo[k + 1]])
         for k in range(n)]
    corr = s.correction(dets, *csr(H), "r2b")
    applied = 0
    for k, sv in enumerate(det_ints(dets, M.D)):
        K = M.k_r2b(H[k], sv)
        lo, hi = corr["late_off"][k], corr["late_off"][k + 1]
        assert dict(zip(map(int, corr["late_axis"][lo:hi]), map(int, corr["late_eta"][lo:hi]))) == K["eta"]
        assert list(corr["late_axis"][lo:hi]) == sorted(K["eta"])
        to, th = corr["tau_off"][k], corr["tau_off"][k + 1]
        assert sum(1 << int(a) for a in corr["tau_axis"][to:th]) == K["tau"]
        assert bool(corr["applied"][k]) == K["applied"]
        assert sum(int(corr["c"][k * M.O + o]) << o for o in range(M.O)) == K["c"]
        applied += K["applied"]
    assert applied > 0
    assert s.counters()["lines_shots"] == 0                              # an export is not a scoring call


def test_correction_refusals():
    s = _xtim._XdemSampler(xdem_export(circuit("rep3T_pre")), 65536)
    dets, obs, cs, fo, fi = s.sample(20, 1, 1, 0, True)
    with pytest.raises(ValueError, match='nothing to export for "none"'):
        s.correction(dets, fo, fi, "none")
    with pytest.raises(ValueError, match="correction"):
        s.correction(dets, fo, fi, "Optimal")
    with pytest.raises(IndexError, match="outside"):
        s.correction(dets, *csr([[s.num_lines]] + [[] for _ in range(19)]), "optimal")
    lossy = (DATA / "synth" / "lossy.xdem").read_text()
    t = _xtim._XdemSampler(lossy.replace("tc 1", "tc 0").replace("rot", "late"), 65536)   # a late rotation on D0
    d, o, c, f0, f1 = t.sample(10, 1, 1, 0, True)
    with pytest.raises(XtimXdemError, match="not in canonical single-stage form at t_c"):
        t.correction(d, f0, f1, "r2b")


def test_correction_shapes():
    text = xdem_export(circuit("toy2obs"))
    s = _xtim._XdemSampler(text, 65536)
    n = 300
    dets, obs, cs, fo, fi = s.sample(n, 3, 1, 0, True)
    O = s.num_observables
    c = s.correction(dets, fo, fi, "optimal")
    assert set(c) == {"axes_off", "axes", "piv_off", "piv", "y", "frame", "lossy"}
    assert c["axes_off"].shape == (n * O + 1,) and c["piv_off"].shape == (n * O + 1,)
    assert c["axes_off"][-1] == len(c["axes"]) and c["piv_off"][-1] == len(c["piv"])
    assert all(c[k].shape == (n * O,) for k in ("y", "frame", "lossy"))
    r = s.correction(dets, fo, fi, "r2b")
    assert set(r) == {"late_off", "late_axis", "late_eta", "tau_off", "tau_axis", "applied", "c"}
    assert r["late_off"].shape == (n + 1,) and r["tau_off"].shape == (n + 1,) and r["applied"].shape == (n,)
    assert r["c"].shape == (n * O,)


def test_k_optimal_without_the_memo_equals_the_oracle():
    """XdemSampler::k_optimal (the public C++ step, from scratch) on hypotheses in any order with repeats; the lossy
    rows' frames (ERRATA E4's c) on the E4 files, whose lossy observables are tied"""
    from xtim import _xtim_testapi as ta
    checked = lossy = 0
    texts = [(synth_text(seed, 0), seed) for seed in range(8)]
    texts += [(E4_PAIR, 100), (E4_INDEP, 101), (xdem_export(E4_FUZZ), 102)]
    for text, seed in texts:
        M = XdemModel.from_text(text)
        L = len(M.lines)
        rng = random.Random(seed)
        for _ in range(40):
            H = [rng.randrange(L) for _ in range(rng.randint(0, 4))]
            sv = rng.getrandbits(M.D)
            got, K = ta.xdem_k_optimal(text, H, sv), M.k_optimal(H, sv)
            for o in range(M.O):
                assert got[o]["lossy"] == K["lossy"][o]
                if K["lossy"][o]:                                        # ERRATA E4: the frame is c there
                    assert got[o]["frame"] == K["frame"][o], (seed, H, sv, o)
                    lossy += 1
                    continue
                assert got[o]["S"] == sorted(K["S"][o]) and sum(1 << j for j in got[o]["P"]) == K["P"][o]
                assert got[o]["y"] == K["y"][o] and got[o]["frame"] == K["frame"][o]
                checked += 1
    assert checked > 0 and lossy > 0


def fast_equals_generic(text, shots, seed, decoder):
    """score_lines("optimal") on the sampler's fast path (the law of [T, K] in beta' coordinates) == the general path
    (law_into over AxisView{file axes, K strings}), per shot and on the lines_* counters; returns the shots compared"""
    from xtim import _xtim_testapi as ta
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, seed, 1, 0, True)
    H = [decoder(list(fi[fo[k]:fo[k + 1]]), k) for k in range(shots)]
    ho, hi = csr(H)
    a = ta.xdem_lines_fixed(text, fo, fi, dets, ho, hi, "optimal", 0, False)
    b = ta.xdem_lines_fixed(text, fo, fi, dets, ho, hi, "optimal", 0, True)
    assert a[3]["fast_path"] and not b[3]["fast_path"]
    assert a[3]["kfast_laws"] > 0 and b[3]["kfast_laws"] == 0           # the beta' law ran (default leg) / never ran
    assert np.array_equal(a[0], b[0])
    skip = ("fast_path", "kfast_laws")
    assert {k: v for k, v in a[3].items() if k not in skip} == {k: v for k, v in b[3].items() if k not in skip}
    return shots


@pytest.mark.parametrize("seed", range(60))
def test_fast_law_equals_the_generic_law_synthetic(seed):     # Task 8 fix round 1
    assert fast_equals_generic(synth_text(seed, 0), 300, seed, synth_decoder) > 0


@pytest.mark.parametrize("name", ["lossy", "e4_fuzz", "e4_pair", "e4_indep", "ex422_obs", "toy2obs", "rm15rep",
                                  "ccz_toy_obs_p5", "rep3T_pre_p1e-2", "msc_d3_canon"])
def test_fast_law_equals_the_generic_law_circuits(name):
    if name == "lossy":
        text = (DATA / "synth" / "lossy.xdem").read_text()
    elif name == "e4_fuzz":
        text = xdem_export(E4_FUZZ)
    elif name in ("e4_pair", "e4_indep"):
        text = E4_PAIR if name == "e4_pair" else E4_INDEP
    elif not present(name):
        pytest.skip("external circuit absent")
    else:
        text = xdem_export(circuit(name))
    L = _xtim._XdemSampler(text, 65536).num_lines

    def dec(T, k):                                                   # H = E on even shots, random 0-3 lines on odd
        rng = random.Random(k)
        return T if k % 2 == 0 else sorted(rng.sample(range(L), rng.randint(0, min(3, L))))

    assert fast_equals_generic(text, 2000, 29, dec) > 0


E2_HAND = """xdem 1 detectors 1 observables 2 gauge 3 hidden 0 tc 1
axis Q0 img D0 L0 gauge g0
axis Q1 img D0 gauge g1
axis Q2 img L1 gauge g2
axis Q3 img L0 L1 gauge g0 g1 g2
error(0.3) rot +Q0 +Q1 +Q2 +Q3
"""


def test_kprime_edge_to_lossy_observable():                               # ERRATA E2 (Task 8 review)
    """T = [], h = [the line], s = 0: L0 is lossy (t_0 = e_D0), L1 is not (S_1 = {Q2, Q3}, A_1 = {L0, g0, g1}), so K holds
    CZ(L1, L0). X_L0 is then random: Pr(o = 0) = 1/4. The spec's kappa-only law gives 1/2 (it declares X_L0 deterministic);
    kappa' = [kappa | delta] gives 1/4. Both the fast and the generic law, and the oracle."""
    from xtim import _xtim_testapi as ta
    M = XdemModel.from_text(E2_HAND)
    K = M.k_optimal([0], 0)
    assert K["lossy"] == [True, False] and sorted(K["S"][1]) == [2, 3]
    assert M.pass_optimal([], [0], 0) == 0.25
    dets = np.zeros((1, 1), np.uint8)
    fo, fi = csr([[]])
    ho, hi = csr([[0]])
    s = _xtim._XdemSampler(E2_HAND, 65536)
    assert s.score_lines(dets, fo, fi, ho, hi, "optimal")[0] == 0.25
    for general in (False, True):
        assert ta.xdem_lines_fixed(E2_HAND, fo, fi, dets, ho, hi, "optimal", 0, general)[0][0] == 0.25
    c = s.correction(dets, ho, hi, "optimal")
    assert list(c["lossy"]) == [1, 0] and list(c["axes"]) == [2, 3]


def lossy_shots_equal_the_oracle(name, shots, seed, cap):
    """H = fired on a real circuit, the shots with a lossy observable: pass and every correction row == the oracle;
    returns (lossy shots checked, of them with a CZ edge from a non-lossy observable to a lossy one)"""
    text = xdem_export(circuit(name))
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, seed, 1, 0, True)
    got = s.score_lines(dets, fo, fi, fo, fi, "optimal")
    corr = s.correction(dets, fo, fi, "optimal")
    O, n, edge = M.O, 0, 0
    sv_all = det_ints(dets, M.D)
    for k in range(shots):
        if not corr["lossy"][k * O:(k + 1) * O].any():
            continue
        T, sv = list(fi[fo[k]:fo[k + 1]]), sv_all[k]
        assert got[k] == M.pass_optimal(T, T, sv), (k, T, sv)
        K = M.k_optimal(T, sv)
        for o in range(O):
            row = k * O + o
            assert bool(corr["lossy"][row]) == K["lossy"][o]
            if K["lossy"][o]:                                            # ERRATA E4: the frame is c there
                assert corr["frame"][row] == K["frame"][o], (k, T, sv, o)
                continue
            assert list(corr["axes"][corr["axes_off"][row]:corr["axes_off"][row + 1]]) == sorted(K["S"][o])
            assert sum(1 << int(j) for j in corr["piv"][corr["piv_off"][row]:corr["piv_off"][row + 1]]) == K["P"][o]
            assert corr["y"][row] == K["y"][o] and corr["frame"][row] == K["frame"][o]
        phi = {o: 0 for o in range(O) if not K["lossy"][o]}
        for o in phi:
            for a in K["S"][o]:
                phi[o] ^= M.axes[a][1]
        edge += any(phi[o] >> (M.D + j) & 1 for o in phi for j in range(O) if K["lossy"][j])
        n += 1
        if n == cap:
            break
    return n, edge


def test_lossy_shots_on_cultivation_d5_equal_the_oracle():                # Task 8 review, Important 1
    n, _ = lossy_shots_equal_the_oracle("cultivation_d5", 8000, 41, 10 ** 9)
    assert n > 0


@pytest.mark.slow
def test_lossy_shots_on_bt27_equal_the_oracle():                          # Task 8 review, Important 1 (O = 9: E2 edges)
    if not present("bt27_p1e-3"):
        pytest.skip("external circuit absent")
    n, edge = lossy_shots_equal_the_oracle("bt27_p1e-3", 2000, 43, 60)
    assert n > 0 and edge > 0



# ERRATA E4 (Task 10 review; owner-approved 2026-11-17): the frame on the lossy observables is a point c of the
# support of law([Ê, K]) restricted to them, given d, not f_K = 0. The rule: the support point whose difference from
# f_Ê there is lexicographically least (observable 0 first), so c = f_Ê there whenever f_Ê is in the support.
# The slow loop fuzz's (seed 54) circuit holding all 10 of its CZ-edge rows: L0, L2 lossy, L0 + L2 determined by d.
E4_FUZZ = """R 0 1 2 3 4 5 6
SQRT_X 2
Z 3
SQRT_XX 6 2
CY 2 4
XCZ 3 1
S 1
ISWAP 1 0
MPP X1
DEPOLARIZE1(0.08) 0
CCZ 5 4 1
PAULI_CHANNEL_1(0.05, 0.2, 0.0) 5
MPP Z5*Y0
DEPOLARIZE1(0.15) 4 5
CCZ 5 4 1
ISWAP_DAG 1 0
S_DAG 1
XCZ 3 1
CY 2 4
SQRT_XX_DAG 6 2
MY(0.2) 3
MPP(0.03) X2*Y1
MRY 0
Z 3
SQRT_X_DAG 2
M(0.08) 0
M !1
M 2
M 3
M 4
M 5
M 6
DETECTOR rec[-3] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-12] rec[-8] rec[-6] rec[-5] rec[-3] rec[-2] rec[-1]
OBSERVABLE_INCLUDE(1) rec[-3] rec[-2] rec[-1]
OBSERVABLE_INCLUDE(2) rec[-12] rec[-8] rec[-6] rec[-5] rec[-2] rec[-1]
"""
E4_ROWS = [(E, d) for E in ([0], [0, 2], [0, 1, 2], [0, 2, 3], [0, 3]) for d in (0, 1)]   # with Ê = E
E4_WAS_ZERO = {((0, 2), 0), ((0, 2), 1), ((0, 1, 2), 0), ((0, 1, 2), 1), ((0, 3), 0), ((0, 3), 1)}   # under f_K = 0
# two lossy observables whose parity is tied to D0 (L0 + L1 = d): at d = 1 "none" (c = f_Ê = 0) and f_K = 0 score 0
E4_PAIR = """xdem 1 detectors 1 observables 2 gauge 2 hidden 0 tc 1
axis Q0 img D0 L0 gauge g0
axis Q1 img D0 L1 gauge g1
error(0.3) rot +Q0 +Q1
"""
# two independent lossy observables: 1/2 each, 1/4 jointly
E4_INDEP = """xdem 1 detectors 2 observables 2 gauge 4 hidden 0 tc 1
axis Q0 img D0 L0 gauge g0
axis Q1 img D0 gauge g1
axis Q2 img D1 L1 gauge g2
axis Q3 img D1 gauge g3
error(0.3) L0 rot +Q0 +Q1 +Q2 +Q3
"""


def one_row(text, T, H, sv, corr):
    """(pass, correction export) of one shot: fired T, hypothesis H, detector row sv (an int)"""
    s = _xtim._XdemSampler(text, 65536)
    D = s.num_detectors
    dets = np.frombuffer(int(sv).to_bytes((D + 7) // 8, "little"), np.uint8).reshape(1, -1)
    fo, fi = csr([T])
    ho, hi = csr([H])
    return s.score_lines(dets, fo, fi, ho, hi, corr)[0], s.correction(dets, ho, hi, "optimal")


def test_e4_minimal_example():
    """the review's minimal row: D0; L0..L2; L0, L2 lossy, L0 + L2 deterministic; E = Ê = [0, 2], d = 1. f_K = 0 scored
    0.0 there ("none" and V^dag 1/2). L0 alone is a fair coin (lossy), so no correction can score above 1/2; E4's c is
    a point of the support (here c = f_Ê = (L0, L2) = (0, 1)) and scores exactly 1/2"""
    x = xdem_export(E4_FUZZ)
    M = XdemModel.from_text(x)
    K = M.k_optimal([0, 2], 1)
    assert [j for j in range(3) if K["lossy"][j]] == [0, 2]
    assert M.compose([0, 2])[0] >> M.D == 0b100 and (K["frame"][0], K["frame"][2]) == (0, 1)
    assert M.pass_optimal([0, 2], [0, 2], 1) == 0.5
    opt, corr = one_row(x, [0, 2], [0, 2], 1, "optimal")
    none, _ = one_row(x, [0, 2], [0, 2], 1, "none")
    assert opt == 0.5 and none == 0.5 and opt >= none
    assert [int(v) for v in corr["frame"]] == K["frame"]


def test_e4_edge_rows_score_the_lossy_coin():
    """the slow fuzz's 10 CZ-edge rows (Ê = E; f_K = 0 scored 0.0 on 6, where V^dag scores 1/2 and "none" 1/2 on 3 and
    0 on 3): "optimal" == the oracle == 1/2 (L0, L2
    lossy with a determined parity: one coin) and >= "none" on every row; the export's frames == the oracle's"""
    x = xdem_export(E4_FUZZ)
    M = XdemModel.from_text(x)
    for E, d in E4_ROWS:
        opt, corr = one_row(x, E, E, d, "optimal")
        none, _ = one_row(x, E, E, d, "none")
        K = M.k_optimal(E, d)
        assert opt == M.pass_optimal(E, E, d) == 0.5, (E, d, opt)
        assert opt >= none, (E, d, opt, none)
        assert [int(v) for v in corr["frame"]] == K["frame"], (E, d)
        f0 = dict(K, frame=[0 if K["lossy"][j] else K["frame"][j] for j in range(M.O)])   # the old rule, f_K = 0
        old = M.pass_of(M.law(M.circuit(E) + _with_frame(M, K, f0["frame"])), d, 0)
        assert (old == 0.0) == ((tuple(E), d) in E4_WAS_ZERO), (E, d, old)


def _with_frame(M, K, frame):
    """K's circuit with its Z frame replaced by `frame` (per observable)"""
    import stim
    c = stim.Circuit()
    for ins in K["circuit"]:
        if ins.name != "Z":
            c.append(ins)
    zs = [M.D + j for j in range(M.O) if frame[j]]
    if zs:
        c.append("Z", zs)
    return c


def test_e4_hand_files():
    """E4_PAIR: L0 + L1 = d; at d = 1, c = (0, 1) (the support point nearest f_Ê = 0, observable 0 first): pass 1/2
    where "none" scores 0. E4_INDEP: independent lossy observables still score 1/2 each (1/4), with c = f_Ê."""
    M = XdemModel.from_text(E4_PAIR)
    for d, frame, none in ((0, [0, 0], 0.5), (1, [0, 1], 0.0)):
        opt, corr = one_row(E4_PAIR, [0], [0], d, "optimal")
        assert opt == M.pass_optimal([0], [0], d) == 0.5
        assert one_row(E4_PAIR, [0], [0], d, "none")[0] == M.pass_none([0], [0], d) == none
        assert M.k_optimal([0], d)["frame"] == frame and [int(v) for v in corr["frame"]] == frame
    M = XdemModel.from_text(E4_INDEP)
    for d in range(4):
        opt, corr = one_row(E4_INDEP, [0], [0], d, "optimal")
        assert opt == M.pass_optimal([0], [0], d) == 0.25
        assert M.k_optimal([0], d)["frame"] == [1, 0] and [int(v) for v in corr["frame"]] == [1, 0]
    for general in (False, True):                                       # both law paths
        from xtim import _xtim_testapi as ta
        dets = np.array([[1]], np.uint8)
        fo, fi = csr([[0]])
        assert ta.xdem_lines_fixed(E4_PAIR, fo, fi, dets, fo, fi, "optimal", 0, general)[0][0] == 0.5


# E4 fix round 1 (owner, I1): d outside law(Ê)'s detector support gives c = 0 on the lossy observables (SPEC §7 Q1).
# E4_INDEP plus a detector D2 that only line 1 flips: under Ê = [0], d with D2 = 1 is outside; f_Ê = L0 there.
E4_OUT = """xdem 1 detectors 3 observables 2 gauge 4 hidden 0 tc 2
axis Q0 img D0 L0 gauge g0
axis Q1 img D0 gauge g1
axis Q2 img D1 L1 gauge g2
axis Q3 img D1 gauge g3
error(0.3) L0 rot +Q0 +Q1 +Q2 +Q3
error(0.1) D2
"""


def test_e4_outside_support_predicts_zero():
    """T = [0, 1], Ê = [0], d with D2 = 1: d is outside law(Ê)'s support, so c = 0 on both lossy observables (not
    f_Ê = (1, 0)); inside (D2 = 0), c = f_Ê. C++ score / export / k_optimal == the oracle; lines_outside counts it"""
    from xtim import _xtim_testapi as ta
    M = XdemModel.from_text(E4_OUT)
    for d, frame in ((0b100, [0, 0]), (0b101, [0, 0]), (0b000, [1, 0]), (0b011, [1, 0])):
        K = M.k_optimal([0], d)
        assert K["lossy"] == [True, True] and K["frame"] == frame, (d, K["frame"])
        T = [0, 1] if d >> 2 & 1 else [0]
        opt, corr = one_row(E4_OUT, T, [0], d, "optimal")
        assert opt == M.pass_optimal(T, [0], d), (d, opt)
        assert [int(v) for v in corr["frame"]] == frame
        assert [r["frame"] for r in ta.xdem_k_optimal(E4_OUT, [0], d)] == frame
    s = _xtim._XdemSampler(E4_OUT, 65536)
    fo, fi = csr([[0, 1]])
    ho, hi = csr([[0]])
    s.score_lines(np.array([[0b100]], np.uint8), fo, fi, ho, hi, "optimal")
    assert s.counters()["lines_outside"] == 1


def optimal_not_below_none(name, shots, seed):
    """H = fired on a real circuit: "optimal" >= "none" on every shot (ERRATA E4 clarification: a lossy parity the
    hypothesis fixes commutes with every non-lossy rotated readout, so K's CZ edges add no coin); returns lossy shots"""
    text = xdem_export(circuit(name))
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, seed, 1, 0, True)
    opt = s.score_lines(dets, fo, fi, fo, fi, "optimal")
    none = s.score_lines(dets, fo, fi, fo, fi, "none")
    bad = np.flatnonzero(opt < none)
    assert bad.size == 0, (name, bad[:10], opt[bad[:10]], none[bad[:10]])
    return s.counters()["lines_lossy"]


def test_optimal_not_below_none_cultivation_d5():
    assert optimal_not_below_none("cultivation_d5", 20000, 11) > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bt27_p1e-3", "bt27_p1e-4"])
def test_optimal_not_below_none_bt27(name):
    if not present(name):
        pytest.skip("external circuit absent")
    assert optimal_not_below_none(name, 200000, 11) > 0
