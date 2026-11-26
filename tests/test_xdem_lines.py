"""score(lines=) corrections "none" and "r2b" (SPEC §4.4, §4.5, §6 T6) against the model oracle, per shot, exactly."""
import random

import numpy as np
import pytest
from xtim.errors import XtimXdemError

from xtim import _xtim
from xtim import _xtim_testapi as ta

from _xdem_circuits import circuit, present, xdem_export
from _xdem_model import XdemModel, cancel
from _xdem_synth import random_text

kXdemMutTcMinus1 = 5                                                    # xdem_sampler.hpp's lever


def csr(lists):
    off = np.zeros(len(lists) + 1, np.int64)
    off[1:] = np.cumsum([len(l) for l in lists])
    ids = np.array([i for l in lists for i in l], np.int32)
    return off, ids


def run(text, shots, seed, decoder, correction):
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, seed, 1, 0, True)
    H = [decoder(list(fi[fo[k]:fo[k + 1]]), k) for k in range(shots)]
    ho, hi = csr(H)
    return s, dets, fo, fi, H, s.score_lines(dets, fo, fi, ho, hi, correction)


def oracle_check(text, shots, seed, decoder, correction):
    M = XdemModel.from_text(text)
    s, dets, fo, fi, H, got = run(text, shots, seed, decoder, correction)
    d = np.unpackbits(dets, axis=1, bitorder="little", count=M.D)
    for k in range(shots):
        sv = int(sum(int(b) << j for j, b in enumerate(d[k])))
        T = list(fi[fo[k]:fo[k + 1]])
        want = {"none": M.pass_none, "r2b": M.pass_r2b}[correction](T, H[k], sv)
        assert got[k] == want, (k, T, H[k], sv)


def r2b_ok(text):
    return ta.xdem_facts(text)["r2b_available"]


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("correction", ["none", "r2b"])
def test_synthetic_per_shot_equals_the_oracle(seed, correction):
    rng = random.Random(seed)
    text = random_text(rng, D=rng.randint(1, 4), O=rng.randint(1, 2), r=rng.randint(1, 3), m=rng.randint(0, 2),
                       A=rng.randint(1, 4), L=6, tc=4)
    if correction == "r2b" and not r2b_ok(text):
        with pytest.raises(XtimXdemError, match="not in canonical single-stage form at t_c"):
            run(text, 10, seed, lambda T, k: T, correction)
        return
    L = ta.xdem_parse(text)["L"]
    dec = lambda T, k: T if k % 3 == 0 else sorted(random.Random(k).sample(range(L), random.randint(0, 2)))  # noqa
    oracle_check(text, 300, seed, dec, correction)


@pytest.mark.parametrize("correction", ["none", "r2b"])
def test_hypothesis_order_and_repeats(correction):                      # Review Focus 2
    text = xdem_export(circuit("rep3T_pre"))
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(2000, 3, 1, 0, True)
    L = s.num_lines
    rng = random.Random(4)
    H = [sorted(rng.sample(range(L), 2)) for _ in range(2000)]
    messy = [[h[1], h[0], h[1], h[1]] for h in H]                        # [b, a, b, b] = [a, b]
    canon = [cancel(m) for m in messy]
    a = s.score_lines(dets, fo, fi, *csr(messy), correction)
    b = s.score_lines(dets, fo, fi, *csr(canon), correction)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("correction", ["none", "r2b"])
def test_hypothesis_order_and_repeats_synthetic(seed, correction):   # Review Focus 2, on non-commuting lines
    """H in reverse file order with a repeated pair ([c, b, a, a, a] = [a, b, c]) scores as its canonical list and as
    the oracle; synthetic lines rarely commute, so composing in the given order would show"""
    rng = random.Random(1000 + seed)
    text = random_text(rng, D=rng.randint(1, 3), O=rng.randint(1, 2), r=rng.randint(1, 3), m=rng.randint(0, 2),
                       A=rng.randint(2, 4), L=6, tc=4)
    if correction == "r2b" and not r2b_ok(text):
        pytest.skip("r2b unavailable on this model (the refusal is pinned in test_synthetic_per_shot_equals_the_oracle)")
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    n = 300
    dets, obs, cs, fo, fi = s.sample(n, seed, 1, 0, True)
    canon = [sorted(rng.sample(range(len(M.lines)), rng.randint(2, 3))) for _ in range(n)]
    messy = [h[::-1] + [h[0], h[0]] for h in canon]
    a = s.score_lines(dets, fo, fi, *csr(messy), correction)
    assert np.array_equal(a, s.score_lines(dets, fo, fi, *csr(canon), correction))
    d = np.unpackbits(dets, axis=1, bitorder="little", count=M.D)
    f = {"none": M.pass_none, "r2b": M.pass_r2b}[correction]
    for k in range(n):
        assert a[k] == f(list(fi[fo[k]:fo[k + 1]]), canon[k], int(sum(int(b) << j for j, b in enumerate(d[k])))), k


def test_reductions():                                                   # T6: "none" == law-aware flips; empty == flips 0
    text = xdem_export(circuit("ccz_toy_obs_p5"))
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(3000, 5, 1, 0, True)
    empty = csr([[] for _ in range(3000)])
    assert np.array_equal(s.score_lines(dets, fo, fi, *empty, "none"),
                          s.score(obs, cs, np.zeros((3000, s.num_observables), np.uint8)))
    M = XdemModel.from_text(text)
    d = np.unpackbits(dets, axis=1, bitorder="little", count=M.D)
    H = [list(fi[fo[k]:fo[k + 1]]) for k in range(3000)]
    flips = np.zeros((3000, M.O), np.uint8)
    for k in range(3000):
        sv = int(sum(int(b) << j for j, b in enumerate(d[k])))
        f, _, _ = M.compose(H[k])
        c = M.definite(M.law(M.circuit(H[k])), sv, f)
        flips[k] = [(c >> j) & 1 for j in range(M.O)]
    assert np.array_equal(s.score_lines(dets, fo, fi, *csr(H), "none"), s.score(obs, cs, flips))


def mle2_table(ps, syn):
    """the most likely set of <= 2 lines per detector pattern (port of loop/mlib.py::mle2_table); ps: line
    probabilities, syn: each line's detector bits as an int"""
    import itertools
    import math
    lw = [math.log(p / (1 - p)) for p in ps]
    table = {0: (0.0, [])}
    for k in (1, 2):
        for ids in itertools.combinations(range(len(ps)), k):
            dd = 0
            for i in ids:
                dd ^= syn[i]
            w = sum(lw[i] for i in ids)
            if dd not in table or w > table[dd][0]:
                table[dd] = (w, list(ids))
    return table


def det_int(row, D):
    return int.from_bytes(np.asarray(row, np.uint8).tobytes(), "little") & ((1 << D) - 1)


def mutant_bad(name, mut, wrong=False, shots=4000):
    """shots where the mutant's r2b pass differs from the oracle's: H = the fired lines, or (wrong) random 0-3 lines"""
    text = xdem_export(circuit(name))
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(shots, 9, 1, 0, True)
    T = [list(fi[fo[k]:fo[k + 1]]) for k in range(shots)]
    rng = random.Random(11)
    H = [sorted(rng.sample(range(len(M.lines)), rng.randint(0, 3))) for _ in range(shots)] if wrong else T
    ho, hi = csr(H)
    d = np.unpackbits(dets, axis=1, bitorder="little", count=M.D)
    got, _, _, _ = ta.xdem_lines_fixed(text, fo, fi, dets, ho, hi, "r2b", mut)
    return sum(got[k] != M.pass_r2b(T[k], H[k], int(sum(int(b) << j for j, b in enumerate(d[k])))) for k in range(shots))


@pytest.mark.parametrize("mut", [1, 2, 3, 4, 5])
def test_mutations_fail(mut):
    """Test-of-test: each lever fails. A lever that changes nothing on rep3T_pre_p1e-2 with H = E is re-run on
    msc_d3_canon (H = E) and on rep3T_pre_p1e-2 with wrong hypotheses (random 0-3 lines), and must fail on one. Found:
    kXdemMutTcMinus1 is law-invisible on rep3T_pre_p1e-2 (line tc - 1 has no late exponent, no tau, no rotation, so
    eta^> and K's place are unchanged); kXdemMutNoCocycle changes K by a Pauli (a sign flip of an odd exponent is a
    factor i Z_a), which pass ignores whenever c is the chosen law's definite value, as it always is with H = E (d lies
    in the law's support): it shows only when c falls back to f_h|_O, i.e. under a wrong hypothesis."""
    if mutant_bad("rep3T_pre_p1e-2", mut) > 0:
        return
    if mut == kXdemMutTcMinus1:
        M = XdemModel.from_text(xdem_export(circuit("rep3T_pre_p1e-2")))
        _, _, tau, rot, late = M.lines[M.tc - 1]
        assert not tau and not rot and not late                          # law-invisible there
    legs = [mutant_bad("rep3T_pre_p1e-2", mut, wrong=True)]
    if present("msc_d3_canon"):
        legs.append(mutant_bad("msc_d3_canon", mut))
    elif legs[0] == 0:
        pytest.skip("msc_d3_canon absent and the lever is invisible on rep3T_pre_p1e-2")
    print(f"\n[mutant {mut}] rep3T_pre_p1e-2 H=E: 0; wrong hypotheses / msc_d3_canon H=E: {legs}")
    assert max(legs) > 0


def test_production_equals_the_oracle_under_wrong_hypotheses():
    assert mutant_bad("rep3T_pre_p1e-2", 0, wrong=True) == 0



def test_binding_refusals_and_lines_csr():
    """correction names (SPEC §4.6), CSR refusals, and lines_csr's CSR (the dc name is gone, preflight S6)"""
    s = _xtim._XdemSampler(xdem_export(circuit("rep3T_pre")), 65536)
    dets, obs, cs, fo, fi = s.sample(50, 1, 1, 0, True)
    with pytest.raises(ValueError, match="correction"):
        s.score_lines(dets, fo, fi, fo, fi, "R2B")
    bo, bi = csr([[s.num_lines]] + [[] for _ in range(49)])           # a fixed id one past the last line
    for corr in ("none", "r2b", "optimal"):
        with pytest.raises(IndexError, match="outside"):
            s.score_lines(dets, fo, fi, bo, bi, corr)
        with pytest.raises(IndexError, match="outside"):
            s.score_lines(dets, bo, bi, fo, fi, corr)
    with pytest.raises(ValueError, match="offsets"):
        s.score_lines(dets, fo, fi, fo[::-1].copy(), fi, "none")
    lists = [[2, 0], [], [1, 1]]
    a = _xtim.lines_csr(lists, 3, s.num_lines, "w")
    assert not hasattr(_xtim, "dc_lines_csr")
    assert list(a[0]) == [0, 2, 2, 4] and list(a[1]) == [2, 0, 1, 1]


@pytest.mark.parametrize("name", ["rep3T_pre_p1e-2", "toy2obs"])
def test_actions_c_and_outside_equal_the_oracle(name):
    """per shot under wrong hypotheses: R2b's action (none / applied / withheld) and c equal the oracle's k_r2b, and
    lines_outside counts the shots whose d is outside law(h)'s detector support (SPEC §4.5)"""
    text = xdem_export(circuit(name))
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    n = 1500
    dets, obs, cs, fo, fi = s.sample(n, 21, 1, 0, True)
    rng = random.Random(5)
    H = [sorted(rng.sample(range(len(M.lines)), rng.randint(0, 3))) for _ in range(n)]
    ho, hi = csr(H)
    d = np.unpackbits(dets, axis=1, bitorder="little", count=M.D)
    _, act, cb, _ = ta.xdem_lines_fixed(text, fo, fi, dets, ho, hi, "r2b", 0)
    s.score_lines(dets, fo, fi, ho, hi, "r2b")
    outside = 0
    for k in range(n):
        sv = int(sum(int(b) << j for j, b in enumerate(d[k])))
        K = M.k_r2b(H[k], sv)
        want = 0 if not K["eta"] else (1 if K["applied"] else 2)
        assert act[k] == want, (k, H[k])
        assert int(sum(int(b) << j for j, b in enumerate(cb[k]))) == K["c"], (k, H[k])
        outside += not M.given(M.law(M.circuit(H[k])), sv)[3]
    assert outside > 0
    assert s.counters()["lines_outside"] == outside


def composite_in_given_order(M, ids):
    """the cocycle of SPEC §1.2 over ids AS GIVEN (no sort, no cancellation): the composite mutant A' would use"""
    f, tau, eta = 0, 0, {}
    for i in ids:
        _, fi, ti, rot, late = M.lines[i]
        for a, e in M.total(rot, late).items():
            eta[a] = (eta.get(a, 0) + (-1) ** (tau >> a & 1) * e) % 4
        f ^= fi
        tau ^= ti
    return f, tau, {a: e for a, e in eta.items() if e}


def optimal_given_order(M, T, i, j, sv):
    """the oracle's optimal score with h = [j, i] composed in the GIVEN order (what mutant A' computes): K from the
    model with lines i and j exchanged (its canonical [i, j] composes M's line j, then line i), applied after M's T"""
    lines = list(M.lines)
    lines[i], lines[j] = lines[j], lines[i]
    K = XdemModel(M.D, M.O, M.r, M.m, M.tc, M.axes, lines).k_optimal([i, j], sv)
    return M.pass_of(M.law(M.circuit(T) + K["circuit"]), sv, 0)


# Circuits on which the hypothesis order can change the optimal score. ccz_toy_obs_p5 is score-invisible: exhaustively
# (its 9 non-commuting pairs x every s x every T of <= 2 lines: 468 configurations) the given and the canonical order
# score the same, so its legs check the canonical scores only (and that no shot discriminates).
ORDER_VISIBLE = {"rep3T_pre_p1e-2", "msc_d3_canon"}


@pytest.mark.parametrize("name", ["ccz_toy_obs_p5", "rep3T_pre_p1e-2", "msc_d3_canon"])
@pytest.mark.parametrize("correction", ["none", "r2b", "optimal"])
def test_hypothesis_order_on_real_circuits(name, correction):         # Review Focus 2 (Task 7 review: mutant A')
    """H = two non-commuting extended lines given in REVERSE file order (sometimes with a third line repeated twice),
    chosen per shot so that the oracle's OPTIMAL score differs between the canonical and the given order
    (optimal_given_order): on an ORDER_VISIBLE circuit the optimal leg catches composing in the given order by
    construction (the in-repo rep3T_pre_p1e-2 included). Every shot scores as the oracle on the canonical list."""
    if not present(name):
        pytest.skip("external circuit absent")
    text = xdem_export(circuit(name))
    if correction == "r2b" and not r2b_ok(text):
        pytest.skip("r2b unavailable on this circuit")
    M = XdemModel.from_text(text)
    L = len(M.lines)
    ext = [i for i in range(L) if any(M.lines[i][2:])]
    rng = random.Random(8)
    pool = [(i, j) for i in ext for j in ext if i < j]
    rng.shuffle(pool)
    nc = [(i, j) for i, j in pool[:4000]
          if composite_in_given_order(M, [i, j]) != composite_in_given_order(M, [j, i])][:200]
    assert len(nc) >= 3
    n = 400 if name == "msc_d3_canon" else 1500
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, fo, fi = s.sample(n, 31, 1, 0, True)
    d = np.unpackbits(dets, axis=1, bitorder="little", count=M.D)
    svs = [int(sum(int(b) << j for j, b in enumerate(d[k]))) for k in range(n)]
    messy, canon, picked = [], [], 0
    for k in range(n):
        T = list(fi[fo[k]:fo[k + 1]])
        extra = [rng.randrange(L)] * 2 if k % 3 == 0 else []
        for tries in range(2 if name == "msc_d3_canon" else 6):
            i, j = nc[rng.randrange(len(nc))]
            h = [j] + extra + [i]
            if extra and extra[0] in (i, j):
                continue
            if M.pass_optimal(T, [i, j], svs[k]) != optimal_given_order(M, T, i, j, svs[k]):
                picked += 1
                break
        messy.append(h)
        canon.append(cancel(h))
    if name in ORDER_VISIBLE:
        assert picked >= 5                                               # shots where the order shows in the score
    else:
        assert picked == 0                                               # score-invisible (the exhaustive check above)
    got = s.score_lines(dets, fo, fi, *csr(messy), correction)
    f = {"none": M.pass_none, "r2b": M.pass_r2b, "optimal": M.pass_optimal}[correction]
    for k in range(n):
        assert got[k] == f(list(fi[fo[k]:fo[k + 1]]), canon[k], svs[k]), (k, messy[k])
