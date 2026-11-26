"""XdemSampler's fast path at any gauge count r and span dimension rho (XFM): coordinates of W = ceil(max(r, rho) / 64)
words. W = 1 is the one-word path (its own coin stream, unchanged). W >= 2 reads the general path's coin stream
(ceil(k / 64) words over the canonical coin basis, xdem_sampler.hpp), so on such a file every output equals the
force_general oracle's byte for byte: sample (dets, obs, status ids, C table) at threads 1/3/4, sample_ps, status_of,
score_lines and correction."""
import random
from pathlib import Path

import numpy as np
import pytest

from xtim import _xtim
from xtim import _xtim_testapi as ta

from _xdem_synth import random_text

ROOT = Path(__file__).resolve().parents[1]

# (r, n = D + O, A, L): rho = dim span(img) = min(r, n) for these random models (A >= r random kappa)
CASES = [(64, 64, 80, 160), (65, 64, 80, 160), (65, 65, 80, 160), (70, 63, 90, 180), (127, 127, 140, 260),
         (128, 128, 140, 260), (129, 129, 140, 260), (129, 66, 140, 260), (200, 190, 215, 380)]


def _text(r, n, A, L, seed=0):
    O = 5
    return random_text(random.Random(9000 + 7 * r + n + seed), D=n - O, O=O, r=r, m=2, A=A, L=L, tc=L // 2)


def _hi_p(text, seed):
    """the same file with every probability redrawn in [0.15, 0.45]: many odd axes per shot, coin rank > 64"""
    import re
    rng = random.Random(seed)
    return re.sub(r"^error\(([^)]*)\)", lambda m: f"error({rng.uniform(0.15, 0.45):.17g})", text, flags=re.M)


def _eq_runs(a, b):
    (ra, ta_, ka), (rb, tb, kb) = a, b
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)
    assert ta_ == tb
    return ka, kb


@pytest.mark.parametrize("case", CASES, ids=[f"r{c[0]}_n{c[1]}" for c in CASES])
def test_fast_path_any_size_equals_general(case):
    r, n, A, L = case
    for text in (_text(r, n, A, L), _hi_p(_text(r, n, A, L), r)):
        s = _xtim._XdemSampler(text, 65536)
        assert s.fast_path and s.num_gauge == r
        if max(r, n) <= 64:
            continue   # W = 1: the one-word stream (pinned by the goldens and the byte oracle, not by the oracle path)
        calls = [(5000, 11), (3000, 12)]
        ref = ta.xdem_sample(text, calls, 1, True)
        assert not ref[2]["fast_path"]
        for th in (1, 3, 4):
            got = ta.xdem_sample(text, calls, th, False)
            k, _ = _eq_runs(got, ref)
            assert k["fast_path"] and k["cache_misses"] == 0


@pytest.mark.parametrize("case", CASES[1:], ids=[f"r{c[0]}_n{c[1]}" for c in CASES[1:]])
def test_postselect_any_size_equals_general(case):
    r, n, A, L = case
    D = n - 5
    rng = np.random.default_rng(r + n)
    masks = [np.zeros(D, np.uint8)] + [(rng.random(D) < q).astype(np.uint8) for q in (0.03, 0.08, 0.3)]
    survivors = 0
    for text in (_text(r, n, A, L), _hi_p(_text(r, n, A, L), 3 * r)):
        for flags in masks:
            for th in (1, 4):
                a = ta.xdem_sample_ps(text, 4000, 21, th, list(flags), False)
                b = ta.xdem_sample_ps(text, 4000, 21, th, list(flags), True)
                assert a[2] and not b[2]
                for x, y in zip(list(a[0]) + list(a[1]), list(b[0]) + list(b[1])):   # (survivors, early), plain
                    assert np.array_equal(np.asarray(x), np.asarray(y))
                assert a[3] == b[3] and a[4] == b[4]
                survivors += len(a[0][2]) if flags.any() else 0
    assert survivors > 0   # a nonempty mask keeps some shots: the survivors' bytes are compared too


def test_coin_rank_above_64_is_exercised():
    """the hi-p r = 129 / 200 files reach laws with > 64 coins (two coin words per shot); D = 61 < rank: dim C > 0"""
    big = dimc = 0
    for r, n, A, L in ((129, 129, 140, 260), (200, 190, 215, 380), (129, 66, 140, 260)):
        text = _hi_p(_text(r, n, A, L), r)
        rng = random.Random(r)
        for k in range(12):
            ids = rng.sample(range(L), L // 2 if k < 6 else 3)
            st, st_g = ta.xdem_status_of(text, ids, False), ta.xdem_status_of(text, ids, True)
            assert st["fast_path"] and not st_g["fast_path"]
            assert st == {**st_g, "fast_path": True}   # the same canonical offset, coins (order) and C
            big = max(big, len(st["coins"]))
            dimc = max(dimc, st["dim"])
    assert big > 64 and dimc > 0


@pytest.mark.parametrize("case", [(300, 280, 310, 300), (400, 300, 410, 300), (530, 300, 560, 300)],
                         ids=["cw5_as_6", "cw7_as_8", "cw9_run_time"])
def test_wide_instances(case):
    """r = 300: 5 words padded to the 6-word instance; r = 400: 7 words padded to the 8-word instance; r = 530: 9
    words, the run-time instance: the oracle's bytes"""
    text = _text(*case)
    calls = [(1500, 7)]
    ref = ta.xdem_sample(text, calls, 1, True)
    got = ta.xdem_sample(text, calls, 3, False)
    k, _ = _eq_runs(got, ref)
    assert k["fast_path"]


def _hyp(off, ids, L, seed):
    """a non-trivial hypothesis: each fired line kept with probability 1/2, plus two random lines"""
    rng = np.random.default_rng(seed)
    out, o = [], [0]
    for s in range(len(off) - 1):
        f = ids[off[s]:off[s + 1]]
        out += list(f[rng.random(len(f)) < 0.5]) + list(rng.integers(0, L, 2))
        o.append(len(out))
    return np.array(o, np.int64), np.array(out, np.int32)


SCORE_CASES = [(63, 63, 80, 160), (64, 64, 80, 160), (65, 65, 80, 160), (70, 63, 90, 180), (127, 127, 140, 260),
               (128, 128, 140, 260), (129, 129, 140, 260), (129, 66, 140, 260), (300, 280, 310, 300),
               (400, 300, 410, 300)]   # every instance: CW 1 (63, 64), 2, 3, 6 (300), 8 (400)


@pytest.mark.parametrize("case", SCORE_CASES, ids=[f"r{c[0]}_n{c[1]}" for c in SCORE_CASES])
@pytest.mark.parametrize("late", [True, False], ids=["late", "nolate"])
def test_scoring_any_size_equals_general(case, late):
    """XFM fix round 1: score_lines ("none", "r2b" when available, "optimal"), the correction exports and law_of run
    the fast laws at every size (multi-word law readers; the optimal correction's beta' law in cw words) and equal the
    force_general oracle: pass, action, c, the lines_* counters, every export member; law_of exactly (canonical) on a
    multi-word file"""
    r, n, A, L = case
    text = random_text(random.Random(9000 + 7 * r + n), D=n - 5, O=5, r=r, m=2, A=A, L=L, tc=L // 2, late_ok=late)
    s = _xtim._XdemSampler(text, 65536)
    d, o, cs, off, ids = s.sample(1200 if r <= 300 else 400, 31, 1, 0, True)
    ho, hi = _hyp(off, ids, s.num_lines, r)
    corrs = ["none", "optimal"] + (["r2b"] if s.r2b_available[0] else [])
    for corr in corrs:
        for h_o, h_i in ((ho, hi), (off, ids)):
            a = ta.xdem_lines_fixed(text, off, ids, d, h_o, h_i, corr, 0, False)
            b = ta.xdem_lines_fixed(text, off, ids, d, h_o, h_i, corr, 0, True)
            assert np.array_equal(a[0], b[0]) and a[0].any()
            if corr != "optimal":   # (the optimal leg fills neither action nor c)
                assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
            skip = ("fast_path", "kfast_laws")
            assert {k: v for k, v in a[3].items() if k not in skip} == {k: v for k, v in b[3].items() if k not in skip}
            if corr == "optimal":
                assert a[3]["kfast_laws"] > 0 and b[3]["kfast_laws"] == 0   # the fast law of [T, K] at any size
        if corr != "none":
            ea = ta.xdem_correction(text, d, ho, hi, corr, False)
            eb = ta.xdem_correction(text, d, ho, hi, corr, True)
            assert ea == eb
    if s.fast_path and max(r, n) > 64:
        for k in range(0, len(off) - 1, 40):
            for x in (list(ids[off[k]:off[k + 1]]), list(hi[ho[k]:ho[k + 1]])):
                assert ta.xdem_law_of(text, x, False) == {**ta.xdem_law_of(text, x, True), "fast_path": True}


def test_bundled_xdem_files_take_the_fast_path():
    """every xdem file shipped in the repo (examples/xdem, tests/data/xdem) is on the fast path; one with W >= 2
    equals the oracle byte for byte"""
    paths = sorted((ROOT / "examples" / "xdem").glob("*.xdem"))
    paths += sorted((ROOT / "tests" / "data" / "xdem").rglob("*.xdem"))
    assert paths
    for p in paths:
        text = p.read_text()
        s = _xtim._XdemSampler(text, 65536)
        assert s.fast_path, p.name
        if s.num_gauge > 64:
            ref = ta.xdem_sample(text, [(3000, 5)], 1, True)
            _eq_runs(ta.xdem_sample(text, [(3000, 5)], 3, False), ref)
