"""Integer geometric skips (xtim.gpu.tables): exact thresholds and the reference skip."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from xtim.gpu.tables import geo_skip_py, geo_thresholds

TWO64 = 1 << 64


@pytest.mark.parametrize("p", [0.5, 1e-3, 1e-6, 1e-12, 0.999])
def test_skip_law_exact(p):
    slots = 1024
    J, thr = geo_thresholds(p, slots)
    assert J == 10 and thr.dtype == np.uint64 and len(thr) == J + 1
    q = 1 - Fraction(p)
    bit = [Fraction(int(thr[1 + j]), TWO64) for j in range(J)]
    p_end = Fraction(int(thr[0]), TWO64)
    # A threshold is floor()ed to a multiple of 2**-64, so the comparison is relative with
    # an absolute floor of 2**-62 for probabilities that are themselves below that grain.
    tol_rel, tol_abs = Fraction(1, 2**60), Fraction(1, 2**62)

    def close(got, want):
        return abs(got - want) <= tol_rel * want + tol_abs

    assert close(p_end, q ** (2**J))
    for k in range(64):
        prob = 1 - p_end
        for j in range(J):
            prob *= bit[j] if (k >> j) & 1 else 1 - bit[j]
        assert close(prob, Fraction(p) * q**k), k


def test_thresholds_are_exact_floors():
    p = 0.01
    J, thr = geo_thresholds(p, 1024)
    q = 1 - Fraction(p)
    for j in range(J):
        t = q ** (2**j) / (1 + q ** (2**j)) * TWO64
        assert abs(int(thr[1 + j]) - int(t)) <= 1
    assert abs(int(thr[0]) - int(q**1024 * TWO64)) <= 1


def test_slots_not_power_of_two():
    J, thr = geo_thresholds(1e-3, 3 * 1024)
    assert J == 12 and len(thr) == 13


def test_small_slots_have_one_bit():
    J, thr = geo_thresholds(0.3, 1)
    assert J == 1 and len(thr) == 2


def test_thresholds_clamp_to_uint64():
    # q rounds to exactly 1, so q**(2**J) * 2**64 would be 2**64 and must clamp.
    _, thr = geo_thresholds(1e-300, 1024)
    assert int(thr[0]) == TWO64 - 1


def test_rejects_p_out_of_range():
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            geo_thresholds(p, 1024)


LOW56 = (1 << 56) - 1


def _thr(*pairs):
    """thresholds from (top byte, low 56 bits) pairs"""
    return [(top << 56) | low for top, low in pairs]


def _word(*bytes_):
    """a draw whose byte i is bytes_[i] (the rest 0xFF)"""
    x = (1 << 64) - 1
    for i, b in enumerate(bytes_):
        x = (x & ~(0xFF << (8 * i))) | (b << (8 * i))
    return x


def _skip(thr, words, ties=()):
    """(skip, tie draws consumed): ``ties[q]`` lists word q's tie draws in order"""
    used = 0

    def tie(q, j):
        nonlocal used
        used += 1
        return ties[q][j]
    out = geo_skip_py(thr, words, tie)
    return out, used


def test_geo_skip_py_semantics():
    thr = _thr((100, 5), (50, 5), (60, 5), (70, 5))
    # one word for all four tests (no tie): byte i against the top byte of thr[i]
    assert _skip(thr, [_word(100, 51, 61, 71)], [[0]]) == (None, 1)   # test 0 ties: its tie draw 0 < 5, the end
    assert _skip(thr, [_word(101, 51, 61, 71)]) == (0, 0)
    assert _skip(thr, [_word(99, 0, 0, 0)]) == (None, 0)
    assert _skip(thr, [_word(101, 49, 61, 69)]) == (1 + 4, 0)
    assert _skip(thr, [_word(101, 51, 59, 71)]) == (2, 0)
    # ties, ascending: test 1 then test 3, each against the low 56 bits (y >> 8 < 5)
    assert _skip(thr, [_word(101, 50, 61, 70)], [[4 << 8, 5 << 8]]) == (1, 2)
    assert _skip(thr, [_word(101, 50, 61, 70)], [[5 << 8, (4 << 8) | 0xFF]]) == (4, 2)
    # the end test's tie lost: the skip read from the same word
    assert _skip(thr, [_word(100, 49, 61, 71)], [[5 << 8]]) == (1, 1)
    # every tie of the pass is decided, whatever the end test says
    assert _skip(thr, [_word(99, 51, 60, 71)], [[0]]) == (None, 1)
    # bytes past the last test are ignored
    assert _skip(_thr((0, 0), (0, 0)), [_word(0xFF, 0xFF, 0, 0, 0, 0, 0, 0)]) == (0, 0)


def test_geo_skip_py_later_words():
    """tests 8 q + i read byte i of word q; bit j of the skip is test 1 + j; a word per 8 tests; a tie's draw is
    indexed by its word and its rank within that word"""
    nt = 20   # J = 19: three words
    thr = [0] + [(128 << 56)] * (nt - 1)   # end test never true; every bit test: byte < 128
    for j in range(nt - 1):
        i = 1 + j
        words = [_word(*[0xFF] * 8) for _ in range(3)]
        words[i // 8] = _word(*[(0 if k == i % 8 else 0xFF) for k in range(8)])
        assert _skip(thr, words) == (1 << j, 0), j
    assert _skip([1 << 63] + thr[1:], [_word(0), _word(0xFF), _word(0xFF)]) == (None, 0)
    # ties in words 1 and 2: tie(q, j) is called with the word q and the tie's rank j within it
    seen = []
    tw = [_word(0xFF), _word(0xFF, 128, 0xFF, 128), _word(128)]

    def tie(q, j):
        seen.append((q, j))
        return 0
    assert geo_skip_py([0] + [(128 << 56) | 1] * (nt - 1), tw, tie) == (1 << 8) | (1 << 10) | (1 << 15)
    assert seen == [(1, 0), (1, 1), (2, 0)]


def _prob(thr, i, nt):
    """test i's probability under uniform draws, read off the mirror: the 2^64 values of (its byte b, the top 56 bits
    v of its tie draw) counted as 2^56 per byte value decided true without a tie, plus, for the byte that ties, the
    number of v decided true (found by bisection: the tie decision is monotone in v). Every other test's byte is
    kept off its threshold's top byte (no tie), the end test false when i != 0."""
    q, k = divmod(i, 8)
    top = [int(x) >> 56 for x in thr]
    nwords = -(-nt // 8)

    def decide(b, v):
        words, ties = [], []
        for w in range(nwords):
            word = [top[8 * w + m] ^ 0x80 if 8 * w + m < nt else 0 for m in range(8)]
            wt = []
            if w == 0 and i != 0:   # the end test false: its byte above the top byte, or a tie lost
                word[0] = 0xFF
                if top[0] == 0xFF:
                    wt.append(LOW56 << 8)
            if w == q:
                word[k] = b
                if b == top[i]:
                    wt.append(v << 8)
            words.append(_word(*word))
            ties.append(wt)
        out, _ = _skip(thr, words, ties)
        if i == 0:
            return out is None
        assert out is not None
        return (out >> (i - 1)) & 1 == 1

    count = sum(1 for b in range(256) if b != top[i] and decide(b, 0))
    assert all(decide(b, 0) == decide(b, LOW56) for b in range(256) if b != top[i])
    lo, hi = 0, 1 << 56   # the first v decided false
    while lo < hi:
        mid = (lo + hi) // 2
        if decide(top[i], mid):
            lo = mid + 1
        else:
            hi = mid
    assert lo == 0 or decide(top[i], 0)
    return count * (1 << 56) + lo


@pytest.mark.parametrize("p", [0.5, 1e-3, 1e-12])
def test_each_test_probability_is_exactly_its_threshold(p):
    """the exact law of the shared-draw skip: test i is true for exactly thr[i] of the 2^64 equally likely values
    of (its byte, its tie draw's top 56 bits), ties included"""
    J, thr = geo_thresholds(p, 1 << 17)
    thr = [int(x) for x in thr]
    for i in range(J + 1):
        assert _prob(thr, i, J + 1) == thr[i], i


def test_edge_thresholds_exact():
    """top bytes 0 and 0xFF, low bits 0 and all ones, 0 and 2^64 - 1, across a word boundary"""
    vals = [0, 1, LOW56, 1 << 56, (1 << 56) + 1, (0xFF << 56), (0xFF << 56) | LOW56, (0x7F << 56) | 12345,
            (0x80 << 56) - 1]
    thr = [1] + vals + vals[:3]
    for i in range(len(thr)):
        assert _prob(thr, i, len(thr)) == thr[i], i


def test_sampled_skips_chi_square():
    p, n, kmax = 0.01, 2_000_000, 400
    J, thr = geo_thresholds(p, 1024)
    rng = np.random.default_rng(12345)
    counts = np.zeros(kmax + 2)  # bins 0..kmax, then the tail (including chain ends)
    batch = 100_000
    for _ in range(n // batch):
        raw = iter(rng.integers(0, 2**64, size=batch * (J + 1), dtype=np.uint64, endpoint=False).tolist())
        for _ in range(batch):
            words = [next(raw) for _ in range(-(-(J + 1) // 8))]
            s = geo_skip_py(thr, words, lambda q, j: next(raw))
            counts[kmax + 1 if s is None or s > kmax else s] += 1
    q = 1 - p
    probs = np.array([p * q**k for k in range(kmax + 1)] + [q ** (kmax + 1)])
    expected = probs * n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    df = kmax + 1
    # Wilson-Hilferty normal approximation; p-value > 1e-6 means z < 4.753.
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    assert z < 4.753, (chi2, df, z)


# ------------------------------------------------------------------------------------------------ the kernel's walk

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
# a file of many chains with extended lines (rm15rep), a conventional DEM, and p = 1, tiny and large p in one file
_MIXED = "\n".join(["error(1) D0", "error(1e-12) D1", "error(0.3) D2", "error(0.3) D3", "error(0.001) D4",
                    "error(0.05) D5 L0"]) + "\n"
WALK_CASES = [("examples/xdem/rm15rep.xdem", 64, 9 * 64 - 5), ("tests/data/xdem/synth/surface_d5r5.dem", 64, 300),
              ("examples/xdem/rm15rep.xdem", 48, 7 * 48 - 1), (None, 64, 40 * 64), (None, 24, 40 * 24 - 3)]


def _flips(t, m):
    x = 0
    for e in range(int(t["mem_beg"][m]), int(t["mem_beg"][m] + t["mem_cnt"][m])):
        x ^= int(t["ent_mask"][e]) << (64 * int(t["ent_word"][e]))
    return x


def _launchers():
    out = [("ref", None)]
    try:
        from xtim.gpu.metal import MetalLauncher
        MetalLauncher()
        out.append(("metal", MetalLauncher))
    except Exception:   # no Metal here: the reference alone
        pass
    return out


def _tables(case):
    import xtim
    path, B, n = WALK_CASES[case]
    text = (ROOT / path).read_text() if path else _MIXED
    return xtim.XdemSampler(text)._s.device_tables(), path, B, n


@pytest.mark.parametrize("case", range(len(WALK_CASES)))
@pytest.mark.parametrize("fires", [None, 0.5, 64.0])
def test_walk_plan_partitions_every_chain(case, fires):
    """every chain's member x shot slots of a block are cut into consecutive segments, each walked with the exact
    thresholds of its own length (geo_thresholds(p, L): its J + 1 tests); every segment in exactly one item"""
    from xtim.gpu import tables as gt

    t, _, B, _ = _tables(case)
    plan = gt.walk_plan(t, B, fires=fires)
    off, cp = np.asarray(t["chain_off"], np.int64), t["chain_p"]
    by_m0 = {}
    for row, nt, m0, t0, L in plan.seg.tolist():
        by_m0.setdefault(m0, []).append((t0, L, row, nt))
    assert int(plan.item_beg[0]) == 0 and int(plan.item_beg[-1]) == len(plan.seg)
    assert all(int(a) < int(b) for a, b in zip(plan.item_beg[:-1], plan.item_beg[1:]))
    seen = set()
    for c in range(len(cp)):
        m0, N, p = int(off[c]), int(off[c + 1] - off[c]) * B, float(cp[c])
        if N == 0 or p == 0.0:
            assert m0 not in by_m0 or N == 0
            continue
        segs = sorted(by_m0[m0])
        assert [x[0] for x in segs] == list(np.cumsum([0] + [x[1] for x in segs[:-1]]))
        assert sum(x[1] for x in segs) == N
        for t0, L, row, nt in segs:
            if p == 1.0:
                assert nt == 0
                continue
            J, thr = gt.geo_thresholds(p, L)
            assert nt == J + 1 and [int(x) for x in plan.thr[row]] == [int(x) for x in thr]
        seen.add(m0)
    assert seen == set(by_m0)


def _walk_lanes(launcher):
    return (1, 37, 256) if launcher == "ref" else (32, 256)


@pytest.mark.parametrize("launcher", [n for n, _ in _launchers()])
@pytest.mark.parametrize("case", range(len(WALK_CASES)))
def test_kernel_walk_equals_the_mirror_draw_for_draw(case, launcher):
    """K1's rows and per-shot extended lists equal walk_block_py's fired slots (geo_skip_py on each item's stream,
    draw for draw) on every block, a first_block offset, block sizes that are and are not powers of two, a partial
    last block, any number of threads per group, rows in group or device memory; the run draws ties and several
    words per skip, so both paths are exercised"""
    from _gpu_ref import RefLauncher

    from xtim.gpu import tables as gt

    t, path, B, n = _tables(case)
    plan = gt.walk_plan(t, B)
    seed, first = 77, 3
    W, E = int(t["W"]), 4096
    nb = -(-n // B)
    want = np.zeros(n * W, np.uint64)
    lists = []
    draws = []
    for b in range(nb):
        ls = [[] for _ in range(B)]
        x = [0] * B
        for m, s in gt.walk_block_py(t, plan, b, seed=seed, first_block=first, draws=draws):
            if b * B + s >= n:
                continue
            x[s] ^= _flips(t, m)
            if t["mem_xo"][m]:
                ls[s].append(int(t["mem_xo"][m]))
        for s in range(min(B, n - b * B)):
            for w in range(W):
                want[(b * B + s) * W + w] = (x[s] >> (64 * w)) & ((1 << 64) - 1)
        lists.append([sorted(v) for v in ls])
    if path:   # (the small file fires too rarely to be sure of a tie)
        assert sum(d[2] for d in draws) > 0, "no tie drawn"
        assert any(nt > 8 for nt in plan.seg[:, 1].tolist()), "no skip of two words"
    for lanes in _walk_lanes(launcher):
        for walk in (None, 0, gt.XK_WALK_TGROWS):
            buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=seed, first_block=first)
            if walk is not None:
                buf["dims"][gt.XK_DIM_WALK] = walk
            L = RefLauncher() if launcher == "ref" else dict(_launchers())[launcher]()
            L.compile(HEADER.read_text(), gt.capacities(t, B, E, plan=plan, walk=walk))
            L.upload(buf)
            L.run("walk", nb, lanes)
            rows = L.download("rows")[:n * W]
            epos = L.download("epos")[:nb * (B + 1)].reshape(nb, B + 1)
            eids = L.download("eids")[:nb * E].reshape(nb, E)
            assert not L.download("overflow")[:nb].any()
            got = [[eids[b, epos[b, s]:epos[b, s + 1]].tolist() for s in range(B)] for b in range(nb)]
            assert got == lists, (lanes, walk)
            assert rows.tobytes() == want.tobytes(), (lanes, walk)
    assert any(int(x) for x in want)
