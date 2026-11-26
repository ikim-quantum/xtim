"""The GPU xdem kernel source (xtim/gpu/xdem_kernels.h), run on the CPU through its reference build (Task 4).

K2 ("resolve") is checked exactly against the CPU sampler's fast-path law: with the fired lines given (mode bits
COIN_IN | GIVEN), the corrected row at coin word 0 is the law's offset up to the coin span, the rows at coin words
e_j span the CPU's coin space, dim C and the status C key equal the CPU's (`xdem_law_of`, `xdem_status_of`). K1
("walk") is checked statistically (fire rates per line) and structurally (the per-shot CSR of extended records).
Internal row layout: detector d = bit d, observable o = bit 64 WD + o.
"""
import re
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as T
from xtim.gpu import tables as gt

from _gpu_ref import MULTIWORD, RefLauncher, gf2_rank, in_span, multiword_text

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
# every bundled example, a conventional DEM (no axes, no extended line) and files with O = 0 (ccz_toy,
# code_switching_*_noobs, ...)
FILES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "examples" / "xdem").glob("*.xdem"))
FILES.append("tests/data/xdem/synth/surface_d5r5.dem")
B = 64
MODE_GIVEN = gt.XK_MODE_COIN_IN | gt.XK_MODE_GIVEN


def _read(name):
    return (ROOT / name).read_text()


@pytest.fixture(scope="module", params=FILES, ids=[Path(f).name for f in FILES])
def text(request):
    return _read(request.param)


def _launcher(t, pair_cap, caps=None):
    L = RefLauncher()
    L.compile(HEADER.read_text(), caps or gt.capacities(t, B, pair_cap))
    return L


def _row_int(words):
    return sum(int(x) << (64 * w) for w, x in enumerate(words))


def _to_file(x, t):
    """an internal-layout row (Python int) in the File layout (detector d = bit d, observable o = bit D + o)"""
    D, WD = t["D"], t["WD"]
    return (x & ((1 << D) - 1)) | ((x >> (64 * WD)) << D)


def _member_flips(t, m):
    x = 0
    beg = int(t["mem_beg"][m])
    for e in range(beg, beg + int(t["mem_cnt"][m])):
        x ^= int(t["ent_mask"][e]) << (64 * int(t["ent_word"][e]))
    return x


def _cases(t, rng):
    """fired member sets: every single extended line, 200 random pairs (of extended lines when there are two),
    200 random sets of 3 to 8 chain members"""
    nm = len(t["mem_line"])
    ext = [m for m in range(nm) if t["mem_xo"][m]]
    out = [[m] for m in ext]
    pool = ext if len(ext) >= 2 else list(range(nm))
    if len(pool) >= 2:
        out += [sorted(rng.choice(pool, 2, replace=False).tolist()) for _ in range(200)]
    if nm >= 3:
        out += [sorted(rng.choice(nm, int(rng.integers(3, min(8, nm) + 1)), replace=False).tolist())
                for _ in range(200)]
    return out


def _given(t, cases, rng, order="shuffle", seed=0):
    """pack() buffers for "resolve" over one shot per case: rows = the XOR of the members' flips, extended lists =
    the members' records in the given order (shuffle / reverse / sorted)"""
    n = len(cases)
    W = t["W"]
    lists = []
    for ms in cases:
        xs = [int(t["mem_xo"][m]) for m in ms if t["mem_xo"][m]]
        if order == "shuffle":
            rng.shuffle(xs)
        elif order == "reverse":
            xs = sorted(xs, reverse=True)
        else:
            xs = sorted(xs)
        lists.append(xs)
    nblocks = -(-n // B)
    E = max([sum(len(lists[s]) for s in range(b * B, min(n, b * B + B))) for b in range(nblocks)] + [1])
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=seed, mode=MODE_GIVEN)
    rows = buf["rows"]
    for s, ms in enumerate(cases):
        x = 0
        for m in ms:
            x ^= _member_flips(t, m)
        for w in range(W):
            rows[s * W + w] = (x >> (64 * w)) & ((1 << 64) - 1)
    for b in range(nblocks):
        acc = 0
        for i in range(B):
            s = b * B + i
            buf["epos"][b * (B + 1) + i] = acc
            if s < n:
                buf["eids"][b * E + acc:b * E + acc + len(lists[s])] = lists[s]
                acc += len(lists[s])
        buf["epos"][b * (B + 1) + B] = acc
    return buf, E


def _resolve(t, buf, E, coin=None, caps=None):
    n = int(buf["dims"][gt.XK_DIM_NSHOTS])
    L = _launcher(t, E, caps)
    b = dict(buf)
    if coin is not None:
        b["coin_in"] = np.full(n, coin, np.uint64) if np.isscalar(coin) else np.asarray(coin, np.uint64)
    L.upload(b)
    L.run("resolve", n)
    return L.download("rows"), L.download("ckey"), L.download("cdim")


def _tables(text):
    s = xtim.XdemSampler(text)
    return s, s._s.device_tables()


# ------------------------------------------------------------------------------------------------- the kernel ABI


def test_dims_indices_match_header():
    pat = r"XK_(?:DIM|MODE|OVERFLOW|CDIM|FLAG|HIST|MAX)_\w+"
    defs = dict(re.findall(r"^#define\s+(" + pat + r")\s+(\d+)\s*$", HEADER.read_text(), re.M))
    assert defs, "no XK_DIM_ / XK_MODE_ / XK_OVERFLOW_ / XK_CDIM_ defines in the header"
    py = {k: getattr(gt, k) for k in dir(gt) if re.fullmatch(pat, k)}
    assert {k: int(v) for k, v in defs.items()} == py
    idx = [v for k, v in py.items() if k.startswith("XK_DIM_") and k != "XK_DIM_COUNT"]
    assert len(set(idx)) == len(idx) and max(idx) < py["XK_DIM_COUNT"]
    assert [py[f"XK_DIM_{k}"] for k in ("W", "WD", "WO", "WF", "WA", "RHO", "RHOL", "R", "B", "E", "NCHAINS",
                                        "NSHOTS", "NBLOCKS", "MODE", "SPAN_RANK", "SPAN_WORDS")] == list(range(16))
    offs = [v for k, v in py.items() if k.startswith("XK_DIM_OFF_")]
    assert min(offs) == 16


def test_pack_places_every_table(text):
    s, t = _tables(text)
    if t["A"] and not t["fast"]:
        pytest.skip("general path")
    buf = gt.pack(t, block=B, pair_cap=7, shots=200, seed=5, first_block=3)
    d = buf["dims"]
    assert d.dtype == np.uint32 and d.shape == (gt.XK_DIM_COUNT,)
    for k in ("W", "WD", "WO", "WF", "WA", "rho", "rhoL", "r"):
        assert d[getattr(gt, "XK_DIM_" + k.upper())] == t[k], k
    assert d[gt.XK_DIM_B] == B and d[gt.XK_DIM_E] == 7 and d[gt.XK_DIM_NSHOTS] == 200
    assert d[gt.XK_DIM_NBLOCKS] == 4 and d[gt.XK_DIM_NCHAINS] == len(t["chain_p"])
    for name, arena in (("ent_mask", "tab64"), ("xa", "tab64"), ("Tc", "tab64"), ("Kc", "tab64"),
                        ("beta", "tab64"), ("betaO", "tab64"), ("bLo", "tab64"), ("chain_off", "tab32"),
                        ("mem_beg", "tab32"), ("mem_cnt", "tab32"), ("mem_xo", "tab32"), ("ent_word", "tab32"),
                        ("bb", "tab32"), ("pL", "tab32")):
        off = int(d[getattr(gt, "XK_DIM_OFF_" + name.upper())])
        a = buf[arena]
        assert [int(x) for x in a[off:off + len(t[name])]] == [int(x) for x in t[name]], name
    # the walk's plan: item starts and segment records in tab32, threshold rows (and their top bytes) in tab64
    plan = gt.walk_plan(t, B)
    o32 = buf["tab32"]
    ni = int(d[gt.XK_DIM_NITEMS])
    assert ni == plan.nitems
    oi, osg = int(d[gt.XK_DIM_OFF_ITEM]), int(d[gt.XK_DIM_OFF_SEG])
    assert o32[oi:oi + ni + 1].tolist() == plan.item_beg.tolist()
    assert o32[osg:osg + plan.seg.size].tolist() == plan.seg.reshape(-1).tolist()
    stride = int(d[gt.XK_DIM_THR_STRIDE])
    s8 = int(d[gt.XK_DIM_THR8_STRIDE])
    assert s8 == -(-stride // 8)
    for r, thr in enumerate(plan.thr):
        J = len(thr) - 1
        assert J + 1 <= stride
        o = int(d[gt.XK_DIM_OFF_CHAIN_THR]) + r * stride
        assert buf["tab64"][o:o + J + 1].tolist() == [int(x) for x in thr]
        # the top bytes, eight tests to a word: byte i of word q = the top byte of test 8 q + i
        o8 = int(d[gt.XK_DIM_OFF_CHAIN_THR8]) + r * s8
        w8 = [int(x) for x in buf["tab64"][o8:o8 + s8]]
        assert [(w8[i // 8] >> (8 * (i % 8))) & 0xFF for i in range(8 * s8)] == \
            [int(x) >> 56 for x in thr] + [0] * (8 * s8 - J - 1)
    sd = buf["seed"]
    assert sd.tolist() == [gt.stream_key(5, gt.K_TAG_GPU_NOISE), gt.stream_key(5, gt.K_TAG_GPU_COIN), 3, t["rmask"]]
    assert buf["rows"].shape == (200 * t["W"],) and buf["epos"].shape == (4 * (B + 1),)
    assert buf["eids"].shape == (4 * 7,) and buf["overflow"].shape == (4,)
    assert buf["ckey"].shape == (200 * max(t["rhoL"], 1),) and buf["cdim"].dtype == np.uint8
    assert buf["coin_in"].shape == (200,)
    caps = gt.capacities(t, B, 7)
    assert caps["XK_CAP_E"] == 7 and caps["XK_CAP_W"] == max(t["W"], 1) and caps["XK_CAP_J"] >= 1


def test_mix_matches_cpp_constants():
    # SplitMix64's finalizer (krng::mix): mix(0) = 0, and a reference value computed by hand from its definition
    assert gt.mix(0) == 0
    z = 1
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
    assert gt.mix(1) == z ^ (z >> 31)
    assert gt.stream_key(7, 9) == gt.mix(7 ^ gt.mix(9))
    assert gt.K_TAG_GPU_NOISE == 0x67704E6F69736501 and gt.K_TAG_GPU_COIN == 0x6770436F696E0002


# ------------------------------------------------------------------------------------------------- K2 resolve


def test_resolve_matches_cpu_law(text):
    s, t = _tables(text)
    rng = np.random.default_rng(1)
    cases = _cases(t, rng)
    assert cases
    buf, E = _given(t, cases, rng)
    W, r, rhoL = t["W"], t["r"], t["rhoL"]
    kw = max(rhoL, 1)
    rows0, ckey, cdim = _resolve(t, buf, E, coin=0)
    rows_j = [_resolve(t, buf, E, coin=1 << j)[0] for j in range(r)]
    lines = t["mem_line"]
    for k, ms in enumerate(cases):
        ids = [int(lines[m]) for m in ms]
        law = T.xdem_law_of(text, ids, False)
        off = _to_file(_row_int(rows0[k * W:(k + 1) * W]), t)
        coins = [_to_file(_row_int(rj[k * W:(k + 1) * W]), t) ^ off for rj in rows_j]
        cpu = [int(c) for c in law["coins"]]
        rk = gf2_rank(cpu)
        assert gf2_rank(coins) == rk == gf2_rank(coins + cpu), (ids, coins, cpu)
        assert in_span(off ^ int(law["offset"]), cpu), (ids, off, law["offset"])
        assert int(cdim[k]) == law["dim"], (ids, int(cdim[k]), law["dim"])
        key = ckey[k * kw:(k + 1) * kw]
        dim = int(cdim[k])
        assert not key[dim:].any()
        if dim:
            st = T.xdem_status_of(text, ids, False)
            assert st["dim"] == dim
            cid = int(s._s.cspace_of_keys(key[:dim].reshape(1, dim))[0])
            got = [sum(1 << i for i, b in enumerate(row) if b) for row in s._s.cspaces(cid, 1)[0]]
            assert got == [int(x) for x in st["cspace"]], (ids, got, st["cspace"])


# ---- the multi-word law (coord_words >= 2): the same exact checks with cw coin words per shot and cw-word keys


def _resolve_mw(t, buf, E, coins):
    """resolve with the given coin words (uint64[n, cw]: shot s draws coins[s])"""
    b = dict(buf)
    b["coin_in"] = np.ascontiguousarray(coins, np.uint64).reshape(-1)
    return _resolve(t, b, E)


@pytest.mark.parametrize("name", MULTIWORD)
def test_resolve_matches_cpu_law_multiword(name):
    """with the fired lines given and coin words t = 0 and t = e_j (j < r, spread over cw words), the corrected rows
    give the CPU law's offset (modulo its coins) and its coin span, dim C and the status C key (cw-word vectors)
    equal the CPU's; cw8_r400 on a subset of the cases (its 400 coin runs are slow)"""
    text = multiword_text(name)
    s, t = _tables(text)
    cw = gt.coord_words(t)
    assert cw >= 2
    rng = np.random.default_rng(1)
    cases = _cases(t, rng)
    if name == "cw8_r400":
        cases = cases[:40]
    buf, E = _given(t, cases, rng)
    n, W, r, rhoL = len(cases), t["W"], t["r"], t["rhoL"]
    kw = max(rhoL, 1) * cw
    zero = np.zeros((n, cw), np.uint64)
    rows0, ckey, cdim = _resolve_mw(t, buf, E, zero)
    rows_j = []
    for j in range(r):
        c = zero.copy()
        c[:, j // 64] = np.uint64(1 << (j % 64))
        rows_j.append(_resolve_mw(t, buf, E, c)[0])
    lines = t["mem_line"]
    dims = set()
    for k, ms in enumerate(cases):
        ids = [int(lines[m]) for m in ms]
        law = T.xdem_law_of(text, ids, False)
        off = _to_file(_row_int(rows0[k * W:(k + 1) * W]), t)
        coins = [_to_file(_row_int(rj[k * W:(k + 1) * W]), t) ^ off for rj in rows_j]
        cpu = [int(c) for c in law["coins"]]
        rk = gf2_rank(cpu)
        assert gf2_rank(coins) == rk == gf2_rank(coins + cpu), ids
        assert in_span(off ^ int(law["offset"]), cpu), ids
        assert int(cdim[k]) == law["dim"], (ids, int(cdim[k]), law["dim"])
        key = ckey[k * kw:(k + 1) * kw]
        dim = int(cdim[k])
        dims.add(dim)
        assert not key[dim * cw:].any()
        if dim:
            st = T.xdem_status_of(text, ids, False)
            assert st["dim"] == dim
            cid = int(s._s.cspace_of_keys(key[:dim * cw].reshape(1, dim * cw))[0])
            got = [sum(1 << i for i, b in enumerate(row) if b) for row in s._s.cspaces(cid, 1)[0]]
            assert got == [int(x) for x in st["cspace"]], ids
    if name != "cw8_r400":   # (its cases, like its sampled shots, all have C = {0})
        assert len(dims) > 1   # not vacuous: some case has a nontrivial C


def test_multiword_coin_stream_is_keyed_per_shot():
    """without COIN_IN, coin word k of global shot g is mix(key_k + (g + 1) gamma), key_0 = the coin key, key_k =
    mix(coin key ^ mix(k)), bits past r cleared: equal to passing those words in"""
    text = multiword_text("boundary_r65")
    _, t = _tables(text)
    cw, r = gt.coord_words(t), int(t["r"])
    rng = np.random.default_rng(3)
    cases = _cases(t, rng)[:300]
    buf, E = _given(t, cases, rng, seed=99)
    buf["seed"][2] = 5   # first block
    n = len(cases)
    key = gt.stream_key(99, gt.K_TAG_GPU_COIN)
    words = np.zeros((n, cw), np.uint64)
    for g in range(n):
        for k in range(cw):
            kk = key if k == 0 else gt.mix(key ^ gt.mix(k))
            bits = max(0, min(64, r - 64 * k))
            words[g, k] = gt.mix((kk + (5 * B + g + 1) * gt.K_GAMMA) & (2**64 - 1)) & ((1 << bits) - 1)
    own = dict(buf)
    own["dims"] = buf["dims"].copy()
    own["dims"][gt.XK_DIM_MODE] = gt.XK_MODE_GIVEN
    got = _resolve(t, own, E)
    want = _resolve_mw(t, buf, E, words)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(got, want))
    assert int(words[:, 1].max()) <= 1 and words[:, 1].any()   # r = 65: one bit in the second word


@pytest.mark.parametrize("name", ["rm15rep", "toy2obs"])
def test_resolve_cases_reach_nontrivial_c(name):
    # the status key check above is not vacuous: these files' cases include laws with dim C = 1 and 2
    _, t = _tables(_read(f"examples/xdem/{name}.xdem"))
    rng = np.random.default_rng(1)
    buf, E = _given(t, _cases(t, rng), rng)
    assert {1, 2} <= set(_resolve(t, buf, E, coin=0)[2].tolist())


def test_resolve_independent_of_list_order(text):
    _, t = _tables(text)
    rng = np.random.default_rng(2)
    cases = [c for c in _cases(t, rng) if len(c) > 1]
    if not cases:
        pytest.skip("no multi-line case")
    coin = rng.integers(0, 2**63, size=len(cases), dtype=np.uint64) * 2 + 1
    outs = []
    for order in ("sorted", "reverse", "shuffle", "shuffle"):
        buf, E = _given(t, cases, np.random.default_rng(len(outs)), order=order)
        outs.append(tuple(a.tobytes() for a in _resolve(t, buf, E, coin=coin)))
    assert all(o == outs[0] for o in outs[1:])


def test_resolve_coin_stream_is_keyed_per_shot(text):
    # without COIN_IN, shot g's coin word is mix(coin_key + (g + 1) gamma) & rmask: equal to passing that word in
    _, t = _tables(text)
    if not t["r"]:
        pytest.skip("no coins")
    rng = np.random.default_rng(3)
    cases = _cases(t, rng)[:300]
    buf, E = _given(t, cases, rng, seed=99)
    buf["seed"][2] = 5   # first block
    n = len(cases)
    key = gt.stream_key(99, gt.K_TAG_GPU_COIN)
    words = [gt.mix((key + (5 * B + g + 1) * gt.K_GAMMA) & (2**64 - 1)) & int(t["rmask"]) for g in range(n)]
    own = dict(buf)
    own["dims"] = buf["dims"].copy()
    own["dims"][gt.XK_DIM_MODE] = gt.XK_MODE_GIVEN
    got = _resolve(t, own, E)
    want = _resolve(t, buf, E, coin=np.array(words, np.uint64))
    assert all(a.tobytes() == b.tobytes() for a, b in zip(got, want))


# ------------------------------------------------------------------------------------------------- K1 walk

_RATES_TEXT = "\n".join(
    ["error(0) D0", "error(1) D1", "error(1) D2", "error(1e-12) D3"]
    + [f"error(0.001) D{d}" for d in range(4, 7)]
    + [f"error(0.3) D{d}" for d in range(7, 12)]
    + ["error(0.05) D12", "error(0.3) L0"]) + "\n"
_RATES_P = [0, 1, 1, 1e-12] + [1e-3] * 3 + [0.3] * 5 + [0.05, 0.3]


def _walk(t, nblocks, pair_cap, seed=11, first_block=0, shots=None, caps=None, extra=None):
    n = nblocks * B if shots is None else shots
    buf = gt.pack(t, block=B, pair_cap=pair_cap, shots=n, seed=seed, first_block=first_block)
    if extra:
        buf.update(extra)
    L = _launcher(t, pair_cap, caps)
    L.upload(buf)
    L.run("walk", int(buf["dims"][gt.XK_DIM_NBLOCKS]))
    return L


def test_walk_fire_rates():
    _, t = _tables(_RATES_TEXT)
    nb = 4096
    n = nb * B
    L = _walk(t, nb, 1)
    rows = L.download("rows").reshape(n, t["W"])
    assert not L.download("overflow").any()
    assert t["W"] == 2 and t["WD"] == 1
    det = rows[:, 0]
    obs = rows[:, 1]
    assert not (det >> np.uint64(13)).any() and not (obs >> np.uint64(1)).any()   # no bit outside the columns
    cols = [(det >> np.uint64(d)) & np.uint64(1) for d in range(13)] + [obs & np.uint64(1)]
    for i, (p, col) in enumerate(zip(_RATES_P, cols)):
        k = int(col.sum())
        if p == 0:
            assert k == 0, i
        elif p == 1:
            assert k == n, i
        else:
            z = (k - n * p) / np.sqrt(n * p * (1 - p))
            assert abs(z) <= 5, (i, p, k, z)
    # lines of one chain fire independently (two p = 0.3 members, and a member with its neighbour shot)
    for a, b, shift in ((cols[7], cols[11], 0), (cols[8], cols[8], 1)):
        both = int((a[shift:] & b[:n - shift]).sum())
        m = n - shift
        z = (both - m * 0.09) / np.sqrt(m * 0.09 * 0.91)
        assert abs(z) <= 5, (shift, both, z)


def _check_csr(t, epos, eids, E, nblocks):
    valid = set(int(x) for x in t["mem_xo"] if x)
    total = 0
    for b in range(nblocks):
        fp = epos[b * (B + 1):(b + 1) * (B + 1)].astype(np.int64)
        assert fp[0] == 0 and (np.diff(fp) >= 0).all() and fp[B] <= E, b
        for s in range(B):
            lst = eids[b * E + fp[s]:b * E + fp[s + 1]].tolist()
            assert set(lst) <= valid and len(set(lst)) == len(lst), (b, s, lst)
        total += int(fp[B])
    return total


def test_walk_pairs_csr_wellformed(text):
    _, t = _tables(text)
    if t["A"] and not t["fast"]:
        pytest.skip("general path")
    nb = 256
    E = int(B * t["mean_ext"] + 10 * np.sqrt(B * t["mean_ext"] + 1) + 64)
    L = _walk(t, nb, E)
    assert not L.download("overflow").any()
    total = _check_csr(t, L.download("epos"), L.download("eids"), E, nb)
    # the number of extended pairs is the number of fired extended lines: mean n * mean_ext
    p = np.array([t["chain_p"][c] for c in range(len(t["chain_p"]))
                  for _ in range(int(t["chain_off"][c]), int(t["chain_off"][c + 1]))])
    ext = t["mem_xo"] != 0
    n = nb * B
    var = n * float((p[ext] * (1 - p[ext])).sum())
    mean = n * float(p[ext].sum())
    assert mean == pytest.approx(n * t["mean_ext"])
    if var == 0:
        assert total == mean
    else:
        assert abs(total - mean) / np.sqrt(var) <= 5, (total, mean)


def test_overflow_flag():
    _, t = _tables(_read("examples/xdem/ccz_toy_obs.xdem"))   # 3.8 extended lines per block of 64 on average
    nb, E, tail = 256, 1, 32
    sentinel = np.uint32(0xDEADBEEF)
    extra = {"eids": np.full(nb * E + tail, sentinel, np.uint32),
             "epos": np.full(nb * (B + 1) + tail, sentinel, np.uint32)}
    L = _walk(t, nb, E, extra=extra)
    ov = L.download("overflow")
    epos, eids = L.download("epos"), L.download("eids")
    assert ov.any() and not ov.all(), ov
    assert set(ov.tolist()) == {0, gt.XK_OVERFLOW_PAIRS}
    assert (eids[nb * E:] == sentinel).all() and (epos[nb * (B + 1):] == sentinel).all()
    for b in range(nb):
        fp = epos[b * (B + 1):(b + 1) * (B + 1)]
        if ov[b]:
            assert not fp.any(), b   # an overflowed block lists nothing (the host relaunches it)
        else:
            assert fp[B] <= E
    good = [b for b in range(nb) if not ov[b]]
    _check_csr(t, epos, eids, E, nb)
    assert good
    # resolve after an overflow reads only the (empty) lists
    L.run("resolve", nb * B)


def test_blocks_depend_only_on_seed_and_index():
    _, t = _tables(_read("examples/xdem/cultivation_d5.xdem"))
    E = 512
    full = _walk(t, 8, E, seed=4)
    part = _walk(t, 3, E, seed=4, first_block=2, shots=2 * B + 17)
    W = t["W"]
    assert (full.download("rows")[2 * B * W:(4 * B + 17) * W] == part.download("rows")).all()
    assert (full.download("epos")[2 * (B + 1):4 * (B + 1)] == part.download("epos")[:2 * (B + 1)]).all()
    assert (full.download("eids")[2 * E:4 * E] == part.download("eids")[:2 * E]).all()
    # the partial block: its shots' lists are the full block's lists of the same shots
    fp_full = full.download("epos")[4 * (B + 1):5 * (B + 1)]
    fp_part = part.download("epos")[2 * (B + 1):3 * (B + 1)]
    ef, ep = full.download("eids")[4 * E:5 * E], part.download("eids")[2 * E:3 * E]
    for s in range(17):
        assert ef[fp_full[s]:fp_full[s + 1]].tolist() == ep[fp_part[s]:fp_part[s + 1]].tolist()
    assert fp_part[B] == fp_part[17]


def test_output_independent_of_capacities():
    tight = T.xdem_gpu_ref_caps()[1]
    E = 256
    used = 0
    for name in FILES:
        _, t = _tables(_read(name))
        need = gt.capacities(t, B, E)
        if any(need[k] > tight[k] for k in tight):
            continue
        used += 1
        outs = []
        for caps in (T.xdem_gpu_ref_caps()[0], tight):
            L = _walk(t, 40, E, seed=21, shots=40 * B - 13, caps=caps)
            L.run("resolve", 40 * B - 13)
            outs.append([L.download(k).tobytes() for k in ("rows", "epos", "eids", "overflow", "ckey", "cdim")])
        assert outs[0] == outs[1], name
    assert used >= 8, used


_OUTS = ("rows", "epos", "eids", "overflow", "ckey", "cdim")


def test_thread_ids_past_the_batch_are_ignored():
    # a GPU grid rounded up to whole threadgroups: ids >= nblocks (walk) / >= nshots (resolve) write nothing
    _, t = _tables(_read("examples/xdem/rm15rep.xdem"))
    E, n, extra, tail = 64, 3 * B - 5, 70, 40
    sentinel = {"rows": np.uint64(0xA5A5A5A5A5A5A5A5), "epos": np.uint32(0xDEADBEEF), "eids": np.uint32(0xDEADBEEF),
                "overflow": np.uint32(0xDEADBEEF), "ckey": np.uint64(0xA5A5A5A5A5A5A5A5), "cdim": np.uint8(0xA5)}
    outs = []
    for over in (0, extra):
        buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=8)
        size = {k: len(buf[k]) for k in _OUTS}
        for k in _OUTS:
            buf[k] = np.concatenate([buf[k], np.full(tail, sentinel[k], buf[k].dtype)])
        L = _launcher(t, E)
        L.upload(buf)
        L.run("walk", 3 + over)
        L.run("resolve", n + over)
        got = {k: L.download(k) for k in _OUTS}
        for k in _OUTS:
            assert (got[k][size[k]:] == sentinel[k]).all(), (over, k)
        outs.append([got[k][:size[k]].tobytes() for k in _OUTS])
    assert outs[0] == outs[1]
    assert (np.frombuffer(outs[0][5], np.uint8) > 0).any()   # the batch did resolve nontrivial laws


def test_capacity_above_its_macro_is_reported():
    # a size above a capacity macro is a detectable result, never a different answer: K1 flags every block
    # (XK_OVERFLOW_CAPACITY, empty lists), K2 flags every shot (XK_CDIM_CAPACITY, row unchanged)
    tight = T.xdem_gpu_ref_caps()[1]
    _, t = _tables(_read("examples/xdem/cultivation_d5.xdem"))
    E = tight["XK_CAP_E"] + 1   # only E is too large
    L = _walk(t, 4, E, caps=tight)
    assert (L.download("overflow") == gt.XK_OVERFLOW_CAPACITY).all()
    assert not L.download("epos").any() and not L.download("rows").any()
    L.run("resolve", 4 * B)
    assert (L.download("cdim") == gt.XK_CDIM_CAPACITY).all() and not L.download("rows").any()
    # rho = 39 > XK_CAP_RHO = 15 on the fast path's boundary file: given rows are left as they are
    _, t = _tables(_read("tests/data/xdem/synth/boundary_r64.xdem"))
    rng = np.random.default_rng(4)
    buf, E = _given(t, _cases(t, rng)[:100], rng)
    for caps, flagged in ((T.xdem_gpu_ref_caps()[0], False), (tight, True)):
        rows, _, cdim = _resolve(t, buf, E, coin=0, caps=caps)
        assert (cdim == gt.XK_CDIM_CAPACITY).all() == flagged and (cdim == gt.XK_CDIM_CAPACITY).any() == flagged
        if flagged:
            assert (rows == buf["rows"]).all()
        else:
            assert (rows != buf["rows"]).any()
