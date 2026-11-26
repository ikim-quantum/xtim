"""The single-line law table and K2's queue (xdem_kernels.h, K2 / K2b; xtim.gpu.tables.law_table).

The table holds, for each distinct nonzero odd mask lo_l of an extended line, the objects xk_law_fast computes from
lo = lo_l (computed on the host by ``tables.line_law``); K2 uses an entry when a shot's composed lo equals a fired
line's lo_l. Under XK_MODE_DEFER, K2 queues the shots whose law it must compute and K2b ("resolve_law") computes them.
Neither may change an output byte: checked against the kernel's own general path (the table left out) for every
extended line of every bundled example alone and in random combinations at several coin words, and through the
backend (sample, sample_ps, estimate) on the reference build and on Metal.
"""
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import tables as gt
from xtim.gpu.backend import GpuBackend

from _gpu_ref import RefLauncher
from test_gpu_kernels import FILES, HEADER, MODE_GIVEN, B, _cases, _given, _read, _tables
from test_gpu_postselect import _mask

ROOT = Path(__file__).parent.parent
COINS = [0, 1, 0x5DEECE66D, (1 << 64) - 1, 0x9E3779B97F4A7C15]


def _one_word(t):
    return bool(t["A"]) and bool(t["fast"]) and not gt.coord_words(t)


def _lo_of(t, xo):
    xa = t["xa"]
    lo = [0] * int(t["WA"])
    for k in range(int(xa[xo]) & 0xFFFF_FFFF):
        lo[int(xa[xo + 1 + 3 * k])] ^= int(xa[xo + 2 + 3 * k])
    return tuple(lo)


def _run(t, buf, E, coin, *, law, defer):
    """resolve (and resolve_law under defer) on given lists; the rows, keys, dims, flags and the queue's length"""
    n = int(buf["dims"][gt.XK_DIM_NSHOTS])
    b = dict(buf)
    st = gt.static_tables(t, block=B, law=law)
    b["tab64"], b["tab32"] = st.tab64, st.tab32
    dims = st.dims.copy()
    for i in (gt.XK_DIM_E, gt.XK_DIM_NSHOTS, gt.XK_DIM_NBLOCKS):
        dims[i] = buf["dims"][i]
    dims[gt.XK_DIM_MODE] = MODE_GIVEN | (gt.XK_MODE_DEFER if defer else 0)
    b["dims"] = dims
    b["coin_in"] = np.full(n, coin, np.uint64)
    b["lawq"] = np.full(n + 1, 0xFFFF_FFFF, np.uint32)   # K1 must empty the queue
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    L.upload(b)
    L.run_batch([("walk", int(dims[gt.XK_DIM_NBLOCKS])), ("resolve", n)]
                + ([("resolve_law", gt.law_threads(n))] if defer else []))
    q = int(L.download("lawq", 1)[0]) if defer else None
    return [L.download(k) for k in ("rows", "ckey", "cdim", "flag")], q


@pytest.fixture(scope="module", params=FILES, ids=[Path(f).name for f in FILES])
def text(request):
    return _read(request.param)


def test_table_equals_the_kernels_general_path(text):
    """every extended line alone, and 400 random member sets, at five coin words: the table and the queue give the
    bytes of the general path; and the table is used (each line with an entry alone is a hit, so nothing is queued
    for it)"""
    s, t = _tables(text)
    if not _one_word(t):
        pytest.skip("no one-word law (no axes, or the multi-word law)")
    xlaw, lawt = gt.law_table(t)
    assert len(lawt) > 1
    rng = np.random.default_rng(7)
    cases = _cases(t, rng)
    buf, E = _given(t, cases, rng)
    for coin in COINS:
        plain, _ = _run(t, buf, E, coin, law=False, defer=False)
        for law, defer in ((True, False), (False, True), (True, True)):
            got, q = _run(t, buf, E, coin, law=(xlaw, lawt) if law else False, defer=defer)
            for a, b in zip(got, plain):
                assert np.array_equal(a, b), (law, defer, coin)
            if defer:
                # the queue: the shots with odd axes and (with the table) no entry
                need = 0
                for ms in cases:
                    xos = [int(t["mem_xo"][m]) for m in ms if t["mem_xo"][m]]
                    lo = [0] * int(t["WA"])
                    for x in xos:
                        lo = [a ^ b for a, b in zip(lo, _lo_of(t, x))]
                    lo = tuple(lo)
                    if not any(lo):
                        continue
                    hit = law and any(xlaw[x] and _lo_of(t, x) == lo for x in xos)
                    need += not hit
                assert q == need
    # non-vacuous: the single-line cases with an entry
    singles = [ms for ms in cases if len(ms) == 1]
    assert sum(1 for (m,) in singles if xlaw[int(t["mem_xo"][m])]) > 0


def test_entries_are_post_fasts_objects(text):
    """each entry: lo_l, omega, npv | dim << 8, the pivots' (col_p, S[p]), the key; line_law's values, distinct
    lo_l per entry, and the table's size within entries * (WA + 2 + 2 rho + rhoL) + 1 words"""
    s, t = _tables(text)
    if not _one_word(t):
        pytest.skip("no one-word law")
    xlaw, lawt = gt.law_table(t)
    WA, rho, rhoL = int(t["WA"]), int(t["rho"]), int(t["rhoL"])
    ents = sorted({int(e) for e in xlaw if e})
    assert len(lawt) <= 1 + len(ents) * (WA + 2 + 2 * rho + rhoL)
    seen = set()
    for e in ents:
        lo = tuple(int(x) for x in lawt[e:e + WA])
        assert lo not in seen and any(lo)
        seen.add(lo)
        odd = [64 * w + b for w in range(WA) for b in range(64) if lo[w] >> b & 1]
        omega, piv, key = gt.line_law(t, odd)
        meta = int(lawt[e + WA + 1])
        assert int(lawt[e + WA]) == omega and meta == len(piv) | len(key) << 8
        o = e + WA + 2
        assert [int(x) for x in lawt[o:o + 2 * len(piv)]] == [v for _, cp, sp in piv for v in (cp, sp)]
        assert [int(x) for x in lawt[o + 2 * len(piv):o + 2 * len(piv) + len(key)]] == key
        assert all(sp & -sp == 1 << p for p, _, sp in piv)   # S[p]'s lowest bit is its pivot
    for xo in {int(x) for x in t["mem_xo"] if x}:
        e = int(xlaw[xo])
        if e:
            assert tuple(int(x) for x in lawt[e:e + WA]) == _lo_of(t, xo)


def test_size_cap_leaves_the_table_out(monkeypatch):
    t = _tables(_read("examples/xdem/cultivation_d5.xdem"))[1]
    xlaw, lawt = gt.law_table(t)
    assert len(lawt) > 100
    monkeypatch.setattr(gt, "LAW_TABLE_MAX_WORDS", 100)
    xl, lt = gt.law_table(t)
    assert len(xl) == 0 and len(lt) == 0
    assert gt.static_tables(t, block=B).dims[gt.XK_DIM_LAWT] == 0


def test_backend_default_follows_law_pays(monkeypatch):
    """GpuBackend's default: the table and the queue where tables.law_pays (mean_ext <= LAW_MAX_MEAN_EXT), neither
    elsewhere; an explicit True / False overrides it"""
    s = xtim.XdemSampler(_read("examples/xdem/cultivation_d5.xdem"))
    t = s._s.device_tables()
    assert 0 < float(t["mean_ext"]) <= gt.LAW_MAX_MEAN_EXT and gt.law_pays(t)
    b = GpuBackend(s._s, RefLauncher())
    assert b._law and len(b._law[1]) and b._defer
    monkeypatch.setattr(gt, "LAW_MAX_MEAN_EXT", 0.5)
    b = GpuBackend(s._s, RefLauncher())
    assert b._law is False and not b._defer
    b = GpuBackend(s._s, RefLauncher(), law_table=True, defer=True)
    assert b._law and b._defer


# ------------------------------------------------------------------------------------------------ through the backend

BACKEND_FILES = ["examples/xdem/cultivation_d5.xdem", "examples/xdem/cultivation_d3_faithful.xdem",
                 "examples/xdem/rm15rep.xdem", "examples/xdem/toy2obs.xdem", "examples/xdem/ccz_toy_obs_p5.xdem",
                 "examples/xdem/distillation_15_1_3.xdem", "tests/data/xdem/synth/boundary_r64.xdem"]
SHOTS, SEED = 5000, 41


def _entry_points(text, launcher, **kw):
    out = []
    s = xtim.XdemSampler(text)
    out.append(GpuBackend(s._s, launcher(), **kw).sample(SHOTS, SEED))
    s = xtim.XdemSampler(text)
    d, o, c, counts = GpuBackend(s._s, launcher(), **kw).sample_ps(SHOTS, SEED, _mask(s.num_detectors, "half"))
    out.append((d, o, c, sorted(counts.items())))
    s = xtim.XdemSampler(text)
    out.append(GpuBackend(s._s, launcher(), max_batch_bytes=1, **kw).estimate(SHOTS, SEED, None))
    s = xtim.XdemSampler(text)
    out.append(GpuBackend(s._s, launcher(), **kw).estimate(SHOTS, SEED, _mask(s.num_detectors, "half")))
    return out


def _same(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("name", BACKEND_FILES, ids=[Path(f).name for f in BACKEND_FILES])
def test_backend_bytes_do_not_depend_on_table_or_queue(name):
    text = _read(name)
    want = _entry_points(text, RefLauncher, law_table=False, defer=False)
    for kw in (dict(law_table=True, defer=False), dict(law_table=False, defer=True), dict(law_table=True, defer=True),
               {}):
        _same(_entry_points(text, RefLauncher, **kw), want)


@pytest.mark.metal
@pytest.mark.parametrize("name", BACKEND_FILES, ids=[Path(f).name for f in BACKEND_FILES])
def test_metal_table_and_queue_match_reference(name):
    from xtim.gpu.metal import MetalLauncher
    text = _read(name)
    want = _entry_points(text, RefLauncher, law_table=False, defer=False)
    _same(_entry_points(text, MetalLauncher, law_table=True, defer=True), want)


@pytest.mark.metal
def test_metal_queue_holds_the_computed_laws():
    """non-vacuous on the device: a cultivation_d5 batch queues some shots, not all, and the table serves others"""
    from xtim.gpu.metal import MetalLauncher
    s = xtim.XdemSampler(_read("examples/xdem/cultivation_d5.xdem"))
    L = MetalLauncher()
    b = GpuBackend(s._s, L)
    n = 1 << 16
    for _ in b._batches(n, 3):
        q = int(L.download("lawq", 1)[0])
    assert 0.1 * n < q < 0.6 * n   # about 30% of its shots (the single-line ones, about 38%, use the table)
