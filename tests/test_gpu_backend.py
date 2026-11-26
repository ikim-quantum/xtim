"""The GPU backend's batched sample() (xtim/gpu/backend.py, Task 6), run through the kernels' C++ reference build
(`RefLauncher`, tests/_gpu_ref.py: the same bytes as the GPU), plus one Metal copy of the first test.

Exact properties: output shapes and dtypes are the CPU sampler's; sample(n) is a prefix of sample(m) for n <= m (off
the block grid too); the bytes do not depend on the batch size or on overflow relaunches. Layout: with the fired
lines forced to the CPU sampler's own (`_given`), the backend's host conversion gives the CPU's dets / obs bytes
(conventional DEM) and the CPU's cspace ids (extended files). Statistics against the CPU sampler: the pass mean,
every detector marginal and every detector x failure joint of tests/test_xdem_cross_route.py::cross_compare.
"""
import math
import time
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim.gpu import tables as gt
from xtim.gpu.backend import DEFAULT_BLOCK, GpuBackend
from xtim.xdem import XdemStatus

from _gpu_ref import RefLauncher, host_outputs
from _xdem_circuits import NAMES_INREPO, circuit, present, xdem_export
from test_xdem_cross_route import Z_MAX, _Acc, _z_mean, _z_prop

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)

# rhoL = 1 (cultivation_d5), rhoL = 2 (rm15rep, toy2obs), many extended lines per block (ccz_toy_obs_p5), a
# conventional DEM (no axes)
FILES = ["examples/xdem/cultivation_d5.xdem", "examples/xdem/rm15rep.xdem", "examples/xdem/toy2obs.xdem",
         "examples/xdem/ccz_toy_obs_p5.xdem", "tests/data/xdem/synth/surface_d5r5.dem"]


def _read(name):
    return (ROOT / name).read_text()


@pytest.fixture(scope="module", params=FILES, ids=[Path(f).name for f in FILES])
def text(request):
    return _read(request.param)


def backend(text, launcher=None, **kw):
    s = xtim.XdemSampler(text)
    return GpuBackend(s._s, launcher or RefLauncher(), **kw)


def _sizes(s):
    return (s.num_detectors + 7) // 8, (s.num_observables + 7) // 8


def _same(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b))


# ------------------------------------------------------------------------------------------------- exact properties


def _shapes_and_prefix(text, launcher):
    s = xtim.XdemSampler(text)
    b = GpuBackend(s._s, launcher)
    nbD, nbO = _sizes(s)
    d100, o100, c100 = b.sample(100, 7)
    d5k, o5k, c5k = b.sample(5000, 7)
    assert d100.shape == (100, nbD) and o100.shape == (100, nbO) and c100.shape == (100,)
    assert d5k.shape == (5000, nbD) and o5k.shape == (5000, nbO) and c5k.shape == (5000,)
    assert d100.dtype == np.uint8 and o100.dtype == np.uint8 and c100.dtype == np.int32
    assert (d5k[:100] == d100).all() and (o5k[:100] == o100).all() and (c5k[:100] == c100).all()
    assert d5k.any()   # not vacuous
    d0, o0, c0 = b.sample(0, 7)
    assert d0.shape == (0, nbD) and o0.shape == (0, nbO) and c0.shape == (0,)
    assert d0.dtype == np.uint8 and o0.dtype == np.uint8 and c0.dtype == np.int32
    assert not _same(b.sample(5000, 8), (d5k, o5k, c5k))   # the seed matters


def test_shapes_and_prefix(text):
    _shapes_and_prefix(text, RefLauncher())


@pytest.mark.metal
def test_shapes_and_prefix_metal(text):
    from xtim.gpu.metal import MetalLauncher
    _shapes_and_prefix(text, MetalLauncher())


@pytest.mark.metal
def test_metal_equals_ref(text):
    from xtim.gpu.metal import MetalLauncher
    assert _same(backend(text, MetalLauncher()).sample(3000, 5), backend(text).sample(3000, 5))


def test_independent_of_batch_size(text):
    one = backend(text)
    want = one.sample(1000, 3)
    assert one.batches_run == 1   # the default: one batch
    for mbb in (1, 5 * 4096):   # one block per batch; a few blocks per batch
        b = backend(text, max_batch_bytes=mbb)
        assert _same(b.sample(1000, 3), want), mbb
        assert b.batches_run > 1


def test_first_block_is_a_stream_offset(text):
    """sample(n, seed, first_block=k) = rows [k * B, k * B + n) of sample(k * B + n, seed); the offset may exceed 2**32"""
    b = backend(text)
    whole = b.sample(7 * B + 20, 5)
    for k, n in ((3, 2 * B + 20), (1, 6 * B), (0, 100)):
        got = b.sample(n, 5, first_block=k)
        assert _same(got, tuple(a[k * B:k * B + n] for a in whole)), k
    big = 2**33
    a = b.sample(3 * B, 6, first_block=big)
    c = b.sample(2 * B, 6, first_block=big + 1)
    assert _same(c, tuple(x[B:] for x in a))
    assert not _same(b.sample(2 * B, 6, first_block=big), c)
    with pytest.raises(ValueError, match="first_block"):
        b.sample(1, 6, first_block=-1)
    with pytest.raises(ValueError, match="first_block"):
        b.sample(1, 6, first_block=2**64)


@pytest.mark.parametrize("entry", ["sample", "sample_ps", "estimate"])
def test_first_block_past_the_last_block_index_is_refused_up_front(entry):
    """first_block + the stream's blocks must fit 64 bits, the CPU sampler's rule and message: refused before any
    batch runs (not after earlier batches, and no block index wraps to block 0's)"""
    from _gpu_compare import mask
    text = _read(FILES[0])
    nb = 5
    shots = nb * B - 3
    cpu = xtim.XdemSampler(text)
    with pytest.raises(ValueError) as cpu_err:   # the CPU sampler's refusal
        cpu._s.sample(1, 1, 1, 2**64 - 1)
    flags = mask(cpu.num_detectors, "all")
    for first_block in (2**64 - 1, 2**64 - nb + 1, 2**64 - 2):
        b = backend(text, max_batch_bytes=1)   # one block per batch
        call = {"sample": lambda: b.sample(shots, 1, first_block=first_block),
                "sample_ps": lambda: b.sample_ps(shots, 1, flags, first_block=first_block),
                "estimate": lambda: b.estimate(shots, 1, flags, first_block=first_block)}[entry]
        with pytest.raises(ValueError, match="first_block out of range") as e:
            call()
        assert str(e.value) == str(cpu_err.value).split(": ", 1)[-1] or str(cpu_err.value) in str(e.value)
        assert b.batches_run == 0
    # the largest first_block that fits: its last block is 2**64 - 1 (the CPU sampler refuses one more)
    b = backend(text)
    ok = 2**64 - 1 - nb
    {"sample": lambda: b.sample(shots, 1, first_block=ok),
     "sample_ps": lambda: b.sample_ps(shots, 1, flags, first_block=ok),
     "estimate": lambda: b.estimate(shots, 1, flags, first_block=ok)}[entry]()
    assert b.batches_run > 0


def test_pack_batch_refuses_a_wrapping_block_index():
    t = xtim.XdemSampler(_read(FILES[0]))._s.device_tables()
    st = gt.static_tables(t, block=B)
    gt.pack_batch(st, pair_cap=64, shots=2 * B, seed=1, first_block=2**64 - 2)   # blocks 2**64 - 2, 2**64 - 1
    with pytest.raises(ValueError, match="last block's index"):
        gt.pack_batch(st, pair_cap=64, shots=2 * B, seed=1, first_block=2**64 - 1)


def test_interrupt_inside_the_c_table_update_keeps_keys_and_ids_together(monkeypatch):
    """_emitted assigns the device C table's keys and ids together: an exception between computing them leaves both
    unchanged (an assignment of one first would leave the lengths different and break every later upload)"""
    text = _read("examples/xdem/cultivation_d5.xdem")
    b = backend(text)
    real = np.concatenate

    def concatenate(arrays, *a, **kw):
        if isinstance(arrays, (list, tuple)) and len(arrays) == 2 and arrays[0] is b._cids and boom:
            raise KeyboardInterrupt
        return real(arrays, *a, **kw)
    boom = True
    monkeypatch.setattr(np, "concatenate", concatenate)
    with pytest.raises(KeyboardInterrupt):
        b.sample(4 * B, 3)
    boom = False
    assert len(b._ckeys) == len(b._cids)
    got = b.sample(4 * B, 3)
    want = backend(text).sample(4 * B, 3)
    assert _same(got, want) and len(b._ckeys) == len(b._cids) > 0


class _Counting(RefLauncher):
    def __init__(self):
        super().__init__()
        self.compiles = []

    def compile(self, source, capacities):
        self.compiles.append(dict(capacities))
        super().compile(source, capacities)


def test_overflow_relaunch_same_bytes(text):
    want = backend(text).sample(2000, 4)
    L = _Counting()
    b = backend(text, L)
    if not b.tables["mean_ext"]:
        pytest.skip("no extended line: nothing to overflow")
    b._pair_cap = 1   # force the first launches to overflow
    assert _same(b.sample(2000, 4), want)
    assert b._pair_cap > 1 and b.relaunches > 0
    # every pair_cap change recompiled with the new XK_CAP_E
    assert L.compiles[-1]["XK_CAP_E"] == b._pair_cap
    assert len({c["XK_CAP_E"] for c in L.compiles}) == len(L.compiles) >= 2


def test_default_pair_cap():
    b = backend(_read("examples/xdem/cultivation_d5.xdem"))
    assert b._pair_cap == max(64, math.ceil(4 * B * b.tables["mean_ext"])) == 2406
    assert backend(_read("examples/xdem/ccz_toy.xdem"), block=8)._pair_cap == 64


class _TightRef(RefLauncher):
    """always the reference build with the small capacities (XK_CAP_E = 256), whatever compile asks for"""

    def compile(self, source, capacities):
        super().compile(source, capacities)
        self._variant = 1


class _Inject(RefLauncher):
    """the reference launcher with one value written into a buffer's middle shot just before "emit" runs (after
    "reduce" counted it), as a kernel's error code would be"""

    def __init__(self, name, value):
        super().__init__()
        self._inj = (name, value)

    def run(self, kernel, n, lanes=None):
        if kernel == "emit":
            self._buf[self._inj[0]][int(self._buf["dims"][gt.XK_DIM_NSHOTS]) // 2] = self._inj[1]
        super().run(kernel, n, lanes)


def test_capacity_overflow_is_an_invariant_error():
    # pair_cap 301 > the build's XK_CAP_E 256: the kernel reports XK_OVERFLOW_CAPACITY, never retried
    b = backend(_read("examples/xdem/cultivation_d5.xdem"), _TightRef())
    with pytest.raises(xtim.XtimInvariantError, match="capacity"):
        b.sample(500, 1)
    assert b.relaunches == 0


@pytest.mark.parametrize("value,match", [(gt.XK_CDIM_CAPACITY, "capacity"), (gt.XK_CDIM_INTERNAL, "internal")])
def test_bad_cdim_is_an_invariant_error(value, match):
    b = backend(_read("examples/xdem/rm15rep.xdem"), _Inject("cdim", value))
    with pytest.raises(xtim.XtimInvariantError, match=match):
        b.sample(500, 1)


def test_degenerate_files():
    cases = [
        ("error(0.1) D0\n", 1, 0),   # conventional, no observable
        (_read("tests/data/xdem/synth/surface_d5r5.dem"), 120, 1),   # conventional
        (_read("examples/xdem/ccz_toy.xdem"), 3, 0),   # O = 0, rhoL = 0
        (_read("examples/xdem/ccz_toy_obs.xdem"), 2, 1),   # rhoL = 0
        (_read("examples/xdem/code_switching_rate_noobs.xdem"), 26, 0),   # rho = 0
    ]
    for txt, D, O in cases:
        s = xtim.XdemSampler(txt)
        t = s._s.device_tables()
        assert (t["D"], t["O"], t["rhoL"]) == (D, O, 0)
        b = GpuBackend(s._s, RefLauncher())
        for shots in (0, 1, 5, B - 1, B, B + 1, 3000):
            d, o, c = b.sample(shots, 2)
            assert d.shape == (shots, (D + 7) // 8) and o.shape == (shots, (O + 7) // 8) and c.shape == (shots,)
            assert not c.any()
        assert d.any()
    # a conventional DEM's line fires at its rate
    d, _, _ = GpuBackend(xtim.XdemSampler("error(0.1) D0\n")._s, RefLauncher()).sample(40_000, 9)
    k = int(d[:, 0].sum())
    assert abs(k - 4000) / math.sqrt(40_000 * 0.09) <= 5, k


def test_unsupported_file_refused():
    """the backend refuses tables off the fast path (the general path: coord_words = 0); a multi-word file is taken"""
    s = xtim.XdemSampler(_read("tests/data/xdem/synth/boundary_r65.xdem"))
    GpuBackend(s._s, RefLauncher())

    class General:
        def device_tables(self):
            t = dict(s._s.device_tables())
            t["coord_words"], t["fast"] = 0, False
            return t
    with pytest.raises(ValueError, match="coord_words = 0"):
        GpuBackend(General(), RefLauncher())


def test_bad_arguments():
    b = backend(_read("examples/xdem/toy2obs.xdem"))
    with pytest.raises(ValueError):
        b.sample(-1, 0)
    with pytest.raises(ValueError):
        b.sample(1, -1)
    with pytest.raises(ValueError):
        b.sample(1, 2**64)
    with pytest.raises(ValueError):
        backend(_read("examples/xdem/toy2obs.xdem"), block=0)


def test_status_scores_like_cpu(text):
    s = xtim.XdemSampler(text)
    b = GpuBackend(s._s, RefLauncher())
    n = 20_000
    _, obs, cs = b.sample(n, 12)
    status = XdemStatus(obs, cs, None, s, s._s.num_cspaces())
    p = s.score(status, flips=np.zeros((n, s.num_observables), bool))
    dims = np.array([len(c) for c in status.cspaces])[cs]
    assert set(np.unique(p).tolist()) <= {0.0} | {2.0 ** -k for k in range(int(dims.max()) + 1)}
    assert ((p == 0) | (p == 2.0 ** -dims)).all()
    if b.tables["rhoL"]:
        assert cs.any()   # nontrivial C reached (cultivation_d5, rm15rep, toy2obs)


# ------------------------------------------------------------------------------------- layout: forced fired sets


def _given(t, fired_off, fired_ids):
    """pack() buffers (modes COIN_IN | GIVEN) for shots whose fired lines are the CSR (fired_off, fired_ids): rows =
    the XOR of the lines' flips, extended lists = their records"""
    n = len(fired_off) - 1
    W = int(t["W"])
    member = {int(l): m for m, l in enumerate(t["mem_line"])}
    flips = []
    for m in range(len(t["mem_line"])):
        x = np.zeros(W, np.uint64)
        beg = int(t["mem_beg"][m])
        for e in range(beg, beg + int(t["mem_cnt"][m])):
            x[int(t["ent_word"][e])] ^= np.uint64(t["ent_mask"][e])
        flips.append(x)
    lists = []
    rows = np.zeros((n, W), np.uint64)
    for s in range(n):
        ms = [member[int(i)] for i in fired_ids[fired_off[s]:fired_off[s + 1]]]
        for m in ms:
            rows[s] ^= flips[m]
        lists.append([int(t["mem_xo"][m]) for m in ms if t["mem_xo"][m]])
    nblocks = -(-n // B)
    E = max([sum(len(lists[s]) for s in range(b * B, min(n, b * B + B))) for b in range(nblocks)] + [1])
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=0, mode=gt.XK_MODE_COIN_IN | gt.XK_MODE_GIVEN)
    buf["rows"] = rows.reshape(-1)
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


def _emit_case(name, n=3000, launcher="ref"):
    """the CPU sampler's fired lines of ``name`` resolved by the kernels on ``launcher`` ("ref" or "metal"):
    (sampler, tables, launcher)"""
    s = xtim.XdemSampler(_read(name))
    _, status = s.sample(n, seed=31, fired=True)
    t = s._s.device_tables()
    buf, E = _given(t, *status.fired)
    if launcher == "ref":
        L = RefLauncher()
    else:
        from xtim.gpu.metal import MetalLauncher
        L = MetalLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    L.upload(buf)
    L.run("resolve", n)
    return s, t, L


@pytest.mark.parametrize("name", ["examples/xdem/rm15rep.xdem", "examples/xdem/toy2obs.xdem",
                                  "tests/data/xdem/synth/boundary_r65.xdem"])   # the last: multi-word keys
@pytest.mark.parametrize("part", ["none", "half", "all"])
@pytest.mark.parametrize("launcher", ["ref", pytest.param("metal", marks=pytest.mark.metal)])
def test_emit_looks_ids_up_in_ctab(name, part, launcher):
    """K4 with a C-id table (tables.cspace_table: key_hash and the probing mirror xk_cspace_lookup) holding no, half
    or all of the batch's distinct keys: a known key's shot gets the table's id, the others 0xFFFFFFFF and exactly
    one (row, shot) pair in cmiss; the backend's finishing gives the host oracle's ids either way"""
    s, t, L = _emit_case(name, launcher=launcher)
    n, kw = 3000, gt.key_words(t)
    cdim = L.download("cdim")[:n]
    keys = L.download("ckey")[:n * kw].reshape(n, kw)
    want = host_outputs(s._s, t, L.download("rows"), L.download("ckey"), cdim)[2]
    nz = np.flatnonzero(cdim)
    distinct = {keys[i].tobytes(): int(want[i]) for i in nz}
    assert len(distinct) >= (1 if "synth" in name else 2)   # boundary_r65: one C, of two-word keys
    pick = list(distinct)[:{"none": 0, "half": len(distinct) // 2, "all": len(distinct)}[part]]
    known = np.array([np.frombuffer(k, np.uint64) for k in pick], np.uint64).reshape(-1, kw)
    # 600 random keys ahead of them, among them one on each known key's home slot: every lookup of a known key probes
    m = 600 + len(known)
    nslots = 1 << (2 * m - 1).bit_length()
    pool = np.random.default_rng(1).integers(1, 1 << 63, (1 << 16, kw), dtype=np.uint64)
    home = gt.key_hash(pool) & np.uint64(nslots - 1)
    block = [pool[np.flatnonzero(home == (h & np.uint64(nslots - 1)))[0]] for h in gt.key_hash(known)]
    extra = np.concatenate([np.array(block, np.uint64).reshape(-1, kw), pool[-(600 - len(block)):]])
    ctab = gt.cspace_table(np.concatenate([extra, known]),
                           np.concatenate([np.arange(600, dtype=np.int32) + 10 ** 6,
                                           np.array([distinct[k] for k in pick], np.int32)]))
    assert int(ctab[0]) == nslots
    L.upload({"ctab": ctab})
    L.run("emit", n)
    cs = L.download("out_cs")[:n]
    inside = np.array([cdim[i] > 0 and keys[i].tobytes() in set(pick) for i in range(n)])
    assert (cs[inside] == want[inside].astype(np.uint32)).all()
    miss = (cdim > 0) & ~inside
    assert (cs[miss] == 0xFFFF_FFFF).all() and (cs[cdim == 0] == 0).all()
    cm = L.download("cmiss")
    pairs = cm[1:1 + 2 * int(cm[0])].reshape(-1, 2)
    assert sorted(pairs[:, 0].tolist()) == np.flatnonzero(miss).tolist() and (pairs[:, 0] == pairs[:, 1]).all()
    b = GpuBackend(s._s, L)
    out = (np.empty((n, b._nbD), np.uint8), np.empty((n, b._nbO), np.uint8), np.empty(n, np.int32))
    b._emitted(n, *out)
    assert np.array_equal(out[2], want)


@pytest.mark.parametrize("name,exact_rows", [("tests/data/xdem/synth/surface_d5r5.dem", True),
                                             ("examples/xdem/cultivation_d5.xdem", False),
                                             ("examples/xdem/rm15rep.xdem", False),
                                             ("examples/xdem/toy2obs.xdem", False)])
def test_layout_with_forced_fired_sets(name, exact_rows):
    # the CPU sampler's own fired lines through the kernel's resolve, emit and the backend's finishing of the C ids:
    # the CPU's cspace ids always (C depends on the fired lines only); the CPU's dets / obs bytes where no coin is
    # drawn
    s = xtim.XdemSampler(_read(name))
    n = 3000
    dets, status = s.sample(n, seed=31, fired=True)
    off, ids = status.fired
    t = s._s.device_tables()
    buf, E = _given(t, off, ids)
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    L.upload(buf)
    L.run("resolve", n)
    L.run("emit", n)
    b = GpuBackend(s._s, L)
    d, o, c = (np.empty((n, b._nbD), np.uint8), np.empty((n, b._nbO), np.uint8), np.empty(n, np.int32))
    b._emitted(n, d, o, c)
    for x, y in zip((d, o, c), host_outputs(s._s, t, L.download("rows"), L.download("ckey"), L.download("cdim"))):
        assert x.dtype == y.dtype and np.array_equal(x, y)
    assert d.shape == dets.shape and o.shape == status.obs.shape and c.dtype == status.cspace.dtype
    assert (c == status.cspace).all()
    if t["rhoL"]:
        assert status.cspace.any()
    if exact_rows:
        assert d.tobytes() == dets.tobytes() and o.tobytes() == status.obs.tobytes() and dets.any()


# ------------------------------------------------------------------------------------- statistics vs the CPU sampler


STAT_NAMES = [n for n in NAMES_INREPO if present(n)]
# 4e6 shots per side on every circuit (the reference build takes about a second per circuit, so none is `slow`)
STAT_SHOTS = 4_000_000
CHUNK = 500_000


def _acc_add(acc, dets, passv, D):
    for i in range(0, len(passv), CHUNK):
        acc.add(dets[i:i + CHUNK], passv[i:i + CHUNK], D)


def stat_compare(text, shots, seed_gpu, seed_cpu, threads=4, launcher=None, **kw):
    """the GPU backend (reference build, or `launcher`; GpuBackend's keywords `kw`) vs the CPU sampler on `shots`
    shots each: pass mean (O >= 1), every detector marginal, every detector x failure joint (O >= 1), the z-scores of
    test_xdem_cross_route.py::cross_compare"""
    S = _xtim._XdemSampler(text, 65536)
    D, O = S.num_detectors, S.num_observables
    A, C = _Acc(D), _Acc(D)
    zero = np.zeros((shots, O), np.uint8)
    t0 = time.time()
    d, o, cs = GpuBackend(S, launcher or RefLauncher(), **kw).sample(shots, seed_gpu)
    t_gpu = time.time() - t0
    _acc_add(A, d, S.score(o, cs, zero), D)
    t0 = time.time()
    d, o, cs = S.sample(shots, seed_cpu, threads, 0, False)
    t_cpu = time.time() - t0
    _acc_add(C, d, S.score(o, cs, zero), D)
    z_pass = _z_mean(A.s, A.ss, C.s, C.ss, shots) if O else None
    z_det = [_z_prop(int(a), int(c), shots) for a, c in zip(A.det, C.det)]
    z_dxf = [_z_mean(A.F[j], A.FF[j], C.F[j], C.FF[j], shots) for j in range(D)] if O else []
    named = ([("pass", z_pass)] if O else []) + [(f"D{j}", z) for j, z in enumerate(z_det)] + \
        [(f"D{j}xfail", z) for j, z in enumerate(z_dxf)]
    arg, zmax = max(named, key=lambda kz: abs(kz[1])) if named else (None, 0.0)
    return dict(D=D, O=O, pass_gpu=A.s / shots, pass_cpu=C.s / shots, z_pass=z_pass, n_tests=len(named),
                max_abs_z=abs(zmax), argmax=arg, t_gpu=t_gpu, t_cpu=t_cpu)


@pytest.mark.parametrize("name", STAT_NAMES)
def test_statistics_vs_cpu(name):
    try:
        text = xdem_export(circuit(name))
    except xtim.XtimDemError as e:   # code_switching_faithful / _rate: a random observable (tests/test_xdem_export.py)
        pytest.skip(f"{name}: refused by the xdem export ({str(e).splitlines()[0]})")
    if not xtim.XdemSampler(text)._s.device_tables()["fast"]:
        pytest.skip(f"{name}: not on the kernels' one-word fast path")
    i = STAT_NAMES.index(name)
    r = stat_compare(text, STAT_SHOTS, 71_000 + i, 72_000 + i)
    msg = (f"{name}: D={r['D']} O={r['O']} 1-pass gpu {1 - r['pass_gpu']:.6g} cpu {1 - r['pass_cpu']:.6g} "
           f"z_pass {r['z_pass']} | {r['n_tests']} tests, max|z| {r['max_abs_z']:.2f} at {r['argmax']} | "
           f"gpu(ref) {r['t_gpu']:.1f}s cpu {r['t_cpu']:.1f}s")
    print(msg)
    assert r["n_tests"] >= r["D"]
    assert r["max_abs_z"] <= Z_MAX, msg
