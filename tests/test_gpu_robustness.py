"""The GPU sampler under stress (final review of the GPU xdem branch): a sampler shared between threads (calls
serialised, each giving its single-threaded result), pair capacities past 2^17 (the walk's list is a device buffer,
no thread-local array grows with it), batches bounded by their expected work (a heavy DEM gets small batches; no byte
changes), KeyboardInterrupt between submissions, and every device failure surfacing as ``XtimGpuError``. Through the
kernels' C++ reference build (`RefLauncher`) unless marked `metal`.
"""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import xtim
import xtim.gpu
import xtim.gpu.backend as GB
from xtim.gpu import tables as gt
from xtim.gpu.backend import DEFAULT_BLOCK, GpuBackend

from _gpu_compare import heavy_dem, heavy_ext
from _gpu_ref import RefLauncher
from test_gpu_postselect import _mask

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)
D5 = (ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text()
SURFACE = (ROOT / "tests" / "data" / "xdem" / "synth" / "surface_d5r5.dem").read_text()


def _metal():
    from xtim.gpu.metal import MetalLauncher
    return MetalLauncher()


def _same(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b))


def _gpu_sampler(monkeypatch, text, launcher=RefLauncher):
    """XdemSampler(text, device="gpu") on the launcher class `launcher` (the device list replaced)"""
    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", launcher),))
    return xtim.XdemSampler(text, device="gpu")


# ------------------------------------------------------------------------------------------------ A1: threads


def _jobs(s):
    D, O = s.num_detectors, s.num_observables
    mask = _mask(D, "half").astype(bool)

    def dec(dets):   # a decoder that reads its input, so a wrong row changes the score
        return np.zeros((len(dets), O), bool) | dets[:, :1]

    def cs_rows(st):   # cspace ids index the shared table: compare each shot's C itself
        return [s._s.cspaces(int(i), 1)[0].tobytes() for i in st.cspace]

    def sample(seed):
        d, st = s.sample(3000, seed=seed)
        return d.tobytes(), st.obs.tobytes(), cs_rows(st)

    def sample_ps(seed):
        d, st, counts = s.sample(3000, seed=seed, postselect=mask)
        return d.tobytes(), st.obs.tobytes(), cs_rows(st), sorted(counts.items())

    def inf_none(seed):
        return s.infidelity(5000, None, seed=seed, postselect=mask)

    def inf_dec(seed):   # chunked: several backend calls per result, interleaved with the other threads'
        return s.infidelity(5000, dec, seed=seed, batch=1024)

    return {"sample": sample, "sample_ps": sample_ps, "inf_none": inf_none, "inf_dec": inf_dec}


def _threaded(s, threads=4, seeds=(11, 12, 13)):
    """every (job, seed) single-threaded, then the same from `threads` threads at once, each thread every pair in its
    own order: every result equal"""
    jobs = _jobs(s)
    want = {(j, sd): f(sd) for j, f in jobs.items() for sd in seeds}
    got, errors = [], []
    start = threading.Barrier(threads)

    def worker(k):
        try:
            start.wait()
            order = list(want)
            np.random.default_rng(k).shuffle(order)
            for j, sd in order * 2:
                got.append(((j, sd), jobs[j](sd)))
        except BaseException as e:   # reported below, not lost in the thread
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
    assert len(got) == threads * 2 * len(want)
    bad = [key for key, v in got if v != want[key]]
    assert not bad, bad[:5]


def test_shared_sampler_threads_get_single_threaded_results(monkeypatch):
    s = _gpu_sampler(monkeypatch, D5)
    _threaded(s)
    assert not s._gpu._lock.locked()


@pytest.mark.metal
def test_shared_sampler_threads_metal():
    s = xtim.XdemSampler(D5, device="metal")
    _threaded(s, threads=6)


def test_backend_lock_covers_every_public_call():
    b = GpuBackend(xtim.XdemSampler(SURFACE)._s, RefLauncher())
    seen = []

    class Probe:
        def __enter__(self):
            seen.append(True)

        def __exit__(self, *exc):
            return False

    b._lock = Probe()
    b.prepare()
    b.sample(10, 1)
    b.sample_ps(10, 1, np.ones(b.tables["D"], np.uint8))
    b.estimate(10, 1, None)
    b.counters()
    assert len(seen) == 5


# ------------------------------------------------------------------------------------------------ B1: pair capacity


def _all_kernels(L, t, E, n, seed):
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=seed)
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    L.upload(buf)
    nb = -(-n // B)
    L.run_batch([("walk", nb), ("resolve", n), ("reduce", nb), ("reduce_groups", gt.XK_HIST_GROUPS *
                                                                    gt.XK_HIST_TOTAL_COUNT),
                 ("reduce2", gt.XK_HIST_TOTAL_COUNT)])
    return {k: L.download(k) for k in buf}


@pytest.mark.parametrize("launcher", ["ref", pytest.param("metal", marks=pytest.mark.metal)])
def test_walk_at_pair_capacity_2_18(launcher):
    """the walk compiled and run at XK_CAP_E = E = 2^18 (Metal refused 2^17 when the list was thread-local): every
    buffer equals the reference at that capacity, and rows / keys / dims equal those at the default capacity"""
    t = xtim.XdemSampler(D5)._s.device_tables()
    n, E = 40 * B - 9, 1 << 18
    ref = _all_kernels(RefLauncher(), t, E, n, 5)
    got = ref if launcher == "ref" else _all_kernels(_metal(), t, E, n, 5)
    for k in ref:
        if k != "psh":   # K1's scratch, in the order of its atomics
            assert got[k].tobytes() == ref[k].tobytes(), k
    small = _all_kernels(RefLauncher(), t, GpuBackend(xtim.XdemSampler(D5)._s, RefLauncher())._pair_cap, n, 5)
    for k in ("rows", "ckey", "cdim", "flag", "hist_total"):
        assert small[k].tobytes() == ref[k].tobytes(), k


@pytest.mark.parametrize("launcher", ["ref", pytest.param("metal", marks=pytest.mark.metal)])
def test_relaunch_grows_pair_capacity_past_2_17(launcher):
    """blocks of 512 shots of heavy_ext(1000) fire about 154k extended lines each: from pair_cap 2^16 the backend
    relaunches up to 2^18 (recompiling), and the bytes are those of a backend that started large enough"""
    text = heavy_ext(1000)
    block = 512
    s = xtim.XdemSampler(text)
    want = GpuBackend(s._s, RefLauncher(), block=block).sample(2 * block + 5, 3)
    b = GpuBackend(s._s, RefLauncher() if launcher == "ref" else _metal(), block=block)
    b._pair_cap = 1 << 16
    got = b.sample(2 * block + 5, 3)
    assert b.relaunches >= 2 and b._pair_cap >= 1 << 18 and b._caps["XK_CAP_E"] == b._pair_cap
    assert _same(got, want)


@pytest.mark.metal
def test_metal_heavy_extended_file_samples():
    """about 510 fired extended lines per shot: a default pair capacity near 2^17 at the first sample (the case that
    exceeded Metal's stack with a thread-local list); the bytes are the reference's"""
    text = heavy_ext(1700)
    s = xtim.XdemSampler(text, device="metal")
    assert s._gpu._pair_cap > 100_000
    d, st = s.sample(3000, seed=2)
    want = GpuBackend(xtim.XdemSampler(text)._s, RefLauncher()).sample(3000, 2)
    assert _same((d, st.obs), want[:2]) and (st.cspace == want[2]).all()


def test_local_arrays_are_bounded():
    """only XK_CAP_WA, XK_CAP_RHO (both at most 64), XK_CAP_DIMC (at most 253) and the structured law's XK_CAP_ST_NC,
    _NG, _T (at most 8, 8, 32) size thread-local arrays (an #error otherwise; no array is sized by rho * CW: the
    multi-word law's scratch is the device buffer mws), and a file with more than 64 axis words or a status C past 253
    dimensions is refused by supports() and the backend"""
    src = HEADER.read_text()
    code = "\n".join(x.split("//")[0] for x in src.splitlines() if not x.lstrip().startswith("#"))
    import re
    sized = set(re.findall(r"\[(XK_CAP_\w+)[^\]]*\]", code))   # every array bound that involves a capacity
    assert sized == {"XK_CAP_WA", "XK_CAP_RHO", "XK_CAP_DIMC", "XK_CAP_ST_NC", "XK_CAP_ST_NG", "XK_CAP_ST_T"}, sized
    assert not re.search(r"\[[^\]]*XK_CAP_CW", code)
    assert "#if XK_CAP_WA > XK_MAX_WA || XK_CAP_RHO > XK_MAX_RHO" in src
    assert "#if XK_CAP_DIMC < 64 || XK_CAP_DIMC > XK_MAX_DIMC" in src
    assert ("#if XK_CAP_ST_NC > XK_MAX_ST_NC || XK_CAP_ST_NG > XK_MAX_ST_NG || XK_CAP_ST_T > XK_MAX_ST_T || "
            "XK_CAP_ST_N > XK_MAX_ST_N") in src
    assert (gt.XK_MAX_ST_NC, gt.XK_MAX_ST_NG, gt.XK_MAX_ST_T, gt.XK_MAX_ST_N) == (8, 8, 32, 16)
    t = dict(xtim.XdemSampler(D5)._s.device_tables())
    t["WA"], t["A"] = gt.XK_MAX_WA + 1, 64 * gt.XK_MAX_WA + 1
    with pytest.raises(ValueError, match="at most 4096 axes"):
        gt.capacities(t, B, 64)

    class Fake:
        def device_tables(self):
            return t
    ok, reason = xtim.gpu._supports_cpu(Fake())
    assert not ok and "4097 axes" in reason and "device='cpu'" in reason
    t = dict(xtim.XdemSampler(D5)._s.device_tables())
    t["rhoL"] = gt.XK_MAX_DIMC + 1
    with pytest.raises(ValueError, match="dim C up to 253"):
        gt.capacities(t, B, 64)
    ok, reason = xtim.gpu._supports_cpu(Fake())
    assert not ok and "rhoL = 254" in reason and "device='cpu'" in reason


# ------------------------------------------------------------------------------------------------ B2: batch work


class _Dry(RefLauncher):
    """records every submission's steps and runs nothing (hist_total: no survivor, no overflow): batch sizing only"""

    def __init__(self):
        super().__init__()
        self.steps = []

    def run_batch(self, steps):
        self.steps.append(list(steps))
        self._buf["hist_total"][:] = 0


def test_heavy_dem_batches_are_bounded_by_work():
    """a DEM firing 4000 lines per shot: (2 + 1.3 * 4000) * 512 work per block, so BATCH_WORK = 1e9 gives 375 blocks
    (192000 shots, about 1.7 s on the M3 Max) per submission instead of 2^24 shots (about 85 s); the bundled
    cultivation_d5 keeps 2^24"""
    s = xtim.XdemSampler(heavy_dem())
    L = _Dry()
    b = GpuBackend(s._s, L)
    assert round(float(b.tables["mean_fired"])) == 4000
    b.estimate(1_000_000, 1, None)
    sizes = [st[1][1] for st in L.steps]   # resolve's threads = the batch's shots
    assert sizes[:-1] == [375 * B] * 5 and sizes[-1] == 1_000_000 - 5 * 375 * B
    assert all(n * b._shot_work() <= GB.BATCH_WORK for n in sizes)
    d5 = GpuBackend(xtim.XdemSampler(D5)._s, _Dry())
    assert d5._batch_blocks() * B == GB.DEFAULT_BATCH_SHOTS < GB.BATCH_WORK / d5._shot_work()
    # max_batch_bytes still bounds by memory, below the work bound
    L = _Dry()
    b = GpuBackend(s._s, L, max_batch_bytes=10 * GpuBackend(s._s, RefLauncher())._block_bytes())
    b.estimate(100 * B, 1, None)
    assert {st[1][1] for st in L.steps} == {10 * B}


@pytest.mark.parametrize("text", [heavy_dem(lines=20_000, p=0.2), heavy_ext(40)], ids=["heavy_dem", "heavy_ext"])
def test_heavy_files_independent_of_batch_size(text, monkeypatch):
    """the work bound only cuts batches: a small BATCH_WORK (several batches), one block per batch and one batch give
    the bytes of sample, sample_ps and estimate"""
    s = xtim.XdemSampler(text)
    flags = _mask(s.num_detectors, "half")
    n = 20 * B + 7

    def outs(**kw):
        """(digest-ready outputs, batches run) of the four entry points on a fresh backend (each cspace id as its C)"""
        S = xtim.XdemSampler(text)._s
        b = GpuBackend(S, RefLauncher(), **kw)
        d, o, c = b.sample(n, 4, first_block=2)
        pd, po, pc, counts = b.sample_ps(n, 4, flags, first_block=2)
        cs = [S.cspaces(int(i), 1)[0].tobytes() for i in np.concatenate([c, pc])]
        e0, e1 = b.estimate(n, 4, None), b.estimate(n, 4, flags)
        return [d.tobytes(), o.tobytes(), pd.tobytes(), po.tobytes(), cs, sorted(counts.items()), e0[:2],
                e0[2].tolist(), e1[:2], e1[2].tolist()], b.batches_run

    one, nb1 = outs()
    monkeypatch.setattr(GB, "BATCH_WORK", 3.5 * B * GpuBackend(s._s, RefLauncher())._shot_work())
    few, nb3 = outs()
    tiny, nb_tiny = outs(max_batch_bytes=1)
    assert nb1 == 4 and nb3 == 4 * 7 and nb_tiny == 4 * 21   # 21 blocks: one batch, batches of 3, of 1
    assert few == one and tiny == one


class _Interrupt(RefLauncher):
    """raises KeyboardInterrupt at its k-th submission (once)"""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def run_batch(self, steps):
        self.k -= 1
        if self.k == 0:
            raise KeyboardInterrupt
        super().run_batch(steps)


@pytest.mark.parametrize("entry", ["sample", "sample_ps", "estimate"])
def test_keyboard_interrupt_between_submissions_leaves_the_backend_usable(entry):
    s = xtim.XdemSampler(D5)
    flags = _mask(s.num_detectors, "half")
    n = 30 * B + 3

    def call(b):
        if entry == "sample":
            return b.sample(n, 7)
        if entry == "sample_ps":
            return b.sample_ps(n, 7, flags)[:3]
        a, p, h = b.estimate(n, 7, flags)
        return np.array([a, p]), h

    want = call(GpuBackend(s._s, RefLauncher(), max_batch_bytes=1))
    b = GpuBackend(s._s, _Interrupt(3), max_batch_bytes=1)   # one block per batch: interrupted mid-call
    with pytest.raises(KeyboardInterrupt):
        call(b)
    assert not b._lock.locked()
    assert _same(call(b), want)   # the same backend afterwards: the full, correct result
    assert _same(call(b), want)


# ------------------------------------------------------------------------------------------------ B3: device errors


class _Failing(RefLauncher):
    """the reference launcher raising `exc` at its first call of `stage`"""
    stage, exc = None, None

    def __init__(self):
        super().__init__()
        for name in ("compile", "upload", "alloc", "run_batch", "download", "view", "free_bytes"):
            if name == self.stage:
                setattr(self, name, self._raise)

    def _raise(self, *a, **k):
        raise self.exc


def _failing(stage, exc):
    return type("F", (_Failing,), {"stage": stage, "exc": exc})


@pytest.mark.parametrize("stage,exc,when", [
    ("compile", RuntimeError("Metal pipeline for 'walk' failed: Compute function exceeds available stack space"),
     "construction"),
    ("free_bytes", RuntimeError("device lost"), "construction"),
    ("upload", MemoryError("Metal could not allocate a buffer of 4096 bytes"), "construction"),
    ("alloc", MemoryError("Metal could not allocate a buffer of 999 bytes"), "sample"),
    ("run_batch", RuntimeError("Metal kernel 'walk' failed: Caused GPU Timeout Error"), "sample"),
    ("download", RuntimeError("CUDARuntimeError: cudaErrorIllegalAddress"), "sample"),
    ("view", ValueError("view out of range"), "sample"),
])
def test_device_failures_are_xtim_gpu_errors(monkeypatch, stage, exc, when):
    L = _failing(stage, exc)
    if when == "construction":
        with pytest.raises(xtim.XtimGpuError) as e:
            _gpu_sampler(monkeypatch, D5, L)
    else:
        s = _gpu_sampler(monkeypatch, D5, L)
        with pytest.raises(xtim.XtimGpuError) as e:
            s.sample(1000, seed=1)
    msg = str(e.value)
    assert str(exc) in msg and 'device="cpu"' in msg and type(exc).__name__ in msg
    if isinstance(exc, MemoryError):
        assert "bytes requested" in msg and "fewer shots per call is not the fix" in msg


def test_opener_failure_is_an_xtim_gpu_error(monkeypatch):
    def opener():
        raise MemoryError("Metal could not allocate a buffer of 16 bytes")
    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", opener),))
    with pytest.raises(xtim.XtimGpuError, match=r"could not be set up: MemoryError.*device=\"cpu\""):
        xtim.XdemSampler(SURFACE, device="metal")


def test_invariant_errors_pass_through(monkeypatch):
    class Inject(RefLauncher):   # the code written into the batch's middle shot before "emit" reads it
        def run(self, kernel, n, lanes=None):
            if kernel == "emit":
                self._buf["cdim"][int(self._buf["dims"][gt.XK_DIM_NSHOTS]) // 2] = gt.XK_CDIM_INTERNAL
            super().run(kernel, n, lanes)
    s = _gpu_sampler(monkeypatch, (ROOT / "examples" / "xdem" / "rm15rep.xdem").read_text(), Inject)
    with pytest.raises(xtim.XtimInvariantError, match="internal"):
        s.sample(500, seed=1)


@pytest.mark.metal
def test_metal_compile_failure_at_construction_is_an_xtim_gpu_error(monkeypatch):
    monkeypatch.setattr(GB, "_SOURCE", type("P", (), {"read_text": lambda self: "XK_FN u32 broken( { }\n"})())
    with pytest.raises(xtim.XtimGpuError, match=r"(?s)kernel compile.*Metal compile.*device=\"cpu\""):
        xtim.XdemSampler(SURFACE, device="metal")


# ------------------------------------------------------------------------------------------------ minor findings


def test_counters_on_a_gpu_sampler(monkeypatch):
    s = _gpu_sampler(monkeypatch, D5)
    cpu = xtim.XdemSampler(D5)
    c = s.counters()
    assert set(cpu.counters()) <= set(c)
    assert c["shots"] == 0 and c["device"] == "metal" and c["gpu_batches"] == 0
    s.sample(1000, seed=1)
    s.sample(500, seed=1, postselect=_mask(s.num_detectors, "half").astype(bool))
    s.infidelity(300, None, seed=2)
    c = s.counters()
    assert c["shots"] == 1800 and c["gpu_batches"] == 3 and c["gpu_relaunches"] == 0
    assert c["ext_shots"] is None and c["coin_shots"] is None and c["gpu_pair_cap"] == 2406
    assert "device" not in cpu.counters() and "gpu_batches" not in cpu.counters()   # the CPU dict is unchanged


def test_repr_names_a_gpu_device(monkeypatch):
    cpu = xtim.XdemSampler(D5)
    s = _gpu_sampler(monkeypatch, D5)
    assert repr(s) == repr(cpu)[:-1] + ", device='metal')"
    assert "device" not in repr(cpu)


def test_infidelity_chunks_are_multiples_of_1024(monkeypatch):
    asked = []

    class Recording(GB.GpuBackend):
        def sample(self, shots, seed, *, first_block=0):
            asked.append((shots, first_block))
            return super().sample(shots, seed, first_block=first_block)

    monkeypatch.setattr(GB, "GpuBackend", Recording)
    s = _gpu_sampler(monkeypatch, SURFACE)
    O = s.num_observables
    r = s.infidelity(2500, lambda d: np.zeros((len(d), O), bool), seed=3, batch=100)   # 100 -> 1024, not 128
    assert asked == [(1024, 0), (1024, 1024 // B), (452, 2048 // B)]
    assert r == s.infidelity(2500, lambda d: np.zeros((len(d), O), bool), seed=3, batch=10**6)


@pytest.mark.parametrize("slots,J", [(1, 1), (2, 1), (3, 2), (1024, 10), (1025, 11), (2**49 + 1, 50),
                                     (2**53 + 1, 54)])
def test_geo_thresholds_J_is_exact(slots, J):
    got, thr = gt.geo_thresholds(0.25, slots)
    assert got == J and len(thr) == J + 1 and 2**got >= slots


def test_mask_tables_cache_is_least_recently_used():
    b = GpuBackend(xtim.XdemSampler(D5)._s, RefLauncher())
    D = b.tables["D"]
    masks = [np.eye(D, dtype=np.uint8)[i] for i in range(GB._MASKS_KEPT + 1)]
    for m in masks[:GB._MASKS_KEPT]:
        b._tables(m)
    first = b._tables(masks[0])[1]   # used again: now the most recent
    b._tables(masks[GB._MASKS_KEPT])   # evicts masks[1], the least recently used
    assert b._tables(masks[0])[1] is first
    assert masks[1].tobytes() not in b._static and len(b._static) == GB._MASKS_KEPT


def test_cuda_import_failure_is_an_xtim_gpu_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def imp(name, *a, **k):
        if name == "cupy":
            raise OSError("libnvrtc.so.12: cannot open shared object file")
        return real(name, *a, **k)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(builtins, "__import__", imp)
    with pytest.raises(xtim.XtimGpuError, match=r"CuPy failed to import \(OSError: libnvrtc"):
        xtim.XdemSampler(SURFACE, device="cuda")


def test_metal_refused_off_arm64(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setitem(sys.modules, "cupy", None)
    from xtim.gpu.metal import MetalLauncher
    with pytest.raises(xtim.XtimGpuError, match=r"x86_64, not an arm64"):
        MetalLauncher()
    assert xtim.gpu.available() is None
    with pytest.raises(xtim.XtimGpuError, match="arm64"):
        xtim.XdemSampler(SURFACE, device="metal")


@pytest.mark.parametrize("name", ["cultivation_d5", "heavy_ext", "bh_like36"])
@pytest.mark.parametrize("block", [64, 48])
def test_packed_walk_records_same_lists(name, block):
    """K1's packed records (XK_DIM_RECSH: offset | shot << shift in psh, one scatter into eids) give the rows, epos
    and eids of the unpacked records (shot in psh, offset in eids, the cycle permutation), on the reference build and
    on Metal; a block size that is not a power of two too"""
    from _xdem_synth import bh_like_text
    text = {"cultivation_d5": (Path(__file__).parent.parent / "examples" / "xdem" / "cultivation_d5.xdem").read_text(),
            "heavy_ext": heavy_ext(60), "bh_like36": bh_like_text(36, 300, seed=3)}[name]
    t = xtim.XdemSampler(text)._s.device_tables()
    assert gt.rec_shift(t, block) == 32 - (block - 1).bit_length()
    assert gt.rec_shift(t, 1) == 0
    E = int(block * t["mean_ext"] + 10 * np.sqrt(block * t["mean_ext"] + 1) + 64)
    n = 37 * block - 5
    launchers = [RefLauncher]
    try:
        from xtim.gpu.metal import MetalLauncher
        MetalLauncher()
        launchers.append(MetalLauncher)
    except Exception:
        pass
    for new in launchers:
        out = []
        for packed in (True, False):
            buf = gt.pack(t, block=block, pair_cap=E, shots=n, seed=5)
            if not packed:
                buf["dims"][gt.XK_DIM_RECSH] = 0
            L = new()
            L.compile(HEADER.read_text(), gt.capacities(t, block, E))
            L.upload(buf)
            L.run("walk", int(buf["dims"][gt.XK_DIM_NBLOCKS]))
            out.append({k: L.download(k) for k in ("rows", "epos", "eids", "overflow")})
        a, b = out
        assert not a["overflow"].any()
        nb = int(buf["dims"][gt.XK_DIM_NBLOCKS])
        assert np.array_equal(a["rows"][:n * int(t["W"])], b["rows"][:n * int(t["W"])])
        assert np.array_equal(a["epos"], b["epos"]) and a["epos"].reshape(nb, block + 1)[:, -1].sum() > 0
        ep = a["epos"].reshape(nb, block + 1)
        for k in range(nb):   # eids: each block's first epos[B] records
            m = int(ep[k, -1])
            assert np.array_equal(a["eids"][k * E:k * E + m], b["eids"][k * E:k * E + m]), (new, k)
