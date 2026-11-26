"""The GPU backend's launch loop (Task 9b): static tables uploaded once per backend (and mask), working and output
buffers allocated once and reused, several kernels per submission (``Launcher.run_batch``), and the estimate's
counters reduced on the device ("reduce_groups" and "reduce2": one ``hist_total`` per batch).

None of this may change an output bit: golden hashes of sample / sample_ps / estimate (pinned, see GOLDEN), reused buffers pre-filled with sentinels giving the bytes of a fresh backend, a partial last batch giving the
bytes of an exact-size run. Through the kernels' C++ reference build (`RefLauncher`) unless marked `metal`.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import tables as gt
from xtim.gpu.backend import DEFAULT_BLOCK, GpuBackend

from _gpu_ref import RefLauncher
from test_gpu_postselect import _mask

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)
HS = gt.XK_HIST_STRIDE
PS = gt.XK_MODE_POSTSELECT

# D % 64 != 0 and O % 64 != 0 in every file; nontrivial C (cultivation_d5, rm15rep, toy2obs); many extended lines
# per block (ccz_toy_obs_p5); rhoL = 0 (cultivation_d3_faithful); a conventional DEM; r = 64 (the last one-word file)
# and r = 65 (two coordinate words: the multi-word law)
FILES = ["examples/xdem/cultivation_d5.xdem", "examples/xdem/cultivation_d3_faithful.xdem",
         "examples/xdem/rm15rep.xdem", "examples/xdem/toy2obs.xdem", "examples/xdem/ccz_toy_obs_p5.xdem",
         "tests/data/xdem/synth/surface_d5r5.dem", "tests/data/xdem/synth/boundary_r64.xdem",
         "tests/data/xdem/synth/boundary_r65.xdem"]
MASKS = [None, "all", "half"]


def _read(name):
    return (ROOT / name).read_text()


@pytest.fixture(scope="module", params=FILES, ids=[Path(f).name for f in FILES])
def path(request):
    return request.param


@pytest.fixture(params=MASKS, ids=[str(m) for m in MASKS])
def mask(request):
    return request.param


def _flags(s, kind):
    return None if kind is None else _mask(s.num_detectors, kind)


def _backend(text, launcher=None, **kw):
    s = xtim.XdemSampler(text)
    return s, GpuBackend(s._s, launcher or RefLauncher(), **kw)


def _metal():
    from xtim.gpu.metal import MetalLauncher
    return MetalLauncher()


class _Recording(RefLauncher):
    """the reference launcher recording every upload (bytes by name, per call) and download (bytes per call)"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.uploads: list[tuple[str, int]] = []
        self.downloads: list[tuple[str, int]] = []

    def upload(self, buffers):
        self.uploads += [(k, np.asarray(v).nbytes) for k, v in buffers.items()]
        super().upload(buffers)

    def download(self, name, count=None):
        a = super().download(name, count)
        self.downloads.append((name, a.nbytes))
        return a

    def bytes_up(self, names=None):
        return sum(n for k, n in self.uploads if names is None or k in names)

    def count_up(self, name):
        return sum(1 for k, _ in self.uploads if k == name)


# ------------------------------------------------------------------------------------------------ golden outputs

GOLDEN_SHOTS = 4 * 1024 + 37   # off the block grid
GOLDEN_SEED = 2026


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(f"{p.dtype.str}{p.shape}".encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def _golden_outputs(text, kind, launcher=None, **kw) -> str:
    """sha256 of every entry point's outputs for (file, mask kind): sample (with and without a first_block offset)
    and estimate(None) without a mask; sample_ps (both offsets) and estimate(flags) with one. A fresh sampler per
    entry point (cspace ids index the sampler's growing table of C)."""
    n, seed = GOLDEN_SHOTS, GOLDEN_SEED
    out = []
    new = launcher or RefLauncher
    for first_block in (0, 5):
        s, b = _backend(text, new(), **kw)
        flags = _flags(s, kind)
        if kind is None:
            out.append(_digest(*b.sample(n, seed, first_block=first_block)))
        else:
            d, o, c, counts = b.sample_ps(n, seed, flags, first_block=first_block)
            out.append(_digest(d, o, c, sorted(counts.items())))
        s, b = _backend(text, new(), **kw)
        att, passed, hist = b.estimate(n, seed, _flags(s, kind), first_block=first_block)
        out.append(_digest(att, passed, hist))
    return _digest(*out)


# re-recorded when the walk became a group kernel over segments of chains (its own stream per item, blocks of
# DEFAULT_BLOCK shots; each hash checked equal on the reference launcher and on Metal, default batches and one block
# per batch, before it was pinned)
GOLDEN = {
    ('cultivation_d5.xdem', None): "5b2971ef0a906ad0b123f22586eadf30a8f1129e17998c423e9e9bf7337f8795",
    ('cultivation_d5.xdem', 'all'): "57c363658e3c3c14f8eb83fcd01a667baaa713816751023fc6331c375ffad282",
    ('cultivation_d5.xdem', 'half'): "ebffaf6cb7a8c022d864f92eb8db12f1b87f9b50c203bef06af99ac7efb34c39",
    ('cultivation_d3_faithful.xdem', None): "5ada54d092f782567433b4de89dd415554841f1950c04f69bd1c109de5049886",
    ('cultivation_d3_faithful.xdem', 'all'): "1b0ea5bbfc7f050fa6ace77e032d1ef68ba54923a7fd3fccfed0c2d1f4a47928",
    ('cultivation_d3_faithful.xdem', 'half'): "c7c1aa5aa71b83c5b965b0eb8f4efcccae3671bad7ae457857af5425a72e1805",
    ('rm15rep.xdem', None): "bc51d62dca46eeacad1c755ec8a1aa1be2815111b892fa66b432305b1eb426cb",
    ('rm15rep.xdem', 'all'): "4019f3a5775766d74ba6198590d5f7aed66386ed09f3f0a1b142b8b8c87a2f3e",
    ('rm15rep.xdem', 'half'): "f008a13573cc80ac618060375066f4e2684969f9751dd4e8f36ffffe0125f87e",
    ('toy2obs.xdem', None): "d4ed713f46279910c6e6dcd82e356345a7eefdbec98b1108cf62a5200c67ad78",
    ('toy2obs.xdem', 'all'): "db041b78541014a848d569bd98a97c0e6902233d20e436aea01a3b8b4c453beb",
    ('toy2obs.xdem', 'half'): "da5f3572c6b73b27e599a437ca2b4f135d9b334dfdf6d78c47a55516a20edd0c",
    ('ccz_toy_obs_p5.xdem', None): "ed14df5c18692fe4e27d43b5b73f9a633a49b308ecf856d5fec769be2b97997e",
    ('ccz_toy_obs_p5.xdem', 'all'): "5ee8ff7fb4ed79de6750be89554a04e7dc300bd6d94563b2b1e0c75c7b23daa3",
    ('ccz_toy_obs_p5.xdem', 'half'): "5ee8ff7fb4ed79de6750be89554a04e7dc300bd6d94563b2b1e0c75c7b23daa3",
    ('surface_d5r5.dem', None): "c1e8dd2a6fcee5dce9ea69ed3d939749e59afe86a87560f046ba5219fe261a83",
    ('surface_d5r5.dem', 'all'): "281d6975b55b5438a5a111334c813839a95c6f2f085163b2e4aba7ebda9b030b",
    ('surface_d5r5.dem', 'half'): "4304a6a498d5930c4b86d1f146a135e9bc40eb859989ff22e13c5f5c1332fe74",
    ('boundary_r64.xdem', None): "b1c5f489e1344c28ec2b276b4a397fb64fbb9edf3ec35e001ca76ab5c19731b1",
    ('boundary_r64.xdem', 'all'): "d1dd863ce12c76c87e278a7de205dcd85acd1aaab6c167c7a5e25622328a5f9a",
    ('boundary_r64.xdem', 'half'): "520336aedee0905e01aa835b84742d545673a3653b62bee7afb5ad7aa93338b6",
    ('boundary_r65.xdem', None): "d4b8e0e69b442a5220de680daebff2db21dd8eb0f65840caf60380a7a8dc0971",
    ('boundary_r65.xdem', 'all'): "e79da8fcb4a11e98dbbbbb9970b359bd5a9dc0e65dd2d676beead6703aff8dca",
    ('boundary_r65.xdem', 'half'): "3539b2fe2142f8bb12cb7a721310cfe94a68d298b6f07bc590f97e5ca50ccb43",
}


def test_outputs_unchanged_by_the_launch_loop(path, mask):
    text = _read(path)
    want = GOLDEN[(Path(path).name, mask)]
    assert _golden_outputs(text, mask) == want
    # many small batches give the same bytes
    assert _golden_outputs(text, mask, max_batch_bytes=1) == want


@pytest.mark.metal
def test_metal_launch_loop_matches_reference(path, mask):
    """every entry point on Metal: the golden bytes of the reference launcher (default batches and many small ones)"""
    text = _read(path)
    want = GOLDEN[(Path(path).name, mask)]
    assert _golden_outputs(text, mask, launcher=_metal) == want
    assert _golden_outputs(text, mask, launcher=_metal, max_batch_bytes=1) == want


# ------------------------------------------------------------------------------------------------ uploads, downloads

ENTRY_POINTS = ["sample", "sample_ps", "estimate", "estimate_ps"]


def _call(b, s, entry, shots, seed):
    flags = _mask(s.num_detectors, "half")
    if entry == "sample":
        return b.sample(shots, seed)
    if entry == "sample_ps":
        return b.sample_ps(shots, seed, flags)
    return b.estimate(shots, seed, flags if entry == "estimate_ps" else None)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_tables_uploaded_once(path, entry):
    """20 batches in each of two calls: the static tables (span tables included) go up once per (backend, mask);
    per batch, only dims and seed are uploaded, and emit's C-id table (ctab, sample and sample_ps only) only when it
    grew: at most once per new key, besides the first (empty) one"""
    s = xtim.XdemSampler(_read(path))
    L = _Recording()
    b = GpuBackend(s._s, L, max_batch_bytes=1)   # one block per batch
    for seed in (3, 4):
        _call(b, s, entry, 20 * B - 7, seed)
    assert b.batches_run == 40
    assert L.count_up("tab64") == 1 and L.count_up("tab32") == 1
    per_batch = L.bytes_up() - L.bytes_up({"tab64", "tab32", "ctab"})
    assert per_batch / b.batches_run < 1024
    assert set(k for k, _ in L.uploads) - {"tab64", "tab32", "dims", "seed", "ctab"} == set()
    if entry.startswith("estimate"):
        assert L.count_up("ctab") == 0
    else:
        assert 1 <= L.count_up("ctab") <= 1 + len(b._cids)
    # a second mask on the same backend: its tables once more, then cached
    if entry in ("sample_ps", "estimate_ps"):
        other = np.zeros(s.num_detectors, np.uint8)   # one detector, a mask other than "half"
        other[0 if _mask(s.num_detectors, "half")[0] == 0 else -1] = 1
        assert other.tobytes() != _mask(s.num_detectors, "half").tobytes()
        for seed in (5, 6):
            if entry == "sample_ps":
                b.sample_ps(3 * B, seed, other)
            else:
                b.estimate(3 * B, seed, other)
        assert L.count_up("tab64") == 2 and L.count_up("tab32") == 2


def test_estimate_downloads_constant_bytes(path, mask):
    """estimate() downloads one hist_total (8 * (XK_HIST_STRIDE + 2) bytes) per batch, whatever the batch size"""
    s = xtim.XdemSampler(_read(path))
    flags = _flags(s, mask)
    for mbb in (1, 64 * 1024, None):
        L = _Recording()
        b = GpuBackend(s._s, L, max_batch_bytes=mbb)
        b.estimate(50 * B + 9, 7, flags)
        assert b.batches_run >= 1
        assert len(L.downloads) == b.batches_run
        assert all(name == "hist_total" and n <= 8 * (HS + 2) for name, n in L.downloads), L.downloads[:3]


def _kernel_buffers(t, n, seed, pair_cap, mode=0, span=None):
    """pack's buffers, cmiss filled with garbage (K1 must empty it under XK_MODE_EMIT)"""
    buf = gt.pack(t, block=B, pair_cap=pair_cap, shots=n, seed=seed, mode=mode | gt.XK_MODE_EMIT, span=span)
    buf["cmiss"][:] = 0xFFFF_FFFF
    return buf


def _settled(name, a):
    """a downloaded buffer with its atomic list in a fixed order (cmiss: the pairs sorted); psh, the walk's working
    list in the order of its atomics (read by nothing past the walk), empty"""
    if name == "psh":
        return a[:0]
    if name != "cmiss":
        return a
    m = int(a[0])
    out = a.copy()
    out[1:1 + 2 * m] = a[1:1 + 2 * m].reshape(m, 2)[np.argsort(a[1:1 + 2 * m:2], kind="stable")].reshape(-1)
    return out


def _run_all(L, buf, n, batch=True, emit=True):
    """the backend's submission: walk, resolve, reduce, reduce_groups, reduce2 and (with ``emit``) scan_blocks under
    postselection, then emit"""
    nb = -(-n // B)
    ps = int(buf["dims"][gt.XK_DIM_MODE]) & gt.XK_MODE_POSTSELECT
    steps = [("walk", nb), ("resolve", n), ("reduce", nb), ("reduce_groups", gt.XK_HIST_GROUPS * (HS + 2)),
             ("reduce2", HS + 2)]
    if emit:
        steps += ([("scan_blocks", nb)] if ps else []) + [("emit", n)]
    if batch:
        L.run_batch(steps)
    else:
        for k, m in steps:
            L.run(k, m)


def test_reduce2_equals_host_sum(path, mask):
    """hist_total[:67] = the per-block hist summed on the host; [67], [68] = the blocks whose overflow is
    XK_OVERFLOW_PAIRS / XK_OVERFLOW_CAPACITY; with pair_cap = 1 the PAIRS count is > 0 and estimate() still gives
    the bytes it gives without the relaunch"""
    if mask is not None and "boundary_r6" in path:
        pytest.skip("no shot survives postselection on the boundary files (the CPU accepts 0 of 20000 too)")
    s = xtim.XdemSampler(_read(path))
    t = s._s.device_tables()
    flags = _flags(s, mask)
    span = None if flags is None else s._s.span_tables(flags)
    mode = 0 if flags is None else PS
    n = 300 * B - 21   # 300 blocks: more blocks than groups
    E0 = GpuBackend(s._s, RefLauncher())._pair_cap
    for E in (E0, 1):
        L = RefLauncher()
        L.compile(HEADER.read_text(), gt.capacities(t, B, E))
        buf = _kernel_buffers(t, n, 9, E, mode, span)
        for k in ("hist", "hist_group", "hist_total"):   # sentinels: every slot read below is one a kernel wrote
            buf[k].view(np.uint8)[:] = 0xA5
        L.upload(buf)
        _run_all(L, buf, n, emit=False)
        nb = -(-n // B)
        hist = L.download("hist").reshape(nb, HS).astype(np.int64)
        # K7: each run of XK_EMIT_RUN shots' first output row = the survivors (flag XK_FLAG_SURVIVE: hist's SURVIVE
        # and error slots) among the shots before it
        L.run("scan_blocks", nb)
        surv = (L.download("flag")[:n] & gt.XK_FLAG_SURVIVE).astype(np.int64)
        assert int(surv.sum()) == int((hist[:, gt.XK_HIST_SURVIVE] + hist[:, HS - 1]).sum())
        runs = -(-n // gt.XK_EMIT_RUN)
        before = np.concatenate([[0], np.cumsum(surv)])[::gt.XK_EMIT_RUN][:runs]
        assert L.download("boff")[:runs].tolist() == before.tolist()
        tot = L.download("hist_total")
        assert tot.dtype == np.uint64 and tot.shape == (HS + 2,)
        assert tot[:HS].astype(np.int64).tolist() == hist.sum(axis=0).tolist()
        groups = L.download("hist_group").reshape(gt.XK_HIST_GROUPS, HS + 2)
        assert groups.sum(axis=0, dtype=np.uint64).tolist() == tot.tolist()
        per = -(-nb // gt.XK_HIST_GROUPS)   # group g: blocks [g * per, (g + 1) * per)
        assert groups[0, :HS].astype(np.int64).tolist() == hist[:per].sum(axis=0).tolist()
        assert not groups[-(-nb // per):].any()   # the groups past the batch are 0
        ov = L.download("overflow")
        assert int(tot[gt.XK_HIST_TOTAL_PAIRS]) == int(np.count_nonzero(ov == gt.XK_OVERFLOW_PAIRS))
        assert int(tot[gt.XK_HIST_TOTAL_CAPACITY]) == int(np.count_nonzero(ov == gt.XK_OVERFLOW_CAPACITY))
        if E == 1 and t["mean_ext"] * B > 2:
            assert tot[gt.XK_HIST_TOTAL_PAIRS] > 0
        if E == E0:
            assert int(tot[gt.XK_HIST_SURVIVE]) > 0 and tot[gt.XK_HIST_TOTAL_PAIRS] == 0
    want = GpuBackend(s._s, RefLauncher()).estimate(n, 9, flags)
    b = GpuBackend(s._s, RefLauncher())
    b._pair_cap = 1
    got = b.estimate(n, 9, flags)
    assert got[:2] == want[:2] and got[2].tolist() == want[2].tolist()
    if t["mean_ext"] * B > 2:
        assert b.relaunches > 0


@pytest.mark.parametrize("fill", [None, 0xFF])
def test_reused_buffers_do_not_leak(path, mask, fill):
    """backend A: every entry point on a large batch, then on a small off-grid count with another seed; backend B
    (fresh): the small run only; byte-equal. ``fill``: the reference launcher's allocations pre-filled with that byte
    (None: its default sentinel)"""
    text = _read(path)
    kw = {} if fill is None else {"fill": fill}

    def run(b, s, shots, seed):
        """the outputs, each cspace id replaced by its C (ids index the sampler's table, which grows with use)"""
        flags = _flags(s, mask)
        if flags is None:
            d, o, c = b.sample(shots, seed)
            counts = None
        else:
            d, o, c, counts = b.sample_ps(shots, seed, flags)
            counts = sorted(counts.items())
        cs = {int(i): s._s.cspaces(int(i), 1)[0].tolist() for i in np.unique(c)}
        return _digest(d, o, [cs[int(i)] for i in c], counts, *b.estimate(shots, seed, flags))

    sa, a = _backend(text, RefLauncher(**kw))
    run(a, sa, 40 * B + 5, 1)
    got = run(a, sa, 3 * B - 17, 2)
    sb, fresh = _backend(text, RefLauncher(**kw))
    assert got == run(fresh, sb, 3 * B - 17, 2)


def test_partial_last_batch_matches_exact_size_run(path):
    """the last batch smaller than the allocated buffers gives the bytes of a backend whose only batch is exact"""
    text = _read(path)
    n = 5 * B + 3
    blk = GpuBackend(xtim.XdemSampler(text)._s, RefLauncher())._block_bytes(True)   # sample's bytes per block
    _, many = _backend(text, max_batch_bytes=2 * blk)
    want = _backend(text)[1].sample(n, 8)
    assert _digest(*many.sample(n, 8)) == _digest(*want)
    assert many.batches_run == 3


# ------------------------------------------------------------------------------------------------ kernels

def test_run_batch_equals_runs(path):
    """run_batch(steps) leaves every buffer as the same run() calls in order do"""
    s = xtim.XdemSampler(_read(path))
    t = s._s.device_tables()
    n, E = 30 * B - 5, GpuBackend(s._s, RefLauncher())._pair_cap
    flags = _mask(s.num_detectors, "half")
    buf = _kernel_buffers(t, n, 4, E, PS, s._s.span_tables(flags))
    out = []
    for batch in (True, False):
        L = RefLauncher()
        L.compile(HEADER.read_text(), gt.capacities(t, B, E))
        L.upload(buf)
        _run_all(L, buf, n, batch)
        out.append({k: _settled(k, L.download(k)) for k in buf})
    for k in buf:
        assert out[0][k].tobytes() == out[1][k].tobytes(), k
    # ctab is empty: emit leaves every survivor with dim C > 0 to the host
    surv = (out[0]["flag"][:n] & gt.XK_FLAG_SURVIVE) != 0
    assert int(out[0]["cmiss"][0]) == int(np.count_nonzero(surv & (out[0]["cdim"][:n] > 0)))


@pytest.mark.metal
def test_metal_run_batch_equals_reference(path):
    """Metal run_batch (one submission) == the reference's run() calls, on every buffer"""
    s = xtim.XdemSampler(_read(path))
    t = s._s.device_tables()
    n, E = 30 * B - 5, GpuBackend(s._s, RefLauncher())._pair_cap
    flags = _mask(s.num_detectors, "half")
    buf = _kernel_buffers(t, n, 4, E, PS, s._s.span_tables(flags))
    out = []
    for L, batch in ((RefLauncher(), False), (_metal(), True)):
        L.compile(HEADER.read_text(), gt.capacities(t, B, E))
        L.upload(buf)
        _run_all(L, buf, n, batch)
        out.append({k: _settled(k, L.download(k)) for k in buf})
    for k in buf:
        assert out[0][k].tobytes() == out[1][k].tobytes(), k


GUARD = 4096   # sentinel elements past each buffer's batch size


def _guarded(buf, value=0xA5):
    """each working buffer with GUARD sentinel elements after its batch size"""
    out = dict(buf)
    for k in ("rows", "epos", "eids", "overflow", "ckey", "cdim", "flag", "hist", "hist_group", "hist_total"):
        a = buf[k]
        g = np.empty(len(a) + GUARD, a.dtype)
        g.view(np.uint8)[:] = value
        g[:len(a)] = a
        out[k] = g
    return out


@pytest.mark.parametrize("launcher", ["ref", pytest.param("metal", marks=pytest.mark.metal)])
def test_kernels_after_a_pair_overflow_stay_in_bounds(launcher):
    """walk overflows (pair_cap = 1) and resolve / reduce / reduce2 / scan_blocks / emit run on the batch anyway (one
    submission): an overflowed block's lists are empty (epos all 0), nothing is written past any buffer's batch size
    (emit's outputs: past the survivors), and hist_total counts the overflowed blocks"""
    s = xtim.XdemSampler(_read("examples/xdem/cultivation_d5.xdem"))
    t = s._s.device_tables()
    flags = _mask(s.num_detectors, "half")
    n = 40 * B - 9
    nb = -(-n // B)
    buf = _guarded(_kernel_buffers(t, n, 2, 1, PS, s._s.span_tables(flags)))
    L = RefLauncher() if launcher == "ref" else _metal()
    L.compile(HEADER.read_text(), gt.capacities(t, B, 1))
    L.upload(buf)
    _run_all(L, buf, n, emit=False)
    ov = L.download("overflow")
    over = np.flatnonzero(ov[:nb] == gt.XK_OVERFLOW_PAIRS)
    assert len(over) > nb // 2
    epos = L.download("epos")[:nb * (B + 1)].reshape(nb, B + 1)
    assert not epos[over].any()
    tot = L.download("hist_total")
    assert int(tot[gt.XK_HIST_TOTAL_PAIRS]) == len(over) and tot[gt.XK_HIST_TOTAL_CAPACITY] == 0
    flag = L.download("flag")[:n]
    surv = (flag & gt.XK_FLAG_SURVIVE) != 0
    k = int(surv.sum())
    kw = max(int(t["rhoL"]), 1)
    W = int(t["W"])
    nbd, nbo = (int(t["D"]) + 7) // 8, (int(t["O"]) + 7) // 8
    outs = {"boff": -(-n // gt.XK_EMIT_RUN), "out_det": k * nbd, "out_obs": k * nbo, "out_cs": k}   # what K7 and K4 write
    L.upload({name: np.zeros(len(buf[name]) + GUARD, buf[name].dtype) for name in outs})
    L.run_batch([("scan_blocks", nb), ("emit", n)])
    for name, size in (("rows", n * W), ("epos", nb * (B + 1)), ("eids", nb * 1), ("overflow", nb),
                       ("ckey", n * kw), ("cdim", n), ("flag", n), ("hist", nb * HS),
                       ("hist_group", gt.XK_HIST_GROUPS * (HS + 2)), ("hist_total", HS + 2)):
        tail = L.download(name)[size:]
        assert len(tail) == GUARD and (tail.view(np.uint8) == 0xA5).all(), name
    for name, size in outs.items():
        assert not L.download(name)[size:].any(), name
    assert L.download("out_det")[:k * nbd].any()


# ------------------------------------------------------------------------------------------------ padding bits

@pytest.mark.parametrize("kind", [None, "half"])
def test_row_padding_bits_are_zero(path, kind):
    """bits past D in the detector words and past O in the observable words of sampled rows are 0 (the zero
    decoder's test reads whole observable words; the CPU's score masks to the O bits first)"""
    s = xtim.XdemSampler(_read(path))
    t = s._s.device_tables()
    D, O, WD, WO = int(t["D"]), int(t["O"]), int(t["WD"]), int(t["WO"])
    assert D % 64 and (O % 64 or O == 0)
    flags = _flags(s, kind)
    span = None if flags is None else s._s.span_tables(flags)
    n = 200 * B - 3
    E = GpuBackend(s._s, RefLauncher())._pair_cap
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    L.upload(_kernel_buffers(t, n, 31, E, 0 if span is None else PS, span))
    L.run("walk", -(-n // B))
    L.run("resolve", n)
    rows = L.download("rows")[:n * int(t["W"])].reshape(n, int(t["W"]))
    keep = (L.download("flag")[:n] & gt.XK_FLAG_EARLY) == 0   # an early-rejected shot's row is the walk's
    rows = rows[keep]
    det_pad = np.zeros(WD, np.uint64)
    for w in range(WD):
        lo = max(0, D - 64 * w)
        det_pad[w] = 0 if lo >= 64 else ~np.uint64((1 << lo) - 1)
    obs_pad = np.zeros(WO, np.uint64)
    for w in range(WO):
        lo = max(0, O - 64 * w)
        obs_pad[w] = 0 if lo >= 64 else ~np.uint64((1 << lo) - 1)
    assert not (rows[:, :WD] & det_pad).any()
    assert not (rows[:, WD:WD + WO] & obs_pad).any()
    assert rows[:, :WD].any()   # not vacuous
    # why: a row is the XOR of flip entries and beta vectors, and none of them has a padding bit
    pad = np.concatenate([det_pad, obs_pad])
    assert not (np.asarray(t["ent_mask"], np.uint64) & pad[np.asarray(t["ent_word"], np.int64)]).any()
    if len(t["beta"]):
        assert not (np.asarray(t["beta"], np.uint64).reshape(-1, WD + WO) & pad).any()
