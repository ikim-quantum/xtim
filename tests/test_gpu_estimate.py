"""The device-only zero-decoder estimate (Task 9): K2 sets flag bit 2 (XK_FLAG_ZERO_PASS) on every surviving shot whose
observable row lies in its C (law_in_c's reduction over bLo / pL, then the C key), K3 ("reduce", xk_reduce_block) counts
a block's survivors and its passing shots by dim C into hist[block * XK_HIST_STRIDE ...], GpuBackend.estimate sums the
per-block counters on the host as int64, and XdemSampler.infidelity(shots, None) is the zero decoder (on the GPU from
the histogram alone; on the CPU the existing loop with a zeros decoder).

Exact properties (reference launcher): flag bit 2 == score(status, zeros) > 0 and the passing score is 2^-dim C, on
sampled shots and on forced fired sets reaching dim C = 1, 2; estimate's counts equal those of sample / sample_ps +
score, whatever the batch size; error-code shots are counted only in the error slot and raise XtimInvariantError;
host sums are exact past 2^32. Statistics: GPU infidelity(None) vs CPU infidelity(None), means within 5 combined
standard errors. One Metal copy of the flag / hist bytes.
"""
import math
from fractions import Fraction
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
from test_gpu_kernels import _cases, _given
from test_gpu_postselect import _mask

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)
HS = gt.XK_HIST_STRIDE
PS = gt.XK_MODE_POSTSELECT
ZP = gt.XK_FLAG_ZERO_PASS

# nontrivial C (cultivation_d5: rhoL = 1; rm15rep, toy2obs: rhoL = 2), many extended lines per block (ccz_toy_obs_p5),
# O = 0 (ccz_toy, code_switching_rate_noobs), rhoL = 0 with O = 1 (ccz_toy_obs), a conventional DEM
FILES = ["examples/xdem/cultivation_d5.xdem", "examples/xdem/rm15rep.xdem", "examples/xdem/toy2obs.xdem",
         "examples/xdem/ccz_toy_obs_p5.xdem", "examples/xdem/ccz_toy.xdem", "examples/xdem/ccz_toy_obs.xdem",
         "examples/xdem/code_switching_rate_noobs.xdem", "tests/data/xdem/synth/surface_d5r5.dem"]


def _read(name):
    return (ROOT / name).read_text()


@pytest.fixture(scope="module", params=FILES, ids=[Path(f).name for f in FILES])
def text(request):
    return _read(request.param)


@pytest.fixture
def gpu_ref(monkeypatch):
    """XdemSampler(device="gpu") on the reference launcher"""
    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", RefLauncher),))
    return lambda text: xtim.XdemSampler(text, device="gpu")


def _zeros(O):
    return lambda d: np.zeros((len(d), O), bool)


def _hist_of_scores(p):
    """int64[65]: the shots with p = 2^-k > 0, by k (p is 0 or a power of two)"""
    h = np.zeros(65, np.int64)
    pos = p[p > 0]
    k = np.round(-np.log2(pos)).astype(np.int64)
    assert (np.ldexp(1.0, -k) == pos).all()
    np.add.at(h, k, 1)
    return h


def _run(t, buf, n, kernels=("walk", "resolve"), E=None, launcher=None):
    L = launcher or RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E or int(buf["dims"][gt.XK_DIM_E])))
    L.upload(buf)
    nb = int(buf["dims"][gt.XK_DIM_NBLOCKS])
    for k in kernels:
        L.run(k, nb if k in ("walk", "reduce") else n)
    return L


# ------------------------------------------------------------------------------------------------- K2: flag bit 2


def _check_flags_vs_score(s, b, rows, ckey, cdim, flag, sel=None):
    """flag bit 2 of the selected shots == score(zeros) > 0, and the passing score is 2^-cdim"""
    sel = np.ones(len(cdim), bool) if sel is None else sel
    W, kw = b._W, b._kw
    n = int(sel.sum())
    _, obs, cs = host_outputs(s._s, b.tables, rows.reshape(-1, W)[sel].reshape(-1),
                              ckey.reshape(-1, kw)[sel].reshape(-1), cdim[sel])
    st = XdemStatus(obs, cs, None, s, s._s.num_cspaces())
    p = s.score(st, np.zeros((n, s.num_observables), bool))
    zp = (flag[sel] & ZP) != 0
    assert (zp == (p > 0)).all(), np.flatnonzero(zp != (p > 0))[:5]
    assert (p[zp] == np.ldexp(1.0, -cdim[sel][zp].astype(int))).all()
    return p, cdim[sel]


def test_zero_pass_flag_matches_score(text):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    b = GpuBackend(s._s, RefLauncher())
    n = 20 * B - 11   # a partial last block
    E = b._pair_cap
    # plain mode: every shot survives
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=41)
    L = _run(t, buf, n)
    flag = L.download("flag")
    assert ((flag & gt.XK_FLAG_SURVIVE) != 0).all() and not (flag & gt.XK_FLAG_EARLY).any()
    p, _ = _check_flags_vs_score(s, b, L.download("rows"), L.download("ckey"), L.download("cdim"), flag)
    if s.num_observables:
        assert (p > 0).any()
    # postselection: bit 2 only on survivors, and right there
    flags = _mask(s.num_detectors, "half")
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=41, mode=PS, span=s._s.span_tables(flags))
    L = _run(t, buf, n)
    flag = L.download("flag")
    surv = (flag & gt.XK_FLAG_SURVIVE) != 0
    assert not (flag[~surv] & ZP).any()
    _check_flags_vs_score(s, b, L.download("rows"), L.download("ckey"), L.download("cdim"), flag, surv)


@pytest.mark.parametrize("name", ["rm15rep", "toy2obs", "cultivation_d5"])
def test_zero_pass_flag_on_forced_sets(name):
    # forced fired sets reach dim C = 1 and 2 (tests/test_gpu_kernels.py::test_resolve_cases_reach_nontrivial_c);
    # at several coin words, flag bit 2 is the CPU score of the resolved row and C; passes at dim C 1 and 2 and a
    # failure at dim C = 1 occur where O >= 2
    s = xtim.XdemSampler(_read(f"examples/xdem/{name}.xdem"))
    t = s._s.device_tables()
    b = GpuBackend(s._s, RefLauncher())
    rng = np.random.default_rng(5)
    cases = _cases(t, rng)
    buf, E = _given(t, cases, rng)
    n = len(cases)
    seen = set()
    for coin in [0] + [int(x) for x in rng.integers(0, 2**63, 4)]:
        L = RefLauncher()
        L.compile(HEADER.read_text(), gt.capacities(t, B, E))
        L.upload({**buf, "coin_in": np.full(n, coin, np.uint64)})
        L.run("resolve", n)
        p, dims = _check_flags_vs_score(s, b, L.download("rows"), L.download("ckey"), L.download("cdim"),
                                        L.download("flag"))
        seen |= {(int(d), bool(x > 0)) for d, x in zip(dims, p)}
    assert (1, True) in seen and (0, False) in seen, seen
    if s.num_observables >= 2:   # with O = 1 (cultivation_d5) dim C = 1 is every value: a pass
        assert {(1, False), (2, True)} <= seen, seen


@pytest.mark.parametrize("variant", [0, 2])
@pytest.mark.parametrize("kind", [None, "half"])
def test_nokey_mode_same_rows_dims_and_flags(text, variant, kind):
    # XK_MODE_NOKEY (the estimate): no C key, dim C and the zero decoder's verdict from the law itself (the kernel
    # W, or the coin span's echelon), with and without the single-line law table; rows, cdim and flags equal to the
    # keyed run's, in both one-word implementations (variant 0: rho > 32's, variant 2: the register form)
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    n = 12 * B - 7
    flags = None if kind is None else _mask(s.num_detectors, kind)
    span = None if flags is None else s._s.span_tables(flags)
    base = PS if flags is not None else 0
    for law in (True, False):
        st = gt.static_tables(t, block=B, span=span, law=law)
        out = []
        for mode in (base, base | gt.XK_MODE_NOKEY):
            buf = gt.pack(t, block=B, pair_cap=256, shots=n, seed=77, mode=mode, span=span)
            buf.update(tab64=st.tab64, tab32=st.tab32)
            buf["dims"][:] = gt.pack_batch(st, pair_cap=256, shots=n, seed=77, mode=mode)["dims"]
            L = _run(t, buf, n, launcher=RefLauncher(variant=variant))
            out.append([L.download(k) for k in ("rows", "cdim", "flag")])
        for a, b_ in zip(*out):
            assert np.array_equal(a, b_)
    with pytest.raises(ValueError, match="NOKEY"):
        gt.pack(t, block=B, pair_cap=8, shots=8, seed=1, mode=gt.XK_MODE_NOKEY | gt.XK_MODE_EMIT)


# ------------------------------------------------------------------------------------------------- K3: reduce


def test_reduce_counts_partial_block_and_ignores_extra_ids():
    s = xtim.XdemSampler(_read("examples/xdem/rm15rep.xdem"))
    t = s._s.device_tables()
    n = 3 * B - 5
    buf = gt.pack(t, block=B, pair_cap=64, shots=n, seed=1, mode=gt.XK_MODE_GIVEN)
    assert buf["hist"].shape == (3 * HS,) and buf["hist"].dtype == np.uint32
    rng = np.random.default_rng(0)
    flag = rng.integers(0, 8, n).astype(np.uint8)
    cdim = rng.integers(0, 3, n).astype(np.uint8)
    cdim[[7, B + 6]] = gt.XK_CDIM_CAPACITY
    cdim[2 * B + 22] = gt.XK_CDIM_INTERNAL
    flag[[7, B + 6, 2 * B + 22]] = gt.XK_FLAG_SURVIVE | ZP
    tail = 30
    sentinel = np.uint32(0xDEADBEEF)
    buf["flag"], buf["cdim"] = flag, cdim
    buf["hist"] = np.concatenate([np.full(3 * HS, sentinel, np.uint32), np.full(tail, sentinel, np.uint32)])
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, 64))
    L.upload({k: buf[k] for k in ("dims", "seed", "tab64", "tab32", "rows", "cdim", "flag", "hist")})
    L.run("reduce", 3 + 50)   # a rounded-up grid
    h = L.download("hist")
    assert (h[3 * HS:] == sentinel).all()
    h = h[:3 * HS].reshape(3, HS)
    for blk in range(3):
        sl = slice(blk * B, min(n, (blk + 1) * B))
        f, c = flag[sl], cdim[sl]
        err = c >= gt.XK_CDIM_CAPACITY
        surv = ((f & gt.XK_FLAG_SURVIVE) != 0) & ~err
        assert h[blk, gt.XK_HIST_SURVIVE] == surv.sum()
        assert h[blk, gt.XK_HIST_ERROR] == (((f & gt.XK_FLAG_SURVIVE) != 0) & err).sum()
        for k in range(65):
            assert h[blk, gt.XK_HIST_PASS + k] == (surv & ((f & ZP) != 0) & (c == k)).sum(), (blk, k)
    assert h[:, gt.XK_HIST_ERROR].tolist() == [1, 1, 1]


# ------------------------------------------------------------------------------------------------- estimate


@pytest.mark.parametrize("kind", [None, "half", "all"])
def test_estimate_equals_sample_then_score(text, kind):
    s = xtim.XdemSampler(text)
    O = s.num_observables
    n = 5000
    flags = None if kind is None else _mask(s.num_detectors, kind)
    ref = GpuBackend(s._s, RefLauncher())
    if flags is None:
        _, obs, cs = ref.sample(n, 9, first_block=3)
        passed = n
    else:
        _, obs, cs, counts = ref.sample_ps(n, 9, flags, first_block=3)
        passed = counts["passed"]
    st = XdemStatus(obs, cs, None, s, s._s.num_cspaces())
    p = s.score(st, np.zeros((len(cs), O), bool))
    want = _hist_of_scores(p)
    for mbb in (None, 1, 9 * 4096):   # one batch; one block per batch; a few blocks per batch
        b = GpuBackend(s._s, RefLauncher(), max_batch_bytes=mbb)
        att, got_passed, h = b.estimate(n, 9, flags, first_block=3)
        assert (att, got_passed) == (n, passed), mbb
        assert h.dtype == np.int64 and h.shape == (65,)
        assert h.tolist() == want.tolist(), mbb
        assert sum(Fraction(int(c), 2**k) for k, c in enumerate(h)) == sum(Fraction(x) for x in p)
        if mbb == 1:
            assert b.batches_run > 1
    if O:
        assert want.sum() > 0
    att, got_passed, h = GpuBackend(s._s, RefLauncher()).estimate(0, 9, flags)
    assert (att, got_passed) == (0, 0) and h.shape == (65,) and not h.any()


def test_estimate_after_overflow_relaunch():
    s = xtim.XdemSampler(_read("examples/xdem/cultivation_d5.xdem"))
    want = GpuBackend(s._s, RefLauncher()).estimate(3000, 4, None)
    b = GpuBackend(s._s, RefLauncher())
    b._pair_cap = 1
    got = b.estimate(3000, 4, None)
    assert b.relaunches > 0
    assert got[:2] == want[:2] and got[2].tolist() == want[2].tolist()


def test_estimate_bad_arguments():
    s = xtim.XdemSampler(_read("examples/xdem/toy2obs.xdem"))
    b = GpuBackend(s._s, RefLauncher())
    with pytest.raises(ValueError, match="shape"):
        b.estimate(10, 1, np.ones(s.num_detectors + 1, np.uint8))
    with pytest.raises(ValueError):
        b.estimate(-1, 1, None)
    with pytest.raises(ValueError):
        b.estimate(1, 2**64, None)


class _Corrupt(RefLauncher):
    """the reference launcher writing one error code into cdim (with the survive flag) after "resolve", on the device
    side (before "reduce" reads it)"""

    def __init__(self, value):
        super().__init__()
        self._value = value

    def run(self, kernel, n, lanes=None):
        super().run(kernel, n, lanes)
        if kernel == "resolve":
            i = len(self._buf["cdim"]) // 2
            self._buf["cdim"][i] = self._value
            self._buf["flag"][i] |= gt.XK_FLAG_SURVIVE


@pytest.mark.parametrize("value,match", [(gt.XK_CDIM_CAPACITY, "capacity"), (gt.XK_CDIM_INTERNAL, "internal")])
@pytest.mark.parametrize("ps", [False, True])
def test_error_code_shot_raises(value, match, ps):
    s = xtim.XdemSampler(_read("examples/xdem/rm15rep.xdem"))
    b = GpuBackend(s._s, _Corrupt(value))
    flags = _mask(s.num_detectors, "half") if ps else None
    with pytest.raises(xtim.XtimInvariantError, match=match):
        b.estimate(500, 1, flags)


class _MaxHist:
    """a launcher that runs nothing: every batch total is that of one block whose hist counters but the error slot
    are 0xFFFFFFFF (no overflow)"""
    name = "mock"

    def compile(self, source, capacities):
        pass

    def upload(self, buffers):
        self._n = {k: len(v) for k, v in buffers.items()}

    def alloc(self, name, dtype, count):
        pass

    def run(self, kernel, n, lanes=None):
        pass

    def run_batch(self, steps):
        assert [k for k, *_ in steps] == ["walk", "resolve", "resolve_law", "reduce", "reduce_groups", "reduce2"]
        assert steps[0][1] == 1   # one block per batch

    def download(self, name, count=None):
        assert name == "hist_total", name   # estimate downloads nothing else
        h = np.zeros(gt.XK_HIST_TOTAL_COUNT, np.uint64)
        h[:HS] = 0xFFFF_FFFF
        h[gt.XK_HIST_ERROR] = 0
        return h[:count]

    def free_bytes(self):
        return 1 << 32


def test_estimate_large_counts_no_overflow():
    s = xtim.XdemSampler("error(0.1) D0 L0\n")
    b = GpuBackend(s._s, _MaxHist(), max_batch_bytes=1)   # one block per batch
    nb = 5000
    att, passed, h = b.estimate(nb * B, 3, None)
    assert b.batches_run == nb
    big = nb * 0xFFFF_FFFF
    assert big > 2**32 and att == nb * B
    assert passed == big and isinstance(passed, int)
    assert h.dtype == np.int64 and h.tolist() == [big] * 65


# ------------------------------------------------------------------------------------------------- infidelity(None)


def _from_hist(passed, h):
    """(mean, stderr) of 1 - pass for `passed` shots, h[k] of them passing with 2^-k, the rest 0: exact moments"""
    q1 = Fraction(passed - int(h.sum()))
    q2 = Fraction(q1)
    for k, c in enumerate(h):
        q = 1 - Fraction(1, 2**k)
        q1 += int(c) * q
        q2 += int(c) * q * q
    m = q1 / passed
    var = (q2 - passed * m * m) / (passed - 1)
    return float(m), math.sqrt(float(var / passed))


@pytest.mark.parametrize("kind", [None, "half"])
def test_infidelity_none_cpu_equals_zero_decoder(text, kind):
    s = xtim.XdemSampler(text)
    ps = None if kind is None else _mask(s.num_detectors, kind).astype(bool)
    for batch in (100_000, 1024):
        a = s.infidelity(5000, None, seed=17, batch=batch, postselect=ps)
        z = s.infidelity(5000, _zeros(s.num_observables), seed=17, batch=batch, postselect=ps)
        assert (a.mean, a.stderr, a.shots, a.accepted, a.giveup_fraction) == \
            (z.mean, z.stderr, z.shots, z.accepted, z.giveup_fraction) or (math.isnan(a.mean) and math.isnan(z.mean))


@pytest.mark.parametrize("kind", [None, "half"])
def test_infidelity_none_gpu_fields(gpu_ref, text, kind):
    s = gpu_ref(text)
    n = 6000
    flags = None if kind is None else _mask(s.num_detectors, kind)
    ps = None if flags is None else flags.astype(bool)
    r = s.infidelity(n, None, seed=23, postselect=ps)
    att, passed, h = s._gpu.estimate(n, 23, flags)
    assert r.shots == att == n and r.accepted == passed and r.giveup_fraction == 0.0
    mean, se = _from_hist(passed, h)
    assert r.mean == mean and r.stderr == se
    if 0 < r.mean < 1:
        assert r.stderr > 0
    # the same stream as the explicit zeros decoder on this GPU sampler (sample's / sample_ps's shots)
    z = s.infidelity(n, _zeros(s.num_observables), seed=23, postselect=ps, batch=2000)
    assert z.accepted == r.accepted and z.shots == r.shots
    assert math.isclose(z.mean, r.mean, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(z.stderr, r.stderr, rel_tol=1e-9, abs_tol=1e-15)


def test_infidelity_none_degenerate(gpu_ref):
    # nothing accepted: NaN mean / stderr, as on the CPU; one accepted: NaN stderr; an empty selection = no mask
    s = gpu_ref("error(1) D0\nerror(0.1) D1 L0\n")
    r = s.infidelity(300, None, seed=1, postselect=[0])
    assert r.accepted == 0 and r.shots == 300 and math.isnan(r.mean) and math.isnan(r.stderr)
    assert s.infidelity(0, None, seed=1).accepted == 0
    one = s.infidelity(1, None, seed=1)
    assert one.accepted == 1 and math.isnan(one.stderr) and one.mean in (0.0, 1.0)
    assert s.infidelity(500, None, seed=2, postselect=[]) == s.infidelity(500, None, seed=2)
    c = xtim.XdemSampler("error(1) D0\nerror(0.1) D1 L0\n")
    rc = c.infidelity(300, None, seed=1, postselect=[0])
    assert rc.accepted == 0 and math.isnan(rc.mean)


def test_infidelity_none_with_lines_refused(gpu_ref):
    text = _read("examples/xdem/rm15rep.xdem")
    for s in (xtim.XdemSampler(text), gpu_ref(text)):
        with pytest.raises(ValueError, match="decoder=None"):
            s.infidelity(10, None, seed=1, lines=True)


# ------------------------------------------------------------------------------------------------- Metal


@pytest.mark.metal
@pytest.mark.parametrize("name", ["cultivation_d5", "rm15rep", "toy2obs", "ccz_toy_obs_p5"])
@pytest.mark.parametrize("kind", [None, "half"])
def test_metal_flag_and_hist_equal_reference(name, kind):
    from xtim.gpu.metal import MetalLauncher
    s = xtim.XdemSampler(_read(f"examples/xdem/{name}.xdem"))
    t = s._s.device_tables()
    E = GpuBackend(s._s, RefLauncher())._pair_cap
    n = 64 * B - 37
    span = None if kind is None else s._s.span_tables(_mask(s.num_detectors, kind))
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=13, first_block=5, mode=0 if kind is None else PS, span=span)
    out = []
    for L in (RefLauncher(), MetalLauncher()):
        _run(t, buf, n, ("walk", "resolve", "reduce"), E, L)
        out.append({k: L.download(k) for k in ("flag", "hist", "cdim")})
    ref, got = out
    for k in ("flag", "hist"):
        assert got[k].dtype == ref[k].dtype and np.array_equal(got[k], ref[k]), k
    assert (ref["flag"] & ZP).any() and ref["hist"].any()


@pytest.mark.metal
def test_metal_estimate_equals_reference():
    from xtim.gpu.metal import MetalLauncher
    s = xtim.XdemSampler(_read("examples/xdem/cultivation_d5.xdem"))
    flags = _mask(s.num_detectors, "all")
    for f in (None, flags):
        a = GpuBackend(s._s, MetalLauncher()).estimate(20_000, 6, f)
        b = GpuBackend(s._s, RefLauncher()).estimate(20_000, 6, f)
        assert a[:2] == b[:2] and a[2].tolist() == b[2].tolist()


# ------------------------------------------------------------------------------------------------- statistics vs CPU


STAT_NAMES = [n for n in NAMES_INREPO if present(n)]
STAT_SHOTS = 2_000_000
STAT_MASKS = [None, "all", "half"]


@pytest.mark.parametrize("name", STAT_NAMES)
def test_infidelity_none_statistics_vs_cpu(gpu_ref, name):
    try:
        text = xdem_export(circuit(name))
    except xtim.XtimDemError as e:   # refused by the xdem export (tests/test_gpu_backend.py's rule)
        pytest.skip(f"{name}: refused by the xdem export ({str(e).splitlines()[0]})")
    if not xtim.XdemSampler(text)._s.device_tables()["fast"]:
        pytest.skip(f"{name}: not on the kernels' one-word fast path")
    i = STAT_NAMES.index(name)
    g = gpu_ref(text)
    c = xtim.XdemSampler(text)
    D = c.num_detectors
    for j, kind in enumerate(STAT_MASKS):
        ps = None if kind is None else _mask(D, kind).astype(bool)
        rg = g.infidelity(STAT_SHOTS, None, seed=91_000 + 3 * i + j, postselect=ps)
        rc = c.infidelity(STAT_SHOTS, None, seed=92_000 + 3 * i + j, postselect=ps, threads=4)
        msg = (f"{name} [{kind}]: gpu {rg.mean:.6g} +- {rg.stderr:.3g} ({rg.accepted}) cpu {rc.mean:.6g} +- "
               f"{rc.stderr:.3g} ({rc.accepted})")
        print(msg)
        if rg.accepted == rc.accepted == 0:
            continue
        se = math.hypot(rg.stderr, rc.stderr)
        if se == 0:
            assert rg.mean == rc.mean, msg
        else:
            assert abs(rg.mean - rc.mean) <= 5 * se, msg
