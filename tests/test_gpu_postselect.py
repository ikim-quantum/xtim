"""Post-selection on the device (Task 8): K2 with mode bit XK_MODE_POSTSELECT writes flag[shot] (bit 0 survive, bit
1 rejected early by the span test) following run_one_block's postselected loop, "gather" (K4) compacts the survivors,
and GpuBackend.sample_ps / XdemSampler(device=...).sample(postselect=) / infidelity(postselect=) return them.

Exact properties (reference launcher, the kernel source built as C++): the decision on forced fired sets against the
CPU's law (xdem_law_of) and the exported span test; sample_ps's survivors are exactly the rows of sample() of the same
seed whose masked detectors are 0, in order (coin words are keyed per shot), C compared by content; counts; an empty
mask is sample(); zero survivors. Statistics against the CPU's sample_ps: acceptance, early rejections, the pass mean
over survivors and the survivors' detector marginals, |z| <= 5. One Metal copy of the exactness check.
"""
import math
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim import _xtim_testapi as T
from xtim.gpu import tables as gt
from xtim.gpu.backend import DEFAULT_BLOCK, GpuBackend

from _gpu_ref import RefLauncher, in_span
from _xdem_circuits import NAMES_INREPO, circuit, present, xdem_export
from test_gpu_kernels import B as KB   # the block of _given's kernel buffers
from test_gpu_kernels import _cases, _given, _row_int, _to_file
from test_xdem_cross_route import Z_MAX

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)
MODE_GIVEN = gt.XK_MODE_COIN_IN | gt.XK_MODE_GIVEN
PS = gt.XK_MODE_POSTSELECT

FILES = ["examples/xdem/cultivation_d5.xdem", "examples/xdem/rm15rep.xdem", "examples/xdem/toy2obs.xdem",
         "examples/xdem/ccz_toy_obs_p5.xdem", "tests/data/xdem/synth/surface_d5r5.dem"]
MASKS = ["half", "all"]
ALL_REJECTED = "error(1) D0\nerror(0.1) D1 L0\n"   # D0 fires on every shot


def _read(name):
    return (ROOT / name).read_text()


@pytest.fixture(scope="module", params=FILES, ids=[Path(f).name for f in FILES])
def text(request):
    return _read(request.param)


def _mask(D, kind, seed=12345):
    """uint8[D]: every detector, or a fixed random half"""
    if kind == "all":
        return np.ones(D, np.uint8)
    return (np.random.default_rng(seed).random(D) < 0.5).astype(np.uint8)


def _masked_zero(dets, flags):
    """bool[n]: the shots whose masked detectors are all 0 (dets packed little-endian, as the samplers return them)"""
    D = len(flags)
    bits = np.unpackbits(dets, axis=1, bitorder="little")[:, :D]
    return ~(bits & flags.astype(bool)).any(axis=1)


def _same(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b))


def _cs_content(S, cs):
    """each shot's C as its basis rows (tuples), from the sampler's table"""
    tab = {int(i): tuple(tuple(int(b) for b in row) for row in S.cspaces(int(i), 1)[0]) for i in np.unique(cs)}
    return [tab[int(i)] for i in cs]


# ------------------------------------------------------------------------------------------------- tables and launchers


def test_pack_span_tables(text):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    with pytest.raises(ValueError, match="span"):
        gt.pack(t, block=B, pair_cap=8, shots=10, seed=1, mode=PS)
    st = s._s.span_tables(_mask(s.num_detectors, "half"))
    buf = gt.pack(t, block=B, pair_cap=8, shots=10, seed=1, mode=PS, span=st)
    d = buf["dims"]
    nw = len(st["widx"])
    assert d[gt.XK_DIM_SPAN_RANK] == st["rank"] and d[gt.XK_DIM_SPAN_WORDS] == nw and d[gt.XK_DIM_MODE] == PS
    for name, arena in (("wmask", "tab64"), ("basis", "tab64"), ("piv_bit", "tab64"), ("widx", "tab32"),
                        ("piv_word", "tab32")):
        off = int(d[getattr(gt, "XK_DIM_OFF_SPAN_" + name.upper())])
        assert buf[arena][off:off + len(st[name])].tolist() == st[name].tolist(), name
    assert buf["flag"].shape == (10,) and buf["flag"].dtype == np.uint8 and not buf["flag"].any()
    plain = gt.pack(t, block=B, pair_cap=8, shots=10, seed=1)
    assert plain["dims"][gt.XK_DIM_SPAN_RANK] == plain["dims"][gt.XK_DIM_SPAN_WORDS] == 0
    # the plain tables sit where they did: the span tables are appended to the arenas
    for arena in ("tab64", "tab32"):
        assert buf[arena][:len(plain[arena]) - 1].tolist() == plain[arena][:-1].tolist()


def _walk_only(t):
    buf = gt.pack(t, block=B, pair_cap=64, shots=100, seed=1)
    return {k: buf[k] for k in ("dims", "seed", "tab64", "tab32", "rows", "epos", "eids", "psh", "overflow", "lawq",
                                "cmiss", "flag")}


def _launchers():
    out = [RefLauncher]
    try:
        from xtim.gpu.metal import MetalLauncher
        MetalLauncher()
        out.append(MetalLauncher)
    except Exception:   # no Metal here: the reference alone
        pass
    return out


@pytest.mark.parametrize("make", _launchers(), ids=lambda m: m.__name__)
def test_per_kernel_buffers(make):
    """a kernel needs only its own buffers uploaded; a missing one of its own is a ValueError naming it (both
    launchers alike)"""
    t = xtim.XdemSampler(_read("examples/xdem/rm15rep.xdem"))._s.device_tables()
    L = make()
    L.compile(HEADER.read_text(), gt.capacities(t, B, 64))
    L.upload(_walk_only(t))
    L.run("walk", 2)   # no ckey, cdim, coin_in, mws, out_* (lawq, cmiss: K2's queue and K4's list, which K1
    # empties; flag: its postselection screen's verdicts)
    with pytest.raises(ValueError, match="ckey"):
        L.run("resolve", 100)
    full = gt.pack(t, block=B, pair_cap=64, shots=100, seed=1)
    L.upload({k: full[k] for k in ("ckey", "cdim", "coin_in")})
    with pytest.raises(ValueError, match="mws"):   # the multi-word law's scratch: bound for every file
        L.run("resolve", 100)
    L.upload({"mws": full["mws"]})
    L.run("resolve", 100)
    with pytest.raises(ValueError, match="boff"):
        L.run("emit", 100)


# ------------------------------------------------------------------------------------------------- K2 decision


def _resolve_ps(t, buf, E, coin):
    n = int(buf["dims"][gt.XK_DIM_NSHOTS])
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, KB, E))
    L.upload({**buf, "coin_in": np.full(n, coin, np.uint64)})
    L.run("resolve", n)
    return L.download("rows"), L.download("flag")


@pytest.mark.parametrize("kind", MASKS)
def test_ps_decision_matches_cpu_on_forced_sets(text, kind):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    D, W, r = t["D"], t["W"], t["r"]
    flags = _mask(D, kind)
    st = s._s.span_tables(flags)
    rng = np.random.default_rng(3)
    cases = _cases(t, rng)
    buf0, E = _given(t, cases, rng)
    n = len(cases)
    buf = gt.pack(t, block=KB, pair_cap=E, shots=n, seed=0, mode=MODE_GIVEN | PS, span=st)
    for k in ("rows", "epos", "eids"):
        buf[k] = buf0[k].copy()
    pre = [_row_int(buf0["rows"][k * W:(k + 1) * W]) for k in range(n)]
    has_ext = [any(t["mem_xo"][m] for m in ms) for ms in cases]
    mbits = sum(1 << d for d in range(D) if flags[d])
    nw = len(st["widx"])

    def span_ok(x):   # XdemSpanTest::in_span from the exported tables (tests/test_gpu_tables.py)
        v = [(x >> (64 * int(w)) & ((1 << 64) - 1)) & int(m) for w, m in zip(st["widx"], st["wmask"])]
        dv = list(v)
        for b in range(st["rank"]):
            if dv[st["piv_word"][b]] & int(st["piv_bit"][b]):
                for k in range(nw):
                    v[k] ^= int(st["basis"][b * nw + k])
        return not any(v)

    laws = {}

    def law_of(k):   # the CPU law of case k (one sampler construction per case)
        if k not in laws:
            laws[k] = T.xdem_law_of(text, [int(t["mem_line"][m]) for m in cases[k]], False)
        return laws[k]

    coins = [0] + [1 << j for j in range(r)] + [int(x) for x in rng.integers(0, 2**63, 3)]
    early0 = None
    seen = {"early": 0, "survive": 0, "late_reject": 0}
    for coin in coins:
        rows_ps, fl = _resolve_ps(t, buf, E, coin)
        rows_pl = _given_rows(t, buf0, E, coin)
        assert not (fl & ~np.uint8(7)).any()
        early = (fl & 2) != 0
        surv = (fl & 1) != 0
        assert not (early & surv).any()   # bit 1 implies bit 0 clear
        assert not (fl[~surv] & 4).any()  # bit 2 (zero decoder passes, Task 9) on survivors only
        if early0 is None:
            early0 = early
        assert (early == early0).all()    # the span test reads no coin
        for k in range(n):
            # bit 1: an extended line fired and the pre-correction row fails the span test (run_one_block's order)
            assert early[k] == (has_ext[k] and not span_ok(pre[k])), (cases[k], coin)
            row_pl = _row_int(rows_pl[k * W:(k + 1) * W])
            if early[k]:
                # sound: no row of the CPU law has its masked detectors 0
                law = law_of(k)
                assert not in_span(int(law["offset"]) & mbits, [int(c) & mbits for c in law["coins"]]), cases[k]
                seen["early"] += 1
                continue
            # otherwise the decision is the corrected row's (the unselected K2's row at the same coin)
            assert surv[k] == ((row_pl & mbits) == 0), (cases[k], coin)
            if surv[k]:
                assert _row_int(rows_ps[k * W:(k + 1) * W]) == row_pl
                if has_ext[k]:
                    law = law_of(k)
                    off = _to_file(row_pl, t)
                    assert in_span(off ^ int(law["offset"]), [int(c) for c in law["coins"]])
                seen["survive"] += 1
            else:
                seen["late_reject"] += 1
    assert seen["survive"] + seen["late_reject"] + seen["early"] == n * len(coins)


def _given_rows(t, buf0, E, coin):
    n = int(buf0["dims"][gt.XK_DIM_NSHOTS])
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, KB, E))
    L.upload({**buf0, "coin_in": np.full(n, coin, np.uint64)})
    L.run("resolve", n)
    return L.download("rows")


def test_ps_decision_reaches_every_outcome():
    """the decision test above is not vacuous: cultivation_d5 (all detectors) has early rejections, survivors after an
    extended line and rejections after the correction"""
    text = _read("examples/xdem/cultivation_d5.xdem")
    s = xtim.XdemSampler(text)
    b = GpuBackend(s._s, RefLauncher())
    flags = _mask(s.num_detectors, "all")
    *_, counts = b.sample_ps(20000, 5, flags)
    assert 0 < counts["early"] < counts["discarded"] and counts["passed"] > 0


# ------------------------------------------------------------------------------------------------- the backend


def _backend(text, launcher=None, **kw):
    S = xtim.XdemSampler(text)._s
    return S, GpuBackend(S, launcher or RefLauncher(), **kw)


def _filtered(text, kind, n, seed, launcher=None, **kw):
    S1, b1 = _backend(text, launcher, **kw)
    flags = _mask(S1.num_detectors, kind)
    d, o, c = b1.sample(n, seed)
    S2, b2 = _backend(text, launcher, **kw)
    dp, op, cp, counts = b2.sample_ps(n, seed, flags)
    keep = _masked_zero(d, flags)
    return (S1, d, o, c, keep), (S2, dp, op, cp, counts)


@pytest.mark.parametrize("kind", MASKS)
def test_ps_survivors_are_filtered_sample(text, kind):
    for kw in ({}, {"max_batch_bytes": 1}):   # one batch; one block per batch (some batches with no survivor)
        (S1, d, o, c, keep), (S2, dp, op, cp, counts) = _filtered(text, kind, 3000, 9, **kw)
        assert dp.dtype == np.uint8 and op.dtype == np.uint8 and cp.dtype == np.int32
        assert dp.tobytes() == d[keep].tobytes() and dp.shape == d[keep].shape
        assert op.tobytes() == o[keep].tobytes() and op.shape == o[keep].shape
        assert _cs_content(S2, cp) == _cs_content(S1, c[keep])
        assert counts["passed"] == int(keep.sum())
        assert 0 < counts["passed"] or kind == "all"   # every batching, not only the last


def test_ps_first_block_is_a_stream_offset(text):
    S, b = _backend(text)
    flags = _mask(S.num_detectors, "half")
    d, *_ = b.sample_ps(7 * B + 20, 5, flags)
    whole, _, _ = b.sample(7 * B + 20, 5)
    assert d.tobytes() == whole[_masked_zero(whole, flags)].tobytes()   # first_block = 0: the whole stream's
    k = 3
    got = b.sample_ps(2 * B + 20, 5, flags, first_block=k)
    part = whole[k * B:k * B + 2 * B + 20]
    assert got[0].tobytes() == part[_masked_zero(part, flags)].tobytes()


@pytest.mark.parametrize("kind", MASKS)
def test_ps_counts(text, kind):
    S, b = _backend(text)
    n = 4000
    dp, op, cp, counts = b.sample_ps(n, 2, _mask(S.num_detectors, kind))
    assert set(counts) == {"attempts", "passed", "discarded", "early"}
    assert counts["attempts"] == n and counts["passed"] == len(dp) == len(op) == len(cp)
    assert counts["discarded"] == n - counts["passed"] and 0 <= counts["early"] <= counts["discarded"]
    if not S.device_tables()["mean_ext"]:
        assert counts["early"] == 0   # no extended line: no span test


def test_ps_empty_mask_is_sample(text):
    S, b = _backend(text)
    want = b.sample(2000, 3)
    dp, op, cp, counts = b.sample_ps(2000, 3, np.zeros(S.num_detectors, np.uint8))
    assert _same((dp, op, cp), want)
    assert counts == {"attempts": 2000, "passed": 2000, "discarded": 0, "early": 0}


def test_ps_all_rejected():
    S, b = _backend(ALL_REJECTED, max_batch_bytes=1)
    flags = np.array([1, 0], np.uint8)
    dp, op, cp, counts = b.sample_ps(1000, 1, flags)
    assert dp.shape == (0, 1) and op.shape == (0, 1) and cp.shape == (0,)
    assert dp.dtype == op.dtype == np.uint8 and cp.dtype == np.int32
    assert counts == {"attempts": 1000, "passed": 0, "discarded": 1000, "early": 0}
    assert b.batches_run > 1
    z = b.sample_ps(0, 1, flags)
    assert z[0].shape == (0, 1) and z[3] == {"attempts": 0, "passed": 0, "discarded": 0, "early": 0}


def test_ps_bad_arguments():
    S, b = _backend(_read("examples/xdem/rm15rep.xdem"))
    with pytest.raises(ValueError, match="flags"):
        b.sample_ps(10, 1, np.zeros(S.num_detectors + 1, np.uint8))
    with pytest.raises(ValueError, match="shots"):
        b.sample_ps(-1, 1, np.zeros(S.num_detectors, np.uint8))


@pytest.mark.metal
def test_ps_metal_equals_ref(text):
    from xtim.gpu.metal import MetalLauncher
    for kind in MASKS:
        _, ref = _backend(text, max_batch_bytes=20000)
        _, met = _backend(text, MetalLauncher(), max_batch_bytes=20000)
        S = ref._cpu
        flags = _mask(S.num_detectors, kind)
        a, b = ref.sample_ps(3000, 5, flags), met.sample_ps(3000, 5, flags)
        assert _same(a[:3], b[:3]) and a[3] == b[3], kind


@pytest.mark.metal
def test_ps_metal_all_rejected():
    from xtim.gpu.metal import MetalLauncher
    _, b = _backend(ALL_REJECTED, MetalLauncher(), max_batch_bytes=1)
    dp, op, cp, counts = b.sample_ps(500, 1, np.array([1, 0], np.uint8))
    assert len(dp) == len(op) == len(cp) == 0 and counts["discarded"] == 500


# ------------------------------------------------------------------------------------------------- the public API


@pytest.fixture
def gpu_ref(monkeypatch):
    """XdemSampler(device="gpu") on the reference launcher"""
    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", RefLauncher),))
    return lambda text: xtim.XdemSampler(text, device="gpu")


def test_public_sample_postselect(gpu_ref):
    text = _read("examples/xdem/rm15rep.xdem")
    s = gpu_ref(text)
    D = s.num_detectors
    flags = _mask(D, "half")
    idx = [int(i) for i in np.flatnonzero(flags)]
    dets, st, counts = s.sample(3000, seed=4, postselect=idx)
    dets_b, st_b, counts_b = s.sample(3000, seed=4, postselect=flags.astype(bool))
    assert dets.tobytes() == dets_b.tobytes() and counts == counts_b
    all_d, all_st = s.sample(3000, seed=4)
    keep = _masked_zero(all_d, flags)
    assert dets.tobytes() == all_d[keep].tobytes() and st.obs.tobytes() == all_st.obs[keep].tobytes()
    assert isinstance(st, xtim.xdem.XdemStatus) and len(st) == counts["passed"] == int(keep.sum())
    assert counts["attempts"] == 3000 and counts["discarded"] == 3000 - counts["passed"]
    assert s.score(st, np.zeros((len(st), s.num_observables), bool)).shape == (len(st),)
    # an empty selection: sample()'s bytes, nothing discarded (as on the CPU)
    d0, st0, c0 = s.sample(3000, seed=4, postselect=[])
    assert d0.tobytes() == all_d.tobytes() and c0 == {"attempts": 3000, "passed": 3000, "discarded": 0, "early": 0}
    cpu = xtim.XdemSampler(text)
    assert cpu.sample(10, seed=1, postselect=[])[2] == {"attempts": 10, "passed": 10, "discarded": 0, "early": 0}
    # the CPU's argument rules and refusal texts
    for bad in ([0, 0], [D], np.ones(D, np.uint8), np.ones(D + 1, bool), [[0]]):
        with pytest.raises(ValueError) as ec:
            cpu.sample(10, seed=1, postselect=bad)
        with pytest.raises(ValueError, match=__import__("re").escape(str(ec.value))):
            s.sample(10, seed=1, postselect=bad)
    with pytest.raises(xtim.XtimGpuError, match="fired lines"):
        s.sample(10, seed=1, postselect=idx, fired=True)


def test_public_infidelity_postselect(gpu_ref):
    text = _read("examples/xdem/rm15rep.xdem")
    s = gpu_ref(text)
    O = s.num_observables
    flags = _mask(s.num_detectors, "half")
    dec = lambda d: np.zeros((len(d), O), bool)   # noqa: E731
    one = s.infidelity(5000, dec, seed=4, batch=10**6, postselect=flags.astype(bool))
    many = s.infidelity(5000, dec, seed=4, batch=1000, postselect=list(np.flatnonzero(flags)) * 2)   # repeats allowed
    assert one == many
    _, st, counts = s.sample(5000, seed=4, postselect=flags.astype(bool))
    assert one.shots == 5000 and one.accepted == counts["passed"] == len(st)
    p = s.score(st, np.zeros((len(st), O), bool))
    assert math.isclose(one.mean, float((1.0 - p).mean()), rel_tol=1e-12, abs_tol=1e-15)
    none = s.infidelity(5000, dec, seed=4, postselect=[])
    assert none == s.infidelity(5000, dec, seed=4) and none.accepted == 5000
    with pytest.raises(xtim.XtimGpuError, match="fired lines"):
        s.infidelity(10, lambda d: [[]] * len(d), seed=1, postselect=[0], lines=True)
    allrej = gpu_ref(ALL_REJECTED).infidelity(300, lambda d: np.zeros((len(d), 1), bool), seed=1, postselect=[0])
    assert allrej.accepted == 0 and allrej.shots == 300 and math.isnan(allrej.mean)


# ------------------------------------------------------------------------------------- statistics vs the CPU sampler


STAT_NAMES = [n for n in NAMES_INREPO if present(n)]
STAT_SHOTS = 2_000_000


def _z_prop2(k1, n1, k2, n2):
    """pooled two-proportion z for k1/n1 vs k2/n2 (0 when the pooled rate is 0 or 1 or a side is empty)"""
    if not n1 or not n2:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    if p <= 0 or p >= 1:
        return 0.0
    return (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def _z_mean2(x1, x2):
    """two-sample z for the means of x1, x2 (unequal sizes, sample variances)"""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = float(x1.mean()), float(x2.mean())
    v = float(x1.var(ddof=1)) / n1 + float(x2.var(ddof=1)) / n2
    if v == 0:
        return 0.0 if m1 == m2 else math.inf
    return (m1 - m2) / math.sqrt(v)


def ps_stat_compare(text, flags, shots, seed_gpu, seed_cpu, threads=4, launcher=None):
    S = _xtim._XdemSampler(text, 65536)
    D, O = S.num_detectors, S.num_observables
    side = []
    for gpu in (True, False):
        if gpu:
            d, o, cs, k = GpuBackend(S, launcher or RefLauncher()).sample_ps(shots, seed_gpu, flags)
        else:
            d, o, cs, k = S.sample_ps(shots, seed_cpu, threads, 0, flags.astype(bool))
        p = S.score(o, cs, np.zeros((len(cs), O), np.uint8)) if O else np.zeros(0)
        bits = np.unpackbits(d, axis=1, bitorder="little")[:, :D]
        side.append(dict(k=k, p=p, det=bits.sum(axis=0, dtype=np.int64), n=len(cs)))
    g, c = side
    named = [("accept", _z_prop2(g["n"], shots, c["n"], shots)),
             ("early", _z_prop2(g["k"]["early"], shots, c["k"]["early"], shots))]
    if O:
        named.append(("pass", _z_mean2(g["p"], c["p"])))
    named += [(f"D{j}", _z_prop2(int(a), g["n"], int(b), c["n"])) for j, (a, b) in enumerate(zip(g["det"], c["det"]))]
    arg, z = max(named, key=lambda kz: abs(kz[1]))
    return dict(D=D, O=O, acc_gpu=g["n"] / shots, acc_cpu=c["n"] / shots, early_gpu=g["k"]["early"],
                early_cpu=c["k"]["early"], max_abs_z=abs(z), argmax=arg, n_tests=len(named))


@pytest.mark.parametrize("name", STAT_NAMES)
def test_ps_statistics_vs_cpu(name):
    try:
        text = xdem_export(circuit(name))
    except xtim.XtimDemError as e:   # refused by the xdem export (tests/test_gpu_backend.py's rule)
        pytest.skip(f"{name}: refused by the xdem export ({str(e).splitlines()[0]})")
    if not xtim.XdemSampler(text)._s.device_tables()["fast"]:
        pytest.skip(f"{name}: not on the kernels' one-word fast path")
    i = STAT_NAMES.index(name)
    D = xtim.XdemSampler(text).num_detectors
    for j, kind in enumerate(MASKS):
        r = ps_stat_compare(text, _mask(D, kind), STAT_SHOTS, 81_000 + 2 * i + j, 82_000 + 2 * i + j)
        msg = (f"{name} [{kind}]: D={r['D']} O={r['O']} accept gpu {r['acc_gpu']:.6g} cpu {r['acc_cpu']:.6g} "
               f"early gpu {r['early_gpu']} cpu {r['early_cpu']} | {r['n_tests']} tests, max|z| "
               f"{r['max_abs_z']:.2f} at {r['argmax']}")
        assert r["max_abs_z"] <= Z_MAX, msg


# ------------------------------------------------------------------------------------------------- the walk's options

def _walk_launchers():
    out = [("ref", RefLauncher)]
    try:
        from xtim.gpu.metal import MetalLauncher
        MetalLauncher()
        out.append(("metal", MetalLauncher))
    except Exception:   # no Metal here: the reference alone
        pass
    return out


@pytest.mark.parametrize("launcher", [n for n, _ in _walk_launchers()])
def test_walk_options_do_not_change_the_outputs(text, launcher, monkeypatch):
    """the walk's options (dims[XK_DIM_WALK]: the postselection screen in the walk or in K2, the rows in group or
    device memory) change no output byte of sample_ps, sample or estimate, on the reference and on Metal; with the
    screen in the walk, K2 skips the shots the walk rejected (their flag is final)"""
    make = dict(_walk_launchers())[launcher]
    s = xtim.XdemSampler(text)
    flags = _mask(s.num_detectors, "half")
    n = 3 * B + 77
    outs = []
    for walk in (0, gt.XK_WALK_SCREEN, gt.XK_WALK_TGROWS, gt.XK_WALK_TGROWS | gt.XK_WALK_SCREEN):
        monkeypatch.setattr(gt, "walk_flags", lambda tables, block, w=walk: w)
        b = GpuBackend(s._s, make())
        d, o, c, counts = b.sample_ps(n, 7, flags, first_block=1)
        d2, o2, c2 = GpuBackend(s._s, make()).sample(n, 7)
        est = GpuBackend(s._s, make()).estimate(n, 7, flags)
        outs.append((d.tobytes(), o.tobytes(), c.tobytes(), sorted(counts.items()), d2.tobytes(), o2.tobytes(),
                     c2.tobytes(), est[0], est[1], est[2].tobytes()))
    assert all(x == outs[0] for x in outs[1:])
    assert dict(outs[0][3])["discarded"] > 0   # the screen had shots to reject
