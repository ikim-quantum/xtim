"""Multi-word files on the GPU (coord_words >= 2: the multi-word law, xk_law_multi, any number of gauge qubits).

Files (tests/_gpu_ref.py's MULTIWORD): boundary_r65 (r = 65, two words, rhoL = 1), cw8_r400 (r = 400, eight words),
wide_c (r = 90, rhoL = 68: about 40% of its shots have dim C > 64, past the one-word counters, and about a quarter
pass the zero decoder there).

Exact properties: Metal gives the reference build's bytes (sample, sample_ps, estimate); estimate's counts are those
of sample / sample_ps followed by score with zero flips, whatever the batch size; a shot with an error code raises
XtimInvariantError. Statistics vs the CPU sampler (the GPU's coins differ from the CPU's, the law is the same):
sample (pass mean, detector marginals, detector x failure joints), sample(postselect=) (acceptance, early
rejections, pass mean, detector marginals) and infidelity(None) (means within 5 combined standard errors). The
statistical tests run on Metal (the reference build takes minutes per million shots of cw8_r400); the reference is
checked byte for byte against Metal above.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim.gpu import tables as gt
from xtim.gpu.backend import GpuBackend
from xtim.xdem import XdemStatus

from _gpu_ref import MULTIWORD, RefLauncher, multiword_text
from test_gpu_backend import stat_compare
from test_gpu_estimate import _Corrupt
from test_gpu_postselect import ps_stat_compare
from test_xdem_cross_route import Z_MAX


def _metal():
    from xtim.gpu.metal import MetalLauncher
    return MetalLauncher()


def _first(D, k):
    """uint8[D]: the first min(k, D) detectors (every multi-word file keeps a usable acceptance)"""
    f = np.zeros(D, np.uint8)
    f[:k] = 1
    return f


def _same(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b))


def _hist_of_scores(p, size):
    """int64[size]: the shots with p = 2^-k > 0, by k (p is 0 or a power of two)"""
    h = np.zeros(size, np.int64)
    pos = p[p > 0]
    k = np.round(-np.log2(pos)).astype(np.int64)
    assert (np.ldexp(1.0, -k) == pos).all()
    np.add.at(h, k, 1)
    return h


SMALL = {"boundary_r65": 3000, "cw8_r400": 300, "wide_c": 1500}


@pytest.mark.parametrize("name", MULTIWORD)
def test_tables_are_multiword(name):
    t = xtim.XdemSampler(multiword_text(name))._s.device_tables()
    assert gt.coord_words(t) >= 2
    assert xtim.gpu.supports(multiword_text(name)) == (True, None)


# ------------------------------------------------------------------------------------------------- exact properties


@pytest.mark.metal
@pytest.mark.parametrize("name", MULTIWORD)
def test_metal_equals_ref(name):
    s = xtim.XdemSampler(multiword_text(name))
    n = SMALL[name]
    ref, met = GpuBackend(s._s, RefLauncher()), GpuBackend(s._s, _metal())
    a = met.sample(n, 5)
    assert _same(a, ref.sample(n, 5)) and a[0].any()
    flags = _first(s.num_detectors, 2)
    a, b = met.sample_ps(n, 6, flags), ref.sample_ps(n, 6, flags)
    assert _same(a[:3], b[:3]) and a[3] == b[3] and len(a[2]) > 0
    for f in (None, flags):
        a, b = met.estimate(n, 7, f), ref.estimate(n, 7, f)
        assert a[:2] == b[:2] and a[2].tolist() == b[2].tolist()


@pytest.mark.parametrize("name", ["boundary_r65", "wide_c"])
@pytest.mark.parametrize("ps", [False, True])
def test_estimate_equals_sample_then_score(name, ps):
    s = xtim.XdemSampler(multiword_text(name))
    t = s._s.device_tables()
    O = s.num_observables
    n = SMALL[name]
    size = gt.dimc_of(t) + 1
    flags = _first(s.num_detectors, 2) if ps else None
    ref = GpuBackend(s._s, RefLauncher())
    if flags is None:
        _, obs, cs = ref.sample(n, 9, first_block=3)
        passed = n
    else:
        _, obs, cs, counts = ref.sample_ps(n, 9, flags, first_block=3)
        passed = counts["passed"]
    st = XdemStatus(obs, cs, None, s, s._s.num_cspaces())
    p = s.score(st, np.zeros((len(cs), O), bool))
    want = _hist_of_scores(p, size)
    assert want.sum() > 0 and passed > 0
    if name == "wide_c":
        dims = np.array([len(c) for c in st.cspaces])[cs]
        assert dims.max() > 64   # past the one-word counters
        assert want[65:].sum() > 0
    for mbb in (None, 1):   # one batch; one block per batch
        b = GpuBackend(s._s, RefLauncher(), max_batch_bytes=mbb)
        att, got_passed, h = b.estimate(n, 9, flags, first_block=3)
        assert (att, got_passed) == (n, passed), mbb
        assert h.dtype == np.int64 and h.shape == (size,)
        assert h.tolist() == want.tolist(), mbb
        assert sum(Fraction(int(c), 2**k) for k, c in enumerate(h)) == sum(Fraction(x) for x in p)
        if mbb == 1:
            assert b.batches_run > 1


@pytest.mark.parametrize("value,match", [(gt.XK_CDIM_CAPACITY, "capacity"), (gt.XK_CDIM_INTERNAL, "internal")])
@pytest.mark.parametrize("entry", ["sample", "sample_ps", "estimate"])
def test_error_code_shot_raises(value, match, entry):
    s = xtim.XdemSampler(multiword_text("boundary_r65"))
    b = GpuBackend(s._s, _Corrupt(value))
    flags = _first(s.num_detectors, 1)
    call = {"sample": lambda: b.sample(500, 1), "sample_ps": lambda: b.sample_ps(500, 1, flags),
            "estimate": lambda: b.estimate(500, 1, flags)}[entry]
    with pytest.raises(xtim.XtimInvariantError, match=match):
        call()


class _TightRef(RefLauncher):
    """the reference build with XK_CAP_CW = 1 and XK_CAP_E = 256 (no multi-word law compiled), whatever compile asks"""

    def compile(self, source, capacities):
        super().compile(source, capacities)
        self._variant = 1


def test_kernel_capacity_code_reaches_the_host():
    """a build whose XK_CAP_CW is below the file's coordinate words computes nothing and says so (the kernels' own
    XK_CDIM_CAPACITY / XK_OVERFLOW_CAPACITY), and every entry raises instead of answering"""
    s = xtim.XdemSampler(multiword_text("boundary_r65"))
    flags = _first(s.num_detectors, 1)
    for call in (lambda b: b.sample(200, 1), lambda b: b.sample_ps(200, 1, flags),
                 lambda b: b.estimate(200, 1, None), lambda b: b.estimate(200, 1, flags)):
        with pytest.raises(xtim.XtimInvariantError, match="capacity"):
            call(GpuBackend(s._s, _TightRef()))


@pytest.mark.metal
def test_metal_capacity_code_reaches_the_host():
    """the same on Metal: cw8_r400 (eight words) under a build compiled for two"""
    from xtim.gpu.metal import MetalLauncher

    class _Narrow(MetalLauncher):
        def compile(self, source, capacities):
            super().compile(source, dict(capacities, XK_CAP_CW=2))
    s = xtim.XdemSampler(multiword_text("cw8_r400"))
    for call in (lambda b: b.sample(100, 1), lambda b: b.estimate(100, 1, None)):
        with pytest.raises(xtim.XtimInvariantError, match="capacity"):
            call(GpuBackend(s._s, _Narrow()))


# ------------------------------------------------------------------------------------- statistics vs the CPU sampler


STAT = {"boundary_r65": 2_000_000, "cw8_r400": 100_000, "wide_c": 300_000}


@pytest.mark.metal
@pytest.mark.parametrize("name", MULTIWORD)
def test_statistics_vs_cpu(name):
    i = MULTIWORD.index(name)
    r = stat_compare(multiword_text(name), STAT[name], 171_000 + i, 172_000 + i, launcher=_metal())
    msg = (f"{name}: D={r['D']} O={r['O']} 1-pass gpu {1 - r['pass_gpu']:.6g} cpu {1 - r['pass_cpu']:.6g} "
           f"z_pass {r['z_pass']} | {r['n_tests']} tests, max|z| {r['max_abs_z']:.2f} at {r['argmax']}")
    print(msg)
    assert r["n_tests"] >= r["D"]
    assert r["max_abs_z"] <= Z_MAX, msg


@pytest.mark.metal
@pytest.mark.parametrize("name", MULTIWORD)
def test_ps_statistics_vs_cpu(name):
    i = MULTIWORD.index(name)
    D = xtim.XdemSampler(multiword_text(name)).num_detectors
    r = ps_stat_compare(multiword_text(name), _first(D, 2), STAT[name], 181_000 + i, 182_000 + i, launcher=_metal())
    msg = (f"{name}: D={r['D']} O={r['O']} accept gpu {r['acc_gpu']:.6g} cpu {r['acc_cpu']:.6g} early gpu "
           f"{r['early_gpu']} cpu {r['early_cpu']} | {r['n_tests']} tests, max|z| {r['max_abs_z']:.2f} at {r['argmax']}")
    print(msg)
    assert r["acc_gpu"] > 0.1
    assert r["max_abs_z"] <= Z_MAX, msg


@pytest.mark.metal
@pytest.mark.parametrize("name", MULTIWORD)
@pytest.mark.parametrize("ps", [False, True])
def test_infidelity_none_statistics_vs_cpu(name, ps):
    text = multiword_text(name)
    g = xtim.XdemSampler(text, device="metal")
    c = xtim.XdemSampler(text)
    i = 2 * MULTIWORD.index(name) + ps
    flags = _first(c.num_detectors, 2).astype(bool) if ps else None
    rg = g.infidelity(STAT[name], None, seed=191_000 + i, postselect=flags)
    rc = c.infidelity(STAT[name], None, seed=192_000 + i, postselect=flags, threads=4)
    msg = (f"{name} [ps={ps}]: gpu {rg.mean:.6g} +- {rg.stderr:.3g} ({rg.accepted}) cpu {rc.mean:.6g} +- "
           f"{rc.stderr:.3g} ({rc.accepted})")
    print(msg)
    assert rg.accepted > 0 and rc.accepted > 0
    assert abs(rg.accepted - rc.accepted) <= 5 * math.sqrt(STAT[name]) or not ps
    se = math.hypot(rg.stderr, rc.stderr)
    assert se > 0 and abs(rg.mean - rc.mean) <= 5 * se, msg
