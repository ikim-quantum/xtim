"""``sample(status=False)``: detectors and observables only (XK_MODE_NOKEY | XK_MODE_NOSTATUS on the GPU).

The rows (dets, obs) equal the default call's byte for byte for the same seed and arguments, with and without
postselection, on one-word files (the generic law, its law table and queue, the structured law) and multi-word files
(the generic multi-word law, the structured multi-word law), over several batchings; the host resolves no C id (the
CPU sampler's cspace_of_keys is never called and its C table does not grow); Metal equals the reference build. The
public API: GPU and CPU samplers, the argument rules (fired=True refused, a non-bool refused).
"""

from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import tables as gt
from xtim.gpu.backend import GpuBackend

from _gpu_ref import RefLauncher, multiword_text
from _xdem_synth import bh_like_text
from test_gpu_postselect import _mask

ROOT = Path(__file__).resolve().parent.parent

# (name, text factory, backend options): the law paths the option must keep byte for byte
CASES = [
    ("cultivation_d5", lambda: (ROOT / "examples/xdem/cultivation_d5.xdem").read_text(), {}),
    ("cultivation_d5_plain", lambda: (ROOT / "examples/xdem/cultivation_d5.xdem").read_text(),
     {"law_table": False, "defer": False}),
    ("rm15rep", lambda: (ROOT / "examples/xdem/rm15rep.xdem").read_text(), {}),
    ("surface_d5r5", lambda: (ROOT / "tests/data/xdem/synth/surface_d5r5.dem").read_text(), {}),
    ("bh_like36_structured", lambda: bh_like_text(36, 300, seed=36), {}),
    ("bh_like36_generic", lambda: bh_like_text(36, 300, seed=36), {"structured": False}),
    ("bh_like70_structured_mw", lambda: bh_like_text(70, 300, seed=70), {}),
    ("bh_like70_multi", lambda: bh_like_text(70, 300, seed=70), {"structured": False}),
    ("boundary_r65", lambda: multiword_text("boundary_r65"), {}),
    ("wide_c", lambda: multiword_text("wide_c"), {}),
]
IDS = [c[0] for c in CASES]


class _NoIds:
    """the CPU sampler with cspace_of_keys refused (the host must resolve no C id)"""

    def __init__(self, cpu):
        self._cpu = cpu

    def cspace_of_keys(self, keys):
        raise AssertionError("status=False resolved a C id on the host")

    def __getattr__(self, attr):
        return getattr(self._cpu, attr)


def _metal():
    try:
        from xtim.gpu.metal import MetalLauncher
        return MetalLauncher()
    except Exception:
        return None


def _pair(text, launcher=None, **kw):
    S = xtim.XdemSampler(text)._s
    return S, GpuBackend(S, launcher or RefLauncher(), **kw), GpuBackend(_NoIds(S), launcher or RefLauncher(), **kw)


@pytest.mark.parametrize("name, make, kw", CASES, ids=IDS)
def test_rows_equal_default(name, make, kw):
    text = make()
    for extra in ({}, {"max_batch_bytes": 200_000}):   # one batch; several
        S, full, bare = _pair(text, **kw, **extra)
        n = 3000
        d, o, c = full.sample(n, 7)
        assert c is not None
        nc = S.num_cspaces()
        d2, o2, c2 = bare.sample(n, 7, status=False)
        assert c2 is None and S.num_cspaces() == nc
        assert d2.dtype == o2.dtype == np.uint8 and d2.shape == d.shape and o2.shape == o.shape
        assert d2.tobytes() == d.tobytes() and o2.tobytes() == o.tobytes(), (name, extra)
        for kind in ("half", "all"):
            flags = _mask(S.num_detectors, kind)
            dp, op, cp, counts = full.sample_ps(n, 9, flags)
            nc = S.num_cspaces()
            dq, oq, cq, counts2 = bare.sample_ps(n, 9, flags, status=False)
            assert cq is None and counts2 == counts and S.num_cspaces() == nc
            assert dq.tobytes() == dp.tobytes() and oq.tobytes() == op.tobytes(), (name, extra, kind)
            assert dq.shape == dp.shape and oq.shape == op.shape


@pytest.mark.parametrize("name, make, kw", CASES, ids=IDS)
def test_first_block(name, make, kw):
    S, full, bare = _pair(make(), **kw)
    d, o, _ = full.sample(700, 3, first_block=5)
    d2, o2, _ = bare.sample(700, 3, first_block=5, status=False)
    assert d2.tobytes() == d.tobytes() and o2.tobytes() == o.tobytes()


@pytest.mark.metal
@pytest.mark.parametrize("name, make, kw", CASES, ids=IDS)
def test_metal_equals_ref(name, make, kw):
    met = _metal()
    if met is None:
        pytest.skip("no Metal device")
    text = make()
    S, ref, _ = _pair(text, **kw)
    Sm, metal_full, metal_bare = _pair(text, met, **kw)
    n = 20_000
    a = ref.sample(n, 11, status=False)
    b = metal_bare.sample(n, 11, status=False)
    full = metal_full.sample(n, 11)
    assert a[0].tobytes() == b[0].tobytes() == full[0].tobytes()
    assert a[1].tobytes() == b[1].tobytes() == full[1].tobytes()
    flags = _mask(S.num_detectors, "half")
    a = ref.sample_ps(n, 13, flags, status=False)
    b = metal_bare.sample_ps(n, 13, flags, status=False)
    full = metal_full.sample_ps(n, 13, flags)
    assert a[0].tobytes() == b[0].tobytes() == full[0].tobytes()
    assert a[1].tobytes() == b[1].tobytes() == full[1].tobytes()
    assert a[3] == b[3] == full[3]


def test_pack_mode_rules():
    t = xtim.XdemSampler((ROOT / "examples/xdem/rm15rep.xdem").read_text())._s.device_tables()
    st = gt.static_tables(t, block=64)
    for bad in (gt.XK_MODE_NOSTATUS, gt.XK_MODE_NOSTATUS | gt.XK_MODE_EMIT,
                gt.XK_MODE_NOSTATUS | gt.XK_MODE_NOKEY):
        with pytest.raises(ValueError, match="NOSTATUS"):
            gt.pack_batch(st, pair_cap=8, shots=8, seed=1, mode=bad)
    gt.pack_batch(st, pair_cap=8, shots=8, seed=1,
                  mode=gt.XK_MODE_NOSTATUS | gt.XK_MODE_NOKEY | gt.XK_MODE_EMIT)


# ------------------------------------------------------------------------------------------------- the public API


@pytest.fixture
def gpu_ref(monkeypatch):
    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", RefLauncher),))
    return lambda text: xtim.XdemSampler(text, device="gpu")


@pytest.mark.parametrize("name", ["rm15rep", "cultivation_d5"])
def test_public_gpu(gpu_ref, name):
    text = (ROOT / f"examples/xdem/{name}.xdem").read_text()
    s = gpu_ref(text)
    d, st = s.sample(2000, seed=4)
    nc = s._s.num_cspaces()
    d2, o2 = s.sample(2000, seed=4, status=False)
    assert d2.tobytes() == d.tobytes() and o2.tobytes() == st.obs.tobytes()
    idx = list(range(0, s.num_detectors, 2))
    dp, stp, counts = s.sample(2000, seed=5, postselect=idx)
    dq, oq, counts2 = s.sample(2000, seed=5, postselect=idx, status=False)
    assert dq.tobytes() == dp.tobytes() and oq.tobytes() == stp.obs.tobytes() and counts2 == counts
    dq, oq, counts2 = s.sample(300, seed=5, postselect=[], status=False)   # empty mask: sample()'s bytes
    assert dq.tobytes() == s.sample(300, seed=5, status=False)[0].tobytes() and counts2["passed"] == 300
    assert s._s.num_cspaces() >= nc


@pytest.mark.parametrize("threads", [1, 3])
def test_public_cpu(threads):
    s = xtim.XdemSampler((ROOT / "examples/xdem/rm15rep.xdem").read_text())
    d, st = s.sample(3000, seed=4, threads=threads)
    d2, o2 = s.sample(3000, seed=4, threads=threads, status=False)
    assert d2.tobytes() == d.tobytes() and o2.tobytes() == st.obs.tobytes()
    idx = list(range(0, s.num_detectors, 2))
    dp, stp, counts = s.sample(3000, seed=5, threads=threads, postselect=idx)
    dq, oq, counts2 = s.sample(3000, seed=5, threads=threads, postselect=idx, status=False)
    assert dq.tobytes() == dp.tobytes() and oq.tobytes() == stp.obs.tobytes() and counts2 == counts


def test_public_argument_rules(gpu_ref):
    text = (ROOT / "examples/xdem/rm15rep.xdem").read_text()
    for s in (xtim.XdemSampler(text), gpu_ref(text)):
        with pytest.raises(ValueError, match="fired"):
            s.sample(10, seed=1, status=False, fired=True)
        with pytest.raises(TypeError, match="status"):
            s.sample(10, seed=1, status=0)
