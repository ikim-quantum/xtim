"""The Metal launcher (xtim/gpu/metal.py) runs the xdem kernels byte-identically to their C++ reference build.

The same packed buffers go through `RefLauncher` (tests/_gpu_ref.py, the kernel source compiled as plain C++) and
`MetalLauncher`; every output buffer of "walk" then "resolve" must be byte-equal. Tests marked `metal` are skipped
(with the reason) when there is no Metal device or pyobjc is missing (tests/conftest.py).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import tables as gt
from xtim.gpu.launch import Launcher

from _gpu_ref import RefLauncher

ROOT = Path(__file__).parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = 64
NBLOCKS = 64
OUTS = ("rows", "epos", "eids", "overflow", "ckey", "cdim")
SENTINEL = {"rows": np.uint64(0xA5A5A5A5A5A5A5A5), "epos": np.uint32(0xDEADBEEF), "eids": np.uint32(0xDEADBEEF),
            "overflow": np.uint32(0xDEADBEEF), "ckey": np.uint64(0xA5A5A5A5A5A5A5A5), "cdim": np.uint8(0xA5)}


def _supported(path):
    t = xtim.XdemSampler(path.read_text())._s.device_tables()
    return not t["A"] or t["fast"]


FILES = sorted(p for p in (ROOT / "examples" / "xdem").glob("*.xdem") if _supported(p))


@pytest.fixture(scope="module", params=FILES, ids=[p.name for p in FILES])
def tables(request):
    return xtim.XdemSampler(request.param.read_text())._s.device_tables()


def _pair_cap(t):
    return int(B * t["mean_ext"] + 10 * np.sqrt(B * t["mean_ext"] + 1) + 64)


def _metal():
    from xtim.gpu.metal import MetalLauncher
    return MetalLauncher()


def _both(t, buf, pair_cap, walk_n, resolve_n):
    """run "walk" (walk_n threads) then "resolve" (resolve_n threads) through both launchers; their outputs"""
    out = []
    for L in (RefLauncher(), _metal()):
        L.compile(HEADER.read_text(), gt.capacities(t, B, pair_cap))
        L.upload(buf)
        L.run("walk", walk_n)
        L.run("resolve", resolve_n)
        out.append({k: L.download(k) for k in OUTS})
    return out


def _assert_equal(ref, got):
    for k in OUTS:
        assert got[k].dtype == ref[k].dtype and got[k].shape == ref[k].shape, k
        if not np.array_equal(got[k], ref[k]):
            i = int(np.flatnonzero(got[k] != ref[k])[0])
            pytest.fail(f"{k}[{i}]: metal {got[k][i]!r} != reference {ref[k][i]!r}")


# ------------------------------------------------------------------------------------------------ no Metal needed


def test_cpu_path_does_not_import_pyobjc():
    code = ("import sys, xtim, xtim.gpu, xtim.gpu.launch, xtim.gpu.tables, xtim.gpu.metal\n"
            "bad = [m for m in ('objc', 'Metal', 'Foundation') if m in sys.modules]\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ref_launcher_satisfies_the_protocol():
    assert isinstance(RefLauncher(), Launcher)


def test_missing_pyobjc_is_a_clear_error(monkeypatch):
    import xtim.gpu.metal as M
    monkeypatch.setitem(sys.modules, "Metal", None)   # `import Metal` raises ImportError
    with pytest.raises(xtim.XtimGpuError, match=r'pip install "xtim\[metal\]"'):
        M.MetalLauncher()


# ------------------------------------------------------------------------------------------------ on the GPU


@pytest.mark.metal
def test_metal_launcher_satisfies_the_protocol():
    L = _metal()
    assert isinstance(L, Launcher) and L.name == "metal"
    assert L.free_bytes() > 0


@pytest.mark.metal
def test_metal_matches_reference_bit_for_bit(tables):
    t = tables
    E = _pair_cap(t)
    buf = gt.pack(t, block=B, pair_cap=E, shots=NBLOCKS * B, seed=17)
    ref, got = _both(t, buf, E, NBLOCKS, NBLOCKS * B)
    _assert_equal(ref, got)
    assert not ref["overflow"].any()


@pytest.mark.metal
def test_metal_matches_reference_partial_block(tables):
    # a shot count that is not a multiple of B (the last block is partial), and a small E so that some blocks
    # overflow (XK_OVERFLOW_PAIRS) when the file has extended lines
    t = tables
    n = NBLOCKS * B - 37
    for E in (_pair_cap(t), 2):
        buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=5, first_block=9)
        nb = int(buf["dims"][gt.XK_DIM_NBLOCKS])
        ref, got = _both(t, buf, E, nb, n)
        _assert_equal(ref, got)


@pytest.mark.metal
def test_metal_matches_reference_rounded_up_grid(tables):
    # n rounded up past the batch (ids >= nblocks / nshots write nothing): the tails past the outputs keep their
    # sentinels on both launchers
    t = tables
    E, n, tail = _pair_cap(t), 3 * B - 5, 40
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=8)
    size = {k: len(buf[k]) for k in OUTS}
    for k in OUTS:
        buf[k] = np.concatenate([buf[k], np.full(tail, SENTINEL[k], buf[k].dtype)])
    ref, got = _both(t, buf, E, 256, 1024)
    _assert_equal(ref, got)
    for k in OUTS:
        assert (got[k][size[k]:] == SENTINEL[k]).all(), k


@pytest.mark.metal
def test_metal_outputs_are_nontrivial():
    # the bit-equality above compares real work: rows fired, extended lists filled, nontrivial laws resolved
    t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text())._s.device_tables()
    E = _pair_cap(t)
    buf = gt.pack(t, block=B, pair_cap=E, shots=NBLOCKS * B, seed=17)
    _, got = _both(t, buf, E, NBLOCKS, NBLOCKS * B)
    assert got["rows"].any() and got["eids"].any() and (got["cdim"] > 0).any() and got["ckey"].any()


@pytest.mark.metal
def test_metal_compile_error_is_reported():
    L = _metal()
    src = HEADER.read_text() + "\nXK_FN u32 xk_broken_fn(u32 x) { return x +; }\n"
    with pytest.raises(RuntimeError, match="xk_broken_fn|expected expression") as e:
        L.compile(src, {"XK_CAP_W": 1, "XK_CAP_WA": 1, "XK_CAP_RHO": 1, "XK_CAP_E": 1, "XK_CAP_J": 1,
                           "XK_CAP_CW": 1, "XK_CAP_DIMC": 64})
    assert "error" in str(e.value)


@pytest.mark.metal
def test_metal_upload_replaces_buffer():
    t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "rm15rep.xdem").read_text())._s.device_tables()
    E = _pair_cap(t)
    L = _metal()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    small = gt.pack(t, block=B, pair_cap=E, shots=2 * B, seed=3)
    L.upload(small)
    L.run("walk", 2)
    L.run("resolve", 2 * B)
    assert len(L.download("rows")) == len(small["rows"])
    big = gt.pack(t, block=B, pair_cap=E, shots=7 * B + 3, seed=4)
    assert len(big["rows"]) != len(small["rows"])
    L.upload(big)
    L.run("walk", 8)
    L.run("resolve", 7 * B + 3)
    got = {k: L.download(k) for k in OUTS}
    R = RefLauncher()
    R.compile(HEADER.read_text(), gt.capacities(t, B, E))
    R.upload(big)
    R.run("walk", 8)
    R.run("resolve", 7 * B + 3)
    _assert_equal({k: R.download(k) for k in OUTS}, got)
    # replacing one buffer alone: a new seed changes the walk
    L.upload({"seed": gt.pack(t, block=B, pair_cap=E, shots=7 * B + 3, seed=99)["seed"],
              "rows": np.zeros_like(big["rows"])})
    L.run("walk", 8)
    assert not np.array_equal(L.download("rows"), got["rows"])
    assert L.download("rows", 5).shape == (5,)


@pytest.mark.metal
def test_metal_unknown_kernel_and_missing_buffer():
    t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "rm15rep.xdem").read_text())._s.device_tables()
    L = _metal()
    L.compile(HEADER.read_text(), gt.capacities(t, B, 64))
    with pytest.raises(ValueError, match="kernel"):
        L.run("nope", 1)
    with pytest.raises(ValueError, match="dims"):
        L.run("walk", 1)
