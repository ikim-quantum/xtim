"""The CUDA launcher (xtim/gpu/cuda.py) and the validation tool (tools/gpu_validate.py).

Three kinds of test, and what each proves:
  - ``[cuda]`` variants (marker ``cuda``): the real thing on an NVIDIA GPU with CuPy, byte-equal to the kernels' C++
    reference build (tests/_gpu_ref.py). Skipped with the reason elsewhere (tests/conftest.py).
  - ``[fake_cupy]`` variants: the same test bodies with ``cupy`` replaced by tests/_fake_cupy.py, whose kernels ARE
    the C++ reference. They exercise CudaLauncher's host-side logic only (binding order and dtypes, grid sizing,
    buffer reuse, run_batch, download / view, errors) and do NOT prove that the CUDA build compiles or runs.
  - ``test_cuda_program_*_host_cxx_syntax``: the exact program text CudaLauncher hands NVRTC, compiled as host C++14
    under a shim for the CUDA-only tokens. It catches typos, type errors and newer-than-C++14 constructs in the text,
    NOT NVRTC-specific rejections.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import _stubs
from xtim.gpu import tables as gt
from xtim.gpu.launch import Launcher

import _gpu_compare as C
from _fake_cupy import cxx, cxx_syntax_check, make_fake_cupy
from _gpu_ref import RefLauncher

ROOT = C.ROOT
HEADER = C.HEADER
B = C.B
SURFACE = (ROOT / "tests" / "data" / "xdem" / "synth" / "surface_d5r5.dem").read_text()
FILES = C.supported_files()
# every capacity macro the kernel source requires, at its smallest (a missing one is an #error, which NVRTC treats
# as fatal: a test compiling a deliberately broken source would then see that error instead of its own)
CAPS1 = {"XK_CAP_W": 1, "XK_CAP_WA": 1, "XK_CAP_RHO": 1, "XK_CAP_E": 1, "XK_CAP_J": 1,
         "XK_CAP_CW": 1, "XK_CAP_DIMC": 64, "XK_CAP_SHM": gt.shm_words(1, 1, 0)}


@pytest.fixture(params=["fake_cupy", pytest.param("cuda", marks=pytest.mark.cuda)])
def new(request, monkeypatch):
    """the CudaLauncher class: on a real CUDA device ([cuda]) or over the fake CuPy ([fake_cupy]: host-side logic
    only, the kernels run as the C++ reference)"""
    if request.param == "fake_cupy":
        monkeypatch.setitem(sys.modules, "cupy", make_fake_cupy())
    from xtim.gpu.cuda import CudaLauncher
    return CudaLauncher


@pytest.fixture(scope="module", params=FILES, ids=[p.name for p in FILES])
def sampler(request):
    return xtim.XdemSampler(request.param.read_text())


def _assert_same(ref, got):
    d = C.first_difference(ref, got)
    if d is not None:
        pytest.fail(f"{d['buffer']}[{d['index']}]: cuda {d['got']!r} != reference {d['ref']!r}")


def _both(new, t, buf, E, n):
    return C.run_kernels(RefLauncher(), t, buf, E, n), C.run_kernels(new(), t, buf, E, n)


# ------------------------------------------------------------------------------------------------ no GPU, no CuPy


def test_cpu_and_metal_paths_do_not_import_cupy():
    code = ("import sys, xtim, xtim.gpu, xtim.gpu.launch, xtim.gpu.tables, xtim.gpu.metal, xtim.gpu.cuda\n"
            f"s = xtim.XdemSampler(open({str(ROOT / 'tests/data/xdem/synth/surface_d5r5.dem')!r}).read())\n"
            "s.sample(10, seed=1)\n"
            "xtim.gpu.supports(open(" + repr(str(ROOT / 'tests/data/xdem/synth/surface_d5r5.dem')) + ").read())\n"
            "assert 'cupy' not in sys.modules, 'cupy imported'\n")
    subprocess.run([sys.executable, "-W", "error", "-c", code], check=True)


def test_missing_cupy_is_a_clear_error(monkeypatch):
    import xtim.gpu.cuda as CU
    monkeypatch.setitem(sys.modules, "cupy", None)   # `import cupy` raises ImportError
    with pytest.raises(xtim.XtimGpuError, match=r"CuPy is not installed.*pip install cupy-cuda12x"):
        CU.CudaLauncher()
    with pytest.raises(xtim.XtimGpuError, match=r"CUDA GPU unavailable.*cupy-cuda12x"):
        xtim.XdemSampler(SURFACE, device="cuda")


def test_no_cuda_device_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", make_fake_cupy(devices=0))
    from xtim.gpu.cuda import CudaLauncher
    with pytest.raises(xtim.XtimGpuError, match=r"CUDA GPU unavailable: no usable CUDA device.*cudaErrorNoDevice"):
        CudaLauncher()


def test_available_prefers_metal_then_cuda(monkeypatch):
    names = [n for n, _ in xtim.gpu._DEVICES]
    assert names == ["metal", "cuda"]
    monkeypatch.setitem(sys.modules, "Metal", None)            # no Metal
    monkeypatch.setitem(sys.modules, "cupy", make_fake_cupy())  # a (fake) CUDA device
    assert xtim.gpu.available() == "cuda"
    s = xtim.XdemSampler(SURFACE, device="gpu")
    assert s.device == "cuda"
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert xtim.gpu.available() is None


def test_metal_stubs_unchanged():
    """the Metal entry stubs, generated from the shared tables (xtim/gpu/_stubs.py), are the pinned bytes (re-pinned
    when the walk's psh buffer was added after eids: one more parameter, the later indices shifted by one; and when
    the multi-word law's scratch mws was appended as the last parameter; and when K2's queue lawq was appended after
    it, with the stub of K2b, resolve_law; and when gather's dst / out_rows / out_ckey / out_cdim gave way to boff,
    ctab, out_det, out_obs, out_cs and cmiss, appended last, with the stubs of K7 scan_blocks and K4 emit replacing
    gather's; and when the walk became a group kernel, xk_walk_group with its group's position, thread, threads and
    threadgroup memory, and took the flag buffer; and when K7 took one thread per block and the flag and hist
    buffers)"""
    import hashlib
    from xtim.gpu import metal
    assert hashlib.sha256(metal._stub_text().encode()).hexdigest() == \
        "80b39dd011d1586d5a898d3ca1352cd848ab3c68b2096b98cfa5308276e141f7"


def test_stub_tables_match_the_reference_host():
    """every kernel's buffer list names known buffers, and the stubs' element types are the reference's dtypes"""
    names = [n for n, _, _ in _stubs.BUFFERS]
    assert len(set(names)) == len(names)
    for kernel, (fn, tid, reads) in _stubs.KERNELS.items():
        assert set(reads) <= set(names) and fn.startswith("xk_") and tid
    buf = gt.pack(xtim.XdemSampler(SURFACE)._s.device_tables(), block=B, pair_cap=64, shots=B, seed=1)
    for name, ty, _ in _stubs.BUFFERS:
        if name in buf:
            assert buf[name].dtype == _stubs.DTYPES[ty], name


@pytest.mark.parametrize("path", [FILES[0], ROOT / "examples" / "xdem" / "cultivation_d5.xdem",
                                  ROOT / "tests" / "data" / "xdem" / "synth" / "boundary_r65.xdem"],
                         ids=lambda p: p.name)
def test_cuda_program_compiles_as_host_cxx_syntax(path):
    """NOT a CUDA compile: the exact text CudaLauncher.compile hands NVRTC (prelude, capacity macros, kernel source,
    extern "C" __global__ stubs) passes `c++ -std=c++14 -fsyntax-only -pedantic-errors -Werror` under a shim for the
    CUDA-only tokens, taking xdem_kernels.h's CUDA branch (boundary_r65: with the multi-word law compiled in)"""
    if cxx() is None:
        pytest.skip("no c++ on PATH")
    from xtim.gpu import cuda
    t = xtim.XdemSampler(path.read_text())._s.device_tables()
    text = cuda.program(HEADER.read_text(), gt.capacities(t, B, C.pair_cap(t)))
    assert "#elif defined(__CUDACC__)" in text and text.count('extern "C" __global__ void xk_entry_') == 8
    ok, log = cxx_syntax_check(text)
    assert ok, log


def test_cuda_program_host_cxx_syntax_check_catches_errors():
    """the syntax check above is live: a broken kernel source fails it, and so does a C++17 construct"""
    if cxx() is None:
        pytest.skip("no c++ on PATH")
    from xtim.gpu import cuda
    ok, log = cxx_syntax_check(cuda.program(HEADER.read_text(), CAPS1))
    assert ok, log   # CAPS1 defines every capacity the source requires (else the checks below pass for that reason)
    ok, log = cxx_syntax_check(cuda.program(HEADER.read_text() + "\nXK_FN u32 xk_broken_fn(u32 x) { return x +; }\n",
                                            CAPS1))
    assert not ok and "xk_broken_fn" in log
    ok, log = cxx_syntax_check(cuda.program(HEADER.read_text() + "\nXK_FN u32 xk_cxx17(u32 x) "
                                            "{ if constexpr (true) { return x; } return 0; }\n", CAPS1))
    # clang: "... is a C++17 extension"; gcc 9: "... only available with '-std=c++17'" (either spelling, any case)
    assert not ok and "c++17" in log.lower()


def test_cuda_program_rejects_bad_capacities():
    from xtim.gpu import cuda
    with pytest.raises(ValueError, match="bad capacity"):
        cuda.program("", {"XK_CAP_W": 0})


# ------------------------------------------------------------------------------------------------ the launcher
# [fake_cupy]: host-side logic over the reference kernels (NOT a CUDA test); [cuda]: on an NVIDIA GPU


def test_cuda_launcher_satisfies_the_protocol(new):
    L = new()
    assert isinstance(L, Launcher) and L.name == "cuda"
    assert L.free_bytes() > 0
    info = L.describe()
    assert info["name"] and "cupy" in info


def test_cuda_matches_reference_bit_for_bit(new, sampler):
    """the backend's submission (walk to reduce2, scan_blocks, emit) as one run_batch, every buffer, with and
    without postselection"""
    s = sampler
    t = s._s.device_tables()
    E = C.pair_cap(t)
    for kind in (None, "half"):
        buf = C.kernel_buffers(s, kind, 64 * B, 17, E)
        ref, got = _both(new, t, buf, E, 64 * B)
        _assert_same(ref, got)
        assert not ref["overflow"].any()


def test_cuda_matches_reference_partial_block(new, sampler):
    # a shot count off the block grid, and E = 2 so that blocks overflow (XK_OVERFLOW_PAIRS) when the file has
    # extended lines
    s = sampler
    t = s._s.device_tables()
    n = 64 * B - 37
    for E in (C.pair_cap(t), 2):
        buf = C.kernel_buffers(s, "all", n, 5, E, first_block=9)
        _assert_same(*_both(new, t, buf, E, n))


def test_cuda_matches_reference_rounded_up_grid(new):
    # thread counts far past the batch (the CUDA grid also rounds up to its block size): ids past nblocks / nshots
    # write nothing, so the tails past each output keep their sentinels
    s = xtim.XdemSampler((ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text())
    t = s._s.device_tables()
    E, n, tail = C.pair_cap(t), 3 * B - 5, 40
    buf = C.kernel_buffers(s, None, n, 8, E)
    size = {k: len(buf[k]) for k in ("rows", "epos", "eids", "overflow", "ckey", "cdim")}
    for k in size:
        buf[k] = np.concatenate([buf[k], np.full(tail, 0xA5, buf[k].dtype)])
    out = []
    for L in (RefLauncher(), new()):
        L.compile(HEADER.read_text(), gt.capacities(t, B, E))
        L.upload(buf)
        L.run("walk", 257)
        L.run("resolve", 1025)
        out.append({k: L.download(k) for k in size})
    _assert_same(*out)
    for k in size:
        assert (out[1][k][size[k]:] == 0xA5).all(), k


def test_cuda_outputs_are_nontrivial(new):
    s = xtim.XdemSampler((ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text())
    t = s._s.device_tables()
    E = C.pair_cap(t)
    got = C.run_kernels(new(), t, C.kernel_buffers(s, "half", 64 * B, 17, E), E, 64 * B)
    assert got["rows"].any() and got["eids"].any() and (got["cdim"] > 0).any() and got["ckey"].any()
    assert got["out_det"].any() and got["out_cs"].dtype == np.uint32 and got["cmiss"][0] > 0
    assert got["hist_total"].any() and got["boff"].any()


def test_cuda_compile_error_is_reported(new):
    if cxx() is None and "fake" in sys.modules["cupy"].__version__:
        pytest.skip("the fake CuPy reports compile errors through the host c++ syntax check")
    L = new()
    with pytest.raises(RuntimeError, match=r"CUDA compile of the xdem kernels failed") as e:
        L.compile(HEADER.read_text() + "\nXK_FN u32 xk_broken_fn(u32 x) { return x +; }\n", CAPS1)
    msg = str(e.value)   # NVRTC's log may not echo the source line: the name or the parser's complaint
    assert "error" in msg and ("xk_broken_fn" in msg or "expected" in msg)


def test_cuda_upload_alloc_reuse_and_views(new):
    t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "rm15rep.xdem").read_text())._s.device_tables()
    E = C.pair_cap(t)
    L = new()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    small = gt.pack(t, block=B, pair_cap=E, shots=2 * B, seed=3)
    L.upload(small)
    L.run("walk", 2)
    L.run("resolve", 2 * B)
    assert len(L.download("rows")) == len(small["rows"])
    big = gt.pack(t, block=B, pair_cap=E, shots=7 * B + 3, seed=4)
    L.upload(big)   # a new size: a new buffer
    L.run_batch([("walk", 8), ("resolve", 7 * B + 3)])
    R = RefLauncher()
    R.compile(HEADER.read_text(), gt.capacities(t, B, E))
    R.upload(big)
    R.run_batch([("walk", 8), ("resolve", 7 * B + 3)])
    names = ("rows", "epos", "eids", "overflow", "ckey", "cdim", "flag")
    _assert_same({k: R.download(k) for k in names}, {k: L.download(k) for k in names})
    # download(count) and view: copies of the first elements, read-only views
    v = L.view("rows", 5)
    assert v.shape == (5,) and not v.flags.writeable and np.array_equal(v, R.download("rows", 5))
    assert L.download("rows", 5).shape == (5,)
    with pytest.raises(ValueError, match="rows"):
        L.view("rows", len(big["rows"]) + 1)
    # same size: overwritten in place; a new seed changes the walk
    before = L.download("rows")
    L.upload({"seed": gt.pack(t, block=B, pair_cap=E, shots=7 * B + 3, seed=99)["seed"],
              "rows": np.zeros_like(big["rows"])})
    L.run("walk", 8)
    assert not np.array_equal(L.download("rows"), before)
    # alloc keeps a large enough buffer (its contents too) and grows a small one
    L.alloc("rows", np.uint64, 10)
    assert len(L.download("rows")) == len(big["rows"])
    L.alloc("hist", np.uint32, 3)
    L.alloc("hist", np.uint32, 1000)
    assert len(L.download("hist")) >= 1000 and L.download("hist").dtype == np.uint32
    L.alloc("hist", np.uint64, 10)   # another dtype: replaced
    assert L.download("hist").dtype == np.uint64


def test_cuda_unknown_kernel_missing_buffer_and_order(new):
    t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "rm15rep.xdem").read_text())._s.device_tables()
    L = new()
    with pytest.raises(RuntimeError, match="before compile"):
        L.run("walk", 1)
    L.compile(HEADER.read_text(), gt.capacities(t, B, 64))
    with pytest.raises(ValueError, match="kernel"):
        L.run("nope", 1)
    with pytest.raises(ValueError, match="dims"):
        L.run("walk", 1)
    with pytest.raises(ValueError, match="out of range"):
        L.run("walk", -1)
    L.run_batch([])   # nothing to do


def test_cuda_launch_loop_matches_reference(new):
    """GpuBackend on CudaLauncher: sample / sample_ps / estimate bytes of the reference launcher (default batches and
    many small ones: alloc reuse, views, run_batch)"""
    for path in ("examples/xdem/cultivation_d5.xdem", "tests/data/xdem/synth/surface_d5r5.dem"):
        text = (ROOT / path).read_text()
        for kind in (None, "half"):
            want = C.entry_outputs(text, kind, RefLauncher, 2 * 1024 + 37, 2026)
            for kw in ({}, {"max_batch_bytes": 1}):
                got = C.entry_outputs(text, kind, new, 2 * 1024 + 37, 2026, **kw)
                _assert_same(want, got)


def test_cuda_launch_grid_and_sync(monkeypatch):
    """[fake_cupy only] the launch configuration CudaLauncher hands CuPy: grid = ceil(n / block) blocks of
    min(256, max_threads_per_block) threads (the walk, a group kernel: n blocks of min(its threads, 256 by default,
    max_threads_per_block)), one deviceSynchronize per run_batch, options ("-std=c++14",); a kernel
    error raised at the synchronize becomes a RuntimeError naming the kernels"""
    fake = make_fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    from xtim.gpu.cuda import CudaLauncher
    s = xtim.XdemSampler(SURFACE)
    t = s._s.device_tables()
    E = C.pair_cap(t)
    L = CudaLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    assert fake.log["options"] == [("-std=c++14",)]
    L.upload(C.kernel_buffers(s, None, 1000, 1, E))
    fake.log["launches"].clear()
    syncs = fake.log["syncs"]
    L.run_batch([("walk", 16), ("resolve", 1000), ("reduce", 0), ("reduce2", 69)])
    assert fake.log["launches"] == [("walk", (16,), (256,)), ("resolve", (4,), (256,)), ("reduce2", (1,), (256,))]
    assert fake.log["syncs"] == syncs + 1
    fake.log["launches"].clear()
    L.run_batch([("walk", 16, 64)])
    assert fake.log["launches"] == [("walk", (16,), (64,))]
    fake.RawKernel.max_threads_per_block = 128
    L.compile(HEADER.read_text(), gt.capacities(t, B, E))
    fake.log["launches"].clear()
    L.run("resolve", 1000)
    L.run("walk", 16, 512)
    assert fake.log["launches"] == [("resolve", (8,), (128,)), ("walk", (16,), (128,))]
    fake.fail_next_sync = fake.cuda.runtime.CUDARuntimeError("cudaErrorIllegalAddress: an illegal memory access")
    with pytest.raises(RuntimeError, match=r"CUDA kernel 'walk' failed.*illegal memory access"):
        L.run("walk", 16)


@pytest.mark.cuda
def test_cuda_sample_roundtrip():
    s = xtim.XdemSampler(SURFACE, device="cuda")
    assert s.device == "cuda"
    d, st = s.sample(10_000, seed=3)
    assert d.shape[0] == 10_000 and d.any()


# ------------------------------------------------------------------------------------------------ the tool


def test_validate_tool_runs_without_gpu(tmp_path):
    """tools/gpu_validate.py --quick with no device: exit 0, device null, every GPU section skipped, the CPU
    benchmark still measured"""
    out = tmp_path / "r.json"
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_validate.py"), "--quick", "--device", "none",
                        "--out", str(out)], capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    r = json.loads(out.read_text())
    assert r["device"] is None and r["passed"] is True
    for name in ("tests", "kernels", "entry_points", "stress"):
        assert r["sections"][name]["status"] == "skipped", name
    bench = r["sections"]["benchmark"]
    assert bench["status"] == "passed"
    assert all(row["device"] is None and row["cpu_1"] > 0 and row["cpu_all"] > 0 for row in bench["rows"])
    inf = [row for row in bench["rows"] if row["entry"].startswith("infidelity")]
    assert inf and all(row["cpu_score_1"] > 0 and row["cpu_score_all"] > 0 for row in inf)   # both CPU baselines
    assert all(row["cpu_best"] == max(v for k, v in row.items() if k.startswith("cpu_") and k != "cpu_best"
                                      and k != "cpu_shots") for row in bench["rows"])
    assert "device: none" in p.stdout and "PASS" in p.stdout


def test_validate_tool_cuda_path_over_fake_cupy(tmp_path, monkeypatch):
    """NOT a CUDA test: the tool's CUDA code path (device description, kernel and entry-point bit-equality,
    benchmark) run in-process over the fake CuPy, so that its first run on an NVIDIA machine does not stumble on
    its own code; a corrupted kernel output is reported with its buffer, index and values"""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import gpu_validate as V
    monkeypatch.setitem(sys.modules, "cupy", make_fake_cupy())
    monkeypatch.setitem(sys.modules, "Metal", None)
    out = tmp_path / "r.json"
    code = V.main(["--quick", "--device", "cuda", "--skip-tests", "--files", "cultivation_d5,toy2obs",
                   "--bench-shots", "3000", "--out", str(out)])
    r = json.loads(out.read_text())
    assert code == 0 and r["passed"], json.dumps(r["sections"], indent=1)[:3000]
    assert r["device"] == "cuda" and r["device_info"]["name"] == "Fake CuPy device"
    assert r["sections"]["tests"]["status"] == "skipped"
    assert r["sections"]["kernels"]["status"] == "passed" and r["sections"]["entry_points"]["status"] == "passed"
    rows = r["sections"]["benchmark"]["rows"]
    assert rows and all(row["device"] > 0 and row["device_compile_s"] >= 0 for row in rows)
    # a launcher that corrupts one element of eids: the kernels section fails and names it
    from xtim.gpu.cuda import CudaLauncher

    class Broken(CudaLauncher):
        def download(self, name, count=None):
            a = super().download(name, count)
            if name == "eids" and len(a) > 3:
                a[3] ^= 1
            return a

    monkeypatch.setattr(V, "_launcher_class", lambda device: Broken)
    code = V.main(["--quick", "--device", "cuda", "--skip-tests", "--skip-bench", "--files", "cultivation_d5",
                   "--out", str(out)])
    r = json.loads(out.read_text())
    assert code == 1 and not r["passed"] and r["sections"]["kernels"]["status"] == "failed"
    bad = [c for c in r["sections"]["kernels"]["cases"] if not c["ok"]]
    assert bad and bad[0]["diff"]["buffer"] == "eids" and bad[0]["diff"]["index"] == 3


def test_validate_tool_stress_section_over_fake_cupy(monkeypatch):
    """NOT a CUDA test: the tool's stress section (threads, pair capacity past 2^17, submission times) over the fake
    CuPy, with a small heavy-DEM shot count"""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import gpu_validate as V
    monkeypatch.setitem(sys.modules, "cupy", make_fake_cupy())
    monkeypatch.setitem(sys.modules, "Metal", None)
    monkeypatch.setattr(V, "HEAVY_DEM_SHOTS", (3000, 3000))
    sec = V._section_stress("cuda", True)
    assert sec["status"] == "passed", json.dumps(sec, indent=1)[:3000]
    cases = {c["case"]: c for c in sec["cases"]}
    assert cases["pair_capacity"]["pair_cap_relaunched"] > 1 << 17 and cases["threads"]["calls"] > 0
    assert cases["submission_time"]["submissions"]
