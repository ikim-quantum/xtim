"""The public GPU API (Task 7): ``XdemSampler(text, device=...)``, ``XtimGpuError``, ``xtim.gpu.available()`` /
``supports()``. The default device is the CPU path, untouched: nothing GPU is imported and the bytes are the pinned
golden's. Every way the GPU can be unusable (no device, extras missing, unsupported file, fired lines) raises
``XtimGpuError`` naming the cause, and ``device="cpu"`` still works on the same text (post-selection on the GPU:
tests/test_gpu_postselect.py)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import xtim
import xtim.gpu
from xtim.errors import XtimError

from _gpu_ref import RefLauncher

ROOT = Path(__file__).parent.parent
SYNTH = ROOT / "tests" / "data" / "xdem" / "synth"
SURFACE = (SYNTH / "surface_d5r5.dem").read_text()
R65 = (SYNTH / "boundary_r65.xdem").read_text()
R64 = (SYNTH / "boundary_r64.xdem").read_text()
PINS = json.loads((ROOT / "tests" / "data" / "xdem" / "plain_sampler_sha256.json").read_text())


def test_cpu_default_unchanged_and_imports_nothing():
    code = f"""
import hashlib, sys
import xtim
s = xtim.XdemSampler(open({str(SYNTH / 'surface_d5r5.dem')!r}).read())
assert s.device == "cpu"
d, st = s.sample(5000, seed=7, threads=1)
print(hashlib.sha256(d.tobytes() + st.obs.tobytes() + st.cspace.tobytes()).hexdigest())
import xtim.gpu
print(sorted(m for m in ("objc", "Metal", "cupy", "xtim.gpu.metal", "xtim.gpu.backend") if m in sys.modules))
"""
    out = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, check=True)
    digest, loaded = out.stdout.split("\n")[:2]
    assert digest == PINS["surface_d5r5"]["5000_7_1"]    # the pinned golden (fixture = the pins' surface_d5r5)
    assert loaded == "[]"


def test_importing_xtim_gpu_is_light():
    code = ("import sys, xtim.gpu\n"
            "print([m for m in ('objc', 'Metal', 'cupy', 'xtim.gpu.metal', 'xtim.gpu.backend') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_error_class():
    assert issubclass(xtim.XtimGpuError, XtimError) and "XtimGpuError" in xtim.__all__
    assert not issubclass(xtim.XtimGpuError, (ValueError, RuntimeError))


def test_unknown_device():
    for bad in ("tpu", "GPU", "", None, 3):
        with pytest.raises(ValueError, match=r'"cpu", "gpu", "metal" or "cuda"'):
            xtim.XdemSampler(SURFACE, device=bad)


def test_gpu_unavailable(monkeypatch):
    monkeypatch.setattr(xtim.gpu, "available", lambda: None)
    with pytest.raises(xtim.XtimGpuError, match=r'(?s)no usable GPU.*xtim\[metal\].*CuPy.*device="cpu"'):
        xtim.XdemSampler(SURFACE, device="gpu")
    assert xtim.XdemSampler(SURFACE, device="cpu").sample(10, seed=1)[0].shape[0] == 10


def test_missing_pyobjc_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "Metal", None)
    assert xtim.gpu.available() is None            # never raises
    for dev in ("gpu", "metal"):
        with pytest.raises(xtim.XtimGpuError, match=r'pip install "xtim\[metal\]"'):
            xtim.XdemSampler(SURFACE, device=dev)


def test_cuda_without_cupy_names_the_wheel(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)   # `import cupy` raises ImportError
    with pytest.raises(xtim.XtimGpuError, match=r"CUDA GPU unavailable: CuPy is not installed.*cupy-cuda12x"):
        xtim.XdemSampler(SURFACE, device="cuda")


def test_supported_files():
    """any number of gauge qubits: r = 64 (one word) and r = 65 (two words) both run on the GPU"""
    assert xtim.gpu.supports(R65) == (True, None)
    assert xtim.gpu.supports(R64) == (True, None)
    assert xtim.gpu.supports(SURFACE) == (True, None)
    assert xtim.XdemSampler(R65).sample(10, seed=1)[0].shape == (10, 8)    # the CPU path serves it too


def test_unsupported_file():
    """a file off the fast path (the general path: coord_words = 0) is refused with the numbers, and so is every
    device name"""
    class General:
        def __init__(self, cpu):
            self.cpu = cpu

        def device_tables(self):
            t = dict(self.cpu.device_tables())
            t["coord_words"], t["fast"] = 0, False
            return t
    ok, reason = xtim.gpu._supports_cpu(General(xtim.XdemSampler(R65)._s))
    assert not ok and "general path" in reason and "coord_words = 0" in reason and "device='cpu'" in reason


class _FakeBackend:
    _block = 64

    def __init__(self, cpu):
        self.cpu = cpu

    def sample(self, shots, seed, *, first_block=0):
        D, O = self.cpu.num_detectors, self.cpu.num_observables
        return (np.full((shots, (D + 7) // 8), 5, np.uint8), np.zeros((shots, (O + 7) // 8), np.uint8),
                np.zeros(shots, np.int32))


@pytest.fixture
def gpu_sampler(monkeypatch):
    """a GPU-routed sampler without a device: the Metal opener and the backend are replaced (routing tests only)"""
    import xtim.gpu.backend as B
    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", lambda: object()),))
    monkeypatch.setattr(B, "GpuBackend", lambda cpu, launcher: _FakeBackend(cpu))
    return xtim.XdemSampler(SURFACE, device="gpu")


def test_gpu_routing(gpu_sampler):
    s = gpu_sampler
    assert s.device == "metal"
    dets, st = s.sample(100, seed=3, threads=4)
    assert (dets == 5).all() and len(st) == 100 and isinstance(st, xtim.xdem.XdemStatus)
    assert s.score(st, np.zeros((100, s.num_observables), bool)).shape == (100,)
    r = s.infidelity(2500, lambda d: np.zeros((len(d), s.num_observables), bool), seed=1, batch=1000)
    assert r.shots == r.accepted == 2500


def test_fired_and_lines_refused(gpu_sampler):
    s = gpu_sampler
    dets, st = s.sample(10, seed=1)
    cpu = xtim.XdemSampler(SURFACE)
    _, cst = cpu.sample(10, seed=1, fired=True)
    lines = [[] for _ in range(10)]
    st.fired, st._dets = cst.fired, cst._dets      # a status with fired lines, to reach score(lines=)
    calls = [lambda: s.sample(10, seed=1, fired=True),
             lambda: s.score(st, lines=lines),
             lambda: s.correction(st, lines=lines),
             lambda: s.infidelity(10, lambda d: [[]] * len(d), seed=1, lines=True)]
    for call in calls:
        with pytest.raises(xtim.XtimGpuError, match=r"fired lines.*device='cpu'"):
            call()


def test_seed_and_shots_validation_shared(gpu_sampler):
    cpu = xtim.XdemSampler(SURFACE)
    for kw in ({"shots": -1, "seed": 1}, {"shots": "x", "seed": 1}, {"shots": 1, "seed": -1},
               {"shots": 1, "seed": 1.5}, {"shots": 1, "seed": 1, "threads": 0}, {"shots": 1, "seed": 1, "fired": 1}):
        shots = kw.pop("shots")
        with pytest.raises((TypeError, ValueError)) as ec:
            cpu.sample(shots, **kw)
        with pytest.raises(type(ec.value), match=re.escape(str(ec.value))):
            gpu_sampler.sample(shots, **kw)


def test_infidelity_chunks_the_stream(monkeypatch):
    """GPU infidelity draws one chunk (batch rounded up to whole blocks) at a time from a real (reference-launcher)
    backend: the same result for any batch, the decoder called once per chunk, never more than a chunk of rows."""
    import xtim.gpu.backend as B
    asked = []

    class Recording(B.GpuBackend):
        def sample(self, shots, seed, *, first_block=0):
            asked.append((shots, first_block))
            return super().sample(shots, seed, first_block=first_block)

    monkeypatch.setattr(xtim.gpu, "_DEVICES", (("metal", RefLauncher),))
    monkeypatch.setattr(B, "GpuBackend", Recording)
    s = xtim.XdemSampler(SURFACE, device="gpu")
    O = s.num_observables
    calls = []

    def dec(dets):
        calls.append(len(dets))
        return np.zeros((len(dets), O), bool) | (dets[:, :O] if dets.shape[1] >= O else False)

    shots = 5000
    one = s.infidelity(shots, dec, seed=4, batch=10**6)
    assert asked == [(shots, 0)] and calls == [shots]
    asked.clear(), calls.clear()
    many = s.infidelity(shots, dec, seed=4, batch=1000)      # 1000 -> 1024 rows per chunk (GPU blocks of 512)
    assert many == one
    g = xtim.gpu.backend.DEFAULT_BLOCK
    assert asked == [(1024, 0), (1024, 1024 // g), (1024, 2048 // g), (1024, 3072 // g), (904, 4096 // g)]
    assert calls == [1024, 1024, 1024, 1024, 904]
    # a failing decoder stops the run after its first chunk
    asked.clear()
    with pytest.raises(ZeroDivisionError):
        s.infidelity(shots, lambda d: 1 / 0, seed=4, batch=1000)
    assert asked == [(1024, 0)]


@pytest.mark.metal
def test_gpu_sample_roundtrip():
    s = xtim.XdemSampler(SURFACE, device="gpu")
    cpu = xtim.XdemSampler(SURFACE)
    assert s.device == "metal" and xtim.gpu.available() == "metal"
    dets, st = s.sample(3000, seed=5, threads=3)
    cd, cst = cpu.sample(3000, seed=5)
    assert dets.dtype == cd.dtype and dets.shape == cd.shape
    assert st.obs.shape == cst.obs.shape and st.cspace.dtype == cst.cspace.dtype and len(st) == 3000
    assert np.array_equal(dets, s.sample(3000, seed=5)[0])
    sc = s.score(st, np.zeros((3000, s.num_observables), bool))
    assert sc.shape == (3000,) and ((sc >= 0) & (sc <= 1)).all()
    ok, _ = xtim.gpu.supports(SURFACE)
    assert ok
