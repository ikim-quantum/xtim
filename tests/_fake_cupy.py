"""A stand-in for CuPy (test helper) so that `xtim.gpu.cuda.CudaLauncher`'s host-side logic runs on a machine without
CUDA. It is NOT a CUDA test: nothing here compiles or runs CUDA code, so passing tests against it prove the launcher's
bookkeeping (buffer binding order and dtypes, grid sizing, alloc / upload reuse, run_batch, download / view, error
paths), never that the CUDA build works. That needs `pytest -m cuda` and tools/gpu_validate.py on an NVIDIA GPU.

    fake = make_fake_cupy()                      # a module object; install it with
    monkeypatch.setitem(sys.modules, "cupy", fake)

What it models, and how:
  - ``ndarray``: a device array is a wrapper around host memory that refuses implicit conversion to numpy (as CuPy
    does), so a launcher that treats device arrays as numpy arrays fails here too. ``set`` / ``get`` / 1-D slicing
    follow CuPy (``set`` needs the same shape and dtype). New arrays from ``empty`` hold the byte 0xA5.
  - ``RawModule(code=..., options=...)``: "compiles" by running the host C++ syntax check below on the exact text (when
    a ``c++`` is on PATH; a compile error raises ``cupy.cuda.compiler.CompileException`` with the compiler's log) and
    reading the capacity macros back from the text; ``get_function(name)`` needs an ``extern "C" __global__`` entry of
    that name in the text.
  - ``RawKernel.__call__(grid, block, args)``: checks the call convention (int tuples, block within
    ``max_threads_per_block``, one argument per stub buffer, device arrays of the stub's element type for every
    buffer the kernel reads) and runs the kernel's C++ reference build (`_xtim_testapi.xdem_gpu_ref`) over
    grid[0] * block[0] thread ids, as the CUDA stub would (ids past 2^32 - 1 exit).
  - ``cuda.runtime``: one device, ``memGetInfo``, ``deviceSynchronize`` (counted), device properties and versions.

`cxx_syntax_check(text)` compiles a CUDA program text as host C++ (`c++ -std=c++14 -fsyntax-only`) under `SHIM`, a
header defining the CUDA-only tokens the text uses; it catches typos, type errors and C++17-isms in the text the
launcher hands NVRTC, not anything CUDA-specific.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import types

import numpy as np

from xtim.gpu import _stubs

# the CUDA-only tokens of the CUDA program text (xtim/gpu/cuda.py's stubs, the CUDA branch of xdem_kernels.h's prelude)
SHIM = r"""// host C++ stand-ins for CUDA-only tokens (tests/_fake_cupy.py): a syntax check, not a CUDA compiler
#define __CUDACC__ 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
struct xk_shim_uint3 { unsigned int x, y, z; };
extern const xk_shim_uint3 threadIdx, blockIdx, blockDim, gridDim;
inline int __ffsll(long long x) { return __builtin_ffsll(x); }
inline int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline unsigned int atomicAdd(unsigned int* p, unsigned int v) { const unsigned int o = *p; *p += v; return o; }
inline unsigned int atomicXor(unsigned int* p, unsigned int v) { const unsigned int o = *p; *p ^= v; return o; }
inline unsigned long long atomicXor(unsigned long long* p, unsigned long long v) {
    const unsigned long long o = *p; *p ^= v; return o;
}
inline void __syncthreads() {}
#define __shared__
"""

CXX_FLAGS = ("-std=c++14", "-fsyntax-only", "-Wall", "-pedantic-errors", "-Werror")
_checked: dict[str, tuple[bool, str]] = {}


def cxx() -> str | None:
    return shutil.which("c++")


def cxx_syntax_check(text: str) -> tuple[bool, str]:
    """(ok, compiler log) of `text` compiled as host C++14 under SHIM (cached by text)"""
    key = hashlib.sha256(text.encode()).hexdigest()
    if key not in _checked:
        p = subprocess.run([cxx(), *CXX_FLAGS, "-x", "c++", "-"], input=SHIM + text, capture_output=True, text=True)
        _checked[key] = (p.returncode == 0, p.stderr)
    return _checked[key]


class CompileException(Exception):
    pass


class CUDARuntimeError(RuntimeError):
    pass


class CUDADriverError(RuntimeError):
    pass


class ndarray:
    """a 1-D device array (host memory underneath; refuses implicit conversion to numpy, as CuPy does)"""

    def __init__(self, a: np.ndarray):
        self._a = a

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def size(self):
        return self._a.size

    @property
    def shape(self):
        return self._a.shape

    @property
    def nbytes(self):
        return self._a.nbytes

    def __len__(self):
        return len(self._a)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("fake cupy: only slicing is modelled")
        return ndarray(self._a[key])

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a "
                        "NumPy array explicitly.")

    def set(self, arr, stream=None):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Only numpy.ndarray can be set to cupy.ndarray, got {type(arr)}")
        if arr.shape != self._a.shape:
            raise ValueError(f"Shape mismatch. Old shape: {self._a.shape}, new shape: {arr.shape}")
        if arr.dtype != self._a.dtype:
            raise TypeError(f"{arr.dtype} cannot be cast to {self._a.dtype}")  # stricter than CuPy, on purpose
        if not self._a.flags.c_contiguous:
            raise RuntimeError("Cannot set to non-contiguous array")
        np.copyto(self._a, arr)

    def get(self, stream=None) -> np.ndarray:
        return self._a.copy()


def make_fake_cupy(*, devices: int = 1, free: int = 6 << 30, total: int = 8 << 30, check_cxx: bool = True):
    """a fresh fake ``cupy`` module (its own launch / sync counters in ``fake.log``)"""
    from _gpu_ref import RefLauncher
    from xtim import _xtim_testapi as _T

    cp = types.ModuleType("cupy")
    cp.__version__ = "0.0.0+fake"
    cp.ndarray = ndarray
    cp.log = {"launches": [], "syncs": 0, "modules": 0, "options": []}
    cp.fail_next_sync = None   # set to an exception to raise at the next deviceSynchronize (a failed kernel)

    def empty(shape, dtype=float):
        a = np.empty(shape, dtype)
        a.view(np.uint8)[...] = 0xA5
        return ndarray(a)

    def zeros(shape, dtype=float):
        return ndarray(np.zeros(shape, dtype))

    cp.empty, cp.zeros = empty, zeros

    class RawKernel:
        max_threads_per_block = 1024

        def __init__(self, kernel, variant):
            self.kernel, self.variant = kernel, variant

        def __call__(self, grid, block, args, *, shared_mem=0):
            for what, v in (("grid", grid), ("block", block)):
                if not isinstance(v, tuple) or not 1 <= len(v) <= 3 or not all(isinstance(x, int) and x >= 1
                                                                                for x in v):
                    raise ValueError(f"fake cupy: {what} must be a tuple of 1 to 3 positive ints, got {v!r}")
            if block[0] > self.max_threads_per_block:
                raise CUDADriverError("CUDA_ERROR_INVALID_VALUE: block larger than max_threads_per_block")
            if not isinstance(args, tuple) or len(args) != len(_stubs.BUFFERS):
                raise ValueError(f"fake cupy: {len(_stubs.BUFFERS)} arguments expected, got {args!r:.200}")
            reads = _stubs.KERNELS[self.kernel][2]
            bufs = {}
            for (name, ty, _), a in zip(_stubs.BUFFERS, args):
                if not isinstance(a, ndarray):
                    raise TypeError(f"fake cupy: argument {name} is not a device array ({type(a)})")
                if name in reads and a.dtype != _stubs.DTYPES[ty]:
                    raise TypeError(f"fake cupy: argument {name} is {a.dtype}, the stub takes {ty}*")
                bufs[name] = a._a
            n = min(grid[0] * block[0], 0xFFFF_FFFF)   # the stub returns for thread ids past u32
            cp.log["launches"].append((self.kernel, grid, block))
            _T.xdem_gpu_ref(self.kernel, n, {**bufs, "_caps": np.array([self.variant], np.uint32)})

    class RawModule:
        def __init__(self, code=None, *, path=None, options=(), backend="nvrtc", **kw):
            if code is None or path is not None or backend != "nvrtc":
                raise ValueError("fake cupy: RawModule(code=..., options=...) only")
            self.code, self.options = code, tuple(options)
            cp.log["modules"] += 1
            cp.log["options"].append(self.options)
            self._variant = None

        def _compile(self):
            if self._variant is not None:
                return
            if check_cxx and cxx() is not None:
                ok, log = cxx_syntax_check(self.code)
                if not ok:
                    raise CompileException(log)
            caps = {k: int(v) for k, v in re.findall(r"^#define (XK_CAP_\w+) (\d+)$", self.code, re.M)}
            R = RefLauncher()
            R.compile("", caps)
            self._variant = R._variant

        def get_function(self, name):
            self._compile()
            for kernel in _stubs.KERNELS:
                if _stubs.entry(kernel) == name and f'extern "C" __global__ void {name}(' in self.code:
                    return RawKernel(kernel, self._variant)
            raise CUDADriverError(f"CUDA_ERROR_NOT_FOUND: named symbol not found ({name})")

    cp.RawModule, cp.RawKernel = RawModule, RawKernel

    class _Pool:
        def free_bytes(self):
            return 1 << 20

        def used_bytes(self):
            return 0

    cp.get_default_memory_pool = lambda: _Pool()

    def device_count():
        if devices < 1:
            raise CUDARuntimeError("cudaErrorNoDevice: no CUDA-capable device is detected")
        return devices

    def device_synchronize():
        cp.log["syncs"] += 1
        err, cp.fail_next_sync = cp.fail_next_sync, None
        if err is not None:
            raise err

    runtime = types.SimpleNamespace(
        CUDARuntimeError=CUDARuntimeError,
        getDeviceCount=device_count,
        memGetInfo=lambda: (free, total),
        deviceSynchronize=device_synchronize,
        getDeviceProperties=lambda i: {"name": b"Fake CuPy device", "major": 0, "minor": 0, "totalGlobalMem": total},
        driverGetVersion=lambda: 0,
        runtimeGetVersion=lambda: 0,
    )
    cp.cuda = types.SimpleNamespace(runtime=runtime, driver=types.SimpleNamespace(CUDADriverError=CUDADriverError),
                                    compiler=types.SimpleNamespace(CompileException=CompileException))
    return cp
