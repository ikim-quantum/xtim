"""The CUDA launcher of the GPU xdem sampler (NVIDIA GPUs; CuPy, design spec section 5.5).

    L = CudaLauncher()                        # XtimGpuError naming the cause when CuPy is missing or there is no
                                              # usable CUDA device (pip install the CuPy wheel of the machine's CUDA,
                                              # e.g. cupy-cuda12x)
    L.compile(source, capacities)             # the kernel source compiled at run time by NVRTC, one function per kernel
    L.upload(buffers); L.alloc("rows", np.uint64, n); L.run("walk", n); L.download("rows")
    L.run_batch([("walk", nb), ("resolve", n)])   # the kernels in order on one stream, one synchronize

The kernel source (xtim/gpu/xdem_kernels.h) has no entry points: ``compile`` puts the capacity macros ahead of it and
the entry stubs below after it (``program``: the exact text handed to ``cupy.RawModule``, compiled with
``-std=c++14``). Each stub is one ``extern "C" __global__`` function taking every buffer of ``_stubs.BUFFERS`` as a
pointer, in order, filling an ``XkCtx`` with ``xk_ctx_init`` and calling the kernel's function with its thread id
``blockIdx.x * blockDim.x + threadIdx.x``; the kernels and their buffer lists are ``_stubs.KERNELS`` (shared with
xtim/gpu/metal.py). A launch is ``ceil(n / block)`` blocks of ``block = min(256, the function's
max_threads_per_block)`` threads: the grid is rounded up and every kernel ignores thread ids past its range (an id
past 2^32 - 1, possible only in the last block, returns in the stub). ``run`` requires the kernel's own buffers to be
uploaded and passes a 16-byte placeholder for any other buffer not uploaded (the kernel never reads it).

Buffers live in device memory (CuPy arrays from its memory pool): upload copies host to device (``ndarray.set``), an
upload of the same dtype and size overwrites the buffer in place, ``alloc`` keeps a buffer that is already large
enough. Device memory is not host-visible, so ``download`` and ``view`` both copy device to host: ``view`` returns
that copy read-only (the protocol's "valid until the next upload, alloc or run" holds trivially). Kernels run on
CuPy's current stream, in order; ``run_batch`` launches its steps back to back and synchronizes the device once at
the end, where an error in any step raises; upload / alloc / download happen between batches, so the host never
writes a buffer a kernel is reading.

Importing this module does not import CuPy; constructing a ``CudaLauncher`` does.
"""

from __future__ import annotations

import numpy as np

from xtim.gpu import _stubs

_INSTALL_HINT = ("pip install the CuPy wheel matching the machine's CUDA version (nvidia-smi's \"CUDA Version\"), "
                 "e.g. pip install cupy-cuda12x")

# every buffer of the stubs (xtim/gpu/_stubs.py), in binding order: (name, CUDA parameter type)
_BUFFERS = tuple((name, f"{'' if written else 'const '}{ty}*") for name, ty, written in _stubs.BUFFERS)

_KERNELS = _stubs.KERNELS
_entry = _stubs.entry

_PRELUDE = "// xtim/gpu/xdem_kernels.h for CUDA (NVRTC through cupy.RawModule; xtim/gpu/cuda.py)\n"
_OPTIONS = ("-std=c++14",)
_BLOCK = 256  # threads per block (at most the function's maximum)


def _stub_text() -> str:
    params = ",\n".join(f"    {ty} {name}" for name, ty in _BUFFERS)
    args = ", ".join(name for name, _ in _BUFFERS)
    out = []
    for kernel, (fn, tid, _) in _KERNELS.items():
        if kernel in _stubs.GROUP_KERNELS:   # a CUDA block per group, its shared memory declared here
            out.append(f'extern "C" __global__ void {_entry(kernel)}(\n{params}) {{\n'
                       f"    __shared__ u32 xk_shm[XK_CAP_SHM];\n"
                       f"    XkCtx c;\n    xk_ctx_init(&c, {args});\n"
                       f"    {fn}(&c, (u32)blockIdx.x, (u32)threadIdx.x, (u32)blockDim.x, xk_shm);\n}}\n")
            continue
        out.append(f'extern "C" __global__ void {_entry(kernel)}(\n{params}) {{\n'
                   f"    const u64 xk_id = (u64)blockIdx.x * blockDim.x + threadIdx.x;\n"
                   f"    if (xk_id > XK_U64(0xFFFFFFFF)) return;\n"
                   f"    const u32 {tid} = (u32)xk_id;\n"
                   f"    XkCtx c;\n    xk_ctx_init(&c, {args});\n    {fn}(&c, {tid});\n}}\n")
    return "\n".join(out)


def program(source: str, capacities: dict[str, int]) -> str:
    """The exact CUDA C++ text ``compile`` hands NVRTC: prelude, capacity macros, kernel source, entry stubs."""
    return _PRELUDE + _stubs.capacity_defines(capacities, "CudaLauncher.compile") + source + "\n" + _stub_text()


def _unavailable(cause: str) -> Exception:
    """the error raised when CUDA cannot be used (the one place its type is chosen)"""
    from xtim.errors import XtimGpuError
    return XtimGpuError(f"CUDA GPU unavailable: {cause} (on an NVIDIA GPU: {_INSTALL_HINT})")


def _describe(err) -> str:
    return f"{type(err).__name__}: {err}"


class _Buffer:
    """a device buffer holding `count` elements of `dtype` (contents unspecified until written)"""

    def __init__(self, cp, dtype, count: int):
        self.dtype, self.count = np.dtype(dtype), int(count)
        # never zero-length (a kernel argument is always a valid pointer)
        self.handle = cp.empty(max(self.count, 1), dtype=self.dtype)

    def view(self):
        """the first `count` elements, on the device"""
        return self.handle if self.handle.size == self.count else self.handle[:self.count]


class CudaLauncher:
    name: str = "cuda"

    def __init__(self) -> None:
        try:
            import cupy
        except ImportError as e:
            raise _unavailable(f"CuPy is not installed ({e})") from None
        except Exception as e:   # CuPy installed but failing to load (a broken CUDA library, a mismatched wheel)
            raise _unavailable(f"CuPy failed to import ({_describe(e)})") from None
        self._cp = cupy
        try:
            ndev = int(cupy.cuda.runtime.getDeviceCount())
        except Exception as e:  # CUDARuntimeError (no driver, no device), or CuPy built for another CUDA
            raise _unavailable(f"no usable CUDA device ({_describe(e)})") from None
        if ndev < 1:
            raise _unavailable("no CUDA device")
        try:
            self._placeholder = cupy.zeros(16, dtype=np.uint8)  # passed for unread buffers
        except Exception as e:
            raise _unavailable(f"the CUDA device cannot allocate ({_describe(e)})") from None
        self._buf: dict[str, _Buffer] = {}
        self._fn: dict[str, object] = {}
        self._module = None

    def compile(self, source: str, capacities: dict[str, int]) -> None:
        """Compile the kernel source with these capacity macros (the actual sizes); one function per kernel. NVRTC
        compiles on the first get_function (CuPy caches the binary on disk)."""
        text = program(source, capacities)
        try:
            module = self._cp.RawModule(code=text, options=_OPTIONS)
            fn = {kernel: module.get_function(_entry(kernel)) for kernel in _KERNELS}
        except Exception as e:  # cupy.cuda.compiler.CompileException (with NVRTC's log), CUDADriverError
            raise RuntimeError(f"CUDA compile of the xdem kernels failed:\n{_describe(e)}") from e
        self._module, self._fn = module, fn

    def upload(self, buffers: dict[str, np.ndarray]) -> None:
        """Create or replace named buffers; one of the same dtype and size is overwritten in place."""
        for k, v in buffers.items():
            a = np.ascontiguousarray(v).reshape(-1)
            cur = self._buf.get(k)
            if cur is None or cur.dtype != a.dtype or cur.count != a.size:
                cur = self._buf[k] = _Buffer(self._cp, a.dtype, a.size)
            if a.size:
                cur.view().set(a)

    def alloc(self, name: str, dtype, count: int) -> None:
        """A buffer of at least `count` elements of `dtype`, contents unspecified; an existing one of that dtype and
        at least that size is kept as it is."""
        cur = self._buf.get(name)
        if cur is None or cur.dtype != np.dtype(dtype) or cur.count < int(count):
            self._buf[name] = _Buffer(self._cp, dtype, count)

    def run(self, kernel: str, n: int, lanes: int | None = None) -> None:
        """Run `kernel` over n threads (thread id = block for "walk" and "reduce", shot for "resolve" and "emit",
        hist_total slot for "reduce2"); blocks until done. The kernel's own buffers must be uploaded; any other
        buffer not uploaded is passed as a placeholder."""
        self.run_batch([(kernel, n) if lanes is None else (kernel, n, lanes)])

    def run_batch(self, steps: list[tuple[str, int]]) -> None:
        """Launch the steps (kernel, threads) in order on the current stream (each launch sees the previous ones'
        writes) and synchronize once at the end. Checked before anything is launched."""
        if not self._fn:
            raise RuntimeError("CudaLauncher.run before compile")
        todo = _stubs.check_steps(steps, "CudaLauncher.run", self._buf)
        if not todo:
            return
        args = tuple(self._buf[name].handle if name in self._buf else self._placeholder for name, _ in _BUFFERS)
        for kernel, n, lanes in todo:
            try:
                fn = self._fn[kernel]
                if lanes:   # a group kernel: n blocks of `lanes` threads (at most the function's maximum)
                    fn((n,), (max(1, min(lanes, int(fn.max_threads_per_block))),), args)
                    continue
                block = max(1, min(_BLOCK, int(fn.max_threads_per_block)))
                fn(((n + block - 1) // block,), (block,), args)
            except Exception as e:  # the launch itself refused (CUDADriverError: resources, a bad configuration)
                raise RuntimeError(f"CUDA kernel {kernel!r} failed to launch ({n} threads): {_describe(e)}") from e
        try:
            self._cp.cuda.runtime.deviceSynchronize()
        except Exception as e:  # CUDARuntimeError: a fault in a kernel, reported at the synchronize
            names = ", ".join(repr(k) for k, *_ in todo)
            raise RuntimeError(f"CUDA kernel {names} failed: {_describe(e)} (raised at the synchronize after the "
                               f"launches {names}; after a fault the CUDA context is unusable)") from e

    def download(self, name: str, count: int | None = None) -> np.ndarray:
        buf = self._buf[name]
        n = buf.count if count is None else int(count)
        if not 0 <= n <= buf.count:
            raise ValueError(f"CudaLauncher.download: {name} has {buf.count} elements, asked for {count}")
        return buf.handle[:n].get() if n else np.empty(0, buf.dtype)

    def view(self, name: str, count: int) -> np.ndarray:
        """The buffer's first `count` elements, read-only. Device memory is not host-visible: this is a host copy
        (``download``), so it stays valid past the next upload, alloc or run (the protocol asks less)."""
        buf = self._buf[name]
        if not 0 <= int(count) <= buf.count:
            raise ValueError(f"CudaLauncher.view: {name} has {buf.count} elements, asked for {count}")
        v = self.download(name, int(count))
        v.flags.writeable = False
        return v

    def free_bytes(self) -> int:
        """The device's free memory plus what CuPy's memory pool holds free."""
        free, _ = self._cp.cuda.runtime.memGetInfo()
        return int(free) + int(self._cp.get_default_memory_pool().free_bytes())

    def describe(self) -> dict:
        """The device and software versions (tools/gpu_validate.py's report)."""
        rt = self._cp.cuda.runtime
        p = rt.getDeviceProperties(0)
        name = p.get("name", b"?")
        out = {"name": name.decode() if isinstance(name, bytes) else str(name),
               "compute_capability": f"{p.get('major', '?')}.{p.get('minor', '?')}",
               "total_bytes": int(p.get("totalGlobalMem", 0)), "driver": int(rt.driverGetVersion()),
               "runtime": int(rt.runtimeGetVersion()), "cupy": str(self._cp.__version__)}
        kernels = {}
        for kernel, fn in self._fn.items():   # after a compile: each function's per-thread resources, when exposed
            k = {}
            for attr in ("local_size_bytes", "num_regs", "max_threads_per_block"):
                try:
                    k[attr] = int(getattr(fn, attr))
                except Exception:
                    pass
            kernels[kernel] = k
        if kernels:
            out["kernels"] = kernels
        return out
