"""The Metal launcher of the GPU xdem sampler (Apple silicon; pyobjc, design spec section 5.5).

    L = MetalLauncher()                       # XtimGpuError naming the cause when there is no Metal device, no
                                              # pyobjc (pip install "xtim[metal]") or the Mac is not arm64 (Intel
                                              # Macs' GPUs are untested and refused)
    L.compile(source, capacities)             # the kernel source compiled at run time, one pipeline per kernel
    L.upload(buffers); L.alloc("rows", np.uint64, n); L.run("walk", n); L.download("rows")
    L.run_batch([("walk", nb), ("resolve", n)])   # one command buffer, one wait

The kernel source (xtim/gpu/xdem_kernels.h) has no entry points: ``compile`` puts the capacity macros ahead of it and
the entry stubs below after it. Each stub is one ``kernel void`` taking every buffer of ``_stubs.BUFFERS`` (at its
index), filling an ``XkCtx`` with ``xk_ctx_init`` and calling the kernel's function with its thread id; the kernels
and their buffer lists are ``_stubs.KERNELS`` (shared with xtim/gpu/cuda.py). ``run`` requires the
kernel's own buffers to be uploaded and binds a 16-byte placeholder for any other buffer not uploaded (the kernel
never reads it), so "walk" and "resolve" never need emit's or reduce's buffers. Buffers live in shared storage
(unified memory): upload and download are copies into and out of the Metal buffer's contents; an upload of the same
dtype and size overwrites the buffer in place, and ``alloc`` keeps a buffer that is already large enough. Every
upload, alloc and download happens between submissions (``run`` / ``run_batch`` wait for completion), so the host
never writes a buffer a kernel is reading.

Errors: the launcher raises ``RuntimeError`` (compile, pipeline, a failed command buffer) and ``MemoryError`` (an
allocation); ``GpuBackend`` turns every one into ``XtimGpuError``. Every capacity but XK_CAP_WA and XK_CAP_RHO (both
<= 64) sizes no thread-local array, so no file the backend takes exceeds the pipelines' stack.

Importing this module does not import pyobjc; constructing a ``MetalLauncher`` does.
"""

from __future__ import annotations

import numpy as np

from xtim.gpu import _stubs

_INSTALL_HINT = 'pip install "xtim[metal]"'

# every buffer of the stubs (xtim/gpu/_stubs.py), in binding order: (name, MSL parameter type)
_BUFFERS = tuple((name, f"device {'' if written else 'const '}{ty}*") for name, ty, written in _stubs.BUFFERS)
assert len(_BUFFERS) <= 31  # Metal's buffer argument table

_KERNELS = _stubs.KERNELS
_entry = _stubs.entry

_PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"
_THREADGROUP = 64  # threads per threadgroup (at most the pipeline's maximum)


def _stub_text() -> str:
    params = ",\n".join(f"    {ty} {name} [[buffer({i})]]" for i, (name, ty) in enumerate(_BUFFERS))
    args = ", ".join(name for name, _ in _BUFFERS)
    out = []
    for kernel, (fn, tid, _) in _KERNELS.items():
        if kernel in _stubs.GROUP_KERNELS:   # a threadgroup per group, its memory declared here
            out.append(f"kernel void {_entry(kernel)}(\n{params},\n    uint {tid} [[threadgroup_position_in_grid]],\n"
                       f"    uint lane [[thread_position_in_threadgroup]],\n"
                       f"    uint lanes [[threads_per_threadgroup]]) {{\n"
                       f"    threadgroup u32 xk_shm[XK_CAP_SHM];\n"
                       f"    XkCtx c;\n    xk_ctx_init(&c, {args});\n    {fn}(&c, {tid}, lane, lanes, xk_shm);\n}}\n")
            continue
        out.append(f"kernel void {_entry(kernel)}(\n{params},\n    uint {tid} [[thread_position_in_grid]]) {{\n"
                   f"    XkCtx c;\n    xk_ctx_init(&c, {args});\n    {fn}(&c, {tid});\n}}\n")
    return "\n".join(out)


def _unavailable(cause: str) -> Exception:
    """the error raised when Metal cannot be used (the one place its type is chosen)"""
    from xtim.errors import XtimGpuError
    return XtimGpuError(f"Metal GPU unavailable: {cause} (on Apple silicon: {_INSTALL_HINT})")


def _describe(err) -> str:
    if err is None:
        return "unknown error"
    try:
        return str(err.localizedDescription())
    except AttributeError:
        return str(err)


class _Buffer:
    """a shared-storage Metal buffer holding `count` elements of `dtype` (contents unspecified until written)"""

    def __init__(self, device, mtl, dtype, count: int):
        self.dtype, self.count = np.dtype(dtype), int(count)
        nbytes = max(self.count * self.dtype.itemsize, 16)  # Metal refuses zero-length buffers
        self.handle = device.newBufferWithLength_options_(nbytes, mtl.MTLResourceStorageModeShared)
        if self.handle is None:
            raise MemoryError(f"Metal could not allocate a buffer of {nbytes} bytes")
        raw = self.handle.contents().as_buffer(nbytes)
        # the buffer's memory; never handed out (download copies), and dropped with the buffer
        self._view = np.frombuffer(raw, dtype=np.uint8, count=self.count * self.dtype.itemsize).view(self.dtype)

    def view(self) -> np.ndarray:
        return self._view


class MetalLauncher:
    name: str = "metal"

    def __init__(self) -> None:
        import platform
        if platform.machine() != "arm64":
            raise _unavailable(f"this machine is {platform.machine() or 'unknown'}, not an arm64 (Apple silicon) Mac; "
                               "the Metal kernels are validated on Apple silicon only")
        try:
            import Metal
        except ImportError as e:
            raise _unavailable(f"pyobjc-framework-Metal is not installed ({e})") from None
        self._mtl = Metal
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise _unavailable("no Metal device")
        self._queue = self._device.newCommandQueue()
        if self._queue is None:
            raise _unavailable("the Metal device gave no command queue")
        self._buf: dict[str, _Buffer] = {}
        self._pso: dict[str, object] = {}
        self._placeholder = _Buffer(self._device, self._mtl, np.uint8, 0)  # bound for unread buffers

    def compile(self, source: str, capacities: dict[str, int]) -> None:
        """Compile the kernel source with these capacity macros (the actual sizes); one compute pipeline per
        kernel."""
        text = _PRELUDE + _stubs.capacity_defines(capacities, "MetalLauncher.compile") + source + "\n" + _stub_text()
        opts = self._mtl.MTLCompileOptions.new()
        lib, err = self._device.newLibraryWithSource_options_error_(text, opts, None)
        if lib is None:
            raise RuntimeError(f"Metal compile of the xdem kernels failed:\n{_describe(err)}")
        pso = {}
        for kernel in _KERNELS:
            fn = lib.newFunctionWithName_(_entry(kernel))
            if fn is None:
                raise RuntimeError(f"Metal library has no function {_entry(kernel)}")
            p, err = self._device.newComputePipelineStateWithFunction_error_(fn, None)
            if p is None:
                raise RuntimeError(f"Metal pipeline for {kernel!r} failed:\n{_describe(err)}")
            pso[kernel] = p
        self._pso = pso

    def upload(self, buffers: dict[str, np.ndarray]) -> None:
        """Create or replace named buffers; one of the same dtype and size is overwritten in place."""
        for k, v in buffers.items():
            a = np.asarray(v)
            cur = self._buf.get(k)
            if cur is None or cur.dtype != a.dtype or cur.count != a.size:
                cur = self._buf[k] = _Buffer(self._device, self._mtl, a.dtype, a.size)
            cur.view()[:] = a.reshape(-1)

    def alloc(self, name: str, dtype, count: int) -> None:
        """A buffer of at least `count` elements of `dtype`, contents unspecified; an existing one of that dtype and
        at least that size is kept as it is."""
        cur = self._buf.get(name)
        if cur is None or cur.dtype != np.dtype(dtype) or cur.count < int(count):
            self._buf[name] = _Buffer(self._device, self._mtl, dtype, count)

    def run(self, kernel: str, n: int, lanes: int | None = None) -> None:
        """Run `kernel` over n threads (thread id = block for "walk" and "reduce", shot for "resolve" and "emit",
        hist_total slot for "reduce2"); blocks until done. The kernel's own buffers must be uploaded; any other
        buffer not uploaded is bound to a placeholder."""
        self.run_batch([(kernel, n) if lanes is None else (kernel, n, lanes)])

    def run_batch(self, steps: list[tuple[str, int]]) -> None:
        """Run the steps (kernel, threads) in order as one command buffer (one serial compute encoder: each dispatch
        sees the previous ones' writes) and wait once at the end. Checked before anything is encoded."""
        if not self._pso:
            raise RuntimeError("MetalLauncher.run before compile")
        todo = _stubs.check_steps(steps, "MetalLauncher.run", self._buf)
        if not todo:
            return
        mtl = self._mtl
        cb = self._queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        for i, (name, _) in enumerate(_BUFFERS):
            enc.setBuffer_offset_atIndex_(self._buf.get(name, self._placeholder).handle, 0, i)
        for kernel, n, lanes in todo:
            pso = self._pso[kernel]
            enc.setComputePipelineState_(pso)
            if lanes:   # a group kernel: n threadgroups of `lanes` threads (at most the pipeline's maximum)
                tg = max(1, min(lanes, int(pso.maxTotalThreadsPerThreadgroup())))
                enc.dispatchThreadgroups_threadsPerThreadgroup_(mtl.MTLSizeMake(n, 1, 1), mtl.MTLSizeMake(tg, 1, 1))
                continue
            tg = max(1, min(_THREADGROUP, int(pso.maxTotalThreadsPerThreadgroup())))
            enc.dispatchThreads_threadsPerThreadgroup_(mtl.MTLSizeMake(n, 1, 1), mtl.MTLSizeMake(tg, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.error() is not None or cb.status() != mtl.MTLCommandBufferStatusCompleted:
            names = ", ".join(repr(k) for k, *_ in todo)
            raise RuntimeError(f"Metal kernel {names} failed: {_describe(cb.error())}")

    def download(self, name: str, count: int | None = None) -> np.ndarray:
        a = self._buf[name].view()
        return (a if count is None else a[:count]).copy()

    def view(self, name: str, count: int) -> np.ndarray:
        """The buffer's first `count` elements, read-only and not copied: the caller copies what it keeps before the
        next upload, alloc or run (which may overwrite or free the memory)."""
        buf = self._buf[name]
        if not 0 <= int(count) <= buf.count:
            raise ValueError(f"MetalLauncher.view: {name} has {buf.count} elements, asked for {count}")
        v = buf.view()[:int(count)]
        v.flags.writeable = False
        return v

    def free_bytes(self) -> int:
        """The device's recommended working set less what is allocated now."""
        d = self._device
        return max(0, int(d.recommendedMaxWorkingSetSize()) - int(d.currentAllocatedSize()))

    def describe(self) -> dict:
        """The device and software versions (tools/gpu_validate.py's report)."""
        import platform
        import objc
        return {"name": str(self._device.name()), "total_bytes": int(self._device.recommendedMaxWorkingSetSize()),
                "macos": platform.mac_ver()[0], "pyobjc": str(objc.__version__)}
