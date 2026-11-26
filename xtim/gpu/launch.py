"""The GPU sampler's launcher interface (design spec section 5.5): one object per device API that compiles the kernel
source xtim/gpu/xdem_kernels.h, holds named buffers and runs the kernels over a 1-D grid.

    L.compile(source, capacities)   # capacities: {"XK_CAP_W": ..., ...} (xtim.gpu.tables.capacities)
    L.upload(buffers)               # create or replace named buffers (xtim.gpu.tables.pack); a buffer of the same
                                    # dtype and size is overwritten in place, not reallocated
    L.alloc("rows", np.uint64, n)   # a device buffer of at least n elements, contents unspecified, no host copy
                                    # (kept as it is when it already has that dtype and at least n elements)
    L.run("walk", nblocks)          # blocks until the kernel is done; thread ids past the batch are ignored
    L.run_batch([("walk", nb), ("resolve", n)])   # the kernels in order as one submission, one wait at the end
    L.download("rows")              # a copy, in the dtype the buffer was uploaded with (count: its first elements)
    L.view("rows", n)               # its first n elements read-only, no copy where the memory is host-visible
                                    # (Metal, ref; CUDA returns a host copy): valid only until the next upload,
                                    # alloc or run (a caller copies what it keeps before then)

Buffers persist across runs: a backend uploads its read-only tables once, allocates its working buffers once at the
largest batch and reuses them (xdem_kernels.h's header comment audits that no kernel reads what the current batch did
not write). ``run`` is ``run_batch`` with one step; an error in any step raises when the submission completes.

Implementations: ``xtim.gpu.metal.MetalLauncher`` ("metal"), ``xtim.gpu.cuda.CudaLauncher`` ("cuda") and the C++
reference build of the tests (tests/_gpu_ref.py, "ref"). The device launchers generate their entry stubs from one
table of kernels and buffers (xtim/gpu/_stubs.py).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Launcher(Protocol):
    name: str  # "metal" | "cuda" | "ref"

    def compile(self, source: str, capacities: dict[str, int]) -> None: ...

    def upload(self, buffers: dict[str, np.ndarray]) -> None: ...  # create, or overwrite in place when same size

    def alloc(self, name: str, dtype, count: int) -> None: ...  # at least count elements, unspecified contents

    # kernel in _stubs.KERNELS ("walk", "resolve", ..., "emit"); blocks until done
    def run(self, kernel: str, n: int, lanes: int | None = None) -> None: ...

    def run_batch(self, steps: list[tuple[str, int]]) -> None: ...  # in order, one submission, one wait

    def download(self, name: str, count: int | None = None) -> np.ndarray: ...  # a copy

    def view(self, name: str, count: int) -> np.ndarray: ...  # read-only (CUDA: a copy); dead at next upload/alloc/run

    def free_bytes(self) -> int: ...
