"""The kernels' entry points, shared by every launcher that compiles xtim/gpu/xdem_kernels.h at run time
(xtim/gpu/metal.py, xtim/gpu/cuda.py): which buffers there are, in binding order, and which kernel reads which.

The kernel source has no entry points: each launcher appends one stub per ``KERNELS`` entry, in its own target's
syntax, taking every buffer of ``BUFFERS`` in this order, filling an ``XkCtx`` with ``xk_ctx_init`` (which takes the
buffers in this same order) and calling the kernel's function with its thread id. Adding a kernel is one ``KERNELS``
entry (with the buffers it reads); adding a buffer is one ``BUFFERS`` entry and one ``xk_ctx_init`` parameter. The
C++ reference host (cpp/testapi/xdem_gpu_ref_py.cpp ``needs()``) holds the same per-kernel lists.

Importing this module imports nothing beyond numpy.
"""
from __future__ import annotations

import numpy as np

# every buffer of the stubs, in binding order: (name, element type, written by some kernel)
BUFFERS = (
    ("dims", "u32", False),
    ("seed", "u64", False),
    ("tab64", "u64", False),
    ("tab32", "u32", False),
    ("rows", "u64", True),
    ("epos", "u32", True),
    ("eids", "u32", True),
    ("psh", "u32", True),
    ("overflow", "u32", True),
    ("ckey", "u64", True),
    ("cdim", "u8", True),
    ("coin_in", "u64", False),
    ("flag", "u8", True),
    ("hist", "u32", True),
    ("hist_group", "u64", True),
    ("hist_total", "u64", True),
    ("mws", "u64", True),
    ("lawq", "u32", True),
    ("boff", "u32", True),
    ("ctab", "u64", False),
    ("out_det", "u8", True),
    ("out_obs", "u8", True),
    ("out_cs", "u32", True),
    ("cmiss", "u32", True),
)

# the numpy dtype of each element type (a launcher's buffers have these dtypes; tables.pack produces them)
DTYPES = {"u8": np.dtype(np.uint8), "u32": np.dtype(np.uint32), "u64": np.dtype(np.uint64)}

COMMON = ("dims", "seed", "tab64", "tab32", "rows")  # xk_ctx_init reads dims and seed; every kernel the rows

# launcher kernel name -> (the source's function, what the thread id is, the buffers it reads: xdem_kernels.h's header
# comment; resolve touches mws only for a multi-word file, but a launcher binds it always; walk, resolve and
# resolve_law touch lawq only under XK_MODE_DEFER, walk cmiss only under XK_MODE_EMIT and flag only under
# XK_MODE_POSTSELECT, emit boff only under XK_MODE_POSTSELECT). Thread ids: "block" for reduce, "shot" for resolve
# and emit, "t" for resolve_law (law_threads(nshots) threads sharing K2's queue), "t" = group * (hist stride + 2) +
# slot for reduce_groups, "slot" (of hist_total) for reduce2, "block" for scan_blocks; every kernel
# ignores ids past its range. The walk is a group kernel (GROUP_KERNELS): n groups (blocks of shots) of any number of
# threads, which share XK_CAP_SHM u32 words of group memory; its function takes (ctx, group, lane, lanes, memory).
KERNELS = {
    "walk": ("xk_walk_group", "block", COMMON + ("epos", "eids", "psh", "overflow", "lawq", "cmiss", "flag")),
    "resolve": ("xk_resolve_shot", "shot", COMMON + ("epos", "eids", "ckey", "cdim", "coin_in", "flag", "mws",
                                                     "lawq")),
    "resolve_law": ("xk_resolve_law", "t", COMMON + ("epos", "eids", "ckey", "cdim", "coin_in", "flag", "mws",
                                                     "lawq")),
    "reduce": ("xk_reduce_block", "block", COMMON + ("cdim", "flag", "hist")),
    "reduce_groups": ("xk_reduce_groups", "t", COMMON + ("overflow", "hist", "hist_group")),
    "reduce2": ("xk_reduce_total", "slot", COMMON + ("hist_group", "hist_total")),
    "scan_blocks": ("xk_scan_blocks", "block", COMMON + ("flag", "hist", "hist_group", "boff")),
    "emit": ("xk_emit", "shot", COMMON + ("ckey", "cdim", "flag", "boff", "ctab", "out_det", "out_obs", "out_cs",
                                          "cmiss")),
}


GROUP_KERNELS = frozenset({"walk"})
# threads per group of a group kernel when the step does not say (a launcher clamps it to its pipeline's maximum; the
# outputs do not depend on it)
GROUP_LANES = 256
MAX_GROUP_LANES = 1024


def entry(kernel: str) -> str:
    """the stub's (entry function's) name of a kernel"""
    return f"xk_entry_{kernel}"


def capacity_defines(capacities: dict[str, int], who: str) -> str:
    """the ``#define XK_CAP_* n`` lines of a compile (ValueError naming ``who`` for a bad one)"""
    defs = []
    for k, v in capacities.items():
        if not k.startswith("XK_CAP_") or int(v) < 1:
            raise ValueError(f"{who}: bad capacity {k}={v}")
        defs.append(f"#define {k} {int(v)}\n")
    return "".join(defs)


def check_steps(steps, who: str, present) -> list[tuple[str, int, int]]:
    """the steps of a run_batch with at least one thread, checked before anything runs, as (kernel, n, lanes): a step
    is (kernel, threads) or, for a group kernel, (kernel, groups) or (kernel, groups, lanes) (lanes: threads per group,
    1 to MAX_GROUP_LANES; default GROUP_LANES; 0 for the other kernels); a known kernel, 0 <= n < 2^32, every buffer it
    reads in ``present`` (ValueError naming ``who``)"""
    todo = []
    for step in steps:
        kernel, n, *rest = step
        lanes = 0
        if kernel in GROUP_KERNELS:
            lanes = int(rest[0]) if rest else GROUP_LANES
            if not 1 <= lanes <= MAX_GROUP_LANES:
                raise ValueError(f"{who}: {kernel!r} lanes = {lanes} out of range")
        elif rest:
            raise ValueError(f"{who}: {kernel!r} takes no lanes")
        if kernel not in KERNELS:
            raise ValueError(f"{who}: unknown kernel {kernel!r} (one of {sorted(KERNELS)})")
        n = int(n)
        if n < 0 or n > 0xFFFF_FFFF:
            raise ValueError(f"{who}: n = {n} out of range")
        missing = [name for name in KERNELS[kernel][2] if name not in present]
        if missing:
            raise ValueError(f"{who}: buffers not uploaded: {', '.join(missing)}")
        if n:
            todo.append((kernel, n, lanes))
    return todo
