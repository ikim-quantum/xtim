"""The GPU xdem kernels run on the CPU (test helper): `RefLauncher`, `host_outputs` and two GF(2) helpers.

`RefLauncher` implements the GPU sampler's `Launcher` protocol (Task 5; `alloc` / `run_batch`, Task 9b) over
`_xtim_testapi.xdem_gpu_ref`, the kernel source `xtim/gpu/xdem_kernels.h` compiled as plain C++. The test extension
holds that source built three times with different capacity macros (the variants of `_xtim_testapi.xdem_gpu_ref_caps()`);
`compile` picks the variant whose capacities equal the requested ones, else the first (largest) one when they fit in
it; ``RefLauncher(variant=k)`` always runs variant k (2: XK_CAP_RHO = 32, xk_law_fast's register implementation;
variant 0 runs the other one). Variants 0 and 2 also hold the structured law (XK_CAP_ST_*, xtim.gpu.structure), variant 1
does not: a request fits a variant when every requested capacity is compiled in at least as large and every capacity the
variant has (but the XK_CAP_ST_*) is requested. `alloc` fills a new buffer with the byte `fill` (default 0xA5), so a kernel or a host read of an element the
current batch did not write shows up as a changed output.
"""
from __future__ import annotations

import numpy as np

from xtim import _xtim_testapi as _T


class RefLauncher:
    name: str = "ref"

    def __init__(self, fill: int = 0xA5, variant: int | None = None, lanes: int | None = None) -> None:
        self.lanes = 37 if lanes is None else int(lanes)   # the walk's threads per group (no output depends on it)
        self._buf: dict[str, np.ndarray] = {}
        self._variant: int | None = None
        self._force = variant
        self._fill = int(fill)

    def compile(self, source: str, capacities: dict[str, int]) -> None:
        """Select the reference build for these capacities (the source is compiled into the test extension)."""
        caps = {k: int(v) for k, v in capacities.items()}
        variants = _T.xdem_gpu_ref_caps()

        def fits(v):   # every requested capacity compiled in, at least as large (the XK_CAP_ST_* only when asked)
            return all(k in v and caps[k] <= v[k] for k in caps) and all(k in caps for k in v
                                                                           if not k.startswith("XK_CAP_ST_"))
        if self._force is not None:
            v = variants[self._force]
            if not fits(v):
                raise ValueError(f"RefLauncher: capacities {caps} exceed reference build {self._force}: {v}")
            self._variant = self._force
            return
        for i, v in enumerate(variants):
            if v == caps:
                self._variant = i
                return
        if not fits(variants[0]):
            raise ValueError(f"RefLauncher: capacities {caps} exceed every reference build {variants}")
        self._variant = 0

    def upload(self, buffers: dict[str, np.ndarray]) -> None:
        for k, v in buffers.items():
            v = np.asarray(v)
            cur = self._buf.get(k)
            if cur is not None and cur.dtype == v.dtype and cur.size == v.size:
                np.copyto(cur, v.reshape(-1))   # in place, as a device buffer is overwritten
            else:
                self._buf[k] = np.array(v, copy=True, order="C").reshape(-1)

    def alloc(self, name: str, dtype, count: int) -> None:
        dtype, count = np.dtype(dtype), int(count)
        cur = self._buf.get(name)
        if cur is not None and cur.dtype == dtype and cur.size >= count:
            return
        a = np.empty(max(count, 1), dtype)
        a.view(np.uint8)[:] = self._fill
        self._buf[name] = a

    def run(self, kernel: str, n: int, lanes: int | None = None) -> None:
        """the kernel over n threads (the walk: n groups of ``lanes`` threads, default ``self.lanes``)"""
        if self._variant is None:
            raise RuntimeError("RefLauncher.run before compile")
        if kernel != "walk" and lanes is not None:
            raise ValueError(f"RefLauncher.run: {kernel!r} takes no lanes")
        _T.xdem_gpu_ref(kernel, int(n), {**self._buf, "_caps": np.array([self._variant], np.uint32)},
                        int(self.lanes if lanes is None else lanes))

    def run_batch(self, steps: list[tuple]) -> None:
        """the steps' run() calls in order (a subclass hooking run() sees every kernel)"""
        for kernel, n, *lanes in steps:
            self.run(kernel, n, *lanes)

    def download(self, name: str, count: int | None = None) -> np.ndarray:
        a = self._buf[name]
        return (a if count is None else a[:count]).copy()

    def view(self, name: str, count: int) -> np.ndarray:
        a = self._buf[name]
        if not 0 <= int(count) <= a.size:
            raise ValueError(f"RefLauncher.view: {name} has {a.size} elements, asked for {count}")
        v = a[:int(count)]
        v.flags.writeable = False
        return v

    def free_bytes(self) -> int:
        """The reference runs in host memory: a fixed budget of 4 GiB."""
        return 1 << 32


def host_outputs(cpu, tables, rows, ckey, cdim):
    """the kernels' per-shot outputs (rows u64[n * W], ckey u64[n * key_words], cdim u8[n], no error code) in the CPU
    sampler's ``sample`` layouts, converted on the host (the oracle for "emit"): dets uint8[n, ceil(D/8)] = bytes
    0.. of each row, obs uint8[n, ceil(O/8)] = bytes 8 WD.., cspace int32[n] = 0 at dim C = 0, else the CPU table's
    id of the key (distinct keys resolved in order of first appearance)"""
    from xtim.gpu import tables as gt
    n, W, WD = len(cdim), int(tables["W"]), int(tables["WD"])
    nbD, nbO, kw = (int(tables["D"]) + 7) // 8, (int(tables["O"]) + 7) // 8, gt.key_words(tables)
    assert not (np.asarray(cdim) >= gt.XK_CDIM_CAPACITY).any()
    rb = np.ascontiguousarray(rows, dtype="<u8").reshape(n, W).view(np.uint8)
    dets, obs = rb[:, :nbD].copy(), rb[:, 8 * WD:8 * WD + nbO].copy()
    cs = np.zeros(n, np.int32)
    keys = np.asarray(ckey, np.uint64).reshape(n, kw)
    seen: dict[bytes, int] = {}
    order = []
    for s in np.flatnonzero(cdim):
        k = keys[s].tobytes()
        if k not in seen:
            seen[k] = len(order)
            order.append(keys[s])
    if order:
        ids = np.asarray(cpu.cspace_of_keys(np.array(order, np.uint64).reshape(-1, kw)), np.int32)
        for s in np.flatnonzero(cdim):
            cs[s] = ids[seen[keys[s].tobytes()]]
    return dets, obs, cs


def _reduced(rows):
    """an echelon basis of the span of rows (Python ints) as {pivot bit: vector}"""
    basis: dict[int, int] = {}
    for v in rows:
        v = int(v)
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                basis[p] = v
                break
            v ^= basis[p]
    return basis


def gf2_rank(rows) -> int:
    """rank over GF(2) of rows (Python ints)"""
    return len(_reduced(rows))


def in_span(v, rows) -> bool:
    """whether v (a Python int) is in the GF(2) span of rows"""
    basis = _reduced(rows)
    v = int(v)
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return False
        v ^= basis[p]
    return True


# ---- multi-word files (coord_words >= 2: more than 64 gauge qubits or span dimensions), for the GPU tests

def _synth(seed, **kw):
    import random
    from _xdem_synth import random_text
    return random_text(random.Random(seed), **kw)


def multiword_text(name: str) -> str:
    """a multi-word xdem text by name: "boundary_r65" (tests/data, r = 65: two words, rhoL = 1), "cw8_r400" (r = 400,
    seven words padded to eight; test_gpu_tables.py's instance), "wide_c" (r = 90, rhoL = 68: about 40% of its shots
    have dim C > 64, past the one-word counters, and about a quarter pass the zero decoder at such a dim)"""
    from pathlib import Path
    if name == "boundary_r65":
        return (Path(__file__).parent / "data" / "xdem" / "synth" / "boundary_r65.xdem").read_text()
    if name == "cw8_r400":
        r, n, A, L, O = 400, 300, 410, 300, 5
        return _synth(9000 + 7 * r + n, D=n - O, O=O, r=r, m=2, A=A, L=L, tc=L // 2)
    if name == "wide_c":
        return _synth(1, D=2, O=68, r=90, m=1, A=150, L=600, tc=300)
    raise KeyError(name)


MULTIWORD = ("boundary_r65", "cw8_r400", "wide_c")
