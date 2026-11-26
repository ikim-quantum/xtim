"""The GPU sampler's backend (design spec sections 5.5 to 5.8): batched launches of the xdem kernels through a
``Launcher``, returning the CPU sampler's arrays.

    b = GpuBackend(cpu, launcher, block=None, max_batch_bytes=None)   # cpu: xtim._xtim._XdemSampler
    dets, obs, cspace = b.sample(shots, seed)                          # the CPU sampler's layouts
    dets, obs, cspace, counts = b.sample_ps(shots, seed, flags)        # postselected: the survivors of shots attempts
    dets, obs, None = b.sample(shots, seed, status=False)              # no status: no C key, no C id (same rows)
    attempts, passed, hist = b.estimate(shots, seed, flags_or_None)    # the zero decoder's scores, device-counted
    b.prepare(); b.counters()                                          # compile + tables now; what has run

Stream: the shots of a seed are cut into blocks of ``block`` shots; block b depends only on (seed, b, block), so
``sample(n, seed)`` is a prefix of ``sample(m, seed)`` for n <= m, and no byte depends on the batch size, on the
launcher's memory or on overflow relaunches. The GPU streams differ from the CPU sampler's (distinct RNG tags,
xtim/gpu/tables.py): only the law is the same.

A batch is a whole number of blocks (the last one may be partial) of at most ``DEFAULT_BATCH_SHOTS`` shots, half the
launcher's free memory (or ``max_batch_bytes``) and ``BATCH_WORK`` units of expected work (``_shot_work``: a model of
the kernels' time per shot from the file's mean fired and extended lines per shot, in nanoseconds of the M3 Max, so a
submission takes about a second at most there; the bundled examples stay at ``DEFAULT_BATCH_SHOTS``, a DEM firing
thousands of lines per shot gets batches of a few hundred thousand shots). Between submissions the host is
responsive (Ctrl-C is serviced; a wait on the device is not interruptible). Each batch is one submission: "walk"
fills each block's fired extended-line lists, at most ``pair_cap`` records per block; "resolve" (and "resolve_law",
for the one-word laws it queues) writes the rows, dim C and the status C key per shot; "reduce" counts per block the
survivors and the zero decoder's passes by dim C; "reduce_groups" and "reduce2" sum those counters over the batch with
the walk's overflow codes into ``hist_total`` (``hist_layout(dimc).count`` u64), the one download every batch makes;
for ``sample`` / ``sample_ps``, "emit" (after "scan_blocks" under postselection) writes each survivor's output row. A block that fires more than ``pair_cap``
extended lines overflows (``XK_OVERFLOW_PAIRS``; the later kernels see it with empty lists, xdem_kernels.h's buffer
audit), and the batch reruns with ``pair_cap`` doubled (``XK_CAP_E``, a checked capacity that sizes no local array:
the kernels are recompiled, and the walk's lists ``eids`` / ``psh`` grow with it).

Outputs: "emit" writes each survivor's detector bytes, observable bytes and global C id (``out_det``, ``out_obs``,
``out_cs``) in the returned layouts, at its row among the batch's survivors, so the host copies each once. The C ids
come from ``ctab``, a read-only device table of the keys the backend has already resolved (``tables.cspace_table``,
at most ``CSPACE_TABLE_MAX_KEYS`` keys; uploaded only when it grows): a key not in it (and a dim C error code) is
listed in ``cmiss``, and the host resolves those keys through the CPU sampler's ``cspace_of_keys`` in shot order,
so the ids are those of resolving every key in shot order.

Without the status (``sample`` / ``sample_ps`` with ``status=False``): "resolve" runs with ``XK_MODE_NOKEY |
XK_MODE_NOSTATUS`` (no law builds C's key or the zero decoder's verdict) and "emit" writes the detector and observable
bytes only (no C id; cmiss lists only a shot whose dim C is an error code, which raises). The rows are those of the
default call, byte for byte (the key is written after the row and nothing else reads it), and the host touches no C
id or CPU C table.

Threads: one lock serialises the public calls (``sample``, ``sample_ps``, ``estimate``, ``prepare``, ``counters``):
they share the launcher's buffers, so a backend shared between threads runs one call at a time, and each call's
result is what it would be alone (cspace ids index the CPU sampler's table, which grows in the order calls see new
C).

Errors: every launcher call goes through ``_Guarded``, which turns any exception of the device (compile, pipeline,
allocation, upload, launch, a fault or a lost device) into ``XtimGpuError`` carrying the original text and the advice
to use ``device="cpu"``; the kernels' own error codes are ``XtimInvariantError``.

Buffers (``_Device``, the one owner of what is uploaded and allocated when): the read-only tables (``tab64``,
``tab32``) are packed once per mask (``None``: no postselection; the packings are cached by the mask's bytes) and
uploaded when the mask differs from the resident one; the working and output buffers are allocated at the largest
batch and grow only; per batch only ``dims`` and ``seed`` go up (and ``ctab`` when it grew).

Postselection (``sample_ps``): "resolve" runs with ``XK_MODE_POSTSELECT`` and the mask's span tables and writes a
flag per shot (survive / rejected early by the span test); "scan_blocks" gives each run of 64 shots its first output
row from "reduce"'s counts and the flags, and "emit" writes the survivors in shot order, of which only the batch's survivors' bytes come
back. The coin words are keyed per shot, so the survivors are exactly the rows of ``sample(shots, seed)`` whose masked
detectors are 0, in order.

The zero decoder's estimate (``estimate``): "resolve" also marks every surviving shot that the zero decoder passes
(score(status, zeros) > 0: the observable row lies in its C), and ``hist_total`` holds the batch's survivors and those
passes by dim C; the host sums the batches as int64. The counts are exactly those of ``sample`` / ``sample_ps``
followed by ``score`` with zero flips. "resolve" runs with ``XK_MODE_NOKEY`` there: the one-word law gives dim C and
the zero decoder's verdict without building C's key (``ckey`` is not written).

Both fast paths are supported: one coordinate word, and several (coord_words >= 2, any number of gauge qubits: the
multi-word law works in the device buffer ``mws``, ``tables.mw_words`` words per shot, which the batch size counts).
The general path (coord_words = 0), more than 4096 axes or rhoL > 253 is refused with ``ValueError``
(``tables.pack``'s message).
"""

from __future__ import annotations

import functools
import math
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from xtim.errors import XtimGpuError, XtimInvariantError
from xtim.gpu import structure as gs
from xtim.gpu import tables as gt
from xtim.gpu.launch import Launcher

_SOURCE = Path(__file__).with_name("xdem_kernels.h")
_U64_MAX = (1 << 64) - 1
_MAX_BATCH_SHOTS = 1 << 30  # dims[] and the kernels' indices are 32-bit
DEFAULT_BATCH_SHOTS = 1 << 24  # shots per batch without max_batch_bytes (M3 Max: 1.15-1.2x the rate of 4 Mi)
# the structured law by default only above this rho (GpuBackend(structured=None)): at or below it the generic law is its
# register form (XK_CAP_RHO <= 32), which was as fast on merlin_bh12 (rho 15: 2.4 vs 2.5 ns per estimated shot, M3 Max)
STRUCT_MIN_RHO = 32
# K1's shots per group of threads (GpuBackend(block=None)) and its threads per group: about 8 expected fires per
# thread's segment, one group per block of DEFAULT_BLOCK shots, WALK_LANES threads taking the segments in turn (M3 Max,
# K1's GPU ns per post-selected estimated shot, rows in device memory, against 256 shots and against 128 threads:
# merlin_bh50 1.53 vs 1.60 and 1.68; cultivation_d5 0.81 vs 0.86 and 0.93; merlin_bh12 0.57 vs 0.55 and 0.60)
DEFAULT_BLOCK = 512
WALK_LANES = 64
_MASKS_KEPT = 8  # packed tables kept per backend (one per postselection mask, least recently used evicted)
# Expected work per submission (``GpuBackend._shot_work``: about nanoseconds of kernel time on the M3 Max): a batch
# holds at most BATCH_WORK / (work per shot) shots, so one submission takes about a second there. Fitted on the M3 Max
# (tools/gpu_validate.py's machine): a fixed cost per shot, each fired line (walk draws and row XORs), each fired
# extended line (resolve's composition and law), and resolve's insertion sort of a shot's extended lines.
BATCH_WORK = 1_000_000_000
_WORK_SHOT, _WORK_FIRED, _WORK_EXT, _WORK_EXT2 = 2.0, 1.3, 3.0, 0.01


def _serialised(fn):
    """a public GpuBackend method run under the backend's lock (released on every exit, KeyboardInterrupt included)"""
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return call


class _Guarded:
    """A launcher whose every call raises only ``XtimGpuError`` (an exception of the device, converted with its
    original text) or what the launcher raised as ``XtimGpuError`` / ``XtimInvariantError`` already. KeyboardInterrupt
    and other non-``Exception`` exits pass through."""

    _STAGES = {"compile": "kernel compile / pipeline build", "upload": "upload", "alloc": "buffer allocation",
               "run": "kernel launch", "run_batch": "kernel launch", "download": "download", "view": "read-back",
               "free_bytes": "memory query"}

    def __init__(self, launcher: Launcher):
        self.inner = launcher
        self.name = getattr(launcher, "name", "?")

    def __getattr__(self, attr):
        fn = getattr(self.inner, attr)
        stage = self._STAGES.get(attr)
        if stage is None or not callable(fn):
            return fn

        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (XtimGpuError, XtimInvariantError):
                raise
            except Exception as e:
                raise _device_error(self.name, attr, stage, e, args) from e
        return call


def _device_error(device: str, attr: str, stage: str, e: Exception, args) -> XtimGpuError:
    """the XtimGpuError of a failed launcher call (``attr`` with ``args``)"""
    what = f"{type(e).__name__}: {e}".strip()
    extra = ""
    if attr in ("alloc", "upload") or isinstance(e, MemoryError):
        need = 0
        try:
            if attr == "alloc":
                need = int(args[2]) * np.dtype(args[1]).itemsize
            elif attr == "upload":
                need = sum(np.asarray(v).nbytes for v in args[0].values())
        except Exception:
            pass
        extra = (f" ({need} bytes requested; the batch is sized automatically from the device's free memory, so "
                 "fewer shots per call is not the fix)") if need else ""
    return XtimGpuError(f"GPU ({device}) {stage} failed{extra}: {what}\nThe GPU sampler cannot continue on this "
                        'device; use device="cpu" (same law, every feature).')


def _count(x, name: str, lo: int) -> int:
    v = int(x)
    if v != x or v < lo:
        raise ValueError(f"GpuBackend: {name} must be an integer >= {lo}, got {x!r}")
    return v


class _Device:
    """What the launcher holds for one backend: the resident read-only tables (by mask key) and the working and
    output buffers (grow-only). The one place that uploads and allocates: tables when the mask changes, buffers when
    a batch outgrows them, ``put`` for the per-batch buffers. A backend owns its launcher: another user of the same
    launcher would leave this record stale."""

    def __init__(self, launcher: Launcher):
        self.L = launcher
        self.tables_key: object = ()  # the mask key of the resident tab64 / tab32 (() = none yet)
        self.sizes: dict[str, int] = {}  # allocated elements per buffer

    def use_tables(self, key, st: gt.StaticTables) -> None:
        if self.tables_key != key:
            self.tables_key = ()  # a failed upload leaves nothing marked resident
            self.L.upload({"tab64": st.tab64, "tab32": st.tab32})
            self.tables_key = key

    def put(self, buffers: dict[str, np.ndarray]) -> None:
        """the per-batch uploads (dims, seed): same size every batch, overwritten in place"""
        self.L.upload(buffers)

    def ensure(self, need: dict[str, tuple[object, int]]) -> None:
        """every buffer of ``need`` (name -> (dtype, elements)) at least that large"""
        for name, (dtype, count) in need.items():
            if self.sizes.get(name, -1) < count:
                self.L.alloc(name, dtype, count)
                self.sizes[name] = count


class GpuBackend:
    """Batched GPU sampling of one compiled xdem file (``cpu``: the ``_xtim._XdemSampler`` holding its tables and
    its table of distinct C, which the returned cspace ids index)."""

    def __init__(self, cpu, launcher: Launcher, *, block: int | None = None, max_batch_bytes: int | None = None,
                 law_table: bool | None = None, defer: bool | None = None, structured: bool | None = None):
        self._cpu = cpu
        self._lock = threading.Lock()   # held by every public call (_serialised)
        self._L = launcher = _Guarded(launcher)
        self._max_batch_bytes = None if max_batch_bytes is None else _count(max_batch_bytes, "max_batch_bytes", 1)
        t = cpu.device_tables()
        # the law's path: the structured law (xk_law_struct, on a multi-word file xk_law_struct_mw) where
        # xtim.gpu.structure accepts the file (its axes renumbered), else the generic one (xk_law_fast, xk_law_multi).
        # structured=None: the structured law where the generic one would be its column-elimination form (rho >
        # STRUCT_MIN_RHO, every multi-word file; the register form is as fast below), True: wherever accepted, False:
        # never. law_path ("structured", "generic"; "multi-word structured", "multi-word") / law_path_reason say which
        # and why. The single-line law table holds the generic law's laws, whose corrections differ from the
        # structured law's by a coin shift: the structured law never uses it (law_table=True: the generic law; with
        # structured=True, a ValueError).
        if structured and law_table:
            raise ValueError("GpuBackend: structured=True and law_table=True (the law table is the generic law's)")
        multi = bool(gt.coord_words(t))
        one = not multi and int(t["A"])
        want = structured if structured is not None else int(t["rho"]) > STRUCT_MIN_RHO and not law_table
        self._plan, why = gs.plan(t) if want and int(t["A"]) else (None, "")
        if self._plan is not None:
            t = gs.renumber(t, self._plan)
            self.law_path, self.law_path_reason = "multi-word structured" if multi else "structured", ""
        elif multi:
            self.law_path = "multi-word"
            self.law_path_reason = why or ("structured=False" if structured is False
                                           else "law_table=True" if law_table else "")
        else:
            self.law_path = "generic"
            self.law_path_reason = why or ("no axes" if not one else "structured=False" if structured is False
                                           else "law_table=True" if law_table else f"rho <= {STRUCT_MIN_RHO}")
        self.tables = t
        # shots per walk group (K1's block): None = DEFAULT_BLOCK (the shots depend on it: block b of a seed is a
        # function of (seed, b, block))
        if block is None:
            block = DEFAULT_BLOCK
        self._block = _count(block, "block", 1)
        if self._block > gt.XK_MAX_BLOCK:
            raise ValueError(f"GpuBackend: block must be at most {gt.XK_MAX_BLOCK}, got {block}")
        self._pair_cap = max(64, math.ceil(4 * self._block * float(t["mean_ext"])))
        self._walk = gt.walk_plan(t, self._block)  # exact skip thresholds: computed once
        # the walk's threads per group: WALK_LANES, or fewer when there are fewer items (whole SIMD groups of 32)
        self._lanes = max(32, min(WALK_LANES, -(-self._walk.nitems // 32) * 32))
        gt.capacities(t, self._block, self._pair_cap, plan=self._walk, struct=self._plan)  # ValueError: unsupported
        # the single-line law table (computed once; False: none) and K2's queue of the shots needing their law
        # computed (XK_MODE_DEFER, run by "resolve_law"); neither changes an output byte. None: both where they pay
        # (tables.law_pays); the structured law: the queue, never the table (above).
        pays = gt.law_pays(t)
        if self._plan is not None:
            law_table = False
            defer = True if defer is None else defer
        self._law = gt.law_table(t) if (pays if law_table is None else law_table) else False
        # the one-word laws only (xdem_kernels.h, K2); the structured multi-word law runs from the queue only (defer
        # is ignored there: K2 queues each of its shots with a fired line)
        self._defer = (bool(pays if defer is None else defer) and not multi) or (multi and self._plan is not None)
        self._W, self._WD = int(t["W"]), int(t["WD"])
        self._nbD, self._nbO = (int(t["D"]) + 7) // 8, (int(t["O"]) + 7) // 8
        self._kw = gt.key_words(t)  # ckey words per shot: max(rhoL, 1) vectors of max(CW, 1) words
        # the multi-word law's scratch words per shot (0: one-word, or the structured law, which needs none)
        self._mw = gt.mw_words(t) if self._plan is None else 0
        self._hl = gt.hist_layout(gt.dimc_of(t))  # the counters' layout (dim C up to max(64, rhoL))
        self._source = _SOURCE.read_text()
        self._caps: dict[str, int] | None = None
        self._dev = _Device(launcher)
        self._static: dict[object, gt.StaticTables] = {}  # packed tables by mask key (None: no mask)
        self._free0: int | None = None  # the launcher's free bytes when first asked (before our buffers)
        # K4's C-id table: the keys resolved so far (u64[m, kw], at most ctab_max) and their ids; ctab_dev: the keys
        # in the device's ctab (None: not uploaded)
        self._ctab_max = min(gt.CSPACE_TABLE_MAX_KEYS, gt.CSPACE_TABLE_MAX_WORDS // self._kw)
        self._ckeys = np.zeros((0, self._kw), np.uint64)
        self._cids = np.zeros(0, np.int32)
        self._ctab_dev: int | None = None
        self.batches_run = 0  # batches launched (over the backend's life)
        self.relaunches = 0  # batches rerun after an XK_OVERFLOW_PAIRS
        self.shots_run = 0  # shots drawn (attempts) over the backend's life

    @_serialised
    def prepare(self) -> None:
        """Compile the kernels, ask the device for its free memory and upload the read-only tables without a mask, so
        that a device that cannot run them fails here (``XtimGpuError``), not at the first call.
        ``XdemSampler(..., device=...)`` calls it at construction."""
        self._compile()
        self._batch_blocks()
        self._dev.use_tables(*self._tables(None))

    @_serialised
    def counters(self) -> dict:
        """what the backend has run: ``shots`` (attempted shots drawn by sample / sample_ps / estimate), ``batches``
        (submissions, relaunches not counted), ``relaunches`` (batches rerun after a pair overflow), ``pair_cap`` (the
        current extended records per block), ``device`` (the launcher's name)"""
        return {"shots": self.shots_run, "batches": self.batches_run, "relaunches": self.relaunches,
                "pair_cap": self._pair_cap, "device": self._L.name}

    # ------------------------------------------------------------------------------------------------ batching

    def _compile(self) -> None:
        caps = gt.capacities(self.tables, self._block, self._pair_cap, plan=self._walk, struct=self._plan)
        if caps != self._caps:
            self._caps = None  # a failed compile leaves nothing marked as compiled
            self._L.compile(self._source, caps)
            self._caps = caps

    def _block_bytes(self, out: bool = False) -> int:
        """device bytes of one block of a batch (the working buffers sized by shots or blocks; with ``out``, emit's
        outputs for every shot surviving)"""
        B = self._block
        per_shot = 8 * self._W + 8 * self._kw + 1 + 1 + 8 * self._mw + 4  # rows, ckey, cdim, flag, mws, lawq
        if out:
            per_shot += self._nbD + self._nbO + 4 + 8  # out_det, out_obs, out_cs, cmiss
        # epos, eids and psh (two words per record), overflow, hist, boff
        return B * per_shot + 4 * (B + 1) + 12 * self._pair_cap + 4 + 4 * self._hl.stride + 4 * out

    def _shot_work(self) -> float:
        """the expected work of one shot (``BATCH_WORK``'s units), from the file's mean fired and extended lines per
        shot (E[ext^2] ~ ext (ext + 1) for the sort)"""
        t = self.tables
        fired, ext = float(t["mean_fired"]), float(t["mean_ext"])
        return _WORK_SHOT + _WORK_FIRED * fired + _WORK_EXT * ext + _WORK_EXT2 * ext * (ext + 1)

    def _batch_blocks(self, out: bool = False) -> int:
        budget = self._max_batch_bytes
        cap = max(1, min(_MAX_BATCH_SHOTS // self._block, (0xFFFF_FFFF // self._pair_cap),
                         int(BATCH_WORK // (self._shot_work() * self._block))))
        if budget is None:
            if self._free0 is None:
                self._free0 = int(self._L.free_bytes())
            budget = self._free0 // 2
            cap = min(cap, max(1, DEFAULT_BATCH_SHOTS // self._block))
        return max(1, min(budget // self._block_bytes(out), cap))

    def _tables(self, flags) -> tuple[object, gt.StaticTables]:
        """(mask key, packed read-only tables) of the mask ``flags`` (uint8[D], checked; None: no postselection),
        packed once per mask"""
        key = None if flags is None else flags.tobytes()
        st = self._static.pop(key, None)   # re-inserted last: the dict's order is least recently used first
        if st is None:
            span = None if flags is None else self._cpu.span_tables(flags)
            st = gt.static_tables(self.tables, block=self._block, span=span, plan=self._walk,
                                  law=self._law, struct=self._plan)
            if len(self._static) >= _MASKS_KEPT:
                self._static.pop(next(iter(self._static)))
        self._static[key] = st
        return key, st

    def _ensure(self, nb: int, out: bool) -> None:
        """the working buffers for a batch of nb blocks (and, with ``out``, emit's)"""
        B = self._block
        n = nb * B
        need = {"rows": (np.uint64, n * self._W), "epos": (np.uint32, nb * (B + 1)),
                "eids": (np.uint32, nb * self._pair_cap), "psh": (np.uint32, 2 * nb * self._pair_cap),
                "overflow": (np.uint32, nb),
                "ckey": (np.uint64, n * self._kw), "cdim": (np.uint8, n), "flag": (np.uint8, n),
                "coin_in": (np.uint64, 1),  # read only under XK_MODE_COIN_IN, which the backend never sets
                "hist": (np.uint32, nb * self._hl.stride),
                "hist_group": (np.uint64, gt.XK_HIST_GROUPS * self._hl.count),
                "hist_total": (np.uint64, self._hl.count),
                "mws": (np.uint64, max(1, n * self._mw)),  # the multi-word law's scratch (one word: unused)
                "lawq": (np.uint32, n + 1),  # K2's queue (XK_MODE_DEFER)
                "cmiss": (np.uint32, 1)}  # bound by walk, written under XK_MODE_EMIT only
        if out:
            need.update({"boff": (np.uint32, -(-n // gt.XK_EMIT_RUN)), "out_det": (np.uint8, n * self._nbD),
                         "out_obs": (np.uint8, n * self._nbO), "out_cs": (np.uint32, n),
                         "cmiss": (np.uint32, 2 * n + 1)})
        self._dev.ensure(need)

    def _batches(self, shots: int, seed: int, flags=None, first_block: int = 0,
                 out: bool = False, status: bool = True) -> Iterator[tuple[int, int, np.ndarray]]:
        """Run the batches of ``shots`` shots of ``seed``: yields ``(first shot, shots, hist_total)`` once the batch's
        "walk", "resolve" ("resolve_law"), "reduce", "reduce_groups" and "reduce2" (with ``out``: "scan_blocks" under
        postselection, then "emit") have run, its outputs in the launcher's buffers (read them before advancing).
        ``flags``: the postselection mask (uint8[D], checked) or None. ``status=False`` (with ``out``): no C key, no C
        id (XK_MODE_NOKEY | XK_MODE_NOSTATUS)."""
        ps = flags is not None
        mode = ((gt.XK_MODE_POSTSELECT if ps else 0) | (gt.XK_MODE_DEFER if self._defer else 0)
                | (gt.XK_MODE_EMIT if out else gt.XK_MODE_NOKEY))
        if out and not status:
            mode |= gt.XK_MODE_NOKEY | gt.XK_MODE_NOSTATUS
        key, st = self._tables(flags)
        L, B, hl = self._L, self._block, self._hl
        nblocks_total = -(-shots // B)
        b0 = 0
        while b0 < nblocks_total:
            nb = min(self._batch_blocks(out), nblocks_total - b0)
            s0 = b0 * B
            while True:
                n = min(nb * B, shots - s0)
                self._compile()
                self._dev.use_tables(key, st)
                self._ensure(nb, out)
                self._dev.put(gt.pack_batch(st, pair_cap=self._pair_cap, shots=n, seed=seed,
                                            first_block=first_block + b0, mode=mode))
                law = [("resolve_law", gt.law_threads(n))] if self._defer else []
                emit = []
                if out:
                    self._put_ctab()
                    emit = ([("scan_blocks", nb)] if ps else []) + [("emit", n)]
                L.run_batch([("walk", nb, self._lanes), ("resolve", n)] + law + [
                    ("reduce", nb), ("reduce_groups", gt.XK_HIST_GROUPS * hl.count), ("reduce2", hl.count)] + emit)
                tot = L.download("hist_total", hl.count)
                if tot[hl.capacity]:
                    raise XtimInvariantError(
                        f"GPU xdem walk: a size exceeds a compiled capacity ({self._caps}); this is a GPU backend "
                        "bug, please report it")
                if not tot[hl.pairs]:
                    break
                # some block fired more than pair_cap extended lines: a larger capacity, then the batch again
                self._pair_cap *= 2
                self.relaunches += 1
                nb = min(nb, self._batch_blocks(out))  # the batch's bytes grew with pair_cap
            self.batches_run += 1
            self.shots_run += n
            yield s0, n, tot
            b0 += nb

    # ------------------------------------------------------------------------------------------------ outputs

    def _put_ctab(self) -> None:
        """upload K4's C-id table when the resolved keys grew past the resident one (none: the empty table)"""
        m = len(self._cids)
        if self._ctab_dev != m:
            self._ctab_dev = None   # a failed upload leaves nothing marked resident
            self._L.upload({"ctab": gt.cspace_table(self._ckeys, self._cids)})
            self._ctab_dev = m

    def _check_cdim(self, cdim: np.ndarray, shots: np.ndarray | None = None) -> None:
        """``XtimInvariantError`` naming the first shot (``shots[i]``, or i) whose cdim is an error code
        (``XK_CDIM_*``), if any"""
        bad = cdim >= gt.XK_CDIM_CAPACITY
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            s = i if shots is None else int(shots[i])
            if cdim[i] == gt.XK_CDIM_CAPACITY:
                raise XtimInvariantError(
                    f"GPU xdem resolve: a size exceeds a compiled capacity ({self._caps}) at shot {s} of the batch; "
                    "this is a GPU backend bug, please report it")
            raise XtimInvariantError(
                f"GPU xdem resolve: the fast-path law's internal invariant failed at shot {s} of the batch (the CPU "
                "sampler raises std::logic_error there); please report it")

    def _emitted(self, k: int, dets: np.ndarray, obs: np.ndarray, cs: np.ndarray | None) -> None:
        """One batch's k output rows of "emit" into dets uint8[k, ceil(D/8)], obs uint8[k, ceil(O/8)], cs int32[k] (the
        CPU sampler's ``sample`` layouts), finishing the C ids emit left to the host (cmiss): the error codes raise
        (``_check_cdim``); the keys not in ctab are deduplicated (a 1-d unique: np.unique(axis=0) is far slower) and
        passed to cspace_of_keys in shot order, so a C new to the table gets the same id whatever the batching (the
        table grows in shot order), and join ctab's keys. ``cs=None`` (no status, XK_MODE_NOSTATUS): the rows only;
        cmiss then lists only error codes, which raise."""
        L = self._L
        if not k:
            return
        dets.reshape(-1)[:] = L.view("out_det", k * self._nbD)
        obs.reshape(-1)[:] = L.view("out_obs", k * self._nbO)
        if cs is not None:
            cs[:] = L.view("out_cs", k).view(np.int32)
        m = int(L.download("cmiss", 1)[0])
        if not m:
            return
        pairs = L.download("cmiss", 1 + 2 * m)[1:].reshape(m, 2)
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]   # shot order (the output rows are in shot order)
        rows, shots = pairs[:, 0], pairs[:, 1].astype(np.int64)
        self._check_cdim(L.view("cdim", int(shots.max()) + 1)[shots], shots)
        if cs is None:
            raise XtimInvariantError("GPU xdem emit: a shot was left to the host without an error code under "
                                     "XK_MODE_NOSTATUS; this is a GPU backend bug, please report it")
        k_ = np.ascontiguousarray(L.view("ckey", (int(shots.max()) + 1) * self._kw).reshape(-1, self._kw)[shots])
        v = k_[:, 0] if self._kw == 1 else k_.view(np.dtype((np.void, 8 * self._kw))).reshape(-1)
        uniq, first, inv = np.unique(v, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        ukeys = np.ascontiguousarray(uniq[order]).view(np.uint64).reshape(-1, self._kw)
        uids = np.asarray(self._cpu.cspace_of_keys(ukeys), np.int32)
        ids = np.empty(len(uniq), np.int32)
        ids[order] = uids
        cs[rows] = ids[inv.reshape(-1)]
        room = self._ctab_max - len(self._cids)
        if room > 0:   # the device table learns them, up to its size (a C's id never changes)
            # both computed before either is stored: an asynchronous exception between two assignments would leave
            # the key and id tables of different lengths
            ckeys = np.concatenate([self._ckeys, ukeys[:room]])
            cids = np.concatenate([self._cids, uids[:room]])
            self._ckeys, self._cids = ckeys, cids

    # ------------------------------------------------------------------------------------------------ sampling

    @_serialised
    def sample(self, shots: int, seed: int, *, first_block: int = 0, status: bool = True):
        """``(dets uint8[shots, ceil(D/8)], obs uint8[shots, ceil(O/8)], cspace int32[shots])``, the CPU sampler's
        ``sample`` layouts; cspace indexes the CPU sampler's table of C. Depends only on (seed, shots, block).
        ``status=False``: cspace is None, and no C key or C id is computed (device or host); dets and obs are the
        same bytes.

        ``first_block``: the stream starts at its block ``first_block`` (shot ``first_block * block``):
        ``sample(n, seed, first_block=k)`` is rows ``[k * block, k * block + n)`` of ``sample(k * block + n, seed)``
        (the kernels index blocks and shots with 64 bits)."""
        shots, seed, first_block = _stream_args(shots, seed, first_block, self._block)
        status = bool(status)
        dets = np.empty((shots, self._nbD), np.uint8)
        obs = np.empty((shots, self._nbO), np.uint8)
        cs = np.empty(shots, np.int32) if status else None
        for s0, n, _ in self._batches(shots, seed, first_block=first_block, out=True, status=status):
            self._emitted(n, dets[s0:s0 + n], obs[s0:s0 + n], cs[s0:s0 + n] if status else None)
        return dets, obs, cs

    @_serialised
    def sample_ps(self, shots: int, seed: int, flags, *, first_block: int = 0, status: bool = True):
        """Postselected ``sample``: ``(dets, obs, cspace, counts)`` for the survivors of ``shots`` attempted shots of
        the stream (``sample(shots, seed, first_block=...)``'s), in order: the shots whose detectors flagged in
        ``flags`` (uint8[D] or bool[D], nonzero = in the mask) are all 0. ``counts = {attempts, passed, discarded,
        early}`` as the CPU sampler's ``sample_ps`` (``early``: rejected by the span test before the law). cspace ids
        are given in the survivors' order of first appearance. An empty mask keeps every shot (``sample``'s
        bytes). ``status=False``: cspace is None (as ``sample``'s)."""
        shots, seed, first_block = _stream_args(shots, seed, first_block, self._block)
        status = bool(status)
        flags = self._mask(flags, "sample_ps")
        L = self._L
        parts = []
        early = 0
        for _, n, tot in self._batches(shots, seed, flags, first_block=first_block, out=True, status=status):
            k = int(tot[gt.XK_HIST_SURVIVE]) + int(tot[self._hl.error])   # the survivors (flag XK_FLAG_SURVIVE)
            fl = L.view("flag", n)   # read in place, before anything else touches the launcher
            early += int(np.count_nonzero(fl & gt.XK_FLAG_EARLY))
            del fl   # the view ends here
            part = (np.empty((k, self._nbD), np.uint8), np.empty((k, self._nbO), np.uint8), np.empty(k, np.int32))
            self._emitted(k, part[0], part[1], part[2] if status else None)
            parts.append(part)
        if len(parts) == 1:
            dets, obs, cs = parts[0]
        elif parts:
            dets, obs, cs = (np.concatenate(x) for x in zip(*parts))
        else:
            dets, obs, cs = (np.zeros((0, self._nbD), np.uint8), np.zeros((0, self._nbO), np.uint8),
                             np.zeros(0, np.int32))
        passed = len(dets)
        if not status:
            cs = None
        counts = {"attempts": shots, "passed": passed, "discarded": shots - passed, "early": early}
        return dets, obs, cs, counts

    def _mask(self, flags, who: str) -> np.ndarray:
        """the mask ``flags`` (uint8[D] or bool[D], nonzero = in the mask) as uint8[D] 0/1, checked"""
        D = int(self.tables["D"])
        f = np.asarray(flags)
        if f.shape != (D,):
            raise ValueError(f"GpuBackend.{who}: flags must have shape (D,) = ({D},), got {f.shape}")
        return np.ascontiguousarray(f != 0, dtype=np.uint8)

    @_serialised
    def estimate(self, shots: int, seed: int, flags, *, first_block: int = 0) -> tuple[int, int, np.ndarray]:
        """The zero decoder's scores over the stream of ``sample(shots, seed, first_block=...)``, with nothing but
        the batch totals downloaded (``hist_total``, ``hist_layout(dimc).count`` u64 per batch): ``(attempts, passed,
        hist)``. ``attempts = shots``; ``passed``: the survivors of ``flags`` (as ``sample_ps``; ``flags=None``: no
        postselection, ``passed = shots``); ``hist`` int64[max(64, rhoL) + 1]: ``hist[k]`` survivors with
        ``score(status, zeros) = 2^-k`` (the others score 0). Exactly the counts of ``sample`` / ``sample_ps``
        followed by ``score`` with zero flips. Sums are 64-bit (on the device per batch, on the host over batches). A shot whose dim C is an error
        code raises ``XtimInvariantError`` as ``sample`` does."""
        shots, seed, first_block = _stream_args(shots, seed, first_block, self._block)
        flags = None if flags is None else self._mask(flags, "estimate")
        L, hl = self._L, self._hl
        total = np.zeros(hl.stride, np.int64)
        for _, n, tot in self._batches(shots, seed, flags, first_block=first_block):
            h = tot[:hl.stride].astype(np.int64)
            if h[hl.error]:
                # cdim is written for survivors only (an early-rejected shot's entry is stale): look at those
                surv = (L.download("flag", n) & gt.XK_FLAG_SURVIVE) != 0
                cdim = L.download("cdim", n)
                self._check_cdim(np.where(surv, cdim, 0).astype(np.uint8))   # names the shot and the code
                raise XtimInvariantError("GPU xdem reduce: a shot with an error code was counted, but cdim shows "
                                         "none; this is a GPU backend bug, please report it")
            total += h
        hist = np.ascontiguousarray(total[gt.XK_HIST_PASS:gt.XK_HIST_PASS + hl.dimc + 1])
        return shots, int(total[gt.XK_HIST_SURVIVE]), hist


def _stream_args(shots, seed, first_block, block: int) -> tuple[int, int, int]:
    """sample's, sample_ps's and estimate's checks of (shots, seed, first_block): the stream's last block index,
    ``first_block + ceil(shots / block) - 1``, must fit 64 bits (the CPU sampler's rule and message), so that no batch
    runs before the refusal and no block index wraps"""
    shots = _count(shots, "shots", 0)
    seed = _count(seed, "seed", 0)
    if seed > _U64_MAX:
        raise ValueError(f"GpuBackend: seed must be < 2**64, got {seed}")
    first_block = _count(first_block, "first_block", 0)
    if first_block > _U64_MAX:
        raise ValueError(f"GpuBackend: first_block must be < 2**64, got {first_block}")
    nb = -(-shots // block)
    if nb and first_block > _U64_MAX - nb:
        raise ValueError("XdemSampler.sample: first_block out of range")
    return shots, seed, first_block
