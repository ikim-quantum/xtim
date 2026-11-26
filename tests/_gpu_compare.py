"""Device-vs-reference comparisons of the GPU xdem sampler (test helper; tools/gpu_validate.py uses it too).

    out = run_kernels(L, t, buf, pair_cap, shots)   # every kernel on launcher L, then every buffer downloaded
    diff = first_difference(ref_out, got_out)        # None, or the first differing buffer / index / values
    outs = entry_outputs(text, kind, new_launcher, shots, seed)   # GpuBackend's sample / sample_ps / estimate
    t, buf, plan = kernel_variant(s, kind, shots, seed, E, structured=..., defer=..., status=...)   # the same, by mode
    heavy_dem(), heavy_ext(copies)                   # synthetic heavy files (many fired / extended lines per shot)

`run_kernels` runs GpuBackend's submission as one ``run_batch``: "walk", "resolve", "reduce", "reduce_groups",
"reduce2", then "scan_blocks" (under postselection) and "emit"; "resolve_law" runs after "resolve" when the buffers'
mode has XK_MODE_DEFER. So every kernel of xtim/gpu/_stubs.py (the result's "_kernels" lists those run) and every
buffer are compared; emit's list ``cmiss`` and K2's queue ``lawq``, filled atomically, are compared with their entries
sorted; the walk's working list ``psh`` is not compared (filled atomically, read by nothing past the walk; on a
block of more than E records which ones it holds depends on the atomics' order). ``kind``: None (no postselection), "all" or "half" (a fixed random half of the detectors).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import xtim
from xtim.gpu import tables as gt
from xtim.gpu.backend import GpuBackend

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = 64
HS = gt.XK_HIST_STRIDE


def supported_files() -> list[Path]:
    """the bundled examples the GPU sampler takes (xtim.gpu.supports)"""
    import xtim.gpu
    return [p for p in sorted((ROOT / "examples" / "xdem").glob("*.xdem")) if xtim.gpu.supports(p.read_text())[0]]


def heavy_dem(lines: int = 100_000, D: int = 1000, p: float = 0.04, seed: int = 0) -> str:
    """a conventional DEM firing about lines * p (4000) lines per shot: two detectors each, every 7th line also L0"""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, D, lines), rng.integers(0, D, lines)
    return "".join(f"error({p}) D{x} D{y}" + (" L0" if i % 7 == 0 else "") + "\n"
                   for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())))


def heavy_ext(copies: int, p: float = 0.1) -> str:
    """examples/xdem/ccz_toy_obs_p5.xdem's three extended lines repeated ``copies`` times at rate p: about 0.3 * copies
    fired extended lines per shot, so a default pair capacity of about 77 * copies per block of 64 shots"""
    src = (ROOT / "examples" / "xdem" / "ccz_toy_obs_p5.xdem").read_text().splitlines()
    axes = [x for x in src if x.startswith("axis")]
    ext = [x.split(")", 1)[1] for x in src if x.startswith("error") and " rot " in x]
    lines = [f"error({p}){e}" for _ in range(copies) for e in ext]
    return "\n".join([f"xdem 1 detectors 2 observables 1 gauge 2 hidden 1 tc {len(lines)}"] + axes + lines) + "\n"


def mask(D: int, kind, seed: int = 12345):
    """uint8[D]: None, every detector ("all") or a fixed random half ("half")"""
    if kind is None:
        return None
    if kind == "all":
        return np.ones(D, np.uint8)
    return (np.random.default_rng(seed).random(D) < 0.5).astype(np.uint8)


def pair_cap(t) -> int:
    return int(B * t["mean_ext"] + 10 * np.sqrt(B * t["mean_ext"] + 1) + 64)


def kernel_buffers(s, kind, shots: int, seed: int, E: int, first_block: int = 0) -> dict[str, np.ndarray]:
    """the packed buffers of one batch of sampler ``s`` under mask ``kind`` (postselection mode with a mask)"""
    flags = mask(s.num_detectors, kind)
    span = None if flags is None else s._s.span_tables(flags)
    mode = (0 if flags is None else gt.XK_MODE_POSTSELECT) | gt.XK_MODE_EMIT
    return gt.pack(s._s.device_tables(), block=B, pair_cap=E, shots=shots, seed=seed, first_block=first_block,
                   mode=mode, span=span)


def kernel_variant(s, kind, shots: int, seed: int, E: int, *, structured: bool = False, defer: bool = False,
                   status: bool = True, first_block: int = 0):
    """``(t, buf, plan)``: the tables (the structured law's renumbered ones with ``structured``), the packed buffers
    and the structured plan (None: the generic law) of one batch of sampler ``s`` under mask ``kind``, in the mode
    GpuBackend would use: ``defer`` (K2 queues the laws, "resolve_law" computes them; the structured multi-word law
    needs it), ``status=False`` (XK_MODE_NOKEY | XK_MODE_NOSTATUS: no C key, no C id). ``run_kernels(L, t, buf, E,
    shots, plan)`` runs it."""
    from xtim.gpu import structure as gs
    t, plan = s._s.device_tables(), None
    if structured:
        plan, why = gs.plan(t)
        if plan is None:
            raise ValueError(f"no structured plan: {why}")
        t = gs.renumber(t, plan)
    flags = mask(s.num_detectors, kind)
    span = None if flags is None else s._s.span_tables(flags)
    mode = ((0 if flags is None else gt.XK_MODE_POSTSELECT) | gt.XK_MODE_EMIT | (gt.XK_MODE_DEFER if defer else 0)
            | (0 if status else gt.XK_MODE_NOKEY | gt.XK_MODE_NOSTATUS))
    return t, gt.pack(t, block=B, pair_cap=E, shots=shots, seed=seed, first_block=first_block, mode=mode, span=span,
                      struct=plan), plan


def run_kernels(L, t, buf, E: int, shots: int, plan=None) -> dict[str, np.ndarray]:
    """compile, upload ``buf``, run the batch's kernels as one run_batch; every buffer downloaded (the inputs too;
    cmiss's pairs sorted; psh, K1's scratch, left out). ``plan``: the structured law's plan (``kernel_variant``'s).
    ``_kernels``: the names of the kernels run"""
    import time
    nb = -(-shots // B)
    hc = gt.hist_layout(gt.dimc_of(t)).count   # the counters per block (wider than HS + 2 when dim C may pass 64)
    t0 = time.perf_counter()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E, struct=plan))
    compile_s = time.perf_counter() - t0
    L.upload(buf)
    mode = int(buf["dims"][gt.XK_DIM_MODE])
    steps = ([("walk", nb), ("resolve", shots)]
             + ([("resolve_law", gt.law_threads(shots))] if mode & gt.XK_MODE_DEFER else [])
             + [("reduce", nb), ("reduce_groups", gt.XK_HIST_GROUPS * hc), ("reduce2", hc)]
             + ([("scan_blocks", nb)] if mode & gt.XK_MODE_POSTSELECT else []) + [("emit", shots)])
    L.run_batch(steps)
    out = {name: L.download(name) for name in buf}
    m = int(out["cmiss"][0])
    pairs = out["cmiss"][1:1 + 2 * m].reshape(m, 2)
    out["cmiss"][1:1 + 2 * m] = pairs[np.argsort(pairs[:, 0], kind="stable")].reshape(-1)
    q = int(out["lawq"][0])   # K2's queue: the shots in the order of the atomics, so its entries are compared sorted
    out["lawq"][1:1 + q] = np.sort(out["lawq"][1:1 + q])
    del out["psh"]
    out["_compile_s"] = compile_s
    out["_kernels"] = [name for name, *_ in steps]
    return out


def first_difference(ref: dict, got: dict) -> dict | None:
    """None when every buffer of ``ref`` is byte-equal in ``got``, else the first differing one:
    ``{"buffer", "index", "ref", "got"}`` (index -1 for a dtype / shape mismatch)"""
    for name, a in ref.items():
        if name.startswith("_"):
            continue
        b = got.get(name)
        if b is None or b.dtype != a.dtype or b.shape != a.shape:
            return {"buffer": name, "index": -1, "ref": f"{a.dtype}{a.shape}",
                    "got": None if b is None else f"{b.dtype}{b.shape}"}
        if a.tobytes() != b.tobytes():
            fa, fb = np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(b).reshape(-1)
            i = int(np.flatnonzero(fa.view(np.uint8) != fb.view(np.uint8))[0]) // a.dtype.itemsize
            return {"buffer": name, "index": i, "ref": int(fa[i]), "got": int(fb[i])}
    return None


def entry_outputs(text: str, kind, new_launcher, shots: int, seed: int, first_block: int = 0,
                  nostatus: bool = False, **kw) -> dict[str, np.ndarray]:
    """GpuBackend's outputs on a launcher of ``new_launcher()`` (a fresh backend per entry point): without a mask
    ``sample`` and ``estimate``; with one ``sample_ps`` and ``estimate``. ``nostatus``: also ``sample`` /
    ``sample_ps`` with ``status=False`` (``sample_nostatus.dets`` / ``.obs``, and ``.counts`` under a mask)"""
    out = {}
    s = xtim.XdemSampler(text)
    b = GpuBackend(s._s, new_launcher(), **kw)
    flags = mask(s.num_detectors, kind)
    if flags is None:
        d, o, c = b.sample(shots, seed, first_block=first_block)
        out.update({"sample.dets": d, "sample.obs": o, "sample.cspace": c})
        if nostatus:
            d, o, c0 = b.sample(shots, seed, first_block=first_block, status=False)
            assert c0 is None
            out.update({"sample_nostatus.dets": d, "sample_nostatus.obs": o})
    else:
        d, o, c, counts = b.sample_ps(shots, seed, flags, first_block=first_block)
        out.update({"sample_ps.dets": d, "sample_ps.obs": o, "sample_ps.cspace": c,
                    "sample_ps.counts": np.array([counts[k] for k in sorted(counts)], np.int64)})
        if nostatus:
            d, o, c0, counts = b.sample_ps(shots, seed, flags, first_block=first_block, status=False)
            assert c0 is None
            out.update({"sample_ps_nostatus.dets": d, "sample_ps_nostatus.obs": o,
                        "sample_ps_nostatus.counts": np.array([counts[k] for k in sorted(counts)], np.int64)})
    s = xtim.XdemSampler(text)
    b = GpuBackend(s._s, new_launcher(), **kw)
    att, passed, hist = b.estimate(shots, seed, mask(s.num_detectors, kind), first_block=first_block)
    out.update({"estimate.counts": np.array([att, passed], np.int64), "estimate.hist": np.asarray(hist)})
    return out
