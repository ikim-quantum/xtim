"""GPU sampling of xdem files (design spec section 5.7): which device is usable, and which files it supports.

    xtim.gpu.available()     # "metal", "cuda" or None: the first device that opens on this machine
    xtim.gpu.supports(text)  # (True, None) or (False, reason): whether a GPU sampler takes this xdem / DEM text

Both are cheap to call and never import a GPU package at module import: importing ``xtim.gpu`` imports neither
``objc`` / ``Metal`` nor ``cupy`` nor a launcher module. ``available()`` probes on every call (a package installed
later in the same process is seen). ``XdemSampler(text, device="gpu")`` resolves its device through
``xtim.gpu.available()``; the launcher of each device is opened by ``_open`` (the one place a backend is added).
"""
from __future__ import annotations

__all__ = ["available", "supports"]


def _open_metal():
    from xtim.gpu.metal import MetalLauncher    # imports pyobjc inside its constructor
    return MetalLauncher()


def _open_cuda():
    from xtim.gpu.cuda import CudaLauncher      # imports CuPy inside its constructor
    return CudaLauncher()


# (name, opener) in the order ``available()`` tries them; an opener raises (XtimGpuError) when its device is unusable.
# Metal first (it opens only on an arm64 Mac with pyobjc's Metal), then CUDA (CuPy and an NVIDIA device); the first
# that opens wins, else None.
_DEVICES = (
    ("metal", _open_metal),
    ("cuda", _open_cuda),
)


def _open(name: str):
    """The launcher of device ``name`` (``XtimGpuError`` if it cannot be opened)."""
    from xtim.errors import XtimGpuError
    for n, opener in _DEVICES:
        if n == name:
            return opener()
    raise XtimGpuError(f"the {name.upper() if name == 'cuda' else name} backend is not available in this build or "
                       f"on this machine (devices here: {', '.join(n for n, _ in _DEVICES)}; use device='cpu')")


def available() -> "str | None":
    """The name of the first usable GPU device of ``_DEVICES`` ("metal", then "cuda": a CUDA device needs CuPy
    importable and a device it can use), or None. Never raises."""
    for name, opener in _DEVICES:
        try:
            opener()
        except Exception:
            continue
        return name
    return None


def _supports_cpu(cpu) -> "tuple[bool, str | None]":
    """``supports`` for an already built ``xtim._xtim._XdemSampler`` (no second parse). Any number of gauge qubits
    and any span dimension: the one-word law (r, rho <= 64) and the multi-word law (coord_words >= 2) both run on the
    device."""
    from xtim.gpu.tables import XK_MAX_DIMC, XK_MAX_WA
    t = cpu.device_tables()
    cw, r, rho = int(t["coord_words"]), int(t["r"]), int(t["rho"])
    if cw == 0:
        return False, (f"this file takes the CPU sampler's general path (coord_words = {cw}, r = {r}): the GPU "
                       f"sampler handles the fast path only (one-word or multi-word coordinates); use device='cpu'")
    if int(t["WA"]) > XK_MAX_WA:
        return False, (f"{t['A']} axes: the GPU sampler handles at most {64 * XK_MAX_WA} axes (its per-thread "
                       f"memory); use device='cpu'")
    if int(t["rhoL"]) > XK_MAX_DIMC:
        return False, (f"rhoL = {t['rhoL']}: the GPU sampler counts dim C up to {XK_MAX_DIMC} (a byte per shot); "
                       f"use device='cpu'")
    return True, None


def supports(text) -> "tuple[bool, str | None]":
    """``(True, None)`` if ``XdemSampler(text, device="gpu")`` can sample this xdem / DEM text (a ``str`` or a
    ``stim.DetectorErrorModel``), else ``(False, reason)`` with the numbers that rule it out. Needs no GPU. A text
    the reader refuses raises ``XtimXdemError`` as ``XdemSampler`` does."""
    from xtim.xdem import XdemSampler
    return _supports_cpu(XdemSampler(text)._s)
