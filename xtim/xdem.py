"""xdem file-only sampling with decoder feedback (docs/xdem.md).

    text = xtim.Circuit(c).detector_error_model_text(format="xdem")    # the conventional DEM when nothing rotates
    dem  = xtim.Circuit(c).detector_error_model(format="xdem")         # stim.DetectorErrorModel; raises if the text
                                                                       # is extended (stim cannot hold it)
    xtim.xdem.canonical_report(circuit)                                # (ok, reason): C1', C2' on the circuit

    s = xtim.XdemSampler(text, *, cache_cap=65536, device="cpu")       # an xdem or any conventional stim DEM (no flag)
    s.device                 # "cpu", or the GPU in use ("metal", "cuda"); device="gpu" | "metal" | "cuda": below
    s.extended               # False for a conventional DEM
    s.num_detectors, s.num_observables, s.num_gauge, s.num_hidden, s.num_axes, s.num_lines, s.num_rotating
    s.tc                     # first late line index (num_lines for a conventional DEM)
    s.late_axes              # tuple of axis ids with a nonzero late exponent on some line
    s.r2b_available          # (bool, reason) per §4.4
    s.invisible_residues     # computed from the file: rank(img[A₁]) − rank(img[A₁]|_D),
                             # A₁ = axes with an odd rot on some line; property only, no warning

    dets, status = s.sample(shots, *, seed, threads=1, postselect=None, fired=False)
    dets, obs = s.sample(shots, *, seed, status=False)           # no status: detectors and observables only
    passes = s.score(status, flips, *, bit_packed=False)
    passes = s.score(status, *, lines=L, correction=None)        # None = "r2b"; "r2b" | "optimal" | "none"
    r = s.infidelity(shots, decoder, *, seed, threads=1, batch=100_000, postselect=None, lines=False,
                     correction=None)                            # None = "r2b" with lines=True; with flips: no value
    r = s.infidelity(shots, None, *, seed, ...)                  # the zero decoder (on a GPU: counted on the device)
    corr = s.correction(status, *, lines=L, correction="r2b")   # export, below: "r2b" | "optimal"
    s.counters()
    xtim.score_plain(obs, flips, *, bit_packed=False)

The default correction is "r2b": ``score`` and
``infidelity`` take ``correction=None``, which means "r2b" with a hypothesis (``lines=``) and must stay None with
flips; ``correction()`` defaults to "r2b". ``"optimal"`` is opt-in: the reference (best-possible) correction, and
slower.

A thin wrapper over the C++ reader / sampler / scorer (``xtim._xtim._XdemSampler``, ``_xtim.score_plain``): every loop
over shots or lines runs in C++; Python only calls the decoder and unpacks bits for it. Refusals of the file are
``xtim.XtimXdemError`` (a subclass of ``xtim.XtimError``, not of ``ValueError``) naming the cause and the text line; a
dc-DEM v1 file is refused by name (re-export the circuit with ``format="xdem"``).

GPU (``device="gpu"``, or "metal" / "cuda" to name one; default "cpu", which imports nothing GPU and changes no byte):
``sample(shots, seed=...)`` runs on the device for the fast-path files, one-word and multi-word
(``xtim.gpu.supports(text)``); its stream differs from the CPU's (same law), and ``threads`` is ignored. ``postselect=`` runs on the device too (the
CPU's argument rules, return types and counts; there the survivors ARE the rows of ``sample(shots, seed=s)`` whose
masked detectors are 0, in order). Everything else raises ``xtim.XtimGpuError`` at construction (no device, extras
not installed, unsupported file) or at the call (``fired=True``, ``lines``-based scoring and ``correction()``: not
available on the GPU yet; ``device="cpu"`` serves them). ``score(status, flips)`` and ``infidelity(...,
lines=False)``, with or without ``postselect``, work as on the CPU. ``infidelity(shots, None)`` (the zero decoder)
runs on the device with only per-block counters coming back.

Readout qubits are the detectors, then the observables; a shot's status keeps the space C of its observables that the
gauge qubits leave undetermined, so ``score(status, flips)`` = 2^-dim C if (v XOR flips) lies in C, else 0.

Stream (one definition): the shots of ``sample(shots, seed=s)`` are cut into blocks of 1024; block b depends only on
(s, b), never on ``threads``. ``infidelity(shots, decoder, seed=s)`` consumes exactly the shots of
``sample(shots, seed=s)``, in order, whatever ``batch`` is (each batch starts at its block offset in the stream).
Calling ``sample`` twice with the same seed returns the same shots: use a new seed for new shots.

Postselection (``sample(..., postselect=m)``, and ``infidelity(..., postselect=...)``) runs in C++: ``shots`` are
attempted shots and only the survivors (every detector of the mask 0) come back. They are distributed exactly as
``sample(shots, seed=s)`` followed by a filter and depend only on (seed, shots), never on ``threads``, but they are not
the same bytes. A shot whose detector row, restricted to the mask, cannot be brought into the mask's zero set by the
axes is rejected before its gauge-qubit work (the span test: exact, never a heuristic; counted in ``early``).

Decoder corrections (docs/xdem.md §4): ``score(status, lines=L, correction=...)`` takes the decoder's hypothesis per
shot as xdem line ids (line i = the file's i-th error line; any order, repeats cancel) on a status from
``sample(..., fired=True)``, and returns pass = Pr(o = c | d, the fired lines, the correction):

- ``"none"``: law-aware flips (c = the hypothesis law's definite observable value given d, else its XOR flips).
- ``"r2b"`` (the default): the hypothesis' late part as one Pauli at t_c, applied by rule R2b (docs/xdem.md §4). Needs
  ``r2b_available[0]``; otherwise ``XtimXdemError`` naming the reason (pass ``correction="optimal"`` or ``"none"``).
- ``"optimal"`` (opt-in): the reference, best-possible correction K (docs/xdem.md §4), c = 0; slower than "r2b". A
  readout whose pivot system has no solution is lossy (pass ½ per lossy observable for every K; the prediction on
  the lossy observables is a point of law([Ê, K])'s support given d); lossy readouts are common on real
  circuits under the true hypothesis (``counters()["lines_lossy"]``).

``correction(status, lines=L, correction=...)`` exports that correction per shot (``XdemCorrection``) for a physical
implementation; mapping axis ids to physical Paulis needs the circuit.
"""
from __future__ import annotations

import math
import operator
import sys
from dataclasses import dataclass

import numpy as np

from . import _xtim

__all__ = ["XdemSampler", "XdemStatus", "XdemCorrection", "InfidelityResult", "canonical_report", "score_plain"]

BLOCK_SHOTS = 1024  # the C++ stream's block (xdem_sampler.hpp kBlockShots)
DEFAULT_CACHE_CAP = 65536  # the law caches' and the K memo's bound (xdem_sampler.hpp kDefaultCacheCap)
DEVICES = ("cpu", "gpu", "metal", "cuda")
CORRECTIONS = ("optimal", "r2b", "none")
DEFAULT_CORRECTION = "r2b"  # the default correction; the C++ binding has no default


def _seed(seed) -> int:
    try:
        s = operator.index(seed)
    except TypeError:
        raise TypeError(f"seed must be an int, got {type(seed).__name__}") from None
    if not 0 <= s < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {s}")
    return s


def _count(x, name: str, lo: int) -> int:
    try:
        v = operator.index(x)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(x).__name__}") from None
    if v < lo:
        raise ValueError(f"{name} must be >= {lo}, got {v}")
    return v


def _unpack(packed: np.ndarray, num_bits: int) -> np.ndarray:
    """stim b8 rows (uint8, little-endian bit order) -> bool[shots, num_bits]."""
    if num_bits == 0:
        return np.zeros((packed.shape[0], 0), dtype=bool)
    return np.unpackbits(packed, axis=1, bitorder="little", count=num_bits).astype(bool)


def _flips01(flips, who: str) -> np.ndarray:
    """flips as C-contiguous uint8 0/1 (bool, or integers that are all 0/1); anything else is refused."""
    f = np.asarray(flips)
    if f.dtype == np.bool_:
        return np.ascontiguousarray(f, dtype=np.uint8)
    if not np.issubdtype(f.dtype, np.integer):
        raise TypeError(f"{who}: values must be bool or integer 0/1, got dtype {f.dtype}")
    if f.size and not ((f == 0) | (f == 1)).all():
        raise ValueError(f"{who}: values must be 0/1")
    return np.ascontiguousarray(f, dtype=np.uint8)


def _flips_matrix(flips, shots: int, O: int, bit_packed: bool, who: str) -> np.ndarray:
    """uint8[shots, O] 0/1 from either layout; a shape mismatch is a ValueError naming the expected shape."""
    if bit_packed:
        f = np.asarray(flips)
        nb = (O + 7) // 8
        if f.dtype != np.uint8 or f.shape != (shots, nb):
            raise ValueError(f"{who}: bit_packed flips must be uint8 with shape (shots, ceil(O/8)) = ({shots}, {nb}),"
                             f" got {f.dtype} {f.shape}")
        return _unpack(f, O).view(np.uint8)
    f = _flips01(flips, who)
    if f.shape != (shots, O):
        raise ValueError(f"{who}: flips must have shape (shots, O) = ({shots}, {O}), got {f.shape}")
    return f


def _lines_csr(lines, shots: int, num_lines: int, who: str):
    """A decoder's hypotheses as a CSR (offsets int64[shots + 1], ids int32): from a list of per-shot int arrays /
    sequences (converted in C++, ``_xtim.lines_csr``), or a tuple (offsets, ids). A tuple is read as CSR only if
    offsets is a 1-d integer array of length shots + 1, non-decreasing from 0, and ids a 1-d integer array of length
    offsets[-1]; any other tuple is a ValueError (a tuple of per-shot arrays is not guessed at: per-shot lists go in a
    list). Ids are range-checked as 64-bit integers before the int32 cast (a wrapped id cannot pass)."""
    if not isinstance(lines, tuple):
        # a list of per-shot sequences / integer arrays: one C++ pass (range and type checked there)
        return _xtim.lines_csr(lines, shots, num_lines, who)

    def refuse(what):
        raise ValueError(f"{who}: a tuple is read as a CSR pair (offsets, ids): offsets a 1-d integer array of length "
                         f"shots + 1 = {shots + 1}, non-decreasing from 0, ids a 1-d integer array of length "
                         f"offsets[-1]; {what}. Per-shot line lists go in a list, not a tuple: lines=[ids_0, ids_1, "
                         f"...]")
    if len(lines) != 2:
        refuse(f"got a tuple of {len(lines)}")
    try:   # numpy's own refusal (e.g. a ragged nested list) is the same misuse: the hint applies
        off = np.asarray(lines[0])
        ids = np.asarray(lines[1])
    except ValueError as e:
        refuse(f"numpy could not read the pair as arrays ({e})")
    if off.ndim != 1 or off.shape[0] != shots + 1:
        refuse(f"CSR offsets must have shape ({shots + 1},), got {off.shape}")
    if not np.issubdtype(off.dtype, np.integer):
        refuse(f"CSR offsets have dtype {off.dtype}")
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        refuse(f"CSR ids have shape {ids.shape} and dtype {ids.dtype}")
    off = np.ascontiguousarray(off, dtype=np.int64)
    if off[0] != 0:
        refuse(f"CSR offsets start at {int(off[0])}")
    if (np.diff(off) < 0).any():
        refuse(f"CSR offsets decrease at shot {int(np.flatnonzero(np.diff(off) < 0)[0])}")
    if int(off[-1]) != ids.shape[0]:
        refuse(f"CSR offsets end at {int(off[-1])} but there are {ids.shape[0]} ids")
    if ids.dtype != np.int32:   # an int32 CSR (status.fired's) is range-checked as it is: it cannot wrap
        ids = np.ascontiguousarray(ids, dtype=np.int64)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= num_lines):
        bad = np.flatnonzero((ids < 0) | (ids >= num_lines))
        i = int(bad[0])
        s = int(np.searchsorted(off, i, side="right")) - 1
        raise ValueError(f"{who}: line id {int(ids[i])} at shot {s} is outside [0, num_lines = {num_lines})")
    return off, np.ascontiguousarray(ids, dtype=np.int32)


def _gpu_backend(cpu, device: str):
    """``(GpuBackend, name)`` for the C++ sampler ``cpu`` on ``device`` ("gpu", "metal" or "cuda"), its kernels
    compiled and its tables uploaded (``GpuBackend.prepare``): ``XtimGpuError`` for an unsupported file, an unusable
    device, a missing extra, or a device that cannot compile or hold the kernels, before any sample."""
    from . import gpu
    from .errors import XtimGpuError
    ok, reason = gpu._supports_cpu(cpu)
    if not ok:
        raise XtimGpuError(f"device={device!r}: {reason}")
    name = device
    if device == "gpu":
        name = gpu.available()
        if name is None:
            raise XtimGpuError('device="gpu": no usable GPU device (on Apple silicon: pip install "xtim[metal]" and '
                               'run on a Metal device; on NVIDIA: install the CuPy wheel matching your CUDA, e.g. '
                               'pip install cupy-cuda12x); '
                               'use device="cpu"')
    from .gpu.backend import GpuBackend
    try:
        b = GpuBackend(cpu, gpu._open(name))
        prepare = getattr(b, "prepare", None)   # (a stand-in backend of the routing tests has none)
        if prepare is not None:
            prepare()
    except XtimGpuError:
        raise
    except Exception as e:   # a launcher's own exception while opening (the backend converts its calls' errors)
        raise XtimGpuError(f"device={name!r}: the GPU sampler could not be set up: {type(e).__name__}: {e}; use "
                           'device="cpu"') from e
    return b, name


def _gpu_needs_cpu(what: str):
    """the refusal of a call that needs the fired lines, which the GPU sampler does not return yet"""
    from .errors import XtimGpuError
    raise XtimGpuError(f"{what} needs the fired lines, which the GPU sampler does not return yet (use device='cpu')")


def _correction_name(correction, who: str) -> str:
    if not isinstance(correction, str) or correction not in CORRECTIONS:
        raise ValueError(f'{who}: correction must be "optimal", "r2b" or "none", got {correction!r}')
    return correction


class XdemStatus:
    """Per-shot status of an ``XdemSampler.sample`` call (docs/xdem.md §5).

    ``obs``: uint8[shots, ceil(O/8)], the sampled observable bits v (stim b8 packing).
    ``cspace``: int32[shots], an index into ``cspaces``.
    ``cspaces``: list of uint8[dim, O] bases; entry i is the space C of observable values the gauge qubits leave
    undetermined given the detectors (fully reduced RREF, pivot = highest observable, descending); entry 0 = {0}
    (dim 0). The table belongs to the sampler that produced the status (append-only over its calls); ``score``
    refuses a status of another sampler. A status from ``sample`` builds ``cspaces`` on first access: a copy of the
    sampler's first ``num_cspaces`` entries (the table's size when the call returned), taken under the sampler's lock.
    ``fired``: None, or with ``sample(..., fired=True)`` the pair ``(offsets int64[shots + 1], ids int32[total])``,
    the CSR of each shot's fired line ids (ascending = file order; a line fires at most once per shot): shot s fired
    ``ids[offsets[s]:offsets[s + 1]]``. Under postselection: the survivors' lists, in the rows' order. With
    ``fired=True`` the status also keeps a reference to the call's detector rows, which ``score(lines=)`` and
    ``correction()`` read (not a copy: changing those rows in place after sampling changes what they read)."""

    __slots__ = ("obs", "cspace", "fired", "_cspaces", "_ncs", "_owner", "_dets")

    def __init__(self, obs: np.ndarray, cspace: np.ndarray, cspaces: "list | None", _owner=None, _ncs=None,
                 fired=None, _dets=None):
        self.obs = obs
        self.cspace = cspace
        self.fired = fired
        self._dets = _dets      # with fired: the call's detector rows (a reference; score(lines=) reads them)
        self._cspaces = cspaces
        self._ncs = len(cspaces) if cspaces is not None else int(_ncs)
        self._owner = _owner

    @property
    def cspaces(self) -> list:
        if self._cspaces is None:
            self._cspaces = self._owner._s.cspaces(0, self._ncs)
        return self._cspaces

    @property
    def num_cspaces(self) -> int:
        return self._ncs

    def __len__(self) -> int:
        return int(self.cspace.shape[0])

    def __repr__(self) -> str:
        return f"XdemStatus(shots={len(self)}, num_cspaces={self._ncs})"


@dataclass(frozen=True, eq=False)
class XdemCorrection:
    """``XdemSampler.correction``'s export (docs/xdem.md §5): the correction per shot, for a physical implementation. The
    members of the other kind are None.

    ``kind == "optimal"`` (docs/xdem.md §4), per shot s and observable k (CSR row = s * O + k):
      ``axes = (offsets int64[shots * O + 1], ids int32)``: S_k, the axis ids (ascending);
      ``pivots = (offsets int64[shots * O + 1], ids int32)``: P_k, the pivot detector ids (ascending);
      ``y``, ``frame``, ``lossy``: bool[shots, O].
      Meaning: measure, in place of observable k, the Hermitian Pauli (-1)^frame · ι · X_{L_k} ∏_{a ∈ S_k} Q_a
      (ι = i iff y), whose noiseless value is +1. ``lossy[s, k]``: the readout has no pivot solution, so observable k
      is a fair coin for every correction (S_k = P_k = empty); it is read plainly and ``frame`` is the predicted value
      (a point of law([Ê, K])'s support on the lossy observables given d, the one nearest Ê's flips,
      observable 0 first; 0 when d is outside law(Ê)'s detector support). Lossy readouts are common on real circuits under the true hypothesis.
    ``kind == "r2b"`` (docs/xdem.md §4), per shot s:
      ``late = (offsets int64[shots + 1], axis int32, eta int8)``: eta^>, the late exponent (1..3) per late axis
      (ascending axis);
      ``late_tau = (offsets int64[shots + 1], ids int32)``: tau_K, the late axes (ascending);
      ``applied``: bool[shots] (rule R2b's guard); ``c``: bool[shots, O], the prediction scored.
      Meaning: a Pauli at t_c anticommuting with exactly the late axes in ``late_tau``, applied iff ``applied``."""

    kind: str
    axes: "tuple | None" = None
    pivots: "tuple | None" = None
    y: "np.ndarray | None" = None
    frame: "np.ndarray | None" = None
    lossy: "np.ndarray | None" = None
    late: "tuple | None" = None
    late_tau: "tuple | None" = None
    applied: "np.ndarray | None" = None
    c: "np.ndarray | None" = None


@dataclass(frozen=True)
class InfidelityResult:
    """``XdemSampler.infidelity``'s result. ``mean`` = 1 - mean(pass) over the accepted shots (nan if none),
    ``stderr`` = sample standard deviation of pass over the accepted shots / sqrt(accepted) (nan if < 2),
    ``shots`` = shots drawn, ``accepted`` = shots kept by ``postselect`` (= shots without it),
    ``giveup_fraction`` = give-up shots / accepted (0 when the decoder reports none)."""

    mean: float
    stderr: float
    shots: int
    accepted: int
    giveup_fraction: float


class XdemSampler:
    """File-only sampler of an xdem text (``Circuit.detector_error_model_text(format="xdem")``) or of any
    conventional stim DEM (no flag), given as text or as a ``stim.DetectorErrorModel`` (read through ``str()``).

    ``cache_cap``: the bound on the sampler's law caches and the optimal correction's memo (a full cache is cleared).
    No output byte depends on it.

    ``device``: "cpu" (the default: the C++ sampler, nothing GPU is imported), "gpu" (the first usable device of
    ``xtim.gpu.available()``), "metal" or "cuda". A GPU device needs its extras (Metal: ``pip install "xtim[metal]"``;
    CUDA: the CuPy wheel of the machine's CUDA, e.g. ``pip install cupy-cuda12x``) and a file ``xtim.gpu.supports``
    accepts; otherwise ``xtim.XtimGpuError`` names the cause, here, not at the first ``sample`` (the kernels are
    compiled and the file's tables uploaded at construction). A device failure later (allocation, launch, a lost
    device) is also ``XtimGpuError``, with the device's own message; ``device="cpu"`` is the remedy.
    ``XdemSampler.device`` is the device in use ("gpu" resolves to its name). A GPU sampler may be shared between
    threads: its calls are serialised (one at a time on the device), each giving what it gives alone."""

    def __init__(self, text, *, cache_cap: int = DEFAULT_CACHE_CAP, device: str = "cpu"):
        if not isinstance(device, str) or device not in DEVICES:
            raise ValueError(f'device must be one of "cpu", "gpu", "metal" or "cuda", got {device!r}')
        stim = sys.modules.get("stim")    # a stim object implies stim is imported
        if stim is not None and isinstance(text, stim.DetectorErrorModel):
            text = str(text)
        if not isinstance(text, str):
            raise TypeError(f"XdemSampler expects xdem / DEM text (str) or a stim.DetectorErrorModel, got "
                            f"{type(text).__name__}")
        self._s = _xtim._XdemSampler(text, _count(cache_cap, "cache_cap", 1))
        self._gpu = None            # the GpuBackend of a GPU sampler; the only thing the CPU path ever checks
        self._device = device
        if device != "cpu":
            self._gpu, self._device = _gpu_backend(self._s, device)

    @property
    def device(self) -> str:
        """"cpu", or the GPU in use ("metal", "cuda"); ``device="gpu"`` resolves to the device's name."""
        return self._device

    @property
    def extended(self) -> bool:
        """False for a conventional DEM (no xdem header)."""
        return self._s.extended

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    @property
    def num_gauge(self) -> int:
        """r, the gauge qubits."""
        return self._s.num_gauge

    @property
    def num_hidden(self) -> int:
        """m, the hidden-syndrome qubits."""
        return self._s.num_hidden

    @property
    def num_axes(self) -> int:
        return self._s.num_axes

    @property
    def num_lines(self) -> int:
        return self._s.num_lines

    @property
    def num_rotating(self) -> int:
        """The error lines with a nonempty rot or late field."""
        return self._s.num_rotating

    @property
    def tc(self) -> int:
        """The first late line's index (num_lines for a conventional DEM)."""
        return self._s.tc

    @property
    def late_axes(self) -> tuple:
        """The axis ids with a nonzero late exponent on some line, ascending."""
        return self._s.late_axes

    @property
    def r2b_available(self) -> tuple:
        """``(True, None)``, or ``(False, reason)`` naming the first line or axis that stops rule R2b (docs/xdem.md §4): a
        line at or after t_c that flips a detector or carries a rotation, or a late axis with a detector in its
        img. ``correction="r2b"`` refuses with that reason."""
        return self._s.r2b_available

    @property
    def invisible_residues(self) -> int:
        """rank(img[A1]) - rank(img[A1] on the detectors), A1 = the axes with an odd rot on some line,
        computed from the file. A property only: no warning, no refusal."""
        return self._s.invisible_residues

    def __repr__(self) -> str:
        dev = "" if self._gpu is None else f", device={self._device!r}"
        return (f"XdemSampler(extended={self.extended}, num_detectors={self.num_detectors}, "
                f"num_observables={self.num_observables}, num_axes={self.num_axes}, num_lines={self.num_lines}, "
                f"num_rotating={self.num_rotating}{dev})")

    def _sample(self, shots: int, seed: int, threads: int, first_block: int):
        dets, obs, cs = self._s.sample(shots, seed, threads, first_block)
        # the C table is copied only if status.cspaces is read (infidelity never reads it)
        return dets, XdemStatus(obs, cs, None, self, self._s.num_cspaces())

    def sample(self, shots: int, *, seed: int, threads: int = 1, postselect=None, fired: bool = False,
               status: bool = True):
        """``(dets, status)``: dets uint8[shots, ceil(D/8)] (stim b8 packing), status an ``XdemStatus``.
        The output depends only on (seed, shots), never on threads; the same seed gives the same shots (shots
        [0, n) of the seed's stream; ``sample(n)`` is a prefix of ``sample(m)`` for n <= m).

        ``postselect``: as xtim's ``TwirlSampler.sample``, a ``dtype=bool`` mask of length D, or distinct detector
        indices (a non-bool array is read as indices: duplicate or out-of-range indices are refused, so pass a 0/1
        mask as bool; empty: no detector flagged). Then ``shots`` are attempted shots and the call returns
        ``(dets, status, counts)``: only the shots whose flagged detectors are all 0, and ``counts = {attempts,
        passed, discarded, early}``. The survivors have the law of ``sample(shots, seed=seed)`` followed by a filter
        and depend only on (seed, shots), never on ``threads``; they are not its bytes (an empty selection rejects
        nothing and gives ``sample()``'s bytes). ``early`` counts the rejected shots the span test rejected before
        their gauge-qubit work.

        ``fired=True``: ``status.fired = (offsets int64[shots + 1], ids int32[total])``, each returned shot's fired
        line ids (CSR, ascending = file order; with ``postselect`` the survivors' lists). dets, ``status.obs`` and
        ``status.cspace`` are the same bytes as with ``fired=False``.

        ``status=False``: ``(dets, obs)`` (with ``postselect``: ``(dets, obs, counts)``), obs uint8[shots,
        ceil(O/8)] = ``status.obs``; no status, so no C. dets and obs are the default call's bytes for the same
        arguments. On a GPU sampler the device builds no C key and the host resolves no C id (the default call's
        cost on files where most shots carry a C new to the device's table, e.g. the Bravyi-Haah files); on the CPU
        the status is computed and dropped (same cost as the default). Not with ``fired=True`` (the fired lists
        belong to the status: ValueError)."""
        shots, seed, threads = _count(shots, "shots", 0), _seed(seed), _count(threads, "threads", 1)
        if not isinstance(fired, bool):
            raise TypeError(f"fired must be a bool, got {type(fired).__name__}")
        if not isinstance(status, bool):
            raise TypeError(f"status must be a bool, got {type(status).__name__}")
        if not status:
            if fired:
                raise ValueError("sample(status=False) returns no status, which fired=True's line lists belong to: "
                                 "pass status=True (the default) with fired=True")
            if self._gpu is not None:
                return self._sample_gpu(shots, seed, postselect, False, status=False)
            if postselect is None:
                dets, st = self._sample(shots, seed, threads, 0)
                return dets, st.obs
            dets, obs, _, counts = self._s.sample_ps(shots, seed, threads, 0, postselect)
            return dets, obs, counts
        if self._gpu is not None:
            return self._sample_gpu(shots, seed, postselect, fired)
        if postselect is None:
            if not fired:
                return self._sample(shots, seed, threads, 0)
            dets, obs, cs, off, ids = self._s.sample(shots, seed, threads, 0, True)
            return dets, XdemStatus(obs, cs, None, self, self._s.num_cspaces(), fired=(off, ids), _dets=dets)
        if not fired:
            dets, obs, cs, counts = self._s.sample_ps(shots, seed, threads, 0, postselect)
            return dets, XdemStatus(obs, cs, None, self, self._s.num_cspaces()), counts
        dets, obs, cs, counts, off, ids = self._s.sample_ps(shots, seed, threads, 0, postselect, True)
        return dets, XdemStatus(obs, cs, None, self, self._s.num_cspaces(), fired=(off, ids), _dets=dets), counts

    def _sample_gpu(self, shots: int, seed: int, postselect, fired: bool, status: bool = True):
        """``sample`` on a GPU sampler (arguments validated): the one place the GPU route is chosen and refused.
        ``status=False``: ``(dets, obs)`` / ``(dets, obs, counts)``, no C key or id computed."""
        if fired:
            _gpu_needs_cpu("sample(fired=True)")
        kw = {} if status else {"status": False}
        if postselect is None:
            dets, obs, cs = self._gpu.sample(shots, seed, **kw)
            if not status:
                return dets, obs
            return dets, XdemStatus(obs, cs, None, self, self._s.num_cspaces())
        # the CPU binding's argument rules and refusal texts (postselect_flags), on no shot
        self._s.sample_ps(0, seed, 1, 0, postselect)
        a = np.asarray(postselect)
        flags = np.zeros(self.num_detectors, np.uint8)
        if a.size:
            if a.dtype == np.bool_:
                flags[a] = 1
            else:
                flags[a.astype(np.int64)] = 1
        if not flags.any():   # no detector flagged: every shot passes, sample()'s bytes (as on the CPU)
            dets, obs, cs = self._gpu.sample(shots, seed, **kw)
            counts = {"attempts": shots, "passed": shots, "discarded": 0, "early": 0}
        else:
            dets, obs, cs, counts = self._gpu.sample_ps(shots, seed, flags, **kw)
        if not status:
            return dets, obs, counts
        return dets, XdemStatus(obs, cs, None, self, self._s.num_cspaces()), counts

    def _own(self, status, who: str) -> int:
        if not isinstance(status, XdemStatus):
            raise TypeError(f"{who} expects the XdemStatus returned by sample(), got {type(status).__name__}")
        if status._owner is not self:
            raise ValueError(f"{who}: this XdemStatus was produced by a different XdemSampler (its cspace ids index "
                             f"that sampler's table)")
        return len(status)

    def _hypotheses(self, status: XdemStatus, lines, who: str):
        """(dets, offsets, ids) of a fired status and a decoder's hypotheses."""
        if status.fired is None or status._dets is None:
            raise ValueError(f"{who} needs status.fired and the call's detector rows: sample with fired=True")
        off, ids = _lines_csr(lines, len(status), self.num_lines, who)
        return status._dets, off, ids

    def score(self, status: XdemStatus, flips=None, *, lines=None, correction: "str | None" = None,
              bit_packed: bool = False) -> np.ndarray:
        """float64[shots], per shot.

        ``score(status, flips)``: pass = 2^-dim C if (v XOR flips) in C else 0. ``flips``: the decoder's flip bits,
        bool / 0-1 integers of shape (shots, O), or with ``bit_packed=True`` uint8[shots, ceil(O/8)] (stim b8
        packing; padding bits ignored). A shape mismatch raises ValueError. ``correction`` applies to ``lines=`` only
        (None, the default, means "r2b" there; any value with flips is a TypeError).

        ``score(status, lines=L, correction=...)``: the decoder's hypothesis per shot as xdem line ids (any order;
        repeats cancel), a list of int arrays (one per shot; empty = apply nothing) or a CSR pair ``(offsets, ids)``
        given as a tuple (offsets: 1-d integer, length shots + 1, non-decreasing from 0; ids: 1-d integer, length
        offsets[-1]; any other tuple is a ValueError: per-shot lists go in a list). pass = Pr(o = c | d, the fired
        lines, the correction) with ``correction`` "r2b" (the default, None; docs/xdem.md §4; refused with
        ``XtimXdemError`` when ``r2b_available[0]`` is False), "optimal" (opt-in: the reference, best-possible
        correction, c = 0; slower) or "none" (law-aware flips); any other value is a ValueError. Needs ``status.fired`` (``sample(...,
        fired=True)``; the status keeps a reference to that call's detector rows). A line id outside [0, num_lines), a
        bare id in place of a shot's list or a list of the wrong length raises ValueError / TypeError."""
        n = self._own(status, "score")
        if lines is not None and self._gpu is not None:
            _gpu_needs_cpu("score(lines=)")
        if lines is not None:
            correction = _correction_name(DEFAULT_CORRECTION if correction is None else correction, "score")
            if flips is not None:
                raise TypeError("score: pass flips or lines=, not both")
            if bit_packed:
                raise TypeError("score: bit_packed applies to flips, not lines=")
            dets, off, ids = self._hypotheses(status, lines, "score(lines=)")
            fo, fi = status.fired
            return self._s.score_lines(dets, fo, fi, off, ids, correction)
        if flips is None:
            raise TypeError("score: missing flips (or lines=)")
        if correction is not None:
            raise TypeError(f"score: correction={correction!r} applies to lines=, not flips")
        f = _flips_matrix(flips, n, self.num_observables, bit_packed, "XdemSampler.score")
        return self._s.score(status.obs, status.cspace, f)

    def correction(self, status: XdemStatus, *, lines, correction: str = DEFAULT_CORRECTION) -> XdemCorrection:
        """The correction of ``score(status, lines=L, correction=...)`` per shot, as an ``XdemCorrection`` (docs/xdem.md §5):
        "r2b" (the default; refused with ``XtimXdemError`` when ``r2b_available[0]`` is False) or "optimal" (opt-in,
        the reference correction; slower); "none"
        has nothing to export (ValueError). Needs ``status.fired`` as ``score(lines=)``; ``lines`` as there. An
        export counts no ``lines_*`` counter."""
        n = self._own(status, "correction")
        if self._gpu is not None:
            _gpu_needs_cpu("correction()")
        correction = _correction_name(correction, "correction")
        if correction == "none":
            raise ValueError('correction: nothing to export for correction="none" (use "optimal" or "r2b")')
        dets, off, ids = self._hypotheses(status, lines, "correction(lines=)")
        d = self._s.correction(dets, off, ids, correction)
        O = self.num_observables
        if correction == "optimal":
            return XdemCorrection("optimal", axes=(d["axes_off"], d["axes"]), pivots=(d["piv_off"], d["piv"]),
                                  y=d["y"].astype(bool).reshape(n, O), frame=d["frame"].astype(bool).reshape(n, O),
                                  lossy=d["lossy"].astype(bool).reshape(n, O))
        return XdemCorrection("r2b", late=(d["late_off"], d["late_axis"], d["late_eta"]),
                              late_tau=(d["tau_off"], d["tau_axis"]), applied=d["applied"].astype(bool),
                              c=d["c"].astype(bool).reshape(n, O))

    def infidelity(self, shots: int, decoder, *, seed: int, threads: int = 1, batch: int = 100_000,
                   postselect=None, lines: bool = False, correction: "str | None" = None) -> InfidelityResult:
        """Sample, decode and score ``shots`` shots in batches; 1 - mean(pass) over the accepted shots.

        ``decoder(dets_bool)`` gets bool[n, D] and returns flips (bool / 0-1 of shape (n, O)) or ``(flips, giveup)``
        with giveup bool[n]; give-up shots are scored with the flips returned and counted in ``giveup_fraction``.
        giveup must be bool or 0/1 integers (anything else is refused like flips).

        ``decoder=None``: the zero decoder (flips all 0; ``lines=True`` is a ValueError). On the CPU this is the
        loop below with a zeros decoder, minus the per-shot unpacking of the detectors it would ignore: the same
        result, bit for bit, as passing ``lambda d: np.zeros((len(d), O), bool)``. On a
        GPU sampler nothing but per-block counters leaves the device (``GpuBackend.estimate``, over the whole run;
        ``batch`` and ``threads`` play no part): the counts of passes by dim C, from which ``mean`` and ``stderr`` are
        computed exactly (rational arithmetic, rounded once to float) with the formulas below, so the result is
        that of the CPU formulas on the same scores up to their float accumulation.
        ``postselect``: a ``dtype=bool`` mask of length D or detector indices (repeats allowed here, unlike
        ``sample()``, which refuses a repeated index); a shot where any of them fires is discarded (not decoded, not
        scored) and ``accepted`` counts the rest. The discarding runs in the sampler: the shots decoded are exactly
        the survivors of ``sample(shots, seed=seed, postselect=postselect)``, in order (whatever ``batch`` is).

        Batch rule: the shots are exactly those of ``sample(shots, seed=seed)`` (with ``postselect``: the survivors of
        ``sample(shots, seed=seed, postselect=...)``), in order. ``batch`` is rounded UP to a multiple of 1024 (the
        stream's block; default 100_000 -> 100_352), and batch i returns shots [i * batch, min((i + 1) * batch,
        shots)) of the stream (with ``postselect``, the survivors among them: a block's survivors depend only on
        (seed, block)). So the result does not depend on ``batch`` or ``threads`` except through the decoder.

        ``lines=True``: the decoder returns its hypotheses as xdem line ids, a list of n int arrays (empty = apply
        nothing), or ``(lines, giveup)`` (a tuple is always this pair; ``lines`` may itself be a CSR pair); each
        batch is sampled with ``fired=True`` and scored as ``score(status, lines=..., correction=correction)``. A
        give-up shot is scored with the list returned and counted in ``giveup_fraction``. ``correction`` applies to
        ``lines=True`` only (None, the default, means "r2b" there; any value without it is a TypeError). "r2b" on a
        file where ``r2b_available[0]`` is False raises ``XtimXdemError`` before any shot is sampled."""
        shots = _count(shots, "shots", 0)
        seed = _seed(seed)
        threads = _count(threads, "threads", 1)
        batch = _count(batch, "batch", 1)
        B = -(-batch // BLOCK_SHOTS) * BLOCK_SHOTS
        D, O = self.num_detectors, self.num_observables
        if postselect is None:
            ps = None
        elif np.asarray(postselect).dtype == np.bool_:   # a bool mask of length D, as sample() takes it
            mask = np.ascontiguousarray(postselect, dtype=bool)
            if mask.shape != (D,):
                raise ValueError(f"postselect: a bool mask must have shape (D,) = ({D},), got {mask.shape}")
            ps = mask if mask.any() else None
        else:
            ps = np.asarray([_count(i, "postselect index", 0) for i in postselect], dtype=np.int64)
            if ps.size and int(ps.max()) >= D:
                raise ValueError(f"postselect: detector index {int(ps.max())} out of range (D = {D})")
            if ps.size == 0:
                ps = None
            else:  # repeated indices allowed here: the sampler takes the bool mask
                mask = np.zeros(D, dtype=bool)
                mask[ps] = True
        if not isinstance(lines, bool):
            raise TypeError(f"lines must be a bool, got {type(lines).__name__}")
        if decoder is None and lines:
            raise ValueError("infidelity: decoder=None is the zero decoder on flips; it takes no lines=True")
        if lines and self._gpu is not None:
            _gpu_needs_cpu("infidelity(lines=True)")
        if lines:
            correction = _correction_name(DEFAULT_CORRECTION if correction is None else correction, "infidelity")
            ok, reason = self.r2b_available
            if correction == "r2b" and not ok:   # refused before any sampling (the same text as score(lines=))
                from .errors import XtimXdemError
                raise XtimXdemError(f'score(lines=, correction="r2b"): the file is not in canonical single-stage form '
                                    f'at t_c: {reason}; pass correction="optimal" or "none"')
        elif correction is not None:
            raise TypeError(f"infidelity: correction={correction!r} applies to lines=True, not flips")
        zero = decoder is None
        if zero and self._gpu is not None:
            _, accepted, hist = self._gpu.estimate(shots, seed, None if ps is None else mask.astype(np.uint8))
            return _zero_decoder_result(shots, accepted, hist)
        accepted = 0
        giveups = 0
        q1 = 0.0  # sum of (1 - pass): failure mass, accumulated directly so a small infidelity keeps its digits
        q2 = 0.0
        if self._gpu is not None:
            gblock = self._gpu._block    # the GPU stream's block (512), which divides B (a multiple of 1024)
            gflags = None if ps is None else mask.astype(np.uint8)
        for start in range(0, shots, B):
            n = min(B, shots - start)
            fo = fi = None
            if self._gpu is not None and ps is None:
                dets_p, obs, cs = self._gpu.sample(n, seed, first_block=start // gblock)
            elif self._gpu is not None:
                dets_p, obs, cs, _ = self._gpu.sample_ps(n, seed, gflags, first_block=start // gblock)
            elif ps is None:
                if lines:
                    dets_p, obs, cs, fo, fi = self._s.sample(n, seed, threads, start // BLOCK_SHOTS, True)
                else:
                    dets_p, status = self._sample(n, seed, threads, start // BLOCK_SHOTS)
                    obs, cs = status.obs, status.cspace
            elif lines:
                dets_p, obs, cs, _, fo, fi = self._s.sample_ps(n, seed, threads, start // BLOCK_SHOTS, mask, True)
            else:
                dets_p, obs, cs, _ = self._s.sample_ps(n, seed, threads, start // BLOCK_SHOTS, mask)
            if zero:   # the zero decoder: a zeros decoder's scores, without unpacking the detectors it ignores
                m = int(dets_p.shape[0])
                if m:
                    p = self._s.score(np.ascontiguousarray(obs), np.ascontiguousarray(cs), np.zeros((m, O), np.uint8))
                    q = 1.0 - p
                    accepted += m
                    q1 += float(q.sum())
                    q2 += float(np.dot(q, q))
                continue
            dets = _unpack(dets_p, D)
            m = int(dets.shape[0])
            if m == 0:
                continue
            out = decoder(dets)
            if isinstance(out, tuple):
                if len(out) != 2:
                    raise ValueError("decoder must return " + ("a list of line-id arrays" if lines else "flips") +
                                     " or (" + ("lines" if lines else "flips") + ", giveup)")
                flips, giveup = out
                g = _flips01(giveup, "XdemSampler.infidelity decoder giveup")
                if g.shape != (m,):
                    raise ValueError(f"decoder giveup must have shape ({m},), got {g.shape}")
                giveups += int(np.count_nonzero(g))
            else:
                flips = out
            if lines:
                off, ids = _lines_csr(flips, m, self.num_lines, "XdemSampler.infidelity decoder output")
                p = self._s.score_lines(np.ascontiguousarray(dets_p), fo, fi, off, ids, correction)
            else:
                f = _flips_matrix(flips, m, O, False, "XdemSampler.infidelity decoder output")
                p = self._s.score(np.ascontiguousarray(obs), np.ascontiguousarray(cs), f)
            q = 1.0 - p
            accepted += m
            q1 += float(q.sum())
            q2 += float(np.dot(q, q))
        if accepted == 0:
            return InfidelityResult(math.nan, math.nan, shots, 0, 0.0)
        mq = q1 / accepted
        if accepted > 1:
            var = max(0.0, (q2 - accepted * mq * mq) / (accepted - 1))
            se = math.sqrt(var / accepted)
        else:
            se = math.nan
        return InfidelityResult(mq, se, shots, accepted, giveups / accepted)

    def counters(self) -> dict:
        """The C++ sampler's counters: shots, ext_shots, coin_shots, cache_*, num_cspaces, thread_slots, fast_path;
        for score(lines=): lines_shots, lines_applied, lines_withheld, lines_coin_after, lines_lossy, lines_outside;
        the optimal correction's memo: kmemo_entries, kmemo_clears, kmemo_law_clears, kfast_laws.

        On a GPU sampler the same keys, with ``shots`` = the attempted shots drawn on the device (sample, sample with
        postselect, infidelity), ``ext_shots`` and ``coin_shots`` None (the device does not count them; the other
        keys are the CPU object's, which on a GPU sampler holds the table of C and scores), plus ``device`` (its
        name), ``gpu_batches`` (submissions), ``gpu_relaunches`` (batches rerun after a pair overflow) and
        ``gpu_pair_cap`` (extended records per block of the GPU walk, 512 shots, now)."""
        c = dict(self._s.counters())
        if self._gpu is not None:
            g = self._gpu.counters()
            c.update(shots=g["shots"], ext_shots=None, coin_shots=None, device=self._device,
                     gpu_batches=g["batches"], gpu_relaunches=g["relaunches"], gpu_pair_cap=g["pair_cap"])
        return c


def _zero_decoder_result(shots: int, accepted: int, hist: np.ndarray) -> InfidelityResult:
    """infidelity()'s result for ``accepted`` shots of which ``hist[k]`` pass with 2^-k and the rest with 0: the
    loop's formulas (q = 1 - pass; mean = sum q / n; var = (sum q^2 - n mean^2) / (n - 1), at least 0; stderr =
    sqrt(var / n)) on the exact first and second moments, in rational arithmetic, rounded once to float."""
    if accepted == 0:
        return InfidelityResult(math.nan, math.nan, shots, 0, 0.0)
    from fractions import Fraction
    fails = accepted - int(hist.sum())
    q1 = Fraction(fails)
    q2 = Fraction(fails)
    for k, c in enumerate(hist.tolist()):
        if c:
            q = 1 - Fraction(1, 1 << k)
            q1 += c * q
            q2 += c * q * q
    mq = q1 / accepted
    se = math.sqrt(float((q2 - accepted * mq * mq) / (accepted - 1) / accepted)) if accepted > 1 else math.nan
    return InfidelityResult(float(mq), se, shots, accepted, 0.0)


def score_plain(obs, flips, *, bit_packed: bool = False) -> np.ndarray:
    """float64[shots]: 1 if a shot's observables equal its flips, else 0 — the plain XOR score, for any bit sampler
    (e.g. ``compile_detector_sampler(seed=...).sample(n, separate_observables=True, bit_packed=True)``).

    ``obs``: uint8[shots, ceil(O/8)] (stim b8 packing). ``flips``: bool / 0-1 of shape (shots, O); with
    ``bit_packed=True`` uint8[shots, ceil(O/8)], compared over all 8*ceil(O/8) bits (padding bits must be 0 in both,
    as xtim's and stim's samplers write them). Unpacked flips define O (their width): obs bits at or beyond O are
    ignored. A shape mismatch raises ValueError."""
    o = np.asarray(obs)
    if o.dtype != np.uint8 or o.ndim != 2:
        raise ValueError(f"score_plain: obs must be bit-packed uint8[shots, ceil(O/8)], got {o.dtype} {o.shape}")
    if bit_packed:
        f = np.asarray(flips)
        if f.dtype != np.uint8 or f.shape != o.shape:
            raise ValueError(f"score_plain: bit_packed flips must be uint8 with obs's shape {o.shape}, got "
                             f"{f.dtype} {f.shape}")
        f = _unpack(f, 8 * o.shape[1]).view(np.uint8)
    else:
        f = _flips01(flips, "score_plain")
        if f.ndim != 2 or f.shape[0] != o.shape[0]:
            raise ValueError(f"score_plain: flips must have shape (shots, O) with shots = {o.shape[0]}, got {f.shape}")
    return _xtim.score_plain(np.ascontiguousarray(o), f)


def canonical_report(circuit) -> "tuple[bool, str | None]":
    """``(True, None)`` if ``circuit`` (an ``xtim.Circuit`` or its text) is in canonical single-stage form, else
    ``(False, reason)``: the exact check and message of ``detector_error_model_text(format="xdem",
    require_canonical=True)`` (which checks it when the output is extended), without exporting (a parse only). With
    t_c = right after the instruction that measures the last detector record: (C1') every observable reads at least
    one record measured after t_c (earlier records are a classical frame), and (C2') no noise acts between t_c and
    the last non-Clifford gate after it (a noise channel with a nonzero probability, or a noisy measurement such as
    ``MX(p)``; noise after that gate is a classical readout flip). Gates, non-Clifford readout dressing included,
    are allowed. A parse error raises ``xtim.XtimParseError``."""
    from .errors import XtimParseError
    text = getattr(circuit, "_text", circuit)
    if not isinstance(text, str):
        raise TypeError(f"canonical_report expects an xtim.Circuit or circuit text (str), got {type(circuit).__name__}")
    r = _xtim.canonical_report(text)
    if r["errors"]:
        raise XtimParseError(r["errors"])
    return (True, None) if not r["reason"] else (False, r["reason"])
