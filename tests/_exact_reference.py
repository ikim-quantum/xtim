"""Stored reference decompositions (.ref) and the invisible reference of the EXACT engine.

TEST API since exact-out T5 (was the public ``xtim.reference`` / ``xtim.Reference`` / ``xtim.cache_dir`` /
``Circuit.compile_reference``): only the exact engine (tests/_exact_engine.py) consumes a reference, and the
engine bindings (``ref_compile_text``, ``ref_verify_cheap``, ``ref_info``, ``deferred_signature``) live in
``xtim._xtim_testapi``. ``XtimReferenceError`` / ``XtimCacheWarning`` are defined here (they left ``xtim.errors``).

The `Reference` class wraps the existing `.ref` text artifact (cpp/include/qeccore/
ref_io.hpp format) — load/save/inspect, format unchanged.

The invisible reference (resolve_reference), computed IN MEMORY (bloat-cut C2, owner rule 2026-10-30: no disk
caches; the former `.xtim_cache/` reference cache, its `cache_dir`, integrity companions and cheap-verify hit path
are deleted):

  * the FULL ref_compile gate battery runs via the bindings, once per circuit text per process (`_MEMO`, keyed on the
    exact text: one process means one engine binary, so a memoised reference is the one this binary compiled);
  * the noiseless-invariants battery is WORK-SIZED (ref_compile.cpp noiseless_invariants): the full 500 shots
    wherever noiseless shots replay a TreePlan, otherwise clamp(program-compile work / shot work, 1, 500);
  * whenever the battery runs FEWER shots than its ceiling, one `XtimVerificationWarning` per compile states the
    shots run vs the ceiling, W_compile and w1, and the miss bound (1-q)^N (`_warn_reduced_battery`); silent when the
    full battery runs (w1 = 0); a memo hit runs no battery and warns nothing;
  * a compile that is unavailable (a higher-chi shape the ref compiler does not build) emits a `XtimCacheWarning`
    and the caller falls back to the engine's deduced bare state (resolve_reference returns None).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

from xtim import _xtim_testapi as _ta
from xtim.errors import XtimError, XtimInvariantError, XtimVerificationWarning, XtimWarning

# The in-memory reference of each circuit text resolved in this process (bloat-cut C2).
_MEMO: "dict[str, Reference]" = {}


class XtimReferenceError(XtimError):
    """Explicit reference operation failed (compile battery / load / save)."""


class XtimCacheWarning(XtimWarning):
    """The reference compile was unavailable. The run stays correct — the exact engine
    falls back to its deduced bare state."""


class Reference:
    """A stored reference decomposition (the existing .ref text artifact)."""

    def __init__(self, text: str):
        if not isinstance(text, str):
            extra = (" — build one with compile_reference(text)"
                     if type(text).__name__ == "Circuit" else "")
            raise XtimReferenceError(
                f"Reference() takes stored .ref text (str), got "
                f"{type(text).__name__}{extra}")
        info = _ta.ref_info(text)
        if not info["ok"]:
            raise XtimReferenceError(f"invalid .ref text: {info['error']}")
        self._text = text
        self._info = info

    # ── power-user surface ───────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | os.PathLike) -> "Reference":
        """Load a stored ``.ref`` from `path`. Raises :class:`XtimReferenceError` if
        the file is missing/unreadable or its text is not a valid reference."""
        try:
            text = Path(path).read_text()
        except OSError as e:  # missing / unreadable file -> the class's typed error
            raise XtimReferenceError(f"could not read .ref file {path!r}: {e}") from e
        except UnicodeDecodeError:  # binary/corrupt .ref -> same typed error, not a raw codecs traceback
            raise XtimReferenceError(
                f".ref file {path!r} is not UTF-8 text (binary/corrupt)") from None
        return cls(text)

    def save(self, path: str | os.PathLike) -> None:
        """Write this reference's ``.ref`` text to `path` (round-trips with
        :meth:`load`)."""
        Path(path).write_text(self._text)

    # ── inspection ──────────────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        return self._text

    @property
    def chi(self) -> int:
        return self._info["chi"]

    @property
    def n(self) -> int:
        return self._info["n"]

    @property
    def source(self) -> str:
        return self._info["source"]

    @property
    def version(self) -> int:
        """The on-disk `.ref` format version of this stored artifact (v2/v3/v4).

        A v4 reference cannot be read by xtim ≤ 0.8.0; this xtim reads v2/v3/v4."""
        return self._info["version"]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Reference(chi={self.chi}, n={self.n}, source={self.source!r})"


def _caller_frame(stacklevel: int) -> dict:
    """Point a warning at the first frame outside the xtim package (3.12+), else stacklevel."""
    import sys
    if sys.version_info >= (3, 12):
        return {"skip_file_prefixes": (os.path.dirname(os.path.abspath(__file__)) + os.sep,)}
    return {"stacklevel": stacklevel}


def _warn_reduced_battery(r: dict, where: str, stacklevel: int = 3) -> None:
    """One XtimVerificationWarning when the work-sized noiseless-invariants battery ran fewer
    shots than its ceiling (ref_compile.cpp noiseless_invariants). Silent when the full battery
    ran — every circuit whose noiseless shots replay a TreePlan (w1 = 0)."""
    n = int(r.get("invariants_shots", 0))
    ceiling = int(r.get("invariants_ceiling", 0))
    if n >= ceiling:
        return
    warnings.warn(
        f"xtim reference {where}: the noiseless-invariants self-check ran {n} of {ceiling} "
        f"shots (work-sized, verification never costs more than compilation: "
        f"W_compile = {int(r['invariants_compile_work'])}, w1 = "
        f"{int(r['invariants_shot_work'])} work units, clamp(W_compile / w1, 1, {ceiling}) = "
        f"{int(r['invariants_sized_shots'])}). The detector verdict equals the full battery's; "
        f"the |PAULI_EXPECTATION| constancy check saw only these {n} shots: a wobble confined "
        f"to outcome paths of probability q is missed with probability (1-q)^{n} "
        f"(full battery: (1-q)^{ceiling}).",
        XtimVerificationWarning, **_caller_frame(stacklevel))


def compile_reference(circuit_text: str, source: str = "xtim-circuit") -> Reference:
    """Run the full ref_compile gate battery; raise XtimReferenceError on failure. Emits one
    XtimVerificationWarning when the work-sized self-check ran below its ceiling."""
    ref, r = _compile_reference_checked(circuit_text, source)
    _warn_reduced_battery(r, "compile")
    return ref


def _compile_reference_checked(circuit_text: str, source: str) -> tuple[Reference, dict]:
    """compile_reference without the warning: the reference and the battery result (the
    cache-miss path stores the reference BEFORE warning, so an "error" warnings filter cannot
    keep a valid reference out of the cache)."""
    # The reference compiler takes no bit values; an IF body would be compiled unconditionally
    # into the reference. Same by-name refusal as the sampling surfaces (xtim.Circuit).
    from xtim.circuit import _refuse_unresolved_if
    _refuse_unresolved_if(circuit_text, "compile_reference")
    r = _ta.ref_compile_text(circuit_text, source=source)
    if not r["ok"]:
        gates = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in r["gates"].items())
        raise XtimReferenceError(
            f"reference compile failed: {r['error']}" + (f" (gates: {gates})" if gates else "")
        )
    return Reference(r["ref_text"]), r


def resolve_reference(circuit_text: str, source: str = "xtim-circuit") -> Reference | None:
    """The invisible-reference resolution of the exact samplers (tests/_exact_engine.py): the
    reference compiled in memory (memoised per circuit text in this process), or None for the
    engine's own deduced-bare-state path (the silent-correct fallback)."""
    ref = _MEMO.get(circuit_text)
    if ref is not None:
        return ref
    # Compile from the caller's FULL text — NOT a noise-stripped canonicalization:
    # normalization (eliminate_hadamards) is sensitive to noise PLACEMENT, so the stripped
    # circuit can land in a different terminal frame than the noisy one (measured
    # |overlap| = 1/sqrt(2) on H;CX;X_ERROR;M).
    try:
        ref, battery = _compile_reference_checked(circuit_text, source)
    except XtimReferenceError as e:
        # An out-of-class / reject failure is NOT a benign fallback: the circuit will raise
        # XtimRejectError when it is actually run, so the "sampling stays exact — no action
        # needed" reassurance would be misleading (e.g. a `CH`, or a Hadamard on a magic
        # wire). Stay silent here and let that real reject be the single loud signal. The
        # benign warning still fires for genuine fallbacks (a higher-χ shape the ref compiler
        # doesn't yet build), where sampling really does stay exact.
        msg = str(e).lower()
        if "not a state" in msg:
            # The compiled state failed its normalisation check. The deduced bare state is the
            # SAME construction, so "sampling stays exact" would be false: refuse loudly.
            raise XtimInvariantError(
                f"reference compile built a non-normalised state ({e}); the deduced bare "
                "state is the same construction, so xtim refuses to sample it") from e
        if "out-of-class" not in msg and "rejected" not in msg:
            warnings.warn(
                f"xtim reference compile unavailable ({e}); using the deduced bare state "
                "(expected for some circuits; sampling stays exact — no action needed)",
                XtimCacheWarning, stacklevel=2)
        return None
    _MEMO[circuit_text] = ref
    # After the memo store: the reference passed every gate it ran; the warning is a note (under an
    # "error" filter it propagates, but the next resolution of the same text is a memo hit).
    _warn_reduced_battery(battery, "compile")
    return ref
