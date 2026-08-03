"""Stored reference decompositions (.ref) and the invisible reference cache.

The `Reference` class wraps the existing `.ref` text artifact (cpp/include/qeccore/
ref_io.hpp format) — load/save/inspect, format unchanged.

The cache (resolve_reference) is the M1 "invisible reference" mechanism:

  * key = SHA-256(circuit text + ref-format version + engine-binary hash);
  * cache dir = `xtim.cache_dir` (default `.xtim_cache/` under the cwd);
  * miss  -> run the FULL ref_compile gate battery via the bindings, store the .ref
             (atomically: tmp + rename) plus a `.sha256` integrity companion;
  * hit   -> integrity check + the cheap verify (load + n-match + noiseless
             invariants), then load;
  * ANY cache failure (tamper, unreadable, verify failure, unwritable dir) is
    SILENT-CORRECT: a `XtimCacheWarning` is emitted and the caller falls back to the
    engine's deduced bare state (resolve_reference returns None) or, when a battery
    compile already succeeded but only the STORE failed, the in-memory reference.
"""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from pathlib import Path

from . import _xtim
from .diagnose import _MEAS_PREFIXES, _NOISE_PREFIXES
from .errors import XtimCacheWarning, XtimReferenceError


_ARG_RE = re.compile(r"\(([^)]*)\)")
_HEAD_RE = re.compile(r"\s*([A-Za-z0-9_]+)")


def _instr_head(line: str) -> str:
    """Instruction name the way the stim tokenizer sees it: leading word of
    [A-Za-z0-9_], UPPERCASED (the parser uppercases names), with any [tag] suffix
    and the arg-paren both outside the name. Robust to lowercase gate names,
    tags, and tab separators, all of which the parser accepts."""
    m = _HEAD_RE.match(line)
    return m.group(1).upper() if m else ""


def _mask_noise_values(text: str) -> str:
    """Canonicalize NOISE VALUES (not noise structure) for the cache key.

    Every noise-channel and measurement probability argument is replaced by a
    placeholder ('*' — or '0' when the value is exactly zero, since the pipeline may
    legitimately prune a p=0 channel). The lines themselves — their kind, placement
    and targets — are preserved verbatim.

    This is deliberately WEAKER than diagnose's `_strip_noise` (which deletes noise
    lines outright). The reference state is provably independent of the noise
    *values* (nothing on the reference path reads `probs`), but it is NOT invariant
    under removing the lines: normalization (`eliminate_hadamards`) makes rewrite
    decisions around a noise site — commuting an H past an X_ERROR transforms the
    channel — so the stripped circuit can normalize to a different terminal frame
    than the noisy one (measured: |overlap| = 1/√2 on `H;CX;X_ERROR;M`). Masking
    values keeps every noise-scaled variant of one protocol (`Task(p=...)` sweeps,
    `scale_noise` rewrites) on ONE cache entry while distinguishing any structural
    difference, including noise present vs absent."""
    out = []
    for line in text.splitlines():
        head = _instr_head(line)
        if head in _NOISE_PREFIXES or head in _MEAS_PREFIXES:
            def _mask(m: "re.Match[str]") -> str:
                try:
                    vals = [float(v) for v in m.group(1).split(",")]
                except ValueError:
                    return m.group(0)          # not numeric args: leave untouched
                return "(" + ",".join("0" if v == 0.0 else "*" for v in vals) + ")"
            line = _ARG_RE.sub(_mask, line, count=1)
        out.append(line)
    return "\n".join(out) + "\n"


def _engine_binary_hash() -> str:
    """SHA-256 of the compiled `_xtim` extension binary itself.

    This is the ROBUST guard against serving a reference compiled by a DIFFERENT engine: the
    deferred-signature key (below) auto-invalidates on changes to the deferred *form*, but a
    reference-COMPILE-algorithm change (e.g. a bare-state rank-reduction tweak)
    leaves the deferred form identical, so without this it would rely on someone remembering to
    bump ENGINE_VERSION by hand. Hashing the binary makes ANY engine change auto-invalidate stale
    `.ref` caches — no manual version bump can be forgotten. (ENGINE_VERSION is kept below as a
    human-readable label.) Computed ONCE at import; falls back to the manual version if `_xtim`
    has no file (frozen/builtin)."""
    try:
        return hashlib.sha256(Path(_xtim.__file__).read_bytes()).hexdigest()
    except Exception:
        return "no-binary:" + str(_xtim.ENGINE_VERSION)


_ENGINE_BINARY_HASH = _engine_binary_hash()


def _effective_cache_dir() -> Path:
    """`xtim.cache_dir` if set, else `.xtim_cache/` under the cwd."""
    import xtim

    configured = getattr(xtim, "cache_dir", None)
    if not configured:
        return Path.cwd() / ".xtim_cache"
    try:
        return Path(configured)
    except TypeError:
        # Honour the warn+fallback contract for a pathological cache_dir (e.g. an int)
        # rather than leaking a raw TypeError up through compile_*/reference_info.
        warnings.warn(
            f"xtim.cache_dir is not a valid path ({configured!r}); using the default "
            ".xtim_cache/ under the cwd",
            XtimCacheWarning, stacklevel=2)
        return Path.cwd() / ".xtim_cache"


def cache_key(circuit_text: str) -> str:
    """SHA-256 over the noise-VALUE-masked circuit text + ref-format version + engine-binary hash.

    The key separates the two — and only two — things a stored reference depends on:

      * WHICH CIRCUIT  -> the circuit text with noise VALUES masked
        (`_mask_noise_values`). The reference is provably independent of the noise
        probabilities (nothing on the reference path reads them), so every
        noise-scaled variant of one protocol — a `Task(p=...)` sweep, a
        `scale_noise` rewrite — shares ONE cache entry: a p-sweep pays the
        reference compile once, not once per point. Noise STRUCTURE (which lines,
        where, on which targets) stays in the key: normalization is sensitive to
        noise placement (see `_mask_noise_values`), so noisy and noise-stripped
        circuits are correctly distinct.
      * WHICH ENGINE   -> the SHA-256 of the compiled _xtim binary (`_ENGINE_BINARY_HASH`).

    A reference can become stale for exactly two reasons, and each axis covers one: the circuit
    changed (different value-masked text), or the engine that compiles/desugars/defers it changed
    (different binary). Because EVERY engine change recompiles the `.so`, the engine-binary hash
    auto-invalidates stale `.ref` entries by construction — including desugar/deferral changes (the
    original stale-ref incident) AND reference-COMPILE-algorithm changes (a bare-state rank fix). No
    manual ENGINE_VERSION bump can be forgotten; the version string below is kept only as a human-
    readable label. (A previous design hashed an engine-computed deferred-circuit *signature* as
    the circuit identity; the engine-binary hash subsumes its correctness role. Until 2.1 the key
    hashed the RAW text verbatim — correct, but it made every p-point of a sweep a separate
    expensive miss; the value-masked text is the exact invariant the entry depends on.)"""
    payload = (
        "NOISEMASKED2\x00" + _mask_noise_values(circuit_text)
        + "\x00ref-format:" + str(_xtim.REF_FORMAT_VERSION)
        + "\x00engine:" + _xtim.ENGINE_VERSION
        + "\x00engine-bin:" + _ENGINE_BINARY_HASH
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Reference:
    """A stored reference decomposition (the existing .ref text artifact)."""

    def __init__(self, text: str):
        if not isinstance(text, str):
            extra = (" — build one with c.compile_reference()"
                     if type(text).__name__ == "Circuit" else "")
            raise XtimReferenceError(
                f"Reference() takes stored .ref text (str), got "
                f"{type(text).__name__}{extra}")
        info = _xtim.ref_info(text)
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
        return f"xtim.Reference(chi={self.chi}, n={self.n}, source={self.source!r})"


def compile_reference(circuit_text: str, source: str = "xtim-circuit") -> Reference:
    """Run the full ref_compile gate battery; raise XtimReferenceError on failure."""
    r = _xtim.ref_compile_text(circuit_text, source=source)
    if not r["ok"]:
        gates = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in r["gates"].items())
        raise XtimReferenceError(
            f"reference compile failed: {r['error']}" + (f" (gates: {gates})" if gates else "")
        )
    return Reference(r["ref_text"])


def _deferred_sig(circuit_text: str) -> str:
    """SHA-256 of the engine's structural deferred-circuit signature (noise values
    excluded). Two texts share it iff they normalize+defer to the SAME frame — the
    exact invariant a cached reference depends on. Stored beside each entry and
    re-derived from the caller's text on every hit, this catches the whole
    frame-divergence class (the `H;CX;X_ERROR;M` stripped-vs-noisy bug) at
    normalize cost, even for circuits where the cheap-verify invariants are weak
    (detector-bearing / expectation-bearing circuits skip the exact overlap gate)."""
    return hashlib.sha256(_xtim.deferred_signature(circuit_text).encode()).hexdigest()


def _store(ref: Reference, ref_path: Path, defsig: str) -> None:
    """Atomic store: writer-unique tmp + rename, plus the .sha256 companion.

    The tmp names carry the pid so concurrent processes racing the same cache
    entry never truncate each other's tmp mid-rename (deterministic content
    makes last-writer-wins safe)."""
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    import threading
    wid = f"{os.getpid()}.{threading.get_ident()}"   # pid+tid: thread-safe tmp names
    tmp = ref_path.parent / f"{ref_path.name}.tmp.{wid}"
    tmp.write_text(ref.text)
    os.replace(tmp, ref_path)
    digest = hashlib.sha256(ref.text.encode()).hexdigest()
    sha_path = ref_path.with_suffix(ref_path.suffix + ".sha256")
    sha_tmp = sha_path.parent / f"{sha_path.name}.tmp.{wid}"
    sha_tmp.write_text(digest + "\n")
    os.replace(sha_tmp, sha_path)
    sig_path = ref_path.with_suffix(ref_path.suffix + ".defsig")
    sig_tmp = sig_path.parent / f"{sig_path.name}.tmp.{wid}"
    sig_tmp.write_text(defsig + "\n")
    os.replace(sig_tmp, sig_path)


def _load_verified(circuit_text: str, ref_path: Path) -> Reference:
    """Cache-hit path: integrity check + cheap verify. Raises on any problem."""
    text = ref_path.read_text()
    sha_path = ref_path.with_suffix(ref_path.suffix + ".sha256")
    expected = sha_path.read_text().strip()
    actual = hashlib.sha256(text.encode()).hexdigest()
    if actual != expected:
        raise XtimReferenceError(f"integrity mismatch for {ref_path} (tampered or corrupt)")
    # Frame-signature gate: the entry must have been compiled for a circuit that
    # normalizes+defers to the SAME frame as the caller's. This is the complete net
    # for the frame-divergence class; the invariants below are weak for circuits
    # with detectors/expectations (they skip the exact overlap gate).
    sig_path = ref_path.with_suffix(ref_path.suffix + ".defsig")
    stored_sig = sig_path.read_text().strip()
    if stored_sig != _deferred_sig(circuit_text):
        raise XtimReferenceError(
            f"deferred-frame signature mismatch for {ref_path}: the cached reference "
            "was compiled for a circuit that normalizes to a different terminal frame")
    v = _xtim.ref_verify_cheap(circuit_text, text)
    if not v["ok"]:
        raise XtimReferenceError(f"cheap verify failed for {ref_path}: {v['error']}")
    return Reference(text)


def resolve_reference(circuit_text: str, source: str = "xtim-circuit") -> Reference | None:
    """The invisible-reference resolution used by Circuit.compile_*.

    Returns the Reference to supply to the engine, or None for the engine's own
    deduced-bare-state path (the silent-correct fallback)."""
    key = cache_key(circuit_text)
    try:
        ref_path = _effective_cache_dir() / f"{key}.ref"
    except Exception as e:  # pathological cache_dir values
        warnings.warn(f"xtim cache dir unusable ({e}); using the deduced bare state",
                      XtimCacheWarning, stacklevel=2)
        ref_path = None

    # hit: integrity + cheap verify. ANY hit failure (tamper, unreadable, verify
    # failure) is the contract's silent-correct fallback: warn + the deduced path.
    if ref_path is not None and ref_path.exists():
        try:
            return _load_verified(circuit_text, ref_path)
        except Exception as e:
            # ANY hit failure (tamper, unreadable, verify-fail) is the contract's
            # silent-correct fallback: warn + the deduced bare state. We deliberately do
            # NOT recompile-over the bad entry — that keeps an integrity failure visible
            # and preserves the byte-parity-with-deduced guarantee on this path.
            warnings.warn(
                f"xtim reference cache hit failed ({e}); using the deduced bare state "
                f"(delete {ref_path} to recompile)",
                XtimCacheWarning, stacklevel=2)
            return None

    # miss: full gate battery, then store. Compile from the caller's FULL text — NOT a
    # noise-stripped canonicalization: normalization (eliminate_hadamards) is sensitive
    # to noise PLACEMENT, so the stripped circuit can land in a different terminal
    # frame than every noisy variant (measured |overlap| = 1/sqrt(2) on H;CX;X_ERROR;M).
    # All variants that share this key differ only in noise VALUES, which nothing on
    # the reference path reads, so whichever p-point arrives first stores the entry
    # every other point verifies and reuses.
    try:
        ref = compile_reference(circuit_text, source=source)
    except XtimReferenceError as e:
        # An out-of-class / reject failure is NOT a benign cache miss: the circuit will
        # raise XtimRejectError when it is actually run, so the "sampling stays exact —
        # no action needed" reassurance would be misleading (e.g. a `CH`, or a Hadamard
        # on a magic wire). Stay silent here and let that real reject be the single loud
        # signal. The benign warning still fires for genuine fallbacks (a higher-χ shape
        # the ref compiler doesn't yet build), where sampling really does stay exact.
        msg = str(e).lower()
        if "out-of-class" not in msg and "rejected" not in msg:
            warnings.warn(
                f"xtim reference compile unavailable ({e}); using the deduced bare state "
                "(expected for some circuits; sampling stays exact — no action needed)",
                XtimCacheWarning, stacklevel=2)
        return None

    if ref_path is not None:
        try:
            _store(ref, ref_path, _deferred_sig(circuit_text))
        except (OSError, ValueError) as e:
            # OSError: unwritable dir / file-as-dir / etc. ValueError: a pathological
            # path the filesystem rejects (e.g. an embedded NUL byte in cache_dir).
            # Both are the silent-correct fallback — warn, proceed without caching.
            warnings.warn(
                f"xtim reference cache store failed ({e}); proceeding without caching",
                XtimCacheWarning, stacklevel=2)
    return ref
