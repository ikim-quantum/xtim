"""xtim.Circuit — the Stim-shaped entry point over the MSP engine.

The workflow mirrors ordinary Stim (spec: docs/superpowers/specs/
2026-06-12-xtim-python-ux-design.md):

    c = xtim.Circuit.from_file("protocol.stim")
    s = c.compile_detector_sampler(seed=7)
    dets, obs, exps = s.sample(10**6, separate_observables=True,
                               return_expectations=True)
    meas = c.compile_sampler(seed=7).sample(10**6)
    dem  = c.detector_error_model()        # a real stim.DetectorErrorModel
    info = c.reference_info()

Engine contract: RAW physics only — measurement bits, detection events, observable
flips (reference-relative, Stim's exact semantics), raw per-shot Pauli expectations,
and the DEM. Every interpretive step (frames, corrections, post-selection) is
user-side analysis.

Determinism: a compiled sampler replays the SAME stream on every `sample(shots)`
call — `seed` pins the run exactly as `run_stim_main <file> <shots> <seed>` does, and
all records of one `sample()` call come from ONE engine run (single RNG stream).

Invisible reference: `compile_*` resolves the bare/reference state automatically via
the reference cache (see xtim/reference.py); any cache trouble degrades, with a
warning, to the engine's own deduced bare state — results stay correct.
"""
from __future__ import annotations

import operator
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np

from . import _xtim
from . import reference as _refmod
from . import twirl as _twirl
from .errors import (
    XtimDemError,
    XtimError,
    XtimParseError,
    XtimPerformanceWarning,
    XtimRejectError,
    XtimStimMissingError,
)
from .reference import Reference

if TYPE_CHECKING:  # pragma: no cover - typing only
    import stim

    from .diagnose import Diagnosis
    from .ler import PostselectedLER

def _dem_export_error(errors) -> "XtimDemError":
    """Build an XtimDemError from DEM-export `errors` (list of (line, msg)). The circuit
    already parsed, so these are DEM-export limitations (e.g. classically-controlled
    feedback), not parse errors — route the feedback case to a correct hint."""
    msg = "; ".join(m for _, m in errors)
    low = msg.lower()
    hint = None
    if "feedback" in low or "not yet handled by dem export" in low:
        hint = (
            "classically-controlled Pauli feedback (CX/CY/CZ rec[-k] q) isn't modeled by "
            "DEM export yet — but the sampler runs it exactly. For a decoder DEM, pull the "
            "feedback out of the circuit and apply the byproduct correction in "
            "post-processing on the records (diagnose() reports the frame), or post-select "
            "instead of decoding."
        )
    return XtimDemError(msg, hint=hint)


@dataclass(frozen=True)
class PostselectFault:
    """One detectable-but-not-Pauli-correctable fault: the detector signature it
    flips/randomizes (no Pauli correction exists, so it cannot be a DEM edge) plus its
    probability. Summed over all faults, the probabilities give the O(p) coherent-error
    floor (a lower bound on the logical infidelity if you decode rather than post-select)."""

    detectors: list[int]   # the syndrome this fault flips/randomizes (no Pauli correction exists)
    probability: float


@dataclass(frozen=True)
class DemWithReject:
    """A DEM paired with the not-Pauli-correctable faults to FLAG. `postselect_faults`
    is the primary output: each is a `PostselectFault` (the detector signature the
    fault flips/randomizes + its probability), letting the user choose a post-select
    policy. `reject_detectors` is the derived convenience union (the detectors any such
    fault touches); both are empty for a Pauli/Clifford circuit (then the DEM is
    byte-identical to `detector_error_model()`)."""

    dem: "stim.DetectorErrorModel"
    reject_detectors: list[int]
    postselect_faults: list["PostselectFault"]

    def __repr__(self) -> str:
        # Compact: the dataclass default embeds the full DEM text + every fault (a huge
        # interactive dump); keep a summary — the full data stays on the attributes.
        return ("DemWithReject(detectors=%d, mechanisms=%d, reject_detectors=%d, "
                "postselect_faults=%d)" % (
                    self.dem.num_detectors, self.dem.num_errors,
                    len(self.reject_detectors), len(self.postselect_faults)))

    def keep_mask(self, dets: np.ndarray) -> np.ndarray:
        """Boolean mask over shots — True where NO reject detector fired (the shots to
        keep / post-select). `dets` is a bool[shots, num_detectors] detector array.
        Decode `dets[keep_mask(dets)]` with `self.dem`; discard the rest.

        This is the BLUNT, conservative policy: it rejects whenever any flagged detector
        fires, so it can over-reject shots where that detector is shared with an ordinary
        Pauli edge — but it is ALWAYS sound, which is why it is the recommended default
        (and on a saturated-reject protocol it is provably minimal). See
        docs/xtim_dem_reject.md for the finer-policy caveats."""
        dets = np.asarray(dets)
        if dets.ndim != 2:
            raise ValueError(
                f"dets must be 2-D bool[shots, num_detectors], got ndim={dets.ndim}")
        num_detectors = self.dem.num_detectors
        if dets.shape[1] != num_detectors:
            raise ValueError(
                f"dets must be 2-D bool[shots, num_detectors] with "
                f"num_detectors={num_detectors}, got shape={tuple(dets.shape)} "
                f"(columns={dets.shape[1]}); pass the full detector array for every "
                f"shot — did you transpose it or concatenate observables?")
        if not self.reject_detectors:
            return np.ones(dets.shape[0], dtype=bool)
        return ~dets[:, self.reject_detectors].any(axis=1)


def _unpack_bits(packed: np.ndarray, num_bits: int) -> np.ndarray:
    """Stim b8 rows (uint8, little-endian bit order) -> bool[shots, num_bits]."""
    shots = packed.shape[0]
    if num_bits == 0:
        return np.zeros((shots, 0), dtype=bool)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :num_bits].astype(bool)


# ── programmatic-construction text helpers (M-D) ─────────────────────────────────
def _format_target(t) -> str:
    """Render one append target to Stim target text.

    Accepts an ``int`` qubit index, a raw target string (passed through —
    ``"rec[-1]"``, ``"!1"``, ``"X0"``), or a ``stim.GateTarget`` (converted to its
    text form). Anything else raises TypeError."""
    if isinstance(t, bool):  # bool is an int subclass; reject to avoid True->"1"
        raise TypeError(f"invalid append target {t!r}: bool is not a qubit index")
    if isinstance(t, int):
        if t < 0:
            raise ValueError(
                f"invalid append target {t!r}: a bare qubit index must be >= 0 "
                f"(use the string 'rec[{t}]' for a measurement record)")
        return str(t)
    if isinstance(t, str):
        return t
    # stim.GateTarget (duck-typed so we never hard-depend on stim here)
    rec = getattr(t, "is_measurement_record_target", False)
    if rec:
        return f"rec[{t.value}]"
    for attr, pauli in (("is_x_target", "X"), ("is_y_target", "Y"),
                        ("is_z_target", "Z")):
        if getattr(t, attr, False):
            inv = "!" if getattr(t, "is_inverted_result_target", False) else ""
            return f"{inv}{pauli}{t.value}"
    if getattr(t, "is_inverted_result_target", False):
        return f"!{t.value}"
    val = getattr(t, "value", None)
    if val is not None:
        return str(val)
    raise TypeError(f"invalid append target {t!r} ({type(t).__name__})")


def _format_instruction(name: str, targets, arg) -> str:
    """Render (name, targets, arg) to a single Stim instruction line."""
    if not isinstance(name, str):
        raise TypeError(
            f"append name must be a gate/instruction string, got {type(name).__name__}")
    if isinstance(targets, (int, str)) or not isinstance(targets, Iterable):
        targets = (targets,)
    tparts = [_format_target(t) for t in targets]

    if arg is None:
        argstr = ""
    else:
        if isinstance(arg, (int, float)):
            args = [arg]
        elif isinstance(arg, Iterable):
            args = list(arg)
        else:
            raise TypeError(
                f"append arg must be a float or iterable of floats, got "
                f"{type(arg).__name__}")
        argstr = "(" + ", ".join(_format_float(a) for a in args) + ")"

    line = name + argstr
    if tparts:
        line += " " + " ".join(tparts)
    return line


def _format_float(x) -> str:
    """Stim renders integral floats without a trailing .0 (e.g. coord/obs index 1)."""
    f = float(x)
    if f == int(f) and abs(f) < 1e16:
        return str(int(f))
    return repr(f)


def _other_text(other) -> str:
    # The concatenated result is re-parsed by the caller (Circuit(...)/_reinit), so a
    # fragment is validated IN CONTEXT — DETECTOR rec[-1] after an M is fine even
    # though the bare line wouldn't parse alone. Only the type is checked here.
    if isinstance(other, Circuit):
        return other._text
    if isinstance(other, str):
        return other
    raise TypeError(
        f"can only add a Circuit or Stim text str, not {type(other).__name__}")


def _concat_text(a: str, b: str) -> str:
    if a == "":
        return b
    if b == "":
        return a
    sep = "" if a.endswith("\n") else "\n"
    return a + sep + b


def _repeat_text(text: str, repeats: int) -> str:
    if not isinstance(repeats, int) or isinstance(repeats, bool):
        raise TypeError(f"can only multiply a Circuit by an int, not "
                        f"{type(repeats).__name__}")
    if repeats < 0:
        raise ValueError("repeat count must be >= 0")
    if repeats == 0 or text.strip() == "":
        return ""
    if repeats == 1:
        return text
    body = "\n".join("    " + ln if ln.strip() else ln
                     for ln in text.rstrip("\n").split("\n"))
    return f"REPEAT {repeats} {{\n{body}\n}}"


def _count_instructions(text: str) -> int:
    """Top-level instruction count: a REPEAT block is one; blank/comment lines and
    the inner lines of a REPEAT body don't count. Mirrors ``len(stim.Circuit)``."""
    count = 0
    depth = 0
    for raw in text.split("\n"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("}"):
            depth -= 1
            continue
        if depth == 0:
            count += 1
        if line.endswith("{"):  # REPEAT N {  (already counted as one above)
            depth += 1
    return count


class Circuit:
    """A parsed extended-dialect (strict Stim superset) circuit."""

    def __init__(self, text: str = "", *, _source: str | None = None):
        if not isinstance(text, str):
            raise TypeError(
                f"circuit text must be a string, got {type(text).__name__}")
        self._text = text
        self._source = _source or "xtim-circuit"
        info = _xtim.parse_info(text)
        if not info["ok"]:
            raise XtimParseError(info["errors"])
        self._info = info
        # invisible-reference memo: (cache_key, cache_dir) -> Reference | None
        self._resolved: dict[tuple[str, str], Reference | None] = {}

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Circuit":
        p = Path(path)
        try:
            text = p.read_text()
        except OSError as e:
            # Match the CLI's clean "cannot open ..." message instead of leaking a raw
            # pathlib traceback — the same actionable surface as every other xtim error.
            raise XtimError("cannot open circuit file %r: %s"
                            % (str(p), e.strerror or e)) from None
        except UnicodeDecodeError:
            # A binary/corrupted file would otherwise leak a raw codecs traceback.
            raise XtimError("%r is not a UTF-8 text circuit (got binary/non-UTF-8 "
                            "bytes) — is the path right?" % str(p)) from None
        return cls(text, _source=p.name)

    # ── programmatic construction (M-D) ──────────────────────────────────────────
    def append(self, name: str, targets=(), arg=None) -> None:
        """Append one instruction, mirroring `stim.Circuit.append`.

        `name` is a gate/instruction string (e.g. ``"H"``, ``"M"``, ``"CX"``, and the
        extended ``"T"``/``"CS"``/``"PAULI_EXPECTATION"``). `targets` is a single
        target or an iterable; each target is an ``int`` qubit index, a raw target
        string (``"rec[-1]"``, ``"!1"``, ``"X0"``), or a ``stim.GateTarget``. `arg`
        is the parens argument(s) — a float or an iterable of floats (noise
        probability, observable/coords indices, …).

        The instruction is rendered to a Stim text line, appended to the circuit
        text, and the whole circuit is re-parsed — an invalid append (unknown gate,
        wrong arity, bad target) raises :class:`XtimParseError` immediately, exactly
        as Stim raises on a bad append, and leaves this Circuit unchanged."""
        line = _format_instruction(name, targets, arg)
        old = self._text
        new = old if old.endswith("\n") or old == "" else old + "\n"
        new = new + line + "\n"
        self._reinit(new)

    def _reinit(self, text: str) -> None:
        """Re-parse `text` and, only if it parses, adopt it as this Circuit's state.

        Any parse failure raises XtimParseError and leaves self untouched (atomic
        mutation — a failed append/`+=` never corrupts the Circuit)."""
        info = _xtim.parse_info(text)
        if not info["ok"]:
            raise XtimParseError(info["errors"])
        self._text = text
        self._info = info
        self._resolved = {}

    def copy(self) -> "Circuit":
        """A new independent Circuit with the same text (mirrors stim.Circuit.copy)."""
        return Circuit(self._text, _source=self._source)

    def to_file(self, file: "str | os.PathLike | object") -> None:
        """Write the circuit text to a path or open text file (mirrors stim's
        ``to_file``). Round-trips with :meth:`from_file`."""
        text = self._text if self._text.endswith("\n") or self._text == "" \
            else self._text + "\n"
        if hasattr(file, "write"):
            file.write(text)
        else:
            Path(file).write_text(text)

    def without_noise(self) -> "Circuit":
        """A copy with all noise removed (mirrors stim's ``without_noise``): noise-channel
        instructions are dropped and a measurement readout-flip probability is stripped
        (``M(0.01)`` -> ``M``). Useful for a quick noiseless reference run."""
        import re
        noise_heads = {
            "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
            "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "PAULI_CHANNEL", "I_ERROR", "II_ERROR",
            "E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR", "HERALDED_ERASE",
            "HERALDED_PAULI_CHANNEL_1",
        }
        out = []
        for line in self._text.splitlines():
            head = line.strip().split("(", 1)[0].split(" ", 1)[0]
            if head in noise_heads:
                continue
            # strip a readout-flip probability from a measurement gate: M(p) -> M
            line = re.sub(r"^(\s*(?:MRX|MRY|MRZ|MR|MXX|MYY|MZZ|MX|MY|MZ|M)\b)\([^)]*\)",
                          r"\1", line)
            out.append(line)
        return Circuit("\n".join(out) + ("\n" if out else ""), _source=self._source)

    # ── operators (M-D) ──────────────────────────────────────────────────────────
    def __add__(self, other: "Circuit | str") -> "Circuit":
        """Concatenate two circuits (or a circuit and raw Stim text), mirroring
        ``stim.Circuit.__add__``. Returns a new Circuit; re-parses the result."""
        return Circuit(_concat_text(self._text, _other_text(other)),
                       _source=self._source)

    def __radd__(self, other: str) -> "Circuit":
        return Circuit(_concat_text(_other_text(other), self._text),
                       _source=self._source)

    def __iadd__(self, other: "Circuit | str") -> "Circuit":
        self._reinit(_concat_text(self._text, _other_text(other)))
        return self

    def __mul__(self, repeats: int) -> "Circuit":
        """Repeat the circuit ``repeats`` times as a ``REPEAT`` block, mirroring
        ``stim.Circuit.__mul__``. ``c * 0`` is the empty circuit; ``c * 1`` is a
        copy of the body (no REPEAT wrapper, as in Stim)."""
        return Circuit(_repeat_text(self._text, repeats), _source=self._source)

    def __rmul__(self, repeats: int) -> "Circuit":
        return self.__mul__(repeats)

    def __imul__(self, repeats: int) -> "Circuit":
        self._reinit(_repeat_text(self._text, repeats))
        return self

    def __len__(self) -> int:
        """The number of top-level instructions (a REPEAT block counts as one),
        mirroring ``len(stim.Circuit)``."""
        return _count_instructions(self._text)

    def __str__(self) -> str:
        return self._text

    def __eq__(self, other) -> bool:
        # Compare on trailing-whitespace-normalized text: a trailing newline is never
        # semantically meaningful (Stim ignores it), and `to_file`/`append` add one, so
        # `c == from_file(to_file(c))` and append-built vs literal circuits must match.
        return (isinstance(other, Circuit)
                and other._text.rstrip() == self._text.rstrip())

    def __hash__(self) -> int:
        return hash(self._text.rstrip())

    # ── inspection ───────────────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        return self._text

    @property
    def num_qubits(self) -> int:
        return self._info["n"]

    @property
    def num_measurements(self) -> int:
        return self._info["num_measurements"]

    @property
    def num_detectors(self) -> int:
        return self._info["num_detectors"]

    @property
    def num_observables(self) -> int:
        return self._info["num_observables"]

    @property
    def num_expectations(self) -> int:
        return self._info["num_expectations"]

    @property
    def expectation_columns(self) -> list[int]:
        """DEM L-column indices of the PAULI_EXPECTATION declarations.

        THE ordering contract (spec M2): the DEM's L-columns are the
        OBSERVABLE_INCLUDE observables first, then one column per PAULI_EXPECTATION
        declaration, each in declaration order — so with O observables and R
        declarations, declaration r is column L(O+r) and this list is
        [O, O+1, ..., O+R-1] (computed from the parse counts, never hardcoded).
        It indexes directly into a DEM decoder's flip output:

            flips = matching.decode_batch(dets)
            corrected = np.where(flips[:, c.expectation_columns], -exps, exps)
        """
        o = self._info["num_observables"]
        return list(range(o, o + self._info["num_expectations"]))

    # ── coordinates (M-B) ────────────────────────────────────────────────────────
    def get_detector_coordinates(
        self, only: "Iterable[int] | None" = None
    ) -> dict[int, list[float]]:
        """Detector coordinates, keyed by detector index (mirrors Stim's method).

        Coordinates come from `DETECTOR(x, y, ...)` declarations with `SHIFT_COORDS`
        already resolved into absolute values, matching real Stim's
        `stim.Circuit.get_detector_coordinates()`. A coordinate-free DETECTOR maps to
        an empty list `[]`. `only` (any iterable of detector indices) restricts the
        result to those keys, exactly like Stim's `only=` argument."""
        coords = self._info["detector_coords"]
        if only is None:
            return {i: list(c) for i, c in enumerate(coords)}
        out: dict[int, list[float]] = {}
        for i in only:
            if i < 0 or i >= len(coords):
                # ValueError (not IndexError) to match stim's get_detector_coordinates.
                raise ValueError(
                    f"detector index {i} out of range (0..{len(coords) - 1})")
            out[i] = list(coords[i])
        return out

    def get_final_qubit_coordinates(self) -> dict[int, list[float]]:
        """Final qubit coordinates, keyed by qubit index (mirrors Stim's method).

        From `QUBIT_COORDS(x, y, ...)` declarations (last write wins) with
        `SHIFT_COORDS` resolved, matching real Stim's
        `stim.Circuit.get_final_qubit_coordinates()`. Qubits without a QUBIT_COORDS
        are absent from the dict (as in Stim)."""
        return {int(q): list(c) for q, c in self._info["qubit_coords"].items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"xtim.Circuit(n={self.num_qubits}, M={self.num_measurements}, "
                f"D={self.num_detectors}, O={self.num_observables}, "
                f"R={self.num_expectations})")

    # ── invisible reference ─────────────────────────────────────────────────────
    def _resolve_reference(self) -> Reference | None:
        key = _refmod.cache_key(self._text)
        memo_key = (key, str(_refmod._effective_cache_dir()))
        if memo_key not in self._resolved:
            self._resolved[memo_key] = _refmod.resolve_reference(
                self._text, source=self._source)
        return self._resolved[memo_key]

    # ── compilation ─────────────────────────────────────────────────────────────
    def compile_sampler(self, *, seed: int = 0) -> "CompiledSampler":
        """A raw measurement-record sampler (mirrors ``stim.Circuit.compile_sampler``).

        ``seed`` pins the engine stream exactly as ``run_stim_main <file> <shots>
        <seed>`` does; the returned sampler replays the SAME stream on every
        ``sample(shots)`` call. The invisible reference is resolved once, here."""
        return CompiledSampler(self, seed, self._resolve_reference())

    def compile_detector_sampler(self, *, seed: int = 0, engine: str = "auto"
                                 ) -> "CompiledDetectorSampler | TwirlDetectorSampler":
        """A detection-event sampler (mirrors ``stim.Circuit.compile_detector_sampler``),
        with the one record Stim cannot give: raw ``PAULI_EXPECTATION`` values.

        ``seed`` pins the engine stream exactly as ``run_stim_main <file> <shots>
        <seed>`` does; the returned sampler replays the SAME stream on every
        ``sample(shots)`` call. The invisible reference is resolved once, here.

        engine (V3-T3 R3, twirl-as-default arc):
          * ``"exact"`` (default for this release): the legacy engine, unchanged.
          * ``"twirl"``: the twirl record engine explicitly — deterministic detector/
            observable channels are distribution-EXACT, gauge detector columns are
            DECLARED fair coins (documented twirl semantics), refused observables
            raise at ``sample()``. Raises RuntimeError under ``QEC_NO_TWIRL=1``.
          * ``"auto"``: run the twirl channel compilation once; use the twirl engine
            iff EVERY detector classifies deterministic (no gauge/anti content), no
            observable is left refused (deterministic observables and one logical
            Born-weighted observable are both emittable), and setup succeeded —
            anything else routes silently to exact. Inspect ``engine_report()`` on
            the returned sampler for which engine won and why.
        ``QEC_NO_TWIRL=1`` (kill switch) forces exact everywhere: ``auto`` routes to
        exact, explicit ``"twirl"`` raises RuntimeError naming the switch."""
        if engine not in ("exact", "twirl", "auto"):
            raise ValueError(f'engine must be "exact", "twirl" or "auto", got {engine!r}')
        if engine == "exact":
            return CompiledDetectorSampler(self, seed, self._resolve_reference())
        kill = os.environ.get("QEC_NO_TWIRL") == "1"
        if engine == "twirl":
            if kill:
                raise RuntimeError(
                    'engine="twirl" requested but the QEC_NO_TWIRL=1 kill switch is set '
                    '(it forces the exact engine everywhere); unset QEC_NO_TWIRL or use '
                    'engine="exact"/"auto"')
            _sc = int(os.environ.get("QEC_TW_SELFCHECK", "0"))
            tw = _twirl.compile_twirl_sampler(self._text, selfcheck=_sc)   # compile refusals raise loudly
            return TwirlDetectorSampler(tw, seed, reason='engine="twirl" requested')
        # engine == "auto": twirl iff eligible, else exact — never an error, never silent-wrong.
        reason = None
        tw = None
        if kill:
            reason = "QEC_NO_TWIRL=1 kill switch"
        else:
            try:
                _sc = int(os.environ.get("QEC_TW_SELFCHECK", "0"))
                tw = _twirl.compile_twirl_sampler(self._text, selfcheck=_sc)
            except (ValueError, RuntimeError) as e:
                # RuntimeError covers rare engine-invariant guards (e.g. a non-Hermitian
                # observable record product) — the spec's auto contract is never-error:
                # anything the twirl cannot do exactly routes to exact (final-review
                # Important #2). Explicit engine="twirl" still raises these loudly.
                reason = f"twirl compile refused: {str(e).splitlines()[0]}"
            if tw is not None:
                rep = tw.channel_report()
                if rep["gauge_detectors"]:
                    reason = (f"gauge detectors {list(rep['gauge_detectors'])}: gauge "
                              "content is never silently twirled")
                elif rep["refused_observables"]:
                    reason = (f"refused observables {list(rep['refused_observables'])}: "
                              "no exact or Born-weighted record channel")
        if tw is not None and reason is None:
            samp = TwirlDetectorSampler(
                tw, seed,
                reason="auto: all detectors deterministic, all observables emittable")
            # auto contract: NEVER an error. Expectation/raw-measurement requests are a
            # sample-time argument the twirl engine cannot serve; under auto they
            # transparently delegate to a lazily-built exact sampler (same circuit,
            # same seed -- the exact engine's own stream). Explicit engine="twirl"
            # keeps the loud refusal.
            samp._exact_thunk = lambda: CompiledDetectorSampler(
                self, seed, self._resolve_reference())
            return samp
        # Once per DISTINCT reason per process: a circuit that is legitimately
        # ineligible on every compile must not spam, but a NEW slow-selection
        # reason must always be announced (never-silent).
        if reason not in _AUTO_EXACT_WARNED:
            _AUTO_EXACT_WARNED.add(reason)
            import warnings as _warnings
            _warnings.warn(
                f"[xtim] engine=auto -> exact: {reason}. Use engine_report() for details.",
                XtimPerformanceWarning, stacklevel=2)
        s = CompiledDetectorSampler(self, seed, self._resolve_reference())
        s._engine_reason = f"auto -> exact: {reason}"
        return s

    # ── DEM ─────────────────────────────────────────────────────────────────────
    def detector_error_model_text(self, *,
                                  include_expectations: bool = True,
                                  decompose_errors: bool = False,
                                  ignore_decomposition_failures: bool = False,
                                  drop_gauge_observables: bool = False) -> str:
        """Stim-format DEM text.

        include_expectations (default ON): PAULI_EXPECTATION declarations become
        L-columns AFTER the observables (see `expectation_columns`); a mechanism
        whose action on a declared Pauli is not a clean +-1 sign refuses with
        XtimDemError naming column and mechanism. include_expectations=False is the
        documented opt-out: the detector/observable-only export (the classic Stim
        question), exactly the pre-M2 text."""
        r = _xtim.export_dem_text(self._text, include_expectations,
                                  decompose_errors, ignore_decomposition_failures,
                                  drop_gauge_observables)
        if r["errors"]:
            raise _dem_export_error(r["errors"])
        if r["rejected"]:
            gi = r["reject_gate_index"]
            raise XtimRejectError(gi, kind="class")
        if not r["ok"]:
            raise XtimDemError(r["error"])
        return r["dem"]

    def detector_error_model(self, *,
                             include_expectations: bool = True,
                             decompose_errors: bool = False,
                             ignore_decomposition_failures: bool = False,
                             drop_gauge_observables: bool = False
                             ) -> "stim.DetectorErrorModel":
        """Our Stim-format DEM text, parsed into a real stim.DetectorErrorModel."""
        text = self.detector_error_model_text(
            include_expectations=include_expectations,
            decompose_errors=decompose_errors,
            ignore_decomposition_failures=ignore_decomposition_failures,
            drop_gauge_observables=drop_gauge_observables)
        try:
            import stim
        except ImportError as e:
            raise XtimStimMissingError(
                "detector_error_model() returns a stim.DetectorErrorModel and needs the "
                "optional `stim` package (pip install stim); use "
                "detector_error_model_text() for the raw DEM text."
            ) from e
        return stim.DetectorErrorModel(text)

    def detector_error_model_with_reject(self, *,
                                         include_expectations: bool = False,
                                         decompose_errors: bool = False,
                                         ignore_decomposition_failures: bool = False,
                                         drop_gauge_observables: bool = False
                                         ) -> "DemWithReject":
        """A sound+complete decoder model: the Pauli-correctable DEM (`dem`) plus
        `postselect_faults` — the PRIMARY output, the not-Pauli-correctable faults
        (those that change a logical magnitude) each flagged as a `PostselectFault`
        (detector signature + probability) for YOU to post-select or decode-and-budget.
        `reject_detectors` is the derived convenience union (and `.keep_mask(dets)` the
        blunt conservative post-select); both are empty for a Pauli/Clifford circuit
        (then `dem` is byte-identical to `detector_error_model()`). `include_expectations`
        defaults to False here: the magic value is read from the expectation channel,
        not decoded. Raises `XtimDemError` if a single fault changes a logical magnitude
        yet fires no detector (an undetectable logical error it cannot post-select away —
        a circuit fault-tolerance defect). See docs/xtim_dem_reject.md."""
        r = _xtim.export_dem_text(self._text, include_expectations,
                                  decompose_errors, ignore_decomposition_failures,
                                  drop_gauge_observables)
        if r["errors"]:
            raise _dem_export_error(r["errors"])
        if r["rejected"]:
            gi = r["reject_gate_index"]
            raise XtimRejectError(gi, kind="class")
        if not r["ok"]:
            raise XtimDemError(r["error"])
        try:
            import stim
        except ImportError as e:
            raise XtimStimMissingError(
                "detector_error_model_with_reject() returns a stim.DetectorErrorModel "
                "and needs the optional `stim` package (pip install stim)."
            ) from e
        return DemWithReject(
            stim.DetectorErrorModel(r["dem"]),
            list(r["reject_detectors"]),
            [PostselectFault(list(sig), p) for sig, p in r["postselect_faults"]])

    # ── reference surface ───────────────────────────────────────────────────────
    def reference_info(self) -> dict:
        """Cache status + reference header info — on request only, never required."""
        key = _refmod.cache_key(self._text)
        cache_dir = _refmod._effective_cache_dir()
        ref_path = cache_dir / f"{key}.ref"
        info: dict = {
            "engine_version": _xtim.ENGINE_VERSION,
            "ref_format_version": _xtim.REF_FORMAT_VERSION,
            "cache_key": key,
            "cache_dir": str(cache_dir),
            "cached": ref_path.exists(),
            "chi": None,
            "n": None,
            "source": None,
        }
        if info["cached"]:
            try:
                ref = Reference(ref_path.read_text())
                info.update(chi=ref.chi, n=ref.n, source=ref.source)
            except Exception:
                info["cached"] = "corrupt"
        return info

    # ── diagnosis ───────────────────────────────────────────────────────────────
    def diagnose(self, *, shots: int = 1024, seed: int = 0) -> "Diagnosis":
        """Onboarding report (v0.2 F1): a `Diagnosis` of this circuit from ONE
        noiseless run — chi + cache status, deterministic-vs-gauge detectors, per
        expectation-column |beta| / sign-constancy / byproduct-frame HINT /
        DEM-expressibility, and an advisory post-selection recipe.

        Reports measured facts; applies nothing (no sign correction, no acceptance
        filtering). Side-effect-free beyond the noiseless run. See xtim/diagnose.py.
        """
        from .diagnose import diagnose as _diagnose
        return _diagnose(self, shots=shots, seed=seed)

    # ── one-call post-selected logical error rate ────────────────────────────────
    def postselected_logical_error_rate(self, *, p: float = 1e-3, shots: int = 10 ** 6,
                                        seed: int = 0, target_k: int = 1,
                                        p0: float = 1e-3) -> "PostselectedLER":
        """Estimate this circuit's post-selected logical error rate in ONE call.

        This reports the **post-selected** logical error rate: every shot in which ANY
        deterministic detector fired is DISCARDED (heralded). This is the pure-post-selection /
        maximal-heralding number — NOT a decoded logical error rate. A decoded /
        partial-post-selection variant is future work.

        Runs :meth:`diagnose`, AUTO-BUILDS the post-selection that keeps only shots
        in which no deterministic detector fired (the step users forget — omitting it
        turns a suppressed O(p^d) answer into an unsuppressed O(p) one), auto-detects
        the scoring mode, and returns a small result object::

            r = circuit.postselected_logical_error_rate(p=1e-3, shots=10**6)
            r.value        # the LER: 1-F (fidelity mode) | post-selected obs LER
            r.sem          # standard error
            r.acceptance   # post-selection acceptance rate
            r.kept         # accepted shots
            r.mode         # "fidelity" | "observable"
            r.target_k     # k used (fidelity denominator 2**k)
            r.warnings     # list[str] advisories

        Fidelity mode (PAULI_EXPECTATION with a well-defined target) reuses the
        collect/Task fidelity path; observable mode (OBSERVABLE_INCLUDE, no scorable
        expectation) reports the post-selected observable-flip fraction; a circuit
        with neither raises :class:`xtim.XtimError`. Warnings flag an incomplete Pauli
        support (partial-support proxy) or an excluded free/logical read. Full story:
        docs/xtim_postselected_logical_error_rate.md.
        See :func:`xtim.postselected_logical_error_rate`."""
        from .ler import postselected_logical_error_rate as _ler
        return _ler(self, p=p, shots=shots, seed=seed, target_k=target_k, p0=p0)

    # ── byproduct-frame extraction ───────────────────────────────────────────
    def extract_frame(self, operator: str, *, minimal: bool = True) -> list[int]:
        """Return the byproduct-frame rec[-k] offsets for *operator*.

        Delegates to :func:`xtim.extract_frame`; see that function's docstring for
        full documentation.  Returns positive integers *k* such that ``rec[-k]`` are
        the sign-controlling records for the given Pauli product.  Raises
        :class:`xtim.XtimError` if the sign is not pinnable.
        """
        from .fidelity_helpers import extract_frame as _extract_frame
        return _extract_frame(self, operator, minimal=minimal)

    def fidelity_from_logicals(self, logicals: dict, *, p: "float | None" = None,
                               shots: int = 10 ** 6, seed: int = 0,
                               p0: float = 1e-3) -> "PostselectedLER":
        """Post-selected full-support logical infidelity ``1 - F`` from k generator pairs.

        Delegates to :func:`xtim.fidelity_from_logicals`; see that function's docstring
        for full documentation.  ``logicals`` maps each logical qubit to its
        ``{"X": ..., "Z": ...}`` generator strings; the number of pairs is ``k`` and the
        fidelity denominator is ``2**k``.  Returns a fidelity-mode
        :class:`PostselectedLER` whose ``.value`` is ``1 - F`` over all ``4**k`` logical
        Pauli products.  ``.target_k`` carries the inferred ``k``.  ``p0`` is the
        circuit's baked-in noise strength (default 1e-3).  Raises
        :class:`xtim.XtimError` if the generators do not form a valid logical Pauli
        algebra.
        """
        from .fidelity_helpers import fidelity_from_logicals as _ffl
        return _ffl(self, logicals, p=p, shots=shots, seed=seed, p0=p0)

    def compile_reference(self, path: str | os.PathLike | None = None) -> Reference:
        """Power-user surface: run the full ref_compile gate battery NOW.

        Saves to `path` when given, else stores into the reference cache. Raises
        XtimReferenceError when any battery gate fails."""
        ref = _refmod.compile_reference(self._text, source=self._source)
        if path is not None:
            ref.save(path)
        else:
            _refmod._store(ref, _refmod._effective_cache_dir() / f"{_refmod.cache_key(self._text)}.ref",
                           _refmod._deferred_sig(self._text))
        return ref


class _CompiledBase:
    def __init__(self, circuit: Circuit, seed: int,
                 ref: Reference | None):
        self._circuit = circuit
        # The engine seed is an unsigned 64-bit int; validate here so a bad seed
        # raises a clear error instead of leaking a raw pybind11 TypeError from sample().
        try:
            seed_i = operator.index(seed)
        except TypeError:
            raise TypeError(f"seed must be an int, got {type(seed).__name__}") from None
        if not 0 <= seed_i < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed_i}")
        self._seed = seed_i
        self._ref = ref
        # Compile ONCE here (deferral / propagation / bare-state / cascade) and hold the stateful
        # program, so every _run(shots) just samples it — no per-call recompile (the ~870ms d5 /
        # ~66ms d3 compile is paid a single time). Byte-identical to the old run_packed path.
        self._program = _xtim.CompiledProgram(
            self._circuit.text,
            self._ref.text if self._ref is not None else "")
        if self._program.ref_error:
            raise XtimRejectError(
                -1, kind="ref_mismatch",
                message="rejected: supplied reference failed to load: "
                f"{self._program.ref_error}")
        if not self._program.ok:  # parse errors (unreachable: __init__ parses first)
            raise XtimParseError(self._program.errors)

    def __repr__(self) -> str:
        return (f"xtim.{type(self).__name__}(circuit={self._circuit!r}, "
                f"seed={self._seed})")

    @property
    def seed(self) -> int:
        return self._seed

    def _run(self, shots: int, *, want_meas: bool = True, want_det: bool = True,
             want_obs: bool = True, want_exp: bool = True) -> dict:
        # Validate the type up front so a str/None/float gives a clear error rather
        # than a leaked comparison TypeError or a silent float->int truncation.
        try:
            shots = operator.index(shots)
        except TypeError:
            raise TypeError(
                f"shots must be an integer, got {type(shots).__name__}") from None
        if shots < 0:
            raise ValueError("shots must be >= 0")
        # The program was compiled (and parse/ref-error checked) ONCE in __init__; just sample it.
        # The want_* mask tells the binding to materialize only the record-buffer channels this
        # sampler will actually read (the others' numpy arrays are skipped, trimming the per-shot
        # marshalling). The scalar counts/chi/reject metadata is always returned. Defaults are
        # all-True so any direct caller is unaffected.
        r = self._program.sample(int(shots), self._seed, want_meas, want_det,
                                 want_obs, want_exp)
        if r["rejected"]:
            raise self._reject_error(r["reject_gate_index"])
        return r

    def _reject_error(self, gate_index: int) -> XtimRejectError:
        """Resolve a run-path reject into a typed XtimRejectError with a DISTINCT
        kind. A gate_index >= 0 is an out-of-class gate (the propagation-class check,
        the SOLE rejection criterion). A -1 is either a supplied-reference n-mismatch
        or a non-gate-localized propagation-class rejection.
        """
        if gate_index >= 0:
            return XtimRejectError(gate_index, kind="class")
        if self._ref is not None:
            return XtimRejectError(
                -1, kind="ref_mismatch", ref_n=self._ref.n,
                circuit_n=self._circuit.num_qubits,
            )
        # no reference supplied: non-gate-localized propagation-class rejection.
        return XtimRejectError(-1, kind="class")


class CompiledSampler(_CompiledBase):
    """Raw measurement-record sampler (Stim's compile_sampler analog)."""

    def sample(self, shots: int, *, bit_packed: bool = False) -> np.ndarray:
        """One engine run -> the raw measurement records ``bool[shots, M]`` (mirrors
        ``stim.CompiledMeasurementSampler.sample``).

        With ``bit_packed=True`` the records come Stim-b8-packed (``uint8[shots,
        ceil(M/8)]``, little-endian bit order). All records of one call come from the
        SAME single engine run (one RNG stream)."""
        # Raw measurement sampler: only the measurements channel is ever read.
        r = self._run(shots, want_meas=True, want_det=False, want_obs=False,
                      want_exp=False)
        if bit_packed:
            return r["measurements"]
        return _unpack_bits(r["measurements"], r["num_measurements"])


class CompiledDetectorSampler(_CompiledBase):
    """Detector-event sampler (Stim's compile_detector_sampler analog) with the one
    record Stim cannot have: raw PAULI_EXPECTATION values."""

    def engine_report(self) -> dict:
        """Which engine backs this sampler and why (twirl-as-default arc, V3-T3 R3)."""
        return {"engine": "exact",
                "reason": getattr(self, "_engine_reason", 'engine="exact" requested')}

    def sample(self, shots: int, *, separate_observables: bool = False,
               append_observables: bool = False,
               return_expectations: bool = False, return_measurements: bool = False,
               bit_packed: bool = False) -> "np.ndarray | tuple[np.ndarray, ...]":
        """One engine run -> (dets[, obs][, exps][, meas]).

        dets bool[shots, D] detection events; obs bool[shots, O] observable flips;
        exps float64[shots, R] RAW expectation values; meas bool[shots, M] raw
        measurement bits. With bit_packed=True the bit records come Stim-b8-packed
        (uint8[shots, ceil(K/8)], little-endian bit order); exps stay float64.
        All requested records come from the SAME single engine run (one RNG stream).
        Returns the dets array alone unless extra records are requested.

        append_observables=True appends the observable-flip columns onto the detector
        array (Stim's combined `(shots, D+O)` layout) instead of returning them
        separately; it is mutually exclusive with separate_observables."""
        if append_observables and separate_observables:
            raise ValueError(
                "pass at most one of append_observables / separate_observables")
        # Request only the channels this call will read (mirrors the r[...] accesses below):
        #   detectors  — always (the primary return, both layouts).
        #   observables — only when appended onto / returned alongside the detectors.
        #   expectations — only with return_expectations.
        #   measurements — only with return_measurements.
        r = self._run(
            shots,
            want_det=True,
            want_obs=append_observables or separate_observables,
            want_exp=return_expectations,
            want_meas=return_measurements,
        )

        def bits(buf, k):
            return buf if bit_packed else _unpack_bits(buf, k)

        if append_observables:
            # Stim's default detector-sampler layout: observable flips appended as
            # extra columns after the detectors. Combine in the unpacked domain, then
            # re-pack if requested.
            dets = _unpack_bits(r["detectors"], r["num_detectors"])
            obs = _unpack_bits(r["observables"], r["num_observables"])
            combined = np.concatenate([dets, obs], axis=1) if obs.shape[1] else dets
            first = (np.packbits(combined, axis=1, bitorder="little")
                     if bit_packed else combined)
        else:
            first = bits(r["detectors"], r["num_detectors"])

        out = [first]
        if separate_observables:
            out.append(bits(r["observables"], r["num_observables"]))
        if return_expectations:
            out.append(r["expectations"])
        if return_measurements:
            out.append(bits(r["measurements"], r["num_measurements"]))
        return out[0] if len(out) == 1 else tuple(out)


_AUTO_EXACT_WARNED: set = set()   # auto->exact perf-warning dedup (once per reason per process)


class TwirlDetectorSampler:
    """Detector-event sampler backed by the twirl record engine
    (``compile_detector_sampler(engine="twirl"/"auto")``, V3-T3 R3).

    ``sample()`` is signature-compatible with :class:`CompiledDetectorSampler` for
    the record path (detectors / ``separate_observables`` / ``append_observables`` /
    ``bit_packed``) and replays the SAME stream on every call (the compile ``seed``
    pins the engine rebuild). DOCUMENTED DELTA: ``return_expectations`` and
    ``return_measurements`` are not part of the twirl record product — they raise
    ValueError; use ``engine="exact"`` for those records. Semantics contract
    (docs/twirl_record_sampler.md): deterministic detector/observable channels are
    distribution-EXACT; gauge detector columns are DECLARED fair coins; no shot is
    ever dropped (guard trips are computed per-shot by the exact engine —
    ``channel_report()["exact_shots"]`` counts them).

    Performance: ``compile_detector_sampler`` compiles with ``selfcheck=0`` (oracle
    disabled on the fast path) by default. Set ``QEC_TW_SELFCHECK=<n>`` to enable the
    oracle window for the first ``n`` fired shots (e.g. ``QEC_TW_SELFCHECK=2000``)."""

    def __init__(self, sampler, seed: int, reason: str):
        try:
            seed_i = operator.index(seed)
        except TypeError:
            raise TypeError(f"seed must be an int, got {type(seed).__name__}") from None
        if not 0 <= seed_i < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed_i}")
        self._s = sampler
        self._seed = seed_i
        self._reason = reason
        self._exact_thunk = None    # set by engine="auto": lazy exact fallback for
        self._exact = None          # sample-time requests the twirl engine cannot serve

    def __repr__(self) -> str:
        return f"xtim.TwirlDetectorSampler(seed={self._seed})"

    @property
    def seed(self) -> int:
        return self._seed

    def engine_report(self) -> dict:
        """Which engine backs this sampler and why, plus the channel classification."""
        rep = self._s.channel_report()
        return {"engine": "twirl", "reason": self._reason,
                "deterministic_detectors": rep["deterministic_detectors"],
                "gauge_detectors": rep["gauge_detectors"],
                "anti_detectors": rep["anti_detectors"],
                "refused_observables": rep["refused_observables"],
                "exact_shots": rep["exact_shots"],
                "disk_path": rep["disk_path"]}

    def channel_report(self) -> dict:
        return self._s.channel_report()

    def sample(self, shots: int, *, separate_observables: bool = False,
               append_observables: bool = False,
               return_expectations: bool = False, return_measurements: bool = False,
               bit_packed: bool = False) -> "np.ndarray | tuple[np.ndarray, ...]":
        """One engine run -> dets[, obs] (see CompiledDetectorSampler.sample)."""
        if append_observables and separate_observables:
            raise ValueError(
                "pass at most one of append_observables / separate_observables")
        if return_expectations or return_measurements:
            if self._exact_thunk is not None:      # auto: delegate, never error
                if self._exact is None:
                    self._exact = self._exact_thunk()
                return self._exact.sample(
                    shots, separate_observables=separate_observables,
                    append_observables=append_observables,
                    return_expectations=return_expectations,
                    return_measurements=return_measurements,
                    bit_packed=bit_packed)
            raise ValueError(
                "the twirl engine emits detector/observable records only "
                "(documented delta); use engine=\"exact\" for expectations or raw "
                "measurements")
        dets_p, obs_p = self._s.sample(int(shots), seed=self._seed)
        ndet = self._s.num_detectors
        nobs = self._s.num_observables
        if append_observables:
            dets = _unpack_bits(dets_p, ndet)
            obs = _unpack_bits(obs_p, nobs)
            combined = np.concatenate([dets, obs], axis=1) if obs.shape[1] else dets
            first = (np.packbits(combined, axis=1, bitorder="little")
                     if bit_packed else combined)
        else:
            first = dets_p if bit_packed else _unpack_bits(dets_p, ndet)
        if separate_observables:
            return first, (obs_p if bit_packed else _unpack_bits(obs_p, nobs))
        return first
