"""xtim.Circuit — the Stim-shaped entry point over the MSP engine.

The workflow mirrors ordinary Stim:

    c = xtim.Circuit.from_file("protocol.stim")
    dets, obs = c.compile_detector_sampler(seed=7).sample(10**6, separate_observables=True)
    dem = c.detector_error_model()         # a real stim.DetectorErrorModel

Detection events and observable flips are reference-relative (Stim's semantics). To score
an output state, use ``xtim.compile_twirl_sampler`` + ``xtim.RecordFidelity`` (docs/usage.md).

Determinism: a compiled sampler replays the SAME stream on every `sample(shots)`
call — `seed` pins the run.
"""
from __future__ import annotations

import numbers
import operator
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np

from ._exact_scope import explicit_exact  # J-b1 P13
from . import _xtim
from . import twirl as _twirl
from .errors import (
    TwirlClass,
    XtimDemError,
    XtimError,
    XtimParseError,
    XtimDemWarning,
    XtimRejectError,
    XtimStimMissingError,
    XtimTwirlRefusal,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import stim

    from .ler import PostselectedLER

def _dem_export_error(errors) -> "XtimDemError":
    """Build an XtimDemError from DEM-export `errors` (list of (line, msg)). The circuit
    already parsed, so these are DEM-export limitations, not parse errors."""
    msg = "; ".join(m for _, m in errors)
    return XtimDemError(msg)


def _disjoint_threshold(approximate_disjoint_errors) -> float:
    """stim's `approximate_disjoint_errors`: False -> 0 (refuse a channel with no exact
    conversion into independent mechanisms), True -> 1, a float t in [0, 1] -> threshold t (a
    channel needing the approximation is refused if one of its probability arguments exceeds t)."""
    if isinstance(approximate_disjoint_errors, bool):
        return 1.0 if approximate_disjoint_errors else 0.0
    t = float(approximate_disjoint_errors)
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"approximate_disjoint_errors must be a bool or a float in [0, 1], got {t!r}")
    return t


_CLIFFORD_CUTOFF_MAX = 2 ** 31 - 1                # the binding's int range; larger t: pass math.inf
CLIFFORD_CUTOFF_DEFAULT: int = 2                  # owner 2026-11-02: the DEM default is t = 2 (exact: math.inf)


def _clifford_cutoff(clifford_cutoff) -> int:
    """The DEM's Clifford-fault weight cutoff t: an int 0 <= t < 2^31 -> itself (the signature default is
    CLIFFORD_CUTOFF_DEFAULT = 2, owner 2026-11-02); None or math.inf -> -1 (EXACT: every pattern, opt-in);
    anything else raises ValueError."""
    if clifford_cutoff is None:
        return -1
    if isinstance(clifford_cutoff, bool):
        raise ValueError(f"clifford_cutoff must be None / math.inf (exact) or an int 0 <= t < 2^31, got {clifford_cutoff!r}")
    if isinstance(clifford_cutoff, float) and clifford_cutoff == float("inf"):
        return -1
    if isinstance(clifford_cutoff, numbers.Integral) and 0 <= int(clifford_cutoff) <= _CLIFFORD_CUTOFF_MAX:
        return int(clifford_cutoff)
    raise ValueError(f"clifford_cutoff must be None / math.inf (exact) or an int 0 <= t < 2^31, got {clifford_cutoff!r}")


@dataclass(frozen=True)
class PostselectFault:
    """One detectable-but-not-Pauli-correctable fault: the detector signature it
    flips/randomizes (no Pauli correction exists, so it cannot be a DEM edge) plus its
    probability. Summed over all faults, the probabilities give the O(p) coherent-error
    floor (a lower bound on the logical infidelity if you decode rather than post-select)."""

    detectors: list[int]   # the syndrome this fault flips/randomizes (no Pauli correction exists)
    probability: float


_DROPPED_COLUMNS_WARNED: set = set()


# A finding the export REPORTED instead of refusing (include_expectations=False only).
_REPORTED_FINDING = "(reported, not refused: include_expectations=False)"


def _warn_dropped_expectation_columns(text: str, r: dict, include_expectations: bool = True) -> None:
    """Once per (circuit, mode, reported set) per process. include_expectations=True: name the
    PAULI_EXPECTATION columns DEM export left out because their noiseless per-shot value varies
    shot by shot. include_expectations=False: name the fault-tolerance findings the export
    REPORTED instead of refusing (no expectation column was requested, so the
    detector/observable DEM is exact; the finding is about a column that is not in it)."""
    dropped = r.get("dropped_expectation_columns") or []
    if not include_expectations:
        dropped = [(i, w) for i, w in dropped if _REPORTED_FINDING in str(w)]
    if not dropped:
        return
    key = (hash(text), bool(include_expectations), tuple(int(i) for i, _ in dropped))
    if key in _DROPPED_COLUMNS_WARNED:
        return
    _DROPPED_COLUMNS_WARNED.add(key)
    import warnings as _warnings
    if include_expectations:
        msg = ("[xtim] DEM export dropped PAULI_EXPECTATION column(s) with no DEM concept "
               "(noiseless per-shot value varies shot by shot): "
               + "; ".join(f"declaration {int(i)} ({w})" for i, w in dropped)
               + ". Declare the named records as the column's frame to make it a DEM column.")
    else:
        msg = ("[xtim] DEM export (include_expectations=False) reports a finding on "
               "PAULI_EXPECTATION column(s) that were not requested (the DEM carries every fault "
               "on its detector/observable signature; with include_expectations=True the export "
               "refuses): "
               + "; ".join(f"declaration {int(i)} ({w})" for i, w in dropped))
    _warnings.warn(msg, XtimDemWarning, stacklevel=3)


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
    # PAULI_EXPECTATION columns with no DEM concept: (declaration index r, reason). Their
    # noiseless per-shot value varies shot by shot (it depends on reads not declared as the
    # column's frame), so column L(O+r) is left out of `dem` entirely (3.1.12).
    # With include_expectations=False (not the default) the same list also carries, per
    # (column, fault location), what the column analysis REFUSES with include_expectations=True
    # (the completeness gate "non-fault-tolerant circuit (<loc>): ..." and the "accepted branch
    # was not verified" refusals), suffixed "(reported, not refused: include_expectations=False)":
    # no column was requested, `dem` carries every fault on its detector/observable signature
    # and nothing is flagged or dropped (owner decision 2026-10-12).
    dropped_expectation_columns: list[tuple[int, str]] = field(default_factory=list)
    # The Clifford records (one per twirled Clifford fault: `error[clifford:f<id>]` lines) and their
    # total probability mass in patterns flipping more than `clifford_cutoff` detectors (dropped; 0.0
    # at clifford_cutoff=math.inf). The DEM text carries the same numbers in its `# clifford` summary
    # line; a stim.DetectorErrorModel does not keep comments, so read them here.
    clifford_faults: int = 0
    clifford_dropped_mass: float = 0.0

    def __repr__(self) -> str:
        # Compact: the dataclass default embeds the full DEM text + every fault (a huge
        # interactive dump); keep a summary — the full data stays on the attributes.
        return ("DemWithReject(detectors=%d, mechanisms=%d, reject_detectors=%d, "
                "postselect_faults=%d, dropped_expectation_columns=%d, clifford_faults=%d, "
                "clifford_dropped_mass=%.3g)" % (
                    self.dem.num_detectors, self.dem.num_errors,
                    len(self.reject_detectors), len(self.postselect_faults),
                    len(self.dropped_expectation_columns), self.clifford_faults,
                    self.clifford_dropped_mass))

    def keep_mask(self, dets: np.ndarray) -> np.ndarray:
        """Boolean mask over shots — True where NO reject detector fired (the shots to
        keep / post-select). `dets` is a bool[shots, num_detectors] detector array.
        Decode `dets[keep_mask(dets)]` with `self.dem`; discard the rest.

        This is the BLUNT, conservative policy: it rejects whenever any flagged detector
        fires, so it can over-reject shots where that detector is shared with an ordinary
        Pauli edge — but it is ALWAYS sound, which is why it is the recommended default
        (and on a saturated-reject protocol it is provably minimal)."""
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

    # ── compilation ─────────────────────────────────────────────────────────────
    def compile_detector_sampler(self, *, seed: int = 0) -> "TwirlDetectorSampler":
        """A detection-event sampler (mirrors ``stim.Circuit.compile_detector_sampler``), backed by the
        twirl record engine: every detector and observable is sampled from the exact joint record law
        (gauge and ANTI records included; docs/twirl_record_sampler.md).

        ``seed`` pins the engine stream; the returned sampler replays the SAME stream on every
        ``sample(shots)`` call. A circuit the engine cannot serve raises :class:`xtim.XtimTwirlRefusal`
        HERE, by name (``twirl_class``: OUTSIDE_T_DEPTH_1, NONCOMMUTING_MAGIC, LAW_LIMITATION_PROPAGATION;
        LAW_LIMITATION_CARRIED_AXIS on carried branches), and
        ``sample(return_expectations=True | return_measurements=True)`` raises it with
        EXPECTATIONS_NOT_SERVED / RAW_RECORDS_NOT_SERVED (score the end state with OUTPUT_QUBITS +
        ``sample_barrier`` + ``RecordFidelity``). There is no other engine, no routing and no fallback."""
        _refuse_unresolved_if(self._text, "compile_detector_sampler")
        # decisions=False: detector/observable records only (see TwirlDetectorSampler). Refusals raise
        # xtim.errors.XtimTwirlRefusal by name here (no routing, no fallback: owner OD-B).
        tw = _twirl.compile_twirl_sampler(self._text, selfcheck=0, decisions=False)
        return TwirlDetectorSampler(tw, seed)

    # ── DEM ─────────────────────────────────────────────────────────────────────
    def detector_error_model_text(self, *,
                                  include_expectations: bool = True,
                                  decompose_errors: bool = False,
                                  ignore_decomposition_failures: bool = False,
                                  drop_gauge_observables: bool = False,
                                  approximate_disjoint_errors: "bool | float" = False,
                                  clifford_cutoff: "int | float | None" = CLIFFORD_CUTOFF_DEFAULT,
                                  erasure: bool = False) -> str:
        """Stim-format DEM text (with erasure=True: an xtim research format, see below).

        include_expectations (default ON): PAULI_EXPECTATION declarations become
        L-columns AFTER the observables (see `expectation_columns`). An L-flip on
        such a column means the fault multiplies P' = P * (Paulis of the declared
        frame records) by -1, i.e. multiplies the reported <P> by -1 (records
        relabelled by the fault's own record flips) — defined for any noiseless
        value: 0, 1/sqrt2, or branch-dependent. A fault whose action on P' is not
        +-1 (a Clifford residual) is flagged for post-selection when it fires a
        detector (detector_error_model_with_reject) and refuses with XtimDemError
        when it fires none; on a column with a nonzero noiseless value the
        magnitude classifier may certify it as a sign instead.

        approximate_disjoint_errors (stim's semantics): a noise channel whose
        disjoint signature distribution has no exact conversion into independent
        mechanisms raises XtimDemError by default; True approximates it (each
        merged signature an independent mechanism with its disjoint probability),
        a float t approximates only if every probability argument is <= t. include_expectations=False is the
        documented opt-out: the detector/observable-only export (the classic Stim
        question). Under it the DEM carries EVERY fault on its detector/observable
        signature: the completeness gate ("non-fault-tolerant circuit (<loc>): ...")
        and the accepted-branch checks do not refuse, flag or drop anything (they
        are statements about the columns, which were not requested); what they
        would have refused is reported instead (`dropped_expectation_columns` on
        detector_error_model_with_reject; an XtimDemWarning here).


        clifford_cutoff=t: keep only each Clifford fault's patterns that flip <= t
        detectors. Default 2: graphlike, matching-friendly lines, but RISK: dropped
        patterns can lower the suppression order, and a fault whose fixed part already
        flips >= 3 detectors vanishes from the DEM entirely (reported as dropped mass).
        clifford_cutoff=math.inf (or None) = EXACT (every pattern).

        clifford_cutoff in detail (dem-erasure): a fault whose propagated error is a Clifford
        (a T/CCZ-crossing S or CZ residue) randomizes the reads it rotates; its
        twirled law is uniform on an affine space c + W of dimension v over the DEM
        targets (an unheralded erasure). Each such fault is its OWN record, never
        merged with other faults: a `# clifford f<id>: p=.. v=.. offset=.. support=..
        loc=.. alt=.. kept=.. dropped=..` header and `error[clifford:f<id>](q)`
        lines, one per kept pattern, each an independent mechanism at the closed
        form q (all equal). The WEIGHT of a pattern is the number of DETECTORS it
        flips (observable / expectation / frame / decision columns never count);
        patterns of weight <= clifford_cutoff are kept (observable-only patterns
        always). 2 (default) = graphlike; math.inf (or None) = every pattern (the
        exact split, at most 2^12 - 1 lines per fault: v > 12 refuses by name,
        naming the exact request as the cause and a finite cutoff as the way
        out), an int t >= 0 = t (the v cap is lifted for
        coset laws: a fault refuses only past 2^12 candidate patterns). Kept patterns are first-order exact (a fault with a
        deterministic offset, c not in W, omits a negative O(p^2 / 2^(v+1))
        correction); each detector's marginal from one fault falls below p/2. A
        dropped weight-w pattern is still reachable as ~ceil(w/t) kept mechanisms,
        so dropping can LOWER the suppression order. Loud: the summary line
        `# clifford faults=<N> cutoff=<t|inf> dropped_mass=<sum>` opens the text
        and `detector_error_model_with_reject` carries `clifford_faults` /
        `clifford_dropped_mass` (0.0 when nothing is dropped).
        approximate_disjoint_errors does not touch a coset record (only when its
        closed form over-mixes, p >= 1/2). A Clifford fault whose law is not a
        uniform coset (`law=support`) is solved exactly alone, or emitted as its
        disjoint law when it has no independent split (`first_order=1`); the Pauli
        alternatives of a channel instance with Clifford alternatives are solved
        alone (an O(p^2) change), first-order if they do not factor. Weight counts
        the offset's detectors too, so the exact split needs clifford_cutoff >= the
        largest pattern weight (which can exceed v), or math.inf.

        erasure=True (dem-erasure-out; a RESEARCH option, default False = the output above, byte for byte):
        every uniform-coset Clifford record is written as ONE line instead of its patterns,
        ``erasure[clifford:f<id>](p) <offset targets> | <gen 1 targets> | ... | <gen v targets>``:
        with probability p (the fault's raw probability) flip the offset plus a uniformly random
        element of the span of the v generators. clifford_cutoff does not apply to these lines and
        nothing is enumerated (cost linear in v; no v cap). Explicit-support records (`law=support`)
        are not cosets and keep their pattern lines (cutoff applies; header ends "(not a coset;
        patterns)"); the summary line ends ``mode=erasure``; Pauli lines are unchanged. The text is
        NOT ingestible by stim (``stim.DetectorErrorModel`` raises on ``erasure``) or by any decoder,
        so only this text entry point takes it (detector_error_model / _with_reject do not).
        Refused together with decompose_errors=True."""
        _refuse_unresolved_if(self._text, "detector_error_model")
        if not isinstance(erasure, bool):
            raise ValueError(f"erasure must be a bool, got {erasure!r}")
        with explicit_exact():                     # J-b1 P13: the DEM export is explicit
            r = _xtim.export_dem_text(self._text, include_expectations,
                                  decompose_errors, ignore_decomposition_failures,
                                  drop_gauge_observables, [], [],
                                  _disjoint_threshold(approximate_disjoint_errors),
                                  _clifford_cutoff(clifford_cutoff), erasure)
        if r["errors"]:
            raise _dem_export_error(r["errors"])
        if r["rejected"]:
            gi = r["reject_gate_index"]
            raise XtimRejectError(gi, kind="class")
        if not r["ok"]:
            raise XtimDemError(r["error"])
        _warn_dropped_expectation_columns(self._text, r, include_expectations)
        return r["dem"]

    def detector_error_model(self, *,
                             include_expectations: bool = True,
                             decompose_errors: bool = False,
                             ignore_decomposition_failures: bool = False,
                             drop_gauge_observables: bool = False,
                             approximate_disjoint_errors: "bool | float" = False,
                             clifford_cutoff: "int | float | None" = CLIFFORD_CUTOFF_DEFAULT
                             ) -> "stim.DetectorErrorModel":
        """Our Stim-format DEM text, parsed into a real stim.DetectorErrorModel.
        Arguments as `detector_error_model_text`.

        clifford_cutoff=t: keep only each Clifford fault's patterns that flip <= t
        detectors. Default 2: graphlike, matching-friendly lines, but RISK: dropped
        patterns can lower the suppression order, and a fault whose fixed part already
        flips >= 3 detectors vanishes from the DEM entirely (reported as dropped mass).
        clifford_cutoff=math.inf (or None) = EXACT (every pattern). A
        stim.DetectorErrorModel drops `#` comments, so the dropped Clifford mass
        of `clifford_cutoff` is not on this object: read it from
        `detector_error_model_text` (its `# clifford` summary line) or
        `detector_error_model_with_reject` (`clifford_dropped_mass`)."""
        text = self.detector_error_model_text(
            include_expectations=include_expectations,
            decompose_errors=decompose_errors,
            ignore_decomposition_failures=ignore_decomposition_failures,
            drop_gauge_observables=drop_gauge_observables,
            approximate_disjoint_errors=approximate_disjoint_errors,
            clifford_cutoff=clifford_cutoff)
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
                                         include_expectations: bool = True,
                                         decompose_errors: bool = False,
                                         ignore_decomposition_failures: bool = False,
                                         drop_gauge_observables: bool = False,
                                         approximate_disjoint_errors: "bool | float" = False,
                                         clifford_cutoff: "int | float | None" = CLIFFORD_CUTOFF_DEFAULT
                                         ) -> "DemWithReject":
        """A sound+complete decoder model: the Pauli-correctable DEM (`dem`) plus
        `postselect_faults` — the PRIMARY output, the not-Pauli-correctable faults
        (those that change a logical magnitude) each flagged as a `PostselectFault`
        (detector signature + probability) for YOU to post-select or decode-and-budget.
        `reject_detectors` is the derived convenience union (and `.keep_mask(dets)` the
        blunt conservative post-select); both are empty for a Pauli/Clifford circuit
        (then `dem` is byte-identical to `detector_error_model()`). The reject region
        exists only when the columns are REQUESTED, so `include_expectations` defaults
        to True here (matching `detector_error_model`; until 2026-10-12 it defaulted to
        False, which after the same-day ruling would have meant empty flags by default):
        `XtimDemError` is raised if a single fault changes a logical magnitude yet fires
        no detector (an undetectable logical error it cannot post-select away — a
        circuit fault-tolerance defect), the detected ones are flagged, and `dem` has
        the expectation L-columns after the ordinary observables (decode it as usual and
        ignore the decoder's predictions for those columns: the magic value is read from
        the expectation channel, never decoded). A caller who wants the plain
        detector/observable DEM with empty flags passes include_expectations=False: no
        column is requested, `dem` carries every fault on its detector/observable
        signature (for every class-admitted circuit byte-identical to the twin circuit without the declarations),
        `postselect_faults` / `reject_detectors` are empty, and what the column analysis
        would have refused is REPORTED in `dropped_expectation_columns`
        ("non-fault-tolerant circuit (<loc>): ... column L<i> ... (reported, not refused:
        include_expectations=False)"). clifford_cutoff: see `detector_error_model_text`;
        `clifford_faults` / `clifford_dropped_mass` report the Clifford records."""
        _refuse_unresolved_if(self._text, "detector_error_model")
        with explicit_exact():                     # J-b1 P13: the DEM export is explicit
            r = _xtim.export_dem_text(self._text, include_expectations,
                                  decompose_errors, ignore_decomposition_failures,
                                  drop_gauge_observables, [], [],
                                  _disjoint_threshold(approximate_disjoint_errors),
                                  _clifford_cutoff(clifford_cutoff))
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
        _warn_dropped_expectation_columns(self._text, r, include_expectations)
        return DemWithReject(
            stim.DetectorErrorModel(r["dem"]),
            list(r["reject_detectors"]),
            [PostselectFault(list(sig), p) for sig, p in r["postselect_faults"]],
            [(int(i), str(w)) for i, w in r["dropped_expectation_columns"]],
            int(r["clifford_faults"]), float(r["clifford_dropped_mass"]))

    # ── one-call post-selected logical error rate ────────────────────────────────
    def postselected_logical_error_rate(self, *, p: float = 1e-3, shots: int = 10 ** 6,
                                        seed: int = 0, p0: float = 1e-3) -> "PostselectedLER":
        """Estimate this circuit's post-selected logical error rate on its OBSERVABLE_INCLUDE
        observable(s) in ONE call: every shot in which ANY deterministic detector fired is
        DISCARDED (the exact noiseless determinism partition), and ``value`` is the fraction of
        accepted shots with a logical observable flipped (the default twirl sampler)::

            r = circuit.postselected_logical_error_rate(p=1e-3, shots=10**6)
            r.value, r.sem, r.acceptance, r.kept, r.upper_bound, r.warnings

        A circuit that declares PAULI_EXPECTATION columns is refused by name (the beta-fidelity
        mode was removed; score the output state with RecordFidelity). Full story:
        docs/xtim_postselected_logical_error_rate.md. See :func:`xtim.postselected_logical_error_rate`."""
        from .ler import postselected_logical_error_rate as _ler
        return _ler(self, p=p, shots=shots, seed=seed, p0=p0)

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


# IF blocks (``INPUT_BITS name W`` + ``IF [!]name[k] { ... }``) are per-shot conditionals on
# bits the CALLER supplies (cpp/include/qeccore/circuit_ir.hpp: "run the span for a shot iff
# decision[cond_port][cond_bit] == cond_value"). No engine path below `xtim.Circuit` takes
# those bits: the DEM export ignores the IfBegin/IfEnd markers (every body would
# run unconditionally -- undocumented, silently wrong), and the twirl partition sampler needs
# `decision_bits` at sample(). Owner rule: exact or refuse by name -- so every surface that
# compiles the circuit as-is refuses, naming the IF input and the escape (resolve the branches
# to a concrete text per bit pattern with `xtim._xtim.resolve_branches_text(text, bits)`, or
# use `xtim.twirl.PartitionedTwirlSampler.sample(decision_bits=...)`).
_IF_COND_RE = re.compile(r"^\s*IF\s+(!?[A-Za-z_]\w*\[\d+\])\s*\{", re.M)


def _unresolved_if_conditions(text: str) -> list:
    """The distinct ``IF name[k]`` conditions whose bits drive a branch, in text order
    (empty for an IF-free circuit; a parse failure is left to the compile that follows)."""
    try:
        if not _xtim.if_driving_bits(text):
            return []
    except (ValueError, RuntimeError):
        return []
    return list(dict.fromkeys(m.group(1).lstrip("!") for m in _IF_COND_RE.finditer(text)))


def _refuse_unresolved_if(text: str, surface: str) -> None:
    conds = _unresolved_if_conditions(text)
    if conds:
        raise ValueError(
            f"{surface}: the circuit has IF blocks on INPUT_BITS "
            f"({', '.join('IF ' + c for c in conds)}) and no bit values are supplied here; "
            "an IF body runs for a shot iff its bit matches, which this surface cannot "
            "evaluate. Resolve the branches first (xtim._xtim.resolve_branches_text(text, "
            "bits) -> one concrete circuit per bit pattern) or use "
            "xtim.twirl.PartitionedTwirlSampler.sample(decision_bits=...)")


_EXPECTATIONS_TEXT = ("the twirl engine does not serve expectations: score the end state with the record route "
                      "(declare OUTPUT_QUBITS, then sample_barrier + RecordFidelity: the exact register fidelity)")
_RAW_RECORDS_TEXT = ("the twirl engine does not serve raw measurement records: it samples detectors, observables "
                     "and DECISION bits")


class TwirlDetectorSampler:
    """Detector-event sampler backed by the twirl record engine (``Circuit.compile_detector_sampler()``).

    ``sample()`` serves detectors / ``separate_observables`` / ``append_observables`` / ``bit_packed``
    and replays the SAME stream on every call (the compile ``seed`` pins the engine rebuild).
    ``return_expectations`` and ``return_measurements`` are not part of the twirl record product — they raise
    :class:`xtim.XtimTwirlRefusal` (EXPECTATIONS_NOT_SERVED / RAW_RECORDS_NOT_SERVED); the end
    state is scored by the record route (``sample_barrier`` + ``RecordFidelity``). ``DECISION`` declarations
    are NOT compiled into this sampler's engine (``compile_twirl_sampler(..., decisions=False)``): it has
    no decision output, and a DECISION is never consumed inside the engine, so its det/obs streams are
    byte-identical to the same circuit with the DECISION lines removed — at that
    circuit's speed and with none of the Born-decision per-shot path's refusals
    (decision bits: ``xtim.port`` / ``sample_barrier``). Semantics contract
    (docs/twirl_record_sampler.md): every detector / observable column is a record of the
    exact joint record law (gauge detectors are exact Born records); every shot is served
    by that law — there is no fallback.

    Performance: ``compile_detector_sampler`` compiles with ``selfcheck=0`` (oracle
    window off on the fast path). For the window, compile with
    ``xtim.compile_twirl_sampler(text, selfcheck=n)``."""

    def __init__(self, sampler, seed: int):
        try:
            seed_i = operator.index(seed)
        except TypeError:
            raise TypeError(f"seed must be an int, got {type(seed).__name__}") from None
        if not 0 <= seed_i < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed_i}")
        self._s = sampler
        self._seed = seed_i

    def __repr__(self) -> str:
        return f"xtim.TwirlDetectorSampler(seed={self._seed})"

    @property
    def seed(self) -> int:
        return self._seed

    def engine_report(self) -> dict:
        """Which engine backs this sampler: ``{"engine": "twirl"}`` (the only engine)."""
        return {"engine": "twirl"}

    def record_report(self) -> dict:
        return self._s.record_report()

    def sample(self, shots: int, *, separate_observables: bool = False,
               append_observables: bool = False,
               return_expectations: bool = False, return_measurements: bool = False,
               bit_packed: bool = False, threads=1) -> "np.ndarray | tuple[np.ndarray, ...]":
        """One engine run -> dets[, obs].

        dets bool[shots, D] detection events; obs bool[shots, O] observable flips
        (``separate_observables=True``), or appended as extra columns (``append_observables=True``,
        Stim's combined ``(shots, D+O)`` layout; mutually exclusive with separate_observables).
        With bit_packed=True the records come Stim-b8-packed (uint8[shots, ceil(K/8)], little-endian
        bit order). Returns the dets array alone unless observables are requested separately.
        threads: an int >= 1 or "auto" (the CPUs this process may use): the shots run on up to that many threads,
        byte-identical for every count (record_report()['threads'] gives the effective count)."""
        if append_observables and separate_observables:
            raise ValueError(
                "pass at most one of append_observables / separate_observables")
        if return_expectations:                    # J-b1 (OD-B): refused by name, no thunk, no fallback
            raise XtimTwirlRefusal(_EXPECTATIONS_TEXT, TwirlClass.EXPECTATIONS_NOT_SERVED)
        if return_measurements:
            raise XtimTwirlRefusal(_RAW_RECORDS_TEXT, TwirlClass.RAW_RECORDS_NOT_SERVED)
        # (dets, obs) 2-tuple: the engine was compiled with decisions=False (no DECISION
        # output channel exists -- see compile_detector_sampler), exactly as for a
        # decision-free circuit; decision bits stay available via xtim.port / sample_barrier.
        dets_p, obs_p = self._s.sample(int(shots), seed=self._seed, threads=threads)
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
