"""Typed exceptions for xtim — mirroring run_stim_main's exit-code cases.

CLI mapping (cpp/apps/run_stim_main.cpp):
  exit 2 (parse errors, line-numbered)   -> XtimParseError
  exit 3 (pipeline reject)               -> XtimRejectError
  exit 4 (not DEM-expressible, --dem)    -> XtimDemError

The three reject exceptions (`XtimParseError`, `XtimRejectError`, `XtimDemError`)
each carry a `.hint` (an actionable, student-facing fix) and a `.location` (line /
deferred gate index / DEM column, labeled HONESTLY — never a fabricated source line),
and a `__str__` that prints cause + location + hint. (`XtimStimMissingError` is a
plain message without `.hint`/`.location`.) The
CLI prints `str(e)` and keys the exit code off the exception TYPE, so the enriched
text rides along on stderr while the 2/3/4 exit-code contract stays pinned.

XtimStimMissingError derives from ImportError so that a missing optional `stim`
dependency surfaces with the conventional exception type.
"""
from __future__ import annotations

import enum


def _compose(cause: str, location: str | None, hint: str | None) -> str:
    """cause + location + hint, each on its own labeled line (skip empties)."""
    parts = [cause]
    if location:
        parts.append(f"  at: {location}")
    if hint:
        parts.append(f"  hint: {hint}")
    return "\n".join(parts)


class XtimError(Exception):
    """Base class for all xtim errors."""


class XtimParseError(XtimError):
    """Circuit parse failure. `errors` is a list of (line, message), 1-based lines.

    `.hint` advises on the nearest-instruction / token context when cheaply
    available from the parse error message; `.location` names the failing line(s)."""

    def __init__(self, errors: list[tuple[int, str]], *, hint: str | None = None):
        self.errors = list(errors)
        self.location = (
            ", ".join(f"line {line}" for line, _ in self.errors) or None
        )
        if hint is None:
            hint = ("fix the flagged line(s); each message names the exact problem "
                    "(token, argument, or range).")
            # Only volunteer the "typo / out-of-dialect" guess when it's actually an
            # unknown name — otherwise it misdirects (e.g. a rec[-k] range error).
            if any("unknown instruction" in msg.lower() for _, msg in self.errors):
                hint += (" The dialect is a strict Stim superset (plus "
                         "T/T_DAG/CS/CS_DAG/CCZ and PAULI_EXPECTATION) — an unknown "
                         "name is usually a typo or an out-of-dialect instruction.")
        self.hint = hint
        cause = (
            "\n".join(f"line {line}: {msg}" for line, msg in self.errors)
            or "parse failed"
        )
        super().__init__(_compose(cause, self.location, self.hint))


class XtimRejectError(XtimError):
    """Pipeline reject (CLI exit 3): the circuit left the v1 simulable class at a
    gate, or a supplied reference mismatched.

    `.kind` distinguishes the causes:
      - "class"        : a propagation-class rejection. When `gate_index` >= 0 it
                         names the offending deferred-stream gate (the SOLE localized
                         rejection criterion). When `gate_index` == -1 the rejection
                         is non-gate-localized (propagation-class, no reference
                         supplied).
      - "ref_mismatch" : a supplied .ref has a different qubit count than the circuit

    `.gate_index` is the deferred-stream gate index for localized "class" rejects,
    else -1.
    `.location` names the deferred gate index (honestly labeled) or the n-mismatch."""

    def __init__(
        self,
        gate_index: int,
        message: str | None = None,
        *,
        kind: str | None = None,
        ref_n: int | None = None,
        circuit_n: int | None = None,
    ):
        self.gate_index = int(gate_index)
        # Default to "class" for any non-ref_mismatch reject.
        if kind is not None:
            self.kind = kind
        else:
            self.kind = "class"
        self.ref_n = ref_n
        self.circuit_n = circuit_n

        self.location: str | None
        if self.kind == "class":
            if self.gate_index >= 0:
                self.location = f"deferred-stream gate index {self.gate_index}"
                cause = (
                    f"rejected: the circuit is outside the simulable class at "
                    f"{self.location}"
                )
            else:
                self.location = "non-gate-localized (non-commuting magic)"
                cause = (
                    "rejected: the circuit is outside the simulable class "
                    f"({self.location})"
                )
            self.hint = (
                "xtim simulates circuits whose non-Clifford gates fold (through the "
                "intervening Cliffords) into ONE mutually-commuting layer of π/8 "
                "Pauli-product rotations; this circuit's don't — 'non-commuting magic'. "
                "Common causes and fixes: (1) you wrote `H` then `M` to read a "
                "magic-carrying wire in the X/Y basis — write `MX`/`MY` directly; "
                "(2) a magic state was fed into a same-wire NON-COMMUTING check (e.g. a "
                "raw T-state into a controlled-H): Clifford-align the state's magic axis "
                "with the measured check first (for the H-eigenstate, prepare "
                "|H> = (H S H) T H |0>); (3) genuinely sequential non-commuting magic "
                "(T-depth ≥ 2 after folding) is not supported. `CH` itself IS "
                "supported whenever its magic commutes with the rest. (The index, when "
                "≥ 0, is the post-deferral stream position, not a source line. See "
                "docs/xtim_simulable_class.md.)"
            )
        elif self.kind == "ref_mismatch":
            self.location = None
            if ref_n is not None and circuit_n is not None:
                self.location = (
                    f"reference (deferred) n={ref_n}, circuit user-qubit n={circuit_n}"
                )
            cause = "rejected: the supplied reference does not match this circuit"
            rn = "?" if ref_n is None else str(ref_n)
            cn = "?" if circuit_n is None else str(circuit_n)
            self.hint = (
                f"the supplied reference declares a deferred space of n={rn} qubits, "
                f"which does not match this circuit's deferred space (its {cn} user "
                "qubits expand under measurement deferral) — the .ref was compiled for "
                "a DIFFERENT circuit. Recompile the reference for THIS circuit (or drop "
                "the supplied reference to use the deduced bare state)."
            )
        else:
            # Unknown kind: default to class-reject messaging.
            self.kind = "class"
            self.location = None
            cause = "rejected: propagation-class rejection"
            self.hint = (
                "the circuit is outside the simulable class (its non-Clifford gates do "
                "not fold into one mutually-commuting π/8 layer). Common fixes: "
                "(1) replace `H` then `M` with `MX`/`MY`; (2) Clifford-align a magic "
                "state's axis with the check that measures it. See "
                "docs/xtim_simulable_class.md."
            )

        if message is not None:
            cause = message
        super().__init__(_compose(cause, self.location, self.hint))


class TwirlClass(enum.Enum):
    """The classes of circuits / requests the twirl engine refuses BY NAME (by design: no silent fallback).
    Carried as ``XtimTwirlRefusal.twirl_class``.

    - OUTSIDE_T_DEPTH_1: the circuit leaves the single commuting magic layer (sub_reason "bare": the bare state's
      pushed magic axes anticommute noiselessly).
    - NONCOMMUTING_MAGIC: the static fixed-axis rule: terms that anticommute only through a fault (the
      surviving axis / Clifford frame depends on the faults) — a LAW LIMITATION, not physics.
    - LAW_LIMITATION_PROPAGATION: the static rule ACCEPTS the circuit (T-depth <= 1), but the gate-by-gate
      propagation table meets a same-axis non-Clifford pair before it merges (a fault before the pair); a law
      limitation.
    - LAW_LIMITATION_CARRIED_AXIS: a carried producer branch (test API) whose law the branch law does not build;
      a law limitation. ``sub_reason``: "branch_frame" (the branch's
      materialize frame differs from the ideal post-barrier frame: a producer coin read hits a dressed row),
      "carried_free_x" / "carried_branch_x" (a combination of the consumer's axes has its X-part only on the
      producer's free rows / on the branch's extended E_min rows).
    - EXPECTATIONS_NOT_SERVED: ``sample(return_expectations=True)`` on the twirl engine.
    - RAW_RECORDS_NOT_SERVED: ``sample(return_measurements=True)`` on the twirl engine."""

    OUTSIDE_T_DEPTH_1 = "OUTSIDE_T_DEPTH_1"
    NONCOMMUTING_MAGIC = "NONCOMMUTING_MAGIC"
    LAW_LIMITATION_PROPAGATION = "LAW_LIMITATION_PROPAGATION"
    LAW_LIMITATION_CARRIED_AXIS = "LAW_LIMITATION_CARRIED_AXIS"
    EXPECTATIONS_NOT_SERVED = "EXPECTATIONS_NOT_SERVED"
    RAW_RECORDS_NOT_SERVED = "RAW_RECORDS_NOT_SERVED"


class XtimTwirlRefusal(XtimError, ValueError):
    """The twirl engine refuses this circuit or request BY NAME. ``twirl_class`` (a :class:`TwirlClass`)
    says which; ``sub_reason`` refines OUTSIDE_T_DEPTH_1 ("bare") and LAW_LIMITATION_CARRIED_AXIS (see above);
    else "". A ValueError, so ``except ValueError`` callers keep working. The request is outside xtim's model;
    nothing is approximated. For the end state, declare OUTPUT_QUBITS and score it with ``RecordFidelity``."""

    def __init__(self, message: str, twirl_class, sub_reason: str = ""):
        self.twirl_class = twirl_class if isinstance(twirl_class, TwirlClass) else TwirlClass(twirl_class)
        self.sub_reason = sub_reason or ""
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (str(self), self.twirl_class.value, self.sub_reason))


class XtimFidelityRefusal(XtimError, ValueError):
    """``RecordFidelity``, or an output-state helper, refuses this circuit or evaluation BY NAME. ``code`` is the
    refusal's name (``"RF_NO_OUTPUT"``, ``"RF_OUTPUT_NOT_STRICT"``, ``"RF_OUTPUT_MEASURED"`` (raised by
    ``compile_twirl_sampler`` and the output-state helpers), ``"RF_RECORD_ON_OUTPUT"``, ``"RF_CORRECTION_OFF_OUTPUT"``,
    ``"RF_SECTOR_RANK"``, ...). The output-state helpers (:func:`xtim.output_is_deterministic`,
    :func:`xtim.output_state`) also raise ``"RF_OUTPUT_UNDECIDED"``, and :func:`xtim.output_state` raises
    ``"RF_OUTPUT_NOT_DETERMINISTIC"`` (``output_is_deterministic(..., explain=True)`` returns the same message). The
    message says what to change. A ValueError, like :class:`XtimTwirlRefusal`."""

    def __init__(self, message: str, code: str):
        self.code = str(code)
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (str(self), self.code))


_MAX_TARGET_RUNS = 20   # the hint names at most this many id runs; XtimDemError.nondeterministic has them all


def _name_targets(names) -> str:
    """"D3 D4 D5 D9 L0" -> "detectors D3-D5, D9 and observable L0" (consecutive ids as ranges; at most
    _MAX_TARGET_RUNS runs, then "...")."""
    groups = {"D": [], "L": []}
    for nm in names:
        groups[nm[0]].append(int(nm[1:]))
    parts = []
    budget = _MAX_TARGET_RUNS
    for kind, word in (("D", "detector"), ("L", "observable")):
        ids = sorted(groups[kind])
        if not ids:
            continue
        runs, lo = [], ids[0]
        for i, cur in enumerate(ids):
            if i + 1 == len(ids) or ids[i + 1] != cur + 1:      # a run ends at cur
                runs.append(f"{kind}{lo}" if lo == cur else f"{kind}{lo}-{kind}{cur}")
                if i + 1 < len(ids):
                    lo = ids[i + 1]
        if budget <= 0:
            parts.append(f"{word}{'s' if len(ids) > 1 else ''} ...")
            continue
        shown = runs[:budget] + (["..."] if len(runs) > budget else [])
        budget -= len(runs)
        parts.append(f"{word}{'s' if len(ids) > 1 else ''} {', '.join(shown)}")
    return " and ".join(parts)


class XtimDemError(XtimError):
    """The circuit's error->flip map is not expressible as a Stim DEM (CLI exit 4).

    Built from the engine's refusal message, which names the DEM column (L<k>), the
    error location, and the mechanism. `.hint` tailors the fix to the cause:
    a channel with no exact independent-mechanism conversion → Stim's
    `approximate_disjoint_errors` refusal (same wording; pass
    approximate_disjoint_errors=True); a fault acting non-Pauli on a declared
    expectation column → export with include_expectations=False and post-select that
    column; a non-deterministic detector/observable → it carries gauge randomness and
    can't be a DEM target (`.nondeterministic` lists every such detector/observable, "D<k>" then "L<k>", when the
    exporter reports them; the hint names them and the fix). An unrecognized cause gets no hint."""

    def __init__(self, engine_message: str, *, hint: str | None = None, nondeterministic=None):
        self.engine_message = str(engine_message)
        self.nondeterministic = list(nondeterministic or [])
        # The engine already suffixes "; not DEM-expressible" on some messages, and the
        # cause line below prefixes "not DEM-expressible: " — strip the redundant tail so
        # the phrase isn't printed twice.
        if self.engine_message.endswith("; not DEM-expressible"):
            self.engine_message = self.engine_message[:-len("; not DEM-expressible")]
        msg = self.engine_message
        low = msg.lower()
        # location: the engine message already embeds the offending location/column.
        self.location = msg if msg else None

        # Stim-shaped disjoint-channel refusal ("Encountered the operation <NAME> during
        # error analysis, but this operation requires the `approximate_disjoint_errors`
        # option to be enabled. ..."): the message itself is the lead line, as in Stim.
        stim_shaped = msg.startswith("Encountered the operation ")
        if hint is None:
            if stim_shaped and "approximate_disjoint_errors" in msg:
                hint = (
                    "this noise channel's record-flip law is not a product of independent "
                    "mechanisms (common when one fault randomizes several correlated reads), "
                    "so an exact DEM would need a negative probability. Pass "
                    "`approximate_disjoint_errors=True` (or a float threshold) to approximate "
                    "it the way Stim does."
                )
            elif "graphlike" in low or "decompose" in low:
                hint = (
                    "this error mechanism could not be decomposed into graphlike "
                    "(<=2-detector) pieces, which `decompose_errors=True` requires for "
                    "matching decoders. Pass `ignore_decomposition_failures=True` to keep "
                    "the undecomposed mechanism (as Stim does), or use a decoder that "
                    "doesn't need graphlike errors. (This is NOT about PAULI_EXPECTATION "
                    "columns — `include_expectations=False` will not help.)"
                )
            elif "non-fault-tolerant" in low:
                hint = (
                    "a single fault changes a logical expectation's MAGNITUDE but fires "
                    "no detector, so it can't be post-selected away — an undetectable "
                    "logical error (a fault-tolerance gap in the circuit, not a tool "
                    "limitation). Declare the missing post-selection syndrome as "
                    "DETECTORs (the cultivation check rounds AND the final output-code "
                    "stabilizer round); then the fault flips one and joins the reject "
                    "region. (Not a PAULI_EXPECTATION-dressing issue — include_expectations "
                    "won't help.)"
                )
            elif "categorical" in low or "fourier coefficient" in low or "over-mix" in low:
                hint = (
                    "a noise channel is so strong it can't be written as an independent "
                    "DEM mechanism (a Fourier coefficient of its signature distribution is "
                    "≤ 0 — e.g. a fully-mixing `DEPOLARIZE(1.0)`). This is a saturated NOISE "
                    "channel, NOT a PAULI_EXPECTATION issue. Reduce the channel's strength — "
                    "a common cause is `scale_noise` with the wrong `p0` driving a "
                    "probability to ~1 (check p/p0), or a hand-written p that is too large."
                )
            elif "fair-coin" in low or "fair coin" in low:
                hint = (
                    "a high-weight Pauli product (an MPP, or a Pauli read used as an "
                    "OBSERVABLE_INCLUDE) is twirled by a single error into more "
                    "measurement-coin reads than the exact DEM walk can expand (its 2^k "
                    "subset sum overflows the engine's weight budget). This is a DEM-export "
                    "limit, NOT a PAULI_EXPECTATION issue — `include_expectations=False` "
                    "does not help. Either reduce the offending observable/MPP weight "
                    "(split the >16-qubit product into smaller pieces) or treat the circuit "
                    "as post-selection / raw-sampling only (it still samples exactly via "
                    "`compile_detector_sampler`)."
                )
            elif "non-deterministic" in low and (
                "observable" in low or "detector" in low
            ) and self.nondeterministic:
                hint = (
                    f"{_name_targets(self.nondeterministic)} "
                    f"{'is' if len(self.nondeterministic) == 1 else 'are'} not deterministic on the noiseless "
                    f"circuit ({len(self.nondeterministic)} in all); such a target carries gauge randomness and "
                    "can't be a DEM target. If the circuit measures Pauli byproducts (teleportation, gauge or "
                    "magic-state-injection outcomes) and these targets read them, add the Pauli feedback that "
                    "corrects them (CX/CY/CZ rec[-k] q), or drop those detectors/observables. For the faithful "
                    "magic protocols this is inherent (probabilistic syndrome reads): those circuits are "
                    "post-selection-only, not decoder-in-the-loop."
                )
            elif "non-deterministic" in low and (
                "observable" in low or "detector" in low
            ):
                hint = (
                    "a declared OBSERVABLE/detector is not deterministic on the noiseless "
                    "reference; it carries gauge randomness and can't be a DEM target. "
                    "For the faithful magic protocols this is inherent (probabilistic "
                    "syndrome reads) — those circuits are post-selection-only, not "
                    "decoder-in-the-loop. Otherwise make it a deterministic parity "
                    "(combine records so the noiseless value is fixed) or drop it."
                )
            elif "non-pauli on a logical expectation" in low:
                hint = (
                    "this error mechanism dresses a declared PAULI_EXPECTATION column "
                    "beyond a sign, so that column can't be a DEM L-column. Export the "
                    "detector/observable DEM with `include_expectations=False` and "
                    "post-select that column instead of decoding it. (Caveat: if the "
                    "detector/observable mechanisms are themselves probabilistic, even "
                    "that DEM refuses — the circuit is then post-selection-only.)"
                )
        self.hint = hint
        cause = (self.engine_message if stim_shaped
                 else f"not DEM-expressible: {self.engine_message}")
        super().__init__(_compose(cause, None, self.hint))


def _rebuild_dem_error(engine_message, hint, nondeterministic):
    return XtimDemError(engine_message, hint=hint, nondeterministic=nondeterministic)


XtimDemError.__reduce__ = lambda self: (_rebuild_dem_error, (self.engine_message, self.hint, self.nondeterministic))


class XtimXdemError(XtimError):
    """An xdem text was refused by the reader (docs/xdem.md §8): a structural or spelling error (unknown version, a header
    field unknown, missing, repeated or out of range, an axis line out of order or after the first error line, an
    undeclared axis or coordinate, a non-canonical spelling, an axis with no qubit, an x/rot/late field in a file with
    no xdem header, more than 2^31 - 1 readout columns), the header condition rank[img | kappa] = rank kappa, a line
    that is not an operator, or a dc-DEM v1 file (re-export the circuit with format="xdem"). Also raised by
    `XdemSampler.score(lines=, correction="r2b")` / `correction(correction="r2b")` on a file whose `r2b_available` is
    False, naming that reason. A subclass of `XtimError`, not of `ValueError`. The message names the cause and, for a
    reader refusal, the text line."""


class XtimGpuError(XtimError):
    """The GPU sampler (`XdemSampler(text, device=...)`, `xtim.gpu`) cannot serve the request, and says why: no usable
    device (no Metal device or no pyobjc: `pip install "xtim[metal]"`; a CUDA backend that is not available), a file
    the GPU kernels do not support (the message names the numbers, e.g. `5000 axes`), or a call the GPU sampler
    does not offer yet (the fired lines of `sample(fired=True)` / `score(lines=)` / `correction()` /
    `infidelity(lines=True)`; `postselect=` is served). `device="cpu"` (the default) serves all of them. A subclass of
    `XtimError`, not of `ValueError` or `RuntimeError`."""


class XtimStimMissingError(XtimError, ImportError):
    """`stim` is required for detector_error_model() but is not installed."""


class XtimCapacityError(XtimError, MemoryError):
    """The circuit's bare state cannot be represented: it needs chi = 2^r stabilizer
    branches with r above the engine limit (2^30), or the allocation at that chi failed.

    Raised by every compile/build path (twirl, DEM); no engine can
    represent the state, so there is nothing to reroute to. The message names r and the limit."""


class XtimInvariantError(XtimError):
    """An engine invariant failed on a built state (e.g. the bare state's squared norm is
    not 1). The engine refuses rather than sample a wrong state; please report it."""


class XtimWarning(UserWarning):
    """Base of every xtim warning category (the diagnostics policy).

    xtim reports through three channels, never bare stderr (the few C++ stderr lines left are the
    shot-loop oracle-failure copies and QEC_TW_PROF's):

    * METRICS -> the API (e.g. ``TwirlSampler.record_report()``);
    * INFORMATIONAL events -> Python ``logging`` on named loggers (``xtim.twirl``, ...), with a
      ``NullHandler`` on the ``xtim`` logger (installed by ``xtim._diagnostics``, where the
      records are raised; silent unless the application configures logging);
    * CONDITIONS needing attention -> ``warnings`` with a specific subclass of this base:
      ``XtimPerformanceWarning``, ``XtimCorrectnessWarning`` and the older ``XtimDemWarning``
      (``XtimVerificationWarning`` is no longer raised by the production samplers).

    The twirl engine releases the GIL while it samples, so its events are DEFERRED: the shot
    loop only counts (plus a first-occurrence payload) and the warnings / log records are
    raised after the run returns. Under an "error" filter (``python -W error``, or
    ``warnings.filterwarnings("error", category=xtim.XtimPerformanceWarning)``) the run
    therefore completes and the warning raises from the sampling call afterwards (its counters,
    e.g. ``record_report()``, reflect the whole run)."""


class XtimDemWarning(XtimWarning):
    """DEM export left something out on purpose and says so once (e.g. a PAULI_EXPECTATION
    column whose noiseless per-shot value varies shot by shot has no DEM column)."""


class XtimVerificationWarning(XtimWarning):
    """Verification is running in a reduced or diagnostic form, and xtim says so.

    The production samplers no longer raise it: the selfcheck oracle-window notice is an INFO log
    record on the ``xtim.twirl`` logger (see CHANGELOG).

    * (Test API only: the exact engine's reference battery.) A reference self-check ran on fewer
      shots than its full battery and says so once per compile. The work-sized noiseless-invariants battery (verification never costs more than
    compilation) runs fewer than its ceiling (500 at a reference compile, 200 at a cache-hit
    verify) only where every noiseless shot collapses a large block live. The detector verdict
    is still exactly the full battery's; the |PAULI_EXPECTATION| constancy check sees only the
    shots run, so a wobble confined to outcome paths of total probability q is missed with
    probability (1-q)^N. The message gives N, the ceiling, W_compile and w1. Sampling is
    unaffected (the reference passed every gate it ran)."""


class XtimPerformanceWarning(XtimWarning):
    """A slower path than the fast one served the request. See the message for details.

    (xtim-engine-13 removed the two former sources: ``engine="auto"`` choosing the exact
    engine, and the twirl engine's per-shot exact-fallback routes. Neither exists any more:
    a circuit or request the twirl engine cannot serve raises :class:`XtimTwirlRefusal`.)"""


class XtimCorrectnessWarning(XtimWarning):
    """An engine self-check found an actual mismatch (an engine bug, please report it).

    Raised by the twirl engine's self-checks (the plan-key derivation check and the local-index
    oracle) after the aborted run returns, immediately before the ``RuntimeError`` "twirl
    sampler oracle abort: ..." that refuses the run's output; the message names the check and
    the shot index."""
