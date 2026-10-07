"""Typed exceptions for xtim — mirroring run_stim_main's exit-code cases.

CLI mapping (cpp/apps/run_stim_main.cpp):
  exit 2 (parse errors, line-numbered)   -> XtimParseError
  exit 3 (pipeline reject)               -> XtimRejectError
  exit 4 (not DEM-expressible, --dem)    -> XtimDemError

The three reject exceptions (`XtimParseError`, `XtimRejectError`, `XtimDemError`)
each carry a `.hint` (an actionable, student-facing fix) and a `.location` (line /
deferred gate index / DEM column, labeled HONESTLY — never a fabricated source line),
and a `__str__` that prints cause + location + hint. (`XtimReferenceError` and
`XtimStimMissingError` are plain messages without `.hint`/`.location`.) The
CLI prints `str(e)` and keys the exit code off the exception TYPE, so the enriched
text rides along on stderr while the 2/3/4 contract stays pinned (v0.2 F2).

XtimStimMissingError derives from ImportError so that a missing optional `stim`
dependency surfaces with the conventional exception type.
"""
from __future__ import annotations


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


class XtimDemError(XtimError):
    """The circuit's error->flip map is not expressible as a Stim DEM (CLI exit 4).

    Built from the engine's refusal message, which names the DEM column (L<k>), the
    error location, and the mechanism. `.hint` tailors the fix to the cause:
    a dressed expectation column → export with include_expectations=False and
    post-select that column; a non-deterministic detector/observable → it carries
    gauge randomness and can't be a DEM target."""

    def __init__(self, engine_message: str, *, hint: str | None = None):
        self.engine_message = str(engine_message)
        # The engine already suffixes "; not DEM-expressible" on some messages, and the
        # cause line below prefixes "not DEM-expressible: " — strip the redundant tail so
        # the phrase isn't printed twice.
        if self.engine_message.endswith("; not DEM-expressible"):
            self.engine_message = self.engine_message[:-len("; not DEM-expressible")]
        msg = self.engine_message
        low = msg.lower()
        # location: the engine message already embeds the offending location/column.
        self.location = msg if msg else None

        if hint is None:
            if "graphlike" in low or "decompose" in low:
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
            ):
                hint = (
                    "a declared OBSERVABLE/detector is not deterministic on the noiseless "
                    "reference; it carries gauge randomness and can't be a DEM target. "
                    "For the faithful magic protocols this is inherent (probabilistic "
                    "syndrome reads) — those circuits are post-selection-only, not "
                    "decoder-in-the-loop. Otherwise make it a deterministic parity "
                    "(combine records so the noiseless value is fixed) or drop it."
                )
            else:
                hint = (
                    "this error mechanism dresses a declared PAULI_EXPECTATION column "
                    "beyond a sign, so that column can't be a DEM L-column. Export the "
                    "detector/observable DEM with `include_expectations=False` and "
                    "post-select that column instead of decoding it. (Caveat: if the "
                    "detector/observable mechanisms are themselves probabilistic, even "
                    "that DEM refuses — the circuit is then post-selection-only.)"
                )
        self.hint = hint
        cause = f"not DEM-expressible: {self.engine_message}"
        super().__init__(_compose(cause, None, self.hint))


class XtimReferenceError(XtimError):
    """Explicit reference operation failed (compile battery / load / save)."""


class XtimStimMissingError(XtimError, ImportError):
    """`stim` is required for detector_error_model() but is not installed."""


class XtimCacheWarning(UserWarning):
    """Reference-cache trouble (tamper, unreadable, unwritable). The run stays
    correct — xtim falls back to the engine's deduced bare state or an in-memory
    compiled reference — but caching did not work as intended."""


class XtimPerformanceWarning(UserWarning):
    """A slow diagnostic selector is active. See the warning message for details."""
