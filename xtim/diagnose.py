"""xtim.diagnose — the onboarding report (v0.2 F1).

A student who has written their OWN magic-state-prep protocol gets the facts they
would otherwise reverse-engineer by hand (which detectors post-select, the
byproduct frame signs, the |beta| magnitudes, whether an expectation can be a DEM
L-column). `Circuit.diagnose()` does ONE noiseless run and REPORTS measured facts.

BINDING boundary (spec docs/superpowers/specs/2026-06-14-xtim-v0.2-onboarding-design.md):
diagnose REPORTS, never APPLIES. It does not sign-correct the returned data, does
not filter shots, does not mutate the circuit or the reference cache (beyond the
ordinary noiseless `sample()`). The `frame_hint` is labelled a HINT — an advisory
the student MAY feed a decoder, discovered (GF(2) record-parity) but not applied.

The semantics mirror scripts/msp_accounting.py — the by-hand discovery this packages:
  * deterministic detector  := the detector's parity Pauli is a ±1 eigen-operator (a
    stabilizer) of the noiseless bare state — checked EXACTLY by the engine
    (`_xtim.detector_determinism`), shot-free, and independent of the noise. This is the
    post-selectable set (a gauge detector's parity is a fair coin noiselessly and cannot be
    post-selected). Classically-controlled feedback is folded exactly (same relabel as the
    sampler), so every circuit the sampler accepts is classified. There is no statistical
    fallback: in the rare case the exact analysis cannot run (an acausal feedback, a shared-wire
    read), diagnose refuses loudly rather than sample.
  * abs_value                := mean |noiseless exps[:, i]| (~ |beta_i|); asserted
    near-constant across shots (a magic channel's MAGNITUDE is constant noiselessly
    even when its SIGN flips with a byproduct frame).
  * signed_value             := the byproduct-frame-CORRECTED beta_i — the target
    expectation <P_i> of the ideal output state, in the canonical (zero-frame) frame.
    Because the circuit now DECLARES its byproduct frame (`PAULI_EXPECTATION(i) <P> rec[-k]...`),
    the engine FOLDS that frame into the per-shot sign, so the noiseless exps[:, i] is
    sign-CONSTANT and signed_value is just that constant sign * abs_value. nan when the
    magnitude is ill-defined, or when the sign still varies (a residual free/logical read,
    e.g. an MPP terminal readout — no fixed byproduct frame governs it). The vector of
    signed_values across columns IS the target Bloch / beta combination.
  * frame_hint               := the DECLARED byproduct-frame records (the `rec[-k]` set on
    the PAULI_EXPECTATION line, echoed exactly from the exact frame-verify binding
    `_xtim.expectation_frames`), or None if the column declares no frame. This is the
    circuit's own declaration, NOT a statistically-discovered guess; diagnose REFUSES (loud)
    if a column has a solvable sign-controlling frame but its declared frame does not SPAN it.
  * dem_expressible          := whether the WHOLE-circuit DEM export succeeds — reusing
    detector_error_model()'s existing refusal (no per-column probe exists; the
    whole-export refusal is the source of truth). A False here is a circuit-level
    fact: the refusal reason MAY be unrelated to the specific column it sits on (a
    non-deterministic observable, or another column's dressing). Not a per-column
    "this expectation is ±1-expressible" verdict — read dem_reason for what refused.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _xtim
from .errors import XtimDemError, XtimError, XtimParseError, XtimRejectError

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import Circuit

# Noise instructions of the benchmark dialect — stripped (rescaled to 0) for the
# noiseless run. Kept in sync with xtim.collect._NOISE_LINE / scripts/msp_accounting.
_NOISE_PREFIXES = (
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "PAULI_CHANNEL",
)

# Measurement instructions that can carry a leading readout-error probability, e.g.
# `MX(0.001) 2 4`. Unlike a standalone noise CHANNEL, a measurement EMITS A RECORD,
# so its line cannot be deleted (that would renumber every downstream rec[-k] and
# corrupt the detectors/observables). To make the circuit noiseless we instead drop
# the probability argument, keeping the measurement and its record intact.
_MEAS_PREFIXES = (
    "M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP",
)


def _strip_noise(text: str) -> str:
    """Return the genuinely-noiseless circuit for `text`.

    Two kinds of noise are neutralised:

    * A standalone noise CHANNEL line (``DEPOLARIZE1``, ``X_ERROR``, …) is deleted —
      it emits no record and adds no qubit, so the deterministic skeleton
      (records/detectors/observables/expectations, qubit count) is unchanged.

    * Readout error baked into a MEASUREMENT (``MX(0.001) …``) cannot be removed by
      deleting the line — the measurement emits a record and deleting it would
      renumber every downstream ``rec[-k]``. Instead the leading probability
      argument is stripped (``MX(0.001) …`` -> ``MX …``), leaving a noiseless
      measurement with its record and all indices intact. This is what makes the
      determinism test see a syndrome bit fed by a noisy readout as the deterministic
      detector it truly is (it fires only on a readout fault, at rate ~p), rather
      than misclassifying it as a gauge bit because it flipped once under noise.
    """
    out = []
    for line in text.splitlines():
        # Head the way the stim tokenizer sees it: uppercase the [A-Za-z0-9_] name;
        # a [tag] suffix and the arg-paren are outside the name. (The parser accepts
        # lowercase names and tags; matching them here keeps stripping correct for
        # the full accepted dialect instead of silently leaving noise in place.)
        m = re.match(r"\s*([A-Za-z0-9_]+)", line)
        head = m.group(1).upper() if m else ""
        if head in _NOISE_PREFIXES:
            continue
        if head in _MEAS_PREFIXES:
            # Drop the readout-error probability argument (if any); keep the record.
            # Match the name case-insensitively with an optional [tag], the way the
            # parser tokenizes it.
            line = re.sub(r"^(\s*[A-Za-z0-9_]+(?:\[[^\]]*\])?)\([^)]*\)", r"\1", line, count=1)
        out.append(line)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class ExpectationInfo:
    """Per declared PAULI_EXPECTATION column (index = declaration order)."""

    index: int
    abs_value: float                 # mean |noiseless exps[:, index]| (~ |beta|); nan if ill-defined
    signed_value: float              # the byproduct-frame-corrected beta_i (the TARGET <P_i>); nan if the
                                     # canonical sign is indeterminate (sign varies, no record-parity explains it)
    sign_constant: bool              # does the raw sign hold across the noiseless run? (True once the
                                     # declared frame is folded; False for a residual free/logical read)
    well_defined: bool = True        # |value| near-constant noiselessly (a "dead channel" is False)
    frame_hint: list[int] | None = None   # the DECLARED byproduct-frame records (echo of the circuit's
                                          # rec[-k] set), or None if the column declares no frame
    frame_const: int | None = None        # the folded canonical sign bit (0:+, 1:-) for a declared frame
    dem_expressible: bool = True     # did the WHOLE-circuit DEM export succeed? (see note)
    dem_reason: str | None = None    # the whole-export refusal message, when False
    # NOTE on dem_expressible: this mirrors detector_error_model()'s whole-circuit
    # refusal (the source of truth — there is no per-column probe), so it is the SAME
    # value on every column. It is NOT an isolated "this column is ±1-expressible"
    # verdict; do not read it as one. __str__ surfaces it ONCE, demoted to a benign
    # bottom note (no exact Pauli DEM is EXPECTED for magic-state prep), rather than
    # tagging each value line — a refusal there read as "the run failed" when it had not.

    def __repr__(self) -> str:
        # Compact repr: the full 9 fields (incl. a multi-line dem_reason) bloat the
        # interactive display; the values stay on the attributes.
        v = "nan" if self.signed_value != self.signed_value else "%+.6g" % self.signed_value
        return ("ExpectationInfo(index=%d, signed_value=%s, sign_constant=%r, "
                "dem_expressible=%r)" % (
                    self.index, v, self.sign_constant, self.dem_expressible))


@dataclass(frozen=True)
class Diagnosis:
    """The onboarding report for one circuit — measured facts, no interpretation."""

    shots: int
    chi: int | None
    chi_source: str                  # "cached" | "deduced"
    num_detectors: int
    deterministic_detectors: list[int]
    gauge_detectors: list[int]
    expectations: list[ExpectationInfo]
    suggested_postselect: str        # advisory numpy recipe (a STRING, never applied)

    def __repr__(self) -> str:
        # Compact one-liner so interactive auto-display isn't a giant nested dump;
        # print(diagnosis) / str() gives the full formatted report.
        return ("Diagnosis(shots=%d, chi=%s, detectors=%d, expectations=%d)" % (
            self.shots, self.chi, self.num_detectors, len(self.expectations)))

    def _repr_html_(self) -> str:
        # Jupyter rich display: show the full formatted report (not the compact repr).
        import html as _html
        return "<pre style='line-height:1.3'>%s</pre>" % _html.escape(str(self))

    def __str__(self) -> str:
        lines = ["xtim diagnosis (noiseless run, %d shots)" % self.shots,
                 "=" * 52]
        # A cold cache reports chi as unknown — say so plainly so the bare "?" doesn't
        # read as a failure (it is deduced on the fly; any compile_* warms it).
        if self.chi is None:
            lines.append("reference chi : not yet cached (deduced on the fly — fine; "
                         "run any compile_* to cache it)")
        else:
            lines.append("reference chi : %s (%s)" % (self.chi, self.chi_source))
        lines.append("detectors     : %d total — %d deterministic, %d gauge"
                     % (self.num_detectors, len(self.deterministic_detectors),
                        len(self.gauge_detectors)))
        lines.append("  deterministic: %s" % (self.deterministic_detectors or "(none)"))
        lines.append("  gauge        : %s" % (self.gauge_detectors or "(none)"))
        if self.expectations:
            lines.append("expectations  : %d declared PAULI_EXPECTATION column(s)"
                         % len(self.expectations))
            for e in self.expectations:
                # report the SIGNED (frame-corrected) target beta_i when it is
                # determined; fall back to the magnitude when the sign is indeterminate.
                if e.signed_value == e.signed_value:  # not nan
                    line = "  [%d] value=%+.12g" % (e.index, e.signed_value)
                else:
                    av = "nan" if not (e.abs_value == e.abs_value) else "%.12g" % e.abs_value
                    line = "  [%d] |value|=%s" % (e.index, av)
                if not e.well_defined:
                    line += "  ILL-DEFINED (|value| not constant noiselessly)"
                elif e.frame_hint is not None:
                    # declared byproduct frame — folded by the engine, so the sign is constant
                    line += ("  declared byproduct frame (folded); records %s"
                             % (e.frame_hint,))
                elif e.sign_constant:
                    line += "  sign constant"
                else:
                    line += "  sign varies (free/logical read; no fixed byproduct frame)"
                lines.append(line)
            # the signed beta combination = the target Bloch vector, when fully determined
            betas = [e.signed_value for e in self.expectations]
            if betas and all(b == b for b in betas):  # no nans
                lines.append("  target (signed beta): [%s]"
                             % ", ".join("%+.6g" % b for b in betas))
            # The whole-circuit DEM refusal (if any) is a CIRCUIT-level fact, not a
            # per-column failure — and for magic-state prep it is EXPECTED, not an
            # error. Emit it once, demoted, framed as benign: the magic value is read
            # from the expectation channel above (not decoded), so a missing Pauli DEM
            # changes nothing about sampling. Keep the technical reason as a detail.
            no_dem = next((e for e in self.expectations if not e.dem_expressible), None)
            if no_dem is not None:
                lines.append("note: no exact Pauli detector-error-model for this circuit"
                             " — expected for magic-state prep. The magic value(s) above are")
                lines.append("      READ from the expectation channel, not decoded; sampling"
                             " stays exact. Post-select as suggested below; nothing is wrong.")
                if no_dem.dem_reason:
                    # Just the core first line — the full reason (with the engine's
                    # own multi-line hint) stays on ExpectationInfo.dem_reason for
                    # anyone who wants it; here it would re-create the wall we demoted.
                    detail = no_dem.dem_reason.split("\n", 1)[0].strip()
                    lines.append("      (DEM detail: %s)" % detail)
        else:
            lines.append("expectations  : (none declared)")
        lines.append("suggested post-selection (advisory; not applied):")
        lines.append("  %s" % self.suggested_postselect)
        if self.expectations and self.deterministic_detectors:
            lines.append("  (this recipe post-selects ALL deterministic detectors — the"
                         " conservative policy. For a decoder DEM that rejects only the")
            lines.append("   not-Pauli-correctable faults and decodes the rest, see"
                         " Circuit.detector_error_model_with_reject() — docs/xtim_dem_reject.md.)")
        # A short legend so the report is self-teaching, not expert-only.
        lines.append("legend: deterministic detector = always 0 noiselessly (post-select"
                     " on it); gauge detector = flips randomly (leave it);")
        lines.append("        signed beta = the target expectation <P_i> of your magic"
                     " state; the declared byproduct frame is ALREADY folded into it by")
        lines.append("        the engine (the per-shot exps sign is deterministic) — read"
                     " exps directly, do not re-apply the frame.")
        return "\n".join(lines)


# ------------------------------------------------------------------------- driver
def diagnose(circuit: "Circuit", *, shots: int = 1024, seed: int = 0) -> Diagnosis:
    """One noiseless run -> a Diagnosis of `circuit` (see module docstring).

    Reports facts; applies nothing. Side-effect-free: the returned data is raw, the
    circuit and reference cache are untouched beyond the ordinary noiseless sample.
    """
    from .circuit import Circuit  # local import: avoid a circular import at module load

    if shots < 1:
        raise ValueError("diagnose needs shots >= 1 (it measures facts over a run)")

    # chi + cache status from the ORIGINAL circuit's reference (the run uses the same).
    info = circuit.reference_info()
    chi = info.get("chi")
    chi_source = "cached" if info.get("cached") is True else "deduced"

    # one noiseless run. The DECLARED byproduct frames are folded into the sign by the engine,
    # so exps[:, i] is sign-CONSTANT for every frame-governed column (the raw magnitude and the
    # frame-corrected sign both come straight from this run — no statistical frame inference).
    noiseless = Circuit(_strip_noise(circuit.text), _source=circuit._source)
    sampler = noiseless.compile_detector_sampler(seed=int(seed))
    dets, exps = sampler.sample(
        int(shots), separate_observables=False, return_expectations=True)

    # detectors: deterministic iff the detector's parity Pauli is a ±1 eigen-operator (a
    # stabilizer) of the noiseless bare state. This is the EXACT, shot-free verdict from the
    # engine (`_xtim.detector_determinism`) — the source of truth, shared with the DEM-export
    # determinism gate, and independent of how well the circuit's noise can be stripped. It
    # answers even when the full DEM refuses (a strong noise channel, a probabilistic
    # observable) and it folds classically-controlled feedback exactly (the same record relabel
    # the sampler applies), so it accepts every circuit the sampler does. There is NO statistical
    # fallback: a determinism verdict that silently switched between an exact check and a sampled
    # estimate would mean two different things under one name. In the rare case the exact analysis
    # cannot run (an acausal / propagation-class-leaving feedback, a shared-wire read — a
    # propagation-class reject surfaces earlier, from the sampler above), diagnose refuses loudly.
    num_det = dets.shape[1]
    det_info = _xtim.detector_determinism(circuit.text)
    if not det_info["ok"]:
        reason = det_info["error"] or (
            "propagation-class reject" if det_info["rejected"] else "unavailable")
        raise XtimError(
            "diagnose cannot classify detector determinism exactly for this circuit: "
            f"{reason}. The determinism verdict is computed exactly from the noiseless bare "
            "state (parity-Pauli eigenvalue); there is no statistical fallback.")
    if det_info["num_detectors"] != num_det:            # engine/sampler disagreement — internal
        raise XtimError(
            "diagnose internal inconsistency: the determinism analysis reports "
            f"{det_info['num_detectors']} detectors but the sampler reports {num_det}")
    deterministic = list(det_info["deterministic_detectors"])
    gauge = list(det_info["gauge_detectors"])

    # whole-DEM expressibility (reuse detector_error_model's refusal) on the ACTUAL
    # circuit — error mechanisms live in the noisy text, so we ask the original.
    dem_ok = True
    dem_reason = None
    # The DEM export can refuse for several reasons, all of which a report should
    # survive: a non-DEM-expressible mechanism (XtimDemError), or a feature the DEM
    # path does not yet handle such as classically-controlled feedback, which the
    # exporter rejects as XtimParseError/XtimRejectError. diagnose() REPORTS the
    # refusal; it never needs the DEM itself, so any of these is a graceful "no".
    try:
        circuit.detector_error_model_text(include_expectations=True)
    except (XtimDemError, XtimParseError, XtimRejectError) as e:
        dem_ok = False
        # Report the cause, not the exception's full __str__: an XtimParseError from
        # the DEM path (e.g. feedback-not-yet-handled) otherwise drags in its generic
        # "fix the flagged line(s) / probably a typo" parse hint, which misleads here.
        if isinstance(e, XtimParseError):
            dem_reason = "; ".join(msg for _, msg in e.errors) or str(e)
        else:
            dem_reason = str(e)

    # EXACT declared-frame verify (shot-free, engine source of truth). A column whose sign is
    # governed by a byproduct frame MUST declare that frame on its PAULI_EXPECTATION line; the
    # engine then folds it and the sign is deterministic. We REFUSE loudly (strict, no statistical
    # fallback) when a column has a SOLVABLE sign-controlling frame (one exists) but its declared
    # frame does not SPAN it (column_ok=0) — a missing / mis-declared byproduct frame. Columns with
    # NO solvable frame (a dead equator channel of a Pauli eigenstate, or a residual free/logical
    # read such as an MPP terminal readout) are legitimately signless and are NOT refused. A
    # determinizing SUPERSET spans and is accepted (column_ok=1), so it is not in `bad`.
    declared_frames: list = []
    if exps.shape[1] > 0:                       # only verify when there ARE expectation columns
        fc = _xtim.expectation_frames(circuit.text)
        if not fc["ok"]:
            reason = fc["error"] or (
                "propagation-class reject" if fc["rejected"] else
                ("; ".join(msg for _, msg in fc["errors"]) if fc["errors"] else "unavailable"))
            raise XtimError(
                "diagnose cannot verify declared PAULI_EXPECTATION frames for this circuit: "
                f"{reason}. The frame verdict is computed exactly from the noiseless bare state; "
                "there is no statistical fallback.")
        declared_frames = list(fc["declared"])
        bad = [i for i in range(fc["num_columns"])
               if fc["solvable"][i] and not fc["column_ok"][i]]
        if bad:
            detail = "; ".join(
                "column L%d: required %s, declared %s"
                % (i, list(fc["required"][i]), list(fc["declared"][i])) for i in bad)
            raise XtimError(
                "diagnose refuses: a PAULI_EXPECTATION column has a sign-controlling record set "
                "but does not DECLARE it as its byproduct frame (rec[-k] on the PAULI_EXPECTATION "
                f"line). Declare the frame so the engine can fold it. {detail}")

    expectations: list[ExpectationInfo] = []
    for i in range(exps.shape[1]):
        col = exps[:, i]
        abs_col = np.abs(col)
        well_defined = bool(abs_col.std() <= 1e-9)
        abs_value = float(abs_col.mean()) if well_defined else math.nan
        # raw sign constant across the run? (True once the declared frame is folded by the engine)
        signs = (col < 0).astype(np.uint8)
        sign_constant = bool(np.all(signs == signs[0])) if len(signs) else True
        # frame_hint echoes the DECLARED frame records (verified above); it is the circuit's own
        # declaration, not a discovered guess. Only meaningful (non-None) when a frame is declared.
        declared = list(declared_frames[i]) if i < len(declared_frames) else []
        frame_hint = declared or None
        # signed (frame-corrected) beta_i: the target <P_i>. The frame is folded, so a well-defined
        # column is sign-constant and signed_value is just constant-sign * magnitude. nan when the
        # magnitude is ill-defined, or when the sign still varies (a residual free/logical read with
        # no fixed byproduct frame — e.g. an MPP terminal readout).
        if not well_defined:
            signed_value = math.nan
        elif sign_constant:
            signed_value = -abs_value if signs[0] else abs_value
        else:
            signed_value = math.nan
        # frame_const := the folded canonical sign bit, reported only for a declared, folded frame.
        frame_const = None
        if frame_hint is not None and signed_value == signed_value:  # not nan
            frame_const = 1 if signed_value < 0 else 0
        expectations.append(ExpectationInfo(
            index=i,
            abs_value=abs_value,
            signed_value=signed_value,
            sign_constant=sign_constant,
            well_defined=well_defined,
            frame_hint=frame_hint,
            frame_const=frame_const,
            dem_expressible=dem_ok,
            dem_reason=None if dem_ok else dem_reason,
        ))

    # advisory post-selection recipe (a STRING; never applied)
    if deterministic:
        recipe = "keep = ~dets[:, %s].any(axis=1)" % deterministic
    else:
        recipe = "keep = np.ones(dets.shape[0], dtype=bool)  # no deterministic detectors"

    return Diagnosis(
        shots=int(shots),
        chi=chi,
        chi_source=chi_source,
        num_detectors=num_det,
        deterministic_detectors=deterministic,
        gauge_detectors=gauge,
        expectations=expectations,
        suggested_postselect=recipe,
    )
