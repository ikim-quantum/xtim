"""xtim.ler — ONE-CALL post-selected logical-error-rate estimation (observable mode).

Estimating the post-selected logical error rate (LER) of a protocol by hand needs the
post-selection mask that keeps only shots in which NO deterministic detector fired — the
step everyone forgets, and forgetting it inflates the answer from the true O(p^d) to O(p).

``Circuit.postselected_logical_error_rate`` (and the module alias
``xtim.postselected_logical_error_rate``) does it in a single call on the circuit's
``OBSERVABLE_INCLUDE`` logical observable(s)::

    import xtim
    c = xtim.load_example("cultivation_d5")
    r = c.postselected_logical_error_rate(p=1e-3, shots=10**6)
    r.value        # post-selected observable LER
    r.sem          # standard error of r.value
    r.upper_bound  # rule-of-three 3/kept (95% CL) when 0 errors observed, else None
    r.acceptance   # post-selection acceptance rate (kept / shots)
    r.kept         # number of accepted shots
    r.warnings     # list[str] of advisories (no noise, zero errors, low acceptance)

The keep set is the exact noiseless determinism partition (``_xtim.detector_determinism``);
the shots come from the default twirl detector sampler (refused circuits raise
``XtimTwirlRefusal`` by name). To score an output state instead, use ``xtim.RecordFidelity``
(docs/usage.md); a circuit with ``PAULI_EXPECTATION`` lines is refused by name.

See ``docs/xtim_postselected_logical_error_rate.md`` for the worked story and pitfalls.
"""
from __future__ import annotations

import dataclasses
import math
import re
import warnings

import numpy as np

from . import _xtim
from .collect import DEFAULT_BAKED_P, _NOISE_LINE, scale_noise
from .errors import XtimError

# Treat estimates at or below this threshold as "consistent with zero logical errors" —
# firing the rule-of-three upper bound and clamping value to 0.0 (64 x machine eps).
_EPS_FLOOR: float = 64.0 * float(np.finfo(float).eps)


@dataclasses.dataclass(frozen=True)
class PostselectedLER:
    """The result of :meth:`Circuit.postselected_logical_error_rate` — a small, plot-ready record.

    Attributes
    ----------
    value : float
        The fraction of ACCEPTED shots in which a logical observable was flipped. When 0
        logical errors are observed, ``value`` is ``0.0`` and ``upper_bound`` is set instead.
    sem : float
        Binomial standard error of ``value`` (nan when fewer than two shots survive). In the
        zero-error regime ``sem`` is 0 and NOT meaningful — read ``upper_bound``.
    upper_bound : float | None
        A rule-of-three 95%-confidence UPPER bound ``3 / kept`` on the LER, set ONLY when 0
        logical errors were observed; ``None`` when a nonzero LER was resolved.
    acceptance : float
        Post-selection acceptance rate ``kept / shots``.
    kept : int
        Number of shots that survived the automatic post-selection.
    shots : int
        Number of shots sampled.
    mode : str
        ``"observable"`` (the only mode since EXP-fix; kept for the record layout).
    p : float | None
        The physical error rate the noise was scaled to (None = run as-is).
    warnings : list[str]
        Advisory messages (no noise, zero-error upper bound, low/small acceptance). Each is
        also re-emitted through :mod:`warnings`. Empty when the estimate is clean.
    """

    value: float
    sem: float
    acceptance: float
    kept: int
    shots: int
    mode: str
    p: "float | None"
    upper_bound: "float | None" = None
    warnings: list = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        raw = self.value
        v = "nan" if raw != raw else "%.6g" % raw
        s = "nan" if self.sem != self.sem else "%.3g" % self.sem
        p = "None" if self.p is None else "%g" % self.p
        parts = ["value=%s" % v, "sem=%s" % s, "mode=%s" % self.mode,
                 "p=%s" % p, "shots=%d" % self.shots,
                 "acceptance=%.4g" % self.acceptance, "kept=%d" % self.kept]
        if self.upper_bound is not None:
            parts.append("upper_bound=%.3g" % self.upper_bound)
        out = "PostselectedLER(" + ", ".join(parts) + ")"
        if self.warnings:
            out += "\n  # %d warning(s) — see .warnings" % len(self.warnings)
        return out


def _keep_from_deterministic(det: list) -> "object | None":
    """A keep(dets, meas) closure post-selecting on the deterministic detectors: keeps only
    shots in which NO deterministic detector fired. None when there is none."""
    if not det:
        return None
    idx = list(det)
    return lambda dets, meas: ~dets[:, idx].any(axis=1)


def _deterministic_detectors(circuit) -> list:
    """The exact noiseless determinism partition (parity-Pauli eigenvalue on the bare state); there
    is no statistical fallback — an unclassifiable circuit is refused by name."""
    from ._exact_scope import explicit_exact   # J-b1 P13: detector_determinism is an explicit entry point
    with explicit_exact():
        info = _xtim.detector_determinism(circuit.text)
    if not info["ok"]:
        reason = info["error"] or ("propagation-class reject" if info["rejected"] else "unavailable")
        raise XtimError(
            "postselected_logical_error_rate cannot classify detector determinism exactly for this "
            f"circuit: {reason}. The keep set is computed exactly from the noiseless bare state; there "
            "is no statistical fallback.")
    return list(info["deterministic_detectors"])


def postselected_logical_error_rate(circuit, *, p: float = 1e-3, shots: int = 10 ** 6,
                                    seed: int = 0, p0: float = DEFAULT_BAKED_P) -> PostselectedLER:
    """Estimate the **post-selected** logical error rate of ``circuit``'s ``OBSERVABLE_INCLUDE``
    observable(s) in ONE call.

    Every shot in which ANY deterministic detector fired is DISCARDED (heralded): the
    pure-post-selection / maximal-heralding number, NOT a decoded logical error rate.

    1. The keep set is the exact noiseless determinism partition
       (``~dets[:, deterministic].any(1)``) — the step users forget, applied unconditionally.
    2. ``value`` = fraction of ACCEPTED shots with any logical observable flipped, sampled by
       the default twirl detector sampler (a refused circuit raises ``XtimTwirlRefusal``).
    3. ``p`` / ``shots`` / ``seed`` rescale the noise (:func:`xtim.scale_noise`) and control
       statistics.

    Refusals (:class:`xtim.XtimError`): a circuit that declares ``PAULI_EXPECTATION`` columns
    (the beta-fidelity mode was removed — score the output state with ``RecordFidelity``, or
    remove the PAULI_EXPECTATION lines to score the observable(s)); a circuit with no
    ``OBSERVABLE_INCLUDE``.

    Warnings (in ``result.warnings`` AND re-emitted via :mod:`warnings`): *no noise* (the value
    is the noiseless logical error rate), *zero-error upper bound*, *low acceptance*.

    Parameters
    ----------
    p : float
        Physical error rate to scale the circuit's noise to (default 1e-3; None = as-is).
    shots : int
        Number of shots (default 1e6).
    seed : int
        Engine seed (default 0).
    p0 : float
        The circuit's baked-in noise strength (default 1e-3).
    """
    from .circuit import Circuit  # local import: avoid a circular import at module load

    if not isinstance(circuit, Circuit):
        tn = type(circuit).__name__
        hint = ""
        if tn == "Circuit":  # almost certainly a stim.Circuit
            hint = (" — to score a stim.Circuit, wrap its text: "
                    "xtim.Circuit(str(the_stim_circuit))")
        raise TypeError(
            "postselected_logical_error_rate expects an xtim.Circuit, got "
            f"{type(circuit).__module__}.{tn}{hint}")
    if p is not None:
        if (isinstance(p, bool) or not isinstance(p, (int, float))
                or p != p or not (0.0 <= p <= 1.0)):
            raise ValueError(f"p must be a float in [0,1] or None, got {p!r}")
        p = float(p)
    shots = int(shots)
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    n_exp = int(circuit.num_expectations)
    if n_exp > 0:
        raise XtimError(
            f"postselected_logical_error_rate: the circuit declares {n_exp} PAULI_EXPECTATION column(s); "
            "the beta-fidelity mode was removed (direct fidelity estimation: declare OUTPUT_QUBITS and use "
            "RecordFidelity, see docs/twirl_record_sampler.md). To score the OBSERVABLE_INCLUDE "
            "observable(s), remove the PAULI_EXPECTATION lines.")
    if int(circuit.num_observables) == 0:
        raise XtimError(
            "postselected_logical_error_rate: this circuit declares no OBSERVABLE_INCLUDE to score. "
            "Declare a logical observable (OBSERVABLE_INCLUDE), or score the output state with "
            "RecordFidelity (OUTPUT_QUBITS).")

    keep = _keep_from_deterministic(_deterministic_detectors(circuit))
    msgs: list[str] = []
    if not any(_NOISE_LINE.match(l) for l in circuit.text.splitlines()):
        msgs.append(
            "circuit declares NO noise instructions: 'value' is the noiseless logical error "
            "rate (varying p rescales nothing). Add a noise channel (DEPOLARIZE1/2, X_ERROR, "
            "...) to measure an LER.")
    result = _observable_mode(circuit, keep, msgs, p=p, p0=p0, shots=shots, seed=seed)
    for m in result.warnings:
        warnings.warn(m, UserWarning, stacklevel=2)
    return result


def _observable_mode(circuit, keep, msgs, *, p, p0, shots, seed) -> PostselectedLER:
    """Observable-mode LER: fraction of ACCEPTED shots with any observable flipped.

    Uses the public detector sampler on its DEFAULT engine, the twirl record engine (the
    fast path; observables only, no expectations) — no sampling is re-implemented here.
    A circuit the twirl engine refuses raises :class:`xtim.XtimTwirlRefusal` by name
    (OUTSIDE_T_DEPTH_1 / NONCOMMUTING_MAGIC / LAW_LIMITATION_PROPAGATION); there is no
    fallback to the exact engine."""
    from .circuit import Circuit

    eff = circuit
    if p is not None:
        eff = Circuit(scale_noise(circuit.text, p, p0),
                      _source=getattr(circuit, "_source", None))

    dets, obs = eff.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)

    if keep is not None:
        keep_mask = np.asarray(keep(dets, None), dtype=bool)
    else:
        keep_mask = np.ones(shots, dtype=bool)
    kept = int(keep_mask.sum())

    # A shot is a logical error if ANY declared logical observable flipped.
    err = obs.any(axis=1) if obs.shape[1] else np.zeros(shots, dtype=bool)
    err_kept = err[keep_mask]
    if kept > 0:
        value = float(err_kept.mean())
        # Binomial standard error of the accepted-shot flip fraction.
        sem = math.sqrt(value * (1.0 - value) / kept) if kept > 1 else math.nan
    else:
        value = math.nan
        sem = math.nan
    acceptance = kept / shots if shots else math.nan
    upper = _upper_bound_if_zero_error(value, kept, msgs)
    if upper is not None:
        value = 0.0  # clamp fp-noise residual to exactly 0
        sem = 0.0    # sem is not meaningful in the zero-error regime; report upper_bound
    _warn_low_acceptance(kept, shots, acceptance, msgs)
    return PostselectedLER(
        value=value, sem=sem, acceptance=acceptance, kept=kept, shots=shots,
        mode="observable", p=p, upper_bound=upper, warnings=msgs)


def _upper_bound_if_zero_error(value: float, kept: int, msgs: list) -> "float | None":
    """Rule-of-three 95%-CL upper bound when 0 logical errors were observed.

    In the zero-error regime the sample SEM collapses to 0, which reads as a misleading
    "LER = 0 +/- 0". When no logical error is seen (``value`` at or below ``_EPS_FLOOR``)
    the honest statistical statement is an UPPER bound ``LER < 3 / kept`` (the rule of
    three at 95% CL).
    Returns that bound and appends a warning; else None.  The caller is expected to
    clamp ``value`` to ``0.0`` when this returns a non-None bound."""
    if kept > 0 and (value == value) and value <= _EPS_FLOOR:
        ub = 3.0 / kept
        msgs.append(
            "0 logical errors in %d accepted shots: the estimate is consistent with 0, "
            "so sem (~0) is NOT meaningful. Report the rule-of-three 95%% upper bound "
            "LER < %.3g (= 3/%d), or raise shots." % (kept, ub, kept))
        return ub
    return None


def _warn_low_acceptance(kept: int, shots: int, acceptance: float, msgs: list) -> None:
    """Advise when the post-selection kept nothing / very little (a soft signal)."""
    if kept == 0:
        msgs.append("post-selection accepted 0 of %d shots — the estimate is nan; "
                    "raise shots or lower p" % shots)
        return
    if shots and acceptance < 1e-3:
        msgs.append("low acceptance (%.3g): few shots survive post-selection, so the "
                    "estimate is noisy — raise shots" % acceptance)
    if 0 < kept < 100:
        msgs.append("only %d shots survived post-selection: the estimate is "
                    "statistically weak regardless of sem — raise shots." % kept)
