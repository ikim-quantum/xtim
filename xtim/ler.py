"""xtim.ler — ONE-CALL post-selected logical-error-rate estimation for magic-state-prep circuits.

Estimating the logical error rate (LER) of a magic-state-prep protocol has, until
now, required a student to know the workflow is bimodal and to reproduce several
by-hand steps that are easy to get wrong:

  * know whether the value is scored by fidelity (PAULI_EXPECTATION) or by a
    post-selected observable (OBSERVABLE_INCLUDE);
  * call ``xtim.diagnose`` and hand-build the post-selection mask that keeps only
    shots in which NO deterministic detector fired — the step everyone forgets,
    and forgetting it inflates the answer from the true O(p^d) to O(p);
  * set ``target_k`` and dig ``infidelity`` out of a ``collect`` stats row.

``Circuit.postselected_logical_error_rate`` (and the module alias
``xtim.postselected_logical_error_rate``) collapses all of that into a single call
that returns a small result object::

    import xtim
    c = xtim.load_example("code_switching_faithful")
    r = c.postselected_logical_error_rate(p=1e-3, shots=10**6)
    r.value        # the LER: 1-F (fidelity mode) or post-selected obs LER (observable mode)
    r.sem          # standard error of r.value
    r.upper_bound  # rule-of-three 3/kept (95% CL) when 0 errors observed, else None
    r.acceptance   # post-selection acceptance rate (kept / shots)
    r.kept         # number of accepted shots
    r.mode         # "fidelity" | "observable"
    r.target_k     # k used for the fidelity denominator (2**k)
    r.warnings     # list[str] of advisories (no-noise floor, incomplete support, ...)

It reuses the shipped machinery verbatim — ``diagnose`` for the auto post-selection,
``collect``/``Task`` for the per-shot fidelity path — so the returned ``value``
matches the manual workflow to within sampling statistics; nothing here
re-implements the sampler.

See ``docs/xtim_postselected_logical_error_rate.md`` for the worked story and pitfalls.
"""
from __future__ import annotations

import dataclasses
import math
import re
import warnings

import numpy as np

from .collect import DEFAULT_BAKED_P, Task, _NOISE_LINE, collect, scale_noise
from .diagnose import diagnose
from .errors import XtimError

# Treat infidelity estimates at or below this threshold as "consistent with zero
# logical errors" — firing the rule-of-three upper bound and clamping value to 0.0.
# The floor catches the fp-noise residual  1 − mean(f_s) ≈ eps/2 ≈ 1.11e-16  that
# arises when the mean fidelity is exactly 1 (e.g. distillation_15_1_3 at low p),
# while staying far below any physically-meaningful LER (≥ 1e-9 in practice).
# 64 × machine-eps ≈ 1.42e-14; comfortable margin on both sides.
_EPS_FLOOR: float = 64.0 * float(np.finfo(float).eps)

# One PAULI_EXPECTATION declaration line (start-of-line, allowing indentation) —
# used to drop free/logical (nan-target) columns from a reduced fidelity circuit.
_EXP_LINE = re.compile(r"^\s*PAULI_EXPECTATION\(", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class PostselectedLER:
    """The result of :meth:`Circuit.postselected_logical_error_rate` — a small, plot-ready record.

    Attributes
    ----------
    value : float
        The logical error rate. In *fidelity* mode this is ``1 - F``; in
        *observable* mode it is the fraction of ACCEPTED shots in which a logical
        observable was flipped.  When 0 logical errors are observed (``value`` is
        at or below the fp-noise floor ``~1.4e-14``), ``value`` is clamped to
        ``0.0`` and ``upper_bound`` is set instead — see below.  Outside that
        regime, in fidelity mode the raw estimate can be a tiny negative near F=1;
        guard ``max(0.0, value)`` before a log plot.
    sem : float
        Standard error of ``value`` (nan when fewer than two shots survive). In the
        zero-error regime ``sem`` collapses to ~0 and is NOT meaningful — read
        ``upper_bound`` instead.
    upper_bound : float | None
        A rule-of-three 95%-confidence UPPER bound ``3 / kept`` on the LER, set ONLY
        when 0 logical errors were observed (``value`` rounds to <= 0). ``None`` when
        a nonzero LER was resolved. Report ``LER < upper_bound`` in that regime; the
        ~0 ``sem`` there does not mean "LER = 0 +/- 0".
    acceptance : float
        Post-selection acceptance rate ``kept / shots``.
    kept : int
        Number of shots that survived the automatic post-selection.
    shots : int
        Number of shots sampled.
    mode : str
        ``"fidelity"`` or ``"observable"`` (see :meth:`Circuit.postselected_logical_error_rate`).
    target_k : int | None
        The ``k`` used for the fidelity denominator ``2**k`` (None in observable mode).
    p : float | None
        The physical error rate the noise was scaled to (None = run as-is).
    warnings : list[str]
        Advisory messages (no-noise floor, incomplete Pauli support, excluded
        free/logical reads, zero-error upper bound, low/small acceptance, ...). Each
        is also re-emitted through :mod:`warnings` so a script that reads only
        ``value`` still sees it on stderr. Empty when the estimate is clean.
    """

    value: float
    sem: float
    acceptance: float
    kept: int
    shots: int
    mode: str
    target_k: "int | None"
    p: "float | None"
    upper_bound: "float | None" = None
    warnings: list = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        raw = self.value
        if raw != raw:
            v = "nan"
        else:
            # Clamp a tiny-negative floating-point 1-F to 0 for display; .value is raw.
            v = "%.6g" % (0.0 if (raw < 0.0 and raw > -1e-9) else raw)
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
    """A keep(dets, meas) closure post-selecting on the deterministic detectors.

    Keeps only shots in which NO deterministic detector fired — the conservative,
    always-correct post-selection ``xtim.diagnose`` recommends. Returns None when
    there are no deterministic detectors (nothing to post-select on)."""
    if not det:
        return None
    idx = list(det)
    return lambda dets, meas: ~dets[:, idx].any(axis=1)


def _drop_expectation_columns(text: str, drop: "set[int]") -> str:
    """Return `text` with the k-th PAULI_EXPECTATION declaration removed for k in `drop`.

    Declarations are counted in file order (matching the column / declaration order
    used everywhere else). Removing a PAULI_EXPECTATION line is sound: it declares
    an expectation channel only and emits no measurement record, so downstream
    ``rec[-k]`` indexing is unaffected."""
    out = []
    seen = 0
    for line in text.splitlines():
        if _EXP_LINE.match(line):
            if seen in drop:
                seen += 1
                continue
            seen += 1
        out.append(line)
    return "\n".join(out) + "\n"


def postselected_logical_error_rate(circuit, *, p: float = 1e-3, shots: int = 10 ** 6,
                                    seed: int = 0, target_k: int = 1,
                                    p0: float = DEFAULT_BAKED_P) -> PostselectedLER:
    """Estimate the **post-selected** logical error rate of ``circuit`` in ONE call.

    This reports the **post-selected** logical error rate: every shot in which ANY
    deterministic detector fired is DISCARDED (heralded). This is the pure-post-selection /
    maximal-heralding number — NOT a decoded logical error rate. A decoded /
    partial-post-selection variant is future work.

    This is the module-level entry point; ``circuit.postselected_logical_error_rate(...)`` is the
    identical method form. It performs, automatically, every manual step of the
    fidelity / post-selection workflow:

    1. Runs :func:`xtim.diagnose` and builds the post-selection that keeps only shots
       in which NO deterministic detector fired (``~dets[:, deterministic].any(1)``).
       **This is the step users forget** — skipping it turns a suppressed O(p^d)
       answer into an unsuppressed O(p) one. It is applied here unconditionally.
    2. Auto-detects the scoring mode:

       * **fidelity** — the circuit declares at least one *scorable*
         ``PAULI_EXPECTATION`` column (well-defined non-nan target AND nonzero
         magnitude). ``value = 1 - F`` via the :func:`xtim.collect` /
         :class:`xtim.Task` per-shot fidelity path. ``target_k`` defaults to 1.
       * **observable** — no scorable ``PAULI_EXPECTATION`` but an
         ``OBSERVABLE_INCLUDE`` is declared. ``value`` = fraction of ACCEPTED shots
         with any logical observable flipped.
       * otherwise :class:`xtim.XtimError` is raised (nothing to score) — including
         when EVERY ``PAULI_EXPECTATION`` is a free/logical (nan) or magnitude-0
         (beta=0) read and there is no observable (the ``mpp_magic`` case).
    3. ``p`` / ``shots`` / ``seed`` rescale the noise and control statistics exactly
       as an :class:`xtim.Task` does.

    Warnings (returned in ``result.warnings`` AND re-emitted via :mod:`warnings` so a
    script that reads only ``value`` still sees them) flag when the answer is a proxy
    rather than a true LER:

    * *no noise* — the circuit declares no noise instruction, so ``value`` is the
      noiseless fidelity floor, NOT an error rate (e.g. ``cube_ccz``).
    * *incomplete support* — the declared ``PAULI_EXPECTATION`` set cannot reach
      fidelity 1 even noiselessly (``sum(beta**2) < 2**target_k - 1``, e.g.
      ``cube_ccz`` at the default ``target_k=1``); ``value`` is a partial-support
      proxy — pass the true ``target_k`` for a real fidelity.
    * *free / logical or magnitude-0 read* — a column has no fixed byproduct frame
      (nan target) or a zero target beta (e.g. the destructive terminal MPP in
      ``mpp_magic``); it carries no fidelity anchor and is excluded from the score.
    * *zero-error upper bound* — when 0 logical errors survive, ``sem`` collapses to
      ~0; ``upper_bound = 3 / kept`` (rule of three, 95% CL) is set instead.

    Parameters
    ----------
    p : float
        Physical error rate to scale the circuit's noise to (default 1e-3).
    shots : int
        Number of shots (default 1e6).
    seed : int
        Engine seed (default 0).
    target_k : int
        Logical-qubit count of the target magic state; the fidelity denominator is
        ``2**target_k`` (fidelity mode only; default 1).
    p0 : float
        The circuit's baked-in noise strength (default 1e-3).

    Returns
    -------
    PostselectedLER
        ``.value .sem .acceptance .kept .mode .target_k .warnings`` (see the class).

    Examples
    --------
    >>> import xtim
    >>> c = xtim.load_example("code_switching_faithful")
    >>> r = c.postselected_logical_error_rate(p=1e-3, shots=10**6)   # doctest: +SKIP
    >>> r.mode, round(r.value, 8)                                     # doctest: +SKIP
    ('fidelity', 1.5e-06)
    """
    from .circuit import Circuit  # local import: avoid a circular import at module load

    # -- input validation (fail fast, before any sampling) --------------------
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

    diag = diagnose(circuit)
    keep = _keep_from_deterministic(diag.deterministic_detectors)
    msgs: list[str] = []

    # A circuit with no noise instructions cannot express an error rate: scaling p
    # rescales nothing and the value is just the noiseless fidelity floor. Flag it
    # LOUDLY so a floor value (e.g. cube_ccz's 0.125) is never mistaken for an LER.
    if not any(_NOISE_LINE.match(l) for l in circuit.text.splitlines()):
        msgs.append(
            "circuit declares NO noise instructions: 'value' is the noiseless "
            "fidelity floor, NOT an error rate (varying p rescales nothing). Add a "
            "noise channel (DEPOLARIZE1/2, X_ERROR, ...) to measure an LER.")

    n_exp = int(circuit.num_expectations)
    n_obs = int(circuit.num_observables)

    # -- mode selection -------------------------------------------------------
    # A PAULI_EXPECTATION column is "scorable" iff its noiseless target beta is
    # well-defined (non-nan) AND has nonzero magnitude. A nan column is a
    # free/logical read (no fixed byproduct frame, e.g. a destructive terminal MPP);
    # a magnitude-0 (beta==0) column carries no fidelity anchor either — both are
    # excluded so a beta=0 column can never masquerade as a perfect F=(1+0)/2 score.
    scorable_cols: list[int] = []
    nan_cols: list[int] = []
    zero_cols: list[int] = []
    if n_exp > 0:
        for e in diag.expectations:
            b = e.signed_value
            if b != b:                 # nan -> free/logical read
                nan_cols.append(e.index)
            elif b == 0.0:             # magnitude-0 -> no fidelity anchor
                zero_cols.append(e.index)
            else:
                scorable_cols.append(e.index)

    if scorable_cols:
        result = _fidelity_mode(circuit, scorable_cols, nan_cols, zero_cols, keep,
                                msgs, p=p, p0=p0, shots=shots, seed=seed,
                                target_k=target_k)
    elif n_obs > 0:
        if nan_cols or zero_cols:
            msgs.append(
                "the declared PAULI_EXPECTATION column(s) are all non-scorable "
                "(free/logical reads with nan target, or magnitude-0 beta=0); scoring "
                "the OBSERVABLE_INCLUDE logical observable(s) instead")
        result = _observable_mode(circuit, keep, msgs,
                                  p=p, p0=p0, shots=shots, seed=seed)
    else:
        # Nothing to score. This is the documented mpp_magic case: every
        # PAULI_EXPECTATION column is a free/logical read (nan) or magnitude-0
        # (beta=0), and there is no OBSERVABLE_INCLUDE — so there is no fidelity
        # anchor. Raise rather than report a plausible-looking (1+0)/2 = 0.5.
        detail = ""
        if n_exp:
            detail = (
                " (its %d PAULI_EXPECTATION column(s) are all non-scorable: "
                "free/logical reads with no fixed sign (nan target) or magnitude-0 "
                "(beta=0), so none carries a fidelity anchor)" % n_exp)
        raise XtimError(
            "postselected_logical_error_rate: this circuit declares no scorable PAULI_EXPECTATION "
            f"and no OBSERVABLE_INCLUDE to score{detail}. Declare a logical observable "
            "(OBSERVABLE_INCLUDE) or a magic expectation (PAULI_EXPECTATION) with a "
            "fixed byproduct-frame sign so there is something to measure against.")

    # FRICTION-4: re-emit every advisory through `warnings` so a script that reads
    # only `.value` still gets a stderr signal — in ADDITION to `.warnings`.
    for m in result.warnings:
        warnings.warn(m, UserWarning, stacklevel=2)
    return result


def _fidelity_mode(circuit, scorable_cols, nan_cols, zero_cols, keep, msgs, *,
                   p, p0, shots, seed, target_k) -> PostselectedLER:
    """Fidelity-mode LER: 1 - F via the collect/Task per-shot fidelity path."""
    from .circuit import Circuit

    eff = circuit
    drop = set(nan_cols) | set(zero_cols)
    if nan_cols:
        msgs.append(
            "excluded %d free/logical PAULI_EXPECTATION column(s) %s from the "
            "fidelity score: they have no fixed byproduct frame (nan target), so they "
            "carry no fidelity anchor" % (len(nan_cols), sorted(nan_cols)))
    if zero_cols:
        msgs.append(
            "excluded %d magnitude-0 PAULI_EXPECTATION column(s) %s from the fidelity "
            "score: their target beta is 0 (a logical/free read), so they carry no "
            "fidelity anchor" % (len(zero_cols), sorted(zero_cols)))
    if drop:
        # Drop the non-scorable columns so the fidelity beta has no nan/zero anchors.
        eff = Circuit(_drop_expectation_columns(circuit.text, drop),
                      _source=getattr(circuit, "_source", None))

    task = Task(circuit=eff, p=p, p0=p0, shots=shots, seed=seed,
                keep=keep, target_k=int(target_k))

    # collect's fidelity path emits the incomplete-support UserWarning; capture it so
    # it becomes a returned warning string rather than noise on stderr.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (row,) = collect([task])
    for w in caught:
        if issubclass(w.category, UserWarning):
            msgs.append(str(w.message).split("\n", 1)[0].strip())

    return _build_fidelity_result(row, shots=shots, target_k=target_k, p=p, msgs=msgs)


def _build_fidelity_result(row, *, shots, target_k, p, msgs) -> PostselectedLER:
    """Assemble a fidelity-mode :class:`PostselectedLER` from a collect stats *row*.

    Shared by :func:`_fidelity_mode` and :func:`xtim.fidelity_from_logicals` so the
    partial-support / zero-error-upper-bound / low-acceptance advisory logic and the
    result construction live in exactly one place."""
    kept = int(row["kept"])
    value = float(row["infidelity"])
    sem = float(row["infidelity_sem"])
    acceptance = float(row["acceptance_rate"])
    tk = int(target_k)

    # BLOCKER-2(a): an incomplete Pauli support (noiseless ceiling < 1) means `value`
    # is a PARTIAL-support proxy, NOT a true infidelity/LER. Add an actionable summary
    # so a value like cube_ccz's 0.125 is un-mistakable for an error rate.
    beta = row.get("target_beta")
    if beta is not None:
        ceiling = (1.0 + float(np.square(np.asarray(beta, dtype=float)).sum())) / float(2 ** tk)
        if ceiling < 1.0 - 1e-6:
            msgs.append(
                "value=%.6g is a PARTIAL-support proxy, NOT a true infidelity/LER: the "
                "declared columns cannot reach fidelity 1 even noiselessly (ceiling "
                "%.4g < 1 at target_k=%d). If this is a k>1 magic state, pass "
                "target_k=<number of logical qubits> and declare every Pauli with a "
                "nonzero ideal expectation." % (value, ceiling, tk))

    upper = _upper_bound_if_zero_error(value, kept, msgs)
    if upper is not None:
        value = 0.0  # clamp fp-noise residual (e.g. eps/2) to exactly 0
        sem = 0.0    # sem is not meaningful in the zero-error regime; report upper_bound
    _warn_low_acceptance(kept, shots, acceptance, msgs)
    return PostselectedLER(
        value=value, sem=sem, acceptance=acceptance, kept=kept, shots=shots,
        mode="fidelity", target_k=tk, p=p, upper_bound=upper, warnings=msgs)


def _observable_mode(circuit, keep, msgs, *, p, p0, shots, seed) -> PostselectedLER:
    """Observable-mode LER: fraction of ACCEPTED shots with any observable flipped.

    Uses the public detector sampler (the same lean run collect performs) — no
    sampling is re-implemented here; the observable channel is simply one collect
    cannot fold into its expectation-only stats row, so it is read directly."""
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
        mode="observable", target_k=None, p=p, upper_bound=upper, warnings=msgs)


def _upper_bound_if_zero_error(value: float, kept: int, msgs: list) -> "float | None":
    """Rule-of-three 95%-CL upper bound when 0 logical errors were observed.

    In the zero-error regime the sample SEM collapses to ~0, which reads as a
    misleading "LER = 0 +/- 0". When no logical error is seen — ``value`` is at or
    below the fp-noise floor ``_EPS_FLOOR`` (~1.4e-14), which catches both an exact
    0.0 (observable mode, zero noise) and the fp-noise residual eps/2 (~1.1e-16) that
    arises in fidelity mode when mean(f_s)==1 exactly — the honest statistical
    statement is an UPPER bound ``LER < 3 / kept`` (the rule of three at 95% CL).
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
