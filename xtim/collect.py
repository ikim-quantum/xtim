"""xtim.collect — batch sampling, composed STRICTLY from the lean primitives.

A `Task` bundles one circuit + decoding policy; `collect(tasks)` runs each task as
exactly the documented lean workflow (docs/xtim_tour.md) — nothing else:

    dets, obs, exps, meas = circuit.compile_detector_sampler(seed=...).sample(
        shots, separate_observables=True, return_expectations=True,
        return_measurements=True)                       # ONE run, one RNG stream
    flips     = decoder(dets, meas)                     # user decoding policy
    corrected = np.where(flips, -exps, exps)            # anticommutation = a sign
    keep      = keep(dets, meas)                        # post-selection = discard
    value     = corrected[keep].mean(axis=0)            # a-bar per declared Pauli

and returns one stats row per task (a plain dict, plot-ready for e.g.
``pandas.DataFrame(rows)``): the task's params/metadata, ``shots``, ``kept``,
``acceptance_rate``, ``value`` (mean corrected expectation per PAULI_EXPECTATION
column), per-entry ``sem``, and ``seconds``. Everything interpretive (the decoder,
the keep mask, any β-combination of ``value``) is user-owned; there is no target
arithmetic in here.

Notes on row fields:
  * Every row also carries ``p0`` (the baked-in noise strength, echoed from the
    Task), ``seed`` (the engine seed used), and ``expectation_paulis`` (the
    Pauli product string behind each ``value[i]``/``sem[i]``, in declaration
    order — so a flattened row stays self-describing once the circuit is gone).
  * ``value``, ``sem``, and ``target_beta`` are numpy arrays — call ``.tolist()``
    before ``json.dumps`` (rows are designed for ``pandas.DataFrame(rows)``, not
    direct JSON serialisation).
  * ``infidelity_sem`` is equally the standard error of ``fidelity`` (since
    std(1 − f) = std(f) — the two are identical, just named for infidelity).
  * ``infidelity`` is the unbiased estimate ``1 - fidelity`` and is NOT clamped:
    near F = 1 (noiseless, or very low p) it can be a tiny negative value (e.g.
    ~ −2e-16 from floating point, or within the error bar at low p). This is
    intentional. Before a log-scale plot, guard it: ``max(0.0, infidelity)``.
  * ``PAULI_EXPECTATION(N)``: the parenthesised N is a free integer LABEL, not the
    output column index — columns appear in declaration order (the i-th declared
    column is ``value[i]``/``target_beta[i]``).

Fidelity (opt-in): set ``Task(target_k=k)`` and collect additionally returns
``fidelity``, ``infidelity`` (= 1 - fidelity), ``infidelity_sem``, plus
``target_k`` and ``target_beta``. It forms the per-shot scalar
``f_s = (1 + corrected_kept @ beta) / 2**k`` (beta = the noiseless target <P_i>
from diagnose) and reports ``mean(f_s)`` with ``std(f_s)/sqrt(kept)``. You are
responsible for declaring EVERY Pauli with nonzero ideal weight: an undeclared
nonzero-beta Pauli makes F silently INCORRECT (its term is dropped, biasing F),
not merely smaller. A gauge column
(undefined target sign) is refused, not silently scored.

Performance: beta (the noiseless target expectation vector) is computed ONCE per
unique noiseless circuit in the parent process before the fork/loop — it is
noise-independent, so a p-sweep over the same base circuit pays the diagnose cost
only once, not once per task. The XtimError refusals (nan beta, zero columns) fire
in the parent (fail-fast before sampling).

Noise rescaling: ``Task(p=...)`` rewrites every noise-instruction argument of the
circuit text by the factor ``p / p0`` (``p0`` = the baked-in strength, default
1e-3 — the committed benchmarks' value) via `scale_noise`, the ported
``scripts/msp_e2e.py`` helper: the rewrite is verified to touch exactly the noise
lines and preserve the line count, failing loudly otherwise.

Workers (num_workers > 1): a ``multiprocessing`` pool with the POSIX **fork**
start method. Tasks are handed to workers by fork inheritance — nothing
user-supplied crosses a pickle boundary on the way IN, so arbitrary closures and
lambdas are fine as ``keep``/``decoder`` (the repo's known pickling quirks do not
apply). Constraints: (a) POSIX only — where fork is unavailable, collect warns
and runs serially (same results, no parallelism); (b) the returned rows ARE
pickled back, so ``metadata`` must be picklable; (c) avoid forking a heavily
multi-threaded parent (the usual POSIX caveat). Reference resolution for every
task happens once, in the parent, before the fork. Results are returned in task
order regardless of ``num_workers``; per-task seeds make them bit-identical to a
``num_workers=1`` run.
"""
from __future__ import annotations

import dataclasses
import re
import time
import warnings
from collections.abc import Iterable
from typing import Any, Callable

import numpy as np

from . import _xtim
from .circuit import Circuit
from .diagnose import diagnose
from .errors import XtimError

# Baked-in noise strength of the committed benchmarks (scripts/msp_e2e.py's P0).
DEFAULT_BAKED_P = 1e-3

# Noise-line shapes of the benchmark dialect (ported from scripts/msp_e2e.py /
# scripts/msp_accounting.py — keep in sync). _NOISE_LINE recognizes a noise
# instruction; _NOISE_ARGS additionally captures its argument list for rewriting.
_NOISE_LINE = re.compile(
    r"^\s*(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL)")
_NOISE_ARGS = re.compile(
    r"^(\s*)(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL\w*)\(([^)]+)\)")


def scale_noise(text: str, p: float, p0: float = DEFAULT_BAKED_P) -> str:
    """Scale every noise-instruction argument of `text` by p/p0.

    The ported scripts/msp_e2e.py helper: the rewrite must be surjective onto the
    noise lines (every recognized noise line gets rescaled) and bijective on line
    count — any mismatch raises ValueError, never silently mis-scales.

    This is a pure TEXT rewrite: it does not validate that the scaled probabilities
    stay in [0, 1] (e.g. p=1.0 on a p0=1e-3 arg yields a >1 probability). Those bounds
    are checked when the resulting circuit is parsed (e.g. inside `collect`), not here."""
    if not isinstance(text, str):
        extra = " — pass its .text" if type(text).__name__ == "Circuit" else ""
        raise TypeError(
            f"scale_noise expects circuit text (str), got {type(text).__name__}{extra}")
    if p0 == 0:
        raise ValueError("p0 (the baked-in noise strength) must be > 0 to rescale by p/p0")
    factor = p / p0
    lines = text.splitlines()
    out = []
    n_noise = sum(1 for l in lines if _NOISE_LINE.match(l))
    n_scaled = 0
    for line in lines:
        m = _NOISE_ARGS.match(line)
        if m:
            args = [float(a) for a in m.group(3).split(",")]
            scaled = ", ".join("%.12g" % (a * factor) for a in args)
            line = f"{m.group(1)}{m.group(2)}({scaled})" + line[m.end():]
            n_scaled += 1
        out.append(line)
    if n_scaled != n_noise:
        raise ValueError(f"noise rewrite mismatch: {n_scaled} lines scaled but "
                         f"{n_noise} noise lines present")
    if len(out) != len(lines):
        raise ValueError("noise rewrite changed the line count")
    return "\n".join(out) + "\n"


@dataclasses.dataclass
class Task:
    """One collect work item: a circuit + the user's decoding policy.

    circuit  : an xtim.Circuit (its text is rescaled when `p` is given).
    p        : physical error rate — noise args scaled by p/p0 (None = run as-is).
    p0       : the circuit's baked-in noise strength (default 1e-3, the
               committed benchmarks' value).
    shots    : number of shots (required).
    seed     : engine seed for this task (the whole task is ONE run/stream).
    keep     : callable (dets, meas) -> bool[shots] keep mask; None = keep all.
               Post-selection modifies nothing; it discards.
    decoder  : callable (dets, meas) -> flip bits, bool-like [shots, R] over the
               PAULI_EXPECTATION columns (declaration order); None = no
               correction. A flip negates the raw expectation (anticommutation).
    metadata : free-form, copied onto the row (picklable under num_workers > 1).
    target_k : logical-qubit count of the TARGET magic state (Hilbert dim
               2**target_k); only used to compute fidelity. None = no fidelity
               computed (output rows unchanged). Must be an int >= 1 if given.
    """
    circuit: Circuit
    _: dataclasses.KW_ONLY
    p: float | None = None
    p0: float = DEFAULT_BAKED_P
    shots: int
    seed: int = 0
    keep: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    decoder: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    metadata: Any = None
    target_k: int | None = None

    def __post_init__(self):
        if not isinstance(self.circuit, Circuit):
            raise TypeError("Task.circuit must be an xtim.Circuit "
                            f"(got {type(self.circuit).__name__})")
        if int(self.shots) < 0:
            raise ValueError("Task.shots must be >= 0")
        if self.target_k is not None:
            if isinstance(self.target_k, bool) or not isinstance(self.target_k, int):
                raise ValueError("Task.target_k must be an int >= 1 (or None)")
            if self.target_k < 1:
                raise ValueError("Task.target_k must be an int >= 1 (or None)")


def _effective_circuit(task: Task) -> Circuit:
    """The circuit the task actually runs: rescaled text when p is given."""
    if task.p is None:
        return task.circuit
    return Circuit(scale_noise(task.circuit.text, task.p, task.p0),
                   _source=f"{task.circuit._source}@p={task.p:g}")


def _run_task(task: Task, circuit: Circuit,
              beta: np.ndarray | None = None) -> dict:
    """The lean workflow, verbatim, for one task -> one stats row.

    `beta` is the precomputed noiseless target expectation vector (from
    _compute_beta), passed in by collect() from the parent so diagnose is called
    only once per unique noiseless circuit text. None when target_k is None."""
    t0 = time.perf_counter()
    shots = int(task.shots)
    s = circuit.compile_detector_sampler(seed=task.seed)
    dets, obs, exps, meas = s.sample(
        shots, separate_observables=True, return_expectations=True,
        return_measurements=True)
    n_cols = exps.shape[1]

    if task.decoder is not None:
        flips = np.asarray(task.decoder(dets, meas), dtype=bool)
        if flips.shape != exps.shape:
            raise ValueError(
                f"decoder returned shape {flips.shape}, expected {exps.shape} "
                "(one flip bit per shot per PAULI_EXPECTATION column)")
        corrected = np.where(flips, -exps, exps)
    else:
        corrected = exps

    if task.keep is not None:
        keep_mask = np.asarray(task.keep(dets, meas), dtype=bool)
        if keep_mask.shape != (shots,):
            raise ValueError(
                f"keep returned shape {keep_mask.shape}, expected ({shots},)")
    else:
        keep_mask = np.ones(shots, dtype=bool)

    kept = int(keep_mask.sum())
    sel = corrected[keep_mask]
    if kept > 0:
        value = sel.mean(axis=0)
    else:
        value = np.full(n_cols, np.nan)
    if kept > 1:
        sem = sel.std(axis=0, ddof=1) / np.sqrt(kept)
    else:
        sem = np.full(n_cols, np.nan)

    row = {
        "p": task.p,
        "p0": task.p0,
        "shots": shots,
        "seed": task.seed,
        "metadata": task.metadata,
        "kept": kept,
        "acceptance_rate": kept / shots if shots else float("nan"),
        "value": value,
        "sem": sem,
        # the Pauli product behind each value[i]/sem[i], in declaration order, so a
        # flattened row stays self-describing once the circuit object is gone.
        "expectation_paulis": _expectation_paulis(circuit.text),
        "seconds": time.perf_counter() - t0,
    }

    if task.target_k is not None:
        assert beta is not None, "beta must be precomputed when target_k is set"
        row.update(_fidelity_row(sel, kept, int(task.target_k), beta))
        row["seconds"] = time.perf_counter() - t0  # include any fidelity overhead

    return row


def _compute_beta(circuit: Circuit, n_cols: int) -> np.ndarray:
    """Compute the noiseless target beta vector for a circuit.

    This is noise-independent (diagnose strips noise internally), so the caller
    should cache the result keyed on the noiseless circuit text and reuse it across
    all tasks that share the same base circuit.

    Raises XtimError immediately if:
      * n_cols == 0 (no PAULI_EXPECTATION columns — fidelity is incomputable).
      * any beta entry is nan (ill-defined target sign or magnitude).
    """
    if n_cols == 0:
        raise XtimError(
            "fidelity requested (target_k set) but the circuit declares no "
            "PAULI_EXPECTATION columns — nothing to score against")

    beta = np.array([e.signed_value for e in diagnose(circuit).expectations],
                    dtype=float)
    bad = [i for i, b in enumerate(beta) if b != b]  # nan => undefined target sign
    if bad:
        raise XtimError(
            f"fidelity is undefined for column(s) {bad}: their noiseless target "
            "beta is ill-defined (the expectation's sign or magnitude is not "
            "constant across the noiseless run). Inspect with xtim.diagnose(circuit); "
            "drop target_k to sample without fidelity")
    return beta


def _fidelity_row(sel: np.ndarray, kept: int, target_k: int,
                  beta: np.ndarray) -> dict:
    """True-state fidelity from the kept, corrected expectation block `sel`.

    beta_i is the noiseless TARGET <P_i> (from _compute_beta / diagnose's
    signed_value). The per-shot fidelity scalar f_s = (1 + sel @ beta) / 2**target_k
    is formed PER SHOT so the error bar std(f_s)/sqrt(kept) captures the correlation
    between columns measured on the same shots.

    Two symmetric UserWarnings bracket a wrong target_k / column set (the raw value
    is always returned unchanged — no clamping):
      * fidelity > 1 (sampled) -> target_k too SMALL, or over-complete columns;
      * noiseless ceiling < 1   -> target_k too LARGE, or INCOMPLETE columns.
    The ceiling (1 + sum(beta**2)) / 2**target_k is the best F the declared set can
    express; for a complete pure-state support sum(beta**2) == 2**target_k - 1 by
    purity, so it is exactly 1."""
    denom = float(2 ** target_k)

    # Completeness guard (computed from the noiseless beta only — no sampling): a
    # ceiling below 1 means the declared columns can never reach F=1, so the support
    # is incomplete or target_k is too large. Symmetric to the fidelity > 1 warning.
    ceiling = (1.0 + float((beta ** 2).sum())) / denom
    if ceiling < 1.0 - 1e-6:
        warnings.warn(
            f"declared PAULI_EXPECTATION set cannot reach fidelity 1 (noiseless "
            f"ceiling={ceiling:.6g} < 1): even with no noise, F maxes out there. "
            "Likely causes: the declared columns are an INCOMPLETE Pauli support for "
            "this target, or target_k is too large (2**target_k is oversize). Declare "
            "every Pauli with a nonzero ideal expectation and set target_k to the "
            "number of logical qubits in the target. (If you intend a partial / "
            "marginal fidelity this is expected — suppress the warning.)",
            UserWarning, stacklevel=4)

    if kept == 0:
        return {"target_k": target_k, "target_beta": beta,
                "fidelity": float("nan"), "infidelity": float("nan"),
                "infidelity_sem": float("nan")}
    f_s = (1.0 + sel @ beta) / denom
    fidelity = float(f_s.mean())
    sem = float(f_s.std(ddof=1) / np.sqrt(kept)) if kept > 1 else float("nan")

    # Warn on physically-impossible fidelity > 1 (sampling-aware slack).
    # Use 5*sem when sem is a real number (kept > 1), otherwise a hard 1e-9 floor.
    slack = max(1e-9, 5 * sem) if not np.isnan(sem) else 1e-9
    if fidelity > 1.0 + slack:
        warnings.warn(
            f"fidelity > 1 detected (observed fidelity={fidelity:.6g}): this is "
            "physically impossible and indicates a misconfiguration. Likely causes: "
            "target_k is too small (2**target_k is then an undersize denominator, "
            "inflating F above 1), or columns are over-complete / double-counted. "
            "Declare ALL nonzero-beta Paulis and set target_k to the number of "
            "logical qubits in the target magic state.",
            UserWarning, stacklevel=4)

    return {"target_k": target_k, "target_beta": beta,
            "fidelity": fidelity, "infidelity": 1.0 - fidelity,
            "infidelity_sem": sem}


_EXP_RE = re.compile(r"^\s*PAULI_EXPECTATION\(-?\d+\)\s+(\S+)", re.MULTILINE)


def _expectation_paulis(text: str) -> list[str]:
    """The `*`-joined Pauli products declared by PAULI_EXPECTATION, in order."""
    return _EXP_RE.findall(text)


# fork-inheritance transport: the prepared (task, circuit, beta) triples live here in
# the PARENT before the pool forks; workers receive only an integer index.
_FORK_WORK: list[tuple[Task, Circuit, np.ndarray | None]] = []


def _run_indexed(i: int) -> dict:
    task, circuit, beta = _FORK_WORK[i]
    return _run_task(task, circuit, beta)


def collect(tasks: Iterable["Task"], num_workers: int = 1, *,
            progress: "Callable[[int, int], None] | None" = None,
            ) -> list[dict[str, Any]]:
    """Run every Task; return one stats row per task, in task order.

    num_workers > 1 fans the tasks over a fork-start multiprocessing pool (see
    the module docstring for the transport contract); results are bit-identical
    to a serial run because each task is one self-seeded engine stream.

    If a Task sets `target_k`, each row also carries `fidelity`, `infidelity`,
    `infidelity_sem`, `target_k`, and `target_beta`. It forms the per-shot scalar
    `f_s = (1 + corrected_kept @ beta) / 2**target_k` (beta = the noiseless target
    <P_i>) and reports `mean(f_s)` with `std(f_s)/sqrt(kept)`. You must declare
    EVERY Pauli with nonzero ideal weight — an undeclared one silently biases F.

    Row fields `value`, `sem`, and `target_beta` are numpy arrays — call
    `.tolist()` before `json.dumps` (rows are built for `pandas.DataFrame(rows)`,
    not direct JSON). `infidelity` is unclamped and can be a tiny negative near
    F = 1; guard `max(0.0, infidelity)` before a log-scale plot.

    progress: optional callback `progress(done, total)` invoked after each task
    completes (in task order) — e.g. `progress=lambda d, n: print(f"{d}/{n}")`."""
    if isinstance(tasks, Task):
        raise TypeError("collect expects an iterable of Task (e.g. a list); "
                        "did you mean collect([task])?")
    tasks = list(tasks)
    for t in tasks:
        if not isinstance(t, Task):
            raise TypeError(f"collect expects xtim.Task items (got {type(t).__name__})")

    # Materialize effective circuits and resolve every reference HERE, in the
    # parent, once per task — workers inherit the resolved state via fork.
    work_pairs = [(t, _effective_circuit(t)) for t in tasks]
    for _, c in work_pairs:
        c._resolve_reference()

    # Precompute beta (the noiseless target expectation vector) ONCE per unique
    # noiseless circuit text — beta is noise-independent, so a p-sweep over the
    # same base circuit pays the diagnose cost only once. Keyed on the stripped
    # (noiseless) circuit text because the effective (rescaled) circuit text varies
    # with p but diagnose strips noise anyway. XtimError refusals (nan beta or zero
    # columns) fire here in the parent, before any sampling, for fast failure.
    beta_cache: dict[str, np.ndarray] = {}
    betas: list[np.ndarray | None] = []
    for t, c in work_pairs:
        if t.target_k is None:
            betas.append(None)
        else:
            # Keyed on the engine's DEFERRED-FRAME signature of the effective
            # circuit: p-variants share it (noise values are excluded), but tasks
            # whose noise PLACEMENT makes normalization land in a different frame
            # get their own diagnose — including its frame-adequacy refusal, which
            # a text-stripped key would silently bypass for colliding tasks.
            key = _xtim.deferred_signature(c.text)
            if key not in beta_cache:
                n_cols = c.num_expectations if hasattr(c, "num_expectations") \
                    else len(_expectation_paulis(c.text))
                beta_cache[key] = _compute_beta(c, n_cols)
            betas.append(beta_cache[key])

    work = [(t, c, b) for (t, c), b in zip(work_pairs, betas)]
    total = len(work)

    def _tick(done: int) -> None:
        if progress is not None:
            progress(done, total)

    if num_workers <= 1 or total <= 1:
        rows = []
        for i, (t, c, b) in enumerate(work, 1):
            rows.append(_run_task(t, c, b))
            _tick(i)
        return rows

    import multiprocessing

    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        warnings.warn(
            "xtim.collect: the fork start method is unavailable on this platform; "
            "running the tasks serially (identical results, no parallelism)",
            RuntimeWarning, stacklevel=2)
        return [_run_task(t, c, b) for t, c, b in work]

    global _FORK_WORK
    _FORK_WORK = work
    try:
        with ctx.Pool(processes=min(int(num_workers), total)) as pool:
            rows = []
            for i, row in enumerate(pool.imap(_run_indexed, range(total)), 1):
                rows.append(row)  # imap yields in task order
                _tick(i)
            return rows
    finally:
        _FORK_WORK = []
