"""xtim.collect — batch sampling of ``PAULI_EXPECTATION`` columns over many tasks.

A `Task` bundles one circuit with a user decoder and keep mask; `collect(tasks)` returns one
plot-ready stats row per task (``shots``, ``kept``, ``acceptance_rate``, ``value`` = the mean
corrected expectation per column, ``sem``, ``seconds``, plus the task's params). It reads
expectations, so it runs ``compile_detector_sampler(engine="exact")``. To score an output
state, use ``xtim.RecordFidelity`` (docs/usage.md) instead.

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

from ._exact_scope import explicit_exact  # J-b1 P13
from .circuit import Circuit

# Baked-in noise strength of the committed benchmarks (scripts/msp_e2e.py's P0).
DEFAULT_BAKED_P = 1e-3

# Noise-line shapes of the benchmark dialect (ported from scripts/msp_e2e.py /
# scripts/msp_accounting.py — keep in sync). _NOISE_LINE recognizes a noise
# instruction; _NOISE_ARGS additionally captures its argument list for rewriting.
_NOISE_LINE = re.compile(
    r"^\s*(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL"
    r"|E\(|CORRELATED_ERROR|ELSE_CORRELATED_ERROR)")
_NOISE_ARGS = re.compile(
    r"^(\s*)(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL\w*)\(([^)]+)\)")
# One term of an E/ELSE_CORRELATED_ERROR chain. Its argument is a SEQUENTIAL probability
# (fires iff no earlier term fired), so the chain is rescaled through its categorical law
# (see scale_noise), never argument by argument.
_CHAIN_ARGS = re.compile(
    r"^(\s*)(E|CORRELATED_ERROR|ELSE_CORRELATED_ERROR)\(([^)]+)\)")


def scale_noise(text: str, p: float, p0: float = DEFAULT_BAKED_P) -> str:
    """Scale every noise-instruction argument of `text` by p/p0.

    The ported scripts/msp_e2e.py helper: the rewrite must be surjective onto the
    noise lines (every recognized noise line gets rescaled) and bijective on line
    count — any mismatch raises ValueError, never silently mis-scales.

    This is a pure TEXT rewrite: it does not validate that the scaled probabilities
    stay in [0, 1] (e.g. p=1.0 on a p0=1e-3 arg yields a >1 probability). Those bounds
    are checked when the resulting circuit is parsed (e.g. inside `collect`), not here.

    An ``E``/``CORRELATED_ERROR`` + ``ELSE_CORRELATED_ERROR`` chain is ONE categorical
    channel whose arguments are sequential probabilities (term k fires with p_k iff no
    earlier term fired), so it is rescaled through its categorical law: every outcome
    probability q_k = p_k * prod_{j<k} (1 - p_j) is multiplied by p/p0 and the chain is
    written back in sequential form (p'_k = q'_k / (1 - sum_{j<k} q'_j)) — exactly what
    multiplying every PAULI_CHANNEL argument by the factor does. Scaling the sequential
    arguments one by one would NOT be that (it changes the outcome law at second order)."""
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
    rem_old = rem_new = 1.0        # chain state: P(no earlier term fired), original / scaled
    for line in lines:
        m = _NOISE_ARGS.match(line)
        c = None if m else _CHAIN_ARGS.match(line)
        if m:
            args = [float(a) for a in m.group(3).split(",")]
            scaled = ", ".join("%.12g" % (a * factor) for a in args)
            line = f"{m.group(1)}{m.group(2)}({scaled})" + line[m.end():]
            n_scaled += 1
        elif c:
            if c.group(2) != "ELSE_CORRELATED_ERROR":
                rem_old = rem_new = 1.0            # E / CORRELATED_ERROR opens a chain
            p_old = float(c.group(3))
            q_new = p_old * rem_old * factor       # the outcome's categorical probability, scaled
            p_new = q_new / rem_new if rem_new > 0.0 else 0.0
            rem_old *= 1.0 - p_old
            rem_new -= q_new
            line = f"{c.group(1)}{c.group(2)}({'%.17g' % p_new})" + line[c.end():]
            n_scaled += 1
        else:
            rem_old = rem_new = 1.0                # any other line ends a chain (parser rule)
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

    def __post_init__(self):
        if not isinstance(self.circuit, Circuit):
            raise TypeError("Task.circuit must be an xtim.Circuit "
                            f"(got {type(self.circuit).__name__})")
        if int(self.shots) < 0:
            raise ValueError("Task.shots must be >= 0")


def _effective_circuit(task: Task) -> Circuit:
    """The circuit the task actually runs: rescaled text when p is given."""
    if task.p is None:
        return task.circuit
    return Circuit(scale_noise(task.circuit.text, task.p, task.p0),
                   _source=f"{task.circuit._source}@p={task.p:g}")


def _run_task(task: Task, circuit: Circuit) -> dict:
    """The lean workflow, verbatim, for one task -> one stats row."""
    t0 = time.perf_counter()
    shots = int(task.shots)
    s = circuit.compile_detector_sampler(seed=task.seed, engine="exact")   # OD-A: the exact oracle, explicitly
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

    return row


_EXP_RE = re.compile(r"^\s*PAULI_EXPECTATION\(-?\d+\)\s+(\S+)", re.MULTILINE)


def _expectation_paulis(text: str) -> list[str]:
    """The `*`-joined Pauli products declared by PAULI_EXPECTATION, in order."""
    return _EXP_RE.findall(text)


# fork-inheritance transport: the prepared (task, circuit) pairs live here in
# the PARENT before the pool forks; workers receive only an integer index.
_FORK_WORK: list[tuple[Task, Circuit]] = []


def _run_indexed(i: int) -> dict:
    task, circuit = _FORK_WORK[i]
    return _run_task(task, circuit)


def collect(tasks: Iterable["Task"], num_workers: int = 1, *,
            progress: "Callable[[int, int], None] | None" = None,
            ) -> list[dict[str, Any]]:
    """Run every Task; return one stats row per task, in task order.

    num_workers > 1 fans the tasks over a fork-start multiprocessing pool (see
    the module docstring for the transport contract); results are bit-identical
    to a serial run because each task is one self-seeded engine stream.

    Row fields `value` and `sem` are numpy arrays — call `.tolist()` before
    `json.dumps` (rows are built for `pandas.DataFrame(rows)`, not direct JSON).

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
    with explicit_exact():          # J-b1 P13: the exact oracle's reference (collect runs engine="exact")
        for _, c in work_pairs:
            c._resolve_reference()

    work = work_pairs
    total = len(work)

    def _tick(done: int) -> None:
        if progress is not None:
            progress(done, total)

    if num_workers <= 1 or total <= 1:
        rows = []
        for i, (t, c) in enumerate(work, 1):
            rows.append(_run_task(t, c))
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
        return [_run_task(t, c) for t, c in work]

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
