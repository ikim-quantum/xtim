"""Exact per-shot REGISTER fidelity of retained twirl output states — the state route (``xtim.RecordFidelity`` at the
top level; the only fidelity route since RF-ref, 2026-10-22: the Target API and the reference / conditioned-reference
semantics were removed).

The retained state of a ``TwirlSampler.sample_barrier`` shot is ``psi = Pi_rec(y) . P . C |B>`` (``|B>`` the bare
state, ``P`` the residual prefix Pauli, ``C`` the plan's Clifford, ``Pi_rec(y)`` the projection onto every record
operator at the shot's recorded raw value ``y`` — exactly ``materialize(i)``).

``RecordFidelity(sampler).evaluate(buf, correction=c)`` scores the OUTPUT_QUBITS register:
``F = <phi| P_c Tr_R(psi psi^dag / <psi|psi>) P_c |phi>`` — the ancillas and every measurement read wire are traced
out; ``|phi>`` is the noiseless output, ``P_c`` the decoder's Pauli on the output (``c`` uint8[shots, 2, n_out]: X
row, Z row, flattened OUTPUT_QUBITS declaration order; ``None`` = no correction). A fired detector therefore costs
nothing by itself; only the output is compared.

STRICT output contract (owner, 2026-10-22), checked at construction: the noiseless output must be pure and fixed,
independent of every outcome — the circuit closes its frames (Pauli feedback), as Stim's deterministic detectors
require. Checked structurally (the bare stabilizer group is a product across output | rest; records live on the
rest); refused by name otherwise (``xtim.XtimFidelityRefusal``, ``.code``): ``RF_OUTPUT_NOT_STRICT`` (the message names
the undetermined output directions on your qubits and the measurements whose outcomes fix them), ``RF_NO_OUTPUT`` (no
OUTPUT_QUBITS).
Evaluation: one rotation-free projector expectation when the output is a stabilizer state, else the rest-sector sum of
the closed-form Gauss-sum kernel (sectors solved per shot); no dense register.

``correction_from_observables(sampler, pred, duals)`` builds ``c`` from a decoder's predicted observable flips and the
user's per-observable duals on the port.

Refused by name besides: buffers from a circuit with a different state model (checked bit for bit) and an evaluation
that would take what the evaluator HOLDS (memo + interned plan contents + kernels + caches, ``stats()["held_bytes"]``)
above ``max_bytes`` (default: no budget; a failed allocation is refused by name).

Memo: a pure speed device keyed on the plan content, the record signs and the prefix's part the value reads (the
correction included). By default it lives for ONE ``evaluate()`` call; ``keep_memo=True`` keeps it across calls
(``stats()["memo_bytes"]``). The interned plan contents and their kernels persist across calls; ``clear()`` drops
everything held. ``cache_budget`` (``None`` = the default, 512 MiB; ``math.inf`` = unbounded, as for the sampler's
``plan_payload_budget``) bounds what persists: an ``evaluate()`` call that finds more held drops it first (values are
pure functions of their keys, so the output is unchanged; ``stats()["cache_flushes"]``).
"""
from __future__ import annotations

import numpy as np

import re

from .errors import XtimError, XtimFidelityRefusal

__all__ = ["RecordFidelity", "correction_from_observables"]

_RF_CODE = re.compile(r"RecordFidelity: (RF_[A-Z_]+): ")


def _named(e: Exception):
    """The engine's RF_* refusal (its text carries the name) as an XtimFidelityRefusal; None for anything else."""
    m = _RF_CODE.search(str(e))
    return XtimFidelityRefusal(str(e), m.group(1)) if m else None


class RecordFidelity:
    """Exact per-shot register fidelity evaluator of one sampler's layout-3 records (see the module docstring).

    ``evaluate(buffer, shot_index=None, *, correction=None, keep_memo=False)`` reads every shot's own record (plan
    content, P_rec, outcomes); ``correction`` is the decoder's Pauli on the output."""

    def __init__(self, sampler, *, max_bytes: int = 0, cache_budget: "int | float | None" = None):
        # None = the engine default (512 MiB), math.inf = unbounded; 0 / negatives refused by name
        try:
            self._rf = sampler.record_fidelity(int(max_bytes), cache_budget)
        except (ValueError, RuntimeError) as e:
            r = _named(e)
            if r is None:
                raise
            raise r from None

    def evaluate(self, buffer, shot_index=None, *, correction=None, keep_memo: bool = False):
        """Exact per-shot fidelity of ``buffer``'s shots: float64[shots] (``shot_index`` None), float64[len] for a
        sequence of indices, a float for one index. ``correction``: uint8[buffer shots, 2, n_out] output-Pauli
        masks (see the class docstring). keep_memo=False (default) drops the memo after the call (the plan contents
        and kernels persist; ``clear()`` drops them)."""
        scalar = shot_index is not None and np.ndim(shot_index) == 0
        shots = None
        if shot_index is not None:
            shots = np.ascontiguousarray(np.atleast_1d(np.asarray(shot_index)).reshape(-1), dtype=np.int64)
        if correction is not None:
            correction = np.ascontiguousarray(correction, dtype=np.uint8)
        try:
            out = np.asarray(self._rf.evaluate(buffer, correction, shots))
        except (ValueError, RuntimeError) as e:
            r = _named(e)
            if r is None:
                raise
            raise r from None
        finally:
            if not keep_memo:
                self._rf.clear_memo()
        return float(out[0]) if scalar else out

    def clear(self) -> None:
        """Drop everything the evaluator holds: the memo, the plan contents and their kernels (values are
        pure functions of their keys)."""
        self._rf.clear()

    def stats(self) -> dict:
        return dict(self._rf.stats())


def correction_from_observables(sampler, pred, duals) -> np.ndarray:
    """The decoder's per-shot output-Pauli correction for ``RecordFidelity.evaluate(buf, correction=...)``.

    ``pred``: uint8/bool[shots, n_obs], the decoder's predicted flip of each OBSERVABLE_INCLUDE (n_obs =
    ``sampler.num_observables``). ``duals``: uint8[n_obs, 2, n_out], per observable the Pauli on the OUTPUT_QUBITS
    port (X row, Z row; flattened OUTPUT_QUBITS declaration order) that undoes its flip — the user's choice (a dual of
    a port logical is fixed only modulo the logicals the records do not fix). Returns
    ``c = pred . duals (mod 2)``, uint8[shots, 2, n_out]. Refused by name: wrong shapes, entries outside {0, 1}."""
    n_obs = int(sampler.num_observables)
    n_out = len(sampler.output_wires())
    p = np.asarray(pred)
    d = np.asarray(duals)
    if p.ndim != 2 or p.shape[1] != n_obs:
        raise XtimError(f"correction_from_observables: pred must be [shots, {n_obs}] (one column per observable); "
                        f"got shape {p.shape}")
    if d.shape != (n_obs, 2, n_out):
        raise XtimError(f"correction_from_observables: duals must be uint8[{n_obs}, 2, {n_out}] (per observable a Pauli "
                        f"on the {n_out} OUTPUT_QUBITS port qubits); got shape {d.shape} — a dual off the port is refused")
    if p.dtype != np.bool_ and np.any((p != 0) & (p != 1)):
        raise XtimError("correction_from_observables: pred entries must be 0 / 1")
    if np.any((d != 0) & (d != 1)):
        raise XtimError("correction_from_observables: duals entries must be 0 / 1")
    c = (p.astype(np.int64) @ d.reshape(n_obs, 2 * n_out).astype(np.int64)) & 1
    return np.ascontiguousarray(c.reshape(-1, 2, n_out).astype(np.uint8))
