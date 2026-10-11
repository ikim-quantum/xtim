"""Forced faults, exact branches (3.1.11).

Compile a circuit ONCE, then ask, for any explicit set of faults, for EVERY
measurement-outcome branch of nonzero probability with its exact probability::

    sim = xtim.Circuit(text).compile_exact_branches()
    sim.noise_sites()                       # the addressable noise instructions
    br = sim.exact_branches([(3, 0, "X"), {"record": 5}])
    br.probabilities                        # float64[B], sums to 1
    br.measurements / br.detectors / br.observables / br.decisions   # bool[B, *]
    br.expectations                         # float64[B, R]: PAULI_EXPECTATION per branch

A fault addresses a NOISE SITE of the circuit — the circuit's noise
instructions (X/Y/Z_ERROR, DEPOLARIZE1/2, PAULI_CHANNEL_1/2), numbered 0, 1, …
in order of appearance with REPEAT blocks unrolled — by ``(site, target,
pauli)``: ``target`` is the qubit position (1-qubit channels) or pair position
(2-qubit channels) inside that instruction, ``pauli`` one letter X/Y/Z, or two
letters over IXYZ for a pair (first letter on the pair's first qubit, not
"II"). Any Pauli may be forced at a site (the site fixes the POSITION; a
channel that could never draw it is still a valid insertion point). A record
flip ``{"record": j}`` flips the RECORDED bit of absolute measurement record
``j`` (the ``M(p)`` readout flip; the state is not flipped) and may name any
measurement. Every other noise channel and readout coin is forced not to fire.

So ``exact_branches(F)`` is exactly the circuit's sampled law conditioned on
the fired set ``F``: the same propagation-table atoms the sampler fires,
composed into one error, conjugated through every terminal read and declared
expectation, and every read resolved in record order on the bare state —
branching wherever both outcomes have conditional probability > 1e-12 (the
engine's own coin threshold). Channels follow the sampler's conventions:
measurements are RECORDED bits (``!`` inverts and forced flips included, then
classically-controlled feedback); detectors and observables are
REFERENCE-RELATIVE (Stim's detector-sampler semantics); decisions are the RAW
recorded parity of each ``DECISION`` line; expectations are the conditional
``<P>`` of each ``PAULI_EXPECTATION`` on that branch, declared frames and
feedback sign flips applied. Branches come in depth-first order, ``+1``
before ``-1`` at every coin.

The query is read-only: no RNG is consumed and the compiled sampler streams of
the circuit are unchanged. Refusals are loud (``ValueError`` naming the
reason): a malformed or repeated address; noise sites on a circuit whose
coherentized feedback inserted readout-flip noise; record flips feeding a
coherentized (non-Clifford-crossing) feedback; more than ``max_branches``
branches.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from . import _xtim
from .errors import XtimParseError, XtimRejectError

__all__ = ["ExactBranches", "ExactBranchSimulator"]


@dataclasses.dataclass(frozen=True)
class ExactBranches:
    """Every nonzero-probability outcome branch of one fault set (see module doc)."""
    probabilities: np.ndarray   # float64[B]
    measurements: np.ndarray    # bool[B, num_measurements]   (recorded bits)
    detectors: np.ndarray       # bool[B, num_detectors]      (reference-relative)
    observables: np.ndarray     # bool[B, num_observables]    (reference-relative)
    decisions: np.ndarray       # bool[B, num_decisions]      (raw recorded parity)
    expectations: np.ndarray    # float64[B, len(columns)]
    columns: tuple              # the PAULI_EXPECTATION columns of `expectations`

    def __len__(self) -> int:
        return int(self.probabilities.shape[0])


def _fault_tuple(f):
    """(kind, index, target, pauli): kind 0 = noise site, 1 = record flip."""
    if isinstance(f, dict):
        keys = set(f)
        if keys == {"record"}:
            return (1, _int(f["record"], "record"), -1, "")
        if keys == {"noise", "target", "pauli"}:
            return (0, _int(f["noise"], "noise"), _int(f["target"], "target"), _pauli(f["pauli"]))
        raise ValueError(f"fault {f!r}: a fault is {{'noise': i, 'target': t, 'pauli': p}} "
                         "or {'record': j}")
    if isinstance(f, (tuple, list)) and len(f) == 3:
        return (0, _int(f[0], "noise"), _int(f[1], "target"), _pauli(f[2]))
    raise ValueError(f"fault {f!r}: a fault is (site, target, pauli), "
                     "{'noise': i, 'target': t, 'pauli': p} or {'record': j}")


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"fault {what} {v!r} is not an integer")
    return int(v)


def _pauli(p):
    if not isinstance(p, str):
        raise ValueError(f"fault pauli {p!r} is not a string")
    return p


class ExactBranchSimulator:
    """A circuit compiled once for forced-fault exact-branch queries."""

    def __init__(self, circuit):
        text = circuit if isinstance(circuit, str) else circuit.text
        self._p = _xtim.ExactBranchProgram(text)
        errs = self._p.errors
        if errs:
            raise XtimParseError(list(errs))
        if not self._p.ok:
            raise XtimRejectError(self._p.reject_gate_index, kind="class")

    def noise_sites(self) -> list:
        """The addressable noise instructions, in site order: dicts with
        ``channel``, ``qubits`` and ``probs``."""
        return self._p.noise_sites()

    @property
    def noise_refusal(self) -> str:
        """Why noise-site faults are refused on this circuit ("" = they are not)."""
        return self._p.noise_refusal

    @property
    def record_refusal(self) -> str:
        """Why record flips are refused on this circuit ("" = they are not)."""
        return self._p.record_refusal

    def exact_branches(self, faults=(), *, expectations=None,
                       max_branches: int = 1 << 20) -> ExactBranches:
        """Every measurement-outcome branch of nonzero probability under exactly
        ``faults`` (see the module docstring for addressing and semantics).
        ``expectations``: PAULI_EXPECTATION column indices to evaluate (None =
        all declared)."""
        ft = [_fault_tuple(f) for f in faults]
        cols = None if expectations is None else [int(c) for c in expectations]
        if cols == []:                      # explicitly none: skip every column
            r = self._p.exact_branches(ft, [], int(max_branches))
            exps = r["expectations"][:, :0]
            columns = ()
        else:
            r = self._p.exact_branches(ft, cols or [], int(max_branches))
            exps = r["expectations"]
            columns = tuple(cols) if cols else tuple(range(exps.shape[1]))
        return ExactBranches(r["probabilities"], r["measurements"], r["detectors"],
                             r["observables"], r["decisions"], exps, columns)
