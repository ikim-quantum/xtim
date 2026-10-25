"""Forced faults, exact law (3.1.11; factored law and provenance addressing: Unreleased).

Compile a circuit ONCE, then ask, for any explicit set of faults, for the EXACT law of the
circuit's detectors, observables and decisions (with the conditional PAULI_EXPECTATION
values) — factored, never enumerated read by read::

    sim = xtim.Circuit(text).compile_exact_branches()
    sim.noise_sites()                       # the addressable noise instructions
    law = sim.exact_branches([(3, 0, "X"), {"record": 5}])
    law.rows_probabilities                  # float64[B]: the structured rows (sum to 1)
    law.rows_detectors / rows_observables / rows_decisions   # bool[B, *] (every coin 0)
    law.rows_expectations                   # float64[B, R]: conditional <P> given the targets
    law.uniform_detectors / uniform_observables / uniform_decisions  # bool[u, *]
    law.uniform_expectation_signs           # bool[u, R]
    law.probabilities / detectors / observables / decisions / expectations  # expanded view

The law: pick a row (its probability), then XOR any subset of the u uniform DIRECTIONS into its
targets, every subset equally likely (independent fair coins); direction j flips the sign of
the expectation columns marked in ``uniform_expectation_signs[j]``. The expanded view lists
every (row, subset) — ``len(law)`` = B * 2^u rows, all distinct, probability = row probability
/ 2^u — and raises ``ExpandedLawTooLarge`` (pointing at the factored fields) beyond
``max_rows``. Expectations are CONDITIONAL on the row's (detectors, observables, decisions)
only: the probability-weighted average of <P> over every measurement outcome with those targets
(0 for a column anticommuting with a read) — every other read, declared frame records
included, is averaged over (``RX 0 1; M 0; PAULI_EXPECTATION(0) X1 rec[-1]``: the sampler
reports +-1 per shot, the law one row with 0). Raw per-branch measurement records are not returned.

A fault addresses a NOISE SITE of the circuit — the circuit's noise
instructions (X/Y/Z_ERROR, DEPOLARIZE1/2, PAULI_CHANNEL_1/2, and each
``E``/``CORRELATED_ERROR`` + ``ELSE_CORRELATED_ERROR`` chain as ONE site),
numbered 0, 1, … in order of appearance with REPEAT blocks unrolled — by
``(site, target, pauli)``: ``target`` is the qubit position (1-qubit channels)
or pair position (2-qubit channels) inside that instruction, ``pauli`` one
letter X/Y/Z, or two letters over IXYZ for a pair (first letter on the pair's
first qubit, not "II"). A chain site has ONE target (0), its whole support:
``pauli`` is one letter per qubit of the site's ``qubits`` (``noise_sites()``
lists them, with the chain's own terms under ``paulis``), e.g. ``"XIZ"``; any
non-identity word is accepted, not only the chain's terms (each chain outcome
is one fault atom). Any Pauli may be forced at a site (the site fixes the POSITION; a
channel that could never draw it is still a valid insertion point), except a
Pauli of probability zero in the site's channel whose propagated error leaves
the simulable class: that one is refused by name (the sampler never fires it,
so it does not reject the circuit). A record
flip ``{"record": j}`` flips the RECORDED bit of absolute measurement record
``j`` (the ``M(p)`` readout flip; the state is not flipped) and may name any
measurement. Every other noise channel and readout coin is forced not to fire.

So ``exact_branches(F)`` is exactly the circuit's sampled law conditioned on
the fired set ``F``: the same propagation-table atoms the sampler fires,
composed into one end-of-circuit error E, conjugated through every terminal
read and declared expectation. Every target is an affine GF(2) function of the
raw reads (``!`` inverts, forced flips and classically-controlled feedback
included); the stabilizer twirl of the bare state splits it into independent
fair coins (the span of the reads' X-parts outside the magic sector) and a
structured part (deterministic parities and at most k magic-sector parities,
whose joint law is walked on its support only). Channels follow the sampler's
conventions: detectors and observables are REFERENCE-RELATIVE (Stim's
detector-sampler semantics); decisions are the RAW recorded parity of each
``DECISION`` line; declared frames and feedback sign flips are applied to the
expectations.

Coherentized feedback (a ``CX rec[-k] q`` whose byproduct crosses a
non-Clifford gate) is addressed by PROVENANCE: every noise instruction of the
normalized circuit carries its origin (user site i, or the readout-flip fold
the coherentizer inserts for an ``M(p)`` it reads), so sites map exactly even
where folds were inserted. A record flip of a folded measurement
(``folded_records``) fires the fold: the recorded bit AND the coherent control
flip together, and where the control is observed again the fold's undo
restores its state — exactly Stim's ``M(p)`` readout flip.

The query is read-only: no RNG is consumed and the compiled sampler streams of
the circuit are unchanged. Refusals are loud (``ValueError`` naming the
reason): a malformed or repeated address; a zero-probability Pauli whose
propagated error leaves the simulable class (see above); a record flip of a readout-NOISELESS
measurement read by coherentized feedback (no fold channel to force;
``refused_records`` lists them); more than ``max_rows`` structured rows, more
than ``max_work`` walk work units, or more than 63 magic-sector parities. Every
bound is a deterministic count, never a clock.
"""
from __future__ import annotations

import os

import numpy as np

from . import _xtim
from .errors import XtimParseError, XtimRejectError

__all__ = ["ExactBranches", "ExactBranchSimulator", "ExpandedLawTooLarge"]


DEFAULT_MAX_WORK = float(_xtim.EXACT_LAW_WORK_DEFAULT)


class ExpandedLawTooLarge(ValueError):
    """The expanded view (rows x 2^u coin subsets) exceeds ``max_rows``: use the factored
    fields (``rows_*`` and ``uniform_*``) instead."""


class ExactBranches:
    """The exact law of one fault set, FACTORED (see module doc).

    Factored form (always available):
      ``rows_probabilities`` float64[B] (sums to 1), ``rows_detectors`` / ``rows_observables`` /
      ``rows_decisions`` bool[B, *], ``rows_expectations`` float64[B, R]: the structured rows,
      every fair coin set to 0, each column the CONDITIONAL expectation given the row's targets;
      ``uniform_detectors`` / ``uniform_observables`` / ``uniform_decisions`` bool[u, *] and
      ``uniform_expectation_signs`` bool[u, R]: the independent fair-coin directions.
    The law = any row XOR any subset of the u directions, each subset equally likely (the row's
    probability split evenly); a direction flips the sign of the columns it marks.

    Expanded form (``probabilities``, ``detectors``, ``observables``, ``decisions``,
    ``expectations``): every (row, subset) as its own row — B * 2^u rows, distinct targets —
    built on first access; ``ExpandedLawTooLarge`` when B * 2^u > ``max_rows``.
    """

    def __init__(self, d, columns, max_rows):
        self.rows_probabilities = d["probabilities"]
        self.rows_detectors = d["detectors"]
        self.rows_observables = d["observables"]
        self.rows_decisions = d["decisions"]
        self.rows_expectations = d["expectations"]
        self.uniform_detectors = d["uniform_detectors"]
        self.uniform_observables = d["uniform_observables"]
        self.uniform_decisions = d["uniform_decisions"]
        self.uniform_expectation_signs = d["uniform_expectation_signs"]
        self.magic_parities = int(d["magic_parities"])
        self.columns = columns
        self.max_rows = int(max_rows)
        self._expander = d["expander"]
        self._expanded = None

    @property
    def num_uniform(self) -> int:
        """u: the number of independent fair-coin directions."""
        return int(self.uniform_detectors.shape[0])

    @property
    def num_rows(self) -> int:
        """B: the number of structured rows."""
        return int(self.rows_probabilities.shape[0])

    @property
    def num_expanded(self) -> int:
        """B * 2^u: the number of rows of the expanded view."""
        return self.num_rows << self.num_uniform

    def _expand(self):
        if self._expanded is None:
            u, B = self.uniform_detectors.shape[0], self.rows_probabilities.shape[0]
            if u > 62 or (B << u) > self.max_rows:
                raise ExpandedLawTooLarge(
                    f"the expanded law has {B} rows x 2^{u} fair-coin subsets > max_rows = "
                    f"{self.max_rows}: use the factored fields rows_* and uniform_* (or raise "
                    "max_rows)")
            if u == 0:                      # no coin: the expanded view IS the rows
                self._expanded = (self.rows_probabilities, self.rows_detectors,
                                  self.rows_observables, self.rows_decisions,
                                  self.rows_expectations)
            else:
                self._expanded = self._expander.expand(self.max_rows)
        return self._expanded

    @property
    def probabilities(self) -> np.ndarray:
        """float64[B*2^u]: expanded probabilities (sum to 1)."""
        e = self._expanded
        return (e if e is not None else self._expand())[0]

    @property
    def detectors(self) -> np.ndarray:
        """bool[B*2^u, D] (reference-relative)."""
        e = self._expanded
        return (e if e is not None else self._expand())[1]

    @property
    def observables(self) -> np.ndarray:
        """bool[B*2^u, O] (reference-relative)."""
        e = self._expanded
        return (e if e is not None else self._expand())[2]

    @property
    def decisions(self) -> np.ndarray:
        """bool[B*2^u, K] (raw recorded parity)."""
        e = self._expanded
        return (e if e is not None else self._expand())[3]

    @property
    def expectations(self) -> np.ndarray:
        """float64[B*2^u, R]: conditional expectations given the targets."""
        e = self._expanded
        return (e if e is not None else self._expand())[4]

    def __len__(self) -> int:
        return self.num_expanded

    def __getattr__(self, name):
        if name == "measurements":
            raise AttributeError(
                "ExactBranches has no 'measurements': the exact law is over detectors, "
                "observables and decisions (raw per-branch records are not enumerated)")
        raise AttributeError(name)


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
        ``channel``, ``qubits`` and ``probs`` (an E/ELSE chain: channel
        ``"CORRELATED_ERROR"``, its sequential probabilities, and ``paulis`` = its
        terms as words over ``qubits``)."""
        return self._p.noise_sites()

    @property
    def noise_refusal(self) -> str:
        """Why noise-site faults are refused on this circuit ("" = they are not)."""
        return self._p.noise_refusal

    @property
    def record_refusal(self) -> str:
        """Why record flips are refused on this circuit ("" = they are not)."""
        return self._p.record_refusal

    @property
    def refused_records(self) -> dict:
        """Records whose forced flip is refused, with the reason (a record read by feedback
        coherentized across a non-Clifford gate whose measurement has no ``M(p)`` readout-flip
        fold to force)."""
        return dict(self._p.refused_records)

    @property
    def folded_records(self) -> list:
        """Records whose ``M(p)`` readout flip is folded into the coherentized feedback: a forced
        flip of one fires the fold (recorded bit AND coherent control flip together)."""
        return list(self._p.folded_records)

    def exact_branches(self, faults=(), *, expectations=None, max_rows: int = 1 << 20,
                       max_work: float = DEFAULT_MAX_WORK) -> ExactBranches:
        """The exact law under exactly ``faults`` (see the module docstring for addressing and
        semantics), factored. ``expectations``: PAULI_EXPECTATION column indices to evaluate
        (None = all declared, [] = none). ``max_rows`` caps the structured rows (refused by name
        beyond) and the expanded view (``ExpandedLawTooLarge`` on access beyond). ``max_work``
        caps the exact walk over the magic-sector parities, in its deterministic work units
        (each Born probability / collapse / leaf read charged (chi+1)*n*(k+1); the default is
        about one second of work): beyond it the query is refused by name, never truncated."""
        ft = [_fault_tuple(f) for f in faults]
        cols = None if expectations is None else [int(c) for c in expectations]
        d = self._p.exact_law(ft, cols or [], int(max_rows), float(max_work), cols == [])
        columns = tuple(cols) if cols is not None else tuple(range(d["expectations"].shape[1]))
        return ExactBranches(d, columns, max_rows)
