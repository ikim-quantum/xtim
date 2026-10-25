"""xtim.fidelity_helpers — byproduct-frame extraction helpers for logical operators.

Primary entry point::

    offsets = xtim.extract_frame(circuit, "X15*X16*X17*X18*X19*X20*X21")
    # offsets is a list[int] of positive k values such that rec[-k] are the records
    # whose parity pins the sign of the given logical Pauli expectation value.

Mechanism
---------
The function reuses the existing ``_xtim.expectation_frames`` binding, which
computes the canonical minimal sign-controlling record set (``required``) for each
declared ``PAULI_EXPECTATION`` column.

To probe an arbitrary operator without conflicts with existing declarations, the
function:

1. Strips all existing ``PAULI_EXPECTATION`` lines from the circuit text.
2. Appends a frameless ``PAULI_EXPECTATION(0) <operator>`` — column 0.
3. Calls ``_xtim.expectation_frames``.
4. If ``solvable[0]`` is falsy, raises ``XtimError`` (sign not pinnable).
5. Converts ``required[0]`` (absolute 0-indexed record indices) to rec[-k] offsets:
   ``k = circuit.num_measurements - abs_index``.

This keeps the column index predictably at 0 and avoids any label-clash with
existing declarations, which must have unique integer labels.
"""
from __future__ import annotations

import re

from . import _xtim
from .errors import XtimError

# This name appears in this module's public signatures.  Until 2026-08-29
# they were spelled as attributes of a package name this module never imports,
# so nothing could resolve them: harmless at runtime under postponed
# annotations, but typing.get_type_hints(), IDEs and doc tooling all failed on
# the public API of a package meant to be read by students.
#
# Imported for REAL, not under TYPE_CHECKING: the guarded form fixes static
# checkers but leaves get_type_hints() still raising, because the names do not
# exist at runtime.  Verified no cycle -- circuit.py imports THIS module only
# from inside functions -- and with these, all 13 callables here resolve.
from .circuit import Circuit          # noqa: F401  (used in annotations)

# Matches any top-level PAULI_EXPECTATION line (single-line instructions only;
# REPEAT blocks with interior indentation are not matched, which is correct —
# PAULI_EXPECTATION cannot legally appear inside a REPEAT body).
_PE_LINE_RE = re.compile(r'^PAULI_EXPECTATION\b.*$', re.MULTILINE)


def _strip_pauli_expectations(text: str) -> str:
    """Remove all PAULI_EXPECTATION instruction lines from circuit text."""
    return _PE_LINE_RE.sub('', text)


def extract_frame(
    circuit: "Circuit",
    operator: str,
    *,
    minimal: bool = True,
) -> list[int]:
    """Return the byproduct-frame rec[-k] offsets for *operator*.

    Parameters
    ----------
    circuit : Circuit
        The circuit containing the protocol.
    operator : str
        A Pauli product in Stim/xtim ``PAULI_EXPECTATION`` notation,
        e.g. ``"X15*X16*X17*X18*X19*X20*X21"``.
    minimal : bool, default True
        If True (default), return the canonical minimal sign-controlling set —
        the smallest set of records whose XOR pins the sign of ``<operator>`` in
        the circuit's bare state.  This is the ``required`` field from
        ``_xtim.expectation_frames``.

        If False, attempt to return a larger "full readout" representative —
        the declared frame from an existing ``PAULI_EXPECTATION`` declaration that
        matches this operator's ``required`` set and is a superset of it, if one
        exists in the circuit.  Falls back to the minimal set when no such
        declaration is found (best-effort with no warning).

    Returns
    -------
    list[int]
        Positive integers *k* such that ``rec[-k]`` are the sign-controlling
        records, ordered by ascending absolute record index.  Convert back to
        0-indexed absolute positions via ``abs_index = circuit.num_measurements - k``.

        An empty list ``[]`` is a **valid** result meaning the operator's sign is
        already determined by the bare state without any measurement conditioning.
        This happens when the operator is in the stabilizer group (ideal expectation
        ±1 regardless of any measurement outcomes) or when its ideal expectation is
        exactly 0 (a "magic direction" such as ⟨Z̄⟩=0 for a T-state).  The resulting
        ``PAULI_EXPECTATION`` line simply carries no ``rec[-k]`` records.  To see
        the numeric expectation value, sample the column on the exact engine
        (``engine="exact"``, ``return_expectations=True``).

    Raises
    ------
    xtim.XtimError
        If the sign of *operator* is not pinnable by any measurement-record frame
        (e.g. the operator touches a free or logical qubit whose sign is not fixed
        by the bare state's stabilizer frame).
    xtim.XtimError
        If *operator* is syntactically invalid in the circuit context.
    """
    num_records = circuit.num_measurements

    # Build a probe text: strip existing PAULI_EXPECTATION declarations to avoid
    # duplicate-label errors, then append a frameless declaration for the operator.
    stripped = _strip_pauli_expectations(circuit.text)
    probe_text = stripped.rstrip() + f'\nPAULI_EXPECTATION(0) {operator}\n'

    # Surface syntax errors early with a clean message.
    info = _xtim.parse_info(probe_text)
    if not info['ok']:
        msgs = '; '.join(msg for _, msg in info.get('errors', []))
        raise XtimError(
            f"extract_frame: operator {operator!r} is invalid in this circuit: {msgs}"
        )

    result = _xtim.expectation_frames(probe_text)

    if not result.get('ok'):
        errs = result.get('errors', [])
        msg = '; '.join(m for _, m in errs) if errs else (result.get('error') or 'unknown')
        raise XtimError(
            f"extract_frame: could not analyze operator {operator!r} "
            "(the circuit or operator may be malformed — check qubit indices): "
            f"{msg}"
        )

    if result['num_columns'] == 0:
        raise XtimError(
            f"extract_frame: operator {operator!r} produced no expectation column "
            "(internal error — circuit may not have parsed correctly)"
        )

    # Column 0 is our probed operator.
    col = 0
    if not result['solvable'][col]:
        raise XtimError(
            f"extract_frame: operator {operator!r}'s sign is not pinnable by a "
            "measurement-record frame (the operator has a free or logical component "
            "whose sign is not determined by the bare-state stabilizer frame)"
        )

    abs_indices: list[int] = result['required'][col]
    minimal_offsets = [num_records - idx for idx in abs_indices]

    if minimal:
        return minimal_offsets

    # minimal=False: try to find a spanning declared frame from the original circuit.
    declared_offsets = _try_declared_frame(
        circuit, abs_indices, num_records
    )
    return declared_offsets if declared_offsets is not None else minimal_offsets


def _try_declared_frame(
    circuit: "Circuit",
    required_abs: list[int],
    num_records: int,
) -> list[int] | None:
    """For ``minimal=False``: look for an existing PAULI_EXPECTATION declaration in
    *circuit* whose ``required`` set equals *required_abs* and whose ``declared`` set
    is a superset.  Returns the declared frame as rec[-k] offsets, or None to signal
    fallback to the minimal set.
    """
    result = _xtim.expectation_frames(circuit.text)
    if not result.get('ok') or result['num_columns'] == 0:
        return None

    required_set = frozenset(required_abs)
    for col_req, col_dec in zip(result['required'], result['declared']):
        if frozenset(col_req) == required_set and required_set.issubset(frozenset(col_dec)):
            return [num_records - idx for idx in col_dec]

    return None
