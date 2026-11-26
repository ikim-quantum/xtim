"""Output-state helpers: the noiseless output state of a circuit on qubits you choose (docs/usage.md, "Output state").

    qs = xtim.output_candidates(circuit)          # live qubits: no measurement or reset on them after their last gate
    xtim.output_is_deterministic(circuit, qs)     # pure and the same for every measurement outcome?
    st = xtim.output_state(circuit, qs)           # |psi> = R|S>: rotations, pauli_stabilizers, clifford_paulis

``circuit`` is an :class:`xtim.Circuit` or circuit text; noise is ignored (the helpers describe the noiseless state).
``qubits`` is an explicit list of the circuit's qubit indices (a declared ``OUTPUT_QUBITS`` port is not read). Helpers 2
and 3 accept any qubit that is not measured after its last gate: live qubits, and qubits left in their reset state
(which ``output_candidates`` does not list). Conversely a candidate need not be deterministic (a qubit measured, then
gated again, depends on the outcome), and an ``OUTPUT_QUBITS`` port refuses a qubit measured by M/MX/MY/MR/...
anywhere in the circuit (MPP data qubits are accepted), so pass the list, not a port. Exact GF(2) algebra on the
circuit's non-Clifford rotations; no amplitudes. Cost: one rotation-form build per call, plus up to r shift steps (r
the odd columns' rank), each roughly O(m r^3 + r^4)/128 word operations for m odd columns (skipped when the columns
are independent).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import _xtim
from .circuit import Circuit, _refuse_unresolved_if
from .errors import TwirlClass, XtimFidelityRefusal, XtimTwirlRefusal

__all__ = ["OutputState", "output_candidates", "output_is_deterministic", "output_state"]

_RF = re.compile(r"(RF_[A-Z_]+): ")


def _text(circuit, surface: str) -> str:
    c = circuit if isinstance(circuit, Circuit) else Circuit(str(circuit))
    _refuse_unresolved_if(c.text, surface)
    return c.text


def _prepare(circuit, qubits, surface: str):
    text = _text(circuit, surface)
    qs = []
    for q in qubits:
        if isinstance(q, bool) or not hasattr(q, "__index__"):
            raise TypeError(f"{surface}: qubits must be ints, got {q!r}")
        q = int(q.__index__())
        if not -2**31 <= q < 2**31:                    # final review m3: refused by name, not by pybind's TypeError
            raise ValueError(f"{surface}: malformed qubit list: qubit {q} is not a qubit of the circuit")
        qs.append(q)
    return text, qs


def _engine(surface: str, fn, *args):
    try:
        return fn(*args)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("bare state rejected: "):
            raise XtimTwirlRefusal(msg, TwirlClass.OUTSIDE_T_DEPTH_1, "bare") from None
        m = _RF.match(msg)
        if m:
            raise XtimFidelityRefusal(msg, m.group(1)) from None
        raise ValueError(f"{surface}: {msg}") from None     # the binding's messages carry no surface name


@dataclass(frozen=True)
class OutputState:
    """The noiseless output state on ``qubits`` as ``|psi> = R |S>``.

    Every Pauli is a pair ``(pauli, sign)``: ``pauli`` an unsigned product in the circuit's qubit indices, in MPP target
    notation (``"X21*Y22*Z27"``), ``sign`` +1 or -1 (the operator is ``sign * pauli``; ``MPP pauli`` reads 1 iff
    ``sign == -1``).

    - ``rotations``: ``R = prod_k exp(-i sign_k pi/8 P_k)``. The axes commute, so the order is immaterial. They are the
      circuit's own (T-count-reduced) rotation axes, localized to the output; they are not reduced against ``|S>``.
    - ``pauli_stabilizers``: generators of the Pauli stabilizer group of ``|psi>`` itself (exact, the full group), in
      reduced row echelon form on the X|Z matrix (X columns, then Z columns, ascending qubit), so a CSS group comes
      back as X-checks, then Z-checks. Measure them directly on the output; decode from them first.
    - ``clifford_paulis``: Paulis P that, with ``pauli_stabilizers``, generate the stabilizer group of ``|S>``. The
      Clifford stabilizers of the output are ``R P R^dag`` (a Clifford, since the axes commute). To measure one, undo
      the rotations (``exp(+i sign pi/8 P)``) and then measure P.
    - The split between R and |S> is deterministic for a given circuit, but not unique (``T_DAG|+>`` may come out as
      ``(Z, -1)`` with ``X``, or as ``(Z, +1)`` with ``-Y``), and no claim is made about which stabilizers are logicals.
      Every field except ``qubits`` (which keeps the caller's order) is the same for any order of ``qubits``.
    - Limitation: an undo built from ``rotations`` may be refused by xtim when the localized axes do not commute in one
      frame with the circuit's own (a localized axis is the circuit's axis times stabilizers of the rest; xtim then
      refuses "non-commuting magic" on "circuit + undo"; in principle also "even-Clifford ladder gap", when a pair still
      anticommutes after the bare build's one settle of the even terms, never observed). The stabilizers are still exact.
    """

    qubits: tuple
    rotations: tuple
    pauli_stabilizers: tuple
    clifford_paulis: tuple


def output_candidates(circuit) -> list:
    """The qubits live at the end of ``circuit``: acted on by a gate or feedback, with no measurement or reset ON THEM
    after their last gate, ascending. An MPP measures an ancilla, so its qubits stay live (as in stim); MX, MR, RX, ...
    end a qubit. No grouping. A candidate is not a promise: :func:`output_is_deterministic` decides it (``M 0; H 0``
    lists qubit 0, whose state depends on the outcome)."""
    return list(_xtim.output_candidates(_text(circuit, "output_candidates")))


def output_is_deterministic(circuit, qubits, explain: bool = False):
    """True iff the noiseless state on ``qubits`` is pure and the same for every measurement outcome. Exact: every
    True and False is proven. ``explain=True`` returns ``(bool, message)``: "" when True, else the
    ``RF_OUTPUT_NOT_DETERMINISTIC`` text naming the cause (undetermined directions and the measurements that fix them;
    a pi/8 rotation coupling the output to the rest; or an entangling stabilizer part).

    It accepts outputs that :class:`xtim.RecordFidelity`'s sufficient-only strict contract refuses (magic on both the
    output and the rest). Refused by name:
    - ``RF_OUTPUT_UNDECIDED``: pi/8 rotations straddle the output and the rest in a way xtim does not decide. This
      usually means the output is entangled with the rest through the magic; the message names the rotations.
    - ``RF_OUTPUT_MEASURED``: a qubit measured after its last gate.
    - :class:`XtimTwirlRefusal`: a circuit outside xtim's model.
    - ValueError: IF blocks, ``INPUT_QUBITS``, or a malformed qubit list (empty, repeated, negative, or a qubit the
      circuit does not have). TypeError: a non-int qubit."""
    text, qs = _prepare(circuit, qubits, "output_is_deterministic")
    ok, why = _engine("output_is_deterministic", _xtim.output_state_check, text, qs)
    return (bool(ok), str(why)) if explain else bool(ok)


def output_state(circuit, qubits) -> OutputState:
    """The noiseless output state of ``circuit`` on ``qubits`` (see :class:`OutputState`). Refused by name when
    :func:`output_is_deterministic` is False (``RF_OUTPUT_NOT_DETERMINISTIC``, with its message), and for the same
    inputs it refuses. Intended use: measure ``pauli_stabilizers``, decode, apply the Pauli correction, undo
    ``rotations``, then measure ``clifford_paulis``. An undo built from ``rotations`` may be refused by xtim when the
    localized axes do not commute in one frame (see :class:`OutputState`); the stabilizers are still exact."""
    text, qs = _prepare(circuit, qubits, "output_state")
    d = _engine("output_state", _xtim.output_state, text, qs)
    conv = lambda key: tuple((str(p), int(s)) for p, s in d[key])
    return OutputState(qubits=tuple(qs), rotations=conv("rotations"),
                       pauli_stabilizers=conv("pauli_stabilizers"), clifford_paulis=conv("clifford_paulis"))
