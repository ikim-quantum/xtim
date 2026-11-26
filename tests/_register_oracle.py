"""Test-only DENSE partial-trace oracle for the register RecordFidelity (never production).

rho_O(state) = 2^-m sum_{P on O} <P> P is built as a dense 2^m x 2^m matrix from xtim's exact per-state Pauli
expectations (FramedSuperposition.pauli_expectation; no RecordFidelity code), O = the OUTPUT_QUBITS wires. The register
fidelity of a shot is F = Re Tr[rho_O(B) . P_c rho_O(psi) P_c] (= <phi| P_c Tr_R(rho) P_c |phi> when rho_O(B) is the
pure |phi><phi|), psi = materialize(i), B = the noiseless bare state, P_c = X^cx Z^cz on O.
"""
import itertools
from functools import reduce

import numpy as np

import xtim
from _public import to_prod

_PM = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]]), np.array([[0, -1j], [1j, 0]])]  # I X Z Y


def rho_out(st, wires):
    m = len(wires)
    n0 = st.pauli_expectation([], [], 0)
    rho = np.zeros((2 ** m, 2 ** m), complex)
    for code in itertools.product(range(4), repeat=m):
        xs = [int(w) for w, c in zip(wires, code) if c in (1, 3)]
        zs = [int(w) for w, c in zip(wires, code) if c in (2, 3)]
        e = st.pauli_expectation(xs, zs, sum(c == 3 for c in code) % 4) / n0
        rho += e * reduce(np.kron, [_PM[c] for c in code])
    return rho / 2 ** m


def bare_rho(txt, wires):
    return rho_out(xtim.bare_state_of(txt), wires)


def register_F(rhoB, buf, i, wires, c=None):
    """The oracle F of shot i (c: uint8[2, n_out] correction rows, or None)."""
    rho = rho_out(to_prod(buf.materialize(i)), wires)
    if c is not None and np.any(c):
        Pc = reduce(np.kron, [_PM[1] if c[0, j] else _PM[0] for j in range(len(wires))]) @ \
            reduce(np.kron, [_PM[2] if c[1, j] else _PM[0] for j in range(len(wires))])
        rho = Pc @ rho @ Pc.conj().T
    return float(np.trace(rhoB @ rho).real)


def purity(rhoB):
    return float(np.trace(rhoB @ rhoB).real)


def bare_terms(txt, wires):
    """The nonzero Pauli expansion of rho_O(B): [(xs, zs, ph, <P>_B)] (for outputs too wide for dense matrices)."""
    B = xtim.bare_state_of(txt)
    n0 = B.pauli_expectation([], [], 0)
    out = []
    for code in itertools.product(range(4), repeat=len(wires)):
        xs = [int(w) for w, c in zip(wires, code) if c in (1, 3)]
        zs = [int(w) for w, c in zip(wires, code) if c in (2, 3)]
        ph = sum(c == 3 for c in code) % 4
        e = B.pauli_expectation(xs, zs, ph) / n0
        if abs(e) > 1e-12:
            out.append((xs, zs, ph, e))
    return out


def register_F_terms(terms, st, m):
    """Tr[rho_O(B) rho_O(st)] = 2^-m sum_P <P>_B <P>_st over the nonzero terms of B."""
    n0 = st.pauli_expectation([], [], 0)
    return float(sum((e * st.pauli_expectation(xs, zs, ph) / n0).real for xs, zs, ph, e in terms) / 2 ** m)
