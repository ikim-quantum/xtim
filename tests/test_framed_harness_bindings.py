"""The FramedSuperposition bindings added for adaptq's segmented Choi harness
(2026-10-07): copy, grow, permute, pauli_expectation(s), framed_from_rows.
All additive; the state algebra is checked against the existing reads and
overlap."""
import math

import numpy as np
import pytest

import xtim._xtim as _x

BELL = "R 0 1 2\nH 0\nCX 0 1\nM 2\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n"


def _bell():
    """The bare state of a Bell pair and the DEFERRED indices of its two
    qubits (every reset allocates a fresh wire, so circuit wires are not
    state indices)."""
    st = _x.bare_state_of(BELL)
    a, b = _x._output_wires_of(BELL)
    return st, int(a), int(b)


def test_copy_is_independent():
    a, p, q = _bell()
    b = a.copy()
    b.apply_clifford(5, p, 0)                       # Z on one Bell qubit of the copy
    assert abs(a.pauli_expectation_xz([p, q], []) - 1.0) < 1e-12
    assert abs(b.pauli_expectation_xz([p, q], []) + 1.0) < 1e-12


def test_grow_appends_zero_qubits():
    a, p, q = _bell()
    n = a.n
    a.grow(3)
    assert a.n == n + 3
    for w in range(n, n + 3):
        assert abs(a.pauli_expectation_z(w) - 1.0) < 1e-12
    assert abs(a.pauli_expectation_xz([p, q], []) - 1.0) < 1e-12


def test_permute_relabels_and_roundtrips():
    a, p, q = _bell()
    b = a.copy()
    n = a.n
    perm = list(range(n))
    perm[p], perm[q] = 0, 1
    perm[0], perm[1] = p, q                          # a transposition pair: Bell -> 0, 1
    b.permute(perm)
    assert abs(b.pauli_expectation_xz([0, 1], []) - 1.0) < 1e-12
    assert abs(b.pauli_expectation_xz([], [0, 1]) - 1.0) < 1e-12
    inv = [0] * n
    for q, t in enumerate(perm):
        inv[t] = q
    b.permute(inv)
    assert abs(a.overlap(b) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        a.copy().permute([0] * n)


def test_pauli_expectation_matches_the_xz_read_and_batches():
    a, p, q = _bell()
    assert abs(a.pauli_expectation([p, q], [], 0).real - a.pauli_expectation_xz([p, q], [])) < 1e-12
    # Y Y on the Bell pair = -1: the literal X^x Z^z with x = z = {p, q} and phase 2 (two Y sites)
    assert abs(a.pauli_expectation([p, q], [p, q], 2).real + 1.0) < 1e-12
    vals = a.pauli_expectations([([p, q], [], 0), ([], [p, q], 0), ([p, q], [p, q], 2)])
    assert [round(v.real, 9) for v in vals] == [1.0, 1.0, -1.0]


def test_framed_from_rows_rebuilds_a_bell_state():
    # Zrows: XX, ZZ ; Xrows: Z_0, X_1 — a symplectic frame; free empty; one amplitude
    st = _x.framed_from_rows(2, [[1, 1], [0, 0]], [[0, 0], [1, 1]], [0, 0],
                             [[0, 0], [0, 1]], [[1, 0], [0, 0]], [0, 0], [0, 0], [], [([], 1.0 + 0j)])
    assert st.n == 2 and st.k == 0
    assert abs(st.pauli_expectation_xz([0, 1], []) - 1.0) < 1e-12
    assert abs(st.pauli_expectation_xz([], [0, 1]) - 1.0) < 1e-12
    ref, p, q = _bell()
    # ref's Bell pair sits at deferred wires (p, q) among ref.n: grow st and move its pair there
    st.grow(ref.n - 2)
    perm = list(range(ref.n))
    perm[0], perm[1] = p, q
    perm[p], perm[q] = 0, 1
    st.permute(perm)
    assert abs(st.overlap(ref) - 1.0) < 1e-12


def test_framed_from_rows_rebuilds_a_magic_state():
    # |T> = (|0> + e^{i pi/4}|1>)/sqrt2 on one qubit: Zrow Z, Xrow X, free [0]
    c = 1 / math.sqrt(2)
    st = _x.framed_from_rows(1, [[0]], [[1]], [0], [[1]], [[0]], [0], [0], [0],
                             [([0], c + 0j), ([1], c * complex(math.cos(math.pi / 4), math.sin(math.pi / 4)))])
    assert st.k == 1
    assert abs(st.pauli_expectation_x(0) - math.cos(math.pi / 4)) < 1e-12
    assert abs(st.pauli_expectation_y(0) - math.sin(math.pi / 4)) < 1e-12
    assert abs(st.pauli_expectation_z(0)) < 1e-12


def test_framed_from_rows_refuses_bad_frames():
    with pytest.raises(ValueError, match="delta"):
        _x.framed_from_rows(2, [[1, 1], [0, 0]], [[0, 0], [1, 1]], [0, 0],
                            [[0, 0], [0, 0]], [[1, 0], [1, 0]], [0, 0], [0, 0], [], [([], 1.0 + 0j)])
    with pytest.raises(ValueError, match="Hermitian"):
        _x.framed_from_rows(1, [[0]], [[1]], [1], [[1]], [[0]], [0], [0], [], [([], 1.0 + 0j)])
    with pytest.raises(ValueError, match="norm"):
        _x.framed_from_rows(1, [[0]], [[1]], [0], [[1]], [[0]], [0], [0], [], [([], 0.5 + 0j)])
