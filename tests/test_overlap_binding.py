"""FramedSuperposition.overlap (3.1.0): the exact overlap VALUE |<a|b>|^2 that
approx_equal only thresholds.  Oracle: closed-form overlaps of small stabilizer
states, reached by mutating a copy of one retained state with apply_clifford."""
import math

import pytest

from xtim import _xtim


def _state(text: str):
    """One retained shot of a portless producer + the state indices of its output wires."""
    from xtim import compile_twirl_sampler
    s = compile_twirl_sampler(text)
    buf = s.sample_barrier(1, 0)
    return buf.materialize(0), list(s.output_wires())


def test_binding_exists():
    assert hasattr(_xtim.FramedSuperposition, "overlap")


BELL = "H 0\nCX 0 1\nR 2\nM 2\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n"


def test_overlap_of_a_state_with_itself_is_one():
    a, _ = _state(BELL)
    b, _ = _state(BELL)
    assert a.overlap(a) == pytest.approx(1.0, abs=1e-12)
    assert a.overlap(b) == pytest.approx(1.0, abs=1e-12)
    assert a.approx_equal(b)


def test_overlap_matches_closed_form_after_single_qubit_mutations():
    a, out = _state(BELL)
    q0, q1 = out
    # X on one half of a Bell pair: orthogonal (|<Φ+|Ψ+>|^2 = 0)
    b, _ = _state(BELL)
    b.apply_clifford(3, q0, 0)
    assert b.overlap(a) == pytest.approx(0.0, abs=1e-12)
    assert not a.approx_equal(b)
    # Z on one half: also orthogonal (Φ-)
    c, _ = _state(BELL)
    c.apply_clifford(5, q1, 0)
    assert c.overlap(a) == pytest.approx(0.0, abs=1e-12)
    # S on one half: |<Φ+|(S⊗I)|Φ+>|^2 = |(1 + i)/2|^2 = 1/2
    d, _ = _state(BELL)
    d.apply_clifford(1, q0, 0)
    assert d.overlap(a) == pytest.approx(0.5, abs=1e-12)
    # H on one half: |<Φ+|(H⊗I)|Φ+>|^2 = |Tr(H)/2|^2 = 0 (H is traceless)
    e, _ = _state(BELL)
    e.apply_clifford(0, q1, 0)
    assert e.overlap(a) == pytest.approx(0.0, abs=1e-12)
    # SDG then H on one half: |Tr(H·SDG)/2|^2 = |(1 - i)/(2√2)|^2 = 1/4
    f, _ = _state(BELL)
    f.apply_clifford(2, q1, 0)
    f.apply_clifford(0, q1, 0)
    assert f.overlap(a) == pytest.approx(0.25, abs=1e-12)
    # symmetric and global-phase insensitive
    assert a.overlap(d) == pytest.approx(d.overlap(a), abs=1e-12)


def test_overlap_refuses_states_on_different_qubit_counts():
    a, _ = _state(BELL)
    b, _ = _state("H 0\nCX 0 1\nCX 1 2\nR 3\nM 3\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1 2\n")
    assert a.n != b.n
    with pytest.raises(ValueError, match="different qubit counts"):
        a.overlap(b)


def test_overlap_is_consistent_with_approx_equal_threshold():
    a, out = _state(BELL)
    d, _ = _state(BELL)
    d.apply_clifford(1, out[0], 0)          # overlap 1/2
    assert a.approx_equal(d, tol=1 - math.sqrt(0.5) + 1e-9)
    assert not a.approx_equal(d, tol=1 - math.sqrt(0.5) - 1e-9)
