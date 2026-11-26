"""Output-state helpers' API (PLAN.md Task 3)."""
import pathlib

import pytest
import stim

import xtim
import _output_state_oracle as O

DATA = pathlib.Path(__file__).parent / "data"
TELEPORT = "RX 0\nT 0\nR 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nM 0 1\nCX rec[-1] 2\nCZ rec[-2] 2\n"
TELEPORT_NO_Z = TELEPORT.replace("CZ rec[-2] 2\n", "")
CULT_OUT = list(range(21, 28))
CULT_CANDIDATES = [0, 3, 7, 9, 10, 12, 13] + CULT_OUT          # I1: the final SWAP leaves the old wires live (|0>)
STRADDLE_4A = "H 0\nT 0\nH 0\nH 0 1\nCX 0 1\nT 1\nCX 0 1\nH 0 1\n"   # X0 and X0X1 rotations on |00>
CCZ = "H 0 1 2\nCCZ 0 1 2\n"


@pytest.mark.parametrize("text,want", [
    ("H 0\nCX 0 1\nM 1\n", [0]), ("H 0\nM 0\nH 0\n", [0]), ("H 0\nMR 0\n", []), ("RX 0\n", []),
    ("H 0\nR 0\n", []), (TELEPORT, [2]),
])
def test_candidates_basic(text, want):
    assert xtim.output_candidates(text) == want


def test_candidates_desugared_instructions():          # Review Focus 3 (C1 / Q2)
    assert xtim.output_candidates("H 0 1\nMPP X0*X1\n") == [0, 1]
    assert xtim.output_candidates("H 0\nMX 0\n") == []
    assert xtim.output_candidates("H 0\nMRX 0\n") == []
    assert xtim.output_candidates("H 0\nRX 0\n") == []
    assert xtim.output_candidates("H 0\nM 0\nCX rec[-1] 1\n") == [1]


def test_candidates_circuit_object_and_repeat():
    c = xtim.load_example("cultivation_d3_faithful")
    assert xtim.output_candidates(c) == CULT_CANDIDATES
    assert xtim.output_candidates(c.text) == CULT_CANDIDATES
    assert xtim.output_candidates("REPEAT 3 {\n  H 0\n  M 0\n}\n") == []
    assert xtim.output_candidates("REPEAT 3 {\n  M 0\n  H 0\n}\n") == [0]


def test_mpp_read_outputs_are_accepted():               # C1 fixture
    c = xtim.load_example("distillation_15_1_3")
    assert set(range(15)) <= set(xtim.output_candidates(c))
    assert xtim.output_is_deterministic(c, range(15)) is True


def test_deterministic_cases():
    assert xtim.output_is_deterministic("H 0\nT 0\n", [0]) is True
    assert xtim.output_is_deterministic(TELEPORT, [2]) is True
    assert xtim.output_is_deterministic("H 0 1\nT 0 1\nM 1\n", [0]) is True      # I2
    assert xtim.output_is_deterministic(CCZ, [0, 1, 2]) is True
    assert xtim.output_is_deterministic("H 0\nCX 0 1\nM 1\nCX rec[-1] 0\n", [0]) is True
    assert xtim.output_is_deterministic("H 0\nT 0\n", [0], explain=True) == (True, "")


def test_not_deterministic_messages():                  # decision 6
    ok, why = xtim.output_is_deterministic("H 0\nCX 0 1\n", [0], explain=True)
    assert not ok and why.startswith("RF_OUTPUT_NOT_DETERMINISTIC:") and "Undetermined output directions" in why
    assert "q1" in why
    ok, why = xtim.output_is_deterministic(TELEPORT_NO_Z, [2], explain=True)
    assert not ok and "measurements [0]" in why and "CZ rec[-k]" in why
    ok, why = xtim.output_is_deterministic(STRADDLE_4A, [0], explain=True)       # Lemma 4a
    assert not ok and "rotation couples" in why and "X0" in why


def test_adaptq_born5_is_not_deterministic():           # the probe's one in-repo straddler: G does not split
    c = xtim.Circuit.from_file(DATA / "adaptq_born5_dec_enum_producer.stim")
    qubits = [int(t) for t in next(l for l in c.text.splitlines() if l.strip().startswith("OUTPUT_QUBITS")).split()[2:]]
    ok, why = xtim.output_is_deterministic(c, qubits, explain=True)
    assert not ok and "Undetermined output directions" in why


def test_undecided_is_refused_by_name():                # A3, Review Focus 5
    with pytest.raises(xtim.XtimFidelityRefusal) as e:
        xtim.output_is_deterministic(CCZ, [0])
    assert e.value.code == "RF_OUTPUT_UNDECIDED"
    assert "entangled with the rest through the magic" in str(e.value) and "Z0" in str(e.value)


def _fixtures():
    for p in sorted((DATA / "output_state").glob("subset_*.stim")):
        text = p.read_text()
        out = [int(t) for t in next(l for l in text.splitlines() if l.startswith("# OUT ")).split()[2:]]
        yield p.name, text, out


LEMMA_4A_FIXTURES = {"subset_221.stim", "subset_222.stim"}


@pytest.mark.parametrize("name,text,out", list(_fixtures()))
def test_probe_fixtures(name, text, out):               # straddle_probe.md: all 13 are entangled
    if name in LEMMA_4A_FIXTURES:
        ok, why = xtim.output_is_deterministic(text, out, explain=True)
        assert not ok and why.startswith("RF_OUTPUT_NOT_DETERMINISTIC:")
    else:
        with pytest.raises(xtim.XtimFidelityRefusal) as e:
            xtim.output_is_deterministic(text, out)
        assert e.value.code == "RF_OUTPUT_UNDECIDED"


def test_refusals_by_name():
    with pytest.raises(xtim.XtimFidelityRefusal) as e:
        xtim.output_is_deterministic("H 0\nM 0\n", [0])
    assert e.value.code == "RF_OUTPUT_MEASURED" and "reset state" in str(e.value)    # M4
    with pytest.raises(xtim.XtimTwirlRefusal) as e:
        xtim.output_is_deterministic("H 0\nT 0\nH 0\nT 0\n", [0])
    assert e.value.twirl_class == xtim.TwirlClass.OUTSIDE_T_DEPTH_1
    for bad in ([], [0, 0], [-1], [5]):                     # Task 2 review m1: refused by name, each one
        with pytest.raises(ValueError, match="malformed qubit list"):
            xtim.output_is_deterministic("H 0\nCX 0 1\n", bad)
        with pytest.raises(ValueError, match="malformed qubit list"):
            xtim.output_state("H 0\nCX 0 1\n", bad)
    with pytest.raises(ValueError, match="INPUT_QUBITS"):
        xtim.output_is_deterministic("INPUT_QUBITS c 0\nH 0\n", [0])
    with pytest.raises(TypeError):
        xtim.output_is_deterministic("H 0\n", [True])


def test_refusal_messages_are_tidy():                   # final review m1
    with pytest.raises(xtim.XtimFidelityRefusal) as e:
        xtim.output_is_deterministic(CCZ, [1])
    msg = str(e.value)
    assert "on your qubits [Z1];" in msg and "measurements []" not in msg and "[q0, q2]" in msg
    for text in ("H 0 1\nCZ 0 1\nT 0\n", "H 0\nCX 0 1\n"):           # no records: no "feedback from those records"
        ok, why = xtim.output_is_deterministic(text, [0], explain=True)
        assert not ok and "measurements" not in why and "from those records" not in why
        assert "the unmeasured qubits [q1]" in why and "include them in the output" in why
    ok, why = xtim.output_is_deterministic(TELEPORT_NO_Z, [2], explain=True)   # records: the feedback advice stays
    assert "measurements [0]" in why and "from those records" in why


def test_validation_messages():                         # final review m3
    for bad in (2**40, -2**40):
        for f in (xtim.output_is_deterministic, xtim.output_state):
            with pytest.raises(ValueError, match="malformed qubit list"):
                f("H 0\n", [bad])
    with pytest.raises(ValueError, match="malformed qubit list") as e:
        xtim.output_is_deterministic("", [0])
    assert "0..-1" not in str(e.value) and "the circuit has no qubits" in str(e.value)
    for f, name in ((xtim.output_is_deterministic, "output_is_deterministic"), (xtim.output_state, "output_state")):
        for args in (("H 0\nCX 0 1\n", [0, 0]), ("H 0\nCX 0 1\n", [5]), ("INPUT_QUBITS c 0\nH 0\n", [0])):
            with pytest.raises(ValueError) as e:
                f(*args)
            assert str(e.value).startswith(name + ": ") and str(e.value).count("output_state:") == (name == "output_state")


IF_TEXT = "INPUT_BITS b 1\nH 0\nIF b[0] {\n  Z 0\n}\n"


def test_if_blocks_refused_by_name():                   # final review m5
    for f, args in ((xtim.output_candidates, ()), (xtim.output_is_deterministic, ([0],)), (xtim.output_state, ([0],))):
        with pytest.raises(ValueError, match=f"{f.__name__}: the circuit has IF blocks on INPUT_BITS"):
            f(IF_TEXT, *args)


def test_output_state_toys():
    st = xtim.output_state("H 0\nT 0\n", [0])
    assert isinstance(st, xtim.OutputState)
    assert st.qubits == (0,) and st.rotations == (("Z0", 1),) and st.pauli_stabilizers == () and st.clifford_paulis == (("X0", 1),)
    # M6: the parser writes T_DAG as `T; S_DAG`, so T_DAG|+> = exp(+i pi/8 Z)|+> = exp(-i pi/8 Z)|-i>; both exact forms
    # pass (decision 2: the completing Pauli may be X or Y) and both wrong-sign forms fail (cpp/tests/test_output_state.cpp).
    st = xtim.output_state("H 0\nT_DAG 0\n", [0])
    assert st.pauli_stabilizers == ()
    assert (st.rotations, st.clifford_paulis) in ((((("Z0", -1),), (("X0", 1),))), ((("Z0", 1),), (("Y0", -1),)))
    assert (st.rotations, st.clifford_paulis) not in (((("Z0", 1),), (("X0", 1),)), ((("Z0", -1),), (("Y0", -1),)))
    st = xtim.output_state(TELEPORT, [2])
    assert st.rotations == (("Z2", 1),) and st.clifford_paulis == (("X2", 1),)
    st = xtim.output_state("H 0 1\nT 0 1\nM 1\n", [0])                            # I2
    assert st.rotations == (("Z0", 1),) and st.clifford_paulis == (("X0", 1),)
    with pytest.raises(xtim.XtimFidelityRefusal) as e:
        xtim.output_state("H 0\nCX 0 1\n", [0])
    assert e.value.code == "RF_OUTPUT_NOT_DETERMINISTIC"


def _ps(p, n):
    s = stim.PauliString(n)
    for t in p.split("*"):
        s[int(t[1:])] = "_XYZ".index(t[0])
    return s


def test_unsorted_qubits_every_field():                 # Review Focus 1, M7
    c = xtim.load_example("cultivation_d3_faithful")
    a = xtim.output_state(c, CULT_OUT)
    b = xtim.output_state(c, [27, 21, 26, 22, 25, 23, 24])
    assert (a.rotations, a.pauli_stabilizers, a.clifford_paulis) == (b.rotations, b.pauli_stabilizers, b.clifford_paulis)
    assert a.qubits == tuple(CULT_OUT) and b.qubits == (27, 21, 26, 22, 25, 23, 24)     # qubits keeps the caller's order


def test_pauli_stabilizers_reduced_form():              # final review N4: RREF on X|Z, X pivots first
    st = xtim.output_state(xtim.load_example("cultivation_d3_faithful"), CULT_OUT)
    assert st.pauli_stabilizers == (("X21*X24*X25*X26", 1), ("X22*X24*X25*X27", 1), ("X23*X25*X26*X27", 1),
                                    ("Z21*Z24*Z25*Z26", 1), ("Z22*Z24*Z25*Z27", 1), ("Z23*Z25*Z26*Z27", 1))
    st = xtim.output_state("H 0\nCX 0 1\nS 1\n", [0, 1])                            # non-CSS: Y0*Y1 -> X0*Y1, Z0*Z1
    assert st.pauli_stabilizers == (("X0*Y1", 1), ("Z0*Z1", 1))


def test_noise_is_ignored():                            # Review Focus 2
    c = xtim.load_example("cultivation_d3_faithful")
    assert xtim.output_state(c, CULT_OUT) == xtim.output_state(c.without_noise(), CULT_OUT)


def test_reset_and_untouched_qubits():                  # Review Focus 4, M4
    st = xtim.output_state("H 0\nT 0\nR 1\nRX 2\nCX 0 4\nCX 0 4\n", [0, 1, 2, 3])
    g = O.SignedGroup(st.pauli_stabilizers, 5)
    assert g.member_sign(_ps("Z1", 5)) == 1                 # R: +Z
    assert g.member_sign(_ps("X2", 5)) == 1                 # RX: +X (its reset state)
    assert g.member_sign(_ps("Z3", 5)) == 1                 # untouched: +Z


def test_candidate_domain_differs_from_helpers_2_3():  # Task 2 review m3, plan M4
    # A candidate need not be deterministic: measured, then re-gated, its state depends on the outcome.
    assert xtim.output_candidates("H 0\nM 0\nH 0\n") == [0]
    ok, why = xtim.output_is_deterministic("H 0\nM 0\nH 0\n", [0], explain=True)
    assert not ok and "measurements [0]" in why
    with pytest.raises(xtim.XtimFidelityRefusal) as e:
        xtim.output_state("H 0\nM 0\nH 0\n", [0])
    assert e.value.code == "RF_OUTPUT_NOT_DETERMINISTIC"
    # ... nor declarable as OUTPUT_QUBITS (the port refuses a qubit measured anywhere): pass the list, not a port.
    with pytest.raises(xtim.XtimParseError, match="measured in the circuit"):
        xtim.Circuit("H 0\nM 0\nH 0\nOUTPUT_QUBITS o 0\n")
    # MPP data qubits are candidates; deterministic or not by the state, as any qubit.
    assert xtim.output_candidates("H 0 1\nMPP Z0*Z1\n") == [0, 1]
    assert xtim.output_is_deterministic("H 0 1\nMPP Z0*Z1\n", [0, 1]) is False
    assert xtim.output_is_deterministic("H 0 1\nMPP X0*X1\n", [0, 1]) is True
    # Helpers 2 and 3 accept qubits candidates does not list: reset and untouched ones.
    assert xtim.output_candidates("H 0\nR 1\n") == [0]
    assert xtim.output_state("H 0\nR 1\n", [1]).pauli_stabilizers == (("Z1", 1),)
