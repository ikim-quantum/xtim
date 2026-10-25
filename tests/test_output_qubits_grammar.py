"""OUTPUT_QUBITS grammar: `OUTPUT_QUBITS <name> q0 q1 ...` — the first target is the port NAME.

A bare qubit index in the name position (`OUTPUT_QUBITS 0 1 2`) used to parse silently as a port
named "0" holding qubits 1 and 2 — qubit 0 dropped from the port without a word. It is now refused
by name (2026-10-13).
"""
import pytest

import xtim


@pytest.mark.parametrize("line", ["OUTPUT_QUBITS 0 1 2", "OUTPUT_QUBITS 7", "OUTPUT_QUBITS 12 0"])
def test_qubit_index_in_name_position_is_refused_by_name(line):
    with pytest.raises(xtim.XtimParseError) as ei:
        xtim.Circuit("R 0 1 2 7 12\n" + line + "\n")
    msg = str(ei.value)
    assert "OUTPUT_QUBITS" in msg and "port NAME" in msg
    assert "OUTPUT_QUBITS <name>" in msg


@pytest.mark.parametrize("name", ["out", "q0", "port_1", "a7"])
def test_named_port_parses_with_every_qubit(name):
    c = xtim.compile_twirl_sampler(
        "R 0 1 2 3\nM 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS %s 0 1 2\n" % name)
    assert len(list(c.output_wires())) == 3


def test_missing_name_still_refused():
    with pytest.raises(xtim.XtimParseError, match="missing output name"):
        xtim.Circuit("R 0\nOUTPUT_QUBITS\n")


# INPUT_QUBITS / INPUT_BITS share the grammar (`<KEYWORD> <name> ...`) and had the same silent
# mis-parse (`INPUT_QUBITS 0 1 2` -> port "0" holding qubits 1 and 2; `INPUT_BITS 3 1` -> port "3").
@pytest.mark.parametrize("line", ["INPUT_QUBITS 0 1 2", "INPUT_QUBITS 5", "INPUT_BITS 3 1"])
def test_number_in_input_name_position_is_refused_by_name(line):
    kw = line.split()[0]
    with pytest.raises(xtim.XtimParseError) as ei:
        xtim.Circuit(line + "\nR 0 1 2 5\nM 0\n")
    msg = str(ei.value)
    assert kw in msg and "port NAME" in msg and ("%s <name>" % kw) in msg


@pytest.mark.parametrize("line", ["INPUT_QUBITS corr 0 1 2", "INPUT_QUBITS q1 5", "INPUT_BITS dec 1"])
def test_named_input_port_parses(line):
    xtim.Circuit(line + "\nR 0 1 2 5\nM 0\n")
