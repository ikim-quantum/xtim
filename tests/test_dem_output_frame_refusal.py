"""Pins the OUTPUT_QUBITS frame refusal of the DEM export (docs/usage.md, "Pauli feedback ... exports too"): an
OUTPUT_QUBITS frame whose feedback controls are noiselessly random outcomes is refused when a fault randomizes one
of them.

The miniature has the structure of a CCZ magic-state factory with a teleported output: three |+> wires, a CCZ,
a one-bit teleport of each wire onto a fresh output wire (CX, then MX of the CCZ wire), and the Z byproduct closed
by Pauli feedback on the output. An X fault before the CCZ leaves a CZ residue on the other two CCZ wires. That
residue randomizes their X readouts, which control the feedback, so the output frame has no Pauli flip.

Controls:
- the same fault after the CCZ (a Pauli) exports;
- the byproduct kept as a classical record frame instead (no feedback; a PAULI_EXPECTATION frame on each output X)
  exports with every column kept, and dropping one record from a frame makes the export report that column (its
  noiseless value then varies with the reads). This pins the frame check, not the completeness of the columns:
  three single-qubit X columns do not see every effect of a CZ residue.
"""
import pytest

import xtim
from xtim.errors import XtimDemError

stim = pytest.importorskip("stim")

HEAD = "RX 0 1 2\nR 3 4 5\n"
TELEPORT = "CX 0 3 1 4 2 5\nMX 0 1 2\n"
FEEDBACK = "CZ rec[-3] 3\nCZ rec[-2] 4\nCZ rec[-1] 5\nOUTPUT_QUBITS out 3 4 5\n"


def _dem(text):
    return xtim.Circuit(text).detector_error_model_text(clifford_cutoff=2)


def test_prefault_ccz_teleport_feedback_refuses_by_name():
    text = HEAD + "X_ERROR(0.01) 0\nCCZ 0 1 2\n" + TELEPORT + FEEDBACK
    with pytest.raises(XtimDemError, match=r"X_ERROR \(user noise site 0, .*randomizes a feedback control record "
                                           r"that sets the Z frame of OUTPUT_QUBITS qubit (4|5) \(port 'out'\); "
                                           r"that frame's noiseless value is not deterministic"):
        _dem(text)
    with pytest.raises(XtimDemError, match="randomizes a feedback control record"):   # the exact export too
        xtim.Circuit(text).detector_error_model_text()


def test_postfault_ccz_teleport_feedback_exports():
    _dem(HEAD + "CCZ 0 1 2\nX_ERROR(0.01) 0\n" + TELEPORT + FEEDBACK)


# The pre-CCZ X fault heralded: CX 0 6 before and after the CCZ (diagonal, so it commutes with the Z0 Z6 check)
# returns ancilla 6 to |0> noiselessly; an X on wire 0 between them flips its read. The byproduct is a record frame.
HERALD = "RX 0 1 2\nR 3 4 5 6\nCX 0 6\nX_ERROR(0.01) 0\nCCZ 0 1 2\nCX 0 6\nM 6\nDETECTOR rec[-1]\n"
FRAMED = HERALD + TELEPORT + ("PAULI_EXPECTATION(0) X3 rec[-3]\nPAULI_EXPECTATION(1) X4 rec[-2]\n"
                              "PAULI_EXPECTATION(2) X5 rec[-1]\n")


def test_record_frame_form_exports_and_keeps_its_columns():
    w = xtim.Circuit(FRAMED).detector_error_model_with_reject(clifford_cutoff=2)
    assert w.dropped_expectation_columns == []
    assert w.dem.num_observables == 3


def test_record_frame_form_reports_a_wrong_frame():
    wrong = FRAMED.replace("PAULI_EXPECTATION(1) X4 rec[-2]", "PAULI_EXPECTATION(1) X4")
    with pytest.warns(Warning):
        w = xtim.Circuit(wrong).detector_error_model_with_reject(clifford_cutoff=2)
    assert [i for i, _ in w.dropped_expectation_columns] == [1]


def test_bundled_miniature_oracle_refuses_by_name():
    """The bundled example closes its teleport frame by feedback after T gates (same class)."""
    with pytest.raises(XtimDemError, match=r"randomizes a feedback control record that sets the Z frame of "
                                           r"OUTPUT_QUBITS qubit 0 \(port 'out'\)"):
        xtim.load_example("miniature_oracle").detector_error_model_text()
