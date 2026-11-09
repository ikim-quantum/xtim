"""OUTPUT_QUBITS adds no DEM column (docs/usage.md, owner 2026-11-03): DEM export ignores the declaration, so a
circuit exports exactly as its twin without it. A logical flip of the output is exported only through a declared
PAULI_EXPECTATION, whose completeness gate (flag / refusal) is unchanged.

The miniature has the structure of a CCZ magic-state factory with a teleported output: three |+> wires, a CCZ,
a one-bit teleport of each wire onto a fresh output wire (CX, then MX of the CCZ wire), and the Z byproduct closed
by Pauli feedback on the output. An X fault before the CCZ leaves a CZ residue on the other two CCZ wires; that
residue randomizes their X readouts, which control the feedback. Until the per-qubit frame columns were removed this
was refused; it now exports.

Controls:
- the byproduct kept as a classical record frame instead (no feedback; a PAULI_EXPECTATION frame on each output X)
  exports with every column kept, and dropping one record from a frame makes the export report that column (its
  noiseless value then varies with the reads).
- a logical PAULI_EXPECTATION after the feedback keeps its exact law; with the syndrome check that resolves the
  logical branch removed the fault is flagged, and with no detector it fires the export refuses (completeness gate).
"""
import math

import pytest

import xtim
from xtim import _xtim
from xtim.errors import XtimDemError

stim = pytest.importorskip("stim")

HEAD = "RX 0 1 2\nR 3 4 5\n"
TELEPORT = "CX 0 3 1 4 2 5\nMX 0 1 2\n"
FEEDBACK = "CZ rec[-3] 3\nCZ rec[-2] 4\nCZ rec[-1] 5\nOUTPUT_QUBITS out 3 4 5\n"
PREFAULT = HEAD + "X_ERROR(0.01) 0\nCCZ 0 1 2\n" + TELEPORT + FEEDBACK

# A CZ residue created directly on two encoded outputs (Bell pairs (4,6), (5,7), X- and Z-checked); no feedback.
D_DIR = ("R 0 6 7\nRX 4 5\nCX 4 6 5 7\nCCZ 0 4 5\nX_ERROR(0.3) 0\nCCZ 0 4 5\nM 0\nDETECTOR rec[-1]\n"
         "MPP X4*X6\nMPP Z4*Z6\nMPP X5*X7\nMPP Z5*Z7\n" + "".join(f"DETECTOR rec[-{k}]\n" for k in (4, 3, 2, 1)) +
         "OUTPUT_QUBITS out 4 5 6 7\n")
# The same residue reached through the teleport + Z feedback (factory d's shape).
C_FB = ("R 0 3 4 5 6 7\nRX 1 2\nCX 1 6 2 7\nCCZ 0 1 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nCX 0 3 1 4 2 5\nMX 0 1 2\n"
        "CZ rec[-3] 3\nCZ rec[-2] 4\nCZ rec[-1] 5\nMPP Z3\nMPP X4*X6\nMPP Z4*Z6\nMPP X5*X7\nMPP Z5*Z7\n" +
        "".join(f"DETECTOR rec[-{k}]\n" for k in (5, 4, 3, 2, 1)) + "OUTPUT_QUBITS out 3 4 5 6 7\n")
# Phase-flip-code output (4, 6, 8) with logical X-bar = X4 declared after the feedback.
G_HEAD = ("R 0 3 4 5 7\nRX 1 2 6 8\nCX 2 7\nCCZ 0 1 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nCX 0 3 1 4 2 5\nMX 0 1 2\n"
          "CZ rec[-3] 3\nCZ rec[-2] 4\nCZ rec[-1] 5\n")
G_OUT = "OUTPUT_QUBITS out 3 4 5 6 7 8\n"
G_PE = (G_HEAD + "MPP Z3\nMPP X4*X6\nMPP X6*X8\nMPP X5*X7\nMPP Z5*Z7\n" +
        "".join(f"DETECTOR rec[-{k}]\n" for k in (5, 4, 3, 2, 1)) + "PAULI_EXPECTATION(0) X4\n" + G_OUT)
G_NOCHK = (G_HEAD + "MPP Z3\nMPP X6*X8\nMPP X5*X7\nMPP Z5*Z7\n" +            # X4*X6 unmeasured
           "".join(f"DETECTOR rec[-{k}]\n" for k in (4, 3, 2, 1)) + "PAULI_EXPECTATION(0) X4\n" + G_OUT)
G_NODET = (G_HEAD + "MPP X6*X8\nMPP Z5*Z7\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"   # no check sees the fault
           "PAULI_EXPECTATION(0) X4\n" + G_OUT)

CUTOFFS = [2, math.inf]


def _strip(text):
    return "".join(line + "\n" for line in text.splitlines() if not line.startswith("OUTPUT_QUBITS"))


def _dem(text, cut):
    return xtim.Circuit(text).detector_error_model_text(clifford_cutoff=cut)


@pytest.mark.parametrize("cut", CUTOFFS)
@pytest.mark.parametrize("name", ["PREFAULT", "D_DIR", "C_FB", "G_PE", "G_NOCHK"])
def test_output_qubits_is_dem_inert(name, cut):
    text = globals()[name]
    assert "OUTPUT_QUBITS" in text
    assert _dem(text, cut) == _dem(_strip(text), cut)


@pytest.mark.parametrize("cut", CUTOFFS)
def test_prefault_ccz_teleport_feedback_exports(cut):
    w = xtim.Circuit(PREFAULT).detector_error_model_with_reject(clifford_cutoff=cut)
    assert w.dem.num_observables == 0 and w.postselect_faults == []


def test_postfault_ccz_teleport_feedback_exports():
    _dem(HEAD + "CCZ 0 1 2\nX_ERROR(0.01) 0\n" + TELEPORT + FEEDBACK, 2)


@pytest.mark.parametrize("text", [D_DIR, C_FB], ids=["D_dir", "C_fb"])
def test_cz_residue_on_outputs_exports_detectors_only(text):
    """Exact law: D0 deterministic, independent fair coins on the two partner X-checks (D1, D3); no L column."""
    dem = _dem(text, math.inf)
    assert "L" not in dem
    errs = sorted(line.split(" ", 1)[1] for line in dem.splitlines() if line.startswith("error"))
    assert errs == ["D0", "D0 D1", "D0 D1 D3", "D0 D3"], dem
    assert xtim.Circuit(text).detector_error_model(clifford_cutoff=math.inf).num_observables == 0


@pytest.mark.parametrize("cut", CUTOFFS)
def test_logical_expectation_after_feedback_keeps_its_law(cut):
    w = xtim.Circuit(G_PE).detector_error_model_with_reject(clifford_cutoff=math.inf)
    assert w.dem.num_observables == 1 and w.postselect_faults == []
    errs = sorted(line.split(" ", 1)[1] for line in str(w.dem).splitlines() if line.startswith("error"))
    assert errs == ["D0", "D0 D1 D3 L0", "D0 D1 L0", "D0 D3"], w.dem
    _dem(G_PE, cut)


@pytest.mark.parametrize("cut", CUTOFFS)
def test_unresolved_logical_branch_is_flagged(cut):
    w = xtim.Circuit(G_NOCHK).detector_error_model_with_reject(clifford_cutoff=cut)
    assert [(f.detectors, f.probability) for f in w.postselect_faults] == [([0, 2], 0.3)]


@pytest.mark.parametrize("cut", CUTOFFS)
def test_undetected_logical_magnitude_change_refuses(cut):
    with pytest.raises(XtimDemError, match="changes the magnitude of a logical expectation but fires no detector"):
        _dem(G_NODET, cut)


def test_column_layout_has_no_output_columns():
    lay = _xtim.dem_column_layout(C_FB + "DECISION(0) rec[-1]\n")
    assert lay["ok"], lay
    assert "frame" not in lay and lay["n_obs"] == 0 and lay["logical_flip"] == [(0, 0)], lay


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


@pytest.mark.parametrize("cut", CUTOFFS)
def test_bundled_miniature_oracle_exports(cut):
    """The bundled example closes its teleport frame by feedback after T gates; it exports like its twin."""
    text = str(xtim.load_example("miniature_oracle"))
    assert "OUTPUT_QUBITS" in text
    assert _dem(text, cut) == _dem(_strip(text), cut)
    assert xtim.load_example("miniature_oracle").detector_error_model(clifford_cutoff=cut).num_observables == 0
