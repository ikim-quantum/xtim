"""Q4 (README stage): RecordFidelity's RF_* refusals raise ``xtim.XtimFidelityRefusal`` (a ValueError) carrying the
refusal's name in ``.code``, and RF_OUTPUT_NOT_STRICT names the user's qubits, not internal deferred wires."""
import pathlib
import re
import warnings

import numpy as np
import pytest

import xtim

DATA = pathlib.Path(__file__).resolve().parent / "data"


def _compile(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(text, selfcheck=0)


def _refusal(text):
    with pytest.raises(xtim.XtimFidelityRefusal) as ei:
        xtim.RecordFidelity(_compile(text))
    e = ei.value
    assert isinstance(e, ValueError) and isinstance(e, xtim.XtimError)
    assert str(e).startswith(f"RecordFidelity: {e.code}: ")
    return e


def test_no_output_is_named():
    assert _refusal("RX 0\nT 0\nMX 0\nDECISION(0) rec[-1]\n").code == "RF_NO_OUTPUT"


def test_open_teleport_frame_names_the_user_qubit():
    """The original miniature oracle (its teleport byproduct closed only by a PAULI_EXPECTATION frame): the output
    is q0..q2, the undetermined direction is Z on q0 (deferred wire 10 internally), fixed by measurement #2."""
    text = (DATA / "miniature_oracle.stim").read_text()
    text = "".join(ln + "\n" for ln in text.splitlines() if not ln.startswith("PAULI_EXPECTATION"))
    e = _refusal(text + "OUTPUT_QUBITS out 0 1 2\n")
    m = str(e)
    assert e.code == "RF_OUTPUT_NOT_STRICT"
    assert "Undetermined output directions (Paulis on your qubits): Z0. " in m, m
    assert "measurements [2]" in m and "Z10" not in m, m
    # closing the frame as the message says serves it, F = 1 on every noiseless shot
    s = xtim.compile_twirl_sampler(text + "CZ rec[-3] 0\nOUTPUT_QUBITS out 0 1 2\n", p_factor=0.0, selfcheck=0)
    F = xtim.RecordFidelity(s).evaluate(s.sample_barrier(200, seed=1))
    assert np.all(np.abs(np.asarray(F) - 1) < 1e-12)


def test_code_switching_open_frame_names_steane_qubits_in_order():
    text = (DATA / "code_switching_faithful.stim").read_text()
    text = "".join(ln + "\n" for ln in text.splitlines() if not ln.startswith("PAULI_EXPECTATION"))
    m = str(_refusal(text + "OUTPUT_QUBITS out 15 16 17 18 19 20 21\n"))
    d = re.search(r"Paulis on your qubits\): ([^.]*)\.", m).group(1)
    qs = [int(t[1:]) for t in d.split("*")]
    assert qs == sorted(qs) and set(qs) <= set(range(15, 22)), m


def test_correction_of_the_wrong_width_is_refused():
    s = _compile(xtim.load_example("distillation_15_1_3").text)
    buf = s.sample_barrier(4, seed=1)
    bad = np.zeros((4, 2, 14), np.uint8)       # wrong port width
    with pytest.raises(ValueError):
        xtim.RecordFidelity(s).evaluate(buf, correction=bad)


_MEASURED_OUT = "RX 0\nT 0\nR 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nM 0 1\nOUTPUT_QUBITS out 0\n"


def test_measured_output_qubit_is_a_named_refusal_with_a_clean_message():
    with pytest.raises(xtim.XtimFidelityRefusal) as ei:
        _compile(_MEASURED_OUT)
    e = ei.value
    assert e.code == "RF_OUTPUT_MEASURED" and isinstance(e, ValueError)
    assert str(e).startswith("RF_OUTPUT_MEASURED: OUTPUT_QUBITS 'out': qubit 0 is measured")
    assert "line " not in str(e) and "R + SWAP" in str(e)


def test_measured_output_with_another_parse_error_stays_a_parse_error():
    with pytest.raises(ValueError) as ei:
        _compile(_MEASURED_OUT + "FOO 1\n")
    assert not isinstance(ei.value, xtim.XtimFidelityRefusal) and "unknown instruction" in str(ei.value)
