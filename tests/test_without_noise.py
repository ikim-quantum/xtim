"""Circuit.without_noise removes every noise instruction and every measurement flip probability, as stim's
without_noise does: tagged (`X_ERROR[t](p)`) and lower-case (`x_error(p)`) noise and noisy MPP / lower-case
measurements included (hunt 2026-11-20 item B), inside REPEAT blocks too. On every bundled example the result is
stim's without_noise (compared by stim on the text with the xtim-only lines replaced by tagged TICKs)."""
import re

import pytest

import xtim
from xtim.examples import example_path, list_examples

stim = pytest.importorskip("stim")

# Instructions stim does not parse; each such line becomes a TICK tagged with its own text (the same map on both sides).
_XTIM_ONLY = {"T", "T_DAG", "CCZ", "CS", "CS_DAG", "CH", "OUTPUT_QUBITS", "INPUT_QUBITS", "INPUT_BITS",
              "PAULI_EXPECTATION", "DECISION"}


def _head(line):
    return re.match(r"\s*([A-Za-z0-9_]*)", line).group(1).upper()


def _stimable(text):
    return "\n".join("TICK[" + re.sub(r"[^A-Za-z0-9_]", "_", l.strip()) + "]" if _head(l) in _XTIM_ONLY else l
                     for l in text.splitlines()) + "\n"


def test_hunt_repro_removes_all_noise():
    text = ("R 0\nX_ERROR[tag](0.1) 0\nx_error(0.1) 0\nMPP(0.1) Z0\nMXX(0.1) 0 1\nM(0.1) 0\nmx(0.1) 0\n")
    assert xtim.Circuit(text).without_noise().text == "R 0\nMPP Z0\nMXX 0 1\nM 0\nmx 0\n"


@pytest.mark.parametrize("line,want", [
    ("X_ERROR[t](0.1) 0", None), ("depolarize1(0.1) 0", None), ("Depolarize2[a b](0.1) 0 1", None),
    ("pauli_channel_1(0.1, 0, 0) 0", None), ("e(0.1) X0", None), ("I_ERROR\t0", None),
    ("MPP(0.1) X0*Z1", "MPP X0*Z1"), ("MPP[t](0.1) Z0", "MPP[t] Z0"), ("mr(0.1) 0", "mr 0"),
    ("MRX[q](0.1) 0", "MRX[q] 0"), ("myy(0.1) 0 1", "myy 0 1"), ("M 0", "M 0"), ("H 0", "H 0"),
])
def test_each_noise_spelling(line, want):
    got = xtim.Circuit("R 0 1\n" + line + "\n").without_noise().text.splitlines()
    assert got == ["R 0 1"] + ([want] if want else [])


def test_inside_repeat_blocks():
    text = "R 0\nREPEAT 3 {\n    m[t](0.2) 0\n    Depolarize1(0.1) 0\n    REPEAT 2 {\n        X_ERROR(0.1) 0\n    }\n}\n"
    w = xtim.Circuit(text).without_noise().text
    assert w == "R 0\nREPEAT 3 {\n    m[t] 0\n    REPEAT 2 {\n    }\n}\n"
    assert stim.Circuit(w) == stim.Circuit(text).without_noise()


def test_matches_stim_on_generated_noisy_circuit():
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=2,
                               after_clifford_depolarization=0.01, before_round_data_depolarization=0.01,
                               before_measure_flip_probability=0.01, after_reset_flip_probability=0.01)
    assert stim.Circuit(xtim.Circuit(str(c)).without_noise().text) == c.without_noise()


@pytest.mark.parametrize("name", list_examples())
def test_matches_stim_on_bundled_examples(name):
    text = open(example_path(name)).read()
    w = xtim.Circuit(text).without_noise().text
    assert stim.Circuit(_stimable(w)) == stim.Circuit(_stimable(text)).without_noise()
