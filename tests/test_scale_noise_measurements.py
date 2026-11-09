"""xtim.scale_noise rescales measurement flip probabilities too (FCS item 3: the 4.1.0 rewrite left M(p), MR(p),
MX(p), MPP(p), ... at the baked-in strength, so a circuit written at p0 = 1e-3 and rescaled to p kept its readout noise
at 1e-3). Every noise argument of the text is scaled by p/p0, the measurement flip probabilities included; a
measurement without an argument is untouched; the line-count / surjectivity checks still hold."""
import re

import numpy as np
import pytest

import xtim

MEAS = ["M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY", "MPP", "MXX", "MYY", "MZZ"]


def _target(name):
    return {"MPP": "X0*X1 Z2", "MXX": "0 1", "MYY": "0 1", "MZZ": "0 1"}.get(name, "0 1")


@pytest.mark.parametrize("name", MEAS)
def test_measurement_flip_probability_is_rescaled(name):
    text = f"R 0 1 2\nX_ERROR(0.001) 0\n{name}(0.001) {_target(name)}\n{name} {_target(name)}\n"
    out = xtim.scale_noise(text, 0.02)
    lines = out.splitlines()
    assert lines[1] == "X_ERROR(0.02) 0"
    assert lines[2] == f"{name}(0.02) {_target(name)}"
    assert lines[3] == f"{name} {_target(name)}"          # no argument: untouched


def test_measurement_scaling_keeps_other_lines_and_indent():
    text = "R 0\n    M(0.002) 0\nDETECTOR rec[-1]\nMPP(0.0005) Z0\nMXX(0.001) 0 1\n"
    out = xtim.scale_noise(text, 1e-4, p0=1e-3)
    assert out.splitlines() == ["R 0", "    M(0.0002) 0", "DETECTOR rec[-1]", "MPP(5e-05) Z0", "MXX(0.0001) 0 1"]


def test_scaled_readout_noise_reaches_the_sampler():
    """The rewritten M(p) is what the engine samples: the flip rate of a noiseless-state readout follows p."""
    text = "R 0\nM(0.001) 0\nDETECTOR rec[-1]\n"
    s = xtim.compile_twirl_sampler(xtim.scale_noise(text, 0.2), selfcheck=0)
    d = np.unpackbits(np.asarray(s.sample(20000, seed=1)[0]), axis=1, bitorder="little")[:, 0]
    assert abs(d.mean() - 0.2) < 0.015


@pytest.mark.parametrize("line,want", [
    ("M[t](0.001) 0", "M[t](0.01) 0"),
    ("MPP[t](0.001) X0*Z1", "MPP[t](0.01) X0*Z1"),
    ("X_ERROR[t](0.001) 0", "X_ERROR[t](0.01) 0"),
    ("MR[a b](0.002) 0 1", "MR[a b](0.02) 0 1"),
    ("PAULI_CHANNEL_1[t](0.001, 0, 0.002) 0", "PAULI_CHANNEL_1[t](0.01, 0, 0.02) 0"),
])
def test_tagged_instructions_are_rescaled(line, want):
    """A Stim tag between the name and the arguments (review M1): the argument is still scaled, the tag kept."""
    assert xtim.scale_noise("R 0 1\n" + line + "\n", 0.01).splitlines()[1] == want


def test_tagged_chain_is_rescaled_like_untagged():
    a = xtim.scale_noise("E(0.001) X0\nELSE_CORRELATED_ERROR(0.002) Z0\n", 0.01)
    b = xtim.scale_noise("E[t](0.001) X0\nELSE_CORRELATED_ERROR[u](0.002) Z0\n", 0.01)
    assert b == a.replace("E(", "E[t](", 1).replace("ELSE_CORRELATED_ERROR(", "ELSE_CORRELATED_ERROR[u](")


def test_tagged_measurement_reaches_the_sampler():
    s = xtim.compile_twirl_sampler(xtim.scale_noise("R 0\nM[tag](0.001) 0\nDETECTOR rec[-1]\n", 0.2), selfcheck=0)
    d = np.unpackbits(np.asarray(s.sample(20000, seed=1)[0]), axis=1, bitorder="little")[:, 0]
    assert abs(d.mean() - 0.2) < 0.015
