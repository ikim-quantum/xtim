"""EXP-fix #8 (E3): frames.projected_expectation expands its sector projector into 2^m Pauli products by
construction; it refuses by name above 16 sector operators (no override). MP-8 (remove the check) turns
test_projected_expectation_refuses_17 red."""
import numpy as np
import pytest

import xtim
from xtim import frames


def _export():
    return xtim.bare_state_of("R 0 1\nH 0\nT 0\n").branch_frames()


def _sector(n, q):
    gx = np.zeros(n, np.uint8)
    gz = np.zeros(n, np.uint8)
    gz[q] = 1
    return ((gx, gz, 0), +1)


def test_projected_expectation_refuses_17():
    fr = _export()
    n = int(fr["n"])
    px = np.zeros(n, np.uint8); pz = np.zeros(n, np.uint8); pz[1] = 1
    with pytest.raises(ValueError, match=r"17 sector operators expand to 2\^m = 131072 Pauli products \(cap 16\)"):
        frames.projected_expectation(fr, px, pz, 0, [_sector(n, 1)] * 17)


def test_projected_expectation_serves_16():
    fr = _export()
    n = int(fr["n"])
    px = np.zeros(n, np.uint8); pz = np.zeros(n, np.uint8); pz[1] = 1
    val, den = frames.projected_expectation(fr, px, pz, 0, [_sector(n, 1)] * 16)
    assert abs(val - 1.0) < 1e-12 and abs(den - 1.0) < 1e-12
