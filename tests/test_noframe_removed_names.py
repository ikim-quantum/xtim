"""dem-noframe (owner 2026-11-03): removed names fail loudly. `_xtim.carried_symplectic_action` is gone (use the
sampler's input_pauli and branches() frames); dem_column_layout has no 'frame' key."""
import pytest

import xtim._xtim as _x


def test_carried_symplectic_action_removed():
    assert not hasattr(_x, "carried_symplectic_action")
    with pytest.raises(AttributeError, match="carried_symplectic_action"):
        _x.carried_symplectic_action("INPUT_QUBITS in 0\nH 0\nOUTPUT_QUBITS out 0\n")


def test_dem_column_layout_has_no_frame_key():
    lay = _x.dem_column_layout("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nR 1\nOUTPUT_QUBITS out 1\n")
    assert lay["ok"], lay
    with pytest.raises(KeyError):
        lay["frame"]
