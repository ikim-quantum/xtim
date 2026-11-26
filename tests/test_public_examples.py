"""The bundled public examples meet the strict output contract.

Every example declares OUTPUT_QUBITS with every frame closed (no PAULI_EXPECTATION): the noiseless RecordFidelity
is exactly 1 on every shot, and a noisy run samples and scores. The repo copy (examples/) and the packaged copy
(xtim/_examples/) are the same file.
"""
import pathlib
import warnings

import numpy as np
import pytest

import xtim

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = xtim.list_examples()


def _sampler(text, p_factor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(text, p_factor=p_factor, selfcheck=0)


def test_every_example_is_listed():
    assert sorted(p.stem for p in (ROOT / "examples").glob("*.stim")) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_public_example_closed_and_scored(name):
    text = xtim.load_example(name).text
    assert (ROOT / "examples" / f"{name}.stim").read_text() == pathlib.Path(xtim.example_path(name)).read_text()
    assert "PAULI_EXPECTATION" not in text and "OUTPUT_QUBITS" in text
    s0 = _sampler(text, 0.0)                                   # noiseless: F = 1 on every shot
    F0 = np.asarray(xtim.RecordFidelity(s0).evaluate(s0.sample_barrier(300, seed=3)))
    assert F0.shape == (300,) and np.all(np.abs(F0 - 1.0) < 1e-12), (F0.min(), F0.max())
    s = _sampler(xtim.scale_noise(text, 1e-2) if "cube_ccz" not in name else text, 1.0)   # a noisy run samples
    buf = s.sample_barrier(300, seed=5)
    F = np.asarray(xtim.RecordFidelity(s).evaluate(buf))
    assert F.shape == (300,) and np.all((F >= -1e-12) & (F <= 1 + 1e-12))
    if name not in ("cube_ccz",):                              # cube_ccz carries no noise (its script adds it)
        assert F.min() < 1.0
