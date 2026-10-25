"""TLC: the exact engine's kernel scratch lives for one compile / run and is released in between.
Its history must never reach an output: a program's runs are byte-identical whether or not other
programs (other shapes, other chi) compiled and ran on the same thread in between, and whether the
program runs once per sampler or twice in a row."""
from pathlib import Path

import numpy as np

import xtim

ROOT = Path(__file__).resolve().parents[1]
CULT = (ROOT / "tests" / "data" / "cultivation_d3_faithful.stim").read_text()
OTHER = (ROOT / "tests" / "data" / "code_switching_faithful.stim").read_text()


def _runs(smp):
    return [np.asarray(smp.sample(300)) for _ in range(2)]


def test_runs_independent_of_interleaved_programs():
    a = _runs(xtim.Circuit(CULT).compile_detector_sampler(seed=3, engine="exact"))
    b_smp = xtim.Circuit(CULT).compile_detector_sampler(seed=3, engine="exact")
    b0 = np.asarray(b_smp.sample(300))
    other = xtim.Circuit(OTHER).compile_detector_sampler(seed=9, engine="exact")
    other.sample(200)                                  # a different program grows/releases its scratch
    b1 = np.asarray(b_smp.sample(300))
    assert np.array_equal(a[0], b0) and np.array_equal(a[1], b1)
