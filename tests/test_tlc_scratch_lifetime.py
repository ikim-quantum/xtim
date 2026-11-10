"""TLC: the exact engine's kernel scratch lives for one compile / run and is released in between.
Its history must never reach an output: a program's runs are byte-identical whether or not other
programs (other shapes, other chi) compiled and ran on the same thread in between, and whether the
program runs once per sampler or twice in a row."""
from pathlib import Path

import numpy as np

import xtim
from _exact_engine import exact_detector_sampler  # noqa: E402  (exact-out T5: the exact engine is test API)

ROOT = Path(__file__).resolve().parents[1]
CULT = (ROOT / "tests" / "data" / "cultivation_d3_faithful.stim").read_text()
OTHER = (ROOT / "tests" / "data" / "code_switching_faithful.stim").read_text()


def _runs(smp):
    return [np.asarray(smp.sample(300)) for _ in range(2)]


def test_runs_independent_of_interleaved_programs():
    a = _runs(exact_detector_sampler(xtim.Circuit(CULT), seed=3))
    b_smp = exact_detector_sampler(xtim.Circuit(CULT), seed=3)
    b0 = np.asarray(b_smp.sample(300))
    other = exact_detector_sampler(xtim.Circuit(OTHER), seed=9)
    other.sample(200)                                  # a different program grows/releases its scratch
    b1 = np.asarray(b_smp.sample(300))
    assert np.array_equal(a[0], b0) and np.array_equal(a[1], b1)
