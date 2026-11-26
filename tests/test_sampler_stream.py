"""compile_detector_sampler: repeated sample() calls continue the stream (stim convention); the first call is unchanged."""
import numpy as np
import pytest

import xtim

BASE_SEED5_FIRST16 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
C = "X_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\n"


def _s(seed=None):
    c = xtim.Circuit(C)
    return c.compile_detector_sampler() if seed is None else c.compile_detector_sampler(seed=seed)


@pytest.mark.parametrize("seed", [None, 5])
def test_two_calls_differ(seed):
    s = _s(seed)
    assert not np.array_equal(s.sample(256), s.sample(256))


def test_first_call_unchanged_and_same_seed_same_sequence():
    a, b = _s(5), _s(5)
    a1, a2 = a.sample(64), a.sample(64)
    b1, b2 = b.sample(64), b.sample(64)
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2)
    # call 1 is the BASE output (captured from abc71bf: seed=5, 64 shots, first 16)
    assert a1[:16, 0].astype(int).tolist() == BASE_SEED5_FIRST16
    assert a.seed == 5


def test_threads_invariant_second_call():
    a, b = _s(9), _s(9)
    a.sample(5000, threads=1)
    b.sample(5000, threads=4)
    assert np.array_equal(a.sample(5000, threads=1), b.sample(5000, threads=4))


def test_statistics_50_calls():
    s = _s(3)
    batches = [s.sample(1000)[:, 0] for _ in range(50)]
    m = np.mean(batches)
    assert abs(m - 0.5) < 4 * 0.5 / np.sqrt(50 * 1000)
    keys = {np.packbits(b).tobytes() for b in batches}
    assert len(keys) == 50


def test_unseeded_samplers_differ_and_seed_roundtrips():
    a, b = _s(), _s()
    assert a.seed != b.seed
    a1 = a.sample(256)
    assert not np.array_equal(a1, b.sample(256))
    assert np.array_equal(_s(a.seed).sample(256), a1)


def test_seed_zero_unchanged():
    # captured with the pre-change default (seed=0) at 0ccb301
    assert _s(0).seed == 0
    assert _s(0).sample(16)[:, 0].astype(int).tolist() == [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]
