"""CAPS-2: shot counts per sample() call (CAPS audit item A7).

A7  TwirlSampler.sample refused shots > INT32_MAX by name ("b8 array shape is int"); the arrays and the engine
    already take a 64-bit count, so the refusal was a stale width cap. The call now reaches the engine with the full
    count: the engine's own named refusal (the cumulative 2^44 shot index of a seed) is what a 2^44 + 1 call meets.

A8 (the record blob above 4 GiB) is tested in cpp/tests/test_plan_arena.cpp (test_blob_over_4gib); A4 (the keyed
stream limits) is a documented, unreachable named refusal (cpp/include/qeccore/keyed_rng.hpp).
"""
import os

import numpy as np
import pytest

from _public import public_sampler

NO_COLUMNS = "R 0\nX_ERROR(0.25) 0\nM 0\n"                          # zero-width arrays: no memory at any count
ONE_DET = "R 0\nX_ERROR(0.25) 0\nM 0\nDETECTOR rec[-1]\n"


def test_shots_beyond_int32_reach_the_engine():
    s = public_sampler(NO_COLUMNS)
    assert s.num_detectors == 0 and s.num_observables == 0
    with pytest.raises(ValueError, match=r"would reach 2\^44"):
        s.sample(2**44 + 1, 1)
    d, o = s.sample(5, 1)                                            # the refused call committed nothing
    assert d.shape == (5, 0) and o.shape == (5, 0)


@pytest.mark.skipif(os.environ.get("XTIM_CAPS2_BIG") != "1",
                    reason="2^31 + 4096 shots (~2 GiB output, ~10 s at 16 threads): set XTIM_CAPS2_BIG=1")
def test_sample_more_than_int32_shots():
    n = 2**31 + 4096
    s = public_sampler(ONE_DET)
    d, o = s.sample(n, 7, threads=16)
    assert d.shape == (n, 1) and o.shape == (n, 0)
    head, _ = public_sampler(ONE_DET).sample(4096, 7)
    assert np.array_equal(d[:4096], head)                            # the same keyed shots
    tail = d[2**31:, 0]                                              # rows past INT32_MAX: drawn, not left zero
    assert abs(tail.mean() - 0.25) < 5 * np.sqrt(0.25 * 0.75 / tail.size)
    big = d[-(1 << 24):, 0]
    assert abs(big.mean() - 0.25) < 5 * np.sqrt(0.25 * 0.75 / big.size)
