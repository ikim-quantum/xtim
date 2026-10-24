"""seed=None CONTINUES the stream: the per-shot coin generator is indexed by a
CUMULATIVE shot counter, not by the shot's index within one call.

Defect (adversarial review C4, 2026-10-14): ``shot_rng.seed(seed_, done)`` used the
per-call ``done`` (reset to 0 at the top of every ``run()``), so with seed=None two
consecutive calls replayed the same twirl/collapse coin sequence shot-for-shot — the
coin-driven column of call 2 equalled call 1 on 100% of shots (independent: ~50%),
while the noise stream (sequential) did continue. Pooled calls were correlated.

Contract tested here:
  * LAW: two consecutive seed=None calls give independent coin columns (agreement
    ~1/2, 6 sigma window) on sample() and sample_barrier().
  * STREAM: sample(N, seed=s) followed by sample(M) is byte-identical to a single
    sample(N+M, seed=s) (and likewise for sample_barrier) on every engine stream, for
    ANY split (S0(b): every per-shot stream is keyed on the cumulative shot index and
    the noise stream is keyed per channel, so the 4096-shot batch partition no longer
    reaches the bytes; tests/test_keyed_streams.py pins random multi-way splits).
  * Explicit seeds are unaffected (byte identity of every seeded call regardless of
    history is pinned by tests/test_set_seed_equivalence.py and the byte corpus).
"""

from pathlib import Path

import numpy as np
import pytest

import xtim

_DATA = Path(__file__).resolve().parent / "data"
_EXAMPLES = Path(__file__).resolve().parent / "data"

# A coin-driven detector: T X T^dag = (X - Y)/sqrt2 ... the MX readout of the
# twirled residual is a fair per-shot coin; detector 1 is plain readout noise.
COIN = ("RX 0\nT 0\nX_ERROR(1.0) 0\nT_DAG 0\nMX 0\nDETECTOR rec[-1]\n"
        "R 1\nX_ERROR(0.3) 1\nM 1\nDETECTOR rec[-1]\n")


def _bits(a, k):
    return np.unpackbits(np.asarray(a, np.uint8), axis=1, bitorder="little")[:, :k]


@pytest.mark.parametrize("surface", ["sample", "sample_barrier"])
def test_unseeded_calls_have_independent_coins(surface):
    s = xtim.compile_twirl_sampler(COIN, selfcheck=0)
    n = 20000

    def call(seed):
        if surface == "sample":
            return _bits(s.sample(n, seed=seed)[0], 2)
        return _bits(s.sample_barrier(n, seed=seed).dets(), 2)

    d1, d2 = call(7), call(None)
    agree = float((d1[:, 0] == d2[:, 0]).mean())
    # independent fair coins: agreement ~ Bin(n, 1/2)/n, sigma = 0.5/sqrt(n)
    assert abs(agree - 0.5) < 6 * 0.5 / np.sqrt(n), (
        f"{surface}: coin column agrees on {agree:.4f} of shots across two seed=None "
        "calls — the per-shot coin stream restarted instead of continuing")
    assert abs(d2[:, 0].mean() - 0.5) < 6 * 0.5 / np.sqrt(n)


# workloads covering: shot coins + noise (all), rf readout-flip stream (cultivation
# d5 M(p)), sequential Born-decision stream (born5_dec), barrier retention.
WORKLOADS = {
    "coin": (COIN, dict(selfcheck=0)),
    "born5_dec": ((_DATA / "adaptq_born5_dec_enum_producer.stim").read_text(),
                  dict(selfcheck=0)),
    "qrm_magic": ((_DATA / "adaptq_qrm_magic_producer.stim").read_text(),
                  dict(selfcheck=0)),
    "cultivation_d5": ((_EXAMPLES / "cultivation_d5.stim").read_text(),
                       dict(selfcheck=0)),
}


def _cat(parts):
    return np.concatenate([np.asarray(p) for p in parts], axis=0)


@pytest.mark.parametrize("name", list(WORKLOADS))
@pytest.mark.parametrize("split", [(4096, 5), (4096, 700), (8192, 300), (1, 4999), (777, 3333)])
def test_sample_split_equals_single_call(name, split):
    text, kw = WORKLOADS[name]
    n1, n2 = split
    one = xtim.compile_twirl_sampler(text, **kw).sample(n1 + n2, seed=11)
    s = xtim.compile_twirl_sampler(text, **kw)
    a, b = s.sample(n1, seed=11), s.sample(n2)
    for i, (w, g1, g2) in enumerate(zip(one, a, b)):
        if w is None:
            assert g1 is None and g2 is None
            continue
        assert np.array_equal(np.asarray(w), _cat([g1, g2])), (
            f"{name} {split}: sample field {i}: split call != single call")


def test_coin_only_split_anywhere():
    """No noise stream: any split continues byte-for-byte."""
    one = xtim.compile_twirl_sampler(COIN.replace("X_ERROR(0.3) 1\n", ""),
                                     selfcheck=0).sample(800, seed=3)
    s = xtim.compile_twirl_sampler(COIN.replace("X_ERROR(0.3) 1\n", ""), selfcheck=0)
    a, b = s.sample(300, seed=3), s.sample(500)
    assert np.array_equal(np.asarray(one[0]), _cat([a[0], b[0]]))


@pytest.mark.parametrize("name", list(WORKLOADS))
def test_barrier_split_equals_single_call(name):
    text, kw = WORKLOADS[name]
    n1, n2 = 1999, 300                 # any split (S0(b)); formerly a 4096 multiple
    one = xtim.compile_twirl_sampler(text, **kw).sample_barrier(n1 + n2, seed=13)
    s = xtim.compile_twirl_sampler(text, **kw)
    b1 = s.sample_barrier(n1, seed=13)
    b2 = s.sample_barrier(n2)
    for acc in ("dets", "obs", "decisions"):
        w = np.asarray(getattr(one, acc)())
        assert np.array_equal(w, _cat([getattr(b1, acc)(), getattr(b2, acc)()])), (
            f"{name}: barrier {acc}: split call != single call")
    # J-b1 F: the layout-3 record keys (sigma retired) continue too
    assert [bytes(k) for k in one.record_keys()] == ([bytes(k) for k in b1.record_keys()] +
                                                     [bytes(k) for k in b2.record_keys()]), name
    coins_one = [bytes(one.outcomes(i)) for i in range(n1 + n2)]
    coins_split = ([bytes(b1.outcomes(i)) for i in range(n1)]
                   + [bytes(b2.outcomes(i)) for i in range(n2)])
    assert coins_one == coins_split, f"{name}: barrier coins: split call != single call"


def test_explicit_seed_restarts_stream():
    """An explicit seed after unseeded continuation reproduces the fresh stream."""
    s = xtim.compile_twirl_sampler(COIN, selfcheck=0)
    ref = [np.asarray(x).copy() for x in s.sample(500, seed=5)]
    s.sample(777)                      # advance the cumulative counter
    again = s.sample(500, seed=5)
    for r, g in zip(ref, again):
        assert np.array_equal(r, np.asarray(g))
