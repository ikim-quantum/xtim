"""v2.7 selfcheck once-per-sampler contract (owner option c).

The selfcheck oracle window (``selfcheck=N``, default 2000: the first N fired
shots also run the full sigma path and must match EXACTLY) runs ONCE per sampler
lifetime — on the first non-zero-shot ``sample()``/``sample_barrier()`` call
after compilation. It never re-arms on later calls: the in-place ``set_seed``
zeroes the ``checked``/``pprchecked`` gating counters, which historically
re-armed the window on EVERY seeded call, silently re-paying the slow oracle
path per call.

The window is a STREAM-NEUTRAL verification overlay: sampled bytes are
byte-identical with the window on or off (ratified when selfcheck=0 became the
fast-path default). Three contracts pinned here:

1. Non-vacuity — the window still RUNS on the first call. Observable:
   ``channel_one_counts()`` accumulates only while the window's channel
   accumulation is active (all-zero on the ratified fast path), and the
   one-line stderr diagnostic announces the active window.
2. Once-per-lifetime — a second seeded call pays NO window: its
   ``channel_one_counts()`` are all-zero (counter observable), the diagnostic
   line does not print again, and the call is not slower than the windowed
   first call (lenient timing bound; the counters are the real gate).
3. Byte equality — same seeds, ``selfcheck=2000`` (window on first call, spent
   afterwards) vs ``selfcheck=0``: every sampled stream is byte-identical,
   including calls AFTER the window is spent.
"""

import time
from pathlib import Path

import numpy as np

import xtim

_DATA = Path(__file__).resolve().parent / "data"
_EXAMPLES = Path(xtim.__file__).resolve().parent / "_examples"
# Counter workload: cultivation d5 has 107 deterministic detector channels, so
# the window's channel accumulation is observable through channel_one_counts().
# (The steane producer compiles to 0 sigma channels — counts are vacuous there.)
_CULT = (_EXAMPLES / "cultivation_d5.stim").read_text()
_CULT_KW = dict(p_factor=1.0, skip_refused_observables=True)
_STEANE = (_DATA / "adaptq_steane_h_producer.stim").read_text()
_SEEDS = (2, 7, 123)
_SHOTS = 400


def _snap(s, shots, seed):
    out = s.sample(shots, seed=seed)
    return tuple(None if a is None else np.asarray(a).copy() for a in out)


def _counts_sum(s):
    return sum(int(c) for c in s.channel_one_counts())


def test_window_runs_first_call_then_never_rearms(capfd):
    """Non-vacuity + once-per-lifetime, via the channel-count observable."""
    s = xtim.compile_twirl_sampler(_CULT, selfcheck=2000, **_CULT_KW)
    capfd.readouterr()                       # drop compile-time chatter

    t0 = time.perf_counter()
    s.sample(_SHOTS, seed=_SEEDS[0])
    t_first = time.perf_counter() - t0
    err_first = capfd.readouterr().err
    # (1) the window ran: channel accumulation is active ONLY with an armed
    # window (the ratified selfcheck=0 fast path leaves the counts all-zero),
    # and this noisy Steane workload fires channels.
    assert _counts_sum(s) > 0, (
        "first call produced all-zero channel_one_counts — the selfcheck "
        "window did not run on the first call (vacuous once-semantics)")
    # The stderr diagnostic prints at most once (process-static dedup).
    assert err_first.count("QEC_TW_SELFCHECK") <= 1

    # (2) second seeded call: set_seed zeroes the counters; a re-armed window
    # would re-accumulate them. Spent window => counts stay all-zero.
    t0 = time.perf_counter()
    s.sample(_SHOTS, seed=_SEEDS[1])
    t_second = time.perf_counter() - t0
    err_second = capfd.readouterr().err
    assert _counts_sum(s) == 0, (
        "second seeded call re-accumulated channel_one_counts — the selfcheck "
        "window re-armed on set_seed (the once-per-sampler latch is broken)")
    assert "QEC_TW_SELFCHECK" not in err_second, (
        "the selfcheck diagnostic printed again on the second call")
    # Lenient timing corroboration (the counters above are the hard gate): the
    # window-free call must not be slower than the windowed one beyond noise.
    assert t_second < max(2.0 * t_first, t_first + 0.05), (
        f"second (window-free) call took {t_second * 1e3:.1f} ms vs "
        f"{t_first * 1e3:.1f} ms windowed — no fast path?")

    # (3) unseeded continuation also stays window-free.
    s.sample(_SHOTS)
    assert _counts_sum(s) == 0
    assert "QEC_TW_SELFCHECK" not in capfd.readouterr().err


def test_zero_shot_call_does_not_spend_window():
    """A zero-shot call checks nothing, so it must not burn the window."""
    s = xtim.compile_twirl_sampler(_CULT, selfcheck=2000, **_CULT_KW)
    s.sample(0, seed=_SEEDS[0])              # zero-shot: window NOT spent
    s.sample(_SHOTS, seed=_SEEDS[0])         # first real run: window active
    assert _counts_sum(s) > 0, (
        "zero-shot call spent the selfcheck window — the first real run "
        "sampled unverified")


def test_selfcheck_window_is_stream_neutral():
    """Byte equality: selfcheck=2000 (once) vs selfcheck=0, same seeds, every
    stream equal — on the first (windowed) call AND on spent-window calls."""
    for text, kw in ((_CULT, _CULT_KW), (_STEANE, {})):
        a = xtim.compile_twirl_sampler(text, selfcheck=2000, **kw)
        b = xtim.compile_twirl_sampler(text, selfcheck=0, **kw)
        # Two passes over the seed set: pass 0 includes a's windowed first
        # call; pass 1 is entirely spent-window vs selfcheck=0.
        for rep in range(2):
            for seed in _SEEDS:
                ga, gb = _snap(a, _SHOTS, seed), _snap(b, _SHOTS, seed)
                assert len(ga) == len(gb)
                for i, (x, y) in enumerate(zip(ga, gb)):
                    if x is None or y is None:
                        assert x is None and y is None
                        continue
                    assert np.array_equal(x, y), (
                        f"rep {rep} seed {seed}: stream {i} differs in "
                        f"{int((x != y).sum())} byte(s) — the selfcheck "
                        f"window is NOT stream-neutral")
