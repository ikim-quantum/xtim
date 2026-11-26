"""perf-glue: per-call overhead fixes of the twirl sampler's public calls (byte-neutral by contract).

- the postselection state (early-reject) is built once per mask and reused while the mask is unchanged;
- a BarrierBuffer shares the sampler's append-only content table (a snapshot of (store, count)) instead of copying it;
- sample() writes its packed rows straight into the returned numpy arrays (fresh, owning, C-contiguous);
- a seeded call at the seed of the last reseed restores the reseeded noise state instead of re-deriving it.
Every output stays byte-identical: repeated / alternating masked calls equal an unmasked stream filtered afterwards,
same-seed calls repeat their bytes, and old buffers keep their records while the table grows.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from xtim.twirl import TwirlBatch

DATA = os.path.join(os.path.dirname(__file__), "data")


def _text(name):
    with open(os.path.join(DATA, name)) as f:
        return f.read()


def _unpack(a, n):
    return np.unpackbits(np.asarray(a), axis=1, bitorder="little")[:, :n].astype(bool)


def _cols(out):
    """(dets, obs[, decisions]) of a sample() result; a masked result's counts dict is dropped."""
    if isinstance(out, tuple) and out and isinstance(out[-1], dict):
        out = out[:-1]
        if len(out) == 1 and isinstance(out[0], TwirlBatch):
            out = out[0]
    if isinstance(out, TwirlBatch):
        return (out.dets, out.obs, out.decisions)
    return tuple(np.asarray(a) for a in out)


# ── §4.2: the postselection state is memoized by its mask ─────────────────────────────────────────────────────────
def test_postselect_state_built_once_per_mask():
    s = TA.compile_family_sampler(_text("cultivation_d3_faithful.stim"), selfcheck=0)
    nd = s.num_detectors
    every, half = [1] * nd, [1, 0] * (nd // 2) + [1] * (nd % 2)
    assert s._postselect_builds() == 0
    assert s._set_postselect(every) is True
    assert s._postselect_builds() == 1
    for _ in range(5):                                # unchanged mask: no rebuild
        assert s._set_postselect(every) is True
    assert s._postselect_builds() == 1
    assert s._set_postselect([0] * nd) is False       # an empty mask arms nothing and builds nothing
    assert s._postselect_builds() == 1
    assert s._set_postselect(every) is True
    assert s._postselect_builds() == 1
    assert s._set_postselect(half) is True            # a new mask builds
    assert s._postselect_builds() == 2
    assert s._set_postselect(every) is True           # one memo entry: the last built mask
    assert s._postselect_builds() == 3


@pytest.mark.parametrize("name", ["cultivation_d3_faithful.stim", "ja_seed23.stim", "distillation_15_1_3.stim"])
def test_repeated_masked_calls_byte_identical(name):
    # one sampler, masked calls with repeated / alternating masks and seeds, against an unmasked twin filtered
    text = _text(name)
    a = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = a.num_detectors
    every = np.ones(nd, dtype=bool)
    half = np.zeros(nd, dtype=bool)
    half[::2] = True
    plan = [(every, 5), (every, None), (half, None), (every, 9), (None, None), (half, None), (half, 9), (every, None)]
    for mask, seed in plan:
        full = _cols(a.sample(400, seed=seed))
        if mask is None:
            out = _cols(b.sample(400, seed=seed))
            keep = np.ones(400, dtype=bool)
        else:
            res = b.sample(400, seed=seed, postselect=mask)
            out = _cols(res)
            keep = ~_unpack(full[0], nd)[:, mask].any(1)
            assert res[-1]["passed"] == int(keep.sum()) and res[-1]["attempts"] == 400
        assert len(full) == len(out)
        for x, y in zip(full, out):
            assert np.asarray(x)[keep].tobytes() == np.asarray(y).tobytes()


def test_repeated_masked_barrier_calls_byte_identical():
    text = _text("ja_seed23.stim")
    a = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = a.num_detectors
    every = np.ones(nd, dtype=bool)
    for seed in (3, None, None, 4):
        full = a.sample_barrier(500, seed=seed)
        buf = b.sample_barrier(500, seed=seed, postselect=every)
        keep = np.flatnonzero(~_unpack(full.dets(), nd).any(1))
        kf = full.record_keys()
        assert [kf[i] for i in keep] == buf.record_keys()
        assert np.asarray(full.dets())[keep].tobytes() == np.asarray(buf.dets()).tobytes()


# ── §4.1: a BarrierBuffer shares the content table ───────────────────────────────────────────────────────────────
def test_barrier_buffers_share_the_content_table():
    s = TA.compile_family_sampler(_text("cultivation_d3_faithful.stim"), selfcheck=0)
    first = s.sample_barrier(50, seed=1)
    keys_first = first.record_keys()
    store_first, n_first = TA._contents_view(first)
    assert n_first > 0
    big = s.sample_barrier(4000)                      # the table grows (past the first store's capacity)
    store_big, n_big = TA._contents_view(big)
    assert n_big > n_first
    views = [TA._contents_view(s.sample_barrier(1)) for _ in range(40)]
    # no copy per call: buffers taken while the table holds the same entries share one store
    for (sa, na), (sb, nb) in zip(views, views[1:]):
        if na == nb:
            assert sa == sb
    assert len({v[0] for v in views}) <= 2
    # an old buffer keeps its own (older) store and reads its records unchanged
    assert TA._contents_view(first) == (store_first, n_first)
    assert first.record_keys() == keys_first


def test_old_buffer_records_survive_table_growth_public():
    text = _text("cultivation_d3_faithful.stim")
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    first = s.sample_barrier(50, seed=1)
    keys = first.record_keys()
    plans = [first.plan_content(i) for i in range(first.num_shots)]
    s.sample_barrier(4000)
    assert first.record_keys() == keys
    assert [first.plan_content(i) for i in range(first.num_shots)] == plans
    t = xtim.compile_twirl_sampler(text, selfcheck=0)   # bytes equal a fresh sampler's
    assert t.sample_barrier(50, seed=1).record_keys() == keys


# ── §4.3: sample() writes into the returned arrays ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["cultivation_d3_faithful.stim", "ja_seed23.stim"])
def test_sample_arrays_are_fresh_owning_arrays(name):
    s = xtim.compile_twirl_sampler(_text(name), selfcheck=0)
    nd = s.num_detectors
    every = np.ones(nd, dtype=bool)
    outs = [(s.sample(64, seed=2), 64), (s.sample(0), 0), (s.sample(1), 1)]
    r = s.sample(300, seed=4, postselect=every)
    outs.append((r[:-1], r[-1]["passed"]))
    r0 = s.sample(0, postselect=every)
    outs.append((r0[:-1], 0))
    for out, rows in outs:
        for a in _cols(out):
            a = np.asarray(a)
            assert a.dtype == np.uint8 and a.ndim == 2 and a.shape[0] == rows
            assert a.flags.c_contiguous and a.flags.owndata and a.flags.writeable and a.base is None
    assert _cols(outs[0][0])[0].shape[1] == (nd + 7) // 8


# ── §4.4: same-seed calls restore the reseeded state ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["cultivation_d3_faithful.stim", "ja_seed23.stim", "distillation_15_1_3.stim"])
def test_same_seed_calls_repeat_bytes(name):
    text = _text(name)
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    f = xtim.compile_twirl_sampler(text, selfcheck=0)
    ref = [_cols(f.sample(200, seed=7)), _cols(f.sample(150))]
    ref8 = _cols(xtim.compile_twirl_sampler(text, selfcheck=0).sample(30, seed=8))
    for seq in range(3):
        got = [_cols(s.sample(200, seed=7)), _cols(s.sample(150))]   # seed=None continues the restored stream
        for g, r_ in zip(got, ref):
            for x, y in zip(g, r_):
                assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
        g8 = _cols(s.sample(30, seed=8))
        for x, y in zip(g8, ref8):
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
    for _ in range(3):                                # back-to-back same seed (the snapshot path)
        g = _cols(s.sample(200, seed=7))
        for x, y in zip(g, ref[0]):
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


# ── §4.2 (fix round 1, F2): the per-call counts of a re-armed memo equal a rebuilt state's ───────────────────────
def _arm_term_view(buf):
    # materialize() of a kind-2 / carried row arms the sampler's term view (once): later masks get term rows
    for i in range(min(40, buf.num_shots)):
        buf.materialize(i)


@pytest.mark.parametrize("name", ["ja_seed23.stim", "cultivation_d3_faithful.stim", "distillation_15_1_3.stim"])
def test_masked_counts_match_a_rebuilding_twin(name):
    # `s` re-arms its memo; `twin` makes a 0-shot call with another mask before every masked call, so each of its masked
    # calls rebuilds the state (the pre-memo behaviour). Counts (early included) and bytes must agree call by call,
    # across the term view being armed between memo uses.
    text = _text(name)
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    twin = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    every = np.ones(nd, dtype=bool)
    half = np.zeros(nd, dtype=bool)
    half[::2] = True
    other = np.zeros(nd, dtype=bool)
    other[-1] = True
    # a recording run arms an unarmed term view while the memo (built unarmed) is set: the next same-mask call re-arms
    # the memo and must then build its term rows (the barrier's term-routed shots read them)
    plan = [("s", 400, 5, every), ("s", 400, None, every), ("b", 300, 6, every), ("b", 300, None, every),
            ("arm", 0, None, None),
            ("s", 400, 7, every), ("s", 400, None, half), ("s", 400, 8, every), ("b", 300, None, every),
            ("s", 400, None, every)]
    early_total = 0
    for kind, n, seed, mask in plan:
        if kind == "arm":
            _arm_term_view(last[0])
            _arm_term_view(last[1])
            continue
        twin.sample(0, postselect=other)
        if kind == "s":
            a, b = s.sample(n, seed=seed, postselect=mask), twin.sample(n, seed=seed, postselect=mask)
            assert a[-1] == b[-1]
            for x, y in zip(_cols(a), _cols(b)):
                assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
            early_total += a[-1]["early"]
        else:
            a, b = s.sample_barrier(n, seed=seed, postselect=mask), twin.sample_barrier(n, seed=seed, postselect=mask)
            assert a.postselection == b.postselection
            assert a.record_keys() == b.record_keys()
            early_total += a.postselection["early"]
            last = (a, b)
    assert early_total > 0


# ── §4.3 (fix round 1, F1): zero-size postselected arrays are built as fresh arrays (strides as before) ──────────
def test_zero_size_postselected_arrays_match_fresh_arrays():
    text = _text("cultivation_d3_faithful.stim")
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    every = np.ones(nd, dtype=bool)
    ref = [np.asarray(a) for a in _cols(s.sample(0))]   # pybind's fresh zero-row arrays
    found_empty = False
    for seed in range(1, 60):
        r = s.sample(1, seed=seed, postselect=every)
        if r[-1]["passed"] == 0:
            found_empty = True
            for a, f in zip(_cols(r), ref):
                a = np.asarray(a)
                assert a.shape == f.shape and a.strides == f.strides
    assert found_empty
    # zero-width columns (no observables / decisions) of a trimmed run
    buf = s.sample_barrier(100, seed=3, postselect=every)
    d = np.asarray(buf.decisions())
    if d.shape[1] == 0:
        assert d.strides == np.asarray(s.sample_barrier(5, seed=3).decisions()).strides
