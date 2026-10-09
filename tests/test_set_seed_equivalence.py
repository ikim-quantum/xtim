"""speed-kill T1 oracle: in-place set_seed == fresh sampler, byte-for-byte.

Every explicitly-seeded ``sample()`` / ``sample_barrier()`` call historically tore
down and reconstructed the whole ``TwirlRecordSampler`` (``rebuild(seed)``: a
~180 ms circuit-compile-scale fixed cost on cultivation d5 — the root cause of the
"90 µs/shot floor", docs adaptq/err-record-cache-rootcause.md). The engine now
reseeds IN PLACE (``TwirlRecordSampler::set_seed``), resetting exactly the
seed-dependent state a fresh construction initializes and reconstructing nothing.

The contract is BYTE IDENTITY, zero tolerance: on a sampler with ARBITRARY prior
history (other seeds, unseeded continuations, mixed sample/sample_barrier calls,
zero-shot calls), a seeded call must emit streams byte-identical to the same call
on a freshly compiled sampler — dets, obs, decisions, sigmas, coins, plan keys,
record keys, record group ids.

Two reference classes:
  * FRESH reference — a newly compiled sampler whose FIRST seeded call is at the
    probe seed (fresh caches, zero counters).
  * CONSTRUCTION-NATIVE reference — a newly compiled sampler's first call with
    seed=None runs on the CONSTRUCTOR's own seed-1 state and never passes through
    set_seed at all; a warm sampler's set_seed(1) stream must match it exactly.
    This pins set_seed to construction bytes, not merely to itself.

Workloads: small Steane stage (selfcheck window ON — exercises the oracle-window
counter reset), the qrm magic producer, the born5 dec-enum producer (sequential
dec_gen Born-decision stream — discriminates a missing dec_gen reset), and
cultivation d5 (the perf-critical adaptq workload). Plus a timing assertion: a
seeded zero-shot call on a warm cultivation-d5 sampler is < 5 ms (was ~180 ms).
"""

import time
from pathlib import Path

import numpy as np
import pytest

import xtim

_DATA = Path(__file__).resolve().parent / "data"
_EXAMPLES = Path(xtim.__file__).resolve().parent / "_examples"

# name -> (path, compile kwargs, shots per call, barrier surface supported)
# Stream-coverage map (each seed-dependent engine stream is discriminated by at
# least one workload): shot coins — all; noise-event stream (smp) — all;
# rf readout-flip stream + counter — cultivation_d5 (5 M(p) records);
# sequential Born-decision stream (dec_gen) — born5_dec/steane_h/qrm_magic;
# born-observable stream (obs_gen) — ch_cultivation (LOGICAL-class O0;
# CH/PPR-class circuit ⇒ sample surface only, retention unsupported);
# selfcheck oracle-window counter gating — steane_h (selfcheck=50).
WORKLOADS = {
    "steane_h": (_DATA / "adaptq_steane_h_producer.stim",
                 dict(selfcheck=50), 300, True),
    "qrm_magic": (_DATA / "adaptq_qrm_magic_producer.stim",
                  dict(selfcheck=0), 300, True),
    "born5_dec": (_DATA / "adaptq_born5_dec_enum_producer.stim",
                  dict(selfcheck=0), 300, True),
    "cultivation_d5": (_EXAMPLES / "cultivation_d5.stim",
                       dict(selfcheck=0, p_factor=1.0,
                            skip_refused_observables=True), 300, True),
    "ch_cultivation": (_EXAMPLES / "ch_cultivation.stim",
                       dict(selfcheck=0, skip_refused_observables=True), 300,
                       False),
}
SEEDS = (2, 7, 123)


def _compile(name):
    path, kw = WORKLOADS[name][0], WORKLOADS[name][1]
    return xtim.compile_twirl_sampler(path.read_text(), **kw)


def _snap_sample(s, shots, seed):
    """One sample() call -> tuple of copied arrays (the TwirlBatch fields;
    absent streams are None)."""
    out = s.sample(shots, seed=seed)
    return tuple(None if a is None else np.asarray(a).copy() for a in out)


def _snap_barrier(s, shots, seed):
    """One sample_barrier() call -> dict of every retained stream, copied."""
    buf = s.sample_barrier(shots, seed=seed)
    return dict(
        dets=np.asarray(buf.dets()).copy(),
        obs=np.asarray(buf.obs()).copy(),
        dec=np.asarray(buf.decisions()).copy(),
        sig=np.asarray(buf.sigmas()).copy(),
        keys=[bytes(k) for k in buf.record_keys()],
        gids=list(buf.record_group_ids()),
        coins=[bytes(bytearray(buf.coins(i))) for i in range(shots)],
        plans=[bytes(bytearray(buf.plan_key(i))) for i in range(shots)],
    )


def _assert_sample_eq(got, want, ctx):
    assert len(got) == len(want), ctx
    for i, (g, w) in enumerate(zip(got, want)):
        if g is None or w is None:
            assert g is None and w is None, f"{ctx}: field {i} None mismatch"
            continue
        assert g.shape == w.shape, f"{ctx}: array {i} shape {g.shape} != {w.shape}"
        assert np.array_equal(g, w), (
            f"{ctx}: array {i} differs in "
            f"{int((g != w).sum())} byte(s) — set_seed is NOT stream-identical")


def _assert_barrier_eq(got, want, ctx):
    for k in ("dets", "obs", "dec", "sig"):
        assert np.array_equal(got[k], want[k]), (
            f"{ctx}: barrier stream '{k}' differs — set_seed is NOT stream-identical")
    for k in ("keys", "gids", "coins", "plans"):
        assert got[k] == want[k], (
            f"{ctx}: barrier stream '{k}' differs — set_seed is NOT stream-identical")


@pytest.fixture(scope="module", params=list(WORKLOADS))
def wl(request):
    """Per-workload bundle: fresh per-seed references + one long-lived DUT.

    References are captured on FRESHLY compiled samplers, one per (surface, seed),
    whose first seeded call is the reference call. The DUT is a single sampler
    reused (and deliberately dirtied) across every test in this module — any
    history-dependence in the seeded path is a failure.
    """
    name = request.param
    shots, has_barrier = WORKLOADS[name][2], WORKLOADS[name][3]
    ref_sample = {s: _snap_sample(_compile(name), shots, s) for s in SEEDS}
    ref_barrier = ({s: _snap_barrier(_compile(name), shots, s) for s in SEEDS}
                   if has_barrier else None)
    # Construction-native seed-1 references: first call ever, seed=None — the
    # stream comes from the CONSTRUCTOR seed (kDefaultSeed = 1), not set_seed.
    native_sample = _snap_sample(_compile(name), shots, None)
    native_barrier = _snap_barrier(_compile(name), shots, None) if has_barrier else None
    dut = _compile(name)
    dut.sample(shots)              # dirty the DUT: unseeded native-stream call
    return dict(name=name, shots=shots, dut=dut, ref_sample=ref_sample,
                ref_barrier=ref_barrier, native_sample=native_sample,
                native_barrier=native_barrier)


def test_interleaved_sample_equivalence(wl):
    """sample(): A,B,A / B,C,B / zero-shot interleavings on one warm sampler,
    every call byte-identical to its fresh-sampler reference."""
    dut, shots = wl["dut"], wl["shots"]
    A, B, C = SEEDS
    order = [A, B, A, C, B, 0, C, A]     # 0 marks a zero-shot seeded call
    for step, s in enumerate(order):
        ctx = f"{wl['name']} sample step {step} seed {s}"
        if s == 0:
            got = _snap_sample(dut, 0, A)          # zero-shot seeded call
            for g in got:
                assert g is None or g.shape[0] == 0, ctx
            continue
        _assert_sample_eq(_snap_sample(dut, shots, s), wl["ref_sample"][s], ctx)


def test_interleaved_barrier_equivalence(wl):
    """sample_barrier(): same interleaved contract on every retained stream
    (dets/obs/decisions/sigmas/coins/plan keys/record keys/group ids)."""
    if wl["ref_barrier"] is None:
        pytest.skip("retention unsupported (CH/PPR-class circuit)")
    dut, shots = wl["dut"], wl["shots"]
    A, B, C = SEEDS
    for step, s in enumerate([B, A, B, 0, C, A]):
        ctx = f"{wl['name']} barrier step {step} seed {s}"
        if s == 0:
            got = _snap_barrier(dut, 0, B)         # zero-shot seeded call
            assert got["dets"].shape[0] == 0, ctx
            continue
        _assert_barrier_eq(_snap_barrier(dut, shots, s), wl["ref_barrier"][s], ctx)


def test_mixed_sample_barrier_sequences(wl):
    """Mixed surfaces: a seeded call on either surface is unaffected by prior
    calls on the other surface (including zero-shot calls in between)."""
    if wl["ref_barrier"] is None:
        pytest.skip("retention unsupported (CH/PPR-class circuit)")
    dut, shots = wl["dut"], wl["shots"]
    A, B, C = SEEDS
    _assert_sample_eq(_snap_sample(dut, shots, A), wl["ref_sample"][A],
                      f"{wl['name']} mixed: sample(A) after barrier history")
    _assert_barrier_eq(_snap_barrier(dut, shots, B), wl["ref_barrier"][B],
                       f"{wl['name']} mixed: barrier(B) after sample(A)")
    _snap_sample(dut, 0, C)                        # zero-shot seeded, surface 1
    _snap_barrier(dut, 0, A)                       # zero-shot seeded, surface 2
    _assert_sample_eq(_snap_sample(dut, shots, C), wl["ref_sample"][C],
                      f"{wl['name']} mixed: sample(C) after zero-shot calls")
    _assert_barrier_eq(_snap_barrier(dut, shots, A), wl["ref_barrier"][A],
                       f"{wl['name']} mixed: barrier(A) last")


def test_set_seed_matches_construction_native_stream(wl):
    """The anchor: set_seed(1) on a WARM sampler reproduces the CONSTRUCTOR's own
    seed-1 stream (captured on a fresh compile via a first-ever seed=None call,
    which never routes through set_seed). This pins the in-place reseed to
    construction bytes, not merely to another set_seed call."""
    dut, shots = wl["dut"], wl["shots"]
    _assert_sample_eq(_snap_sample(dut, shots, 1), wl["native_sample"],
                      f"{wl['name']} native: sample(seed=1) vs construction")
    if wl["native_barrier"] is not None:
        _assert_barrier_eq(_snap_barrier(dut, shots, 1), wl["native_barrier"],
                           f"{wl['name']} native: barrier(seed=1) vs construction")


def test_unseeded_continuation_after_seeded_zero_shot(wl):
    """seed continuation: sample(0, seed=s) then sample(N, seed=None) on a warm
    sampler == the same two calls on a fresh sampler (the zero-shot seeded call
    fully re-bases the stream; the unseeded call continues from it)."""
    name, shots = wl["name"], wl["shots"]
    s = SEEDS[1]
    fresh = _compile(name)
    _snap_sample(fresh, 0, s)
    want = _snap_sample(fresh, shots, None)
    dut = wl["dut"]                                # warm, heavily used
    _snap_sample(dut, 0, s)
    got = _snap_sample(dut, shots, None)
    _assert_sample_eq(got, want,
                      f"{name}: unseeded continuation after zero-shot seeded call")


def test_used_counter_parity_with_fresh(wl):
    """channel_report()['used'] resets on a seeded call exactly like a fresh
    sampler (the rebuild path's observable counter semantics are preserved)."""
    dut, shots = wl["dut"], wl["shots"]
    dut.sample(shots, seed=SEEDS[0])
    assert dut.channel_report()["used"] == shots


def test_seeded_zero_shot_is_fast():
    """The perf cliff this arc kills: a seeded zero-shot call on a warm
    cultivation-d5 sampler must not pay the ~180 ms reconstruction. Bound is
    generous (5 ms, was ~176-184 ms — a 36x margin) and taken as a min over
    several calls to be robust to host load."""
    s = _compile("cultivation_d5")
    s.sample(2000, seed=2)                         # warm the caches
    best = min(_timed_zero_shot(s, 200 + k) for k in range(5))
    assert best < 5e-3, (
        f"seeded zero-shot call took {best * 1e3:.1f} ms — the per-call sampler "
        f"reconstruction is back")


def _timed_zero_shot(s, seed):
    t0 = time.perf_counter()
    s.sample(0, seed=seed)
    return time.perf_counter() - t0
