"""S0(b) of the twirl consolidation: ONE keyed counter-based generator for every twirl stream, ONE
coin convention on every route (design §3.4 [E4, B1], D10, D11 — the owner-signed re-pin).

Contract (cpp/include/qeccore/keyed_rng.hpp): every draw of shot i is

    draw(seed, i, TAG, ctr) = mix(mix(seed ^ mix(TAG)) + (i * 2^20 + ctr + 1) * G)

(mix = the SplitMix64 finalizer, G = 0x9E3779B97F4A7C15), i the CUMULATIVE shot index; the noise
stream is keyed per CHANNEL k (geometric skips span shots): its draw c is
mix(mix(mix(seed ^ mix(NOISE)) + k*G) + (c + 1) * G). Pinned here, independently
of the engine (a pure-Python reference of the formula):
  * fair coin k of a shot = bit k of its COIN words (barrier coins, on the structural route);
  * readout flip slot sl fires iff RF word sl < floor(p * 2^64);
  * gauge detector g = bit g of the GAUGE words (sample() AND sample_barrier()) — on the LEGACY sampler only
    since J-b1 F: production's gauge detectors are exact Born records (record-trie directions read the BORN words;
    the GAUGE stream is not drawn), RE-PINNED 2026-10-20 (J-b1, ENGINE_VERSION xtim-engine-13): M-GAUGE;
  * channel k's noise events = the geometric skips drawn from (seed, k, NOISE, c), c = 0, 1, ...
Determinism (D10): bytes independent of the route (sample() == sample_barrier()), of the plan-cache
state (cold == in-memory warm == TWPL-warm), and of the call split (any split, not only multiples
of the 4096 noise batch); seed=None continues every stream (law test in
tests/test_coin_stream_continuation.py).
"""
import math
from pathlib import Path

import numpy as np
import pytest

import xtim
import xtim._xtim as _x

M64 = (1 << 64) - 1
G = 0x9E3779B97F4A7C15
NOISE, COIN, BORN, OBS, DEC, RF, GAUGE, EXACT = range(1, 9)
_DATA = Path(__file__).resolve().parent / "data"
_EXAMPLES = Path(__file__).resolve().parent / "data"


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def draw(seed, shot, tag, ctr):
    key = mix(seed ^ mix(tag))
    return mix((key + ((shot << 20) + ctr + 1) * G) & M64)


def _bits(a, k):
    return np.unpackbits(np.asarray(a, np.uint8), axis=1, bitorder="little")[:, :k]


def test_known_answers():
    """The generator's definition is part of the byte contract (ENGINE_VERSION)."""
    assert _x.ENGINE_VERSION == "xtim-engine-13"   # RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-VER
    assert mix(0) == 0
    assert mix(G) == 0xE220A8397B1DCDAF          # SplitMix64(seed=0) first output
    assert draw(0, 0, COIN, 0) == mix((mix(0 ^ mix(COIN)) + G) & M64)
    assert draw(5, 3, RF, 7) == mix((mix(5 ^ mix(RF)) + (3 * 2**20 + 8) * G) & M64)


def test_rf_stream_is_the_keyed_rf_word():
    """Readout flips (Stim M(p)) — the tag goes through the hash (no XOR-linear seed)."""
    p, seed, N = 0.3, 4242, 3000
    s = _x.TwirlSampler("R 0\nM(0.3) 0\nDETECTOR rec[-1]\n", 1.0, "", 0)
    d = _bits(s.sample(N, seed=seed)[0], 1)[:, 0]
    thresh = int(p * 18446744073709551616.0)
    want = np.array([draw(seed, i, RF, 0) < thresh for i in range(N)], np.uint8)
    assert np.array_equal(d, want)


def test_noise_stream_is_keyed_per_channel():
    """A single Bernoulli(p) channel: its firing shots are the keyed geometric skips of channel 0
    (U in (0,1] from the top 53 bits; skip = floor(log U / log1p(-p)))."""
    p, seed, N = 0.2, 99, 5000
    s = _x.TwirlSampler("R 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]\n", 1.0, "", 0)
    d = _bits(s.sample(N, seed=seed)[0], 1)[:, 0]
    sc = 1.0 / math.log1p(-p)
    key = mix(seed ^ mix(NOISE))
    pos = mix(key)                                   # channel_base(seed, 0) = mix(key + 0*G)

    def u():
        nonlocal pos
        pos = (pos + G) & M64
        return ((mix(pos) >> 11) + 1) * 2.0 ** -53

    want = np.zeros(N, np.uint8)
    s_next = int(math.log(u()) * sc)
    while s_next < N:
        want[s_next] = 1
        s_next += 1 + int(math.log(u()) * sc)
    assert np.array_equal(d, want)


# Two LOGICAL-class (gauge) detectors — MX of T|+> — plus one deterministic noisy detector.
def test_born_observable_reads_born_word_0_with_the_one_born_convention():
    """S2b: the Born observable is a record row; on a plan where it introduces the only record
    direction its RAW outcome is the record trie's z_0: +1 (bit 0) iff u < p(+1) with u = the shot's
    BORN word 0 (Born word j = canonical direction j; gate M-2). Noiseless T|+> read in X: every
    shot is clean, p(+1) = (1 + 1/sqrt 2)/2; the reported column is reference-relative (one constant
    XOR). The OBS stream is never drawn on a diagonal record route."""
    txt = "RX 0\nT 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 1\nM 1\nDETECTOR rec[-1]\n"
    s = _x.TwirlSampler(txt, 1.0, "", 0)
    seed, N = 8, 4000
    o = _bits(s.sample(N, seed=seed)[1], 1)[:, 0]
    p_plus = (1 + 2 ** -0.5) / 2
    raw = np.array([0 if ((draw(seed, i, BORN, 0) >> 11) * 2.0 ** -53) < p_plus else 1 for i in range(N)],
                   np.uint8)
    assert len(set((o ^ raw).tolist())) == 1          # equal up to the reference constant
    assert 0.1 < raw.mean() < 0.2                      # 1 - p(+1) = 0.146...
    assert s.record_report()["rec"]["obs_draws"]["record"] == 0


GAUGE_TXT = ("RX 0\nT 0\nMX 0\nDETECTOR rec[-1]\nRX 1\nT 1\nMX 1\nDETECTOR rec[-1]\n"
             "R 2\nX_ERROR(0.1) 2\nM 2\nDETECTOR rec[-1]\n")


@pytest.mark.parametrize("surface", ["sample", "sample_barrier"])
def test_gauge_detectors_are_born_records_on_the_keyed_born_words(surface):
    """J-b1 F (M-GAUGE): production's two gauge detectors (MX of T|+> each, independent) are exact Born records —
    the record trie's directions in record order, direction j's outcome z_j = 1 iff u_j >= p(+1) with u_j the shot's
    BORN word j (the one Born convention) — reported reference-relative (one constant per column). The GAUGE stream
    is not drawn."""
    s = _x.TwirlSampler(GAUGE_TXT, 1.0, "", 0)
    rep = s.record_report()
    assert rep["classes"]["detectors"] == {0: "born", 1: "born", 2: "deterministic"}
    seed, N = 17, 3000
    d = (_bits(s.sample(N, seed=seed)[0], 3) if surface == "sample"
         else _bits(s.sample_barrier(N, seed=seed).dets(), 3))
    p_plus = (1 + 2 ** -0.5) / 2
    for j in (0, 1):
        raw = np.array([0 if ((draw(seed, i, BORN, j) >> 11) * 2.0 ** -53) < p_plus else 1 for i in range(N)],
                       np.uint8)
        assert len(set((d[:, j] ^ raw).tolist())) == 1, j     # equal up to the reference constant
        assert 0.1 < raw.mean() < 0.2


def test_structural_route_coins_are_the_keyed_coin_bits():
    """ONE coin convention: the retention route reads canonical coin k (the value of pivot record k)
    as bit k of the shot's COIN words — the same bits the fast paths read. S2b: the record's coin
    segment is (u: rho entries) of record_layout() version 2. Exact per shot, on every shot that drew
    coins."""
    # the adv-A ANTI-decision fixture: ANTI decisions are pivot records (fair coins), and the noisy
    # T makes their plans vary
    text = ("RX 0\nR 1 2 3 4\nX_ERROR(0.25) 0\nT 0\nH 1 3\nCX 1 2 3 4\nCX 0 1\nCZ 0 3\n"
            "DEPOLARIZE1(0.06) 0 1 2 3 4\nMX 1 2 3 4 0\nDETECTOR rec[-5] rec[-4]\n"
            "DECISION(0) rec[-5]\nDECISION(1) rec[-3]\nDECISION(2) rec[-1]\n")
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    seed, N = 23, 3000
    b = s.sample_barrier(N, seed)
    checked = 0
    for i in range(N):
        r = int(b.plan_structure(i)["rho"][0])      # J-b1 F: the layout-3 plan's rho (no exact class)
        if r == 0:
            continue
        c = list(b.outcomes(i))
        words = [draw(seed, i, COIN, w) for w in range((r + 63) // 64)]
        want = [(words[k >> 6] >> (k & 63)) & 1 for k in range(r)]
        assert c[:r] == want, i
        checked += 1
    assert checked > 100, checked


# ── D10: bytes independent of route, cache state and call split ───────────────────────────────
WORKLOADS = {
    "qrm_magic": (_DATA / "adaptq_qrm_magic_producer.stim").read_text(),
    "born5_dec": (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text(),
    "cultivation_d5": (_EXAMPLES / "cultivation_d5.stim").read_text(),
    "gauge": GAUGE_TXT,
}


def _compile(text, **kw):
    return xtim.compile_twirl_sampler(text, selfcheck=0, **kw)


def _fields(out):
    return [None if f is None else np.asarray(f) for f in out]


@pytest.mark.parametrize("name", list(WORKLOADS))
def test_sample_equals_sample_barrier(name):
    """Route independence: the retention route and the sample() fast paths read the same keyed
    draws (no exact-fallback route fires on these workloads)."""
    s = _compile(WORKLOADS[name])
    a = _fields(s.sample(3000, seed=31))
    b = s.sample_barrier(3000, seed=31)
    rr = s.record_report()["route_counts"]
    assert not any(k.startswith("exact") for k in rr), rr       # J-b1 F: no exact route exists
    assert np.array_equal(a[0], np.asarray(b.dets()))
    assert np.array_equal(a[1], np.asarray(b.obs()))
    if len(a) > 2 and a[2] is not None:
        assert np.array_equal(a[2], np.asarray(b.decisions()))


@pytest.mark.parametrize("name", list(WORKLOADS))
def test_cold_warm_and_twpl_warm_bytes_identical(name, tmp_path):
    text = WORKLOADS[name]
    cache = str(tmp_path / "plans.twpl")
    cold = _fields(_compile(text, disk_cache=cache).sample(4000, seed=7))       # saves the TWPL file
    s = _compile(text, disk_cache=cache)                                        # loads it: TWPL-warm
    twpl = _fields(s.sample(4000, seed=7))
    misses_twpl = s.record_report()["plan_misses"]
    warm = _fields(s.sample(4000, seed=7))                                      # in-memory warm
    assert s.record_report()["plan_misses"] == misses_twpl                      # no build: warm
    fresh = _fields(_compile(text).sample(4000, seed=7))                        # cold, no disk
    for f in (twpl, warm, fresh):
        for x, y in zip(cold, f):
            assert (x is None) == (y is None)
            if x is not None:
                assert np.array_equal(x, y)


@pytest.mark.parametrize("name", list(WORKLOADS))
def test_any_split_equals_single_call(name):
    """The noise stream is keyed per channel, so ANY split continues byte-for-byte (the former
    4096-shot batch-multiple restriction is gone)."""
    text = WORKLOADS[name]
    N = 5000
    one = _fields(_compile(text).sample(N, seed=11))
    rng = np.random.default_rng(1)
    for trial in range(3):
        cuts = sorted(set(int(x) for x in rng.integers(1, N, size=1 + trial * 3)))
        sizes = np.diff([0] + cuts + [N])
        s = _compile(text)
        parts = [_fields(s.sample(int(sizes[0]), seed=11))] + [_fields(s.sample(int(k))) for k in sizes[1:]]
        for i, w in enumerate(one):
            if w is None:
                continue
            assert np.array_equal(w, np.concatenate([p[i] for p in parts])), (name, list(sizes), i)


@pytest.mark.parametrize("name", ["qrm_magic", "born5_dec"])
def test_barrier_any_split_equals_single_call(name):
    text = WORKLOADS[name]
    N = 3000
    one = _compile(text).sample_barrier(N, seed=13)
    s = _compile(text)
    sizes = [1, 1000, 7, N - 1008]
    parts = [s.sample_barrier(sizes[0], seed=13)] + [s.sample_barrier(k) for k in sizes[1:]]
    for acc in ("dets", "obs", "decisions"):
        w = np.asarray(getattr(one, acc)())
        got = np.concatenate([np.asarray(getattr(p, acc)()) for p in parts])
        assert np.array_equal(w, got), acc
    assert [bytes(k) for k in one.record_keys()] == [bytes(k) for p in parts for k in p.record_keys()]
    coins_one = [bytes(one.outcomes(i)) for i in range(N)]
    coins_split = [bytes(p.outcomes(i)) for p, k in zip(parts, sizes) for i in range(k)]
    assert coins_one == coins_split
