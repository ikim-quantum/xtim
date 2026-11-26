"""XdemSampler (SPEC §4.7, §6 T5): ordered composition, status C, stream, postselection, fired lines.

Oracles (test-only): tests/_xdem_model.py (`XdemModel`: an independent reader, the stabilizer model as a stim circuit
and its X-readout law), the in-test definition C = {e_L : e in span(coins), e_D = 0} (SPEC §1), and
`_xdem_model.exact_file_law` (the file's full outcome distribution, shared with tests/test_xdem_fuzz.py). Hooks (cpp/testapi/xdem_sampler_py.cpp, registered in
cpp/testapi/record_law_oracle_py.cpp): `xdem_fire_counts` (the production noise walk), `xdem_status_of` /
`xdem_law_of` (the production status / law of a forced fired set), `xdem_sample_fixed`, `xdem_span_check`,
`xdem_sample`.

Two production paths (xdem_sampler.hpp): the one-word coordinate path ("fast": rho <= 64, r <= 64) and the cached
exact-law path ("general"; forced with the testapi lever force_general). Exactness legs run on both.

Ported from tests/test_dc_sampler.py and tests/test_dc_postselect.py (19f2f78): status C (Review Focus 4 of the dc
plan), chi-square against the exact file law, line firing and co-firing, the score hand cases, thread and seed
independence, the rho / r = 64 boundary.
"""
import hashlib
import json
import math
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import stim

import xtim
from xtim import _xtim
from xtim import _xtim_testapi as ta
from xtim.errors import XtimXdemError

from _xdem_model import XdemModel, affine, exact_file_law, reduce, rref
from _xdem_synth import boundary_text, random_text

ROOT = Path(__file__).parent.parent
DATA = ROOT / "tests" / "data" / "xdem"
PINS = json.loads((DATA / "plain_sampler_sha256.json").read_text())

ORDER = """xdem 1 detectors 1 observables 0 gauge 1 hidden 1 tc 3
axis Q0 img D0 gauge g0
axis Q1 img D0 gauge g0 hidden h0
error(0.5) rot +Q1
error(0.6) x Q0
error(0.5) rot +Q0
"""


def unpack(dets, obs, D, O):
    d = np.unpackbits(dets, axis=1, bitorder="little", count=D).astype(np.int64)
    o = np.unpackbits(obs, axis=1, bitorder="little", count=O).astype(np.int64) if O else np.zeros((len(d), 0), np.int64)
    w = (1 << np.arange(D + O, dtype=object))
    return [int(sum(int(b) * int(x) for b, x in zip(np.concatenate([dr, orow]), w))) for dr, orow in zip(d, o)]


def in_law(M, ids, v):
    off, coins = affine(M.law(M.circuit(ids)), M.n)
    return reduce(v ^ off, coins) == 0


def check_every_shot(text, shots, seed, threads=1):
    M = XdemModel.from_text(text)
    s = _xtim._XdemSampler(text, 65536)
    dets, obs, cs, off, ids = s.sample(shots, seed, threads, 0, True)
    vals = unpack(dets, obs, M.D, M.O)
    for k in range(shots):
        assert in_law(M, list(ids[off[k]:off[k + 1]]), vals[k]), (k, list(ids[off[k]:off[k + 1]]))


def test_out_of_order_chains_compose_in_file_order():                   # Review Focus 1
    """lines 0 and 2 share p = 0.5 (one chain, walked first), line 1 (p = 0.6) negates line 2's rotation: with all three
    fired the file-order composite gives D0 = 0; the walk order 0, 2, 1 would give D0 = 1"""
    s = _xtim._XdemSampler(ORDER, 65536)
    dets, obs, cs, off, ids = s.sample(20000, 1, 1, 0, True)
    d0 = np.unpackbits(dets, axis=1, bitorder="little", count=1)[:, 0]
    allf = [k for k in range(20000) if list(ids[off[k]:off[k + 1]]) == [0, 1, 2]]
    assert len(allf) > 1000 and not d0[allf].any()
    assert np.array_equal(s.sample(20000, 1, 1, 0, False)[0], dets)     # sample() composes in file order too
    check_every_shot(ORDER, 3000, 2)


@pytest.mark.parametrize("seed", range(30))
def test_every_shot_lies_in_its_fired_law(seed):
    rng = random.Random(seed)
    text = random_text(rng, D=rng.randint(1, 5), O=rng.randint(0, 2), r=rng.randint(1, 3), m=rng.randint(0, 2),
                       A=rng.randint(1, 5), L=rng.randint(2, 8))
    check_every_shot(text, 500, seed, threads=1 + seed % 3)


def test_fired_true_is_the_same_bytes_and_counts():                      # T5(c)
    text = random_text(random.Random(7), D=5, O=2, r=3, m=2, A=5, L=8)
    s = _xtim._XdemSampler(text, 65536)
    a = s.sample(5000, 3, 1, 0, False)
    b = s.sample(5000, 3, 1, 0, True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b[:3]))
    counts = np.bincount(b[4], minlength=8)
    assert list(counts) == list(ta.xdem_fire_counts(text, 5000, 3)[0])     # (text, shots, seed, pairs=[], first_block=0)


# port: test_dc_sampler.py::test_thread_and_seed_independence
def test_threads_and_seeds():
    text = random_text(random.Random(8), D=6, O=1, r=3, m=1, A=6, L=10)
    s = _xtim._XdemSampler(text, 65536)
    base = s.sample(5000, 11, 1, 0, False)
    for th in (2, 8):
        assert all(np.array_equal(x, y) for x, y in zip(base, s.sample(5000, 11, th, 0, False)))
    assert not np.array_equal(base[0], s.sample(5000, 12, 1, 0, False)[0])


@pytest.mark.parametrize("r,fast", [(64, True), (65, True)])   # XFM: r = 65 is the multi-word fast path
def test_fast_path_boundary(r, fast):
    text = boundary_text(r, 60, 4)
    assert ta.xdem_status_of(text, [0], False)["fast_path"] == fast
    rng = random.Random(r)
    for _ in range(40):
        ids = sorted(rng.sample(range(2 * r), rng.randint(1, 20)))
        st, st_g = ta.xdem_status_of(text, ids, False), ta.xdem_status_of(text, ids, True)   # (text, ids, force_general)
        C, Cg = rref(st["coins"]), rref(st_g["coins"])          # offset / coins come unreduced (status_of's contract)
        assert C == Cg and reduce(st["offset"], C) == reduce(st_g["offset"], C)
        assert st["cspace"] == st_g["cspace"] and st["dim"] == st_g["dim"]


def _ext_lines(M):
    """the sampler's extended lines: a nonzero total exponent, or a tau on an axis some line rotates about"""
    rotated = 0
    for (_, _, _, rot, late) in M.lines:
        for a in list(rot) + list(late):
            rotated |= 1 << a
    return [bool(M.total(rot, late)) or bool(tau & rotated) for (_, _, tau, rot, late) in M.lines]


def _in_span_masked(v, vecs, mask):
    return reduce(v & mask, rref([x & mask for x in vecs])) == 0


# port: test_dc_postselect.py (survivors in law, count vs filter) + preflight S5
def test_postselect_on_extended_files():                                 # Review Focus 5
    for seed in range(6):
        text = random_text(random.Random(40 + seed), D=6, O=1, r=3, m=2, A=5, L=9)
        M = XdemModel.from_text(text)
        s = _xtim._XdemSampler(text, 65536)
        mask = np.zeros(6, bool)
        mask[[0, 3]] = True
        dets, obs, cs, counts, off, ids = s.sample_ps(20000, 5, 1 + seed % 3, 0, mask, True)
        vals = unpack(dets, obs, M.D, M.O)
        for k in range(counts["passed"]):
            assert in_law(M, list(ids[off[k]:off[k + 1]]), vals[k])
            assert not (vals[k] >> 0 & 1) and not (vals[k] >> 3 & 1)
        full = s.sample(20000, 5, 1, 0, False)
        dfull = np.unpackbits(full[0], axis=1, bitorder="little", count=6)
        acc = int((~dfull[:, [0, 3]].any(axis=1)).sum())
        z = (counts["passed"] - acc) / max(1.0, np.sqrt(2 * acc * (1 - acc / 20000)))
        assert abs(z) <= 5 and counts["early"] <= counts["discarded"]
        # preflight S5: the early rejections are exact. (a) the span test's premise on the stream: no correction moves
        # the detector row outside span(img)_D; (b) early = the shots with an extended line whose row after the line
        # XOR, restricted to the mask, is outside span(img)_mask (recomputed here from the oracle's lines and the same
        # walk's fired lists).
        bad, xs = ta.xdem_span_check(text, 20000, 5)
        assert bad == 0 and xs > 0
        ext = _ext_lines(M)
        _, _, _, foff, fids = s.sample(20000, 5, 1, 0, True)
        mbits = (1 << 0) | (1 << 3)
        imgs = [M.img(a) for a in range(M.A)]
        early = 0
        for k in range(20000):
            L = list(fids[foff[k]:foff[k + 1]])
            if not any(ext[i] for i in L):
                continue
            d0 = 0
            for i in L:
                d0 ^= M.lines[i][1]
            early += not _in_span_masked(d0, imgs, mbits)
        assert counts["early"] == early, (seed, counts, early)


# ---------------------------------------------------------------------------------------------- postselection, both paths
# port: test_dc_postselect.py::test_postselect_both_paths (19f2f78), on the xdem fixtures plus the general-path boundary
# file (r = 65 > 64: general whatever the lever). Per path: the span test's premise (xdem_span_check), every mask (all,
# half, one detector, empty) at threads 1 and 8 (accept rule; byte identity across threads, which also runs the
# multi-thread survivor compaction), and the law of the survivors against sample() + filter on independent seeds.

PS_SHOTS, PS_LAW_SHOTS, PS_ZMAX = 2500, 60_000, 5.5
PS_FILES = sorted(p.name for p in DATA.glob("*.xdem")) + sorted(p.name for p in DATA.glob("*.dem")) + \
    ["synth/boundary_r64.xdem", "synth/boundary_r65.xdem"]


def _masks(D, seed, empty=True):
    rng = np.random.default_rng(seed)
    one = np.zeros(D, bool)
    if D:
        one[rng.integers(D)] = True
    m = {"all": np.ones(D, bool), "half": rng.random(D) < 0.5, "one": one}
    if empty:
        m["empty"] = np.zeros(D, bool)
    return m


def _bits(packed, n):
    if n == 0:
        return np.zeros((packed.shape[0], 0), bool)
    return np.unpackbits(packed, axis=1, bitorder="little", count=n).astype(bool)


def _ztest(k1, n1, k2, n2, what):
    """two-proportion z-test (pooled); |z| <= PS_ZMAX. Skipped when the pooled proportion is 0 or 1."""
    if n1 == 0 or n2 == 0:
        return
    p = (k1 + k2) / (n1 + n2)
    if p <= 0 or p >= 1:
        assert k1 == k2 == 0 or (k1 == n1 and k2 == n2), what
        return
    z = (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    assert abs(z) <= PS_ZMAX, (what, z, k1, n1, k2, n2)


def _compare_law(dps, ops, dims_ps, dref, oref, dims_ref, attempts, D, O, where):
    """survivors vs the filtered reference of `attempts` attempted shots each: acceptance, per-detector and
    per-observable marginals, dim-C histogram"""
    n1, n2 = len(dps), len(dref)
    _ztest(n1, attempts, n2, attempts, (where, "acceptance"))
    for name, a, b in (("det", _bits(dps, D), _bits(dref, D)), ("obs", _bits(ops, O), _bits(oref, O))):
        ka, kb = a.sum(axis=0), b.sum(axis=0)
        for j in range(a.shape[1]):
            _ztest(int(ka[j]), n1, int(kb[j]), n2, (where, name, j))
    for dim in sorted(set(dims_ps) | set(dims_ref)):
        _ztest(int((dims_ps == dim).sum()), n1, int((dims_ref == dim).sum()), n2, (where, "dimC", dim))


@pytest.mark.parametrize("name", PS_FILES)
def test_postselect_both_paths(name):
    text = (DATA / name).read_text()
    s = _xtim._XdemSampler(text, 65536)
    D, O = s.num_detectors, s.num_observables
    for general in (False, True):
        bad, xs = ta.xdem_span_check(text, 6000, 5, general)
        assert bad == 0, (general, bad, xs)
        for mname, m in _masks(D, 7).items():
            f = m.astype(np.uint8).tolist()
            outs = [ta.xdem_sample_ps(text, PS_SHOTS, 21, th, f, general) for th in (1, 8)]
            (pd, po, pc, _), (d0, o0, c0, _), fast, _, _ = outs[0]
            assert not (general and fast)
            if name == "synth/boundary_r65.xdem":                    # r = 65: two-word coordinates (XFM), the
                assert fast == (not general)                         # general path's stream: the same bytes
                if not general:
                    ref = ta.xdem_sample_ps(text, PS_SHOTS, 21, 1, f, True)
                    assert all(np.array_equal(np.asarray(a), np.asarray(b))
                               for a, b in zip(list(outs[0][0]) + list(outs[0][1]), list(ref[0]) + list(ref[1])))
            assert not _bits(pd, D)[:, m].any(), (general, mname)
            assert all(a.tobytes() == b.tobytes() for a, b in zip(outs[0][0][:3], outs[1][0][:3])), (general, mname)
            if mname == "empty":                                      # nothing flagged: every shot survives,
                assert pd.tobytes() == d0.tobytes() and pc.tobytes() == c0.tobytes()   # sample()'s bytes, no early
                assert outs[0][0][3] == 0
                continue
            # the law on independent seeds (the plain call of a second xdem_sample_ps call, seed + 1)
            (pd, po, pc, _), _, _, dims_ps, _ = ta.xdem_sample_ps(text, PS_LAW_SHOTS, 31, 8, f, general)
            _, (d, o, c, _), _, _, dims_ref = ta.xdem_sample_ps(text, PS_LAW_SHOTS, 32, 8, f, general)
            keep = ~_bits(d, D)[:, m].any(axis=1)
            dp, dr = np.asarray(dims_ps)[pc], np.asarray(dims_ref)[c[keep]]
            _compare_law(pd, po, dp, d[keep], o[keep], dr, PS_LAW_SHOTS, D, O, (general, mname))


# ---------------------------------------------------------------------------------------------- conventional DEM bytes

def _plain_text(name):
    if name.endswith(".stim"):
        return xtim.Circuit((ROOT / name).read_text()).detector_error_model_text()
    # the committed text (stim 1.16.0 on Linux x86_64), not live stim: stim prints these probabilities with other
    # digits on other platforms (and some differ in the last bits), which moves the sampler's thresholds
    return (DATA / "synth" / (name + ".dem")).read_text()[:-1]


def _stim_text(name):
    g = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                               after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3)
    return {"surface_d5r5": str(g.detector_error_model()),
            "surface_d5r5_decomposed": str(g.detector_error_model(decompose_errors=True)),
            "surface_d5r5_flat": str(g.detector_error_model().flattened())}[name]


@pytest.mark.parametrize("name", sorted(k for k, v in PINS.items() if "5000_7_1" in v))
def test_conventional_dem_bytes_equal_base(name):
    """SPEC §6 T4: a conventional DEM samples identically to today (BASE 19f2f78 v1-sampler plain-DEM pins)"""
    s = _xtim._XdemSampler(_plain_text(name), 65536)
    assert not s.extended and s.num_axes == 0 and s.tc == s.num_lines
    for case, want in PINS[name].items():
        if case.startswith("fired"):
            d, o, cs, off, ids = s.sample(5000, 7, 1, 0, True)
            got = hashlib.sha256(d.tobytes() + off.tobytes() + ids.tobytes()).hexdigest()
        else:
            shots, seed, th = map(int, case.split("_"))
            d, o, cs = s.sample(shots, seed, th, 0, False)
            got = hashlib.sha256(d.tobytes() + o.tobytes() + cs.tobytes()).hexdigest()
        assert got == want, (name, case)


def test_synth_boundary_fixtures_are_the_generated_text():
    for r in (64, 65):
        assert (DATA / "synth" / f"boundary_r{r}.xdem").read_text() == boundary_text(r, 60, 4)


@pytest.mark.skipif(not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")),
                    reason="stim prints these probabilities differently off Linux x86_64 (the reference platform)")
@pytest.mark.parametrize("name", ["surface_d5r5", "surface_d5r5_decomposed", "surface_d5r5_flat"])
def test_synth_surface_fixture_is_the_pinned_text(name):
    """tests/data/xdem/synth/<name>.dem (the pins' conventional texts; surface_d5r5 is also the golden ctest's) =
    live stim's text on the reference platform: a check for stim drift"""
    assert (DATA / "synth" / (name + ".dem")).read_text() == _stim_text(name) + "\n"


# ---------------------------------------------------------------------------------------------- properties

def test_properties():
    text = (DATA / "ex422_obs.xdem").read_text()
    s = _xtim._XdemSampler(text, 65536)
    assert (s.extended, s.num_detectors, s.num_observables, s.num_gauge, s.num_hidden, s.num_axes, s.num_lines,
            s.num_rotating, s.tc, s.fast_path) == (True, 3, 2, 3, 2, 8, 3, 2, 3, True)
    s.sample(3000, 1, 2, 0, False)
    k = s.counters()
    assert k["shots"] == 3000 and k["thread_slots"] == 2 and k["cache_cap"] == 65536 and k["fast_path"]
    assert 0 < k["ext_shots"] <= 3000 and k["coin_shots"] <= k["ext_shots"]
    assert all(k[f] == 0 for f in ("lines_shots", "lines_lossy", "lines_outside", "kmemo_entries", "kmemo_clears"))
    with pytest.raises(ValueError):
        s.sample(10, 1, 0, 0, False)                                      # threads >= 1
    with pytest.raises(XtimXdemError):
        _xtim._XdemSampler(text.replace("tc 3", "tc 9"), 65536)          # a reader refusal surfaces by name


# ---------------------------------------------------------------------------------------------- status C

def c_definition(D, coins):
    """C = {e_L : e in span(coins), e_D = 0} as its RREF dict, by elimination on the detector bits (independent of
    the C++), cross-checked by brute-force span enumeration when the rank is <= 12"""
    dmask = (1 << D) - 1
    rows = list(coins)
    kern, piv = [], []
    for v in rows:
        for p, r in piv:
            if v >> p & 1:
                v ^= r
        if v & dmask:
            piv.append(((v & dmask).bit_length() - 1, v))
        else:
            kern.append(v >> D)
    by_elim = rref(kern)
    if len(rows) <= 12:
        span = {0}
        for c in rows:
            span |= {x ^ c for x in span}
        assert rref([x >> D for x in span if not x & dmask]) == by_elim
    return by_elim


def _status_check(text, cfgs, general):
    M = XdemModel.from_text(text)
    dims = Counter()
    for cfg in cfgs:
        off, coins = affine(M.law(M.circuit(cfg)), M.n)
        want = c_definition(M.D, coins.values())
        st = ta.xdem_status_of(text, cfg, general)
        lw = ta.xdem_law_of(text, cfg, general)
        assert st["fast_path"] == lw["fast_path"]
        if general:
            assert not st["fast_path"]
        assert rref(st["cspace"]) == want and st["dim"] == len(want), (cfg, st["cspace"], want)
        for got in (st, lw):                                          # the sampler's law is the oracle's law
            assert rref(got["coins"]) == coins, cfg
            assert reduce(got["offset"], coins) == off, cfg
        assert lw["dim"] == len(want)
        dims[len(want)] += 1
    return dims


PATHS = pytest.mark.parametrize("general", [False, True], ids=["fast", "general"])


# port: test_dc_sampler.py::test_status_multiobs / test_status_hand_circuit (C vs its definition, Review Focus 4)
@PATHS
def test_status_synthetic(general):
    """random valid files (third-party gauges) with O = 2..3: every single line, random multisets with unsorted and
    repeated ids (dim C 0 and 1 occur); and the hand file TWO_COIN on every subset (dims 0, 1, 2)"""
    dims = Counter()
    for seed in range(12):
        rng = random.Random(500 + seed)
        text = random_text(rng, D=rng.randint(1, 4), O=rng.randint(2, 3), r=rng.randint(1, 4), m=rng.randint(0, 2),
                           A=rng.randint(2, 6), L=rng.randint(4, 10))
        L = len(XdemModel.from_text(text).lines)
        cfgs = [[i] for i in range(L)] + [[rng.randrange(L) for _ in range(rng.randint(2, 7))] for _ in range(25)]
        dims += _status_check(text, cfgs, general)
    print(f"synthetic ({'general' if general else 'fast'}): dims {dict(dims)}")
    assert dims[1] > 0
    two = _status_check(TWO_COIN, [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2], [0, 0], [0, 1, 1], [2, 1, 0]],
                        general)
    assert two == {0: 2, 1: 5, 2: 3}, two


# port: test_dc_sampler.py::test_status_fixture
@PATHS
@pytest.mark.parametrize("name", ["ex422_obs", "cultivation_d3_faithful", "tdag_cycle"])
def test_status_fixture(name, general):
    text = (DATA / f"{name}.xdem").read_text()
    M = XdemModel.from_text(text)
    rng = random.Random(len(text))
    ext = [i for i, e in enumerate(_ext_lines(M)) if e]
    cfgs = [[i] for i in ext[:60]] + [[rng.choice(ext) for _ in range(rng.randint(2, 8))] for _ in range(80)]
    dims = _status_check(text, cfgs, general)
    print(f"{name} ({'general' if general else 'fast'}): {len(cfgs)} configs, dims {dict(dims)}")


# ---------------------------------------------------------------------------------------------- the exact file law

def chi_z(counts, probs, n):
    """Pooled chi-square z for observed counts vs exact probabilities (cells with n*p < 5 pooled)."""
    big = n * probs >= 5
    obs = list(counts[big]) + [counts[~big].sum()]
    exp = list(n * probs[big]) + [n * probs[~big].sum()]
    if exp[-1] < 5:            # fold a small pooled cell into the smallest big cell
        obs[-2] += obs[-1]
        exp[-2] += exp[-1]
        obs, exp = obs[:-1], exp[:-1]
    obs, exp = np.array(obs, float), np.array(exp, float)
    chi = float(((obs - exp) ** 2 / exp).sum())
    df = len(obs) - 1
    return (chi - df) / np.sqrt(2 * df) if df > 0 else 0.0, df


def _rescaled(text, seed, lo=0.05, hi=0.4):
    """the same file with every probability redrawn in [lo, hi] (a valid edit): richer laws per shot"""
    rng = random.Random(seed)
    return re.sub(r"^error\(([^)]*)\)", lambda m: f"error({rng.uniform(lo, hi):.17g})", text, flags=re.M)


def _chi_cases():
    rng1, rng2 = random.Random(7301), random.Random(7302)
    return [("ex422_obs_rescaled", _rescaled((DATA / "ex422_obs.xdem").read_text(), 7300)),
            ("synth_a", random_text(rng1, D=3, O=2, r=3, m=2, A=5, L=10)),
            ("synth_b", random_text(rng2, D=4, O=1, r=2, m=1, A=4, L=11))]


# port: test_dc_sampler.py::test_exact_law_agreement_4e6
@pytest.mark.slow
@PATHS
@pytest.mark.parametrize("case", range(3), ids=[c[0] for c in _chi_cases()])
def test_exact_law_agreement_4e6(case, general):
    name, text = _chi_cases()[case]
    M = XdemModel.from_text(text)
    law = exact_file_law(text)
    n_out, shots = 1 << M.n, 4_000_000
    probs = np.zeros(n_out)
    for k, v in law.items():
        probs[k] = v
    (dets, obs, cs), = ta.xdem_sample(text, [(shots, 7401 + case + 50 * general)], 4, general)[0]
    db = np.unpackbits(dets, axis=1, bitorder="little", count=M.D).astype(np.int64)
    ob = np.unpackbits(obs, axis=1, bitorder="little", count=M.O).astype(np.int64)
    v = db @ (1 << np.arange(M.D)) + (ob @ (1 << np.arange(M.D, M.n)) if M.O else 0)
    counts = np.bincount(v, minlength=n_out).astype(float)
    outside = int(counts[probs == 0].sum())
    z, df = chi_z(counts, probs, shots)
    print(f"{name} ({'general' if general else 'fast'}): {shots} shots, outside support {outside}, chi-square z "
          f"{z:.2f} (df {df})")
    assert outside == 0 and abs(z) <= 5


# ---------------------------------------------------------------------------------------------- line firing

def _firing_text():
    """cultivation_d3_faithful.xdem with probabilities redrawn in [0.02, 0.5], every 7th line p = 0.25 (equal-p
    chains kept): large p makes the geometric skip's off-by-one (relative over-firing ~ p) visible"""
    rng = random.Random(81)
    k = [0]

    def sub(m):
        k[0] += 1
        return f"error({0.25 if k[0] % 7 == 0 else rng.uniform(0.02, 0.5):.17g})"
    return re.sub(r"^error\(([^)]*)\)", sub, (DATA / "cultivation_d3_faithful.xdem").read_text(), flags=re.M)


# port: test_dc_sampler.py::test_line_firing_bernoulli
def test_line_firing_and_co_firing():
    text = _firing_text()
    ps = np.array([ln[0] for ln in XdemModel.from_text(text).lines])
    n = 200_000
    eq = [i for i in range(len(ps)) if ps[i] == 0.25]
    pairs = [(eq[0], eq[1]), (eq[2], eq[-1]), (0, 1), (3, eq[3])]   # same chain (x2), different chains (x2)
    counts, pc = ta.xdem_fire_counts(text, n, 8101, pairs)
    z = (counts - n * ps) / np.sqrt(n * ps * (1 - ps))
    print(f"line firing: {len(ps)} lines, {n} shots, max |z| {np.abs(z).max():.2f}")
    assert np.abs(z).max() <= 5
    for (i, j), c in zip(pairs, pc):
        q = ps[i] * ps[j]
        zc = (c - n * q) / np.sqrt(n * q * (1 - q))
        print(f"co-firing ({i},{j}) p_i p_j = {q:.4f}: z {zc:.2f}")
        assert abs(zc) <= 5
    # first_block: counts over blocks [5, 10) are the same walk as blocks 5..9 of a longer run
    c1, _ = ta.xdem_fire_counts(text, 5 * 1024, 8102, [], 5)
    c2, _ = ta.xdem_fire_counts(text, 10 * 1024, 8102, [], 0)
    c3, _ = ta.xdem_fire_counts(text, 5 * 1024, 8102, [], 0)
    assert np.array_equal(c1 + c3, c2)


# ---------------------------------------------------------------------------------------------- score

TWO_COIN = """xdem 1 detectors 1 observables 3 gauge 3 hidden 0 tc 3
axis Q0 img L0 gauge g0
axis Q1 img L1 gauge g1
axis Q2 img D0 L2 gauge g2
error(0.1) rot +Q0
error(0.1) rot +Q1
error(0.1) rot +Q2
"""


# port: test_dc_sampler.py::test_score_hand_cases
def test_score_hand_cases():
    """D = 1, O = 3: the coin of Q0 opens L0 alone (dim C 1), Q0 and Q1 together dim 2, Q2's coin moves D0 too (dim 0)"""
    s1, s2, s0 = (ta.xdem_status_of(TWO_COIN, ids) for ids in ([0], [0, 1], [2]))
    assert (s1["dim"], s2["dim"], s0["dim"]) == (1, 2, 0)
    assert sorted(s2["cspace"]) == [1, 2] and s1["cspace"] == [1]
    S = _xtim._XdemSampler(TWO_COIN, 65536)
    d, o, c = S.sample(20_000, 9301, 1, 0, False)          # the public sample path fills the table
    tab = [frozenset(int("".join(map(str, row[::-1])), 2) for row in a) for a in S.cspaces()]
    id1, id2 = tab.index(frozenset(s1["cspace"])), tab.index(frozenset(s2["cspace"]))
    obs = np.packbits(np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 1, 1]],
                               np.uint8), axis=1, bitorder="little")
    cs = np.array([0, 0, id1, id2, id2, 0, id2], np.int32)
    fl = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 0]], np.uint8)
    assert list(S.score(obs, cs, fl)) == [1.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.0]
    with pytest.raises(IndexError, match="not in this sampler's table"):     # std::out_of_range
        S.score(obs, np.array([0, 0, 0, 0, 0, 0, 999], np.int32), fl)
    with pytest.raises(ValueError, match="flips must be 0 or 1"):          # std::invalid_argument
        S.score(obs, cs, fl * 2)


# ---------------------------------------------------------------------------------------------- stream identity

# port: test_dc_sampler.py::test_general_path_thread_independence_and_cache
def test_stream_identity_fixture():
    """cultivation_d5.xdem: identical bytes and C table at threads 1/2/8, both paths (the general path also with a
    4-entry cache, cleared constantly); prefix and first_block slicing of the one stream"""
    text = (DATA / "cultivation_d5.xdem").read_text()
    for general in (False, True):
        ref, rtab, rk = ta.xdem_sample(text, [(30_000, 41), (30_000, 42)], 1, general)
        assert rk["fast_path"] == (not general)
        for th, cap in ((2, 65536), (8, 65536), (3, 4)):
            got, tab, k = ta.xdem_sample(text, [(30_000, 41), (30_000, 42)], th, general, cap)
            assert all(np.array_equal(a, b) for x, y in zip(got, ref) for a, b in zip(x, y)) and tab == rtab
        assert k["cache_clears"] > 0 or not general
        assert not np.array_equal(ref[0][0], ref[1][0])
    s = _xtim._XdemSampler(text, 65536)
    d0, o0, c0 = s.sample(20_000, 1, 1, 0, False)
    d1, o1, c1 = s.sample(4096, 1, 4, 7, False)
    p0, _, _ = s.sample(5000, 1, 3, 0, False)
    assert np.array_equal(d1, d0[7 * 1024: 7 * 1024 + 4096]) and np.array_equal(o1, o0[7 * 1024: 7 * 1024 + 4096])
    assert np.array_equal(c1, c0[7 * 1024: 7 * 1024 + 4096]) and np.array_equal(p0, d0[:5000])
