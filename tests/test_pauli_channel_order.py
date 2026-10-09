"""PAULI_CHANNEL_1 / PAULI_CHANNEL_2 argument order on EVERY noise-consuming path, against Stim.

Stim's argument order (stim gate docs): PAULI_CHANNEL_1(px, py, pz) and
PAULI_CHANNEL_2(p_IX, p_IY, p_IZ, p_XI, p_XX, p_XY, p_XZ, p_YI, p_YX, p_YY, p_YZ, p_ZI,
p_ZX, p_ZY, p_ZZ), where the FIRST letter acts on the FIRST target of the pair. (3.1.6 and
earlier: the twirl sampler used a transposed table — only ZZ and DEPOLARIZE2 came out right.)

Probe: two pairs share one PAULI_CHANNEL_2 instruction — pair (0,1) is Z-prepared and
Z-measured (detectors = X-parts of the Pauli), pair (2,3) is X-prepared and X-measured
(detectors = Z-parts). The 4 detector bits identify the fired two-qubit Pauli uniquely.

Paths: compile_sampler (records), exact detector sampler, twirl (compile_twirl_sampler and
engine="twirl"), port bare route and port record route (sample_barrier), DEM export.
Deterministic components (probability 1) must match Stim EXACTLY on every shot; a channel
with 15 distinct probabilities must match Stim's DEM exactly and Stim's sampled
statistics at z < 5.
"""
from __future__ import annotations

import numpy as np
import pytest

import xtim
from xtim import twirl as _twirl
from xtim.port import Consume, compile as port_compile

stim = pytest.importorskip("stim")

# Stim's documented order, written out literally (independent oracle; NOT the engine's table).
STIM_P2 = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
STIM_P1 = ["X", "Y", "Z"]
_XPART = {"I": 0, "X": 1, "Y": 1, "Z": 0}
_ZPART = {"I": 0, "X": 0, "Y": 1, "Z": 1}


def _p2_text(probs):
    a = ",".join(repr(float(p)) for p in probs)
    return ("R 0 1\nRX 2 3\n"
            f"PAULI_CHANNEL_2({a}) 0 1 2 3\n"
            "M 0 1\nMX 2 3\n"
            "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")


def _p1_text(probs):
    a = ",".join(repr(float(p)) for p in probs)
    return ("R 0\nRX 1\n"
            f"PAULI_CHANNEL_1({a}) 0 1\n"
            "M 0\nMX 1\n"
            "DETECTOR rec[-2]\nDETECTOR rec[-1]\n")


def _expected_p2(word):
    a, b = word
    return [_XPART[a], _XPART[b], _ZPART[a], _ZPART[b]]


def _expected_p1(word):
    return [_XPART[word], _ZPART[word]]


def _unpack(packed, k):
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, bitorder="little")[:, :k].astype(bool)


def _dets_all_paths(text, shots, seed):
    """{path name: bool[shots, D]} for every xtim sampling path."""
    c = xtim.Circuit(text)
    D = c.num_detectors
    out = {}
    meas = np.asarray(c.compile_sampler(seed=seed).sample(shots), dtype=bool)
    # detectors = the records themselves (both preps are deterministic +1 eigenstates)
    out["compile_sampler"] = meas
    out["exact"] = np.asarray(c.compile_detector_sampler(seed=seed, engine="exact").sample(shots), dtype=bool)
    out["twirl_engine"] = np.asarray(c.compile_detector_sampler(seed=seed, engine="twirl").sample(shots), dtype=bool)
    tw = _twirl.compile_twirl_sampler(text, selfcheck=0)
    d, _ = tw.sample(shots, seed=seed)
    out["compile_twirl_sampler"] = _unpack(d, D)
    out["port_bare"] = np.asarray(port_compile(text, Consume(dets=True)).run(shots=shots, seed=seed).dets, dtype=bool)
    rec = port_compile(text, Consume(dets=True, groups=True)).run(shots=shots, seed=seed)
    out["port_record"] = np.asarray(rec.dets, dtype=bool)
    return out


def _dem_map(dem):
    """{frozenset(target strings): probability} of a flattened DEM's error instructions."""
    m = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        key = frozenset(str(t) for t in inst.targets_copy() if not t.is_separator())
        p = inst.args_copy()[0]
        q = m.get(key, 0.0)
        m[key] = q * (1 - p) + (1 - q) * p          # independent mechanisms XOR-combine
    return m


def _assert_dem_equal(text):
    got = _dem_map(xtim.Circuit(text).detector_error_model())
    ref = _dem_map(stim.Circuit(text).detector_error_model())
    assert set(got) == set(ref), (sorted(map(sorted, got)), sorted(map(sorted, ref)))
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-12, abs=1e-15), (k, got[k], ref[k])


# ── deterministic: each component alone at probability 1, exact on every shot ───────────────

@pytest.mark.parametrize("t", range(15), ids=STIM_P2)
def test_pauli_channel_2_component_exact_all_paths(t):
    probs = [0.0] * 15
    probs[t] = 1.0
    text = _p2_text(probs)
    want = np.array(_expected_p2(STIM_P2[t]), dtype=bool)
    ref = stim.Circuit(text).compile_detector_sampler(seed=1).sample(64)
    assert (ref == want).all(), "oracle self-check: stim disagrees with its documented order"
    for path, dets in _dets_all_paths(text, 64, seed=3).items():
        assert dets.shape == (64, 4), path
        bad = ~(dets == want).all(axis=1)
        assert not bad.any(), f"{path}: PAULI_CHANNEL_2 slot {t} ({STIM_P2[t]}) fired {dets[bad][0].astype(int)} != stim {want.astype(int)}"


@pytest.mark.parametrize("t", range(3), ids=STIM_P1)
def test_pauli_channel_1_component_exact_all_paths(t):
    probs = [0.0] * 3
    probs[t] = 1.0
    text = _p1_text(probs)
    want = np.array(_expected_p1(STIM_P1[t]), dtype=bool)
    ref = stim.Circuit(text).compile_detector_sampler(seed=1).sample(64)
    assert (ref == want).all()
    for path, dets in _dets_all_paths(text, 64, seed=3).items():
        assert (dets == want).all(), f"{path}: PAULI_CHANNEL_1 slot {t} ({STIM_P1[t]})"


# ── DEM export: exact equality with Stim's DEM, per component and for a mixed channel ───────

@pytest.mark.parametrize("t", range(15), ids=STIM_P2)
def test_pauli_channel_2_component_dem_matches_stim(t):
    probs = [0.0] * 15
    probs[t] = 0.25
    _assert_dem_equal(_p2_text(probs))


def _single(c, x, z):
    """P(single-qubit Pauli c) for independent X(x) then Z(z) flips."""
    return {(0, 0): (1 - x) * (1 - z), (1, 0): x * (1 - z),
            (1, 1): x * z, (0, 1): (1 - x) * z}[(c in "XY", c in "YZ")]


def test_pauli_channel_dem_mixed_matches_analytic():
    # A mixed channel Stim itself only converts approximately (approximate_disjoint_errors),
    # so the oracle is analytic: PAULI_CHANNEL_2 built, in Stim's documented order, as the
    # product of independent X_a(xa) Z_a(za) X_b(xb) Z_b(zb) flips is EXACTLY four
    # independent single-detector mechanisms (all 15 slots nonzero and distinct).
    xa, za, xb, zb = 0.03, 0.07, 0.11, 0.13
    probs = [_single(w[0], xa, za) * _single(w[1], xb, zb) for w in STIM_P2]
    assert len(set(probs)) == 15
    got = _dem_map(xtim.Circuit(_p2_text(probs)).detector_error_model())
    want = {frozenset({"D0"}): xa, frozenset({"D1"}): xb, frozenset({"D2"}): za, frozenset({"D3"}): zb}
    assert set(got) == set(want), got
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-9), (k, got[k], want[k])
    # PAULI_CHANNEL_1(px, py, pz): Z-read qubit 0 flips on X|Y, X-read qubit 1 flips on Y|Z.
    px, py, pz = 0.05, 0.11, 0.17
    got = _dem_map(xtim.Circuit(_p1_text([px, py, pz])).detector_error_model())
    want = {frozenset({"D0"}): px + py, frozenset({"D1"}): py + pz}
    assert set(got) == set(want), got
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-9), (k, got[k], want[k])


# ── stochastic: 15 distinct probabilities, every path vs Stim sampling (z < 5) ─────────────

def _pattern_freqs(dets, cols):
    """Frequencies of the 4 joint patterns of the detector pair `cols`."""
    idx = dets[:, cols[0]].astype(int) + 2 * dets[:, cols[1]].astype(int)
    return np.bincount(idx, minlength=4) / len(idx)


@pytest.mark.parametrize("kind", ["p2", "p1"])
def test_pauli_channel_mixed_statistics_match_stim(kind):
    S = 200_000
    if kind == "p2":
        text = _p2_text([(t + 1) / 150.0 for t in range(15)])     # sum 0.8, all distinct
        groups = [(0, 1), (2, 3)]
    else:
        text = _p1_text([0.1, 0.25, 0.4])
        groups = [(0, 1)]
    ref = stim.Circuit(text).compile_detector_sampler(seed=11).sample(S)
    for path, dets in _dets_all_paths(text, S, seed=7).items():
        for g in groups:
            fx, fr = _pattern_freqs(dets, g), _pattern_freqs(ref, g)
            se = np.sqrt(fx * (1 - fx) / S + fr * (1 - fr) / S) + 1e-12
            z = float(np.max(np.abs(fx - fr) / se))
            assert z < 5.0, f"{path} {kind} detectors {g}: xtim {fx} vs stim {fr} (z={z:.1f})"
