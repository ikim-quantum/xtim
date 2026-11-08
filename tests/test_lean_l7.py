"""lean L7 (byte-identical): two touched-set rewrites on the sampling hot path.

1. The early-reject C-support mask (er_diag_reject_) is the union of the flagged-channel rows of every wire in
   supp(a) ∪ the endpoints of the key's CZ pairs. It ORed two rows per key PAIR (1.9–3.7 k pairs per shot on the full d5
   T circuit, ~21–140 distinct wires); now each DISTINCT wire's row is ORed once (OR is idempotent: the same bits).
   Pinned here: the early / passed counts of an all-detector mask on the corpus (a mask that drops an endpoint
   rejects more shots, a mask that over-covers rejects fewer) and the masked call == the unmasked call filtered.
2. The readout coins (rf_coins, RFW words) are touched only in the words draw_rf fired: draw_rf clears the previous
   shot's fired words and records this shot's; apply_rf / er_rf_into_ / er_clean_ walk the fired words only. Every
   full-array reader (the family extras' emit, the Born decisions) still sees the exact coins. Pinned here on circuits
   with RFW = 6: every shot's detectors == the pure-Python reference of the RF stream (test_lean_l5's), sparse firing
   (stale words would leak), the masked call on the clean-shot path, and the bytes of a T tail whose extras and Born
   observable read high slots (digest == the L6 tip, 3c29512).
Test-of-test (lean/L7/REPORT.md): an endpoint dropped / one fired word skipped / a stale word left → red.
"""
import hashlib
import os

import numpy as np
import pytest

import xtim
from xtim.twirl import TwirlBatch
from test_lean_l5 import rf_groups, rf_reference

pytestmark = pytest.mark.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _bits(a, k):
    return np.unpackbits(np.asarray(a, np.uint8), axis=1, bitorder="little")[:, :k]


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        if not isinstance(a, dict):
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


# ── 1. the early-reject mask ────────────────────────────────────────────────────────────────────────────────────────
# (circuit, shots, seed, mask: "all" detectors | "even" detectors) -> (passed, early, digest of the accepted rows), values of
# the L6 tip (3c29512). RE-PINNED 2026-11-06 (lean D1-S2, owner-signed stream change; every key is a T circuit on the rotation
# loop): the accepted SET moves with the coin / Born gauges (each moved attempt has a flagged column with A.g = 1 or B != 0:
# lean/D1/S2/equivalence.md) and `early` is the loop's column read over the flagged channels no set rotation touches
ER_PINS = {   # RE-PINNED 2026-11-07 (lean N1, then lean N2: owner-signed NOISE-stream changes; previous values: git history)
    ('cultivation_d3_faithful.stim', 4000, 11, 'all'): (2685, 1315, "61e5b6d176d4f9ba"),
    ('cultivation_d5.stim', 2000, 11, 'all'): (279, 1721, "8666d5ba3799c48a"),
    ('magna_color5_fid.stim', 2000, 11, 'all'): (279, 1721, "0b2a1e5782c2c427"),
    ('magna_rp2_fid.stim', 2000, 11, 'all'): (199, 1800, "4add9b990a3f15ad"),
    ('distillation_15_1_3.stim', 4000, 11, 'all'): (3935, 65, "de5bfc838bdc1f1c"),
    ('code_switching_faithful.stim', 4000, 11, 'all'): (3343, 651, "62242fecd291686f"),
    ('adaptq_qrm_magic_producer.stim', 4000, 11, 'all'): (3571, 429, "19b7a0d21aa16c57"),
    ('miniature_oracle.stim', 4000, 11, 'all'): (3956, 43, "b98afbc529ebbd35"),
    ('cultivation_d5.stim', 3000, 12, 'even'): (663, 2337, "8ad3ead70652a0f0"),
    ('magna_color5_fid.stim', 3000, 12, 'even'): (684, 2316, "5b328698f0e12a26"),
    ('magna_rp2_fid.stim', 3000, 12, 'even'): (357, 2585, "139b640134df2ac5"),
    ('cultivation_d3_faithful.stim', 4000, 12, 'even'): (2898, 998, "3c6f261b01ededc0"),
}


@pytest.mark.parametrize("key", sorted(ER_PINS))
def test_early_reject_counts_pinned(key):
    """The early count is exactly the shots on which a flagged C-untouched channel fires: it pins the C-support mask in both
    directions (a missing wire rejects more, an extra wire fewer); the accepted rows equal the unmasked call filtered."""
    name, shots, seed, which = key
    text = open(os.path.join(DATA, name)).read()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    mask = list(range(nd)) if which == "all" else list(range(0, nd, 2))
    out = s.sample(shots, seed=seed, postselect=mask)
    c = out[-1]
    full = xtim.compile_twirl_sampler(text, selfcheck=0).sample(shots, seed=seed)
    if isinstance(full, TwirlBatch):           # a decision circuit's wrapper: (batch, counts)
        full = (full.dets, full.obs, full.decisions)
        out = (out[0].dets, out[0].obs, out[0].decisions, c)
    passed, early, dig = ER_PINS[key]
    assert (c["attempts"], c["passed"], c["early"]) == (shots, passed, early)
    assert _h(out[:2]) == dig
    keep = ~_bits(full[0], nd)[:, mask].astype(bool).any(1)
    for a, b in zip(full, out[:-1]):
        assert np.array_equal(np.asarray(a)[keep], np.asarray(b))


# ── 2. the readout coins on RFW = 6 words ───────────────────────────────────────────────────────────────────────────
# 330 flip-only slots (6 words): six p groups interleaved, every 7th slot p = 1e-4 (sparse words)
RF_PS = [([0.3, 1e-3, 0.02, 1.0, 2e-4, 0.5][i % 6] if i % 7 else 1e-4) for i in range(330)]
RF_TXT = "".join(f"R 0\nM({p!r}) 0\nDETECTOR rec[-1]\n" for p in RF_PS)
# + a T tail: a gauge detector on an M(p) record after T (a family extra reading slot 330), a feedback CX on it, and a
# Born observable on an MX(p) record (rf_obs_bit) — the readers of the high word
RF_TAIL = ("RX 1\nT 1\nX_ERROR(0.1) 2\nM 2\nMX(0.1) 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCX rec[-1] 3\nM(0.05) 3\n"
           "DETECTOR rec[-1]\nRX 4\nT 4\nMX(0.2) 4\nOBSERVABLE_INCLUDE(0) rec[-1]\n")


@pytest.mark.parametrize("p_factor", [1.0, 0.01])
def test_rf_multiword_is_the_reference_stream(p_factor):
    """Every shot's 330 detectors == the reference's fired slots (seeded, then continued); at p_factor 0.01 most words
    are empty most shots, so a word left set from an earlier shot or a fired word skipped is visible."""
    probs = [p * p_factor for p in RF_PS]
    groups = rf_groups(probs)
    s = xtim.compile_twirl_sampler(RF_TXT, p_factor=p_factor, selfcheck=0)
    seed, N = 77, 3000
    d = _bits(s.sample(N, seed=seed)[0], len(RF_PS))
    d2 = _bits(s.sample(N, seed=None)[0], len(RF_PS))
    for blk, off in ((d, 0), (d2, N)):
        for i in range(N):
            want = np.zeros(len(RF_PS), np.uint8)
            want[rf_reference(seed, off + i, groups)] = 1
            assert np.array_equal(blk[i], want), off + i
    assert d[:, 300:].any()                    # the high words fire


@pytest.mark.parametrize("p_factor", [1.0, 0.01])
def test_rf_multiword_postselect_clean_path(p_factor):
    """A mask on high-word detectors (the clean-shot early test reads the fired words): the accepted rows are the
    reference's accepted shots, every rejection is early (flip-only: K = 0, every flagged detector is an early channel)."""
    probs = [p * p_factor for p in RF_PS]
    groups = rf_groups(probs)
    mask = [i for i in range(len(RF_PS)) if i >= 200 and RF_PS[i] < 0.5]
    seed, N = 5, 3000
    s = xtim.compile_twirl_sampler(RF_TXT, p_factor=p_factor, selfcheck=0)
    out = s.sample(N, seed=seed, postselect=mask)
    ref = np.zeros((N, len(RF_PS)), np.uint8)
    for i in range(N):
        ref[i, rf_reference(seed, i, groups)] = 1
    keep = ~ref[:, mask].astype(bool).any(1)
    assert np.array_equal(_bits(out[0], len(RF_PS)), ref[keep])
    c = out[-1]
    assert c["passed"] == int(keep.sum()) and c["early"] == N - int(keep.sum())
    assert 0 < c["passed"] < N


# digest of (dets, obs) of RF_TXT + RF_TAIL, 4000 shots seed 3, values of the L6 tip; masked: all detectors of the tail
# + every 5th flip slot with p < 0.3 (the gauge detectors take the late filter)
RF_TAIL_PINS = {1.0: ("2942372cb87132f1", ("2a1a3a8d46e37eb7", 2132, 1075)),   # RE-PINNED lean N1, N2 (owner-signed)
                0.01: ("1b55a59043ccaf85", ("f6cd1e8d7431328b", 3382, 17))}


def _tail_digests(p_factor):
    text = RF_TXT + RF_TAIL
    s = xtim.compile_twirl_sampler(text, p_factor=p_factor, selfcheck=0)
    a = _h(s.sample(4000, seed=3))
    nd = s.num_detectors
    mask = [i for i in range(0, len(RF_PS), 5) if RF_PS[i] < 0.3] + list(range(len(RF_PS), nd))
    m = xtim.compile_twirl_sampler(text, p_factor=p_factor, selfcheck=0).sample(4000, seed=3, postselect=mask)
    c = m[-1]
    return a, (_h(m[:-1]), c["passed"], c["early"])


@pytest.mark.parametrize("p_factor", [1.0, 0.01])
def test_rf_tail_readers_bytes_pinned(p_factor):
    """The full-array readers (the family extra's emit on slot 330, the Born observable's flip) and the masked call on
    the diagonal routes (er_rf_into_ inside er_diag_reject_): bytes == the L6 tip."""
    assert _tail_digests(p_factor) == RF_TAIL_PINS[p_factor]
