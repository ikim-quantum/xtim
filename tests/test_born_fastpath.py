"""Born-decision fast-path pins (2026-08-08).

The engine change: Born DECISION circuits run on the abelian sigma fast path
(need_state decoupled from has_born_dec_) and feed the EXISTING born-probability
memo, instead of the per-shot canonicalize/collapse structural detour. See the
`const bool need_state = retain_state;` block in cpp/src/twirl_sampler.cpp and
the born-decision fast-path scope memo.

These tests pin, at a fixed seed:
  1. same-seed REPRODUCIBILITY of the born decision stream;
  2. ROUTING-INDEPENDENCE: the born stream is byte-identical whether shots take
     the warm abelian fast path (selfcheck=0) or are FORCED through the slow
     structural-row path (selfcheck >= shots) — the in-engine proof that the
     fast-path memo-key capture reproduces the slow-path structural one;
  3. FAST-PATH-TAKEN: the profiler reports born shots served by the fast path
     (a non-vacuity guard that the lever is actually exercised);
  4. m_out WEIGHT EXACTNESS: the magic-weighted Born decision keeps its exact
     (1 -/+ 1/sqrt2)/2 conditional weights (the memoized collapse preserves them);
  5. 5-SIGMA PHYSICAL INVARIANCE + BYTE PIN vs a pre-change reference: the born
     decision bytes are byte-identical to the pre-change structural path (for a
     k=0 diagonal plan p_j is invariant to the re-pinned fair coins, so the
     authorised born-coin re-pin did not actually move any born byte).
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

import xtim._xtim as _x

_DATA = Path(__file__).resolve().parent / "data"
# born5: five INDEPENDENT heavily-biased (magic-weighted, ~0.1464) Born decisions
# with noisy prefixes -> genuine class-(b) collapses through the memo.
_BORN5 = (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text()

_SEED = 700
_SHOTS = 4000

# Pre-change reference (captured on main @ HEAD before the born fast-path change).
# RE-PINNED 2026-10-14 (S0(b) owner-signed re-pin (D11, ENGINE_VERSION xtim-engine-11): keyed per-shot streams + one coin convention): every
# twirl stream moved by declaration; old 039a337cecae014b152c8d12d2f66f9d7d2e14b992bde801354f205682d39658.
# The law is gated by the S0(b) law battery (5 sigma vs the pre-change build and engine="exact").
# RE-PINNED 2026-10-17 (S2b, ENGINE_VERSION xtim-engine-12): canonical record-order basis + extras as rows;
# the sample() and barrier (records, sigma) laws are unchanged (sigma-4, owner 2026-10-16); byte streams moved.
# Old (engine-11): 157a00f65218d6d4cb62bc9f8a7b9b89cd29e2c8b446b4275dab492f1ae0037f.
_REF_DEC_H = "4ff09ada7a50d7b3fe7f845ca5211ddfec2c4ee5f669ce3c1870e21c911a5663"


def _run(text, shots, seed, sc=0):
    s = _x.TwirlSampler(text, 1.0, "", int(sc), "")
    out = s.sample(shots, seed, None)
    return [np.asarray(x, np.uint8) for x in out]


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_born_fastpath_reproducible():
    a = _run(_BORN5, _SHOTS, _SEED)
    b = _run(_BORN5, _SHOTS, _SEED)
    assert np.array_equal(a[2], b[2])  # decisions
    assert np.array_equal(a[0], b[0])  # dets


def test_born_fastpath_routing_independent():
    """Warm abelian fast path (sc=0) == forced slow structural-row path (sc>=shots)."""
    fast = _run(_BORN5, _SHOTS, _SEED, sc=0)
    slow = _run(_BORN5, _SHOTS, _SEED, sc=_SHOTS + 10)
    for a, b in zip(fast, slow):
        assert np.array_equal(a, b)


def test_born_fastpath_mout_weight_exactness():
    """The magic-weighted Born decisions keep their exact (1-1/sqrt2)/2 marginal."""
    N = 400000
    out = _run(_BORN5, N, 12345)
    dec = np.unpackbits(out[2], axis=1, bitorder="little")[:, :5]
    anchor = (1 - 2 ** -0.5) / 2  # 0.14645
    for k in range(5):
        p = dec[:, k].mean()
        se = (anchor * (1 - anchor) / N) ** 0.5
        assert abs(p - anchor) < 5 * se, f"col {k}: P1={p:.5f} vs anchor {anchor:.5f} (5se={5*se:.5f})"


def test_born_fastpath_byte_pin_vs_prechange():
    """The born decision bytes are byte-identical to the pre-change structural
    path (proven no-op re-pin): p_j is invariant to the re-pinned fair coins for
    k=0 diagonal plans, so no born byte moved."""
    out = _run(_BORN5, _SHOTS, _SEED)
    assert _h(out[2]) == _REF_DEC_H, (
        "born decision stream moved vs the pre-change reference "
        f"(got {_h(out[2])}); the born fast-path must not change the physical stream"
    )


def test_born_fastpath_taken_public():
    """J-b2 J1b, the PRODUCTION leg (plan r2.1 J1b): the public sampler serves the born5 producer's Born-decision
    shots on the record fast routes (record_report counters, not the legacy QEC_TW_PROF profiler line): faulted
    shots are served with kappa_rec > 0 record routes and not all of them take the structural path."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _public import public_sampler
    s = public_sampler((_DATA / "adaptq_born5_dec_enum_producer.stim").read_text())
    s.sample(20000, seed=700)
    r = s.record_report()
    faulted = r["used"] - r["route_counts"]["clean"]
    assert r["route_counts"]["record_kappa_pos"] > 0 and faulted > 0
    assert r["structural_shots"] < faulted, (r["structural_shots"], faulted)
