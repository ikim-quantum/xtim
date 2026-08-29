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
import pytest

import xtim._xtim as _x

_DATA = Path(__file__).resolve().parent / "data"
# born5: five INDEPENDENT heavily-biased (magic-weighted, ~0.1464) Born decisions
# with noisy prefixes -> genuine class-(b) collapses through the memo.
_BORN5 = (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text()

_SEED = 700
_SHOTS = 4000

# Pre-change reference (captured on main @ HEAD before the born fast-path change).
_REF_DEC_H = "039a337cecae014b152c8d12d2f66f9d7d2e14b992bde801354f205682d39658"


def _run(text, shots, seed, sc=0):
    s = _x.TwirlSampler(text, 1.0, "", int(sc), False, "")
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


def test_born_fastpath_taken():
    """The profiler must report born shots served by the abelian fast path (the
    lever is actually exercised, not silently routed slow)."""
    prog = (
        "import numpy as np, xtim._xtim as _x;"
        f"s=_x.TwirlSampler(open({str(_DATA / 'adaptq_born5_dec_enum_producer.stim')!r}).read(),1.0,'',0,False,'');"
        "s.sample(20000,700,None)"
    )
    env = {"QEC_TW_PROF": "1", "PATH": "/usr/bin:/bin"}
    import os
    env2 = dict(os.environ); env2["QEC_TW_PROF"] = "1"
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env2)
    line = [l for l in r.stderr.splitlines() if "PROF born" in l]
    assert line, f"no born prof line in stderr:\n{r.stderr}"
    # fast_born_shots=<N> with N>0 (born5 has noisy diagonal shots that hit the fast path)
    import re
    m = re.search(r"fast_born_shots=(\d+)", line[0])
    assert m and int(m.group(1)) > 0, f"fast path not taken: {line[0]}"


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
