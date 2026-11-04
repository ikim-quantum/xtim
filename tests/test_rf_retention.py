"""plancache-design §5 F2 (owner-signed rev 4, 2026-11-01): RecordFidelity content retention at the call boundary.

Before: a call whose held bytes exceeded cache_budget flushed EVERYTHING to the rest structure (then rebuilt every
content). Now the boundary pass drops the memo and the conditioned-norm cache, then evicts contents coldest first
(rows served in the last two calls, ties by id) until held <= budget − reserve (the previous call's new-content
bytes; saturating — budget 1 still evicts everything). Gates here:
  * values identical to an unbounded evaluator (every evicted value is a pure function of its key);
  * the pass evicts SOME contents, not all, when the budget admits the hot ones: boundary_evictions > 0 and the
    hottest content of the last two calls survives (its rows cost no kernel compile on the next call) —
    MUT-RF-COLD-TAIL-KEPT (the hot ones evicted first) is RED;
  * held_bytes / cache_flushes / misses / boundary_evictions identical at threads 1 and 4 (M1);
  * review L1: a refused call (max_bytes) commits nothing user-visible — the retention ranking (rows of the last two
    calls), the thread report and every counter but the cached-work ones are as at the call's entry;
  * review L2: the bytes held are a ledger — Σ live contents' charged bytes + the shared charges == held_l3, after
    builds, boundary passes and mid-call drops alike (RecordFidelity._audit_bytes).
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

import xtim

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _sampler(name="magna_color5_fid.stim"):
    p = DATA / name
    if not p.is_file():
        pytest.skip("fixture absent")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(p.read_text(), selfcheck=0)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def test_boundary_pass_keeps_the_hot_contents_and_values():
    s = _sampler()
    A = s.sample_barrier(6000, seed=1)                   # hot rows: many rows per content, plus a singleton tail
    B = s.sample_barrier(1500, seed=2)
    ref = xtim.RecordFidelity(s)
    fa = np.asarray(ref.evaluate(A, keep_memo=True))
    held_a = ref.stats()["held_bytes"]
    fb = ref.evaluate(B, keep_memo=True)
    held_ab = ref.stats()["held_bytes"]
    assert held_ab > held_a > 0
    # the hottest content of A: its rows
    by = {}
    for i in range(6000):
        by.setdefault(A.plan_content(i), []).append(i)
    hot = max(by.values(), key=len)
    assert len(hot) >= 20, len(hot)
    budget = held_a + (held_ab - held_a) // 4
    rf = xtim.RecordFidelity(s, cache_budget=budget)
    assert _bits(rf.evaluate(A, keep_memo=True)) == _bits(fa)
    assert _bits(rf.evaluate(B, keep_memo=True)) == _bits(fb)
    kc = rf.stats()["kernel_compiles"]
    m0 = rf.stats()["misses"]
    # the ranked boundary pass runs before this call; the hottest content survives it, so its rows compile nothing
    # (MUT-RF-COLD-TAIL-KEPT evicts the hot ones first: RED here) — and, speed fix 4, it keeps its memo entries, so its
    # rows (all evaluated before the pass) are memo hits
    assert _bits(rf.evaluate(A, shot_index=hot, keep_memo=True)) == _bits(fa[hot])
    st = rf.stats()
    assert st["cache_flushes"] >= 1 and st["boundary_evictions"] >= 1, st
    assert st["kernel_compiles"] == kc, (st["kernel_compiles"] - kc, len(hot), st)
    assert st["misses"] == m0, (st["misses"] - m0, len(hot), st)
    assert st["boundary_evictions"] < st["plans"], st     # some, not all
    assert _bits(rf.evaluate(A, keep_memo=True)) == _bits(fa)
    assert rf.stats()["held_bytes"] <= held_ab

def test_recycled_slots_serve_exact_values():
    """speed fix 4: evicted slots are recycled (no renumbering) and a kept content keeps its memo entries; a content
    interned into a recycled slot never reads the evicted one's entries (MUT-RF-RECYCLE-KEEPS-MEMO: RED here)."""
    s = _sampler()
    bufs = [s.sample_barrier(3000, seed=20 + k) for k in range(6)]
    ref = xtim.RecordFidelity(s)
    want = [_bits(ref.evaluate(b, keep_memo=True)) for b in bufs]
    h1 = xtim.RecordFidelity(s)
    h1.evaluate(bufs[0], keep_memo=True)
    held1 = h1.stats()["held_bytes"]
    rf = xtim.RecordFidelity(s, cache_budget=held1)
    for rnd in range(3):
        for k, b in enumerate(bufs):
            assert _bits(rf.evaluate(b, keep_memo=True)) == want[k], (rnd, k, rf.stats())
            a, c = rf._audit_bytes()
            assert a == c, (a, c)
            lb, hm, stray = rf._audit_memo()          # every memo entry is a live content's, listed with it
            assert lb == hm and stray == 0, (rnd, k, lb, hm, stray)
    st = rf.stats()
    assert st["boundary_evictions"] >= 1 and st["boundary_evictions"] < st["plans"], st


def test_budget_one_is_todays_flush():
    s = _sampler()
    A = s.sample_barrier(2000, seed=5)
    fresh = xtim.RecordFidelity(s)
    fresh.evaluate(A)
    hA = fresh.stats()["held_bytes"]
    tiny = xtim.RecordFidelity(s, cache_budget=1)
    tiny.evaluate(A)
    tiny.evaluate(A)
    st = tiny.stats()
    assert st["cache_flushes"] >= 1 and st["held_bytes"] == hA, st


def test_retention_counters_thread_identical():
    s = _sampler()
    buf = s.sample_barrier(6000, seed=5)
    h0 = xtim.RecordFidelity(s).stats()["held_bytes"]
    rfs = {T: xtim.RecordFidelity(s, cache_budget=h0 + 4_000_000) for T in (1, 4)}
    for _ in range(3):
        outs = {T: rf.evaluate(buf, keep_memo=True, threads=T) for T, rf in rfs.items()}
        assert _bits(outs[1]) == _bits(outs[4])
        st = {T: rf.stats() for T, rf in rfs.items()}
        for k in ("held_bytes", "cache_flushes", "misses", "boundary_evictions", "midcall_evictions", "plans"):
            assert st[1][k] == st[4][k], (k, st[1][k], st[4][k])


def test_refused_call_commits_nothing():
    s = _sampler()
    A = s.sample_barrier(2000, seed=12)
    h0 = xtim.RecordFidelity(s).stats()["held_bytes"]
    rf = xtim.RecordFidelity(s, max_bytes=h0 + 1)
    before = rf.stats()
    with pytest.raises(RuntimeError, match="above the byte budget"):
        rf.evaluate(A, threads=4)
    after = rf.stats()
    cached_work = {"midcall_evictions", "cache_flushes", "held_bytes", "memo_bytes", "boundary_evictions"}
    diff = {k for k in before if k not in cached_work and before[k] != after[k]}
    assert not diff, {k: (before[k], after[k]) for k in diff}
    assert after["threads"] == before["threads"]
    # the ranking saw no rows: a budgeted evaluator that served A then takes its boundary pass as if A were never
    # evaluated by the refused one — the reserve (the previous call's new-content bytes) is unchanged
    ok = xtim.RecordFidelity(s)
    ok.evaluate(A, keep_memo=True)
    held = ok.stats()["held_bytes"]
    rf2 = xtim.RecordFidelity(s, cache_budget=held + 1_000_000)
    rf2.evaluate(A, keep_memo=True)
    st1 = rf2.stats()
    rf3 = xtim.RecordFidelity(s, cache_budget=held + 1_000_000)
    try:
        rf3.evaluate(A, shot_index=np.arange(2000, 4000), keep_memo=True)   # a failed call (rows out of range)
    except Exception:
        pass
    rf3.evaluate(A, keep_memo=True)
    st3 = rf3.stats()
    for k in ("cache_flushes", "boundary_evictions", "plans", "misses"):
        assert st1[k] == st3[k], (k, st1[k], st3[k])


def test_byte_ledger_is_exact():
    s = _sampler()
    A = s.sample_barrier(6000, seed=1)
    B = s.sample_barrier(1500, seed=2)
    ref = xtim.RecordFidelity(s)
    assert ref._audit_bytes()[0] == ref._audit_bytes()[1]
    ref.evaluate(A, keep_memo=True)
    held_a = ref.stats()["held_bytes"]
    assert ref._audit_bytes()[0] == ref._audit_bytes()[1]
    ref.evaluate(B, keep_memo=True, threads=4)
    held_ab = ref.stats()["held_bytes"]
    assert ref._audit_bytes()[0] == ref._audit_bytes()[1]
    # the boundary pass: the ledger holds after evictions and the compaction
    rf = xtim.RecordFidelity(s, cache_budget=held_a + (held_ab - held_a) // 4)
    for buf in (A, B, A, B):
        rf.evaluate(buf, keep_memo=True)
        a, b = rf._audit_bytes()
        assert a == b, (a, b, rf.stats())
    assert rf.stats()["boundary_evictions"] >= 1
    # the mid-call passes (max_bytes = half the held bytes): the ledger holds after the drops
    half = xtim.RecordFidelity(s, max_bytes=held_a // 2)
    half.evaluate(A)
    a, b = half._audit_bytes()
    assert a == b and half.stats()["midcall_evictions"] >= 1, (a, b, half.stats())


def test_most_must_go_is_one_flush():
    """re-review ruling (b): when most of the droppable bytes must go — the need (here: a call whose rows scale the
    reserve to ~3/4 of the budget), or the bytes served in neither of the last two calls, reaching 3/4 — the boundary
    pass is one clear() (every content goes); a smaller need keeps the signed partial pass (MUT-RF-NO-MOST-FLUSH: RED
    here — the same need evicts part of the cache)."""
    s = _sampler()
    bufs = [s.sample_barrier(400, seed=300 + k) for k in range(3)]
    big = s.sample_barrier(4000, seed=399)
    h = xtim.RecordFidelity(s)
    hs = []
    for b in bufs:
        h.evaluate(b, keep_memo=True)
        hs.append(h.stats()["held_bytes"])
    fresh = xtim.RecordFidelity(s).stats()["held_bytes"]   # the register + the rest structure (never dropped)
    budget = hs[-1] - fresh - 1                             # the first pass: before the fourth call
    per_call = hs[-1] - hs[-2]

    def run(frac):
        rows = int(400 * frac * budget / per_call)          # the reserve scales with the coming call's rows
        rf = xtim.RecordFidelity(s, cache_budget=budget)
        ref = xtim.RecordFidelity(s)
        for b in bufs:
            assert _bits(rf.evaluate(b, keep_memo=True)) == _bits(ref.evaluate(b, keep_memo=True))
        st0 = rf.stats()
        assert st0["boundary_evictions"] == 0, st0
        ix = np.arange(rows)
        assert _bits(rf.evaluate(big, shot_index=ix, keep_memo=True)) == _bits(ref.evaluate(big, shot_index=ix, keep_memo=True))
        st = rf.stats()
        assert st["cache_flushes"] == st0["cache_flushes"] + 1, st
        return st["boundary_evictions"], st0["plans"]

    ev, live = run(0.4)                                     # most must go: one flush
    assert ev == live, (ev, live)
    ev, live = run(0.1)                                     # a small need: the partial pass
    assert 0 < ev < live, (ev, live)
