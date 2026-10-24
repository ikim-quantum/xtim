"""S8-2a (plan r4.1 M3): one thread at a time per sampler and per plan cache, as a mechanism. A flag held by another
thread (the test hook sets it as that thread would) makes every guarded entry refuse by name, nothing waits; after
the flag is released the same calls run."""
import os
import threading
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _text():
    return open(os.path.join(DATA, "magna_rp2_fid.stim")).read()


def _fam():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(_text(), selfcheck=0)


def test_sampler_flag_refuses_runs_and_gil_held_entries():
    s = _fam()
    rec = s.sample_records(50, seed=3)[3]
    s._busy_flag("sampler", True)
    for call in (lambda: s.sample(10, seed=1), lambda: s.sample_records(10, seed=1),
                 lambda: s.record_report(), lambda: s.route_stats(), lambda: s.frames(rec),
                 lambda: s.term_rank()):
        with pytest.raises(RuntimeError, match="XT_SAMPLER_BUSY"):
            call()
    s._busy_flag("sampler", False)
    s.sample(10, seed=1)
    s.record_report()


def test_cache_flag_refuses_runs_rf_evaluate_and_the_handle():
    s = _fam()
    buf = s.sample_barrier(50, seed=3)
    rf = s.record_fidelity()
    h = s._cache_handle()
    s._busy_flag("cache", True)
    with pytest.raises(RuntimeError, match="XT_CACHE_BUSY"):
        s.sample(10, seed=1)
    with pytest.raises(RuntimeError, match="XT_CACHE_BUSY"):
        rf.evaluate(buf)
    with pytest.raises(RuntimeError, match="XT_CACHE_BUSY"):
        h.resident_bytes()
    with pytest.raises(RuntimeError, match="XT_CACHE_BUSY"):
        h.evict_all()
    s._busy_flag("cache", False)
    assert np.all(np.isfinite(np.asarray(rf.evaluate(buf))))
    assert h.resident_bytes() >= 0


def test_flags_released_after_every_call():
    """Every entry releases its flags on return (BUSY-LEAK). (A run that throws inside the flag scope: the only in-scope
    throw found is the negative-seed defect, s8/fixplan-negative-seed.md, not patched; the release is RAII.)"""
    s = _fam()
    a = s.sample(20, seed=5)
    b = s.sample(20, seed=5)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    buf = s.sample_barrier(20, seed=5)
    rf = s.record_fidelity()
    rf.evaluate(buf)
    rf.evaluate(buf)
    s.record_report()
    s.record_report()
    s.sample(20, seed=5)


def test_production_sampler_refuses_a_concurrent_run():
    """Two threads on one production sampler: the second run refuses by name while the first holds the flags (the
    first run releases the GIL); both threads end and the sampler is usable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(_text(), selfcheck=0)
    errs, done = [], []

    def run():
        try:
            s.sample(200000, seed=11)
            done.append(1)
        except RuntimeError as e:
            errs.append(str(e))
    for _ in range(3):
        errs.clear(); done.clear()
        ts = [threading.Thread(target=run) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert len(done) >= 1 and all(("XT_SAMPLER_BUSY" in e or "XT_CACHE_BUSY" in e) for e in errs)
        if errs:
            break
    s.sample(10, seed=1)


def test_negative_seed_raises_and_the_sampler_stays_usable():
    """OD-2026-10-23-s8-seed: a seed that fails the uint64 cast raises (it used to crash the process: the cast ran
    with the GIL released); the flags are released and the sampler samples the same bytes afterwards."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(_text(), selfcheck=0)
    a = s.sample(20, seed=5)
    with pytest.raises((TypeError, ValueError, OverflowError, RuntimeError)):
        s.sample(20, seed=-1)
    b = s.sample(20, seed=5)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def _race(run_a, run_b):
    """Thread A runs run_a() (its run releases the GIL); thread B calls run_b() repeatedly while A is alive. Returns
    (A's result or its error, B's error messages, B's successful calls)."""
    res, errs, oks = {}, [], []

    def a():
        try:
            res["a"] = run_a()
        except Exception as e:                     # noqa: BLE001 — reported by the assertion below
            res["a"] = e
    t = threading.Thread(target=a)
    t.start()
    while t.is_alive():
        try:
            run_b()
            oks.append(1)
        except RuntimeError as e:
            errs.append(str(e))
    t.join()
    return res["a"], errs, oks


def _same(x, y):
    return all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(x, y))


def test_concurrent_sample_barrier_refused_before_touching_the_running_call():
    """Final review M1: the flags cover the GIL-held setup of sample_barrier (the record sink). A second
    sample_barrier on the same sampler while one runs is refused by name (XT_SAMPLER_BUSY) and the running call's bytes
    (detectors, observables, records) equal a solo run. MUT-BUSY-RUNCORE (-DXTIM_MUTATE_BUSY_RUNCORE: the flags back in
    run_core_) turns this red (the refused call swaps / nulls the running call's record sink: heap corruption)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(_text(), selfcheck=0)
    N = 100000

    def view(b):
        return (b.dets(), b.obs(), b.record_keys()[:5000])
    ref = view(s.sample_barrier(N, seed=11))
    a, errs, oks = _race(lambda: view(s.sample_barrier(N, seed=11)), lambda: s.sample_barrier(10, seed=1))
    assert not isinstance(a, Exception), a
    assert errs and all("XT_SAMPLER_BUSY" in e for e in errs), errs[:3]
    assert ref[2] == a[2] and _same(ref[:2], a[:2])
    s.sample_barrier(10, seed=1)


def test_concurrent_input_branch_sample_refused_before_touching_the_running_call():
    """Final review M1, the sample(input_branch=) half: a refused concurrent call must not reassign eng_ib_ or reset
    the engine's input branch under the running call. The running call's bytes equal a solo run."""
    from xtim import _xtim
    # tests/test_jb1_branch_law_e0.py "port_delta": a producer whose branches change the consumer's bytes
    prod = ("RX 0 1\nR 2\nX_ERROR(0.2) 1\nS 1\nH 0\nH 0\nCX 0 1\nS 0\nT 1\nCZ 0 2\nCX 1 2\nM 2\n"
            "DETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1\n")
    cons = "INPUT_QUBITS in 0 1\nH 1\nH 1\nS 1\nH 1\nT 0\nMX 0\nDETECTOR rec[-1]\nM 1\nDETECTOR rec[-1]\n"
    p = _xtim.TwirlSampler(prod, 1.0, "", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = _xtim.TwirlSampler.from_ideal_state(ideal_state=p.ideal_state(), circuit_text=cons, selfcheck=0)
    R = p.sample_barrier(300, seed=1)
    H, fr = p.branches(R)
    ids = {}
    ib = np.array([ids.setdefault((bytes(h["digest"]), bytes(h["z"])), c.add_branch(h)) for h in H], np.int32)
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
    assert (ib != 0).any()
    reps = 4000
    IB, IP = np.tile(ib, reps), np.tile(ip, (reps, 1))
    n = len(IB)
    ref = c.sample(n, seed=5, input_pauli=IP, input_branch=IB)
    assert not _same(ref, c.sample(n, seed=5, input_pauli=IP, input_branch=np.zeros(n, np.int32)))   # non-vacuous
    a, errs, oks = _race(lambda: c.sample(n, seed=5, input_pauli=IP, input_branch=IB),
                         lambda: c.sample(300, seed=1, input_pauli=ip, input_branch=np.zeros(300, np.int32)))
    assert not isinstance(a, Exception), a
    assert errs and all("XT_SAMPLER_BUSY" in e for e in errs), errs[:3]
    assert _same(ref, a)
    c.sample(300, seed=1, input_pauli=ip, input_branch=ib)


def test_memo_hit_does_not_skip_the_undressed_record_refusal():
    """Final review m1: the undressed-R_s impossible-record refusal runs before the memo lookup. A tampered row (one
    stored record value flipped) that shares its memo key with an earlier valid row is refused exactly as with the
    memo bypassed (the row log on): for every record column, memo on and memo off agree (refusal or value), and some
    flip is refused. The fixture is tests/test_rfref_route.py's _STAB (route 3a: the key carries only the used records'
    signs). MUT-RF-UNDRESSED-AFTER-MEMO (-DXTIM_MUTATE_RF_UNDRESSED_AFTER_MEMO) turns this red."""
    import collections
    stab = ("R 0\nDEPOLARIZE1(0.3) 0\nRX 1 2\nT 1\nCZ 1 2\nT_DAG 2\nX_ERROR(0.1) 1\nMX 1\nDECISION(0) rec[-1]\nMX 2\n"
            "DECISION(1) rec[-1]\nR 3\nX_ERROR(0.2) 3\nM 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = TA.compile_family_sampler(stab, selfcheck=0)
    N = 200
    buf = s.sample_barrier(N, seed=3)
    rec = s.sample_records(N, seed=3)[3]
    groups = collections.defaultdict(list)
    for i in range(rec.num_records):
        groups[bytes(rec.key(i))].append(i)
    js = [v[1] for v in groups.values() if len(v) >= 2][:3]   # rows whose memo key an earlier row already filled
    assert js
    m = len(rec.y(0))

    def ev(memo_off):
        rf = s.record_fidelity()
        if memo_off:
            TA._rf_row_log(rf, True)
        try:
            return np.asarray(rf.evaluate(buf))
        except RuntimeError as e:
            assert "impossible record" in str(e), e
            return "refused"
    refused = 0
    for j in js:
        for c in range(m):
            TA._flip_record_value(buf, j, c)
            on, off = ev(False), ev(True)
            TA._flip_record_value(buf, j, c)
            if isinstance(off, str):
                refused += 1
                assert isinstance(on, str), ("a memo hit served a refused row", j, c)
            else:
                assert not isinstance(on, str) and np.array_equal(on, off), (j, c)
    assert refused > 0
