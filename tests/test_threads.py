"""threads (fixplan threads/fixplan.md): sampling with ``threads=`` keeps one copy of the sampler's construction
state and gives today's bytes for every thread count.

* T0: the retained-heap inventory (``_footprint``, test API) — the fixplan §1 table as a test.
* T2 / G1: ``sample(..., threads=T)`` equals ``threads=1`` byte for byte (dets / obs / decisions) and in the
  shot-determined counters (used, clean, the route sum, channel_one_counts), for T in {2, 7, 48}, fresh and warm
  samplers, seeded and seed=None continuations, block-boundary shot counts, with and without the selfcheck window.
  A run the lanes do not serve runs serially and record_report()['threads'] names why (owner ruling D3: reported).
"""
import glob
import os

import numpy as np
import pytest

import xtim
from xtim import _xtim as X
from xtim import _xtim_testapi as TA

from _private import PRIVATE_DIR

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D5 = os.path.join(HERE, "data", "cultivation_d5.stim")
D3R = os.path.join(HERE, "data", "cultivation_d3_rate.stim")
FAC_A = os.path.join(str(PRIVATE_DIR), "ccz_factory_a.stim")

SHARED_GROUPS = ("table", "G", "G_chunk", "gens", "sides", "rotation", "alt_is_ppr",   # lean D1-S3: the alt substrate deleted
                 "sch", "rec_cols", "qcol", "prefix_rows", "readout_flip", "noise")   # lean N1: the mechanism tables


def test_footprint_inventory():
    """T0: every §1 member group is reported; the construction state dominates the per-shot scratch."""
    s = TA.compile_family_sampler(open(D5).read(), selfcheck=0)
    s.sample(2000, 1)
    fp = s._footprint()
    for k in SHARED_GROUPS + ("shot_scratch", "shot_block", "plan_headers", "plan_payloads", "born_root_memo", "sem_work"):
        assert k in fp, (k, sorted(fp))
    shared = sum(fp[k] for k in SHARED_GROUPS)
    assert shared > 20 * fp["shot_scratch"] > 0, fp


# ── G1: byte identity ──────────────────────────────────────────────────────────────────────────────────────────────
CORPUS = sorted(glob.glob(os.path.join(HERE, "data", "*.stim")) +
                [os.path.join(HERE, "..", "examples", "color5_cultivation.stim")] +
                [os.path.join(str(PRIVATE_DIR), n) for n in ("ccz_factory_b.stim", "ccz_factory_c.stim",
                                                             "ccz_factory_d.stim", "ccz_factory_a_out.stim")
                 if os.path.isfile(os.path.join(str(PRIVATE_DIR), n))])


def _sampler(text, selfcheck=0):
    return X.TwirlSampler(text, selfcheck=selfcheck)


def _counters(s):
    r = s.record_report()
    rc = r["route_counts"]
    pm = r.get("plan_memory", {})
    # scaling S4: the plan cache holds each distinct key once at every T (lane overlays adopted, duplicates dropped);
    # its index grows as the serial inserts grow it. (Resident payload bytes are not invariant: Born-trie capacities.)
    return {"used": r["used"], "clean": rc["clean"], "routes": sum(rc.values()),
            "chan": list(s.channel_one_counts()), "plans": r.get("plans"), "header_bytes": pm.get("header_bytes")}


def _calls(s, plan, threads):
    """plan: [(shots, seed)] — seed None continues the stream."""
    out = []
    for shots, seed in plan:
        t = s.sample(shots, seed, threads=threads)
        out.append(tuple(np.asarray(a).copy() for a in t))
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            assert u.dtype == v.dtype and u.shape == v.shape and np.array_equal(u, v)


def _check(text, plan, threads_list, selfcheck=0):
    s1 = _sampler(text, selfcheck)
    ref = _calls(s1, plan, 1)
    c1 = _counters(s1)
    for T in threads_list:
        sT = _sampler(text, selfcheck)
        got = _calls(sT, plan, T)
        _same(ref, got)
        assert _counters(sT) == c1, T
        if selfcheck:
            assert sT.record_report()["semantic_oracle_shots"] == s1.record_report()["semantic_oracle_shots"]


@pytest.mark.parametrize("path", CORPUS, ids=[os.path.basename(p) for p in CORPUS])
def test_g1_corpus(path):
    """Every corpus circuit: threads 2 and 7 == threads 1, cold, a seeded call then two seed=None continuations."""
    text = open(path).read()
    try:
        _sampler(text)
    except Exception as e:  # a circuit outside the twirl class: nothing to sample
        pytest.skip(f"not a twirl circuit: {type(e).__name__}")
    _check(text, [(4097, 11), (1, None), (5000, None)], [2, 7])


@pytest.mark.parametrize("shots", [1, 4095, 4097, 100000])
def test_g1_d5_shot_counts(shots):
    _check(open(D5).read(), [(shots, 3)], [2, 7, 48])


def test_g1_d5_window_and_continuation():
    """The selfcheck window runs serially (the same shots are checked), the rest on lanes; seed=None continues."""
    _check(open(D5).read(), [(30000, 5), (4097, None), (20000, None), (7, 9)], [2, 7, 48], selfcheck=2000)


def test_g1_superblock_shapes():
    """scaling S3: the block ramp kept across calls, super-blocks of up to 16 blocks per lane, a short remainder folded
    into the last super-block; S1: the next super-block's noise drawn while the lanes run (many barriers per call)."""
    _check(open(D5).read(), [(150000, 3), (70001, None), (1, None), (250000, None), (4097, 8)], [2, 7, 48])


def test_g1_d3_rate_warm():
    _check(open(D3R).read(), [(20000, 1), (20000, 2), (60000, None)], [2, 7, 48], selfcheck=500)


def _fac_a():
    if not os.path.isfile(FAC_A):
        pytest.skip("private fixture ccz_factory_a.stim not present")
    return open(FAC_A).read()


def test_g1_factory_a():
    _check(_fac_a(), [(4097, 1), (30000, None)], [2, 7, 48])


def test_g1_factory_a_window():
    _check(_fac_a(), [(20000, 7)], [7], selfcheck=2000)


# ── threads_effective (owner ruling D3: reported, never silent) ─────────────────────────────────────────────────────
def test_threads_reported():
    s = _sampler(open(D5).read())
    s.sample(10000, 1, threads=4)
    th = s.record_report()["threads"]
    assert th == {"requested": 4, "effective": 4, "limit": "", "deferred_shots": 0, "budget_fallbacks": 0}, th
    s.sample(10, 1)
    assert s.record_report()["threads"] == {"requested": 1, "effective": 1, "limit": "", "deferred_shots": 0,
                                            "budget_fallbacks": 0}


A1 = "INPUT_QUBITS in 0\nR 1\nX 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n"


def test_threads_serial_runs_are_named():
    """A run the lanes do not serve (here oracle_all_shots: serial by owner ruling D2) runs serially, with the reason."""
    text = open(D3R).read()
    s = TA.compile_family_sampler(text, selfcheck=0, oracle_all_shots=True)
    a = s.sample(3000, 2, threads=4)
    th = s.record_report()["threads"]
    assert th["requested"] == 4 and th["effective"] == 1 and "oracle_all_shots" in th["limit"], th
    b = TA.compile_family_sampler(text, selfcheck=0, oracle_all_shots=True).sample(3000, 2)
    _same([a], [b])


def test_threads_term_route_on_lanes():
    """T4: a CH/PPR circuit runs on lanes. RE-PINNED lean D1-S3 (owner-signed): its shots (PPR-retry ones included) are
    served by the rotation loop on the lanes — none is deferred to the home; bytes == threads 1 (_check)."""
    for name in ("ch_cultivation.stim", "ja_seed23.stim"):
        text = open(os.path.join(HERE, "data", name)).read()
        s = _sampler(text)
        s.sample(3000, 2, threads=4)
        th = s.record_report()["threads"]
        assert th["effective"] == 4 and th["limit"] == "" and th["deferred_shots"] == 0, (name, th)
        _check(text, [(9000, 2), (4097, None)], [2, 7, 48])


def test_threads_argument():
    s = _sampler(open(D3R).read())
    for bad in (0, -1, True, False, 1.5, "all", None):
        with pytest.raises((ValueError, TypeError)):
            s.sample(10, 1, threads=bad)
    s.sample(10, 1, threads="auto")
    assert s.record_report()["threads"]["requested"] == len(os.sched_getaffinity(0))


# ── T3: postselect (early-reject lanes; accepted rows compacted in shot order) ─────────────────────────────────────
def _ps_calls(s, plan, mask, threads):
    out = []
    for shots, seed in plan:
        t = s.sample(shots, seed, postselect=mask, threads=threads)
        out.append((tuple(np.asarray(a).copy() for a in t[:-1]), dict(t[-1])))
    return out


def _check_ps(text, plan, mask_of, threads_list, selfcheck=0):
    s1 = _sampler(text, selfcheck)
    mask = mask_of(s1)
    ref = _ps_calls(s1, plan, mask, 1)
    c1 = _counters(s1)
    for T in threads_list:
        sT = _sampler(text, selfcheck)
        got = _ps_calls(sT, plan, mask, T)
        assert len(got) == len(ref)
        for (ga, gc), (ra, rc) in zip(got, ref):
            _same([ga], [ra])
            assert gc == rc, (T, gc, rc)
        assert _counters(sT) == c1, T
        assert sT.record_report()["threads"]["effective"] == T


def _every_third(s):
    return np.array([d % 3 == 0 for d in range(s.num_detectors)], dtype=bool)


def test_g1_postselect_d5():
    _check_ps(open(D5).read(), [(30000, 4), (4097, None), (1, None)], _every_third, [2, 7, 48])


def test_g1_postselect_d5_window():
    _check_ps(open(D5).read(), [(20000, 4)], _every_third, [7], selfcheck=1500)


def test_g1_postselect_factory_a():
    _check_ps(_fac_a(), [(20000, 6), (5000, None)], lambda s: np.ones(s.num_detectors, dtype=bool), [2, 48])


@pytest.mark.parametrize("path", [p for p in CORPUS if "producer" in p or "code_switching" in p],
                         ids=lambda p: os.path.basename(p))
def test_g1_postselect_corpus(path):
    text = open(path).read()
    s = _sampler(text)
    if s.num_detectors == 0:
        pytest.skip("no detectors")
    _check_ps(text, [(6000, 2), (3000, None)], lambda s: np.ones(s.num_detectors, dtype=bool), [3])


# ── T4: recording runs (sample_barrier) on lanes: records compared by content ───────────────────────────────────────
def _barrier(s, shots, seed, threads, mask=None):
    kw = {} if mask is None else {"postselect": mask}
    b = s.sample_barrier(shots, seed, threads=threads, **kw)
    n = len(np.asarray(b.dets()))
    keys = list(b.record_keys())
    outs = [np.asarray(b.outcomes(i)).copy() for i in range(n)]
    fr = b.frames()
    frames = {k: np.asarray(v).copy() for k, v in fr.items() if hasattr(v, "shape") or isinstance(v, np.ndarray)}
    return (np.asarray(b.dets()).copy(), np.asarray(b.obs()).copy(), np.asarray(b.decisions()).copy(), keys, outs,
            frames, b)


def _same_barrier(x, y):
    for u, v in zip(x[:3], y[:3]):
        assert np.array_equal(u, v)
    assert x[3] == y[3]
    assert len(x[4]) == len(y[4]) and all(np.array_equal(u, v) for u, v in zip(x[4], y[4]))
    assert sorted(x[5]) == sorted(y[5])
    for k in x[5]:
        assert np.array_equal(x[5][k], y[5][k]), k


def _check_barrier(text, plan, threads_list, mask_of=None, selfcheck=0):
    s1 = _sampler(text, selfcheck)
    mask = mask_of(s1) if mask_of else None
    ref = [_barrier(s1, n, sd, 1, mask) for n, sd in plan]
    c1 = _counters(s1)
    for T in threads_list:
        sT = _sampler(text, selfcheck)
        got = [_barrier(sT, n, sd, T, mask) for n, sd in plan]
        for g, r in zip(got, ref):
            _same_barrier(g, r)
        assert _counters(sT) == c1, T
        assert sT.record_report()["threads"]["effective"] == T
    return ref, got


def test_g1_barrier_d5():
    ref, got = _check_barrier(open(D5).read(), [(9000, 3), (4097, None)], [2, 7])
    b1, bT = ref[0][-1], got[0][-1]
    n = len(np.asarray(b1.frames()["record_xz"][0])[0])
    P = []
    for q in range(0, n, max(1, n // 6)):
        x, z = [0] * n, [0] * n
        z[q] = 1
        P.append((x, z, 0))
    assert np.array_equal(np.asarray(b1.signs(P)), np.asarray(bT.signs(P)))
    for i in (0, 1, 17, 4000):
        a, b = b1.materialize(i), bT.materialize(i)
        assert a.n == b.n and abs(abs(a.overlap(b)) - 1.0) < 1e-12
        assert np.array_equal(np.asarray(b1.record_outcomes(i)), np.asarray(bT.record_outcomes(i)))


def test_g1_barrier_factory_a():
    _check_barrier(_fac_a(), [(6000, 3), (3000, None)], [3, 48])


def test_g1_barrier_postselect():
    _check_barrier(open(D5).read(), [(9000, 8), (5000, None)], [2, 7], mask_of=_every_third)


def test_g1_barrier_term_route():
    text = open(os.path.join(HERE, "data", "ja_seed23.stim")).read()
    _check_barrier(text, [(6000, 3), (2000, None)], [2, 7])


@pytest.mark.parametrize("path", [p for p in CORPUS if "producer" in p or "code_switching" in p or "factory_d" in p],
                         ids=lambda p: os.path.basename(p))
def test_g1_barrier_corpus(path):
    _check_barrier(open(path).read(), [(5000, 2), (2000, None)], [3])


def test_g1_postselect_term_route():
    text = open(os.path.join(HERE, "data", "ch_cultivation.stim")).read()
    s = _sampler(text)
    if s.num_detectors == 0:
        pytest.skip("no detectors")
    _check_ps(text, [(9000, 2), (3000, None)], lambda s: np.ones(s.num_detectors, dtype=bool), [2, 7])


# ── T5: input frames (INPUT_QUBITS ports) and carried branches: framed / branch shots deferred to the home ───────────
def test_g1_input_frames():
    shots = 9000
    fr = (np.random.default_rng(1).random((shots, 2)) < 0.3).astype(np.uint8)
    ref = _sampler(A1).sample(shots, 4, input_pauli=fr)
    for T in (2, 7):
        s = _sampler(A1)
        got = s.sample(shots, 4, input_pauli=fr, threads=T)
        _same([got], [ref])
        th = s.record_report()["threads"]
        assert th["effective"] == T and 0 < th["deferred_shots"] < shots, th
        ps = _sampler(A1).sample(shots, 4, input_pauli=fr, postselect=[0, 1], threads=T)
        ps1 = _sampler(A1).sample(shots, 4, input_pauli=fr, postselect=[0, 1])
        _same([ps[:-1]], [ps1[:-1]])
        assert ps[-1] == ps1[-1]
        b1, bT = _barrier_fr(A1, shots, fr, 1), _barrier_fr(A1, shots, fr, T)
        _same_barrier(b1, bT)


def _barrier_fr(text, shots, fr, T):
    s = _sampler(text)
    b = s.sample_barrier(shots, 4, input_pauli=fr, threads=T)
    n = len(np.asarray(b.dets()))
    return (np.asarray(b.dets()).copy(), np.asarray(b.obs()).copy(), np.asarray(b.decisions()).copy(),
            list(b.record_keys()), [np.asarray(b.outcomes(i)).copy() for i in range(n)], {}, b)


PER_E = "RX 0\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nOUTPUT_QUBITS out 0\n"
CONS = "INPUT_QUBITS in 0\nH 0\nT 0\nH 0\nX_ERROR(0.05) 0\nM 0\nDETECTOR rec[-1]\n"


def test_g1_carried_branches():
    """input_branch: branch-0 shots on lanes, branch shots deferred; E_min != {} classes run on their per-E samplers."""
    N = 6000
    p = xtim.compile_twirl_sampler(PER_E, selfcheck=0)
    H, fr = p.branches(p.sample_barrier(N, seed=1))
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)

    def consumer():
        c = xtim.compile_twirl_sampler_from_ideal_state(CONS, ideal_state=p.ideal_state(), selfcheck=0)
        ids = {}
        ib = np.zeros(N, np.int32)
        for i, h in enumerate(H):
            k = (bytes(h["digest"]), bytes(h["z"]))
            if k not in ids:
                ids[k] = c.add_branch(h)
            ib[i] = ids[k]
        ib[::3] = 0                                   # branch-0 shots too
        return c, ib

    c1, ib = consumer()
    ref = c1.sample(N, seed=7, input_pauli=ip, input_branch=ib)
    for T in (2, 7):
        cT, ibT = consumer()
        got = cT.sample(N, seed=7, input_pauli=ip, input_branch=ibT, threads=T)
        _same([got], [ref])
        cP, ibP = consumer()
        c1P, ib1P = consumer()
        a = cP.sample(N, seed=7, input_pauli=ip, input_branch=ibP, postselect=[0], threads=T)
        b = c1P.sample(N, seed=7, input_pauli=ip, input_branch=ib1P, postselect=[0])
        _same([a[:-1]], [b[:-1]])
        assert a[-1] == b[-1]


# ── T7: the Python wrappers forward threads= ────────────────────────────────────────────────────────────────────────
def test_python_wrappers_forward_threads():
    text = open(os.path.join(HERE, "data", "adaptq_steane_h_producer.stim")).read()   # declares DECISIONs
    a = xtim.compile_twirl_sampler(text, selfcheck=0).sample(5000, 3)
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = s.sample(5000, 3, threads=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert s.record_report()["threads"]["effective"] == 4
    c = xtim.Circuit(open(D3R).read())
    ref = c.compile_detector_sampler(seed=7).sample(6000, separate_observables=True)
    got = c.compile_detector_sampler(seed=7).sample(6000, separate_observables=True, threads="auto")
    assert all(np.array_equal(x, y) for x, y in zip(ref, got))


# ── T6: RecordFidelity.evaluate(threads=): bitwise float64, each content's rows on one lane ─────────────────────────
FID = [os.path.join(HERE, "data", n) for n in ("magna_color5_fid.stim", "magna_rp2_fid.stim",
                                               "adaptq_qrm_magic_producer.stim", "adaptq_steane_h_producer.stim")] + \
      [os.path.join(HERE, "..", "examples", n) for n in ("distillation_15_1_3.stim", "color5_cultivation.stim",
                                                         "cube_ccz.stim", "code_switching_faithful.stim")]


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def test_record_handles_own_vs_foreign_sampler():
    """scaling F2: a sampler's own records carry its content handles as they are (its append-only table); a buffer of
    another sampler of the same model is resolved by content — here into a table numbered differently (it sampled
    other shots first): the same values, and its own buffers still evaluate."""
    text = open(FID[0]).read()
    sA = xtim.compile_twirl_sampler(text, selfcheck=0)
    sB = xtim.compile_twirl_sampler(text, selfcheck=0)
    sB.sample_barrier(3000, seed=99)                  # sB's content table: other contents, other numbering
    buf = sA.sample_barrier(4000, seed=5, threads=4)
    own = xtim.RecordFidelity(sA).evaluate(buf)
    for T in (1, 4):
        assert np.array_equal(_bits(xtim.RecordFidelity(sB).evaluate(buf, threads=T)), _bits(own)), T
    bufB = sB.sample_barrier(2000, seed=6)
    assert np.array_equal(_bits(xtim.RecordFidelity(sB).evaluate(bufB)), _bits(xtim.RecordFidelity(sA).evaluate(bufB)))


@pytest.mark.parametrize("path", FID, ids=lambda p: os.path.basename(p))
def test_g1_record_fidelity(path):
    text = open(path).read()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s.sample_barrier(6000, seed=5)
    n_out = len(s.output_wires())
    corr = (np.random.default_rng(3).random((6000, 2, n_out)) < 0.2).astype(np.uint8)
    keep = np.arange(0, 6000, 3)
    try:
        ref_rf = xtim.RecordFidelity(s)
    except xtim.XtimFidelityRefusal:
        pytest.skip("outside the record fidelity's output contract")
    ref = [ref_rf.evaluate(buf), ref_rf.evaluate(buf, correction=corr), ref_rf.evaluate(buf, keep, keep_memo=True),
           ref_rf.evaluate(buf)]
    for T in (2, 7):
        rf = xtim.RecordFidelity(s)
        got = [rf.evaluate(buf, threads=T), rf.evaluate(buf, correction=corr, threads=T),
               rf.evaluate(buf, keep, keep_memo=True, threads=T), rf.evaluate(buf, threads=T)]
        for g, r in zip(got, ref):
            assert np.array_equal(_bits(g), _bits(r)), (T, path)
        st, st1 = rf.stats(), ref_rf.stats()
        assert st["threads"]["requested"] == T and (st["threads"]["effective"] > 1 or st["threads"]["limit"]), st
        for k in ("shots", "misses", "memo_hits", "held_bytes", "memo_bytes", "cache_flushes", "plans"):
            assert st[k] == st1[k], (k, st[k], st1[k])


@pytest.mark.parametrize("path", FID[:3], ids=lambda p: os.path.basename(p))
def test_g1_record_fidelity_hot_content_split(path):
    """scaling F3: one content's rows (the buffer's hottest) split into chunks on several lanes — each lane with its own
    previous-row projection cache, the content's structure completed first on the home: values bit for bit, the
    shot / miss / memo-hit / byte counters as threads=1 (two lanes' copies of one key fold back to one computation)."""
    text = open(path).read()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s.sample_barrier(20000, seed=11)
    try:
        xtim.RecordFidelity(s)
    except xtim.XtimFidelityRefusal:
        pytest.skip("outside the record fidelity's output contract")
    keys = [buf.plan_content(i)[1] for i in range(buf.num_shots)]
    hot = max(set(keys), key=keys.count)
    rows = np.array([i for i, k in enumerate(keys) if k == hot], dtype=np.int64)
    assert len(rows) > 1024, len(rows)
    ref_rf = xtim.RecordFidelity(s)
    ref = [ref_rf.evaluate(buf, rows, keep_memo=True), ref_rf.evaluate(buf, keep_memo=True)]
    st1 = ref_rf.stats()
    for T in (2, 7):
        rf = xtim.RecordFidelity(s)
        got = [rf.evaluate(buf, rows, keep_memo=True, threads=T), rf.evaluate(buf, keep_memo=True, threads=T)]
        for g, r in zip(got, ref):
            assert np.array_equal(_bits(g), _bits(r)), (T, path)
        st = rf.stats()
        for k in ("shots", "misses", "memo_hits", "held_bytes", "memo_bytes", "plans", "rn_misses", "rotation_misses",
                  "sectors", "sectors_nonzero", "orthogonal_rows", "value_builds", "kernel_compiles"):
            assert st[k] == st1[k], (k, st[k], st1[k], T)
    rf = xtim.RecordFidelity(s)
    rf.evaluate(buf, rows, threads=7)
    assert rf.stats()["threads"]["effective"] > 1, rf.stats()["threads"]     # one content: split, not serial


def test_g1_record_fidelity_factory():
    p = os.path.join(str(PRIVATE_DIR), "ccz_factory_a_out.stim")
    if not os.path.isfile(p):
        pytest.skip("private fixture absent")
    s = xtim.compile_twirl_sampler(open(p).read(), selfcheck=0)
    buf = s.sample_barrier(4000, seed=2)
    try:
        ref = xtim.RecordFidelity(s).evaluate(buf)
    except Exception as e:                            # the output contract refuses this one: the same refusal
        with pytest.raises(type(e)):
            xtim.RecordFidelity(s).evaluate(buf, threads=7)
        return
    got = xtim.RecordFidelity(s).evaluate(buf, threads=7)
    assert np.array_equal(_bits(got), _bits(ref))


def test_record_fidelity_held_bytes_and_cache_budget():
    """Fix F1: the lanes' structure bytes are the evaluator's: held_bytes and the cache_budget's flushes equal the
    serial evaluator's at every call."""
    text = open(os.path.join(HERE, "data", "magna_color5_fid.stim")).read()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s.sample_barrier(6000, seed=5)
    rfs = {T: xtim.RecordFidelity(s, cache_budget=20_000_000) for T in (1, 4)}
    for _ in range(4):
        outs = {T: rf.evaluate(buf, threads=T) for T, rf in rfs.items()}
        assert np.array_equal(_bits(outs[1]), _bits(outs[4]))
        st = {T: rf.stats() for T, rf in rfs.items()}
        assert st[4]["threads"]["effective"] == 4
        for k in ("held_bytes", "cache_flushes", "memo_bytes"):
            assert st[1][k] == st[4][k], (k, st[1][k], st[4][k])
    assert st[1]["cache_flushes"] > 0                 # the budget binds (the repro's flushes)


_BUDGET_SCRIPT = """
import sys, hashlib, numpy as np, warnings
warnings.simplefilter("ignore")
import xtim._xtim_testapi as TA
s = TA.compile_family_sampler(open(sys.argv[1]).read(), selfcheck=0)
s._set_lane_bytes(1)                                  # the D3 estimate off: lanes start that the budget then refuses
out = s.sample(20000, 3, threads=int(sys.argv[2]))
th = s.record_report()["threads"]
print(hashlib.sha1(b"".join(np.asarray(a).tobytes() for a in out)).hexdigest(), th["budget_fallbacks"], th["effective"])
"""


def test_lane_budget_refusal_reruns_serially():
    """Fix F2: under a tight XTIM_MAX_STATE_BYTES a lane's allocation is refused; the call is not: the super-block is
    re-run on the calling thread from its drawn events and the call continues serially (the serial bytes)."""
    import subprocess, sys
    path = os.path.join(str(PRIVATE_DIR), "ccz_factory_a.stim")
    if not os.path.isfile(path):
        pytest.skip("private fixture absent")
    env = dict(os.environ, XTIM_MAX_STATE_BYTES="140000", XTIM_QUIET="1")
    root = os.path.dirname(HERE)
    run = lambda T: subprocess.run([sys.executable, "-c", _BUDGET_SCRIPT, path, str(T)], env=env, cwd=root,
                                   capture_output=True, text=True, check=True).stdout.split()
    d1, f1, e1 = run(1)
    d8, f8, e8 = run(8)
    assert d1 == d8 and f1 == "0" and int(f8) >= 1 and e8 == "1", (d1, d8, f8, e8)


@pytest.mark.parametrize("T", [1, 4])
def test_record_fidelity_one_call_at_a_time(T):
    """Fix R1: evaluate runs without the GIL (scaling: at threads=1 too); a concurrent clear / clear_memo / stats /
    evaluate on the same evaluator is refused by name (XT_FIDELITY_BUSY) and the running call's values are
    untouched."""
    import threading
    text = open(os.path.join(HERE, "data", "magna_color5_fid.stim")).read()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s.sample_barrier(30000, seed=5)
    ref = xtim.RecordFidelity(s).evaluate(buf)
    rf = xtim.RecordFidelity(s)
    raw = rf._rf
    out, refused, done = {}, [], threading.Event()

    def run():
        out["F"] = np.asarray(raw.evaluate(buf, None, None, T))
        done.set()

    t = threading.Thread(target=run)
    t.start()
    while not done.is_set():
        for call in (raw.clear, raw.clear_memo, raw.stats):
            try:
                call()
            except RuntimeError as e:
                assert "XT_FIDELITY_BUSY" in str(e), e
                refused.append(1)
    t.join()
    assert refused, "no concurrent call met the running evaluate"
    assert np.array_equal(_bits(out["F"]), _bits(ref))


def test_record_fidelity_releases_the_gil_at_one_thread():
    """scaling: evaluate(threads=1) runs its C++ (the handle resolution, the intern, the rows) without the GIL — a
    Python thread keeps its pace during the call (with the GIL held it would get a switch interval or two at most)."""
    import threading
    import time
    text = open(os.path.join(HERE, "data", "magna_color5_fid.stim")).read()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s.sample_barrier(60000, seed=5)
    raw = xtim.RecordFidelity(s)._rf
    ticks, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1

    t = threading.Thread(target=spin)
    t.start()
    try:
        a = ticks[0]; t0 = time.perf_counter(); time.sleep(0.1); idle = (ticks[0] - a) / (time.perf_counter() - t0)
        a = ticks[0]; t0 = time.perf_counter()
        raw.evaluate(buf, None, None, 1)
        dt = time.perf_counter() - t0
        busy = (ticks[0] - a) / dt
    finally:
        stop.set()
        t.join()
    assert dt > 0.1, dt                               # long enough that a switch interval or two would not pass
    assert busy > 0.25 * idle, (busy, idle, dt)   # (held: ~0.01; a loaded host slows the spinner too)


# ── scaling fix-3: a failed Python call commits nothing (multi-engine input-branch call, reseed, counters) ─────────────
def _eclass_consumer(N):
    """A test-API consumer of PER_E's producer with E_min != {} branch rows (E-class inner samplers) and branch-0 rows."""
    p = TA.compile_family_sampler(PER_E, selfcheck=0)
    H, fr = p.branches(p.sample_barrier(N, seed=1))
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)

    def consumer():
        c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), CONS)
        ids = {}
        ib = np.zeros(N, np.int32)
        for i, h in enumerate(H):
            k = (bytes(h["digest"]), bytes(h["z"]))
            if k not in ids:
                ids[k] = c.add_branch(h)
            ib[i] = ids[k]
        ib[::3] = 0
        return c, ib

    return consumer, ip


@pytest.mark.parametrize("T", [1, 4])
def test_failed_input_branch_call_commits_nothing(T):
    """(1) the outer sampler's run succeeds, an E-class inner run succeeds, then the call fails (test seam): the outer and
    the inner samplers go back to the call's entry — a seed=None continuation equals a never-failed run; (3) a failed
    SEEDED call leaves the sampler as before its reseed; (4) record_report()['used'] does not count a failed call."""
    N = 3000
    consumer, ip = _eclass_consumer(N)
    r, ib = consumer()
    first = r.sample(N, 7, input_pauli=ip, input_branch=ib)
    want = r.sample(N, None, input_pauli=ip, input_branch=ib)
    want2 = r.sample(N, None, input_pauli=ip, input_branch=ib)
    c, ibc = consumer()
    _same([c.sample(N, 7, input_pauli=ip, input_branch=ibc, threads=T)], [first])
    used0 = c.record_report()["used"]
    c._fail_after_inner_run(1)
    with pytest.raises(RuntimeError, match="test seam"):
        c.sample(N, None, input_pauli=ip, input_branch=ibc, threads=T)
    assert c.record_report()["used"] == used0
    _same([c.sample(N, None, input_pauli=ip, input_branch=ibc, threads=T)], [want])
    used1 = c.record_report()["used"]
    c._fail_after_inner_run(1)                       # (3) a failed call with a NEW seed: as if never made
    with pytest.raises(RuntimeError, match="test seam"):
        c.sample(N, 12345, input_pauli=ip, input_branch=ibc, threads=T)
    assert c.record_report()["used"] == used1
    _same([c.sample(N, None, input_pauli=ip, input_branch=ibc, threads=T)], [want2])


# ── scaling fix-4/5 (A): STRUCTURAL gate — a failed call leaves EVERY Python-visible report as at its entry ───────────
# Discovery is loud: every zero-argument dict-returning method of the sampler's class (read from its pybind signature,
# never called to find out) must be exactly the expected set below — a new or removed report accessor fails here until
# this list (and, if it moves, the allowlist) is updated. channel_one_counts (a list) is added explicitly.
# The allowlist is an EXACT set of flattened paths of CACHED WORK a failed call may legitimately have built and keeps
# (the next call reuses it; outputs never depend on it), each with its check and reason: ">=" cumulative counters
# (never decrease — fix-4 B: a reseeding failed call never loses their history), "any" gauges (resident sizes).
# Every other path — shot / route counts, `used`, branch_shots, branch_map_probes, channel and oracle counts, the
# thread report, term_merge_stats — must be EXACTLY equal.
_EXPECTED_DICT_ACCESSORS = {
    "FamilySampler": frozenset(("_footprint", "branch_counters", "carried_compose_report", "channel_report",
                                "oracle_counts", "plan_payload_stats", "record_classes", "record_counters",
                                "record_layout", "record_report", "route_stats", "term_merge_stats")),
    "TwirlSampler": frozenset(("born_dec", "branch_counters", "record_layout", "record_reads", "record_report")),
}
_EXTRA_ACCESSORS = {"FamilySampler": (), "TwirlSampler": ("channel_one_counts",)}
_R_READER = ("a plan's second use promotes its parked record table to a reader table (as a successful call does); "
             "the table stays cached and the next call reuses it")
_CACHED_PATHS = {   # flattened path -> (check, reason). Nothing else is exempt.
    ("record_report", "counters", "reader_tables"): (">=", _R_READER),
    ("record_report", "plan_memory", "reader_tables"): (">=", _R_READER),
    ("record_report", "counters", "reader_lookups"): (">=", "reader-cache lookups made building / finding those tables"),
    ("record_report", "plan_memory", "reader_lookups"): (">=", "the same counter in the plan-memory view"),
    ("record_report", "plan_memory", "reader_peak_bytes"): (">=", "peak resident bytes of the reader cache"),
    ("plan_payload_stats", "reader_peak_bytes"): (">=", "the same peak in the payload view"),
    ("record_report", "plan_memory", "reader_bytes"): ("any", "resident bytes of the reader cache (gauge)"),
    ("plan_payload_stats", "reader_bytes"): ("any", "the same gauge in the payload view"),
    ("_footprint", "shot_block"): ("any", "retained capacity of the S-b shot-block buffers (gauge; no state)"),
    ("_footprint", "shot_scratch"): ("any", "retained capacity of the per-shot scratch buffers (gauge; no state)"),
    ("_footprint", "G"): (">=", "lean D1-S2: the certified planes' lazily built indexes (xcol_index: a threaded call's "
                                "freeze builds it; a pure function of G — cached work)"),
    ("_footprint", "term_view"): ("any", "lean D1-S1: the term view's retained bytes (gauge; no state) — the test API's accessors "
                                         "(ideal_fs / branch handles / _engine) arm it on demand, never a run path"),
    # plancache-design: classes a failed call created stay (pure functions of their keys, as plan headers do)
    ("record_report", "plan_memory", "classes"): (">=", "plan classes created (cached work, never dropped)"),
    ("record_report", "plan_memory", "class_builds"): (">=", "class payload builds (cached work)"),
    ("record_report", "plan_memory", "class_rep_diffs"): (">=", "O1 representation diffs of built classes (diagnostic)"),
    ("record_counters", "classes"): (">=", "the same counter in the test-API view"),
    ("record_counters", "class_builds"): (">=", "the same counter in the test-API view"),
    ("record_counters", "class_rep_diffs"): (">=", "the same counter in the test-API view"),
    ("record_report", "plan_memory", "arena_chunk_bytes"): ("any", "the plan arena's mapped chunks (gauge)"),
    ("record_report", "plan_memory", "arena_free_bytes"): ("any", "the plan arena's free-list bytes (gauge)"),
    ("record_report", "plan_memory", "arena_live_bytes"): ("any", "the plan arena's live blocks (gauge)"),
    ("record_report", "plan_memory", "second_chances"): (">=", "FIFO re-queues of hit classes (cached work)"),
}


def _dict_accessors(cls):
    import re
    found = set()
    for name in dir(cls):
        if name.startswith("__"):
            continue
        for line in (getattr(cls, name).__doc__ or "").splitlines():
            if re.fullmatch(rf"{re.escape(name)}\(self: [\w.]+\) -> dict", line.strip()):
                found.add(name)
    return found


def _flat(v, path, out):
    if isinstance(v, dict):
        for k, x in v.items():
            _flat(x, path + (str(k),), out)
    elif isinstance(v, np.ndarray):
        out[path] = (str(v.dtype), v.shape, v.tolist())
    else:
        out[path] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def _flat_reports(s):
    cls = type(s)
    want = _EXPECTED_DICT_ACCESSORS[cls.__name__]
    got = _dict_accessors(cls)
    assert got == want, ("report accessors changed: update _EXPECTED_DICT_ACCESSORS", cls.__name__, got ^ want)
    out = {}
    for a in sorted(want) + list(_EXTRA_ACCESSORS[cls.__name__]):
        _flat(getattr(cls, a)(s), (a,), out)
    return out


def _assert_reports_as_at_entry(before, after, what):
    assert set(before) == set(after), (what, set(before) ^ set(after))
    bad = []
    for path in before:
        b, a = before[path], after[path]
        check = _CACHED_PATHS.get(path, ("==", ""))[0]
        if check == "any":
            continue
        if check == ">=":
            ok = (len(a) >= len(b) and all(x >= y for x, y in zip(a, b))) if isinstance(b, list) else a >= b
        else:
            ok = a == b
        if not ok:
            bad.append((path, b, a))
    assert not bad, (what, bad)


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("seed", [None, 12345])
def test_failed_call_whole_report_unchanged_input_branch(T, seed):
    """(A) an input-branch call (outer + E-class inner samplers) that fails after an inner run: the OUTER sampler's
    whole report is as at the call's entry; the entry call ran at threads=1, the failed one at threads=T (E: the
    thread report); a SEEDED failure (B) keeps the cached-work history set_seed zeroed. The E-class inner samplers
    expose no report to Python: their restore is covered by the byte-continuation test
    (test_failed_input_branch_call_commits_nothing), not by this comparison."""
    N = 3000
    consumer, ip = _eclass_consumer(N)
    c, ib = consumer()
    c.sample(N, 7, input_pauli=ip, input_branch=ib)
    c.sample(N, None, input_pauli=ip, input_branch=ib)
    before = _flat_reports(c)
    c._fail_after_inner_run(1)
    with pytest.raises(RuntimeError, match="test seam"):
        c.sample(N, seed, input_pauli=ip, input_branch=ib, threads=T)
    _assert_reports_as_at_entry(before, _flat_reports(c), "input-branch")


_PART_TEXT = ("INPUT_BITS drv 1\nRX 0 1\nR 2\nT 0\nIF drv[0] {\n  Z 0\n}\nX_ERROR(0.05) 0 1 2\nMX 0\nCZ rec[-1] 1\n"
              "MX 1\nM 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")


class _GroupProxy:
    """Stands in for one partition's inner sampler: runs the real sample and then (fail_sample) raises; its restore
    restores the real sampler and then (fail_restore) raises."""

    def __init__(self, s, fail_sample=None, fail_restore=None):
        self._s, self._fail_sample, self._fail_restore = s, fail_sample, fail_restore

    def sample(self, *a, **kw):
        out = self._s.sample(*a, **kw)
        if self._fail_sample is not None:
            raise self._fail_sample()
        return out

    def _save_call_state(self):
        return self._s._save_call_state()

    def _restore_call_state(self, st):
        self._s._restore_call_state(st)
        if self._fail_restore is not None:
            raise self._fail_restore


def _partitioned(N, monkeypatch=None, family=False):
    if family:   # the groups' samplers are test-API FamilySamplers (their C++ failure seam)
        monkeypatch.setattr(xtim.twirl, "_make_inner_sampler",
                            lambda text, p, dc, sc, dec=True, b=None: TA.compile_family_sampler(
                                text, p_factor=p, selfcheck=0, decisions=dec))
    bits = np.array([[0], [1]] * (N // 2), np.uint8)
    ref = xtim.compile_twirl_sampler(_PART_TEXT, selfcheck=0)
    ref.sample(N, 3, decision_bits=bits)
    want = ref.sample(N, None, decision_bits=bits)
    s = xtim.compile_twirl_sampler(_PART_TEXT, selfcheck=0)
    assert isinstance(s, xtim.twirl.PartitionedTwirlSampler)
    s.sample(N, 3, decision_bits=bits)
    keys = sorted(s._sampler_cache)
    assert len(keys) == 2
    return s, bits, want, keys, {k: s._sampler_cache[k] for k in keys}


def _same_batch(x, y):
    x, y = list(x), list(y)
    assert len(x) == len(y)
    for u, v in zip(x, y):
        assert (u is None) == (v is None)
        if u is not None:
            assert u.dtype == v.dtype and u.shape == v.shape and np.array_equal(u, v)


def _check_partition_as_at_entry(s, inner, before, summed, what):
    for k, v in inner.items():
        _assert_reports_as_at_entry(before[k], _flat_reports(v), (what, k))
    _assert_reports_as_at_entry(_flat(summed, ("record_report",), {}), _flat(s.record_report(), ("record_report",), {}),
                                (what, "sum"))


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("seed", [None, 12345])
def test_failed_call_whole_report_unchanged_partitioned(T, seed):
    """(A, D) PartitionedTwirlSampler: the second group's sampler fails after its run (a Python failure); both group
    samplers' whole reports (and the summed record_report) are as at the call's entry, and a seed=None continuation
    equals a never-failed sampler's."""
    N = 2000
    s, bits, want, keys, inner = _partitioned(N)
    before = {k: _flat_reports(v) for k, v in inner.items()}
    summed = s.record_report()
    s._sampler_cache[keys[1]] = _GroupProxy(inner[keys[1]], fail_sample=lambda: RuntimeError("test: group failed"))
    with pytest.raises(RuntimeError, match="test: group failed"):
        s.sample(N, seed, decision_bits=bits, threads=T)
    s._sampler_cache[keys[1]] = inner[keys[1]]
    _check_partition_as_at_entry(s, inner, before, summed, "partition")
    _same_batch(s.sample(N, None, decision_bits=bits), want)


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("seed", [None, 12345])
def test_failed_call_cpp_failure_inside_partition_group(T, seed, monkeypatch):
    """fix-5: a C++ failure INSIDE a group's call (test-API seam after its engine run): the binding's C++ rollback
    restores that sampler, then the partition's Python restore restores it again (idempotent double restore) — the
    same whole reports and the same continuation as a never-failed sampler."""
    N = 2000
    s, bits, want, keys, inner = _partitioned(N, monkeypatch, family=True)
    before = {k: _flat_reports(v) for k, v in inner.items()}
    summed = s.record_report()
    inner[keys[1]]._fail_after_run(1)
    with pytest.raises(RuntimeError, match="test seam: the call failed after its engine run"):
        s.sample(N, seed, decision_bits=bits, threads=T)
    _check_partition_as_at_entry(s, inner, before, summed, "cpp-in-group")
    _same_batch(s.sample(N, None, decision_bits=bits), want)


def test_failed_partition_restore_errors_kept_original_unchanged():
    """fix-5 (D): when restores fail too, the ORIGINAL exception propagates unchanged (its own __cause__ intact) and
    EVERY restore error is kept (a note each + .xtim_restore_errors); every sampler is still restored."""
    N = 2000
    s, bits, want, keys, inner = _partitioned(N)
    before = {k: _flat_reports(v) for k, v in inner.items()}
    summed = s.record_report()

    def failing():
        try:
            raise KeyError("root cause")
        except KeyError as e:
            err = RuntimeError("test: group failed")
            err.__cause__ = e
            return err

    r0, r1 = ValueError("restore 0 failed"), ValueError("restore 1 failed")
    s._sampler_cache[keys[0]] = _GroupProxy(inner[keys[0]], fail_restore=r0)
    s._sampler_cache[keys[1]] = _GroupProxy(inner[keys[1]], fail_sample=failing, fail_restore=r1)
    with pytest.raises(RuntimeError, match="test: group failed") as ei:
        s.sample(N, None, decision_bits=bits)
    e = ei.value
    assert isinstance(e.__cause__, KeyError) and e.__cause__.args == ("root cause",)
    assert set(e.xtim_restore_errors) == {r0, r1}
    notes = "\n".join(getattr(e, "__notes__", []))
    assert "restore 0 failed" in notes and "restore 1 failed" in notes
    s._sampler_cache.update(inner)
    _check_partition_as_at_entry(s, inner, before, summed, "restore-errors")
    _same_batch(s.sample(N, None, decision_bits=bits), want)
