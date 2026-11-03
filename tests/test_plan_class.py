"""plancache-design (owner-signed rev 4, 2026-11-01): the two-level plan cache — headers by the FULL key, one record
table / emission block / Born trie per CLASS key (a ∧ U ∧ ¬prod, Π(B)), Π's kernel K2' = theorem T3's subspace.

Gates pinned here (plan §6):
  * NULL OPTION: the sample bytes with Π = id (the test lever) equal the bytes with Π = K2' — on every circuit, at
    T = 1 and T > 1 — and the counters the plan keeps thread-identical (plans, plan_misses total, route counts) agree;
  * classes <= plans; every walked plan carries a class id (plan_structure); class_rep_diffs == 0 with the selfcheck
    window on (O1 never fires on the corpus: the class table equals every member key's own table);
  * the class map is reported (record_report()["plan_memory"]: classes, class_builds, class_kernel_dim, class_cap)
    and logged once per compile (the plan_class diagnostic);
  * readers read the class table: plan_structure(content) of two keys of one class returns the same table and class id;
  * the payload budget evicts and rebuilds CLASSES byte-neutrally (the S2b bytes gate, now per class);
  * review M1: plan_structure()["class_id"] is identical at every thread count (the serial first-seen order);
  * review H1: a lane's arena is bounded independent of the call count (lent tails, no chunk per threaded call).
"""
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import xtim

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXAMPLES = HERE.parent / "examples"
from _private import private_fixture  # noqa: E402

CIRCUITS = {
    "cultivation_d5": EXAMPLES / "cultivation_d5.stim",
    "color5": DATA / "magna_color5_fid.stim",
    "rp2": DATA / "magna_rp2_fid.stim",
    # plan §7f: circuits with EXTRAS (Born decisions / observables, gauge detectors) — every column enters U, (i), (ii)
    "born5_dec": DATA / "adaptq_born5_dec_enum_producer.stim",
    "cultivation_d3_rate": EXAMPLES / "cultivation_d3_rate.stim",
}
CIRCUITS = {k: v for k, v in CIRCUITS.items() if v.is_file()}
M3_OBS = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
          "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
          "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\nMX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
          "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1 2\n")


def _texts():
    out = [("m3_obs", M3_OBS)]
    for k, p in CIRCUITS.items():
        out.append((k, p.read_text()))
    return out


def _sampler(text, identity=False, selfcheck=0):
    # the null-option lever of the PRODUCTION extension (a process-global read at construction; each extension links
    # its own copy of the engine, so the test API's lever does not reach _xtim's samplers)
    import xtim._xtim as _x
    _x._plan_class_identity(identity)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xtim.compile_twirl_sampler(text, selfcheck=selfcheck)
    finally:
        _x._plan_class_identity(False)


def _ids():
    return [n for n, _ in _texts()]


def _digest(s, shots, seed, threads=1):
    out = s.sample(shots, seed=seed, threads=threads)
    return b"".join(np.ascontiguousarray(np.asarray(a)).tobytes() for a in out)


def _barrier_digest(s, shots, seed):
    b = s.sample_barrier(shots, seed=seed)
    return (np.asarray(b.dets()).tobytes() + np.asarray(b.obs()).tobytes()
            + b"".join(bytes(np.asarray(b.outcomes(i), np.uint8)) for i in range(shots)))


@pytest.mark.parametrize("name,text", _texts(), ids=_ids())
def test_null_option_bytes_identical_and_classes_reported(name, text):
    shots = 6000 if name != "m3_obs" else 3000
    ref = _sampler(text, identity=True)
    ref_bytes = _digest(ref, shots, 11)
    rr = ref.record_report()
    assert rr["plan_memory"]["class_cap"] == "identity" and rr["plan_memory"]["class_kernel_dim"] == 0
    s = _sampler(text)
    got = _digest(s, shots, 11)
    assert got == ref_bytes, name
    r = s.record_report()
    pm = r["plan_memory"]
    assert pm["class_cap"] == "" and pm["classes"] >= 1 and pm["classes"] <= r["plans"], pm
    assert pm["class_rep_diffs"] == 0
    assert pm["class_builds"] >= pm["classes"]
    for k in ("plans", "route_counts", "used"):
        assert r[k] == rr[k], (k, r[k], rr[k])
    assert sum(r["plan_misses"].values()) == sum(rr["plan_misses"].values())   # (the kappa split is the class rep's)
    # the Π = id run classes by the a ∧ U strip alone; K2' merges further (fewer builds) where its kernel is nonzero
    assert rr["plan_memory"]["classes"] <= rr["plans"]
    assert pm["classes"] <= rr["plan_memory"]["classes"]


@pytest.mark.parametrize("name,text", _texts(), ids=_ids())
def test_thread_counts_identical(name, text):
    # the counters the threads gate (tests/test_threads.py) keeps T-identical (route splits are not: two lanes may
    # build one plan in a super-block), plus the class count and the plan_misses total
    def cnt(s):
        r = s.record_report()
        rc = r["route_counts"]
        pm = r["plan_memory"]
        return {"used": r["used"], "clean": rc["clean"], "routes": sum(rc.values()), "plans": r["plans"],
                "misses": sum(r["plan_misses"].values()), "header_bytes": pm["header_bytes"],
                "classes": pm["classes"], "chan": list(s.channel_one_counts())}
    shots = 20000 if name != "m3_obs" else 4000
    s1 = _sampler(text)
    ref = _digest(s1, shots, 5)
    c1 = cnt(s1)
    for T in (2, 7):
        s = _sampler(text)
        got = _digest(s, shots, 5, threads=T)
        assert got == ref, (name, T)
        r = s.record_report()
        assert r["threads"]["effective"] > 1 or r["threads"]["limit"], r["threads"]
        assert cnt(s) == c1, (T, cnt(s), c1)
        assert r["plan_memory"]["class_rep_diffs"] == 0


@pytest.mark.parametrize("name,text", _texts(), ids=_ids())
def test_selfcheck_window_o1_never_fires(name, text, caplog):
    caplog.set_level(logging.INFO, logger="xtim.twirl")
    s = _sampler(text, selfcheck=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.sample(2000 if name != "m3_obs" else 1500, seed=3)
    pm = s.record_report()["plan_memory"]
    assert pm["class_rep_diffs"] == 0, pm
    texts = [rec.getMessage() for rec in caplog.records]
    assert any("plan classes:" in t for t in texts), texts
    assert not any("record table differs from its class" in t for t in texts), texts


def test_readers_read_the_class_table_and_class_ids():
    from xtim import _xtim_testapi as TA
    text = CIRCUITS.get("cultivation_d5", None)
    if text is None:
        pytest.skip("example absent")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = TA.compile_family_sampler(text.read_text(), selfcheck=0)
    b = s.sample_barrier(4000, seed=2)
    by_class = {}
    for i in range(4000):
        ps = b.plan_structure(i)
        cid = int(ps["class_id"])
        if cid == 0:
            continue
        key = (int(np.asarray(ps["rho"])[0]), int(np.asarray(ps["kappa_rec"])[0]),
               np.asarray(ps["pivot_rec"]).tobytes() if "pivot_rec" in ps else b"")
        by_class.setdefault(cid, set()).add(key)
    assert by_class
    for cid, keys in by_class.items():
        assert len(keys) == 1, (cid, keys)          # one table per class: every member reports the same shape
    rc = s.record_counters()
    assert rc["classes"] == len({c for c in by_class}) or rc["classes"] >= len(by_class)


def test_budget_evicts_classes_byte_neutral():
    from xtim import _xtim_testapi as TA
    text = M3_OBS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s0 = TA.compile_family_sampler(text, selfcheck=0)
        t = TA.compile_family_sampler(text, selfcheck=0)
    cold = _barrier_digest(s0, 3000, 21)
    r0 = s0.record_report()
    t.set_plan_payload_budget(max(1, 2 * r0["plan_memory"]["resident_bytes"] // 3))
    part = _barrier_digest(t, 3000, 21)
    assert cold == part
    rt = t.record_report()["plan_memory"]
    assert rt["evictions"] > 0 and rt["payload_builds"] > 0, rt
    assert rt["class_builds"] == rt["classes"] + rt["payload_builds"]
    t.set_plan_payload_budget(1 << 40)                 # the tries rebuilt, then incremental == from scratch
    assert _barrier_digest(t, 3000, 21) == cold
    checked, bad = t.born_trie_check()
    assert checked > 0 and bad == 0


def test_private_factory_classes():
    p = private_fixture("ccz_factory_a.stim")
    if not os.path.isfile(str(p)):
        pytest.skip("private fixture absent")
    text = p.read_text()
    ref = _sampler(text, identity=True)
    rb = _digest(ref, 20000, 9)
    s = _sampler(text)
    assert _digest(s, 20000, 9) == rb
    pm = s.record_report()["plan_memory"]
    assert pm["class_cap"] == "" and pm["class_rep_diffs"] == 0 and 0 < pm["classes"] <= s.record_report()["plans"]


def test_record_table_builder_callers_are_the_choke_points():
    """plan §1 (review F2, low): build_record_table_recB's production callers are grep-gated to the readers' ONE choke
    point (rec_table_of_: the class representative's table, or an unclassed route-1 content's own), the class build
    (build_rec_table_fam_, the parked-core miss path), O1's per-key oracle (class_rep_check_) and the DEM path
    (fault_record_law: not a plan reader). Every other reader goes through rec_table_memo_ / rec_table_held_ /
    content_table_ / reader_table_fresh_."""
    import re
    root = HERE.parent / "cpp"
    allowed = {
        "src/record_space_b.cpp",            # the definition
        "src/twirl_sampler_impl.inc.hpp",    # rec_table_of_ (the choke point's builder)
        "src/twirl_sampler_fam.cpp",         # build_rec_table_fam_ (no parked core: expected 0, counted)
        "src/twirl_sampler.cpp",             # class_rep_check_ (O1)
        "src/fault_record_law.cpp",          # the DEM's per-law table (not a plan reader)
    }
    found = set()
    for sub in ("src", "bindings"):
        for p in sorted((root / sub).rglob("*")):
            if p.suffix not in (".cpp", ".hpp"):
                continue
            t = p.read_text(errors="replace")
            if re.search(r"\bbuild_record_table_recB\s*\(", t):
                found.add(str(p.relative_to(root)))
    assert found == allowed, (found - allowed, allowed - found)
    impl = (root / "src" / "twirl_sampler_impl.inc.hpp").read_text()
    assert impl.count("build_record_table_recB(") == 1          # rec_table_of_ only
    ts = (root / "src" / "twirl_sampler.cpp").read_text()
    assert ts.count("build_record_table_recB(") == 1            # class_rep_check_ only (plan_structure reads the class)


def test_second_chance_eviction_keeps_hit_classes(caplog):
    """plan §2 R2: under a binding budget the FIFO skips (once) a class served since the last pass, so singletons go
    first; the mechanism's trace is plan_memory.second_chances > 0 (MUT-R2-NO-SECOND-CHANCE: 0), and the bytes do
    not move (every evicted value is a pure function of its key)."""
    text = CIRCUITS.get("cultivation_d5")
    if text is None:
        pytest.skip("example absent")
    text = text.read_text()
    s0 = _sampler(text)
    ref = _digest(s0, 20000, 7)
    r0 = s0.record_report()["plan_memory"]
    t = _sampler(text)
    t.set_plan_payload_budget(max(1, r0["resident_bytes"] // 3))
    got = _digest(t, 20000, 7)
    assert got == ref
    pm = t.record_report()["plan_memory"]
    assert pm["evictions"] > 0 and pm["payload_builds"] > 0, pm
    assert pm["second_chances"] > 0, pm
    assert pm["resident_bytes"] <= max(1, r0["resident_bytes"] // 3) + pm["overshoot_bytes"]


@pytest.mark.parametrize("name", ["cultivation_d5", "ch_cultivation", "factory_a"])
def test_class_ids_thread_identical(name):
    """review M1: the class id a plan reports is the serial first-seen order of its class at EVERY thread count
    (a super-block's new classes are ordered by the first shot that needed them; deferred shots included)."""
    if name == "factory_a":
        p = private_fixture("ccz_factory_a.stim")
        if not os.path.isfile(str(p)):
            pytest.skip("private fixture absent")
    elif name == "ch_cultivation":
        p = DATA / "ch_cultivation.stim"
    else:
        p = CIRCUITS.get(name)
    if p is None or not os.path.isfile(str(p)):
        pytest.skip("fixture absent")
    text = p.read_text()
    N = 40000 if name != "factory_a" else 12000

    def run(T):
        s = _sampler(text)
        b = s.sample_barrier(N, seed=3, threads=T)
        cids, conts = [], []
        for i in range(0, N, 7):
            ps = b.plan_structure(i)
            cids.append(int(ps["class_id"]))
            conts.append(bytes(ps["plan_content"]))
        th = s.record_report()["threads"]
        return cids, conts, s.record_report()["plan_memory"]["classes"], th

    c1, p1, n1, _ = run(1)
    c8, p8, n8, th = run(8)
    assert th["effective"] > 1 or th["limit"], th
    assert p1 == p8 and n1 == n8
    assert c1 == c8, (name, sum(a != b for a, b in zip(c1, c8)), len(c1))


def test_lane_arena_bounded_over_threaded_calls():
    """review H1: many threaded calls map no chunk per call — the lanes bump from the home's lent tails; the mapped
    capacity grows only with the live bytes (plus at most one chunk per lane)."""
    text = CIRCUITS.get("cultivation_d5")
    if text is None:
        pytest.skip("example absent")
    s = _sampler(text.read_text())
    T, calls = 8, 60

    def gauges():
        pm = s.record_report()["plan_memory"]
        return pm["arena_chunk_bytes"], pm["arena_live_bytes"], pm["arena_free_bytes"]

    for c in range(calls):
        s.sample(4000, seed=c, threads=T)
        if c == 19:
            chunk_a, live_a, _ = gauges()
    chunk_b, live_b, _ = gauges()
    assert s.record_report()["threads"]["effective"] == T
    grown = chunk_b - chunk_a
    # the base (3052455) mapped ~1 MiB per lane per call here (4 MiB / call at T = 8): 40 calls => ~160 MiB
    assert grown <= (live_b - live_a) + (T + 1) * (1 << 20), (chunk_a, chunk_b, live_a, live_b)
