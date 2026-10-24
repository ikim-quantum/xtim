"""S0(c) / D6: content-derived plan identity, PAYLOAD-only eviction, caller payload budget.

Claims under test (design S0(c)):
  * every output byte is identical for EVERY payload budget (None = unbounded, the smallest
    non-zero budget 1 byte = evict everything at every shot boundary, and in between) — on
    sample(), sample_barrier() (legacy and compact record stores: plan keys, record groups,
    frames' plan ids) and the record-fidelity surfaces that read them;
  * plan ids are stable across evict/rebuild: record_groups() partitions and frames() plan ids
    do not move (a rebuilt payload keeps its permanent header and id);
  * a budget of 0 (or negative) is refused by name; None is unbounded;
  * eviction really happens under a small budget (evictions, payload rebuilds > 0) and the
    resident bytes respect the budget between calls; the transient overshoot is reported;
  * TWPL stores KEYS only: a TWPL-warm sampler loads the headers (no law builds: plan_misses 0)
    and rebuilds every payload on first use — bytes identical to cold.
"""
import hashlib
import math
import os
import pathlib

import numpy as np
import pytest

import xtim
import xtim._xtim as _x
from _private import private_fixture, private_text

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EX = _ROOT / "tests" / "data"
_DATA = _ROOT / "tests" / "data"

_M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
       "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
       "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
_DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
_M3_OBS_DEC = (_M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET +
               "MX 0\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1 2\n")

_CIRCUITS = {
    "cultivation_d3_rate": (_EX / "cultivation_d3_rate.stim").read_text(),
    "ch_cultivation (PPR)": (_EX / "ch_cultivation.stim").read_text(),
    "born5_dec": (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text(),
    "m3_obs_dec": _M3_OBS_DEC,
    "ccz_factory_a (kappa>0)": private_text("ccz_factory_a.stim"),   # private: None when absent (the tests skip)
}
_SHOTS = {"ccz_factory_a (kappa>0)": (600, 200)}


def _h(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


_DEFAULT = "default"     # termroute-a: the engine default budget (512 MiB); math.inf = unbounded (set explicitly)


def _fingerprint(text, budget, disk=""):
    ns, nb = _SHOTS.get(next((k for k, v in _CIRCUITS.items() if v == text), ""), (3000, 600))
    s = _x.TwirlSampler(text, 1.0, disk, 0)
    if budget != _DEFAULT:
        s.set_plan_payload_budget(budget)
    out = {"sample": _h(*s.sample(ns, seed=11))}
    # J-b1 F: the layout-3 barrier (always compact; CH / PPR retention served) — every record surface
    b = s.sample_barrier(nb, seed=7)
    keys, counts = b.record_groups()
    pc = b"".join(bytes(b.plan_content(i)[0]) for i in range(nb))
    F = b.frames()
    out["barrier"] = _h(b.dets(), b.obs(), b.decisions(), np.frombuffer(b"".join(bytes(k) for k in keys) or b"\0",
                        np.uint8), counts, b.record_group_ids(), np.frombuffer(pc or b"\0", np.uint8),
                        F["plan_index"], F["prefix_words"])
    return out, s.record_report()


@pytest.mark.parametrize("name", list(_CIRCUITS))
def test_bytes_identical_for_every_budget(name):
    text = _CIRCUITS[name]
    if text is None:
        private_fixture("ccz_factory_a.stim")
    ref, rr0 = _fingerprint(text, math.inf)
    assert rr0["plan_memory"]["budget"] == math.inf
    assert rr0["plan_memory"]["evictions"] == 0
    for budget in (1, 4096, 1 << 20, _DEFAULT):
        got, rr = _fingerprint(text, budget)
        assert got == ref, f"{name}: bytes moved under plan_payload_budget={budget}"
        pm = rr["plan_memory"]
        if budget == _DEFAULT:
            budget = 512 << 20                           # termroute-a: the default bounds every sampler
        assert pm["budget"] == budget
        # between calls the resident payloads honour the budget
        assert pm["resident_bytes"] <= budget, pm
        # identity is content: a rebuild never creates a new plan (headers are permanent)
        assert rr["plans"] == rr0["plans"] and rr["plan_misses"] == rr0["plan_misses"]
        assert rr["term_route"]["plans_kind2"] == rr0["term_route"]["plans_kind2"]   # J-b1 F: was ppr_plans


def test_small_budget_really_evicts_and_reports_overshoot():
    text = _CIRCUITS["cultivation_d3_rate"]
    _, rr = _fingerprint(text, 1)
    pm = rr["plan_memory"]
    assert pm["evictions"] > 0 and pm["payload_builds"] > 0, pm
    assert pm["overshoot_bytes"] > 0 and pm["resident_bytes"] == 0, pm
    assert pm["header_bytes"] > 0


@pytest.mark.parametrize("bad", [0, -5, -math.inf, math.nan, 1.5])
def test_zero_or_negative_budget_refused_by_name(bad):
    s = _x.TwirlSampler(_CIRCUITS["cultivation_d3_rate"], 1.0, "", 0)
    with pytest.raises(ValueError, match="plan_payload_budget"):
        s.set_plan_payload_budget(bad)
    with pytest.raises(ValueError, match="plan_payload_budget"):
        xtim.compile_twirl_sampler(_CIRCUITS["cultivation_d3_rate"], plan_payload_budget=bad)


def test_none_is_the_default_and_inf_is_unbounded():
    """termroute-a m3: one meaning everywhere — None = the 512 MiB default, math.inf = unbounded (reported as
    math.inf), for set_plan_payload_budget, compile_twirl_sampler(plan_payload_budget=) and RecordFidelity."""
    s = _x.TwirlSampler(_CIRCUITS["cultivation_d3_rate"], 1.0, "", 0)
    s.set_plan_payload_budget(1)
    s.set_plan_payload_budget(None)
    assert s.record_report()["plan_memory"]["budget"] == 512 << 20
    s.set_plan_payload_budget(math.inf)
    assert s.record_report()["plan_memory"]["budget"] == math.inf
    for b, want in ((None, 512 << 20), (math.inf, math.inf), (4096, 4096)):
        t = xtim.compile_twirl_sampler(_CIRCUITS["cultivation_d3_rate"], selfcheck=0, plan_payload_budget=b)
        assert t.record_report()["plan_memory"]["budget"] == want, b


def test_compile_twirl_sampler_forwards_budget():
    t = xtim.compile_twirl_sampler(_CIRCUITS["cultivation_d3_rate"], selfcheck=0,
                                   plan_payload_budget=1)
    t.sample(2000, seed=3)
    assert t.record_report()["plan_memory"]["budget"] == 1


@pytest.mark.parametrize("name", ["cultivation_d3_rate", "born5_dec", "ccz_factory_a (kappa>0)"])
def test_twpl_keys_only_round_trip(name, tmp_path):
    text = _CIRCUITS[name]
    if text is None:
        private_fixture("ccz_factory_a.stim")
    disk = str(tmp_path / "plans.twpl")
    cold, rr_cold = _fingerprint(text, math.inf, disk)       # writes the file (keys only)
    assert os.path.getsize(disk) > 0
    warm, rr_warm = _fingerprint(text, math.inf, disk)       # loads the keys, rebuilds payloads
    assert warm == cold, f"{name}: TWPL-warm bytes differ from cold"
    assert rr_warm["plans"] == rr_cold["plans"]
    assert sum(rr_warm["plan_misses"].values()) == 0, rr_warm["plan_misses"]
    assert rr_warm["plan_memory"]["payload_builds"] == rr_cold["plans"]
    # keys-only + a budget: still identical
    warm_b, _ = _fingerprint(text, 1, disk)
    assert warm_b == cold


# ── r6.4 (code review MINOR-4, review R6-2): the readers' record-table memo is bounded and charged
# to the SAME payload budget; reader tables are evicted first, sampler payloads only after.
_READER_TEXT = _M3_OBS_DEC


def _materialize_digest(b, shots):
    h = hashlib.sha256()
    for i in shots:
        bf = b.branch_frames(int(i))                 # J-b1 F: materialize(i).branch_frames()
        for k in sorted(bf):
            if k != "n":
                h.update(np.ascontiguousarray(np.asarray(bf[k])).tobytes())
    return h.hexdigest()


def _distinct_plan_shots(b):
    pid = np.asarray(b.frames()["plan_index"])       # J-b1 F: the export-local plan index (no exact class)
    first = {}
    for i in range(len(pid)):
        first.setdefault(int(pid[i]), i)
    return [first[k] for k in sorted(first)]


def test_reader_memo_bounded_charged_and_value_neutral():
    """(a) a reader-only workload under a tiny budget: every table is charged (reader_bytes > 0 at
    the peak), unheld tables are evicted, nothing outlives the budget beyond the one table in use, and
    the materialized states equal the unbounded sampler's."""
    ref_s = _x.TwirlSampler(_READER_TEXT, 1.0, "", 0)
    rb = ref_s.sample_barrier(800, seed=4, )
    shots = _distinct_plan_shots(rb)
    assert len(shots) > 3, shots
    ref = _materialize_digest(rb, shots)
    pm0 = ref_s.record_report()["plan_memory"]
    # RE-PINNED 2026-10-20 (J-b1 F): resident plans are reused by the readers (P12 F3), so the unbounded sampler's
    # reader memo holds at most the non-resident contents' tables and never evicts
    assert pm0["reader_evictions"] == 0 and pm0["reader_tables"] <= len(shots), pm0

    s = _x.TwirlSampler(_READER_TEXT, 1.0, "", 0)
    s.set_plan_payload_budget(1)
    b = s.sample_barrier(800, seed=4, )
    assert _materialize_digest(b, shots) == ref
    pm = s.record_report()["plan_memory"]
    assert pm["reader_peak_bytes"] > 0, pm                       # charged (R64 red)
    assert pm["reader_evictions"] >= len(shots) - 1, pm          # every older unheld table evicted
    assert pm["reader_tables"] <= 1, pm                          # only the last one survives


def test_one_table_over_budget_is_reported_not_refused():
    """(b) the payload budget's contract (overshoot reported, never refused) holds for the readers:
    a table larger than the whole budget is served and its excess reported."""
    s = _x.TwirlSampler(_READER_TEXT, 1.0, "", 0)
    s.set_plan_payload_budget(1)
    b = s.sample_barrier(400, seed=4, )
    shots = _distinct_plan_shots(b)
    b.materialize(int(shots[-1]))
    pm = s.record_report()["plan_memory"]
    assert pm["reader_overshoot_bytes"] > 0, pm


def _run_counters(s):
    pm = s.record_report()["plan_memory"]
    return {k: pm[k] for k in ("resident_bytes", "peak_bytes", "overshoot_bytes", "payload_builds", "evictions")}


@pytest.mark.parametrize("budget", [None, 1, 20000])
def test_readers_between_runs_leave_run_counters(budget):
    """(R6-2) run() -> readers -> run() leaves run()'s exact payload counters exactly as run() -> run():
    the readers' unheld tables are evicted before any sampler payload."""
    def make():
        s = _x.TwirlSampler(_READER_TEXT, 1.0, "", 0)
        if budget is not None:
            s.set_plan_payload_budget(budget)
        return s
    a = make()
    a.sample_barrier(600, seed=4, )
    a.sample(2000, seed=5)
    ctrl = _run_counters(a)
    r = make()
    br = r.sample_barrier(600, seed=4, )
    for i in _distinct_plan_shots(br):                           # reader activity in between
        br.materialize(int(i))
    br.frames()
    pm = r.record_report()["plan_memory"]
    # RE-PINNED 2026-10-20 (J-b1 F, layout 3): the readers resolve a kind-0 content to the sampler's RESIDENT plan
    # first (P12 fixplan-p12-frames F3: no reader rebuild of a plan the run built), so with the payloads resident
    # (budget None / 20000) the reader memo is not touched; under budget 1 the payloads are evicted and the readers
    # go through the memo (non-vacuous).
    if budget == 1:
        assert pm["reader_tables"] >= 1 and pm["reader_lookups"] > 0, pm
    else:
        assert pm["reader_tables"] <= 1 and pm["reader_evictions"] == 0, pm
    r.sample(2000, seed=5)
    assert _run_counters(r) == ctrl, (_run_counters(r), ctrl)
    if budget == 1:                    # unheld reader tables never outlive the budget beyond the one table in use
        assert r.record_report()["plan_memory"]["reader_tables"] <= 1


def test_buffer_held_tables_never_evict_sampler_payloads():
    """r8 (final review MAJOR-1; coordinator ruling): tables a buffer holds sit OUTSIDE the plan payload
    budget, and readers never evict a sampler payload (R6-2). At a budget the sampler alone fits under
    (its peak + 64 B), a compact buffer kept alive after frames() leaves run()'s payload builds exactly
    at the no-reader control (it was 4 -> 10,365 at c8d1ff5)."""
    s0 = _x.TwirlSampler(_READER_TEXT, 1.0, "", 0)
    s0.sample(20000, seed=5)
    B = s0.record_report()["plan_memory"]["peak_bytes"] + 64
    builds = {}
    for hold in (False, True):
        s = _x.TwirlSampler(_READER_TEXT, 1.0, "", 0)
        s.set_plan_payload_budget(B)
        br = s.sample_barrier(600, seed=4, )
        if hold:
            br.frames()      # J-b1 F: a layout-3 buffer holds records only; its tables live in the reader memo
        else:
            del br
        p0 = s.record_report()["plan_memory"]["payload_builds"]
        s.sample(20000, seed=5)
        builds[hold] = s.record_report()["plan_memory"]["payload_builds"] - p0
    assert builds[True] == builds[False], builds
