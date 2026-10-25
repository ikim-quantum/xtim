"""S0(a) of the twirl consolidation: ``TwirlSampler.record_report()`` — route + memo/miss counters.

Contracts pinned here:
  * every shot takes exactly ONE route: ``route_counts`` sums to ``used``, and the ``exact_*``
    routes sum to the per-shot exact-fallback count (``channel_report()["exact_shots"]``);
  * the counters are EXACT at a fixed seed: two fresh samplers, and two separate processes,
    report identical dicts; a seeded call resets them (like every run counter);
  * each exact-fallback trigger raises ONE ``xtim.XtimPerformanceWarning`` per run in which it
    fired (S0d: after the run returns; formerly a stderr line) — including the formerly SILENT
    non-commuting PPR route;
  * plan misses split by the built plan's class sum to ``channel_report()["plan_misses"]``;
  * the counters are reads only: sampled bytes are identical to the same seed's stream.
Route attribution per trigger is checked on fixtures whose trigger is known by construction.

RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-MEMCTR / M-API — the family sampler has no exact route (no
``exact_*`` / ``ppr`` key, no performance warning); channel_report is gone (its used / plans / plan_misses live in
record_report). ``route_counts["fast"]`` stays the DERIVED bucket (used minus every counted route), so the
term-route shots of a CH / PPR circuit land in it; ``record_report()["term_route"]`` counts them (cumulative since
construction). The cultivation d5 counts are unchanged (diagonal).
"""
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import xtim
import xtim._xtim as _x

_ROOT = Path(__file__).resolve().parents[1]
_DATA = Path(__file__).resolve().parent / "data"

# Born observable on kappa>0 plans. S2b: the observable is a record row — its kappa>0 plans take the
# record route (record_kappa_pos / retain_kappa_pos), never an exact route.
_M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
       "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
       "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
_DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
M3_OBS = _M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET + "OUTPUT_QUBITS out 0 1 2\n"
M3_DET = _M3 + "MX 3\nMX 4\n" + _DET + "OUTPUT_QUBITS out 0 1 2\n"          # kappa>0 record path
Q3_DEC = ("R 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\nMY 0\nMPP Z0\nMPP Z0\nH 3\nT 3\nX_ERROR(0.5) 2\n"
          "CCZ 0 3 2\nDECISION(0) rec[-3]\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\nOUTPUT_QUBITS out 2 3\n")


def _mirror_ppr(seed, n=4):
    """adv-A fuzz family (Clifford-T-Clifford mirror with noise): PPR atoms, some non-commuting."""
    r = random.Random(seed)

    def rc(d):
        g = []
        for _ in range(d):
            k = r.randrange(4)
            if k == 0:
                g.append(("H", [r.randrange(n)]))
            elif k == 1:
                g.append(("S", [r.randrange(n)]))
            else:
                a, b = r.sample(range(n), 2)
                g.append(("CX" if k == 2 else "CZ", [a, b]))
        return g
    inv = {"H": "H", "S": "S_DAG", "CX": "CX", "CZ": "CZ", "T": "T_DAG"}

    def emit(gs, p):
        out = []
        for name, q in gs:
            out.append(f"{name} " + " ".join(map(str, q)))
            if p:
                out.append(f"DEPOLARIZE{len(q)}({p}) " + " ".join(map(str, q)))
        return out
    C1, C2 = rc(8), rc(8)
    T = [("T", [q]) for q in r.sample(range(n), r.randrange(1, n + 1))]
    dag = lambda gs: [(inv[a], q) for a, q in reversed(gs)]
    L = ["R " + " ".join(map(str, range(n)))]
    L += emit(C1, 0.02) + emit(T, 0) + emit(C2, 0.02) + emit(dag(C2), 0.02) + emit(dag(T), 0)
    L += emit(dag(C1), 0.02)
    L.append("M " + " ".join(map(str, range(n))))
    L += [f"DETECTOR rec[-{i + 1}]" for i in range(n)]
    L.append("DETECTOR " + " ".join(f"rec[-{i + 1}]" for i in range(n)))
    return "\n".join(L) + "\n"


PPR23 = _mirror_ppr(23)          # both non-commuting PPR routes fire (alt and composed)

def _sampler(text):
    return _x.TwirlSampler(text, 1.0, "", 0)


def _check_sums(s):
    r = s.record_report()
    rc = r["route_counts"]
    assert not [k for k in rc if k.startswith("exact") or k == "ppr"], rc     # no legacy route exists
    assert all(v >= 0 for v in rc.values()), rc
    assert sum(rc.values()) == r["used"]
    pm = r["plan_misses"]
    assert pm["kappa0"] + pm["kappa_pos"] + pm["fallback"] == r["counters"]["plan_misses"]
    return r


@pytest.mark.parametrize("text,mode,want", [
    (M3_OBS, "sample", {"record_kappa_pos", "slow_kappa0"}),
    # J-b1 F: a barrier batch is the FAM recording run (layout 3), on the sample-mode routes (P4: fast / accumulator
    # paths kept); the legacy retain_* routes are not taken
    (M3_OBS, "barrier", {"record_kappa_pos", "fast_accumulator"}),
    # S2b: m3_det's kappa_sigma direction is read by no record (kappa_rec = 0): the fast routes
    (M3_DET, "sample", {"fast"}),
    (M3_DET, "barrier", {"fast_accumulator"}),
    (Q3_DEC, "sample", {"fast_accumulator"}),
    (PPR23, "sample", {"fast"}),          # J-b1 F: the formerly exact PPR shots, served on the term route
])
def test_routes_sum_and_attribute(text, mode, want):
    s = _sampler(text)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        if mode == "sample":
            s.sample(4000, seed=11)
        else:
            s.sample_barrier(2000, seed=7, compact=True)
    r = _check_sums(s)
    rc = r["route_counts"]
    for k in want:
        assert rc[k] > 0, (k, rc)
    # S2a: the kappa chain and its counters are deleted; every kappa_rec>0 route walks the plan's
    # record trie — S2b: including CLEAN shots when the identity plan has a Born record (kappa_rec >
    # 0 at R = I, e.g. a Born observable); kappa_rec-free fixtures walk none.
    kpos = sum(rc[k] for k in ("record_kappa_pos", "retain_kappa_pos"))
    assert "kappa_chain" not in r                  # S2a: the chain and its counters are deleted
    # the identity plan's content: every shot of a noiseless run (layout 3: plan_structure takes plan content)
    ident = bytes(_x.TwirlSampler(text, 0.0, "", 0).sample_barrier(1, seed=1).plan_content(0)[0])
    clean_walks = rc["clean"] if sum(s.plan_structure(ident)["kappa_rec"]) > 0 else 0
    assert r["born_trie"]["walks"] == kpos + clean_walks
    assert r["born_trie"]["steps"] >= kpos
    # No exact route, no performance warning (J-b1 F); the term route serves the PPR circuit.
    assert not [w for w in rec if issubclass(w.category, xtim.XtimPerformanceWarning)]
    if text is PPR23:
        assert r["term_route"]["shots_kind2"] > 0 and r["term_route"]["plans_kind2"] > 0


def test_fast_path_is_derived_and_cult_d5_counts():
    s = _sampler((Path(__file__).resolve().parent / "data" / "cultivation_d5.stim").read_text())
    s.sample(20000, seed=11)
    r = _check_sums(s)
    rc = r["route_counts"]
    # Pinned exact counts (fixed seed): hot loop, clean shots, cold misses (= plan builds here).
    # Re-pinned at S0(b) (keyed streams, ENGINE_VERSION xtim-engine-11): was (14318, 647, 5035).
    assert (rc["fast"], rc["clean"], rc["slow_kappa0"]) == (14336, 569, 5095)
    assert r["plan_misses"] == {"kappa0": 5095, "kappa_pos": 0, "fallback": 0}


# S2b: a Born DECISION on kappa>0 plans walks the record trie (q3_dec reads no Born direction).
M3_DEC = _M3 + "MX 3\nDECISION(0) rec[-1]\n" + _DET + "OUTPUT_QUBITS out 0 1 2\n"


def test_counters_deterministic_and_reset_by_seeded_call():
    a, b = _sampler(M3_DEC), _sampler(M3_DEC)
    a.sample(3000, seed=5)
    b.sample(3000, seed=5)
    ra = a.record_report()
    assert ra == b.record_report()
    a.sample(3000, seed=5)                      # seeded call: run counters restart
    rb = a.record_report()
    assert rb["used"] == sum(rb["route_counts"].values()) == 3000
    # Same shots, same triggers; only the cold/warm split moves (the plan cache stays warm).
    exact = lambda r: {k: v for k, v in r["route_counts"].items() if k == "clean"}
    assert exact(rb) == exact(ra)
    assert "memo_nodes" not in rb                # S2b: the Born memos are deleted
    # S2a: same walks and steps; the trie (plan payload) stays warm, so no node is re-evaluated.
    assert (rb["born_trie"]["walks"], rb["born_trie"]["steps"]) == (ra["born_trie"]["walks"],
                                                                    ra["born_trie"]["steps"])
    assert ra["born_trie"]["node_evals"] > 0 and rb["born_trie"]["node_evals"] == 0
    assert rb["route_counts"]["fast_accumulator"] > ra["route_counts"]["fast_accumulator"]


def _subprocess_report(text):
    code = ("import json, sys, warnings, xtim._xtim as x\n"
            "s = x.TwirlSampler(sys.stdin.read(), 1.0, '', 0)\n"
            "with warnings.catch_warnings(record=True) as rec:\n"
            "    warnings.simplefilter('always')\n"
            "    s.sample(3000, seed=9); s.sample(2000)\n"
            "print(json.dumps(s.record_report(), sort_keys=True))\n"
            "print(json.dumps([[w.category.__name__, str(w.message)] for w in rec]))\n")
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    p = subprocess.run([sys.executable, "-c", code], input=text, capture_output=True, text=True,
                       env=env, check=True)
    rep, warned = p.stdout.splitlines()[-2:]
    return json.loads(rep), json.loads(warned)


def test_counters_identical_across_processes():
    r1, w1 = _subprocess_report(PPR23)
    r2, w2 = _subprocess_report(PPR23)
    assert r1 == r2 and w1 == w2
    # J-b1 F: the PPR circuit's formerly exact shots are law-served — no XtimPerformanceWarning at all.
    assert not [m for c, m in w1 if c == "XtimPerformanceWarning"], w1


def test_counters_do_not_touch_the_streams():
    s = _sampler(PPR23)
    d1, o1 = s.sample(3000, seed=4)
    s.record_report()
    d2, o2 = s.sample(3000, seed=4)
    assert np.array_equal(np.asarray(d1), np.asarray(d2))
    assert np.array_equal(np.asarray(o1), np.asarray(o2))


def test_partitioned_sampler_record_report_sums_the_patterns():
    """J-b1 F (plan §1.7, ruled): PartitionedTwirlSampler.record_report() is the reference pattern's static schema
    plus route_counts / counters summed over the inner samplers built so far, plus `patterns` (it was refused)."""
    text = "INPUT_BITS b 1\nR 0\nIF b[0] {\nX 0\n}\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    assert isinstance(s, xtim.twirl.PartitionedTwirlSampler)
    s.sample(400, seed=3, decision_bits=np.array([[0], [1]] * 200, np.uint8))
    r = s.record_report()
    assert r["patterns"] == 2 and sum(r["route_counts"].values()) == 400
    assert r["classes"] == s._ref_sampler.record_report()["classes"]
