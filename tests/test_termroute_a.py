"""termroute-a (stage/termroute-a): the term route's per-column memo (F2) is owned by its sampler, never keyed on an
address; the cache bounds are byte-neutral.

* F2: build_k2's per-column memo (phi(M_c), free_dir_of(bare, M_c)) is a constant of ONE sampler's (group, bare,
  term view, record columns), so it lives in that sampler's FamRoute. A sampler constructed where a dead one lived
  (same heap address) must not see the dead one's memo: bytes of construct / sample / destroy cycles over two
  kind-2 circuits with different record columns equal fresh-process references (MUT-F2-STATIC goes RED).
* cache bound: a 1-byte budget (plan payloads + term-route dispatch entries; RecordFidelity held bytes) gives the
  same bytes as an unbounded one, really evicts, and keeps only live entries charged (MUT-CB-NOUNCHARGE,
  MUT-CB-RF-NOCLEAR go RED).
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import xtim

ROOT = Path(__file__).resolve().parents[1]


def _no_expect(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("PAULI_EXPECTATION")) + "\n"


_CH = _no_expect((ROOT / "tests" / "data" / "ch_cultivation.stim").read_text()).replace("0.001", "0.05")
_RECS = ("MPP !Z0\nMPP Z0\nMX 0\nM 0\nDETECTOR rec[-4]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
         "OBSERVABLE_INCLUDE(1) rec[-1]\n")
# two kind-2 circuits with the same record count whose columns differ off the terms: a fresh wire read in Z (a
# deterministic record, phi = 0) or in X (a coin, phi != 0) — a memo carried from one to the other mis-serves them
_FRESH = "R 9\nMPP {}9\nOBSERVABLE_INCLUDE(2) rec[-1]\n"
KIND2 = {"z_fresh": _CH + _RECS + _FRESH.format("Z"), "x_fresh": _CH + _RECS + _FRESH.format("X")}
SHOTS = 400

_FINGERPRINT = r'''
import hashlib, sys, warnings
import numpy as np
sys.path.insert(0, ROOT)
import xtim
warnings.simplefilter("ignore")
def fp(text, shots, seed, check=False):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    out = s.sample(shots, seed=seed)
    if check:
        st = s.record_report()["term_route"]
        assert st["shots_kind2"] > 0, st
    s = None                                   # freed here: the next sampler may take its address
    h = hashlib.sha256()
    for a in out:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()
'''


def _fresh(name):
    code = "ROOT = %r\n" % str(ROOT) + _FINGERPRINT + "print(fp(%r, %d, 5, True))\n" % (KIND2[name], SHOTS)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


def test_f2_memo_is_owned_by_its_sampler():
    ref = {k: _fresh(k) for k in KIND2}
    assert ref["z_fresh"] != ref["x_fresh"]
    ns = {"ROOT": str(ROOT)}
    exec(_FINGERPRINT, ns)
    bad = []
    for cycle in range(30):                    # alternate: a new sampler often lands on a dead one's address
        for k in ("z_fresh", "x_fresh") if cycle % 2 == 0 else ("x_fresh", "z_fresh"):
            if ns["fp"](KIND2[k], SHOTS, 5) != ref[k]:
                bad.append((cycle, k))
    assert not bad, f"bytes differ from the fresh-process references (a stale column memo): {bad}"


# ── cache bound (OD-2026-10-23-cache-bound): one default byte budget per object, plain drop-oldest eviction ──
_CH_OUT = _CH + "OUTPUT_QUBITS out 0\n"          # kind-2 shots, and a strict output contract (RecordFidelity)
UNBOUNDED = math.inf                           # m3: the one documented unbounded value


def _h(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _route_run(budget):
    s = xtim.compile_twirl_sampler(_CH_OUT, selfcheck=0, plan_payload_budget=budget)
    out = [_h(*s.sample(4000, seed=3))]
    b = s.sample_barrier(1500, seed=4)
    keys, counts = b.record_groups()
    out.append(_h(b.dets(), b.obs(), np.frombuffer(b"".join(bytes(k) for k in keys) or b"\0", np.uint8), counts,
                  b.record_group_ids(), b.frames()["plan_index"]))
    return out, s.record_report()


def test_route_keys_tiny_budget_byte_identical_and_bounded():
    ref, r0 = _route_run(UNBOUNDED)
    got, r1 = _route_run(1)
    assert got == ref, "bytes moved under a 1-byte budget (dispatch-entry eviction must be byte-neutral)"
    t0, t1 = r0["term_route"], r1["term_route"]
    assert t0["shots_kind2"] > 0 and t0["key_evictions"] == 0, t0
    assert t1["key_evictions"] > 0, t1                    # entries were dropped and re-created
    # what stays charged is only the live entries: under the tiny budget, fewer bytes than the unbounded run holds
    # (MUT-CB-NOUNCHARGE: dropped entries stay charged, so the re-created ones push it above)
    assert 0 < t1["key_bytes"] < t0["key_bytes"], (t0, t1)
    for k in ("shots_kind2", "plans_kind2", "shots_kind0_general", "plans_kind0_general", "shots_framed"):
        assert t1[k] == t0[k], (k, t0, t1)               # plan counters: a re-created entry is not a new plan
    assert r1["plans"] == r0["plans"] and r1["plan_misses"] == r0["plan_misses"]
    # the default-on budget skips a shot boundary only when nothing it reads moved: the payloads are still evicted
    # shot by shot, so the overshoot stays far below the unbounded run's payload total (MUT-CB-SKIP-BOUNDARY: one
    # boundary per call lets a whole call's payloads pile up)
    assert r1["plan_memory"]["evictions"] > 0
    assert 0 < r1["plan_memory"]["overshoot_bytes"] < r0["plan_memory"]["peak_bytes"] // 4, (r0["plan_memory"],
                                                                                        r1["plan_memory"])


def test_default_budget_reported():
    s = xtim.compile_twirl_sampler(_CH_OUT, selfcheck=0)
    s.sample(500, seed=1)
    assert s.record_report()["plan_memory"]["budget"] == 512 << 20
    assert xtim.RecordFidelity(s).stats()["cache_budget"] == 512 << 20


def test_record_fidelity_tiny_cache_budget_byte_identical_and_dropped():
    s = xtim.compile_twirl_sampler(_CH_OUT, selfcheck=0)
    A, B = s.sample_barrier(1500, seed=5), s.sample_barrier(1500, seed=6)
    ref = xtim.RecordFidelity(s, cache_budget=UNBOUNDED)
    fa, fb = np.asarray(ref.evaluate(A, keep_memo=True)), np.asarray(ref.evaluate(B, keep_memo=True))
    assert ref.stats()["cache_flushes"] == 0
    tiny = xtim.RecordFidelity(s, cache_budget=1)
    ga, gb = np.asarray(tiny.evaluate(A, keep_memo=True)), np.asarray(tiny.evaluate(B, keep_memo=True))
    assert ga.tobytes() == fa.tobytes() and gb.tobytes() == fb.tobytes()
    st = tiny.stats()
    assert st["cache_flushes"] >= 1, st                   # dropped at a call start (between calls only)
    fresh = xtim.RecordFidelity(s, cache_budget=UNBOUNDED)
    fresh.evaluate(B, keep_memo=True)
    # after the drop the evaluator holds B's contents alone: no more than a fresh evaluator of B (which also charges
    # the register's rest structure); MUT-CB-RF-NOCLEAR keeps A's contents too
    assert st["held_bytes"] <= fresh.stats()["held_bytes"], (st["held_bytes"], fresh.stats()["held_bytes"])
    assert ref.stats()["held_bytes"] > fresh.stats()["held_bytes"]   # non-vacuous: A holds contents B lacks


def test_record_fidelity_rest_structure_stays_charged():
    """termroute-a m4: the register's rest structure, built and charged at construction, is never dropped, so
    clear() and a cache flush leave it charged (held_bytes, which the max_bytes check also reads)."""
    s = xtim.compile_twirl_sampler(_CH_OUT, selfcheck=0)
    A = s.sample_barrier(800, seed=5)
    rf = xtim.RecordFidelity(s, cache_budget=UNBOUNDED)
    h0 = rf.stats()["held_bytes"]                         # bare register + rest structure, nothing evaluated yet
    rf.evaluate(A)
    hA = rf.stats()["held_bytes"]
    assert hA > h0
    rf.clear()
    assert rf.stats()["held_bytes"] == h0                 # was 16 * register only (the rest structure lost)
    tiny = xtim.RecordFidelity(s, cache_budget=1)
    tiny.evaluate(A)
    tiny.evaluate(A)                                      # flushed at the call start, then holds A's contents again
    assert tiny.stats()["cache_flushes"] >= 1
    assert tiny.stats()["held_bytes"] == hA


def test_record_fidelity_budget_none_default_inf_unbounded():
    s = xtim.compile_twirl_sampler(_CH_OUT, selfcheck=0)
    assert xtim.RecordFidelity(s, cache_budget=None).stats()["cache_budget"] == 512 << 20
    assert xtim.RecordFidelity(s, cache_budget=math.inf).stats()["cache_budget"] == math.inf
    assert xtim.RecordFidelity(s, cache_budget=4096).stats()["cache_budget"] == 4096


@pytest.mark.parametrize("bad", [0, -1, -math.inf, math.nan, 2.0])
def test_record_fidelity_cache_budget_refused_by_name(bad):
    s = xtim.compile_twirl_sampler(_CH_OUT, selfcheck=0)
    with pytest.raises(ValueError, match="cache_budget"):
        xtim.RecordFidelity(s, cache_budget=bad)
