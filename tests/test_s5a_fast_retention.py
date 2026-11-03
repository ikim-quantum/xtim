"""S5a: the barrier record (and the Born observable) on the abelian accumulator fast path.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

* T1 mechanism: on a warm replay (a second, seeded sample_barrier call on the same sampler, selfcheck
  spent) no shot enters the structural slow path: record_report()["structural_shots"] == 0 exactly,
  and no warm shot calls the plan cache's get_or_build (channel_report()["plan_hits"] does not move).
* T2 route neutrality on the barrier (S2b Q6's barrier counterpart): selfcheck >= shots forces every
  shot through the structural path; selfcheck = 0 serves warm shots on the fast path. Every record
  field is identical, on the compact and the legacy store. zflip_neg puts a -Z product wire in a
  residual CZ pair (the canonical strip's prefix Z-flip; mutation XTIM_MUTATE_S5A_NO_ZFLIP).
* T3 route attribution: the fast path counts with the structural path's table, so the Born-trie walks
  equal the kappa_rec > 0 record routes plus the clean walks on every surface.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from test_s2b_record_table import FIXTURES  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "s2b"))
from test_sigma_joint_law import FIXTURES as SIGMA_FIXTURES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
CULT_D5 = (Path(__file__).resolve().parent / "data" / "cultivation_d5.stim").read_text()
D3_RATE = (_ROOT / "benchmarks" / "cultivation_d3_rate.stim").read_text()

# A -Z product wire (wire 1: R then X = |1>) in the residual CZ(1, 2) that an X fault on wire 0 before
# CCZ 0 1 2 leaves; wire 2 is magic (|+> then T) and retained.
ZFLIP_NEG = ("R 0 1\nX 1\nRX 2\nT 2\nRX 3\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\n"
             "MPP X2*X3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
             "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 2 3\n")

ALL = dict(SIGMA_FIXTURES)
ALL.update(FIXTURES)
ALL["zflip_neg"] = ZFLIP_NEG


def _eq(a, b):
    if isinstance(a, tuple):
        return all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


# ── J-b2 J1b: the PRODUCTION leg (plan r2.1 J1b; review MINOR-4: the T1 mechanism reads record_report's
# structural_shots / plan_misses, not the legacy channel_report's plan_hits). Production's barrier is the layout-3
# record; its fields are compared field by field between the fast (warm, selfcheck 0) and the structural
# (selfcheck >= shots: the first run's window) routes, and under a payload budget.
from _public import public_sampler  # noqa: E402


def _pmisses(r):
    return r["plan_misses"] if isinstance(r["plan_misses"], int) else sum(r["plan_misses"].values())


@pytest.mark.parametrize("name,text,n", [("cult_d5", CULT_D5, 2000), ("d3_rate", D3_RATE, 4000)],
                         ids=["cult_d5", "d3_rate"])
def test_warm_barrier_replay_never_enters_the_structural_path_public(name, text, n):
    s = public_sampler(text)
    s.sample_barrier(n, seed=7)                               # cold: plan builds (structural path)
    cold = s.record_report()
    assert _pmisses(cold) > 0                                 # non-vacuous: the cold run built plans
    s.sample_barrier(n, seed=7)                               # warm replay of the same shots
    r = s.record_report()
    assert r["structural_shots"] == 0, r
    assert _pmisses(r) == _pmisses(cold), (r["plan_misses"], cold["plan_misses"])   # no plan build
    assert not any(k.startswith("exact") for k in r["route_counts"]), r["route_counts"]


def _barrier_fields_public(b, n):
    out = {"dets": np.asarray(b.dets()), "obs": np.asarray(b.obs()), "dec": np.asarray(b.decisions())}
    out["outcomes"] = [bytes(np.asarray(b.outcomes(i), np.uint8)) for i in range(n)]
    out["record_outcomes"] = [bytes(np.asarray(b.record_outcomes(i), np.uint8)) for i in range(n)]
    out["plan_content"] = [repr(b.plan_content(i)) for i in range(n)]
    out["groups"] = tuple(np.asarray(x) for x in b.record_groups())
    out["group_ids"] = np.asarray(b.record_group_ids())
    out["keys"] = np.asarray(b.record_keys())
    return out


def _barrier_public(text, selfcheck, n, budget=None):
    s = public_sampler(text, selfcheck=selfcheck)
    if budget is not None:
        s.set_plan_payload_budget(budget)
    if selfcheck == 0:
        s.sample(n, seed=11)
    b = s.sample_barrier(n, seed=7)
    return s, _barrier_fields_public(b, n)


@pytest.mark.parametrize("name", sorted(ALL))
def test_barrier_fast_route_equals_structural_route_public(name):
    text, n = ALL[name], 1500
    sf, fast = _barrier_public(text, 0, n)
    ss, slow = _barrier_public(text, 10 ** 6, n)
    for k in fast:
        assert _eq(fast[k], slow[k]), (name, k)


@pytest.mark.parametrize("name", ["zflip_neg", "m3_obs", "a4_minus_z"])
def test_barrier_fast_route_under_a_payload_budget_public(name):
    text, n = ALL[name], 1500
    _, ref = _barrier_public(text, 10 ** 6, n)
    s0, _ = _barrier_public(text, 0, n)
    peak = s0.record_report()["plan_memory"]["peak_bytes"]
    s, got = _barrier_public(text, 0, n, budget=max(1, 3 * peak // 4))
    for k in ref:
        assert _eq(got[k], ref[k]), (name, k)
    r = s.record_report()
    assert r["plan_memory"]["evictions"] > 0, r["plan_memory"]
    assert r["structural_shots"] > 0, r["structural_shots"]


def test_fast_route_is_reached_on_the_neutrality_fixtures_public():
    for name in ("zflip_neg", "m3_obs", "m3_det", "a4_minus_z", "ccz_obs"):
        s = public_sampler(ALL[name])
        s.sample(1500, seed=11)
        s.sample_barrier(1500, seed=7)
        r = s.record_report()
        faulted = r["used"] - r["route_counts"]["clean"]
        assert r["structural_shots"] < faulted, (name, r["structural_shots"], faulted)


@pytest.mark.parametrize("name", ["m3_obs", "m3_obs_dec", "m3_det", "q3_dec"])
@pytest.mark.parametrize("mode", ["sample", "barrier"])
def test_trie_walks_equal_the_kappa_pos_routes_public(name, mode):
    """T3 on production: the Born-trie walks equal the kappa_rec > 0 record routes plus the clean walks."""
    from _public import identity_kappa_rec
    s = public_sampler(FIXTURES[name])
    clean_walks = identity_kappa_rec(FIXTURES[name]) > 0
    for _ in range(2):                                       # the second call is warm
        if mode == "sample":
            s.sample(3000, seed=5)
        else:
            s.sample_barrier(3000, seed=5)
        r = s.record_report()
        rc = r["route_counts"]
        kpos = rc["record_kappa_pos"] + rc["retain_kappa_pos"]
        assert r["born_trie"]["walks"] == kpos + (rc["clean"] if clean_walks else 0), (name, mode, rc, r["born_trie"])
    if name.startswith("m3_obs"):
        assert kpos > 0 and r["structural_shots"] == 0, r    # warm: the Born observable is served fast
