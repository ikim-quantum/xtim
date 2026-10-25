"""S2a (twirl consolidation): the per-plan Born trie replaces the kappa chain, byte-identically.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

Gates pinned here (design.final.md §5.1 S2a):
  * the chain is gone (no kappa_chain key), and a trie walk takes EXACTLY kappa steps per walked shot;
  * node counts == the distinct (plan, z-prefix) set recorded in the barrier records (exact);
  * every stored node equals its FROM-SCRATCH definition bit for bit (incremental == from scratch),
    including after payload eviction and rebuild, AND the deleted kappa chain's semantics on the
    n-qubit state (bare copy + measure_pauli of the derived directions + born_p1);
  * the n-qubit kernel logical DERIVED from the compact free-register form equals the builder's
    representative exactly, phase included, on every kappa>0 plan;
  * node work is linear in (support reached)*j: passes <= node_evals * max_depth, entries touched
    <= passes * max_support;
  * bytes are independent of the trie's warmth (fresh cache vs warm cache vs smallest budget).
"""
from pathlib import Path

import numpy as np
import pytest

import xtim
import xtim._xtim as _x

DATA = Path(__file__).resolve().parent / "data"
CUBES = sorted((DATA / "dem_factored_law").glob("v2_777_01[24]9_cube.stim"))

M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
      "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
      "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
M3_DET = M3 + "MX 3\nMX 4\nR 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1 2\n"
# S2b: the record trie walks only the directions a RECORD reads (kappa_rec). m3_det reads none (its
# kappa_sigma direction is unread: no walk at all, gate C-3); m3_obs's Born observable reads one.
M3_OBS = M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\nMX 4\nR 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1 2\n"


def _texts():
    return [("m3_obs", M3_OBS)] + [(p.name, p.read_text()) for p in CUBES]


# ── decision 7: the large-chi INFO event (one logging INFO on xtim.twirl per call) ─────────────
import logging  # noqa: E402

KAPPA0 = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"


def _born_records(caplog):
    return [r for r in caplog.records
            if r.name == "xtim.twirl" and "Born-trie node evaluation" in r.getMessage()]


def test_large_chi_info_event_is_logging_not_a_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # an INFO record never raises
        import xtim._xtim as _xe
        _xe.TwirlSampler(M3_OBS, 1.0, "", 0).sample(2000, seed=3)   # the public engine object


# ── J-b2 J1b: the PRODUCTION leg (plan r2.1 J1b). The production engine's record trie (the diagonal cache's,
# the same trie the legacy sampler walked) on the family sampler in the production configuration (test API
# compile_family_sampler(classes="all"): FAMP == NEW, with the trie oracles _born_trie_check / _born_trie_oracle, m6) and on the
# public sampler (record_report counters, bytes). Prefixes from the layout-3 records: plan_structure(i)'s rho /
# kappa_rec, outcomes(i) = (u: rho coins, z: kappa_rec Born bits). The sigma builder's kernel representative
# (test_derived_lambda_equals_builder_rep) has no production counterpart (the run path's table is recB): retired
# with the sigma builder (jb2 mapping).
import warnings as _w  # noqa: E402

from _public import public_sampler  # noqa: E402
from _private import private_fixture  # noqa: E402


def _family(text, budget=None):
    from xtim import _xtim_testapi as TA
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        s = TA.compile_family_sampler(text, selfcheck=0)   # the production configuration
    if budget is not None:
        s.set_plan_payload_budget(budget)
    return s


def _recorded_prefixes_l3(s, shots, seed):
    b = s.sample_barrier(shots, seed=seed)
    seen, steps, walks = set(), 0, 0
    for i in range(shots):
        ps = b.plan_structure(i)
        k, r = int(np.asarray(ps["kappa_rec"])[0]), int(np.asarray(ps["rho"])[0])
        if k == 0:
            continue
        z = bytes(np.asarray(b.outcomes(i), np.uint8))[r:r + k]
        walks += 1
        steps += len(z)
        key = bytes(ps["plan_content"])
        for j in range(len(z)):
            seen.add((key, z[:j]))
    return b, seen, walks, steps


@pytest.mark.parametrize("name,text", _texts())
def test_steps_and_node_counts_exact_family(name, text):
    s = _family(text)
    _, seen, walks, steps = _recorded_prefixes_l3(s, 3000, 7)
    bt = s.record_report()["born_trie"]
    assert walks > 0
    assert (bt["walks"], bt["steps"]) == (walks, steps)
    # S8-4: a root node is evaluated OR served by the per-trie root memo (the same double): evals + memo hits = nodes
    assert bt["node_evals"] + bt["root_memo_hits"] == bt["nodes"] == len(seen)
    before = s.record_report()["born_trie"]
    assert tuple(s.born_trie_check()) == (len(seen), 0)          # incremental == from scratch
    assert s.record_report()["born_trie"] == before
    assert tuple(s.born_trie_oracle()) == (len(seen), 0)         # == the n-qubit chain semantics


@pytest.mark.parametrize("name,text", _texts())
def test_node_work_linear_in_chi_j_public(name, text):
    s = public_sampler(text)
    s.sample(4000, seed=3)
    bt = s.record_report()["born_trie"]
    chi = int(len(xtim.bare_state_of(text).branch_frames()["coeff"]))
    assert bt["passes"] >= bt["node_evals"] > 0
    assert bt["passes"] <= bt["node_evals"] * bt["max_depth"]
    assert bt["chi_j"] <= bt["passes"] * bt["max_support"] and bt["max_support"] <= chi


def _digest_l3(s, mode, shots, seed):
    if mode == "sample":
        out = s.sample(shots, seed=seed)
        return b"".join(np.ascontiguousarray(np.asarray(a)).tobytes() for a in out)
    b = s.sample_barrier(shots, seed=seed)
    return np.asarray(b.dets()).tobytes() + b"".join(bytes(np.asarray(b.outcomes(i), np.uint8)) for i in range(shots))


@pytest.mark.parametrize("mode", ["sample", "barrier"])
def test_bytes_independent_of_trie_warmth_and_eviction_public(mode):
    text = M3_OBS
    s0 = public_sampler(text)
    cold = _digest_l3(s0, mode, 3000, 21)
    r0 = s0.record_report()
    s = public_sampler(text)
    _digest_l3(s, mode, 3000, 5)
    warm = _digest_l3(s, mode, 3000, 21)
    t1 = public_sampler(text)
    t1.set_plan_payload_budget(1)
    tiny = _digest_l3(t1, mode, 3000, 21)
    t = _family(text, budget=max(1, 2 * r0["plan_memory"]["resident_bytes"] // 3))
    part = _digest_l3(t, mode, 3000, 21)
    assert cold == warm == tiny == part
    rt = t.record_report()
    assert rt["plan_memory"]["evictions"] > 0 and rt["plan_memory"]["payload_builds"] > 0
    checked, bad = t.born_trie_check()
    assert checked > 0 and bad == 0


def test_factory_trie_deep_nodes_family():
    s = _family(private_fixture("ccz_factory_a.stim").read_text())
    _, seen, walks, steps = _recorded_prefixes_l3(s, 1500, 7)
    bt = s.record_report()["born_trie"]
    assert (bt["walks"], bt["steps"]) == (walks, steps)
    assert bt["node_evals"] + bt["root_memo_hits"] == len(seen) and bt["max_depth"] >= 2   # S8-4: root memo
    assert tuple(s.born_trie_check()) == (len(seen), 0)
    assert tuple(s.born_trie_oracle()) == (len(seen), 0)


@pytest.mark.parametrize("mode", ["sample", "barrier"])
def test_large_chi_info_event_once_per_call_at_first_node_public(mode, caplog, capfd):
    caplog.set_level(logging.INFO, logger="xtim.twirl")
    capfd.readouterr()
    s = public_sampler(M3_OBS)
    _digest_l3(s, mode, 2000, 3)
    nodes = s.record_report()["born_trie"]["node_evals"]
    recs = _born_records(caplog)
    assert nodes >= 1 and len(recs) == 1 and recs[0].levelno == logging.INFO
    assert "Born-trie" not in capfd.readouterr().err
    caplog.clear()
    _digest_l3(s, mode, 2000, 3)
    assert s.record_report()["born_trie"]["node_evals"] == 0 and _born_records(caplog) == []


def test_large_chi_info_event_absent_without_born_directions_public(caplog):
    caplog.set_level(logging.INFO, logger="xtim.twirl")
    s = public_sampler(KAPPA0)
    s.sample(2000, seed=1)
    assert s.record_report()["born_trie"]["walks"] == 0 and _born_records(caplog) == []


def test_unread_direction_is_not_walked_public():
    s = public_sampler(M3_DET)
    s.sample(3000, seed=7)
    s.sample_barrier(3000, seed=8)
    bt = s.record_report()["born_trie"]
    assert bt["walks"] == 0 and bt["steps"] == 0
