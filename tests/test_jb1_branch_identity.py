"""J-b1 P9a: the carried-branch IDENTITY layer (design §2.10 "Identity by CONTENT"; plan §2.1 P9a, §3.6), test API.

Producer: a FAM sampler exports its ideal state (ideal_state(): the frame-explicit carried state, its OUTPUT_QUBITS
wires and the producer frame token) and the content handles of its layout-3 records (branch_handles: frame token,
128-bit content digest, Born word z). Consumer: compile_family_sampler_from_ideal_state carries the ideal state as
branch 0 (state-level injection) and registers producer branches with add_branch(handle). Gates:
  * wrong token raises (MUT-BR-TOKEN -> red); a rows-only ideal state refuses producer branches; a digest mismatch
    raises; a forced digest collision is a named refusal (MUT-BR-DIGEST -> red);
  * cold / warm producer caches and permuted registration orders give bit-identical branch vectors and the same
    branch_table_digest;
  * ZERO map probes per shot on branch-0 workloads (MUT-BR-MAPPROBE -> red);
  * the pivot row of every barrier read is chosen by the read's pattern only (counter = 0; design Lemma R1);
  * a producer branch (id != 0) is sampled by the consumer branch law (P9b; refused by name at P9a).
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_ja_branch import FX_XSTAGE  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

# The CE1 MX0 variant (carried wire 1): its dressed branch is frame-equal with E_min != {} (the CE1-chain's dressed
# branch, OUTPUT_QUBITS out 0, is a declared refusal since P9b: tests/test_jb1_branch_frame.py).
PRODUCER = "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nMX 0\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n"
CONSUMER = "INPUT_QUBITS in 0\nH 0\nM 0\nR 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def consumer(ideal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler_from_ideal_state(ideal, CONSUMER)


@pytest.fixture(scope="module")
def prod():
    p = fam(PRODUCER)
    d, o, dc, R = p.sample_records(400, seed=3)
    H = p.branch_handles(R)
    return p, R, H


def distinct(H):
    seen, out = set(), []
    for h in H:
        k = (h["digest"], h["z"])
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def test_tokens():
    a, b = fam(PRODUCER), fam(PRODUCER)
    assert a.ideal_state().token == b.ideal_state().token and a.ideal_state().provenance
    assert fam(PRODUCER.replace("0.3", "0.2")).ideal_state().token != a.ideal_state().token   # the circuit digest
    r = a.ideal_state(provenance=False)
    assert not r.provenance and r.token != a.ideal_state().token


def test_branch0_unchanged_and_zero_probes(prod):
    p, R, H = prod
    c = consumer(p.ideal_state())
    for h in distinct(H):
        c.add_branch(h)
    a = c.sample(2000, seed=7)
    b = c.sample(2000, seed=7, input_branch=np.zeros(2000, np.int32))
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert c.branch_counters()["map_probes"] == 0


def test_registration_and_identity(prod):
    p, R, H = prod
    c = consumer(p.ideal_state())
    D = distinct(H)
    ids = [c.add_branch(h) for h in D]
    assert ids == list(range(1, len(D) + 1))
    assert [c.add_branch(h) for h in D] == ids                   # the same handle -> the same branch
    assert c.branch_counters()["branches"] == len(D) + 1


def test_wrong_token_raises(prod):
    p, R, H = prod
    c = consumer(p.ideal_state())
    h = dict(H[0])
    h["token"] = bytes(16)
    with pytest.raises(RuntimeError, match="frame token is not this ideal state"):
        c.add_branch(h)


def test_rows_only_refuses_producer_branches(prod):
    p, R, H = prod
    c = consumer(p.ideal_state(provenance=False))
    with pytest.raises(RuntimeError, match="rows-only frame token"):
        c.add_branch(H[0])
    c.sample(100, seed=1)                                        # branch 0 serves


def test_digest_mismatch_raises(prod):
    p, R, H = prod
    c = consumer(p.ideal_state())
    h = dict(H[0])
    h["digest"] = bytes(range(16))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        c.add_branch(h)


def test_forced_digest_collision_is_refused(prod):
    p, R, H = prod
    keys = {}
    for h in distinct(H):
        keys.setdefault(h["digest"], h)
    assert len(keys) >= 2
    a, b = list(keys.values())[:2]
    c = consumer(p.ideal_state())
    c._force_branch_digest(bytes(16))
    c.add_branch(a)
    with pytest.raises(RuntimeError, match="digest collision"):
        c.add_branch(b)


def test_cold_warm_and_permuted_resolution(prod):
    p, R, H = prod
    D = distinct(H)
    c1, c2 = consumer(p.ideal_state()), consumer(p.ideal_state())
    for h in D:
        c1.add_branch(h)
    for h in reversed(D):
        c2.add_branch(h)
    assert c1.branch_table_digest() == c2.branch_table_digest()
    # a warm vs an evicted (cold) producer payload: the rebuilt branch vectors are bit-identical
    v_warm = [c1.branch_vector(i + 1) for i in range(len(D))]
    p.set_plan_payload_budget(1)
    p.sample(50, seed=99)                                        # evict at the shot boundary
    c3 = consumer(p.ideal_state())
    for h in D:
        c3.add_branch(h)
    for i in range(len(D)):
        a, b = v_warm[i], c3.branch_vector(i + 1)
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    p.set_plan_payload_budget(0)


def test_nonzero_input_branch_is_served_since_p9b(prod):
    """P9a refused a producer branch (id != 0) by name; P9b serves it by the consumer branch law (its gates:
    tests/test_jb1_branch_law_e0.py)."""
    p, R, H = prod
    c = consumer(p.ideal_state())
    b = c.add_branch(H[0])
    c.sample(10, seed=1, input_branch=np.full(10, b, np.int32))
    assert c.branch_counters()["branch_shots"] == 10 or c.branch_emin(b)   # E_min != {}: its per-E consumer


@pytest.mark.parametrize("name", sorted(FX_XSTAGE))
def test_pivot_rows_pattern_only(name):
    before = TA.pivot_nonpattern_count()
    s = fam(FX_XSTAGE[name])
    d, o, dc, R = s.sample_records(300, seed=5)
    for i in range(0, 300, 3):
        s.materialize(R, i)
    assert TA.pivot_nonpattern_count() == before
