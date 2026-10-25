"""J-b1 F, fixplan-F-branch-readers (jc/plan.md §0.1 F-1 / F-2): the PUBLIC layout-3 readers that score or key the
retained state on branch 0's bare refuse carried-branch rows (input_branch != 0) BY NAME; J-c serves them.

  F-1  RecordFidelity.evaluate (and frames / signs) on a buffer holding branch rows -> ValueError naming the reader;
       a branch-0-only subset of the same buffer (evaluate(buf, shots)) is served, and equals the same shots scored
       on an all-branch-0 run. MUT-F1-RF-BRANCH (-DXTIM_MUTATE_F1_RF_BRANCH) turns test_f1 red.
  F-2  record_keys / record_group_ids / record_groups (and the port's RecordGroups) -> ValueError by name: the
       layout-3 key carries no branch id, so a branch row and a branch-0 row would share a group although their
       states differ. MUT-F2-GROUPS-BRANCH (-DXTIM_MUTATE_F2_GROUPS_BRANCH) turns test_f2 red.
materialize keeps serving branch rows (the branch's own consumer bare).
"""
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim

pytestmark = pytest.mark.filterwarnings("ignore")

# tests/test_jb1_branch_law_e0.py CE2 (a producer whose records give several branches) and a consumer on its port
CE2 = ("R 0 1 2\nH 0\nCX 0 1\nDEPOLARIZE1(0.3) 0\nT 0\nT_DAG 1\nH 2\nCX 2 0\nCX 2 1\nH 2\nM 2\nDETECTOR rec[-1]\n"
       "OUTPUT_QUBITS out 0 1\n")
CONS = "INPUT_QUBITS in 0 1\nT 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"
N = 300


@pytest.fixture(scope="module")
def mixed():
    p = _xtim.TwirlSampler(CE2, 1.0, "", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = _xtim.TwirlSampler.from_ideal_state(ideal_state=p.ideal_state(), circuit_text=CONS, selfcheck=0)
    R = p.sample_barrier(N, seed=1)
    H, fr = p.branches(R)
    ids = {}
    ib = np.array([ids.setdefault((bytes(h["digest"]), bytes(h["z"])), c.add_branch(h)) for h in H], np.int32)
    ib[::2] = 0                                      # every registered branch id is >= 1: mix in branch-0 rows
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
    b = c.sample_barrier(N, seed=3, input_pauli=ip, input_branch=ib)
    b0 = c.sample_barrier(N, seed=3, input_pauli=ip, input_branch=np.zeros(N, np.int32))
    rows = np.flatnonzero(ib != 0)
    assert 0 < len(rows) < N, "fixture must mix branch and branch-0 rows"
    return c, b, b0, ib


CONS_RF = "INPUT_QUBITS in 0 1\nT 1\nR 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1\n"


def test_f1_record_fidelity_refuses_branch_rows():
    """The branch-0 evaluator refuses carried-branch rows by name (a consumer whose output register meets the strict
    contract: the producer's Bell pair is the output)."""
    p = _xtim.TwirlSampler(CE2, 1.0, "", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = _xtim.TwirlSampler.from_ideal_state(ideal_state=p.ideal_state(), circuit_text=CONS_RF, selfcheck=0)
    R = p.sample_barrier(N, seed=1)
    H, fr = p.branches(R)
    ids = {}
    ib = np.array([ids.setdefault((bytes(h["digest"]), bytes(h["z"])), c.add_branch(h)) for h in H], np.int32)
    ib[::2] = 0
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
    b = c.sample_barrier(N, seed=3, input_pauli=ip, input_branch=ib)
    b0 = c.sample_barrier(N, seed=3, input_pauli=ip, input_branch=np.zeros(N, np.int32))
    assert 0 < int((ib != 0).sum()) < N
    rf = xtim.RecordFidelity(c)
    with pytest.raises(ValueError, match=r"RecordFidelity: record \d+ is a carried-branch row"):
        rf.evaluate(b)
    keep = np.flatnonzero(ib == 0)
    got = rf.evaluate(b, keep)                       # a branch-0-only subset is served
    assert got.shape == (len(keep),) and np.isfinite(got).all()
    assert np.array_equal(got, rf.evaluate(b0, keep))   # same shots (seed, frames) on an all-branch-0 run
    assert rf.evaluate(b0).shape == (N,)


def test_f1_frames_and_signs_refuse_branch_rows(mixed):
    c, b, b0, ib = mixed
    with pytest.raises(ValueError, match="frames: record .* carried-branch row"):
        b.frames()
    with pytest.raises(ValueError, match="signs: record .* carried-branch row"):
        b.signs([])
    i = int(np.flatnonzero(ib != 0)[0])
    assert type(b.materialize(i)).__name__ == "FramedSuperposition"   # materialize serves the branch row


def test_f2_record_keys_and_groups_refuse_branch_rows(mixed):
    c, b, b0, ib = mixed
    with pytest.raises(ValueError, match="record_keys: record .* carried-branch row"):
        b.record_keys()
    with pytest.raises(ValueError, match="record groups: record .* carried-branch row"):
        b.record_group_ids()
    with pytest.raises(ValueError, match="record groups: record .* carried-branch row"):
        b.record_groups()
    from xtim._record_groups import RecordGroups
    with pytest.raises(ValueError, match="carried-branch row"):
        RecordGroups(b)
    # branch-0 buffers are unchanged: keys, ids and groups served
    k = b0.record_keys()
    gid = np.asarray(b0.record_group_ids())
    keys, cnt = b0.record_groups()
    assert len(k) == N and gid.shape == (N,) and int(np.sum(cnt)) == N
    assert [bytes(x) for x in keys] == list(dict.fromkeys(bytes(x) for x in k))


def test_branch_row_refusals_are_typed(mixed):
    """J-b1 final review m2: the branch-row reader refusals are XtimTwirlRefusal(LAW_LIMITATION_CARRIED_AXIS,
    "branch_row") (still ValueError subclasses)."""
    from xtim.errors import TwirlClass, XtimTwirlRefusal
    c, b, b0, ib = mixed
    with pytest.raises(XtimTwirlRefusal) as ei:
        b.frames()
    assert ei.value.twirl_class == TwirlClass.LAW_LIMITATION_CARRIED_AXIS and ei.value.sub_reason == "branch_row"
