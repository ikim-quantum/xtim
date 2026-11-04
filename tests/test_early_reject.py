"""early-reject (OD-2026-10-25-early-reject-stage): the opt-in postselection mask of the twirl sampler.

``sample(..., postselect=mask)`` / ``sample_barrier(..., postselect=mask)`` return only the shots whose flagged detectors
are all 0. The engine drops a shot before any plan work when a flagged deterministic detector fires for certain (the
shot's Clifford part C shares no qubit with the detector's operator, so the bit is ⟨P, M_d⟩ ⊕ readout flips); every
emitted shot also passes the exact filter. Contract pinned here: the accepted shots are BYTE-IDENTICAL to an unmasked
call at the same seed filtered afterwards (detectors, observables, decisions, layout-3 records, RecordFidelity
values), the counts agree, and the early path really runs (early > 0 where the probe measured it).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import xtim
from _private import private_fixture
from xtim.twirl import TwirlBatch

DATA = os.path.join(os.path.dirname(__file__), "data")


def _text(name):
    with open(os.path.join(DATA, name)) as f:
        return f.read()


def _unpack(a, n):
    return np.unpackbits(np.asarray(a), axis=1, bitorder="little")[:, :n].astype(bool)


# (circuit, shots, the early path must reject at least one shot)
CIRCUITS = [
    ("cultivation_d3_faithful.stim", 4000, True),
    ("cultivation_d3_rate.stim", 4000, True),
    ("distillation_15_1_3.stim", 4000, True),
    ("code_switching_faithful.stim", 4000, True),
    ("code_switching_rate.stim", 4000, True),
    ("miniature_oracle.stim", 4000, True),
    ("ch_cultivation.stim", 4000, False),       # CH (term route): its three detectors meet C on every faulted shot
    ("ja_seed23.stim", 4000, True),            # term-route (kind-2) shots
    ("adaptq_qrm_magic_producer.stim", 4000, True),
    ("cultivation_d5.stim", 2000, True),
    ("magna_color5_fid.stim", 2000, True),
    ("magna_rp2_fid.stim", 2000, True),
]


def _check_sample(text, shots, accept, seed=11, expect_early=False):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    full = s.sample(shots, seed=seed)
    keep = ~_unpack(full[0], nd)[:, accept].any(1)
    s2 = xtim.compile_twirl_sampler(text, selfcheck=0)
    out = s2.sample(shots, seed=seed, postselect=accept)
    if isinstance(full, TwirlBatch):          # a DECISION circuit's wrapper: (batch, counts)
        full = (full.dets, full.obs, full.decisions)
        out = (out[0].dets, out[0].obs, out[0].decisions, out[1])
    counts = out[-1]
    assert len(out) == len(full) + 1
    for a, b in zip(full, out[:-1]):
        assert np.array_equal(np.asarray(a)[keep], np.asarray(b))
    assert counts["attempts"] == shots
    assert counts["passed"] == int(keep.sum())
    assert counts["discarded"] == shots - int(keep.sum())
    assert 0 <= counts["early"] <= counts["discarded"]
    if expect_early:
        assert counts["early"] > 0
    return counts


@pytest.mark.parametrize("name,shots,early", CIRCUITS)
def test_sample_all_detectors_byte_identical(name, shots, early):
    text = _text(name)
    nd = xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors
    _check_sample(text, shots, list(range(nd)), expect_early=early)


@pytest.mark.parametrize("name,shots,early", CIRCUITS[:6] + CIRCUITS[9:])
def test_sample_partial_mask_byte_identical(name, shots, early):
    text = _text(name)
    nd = xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors
    sub = list(range(0, nd, 2))
    mask = np.zeros(nd, dtype=bool)
    mask[sub] = True
    _check_sample(text, shots, sub, seed=5)
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    a = s.sample(shots, seed=5, postselect=sub)
    b = s.sample(shots, seed=5, postselect=mask)    # the boolean form is the same mask
    for x, y in zip(a[:-1], b[:-1]):
        assert np.array_equal(x, y)


def _check_barrier(text, shots, accept, seed=3, fid=True):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    full = s.sample_barrier(shots, seed=seed)
    keep = ~_unpack(full.dets(), nd)[:, accept].any(1)
    idx = np.flatnonzero(keep)
    s2 = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s2.sample_barrier(shots, seed=seed, postselect=accept)
    assert full.postselection is None
    c = buf.postselection
    assert (c["attempts"], c["passed"], c["discarded"]) == (shots, len(idx), shots - len(idx))
    assert buf.num_shots == len(idx)
    assert np.array_equal(np.asarray(full.dets())[keep], np.asarray(buf.dets()))
    assert np.array_equal(np.asarray(full.obs())[keep], np.asarray(buf.obs()))
    assert np.array_equal(np.asarray(full.decisions())[keep], np.asarray(buf.decisions()))
    kf = full.record_keys()
    assert [kf[i] for i in idx] == buf.record_keys()
    for j, i in enumerate(idx):
        assert np.array_equal(np.asarray(full.outcomes(int(i))), np.asarray(buf.outcomes(j)))
        assert np.array_equal(np.asarray(full.record_outcomes(int(i))), np.asarray(buf.record_outcomes(j)))
        assert full.frame_folded(int(i)) == buf.frame_folded(j)
    if fid and len(idx):
        Ff = np.asarray(xtim.RecordFidelity(s).evaluate(full, idx))
        Fm = np.asarray(xtim.RecordFidelity(s2).evaluate(buf))
        assert Ff.tobytes() == Fm.tobytes()
    return c


@pytest.mark.parametrize("name,shots,fid", [("cultivation_d3_faithful.stim", 3000, False),
                                            ("magna_rp2_fid.stim", 1500, True), ("magna_color5_fid.stim", 1500, True),
                                            ("distillation_15_1_3.stim", 3000, False),
                                            ("code_switching_faithful.stim", 3000, False)])
def test_barrier_and_fidelity_byte_identical(name, shots, fid):
    text = _text(name)
    nd = xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors
    c = _check_barrier(text, shots, list(range(nd)), fid=fid)
    assert c["early"] > 0


@pytest.mark.parametrize("name", ["ja_seed23.stim", "ch_cultivation.stim"])
def test_barrier_term_route_byte_identical(name):
    text = _text(name)
    nd = xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors
    _check_barrier(text, 3000, list(range(nd)), fid=False)


def test_counter_streams_continue_after_a_masked_call():
    # shot_ advances for every shot, rejected or not: a masked call followed by an unmasked continuation (seed=None)
    # equals the unmasked stream.
    text = _text("cultivation_d3_faithful.stim")
    nd = 20
    a = xtim.compile_twirl_sampler(text, selfcheck=0)
    ref = a.sample(6000, seed=9)
    b = xtim.compile_twirl_sampler(text, selfcheck=0)
    m = b.sample(3000, seed=9, postselect=list(range(nd)))
    rest = b.sample(3000)
    keep = ~_unpack(ref[0][:3000], nd).any(1)
    assert np.array_equal(ref[0][:3000][keep], m[0])
    assert np.array_equal(ref[0][3000:], rest[0]) and np.array_equal(ref[1][3000:], rest[1])


def test_input_frames_byte_identical():
    # framed shots take the term route with the frame atoms composed into (P, S)
    text = "INPUT_QUBITS in 0\nR 1\nX 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n"
    shots = 4000
    rng = np.random.default_rng(1)
    fr = (rng.random((shots, 2)) < 0.3).astype(np.uint8)
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    full = s.sample(shots, seed=4, input_pauli=fr)
    keep = ~_unpack(full[0], 2).any(1)
    out = xtim.compile_twirl_sampler(text, selfcheck=0).sample(shots, seed=4, input_pauli=fr, postselect=[0, 1])
    assert np.array_equal(full[0][keep], out[0]) and np.array_equal(full[1][keep], out[1])
    assert out[-1]["passed"] == int(keep.sum()) and out[-1]["early"] > 0


def test_empty_mask_and_validation():
    text = _text("cultivation_d3_faithful.stim")
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    ref = s.sample(500, seed=2)
    out = s.sample(500, seed=2, postselect=[])
    assert np.array_equal(ref[0], out[0]) and out[-1] == {"attempts": 500, "passed": 500, "discarded": 0, "early": 0}
    out = s.sample(500, seed=2, postselect=np.zeros(20, dtype=bool))
    assert np.array_equal(ref[0], out[0])
    with pytest.raises(ValueError):
        s.sample(10, seed=1, postselect=np.zeros(19, dtype=bool))
    with pytest.raises(ValueError):
        s.sample(10, seed=1, postselect=[20])
    with pytest.raises(ValueError):
        s.sample(10, seed=1, postselect=[0.5])
    # empty forms agree: an empty list and an empty array of any dtype flag no detector
    for empty in ([], np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64), np.zeros(0)):
        out = s.sample(500, seed=2, postselect=empty)
        assert np.array_equal(ref[0], out[0]) and out[-1]["passed"] == 500
        assert s.sample_barrier(50, seed=2, postselect=empty).postselection["passed"] == 50
    # the mask is per call: a later unmasked call is the plain sampler
    s.sample(500, seed=2, postselect=list(range(20)))
    again = s.sample(500, seed=2)
    assert np.array_equal(ref[0], again[0]) and np.array_equal(ref[1], again[1])


@pytest.mark.parametrize("fixture", ["ccz_factory_a.stim", "ccz_factory_b.stim", "ccz_factory_d.stim"])
def test_private_factories_byte_identical(fixture):
    text = private_fixture(fixture).read_text()
    nd = xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors
    _check_sample(text, 1500, list(range(nd)), expect_early=True)


# ── fix round (review MINOR-1): a non-bool array is read as detector indices; duplicates / out-of-range refused ──────
def test_integer_mask_is_refused_not_misread():
    s = xtim.compile_twirl_sampler(_text("cultivation_d3_faithful.stim"), selfcheck=0)
    nd = s.num_detectors
    ones = np.ones(nd, dtype=np.uint8)
    unpacked = np.unpackbits(np.full((nd + 7) // 8, 0b10110001, np.uint8), bitorder="little")[:nd]   # uint8 0/1
    assert unpacked.dtype == np.uint8
    for bad in (ones, unpacked, np.zeros(nd, dtype=np.int64), [3, 3]):
        with pytest.raises(ValueError, match="duplicate detector index"):
            s.sample(10, seed=1, postselect=bad)
        with pytest.raises(ValueError, match="duplicate detector index"):
            s.sample_barrier(10, seed=1, postselect=bad)
    with pytest.raises(ValueError, match="out of range"):
        s.sample(10, seed=1, postselect=[nd])
    with pytest.raises(ValueError, match="out of range"):
        s.sample(10, seed=1, postselect=np.array([-1]))
    # the same 0/1 values cast to bool are the mask; as indices (np.flatnonzero) the same selection
    ref = xtim.compile_twirl_sampler(_text("cultivation_d3_faithful.stim"), selfcheck=0)
    a = ref.sample(500, seed=4, postselect=unpacked.astype(bool))
    b = s.sample(500, seed=4, postselect=np.flatnonzero(unpacked))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[-1] == b[-1]


# ── fix round (review MINOR-4): paths the review verified, pinned here ───────────────────────────────────────────────
@pytest.mark.filterwarnings("ignore::xtim.XtimVerificationWarning")
@pytest.mark.parametrize("name", ["cultivation_d3_faithful.stim", "distillation_15_1_3.stim"])
def test_first_masked_call_inside_the_selfcheck_window(name):
    # default selfcheck: the first (masked) call runs the self-check window; then an unmasked continuation and a masked
    # barrier continuation, against one unmasked stream of the same length.
    text = _text(name)
    a = xtim.compile_twirl_sampler(text)
    nd = a.num_detectors
    ref = a.sample(3000, seed=8)
    refd = _unpack(ref[0], nd)
    b = xtim.compile_twirl_sampler(text)
    acc = list(range(nd))
    m = b.sample(1000, seed=8, postselect=acc)
    keep = ~refd[:1000].any(1)
    assert np.array_equal(ref[0][:1000][keep], m[0]) and np.array_equal(ref[1][:1000][keep], m[1])
    assert m[-1]["passed"] == int(keep.sum())
    mid = b.sample(1000)
    assert np.array_equal(ref[0][1000:2000], mid[0]) and np.array_equal(ref[1][1000:2000], mid[1])
    buf = b.sample_barrier(1000, postselect=acc)
    keep = ~refd[2000:].any(1)
    assert np.array_equal(ref[0][2000:][keep], np.asarray(buf.dets()))
    assert buf.postselection["passed"] == int(keep.sum())


def _wide(n):
    # n > 512 detectors (the FW = 0 masked instance) plus one T so the sampler is non-Clifford
    q = " ".join(map(str, range(n)))
    return ("R %s %d\nH %d\nT %d\nH %d\nX_ERROR(0.002) %s\nM %s %d\n" % (q, n, n, n, n, q, q, n)
            + "".join("DETECTOR rec[-%d]\n" % (k + 2) for k in range(n)) + "M %d\nOBSERVABLE_INCLUDE(0) rec[-1]\n" % n)


def test_more_than_512_detectors():
    text = _wide(600)
    assert xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors == 600
    c = _check_sample(text, 2000, list(range(600)), seed=3, expect_early=True)
    assert c["passed"] > 0
    _check_sample(text, 2000, list(range(1, 600, 3)), seed=6, expect_early=True)
    _check_barrier(text, 1000, list(range(0, 600, 2)), seed=2, fid=False)


PER_E = "RX 0\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nOUTPUT_QUBITS out 0\n"


@pytest.mark.filterwarnings("ignore::xtim.XtimVerificationWarning")
def test_e_class_branches_late_filter():
    # an E_min != {} carried branch runs on its per-E consumer and is filtered in the binding after the scatter
    N = 3000
    p = xtim.compile_twirl_sampler(PER_E, selfcheck=0)
    H, fr = p.branches(p.sample_barrier(N, seed=1))
    ctext = "INPUT_QUBITS in 0\nH 0\nT 0\nH 0\nX_ERROR(0.05) 0\nM 0\nDETECTOR rec[-1]\n"

    def consumer():
        c = xtim.compile_twirl_sampler_from_ideal_state(ctext, ideal_state=p.ideal_state(), selfcheck=0)
        ids = {}
        ib = np.zeros(N, np.int32)
        for i, h in enumerate(H):
            k = (bytes(h["digest"]), bytes(h["z"]))
            if k not in ids:
                ids[k] = c.add_branch(h)
            ib[i] = ids[k]
        return c, ib

    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
    c1, ib = consumer()
    full = c1.sample(N, seed=7, input_pauli=ip, input_branch=ib)
    assert c1.branch_counters()["distinct_emin"] >= 2              # an E_min != {} branch was sampled (late filter)
    keep = ~_unpack(full[0], 1)[:, 0]
    assert 0 < keep.sum() < N
    c2, ib2 = consumer()
    out = c2.sample(N, seed=7, input_pauli=ip, input_branch=ib2, postselect=[0])
    assert np.array_equal(full[0][keep], out[0]) and np.array_equal(full[1][keep], out[1])
    assert out[-1]["attempts"] == N and out[-1]["passed"] == int(keep.sum())


@pytest.mark.parametrize("name,fid", [("magna_rp2_fid.stim", True), ("magna_color5_fid.stim", True),
                                      ("cultivation_d3_faithful.stim", False), ("ja_seed23.stim", False)])
def test_barrier_and_fidelity_partial_masks(name, fid):
    text = _text(name)
    nd = xtim.compile_twirl_sampler(text, selfcheck=0).num_detectors
    rng = np.random.default_rng(nd)
    for accept in (list(range(0, nd, 2)), sorted(rng.choice(nd, max(1, nd * 3 // 10), replace=False).tolist()),
                   [nd - 1]):
        _check_barrier(text, 1200, accept, seed=13, fid=fid)
