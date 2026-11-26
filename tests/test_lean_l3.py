"""lean L3 (2026-11-04): Born-class observables (and every other record extra) on the streaming fast path. A circuit whose
observable is Born-class used to force every fired shot onto the accumulator route (the extras' columns need t(P) = <P, W_e>
from the residual prefix, which the streaming route never builds). The extras are now emitted like the channels: a per-
alternative extras row plus the pair term consumed through the per-wire extras rows (xrow_z_) — the same decomposition the
channel bits already use. Byte-identical: the window / oracle_all_shots agreement below (the window's ABELIAN EXTRAS ORACLE
compares the decomposition with t(nf.prefix) on every checked shot), the postselect (early-reject) and lane instances, and
the byte gate.
"""
import hashlib
import os
import re

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D5 = os.path.join(HERE, "..", "examples", "cultivation_d5.stim")
CS_FAITHFUL = os.path.join(HERE, "data", "code_switching_faithful.stim")   # Born observable, kappa_rec = 0 plans
CS_RATE = os.path.join(HERE, "data", "code_switching_rate.stim")           # Born observable, kappa_rec > 0 plans
BORN5 = os.path.join(HERE, "data", "adaptq_born5_dec_enum_producer.stim")  # Born decisions

M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
      "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
      "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
M3_OBS = M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET
M3_OBS_DEC = M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET + "MX 0\nDECISION(0) rec[-1]\n"
# Two circuits (found by a random search) on which the PAIR TERM reaches the Born observable: the extras' rows of the
# ordered-pair z-corrections (xrow_z_ on the pair-term wires) move the observable's bytes on multi-fault shots. A mutant
# that drops those rows on the streaming route fails the window equality below on exactly these (the test-of-test).
PAIR_CCZ = ("RX 0\nRX 1\nRX 2\nRX 3\nH 1\nX_ERROR(0.3) 3\nDEPOLARIZE1(0.3) 0\nCS 2 0\nX_ERROR(0.3) 2\nZ_ERROR(0.3) 1\nT 3\n"
            "X_ERROR(0.3) 1\nZ_ERROR(0.3) 0\nS 1\nX_ERROR(0.3) 0\nZ_ERROR(0.3) 3\nS 0\nS 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nMY 1\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")          # kappa_rec = 0 plans
PAIR_CS = ("RX 0\nRX 1\nCS 1 0\nX_ERROR(0.3) 1\nS 1\nZ_ERROR(0.3) 1\nT 1\nX_ERROR(0.3) 1\nZ_ERROR(0.3) 1\nH 0\nX_ERROR(0.3) 0\n"
           "Z_ERROR(0.3) 1\nH 0\nX_ERROR(0.3) 0\nCS 0 1\nDEPOLARIZE1(0.3) 1\nZ_ERROR(0.3) 1\nS 0\nX_ERROR(0.3) 1\nZ_ERROR(0.3) 0\n"
           "MX 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")      # kappa_rec > 0 plans too


def standin(text):
    """The circuit's T -> S stand-in (the canary tool's rule): cultivation_d5's observable becomes Born-class."""
    return re.sub(r"^(\s*)T_DAG\b", r"\1S_DAG", re.sub(r"^(\s*)T\b", r"\1S", text, flags=re.M), flags=re.M)


def _fixtures():
    return {
        "d5_standin": standin(open(D5).read()),
        "cs_faithful": open(CS_FAITHFUL).read(),
        "cs_rate": open(CS_RATE).read(),
    }


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _arrs(out):
    """The sampled arrays of a public sample() result (a tuple, or a TwirlBatch for a DECISION circuit)."""
    if isinstance(out, tuple):
        return [np.asarray(a) for a in out]
    return [np.asarray(out.dets), np.asarray(out.obs), np.asarray(out.decisions)]


@pytest.mark.parametrize("name", ["d5_standin", "cs_faithful", "cs_rate"])
def test_born_class_shots_stream(name):
    """A Born-class observable no longer forces the accumulator route: warm, every fired shot streams; the only shots off
    the streaming route are the plan misses (the structural path)."""
    text = _fixtures()[name]
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    s.sample(20000, seed=1)
    r = s.record_report()
    rc = r["route_counts"]
    assert r["nondeterministic_observables"] == [0], r["nondeterministic_observables"]
    assert rc["fast_accumulator"] == 0, rc
    faulted = r["used"] - rc["clean"]
    assert faulted > 0
    # kappa_rec = 0 plans: hits are the streaming remainder (`fast`), misses are `slow_kappa0`;
    # kappa_rec > 0 plans: hits and misses are both `record_kappa_pos` (the trie walk), misses = structural shots
    assert rc["fast"] + rc["slow_kappa0"] + rc["record_kappa_pos"] == faulted, rc
    assert rc["slow_kappa0"] + (rc["record_kappa_pos"] if rc["fast"] == 0 else 0) >= r["structural_shots"]
    assert r["structural_shots"] <= r["plans"], r
    if name == "cs_rate":
        assert rc["record_kappa_pos"] == faulted and rc["fast"] == 0
    else:
        assert rc["fast"] == faulted - r["structural_shots"], (rc, r["structural_shots"])


@pytest.mark.parametrize("name", ["d5_standin", "cs_faithful", "cs_rate"])
def test_streaming_equals_window_and_oracle_all_shots(name):
    """selfcheck = 0 (every fired shot on the streaming route) vs selfcheck = 200 (the first fired shots through the exact
    window with its oracles, the new ABELIAN EXTRAS ORACLE included) vs oracle_all_shots (EVERY shot through the structural
    path with every oracle): the same bytes."""
    text = _fixtures()[name]
    a = TA.compile_family_sampler(text, selfcheck=0)
    b = TA.compile_family_sampler(text, selfcheck=200)
    c = TA.compile_family_sampler(text, selfcheck=0, oracle_all_shots=True)
    assert a._ctor_digest() == b._ctor_digest() == c._ctor_digest()
    ha, hb, hc = (_h(x.sample(3000, 9)) for x in (a, b, c))
    assert ha == hb == hc
    assert _h(a.sample(3000, None)) == _h(b.sample(3000, None)) == _h(c.sample(3000, None))


@pytest.mark.parametrize("text", [M3_OBS, M3_OBS_DEC, open(BORN5).read(), PAIR_CCZ, PAIR_CS],
                         ids=["m3_obs", "m3_obs_dec", "born5_dec", "pair_ccz", "pair_cs"])
def test_public_extras_equal_window(text):
    """The public sampler on Born observables and Born DECISIONs: streaming (selfcheck = 0) equals the window route (the
    structural path's t(nf.prefix) for the extras on the first fired shots, the ABELIAN EXTRAS ORACLE on each of them)."""
    a = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = xtim.compile_twirl_sampler(text, selfcheck=300)
    assert _h(_arrs(a.sample(4000, seed=3))) == _h(_arrs(b.sample(4000, seed=3)))
    assert _h(_arrs(a.sample(4000, seed=None))) == _h(_arrs(b.sample(4000, seed=None)))
    r = a.record_report()
    assert r["route_counts"]["fast_accumulator"] == 0, r["route_counts"]


def _unpack(bits, n):
    return np.unpackbits(np.asarray(bits), axis=1, bitorder="little")[:, :n].astype(bool)


@pytest.mark.parametrize("name", ["d5_standin", "cs_faithful"])
def test_postselect_equals_filter(name):
    """The early-reject instances with a Born observable (the streaming route under ER): the postselected sample is the
    plain sample filtered on the flagged detectors."""
    text = _fixtures()[name]
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    full = s.sample(6000, seed=11)
    accept = np.zeros(nd, bool)
    accept[: max(1, nd // 2)] = True
    keep = ~_unpack(full[0], nd)[:, accept].any(1)
    out = xtim.compile_twirl_sampler(text, selfcheck=0).sample(6000, seed=11, postselect=accept)
    counts = out[-1]
    assert len(out) == len(full) + 1
    for a, b in zip(full, out[:-1]):
        assert np.array_equal(np.asarray(a)[keep], np.asarray(b))
    assert counts["passed"] == int(keep.sum()) and counts["attempts"] == 6000


@pytest.mark.parametrize("name", ["d5_standin", "cs_rate"])
def test_threads_equal_serial(name):
    """The lane instances (threads = 4) stream Born-class shots too: the same bytes as the serial run."""
    text = _fixtures()[name]
    a = xtim.compile_twirl_sampler(text, selfcheck=0).sample(12000, seed=2, threads=4)
    b = xtim.compile_twirl_sampler(text, selfcheck=0).sample(12000, seed=2)
    assert _h(_arrs(a)) == _h(_arrs(b))
