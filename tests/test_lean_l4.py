"""lean L4 (2026-11-04): cheaper plan builds on T circuits, byte-identical. A plan miss used to spend most of its ~15 M
instructions in dense scans whose content is lightcone-sized: every channel mask scanned over all GW generator words
(the record-column index now walks the active generators' channels), the C_det signs by an exact RREF reduction per
combination (now a closed form over the signed pure-Z group content and the Pi matrix, the reduction kept as the
window's teeth), dense generator columns XORed per CZ endpoint (now the sparse X-column lists), the Bits-form
canonicalization forced by the extras (now the one-word form whenever it fits). The mechanisms are asserted through
record-space B's op counts; the bytes through the window / oracle_all_shots / lanes agreement below and the byte gate.
"""
import hashlib
import os
import re
import sys

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA

sys.path.insert(0, os.path.dirname(__file__))
from test_ja_record_space_b import plan_keys  # noqa: E402
from test_jb2_kappa_rec_law import circuit_text as kappa_text  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D5 = os.path.join(HERE, "..", "examples", "cultivation_d5.stim")           # the real T circuit (kappa_rec = 1 plans)
CS_RATE = os.path.join(HERE, "data", "code_switching_rate.stim")           # Born observable, kappa_rec > 0 plans
COLOR5 = os.path.join(HERE, "data", "color5_xmix_prep_T_depolarizing_p1e-3_r3.stim")
SIGN_BYTES = os.path.join(HERE, "data", "sign_live_bytes_c40022.stim")       # L4 review F1: the sign term moves bytes


# A circuit (found by a random search over small T / CS / CZ circuits with negated stabilizers) on which the C_det SIGN
# TERM of the shot law is live: a deterministic generator combination whose dressing reduces through a pure-Z group
# element of sign -1 (GroupZCache::gz_sign). A mutant that drops that term trips the window by name on exactly this
# circuit ("the C_det sign formula disagrees with the exact reduction") — the test-of-test. Its selfcheck-0 BYTES do not
# move under that mutant (L4 review F1: the 5297ec6 message's "moves its bytes" was wrong; the mutant is caught by the
# window throw only). The second fixture below, sign_bytes (tests/data/sign_live_bytes_c40022.stim, a random stress
# circuit from the L4 review), DOES move its selfcheck-0 bytes under the mutant (a7426cc8... vs 4c1f556a...).
SIGN_C = ("R 0\nR 1\nRX 2\nCX 1 2\nCZ 1 2\nCZ 1 2\nCX 2 1\nZ 2\nY 2\nT 2\nDEPOLARIZE1(0.3) 1\nCS 2 0\nX_ERROR(0.3) 0\n"
          "CZ 1 2\nX_ERROR(0.3) 1\nT_DAG 0\nDEPOLARIZE1(0.3) 0\nCS 2 0\nZ_ERROR(0.3) 2\nT 1\nZ_ERROR(0.3) 1\nZ_ERROR(0.3) 2\n"
          "M 0\nDETECTOR rec[-1]\nMX 1\nDETECTOR rec[-1]\nMX 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")


def standin(text):
    return re.sub(r"^(\s*)T_DAG\b", r"\1S_DAG", re.sub(r"^(\s*)T\b", r"\1S", text, flags=re.M), flags=re.M)


def _fixtures():
    fx = {
        "d5_T": open(D5).read(),
        "d5_standin": standin(open(D5).read()),
        "cs_rate": open(CS_RATE).read(),
        "kappa2": kappa_text(2),                                           # law kappa = 2 (kernel directions)
        "sign_c": SIGN_C,                                                  # the C_det sign term live
        "sign_bytes": open(SIGN_BYTES).read(),                             # the sign term live AND byte-visible
    }
    if os.path.exists(COLOR5):
        fx["color5"] = open(COLOR5).read()
    return fx


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _arrs(out):
    if isinstance(out, tuple):
        return [np.asarray(a) for a in out]
    return [np.asarray(out.dets), np.asarray(out.obs), np.asarray(out.decisions)]


@pytest.mark.parametrize("name", ["d5_T", "cs_rate", "kappa2"])
def test_plan_build_work_is_lightcone_sized(name):
    """Record-space B on sampled plan keys: no channel mask is scanned densely (mask_words == 0: the record-column
    index serves the active generators' channels) and the law's production path calls G.reduce only for its kappa
    kernel representatives (law_reduces == kappa: the C_det signs come from the signed G_z closed form)."""
    s = TA.compile_family_sampler(_fixtures()[name], selfcheck=0)
    keys = plan_keys(s, shots=600, seed=5)
    assert len(keys) > 1
    checked = 0
    for k in keys[: min(len(keys), 40)]:
        ps = TA.recb_plan_structure(s, k)
        if ps["fallback"]:
            continue
        ops = ps["ops"]
        assert ops["mask_words"] == 0, ops
        assert ops["law_reduces"] == ps["kappa"], (ops, ps["kappa"])
        assert ops["touched"] >= 0
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", ["d5_T", "d5_standin", "cs_rate", "kappa2", "sign_c", "sign_bytes"])
def test_streaming_equals_window_and_oracle_all_shots(name):
    """selfcheck = 0 vs selfcheck = 200 (the first fired shots through the exact window: the law's teeth, record-space
    B's teeth, the key / abelian / extras oracles, class_rep_check_) vs oracle_all_shots: the same bytes, seeded and
    continued."""
    text = _fixtures()[name]
    a = TA.compile_family_sampler(text, selfcheck=0)
    b = TA.compile_family_sampler(text, selfcheck=200)
    c = TA.compile_family_sampler(text, selfcheck=0, oracle_all_shots=True)
    assert a._ctor_digest() == b._ctor_digest() == c._ctor_digest()
    ha, hb, hc = (_h(x.sample(3000, 9)) for x in (a, b, c))
    assert ha == hb == hc
    assert _h(a.sample(3000, None)) == _h(b.sample(3000, None)) == _h(c.sample(3000, None))


@pytest.mark.parametrize("name", ["d5_T", "kappa2", "cs_rate", "sign_c", "sign_bytes"])
def test_public_window_equals_streaming(name):
    """The public sampler: selfcheck = 0 vs a 400-shot window (every plan built in the window runs the teeth)."""
    text = _fixtures()[name]
    a = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = xtim.compile_twirl_sampler(text, selfcheck=400)
    assert _h(_arrs(a.sample(4000, seed=3))) == _h(_arrs(b.sample(4000, seed=3)))
    assert _h(_arrs(a.sample(4000, seed=None))) == _h(_arrs(b.sample(4000, seed=None)))


@pytest.mark.parametrize("name", ["d5_T", "kappa2"])
def test_threads_equal_serial(name):
    """The lanes build plans too (their overlay caches read the planes' lazily built indexes, frozen before they start):
    the same bytes as the serial run."""
    text = _fixtures()[name]
    a = xtim.compile_twirl_sampler(text, selfcheck=0).sample(12000, seed=2, threads=4)
    b = xtim.compile_twirl_sampler(text, selfcheck=0).sample(12000, seed=2)
    assert _h(_arrs(a)) == _h(_arrs(b))


@pytest.mark.skipif(not os.path.exists(COLOR5), reason="fixture absent")
def test_color5_window_equals_streaming():
    text = _fixtures()["color5"]
    a = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = xtim.compile_twirl_sampler(text, selfcheck=300)
    assert _h(_arrs(a.sample(3000, seed=7))) == _h(_arrs(b.sample(3000, seed=7)))
