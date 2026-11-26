"""QW-D pins: a Clifford sampler builds neither the certified group's G_z cache nor (selfcheck 0) the propagation table.

* G_z (CertifiedGroupPlanes::group_z_cache) is read only by a shot law with an active generator; a circuit without
  T / CS / CCZ / CH has the identity plan only (no active generator), so compile + sample (threads 1 and > 1) never
  builds it (process-wide count _xtim._group_z_cache_builds). A magic circuit's threaded call still builds it first
  (freeze_), and so does a magic plan build.
* The propagation table of a circuit without non-Clifford gates is all in class with no PPR atom (decided from the gate
  kinds); with selfcheck 0 and no all-shots oracle nothing per shot reads it, so the sampler never builds it
  (_xtim._sampler_table_builds; _footprint()["table"] == 0). The selfcheck window and magic circuits still get it;
  the test API's digest and diag_shot rebuild it on demand, identically, and do not keep it.
"""
import os

import numpy as np
import pytest
import stim

import xtim
from xtim import _xtim
from xtim import _xtim_testapi as TA

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = os.path.join(HERE, "data", "cultivation_d3_faithful.stim")


def surface(d):
    return str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def chain(n):
    return "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(n))


CASES = {"surface_d5": surface(5), "surface_d11": surface(11), "chain_300": chain(300)}


def builds(fn):
    g0 = _xtim._group_z_cache_builds()
    out = fn()
    return _xtim._group_z_cache_builds() - g0, out


def table_builds(fn):
    t0 = _xtim._sampler_table_builds()
    out = fn()
    return _xtim._sampler_table_builds() - t0, out


@pytest.mark.parametrize("name", sorted(CASES))
def test_clifford_compile_and_sample_build_neither(name):
    c = xtim.Circuit(CASES[name])
    g, s = builds(lambda: c.compile_detector_sampler(seed=1))
    assert g == 0, (name, "compile", g)
    g, _ = builds(lambda: s.sample(1000))
    assert g == 0, (name, "sample(1000)", g)
    g, _ = builds(lambda: s.sample(1000, threads=3))
    assert g == 0, (name, "sample(1000, threads=3)", g)


def test_magic_threaded_call_builds_g_z_first():
    s = xtim.compile_twirl_sampler(open(MAGIC).read(), selfcheck=0)
    g, _ = builds(lambda: s.sample(2000))
    assert g == 0, g                                     # the rotation loop's single-threaded shots read no G_z
    g, _ = builds(lambda: s.sample(2000, threads=3))
    assert g == 1, g                                     # freeze_ built it before the lanes


def test_magic_plan_build_builds_g_z():
    s = xtim.compile_twirl_sampler(open(MAGIC).read(), selfcheck=50)
    g, _ = builds(lambda: s.sample(2000))                # the window builds the shots' diagonal plans
    assert g == 1, g


@pytest.mark.parametrize("name", sorted(CASES))
def test_clifford_compile_and_sample_build_no_table(name):
    c = xtim.Circuit(CASES[name])
    t, s = table_builds(lambda: c.compile_detector_sampler(seed=1))
    assert t == 0, (name, "compile", t)
    t, _ = table_builds(lambda: (s.sample(1000), s.sample(1000, threads=3)))
    assert t == 0, (name, "sample", t)
    f = TA.compile_family_sampler(CASES[name], selfcheck=0)
    f.sample(1000, 1)
    assert int(f._footprint()["table"]) == 0


def test_selfcheck_and_magic_still_build_the_table():
    t, s = table_builds(lambda: xtim.compile_twirl_sampler(CASES["surface_d5"], selfcheck=50))
    assert t == 1, t
    s.sample(1000)                                       # the window reads it (compose_fired against each plan)
    t, s = table_builds(lambda: xtim.compile_twirl_sampler(open(MAGIC).read(), selfcheck=0))
    assert t == 1, t
    assert int(TA.compile_family_sampler(open(MAGIC).read(), selfcheck=0)._footprint()["table"]) > 0


def test_clifford_digest_and_diag_shot_rebuild_the_table_on_demand():
    text = CASES["surface_d5"]
    a = TA.compile_family_sampler(text, selfcheck=0)
    b = TA.compile_family_sampler(text, selfcheck=50)    # holds the table (the window's): the reference
    assert int(a._footprint()["table"]) == 0 < int(b._footprint()["table"])
    assert a._ctor_digest() == b._ctor_digest()
    for alts in ([0], [3, 17], [5, 9, 40]):
        da, db = a.diag_shot(alts), b.diag_shot(alts)
        assert da.keys() == db.keys()
        for k in da:
            assert np.array_equal(np.asarray(da[k]), np.asarray(db[k])), (alts, k)
    assert int(a._footprint()["table"]) == 0             # rebuilt per call, never kept
