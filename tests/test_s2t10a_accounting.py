"""S2-T10a: the sparse G_z cache and the sampler's memory accounting (test API ``_footprint``).

* The G_z cache is stored sparse (CSR rows): on a mixed circuit (a surface code plus one T ancilla) whose threaded call
  builds it (freeze_), the footprint's "G" group grows by O(weight), far below the pre-S2 dense rows (rank x n'/64
  words). Its bytes against the dense oracle are pinned in C++ (cpp/tests/test_planes_sparse.cpp).
* "bare": the caller-owned bare state the sampler reads is accounted (its frame compact on Clifford circuits: O(n'),
  not the n'^2 / 2 bytes of a dense tableau).
* "lanes": a threaded call's lane copies are accounted (0 before the first threaded call).
"""
import os

import pytest
import stim

import xtim
from xtim import _xtim
from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = os.path.join(HERE, "data", "cultivation_d3_faithful.stim")


def surface(d):
    return str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def surface_plus_t(d):
    c = surface(d)
    nq = stim.Circuit(c).num_qubits
    a, b = nq, nq + 1
    return f"R {a} {b}\nH {a}\nT {a}\nCX {a} {b}\n" + c + f"\nMX {a}\nM {b}\n"


def fp(s):
    return {k: int(v) for k, v in s._footprint().items()}


def test_mixed_threaded_call_keeps_the_g_z_cache_sparse():
    s = TA.compile_family_sampler(surface_plus_t(7), selfcheck=0)
    n = int(s.record_layout()["n"])
    g0 = fp(s)["G"]
    s.sample(2000, 1, threads=3)
    assert s.record_report()["threads"]["effective"] > 1
    grown = fp(s)["G"] - g0
    assert grown > 0                                     # freeze_ built the cache (a plan can read it)
    dense_rows = (n // 2) * 8 * ((n + 63) // 64)         # even half of n' as dense rows would be this many bytes
    assert grown < dense_rows // 2, (grown, dense_rows, n)
    assert grown < 64 * n, (grown, n)                    # linear: a few entries per wire


def test_bare_and_lanes_groups():
    s = TA.compile_family_sampler(surface(9), selfcheck=0)
    n = int(s.record_layout()["n"])
    f = fp(s)
    dense = 2 * n * 16 * ((n + 63) // 64)                # a dense tableau's forward rows alone (x and z words)
    assert 0 < f["bare"] < dense // 8, (f["bare"], dense, n)   # the compact frame: O(n')
    assert f["lanes"] == 0
    s.sample(2000, 1, threads=3)
    th = s.record_report()["threads"]["effective"]
    assert th > 1
    f = fp(s)
    assert f["lanes"] >= (th - 1) * f["shot_scratch"] > 0, f   # each lane copies at least the per-shot scratch


def test_magic_bare_group_counts_the_frame():
    s = TA.compile_family_sampler(open(MAGIC).read(), selfcheck=0)
    assert fp(s)["bare"] > 0
    s.sample(2000, 1, threads=3)
    assert fp(s)["lanes"] > 0


# m4: every footprint group is classified, as shared with the lanes (Frozen, the caller-owned bare state: kShared in
# TwirlRecordSampler::family_footprint) or per lane (summed into "lanes"); a new group fails here until it is.
SHARED = {"table", "bare", "G", "G_chunk", "gens", "sides", "rotation", "rotation_plists", "alt_is_ppr", "sch",
          "rec_cols", "noise", "qcol", "rf_cols"}   # S2-T7: the per-wire rows and readout-flip columns (Frozen)
PER_LANE = {"rotation_store", "term_view", "readout_flip", "shot_scratch", "shot_block",
            "sem_work", "plan_headers", "plan_payloads", "born_root_memo"}


def test_every_footprint_group_is_classified():
    s = TA.compile_family_sampler(open(MAGIC).read(), selfcheck=0)
    s.sample(2000, 1, threads=3)
    f = fp(s)
    assert set(f) == SHARED | PER_LANE | {"lanes"}, sorted(set(f) ^ (SHARED | PER_LANE | {"lanes"}))
    assert f["lanes"] >= f["shot_scratch"] > 0, f   # each lane copies the per-lane arrays


@pytest.mark.parametrize("which", ["surface", "magic"])
def test_model_key_and_ctor_digest_keep_no_rows_cache(which):
    # S2-CLEAN (interim review M4): family_model_key (production: record buffers) and family_ctor_bytes read the
    # compact frame's rows through transient sweeps; they no longer build the frame's rows cache and keep it in the
    # sampler's bare state for its lifetime (which "bare" did not report before S2-T10a counted it).
    text = surface(9) if which == "surface" else open(MAGIC).read()
    s = TA.compile_family_sampler(text, selfcheck=0)
    s.sample(200, 1)                                     # whatever a plain run builds lazily, built first
    b0 = fp(s)["bare"]
    d0 = s._ctor_digest()
    s.sample_records(200, 1)                             # the records' model key (memoised by the binding)
    assert s._ctor_digest() == d0
    assert fp(s)["bare"] == b0
