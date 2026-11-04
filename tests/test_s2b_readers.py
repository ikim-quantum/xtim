"""S2b r6.1 (code review MAJOR-3): the record READERS resolve a plan's record table once per distinct
plan and derive every shot's y word-parallel; exact-fallback shots never touch the table lookup
(their y is their stored record, Q4).

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

* frames()['y'] == record_outcomes(i) on every shot (the readers' per-plan path vs the per-shot
  accessor), on the compact and the legacy store.
* record_report()['plan_memory']['reader_lookups'] grows by exactly the number of DISTINCT plans of
  the record-route shots per reader call (frames, RecordFidelity.evaluate) — never per shot.
* Under QEC_TW_FORCE_RECORD_FALLBACK=1 (child process) every faulted shot is exact: the lookups
  count only the record-route shots' plans, so an exact shot reaching the lookup is red.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from test_s2b_record_table import FIXTURES  # noqa: E402

NAMES = ["m3_obs_dec", "m3_obs_2dec", "a4_comb", "a4_minus_z", "q3_dec", "sig_comb"]


# ── J-b2 J1b: the PRODUCTION legs (plan r2.1 J1b). frames()['y'] == record_outcomes(i) on the public layout-3
# barrier; a reader builds each needed table at most once (production serves a resident plan's table from the run's
# own plan — J-b1 P12 F3 — and memoizes the rest: a SECOND frames() / RecordFidelity.evaluate builds no new reader
# table, and RecordFidelity's values repeat). Production has no exact shot, so the legacy exact-bypass gate has no
# production object (retired with QEC_TW_FORCE_RECORD_FALLBACK, J-b1 F).
from _public import public_sampler  # noqa: E402


@pytest.mark.parametrize("name", NAMES)
def test_frames_y_equals_record_outcomes_public(name):
    s = public_sampler(FIXTURES[name])
    b = s.sample_barrier(3000, seed=9)
    F = b.frames()
    y = np.asarray(F["y"])
    for i in range(0, 3000, 7):
        assert np.array_equal(y[i], np.asarray(b.record_outcomes(i))), (name, i)
    s.record_report()                                      # record_report itself reads the identity plan's table
    t1 = s.record_report()["plan_memory"]["reader_tables"]
    F2 = b.frames()
    assert np.array_equal(np.asarray(F2["y"]), y)
    assert s.record_report()["plan_memory"]["reader_tables"] == t1          # a second reader builds nothing
    assert len(set(np.asarray(F["plan_index"]).tolist())) > 1              # non-vacuous: several plans


@pytest.mark.parametrize("name", ["m3_obs", "m3_det", "wlam", "ccz_obs"])
def test_record_fidelity_builds_each_table_once_public(name):
    from xtim.record_fidelity import RecordFidelity
    s = public_sampler(FIXTURES[name] + "R 99\nOUTPUT_QUBITS out 99\n")   # an output register (strict contract)
    b = s.sample_barrier(3000, seed=2)
    rf = RecordFidelity(s)
    f1 = np.asarray(rf.evaluate(b))
    s.record_report()                                      # record_report itself reads the identity plan's table
    t1 = s.record_report()["plan_memory"]["reader_tables"]
    f2 = np.asarray(rf.evaluate(b))
    assert s.record_report()["plan_memory"]["reader_tables"] == t1
    assert np.array_equal(f1, f2)
