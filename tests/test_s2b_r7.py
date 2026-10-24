"""S2b r7 mechanism gates (the B5 regressions fixed; plan.md "## r7").

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

* r7.4: the local index oracle runs on the selfcheck WINDOW's slow-path shots only (mutation R74 makes it
  fail on every shot it runs on: red inside the window, silent outside it).
"""
import os
import sys
import warnings

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from test_s2b_record_table import FIXTURES  # noqa: E402
from _public import compile_public_twirl_sampler, public_sampler  # noqa: E402


def _slow_shots(s):
    return s.record_report()["route_counts"].get("slow_kappa0", 0)


def _sample_no_oracle_failure(selfcheck):
    s = compile_public_twirl_sampler(FIXTURES["m3_det"], selfcheck=selfcheck)   # J-b2 J1b: the public sampler
    with warnings.catch_warnings():
        warnings.simplefilter("error", xtim.XtimCorrectnessWarning)
        s.sample(6000, seed=3)
    return s


def test_index_oracle_runs_inside_the_window():
    s = _sample_no_oracle_failure(6000)
    assert _slow_shots(s) > 0                     # the oracle's site is reached in the window


def test_index_oracle_does_not_run_outside_the_window():
    s = _sample_no_oracle_failure(0)
    assert _slow_shots(s) > 0                     # slow-path shots exist outside the window


# ── J-b2 J1b: the PRODUCTION legs (plan r2.1 J1b). r7.1's trigger on the public layout-3 record (prefix_words of the
# shot, record_xz of the record operators: a shot whose P_rec has X on a wire some record operator has Z on), with
# S-1 on the public sampler; r7.2's L-3 on the production table (record-space B against the dense oracle,
# test_s2b_record_table.compare_recb). The sigma payload fields r7.2's trigger read (det_signs, coin_masks,
# kernel_masks) have no production counterpart; the transpose (record_table.cpp R72) is the sigma builder's and
# retires with it (J7b).
def test_column_rows_trigger_reached_and_state_equals_bits_public():
    from test_s2b_state_equals_bits import check_state_equals_bits_public
    text = FIXTURES["m3_obs"]
    b = public_sampler(text).sample_barrier(1000, seed=3)
    F = b.frames()
    N = F["record_layout"]["n"]
    pw = np.asarray(F["prefix_words"])
    zc = np.asarray(F["record_xz"][1]).any(0)
    reached = any(((pw[:, q >> 6] >> np.uint64(q & 63)) & np.uint64(1)).astype(bool).any() and zc[q] for q in range(N))
    assert reached
    assert check_state_equals_bits_public(text) > 0


@pytest.mark.parametrize("name", ["m3_obs_2dec", "q3_dec", "sig_comb"])
def test_table_equals_oracle_public(name):
    from test_s2b_record_table import compare_recb
    assert compare_recb(FIXTURES[name]) > 0
