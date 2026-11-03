"""fid-orth (OD-2026-10-24-fid-orth): RecordFidelity route 3b decides orthogonality first.

Per content the constraint system is row-reduced once (stage 1); per row N^T b decides consistency (orthogonal rows
score 0 with nothing built or stored), R b is the particular solution; a row whose b equals the content's previous b
reuses that projection; the value structure (stage 2) is built on the content's first consistent row. Values are
byte-identical to the parent (the pinned per-shot SHAs in test_jb1_fidelity_vs_production.py are unchanged).
Mutations: XTIM_MUTATE_RF_ORTH_SKIP (every row consistent) breaks the pins and the classes below;
XTIM_MUTATE_RF_LASTB_STALE (a one-word compare of b) breaks the order independence below.
"""
import os
import sys

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from xtim.record_fidelity import RecordFidelity
from _private import private_fixture

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "ledger"))
try:  # the performance ledger (benchmarks/ledger) is not part of the public tree; its legs skip when absent
    import s8_fiddec as LEGS  # noqa: E402
except ImportError:
    LEGS = None

ROWS = 3000


def _inputs(leg):
    if LEGS is None:
        pytest.skip("benchmarks/ledger (s8_fiddec) not present")
    return LEGS.inputs(ROOT, leg, testapi=True)


def _ev(s, buf, corr, shots, log=True):
    rf = RecordFidelity(s)
    if log:
        TA._rf_row_log(rf._rf, True)
    F = np.asarray(rf.evaluate(buf, shots, correction=corr))
    cls = np.asarray(TA._rf_take_row_log(rf._rf)) if log else None
    return F, cls, rf.stats()


@pytest.mark.parametrize("leg", ["color5/p1e-3/dec", "rp2/p1e-3/dec", "color5/p1e-3/zero"])
def test_projection_reuse_is_order_independent(leg):
    """The per-content previous-b projection is a pure speed device: any row order gives the same per-row bytes and
    classes (memo bypassed, so the projection reuse is the only state carried between rows), and it is exercised."""
    s, buf, corr = _inputs(leg)
    rows = np.arange(ROWS, dtype=np.int64)
    F, cls, st = _ev(s, buf, corr, rows)
    assert st["projection_skips"] > 0
    for perm in (rows[::-1].copy(), np.random.default_rng(5).permutation(rows),
                 np.concatenate([rows[0::2], rows[1::2]])):
        Fp, clsp, _ = _ev(s, buf, corr, perm)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        assert Fp[inv].tobytes() == F.tobytes() and np.array_equal(clsp[inv], cls), leg


@pytest.mark.parametrize("leg", ["color5/p1e-3/rand", "color5/p1e-3/zero", "color5/p1e-4/zero"])
def test_orthogonal_rows_build_and_store_nothing(leg):
    """Orthogonal rows (class 0) are exactly the rows counted orthogonal; they score 0, are neither memo hits nor
    misses (no memo entry), and a content whose rows are all orthogonal never builds its value structure. (The decoded
    legs and rp2 have no orthogonal rows: their constraint test always passes.)"""
    s, buf, corr = _inputs(leg)
    rows = np.arange(ROWS, dtype=np.int64)
    F, cls, st = _ev(s, buf, corr, rows)
    orth = int((cls == 0).sum())
    assert st["orthogonal_rows"] == orth > 0 and np.all(F[cls == 0] == 0.0)
    assert st["misses"] == ROWS - orth and st["memo_hits"] == 0           # (row log: memo bypassed)
    assert 0 < st["value_builds"] <= st["kernel_compiles"] == st["plans"]
    rf = RecordFidelity(s)
    Fm = np.asarray(rf.evaluate(buf, rows, correction=corr, keep_memo=True))
    stm = rf.stats()
    assert Fm.tobytes() == F.tobytes()
    assert stm["memo_hits"] + stm["misses"] == ROWS - orth and stm["orthogonal_rows"] == orth
    if leg.endswith("rand"):          # random corrections: most contents never see a consistent row
        assert st["value_builds"] < st["kernel_compiles"]


# The parent (52577f7) held 60330494 bytes on this buffer (ccz_factory_d, 5000 shots, seed 8,
# cache_budget = inf); its factor-list contents freed their record images at the first-row build.
FAC_D_BASE_HELD = 60330494


def test_factor_list_held_bytes_not_above_parent():
    """Review MAJOR-1 / MINOR-2 (fid-orth fix round): a factor-list (kind-2) content builds its record split at stage 2
    from its own factor list (never at intern, never charged twice), so contents that only see orthogonal rows hold no
    record images: the evaluator holds no more than the parent, and a 72 MiB budget the parent met is met (the tip
    before the fix held 79 MB here and refused it). Values are unchanged under the budget."""
    s = xtim.compile_twirl_sampler(private_fixture("ccz_factory_d.stim").read_text(), selfcheck=0)
    buf = s.sample_barrier(5000, seed=8)
    rf = RecordFidelity(s, cache_budget=float("inf"))
    F = np.asarray(rf.evaluate(buf))
    st = rf.stats()
    assert st["cache_flushes"] == 0 and 0 < st["value_builds"] < st["plans"]
    assert st["held_bytes"] <= FAC_D_BASE_HELD, st["held_bytes"]
    Fb = np.asarray(RecordFidelity(s, max_bytes=72 * 2**20).evaluate(buf))
    assert Fb.tobytes() == F.tobytes()
