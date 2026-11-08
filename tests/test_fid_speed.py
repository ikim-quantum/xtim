"""FID (fidelity-mode speed, lean/FID): the four fixes of the analysis, each with its oracle.

fix 1: a raw kind-2 content with Z-type axes only (a rotation circuit's) is interned by its closed diagonal form
Δ·S^{a&1}·CZ — the factor route is the oracle on every content of the buffer (max |ΔF| measured: 0.0).
fix 2: the kind-2 content readers that read the term view's axes only (RecordFidelity, signs, frames, the plan
structure, a branch-0 materialize) never arm (complete) a view that exists — a fresh sampler's results equal an armed
sampler's bit for bit, and the fresh sampler's z-screened view stays unarmed.
fix 3: the factor route's expansion (non-diagonal CH / PPR contents) is value-identical to the 6ea3308 engine (pins).
fix 4: the boundary pass is always the coldest-first partial eviction — test_rf_retention.py.
"""
import os
import warnings

import numpy as np
import pytest

from xtim import _xtim_testapi as TA
from xtim.record_fidelity import RecordFidelity

pytestmark = pytest.mark.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _fam(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(open(os.path.join(DATA, name)).read(), selfcheck=0)


def _z(n, qs):
    z = np.zeros(n, np.uint8)
    z[list(qs)] = 1
    return (np.zeros(n, np.uint8), z, 0)


def _same(a, b):
    """deep equality of the readers' exports (dicts / lists / tuples / arrays), arrays bit for bit"""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b)
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    else:
        assert a == b or (a != a and b != b), (a, b)


# ── fix 2 ────────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["magna_rp2_fid.stim", "magna_color5_fid.stim"])
def test_fix2_axes_readers_do_not_arm_and_equal_the_armed_sampler(name):
    fresh, armed = _fam(name), _fam(name)
    assert fresh._term_view_state() == 0 and armed._term_view_state() == 0   # a z-screened view (rotation circuit)
    armed.term_rank()                                                        # the test API arms the view
    assert armed._term_view_state() == 1
    out = {}
    for tag, s in (("fresh", fresh), ("armed", armed)):
        buf = s.sample_barrier(600, seed=11)
        F = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True), np.float64)
        F4 = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True, threads=4), np.float64)
        R = s.sample_records(60, seed=5)[3]
        ops = s.record_ops()
        n = len(ops[0][0])
        g = list(ops[:4]) + [_z(n, [0]), _z(n, [1, 2])]
        sg = np.asarray(s.signs(R, g))
        fr = s.frames(R)
        ps = s.record_plan_structure(R, 0)
        mat = [s.state_expectations(R, i, g) for i in range(3)]
        out[tag] = (F, F4, sg, fr, ps, mat)
    assert fresh._term_view_state() == 0                                     # fix 2: nothing armed it
    a, b = out["fresh"], out["armed"]
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    assert np.any(a[0] > 0) and np.any(a[0] != a[0][0])                      # not vacuous
    assert np.array_equal(a[2], b[2])
    _same(a[3], b[3])
    _same(a[4], b[4])
    for x, y in zip(a[5], b[5]):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


# ── fix 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# A raw kind-2 content whose axes are all Z-type (a rotation circuit's) is interned by its closed diagonal form
# Δ·S^{a&1}·CZ (Δ folded into the row's Pauli) instead of its factor list. Oracle: the factor route (test API
# _fid_routes(oracle=True)) on the same buffer, every row — hence every content the buffer reads — to float precision.
# MUT-FID-DELTA (Δ's Z part dropped) and MUT-FID-SIGN (S^{+c} for S^{−c}) are RED here.
EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
FIX1 = {   # name: (path, shots, the circuit has rotation contents)
    "cultivation_d5": (os.path.join(EX, "cultivation_d5.stim"), 3000, True),
    "color5_cultivation": (os.path.join(EX, "color5_cultivation.stim"), 3000, True),
    "magna_color5": (os.path.join(DATA, "magna_color5_fid.stim"), 3000, True),
    "magna_rp2": (os.path.join(DATA, "magna_rp2_fid.stim"), 3000, True),
    "cultivation_d3_faithful": (os.path.join(EX, "cultivation_d3_faithful.stim"), 3000, False),
    "ch_cultivation": (os.path.join(EX, "ch_cultivation.stim"), 3000, False),
    "cube_ccz": (os.path.join(EX, "cube_ccz.stim"), 3000, False),
    "code_switching_faithful": (os.path.join(EX, "code_switching_faithful.stim"), 3000, False),
    "distillation_15_1_3": (os.path.join(EX, "distillation_15_1_3.stim"), 3000, False),
    "factory_d": ("private:ccz_factory_d.stim", 1500, False),
}
FIX1_MAX = {}


def _fam_path(path):
    if path.startswith("private:"):
        from _private import private_fixture
        path = str(private_fixture(path.split(":", 1)[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(open(path).read(), selfcheck=0)


@pytest.mark.parametrize("name", sorted(FIX1))
def test_fix1_diag_route_equals_the_factor_route_on_every_content(name):
    path, N, rot = FIX1[name]
    s = _fam_path(path)
    try:
        RecordFidelity(s)
    except ValueError as e:
        pytest.skip(f"not served by RecordFidelity: {str(e)[:60]}")
    buf = s.sample_barrier(N, seed=21)
    F = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True), np.float64)
    nd, nf = s._fid_routes()
    F4 = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True, threads=4), np.float64)   # the parallel interns
    assert F.tobytes() == F4.tobytes()
    s._fid_routes(oracle=True)
    nd0, nf0 = s._fid_routes()
    G = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True), np.float64)
    nd1, nf1 = s._fid_routes()
    s._fid_routes(oracle=False)
    if rot:
        assert nd > 0 and nf == 0, (nd, nf)                     # every rotation content by the closed form
        assert nd1 == nd0 and nf1 - nf0 == nd, (nd0, nf0, nd1, nf1)   # the oracle: every one by its factors
    assert np.any(F != F[0]) or rot is False                    # not vacuous
    d = float(np.max(np.abs(F - G))) if len(F) else 0.0
    FIX1_MAX[name] = d
    print(f"FIX1 {name}: contents diag {nd} factor {nf}; max |dF| = {d:.3e}")
    assert d <= 1e-12, d


# ── fix 3 ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The factor route's expansion (one parse per content, flat arrays, the pivot × pivot entries of the change of variables
# filled from T directly, rot lists without factor terms) is value-identical: the per-shot F of circuits with
# NON-diagonal kind-2 contents (CH / PPR-retry: the factor route, test_fix1's counters) equals the 6ea3308 engine's bit for
# bit (sha256 of float64 F, sample_barrier(3000, seed 21), keep_memo; threads 1 and 4).
FIX3 = {
    "ch_cultivation": (os.path.join(EX, "ch_cultivation.stim"),
                       "4a0d3ef5fe4c2eb0f3ed70332f70153a8ea52f5134ca3669bc1c4ea020608f3e"),
    "factory_d": ("private:ccz_factory_d.stim", "1e73a640156345efb1e5dc2087b806a0d1c2198c681dc3aea5804e5c8778b469"),
}


@pytest.mark.parametrize("name", sorted(FIX3))
def test_fix3_factor_route_values_unchanged(name):
    import hashlib
    path, pin = FIX3[name]
    s = _fam_path(path)
    buf = s.sample_barrier(3000, seed=21)
    F = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True), np.float64)
    assert s._fid_routes()[1] > 0                                    # factor-route contents were interned
    F4 = np.asarray(RecordFidelity(s).evaluate(buf, keep_memo=True, threads=4), np.float64)
    assert hashlib.sha256(F.tobytes()).hexdigest() == pin
    assert F.tobytes() == F4.tobytes()
