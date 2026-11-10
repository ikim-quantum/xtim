"""lean P3 S-a / S-b (byte-identical speed stages): the sampled bytes, the postselection counts (incl. `early`) and the
plan-memory counters equal the pinned values on every case and leg below. GOLDEN was captured on 5b17329 (xtim-engine-16)
with capture_all() (lean/P3/Sab/tools/golden.py) and RE-PINNED once by the owner-signed lean N1 NOISE-stream change
(xtim-engine-17; lean/N1/REPORT.md "Moved pins") and once by lean N2 (owner-signed disjoint noise counter ranges;
lean/N2/REPORT.md); any other moved digest is a byte change (STOP, not a re-pin).

S-a: the early reject's column read clears a set rotation's flagged columns one WORD at a time (was one bit at a time);
a circuit without rotations / extras skips the identity plan's lookup and its all-zero record rows per fired shot."""
import hashlib
import os
import re

import numpy as np
import pytest

import xtim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_ROOT, rel)) as f:
        return f.read()


def _standin(text):   # T -> S (a circuit without rotations, Pauli faults only)
    return re.sub(r"^(\s*)T_DAG\b", r"\1S_DAG", re.sub(r"^(\s*)T\b", r"\1S", text, flags=re.M), flags=re.M)


def _surface():
    import stim
    return str(stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                                      after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3,
                                      after_reset_flip_probability=1e-3, before_round_data_depolarization=1e-3))


CASES = {
    "cult_d5": lambda: _read("tests/data/cultivation_d5.stim"),
    "cult_d5_S": lambda: _standin(_read("tests/data/cultivation_d5.stim")),
    "cult_d3_rate": lambda: _read("tests/data/cultivation_d3_rate.stim"),
    "cult_d3_faithful": lambda: _read("tests/data/cultivation_d3_faithful.stim"),
    "distill": lambda: _read("tests/data/distillation_15_1_3.stim"),
    "color5": lambda: _read("examples/color5_cultivation.stim"),
    "ch_cult": lambda: _read("tests/data/ch_cultivation.stim"),
    "ja_seed23": lambda: _read("tests/data/ja_seed23.stim"),
    "surface_d5": _surface,
}
LEGS = ("none", "ps_all", "ps_part", "thr4", "budget1", "window")


def _h(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        if a is None:
            h.update(b"-")
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:24]


def _batch(b):   # (dets, obs) or a TwirlBatch (dets, obs, decisions)
    if isinstance(b, tuple):
        return list(b)
    return [b.dets, b.obs, b.decisions]


def _counts(c):
    return [int(c[k]) for k in ("attempts", "passed", "discarded", "early")]


def run_leg(text, leg):
    """One leg: a seeded call and a continued call; returns (digest, counts / counters)."""
    sc = 300 if leg == "window" else 0
    s = xtim.compile_twirl_sampler(text, selfcheck=sc, **({"plan_payload_budget": 1} if leg == "budget1" else {}))
    ndet = s.num_detectors
    out, extra = [], []
    mask = None
    if leg == "ps_all":
        mask = np.ones(ndet, bool)
    elif leg == "ps_part":
        mask = np.zeros(ndet, bool)
        mask[::3] = True
    for i, n in enumerate((1500, 701)):
        kw = {"seed": 7} if i == 0 else {}
        if leg == "thr4":
            kw["threads"] = 4
        if mask is not None:
            r = s.sample(n, postselect=mask, **kw)
            c = r[-1]
            out += _batch(r[0] if len(r) == 2 and not isinstance(r[0], np.ndarray) else tuple(r[:-1]))
            extra += _counts(c)
        else:
            out += _batch(s.sample(n, **kw))
    if leg == "budget1":
        pm = s.record_report()["plan_memory"]
        extra += [int(pm[k]) for k in ("evictions", "payload_builds", "second_chances")]
    return _h(*out), extra


def capture_all():
    return {f"{c}/{leg}": list(run_leg(f(), leg)) for c, f in CASES.items() for leg in LEGS}


GOLDEN = {   # RE-PINNED 2026-11-07 (lean N2, OWNER-SIGNED disjoint noise counter ranges; still xtim-engine-17):
    # capture_all() on the N2 tree (was the N1 capture, before it the 5b17329 one; every leg of a gate-noise case moved,
    # the window / thr4 / budget1 legs still equal the `none` leg of their case: the window, lanes and eviction stay
    # stream-neutral)
    "ch_cult/budget1": ["1249f18c05cc48e9f6464e92", [0, 0, 0]],
    "ch_cult/none": ["1249f18c05cc48e9f6464e92", []],
    "ch_cult/ps_all": ["7a17582310921fd67d474705", [1500, 1498, 2, 0, 701, 697, 4, 0]],
    "ch_cult/ps_part": ["a39c7aabd7ed50693205cd23", [1500, 1499, 1, 0, 701, 699, 2, 0]],
    "ch_cult/thr4": ["1249f18c05cc48e9f6464e92", []],
    "ch_cult/window": ["1249f18c05cc48e9f6464e92", []],
    "color5/budget1": ["2fee895c1982dcdbe0aaa599", [0, 0, 0]],
    "color5/none": ["2fee895c1982dcdbe0aaa599", []],
    "color5/ps_all": ["ba054df69856211745206175", [1500, 226, 1274, 1273, 701, 108, 593, 593]],
    "color5/ps_part": ["48898d62d31196a9c02d876e", [1500, 355, 1145, 1066, 701, 172, 529, 488]],
    "color5/thr4": ["2fee895c1982dcdbe0aaa599", []],
    "color5/window": ["2fee895c1982dcdbe0aaa599", []],
    "cult_d3_faithful/budget1": ["ebcad9b856989b0293895ed5", [0, 0, 0]],
    "cult_d3_faithful/none": ["ebcad9b856989b0293895ed5", []],
    "cult_d3_faithful/ps_all": ["74c4fcc4659fae013f85417d", [1500, 1023, 477, 477, 701, 494, 207, 207]],
    "cult_d3_faithful/ps_part": ["5ac1b7ccd6d4d9da8ad4fc12", [1500, 1150, 350, 330, 701, 545, 156, 145]],
    "cult_d3_faithful/thr4": ["ebcad9b856989b0293895ed5", []],
    "cult_d3_faithful/window": ["ebcad9b856989b0293895ed5", []],
    "cult_d3_rate/budget1": ["acaf2beb216305467f4986a1", [683, 682, 683]],
    "cult_d3_rate/none": ["acaf2beb216305467f4986a1", []],
    "cult_d3_rate/ps_all": ["89bdc26b4d1ccd30e6f3538e", [1500, 1040, 460, 460, 701, 479, 222, 222]],
    "cult_d3_rate/ps_part": ["fc89aeeb1e071feaf4d1c214", [1500, 1150, 350, 350, 701, 532, 169, 169]],
    "cult_d3_rate/thr4": ["acaf2beb216305467f4986a1", []],
    "cult_d3_rate/window": ["acaf2beb216305467f4986a1", []],
    "cult_d5/budget1": ["eebd5708da329580d3978801", [0, 0, 0]],
    "cult_d5/none": ["eebd5708da329580d3978801", []],
    "cult_d5/ps_all": ["d4df4acb65cd713759d29711", [1500, 225, 1275, 1274, 701, 107, 594, 594]],
    "cult_d5/ps_part": ["56930d166a5fa1e0db40c808", [1500, 357, 1143, 1063, 701, 159, 542, 502]],
    "cult_d5/thr4": ["eebd5708da329580d3978801", []],
    "cult_d5/window": ["eebd5708da329580d3978801", []],
    "cult_d5_S/budget1": ["8c7b2397cde06ddceabaee9a", [2201, 2200, 2201]],
    "cult_d5_S/none": ["8c7b2397cde06ddceabaee9a", []],
    "cult_d5_S/ps_all": ["d076554655471ab7e60681d7", [1500, 223, 1277, 1277, 701, 108, 593, 593]],
    "cult_d5_S/ps_part": ["919240a529af1750f657074c", [1500, 366, 1134, 1134, 701, 162, 539, 539]],
    "cult_d5_S/thr4": ["8c7b2397cde06ddceabaee9a", []],
    "cult_d5_S/window": ["8c7b2397cde06ddceabaee9a", []],
    "distill/budget1": ["10251a68111fe4edee3046d5", [0, 0, 0]],
    "distill/none": ["10251a68111fe4edee3046d5", []],
    "distill/ps_all": ["e64f52dea5b190193fff87da", [1500, 1482, 18, 18, 701, 691, 10, 10]],
    "distill/ps_part": ["c78974818802aa51075de1d9", [1500, 1484, 16, 16, 701, 694, 7, 7]],
    "distill/thr4": ["10251a68111fe4edee3046d5", []],
    "distill/window": ["10251a68111fe4edee3046d5", []],
    "ja_seed23/budget1": ["4633f74ee289add07e46b43a", [0, 0, 0]],
    "ja_seed23/none": ["4633f74ee289add07e46b43a", []],
    "ja_seed23/ps_all": ["c72f71df66d5d0dd2b0a0ead", [1500, 965, 535, 498, 701, 426, 275, 253]],
    "ja_seed23/ps_part": ["91bbf1f4f266cf68060ea5c9", [1500, 1126, 374, 315, 701, 513, 188, 154]],
    "ja_seed23/thr4": ["4633f74ee289add07e46b43a", []],
    "ja_seed23/window": ["4633f74ee289add07e46b43a", []],
    "surface_d5/budget1": ["be05dd9d965decb40a493099", [1251, 1250, 1251]],
    "surface_d5/none": ["be05dd9d965decb40a493099", []],
    "surface_d5/ps_all": ["13aa64558855f9419daa22bb", [1500, 646, 854, 854, 701, 304, 397, 397]],
    "surface_d5/ps_part": ["48ed5cd6f0d73859c91e7a71", [1500, 967, 533, 533, 701, 475, 226, 226]],
    "surface_d5/thr4": ["be05dd9d965decb40a493099", []],
    "surface_d5/window": ["be05dd9d965decb40a493099", []],
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("leg", LEGS)
def test_bytes_and_counts_equal_base(case, leg):
    dig, extra = run_leg(CASES[case](), leg)
    assert [dig, extra] == GOLDEN[f"{case}/{leg}"]


# ── S-b: the shot block (serial runs) == the event list (the oracle route, TA.shot_block(s, False)) ─────────────────────
# Same sampler construction, the same call sequence (a seeded call, then continued calls across block boundaries);
# every array, the PS counts incl. `early`, the plan store's / plan cache's counters must be equal, and the block route
# must have served every shot (the event-list sampler none).
from xtim import _xtim_testapi as TA   # noqa: E402

CALLS = (3000, 1, 4097, 777)


def _mask(s, kind):
    nd = s.num_detectors
    if kind == "none":
        return None
    m = np.zeros(nd, bool)
    m[:: (1 if kind == "all" else 3)] = True
    return m


def _seq(s, mask, calls=CALLS, threads=1, seed=5):
    out = []
    for i, n in enumerate(calls):
        sd = seed if i == 0 else None
        if mask is None:
            out += [np.asarray(a).copy() for a in s.sample(n, sd, threads=threads)]
        else:
            *arrs, c = s._sample_ps(n, sd, mask, threads)
            out += [np.asarray(a).copy() for a in arrs] + [sorted(dict(c).items())]
    return out


def _state(s):
    st = TA.rot_store_stats(s)
    pm = s.record_report()["plan_memory"]
    return ({k: st[k] for k in ("plans", "tuples", "misses_k0", "misses_kpos")} if st["on"] else None,
            {k: pm[k] for k in ("payload_builds", "evictions", "second_chances", "classes")})


def _eq(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            assert x.shape == y.shape and np.array_equal(x, y)
        else:
            assert x == y


def _pair(text, **kw):
    a = TA.compile_family_sampler(text, selfcheck=kw.pop("selfcheck_a", 0))
    b = TA.compile_family_sampler(text, selfcheck=kw.pop("selfcheck_b", 0))
    TA.shot_block(a, False)
    for s in (a, b):
        if "budget" in kw:
            s.set_plan_payload_budget(kw["budget"])
    return a, b


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("kind", ["none", "all", "part"])
def test_shot_block_equals_event_list(case, kind):
    a, b = _pair(CASES[case]())
    ra, rb = _seq(a, _mask(a, kind)), _seq(b, _mask(b, kind))
    _eq(ra, rb)
    assert _state(a) == _state(b)
    assert TA.shot_block_shots(a) == 0
    attempts = sum(CALLS)
    assert TA.shot_block_shots(b) == attempts


@pytest.mark.parametrize("case", ["cult_d3_rate", "cult_d5_S", "surface_d5", "cult_d5", "color5"])
def test_shot_block_under_budget(case):
    a, b = _pair(CASES[case](), budget=1)
    _eq(_seq(a, None), _seq(b, None))
    assert _state(a) == _state(b)
    assert TA.shot_block_shots(b) == sum(CALLS)


@pytest.mark.parametrize("case", ["cult_d5", "surface_d5", "ja_seed23"])
def test_shot_block_window_then_blocks(case):
    """The window's shots take the event list (its oracles read the events); every later shot takes shot blocks."""
    a, b = _pair(CASES[case](), selfcheck_b=300)
    _eq(_seq(a, None), _seq(b, None))
    assert sum(CALLS[1:]) <= TA.shot_block_shots(b) < sum(CALLS)


@pytest.mark.parametrize("case", ["cult_d5", "surface_d5", "color5"])
@pytest.mark.parametrize("kind", ["none", "all"])
def test_threads_equal_shot_block(case, kind):
    """threads N (lanes: the event list) == threads 1 (shot blocks)."""
    a, b = _pair(CASES[case]())
    TA.shot_block(a, True)
    _eq(_seq(a, _mask(a, kind), threads=4), _seq(b, _mask(b, kind), threads=1))
    assert TA.shot_block_shots(b) == sum(CALLS)


# ── M3: a call that straddles the window boundary: window shots on the event list, the rest on shot blocks ─────────────
@pytest.mark.parametrize("case", ["cult_d5", "surface_d5", "ja_seed23"])
@pytest.mark.parametrize("kind", ["none", "all"])
def test_shot_block_straddles_window(case, kind):
    calls = (5000, 1, 4097)
    text = CASES[case]()
    a = TA.compile_family_sampler(text, selfcheck=500)        # (a) block route forced off
    b = TA.compile_family_sampler(text, selfcheck=500)        # the straddling sampler
    c = TA.compile_family_sampler(text, selfcheck=500)        # (b) event list throughout (also through a spent window)
    TA.shot_block(a, False)
    TA.shot_block(c, False)
    rb = _seq(b, _mask(b, kind), calls=calls, seed=11)
    _eq(_seq(a, _mask(a, kind), calls=calls, seed=11), rb)
    _eq(_seq(c, _mask(c, kind), calls=calls, seed=11), rb)
    assert _state(a) == _state(b) == _state(c)
    assert TA.shot_block_shots(a) == 0 == TA.shot_block_shots(c)
    sb = TA.shot_block_shots(b)
    assert sum(calls[1:]) <= sb < sum(calls)                 # the window's shots stayed on the event list, 1st call split
