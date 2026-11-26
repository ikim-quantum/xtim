"""The public xdem API (SPEC §4.6)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import xtim

from _xdem_circuits import circuit, xdem_export

NOTHING_BEFORE_TC = """R 0 1
X_ERROR(0.05) 0
T 0
M 0 1
OBSERVABLE_INCLUDE(0) rec[-2]
OBSERVABLE_INCLUDE(1) rec[-2]
OBSERVABLE_INCLUDE(2) rec[-1]
"""
# checked with the v1 export at 19f2f78: D = 0, O = 3, one axis (Z0) with no img and no kappa (so one hidden-syndrome
# qubit), one rotating line flipping L0 L1; no detector record, so t_c is the start: tc 0, the rotation in `late`;
# L2 is an observable no line touches


def test_properties_and_shapes():
    s = xtim.XdemSampler(xdem_export(circuit("ex422_obs")))
    assert (s.extended, s.num_detectors, s.num_observables, s.num_gauge, s.num_hidden, s.num_axes, s.num_lines,
            s.num_rotating, s.tc) == (True, 3, 2, 3, 2, 8, 3, 2, 3)
    assert s.late_axes == () and s.r2b_available[0] is True and s.invisible_residues >= 0
    dets, st = s.sample(1000, seed=1)
    assert dets.shape == (1000, 1) and st.obs.shape == (1000, 1) and st.cspace.shape == (1000,)
    p = s.score(st, np.zeros((1000, 2), bool))
    assert p.dtype == np.float64 and p.shape == (1000,)
    with pytest.raises(TypeError):
        s.sample(10, 1)                                                     # seed is keyword-only


def test_defaults_are_r2b():
    """Owner decision 2026-11-16 (overrides the brief's "optimal"): score / infidelity / correction default to r2b."""
    import inspect
    sig = lambda f: inspect.signature(f).parameters["correction"].default            # noqa: E731
    assert sig(xtim.XdemSampler.correction) == "r2b"
    assert sig(xtim.XdemSampler.score) is None and sig(xtim.XdemSampler.infidelity) is None   # None = r2b on lines=
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre_p1e-2")))
    _, st = s.sample(2000, seed=5, fired=True)
    L = fired_lists(st)
    assert np.array_equal(s.score(st, lines=L), s.score(st, lines=L, correction="r2b"))
    dec = lambda d: [[]] * len(d)                                                     # noqa: E731
    assert s.infidelity(3000, dec, seed=6, lines=True) == s.infidelity(3000, dec, seed=6, lines=True,
                                                                       correction="r2b")
    assert s.correction(st, lines=L).kind == "r2b"


def test_score_lines_corrections_and_status_requirements():
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre")))
    dets, st = s.sample(500, seed=2, fired=True)
    L = [st.fired[1][st.fired[0][k]:st.fired[0][k + 1]] for k in range(500)]
    for corr in ("optimal", "r2b", "none"):
        assert s.score(st, lines=L, correction=corr).shape == (500,)
    _, st0 = s.sample(10, seed=2)
    with pytest.raises(ValueError, match="fired=True"):
        s.score(st0, lines=[[]] * 10)
    with pytest.raises(ValueError, match="correction"):
        s.score(st, lines=L, correction="vdag")
    assert (s.score(st, lines=L) == s.score(st, lines=L, correction="r2b")).all()   # the default (owner 2026-11-16)


def test_correction_export_kinds():
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre")))
    dets, st = s.sample(300, seed=3, fired=True)
    L = [st.fired[1][st.fired[0][k]:st.fired[0][k + 1]] for k in range(300)]
    c = s.correction(st, lines=L, correction="optimal")
    assert c.kind == "optimal" and c.y.shape == (300, s.num_observables) and c.lossy.dtype == bool
    r = s.correction(st, lines=L)                                           # the default (owner 2026-11-16)
    assert r.kind == "r2b" and r.applied.shape == (300,)


def test_infidelity_lines_equals_sample_then_score():
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre")))
    dec = lambda d: [[] for _ in range(len(d))]                              # noqa: E731
    r = s.infidelity(5000, dec, seed=4, lines=True, correction="r2b", batch=2048)
    dets, st = s.sample(5000, seed=4, fired=True)
    p = s.score(st, lines=[[]] * 5000, correction="r2b")
    assert abs(r.mean - (1 - p.mean())) < 1e-12 and r.accepted == 5000


def test_conventional_dem_is_read_without_a_flag():
    import stim
    g = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.01)
    s = xtim.XdemSampler(g.detector_error_model(decompose_errors=True))
    assert not s.extended and s.tc == s.num_lines and s.num_axes == 0


def test_errors_and_gone_names():
    v1 = Path(__file__).parent / "data/xdem/v1/cultivation_d3_faithful.dem"
    with pytest.raises(xtim.XtimXdemError, match="dc-DEM v1"):
        xtim.XdemSampler(v1.read_text())
    assert issubclass(xtim.XtimXdemError, xtim.XtimError)


def test_degenerate_shapes():                                               # Review Focus 4
    x = xdem_export(NOTHING_BEFORE_TC)
    assert x.startswith("xdem 1 detectors 0 observables 3 gauge 0 hidden 1 tc 0\n")
    s = xtim.XdemSampler(x)
    assert (s.num_detectors, s.num_observables, s.tc, s.num_rotating) == (0, 3, 0, 1)
    ok, why = s.r2b_available
    assert ok is False and "carries a rotation" in why
    dets, st = s.sample(200, seed=1, fired=True)
    assert dets.shape == (200, 0) and st.obs.shape == (200, 1)
    L = [st.fired[1][st.fired[0][k]:st.fired[0][k + 1]] for k in range(200)]
    for corr in ("optimal", "none"):
        assert s.score(st, lines=L, correction=corr).shape == (200,)
    with pytest.raises(xtim.XtimXdemError, match='carries a rotation; pass correction="optimal" or "none"'):
        s.score(st, lines=L, correction="r2b")
    c = s.correction(st, lines=L, correction="optimal")
    assert c.y.shape == (200, 3) and not c.lossy.any()
    r = s.infidelity(500, lambda d: [[] for _ in range(len(d))], seed=2, lines=True, correction="optimal")
    assert r.accepted == 500
    for call in (lambda: s.score(st, lines=L), lambda: s.correction(st, lines=L),     # the default is "r2b"
                 lambda: s.infidelity(10, lambda d: [[]] * len(d), seed=2, lines=True)):
        with pytest.raises(xtim.XtimXdemError, match='carries a rotation; pass correction="optimal" or "none"'):
            call()


# ── ported from tests/test_dc_api.py: postselect forms, seed rules, giveup, bit_packed, score_plain, batch rule
DATA = Path(__file__).parent / "data" / "xdem"

# Several observables with gauge-qubit readouts (tests/test_dc_api.py::HAND): dim C in {0, 1} across shots.
HAND = """
RX 0 1 2
T 0 2
X_ERROR(0.3) 0
X_ERROR(0.4) 2
T_DAG 0 2
Z_ERROR(0.2) 1
MX 0 1 2
DETECTOR rec[-3] rec[-1]
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-3]
OBSERVABLE_INCLUDE(1) rec[-3] rec[-2]
OBSERVABLE_INCLUDE(2) rec[-1]
"""

# A Clifford-only circuit: its xdem export is the conventional DEM (no axes, no gauge qubits).
CLIFF = """
R 0 1 2
X_ERROR(0.1) 0 1 2
CX 0 1
DEPOLARIZE2(0.05) 1 2
M 0 1 2
DETECTOR rec[-3] rec[-2]
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
OBSERVABLE_INCLUDE(1) rec[-3]
"""


def hand():
    return xtim.XdemSampler(xdem_export(HAND))


def d5():
    return xtim.XdemSampler((DATA / "cultivation_d5.xdem").read_text())


def unpack(packed, n):
    if n == 0:
        return np.zeros((packed.shape[0], 0), dtype=bool)
    return np.unpackbits(packed, axis=1, bitorder="little", count=n).astype(bool)


def zeros_for(s):
    return lambda dets: np.zeros((dets.shape[0], s.num_observables), dtype=bool)


def fired_lists(st):
    off, ids = st.fired
    return [ids[off[k]:off[k + 1]] for k in range(len(st))]


def test_exports():
    assert xtim.XdemSampler is xtim.xdem.XdemSampler
    assert xtim.score_plain is xtim.xdem.score_plain                       # preflight S8
    assert xtim.XtimXdemError is xtim.errors.XtimXdemError
    for name in ("XdemSampler", "XtimXdemError", "score_plain"):
        assert name in xtim.__all__
    assert set(xtim.xdem.__all__) == {"XdemSampler", "XdemStatus", "XdemCorrection", "InfidelityResult",
                                      "canonical_report", "score_plain"}


def test_shapes_and_dtypes():
    s = hand()
    assert s.extended and (s.num_detectors, s.num_observables) == (2, 3)
    dets, st = s.sample(5000, seed=701)
    assert isinstance(st, xtim.xdem.XdemStatus) and len(st) == 5000 and st.fired is None
    assert dets.dtype == np.uint8 and dets.shape == (5000, 1)
    assert st.obs.dtype == np.uint8 and st.obs.shape == (5000, 1)
    assert st.cspace.dtype == np.int32 and st.cspace.shape == (5000,)
    assert isinstance(st.cspaces, list) and st.cspaces[0].shape == (0, 3)
    assert all(c.dtype == np.uint8 and c.ndim == 2 and c.shape[1] == 3 for c in st.cspaces)
    assert {c.shape[0] for c in st.cspaces} == {0, 1}                       # HAND reaches dim C = 1
    assert 0 <= st.cspace.min() and st.cspace.max() < len(st.cspaces)
    sc = s.score(st, np.zeros((5000, 3), dtype=bool))
    assert sc.dtype == np.float64 and sc.shape == (5000,)
    assert set(np.unique(sc)) <= {0.0, 0.5, 1.0}
    c = s.counters()
    assert c["shots"] == 5000 and "fast_path" in c
    for k in ("lines_shots", "lines_applied", "lines_withheld", "lines_coin_after", "lines_lossy", "kmemo_entries",
              "kmemo_clears", "kfast_laws"):
        assert k in c
    s5 = d5()                                                               # 107 detectors, 1 observable
    dets5, st5 = s5.sample(3000, seed=702)
    assert dets5.shape == (3000, 14) and st5.obs.shape == (3000, 1)


def test_seed_keyword_only_and_required():
    s = hand()
    with pytest.raises(TypeError):
        s.sample(10)
    with pytest.raises(TypeError):
        s.sample(10, 3)
    with pytest.raises(TypeError):
        s.infidelity(10, zeros_for(s))
    with pytest.raises(TypeError):
        s.infidelity(10, zeros_for(s), 3)
    with pytest.raises(TypeError):
        s.sample(10, seed=1.5)
    with pytest.raises(ValueError):
        s.sample(10, seed=-1)
    with pytest.raises(ValueError):
        s.sample(10, seed=2**64)
    with pytest.raises(ValueError):
        s.sample(10, seed=1, threads=0)
    with pytest.raises(ValueError):
        s.sample(-1, seed=1)
    with pytest.raises(TypeError):
        s.sample(10, seed=1, fired=1)


def test_sample_is_the_binding_stream_and_replays():
    text = (DATA / "cultivation_d5.xdem").read_text()
    s = xtim.XdemSampler(text)
    raw = xtim._xtim._XdemSampler(text)
    d, st = s.sample(20000, seed=703, threads=4)
    rd, ro, rc = raw.sample(20000, 703, 1)
    assert np.array_equal(d, rd) and np.array_equal(st.obs, ro) and np.array_equal(st.cspace, rc)
    d2, st2 = s.sample(20000, seed=703, threads=1)                          # same seed replays (any threads)
    assert np.array_equal(d, d2) and np.array_equal(st.obs, st2.obs)
    d3, _ = s.sample(20000, seed=704)                                       # new seed, new shots
    assert not np.array_equal(d, d3)
    df, stf = s.sample(20000, seed=703, fired=True)                         # fired=True: the same bytes
    assert df.tobytes() == d.tobytes() and stf.obs.tobytes() == st.obs.tobytes()
    assert stf.fired[0].shape == (20001,) and stf.fired[0].dtype == np.int64 and stf.fired[1].dtype == np.int32


def test_score_layouts_and_shape_errors():
    s = hand()
    _, st = s.sample(4000, seed=705)
    rng = np.random.default_rng(705)
    f = rng.integers(0, 2, size=(4000, 3)).astype(bool)
    a = s.score(st, f)
    b = s.score(st, f.astype(np.int64))
    packed = np.packbits(f, axis=1, bitorder="little")
    c = s.score(st, packed, bit_packed=True)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    assert np.array_equal(a, s.score(st, packed | np.uint8(0xF8), bit_packed=True))   # padding bits ignored
    for bad in (np.zeros((4000, 2), bool), np.zeros((3999, 3), bool), np.zeros(4000, bool)):
        with pytest.raises(ValueError):
            s.score(st, bad)
    with pytest.raises(ValueError):
        s.score(st, np.zeros((4000, 2), np.uint8), bit_packed=True)
    with pytest.raises(ValueError):
        s.score(st, np.full((4000, 3), 2))
    with pytest.raises(ValueError):                                         # a status of another sampler
        hand().score(st, f)
    with pytest.raises(TypeError):
        s.score(st)                                                         # neither flips nor lines=
    with pytest.raises(TypeError):
        s.score(st, f, correction="optimal")                                # correction= applies to lines=
    for corr in ("r2b", "none", "vdag"):                                    # any value, the default's name included
        with pytest.raises(TypeError, match="applies to lines="):
            s.score(st, f, correction=corr)
    assert np.array_equal(s.score(st, f, correction=None), a)
    with pytest.raises(TypeError):
        s.score(st.obs, f)


def test_score_lines_argument_rules():
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre")))
    dets, st = s.sample(400, seed=706, fired=True)
    L = fired_lists(st)
    a = s.score(st, lines=L, correction="none")
    assert np.array_equal(a, s.score(st, lines=st.fired, correction="none"))      # a CSR pair: the same scores
    with pytest.raises(TypeError):
        s.score(st, np.zeros((400, 1), bool), lines=L)
    with pytest.raises(TypeError):
        s.score(st, lines=L, bit_packed=True)
    with pytest.raises(ValueError, match="CSR"):
        s.score(st, lines=tuple(L[:3]))
    with pytest.raises(ValueError, match="outside"):
        s.score(st, lines=[[s.num_lines]] + [[]] * 399)
    with pytest.raises(ValueError):
        s.score(st, lines=[[]] * 399)
    with pytest.raises(ValueError, match="correction"):
        s.correction(st, lines=L, correction="none")                       # nothing to export
    _, st0 = s.sample(10, seed=706)
    with pytest.raises(ValueError, match="fired=True"):
        s.correction(st0, lines=[[]] * 10)


def test_correction_layout():
    """XdemCorrection maps the C++ export (task-8 report): "optimal" rows are shot * O + k; "r2b" rows are shots."""
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre_p1e-2")))
    n, O = 2000, s.num_observables
    dets, st = s.sample(n, seed=707, fired=True)
    L = fired_lists(st)
    raw = s._s.correction(dets, *xtim.xdem._lines_csr(L, n, s.num_lines, "t"), "optimal")
    c = s.correction(st, lines=L, correction="optimal")
    for name, (off, ids) in (("axes", c.axes), ("piv", c.pivots)):
        assert off.shape == (n * O + 1,) and np.array_equal(off, raw[name + "_off"])
        assert np.array_equal(ids, raw[name])
    for name in ("y", "frame", "lossy"):
        v = getattr(c, name)
        assert v.dtype == bool and v.shape == (n, O)
        assert np.array_equal(v.reshape(-1), raw[name].astype(bool))
    assert c.late is None and c.late_tau is None and c.applied is None and c.c is None
    r = s.correction(st, lines=L)                                           # the default: "r2b"
    raw = s._s.correction(dets, *xtim.xdem._lines_csr(L, n, s.num_lines, "t"), "r2b")
    assert np.array_equal(r.late[0], raw["late_off"]) and np.array_equal(r.late[1], raw["late_axis"])
    assert np.array_equal(r.late[2], raw["late_eta"]) and r.late[0].shape == (n + 1,)
    assert np.array_equal(r.late_tau[0], raw["tau_off"]) and np.array_equal(r.late_tau[1], raw["tau_axis"])
    assert r.applied.dtype == bool and np.array_equal(r.applied, raw["applied"].astype(bool))
    assert r.c.dtype == bool and r.c.shape == (n, O) and np.array_equal(r.c.reshape(-1), raw["c"].astype(bool))
    assert r.axes is None and r.pivots is None and r.y is None and r.frame is None and r.lossy is None
    # semantic invariants, both kinds
    D, A, late = s.num_detectors, s.num_axes, set(s.late_axes)

    def rows(off, ids, hi):
        assert off[0] == 0 and (np.diff(off) >= 0).all() and off[-1] == len(ids)
        out = [ids[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        assert all(((x >= 0) & (x < hi)).all() and (np.diff(x) > 0).all() for x in out)   # in range, ascending
        return out
    S, P = rows(*c.axes, A), rows(*c.pivots, D)
    lossy = c.lossy.reshape(-1)
    assert all(len(S[i]) == 0 and len(P[i]) == 0 for i in np.flatnonzero(lossy))
    assert not c.frame.reshape(-1)[lossy].any()
    assert sum(len(x) > 0 for x in S) > 0                                   # some rotated readout is exercised
    eta_rows = rows(r.late[0], r.late[1], A)
    assert all(set(x.tolist()) <= late for x in eta_rows) and ((r.late[2] >= 1) & (r.late[2] <= 3)).all()
    assert all(set(x.tolist()) <= late for x in rows(*r.late_tau, A))
    assert all(len(eta_rows[i]) > 0 for i in np.flatnonzero(r.applied))    # applied => eta^> != 0
    # with the true hypothesis (the fired lines), a shot with no lossy readout passes with certainty (SPEC §4.3)
    p = s.score(st, lines=L, correction="optimal")
    ok = ~c.lossy.any(axis=1)
    assert ok.sum() > 0 and (p[ok] == 1.0).all()
    # r2b's export agrees with score(lines=, correction="r2b") through the counters: applied shots = lines_applied
    before = s.counters()["lines_applied"]
    s.score(st, lines=L, correction="r2b")
    assert s.counters()["lines_applied"] - before == int(r.applied.sum())


def test_score_plain_on_existing_sampler():
    c = xtim.Circuit(HAND)
    dets, obs = c.compile_detector_sampler(seed=708).sample(6000, separate_observables=True, bit_packed=True)
    v = unpack(obs, 3)
    rng = np.random.default_rng(708)
    f = rng.integers(0, 2, size=(6000, 3)).astype(bool)
    want = (v == f).all(axis=1).astype(np.float64)
    got = xtim.score_plain(obs, f)
    assert got.dtype == np.float64 and np.array_equal(got, want)
    assert np.array_equal(xtim.score_plain(obs, np.packbits(f, axis=1, bitorder="little"), bit_packed=True), want)
    assert np.array_equal(xtim.score_plain(obs, v), np.ones(6000))
    s = hand()                                    # on an xdem status, score_plain == score wherever dim C = 0
    _, st = s.sample(6000, seed=709)
    sp, sd = xtim.score_plain(st.obs, f), s.score(st, f)
    dim0 = np.array([st.cspaces[i].shape[0] == 0 for i in st.cspace])
    assert dim0.sum() > 1000 and np.array_equal(sp[dim0], sd[dim0])
    with pytest.raises(ValueError):
        xtim.score_plain(obs, f[:, :2].copy().repeat(5, axis=1))
    with pytest.raises(ValueError):
        xtim.score_plain(obs, f[:10])
    with pytest.raises(ValueError):
        xtim.score_plain(obs, np.zeros((6000, 2), np.uint8), bit_packed=True)


def test_clifford_only_pass_is_binary():
    c = xtim.Circuit(CLIFF)
    for text in (c.detector_error_model_text(), xdem_export(CLIFF)):
        s = xtim.XdemSampler(text)
        assert not s.extended and s.num_observables == 2 and s.num_axes == 0
        assert s.late_axes == () and s.r2b_available == (True, None) and s.invisible_residues == 0
        _, st = s.sample(20000, seed=710)
        assert len(st.cspaces) == 1 and not st.cspace.any()
        f = np.random.default_rng(710).integers(0, 2, size=(20000, 2)).astype(bool)
        sc = s.score(st, f)
        assert set(np.unique(sc)) == {0.0, 1.0}
        assert np.array_equal(sc, xtim.score_plain(st.obs, f))


@pytest.mark.parametrize("shots,batch", [(20000, 3 * 1024), (20000, 5000), (20480, 1024), (7000, 100_000)])
def test_infidelity_zero_decoder_equals_sample_score(shots, batch):
    s = d5()
    seed = 711
    dets, st = s.sample(shots, seed=seed)
    want = 1.0 - s.score(st, np.zeros((shots, 1), bool)).mean()
    seen = []

    def rec(d):
        seen.append(d.copy())
        return np.zeros((d.shape[0], 1), bool)

    r = s.infidelity(shots, rec, seed=seed, batch=batch, threads=3)
    assert isinstance(r, xtim.xdem.InfidelityResult)
    assert (r.shots, r.accepted, r.giveup_fraction) == (shots, shots, 0.0)
    assert np.array_equal(np.concatenate(seen), unpack(dets, s.num_detectors))   # identical shots, in order
    B = -(-batch // 1024) * 1024
    assert [len(x) for x in seen] == [min(B, shots - i) for i in range(0, shots, B)]
    assert abs(r.mean - want) <= 1e-12 and r.mean > 0
    p = s.score(st, np.zeros((shots, 1), bool))
    assert math.isclose(r.stderr, p.std(ddof=1) / math.sqrt(shots), rel_tol=1e-9)


@pytest.mark.parametrize("corr", ["optimal", "r2b", "none"])
def test_infidelity_lines_batch_rule(corr):
    """lines=True: each batch is sampled with fired=True from its block offset; the result is sample(fired=True)
    then score(lines=, correction=) for every batch size (decoder: the fired lines themselves)."""
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre_p1e-2")))
    shots = 6000
    dets, st = s.sample(shots, seed=712, fired=True)
    L = fired_lists(st)
    want = 1.0 - s.score(st, lines=L, correction=corr).mean()
    for batch in (1024, 2500, 100_000):
        it = iter(L)
        r = s.infidelity(shots, lambda d: [next(it) for _ in range(len(d))], seed=712, lines=True, correction=corr,
                         batch=batch)
        assert abs(r.mean - want) <= 1e-12 and r.accepted == shots
    with pytest.raises(ValueError, match="correction"):
        s.infidelity(10, lambda d: [[]] * len(d), seed=1, lines=True, correction="vdag")
    with pytest.raises(TypeError):
        s.infidelity(10, zeros_for(s), seed=1, correction="optimal")       # correction= applies to lines=True
    with pytest.raises(TypeError, match="applies to lines=True"):
        s.infidelity(10, zeros_for(s), seed=1, correction="r2b")


def test_infidelity_threads_and_batch_independent():
    s = hand()
    rs = {(t, b): s.infidelity(30000, zeros_for(s), seed=713, threads=t, batch=b)
          for t in (1, 4) for b in (1024, 7777, 100_000)}
    vals = list(rs.values())
    assert all(abs(v.mean - vals[0].mean) <= 1e-12 and v.accepted == vals[0].accepted for v in vals)
    assert s.infidelity(30000, zeros_for(s), seed=714).mean != vals[0].mean


def test_infidelity_giveup():
    s = hand()
    dets, st = s.sample(25000, seed=715)
    fired = unpack(dets, 2).any(axis=1)

    def dec(d):
        return np.zeros((d.shape[0], 3), bool), d.any(axis=1)

    r = s.infidelity(25000, dec, seed=715, batch=4096)
    assert math.isclose(r.giveup_fraction, fired.mean(), rel_tol=1e-12) and 0 < r.giveup_fraction < 1
    assert abs(r.mean - (1 - s.score(st, np.zeros((25000, 3), bool)).mean())) <= 1e-12
    with pytest.raises(ValueError):
        s.infidelity(100, lambda d: (np.zeros((d.shape[0], 3)), np.zeros(d.shape[0] + 1, bool)), seed=1)
    with pytest.raises(ValueError):
        s.infidelity(100, lambda d: np.zeros((d.shape[0], 2), bool), seed=1)
    with pytest.raises(ValueError):
        s.infidelity(100, lambda d: (np.zeros((d.shape[0], 3)), d.any(axis=1), 0), seed=1)


def test_infidelity_giveup_validated():
    s = hand()
    zeros = lambda d: np.zeros((d.shape[0], 3), bool)                       # noqa: E731
    with pytest.raises(TypeError):
        s.infidelity(2000, lambda d: (zeros(d), np.full(d.shape[0], 0.3)), seed=716)
    with pytest.raises(ValueError):
        s.infidelity(2000, lambda d: (zeros(d), np.full(d.shape[0], 2)), seed=716)
    with pytest.raises(ValueError):
        s.infidelity(2000, lambda d: (zeros(d), -np.ones(d.shape[0], np.int64)), seed=716)
    a = s.infidelity(2000, lambda d: (zeros(d), d.any(axis=1)), seed=716)
    b = s.infidelity(2000, lambda d: (zeros(d), d.any(axis=1).astype(np.int32)), seed=716)
    assert a == b and 0 < a.giveup_fraction < 1


def test_infidelity_lines_giveup():
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre_p1e-2")))
    dets, st = s.sample(5000, seed=717, fired=True)
    gv = unpack(dets, s.num_detectors).any(axis=1)
    r = s.infidelity(5000, lambda d: ([[]] * len(d), d.any(axis=1)), seed=717, lines=True, correction="none")
    assert math.isclose(r.giveup_fraction, gv.mean(), rel_tol=1e-12) and 0 < r.giveup_fraction < 1
    assert abs(r.mean - (1 - s.score(st, lines=[[]] * 5000, correction="none").mean())) <= 1e-12


def test_postselect_forms():
    """sample(postselect=): a bool mask of length D or distinct detector indices; the survivors have the law of
    sample()'s rows with the flagged detectors all 0 (survivor counts agree within 5 sigma; test_xdem_sampler.py tests
    the law exactly), and an empty selection gives sample()'s bytes."""
    s = d5()
    D, n = s.num_detectors, 20000
    mask = np.zeros(D, bool)
    mask[::3] = True
    dets, st, counts = s.sample(n, seed=718, postselect=mask)
    assert set(counts) == {"attempts", "passed", "discarded", "early"}
    assert counts["attempts"] == n and counts["passed"] + counts["discarded"] == n and len(st) == counts["passed"]
    assert not unpack(dets, D)[:, mask].any()
    full, _ = s.sample(n, seed=718)
    k = int((~unpack(full, D)[:, mask].any(axis=1)).sum())     # the same law, not the same bytes: 5 sigma
    q = k / n
    assert 0.05 < q < 0.95                                    # the mask is not degenerate (the bound is not vacuous)
    assert abs(counts["passed"] - k) <= 5 * math.sqrt(2 * n * q * (1 - q)) + 1
    di, sti, ci = s.sample(n, seed=718, postselect=np.flatnonzero(mask))
    assert di.tobytes() == dets.tobytes() and sti.obs.tobytes() == st.obs.tobytes() and ci == counts
    de, ste, ce = s.sample(n, seed=718, postselect=np.zeros(D, bool))
    assert de.tobytes() == full.tobytes() and ce["passed"] == n and ce["early"] == 0
    with pytest.raises(ValueError):
        s.sample(10, seed=1, postselect=[0, 0])
    with pytest.raises(ValueError):
        s.sample(10, seed=1, postselect=[D])
    with pytest.raises(ValueError):
        s.sample(10, seed=1, postselect=np.zeros(D - 1, bool))
    # with fired=True: the survivors' fired CSR, and score(lines=) takes the status
    df, stf, cf = s.sample(n, seed=718, postselect=mask, fired=True)
    assert df.tobytes() == dets.tobytes() and cf == counts and stf.fired[0].shape == (counts["passed"] + 1,)
    assert s.score(stf, lines=fired_lists(stf), correction="none").shape == (counts["passed"],)


def test_infidelity_postselect_mask_and_giveup():
    """infidelity(postselect=) takes a length-D bool mask as sample() does (== the indices form); give-ups are
    counted over the accepted shots only."""
    s = d5()
    D, shots = s.num_detectors, 20000
    mask = np.zeros(D, bool)
    mask[::4] = True
    dets, st, c = s.sample(shots, seed=725, postselect=mask)
    gv = unpack(dets, D).any(axis=1)

    def dec(d):
        return np.zeros((d.shape[0], 1), bool), d.any(axis=1)

    a = s.infidelity(shots, dec, seed=725, postselect=mask, batch=4096)
    b = s.infidelity(shots, dec, seed=725, postselect=np.flatnonzero(mask), batch=4096)
    assert a == b and a.accepted == c["passed"] < shots
    assert math.isclose(a.giveup_fraction, gv.mean(), rel_tol=1e-12) and 0 < a.giveup_fraction < 1
    assert abs(a.mean - (1 - s.score(st, np.zeros((len(st), 1), bool)).mean())) <= 1e-12
    assert s.infidelity(100, dec, seed=1, postselect=np.zeros(D, bool)).accepted == 100
    with pytest.raises(ValueError, match="shape"):
        s.infidelity(10, dec, seed=1, postselect=np.zeros(D - 1, bool))


def test_infidelity_postselect():
    s = d5()
    shots = 30000
    ps = list(range(0, s.num_detectors, 3))
    mask = np.zeros(s.num_detectors, bool)
    mask[ps] = True
    dets, st, _ = s.sample(shots, seed=719, postselect=mask)
    p = s.score(st, np.zeros((len(dets), 1), bool))
    seen = []

    def rec(x):
        seen.append(x.shape[0])
        return np.zeros((x.shape[0], 1), bool)

    r = s.infidelity(shots, rec, seed=719, batch=8192, postselect=ps)
    assert r.shots == shots and r.accepted == len(dets) and 0 < r.accepted < shots
    assert sum(seen) == r.accepted                                          # discarded shots are not decoded
    assert abs(r.mean - (1 - p.mean())) <= 1e-12
    full = s.infidelity(shots, rec, seed=719, postselect=list(range(s.num_detectors)))
    assert full.accepted == s.sample(shots, seed=719, postselect=np.ones(s.num_detectors, bool))[2]["passed"]
    assert s.infidelity(shots, rec, seed=719, postselect=[]).accepted == shots
    with pytest.raises(ValueError):
        s.infidelity(10, rec, seed=1, postselect=[s.num_detectors])
    with pytest.raises(ValueError):
        s.infidelity(10, rec, seed=1, postselect=[-1])


def test_infidelity_postselect_lines():
    s = xtim.XdemSampler(xdem_export(circuit("rep3T_pre_p1e-2")))
    ps = [0, 2]
    mask = np.zeros(s.num_detectors, bool)
    mask[ps] = True
    dets, st, c = s.sample(8000, seed=720, postselect=mask, fired=True)
    want = 1 - s.score(st, lines=[[]] * len(st), correction="r2b").mean()
    r = s.infidelity(8000, lambda d: [[]] * len(d), seed=720, postselect=ps, lines=True, batch=3000)   # default r2b
    assert r.accepted == c["passed"] < 8000 and abs(r.mean - want) <= 1e-12


def test_infidelity_nothing_accepted():
    s = xtim.XdemSampler("error(1) D0\n")
    r = s.infidelity(100, lambda d: np.zeros((d.shape[0], 0), bool), seed=721, postselect=[0])
    assert r.accepted == 0 and math.isnan(r.mean) and math.isnan(r.stderr)


def test_accepts_stim_detector_error_model_object():
    import stim
    text = xdem_export(CLIFF)
    s1, s2 = xtim.XdemSampler(stim.DetectorErrorModel(text)), xtim.XdemSampler(text)
    d1, st1 = s1.sample(3000, seed=722)
    d2, st2 = s2.sample(3000, seed=722)
    assert d1.tobytes() == d2.tobytes() and st1.obs.tobytes() == st2.obs.tobytes()
    for bad in (b"error(0.1) D0", 3, None):
        with pytest.raises(TypeError, match="stim.DetectorErrorModel"):
            xtim.XdemSampler(bad)


def test_cache_cap_keyword():
    text = (DATA / "cultivation_d3_faithful.xdem").read_text()
    a, b = xtim.XdemSampler(text, cache_cap=7), xtim.XdemSampler(text)
    assert a.counters()["cache_cap"] == 7 and b.counters()["cache_cap"] == 65536
    da, sa = a.sample(5000, seed=723)
    db, sb = b.sample(5000, seed=723)
    assert da.tobytes() == db.tobytes() and sa.cspace.tobytes() == sb.cspace.tobytes()
    with pytest.raises(ValueError, match="cache_cap"):
        xtim.XdemSampler(text, cache_cap=0)
    with pytest.raises(TypeError, match="cache_cap"):
        xtim.XdemSampler(text, cache_cap=1.5)
    with pytest.raises(TypeError):
        xtim.XdemSampler(text, 7)                                           # cache_cap is keyword-only


def test_file_facts_match_the_reader():
    """late_axes / r2b_available / invisible_residues equal the test API's file facts (xdem_facts, SPEC §4.4)."""
    from xtim import _xtim_testapi as T
    for name in ("cultivation_d3_faithful", "cultivation_d5", "ex422_obs", "tdag_cycle", "miniature_oracle"):
        text = (DATA / f"{name}.xdem").read_text()
        s, f = xtim.XdemSampler(text), T.xdem_facts(text)
        assert s.late_axes == tuple(f["late_axes"]), name
        assert s.r2b_available == (f["r2b_available"], None if f["r2b_available"] else f["r2b_reason"]), name
        assert s.invisible_residues == f["invisible_residues"], name
    x = xdem_export(NOTHING_BEFORE_TC)
    assert xtim.XdemSampler(x).late_axes == tuple(T.xdem_facts(x)["late_axes"]) and xtim.XdemSampler(x).late_axes


def test_canonical_report():
    assert xtim.xdem.canonical_report(circuit("rep3T_pre")) == (True, None)
    ok, why = xtim.xdem.canonical_report(xtim.Circuit(circuit("rep3T_pre")))
    assert ok is True and why is None
    with pytest.raises(TypeError):
        xtim.xdem.canonical_report(3)


def test_repr():
    s = hand()
    assert repr(s).startswith("XdemSampler(extended=True, num_detectors=2, num_observables=3, ")
    _, st = s.sample(10, seed=724)
    assert repr(st) == f"XdemStatus(shots=10, num_cspaces={st.num_cspaces})"


def test_docs_example_runs():
    """docs/xdem.md: the first python block runs, and the values its comments show are the values it prints"""
    import contextlib
    import io
    import re
    md = (Path(__file__).parent.parent / "docs" / "xdem.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", md, re.S)
    assert blocks, "docs/xdem.md has no python example"
    ns = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(blocks[0], "docs/xdem.md", "exec"), ns)
    out = buf.getvalue().splitlines()
    assert len(out) == 2, out
    assert float(out[0]) == 0.0                                        # the genie decoder: every pass is 1
    shown = re.search(r"# (0\.\d+) (0\.\d+)\.\.\.", blocks[0])
    mean, stderr = (float(x) for x in out[1].split())
    assert f"{mean:.5f}" == shown[1] and str(stderr).startswith(shown[2]), (out[1], shown.groups())
    assert ns["r"].shots == 100_000 and ns["r"].accepted == 100_000


_CAPACITY_SCRIPT = r"""
import resource, sys
import xtim
resource.setrlimit(resource.RLIMIT_AS, (1500 << 20, 1500 << 20))
for t in ["error(0.1) D2000000000\n",
          "xdem 1 detectors 2000000000 observables 0 gauge 1 hidden 0 tc 1\naxis Q0 img D0 gauge g0\nerror(0.1) D0 rot +Q0\n",
          "repeat 1000000000 {\nerror(0.1) D0\n}\n"]:
    try:
        xtim.XdemSampler(t)
        print("accepted")
    except xtim.XtimCapacityError as e:
        print("capacity", isinstance(e, MemoryError), str(e).split(" (")[0])
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS bounds the address space on Linux")
def test_huge_sizes_raise_a_capacity_error():
    """final review M8: a size stim accepts but this process cannot hold (a huge target index, header count or repeat
    count) raises xtim.XtimCapacityError (a MemoryError) naming the stage, not a bare std::bad_alloc; no new cap
    (run under a 1.5 GB address-space limit)"""
    import os
    import subprocess
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(Path(xtim.__file__).parent.parent), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", _CAPACITY_SCRIPT], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["capacity True xdem: out of memory building the sampler",
                                     "capacity True xdem: out of memory building the sampler",
                                     "capacity True xdem: out of memory reading the file at line 2"], r.stdout


def test_infidelity_lines_r2b_refused_before_sampling():
    """final review M3: with lines=True, "r2b" (the default) on a file without R2b is refused up front, with
    score(lines=)'s text, before any shot is sampled or decoded"""
    lossy = (Path(__file__).parent / "data/xdem/synth/lossy.xdem").read_text()
    s = xtim.XdemSampler(lossy.replace("tc 1", "tc 0").replace("rot", "late"))   # a late rotation on D0
    assert s.r2b_available[0] is False
    calls = []
    dec = lambda d: calls.append(1) or [[] for _ in range(len(d))]                  # noqa: E731
    for corr in (None, "r2b"):
        with pytest.raises(xtim.XtimXdemError, match="not in canonical single-stage form at t_c") as e:
            s.infidelity(5000, dec, seed=1, lines=True, correction=corr)
        assert s.r2b_available[1] in str(e.value) and 'pass correction="optimal" or "none"' in str(e.value)
    assert calls == [] and s.counters()["shots"] == 0
    dets, st = s.sample(10, seed=1, fired=True)
    with pytest.raises(xtim.XtimXdemError) as e2:                                    # the same text as score(lines=)
        s.score(st, lines=[[]] * 10)
    assert str(e2.value) == str(e.value)
    assert s.infidelity(2048, dec, seed=1, lines=True, correction="optimal").accepted == 2048


def test_one_sampler_from_several_python_threads():
    """final review M11: one XdemSampler used from 8 Python threads at once (the binding releases the GIL; every
    mutable member is under the sampler's lock) gives each call the bytes of a single-threaded run. cache_cap = 8
    forces cache and memo clears. cspace ids index the shared table in discovery order, so they are compared by
    content."""
    from concurrent.futures import ThreadPoolExecutor
    text = xdem_export(circuit("rm15rep"))
    n = 3000

    def run(s, seed):
        dets, st = s.sample(n, seed=seed, threads=2, fired=True)
        off, ids = st.fired
        H = (off, ids)                                                               # the genie hypothesis
        cs = st.cspaces
        return (dets.tobytes(), st.obs.tobytes(), [tuple(np.asarray(cs[i]).tolist()) for i in st.cspace],
                off.tobytes(), ids.tobytes(), s.score(st, np.zeros((n, s.num_observables), bool)).tobytes(),
                s.score(st, lines=H, correction="optimal").tobytes(),
                s.score(st, lines=H, correction="none").tobytes(),
                (s.score(st, lines=H).tobytes() if s.r2b_available[0] else None),
                s.correction(st, lines=H, correction="optimal").frame.tobytes())

    seeds = list(range(1, 25))
    ref = {k: run(xtim.XdemSampler(text, cache_cap=8), k) for k in seeds}
    shared = xtim.XdemSampler(text, cache_cap=8)
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = dict(zip(seeds, ex.map(lambda k: run(shared, k), seeds)))
    assert [k for k in seeds if got[k] != ref[k]] == []
