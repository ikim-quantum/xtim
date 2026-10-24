"""J-b1 P11b: the record API's export builders (cpp/bindings/record_api.hpp; design §4), bound in the test API.

Gates (plan §2.1 P11b):
  G-frames  the state the frames() export defines — P_rec · C0 |B> projected onto the records at y, C0 the plan's
            term rotations — evaluated by xtim.frames (record_state / record_expectation) equals the engine's
            materialize on every sampled row: kind 0 and kind 2 plans, dressed unread generators, ANTI / gauge records,
            framed rows (the frames corpus), and the r5check (2) / (2b) and r6check circuits;
  G-signs   signs() (now through the shared builder) never disagrees with the export's state on a definite value;
  schema    a / cz present iff every plan of the export is kind 0; σ-space fields never exported; plan_index is
            export-local (first appearance) and plan_key_digest the record's content digest; record_layout matches;
            the CSR shape (term_index / term_offsets into a shared axis_table deduplicated by (x, z, phase); Born rows
            and kind-0 CZ pairs CSR; P11b's `terms` / `term_axes` absent); a kind-0 plan's repeated −Z_q is a repeated
            index (MUT-TERMS-SET -> red); +Z_q and −Z_q are distinct entries (MUT-AXTAB-NOPHASE -> red); a route-1 row
            is exported; frames() on resident kind-0 plans rebuilds no reader table (MUT-FR-NO-CACHE-REC -> red);
            the layout-3 key; record_report's classes equal record_classes() and its route counts sum to the shots;
  frame     frame_folded = 1 exactly on the rows of an input_pauli run (the frame is folded into P_rec, and the
            export's state carries it).
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from test_ja_frames import FIXTURES as FR_FIX, frame_rows  # noqa: E402
from test_ja_retention import DIAG, PPRS  # noqa: E402
from test_jb1_c0hat import MIRROR  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402
from xtim import frames as XF  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
NMAX = 22
R5_2 = "RX 0\nR 1\nX_ERROR(0.2) 1\nM 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
R5_2B = "RX 0\nT 0\nX_ERROR(0.5) 0\nT_DAG 0\nR 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\n"
R6 = "R 0\nX_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\n"
# kind 0 (the T·X·T_DAG dressing: −Z_0) and kind 2 (CH terms with the T term +Z_0 in S) in one export
MIXED_PM_Z = ("RX 0\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nR 1\nH 2\nX_ERROR(0.3) 1\nCH 2 1\nX_ERROR(0.3) 1\nCH 2 1\n"
              "MX 0\nM 1\nMX 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")
CORPUS = {"r5_2": R5_2, "r5_2b": R5_2B, "r6": R6, "mirror": MIRROR}
CORPUS.update({"diag_" + k: v for k, v in DIAG.items()})
CORPUS.update({"ppr_" + k: v for k, v in PPRS.items()})


def records(text, shots=120, seed=5, P=None):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = TA.compile_family_sampler(text, selfcheck=0)
            d, o, dc, R = s.sample_records(shots, seed=seed, input_pauli=None if P is None else frame_rows(shots, P))
        return s, R
    except (ValueError, RuntimeError):
        return None, None


def probes(n, rng, ops, k=12):
    out = [tuple(o) for o in ops]
    for _ in range(k):
        x = (rng.random(n) < 0.2).astype(np.uint8)
        z = (rng.random(n) < 0.2).astype(np.uint8)
        out.append((x, z, int(np.dot(x, z)) & 3))
    return out


def check_rows(s, R, rows, rng):
    fr = s.frames(R)
    bf = TA.bare_frames(s)
    n = int(bf["n"])
    ops = [tuple(o) for o in TA.record_operators(s)]
    P = probes(n, rng, ops)
    worst = 0.0
    for i in rows:
        a = XF.record_state(bf, fr, i)
        b = XF._state_vec(s.materialize(R, i))
        worst = max(worst, abs(abs(np.vdot(a, b)) - 1.0))
        e = s.state_expectations(R, i, P)
        for k, (x, z, ph) in enumerate(P):
            v = np.vdot(a, XF._pauli_apply(a, n, x, z, ph))
            worst = max(worst, abs(v.real - e[k]))
    return fr, worst


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_gframes_export_state_equals_materialize(name):
    s, R = records(CORPUS[name])
    if s is None:
        pytest.skip("refused by name")
    if int(TA.bare_frames(s)["n"]) > NMAX:
        pytest.skip("n > %d (dense oracle)" % NMAX)
    n = int(TA.bare_frames(s)["n"])
    fr, worst = check_rows(s, R, range(0, R.num_records, 7 if n <= 16 else 40), np.random.default_rng(3))
    assert worst < 1e-10, worst


@pytest.mark.parametrize("name", sorted(FR_FIX))
@pytest.mark.parametrize("P", ["X", "Z"])
def test_gframes_framed_rows(name, P):
    s, R = records(FR_FIX[name], P=P)
    if s is None:
        pytest.skip("refused by name")
    if int(TA.bare_frames(s)["n"]) > NMAX:
        pytest.skip("n > %d" % NMAX)
    n = int(TA.bare_frames(s)["n"])
    fr, worst = check_rows(s, R, range(0, R.num_records, 11 if n <= 16 else 40), np.random.default_rng(4))
    assert worst < 1e-10, worst
    assert np.all(np.asarray(fr["frame_folded"]) == 1)          # input_pauli route: the frame folded into P_rec
    s0, R0 = records(FR_FIX[name])
    assert np.all(np.asarray(s0.frames(R0)["frame_folded"]) == 0)


@pytest.mark.parametrize("name", ["r5_2", "r5_2b", "r6", "ppr_ch_rec", "diag_port"])
def test_gsigns_agree_with_the_export(name):
    s, R = records(CORPUS[name], shots=200)
    fr = s.frames(R)
    bf = TA.bare_frames(s)
    n = int(bf["n"])
    P = probes(n, np.random.default_rng(9), [tuple(o) for o in TA.record_operators(s)], k=16)
    got = np.asarray(s.signs(R, P))
    for i in range(0, R.num_records, 5):
        a = XF.record_state(bf, fr, i)
        for k, (x, z, ph) in enumerate(P):
            if got[i, k]:
                v = np.vdot(a, XF._pauli_apply(a, n, x, z, ph)).real
                assert abs(v - got[i, k]) < 1e-9, (i, k, v, got[i, k])


def test_schema_kind0_view_only_when_all_kind0():
    s, R = records(CORPUS["ppr_ch_rec"])
    fr = s.frames(R)
    kinds = set(np.asarray(fr["kind"]).tolist())
    assert 2 in kinds and "a" not in fr and "cz_pairs" not in fr
    s0, R0 = records(CORPUS["mirror"])
    fr0 = s0.frames(R0)
    assert set(np.asarray(fr0["kind"]).tolist()) == {0} and "a" in fr0 and "cz_pairs" in fr0
    for removed in ("r", "kappa", "kernel_xz", "kernel_phase", "fallback", "plan_id", "exact", "terms", "term_axes",
                    "cz"):
        assert removed not in fr and removed not in fr0


def test_schema_plan_index_digest_layout():
    s, R = records(CORPUS["ppr_ch_rec"], shots=300)
    fr = s.frames(R)
    pi = np.asarray(fr["plan_index"])
    seen = []
    for v in pi.tolist():
        if v not in seen:
            seen.append(v)
    assert seen == list(range(len(seen))) == list(range(len(fr["kind"])))   # export-local, first appearance
    for i in range(0, R.num_records, 13):
        content, digest = R.plan_content(i)
        assert bytes(fr["plan_key_digest"][pi[i]]) == digest
        ps = s.record_plan_structure(R, i)
        assert bytes(ps["plan_key_digest"][0]) == digest and ps["plan_content"] == content
        assert ps["rho"][0] == fr["rho"][pi[i]] and ps["kappa_rec"][0] == fr["kappa_rec"][pi[i]]
        # the same C0 multiset, read through each export's own axis table
        def axes(d, k):
            ax, az, ph = d["axis_table"]
            off = d["term_offsets"]
            return sorted((bytes(ax[t]), bytes(az[t]), int(ph[t])) for t in d["term_index"][off[k]:off[k + 1]])
        assert axes(ps, 0) == axes(fr, pi[i])
        assert len(R.outcomes(i)) == fr["n_outcomes"][i] == ps["rho"][0] + ps["kappa_rec"][0]
        lay = fr["record_layout"]
        assert len(R.key(i)) == len(content) + 8 * (lay["prefix_words"] + lay["y_words"])
        assert np.array_equal(fr["y"][i], R.y(i))
    assert s.record_layout() == fr["record_layout"]


def test_record_report_schema():
    s, R = records(CORPUS["ppr_ch_rec"], shots=500)
    rr = s.record_report()
    old = s.record_classes()
    assert rr["classes"] == old["classes"]
    assert rr["nondeterministic_detectors"] == old["nondeterministic_detectors"]
    assert rr["nondeterministic_observables"] == old["nondeterministic_observables"]
    assert sum(rr["route_counts"].values()) == 500
    assert {"plan_misses", "payload_builds", "header_bytes", "trie_walks"} <= set(rr["counters"])


def test_require_kappa_rec_zero():
    s, R = records(CORPUS["mirror"])
    fr = s.frames(R)
    if np.any(np.asarray(fr["kappa_rec"]) > 0):
        with pytest.raises(ValueError, match="kappa_rec > 0"):
            XF.require_kappa_rec_zero(fr)
    else:
        XF.require_kappa_rec_zero(fr)
    fr2 = dict(fr)
    fr2["kappa_rec"] = np.ones_like(np.asarray(fr["kappa_rec"]))
    with pytest.raises(ValueError, match="kappa_rec > 0"):
        XF.require_kappa_rec_zero(fr2)


def _axes_of(fr, k):
    ax, az, ph = fr["axis_table"]
    off = fr["term_offsets"]
    return [int(t) for t in fr["term_index"][off[k]:off[k + 1]]]


def test_kind0_multiset_has_repeats():
    """cult_d5's kind-0 plans: −Z_q appears for a_q and for each CZ on q, so some plan repeats an index."""
    s, R = records(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "tests", "data", "cultivation_d5.stim")).read(), shots=400)
    fr = s.frames(R)
    reps = 0
    for k in range(len(fr["kind"])):
        t = _axes_of(fr, k)
        reps += len(t) - len(set(t))
    assert reps > 0


def test_mixed_kinds_keep_plus_and_minus_z_apart():
    """An export mixing kind 0 (−Z_q axes) and kind 2 (whose term list holds +Z_q): the table keeps +Z_q and −Z_q as
    two entries, and the export's state equals materialize (G-frames on the mixed export)."""
    s, R = records(MIXED_PM_Z, shots=3000)
    fr = s.frames(R)
    assert {0, 2} <= set(np.asarray(fr["kind"]).tolist())
    ax, az, ph = fr["axis_table"]
    keys = {}
    for t in range(len(ph)):
        keys.setdefault((bytes(ax[t]), bytes(az[t])), set()).add(int(ph[t]) % 4)
    assert any(v >= {0, 2} for v in keys.values()), "no +Z / -Z pair in the export"
    fr2, worst = check_rows(s, R, range(0, R.num_records, 9), np.random.default_rng(5))
    assert worst < 1e-10, worst


def test_route1_row_exported():
    from test_ja_ppr_records import K0G_OPEN, biased_fuzz_records
    s, R = records(biased_fuzz_records(K0G_OPEN[0], True), shots=4000)
    assert s.route_stats()["plans_kind0_general"] > 0
    route = [R.route(i) for i in range(R.num_records)]
    assert 1 in route, "no route-1 row recorded"
    if int(TA.bare_frames(s)["n"]) <= NMAX:
        rows = [i for i in range(R.num_records) if route[i] == 1][:5]
        fr, worst = check_rows(s, R, rows, np.random.default_rng(6))
        assert worst < 1e-10, worst


def test_frames_reads_the_resident_kind0_tables():
    """F-ruling COLD leg: frames() on kind-0 plans still resident in the diagonal cache reads the cache's record table;
    it rebuilds no shot law (reader_table_rebuilds stays 0). MUT-FR-NO-CACHE-REC -> red here."""
    s, R = records(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "tests", "data", "cultivation_d3_faithful.stim")).read(), shots=400)
    before = s.route_stats()["reader_table_rebuilds"]
    fr = s.frames(R)
    assert 0 in set(np.asarray(fr["kind"]).tolist())
    assert s.route_stats()["reader_table_rebuilds"] == before == 0
