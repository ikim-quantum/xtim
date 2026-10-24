"""J-b1 P4: the FAM recording run serves its DIAGONAL shots on the (D) fast / accumulator / structural paths, which
write the layout-3 kind-0 record (fam_rec_write_d_); only PPR-class, route-1 and framed shots take the term route.

G-REC-EQ (plan P4): the layout-3 records are BITWISE the term route's — compiled once with the default route and once
with the test-API oracle option record_route="term" (every shot term-routed, the J-a route) — on every corpus circuit
and fixture: handle, kind, content, digest, P_rec, y, outcomes, route, frame_folded, and the sampled columns. The
counter shots_record_d proves the (D) paths wrote them (MUT-RET-TERM: the (D) route disabled -> red here)."""
import glob
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import _ja_fixtures as F  # noqa: E402

from _private import PRIVATE_CORPUS  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

def _fam(text, route):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0, record_route=route)


def _records(s, shots, seed, frame=None):
    d, o, dc, R = s.sample_records(shots, seed=seed, input_pauli=frame)
    rows = []
    for i in range(R.num_records):
        x, z = R.prec(i)
        rows.append((R.handle(i), R.kind(i), bytes(R.content(i)), tuple(R.digest(i)), bytes(np.asarray(x, np.uint8)),
                     bytes(np.asarray(z, np.uint8)), bytes(np.asarray(R.y(i), np.uint8)),
                     bytes(np.asarray(R.outcomes(i), np.uint8)), R.route(i), R.frame_folded(i)))
    return [np.asarray(d), np.asarray(o), np.asarray(dc)], rows


def _corpus():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "examples", "*.stim"))):
        out["ex:" + os.path.basename(f)] = open(f).read()
    for f in sorted(glob.glob(os.path.join(ROOT, "tests", "data", "*.stim"))) + PRIVATE_CORPUS:
        out["data:" + os.path.basename(f)] = open(f).read()
    out["DUP_GAUGE"] = F.DUP_GAUGE
    for k, t in sorted(F.DESIGN.items()):
        out["design:" + k] = t
    for k, t in sorted(F.CONVENTION.items()):
        out["convention:" + k] = t
    return out


CORPUS = _corpus()


def _equal(text, shots, seed=5, frame=None, expect_d=True):
    try:
        a = _fam(text, "d")
    except ValueError as e:                       # refused at construction (by name): both routes refuse alike
        with pytest.raises(ValueError, match=str(e)[:40].replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
            _fam(text, "term")
        return None
    b = _fam(text, "term")
    ca, ra = _records(a, shots, seed, frame)
    cb, rb = _records(b, shots, seed, frame)
    for x, y in zip(ca, cb):
        assert np.array_equal(x, y)
    assert len(ra) == len(rb) == shots
    for i, (p, q) in enumerate(zip(ra, rb)):
        assert p == q, (i, p, q)
    assert b.route_stats()["shots_record_d"] == 0
    nd = a.route_stats()["shots_record_d"]
    if expect_d:
        assert nd > 0
    return nd


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_g_rec_eq_corpus(name):
    text = CORPUS[name]
    shots = 300 if len(text) > 20000 else 1200
    _equal(text, shots, expect_d=False)


def test_g_rec_eq_covers_all_routes():
    """The corpus run reaches every record route: (D) kind 0 written by the (D) paths, and the term route's kind 0
    (PPR-composed), route 1 and kind 2 on the PPR fixtures (ch_cultivation, ja_seed23)."""
    routes, d_shots = set(), 0
    for name in ("ex:ch_cultivation.stim", "data:ja_seed23.stim", "ex:cultivation_d3_faithful.stim"):
        text = CORPUS[name]
        s = _fam(text, "d")
        _, rows = _records(s, 3000, 7)
        routes |= {r[8] for r in rows}
        d_shots += s.route_stats()["shots_record_d"]
        assert _equal(text, 3000, seed=7, expect_d=False) is not None
    assert d_shots > 0 and {0, 2} <= routes, (routes, d_shots)


def _k0g_and_frames():
    import test_ja_frames as TF
    import test_ja_retention as TR
    out = {k: v for k, v in sorted(TR.K0G.items())}
    out.update({k: v for k, v in sorted(TR.PPRS.items())})
    return out, TF


def test_g_rec_eq_route1_and_ppr_fixtures():
    """The route-1 (kind 0 outside the pair universe, N4-D1) and PPR fixtures of the J-a retention gates."""
    fx, _ = _k0g_and_frames()
    routes = set()
    for name, text in fx.items():
        _equal(text, 1500, seed=9, expect_d=False)
        s = _fam(text, "d")
        _, rows = _records(s, 1500, 9)
        routes |= {r[8] for r in rows}
    assert routes == {0, 1, 2}, routes


@pytest.mark.parametrize("name", ["c6", "c6_noisy", "l4", "strip", "ch_fr"])
def test_g_rec_eq_frames_fixtures(name):
    _, TF = _k0g_and_frames()
    for P in TF.FRAMES[name]:
        _equal(TF.FIXTURES[name], 800, seed=13, frame=TF.frame_rows(800, P), expect_d=False)


def test_g_rec_eq_framed_shots():
    """Framed shots stay on the term route (frame folded into P_rec); the unframed shots of the same run are (D)."""
    rng = np.random.default_rng(3)
    frame = (rng.random((800, 2)) < 0.3).astype(np.uint8)
    nd = _equal(F.PORT, 800, seed=11, frame=frame)
    assert 0 < nd < 800


def test_recording_run_on_an_unarmed_sampler_builds_no_term_view():
    """R9a / R9b: a diagonal circuit's FAM sampler is unarmed; its recording run serves every shot on (D) and builds
    no term view (the run path never arms it lazily)."""
    text = CORPUS["ex:cultivation_d3_faithful.stim"]
    s = _fam(text, "d")
    _, rows = _records(s, 2000, 3)
    st = s.route_stats()
    assert st["shots_record_d"] == 2000 and st["keys"] == 0 and st["shots_kind0"] == 0
    assert {r[8] for r in rows} == {0}


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux/glibc (LD_PRELOAD counting allocator)")
def test_zero_allocations_per_shot_recording_run_d_paths(tmp_path):
    """m9 / G2z: a warm recording run allocates per CALL, never per shot, on the (D) paths (cultivation d3 / d5,
    DUP_GAUGE: every shot (D)) and on mixed (D) + term runs (ch_cultivation, the route-1 fixture)."""
    import json
    import subprocess
    from pathlib import Path

    import test_ja_retention as TR
    from test_zero_alloc_per_shot import SHIM, _cc
    if _cc() is None:
        pytest.skip("no C compiler")
    (tmp_path / "za.c").write_text(SHIM)
    so = tmp_path / "libza.so"
    subprocess.run([_cc(), "-O2", "-fPIC", "-shared", str(tmp_path / "za.c"), "-o", str(so)], check=True)
    import xtim
    pkg_parent = str(Path(xtim.__file__).resolve().parents[1])
    wl = [["cult_d3", CORPUS["ex:cultivation_d3_faithful.stim"], 4000, 12000],
          ["cult_d5", CORPUS["ex:cultivation_d5.stim"], 2000, 6000],
          ["dup_gauge", F.DUP_GAUGE, 4000, 12000],
          ["ch_cult", CORPUS["ex:ch_cultivation.stim"], 4000, 12000],
          ["open1_k0g", TR.K0G["k0g_open1"], 4000, 12000]]
    env = dict(os.environ, LD_PRELOAD=str(so))
    r = subprocess.run([sys.executable, "-I", "-c", TR.ALLOC_CHILD, pkg_parent, str(so), json.dumps(wl)], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    for name, c in res.items():
        assert c["a1"][0] > 0, (name, c)
        assert c["a2"][0] == c["a1"][0], (name, "allocations per shot", (c["a2"][0] - c["a1"][0]) / (c["n2"] - c["n1"]), c)
        assert c["a2b"] == c["a2"], (name, c)
