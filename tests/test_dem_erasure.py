"""dem-erasure (design note signed 2026-10-30): Clifford faults as unheralded erasures in the DEM.

T4 (test-of-test): every DEM parser the suite uses must see the Clifford records' tagged lines
`error[clifford:f<id>](q) ...` and XOR-compose duplicate signatures. A parser matching only `error(`
silently skips them; test_parsers_see_tagged_clifford_lines is the named RED test for that mutation.
"""
import exact_sv as E
from _dem_compare import _parse as cmp_parse
from _dem_parse import parse_error_line, xor_compose
from test_expfix_dem_law import _mechs as expfix_mechs

TAGGED = """# clifford faults=2 cutoff=2 dropped_mass=0
error(0.01) D0
error(0.02) D1 L0
# clifford f0: p=0.02 v=1 offset=0 support=D0 loc=X_ERROR@0 alt=X1 kept=1 dropped=0
error[clifford:f0](0.01) D0
# clifford f1: p=0.03 v=2 offset=1 qW=-1e-5 support=D2,D3 loc=X_ERROR@1 alt=X2 kept=2 dropped=0
error[clifford:f1](0.004) D2 D3
error[clifford:f1](0.005) D2 ^ D3
detector D4
"""


def test_parse_error_line_reads_tags():
    assert parse_error_line("error(0.5) D0 L1") == (0.5, [["D0", "L1"]], None)
    assert parse_error_line("error[clifford:f7](0.25) D1 ^ D2") == (0.25, [["D1"], ["D2"]], "clifford:f7")
    assert parse_error_line("# clifford f0: p=0.1") is None
    assert parse_error_line("detector D3") is None


def test_parsers_see_tagged_clifford_lines():
    d0 = xor_compose(0.01, 0.01)                  # the Pauli D0 line and record f0's D0 line compose
    d23 = xor_compose(0.004, 0.005)               # f1's two lines on {D2, D3} (one decomposed) compose
    # exact_sv.dem_mechs (the exact-oracle comparator)
    m = E.dem_mechs(TAGGED, 5)
    assert m == {frozenset({0}): d0, frozenset({1, 5}): 0.02, frozenset({2, 3}): d23}
    # tests/_dem_compare (the golden comparator)
    other, mech = cmp_parse(TAGGED)
    assert {k: p for k, (p, _) in mech.items()} == {("D0",): d0, ("D1", "L0"): 0.02, ("D2", "D3"): d23}
    assert "detector D4" in other
    # test_expfix_dem_law._mechs
    assert expfix_mechs(TAGGED) == {("D0",): d0, ("D1", "L0"): 0.02, ("D2", "D3"): d23}


# ── T5: the default cutoff t = 2 against the exact split (clifford_cutoff = inf) ─────────────────────────────────
import math
import re
import subprocess
import sys

import numpy as np
import pytest

import xtim
from _orig_examples import load_orig
from test_expfix_dem_law import census, fanout

_HDR = re.compile(r"^# clifford f(\d+): p=(\S+) v=(\d+) (offset=([01])(?: qW=\S+)?|law=support(?: first_order=1)?) "
                  r"support=(\S*) loc=(\S+) alt=(\S+) kept=(\d+) dropped=(\S+)$")
_SUM = re.compile(r"^# clifford faults=(\d+) cutoff=(\S+) dropped_mass=(\S+)$")


def records(dem_text):
    """{id: dict(p, v, offset, kept, dropped, lines={sig tuple: q})} and the summary (faults, cutoff, mass)."""
    recs, summary = {}, None
    for line in dem_text.splitlines():
        m = _SUM.match(line)
        if m:
            summary = (int(m.group(1)), m.group(2), float(m.group(3)))
            continue
        m = _HDR.match(line)
        if m:
            recs[int(m.group(1))] = dict(p=float(m.group(2)), v=int(m.group(3)), support=m.group(4).startswith("law"),
                                         offset=m.group(5) == "1", kept=int(m.group(9)),
                                         dropped=float(m.group(10)), lines={})
            continue
        e = parse_error_line(line)
        if e and e[2]:
            rid = int(e[2].split(":f")[1])
            acc = set()
            for comp in e[1]:                          # a decomposed line: its components XOR to the pattern
                acc ^= set(comp)
            sig = tuple(sorted(acc, key=lambda t: (t[0] != "D", int(t[1:]))))
            assert sig not in recs[rid]["lines"], ("one pattern twice in a record", rid, sig)
            recs[rid]["lines"][sig] = e[0]
    return recs, summary


def _dw(sig):
    return sum(1 for t in sig if t[0] == "D")


def _q(p, v, offset):
    if offset:
        return -math.expm1(math.log1p(-2 * p) * 2.0 ** -v) / 2
    return -math.expm1(math.log1p(-p) * 2.0 ** (1 - v)) / 2


def _check_cutoff_against_exact(text, t, **kw):
    c = xtim.Circuit(text)
    cut = c.detector_error_model_text(clifford_cutoff=t, **kw)
    ex = c.detector_error_model_text(clifford_cutoff=math.inf, **kw)
    rc, sc = records(cut)
    re_, se = records(ex)
    assert sc[0] == se[0] == len(rc) == len(re_) and sc[1] == str(t) and se[1] == "inf" and se[2] == 0.0
    total_dropped = 0.0
    n_obs_only = 0
    for i, r in rc.items():
        e = re_[i]
        assert (r["p"], r["v"], r["offset"], r["support"]) == (e["p"], e["v"], e["offset"], e["support"])
        assert e["dropped"] == 0.0 and e["kept"] == len(e["lines"])
        # kept lines are exactly the exact split's patterns of detector weight <= t, at the same q
        assert r["lines"] == {s: q for s, q in e["lines"].items() if _dw(s) <= t}, i
        assert r["kept"] == len(r["lines"])
        n_obs_only += sum(1 for s in r["lines"] if _dw(s) == 0)
        if not r["support"]:
            q = _q(r["p"], r["v"], r["offset"])
            assert all(abs(x - q) <= 1e-15 * q for x in r["lines"].values()), i
            n_total = 2 ** r["v"] - (0 if r["offset"] else 1)
            assert len(e["lines"]) == n_total
            want = r["p"] * (n_total - r["kept"]) / 2 ** r["v"]
            assert abs(r["dropped"] - want) <= 1e-15 * max(want, 1e-300), (i, r["dropped"], want)
        total_dropped += r["dropped"]
    assert abs(sc[2] - total_dropped) <= 1e-12 * max(total_dropped, 1e-300)
    # the Pauli (untagged) lines are the same at every cutoff
    untagged = lambda d: [l for l in d.splitlines() if l.startswith("error(")]
    assert untagged(cut) == untagged(ex)
    return rc, n_obs_only, sc


def test_fanout5_cutoff2_closed_form():
    p = 0.02
    rc, _, sc = _check_cutoff_against_exact(fanout(5, p), 2)
    (r,) = rc.values()
    assert r["v"] == 5 and not r["offset"] and r["kept"] == 5 + 10
    assert abs(sc[2] - p * (31 - 15) / 32) < 1e-18
    # the default is t = 2 (owner 2026-11-02, dem-t2); the binding / DemWithReject carry the same numbers
    c = xtim.Circuit(fanout(5, p))
    assert c.detector_error_model_text() == c.detector_error_model_text(clifford_cutoff=2)
    assert "cutoff=2 dropped_mass=" in c.detector_error_model_text()
    w = c.detector_error_model_with_reject()
    assert w.clifford_faults == 1 and w.clifford_dropped_mass == sc[2]
    assert w.dem.num_errors == 15                     # stim keeps the tagged lines (and drops the comments)


def test_cultivation_d3_default_cutoff_against_exact():
    text = load_orig("cultivation_d3_faithful").text
    for kw in ({}, {"include_expectations": False}):
        rc, n_obs_only, sc = _check_cutoff_against_exact(text, 2, **kw)
        assert sc[0] > 500 and sc[2] > 0, sc


def test_observable_only_patterns_always_kept():
    # fanout(3) with the third read an OBSERVABLE: weight counts detectors only; t = 0 keeps exactly {L0}
    t = fanout(3).replace("DETECTOR rec[-1]\n", "OBSERVABLE_INCLUDE(0) rec[-1]\n")
    for cut in (0, 1, 2):
        rc, n_obs_only, _ = _check_cutoff_against_exact(t, cut)
        (r,) = rc.values()
        assert ("L0",) in r["lines"] and n_obs_only == 1
        assert all(_dw(s) <= cut for s in r["lines"])
    rc, _, _ = _check_cutoff_against_exact(t, 0)
    assert list(next(iter(rc.values()))["lines"]) == [("L0",)]


def test_cutoff_at_least_max_weight_equals_exact():
    """t >= the largest detector weight of any pattern == t = inf, byte for byte (but the summary's cutoff). NOTE: that
    weight can exceed v: an offset record's deterministic base detectors count (fanout(2) + a control read: v = 2,
    pattern D0 D1 D2 of weight 3), so t >= v alone does NOT give the exact split (it drops mass)."""
    off = fanout(2, ctrl_read=True)
    for text in (fanout(5), off, load_orig("cultivation_d3_faithful").text):
        c = xtim.Circuit(text)
        ex_text = c.detector_error_model_text(clifford_cutoff=math.inf)
        ex = ex_text.splitlines()
        wmax = max(_dw(s) for r in records(ex_text)[0].values() for s in r["lines"])
        for t in (wmax, wmax + 1, 40):
            got = c.detector_error_model_text(clifford_cutoff=t).splitlines()
            assert got[0] == ex[0].replace("cutoff=inf", f"cutoff={t}")
            assert got[1:] == ex[1:]
        if wmax > 0:
            assert records(c.detector_error_model_text(clifford_cutoff=wmax - 1))[1][2] > 0
    rc, _ = records(xtim.Circuit(off).detector_error_model_text(clifford_cutoff=2))      # t = v = 2, offset
    (r,) = rc.values()
    assert r["v"] == 2 and r["offset"] and r["dropped"] > 0


def test_decompose_errors_records():
    c = xtim.Circuit(load_orig("cultivation_d3_faithful").text)
    import stim
    for t in (2, math.inf):
        d = c.detector_error_model_text(clifford_cutoff=t, decompose_errors=True)
        plain = c.detector_error_model_text(clifford_cutoff=t)
        assert stim.DetectorErrorModel(d).num_errors == stim.DetectorErrorModel(plain).num_errors
        rd, _ = records(d)
        rp, _ = records(plain)
        assert {i: r["lines"] for i, r in rd.items()} == {i: r["lines"] for i, r in rp.items()}   # components XOR back
        for line in d.splitlines():
            e = parse_error_line(line)
            if e and e[2]:
                assert all(sum(1 for x in comp if x[0] == "D") <= 2 for comp in e[1]), line
    # t = 2 record lines are graphlike already: decomposition leaves them untouched
    assert "^" not in "".join(l for l in c.detector_error_model_text(decompose_errors=True, clifford_cutoff=2).splitlines()
                              if l.startswith("error[clifford"))


@pytest.mark.parametrize("bad", [-1, -2, 2.5, "2", True, float("nan"), -math.inf])
def test_clifford_cutoff_argument_validation(bad):
    with pytest.raises(ValueError, match="clifford_cutoff"):
        xtim.Circuit(fanout(2)).detector_error_model_text(clifford_cutoff=bad)


def test_clifford_cutoff_accepted_values():
    c = xtim.Circuit(fanout(4))
    assert c.detector_error_model_text(clifford_cutoff=np.int64(3)) == c.detector_error_model_text(clifford_cutoff=3)
    assert "cutoff=inf" in c.detector_error_model_text(clifford_cutoff=math.inf)
    assert "cutoff=inf" in c.detector_error_model_text(clifford_cutoff=None)
    with pytest.raises(ValueError, match="clifford_cutoff"):
        c.detector_error_model_text(clifford_cutoff=2 ** 40)          # outside the binding's int: ValueError
    assert "cutoff=2147483647" in c.detector_error_model_text(clifford_cutoff=2 ** 31 - 1)
    with pytest.raises(ValueError, match="clifford_cutoff"):
        xtim._xtim.export_dem_text(fanout(2), True, False, False, False, [], [], 0.0, -2)   # twirl-routes T2: no pe_max_work


def test_pauli_only_circuit_has_no_clifford_lines():
    text = "R 0 1\nDEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
    c = xtim.Circuit(text)
    d = c.detector_error_model_text()
    assert "#" not in d and "[clifford" not in d
    assert d == c.detector_error_model_text(clifford_cutoff=math.inf) == c.detector_error_model_text(clifford_cutoff=0)
    w = c.detector_error_model_with_reject()
    assert w.clifford_faults == 0 and w.clifford_dropped_mass == 0.0


def test_finite_cutoff_lifts_the_v_cap_with_a_candidate_budget():
    # fanout(13): v = 13 refuses at inf (cap 12); at t = 2: 91 lines. A 2^12 candidate budget refuses by name with the
    # cutoff in the text: fanout(13) at t = 7 has 1 + 13 + 78 + 286 + 715 + 1287 + 1716 + 1716 = 5812 > 4096.
    r = census(fanout(13), clifford_cutoff=2)
    assert r["ok"] and r["clifford_faults"] == 1
    r = census(fanout(13), clifford_cutoff=7)
    assert not r["ok"] and "triggers too many detectors" in r["error"]
    assert "(cutoff t = 7, 5812 candidate patterns)" in r["error"], r["error"]
    r = census(fanout(13), clifford_cutoff=6)                         # 4096 candidates: exactly the budget
    assert r["ok"] and r["instances"][0]["kept"] == 4096 - 1


def test_cli_clifford_cutoff(tmp_path):
    f = tmp_path / "c.stim"
    f.write_text(fanout(3))
    out = {}
    for arg in ([], ["--clifford-cutoff", "inf"], ["--clifford-cutoff", "1"]):
        r = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--in", str(f)] + arg,
                           capture_output=True, text=True, check=True)
        out[tuple(arg)] = r.stdout
    assert out[()].splitlines()[0].startswith("# clifford faults=1 cutoff=2 dropped_mass=")   # default: t = 2 (dem-t2)
    assert out[("--clifford-cutoff", "inf")].splitlines()[0].endswith("cutoff=inf dropped_mass=0")   # exact on request
    assert out[()] != out[("--clifford-cutoff", "inf")] and "cutoff=1" in out[("--clifford-cutoff", "1")]
    assert out[()] == xtim.Circuit(fanout(3)).detector_error_model_text(clifford_cutoff=2)
    r = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--in", str(f), "--clifford-cutoff", "foo"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "usage:" in r.stderr and "--clifford-cutoff" in r.stderr     # the CLI usage error (exit 1)


def test_exact_cap_refusal_states_reason_and_cutoff_options(tmp_path):
    """Owner (fix round 1; dem-t2): a refusal of an EXPLICIT exact request caused by a Clifford fault's size names the
    exact request as the cause and shows the finite-cutoff options — in the binding error, the Python exception and the
    CLI error."""
    want = ("over v = 13 signature directions", "needs 2^13 - 1 = 8191 mechanisms", "exact-output cap (v <= 12)",
            "Cause: the EXACT expansion was requested (clifford_cutoff=inf)",
            "the default clifford_cutoff=2 (matching-friendly) or any finite clifford_cutoff=t", "dropped Clifford mass",
            "error-suppression order")
    r = census(fanout(13), clifford_cutoff=-1)
    assert not r["ok"] and all(w in r["error"] for w in want), r["error"]
    for exact in (math.inf, None):
        with pytest.raises(xtim.XtimDemError) as ei:
            xtim.Circuit(fanout(13)).detector_error_model(clifford_cutoff=exact)
        assert all(w in str(ei.value) for w in want), str(ei.value)
    f = tmp_path / "c.stim"
    f.write_text(fanout(13))
    p = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--in", str(f), "--clifford-cutoff", "inf"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and all(w in p.stderr for w in want), p.stderr
    # and the default (t = 2) exports it, in Python and the CLI
    assert "cutoff=2" in xtim.Circuit(fanout(13)).detector_error_model_text()
    p = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--in", str(f)], capture_output=True, text=True,
                       check=True)
    assert p.stdout.startswith("# clifford faults=1 cutoff=2 ")


def test_default_clifford_cutoff_is_2():
    """dem-t2 (owner 2026-11-02): every default is t = 2 — the Python signatures, the binding, the census test API."""
    import inspect
    from xtim.circuit import CLIFFORD_CUTOFF_DEFAULT
    assert CLIFFORD_CUTOFF_DEFAULT == 2
    for fn in ("detector_error_model_text", "detector_error_model", "detector_error_model_with_reject"):
        assert inspect.signature(getattr(xtim.Circuit, fn)).parameters["clifford_cutoff"].default == 2, fn
    c = xtim.Circuit(fanout(5))
    d = c.detector_error_model_text()
    assert d.splitlines()[0].startswith("# clifford faults=1 cutoff=2 dropped_mass=")
    assert d == c.detector_error_model_text(clifford_cutoff=2) != c.detector_error_model_text(clifford_cutoff=math.inf)
    from xtim._exact_scope import explicit_exact
    with explicit_exact():                                                       # J-b1 P13: an explicit DEM entry
        assert xtim._xtim.export_dem_text(fanout(5))["dem"] == d                 # the binding default
    assert census(fanout(5))["instances"][0]["kept"] == 5 + 10                   # the census default: t = 2
    w = c.detector_error_model_with_reject()
    assert w.clifford_dropped_mass > 0 and w.dem == c.detector_error_model()


def test_exact_is_still_reachable():
    """dem-t2: exact stays one argument away (math.inf or None): every pattern, nothing dropped."""
    c = xtim.Circuit(fanout(5))
    for exact in (math.inf, None):
        d = c.detector_error_model_text(clifford_cutoff=exact)
        assert d.splitlines()[0] == "# clifford faults=1 cutoff=inf dropped_mass=0"
        assert sum(l.startswith("error[clifford:") for l in d.splitlines()) == 2 ** 5 - 1
        assert c.detector_error_model_with_reject(clifford_cutoff=exact).clifford_dropped_mass == 0.0
