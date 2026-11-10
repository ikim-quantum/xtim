"""dem-erasure-out (owner 2026-11-03): the research option `erasure=True` of the DEM text export.

Every uniform-coset Clifford record is ONE line `erasure[clifford:f<id>](p) <offset> | <gen 1> | ... | <gen v>`
(with probability p flip offset + a uniform element of span(gens)); explicit-support records keep their pattern lines.
(a) erasure=False is the default output byte for byte; (b) expanding each erasure line to its patterns at the closed
form q reproduces the exact split (clifford_cutoff=inf) record for record; (c) v > 12 (refused at the exact cutoff)
exports; (d) the CLI flag. The output is not stim-ingestible: stim must refuse it loudly.
"""
import math
import os
import re
import subprocess
import sys
from itertools import product

import pytest

import xtim
from _dem_parse import parse_error_line
from _orig_examples import load_orig
from test_dem_erasure import _q
from test_expfix_dem_law import fanout

HERE = os.path.dirname(__file__)
GOLD_DIR = os.path.join(HERE, "data", "dem_factored_law")

_HDR = re.compile(r"^# clifford f(\d+): p=(\S+) v=(\d+) (offset=([01])(?: qW=\S+)?|law=support(?: first_order=1)?) "
                  r"support=(\S*) loc=(\S+) alt=(\S+) kept=(\d+) dropped=(\S+)( \(not a coset; patterns\))?$")
_ERAS = re.compile(r"^erasure\[clifford:f(\d+)\]\((\S+)\)(.*)$")


def _sig(targets):
    return tuple(sorted(targets, key=lambda t: (t[0] != "D", int(t[1:]))))


def parse(dem_text):
    """(records {id: dict}, untagged lines, summary line). A record: p, v, offset, support, tail (the erasure-mode
    header suffix), lines {sig: q} (pattern lines), erasure (p, offset sig, [gen sigs]) or None."""
    recs, other, summary = {}, [], None
    for line in dem_text.splitlines():
        if line.startswith("# clifford faults="):
            summary = line
            continue
        m = _HDR.match(line)
        if m:
            recs[int(m.group(1))] = dict(p=float(m.group(2)), v=int(m.group(3)), support=m.group(4).startswith("law"),
                                         offset=m.group(5) == "1", supp=m.group(6), tail=m.group(11) is not None,
                                         kept=int(m.group(9)), dropped=float(m.group(10)), lines={}, erasure=None)
            continue
        assert not line.startswith("# clifford"), line
        m = _ERAS.match(line)
        if m:
            parts = m.group(3).split("|")
            off = _sig(parts[0].split())
            gens = [_sig(g.split()) for g in parts[1:]]
            r = recs[int(m.group(1))]
            assert r["erasure"] is None
            r["erasure"] = (float(m.group(2)), off, gens)
            continue
        e = parse_error_line(line)
        if e and e[2]:
            acc = set()
            for comp in e[1]:
                acc ^= set(comp)
            recs[int(e[2].split(":f")[1])]["lines"][_sig(acc)] = e[0]
        else:
            other.append(line)
    return recs, other, summary


def expand(off, gens):
    """Every element of off + span(gens), as sorted signatures (2^v of them: gens must be independent)."""
    out = []
    for bits in product((0, 1), repeat=len(gens)):
        acc = set(off)
        for b, g in zip(bits, gens):
            if b:
                acc ^= set(g)
        out.append(_sig(acc))
    assert len(set(out)) == len(out), "dependent generators"
    return out


def check_against_exact(text, cutoff=math.inf, **kw):
    """(b): the erasure-mode text expands, record for record, to the exact split's lines. Returns the records."""
    c = xtim.Circuit(text)
    era = c.detector_error_model_text(erasure=True, clifford_cutoff=cutoff, **kw)
    ex = c.detector_error_model_text(clifford_cutoff=math.inf, **kw)
    sup_ref = c.detector_error_model_text(clifford_cutoff=cutoff, **kw)     # support records keep the cutoff
    re_, oe, se = parse(era)
    rx, ox, sx = parse(ex)
    rs, _, _ = parse(sup_ref)
    assert oe == ox                                       # Pauli lines + detector / observable lines unchanged
    assert se.endswith(" mode=erasure") and len(re_) == len(rx) == int(se.split()[2].split("=")[1])
    for i, r in re_.items():
        x = rx[i]
        assert (r["p"], r["v"], r["support"], r["supp"]) == (x["p"], x["v"], x["support"], x["supp"]), i
        if r["support"]:                                  # not a coset: its pattern lines, as without erasure
            assert r["erasure"] is None and r["tail"]
            assert r["lines"] == rs[i]["lines"] and r["kept"] == rs[i]["kept"] and r["dropped"] == rs[i]["dropped"]
            continue
        assert not r["tail"] and not r["lines"] and r["erasure"] is not None, i
        p, off, gens = r["erasure"]
        assert p == r["p"] and len(gens) == r["v"] and r["kept"] == 1 and r["dropped"] == 0.0
        pats = expand(off, gens)
        assert (() in pats) == (not r["offset"]) == (off == ()), i      # pure iff the offset is in the span
        assert r["offset"] == x["offset"]
        q = _q(r["p"], r["v"], r["offset"])
        want = {s: q for s in pats if s != ()}
        assert set(want) == set(x["lines"]), (i, sorted(set(want) ^ set(x["lines"]))[:4])
        assert all(abs(x["lines"][s] - q) <= 1e-15 * q for s in want), i
    return re_


CASES = {
    "fanout5": fanout(5),
    "fanout2_ctrl": fanout(2, ctrl_read=True),
    "fanout3_obs": fanout(3).replace("DETECTOR rec[-1]\n", "OBSERVABLE_INCLUDE(0) rec[-1]\n"),
    "depol_fanout4": fanout(4, channel="DEPOLARIZE1"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_erasure_expands_to_exact_split_small(name):
    recs = check_against_exact(CASES[name])
    assert recs and all(not r["support"] for r in recs.values())


@pytest.mark.parametrize("name", ["v2_777_0010_cube.stim", "v2_777_0045_rm.stim", "v2_777_0062_cube+rm.stim",
                                  "v2_777_0089_cube+rm.stim"])
def test_erasure_expands_to_exact_split_golden_circuits(name):
    with open(os.path.join(GOLD_DIR, name)) as fh:
        text = fh.read()
    for inc in (True, False):
        assert check_against_exact(text, include_expectations=inc)


def test_erasure_cultivation_d3_cosets_and_support_records():
    text = load_orig("cultivation_d3_faithful").text
    recs = check_against_exact(text, include_expectations=False)            # 814 coset records
    assert len(recs) > 500 and all(not r["support"] for r in recs.values())
    for cut in (2, math.inf):                                                # support (list-law) records: patterns
        recs = check_against_exact(text, cutoff=cut)
        assert len(recs) > 500 and all(r["support"] for r in recs.values())


def test_erasure_ignores_cutoff_for_coset_lines():
    c = xtim.Circuit(fanout(6))
    a = c.detector_error_model_text(erasure=True, clifford_cutoff=0)
    b = c.detector_error_model_text(erasure=True, clifford_cutoff=math.inf)
    assert a.splitlines()[1:] == b.splitlines()[1:]
    assert a.splitlines()[0] == "# clifford faults=1 cutoff=0 dropped_mass=0 mode=erasure"


def test_erasure_false_is_the_default_byte_for_byte():
    texts = [fanout(5), fanout(2, ctrl_read=True), load_orig("cultivation_d3_faithful").text]
    for name in sorted(os.listdir(GOLD_DIR))[:6]:
        with open(os.path.join(GOLD_DIR, name)) as fh:
            texts.append(fh.read())
    for t in texts:
        c = xtim.Circuit(t)
        for kw in ({}, {"clifford_cutoff": math.inf}, {"include_expectations": False}):
            try:
                d = c.detector_error_model_text(**kw)
            except xtim.XtimDemError:
                continue
            assert c.detector_error_model_text(erasure=False, **kw) == d
            assert "erasure" not in d and "mode=erasure" not in d


def test_v_above_cap_refuses_exact_but_exports_in_erasure_mode():
    with pytest.raises(xtim.XtimDemError, match="exact-output cap"):
        xtim.Circuit(fanout(13)).detector_error_model_text(clifford_cutoff=math.inf)
    for n in (13, 40):
        for cut in (math.inf, 2, 7):                      # t = 7 refuses on candidates without erasure (fanout(13))
            d = xtim.Circuit(fanout(n)).detector_error_model_text(erasure=True, clifford_cutoff=cut)
            recs, _, _ = parse(d)
            (r,) = recs.values()
            assert r["v"] == n and r["erasure"] == (0.02, (), [(f"D{i}",) for i in range(n)])
    with pytest.raises(xtim.XtimDemError, match="triggers too many detectors"):
        xtim.Circuit(fanout(13)).detector_error_model_text(clifford_cutoff=7)


def test_erasure_text_is_not_stim_ingestible_and_not_on_stim_entry_points():
    stim = pytest.importorskip("stim")
    d = xtim.Circuit(fanout(3)).detector_error_model_text(erasure=True)
    with pytest.raises(Exception, match="erasure"):
        stim.DetectorErrorModel(d)
    for fn in ("detector_error_model", "detector_error_model_with_reject"):
        with pytest.raises(TypeError):
            getattr(xtim.Circuit(fanout(3)), fn)(erasure=True)
    with pytest.raises(xtim.XtimDemError, match="decompose_errors"):
        xtim.Circuit(fanout(3)).detector_error_model_text(erasure=True, decompose_errors=True)
    with pytest.raises(ValueError, match="erasure"):
        xtim.Circuit(fanout(3)).detector_error_model_text(erasure=1)


def test_cli_erasure(tmp_path):
    f = tmp_path / "c.stim"
    f.write_text(fanout(13))
    r = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--in", str(f), "--erasure",
                        "--clifford-cutoff", "inf"], capture_output=True, text=True, check=True)
    assert r.stdout == xtim.Circuit(fanout(13)).detector_error_model_text(erasure=True, clifford_cutoff=math.inf)
    assert r.stdout.splitlines()[0].endswith("mode=erasure")
    r = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--in", str(f)], capture_output=True,
                       text=True, check=True)
    assert "erasure" not in r.stdout
    h = subprocess.run([sys.executable, "-m", "xtim", "analyze_errors", "--help"], capture_output=True, text=True,
                       check=True)
    assert "--erasure" in h.stdout and "NOT ingestible" in h.stdout
