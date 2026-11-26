"""xdem reader (SPEC §3): grammar, canonical spelling, §3.5 refusals, conventional-DEM semantics (§3.3), the composite
(§1.2) and the law (§1.3) against the model oracle, the canonical emitter's fixed point, the third-party texts."""
import random
from pathlib import Path

import pytest

from xtim import _xtim_testapi as ta
from xtim.errors import XtimXdemError

from _xdem_model import XdemModel, affine, cancel, reduce, rref
from _xdem_synth import random_text

DATA = Path(__file__).parent / "data" / "xdem"
EX = (DATA / "ex422_obs.xdem").read_text()


def laws_equal(text, cfgs):
    M = XdemModel.from_text(text)
    got = ta.xdem_laws(text, cfgs)
    for ids, (off_hex, coins_hex) in zip(cfgs, got):
        C = rref([int(c, 16) for c in coins_hex])
        o2, C2 = affine(M.law(M.circuit(ids)), M.n)
        assert C == C2 and reduce(int(off_hex, 16), C) == o2, (ids,)


def test_worked_file_parses_and_its_laws_equal_the_oracle():
    d = ta.xdem_parse(EX)
    assert (d["extended"], d["D"], d["O"], d["r"], d["m"], d["tc"], d["A"], d["L"]) == (True, 3, 2, 3, 2, 3, 8, 3)
    laws_equal(EX, [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])


@pytest.mark.parametrize("seed", range(200))
def test_random_valid_files(seed):
    rng = random.Random(seed)
    text = random_text(rng, D=rng.randint(1, 6), O=rng.randint(0, 3), r=rng.randint(0, 4), m=rng.randint(0, 3),
                       A=rng.randint(1, 6), L=rng.randint(1, 8))
    assert ta.xdem_emit(text) == text                                  # canonical text is a fixed point
    L = ta.xdem_parse(text)["L"]
    cfgs = [sorted(rng.sample(range(L), rng.randint(0, L))) for _ in range(8)] + [[0, 0], [L - 1, 0, L - 1]]
    laws_equal(text, cfgs)
    M = XdemModel.from_text(text)
    for ids in cfgs:
        f, tau, eta = M.compose(ids)
        fh, tl, el = ta.xdem_compose(text, ids)
        assert int(fh, 16) == f and sum(1 << a for a in tl) == tau and dict(el) == eta


def test_hypothesis_order_and_repeats_in_compose():                   # Review Focus 2 (reader half)
    assert ta.xdem_compose(EX, [2, 0, 2]) == ta.xdem_compose(EX, [0])
    assert ta.xdem_compose(EX, [1, 1]) == ta.xdem_compose(EX, [])


def regauge(text):
    """kappa -> G kappa with g0' = g0 + g1 (an invertible change of the gauge basis) plus one unused gauge qubit: a
    different text of the same model (checked by the plan's author with the oracle on all 7 subsets of ex422)"""
    out = []
    for l in text.splitlines():
        tok = l.split()
        if tok and tok[0] == "xdem":
            i = tok.index("gauge") + 1
            tok[i] = str(int(tok[i]) + 1)
        elif tok and tok[0] == "axis" and "gauge" in tok:
            i = tok.index("gauge")
            j = i + 1
            while j < len(tok) and tok[j].startswith("g"):
                j += 1
            g = {int(t[1:]) for t in tok[i + 1:j]}
            if 1 in g:
                g ^= {0}
            tok = tok[:i] + (["gauge"] + [f"g{t}" for t in sorted(g)] if g else []) + tok[j:]
        out.append(" ".join(tok))
    return "\n".join(out) + "\n"


def test_third_party_texts():                                          # Review Focus 3
    cfgs = [[0], [1], [0, 1], [0, 2], [0, 1, 2]]
    for t in (EX.replace("\n", "\r\n"), EX.rstrip("\n")):              # CRLF; no final newline
        assert ta.xdem_laws(t, cfgs) == ta.xdem_laws(EX, cfgs)
    R = regauge(EX)                                                    # non-canonical gauge basis + unused g3
    assert "gauge 4" in R and "axis Q1 img D1 D2 gauge g0 g1" in R
    laws_equal(R, cfgs)
    M = XdemModel.from_text(EX)
    for (o1, c1), (o2, c2) in zip(ta.xdem_laws(R, cfgs), ta.xdem_laws(EX, cfgs)):
        C = rref([int(c, 16) for c in c1])
        assert C == rref([int(c, 16) for c in c2]) and reduce(int(o1, 16), C) == reduce(int(o2, 16), C)
    # the X2 line's chi needs a gauge qubit (chi = X[g2 h0], SPEC §3.7): accepted, it is an operator
    assert M.chi(M.lines[0][2]) >> M.n & ((1 << M.r) - 1)


@pytest.mark.parametrize("edit,match", [
    (lambda t: t.replace("xdem 1 ", "xdem 2 ", 1), "unknown xdem version"),
    (lambda t: t.replace("observables 2 gauge", "gauge 3 observables 2 gauge"), "header field"),
    (lambda t: t.replace("tc 3", "tc 4"), "tc 4 out of range"),
    (lambda t: t.replace("tc 3", "tc 03"), "non-canonical"),
    (lambda t: t.replace("axis Q1 ", "axis Q2 ", 1), "out of order"),
    (lambda t: t.replace("D0 D1 L1 x Q2 Q3", "D0 D1 L1 x Q2 Q9"), "undeclared axis Q9"),
    (lambda t: t.replace("error(0.001) D2 L0", "axis Q8 img D0 gauge g0\nerror(0.001) D2 L0"), "after the first error line"),
    (lambda t: t.replace("axis Q0 img D0 D1 D2", "axis Q0 img D1 D0 D2"), "unsorted"),
    (lambda t: t.replace("axis Q0 img D0 D1 D2", "axis Q0 img D0 D0 D1 D2"), "duplicate"),
    (lambda t: t.replace("axis Q0 img", "axis Q0 sign + img"), "sign"),
    (lambda t: t.replace("axis Q0 img", "axis Q0  img"), "double space"),
    (lambda t: t.replace("axis Q1 img D1 D2 gauge g1\n", "axis Q1 img D1 D2 gauge g1 \n"), "trailing space"),
    (lambda t: t.replace("gauge g0\n", "gauge g0 hidden\n", 1), "empty"),
    (lambda t: t.replace("rot -Q2 +Q3", "rot Q2 +Q3", 1), "rot token"),
    (lambda t: t.replace("rot -Q2 +Q3", "rot -Q2 -Q2", 1), "repeated"),
    (lambda t: t.replace("x Q2 Q3 rot", "rot -Q2 +Q3 x Q2 Q3 rot", 1), "field out of order"),
    (lambda t: t.replace("gauge g0\n", "gauge g3\n", 1), "g3"),
    (lambda t: t.replace("hidden h0\n", "hidden h2\n", 1), "h2"),
    (lambda t: t.replace("axis Q1 img D1 D2 gauge g1", "axis Q1"), "no qubit"),
    (lambda t: t.replace("axis Q1 img D1 D2 gauge g1", "axis Q1 img D0 D2 gauge g1"), "rank"),          # §3.5(b)
    (lambda t: t.replace("D2 L0 x Q2 Q3", "D2 L0 x Q2 Q3 Q4"), "not an operator"),                     # §3.5(c)
    (lambda t: t.replace("error(0.001) D2", "error[t](0.001) D2"), "tag"),
    (lambda t: "\n".join(l for l in t.splitlines() if not l.startswith(("xdem", "axis"))) + "\n", "x/rot/late field"),
    (lambda t: t.replace("axis Q0", "AXIS Q0"), "instruction 'AXIS' \\(xdem spells it 'axis'\\)"),     # one spelling
    (lambda t: t.replace("error(0.001) D2 L0 x", "ERROR(0.001) D2 L0 x"), "instruction 'ERROR' \\(xdem spells it 'error'\\)"),
    (lambda t: t.replace("axis Q0 img", "axis Q0\timg"), "tab"),
    (lambda t: t + "foo 1\n", "unknown instruction 'foo'"),
    (lambda t: t + "logical_observable L2\n", "declares observable L2 \\(header: 2 observables\\)"),
    (lambda t: t.replace("error(0.001) D0 D1 D2 L0 L1", "repeat 2 {\nerror(0.001) D0 D1 D2 L0 L1"), "repeat block not closed"),
])
def test_refusals_name_the_cause_and_the_line(edit, match):
    text = edit(EX)
    with pytest.raises(XtimXdemError, match=match) as e:
        ta.xdem_parse(text)
    assert "line " in str(e.value)


def test_compose_refuses_an_out_of_range_id():
    with pytest.raises(XtimXdemError, match=r"^xdem: line id 3 out of range \(the file has 3 error lines\)$"):
        ta.xdem_compose(EX, [0, 3])


HDR1 = "xdem 1 detectors 2 observables 1 gauge 1 hidden 0 tc 1\n"


@pytest.mark.parametrize("text,facts", [
    (EX, ([], True, "", 0)),
    (HDR1 + "axis Q0 img D0 gauge g0\nerror(0.1) D0 rot +Q0\nerror(0.2) D1\n",
     ([], False, "line 4 (index \u2265 tc) flips detector D1", 0)),
    (HDR1 + "axis Q0 img D0 gauge g0\nerror(0.1) D0\nerror(0.2) L0 rot +Q0\n",
     ([], False, "line 4 (index \u2265 tc) carries a rotation", 0)),
    ("xdem 1 detectors 1 observables 1 gauge 1 hidden 0 tc 1\naxis Q0 img D0 L0 gauge g0\nerror(0.1) L0 late +Q0\n",
     ([0], False, "late axis Q0 has detector D0 in its img", 0)),
    ("xdem 1 detectors 1 observables 1 gauge 1 hidden 0 tc 1\naxis Q0 img L0 gauge g0\nerror(0.1) D0 rot +Q0\n",
     ([], True, "", 1)),                                              # rank img[A1] = 1, rank img[A1]|_D = 0
])
def test_file_facts(text, facts):
    """SPEC §4.4 / §4.6: late_axes, r2b_available (exact reasons), invisible_residues"""
    f = ta.xdem_facts(text)
    assert (f["late_axes"], f["r2b_available"], f["r2b_reason"], f["invisible_residues"]) == facts


def test_dc_v1_files_are_refused_with_the_reexport_message():
    for name in ("cultivation_d3_faithful.dem", "distillation_15_1_3.dem"):
        text = (DATA / "v1" / name).read_text()
        with pytest.raises(XtimXdemError, match='dc-DEM v1 file: re-export the circuit with format="xdem"'):
            ta.xdem_parse(text)
    marker_only = "".join(l for l in (DATA / "v1" / "distillation_15_1_3.dem").read_text().splitlines(True)
                          if "dc:" not in l)                           # no dc tag left: the marker alone
    assert "dc:" not in marker_only and "6463" in marker_only
    with pytest.raises(XtimXdemError, match="dc-DEM v1 file") as e:
        ta.xdem_parse(marker_only)
    assert f"line {len(marker_only.splitlines())}:" in str(e.value)


def test_conventional_dem_semantics():
    """§3.3: repeat unrolled, shift_detectors shifts later D's, ^ is a hint, declarations accepted (stim's meaning)"""
    stim = pytest.importorskip("stim")
    g = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01)
    for dem in (g.detector_error_model(), g.detector_error_model(decompose_errors=True)):
        flat = dem.flattened()
        a, b = ta.xdem_parse(str(dem)), ta.xdem_parse(str(flat))
        assert not a["extended"] and a["D"] == dem.num_detectors and a["O"] == dem.num_observables
        assert [l[:2] for l in a["lines"]] == [l[:2] for l in b["lines"]]
        assert a["tc"] == a["L"]


def test_conventional_lines_inside_an_extended_file():
    """§3.3 inside an extended file: shift_detectors shifts later axis and error lines; repeat unrolls and counts in tc"""
    t = ("xdem 1 detectors 4 observables 0 gauge 1 hidden 0 tc 3\n"
         "shift_detectors 2\n"
         "axis Q0 img D0 D1 gauge g0\n"
         "repeat 2 {\n  error(0.1) D0 rot +Q0\n}\n"
         "error(0.2) D1 ^ D0\n")
    d = ta.xdem_parse(t)
    assert d["axes"][0][1] == [2, 3]
    assert [l[1] for l in d["lines"]] == [[2], [2], [2, 3]] and d["tc"] == 3
    with pytest.raises(XtimXdemError, match="declares detector D4"):
        ta.xdem_parse(t + "detector D2\n")                                 # shifted: D4 >= D = 4


def test_truncated_file_is_a_smaller_model():                             # SPEC §7 Q6 (documented, no guard)
    """preflight B3: a cut at or after tc parses as a smaller model; a cut below tc is refused by the tc range"""
    t2 = EX.replace("tc 3", "tc 2")
    cut = "\n".join(t2.splitlines()[:-1]) + "\n"
    assert ta.xdem_parse(cut)["L"] == 2
    with pytest.raises(XtimXdemError, match=r"tc 3 out of range \(the file has 2 error lines\)"):
        ta.xdem_parse("\n".join(EX.splitlines()[:-1]) + "\n")


@pytest.mark.parametrize("seed", range(12))
def test_random_valid_files_with_no_gauge_qubit(seed):
    """preflight B1: r == 0 and m == 0 requested; the generator raises m to 1 (an axis needs a qubit), every axis
    is a hidden-syndrome string with no img, and the law is the oracle's"""
    rng = random.Random(1000 + seed)
    text = random_text(rng, D=rng.randint(1, 5), O=rng.randint(0, 2), r=0, m=0, A=rng.randint(1, 5),
                       L=rng.randint(1, 6))
    d = ta.xdem_parse(text)
    assert (d["r"], d["m"]) == (0, 1)
    assert all(not img and not gauge and hidden == [0] for (_, img, gauge, hidden) in d["axes"])
    assert ta.xdem_emit(text) == text
    L = d["L"]
    cfgs = [sorted(rng.sample(range(L), rng.randint(0, L))) for _ in range(6)] + [list(range(L))]
    laws_equal(text, cfgs)


# stim parity of the conventional reader (final review I2, M2): the reader accepts exactly what stim accepts on these
# inputs (stim 1.16 behaviour, pinned here so the test also runs without stim; checked against stim when installed).
# Lower-case target prefixes are accepted (stim's d/D, l/L), and the separator, declaration and number rules refused.
_STIM_PARITY = [
    ("error(0.1) d0", True), ("error(0.1) l0", True), ("error(0.1) d0 L0 ^ D1 l1", True),
    ("detector d0", True), ("detector(1, 2) d0", True), ("logical_observable l0", True),
    ("ERROR(0.1) d0", True), ("shift_detectors 1\nerror(0.1) d0", True),
    ("repeat 2 {\n error(0.1) d0\n shift_detectors 1\n}", True),
    ("error(0.1) D0 ^ D1", True), ("error(0.1)", True), ("error() D0", True), ("error(+0.1) D0", True),
    ("error(1e-1) D0", True), ("error(.1) D0", True), ("error( 0.1 ) D0", True), ("detector(-1) D0", True),
    ("shift_detectors(1, 2) 1", True), ("detector(1,) D0", True),
    ("error(0.1) D0 ^", False), ("error(0.1) ^ D0", False), ("error(0.1) D0 ^ ^ D1", False), ("error(0.1) ^", False),
    ("detector D0 D1", False), ("detector(1) D0 D1", False), ("detector", False),
    ("logical_observable L0 L1", False), ("logical_observable", False), ("logical_observable(1) L0", False),
    ("logical_observable() L0", False), ("detector L0", False), ("logical_observable d0", False),
    ("error(0x1p-3) D0", False), ("error(0X1P-3) D0", False), ("error(inf) D0", False), ("error(nan) D0", False),
    ("detector(0x1p1) D0", False), ("detector(nan) D0", False), ("detector(1e400) D0", False),
    ("shift_detectors(0x1p1) 1", False), ("error(0.1,0.2) D0", False), ("error(1_0) D0", False),
    ("error(0.1) x0", False), ("error(0.1) D", False),
]


@pytest.mark.parametrize("text,ok", _STIM_PARITY)
def test_conventional_reader_accepts_what_stim_accepts(text, ok):
    try:
        import stim
    except ImportError:
        stim = None
    if stim is not None:
        try:
            stim.DetectorErrorModel(text)
            stim_ok = True
        except Exception:  # noqa: BLE001
            stim_ok = False
        assert stim_ok == ok, "the pinned expectation disagrees with the installed stim"
    if ok:
        d = ta.xdem_parse(text)
        up = text.replace(" d", " D").replace(" l", " L")                  # the lower-case prefixes read as upper
        assert d == ta.xdem_parse(up)
    else:
        with pytest.raises(XtimXdemError, match="line "):
            ta.xdem_parse(text)


def test_lowercase_targets_and_separator_rules_in_an_extended_file():
    """§3.2/§3.3: an extended file's targets are stim targets (case-blind prefixes, stim's ^ rules); xdem's own fields
    (img, gauge, hidden) keep their one spelling"""
    t = ("xdem 1 detectors 2 observables 1 gauge 1 hidden 0 tc 2\n"
         "axis Q0 img D0 gauge g0\n"
         "error(0.1) D0 rot +Q0\n"
         "error(0.2) D1 ^ L0\n")
    low = t.replace("error(0.1) D0 rot", "error(0.1) d0 rot").replace("D1 ^ L0", "d1 ^ l0")
    assert ta.xdem_parse(low) == ta.xdem_parse(t)
    assert ta.xdem_parse(t + "detector d1\nlogical_observable l0\n") == ta.xdem_parse(t)
    for bad, match in [("D1 ^ L0", "D1 ^ L0 ^"), ("D1 ^ L0", "^ D1 ^ L0"), ("D1 ^ L0", "D1 ^ ^ L0")]:
        with pytest.raises(XtimXdemError, match="separator"):
            ta.xdem_parse(t.replace(bad, match))
    with pytest.raises(XtimXdemError, match="separator"):
        ta.xdem_parse(t.replace("error(0.1) D0 rot", "error(0.1) D0 ^ rot"))
    with pytest.raises(XtimXdemError, match="img token 'd0'"):
        ta.xdem_parse(t.replace("axis Q0 img D0", "axis Q0 img d0"))
