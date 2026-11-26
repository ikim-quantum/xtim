"""format="xdem" export (SPEC §2, §3.4, §6 T1(a,b,e,f,g,h,i), T4 safety)."""
import math
import re
from pathlib import Path

import pytest
import stim
import xtim
from xtim import _xtim_testapi as ta
from xtim._exact_scope import explicit_exact
from xtim.errors import XtimDemError

from _exact_law import xtim_exact_law
from _xdem_circuits import NAMES_EXTERNAL, NAMES_INREPO, circuit, present, xdem_export
from _xdem_engine import residue_terms
from _xdem_model import XdemModel, affine
from test_xdem_oracle_configs import Gate, compare

DATA = Path(__file__).parent / "data" / "xdem"
GATE_TOKENS = re.compile(r"\bT\b|T_DAG|\bCS\b|CS_DAG|CCZ|\bG[0-9]|dc:|group")
LINE_COUNTS_SPEC = {"msc_d3_canon": 472, "cultivation_d3_faithful": 470, "msc_d5_canon": 3237, "bt27_p1e-4": 13050}
# The lines this writer emits under the merge guard (fix round 1, refined): a fault joins its key's line only past
# lines whose swap with it is invisible (readout image 0). On these four circuits no swap is visible, so L = the
# distinct (side, key) count; the commutation guard gave 1129 / 1113 / 7098 / 22495.
LINE_COUNTS = {"msc_d3_canon": 473, "cultivation_d3_faithful": 471, "msc_d5_canon": 3243, "bt27_p1e-4": 13083}


def names():
    return [n for n in NAMES_INREPO + NAMES_EXTERNAL if present(n)]


def atom_lines(text):
    with explicit_exact():                                               # the export is explicit (J-b1)
        return ta.xdem_atom_lines(text)


def test_worked_file_bytes():                                            # T1(g)
    assert xdem_export(circuit("ex422_obs")) == (DATA / "ex422_obs.xdem").read_text()


@pytest.mark.parametrize("name", NAMES_INREPO + NAMES_EXTERNAL)
def test_deterministic_canonical_and_gate_free(name):                    # T1(a), T1(b), "no gate names"
    if not present(name):
        pytest.skip("external circuit absent")
    plain = _outcome(lambda: xtim.Circuit(circuit(name)).detector_error_model_text())
    got = _outcome(lambda: xdem_export(circuit(name)))
    if plain[0] != "ok":
        # refused by the default export: xdem refuses alike (code_switching_*: a random observable), or is extended
        # (bt27: the default export's Clifford-fault cap; the skeleton route has no Clifford faults)
        assert got == plain or (got[0] == "ok" and got[1].startswith("xdem 1")), (got[0], plain)
        if got[0] != "ok":
            return
    a = got[1]
    assert xdem_export(circuit(name)) == a
    if a.startswith("xdem 1"):
        assert ta.xdem_emit(a) == a           # canonical; the grammar admits only +Q / -Q, so no exponent but +-1 (T1(i))
        assert not GATE_TOKENS.search(a)
    else:
        assert a == xtim.Circuit(circuit(name)).detector_error_model_text()   # §3.4: the conventional DEM


@pytest.mark.parametrize("name", ["msc_d3_canon", "cultivation_d3_faithful", "msc_d5_canon", "bt27_p1e-4"])
def test_strict_merge_counts_and_no_line_straddles_tc(name):             # T1(e), T1(f)
    if not present(name):
        pytest.skip("external circuit absent")
    text = circuit(name)
    x = xdem_export(text)
    d = ta.xdem_parse(x)
    # The SPEC counts refine today's (f, x) lines by (f, tau, E^<, E^>) only. Not in them: (1) a fault with no flip and
    # no x but a nonempty strict tau has no (f, x) line; such a tau-only line is kept (only an all-empty key is
    # dropped); (2) the side of t_c: a line after t_c whose (f, tau, E^<, E^>) equals a line's before it (bt27: the
    # 9 observable readout flips after t_c).
    keys = [(tuple(tg), tuple(tau), tuple(map(tuple, rot)), tuple(map(tuple, late))) for _, tg, tau, rot, late
            in d["lines"]]
    distinct = len(set(keys[:d["tc"]])) + len(set(keys[d["tc"]:]))  # (side, key) classes = the pre-erratum lines
    tau_only = len({k for k in keys if k[1] and not (k[0] or k[2] or k[3])})
    twins = len(set(keys[d["tc"]:]) & set(keys[:d["tc"]]))
    print(f"{name}: {d['L']} lines, {distinct} distinct (side, key), {tau_only} tau-only, {twins} after-t_c twins "
          f"(SPEC strict count before folding: {LINE_COUNTS_SPEC[name]})")
    assert d["L"] == LINE_COUNTS[name]
    assert distinct - tau_only - twins <= LINE_COUNTS_SPEC[name]
    tc_step = ta.xdem_header_debug(text)["tc_step"]
    n = 0
    for at in atom_lines(text):
        if at["line"] < 0:                                               # preflight S2: an atom on no line
            continue
        n += 1
        assert (at["line"] < d["tc"]) == (at["step"] <= tc_step), at
    assert n > 0


# The in-repo circuits format="xdem" refuses by name although the default export accepts them (fix round 1, controller
# ruling: kept, as in dc; a PAULI_EXPECTATION circuit exports with include_expectations=False).
XDEM_REFUSALS = {
    "tests/data/ch_cultivation.stim": 'format="xdem" with include_expectations=True',
    "tests/data/adaptq_born5_dec_enum_producer.stim": 'format="xdem": the circuit declares DECISION columns',
    "tests/data/adaptq_qrm_magic_producer.stim": 'format="xdem": the circuit declares DECISION columns',
    "tests/data/cultivation_d3_faithful.stim": 'format="xdem" with include_expectations=True',
    "tests/data/miniature_oracle.stim": 'format="xdem" with include_expectations=True',
}


# Hand refusals of the writer (Task 11 fix round 1: the diagonality check had no xdem-side test after the dc ctests
# went). A location's Z carried to a later location as X: refused, both locations named.
def test_residue_that_does_not_stay_diagonal_is_refused_naming_both_locations():
    c = xtim.Circuit("R 0\nH 0\nT 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n")
    with pytest.raises(XtimDemError, match="does not stay diagonal") as e:
        c.detector_error_model_text(format="xdem")
    assert "xdem writer: " in str(e.value) and "G0" in str(e.value) and "G1" in str(e.value), str(e.value)


def test_clifford_and_nonrotating_circuits_are_the_conventional_dem():  # T1(h)
    root = Path(__file__).parent.parent
    seen = set()
    for p in sorted((root / "xtim/_examples").glob("*.stim")) + sorted((root / "tests/data").glob("*.stim")):
        key = str(p.relative_to(root))
        c = xtim.Circuit(p.read_text())
        want = _outcome(lambda: c.detector_error_model_text())
        got = _outcome(lambda: c.detector_error_model_text(format="xdem"))
        if key in XDEM_REFUSALS:
            seen.add(key)
            assert want[0] == "ok" and got[0] == "XtimDemError" and XDEM_REFUSALS[key] in got[1], (key, got)
            if "include_expectations" in XDEM_REFUSALS[key]:   # the documented way out
                x = c.detector_error_model_text(format="xdem", include_expectations=False)
                assert x.startswith("xdem 1"), key
            continue
        if want[0] != "ok":                                  # refused by the default export: the same refusal
            assert got == want, key
            continue
        assert got[0] == "ok", (key, got)
        if not got[1].startswith("xdem 1"):
            assert got[1] == want[1], key
            stim.DetectorErrorModel(got[1])
    assert seen == set(XDEM_REFUSALS)
    d = xtim.Circuit(xtim.load_example("distillation_15_1_3").text)     # non-Clifford, Z noise only: nothing rotates
    assert d.detector_error_model_text(format="xdem") == d.detector_error_model_text()


TT_EARLY = """R 0
H 0
X_ERROR(0.1) 0
T 0
T 0
S_DAG 0
H 0
M 0
DETECTOR rec[-1]
"""
TT_SPLIT = """R 0 1
H 0
X_ERROR(0.1) 0
T 0
M 1
DETECTOR rec[-1]
T 0
S_DAG 0
H 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def test_two_t_gates_on_one_qubit():                                     # T1(i)
    a = xdem_export(TT_EARLY)
    assert not a.startswith("xdem 1") and a == xtim.Circuit(TT_EARLY).detector_error_model_text()
    b = xdem_export(TT_SPLIT)
    assert b.startswith("xdem 1")
    rot_lines = [l for l in b.splitlines() if " rot " in l]
    assert len(rot_lines) == 1 and re.search(r" rot ([+-])Q(\d+) late ([+-])Q\2$", rot_lines[0])


TT_TC = TT_SPLIT.replace("DETECTOR rec[-1]\n", "DETECTOR rec[-1]\nX_ERROR(0.05) 0\n").replace("M 0\n", "M(0.02) 0\n")


def test_lines_after_tc_in_repo():                                       # T1(f) on an in-repo circuit
    """TT_SPLIT plus a fault after t_c (before the late T: its rotation is all E^>) and a readout flip after t_c: the
    lines after t_c come last, tc counts the others, and an after-t_c line carries no E^<."""
    x = xdem_export(TT_TC)
    d = ta.xdem_parse(x)
    tc_step = ta.xdem_header_debug(TT_TC)["tc_step"]
    ats = atom_lines(TT_TC)
    assert {a["step"] > tc_step for a in ats} == {False, True}
    for at in ats:
        assert at["line"] >= 0 and (at["line"] < d["tc"]) == (at["step"] <= tc_step), at
    after = d["lines"][d["tc"]:]
    assert 0 < d["tc"] < d["L"] and len(after) == 2
    assert all(not rot for _, _, _, rot, _ in after)
    assert any(late for _, _, _, _, late in after)


# preflight B5: an option is refused only when the output is extended; otherwise format="xdem" is the ordinary export
# with the caller's options (S14: stim-identical Clifford output "with the same options").
OPTIONS = (dict(decompose_errors=True), dict(erasure=True), dict(clifford_cutoff=2), dict(clifford_cutoff=0),
           dict(clifford_cutoff=math.inf), dict(drop_gauge_observables=True), dict(include_expectations=False),
           dict(approximate_disjoint_errors=True), dict(ignore_decomposition_failures=True, decompose_errors=True))


def test_option_refusals_and_detector_error_model():
    c = xtim.Circuit(circuit("ex422_obs"))
    for kw in (dict(decompose_errors=True), dict(erasure=True), dict(clifford_cutoff=2),
               dict(drop_gauge_observables=True)):
        with pytest.raises(ValueError, match='format="xdem" does not take'):
            c.detector_error_model_text(format="xdem", **kw)
    with pytest.raises(ValueError, match="extended"):
        c.detector_error_model(format="xdem")
    g = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.01)
    cl = xtim.Circuit(str(g))
    assert isinstance(cl.detector_error_model(format="xdem"), stim.DetectorErrorModel)
    with pytest.raises(ValueError, match='format="xdem"'):
        c.detector_error_model_with_reject(format="xdem")


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as e:                                               # noqa: BLE001 - compared as (type, text)
        return (type(e).__name__, str(e))


_CLIFFORD = {
    "rep_d3": str(stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2,
                                          after_clifford_depolarization=0.01)),
    "surface_d3": str(stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=2,
                                              after_clifford_depolarization=0.001,
                                              before_measure_flip_probability=0.001)),
}


@pytest.mark.parametrize("kw", OPTIONS, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()))
@pytest.mark.parametrize("name", ["rep_d3", "surface_d3", "distillation_15_1_3", "TT_EARLY"])
def test_option_equality_when_the_output_is_conventional(name, kw):     # preflight B5, S14
    text = (_CLIFFORD.get(name) or (TT_EARLY if name == "TT_EARLY" else circuit(name)))
    c = xtim.Circuit(text)
    want = _outcome(lambda: c.detector_error_model_text(**kw))
    got = _outcome(lambda: c.detector_error_model_text(format="xdem", **kw))
    assert got == want
    if got[0] == "ok" and not kw.get("erasure"):
        assert not got[1].startswith("xdem 1")
        assert _outcome(lambda: str(c.detector_error_model(format="xdem", **{k: v for k, v in kw.items()
                                                                              if k != "erasure"}))) == \
            _outcome(lambda: str(c.detector_error_model(**{k: v for k, v in kw.items() if k != "erasure"})))


def test_require_canonical_is_never_forwarded_to_the_ordinary_export():
    c = xtim.Circuit(_CLIFFORD["rep_d3"])
    assert c.detector_error_model_text(format="xdem", require_canonical=True) == c.detector_error_model_text()
    x = xtim.Circuit(circuit("ex422_obs"))
    assert x.detector_error_model_text(format="xdem", require_canonical=True) == xdem_export(circuit("ex422_obs"))


@pytest.mark.parametrize("name", NAMES_INREPO)
def test_stim_fails_at_the_first_extended_token(name):                   # T4 / SPEC §3.6 (stim leg)
    x = _outcome(lambda: xdem_export(circuit(name)))
    if x[0] != "ok":                                                     # refused like the default export (above)
        assert x[0] == "XtimDemError"
        return
    x = x[1]
    if not x.startswith("xdem 1"):
        stim.DetectorErrorModel(x)
        return
    no_hdr = "\n".join(x.splitlines()[1:]) + "\n"
    no_axes = "\n".join(l for l in x.splitlines() if l.startswith("error")) + "\n"
    cases = [(x, "xdem"), (no_hdr, "axis")]
    if re.search(r" (x|rot|late) ", no_axes):
        cases.append((no_axes, "Unrecognized"))
    for t, msg in cases:
        with pytest.raises((ValueError, IndexError), match=msg):   # stim 1.16: IndexError for an instruction name
            stim.DetectorErrorModel(t)


# ── the late split E^> on real circuits with groups after t_c (carried from Task 3) ─────────────────────────────────
def _neg(e):
    return (-e) % 4


@pytest.mark.parametrize("name", ["rep3T_pre", "rm15_x"])
def test_late_fields_against_the_atom_x_patterns(name):
    """Every atom's xdem line against an independent recomputation from the atom's own readout flips and x-patterns
    (the export's census, xdem_atom_lines), the writer's groups (xdem_header_debug: user gate, (axis, sign) per v) and
    the Python residue table (tests/_xdem_engine.py): f = the atom's flips + the folded img's; rot = -E' summed over
    the groups g < Gl, late = over g >= Gl, each field folded (an exponent 2 -> img flip). (Until Task 11 the flips
    and x-patterns came from the dc-DEM v1 export of the same circuit, with the same result.)"""
    if not present(name):
        pytest.skip("external circuit absent")
    text = circuit(name)
    h = ta.xdem_header_debug(text)
    Gl, groups = h["Gl"], h["groups"]
    assert 0 < Gl < len(groups), (Gl, len(groups))
    x = xdem_export(text)
    d = ta.xdem_parse(x)
    img = {a: sum(1 << j for j in ax[1]) for a, ax in enumerate(d["axes"])}
    checked = late_lines = 0
    for at in atom_lines(text):
        if at["line"] < 0:
            continue
        fl = sum(1 << j for j in at["flips"])
        fields = []
        for early in (True, False):
            E = {}
            for g, xv in at["x"]:
                if (g < Gl) == early:
                    gate, v_axis = groups[g]
                    for v, e in residue_terms(gate, xv):
                        a, sg = v_axis[v - 1]
                        if a >= 0:
                            E[a] = (E.get(a, 0) + _neg(sg * e)) % 4
            for a in [a for a, e in E.items() if e == 2]:
                fl ^= img[a]
                del E[a]
            fields.append(sorted((a, e) for a, e in E.items() if e))
        p, tg, tau, rot, late = d["lines"][at["line"]]
        assert sum(1 << j for j in tg) == fl, at
        assert [tuple(r) for r in rot] == fields[0], at
        assert [tuple(r) for r in late] == fields[1], at
        checked += 1
        late_lines += bool(late)
    assert checked > 0 and late_lines > 0, (checked, late_lines)
    assert " late " in x


def test_xdem_atom_lines_shape():
    text = circuit("ex422_obs")
    ats = atom_lines(text)
    assert ats and all(set(a) == {"site", "addr", "line", "p", "step", "x", "flips"} for a in ats)
    d = ta.xdem_parse(xdem_export(text))
    assert sorted({a["line"] for a in ats}) == list(range(d["L"]))
    G = len(ta.xdem_header_debug(text)["groups"])
    assert all(0 <= g < G and xv > 0 for a in ats for g, xv in a["x"]), ats
    assert any(a["x"] for a in ats), ats                                  # the worked file's lines rotate


# ── writer fixtures (Step 5) ─────────────────────────────────────────────────────────────────────────────────────────
TDAG_CYCLE = """RX 0
X_ERROR(0.2) 0
T 0
X_ERROR(0.1) 0
T_DAG 0
X_ERROR(0.15) 0
T 0
X_ERROR(0.12) 0
T_DAG 0
MX 0
DETECTOR rec[-1]
"""
FIXTURES = ("cultivation_d3_faithful", "cultivation_d5", "distillation_15_1_3", "distillation_15_1_3:gatenoise",
            "color5_cultivation", "miniature_oracle", "code_switching_faithful:noobs", "code_switching_rate:noobs",
            "tdag_cycle")


def fixture_text(name):
    return TDAG_CYCLE if name == "tdag_cycle" else circuit(name)


def fixture_path(name, text):
    return DATA / (name.replace(":", "_") + (".xdem" if text.startswith("xdem 1") else ".dem"))


@pytest.mark.parametrize("name", FIXTURES)
def test_fixtures_are_this_writer(name):
    x = xdem_export(fixture_text(name))
    f = fixture_path(name, x)
    assert f.exists(), f
    assert x == f.read_text()


# ── fix round 1: the merge erratum (a fault joins its key's line only past commuting lines) ────────────────────────
def _law_key(o, c):
    return int(o, 16), tuple(sorted(int(x, 16) for x in c))


def _subset_law_mismatches(name, text, subsets=None, seed=5):
    """Per fault subset (indices into the exact-engine gate's atoms): the xdem file's law of the lines carrying the
    faults (each fault fires its line; repeats cancel) against xtim's exact law of those faults (the per-configuration
    gate of tests/test_xdem_oracle_configs.py: exact over GF(2)). Subsets holding two atoms of one channel instance
    (or one record twice) cannot be forced and are skipped. Until Task 11 the reference was the dc-DEM v1 file's law.
    Returns (number compared, disagreements, mixtures, examples)."""
    G = Gate(name, text)
    if subsets is None:
        subsets = _subsets(G, seed)
    cfgs = [list(S) for S in subsets if G.ok_slots(S)]
    agree, dis, mix, ex = compare(G, cfgs)
    return len(cfgs), dis, mix, ex


def _subsets(G, seed=5):
    import itertools
    import random
    n = len(G.atoms)
    if n <= 12:
        return [S for k in range(n + 1) for S in itertools.combinations(range(n), k)]
    rnd = random.Random(seed)
    out = [(i,) for i in range(n)]
    if n <= 200:
        out += list(itertools.combinations(range(n), 2))
    else:                                       # pairs: a later member of a multi-member line with any other atom
        later = [i for v in G.line_atoms.values() for i in v[1:]]
        out += [(i, rnd.randrange(n)) for i in later for _ in range(4)]
        out += [tuple(rnd.sample(range(n), 2)) for _ in range(5000)]
    out += [tuple(rnd.sample(range(n), rnd.randint(3, min(8, n)))) for _ in range(2000)]
    return out


# tdag_cycle with its qubit read by no detector (MX 0 unread) and a separate detected qubit: the axis has img = 0, so
# the swap sets {Q0} of `x Q0` / `x Q0 rot -Q0` are invisible and the equal keys merge (plusmodel/REPORT.md §3(c)).
TDAG_UNREAD = TDAG_CYCLE.replace("MX 0\nDETECTOR rec[-1]\n", "MX 0\nX_ERROR(0.05) 1\nM 1\nDETECTOR rec[-1]\n").replace(
    "RX 0\n", "RX 0\nR 1\n")
HAND = {"tdag_cycle": TDAG_CYCLE, "tdag_unread": TDAG_UNREAD}


def test_an_invisible_swap_set_merges():
    x = xdem_export(TDAG_UNREAD)
    assert "axis Q0 gauge g0\n" in x                                     # img(Q0) = 0
    lines = [l for l in x.splitlines() if l.startswith("error")]
    assert lines == ["error(0.28999999999999998) x Q0", "error(0.19600000000000001) x Q0 rot -Q0",
                     "error(0.050000000000000003) D0"]
    n, dis, mix, ex = _subset_law_mismatches("tdag_unread", TDAG_UNREAD)    # every subset (n atoms <= 12)
    assert n > 0 and dis == 0 and mix == 0, (n, dis, mix, ex)


@pytest.mark.parametrize("name", ["tdag_cycle", "tdag_unread", "miniature_oracle", "rep3T_pre", "rm15_x",
                                  "cultivation_d3_faithful"])
def test_law_per_fault_subset_against_the_exact_engine(name):
    if name not in HAND and not present(name):
        pytest.skip("external circuit absent")
    text = HAND[name] if name in HAND else circuit(name)
    n, dis, mix, ex = _subset_law_mismatches(name, text)
    print(f"{name}: {n} subsets, {dis} mismatches, {mix} mixtures")
    assert n > 0 and dis == 0 and mix == 0, ex


def test_tdag_cycle_is_not_merged_across_a_non_commuting_line():
    lines = [l for l in xdem_export(TDAG_CYCLE).splitlines() if l.startswith("error")]
    assert lines == ["error(0.20000000000000001) x Q0", "error(0.10000000000000001) x Q0 rot -Q0",
                     "error(0.14999999999999999) x Q0", "error(0.12) x Q0 rot -Q0"]


def test_tdag_cycle_sampled_p_d0():
    """P(D0 = 1): the file's exact value (every line subset, its law) equals xtim's exact law of the circuit (every
    fault configuration, tests/_exact_law.py; rel 1e-12; until Task 11: the dc-DEM v1 file's exact value), and the
    circuit sampler (8e6 shots) lies within 5 sigma of it. The pre-erratum merge gave 0.09800 (about 17 sigma off
    the sampler's 0.09976)."""
    import itertools
    import numpy as np

    def exact(P, laws):
        configs = [S for k in range(len(P) + 1) for S in itertools.combinations(range(len(P)), k)]
        tot = 0.0
        for S, (o, c) in zip(configs, laws([list(S) for S in configs])):
            w = 1.0
            for i, p in enumerate(P):
                w *= p if i in S else 1 - p
            rnd = any(int(x, 16) & 1 for x in c)
            tot += w * (0.5 if rnd else int(o, 16) & 1)
        return tot
    x = xdem_export(TDAG_CYCLE)
    px = exact([p for p, *_ in ta.xdem_parse(x)["lines"]], lambda cf: ta.xdem_laws(x, cf))
    law, tail = xtim_exact_law(TDAG_CYCLE)
    assert tail == 0, tail
    pv = sum(q for o, q in law.items() if o & 1)                         # D0 = bit 0
    assert abs(px - pv) <= 1e-12 * pv, (px, pv)
    n = 8_000_000
    d = xtim.Circuit(TDAG_CYCLE).compile_detector_sampler(seed=20261116).sample(n)
    m = float(np.mean(d[:, 0]))
    sig = (px * (1 - px) / n) ** 0.5
    assert abs(m - px) < 5 * sig, (m, px, sig)


# ── require_canonical (Task 11 fix round 1: the Python surface the deleted dc canonical tests covered) ─────────────
def test_require_canonical_refusals():
    c = xtim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\n")
    with pytest.raises(ValueError, match='require_canonical=True is a format="xdem" option'):
        c.detector_error_model_text(require_canonical=True)
    with pytest.raises(ValueError, match="require_canonical must be a bool"):
        c.detector_error_model_text(format="xdem", require_canonical=1)
    with pytest.raises(xtim.XtimParseError):
        xtim.xdem.canonical_report("H 0\nNOPE 1\n")
    with pytest.raises(TypeError):
        xtim.xdem.canonical_report(3)


_ROOT = Path(__file__).resolve().parents[1]
_CIRCS = sorted(str(p) for p in (_ROOT / "tests/data").glob("*.stim")) + \
    sorted(str(p) for p in (_ROOT / "xtim/_examples").glob("*.stim"))


# the in-repo circuits the xdem writer refuses (DECISION columns, a non-deterministic detector or observable, a channel
# needing approximate_disjoint_errors); any other refusal fails the test
_OUT_OF_CLASS = frozenset({
    "adaptq_born5_dec_enum_producer.stim", "adaptq_qrm_magic_producer.stim",
    "code_switching_faithful.stim", "code_switching_rate.stim", "n1_mechanisms.stim", "sign_live_bytes_c40022.stim",
})


@pytest.mark.parametrize("path", _CIRCS, ids=[Path(p).name for p in _CIRCS])
def test_require_canonical_agrees_with_canonical_report(path):
    """require_canonical=False is the default; on an extended export =True gives the same bytes on a canonical
    circuit and refuses a non-canonical one with canonical_report's reason (a conventional export never checks)."""
    c = xtim.Circuit(Path(path).read_text())
    if Path(path).name in _OUT_OF_CLASS:
        with pytest.raises(XtimDemError):
            c.detector_error_model_text(format="xdem", include_expectations=False)
        pytest.skip("outside the xdem writer's class (refusal checked)")
    base = c.detector_error_model_text(format="xdem", include_expectations=False)
    assert c.detector_error_model_text(format="xdem", include_expectations=False, require_canonical=False) == base
    if not base.startswith("xdem 1"):
        pytest.skip("conventional export: require_canonical is never forwarded (tested above)")
    ok, why = xtim.xdem.canonical_report(c)
    if ok:
        assert c.detector_error_model_text(format="xdem", include_expectations=False, require_canonical=True) == base
    else:
        with pytest.raises(XtimDemError) as e:
            c.detector_error_model_text(format="xdem", include_expectations=False, require_canonical=True)
        assert "require_canonical=True" in str(e.value) and why in str(e.value)
