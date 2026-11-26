"""LEAN-EXPORT-PLAN: the lean xdem route (export_dem's default) is byte-identical to the oracle route (the
exporter as it was before the plan). Fields compared: FIELDS. The only exemption is owner decision OD3 (the oracle
raised BareStateCapacityError and the lean route, which builds no bare state, exported), and it is counted."""
from collections import Counter
from pathlib import Path

import pytest

from xtim import _xtim_testapi as ta
from xtim.errors import XtimDemError
from _xdem_circuits import NAMES_EXTERNAL, NAMES_INREPO, circuit, present
import _xdem_lean_fuzz as lf

REPO = Path(__file__).resolve().parent.parent
FILES = sorted({p.resolve() for d in ("tests/data", "xtim/_examples", "examples", "benchmarks")
                for p in (REPO / d).rglob("*.stim")
                if not p.relative_to(REPO).as_posix().startswith("examples/xdem/")})   # the pairs duplicate covered circuits; own test
FIELDS = ("ok", "dem", "error", "rejected", "reject_gate_index", "parse_errors", "parse_error_lines",
          "dropped_observables", "reject_detectors", "postselect_faults", "dropped_expectation_columns",
          "clifford_faults", "clifford_dropped_mass", "nondeterministic_targets", "exception",
          "exception_message")   # every DemExportResult field
FAST = lf.generate(1, 450)
BETA = lf.generate_beta(12345, 200)
BOUNDARY = lf.generate_boundary(77, 300)
SENTINEL = "xdem lean route"
NONDET = "non-deterministic "    # the determinism refusal's prefix (dem_export.cpp nondeterministic_message)
NO_GROUPS_FLOOR = 25           # Task 3 review N2; measured 31 at Task 4 (in-repo + fast)
NO_GROUPS_FLOOR_BOUNDARY = 15  # measured 20 at Task 4


def inrepo_texts():
    for n in NAMES_INREPO:
        yield n, circuit(n)
    for p in FILES:
        yield str(p.relative_to(REPO)), p.read_text()


def sweep(items, delegated=None, **kw):
    """The identity sweep. delegated (a list): receives (name, text) of every circuit the oracle exports (ok) and the
    lean route delegates, for assert_no_coverage_loss (Task 3 review M1)."""
    stats = Counter()
    for name, text in items:
        o = ta.xdem_export_route(text, "oracle", **kw)
        lean = ta.xdem_export_route(text, "lean", **kw)
        assert SENTINEL not in lean["error"] and SENTINEL not in o["error"], name
        assert lean["lean_exception"] == "", (name, lean["lean_exception"])   # review I2: no swallowed lean throw
        if o["exception"] == "BareStateCapacityError" and lean["ok"]:     # OD3 (a)
            if lean["dem"].startswith("xdem 1"):
                ta.xdem_parse(lean["dem"])     # raises XtimXdemError if §3.5(b)/(c) fail
            stats["od3_capacity_exempt"] += 1
            continue
        if o["exception"] == "BareStateCapacityError" and lean["error"].startswith(NONDET):   # nondet-message
            assert lean["lean_used"] and lean["nondeterministic_targets"], name
            stats["nondet_capacity_named"] += 1
            continue
        bad = [f for f in FIELDS if o[f] != lean[f]]
        assert not bad, (name, bad, o["error"][:200], lean["error"][:200])
        stats["lean" if lean["lean_used"] else "delegated"] += 1
        if delegated is not None and o["ok"] and not lean["lean_used"]:
            delegated.append((name, text))
        stats["extended" if lean["dem"].startswith("xdem 1") else "other"] += 1
    print("SWEEP", kw, len(items) if hasattr(items, "__len__") else "?", dict(stats))
    return stats


def assert_no_coverage_loss(delegated):
    """Task 3 review M1: OD4 admits every circuit with no PAULI_EXPECTATION, a writer-OK model and the H-condition
    false, so none of them may be delegated while the oracle exports it (a spurious lean refusal keeps the bytes
    equal, so the identity sweep alone cannot see it)."""
    lost = []
    for name, text in delegated:
        if "PAULI_EXPECTATION" in text:
            continue                        # OD4: oracle-only
        try:
            c = ta.xdem_class_probe(text)
        except XtimDemError:
            continue
        if c["writer"] == "ok" and not c["h_meets_location"]:
            lost.append(name)
    assert not lost, lost


def test_identity_inrepo():
    delegated = []
    s = sweep(inrepo_texts(), delegated)
    assert s["extended"] >= 30, s
    assert s["lean"] >= 50, s               # Task 3: 43 lean-served; nondet-message: 52 of 113 (lean_used now also
    #                                         counts the determinism refusals the lean route answers itself)
    assert_no_coverage_loss(delegated)


def test_identity_inrepo_approx():
    sweep(inrepo_texts(), approximate_disjoint_errors=0.1)


def test_identity_inrepo_canonical():
    sweep(inrepo_texts(), require_canonical=True)


def test_identity_fuzz_fast():
    delegated = []
    s = sweep(FAST, delegated)
    assert s["extended"] >= 150, s
    assert s["lean"] >= 300, s              # Task 3 review M1 (Task 3: 210 of 450; nondet-message: 309, the 99 more
    #                                         are determinism refusals the lean route now answers itself)
    assert_no_coverage_loss(delegated)


@pytest.mark.slow
def test_identity_external():
    sweep((n, circuit(n)) for n in NAMES_EXTERNAL if present(n))


@pytest.mark.slow
def test_identity_fuzz_slow():
    s = sweep(lf.generate(2, 9000))
    assert s["extended"] >= 3000, s


@pytest.mark.slow
def test_identity_xdem_fuzz_gen():
    from _xdem_fuzz_gen import generate
    sweep((f"g{i}", t) for i, t in enumerate(generate(5101, 300)))


def test_algebraic_determinism_matches_the_bare_state():
    det = non = 0
    for name, text in list(inrepo_texts()) + FAST:
        try:
            v = ta.xdem_determinism_verdicts(text)
        except XtimDemError:
            continue
        if v["capacity"] or v["bare_rejected"]:
            continue                       # no oracle verdict (OD3; F5 has its own test)
        assert v["lean"] == v["oracle"], name
        non += len(v["oracle"])
        det += v["ncols"] - len(v["oracle"])
    # plan time, fast fuzz alone (det.cpp): 1 551 deterministic and 449 non-deterministic columns
    assert non >= 300 and det >= 1000, (non, det)


def test_determinism_pair_coefficient():
    """CCZ|+++>, <X_0> = 1/2: every linear coefficient is 0 mod 4 and only the pair coefficient beta_12 is nonzero
    (no in-repo or fuzz circuit has such a column; Task 1 mutation check)."""
    v = ta.xdem_determinism_verdicts("RX 0 1 2\nX_ERROR(0.01) 0\nCCZ 0 1 2\nMX 0\nDETECTOR rec[-1]\n")
    assert v["lean"] == v["oracle"] == [0], v


def test_determinism_beta_fuzz():
    """The CCZ-heavy family (review M1): columns whose verdict rests on the pair coefficients, against the bare state."""
    adm = cols = non = 0
    for name, text in BETA:
        try:
            v = ta.xdem_determinism_verdicts(text)
        except XtimDemError:
            continue
        assert not v["capacity"] and not v["bare_rejected"], name
        assert v["lean"] == v["oracle"], name
        adm += 1
        cols += v["ncols"]
        non += len(v["oracle"])
    assert adm >= 150 and non >= 300 and cols - non >= 50, (adm, cols, non)


def test_identity_beta_fuzz():
    sweep(BETA)

def test_no_noncapacity_bare_refusal_on_admitted_circuits():
    checked = 0
    for name, text in list(inrepo_texts()) + FAST:
        try:
            v = ta.xdem_determinism_verdicts(text)
        except XtimDemError:
            continue                       # not admitted (writer refusal, diag, no groups)
        if v["capacity"]:
            continue                       # OD3, not this obligation
        assert not v["rotation_form_rejected"] and not v["bare_rejected"], name   # F5
        checked += 1
    assert checked >= 200, checked


def _t_wall(k):
    """k |+> qubits each with an X_ERROR then a T, never read (bare rank k), and one deterministic Z detector."""
    q = " ".join(map(str, range(k)))
    return (f"RX {q}\nX_ERROR(0.01) {q}\nT {q}\nR {k}\nX_ERROR(0.01) {k}\nM {k}\nDETECTOR rec[-1]\n")


def test_od3_bare_rank_cap_is_not_a_lean_refusal():
    o = ta.xdem_export_route(_t_wall(12), "oracle")
    lean = ta.xdem_export_route(_t_wall(12), "lean")
    assert o["ok"] and lean["ok"] and o["dem"] == lean["dem"] and lean["lean_used"]
    o = ta.xdem_export_route(_t_wall(31), "oracle")
    lean = ta.xdem_export_route(_t_wall(31), "lean")
    assert o["exception"] == "BareStateCapacityError", o
    assert lean["ok"] and lean["lean_used"] and lean["dem"].startswith("xdem 1"), lean
    assert sum(1 for ln in lean["dem"].splitlines() if ln.startswith("axis Q")) == 31
    facts = ta.xdem_parse(lean["dem"])     # the reader's §3.5 checks (header consistency, every line an operator)
    assert facts["extended"]


def test_h_condition_implies_the_diagonal_class():
    combos = Counter()
    no_groups = 0
    for name, text in list(inrepo_texts()) + FAST:
        try:
            c = ta.xdem_class_probe(text)
        except XtimDemError:
            continue
        if c["writer"] == "no_groups" and not c["h_meets_location"]:
            assert c["tdef"], name          # review M4: the lean route serves these too (its early `ok` return)
            no_groups += 1
        if c["writer"] != "ok":
            continue
        combos[(c["tdef"], not c["h_meets_location"])] += 1
        if not c["h_meets_location"]:
            assert c["tdef"], name          # soundness (F2): no H meets a location's Z => default table in class
    # plan-time counts on the fast fuzz alone: (True, True) 310, (True, False) 31, (False, False) 46. They were
    # measured with the r1 probe (writer without dem_preamble's feedback fold). If a bound fails only because of the
    # M6 change (circuits now "preamble_refused"), lower it to the new count, and record both counts in STATUS.
    assert combos[(True, True)] >= 300 and combos[(False, False)] >= 40, combos
    assert no_groups >= NO_GROUPS_FLOOR, no_groups   # Task 3 review N2: the M4 check is not vacuous


OUT_OF_CLASS = ("tests/data/xdem/circuits/ex422_obs.stim", "tests/data/ja_seed23.stim",
                "tests/data/dem_ch/nondiag_small.stim")   # every in-repo writer-OK circuit with tdef False (Task 2)


@pytest.mark.parametrize("path", OUT_OF_CLASS)
def test_diag_check_does_not_imply_the_class_gate(path):
    c = ta.xdem_class_probe((REPO / path).read_text())
    assert c["writer"] == "ok" and not c["tdef"] and not c["tdc"] and c["t3"] and c["h_meets_location"], c


def test_general_route_feedback_refusal_is_not_a_writer_refusal():
    hits = []
    for name, text in FAST:
        try:
            c = ta.xdem_class_probe(text)
        except XtimDemError:
            continue
        if c["writer"] == "ok" and not c["tdef"] and not c["tdc"] and c["t3"] and c["feedback"]:
            hits.append(name)
            o = ta.xdem_export_route(text, "oracle")
            assert "CH-class fault residuals" in o["error"], (name, o["error"])
    assert len(hits) >= 1, "plan time: 4 such circuits in the fast leg"


@pytest.mark.slow
def test_h_condition_implies_the_diagonal_class_slow():
    for name, text in lf.generate(2, 9000):
        try:
            c = ta.xdem_class_probe(text)
        except XtimDemError:
            continue
        if c["writer"] == "ok" and not c["h_meets_location"]:
            assert c["tdef"], name


def test_h_condition_implies_the_diagonal_class_boundary():
    # Task 2 review M1: the reviewer's boundary family (H next to forwarded location Z's; REPEAT, Y, CY/CZ-rec
    # feedback, SQRT_X/C_XYZ, SWAP/ISWAP, Y-basis MPP, zero-probability noise), a seeded sample in the fast leg
    combos = Counter()
    no_groups = 0
    for name, text in BOUNDARY:
        try:
            c = ta.xdem_class_probe(text)
        except XtimDemError:
            continue
        if c["writer"] in ("ok", "no_groups") and not c["h_meets_location"]:
            assert c["tdef"], name
            no_groups += c["writer"] == "no_groups"
        if c["writer"] == "ok":
            combos[(c["tdef"], not c["h_meets_location"])] += 1
    assert combos[(True, True)] >= 100 and combos[(True, False)] >= 20, combos
    assert no_groups >= NO_GROUPS_FLOOR_BOUNDARY, no_groups   # Task 3 review N2


def test_identity_boundary_fuzz():
    delegated = []
    s = sweep(BOUNDARY, delegated)
    assert s["lean"] >= 150, s              # Task 3 review M1 (Task 3: 3 of 300; nondet-message: 164, the 161 more
    #                                         are determinism refusals the lean route now answers itself)
    assert_no_coverage_loss(delegated)


BENCH_LEAN = ["cultivation_d3_faithful", "cultivation_d5", "color5_cultivation", "miniature_oracle",
              "distillation_15_1_3:gatenoise", "rep3T_pre", "toy2obs", "rm15rep", "ccz_toy", "ccz_toy_obs"]


def test_lean_serves_the_benchmarks():
    for n in BENCH_LEAN:
        r = ta.xdem_export_route(circuit(n), "lean")
        assert r["ok"] and r["dem"].startswith("xdem 1") and r["lean_used"], n


@pytest.mark.slow
def test_lean_serves_the_external_benchmarks():
    for n in ("bt27_p1e-3", "bt27_p1e-4", "msc_d3_canon", "msc_d5_canon", "rm15_x"):
        if present(n):
            r = ta.xdem_export_route(circuit(n), "lean")
            assert r["ok"] and r["lean_used"], n


def test_delegated_class_counterexamples_stay_identical():
    for path in OUT_OF_CLASS:   # review M5: nondiag_small too
        text = (REPO / path).read_text()
        o, lean = ta.xdem_export_route(text, "oracle"), ta.xdem_export_route(text, "lean")
        assert all(o[f] == lean[f] for f in FIELDS) and not lean["lean_used"], path


def test_lean_serves_feedback_circuits_in_class():
    # F2: residual feedback is admissible when the H-condition holds (the general-route feedback refusal needs
    # tdef == False, which the H-condition excludes)
    served = 0
    for name, text in FAST:
        try:
            c = ta.xdem_class_probe(text)
        except XtimDemError:
            continue
        if c["writer"] == "ok" and c["feedback"] and not c["h_meets_location"]:
            r = ta.xdem_export_route(text, "lean")
            served += bool(r["lean_used"] and r["ok"])
    assert served >= 5, served   # plan time: 26 feedback circuits with an extended export in the fast leg


def test_lean_export_is_reentrant():
    import threading
    a, b = circuit("cultivation_d5"), circuit("color5_cultivation")
    ref_a = ta.xdem_export_route(a, "oracle")["dem"]
    ref_b = ta.xdem_export_route(b, "oracle")["dem"]
    assert ta.xdem_export_route(a, "lean")["dem"] == ref_a
    assert ta.xdem_export_route(a, "lean")["dem"] == ref_a          # a second call in the same thread
    outs = [None] * 8

    def run(i):
        outs[i] = ta.xdem_export_route(a if i % 2 == 0 else b, "lean")["dem"]

    ts = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(o == (ref_a if i % 2 == 0 else ref_b) for i, o in enumerate(outs))


# ── nondet-message: the lean route answers the determinism refusal itself (no delegation, so no capacity error) ──
_FEEDBACK = __import__("re").compile(r"^\s*C[XYZ]\s.*rec\[")


def _strip_feedback(text):
    """Drop every classically controlled Pauli (CX/CY/CZ rec[-k] q): the circuit's byproducts go uncorrected."""
    return "\n".join(ln for ln in text.splitlines() if not _FEEDBACK.match(ln)) + "\n"


def _shift(text, qoff, ooff):
    """The circuit on qubits + qoff with observables + ooff (records are relative, so copies can be concatenated)."""
    import re
    out = []
    for ln in text.splitlines():
        s = ln.split("#")[0].rstrip()
        m = re.match(r"^(\s*)([A-Z_0-9]+)(\([^)]*\))?(.*)$", s)
        if not m:
            continue
        ind, g, args, rest = m.group(1), m.group(2), m.group(3) or "", m.group(4)
        if g in ("DETECTOR", "TICK", "SHIFT_COORDS"):
            out.append(s)
        elif g == "OBSERVABLE_INCLUDE":
            out.append(f"{ind}{g}({int(args[1:-1]) + ooff}){rest}")
        elif g != "OUTPUT_QUBITS":
            toks = [t if t.startswith("rec[") else re.sub(r"\d+", lambda d: str(int(d.group()) + qoff), t)
                    for t in rest.split()]
            out.append(f"{ind}{g}{args} " + " ".join(toks))
    return "\n".join(out) + "\n"


def _stripped_cultivation_copies(k):
    """k parallel copies of the public cultivation_d5 example with its Pauli feedback stripped: 10 non-deterministic
    detectors per copy (D38 D40 D42 D43 D46 D47 D69 D99 D101 D103 of the copy)."""
    base = _strip_feedback(circuit("cultivation_d5"))
    return "".join(_shift(base, 100 * i, 2 * i) for i in range(k))


def test_nondet_refusal_above_the_capacity_limit_names_the_detectors():
    """(a) 8 stripped cultivation_d5 copies: the oracle's bare state is above the rank limit (it raises
    BareStateCapacityError before its determinism check); the public export now names the detectors."""
    import xtim
    from xtim.errors import XtimCapacityError
    text = _stripped_cultivation_copies(8)
    o = ta.xdem_export_route(text, "oracle")
    assert o["exception"] == "BareStateCapacityError", o["exception"]
    lean = ta.xdem_export_route(text, "lean")
    first = "non-deterministic detector D38 (noiseless parity expectation 0.000000, not +-1); not DEM-expressible"
    assert lean["lean_used"] and lean["lean_exception"] == "" and lean["error"] == first, lean["error"]
    D = 107                                                  # detectors per copy
    per = [38, 40, 42, 43, 46, 47, 69, 99, 101, 103]
    assert lean["nondeterministic_targets"] == [f"D{d + D * i}" for i in range(8) for d in per]
    with pytest.raises(XtimDemError) as ei:
        xtim.Circuit(text).detector_error_model_text(format="xdem")
    assert not isinstance(ei.value, XtimCapacityError)
    msg = str(ei.value)
    assert msg.startswith("not DEM-expressible: non-deterministic detector D38 (noiseless parity expectation"), msg
    assert "detectors D38, D40, D42-D43, D46-D47, D69" in msg and "(80 in all)" in msg, msg
    assert "add the Pauli feedback" in msg and "drop those detectors/observables" in msg, msg
    assert len(ei.value.nondeterministic) == 80


SMALL_NONDET = [
    "RX 0\nX_ERROR(0.01) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n",                       # <X> = 1/sqrt2
    "RX 0\nX_ERROR(0.01) 0\nT_DAG 0\nMY 0\nDETECTOR rec[-1]\n",                   # <Y> = -1/sqrt2 (an inverted deferred read)
    "RX 0 1 2\nX_ERROR(0.01) 0\nCCZ 0 1 2\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",  # <X0> = 1/2, an observable
    "RX 0 1\nX_ERROR(0.01) 0 1\nT 0 1\nMX 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\n"
    "OBSERVABLE_INCLUDE(1) rec[-1]\n",
]


@pytest.mark.parametrize("text", SMALL_NONDET + [_strip_feedback(circuit("cultivation_d5")),
                                                 _strip_feedback(circuit("color5_cultivation"))])
def test_nondet_refusal_lean_equals_oracle(text):
    """(b) a small non-deterministic circuit: the lean route answers (no delegation) with the oracle's exception
    type and text, field for field; the public export raises XtimDemError with it."""
    import xtim
    o, lean = ta.xdem_export_route(text, "oracle"), ta.xdem_export_route(text, "lean")
    assert o["error"].startswith(NONDET) and o["nondeterministic_targets"], o["error"]
    assert all(o[f] == lean[f] for f in FIELDS), [f for f in FIELDS if o[f] != lean[f]]
    assert lean["lean_used"] and lean["lean_exception"] == ""
    with pytest.raises(XtimDemError) as ei:
        xtim.Circuit(text).detector_error_model_text(format="xdem")
    assert ei.value.engine_message == o["error"].removesuffix("; not DEM-expressible")
    assert ei.value.nondeterministic == o["nondeterministic_targets"]


def test_nondet_refusals_agree_on_the_fuzz_families():
    """(c) every oracle determinism refusal in the fast, beta and boundary families and the in-repo set: the lean
    route gives the same fields; the ones it admits (writer OK, H-condition false, no PAULI_EXPECTATION) it answers
    itself, and it answers no determinism refusal the oracle does not give."""
    answered = delegated = refusals = 0
    for name, text in list(inrepo_texts()) + FAST + BETA + BOUNDARY:
        o, lean = ta.xdem_export_route(text, "oracle"), ta.xdem_export_route(text, "lean")
        if lean["error"].startswith(NONDET):
            assert o["error"] == lean["error"], name     # lean never refuses where the oracle does not
        if not o["error"].startswith(NONDET):
            continue
        refusals += 1
        assert all(o[f] == lean[f] for f in FIELDS), (name, [f for f in FIELDS if o[f] != lean[f]])
        assert o["nondeterministic_targets"] and o["error"].startswith(
            f"non-deterministic {'detector' if o['nondeterministic_targets'][0][0] == 'D' else 'observable'} "
            f"{o['nondeterministic_targets'][0]} "), name
        if lean["lean_used"]:
            answered += 1
        else:
            delegated += 1
            if "PAULI_EXPECTATION" not in text:
                c = ta.xdem_class_probe(text)
                assert c["writer"] != "ok" or c["h_meets_location"], name   # delegated only outside OD4
    print("NONDET", refusals, answered, delegated)
    # measured at nondet-message: 552 refusals, 434 answered by the lean route, 118 delegated (outside OD4)
    assert answered >= 400 and refusals >= 500, (refusals, answered, delegated)


def test_column_expectation_matches_the_bare_state():
    """The lean refusal's printed value: column_expectation (exact, from the writer model) against the bare state's
    <P> (framed_expectation) on every readout column of the in-repo and fuzz circuits with an oracle verdict."""
    kinds = Counter()
    for name, text in list(inrepo_texts()) + FAST + BETA + BOUNDARY:
        try:
            v = ta.xdem_determinism_verdicts(text)
        except XtimDemError:
            continue
        if v["oracle"] is None:
            continue
        for t, a, b in zip(v["ev_cols"], v["lean_ev"], v["oracle_ev"]):
            assert abs(a - b) <= 1e-9, (name, t, a, b)
            kinds["+1" if a == 1 else "-1" if a == -1 else "0" if a == 0 else "+frac" if a > 0 else "-frac"] += 1
    print("EV", dict(kinds))
    # measured at nondet-message: +1 3586, 0 1451, +frac 415, -1 47, -frac 25 (5 524 columns)
    assert kinds["+1"] >= 3000 and kinds["0"] >= 1000 and kinds["+frac"] >= 300 and kinds["-1"] >= 30 \
        and kinds["-frac"] >= 15, kinds


_KEEP_ARGS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "PAULI_EXPECTATION"}


def _noiseless(text):
    """Every noise / measurement-flip probability set to 0 (the noiseless circuit, for the sampler's <parity>)."""
    import re
    out = []
    for ln in text.splitlines():
        m = re.match(r"^(\s*)([A-Z_0-9]+)\(([^)]*)\)(.*)$", ln)
        if m and m.group(2) not in _KEEP_ARGS:
            ln = f"{m.group(1)}{m.group(2)}({','.join('0' for _ in m.group(3).split(','))}){m.group(4)}"
        out.append(ln)
    return "\n".join(out) + "\n"


def _printed_value(err):
    import re
    return float(re.search(r"noiseless parity expectation (-?[0-9.]+),", err).group(1))


def _sampled_value(text, col, shots=40000):
    """The exact sampler's noiseless <(-1)^{column}> (detector col, or observable when col = ("L", k))."""
    import numpy as np
    import xtim
    s = xtim.Circuit(_noiseless(text)).compile_detector_sampler(seed=7)
    if col[0] == "D":
        bits = np.asarray(s.sample(shots))[:, col[1]]
    else:
        bits = np.asarray(s.sample(shots, separate_observables=True)[1])[:, col[1]]
    return 1.0 - 2.0 * bits.astype(float).mean()


_GATES = ["T 0\n", "T_DAG 0\n", "T 0\nS 0\n", "T_DAG 0\nS_DAG 0\n", "T 0\nS_DAG 0\n", "T_DAG 0\nZ 0\n"]
_READS = ["MX 0\n", "MY 0\n", "M 0\n", "MX !0\n", "MY !0\n", "H 0\nM 0\n"]


@pytest.mark.parametrize("gate", _GATES)
@pytest.mark.parametrize("read", _READS)
def test_nondet_refusal_value_matches_the_sampler(gate, read):
    """Fix round 1: the refusal's printed <parity> is the column's noiseless value as the exact sampler reports it
    (40 000 shots, |error| < 0.03), sign included, on both routes: the invert eliminate_hadamards stamps when it folds
    S_DAG / Z into the read is physics; a user `!` cancels against the sampler's reference (as in stim)."""
    text = "RX 0\nX_ERROR(0.01) 0\n" + gate + read + "DETECTOR rec[-1]\n"
    o, lean = ta.xdem_export_route(text, "oracle"), ta.xdem_export_route(text, "lean")
    if not o["error"].startswith(NONDET):
        return                                              # a deterministic case (+-1): nothing printed
    assert lean["error"] == o["error"] and lean["lean_used"], (o["error"], lean["error"])
    v, s = _printed_value(o["error"]), _sampled_value(text, ("D", 0))
    assert abs(v - s) < 0.03, (text, v, s)


_MIXED_DET = [("R 1\nX 1\n", "M 1\n"), ("RX 1\nZ 1\n", "MX 1\n"), ("RX 1\nS_DAG 1\n", "MY 1\n"),
              ("R 1\nX 1\n", "MR 1\n"), ("R 1\n", "M !1\n"), ("R 1\nX 1\n", "MPP Z1\n"), ("R 1\n", "MPP !Z1\n"),
              ("RY 1\n", "MPP Y1\n")]


@pytest.mark.parametrize("det1", _MIXED_DET)
@pytest.mark.parametrize("gate,read0", [("T 0\n", "MX 0\n"), ("T_DAG 0\n", "MY 0\n"), ("T_DAG 0\n", "MX 0\n")])
@pytest.mark.parametrize("col", ["DETECTOR rec[-1] rec[-2]\n", "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n"])
def test_nondet_refusal_value_mixed_columns_match_the_sampler(det1, gate, read0, col):
    """Fix round 2: a column mixing a non-deterministic read with a deterministic read that carries a folded or user
    sign. The printed value is the sampler's noiseless column average, on both routes."""
    prep1, read1 = det1
    text = "RX 0\n" + prep1 + gate + "X_ERROR(0.01) 0\n" + read0 + read1 + col
    o, lean = ta.xdem_export_route(text, "oracle"), ta.xdem_export_route(text, "lean")
    assert o["error"].startswith(NONDET), o["error"]
    assert all(o[f] == lean[f] for f in FIELDS) and lean["lean_used"], (o["error"], lean["error"])
    which = ("D", 0) if col.startswith("DETECTOR") else ("L", 0)
    v, s = _printed_value(o["error"]), _sampled_value(text, which)
    assert abs(v - s) < 0.03, (text, v, s)


def test_nondet_refusal_values_match_the_sampler_on_mixed_fuzz():
    """Fix round 2, the mixed family (_xdem_lean_fuzz.generate_mixed): every oracle determinism refusal is
    field-identical on the lean route, and its printed value is the sampler's noiseless column average."""
    import xtim
    checked = signed = 0
    for name, text in lf.generate_mixed(31, 400):
        o, lean = ta.xdem_export_route(text, "oracle"), ta.xdem_export_route(text, "lean")
        assert all(o[f] == lean[f] for f in FIELDS), (name, o["error"], lean["error"])
        if not o["error"].startswith(NONDET):
            continue
        first = o["nondeterministic_targets"][0]
        try:
            s = _sampled_value(text, (first[0], int(first[1:])), shots=20000)
        except xtim.XtimError:
            continue
        v = _printed_value(o["error"])
        assert abs(v - s) < 0.04, (name, text, first, v, s)
        checked += 1
        signed += v < -0.01
    print("MIXED", checked, signed)
    assert checked >= 300 and signed >= 10, (checked, signed)   # measured at fix round 2: 385 checked, 19 negative


def test_detector_determinism_signs_are_the_recorded_bit_value():
    """detector_determinism's signed parities are the recorded-bit convention (dem_export.hpp): the noiseless
    (-1)^(recorded bit), the user's `!` and the sign eliminate_hadamards folds into a read (Z;MX, S_DAG;MY) both
    included. Through e4604aa (pinned here by F2 as "unchanged") both were ignored; the values below are the exact
    sampler's recorded bits (tests/test_determinism_signs.py checks them against it)."""
    from xtim import _xtim
    for text, want in (("R 0\nX 0\nM 0\n", -1.0), ("R 0\nM !0\n", -1.0), ("R 0\nMPP !Z0\n", -1.0),
                       ("RX 0\nZ 0\nMX 0\n", -1.0), ("RX 0\nS_DAG 0\nMY 0\n", -1.0),
                       ("R 0\nH 0\nS 0\nMY 0\n", 1.0)):
        assert _xtim.detector_determinism(text + "DETECTOR rec[-1]\n")["detector_parity"] == [want], text


def test_nondet_refusal_values_match_the_sampler_on_fuzz():
    """Fix round 1, fuzz: the first refused column's printed value against the exact sampler (fast, beta, boundary)."""
    import xtim
    checked = signed = 0
    for name, text in FAST + BETA + BOUNDARY:
        o = ta.xdem_export_route(text, "oracle")
        if not o["error"].startswith(NONDET):
            continue
        first = o["nondeterministic_targets"][0]
        try:
            s = _sampled_value(text, (first[0], int(first[1:])), shots=20000)
        except xtim.XtimError:
            continue                                        # a circuit the sampler refuses
        v = _printed_value(o["error"])
        assert abs(v - s) < 0.04, (name, first, v, s)
        checked += 1
        signed += v < -0.01
    print("SAMPLED", checked, signed)
    assert checked >= 400 and signed >= 5, (checked, signed)   # measured at fix round 1: 543 checked, 9 negative


def test_nondet_hint_caps_the_runs_and_the_error_pickles():
    import pickle
    names = [f"D{2 * i}" for i in range(30)] + ["L0"]
    e = XtimDemError("non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1); "
                     "not DEM-expressible", nondeterministic=names)
    assert "D38, ..." in str(e) and "D40" not in str(e) and "(31 in all)" in str(e), str(e)
    assert e.nondeterministic == names                     # the full list
    f = pickle.loads(pickle.dumps(e))
    assert type(f) is XtimDemError and str(f) == str(e) and f.nondeterministic == names


def test_nondet_refusal_above_the_capacity_limit_prints_an_unsigned_magnitude():
    """Fix round 2: above the bare-state capacity limit the noiseless reference (the value's sign) is not available;
    a nonzero value prints as +-|v| (the oracle raises BareStateCapacityError there)."""
    text = _t_wall(31) + "RX 40\nX_ERROR(0.01) 40\nT 40\nMX 40\nDETECTOR rec[-1]\n"
    assert ta.xdem_export_route(text, "oracle")["exception"] == "BareStateCapacityError"
    lean = ta.xdem_export_route(text, "lean")
    assert lean["lean_used"] and lean["nondeterministic_targets"] == ["D1"], lean
    assert lean["error"] == ("non-deterministic detector D1 (noiseless parity expectation +-0.707107, not +-1); "
                             "not DEM-expressible"), lean["error"]


def test_nondet_refusal_reference_over_the_memory_budget_prints_an_unsigned_magnitude():
    """Fix round 3: the bare state fits but the noiseless reference run's scratch exceeds the memory budget (default
    2 GiB): the lean route still names the refusal, with +-|v| (before: lean threw, the oracle raised capacity)."""
    text = _t_wall(22) + "RX 40\nX_ERROR(0.01) 40\nT_DAG 40\nMY 40\nDETECTOR rec[-1]\n"
    lean = ta.xdem_export_route(text, "lean")
    assert lean["lean_used"] and lean["lean_exception"] == "" and lean["nondeterministic_targets"] == ["D1"], lean
    assert lean["error"] == ("non-deterministic detector D1 (noiseless parity expectation +-0.707107, not +-1); "
                             "not DEM-expressible"), lean["error"]
