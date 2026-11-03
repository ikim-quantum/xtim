"""demfb (option OC): DEM export of circuits with residual classical Pauli feedback.

Feedback whose Pauli reaches a T/CS/CCZ/CH is coherentized (as in the sampler); the rest (residual) is stripped and
folded back exactly: the reported records are a unitriangular GF(2) image L of the stripped circuit's records, so
every record-defined column reads its L^T-relabelled record set (detectors, observables, DECISIONs, expectation
frames + feedback signs), and an OUTPUT_QUBITS frame column gains the controls whose feedback leaves a Pauli on its
wire. The per-fault correction Q_j is the identity (the commutator of a residual op and a later fault is conjugated by
every later gate and starts as a Pauli commutator): export_dem checks it per alternative.

Oracles: exact_sv (an independent branching statevector simulator that applies the feedback per branch) for every
alternative of every channel (check_circuit, exact to 1e-12); the measured twin for the frame columns; the operator
rule for the expectation signs.
"""
import pytest

import _exact  # noqa: E402  (J-b1 F: raw exact oracle calls inside the explicit-exact scope)
import exact_sv as E
import xtim
from xtim import _xtim

stim = pytest.importorskip("stim")
census_api = pytest.importorskip("xtim._xtim_testapi")


def export(text, approx):
    return _exact.export_dem_text(text, True, False, False, True, [], [], approx, -1)


def census(text):
    from xtim._exact_scope import explicit_exact
    with explicit_exact():
        return census_api.dem_law_census(text, True, 0.0, -1)


# Teleportation of |+> (q0 -> q2): X / Z corrections by feedback, the teleported state read in X.
TELEPORT = """RX 0
R 1 2
H 1
CX 1 2
DEPOLARIZE1(0.02) 0 1 2
CX 0 1
H 0
DEPOLARIZE2(0.03) 0 1
M 0 1
CX rec[-1] 2
CZ rec[-2] 2
DEPOLARIZE1(0.02) 2
MX 2
DETECTOR rec[-1]
"""

# A chain: op 1's control (record 1) is itself flipped by op 0.
CHAIN = """R 0 1 2
H 0
CX 0 1
X_ERROR(0.1) 0
M 0
CX rec[-1] 1
DEPOLARIZE1(0.05) 1
M 1
CX rec[-1] 2
X_ERROR(0.05) 2
M 2
DETECTOR rec[-2]
DETECTOR rec[-1]
"""

# The fixplan's synthetic Q case: residual X feedback AFTER a T onto the T's wire, controlled by a random record,
# with faults before the T (their residual reaches the wire). The fault precedes the op, so no correction applies
# (the ordering-blind formula P C P C^dag would be Z on wire 1 here); exact_sv arbitrates.
T_AFTER = """R 0 1 2
H 0 1
CX 1 2
X_ERROR(0.1) 1
Y_ERROR(0.07) 1
DEPOLARIZE1(0.05) 2
T 1
T_DAG 2
M 0
CX rec[-1] 1
CX rec[-1] 2
DEPOLARIZE2(0.04) 1 2
MX 1 2
DETECTOR rec[-1] rec[-2]
"""

# Feedback onto a wire read later in the Y basis, a Clifford coupling after the feedback, and a DECISION.
MIXED = """R 0 1 2 3
H 0 2
CX 2 3
X_ERROR(0.03) 0
T 2
T_DAG 3
M 0
CZ rec[-1] 2
CY rec[-1] 3
PAULI_CHANNEL_1(0.01,0.02,0.03) 2
CX 2 3
Z_ERROR(0.05) 3
MX 3
MX 2
DETECTOR rec[-1]
"""


# Review fix round 1: Hadamard elimination splits a noise instruction whose qubits carry different pending frames
# into several deferred instructions; the fault-vs-feedback order is read off the provenance tag, not the position.
# (check_circuit inserts single-qubit one-hots, so it never splits: the split itself is pinned by the full-vs-twin
# equality below.)
SPLIT_NOISE = """R 0 1 2
H 2
M 2
CZ rec[-1] 1
H 0
X_ERROR(0.1) 0 1
H 0
M 0 1
DETECTOR rec[-1]
DETECTOR rec[-2]
"""
SPLIT_NOISE_READ = """R 0 1 2
H 2
M 2
CX rec[-1] 1
H 0
X_ERROR(0.1) 0 1
M 0 1
DETECTOR rec[-1] rec[-3]
"""


@pytest.mark.parametrize("text", [TELEPORT, CHAIN, T_AFTER, MIXED, SPLIT_NOISE, SPLIT_NOISE_READ],
                         ids=["teleport", "chain", "t_after", "mixed", "split_noise", "split_noise_read"])
def test_feedback_circuits_exact_per_alternative(text):
    r = export(text, 0.0)
    assert r["ok"], r
    cnt = E.check_circuit(text, export)
    assert cnt["alts"] > 0, cnt
    c = census(text)
    assert c["fb_ops"] > 0, c
    assert c["fb_q_corrections"] == 0 and c["fb_q_pairs"] == 0, c


def test_t_after_feedback_has_clifford_faults():
    # the Q question needs a Clifford fault whose residual reaches the feedback's wire
    c = census(T_AFTER)
    kinds = {cm["kind"] for inst in c["instances"] for cm in inst["comps"]}
    assert kinds & {"C", "M", "L"}, kinds


def test_feedback_free_and_coherentized_circuits_fold_nothing():
    assert census(TELEPORT.replace("CX rec[-1] 2\nCZ rec[-2] 2\n", ""))["fb_ops"] == 0
    # the feedback crosses a T: coherentized, no residual op
    crossing = "R 0 1\nH 0\nM 0\nCX rec[-1] 1\nT 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1] rec[-2]\n"
    c = census(crossing)
    assert c["ok"], c
    assert c["fb_ops"] == 0, c


def _frame_twins(state_prep, read, frame_col):
    """(A) teleport with OUTPUT_QUBITS 2, (B) the same with the output read and declared an observable. A's frame
    column `frame_col` (0 X, 1 Z) must carry B's observable law, jointly with the detectors."""
    body = TELEPORT.replace("RX 0\n", state_prep).replace("MX 2\nDETECTOR rec[-1]\n", "")
    body += "M 1\nDETECTOR rec[-1] rec[-2]\n"          # keep a detector: re-read the X-correction control
    a = body + "OUTPUT_QUBITS out 2\n"
    b = body + f"{read} 2\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    return a, b


@pytest.mark.parametrize("prep,read,col", [("RX 0\n", "MX", 1), ("R 0\n", "M", 0)], ids=["frameZ", "frameX"])
def test_output_frame_columns_match_the_measured_twin(prep, read, col):
    a, b = _frame_twins(prep, read, col)
    ra, rb = export(a, 0.0), export(b, 0.0)
    assert ra["ok"], ra
    assert rb["ok"], rb
    D = 1
    ma = E.dem_mechs(ra["dem"], D)
    mb = E.dem_mechs(rb["dem"], D)
    # A's frame column col is target D + col (O = R = 0): project it onto B's observable index D + 0
    pa = {}
    for s, p in ma.items():
        t = frozenset(x for x in s if x < D) | (frozenset([D]) if D + col in s else frozenset())
        pa[t] = pa.get(t, 0.0) + p - 2 * pa.get(t, 0.0) * p
    ga, gb = E.mech_dist(pa, frozenset([0, D])), E.mech_dist(mb, frozenset([0, D]))
    for k in set(ga) | set(gb):
        assert abs(ga.get(k, 0.0) - gb.get(k, 0.0)) <= 1e-12, (sorted(k), ga.get(k), gb.get(k))
    assert any(D + col in s for s in ma), "the frame column never flips (the feedback part is missing)"
    E.check_circuit(b, export)                          # B against the exact oracle


def test_operator_oracle_bell_feedback_sign():
    # Z_ERROR(0.2) flips MX 0; the feedback then applies Z1, which flips <X1>: L0 at 0.2.
    text = "R 0 1\nH 0\nCX 0 1\nZ_ERROR(0.2) 0\nMX 0\nCZ rec[-1] 1\nPAULI_EXPECTATION(0) X1\n"
    r = export(text, 0.0)
    assert r["ok"], r
    assert E.dem_mechs(r["dem"], 0) == pytest.approx({frozenset([0]): 0.2}, rel=1e-15)


def test_more_than_64_expectation_columns_with_feedback():
    # the feedback's expectation signs fold into record functionals: no 64-column limit (esign is a word vector)
    text = "R 0 1\nH 0\nCX 0 1\nZ_ERROR(0.2) 0\nMX 0\nCZ rec[-1] 1\n"
    text += "".join(f"PAULI_EXPECTATION({r}) X1\n" for r in range(65))
    r = export(text, 0.0)
    assert r["ok"], r
    assert E.dem_mechs(r["dem"], 0) == pytest.approx({frozenset(range(65)): 0.2}, rel=1e-15)


# Owner decision 2026-11-01: a PAULI_EXPECTATION declared BEFORE a feedback carries that feedback's sign (the engine's
# convention, docs/usage.md: the declared Pauli is read against the end-of-circuit state). exact_sv used to evaluate in
# place with no flip and disagreed with the DEM on every *_flips case; it now applies the sign. The X fault on q0 fires
# the feedback; the column flips iff the feedback Pauli anticommutes with the declared Pauli.
EXP_BEFORE_FB = {
    "cz_flips": ("R 0\nRX 1\nX_ERROR(0.2) 0\nM 0\nPAULI_EXPECTATION(0) X1\nCZ rec[-1] 1\n", {frozenset([0]): 0.2}),
    "s_between_flips": ("R 0 1\nX_ERROR(0.2) 0\nM 0\nPAULI_EXPECTATION(0) Z1\nS 1\nCX rec[-1] 1\n",
                        {frozenset([0]): 0.2}),
    "two_qubit_flips": ("R 0 2\nRX 1\nH 2\nX_ERROR(0.2) 0\nM 0\nPAULI_EXPECTATION(0) X1*X2\nCY rec[-1] 2\n",
                        {frozenset([0]): 0.2}),
    "cx_commutes": ("R 0\nRX 1\nX_ERROR(0.2) 0\nM 0\nPAULI_EXPECTATION(0) X1\nCX rec[-1] 1\n", {}),
}


@pytest.mark.parametrize("name", sorted(EXP_BEFORE_FB))
def test_expectation_before_feedback_oracle_agrees_with_dem(name):
    text, want = EXP_BEFORE_FB[name]
    r = export(text, 0.0)
    assert r["ok"], r
    assert E.dem_mechs(r["dem"], 0) == pytest.approx(want, rel=1e-15)
    orc = E.Oracle(text)
    law = orc.law(orc.noise_idx[0], {0: "X"})
    flips = frozenset().union(*want) if want else frozenset()
    assert law == pytest.approx({(flips, False): 1.0}), law                # the oracle applies the sign itself
    E.check_circuit(text, export)


# Fix round 1: the same end-of-circuit convention for EVERY Pauli after a declaration (an unconditional X/Y/Z gate, an
# inserted fault), not only feedback. exact_sv (before and after ce92fd8) read these columns in place and missed the
# flip (check_circuit failed with 'alternative law' on z_decl_x_fault). Each case: per noise line and fault, the
# oracle's flipped targets (D + O + column numbering); the noiseless value against the exact engine; DEM == oracle for
# every alternative (check_circuit).
LATER_PAULI = {
    "z_decl_x_fault": ("R 1\nPAULI_EXPECTATION(0) Z1\nX_ERROR(0.2) 1\n", {(0, "X"): {0}}),
    "z_decl_z_fault": ("R 1\nPAULI_EXPECTATION(0) Z1\nZ_ERROR(0.2) 1\n", {(0, "Z"): set()}),
    "x_decl_y_fault": ("RX 1\nPAULI_EXPECTATION(0) X1\nY_ERROR(0.2) 1\n", {(0, "Y"): {0}}),
    "mid_circuit_zz": ("R 0 1\nH 0\nCX 0 1\nPAULI_EXPECTATION(0) Z0*Z1\nDEPOLARIZE1(0.15) 1\nM 0 1\n"
                       "DETECTOR rec[-1] rec[-2]\n",
                       {(0, "X"): {0, 1}, (0, "Y"): {0, 1}, (0, "Z"): set()}),
    "x_gate_after": ("R 1\nPAULI_EXPECTATION(0) Z1\nX 1\nX_ERROR(0.2) 1\nZ_ERROR(0.1) 1\n",
                     {(0, "X"): {0}, (1, "Z"): set()}),
    "fault_and_feedback": ("R 0\nRX 1\nX_ERROR(0.2) 0\nM 0\nPAULI_EXPECTATION(0) X1\nZ_ERROR(0.1) 1\n"
                           "CZ rec[-1] 1\nX_ERROR(0.05) 1\n",
                           {(0, "X"): {0}, (1, "Z"): {0}, (2, "X"): set()}),
}


@pytest.mark.parametrize("name", sorted(LATER_PAULI))
def test_pauli_after_declaration_oracle_agrees_with_dem(name):
    import numpy as np
    import _exact_engine
    text, want = LATER_PAULI[name]
    assert export(text, 0.0)["ok"]
    orc = E.Oracle(text)
    for (k, P), flips in want.items():
        line = orc.noise_idx[k]
        q = int(orc.lines[line].split()[-1])
        law = orc.law(line - k, {q: P})
        assert law == pytest.approx({(frozenset(flips), False): 1.0}), (k, P, law)
    clean = "\n".join(orc.base) + "\n"                 # the noiseless value: oracle (in place + signs) == engine
    _, e = _exact_engine.exact_detector_sampler(xtim.Circuit(clean), seed=1).sample(4, return_expectations=True)
    assert np.allclose(e, orc.v0[0]), (e[0], orc.v0)
    E.check_circuit(text, export)


def test_pauli_after_changed_declaration_is_outside_exact_sv():
    # a fault after a declaration that a later H changed: raise, never compare in place
    with pytest.raises(ValueError, match="end-of-circuit"):
        E.check_circuit("R 1\nPAULI_EXPECTATION(0) Z1\nH 1\nX_ERROR(0.2) 1\n", export)


def test_expectation_changed_before_feedback_is_outside_exact_sv():
    # A later H changes the declared X1: the engine reads X1 on the end state (0 here), exact_sv in place (+-1). The
    # oracle refuses instead of comparing two different quantities.
    text = "R 0\nRX 1\nX_ERROR(0.2) 0\nM 0\nPAULI_EXPECTATION(0) X1\nH 1\nCX rec[-1] 1\n"
    with pytest.raises(ValueError, match="end-of-circuit"):
        E.Oracle(text)


def test_column_layout_accepts_feedback():
    a, _ = _frame_twins("RX 0\n", "MX", 1)
    lay = _xtim.dem_column_layout(a)   # structural (no exact construction)
    assert lay["ok"], lay
    assert lay["frame"] == [(2, 0, 1)], lay


def test_python_api_exports_feedback():
    dem = xtim.Circuit(TELEPORT).detector_error_model()   # the public API enters the scope itself
    assert dem.num_detectors == 1


@pytest.mark.parametrize("text", [SPLIT_NOISE, SPLIT_NOISE_READ], ids=["split_noise", "split_noise_read"])
def test_split_noise_matches_the_sampler(text):
    # DEM per-detector rates vs the sampler (the exact per-alternative check is above)
    import numpy as np
    c = xtim.Circuit(text)
    dem = stim.DetectorErrorModel(c.detector_error_model_text())
    shots = 200000
    a = dem.compile_sampler(seed=7).sample(shots)[0].mean(axis=0)
    b = c.compile_detector_sampler(seed=7).sample(shots).mean(axis=0)
    se = np.sqrt(np.maximum((a + b) / 2 * (1 - (a + b) / 2), 1.0 / shots) * 2 / shots)
    assert (np.abs(a - b) / se).max() < 5, (a, b)


def test_residual_feedback_reaching_a_coherentized_control_is_refused_by_name():
    # a residual op (CY onto wire 0) reaches the control of a coherentized feedback (CY rec -> 3, crossing T 3): the
    # coherent gate carries it into the T. The relabel preset refuses (as detector_determinism does); no engine bug.
    text = ("R 0 1 2 3 4\nMX 4\nCY rec[-1] 0\nS 0\nCY rec[-1] 1\nT_DAG 4\nM 3\nM(0.01) 0\n"
            "DEPOLARIZE1(0.05) 0 4 1 2\nX 3\nCY rec[-1] 3\nT 3\nZ 3\nM(0.01) 4\nM 0 1 2 3 4\nDETECTOR rec[-1]\n")
    r = export(text, 0.0)
    assert not r["ok"] and "is not coherentized" in r["error"] and "`CY rec[-1] " in r["error"], r
    assert "engine bug" not in r["error"] and "deferred gate" not in r["error"]
    assert _xtim.detector_determinism(text)["rejected"]


# Re-review: every multi-qubit noise line vs its twin (one line per qubit / pair, so nothing splits). The full DEM must
# equal the twin's; the twin is checked against exact_sv where the oracle applies. Cases (reviewer's adversarial set):
# H / S / SQRT_X partial frames, 2-qubit channels across mixed frames, T-crossing + residual feedback with M(p), a
# readout fold, REPEAT (feedback in / out of the body), zero-probability noise, an OUTPUT_QUBITS split.
SPLIT_CASES = {
 # H split, feedback before/after the split instruction, read in Z and X
 "h_before": "R 0 1 2\nH 2\nM 2\nCZ rec[-1] 1\nH 0\nX_ERROR(0.1) 0 1\nH 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n",
 "h_after_fb": "R 0 1 2\nH 2\nM 2\nH 0\nX_ERROR(0.1) 0 1 2\nCX rec[-1] 1\nM 0 1\nDETECTOR rec[-1] rec[-3]\n",
 # S fold on part of the qubits (X -> Y on 0 only)
 "s_split": "R 0 1 2\nH 2\nM 2\nCX rec[-1] 0\nH 0 1\nS 0\nZ_ERROR(0.1) 0 1\nS_DAG 0\nH 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2] rec[-3]\n",
 "sqrtx_split": "R 0 1 2\nH 2\nM 2\nCZ rec[-1] 1\nSQRT_X 0\nDEPOLARIZE1(0.1) 1 0 2\nSQRT_X_DAG 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n",
 # two-qubit channels across mixed frames
 "dep2_split": "R 0 1 2 3\nH 3\nM 3\nCX rec[-1] 2\nH 0\nDEPOLARIZE2(0.1) 0 1 2 3\nH 0\nM 0 1 2\nDETECTOR rec[-1] rec[-4]\nDETECTOR rec[-2]\nDETECTOR rec[-3]\n",
 "pc2_split": "R 0 1 2 3\nH 3\nM 3\nCZ rec[-1] 2\nH 2\nPAULI_CHANNEL_2(0.01,0.02,0.03,0.01,0.02,0.03,0.01,0.02,0.03,0.01,0.02,0.03,0.01,0.02,0.03) 0 1 2 3\nH 2\nM 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-3]\n",
 # feedback near magic: coherentized (crossing T) + residual, split noise between them and noisy readouts
 "hoist_mix": "R 0 1 2 3\nH 3\nM(0.05) 3\nCX rec[-1] 1\nT 1\nH 0\nX_ERROR(0.1) 0 1 2\nCX rec[-1] 2\nH 0\nT_DAG 1\nM 0 1 2\nDETECTOR rec[-1] rec[-4]\nDETECTOR rec[-3]\n",
 "readout_fold_split": "R 0 1 2\nH 2\nM(0.1) 2\nCX rec[-1] 1\nH 0\nZ_ERROR(0.1) 0 1\nH 0\nM 0 1\nDETECTOR rec[-1] rec[-3]\nDETECTOR rec[-2]\n",
 # REPEAT body: one user instruction, several deferred copies
 "repeat": "R 0 1 2\nH 2\nM 2\nCX rec[-1] 1\nREPEAT 2 {\nH 0\nX_ERROR(0.1) 0 1\nH 0\n}\nM 0 1\nDETECTOR rec[-1] rec[-3]\nDETECTOR rec[-2]\n",
 "repeat_fb_inside": "R 0 1 2 3\nREPEAT 2 {\nH 2\nM 2\nCX rec[-1] 1\nH 0\nX_ERROR(0.1) 0 1\nH 0\nR 2\n}\nM 0 1\nDETECTOR rec[-1] rec[-3] rec[-4]\nDETECTOR rec[-2]\n",
 # zero-probability noise (transparent to frames) mixed in
 "zero_noise": "R 0 1 2\nH 2\nM 2\nCZ rec[-1] 1\nH 0\nX_ERROR(0) 0 1\nX_ERROR(0.1) 0 1\nH 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n",
 # OUTPUT_QUBITS frame with a split
 "output_split": "R 0 1 2\nH 2\nM 2\nCX rec[-1] 1\nH 0\nZ_ERROR(0.1) 0 1\nH 0\nM 0\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n",
}


_ONEQ = ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1")
_TWOQ = ("DEPOLARIZE2", "PAULI_CHANNEL_2")


def _twin(t):
    import re
    out = []
    for line in t.splitlines():
        m = re.match(r"^(\w+)\(([^)]*)\)\s+(.*)$", line)
        if m and m.group(1) in _ONEQ + _TWOQ:
            qs = m.group(3).split()
            k = 2 if m.group(1) in _TWOQ else 1
            out += [f"{m.group(1)}({m.group(2)}) {' '.join(qs[i:i + k])}" for i in range(0, len(qs), k)]
        else:
            out.append(line)
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("name", sorted(SPLIT_CASES))
def test_split_noise_full_equals_twin(name):
    t = SPLIT_CASES[name]
    assert _twin(t) != t
    a, b = export(t, 0.0), export(_twin(t), 0.0)
    assert a["ok"] and b["ok"], (a["error"], b["error"])
    D = sum(1 for line in t.splitlines() if line.startswith("DETECTOR"))
    ma, mb = E.dem_mechs(a["dem"], D), E.dem_mechs(b["dem"], D)
    for k in set(ma) | set(mb):
        assert abs(ma.get(k, 0.0) - mb.get(k, 0.0)) <= 1e-12, (sorted(k), ma.get(k), mb.get(k))


def test_split_noise_twins_exact():
    ran = 0
    for name, t in sorted(SPLIT_CASES.items()):
        if "OUTPUT_QUBITS" in t:
            continue                                   # exact_sv has no frame columns
        try:
            E.check_circuit(_twin(t), export)
        except AssertionError as e:
            if "non-deterministic" in str(e):
                continue
            raise AssertionError((name, str(e)[:300]))
        except Exception:                              # outside exact_sv's syntax (e.g. REPEAT, SQRT_X)
            continue
        ran += 1
    assert ran >= 8, ran
