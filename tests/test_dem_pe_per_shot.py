"""DEM export judges every fault's action on a PAULI_EXPECTATION column PER SHOT (3.1.12).

The sampler reports a column per shot: <P> on the state conditioned on EVERY measurement outcome,
with the declared frame records fixing only the sign convention. Through 3.1.11 the DEM exporter
classified a fault's action on a column (sign flip -> L-column mechanism; magnitude change ->
refused or post-select flagged) from the READ-AVERAGED <D^dag P' D> on the bare state. Whenever a
read that is neither a detector nor a declared frame record is correlated with the column the two
disagree, in both directions (adversarial DEM audit, 2026-10-11):

  SW1  `RX 0; RX 1; T 1; Y_ERROR(0.1) 1; CS 1 0; MX 1; PAULI_EXPECTATION(0) X0`: per shot 0.5 ->
       +-0.7071 on every faulted shot (no detector) — exported as `logical_observable L0`, no
       mechanism, no flag (silently wrong).
  SW2  a zero column becoming +-1 per shot — exported as a sign flip `error(0.1) L0`.
  FR1  a zero column that stays 0 per shot — refused as non-fault-tolerant.
  FR2  a column whose per-shot sign is randomized 50/50 (|value| kept, no detector) — refused.
  FF1  a detector-correlated per-shot sign law — post-select flagged instead of exported.

Every expected verdict below is DERIVED inside the test from the exact per-shot law — the
independent dense statevector oracle (`_forced_oracle`, own numpy: every measurement branch,
conditioned on EVERY outcome exactly as the sampler reports a column) — never from hard-coded
numbers alone: per alternative, the per-branch (detector flips, observable
flips, expectation-column sign flips) law plus a magnitude-change marker; a magnitude change on a
branch that fires no detector must be refused as non-fault-tolerant; otherwise the exported
single-channel DEM must induce exactly the channel's law.
"""
import math
import os
import re

import numpy as np
import pytest

import xtim
from _forced_oracle import branches as dense_branches
import _exact  # noqa: E402  (J-b1 F: raw exact oracle calls inside the explicit-exact scope)

SW1 = """RX 0
RX 1
T 1
Y_ERROR(0.1) 1
CS 1 0
MX 1
PAULI_EXPECTATION(0) X0
"""
SW2 = """RX 0
T 0
X_ERROR(0.1) 0
T_DAG 0
R 1
CX 0 1
MX 0
PAULI_EXPECTATION(0) Y1
"""
SW2_FRAMED = SW2.replace("PAULI_EXPECTATION(0) Y1", "PAULI_EXPECTATION(0) Y1 rec[-1]")
FR1 = """R 0 1
Y_ERROR(0.1) 0
T 0
PAULI_EXPECTATION(0) X0
"""
FR2 = """RY 0
RX 1
T_DAG 1
DEPOLARIZE2(0.15) 1 0
SWAP 0 1
T 1
T 0
CXSWAP 1 0
SQRT_ZZ 1 0
T_DAG 0
T 0
M 1
PAULI_EXPECTATION(1) X0 rec[-1]
"""
FF1 = """RX 1
RY 2
CX 1 2
SQRT_X 2
DEPOLARIZE1(0.15) 1
T_DAG 1
CX 2 0
T_DAG 2
C_XYZ 0
H_YZ 0
CZ 1 0
MX 1
MY 2
DETECTOR rec[-2] rec[-1]
PAULI_EXPECTATION(0) X1 rec[-1]
"""
# A column whose noiseless per-shot SIGN is the outcome of an undeclared read (Bell pair, MX of
# one half): no DEM concept -> dropped, the reason naming the record; with the record declared as
# its frame the column is deterministic and the fault (|<X0>| 1 -> 1/sqrt2, no detector) is
# refused as non-fault-tolerant.
UNFRAMED = """RX 0
R 1
CX 0 1
T 0
X_ERROR(0.1) 0
T_DAG 0
MX 1
PAULI_EXPECTATION(0) X0
"""
FRAMED = UNFRAMED.replace("PAULI_EXPECTATION(0) X0", "PAULI_EXPECTATION(0) X0 rec[-1]")

PC2 = [a + b for a in "IXYZ" for b in "IXYZ"][1:]


def export(text, incl=True, approx=False):
    return _exact.export_dem_text(text, incl, False, False, False, [], [], 1.0 if approx else 0.0)


def _alternatives(site):
    ch, ps = site["channel"], site["probs"]
    if ch in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        return [(ps[0], ch[0])]
    if ch == "DEPOLARIZE1":
        return [(ps[0] / 3, P) for P in "XYZ"]
    if ch == "PAULI_CHANNEL_1":
        return list(zip(ps, "XYZ"))
    if ch == "DEPOLARIZE2":
        return [(ps[0] / 15, P) for P in PC2]
    return list(zip(ps, PC2))


def _per_shot(text, faults=None):
    """Dense per-branch law: [(p, raw det bits, raw obs bits, expectations)] (every read
    resolved, exactly the per-shot conditioning of the sampler)."""
    n = xtim.Circuit(text).num_qubits
    return [(o["p"], o["det"], o["obs"], o["exp"]) for o in dense_branches(text, n, faults or {}).values()]


def exact_channel_law(text):
    """The circuit's single noise channel instance -> (law {frozenset(targets): prob},
    harmful_undetected). Targets: ('D', d), ('L', o) for observables, ('E', r) for the sign flip of
    a column whose noiseless per-shot value is one nonzero constant (or ('Z', r) bookkeeping for a
    zero column that must stay 0). Built from the dense oracle's exact per-shot branches only;
    detectors / observables relative to the (deterministic) noiseless parities."""
    sim = xtim.Circuit(text).compile_exact_branches()
    sites = sim.noise_sites()
    assert len(sites) == 1 and len(sites[0]["qubits"]) in (1, 2)
    b0 = _per_shot(text)
    R = len(b0[0][3])
    refd, refo = b0[0][1], b0[0][2]
    assert all(b[1] == refd and b[2] == refo for b in b0), "test circuits are deterministic"
    c0 = []
    for r in range(R):
        v = [b[3][r] for b in b0]
        assert np.allclose(v, v[0], atol=1e-12), "test columns are per-shot deterministic"
        c0.append(float(v[0]))
    law, harmful = {}, False
    ptot = 0.0
    for p, P in _alternatives(sites[0]):
        if p <= 0:
            continue
        ptot += p
        for q, det, obs, ex in _per_shot(text, {(0, 0): P}):
            s = {("D", d) for d in range(len(det)) if det[d] != refd[d]}
            s |= {("L", o) for o in range(len(obs)) if obs[o] != refo[o]}
            mag = False
            for r in range(R):
                v = float(ex[r])
                if abs(c0[r]) < 1e-12:
                    mag |= abs(v) > 1e-9
                elif abs(abs(v) - abs(c0[r])) > 1e-9:
                    mag = True
                elif (v < 0) != (c0[r] < 0):
                    s.add(("E", r))
            if mag and not any(t[0] == "D" for t in s):
                harmful = True
            key = frozenset(s)
            law[key] = law.get(key, 0.0) + p * q
    law[frozenset()] = law.get(frozenset(), 0.0) + 1.0 - ptot
    return law, harmful, sim


def _num_obs(text):
    return len(_per_shot(text)[0][2])


def dem_law(dem_text, O):
    """Distribution of the XOR of the DEM's independent mechanisms; L(O+r) -> ('E', r)."""
    mechs = []
    for line in dem_text.splitlines():
        m = re.match(r"^error\(([^)]*)\)(.*)$", line.strip())
        if not m:
            continue
        s = set()
        for t in m.group(2).split():
            if t == "^":
                continue
            j = int(t[1:])
            x = ("D", j) if t[0] == "D" else (("L", j) if j < O else ("E", j - O))
            s ^= {x}
        mechs.append((frozenset(s), float(m.group(1))))
    dist = {frozenset(): 1.0}
    for s, q in mechs:
        nd = {}
        for k, p in dist.items():
            nd[k] = nd.get(k, 0.0) + p * (1 - q)
            nd[k ^ s] = nd.get(k ^ s, 0.0) + p * q
        dist = nd
    return dist


def assert_same_law(a, b, tol=1e-12):
    for k in set(a) | set(b):
        assert abs(a.get(k, 0.0) - b.get(k, 0.0)) <= tol, (sorted(k), a.get(k), b.get(k))


# --------------------------------------------------------------------------------------------
# the silent-wrong reproducers: now refused, because the exact per-shot law changes a magnitude
# on a branch that fires no detector
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", [("SW1", SW1), ("SW2", SW2), ("SW2_framed", SW2_FRAMED)])
def test_silent_wrong_repros_now_refused_per_their_exact_law(name, text):
    """With the column requested: refused as non-fault-tolerant. With include_expectations=False
    (owner decision 2026-10-12): exported — the detector/observable DEM induces exactly the
    oracle's law marginalised over the column — and the finding is reported, naming the column."""
    law, harmful, _ = exact_channel_law(text)
    assert harmful, f"{name}: the exact per-shot law must show an undetected magnitude change"
    O = _num_obs(text)
    for approx in (False, True):
        r = export(text, True, approx)
        assert not r["ok"], (name, approx, r["dem"])
        assert "non-fault-tolerant" in r["error"], r["error"]
        r0 = export(text, False, approx)
        assert r0["ok"] and not r0["postselect_faults"], (name, approx, r0["error"])
        assert_same_law(dem_law(r0["dem"], O), _marginal(law))
        findings = [w for i, w in r0["dropped_expectation_columns"] if i == 0]
        assert findings and all(w.startswith("non-fault-tolerant circuit (") and
                                f"logical expectation column L{O}" in w for w in findings), findings


def test_sw1_per_shot_values_are_what_the_refusal_says():
    # the per-shot values (not the read average 0.5) are +-1/sqrt2 on the faulted shots
    br = _per_shot(SW1, {(0, 0): "Y"})
    assert np.allclose([abs(b[3][0]) for b in br], 1 / math.sqrt(2))
    avg = float(sum(b[0] * b[3][0] for b in br))
    assert abs(avg - 0.5) < 1e-9          # the read-averaged value the 3.1.11 classifier saw


# --------------------------------------------------------------------------------------------
# the false refusals / false flags: exported, the DEM inducing exactly the per-shot law
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", [("FR1", FR1), ("FR2", FR2), ("FF1", FF1)])
def test_false_refusal_repros_now_exported_with_the_exact_law(name, text):
    law, harmful, sim = exact_channel_law(text)
    assert not harmful
    O = _num_obs(text)
    for incl in (True, False):
        r = export(text, incl, False)
        assert r["ok"], (name, r["error"])
        assert not r["postselect_faults"], (name, r["postselect_faults"])
        want = law if incl else _marginal(law)
        assert_same_law(dem_law(r["dem"], O), want)


def _marginal(law):
    out = {}
    for k, v in law.items():
        k2 = frozenset(t for t in k if t[0] != "E")
        out[k2] = out.get(k2, 0.0) + v
    return out


def test_fr2_sign_coin_law_is_nontrivial():
    # guard the guard: FR2's exact law really is a per-shot random sign (so the export is a real
    # sign mechanism, not an empty DEM)
    law, _, _ = exact_channel_law(FR2)
    assert law.get(frozenset({("E", 0)}), 0.0) > 0.01


def test_ff1_sign_is_detector_correlated():
    law, _, _ = exact_channel_law(FF1)
    assert law.get(frozenset({("D", 0), ("E", 0)}), 0.0) > 0.01


# --------------------------------------------------------------------------------------------
# a column whose NOISELESS per-shot value varies has no DEM concept (owner rule): it is dropped
# (no L line, no mechanism, no flag, no refusal on its account), reported with the reason
# --------------------------------------------------------------------------------------------
# the audit's FR4 / SW4 reproducers (their L1 / L2 columns' sign is an undeclared record)
FR4 = """RX 0
RX 1
DEPOLARIZE2(0.15) 1 0
T 0
PAULI_CHANNEL_2(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.0,0.0,0.0,0.0,0.0,0.0,0.0) 0 1
S_DAG 0
M(0.1) !1
DEPOLARIZE2(0.2) 1 0
CY rec[-1] 0
CZ 0 1
CY 1 0
T_DAG 1
M 0
MY(0.2) 0
T_DAG 1
M 0
M 0
MY 1
DETECTOR rec[-3] rec[-2]
DETECTOR rec[-3] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2]
PAULI_EXPECTATION(0) Y1 rec[-5]
"""
SW4 = """RX 0
RX 1
R 2
T_DAG 1
MX(0.01) 1
T_DAG 1
M(0.2) !0
MX(0.1) 1
DEPOLARIZE2(0.2) 2 1
CZ rec[-2] 1
T 1
MRX(0.1) !1
T 1
CZ 0 1
T 2
X 2
SWAP 1 0
MX(0.15) 0
MX 1
M 2
DETECTOR rec[-1]
DETECTOR rec[-1]
DETECTOR rec[-1]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
PAULI_EXPECTATION(0) Z1*Y0 rec[-3]
PAULI_EXPECTATION(1) X1*X0 rec[-6]
"""


def _noiseless_varies(text):
    b0 = _per_shot(text)
    return [not np.allclose([b[3][r] for b in b0], b0[0][3][r], atol=1e-12)
            for r in range(len(b0[0][3]))]


@pytest.mark.parametrize("name,text", [("UNFRAMED", UNFRAMED), ("FR4", FR4), ("SW4", SW4)])
def test_varying_column_is_dropped_and_reported(name, text):
    varies = _noiseless_varies(text)          # derived from the exact per-shot oracle
    assert any(varies)
    O = _num_obs(text)
    r = export(text)
    assert r["ok"], (name, r["error"])
    assert [i for i, _ in r["dropped_expectation_columns"]] == [i for i, v in enumerate(varies) if v]
    assert not r["postselect_faults"]
    for i, why in r["dropped_expectation_columns"]:
        assert f"L{O + i}" in why
        assert re.search(rf"\bL{O + i}\b", r["dem"]) is None, r["dem"]     # no line, no mechanism
    # everything else is the D/L DEM (plus the kept columns)
    r0 = export(text, incl=False)
    assert r0["ok"]
    strip = lambda d: dem_law(d, 10 ** 6)
    keep_dl = {}
    for k, v in strip(r["dem"]).items():
        k2 = frozenset(t for t in k if t[0] == "D" or (t[0] == "L" and t[1] < O))
        keep_dl[k2] = keep_dl.get(k2, 0.0) + v
    assert_same_law(keep_dl, strip(r0["dem"]))


def test_unframed_reason_names_the_record():
    r = export(UNFRAMED)
    (i, why), = r["dropped_expectation_columns"]
    assert i == 0 and "declaring its frame as rec[-1] makes it deterministic" in why, why
    # declared: deterministic per shot, kept, and the fault's per-shot magnitude change is caught
    law, harmful, _ = exact_channel_law(FRAMED)
    assert harmful
    r2 = export(FRAMED)
    assert not r2["ok"] and "non-fault-tolerant" in r2["error"], r2
    assert r2["dropped_expectation_columns"] == []


def test_dropped_columns_on_python_api_and_warned_once():
    import warnings
    from xtim.errors import XtimDemWarning
    c = xtim.Circuit(SW4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c.detector_error_model_text()
        c.detector_error_model_text()
        res = c.detector_error_model_with_reject(include_expectations=True)
    got = [x for x in w if issubclass(x.category, XtimDemWarning)]
    assert len(got) == 1 and "L2" in str(got[0].message), [str(x.message) for x in w]
    assert [i for i, _ in res.dropped_expectation_columns] == [1]
    assert res.postselect_faults == []


# --------------------------------------------------------------------------------------------
# unchanged where the read-averaged and per-shot judgements agree
# --------------------------------------------------------------------------------------------
def test_pauli_only_columns_untouched():
    # Bell pair, Z fault before MX of the frame half: the 3.1.10 frame-record law, Pauli only
    text = """R 0 1
H 0
CX 0 1
Z_ERROR(0.2) 0
MX 0
PAULI_EXPECTATION(0) X1 rec[-1]
"""
    r = export(text)
    assert r["ok"] and r["dem"] == "error(0.20000000000000001) L0\n", r


# --------------------------------------------------------------------------------------------
# review of e9f228e (I1): a Pauli fault that FLIPS an undeclared read correlated with a column
# whose noiseless per-shot value varies. Those columns have no DEM concept: dropped, not exported
# by the operator rule (which was exact only under a record relabelling the DEM does not carry).
# --------------------------------------------------------------------------------------------
REVIEW_VARMAG = "RX 1\nRY 2\nCS 2 1\nX_ERROR(0.1) 1\nCZ 1 2\nMX 2\nPAULI_EXPECTATION(0) Z1\n"
REVIEW_VARSIGN = "RX 0\nR 1\nCX 0 1\nX_ERROR(0.1) 1\nM 1\nPAULI_EXPECTATION(0) Z0\n"
REVIEW_VARSIGN_FRAMED = REVIEW_VARSIGN.replace("Z0\n", "Z0 rec[-1]\n")


@pytest.mark.parametrize("text", [REVIEW_VARMAG, REVIEW_VARSIGN])
def test_review_pauli_read_flip_on_varying_column_is_dropped(text):
    assert _noiseless_varies(text) == [True]
    for incl in (True, False):
        r = export(text, incl)
        assert r["ok"] and [i for i, _ in r["dropped_expectation_columns"]] == [0], r
        assert "L0" not in r["dem"] and not r["postselect_faults"]


def test_review_framed_twin_keeps_the_column_with_its_exact_law():
    # with the record declared the column is deterministic per shot; the X fault is a pure sign
    assert _noiseless_varies(REVIEW_VARSIGN_FRAMED) == [False]
    law, harmful, sim = exact_channel_law(REVIEW_VARSIGN_FRAMED)
    assert not harmful
    r = export(REVIEW_VARSIGN_FRAMED)
    assert r["ok"] and r["dropped_expectation_columns"] == []
    assert_same_law(dem_law(r["dem"], 0), law)


def test_fast_path_pauli_read_flip_on_constant_column_is_exact():
    # the fast path (Pauli action, no randomized read) on a CONST column whose value depends on
    # a flipped read through its declared frame: exported law == exact per-shot law
    text = "RX 0\nT 0\nR 1\nCX 0 1\nZ_ERROR(0.2) 0\nX_ERROR(0.1) 1\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n"
    sim = xtim.Circuit(text).compile_exact_branches()
    for i, site in enumerate(sim.noise_sites()):
        one = text.replace("X_ERROR(0.1) 1\n", "") if i == 0 else text.replace("Z_ERROR(0.2) 0\n", "")
        law, harmful, _ = exact_channel_law(one)
        r = export(one)
        assert r["ok"] and not harmful
        assert_same_law(dem_law(r["dem"], 0), law)


# I3: include_expectations=False never refuses on column grounds where 3.1.11 exported
REVIEW_I3 = "RX 0 1 2\nX_ERROR(0.01) 0\nT 0 1\nCZ 0 2\nCZ 1 2\nMX 0 1\nPAULI_EXPECTATION(0) X2\n"


def test_review_i3_include_expectations_false_exports():
    r = export(REVIEW_I3, incl=False)
    assert r["ok"] and r["dem"] == "", r
    r = export(REVIEW_I3, incl=True)
    assert r["ok"], r


# I2: the exact analysis is budgeted; past the budget it refuses BY NAME and FAST
def _t_ladder(k):
    c = [f"RX {' '.join(map(str, range(k + 1)))}", "X_ERROR(0.01) 0",
         f"T {' '.join(map(str, range(k)))}"] + [f"CZ {i} {k}" for i in range(k)] + \
        [f"MX {' '.join(map(str, range(k)))}", f"PAULI_EXPECTATION(0) X{k}"]
    return "\n".join(c) + "\n"


@pytest.mark.parametrize("k", [13, 17])
def test_t_ladder_refuses_fast_by_name(k):
    import time
    t0 = time.time()
    r = export(_t_ladder(k))
    assert time.time() - t0 < 5.0
    assert not r["ok"] and "exact-analysis budget" in r["error"], r["error"]
    assert "raise pe_max_work" in r["error"], r["error"]   # the work cap binds; the message names it
    r0 = export(_t_ladder(k), incl=False)                 # include_expectations=False: reported, not refused
    assert r0["ok"] and [i for i, _ in r0["dropped_expectation_columns"]] == [0]
    assert "pe_max_work" in r0["dropped_expectation_columns"][0][1]


def _t_ladder_raised_budget_check(k, pe_max_work):
    """The default cap refuses the ladder by name (naming pe_max_work); a raised pe_max_work
    exports it, and the exported DEM induces exactly the fault's per-shot law (dense oracle). For
    odd k the column's per-shot value is identically 0 (the oracle certifies it, on every branch,
    faulted or not), so the DEM's `error(p) L0` sign flip of it is vacuous per shot (the operator
    E bit of a zero column, the pre-existing M1 class of the per-shot work); the laws are compared
    with that vacuous bit marginalised. The noiseless twin exports too."""
    text = _t_ladder(k)
    c = xtim.Circuit(text)
    with pytest.raises(xtim.XtimDemError, match="raise pe_max_work on detector_error_model"):
        c.detector_error_model_text()
    twin = xtim.Circuit(text.replace("X_ERROR(0.01) 0\n", ""))
    with pytest.raises(xtim.XtimDemError, match="raise pe_max_work"):
        twin.detector_error_model_text()
    law, harmful, _ = exact_channel_law(text)      # asserts the noiseless column is per-shot constant
    assert not harmful
    zero = all(abs(float(b[3][0])) < 1e-9 for b in _per_shot(text) + _per_shot(text, {(0, 0): "X"}))
    assert zero == (k % 2 == 1)
    dem = c.detector_error_model_text(pe_max_work=pe_max_work)
    assert_same_law(_marginal(dem_law(dem, 0)) if zero else dem_law(dem, 0),
                    _marginal(law) if zero else law)
    r = c.detector_error_model_with_reject(include_expectations=True, pe_max_work=pe_max_work)
    assert not r.postselect_faults and not r.dropped_expectation_columns
    assert twin.detector_error_model_text(pe_max_work=pe_max_work) == "logical_observable L0\n"
    return dem


def test_t_ladder_budget_names_argument_and_a_raised_budget_exports():
    # k=9: refused by name at the default cap (3.1.12), a complete walk in ~1 s at 1e9
    _t_ladder_raised_budget_check(9, 1e9)


@pytest.mark.skipif(not os.environ.get("XTIM_SLOW_TESTS"),
                    reason="k=13 needs about 1e11 work units (about an hour); XTIM_SLOW_TESTS=1 runs it")
def test_t_ladder_k13_exports_with_a_raised_budget():
    _t_ladder_raised_budget_check(13, 1e13)


def test_pe_max_work_is_validated():
    with pytest.raises(ValueError, match="pe_max_work must be a positive number"):
        xtim.Circuit(_t_ladder(9)).detector_error_model_text(pe_max_work=0)
    with pytest.raises(ValueError, match="pe_max_work must be a positive number"):
        xtim.Circuit(_t_ladder(9)).detector_error_model_text(pe_max_work=float("nan"))


def test_pe_max_work_default_is_the_engine_constant_and_inf_lifts_it():
    # single source of the default: the C++ constant, exported to Python
    assert xtim.circuit.PE_MAX_WORK_DEFAULT == xtim._xtim.PE_MAX_WORK_DEFAULT == 1.5e7
    text = _t_ladder(9)
    c = xtim.Circuit(text)
    with pytest.raises(xtim.XtimDemError, match="15000000 work units = pe_max_work"):
        c.detector_error_model_text(pe_max_work=xtim.circuit.PE_MAX_WORK_DEFAULT)   # == omitted
    # float('inf') is the documented lift: same export as a large finite cap
    assert c.detector_error_model_text(pe_max_work=float("inf")) == c.detector_error_model_text(pe_max_work=1e9)
    # a Pauli-only circuit is untouched by the argument (byte-identical with any value)
    pauli = "R 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-1] rec[-2]\n"
    assert xtim.Circuit(pauli).detector_error_model_text(pe_max_work=1) == xtim.Circuit(pauli).detector_error_model_text()
