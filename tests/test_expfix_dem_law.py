"""EXP-fix E1b (owner design 2026-10-21, docs/twirl_pauli_condition.tex "Noise model"): each fault of a channel
instance is classified (Pauli / uniform coset / general / list law) and solved per instance — the owner's closed
form for a single coset fault, one FWHT for a joint instance — with the owner's support cap (v > 12 refused by
name: "the Clifford error this fault leaves triggers too many detectors").

Fixture family `fanout(n)`: one X_ERROR on a control fanned out by CX onto n |+> read wires between T and T_DAG; the
fault leaves an odd S-power on every read, so the n reads are independent fair coins — a uniform coset of
dimension v = n with c' in W (Case A), or c' not in W (Case B) when the control is also read.

Gates here: the census routes and dims; the closed-form values; the cap (MX-CAP, MX-CAPL, MX-MSG, MX-ORDER);
G-DEM-SAMPLE (the DEM-implied detector distribution vs the twirl sampler's, G-test at 1e6 shots)."""
import math
import re

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from xtim._exact_scope import explicit_exact


def fanout(n, p=0.02, ctrl_read=False, channel="X_ERROR", pe_frame=False):
    t = " ".join(str(i) for i in range(1, n + 1))
    s = f"R 0\n{channel}({p}) 0\nRX {t}\nT {t}\n"
    s += "".join(f"CX 0 {i}\n" for i in range(1, n + 1))
    s += f"T_DAG {t}\nMX {t}\n"
    s += "".join(f"DETECTOR rec[-{k}]\n" for k in range(n, 0, -1))
    if ctrl_read:
        s += "M 0\nDETECTOR rec[-1]\n"
    if pe_frame:
        s += "R 40\nPAULI_EXPECTATION(0) Z40 rec[-1]\n"
    return s


def census(text, **kw):
    with explicit_exact():                     # the DEM export is an explicit exact entry point (J-b1 P13)
        return TA.dem_law_census(text, **kw)


def _mechs(dem_text):
    out = {}
    for line in dem_text.splitlines():
        m = re.match(r"error\(([^)]*)\)\s+(.*)$", line.strip())
        if m:
            out[tuple(sorted(m.group(2).split()))] = float(m.group(1))
    return out


@pytest.mark.parametrize("n", [1, 3, 10, 12])
def test_case_a_closed_form(n):
    p = 0.02
    r = census(fanout(n, p))
    assert r["ok"], r["error"]
    (inst,) = r["instances"]
    assert inst["route"] == "coset_closed" and inst["status"] == "ok"
    (comp,) = inst["comps"]
    assert comp["kind"] == "C" and comp["v"] == n and comp["v0"] == n
    q = -math.expm1(math.log1p(-p) * 2.0 ** (1 - n)) / 2
    mech = _mechs(r["dem"])
    assert len(mech) == 2 ** n - 1
    assert all(abs(v - q) <= 1e-15 * q for v in mech.values())
    if n == 1:
        assert abs(q - p / 2) < 1e-17


def test_case_b_needs_approximation_and_disjoint_fallback():
    p = 0.02
    r = census(fanout(2, p, ctrl_read=True))
    assert not r["ok"] and "requires the `approximate_disjoint_errors` option" in r["error"]
    (inst,) = r["instances"]
    assert inst["route"] == "coset_closed" and inst["status"] == "negative"
    r = census(fanout(2, p, ctrl_read=True), approximate_disjoint_errors=1.0)
    assert r["ok"], r["error"]
    assert r["instances"][0]["status"] == "approx_disjoint"
    mech = _mechs(r["dem"])
    assert len(mech) == 4 and all(abs(v - p / 4) < 1e-18 for v in mech.values())   # c' + W, p / 2^v each


def test_support_cap_refuses_v13_by_name_before_any_born_sum():
    for kw in ({}, {"approximate_disjoint_errors": 1.0}):
        r = census(fanout(13), **kw)
        assert not r["ok"]
        e = r["error"]
        assert "X_ERROR at deferred stream index" in e
        assert re.search(r"\(alternative X on deferred qubit \d+; user noise site 0\)", e), e
        assert "the Clifford error this fault leaves triggers too many detectors" in e
        assert "randomizes 13 detector(s) (and 0 other DEM target(s))" in e
        assert "support of dimension v = 13 > 12" in e and "2^v - 1 = 8191 hyperedges" in e
    with pytest.raises(xtim.XtimDemError, match="triggers too many detectors"):
        xtim.Circuit(fanout(13)).detector_error_model_text()
    # MX-ORDER: the v0 cap precedes every Born sum — a v0 = 11 fault with a nonzero Fourier support (wdim = 1: one
    # read CS-coupled to a T qubit) refuses with ZERO Born sums evaluated.
    t = fanout(13).replace("RX ", "RX 50 51\nT 51\nCS 50 51\nRX ", 1).replace("T_DAG", "CX 0 50\nCS_DAG 50 51\nT_DAG", 1)
    t += "MX 50\nDETECTOR rec[-1]\n"
    r = census(t)
    assert not r["ok"] and "randomizes 14 detector(s)" in r["error"] and "v = 13 > 12" in r["error"], r["error"]
    assert r["born_sums"] == 0



def test_support_cap_list_path_precheck():
    # A PAULI_EXPECTATION whose declared frame is a randomized read: a per-shot random column sign (the LIST law).
    r = census(fanout(4, pe_frame=True))
    assert r["ok"], r["error"]
    assert [c["kind"] for c in r["instances"][0]["comps"]] == ["L"]
    for n in (13, 14):                                   # the walk alone would hit its 4096-leaf cap (MX-CAPL)
        r = census(fanout(n, pe_frame=True))
        assert not r["ok"] and f"support of dimension v = {n} > 12" in r["error"], r["error"]


def test_all_case_a_mixture_routes_joint():
    r = census(fanout(3, p=0.03, channel="DEPOLARIZE1"))
    assert r["ok"], r["error"]
    (inst,) = r["instances"]
    assert inst["route"] == "joint_fwht" and inst["status"] == "ok" and inst["rank"] == 3
    assert sorted(c["kind"] for c in inst["comps"]) == ["C", "C"]
    # X and Y leave the same uniform law on W: one coset fault at 2p/3, closed form
    q = -math.expm1(math.log1p(-0.02) * 2.0 ** (1 - 3)) / 2
    assert all(abs(v - q) <= 1e-12 * q for v in _mechs(r["dem"]).values())


def test_pauli_circuits_route_the_old_body():
    text = "R 0 1\nDEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
    r = census(text)
    assert r["ok"] and {i["route"] for i in r["instances"]} == {"pauli_old"}


# ── G-DEM-SAMPLE: the DEM-implied detector distribution vs the twirl sampler (independent) ───────────────────────
def _dem_distribution(dem_text, D):
    P = np.zeros(1 << D)
    P[0] = 1.0
    for sig, q in _mechs(dem_text).items():
        mask = 0
        for t in sig:
            assert t[0] == "D"
            mask |= 1 << int(t[1:])
        idx = np.arange(1 << D)
        P = (1 - q) * P + q * P[idx ^ mask]
    return P


@pytest.mark.parametrize("text", [fanout(3), fanout(4, p=0.02), fanout(3, p=0.03, channel="DEPOLARIZE1")],
                         ids=["caseA_v3", "caseA_v4", "mixture_v3"])
def test_sample_matches_dem(text):
    from scipy.stats import chi2
    c = xtim.Circuit(text)
    D = c.num_detectors
    P = _dem_distribution(c.detector_error_model_text(), D)
    dets = c.compile_detector_sampler(seed=20261021).sample(1_000_000)
    idx = (dets.astype(np.int64) << np.arange(D)).sum(axis=1)
    obs = np.bincount(idx, minlength=1 << D).astype(float)
    exp = P * len(idx)
    nz = obs > 0
    G = 2.0 * float((obs[nz] * np.log(obs[nz] / exp[nz])).sum())
    pval = chi2.sf(G, (1 << D) - 1)
    assert pval > 1e-4, (G, pval)


# ── R1 re-detection: a law the character test misses is still a coset (review M1's T-type <Z_f> = 0 case) ──────────
_TZ0 = "RX 0\nT 0\nRX 1\nCS 1 0\nX_ERROR(0.01) 1\nCS_DAG 1 0\nMX 1\nDETECTOR rec[-1]\n"


def test_redetected_coset_routes_closed_form():
    r = census(_TZ0)
    assert r["ok"], r["error"]
    (inst,) = r["instances"]
    assert inst["route"] == "coset_closed"
    (comp,) = inst["comps"]
    assert comp["kind"] == "R" and comp["wdim"] == 1 and comp["v0"] == 0 and comp["v"] == 1
    assert abs(_mechs(r["dem"])[("D0",)] - 0.005) < 1e-17           # S on an X-read: p/2


# ── the joint rank cap (exact mode) and its approximate fallback (MX-APX20) ──────────────────────────────────────────
def two_block(nA=10, nB=10, p=0.01):
    """One PAULI_CHANNEL_2 instance on controls (0, 99) with only XI and IX: X_0 fans out onto block A, X_99 onto
    block B (v = 10 each); M 0 reads X_0 too (a deterministic c'), so the joint signature rank is 10 + 10 + 1 = 21."""
    A = list(range(1, nA + 1))
    B = list(range(nA + 1, nA + nB + 1))
    ab = " ".join(map(str, A + B))
    args = ", ".join([str(p), "0", "0", str(p)] + ["0"] * 11)
    s = f"R 0 99\nPAULI_CHANNEL_2({args}) 0 99\nRX {ab}\nT {ab}\n"
    s += "".join(f"CX 0 {i}\n" for i in A) + "".join(f"CX 99 {j}\n" for j in B)
    s += f"T_DAG {ab}\nMX {ab}\n" + "".join(f"DETECTOR rec[-{k}]\n" for k in range(nA + nB, 0, -1))
    s += "M 0\nDETECTOR rec[-1]\n"
    return s


def test_joint_rank_over_20_refused_in_exact_mode_and_disjoint_in_approx():
    r = census(two_block())
    assert not r["ok"]
    assert "the alternatives' joint signature group has rank m = 21 > 20 (component supports v = [10, 10])" in r["error"]
    assert r["instances"][-1]["route"] == "joint_fwht" and r["instances"][-1]["status"] == "rank"
    r = census(two_block(), approximate_disjoint_errors=1.0)
    assert r["ok"], r["error"]
    inst = [i for i in r["instances"] if i["status"] == "approx_disjoint"]
    assert len(inst) == 1 and inst[0]["rank"] == 21
    mech = _mechs(r["dem"])
    assert len(mech) == 2 * 1024 - 1                                  # c'_X + W_A (1024) and W_B \ {0} (1023)


def test_small_joint_instance_routes_the_naive_solve():
    # X and Y on the fanned-out control leave the same v = 1 coset law: the joint rank bound sum(v_i + 1) = 4, so the
    # instance takes the naive G <= 16 solve on the merged signatures (as before EXP-fix), with the closed-form value.
    r = census(fanout(1, p=0.03, channel="DEPOLARIZE1"))
    assert r["ok"], r["error"]
    (inst,) = r["instances"]
    assert inst["route"] == "joint_small" and inst["status"] == "ok"
    assert abs(_mechs(r["dem"])[("D0",)] - 0.01) < 1e-15            # (X + Y) at 2p/3 = 0.02 on a v = 1 coset: q = 0.01
