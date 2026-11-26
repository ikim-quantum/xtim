"""CAPS-1: storage-width limits that refused or were silently wrong (CAPS audit items A1, A2, A3).

A2  per_shot_accepted_violation (DEM export) kept the PK_RAND / PK_MAG columns' coin bits of one fault alternative in
    ONE uint64: `1ULL << q` with q >= 64 aliased column q onto column q mod 64, so the accepted-branch judgement of a
    column read another column's coin. Silent: with 128 columns the include_expectations=False finding named a column
    the fault never harms (and missed the harmed one).
A1  PeShotAnalyzer::structure kept its masks over the fault's changed reads J in one uint64 and refused nJ > 63
    ("a single fault changes 72 (> 63) terminal reads ..."): now multiword, the same lowest-bit pivots.
A3  the sampler preset refused classically-controlled Pauli feedback with > 64 PAULI_EXPECTATION records (a 64-bit sign
    mask; the refusal surfaced as "outside the simulable class"); its relabel cut r < 64 and skipped the whole sign block
    when the first sign word was 0. The framed shot engine's zero-event fast path used sign 0 for every column r >= 64
    (eval_channel_shot_ already read the fired nodes' masks) — silently wrong exact-engine columns past 63.

Every expected value is derived from an independent route: the dense statevector oracle (_forced_oracle) on the small
instance, the exact engine's per-shot sampler, the recorded feedback bit, or the same circuit's own first columns
(duplicated columns are equal).
"""
import re

import numpy as np
import pytest

import xtim
import _exact  # noqa: E402  (raw exact DEM export: the explicit-exact scope)
from _exact_engine import compile_exact_branches, exact_detector_sampler
from _forced_oracle import branches as dense_branches


def export(text, incl=True):
    # clifford_cutoff = inf: the exact split
    return _exact.export_dem_text(text, incl, False, False, False, [], [], 0.0, -1)


def with_columns(body, cols):
    return body + "".join(f"PAULI_EXPECTATION({i}) {c}\n" for i, c in enumerate(cols))


# ------------------------------------------------------------------------------------------------------------------
# A2: > 64 PK_RAND / PK_MAG columns in one alternative
# ------------------------------------------------------------------------------------------------------------------
# One correlated fault X0 X1, each between T and T_DAG (S-type residuals), fanned into the magic wire 3, the |0>-H wire 4
# and the detector wire 2: the alternative randomizes D0 and is undetectable on part of the accepted branch.
A2_BODY = ("RX 0 1 2\nRY 3\nT 3\nT 3\nS 3\nR 4\nH 4\nT 0\nT 1\nE(0.1) X0 X1\nT_DAG 0\nT_DAG 1\nCX 3 0\nCX 2 1\nCX 4 1\n"
           "MY 0\nM 1\nMX 2\nDETECTOR rec[-1]\n")
A2_COLS = ["X4", "Z3", "X3"]


def _harmed_on_accepted(body, cols):
    """Dense oracle: the columns whose per-shot value differs from the noiseless one on some accepted (no detector
    fired) branch of the forced alternative."""
    text = with_columns(body, cols)
    n = xtim.Circuit(text).num_qubits
    nb = list(dense_branches(text, n, {}).values())
    c0 = [float(v) for v in nb[0]["exp"]]
    assert all(np.allclose([float(v) for v in b["exp"]], c0, atol=1e-12) and b["det"] == nb[0]["det"] for b in nb)
    harmed, accepted = set(), 0.0
    for b in dense_branches(text, n, {(0, 0): "XX"}).values():
        if b["p"] <= 1e-12 or b["det"] != nb[0]["det"]:
            continue
        accepted += b["p"]
        harmed |= {cols[r] for r in range(len(cols)) if abs(float(b["exp"][r]) - c0[r]) > 1e-9}
    assert accepted > 0.1
    return harmed


def _named(res, cols):
    """The columns the include_expectations=False report names as harmed on the accepted branch."""
    out = set()
    for _, why in res["dropped_expectation_columns"]:
        if "accepted branch" in why:
            out.add(cols[int(re.search(r"logical expectation column L(\d+)", why).group(1))])
    return out


def test_a2_oracle_names_the_harmed_column():
    harmed = _harmed_on_accepted(A2_BODY, A2_COLS)
    assert harmed == {"Z3"}                      # X4 stays +1 on the accepted branch; Z3 goes 0 -> +-1
    r = export(with_columns(A2_BODY, A2_COLS), False)
    assert r["ok"], r["error"]
    assert _named(r, A2_COLS) == harmed


@pytest.mark.parametrize("arrangement", ["cycled_twice", "harmless_first_64"])
def test_a2_more_than_64_columns_judged_per_column(arrangement):
    """128 / 65+ columns in one alternative: every named column is one the oracle finds harmed (duplicated columns are
    the same column), and the harmed one is named. Before CAPS-1 the 128-column export named only X4."""
    harmed = _harmed_on_accepted(A2_BODY, A2_COLS)
    if arrangement == "cycled_twice":
        cyc = [A2_COLS[i % len(A2_COLS)] for i in range(64)]
        cols = cyc + cyc
    else:
        cols = ["X4"] * 64 + ["X3", "Z3"]
    r = export(with_columns(A2_BODY, cols), False)
    assert r["ok"], r["error"]
    assert _named(r, cols) == harmed, (_named(r, cols), harmed)
    r1 = export(with_columns(A2_BODY, cols), True)
    assert not r1["ok"] and "accepted branch" in r1["error"], r1["error"]


# ------------------------------------------------------------------------------------------------------------------
# A1: a fault whose residual randomizes > 63 terminal reads feeding an expectation column
# ------------------------------------------------------------------------------------------------------------------
def fan(n, p=0.1):
    """X fault between T and T_DAG on wire 0 (S-type residual), fanned (CX e 0) into the magic column wire 1 and n
    detector wires: the fault flips all n detectors and the column's sign together, with probability 1/2."""
    es = list(range(2, 2 + n))
    t = f"RX 0 1 {' '.join(map(str, es))}\nT 1\nT 0\nX_ERROR({p}) 0\nT_DAG 0\nCX 1 0\n"
    t += "".join(f"CX {e} 0\n" for e in es) + "M 0\n" + "".join(f"MX {e}\n" for e in es)
    t += "".join(f"DETECTOR rec[-{k}]\n" for k in range(1, n + 1)) + "PAULI_EXPECTATION(0) X1\n"
    return t


def _mechs(dem):
    out = {}
    for line in dem.splitlines():
        if line.startswith("error"):
            p = float(line[line.index("(") + 1:line.index(")")])
            out[frozenset(line.split(") ", 1)[1].split())] = p
    return out


def test_a1_fan_law_small_instance_matches_the_dense_oracle():
    n = 4
    text = fan(n)
    br = list(dense_branches(text, xtim.Circuit(text).num_qubits, {(0, 0): "X"}).values())
    c0 = 1 / np.sqrt(2)
    law = {}
    for b in br:
        key = (tuple(b["det"]), round(float(b["exp"][0]) / c0))
        law[key] = law.get(key, 0.0) + b["p"]
    assert law == pytest.approx({((0,) * n, 1): 0.5, ((1,) * n, -1): 0.5})
    r = export(text)
    assert r["ok"], r["error"]
    assert _mechs(r["dem"]) == pytest.approx({frozenset([f"D{d}" for d in range(n)] + ["L0"]): 0.05})


@pytest.mark.parametrize("n", [64, 72])
def test_a1_more_than_63_changed_reads_served(n):
    """Refused before CAPS-1 ("a single fault changes 72 (> 63) terminal reads"); now the same law as at n <= 63, and
    the exact engine's shot sampler (an independent route: per-shot expectations, no PeShotAnalyzer) samples exactly
    that joint law of (detectors, column sign)."""
    text = fan(n)
    r = export(text)
    assert r["ok"], r["error"]
    want = {frozenset([f"D{d}" for d in range(n)] + ["L0"]): 0.05}
    assert _mechs(r["dem"]) == pytest.approx(want)
    r0 = export(text, False)
    assert r0["ok"] and r0["dropped_expectation_columns"] == [], r0["dropped_expectation_columns"]
    shots = 20000
    dets, ex = exact_detector_sampler(xtim.Circuit(text), seed=11).sample(shots, return_expectations=True)
    dets = np.asarray(dets, dtype=np.uint8)
    flip = np.asarray(ex)[:, 0] < 0
    nfired = dets.sum(axis=1)
    assert set(np.unique(nfired)) <= {0, n}
    assert np.array_equal(nfired == n, flip)              # all detectors and the column sign flip together
    assert np.allclose(np.abs(np.asarray(ex)[:, 0]), 1 / np.sqrt(2))
    frac = float(flip.mean())
    assert abs(frac - 0.05) < 5 * np.sqrt(0.05 * 0.95 / shots), frac


# ------------------------------------------------------------------------------------------------------------------
# A3: > 64 PAULI_EXPECTATION records with classical Pauli feedback; the exact engine's columns >= 64
# ------------------------------------------------------------------------------------------------------------------
# a fair record m (DECISION 0); CZ rec[-1] 1 applies Z1 iff m = 1: <X1> = (-1)^m / sqrt2 on the magic wire, <X2> = 1
# (commutes with the feedback). Layouts: every column X1; or 64 X2 columns first, so the first sign word is 0.
FB_HEAD = "RX 0 1 2\nT 1\nM 0\nDECISION(0) rec[-1]\nCZ rec[-1] 1\n"
FB_LAYOUTS = {
    "x1_64": ["X1"] * 64,
    "x1_65": ["X1"] * 65,
    "x1_130": ["X1"] * 130,
    "x2_64_then_x1": ["X2"] * 64 + ["X1"] * 3,
    "mixed_130": (["X2"] * 64 + ["X1", "X2"] * 33),
}


def _fb_want(cols, m):
    m = np.asarray(m, dtype=np.int64)
    return np.stack([(1 - 2 * m) / np.sqrt(2) if c == "X1" else np.ones(len(m)) for c in cols], axis=1)


@pytest.mark.parametrize("layout", sorted(FB_LAYOUTS))
def test_a3_feedback_with_more_than_64_expectations_sampled(layout):
    cols = FB_LAYOUTS[layout]
    shots = 400
    _, ex, meas = exact_detector_sampler(xtim.Circuit(with_columns(FB_HEAD, cols)), seed=7).sample(
        shots, return_expectations=True, return_measurements=True)
    ex, m = np.asarray(ex), np.asarray(meas, dtype=np.int64)[:, 0]
    assert ex.shape == (shots, len(cols)) and 0 < m.sum() < shots
    bad = [r for r in range(len(cols)) if not np.allclose(ex[:, r], _fb_want(cols, m)[:, r], atol=1e-12)]
    assert bad == []


@pytest.mark.parametrize("layout", sorted(FB_LAYOUTS))
def test_a3_feedback_with_more_than_64_expectations_exact_law(layout):
    """The exact law (compiled linear forms): each row's conditional expectations given its DECISION bit."""
    cols = FB_LAYOUTS[layout]
    law = compile_exact_branches(xtim.Circuit(with_columns(FB_HEAD, cols))).exact_branches([])
    d = np.asarray(law.decisions, dtype=np.int64)[:, 0]
    assert sorted(d.tolist()) == [0, 1]
    assert np.allclose(np.asarray(law.probabilities), 0.5)
    ex = np.asarray(law.expectations)
    bad = [r for r in range(len(cols)) if not np.allclose(ex[:, r], _fb_want(cols, d)[:, r], atol=1e-12)]
    assert bad == []


def test_a3_exact_engine_columns_past_63_equal_their_twins():
    """miniature_oracle with its 2 columns cycled to 70: each column r equals column r mod 2 on every shot (the
    expectations are computed per shot, not drawn). Before CAPS-1 columns 64..69 read sign 0 on zero-event shots."""
    from pathlib import Path
    text = (Path(__file__).resolve().parent / "data" / "miniature_oracle.stim").read_text()
    lines = text.strip().splitlines()
    cols = [ln.split(")", 1)[1] for ln in lines if ln.startswith("PAULI_EXPECTATION")]
    body = "".join(ln + "\n" for ln in lines if not ln.startswith("PAULI_EXPECTATION"))
    cc = [cols[i % len(cols)] for i in range(70)]
    big = body + "".join(f"PAULI_EXPECTATION({i}){c}\n" for i, c in enumerate(cc))
    _, ex = exact_detector_sampler(xtim.Circuit(big), seed=5).sample(2000, return_expectations=True)
    ex = np.asarray(ex)
    bad = [r for r in range(70) if not np.allclose(ex[:, r], ex[:, r % len(cols)], atol=1e-12)]
    assert bad == []
