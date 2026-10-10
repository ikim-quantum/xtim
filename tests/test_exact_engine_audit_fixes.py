"""Exact-engine fixes for the 2026-10-09 sampling audit (fix/audit-exact), against an independent
dense density-matrix oracle (tests/_dense_oracle.py — no xtim code).

  B, H  noisy NON-DIAGONAL magic (CH / conjugated CS): a fired noise atom whose PPR residual has an
        axis with #Y ≡ 2,3 (mod 4) was compiled into the error tableau with the wrong rotation
        sense (ppr_residual.cpp clifford_tableau: the stored canonical phase fixes the
        representative, not the letter sign) — every read anticommuting with that axis flipped.
  G, D  noiseless multi-magic: the case-B collapse (framed_measure_anticommuting_general, used by
        batch_measure / the TreePlan) re-referenced the frame to survivor 0 but kept the branch
        labels relative to the projected σ=0 state — a relative Pauli frame error on the
        post-measurement state whenever survivor 0 was not the σ=0 branch: wrong record law (G)
        and wrong exact PAULI_EXPECTATION signs (D).
  C     exact return_expectations sign on a noisy conjugated-CS circuit — the B/H root cause.
  L1    engine="auto" raised "exact_ppr_shot: conjugated generator not Hermitian": the guard
        tested `phase & 1`, but a Hermitian i^{phase} X^x Z^z has phase ≡ #Y (mod 2).

Exact checks wherever the quantity is deterministic: impossible records (oracle probability 0)
and per-shot PAULI_EXPECTATION values given the shot's records (noise forced to probability 0/1
so the conditional state is fixed by the records). Record LAWS are statistical only where no
exact handle exists (finding G and its family), at |z| > 6 with ≥ 25σ power on the 3.1.9 defect.
The PAULI_EXPECTATION families measure in ascending wire order: finding A (record layout with a
PAULI_EXPECTATION and non-ascending measurements) is a separate front-end defect.
"""
from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("stim")   # the oracle takes Clifford unitaries from stim

import xtim  # noqa: E402
from _dense_oracle import Oracle  # noqa: E402


@pytest.fixture(autouse=True)
def _private_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(xtim, "cache_dir", str(tmp_path))


def _exact_sample(text, shots, seed=5):
    s = xtim.Circuit(text).compile_detector_sampler(seed=seed, engine="exact")
    _, exps, meas = s.sample(shots, return_expectations=True, return_measurements=True)
    return meas.astype(np.int64), exps


def _oracle(text):
    o = Oracle(text)
    o.run()
    return o


def exact_violations(text, shots=2000, seed=5):
    """Per-shot EXACT checks: every record has oracle probability > 0 and every
    PAULI_EXPECTATION column equals the oracle's conditional value given the record
    (frame sign applied). Returns a list of violation strings (empty = pass)."""
    meas, exps = _exact_sample(text, shots, seed)
    o = _oracle(text)
    law = o.record_dist()
    order = sorted(range(len(o.expectations)), key=lambda i: o.expectations[i][0])
    finals = [(o.expectations[i][2], o.final_expect(o.expectations[i][1])) for i in order]
    bad = []
    for r in range(shots):
        k = tuple(int(b) for b in meas[r])
        if law.get(k, 0.0) < 1e-12:
            bad.append(f"shot {r}: impossible record {k}")
            continue
        for col, (recs, fe) in enumerate(finals):
            tgt = (-1) ** (sum(k[i] for i in recs) & 1) * fe[k]
            if abs(exps[r, col] - tgt) > 1e-9:
                bad.append(f"shot {r}: record {k} PE[{col}] {exps[r, col]:+.6f} vs {tgt:+.6f}")
    return bad


def law_z(text, shots, seed=7):
    """Max |z| of the sampled record law vs the oracle (compile_sampler), and the
    impossible-record count."""
    s = xtim.Circuit(text).compile_sampler(seed=seed).sample(shots).astype(np.int64)
    law = _oracle(text).record_dist()
    c = Counter(tuple(int(b) for b in row) for row in s)
    imp = sum(v for k, v in c.items() if law.get(k, 0.0) < 1e-12)
    z = 0.0
    for k, p in law.items():
        if 1e-12 < p < 1 - 1e-12:
            z = max(z, abs(c.get(k, 0) / shots - p) / np.sqrt(p * (1 - p) / shots))
    return z, imp


# ── B / H / C: the PPR residual axis sign ─────────────────────────────────────────────────────────

AUDIT_B = "Z_ERROR(0.5) 1\nCH 0 1\nH 0\nS 0\nM 1\n"
AUDIT_H = "Y_ERROR(0.3) 1\nSQRT_YY 1 0\nCS 1 0\nSQRT_YY_DAG 1 0\nM 1\n"
AUDIT_C = ("X_ERROR(0.3) 1\nM 1\nCZ rec[-1] 0\nC_NXYZ 0\nXCY 1 0\nCS_DAG 0 1\n"
           "SQRT_YY_DAG 0 1\nH_NXY 0\nPAULI_EXPECTATION(1) Y1*X0\n")


@pytest.mark.parametrize("text", [
    AUDIT_B,                                                    # truth: M 1 is always 0
    "X_ERROR(1) 1\nCH 0 1\nH 0\nS 0\nM 1\n",                    # truth: always 1
    "X_ERROR(1) 1\nCH 0 1\nH 0\nS_DAG 0\nM 1\n",
    "Z_ERROR(1) 1\nCH 0 1\nH 0\nS 0\nM 1\n",
    "X_ERROR(1) 1\nCH 0 1\nH 0\nS 0\nM 1\nPAULI_EXPECTATION(0) Y0\n",
    "X_ERROR(1) 1\nCH 0 1\nH 0\nS 0\nM 1\nPAULI_EXPECTATION(0) X0\n",
], ids=["audit_B", "x_hs", "x_hsdg", "z_hs", "x_hs_peY", "x_hs_peX"])
def test_noisy_ch_records_and_expectations_exact(text):
    bad = exact_violations(text)
    assert not bad, f"{len(bad)} violations, first: {bad[:3]}\n{text}"


def test_audit_C_expectation_sign_exact():
    bad = exact_violations(AUDIT_C, shots=4000)
    assert not bad, f"{len(bad)} violations, first: {bad[:3]}"


def test_audit_H_record_law():
    # P(M=1) = 0.65 exactly; 3.1.9 sampled 0.347 (|z| ≈ 37 at 40k shots).
    z, imp = law_z(AUDIT_H, 40000)
    assert imp == 0 and z < 6, (z, imp)


_ONE = ["H", "S", "S_DAG", "SQRT_X", "SQRT_Y", "H_YZ", "X", "Z"]
_TWO = ["CX", "CZ", "CY", "SWAP"]


def _offdiag_noisy_circuit(rng: random.Random, n=3, length=12) -> str:
    """Forced (p=1) Pauli errors before ONE non-diagonal magic gate (CH or a Clifford-conjugated
    CS) on a Z-basis control, then a random Clifford tail: one terminal read + a
    PAULI_EXPECTATION on the other wires (state fixed by the record ⇒ exact per shot)."""
    out = []
    a, b = rng.sample(range(n), 2)
    if rng.random() < 0.5:
        out.append(f"X {a}")
    out.append(f"{rng.choice(_ONE)} {b}")
    for q in range(n):
        if rng.random() < 0.6:
            out.append(f"{rng.choice('XYZ')}_ERROR(1) {q}")
    if rng.random() < 0.5:
        out.append(f"CH {a} {b}")
    else:
        c = rng.choice(["SQRT_YY", "SQRT_XX", "SQRT_ZZ"])
        out += [f"{c} {a} {b}", f"CS {a} {b}", f"{c}_DAG {a} {b}"]
    for _ in range(length):
        if rng.random() < 0.7:
            out.append(f"{rng.choice(_ONE)} {rng.randrange(n)}")
        else:
            x, y = rng.sample(range(n), 2)
            out.append(f"{rng.choice(_TWO)} {x} {y}")
    m = rng.randrange(n)
    out.append(f"{rng.choice(['M', 'MX', 'MY'])} {m}")
    P = "*".join(f"{rng.choice('XYZ')}{q}" for q in range(n) if q != m)
    out.append(f"PAULI_EXPECTATION(0) {P}")
    return "\n".join(out) + "\n"


def test_offdiagonal_noise_family_exact():
    # 3.1.9: 13 of these 300 circuits sample impossible records or wrong exact expectations.
    rng = random.Random(3)
    accepted, failures = 0, []
    for i in range(300):
        text = _offdiag_noisy_circuit(rng)
        try:
            bad = exact_violations(text, shots=300)
        except xtim.XtimRejectError:
            continue
        accepted += 1
        if bad:
            failures.append((i, bad[0], text))
    assert accepted >= 250
    assert not failures, f"{len(failures)} failing circuits; first: {failures[0]}"


# ── G / D: the case-B collapse reference ──────────────────────────────────────────────────────────

AUDIT_G = ("RX 0\nRY 1\nH 2\nT 2\nMPP Z2*Z0\nRY 2\nCCZ 0 2 1\nCS 2 1\nMPP X1\nMPP X1\nMR 0\n"
           "MRY 1\nR 2\nMX 1\n")
AUDIT_D = ("RX 0\nMPP Y0*Y3\nSWAP 1 3\nRY 5\nT 1\nMX 4\nRX 4\nCCZ 4 0 5\nMX 0\nMY 4\n"
           "PAULI_EXPECTATION(1) X5\n")


def test_audit_G_record_law():
    # 3.1.9: cell 111000 sampled 0.016 vs 0.047 (|z| ≈ 37 at 40k shots).
    z, imp = law_z(AUDIT_G, 40000)
    assert imp == 0 and z < 6, (z, imp)


def test_audit_D_expectation_sign_exact():
    bad = exact_violations(AUDIT_D, shots=4000)
    assert not bad, f"{len(bad)} violations, first: {bad[:3]}"


def _multimagic_circuit(rng: random.Random, n=5, length=18) -> str:
    out = [f"{rng.choice(['R', 'RX', 'RY'])} {q}" for q in range(n)]
    for _ in range(length):
        r = rng.random()
        if r < 0.15:
            out.append(f"T {rng.randrange(n)}")
        elif r < 0.3:
            a, b, c = rng.sample(range(n), 3)
            out.append(f"CCZ {a} {b} {c}")
        elif r < 0.45:
            a, b = rng.sample(range(n), 2)
            out.append(f"{rng.choice(['CS', 'CS_DAG'])} {a} {b}")
        elif r < 0.6:
            a, b = rng.sample(range(n), 2)
            out.append(f"MPP {rng.choice('XYZ')}{a}*{rng.choice('XYZ')}{b}")
        elif r < 0.75:
            out.append(f"{rng.choice(['M', 'MX', 'MY'])} {rng.randrange(n)}")
        elif r < 0.85:
            out.append(f"{rng.choice(['R', 'RX', 'RY'])} {rng.randrange(n)}")
        else:
            a, b = rng.sample(range(n), 2)
            out.append(f"{rng.choice(['CX', 'CZ', 'SWAP'])} {a} {b}")
    return "\n".join(out) + "\n"


def test_noiseless_multimagic_family_record_law():
    # 3.1.9: 6 of these 300 circuits sample a wrong record law (one an impossible record).
    rng = random.Random(7)
    failures, checked = [], 0
    for i in range(300):
        text = _multimagic_circuit(rng)
        if not re.search(r"^M", text, re.M):
            continue
        try:
            z, imp = law_z(text, 6000, seed=5)
        except xtim.XtimRejectError:
            continue
        checked += 1
        if imp or z > 6:
            failures.append((i, z, imp, text))
    assert checked >= 250
    assert not failures, f"{len(failures)} failing circuits; first: {failures[0]}"


# ── L1: auto never errors on an accepted circuit ──────────────────────────────────────────────────

AUDIT_L1 = ("PAULI_CHANNEL_2(0,0,0,0,0,0.165,0.0713,0,0,0,0,0,0,0.0536,0) 2 1\nCCZ 0 1 2\nM 2\n"
            "MPP Z1*Z2*Z0\nDETECTOR rec[-1] rec[-2]\nCY 2 1\nCXSWAP 0 2\nSQRT_X 2\n"
            "SQRT_YY_DAG 0 1\nSQRT_X_DAG 2\n")


@pytest.mark.parametrize("engine", ["auto", "twirl"])
def test_audit_L1_no_error_and_exact_detector_law(engine):
    law = _oracle(AUDIT_L1).record_dist()
    p_det = sum(p for k, p in law.items() if k[0] ^ k[1])
    d = xtim.Circuit(AUDIT_L1).compile_detector_sampler(seed=3, engine=engine).sample(100000)
    z = abs(d.mean() - p_det) / np.sqrt(p_det * (1 - p_det) / d.shape[0])
    assert z < 6, (d.mean(), p_det)
