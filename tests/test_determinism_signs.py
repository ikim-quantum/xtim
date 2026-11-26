"""detector_determinism's signed parities are the RECORDED-bit convention (dem_export.hpp, DeterminismResult):
detector_parity[d] / observable_parity[o] = the noiseless E (-1)^(XOR of the column's records as the sampler records
them) — the user's `!`, the sign eliminate_hadamards folds into a read and classical feedback all included. Every
value here is checked against the exact sampler's raw measurement records (tests/_exact_engine.py; no statevector)."""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from xtim import _xtim
from _exact_engine import exact_sampler


def _cols(text):
    """[(kind, record indices)] of every DETECTOR (kind "D") and OBSERVABLE_INCLUDE (kind ("L", k)) of a flat or
    REPEAT text, in absolute record order (a REPEAT is unrolled by stim's flattener; text only, no simulation)."""
    if "REPEAT" in text:
        import stim
        flat = str(stim.Circuit(text).flattened())
    else:
        flat = text
    m, dets, obs = 0, [], {}
    for ln in flat.splitlines():
        t = ln.split()
        if not t:
            continue
        head = t[0].split("(")[0]
        if head == "DETECTOR":
            dets.append([m + int(x[4:-1]) for x in t[1:] if x.startswith("rec[")])
        elif head == "OBSERVABLE_INCLUDE":
            k = int(t[0][t[0].index("(") + 1:-1])
            obs.setdefault(k, []).extend(m + int(x[4:-1]) for x in t[1:] if x.startswith("rec["))
        elif head == "MPP":
            m += len(" ".join(t[1:]).replace(" * ", "*").split())
        elif head in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
            m += len(t) - 1
    return dets, obs


def _exact_values(text, shots):
    """Per-column E (-1)^(XOR of the recorded bits) from the exact sampler's raw records."""
    recs = np.asarray(exact_sampler(text, seed=11).sample(shots), dtype=np.uint8)
    dets, obs = _cols(text)
    val = lambda idx: 1.0 - 2.0 * (recs[:, idx].sum(axis=1) % 2).mean() if idx else 1.0
    return [val(i) for i in dets], {k: val(i) for k, i in obs.items()}


def _check(text, shots=64, tol=None):
    r = _xtim.detector_determinism(text)
    assert r["ok"], (text, r)
    ed, eo = _exact_values(text, shots)
    for d, (v, e) in enumerate(zip(r["detector_parity"], ed)):
        if d in r["deterministic_detectors"]:
            assert v == e, (text, "D", d, v, e)
        elif tol is not None:
            assert abs(v - e) < tol, (text, "D", d, v, e)
    for k, e in eo.items():
        v = r["observable_parity"][k]
        if k in r["deterministic_observables"]:
            assert v == e, (text, "L", k, v, e)
        elif tol is not None:
            assert abs(v - e) < tol, (text, "L", k, v, e)
    return r


# The six cases of the bug report, with their recorded-bit truth.
@pytest.mark.parametrize("text,want", [
    ("R 0\nX 0\nM 0\n", -1.0), ("R 0\nM !0\n", -1.0), ("R 0\nMPP !Z0\n", -1.0),
    ("R 0\nH 0\nS 0\nMY 0\n", 1.0), ("RX 0\nZ 0\nMX 0\n", -1.0), ("RX 0\nS_DAG 0\nMY 0\n", -1.0)])
def test_reported_cases(text, want):
    r = _check(text + "DETECTOR rec[-1]\n")
    assert r["detector_parity"] == [want]


_ONE = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "H_YZ", "C_XYZ"]


@pytest.mark.parametrize("reset", ["R", "RX", "RY"])
@pytest.mark.parametrize("read", ["M", "MX", "MY", "MR", "MRX", "MRY"])
@pytest.mark.parametrize("bang", ["", "!"])
def test_single_qubit_clifford_folds(reset, read, bang):
    """Every single-qubit Clifford word of length <= 2 between a reset and a read, with and without `!`."""
    n = 0
    for L in range(3):
        for seq in itertools.product(_ONE, repeat=L):
            text = f"{reset} 0\n" + "".join(f"{g} 0\n" for g in seq) + f"{read} {bang}0\nDETECTOR rec[-1]\n"
            r = _check(text)
            n += bool(r["deterministic_detectors"])
    assert n > 0


@pytest.mark.parametrize("text", [
    "R 0 1\nH 0\nCX 0 1\nMPP X0*X1 !Z0*Z1 Y0*Y1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "R 0 1\nH 0\nCX 0 1\nZ 1\nMPP !X0*X1 Z0*Z1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-1] rec[-2]\n",
    "RX 0 1 2\nS_DAG 0\nMPP Y0*X1*X2 !X0*X1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    "R 0\nX 0\nMR 0\nM 0\nMR !0\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
    # feedback: the recorded control bit (with its `!`) drives the target
    "R 0 1\nM !0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-1] rec[-2]\n",
    "RX 0\nR 1\nZ 0\nMX 0\nCX rec[-1] 1\nM !1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "R 0 1\nX 0\nM 0\nCZ rec[-1] 1\nRX 1\nZ 1\nMX 1\nDETECTOR rec[-1] rec[-2]\n",
    # REPEAT
    "R 0\nREPEAT 3 {\n    X 0\n    M !0\n    DETECTOR rec[-1]\n}\nOBSERVABLE_INCLUDE(1) rec[-1] rec[-2]\n",
    "RX 0\nREPEAT 2 {\n    S 0\n    MY 0\n    DETECTOR rec[-1]\n    S_DAG 0\n}\nMX !0\nDETECTOR rec[-1] rec[-2]\n",
    # observables, multi-record columns, a gauge column beside them
    "R 0 1 2\nH 0\nCX 0 1\nX 2\nM 0 1 !2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(2) rec[-1]\n",
])
def test_products_resets_feedback_repeat_observables(text):
    _check(text, shots=2000, tol=0.15)


def _random_circuit(rng):
    n = rng.randint(2, 4)
    out = [rng.choice(["R", "RX", "RY"]) + " " + " ".join(map(str, range(n)))]
    nrec = 0
    for _ in range(rng.randint(3, 10)):
        k = rng.random()
        q = rng.randrange(n)
        if k < 0.35:
            out.append(f"{rng.choice(_ONE)} {q}")
        elif k < 0.50:
            a, b = rng.sample(range(n), 2)
            out.append(f"{rng.choice(['CX', 'CZ', 'CY'])} {a} {b}")
        elif k < 0.55:
            out.append(f"{rng.choice(['T', 'T_DAG'])} {q}")
        elif k < 0.80:
            out.append(f"{rng.choice(['M', 'MX', 'MY', 'MR', 'MRX', 'MRY'])} {rng.choice(['', '!'])}{q}")
            nrec += 1
        elif k < 0.90:
            a, b = rng.sample(range(n), 2)
            out.append(f"MPP {rng.choice(['', '!'])}{rng.choice('XYZ')}{a}*{rng.choice('XYZ')}{b}")
            nrec += 1
        elif nrec:
            out.append(f"{rng.choice(['CX', 'CZ', 'CY'])} rec[-{rng.randint(1, nrec)}] {q}")
    out.append(f"{rng.choice(['M', 'MX', 'MY'])} " + " ".join(f"{rng.choice(['', '!'])}{q}" for q in range(n)))
    nrec += n
    for _ in range(rng.randint(1, 4)):
        recs = rng.sample(range(1, nrec + 1), rng.randint(1, min(3, nrec)))
        out.append("DETECTOR " + " ".join(f"rec[-{r}]" for r in recs))
    recs = rng.sample(range(1, nrec + 1), rng.randint(1, min(3, nrec)))
    out.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{r}]" for r in recs))
    return "\n".join(out) + "\n"


def test_fuzz_against_the_exact_sampler():
    """Random Clifford+T circuits with `!`, MPP, MR, feedback: every deterministic column's sign equals the exact
    sampler's recorded parity; every gauge column's value is its recorded average (2000 shots)."""
    rng = random.Random(20261119)
    det_cols = signed = checked = 0
    for _ in range(1500):
        text = _random_circuit(rng)
        r = _xtim.detector_determinism(text)
        if not r["ok"]:
            continue
        try:
            _check(text, shots=2000, tol=0.15)
        except Exception as e:   # a circuit the exact sampler refuses (class reject) is out of scope
            import xtim
            if isinstance(e, xtim.XtimError):
                continue
            raise
        checked += 1
        det_cols += len(r["deterministic_detectors"]) + len(r["deterministic_observables"])
        signed += sum(r["detector_parity"][d] < 0 for d in r["deterministic_detectors"])
    # measured at the fix: 1496 checked, 618 deterministic columns, 236 of them negative
    assert checked >= 1400 and det_cols >= 450 and signed >= 180, (checked, det_cols, signed)


@pytest.mark.parametrize("text,noiseless,want", [
    ("R 0\nM(0.3) !0\nDETECTOR rec[-1]\n", "R 0\nM !0\nDETECTOR rec[-1]\n", -1.0),
    ("R 0 1\nX 0\nMPP(0.2) !Z0*Z1\nDETECTOR rec[-1]\n", "R 0 1\nX 0\nMPP !Z0*Z1\nDETECTOR rec[-1]\n", 1.0),
    ("R 0 1\nMPP(0.2) !Z0*Z1\nDETECTOR rec[-1]\n", "R 0 1\nMPP !Z0*Z1\nDETECTOR rec[-1]\n", -1.0),
])
def test_readout_noise_is_excluded(text, noiseless, want):
    """The value is NOISELESS: a measurement-flip probability M(p) / MPP(p) is noise and is not in it (the `!` is).
    `want` is the exact sampler's recorded value on the circuit with the flip probability removed."""
    r = _xtim.detector_determinism(text)
    assert r["ok"] and r["deterministic_detectors"] == [0], r
    assert r["detector_parity"] == [want]
    assert _check(noiseless)["detector_parity"] == [want]
