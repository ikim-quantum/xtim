"""QW-E pins: the DEM export's determinism gate and readout-flip loop lost their O(n') / O(M) / O(T) per-target scans.

* D1/D2: check_deterministic reads the prefetched U^dag P U sparsely and builds the O(M) inclusion mask only when it
  refuses. The refusal messages (non-deterministic detector / observable, the first one named, the others listed) and
  the DEM text of deterministic circuits (constant parities, repeated records, Y reads, dropped gauge observables) are
  pinned to the pre-change build (2e057d3).
* D5: an M(p) readout flip's signature comes from the transposed read -> targets index instead of a scan over every
  target. A random-M(p) fuzz (classical-state circuits: every Z / X / Y read deterministic, random noise, random
  detectors and observables with repeated records, gauge observables dropped) pins the stim and xdem text digests.
* D4: the dense T x MW / NDEC x MW record masks became parity-reduced record rows (CSR) and their transposes; the
  randomized-read target sets, DECISION flips and readout flips read them. Pinned: the fuzz above with DECISION
  columns and a PAULI_EXPECTATION (include_expectations on and off; stim and xdem), and magic circuits whose faults
  randomize reads (correlated fair-coin reads feeding several detectors, with DECISION columns).
The pins were produced by the reference build of 2e057d3 (_xtim 7131eac9).
"""
import hashlib
import random

import pytest

import xtim


REFUSALS = {
    "nondet_detector": ("H 0\nM 0\nDETECTOR rec[-1]\n", {}),
    "nondet_second_of_three": ("R 0 1 2\nH 1\nX_ERROR(0.1) 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\n"
                               "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\n", {}),
    "nondet_observable": ("R 0 1\nH 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n", {}),
    "nondet_y_parity": ("R 0 1\nH 0\nCX 0 1\nS 0\nMY 0\nM 1\nDETECTOR rec[-1] rec[-2]\n", {}),
    "nondet_bell_half": ("R 0 1\nH 0\nCX 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-1]\n",
                         {}),
    "nondet_xdem": ("H 0\nM 0\nDETECTOR rec[-1]\n", {"format": "xdem"}),
    "nondet_magic": ("R 0\nH 0\nT 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n", {}),
    "nondet_magic_xdem": ("R 0\nH 0\nT 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n", {"format": "xdem"}),
    "nondet_magic_partial": ("R 0 1\nH 0\nT 0\nH 0\nX 0\nX_ERROR(0.1) 0 1\nM 1 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
                             {}),
    "nondet_magic_partial_xdem": ("R 0 1\nH 0\nT 0\nH 0\nX 0\nX_ERROR(0.1) 0 1\nM 1 0\nDETECTOR rec[-2]\n"
                                  "DETECTOR rec[-1]\n", {"format": "xdem"}),
    # S2-CLEAN (QW-E review m1): formerly DETERMINISTIC["y_parity"]; the S makes <YY> = 0, so it is a refusal
    "nondet_y_parity_noisy": ("R 0 1\nH 0\nCX 0 1\nS 0\nDEPOLARIZE1(0.01) 0 1\nMY 0\nMY 1\nDETECTOR rec[-1] rec[-2]\n",
                              {}),
}

DETERMINISTIC = {
    "constant_parity": "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1] rec[-1]\nDETECTOR rec[-1]\n",
    "bell_parity": "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nM(0.02) 0 1\nDETECTOR rec[-1] rec[-2]\n",
    # the Bell pair read in Y without the S: <YY> = -1, a deterministic Y-read parity (S2-CLEAN, QW-E review m1)
    "y_parity": "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE1(0.01) 0 1\nMY 0\nMY 1\nDETECTOR rec[-1] rec[-2]\n",
    "gauge_dropped": "R 0 1\nH 1\nX_ERROR(0.1) 0\nM(0.01) 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                     "OBSERVABLE_INCLUDE(1) rec[-2]\n",
}


def fuzz_circuit(rng):
    """A classical-basis circuit: every read deterministic; M(p) / MX(p) / MY(p) / MR(p) readout noise."""
    n = rng.randint(1, 7)
    lines, nrec, basis = [], 0, [rng.choice("ZXY") for _ in range(n)]
    for q in range(n):
        lines.append(f"R{'' if basis[q] == 'Z' else basis[q]} {q}")
    gauge = rng.random() < 0.3
    for _ in range(rng.randint(1, 4)):
        for q in range(n):
            if rng.random() < 0.5:
                lines.append(f"{rng.choice(['X_ERROR', 'Y_ERROR', 'Z_ERROR', 'DEPOLARIZE1'])}({rng.uniform(0, 0.2):.4f}) {q}")
        for q in range(n):
            if rng.random() < 0.7:
                p = rng.choice([0.0, rng.uniform(0.001, 0.3)])
                b = "" if basis[q] == "Z" else basis[q]
                kind = "MR" if (b == "" and rng.random() < 0.3) else "M"
                lines.append(f"{kind}{b}({p:.4f}) {q}" if p else f"{kind}{b} {q}")
                nrec += 1
        for _ in range(rng.randint(0, 3)):
            if nrec:
                recs = [rng.randint(1, nrec) for _ in range(rng.randint(1, 4))]
                lines.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in recs))
    for o in range(rng.randint(0, 2)):
        if nrec:
            recs = [rng.randint(1, nrec) for _ in range(rng.randint(1, 3))]
            lines.append(f"OBSERVABLE_INCLUDE({o}) " + " ".join(f"rec[-{k}]" for k in recs))
    if gauge:   # a fresh |+> read in Z: a gauge observable (dropped), its M(p) feeds no emitted target
        lines += [f"H {n}", f"M(0.05) {n}", "OBSERVABLE_INCLUDE(2) rec[-1]"]
    return "\n".join(lines) + "\n", gauge


def outcome(text, kw):
    try:
        return "OK " + xtim.Circuit(text).detector_error_model_text(**kw)
    except Exception as e:   # the refusal text is part of the contract
        return f"ERR {type(e).__name__}: {e}"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def fuzz_digest(count=120, seed=20261122):
    rng = random.Random(seed)
    h = hashlib.sha256()
    for _ in range(count):
        text, gauge = fuzz_circuit(rng)
        kw = {"drop_gauge_observables": True} if gauge else {}
        for k2 in ({}, {"format": "xdem"}):
            h.update(outcome(text, {**kw, **k2}).encode())
            h.update(b"\0")
    return h.hexdigest()


RANDOMIZED = {   # T-gate circuits whose faults randomize reads (the correlated-read law; tests/test_dem_correlated_reads)
    "cancelling_pair": "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE2(0.03) 0 1\nT 0\nT_DAG 1\nMX 0 1\nDETECTOR rec[-1] rec[-2]\n",
    "correlated_pair": "R 0 1 2\nH 0\nCX 0 1\nCX 0 2\nR 3 4\nH 3\nCX 3 4\nX_ERROR(0.02) 3\nT 0\nT_DAG 2\n"
                       "T 4\nT_DAG 3\nCX 1 2\nCX 0 3\nMX 0\nMX 1\nMX 2\nMX 3\nMX 4\n"
                       "DETECTOR rec[-2] rec[-1]\nDETECTOR rec[-5] rec[-4] rec[-2]\nDETECTOR rec[-5] rec[-4] rec[-1]\n",
    "negative_parity_pair": "R 0 1 2 3\nH 0\nCX 0 1\nH 2\nCX 2 3\nX_ERROR(0.05) 0\nT 0\nT_DAG 1\n"
                            "MY 0 1 2 3\nDETECTOR rec[-4] rec[-3]\nDETECTOR rec[-4] rec[-3] rec[-2] rec[-1]\n",
    "y_before_t_cx": "R 0 1 2 3 4\nH 0\nH 1\nH 2\nH 3\nCZ 4 1\nCZ 4 2\nCZ 0 2\nX_ERROR(0.02) 1\n"
                     "Y_ERROR(0.02) 2\nT 1\nT_DAG 2\nCX 3 2\nCZ 2 4\nCX 1 3\nMX 0\nMY 1\nMX 2\nMX 3\nMX 4\n"
                     "DETECTOR rec[-3] rec[-2]\n",
}


def randomized_digest():
    h = hashlib.sha256()
    for name in sorted(RANDOMIZED):
        base = RANDOMIZED[name]
        for text in (base, base + "DETECTOR rec[-1] rec[-1] rec[-2]\n", base + "DECISION(0) rec[-1]\nDECISION(1) rec[-2] rec[-2]\n"):
            for kw in ({}, {"format": "xdem"}, {"include_expectations": False}):
                h.update(outcome(text, kw).encode())
                h.update(b"\0")
    return h.hexdigest()


def decision_digest(count=120, seed=20261123):
    rng = random.Random(seed)
    h = hashlib.sha256()
    for _ in range(count):
        text, gauge = fuzz_circuit(rng)
        nrec = sum(1 for ln in text.splitlines() if ln.startswith("M"))
        extra = []
        if nrec:
            for k in range(rng.randint(1, 3)):
                recs = [rng.randint(1, nrec) for _ in range(rng.randint(1, 3))]
                extra.append(f"DECISION({k}) " + " ".join(f"rec[-{r}]" for r in recs))
            if rng.random() < 0.5:
                extra.append("PAULI_EXPECTATION(0) Z0")
        text += "\n".join(extra) + ("\n" if extra else "")
        kw = {"drop_gauge_observables": True} if gauge else {}
        for k2 in ({}, {"include_expectations": False}, {"format": "xdem"}, {"format": "xdem", "include_expectations": False}):
            h.update(outcome(text, {**kw, **k2}).encode())
            h.update(b"\0")
    return h.hexdigest()


REFUSAL_PINS = {   # name: (first line, sha256 of the whole refusal text incl. the hint)
    'nondet_bell_half': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D1 (noiseless parity expectation 0.000000, not +-1)',
        '92e25dafbb75936c473726c1849f1f67ac9ccb386e67b2dec7487439946a88c8'),
    'nondet_detector': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1)',
        'e9d8c2f2869198263c15532ff1a509f334373f600fa913e887cdda7a7c0ff44d'),
    'nondet_magic': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1)',
        'e9d8c2f2869198263c15532ff1a509f334373f600fa913e887cdda7a7c0ff44d'),
    'nondet_magic_partial': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D1 (noiseless parity expectation -0.707107, not +-1)',
        'e1f62620a428d075ef3fc2f89157708aa2918e0d0bf74ce6b3017015a6725844'),
    'nondet_magic_partial_xdem': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D1 (noiseless parity expectation -0.707107, not +-1)',
        'e1f62620a428d075ef3fc2f89157708aa2918e0d0bf74ce6b3017015a6725844'),
    'nondet_magic_xdem': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1)',
        'e9d8c2f2869198263c15532ff1a509f334373f600fa913e887cdda7a7c0ff44d'),
    'nondet_observable': ('ERR XtimDemError: not DEM-expressible: non-deterministic observable L0 (noiseless parity expectation 0.000000, not +-1)',
        '7f37c2bedb7beb4b8d8e2ebe2f73d162ea3f636e80f499f68b0575fdf96f9943'),
    'nondet_second_of_three': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D1 (noiseless parity expectation 0.000000, not +-1)',
        'ff2d7a00617100cbdbe0eecfe3abc8c1b576aed9e9b8cf9b731dbfea7632b994'),
    'nondet_xdem': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1)',
        'e9d8c2f2869198263c15532ff1a509f334373f600fa913e887cdda7a7c0ff44d'),
    'nondet_y_parity': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1)',
        'e9d8c2f2869198263c15532ff1a509f334373f600fa913e887cdda7a7c0ff44d'),
    'nondet_y_parity_noisy': ('ERR XtimDemError: not DEM-expressible: non-deterministic detector D0 (noiseless parity expectation 0.000000, not +-1)',
        'e9d8c2f2869198263c15532ff1a509f334373f600fa913e887cdda7a7c0ff44d'),
}
DETERMINISTIC_PINS = {
    'bell_parity': 'cbb72d4010506fe517cb2ee0d415ab00e0f5347730b6e88d8e4c524651bd318f',
    'constant_parity': '8ee6786f29acf9add2859f855191b510e1c152104974d4ce69ca64d869715af2',
    'gauge_dropped': '5fdcdab35a8b3ebfea53ecc817bd563616958405e2af81fca5ddffe4440949f7',
    'y_parity': 'b09e402b4f84fd4267ed90014f732fac1c154bd3e105f639451ab6c0bc9d8f03',   # S2-CLEAN: 9fb442f's bytes; stim: error(0.01324...) D0
}
FUZZ_PIN = 'e6d656f636e6a4ec1fe046ddfa797d8ebbf67b4f9e4ac76deda40d71e6be40ed'   # 120 circuits, 111 with readout noise, all deterministic


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_refusal_text_pinned(name):
    text, kw = REFUSALS[name]
    got = outcome(text, kw)
    assert got.splitlines()[0] == REFUSAL_PINS[name][0]
    assert sha(got) == REFUSAL_PINS[name][1]


@pytest.mark.parametrize("name", sorted(DETERMINISTIC))
def test_deterministic_dem_pinned(name):
    kw = {"drop_gauge_observables": True} if name == "gauge_dropped" else {}
    assert sha(outcome(DETERMINISTIC[name], kw)) == DETERMINISTIC_PINS[name]


def test_readout_flip_fuzz_pinned():
    assert fuzz_digest() == FUZZ_PIN


def test_randomized_reads_pinned():
    assert randomized_digest() == RANDOMIZED_PIN


def test_decision_fuzz_pinned():
    assert decision_digest() == DECISION_PIN


RANDOMIZED_PIN = "ed9a441275f89209632f3ea25bfc7d95cb3726c2e920850f16e89c4212b524e7"
DECISION_PIN = "688ca67fad6b9b1adc055440bf36357e6d1169bcbd842b36c37c6558c4f2a729"
