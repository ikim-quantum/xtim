"""DEM export: measurement-record flip probabilities (`M(p) q`, `MX(p)`, `MY(p)`, `MR(p)`,
`MRX(p)`, `MPP(p)`, `MXX(p)`...) are error mechanisms, exactly as Stim models them.

Stim's `detector_error_model()` turns `M(p) q` into `error(p) <every detector / observable
that reads that record>`. Before the fix (3.1.3) xtim's exporter walked ONLY the Noise
instructions of the deferred stream and never looked at `Instr::readout_flip_p`, so every
readout flip was silently absent from the decoder's model (found 2026-10-06 in adaptq: a
policy-noised Steane memory decoded at the raw readout-flip rate).

Oracle: `stim.Circuit(text).detector_error_model()` on Clifford circuits, compared after
canonicalisation (parse both DEMs, key each mechanism by its sorted target set, compare the
probabilities to 1e-12 relative). For magic-bearing circuits (no Stim oracle) the oracle is
an independent Python composition: the readout-free xtim DEM plus the record-flip mechanisms
built from the circuit's declared record sets, XOR-composed (p (+) q = p + q - 2pq).
"""
import math
import re

import pytest

import xtim

stim = pytest.importorskip("stim")


# ---------------------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------------------
def canon(dem: "stim.DetectorErrorModel"):
    """{sorted-target-tuple: probability} over the error mechanisms, plus the set of
    detector / observable ids the DEM mentions (coordinates ignored)."""
    errors = {}
    dets, obs = set(), set()
    for ins in dem.flattened():
        if ins.type == "error":
            key = []
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    key.append(("D", t.val))
                    dets.add(t.val)
                elif t.is_logical_observable_id():
                    key.append(("L", t.val))
                    obs.add(t.val)
                # separators ("^") carry no information for an undecomposed comparison
            key = tuple(sorted(key))
            assert key not in errors, f"duplicate mechanism {key}"
            errors[key] = ins.args_copy()[0]
        elif ins.type == "detector":
            for t in ins.targets_copy():
                dets.add(t.val)
        elif ins.type == "logical_observable":
            for t in ins.targets_copy():
                obs.add(t.val)
    return errors, dets, obs


def assert_dem_equal(got: "stim.DetectorErrorModel", want: "stim.DetectorErrorModel",
                     rel=1e-12):
    ge, gd, go = canon(got)
    we, wd, wo = canon(want)
    assert gd == wd, f"detector sets differ: {sorted(gd)} vs {sorted(wd)}"
    assert go == wo, f"observable sets differ: {sorted(go)} vs {sorted(wo)}"
    assert set(ge) == set(we), (
        f"mechanism sets differ:\n  only xtim: {sorted(set(ge) - set(we))}\n"
        f"  only stim: {sorted(set(we) - set(ge))}")
    for k in we:
        assert math.isclose(ge[k], we[k], rel_tol=rel, abs_tol=0.0), (k, ge[k], we[k])


def xtim_dem(text: str, **kw) -> "stim.DetectorErrorModel":
    return xtim.Circuit(text).detector_error_model(**kw)


def stim_dem(text: str) -> "stim.DetectorErrorModel":
    return stim.Circuit(text).detector_error_model()


# ---------------------------------------------------------------------------------------
# minimal cases — each readout flavour, one record, one detector
# ---------------------------------------------------------------------------------------
MINIMAL = {
    "M":   "R 0\nM(0.1) 0\nDETECTOR rec[-1]\n",
    "MX":  "RX 0\nMX(0.1) 0\nDETECTOR rec[-1]\n",
    "MY":  "RY 0\nMY(0.1) 0\nDETECTOR rec[-1]\n",
    "MR":  "R 0\nMR(0.1) 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n",
    "MRX": "RX 0\nMRX(0.1) 0\nDETECTOR rec[-1]\nMX 0\nDETECTOR rec[-1]\n",
    "MRY": "RY 0\nMRY(0.1) 0\nDETECTOR rec[-1]\nMY 0\nDETECTOR rec[-1]\n",
    "MPP": "R 0 1\nMPP(0.1) Z0*Z1\nDETECTOR rec[-1]\n",
    "MPP_X": "RX 0 1\nMPP(0.1) X0*X1\nDETECTOR rec[-1]\n",
    "MZZ": "R 0 1\nMZZ(0.1) 0 1\nDETECTOR rec[-1]\n",
    "MXX": "RX 0 1\nMXX(0.1) 0 1\nDETECTOR rec[-1]\n",
    "M_invert": "R 0\nM(0.1) !0\nDETECTOR rec[-1]\n",
    "M_half": "R 0\nM(0.5) 0\nDETECTOR rec[-1]\n",
    "M_zero_is_no_mechanism": "R 0\nM(0) 0\nDETECTOR rec[-1]\n",
    "M_observable": "R 0\nM(0.1) 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "M_det_and_obs": "R 0\nM(0.1) 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "M_unread_record_dropped": "R 0 1\nM(0.1) 0\nM(0.2) 1\nDETECTOR rec[-1]\n",
}


@pytest.mark.parametrize("name", sorted(MINIMAL))
def test_minimal_readout_flip_matches_stim(name):
    text = MINIMAL[name]
    assert_dem_equal(xtim_dem(text), stim_dem(text))


def test_minimal_M_is_literally_one_mechanism():
    # The one-line answer: `M(0.1) 0` + `DETECTOR rec[-1]` == `error(0.1) D0`.
    errors, dets, obs = canon(xtim_dem(MINIMAL["M"]))
    assert errors == {(("D", 0),): pytest.approx(0.1, rel=1e-15)}


# ---------------------------------------------------------------------------------------
# multiple targets, overlapping detectors, composition with gate noise
# ---------------------------------------------------------------------------------------
MULTI = {
    # one M line, three records, chained detectors + an observable on the last record
    "three_targets": (
        "R 0 1 2\nM(0.1) 0 1 2\n"
        "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"),
    # distinct probabilities per record, written as separate lines
    "distinct_p": (
        "R 0 1 2\nM(0.1) 0\nM(0.2) 1\nM(0.3) 2\n"
        "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"),
    # readout flip composes (p (+) q) with an X_ERROR that has the SAME signature
    "compose_same_signature": (
        "R 0\nX_ERROR(0.05) 0\nM(0.1) 0\nDETECTOR rec[-1]\n"),
    # the authored rewrite that reproduced the adaptq defect: X_ERROR(p)+M  vs  M(p)
    "rep3_authored_vs_readout": (
        "R 0 1 2\nX_ERROR(0.01) 0 1 2\nM(0.02) 0 1 2\n"
        "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"),
    # a mid-circuit MR(p) whose qubit is reused afterwards: the flip is a RECORD flip only,
    # the state is untouched (unlike X_ERROR before M)
    "mid_circuit_MR_reuse": (
        "R 0 1\nCX 0 1\nMR(0.1) 1\nCX 0 1\nM(0.2) 1\nM 0\n"
        "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-3] rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"),
    # MPP(p) on a product, inside a stabilizer round, followed by data readout
    "mpp_round": (
        "R 0 1 2\nMPP(0.1) Z0*Z1 Z1*Z2\nDEPOLARIZE1(0.01) 0 1 2\nMPP(0.1) Z0*Z1 Z1*Z2\n"
        "DETECTOR rec[-4] rec[-2]\nDETECTOR rec[-3] rec[-1]\nM(0.05) 0 1 2\n"
        "DETECTOR rec[-5] rec[-3] rec[-2]\nDETECTOR rec[-4] rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"),
}


@pytest.mark.parametrize("name", sorted(MULTI))
def test_multi_record_readout_flip_matches_stim(name):
    text = MULTI[name]
    assert_dem_equal(xtim_dem(text), stim_dem(text))


def test_authored_rewrite_equivalence_rep3():
    # Terminal readout: `X_ERROR(p) q; M q` and `M(p) q` are the SAME mechanism set
    # (an X right before a terminal Z-read only flips that record). Both must agree with
    # Stim; before the fix the M(p) form lost the mechanisms entirely.
    authored = ("R 0 1 2\nX_ERROR(0.02) 0 1 2\nM 0 1 2\n"
                "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    readout = ("R 0 1 2\nM(0.02) 0 1 2\n"
               "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert_dem_equal(xtim_dem(readout), xtim_dem(authored))
    assert_dem_equal(xtim_dem(readout), stim_dem(readout))


# ---------------------------------------------------------------------------------------
# a real stabilizer memory: Stim's generated repetition / surface codes carry MR(p) / M(p)
# (before_measure_flip_probability) alongside gate noise, DETECTOR coords, OBSERVABLE_INCLUDE
# ---------------------------------------------------------------------------------------
GENERATED = [
    ("repetition_code:memory", dict(rounds=3, distance=3,
                                    before_measure_flip_probability=0.01)),
    ("repetition_code:memory", dict(rounds=3, distance=3,
                                    before_measure_flip_probability=0.01,
                                    after_clifford_depolarization=0.001,
                                    before_round_data_depolarization=0.002)),
    ("surface_code:rotated_memory_z", dict(rounds=2, distance=3,
                                           before_measure_flip_probability=0.01,
                                           after_clifford_depolarization=0.001)),
    ("surface_code:rotated_memory_x", dict(rounds=2, distance=3,
                                           before_measure_flip_probability=0.01,
                                           after_reset_flip_probability=0.002)),
]


def readoutify(text: str) -> str:
    """Stim's generators author readout noise as `X_ERROR(p) q..` immediately before `M q..`
    (`Z_ERROR` before `MX`). Rewrite each such pair into the `M(p) q..` form — the form
    adaptq's runtime noise policy writes — which is the SAME Stim model (an X right before a
    Z-read is a record flip)."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(X_ERROR|Z_ERROR)\(([^)]*)\) (.*)$", lines[i])
        if m and i + 1 < len(lines):
            m2 = re.match(r"^(M|MR|MX|MRX|MY|MRY) (.*)$", lines[i + 1])
            if m2 and m2.group(2) == m.group(3) and (
                    (m.group(1) == "X_ERROR" and m2.group(1) in ("M", "MR", "MY", "MRY")) or
                    (m.group(1) == "Z_ERROR" and m2.group(1) in ("MX", "MRX", "MY", "MRY"))):
                out.append(f"{m2.group(1)}({m.group(2)}) {m2.group(2)}")
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("kind,kw", GENERATED, ids=[g[0] + "/" + "+".join(sorted(g[1])) for g in GENERATED])
def test_generated_memory_with_readout_flips_matches_stim(kind, kw):
    authored = str(stim.Circuit.generated(kind, **kw).flattened())
    text = readoutify(authored)
    assert text != authored and re.search(r"^M[RXY]*\(", text, re.M)   # really rewritten
    assert_dem_equal(xtim_dem(text), stim_dem(text))
    # ... and the Stim model of the rewritten circuit is the model of the authored one
    assert_dem_equal(stim_dem(text), stim_dem(authored))


# ---------------------------------------------------------------------------------------
# Steane 1-round memory, SD6-style readout noise — the adaptq reproduction shape
# ---------------------------------------------------------------------------------------
STEANE_H = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]


def steane_memory(p_meas: float, p_data: float) -> str:
    lines = ["R 0 1 2 3 4 5 6"]
    # Z-type stabilizer round via MPP (3 records), noisy readout
    lines.append(f"MPP({p_meas}) " + " ".join(
        "*".join(f"Z{q}" for q in range(7) if row[q]) for row in STEANE_H))
    lines.append(f"X_ERROR({p_data}) 0 1 2 3 4 5 6")
    lines.append(f"MPP({p_meas}) " + " ".join(
        "*".join(f"Z{q}" for q in range(7) if row[q]) for row in STEANE_H))
    for i in range(3):
        lines.append(f"DETECTOR rec[{-6 + i}] rec[{-3 + i}]")
    lines.append(f"M({p_meas}) 0 1 2 3 4 5 6")
    for i, row in enumerate(STEANE_H):
        recs = " ".join(f"rec[{q - 7}]" for q in range(7) if row[q])
        lines.append(f"DETECTOR rec[{-10 + i}] {recs}")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-7] rec[-6] rec[-5] rec[-4] rec[-3] rec[-2] rec[-1]")
    return "\n".join(lines) + "\n"


def test_steane_memory_readout_noise_matches_stim():
    text = steane_memory(p_meas=0.0066, p_data=0.001)
    got, want = xtim_dem(text), stim_dem(text)
    assert_dem_equal(got, want)
    # and the readout mechanisms are really there: every data-readout record feeds
    # the observable, so 7 single-record mechanisms carry L0
    errors, _, _ = canon(got)
    with_L0 = [k for k in errors if ("L", 0) in k]
    assert len(with_L0) >= 7


# ---------------------------------------------------------------------------------------
# non-readout corpus must be BYTE-identical before/after the fix (the refactoring oracle)
# ---------------------------------------------------------------------------------------
def test_readout_free_circuit_unchanged_by_the_readout_pass():
    # A circuit with no M(p) carries no readout mechanism: the DEM text must equal the
    # text the exporter produced before the readout pass existed (pinned literal).
    text = ("R 0 1 2\nX_ERROR(0.05) 0 1 2\nM 0 1 2\n"
            "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    assert xtim.Circuit(text).detector_error_model_text() == (
        "error(0.050000000000000003) D0\n"
        "error(0.050000000000000003) D0 D1\n"
        "error(0.050000000000000003) D1 L0\n")


# ---------------------------------------------------------------------------------------
# magic-bearing circuit: no Stim oracle; independent Python composition oracle
# ---------------------------------------------------------------------------------------
def _records_feeding(text: str):
    """Absolute record index -> list of DEM target labels ('D#'/'L#') that read it, from the
    circuit's DETECTOR / OBSERVABLE_INCLUDE / PAULI_EXPECTATION-frame declarations."""
    c = xtim.Circuit(text)
    n_meas = 0
    dets, obs, frames = [], {}, []
    rec_re = re.compile(r"rec\[(-\d+)\]")
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split()[0]
        base = name.split("(", 1)[0]
        if base in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            n_meas += len([t for t in line.split()[1:]])
        elif base in ("MPP",):
            n_meas += len(line.split()[1:])
        elif base == "DETECTOR":
            dets.append([n_meas + int(k) for k in rec_re.findall(line)])
        elif base == "OBSERVABLE_INCLUDE":
            idx = int(re.match(r"OBSERVABLE_INCLUDE\((\d+)\)", name).group(1))
            obs.setdefault(idx, []).extend(n_meas + int(k) for k in rec_re.findall(line))
        elif base == "PAULI_EXPECTATION":
            frames.append([n_meas + int(k) for k in rec_re.findall(line)])
    assert n_meas == c.num_measurements
    O = (max(obs) + 1) if obs else 0
    feeds = {}
    for d, recs in enumerate(dets):
        for r in recs:
            feeds.setdefault(r, []).append(("D", d))
    for o, recs in obs.items():
        for r in recs:
            feeds.setdefault(r, []).append(("L", o))
    for i, recs in enumerate(frames):            # expectation column r is L(O + r)
        for r in recs:
            feeds.setdefault(r, []).append(("L", O + i))
    # parity: a record listed twice in one target cancels
    out = {}
    for r, lst in feeds.items():
        s = {}
        for t in lst:
            s[t] = s.get(t, 0) ^ 1
        out[r] = tuple(sorted(t for t, v in s.items() if v))
    return out


def _readout_probs(text: str):
    """Absolute record index -> flip probability, from the M-family (p) arguments."""
    probs = {}
    n = 0
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split()[0]
        base = name.split("(", 1)[0]
        if base not in ("M", "MX", "MY", "MR", "MRX", "MRY", "MPP"):
            continue
        m = re.match(r"[A-Z]+\(([^)]*)\)", name)
        p = float(m.group(1)) if m else 0.0
        for _ in line.split()[1:]:
            if p > 0:
                probs[n] = p
            n += 1
    return probs


def compose_readout_oracle(text_noisy: str, text_clean: str):
    """Independent oracle: xtim DEM of the readout-FREE circuit, composed with the record-flip
    mechanisms derived from the declarations (XOR composition p (+) q = p + q - 2pq)."""
    errors, dets, obs = canon(xtim_dem(text_clean))
    feeds = _records_feeding(text_noisy)
    for r, p in _readout_probs(text_noisy).items():
        key = feeds.get(r, ())
        if not key:
            continue
        a = errors.get(key, 0.0)
        errors[key] = a + p - 2.0 * a * p
    return errors


def strip_readout(text: str) -> str:
    return re.sub(r"^(M|MX|MY|MR|MRX|MRY|MPP|MXX|MYY|MZZ)\([^)]*\)", r"\1", text, flags=re.M)


def test_magic_circuit_readout_flips_compose_with_gate_noise():
    # A magic-bearing (T-gate, post-selected) protocol: the bundled cultivation_d3_rate
    # example, with its authored `X_ERROR(p) q; M q` readout noise rewritten into `M(p) q`
    # (the form adaptq's runtime noise policy writes). No Stim oracle exists for a T
    # circuit, so the oracle is the independent Python composition above. (cultivation_d5,
    # which ships with M(0.001) lines, refuses its DEM for an unrelated reason — 19 twirled
    # fair-coin reads on one alternative — so it cannot serve here.)
    authored = xtim.load_example("cultivation_d3_rate").text
    noisy = readoutify(authored)                 # X_ERROR(p) q; M q   ->   M(p) q
    clean = strip_readout(noisy)                 # M(p) q             ->   M q  (no readout noise)
    assert noisy != authored and re.search(r"^M[RXY]*\(", noisy, re.M)
    assert not re.search(r"^M[RXY]*\(", clean, re.M)
    c = xtim.Circuit(noisy)
    res_noisy = c.detector_error_model_with_reject(include_expectations=True)
    # the M(p) form is the authored form's model (an X right before a terminal Z-read is a
    # record flip) — the equivalence the adaptq reproduction rewrote
    assert_dem_equal(res_noisy.dem,
                     xtim.Circuit(authored).detector_error_model_with_reject(
                         include_expectations=True).dem)
    got, _, _ = canon(res_noisy.dem)
    want = compose_readout_oracle(noisy, clean)
    assert set(got) == set(want), (
        f"only xtim: {sorted(set(got) - set(want))[:5]}\n"
        f"only oracle: {sorted(set(want) - set(got))[:5]}")
    for k in want:
        assert math.isclose(got[k], want[k], rel_tol=1e-12, abs_tol=0.0), (k, got[k], want[k])
    # the readout-flip mechanisms are a strict addition, never a post-select flag
    res_clean = xtim.Circuit(clean).detector_error_model_with_reject(include_expectations=True)
    assert res_noisy.postselect_faults == res_clean.postselect_faults


def test_expectation_frame_record_flip_flips_the_expectation_column():
    # A readout flip on a record that a PAULI_EXPECTATION frame folds in flips the FOLDED
    # sign the sampler reports (rec_flip is applied before the emask fold), so the
    # mechanism must carry that expectation's L-column. Teleport-style: measure q0 in X,
    # frame the sign of Z1 on that record.
    text = ("R 0 1\nH 0\nCX 0 1\nT 1\nMX(0.1) 0\nPAULI_EXPECTATION(0) Z1 rec[-1]\n")
    errors, _, obs = canon(xtim_dem(text, include_expectations=True))
    assert errors == {(("L", 0),): pytest.approx(0.1, rel=1e-15)}
    # and the opt-out (no expectation columns) drops it as an unread record
    errors_off, _, _ = canon(xtim_dem(text, include_expectations=False))
    assert errors_off == {}
