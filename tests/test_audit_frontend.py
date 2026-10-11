"""Front-end / record-layout regressions from the 2026-10-09 sampling audit (fixed in 3.1.10).

Every assertion is EXACT (expected record patterns cross-checked against an independent
density-matrix oracle, scratchpad audit-sampling/oracle.py): the circuits are built so the pinned records are deterministic (an
impossible outcome on any shot fails), or the per-shot quantity is a fixed function of the
shot's own records. The single statistical check (finding K's twirl column) has > 100 sigma of
separation between the right and the wrong convention.

A  (records swapped). ``eliminate_hadamards``' Fold 1 (``H q; ...; M q`` -> ``MX q``) emitted the
   folded Measure at the H's stream position. A Measure on another wire lying between the two
   was overtaken, so the two records swapped. The H only survives to that fold when something
   flushes the frame pass first — a PAULI_EXPECTATION does (``flush_all``), which is why ~98% of
   the audit's flags carried one. Detectors / observables / DECISIONs / expectation pairing read
   the swapped records too. Fixed in cpp/src/clifford_frames.cpp (the Measure stays in place).

F  (`!` + MPP ancilla collision). ``scan_max_qubit`` (cpp/src/stim_parse.cpp) skipped a plain
   inverted target ``!q``, so when q was referenced ONLY inverted the MPP/MPAD/MZZ desugar ancilla
   was placed on q itself. F2 is the same defect surfacing as a spurious class rejection.

K  (random-observable reference). The exact engine reports every observable relative to its
   noiseless u=0 reference sample — Stim's convention (``reference_sample()`` biases every
   collapse to +1, i.e. record 0; m2d XORs that reference parity). The twirl engine's Born
   (LOGICAL) observable channel reported the ABSOLUTE parity, so ``engine="auto"`` changed the
   meaning of the observable bit with the engine it picked. Fixed in the twirl engine (an
   engine-side, not front-end, defect): the Born bit is XORed with the same u=0 reference parity.
"""
import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("XTIM_QUIET", "1")
import xtim  # noqa: E402

N = 4000


def _exact_meas(text, n=N):
    _, me = xtim.Circuit(text).compile_detector_sampler(seed=11, engine="exact").sample(
        n, return_measurements=True)
    return np.asarray(me, dtype=np.uint8)


def _sampler_meas(text, n=N):
    return np.asarray(xtim.Circuit(text).compile_sampler(seed=12).sample(n), dtype=np.uint8)


def _decision_text(text, nm):
    return text + "".join(f"DECISION({j}) rec[-{nm - j}]\n" for j in range(nm))


def _barrier_meas(text, nm, n=N):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(_decision_text(text, nm), selfcheck=0)
        b = s.sample_barrier(n, seed=13)
    return np.unpackbits(b.decisions(), axis=1, bitorder="little")[:, :nm]


def _port_meas(text, nm, n=N):
    from xtim.port import Consume, compile as port_compile
    r = port_compile(_decision_text(text, nm), Consume(dets=True, decisions=True)).run(
        shots=n, seed=14)
    return np.asarray(r.decisions, dtype=np.uint8)[:, :nm]


ENGINES = ["exact", "sampler", "barrier", "port"]


def _records(engine, text, nm):
    if engine == "exact":
        return _exact_meas(text)
    if engine == "sampler":
        return _sampler_meas(text)
    if engine == "barrier":
        return _barrier_meas(text, nm)
    return _port_meas(text, nm)


def _check_records(rec, fixed, random_cols=()):
    """fixed: {col: bit} must hold on EVERY shot; random_cols must show both values."""
    for col, bit in fixed.items():
        bad = int((rec[:, col] != bit).sum())
        assert bad == 0, f"record {col}: {bad}/{len(rec)} shots != {bit} (impossible outcome)"
    for col in random_cols:
        assert 0 < int(rec[:, col].sum()) < len(rec), f"record {col} should be random"


# ---------------------------------------------------------------- A: record order
A_CASES = {
    # the audit's minimal repro: records (1, random); 3.1.9 gave (random, 1)
    "A_min": ("X 1\nM 1\nH 0\nM 0\nPAULI_EXPECTATION(0) Z2\n", 2, {0: 1}, (1,)),
    # ascending wires — not about wire order: any Measure between the H and its M is overtaken
    "A_ascending": ("X 0\nM 0\nH 1\nM 1\nPAULI_EXPECTATION(0) Z2\n", 2, {0: 1}, (1,)),
    # three records, the folded one overtakes two
    "A_three": ("X 1\nM 1\nH 0\nM 2\nM 0\nPAULI_EXPECTATION(0) Z3\n", 3, {0: 1, 1: 0}, (2,)),
    # SDG;H -> MY fold (Y basis), same overtaking
    "A_my": ("X 1\nM 1\nS_DAG 0\nH 0\nM 0\nPAULI_EXPECTATION(0) Z2\n", 2, {0: 1}, (1,)),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", sorted(A_CASES))
def test_A_record_order(engine, case):
    text, nm, fixed, rnd = A_CASES[case]
    _check_records(_records(engine, text, nm), fixed, rnd)


def test_A_detector_and_observable_read_the_right_record():
    # DETECTOR on the deterministic M 1 (noiseless parity 1 -> event 0 every shot); with the swap
    # it read the random MX 0 record and fired on ~half the shots.
    t = ("X 1\nM 1\nH 0\nM 0\nPAULI_EXPECTATION(0) Z2\n"
         "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\n")
    for engine in ("exact", "twirl", "auto"):
        d, o = xtim.Circuit(t).compile_detector_sampler(seed=3, engine=engine).sample(
            N, separate_observables=True)
        assert int(np.asarray(d).sum()) == 0, engine
        assert int(np.asarray(o).sum()) == 0, engine


def test_A_expectation_pairs_with_its_own_record():
    # Bell pair (2,0); MX 0 = r and <X2 | r> = (-1)^r EXACTLY. The X 1; M 1 record sits between
    # the folded H and M 0 — 3.1.9 swapped the records, so the expectation column paired with
    # the wrong bit. (sample() returns (dets, expectations, measurements).)
    t = "H 2\nCX 2 0\nX 1\nM 1\nH 0\nM 0\nPAULI_EXPECTATION(0) X2\n"
    _, ev, me = xtim.Circuit(t).compile_detector_sampler(seed=5, engine="exact").sample(
        N, return_measurements=True, return_expectations=True)
    me = np.asarray(me, dtype=np.uint8)
    ev = np.asarray(ev)
    _check_records(me, {0: 1}, (1,))
    np.testing.assert_allclose(ev[:, 0], (-1.0) ** me[:, 1], atol=1e-12)


# ---------------------------------------------------------------- F / F2: `!` + desugar ancilla
F_CASES = {
    # MX !1 random; MPP Z0 on |0> is 0. 3.1.9 put the MPP ancilla ON qubit 1.
    "F_min": ("MX !1\nMPP Z0\n", 2, {1: 0}, (0,)),
    "F_b": ("MPP X0\nM !1\n", 2, {1: 1}, (0,)),
    # the audit's longer form
    "F_long": ("MX !1\nMPP Z0\nMPP X0\nM !1\nX 0\nMZZ 0 1\nM !2\n", 6,
               {1: 0, 5: 1}, (0, 2, 3, 4)),
    # factor-level `!` inside a product must also count its qubit
    "F_factor": ("MPP X0*!Z1\nM !1\n", 2, {1: 1}, (0,)),
    # F2: was rejected as outside the simulable class (pure Clifford)
    "F2": ("MZZ !1 0\nMPP Z0\n", 2, {0: 1, 1: 0}, ()),
    "F2_mxx": ("RX 0\nRX 1\nMXX !1 0\nMPP X0\n", 2, {0: 1, 1: 0}, ()),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", sorted(F_CASES))
def test_F_inverted_target_with_desugar_ancilla(engine, case):
    text, nm, fixed, rnd = F_CASES[case]
    _check_records(_records(engine, text, nm), fixed, rnd)


def test_F_ancilla_sits_above_inverted_only_qubits():
    # parse_info's internal qubit count includes the desugar ancillas: user qubits 0..2 (q2 only
    # as `!2`), one MPP ancilla -> it must be wire 3, i.e. 4 internal qubits.
    info = xtim._xtim.parse_info("M !2\nMPP Z0\n")
    assert info["ok"] and info["n"] == 4, info


# ---------------------------------------------------------------- K: random-observable reference
K_TEXT = ("RX 0\nT 0\nMY 0\nX 1\nM 1\n"
          "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\nDETECTOR rec[-1]\n")
# MY on T|+>: P(r0=0) = (1 + 1/sqrt2)/2; r1 = 1. Stim reference: r0 -> 0, r1 = 1 -> parity 1.
# Reported flip = raw parity XOR 1 = r0, so P(flip) = (1 - 1/sqrt2)/2.
K_P = (1 - 2 ** -0.5) / 2


def test_K_exact_is_stim_reference_relative():
    _, o, me = xtim.Circuit(K_TEXT).compile_detector_sampler(seed=1, engine="exact").sample(
        N, separate_observables=True, return_measurements=True)
    o = np.asarray(o, dtype=np.uint8)[:, 0]
    me = np.asarray(me, dtype=np.uint8)
    np.testing.assert_array_equal(o ^ me[:, 0] ^ me[:, 1], np.ones(N, np.uint8))


def test_K_clifford_analogue_matches_stim_m2d():
    stim = pytest.importorskip("stim")
    t = "RX 0\nMY 0\nX 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\nDETECTOR rec[-1]\n"
    _, o, me = xtim.Circuit(t).compile_detector_sampler(seed=2, engine="exact").sample(
        500, separate_observables=True, return_measurements=True)
    _, so = stim.Circuit(t).compile_m2d_converter().convert(
        measurements=np.asarray(me, dtype=bool), separate_observables=True)
    np.testing.assert_array_equal(np.asarray(o, dtype=bool), so)


@pytest.mark.parametrize("engine", ["twirl", "auto"])
def test_K_all_engines_agree_on_reference(engine):
    n = 40000
    s = xtim.Circuit(K_TEXT).compile_detector_sampler(seed=4, engine=engine)
    _, o = s.sample(n, separate_observables=True)
    m = float(np.asarray(o, dtype=np.uint8)[:, 0].mean())
    sig = (K_P * (1 - K_P) / n) ** 0.5
    assert abs(m - K_P) < 5 * sig, (engine, m, K_P)   # the absolute convention gives 1 - K_P


def _random_clifford(seed):
    """Small noiseless Clifford circuit (no plain R/RX/RY — see the docs' reset caveat) with
    random-basis, inverted, product and feedback measurements."""
    import random
    r = random.Random(seed)
    n = r.randint(1, 5)
    g1 = ["H", "S", "S_DAG", "SQRT_X", "SQRT_Y_DAG", "C_XYZ", "H_XY", "X", "Y", "Z"]
    g2 = ["CX", "CY", "CZ", "XCY", "SWAP", "ISWAP", "SQRT_XX"]
    lines, nm = [], 0
    for _ in range(r.randint(3, 14)):
        k = r.random()
        if k < 0.35:
            lines.append(f"{r.choice(g1)} {r.randrange(n)}")
        elif k < 0.55 and n >= 2:
            a, b = r.sample(range(n), 2)
            lines.append(f"{r.choice(g2)} {a} {b}")
        elif k < 0.85:
            m = r.choice(["M", "MX", "MY", "MR", "MRX", "MRY"])
            lines.append(f"{m} {'!' if r.random() < 0.3 else ''}{r.randrange(n)}")
            nm += 1
        elif k < 0.93:
            qs = r.sample(range(n), r.randint(1, min(3, n)))
            lines.append("MPP " + ("!" if r.random() < 0.3 else "")
                         + "*".join(f"{r.choice('XYZ')}{q}" for q in qs))
            nm += 1
        elif nm:
            lines.append(f"C{r.choice('XYZ')} rec[-{r.randint(1, nm)}] {r.randrange(n)}")
    return "\n".join(lines) + "\n", nm


def test_K_random_observable_reference_matches_stim_reference_sample():
    """Every observable (deterministic or random) of the exact engine is parity XOR Stim's
    reference_sample() parity, per shot. 3.1.9 disagreed on ~1/3 of these circuits (an
    eliminate_hadamards sign stamp or a magic-free out-of-order batch collapse)."""
    import random
    stim = pytest.importorskip("stim")
    checked = 0
    for seed in range(250):
        t, nm = _random_clifford(seed)
        if nm == 0:
            continue
        r = random.Random(10_000 + seed)
        sets = [sorted(r.sample(range(nm), r.randint(1, nm))) for _ in range(3)]
        full = t + "".join(f"OBSERVABLE_INCLUDE({j}) " + " ".join(f"rec[-{nm - i}]" for i in s)
                           + "\n" for j, s in enumerate(sets))
        ref = stim.Circuit(full).reference_sample()
        _, o, me = xtim.Circuit(full).compile_detector_sampler(seed=1, engine="exact").sample(
            32, separate_observables=True, return_measurements=True)
        o = np.asarray(o, dtype=np.uint8)
        me = np.asarray(me, dtype=np.uint8)
        for j, s in enumerate(sets):
            want = int(ref[s].sum() % 2)
            got = o[:, j] ^ (me[:, s].sum(1) % 2).astype(np.uint8)
            assert np.all(got == want), (seed, j, s, full)
        checked += 1
    assert checked > 200


def test_K_barrier_obs_column_is_reference_relative_too():
    # The Born reference is applied by the reporting layer on BOTH twirl surfaces (sample and
    # sample_barrier); the engine itself carries the absolute bit.
    n = 40000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(K_TEXT, selfcheck=0)
        b = s.sample_barrier(n, seed=6)
    o = np.unpackbits(np.asarray(b.obs()), axis=1, bitorder="little")[:, 0]
    sig = (K_P * (1 - K_P) / n) ** 0.5
    assert abs(float(o.mean()) - K_P) < 5 * sig, float(o.mean())
