"""Stage S1: the exact record-law oracle (X) against the INDEPENDENT dense oracle.

(X) = `xtim._xtim_testapi.RecordLawOracle` (cpp/testapi/record_law_oracle.cpp): per shot, the joint law of every
record column from the term-indexed residual R = P * prod_{j in S} rot(B_j) — design Theorem 1
(pivots, A, K_c, Lambda_j, B, beta, P0 on the free register) plus Lemma 2 (t_c = <P, M_c>),
support-sparse (term rotations on the records), no per-alternative tableau, no dense state.

The independent oracle = tests/_record_oracle.py: a dense statevector run of the ORIGINAL circuit
text with exact enumeration of every noise branch, measurement branch and readout-flip coin. It
imports no xtim code, so it shares none of the engine's upstream (normalize, deferral, term set,
propagation, certified group, noiseless reference).

Gate (design S1): the two exact laws of the EMITTED columns (detector / observable parities
relative to the noiseless reference, decisions raw) agree to 1e-12 on >= 500 random circuits that
cover diagonal, PPR (CH, H T H), mirror pairs, ANTI, gauge, Born observables and decisions,
frames (clean, dressed, poisoned) and the A4 sign case — coverage is asserted, not assumed.
"""
import functools

import numpy as np
import pytest

from _record_oracle import emitted_law
from _record_law_x import emitted_law_x, fault_configs, frame_arg, max_abs_diff, oracle, random_circuit

N_RANDOM = 520
TOL = 1e-12


@functools.lru_cache(maxsize=None)
def _oracle(text):
    return oracle(text)


@pytest.mark.parametrize("seed", range(N_RANDOM))
def test_x_equals_dense_on_random_circuits(seed):
    text, frames = random_circuit(seed)
    orc = _oracle(text)
    for fr in [None] + frames:
        d = max_abs_diff(emitted_law_x(orc, fr), emitted_law(text, fr))
        assert d <= TOL, (seed, fr, d, text)


def _classes(seed):
    """Which design classes circuit `seed` exercises, read off (X) itself."""
    text, frames = random_circuit(seed)
    orc = _oracle(text)
    xz, _, _ = orc.terms()
    n = orc.n
    cols = orc.columns()
    T = orc.num_terms
    out = set()
    if T:
        out.add("ppr" if xz[:, :n].any() else "diagonal")
        rows = {tuple(r) for r in xz}
        if len(rows) < T:
            out.add("mirror")
    law0 = orc.law(np.zeros(T, np.uint8))
    for (kind, _, _), c in zip(cols, law0["cls"]):
        if c == 1:
            out.add("anti")
        if c == 2:
            out.add({"detector": "gauge", "observable": "born_observable", "decision": "born_decision"}[kind])
    for pr, faults in fault_configs(orc):
        S, Px, Pz = orc.residual(faults, None)
        L = orc.law(S)
        if L["kappa"] and L["rho"]:
            out.add("born_with_coins")
        if L["s_sign"].any():
            out.add("a4_sign")
    for fr in frames:
        S_f, _, _ = orc.residual([], frame_arg(fr))
        if orc.term_set_pauli(S_f) is None:
            out.add("frame_poisoned")
        else:
            out.add("frame_dressed" if _frame_is_dressed(orc, fr) else "frame_clean")
    return out


def _shot_law(orc, faults, frame):
    S, Px, Pz = orc.residual(faults, frame_arg(frame))
    keys, probs = orc.enumerate(S, orc.relabel(Px, Pz))
    return {int(k): float(p) for k, p in zip(keys, probs) if p > 1e-14}


def _anchor_image(orc, frame):
    """P' = U F U^dag for a Pauli-image frame: the frame's residual (P_f, S_f) with rot(S_f) = N."""
    S_f, Px, Pz = orc.residual([], frame_arg(frame))
    Nx, Nz = orc.term_set_pauli(S_f)
    return Px ^ Nx, Pz ^ Nz


def _frame_is_dressed(orc, frame):
    """The frame is CLEAN iff the STATIC relabel t = <P', M_c> (P' the anchor image) maps every
    fault configuration's frame-free shot law onto its framed shot law; DRESSED otherwise (the
    design's per-shot term-set relabel is then needed)."""
    m = len(orc.columns())
    t = sum(int(b) << c for c, b in enumerate(orc.relabel(*_anchor_image(orc, frame))))
    for pr, faults in fault_configs(orc):
        lf, l0 = _shot_law(orc, faults, frame), _shot_law(orc, faults, None)
        if max_abs_diff({k ^ t: p for k, p in l0.items()}, lf) > 1e-12:
            return True
    return False


def test_random_family_covers_every_class():
    counts = {}
    for seed in range(N_RANDOM):
        for c in _classes(seed):
            counts[c] = counts.get(c, 0) + 1
    need = ["diagonal", "ppr", "mirror", "anti", "gauge", "born_observable", "born_decision",
            "born_with_coins", "a4_sign", "frame_clean", "frame_dressed", "frame_poisoned"]
    missing = [c for c in need if counts.get(c, 0) < 10]
    assert not missing, (missing, counts)


@pytest.mark.parametrize("seed", range(0, N_RANDOM, 7))
def test_frame_relabel_rule_equals_composition(seed):
    """Section 2.7: for a frame component whose own term set is EMPTY (Pauli anchor image P'), the
    per-term relabel t = <P,M> ^ <P',M> ^ XOR_{j in S} <P',B_j><M,B_j> equals composing the frame
    into the residual — on every fault configuration."""
    text, frames = random_circuit(seed)
    orc = _oracle(text)
    for fr in frames:
        S_f, Pfx, Pfz = orc.residual([], frame_arg(fr))
        if S_f.any():
            continue
        for pr, faults in fault_configs(orc):
            S, Px, Pz = orc.residual(faults, frame_arg(fr))
            S0, P0x, P0z = orc.residual(faults, None)
            assert np.array_equal(S, S0)
            rule = orc.relabel(P0x, P0z) ^ orc.frame_relabel(Pfx, Pfz, S0)
            assert np.array_equal(orc.relabel(Px, Pz), rule)


# ── named fixtures (adversary probes of the design rounds) ─────────────────────────────────────
L4 = """INPUT_QUBITS in 1
RX 0
R 2
T 0
CX 0 1
T 1
CX 0 1
X_ERROR(1) 0
CX 0 1
T_DAG 1
CX 0 1
DEPOLARIZE1(0.001) 2
MX 0
M 2
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]
"""

STRIP = """INPUT_QUBITS in 0
RX 1 2
CCZ 0 1 2
X_ERROR(0.5) 1
CCZ 0 1 2
DEPOLARIZE1(0.0001) 1 2
MX 1 2
DETECTOR rec[-2]
DETECTOR rec[-1]
"""

A4 = """RX 0 1
T 0
T 1
X_ERROR(0.1) 0
MPP Z0*Z1 X0*X1 Y0*Y1
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
"""

GAUGE = "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX 0\nMX 0\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
BORN_OBS_INV = "RX 0\nT 0\nX_ERROR(0.1) 1\nM 1\nMX !0\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"


@pytest.mark.parametrize("text,frame", [
    (L4, None), (L4, {1: "X"}), (L4, {1: "Y"}), (L4, {1: "Z"}),
    (STRIP, None), (STRIP, {0: "X"}), (STRIP, {0: "Z"}),
    (A4, None), (GAUGE, None), (BORN_OBS_INV, None),
])
def test_named_fixtures(text, frame):
    a = emitted_law_x(_oracle(text), frame)
    assert max_abs_diff(a, emitted_law(text, frame)) <= TOL


def test_l4_dressed_frame_truth():
    """adv-A l4.py: with an X frame on the port the Born observable is 1 with probability 0.146
    (the pre-fix static relabel said 0.852); the frame is DRESSED."""
    assert _frame_is_dressed(_oracle(L4), {1: "X"})
    law = emitted_law_x(_oracle(L4), {1: "X"})
    p_obs1 = sum(p for k, p in law.items() if k[1] == 1)
    assert abs(p_obs1 - 0.14644660940672624) < 1e-3   # ~ sin^2(pi/8) mixed with the 1e-3 noise


def test_strip_dressed_frame_truth():
    """adv-A strip.py: the X frame on the CCZ control crosses both CCZs (a Pauli image: CCZ^2 = I)
    but anticommutes with the second CCZ's terms that the X_ERROR(0.5) fault puts in the residual
    — a DRESSED frame; detector 1 (MX 2) is a fair coin (truth 0.500; today's strip gave 0.0001)."""
    orc = _oracle(STRIP)
    S_f, _, _ = orc.residual([], frame_arg({0: "X"}))
    assert S_f.any() and orc.term_set_pauli(S_f) is not None
    assert _frame_is_dressed(orc, {0: "X"})
    law = emitted_law_x(orc, {0: "X"})
    assert abs(sum(p for k, p in law.items() if k[1] == 1) - 0.5) < 1e-12


def test_a4_sign():
    """Z0Z1 * X0X1 = -Y0Y1 on a two-qubit free register (adv-A A4): the Y0Y1 record is the product
    of the two Born directions, and the ordered product of their Hermitian forms carries the sign
    s_c = 1 (round 1 dropped it). The emitted relation (reference-relative) is y3 = y1 ^ y2 on
    every shot, and the joint law equals the dense oracle."""
    orc = _oracle(A4)
    L = orc.law(np.zeros(orc.num_terms, np.uint8))
    assert L["kappa"] == 2 and L["rho"] == 0 and L["born"] == [0, 1]
    assert L["B"][2].tolist() == [1, 1] and L["s_sign"].tolist() == [0, 0, 1]
    law = emitted_law_x(orc)
    assert max_abs_diff(law, emitted_law(A4)) <= TOL
    for k, p in law.items():
        assert p < 1e-15 or k[2] == k[0] ^ k[1]
