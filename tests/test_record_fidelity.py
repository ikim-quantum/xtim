"""xtim.record_fidelity — the REGISTER RecordFidelity (the only route; RF-ref 2026-10-22: the Target API and the
reference / conditioned-reference semantics were removed; the value oracles are tests/test_rfref_route.py and
tests/test_rfref_correction.py, a dense partial trace). Here: the evaluator's surface — shot subsets, the memo scope,
corrections gathered with the subset, the byte budget, the state-model checks (refusals by name), input_pauli buffers.

J-b1 F (xtim-engine-13): the layout-3 evaluator — ``evaluate(buffer, shot_index=None, *, correction=None,
keep_memo=False)`` reads each shot's own record; a decoder correction is a Pauli on the OUTPUT qubits
(``correction`` uint8[shots, 2, n_out]).
"""
import os

import numpy as np
import pytest

from _public import to_prod
from xtim import _xtim
from xtim import record_fidelity as rf


def _corr(cx, cz):
    """The output-Pauli correction X^cx Z^cz as evaluate()'s uint8[shots, 2, n_out]."""
    return np.ascontiguousarray(np.stack([np.asarray(cx, np.uint8), np.asarray(cz, np.uint8)], axis=1))

FAULTS = ["", "X_ERROR(1)", "Z_ERROR(1)", "Y_ERROR(1)"]


def _circuit(rng):
    lines, outs, nq, mpp = [], [], 0, []

    def q():
        nonlocal nq
        nq += 1
        return nq - 1

    first = str(rng.choice(["bell", "tele"]))
    extra = ["ccz", "cs", "gauge", "bell2"] + (["tele2"] if first == "tele" else [])
    blocks = [first] + list(rng.choice(extra, size=rng.integers(1, 3), replace=False))
    tele = None
    for b in blocks:
        if b in ("bell", "bell2"):
            a, c, anc = q(), q(), q()
            lines += [f"RX {a}", f"R {c} {anc}", f"CX {a} {c}"]
            f = rng.choice(FAULTS)
            if f:
                lines.append(f"{f} {rng.choice([a, c])}")
            lines.append(f"{rng.choice(['T', 'T_DAG'])} {a}")
            lines += [f"CX {a} {anc}", f"CX {c} {anc}", f"M {anc}", "DETECTOR rec[-1]"]
            outs += [a, c]
            mpp.append((a, c))
        elif b == "tele":
            # X fault on a |0> control before a CCZ leaves CZ(b, c) on an MX-read wire: the
            # record channel X_c is DRESSED (X_c Z_b) and its projection sets the output state.
            a, c, d, e = q(), q(), q(), q()
            lines += [f"R {a} {d}", f"RX {c} {e}", f"CX {c} {d}"]
            if rng.random() < 0.85:
                lines.append(f"X_ERROR(1) {a}")
            lines += [f"CCZ {a} {c} {e}", f"MX {e}", "DETECTOR rec[-1]"]
            outs += [a, c, d]
            tele = (a, c)
        elif b == "tele2":
            if tele is None:
                continue
            a, c = tele
            c2, e2 = q(), q()
            lines += [f"RX {c2} {e2}", f"CCZ {a} {c2} {e2}", f"MX {e2}", "DETECTOR rec[-1]"]
            outs.append(c2)
        elif b == "ccz":
            a, c, d = q(), q(), q()
            lines.append(f"RX {a} {c} {d}")
            if rng.random() < 0.7:
                lines.append(f"X_ERROR(1) {rng.choice([a, c, d])}")
            lines.append(f"CCZ {a} {c} {d}")
            outs += [a, c, d]
        elif b == "cs":
            a, c = q(), q()
            lines.append(f"RX {a} {c}")
            if rng.random() < 0.7:
                lines.append(f"X_ERROR(1) {rng.choice([a, c])}")
            lines.append(f"CS {a} {c}")
            outs += [a, c]
        else:  # gauge outcome + feedback
            d, e = q(), q()
            lines += [f"RX {d}", f"R {e}", f"CX {d} {e}", f"MX {d}", f"CZ rec[-1] {e}"]
            outs.append(e)
    if mpp and rng.random() < 0.7:
        a, c = mpp[0]
        lines.append(f"MPP {rng.choice(['X', 'Y'])}{a}*X{c}")
    if rng.random() < 0.5:
        lines.append(f"Z_ERROR(1) {rng.choice(outs)}")
    lines.append("OUTPUT_QUBITS out " + " ".join(map(str, outs)))
    return "\n".join(lines) + "\n", outs




def _random_sampler(seed):
    rng = np.random.default_rng(seed)
    txt, outs = _circuit(rng)
    return txt, outs, _xtim.TwirlSampler(txt, 1.0, "", 0)


# a circuit that meets the strict contract (the Born M 0 fed back), fractional F, an unrelated detector
_STRICT = ("R 0 1 2\nH 0\nCX 0 1\nRX 2\nT 2\nDEPOLARIZE1(0.3) 1 2\nM 0\nCX rec[-1] 1\nDECISION(0) rec[-1]\n"
           "R 3\nX_ERROR(0.25) 3\nM 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1 2\n")


def _strict_sampler():
    return _xtim.TwirlSampler(_STRICT, 1.0, "", 0)


def test_shot_subset_correction_and_memo_are_consistent():
    s = _strict_sampler()
    ev = rf.RecordFidelity(s)
    buf = s.sample_barrier(80, seed=5)
    full = ev.evaluate(buf, keep_memo=True)
    sub = list(range(79, -1, -3))
    assert np.array_equal(ev.evaluate(buf, sub, keep_memo=True), full[sub])
    fresh = rf.RecordFidelity(s)
    assert np.array_equal(fresh.evaluate(buf, sub), full[sub])            # memo order-independent
    no = len(s.output_wires())
    rng = np.random.default_rng(2)
    c = _corr(rng.integers(0, 2, (80, no)).astype(np.uint8), rng.integers(0, 2, (80, no)).astype(np.uint8))
    Fc = ev.evaluate(buf, correction=c, keep_memo=True)
    assert np.array_equal(rf.RecordFidelity(s).evaluate(buf, sub, correction=c), Fc[sub])   # the subset gathers c
    assert ev.stats()["memo_hits"] > 0

def test_compact_false_is_refused_and_the_read_maps_are_exported():
    """J-b1 F: a layout-3 buffer is always compact (compact=False refused by name); the exact class and
    exact_flags() are gone; the record / port maps stay."""
    _txt, _outs, s = _random_sampler(2)
    with pytest.raises(ValueError, match="compact=False"):
        s.sample_barrier(4, seed=2, compact=False)
    assert not hasattr(s.sample_barrier(4, seed=2), "exact_flags")
    rr = s.record_reads()
    assert set(rr) >= {"wire", "basis", "invert", "flip", "n"}
    assert s.output_ports()[0][0] == "out"


# ── refusals by name ─────────────────────────────────────────────────────────────────────
def test_refuses_register_above_byte_budget():
    txt = ("RX 0 1 2\nCCZ 0 1 2\nRX 3\nR 4 5\nCX 3 4\nT 3\nCX 3 5\nCX 4 5\nM 5\nDETECTOR rec[-1]\n"
           "OUTPUT_QUBITS out 0 1 2 3 4\n")
    s = _xtim.TwirlSampler(txt, 1.0, "", 0)
    k = len(_xtim.bare_state_of(txt).branch_frames()["free"])
    need = 16 * (2 * 2 ** k + 2 ** k)
    with pytest.raises(RuntimeError, match="byte budget"):
        rf.RecordFidelity(s, max_bytes=need - 1)
    # RF-ref (ruling Q2): max_bytes also budgets everything the evaluator HOLDS (register + memo + contents + kernels)
    buf = s.sample_barrier(5, seed=1)
    probe = rf.RecordFidelity(s)
    probe.evaluate(buf, keep_memo=True)                         # the peak: the memo of one call included
    held = probe.stats()["held_bytes"]
    assert held > 16 * 2 ** k
    with pytest.raises(RuntimeError, match="above the byte budget"):
        rf.RecordFidelity(s, max_bytes=need).evaluate(buf)
    ev = rf.RecordFidelity(s, max_bytes=max(need, held))   # the exact holding is accepted
    assert ev.evaluate(buf).shape == (5,)


_DET9 = "R 9\nM 9\nDETECTOR rec[-1]\n"
_SAME_GROUP_A = "RX 0\nT 0\nRX 2\nMX 2\n" + _DET9 + "OUTPUT_QUBITS out 0\n"
_SAME_GROUP_B = "RX 0\nT 0\nRX 2\nMX 2\nDETECTOR rec[-1]\n" + _DET9 + "OUTPUT_QUBITS out 0\n"


def test_refuses_buffer_from_a_same_group_different_circuit():
    """Review I1 repro: same certified group, different record channels — must refuse, never score."""
    sA = _xtim.TwirlSampler(_SAME_GROUP_A, 1.0, "", 0)
    sB = _xtim.TwirlSampler(_SAME_GROUP_B, 1.0, "", 0)
    bB = sB.sample_barrier(8, seed=2)
    own = rf.RecordFidelity(sB).evaluate(bB)
    with pytest.raises(ValueError, match="state model"):
        rf.RecordFidelity(sA).evaluate(bB)
    # a second sampler compiled from the SAME circuit has the same model and is accepted
    sB2 = _xtim.TwirlSampler(_SAME_GROUP_B, 1.0, "", 0)
    assert np.array_equal(rf.RecordFidelity(sB2).evaluate(bB), own)

def test_refuses_buffer_with_a_different_record_operator_fold():
    """Review I1 (re-review), layout 3: an inverted MX record changes the record operators' inv/ref fold, i.e. the
    state model the evaluator was built on (family_model_key) — C's buffer is refused by name, never scored."""
    sB = _xtim.TwirlSampler(_SAME_GROUP_B, 1.0, "", 0)
    sC = _xtim.TwirlSampler(_SAME_GROUP_B.replace("MX 2", "MX !2"), 1.0, "", 0)
    evB = rf.RecordFidelity(sB)
    with pytest.raises(ValueError, match="state model"):
        evB.evaluate(sC.sample_barrier(8, seed=2))
    assert evB.evaluate(sB.sample_barrier(8, seed=2)).shape == (8,)

# ── a Born-sampled OBSERVABLE is part of the state model (owner decision 2026-10-13: every retained shot is projected
# onto it at its recorded outcome) — here on a rest wire, so the strict contract holds ──
_DET5 = "R 5\nX_ERROR(0.3) 5\nM 5\nDETECTOR rec[-1]\n"
_BORN = "RX 0\nT 0\nRX 3\nM 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET5 + "OUTPUT_QUBITS out 0\n"
_NOBORN = "RX 0\nT 0\nRX 3\nM 3\n" + _DET5 + "OUTPUT_QUBITS out 0\n"


def _born_fixture(shots=300, seed=7):
    s = _xtim.TwirlSampler(_BORN, 1.0, "", 0)
    buf = s.sample_barrier(shots, seed=seed)
    return s, buf


def test_born_observable_is_part_of_the_state_model():
    """The observable's declaration changes the retained state (every shot is projected onto W)
    but neither the bare state nor the record channels: an evaluator built on the circuit
    WITHOUT the observable must refuse a buffer sampled WITH it (and vice versa), by name."""
    sB, bB = _born_fixture(shots=40)
    sA = _xtim.TwirlSampler(_NOBORN, 1.0, "", 0)
    bA = sA.sample_barrier(40, seed=7)
    with pytest.raises(ValueError, match="state model"):          # J-b1 F: the observable is a record operator
        rf.RecordFidelity(sA).evaluate(bB)
    with pytest.raises(ValueError, match="state model"):
        rf.RecordFidelity(sB).evaluate(bA)
    assert rf.RecordFidelity(sA).evaluate(bA).shape == (40,)

def test_input_pauli_buffer_is_served_exactly():
    """J-b1 (plan §5.3; M-FRAME): an input_pauli frame is folded into the record's P_rec, so the evaluator serves the
    buffer — an X frame on the port's qubit 0 (|0> in the bare) scores 0 against the reference; decoded with the X
    correction on qubit 0 it scores 1."""
    txt = "INPUT_QUBITS in 0\nRX 1\nT 1\nCX 0 2\nM 2\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1\n"
    s = _xtim.TwirlSampler(txt, 1.0, "", 0)
    ev = rf.RecordFidelity(s)
    frame = np.zeros((4, 2), np.uint8)
    frame[:, 0] = 1
    buf = s.sample_barrier(4, 1, frame, True)
    assert np.allclose(ev.evaluate(buf), 0.0, atol=1e-12)
    cx = np.zeros((4, 2), np.uint8)
    cx[:, 0] = 1
    assert np.allclose(ev.evaluate(buf, correction=_corr(cx, np.zeros_like(cx))), 1.0, atol=1e-12)

def test_memo_is_scoped_to_one_call_unless_kept():
    s = _strict_sampler()
    ev = rf.RecordFidelity(s)
    buf = s.sample_barrier(60, seed=5)
    a = ev.evaluate(buf)
    assert ev.stats()["memo_bytes"] == 0
    b = ev.evaluate(buf, keep_memo=True)
    assert ev.stats()["memo_bytes"] > 0 and np.array_equal(a, b)
    hits = ev.stats()["memo_hits"]
    orth = ev.stats()["orthogonal_rows"]
    c = ev.evaluate(buf, keep_memo=True)                       # served from the kept memo
    # fid-orth: orthogonal rows are decided before the memo (no entry); every other row is a hit
    orth = ev.stats()["orthogonal_rows"] - orth
    assert np.array_equal(a, c) and ev.stats()["memo_hits"] >= hits + 60 - orth and 60 - orth > 0
    ev.clear()
    assert ev.stats()["memo_bytes"] == 0


def test_born_decisions_are_served_exactly():
    """J-b1 (plan §5.3; M-DECPPR): a Born-class DECISION is a record of the shot's plan; with an output register (a fresh
    wire beside the Born MPP) the evaluator serves the circuit and equals the dense oracle on every shot."""
    from _register_oracle import bare_rho, register_F
    txt = ("RX 0\nR 1 2\nCX 0 1\nT 0\nCX 0 2\nCX 1 2\nX_ERROR(0.2) 0\nM 2\nDETECTOR rec[-1]\n"
           "MPP X0*X1\nDECISION(0) rec[-1]\nRX 5\nDEPOLARIZE1(0.2) 5\nOUTPUT_QUBITS out 5\n")
    s = _xtim.TwirlSampler(txt, 1.0, "", 0)
    assert s.born_dec()["nborn"] == 1
    buf = s.sample_barrier(64, seed=3)
    F = rf.RecordFidelity(s).evaluate(buf)
    ow = list(s.output_wires())
    rhoB = bare_rho(txt, ow)
    assert max(abs(F[i] - register_F(rhoB, buf, i, ow)) for i in range(64)) < 1e-12
    dec = np.unpackbits(np.asarray(buf.decisions()), axis=1, bitorder="little")[:, 0]
    assert dec.any() and (~dec.astype(bool)).any()             # both decision outcomes occur
