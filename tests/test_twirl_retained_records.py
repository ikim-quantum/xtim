"""The twirl sampler's RETAINED state is conditioned on its own records.

Defect (CHANGELOG [Unreleased], 2026-10-08): a fault that crosses a T gate
leaves an S-tail residual that mixes the sector of a split check as a fair coin.  The coin reaches the shot's
records (a DETECTOR / DECISION / OBSERVABLE that reads the check), but the
retained state (``BarrierBuffer.materialize``) was the UNPROJECTED residual state
R|psi>: a superposition over the coin's sectors, so a record-bearing check read
<g> = 0 on every faulted shot while its record said +1 or -1.

Oracle: a dense state vector of the same circuit with the shot's fault
inserted and every measured ancilla PROJECTED onto the outcome the shot
recorded; the reduced state on the OUTPUT_QUBITS (every Pauli expectation) must
equal the engine's retained state shot by shot.  Checks nobody measures must
stay coherent (an over-projecting fix is caught by the unmeasured-check case).
"""
import itertools
import re

import numpy as np
import pytest

import xtim._xtim as _x

# ── dense oracle ──────────────────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], complex)
_Z = np.diag([1, -1]).astype(complex)
_Y = 1j * _X @ _Z
_H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
_ONE = {"H": _H, "S": np.diag([1, 1j]), "S_DAG": np.diag([1, -1j]),
        "T": np.diag([1, np.exp(1j * np.pi / 4)]), "T_DAG": np.diag([1, np.exp(-1j * np.pi / 4)]),
        "X": _X, "Y": _Y, "Z": _Z}
_ERR = {"X_ERROR": _X, "Y_ERROR": _Y, "Z_ERROR": _Z}


def _apply1(psi, n, g, q):
    v = np.moveaxis(psi.reshape([2] * n), q, 0)
    v = np.tensordot(g, v, axes=([1], [0]))
    return np.moveaxis(v, 0, q).reshape(-1)


def _apply_cx(psi, n, c, t):
    v = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[c] = 1
    sub = v[tuple(idx)]
    v[tuple(idx)] = np.flip(sub, axis=t - 1 if t > c else t)
    return v.reshape(-1)


def _project_z(psi, n, q, bit):
    v = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[q] = 1 - bit
    v[tuple(idx)] = 0
    return v.reshape(-1)


def _dense_run(text, fired, outcomes=None):
    """Dense run of ``text``.  ``fired`` = indices (in text order) of the *_ERROR
    instructions that fire; ``outcomes`` = the raw measurement outcomes to project
    onto (None: return the noiseless outcome probabilities instead).  Returns
    (psi, output qubits, list of measured outcome probabilities)."""
    qubits = {int(t) for ln in text.splitlines() for t in ln.split()[1:] if t.isdigit()}
    n = max(qubits) + 1
    psi = np.zeros(1 << n, complex)
    psi[0] = 1
    out, probs, k_err, k_meas = None, [], 0, 0
    for ln in text.splitlines():
        tok = ln.split()
        if not tok:
            continue
        g = re.sub(r"\(.*\)", "", tok[0])
        qs = [int(t) for t in tok[1:] if t.isdigit()]
        if g == "R":
            continue                     # every wire is fresh in the test circuits
        if g in _ERR:
            for q in qs:
                if k_err in fired:
                    psi = _apply1(psi, n, _ERR[g], q)
                k_err += 1
        elif g in _ONE:
            for q in qs:
                psi = _apply1(psi, n, _ONE[g], q)
        elif g == "CX":
            for c, t in zip(qs[::2], qs[1::2]):
                psi = _apply_cx(psi, n, c, t)
        elif g == "M":
            for q in qs:
                p1 = np.linalg.norm(_project_z(psi, n, q, 1)) ** 2 / np.linalg.norm(psi) ** 2
                probs.append(p1)
                if outcomes is not None:
                    psi = _project_z(psi, n, q, int(outcomes[k_meas]))
                    assert np.linalg.norm(psi) > 1e-9, "recorded outcome has zero probability"
                    psi = psi / np.linalg.norm(psi)
                k_meas += 1
        elif g == "OUTPUT_QUBITS":
            out = qs
    return psi, n, out, probs


def _paulis(k):
    return list(itertools.product("IXYZ", repeat=k))


def _dense_expect(psi, n, qs, word):
    v = psi
    for q, p in zip(qs, word):
        if p != "I":
            v = _apply1(v, n, {"X": _X, "Y": _Y, "Z": _Z}[p], q)
    return np.vdot(psi, v)


def _engine_reads(ow, word):
    xs = [w for w, p in zip(ow, word) if p in "XY"]
    zs = [w for w, p in zip(ow, word) if p in "ZY"]
    return (xs, zs, sum(p == "Y" for p in word) % 4)    # Y = i·X·Z


def _records_per_measurement(text, buf, shot, n_meas):
    """Each test circuit gives every M its own single-record DETECTOR / DECISION /
    OBSERVABLE; all three report the raw record value.  Map them back."""
    dets = np.unpackbits(np.asarray(buf.dets(), np.uint8)[shot], bitorder="little")
    decs = np.unpackbits(np.asarray(buf.decisions(), np.uint8)[shot], bitorder="little") \
        if np.asarray(buf.decisions()).size else None
    obs = np.unpackbits(np.asarray(buf.obs(), np.uint8)[shot], bitorder="little") \
        if np.asarray(buf.obs()).size else None
    m = [None] * n_meas
    k_meas, k_det = 0, 0
    for ln in text.splitlines():
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "M":
            k_meas += len(tok) - 1
            continue
        mm = re.findall(r"rec\[-(\d+)\]", ln)
        if not mm:
            continue
        assert len(mm) == 1
        j = k_meas - int(mm[0])
        if tok[0] == "DETECTOR":
            m[j] = int(dets[k_det]); k_det += 1
        elif tok[0].startswith("DECISION"):
            m[j] = int(decs[int(re.findall(r"\((\d+)\)", tok[0])[0])])
        elif tok[0].startswith("OBSERVABLE_INCLUDE"):
            m[j] = int(obs[int(re.findall(r"\((\d+)\)", tok[0])[0])])
    assert all(v is not None for v in m), m
    return m


def _check(text, shots, seed, fired_of_shot):
    """Per shot: the retained state's reduced output state == the dense physical
    state conditioned on the shot's records.  Returns #shots whose records were
    non-deterministic (coin shots) so callers can assert non-vacuity."""
    S = _x.TwirlSampler(text, 1.0, "", 0, False, "")
    buf = S.sample_barrier(shots, seed)
    ow = list(S.output_wires())
    n_meas = sum(len(ln.split()) - 1 for ln in text.splitlines() if ln.startswith("M "))
    words = _paulis(len(ow))
    eng = np.asarray(buf.pauli_expectations_all([_engine_reads(ow, w) for w in words]))
    coin_shots = 0
    for i in range(shots):
        fired = fired_of_shot(buf, i)
        _, _, _, probs = _dense_run(text, fired)
        coin_shots += any(1e-9 < p < 1 - 1e-9 for p in probs)
        m = _records_per_measurement(text, buf, i, n_meas)
        psi, n, out, _ = _dense_run(text, fired, m)
        want = np.array([_dense_expect(psi, n, out, w) for w in words])
        bad = np.flatnonzero(np.abs(eng[i] - want) > 1e-9)
        assert bad.size == 0, (i, fired, m, [("".join(words[b]), eng[i][b], want[b]) for b in bad[:6]])
    return coin_shots


# ── circuits ──────────────────────────────────────────────────────────────────
# The adaptq harness3 §D2 4-qubit engine repro: Bell pair, X before T, X0X1 read
# through an ancilla with a DETECTOR (+ a trivial decision).
REPRO = ("R 0 1\nH 0\nCX 0 1\nX_ERROR(0.5) 0\nT 0\nT_DAG 1\nR 3\nH 3\nCX 3 0\nCX 3 1\n"
         "H 3\nM 3\nDETECTOR rec[-1]\nR 4\nM 4\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n")


def _xx_check(anc, a, b, rec):
    return f"R {anc}\nH {anc}\nCX {anc} {a}\nCX {anc} {b}\nH {anc}\nM {anc}\n{rec} rec[-1]\n"


def _zz_check(anc, a, b, rec):
    return f"R {anc}\nCX {a} {anc}\nCX {b} {anc}\nM {anc}\n{rec} rec[-1]\n"


def _two_pairs(faults, xx01="DETECTOR", second_round=True):
    """Two Bell pairs (0,1), (2,3), deterministic faults (p = 1) before
    T0 T_DAG1 T2 T_DAG3.  X0X1 is read (record kind ``xx01``) — twice when
    ``second_round`` — Z0Z1 and Z2Z3 are read; X2X3 is NEVER read (its coin
    must stay a coherent superposition in the retained state)."""
    t = "R 0 1 2 3\nH 0 2\nCX 0 1 2 3\n" + "".join(faults) + "T 0 2\nT_DAG 1 3\n"
    t += _xx_check(4, 0, 1, xx01)
    if second_round:
        t += _xx_check(5, 0, 1, "DETECTOR")
    t += _zz_check(6, 0, 1, "DETECTOR") + _zz_check(7, 2, 3, "DECISION(1)")
    return t + "OUTPUT_QUBITS out 0 1 2 3\n"


def _all_fired(text):
    k = sum(len(ln.split()) - 1 for ln in text.splitlines() if "_ERROR" in ln)
    return lambda buf, i: set(range(k))


def test_d2_repro_retained_state_is_conditioned_on_its_detector():
    # p = 0.5 on ONE location: a shot is faulted iff its residual plan is non-empty
    n_coin = _check(REPRO, 200, 3, lambda buf, i: {0} if len(buf.plan_key(i)) else set())
    assert n_coin > 60          # non-vacuous: ~half the shots carry the coin


def test_d2_repro_xx_matches_the_detector_exactly():
    """The owner's statement of the physics, verbatim: <X0X1> = 1 − 2·det."""
    S = _x.TwirlSampler(REPRO, 1.0, "", 0, False, "")
    n = 200
    buf = S.sample_barrier(n, 3)
    det = np.unpackbits(np.asarray(buf.dets(), np.uint8), axis=1, bitorder="little")[:, 0]
    ow = list(S.output_wires())
    xx = np.asarray(buf.pauli_expectations_all([(ow, [], 0)])).real[:, 0]
    assert np.allclose(xx, 1 - 2 * det.astype(float), atol=1e-9)
    assert 20 < det.sum() < 80       # ~half the shots faulted, half of those read −1


@pytest.mark.parametrize("faults", [
    ["X_ERROR(1) 0\n"],                         # coin on the READ X0X1 sector
    ["Y_ERROR(1) 1\n"],                         # Y fault: S-tail + Z part
    ["X_ERROR(1) 2\n"],                         # coin on the UNREAD X2X3 sector only
    ["X_ERROR(1) 0 2\n"],                       # both: one read, one coherent
    ["X_ERROR(1) 0\n", "Y_ERROR(1) 3\n"],
])
def test_two_pairs_retained_state_matches_dense_conditioned_state(faults):
    text = _two_pairs(faults)
    _check(text, 64, 11, _all_fired(text))


@pytest.mark.parametrize("kind", ["DECISION(0)", "OBSERVABLE_INCLUDE(0)"])
def test_record_kinds_decision_and_observable_condition_too(kind):
    """The coin reaches the record through a DECISION or an OBSERVABLE only."""
    text = _two_pairs(["X_ERROR(1) 0\n"], xx01=kind, second_round=False)
    n_coin = _check(text, 64, 5, _all_fired(text))
    assert n_coin == 64


def _born_after_coin():
    """X0X1 read with a DETECTOR, then Y0X1 read through an ancilla as a Born
    DECISION (ANTI to the certified group: it anticommutes with X0X1).  The
    decision is Born-measured on the retained state AFTER the record projection
    (materialize_shot step 3b): the per-shot state check covers the projected
    and then Born-collapsed state, and the joint (record, decision) statistics
    must stay the dense ones (P(dec | det) = 1/2).  This one passes on the
    pre-fix build too — the decision's own deferred wire makes its marginal
    equal its conditional here — so it guards the born path, not the defect."""
    cy = "S_DAG 0\nCX 8 0\nS 0\n"                    # controlled-Y(8 -> 0) = S·CX·S†
    return ("R 0 1\nH 0\nCX 0 1\nX_ERROR(1) 0\nT 0\nT_DAG 1\n" + _xx_check(4, 0, 1, "DETECTOR")
            + "R 8\nH 8\n" + cy + "CX 8 1\nH 8\nM 8\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n")


def test_born_decision_after_a_record_coin_is_conditioned():
    text = _born_after_coin()
    assert _check(text, 64, 9, _all_fired(text)) == 64
    S = _x.TwirlSampler(text, 1.0, "", 0, False, "")
    n = 4000
    buf = S.sample_barrier(n, 4)
    dec = np.unpackbits(np.asarray(buf.decisions(), np.uint8), axis=1, bitorder="little")[:, 0]
    det = np.unpackbits(np.asarray(buf.dets(), np.uint8), axis=1, bitorder="little")[:, 0]
    # dense: P(dec = 1 | det) = 1/2 for both records (5σ binomial bound per bucket)
    for d in (0, 1):
        k = int(det.sum()) if d else int(n - det.sum())
        f = dec[det == d].mean()
        assert abs(f - 0.5) < 5 * 0.5 / np.sqrt(k), (d, f, k)


def test_unread_check_stays_coherent():
    """Fault on pair (2,3) only: X2X3 is never read, so the retained state must
    keep the coherent residual state — <X2 Y3> = ±1 there, which ANY projection
    onto an X2X3 sector would zero."""
    text = _two_pairs(["X_ERROR(1) 2\n"])
    S = _x.TwirlSampler(text, 1.0, "", 0, False, "")
    buf = S.sample_barrier(16, 2)
    ow = list(S.output_wires())
    xy = np.asarray(buf.pauli_expectations_all([([ow[2], ow[3]], [ow[3]], 1)])).real[:, 0]
    assert np.allclose(np.abs(xy), 1.0, atol=1e-9)


def test_fault_free_shots_untouched():
    text = _two_pairs([])
    assert _check(text, 16, 1, lambda buf, i: set()) == 0
