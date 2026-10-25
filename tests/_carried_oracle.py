"""J-b1 P9b test oracle (dense, n <= ~12): the consumer's bare on a carried ideal state, by direct statevector
evolution — the producer's post-barrier ideal state (materialize of an identity-content record at z = 0) tensored
with |0> on the consumer's fresh wires, then the consumer's family-normalized deferred Gate stream
(_xtim_testapi._deferred_stream) with its wires mapped as the injection maps them (port j -> the ideal state's
j-th OUTPUT_QUBITS wire, fresh w -> n_producer + w). Qubit 0 is the most significant bit (xtim.frames.dense_state)."""
import re

import numpy as np

from xtim import _xtim_testapi as TA
from xtim.frames import _pauli_apply

_S = np.exp(1j * np.pi / 4)
G1 = {0: np.array([[0, 1], [1, 0]], complex), 1: np.array([[0, -1j], [1j, 0]], complex),
      2: np.diag([1, -1]).astype(complex), 3: np.diag([1, 1j]), 4: np.diag([1, -1j]),
      7: np.array([[1, 1], [1, -1]], complex) / np.sqrt(2), 8: np.diag([1, _S])}
# GateKind: X Y Z S SDG CZ CX H T CS CCZ CH


def apply1(psi, n, q, U):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def apply_ctrl(psi, n, ctrls, t, U):
    """U on t when every control is 1."""
    out = psi.copy().reshape([2] * n)
    idx = [slice(None)] * n
    for c in ctrls:
        idx[c] = 1
    sub = out[tuple(idx)]
    # t's axis index inside sub: count the remaining axes before t
    ax = t - sum(1 for c in ctrls if c < t)
    sub = np.moveaxis(np.tensordot(U, sub, axes=([1], [ax])), 0, ax)
    out[tuple(idx)] = sub
    return out.reshape(-1)


def apply_gate(psi, n, kind, tg):
    if kind in G1:
        return apply1(psi, n, tg[0], G1[kind])
    if kind == 5:                                     # CZ
        return apply_ctrl(psi, n, [tg[0]], tg[1], G1[2])
    if kind == 6:                                     # CX
        return apply_ctrl(psi, n, [tg[0]], tg[1], G1[0])
    if kind == 9:                                     # CS
        return apply_ctrl(psi, n, [tg[0]], tg[1], G1[3])
    if kind == 10:                                    # CCZ
        return apply_ctrl(psi, n, [tg[0], tg[1]], tg[2], G1[2])
    if kind == 11:                                    # CH
        return apply_ctrl(psi, n, [tg[0]], tg[1], G1[7])
    raise ValueError("gate kind %d" % kind)


def dense_state(fr):
    """The statevector of a frames export, matrix-free (O(n 2^n)): |ref> by projecting a seeded Gaussian onto the
    signed generators' joint +1 eigenspace, then sum_i c_i D_i |ref>."""
    n = int(fr["n"])
    rng = np.random.default_rng(20261020)
    for _ in range(4):
        v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
        for a in range(n):
            v = (v + _pauli_apply(v, n, fr["stab_x"][a], fr["stab_z"][a], int(fr["stab_phase"][a]))) / 2.0
        if np.linalg.norm(v) > 1e-9:
            break
    ref = v / np.linalg.norm(v)
    psi = np.zeros(1 << n, complex)
    c = np.asarray(fr["coeff"], complex)
    for i in range(c.shape[0]):
        psi += c[i] * _pauli_apply(ref, n, fr["branch_x"][i], fr["branch_z"][i], int(fr["branch_phase"][i]))
    return psi


def ideal_record(R, H):
    """A record of the ideal state: identity content, P_rec = I and every outcome (u ‖ z) 0."""
    for i in range(R.num_records):
        ct = R.content(i)
        x, z = R.prec(i)
        if (R.kind(i) == 0 and not any(ct[:-1]) and not any(bytes(H[i]["z"])) and not np.any(x) and not np.any(z)
                and not np.any(np.asarray(R.outcomes(i)))):
            return i
    return None


def ideal_dense(prod, R, H):
    i = ideal_record(R, H)
    if i is None:
        raise AssertionError("no ideal record (identity content, P_rec = I, outcomes 0)")
    fr = prod.materialize(R, i)
    v = dense_state(fr)
    return v / np.linalg.norm(v), int(fr["n"])


def consumer_oracle(prod_vec, n_p, wires, ctext):
    m = re.search(r"INPUT_QUBITS\s+\w+((?:\s+\d+)+)", ctext)
    ports = [int(x) for x in m.group(1).split()]
    nc, stream = TA._deferred_stream(ctext, 1.0)
    mp = {p: wires[j] for j, p in enumerate(ports)}
    n = n_p + nc
    psi = np.kron(prod_vec, np.eye(1, 1 << nc, 0).reshape(-1))
    for ins in stream:
        if ins["kind"] != 0:                          # Gate only (the bare is the noiseless pre-measurement state)
            continue
        tg = [mp.get(q, n_p + q) for q in ins["targets"]]
        psi = apply_gate(psi, n, ins["gate"], tg)
    return psi, n


def bare_dense(consumer):
    v = dense_state(TA.bare_frames(consumer))
    return v / np.linalg.norm(v)
