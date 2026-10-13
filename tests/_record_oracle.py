"""Exact record-law oracle for small Stim circuits (test helper, no xtim code).

Enumerates every noise event and every measurement branch on dense pure states, so the
returned law is EXACT: a dict mapping the full record tuple (one bit per measurement, Stim
record order, recorded bits incl. `!` inverts) to its probability.  `parity_law` folds a
record law onto DETECTOR / OBSERVABLE_INCLUDE / DECISION parities.

Scope: the gate set the twirl audit repros use — R/RX/RY, H, S, S_DAG, T, T_DAG, X, Y, Z,
CX, CY, CZ, CH, CS, CS_DAG, CCZ, X/Y/Z_ERROR, PAULI_CHANNEL_1, M/MX/MY/MR (optionally `!q`),
MPAD, MPP (Pauli products), feedback `CX/CY/CZ rec[-k] q`.  Anything else raises.
"""
import re

import numpy as np

_X = np.array([[0, 1], [1, 0]], complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_Y = 1j * _X @ _Z
_H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
_P1 = {"X": _X, "Y": _Y, "Z": _Z}
_ONE = {"H": _H, "S": np.diag([1, 1j]), "S_DAG": np.diag([1, -1j]),
        "T": np.diag([1, np.exp(1j * np.pi / 4)]), "T_DAG": np.diag([1, np.exp(-1j * np.pi / 4)]),
        "X": _X, "Y": _Y, "Z": _Z, "I": np.eye(2, dtype=complex)}
# diagonal 2-/3-qubit gates: phase on the all-ones control pattern
_DIAG = {"CZ": -1.0, "CS": 1j, "CS_DAG": -1j}
_ERR = {"X_ERROR": "X", "Y_ERROR": "Y", "Z_ERROR": "Z"}


def _apply1(psi, n, g, q):
    v = np.moveaxis(psi.reshape([2] * n), q, 0)
    v = np.tensordot(g, v, axes=([1], [0]))
    return np.moveaxis(v, 0, q).reshape(-1)


def _bits(n):
    idx = np.arange(1 << n)
    return [(idx >> (n - 1 - q)) & 1 for q in range(n)]   # qubit 0 = most significant


def _pauli_apply(psi, n, word):
    for q, p in word:
        psi = _apply1(psi, n, _P1[p], q)
    return psi


def _project(psi, n, word, sign):
    """(1 + sign*P)/2 |psi>, unnormalised."""
    return 0.5 * (psi + sign * _pauli_apply(psi, n, word))


def _qubits(lines):
    m = -1
    for ln in lines:
        for t in ln.split()[1:]:
            if t.startswith("rec[") or t.startswith("out") or not t:
                continue
            for f in t.lstrip("!").split("*"):
                f = f.lstrip("!").lstrip("XYZ")
                if f.isdigit():
                    m = max(m, int(f))
    return m + 1


def _args(tok0):
    m = re.match(r"([A-Z_0-9]+)(?:\(([^)]*)\))?", tok0)
    name, a = m.group(1), m.group(2)
    return name, ([float(x) for x in a.split(",")] if a else [])


def record_law(text, max_branches=1 << 16):
    lines = [ln.split("#")[0].strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    n = _qubits(lines)
    psi0 = np.zeros(1 << n, complex)
    psi0[0] = 1.0
    branches = [(1.0, psi0, ())]           # (weight, normalised state, records)
    zb = _bits(n)

    def measure(brs, word, inv, reset_to=None):
        out = []
        for w, psi, rec in brs:
            for sign, bit in ((+1, 0), (-1, 1)):
                ph = _project(psi, n, word, sign)
                p = float(np.vdot(ph, ph).real)
                if p < 1e-14:
                    continue
                ph = ph / np.sqrt(p)
                if reset_to is not None and bit:
                    ph = _pauli_apply(ph, n, reset_to)   # flip back to the +1 eigenstate
                out.append((w * p, ph, rec + (bit ^ inv,)))
        return out

    for ln in lines:
        tok = ln.split()
        name, args = _args(tok[0])
        tg = tok[1:]
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "OUTPUT_QUBITS",
                    "PAULI_EXPECTATION", "QUBIT_COORDS", "TICK"):
            continue
        new = []
        if name in _ONE:
            for w, psi, rec in branches:
                for t in tg:
                    psi = _apply1(psi, n, _ONE[name], int(t))
                new.append((w, psi, rec))
        elif name in ("CX", "CY", "CZ") and any(t.startswith("rec[") for t in tg):
            for w, psi, rec in branches:
                for c, t in zip(tg[::2], tg[1::2]):
                    k = int(re.match(r"rec\[-(\d+)\]", c).group(1))
                    if rec[-k]:
                        psi = _apply1(psi, n, _P1[name[1]], int(t))
                new.append((w, psi, rec))
        elif name in ("CX", "CY", "CH"):
            g = _H if name == "CH" else _P1[name[1]]
            for w, psi, rec in branches:
                for c, t in zip(tg[::2], tg[1::2]):
                    c, t = int(c), int(t)
                    on = _project(psi, n, [(c, "Z")], -1)
                    psi = psi - on + _apply1(on, n, g, t)
                new.append((w, psi, rec))
        elif name in _DIAG or name == "CCZ":
            ph = -1.0 if name == "CCZ" else _DIAG[name]
            ar = 3 if name == "CCZ" else 2
            for w, psi, rec in branches:
                for i in range(0, len(tg), ar):
                    qs = [int(x) for x in tg[i:i + ar]]
                    mask = np.ones(1 << n, bool)
                    for q in qs:
                        mask &= zb[q] == 1
                    psi = psi.copy()
                    psi[mask] *= ph
                new.append((w, psi, rec))
        elif name in _ERR or name == "PAULI_CHANNEL_1":
            for t in tg:
                q = int(t)
                if name == "PAULI_CHANNEL_1":
                    chans = [(args[0], "X"), (args[1], "Y"), (args[2], "Z")]
                else:
                    chans = [(args[0], _ERR[name])]
                nxt = []
                for w, psi, rec in branches:
                    p_none = 1.0 - sum(p for p, _ in chans)
                    if p_none > 0:
                        nxt.append((w * p_none, psi, rec))
                    for p, P in chans:
                        if p > 0:
                            nxt.append((w * p, _apply1(psi, n, _P1[P], q), rec))
                branches = nxt
            new = branches
        elif name in ("R", "RX", "RY"):
            basis = {"R": "Z", "RX": "X", "RY": "Y"}[name]
            flip = {"Z": "X", "X": "Z", "Y": "X"}[basis]
            for t in tg:
                q = int(t)
                nxt = []
                for w, psi, rec in branches:
                    for sign in (+1, -1):
                        ph = _project(psi, n, [(q, basis)], sign)
                        p = float(np.vdot(ph, ph).real)
                        if p < 1e-14:
                            continue
                        ph = ph / np.sqrt(p)
                        if sign < 0:
                            ph = _apply1(ph, n, _P1[flip], q)
                        nxt.append((w * p, ph, rec))
                branches = nxt
            new = branches
        elif name in ("M", "MX", "MY", "MR"):
            basis = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z"}[name]
            for t in tg:
                inv = 1 if t.startswith("!") else 0
                q = int(t.lstrip("!"))
                branches = measure(branches, [(q, basis)], inv,
                                   reset_to=[(q, "X")] if name == "MR" else None)
            new = branches
        elif name == "MPAD":
            new = [(w, psi, rec + tuple(int(t) for t in tg)) for w, psi, rec in branches]
        elif name == "MPP":
            for t in tg:
                inv = 1 if t.startswith("!") else 0
                word = [(int(f[1:]), f[0]) for f in t.lstrip("!").split("*")]
                branches = measure(branches, word, inv)
            new = branches
        else:
            raise ValueError(f"record_law: unsupported instruction {name!r}")
        branches = new
        if len(branches) > max_branches:
            raise ValueError("record_law: branch cap exceeded")
    law = {}
    for w, _, rec in branches:
        law[rec] = law.get(rec, 0.0) + w
    return law


def parities(text):
    """(detectors, observables{k: recs}, decisions{k: recs}) as absolute record indices."""
    dets, obs, decs = [], {}, {}
    nm = 0
    for ln in text.splitlines():
        tok = ln.split("#")[0].split()
        if not tok:
            continue
        name, args = _args(tok[0])
        if name in ("M", "MX", "MY", "MR", "MPAD", "MPP"):
            nm += len(tok) - 1
            continue
        recs = [nm - int(k) for k in re.findall(r"rec\[-(\d+)\]", ln)]
        if name == "DETECTOR":
            dets.append(recs)
        elif name == "OBSERVABLE_INCLUDE":
            obs.setdefault(int(args[0]), []).extend(recs)
        elif name == "DECISION":
            decs.setdefault(int(args[0]), []).extend(recs)
    return dets, obs, decs


def parity_law(text, columns):
    """Exact law of the given record-parity columns (list of record-index lists)."""
    out = {}
    for rec, p in record_law(text).items():
        key = tuple(sum(rec[i] for i in col) & 1 for col in columns)
        out[key] = out.get(key, 0.0) + p
    return out


def max_z(samples, law):
    """Largest per-cell |z| of the empirical joint `samples` (shots x cols, 0/1) vs `law`,
    plus the count of samples landing in a zero-probability cell."""
    N = len(samples)
    keys, cnt = np.unique(np.asarray(samples, np.uint8), axis=0, return_counts=True)
    emp = {tuple(int(x) for x in k): int(c) for k, c in zip(keys, cnt)}
    worst, impossible = 0.0, 0
    for k in set(emp) | set(law):
        p = law.get(k, 0.0)
        c = emp.get(k, 0)
        if p < 1e-15:
            impossible += c
            continue
        z = (c - N * p) / np.sqrt(N * p * (1 - p)) if p < 1 else 0.0
        worst = max(worst, abs(z))
    return worst, impossible
