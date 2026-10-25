"""Exact record-law oracle on DENSITY MATRICES (test helper, no xtim code) — J-a N4a, G-12.

tests/_record_oracle.py enumerates every noise event as a separate pure-state branch, which is exact but
exponential in the number of noisy locations: the adv-A fuzz corpus (a depolarizing channel after every gate)
is out of its reach. This module evolves, per distinct record prefix, the UNNORMALISED density matrix of the
branches sharing it (Kraus sums for the noise channels, projectors for the reads), so its cost is linear in
the number of noisy locations. It reads the ORIGINAL text with the same gate semantics: every gate matrix,
the instruction set and the parsing helpers are _record_oracle's (imported, not copied); the noiseless
reference sample is _record_oracle.reference_records. `record_law_mixed(text, readout=True)` equals
`_record_oracle.record_law(text, readout=True)` wherever the latter is feasible
(tests/test_ja_ppr_records.py checks it at 1e-12).

rho (2^n x 2^n) is stored as a 2n-qubit vector: ket wire q is axis q, bra wire q is axis n + q; an operator A
acts as rho -> A rho A^dagger, i.e. A on the ket axes and conj(A) on the bra axes.
"""
import re

import numpy as np

from _record_oracle import (_DIAG, _ERR, _H, _ONE, _P1, _apply1, _args, _bits, _qubits, emitted_columns,
                            reference_records)


def record_law_mixed(text, *, readout=False):
    """Exact law {record tuple: probability} of `text` (noise channels as Kraus sums)."""
    lines = [ln.split("#")[0].strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    n = _qubits(lines)
    N2 = 2 * n
    rho0 = np.zeros(1 << N2, complex)
    rho0[0] = 1.0
    zb = _bits(N2)
    branches = {(): rho0}                    # record prefix -> unnormalised rho (trace = probability)

    def op1(v, g, q):                        # g on wire q: ket and bra
        return _apply1(_apply1(v, N2, g, q), N2, np.conj(g), n + q)

    def word_op(v, word, side):              # a Pauli word on one side (side 0 ket, 1 bra)
        for q, P in word:
            g = _P1[P] if side == 0 else np.conj(_P1[P])
            v = _apply1(v, N2, g, q + side * n)
        return v

    def project(v, word, sign):              # (1 + sign W)/2 rho (1 + sign W)/2
        v = 0.5 * (v + sign * word_op(v, word, 0))
        return 0.5 * (v + sign * word_op(v, word, 1))

    def trace(v):
        return float(np.real(np.trace(v.reshape(1 << n, 1 << n))))

    def pauli_both(v, word):
        return word_op(word_op(v, word, 0), word, 1)

    def add(d, rec, v):
        if rec in d:
            d[rec] = d[rec] + v
        else:
            d[rec] = v

    def measure(word, inv, reset_to=None, flip_p=0.0):
        nonlocal branches
        out = {}
        for rec, v in branches.items():
            for sign, bit in ((+1, 0), (-1, 1)):
                ph = project(v, word, sign)
                if trace(ph) < 1e-15:
                    continue
                if reset_to is not None and bit:
                    ph = pauli_both(ph, reset_to)
                if readout and flip_p > 0:
                    add(out, rec + (bit ^ inv,), (1 - flip_p) * ph)
                    add(out, rec + (bit ^ inv ^ 1,), flip_p * ph)
                else:
                    add(out, rec + (bit ^ inv,), ph)
        branches = out

    def controlled(v, c, t, g, side):        # ket (side 0) or bra (side 1) map of a controlled-g gate
        gg = g if side == 0 else np.conj(g)
        cz = _P1["Z"]
        on = 0.5 * (v - _apply1(v, N2, cz, c + side * n))     # (1 - Z_c)/2 is real: same on both sides
        return v - on + _apply1(on, N2, gg, t + side * n)

    def channel(v, chans):                   # sum_k p_k P_k rho P_k + p_none rho
        p_none = 1.0 - sum(p for p, _ in chans)
        out = p_none * v if p_none > 0 else np.zeros_like(v)
        for p, word in chans:
            if p > 0:
                out = out + p * pauli_both(v, word)
        return out

    for ln in lines:
        tok = ln.split()
        name, args = _args(tok[0])
        tg = tok[1:]
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "OUTPUT_QUBITS",
                    "INPUT_QUBITS", "PAULI_EXPECTATION", "QUBIT_COORDS", "TICK"):
            continue
        if name in _ONE:
            branches = {r: _iter(v, tg, lambda w, t: op1(w, _ONE[name], int(t))) for r, v in branches.items()}
        elif name in ("CX", "CY", "CZ") and any(t.startswith("rec[") for t in tg):
            nb = {}
            for rec, v in branches.items():
                for c, t in zip(tg[::2], tg[1::2]):
                    k = int(re.match(r"rec\[-(\d+)\]", c).group(1))
                    if rec[-k]:
                        v = op1(v, _P1[name[1]], int(t))
                nb[rec] = v
            branches = nb
        elif name in ("CX", "CY", "CH"):
            g = _H if name == "CH" else _P1[name[1]]
            nb = {}
            for rec, v in branches.items():
                for c, t in zip(tg[::2], tg[1::2]):
                    v = controlled(controlled(v, int(c), int(t), g, 0), int(c), int(t), g, 1)
                nb[rec] = v
            branches = nb
        elif name in _DIAG or name == "CCZ":
            ph = -1.0 if name == "CCZ" else _DIAG[name]
            ar = 3 if name == "CCZ" else 2
            nb = {}
            for rec, v in branches.items():
                for i in range(0, len(tg), ar):
                    qs = [int(x) for x in tg[i:i + ar]]
                    mk, mb = np.ones(1 << N2, bool), np.ones(1 << N2, bool)
                    for q in qs:
                        mk &= zb[q] == 1
                        mb &= zb[n + q] == 1
                    v = v.copy()
                    v[mk] *= ph
                    v[mb] *= np.conj(ph)
                nb[rec] = v
            branches = nb
        elif name in _ERR or name in ("PAULI_CHANNEL_1", "DEPOLARIZE1"):
            for t in tg:
                q = int(t)
                if name == "PAULI_CHANNEL_1":
                    chans = [(args[0], "X"), (args[1], "Y"), (args[2], "Z")]
                elif name == "DEPOLARIZE1":
                    chans = [(args[0] / 3, "X"), (args[0] / 3, "Y"), (args[0] / 3, "Z")]
                else:
                    chans = [(args[0], _ERR[name])]
                chans = [(p, [(q, P)]) for p, P in chans]
                branches = {r: channel(v, chans) for r, v in branches.items()}
        elif name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
            letters = "IXYZ"
            for a, b in zip(tg[::2], tg[1::2]):
                a, b = int(a), int(b)
                chans = []
                for t in range(15):
                    p = args[0] / 15 if name == "DEPOLARIZE2" else args[t]
                    Pa, Pb = letters[(t + 1) >> 2], letters[(t + 1) & 3]
                    word = [(a, Pa)] if Pa != "I" else []
                    word += [(b, Pb)] if Pb != "I" else []
                    chans.append((p, word))
                branches = {r: channel(v, chans) for r, v in branches.items()}
        elif name in ("R", "RX", "RY"):
            basis = {"R": "Z", "RX": "X", "RY": "Y"}[name]
            flip = {"Z": "X", "X": "Z", "Y": "X"}[basis]
            for t in tg:
                q = int(t)
                branches = {r: project(v, [(q, basis)], +1) + pauli_both(project(v, [(q, basis)], -1), [(q, flip)])
                            for r, v in branches.items()}
        elif name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            basis = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}[name]
            flip = {"Z": "X", "X": "Z", "Y": "X"}[basis]
            for t in tg:
                inv = 1 if t.startswith("!") else 0
                q = int(t.lstrip("!"))
                measure([(q, basis)], inv, reset_to=[(q, flip)] if name.startswith("MR") else None,
                        flip_p=args[0] if args else 0.0)
        elif name == "MPAD":
            branches = {r + tuple(int(t) for t in tg): v for r, v in branches.items()}
        elif name == "MPP":
            for t in tg:
                inv = 1 if t.startswith("!") else 0
                word = [(int(f[1:]), f[0]) for f in t.lstrip("!").split("*")]
                measure(word, inv, flip_p=args[0] if args else 0.0)
        else:
            raise ValueError(f"record_law_mixed: unsupported instruction {name!r}")
    law = {}
    for rec, v in branches.items():
        p = trace(v)
        if p > 0:
            law[rec] = law.get(rec, 0.0) + p
    return law


def _iter(v, targets, f):
    for t in targets:
        v = f(v, t)
    return v


def emitted_law_mixed(text):
    """_record_oracle.emitted_law (readout flips honoured; detector / observable parities relative to the
    noiseless reference, decision parities raw), from record_law_mixed."""
    cols = emitted_columns(text)
    ref = reference_records(text)
    off = [sum(ref[i] for i in recs) & 1 if kind != "decision" else 0 for kind, _, recs in cols]
    out = {}
    for rec, p in record_law_mixed(text, readout=True).items():
        key = tuple((sum(rec[i] for i in recs) + o) & 1 for (_, _, recs), o in zip(cols, off))
        out[key] = out.get(key, 0.0) + p
    return out
