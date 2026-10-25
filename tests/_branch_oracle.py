"""Independent dense branch oracle (test helper, no xtim code; written for the
fix/feedback-after-t review and adopted as the feedback-hoist regression oracle).

Stim semantics: M(p) collapses to the TRUE outcome, the RECORD is flipped w.p. p;
`!q` inverts the record; MR/MRX/MRY measure then reset; feedback reads the RECORD.
Supports REPEAT (textual unroll), [tags], DEPOLARIZE1/2, PAULI_CHANNEL_1, X/Y/Z_ERROR,
MPP, CX/CY/CZ (+rec), CS/CS_DAG/CCZ/CH, 1q Cliffords, T/T_DAG.
Returns branches (w, psi, rec)."""
import itertools
import re

import numpy as np

X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)
Y = 1j * X @ Z
H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
P1 = {"X": X, "Y": Y, "Z": Z, "I": np.eye(2, dtype=complex)}
ONE = {"H": H, "S": np.diag([1, 1j]), "S_DAG": np.diag([1, -1j]), "SQRT_Z": np.diag([1, 1j]),
       "T": np.diag([1, np.exp(1j * np.pi / 4)]), "T_DAG": np.diag([1, np.exp(-1j * np.pi / 4)]),
       "X": X, "Y": Y, "Z": Z, "I": np.eye(2, dtype=complex),
       "SQRT_X": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
       "SQRT_X_DAG": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]),
       "H_YZ": np.array([[1, -1j], [1j, -1]]) / np.sqrt(2)}
DIAG = {"CZ": -1.0, "CS": 1j, "CS_DAG": -1j}
BAS = {"M": "Z", "MX": "X", "MY": "Y", "MZ": "Z", "MR": "Z", "MRZ": "Z", "MRX": "X", "MRY": "Y"}


def unroll(text):
    lines = [ln.split("#")[0].rstrip() for ln in text.splitlines()]
    out, stack = [], [[]]
    reps = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"REPEAT\s+(\d+)\s*\{", s)
        if m:
            reps.append(int(m.group(1)))
            stack.append([])
            continue
        if s == "}":
            body = stack.pop()
            stack[-1].extend(body * reps.pop())
            continue
        stack[-1].append(strip_tag(s))
    return stack[0]


def strip_tag(s):
    tok = s.split(None, 1)
    head = re.sub(r"^([A-Z_0-9]+)\[[^\]]*\]", r"\1", tok[0])
    return head + ((" " + tok[1]) if len(tok) > 1 else "")


def apply1(psi, n, g, q):
    v = np.moveaxis(psi.reshape([2] * n), q, 0)
    v = np.tensordot(g, v, axes=([1], [0]))
    return np.moveaxis(v, 0, q).reshape(-1)


def bits(n):
    idx = np.arange(1 << n)
    return [(idx >> (n - 1 - q)) & 1 for q in range(n)]


def pword(psi, n, word):
    for q, p in word:
        psi = apply1(psi, n, P1[p], q)
    return psi


def proj(psi, n, word, sign):
    return 0.5 * (psi + sign * pword(psi, n, word))


def nqubits(lines):
    m = -1
    for ln in lines:
        for t in ln.split()[1:]:
            if t.startswith("rec[") or not t:
                continue
            for f in t.lstrip("!").split("*"):
                f = f.lstrip("!").lstrip("XYZ")
                if f.isdigit():
                    m = max(m, int(f))
    return m + 1


def args_of(tok0):
    m = re.match(r"([A-Z_0-9]+)(?:\(([^)]*)\))?", tok0)
    return m.group(1), ([float(x) for x in m.group(2).split(",")] if m.group(2) else [])


def run(text, max_branches=1 << 18):
    lines = unroll(text)
    n = nqubits(lines)
    psi0 = np.zeros(1 << n, complex)
    psi0[0] = 1
    br = [(1.0, psi0, ())]
    zb = bits(n)
    outq = None

    def measure(brs, word, inv, p_flip=0.0, reset=None):
        out = []
        for w, psi, rec in brs:
            for sign, bit in ((+1, 0), (-1, 1)):
                ph = proj(psi, n, word, sign)
                pr = float(np.vdot(ph, ph).real)
                if pr < 1e-14:
                    continue
                ph = ph / np.sqrt(pr)
                if reset is not None and bit:
                    ph = pword(ph, n, reset)
                for flip, pf in ((0, 1 - p_flip), (1, p_flip)):
                    if pf > 0:
                        out.append((w * pr * pf, ph, rec + (bit ^ inv ^ flip,)))
        return out

    def mix(brs, chans):  # chans: list of (p, word)
        nxt = []
        for w, psi, rec in brs:
            p0 = 1 - sum(p for p, _ in chans)
            if p0 > 1e-15:
                nxt.append((w * p0, psi, rec))
            for p, word in chans:
                if p > 0:
                    nxt.append((w * p, pword(psi, n, word), rec))
        return nxt

    for ln in lines:
        tok = ln.split()
        name, a = args_of(tok[0])
        tg = tok[1:]
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "PAULI_EXPECTATION", "QUBIT_COORDS", "TICK"):
            continue
        if name == "OUTPUT_QUBITS":
            outq = [int(t) for t in tg if t.isdigit()]
            continue
        if name in ONE:
            br = [(w, _seq1(psi, n, ONE[name], tg), r) for w, psi, r in br]
        elif name in ("CX", "CY", "CZ") and any(t.startswith("rec[") for t in tg):
            nb = []
            for w, psi, rec in br:
                for c, t in zip(tg[::2], tg[1::2]):
                    if c.startswith("rec["):
                        k = int(re.match(r"rec\[-(\d+)\]", c).group(1))
                        if rec[-k]:
                            psi = apply1(psi, n, P1[name[1]], int(t))
                    else:
                        raise ValueError("mixed")
                nb.append((w, psi, rec))
            br = nb
        elif name in ("CX", "CY", "CH", "CNOT"):
            g = H if name == "CH" else P1["X" if name == "CNOT" else name[1]]
            nb = []
            for w, psi, rec in br:
                for c, t in zip(tg[::2], tg[1::2]):
                    on = proj(psi, n, [(int(c), "Z")], -1)
                    psi = psi - on + apply1(on, n, g, int(t))
                nb.append((w, psi, rec))
            br = nb
        elif name in DIAG or name == "CCZ":
            ph = -1.0 if name == "CCZ" else DIAG[name]
            ar = 3 if name == "CCZ" else 2
            nb = []
            for w, psi, rec in br:
                for i in range(0, len(tg), ar):
                    mask = np.ones(1 << n, bool)
                    for q in tg[i:i + ar]:
                        mask &= zb[int(q)] == 1
                    psi = psi.copy()
                    psi[mask] *= ph
                nb.append((w, psi, rec))
            br = nb
        elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            for t in tg:
                br = mix(br, [(a[0], [(int(t), name[0])])])
        elif name == "PAULI_CHANNEL_1":
            for t in tg:
                br = mix(br, [(a[0], [(int(t), "X")]), (a[1], [(int(t), "Y")]), (a[2], [(int(t), "Z")])])
        elif name == "DEPOLARIZE1":
            for t in tg:
                br = mix(br, [(a[0] / 3, [(int(t), P)]) for P in "XYZ"])
        elif name == "DEPOLARIZE2":
            for c, t in zip(tg[::2], tg[1::2]):
                ch = []
                for P, Q in itertools.product("IXYZ", repeat=2):
                    if P == Q == "I":
                        continue
                    ch.append((a[0] / 15, [(int(c), P), (int(t), Q)]))
                br = mix(br, ch)
        elif name in ("R", "RZ", "RX", "RY"):
            b = {"R": "Z", "RZ": "Z", "RX": "X", "RY": "Y"}[name]
            flip = {"Z": "X", "X": "Z", "Y": "X"}[b]
            for t in tg:
                q = int(t)
                nb = []
                for w, psi, rec in br:
                    for sign in (+1, -1):
                        ph = proj(psi, n, [(q, b)], sign)
                        pr = float(np.vdot(ph, ph).real)
                        if pr < 1e-14:
                            continue
                        ph = ph / np.sqrt(pr)
                        if sign < 0:
                            ph = apply1(ph, n, P1[flip], q)
                        nb.append((w * pr, ph, rec))
                br = nb
        elif name in BAS:
            b = BAS[name]
            pf = a[0] if a else 0.0
            for t in tg:
                inv = 1 if t.startswith("!") else 0
                q = int(t.lstrip("!"))
                reset = None
                if name.startswith("MR"):
                    reset = [(q, {"Z": "X", "X": "Z", "Y": "X"}[b])]
                br = measure(br, [(q, b)], inv, pf, reset)
        elif name == "MPP":
            pf = a[0] if a else 0.0
            for t in tg:
                inv = 1 if t.startswith("!") else 0
                word = [(int(f.lstrip("!")[1:]), f.lstrip("!")[0]) for f in t.lstrip("!").split("*")]
                br = measure(br, word, inv, pf)
        else:
            raise ValueError(f"oracle: unsupported {name!r}")
        if len(br) > max_branches:
            raise ValueError("branch cap")
    return br, n, outq


def _seq1(psi, n, g, tg):
    for t in tg:
        psi = apply1(psi, n, g, int(t))
    return psi


def decisions(text):
    """{k: [abs record indices]} for DECISION(k), counting records across REPEAT/MPP/MR."""
    nm = 0
    decs = {}
    for ln in unroll(text):
        tok = ln.split()
        name, a = args_of(tok[0])
        if name in BAS or name in ("MPP", "MPAD"):
            nm += len(tok) - 1
            continue
        if name == "DECISION":
            decs.setdefault(int(a[0]), []).extend(nm - int(k) for k in re.findall(r"rec\[-(\d+)\]", ln))
    return [decs[k] for k in sorted(decs)]


def cond_table(text, words):
    """-> {dec_tuple: (prob, np.array of E[<P_w>|dec] over words)} with words over OUTPUT order."""
    br, n, outq = run(text)
    cols = decisions(text)
    acc = {}
    for w, psi, rec in br:
        key = tuple(sum(rec[i] for i in c) & 1 for c in cols)
        ex = []
        for word in words:
            v = psi
            for q, p in zip(outq or [], word):
                if p != "I":
                    v = apply1(v, n, P1[p], q)
            ex.append(np.vdot(psi, v).real)
        pr, e = acc.get(key, (0.0, np.zeros(len(words))))
        acc[key] = (pr + w, e + w * np.array(ex))
    return {k: (p, e / p) for k, (p, e) in acc.items()}
