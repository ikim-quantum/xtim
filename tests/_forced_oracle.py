"""Independent dense-statevector oracle for xtim's forced-fault exact branches.

Own numpy, no xtim code: a fixed-width statevector over the circuit's qubits, branching
EXACTLY on every measurement outcome whose conditional probability exceeds 1e-12. Faults are
applied as Pauli gates at their noise site (every unforced noise site is the identity); a record
flip flips the RECORDED bit only. Classically-controlled feedback (CX/CY/CZ rec[-k] q) reads the
recorded bit. Returns, per branch keyed by its recorded bits: probability, raw detector /
observable parities (NOT reference-relative), decision parities, and the conditional
PAULI_EXPECTATION values at the end of the circuit (declared frame records applied).

Supported: R (only before any other use of the qubit), M MX MY MR MRX (with !q and M(p)), H S
S_DAG T T_DAG X Y Z CX CZ CS CCZ, feedback CX/CY/CZ rec[-k] q, the seven noise channels,
DETECTOR, OBSERVABLE_INCLUDE, DECISION, PAULI_EXPECTATION(i) [rec frame], TICK.
"""
import math
import re

import numpy as np

EPS = 1e-12
_S2 = 1 / math.sqrt(2)
G1 = {
    "H": np.array([[_S2, _S2], [_S2, -_S2]], complex),
    "S": np.diag([1, 1j]).astype(complex),
    "S_DAG": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * math.pi / 4)]).astype(complex),
    "T_DAG": np.diag([1, np.exp(-1j * math.pi / 4)]).astype(complex),
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.diag([1, -1]).astype(complex),
    "I": np.eye(2, dtype=complex),
}
NOISE1 = ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1")
NOISE2 = ("DEPOLARIZE2", "PAULI_CHANNEL_2")


def _apply1(psi, U, q, n):
    t = psi.reshape([2] * n)
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def _bits(n):
    idx = np.arange(2 ** n)
    return [(idx >> (n - 1 - q)) & 1 for q in range(n)]      # qubit 0 = most significant


def _diag(psi, phase):
    return psi * phase


def _pauli_vec(psi, ps, n):
    for q, ch in ps:
        psi = _apply1(psi, G1[ch], q, n)
    return psi


def _parse(line):
    m = re.match(r"^([A-Z_0-9]+)(?:\(([^)]*)\))?\s*(.*)$", line.strip())
    args = [float(a) for a in m.group(2).split(",")] if m.group(2) else []
    return m.group(1), args, m.group(3).split()


def branches(text, n, faults=(), record_flips=()):
    """faults: {(site, target): pauli string}; record_flips: set of absolute records."""
    faults = dict(faults)
    record_flips = set(record_flips)
    B = _bits(n)
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    br = [(psi, [])]           # (unnormalized state, recorded bits); |psi|^2 = probability
    dets, obs, decs, pes = [], {}, [], []
    site = 0
    started = set()
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("TICK"):
            continue
        name, args, toks = _parse(line)
        nrec = len(br[0][1])
        if name in ("DETECTOR", "DECISION", "OBSERVABLE_INCLUDE"):
            recs = [nrec + int(t[4:-1]) for t in toks]
            if name == "DETECTOR":
                dets.append(recs)
            elif name == "DECISION":
                decs.append(recs)
            else:
                obs.setdefault(int(args[0]), []).extend(recs)
            continue
        if name == "PAULI_EXPECTATION":
            frame = [nrec + int(t[4:-1]) for t in toks if t.startswith("rec[")]
            ps = [(int(t[1:]), t[0]) for t in toks if not t.startswith("rec[")
                  for t in t.split("*")]
            pes.append((ps, frame))
            continue
        if name in NOISE1 or name in NOISE2:
            two = name in NOISE2
            qs = [int(t) for t in toks]
            groups = [qs[i:i + 2] for i in range(0, len(qs), 2)] if two else [[q] for q in qs]
            for t, g in enumerate(groups):
                p = faults.get((site, t))
                if p:
                    br = [(_pauli_vec(v, [(q, c) for q, c in zip(g, p) if c != "I"], n), r)
                          for v, r in br]
            site += 1
            continue
        if name == "R":
            for t in toks:
                assert int(t) not in started, "oracle: R only on fresh qubits"
            continue
        if name in ("CX", "CY", "CZ") and toks and toks[0].startswith("rec["):
            nb = []
            for i in range(0, len(toks), 2):
                k = int(toks[i][4:-1])
                q = int(toks[i + 1])
                started.add(q)
                nb = []
                for v, r in br:
                    if r[len(r) + k]:
                        v = _apply1(v, G1[name[1]], q, n)
                    nb.append((v, r))
                br = nb
            continue
        if name in G1:
            for t in toks:
                started.add(int(t))
                br = [(_apply1(v, G1[name], int(t), n), r) for v, r in br]
            continue
        if name in ("CX", "CZ", "CS", "CCZ"):
            ar = 3 if name == "CCZ" else 2
            qs = [int(t) for t in toks]
            for i in range(0, len(qs), ar):
                g = qs[i:i + ar]
                started.update(g)
                if name == "CX":
                    c, tq = g
                    nb = []
                    for v, r in br:
                        flip = _apply1(v, G1["X"], tq, n)
                        nb.append((np.where(B[c] == 1, flip, v), r))
                    br = nb
                else:
                    on = np.ones(2 ** n, bool)
                    for q in g:
                        on &= B[q] == 1
                    ph = np.where(on, 1j if name == "CS" else -1, 1)
                    br = [(v * ph, r) for v, r in br]
            continue
        if name in ("M", "MX", "MY", "MR", "MRX"):
            basis = "X" if name in ("MX", "MRX") else ("Y" if name == "MY" else "Z")
            reset = name.startswith("MR")
            for t in toks:
                inv = t.startswith("!")
                q = int(t.lstrip("!"))
                started.add(q)
                nb = []
                for v, r in br:
                    w = v
                    if basis == "X":
                        w = _apply1(w, G1["H"], q, n)
                    elif basis == "Y":
                        w = _apply1(_apply1(w, G1["S_DAG"], q, n), G1["H"], q, n)
                    tot = np.vdot(v, v).real
                    for b in (0, 1):
                        u = np.where(B[q] == b, w, 0)
                        p = np.vdot(u, u).real
                        if p <= EPS * tot:
                            continue
                        if basis == "X":
                            u = _apply1(u, G1["H"], q, n)
                        elif basis == "Y":
                            u = _apply1(_apply1(u, G1["H"], q, n), G1["S"], q, n)
                        if reset and b:
                            u = _apply1(u, G1["Z" if basis == "X" else "X"], q, n)
                        rec = b ^ inv ^ (len(r) in record_flips)
                        nb.append((u, r + [rec]))
                br = nb
            continue
        raise ValueError(f"oracle: unsupported line {line!r}")
    out = {}
    for v, r in br:
        p = np.vdot(v, v).real
        vn = v / math.sqrt(p)
        ex = []
        for ps, frame in pes:
            val = np.vdot(vn, _pauli_vec(vn, ps, n)).real
            if sum(r[j] for j in frame) & 1:
                val = -val
            ex.append(val)
        out[tuple(r)] = {
            "p": p,
            "det": [sum(r[j] for j in d) & 1 for d in dets],
            "obs": [sum(r[j] for j in obs[k]) & 1 for k in range(max(obs, default=-1) + 1)
                    ] if obs else [],
            "dec": [sum(r[j] for j in d) & 1 for d in decs],
            "exp": ex,
        }
    return out
