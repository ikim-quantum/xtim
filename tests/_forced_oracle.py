"""Independent dense-statevector oracle for xtim's forced-fault exact branches.

Own numpy, no xtim code: a fixed-width statevector over the circuit's qubits, branching
EXACTLY on every measurement outcome whose conditional probability exceeds 1e-12. Faults are
applied as Pauli gates at their noise site (every unforced noise site is the identity); a record
flip flips the RECORDED bit only. Classically-controlled feedback (CX/CY/CZ rec[-k] q) reads the
recorded bit. Returns, per branch keyed by its recorded bits: probability, raw detector /
observable parities (NOT reference-relative), decision parities, and the conditional
PAULI_EXPECTATION values at the end of the circuit (declared frame records applied).

Supported: R RX RY (only before any other use of the qubit), M MX MY MR MRX MRY (with !q and
M(p)), H S S_DAG T T_DAG X Y Z CX CZ CS CCZ, any other unitary 1-/2-qubit Clifford gate stim
knows (its unitary from stim's tableau, imported lazily), feedback CX/CY/CZ rec[-k] q, the seven
noise channels, DETECTOR, OBSERVABLE_INCLUDE, DECISION, PAULI_EXPECTATION(i) [rec frame], TICK.

`aggregate` / `law_shifts` compare a factored forced-fault law (expanded view) with these
branches aggregated by (detectors, observables, decisions), conditional expectations
probability-weighted, up to one constant detector/observable reference XOR per circuit.
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


_STIM_U = {}


def _stim_unitary(name):
    """(arity, unitary) of a stim Clifford gate, qubit 0 = most significant (stim 'big')."""
    if name not in _STIM_U:
        import stim
        try:
            g = stim.gate_data(name)
        except (IndexError, ValueError):
            _STIM_U[name] = None
            return None
        if not g.is_unitary:
            _STIM_U[name] = None
            return None
        ar = 2 if g.is_two_qubit_gate else 1
        c = stim.Circuit(f"{name} " + " ".join(map(str, range(ar))))
        U = np.asarray(c.to_tableau().to_unitary_matrix(endian="big"), complex)   # complex64
        # snap every real / imaginary part to the exact Clifford value {0, +-1/2, +-1/sqrt2, +-1}
        grid = np.array([0.0, 0.5, _S2, 1.0])
        snap = lambda x: np.sign(x) * grid[np.abs(np.abs(x)[..., None] - grid).argmin(-1)]
        U = snap(U.real) + 1j * snap(U.imag)
        assert np.allclose(U.conj().T @ U, np.eye(len(U)), atol=1e-15), name
        _STIM_U[name] = (ar, U)
    return _STIM_U[name]


def _apply2(psi, U, a, b, n):
    t = psi.reshape([2] * n)
    t = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [a, b]))
    t = np.moveaxis(t, [0, 1], [a, b])
    return t.reshape(-1)


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
    # An E/CORRELATED_ERROR + ELSE_CORRELATED_ERROR chain is ONE site (numbered at its E line)
    # whose single target is the chain's whole support: the distinct qubits of all its terms in
    # first-appearance order (the engine's documented order). Pre-scan the supports so the
    # forced word (one letter per support qubit) can be applied at the E line.
    chain_support = {}
    pre_site, open_chain = 0, None
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("TICK"):
            open_chain = None
            continue
        name, _, toks = _parse(line)
        if name in NOISE1 or name in NOISE2:
            pre_site += 1
            open_chain = None
        elif name in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"):
            if name != "ELSE_CORRELATED_ERROR":
                open_chain = pre_site
                chain_support[open_chain] = []
                pre_site += 1
            for t in toks:
                q = int(t.lstrip("!")[1:])
                if q not in chain_support[open_chain]:
                    chain_support[open_chain].append(q)
        else:
            open_chain = None
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("TICK"):
            continue
        name, args, toks = _parse(line)
        nrec = len(br[0][1])
        if name == "ELSE_CORRELATED_ERROR":
            continue                                   # part of the site opened by its E line
        if name in ("E", "CORRELATED_ERROR"):
            p = faults.get((site, 0))
            if p:
                g = chain_support[site]
                assert len(p) == len(g), "oracle: chain word length != support size"
                br = [(_pauli_vec(v, [(q, c) for q, c in zip(g, p) if c != "I"], n), r)
                      for v, r in br]
            site += 1
            continue
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
        if name in ("R", "RZ", "RX", "RY"):
            for t in toks:
                q = int(t)
                assert q not in started, "oracle: resets only on fresh qubits"
                started.add(q)
                if name == "RX":
                    br = [(_apply1(v, G1["H"], q, n), r) for v, r in br]
                elif name == "RY":
                    br = [(_apply1(_apply1(v, G1["H"], q, n), G1["S"], q, n), r) for v, r in br]
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
        if name in ("M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"):
            basis = "X" if name in ("MX", "MRX") else ("Y" if name in ("MY", "MRY") else "Z")
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
                            u = _apply1(u, G1["Z" if basis in ("X", "Y") else "X"], q, n)
                        rec = b ^ inv ^ (len(r) in record_flips)
                        nb.append((u, r + [rec]))
                br = nb
            continue
        su = _stim_unitary(name)
        if su is not None:
            ar, U = su
            qs = [int(t) for t in toks]
            for i in range(0, len(qs), ar):
                g = qs[i:i + ar]
                started.update(g)
                if ar == 1:
                    br = [(_apply1(v, U, g[0], n), r) for v, r in br]
                else:
                    br = [(_apply2(v, U, g[0], g[1], n), r) for v, r in br]
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


def aggregate(orc):
    """Dense branches -> {(det, obs, dec): (probability, conditional expectations)} (raw
    detector/observable parities)."""
    agg = {}
    for o in orc.values():
        k = (tuple(o["det"]), tuple(o["obs"]), tuple(o["dec"]))
        a = agg.setdefault(k, [0.0, np.zeros(len(o["exp"]))])
        a[0] += o["p"]
        a[1] = a[1] + o["p"] * np.array(o["exp"], float)
    return {k: (p, v / p) for k, (p, v) in agg.items()}


def _law_rows(law, sd, so):
    out = {}
    sd, so = np.array(sd, int), np.array(so, int)
    for i in range(len(law.probabilities)):
        k = (tuple((law.detectors[i].astype(int) ^ sd).tolist()),
             tuple((law.observables[i].astype(int) ^ so).tolist()),
             tuple(law.decisions[i].astype(int).tolist()))
        assert k not in out, "expanded law: duplicate target row"
        out[k] = (float(law.probabilities[i]), law.expectations[i])
    return out


def law_shifts(law, agg, cands=None, tol=1e-12):
    """The constant reference shifts (det, obs) under which the expanded law equals the
    aggregated dense branches (probabilities and conditional expectations to `tol`). `cands`
    (the survivors of earlier queries on the same circuit) narrows them; None derives them."""
    if cands is None:
        k0 = max(agg, key=lambda k: agg[k][0])
        cands = sorted({(tuple((law.detectors[i].astype(int) ^ np.array(k0[0], int)).tolist()),
                          tuple((law.observables[i].astype(int) ^ np.array(k0[1], int)).tolist()))
                         for i in range(len(law.probabilities))
                         if tuple(law.decisions[i].astype(int).tolist()) == k0[2]})
    keep = []
    for sd, so in cands:
        lk = _law_rows(law, sd, so)
        if any(abs(lk.get(k, (0.0,))[0] - agg.get(k, (0.0,))[0]) > tol for k in set(lk) | set(agg)):
            continue
        if all(np.abs(lk[k][1] - agg[k][1]).max(initial=0.0) <= tol
               for k in agg if agg[k][0] > 1e-9 and k in lk):
            keep.append((sd, so))
    return keep


def max_diff(law, agg, shift):
    """(max |dp|, max |dexp|) of the expanded law vs the aggregated branches under `shift`."""
    lk = _law_rows(law, *shift)
    dp = max((abs(lk.get(k, (0.0,))[0] - agg.get(k, (0.0,))[0]) for k in set(lk) | set(agg)),
             default=0.0)
    de = max((float(np.abs(lk[k][1] - agg[k][1]).max(initial=0.0))
              for k in agg if agg[k][0] > 1e-9 and k in lk), default=0.0)
    return dp, de
