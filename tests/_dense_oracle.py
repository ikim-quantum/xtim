"""TEST ORACLE (no xtim code): an independent dense density-matrix simulator of the xtim dialect
subset, with exact record branching — the arbiter of the 2026-10-09 sampling audit (validated
against stim on 2,300 random Clifford circuits, 0 failures). Written from the docs only; it uses
stim ONLY for the unitary matrices of named Clifford gates. Kept verbatim as an ORACLE (good
duplication): do not "fix" it toward the engine.

Independent dense density-matrix oracle for the xtim dialect (subset used by the
audit generator).  Written from the docs only; uses stim ONLY for the unitary
matrices of named Clifford gates (stim.Tableau.to_unitary_matrix), never xtim.

State model: dict  record-tuple -> unnormalised density matrix (ndarray, shape (2,)*2n,
axis q = row index of qubit q, axis n+q = column index of qubit q).  Records are the
RECORDED bits (after `!` inversion and M(p) readout flips) — exactly what feedback,
detectors etc. read.
"""
from __future__ import annotations

import itertools
import math
import re
from functools import lru_cache

import numpy as np

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], complex)
PY = np.array([[0, -1j], [1j, 0]], complex)
PZ = np.array([[1, 0], [0, -1]], complex)
PAULI = {"I": I2, "X": PX, "Y": PY, "Z": PZ}
H1 = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)

NONCLIFF = {
    "T": np.diag([1, np.exp(1j * np.pi / 4)]),
    "T_DAG": np.diag([1, np.exp(-1j * np.pi / 4)]),
    "CS": np.diag([1, 1, 1, 1j]),
    "CS_DAG": np.diag([1, 1, 1, -1j]),
    "CCZ": np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex),
}
_ch = np.zeros((4, 4), complex)
_ch[:2, :2] = I2
_ch[2:, 2:] = H1
NONCLIFF["CH"] = _ch
ALIAS = {"H_XZ": "H", "SQRT_Z": "S", "SQRT_Z_DAG": "S_DAG", "CNOT": "CX", "ZCX": "CX",
         "ZCZ": "CZ", "ZCY": "CY", "SWAPCZ": "CZSWAP", "MZ": "M", "RZ": "R", "MRZ": "MR",
         "CORRELATED_ERROR": "E"}


@lru_cache(maxsize=None)
def clifford_unitary(name):
    import stim
    U = np.asarray(stim.Tableau.from_named_gate(name).to_unitary_matrix(endian="big"),
                   dtype=complex)
    # stim returns complex64; snap to the exact Clifford entry set
    grid = np.array([0, 0.5, -0.5, 1, -1, 1 / math.sqrt(2), -1 / math.sqrt(2)])
    snap = lambda x: grid[np.abs(grid[None, :] - x.reshape(-1, 1)).argmin(1)].reshape(x.shape)
    U = snap(U.real) + 1j * snap(U.imag)
    assert np.allclose(U @ U.conj().T, np.eye(len(U)), atol=1e-12)
    return U


def gate_matrix(name):
    if name in NONCLIFF:
        return NONCLIFF[name]
    return clifford_unitary(name)


def gate_arity(name):
    if name in ("CCZ",):
        return 3
    if name in ("CS", "CS_DAG", "CH"):
        return 2
    if name in ("T", "T_DAG"):
        return 1
    import stim
    gd = stim.gate_data(name)
    return 2 if gd.is_two_qubit_gate else 1


class Rho:
    """helpers on a (2,)*2n tensor"""

    @staticmethod
    def rows(rho, M, qs, n):
        k = len(qs)
        Mt = M.reshape((2,) * (2 * k))
        out = np.tensordot(Mt, rho, axes=(list(range(k, 2 * k)), list(qs)))
        return np.moveaxis(out, list(range(k)), list(qs))

    @staticmethod
    def cols(rho, A, qs, n):
        cq = [n + q for q in qs]
        k = len(qs)
        At = A.reshape((2,) * (2 * k))
        out = np.tensordot(At, rho, axes=(list(range(k, 2 * k)), cq))
        return np.moveaxis(out, list(range(k)), cq)

    @staticmethod
    def unitary(rho, U, qs, n):
        r = Rho.rows(rho, U, qs, n)
        return Rho.cols(r, U.conj(), qs, n)

    @staticmethod
    def pauli_rows(rho, P, n):  # P list of (q, letter)
        for q, l in P:
            rho = Rho.rows(rho, PAULI[l], [q], n)
        return rho

    @staticmethod
    def pauli_cols(rho, P, n):  # rho -> rho P
        for q, l in P:
            rho = Rho.cols(rho, PAULI[l].T, [q], n)
        return rho

    @staticmethod
    def conj_pauli(rho, P, n):
        return Rho.pauli_cols(Rho.pauli_rows(rho, P, n), P, n)

    @staticmethod
    def project(rho, P, sign, n):
        """(I + sign P)/2 rho (I + sign P)/2"""
        Pr = Rho.pauli_rows(rho, P, n)
        rP = Rho.pauli_cols(rho, P, n)
        PrP = Rho.pauli_cols(Pr, P, n)
        return (rho + sign * Pr + sign * rP + PrP) / 4

    @staticmethod
    def trace(rho, n):
        m = rho.reshape(2 ** n, 2 ** n)
        return np.trace(m).real

    @staticmethod
    def expval(rho, P, n):
        Pr = Rho.pauli_rows(rho, P, n)
        return np.trace(Pr.reshape(2 ** n, 2 ** n))


def parse_targets(tokens):
    return tokens


LINE_RE = re.compile(r"^\s*([A-Za-z_0-9]+)(?:\[[^\]]*\])?(?:\(([^)]*)\))?\s*(.*)$")


def parse(text):
    ops = []
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        name = m.group(1).upper()
        args = [float(a) for a in m.group(2).split(",")] if m.group(2) else []
        toks = m.group(3).split()
        name = ALIAS.get(name, name)
        ops.append((name, args, toks))
    return ops


def qubit_count(ops):
    n = 0
    for name, args, toks in ops:
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "SHIFT_COORDS", "TICK",
                    "QUBIT_COORDS", "MPAD"):
            continue
        if name == "OUTPUT_QUBITS":
            toks = toks[1:]
        if name == "PAULI_EXPECTATION":
            toks = toks[:1]
        for t in toks:
            if t.startswith("rec"):
                continue
            for d in re.findall(r"\d+", t):
                n = max(n, int(d) + 1)
    return n


def pauli_terms(tok):
    P = []
    inv = False
    for piece in tok.split("*"):
        if piece.startswith("!"):
            inv = not inv
            piece = piece[1:]
        P.append((int(piece[1:]), piece[0].upper()))
    return P, inv


BASIS_FLIP = {"Z": "X", "X": "Z", "Y": "X"}  # a Pauli anticommuting with the basis


class Oracle:
    def __init__(self, text, noiseless=False, max_branches=1 << 14):
        self.text = text
        self.ops = parse(text)
        self.n = max(1, qubit_count(self.ops))
        self.noiseless = noiseless
        self.max_branches = max_branches
        self.detectors = []
        self.observables = {}
        self.decisions = {}
        self.expectations = []  # (label, P, rec_abs, branch_index_at_decl)
        self.outputs = []
        self.nmeas = 0

    def run(self):
        n = self.n
        rho = np.zeros((2,) * (2 * n), complex)
        rho[(0,) * (2 * n)] = 1
        br = {(): rho}
        exp_snap = []
        for name, args, toks in self.ops:
            br = self._op(br, name, args, toks)
            if name == "PAULI_EXPECTATION":
                pass
            if len(br) > self.max_branches:
                raise RuntimeError("oracle branch blowup")
        self.branches = br
        return br

    # --- helpers
    def _map(self, br, f):
        return {k: f(v) for k, v in br.items()}

    def _mix(self, br, terms):
        """terms: list of (prob, fn)"""
        out = {}
        for k, v in br.items():
            acc = None
            for p, fn in terms:
                if p == 0:
                    continue
                t = p * fn(v)
                acc = t if acc is None else acc + t
            out[k] = acc
        return out

    def _measure(self, br, P, inv, p_flip, reset_basis=None, reset_q=None):
        n = self.n
        out = {}
        for k, v in br.items():
            parts = {}
            for o in (0, 1):
                pr = Rho.project(v, P, 1 if o == 0 else -1, n)
                tr = Rho.trace(pr, n)
                if tr < 1e-15:
                    continue
                if reset_basis is not None and o == 1:
                    pr = Rho.conj_pauli(pr, [(reset_q, BASIS_FLIP[reset_basis])], n)
                parts[o] = pr
            for o, pr in parts.items():
                for flip, pf in ((0, 1 - p_flip), (1, p_flip)):
                    if pf == 0:
                        continue
                    bit = o ^ inv ^ flip
                    key = k + (bit,)
                    out[key] = out.get(key, 0) + pf * pr
        self.nmeas += 1
        return out

    def _recs(self, toks):
        res = []
        for t in toks:
            m = re.match(r"rec\[-(\d+)\]", t)
            res.append(self.nmeas - int(m.group(1)))
        return res

    def _op(self, br, name, args, toks):
        n = self.n
        nl = self.noiseless
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS", "I"):
            return br
        if name == "DETECTOR":
            self.detectors.append(self._recs(toks))
            return br
        if name == "OBSERVABLE_INCLUDE":
            self.observables.setdefault(int(args[0]), []).extend(self._recs(toks))
            return br
        if name == "DECISION":
            self.decisions.setdefault(int(args[0]), []).extend(self._recs(toks))
            return br
        if name == "OUTPUT_QUBITS":
            self.outputs.extend(int(t) for t in toks[1:])
            return br
        if name == "PAULI_EXPECTATION":
            P, _ = pauli_terms(toks[0])
            self.expectations.append((int(args[0]), P, self._recs(toks[1:])))
            # evaluate on the CURRENT state, per current branch key
            vals = {}
            for k, v in br.items():
                tr = Rho.trace(v, n)
                vals[k] = (Rho.expval(v, P, n), tr)
            self._exp_vals = getattr(self, "_exp_vals", [])
            self._exp_vals.append(vals)
            return br
        if name in NONCLIFF or name in ("H", "S", "S_DAG", "X", "Y", "Z") or _is_cliff(name):
            if name in ("CX", "CY", "CZ") and any(t.startswith("rec") for t in toks):
                # pairs, possibly mixed
                for i in range(0, len(toks), 2):
                    c, t = toks[i], toks[i + 1]
                    if c.startswith("rec"):
                        idx = self._recs([c])[0]
                        q = int(t)
                        P = [(q, name[1])]
                        br = {k: (Rho.conj_pauli(v, P, n) if k[idx] else v) for k, v in br.items()}
                    else:
                        U = gate_matrix(name)
                        qs = [int(c), int(t)]
                        br = self._map(br, lambda v, U=U, qs=qs: Rho.unitary(v, U, qs, n))
                return br
            U = gate_matrix(name)
            a = gate_arity(name)
            qs_all = [int(t) for t in toks]
            for i in range(0, len(qs_all), a):
                qs = qs_all[i:i + a]
                br = self._map(br, lambda v, U=U, qs=qs: Rho.unitary(v, U, qs, n))
            return br
        # ---- noise
        if name in ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1"):
            if nl:
                return br
            if name == "PAULI_CHANNEL_1":
                probs = dict(zip("XYZ", args))
            elif name == "DEPOLARIZE1":
                probs = {l: args[0] / 3 for l in "XYZ"}
            else:
                probs = {name[0]: args[0]}
            for t in toks:
                q = int(t)
                terms = [(1 - sum(probs.values()), lambda v: v)]
                for l, p in probs.items():
                    terms.append((p, lambda v, l=l, q=q: Rho.conj_pauli(v, [(q, l)], n)))
                br = self._mix(br, terms)
            return br
        if name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
            if nl:
                return br
            pairs = [p for p in itertools.product("IXYZ", repeat=2) if p != ("I", "I")]
            if name == "DEPOLARIZE2":
                probs = [args[0] / 15] * 15
            else:
                probs = args
            qs = [int(t) for t in toks]
            for i in range(0, len(qs), 2):
                a, b = qs[i], qs[i + 1]
                terms = [(1 - sum(probs), lambda v: v)]
                for (la, lb), p in zip(pairs, probs):
                    P = [(q, l) for q, l in ((a, la), (b, lb)) if l != "I"]
                    terms.append((p, lambda v, P=P: Rho.conj_pauli(v, P, n)))
                br = self._mix(br, terms)
            return br
        if name in ("E", "ELSE_CORRELATED_ERROR"):
            P = [(int(t[1:]), t[0].upper()) for t in toks]
            if name == "E":
                self._chain = [(args[0], P)]
            else:
                self._chain.append((args[0], P))
            # Correlated chains are applied lazily: we re-apply at the end of the chain.
            # Simplest exact handling: store pre-chain state and recompute on each element.
            if name == "E":
                self._pre_chain = br
            if nl:
                return br
            pre = self._pre_chain
            terms = []
            rem = 1.0
            for p, PP in self._chain:
                terms.append((rem * p, lambda v, PP=PP: Rho.conj_pauli(v, PP, n)))
                rem *= (1 - p)
            terms.append((rem, lambda v: v))
            return self._mix(pre, terms)
        if name == "HERALDED_ERASE":
            p = 0.0 if nl else args[0]
            for t in toks:
                q = int(t)
                out = {}
                for k, v in br.items():
                    if 1 - p > 0:
                        out[k + (0,)] = out.get(k + (0,), 0) + (1 - p) * v
                    if p > 0:
                        mixed = sum(Rho.conj_pauli(v, [(q, l)], n) for l in "XYZ") + v
                        out[k + (1,)] = out.get(k + (1,), 0) + p * mixed / 4
                self.nmeas += 1
                br = out
            return br
        if name == "MPAD":
            p = (args[0] if args else 0.0) if not nl else 0.0
            for t in toks:
                b = int(t)
                out = {}
                for k, v in br.items():
                    for flip, pf in ((0, 1 - p), (1, p)):
                        if pf:
                            out[k + (b ^ flip,)] = out.get(k + (b ^ flip,), 0) + pf * v
                self.nmeas += 1
                br = out
            return br
        # ---- measurement / reset
        pflip = (args[0] if args else 0.0) if not nl else 0.0
        if name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            basis = {"M": "Z", "MR": "Z"}.get(name, name[-1])
            reset = name.startswith("MR")
            for t in toks:
                inv = t.startswith("!")
                q = int(t.lstrip("!"))
                br = self._measure(br, [(q, basis)], int(inv), pflip,
                                   reset_basis=basis if reset else None, reset_q=q)
            return br
        if name in ("R", "RX", "RY"):
            basis = {"R": "Z"}.get(name, name[-1])
            for t in toks:
                q = int(t)
                P = [(q, basis)]
                br = {k: Rho.project(v, P, 1, n) +
                      Rho.conj_pauli(Rho.project(v, P, -1, n), [(q, BASIS_FLIP[basis])], n)
                      for k, v in br.items()}
            return br
        if name == "MPP":
            for t in toks:
                P, inv = pauli_terms(t)
                br = self._measure(br, P, int(inv), pflip)
            return br
        if name in ("MXX", "MYY", "MZZ"):
            l = name[1]
            for i in range(0, len(toks), 2):
                a, b = toks[i], toks[i + 1]
                inv = a.startswith("!") ^ b.startswith("!")
                P = [(int(a.lstrip("!")), l), (int(b.lstrip("!")), l)]
                br = self._measure(br, P, int(inv), pflip)
            return br
        raise NotImplementedError(name)

    # ---- outputs
    def record_dist(self):
        """{record tuple: probability}"""
        return {k: Rho.trace(v, self.n) for k, v in self.branches.items()}

    def final_expect(self, P):
        return {k: (Rho.expval(v, P, self.n) / Rho.trace(v, self.n)).real
                for k, v in self.branches.items() if Rho.trace(v, self.n) > 1e-14}

    def reduced(self, k, qs):
        """normalised reduced density matrix of branch k on qubits qs (in order)."""
        n = self.n
        v = self.branches[k]
        keep = list(qs)
        rest = [q for q in range(n) if q not in keep]
        # trace out rest
        t = v
        # move axes: rows keep, rows rest, cols keep, cols rest
        perm = keep + rest + [n + q for q in keep] + [n + q for q in rest]
        t = np.transpose(t, perm)
        a, b = len(keep), len(rest)
        t = t.reshape(2 ** a, 2 ** b, 2 ** a, 2 ** b)
        r = np.einsum("ijkj->ik", t)
        return r / np.trace(r).real


def _is_cliff(name):
    try:
        import stim
        gd = stim.gate_data(name)
        return gd.is_unitary
    except Exception:
        return False


def det_values(rec, sets):
    return [sum(rec[i] for i in s) & 1 for s in sets]
