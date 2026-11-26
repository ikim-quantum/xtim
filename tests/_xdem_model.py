"""Test-only ORACLE for the xdem format (SPEC-approved-2026-11-15; never production code).

An independent reader of xdem text, the stabilizer model of SPEC §1 as a stim circuit on N = D + O + r + m qubits
(readout qubits | gauge qubits | hidden-syndrome qubits), its X-readout law from stim tableaux, the §1.2 composite, the
§1.3 closed form, and the decoder loop of SPEC §4 (K "optimal", "r2b", "none") evaluated on the model register.
Ported from internal working scripts: ex422_verify.py (md5 c33e8eae978779919d9a13762b35bf87),
identities_check.py (611f12c244c2f07de9e8f27890d4f018) and kstrings_check.py
(f55cd6c739154c4e8c4753eb5e01c47b). Bit conventions: readout qubit j = bit j (detectors 0..D-1, observables D..D+O-1),
gauge qubit t = bit D+O+t, hidden-syndrome qubit t = bit D+O+r+t; an axis or line set = a bitmask over axis ids.
"""
from __future__ import annotations

import collections
import re

import numpy as np
import stim


def pc(x: int) -> int:
    return bin(x).count("1")


def cancel(ids) -> list:
    """file order, a line listed twice cancels (M_i^2 = I, SPEC §1.2)"""
    out = []
    for i in sorted(int(i) for i in ids):
        if out and out[-1] == i:
            out.pop()
        else:
            out.append(i)
    return out


def rref(vecs) -> dict:
    """fully reduced echelon basis {pivot (highest bit): vector}"""
    B = {}
    for v in vecs:
        for p in sorted(B, reverse=True):
            if v >> p & 1:
                v ^= B[p]
        if v:
            p = v.bit_length() - 1
            for q in B:
                if B[q] >> p & 1:
                    B[q] ^= v
            B[p] = v
    return B


def reduce(v: int, B: dict) -> int:
    for p in sorted(B, reverse=True):
        if v >> p & 1:
            v ^= B[p]
    return v


def solve(rows, rhs):
    """x with parity(rows[i] & x) == rhs[i] for all i, or None"""
    piv = {}
    for r, b in zip(rows, rhs):
        for p in sorted(piv, reverse=True):
            if r >> p & 1:
                r ^= piv[p][0]
                b ^= piv[p][1]
        if r == 0:
            if b:
                return None
            continue
        piv[r.bit_length() - 1] = (r, b)
    x = 0
    for p in sorted(piv):
        r, b = piv[p]
        if (pc(r & x) & 1) ^ b:
            x ^= 1 << p
    return x


def affine(cons, n: int):
    """the outcome space {x : parity(J & x) = v for (J, v) in cons} as (offset reduced mod coins, rref coins)"""
    x0 = solve([J for J, _ in cons], [v for _, v in cons])
    assert x0 is not None, "inconsistent law"
    R = rref([J for J, _ in cons])
    piv = set(R)
    coins = []
    for j in range(n):                      # null space: one vector per free column
        if j in piv:
            continue
        v = 1 << j
        for p, row in R.items():
            if row >> j & 1:
                v |= 1 << p
        coins.append(v)
    C = rref(coins)
    return reduce(x0, C), C


def _bits(arr) -> int:
    return int.from_bytes(np.packbits(np.asarray(arr, bool), bitorder="little").tobytes(), "little")


def _pauli(xmask: int, zmask: int, N: int) -> stim.PauliString:
    P = stim.PauliString(N)
    for q in range(N):
        x, z = xmask >> q & 1, zmask >> q & 1
        P[q] = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(x, z)]
    return P


def _tab(c: stim.Circuit, N: int) -> stim.Tableau:
    c = c.copy()
    c.append("I", [N - 1])
    return stim.Tableau.from_circuit(c)


def _rot(c: stim.Circuit, qs, e: int, b: int) -> None:
    """append e^{i pi/4 e Zhat}, Zhat = (-1)^b Z[qs] (up to a global phase; spec/scratch/identities_check.py)"""
    ep = (e if not b else -e) % 4
    if not ep or not qs:
        return
    c.append({1: "S_DAG", 2: "Z", 3: "S"}[ep], qs)
    if ep % 2 and len(qs) > 1:
        c.append("CZ", [t for i, u in enumerate(qs) for v in qs[i + 1:] for t in (u, v)])


def _unroll(lines) -> list:
    out, i = [], 0
    while i < len(lines):
        raw = lines[i].split("#", 1)[0].strip()
        i += 1
        if not raw:
            continue
        m = re.fullmatch(r"repeat\s+(\d+)\s*\{", raw)
        if m:
            depth, body = 1, []
            while True:
                t = lines[i].split("#", 1)[0].strip()
                i += 1
                if t.endswith("{"):
                    depth += 1
                elif t == "}":
                    depth -= 1
                    if depth == 0:
                        break
                body.append(t)
            out += _unroll(body) * int(m.group(1))
            continue
        out.append(raw)
    return out


class XdemModel:
    """axes: list of (bhat, Phi) with Phi a bitmask over the N qubits; lines: list of (p, f, tau, rot, late) with f over
    the readout qubits, tau over the axes, rot / late {axis: exponent in Z4} (a file holds 1 and 3 only; tests may build
    a model with 2's directly)."""

    def __init__(self, D, O, r, m, tc, axes, lines):
        self.D, self.O, self.r, self.m, self.tc = D, O, r, m, tc
        self.n = D + O
        self.N = self.n + r + m
        self.axes = list(axes)
        self.lines = list(lines)
        self.A = len(self.axes)

    # ------------------------------------------------------------------------------------------- reader
    @classmethod
    def from_text(cls, text: str) -> "XdemModel":
        hdr, axes, lines, shift = None, {}, [], 0

        def q(t):
            k, v = t[0], int(t[1:])
            D, O, r = hdr[0], hdr[1], hdr[2]
            return {"D": v + shift, "L": D + v, "g": D + O + v, "h": D + O + r + v}[k]

        for raw in _unroll(text.splitlines()):
            tok = raw.split()
            if tok[0] == "xdem":
                kv = dict(zip(tok[2::2], (int(x) for x in tok[3::2])))
                hdr = (kv["detectors"], kv["observables"], kv["gauge"], kv["hidden"], kv["tc"])
            elif tok[0] == "axis":
                b, phi, cur = 0, 0, None
                for t in tok[2:]:
                    if t in ("sign", "img", "gauge", "hidden"):
                        cur = t
                        continue
                    if cur == "sign":
                        b = 1
                        continue
                    phi |= 1 << q(t)
                axes[int(tok[1][1:])] = (b, phi)
            elif tok[0].startswith("error"):
                p = float(re.fullmatch(r"error(?:\[[^\]]*\])?\((.*)\)", tok[0]).group(1))
                f, tau, rot, late, cur = 0, 0, {}, {}, "t"
                for t in tok[1:]:
                    if t in ("x", "rot", "late"):
                        cur = t
                    elif cur == "t":
                        if t != "^":
                            f ^= 1 << q(t)
                    elif cur == "x":
                        tau |= 1 << int(t[1:])
                    else:
                        (rot if cur == "rot" else late)[int(t[2:])] = 1 if t[0] == "+" else 3
                lines.append((p, f, tau, rot, late))
            elif tok[0].startswith("shift_detectors"):
                shift += int(tok[-1])
            elif tok[0].startswith(("detector", "logical_observable")):
                continue
            else:
                raise ValueError("oracle reader: unknown line: " + raw)
        assert hdr is not None, "oracle reader: extended files only (no xdem header)"
        return cls(*hdr, [axes[a] for a in range(len(axes))], lines)

    # ------------------------------------------------------------------------------------------- structure
    def img(self, a: int) -> int:
        return self.axes[a][1] & ((1 << self.n) - 1)

    def late_axes(self) -> list:
        return sorted({a for (_, _, _, _, late) in self.lines for a in late})

    def chi(self, tau: int) -> int:
        """an X-string on the gauge and hidden-syndrome qubits anticommuting with exactly the axes in tau"""
        x = solve([phi >> self.n for (_, phi) in self.axes], [tau >> a & 1 for a in range(self.A)])
        assert x is not None, "tau not representable: the line is not an operator (SPEC §3.5(c))"
        return x << self.n

    @staticmethod
    def total(rot: dict, late: dict) -> dict:
        E = dict(rot)
        for a, e in late.items():
            E[a] = (E.get(a, 0) + e) % 4
        return {a: e for a, e in E.items() if e}

    def compose(self, ids):
        """SPEC §1.2: (f, tau, eta) of the lines in file order; eta {axis: nonzero exponent mod 4}"""
        f, tau, eta = 0, 0, {}
        for i in cancel(ids):
            _, fi, ti, rot, late = self.lines[i]
            for a, e in self.total(rot, late).items():
                eta[a] = (eta.get(a, 0) + (-1) ** (tau >> a & 1) * e) % 4
            f ^= fi
            tau ^= ti
        return f, tau, {a: e for a, e in eta.items() if e}

    # ------------------------------------------------------------------------------------------- circuits and laws
    def qubits(self, phi: int) -> list:
        return [q for q in range(self.N) if phi >> q & 1]

    def line_circuit(self, f: int, tau: int, E: dict) -> stim.Circuit:
        """rotation first, then Z[f], then X[chi] (SPEC §1.1)"""
        c = stim.Circuit()
        for a, e in sorted(E.items()):
            b, phi = self.axes[a]
            _rot(c, self.qubits(phi), e, b)
        zs = [q for q in range(self.n) if f >> q & 1]
        if zs:
            c.append("Z", zs)
        xs = self.qubits(self.chi(tau))
        if xs:
            c.append("X", xs)
        return c

    def circuit(self, ids, inserts=(), prep=True) -> stim.Circuit:
        """|+>^{n+r}|0>^m, the lines of ids in file order; inserts: (index, circuit) placed before line `index`"""
        c = stim.Circuit()
        if prep:
            c.append("H", list(range(self.n + self.r)))
        ins = sorted(inserts, key=lambda t: t[0])
        k = 0
        for i in cancel(ids):
            while k < len(ins) and ins[k][0] <= i:
                c += ins[k][1]
                k += 1
            _, f, tau, rot, late = self.lines[i]
            c += self.line_circuit(f, tau, self.total(rot, late))
        for _, extra in ins[k:]:
            c += extra
        return c

    def law(self, circ: stim.Circuit) -> list:
        """the X-readout law: the deterministic readout parities [(J, value)] (generators)"""
        sim = stim.TableauSimulator()
        sim.set_num_qubits(self.N)
        sim.do(circ)
        rows = []
        for P in sim.canonical_stabilizers():
            xs, zs = P.to_numpy()
            rows.append((_bits(xs), _bits(zs), P))
        hid = ((1 << self.N) - 1) ^ ((1 << self.n) - 1)
        piv, ker = {}, []
        for gi, (xi, zi, _) in enumerate(rows):
            v, cmb = (zi << self.N) | (xi & hid), 1 << gi
            while v:
                p = v.bit_length() - 1
                if p in piv:
                    v ^= piv[p][0]
                    cmb ^= piv[p][1]
                else:
                    piv[p] = (v, cmb)
                    break
            if not v:
                ker.append(cmb)
        out = []
        for cmb in ker:
            P = stim.PauliString(self.N)
            for gi in range(len(rows)):
                if cmb >> gi & 1:
                    P *= rows[gi][2]
            xs, zs = P.to_numpy()
            assert not zs.any() and P.sign in (1, -1)
            out.append((_bits(xs) & ((1 << self.n) - 1), 0 if P.sign == 1 else 1))
        return out

    def law13(self, ids):
        """SPEC §1.3 closed form (enumerates the 2^n parities: small models only); [(J, value)]"""
        f, _, eta = self.compose(ids)
        kap = [(phi >> self.n) & ((1 << self.r) - 1) for (_, phi) in self.axes]
        cons = []
        for J in range(1, 1 << self.n):
            AJ = [a for a in range(self.A) if pc(self.img(a) & J) & 1]
            BJ = [a for a in AJ if eta.get(a, 0) % 2]
            kb = 0
            for a in BJ:
                kb ^= kap[a]
            if kb:
                continue
            s = sum(eta.get(a, 0) for a in AJ)
            assert s % 2 == 0
            cons.append((J, (pc(f & J) + s // 2 + sum(self.axes[a][0] for a in BJ)) & 1))
        return cons

    # ------------------------------------------------------------------------------------------- conditioning
    def given(self, cons, s: int):
        """(k, c0, rows, ok): the observable law given detectors s; k = its dimension, c0 one consistent observable
        vector (bits 0..O-1), ok = Pr(s) > 0"""
        D = self.D
        piv, ok = {}, True
        for J, v in cons:
            jo, vv = J >> D, v ^ (pc(J & ((1 << D) - 1) & s) & 1)
            for p in sorted(piv, reverse=True):
                if jo >> p & 1:
                    jo ^= piv[p][0]
                    vv ^= piv[p][1]
            if not jo:
                ok = ok and not vv
                continue
            piv[jo.bit_length() - 1] = (jo, vv)
        c0 = 0
        for p in sorted(piv):
            jo, vv = piv[p]
            if (pc(jo & c0) & 1) ^ vv:
                c0 ^= 1 << p
        return self.O - len(piv), c0, list(piv.values()), ok

    def pass_of(self, cons, s: int, c: int):
        k, _, rows, ok = self.given(cons, s)
        if not ok:
            return None
        return 2.0 ** -k if all((pc(jo & c) & 1) == vv for jo, vv in rows) else 0.0

    def definite(self, cons, s: int, f: int) -> int:
        """the law's observable value at s when it is definite (k = 0, Pr(s) > 0), else the XOR flips f|_O"""
        k, c0, _, ok = self.given(cons, s)
        return c0 if (ok and k == 0) else f >> self.D

    # ------------------------------------------------------------------------------------------- the loop (SPEC §4)
    def pass_none(self, T, H, s: int):
        f, _, _ = self.compose(H)
        c = self.definite(self.law(self.circuit(H)), s, f)
        return self.pass_of(self.law(self.circuit(T)), s, c)

    def k_r2b(self, H, s: int) -> dict:
        """SPEC §4.4 (the file must satisfy its availability condition)"""
        tc, H = self.tc, cancel(H)
        eta, tauc = {}, 0
        for i in H:
            if i >= tc:
                continue
            _, _, ti, _, late = self.lines[i]
            for a, e in late.items():
                eta[a] = (eta.get(a, 0) + (-1) ** (tauc >> a & 1) * e) % 4
            tauc ^= ti
        eta = {a: e for a, e in eta.items() if e}
        tauK = sum(1 << a for a in self.late_axes() if tauc >> a & 1)
        Kc = stim.Circuit()
        for a, e in sorted(eta.items()):
            b, phi = self.axes[a]
            _rot(Kc, self.qubits(phi), e, b)
        f, _, _ = self.compose(H)
        L0 = self.law(self.circuit(H))
        L1 = self.law(self.circuit(H, inserts=[(tc, Kc)]))
        k0, k1 = self.given(L0, s)[0], self.given(L1, s)[0]
        applied = bool(eta) and k1 <= k0
        c = self.definite(L1 if applied else L0, s, f)
        return dict(eta=eta, tau=tauK, applied=applied, c=c, circuit=Kc)

    def pass_r2b(self, T, H, s: int):
        K = self.k_r2b(H, s)
        ins = [(self.tc, K["circuit"])] if K["applied"] else []
        return self.pass_of(self.law(self.circuit(T, inserts=ins)), s, K["c"])

    def k_optimal(self, H, s: int) -> dict:
        """SPEC §4.3 by stim tableaux: per observable k, lossy / P_k / S_k / y_k / frame f_K,k; the K circuit"""
        D, O, n, N = self.D, self.O, self.n, self.N
        f, tau, eta = self.compose(H)
        Od = [a for a, e in eta.items() if e % 2]
        mD = (1 << D) - 1
        iota = [self.img(a) & mD for a in range(self.A)]
        col = [0] * D
        for a in Od:
            for j in range(D):
                if iota[a] >> j & 1:
                    col[j] ^= iota[a]
        ech, piv = {}, []                                        # greedy independent columns, detector order
        for j in range(D):
            v, cmb = col[j], 1 << j
            while v:
                hb = v.bit_length() - 1
                if hb not in ech:
                    ech[hb] = (v, cmb)
                    piv.append(j)
                    break
                v ^= ech[hb][0]
                cmb ^= ech[hb][1]
        Th = _tab(self.circuit(H, prep=False), N)
        out = dict(lossy=[], P=[], S=[], y=[], eps=[], frame=[], pivots=list(piv))
        Kc, edges, info = stim.Circuit(), set(), []
        for k in range(O):
            L = D + k
            t = 0
            for a in Od:
                if self.img(a) >> L & 1:
                    t ^= iota[a]
            v, cmb = t, 0                                        # t = XOR of the pivot columns in cmb, or lossy
            while v:
                hb = v.bit_length() - 1
                if hb not in ech:
                    break
                v ^= ech[hb][0]
                cmb ^= ech[hb][1]
            if v:
                out["lossy"].append(True)
                out["P"].append(None)
                out["S"].append(None)
                out["y"].append(None)
                out["eps"].append(None)
                continue
            Pk = cmb
            G = (1 << L) | Pk
            Y = Th(_pauli(G, 0, N))                              # M X_G M^dag
            xs, zs = Y.to_numpy()
            zm = _bits(zs)
            assert zm & mD == 0, "a non-lossy rotated readout has a Z on a detector (LOOP §4(ii))"
            y = zm >> L & 1
            S = [a for a in Od if pc(self.img(a) & G) & 1]
            out["lossy"].append(False)
            out["P"].append(Pk)
            out["S"].append(S)
            out["y"].append(y)
            out["eps"].append(int(Y.sign.real))
            if y:
                Kc.append("S", [L])
            for q in range(N):
                if q == L or not zm >> q & 1:
                    continue
                e = (min(L, q), max(L, q))
                if D <= q < n and e in edges:                    # an observable-observable edge, once
                    continue
                edges.add(e)
                Kc.append("CZ", [L, q])
            info.append((k, L, Pk, zm))
        Kt = _tab(Kc, N).inverse()                               # P -> K_diag^dag P K_diag
        frame = 0
        out["frame"] = [0] * O
        for k, L, Pk, zm in info:
            Q = Kt(_pauli(1 << L, 0, N))
            assert _bits(Q.to_numpy()[1]) == zm, "K_diag does not realise the rotated readout's letters"
            fk = (pc(s & Pk) & 1) ^ (out["eps"][k] != int(Q.sign.real))
            out["frame"][k] = fk
            frame |= fk << L
        lossy = [j for j in range(O) if out["lossy"][j]]
        if lossy:                                                # ERRATA E4
            out["frame"] = list(out["frame"])
            c = self.lossy_c(H, s, Kc, frame, lossy, f >> D)
            for j in lossy:
                out["frame"][j] = c >> j & 1
                frame |= (c >> j & 1) << (D + j)
        if frame:
            Kc.append("Z", [q for q in range(n) if frame >> q & 1])
        out["circuit"] = Kc
        return out

    def lossy_c(self, H, s: int, Kc, frame: int, lossy: list, fO: int) -> int:
        """ERRATA E4, by enumeration: the support of law([H, K]) (K = K_diag and the non-lossy frame Z's) at s,
        projected on the lossy observables; c = the support point x minimising the bit tuple of x ^ f_H on the lossy
        observables, observable 0 first (so c = f_H there when f_H is in the support); 0 there when Pr(s) = 0 (SPEC §7 Q1)"""
        D, O = self.D, self.O
        Kf = stim.Circuit() + Kc
        if frame:
            Kf.append("Z", [q for q in range(self.n) if frame >> q & 1])
        _, _, rows, ok = self.given(self.law(self.circuit(H) + Kf), s)
        lm = sum(1 << j for j in lossy)
        fL = fO & lm
        if not ok:
            return 0
        assert O <= 14, "lossy_c: small O only (enumeration)"
        pts = {o & lm for o in range(1 << O) if all((pc(jo & o) & 1) == vv for jo, vv in rows)}
        assert pts, "Pr(s) > 0 but no observable value"
        return min(pts, key=lambda x: [(x ^ fL) >> j & 1 for j in lossy])

    def pass_optimal(self, T, H, s: int):
        K = self.k_optimal(H, s)
        return self.pass_of(self.law(self.circuit(T) + K["circuit"]), s, 0)


def exact_file_law(text):
    """{outcome int over D+O: probability} = sum over line subsets of prod p (1 - p) x the uniform affine law (the
    shared oracle of tests/test_xdem_fuzz.py and tests/test_xdem_sampler.py)"""
    M = XdemModel.from_text(text)
    out = collections.Counter()
    L = len(M.lines)
    assert L <= 14, "exact_file_law: small files only"
    for mask in range(1 << L):
        ids = [i for i in range(L) if mask >> i & 1]
        w = 1.0
        for i in range(L):
            w *= M.lines[i][0] if mask >> i & 1 else 1.0 - M.lines[i][0]
        off, coins = affine(M.law(M.circuit(ids)), M.n)
        span = [0]
        for c in coins.values():
            span += [s ^ c for s in span]
        for s in span:
            out[off ^ s] += w / len(span)
    return dict(out)
