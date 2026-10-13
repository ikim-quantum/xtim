"""Dense-oracle law identity for twirl plans with kappa >= 2 kernel directions (the canonical
dual-basis fold, derivation note 2026-10-12, owner-signed).

Law under test (Prop. 2 of the note): with a diagonal residual R = P.C on the bare state psi and
certified generators g_i, the per-shot record law is
    p(sigma) = 2^{-r} 1[sigma in beta + C_det^perp] Born_psi(y(sigma)),  y_j = c_j.sigma ^ s_j,
and the sampler ASSEMBLES sigma = det_signs ^ sum_k a_k coin_k ^ sum_j f_j delta_j with fair coins
a and f_j = y_j ^ kernel_base_j, y ~ Born (Prop. 3). This test enumerates the assembled law from
the engine's exported plan (plan_structure: det_signs, coin_masks, kernel_masks, kernel_base,
kernel logicals) and compares it, support and values at 1e-9, with the dense law
    p_dense(sigma) = <R psi| prod_i (I + (-1)^{sigma_i} g_i)/2 |R psi>
computed by an INDEPENDENT oracle: the bare state is a statevector simulation of the fixture
circuit written here (not the engine); the certified group is read from certified_symplectic and
every generator is checked to be an exact +1 stabilizer of that statevector; the engine's deferred
ancilla wires (product |0>/|+> wires) are separated through their single-wire stabilizers found
by GF(2) algebra on the certified rows.

Fixtures: B in {2, 3} blocks of [[4,2,2]] with BOTH logicals magic (HT|+>, X.HT|+>, T|+> in blocks
0, 1, 2: no logical is certified and the blocks' Born laws differ), the X
ancillas chained by CX before the encoders so the engine's basis carries NESTED X-type generators
(X-all, X-blocks<b), and the residual S on qubits (4b, 4b+1) of every block handed to
plan_structure as a hand-built plan key (prefix / cz variants included). These realise the note's
section 4.4 instances in the engine's own pipeline: kappa = B, and the LEGACY fold choice
(delta_j := the "[qubit rows | C_det] probe" preimage, else the split pick) has a pairing matrix
c_j.delta_k != delta_jk, so the 3.1.12 engine fell back to the exact per-shot path on every one of
these keys (verified offline with the 3.1.12 binary: plan_structure(key)["fallback"] == True for
all six keys). test_legacy_fold_trips re-derives that legacy pick here (a faithful port of the
pre-change construction) and asserts it is NOT a dual basis, so the fixtures demonstrably exercise
the repair; test_kernel_dirs_match_engine pins the port to the engine's kernel directions.

Mutations (note section 5.2): m1 (legacy folds) / m2 (one kernel_base flipped) / m3 (delta_k ^=
c_j, j != k) must FAIL the dense identity; m4 (delta ^= a coin vector) must PASS — the coin gauge
of Thm. C, recorded as such (it is not a correctness check).
"""
import itertools

import numpy as np
import pytest

import xtim
import xtim._xtim as _xtim


# ---------------------------------------------------------------------------
# fixture circuits (physical qubits 0..4B-1) and plan keys (deferred wires)
# ---------------------------------------------------------------------------

def circuit_text(B):
    L = []
    for b in range(B):
        L.append("RX %d" % (4 * b))
    for b in range(B - 1, 0, -1):               # nest: X_{4b} -> X_{4b} X_{4(b-1)} ...
        L.append("CX %d %d" % (4 * b, 4 * (b - 1)))
    for b in range(B):
        for d in (4 * b + 1, 4 * b + 2):
            # block b's logicals, one T each (the engine refuses T.H.T on a wire): block 0
            # HT|+> (<Z> = +1/sqrt2), block 1 X.HT|+> (-1/sqrt2), block 2 T|+> (0) — distinct
            # Born factors, so a PERMUTED pairing matrix cannot hide behind a block symmetry
            L += ["RX %d" % d, "T %d" % d] + (["H %d" % d] if b != 2 else []) + (["X %d" % d] if b == 1 else [])
        L.append("R %d" % (4 * b + 3))
    for b in range(B):
        a0, d1, d2, a3 = 4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3
        L += ["CX %d %d" % (a0, d1), "CX %d %d" % (a0, d2), "CX %d %d" % (d1, a3),
              "CX %d %d" % (d2, a3), "CX %d %d" % (a0, a3)]
    L.append("X_ERROR(0.01) %d" % (4 * B - 1))
    for b in range(B):
        q = [4 * b + i for i in range(4)]
        L.append("MPP " + "*".join("X%d" % i for i in q))
        L.append("DETECTOR rec[-1]")
        L.append("MPP " + "*".join("Z%d" % i for i in q))
        L.append("DETECTOR rec[-1]")
    L.append("OUTPUT_QUBITS out " + " ".join(str(q) for q in range(4 * B)))
    return "\n".join(L) + "\n"


def plan_key(n, a_wires, cz_wires=(), px_wires=(), pz_wires=()):
    """plan_structure key: prefix.x words | prefix.z words | a bytes | 0xFF | cz u16 pairs."""
    NW = (n + 63) // 64
    px = np.zeros(NW, np.uint64)
    pz = np.zeros(NW, np.uint64)
    for q in px_wires:
        px[q // 64] |= np.uint64(1) << np.uint64(q % 64)
    for q in pz_wires:
        pz[q // 64] |= np.uint64(1) << np.uint64(q % 64)
    key = px.tobytes() + pz.tobytes()
    a = bytearray(n)
    for q in a_wires:
        a[q] = 1
    key += bytes(a) + b"\xff"
    for (f, s) in cz_wires:
        key += bytes([f & 255, f >> 8, s & 255, s >> 8])
    return key


# (B, cz pairs (physical), prefix X qubits, prefix Z qubits, extra S qubits)
_FIXTURES = [
    (2, (), (), (), ()),
    (2, (), (2,), (5,), ()),
    (2, ((1, 2),), (), (), ()),
    (2, (), (0,), (), "anc_x"),       # extra S on a |+> ancilla wire riding an X generator: r > 0
    (3, (), (), (), ()),
    (3, ((1, 2),), (6,), (0,), ()),
]
_IDS = ["B2", "B2_prefix", "B2_cz", "B2_coin", "B3", "B3_cz_prefix"]


# ---------------------------------------------------------------------------
# GF(2) helpers (the test's own)
# ---------------------------------------------------------------------------

def _rref(rows):
    R, piv = [], []
    for r in rows:
        v = r.copy()
        for k, p in enumerate(piv):
            if v[p]:
                v ^= R[k]
        nz = np.flatnonzero(v)
        if nz.size == 0:
            continue
        p = int(nz[0])
        for k in range(len(R)):
            if R[k][p]:
                R[k] ^= v
        R.append(v)
        piv.append(p)
    return R, piv


def _reduce(v, R, piv):
    v = v.copy()
    for k, p in enumerate(piv):
        if v[p]:
            v ^= R[k]
    return v


def _rank(rows):
    return len(_rref(rows)[0])


def _nullspace(rows, n):
    R, piv = _rref(rows)
    isp = set(piv)
    out = []
    for f in range(n):
        if f in isp:
            continue
        x = np.zeros(n, np.uint8)
        x[f] = 1
        for k, p in enumerate(piv):
            if R[k][f]:
                x[p] = 1
        out.append(x)
    return out


def _solve(A, b, ncols):
    """particular solution (free vars 0) of A x = b, None if inconsistent — the engine's rule"""
    red = []
    for row, rhs in zip(A, b):
        v = row.copy()
        rr = int(rhs) & 1
        for (ev, er, ep) in red:
            if v[ep]:
                v ^= ev
                rr ^= er
        nz = np.flatnonzero(v)
        if nz.size == 0:
            if rr:
                return None
            continue
        p = int(nz[0])
        for i in range(len(red)):
            if red[i][0][p]:
                red[i] = (red[i][0] ^ v, red[i][1] ^ rr, red[i][2])
        red.append((v, rr, p))
    x = np.zeros(ncols, np.uint8)
    for (ev, er, ep) in red:
        if er:
            x[ep] = 1
    return x


def _left_deps(rows):
    """combos c with sum_i c_i rows[i] = 0 (reference _left_deps / engine left_deps)"""
    red, deps = [], []
    m = len(rows)
    for i in range(m):
        v = rows[i].copy()
        c = np.zeros(m, np.uint8)
        c[i] = 1
        for (ev, ec, ep) in red:
            if v[ep]:
                v ^= ev
                c ^= ec
        nz = np.flatnonzero(v)
        if nz.size == 0:
            deps.append(c)
        else:
            red.append((v, c, int(nz[0])))
    return deps


def _words_to_bits(words, nbits):
    out = np.zeros(nbits, np.uint8)
    for i in range(nbits):
        out[i] = (int(words[i // 64]) >> (i % 64)) & 1
    return out


# ---------------------------------------------------------------------------
# exact Pauli algebra: (x, z, phase) = i^phase X^x Z^z (engine convention, pauli.hpp)
# ---------------------------------------------------------------------------

def _pmul(a, b):
    ax, az, ap = a
    bx, bz, bp = b
    return (ax ^ bx, az ^ bz, (ap + bp + 2 * int(np.sum(az & bx) & 1)) % 4)


def _pid(n):
    return (np.zeros(n, np.uint8), np.zeros(n, np.uint8), 0)


# ---------------------------------------------------------------------------
# dense statevector oracle on the physical qubits (local index = physical qubit index)
# ---------------------------------------------------------------------------

def _sim_circuit(text, nq):
    """statevector of the NOISELESS unitary part of circuit_text (RX/R/T/H/X/CX only)."""
    dim = 1 << nq
    psi = np.zeros(dim, complex)
    psi[0] = 1.0
    idx = np.arange(dim)
    bit = lambda q: (idx >> q) & 1
    for line in text.splitlines():
        t = line.split()
        if not t or t[0] in ("X_ERROR(0.01)", "MPP", "DETECTOR", "OUTPUT_QUBITS"):
            continue
        g = t[0]
        qs = [int(x) for x in t[1:]]
        if g == "RX":                          # fresh |+> (all qubits start |0>)
            q = qs[0]
            psi = (psi + psi[idx ^ (1 << q)]) / np.sqrt(2.0)
        elif g == "R":
            continue                           # already |0>
        elif g == "T":
            psi = psi * np.where(bit(qs[0]) == 1, np.exp(1j * np.pi / 4), 1.0)
        elif g == "H":
            q = qs[0]
            p0, p1 = psi * (bit(q) == 0), psi * (bit(q) == 1)
            psi = (p0 + p0[idx ^ (1 << q)] + p1[idx ^ (1 << q)] - p1) / np.sqrt(2.0)
        elif g == "X":
            psi = psi[idx ^ (1 << qs[0])]
        elif g == "CX":
            c, tg = qs
            flip = idx ^ (((idx >> c) & 1) << tg)
            psi = psi[flip]
        else:
            raise AssertionError("unexpected gate in fixture: " + g)
    assert abs(np.vdot(psi, psi) - 1) < 1e-12
    return psi


def _parity(v):
    v = v.astype(np.uint64)
    for sh in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(sh)
    return (v & np.uint64(1)).astype(np.int64)


def _apply_pauli(psi, x, z, phase, nq):
    """i^phase X^x Z^z |psi> on the local register"""
    idx = np.arange(psi.size)
    zmask = sum(1 << int(q) for q in np.flatnonzero(z))
    xmask = sum(1 << int(q) for q in np.flatnonzero(x))
    sgn = 1.0 - 2.0 * _parity(idx & zmask)
    out = psi * sgn
    out = out[idx ^ xmask]
    return out * (1j ** (phase % 4))


def _apply_residual(psi, a, cz, px, pz, nq):
    """R = P.C on the local register: C = S^a CZ(cz) first, then P = X^px Z^pz (global phase free)"""
    idx = np.arange(psi.size)
    ph = np.ones(psi.size, complex)
    for q in np.flatnonzero(a):
        ph = ph * np.where((idx >> q) & 1, 1j, 1.0)
    for (f, s) in cz:
        ph = ph * np.where(((idx >> f) & 1) & ((idx >> s) & 1), -1.0, 1.0)
    out = psi * ph
    return _apply_pauli(out, px, pz, 0, nq)


# ---------------------------------------------------------------------------
# the engine side: group rows, ancilla separation, plan export
# ---------------------------------------------------------------------------

class Plan:
    def __init__(self, B, cz, pxq, pzq, extra_s):
        text = circuit_text(B)
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
        inner = s._inner if hasattr(s, "_inner") else s
        bare = _xtim.bare_state_of(text)
        self.n = n = bare.n
        cs = bare.certified_symplectic()
        self.ng = ng = int(cs["ngens"])
        xz = np.asarray(cs["xz"]).astype(np.uint8)
        ph = np.asarray(cs["phase"]).astype(np.int64)
        self.gens = [(xz[i][:n].copy(), xz[i][n:].copy(), int(ph[i])) for i in range(ng)]
        self.wires = list(inner.output_wires())          # physical qubit q -> deferred wire
        self.nq = 4 * B
        assert len(self.wires) == self.nq and len(set(self.wires)) == self.nq
        W = lambda qs: [self.wires[q] for q in qs]
        # dense bare state on the physical register
        self.psi = _sim_circuit(text, self.nq)
        self.bare = bare
        self._separate_ancillas()
        a_q = [q for b in range(B) for q in (4 * b, 4 * b + 1)]
        a_w = W(a_q)
        self.reg = list(self.wires)                      # dense register: code wires (+ extras)
        if extra_s == "anc_x":
            # the first |+> ancilla wire that some certified generator pairs with code support:
            # an S there dresses that generator alone -> a fair coin (r >= 1), kappa untouched
            pick = None
            for w, sw in sorted(self.anc_stab.items()):
                if sw[0][w] and not sw[1][w] and any(
                        g[0][w] and any(g[0][c] for c in self.wires) for g in self.gens):
                    pick = w
                    break
            assert pick is not None, "no |+> ancilla wire rides an X generator"
            a_w.append(pick)
            self.reg.append(pick)
        else:
            a_w += W(list(extra_s))
        self._build_register()
        key = plan_key(n, a_w, [(self.wires[f], self.wires[t]) for (f, t) in cz], W(pxq), W(pzq))
        self.ps = ps = inner.plan_structure(key)
        self.r, self.kappa = int(ps["r"]), int(ps["kappa"])
        self.fallback = bool(ps["fallback"])
        self.det = _words_to_bits(np.asarray(ps["det_signs"]), ng)
        gw = int(ps["gw"])
        rows = lambda key, k: [_words_to_bits(np.asarray(ps[key]).reshape(-1)[i * gw:(i + 1) * gw], ng)
                               for i in range(k)]
        self.coins = rows("coin_masks", self.r)
        self.delta = rows("kernel_masks", self.kappa)
        self.pre = rows("kernel_pre", self.kappa)
        self.kbase = [int(b) for b in np.asarray(ps["kernel_base"])]
        kxz = np.asarray(ps["kernel_xz"]).astype(np.uint8).reshape(self.kappa, 2 * n)
        kph = np.asarray(ps["kernel_phase"]).astype(np.int64)
        self.L = [(kxz[j][:n].copy(), kxz[j][n:].copy(), int(kph[j])) for j in range(self.kappa)]
        # residual on the deferred register (what the key says) — must be code-wire only
        self.a_w = np.asarray(ps["a"]).astype(np.uint8)
        self.cz_w = [tuple(int(v) for v in row) for row in np.asarray(ps["cz"]).reshape(-1, 2)]
        pxz = np.asarray(ps["prefix_xz"]).astype(np.uint8)
        self.px_w, self.pz_w = pxz[:n], pxz[n:]

    # wire helpers --------------------------------------------------------------------------
    def _build_register(self):
        """psi_reg = psi_code (x) (product state of the extra ancilla wires in self.reg), each
        extra wire's state the +1 eigenstate of its single-wire certified stabilizer."""
        psi = self.psi
        for w in self.reg[self.nq:]:
            x, z, p = self.anc_stab[w]
            # single-wire i^p X^x Z^z: +Z->|0>, -Z->|1>, +X->|+>, -X->|->, +Y->|+i>, -Y->|-i>
            m = np.eye(2, dtype=complex)
            if z[w]:
                m = np.array([[1, 0], [0, -1]], complex) @ m
            if x[w]:
                m = np.array([[0, 1], [1, 0]], complex) @ m
            m = (1j ** p) * m
            vals, vecs = np.linalg.eigh(m)
            v1 = vecs[:, np.argmax(vals)]
            assert abs(np.max(vals) - 1) < 1e-12
            # kron: the new wire is the NEXT register bit (most significant index position)
            psi = np.kron(v1, psi)
        self.psi_reg = psi
        self.nreg = len(self.reg)
        self.reg_pos = {w: i for i, w in enumerate(self.reg)}

    def _code_local(self, x, z):
        """restrict a deferred-wire Pauli to the dense register, in register order"""
        return (np.array([x[w] for w in self.reg], np.uint8),
                np.array([z[w] for w in self.reg], np.uint8))

    def _separate_ancillas(self):
        """every deferred wire outside the code register must carry a single-wire certified
        stabilizer s_w (|0>/|+>/|i> product wire); then each generator g = (anc part)(code part)
        with the anc part a product of those s_w — the code part h (exact phase) is what the
        joint law on the code register sees (outcome(g) = outcome(h), the s_w being +1)."""
        n, ng = self.n, self.ng
        anc = [w for w in range(n) if w not in self.wires]
        self.anc_stab = {}
        rows = [np.concatenate([g[0], g[1]]) for g in self.gens]
        for w in anc:
            keep = [i for i in range(2 * n) if i not in (w, n + w)]
            deps = _left_deps([r[keep] for r in rows])
            found = None
            for c in deps:
                P = _pid(n)
                for i in np.flatnonzero(c):
                    P = _pmul(P, self.gens[i])
                if P[0][w] or P[1][w]:
                    found = P
                    break
            assert found is not None, "deferred wire %d is not a product wire" % w
            assert not np.any(np.delete(found[0], w)) and not np.any(np.delete(found[1], w))
            self.anc_stab[w] = found
        self.h = []
        for (x, z, p) in self.gens:
            P = (x.copy(), z.copy(), p)
            for w in anc:
                if x[w] or z[w]:
                    s = self.anc_stab[w]
                    assert (x[w], z[w]) == (s[0][w], s[1][w]), "generator letter != wire stabilizer"
                    P = _pmul(P, s)
            assert not any(P[0][w] or P[1][w] for w in anc)
            self.h.append(P)

    def anc_factor(self, x, z):
        """<psi_anc| (ancilla part of X^x Z^z, letters only) |psi_anc> with the i-phases of XZ"""
        val = 1.0 + 0j
        for w, s in self.anc_stab.items():
            if w in self.reg_pos or not (x[w] or z[w]):
                continue
            if (x[w], z[w]) != (s[0][w], s[1][w]):
                return 0.0
            # single-wire operator X^x Z^z vs stabilizer i^{p} X^x Z^z: <X^xZ^z> = i^{-p}
            val *= 1j ** ((-s[2]) % 4)
        return val

    def expect(self, P, psi_local=None):
        """<psi| i^p X^x Z^z |psi> on the deferred register (psi = psi_anc (x) psi_code)"""
        x, z, p = P
        psi_c = self.psi_reg if psi_local is None else psi_local
        xl, zl = self._code_local(x, z)
        v = _apply_pauli(psi_c, xl, zl, 0, self.nreg)
        return (1j ** p) * self.anc_factor(x, z) * np.vdot(psi_c, v)

    # dense law -------------------------------------------------------------------------------
    def dense_law(self):
        n, ng, nq = self.n, self.ng, self.nreg
        a_l = np.array([self.a_w[w] for w in self.reg], np.uint8)
        wpos = self.reg_pos
        assert all(w in wpos for w in np.flatnonzero(self.a_w))
        cz_l = [(wpos[f], wpos[s]) for (f, s) in self.cz_w]
        px_l, pz_l = self._code_local(self.px_w, self.pz_w)
        assert not any((self.px_w[w] or self.pz_w[w]) and w not in wpos for w in self.anc_stab)
        psi_r = _apply_residual(self.psi_reg, a_l, cz_l, px_l, pz_l, nq)
        # the register-part generators: strip only the ancilla wires OUTSIDE the register
        hl = []
        for (x, z, p) in self.gens:
            P = (x.copy(), z.copy(), p)
            for w, s in self.anc_stab.items():
                if w in wpos or not (x[w] or z[w]):
                    continue
                P = _pmul(P, s)
            assert not any((P[0][w] or P[1][w]) for w in self.anc_stab if w not in wpos)
            hl.append(self._code_local(P[0], P[1]) + (P[2],))
        # independent subset (symplectic rank), dependents with signs
        rows = [np.concatenate([x, z]) for (x, z, _) in hl]
        R, piv, ind = [], [], []
        for i, r in enumerate(rows):
            v = r.copy()
            for k, p in enumerate(piv):
                if v[p]:
                    v ^= R[k]
            nz = np.flatnonzero(v)
            if nz.size:
                p = int(nz[0])
                for k in range(len(R)):
                    if R[k][p]:
                        R[k] ^= v
                R.append(v)
                piv.append(p)
                ind.append(i)
        dep = {}
        for i in range(ng):
            if i in ind:
                continue
            # rows[i] in span of the independent rows: solve for coefficients over the basis
            A = [np.array([rows[j][c] for j in ind], np.uint8) for c in range(2 * nq)]
            t = _solve(A, rows[i], len(ind))
            assert t is not None
            P = _pid(nq)
            for jj in np.flatnonzero(t):
                P = _pmul(P, hl[ind[jj]])
            ph = (hl[i][2] - P[2]) % 4
            assert ph in (0, 2), "dependent generator with non-real relative phase"
            dep[i] = (t, ph // 2)
        law = {}
        for bits in itertools.product((0, 1), repeat=len(ind)):
            v = psi_r
            for b, i in zip(bits, ind):
                x, z, p = hl[i]
                v = (v + ((-1) ** b) * _apply_pauli(v, x, z, p, nq)) / 2.0
            pr = float(np.vdot(v, v).real)
            if pr < 1e-12:
                continue
            sigma = np.zeros(ng, np.uint8)
            for b, i in zip(bits, ind):
                sigma[i] = b
            for i, (t, s) in dep.items():
                sigma[i] = (s + sum(int(bits[jj]) for jj in np.flatnonzero(t))) % 2
            law[tuple(int(b) for b in sigma)] = law.get(tuple(int(b) for b in sigma), 0.0) + pr
        assert abs(sum(law.values()) - 1.0) < 1e-9
        return law

    # assembled law ---------------------------------------------------------------------------
    def born(self, y, L=None):
        L = self.L if L is None else L
        k = len(L)
        tot = 0.0
        for S in itertools.product((0, 1), repeat=k):
            P = _pid(self.n)
            for j in range(k):
                if S[j]:
                    P = _pmul(P, L[j])
            e = self.expect(P)
            assert abs(e.imag) < 1e-9, e
            par = sum(int(S[j]) & int(y[j]) for j in range(k)) % 2
            tot += ((-1) ** par) * e.real
        return tot / 2 ** k

    def assembled_law(self, delta=None, kbase=None, coins=None):
        delta = self.delta if delta is None else delta
        kbase = self.kbase if kbase is None else kbase
        coins = self.coins if coins is None else coins
        law = {}
        for y in itertools.product((0, 1), repeat=self.kappa):
            pb = self.born(y) / 2 ** len(coins)
            if pb <= 1e-14:
                continue
            for cb in itertools.product((0, 1), repeat=len(coins)):
                sg = self.det.copy()
                for b, m in zip(cb, coins):
                    if b:
                        sg ^= m
                for j in range(self.kappa):
                    if y[j] ^ kbase[j]:
                        sg ^= delta[j]
                key = tuple(int(b) for b in sg)
                law[key] = law.get(key, 0.0) + pb
        return law

    # the pre-change (3.1.12) fold choice, ported step by step from build_shot_law ---------
    def legacy_fold(self):
        n, ng = self.n, self.ng
        xs = [g[0] for g in self.gens]
        zs = [g[1] for g in self.gens]
        M = np.zeros((n, n), np.uint8)
        for w in np.flatnonzero(self.a_w):
            M[w, w] = 1
        for (f, s) in self.cz_w:
            M[f, s] ^= 1
            M[s, f] ^= 1
        v = [(M @ x) % 2 for x in xs]
        active = [bool(vi.any()) for vi in v]
        Gz = []
        for c in _left_deps(xs):
            zc = np.zeros(n, np.uint8)
            for i in np.flatnonzero(c):
                zc ^= zs[i]
            if zc.any():
                Gz.append(zc)
        GzR, GzP = _rref(Gz)
        aidx = [i for i in range(ng) if active[i]]
        Bred = [_reduce(v[i], GzR, GzP) for i in aidx]
        Cdet = []
        for c in _left_deps(Bred):
            cs = np.zeros(ng, np.uint8)
            for a2 in np.flatnonzero(c):
                cs[aidx[a2]] = 1
            Cdet.append(cs)
        Cdet_full = list(Cdet) + [np.eye(ng, dtype=np.uint8)[i] for i in range(ng) if not active[i]]
        qrows = [np.array([v[i][q] if active[i] else 0 for i in range(ng)], np.uint8) for q in range(n)]
        Pi = [np.array([int(v[i] @ x) % 2 for x in xs], np.uint8) for i in aidx]
        coinR, coinP = _rref([row for row in Pi if row.any()])
        Vb, _ = _rref([v[i] for i in aidx])
        prow = [np.array([int(u @ x) % 2 for x in xs], np.uint8) for u in Vb]
        Kz = []
        for c in _left_deps(prow):
            u = np.zeros(n, np.uint8)
            for k in np.flatnonzero(c):
                u ^= Vb[k]
            Kz.append(u)
        D = []
        for c in _left_deps([_reduce(u, GzR, GzP) for u in Kz]):
            d = np.zeros(n, np.uint8)
            for k in np.flatnonzero(c):
                d ^= Kz[k]
            D.append(d)
        kdirs = []
        acc = list(D)
        detR, detP = _rref(acc)
        for u in Kz:
            if _reduce(u, detR, detP).any():
                kdirs.append(u)
                acc.append(u)
                detR, detP = _rref(acc)
        assert len(coinR) == self.r and len(kdirs) == self.kappa
        Wb = _nullspace(Cdet_full, ng)
        pre, delta, trip = [], [], None
        for u in kdirs:
            mA = qrows + Cdet_full
            mb = [int(u[q]) for q in range(n)] + [0] * len(Cdet_full)
            cl = _solve(mA, mb, ng)
            if cl is not None and _reduce(cl, coinR, coinP).any():
                cj, dl = cl, cl
            else:
                cj = _solve(qrows, [int(u[q]) for q in range(n)], ng)
                assert cj is not None
                if any(int(cj @ m) % 2 for m in coinR):
                    trip = "pre_not_perp_coins"
                    break
                dl = None
                for w in Wb:
                    wr = _reduce(w, coinR, coinP)
                    if wr.any() and int(cj @ wr) % 2:
                        dl = wr
                        break
                if dl is None:
                    trip = "no_fold_dir"
                    break
            pre.append(cj)
            delta.append(dl)
        if trip is None:
            if _rank(list(coinR) + delta) != self.r + self.kappa:
                trip = "rank"
            else:
                P = [[int(pre[j] @ delta[k]) % 2 for k in range(self.kappa)] for j in range(self.kappa)]
                if P != [[int(j == k) for k in range(self.kappa)] for j in range(self.kappa)]:
                    trip = "pairing"
        return dict(kdirs=kdirs, v=v, pre=pre, delta=delta, trip=trip, Cdet_full=Cdet_full,
                    coins=coinR)


def _maxdiff(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


@pytest.fixture(scope="module", params=_FIXTURES, ids=_IDS)
def plan(request):
    return Plan(*request.param)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_fixture_is_the_intended_instance(plan):
    """kappa = B kernel directions, no fallback, the oracle statevector is stabilized (+1, exact
    phase) by every certified generator, and the oracle agrees with the engine's own bare-state
    expectations on random code-wire Paulis (the two bare states are the same state)."""
    B = plan.nq // 4
    assert plan.kappa == B and not plan.fallback
    for g in plan.gens:
        e = plan.expect(g)
        assert abs(e - 1.0) < 1e-9, (g, e)
    rng = np.random.default_rng(20261012)
    for _ in range(200):
        x = rng.integers(0, 2, plan.nq).astype(np.uint8)
        z = rng.integers(0, 2, plan.nq).astype(np.uint8)
        z &= 1 - x                                   # Hermitian X^x Z^z (no Y sites)
        xw = [plan.wires[q] for q in np.flatnonzero(x)]
        zw = [plan.wires[q] for q in np.flatnonzero(z)]
        got = plan.bare.pauli_expectation_xz(xw, zw)
        want = np.vdot(plan.psi, _apply_pauli(plan.psi, x, z, 0, plan.nq)).real
        assert abs(float(got) - want) < 1e-9, (x, z, got, want)


def test_kernel_dirs_match_engine(plan):
    """The ported lattice classification reproduces the engine's kernel directions: the
    engine's exported preimages map onto exactly the port's kernel directions (u_j = sum_i
    c_ji v_i), so the legacy-fold port below speaks about the engine's plan."""
    lg = plan.legacy_fold()
    for j in range(plan.kappa):
        u = np.zeros(plan.n, np.uint8)
        for i in np.flatnonzero(plan.pre[j]):
            u ^= lg["v"][i]
        assert np.array_equal(u, lg["kdirs"][j]), j


def test_legacy_fold_trips(plan):
    """The pre-change fold choice is NOT a dual basis on this plan (pairing != I or
    rank-deficient): the 3.1.12 engine fell back here (verified offline on the binary); the
    fixture exercises the repair."""
    lg = plan.legacy_fold()
    assert lg["trip"] in ("pairing", "rank"), lg["trip"]


def test_dual_basis_structure(plan):
    """kernel_masks are a dual basis of kernel_pre inside C_det^perp, independent of the coins."""
    lg = plan.legacy_fold()
    for k in range(plan.kappa):
        for d in lg["Cdet_full"]:
            assert int(plan.delta[k] @ d) % 2 == 0
        for j in range(plan.kappa):
            assert int(plan.pre[j] @ plan.delta[k]) % 2 == int(j == k)
        for m in plan.coins:
            assert int(plan.pre[k] @ m) % 2 == 0
    assert _rank(list(plan.coins) + list(plan.delta)) == plan.r + plan.kappa


def test_dense_law_identity(plan):
    """The assembled law equals the dense law: same support, values at 1e-9."""
    dense = plan.dense_law()
    asm = plan.assembled_law()
    assert set(dense) == set(asm), (len(dense), len(asm))
    assert _maxdiff(dense, asm) < 1e-9
    assert len(dense) == 2 ** (plan.r + plan.kappa)


def test_mutations(plan):
    """m1 legacy folds, m2 flipped base bit, m3 delta_k ^= c_j (j != k): FAIL the identity;
    m4 delta ^= coin: PASSES (coin gauge, Thm. C) — recorded, not a correctness check."""
    dense = plan.dense_law()
    lg = plan.legacy_fold()
    if len(lg["delta"]) == plan.kappa and all(d is not None for d in lg["delta"]):
        m1 = plan.assembled_law(delta=lg["delta"])
        assert _maxdiff(dense, m1) > 1e-6 or set(dense) != set(m1)
    kb = list(plan.kbase)
    kb[0] ^= 1
    m2 = plan.assembled_law(kbase=kb)
    assert _maxdiff(dense, m2) > 1e-6
    d3 = [d.copy() for d in plan.delta]
    d3[1] ^= plan.pre[0]
    m3 = plan.assembled_law(delta=d3)
    assert _maxdiff(dense, m3) > 1e-6 or set(dense) != set(m3)
    if plan.r > 0:
        d4 = [d.copy() for d in plan.delta]
        d4[0] ^= plan.coins[0]
        m4 = plan.assembled_law(delta=d4)
        assert set(dense) == set(m4) and _maxdiff(dense, m4) < 1e-9
