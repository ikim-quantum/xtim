"""Engine oracles of the xdem decoder loop (SPEC §6 T3 loop part, T6) -- TEST ONLY.

Ports of loop/mlib.py (`engine_r2b_multi`, `engine_vdag_proj`) and idealcorr/ilib.py (`Circ.engine_text`) onto the
xdem export's own deferred circuit (test API `xdem_deferred_debug`) and the writer's atom map (`xdem_atom_lines`).
Everything physical is xtim's exact engine (`_exact_engine.compile_exact_branches`, forced faults, never a
statevector) on one engine text:

  the REAL deferred circuit (non-Clifford gates kept, user noise dropped), the actual faults forced as
  X/Y/Z_ERROR(0.5) sites at their deferred positions (an unfolded readout flip: a classical flip of its record's
  outcome bits, as dccore's X on a record ancilla that nothing touches again), then (V^dag) MPP of every detector,
  C_H^dag by rewind/replay,
  MPP of the pivot detectors b_i, MPP of every observable; or (R2b) a flag-controlled Pauli block at t_c and the
  detector / observable MPPs at the end.

The engine reports every value relative to the circuit's noiseless run (every forced site unfired), so the reference
is the plain circuit (rewind/replay without H: B B^-1 = 1; the R2b flag unfired: no block). A hypothesis and an
actual set are given as ATOMS: (site, addr) as `xdem_atom_lines` names them; each xdem line's representative is its
highest-p atom (ties: the first), `representative_atoms`.

  representative_atoms(text)            xdem line -> (site, addr)
  engine_vdag_proj(text, act, hyp, d, pivots)
        the sign-free projector form 2^r Pr(b = 0, o = 0 | d) (vdag REPORT §2), asserting 2^r Pr(b = 0 | d) = 1
        (both engine functions snap their float result to {0} u {2^-k} within 1e-12; callers compare exactly)
  engine_r2b(text, act, hyp, d)
        Pr(o = c ⊕ F | d): R2b's guard, c and eta^> from the C++ correction export; when applied, the Pauli at t_c =
        the product of the early hypothesis atoms' Paulis propagated through the Clifford skeleton (non-Clifford
        gates removed) to t_c, restricted to the wires not read before t_c, as a flag-controlled block. F is the
        observable frame of that Pauli against the model's K_R2b (which carries no readout flips): its own Pauli
        flips through the skeleton, plus img_a on every axis whose late exponent differs by 2 between the atoms'
        residues (computed here from the deferred gates, composed per atom by the cocycle) and eta^> (a folded E = 2
        of a line is a flip in its f, not in eta^>; a T_DAG crossed as T then S_DAG shifts both by the same 2)
  engine_beta(text, axes)               the sign bit of <prod_a Q_a> on the noiseless real circuit's output
  rep3_exact_ler(correction, p)         rep3T_pre summed LER with the exact MLE (tests/test_dc_lines.py::rep3_exact)
"""
import itertools
import math
from collections import defaultdict

import numpy as np
import stim

from xtim import _xtim
from xtim import _xtim_testapi as ta
from xtim._exact_scope import explicit_exact

from _exact_engine import compile_exact_branches
from _xdem_circuits import circuit, xdem_export
from _xdem_model import XdemModel, cancel, reduce, rref

NONCLIFFORD = {"T", "CS", "CCZ", "CH"}
INV = {"T": "T_DAG", "S": "S_DAG", "S_DAG": "S", "CS": "CS_DAG"}
SELF_INV = {"H", "X", "Y", "Z", "CX", "CZ", "CCZ", "CH"}
# a non-Clifford gate's phase in units of pi/4 on the computational basis z (the writer's PHASE table, ported from the
# dc-DEM v1 reference model); the deferred stream holds T, CS and CCZ only (T_DAG = T; S_DAG, CS_DAG = CS; CZ), the
# writer's groups are named by the user gate (T_DAG and CS_DAG too)
PHASE = {"T": lambda z: z[0], "T_DAG": lambda z: -z[0], "CS": lambda z: 2 * z[0] * z[1],
         "CS_DAG": lambda z: -2 * z[0] * z[1], "CCZ": lambda z: 4 * z[0] * z[1] * z[2]}
ARITY = {"T": 1, "T_DAG": 1, "CS": 2, "CS_DAG": 2, "CCZ": 3}
_RESIDUE = {}


def residue_terms(gate, x):
    """G X(x) G^dag = X(x) D_x with D_x = prod_v S^e_v(Z(v)) (S(Z_v) = diag i^{[v.z odd]}), the minimal-weight Z4
    solution, ties to the first in lexicographic order of e -- the writer's convention (the dc-DEM v1 reference
    model's residue_terms, ported): [(v, e)], v a bitmask over the gate's qubits"""
    key = (gate, x)
    r = _RESIDUE.get(key)
    if r is None:
        m = ARITY[gate]
        f = PHASE[gate]
        zs = list(itertools.product([0, 1], repeat=m))
        xb = [(x >> i) & 1 for i in range(m)]
        g = [(f([a ^ b for a, b in zip(z, xb)]) - f(list(z))) % 8 for z in zs]
        h = [(gi - g[0]) % 8 for gi in g]
        assert all(v % 2 == 0 for v in h)
        h = [v // 2 for v in h]
        vs = list(range(1, 1 << m))
        r = None
        for es in sorted(itertools.product(range(4), repeat=len(vs)), key=lambda e: sum(1 for t in e if t)):
            if all(sum(e * (bin(v & sum(b << i for i, b in enumerate(z))).count("1") & 1) for v, e in zip(vs, es)) % 4
                   == h[zi] for zi, z in enumerate(zs)):
                r = [(v, e) for v, e in zip(vs, es) if e]
                break
        assert r is not None, (gate, x)
        _RESIDUE[key] = r
    return r




def toint(row) -> int:
    v = 0
    for i in np.flatnonzero(np.asarray(row, bool)):
        v |= 1 << int(i)
    return v


def _odd(items):
    c = defaultdict(int)
    for x in items:
        c[x] ^= 1
    return [x for x, v in c.items() if v]


class Deferred:
    """One user circuit's xdem deferred circuit (test API xdem_deferred_debug), its xdem file and atom map."""

    def __init__(self, text: str):
        self.text = text
        d = ta.xdem_deferred_debug(text)
        self.lines = list(d["deferred_text"])
        self.N = d["num_wires"]
        self.flag = self.N                                   # the R2b block's control wire
        self.idle = self.N + 1                               # an idle wire (the MPP of an empty detector)
        self.reads = list(d["reads"])
        self.drecs = list(d["detector_recs"])
        self.orecs = list(d["observable_recs"])
        self.sites = d["sites"]
        self.folds = d["readout_folds"]
        self.tc_index = d["tc_index"]
        early = {w for _, w in self.reads[:d["early_reads"]]}
        self.live = [w for w in range(self.N) if w not in early]   # every wire not read before t_c
        self.D, self.O = len(self.drecs), len(self.orecs)
        self.end = len(self.lines)
        self.det_mpp = [self._mpp(r) for r in self.drecs]
        self.obs_mpp = [self._mpp(r) for r in self.orecs]
        self.det_pauli = [self._pauli(r) for r in self.drecs]
        self.obs_pauli = [self._pauli(r) for r in self.orecs]
        self.base = [(i, 0, l) for i, l in enumerate(self.lines)
                     if l.split()[0] not in ("NOISE", "M", "MX", "MY", "OBS")]
        self.xdem = xdem_export(text)
        self.F = XdemModel.from_text(self.xdem) if self.xdem.startswith("xdem 1") else None
        with explicit_exact():
            self.atoms = ta.xdem_atom_lines(text)
        self.atom_line = {}
        for a in self.atoms:
            self.atom_line.setdefault((a["site"], tuple(a["addr"])), a["line"])
        self._laws = {}
        self._skel = {}
        self._zend = {}
        self._sampler = None
        self._hdr = None

    # ------------------------------------------------------------------------------------------- pieces
    def _mpp(self, recs):
        if not recs:
            return f"MPP Z{self.idle}"
        return "MPP " + "*".join(f"{self.reads[r][0]}{self.reads[r][1]}" for r in recs)

    def _pauli(self, recs):
        P = stim.PauliString(self.N + 2)
        for r in recs:
            b, w = self.reads[r]
            Q = stim.PauliString(self.N + 2)
            Q[w] = b
            P *= Q
        return P

    def atom_items(self, atom):
        """[(deferred index, letter, wire)] of one atom: its Paulis at their deferred noise instruction; [] for a
        readout flip without a fold (a classical flip of its record: `record_flip`)"""
        site, addr = atom
        if site == -1:
            j = addr[1]
            if j in self.folds:
                i, w, L = self.folds[j]
                return [(i, L, w)]
            return []
        target, word = addr
        out = []
        for i, L in enumerate(word):
            if L == "I":
                continue
            pos = target if len(word) == 1 else target * len(word) + i
            hit = [(k, w, lt) for k, part in self.sites[site] for (pp, w, lt) in part if pp == pos]
            assert len(hit) == 1, ("atom position not in exactly one deferred part", atom, pos)
            k, w, lt = hit[0]
            out.append((k, "?XYZ"[lt["XYZ".index(L)]], w))
        return out

    def skeleton(self, i0: int, i1: int) -> stim.Circuit:
        """the Clifford skeleton of deferred instructions [i0, i1) (non-Clifford gates are identity there)"""
        key = (i0, i1)
        c = self._skel.get(key)
        if c is None:
            c = stim.Circuit()
            for i in range(i0, min(i1, self.end)):
                nm = self.lines[i].split()[0]
                if nm in NONCLIFFORD or nm in ("NOISE", "M", "MX", "MY", "OBS"):
                    continue
                c.append_from_stim_program_text(self.lines[i])
            c.append("I", [self.N + 1])
            self._skel[key] = c
        return c

    def pauli_of(self, items) -> stim.PauliString:
        P = stim.PauliString(self.N + 2)
        for _, L, w in items:
            Q = stim.PauliString(self.N + 2)
            Q[w] = L
            P *= Q
        return P

    # ------------------------------------------------------------------------------------------- engine texts
    def engine_text(self, act, dec=(), block=(), pivots=None, last_obs=()):
        """(text, forced addresses): the deferred real circuit with the forced items `act` [(index, letter, wire)],
        an R2b block [(letter, wire)] at t_c, the detector MPPs, C_H^dag by rewind / replay of `dec` (V^dag), the
        pivot MPPs and the MPPs of the observables not in `last_obs`; then, for `last_obs` (the lossy observables),
        the replay undone and redone with `dec` (back to the fired frame) and their MPPs. The outcome layout is always
        (detectors, pivots, observables)."""
        addrs, out = [], []

        def forced(L, w):
            addrs.append((len(addrs), 0, L))
            out.append(f"{L}_ERROR(0.5) {w}")
            out.append("TICK")

        items = [(i, s, "G", l) for i, s, l in self.base]
        items += [(i, 1, "F", (L, w)) for i, L, w in act]
        if block:
            items.append((self.tc_index, -2, "B", None))
        items.sort(key=lambda t: (t[0], t[1]))

        def emit(it, inverse=False):
            _, _, k, t = it
            if k == "G":
                if not inverse:
                    out.append(t)
                    return
                nm = t.split()[0]
                assert nm in INV or nm in SELF_INV, t
                out.append(INV.get(nm, nm) + t[len(nm):])
            elif k == "F":
                forced(*t)
            else:
                forced("X", self.flag)
                for L, w in block:
                    out.append(f"C{L} {self.flag} {w}")

        for it in items:
            emit(it)
        out += self.det_mpp
        last = sorted(set(last_obs))
        seg = segH = []
        if dec:
            ditems = [(i, 1, "F", (L, w)) for i, L, w in dec]
            k0 = min((i, s) for i, s, _, _ in ditems)
            seg = [(i, s, "G", l) for i, s, l in self.base if (i, s) >= k0]
            segH = sorted(seg + ditems, key=lambda t: (t[0], t[1]))
            for it in reversed(segH):
                emit(it, inverse=True)
            for it in seg:
                emit(it)
        piv = list(pivots or [])
        out += [self.det_mpp[p] for p in piv]
        first = [k for k in range(self.O) if k not in last]
        out += [self.obs_mpp[k] for k in first]
        if last and dec:                                   # back to the fired frame: B_H B^-1 undoes the replay
            for it in reversed(seg):
                emit(it, inverse=True)
            for it in segH:
                emit(it)
        out += [self.obs_mpp[k] for k in last]
        order = list(range(self.D)) + [("p", i) for i in range(len(piv))] + [("o", k) for k in first] + \
            [("o", k) for k in last]
        pos = {key: n for n, key in enumerate(order)}
        tot = len(order)
        out += [f"DETECTOR rec[{pos[j] - tot}]" for j in range(self.D)]
        out += [f"DETECTOR rec[{pos[('p', i)] - tot}]" for i in range(len(piv))]
        out += [f"OBSERVABLE_INCLUDE({k}) rec[{pos[('o', k)] - tot}]" for k in range(self.O)]
        return "\n".join(out) + "\n", addrs

    def law(self, text, addrs):
        """(rows, probs, B): the exact engine's factored law over (detectors, pivots, observables), reference-relative
        (branch rows, their probabilities, the uniform directions rref'd)"""
        key = (text, tuple(addrs))
        r = self._laws.get(key)
        if r is None:
            L = compile_exact_branches(text).exact_branches(addrs, expectations=[])
            rows = np.concatenate([L.rows_detectors, L.rows_observables], 1)
            U = np.concatenate([L.uniform_detectors, L.uniform_observables], 1)
            probs = list(map(float, L.rows_probabilities)) if rows.shape[0] > 1 else [1.0]
            r = ([toint(x) for x in rows], probs, rref([toint(u) for u in U]))
            self._laws[key] = r
        return r

    # ------------------------------------------------------------------------------------------- file-side helpers
    def sampler(self):
        if self._sampler is None:
            self._sampler = _xtim._XdemSampler(self.xdem, 65536)
        return self._sampler

    def header(self):
        if self._hdr is None:
            self._hdr = ta.xdem_header_debug(self.text)
        return self._hdr

    def lines_of(self, atoms):
        return cancel([self.atom_line[(s, tuple(a))] for s, a in atoms])

    def dets_row(self, d):
        return np.frombuffer(int(d).to_bytes((self.D + 7) // 8, "little"), np.uint8).reshape(1, -1)

    def z_end(self, t, qs):
        """the end Pauli of Z_qs at deferred time t (just after instruction t), through the skeleton"""
        key = (t, qs)
        P = self._zend.get(key)
        if P is None:
            Z = stim.PauliString(self.N + 2)
            for q in qs:
                Z[q] = "Z"
            P = Z.after(self.skeleton(t + 1, self.end))
            self._zend[key] = P
        return P


_DEF = {}


def deferred(text: str) -> Deferred:
    D = _DEF.get(text)
    if D is None:
        if len(_DEF) > 64:
            _DEF.clear()
        D = _DEF[text] = Deferred(text)
    return D


def representative_atoms(text: str) -> dict:
    """xdem line -> (site, addr) of its highest-p atom (ties: the first in the writer's atom order)"""
    best = {}
    for a in deferred(text).atoms:
        if a["line"] < 0:
            continue
        cur = best.get(a["line"])
        if cur is None or a["p"] > cur[0]:
            best[a["line"]] = (a["p"], (a["site"], tuple(a["addr"])))
    return {l: v[1] for l, v in best.items()}


def _dyadic(v):
    """the engine's float sums over its branches, snapped to the pass alphabet {0} u {2^-k} when within 1e-12 of it
    (SPEC §4.5: pass = 2^-k [consistent]); any other value is returned as computed (and then fails an exact compare)"""
    if v is None:
        return None
    if abs(v) <= 1e-12:
        return 0.0
    k = round(-math.log2(v))
    return 2.0 ** -k if k >= 0 and abs(v - 2.0 ** -k) <= 1e-12 else v


def _items(Dd, atoms):
    return [it for a in _odd([(s, tuple(x)) for s, x in atoms]) for it in Dd.atom_items(a)]


def _flips(Dd, atoms, pivots=(), detectors=True):
    """the classical outcome flips of the atoms' unfolded readout flips, over (detectors, pivots, observables): a
    readout flip changes its record after the measurement (dccore's X on the record ancilla, which the correction
    never touches); xtim's deferral reads an abandoned wire in place, so the engine text cannot carry it as a Pauli
    there -- it is applied to the outcome rows instead. `detectors` = False for a hypothesis (its flips are part of
    C_H^dag, after the detector MPPs)."""
    m = 0
    for s, a in _odd([(s, tuple(x)) for s, x in atoms]):
        if s != -1 or a[1] in Dd.folds:
            continue
        j = a[1]
        for k, r in enumerate(Dd.drecs):
            if detectors and j in r:
                m ^= 1 << k
        for i, p in enumerate(pivots):
            if j in Dd.drecs[p]:
                m ^= 1 << (Dd.D + i)
        for k, r in enumerate(Dd.orecs):
            if j in r:
                m ^= 1 << (Dd.D + len(pivots) + k)
    return m


def engine_vdag_proj(text: str, act: list, hyp: list, d: int, pivots: list, lossy=()):
    """2^r Pr(b = 0, o = 0 | d) from the engine (None if Pr(d) = 0), asserting 2^r Pr(b = 0 | d) = 1.

    lossy: the observables whose rotated readout is lossy under the hypothesis (SPEC §4.3 step 3). The optimal K
    leaves them alone (§7 Q4: K_diag over the non-lossy observables only, no V^dag fallback). The K-law is a
    commuting measurement set, realised in sequence: the non-lossy rotated readouts via C_H^dag and the pivots, then
    the replay undone and redone with H (back to the fired frame), then the lossy observables read plainly, LAST,
    without the hypothesis's readout flips. Read last, a lossy observable that anticommutes with an edged rotated
    readout (a CZ edge to it, ERRATA E2) is measured in the right order, so every lossy row has this engine form.
    The predicted value c on the lossy observables (ERRATA E4) is computed here, by the engine, from the same
    measurement sequence with the hypothesis's atoms as the actual error (engine_lossy_c); the C++ correction export's
    frame bits on them must equal it (asserted per call)."""
    Dd = deferred(text)
    D, r = Dd.D, len(pivots)
    etext, addrs = Dd.engine_text(_items(Dd, act), _items(Dd, hyp), pivots=pivots, last_obs=lossy)
    rows, probs, B = Dd.law(etext, addrs)
    lossy_mask = sum(1 << (D + r + k) for k in lossy)
    fl = _flips(Dd, act, pivots) ^ (_flips(Dd, hyp, pivots, detectors=False) & ~lossy_mask)
    if lossy:                                                  # ERRATA E4: c by the engine, checked on the export
        c = engine_lossy_c(text, hyp, d, pivots, lossy)
        ids = Dd.lines_of(hyp)
        fr = Dd.sampler().correction(Dd.dets_row(d), np.array([0, len(ids)], np.int64), np.array(ids, np.int32),
                                     "optimal")["frame"]
        fr = np.asarray(fr).reshape(-1)[:Dd.O]
        assert all(int(fr[k]) == c >> k & 1 for k in lossy), ("export lossy frame != engine c (E4)", ids, d, c, fr)
        fl ^= sum(1 << (D + r + k) for k in lossy if c >> k & 1)
    rows = [v ^ fl for v in rows]
    mD = (1 << D) - 1
    mDb = (1 << (D + r)) - 1
    BD = rref([g & mD for g in B.values()])
    BDb = rref([g & mDb for g in B.values()])
    pd = pdb = pdbo = 0.0
    for v, p in zip(rows, probs):
        if reduce((v ^ d) & mD, BD) == 0:
            pd += p * 2.0 ** -len(BD)
            if reduce((v ^ d) & mDb, BDb) == 0:
                pdb += p * 2.0 ** -len(BDb)
                if reduce(v ^ d, B) == 0:
                    pdbo += p * 2.0 ** -len(B)
    if pd < 1e-15:
        return None
    norm = 2.0 ** r * pdb / pd
    assert abs(norm - 1.0) < 1e-9, f"2^r Pr(b = 0 | d) = {norm}"
    return _dyadic(2.0 ** r * pdbo / pd)


def engine_lossy_c(text: str, hyp: list, d: int, pivots: list, lossy) -> int:
    """ERRATA E4's c on the lossy observables (bits k of an O-bit int), from the engine: engine_vdag_proj's sequence
    (the K-law: C_H^dag, the pivots, the non-lossy rotated readouts, then the lossy observables last in the fired
    frame) with the hypothesis's atoms forced as the actual error. Its outcomes given d and the pivot readouts b = 0
    (the score's own conditioning), projected on the lossy observables, are the support of law([Ê, K]) there; c = the
    support point x whose x ^ f_Ê (Ê's flips on the lossy observables, read off the file) is lexicographically least,
    observable 0 first; 0 there when Pr(d) = 0 (SPEC §7 Q1)."""
    Dd = deferred(text)
    D, r, O = Dd.D, len(pivots), Dd.O
    fO = Dd.F.compose(Dd.lines_of(hyp))[0] >> D
    lm = sum(1 << k for k in lossy)
    etext, addrs = Dd.engine_text(_items(Dd, hyp), _items(Dd, hyp), pivots=pivots, last_obs=lossy)
    rows, probs, B = Dd.law(etext, addrs)
    fl = _flips(Dd, hyp, pivots)
    m = ((1 << (D + r)) - 1) | (lm << (D + r))                 # the detectors, the pivots (b = 0), the lossy obs.
    Bm = rref([g & m for g in B.values()])
    pts = set()
    for v in rows:
        for x in range(1 << O):
            if x & ~lm or x in pts:
                continue
            if reduce(((v ^ fl) ^ d ^ (x << (D + r))) & m, Bm) == 0:
                pts.add(x)
    if not pts:
        return 0
    return min(pts, key=lambda x: [(x ^ fO) >> k & 1 for k in sorted(lossy)])


def vdag_args(M, H, d):
    """(pivots, lossy, edge) for engine_vdag_proj from the model oracle (tests/_xdem_model.py k_optimal): the greedy
    pivots of Omega, the lossy observables, and whether some non-lossy rotated readout has a Z on a lossy observable
    (Phi_{S_k} meets L_j: a CZ edge to it, ERRATA E2; counted by callers -- engine_vdag_proj reads the lossy
    observables last, in the fired frame, which covers these rows too)"""
    k = M.k_optimal(H, d)
    lossy = [j for j, l in enumerate(k["lossy"]) if l]
    edge = False
    for j, S in enumerate(k["S"]):
        if S is None:
            continue
        img = 0
        for a in S:
            img ^= M.img(a)
        edge |= any(img >> (M.D + l) & 1 for l in lossy)
    return k["pivots"], lossy, edge


def _late_exponents(Dd, P):
    """{axis: E mod 4} of the Pauli P at t_c through the deferred non-Clifford gates after t_c: at each gate, the
    residue D_x of P's X part x there (residue_terms; S^e(Z_v) = a rotation exp(i pi/4 (-e) Z_v) up to phase), each Z_v
    mapped to its end Pauli through the skeleton (+- one axis; an end Pauli that is no axis must be the identity)"""
    ax = {s: a for a, s in enumerate(Dd.header()["axis_paulis"])}
    E = defaultdict(int)
    Pt = P
    i = Dd.tc_index
    for t in range(Dd.tc_index, Dd.end):
        nm = Dd.lines[t].split()[0]
        if nm not in NONCLIFFORD:
            continue
        assert nm in PHASE, ("no residue rule for", Dd.lines[t])
        Pt = Pt.after(Dd.skeleton(i, t))
        i = t
        qs = [int(q) for q in Dd.lines[t].split()[1:]]
        x = sum(1 << j for j, q in enumerate(qs) if Pt[q] in (1, 2))
        if not x:
            continue
        for v, e in residue_terms(nm, x):
            Q = Dd.z_end(t, tuple(q for j, q in enumerate(qs) if v >> j & 1))
            s = str(Q)
            a = ax.get(s[1:Dd.N + 1])
            if a is None:
                assert all(c == "_" for c in s[1:]), ("a residue on a non-axis end Pauli", Dd.lines[t], s)
                continue
            E[a] -= (1 if s[0] == "+" else -1) * e
    return {a: e % 4 for a, e in E.items() if e % 4}


def r2b_block(text: str, hyp: list):
    """(block [(letter, wire)], P at t_c as a stim.PauliString, its late exponents): the early hypothesis atoms'
    Paulis propagated through the skeleton to t_c, restricted to the wires not read before t_c (an early record's
    wire carries the record; a measured wire the deferral copied stays live, e.g. as a coherent feedback control).
    The late exponents are composed per atom, in file order of their lines, by the cocycle of SPEC §1.2: atom i's own
    residues (`_late_exponents` of its restricted Pauli R_i) negated on every axis an earlier atom's R_j anticommutes
    with at the end. (One residue of the product Pauli is the same operator, but its minimal-weight phase polynomial
    can name other axes -- e.g. CCZ through two X's -- so the per-atom form is the one comparable with eta^>.)"""
    Dd = deferred(text)
    tc = Dd.F.tc
    names = Dd.header()["axis_paulis"]
    axes = [stim.PauliString("+" + nm + "__") for nm in names]
    atoms = sorted((Dd.atom_line[(s, a)], s, a) for s, a in _odd([(s, tuple(x)) for s, x in hyp]))
    P = stim.PauliString(Dd.N + 2)
    E = defaultdict(int)
    tau = 0
    tail = Dd.skeleton(Dd.tc_index, Dd.end)
    for line, s, a in atoms:
        if line >= tc:
            for i, _, _ in Dd.atom_items((s, a)):
                assert i >= Dd.tc_index, ("a late atom before t_c", s, a, i, Dd.tc_index)
            continue
        Pi = stim.PauliString(Dd.N + 2)
        for i, L, w in Dd.atom_items((s, a)):                 # (an unfolded readout flip has no Pauli here)
            assert i < Dd.tc_index, ("an early atom after t_c", s, a, i, Dd.tc_index)
            Q = stim.PauliString(Dd.N + 2)
            Q[w] = L
            Pi *= Q.after(Dd.skeleton(i + 1, Dd.tc_index))
        Ri = stim.PauliString(Dd.N + 2)
        for w in Dd.live:
            Ri[w] = Pi[w]
        for ax, e in _late_exponents(Dd, Ri).items():
            E[ax] += -e if tau >> ax & 1 else e
        end = Ri.after(tail)
        tau ^= sum(1 << k for k, Q in enumerate(axes) if not end.commutes(Q))
        P *= Ri
    block = [("_XYZ"[P[w]], w) for w in Dd.live if P[w]]
    return block, P, {ax: e % 4 for ax, e in E.items() if e % 4}


def engine_r2b(text: str, act: list, hyp: list, d: int):
    """Pr(o = c ⊕ F | d) from the engine under R2b (None if Pr(d) = 0). The guard (applied), c and eta^> come from
    the independent stim-tableau model (tests/_xdem_model.py XdemModel.k_r2b); the C++ correction export must equal
    them (asserted per call), so the engine leg also checks the export's guard, c and eta^> mod 4."""
    Dd = deferred(text)
    D, O = Dd.D, Dd.O
    ids = Dd.lines_of(hyp)
    m = Dd.F.k_r2b(ids, d)
    applied, c, eta = bool(m["applied"]), int(m["c"]), {a: e % 4 for a, e in m["eta"].items() if e % 4}
    k = Dd.sampler().correction(Dd.dets_row(d), np.array([0, len(ids)], np.int64), np.array(ids, np.int32), "r2b")
    lo, la, le = k["late_off"], k["late_axis"], k["late_eta"]
    eta_x = {int(la[j]): int(le[j]) % 4 for j in range(lo[0], lo[1]) if int(le[j]) % 4}
    c_x = sum(int(b) << j for j, b in enumerate(np.asarray(k["c"]).reshape(-1)[:O]))
    assert (bool(k["applied"][0]), c_x, eta_x) == (applied, c, eta), ("R2b export != model", ids, d)
    block, F = (), 0
    if applied:
        block, R, E = r2b_block(text, hyp)
        for a in set(E) | set(eta):
            assert E.get(a, 0) % 2 == eta.get(a, 0) % 2, ("odd late exponent differs from eta^>", a, E, eta)
            if (E.get(a, 0) - eta.get(a, 0)) % 4 == 2:
                im = Dd.F.img(a)
                assert im & ((1 << D) - 1) == 0, "a late axis with a detector image"
                F ^= im >> D
        Pend = R.after(Dd.skeleton(Dd.tc_index, Dd.end))
        for j in range(O):
            if not Pend.commutes(Dd.obs_pauli[j]):
                F ^= 1 << j
    etext, addrs = Dd.engine_text(_items(Dd, act), block=block)
    rows, probs, B = Dd.law(etext, addrs)
    fl = _flips(Dd, act)
    rows = [v ^ fl for v in rows]
    mD = (1 << D) - 1
    BD = rref([g & mD for g in B.values()])
    w = d | ((c ^ F) << D)
    pd = pdo = 0.0
    for v, p in zip(rows, probs):
        if reduce((v ^ d) & mD, BD) == 0:
            pd += p * 2.0 ** -len(BD)
            if reduce(v ^ w, B) == 0:
                pdo += p * 2.0 ** -len(B)
    return _dyadic(pdo / pd) if pd > 1e-15 else None


def engine_detector_law(text: str, act: list) -> dict:
    """{d: Pr(d)} under the actual atoms (no correction)"""
    Dd = deferred(text)
    etext, addrs = Dd.engine_text(_items(Dd, act))
    rows, probs, B = Dd.law(etext, addrs)
    fl = _flips(Dd, act)
    rows = [v ^ fl for v in rows]
    mD = (1 << Dd.D) - 1
    BD = rref([g & mD for g in B.values()])
    span = [0]
    for g in BD.values():
        span += [x ^ g for x in span]
    out = defaultdict(float)
    for v, p in zip(rows, probs):
        for e in span:
            out[(v ^ e) & mD] += p / len(span)
    return dict(out)


def engine_beta(text: str, axes: list) -> int:
    """1 iff <prod_{a in axes} Q_a> = -1 on the noiseless real circuit's output (Q_a the unsigned end Paulis of the
    writer's axes, multiplied in the given order), by the engine's absolute PAULI_EXPECTATION"""
    Dd = deferred(text)
    names = Dd.header()["axis_paulis"]
    P = stim.PauliString(Dd.N)
    for a in axes:
        P *= stim.PauliString("+" + names[a])
    assert P.sign in (1, -1), ("the axis product is not Hermitian", axes)
    terms = [f"{'_XYZ'[P[q]]}{q}" for q in range(Dd.N) if P[q]]
    if not terms:
        return int(P.sign == -1)
    body = [l for _, _, l in Dd.base]
    prog = "\n".join(body + ["PAULI_EXPECTATION(0) " + "*".join(terms)]) + "\n"
    v = float(compile_exact_branches(prog).exact_branches([]).expectations[0][0])
    assert abs(abs(v) - 1.0) < 1e-9, ("the axis product is not a stabilizer of the noiseless output", axes, v)
    return int(v * P.sign.real < 0)


# ----------------------------------------------------------------------------------------------- rep3T_pre exact LER
def rep3_exact(correction: str, p: float, hypothesis: str = "mle", engine: bool = True):
    """(LER, max |pass - engine|, (configuration, d) rows) for rep3T_pre (tests/test_dc_lines.py::rep3_exact, on the
    xdem file): every configuration of the circuit's atoms (each X_ERROR target fires alone with its p) and every d
    with Pr(d) > 0, scored by the C++ (`xdem_lines_fixed`) with the exact MLE hypothesis (stage_rules' rule: the
    largest Pr(T) * Pr(d, o | law(T)) over (T, o), ties to the lexicographically smallest ids) or the true lines
    (hypothesis "E"); engine = True also checks every row against the engine oracle of that correction."""
    from _xdem_model import affine
    text = circuit("rep3T_pre" if p == 0.001 else "rep3T_pre_p1e-2")
    assert abs(float(text.split("X_ERROR(")[1].split(")")[0]) - p) < 1e-15
    Dd = deferred(text)
    F, D, x = Dd.F, Dd.D, Dd.xdem
    L = len(F.lines)
    best = {}
    for k in range(L + 1):
        for T in itertools.combinations(range(L), k):
            pr = 1.0
            for i in range(L):
                pr *= F.lines[i][0] if i in T else 1 - F.lines[i][0]
            off, coins = affine(F.law(F.circuit(list(T))), F.n)
            span = [0]
            for b in coins.values():
                span += [y ^ b for y in span]
            for e in span:
                dd = (off ^ e) & ((1 << D) - 1)
                key = (pr / len(span), [-i for i in T])
                if dd not in best or key > best[dd][0]:
                    best[dd] = (key, list(T))
    mle = {dd: T for dd, (_, T) in best.items()}
    atoms = [(a["site"], tuple(a["addr"]), a["p"], a["line"]) for a in Dd.atoms]
    rep = representative_atoms(text)
    assert all(a[0] >= 0 for a in atoms), "rep3T_pre has no readout flips"
    rows = []
    for m in range(1 << len(atoms)):
        cfg = [atoms[j] for j in range(len(atoms)) if m >> j & 1]
        w = 1.0
        for j, a in enumerate(atoms):
            w *= a[2] if m >> j & 1 else 1 - a[2]
        act = [(a[0], a[1]) for a in cfg]
        T = cancel([a[3] for a in cfg if a[3] >= 0])
        for dd, q in engine_detector_law(text, act).items():
            if q > 0:
                rows.append((w, q, act, T, dd, mle.get(dd, []) if hypothesis == "mle" else T))
    from test_xdem_lines import csr
    fo, fi = csr([r[3] for r in rows])
    ho, hi = csr([r[5] for r in rows])
    dets = np.zeros((len(rows), (D + 7) // 8), np.uint8)
    for k, r in enumerate(rows):
        dets[k] = Dd.dets_row(r[4])[0]
    pas, _, _, _ = ta.xdem_lines_fixed(x, fo, fi, dets, ho, hi, correction)
    ler = sum(r[0] * r[1] * (1.0 - pas[k]) for k, r in enumerate(rows))
    worst = 0.0
    if engine:
        for k, (w, q, act, T, dd, h) in enumerate(rows):
            ha = [rep[i] for i in h]
            if correction == "r2b":
                e = engine_r2b(text, act, ha, dd)
            else:
                piv, lossy, _ = vdag_args(F, h, dd)
                e = engine_vdag_proj(text, act, ha, dd, piv, lossy)
            worst = max(worst, abs(e - pas[k]))
    return ler, worst, len(rows)


def rep3_exact_ler(correction: str, p: float, hypothesis: str = "mle") -> float:
    return rep3_exact(correction, p, hypothesis, engine=False)[0]
