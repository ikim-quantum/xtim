"""Exact per-fault oracle for DEM export: a branching statevector simulator (no sampling).

Independent of xtim's engine: a dense statevector over the LIVE qubits only (a qubit is
allocated at its first touch and dropped after a measurement that is followed by a reset or by
nothing), branching exactly on every measurement / reset outcome with nonzero probability.
Each final branch carries its exact probability, its records and its exact PAULI_EXPECTATION
values (frame sign applied). From that, `fault_law` returns the EXACT joint law of
(detector flips, observable flips, expectation-column sign flips, magnitude-change marker) of a
single Pauli fault inserted as a gate, relative to the noiseless circuit.

Supported: R RX RY M MX MY MR MRX MRY MPP H S S_DAG SQRT_X SQRT_X_DAG C_XYZ T T_DAG X Y Z CX CZ,
DETECTOR, OBSERVABLE_INCLUDE, PAULI_EXPECTATION (with frame records), TICK, *_COORDS.
"""
import math
import re

import numpy as np

_SQ2 = 1 / math.sqrt(2)
G1 = {
    "H": np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex),
    "S": np.diag([1, 1j]).astype(complex),
    "S_DAG": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * math.pi / 4)]).astype(complex),
    "T_DAG": np.diag([1, np.exp(-1j * math.pi / 4)]).astype(complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
    "SQRT_X": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    "SQRT_X_DAG": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex),
}
# C_XYZ: X -> Y -> Z -> X  (stim convention: equals S_DAG then H? use explicit matrix)
G1["C_XYZ"] = 0.5 * np.array([[1 - 1j, -1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
TO_Z = {"Z": np.eye(2, dtype=complex), "X": G1["H"], "Y": G1["H"] @ G1["S_DAG"]}
NOISE = ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1",
         "PAULI_CHANNEL_2")
SKIP = ("TICK", "QUBIT_COORDS", "SHIFT_COORDS")


def _split(line):
    m = re.match(r"^([A-Z_0-9]+)(?:\(([^)]*)\))?\s*(.*)$", line.strip())
    return m.group(1), m.group(2), m.group(3).split()


class State:
    """Branches of (amplitude tensor over live slots, records, expectation values)."""

    def __init__(self):
        self.slot = {}          # qubit -> axis position in the tensor
        self.order = []         # axis -> qubit
        self.br = [(np.ones((), dtype=complex), [], [])]

    # tensor helpers -------------------------------------------------------------------
    def _alloc(self, q):
        if q in self.slot:
            return
        self.slot[q] = len(self.order)
        self.order.append(q)
        nb = []
        for v, r, e in self.br:
            w = np.zeros(v.shape + (2,), dtype=complex)
            w[..., 0] = v
            nb.append((w, r, e))
        self.br = nb

    def _drop(self, q, val):
        """Remove qubit q (known to be in Z-basis state `val` on every branch)."""
        ax = self.slot.pop(q)
        self.order.pop(ax)
        for qq in self.order[ax:]:
            self.slot[qq] -= 1
        self.br = [(np.take(v, val, axis=ax), r, e) for v, r, e in self.br]

    def g1(self, U, q):
        self._alloc(q)
        ax = self.slot[q]
        self.br = [(np.moveaxis(np.tensordot(U, v, axes=([1], [ax])), 0, ax), r, e)
                   for v, r, e in self.br]

    def cx(self, c, t):
        self._alloc(c); self._alloc(t)
        a, b = self.slot[c], self.slot[t]
        nb = []
        for v, r, e in self.br:
            v = v.copy()
            idx1 = [slice(None)] * v.ndim
            idx1[a] = 1
            sub = v[tuple(idx1)]
            bb = b if b < a else b - 1
            v[tuple(idx1)] = np.flip(sub, axis=bb)
            nb.append((v, r, e))
        self.br = nb

    def cz(self, c, t):
        self._alloc(c); self._alloc(t)
        a, b = self.slot[c], self.slot[t]
        nb = []
        for v, r, e in self.br:
            v = v.copy()
            idx = [slice(None)] * v.ndim
            idx[a] = 1
            idx[b] = 1
            v[tuple(idx)] *= -1
            nb.append((v, r, e))
        self.br = nb

    def _pauli_apply(self, v, P):
        """P = [(qubit, 'X'|'Y'|'Z')] applied to tensor v (all qubits live)."""
        for q, p in P:
            ax = self.slot[q]
            v = np.moveaxis(np.tensordot(G1[p], v, axes=([1], [ax])), 0, ax)
        return v

    # measurement / reset ----------------------------------------------------------------
    def measure(self, q, basis, record, dead, reset):
        self._alloc(q)
        U = TO_Z[basis]
        ax = self.slot[q]
        nb = []
        for v, r, e in self.br:
            v = np.moveaxis(np.tensordot(U, v, axes=([1], [ax])), 0, ax)
            for out in (0, 1):
                idx = [slice(None)] * v.ndim
                idx[ax] = 1 - out
                w = v.copy()
                w[tuple(idx)] = 0
                if np.vdot(w, w).real < 1e-26:
                    continue
                if reset and out == 1:          # back to |0> in the measured basis frame
                    w = np.flip(w, axis=ax)
                    out_state = 0
                else:
                    out_state = out
                if not dead:
                    Ud = U.conj().T
                    w = np.moveaxis(np.tensordot(Ud, w, axes=([1], [ax])), 0, ax)
                nb.append((w, r + ([out] if record else []), e, out_state))
        if dead:
            # every branch is in a Z eigenstate of q (rotated frame): split by value and drop
            groups = {0: [], 1: []}
            for w, r, e, o in nb:
                groups[o].append((w, r, e))
            res = []
            ax = self.slot.pop(q)
            self.order.pop(ax)
            for qq in self.order[ax:]:
                self.slot[qq] -= 1
            for o in (0, 1):
                res += [(np.take(w, o, axis=ax), r, e) for w, r, e in groups[o]]
            self.br = res
        else:
            self.br = [(w, r, e) for w, r, e, _ in nb]

    def mpp(self, P, record=True):
        for q, _ in P:
            self._alloc(q)
        nb = []
        for v, r, e in self.br:
            pv = self._pauli_apply(v, P)
            for out, s in ((0, 1), (1, -1)):
                w = (v + s * pv) / 2
                if np.vdot(w, w).real < 1e-26:
                    continue
                nb.append((w, r + [out], e))
        self.br = nb

    def expectation(self, P, frame):
        for q, _ in P:
            self._alloc(q)
        nb = []
        for v, r, e in self.br:
            n2 = np.vdot(v, v).real
            val = np.vdot(v, self._pauli_apply(v, P)).real / n2
            if sum(r[a] for a in frame) & 1:
                val = -val
            nb.append((v, r, e + [val]))
        self.br = nb


def parse_pauli(tok):
    out = []
    for term in tok.split("*"):
        out.append((int(term[1:]), term[0]))
    return out


def _liveness(lines):
    """For each (line index, qubit) of a terminal measurement: is the qubit dead afterwards
    (next touch is a reset, or none)?"""
    touches = []
    for i, l in enumerate(lines):
        name, arg, tg = _split(l)
        if name in NOISE or name in SKIP or name in ("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION"):
            continue
        qs = []
        for t in tg:
            if t.startswith("rec["):
                continue
            qs += [int(x) for x in re.findall(r"(\d+)", t.replace("!", ""))]
        touches.append((i, name, qs))
    dead = {}
    for k, (i, name, qs) in enumerate(touches):
        if name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            for q in qs:
                nxt = None
                for j, n2, q2 in touches[k + 1:]:
                    if q in q2:
                        nxt = n2
                        break
                dead[(i, q)] = nxt is None or nxt in ("R", "RX", "RY") or name.startswith("MR")
    return dead


def run(lines, dead, snapshots=None):
    """Simulate `lines` (noise-free; Pauli gates allowed). Returns branches (prob, records, exps).
    `snapshots`: optional set of line indices; the state before each is recorded in
    `run.snaps[i]` (for resuming with `run_from`)."""
    st = State()
    run.snaps = {}
    return run_from(st, lines, dead, 0, 0, set(), snapshots)


def _copy_state(st):
    c = State()
    c.slot = dict(st.slot); c.order = list(st.order)
    c.br = [(v.copy(), list(r), list(e)) for v, r, e in st.br]
    return c


def run_from(st, lines, dead, start, rec_count, gone, snapshots=None, extra=None):
    """Continue from state `st` at line `start`; `extra` = Pauli gate lines applied first."""
    gone = set(gone)
    todo = [(None, l) for l in (extra or [])] + [(i, lines[i]) for i in range(start, len(lines))]
    for i, l in todo:
        if snapshots is not None and i in snapshots and i not in run.snaps:
            run.snaps[i] = (_copy_state(st), rec_count, set(gone))
        name, arg, tg = _split(l)
        if name in SKIP or name in ("DETECTOR", "OBSERVABLE_INCLUDE", "DECISION"):
            continue
        if name in ("R", "RX", "RY"):
            for t in tg:
                q = int(t)
                if q in st.slot:
                    st.measure(q, "Z", False, True, False)       # trace out (reset)
                gone.discard(q)
                st._alloc(q)
                if name == "RX":
                    st.g1(G1["H"], q)
                elif name == "RY":
                    st.g1(G1["H"], q); st.g1(G1["S"], q)
            continue
        if name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            basis = {"M": "Z", "MR": "Z", "MX": "X", "MRX": "X", "MY": "Y", "MRY": "Y"}[name]
            for t in tg:
                q = int(t)
                if q in gone:
                    raise ValueError("measurement of a dropped qubit")
                d = dead.get((i, q), False) if i is not None else False
                st.measure(q, basis, True, d, name.startswith("MR"))
                rec_count += 1
                if d:
                    gone.add(q)
                if name.startswith("MR"):              # measure-and-reset: fresh |0>/|+>/|+i>
                    gone.discard(q)
                    if q in st.slot:
                        st.measure(q, "Z", False, True, False)
                    st._alloc(q)
                    if name == "MRX":
                        st.g1(G1["H"], q)
                    elif name == "MRY":
                        st.g1(G1["H"], q); st.g1(G1["S"], q)
            continue
        if name == "MPP":
            for t in tg:
                st.mpp(parse_pauli(t))
                rec_count += 1
            continue
        if name == "PAULI_EXPECTATION":
            P = [x for x in tg if not x.startswith("rec[")]
            frame = [rec_count + int(x[4:-1]) for x in tg if x.startswith("rec[")]
            st.expectation(parse_pauli(P[0]), frame)
            continue
        if name in ("CX", "CY", "CZ") and any(t.startswith("rec[") for t in tg):
            for a, b in zip(tg[0::2], tg[1::2]):        # classical feedback: rec controls a Pauli
                assert a.startswith("rec[") and not b.startswith("rec["), (a, b)
                j = rec_count + int(a[4:-1]); q = int(b)
                P = {"CX": "X", "CY": "Y", "CZ": "Z"}[name]
                st._alloc(q)
                ax = st.slot[q]
                st.br = [((np.moveaxis(np.tensordot(G1[P], v, axes=([1], [ax])), 0, ax) if r[j] else v), r, e)
                         for v, r, e in st.br]
            continue
        if name in ("CX", "CZ"):
            qs = [int(t) for t in tg]
            for a, b in zip(qs[0::2], qs[1::2]):
                if a in gone or b in gone:
                    raise ValueError("2-qubit gate on a dropped qubit")
                (st.cx if name == "CX" else st.cz)(a, b)
            continue
        if name in G1:
            for t in tg:
                q = int(t)
                if q in gone:
                    if name in ("X", "Y", "Z"):
                        continue                 # a Pauli on a measured-and-dead qubit: no effect
                    raise ValueError("gate on a dropped qubit")
                st.g1(G1[name], q)
            continue
        raise ValueError(f"unsupported instruction {name}")
    return [(np.vdot(v, v).real, r, e) for v, r, e in st.br]


def strip_noise(text):
    lines = [l.split("#")[0].strip() for l in text.strip().splitlines()]
    lines = [l for l in lines if l]
    noise_idx = [i for i, l in enumerate(lines) if _split(l)[0] in NOISE]
    base = [re.sub(r"^(M|MX|MY|MR|MRX|MRY|MPP)\([^)]*\)", r"\1", l)
            for i, l in enumerate(lines) if i not in noise_idx]
    return lines, noise_idx, base


class Oracle:
    """Exact laws of single Pauli faults on `text` (detectors, deterministic observables,
    sign-constant nonzero expectation columns)."""

    def __init__(self, text):
        self.lines, self.noise_idx, self.base = strip_noise(text)
        self.dead = _liveness(self.base)
        recs, nrec = [], 0
        for l in self.base:
            name, arg, tg = _split(l)
            if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                recs.append((name, arg, [nrec + int(t[4:-1]) for t in tg if t.startswith("rec[")]))
            if name in ("M", "MX", "MY", "MR", "MRX", "MRY", "MPP"):
                nrec += len(tg)
        self.recs = recs
        self.D = sum(1 for k, *_ in recs if k == "DETECTOR")
        self.O = 1 + max([int(a) for k, a, _ in recs if k == "OBSERVABLE_INCLUDE"] or [-1])
        npos = sorted({li - sum(1 for j in self.noise_idx if j < li) for li in self.noise_idx})
        br0 = run(self.base, self.dead, snapshots=set(npos))
        self.snaps = run.snaps
        tot = sum(p for p, *_ in br0)
        assert abs(tot - 1) < 1e-10, tot
        par = [self._par(r) for _, r, _ in br0]
        self.ref = par[0]
        self.det_ok = all(par[k][d] == self.ref[d] for k in range(len(par)) for d in range(self.D))
        self.keep_obs = [o for o in range(self.O) if all(pp[self.D + o] == self.ref[self.D + o] for pp in par)]
        self.R = len(br0[0][2])
        self.v0 = []
        for c in range(self.R):
            vals = [e[c] for _, _, e in br0]
            const = max(vals) - min(vals) < 1e-9 and abs(vals[0]) > 1e-9
            self.v0.append(vals[0] if const else None)

    def _par(self, r):
        out = [0] * (self.D + self.O)
        d = 0
        for kind, arg, rr in self.recs:
            if kind == "DETECTOR":
                out[d] = sum(r[a] for a in rr) & 1
                d += 1
            else:
                out[self.D + int(arg)] ^= sum(r[a] for a in rr) & 1
        return out

    def targets(self):
        """Target ids in DEM numbering: detectors, kept observables, sign-constant columns."""
        return (list(range(self.D)) + [self.D + o for o in self.keep_obs]
                + [self.D + self.O + c for c in range(self.R) if self.v0[c] is not None])

    def law(self, pos, pauli):
        """Exact law {(frozenset(flipped targets), magnitude_changed): prob} of the Pauli fault
        {qubit: 'X'|'Y'|'Z'} inserted before base line `pos`."""
        extra = [f"{P} {q}" for q, P in sorted(pauli.items())]
        if pos in self.snaps:
            st, rc, gone = self.snaps[pos]
            br = run_from(_copy_state(st), self.base, self.dead, pos, rc, gone, extra=extra)
        else:
            fl = list(self.base)
            fl[pos:pos] = extra
            br = run(fl, _liveness(fl))
        out = {}
        for p, r, e in br:
            if p < 1e-30:
                continue
            pp = self._par(r)
            s = {t for t in range(self.D) if pp[t] != self.ref[t]}
            s |= {self.D + o for o in self.keep_obs if pp[self.D + o] != self.ref[self.D + o]}
            mag = False
            for c in range(self.R):
                if self.v0[c] is None:
                    continue
                if abs(abs(e[c]) - abs(self.v0[c])) > 1e-9:
                    mag = True
                elif (e[c] < 0) != (self.v0[c] < 0):
                    s.add(self.D + self.O + c)
            k = (frozenset(s), mag)
            out[k] = out.get(k, 0.0) + p
        return out


# ---------------------------------------------------------------------------------------------
# exporter checks against the exact laws
# ---------------------------------------------------------------------------------------------
_PC1 = ["X", "Y", "Z"]
_PC2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]


def channel_alternatives(name, arg, targets):
    ps = [float(x) for x in arg.split(",")] if arg else []
    qs = [int(t) for t in targets]
    out = []
    if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        for q in qs:
            out.append(([q], [(ps[0], {q: name[0]})]))
    elif name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
        for q in qs:
            pr = [ps[0] / 3] * 3 if name == "DEPOLARIZE1" else ps
            out.append(([q], [(pr[i], {q: _PC1[i]}) for i in range(3)]))
    else:
        for a, b in zip(qs[0::2], qs[1::2]):
            pr = [ps[0] / 15] * 15 if name == "DEPOLARIZE2" else ps
            alts = []
            for i, (pa, pb) in enumerate(_PC2):
                d = {}
                if pa != "I": d[a] = pa
                if pb != "I": d[b] = pb
                alts.append((pr[i], d))
            out.append(([a, b], alts))
    return out


def one_hot(d, qs, p):
    if len(qs) == 1:
        ps = [0.0, 0.0, 0.0]; ps[_PC1.index(d[qs[0]])] = p
        return f"PAULI_CHANNEL_1({','.join(repr(x) for x in ps)}) {qs[0]}"
    a, b = qs
    ps = [0.0] * 15; ps[_PC2.index((d.get(a, "I"), d.get(b, "I")))] = p
    return f"PAULI_CHANNEL_2({','.join(repr(x) for x in ps)}) {a} {b}"


def dem_mechs(dem_text, D):
    """{frozenset(target ids): p} of a DEM text (Dk -> k, Lk -> D+k), duplicates XOR-composed."""
    out = {}
    for line in dem_text.splitlines():
        if not line.startswith("error("):
            continue
        p = float(line[6:line.index(")")])
        ts = frozenset(int(t[1:]) if t[0] == "D" else D + int(t[1:])
                       for t in line[line.index(")") + 1:].replace("^", " ").split())
        q = out.get(ts, 0.0)
        out[ts] = q + p - 2 * q * p
    return out


def mech_dist(mechs, keep):
    dist = {frozenset(): 1.0}
    for s, p in mechs.items():
        s = s & keep
        nd = {}
        for k, q in dist.items():
            nd[k] = nd.get(k, 0.0) + q * (1 - p)
            nd[k ^ s] = nd.get(k ^ s, 0.0) + q * p
        dist = nd
    return dist


def factorable(dist):
    """Exact independent-mechanism factoring of a distribution over GF(2) signatures."""
    sigs = [s for s in dist if s]
    if not sigs:
        return True
    elems = sorted(set().union(*sigs)); idx = {e: i for i, e in enumerate(elems)}
    vec = lambda s: sum(1 << idx[e] for e in s)
    basis = []
    for s in sigs:
        v = vec(s)
        for b in basis: v = min(v, v ^ b)
        if v: basis.append(v)
    dim = len(basis)
    span = {}
    for c in range(1 << dim):
        v = 0
        for i in range(dim):
            if c >> i & 1: v ^= basis[i]
        span[v] = c
    mu = np.zeros(1 << dim)
    for s, pr in dist.items():
        mu[span[vec(s) if s else 0]] += pr
    H = np.array([[(-1) ** bin(x & c).count("1") for c in range(1 << dim)] for x in range(1 << dim)], float)
    hat = H @ mu
    if np.any(hat <= 1e-14):
        return False
    A = np.array([[bin(x & c).count("1") & 1 for c in range(1, 1 << dim)] for x in range(1 << dim)], float)
    l, *_ = np.linalg.lstsq(A, np.log(hat), rcond=None)
    return bool(np.all((1 - np.exp(l)) / 2 >= -1e-12))


def check_circuit(text, export, tol=1e-12, every=1, approx_check=True, report=None):
    """Every Pauli alternative and every channel of `text` against the exact laws.
    `export(text, approx)` -> the binding's result dict (include_expectations=True,
    drop_gauge_observables=True). Raises AssertionError on the first disagreement; returns counters."""
    orc = Oracle(text)
    assert orc.det_ok, "non-deterministic detector"
    keep = frozenset(orc.targets())
    D, O = orc.D, orc.O
    cnt = dict(alts=0, exact=0, approximated=0, flagged=0, refused_nonft=0, random=0, channels=0)
    for n_line, li in enumerate(orc.noise_idx):
        if n_line % every:
            continue
        name, arg, tg = _split(orc.lines[li])
        pos = li - sum(1 for j in orc.noise_idx if j < li)
        for qs, alts in channel_alternatives(name, arg, tg):
            cnt["channels"] += 1
            mix, ptot, special = {frozenset(): 0.0}, 0.0, False
            for p, d in alts:
                if p <= 0:
                    continue
                law = orc.law(pos, d)
                cnt["alts"] += 1
                if len(law) > 1:
                    cnt["random"] += 1
                oh = "\n".join(orc.base[:pos] + [one_hot(d, qs, 0.1)] + orc.base[pos:]) + "\n"
                r = export(oh, 0.0)
                ctx = (orc.lines[li], d, {(tuple(sorted(k[0])), k[1]): v for k, v in law.items()})
                if r["ok"] and r["postselect_faults"]:
                    cnt["flagged"] += 1; special = True
                    assert any(p2 > 0 and (k[0] & set(range(D))) for k, p2 in law.items()), ("flagged, never detected", ctx)
                    for (s, mag), p2 in law.items():
                        if not (s & set(range(D))):
                            assert not mag and not (s & set(range(D + O, D + O + orc.R))), ("flagged fault harms the accepted branch", ctx)
                    continue
                if not r["ok"]:
                    err = r["error"]
                    if "non-fault-tolerant" in err:
                        cnt["refused_nonft"] += 1; special = True
                        assert any(not (s & set(range(D))) and (mag or (s & set(range(D + O, D + O + orc.R))))
                                   for (s, mag), p2 in law.items() if p2 > 1e-15) and len(law) >= 1, ("non-FT refusal of a harmless fault", ctx)
                        continue
                    assert "negative probability" in err or "over-mixing" in err, (err, ctx)
                    assert not any(m for _, m in law), ctx
                    want = {frozenset(): 0.9}
                    for (s, _), v in law.items():
                        want[s] = want.get(s, 0.0) + 0.1 * v
                    assert not factorable(want), ("refused a factorable law", ctx)
                    if approx_check:
                        ra = export(oh, 1.0)
                        assert ra["ok"], ra["error"]
                        mech = {k & keep: v for k, v in dem_mechs(ra["dem"], D).items()}
                        for (s, _), v in law.items():
                            if s:
                                assert abs(mech.get(s, 0.0) - 0.1 * v) <= tol, ("approximation mass", s, mech.get(s), ctx)
                    cnt["approximated"] += 1
                    ptot += p
                    for (s, _), v in law.items():
                        mix[s] = mix.get(s, 0.0) + p * v
                    continue
                assert not any(m for _, m in law), ("emitted a magnitude-changing fault", ctx)
                got = mech_dist(dem_mechs(r["dem"], D), keep)
                want = {frozenset(): 0.9}
                for (s, _), v in law.items():
                    want[s] = want.get(s, 0.0) + 0.1 * v
                for k in set(got) | set(want):
                    assert abs(got.get(k, 0.0) - want.get(k, 0.0)) <= tol, ("alternative law", sorted(k), got.get(k), want.get(k), ctx)
                cnt["exact"] += 1
                ptot += p
                for (s, _), v in law.items():
                    mix[s] = mix.get(s, 0.0) + p * v
            if special:
                continue
            mix[frozenset()] += 1 - ptot
            iso = "\n".join(orc.base[:pos] + [f"{name}({arg}) " + " ".join(map(str, qs))] + orc.base[pos:]) + "\n"
            r = export(iso, 0.0)
            if r["ok"]:
                got = mech_dist(dem_mechs(r["dem"], D), keep)
                for k in set(got) | set(mix):
                    assert abs(got.get(k, 0.0) - mix.get(k, 0.0)) <= tol, ("channel law", orc.lines[li], qs, sorted(k), got.get(k), mix.get(k))
            else:
                assert not factorable(mix), ("refused a factorable channel", orc.lines[li], qs, r["error"])
    if report is not None:
        report.update(cnt)
    return cnt
