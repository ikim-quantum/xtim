"""The structured law (xdem_kernels.h, xk_law_struct for one-word files, xk_law_struct_mw for multi-word ones): host
side.

Some files (the Bravyi-Haah distillation files are the example) have a law that factors: a small core of
gauge bits and beta_D coordinates, k "leaves" (one gauge bit g and one coordinate f each, g = f), a few "dense" axes
along one all-leaves direction, and every other axis local to the core plus one leaf. For such a file the kernel
computes the law from one word per axis *type* (the type's odd axes, one bit per leaf) instead of one elimination over
the rho coordinates: dim C, C's key and the zero decoder's verdict exactly as the generic law (xk_law_fast, or
xk_law_multi for a multi-word file: coord_words >= 2, more than 64 gauge bits or coordinates), and the
correction up to an element of the coin span (the same law: the coin word t -> t ^ t0 that maps one onto the other is
a bijection). The derivation is the GPU sampler's design note on the structured law; this module

  ``axes(t)``       every axis's kappa_a and T_a (beta coordinates) as Python ints: a one-word file's Kc and Tc; a
                    multi-word file's KcW and GcW, whose gamma coordinates are mapped to beta ones by inverting GbW,
  ``detect(t)``     finds the structure (exact rule, no tolerance; refusing is always safe),
  ``certified(t)``  checks T_a = Lambda kappa_a for one linear map (rank [kappa] = rank [kappa | T]), under which the
                    law's two internal checks cannot fail, so the kernel skips them,
  ``plan(t)``       both, plus the kernel's own limits: a ``Plan`` or None with the reason,
  ``renumber(t, plan)``  the device tables with the axes renumbered (non-leaf axes first, then each axis type's
                    leaves in slot order, so that a type's odd axes are one shift and mask of the lo plane),
  ``st_table(plan)``     the kernel's read-only table of the structure (``sct``; a multi-word plan's layout is
                    xk_law_struct_mw's),
  ``beta_rows(t)``  a multi-word file's beta vectors (rho rows of WF words: beta_j = sum_l GbW[j]_l gamma_l), the
                    rows the structured law's correction (in beta coordinates) is XORed from.

A file that is not accepted keeps the generic law (``GpuBackend.law_path`` says which is used).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

CORE_MAX = 8  # core bits beyond beta_D the detector may add
MIN_LEAVES = 4  # fewer leaves (or fewer than twice the core bits): nothing to gain
# the kernel's limits (the capacities XK_CAP_ST_* never exceed these)
ST_MAX_NC = 8  # core coordinates of the lifted core: rD + 1 (beta_D and the auxiliary coordinate tau = bit rho)
ST_MAX_NG = 8  # core gauge bits
ST_MAX_T = 32  # type records (a type whose leaves are not consecutive slots takes one record per run)
ST_MAX_N = 16  # non-leaf axes (core-only and dense)
ST_HDR = 16  # sct: header words before the records
ST_MAX_W = 4  # slot words of a multi-word file (rho <= 256: a type record's slot field is 8 bits; see st_table)


def _bits(x: int):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def _pc(x: int) -> int:
    return bin(x).count("1")


def _ints(a) -> list[int]:
    return [int(x) for x in np.asarray(a).reshape(-1)]


def _wide(a, n: int, cw: int) -> list[int]:
    """n vectors of cw u64 words (row-major) as Python ints"""
    a = np.asarray(a, np.uint64).reshape(n, cw) if n else np.zeros((0, cw), np.uint64)
    return [sum(int(a[i, w]) << (64 * w) for w in range(cw)) for i in range(n)]


def axes(t: dict) -> tuple[list[int], list[int]]:
    """(kappa_a, T_a) of every axis as Python ints, T_a in beta coordinates (beta_D = bits [0, rho - rhoL), then
    beta_L). One-word file: Kc and Tc. Multi-word file (coord_words >= 2): KcW, and GcW (T_a over the canonical basis
    gamma of span(img)) mapped to beta coordinates: GbW[j] is beta_j over gamma, so T = sum_j x_j beta_j solves
    sum_j x_j GbW[j] = GcW[a] (GbW is invertible: both are bases of span(img))."""
    cw, A = int(t["coord_words"]), int(t["A"])
    if cw < 2:
        return _ints(t["Kc"]), _ints(t["Tc"])
    rho = int(t["rho"])
    basis: dict = {}   # echelon of GbW (pivot = highest bit) with each vector's combination of the beta_j
    for j, v in enumerate(_wide(t["GbW"], rho, cw)):
        x = 1 << j
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                basis[p] = (v, x)
                break
            v ^= basis[p][0]
            x ^= basis[p][1]
    if len(basis) != rho:
        raise ValueError("structure.axes: GbW is not invertible")
    T = []
    for v in _wide(t["GcW"], A, cw):
        x = 0
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                raise ValueError("structure.axes: an axis image outside span(GbW)")
            v ^= basis[p][0]
            x ^= basis[p][1]
        T.append(x)
    return _wide(t["KcW"], A, cw), T


def beta_rows(t: dict) -> np.ndarray:
    """a multi-word file's beta vectors, u64[rho * WF] (beta_j = sum_l GbW[j]_l gamma_l, WF words each): the table the
    structured law's correction, in beta coordinates, is XORed from (the generic multi-word law XORs gamma vectors)"""
    rho, WF, cw = int(t["rho"]), int(t["WF"]), int(t["coord_words"])
    gamma = np.asarray(t["gamma"], np.uint64).reshape(rho, WF)
    out = np.zeros((rho, WF), np.uint64)
    for j, v in enumerate(_wide(t["GbW"], rho, cw)):
        for b in _bits(v):
            out[j] ^= gamma[b]
    return out.reshape(-1)


def detect(t: dict) -> dict:
    """The structure of a one-word file's law tables, or ``{"structure": False, "reason": ...}``.

    Rule (exact): the core coordinates start as beta_D ([0, rD)), the core gauge bits empty. Repeat: with Lg / Lf the
    gauge bits / coordinates outside the core, an axis is dense when |kappa_a & Lg| + |T_a & Lf| >= max(4, (|Lg| +
    |Lf|) / 4), else local; every local axis joins its non-core gauge bits and coordinates into one component. If
    every component is one gauge bit and one coordinate (a leaf), stop; else move the node of highest local degree of
    the largest offending component into the core (ties: gauge first, then lower index), refusing a coordinate
    outside beta_D or a core of more than CORE_MAX bits beyond beta_D. Then the dense axes' non-core kappa parts take
    at most one nonzero value u and their T parts at most one v; and there must be at least MIN_LEAVES leaves and
    twice as many leaves as core bits."""
    if int(t["coord_words"]) < 1 or not int(t["A"]):
        return {"structure": False, "reason": "not a fast-path file with axes"}
    r, rho, rhoL, A = int(t["r"]), int(t["rho"]), int(t["rhoL"]), int(t["A"])
    rD = rho - rhoL
    Kc, Tc = axes(t)
    D = (1 << rD) - 1
    gamma, phi = 0, D
    allg, allf = (1 << r) - 1, (1 << rho) - 1
    while True:
        Lg, Lf = allg & ~gamma, allf & ~phi
        thr = max(4, (_pc(Lg) + _pc(Lf)) / 4)
        dense = [a for a in range(A) if _pc(Kc[a] & Lg) + _pc(Tc[a] & Lf) >= thr]
        dset = set(dense)
        parent: dict = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        deg: dict = {}
        for b in _bits(Lg):
            find(("g", b))
        for b in _bits(Lf):
            find(("f", b))
        for a in range(A):
            if a in dset:
                continue
            nodes = [("g", b) for b in _bits(Kc[a] & Lg)] + [("f", b) for b in _bits(Tc[a] & Lf)]
            for x in nodes:
                deg[x] = deg.get(x, 0) + 1
            for x in nodes[1:]:
                parent[find(x)] = find(nodes[0])
        comps: dict = {}
        for x in list(parent):
            comps.setdefault(find(x), []).append(x)
        bad = [c for c in comps.values() if not (len(c) == 2 and {k for k, _ in c} == {"g", "f"})]
        if not bad:
            break
        c = max(bad, key=lambda c: (len(c), -min(b for _, b in c)))
        x = max(c, key=lambda x: (deg.get(x, 0), x[0] == "g", -x[1]))
        if x[0] == "g":
            gamma |= 1 << x[1]
        else:
            if not D >> x[1] & 1:
                return {"structure": False, "reason": f"the core would need coordinate {x[1]}, outside beta_D"}
            phi |= 1 << x[1]
        if _pc(gamma) + _pc(phi & ~D) > CORE_MAX:
            return {"structure": False, "reason": f"no core of at most {CORE_MAX} bits separates the leaves"}
    leaves = sorted(((next(b for k, b in c if k == "g"), next(b for k, b in c if k == "f")) for c in comps.values()),
                    key=lambda p: p[1])
    g2f = dict(leaves)
    us = {Kc[a] & Lg for a in dense} - {0}
    vs = {Tc[a] & Lf for a in dense} - {0}
    if len(us) > 1 or len(vs) > 1:
        return {"structure": False, "reason": "the dense axes have more than one leaf gauge part or coordinate part"}
    u = us.pop() if us else 0
    v = vs.pop() if vs else 0
    if len(leaves) < MIN_LEAVES or len(leaves) < 2 * (_pc(gamma) + _pc(phi)):
        return {"structure": False, "reason": f"{len(leaves)} leaves for a core of {_pc(gamma)} + {_pc(phi)} bits: "
                "nothing to gain"}
    cls = []
    for a in range(A):
        if a in dset:
            cls.append("dense")
            continue
        g, f = Kc[a] & Lg, Tc[a] & Lf
        slot = g2f[g.bit_length() - 1] if g else (f.bit_length() - 1 if f else None)
        cls.append("core" if slot is None else f"leaf:{slot}")
    return {"structure": True, "reason": "", "rD": rD, "core_gauge": list(_bits(gamma)), "core_coords": list(_bits(phi)),
            "leaves": leaves, "dense": dense, "u": u, "v": v, "classes": cls}


def _rank(vs) -> int:
    basis: dict = {}
    for v in vs:
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                basis[p] = v
                break
            v ^= basis[p]
    return len(basis)


def certified(t: dict) -> bool:
    """T_a = Lambda kappa_a for one linear map Lambda over every axis (rank [kappa_a] = rank [kappa_a | T_a]): then
    the law's two internal checks (odd phase on the kernel, a non-linear phase) cannot fail for any odd set"""
    r = int(t["r"])
    Kc, Tc = axes(t)
    return _rank(Kc) == _rank([k | tt << r for k, tt in zip(Kc, Tc)])


class Plan(NamedTuple):
    """The layout the structured kernel reads (``st_table``) and the axis renumbering it needs (``renumber``)."""
    perm: list  # perm[a] = the new index of axis a
    records: list  # (off, n, s0, kg, x, y, bb, tD): axes [off, off + n) are a type's leaves at slots [s0, s0 + n)
    nonleaf: list  # (T, kg, eps, bb) of the non-leaf axes, which are the new axes 0 .. len - 1
    nG: int  # core gauge bits
    gbits: list  # their gauge bits
    rD: int
    Lall: int  # the leaf slots
    u: int  # the dense direction (leaf gauge bits = leaf slots)
    sw: int = 0  # a multi-word file's slot words (ceil(rho / 64); 0: a one-word file, xk_law_struct)


def plan(t: dict) -> tuple[Plan | None, str]:
    """(the structured layout, "") for a file the structured kernel takes, else (None, the reason). One-word files
    (xk_law_struct: rho + 1 <= 64, the auxiliary coordinate is bit rho) and multi-word ones (xk_law_struct_mw: at most
    ST_MAX_W slot words; a type record never crosses a slot word, so a run of slots is cut at multiples of 64)."""
    cw, rho, r = int(t["coord_words"]), int(t["rho"]), int(t["r"])
    if not int(t["A"]) or not (cw >= 2 or (cw == 1 and t["fast"])):
        return None, "not a fast-path file with axes"
    sw = (rho + 63) // 64 if cw >= 2 else 0
    if sw > ST_MAX_W:
        return None, f"rho = {rho}: {sw} slot words (at most {ST_MAX_W})"
    if not sw and (rho + 1 > 64 or r > 64):
        return None, f"rho + 1 = {rho + 1} > 64 (the auxiliary coordinate needs a free bit)"
    d = detect(t)
    if not d["structure"]:
        return None, "no structure: " + d["reason"]
    if not certified(t):
        return None, "not certified: T_a is not a linear function of kappa_a"
    if any(g != f for g, f in d["leaves"]):
        return None, "a leaf's gauge bit differs from its coordinate"
    rD, G = d["rD"], d["core_gauge"]
    if rD + 1 > ST_MAX_NC or len(G) > ST_MAX_NG:
        return None, f"core of {rD} coordinates and {len(G)} gauge bits (at most {ST_MAX_NC - 1}, {ST_MAX_NG})"
    if sw and any(g >= 64 * sw for g in G):
        return None, "a core gauge bit past the slot words"
    Kc, Tc = axes(t)
    bb = _ints(t["bb"])
    A = int(t["A"])
    Dm = (1 << rD) - 1
    u = d["u"]
    types: dict = {}
    nonleaf = []
    for a in range(A):
        cl = d["classes"][a]
        kg = sum((Kc[a] >> g & 1) << i for i, g in enumerate(G))
        if not cl.startswith("leaf:"):
            nonleaf.append((a, Tc[a], kg, int(cl == "dense" and (Kc[a] & u) != 0), bb[a]))
            continue
        f = int(cl[5:])   # g = f
        key = (kg, Kc[a] >> f & 1, Tc[a] & Dm, Tc[a] >> f & 1, bb[a])
        occ = 0
        while f in types.get((key, occ), {}):
            occ += 1
        types.setdefault((key, occ), {})[f] = a
    if len(nonleaf) > ST_MAX_N:
        return None, f"{len(nonleaf)} non-leaf axes (at most {ST_MAX_N})"
    perm = [0] * A
    for i, (a, *_) in enumerate(nonleaf):
        perm[a] = i
    at = len(nonleaf)
    records = []
    for (key, _), m in sorted(types.items()):
        kg, x, tD, y, b = key
        slots = sorted(m)
        runs = []
        for f in slots:   # maximal runs of consecutive slots (multi-word: within one slot word)
            if runs and runs[-1][-1] == f - 1 and not (sw and f % 64 == 0):
                runs[-1].append(f)
            else:
                runs.append([f])
        for run in runs:
            records.append((at, len(run), run[0], kg, x, y, b, tD))
            for f in run:
                perm[m[f]] = at
                at += 1
    if len(records) > ST_MAX_T:
        return None, f"{len(records)} axis type records (at most {ST_MAX_T})"
    Lall = sum(1 << f for _, f in d["leaves"])
    return Plan(perm, records, [(T, kg, eps, b) for _, T, kg, eps, b in nonleaf], len(G), list(G), rD, Lall,
                u, sw), ""


def renumber(t: dict, p: Plan) -> dict:
    """the device tables with axis a renumbered p.perm[a]: Kc, Tc, bb permuted (a multi-word file: KcW, GcW and bb,
    and its beta vectors added as "beta", ``beta_rows``), every extended line's record in xa rebuilt (its exponent
    groups and tau entries regrouped by the new axis words, the records in the same order, so record offsets still
    increase with the line id) and mem_xo mapped to the new offsets. The law is the same: the composition is per axis,
    and every law object is a function of the axis sets."""
    A = int(t["A"])
    perm = p.perm
    inv = [0] * A
    for a, n in enumerate(perm):
        inv[n] = a
    out = dict(t)
    ix = np.asarray(inv, np.int64)
    for k in ("Kc", "Tc", "bb"):
        a = np.asarray(t[k])
        if len(a) == A:   # (a multi-word file's Kc, Tc are empty)
            out[k] = a[ix].copy()
    if int(t["coord_words"]) >= 2:
        cw = int(t["coord_words"])
        for k in ("KcW", "GcW"):
            out[k] = np.asarray(t[k], np.uint64).reshape(A, cw)[ix].reshape(-1).copy()
        out["beta"] = beta_rows(t)
    xa = _ints(t["xa"])
    recs = sorted({int(x) for x in np.asarray(t["mem_xo"]) if int(x)})

    def remap(pairs, words):   # [(word, mask...)] -> {new word: [masks]} with every bit moved to its new axis
        acc: dict = {}
        for w, *ms in pairs:
            for j, m in enumerate(ms):
                for b in _bits(m):
                    n = perm[64 * w + b]
                    acc.setdefault(n >> 6, [0] * words)[j] |= 1 << (n & 63)
        return sorted(acc.items())

    new = xa[:recs[0]] if recs else list(xa)
    where = {}
    for i, o in enumerate(recs):
        h = xa[o]
        nE, nT = h & 0xFFFF_FFFF, h >> 32
        end = o + 1 + 3 * nE + 2 * nT
        if i + 1 < len(recs) and end != recs[i + 1]:
            raise ValueError("renumber: the extended records are not contiguous")
        groups = remap([(xa[o + 1 + 3 * k], xa[o + 2 + 3 * k], xa[o + 3 + 3 * k]) for k in range(nE)], 2)
        taus = remap([(xa[o + 1 + 3 * nE + 2 * k], xa[o + 2 + 3 * nE + 2 * k]) for k in range(nT)], 1)
        where[o] = len(new)
        new.append(len(groups) | len(taus) << 32)
        for w, (e0, e1) in groups:
            new += [w, e0, e1]
        for w, (m,) in taus:
            new += [w, m]
        if i + 1 == len(recs):
            new += xa[end:]
    out["xa"] = np.array(new, np.uint64)
    out["mem_xo"] = np.array([where[int(x)] if int(x) else 0 for x in np.asarray(t["mem_xo"])], np.uint32)
    return out


def st_table(p: Plan) -> np.ndarray:
    """The kernel's read-only structure table ``sct`` (u64): header [0] the leaf slots Lall, [1] the dense direction
    u (leaf gauge bits = slots), [2] the non-leaf axes' mask (they are axes 0 .. N - 1), [3] 1 when some axis has a
    bb bit (the linear term of the phase), [8 + i] core gauge bit i as a one-bit word; then per type record two words
    (off | n << 16 | s0 << 24 | x << 32 | y << 33 | bb << 34 | kg << 40, and its beta_D part tD); then per non-leaf
    axis two words (T_a, and kg | eps << 8 | bb << 9).

    A multi-word plan (``p.sw`` slot words; xk_law_struct_mw): header [3] as above, [4] sw, [8 + i] core gauge bit i's
    index; then Lall and u as sw words each; the type records as above (a record's slots [s0, s0 + n) lie in one slot
    word, n <= 64); then per non-leaf axis sw + 2 words: T_a's beta_D part (bits < rD), its slot part (T_a without
    those bits) as sw words, and kg | eps << 8 | bb << 9."""
    if p.sw:
        return _st_table_mw(p)
    hdr = [0] * ST_HDR
    hdr[0] = p.Lall
    hdr[1] = p.u
    hdr[2] = (1 << len(p.nonleaf)) - 1
    hdr[3] = int(any(r[6] for r in p.records) or any(n[3] for n in p.nonleaf))
    for i, g in enumerate(p.gbits):
        hdr[8 + i] = 1 << g
    words = list(hdr)
    for off, n, s0, kg, x, y, b, tD in p.records:
        words += [off | n << 16 | s0 << 24 | x << 32 | y << 33 | b << 34 | kg << 40, tD]
    for T, kg, eps, b in p.nonleaf:
        words += [T, kg | eps << 8 | b << 9]
    return np.array(words, np.uint64)


def _sw_words(x: int, sw: int) -> list[int]:
    return [(x >> (64 * w)) & 0xFFFF_FFFF_FFFF_FFFF for w in range(sw)]


def _st_table_mw(p: Plan) -> np.ndarray:
    """st_table of a multi-word plan"""
    sw, Dm = p.sw, (1 << p.rD) - 1
    hdr = [0] * ST_HDR
    hdr[3] = int(any(r[6] for r in p.records) or any(n[3] for n in p.nonleaf))
    hdr[4] = sw
    for i, g in enumerate(p.gbits):
        hdr[8 + i] = g
    words = hdr + _sw_words(p.Lall, sw) + _sw_words(p.u, sw)
    for off, n, s0, kg, x, y, b, tD in p.records:
        if not (n <= 64 and s0 % 64 + n <= 64 and s0 < 256 and off < 1 << 16):
            raise ValueError("st_table: a type record crosses a slot word")
        words += [off | n << 16 | s0 << 24 | x << 32 | y << 33 | b << 34 | kg << 40, tD]
    for T, kg, eps, b in p.nonleaf:
        words += [T & Dm] + _sw_words(T & ~Dm, sw) + [kg | eps << 8 | b << 9]
    return np.array(words, np.uint64)
