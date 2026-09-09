"""Consumer-side algebra on ``FramedSuperposition.branch_frames()`` exports.

See ``docs/branch_frames.md`` §2 for the contract.  The state is

    |psi> = sum_i c_i D_i |ref>,

``|ref>`` the joint +1 eigenstate of the n signed generators ``G_a`` (rows of
``stab_*``, eps folded into ``stab_phase``), ``D_i`` the branch Paulis (rows of
``branch_*``).  Every function here is pure numpy Pauli algebra on those arrays —
no engine call — and is tested bit-for-bit against the engine's own
``pauli_expectation`` (tests/test_branch_frames.py).

Pauli convention (the engine's): ``i^phase X^x Z^z``; product
``(a·b).phase = a.phase + b.phase + 2·|a.z ∧ b.x| (mod 4)``.
"""
from __future__ import annotations

import numpy as np

_I4 = np.array([1, 1j, -1, -1j], dtype=complex)


def _mul(ax, az, ap, bx, bz, bp):
    """Pauli product a·b on bit arrays; returns (x, z, phase)."""
    sign = int(np.count_nonzero(az & bx)) & 1
    return ax ^ bx, az ^ bz, (int(ap) + int(bp) + 2 * sign) & 3


def _adjoint(px, pz, pp):
    """(i^p X^x Z^z)^dagger = (-i)^p Z^z X^x = (-i)^p (-1)^{x.z} X^x Z^z."""
    xz = int(np.count_nonzero(px & pz)) & 1
    return px, pz, (-int(pp) + 2 * xz) & 3


def _anticommute(ax, az, bx, bz) -> int:
    return (int(np.count_nonzero(ax & bz)) + int(np.count_nonzero(az & bx))) & 1


def ref_expectation(fr: dict, px, pz, pp) -> complex:
    """<ref| P |ref> for P = i^pp X^px Z^pz (bit arrays over the state's n wires).

    P is in the signed stabilizer group of |ref> up to a phase iff it commutes
    with every generator; then P = i^m R with R the product of the generators
    a for which P anticommutes with the destabilizer d_a, and <ref|P|ref> = i^m.
    Otherwise 0.
    """
    px = np.asarray(px, np.uint8); pz = np.asarray(pz, np.uint8)
    n = int(fr["n"])
    sx, sz, sp = fr["stab_x"], fr["stab_z"], fr["stab_phase"]
    dx, dz = fr["destab_x"], fr["destab_z"]
    rx = np.zeros(n, np.uint8); rz = np.zeros(n, np.uint8); rp = 0
    for a in range(n):
        if _anticommute(px, pz, sx[a], sz[a]):
            return 0.0 + 0.0j
        if _anticommute(px, pz, dx[a], dz[a]):
            rx, rz, rp = _mul(rx, rz, rp, sx[a], sz[a], sp[a])
    if not (np.array_equal(rx, px) and np.array_equal(rz, pz)):
        raise AssertionError("branch_frames: P commutes with every generator but is "
                             "not their product — the export is not a full frame")
    m = (int(pp) - rp) & 3
    return complex(_I4[m])


def branch_pauli(fr: dict, i: int):
    return (np.asarray(fr["branch_x"][i], np.uint8), np.asarray(fr["branch_z"][i], np.uint8),
            int(fr["branch_phase"][i]))


def cross_term(fr: dict, i: int, j: int, px, pz, pp) -> complex:
    """<ref| D_i^dagger P D_j |ref>."""
    dix, diz, dip = _adjoint(*branch_pauli(fr, i))
    djx, djz, djp = branch_pauli(fr, j)
    qx, qz, qp = _mul(dix, diz, dip, np.asarray(px, np.uint8), np.asarray(pz, np.uint8), pp)
    qx, qz, qp = _mul(qx, qz, qp, djx, djz, djp)
    return ref_expectation(fr, qx, qz, qp)


def expectation(fr: dict, px, pz, pp=0, mask=None) -> complex:
    """<psi_S| P |psi_S> / <psi_S|psi_S> over the branch subset S (``mask``: bool
    array over branches, or None for the full state).  Exact; for the full
    state equals ``FramedSuperposition.pauli_expectation`` for P = i^pp X^px Z^pz.
    """
    c = np.asarray(fr["coeff"], complex)
    chi = c.shape[0]
    idx = np.arange(chi) if mask is None else np.nonzero(np.asarray(mask, bool))[0]
    if idx.size == 0:
        raise ValueError("branch_frames.expectation: empty branch subset")
    norm = float(np.sum(np.abs(c[idx]) ** 2))
    acc = 0.0 + 0.0j
    for i in idx:
        for j in idx:
            t = cross_term(fr, int(i), int(j), px, pz, pp)
            if t != 0:
                acc += np.conj(c[i]) * c[j] * t
    return acc / norm


def projected_expectation(fr: dict, px, pz, pp=0, sectors=()) -> tuple:
    """(<Π P Π> / <Π>, <Π>) for the Born SECTOR selected by ``sectors`` =
    [((gx, gz, gp), s), ...]: Π = Π_j (I + s_j·g_j)/2 with s_j = ±1 and each g_j a
    Hermitian Pauli that commutes with P and with the other g_j.

    This is the quantity a consumer needs when a state is a coherent
    superposition over an operator g that is NOT one of its stabilizers (a
    twirl-split "anti" check): the sector states (I ± g)/2 |psi> are the
    Born branches the downstream classifier enumerates, and their charge is
    the expectation of P in the projected state.  Expanding Π into Pauli
    products gives 2^m full-state expectations for the numerator and 2^m for
    the denominator — pure Pauli algebra on the export, no engine read.
    Raises if some g_j does not commute with P (the sector charge would be
    undefined) or if <Π> = 0 (an empty sector)."""
    px = np.asarray(px, np.uint8); pz = np.asarray(pz, np.uint8)
    n = int(fr["n"])
    gs = [(np.asarray(g[0], np.uint8), np.asarray(g[1], np.uint8), int(g[2]), int(s))
          for g, s in sectors]
    for gx, gz, _gp, _s in gs:
        if _anticommute(gx, gz, px, pz):
            raise ValueError("projected_expectation: a sector operator anticommutes with P")
    num = 0.0 + 0.0j; den = 0.0 + 0.0j
    m = len(gs)
    for J in range(1 << m):
        qx = np.zeros(n, np.uint8); qz = np.zeros(n, np.uint8); qp = 0; sgn = 1
        for j in range(m):
            if (J >> j) & 1:
                gx, gz, gp, s = gs[j]
                qx, qz, qp = _mul(qx, qz, qp, gx, gz, gp); sgn *= s
        den += sgn * expectation(fr, qx, qz, qp)
        rx, rz, rp = _mul(qx, qz, qp, px, pz, pp)
        num += sgn * expectation(fr, rx, rz, rp)
    # both sums carry the common factor 2^-m; it cancels in the ratio
    if abs(den) < 1e-12:
        raise ValueError("projected_expectation: empty sector (<Π> = 0)")
    return num / den, den / (1 << m)


def branch_sign(fr: dict, i: int, px, pz, pp=0) -> complex:
    """<psi_i| P |psi_i> for the single branch i (a stabilizer state: +-1, +-i or 0)."""
    return cross_term(fr, i, i, px, pz, pp)


def pauli_from_sites(n: int, xs=(), zs=(), ys=()):
    """Bit arrays + phase for the HERMITIAN Pauli with X on ``xs``, Z on ``zs``,
    Y on ``ys``.  A site carrying both an X and a Z bit (a Y site, whether it
    came from ``ys`` or from ``xs`` ∩ ``zs``) contributes a factor i, so the
    Hermitian canonical is ``i^{|x ∧ z|} X^x Z^z`` — the engine's own
    convention (``FramedSuperposition::pauli_expectation``)."""
    px = np.zeros(n, np.uint8); pz = np.zeros(n, np.uint8)
    for q in xs: px[q] ^= 1
    for q in zs: pz[q] ^= 1
    for q in ys: px[q] ^= 1; pz[q] ^= 1
    return px, pz, int(np.count_nonzero(px & pz)) & 3


# ─────────────────────────────────────────────────────────────────────────────
# 3.1.2 "frames at the port" (docs/branch_frames.md §7): consumer algebra on
# BarrierBuffer.frames() — the per-shot residual PREFIX P and the per-plan
# S-layer mask a.  Every retained shot's state is P · S^a · |collapsed>, and for a
# plan with kappa = 0 the collapsed sector IS the bare reference, so the state is a
# superposition of the Pauli-frame copies  P · Z^v |ref>,  v ⊆ supp(a):
#     S^a |ref> ∝ Σ_{v ⊆ supp a} (−i)^{|v|} Z^v |ref>.
# A Born SECTOR of the split ("anti") checks g_j with sign vector s selects the
# frame coset v(s) with <Z^v, g_j> = s_j ⊕ ref_j  (one GF(2) solve per plan), and the
# recipe charge of that sector on shot P is the symplectic product
#     charge(shot, O, s) = <P, O> ⊕ <Z^{v(s')}, O> ⊕ ref(O),   s' = s ⊕ <P, g>,
# — pure bit algebra, no state, no replay.  Everything below is numpy over the
# exported arrays; tested against materialize(i) and projected_expectation.
# ─────────────────────────────────────────────────────────────────────────────

def unpack_prefix(words: np.ndarray, n: int) -> tuple:
    """(prefix_x, prefix_z) as uint8 (shots, n) from frames()['prefix_words']
    (uint64 (shots, 2*pnw): X words then Z words; bit b of word w <-> wire 64*w+b).
    One vectorized np.unpackbits over the little-endian byte view (ns per shot)."""
    words = np.ascontiguousarray(np.asarray(words, np.uint64))
    shots, two_pnw = words.shape
    pnw = two_pnw // 2
    le = words.astype("<u8", copy=False).view(np.uint8).reshape(shots, two_pnw * 8)
    bits = np.unpackbits(le, axis=1, bitorder="little")          # (shots, two_pnw*64)
    return (np.ascontiguousarray(bits[:, :n]), np.ascontiguousarray(bits[:, pnw * 64: pnw * 64 + n]))


def anticommute_rows(px, pz, gx, gz):
    """<P, g> for P given as (shots, n) bit rows and one Pauli g (n,): uint8 (shots,)."""
    px = np.asarray(px, np.uint8); pz = np.asarray(pz, np.uint8)
    gx = np.asarray(gx, np.uint8); gz = np.asarray(gz, np.uint8)
    return ((px.astype(np.int64) @ gz.astype(np.int64) + pz.astype(np.int64) @ gx.astype(np.int64)) & 1).astype(np.uint8)


def plan_frame_cosets(ref: dict, a_mask) -> list:
    """The Pauli-frame SECTORS of the S-layer plan ``a`` acting on the reference state
    ``ref`` (its ``branch_frames()`` export: |T> = Σ_i c_i D_i |r>, r the frame's stabilizer
    state; chi = 1 for a stabilizer reference, chi = 2 for a magic one).

        S^a |T> ∝ Σ_{v ⊆ supp a} (−i)^{|v|} Z^v |T>,   Z^v D_i |r> = (−1)^{<Z^v,D_i>} λ_{v,c} D_i Z^{v_c} |r>

    where v_c is the representative of v's coset modulo the frame's stabilizers (same
    anticommutation pattern with the n generators) and λ_{v,c} = <r| Z^{v ⊕ v_c} |r> = ±1.
    Collecting the coefficient of D_i Z^{v_c}|r>:  A_{c,i} = c_i Σ_{v∈c} (−i)^{|v|} (−1)^{<Z^v,D_i>} λ_{v,c}.
    The Born sector c has weight Σ_i |A_{c,i}|² and state Z^{v_c} Σ_i B_{c,i} D_i |r> with
    B_{c,i} = A_{c,i} (−1)^{<Z^{v_c},D_i>}.  It is a FRAME COPY of |T> iff B_{c,i} = μ ε_i c_i
    with ε_i = (−1)^{<G,D_i>} for a product G of FREE generators (the logical-frame flips);
    then the sector's frame is F_c = Z^{v_c} · G  and every recipe charge on that sector is a
    symplectic product with F_c.  Returns, per non-zero-weight sector, a dict
    {"vz", "fx", "fz" (F_c as X/Z bit rows), "logical_flip" (G's index bits over the free
    rows), "weight", "frame_copy": bool}.  One entry = a definite plan (possibly with a
    logical flip even though the state is definite); several = the coherent sectors.
    Exact bit algebra, 2^{|a|}·chi terms per plan; no state, no replay.  COST: pure
    Python, O(2^{|a|}·chi) ref_expectation calls (median 1.4 ms, max 170 ms at |a| = 9 on
    the u2 unit) — CACHE PER PLAN (`plan_frame_cosets_cached`), never call per group or per
    shot; a compiled version is a documented follow-up (review MINOR-4)."""
    a_mask = np.asarray(a_mask, np.uint8)
    n = int(ref["n"])
    supp = np.nonzero(a_mask)[0]
    sx = np.asarray(ref["stab_x"], np.uint8); sz = np.asarray(ref["stab_z"], np.uint8)
    coef = np.asarray(ref["coeff"], complex); chi = coef.shape[0]
    bx = np.asarray(ref["branch_x"], np.uint8); bz = np.asarray(ref["branch_z"], np.uint8)
    free = [int(f) for f in ref["free"]]
    cosets = {}; order = []
    for mask in range(1 << len(supp)):
        vz = np.zeros(n, np.uint8)
        for t, q in enumerate(supp):
            if (mask >> t) & 1:
                vz[q] = 1
        key = tuple(((sx.astype(np.int64) @ vz.astype(np.int64)) & 1).tolist())
        if key not in cosets:
            cosets[key] = {"vz": vz, "A": np.zeros(chi, complex)}
            order.append(key)
        c = cosets[key]
        lam = ref_expectation(ref, np.zeros(n, np.uint8), c["vz"] ^ vz, 0)     # ±1
        amp_v = (-1j) ** int(vz.sum())
        for i in range(chi):
            sgn = -1.0 if (int(np.count_nonzero(vz & bx[i])) & 1) else 1.0    # <Z^v, D_i>
            c["A"][i] += coef[i] * amp_v * sgn * lam
    # branch index of each D_i over the free rows (sigma), for the logical-frame search
    sigma = np.asarray(ref["sigma"], np.uint8).reshape(chi, len(free))
    idx_of = {tuple(int(b) for b in sigma[i]): i for i in range(chi)}
    k = len(free)
    out = []
    for key in order:
        c = cosets[key]
        w = float(np.sum(np.abs(c["A"]) ** 2))
        if w < 1e-12:
            continue
        vz = c["vz"]
        B = np.array([c["A"][i] * (-1.0 if (int(np.count_nonzero(vz & bx[i])) & 1) else 1.0)
                      for i in range(chi)])
        # FRAME-COPY test: is  Σ_i B_i D_i|r>  ∝  F · Σ_i c_i D_i|r>  for a LOGICAL Pauli
        # F = D^x · G^z over the free rows (D^x: product of free DESTABILIZERS x_f = 1, which
        # maps branch σ -> σ ⊕ x; G^z: product of free signed GENERATORS, eigenvalue +1 on |r>,
        # anticommuting with D_i where σ_i·z is odd)?  Then the sector's frame is
        # F_c = Z^{v_c} · D^x · G^z and every recipe charge is a symplectic product with it.
        frame_copy = False; fx = np.zeros(n, np.uint8); fz = vz.copy(); flip = None
        for code in range(1 << (2 * k)):
            xb = [(code >> t) & 1 for t in range(k)]
            zb = [(code >> (k + t)) & 1 for t in range(k)]
            cprime = np.zeros(chi, complex); ok = True
            for i in range(chi):
                tgt = tuple(int(sigma[i][t]) ^ xb[t] for t in range(k))
                if tgt not in idx_of:
                    ok = False; break
                sgn = -1.0 if (sum(int(sigma[i][t]) & zb[t] for t in range(k)) & 1) else 1.0
                cprime[idx_of[tgt]] += coef[i] * sgn
            if not ok:
                continue
            # proportional up to a global phase?
            nz = np.nonzero(np.abs(cprime) > 1e-12)[0]
            if nz.size == 0 or np.any(np.abs(B[np.abs(cprime) <= 1e-12]) > 1e-9):
                continue
            mu = B[nz[0]] / cprime[nz[0]]
            if abs(mu) < 1e-12 or np.max(np.abs(B - mu * cprime)) > 1e-9:
                continue
            frame_copy = True; flip = (xb, zb)
            gx = np.zeros(n, np.uint8); gz = np.zeros(n, np.uint8)
            for t, f in enumerate(free):
                if xb[t]:
                    gx ^= np.asarray(ref["destab_x"][f], np.uint8); gz ^= np.asarray(ref["destab_z"][f], np.uint8)
                if zb[t]:
                    gx ^= sx[f]; gz ^= sz[f]
            fx = gx; fz = vz ^ gz
            break
        out.append({"vz": vz, "fx": fx, "fz": fz, "logical_flip": flip, "weight": w,
                    "frame_copy": frame_copy})
    tot = sum(c["weight"] for c in out)
    for c in out:
        c["weight"] /= tot
    return out


def lambda_functional(ref: dict, supp) -> tuple:
    """``(free_cols, lb)`` such that ``<r| Z^w |r> = (−1)^{w[free_cols] · lb}`` for every
    ``w`` supported on ``supp`` whose ``Z^w`` commutes with all n frame generators.

    PROOF (why ``λ`` is a GF(2) LINEAR functional, not a table).  Z-type Paulis commute and
    multiply without phase: ``Z^{w1} Z^{w2} = Z^{w1 ⊕ w2}``.  On the frame's stabilizer state
    ``|r⟩`` a Pauli that commutes with every generator is ± a stabilizer, so ``w ↦ ⟨r|Z^w|r⟩``
    is a map from the subgroup ``K = {w : Z^w commutes with every generator}`` to ``{±1}``, and
    for ``w1, w2 ∈ K``: ``⟨r|Z^{w1⊕w2}|r⟩ = ⟨r|Z^{w1} Z^{w2}|r⟩ = ⟨r|Z^{w1}|r⟩⟨r|Z^{w2}|r⟩``
    (both are ±1 eigen-operators of ``|r⟩``).  A group homomorphism ``K → {±1}`` is a GF(2)
    linear functional on ``K``, hence determined by its values on a basis of ``K``.  ``K`` is
    the null space of ``M = stab_x[:, supp]`` (``Z^w`` anticommutes with generator ``a`` iff
    ``stab_x[a] · w`` is odd); its RREF gives a basis indexed by the FREE columns, and one
    ``ref_expectation`` read per basis vector fixes ``λ`` on all of ``K`` — ``|supp|`` reads at
    most, instead of one read per ``v`` (``2^{|supp|}``).  Every ``w = v ⊕ v_c`` the sector
    table needs lies in ``K`` by construction (same coset ⇒ same anticommutation pattern).
    Refuses if a basis read is not ±1 (the export would not be a full frame)."""
    n = int(ref["n"])
    supp = [int(q) for q in supp]
    m = len(supp)
    sx = np.asarray(ref["stab_x"], np.uint8)
    M = np.ascontiguousarray(sx[:, supp] & 1)          # (n, m): anticommutation key = M @ w
    piv: list = []
    r = 0
    for c in range(m):
        rows = np.nonzero(M[r:, c])[0]
        if rows.size == 0:
            continue
        i = r + int(rows[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        sel = np.nonzero(M[:, c])[0]
        sel = sel[sel != r]
        if sel.size:
            M[sel] ^= M[r]
        piv.append(c)
        r += 1
        if r == M.shape[0]:
            break
    pivset = set(piv)
    free_cols = [c for c in range(m) if c not in pivset]
    lb = np.zeros(len(free_cols), np.uint8)
    zero = np.zeros(n, np.uint8)
    for t, fc in enumerate(free_cols):
        w = np.zeros(m, np.uint8)
        w[fc] = 1
        for i, pc in enumerate(piv):
            w[pc] = M[i, fc]
        wz = np.zeros(n, np.uint8)
        wz[[supp[q] for q in range(m) if w[q]]] = 1
        val = ref_expectation(ref, zero, wz, 0)
        if abs(abs(val) - 1.0) > 1e-9 or abs(val.imag) > 1e-9:
            raise ValueError("lambda_functional: <r|Z^w|r> = %r is not ±1 for a null-space "
                             "basis vector — the export is not a full frame" % (val,))
        lb[t] = 1 if val.real < 0 else 0
    return free_cols, lb


def plan_frame_cosets_fast(ref: dict, a_mask) -> list:
    """VECTORIZED :func:`plan_frame_cosets` — same sectors, same (first-encounter) order, same
    fields; the PRODUCTION path.  :func:`plan_frame_cosets` is the reference ORACLE and the
    suite pins the two equal field by field (multi-sector fixtures and synthetic masks up to
    |supp a| = 9).  The only thing that changes is HOW the ``2^{|a|}`` terms are summed:

    * coset keys, the ``(−i)^{|v|}`` amplitudes and the branch signs ``(−1)^{⟨Z^v, D_i⟩}``
      are three numpy expressions over all ``v`` at once;
    * ``λ_{v,c} = ⟨r|Z^{v ⊕ v_c}|r⟩`` is ONE GF(2) dot product per ``v`` via
      :func:`lambda_functional` (``|supp a|`` engine-free reads instead of ``2^{|supp a|}``) —
      the term that made the reference table milliseconds per plan.

    The frame-copy search over logical images ``D^x G^z`` is the reference's, verbatim
    (``2^{2k}`` cheap terms; the convention-bearing part is not re-derived)."""
    a_mask = np.asarray(a_mask, np.uint8)
    n = int(ref["n"])
    supp = np.nonzero(a_mask)[0]
    m = int(supp.size)
    sx = np.asarray(ref["stab_x"], np.uint8); sz = np.asarray(ref["stab_z"], np.uint8)
    coef = np.asarray(ref["coeff"], complex); chi = int(coef.shape[0])
    bx = np.asarray(ref["branch_x"], np.uint8)
    free = [int(f) for f in ref["free"]]
    k = len(free)

    masks = np.arange(1 << m, dtype=np.int64)
    Vb = ((masks[:, None] >> np.arange(m)) & 1).astype(np.uint8)          # (2^m, m)
    KEY = ((Vb.astype(np.int64) @ sx[:, supp].T.astype(np.int64)) & 1).astype(np.uint8)
    keyb = np.packbits(KEY, axis=1) if n else np.zeros((1 << m, 1), np.uint8)
    _u, first_idx, inv = np.unique(keyb, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    order_c = np.argsort(first_idx, kind="stable")                          # first-ENCOUNTER
    rank = np.empty(order_c.size, np.int64); rank[order_c] = np.arange(order_c.size)
    cid = rank[inv]                                                         # coset of each v
    reps = first_idx[order_c]                                               # rep v per coset
    ncos = int(order_c.size)

    free_cols, lb = lambda_functional(ref, supp)
    Wb = Vb ^ Vb[reps[cid]]                                                 # w = v ⊕ v_c
    if len(free_cols):
        lam_bit = (Wb[:, free_cols].astype(np.int64) @ lb.astype(np.int64)) & 1
    else:
        lam_bit = np.zeros(1 << m, np.int64)
    lam = (1.0 - 2.0 * lam_bit).astype(float)
    amp = ((-1j) ** Vb.sum(axis=1)).astype(complex)
    sgn = 1.0 - 2.0 * ((Vb.astype(np.int64) @ bx[:, supp].T.astype(np.int64)) & 1).astype(float)
    term = (amp * lam)[:, None] * sgn * coef[None, :]                       # (2^m, chi)
    A = np.zeros((ncos, chi), complex)
    np.add.at(A, cid, term)

    sigma = np.asarray(ref["sigma"], np.uint8).reshape(chi, k)
    idx_of = {tuple(int(b) for b in sigma[i]): i for i in range(chi)}
    out: list = []
    for c in range(ncos):
        w = float(np.sum(np.abs(A[c]) ** 2))
        if w < 1e-12:
            continue
        vz = np.zeros(n, np.uint8)
        vz[supp[Vb[reps[c]].astype(bool)]] = 1
        B = np.array([A[c][i] * (-1.0 if (int(np.count_nonzero(vz & bx[i])) & 1) else 1.0)
                      for i in range(chi)])
        frame_copy = False; fx = np.zeros(n, np.uint8); fz = vz.copy(); flip = None
        for code in range(1 << (2 * k)):
            xb = [(code >> t) & 1 for t in range(k)]
            zb = [(code >> (k + t)) & 1 for t in range(k)]
            cprime = np.zeros(chi, complex); ok = True
            for i in range(chi):
                tgt = tuple(int(sigma[i][t]) ^ xb[t] for t in range(k))
                if tgt not in idx_of:
                    ok = False; break
                s = -1.0 if (sum(int(sigma[i][t]) & zb[t] for t in range(k)) & 1) else 1.0
                cprime[idx_of[tgt]] += coef[i] * s
            if not ok:
                continue
            nz = np.nonzero(np.abs(cprime) > 1e-12)[0]
            if nz.size == 0 or np.any(np.abs(B[np.abs(cprime) <= 1e-12]) > 1e-9):
                continue
            mu = B[nz[0]] / cprime[nz[0]]
            if abs(mu) < 1e-12 or np.max(np.abs(B - mu * cprime)) > 1e-9:
                continue
            frame_copy = True; flip = (xb, zb)
            gx = np.zeros(n, np.uint8); gz = np.zeros(n, np.uint8)
            for t, f in enumerate(free):
                if xb[t]:
                    gx ^= np.asarray(ref["destab_x"][f], np.uint8); gz ^= np.asarray(ref["destab_z"][f], np.uint8)
                if zb[t]:
                    gx ^= sx[f]; gz ^= sz[f]
            fx = gx; fz = vz ^ gz
            break
        out.append({"vz": vz, "fx": fx, "fz": fz, "logical_flip": flip, "weight": w,
                    "frame_copy": frame_copy})
    tot = sum(c["weight"] for c in out)
    for c in out:
        c["weight"] /= tot
    return out


def plan_frame_cosets_cached(ref: dict, a_mask, cache: dict, *, reference: bool = False) -> list:
    """The per-plan sector table memoized in ``cache`` by the a-mask bytes (the plan datum it
    depends on, given a fixed reference); a batch has far fewer plans than groups (287 vs
    3 995 on the u2 unit).  Production path = :func:`plan_frame_cosets_fast`; pass
    ``reference=True`` for the slow oracle (:func:`plan_frame_cosets`)."""
    key = np.asarray(a_mask, np.uint8).tobytes()
    hit = cache.get(key)
    if hit is None:
        fn = plan_frame_cosets if reference else plan_frame_cosets_fast
        hit = cache[key] = fn(ref, a_mask)
    return hit


def sector_coset(cosets: list, px, pz, checks, sector_signs, ref_signs) -> dict:
    """Pick the sector of ``cosets`` (plan_frame_cosets) a Born sector selects on a shot with
    prefix (px, pz): the one whose frame P·F_c has  <P·F_c, g_j> = [s_j ≠ ref_j]  for every
    split check g_j ((gx, gz) rows; s_j the sector's eigenvalue, ref_j the reference's).
    Raises ValueError if none or more than one matches."""
    px = np.asarray(px, np.uint8).reshape(-1); pz = np.asarray(pz, np.uint8).reshape(-1)
    want = [1 if int(s) != int(r) else 0 for s, r in zip(sector_signs, ref_signs)]
    hits = []
    for c in cosets:
        ok = True
        for (gx, gz), w in zip(checks, want):
            gx = np.asarray(gx, np.uint8); gz = np.asarray(gz, np.uint8)
            bit = (int(np.count_nonzero((px ^ c["fx"]) & gz)) + int(np.count_nonzero((pz ^ c["fz"]) & gx))) & 1
            if bit != w:
                ok = False; break
        if ok:
            hits.append(c)
    if len(hits) != 1:
        raise ValueError("sector_coset: %d sector(s) match — expected exactly one" % len(hits))
    if not hits[0].get("frame_copy", False):
        # review MINOR-3: a sector that is NOT a frame copy of the reference has no single
        # frame F_c and therefore no charge to read off — refuse, never charge it silently.
        raise ValueError("sector_coset: the matched sector is not a frame copy of the reference "
                         "(frame_copy=False) — its recipe charge is undefined; materialize and "
                         "read projected_expectation instead")
    return hits[0]


def require_kappa_zero(fr: dict) -> None:
    """The consumer-side twin of BarrierBuffer.frames(allow_kappa=False): refuse, naming the
    plan ids, if any plan of a frames() export has kappa > 0 OR a fallback law (the
    ``fallback`` field, exported since review MINOR-1) — their collapsed sector depends on the
    recorded kernel-chain outcomes and is not a single frame coset (docs §7)."""
    kappa = np.asarray(fr["kappa"], np.int64)
    fallback = np.asarray(fr.get("fallback", np.zeros_like(kappa)), np.int64)
    bad = np.nonzero((kappa > 0) | (fallback != 0))[0]
    if bad.size:
        raise ValueError("frames: %d plan(s) with kappa > 0 or a fallback law (plan ids %s) — the "
                         "frame algebra does not apply; materialize those shots explicitly, never "
                         "silently" % (bad.size, ",".join(str(int(b)) for b in bad[:10])))


def frame_charge(px, pz, fz, ox, oz, ref_bit=0, fx=None):
    """The recipe charge bit of O on the frame P·F (F = X^fx Z^fz, the sector's frame from
    plan_frame_cosets):  <P, O> ⊕ <F, O> ⊕ ref_bit  (uint8, vectorized over shots when
    px/pz are 2-D).

    TWO CONVENTIONS, pick one and never both (review MINOR-2):
      * ``ref_bit=0`` — the FLIP relative to the reference: 1 iff the sector's <O> has the
        opposite sign to the reference's <O>.  This is adaptq's pin convention
        (``b = 1 iff eX·eX0 < 0``).
      * ``ref_bit = sign bit of <O> on the reference`` (1 iff <O>_ref < 0) — the ABSOLUTE
        sign bit of <O> on the sector (docs/branch_frames.md §7's ``ref(O)``).
    The two differ by exactly the reference's own bit; applying ``ref`` on top of a flip
    (or vice versa) double-counts it."""
    px = np.atleast_2d(np.asarray(px, np.uint8)); pz = np.atleast_2d(np.asarray(pz, np.uint8))
    fz = np.asarray(fz, np.uint8); ox = np.asarray(ox, np.uint8); oz = np.asarray(oz, np.uint8)
    fx = np.zeros_like(fz) if fx is None else np.asarray(fx, np.uint8)
    c = anticommute_rows(px, pz, ox, oz)
    c ^= (int(np.count_nonzero(fz & ox)) + int(np.count_nonzero(fx & oz))) & 1
    return (c ^ (ref_bit & 1)).astype(np.uint8)


def dense_state(fr: dict) -> np.ndarray:
    """Rebuild the full 2^n statevector from the export (oracle use only; n <= ~10).

    |ref> is obtained by projecting a fixed generic vector onto the joint +1
    eigenspace of the signed generators (rank one); then |psi> = sum c_i D_i |ref>.
    """
    n = int(fr["n"])
    dim = 1 << n
    X = np.array([[0, 1], [1, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
    I2 = np.eye(2, dtype=complex)

    def op(px, pz, pp):
        M = np.array([[1.0 + 0j]])
        for q in range(n):      # qubit 0 = most significant (kron order); consistent throughout
            g = I2
            if px[q] and pz[q]: g = X @ Z
            elif px[q]: g = X
            elif pz[q]: g = Z
            M = np.kron(M, g)
        return (_I4[int(pp) & 3]) * M

    v = np.ones(dim, complex) + 1j * np.linspace(0.1, 0.9, dim)
    for a in range(n):
        G = op(fr["stab_x"][a], fr["stab_z"][a], fr["stab_phase"][a])
        v = (v + G @ v) / 2.0
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        raise AssertionError("dense_state: the generic vector is orthogonal to |ref> — retry vector")
    ref = v / nv
    psi = np.zeros(dim, complex)
    c = np.asarray(fr["coeff"], complex)
    for i in range(c.shape[0]):
        D = op(fr["branch_x"][i], fr["branch_z"][i], fr["branch_phase"][i])
        psi += c[i] * (D @ ref)
    return psi
