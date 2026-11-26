"""Test-only generator of VALID xdem texts (SPEC §3): random models that satisfy the header condition §3.5(b)
(img = a linear function of kappa) and the operator condition §3.5(c) (each tau is the anticommutation pattern of a
random X-string on the gauge and hidden-syndrome qubits), written in canonical spelling. Not a writer: the gauge
basis is random (any GL(r) is a valid model), so these files also exercise third-party (non-writer) gauges."""
from __future__ import annotations

import random


def _pc(x: int) -> int:
    return bin(x).count("1")


def _targets(mask: int, D: int, n: int) -> list:
    return [f"D{j}" if j < D else f"L{j - D}" for j in range(n) if mask >> j & 1]


def random_text(rng: random.Random, D: int, O: int, r: int, m: int, A: int, L: int, tc: int | None = None,
                late_ok: bool = True) -> str:
    n = D + O
    if r == 0:                                              # preflight B1: with no gauge qubit an axis needs a
        m = max(m, 1)                                       # hidden-syndrome qubit, else the axis loop never ends
    Lm = [rng.getrandbits(r) for _ in range(n)]            # img_a[j] = parity(Lm[j] & kappa_a): §3.5(b) holds
    axes = []
    while len(axes) < A:
        k = rng.getrandbits(r)
        img = sum((_pc(Lm[j] & k) & 1) << j for j in range(n))
        sig = rng.getrandbits(m) if m else 0
        if k or sig:                                        # an axis with no qubit at all is refused (§3.5(a))
            axes.append((rng.getrandbits(1), img, k, sig))
    tc = rng.randint(0, L) if tc is None else tc
    out = [f"xdem 1 detectors {D} observables {O} gauge {r} hidden {m} tc {tc}"]
    for a, (b, img, k, sig) in enumerate(axes):
        s = f"axis Q{a}" + (" sign -" if b else "")
        if img:
            s += " img " + " ".join(_targets(img, D, n))
        if k:
            s += " gauge " + " ".join(f"g{t}" for t in range(r) if k >> t & 1)
        if sig:
            s += " hidden " + " ".join(f"h{t}" for t in range(m) if sig >> t & 1)
        out.append(s)
    for i in range(L):
        chi = rng.getrandbits(r + m)                        # X-string on gauge | hidden
        tau = [a for a, (_, _, k, sig) in enumerate(axes) if _pc(chi & (k | (sig << r))) & 1]
        rot = sorted(rng.sample(range(A), rng.randint(0, min(A, 3))))
        late = sorted(rng.sample(range(A), rng.randint(0, min(A, 2)))) if late_ok else []
        s = f"error({rng.uniform(0.001, 0.2):.17g})"            # preflight B2: the emit/export spelling of p
        f = rng.getrandbits(n)
        if f:
            s += " " + " ".join(_targets(f, D, n))
        if tau:
            s += " x " + " ".join(f"Q{a}" for a in tau)
        if rot:
            s += " rot " + " ".join(rng.choice("+-") + f"Q{a}" for a in rot)
        if late:
            s += " late " + " ".join(rng.choice("+-") + f"Q{a}" for a in late)
        out.append(s)
    return "\n".join(out) + "\n"


def boundary_text(r: int, D: int, O: int, seed: int = 9171) -> str:
    """r pivot axes (axis a: gauge g_a alone, img = one readout column) plus r random non-pivot axes on the same
    gauge span; 2r rotating lines and 20 plain lines. r = 64: the sampler's one-word fast path at its bound
    (rho = dim span(img) <= r always under §3.5(b)); r = 65: two-word coordinates (the general path's stream)."""
    rng = random.Random(seed + r)
    n = D + O
    col = [rng.randrange(n) for _ in range(r)]
    axes = []
    for a in range(r):
        axes.append((rng.getrandbits(1), 1 << col[a], 1 << a))
    for _ in range(r):
        k = rng.getrandbits(r) or 1
        img = 0
        for t in range(r):
            if k >> t & 1:
                img ^= 1 << col[t]
        axes.append((rng.getrandbits(1), img, k))
    out = [f"xdem 1 detectors {D} observables {O} gauge {r} hidden 0 tc {2 * r + 20}"]
    for a, (b, img, k) in enumerate(axes):
        s = f"axis Q{a}" + (" sign -" if b else "")
        if img:
            s += " img " + " ".join(_targets(img, D, n))
        s += " gauge " + " ".join(f"g{t}" for t in range(r) if k >> t & 1)
        out.append(s)
    for a in range(2 * r):
        j = rng.randrange(n)
        out.append(f"error({rng.uniform(0.05, 0.3):.17g}) {_targets(1 << j, D, n)[0]} rot {rng.choice('+-')}Q{a}")
    for _ in range(20):
        j = rng.randrange(n)
        out.append(f"error({rng.uniform(0.01, 0.1):.17g}) {_targets(1 << j, D, n)[0]}")
    return "\n".join(out) + "\n"


def bh_like_text(k: int, lines: int = 300, seed: int = 1, signs: bool = False, extra_D: int = 8) -> str:
    """A valid xdem text shaped like a Bravyi-Haah distillation file's (merlin_bh*: k output observables), for the
    structured law (xtim.gpu.structure): a core of gauge bits g0, g1, g2 over detectors D0, D1, D2, a dense gauge bit g3
    whose image holds every observable, k leaves (leaf i < k - 1: gauge bit g_{4+i}, image D0 + L_i; leaf k - 1 reached
    through the sum of the others) with three or four axes each, and every axis but the four core / dense generators on
    its own hidden qubit. img = Lambda kappa for the linear Lambda below (§3.5(b)); each line's x set is the
    anticommutation pattern of an X-string on one or two random gauge / hidden qubits (§3.5(c)), its rot / late sets
    random subsets of the axes. signs: a random sign per kind of axis (the phase's linear term), and leaf 1's first
    axis flipped (its type then splits into runs of slots: several type records)."""
    rng = random.Random(seed)
    r = k + 3
    D, O = 3 + extra_D, k
    allL = sum(1 << (D + i) for i in range(k))
    lam = [1 << 2, 1 << 0 | 1 << 2, 1 << 1 | 1 << 2, 1 << 2 | allL] + [1 << 0 | 1 << (D + i) for i in range(k - 1)]
    G = [1 << (4 + i) for i in range(k - 1)]
    rest = sum(G) | 1 << 3   # the dense bit and every leaf bit: the last leaf's gauge part
    g0, g1, g2, g3 = 1, 2, 4, 8
    kap = [g0, g1, g2, g0 | g1 | g2, g3, g0 | g1 | g3, g0 | g2 | g3, g1 | g2 | g3]
    kind = list(range(8))
    for x in G:
        kap += [x, g1 | g2 | x, g0 | g2 | x]
        kind += [8, 9, 10]
    kap += [g0 | rest, g0 | g1 | g2 | rest, g2 | rest]
    kap += [g0 | g1 | x for x in G] + [g1 | rest]
    kind += [11, 12, 13] + [14] * len(G) + [15]
    sgn = [rng.getrandbits(1) if signs else 0 for _ in range(16)]
    axes = []
    m = 0
    for a, kk in enumerate(kap):
        img = 0
        for t in range(r):
            if kk >> t & 1:
                img ^= lam[t]
        hid = 0
        if a not in (0, 1, 2, 4):
            hid = 1 << m
            m += 1
        axes.append((sgn[kind[a]] ^ int(signs and a == 11), img, kk, hid))
    out = [f"xdem 1 detectors {D} observables {O} gauge {r} hidden {m} tc {lines}"]
    n = D + O
    for a, (b, img, kk, hid) in enumerate(axes):
        s = f"axis Q{a}" + (" sign -" if b else "")
        s += " img " + " ".join(_targets(img, D, n))
        s += " gauge " + " ".join(f"g{t}" for t in range(r) if kk >> t & 1)
        if hid:
            s += " hidden " + " ".join(f"h{t}" for t in range(m) if hid >> t & 1)
        out.append(s)
    A = len(axes)
    for _ in range(lines):
        chi = 0
        for _ in range(rng.randint(1, 2)):
            chi |= 1 << rng.randrange(r + m)
        tau = [a for a, (_, _, kk, hid) in enumerate(axes) if _pc(chi & (kk | hid << r)) & 1]
        rot = sorted(rng.sample(tau, rng.randint(0, len(tau)))) if tau else []
        late = sorted(rng.sample(range(A), rng.randint(0, 1)))
        s = f"error({rng.uniform(0.001, 0.05):.17g})"
        f = 0
        for _ in range(rng.randint(0, 3)):
            f |= 1 << rng.randrange(n)
        if f:
            s += " " + " ".join(_targets(f, D, n))
        if tau:
            s += " x " + " ".join(f"Q{a}" for a in tau)
        if rot:
            s += " rot " + " ".join(rng.choice("+-") + f"Q{a}" for a in rot)
        if late:
            s += " late " + " ".join(rng.choice("+-") + f"Q{a}" for a in late)
        out.append(s)
    return "\n".join(out) + "\n"
