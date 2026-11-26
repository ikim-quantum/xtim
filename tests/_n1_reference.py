"""lean N1 (owner-signed 2026-11-07, "Signing b"): a PURE-PYTHON reference of the twirl sampler's NOISE stream
(cpp/include/qeccore/noise_chain.hpp), written from the contract, not from the C++:

* the mechanism tables from the channel table (per flat alternative: p, channel) and the per-alternative records
  (u32 words; NULL = all zero; two alternatives have the same effect iff their records are identical):
  PRUNED = per channel the non-null alternatives (p > 0) grouped by identical record, in first-appearance order, each
  class's probability the sum of its alternatives' (alternative order), its representative the first alternative;
  FULL = per channel every alternative with p > 0. Mechanism P = min(1, sum of its outcome probabilities in order); a
  Vose alias table per mechanism in outcome order; groups = mechanisms of bitwise-equal P in ascending P bits, channel
  order inside a group;
* the chain of block b (4096 shots, aligned on the cumulative shot index): per group g one chain on its own counter
  range of the tag's keyed sequence (lean N2, owner-signed 2026-11-07): draw c = mix(key(seed, tag) + (b*R + r0_g + c + 1)*G)
  with r_g = 2 * M_g * 4096 + 1 (M_g = the group's mechanism count), r0_g = sum of r over the groups before g, R = sum of
  all r_g; walked mechanism-major over (mechanism x shot) cells by inverse-CDF geometric gaps; a fired cell of a
  multi-outcome mechanism draws its outcome (alias) on the next word; a P >= 1 group fires every cell without a gap word.
  A block with (b + 1) * R > 2^64 is refused.
"""
import math
import struct

M64 = (1 << 64) - 1
G = 0x9E3779B97F4A7C15
NOISE, NOISE_FULL = 1, 9
B = 4096


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def stream_key(seed, tag):
    return mix(seed ^ mix(tag))


def _bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _alias(w):
    n = len(w)
    W = 0.0
    for x in w:
        W += x
    pr = [x * n / W for x in w]
    sm = [i for i in range(n) if pr[i] < 1.0]
    lg = [i for i in range(n) if not pr[i] < 1.0]
    prob, alias = [0.0] * n, [0] * n
    while sm and lg:
        l = sm.pop()
        g = lg.pop()
        prob[l] = pr[l]
        alias[l] = g
        pr[g] = (pr[g] + pr[l]) - 1.0
        (sm if pr[g] < 1.0 else lg).append(g)
    for i in lg + sm:
        prob[i] = 2.0
        alias[i] = i
    return prob, alias


def tables(alt_p, alt_k, rec, off):
    """(pruned, full) as dicts with the engine's TA.noise_table layout."""
    chans = {}
    for a, k in enumerate(alt_k):
        chans.setdefault(k, []).append(a)
    recs = [tuple(int(x) for x in rec[off[a]:off[a + 1]]) for a in range(len(alt_k))]

    def build(tag, full):
        T = dict(tag=tag, mo0=[0], oalt=[], ow=[], oprob=[], oalias=[], mch=[], mP=[])
        for k in sorted(chans):
            alts, w = [], []
            for a in chans[k]:
                if not alt_p[a] > 0.0:
                    continue
                if full:
                    alts.append(a); w.append(alt_p[a]); continue
                if not any(recs[a]):
                    continue                                   # null: no record column sees it
                for j, r in enumerate(alts):
                    if recs[r] == recs[a]:
                        w[j] += alt_p[a]; break
                else:
                    alts.append(a); w.append(alt_p[a])
            if not alts:
                continue
            P = 0.0
            for x in w:
                P += x
            pr, al = _alias(w)
            T["oalt"] += alts; T["ow"] += w; T["oprob"] += pr; T["oalias"] += al
            T["mo0"].append(len(T["oalt"]))
            T["mch"].append(k)
            T["mP"].append(min(1.0, P))
        byp = {}
        for m, P in enumerate(T["mP"]):
            byp.setdefault(_bits(P), []).append(m)
        T["groups"], T["gm"], T["events"], T["r0"], T["r"] = [], [], 0.0, [], []
        for bits in sorted(byp):
            P = T["mP"][byp[bits][0]]
            sc = 1.0 if P >= 1.0 else 1.0 / math.log1p(-P)
            m0 = len(T["gm"])
            T["gm"] += byp[bits]
            T["groups"].append((P, sc, m0, len(T["gm"])))
            T["events"] += P * len(byp[bits])
            T["r0"].append(sum(T["r"]))
            T["r"].append(2 * len(byp[bits]) * B + 1)
        T["R"] = sum(T["r"])
        T["blast"] = (1 << 64) // T["R"] - 1 if T["R"] else M64     # the last block b with (b + 1) * R <= 2^64
        return T

    return build(NOISE, False), build(NOISE_FULL, True)


def chain_ctr0(T, g, b):
    """The first counter of chain (group g, block b)."""
    assert (b + 1) * T["R"] <= 1 << 64, "refused: the block's counter range passes 2^64"
    return b * T["R"] + T["r0"][g]


def block(T, seed, b, used=None):
    """Block b's events [(alt, s)] in chain order; `used` (a list) receives each chain's draw count."""
    key = stream_key(seed, T["tag"])
    mo0, oalt, opr, oal, gm = T["mo0"], T["oalt"], T["oprob"], T["oalias"], T["gm"]
    out = []
    for g, (P, sc, m0, m1) in enumerate(T["groups"]):
        ctr = chain_ctr0(T, g, b)               # the next draw's counter
        end = (m1 - m0) * B
        c0 = ctr

        def word():
            nonlocal ctr
            ctr += 1
            return mix((key + ctr * G) & M64)

        def emit(t):
            m = gm[m0 + t // B]
            o0, n = mo0[m], mo0[m + 1] - mo0[m]
            o = o0
            if n > 1:
                x = float((word() >> 11) + 1) * 2.0 ** -53 * n
                i = min(int(x), n - 1)
                o = o0 + i if x - i < opr[o0 + i] else o0 + oal[o0 + i]
            out.append((oalt[o], t % B))

        if sc > 0.0:
            for t in range(end):
                emit(t)
            if used is not None:
                used.append(ctr - c0)
            continue

        def gap():
            gp = math.log(float((word() >> 11) + 1) * 2.0 ** -53) * sc
            return int(gp) if gp < 4.0e18 else 1 << 62

        t = gap()
        while t < end:
            emit(t)
            t += 1 + gap()
        if used is not None:
            used.append(ctr - c0)
    return out


def shots(T, seed, c0, n):
    """The event lists of cumulative shots [c0, c0 + n): per shot ascending alternative id."""
    out = []
    cache = {}
    for c in range(c0, c0 + n):
        b = c // B
        if b not in cache:
            per = [[] for _ in range(B)]
            for a, s in block(T, seed, b):
                per[s].append(a)
            cache = {b: [sorted(x) for x in per]}
        out.append(cache[b][c % B])
    return out
