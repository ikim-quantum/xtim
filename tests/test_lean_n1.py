"""lean N1 (OWNER-SIGNED 2026-11-07, "Signing b"; ENGINE_VERSION xtim-engine-17): the twirl sampler's NOISE stream is
a mechanism table (level 0: null alternatives pruned, identical-record alternatives of a channel merged; no cross-channel
convolution) walked by ONE geometric-skip chain per (seed, aligned 4096-shot block, equal-P group) over
(mechanisms x shots), mechanism-major, the outcome an alias draw on the next word. Runs that read P_f itself (recording,
framed / term-recorded) draw the FULL channel table (option b: exact P_rec).

This file pins (1) the tables and the chain against a pure-Python reference of the contract (tests/_n1_reference.py) bit
for bit, and the engine's event lists against it; (2) the law independently of any reference: per-outcome rates, the
group's per-shot counts (Binomial), the shot position inside the block, pairwise independence of mechanisms, the
exactness of the pruning (the full table's record-class law == the pruned table's); (3) the equalities the contract keeps:
threads N == serial, any call split == one call (mid-block ends included), seed=None continuation, window on == off,
the shot-block route == the event list, a recording run == a full-table sample run on the detectors.
"""
import hashlib
import math
import os
import sys

import numpy as np
import pytest
from scipy.stats import chi2

from xtim import _xtim as X
from xtim import _xtim_testapi as TA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _n1_reference as R   # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CIRC = {
    "mech": os.path.join(DATA, "n1_mechanisms.stim"),          # nulls, merges, forced, p = 0.75, mixed p, M(p)
    "cult_d3": os.path.join(DATA, "cultivation_d3_rate.stim"),  # T circuit (rotation loop)
    "ch": os.path.join(DATA, "ch_cultivation.stim"),            # CH / PPR circuit
    "d5": os.path.join(DATA, "cultivation_d5.stim"),            # T circuit, many channels
}


def _fam(name, **kw):
    kw.setdefault("selfcheck", 0)
    return TA.compile_family_sampler(open(CIRC[name]).read(), **kw)


def _ref_tables(s):
    p, k = TA.noise_alts(s)
    rr = TA.rot_records(s)
    return R.tables(p, k, rr["rec"], rr["off"])


# ── (1) the reference pins ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch", "d5"])
@pytest.mark.parametrize("full", [False, True])
def test_tables_match_reference(name, full):
    s = _fam(name)
    T, ref = TA.noise_table(s, full), _ref_tables(s)[1 if full else 0]
    for key in ("tag", "mo0", "oalt", "ow", "oprob", "oalias", "mch", "mP", "gm"):
        assert list(np.atleast_1d(T[key])) == list(np.atleast_1d(ref[key])), key
    assert [tuple(g) for g in T["groups"]] == ref["groups"]
    for key in ("r0", "r", "R", "blast"):                      # lean N2: the chains' counter ranges
        assert T[key] == ref[key], key
    assert abs(T["events"] - ref["events"]) <= 1e-12 * max(1.0, ref["events"])
    assert T["tag"] == (R.NOISE_FULL if full else R.NOISE)


@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("seed,b", [(1, 0), (77, 3), (2**63 + 5, 1234567)])
def test_chain_matches_reference(name, full, seed, b):
    s = _fam(name)
    ref = _ref_tables(s)[1 if full else 0]
    got = list(TA.noise_block(s, full, seed, b))
    assert got == [(a << 32) | x for a, x in R.block(ref, seed, b)]
    assert len(got) > 0


@pytest.mark.parametrize("name", ["mech", "cult_d3"])
def test_event_lists_match_reference(name):
    """The engine's event lists (what run() draws after set_seed) == the reference's, shot-major over 2.2 blocks."""
    s = _fam(name)
    pr, _ = _ref_tables(s)
    assert not TA.noise_full(s)
    N = 9000
    assert [list(e) for e in s.term_events(N, 5)] == R.shots(pr, 5, 0, N)


# ── lean N2 (owner-signed 2026-11-07): every (seed, group, block) chain owns a disjoint counter range ─────────────
_GINV = pow(R.G, -1, 1 << 64)


@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch"])
@pytest.mark.parametrize("full", [False, True])
def test_chain_counter_ranges_are_disjoint(name, full):
    """The ENGINE's chain start positions, inverted to counters (pos = key + ctr0 * G, G odd), are b * R + r0_g; the
    ranges [ctr0, ctr0 + r_g) tile [b * R, (b + 1) * R) in group order and abut across blocks, so no two chains of one
    (seed, tag) can read the same word of the keyed sequence. A hashed per-chain base (the N1 layout) fails here."""
    s = _fam(name)
    T = TA.noise_table(s, full)
    Rb, r0, r, nG = T["R"], T["r0"], T["r"], len(T["groups"])
    assert nG > 0 and Rb == sum(r) and all(x == 2 * (g[3] - g[2]) * R.B + 1 for x, g in zip(r, T["groups"]))
    blocks = [0, 1, 2, 3, 1234567, T["blast"] - 1, T["blast"]]
    for seed in (1, 2**63 + 5):
        key = R.stream_key(seed, T["tag"])
        iv = []
        for b in blocks:
            for g in range(nG):
                ctr0 = ((TA.noise_chain_pos(s, full, seed, g, b) - key) * _GINV) % (1 << 64)
                assert ctr0 == b * Rb + r0[g], (seed, b, g)
                iv.append((ctr0, ctr0 + r[g]))
        iv.sort()
        assert all(a[1] <= c[0] for a, c in zip(iv, iv[1:]))                 # pairwise disjoint
        assert iv[-1][1] <= 1 << 64
        assert [x for x in iv if x[0] < 4 * Rb] == [(b * Rb + r0[g], b * Rb + r0[g] + r[g])
                                                    for b in range(4) for g in range(nG)]   # contiguous tiling
        assert iv[0][0] == 0 and all(a[1] == c[0] for a, c in zip(iv[:4 * nG], iv[1:4 * nG]))


@pytest.mark.parametrize("full", [False, True])
def test_chain_draws_fit_their_range(full):
    """A chain draws at most r_g = 2 * M_g * 4096 + 1 words (one gap per fired cell + 1, one outcome word per fired cell):
    counted on the reference (bit-for-bit the engine) over the mech circuit's groups, incl. the forced and P = 0.75 ones."""
    s = _fam("mech")
    ref = _ref_tables(s)[1 if full else 0]
    assert any(g[0] >= 0.75 for g in ref["groups"])
    for b in range(3):
        used = []
        R.block(ref, 9, b, used)
        assert len(used) == len(ref["groups"])
        assert all(0 <= u <= x for u, x in zip(used, ref["r"])), (used, ref["r"])


@pytest.mark.parametrize("full", [False, True])
def test_block_past_the_counter_range_is_refused(full):
    s = _fam("mech")
    T = TA.noise_table(s, full)
    assert (T["blast"] + 1) * T["R"] <= 1 << 64 < (T["blast"] + 2) * T["R"]
    assert len(TA.noise_block(s, full, 3, T["blast"])) > 0
    with pytest.raises(ValueError, match=r"2\^64 counters"):
        TA.noise_block(s, full, 3, T["blast"] + 1)


# the sampled bytes of a fixed call sequence (seeded, then continued across block ends), digests re-fixed at N2
GOLDEN = {
    "cult_d3": "f249987dbe10a90e",
    "mech": "5be9b3e05785e9de",
}


def _digest(s, calls=(1000, 3096, 5000, 1)):
    h = hashlib.sha256()
    for i, n in enumerate(calls):
        for a in s.sample(n, 11 if i == 0 else None):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_sample_bytes_golden(name):
    assert _digest(_fam(name)) == GOLDEN[name]


# ── (2) the law, independent of the reference ──────────────────────────────────────────────────────────────────────
def _counts(s, full, seeds, nb):
    """Per outcome slot: fired cells over `nb` blocks x seeds; per mechanism a (shot -> fired) indicator sample."""
    T = TA.noise_table(s, full)
    mo0, oalt = np.asarray(T["mo0"]), np.asarray(T["oalt"])
    slot_of = {int(a): i for i, a in enumerate(oalt)}
    cnt = np.zeros(len(oalt))
    pos = np.zeros(64)
    shots_per_mech = []
    for sd in seeds:
        for b in range(nb):
            ev = np.asarray(TA.noise_block(s, full, sd, b), dtype=np.uint64)
            alt = (ev >> np.uint64(32)).astype(np.int64)
            sh = (ev & np.uint64(0xFFFFFFFF)).astype(np.int64)
            for a in alt:
                cnt[slot_of[int(a)]] += 1
            np.add.at(pos, sh // 64, 1)
            shots_per_mech.append((alt, sh))
    return T, cnt, pos, shots_per_mech


@pytest.mark.parametrize("full", [False, True])
def test_law_outcome_rates(full):
    s = _fam("mech")
    seeds, nb = range(1, 9), 6                                  # 196,608 shots per mechanism
    T, cnt, pos, _ = _counts(s, full, seeds, nb)
    N = len(seeds) * nb * R.B
    w = np.asarray(T["ow"])
    forced = w >= 1.0
    z = (cnt - N * w) / np.sqrt(np.maximum(N * w * (1 - np.minimum(w, 1.0)), 1e-300))
    assert np.all(cnt[forced] == N)
    assert np.abs(z[~forced]).max() < 5.0, z
    # the shot position inside the block: events spread uniformly over the 64 sub-ranges of 64 shots
    st = ((pos - pos.mean()) ** 2 / pos.mean()).sum()
    assert chi2.sf(st, 63) > 1e-4, st


def test_law_group_counts_and_independence():
    """Per shot, a group's fired mechanisms ~ Binomial(M_g, P) (gaps exactly geometric over the grid), and two
    mechanisms of one group co-fire as independent events."""
    s = _fam("cult_d3")
    T = TA.noise_table(s, False)
    groups = T["groups"]
    mo0, oalt, gm = T["mo0"], T["oalt"], T["gm"]
    mech_of = {}
    for m in range(len(mo0) - 1):
        for o in range(mo0[m], mo0[m + 1]):
            mech_of[oalt[o]] = m
    g = max(range(len(groups)), key=lambda i: groups[i][3] - groups[i][2])   # the largest group
    P, _, m0, m1 = groups[g]
    Mg = m1 - m0
    members = set(gm[m0:m1])
    per_shot = []
    pair = np.zeros((2, 2))
    ma, mb = gm[m0], gm[m0 + 1]
    for sd in range(1, 13):
        ev = np.asarray(TA.noise_block(s, False, sd, 0), dtype=np.uint64)
        alt = (ev >> np.uint64(32)).astype(np.int64)
        sh = (ev & np.uint64(0xFFFFFFFF)).astype(np.int64)
        c = np.zeros(R.B, np.int64)
        fa, fb = np.zeros(R.B, bool), np.zeros(R.B, bool)
        for a, x in zip(alt, sh):
            m = mech_of[int(a)]
            if m in members:
                c[x] += 1
            if m == ma:
                fa[x] = True
            if m == mb:
                fb[x] = True
        per_shot.append(c)
        for i in (0, 1):
            for j in (0, 1):
                pair[i, j] += np.sum((fa == bool(i)) & (fb == bool(j)))
    c = np.concatenate(per_shot)
    kmax = max(3, int(c.max()))
    obs = np.array([np.sum(c == k) for k in range(kmax)] + [np.sum(c >= kmax)], float)
    from scipy.stats import binom
    exp = np.array([binom.pmf(k, Mg, P) for k in range(kmax)] + [binom.sf(kmax - 1, Mg, P)]) * len(c)
    keep = exp >= 5
    st = (((obs - exp) ** 2 / exp)[keep]).sum()
    assert chi2.sf(st, max(1, keep.sum() - 1)) > 1e-4, (obs, exp)
    n = pair.sum()
    pa, pb = pair[1].sum() / n, pair[:, 1].sum() / n
    z11 = (pair[1, 1] - n * pa * pb) / math.sqrt(max(n * pa * pb * (1 - pa * pb), 1e-300))
    assert abs(z11) < 5.0, pair


@pytest.mark.parametrize("name", ["mech", "cult_d3", "d5"])
def test_pruning_is_exact(name):
    """Independently of the reference: per channel, the full table's alternatives grouped by record give exactly the
    pruned outcomes' probabilities (the record XOR of a shot keeps its law), and the dropped mass is the null records'."""
    s = _fam(name)
    p, k = TA.noise_alts(s)
    rr = TA.rot_records(s)
    rec, off = rr["rec"], rr["off"]
    Tp, Tf = TA.noise_table(s, False), TA.noise_table(s, True)
    key = lambda a: bytes(np.asarray(rec[off[a]:off[a + 1]], dtype=np.uint32))
    law_p = {}
    for m in range(len(Tp["mo0"]) - 1):
        ch = Tp["mch"][m]
        for o in range(Tp["mo0"][m], Tp["mo0"][m + 1]):
            law_p[(ch, key(Tp["oalt"][o]))] = Tp["ow"][o]
    law_f = {}
    for m in range(len(Tf["mo0"]) - 1):
        ch = Tf["mch"][m]
        for o in range(Tf["mo0"][m], Tf["mo0"][m + 1]):
            a = Tf["oalt"][o]
            kk = key(a)
            if not any(np.asarray(rec[off[a]:off[a + 1]])):
                continue                                         # null: invisible in every record column
            law_f[(ch, kk)] = law_f.get((ch, kk), 0.0) + p[a]
    assert set(law_p) == set(law_f)
    for kk, v in law_f.items():
        assert abs(law_p[kk] - v) <= 1e-15 * max(1.0, v), kk
    assert len(Tf["mo0"]) - 1 == len({k[a] for a in range(len(p)) if p[a] > 0})


# ── (2b) review M2: a pruned alternative is invisible, by DIRECT COMPOSITION (not the engine's records, not a table) ──
# The engine's pruned table says which alternatives a sample run never draws: a dropped alternative a of channel c is
# either NULL (its record is all zero) or MERGED into an outcome representative r of c (identical records; the outcome's
# probability is the class's). Independently of the records and of the tables, a shot is the composition of its fired
# alternatives: the term route (composed term set S, kind, reduced key, plan id, t(P) over every storage column:
# detectors / observables / decisions and the extras <P, M_e>) and, when diagonal, the diagonal route's canonical normal
# form (plan id, a, cz, t(prefix)); a PPR shot is the term route's (CH / PPR content). For every dropped alternative,
# alone and on random backgrounds E of other channels: shot(E + [a]) == shot(E) (null) or == shot(E + [r]) (merged).
COMPOSE = {
    "mech": CIRC["mech"],
    "cult_d3": CIRC["cult_d3"],
    "d5": CIRC["d5"],
    "qrm": os.path.join(DATA, "adaptq_qrm_magic_producer.stim"),
    "ppr": os.path.join(DATA, "ja_seed23.stim"),                 # CH / PPR alternatives among the dropped ones
    "extras": os.path.join(DATA, "sign_live_bytes_c40022.stim"),   # faults whose only record effect is an extra
    "cs": os.path.join(DATA, "code_switching_faithful.stim"),      # extras + many merges
}


def _composed(s, alts, n):
    alts = sorted(alts)
    rel = lambda xz: np.asarray(s.term_relabel(list(np.asarray(xz[0], np.uint8)),
                                               list(np.asarray(xz[1], np.uint8)))).tobytes()
    t = s.term_shot(alts)
    out = (tuple(t["S"]), int(t["kind"]), bytes(t["key"]), int(t["plan_id"]), rel(t["P"]))
    d = s.diag_shot(alts)
    if d is None:
        n["term"] += 1
        return out
    n["diag"] += 1
    return out + (int(d["plan_id"]), tuple(int(v) for v in d["a"]), tuple(map(tuple, d["cz"])), rel(d["prefix"]))


@pytest.mark.parametrize("name", sorted(COMPOSE))
def test_dropped_alternatives_invisible_by_composition(name):
    import random
    s = TA.compile_family_sampler(open(COMPOSE[name]).read(), selfcheck=0)
    p, k = TA.noise_alts(s)
    rr = TA.rot_records(s)
    rec, off = np.asarray(rr["rec"]), rr["off"]
    rkey = lambda a: rec[off[a]:off[a + 1]].tobytes()
    Tp = TA.noise_table(s, False)
    reps = {}
    for m in range(len(Tp["mo0"]) - 1):
        for o in range(Tp["mo0"][m], Tp["mo0"][m + 1]):
            reps.setdefault(int(Tp["mch"][m]), []).append((int(Tp["oalt"][o]), float(Tp["ow"][o])))
    chans = {}
    for a in range(len(p)):
        if p[a] > 0:
            chans.setdefault(int(k[a]), []).append(a)
    dropping = sorted(c for c, v in chans.items() if len(v) > len(reps.get(c, [])))
    rng = random.Random(name)
    pick = dropping if len(dropping) <= 60 else sorted(rng.sample(dropping, 60))
    n = dict(diag=0, term=0, null=0, merged=0, checks=0)
    for c in pick:
        R = reps.get(c, [])
        rset = {r for r, _ in R}
        cls, mass = {}, {r: p[r] for r in rset}
        for a in chans[c]:
            if a in rset:
                continue
            same = [r for r in sorted(rset) if rkey(r) == rkey(a)]
            if same:
                cls[a] = same[0]
                mass[same[0]] += p[a]
                n["merged"] += 1
            else:
                assert not rec[off[a]:off[a + 1]].any(), (c, a, "dropped but neither null nor merged")
                cls[a] = None
                n["null"] += 1
        for r, w in R:
            assert abs(mass[r] - w) <= 1e-15 * max(1.0, w), (c, r, mass[r], w)
        other = [x for x in sorted(chans) if x != c]
        for rep in range(4):
            E = [] if rep == 0 else [rng.choice(chans[x]) for x in
                                     rng.sample(other, min(len(other), rng.choice([1, 2, 4])))]
            x0 = _composed(s, E, n)
            for a, r in cls.items():
                y = _composed(s, E + [a], n)
                assert y == (x0 if r is None else _composed(s, E + [r], n)), (c, a, r, E)
                n["checks"] += 1
    assert n["checks"] > 0 and n["null"] + n["merged"] > 0, n
    if name == "ppr":
        assert n["term"] > 0, n


# ── (3) the equalities the contract keeps ──────────────────────────────────────────────────────────────────────────
CALLS = (1, 4095, 4097, 1000, 3000)


def _seq(s, calls=CALLS, seed=5, threads=1, mask=None):
    out = []
    for i, n in enumerate(calls):
        sd = seed if i == 0 else None
        if mask is None:
            out += [np.asarray(a).copy() for a in s.sample(n, sd, threads=threads)]
        else:
            *arrs, c = s._sample_ps(n, sd, mask, threads)
            out += [np.asarray(a).copy() for a in arrs] + [sorted(dict(c).items())]
    return out


def _cat(seq, k):
    """Concatenate the call sequence's arrays of output k (0 = detectors, 1 = observables)."""
    n = len(seq) // len(CALLS)
    return np.concatenate([seq[i * n + k] for i in range(len(CALLS))])


def _eq(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            assert x.shape == y.shape and np.array_equal(x, y)
        else:
            assert x == y


@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch", "d5"])
def test_call_split_equals_one_call(name):
    a, b = _fam(name), _fam(name)
    sa = _seq(a)
    one = [np.asarray(x) for x in b.sample(sum(CALLS), 5)]
    for k in range(len(one)):
        assert np.array_equal(_cat(sa, k), one[k])


@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch", "d5"])
@pytest.mark.parametrize("kind", ["none", "all"])
def test_threads_equal_serial(name, kind):
    a, b = _fam(name), _fam(name)
    mask = None if kind == "none" else np.ones(a.num_detectors, bool)
    _eq(_seq(a, threads=4, mask=mask), _seq(b, threads=1, mask=mask))


@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch"])
def test_window_on_equals_off(name):
    """The window's shots read the run's table (the pruned one on a sample run), so window on == off."""
    a, b, c = _fam(name), _fam(name, selfcheck=2500), _fam(name, oracle_all_shots=True)
    ra = _seq(a)
    _eq(_seq(b), ra)
    _eq(_seq(c), ra)


@pytest.mark.parametrize("name", ["mech", "cult_d3", "ch"])
def test_shot_block_equals_event_list(name):
    a, b = _fam(name), _fam(name)
    TA.shot_block(a, False)
    _eq(_seq(a), _seq(b))
    assert TA.shot_block_shots(a) == 0 and TA.shot_block_shots(b) == sum(CALLS)


@pytest.mark.parametrize("name", ["mech", "cult_d3"])
def test_recording_run_reads_the_full_table(name):
    """Option (b): a recording run draws the FULL table; its detectors == a sample run forced onto the full table (same
    seed), and differ from the default (pruned) sample run's bytes in general — the stated relation."""
    s = _fam(name)
    full = _fam(name)
    TA.noise_force_full(full, True)
    assert not TA.noise_full(s) and TA.noise_full(full)
    n = 6000
    d_rec = s.sample_records(n, seed=3)[0]
    d_full = full.sample(n, 3)[0]
    d_pruned = _fam(name).sample(n, 3)[0]
    assert np.array_equal(np.asarray(d_rec), np.asarray(d_full))
    assert not np.array_equal(np.asarray(d_rec), np.asarray(d_pruned))


@pytest.mark.parametrize("name", ["mech", "cult_d3"])
def test_recording_threads_and_split(name):
    """A recording run (sample_barrier: the full table): threads 4 == serial, a split call == one call."""
    text = open(CIRC[name]).read()
    a, b, c = (X.TwirlSampler(text, selfcheck=0) for _ in range(3))
    ra = a.sample_barrier(9000, 4)
    rb = b.sample_barrier(9000, 4, threads=4)
    r1, r2 = c.sample_barrier(5000, 4), c.sample_barrier(4000)
    for f in ("dets", "obs"):
        x, y = np.asarray(getattr(ra, f)()), np.asarray(getattr(rb, f)())
        assert np.array_equal(x, y)
        assert np.array_equal(x, np.concatenate([np.asarray(getattr(r1, f)()), np.asarray(getattr(r2, f)())]))
    assert np.array_equal(np.asarray(ra.dets()), np.asarray(_fam(name).sample_records(9000, seed=4)[0]))


def test_full_table_law_equals_pruned():
    """The two tables give the same detector law (pruned sample run vs a full-table run, disjoint seeds)."""
    a, b = _fam("mech"), _fam("mech")
    TA.noise_force_full(b, True)
    n = 200_000
    da = np.unpackbits(np.asarray(a.sample(n, 1)[0]), axis=1, bitorder="little")[:, :a.num_detectors].astype(float)
    db = np.unpackbits(np.asarray(b.sample(n, 2)[0]), axis=1, bitorder="little")[:, :b.num_detectors].astype(float)
    pa, pb = da.mean(0), db.mean(0)
    p = (pa + pb) / 2
    z = (pa - pb) / np.sqrt(np.maximum(p * (1 - p) * 2 / n, 1e-300))
    live = (p > 0) & (p < 1)
    assert np.abs(z[live]).max() < 5.0, z
    ca, cb = da.T @ da / n, db.T @ db / n
    iu = np.triu_indices(a.num_detectors, 1)
    pc = (ca[iu] + cb[iu]) / 2
    zc = (ca[iu] - cb[iu]) / np.sqrt(np.maximum(pc * (1 - pc) * 2 / n, 1e-300))
    lv = (pc > 0) & (pc < 1)
    assert np.abs(zc[lv]).max() < 5.0, zc
