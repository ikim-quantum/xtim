"""lean L5 (2026-11-05, owner-signed stream change): the readout flips (Stim M(p)) are skip-sampled. Slot sl fired iff the
shot's RF word sl was below floor(p * 2^64) — one keyed hash per noisy record slot per shot (2,635 slots = 2.9 us/shot on
the full d5 cultivation circuit at any p). Now the slots of equal p form a group and each group is one chain of
inverse-CDF geometric gaps over its slots (the gate-noise layer's method over shots): the cost is the fired flips. The law
is unchanged: independent Bernoulli(p_slot) per slot, independent of every other stream; the RF stream stays keyed per
(seed, cumulative shot) and sequential, so the bytes stay route-, lane- and call-split-independent (D10).

This file pins the new stream against a pure-Python reference of the design (lean/L5/plan.md), tests the law (rates,
Poisson-binomial flips per shot, the geometric gap, pairwise independence, independence of the gate-noise bits), and the
equalities the window / oracle_all_shots / lanes / call split must keep on noisy-M circuits.
Test-of-test (lean/L5/REPORT.md): `mut_rf_bias` (the gap scale off by 10 %) is red on the pin and the law tests;
`mut_rf_lane` (a lane's chain positioned one shot off) is red on the lanes test only.
"""
import hashlib
import math
import os

import numpy as np
import pytest
from scipy.stats import chi2

import xtim
from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D5 = os.path.join(HERE, "..", "examples", "cultivation_d5.stim")   # the real T circuit: 13 noisy slots at p = 1e-3

M64 = (1 << 64) - 1
G = 0x9E3779B97F4A7C15
RF = 6


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def rf_groups(probs):
    """The design's grouping: slots of EQUAL p, groups in order of their first slot, slots ascending inside a group."""
    groups = {}
    order = []
    for sl, p in enumerate(probs):
        if p not in groups:
            groups[p] = []
            order.append(p)
        groups[p].append(sl)
    return [(p, groups[p]) for p in order]


def rf_reference(seed, shot, groups):
    """The fired slots of one shot: per group one chain of geometric gaps on the shot's sequential RF words."""
    base = (mix(seed ^ mix(RF)) + (shot << 20) * G) & M64
    ctr = 0
    fired = []
    for p, slots in groups:
        k = len(slots)
        if p >= 1.0:
            fired += slots                                   # fires always, no draw
            continue
        il = 1.0 / math.log1p(-p)
        pos = 0
        while pos < k:
            ctr += 1
            u = ((mix((base + ctr * G) & M64) >> 11) + 1) * 2.0 ** -53
            gp = math.log(u) * il
            if not (gp < float(k - pos)):
                break
            pos += int(gp)
            fired.append(slots[pos])
            pos += 1
    return fired


def _bits(a, k):
    return np.unpackbits(np.asarray(a, np.uint8), axis=1, bitorder="little")[:, :k]


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        if not isinstance(a, dict):
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


# ── fixtures ───────────────────────────────────────────────────────────────────────────────────────────────────────────
# A flip-only circuit: detector i = the i-th noisy measurement (slot i), no gate noise.
PIN_PS = [1e-3, 0.02, 0.3, 1.0, 0.3, 1e-3, 0.5, 2e-3, 0.02, 1.0, 0.3, 1e-9, 0.3, 0.999, 1e-3, 0.02] * 3
PIN_TXT = "".join(f"R 0\nM({p!r}) 0\nDETECTOR rec[-1]\n" for p in PIN_PS)

# The law fixture: 100 slots each at p = 0.3 / 0.02 / 1e-3 interleaved with 50 gate-noise-only detectors (X_ERROR(0.1) + M).
LAW_PS = [0.3, 0.02, 1e-3] * 100
LAW_IS_FLIP = []
LAW_TXT = ""
for _i, _p in enumerate(LAW_PS):
    LAW_TXT += f"R 0\nM({_p!r}) 0\nDETECTOR rec[-1]\n"
    LAW_IS_FLIP.append(True)
    if _i % 6 == 5:
        LAW_TXT += "R 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\n"
        LAW_IS_FLIP.append(False)
LAW_FLIP = np.array(LAW_IS_FLIP)
LAW_P = np.array(LAW_PS)                      # the flip-only slots in circuit order
# the expected rate of every LAW detector in circuit order: flip-only = its p, gate-noise = 0.1
_it = iter(LAW_PS)
LAW_EXP = np.array([next(_it) if f else 0.1 for f in LAW_IS_FLIP])

# A mixed-p circuit WITH gate noise and a Born observable (the window / lanes / split equalities)
MIXED_TXT = ("RX 0\nR 1 2 3\nT 0\nDEPOLARIZE1(0.05) 0 1 2 3\nCX 0 1\nCZ 1 2\nCX 2 3\nDEPOLARIZE2(0.02) 0 1 2 3\n"
             "M(0.1) 1\nDETECTOR rec[-1]\nM(0.001) 2\nDETECTOR rec[-1]\nMX(0.3) 3\nDETECTOR rec[-1]\n"
             "MX(0.02) 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
             + "".join(f"R {q}\nX_ERROR(0.05) {q}\nM({p!r}) {q}\nDETECTOR rec[-1]\n" for q, p in
                       [(4, 0.1), (5, 0.001), (6, 0.3), (7, 0.02), (8, 0.001), (9, 0.5)]))


def _sample_dets(text, n, seed, ndet, **kw):
    s = xtim.compile_twirl_sampler(text, selfcheck=0, **kw)
    return _bits(s.sample(n, seed=seed)[0], ndet), s


# ── 1. the stream pin (RED on L4: the per-slot threshold stream) ───────────────────────────────────────────────────────
@pytest.mark.parametrize("p_factor", [1.0, 0.5])
def test_rf_stream_is_the_grouped_geometric_chain(p_factor):
    """Every shot's flips are the pure-Python reference of the design (the shot's sequential RF words, groups of equal p
    in first-slot order, one inverse-CDF geometric chain per group; p >= 1 fires without a draw), seeded and continued."""
    probs = [p * p_factor for p in PIN_PS]
    groups = rf_groups(probs)
    assert len(groups) == 8
    s = xtim.compile_twirl_sampler(PIN_TXT, p_factor=p_factor, selfcheck=0)
    seed, N = 4242, 6000
    d = _bits(s.sample(N, seed=seed)[0], len(PIN_PS))
    d2 = _bits(s.sample(N, seed=None)[0], len(PIN_PS))                 # continued: cumulative shot index
    for i in range(N):
        want = np.zeros(len(PIN_PS), np.uint8)
        want[rf_reference(seed, i, groups)] = 1
        assert np.array_equal(d[i], want), i
        want2 = np.zeros(len(PIN_PS), np.uint8)
        want2[rf_reference(seed, N + i, groups)] = 1
        assert np.array_equal(d2[i], want2), N + i
    # p = 1 fires always without a draw (p_factor 1), p = 1e-9 (scaled) never in 12k shots
    assert not d[:, 11].any()
    if p_factor == 1.0:
        assert d[:, 3].all() and d[:, 9].all() and d[:, 13].mean() > 0.99
    else:
        assert 0.47 < d[:, 3].mean() < 0.53


# ── 2. the law ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _law_sample(seed=5, N=60000):
    d, _ = _sample_dets(LAW_TXT, N, seed, len(LAW_IS_FLIP))
    return d.astype(np.float64)


def test_rf_marginal_rates():
    """Per-slot flip rates = p: max |z| over 300 slots (+ 50 gate-noise detectors at 0.1) and the aggregate z per group."""
    X = _law_sample()
    N = X.shape[0]
    m = X.mean(axis=0)
    z = (m - LAW_EXP) / np.sqrt(LAW_EXP * (1 - LAW_EXP) / N)
    assert np.abs(z).max() < 5.0, (np.abs(z).max(), int(np.abs(z).argmax()))
    for p in (0.3, 0.02, 1e-3, 0.1):
        sel = LAW_EXP == p
        tot = X[:, sel].sum()
        exp = N * sel.sum() * p
        zg = (tot - exp) / math.sqrt(exp * (1 - p))
        assert abs(zg) < 4.0, (p, zg)


def _pb_pmf(ps):
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf


def _chi2_pvalue(obs, exp):
    """Chi-square with bins merged until every expected count is >= 5 (bins merged from the right)."""
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    o, e = [], []
    ao = ae = 0.0
    for x, y in zip(obs, exp):
        ao += x; ae += y
        if ae >= 5:
            o.append(ao); e.append(ae); ao = ae = 0.0
    if ae > 0:
        if e:
            o[-1] += ao; e[-1] += ae
        else:
            o.append(ao); e.append(ae)
    o, e = np.array(o), np.array(e)
    stat = ((o - e) ** 2 / e).sum()
    return chi2.sf(stat, max(len(o) - 1, 1)), stat, len(o)


def test_rf_flips_per_shot_is_poisson_binomial():
    """The number of fired flip-only slots per shot follows the Poisson-binomial law of the 300 independent slots."""
    X = _law_sample()
    k = X[:, LAW_FLIP].sum(axis=1).astype(int)
    pmf = _pb_pmf(LAW_P)
    obs = np.bincount(k, minlength=len(pmf))[: len(pmf)]
    pv, stat, nb = _chi2_pvalue(obs, pmf * X.shape[0])
    assert pv > 1e-4, (pv, stat, nb)


def test_rf_gap_law_is_geometric():
    """Within a shot, the position of the FIRST fired slot of the p = 0.3 group (100 slots) is geometric: P(g) = p(1-p)^g,
    P(none) = (1-p)^100; and after a fired slot at position i <= 59 the gap to the next fired slot is geometric again
    (memoryless chain): P(gap = g) = p(1-p)^g for g < 40, P(gap >= 40 or none) = (1-p)^40."""
    X = _law_sample()
    p = 0.3
    cols = np.where(LAW_EXP == p)[0]
    B = X[:, cols].astype(bool)
    k = len(cols)
    first = np.where(B.any(axis=1), B.argmax(axis=1), k)
    pmf = np.array([p * (1 - p) ** g for g in range(k)] + [(1 - p) ** k])
    obs = np.bincount(first, minlength=k + 1)
    pv, stat, nb = _chi2_pvalue(obs, pmf * X.shape[0])
    assert pv > 1e-4, ("first", pv, stat, nb)
    gmax = 40
    cnt = np.zeros(gmax + 1, int)
    for row in B:
        idx = np.flatnonzero(row)
        for j, i in enumerate(idx):
            if i > k - 1 - gmax:
                continue
            g = (idx[j + 1] - i - 1) if j + 1 < len(idx) else gmax
            cnt[min(g, gmax)] += 1
    pmf = np.array([p * (1 - p) ** g for g in range(gmax)] + [(1 - p) ** gmax])
    pv, stat, nb = _chi2_pvalue(cnt, pmf * cnt.sum())
    assert pv > 1e-4, ("gaps", pv, stat, nb)


def test_rf_pairwise_independence_and_gate_noise_independence():
    """Every pair of detectors (flip-only x flip-only, flip-only x gate-noise) co-occurs at rate p_i p_j: a z-score per
    pair whose expected co-occurrence count is >= 30 (max |z| over those pairs; the expected max of that many null
    z-scores is ~4.3) and, for the sparse pairs (expected < 30: the 1e-3 x 1e-3 / 1e-3 x 0.02 / 0.02 x 0.02 / 1e-3 x 0.1
    classes), an exact two-sided Poisson tail per pair, Bonferroni-corrected over the sparse pairs."""
    from scipy.stats import poisson
    X = _law_sample()
    N, m = X.shape
    C = X.T @ X
    pp = np.outer(LAW_EXP, LAW_EXP)
    E = N * pp
    iu = np.triu_indices(m, 1)
    Cp, Ep, ppp = C[iu], E[iu], pp[iu]
    dense = Ep >= 30
    z = (Cp[dense] - Ep[dense]) / np.sqrt(Ep[dense] * (1 - ppp[dense]))
    assert dense.sum() > 20000
    assert np.abs(z).max() < 5.5, (np.abs(z).max(), int(np.abs(z).argmax()))
    sp = ~dense
    up = poisson.sf(Cp[sp] - 1, Ep[sp])                 # P(X >= C)
    lo = poisson.cdf(Cp[sp], Ep[sp])                    # P(X <= C)
    pv = np.minimum(up, lo) * 2 * sp.sum()               # two-sided, Bonferroni over the sparse pairs
    assert pv.min() > 1e-4, (pv.min(), int(pv.argmin()), Cp[sp][pv.argmin()], Ep[sp][pv.argmin()])
    # the flip x gate-noise block specifically (its dense pairs: 0.3 x 0.1 and 0.02 x 0.1)
    f, g = np.where(LAW_FLIP)[0], np.where(~LAW_FLIP)[0]
    blk = (E[np.ix_(f, g)] >= 30)
    zb = ((C - E) / np.sqrt(E * (1 - pp)))[np.ix_(f, g)][blk]
    assert len(zb) > 5000 and np.abs(zb).max() < 5.0, np.abs(zb).max()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rf_rates_across_seeds_and_shots_are_independent(seed):
    """Across shots a slot's flips are i.i.d.: the lag-1 autocorrelation of every p = 0.3 slot over 60k shots is ~0."""
    X = _law_sample(seed=seed, N=60000)
    cols = np.where(LAW_EXP == 0.3)[0]
    B = X[:, cols]
    a, b = B[:-1], B[1:]
    N = a.shape[0]
    c11 = (a * b).sum(axis=0)
    z = (c11 - N * 0.09) / math.sqrt(N * 0.09 * 0.91)
    assert np.abs(z).max() < 5.0, np.abs(z).max()


# ── 3. equalities the stream must keep on noisy-M circuits ─────────────────────────────────────────────────────────────
FIX = {"d5_T": lambda: open(D5).read(), "mixed": lambda: MIXED_TXT, "law": lambda: LAW_TXT}


@pytest.mark.parametrize("name", ["d5_T", "mixed"])
def test_rf_threads_equal_serial(name):
    """Lanes (threads = 4) draw each shot's chain at the shot's cumulative index: the same bytes as the serial run."""
    text = FIX[name]()
    a = xtim.compile_twirl_sampler(text, selfcheck=0).sample(12000, seed=2, threads=4)
    b = xtim.compile_twirl_sampler(text, selfcheck=0).sample(12000, seed=2)
    assert _h(a) == _h(b)


@pytest.mark.parametrize("name", ["d5_T", "mixed", "law"])
def test_rf_window_and_oracle_all_shots_equal_streaming(name):
    """The selfcheck window and oracle_all_shots read the flips draw_rf produced (ONE RF reader per shot): the same
    bytes as selfcheck = 0, seeded and continued."""
    text = FIX[name]()
    a = TA.compile_family_sampler(text, selfcheck=0)
    b = TA.compile_family_sampler(text, selfcheck=200)
    c = TA.compile_family_sampler(text, selfcheck=0, oracle_all_shots=True)
    assert _h(a.sample(3000, 9)) == _h(b.sample(3000, 9)) == _h(c.sample(3000, 9))
    assert _h(a.sample(3000, None)) == _h(b.sample(3000, None)) == _h(c.sample(3000, None))
    pa = xtim.compile_twirl_sampler(text, selfcheck=0)
    pb = xtim.compile_twirl_sampler(text, selfcheck=400)
    assert _h(pa.sample(4000, seed=3)) == _h(pb.sample(4000, seed=3))


@pytest.mark.parametrize("name", ["mixed", "law"])
def test_rf_any_split_equals_single_call(name):
    text = FIX[name]()
    N = 5000
    one = xtim.compile_twirl_sampler(text, selfcheck=0).sample(N, seed=11)[0]
    for sizes in ([1, 4999], [1000, 7, 3993], [4096, 904]):
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
        parts = [s.sample(sizes[0], seed=11)[0]] + [s.sample(k)[0] for k in sizes[1:]]
        assert np.array_equal(np.asarray(one), np.concatenate([np.asarray(p) for p in parts])), sizes


# ── L5 review F1: an invalid p_factor is refused by name at every entry point ────────────────────────────────────────
# A negative p_factor made a readout slot's scaled p negative: log1p(-p) > 0, the geometric gap negative, and draw_rf
# wrote before its group (SIGSEGV on 5c86b0c); NaN gave all-zeros flips. Every sampler entry point reaches the engine
# through the one family-sampler ctor, which refuses a non-finite or negative p_factor before any compile work. Run in
# a subprocess: the unfixed engine crashes.
_F1_SCRIPT = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
import xtim
from xtim import _xtim, _xtim_testapi as TA
pf = float(sys.argv[1])
NOISY = "R 0\nX_ERROR(0.1) 0\nM(0.1) 0\nDETECTOR rec[-1]\n"
PORT = "INPUT_QUBITS in 0\nR 1\nM(0.1) 1\nDETECTOR rec[-1]\n"
IF_TEXT = ("INPUT_BITS drv 1\nRX 0 1\nR 2\nT 0\nIF drv[0] {\n  Z 0\n}\nMX 0\nCZ rec[-1] 1\nMX 1\nM(0.1) 2\n"
           "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")
PRODUCER = "R 0\nH 0\nOUTPUT_QUBITS out 0\n"
ideal = lambda: xtim.input_state_of(PRODUCER)
entries = {
    "compile_twirl_sampler": lambda: xtim.compile_twirl_sampler(NOISY, p_factor=pf, selfcheck=0),
    "TwirlSampler": lambda: _xtim.TwirlSampler(NOISY, pf, 0, True),
    "from_ideal_state": lambda: xtim.compile_twirl_sampler_from_ideal_state(PORT, ideal_state=ideal(), p_factor=pf,
                                                                            selfcheck=0),
    "partitioned_IF": lambda: xtim.compile_twirl_sampler(IF_TEXT, p_factor=pf, selfcheck=0),
    "compile_family_sampler": lambda: TA.compile_family_sampler(NOISY, p_factor=pf, selfcheck=0),
    "compile_family_sampler_from_ideal_state": lambda: TA.compile_family_sampler_from_ideal_state(
        TA.compile_family_sampler(PRODUCER, selfcheck=0).ideal_state(), PORT, p_factor=pf),
}
out = {}
for name, make in entries.items():
    try:
        s = make()
    except ValueError as e:
        out[name] = "refused: " + str(e)
        continue
    if name == "partitioned_IF":
        s.sample(2000, seed=1, decision_bits=np.zeros((2000, 1), np.uint8))
    else:
        s.sample(2000, seed=1)
    out[name] = "sampled"
print(json.dumps(out))
"""


def _f1_run(pf):
    import json
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _F1_SCRIPT, repr(pf)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"p_factor={pf}: rc {r.returncode}\n{r.stderr[-2000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("pf", [-1.0, -1e-6, float("nan"), float("inf"), float("-inf")])
def test_invalid_p_factor_is_refused_by_name_at_every_entry(pf):
    out = _f1_run(pf)
    assert len(out) == 6
    for name, v in out.items():
        assert v.startswith("refused: p_factor must be finite and >= 0"), (name, v)


def test_valid_p_factor_boundary_is_served():
    """p_factor = 0 (noiseless) and an ordinary factor stay served at every entry point."""
    for pf in (0.0, 0.5):
        assert set(_f1_run(pf).values()) == {"sampled"}, pf
