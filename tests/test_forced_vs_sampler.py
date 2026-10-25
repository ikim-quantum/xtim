"""Forced faults vs the EXISTING exact path: the same faults written into the circuit as
deterministic Pauli gates (every other noise channel and readout coin removed; a forced record
flip as M(1)), recompiled and sampled by the exact sampler. Every sampled (detectors,
observables) outcome must be a row of the forced law's expanded view (up to ONE constant
reference XOR: the written-in circuit's reference contains the fault), the empirical row
frequencies must match the exact probabilities (two-sided binomial, Bonferroni), every row of
probability >= 20/N must be observed, and the sampled per-shot PAULI_EXPECTATION values averaged
within a row must match the law's conditional expectation (6 sigma). Plus Clifford circuits vs
Stim (detectors vs the unfaulted reference)."""
import math
import random
import re
import zlib

import numpy as np
import pytest

import xtim
from _orig_examples import ORIG_EXAMPLES, load_orig  # noqa: E402,F401
from xtim import _xtim
from xtim._exact_scope import explicit_exact  # noqa: E402

NOISE = re.compile(r"^(X_ERROR|Y_ERROR|Z_ERROR|DEPOLARIZE1|DEPOLARIZE2|PAULI_CHANNEL_1|"
                   r"PAULI_CHANNEL_2)\(([^)]*)\)\s+(.*)$")
MEAS = re.compile(r"^(M|MX|MY|MZ|MR|MRX|MRY|MRZ)(\([^)]*\))?\s+(.*)$")


def _log_binom_pmf(k, n, p):
    """log P(X = k) for X ~ Binomial(n, p), entirely in log-space (math.lgamma for the binomial
    coefficient). Deliberately NOT math.comb(n, k): that is an exact bigint, and for n in the
    thousands summing O(n) of them (the tail sums below) does O(n) bigint multiplications of
    ~n-digit integers each -- a real hang, not a style nit (measured: >10 min and climbing on this
    file's n=3000-4000 shots before this fix landed). lgamma is a single O(1) float op per term,
    accurate to ~1e-15 relative, which easily survives the O(sqrt(n)) summation error growth at
    the n ~ 1e4 scale this file uses (see test_binom_cdf_sf_match_scipy for the measured bound)."""
    if p <= 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if k == n else float("-inf")
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_comb + k * math.log(p) + (n - k) * math.log1p(-p)


def _logsumexp(logs):
    m = max(logs, default=float("-inf"))
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(math.fsum(math.exp(x - m) for x in logs))


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p). No scipy dependency: entirely log-space (lgamma-based
    log-pmf, logsumexp'd), accurate to the tolerance proved in test_binom_cdf_sf_match_scipy."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.exp(_logsumexp([_log_binom_pmf(j, n, p) for j in range(k + 1)]))


def binom_sf(k, n, p):
    """P(X > k) for X ~ Binomial(n, p) (survival function), same no-scipy construction."""
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return math.exp(_logsumexp([_log_binom_pmf(j, n, p) for j in range(k + 1, n + 1)]))


def written_in(text, faults, flips):
    """faults: {(site, target): pauli}; flips: set of records."""
    out, site, rec = [], 0, 0
    for raw in text.splitlines():
        line = raw.strip()
        m = NOISE.match(line)
        if m:
            two = m.group(1) in ("DEPOLARIZE2", "PAULI_CHANNEL_2")
            qs = m.group(3).split()
            groups = [qs[i:i + 2] for i in range(0, len(qs), 2)] if two else [[q] for q in qs]
            for t, g in enumerate(groups):
                p = faults.get((site, t))
                if p:
                    out += [f"{c} {q}" for q, c in zip(g, p) if c != "I"]
            site += 1
            continue
        m = MEAS.match(line)
        if m:
            for tok in m.group(3).split():
                out.append(f"{m.group(1)}{'(1)' if rec in flips else ''} {tok}")
                rec += 1
            continue
        if line.startswith("MPP"):
            rec += len(line.split()) - 1
        out.append(raw)
    return "\n".join(out) + "\n"


def _marginal(law):
    """The expanded law marginalized onto (detectors, observables): {key: (p, cond. exps)}."""
    out = {}
    for i in range(len(law.probabilities)):
        k = tuple(law.detectors[i].astype(int)) + ("|",) + tuple(law.observables[i].astype(int))
        a = out.setdefault(k, [0.0, np.zeros(law.expectations.shape[1])])
        p = float(law.probabilities[i])
        a[0] += p
        a[1] = a[1] + p * law.expectations[i]
    return {k: (p, v / p if p > 0 else v) for k, (p, v) in out.items()}


def check(text, faults, flips, shots=4000, seed=1):
    sim = xtim.Circuit(text).compile_exact_branches()
    q = [(s, t, p) for (s, t), p in faults.items()] + [{"record": j} for j in flips]
    law = _marginal(sim.exact_branches(q))
    keys = list(law)
    with explicit_exact():                    # J-b1 F: the written-in circuit on the exact engine is the oracle
        prog = _xtim.CompiledProgram(written_in(text, faults, flips))
        r = prog.sample(shots, seed)
    assert r["ok"] and not r["rejected"], r
    D, O = r["num_detectors"], r["num_observables"]
    det = np.unpackbits(r["detectors"], axis=1, bitorder="little")[:, :D].astype(int)
    obs = np.unpackbits(r["observables"], axis=1, bitorder="little")[:, :O].astype(int)
    exps = r["expectations"]
    samp = np.concatenate([det, obs], axis=1)
    # the constant reference XOR: a shift mapping EVERY sampled outcome onto a law row
    rows = {tuple(int(x) for x in k if x != "|") for k in keys}
    cands = {tuple(np.array(k) ^ samp[0]) for k in rows}
    uniq = {tuple(x) for x in samp}
    cands = [c for c in cands if all(tuple(np.array(u) ^ np.array(c)) in rows for u in uniq)]
    assert cands, "a sampled outcome is not a row of the forced law (for any constant reference)"
    best = None
    for c in cands:                       # a shift under which the frequencies also match
        shifted = samp ^ np.array(c)
        idx = {tuple(int(x) for x in k if x != "|"): i for i, k in enumerate(keys)}
        cnt = np.zeros(len(keys), int)
        sums = np.zeros((len(keys), exps.shape[1]))
        for s_ in range(shots):
            i = idx[tuple(shifted[s_])]
            cnt[i] += 1
            sums[i] += exps[s_]
        ok = True
        for i, k in enumerate(keys):
            p = law[k][0]
            lo, hi = binom_cdf(cnt[i], shots, p), binom_sf(cnt[i] - 1, shots, p)
            ok &= min(lo, hi) * 2 * len(keys) > 1e-7 and (p < 20 / shots or cnt[i] > 0)
            if cnt[i]:
                ok &= bool(np.all(np.abs(sums[i] / cnt[i] - law[k][1]) <= 6 / np.sqrt(cnt[i]) + 1e-9))
        if ok:
            best = c
            break
    assert best is not None, "frequencies / conditional expectations do not match the forced law"
    return len(keys)


CASES = [("cultivation_d3_faithful", True), ("ch_cultivation", True),
         ("code_switching_faithful", True), ("miniature_oracle", True),
         ("distillation_15_1_3", False), ("cultivation_d5", True)]


@pytest.mark.parametrize("name,flips_ok", CASES)
def test_forced_vs_written_in(name, flips_ok):
    text = load_orig(name).text
    sim = xtim.Circuit(text).compile_exact_branches()
    sites = sim.noise_sites()
    nrec = xtim.Circuit(text).num_measurements
    rng = random.Random(zlib.crc32(name.encode()) & 0xFFFF)   # stable across PYTHONHASHSEED
    total = 0
    for trial in range(12):
        faults, flips = {}, set()
        for _ in range(rng.randint(0, 3)):
            s = rng.randrange(len(sites))
            two = sites[s]["channel"] in ("DEPOLARIZE2", "PAULI_CHANNEL_2")
            ntg = len(sites[s]["qubits"]) // (2 if two else 1)
            t = rng.randrange(ntg)
            faults[(s, t)] = (rng.choice([a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"])
                              if two else rng.choice("XYZ"))
        if flips_ok and nrec and rng.random() < 0.5:
            flips.add(rng.randrange(nrec))
        total += check(text, faults, flips, shots=3000, seed=trial + 1)
    assert total > 0


def test_clifford_vs_stim():
    """Clifford memories under Pauli faults: the law is ONE row (every detector deterministic,
    no coin, however many reads are random), and every stim sample of the written-in circuit
    has exactly that row's detectors relative to the UNFAULTED circuit's reference sample."""
    stim = pytest.importorskip("stim")
    for seed in range(40):
        rng = random.Random(seed)
        task = rng.choice(["surface_code:rotated_memory_z", "surface_code:rotated_memory_x",
                           "repetition_code:memory", "color_code:memory_xyz"])
        c = stim.Circuit.generated(task, rounds=2, distance=3, after_clifford_depolarization=0.01,
                                   before_measure_flip_probability=0.01,
                                   after_reset_flip_probability=0.01)
        text = str(c.flattened())
        sim = xtim.Circuit(text).compile_exact_branches()
        sites = sim.noise_sites()
        faults = {}
        for _ in range(rng.randint(1, 4)):
            s = rng.randrange(len(sites))
            two = sites[s]["channel"] in ("DEPOLARIZE2", "PAULI_CHANNEL_2")
            ntg = len(sites[s]["qubits"]) // (2 if two else 1)
            faults[(s, rng.randrange(ntg))] = rng.choice(
                ["XZ", "ZI", "YY", "IX"] if two else ["X", "Z", "Y"])
        law = sim.exact_branches([(s, t, p) for (s, t), p in faults.items()])
        # Pauli faults on a memory: every detector stays deterministic -> one row, no coin
        assert law.num_rows == 1 and law.num_uniform == 0 and len(law) == 1
        assert abs(law.probabilities[0] - 1) < 1e-15
        ref = np.array(c.reference_sample(), int)
        wc = stim.Circuit(written_in(text, faults, set()))
        meas = wc.compile_sampler(seed=seed).sample(3000).astype(int)
        det_lists = []
        nrec = 0
        for inst in c.flattened():
            if inst.name == "DETECTOR":
                det_lists.append([nrec + t.value for t in inst.targets_copy()])
            elif stim.gate_data(inst.name).produces_measurements:
                nrec += len(inst.targets_copy())
        want = law.detectors[0].astype(int).tolist()
        for m in meas:   # detectors: parity of the sample XOR the UNFAULTED reference parity
            assert [(m[d].sum() + ref[d].sum()) % 2 for d in det_lists] == want


def test_binom_cdf_sf_match_scipy():
    """Prove binom_cdf/binom_sf (lgamma-based log-space pmf, no scipy) equal scipy.stats.binom on
    a grid, including the edge cases p=0 and p=1. scipy is optional: this proof-of-equality test
    is the only place in this file that touches it; every other test in this module is
    scipy-free."""
    sp_stats = pytest.importorskip("scipy.stats")
    ns = [1, 2, 5, 37, 400, 3000, 4000]
    ps = [0.0, 1e-6, 0.001, 0.02, 0.2, 0.5, 0.8, 0.98, 1 - 1e-6, 1.0]
    for n in ns:
        for p in ps:
            for k in (0, 1, n // 3, n // 2, n - 1, n):
                if k < 0 or k > n:
                    continue
                for kk in (k, k - 1):
                    want_cdf = sp_stats.binom.cdf(kk, n, p)
                    want_sf = sp_stats.binom.sf(kk, n, p)
                    got_cdf = binom_cdf(kk, n, p)
                    got_sf = binom_sf(kk, n, p)
                    assert math.isclose(got_cdf, want_cdf, rel_tol=1e-9, abs_tol=1e-12), \
                        (n, p, kk, got_cdf, want_cdf)
                    assert math.isclose(got_sf, want_sf, rel_tol=1e-9, abs_tol=1e-12), \
                        (n, p, kk, got_sf, want_sf)
