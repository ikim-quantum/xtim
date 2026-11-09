"""DEM export of CH-class fault residuals (dem-ch, Unreleased).

A fault that crosses a controlled-Hadamard leaves a GENERAL Clifford end-of-circuit error E (the samplers' PPR retry
class). When both diagonal builds reject, the DEM export rebuilds the table with that retry and states every DEM fact
about the pullbacks Q = E^dag M E: a read with Q = +-R is a deterministic flip, any other read feeds the null-form
record table over the pullbacks. Every circuit that exported before is byte-identical (the CH table is consulted only
after both diagonal builds reject).

Oracles: the EXACT forced-fault engine (compile_exact_branches: the sampler's own propagation incl. PPR residuals; no
dense simulation) per user noise site, and the twirl detector sampler vs stim's sampler of the exported DEM.

Tolerance of the exact comparison: one channel instance's Clifford records are its mixture to FIRST order (the
dem-erasure rulings: an offset coset omits an O(p^2) correction, a mixed instance's Pauli remnant is solved alone), so a
site's law agrees to atol = 3 p_site^2 (p_site = the instance's total fault probability); pure cosets agree to rounding.
"""
import itertools
import math
import os
import re

import numpy as np
import pytest

import xtim
import _exact
from _exact_engine import compile_exact_branches

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dem_ch")
EXAMPLE = os.path.join(HERE, "..", "examples", "ch_cultivation.stim")

NOISE1 = ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1")
NOISE2 = ("DEPOLARIZE2", "PAULI_CHANNEL_2")
NOISE = NOISE1 + NOISE2
P2 = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]

ACCEPTED = ["ch_stab", "ch_obs", "ch_depol2", "ch_expect", "ch_frame", "ch_feedback", "nondiag_small"]


def read(name):
    if name == "ch_cultivation":
        return open(EXAMPLE).read()
    if name == "ja_seed23":   # T/H/CZ random circuit: general (non-diagonal) residuals, no CH
        return open(os.path.join(HERE, "data", "ja_seed23.stim")).read()
    return open(os.path.join(DATA, name + ".stim")).read()


def export(text, include_expectations=True):
    # the exact split (clifford_cutoff = inf), as every exact-oracle DEM test
    return _exact.export_dem_text(text, include_expectations, False, False, True, [], [], 0.0, -1)


def noise_lines(lines):
    return [i for i, l in enumerate(lines) if l.strip().split("(")[0] in NOISE]


def zero_others(lines, idx, keep):
    out = []
    for i, l in enumerate(lines):
        if i in idx and i != keep:
            name, rest = l.split("(", 1)
            args, tail = rest.split(")", 1)
            l = f"{name}({','.join(['0'] * len(args.split(',')))}){tail}"
        out.append(l)
    return "\n".join(out) + "\n"


def site_alternatives(line):
    """[(target index, pauli string, prob)] of one noise instruction."""
    name, rest = line.strip().split("(", 1)
    args, tail = rest.split(")", 1)
    probs = [float(a) for a in args.split(",")]
    qubits = tail.split()
    out = []
    if name in NOISE1:
        for t in range(len(qubits)):
            if name == "X_ERROR": out.append((t, "X", probs[0]))
            elif name == "Y_ERROR": out.append((t, "Y", probs[0]))
            elif name == "Z_ERROR": out.append((t, "Z", probs[0]))
            elif name == "DEPOLARIZE1": out += [(t, P, probs[0] / 3) for P in "XYZ"]
            else: out += [(t, P, pr) for P, pr in zip("XYZ", probs)]
    else:
        for t in range(len(qubits) // 2):
            if name == "DEPOLARIZE2": out += [(t, P, probs[0] / 15) for P in P2]
            else: out += [(t, P, pr) for P, pr in zip(P2, probs)]
    return [a for a in out if a[2] > 0]


def dem_dist(dem, D, O, R):
    """The implied law of the DEM's independent mechanisms over D + O + R bits."""
    K = D + O + R
    dist = np.zeros(1 << K)
    dist[0] = 1.0
    idx = np.arange(1 << K)
    for line in dem.splitlines():
        m = re.match(r"error(\[[^\]]*\])?\(([^)]*)\)\s*(.*)$", line.strip())
        if not m:
            continue
        p = float(m.group(2))
        mask = 0
        for t in m.group(3).split():
            assert t != "^", "decomposed DEM not expected"
            mask |= 1 << (int(t[1:]) if t[0] == "D" else D + int(t[1:]))
        new = dist * (1 - p)
        new[idx ^ mask] += dist * p
        dist = new
    return dist


def bits(row):
    return sum(1 << i for i, b in enumerate(row) if b)


def expand_law(law, D, R):
    """[(prob, det|obs pattern, expectation values[R])] with the uniform directions expanded."""
    rows = []
    u = law.uniform_detectors.shape[0] if law.uniform_detectors.size else 0
    ud = [bits(law.uniform_detectors[k]) | (bits(law.uniform_observables[k]) << D) for k in range(u)]
    us = [np.where(law.uniform_expectation_signs[k], -1.0, 1.0) if R else np.ones(0) for k in range(u)]
    for b in range(len(law.rows_probabilities)):
        pr = float(law.rows_probabilities[b])
        pat = bits(law.rows_detectors[b]) | (bits(law.rows_observables[b]) << D)
        ex = np.array(law.rows_expectations[b], dtype=float) if R else np.zeros(0)
        for sub in itertools.product([0, 1], repeat=u):
            x, s = pat, np.ones(R)
            for k, on in enumerate(sub):
                if on:
                    x ^= ud[k]
                    s = s * us[k]
            rows.append((pr / (1 << u), x, ex * s))
    return rows


def exact_check(text):
    """Per user noise site: max |DEM law - exact law| / (3 p_site^2) over detector/observable pattern probabilities and
    per-pattern expectation moments; the not-Pauli-correctable flags must match the exact magnitude changes."""
    c = xtim.Circuit(text)
    D, O, R = c.num_detectors, c.num_observables, c.num_expectations
    lines = text.splitlines()
    idx = noise_lines(lines)
    l0 = compile_exact_branches(xtim.Circuit(text)).exact_branches([])
    assert len(l0.rows_probabilities) == 1 and l0.uniform_detectors.shape[0] == 0
    c0 = np.array(l0.rows_expectations[0], dtype=float) if R else np.zeros(0)
    worst, devs = 0.0, []
    for site, li in enumerate(idx):
        alts = site_alternatives(lines[li])
        if not alts:
            continue
        p_site = sum(a[2] for a in alts)
        iso = zero_others(lines, idx, li)
        r = export(iso)
        assert r["ok"], (site, r["error"], r["rejected"])
        dist = dem_dist(r["dem"], D, O, R)
        KDO = D + O
        dem_do = np.zeros(1 << KDO)
        dem_mom = np.zeros((1 << KDO, R))
        for x in np.nonzero(dist)[0]:
            xd, xf = x & ((1 << KDO) - 1), x >> KDO
            dem_do[xd] += dist[x]
            for rr in range(R):
                dem_mom[xd, rr] += dist[x] * (-1.0 if (xf >> rr) & 1 else 1.0) * c0[rr]
        sim = compile_exact_branches(xtim.Circuit(iso))
        ex_do = np.zeros(1 << KDO)
        ex_mom = np.zeros((1 << KDO, R))
        kept, flagged = 0.0, {}
        for t, P, pr in alts:
            rows = expand_law(sim.exact_branches([(site, t, P)]), D, R)
            mag = any(q > 0 and abs(abs(ex[rr]) - abs(c0[rr])) > 1e-6 for q, x, ex in rows for rr in range(R))
            if mag:
                sig = tuple(sorted({d for q, x, ex in rows if q > 0 for d in range(D) if (x >> d) & 1}))
                flagged[sig] = flagged.get(sig, 0.0) + pr
                continue
            kept += pr
            for q, x, ex in rows:
                ex_do[x] += pr * q
                ex_mom[x] += pr * q * ex
        ex_do[0] += 1.0 - kept
        ex_mom[0] += (1.0 - kept) * c0
        fl_dem = sorted((tuple(s), q) for s, q in r["postselect_faults"])
        fl_ex = sorted(flagged.items())
        assert len(fl_dem) == len(fl_ex) and all(a[0] == b[0] and abs(a[1] - b[1]) < 1e-12
                                                  for a, b in zip(fl_dem, fl_ex)), (site, fl_dem, fl_ex)
        dev = max(np.abs(dem_do - ex_do).max(), np.abs(dem_mom - ex_mom).max() if R else 0.0)
        devs.append(dev)
        worst = max(worst, dev / (3 * p_site * p_site))
    return worst, devs


def at_p(text, p_old, p_new):
    return text.replace(f"({p_old})", f"({p_new})")


@pytest.mark.parametrize("name", ["ch_cultivation", "ja_seed23"] + ACCEPTED)
def test_ch_class_dem_equals_exact_forced_fault_law(name):
    worst, devs = exact_check(read(name))
    assert devs, name
    assert worst <= 1.0, (name, worst, devs)


@pytest.mark.parametrize("name", ["ch_stab", "ch_expect", "nondiag_small"])
def test_ch_class_deviation_is_second_order(name):
    """The first-order tolerance is earned: the deviation drops ~100x from p = 1e-3 to 1e-4 (or is rounding)."""
    _, d3 = exact_check(read(name))
    _, d4 = exact_check(at_p(read(name), 0.001, 0.0001))
    for a, b in zip(d3, d4):
        assert b <= max(a / 50.0, 1e-14), (name, d3, d4)


def test_ch_cultivation_exports():
    """The example exports at the default cutoff (the weight-3 coset pattern of round 1 dropped and reported) and at
    the exact cutoff as pure-coset Clifford records; the X fault on a |+> ancilla is the identity."""
    text = read("ch_cultivation")
    c = xtim.Circuit(text)
    dem = c.detector_error_model_text(clifford_cutoff=math.inf)
    errs = [l for l in dem.splitlines() if l.startswith("error")]
    cliff = [l for l in errs if l.startswith("error[clifford")]
    assert len(cliff) == 6 and len(errs) == 6, dem
    probs = [float(re.match(r"error\[clifford:f\d+\]\(([^)]*)\) ", l).group(1)) for l in cliff]
    assert all(abs(p - 0.0005) < 1e-15 for p in probs), cliff
    assert c.detector_error_model_text()                       # default cutoff exports too
    w = c.detector_error_model_with_reject()
    assert len(w.postselect_faults) == 0


def test_ch_class_sampler_matches_dem():
    """The twirl detector sampler (full noisy circuit) vs stim's sampler of the exported DEM (exact cutoff, no
    expectation columns): every pattern with >= 20 counts within 5 sigma."""
    stim = pytest.importorskip("stim")
    N = 1_000_000
    for name in ("ch_cultivation", "ch_depol2", "ch_stab", "ch_obs"):
        text = read(name)
        c = xtim.Circuit(text)
        D, O = c.num_detectors, c.num_observables
        r = export(text, include_expectations=False)
        assert r["ok"], r["error"]
        dem_text = "\n".join(re.sub(r"^error\[[^\]]*\]", "error", l) for l in r["dem"].splitlines()
                             if not l.startswith("#"))
        dd, oo, _ = stim.DetectorErrorModel(dem_text).compile_sampler(seed=7).sample(N)
        xd, xo = c.compile_detector_sampler(seed=11).sample(N, separate_observables=True)
        w = (1 << np.arange(D + O)).astype(np.int64)
        key = lambda d, o: (np.concatenate([d, o[:, :O]], axis=1) if O else d).astype(np.int64) @ w
        c1 = np.bincount(key(xd, xo), minlength=1 << (D + O))
        c2 = np.bincount(key(dd, oo), minlength=1 << (D + O))
        m = (c1 + c2) >= 20
        z = np.abs(c1[m] - c2[m]) / np.sqrt(c1[m] + c2[m])
        assert z.max() < 5.0, (name, z.max())


def test_ch_class_with_residual_feedback_is_refused_by_name():
    """CH-class residuals + RESIDUAL classical feedback: the one named refusal (it was the class reject before)."""
    r = export(read("ch_resfeedback"))
    assert not r["ok"] and not r["rejected"]
    assert "CH-class fault residuals" in r["error"] and "feedback" in r["error"], r["error"]


def test_unaligned_t_state_into_ch_still_rejects():
    """Non-commuting magic stays outside the class: the class boundary is unchanged."""
    r = export(read("ch_unaligned"))
    assert not r["ok"] and r["rejected"], r
