"""Forced faults, exact branches (3.1.11) vs an INDEPENDENT dense statevector oracle.

Random small circuits with T/CS/CCZ, mid-circuit measurements (incl. after T), MR resets,
`!`/M(p) records, classically-controlled feedback, and forced faults before/after the magic.
For every accepted circuit and several fault sets: the branch SET (keyed by recorded bits),
probabilities (1e-12), decision bits, expectations and the raw detector/observable parities
(the engine's are reference-relative: engine XOR oracle must be ONE constant vector per circuit)
must agree. XTIM_FORCED_CASES scales the case count (default 400 circuits)."""
import os
import random

import numpy as np
import pytest

import xtim
from _forced_oracle import branches as oracle_branches

CASES = int(os.environ.get("XTIM_FORCED_CASES", "400"))
NOISE = [("DEPOLARIZE1", 1), ("X_ERROR", 1), ("Z_ERROR", 1), ("PAULI_CHANNEL_1", 1),
         ("DEPOLARIZE2", 2), ("PAULI_CHANNEL_2", 2)]


def _noise_line(rng, n):
    name, ar = rng.choice(NOISE)
    if ar == 2:
        if n < 2:
            return None, None
        qs = rng.sample(range(n), 2)
    else:
        qs = rng.sample(range(n), rng.choice([1, 1, 2]) if n >= 2 else 1)
    arg = {"PAULI_CHANNEL_1": "0.01,0.02,0.03",
           "PAULI_CHANNEL_2": ",".join(["0.001"] * 15)}.get(name, "0.01")
    return f"{name}({arg}) " + " ".join(map(str, qs)), (ar, len(qs) // ar)


def random_circuit(rng):
    n = rng.randint(2, 5)
    lines, sites, nrec = [], [], 0
    for _ in range(rng.randint(5, 18)):
        r = rng.random()
        q = rng.randrange(n)
        if r < 0.22:
            lines.append(f"{rng.choice(['H', 'S', 'S_DAG', 'X', 'Z', 'H'])} {q}")
        elif r < 0.36:
            lines.append(f"{rng.choice(['T', 'T_DAG'])} {q}")
        elif r < 0.50 and n >= 2:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['CX', 'CZ', 'CX'])} {a} {b}")
        elif r < 0.55 and n >= 2:
            a, b = rng.sample(range(n), 2)
            lines.append(f"CS {a} {b}")
        elif r < 0.58 and n >= 3:
            lines.append("CCZ " + " ".join(map(str, rng.sample(range(n), 3))))
        elif r < 0.72:
            m = rng.choice(["M", "MX", "MY", "MR", "MRX", "M"])
            arg = "(0.01)" if rng.random() < 0.3 else ""
            inv = "!" if rng.random() < 0.2 else ""
            lines.append(f"{m}{arg} {inv}{q}")
            nrec += 1
        elif r < 0.78 and nrec:
            k = rng.randint(1, min(nrec, 3))
            lines.append(f"{rng.choice(['CX', 'CZ', 'CY'])} rec[-{k}] {q}")
        else:
            ln, info = _noise_line(rng, n)
            if ln:
                lines.append(ln)
                sites.append(info)
    for _ in range(rng.randint(0, 2)):          # a terminal read or two
        q = rng.randrange(n)
        lines.append(f"{rng.choice(['M', 'MX'])} {q}")
        nrec += 1
    if nrec:
        for _ in range(rng.randint(0, 3)):
            ks = rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 3)))
            lines.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in ks))
        if rng.random() < 0.5:
            ks = rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 2)))
            lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{k}]" for k in ks))
        if rng.random() < 0.5:
            ks = rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 2)))
            lines.append("DECISION(0) " + " ".join(f"rec[-{k}]" for k in ks))
    for i in range(rng.randint(1, 3)):
        qs = rng.sample(range(n), rng.randint(1, n))
        lines.append(f"PAULI_EXPECTATION({i}) " + "*".join(f"{rng.choice('XYZ')}{q}" for q in qs))
    return n, "\n".join(lines) + "\n", sites, nrec


def random_faults(rng, sites, nrec):
    faults, oracle_f, flips = [], {}, set()
    for s, (ar, ntg) in enumerate(sites):
        for t in range(ntg):
            if rng.random() < 0.35:
                if ar == 1:
                    p = rng.choice("XYZ")
                else:
                    p = rng.choice([a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"])
                faults.append((s, t, p))
                oracle_f[(s, t)] = p
    for j in range(nrec):
        if rng.random() < 0.15:
            faults.append({"record": j})
            flips.add(j)
    rng.shuffle(faults)
    return faults, oracle_f, flips


def compare(sim, n, text, faults, oracle_f, flips, ref):
    br = sim.exact_branches(faults)
    orc = oracle_branches(text, n, oracle_f, flips)
    keys = [tuple(int(b) for b in row) for row in br.measurements]
    assert len(set(keys)) == len(keys), "duplicate branches"
    assert set(keys) == set(orc), f"branch sets differ: {sorted(keys)} vs {sorted(orc)}"
    assert abs(br.probabilities.sum() - 1) < 1e-12
    worst = 0.0
    for i, k in enumerate(keys):
        o = orc[k]
        assert abs(br.probabilities[i] - o["p"]) < 1e-12, (k, br.probabilities[i], o["p"])
        assert list(br.decisions[i].astype(int)) == o["dec"]
        dx = np.bitwise_xor(br.detectors[i].astype(int), np.array(o["det"], int)).tolist()
        ox = np.bitwise_xor(br.observables[i].astype(int), np.array(o["obs"], int)).tolist()
        if ref[0] is None:
            ref[0] = (dx, ox)
        assert (dx, ox) == ref[0], "detector/observable reference not constant"
        d = np.abs(br.expectations[i] - np.array(o["exp"])).max(initial=0.0)
        worst = max(worst, float(d))
        assert d < 1e-12, (k, br.expectations[i], o["exp"])
    return len(keys), worst


def run_cases(seed0, cases, stats):
    for seed in range(seed0, seed0 + cases * 20):
        if stats["accepted"] >= cases:
            break
        rng = random.Random(seed)
        n, text, sites, nrec = random_circuit(rng)
        try:
            sim = xtim.Circuit(text).compile_exact_branches()
        except xtim.XtimRejectError:
            stats["rejected"] += 1
            continue
        if sim.noise_refusal or sim.record_refusal:
            stats["refused_by_name"] += 1        # coherentized feedback: loud refusal
            with pytest.raises(ValueError, match="coherentized"):
                sim.exact_branches([(0, 0, "X")] if sim.noise_refusal else [{"record": 0}])
            if sim.noise_refusal:
                continue
            nrec_ok = 0                          # record flips refused: noise faults only
        else:
            nrec_ok = nrec
        stats["accepted"] += 1
        stats["magic"] += any(t in text for t in ("T ", "T_DAG", "CS ", "CCZ"))
        ref = [None]
        for _ in range(4):
            faults, oracle_f, flips = random_faults(rng, sites, nrec_ok)
            try:
                nb, w = compare(sim, n, text, faults, oracle_f, flips, ref)
            except AssertionError as ex:
                raise AssertionError(f"seed {seed}: {ex}\n{text}\nfaults={faults}") from None
            stats["queries"] += 1
            stats["branches"] += nb
            stats["worst_exp"] = max(stats["worst_exp"], w)
            stats["faulted"] += bool(faults)


def test_forced_branches_vs_dense_oracle():
    stats = dict(accepted=0, rejected=0, refused_by_name=0, magic=0, queries=0, branches=0, worst_exp=0.0, faulted=0)
    run_cases(int(os.environ.get("XTIM_FORCED_SEED", "0")), CASES, stats)
    print(stats)
    assert stats["accepted"] == CASES
    assert stats["magic"] > CASES // 3


def test_addressing_errors():
    sim = xtim.Circuit("H 0\nDEPOLARIZE1(0.1) 0\nDEPOLARIZE2(0.1) 0 1\nM 0 1\n"
                       ).compile_exact_branches()
    assert [s["channel"] for s in sim.noise_sites()] == ["DEPOLARIZE1", "DEPOLARIZE2"]
    for bad in ([(2, 0, "X")], [(0, 1, "X")], [(1, 1, "XX")], [(0, 0, "XX")], [(1, 0, "X")],
                [(1, 0, "II")], [(0, 0, "Q")], [(0, 0, "X"), (0, 0, "Z")],
                [{"record": 2}], [{"record": 0}, {"record": 0}], [("a", 0, "X")]):
        with pytest.raises(ValueError):
            sim.exact_branches(bad)


def test_noiseless_matches_branch_free_detectors():
    c = xtim.load_example("cultivation_d3_faithful")
    sim = c.compile_exact_branches()
    br = sim.exact_branches([])
    assert abs(br.probabilities.sum() - 1) < 1e-12
    assert not br.detectors.any()           # every detector deterministic, reference-relative 0
