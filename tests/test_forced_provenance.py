"""Forced faults through COHERENTIZED feedback (provenance addressing) vs an inserted-circuit
oracle evaluated by xtim's own exact law (and, on the 4-qubit held-fold circuit, by the
independent dense statevector oracle).

For a fault set F on a circuit C the forced law ``exact_branches(F)`` must equal the NOISELESS
exact law of C_F: C with every noise instruction removed, each forced Pauli written in as a
deterministic Pauli gate at its site, every ``M(p)`` made readout-noiseless, and each forced
record flip realized as a deterministic flip of the REPORTED bit (Stim ``M(1)`` semantics,
written as the ``!`` record inversion — the coherentizer then drives the feedback from the
inverted record, a path independent of the readout-flip fold the forced query fires).

Compared per (detectors, observables, decisions) row of the expanded law: probability and
conditional expectations (1e-12), decisions exactly, detectors/observables up to one constant
reference XOR per fault set (C and C_F have different noiseless references).
"""
import os
import random
import re

import numpy as np
import pytest

import xtim
from _forced_oracle import aggregate, branches as dense_branches, law_shifts, max_diff

NOISE = re.compile(r"^(X_ERROR|Y_ERROR|Z_ERROR|DEPOLARIZE1|DEPOLARIZE2|PAULI_CHANNEL_1|"
                   r"PAULI_CHANNEL_2)\(([^)]*)\)\s+(.*)$")
MEAS = re.compile(r"^(M|MX|MY|MZ|MR|MRX|MRY|MRZ)(\([^)]*\))?\s+(.*)$")
FEEDBACK = re.compile(r"^(CX|CY|CZ)\s+rec\[-(\d+)\]\s+(\d+)$")


def inserted(text, faults, flips):
    """C_F. faults: {(site, target): pauli}; flips: set of records."""
    out, site, rec = [], 0, 0
    for raw in text.splitlines():
        line = raw.strip()
        assert not line.startswith("REPEAT"), "flat circuits only"
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
                if rec in flips:
                    tok = tok[1:] if tok.startswith("!") else "!" + tok
                out.append(f"{m.group(1)} {tok}")
                rec += 1
            continue
        if line.startswith("MPP"):
            toks = line.split()[1:]
            for i, tok in enumerate(toks):
                if rec in flips:
                    toks[i] = tok.replace("!", "", 1) if "!" in tok else "!" + tok
                rec += 1
            out.append("MPP " + " ".join(toks))
            continue
        out.append(raw)
    return "\n".join(out) + "\n"


def oracle_law(text, faults, flips):
    sim = xtim.Circuit(inserted(text, faults, flips)).compile_exact_branches()
    assert not sim.noise_sites()
    return sim.exact_branches([])


def _as_agg(law):
    out = {}
    for i in range(len(law.probabilities)):
        k = (tuple(law.detectors[i].astype(int).tolist()), tuple(law.observables[i].astype(int).tolist()),
             tuple(law.decisions[i].astype(int).tolist()))
        assert k not in out, "duplicate law row"
        out[k] = (float(law.probabilities[i]), np.array(law.expectations[i], float))
    return out


def compare(br, orc):
    """Max |diff| over probabilities and conditional expectations; asserts everything else."""
    assert abs(br.probabilities.sum() - 1) < 1e-12
    agg = _as_agg(orc)
    shifts = law_shifts(br, agg)
    assert shifts, "the laws differ beyond a constant detector/observable reference"
    dp, de = max_diff(br, agg, shifts[0])
    worst = max(dp, de)
    assert worst < 1e-12, worst
    return worst


def feedback_records(text):
    """Records read by any `C? rec[-k] q` feedback line."""
    rec, out = 0, set()
    for raw in text.splitlines():
        line = raw.strip()
        m = MEAS.match(line)
        if m:
            rec += len(m.group(3).split())
            continue
        if line.startswith("MPP"):
            rec += len(line.split()) - 1
            continue
        m = FEEDBACK.match(line)
        if m:
            out.add(rec - int(m.group(2)))
    return out


def random_fault_set(rng, sites, nrec, weight, fb_recs, refused):
    faults, flips = {}, set()
    recs_ok = [j for j in range(nrec) if j not in refused]
    fb_ok = [j for j in sorted(fb_recs) if j not in refused]
    while len(faults) + len(flips) < weight:
        if rng.random() < 0.35:
            j = rng.choice(fb_ok) if fb_ok and rng.random() < 0.6 else rng.choice(recs_ok)
            flips.add(j)
        else:
            s = rng.randrange(len(sites))
            two = sites[s]["channel"] in ("DEPOLARIZE2", "PAULI_CHANNEL_2")
            ntg = len(sites[s]["qubits"]) // (2 if two else 1)
            t = rng.randrange(ntg)
            faults[(s, t)] = (rng.choice([a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"])
                              if two else rng.choice("XYZ"))
    return faults, flips


def query(sim, faults, flips):
    return sim.exact_branches([(s, t, p) for (s, t), p in faults.items()] +
                              [{"record": j} for j in sorted(flips)])


def test_cultivation_d5_no_refusal_and_matches_inserted_oracle():
    text = xtim.load_example("cultivation_d5").text
    sim = xtim.Circuit(text).compile_exact_branches()
    assert sim.noise_refusal == "" and sim.record_refusal == ""
    assert sim.refused_records == {}
    folded = set(sim.folded_records)
    assert len(folded) == 7                      # the 7 folds that broke 1:1 order matching
    fb = feedback_records(text)
    assert folded <= fb
    sites = sim.noise_sites()
    assert len(sites) == 180
    nrec = xtim.Circuit(text).num_measurements
    rng = random.Random(20261011)
    n1 = int(os.environ.get("XTIM_PROV_SINGLE", "200"))
    n23 = int(os.environ.get("XTIM_PROV_MULTI", "100"))
    stats = dict(single=0, multi=0, folded_flips=0, worst=0.0, branches=0)
    # every folded record flipped alone, first
    plan = [({}, {j}) for j in sorted(folded)]
    plan += [random_fault_set(rng, sites, nrec, 1, folded, set()) for _ in range(n1 - len(plan))]
    plan += [random_fault_set(rng, sites, nrec, rng.choice([2, 3]), folded, set()) for _ in range(n23)]
    for faults, flips in plan:
        br = query(sim, faults, flips)
        w = compare(br, oracle_law(text, faults, flips))
        stats["worst"] = max(stats["worst"], w)
        stats["branches"] += br.num_rows
        stats["single" if len(faults) + len(flips) == 1 else "multi"] += 1
        stats["folded_flips"] += bool(flips & folded)
    print(stats)
    assert stats["single"] >= 200 and stats["multi"] >= 100
    assert stats["folded_flips"] >= 50


# Hand-built: noisy M -> feedback across a T, the control observed again (exercises the HELD fold
# and its undo), plus an X-basis noisy M whose control is never observed again (plain fold).
HAND = """R 0 1 2 3
H 0 1 3
T 1
X_ERROR(0.01) 0 1
M(0.05) 0
CX rec[-1] 1
DEPOLARIZE1(0.02) 0 1
T 1
CX 0 2
DEPOLARIZE2(0.01) 0 2
M(0.03) 0
T 3
MX(0.04) 3
CZ rec[-1] 2
T 2
H 1 2
Z_ERROR(0.02) 1
M(0.02) 1 2
DETECTOR rec[-2] rec[-5]
DECISION(0) rec[-1] rec[-4]
PAULI_EXPECTATION(0) Z0
PAULI_EXPECTATION(1) X1*Z2
"""


def _hand_cases():
    sim = xtim.Circuit(HAND).compile_exact_branches()
    sites = sim.noise_sites()
    cases = [({}, set())]
    for s, st in enumerate(sites):
        two = st["channel"] in ("DEPOLARIZE2", "PAULI_CHANNEL_2")
        ntg = len(st["qubits"]) // (2 if two else 1)
        paulis = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"] if two else list("XYZ")
        for t in range(ntg):
            for p in paulis:
                cases.append(({(s, t): p}, set()))
    nrec = xtim.Circuit(HAND).num_measurements
    for j in range(nrec):
        cases.append(({}, {j}))
    for a in range(nrec):
        for b in range(a + 1, nrec):
            cases.append(({}, {a, b}))
    rng = random.Random(7)
    for _ in range(150):
        f, fl = random_fault_set(rng, sites, nrec, rng.choice([2, 3]), feedback_records(HAND), set())
        cases.append((f, fl))
    return sim, cases


def test_hand_built_held_fold_all_faults():
    sim, cases = _hand_cases()
    assert sim.noise_refusal == "" and sim.refused_records == {}
    assert sorted(sim.folded_records) == [0, 2]  # M(0.05) 0 (held + undo) and MX(0.04) 3
    worst, ref = 0.0, None
    for faults, flips in cases:
        br = query(sim, faults, flips)
        worst = max(worst, compare(br, oracle_law(HAND, faults, flips)))
        # and the independent dense statevector oracle (reads the recorded bit for feedback),
        # aggregated by (detectors, observables, decisions); one constant reference per circuit
        agg = aggregate(dense_branches(HAND, 4, faults, flips))
        ref = law_shifts(br, agg, ref)
        assert ref, (faults, flips)
    assert worst < 1e-12


def test_noiseless_coherentized_record_refused_by_name():
    # M without readout noise feeding feedback across a T: no fold channel to force.
    text = "R 0 1\nH 0 1\nM 0\nCX rec[-1] 1\nT 1\nX_ERROR(0.1) 1\nM 1\nMX 1\n"
    sim = xtim.Circuit(text).compile_exact_branches()
    assert sim.noise_refusal == "" and sim.record_refusal == ""
    assert list(sim.refused_records) == [0]
    with pytest.raises(ValueError, match="readout-noiseless"):
        sim.exact_branches([{"record": 0}])
    # other faults on the same circuit are fine and match the oracle
    for faults, flips in [({(0, 0): "X"}, set()), ({}, {1}), ({(0, 0): "Z"}, {2})]:
        compare(query(sim, faults, flips), oracle_law(text, faults, flips))
