"""Forced faults vs the EXISTING exact path: the same faults written into the circuit as
deterministic Pauli gates (every other noise channel and readout coin removed; a forced record
flip as M(1)), recompiled and sampled by the exact sampler, whose per-shot PAULI_EXPECTATION
values are exact. Every sampled shot must be one of the enumerated branches with IDENTICAL
detectors/observables/expectations, the empirical branch frequencies must match the exact
probabilities (chi-square-style 6 sigma per branch), and every branch of probability >= 20/N
must be observed. Plus Clifford circuits vs stim's exact branch distribution."""
import random
import re

import numpy as np
import pytest

import xtim
from xtim import _xtim

NOISE = re.compile(r"^(X_ERROR|Y_ERROR|Z_ERROR|DEPOLARIZE1|DEPOLARIZE2|PAULI_CHANNEL_1|"
                   r"PAULI_CHANNEL_2)\(([^)]*)\)\s+(.*)$")
MEAS = re.compile(r"^(M|MX|MY|MZ|MR|MRX|MRY|MRZ)(\([^)]*\))?\s+(.*)$")


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


def check(text, faults, flips, shots=4000, seed=1):
    sim = xtim.Circuit(text).compile_exact_branches()
    q = [(s, t, p) for (s, t), p in faults.items()] + [{"record": j} for j in flips]
    br = sim.exact_branches(q)
    idx = {tuple(row.astype(int)): i for i, row in enumerate(br.measurements)}
    prog = _xtim.CompiledProgram(written_in(text, faults, flips))
    r = prog.sample(shots, seed)
    assert r["ok"] and not r["rejected"], r
    M, D, O = r["num_measurements"], r["num_detectors"], r["num_observables"]
    meas = np.unpackbits(r["measurements"], axis=1, bitorder="little")[:, :M]
    det = np.unpackbits(r["detectors"], axis=1, bitorder="little")[:, :D]
    obs = np.unpackbits(r["observables"], axis=1, bitorder="little")[:, :O]
    exps = r["expectations"]
    counts = np.zeros(len(br), int)
    ref = None      # the written-in circuit's reference contains the fault: a constant XOR
    for s in range(shots):
        i = idx.get(tuple(meas[s].astype(int)))
        assert i is not None, f"sampled record {meas[s]} is not an enumerated branch"
        counts[i] += 1
        x = (tuple(det[s] ^ br.detectors[i]), tuple(obs[s] ^ br.observables[i]))
        ref = x if ref is None else ref
        assert x == ref, "detector/observable channels differ beyond a constant reference"
        assert np.abs(exps[s] - br.expectations[i]).max(initial=0) < 1e-9
    from scipy.stats import binom
    B = len(br)
    for i, p in enumerate(br.probabilities):       # two-sided binomial tail, Bonferroni over B
        lo = binom.cdf(counts[i], shots, p)
        hi = binom.sf(counts[i] - 1, shots, p)
        assert min(lo, hi) * 2 * B > 1e-7, (i, counts[i], shots * p, B)
        if p >= 20 / shots:
            assert counts[i] > 0
    return len(br)


CASES = [("cultivation_d3_faithful", True), ("ch_cultivation", True),
         ("code_switching_faithful", True), ("miniature_oracle", True),
         ("distillation_15_1_3", False)]


@pytest.mark.parametrize("name,flips_ok", CASES)
def test_forced_vs_written_in(name, flips_ok):
    text = xtim.load_example(name).text
    sim = xtim.Circuit(text).compile_exact_branches()
    sites = sim.noise_sites()
    nrec = xtim.Circuit(text).num_measurements
    rng = random.Random(hash(name) & 0xFFFF)
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
    """Clifford circuits: the enumerated branches == stim's exact branch distribution (uniform
    over 2^r outcome patterns, r = the GF(2) rank of the random measurement part), every stim
    sample is a branch, and the detectors match stim's parities relative to the UNFAULTED
    circuit's reference sample."""
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
        br = sim.exact_branches([(s, t, p) for (s, t), p in faults.items()])
        ref = np.array(c.reference_sample(), int)
        wc = stim.Circuit(written_in(text, faults, set()))
        meas = wc.compile_sampler(seed=seed).sample(3000).astype(int)
        rows = {tuple(r): i for i, r in enumerate(br.measurements.astype(int))}
        for m in meas:
            assert tuple(m) in rows, "a stim sample is not an enumerated branch"
        # GF(2) rank of the random part
        D = (meas ^ meas[0]) % 2
        r = 0
        D = D.copy()
        for col in range(D.shape[1]):
            piv = np.nonzero(D[r:, col])[0]
            if not len(piv):
                continue
            D[[r, r + piv[0]]] = D[[r + piv[0], r]]
            rows_ = np.nonzero(D[:, col])[0]
            D[rows_[rows_ != r]] ^= D[r]
            r += 1
            if r == D.shape[0]:
                break
        assert len(br) == 2 ** r
        assert np.allclose(br.probabilities, 1.0 / len(br), atol=1e-12)
        # detectors: parity of the branch's records XOR the unfaulted reference parity
        det_lists = []
        nrec = 0
        for inst in c.flattened():
            if inst.name == "DETECTOR":
                det_lists.append([nrec + t.value for t in inst.targets_copy()])
            elif stim.gate_data(inst.name).produces_measurements:
                nrec += len(inst.targets_copy())
        for i, row in enumerate(br.measurements.astype(int)):
            want = [(row[d].sum() + ref[d].sum()) % 2 for d in det_lists]
            assert list(br.detectors[i].astype(int)) == want
