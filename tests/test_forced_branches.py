"""Forced faults: the exact factored law vs an INDEPENDENT dense statevector oracle.

Random small circuits with T/CS/CCZ, mid-circuit measurements (incl. after T), MR resets,
`!`/M(p) records, classically-controlled feedback, and forced faults before/after the magic.
For every accepted circuit and several fault sets the EXPANDED law must equal the oracle's
branches aggregated by (detectors, observables, decisions) — probabilities and conditional
expectations to 1e-12 — up to ONE constant detector/observable reference XOR per circuit (the
engine's are reference-relative, the oracle's raw). XTIM_FORCED_CASES scales the case count
(default 400 circuits)."""
import os
import random

import numpy as np
import pytest

import xtim
from _forced_oracle import aggregate, branches as oracle_branches, law_shifts, max_diff

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


def random_faults(rng, sites, nrec, refused=()):
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
        if j not in refused and rng.random() < 0.15:
            faults.append({"record": j})
            flips.add(j)
    rng.shuffle(faults)
    return faults, oracle_f, flips


def compare(sim, n, text, faults, oracle_f, flips, ref):
    """ref[0]: the surviving constant reference shifts of this circuit (None = not yet)."""
    law = sim.exact_branches(faults)
    agg = aggregate(oracle_branches(text, n, oracle_f, flips))
    assert abs(law.rows_probabilities.sum() - 1) < 1e-12
    assert abs(law.probabilities.sum() - 1) < 1e-12
    ref[0] = law_shifts(law, agg, ref[0])
    assert ref[0], "no constant detector/observable reference makes the law equal the oracle"
    dp, de = max_diff(law, agg, ref[0][0])
    return len(law), law.num_rows, law.num_uniform, max(dp, de)


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
        # Provenance addressing (fix/forced-provenance): coherentized feedback no longer refuses
        # the circuit; only a record read by coherentized feedback from a readout-NOISELESS
        # measurement (no fold to force) is refused, by name, per record.
        assert not sim.noise_refusal and not sim.record_refusal, (sim.noise_refusal, sim.record_refusal)
        refused = sim.refused_records
        for j in refused:
            stats["refused_by_name"] += 1
            with pytest.raises(ValueError, match="readout-noiseless"):
                sim.exact_branches([{"record": j}])
        stats["folded"] += len(sim.folded_records)
        nrec_ok = nrec
        stats["accepted"] += 1
        stats["magic"] += any(t in text for t in ("T ", "T_DAG", "CS ", "CCZ"))
        ref = [None]
        for _ in range(4):
            faults, oracle_f, flips = random_faults(rng, sites, nrec_ok, refused)
            try:
                nb, nrows, u, w = compare(sim, n, text, faults, oracle_f, flips, ref)
            except AssertionError as ex:
                raise AssertionError(f"seed {seed}: {ex}\n{text}\nfaults={faults}") from None
            stats["queries"] += 1
            stats["branches"] += nb
            stats["magic_rows"] += nrows > 1
            stats["directions"] += u > 0
            stats["worst_exp"] = max(stats["worst_exp"], w)
            stats["faulted"] += bool(faults)


def test_forced_branches_vs_dense_oracle():
    stats = dict(accepted=0, rejected=0, refused_by_name=0, folded=0, magic=0, queries=0, branches=0,
                 magic_rows=0, directions=0, worst_exp=0.0, faulted=0)
    run_cases(int(os.environ.get("XTIM_FORCED_SEED", "0")), CASES, stats)
    print(stats)
    assert stats["accepted"] == CASES
    assert stats["magic"] > CASES // 3
    assert stats["magic_rows"] > 0 and stats["directions"] > 0


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


# ── The wide generator (review I1): record frames on PAULI_EXPECTATION, several DECISIONs and
# observables, RX/RY/MY/MRY, SQRT_X/H_YZ, CY, feedback chains off the latest record. ──────────
WIDE_NOISE = ["DEPOLARIZE1(0.01)", "X_ERROR(0.01)", "Z_ERROR(0.01)", "Y_ERROR(0.01)"]
WIDE_CASES = int(os.environ.get("XTIM_FORCED_WIDE_CASES", "300"))


def random_circuit_wide(rng):
    n = rng.randint(2, 5)
    L, sites, nrec = [], [], 0
    for q in range(n):
        if rng.random() < 0.4:
            L.append(f"{rng.choice(['RX', 'RY', 'R'])} {q}")
    for _ in range(rng.randint(6, 22)):
        r = rng.random()
        q = rng.randrange(n)
        if r < 0.15:
            L.append(f"{rng.choice(['H', 'S', 'S_DAG', 'H', 'SQRT_X', 'H_YZ'])} {q}")
        elif r < 0.25:
            L.append(f"{rng.choice(['T', 'T_DAG'])} {q}")
        elif r < 0.35:
            a, b = rng.sample(range(n), 2)
            L.append(f"{rng.choice(['CX', 'CZ', 'CY'])} {a} {b}")
        elif r < 0.39:
            a, b = rng.sample(range(n), 2)
            L.append(f"CS {a} {b}")
        elif r < 0.42 and n >= 3:
            L.append("CCZ " + " ".join(map(str, rng.sample(range(n), 3))))
        elif r < 0.60:
            m = rng.choice(["M", "MX", "MY", "MR", "MRX", "MRY", "M"])
            arg = "(0.01)" if rng.random() < 0.4 else ""
            inv = "!" if rng.random() < 0.3 else ""
            L.append(f"{m}{arg} {inv}{q}")
            nrec += 1
        elif r < 0.78 and nrec:
            k = 1 if rng.random() < 0.6 else rng.randint(1, min(nrec, 4))
            L.append(f"{rng.choice(['CX', 'CZ', 'CY'])} rec[-{k}] {q}")
        else:
            qs = rng.sample(range(n), rng.choice([1, 2]))
            L.append(f"{rng.choice(WIDE_NOISE)} " + " ".join(map(str, qs)))
            sites.append((1, len(qs)))
    for _ in range(rng.randint(0, 2)):
        L.append(f"{rng.choice(['M', 'MX', 'MY'])} {rng.randrange(n)}")
        nrec += 1
    if nrec:
        for _ in range(rng.randint(0, 3)):
            ks = rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 3)))
            L.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in ks))
        for o in range(rng.randint(0, 2)):
            ks = rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 2)))
            L.append(f"OBSERVABLE_INCLUDE({o}) " + " ".join(f"rec[-{k}]" for k in ks))
        for d in range(rng.randint(0, 3)):
            ks = rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 3)))
            L.append(f"DECISION({d}) " + " ".join(f"rec[-{k}]" for k in ks))
    for i in range(rng.randint(1, 4)):
        qs = rng.sample(range(n), rng.randint(1, n))
        fr = ""
        if nrec and rng.random() < 0.5:
            fr = " " + " ".join(f"rec[-{k}]" for k in
                                rng.sample(range(1, nrec + 1), rng.randint(1, min(nrec, 2))))
        L.append(f"PAULI_EXPECTATION({i}) " + "*".join(f"{rng.choice('XYZ')}{q}" for q in qs) + fr)
    return n, "\n".join(L) + "\n", sites, nrec


def test_forced_wide_generator_vs_dense_oracle():
    st = dict(accepted=0, queries=0, frames=0, multi_dec=0, directions=0, sign_flips=0, magic_rows=0)
    for seed in range(100000, 100000 + 50 * WIDE_CASES):
        if st["accepted"] >= WIDE_CASES:
            break
        rng = random.Random(seed)
        n, text, sites, nrec = random_circuit_wide(rng)
        try:
            sim = xtim.Circuit(text).compile_exact_branches()
        except xtim.XtimRejectError:
            continue
        st["accepted"] += 1
        st["frames"] += any(ln.startswith("PAULI_EXPECTATION") and "rec[" in ln for ln in text.splitlines())
        st["multi_dec"] += text.count("DECISION(") >= 2
        ref = [None]
        for _ in range(4):
            faults, oracle_f, flips = random_faults(rng, sites, nrec, sim.refused_records)
            try:
                compare(sim, n, text, faults, oracle_f, flips, ref)
            except AssertionError as ex:
                raise AssertionError(f"seed {seed}: {ex}\n{text}\nfaults={faults}") from None
            law = sim.exact_branches(faults)
            st["queries"] += 1
            st["directions"] += law.num_uniform > 0
            st["sign_flips"] += bool(law.uniform_expectation_signs.any())
            st["magic_rows"] += law.num_rows > 1
    print(st)
    assert st["accepted"] == WIDE_CASES
    assert min(st["frames"], st["multi_dec"], st["directions"], st["sign_flips"], st["magic_rows"]) > 0, st
