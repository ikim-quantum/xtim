"""REVIEWER: xtim's EXACT outcome law of a small noisy circuit, by full enumeration of fault configurations.

Per configuration (one atom or nothing per channel instance, independent readout flips) the law is xtim's exact
engine (_exact_engine.compile_exact_branches(text).exact_branches(cfg)), expanded view, reference-relative.
The configuration weights come from the channel DEFINITIONS (stim docs), written here independently of dcall:
  X/Y/Z_ERROR(p): the Pauli w.p. p; DEPOLARIZE1(p): X,Y,Z each p/3; DEPOLARIZE2(p): 15 Paulis each p/15;
  PAULI_CHANNEL_1(px,py,pz); PAULI_CHANNEL_2(15 args, order IX IY IZ XI .. ZZ);
  E(p1) ELSE(p2) ..: term k w.p. p_k prod_{j<k}(1-p_j) (a term with the same word as an earlier one is summed);
  M(p)/MPP(p)/MXX(p)..: recorded bit flipped w.p. p, independently per record.
Nothing from dcall/dcrec/common is imported.
TEST ORACLE copy (2026-11-11) of an internal review script (md5 61c4aeda96e081a4af233d83aed5664d); verbatim except:
its hardcoded sys.path insert is removed (xtim and _exact_engine come from the PYTHONPATH of the checkout under test),
the __main__ block is removed, and xtim_exact_law(text, max_faults=None) -> (law, tail) is added.
Returns {outcome int over D+O bits (detector d = bit d, observable o = bit D+o): probability}."""
import itertools
import re
import sys

import numpy as np

from _exact_engine import compile_exact_branches  # noqa: E402  (this checkout's tests/, via PYTHONPATH)

P2 = [a + b for a in "IXYZ" for b in "IXYZ"][1:]


def instances(site):
    """list of channel instances; each = (target, [(word, p), ...]) disjoint atoms (identity implied)."""
    ch, qs, ps = site["channel"], site["qubits"], site["probs"]
    out = []
    if ch in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        for t in range(len(qs)):
            out.append((t, [(ch[0], ps[0])]))
    elif ch == "DEPOLARIZE1":
        for t in range(len(qs)):
            out.append((t, [(w, ps[0] / 3) for w in "XYZ"]))
    elif ch == "PAULI_CHANNEL_1":
        for t in range(len(qs)):
            out.append((t, list(zip("XYZ", ps))))
    elif ch == "DEPOLARIZE2":
        for t in range(len(qs) // 2):
            out.append((t, [(w, ps[0] / 15) for w in P2]))
    elif ch == "PAULI_CHANNEL_2":
        for t in range(len(qs) // 2):
            out.append((t, list(zip(P2, ps))))
    elif ch == "CORRELATED_ERROR":
        acc, rem = {}, 1.0
        for w, p in zip(site["paulis"], ps):
            if set(w) != {"I"}:
                acc[w] = acc.get(w, 0.0) + rem * p
            rem *= 1 - p
        out.append((0, list(acc.items())))
    else:
        raise ValueError("channel " + ch)
    return [(t, [(w, p) for w, p in a if p > 0]) for t, a in out if any(p > 0 for _, p in a)]


def readout_records(text):
    """[(absolute record, p)] for measurements with a flip probability (own parse: non-Clifford lines dropped)."""
    import stim
    keep = []
    for ln in text.splitlines():
        s = ln.split("#")[0].strip()
        m = re.match(r"[A-Z_0-9]+", s)
        if m and m.group(0) in ("T", "T_DAG", "CS", "CS_DAG", "CCZ", "CH", "OUTPUT_QUBITS", "PAULI_EXPECTATION",
                                "DECISION"):
            continue
        keep.append(ln)
    c = stim.Circuit("\n".join(keep)).flattened()
    out, j = [], 0
    for inst in c:
        n = inst.num_measurements if hasattr(inst, "num_measurements") else 0
        if n:
            a = inst.gate_args_copy()
            if a and a[0] > 0 and not inst.name.startswith("HERALDED"):
                out += [(j + k, a[0]) for k in range(n)]
            j += n
    return out


def _combos(slots, K):
    """configurations with at most K fired slots: yields list of (addr, p_fired) and the product weight."""
    n = len(slots)
    p0 = [sl[0][1] for sl in slots]
    base = 1.0
    for q in p0:
        base *= q

    def rec(start, k, chosen, w):
        yield chosen, w
        if k == K:
            return
        for i in range(start, n):
            if p0[i] == 0:
                continue
            for a, p in slots[i][1:]:
                yield from rec(i + 1, k + 1, chosen + [a], w * p / p0[i])
    # slots with p0 == 0 cannot be 'not fired': unsupported here
    assert all(q > 0 for q in p0)
    yield from rec(0, 0, [], base)


def exact_law(text, max_configs=2_000_000, return_meta=False, K=None):
    sim = compile_exact_branches(text)
    sites = sim.noise_sites()
    slots = []          # list of [(addr or None, p)]
    for s, site in enumerate(sites):
        for t, atoms in instances(site):
            p0 = 1.0 - sum(p for _, p in atoms)
            slots.append([(None, p0)] + [((s, t, w), p) for w, p in atoms])
    for j, p in readout_records(text):
        slots.append([(None, 1 - p), ({"record": j}, p)])
    ncfg = 1
    for sl in slots:
        ncfg *= len(sl)
    if K is None and ncfg > max_configs:
        raise OverflowError(f"{ncfg} configurations > {max_configs}")
    # exact tail mass P(more than K slots fire) (Poisson-binomial), for the truncated mode
    pb = [1.0]
    for sl in slots:
        q = 1 - sl[0][1]
        new = [0.0] * (len(pb) + 1)
        for i, c in enumerate(pb):
            new[i] += c * (1 - q)
            new[i + 1] += c * q
        pb = new
    tail = 0.0 if K is None else float(sum(pb[K + 1:]))
    L0 = sim.exact_branches([], expectations=[])
    D = L0.rows_detectors.shape[1]
    O = L0.rows_observables.shape[1]
    w = 1 << np.arange(D + O, dtype=object)
    law = {}
    det0 = (L0.rows_detectors.shape[0] == 1 and L0.uniform_detectors.shape[0] == 0)
    def gen():
        if K is None:
            for combo in itertools.product(*slots):
                pr = 1.0
                for _, p in combo:
                    pr *= p
                yield [a for a, _ in combo if a is not None], pr
        else:
            yield from _combos(slots, K)
    nrun = 0
    for cfg, pr in gen():
        if pr == 0:
            continue
        nrun += 1
        L = sim.exact_branches(cfg, expectations=[])
        bits = np.concatenate([np.asarray(L.detectors, bool).reshape(len(L.probabilities), D),
                               np.asarray(L.observables, bool).reshape(len(L.probabilities), O)], 1)
        for row, q in zip(bits, L.probabilities):
            v = 0
            for i in np.flatnonzero(row):
                v |= 1 << int(i)
            law[v] = law.get(v, 0.0) + pr * float(q)
    if return_meta:
        return law, dict(D=D, O=O, ncfg=ncfg, noiseless_rows=int(L0.rows_detectors.shape[0]),
                         noiseless_uniform=int(L0.uniform_detectors.shape[0]), deterministic=bool(det0),
                         nsites=len(sites), K=K, tail=tail, nrun=nrun, nslots=len(slots))
    return law


def xtim_exact_law(text, max_faults=None, max_configs=2_000_000):
    """(law, tail): xtim's exact outcome law {outcome: probability} of a noisy circuit by enumeration of fault
    configurations.  max_faults=None: all configurations (tail = 0.0; refused above max_configs).  max_faults=K:
    only configurations with <= K fired slots (law sums to 1 - tail; tail = exact P(> K slots fire))."""
    law, meta = exact_law(text, max_configs=max_configs, return_meta=True, K=max_faults)
    return law, meta["tail"]
