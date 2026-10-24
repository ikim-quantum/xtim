"""Stage S1 helpers: the exact oracle (X) folded into the engine's emitted-column convention, and the
random-circuit family of its gate against the independent dense oracle (tests/_record_oracle.py).

(X) itself is `xtim._xtim_testapi.RecordLawOracle` (cpp/testapi/record_law_oracle.cpp): per shot, the joint law of
every record column from the term-indexed residual (design Theorem 1 + Lemma 2). Here the shot laws
are mixed EXACTLY over every fault configuration of the deferred circuit's noise sites and every
readout-flip coin, then shifted by the emission offsets (record inverts; detector / observable
columns relative to the noiseless reference, decisions raw) — the law the engine is meant to emit.
"""
import itertools
import random

import numpy as np

from xtim import _xtim_testapi


def oracle(text):
    """(X) for `text`: the test-API extension's RecordLawOracle (cpp/testapi)."""
    return _xtim_testapi.RecordLawOracle(text)

CODE = {"X": 1, "Y": 2, "Z": 3}


def frame_arg(frame):
    if not frame:
        return None
    qs = sorted(frame)
    return (qs, [CODE[frame[q]] for q in qs])


def fault_configs(orc, max_configs=1 << 15):
    """Every fault configuration of the deferred noise sites: (probability, faults)."""
    sites = orc.noise_sites()
    sizes = [len(s["probs"]) for s in sites]
    if int(np.prod(sizes, dtype=float)) > max_configs:
        raise ValueError("too many fault configurations")
    for combo in itertools.product(*[range(k) for k in sizes]):
        pr = 1.0
        faults = []
        for s, i in zip(sites, combo):
            pr *= s["probs"][i]
            if i:
                faults.append((s["op_pos"], s["qubits"], s["outcomes"][i]))
        if pr > 0:
            yield pr, faults


def emitted_law_x(orc, frame=None):
    """Exact joint law {column tuple: probability} of the emitted columns, from (X)."""
    m = len(orc.columns())
    acc = {}
    fr = frame_arg(frame)
    for pr, faults in fault_configs(orc):
        S, Px, Pz = orc.residual(faults, fr)
        keys, probs = orc.enumerate(S, orc.relabel(Px, Pz))
        for k, p in zip(keys.tolist(), probs.tolist()):
            acc[k] = acc.get(k, 0.0) + pr * p
    cols = orc.columns()
    for r, p in enumerate(orc.rec_flip):
        if p <= 0:
            continue
        mask = sum(1 << c for c, (_, _, recs) in enumerate(cols) if r in recs)
        new = {}
        for k, v in acc.items():
            new[k] = new.get(k, 0.0) + v * (1 - p)
            new[k ^ mask] = new.get(k ^ mask, 0.0) + v * p
        acc = new
    off = sum(int(b) << c for c, b in enumerate(orc.emission_offset))
    return {tuple(((k ^ off) >> c) & 1 for c in range(m)): v for k, v in acc.items()}


def max_abs_diff(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


# ── the random-circuit family ────────────────────────────────────────────────────────────────
# Flavours (each circuit draws one): "diag" (T / CS / CCZ magic on |+>), "ppr" (magic in a rotated
# frame: H T H, CH), "mirror" (T ... T_DAG pairs sandwiched by Cliffords, faults between), "a4"
# (Z0Z1 / X0X1 / Y0Y1 product reads of a two-qubit magic register). Every circuit gets random
# detectors (deterministic repeats, gauge and ANTI parities), observables, decisions and, half the
# time, an INPUT_QUBITS port evaluated under random frames.

_NOISE = ["X_ERROR", "Z_ERROR", "Y_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1", "DEPOLARIZE2"]


def _noise(r, n, L, budget):
    if budget[0] <= 0:
        return
    kind = r.choice(_NOISE)
    p = r.choice([0.05, 0.1, 0.2, 0.3, 0.5])
    if kind == "DEPOLARIZE2":
        if n < 2:
            return
        a, b = r.sample(range(n), 2)
        L.append(f"DEPOLARIZE2({p}) {a} {b}")
        budget[0] -= 2
        return
    q = r.randrange(n)
    if kind == "PAULI_CHANNEL_1":
        ps = [round(r.uniform(0, 0.2), 3) for _ in range(3)]
        L.append(f"PAULI_CHANNEL_1({ps[0]},{ps[1]},{ps[2]}) {q}")
    else:
        L.append(f"{kind}({p}) {q}")
    budget[0] -= 1


def _cliff(r, n, L, k=1):
    for _ in range(k):
        g = r.choice(["H", "S", "S_DAG", "CX", "CZ", "X", "Z"] if n > 1 else ["H", "S", "S_DAG", "X", "Z"])
        if g in ("CX", "CZ"):
            a, b = r.sample(range(n), 2)
            L.append(f"{g} {a} {b}")
        else:
            L.append(f"{g} {r.randrange(n)}")


def random_circuit(seed):
    """(text, frames) — frames: list of {qubit: letter} (empty when the circuit has no port)."""
    r = random.Random(seed)
    flavour = ["diag", "ppr", "mirror", "a4"][seed % 4]
    n = r.randrange(2, 5) if flavour != "a4" else r.randrange(2, 4)
    budget = [r.randrange(1, 4)]
    has_port = r.random() < 0.5 and flavour != "a4"
    port = sorted(r.sample(range(n), r.randrange(1, min(2, n) + 1))) if has_port else []
    lever = None
    L = []
    if port:
        L.append("INPUT_QUBITS in " + " ".join(map(str, port)))
    for q in range(n):
        if q in port:
            continue
        L.append(r.choice(["R", "RX", "RX", "RY"]) + f" {q}")
    if flavour == "diag":
        mq = r.sample(range(n), r.randrange(1, n + 1))
        _noise(r, n, L, budget)
        for q in mq:
            L.append(r.choice(["T", "T_DAG"]) + f" {q}")
        if n >= 2 and r.random() < 0.5:
            a, b = r.sample(range(n), 2)
            L.append(r.choice(["CS", "CS_DAG", "CZ"]) + f" {a} {b}")
        if n >= 3 and r.random() < 0.4:
            L.append("CCZ " + " ".join(map(str, r.sample(range(n), 3))))
        _noise(r, n, L, budget)
        for _ in range(r.randrange(0, 3)):
            g = r.choice(["CX", "CZ", "S"])
            if g == "S":
                L.append(f"S {r.randrange(n)}")
            else:
                a, b = r.sample(range(n), 2)
                L.append(f"{g} {a} {b}")
    elif flavour == "ppr":
        q = r.randrange(n)
        _noise(r, n, L, budget)
        if n >= 2 and r.random() < 0.5:
            c = r.choice([x for x in range(n) if x != q])
            L.append(f"CH {c} {q}")
        else:
            L.append(f"H {q}")
            L.append(r.choice(["T", "T_DAG"]) + f" {q}")
            L.append(f"H {q}")
        _noise(r, n, L, budget)
        _cliff(r, n, L, r.randrange(0, 3))
    elif flavour == "mirror":
        q = r.randrange(n)
        D = []
        others = [x for x in range(n) if x not in port]
        if port and others:          # l4 shape: the port is crossed twice by a Z_c Z_q mirror
            q = r.choice(port)
            lever = r.choice(others)
            L.append(f"RX {lever}")
            L.append(f"T {lever}")
            D.append(("CX", [lever, q]))
        for _ in range(r.randrange(1, 4)):
            if n > 1 and r.random() < 0.7:
                a, b = r.sample(range(n), 2)
                D.append((r.choice(["CX", "CZ"]), [a, b]))
            else:
                D.append(("S", [r.randrange(n)]))
        inv = {"S": "S_DAG", "CX": "CX", "CZ": "CZ"}
        fwd = [f"{g} " + " ".join(map(str, t)) for g, t in D]
        bwd = [f"{inv[g]} " + " ".join(map(str, t)) for g, t in reversed(D)]
        split = lever is not None or r.random() < 0.5
        # split (adv-A l4 shape): D T D^-1 . fault . D T_DAG D^-1 — the fault crosses only the
        # second term of the mirror pair; otherwise D T . fault . T_DAG D^-1.
        L += fwd + [f"T {q}"] + (bwd if split else [])
        if lever is not None:
            L.append(f"{r.choice(['X_ERROR', 'Y_ERROR', 'DEPOLARIZE1'])}({r.choice([0.1, 0.3, 0.5])}) {lever}")
        else:
            _noise(r, n, L, budget)
        if r.random() < 0.5:
            _noise(r, n, L, budget)
        L += (fwd if split else []) + [f"T_DAG {q}"] + bwd
        _noise(r, n, L, budget)
    else:   # a4: product reads on a two-qubit magic register
        L.append("RX 0 1")
        L.append("T 0")
        L.append(r.choice(["T", "T_DAG", "S"]) + " 1")
        if r.random() < 0.5:
            L.append("CX 0 1")
        _noise(r, n, L, budget)
        L.append("MPP Z0*Z1 X0*X1 Y0*Y1")
        L += ["DETECTOR rec[-3]", "DETECTOR rec[-2]", "DETECTOR rec[-1]"]
    # measurements: some mid-circuit repeats (deterministic pairs), then terminal reads
    recs = 3 if flavour == "a4" else 0
    if r.random() < 0.5:
        q = r.randrange(n)
        b = r.choice(["M", "MX"])
        L.append(f"{b} {q}")
        _noise(r, n, L, budget)
        L.append(f"{b} {q}")
        recs += 2
        L.append("DETECTOR rec[-1] rec[-2]")
    if r.random() < 0.3:
        q = r.randrange(n)
        t = r.choice([x for x in range(n) if x != q]) if n > 1 else q
        if t != q:
            L.append(f"M {q}")
            L.append(f"CX rec[-1] {t}")
            recs += 1
    final = []
    for q in range(n):
        b = "MX" if q == lever else r.choice(["M", "MX", "MY", "M", "MX"])
        rp = r.random() < 0.15
        inv = "!" if r.random() < 0.15 else ""
        final.append(f"{b}{'(0.1)' if rp else ''} {inv}{q}")
    L += final
    recs += n
    if lever is not None:   # the l4 read: the magic lever in X, a Born-valued detector
        L.append(f"DETECTOR rec[-{n - lever}]")
    # random parities over the records: detectors, observables, decisions
    nd = r.randrange(1, 4)
    for _ in range(nd):
        k = r.randrange(1, min(3, recs) + 1)
        sel = r.sample(range(1, recs + 1), k)
        L.append("DETECTOR " + " ".join(f"rec[-{s}]" for s in sel))
    for o in range(r.randrange(0, 3)):
        sel = r.sample(range(1, recs + 1), r.randrange(1, min(2, recs) + 1))
        L.append(f"OBSERVABLE_INCLUDE({o}) " + " ".join(f"rec[-{s}]" for s in sel))
    for d in range(r.randrange(0, 3)):
        sel = r.sample(range(1, recs + 1), r.randrange(1, min(2, recs) + 1))
        L.append(f"DECISION({d}) " + " ".join(f"rec[-{s}]" for s in sel))
    text = "\n".join(L) + "\n"
    frames = []
    if port:
        for _ in range(2):
            frames.append({q: r.choice("XYZ") for q in port})
    return text, frames
