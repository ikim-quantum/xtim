"""Lean-exporter fuzz (TEST ONLY, LEAN-EXPORT-PLAN Task 0): small Clifford+T circuits with NO determinism or class
filter, so every refusal of the xdem export is reachable. Three families, cycled:
  * "free":   a random gate/noise/measurement soup (mostly non-deterministic detectors: the refusal paths);
  * "mirror": C then C^dag with noise in between, read in the prepared basis (deterministic detectors);
  * "mid":    1-3 segments C_k C_k^dag, each followed by a mid-circuit read of one qubit in the prepared basis
              (a live-wire MX: the diagonal-copy deferral) and, in 30 % of segments, a `CX rec[-1] q` feedback.
generate(seed, count) is a pure function of its arguments (one random.Random(seed) stream)."""
import random

INV = {"T": "T_DAG", "T_DAG": "T", "CS": "CS_DAG", "CS_DAG": "CS", "S": "S_DAG", "S_DAG": "S"}


def _free(rng):
    n = rng.randint(3, 5)
    lines, nrec = [], 0
    for _ in range(rng.randint(6, 22)):
        r = rng.random()
        q = rng.sample(range(n), 3)
        if r < 0.18:
            lines.append(f"{rng.choice(['T', 'T_DAG'])} {q[0]}")
        elif r < 0.22:
            lines.append(f"{rng.choice(['CS', 'CS_DAG'])} {q[0]} {q[1]}")
        elif r < 0.25:
            lines.append(f"CCZ {q[0]} {q[1]} {q[2]}")
        elif r < 0.40:
            lines.append(f"{rng.choice(['H', 'S', 'S_DAG', 'X', 'Z'])} {q[0]}")
        elif r < 0.55:
            lines.append(f"{rng.choice(['CX', 'CZ'])} {q[0]} {q[1]}")
        elif r < 0.72:
            lines.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'Y_ERROR', 'DEPOLARIZE1'])}(0.01) {q[0]}")
        elif r < 0.76:
            lines.append(f"DEPOLARIZE2(0.01) {q[0]} {q[1]}")
        elif r < 0.86:
            lines.append(f"{rng.choice(['M', 'MX', 'MR', 'MRX', 'MY'])} {q[0]}")
            nrec += 1
            if rng.random() < 0.5:
                lines.append("DETECTOR rec[-1]")
        elif r < 0.90 and nrec:
            lines.append(f"CX rec[-1] {q[0]}")
        elif r < 0.95:
            lines.append(f"{rng.choice(['R', 'RX'])} {q[0]}")
        else:
            lines.append(f"MPP {rng.choice('XZ')}{q[0]}*{rng.choice('XZ')}{q[1]}")
            nrec += 1
            lines.append("DETECTOR rec[-1]")
    lines.append(rng.choice(["M", "MX"]) + " " + " ".join(map(str, range(n))))
    for k in range(1, n + 1):
        if rng.random() < 0.5:
            lines.append(f"DETECTOR rec[-{k}]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]")
    return "\n".join(lines) + "\n"


def _body(rng, n, lo, hi):
    b = []
    for _ in range(rng.randint(lo, hi)):
        r = rng.random()
        q = rng.sample(range(n), 3)
        if r < 0.3:
            b.append(f"{rng.choice(['T', 'T_DAG'])} {q[0]}")
        elif r < 0.4:
            b.append(f"{rng.choice(['CS', 'CS_DAG'])} {q[0]} {q[1]}")
        elif r < 0.45:
            b.append(f"CCZ {q[0]} {q[1]} {q[2]}")
        elif r < 0.67:
            b.append(f"{rng.choice(['H', 'S', 'S_DAG'])} {q[0]}")
        else:
            b.append(f"{rng.choice(['CX', 'CZ'])} {q[0]} {q[1]}")
    return b


def _inv(line):
    g, *t = line.split()
    return " ".join([INV.get(g, g)] + t)


def _mirror(rng, mid):
    n = rng.randint(3, 5) if not mid else rng.randint(3, 4)
    lines = []

    def noise():
        if rng.random() < (0.7 if mid else 0.6):
            lines.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1'])}(0.01) {rng.randrange(n)}")

    xb = rng.random() < 0.5
    if xb or mid:
        lines.append(("RX " if xb else "R ") + " ".join(map(str, range(n))))
    for _ in range(rng.randint(1, 3) if mid else 1):
        b = _body(rng, n, 2, 7) if mid else _body(rng, n, 3, 10)
        for ln in b:
            lines.append(ln)
            noise()
        for ln in reversed(b):
            lines.append(_inv(ln))
            noise()
        if mid:
            lines.append(("MX " if xb else "M ") + str(rng.randrange(n)))
            lines.append("DETECTOR rec[-1]")
            if rng.random() < 0.3:
                lines.append(f"CX rec[-1] {rng.randrange(n)}")
    lines.append(("MX " if xb else "M ") + " ".join(map(str, range(n))))
    for k in range(1, n + 1):
        lines.append(f"DETECTOR rec[-{k}]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return "\n".join(lines) + "\n"


def generate(seed, count):
    """count circuits as (name, text), families cycled free / mirror / mid."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        fam = ("free", "mirror", "mid")[i % 3]
        text = _free(rng) if fam == "free" else _mirror(rng, fam == "mid")
        out.append((f"{fam}{seed}_{i}", text))
    return out


def generate_beta(seed, count):
    """count CCZ-heavy circuits as (name, text) for the lean determinism check's pair-coefficient (beta) path (Task 1
    review M1; a port of the reviewer's beta_fuzz family): 3-5 qubits in |+>, 1-5 layers of mostly CCZ plus CS/CS_DAG,
    T/T_DAG, CX (kappa weight >= 2) and CZ, MX of every qubit, 2-5 detectors on random parities. Pure in its
    arguments (one random.Random(seed) stream)."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        n = rng.randint(3, 5)
        q = " ".join(map(str, range(n)))
        lines = [f"RX {q}", f"DEPOLARIZE1(0.01) {q}"]
        for _ in range(rng.randint(1, 5)):
            op = rng.randrange(10)
            v = rng.sample(range(n), 3)
            if op < 4:
                lines.append(f"CCZ {v[0]} {v[1]} {v[2]}")
            elif op < 5:
                lines.append(f"{rng.choice(['CS', 'CS_DAG'])} {v[0]} {v[1]}")
            elif op < 7:
                lines.append(f"CX {v[0]} {v[1]}")
            elif op < 8:
                lines.append(f"{rng.choice(['T', 'T_DAG'])} {v[0]}")
            else:
                lines.append(f"CZ {v[0]} {v[1]}")
            if rng.randrange(3) == 0:
                lines.append(f"X_ERROR(0.01) {rng.randrange(n)}")
        lines.append(f"MX {q}")
        for _ in range(rng.randint(2, 5)):
            recs = [f"rec[-{n - k}]" for k in range(n) if rng.random() < 0.5] or ["rec[-1]"]
            lines.append("DETECTOR " + " ".join(recs))
        out.append((f"beta{seed}_{i}", "\n".join(lines) + "\n"))
    return out


def generate_boundary(seed, count):
    """count boundary circuits as (name, text) for the H-condition soundness check (Task 2 review M1; a port of the
    reviewer's genb.py): H placed among forwarded location Z's, CX/CZ/CY/SWAP/XCZ/ISWAP spreading, SQRT_X/SQRT_Y/
    H_YZ/C_XYZ, X/Y/Z/DEPOLARIZE noise at p = 0.01 or 0, M/MX/MY/MR/MRX/M(p), CX/CY/CZ rec feedback, R/RX/RY resets,
    MPP over X/Y/Z, and a REPEAT block in 40 % of circuits. Pure in its arguments (one random.Random(seed) stream)."""
    rng = random.Random(seed)

    def one():
        n = rng.randint(3, 6)
        nrec = 0
        lines = [rng.choice(["R", "RX"]) + " " + " ".join(map(str, range(n)))]

        def body(k):
            nonlocal nrec
            out = []
            for _ in range(k):
                r = rng.random()
                q = rng.sample(range(n), 3)
                if r < 0.15:
                    out.append(f"{rng.choice(['T', 'T_DAG'])} {q[0]}")
                elif r < 0.2:
                    out.append(f"{rng.choice(['CS', 'CS_DAG'])} {q[0]} {q[1]}")
                elif r < 0.23:
                    out.append(f"CCZ {q[0]} {q[1]} {q[2]}")
                elif r < 0.40:
                    out.append(f"H {q[0]}")
                elif r < 0.47:
                    out.append(f"{rng.choice(['S', 'S_DAG', 'SQRT_X', 'SQRT_Y', 'H_YZ', 'C_XYZ', 'SQRT_X_DAG'])} {q[0]}")
                elif r < 0.62:
                    out.append(f"{rng.choice(['CX', 'CZ', 'CY', 'SWAP', 'XCZ', 'ISWAP'])} {q[0]} {q[1]}")
                elif r < 0.74:
                    out.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'Y_ERROR', 'DEPOLARIZE1'])}"
                               f"({rng.choice([0.01, 0.0])}) {q[0]}")
                elif r < 0.77:
                    out.append(f"DEPOLARIZE2(0.01) {q[0]} {q[1]}")
                elif r < 0.84:
                    out.append(f"{rng.choice(['M', 'MX', 'MR', 'MRX', 'MY', 'M(0.01)'])} {q[0]}")
                    nrec += 1
                    if rng.random() < 0.5:
                        out.append("DETECTOR rec[-1]")
                elif r < 0.89 and nrec:
                    out.append(f"{rng.choice(['CX', 'CZ', 'CY'])} rec[-1] {q[0]}")
                elif r < 0.93:
                    out.append(f"{rng.choice(['R', 'RX', 'RY'])} {q[0]}")
                else:
                    out.append(f"MPP {rng.choice('XYZ')}{q[0]}*{rng.choice('XYZ')}{q[1]}")
                    nrec += 1
                    out.append("DETECTOR rec[-1]")
            return out

        lines += body(rng.randint(3, 12))
        if rng.random() < 0.4:
            b = body(rng.randint(2, 6))
            reps = rng.randint(2, 3)
            lines += [f"REPEAT {reps} {{"] + b + ["}"]
        lines += body(rng.randint(0, 6))
        lines.append(rng.choice(["M", "MX"]) + " " + " ".join(map(str, range(n))))
        for k in range(1, n + 1):
            if rng.random() < 0.6:
                lines.append(f"DETECTOR rec[-{k}]")
        lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
        return "\n".join(lines) + "\n"

    return [(f"boundary{seed}_{i}", one()) for i in range(count)]


def generate_mixed(seed, count):
    """count small circuits as (name, text) whose columns mix non-deterministic reads (a T / T_DAG / CCZ-dressed
    |+>) with deterministic reads carrying a folded or user sign (X;M, Z;MX, S_DAG;MY, MR, M !q, MPP, MPP !P):
    the refusal value's sign convention (nondet-message fix round 2)."""
    rng = random.Random(seed)

    def one():
        n = rng.randrange(2, 5)
        lines, reads = [], []
        magic = rng.sample(range(n), rng.randrange(1, n))       # at least one plain qubit
        for q in range(n):
            lines.append(rng.choice(["R", "RX", "RY"]) + f" {q}")
        for q in magic:
            lines.append(f"RX {q}")
            lines.append(rng.choice(["T", "T_DAG", "T", "S"]) + f" {q}")
        if len(magic) >= 2 and rng.random() < 0.3:
            lines.append(f"CZ {magic[0]} {magic[1]}")
        lines.append(f"X_ERROR(0.01) {magic[0]}")
        for q in range(n):
            for _ in range(rng.randrange(0, 3)):
                lines.append(rng.choice(["X", "Z", "S", "S_DAG", "Y"]) + f" {q}")
        for q in range(n):
            r = rng.random()
            if r < 0.15:
                lines.append(f"MR {q}")
            elif r < 0.3:
                lines.append(f"M{rng.choice(['', 'X', 'Y'])} !{q}")
            elif r < 0.45:
                lines.append(f"MPP {'!' if rng.random() < 0.5 else ''}{rng.choice('XYZ')}{q}")
            else:
                lines.append(f"M{rng.choice(['', 'X', 'Y'])} {q}")
            reads.append(q)
        if rng.random() < 0.3:
            a, b = rng.sample(range(n), 2)
            lines.append(f"MPP {rng.choice('XYZ')}{a}*{rng.choice('XYZ')}{b}")
            reads.append(-1)
        m = len(reads)
        for _ in range(rng.randrange(1, 4)):
            ks = rng.sample(range(1, m + 1), rng.randrange(1, min(m, 3) + 1))
            lines.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in ks))
        if rng.random() < 0.4:
            lines.append(f"OBSERVABLE_INCLUDE(0) rec[-{rng.randrange(1, m + 1)}] rec[-{rng.randrange(1, m + 1)}]")
        return "\n".join(lines) + "\n"

    return [(f"mixed{seed}_{i}", one()) for i in range(count)]
