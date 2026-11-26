"""Generator note for tests/data/ladder_settle/pairs.json (fix/ladder-settle review round 1; the reviewer's gen_pairs.py,
adapted). Run on the PRE-fix build (dev/next b0a06fc):

  usage: PYTHONPATH=<wt>:<wt>/tests python gen_pairs.py SEEDS OUT.json

Pairs (A, B): A carries magic doubles (T;T, T;diag;T, CS;CS, CH;CH, CCZ;CCZ), B is A with each double rewritten to its
Clifford (T;T -> S, T;D;T -> D;S, CS;CS -> CZ, CH;CH -> I, CCZ;CCZ -> I), so A and B are the same unitary. Kept: A refused
by the plain build "even-Clifford ladder gap", B accepted (it needs no settle). Over seeds 0..SEEDS-1 of both generator
modes (Clifford-only and with single magic gates), the fixture keeps the 3 shortest A per (double kind, mode)."""
import json
import re
import sys

import numpy as np
from xtim import _xtim

ONE = ["H", "S", "S_DAG", "H_YZ", "SQRT_X", "X", "Z"]
ODD1 = ["T", "T_DAG"]
ODD2 = ["CS", "CS_DAG"]

def gen(seed, clifford_only):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    A, B = [], []
    for _ in range(int(rng.integers(6, 22))):
        r = rng.random()
        q = int(rng.integers(n))
        o = int((q + 1 + rng.integers(n - 1)) % n)
        if r < 0.30:   # doubles
            k = rng.random()
            if k < 0.4:
                g = rng.choice(ODD1)
                if rng.random() < 0.5:
                    A += [f"{g} {q}", f"{g} {q}"]
                else:   # separated by a diagonal Clifford on q
                    sep = rng.choice([f"CZ {q} {o}", f"S {q}", f"Z {q}"])
                    A += [f"{g} {q}", sep, f"{g} {q}"]
                    B += [sep]
                B += ["S " + str(q) if g == "T" else "S_DAG " + str(q)]
            elif k < 0.65:
                g = rng.choice(ODD2)
                A += [f"{g} {q} {o}", f"{g} {q} {o}"]
                B += [f"CZ {q} {o}"]
            elif k < 0.85:
                A += [f"CH {q} {o}", f"CH {q} {o}"]
            elif n >= 3:
                c = sorted(rng.choice(n, 3, replace=False).tolist())
                A += ["CCZ " + " ".join(map(str, c))] * 2
        elif r < 0.30 + (0 if clifford_only else 0.15):
            if rng.random() < 0.6:
                l = f"{rng.choice(ODD1)} {q}"
            else:
                l = f"{rng.choice(ODD2)} {q} {o}"
            A.append(l); B.append(l)
        elif r < 0.75:
            l = f"{rng.choice(ONE)} {q}"; A.append(l); B.append(l)
        else:
            l = f"{rng.choice(['CX', 'CZ'])} {q} {o}"; A.append(l); B.append(l)
    return n, "\n".join(A) + "\n", "\n".join(B) + "\n"

def status(t):
    try:
        _xtim.bare_state_of(t); return "ok"
    except Exception as e:
        return "ERR:" + str(e).split("\n")[0][:160]


KINDS = {   # double kind -> regex on A (one gate per line)
    "t_t": r"^(T|T_DAG) (\d+)\n\1 \2$",
    "t_diag_t": r"^(T|T_DAG) (\d+)\n(CZ \2 \d+|S \2|Z \2)\n\1 \2$",
    "cs_cs": r"^(CS|CS_DAG) (\d+ \d+)\n\1 \2$",
    "ch_ch": r"^CH (\d+ \d+)\nCH \1$",
    "ccz_ccz": r"^CCZ (\d+ \d+ \d+)\nCCZ \1$",
}


def main():
    N, path = int(sys.argv[1]), sys.argv[2]
    found = []
    for cl in (True, False):
        for seed in range(N):
            n, a, b = gen(seed, cl)
            if "ladder gap" in status(a) and status(b) == "ok":
                found.append({"seed": seed, "clifford": cl, "n": n, "A": a, "B": b})
    keep = {}
    for kind, rx in KINDS.items():
        for cl in (True, False):
            hits = sorted((c for c in found if c["clifford"] == cl and re.search(rx, c["A"], re.M)),
                          key=lambda c: (len(c["A"]), c["seed"]))
            for c in hits[:3]:
                keep.setdefault((c["clifford"], c["seed"]), dict(c, kinds=[]))["kinds"].append(kind)
    out = sorted(keep.values(), key=lambda c: (not c["clifford"], c["seed"]))
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(found)} pairs found, {len(out)} kept")


if __name__ == "__main__":
    main()
