"""S2-T6: the sampler's record columns and the DEM's twirl-law columns against the pre-S2-T6 provenance oracle.

`_xtim_testapi._record_columns_oracle(text)` compiles `text` (compile_family_sampler, selfcheck 0) and holds every
record column to the dense combination-tracking echelon over the certified generators (ProvenanceRREF, the
computation production used before S2-T6): a channel's generator mask and ref equal the oracle's decomposition of its
operator M and G.reduce's sign, the product of the mask's generators is M up to that sign, every extra reduces to
LOGICAL / ANTI; RecordColumns (fault_record_law, the DEM's tw_get route) on the same generators is held to the same
oracle. Inputs: the corpus (tests/data, examples, the private corpus by opaque index), chain-300, surface codes and
seeded random Clifford + T circuits (detectors, observables, MPP, feedback).
"""
import os
import random
from pathlib import Path

import pytest
import stim

from xtim import _xtim_testapi as TA

HERE = Path(__file__).resolve().parent
DIRS = [HERE / "data", HERE.parent / "examples", HERE / "private"]


def corpus():
    out = []
    for d in DIRS:
        if d.is_dir():
            out += sorted(str(p) for p in d.rglob("*.stim"))
    return out


def check(text, tag):
    r = TA._record_columns_oracle(text)
    if r is None:
        return None
    assert r["mismatches"] == [], (tag, r["mismatches"])
    assert r["rank"] == r["gens"], (tag, "dependent certified generators")
    assert r["record_columns"] == r["channels"] + r["extras"]
    return r


def test_corpus():
    paths = corpus()
    done = cols = magic = 0
    for i, p in enumerate(paths):
        text = Path(p).read_text()
        r = check(text, f"corpus#{i}")          # opaque index: the private corpus is never named
        if r is None:
            continue
        done += 1
        cols += r["channels"]
        if any(g in text for g in ("T ", "T_DAG", "CCZ", "CS", "CH")):
            magic += 1
    assert done >= 40 and magic >= 20 and cols > 1000, (done, magic, cols)


def test_chain_and_surface():
    chain = "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(300))
    assert check(chain, "chain-300")["channels"] == 300
    for d in (3, 5, 7):
        for kind in ("surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "color_code:memory_xyz"):
            c = stim.Circuit.generated(kind, distance=d, rounds=d, after_clifford_depolarization=0.001,
                                       before_measure_flip_probability=0.001, after_reset_flip_probability=0.001)
            r = check(str(c), f"{kind} d{d}")
            assert r is not None and r["channels"] == c.num_detectors + c.num_observables, (kind, d, r)


def random_circuit(seed):
    rnd = random.Random(seed)
    n = rnd.choice([1, 2, 3, 5, 8, 13, 63, 64, 65])
    act = list(range(n))
    L = ["R " + " ".join(map(str, act))]
    nmeas = nt = 0
    for _ in range(rnd.randint(4, 40)):
        u = rnd.random()
        a = rnd.choice(act)
        if u < 0.2:
            L.append(f"H {a}")
        elif u < 0.3:
            L.append(f"{rnd.choice(['S', 'S_DAG', 'X', 'Z', 'Y', 'SQRT_X'])} {a}")
        elif u < 0.5 and n > 1:
            b = rnd.choice([q for q in act if q != a])
            L.append(f"{rnd.choice(['CX', 'CZ', 'CY'])} {a} {b}")
        elif u < 0.58 and nt < 6:
            L.append(f"{rnd.choice(['T', 'T_DAG'])} {a}")
            nt += 1
        elif u < 0.65:
            L.append(f"{rnd.choice(['DEPOLARIZE1', 'X_ERROR', 'Z_ERROR'])}(0.01) {a}")
        elif u < 0.8:
            L.append(f"{rnd.choice(['M', 'MX', 'MY', 'MR'])} {a}")
            nmeas += 1
            if rnd.random() < 0.5:
                L.append("DETECTOR rec[-1]")
            if nmeas >= 2 and rnd.random() < 0.3:
                L.append("DETECTOR rec[-1] rec[-2]")
            if n > 1 and rnd.random() < 0.15:
                L.append(f"CX rec[-1] {rnd.choice([q for q in act if q != a])}")
        elif u < 0.88 and n > 1:
            b = rnd.choice([q for q in act if q != a])
            L.append(f"MPP {rnd.choice('XZ')}{a}*{rnd.choice('XZ')}{b}")
            nmeas += 1
            if rnd.random() < 0.5:
                L.append("DETECTOR rec[-1]")
        else:
            L.append("TICK")
    for a in act:
        L.append(f"M {a}")
    L.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    if n > 1:
        L.append("DETECTOR rec[-1] rec[-2]")
    return "\n".join(L) + "\n"


def test_random():
    done = cols = 0
    for seed in range(400):
        r = check(random_circuit(seed), f"random seed {seed}")
        if r is not None:
            done += 1
            cols += r["channels"] + r["extras"]
    assert done >= 150 and cols > 500, (done, cols)


def family(kind, n):
    """Chain-structured Clifford families from the S2-T6 reviews (GHZ chains, chain + fan-out, measured twice)."""
    allq = " ".join(map(str, range(n)))
    L = [f"R {allq}"]
    m = n // 2
    if kind == "ghz":
        L += ["H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)]
    elif kind == "ghzrev":
        L += [f"H {n - 1}"] + [f"CX {i} {i - 1}" for i in range(n - 1, 0, -1)]
    elif kind in ("chainfan", "chainfanrev"):
        chain = [f"CX {i} {i + 1}" for i in range(m - 1)]
        fan = [f"CX 0 {j}" for j in range(m, n)]
        L += ["H 0"] + (chain + fan if kind == "chainfan" else fan + chain)
    elif kind == "redund" or kind.startswith("redk"):
        L += ["H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)]
    elif kind == "star":
        L += ["H 0"] + [f"CX 0 {i}" for i in range(1, n)]
    elif kind.startswith("cfx") or kind.startswith("qfx3"):
        L += ["H 0"] + [f"CX {i} {i + 1}" for i in range(m - 1)] + [f"CX 0 {j}" for j in range(m, n)]
    elif kind.startswith("ghzl") or kind.startswith("rkf"):
        L += ["H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)]
    L.append(f"X_ERROR(0.01) {allq}")
    if kind.startswith("rkf"):     # fix 4 (rev4): measured twice, plus a third measurement of the first pct %
        f = n * int(kind[3:]) // 100
        T = 2 * n + f
        L += [f"M {allq}"] * 2 + ([f"M {' '.join(map(str, range(f)))}"] if f else [])
        L += [f"DETECTOR rec[-{T - i}] rec[-{T - (n + i)}]" for i in range(n)]
        L += [f"DETECTOR rec[-{T - (n + i)}] rec[-{T - (2 * n + i)}]" for i in range(f)]
        L += [f"DETECTOR rec[-{T - (n + i)}] rec[-{T - (n + i + 1)}]" for i in range(n - 1)]
        return "\n".join(L) + "\n"
    K = 2 if kind == "redund" else int(kind[4:]) if kind.startswith("redk") else 1
    L += [f"M {allq}"] * K
    for r in range(1, K):
        L += [f"DETECTOR rec[-{(K - r + 1) * n - i}] rec[-{(K - r) * n - i}]" for i in range(n)]
    L += [f"DETECTOR rec[-{n - i}] rec[-{n - i - 1}]" for i in range(n - 1)]
    if kind.startswith("cfx"):
        rr = random.Random(11)
        for _ in range(int(kind[3:]) * n):
            a, b = rr.randrange(m, n), rr.randrange(m, n)
            if a != b:
                L.append(f"DETECTOR rec[-{n - a}] rec[-{n - b}]")
    if kind.startswith("qfx3"):    # fix 4 (rev4 F1): weight-4 parities of distinct fan qubits
        rr = random.Random(13)
        for _ in range(int(kind[4:]) * n):
            q = [rr.randrange(m, n) for _ in range(4)]
            if len(set(q)) == 4:
                L.append("DETECTOR " + " ".join(f"rec[-{n - v}]" for v in q))
    if kind.startswith("ghzl"):    # fix 4 (rev4 F2): far pairs, the first tenth x the last tenth (long masks)
        rr = random.Random(17)
        t = max(1, n // 10)
        for _ in range(int(kind[4:]) * n):
            a, b = rr.randrange(t), n - 1 - rr.randrange(t)
            if a != b:
                L.append(f"DETECTOR rec[-{n - a}] rec[-{n - b}]")
    return "\n".join(L) + "\n"


FAMILIES = ["ghz", "ghzrev", "chainfan", "chainfanrev", "redund", "star", "cfx20", "redk3", "redk4", "redk6", "qfx32",
            "qfx35", "ghzl20", "rkf40"]


@pytest.mark.parametrize("kind", FAMILIES)
def test_chain_families(kind):
    for n in (3, 64, 65, 200, 401):
        r = check(family(kind, n), f"{kind} n={n}")
        assert r is not None and r["channels"] > 0, (kind, n)


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_chain_families_forced_layout(layout):
    """fix 4: the same oracle with both planes storage switches forced (XTIM_PLANES_LAYOUT is read once per process: a
    subprocess). Forced sparse takes the forward form on every group (and its time budget and storage bound), forced
    dense the prref route."""
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, %r); import test_s2t6_record_columns as T\n"
            "for k in T.FAMILIES:\n"
            "    for n in (3, 64, 65, 200, 401):\n"
            "        r = T.check(T.family(k, n), f'{k} n={n}')\n"
            "        assert r is not None and r['channels'] > 0, (k, n)\n"
            "print('ok')\n") % str(HERE)
    env = dict(os.environ, XTIM_PLANES_LAYOUT=layout)
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), (p.stdout[-2000:], p.stderr[-4000:])
