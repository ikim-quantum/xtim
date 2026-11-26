"""LM-2: the lean route's sparse algebraic determinism check (xdem::nondeterministic_columns) against the dense form it
replaced (cpp/testapi/xdem_lean_dense_oracle.hpp, verbatim 9fb442f), verdict list and order.

`_xtim_testapi.xdem_nondet_sparse_dense(text)` builds the writer model and fault columns once and runs both forms on
them. Inputs: the corpus (tests/data, examples, xtim/_examples, benchmarks, the private corpus by opaque index; magic
circuits included), the lean fuzz families (tests/_xdem_lean_fuzz.py: free / mirror / mid, beta, boundary, mixed),
seeded random Clifford + magic circuits at word-boundary sizes (wires and columns 63/64/65/127/128/129), the lean-magic
probe's families at small sizes (surface / chain / GHZ / repetition + noisy T, one T per wire) and a wide-support family
whose column lists exceed the per-chunk bound (several chunks: both sides of the chunking)."""
import random
from pathlib import Path

import pytest
import stim

from xtim import _xtim_testapi as TA
from xtim.errors import XtimDemError
import _xdem_lean_fuzz as lf

HERE = Path(__file__).resolve().parent
DIRS = [HERE / "data", HERE.parent / "examples", HERE.parent / "xtim" / "_examples", HERE.parent / "benchmarks",
        HERE / "private"]
BOUNDARY = (63, 64, 65, 127, 128, 129)
MAGIC = ("T ", "T_DAG", "CCZ", "CS", "CH")


def check(text, tag):
    """The sparse and dense verdicts on `text` (None when the lean check is not reached)."""
    try:
        r = TA.xdem_nondet_sparse_dense(text)
    except XtimDemError:
        return None
    assert r["sparse"] == r["dense"], (tag, r["sparse"][:20], r["dense"][:20])
    assert r["chunks"] >= (1 if r["T"] > 0 else 0), (tag, r)
    return r


def sweep(items):
    done = non = cols = 0
    for tag, text in items:
        r = check(text, tag)
        if r is None:
            continue
        done += 1
        non += len(r["dense"])
        cols += r["T"]
    return done, non, cols


def test_corpus():
    paths = []
    for d in DIRS:
        if d.is_dir():
            paths += sorted(str(p) for p in d.rglob("*.stim"))
    done = magic = 0
    for i, p in enumerate(paths):
        text = Path(p).read_text()
        r = check(text, f"corpus#{i}")          # opaque index: the private corpus is never named
        if r is None:
            continue
        done += 1
        magic += any(g in text for g in MAGIC)
    assert done >= 40 and magic >= 30, (done, magic)


def test_lean_fuzz_families():
    fams = {"fast": lf.generate(1, 450), "beta": lf.generate_beta(12345, 200),
            "boundary": lf.generate_boundary(77, 300), "mixed": lf.generate_mixed(5, 200)}
    det_total = 0
    for name, items in fams.items():
        done, non, cols = sweep(items)
        assert done >= 100 and non >= 50, (name, done, non, cols)
        det_total += cols - non
    assert det_total >= 500, det_total


@pytest.mark.slow
def test_lean_fuzz_slow():
    done, non, cols = sweep(lf.generate(2, 3000))
    assert done >= 1000, done


def random_magic(rng, n, m):
    """n wires, m readout columns (detectors and observables), random Clifford + T / CS / CCZ with noise."""
    allq = " ".join(map(str, range(n)))
    L = [("RX " if rng.random() < 0.5 else "R ") + allq]
    for _ in range(rng.randint(n // 2, 2 * n)):
        r = rng.random()
        q = rng.sample(range(n), 3)
        if r < 0.15:
            L.append(f"{rng.choice(['T', 'T_DAG'])} {q[0]}")
        elif r < 0.2:
            L.append(f"{rng.choice(['CS', 'CS_DAG'])} {q[0]} {q[1]}")
        elif r < 0.22:
            L.append(f"CCZ {q[0]} {q[1]} {q[2]}")
        elif r < 0.5:
            L.append(f"{rng.choice(['H', 'S', 'S_DAG'])} {q[0]}")
        else:
            L.append(f"{rng.choice(['CX', 'CZ'])} {q[0]} {q[1]}")
        if rng.random() < 0.3:
            L.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1'])}(0.01) {q[0]}")
    L.append(f"{rng.choice(['MX', 'M'])} {allq}")
    nobs = max(1, m // 10)
    for _ in range(m - nobs):
        k = rng.randint(1, 3)
        L.append("DETECTOR " + " ".join(f"rec[-{rng.randint(1, n)}]" for _ in range(k)))
    for o in range(nobs):
        L.append(f"OBSERVABLE_INCLUDE({o}) rec[-{rng.randint(1, n)}]")
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("size", BOUNDARY)
def test_word_boundary_sizes(size):
    rng = random.Random(size)
    done = 0
    for i in range(6):
        n, m = (size, rng.choice(BOUNDARY)) if i % 2 == 0 else (rng.choice((3, 5, 17)), size)
        r = check(random_magic(rng, n, m), f"random n={n} m={m} #{i}")
        if r is not None:
            assert r["T"] == m
            done += 1
    assert done >= 3, done


def test_random_small():
    rng = random.Random(20261123)
    done = 0
    for i in range(300):
        r = check(random_magic(rng, rng.randint(3, 9), rng.randint(1, 12)), f"random small #{i}")
        done += r is not None
    assert done >= 150, done


# The lean-magic probe's families (stage2/lean-magic/lwork.py), small sizes.
def _surf(d):
    return str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def _rep(d):
    return str(stim.Circuit.generated("repetition_code:memory", distance=d, rounds=2,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def _tpair(a, b):
    return f"R {a} {b}\nH {a}\nX_ERROR(0.001) {a}\nT {a}\nCX {a} {b}\n", f"MX {a}\nM {b}\n"


def surf_noisy_t(d):
    c = _surf(d)
    nq = stim.Circuit(c).num_qubits
    pre, post = _tpair(nq, nq + 1)
    return pre + c + "\n" + post


def chain_noisy_t(n):
    pre, post = _tpair(n, n + 1)
    return pre + "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(n)) + post


def ghz_noisy_t(n):
    L = ["R " + " ".join(map(str, range(n))), "H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)]
    L += ["X_ERROR(0.001) 0\nH 0\nT 0\nH 0", "X_ERROR(0.01) " + " ".join(map(str, range(n))),
          "M " + " ".join(map(str, range(n)))]
    L += [f"DETECTOR rec[-{n - i}] rec[-{n - i - 1}]" for i in range(1, n - 1)]
    return "\n".join(L) + "\n"


def one_t_per_wire(n):
    return "".join(f"R {q}\nX_ERROR(0.01) {q}\nT {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(n))


def rep_two_noisy_t(d):
    c = _rep(d)
    nq = stim.Circuit(c).num_qubits
    p1, q1 = _tpair(nq, nq + 1)
    p2, q2 = _tpair(nq + 2, nq + 3)
    return p1 + p2 + c + "\n" + q1 + q2


@pytest.mark.parametrize("tag,text", [
    *[(f"surface+T d{d}", surf_noisy_t(d)) for d in (3, 5)],
    *[(f"chain+T {n}", chain_noisy_t(n)) for n in (63, 64, 65, 300)],
    *[(f"ghz+T {n}", ghz_noisy_t(n)) for n in (63, 64, 65, 200)],
    *[(f"one T per wire {n}", one_t_per_wire(n)) for n in (63, 64, 65, 200)],
    *[(f"rep+2T d{d}", rep_two_noisy_t(d)) for d in (5, 33)],
])
def test_lean_magic_families(tag, text):
    r = check(text, tag)
    assert r is not None, tag


def wide(n, seed):
    """Dense start supports: |+>^n, a T on every wire (n axes, every wire a pivot), a random CX network, every wire
    read in Z (its pull-back is a dense X support at the start), a detector per read."""
    rng = random.Random(seed)
    allq = " ".join(map(str, range(n)))
    L = ["RX " + allq, "X_ERROR(0.01) 0", "T " + allq]
    for _ in range(6 * n):
        a, b = rng.sample(range(n), 2)
        L.append(f"CX {a} {b}")
    L.append("M " + allq)
    L += [f"DETECTOR rec[-{k}]" for k in range(1, n + 1)]
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("n", (129, 200, 300))
def test_wide_supports_chunked(n):
    """The per-chunk bound (x_start's words) splits the columns into several chunks: chunk boundaries are exercised."""
    r = check(wide(n, n), f"wide {n}")
    assert r is not None
    assert r["x_nnz"] * 32 > r["N"] * r["T"], r         # dense supports (above the 1/32 density of the bound)
    assert r["chunks"] >= 2, r


def test_stats_sparse_family():
    """On a sparse family one chunk holds every column and the lists are the supports (no dense blow-up)."""
    r = check(chain_noisy_t(300), "chain+T 300")
    assert r["chunks"] == 1 and r["x_nnz"] <= 4 * r["T"], r
