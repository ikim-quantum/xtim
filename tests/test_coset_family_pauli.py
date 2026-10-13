"""The coset-family construction read off the Pauli algebra is BYTE-IDENTICAL to the 3.1.8 one.

3.1.8 (CanonicalStabSum::from_coset_family_rays, kept in the engine as the ORACLE) materialised the
rays of weight <= 2, disentangled each unit ray for its frame syndrome and took an exact inner
product per weight-1/2 gauge.  3.1.9 never builds those rays: ray_g = M_g ray_0 with M_g a product of
stabilizers s_i of the support psi (s_i anticommutes with the i-th projected Z-string alone), so a
unit ray's syndrome is the commutation of the frame generators with s_i, and a gauge is the anchor
expectation of the Pauli product M_g . D_sigma(g).  `_xtim._ray_syndrome_oracle_diff` builds the
bare state both ways and compares every unit-ray syndrome, the re-based syndrome columns, every
weight-1/2 gauge exponent, b_ij, and then the whole CanonicalStabSum (frame, eps, anchor, free,
branch sigma, coefficient BYTES).
"""
import collections
import glob
import pathlib
import random

import pytest

from xtim import _xtim

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = sorted(
    glob.glob(str(ROOT / "examples" / "*.stim"))
    + glob.glob(str(ROOT / "tests" / "data" / "*.stim"))
    + glob.glob(str(ROOT / "benchmarks" / "*.stim"))
)


def _check(text):
    diff, chi, r = _xtim._ray_syndrome_oracle_diff(text)
    assert diff == "", f"first difference at {diff} (chi = {chi}, r = {r})\n{text if len(text) < 4000 else ''}"
    return chi, r


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: pathlib.Path(p).name)
def test_corpus_byte_identical(path):
    try:
        _check(pathlib.Path(path).read_text())
    except ValueError:
        pytest.skip("normalize refuses this circuit")


@pytest.mark.parametrize("n", range(1, 13))
def test_all_t_byte_identical(n):
    qs = " ".join(map(str, range(n)))
    chi, r = _check(f"RX {qs}\nT {qs}\nMX {qs}\n")
    assert r == n and chi == 2 ** n


MAGIC1 = ["T", "T_DAG"]
CLIFF1 = ["H", "S", "S_DAG", "X", "Z", "SQRT_X"]
CLIFF2 = ["CX", "CZ", "SWAP"]


def _random_in_class(rng: random.Random) -> str:
    """|+>/|0> preps, Clifford mixing, a magic layer (T/T_DAG/CS/CS_DAG/CCZ/CH; repeats, magic on
    |0> wires and CX-correlated parities make the magic columns rank-deficient), more Cliffords,
    measure everything.  The magic count is drawn so that r spans 0..12 and beyond."""
    n = rng.randint(2, 16)
    lines = []
    plus = [q for q in range(n) if rng.random() < 0.7]
    zero = [q for q in range(n) if q not in plus]
    if plus:
        lines.append("RX " + " ".join(map(str, plus)))
    if zero:
        lines.append("R " + " ".join(map(str, zero)))
    for _ in range(rng.randint(0, 2 * n)):
        if rng.random() < 0.6:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(CLIFF2)} {a} {b}")
        else:
            lines.append(f"{rng.choice(['S', 'S_DAG', 'Z', 'X'])} {rng.randrange(n)}")
    target = rng.randint(0, 24)
    for _ in range(target):
        u = rng.random()
        if u < 0.7:
            lines.append(f"{rng.choice(MAGIC1)} {rng.randrange(n)}")
        elif u < 0.85:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['CS', 'CS_DAG'])} {a} {b}")
        elif u < 0.97 and n >= 3:
            lines.append("CCZ " + " ".join(map(str, rng.sample(range(n), 3))))
        else:                                        # CH: often non-commuting (refused) — kept rare
            a, b = rng.sample(range(n), 2)
            lines.append(f"CH {a} {b}")
        if rng.random() < 0.1:                       # a repeat: T T = S (an even column)
            lines.append(lines[-1])
    for _ in range(rng.randint(0, n)):
        if rng.random() < 0.5:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(CLIFF2)} {a} {b}")
        else:
            lines.append(f"{rng.choice(CLIFF1)} {rng.randrange(n)}")
    lines.append("M " + " ".join(map(str, range(n))))
    return "\n".join(lines) + "\n"


def _random_high_rank(rng: random.Random) -> str:
    """r ~ 6..13: an all-|+> register mixed by CX/CZ/SWAP/S, one T or T_DAG per qubit (independent
    parities), a few CS/CCZ on top, Clifford tail, measure."""
    n = rng.randint(6, 14)
    lines = ["RX " + " ".join(map(str, range(n)))]
    for _ in range(rng.randint(0, 2 * n)):
        a, b = rng.sample(range(n), 2)
        lines.append(f"{rng.choice(CLIFF2 + ['CX'])} {a} {b}" if rng.random() < 0.8 else f"S {a}")
    for q in range(n):
        if rng.random() < 0.9:
            lines.append(f"{rng.choice(MAGIC1)} {q}")
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.6:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['CS', 'CS_DAG'])} {a} {b}")
        else:
            lines.append("CCZ " + " ".join(map(str, rng.sample(range(n), 3))))
    for _ in range(rng.randint(0, n)):
        a, b = rng.sample(range(n), 2)
        lines.append(f"{rng.choice(CLIFF2)} {a} {b}" if rng.random() < 0.5 else f"{rng.choice(CLIFF1)} {a}")
    lines.append("M " + " ".join(map(str, range(n))))
    return "\n".join(lines) + "\n"


def test_random_circuits_byte_identical():
    rng = random.Random(3191)
    ranks = collections.Counter()
    for k in range(1500):
        text = _random_in_class(rng) if k % 3 else _random_high_rank(rng)
        try:
            _, r = _check(text)
        except ValueError:
            continue                                  # out of class (normalize refuses)
        if r >= 0:
            ranks[r] += 1
    assert sum(ranks.values()) >= 800, ranks
    assert all(ranks[r] >= 10 for r in range(0, 13)), sorted(ranks.items())

