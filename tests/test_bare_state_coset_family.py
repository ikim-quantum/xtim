"""The coset-family bare-state construction is BYTE-IDENTICAL to the 3.1.7 per-branch one.

3.1.8 builds the chi = 2^r bare state without materialising the chi rays: the frame syndromes are
linear in the branch index and the per-branch gauge is a quadratic character, so both are read off
the rays of weight <= 2 (CanonicalStabSum::from_coset_family). The 3.1.7 construction (every ray
projected, disentangled and inner-producted) is kept in the engine as the oracle
(AssemblePath::ray_oracle); `_xtim._bare_state_oracle_diff` builds both and compares frame, anchor
signs, anchor state, free set, branch sign patterns and coefficient BYTES.
"""
import glob
import pathlib
import random

import pytest

import xtim
from xtim import _xtim

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = sorted(
    glob.glob(str(ROOT / "examples" / "*.stim"))
    + glob.glob(str(ROOT / "tests" / "data" / "*.stim"))
    + glob.glob(str(ROOT / "benchmarks" / "*.stim"))
)
CORPUS = [p for p in CORPUS if "merlin_switch3" not in p]   # covered (slowly) in the chi-guard test


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: pathlib.Path(p).name)
def test_corpus_byte_identical(path):
    diff, chi = _xtim._bare_state_oracle_diff(pathlib.Path(path).read_text())
    assert diff == "", f"{path}: first difference at {diff} (chi = {chi})"


@pytest.mark.parametrize("n", range(1, 11))
def test_all_t_byte_identical(n):
    qs = " ".join(map(str, range(n)))
    diff, chi = _xtim._bare_state_oracle_diff(f"RX {qs}\nT {qs}\nMX {qs}\n")
    assert chi == 2 ** n
    assert diff == "", diff


def _random_magic_circuit(rng: random.Random) -> str:
    """A random in-class magic circuit: |+>/|0> preps, Clifford mixing, one commuting magic layer
    of T / T_DAG / CCZ on CX-conjugated parities, more Cliffords, measure everything."""
    n = rng.randint(3, 9)
    lines = []
    plus = [q for q in range(n) if rng.random() < 0.6]
    zero = [q for q in range(n) if q not in plus]
    if plus:
        lines.append("RX " + " ".join(map(str, plus)))
    if zero:
        lines.append("R " + " ".join(map(str, zero)))
    for _ in range(rng.randint(0, 2 * n)):
        a, b = rng.sample(range(n), 2)
        lines.append(f"CX {a} {b}")
    for _ in range(rng.randint(1, 2 * n)):
        r = rng.random()
        if r < 0.45:
            lines.append(f"T {rng.randrange(n)}")
        elif r < 0.8:
            lines.append(f"T_DAG {rng.randrange(n)}")
        elif r < 0.9 and n >= 3:
            a, b, c = rng.sample(range(n), 3)
            lines.append(f"CCZ {a} {b} {c}")
        else:
            lines.append(f"S {rng.randrange(n)}")
    for _ in range(rng.randint(0, n)):
        a, b = rng.sample(range(n), 2)
        lines.append(f"CX {a} {b}" if rng.random() < 0.5 else f"CZ {a} {b}")
    for q in range(n):
        if rng.random() < 0.5:
            lines.append(f"H {q}")
    lines.append("M " + " ".join(map(str, range(n))))
    return "\n".join(lines) + "\n"


def test_random_magic_circuits_byte_identical():
    rng = random.Random(318)
    built = big = 0
    for _ in range(150):
        text = _random_magic_circuit(rng)
        try:
            diff, chi = _xtim._bare_state_oracle_diff(text)
        except (ValueError, RuntimeError):
            continue                                   # out of class: nothing to compare
        assert diff == "", f"first difference at {diff} (chi = {chi}) on\n{text}"
        built += 1
        big += chi >= 8
    assert built >= 100 and big >= 30, (built, big)
