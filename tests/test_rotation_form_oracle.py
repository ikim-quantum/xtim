"""build_pauli_rotation_form's conjugation into the |0> frame is BYTE-IDENTICAL to the 3.1.8 walk.

Every magic generator (T: Z_q; CS: Z_a, Z_b, Z_aZ_b; CCZ: 7 Z-strings; CH: Y_t twice) is mapped to
P = C^-1 . Q . C, C the Clifford list accumulated so far. 3.1.8 walked the WHOLE list backwards per
generator (one n-qubit `conjugate` per gate); that walk is kept in the engine as the ORACLE.
`_xtim._rotation_form_oracle_diff` compares the production image with the walk's term by term (bits
AND phase) and the two whole forms (terms, coeffs, Clifford list, global-phase bytes, reject fields).
"""
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


def _check(text, normalize=True):
    diff, checked, frame, nterms = _xtim._rotation_form_oracle_diff(text, normalize=normalize)
    assert diff == "", f"first difference: {diff}\n{text if len(text) < 4000 else ''}"
    assert frame == checked, f"{checked - frame} of {checked} generators fell back to the walk"
    return checked


@pytest.mark.parametrize("normalize", [True, False], ids=["normalized", "raw"])
@pytest.mark.parametrize("path", CORPUS, ids=lambda p: pathlib.Path(p).name)
def test_corpus_byte_identical(path, normalize):
    try:
        _check(pathlib.Path(path).read_text(), normalize=normalize)
    except ValueError:
        pytest.skip("normalize refuses this circuit")


ONE_Q = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "H_YZ"]
TWO_Q = ["CX", "CZ", "CY", "SWAP", "ISWAP", "CXSWAP"]
MAGIC = ["T", "T_DAG", "CS", "CS_DAG", "CCZ", "CH"]
ARITY = {"CS": 2, "CS_DAG": 2, "CCZ": 3, "CH": 2}


def _random_circuit(rng: random.Random, n: int, length: int) -> str:
    lines = []
    for _ in range(length):
        r = rng.random()
        if r < 0.45:
            lines.append(f"{rng.choice(ONE_Q)} {rng.randrange(n)}")
        elif r < 0.8:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(TWO_Q)} {a} {b}")
        else:
            g = rng.choice(MAGIC)
            k = ARITY.get(g, 1)
            if k > n:
                continue
            lines.append(g + " " + " ".join(map(str, rng.sample(range(n), k))))
    lines.append("M " + " ".join(map(str, range(n))))
    return "\n".join(lines) + "\n"


def test_random_circuits_byte_identical():
    """Raw gate streams (no normalize): every Clifford kind, T/T_DAG/CS/CS_DAG/CCZ/CH interleaved
    with Cliffords AFTER the magic (non-commuting forms too — the walk runs before the decision)."""
    rng = random.Random(319)
    total = 0
    for seed in range(1500):
        n = rng.randint(2, 40) if seed % 3 else rng.randint(3, 8)
        total += _check(_random_circuit(rng, n, rng.randint(1, 6 * n + 10)), normalize=False)
    assert total > 20000, total


def test_random_circuits_normalized_byte_identical():
    """The production input: normalize_for_twirl first (preps, Cliffords, one magic layer)."""
    rng = random.Random(3190)
    total = checked_circuits = 0
    for _ in range(300):
        n = rng.randint(3, 24)
        body = _random_circuit(rng, n, rng.randint(1, 4 * n))
        text = "RX " + " ".join(map(str, range(n))) + "\n" + body
        try:
            total += _check(text, normalize=True)
            checked_circuits += 1
        except ValueError:
            continue
    assert checked_circuits > 200 and total > 1000, (checked_circuits, total)


@pytest.mark.parametrize("pre", ["", "H", "S", "H S", "S H", "S_DAG H", "H S_DAG", "SQRT_X"])
@pytest.mark.parametrize("cliff", ["X", "Y", "Z", "H", "S", "S_DAG", "CX", "CZ"])
def test_every_clifford_kind(cliff, pre):
    """Each conj_generators / two-qubit case, with every basis prefix on both qubits, before
    every magic kind on both orientations."""
    lines = []
    for g in pre.split():
        lines.append(f"{g} 0 1")
    for a, b in [(0, 1), (1, 0)]:
        lines.append(f"{cliff} {a}" if cliff not in ("CX", "CZ") else f"{cliff} {a} {b}")
        for m in ["T 0", "T_DAG 1", "CS 0 1", "CH 0 1", "CH 1 0", "CCZ 0 1 2"]:
            lines.append(m)
        lines.append(f"{cliff} {b}" if cliff not in ("CX", "CZ") else f"{cliff} {b} {a}")
    assert _check("\n".join(lines) + "\nM 0 1 2\n", normalize=False) > 0
