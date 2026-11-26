"""Deferral liveness in one backward pass (Stage 1 T2) vs the per-Measure forward scan it replaced.

`defer_measurements` decides, for every Measure, whether its wire is used again (Gate/Measure/Observable)
before its next Reset. The production pass computes this in O(L); the old O(M*L) forward scan is kept as
the test oracle (`defer_measurements_bruteforce_liveness`). `_xtim_testapi._defer_oracle_diff` runs both on the
parsed circuit and compares the outcome: the refusal (error type and message) or the deferred stream
instruction by instruction, the deferred qubit count and the whole DeferralMap.
"""
import glob
import os
import random

import pytest

from xtim import _xtim_testapi as TA
from _private import PRIVATE_DIR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PUBLIC = sorted(glob.glob(os.path.join(_ROOT, "examples", "*.stim")) +
                 glob.glob(os.path.join(_ROOT, "tests", "data", "*.stim")))
_PRIVATE = sorted(glob.glob(os.path.join(str(PRIVATE_DIR), "*.stim")))


def _check(text):
    for diagonal_copy in (False, True):
        for strip in (False, True):
            diff, _n, _t, _t_old, _r = TA._defer_oracle_diff(text, diagonal_copy=diagonal_copy, strip=strip)
            assert diff == "", (diff, diagonal_copy, strip)


@pytest.mark.parametrize("path", _PUBLIC, ids=[os.path.relpath(p, _ROOT) for p in _PUBLIC])
def test_corpus_matches_forward_scan(path):
    _check(open(path).read())


@pytest.mark.parametrize("i", range(len(_PRIVATE)), ids=[f"private-{i}" for i in range(len(_PRIVATE))])
def test_private_corpus_matches_forward_scan(i):
    _check(open(_PRIVATE[i]).read())


def _random_circuit(rng):
    n = rng.randint(1, 5)
    q = lambda: rng.randrange(n)
    nmeas = [0]
    lines = ["INPUT_BITS d 2"] if rng.random() < 0.3 else []

    def body(depth, length):
        out = []
        for _ in range(length):
            r = rng.random()
            if r < 0.14:
                out.append(f"{rng.choice(['R', 'RX', 'RY'])} {q()}")
            elif r < 0.30:
                b = rng.choice(["M", "MX", "MY"])
                if rng.random() < 0.15:
                    b += "(0.125)"
                out.append(f"{b} {'!' if rng.random() < 0.1 else ''}{q()}")
                nmeas[0] += 1
            elif r < 0.38:
                out.append(f"{rng.choice(['MR', 'MRX', 'MRY'])} {q()}")
                nmeas[0] += 1
            elif r < 0.50:
                out.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1'])}(0.01) {q()}")
            elif r < 0.53 and n >= 2:
                a, c = rng.sample(range(n), 2)
                out.append(f"DEPOLARIZE2(0.01) {a} {c}")
            elif r < 0.64:
                out.append(f"{rng.choice(['H', 'S', 'X', 'SQRT_X', 'T'])} {q()}")
            elif r < 0.71 and n >= 2:
                a, c = rng.sample(range(n), 2)
                out.append(f"{rng.choice(['CX', 'CZ'])} {a} {c}")
            elif r < 0.75 and n >= 2:
                a, c = rng.sample(range(n), 2)
                out.append(f"MPP X{a}*Z{c}")
                nmeas[0] += 1
            elif r < 0.79 and nmeas[0] > 0:
                out.append(f"{rng.choice(['CX', 'CZ'])} rec[-1] {q()}")     # feedback
            elif r < 0.83 and nmeas[0] > 0:
                out.append("DETECTOR rec[-1]")
            elif r < 0.86 and nmeas[0] > 0:
                out.append("OBSERVABLE_INCLUDE(0) rec[-1]")
            elif r < 0.89 and depth == 0:          # labels must be unique: never inside a REPEAT
                out.append(f"PAULI_EXPECTATION({rng.randrange(1 << 30)}) {rng.choice('XYZ')}{q()}")
            elif r < 0.93 and depth < 2:
                out.append(f"REPEAT {rng.randint(1, 3)} {{")
                out += body(depth + 1, rng.randint(1, 6))
                out.append("}")
            elif r < 0.96 and depth < 2 and lines and lines[0].startswith("INPUT_BITS"):
                out.append(f"IF {'!' if rng.random() < 0.3 else ''}d[{rng.randrange(2)}] {{")
                out += body(depth + 1, rng.randint(0, 4))
                out.append("}")
            else:
                out.append("TICK")
        return out

    lines += body(0, rng.randint(1, 30))
    return "\n".join(lines) + "\n"


def test_random_circuits_match_forward_scan():
    rng = random.Random(20261121)
    checked = refused = 0
    for _ in range(2500):
        text = _random_circuit(rng)
        try:
            TA._defer_oracle_diff(text)
        except ValueError:          # the generator's rare unparsable draw
            continue
        _check(text)
        checked += 1
        refused += TA._defer_oracle_diff(text)[1] == -1
    assert checked >= 2000
    assert 100 <= refused <= checked - 1000     # both refusals and deferred streams are exercised


def test_refusal_precedence_first_marker_wins():
    # Feedback before an IF block -> the feedback error; IF block before feedback -> the IF refusal.
    fb_first = "INPUT_BITS d 1\nR 0\nM 0\nCX rec[-1] 0\nIF d[0] {\nX 0\n}\nH 0\nM 0\n"
    if_first = "INPUT_BITS d 1\nR 0\nM 0\nIF d[0] {\nX 0\n}\nCX rec[-1] 0\nH 0\nM 0\n"
    for text, needle in ((fb_first, "runtime_error: defer_measurements: classically-controlled"),
                         (if_first, "invalid_argument: the circuit has IF blocks")):
        diff, n, _t, _t_old, refusal = TA._defer_oracle_diff(text)
        assert diff == "" and n == -1 and refusal.startswith(needle), refusal


def test_abandoned_and_live_measurements():
    # Wire 0: M then noise then R (abandoned: retired, noise lands on the dead wire).
    # Wire 1: M then H (live: CNOT copy onto a fresh ancilla). The final M 0 1 are both abandoned (retired).
    # n' = 2 + retire(0) + R(0) + copy(1) + 2 retires = 7.
    diff, n, _t, _t_old, _r = TA._defer_oracle_diff("M 0 1\nX_ERROR(0.1) 0\nR 0\nH 1\nM 0 1\n")
    assert diff == "" and n == 7
