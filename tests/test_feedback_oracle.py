"""M1 (magic lane, 2026-11-21): the feedback plan's two speed changes, each held to the old path kept as its oracle.

* H6b: feedback_crosses_nonclifford is one backward reachability pass; it must equal the per-feedback forward
  lightcone scan (feedback_crosses_nonclifford_bruteforce) on every input.
* H6a: build_feedback_plan propagates the probes alone (the probe circuit's user noise is dropped after deferral);
  the plan must equal build_feedback_plan_full_table (the table over every noise location, as before) field for
  field — ops, reads, refusal kind / op / message and the deferred-stream reject index — under both deferrals.
"""
import glob
import os
import random

import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sorted(glob.glob(os.path.join(HERE, "data", "*.stim")) +
                glob.glob(os.path.join(HERE, "..", "examples", "**", "*.stim"), recursive=True))


def _check(res):
    assert not res["cross_mismatch"], res
    assert res["plan_mismatches"] == 0, res


@pytest.mark.parametrize("path", CORPUS, ids=[os.path.relpath(p, HERE) for p in CORPUS])
def test_feedback_oracle_corpus(path):
    try:
        res = TA._feedback_oracle(open(path).read())
    except (ValueError, RuntimeError):     # a circuit the parser / coherentizer refuses
        return
    _check(res)


# A non-crossing feedback (X on 1) whose Pauli reaches the control of a crossing, coherentized one (onto 2, then T):
# the plan refuses it, naming a deferred-stream index that lies after user noise (the index map is exercised).
_REFUSED = """R 0 1 2
H 0
DEPOLARIZE1(0.01) 0 1 2
M 0
X_ERROR(0.01) 1
CX rec[-1] 1
DEPOLARIZE1(0.01) 0 1 2
M 1
Z_ERROR(0.01) 2
CX rec[-1] 2
DEPOLARIZE1(0.01) 2
T 2
M 2
"""


def test_feedback_oracle_refused_plan_after_noise():
    res = TA._feedback_oracle(_REFUSED)
    assert res["plan_mismatches"] == 0, res
    assert res["rejected"] == 2 and res["crossing"] == 1 and not res["cross_mismatch"], res


_ONE = ["H", "S", "S_DAG", "X", "Z", "T", "T_DAG"]


def _random_feedback_circuit(rng, n, depth):
    lines, meas = [], 0
    for _ in range(depth):
        r = rng.random()
        if r < 0.25:
            lines.append(f"{rng.choice(_ONE)} {rng.randrange(n)}")
        elif r < 0.45:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['CX', 'CZ', 'CX', 'CZ', 'CS'])} {a} {b}")
        elif r < 0.48 and n >= 3:
            a, b, c = rng.sample(range(n), 3)
            lines.append(f"CCZ {a} {b} {c}")
        elif r < 0.62:
            lines.append(f"{rng.choice(['DEPOLARIZE1(0.01)', 'X_ERROR(0.01)', 'Z_ERROR(0.02)'])} {rng.randrange(n)}")
        elif r < 0.68:
            a, b = rng.sample(range(n), 2)
            lines.append(f"DEPOLARIZE2(0.01) {a} {b}")
        elif r < 0.80:
            lines.append(f"{rng.choice(['M', 'MX', 'M(0.01)'])} {rng.randrange(n)}")
            meas += 1
        elif r < 0.84:
            lines.append(f"R {rng.randrange(n)}")
        elif meas:
            lines.append(f"{rng.choice(['CX', 'CY', 'CZ'])} rec[-{rng.randrange(1, min(meas, 4) + 1)}] "
                         f"{rng.randrange(n)}")
    lines.append("M " + " ".join(str(q) for q in range(n)))
    return "\n".join(lines) + "\n"


def test_feedback_oracle_random():
    rng = random.Random(20261121)
    tot = {"circuits": 0, "feedbacks": 0, "crossing": 0, "plans": 0, "rejected": 0, "ops": 0, "refused": 0}
    for _ in range(600):
        text = _random_feedback_circuit(rng, rng.choice([2, 3, 5, 8]), rng.randrange(10, 70))
        try:
            res = TA._feedback_oracle(text)
        except (ValueError, RuntimeError):
            tot["refused"] += 1
            continue
        _check(res)
        tot["circuits"] += 1
        for k in ("feedbacks", "crossing", "plans", "rejected", "ops"):
            tot[k] += res[k]
    # both crossing answers, accepted plans with ops and refused plans all occur
    assert tot["circuits"] > 400 and tot["feedbacks"] > 1000, tot
    assert 100 < tot["crossing"] < tot["feedbacks"] - 100, tot
    assert tot["ops"] > 500 and tot["rejected"] > 5, tot
