"""Compile stage (2026-10-24, OD-2026-10-24-compile): the oracles behind the construction-speed changes.

* F-G: eliminate_hadamards' frame-push table is built by an inverted search; it must equal the brute-force search
  (try_push, kept as the oracle) on all 2 x 24 x 24 inputs.
* F-F: the propagation table's walk skips gates whose targets all lie outside the atom's support; every atom must
  equal the unskipped propagate_atom (and every PPR retry the unskipped gate-by-gate PPR loop), on the corpus and on
  random circuits over every gate kind (CH included: it is never skipped).
* F-D: the early-stopping class check answers as the full table does, and returns the full table when in class.
* F-A: the table built without the dense PPR tableaux equals the default one minus the tableaux.
* czrows: the sparse-CZ tables (the twirl sampler's table, normalize's class_table) hold every atom's CZ layer as its
  pair list with the dense rows freed, and rehydrate to the dense tables bit for bit; sparsify_cz refuses (engine-bug
  logic_error) a dense layer its pair list does not reproduce, before freeing it (F1).
"""
import glob
import os
import random

import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sorted(glob.glob(os.path.join(HERE, "data", "*.stim")) +
                glob.glob(os.path.join(HERE, "..", "examples", "*.stim")))


def _check(res):
    assert res["atom_mismatches"] == 0, res
    assert res["ppr_mismatches"] == 0, res
    assert res["class_check_agrees"], res
    assert res["sparse_mismatches"] == 0, res


def test_push_table_equals_brute_force_search():
    assert TA._push_table_oracle_mismatches() == 0


@pytest.mark.parametrize("path", CORPUS, ids=[os.path.basename(p) for p in CORPUS])
def test_propagation_walk_matches_unskipped_corpus(path):
    text = open(path).read()
    _check(TA._propagation_walk_oracle(text, normalized=False))
    try:
        res = TA._propagation_walk_oracle(text, normalized=True)
    except ValueError:                       # a circuit the twirl normalization refuses (feedback kept)
        return
    _check(res)


_ONE = ["H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG"]
_TWO = ["CX", "CZ", "CS", "CH"]


def _random_circuit(rng, n, depth):
    lines = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.35:
            lines.append(f"{rng.choice(_ONE)} {rng.randrange(n)}")
        elif r < 0.65:
            a, b = rng.sample(range(n), 2)
            g = rng.choice(_TWO) if rng.random() < 0.5 else rng.choice(["CX", "CZ"])
            lines.append(f"{g} {a} {b}")
        elif r < 0.72:
            a, b, c = rng.sample(range(n), 3)
            lines.append(f"CCZ {a} {b} {c}")
        elif r < 0.85:
            lines.append(f"DEPOLARIZE1(0.01) {rng.randrange(n)}")
        elif r < 0.93:
            a, b = rng.sample(range(n), 2)
            lines.append(f"DEPOLARIZE2(0.01) {a} {b}")
        else:
            lines.append(f"{rng.choice(['X_ERROR', 'Z_ERROR'])}(0.01) {rng.randrange(n)}")
    lines.append("M " + " ".join(str(q) for q in range(n)))
    return "\n".join(lines) + "\n"


def test_propagation_walk_matches_unskipped_random():
    rng = random.Random(20261024)
    tot = {"atoms": 0, "rejected_atoms": 0, "ppr_accepted": 0, "in_class": 0, "out": 0, "sparse_pairs": 0}
    for i in range(300):
        n = rng.choice([3, 4, 6, 10, 24])
        res = TA._propagation_walk_oracle(_random_circuit(rng, n, rng.randrange(10, 80)), normalized=False)
        _check(res)
        tot["atoms"] += res["atoms"]
        tot["rejected_atoms"] += res["rejected_atoms"]
        tot["in_class"] += res["in_class"]
        tot["out"] += not res["in_class"]
        tot["ppr_accepted"] += res["has_general"]
        tot["sparse_pairs"] += res["sparse_pairs"]
    # the corpus of random circuits exercises both branches of every check
    assert tot["atoms"] > 5000 and tot["rejected_atoms"] > 500 and tot["ppr_accepted"] > 10
    assert tot["in_class"] > 10 and tot["out"] > 10
    assert tot["sparse_pairs"] > 1000          # the sparse-CZ check sees real CZ layers


def test_sparsify_cz_check_has_teeth():
    """F1: a corrupted dense CZ layer (lower-only, diagonal or padding bit) makes sparsify_cz throw; clean ones pass."""
    tot = {"atoms": 0, "cases": 0, "thrown": 0, "clean_ok": 0, "dense_reads": 0, "dense_thrown": 0}
    for path in CORPUS:
        res = TA._sparsify_cz_teeth(open(path).read(), 20)
        for k in tot:
            tot[k] += res[k]
    assert tot["atoms"] > 50 and tot["clean_ok"] == tot["atoms"], tot
    assert tot["cases"] > 2 * tot["atoms"] and tot["thrown"] == tot["cases"], tot
    # threads L1: cz_pairs() on a dense atom throws (engine-bug logic_error), never an empty pair list
    assert tot["dense_reads"] > 50 and tot["dense_thrown"] == tot["dense_reads"], tot
