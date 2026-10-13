"""Forced faults: the exact FACTORED law (structured rows + independent fair-coin directions).

* No 2^r blow-up: random reads that feed no target cost nothing; random targets become fair-coin
  DIRECTIONS (u of them), never enumerated — rows stay 1 whatever r is.
* Structured magic rows: X-measured T states feeding DECISIONs give exact biased rows
  (cos^2(pi/8)), refused BY NAME beyond max_rows.
* The expanded view is the factored law expanded (probability split evenly, signs flipped by the
  directions) and refuses by name beyond max_rows; no raw records are returned.
* The factored law vs the independent dense statevector oracle aggregated by (detectors,
  observables, decisions) with conditional expectations, on circuits built to exercise
  directions that flip expectation signs together with magic rows.
"""
import math
import random

import numpy as np
import pytest

import xtim
from _forced_oracle import aggregate, branches as dense_branches, law_shifts

stim = pytest.importorskip("stim")


def _memory_with_ancillas(r, decisions):
    base = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=3,
                                  after_clifford_depolarization=0.001,
                                  before_measure_flip_probability=0.001)
    text = str(base.flattened()) + "\n"
    anc = list(range(base.num_qubits, base.num_qubits + r))
    text += "R " + " ".join(map(str, anc)) + "\nMX " + " ".join(map(str, anc)) + "\n"
    if decisions:
        text += "".join(f"DECISION({i}) rec[-{r - i}]\n" for i in range(r))
    return text


@pytest.mark.parametrize("r", [10, 40])
def test_random_reads_cost_nothing(r):
    """r fresh MX reads (2^(r+8) branches for a walk): one row; with a DECISION on each, r
    independent fair-coin directions, still one row."""
    for dec in (False, True):
        sim = xtim.Circuit(_memory_with_ancillas(r, dec)).compile_exact_branches()
        law = sim.exact_branches([(5, 0, "X")])
        assert law.num_rows == 1 and law.magic_parities == 0
        assert abs(law.rows_probabilities.sum() - 1) < 1e-12
        assert law.num_uniform == (r if dec else 0)
        if dec:
            # each direction flips exactly one decision; no detector / observable
            assert (law.uniform_decisions.sum(axis=1) == 1).all()
            assert not law.uniform_detectors.any() and not law.uniform_observables.any()
            if r <= 16:
                assert len(law) == 1 << r
                assert np.allclose(law.probabilities, 1.0 / (1 << r), atol=1e-15)
                assert len({tuple(x) for x in law.decisions.astype(int)}) == 1 << r
            else:
                with pytest.raises(xtim.ExpandedLawTooLarge, match="factored fields"):
                    law.probabilities


def test_magic_rows_exact_and_refused_by_name():
    """k |T> states read in X, each a DECISION: 2^k structured rows with the exact biased law,
    no directions; more rows than max_rows is refused by name."""
    k = 3
    text = "".join(f"H {q}\nT {q}\n" for q in range(k)) + "MX " + " ".join(map(str, range(k))) + "\n"
    text += "".join(f"DECISION({q}) rec[-{k - q}]\n" for q in range(k))
    sim = xtim.Circuit(text).compile_exact_branches()
    law = sim.exact_branches([])
    assert law.num_rows == 1 << k and law.num_uniform == 0 and law.magic_parities == k
    p0 = math.cos(math.pi / 8) ** 2
    for i in range(law.num_rows):
        ones = int(law.rows_decisions[i].sum())
        assert abs(law.rows_probabilities[i] - p0 ** (k - ones) * (1 - p0) ** ones) < 1e-12
    with pytest.raises(ValueError, match="max_rows"):
        sim.exact_branches([], max_rows=4)


def test_no_records_and_named_errors():
    sim = xtim.Circuit("H 0\nDEPOLARIZE1(0.1) 0\nM 0\nDETECTOR rec[-1]\n").compile_exact_branches()
    law = sim.exact_branches([(0, 0, "Z")])
    with pytest.raises(AttributeError, match="raw per-branch records"):
        law.measurements
    assert law.num_rows == 1 and law.num_uniform == 1
    assert len(law) == 2 and np.allclose(law.probabilities, 0.5)


# Built to exercise every piece at once: a random read (fair coin) that is a detector AND flips
# a declared expectation frame, magic reads feeding a DECISION and a detector, feedback across a
# T, `!` and M(p) records, and expectations conditioned on the magic outcome.
MIXED = """R 0 1 2 3 4
H 0 3
T 0
CX 0 1
X_ERROR(0.01) 0 1
MX 4
DETECTOR rec[-1]
H 2
M(0.02) 2
CZ rec[-1] 3
T 3
DEPOLARIZE1(0.01) 0 3
MX !0
DECISION(0) rec[-1]
M 1
DETECTOR rec[-1] rec[-2]
PAULI_EXPECTATION(0) X3 rec[-3]
PAULI_EXPECTATION(1) Z1
PAULI_EXPECTATION(2) Y3*Z4
PAULI_EXPECTATION(3) Z1 rec[-4]
"""


def test_mixed_vs_dense_oracle_every_single_fault():
    sim = xtim.Circuit(MIXED).compile_exact_branches()
    sites = sim.noise_sites()
    cases = [({}, set())]
    for s, st in enumerate(sites):
        for t in range(len(st["qubits"])):
            for p in "XYZ":
                cases.append(({(s, t): p}, set()))
    nrec = xtim.Circuit(MIXED).num_measurements
    cases += [({}, {j}) for j in range(nrec) if j not in sim.refused_records]
    rng = random.Random(3)
    for _ in range(60):
        f = {}
        for s, st in enumerate(sites):
            for t in range(len(st["qubits"])):
                if rng.random() < 0.3:
                    f[(s, t)] = rng.choice("XYZ")
        cases.append((f, {j for j in range(nrec) if j not in sim.refused_records and rng.random() < 0.2}))
    cands = None
    seen = dict(directions=0, sign_flips=0, magic_rows=0)
    for faults, flips in cases:
        law = sim.exact_branches([(s, t, p) for (s, t), p in faults.items()] +
                                 [{"record": j} for j in sorted(flips)])
        cands = law_shifts(law, aggregate(dense_branches(MIXED, 5, faults, flips)), cands)
        assert cands, f"no constant reference makes the law equal the oracle: {faults} {flips}"
        seen["directions"] += law.num_uniform > 0
        seen["sign_flips"] += bool(law.uniform_expectation_signs.any())
        seen["magic_rows"] += law.num_rows > 1
    assert seen["directions"] and seen["sign_flips"] and seen["magic_rows"], seen


def _t_decisions(u, k):
    """k |T> states and u fresh ancillas, all read in X, every read a DECISION: 2^k rows, u coins."""
    t = "".join(f"H {q}\nT {q}\n" for q in range(k))
    t += "MX " + " ".join(map(str, range(k))) + "\n"
    if u:
        t += "MX " + " ".join(map(str, range(k, k + u))) + "\n"
    t += "".join(f"DECISION({i}) rec[-{k + u - i}]\n" for i in range(k + u))
    return t


@pytest.mark.parametrize("u,k", [(61, 3), (62, 2), (60, 4), (63, 1), (200, 2)])
def test_huge_expanded_view_refused_not_overflowed(u, k):
    """B * 2^u >= 2^63 once overflowed a signed shift (segfault): now a named refusal."""
    law = xtim.Circuit(_t_decisions(u, k)).compile_exact_branches().exact_branches([])
    assert law.num_rows == 1 << k and law.num_uniform == u
    assert abs(law.rows_probabilities.sum() - 1) < 1e-12
    for field in ("probabilities", "detectors", "observables", "decisions", "expectations"):
        with pytest.raises(xtim.ExpandedLawTooLarge, match="factored fields"):
            getattr(law, field)
    with pytest.raises(ValueError, match="max_rows"):          # the C++ expander's own guard
        law._expander.expand(1 << 62)


def test_magic_walk_work_budget_refuses_by_name():
    """The magic-parity walk is capped by a deterministic WORK count: k = 8 answers under the
    default, k = 9 is refused by name (naming the magic rank), and a small max_work refuses
    early; raising max_work answers."""
    sim8 = xtim.Circuit(_t_decisions(0, 8)).compile_exact_branches()
    assert sim8.exact_branches([]).num_rows == 256
    with pytest.raises(ValueError, match=r"8 magic-sector parities \(magic rank 8.*max_work"):
        sim8.exact_branches([], max_work=1e6)
    sim9 = xtim.Circuit(_t_decisions(0, 9)).compile_exact_branches()
    with pytest.raises(ValueError, match=r"magic rank 9.*exceeds max_work"):
        sim9.exact_branches([])
    small = xtim.Circuit(_t_decisions(0, 4)).compile_exact_branches()
    with pytest.raises(ValueError, match="max_work"):
        small.exact_branches([], max_work=10.0)
    assert small.exact_branches([], max_work=1e9).num_rows == 16


def test_expanded_view_is_lazy_and_columns_on_request():
    sim = xtim.Circuit(MIXED).compile_exact_branches()
    law = sim.exact_branches([(0, 0, "Z")])
    assert law._expanded is None                       # nothing expanded until asked
    assert law.num_uniform >= 1 and law._expanded is None
    law.probabilities
    assert law._expanded is not None
    none = sim.exact_branches([(0, 0, "Z")], expectations=[])
    assert none.rows_expectations.shape[1] == 0 and none.expectations.shape[1] == 0
    assert none.uniform_expectation_signs.shape[1] == 0 and none.columns == ()
    sub = sim.exact_branches([(0, 0, "Z")], expectations=[3, 1])
    assert sub.columns == (3, 1)
    assert np.allclose(sub.rows_expectations, law.rows_expectations[:, [3, 1]])
