"""EXACT per-fault oracle for DEM export (3.1.10): no sampling.

`exact_sv` is an independent branching statevector simulator over the live qubits: every
measurement / reset outcome with nonzero probability is a branch carrying its exact probability,
records and PAULI_EXPECTATION values. For every Pauli alternative of every channel, the exact
joint law of (detector flips, deterministic-observable flips, sign-constant expectation-column
flips, magnitude change) is compared with the exported one-hot DEM to 1e-12; refusals must be
non-factorable laws (or, for "non-fault-tolerant", a fault that harms the accepted branch);
flagged faults must be harmless on the accepted branch; approximate_disjoint_errors must carry
each signature's exact first-order mass; each whole channel's export must equal the mixture.
The full corpus / cultivation / random-family runs are in the 3.1.10 report; this file runs the
small circuits in full and the large ones on a subset of channels.
"""
import random

import pytest

import xtim
import exact_sv as E
import test_dem_correlated_reads as C


def export(text, approx):
    return xtim._xtim.export_dem_text(text, True, False, False, True, [], [], approx)


@pytest.mark.parametrize("name", ["miniature_oracle", "cube_ccz"])
def test_bundled_small_examples_exact(name):
    E.check_circuit(xtim.load_example(name).text, export)


def test_cultivation_d3_faithful_exact_subset():
    cnt = E.check_circuit(xtim.load_example("cultivation_d3_faithful").text, export, every=18)
    assert cnt["alts"] > 15, cnt


@pytest.mark.parametrize("name", sorted(C.SILENT_319))
def test_correlated_repros_exact(name):
    E.check_circuit(C.SILENT_319[name], export)


def test_random_bell_ghz_family_exact():
    rng = random.Random(22)
    done = rand = 0
    while done < 60:
        body = C._random_case(rng)
        if body is None or export(body, 0.0)["rejected"]:
            continue
        cnt = E.check_circuit(body, export)
        done += 1
        rand += cnt["random"]
    assert rand >= 5, rand


@pytest.mark.parametrize("cn,ch,text", list(C._cases()))
def test_clifford_set_exact(cn, ch, text):
    E.check_circuit(text, export)


def test_oracle_detects_the_3_1_9_independent_coin_law():
    """Test-of-test: the oracle's verdict on a law it must reject (the 3.1.9 model of the
    correlated pair: three independent pairs instead of {D0, D2} together)."""
    orc = E.Oracle(C.SILENT_319["correlated_pair"])
    li = orc.noise_idx[0]
    law = orc.law(li - 0, {3: "X"})
    assert {k[0]: v for k, v in law.items()} == pytest.approx(
        {frozenset(): 0.5, frozenset({0, 2}): 0.5}, abs=1e-12)
