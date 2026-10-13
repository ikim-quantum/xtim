"""DEM text of code-block circuits with MAGIC left in the bare state, pinned against a golden produced
by the pre-factoring base build (fd4f31c, before the factored correlated-read law of perf/dem-factored-law).

Why these circuits: a deterministic single-qubit read of a bare state must be I/Z on every magic wire,
so on a plain Clifford+T circuit a pre-T Pauli fault only FLIPS it (A_t = -D_t) and the correlated-read
law never meets magic (r_free = 0 after T noise T_DAG, which is Clifford noiselessly). The law sees
magic only when the T layer is a LOGICAL non-Clifford on a code: here [[8,3,2]] cube blocks (bipartite
T/T_DAG = logical CCZ, r_free 3) and [[15,1,3]] quantum Reed-Muller blocks (T^15, r_free 1), one or two
blocks (r_free up to 6, chi up to 64), with noise before/after the T layer, ancilla stabilizer checks,
destructive X/Z block reads with stabilizer-parity detectors, observables and expectations. Vendored
as tests/data/dem_factored_law/*.stim (generation needs stim's tableau elimination, whose output is
stim-version dependent, so the circuits are data, not code); the generator and its provenance are
recorded in the golden. Every circuit's correlated-law call count on the factored build is recorded
too (gdb breakpoint count); each pinned circuit hits the law at least once.

Four legs per circuit (default / approximate_disjoint_errors x with / without expectations), compared
as exact text — a refusal is pinned as "REFUSED <type>: <message>" and must refuse identically.
"""
import json
import os

import pytest

import xtim

HERE = os.path.dirname(__file__)
GOLDEN = os.path.join(HERE, "data", "dem_factored_law_golden.json")
CIRC_DIR = os.path.join(HERE, "data", "dem_factored_law")

LEGS = {
    "dem": dict(),
    "dem_noexp": dict(include_expectations=False),
    "dem_approx": dict(approximate_disjoint_errors=True),
    "dem_approx_noexp": dict(approximate_disjoint_errors=True, include_expectations=False),
}

with open(GOLDEN) as fh:
    DATA = json.load(fh)


def _leg(circuit, kw):
    try:
        return circuit.detector_error_model_text(**kw)
    except Exception as e:  # noqa: BLE001 - the refusal text is the pinned value
        return "REFUSED " + type(e).__name__ + ": " + str(e)


def test_golden_provenance():
    assert DATA["commit"] == "fd4f31c"
    assert "gen_v2" in DATA["generator"]
    circs = DATA["circuits"]
    assert len(circs) >= 20
    assert {c["r_free"] for c in circs.values()} >= {1, 2, 3, 4, 6}
    assert max(c["chi"] for c in circs.values()) == 64
    assert all(c["law_calls"] >= 1 for c in circs.values())
    assert all(set(c) >= set(LEGS) for c in circs.values())
    for name in circs:
        assert os.path.exists(os.path.join(CIRC_DIR, name)), name


@pytest.mark.parametrize("name", sorted(DATA["circuits"]))
def test_dem_text_matches_fd4f31c_golden(name):
    with open(os.path.join(CIRC_DIR, name)) as fh:
        text = fh.read()
    want = DATA["circuits"][name]
    bf = xtim.bare_state_of(text).branch_frames()
    assert (len(bf["free"]), len(bf["coeff"])) == (want["r_free"], want["chi"])
    c = xtim.Circuit(text)
    for leg, kw in LEGS.items():
        got = _leg(c, kw)
        assert got == want[leg], (name, leg)
