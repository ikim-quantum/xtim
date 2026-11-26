"""fix/ladder-settle: the even-Clifford ladder settle (J-b1 P5) runs inside EVERY bare build (build_parity_form), not only
on the family build. Before, the plain build (bare_state_of, the exact engine, detector_determinism, the stim DEM export,
ref_compile, ...) refused these circuits "rotation-form terms anticommute (even-Clifford ladder gap)" although the
family build served them exactly.

Fixtures (tests/data/ladder_settle/cases.json; generator note: gen_cases.py there): the 19 ladder-gap "circuit + undo"
refusals of the output_state undo probe, and 24 probe circuits that output_state accepts but the plain build refused.

Oracles (exact, xtim/stim only): the probe's recorded clifford_paulis signs (the family build's exact bare
expectations, pre-fix); xtim.output_state's GF(2) result checked by tests/_output_state_oracle (MPP reads through
detector_determinism + the exact engine, Heisenberg fallback). The noisy-circuit rule is NOT touched: the named
fault-dependent refusals (ch_pair, t_h_ch) still refuse. Independent oracle: magic-double circuits vs their
Clifford rewrite (pairs.json; overlap, exact PAULI_EXPECTATIONs, stim tableau on the Clifford-only ones). Mutation XTIM_MUTATE_NC_NOFOLD (the settle removed) -> red."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _output_state_oracle as O  # noqa: E402
import _exact_engine as E  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim  # noqa: E402
from xtim.errors import XtimTwirlRefusal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CASES = {c["name"]: c for c in json.load(open(os.path.join(os.path.dirname(__file__), "data", "ladder_settle",
                                                           "cases.json")))}
UNDO = sorted(k for k, c in CASES.items() if c["kind"] == "undo")
BASE = sorted(k for k, c in CASES.items() if c["kind"] == "base")


def test_fixture_counts():
    assert len(UNDO) == 19 and len(BASE) == 24


@pytest.mark.parametrize("name", UNDO + BASE)
def test_plain_bare_build_accepts(name):
    _xtim.bare_state_of(O.base_text(CASES[name]["text"]))          # refused "ladder gap" before the fix


@pytest.mark.parametrize("name", UNDO)
def test_undo_reads_have_the_family_signs(name):
    """circuit + undo: every clifford_pauli reads deterministically with the family build's (probe) sign, through
    detector_determinism and the exact engine (assert_measures returned False = 'xtim refused' before the fix)."""
    assert O.assert_measures(CASES[name]["text"], [tuple(r) for r in CASES[name]["clifford_paulis"]])


@pytest.mark.parametrize("name", BASE)
def test_output_state_oracle_closes(name):
    """assert_state failed 'xtim refused the circuit itself' on these before the fix."""
    c = CASES[name]
    st = xtim.output_state(c["text"], c["qubits"])
    assert st.rotations
    assert O.assert_state(c["text"], st) in ("undo", "heisenberg")


@pytest.mark.parametrize("name", BASE[:8])
def test_plain_sampler_matches_family_on_deterministic_reads(name):
    """The plain (exact-engine) sampler and the family detector sampler agree on the deterministic stabilizer reads
    (all detectors 0: every read has its stated sign)."""
    c = CASES[name]
    st = xtim.output_state(c["text"], c["qubits"])
    reads = list(st.pauli_stabilizers)
    if not reads:
        pytest.skip("no pauli_stabilizers")
    t = O.base_text(c["text"]) + "".join(f"MPP {p}\nDETECTOR rec[-1]\n" for p, _ in reads)
    flips = np.array([s < 0 for _, s in reads])
    m = np.asarray(E.exact_sampler(xtim.Circuit(t), seed=5).sample(16))
    assert (m[:, -len(reads):] == flips).all()
    d = np.asarray(xtim.Circuit(t).compile_detector_sampler(seed=3).sample(16))
    assert not d.any()


import test_jb1_noncommuting as NC  # noqa: E402

NC_NAMES = ["ch_pair", "t_h_ch"]   # noise between the terms: the oracle REFUSES, FAM "NC"
NC_REFUSAL = r"fault-dependent surviving axis/Clifford frame: terms \d+ .* merge only when no fault occurs"


@pytest.mark.parametrize("name", NC_NAMES)
def test_noise_between_terms_still_refuses_by_name(name):
    text, oracle, fam = NC.FIXTURES[name]
    assert (oracle, fam) == ("REFUSE", "NC")
    with pytest.raises(XtimTwirlRefusal, match=NC_REFUSAL):
        xtim.Circuit(text).compile_detector_sampler(seed=0)


# Independent oracle (review round 1): A carries magic doubles and hit the ladder gap at b0a06fc; B is A with every
# double rewritten to its Clifford (T;T -> S, T;D;T -> D;S, CS;CS -> CZ, CH;CH, CCZ;CCZ -> I), the same unitary, which
# the old build accepted WITHOUT the settle. Fixtures: tests/data/ladder_settle/pairs.json (generator: gen_pairs.py).
PAIRS = json.load(open(os.path.join(os.path.dirname(__file__), "data", "ladder_settle", "pairs.json")))
PAIR_IDS = [f"{'cliff' if c['clifford'] else 'magic'}_{c['seed']}" for c in PAIRS]


def test_pair_fixture_covers_every_double():
    kinds = {(k, c["clifford"]) for c in PAIRS for k in c["kinds"]}
    assert kinds == {(k, cl) for k in ("t_t", "t_diag_t", "cs_cs", "ch_ch", "ccz_ccz") for cl in (True, False)}


def _pad(c):
    return "I " + " ".join(map(str, range(c["n"]))) + "\n"


def _labels(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < 12:
        f = rng.integers(0, 4, n)
        if f.any():
            out.append("*".join(f"{'_XYZ'[v]}{q}" for q, v in enumerate(f) if v))
    return out


@pytest.mark.parametrize("c", PAIRS, ids=PAIR_IDS)
def test_settled_build_equals_the_clifford_rewrite(c):
    import stim
    A, B = _pad(c) + c["A"], _pad(c) + c["B"]
    sa, sb = _xtim.bare_state_of(A), _xtim.bare_state_of(B)
    assert abs(complex(sb.overlap(sa)) - 1) < 1e-9                     # |<B|A>|^2 = 1: the same state up to phase
    labels = _labels(c["n"], c["seed"])
    tail = "".join(f"PAULI_EXPECTATION({j}) {l}\n" for j, l in enumerate(labels))
    _, ea = E.exact_detector_sampler(xtim.Circuit(A + tail), seed=1).sample(2, return_expectations=True)
    _, eb = E.exact_detector_sampler(xtim.Circuit(B + tail), seed=1).sample(2, return_expectations=True)
    ea, eb = np.asarray(ea)[0], np.asarray(eb)[0]
    assert np.allclose(ea, eb, atol=1e-9)
    if c["clifford"]:                                                   # B is a stim circuit: tableau agreement
        sim = stim.TableauSimulator()
        sim.do(stim.Circuit(B))
        want = [sim.peek_observable_expectation(_ps(l, c["n"])) for l in labels]
        assert np.allclose(np.real(ea), want, atol=1e-9)


def _ps(label, n):
    import stim
    p = stim.PauliString(n)
    for f in label.split("*"):
        p[int(f[1:])] = f[0]
    return p
