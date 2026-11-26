"""J-b1 P7: record_fidelity on layout-3 records — the test-API evaluator (xtim._xtim_testapi.FamilyRecordFidelity),
REGISTER fidelity on OUTPUT_QUBITS under the strict contract (RF-ref 2026-10-22). Gates:
  * law: per record, F equals the dense partial-trace oracle (tests/_register_oracle.py) on the materialized retained
    state at 1e-12 — CH/PPR kind-2 records, framed records (input_pauli frames);
  * the kappa_rec > 0 / Born-observable fixtures whose outputs depend on unfed-back outcomes are refused by name;
  * the test-API evaluator equals the production one on the same shots, with and without a decoded correction;
  * G-3: records from a sampler whose state model differs only in a gauge detector are refused by name;
  * m10: kind-2 reader tables are charged to the readers' bytes (plan_payload_stats reader_bytes grows);
  * MUT-RF-L3 (P_rec read from the wrong word) -> red here.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from test_s2b_record_table import FIXTURES as S2B  # noqa: E402
from test_jb2_kappa_rec_law import kappa_dual_circuit_text as circuit_text  # noqa: E402
from _public import to_prod, public_sampler  # noqa: E402
from _register_oracle import bare_terms, register_F_terms  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-12


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def check_law(text, s, R, rows):
    ow = list(s.ideal_state(provenance=False).wires)
    terms = bare_terms(text, ow)
    Fv = np.asarray(TA.FamilyRecordFidelity(s).evaluate(s, R))
    worst = max(abs(Fv[i] - register_F_terms(terms, to_prod(s.materialize(R, i)), len(ow))) for i in rows)
    assert worst < TOL, worst
    assert np.any(Fv > 1e-6)
    return Fv


KAPPA = {"kappa_B2": circuit_text(2)}                    # 8 output wires (the oracle enumerates 4^8 Paulis once)
BORN = {k: S2B[k] for k in ("m3_obs", "m3_obs_dec", "m3_obs_2dec", "q3_dec") if k in S2B}


@pytest.mark.parametrize("name", sorted(KAPPA))
def test_law_vs_materialize(name):
    s = fam(KAPPA[name])
    d, o, dc, R = s.sample_records(300, seed=5)
    check_law(KAPPA[name], s, R, range(0, 300, 10))


@pytest.mark.parametrize("name", sorted(BORN))
def test_born_fixtures_served_exactly(name):
    """The Born-observable / decision fixtures with an output register added (a fresh wire): served, every row equal to
    the dense oracle (the Born records live on the rest)."""
    text = BORN[name] + "R 99\nOUTPUT_QUBITS out 99\n"
    s = fam(text)
    d, o, dc, R = s.sample_records(300, seed=5)
    check_law(text, s, R, range(0, 300, 3))


def test_kind2_records_vs_materialize():
    text = open(os.path.join(ROOT, "tests", "data", "ch_cultivation.stim")).read()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("PAULI_EXPECTATION")) + "\n"
    text += "OUTPUT_QUBITS out 0\n"
    s = fam(text)
    d, o, dc, R = s.sample_records(400, seed=7)
    assert any(R.kind(i) == 2 for i in range(400))
    check_law(text, s, R, range(0, 400, 7))


def test_framed_records_vs_materialize():
    port = F.PORT + "OUTPUT_QUBITS out 5\n"              # the input port qubit is the output register
    s = fam(port)
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 2, size=(200, 2)).astype(np.uint8)   # (x, z) per port qubit
    d, o, dc, R = s.sample_records(200, seed=3, input_pauli=frames)
    Fv = check_law(port, s, R, range(0, 200, 4))
    assert np.any(Fv < 1 - 1e-6)                                  # frames lower F


def test_gauge_only_model_difference_is_refused():
    base = "RX 0\nT 0\nH 0\nX_ERROR(0.1) 0\nMX 0\nR 2\nX_ERROR(0.1) 2\nM 2\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n"
    a = fam(base + "DETECTOR rec[-2]\n")       # an extra gauge detector: the only model difference
    b = fam(base)
    rf = TA.FamilyRecordFidelity(a)
    *_, Rb = b.sample_records(10, seed=1)
    with pytest.raises(ValueError, match="different state model"):
        rf.evaluate(b, Rb)
    *_, Ra = a.sample_records(10, seed=1)
    rf.evaluate(a, Ra)


def test_kind2_reader_tables_are_charged():
    """m10: a kind-2 content's (G) table goes through the readers' memo and is CHARGED (XTIM_MUTATE_M10_NO_CHARGE
    -> red): materializing only kind-2 records grows reader_bytes by at least one table."""
    s = fam(open(os.path.join(ROOT, "tests", "data", "ch_cultivation.stim")).read())
    d, o, dc, R = s.sample_records(400, seed=11)
    k2 = [i for i in range(400) if R.kind(i) == 2]
    assert k2, "no kind-2 record in the sample"
    before = s.plan_payload_stats()["reader_bytes"]
    for i in k2[:8]:
        s.materialize(R, i)
    after = s.plan_payload_stats()["reader_bytes"]
    assert after > before, (before, after)


@pytest.mark.parametrize("name", ["tests/data/magna_color5_fid.stim", "tests/data/magna_rp2_fid.stim"])
def test_testapi_equals_production_with_and_without_correction(name):
    """The test-API evaluator on the test-API records equals the production evaluator on the public buffer of the same
    shots (same seed: the detector bytes are asserted equal), plain and decoded (random output corrections)."""
    text = open(os.path.join(ROOT, name)).read()
    s = fam(text)
    N = 400
    d, o, dc, R = s.sample_records(N, seed=9)
    ps = public_sampler(text)
    b = ps.sample_barrier(N, seed=9)
    assert np.array_equal(np.asarray(b.dets()), np.asarray(d))
    Fv = np.asarray(TA.FamilyRecordFidelity(s).evaluate(s, R))
    assert np.array_equal(Fv, np.asarray(xtim.RecordFidelity(ps).evaluate(b)))
    no = len(ps.output_wires())
    c = np.random.default_rng(9).integers(0, 2, (N, 2, no)).astype(np.uint8)
    Fc = np.asarray(TA.FamilyRecordFidelity(s).evaluate(s, R, correction=c))
    assert np.array_equal(Fc, np.asarray(xtim.RecordFidelity(ps).evaluate(b, correction=c)))
    assert np.any(np.abs(Fc - Fv) > 1e-6)
    with pytest.raises(ValueError, match="shape mismatch"):
        TA.FamilyRecordFidelity(s).evaluate(s, R, correction=c[:, :1])
