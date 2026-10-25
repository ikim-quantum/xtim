"""J-b1 P13 (plan r3.2): the typed twirl refusal on the public TwirlDetectorSampler
(xtim.Circuit(text).compile_detector_sampler(); J-b2 J1: the pre-flip test-API twin tests/_jb1_twin.py is retired).

Every TwirlClass value is raised by at least one fixture here (no dead values).
"""
import os
import pickle
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import xtim  # noqa: E402
from xtim.errors import TwirlClass, XtimTwirlRefusal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

TAIL = "R 15\nX_ERROR(0.1) 15\nM 15\nDETECTOR rec[-1]\n"
BARE = "RX 0\nT 0\nH 0\nT 0\nMX 0\nDETECTOR rec[-1]\n" + TAIL
T_X_T_H_T = "RX 0\nT 0\nX_ERROR(0.2) 0\nT 0\nH 0\nT 0\nMY 0\nDETECTOR rec[-1]\n" + TAIL
# a rule-ACCEPT circuit the propagation table refuses (one of the 29 declared, fixplan-p5-c14)
C14 = ("RX 0 1\nY_ERROR(0.1) 0\nT 0\nT 0\nZ 0\nX_ERROR(0.1) 1\nH 0\nCX 0 1\nCS 1 0\nH 0\nS 0\nH 0\nCS_DAG 1 0\n"
       "CS 1 0\nT_DAG 1\nMX 0 1\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")
CH_PAIR = "H 5\nT 5\nCH 5 1\nH 0\nCH 1 0\nX_ERROR(0.35) 0\nCH 1 0\nMX 0 5\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n" + TAIL
T_H_CH = "T 0\nH 0\nH 1\nCH 0 1\nX_ERROR(0.35) 1\nCH 0 1\nMX 1 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n" + TAIL
CS_WITNESS = ("RX 0 1\nT 1\nH 1\nCZ 0 1\nCS_DAG 0 1\nDEPOLARIZE1(0.1) 0\nCX 0 1\nCS_DAG 1 0\nH 1\n"
              "DEPOLARIZE1(0.1) 1\nS_DAG 1\nMX 0 1\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")
SERVED = {
    "noiseless": "RX 0\nT 0\nMX 0\nDETECTOR rec[-1]\n" + TAIL,
    # setup_error 2's shape in production ("no deterministic detector channels"): served by FAM(all)
    "no_det_channels": "RX 0\nT 0\nX_ERROR(0.1) 0\nMX 0\nDETECTOR rec[-1]\n",
    "born_decision_on_ch": ("RX 0\nH 1\nCH 1 0\nDEPOLARIZE1(0.1) 0\nMX 0 1\nDETECTOR rec[-1]\nDECISION(0) rec[-2]\n"
                            + TAIL),
    "gauge": "RX 0\nT 0\nH 0\nX_ERROR(0.1) 0\nMX 0\nDETECTOR rec[-1]\n" + TAIL,
    "anti": "RX 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n" + TAIL,
}

SEEN = set()


def compile_detector_sampler_twin(text, *, seed=0):
    """The public twirl detector sampler (the name is kept from the retired pre-flip twin)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.Circuit(text).compile_detector_sampler(seed=seed)


def refusal(text):
    with pytest.raises(XtimTwirlRefusal) as ei:
        compile_detector_sampler_twin(text, seed=1)
    e = ei.value
    assert isinstance(e, ValueError) and isinstance(e, xtim.XtimError)
    SEEN.add(e.twirl_class)
    return e


def test_bare_reject_is_outside_t_depth_1_bare():
    e = refusal(BARE)
    assert e.twirl_class == TwirlClass.OUTSIDE_T_DEPTH_1 and e.sub_reason == "bare"
    m = str(e)
    assert m.startswith("bare state rejected: non-commuting magic")
    assert "in user qubits: " in m and "q0" in m and m.endswith("the circuit is outside xtim's model (its non-Clifford "
                                                                 "gates do not fold into one commuting layer of pi/8 "
                                                                 "rotations)")


def test_class14_rule_accept_is_a_law_limitation_propagation():
    e = refusal(C14)
    assert e.twirl_class == TwirlClass.LAW_LIMITATION_PROPAGATION and e.sub_reason == ""
    m = str(e)
    assert re.match(r"twirl sampler: law limitation \(propagation\): the static rule accepts this circuit \(T-depth "
                    r"<= 1\), but a [XYZ] fault at deferred noise site \d+ on deferred wire \d+ \(deferred wire \d+ = q\d\) "
                    r"leaves the gate-by-gate propagation table's diagonal \+ Pauli class at deferred gate \d+: a fault "
                    r"meets a same-axis non-Clifford pair before it merges; a law limitation of the twirl engine", m), m
    assert "OUTSIDE" not in m and "T-depth 1" not in m.replace("T-depth <= 1", "")


def test_class14_rule_refuse_is_noncommuting_magic():
    e = refusal(T_X_T_H_T)
    assert e.twirl_class == TwirlClass.NONCOMMUTING_MAGIC
    assert "(the propagation table refused it first: " in str(e)


@pytest.mark.parametrize("text", [CH_PAIR, T_H_CH, CS_WITNESS], ids=["ch_pair", "t_h_ch", "cs_witness"])
def test_noncommuting_magic(text):
    e = refusal(text)
    assert e.twirl_class == TwirlClass.NONCOMMUTING_MAGIC and e.sub_reason == ""
    m = str(e)
    assert m.startswith("twirl sampler: law limitation: fault-dependent surviving axis/Clifford frame: ")
    assert "; in user qubits: " in m and "the non-Clifford gates on these qubits" in m, m


def test_expectations_and_raw_records_are_refused_by_name():
    s = compile_detector_sampler_twin(SERVED["noiseless"], seed=1)
    with pytest.raises(XtimTwirlRefusal) as ei:
        s.sample(10, return_expectations=True)
    assert ei.value.twirl_class == TwirlClass.EXPECTATIONS_NOT_SERVED and "RecordFidelity" in str(ei.value)
    SEEN.add(ei.value.twirl_class)
    with pytest.raises(XtimTwirlRefusal) as ei:
        s.sample(10, return_measurements=True)
    assert ei.value.twirl_class == TwirlClass.RAW_RECORDS_NOT_SERVED and "detectors, observables" in str(ei.value)
    SEEN.add(ei.value.twirl_class)


def test_explicit_exact_serves_expectations():
    text = "RX 0\nT 0\nMX 0\nDETECTOR rec[-1]\nPAULI_EXPECTATION(0) X0\n"
    s = xtim.Circuit(text).compile_detector_sampler(seed=1, engine="exact")
    out = s.sample(4, return_expectations=True)
    assert out is not None


@pytest.mark.parametrize("name", sorted(SERVED))
def test_served_rows(name):
    s = compile_detector_sampler_twin(SERVED[name], seed=3)
    d, o = s.sample(200, separate_observables=True)
    assert d.shape[0] == 200 and o.shape[0] == 200


def _carried_axis_refusal():
    """J-b1 P9b: LAW_LIMITATION_CARRIED_AXIS (test API until F): the CE1-chain dressed branch, whose frame leaves the
    ideal post-barrier frame, is refused by name at add_branch (sub "branch_frame")."""
    import warnings
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = TA.compile_family_sampler("RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\n"
                                      "OUTPUT_QUBITS out 0\n", selfcheck=0)
        c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), "INPUT_QUBITS in 0\nI 0\n")
    R = p.sample_records(300, seed=1)[-1]
    for h in p.branch_handles(R):
        try:
            c.add_branch(h)
        except XtimTwirlRefusal as e:
            return e
    raise AssertionError("the CE1-chain dressed branch was not refused")


def test_carried_axis_is_refused_by_name():
    e = _carried_axis_refusal()
    assert e.twirl_class == TwirlClass.LAW_LIMITATION_CARRIED_AXIS and e.sub_reason == "branch_frame"
    SEEN.add(e.twirl_class)


PLUS = "R 0\nH 0\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"


def test_register_fidelity_honours_a_correction():
    """The register fidelity takes the decoder's correction (formerly CORRECTION_NOT_SERVED on reference=True): |+>
    under a Z correction scores 0, uncorrected 1."""
    s = xtim.compile_twirl_sampler(PLUS, selfcheck=0)
    b = s.sample_barrier(4, seed=1)
    corr = np.zeros((4, 2, 1), np.uint8)
    corr[:, 1, 0] = 1                                  # Z on the output: |+> -> |->
    assert np.allclose(xtim.RecordFidelity(s).evaluate(b), 1.0)
    assert np.allclose(xtim.RecordFidelity(s).evaluate(b, correction=corr), 0.0, atol=1e-12)


def test_every_twirl_class_is_raised_by_a_fixture():
    # the fixtures above, re-run here so the check does not depend on test order
    for t in (BARE, C14, CH_PAIR, T_X_T_H_T):
        refusal(t)
    SEEN.add(_carried_axis_refusal().twirl_class)
    s = compile_detector_sampler_twin(SERVED["noiseless"], seed=1)
    for kw in ({"return_expectations": True}, {"return_measurements": True}):
        with pytest.raises(XtimTwirlRefusal) as ei:
            s.sample(1, **kw)
        SEEN.add(ei.value.twirl_class)
    assert SEEN == set(TwirlClass), set(TwirlClass) - SEEN


def test_refusal_pickles():
    e = refusal(CH_PAIR)
    e2 = pickle.loads(pickle.dumps(e))
    assert type(e2) is XtimTwirlRefusal and e2.twirl_class == e.twirl_class and str(e2) == str(e)


def test_public_bytes_equal_the_family_sampler():
    from xtim import _xtim_testapi as TA
    text = SERVED["gauge"]
    a = compile_detector_sampler_twin(text, seed=5).sample(500, separate_observables=True, bit_packed=True)
    b = TA.compile_family_sampler(text, selfcheck=0, decisions=False).sample(500, seed=5)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


# J-b1 F (F item 4, MUT-AUTO-FALLBACK): the PUBLIC entry points refuse by name, with no exact construction. A
# silent exact route on any of them (the thunk, a class -> exact branch) builds an exact program here: the
# counter delta below (and the session fixture, tests/conftest.py) turn red.
PUBLIC_REFUSED = {"bare": (BARE, TwirlClass.OUTSIDE_T_DEPTH_1), "c14": (C14, TwirlClass.LAW_LIMITATION_PROPAGATION),
                  "ch_pair": (CH_PAIR, TwirlClass.NONCOMMUTING_MAGIC), "t_h_ch": (T_H_CH, TwirlClass.NONCOMMUTING_MAGIC),
                  "cs_witness": (CS_WITNESS, TwirlClass.NONCOMMUTING_MAGIC)}


@pytest.mark.parametrize("name", sorted(PUBLIC_REFUSED))
def test_public_entry_points_refuse_by_name_without_exact(name):
    from xtim._exact_scope import exact_constructions
    text, cls = PUBLIC_REFUSED[name]
    a = exact_constructions()
    for build in (lambda: xtim.Circuit(text).compile_detector_sampler(seed=1).sample(10),
                  lambda: xtim.compile_twirl_sampler(text, selfcheck=0).sample(10, seed=1)):
        with pytest.raises(XtimTwirlRefusal) as ei:
            build()
        assert ei.value.twirl_class == cls, (name, ei.value.twirl_class)
    assert exact_constructions() == a, name
    s = xtim.Circuit(SERVED["noiseless"]).compile_detector_sampler(seed=1)
    with pytest.raises(XtimTwirlRefusal) as ei:
        s.sample(10, return_expectations=True)
    assert ei.value.twirl_class == TwirlClass.EXPECTATIONS_NOT_SERVED
    assert exact_constructions() == a, name


# README stage (Q4): no refusal text points at the exact engine (the README has no exact mode); every refusal says
# the request is outside xtim's model. (Was J-b1 MAJOR-2: the texts' claims about the exact engine's verdict.)
P18 = ("R 0\nH_YZ 0\nT 0\nZ_ERROR(0.1) 0\nH_YZ 0\nT_DAG 0\nDEPOLARIZE1(0.1) 0\nT 0\nH 0\nH_YZ 0\nZ_ERROR(0.1) 0\n"
       "M 0\nDETECTOR rec[-1]\n")


def _c14_texts():
    path = os.path.join(os.path.dirname(__file__), "data_jb1_nc_c14.txt")
    out, cur = [], []
    for line in open(path):
        if line.startswith("#"):
            if cur:
                out.append("".join(cur))
            cur = []
        elif line.strip():
            cur.append(line)
    if cur:
        out.append("".join(cur))
    return out


@pytest.mark.parametrize("which", ["c14", "nc"])
def test_refusal_text_names_no_exact_engine(which):
    import test_ja_term_key as K
    texts = _c14_texts() if which == "c14" else [CH_PAIR, T_H_CH, CS_WITNESS, T_X_T_H_T, P18, BARE] + \
        [t for t in getattr(K, "NONCOMM", {}).values() if isinstance(t, str)]
    assert texts
    for t in texts:
        with pytest.raises(XtimTwirlRefusal) as ei:
            xtim.compile_twirl_sampler(t, selfcheck=0).sample(5, seed=1)
        m = str(ei.value)
        assert "exact" not in m.replace("exact record", ""), m
        assert "outside xtim's model" in m, m
    assert len(texts) >= (29 if which == "c14" else 6)


# J-b1 final review MAJOR-4: the carried-axis / injection refusals, on the PUBLIC API and on the test API, are typed
# and take no exact construction in EITHER extension (xtim._exact_scope sums both counters). MUT-INJ-FALLBACK (each
# such refusal first takes a silent exact construction in the shared code) turns these red.
def _carried_cases(api):
    import test_jb1_branch_law_e0 as E0
    import test_jb1_carried_compose as CC
    from xtim import _xtim_testapi as TA

    def producer(text):
        return (xtim.compile_twirl_sampler(text, selfcheck=0) if api == "public"
                else TA.compile_family_sampler(text, selfcheck=0))

    def consumer(p, text):
        return (xtim.compile_twirl_sampler_from_ideal_state(text, ideal_state=p.ideal_state(), selfcheck=0)
                if api == "public" else TA.compile_family_sampler_from_ideal_state(p.ideal_state(), text))

    def records(p, shots):
        return p.sample_barrier(shots, seed=1) if api == "public" else p.sample_records(shots, seed=1)[-1]

    def free_x():
        consumer(producer(CC.MINIMAL_TD2[0]), CC.MINIMAL_TD2[1])

    def add_all(ptext, ctext):
        p = producer(ptext)
        c = consumer(p, ctext)
        for h in p.branch_handles(records(p, 400)):
            c.add_branch(h)

    return {"carried_free_x": free_x,
            "carried_branch_x": lambda: add_all(E0.PER_E, "INPUT_QUBITS in 0\nT 0\nMX 0\nDETECTOR rec[-1]\n"),
            "branch_frame": lambda: add_all("RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\n"
                                            "OUTPUT_QUBITS out 0\n", "INPUT_QUBITS in 0\nI 0\n"),
            "consumer_t_depth_2": lambda: consumer(producer("R 0\nH 0\nOUTPUT_QUBITS out 0\n"),
                                                   "INPUT_QUBITS in 0\nR 1\nH 1\nT 1\nH 1\nT 1\nMX 1\n"
                                                   "DETECTOR rec[-1]\nMX 0\nDETECTOR rec[-1]\n")}


@pytest.mark.parametrize("api", ["public", "testapi"])
@pytest.mark.parametrize("sub", ["carried_free_x", "carried_branch_x", "branch_frame", "consumer_t_depth_2"])
def test_carried_refusals_typed_without_exact(api, sub):
    from xtim._exact_scope import exact_constructions
    case = _carried_cases(api)[sub]
    a = exact_constructions()
    with pytest.raises(XtimTwirlRefusal) as ei:
        case()
    want = TwirlClass.OUTSIDE_T_DEPTH_1 if sub == "consumer_t_depth_2" else TwirlClass.LAW_LIMITATION_CARRIED_AXIS
    assert ei.value.twirl_class == want, (api, sub, ei.value.twirl_class)
    if sub != "consumer_t_depth_2":
        assert ei.value.sub_reason == sub, (api, sub, ei.value.sub_reason)
    assert exact_constructions() == a, (api, sub)


def test_foreign_records_are_refused_by_type():
    """J-b1 final review m1: a test-API FamilyRecords object never reaches the production readers (module_local), so
    the per-module sampler serial (salted per extension) cannot short-circuit the bit-exact model check."""
    from xtim import _xtim_testapi as TA
    prod = xtim.compile_twirl_sampler("RX 0\nT 0\nOUTPUT_QUBITS out 0\n", selfcheck=0)
    ta = TA.compile_family_sampler("RX 0\nR 1\nT 0\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n",
                                   selfcheck=0)
    R = ta.sample_records(10, seed=1)[-1]
    with pytest.raises(TypeError):
        prod.branch_handles(R)
    assert "_born_trie_check" not in dir(prod) and "_born_trie_oracle" not in dir(prod)   # m6: test API only
