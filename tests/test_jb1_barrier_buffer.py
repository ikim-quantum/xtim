"""J-b1 P12: the layout-3 BarrierBuffer and RecordFidelity (cpp/bindings/record_buffer.hpp), bound in the test API as
the classes F will expose (py::module_local; plan r2.3 / r3.3).

Gates:
  * the buffer's columns, keys, groups, outcomes, plan contents, frame_folded, frames(), signs() and materialize
    equal the FamilyRecords readers on the same run; dets / obs / decisions equal sample() at the same seed;
  * σ is absent: sigmas / sig_bits / plan_key / exact_flags / coins do not exist (AttributeError);
  * RecordFidelity.evaluate(buffer) reads every shot's own record and equals FamilyRecordFidelity on the same records;
  * the per-shot output-Pauli correction (`correction=`, uint8[shots, 2, n_out] on OUTPUT_QUBITS) gives the same
    values through both evaluators (the register-fidelity oracles: tests/test_rfref_route.py, test_rfref_correction.py);
  * (the CCZ factory decoded-fidelity gate is OPEN: fixplan-p12-fidelity);
  * the two extensions import in either order, and a test-API buffer handed to another module's evaluator is
    refused by name.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-12
CH = open(os.path.join(ROOT, "tests", "data", "ch_cultivation.stim")).read()
CH_OUT = "\n".join(line for line in CH.splitlines() if not line.startswith("PAULI_EXPECTATION")) + "\nOUTPUT_QUBITS out 0\n"


def fam(text, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0, **kw)


def test_buffer_equals_the_record_readers():
    s = fam(CH)
    b = s.sample_barrier(300, seed=11)
    d, o, dc, R = s.sample_records(300, seed=11)
    for x, y in ((b.dets(), d), (b.obs(), o), (b.decisions(), dc)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    d2, o2, dc2 = s.sample(300, seed=11)
    assert np.array_equal(np.asarray(b.dets()), np.asarray(d2))
    keys = b.record_keys()
    assert keys == [R.key(i) for i in range(300)]
    gid = np.asarray(b.record_group_ids())
    gk, cnt = b.record_groups()
    assert int(np.sum(cnt)) == 300 and all(keys[i] == gk[gid[i]] for i in range(300))
    for i in range(0, 300, 17):
        assert np.array_equal(b.outcomes(i), R.outcomes(i)) and np.array_equal(b.record_outcomes(i), R.y(i))
        assert b.plan_content(i) == R.plan_content(i) and b.frame_folded(i) == R.frame_folded(i)
        # J-b1 F (fixplan-F-materialize-c0hat r2): the buffer's materialize is the handle on the gate-level route
        m1, m2 = b.materialize(i).branch_frames(), s.materialize(R, i, True)
        assert all(np.array_equal(np.asarray(m1[k]), np.asarray(m2[k])) for k in m2)
    f1 = b.frames()
    assert b.frames() is not None and np.array_equal(f1["plan_index"], s.frames(R)["plan_index"])
    g = list(TA.certified_generators(s))[:6]
    assert np.array_equal(np.asarray(b.signs(g)), np.asarray(s.signs(R, g)))
    assert b.record_layout() == s.record_layout()
    assert b._engine_module().__name__ == "xtim._xtim_testapi"


@pytest.mark.parametrize("name", ["sigmas", "sig_bits", "plan_key", "exact_flags", "coins", "sigma", "frame_dressing",
                                  "input_frame_advance", "input_frame_dressing", "record_tables_held"])
def test_sigma_is_absent(name):
    """σ is absent, and the b9c2e92 methods listed REMOVED in plan §4.1 (P12 ruling) raise AttributeError."""
    b = fam(CH).sample_barrier(20, seed=1)
    with pytest.raises(AttributeError):
        getattr(b, name)


def test_record_fidelity_equals_the_family_evaluator():
    """The production-bound layout-3 evaluator equals the test API's FamilyRecordFidelity on the same records, plain and
    decoded (random output corrections)."""
    s = fam(CH_OUT)
    b = s.sample_barrier(200, seed=13)
    d, o, dc, R = s.sample_records(200, seed=13)
    F_new = np.asarray(s.record_fidelity().evaluate(b))
    F_old = np.asarray(TA.FamilyRecordFidelity(s).evaluate(s, R))
    assert np.array_equal(F_new, F_old)
    c = np.random.default_rng(3).integers(0, 2, (200, 2, 1)).astype(np.uint8)
    Fc = np.asarray(s.record_fidelity().evaluate(b, correction=c))
    assert np.array_equal(Fc, np.asarray(TA.FamilyRecordFidelity(s).evaluate(s, R, correction=c)))
    assert np.any(np.abs(Fc - F_new) > 1e-6)


def test_two_modules_import_in_either_order():
    code = ("import importlib, sys\n"
            "mods = sys.argv[1:]\n"
            "for m in mods: importlib.import_module(m)\n"
            "from xtim import _xtim_testapi as TA\n"
            "s = TA.compile_family_sampler('R 0\\nX_ERROR(0.1) 0\\nM 0\\nDETECTOR rec[-1]\\n', selfcheck=0)\n"
            "b = s.sample_barrier(10, seed=1)\n"
            "assert b.num_shots == 10 and type(b).__module__ == 'xtim._xtim_testapi'\n")
    for order in (["xtim._xtim", "xtim._xtim_testapi"], ["xtim._xtim_testapi", "xtim._xtim"]):
        r = subprocess.run([sys.executable, "-c", code] + order, cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]


def test_cross_module_buffer_is_refused():
    s = fam(CH_OUT)
    rf = s.record_fidelity()
    with pytest.raises(TypeError, match="another extension"):
        rf.evaluate(object())


def test_record_groups_reader():
    """xtim._record_groups (port.py's layout-3 parser, duck-typed on the buffer): every shot of a group has the
    group's plan content, outcomes, P_rec and signs; group_plan's kind-0 view matches FamilyRecords.plan."""
    from xtim._record_groups import RecordGroups
    s = fam(CH)
    b = s.sample_barrier(400, seed=17)
    d, o, dc, R = s.sample_records(400, seed=17)
    G = RecordGroups(b)
    assert G.n_groups == len(G.group_keys) and int(G.group_counts.sum()) == 400
    pc, outc, pre = G.group_plan_content(), G.group_outcomes(), G.group_prefix()
    g = list(TA.certified_generators(s))[:5]
    sg = G.group_signs(g)
    allsg = np.asarray(b.signs(g))
    for i in range(400):
        k = G.group_id[i]
        content, digest = b.plan_content(i)
        assert pc[k][2] == digest and pc[k][1] == content[5:]
        assert np.array_equal(outc[k], b.outcomes(i))
        x, z = R.prec(i)
        n = len(x)
        W = pre.shape[1] // 2
        bits = np.unpackbits(pre[k].astype("<u8").view(np.uint8), bitorder="little")
        assert np.array_equal(bits[:n], x) and np.array_equal(bits[64 * W:64 * W + n], z)
        assert np.array_equal(sg[k], allsg[i])
    for k in range(min(G.n_groups, 20)):
        p = G.group_plan(k)
        q = R.plan(int(G.group_first_shot[k]))
        if p["kind"] == 0:
            assert p["a"] == list(q["a"]) and p["cz"] == [tuple(t) for t in q["cz"]]


def test_partitioned_record_report_shape():
    """Plan §1.7 (ruled): the reference pattern's static schema, plus route counts / counters summed over the inner
    samplers built so far, plus `patterns`. J-b1 F: the public PartitionedTwirlSampler (its inner samplers are the
    family sampler; compile_family_partitioned / _make_inner removed, m5)."""
    text = F.IF_TEXT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = xtim.compile_twirl_sampler(text, selfcheck=0)
    p.sample(400, seed=3, decision_bits=np.array([[0], [1]] * 200, np.uint8), return_mask=True)
    r = p.record_report()
    assert set(r) >= {"classes", "nondeterministic_detectors", "nondeterministic_observables", "record_layout",
                      "route_counts", "counters", "patterns"}
    inner = list(p._sampler_cache.values())
    assert r["patterns"] == len(inner) >= 1
    assert sum(r["route_counts"].values()) == sum(sum(s.record_report()["route_counts"].values()) for s in inner)
    assert r["classes"] == p._ref_sampler.record_report()["classes"]



def test_b9c2e92_methods_on_layout3():
    """P12 ruling (fixplan-p12-review): state / branch_frames (aliases of materialize), plan_structure (no σ fields),
    born_dec (pure data, = production's), pauli_expectations_all (every shot's state read on literal Paulis = the
    per-shot materialize oracle), compact=True accepted / compact=False refused by name."""
    text = open(os.path.join(ROOT, "tests", "data", "adaptq_born5_dec_enum_producer.stim")).read()
    s = fam(text)
    b = s.sample_barrier(60, seed=5, compact=True)
    d, o, dc, R = s.sample_records(60, seed=5)
    for i in (0, 7, 31):
        st = b.materialize(i)                          # J-b1 F r2: the FramedSuperposition handle (as b9c2e92)
        assert type(st).__name__ == "FramedSuperposition" and type(b.state(i)) is type(st)
        m = st.branch_frames()
        for alias in (b.state(i).branch_frames(), b.branch_frames(i)):
            assert all(np.array_equal(np.asarray(alias[k]), np.asarray(m[k])) for k in m)
        ps, ref = b.plan_structure(i), s.record_plan_structure(R, i)
        assert set(ps) == set(ref)
        for k in ref:
            a, c = ref[k], ps[k]
            if isinstance(a, (list, tuple)) and a and isinstance(a[0], np.ndarray):
                assert len(a) == len(c) and all(np.array_equal(u, v) for u, v in zip(a, c)), k
            else:
                assert np.array_equal(np.asarray(a, dtype=object), np.asarray(c, dtype=object)) or a == c, k
        assert not {"sigma", "sig_bits", "sigmas", "r", "kappa", "fallback"} & set(ps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prod = xtim._xtim.TwirlSampler(text, 1.0, "", 0)
    pb = prod.sample_barrier(10, seed=1, compact=True).born_dec()
    fb = b.born_dec()
    assert set(fb) == set(pb) and fb["nborn"] == pb["nborn"] and fb["nborn"] > 0
    for k in ("dec_index", "xz", "phase", "inv"):
        assert np.array_equal(np.asarray(fb[k]), np.asarray(pb[k])), k
    n = len(s.record_ops()[0][0])
    reads = [([0], [], 0), ([], [0, 1], 0), ([2], [2], 1)]
    E = np.asarray(b.pauli_expectations_all(reads))
    assert E.shape == (60, 3)
    for i in range(0, 60, 9):
        P = []
        for xs, zs, ph in reads:
            x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8)
            x[xs] = 1; z[zs] = 1
            P.append((x, z, ph))
        assert np.allclose(E[i].real, s.state_expectations(R, i, P), atol=1e-12)
    with pytest.raises(ValueError, match="compact=False"):
        s.sample_barrier(5, seed=1, compact=False)
