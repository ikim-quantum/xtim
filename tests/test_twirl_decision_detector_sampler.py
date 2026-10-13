"""Twirl/auto detector samplers on circuits that declare DECISIONs (dec-fix1, root fix).

``compile_detector_sampler(engine="twirl"/"auto")`` compiles its twirl engine with
``decisions=False``: no DECISION output channel exists, so the engine is exactly the engine of
the same circuit with its DECISION lines removed (the exact detector sampler has no decision
output either; a DECISION is never consumed inside the engine -- IF conditions read INPUT_BITS
supplied by the caller). Consequences tested here:

* shapes/dtypes equal exact for every kwarg; (det, obs) law equals the exact engine's;
* CS + Born-decision circuits (the reviewer's circ9 class), which the decision-compiled engine
  refuses at sample() ("Born decisions unsupported for CH/PPR circuits"), sample under
  twirl/auto and match the dense oracle law (tests/_forced_oracle, full fault enumeration);
* inertness: seeded det/obs streams are byte-identical with vs without the DECISION lines;
* engine_report() never says "twirl" for a sampler whose sample() raises (IF circuits route
  auto -> exact; explicit twirl refuses at compile);
* mechanism + mutation: the detector sampler's engine has num_decisions == 0, and the same
  circuit compiled with decisions=True (the pre-fix detector-sampler engine) raises -- so
  re-enabling decision channels in compile_detector_sampler fails the circ9 tests;
* compile_twirl_sampler_from_state wraps DECISION circuits like compile_twirl_sampler;
* the circuit-channels banner is opt-in (XTIM_TWIRL_VERBOSE=1).
"""
import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

import xtim
from xtim import _xtim, twirl as _twirl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _forced_oracle as orc  # noqa: E402

REPRO = "RX 0\nT 0\nX_ERROR(0.1) 0\nMX 0\nDECISION(0) rec[-1]\nM 1\nDETECTOR rec[-1]\n"
# deterministic decision + observable + noise
DET_DEC = ("R 0 1\nX_ERROR(0.2) 0\nM 0 1\nDECISION(0) rec[-2]\nDETECTOR rec[-1]\n"
           "OBSERVABLE_INCLUDE(0) rec[-2]\n")
# Born-random decision, deterministic observable and detector
BORN_OBS = ("RX 0\nR 1\nX_ERROR(0.15) 1\nMX 0\nDECISION(0) rec[-1]\nM 1\n"
            "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
# CS + Born decision (3 qubits): the decision-compiled engine refuses these at sample().
CS_BORN = ("RX 0\nRX 1\nR 2\nCS 0 1\nX_ERROR(0.1) 2\nMX 0\nDECISION(0) rec[-1]\n"
           "M 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
CIRC9 = """R 0
RX 1
R 2
H 0
T_DAG 2
CZ 0 1
CZ 2 0
Y_ERROR(0.3) 2
CS 0 2
Z 1
T 2
X_ERROR(0.1) 2
MX 1
M 0
MX 2
DECISION(0) rec[-1]
Y_ERROR(0.1) 2
Y_ERROR(0.1) 0
M 0
DETECTOR rec[-1] rec[-3]
MX 2
DETECTOR rec[-1] rec[-3]
MX 1
DETECTOR rec[-1] rec[-6]
DECISION(1) rec[-1] rec[-6]
OBSERVABLE_INCLUDE(0) rec[-2] rec[-4]
"""
CIRC2 = """RX 0
R 1
R 2
CS 2 0
S 1
X_ERROR(0.2) 1
T_DAG 1
T 1
CZ 2 1
Z 0
Z_ERROR(0.3) 2
MX 1
M 2
M 0
DECISION(0) rec[-3]
DECISION(1) rec[-2]
DECISION(2) rec[-1]
Z_ERROR(0.2) 0
Z_ERROR(0.2) 1
MX 1
DETECTOR rec[-1] rec[-4]
DECISION(3) rec[-1] rec[-4]
M 2
DETECTOR rec[-1] rec[-4]
M 0
DETECTOR rec[-1] rec[-4]
DECISION(4) rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-2] rec[-5]
"""
CIRC13 = """R 0
R 1
R 2
X 0
T 1
S 0
S_DAG 1
DEPOLARIZE1(0.2) 0
X 2
CZ 0 2
DEPOLARIZE1(0.2) 0
M 1
M 2
M 0
DECISION(0) rec[-2]
DECISION(1) rec[-1]
DEPOLARIZE1(0.3) 1
X_ERROR(0.3) 2
M 1
DETECTOR rec[-1] rec[-4]
DECISION(2) rec[-1] rec[-4]
M 2
DETECTOR rec[-1] rec[-4]
DECISION(3) rec[-1] rec[-4]
M 0
DETECTOR rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-6]
"""
CIRC16 = """RX 0
R 1
R 2
CS 1 0
T 1
S_DAG 2
S 2
Y_ERROR(0.3) 2
T_DAG 2
H 1
S 1
CZ 1 0
DEPOLARIZE1(0.1) 2
MX 1
MX 2
MX 0
DECISION(0) rec[-3]
DECISION(1) rec[-2]
DECISION(2) rec[-1]
X_ERROR(0.1) 1
MX 0
DETECTOR rec[-1] rec[-2]
MX 1
DETECTOR rec[-1] rec[-5]
MX 2
DETECTOR rec[-1] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-2] rec[-6]
"""
IF_CIRC = ("INPUT_BITS b 1\nR 0\nX_ERROR(0.1) 0\nIF b[0] {\nX 0\n}\nM 0\n"
           "DETECTOR rec[-1]\n")
CIRCS = {"repro": REPRO, "det_dec_obs": DET_DEC, "born_obs": BORN_OBS, "cs_born": CS_BORN,
         "circ9": CIRC9, "circ2": CIRC2, "circ13": CIRC13}
CS_CIRCS = {"cs_born": CS_BORN, "circ9": CIRC9, "circ2": CIRC2}
N = 40000


def strip_dec(text):
    return "".join(l + "\n" for l in text.splitlines() if not l.startswith("DECISION"))


def oracle_law(text, nq=3):
    """{(det, obs): p} over the full fault enumeration (dense oracle)."""
    sites = []
    for raw in text.splitlines():
        name = raw.split("(")[0].strip()
        if name in orc.NOISE1:
            p = float(raw.split("(")[1].split(")")[0])
            if name == "DEPOLARIZE1":
                sites.append([("I", 1 - p), ("X", p / 3), ("Y", p / 3), ("Z", p / 3)])
            else:
                sites.append([("I", 1 - p), (name[0], p)])
    law = {}
    for combo in itertools.product(*sites):
        w = 1.0
        faults = {}
        for s, (pa, pr) in enumerate(combo):
            w *= pr
            if pa != "I":
                faults[(s, 0)] = pa
        for o in orc.branches(text, nq, faults=faults).values():
            k = (tuple(o["det"]), tuple(o["obs"]))
            law[k] = law.get(k, 0.0) + w * o["p"]
    assert abs(sum(law.values()) - 1) < 1e-9
    return law


def best_shift_z(dets, obs, law):
    """max |z| and support violations under the best constant reference XOR shift."""
    D, O = dets.shape[1], obs.shape[1]
    cols = np.hstack([dets, obs]).astype(np.int64)
    n = len(cols)
    cnt = np.bincount(cols @ (1 << np.arange(D + O)), minlength=1 << (D + O))
    best = None
    for sd in itertools.product((0, 1), repeat=D):
        for so in itertools.product((0, 1), repeat=O):
            zmax, viol = 0.0, 0
            for idx in range(1 << (D + O)):
                bits = [(idx >> i) & 1 for i in range(D + O)]
                k = (tuple(b ^ s for b, s in zip(bits[:D], sd)),
                     tuple(b ^ s for b, s in zip(bits[D:], so)))
                p = law.get(k, 0.0)
                if p <= 0:
                    viol += int(cnt[idx])
                    continue
                if p < 1:
                    zmax = max(zmax, abs((cnt[idx] / n - p) / np.sqrt(p * (1 - p) / n)))
            if best is None or (viol, zmax) < best:
                best = (viol, zmax)
    return best[1], best[0]


@pytest.mark.parametrize("name", sorted(CIRCS))
@pytest.mark.parametrize("engine", ["twirl", "auto"])
def test_matches_exact(name, engine):
    c = xtim.Circuit(CIRCS[name])
    ex = c.compile_detector_sampler(seed=3, engine="exact")
    tw = c.compile_detector_sampler(seed=3, engine=engine)
    assert tw.engine_report()["engine"] == "twirl"
    # plain
    a, b = ex.sample(N), tw.sample(N)
    assert a.shape == b.shape and a.dtype == b.dtype
    # separate / append / bit_packed
    for kw in ({"separate_observables": True}, {"append_observables": True},
               {"bit_packed": True},
               {"separate_observables": True, "bit_packed": True},
               {"append_observables": True, "bit_packed": True}):
        ra, rb = ex.sample(64, **kw), tw.sample(64, **kw)
        if isinstance(ra, tuple):
            assert len(ra) == len(rb)
            for x, y in zip(ra, rb):
                assert x.shape == y.shape and x.dtype == y.dtype, kw
        else:
            assert ra.shape == rb.shape and ra.dtype == rb.dtype, kw
    # statistics: marginals and joint
    da, oa = ex.sample(N, separate_observables=True)
    db, ob = tw.sample(N, separate_observables=True)
    ca, cb = np.hstack([da, oa]).astype(int), np.hstack([db, ob]).astype(int)
    tol = 5 * np.sqrt(0.25 / N * 2)
    assert np.all(np.abs(ca.mean(0) - cb.mean(0)) < tol)
    w = 1 << np.arange(ca.shape[1])
    ja = np.bincount(ca @ w, minlength=1 << ca.shape[1]) / N
    jb = np.bincount(cb @ w, minlength=1 << ca.shape[1]) / N
    assert np.all(np.abs(ja - jb) < tol)


@pytest.mark.parametrize("name", sorted(CS_CIRCS))
@pytest.mark.parametrize("engine", ["twirl", "auto", "exact"])
def test_cs_born_decision_matches_dense_oracle(name, engine):
    """CS + Born-decision circuits: the engine the detector sampler compiles has no decision
    channel, so it samples (the decision-compiled engine refuses, see
    test_decision_compiled_engine_refuses_cs_born) and its law is the dense oracle's."""
    text = CS_CIRCS[name]
    law = oracle_law(text)
    s = xtim.Circuit(text).compile_detector_sampler(seed=11, engine=engine)
    assert s.engine_report()["engine"] == ("exact" if engine == "exact" else "twirl")
    d, o = s.sample(100000, separate_observables=True)
    z, viol = best_shift_z(d, o, law)
    assert viol == 0
    assert z < 4.5, z


@pytest.mark.parametrize("name", sorted(CIRCS))
@pytest.mark.parametrize("engine", ["twirl", "auto"])
def test_decision_lines_inert_byte_identical(name, engine):
    """Seeded det/obs streams with vs without the DECISION lines: byte-identical."""
    a = xtim.Circuit(CIRCS[name]).compile_detector_sampler(seed=9, engine=engine)
    b = xtim.Circuit(strip_dec(CIRCS[name])).compile_detector_sampler(seed=9, engine=engine)
    da, oa = a.sample(20000, separate_observables=True)
    db, ob = b.sample(20000, separate_observables=True)
    assert np.array_equal(da, db) and np.array_equal(oa, ob)
    assert a.channel_report()["exact_shots"] == b.channel_report()["exact_shots"]


def test_detector_sampler_engine_has_no_decision_channel():
    """Mechanism: the detector sampler's engine is compiled with decisions=False."""
    for engine in ("twirl", "auto"):
        s = xtim.Circuit(CIRC9).compile_detector_sampler(seed=1, engine=engine)
        assert isinstance(s._s, _xtim.TwirlSampler)       # raw engine, no decision wrapper
        assert s._s.num_decisions == 0
        assert len(s._s.sample(8, seed=1)) == 2            # the decision-free 2-tuple


def test_compile_twirl_sampler_decisions_flag():
    """decisions=False == the engine of the DECISION-stripped circuit (bytes); the default
    keeps the TwirlBatch surface with .decisions."""
    off = _twirl.compile_twirl_sampler(CIRC13, selfcheck=0, decisions=False)
    ref = _twirl.compile_twirl_sampler(strip_dec(CIRC13), selfcheck=0)
    assert isinstance(off, _xtim.TwirlSampler) and off.num_decisions == 0
    a, b = off.sample(5000, seed=2), ref.sample(5000, seed=2)
    assert len(a) == 2 and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    on = _twirl.compile_twirl_sampler(CIRC13, selfcheck=0)
    batch = on.sample(5000, seed=2)
    assert on.num_decisions == 4 and batch.decisions is not None
    assert np.array_equal(batch.dets, a[0]) and np.array_equal(batch.obs, a[1])


def test_decision_compiled_engine_refuses_cs_born_at_compile():
    """The pre-fix detector-sampler engine (decisions compiled) cannot serve a Born-class
    DECISION on a CH/PPR circuit: it now refuses AT COMPILE, by name (was an "oracle abort"
    RuntimeError at the first sample()). The detector sampler no longer builds it (mutation
    anchor for the circ9 tests above); decisions=False compiles and samples."""
    with pytest.raises(ValueError, match="Born-class DECISION on a CH/PPR circuit"):
        _twirl.compile_twirl_sampler(CIRC9, selfcheck=0)
    with pytest.raises(ValueError, match="Born-class DECISION on a CH/PPR circuit"):
        _twirl.compile_twirl_sampler(CIRC16, selfcheck=0)
    off = _twirl.compile_twirl_sampler(CIRC9, selfcheck=0, decisions=False)
    assert len(off.sample(100, seed=1)) == 2
    xtim.Circuit(CIRC9).compile_detector_sampler(seed=1, engine="twirl").sample(100)
    # det-class decisions on a CS circuit (circ2) and Born decisions on a diagonal circuit
    # (circ13) are unaffected: still compiled and sampled with decisions.
    for text, ndec in ((CIRC2, 5), (CIRC13, 4)):
        s = _twirl.compile_twirl_sampler(text, selfcheck=0)
        assert s.num_decisions == ndec
        assert s.sample(100, seed=1).decisions is not None


def test_engine_report_agrees_with_sample():
    """engine_report()['engine'] == 'twirl' implies sample() does not raise."""
    for text in list(CIRCS.values()) + [strip_dec(CIRC9)]:
        s = xtim.Circuit(text).compile_detector_sampler(seed=1, engine="auto")
        assert s.engine_report()["engine"] == "twirl"
        s.sample(100)


IF_CIRC_NOT = ("INPUT_BITS b 2\nR 0\nX_ERROR(0.1) 0\nIF !b[1] {\nX 0\n}\nM 0\n"
               "DETECTOR rec[-1]\n")


def test_if_blocks_without_bits_refuse_by_name_on_every_surface():
    """IF bodies are per-shot conditionals on caller-supplied INPUT_BITS. No surface under
    xtim.Circuit takes those bits, and the exact engine / DEM export would otherwise run
    every IF body unconditionally (undocumented): every such surface refuses at compile,
    naming the IF condition and the escape. With bits supplied (resolve_branches_text /
    PartitionedTwirlSampler.sample(decision_bits=...)) behaviour is unchanged."""
    for text, cond in ((IF_CIRC, "IF b[0]"), (IF_CIRC_NOT, "IF b[1]")):
        c = xtim.Circuit(text)
        for engine in ("exact", "twirl", "auto"):
            with pytest.raises(ValueError, match="IF blocks on INPUT_BITS") as ei:
                c.compile_detector_sampler(seed=1, engine=engine)
            assert cond in str(ei.value) and "resolve_branches_text" in str(ei.value)
        with pytest.raises(ValueError, match="IF blocks on INPUT_BITS"):
            c.compile_sampler(seed=1)
        with pytest.raises(ValueError, match="IF blocks on INPUT_BITS"):
            c.detector_error_model_text()
        with pytest.raises(ValueError, match="IF blocks on INPUT_BITS"):
            c.detector_error_model_with_reject()
        # twirl partition sampler: refuses at sample() without bits (pre-existing, by name).
        # (Its sample_barrier delegation to the all-zeros pattern is left as is: adaptq's
        # Choi harness may reach it -- owner decision, see CHANGELOG.)
        ps = _twirl.compile_twirl_sampler(text, selfcheck=0)
        assert isinstance(ps, _twirl.PartitionedTwirlSampler)
        with pytest.raises(ValueError, match="decision_bits must be provided"):
            ps.sample(10, seed=1)
    # Bits supplied (unchanged): the resolved texts are what each branch means (X 0 present
    # iff the bit is set), each compiles on every engine, and the partition sampler serves
    # both patterns per shot. Detectors are reference-relative, so both branches report the
    # X_ERROR flip with p = 0.1 (the exact law of either resolved circuit).
    n = 4000
    bits = np.zeros((2 * n, 1), np.uint8)
    bits[n:, 0] = 1
    ps = _twirl.compile_twirl_sampler(IF_CIRC, selfcheck=0)
    dets_p, _obs_p = ps.sample(2 * n, seed=1, decision_bits=bits)
    dets = np.unpackbits(dets_p, axis=1, bitorder="little")[:, 0]
    r0 = xtim.Circuit(_xtim.resolve_branches_text(IF_CIRC, [0]))
    r1 = xtim.Circuit(_xtim.resolve_branches_text(IF_CIRC, [1]))
    assert "X 0" not in r0.text and "X 0" in r1.text
    tol = 5 * np.sqrt(0.09 / n)
    assert abs(dets[:n].mean() - 0.1) < tol and abs(dets[n:].mean() - 0.1) < tol
    for r in (r0, r1):
        for engine in ("exact", "twirl", "auto"):
            d = r.compile_detector_sampler(seed=1, engine=engine).sample(n)
            assert abs(d.mean() - 0.1) < tol
        r.detector_error_model_text()
    # INPUT_BITS without any IF block drives nothing: not refused.
    xtim.Circuit("INPUT_BITS b 1\nR 0\nM 0\nDETECTOR rec[-1]\n").compile_detector_sampler(
        seed=1, engine="exact").sample(4)


def test_exact_law_deterministic_decision_circuit():
    # X_ERROR on qubit 0 never touches rec[-1] (qubit 1) -> the detector is identically 0;
    # observable = rec[-2] = the X_ERROR flip of qubit 0 -> p = 0.2 (exact law).
    c = xtim.Circuit(DET_DEC)
    for eng in ("twirl", "auto"):
        d, o = c.compile_detector_sampler(seed=5, engine=eng).sample(
            N, separate_observables=True)
        assert d.sum() == 0
        assert abs(o.mean() - 0.2) < 5 * np.sqrt(0.16 / N)


def test_from_state_wraps_decisions_like_compile_twirl_sampler():
    a = _twirl.compile_twirl_sampler(REPRO, selfcheck=0)
    b = _twirl.compile_twirl_sampler_from_state(_twirl._bare_state_of(REPRO), REPRO,
                                                selfcheck=0)
    ra, rb = a.sample(2000, seed=4), b.sample(2000, seed=4)
    assert isinstance(ra, _twirl.TwirlBatch) and isinstance(rb, _twirl.TwirlBatch)
    assert np.array_equal(ra.dets, rb.dets) and np.array_equal(ra.obs, rb.obs)
    assert np.array_equal(ra.decisions, rb.decisions)
    off = _twirl.compile_twirl_sampler_from_state(_twirl._bare_state_of(REPRO), REPRO,
                                                  selfcheck=0, decisions=False)
    assert isinstance(off, _xtim.TwirlSampler) and off.num_decisions == 0


def test_quiet_by_default_and_opt_in():
    code = ("import xtim\nc=xtim.Circuit(%r)\n"
            "c.compile_detector_sampler(engine='twirl').sample(4)\n" % REPRO)
    env = {k: v for k, v in os.environ.items()
           if k not in ("XTIM_QUIET", "XTIM_TWIRL_VERBOSE")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 0, r.stderr
    assert "circuit channels" not in r.stderr
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(env, XTIM_TWIRL_VERBOSE="1"))
    assert "circuit channels" in r.stderr


def test_quiet_wins_over_verbose():
    """Precedence: XTIM_QUIET=1 silences the banner even with XTIM_TWIRL_VERBOSE=1."""
    code = ("import xtim\nc=xtim.Circuit(%r)\n"
            "c.compile_detector_sampler(engine='twirl').sample(4)\n" % REPRO)
    env = {k: v for k, v in os.environ.items()
           if k not in ("XTIM_QUIET", "XTIM_TWIRL_VERBOSE")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(env, XTIM_TWIRL_VERBOSE="1", XTIM_QUIET="1"))
    assert r.returncode == 0, r.stderr
    assert "circuit channels" not in r.stderr
