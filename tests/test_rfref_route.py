"""RF-ref: the REGISTER RecordFidelity (owner rulings 2026-10-22: the STRICT output contract; review adoption (i)-(v)).

F = <phi| P_c Tr_R(rho_shot) P_c |phi> on OUTPUT_QUBITS (ancillas and read wires traced out). Compile-time check:
Theorem 1 on the noiseless bare stabilizer group S0 (no records) — the bare state is a product across output | rest —
and records off the output wires; refused by name otherwise (RF_OUTPUT_NOT_STRICT with the undetermined directions,
(a, b, c, e) and the measurements that fix them; RF_NO_OUTPUT; RF_RECORD_ON_OUTPUT in the C++ oracle). Routes: a = 0 ->
3a (one rotation-free projector expectation), a > 0 -> 3b (the rest-sector sum of the Gauss-sum kernel, the sectors
solved per shot). Oracle: tests/_register_oracle.py, a dense partial trace (test-only); the C++ statevector oracle is
cpp/tests/test_record_fidelity_ref.cpp.
"""
import os
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from xtim.record_fidelity import RecordFidelity
from _register_oracle import bare_rho, purity, register_F
from _private import private_fixture

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
TOL = 1e-12


def _compile(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(text, selfcheck=0)


def _fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def _kappa_text(B, p=0.3, reads="MX"):
    L = []
    for b in range(B):
        L.append(f"RX {4 * b}")
    for b in range(B - 1, 0, -1):
        L.append(f"CX {4 * b} {4 * (b - 1)}")
    for b in range(B):
        for d in (4 * b + 1, 4 * b + 2):
            L += [f"RX {d}", f"X_ERROR({p}) {d}", f"T {d}"] + ([f"H {d}"] if b != 2 else []) + ([f"X {d}"] if b == 1 else [])
        L.append(f"R {4 * b + 3}")
    for b in range(B):
        a0, d1, d2, a3 = 4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3
        L += [f"CX {a0} {d1}", f"CX {a0} {d2}", f"CX {d1} {a3}", f"CX {d2} {a3}", f"CX {a0} {a3}"]
    L.append(f"X_ERROR(0.01) {4 * B - 1}")
    for b in range(B):
        q = [4 * b + i for i in range(4)]
        L += ["MPP " + "*".join(f"X{i}" for i in q), "DETECTOR rec[-1]", "MPP " + "*".join(f"Z{i}" for i in q),
              "DETECTOR rec[-1]"]
    k = 0
    for b in range(B):
        for q in (4 * b + 1, 4 * b + 2):
            L += [f"{reads} {q}", f"DECISION({k}) rec[-1]"]
            k += 1
    L.append("OUTPUT_QUBITS out " + " ".join(str(x) for x in [4 * b for b in range(B)] + [4 * b + 3 for b in range(B)]))
    return "\n".join(L) + "\n"


# the Bell pair with its frame CLOSED (the Born M 0 fed back onto 1), a magic output wire and an unrelated detector
_BELL = ("R 0 1 2\nH 0\nCX 0 1\nRX 2\nT 2\n{err}\nM 0\nCX rec[-1] 1\nDECISION(0) rec[-1]\n"
         "R 3\nX_ERROR(0.25) 3\nM 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1 2\n")
_BELL_OPEN = "R 0 1 2\nH 0\nCX 0 1\nRX 2\nT 2\nM 0\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1 2\n"
# a = 0 (route 3a): a stabilizer output, magic and Born reads on the rest
_STAB = ("R 0\nDEPOLARIZE1(0.3) 0\nRX 1 2\nT 1\nCZ 1 2\nT_DAG 2\nX_ERROR(0.1) 1\nMX 1\nDECISION(0) rec[-1]\nMX 2\n"
         "DECISION(1) rec[-1]\nR 3\nX_ERROR(0.2) 3\nM 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n")
_B1 = "R 0 1\nH 0\nT 0\nCX 0 1\nM 1\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0\n"


def _random_text(rng, nq):
    L = ["RX " + " ".join(map(str, range(nq)))]
    for _layer in range(int(rng.integers(2, 5))):
        g = rng.choice(["H", "S", "CX", "CZ", "T", "DEP"])
        if g in ("H", "S", "T"):
            L.append(f"{g} {rng.integers(nq)}")
        elif g == "DEP":
            L.append(f"DEPOLARIZE1(0.2) {rng.integers(nq)}")
        else:
            a, b = rng.choice(nq, 2, replace=False)
            L.append(f"{g} {a} {b}")
    L.append(f"T {rng.integers(nq)}")
    L.append("DEPOLARIZE1(0.15) " + " ".join(map(str, range(nq))))
    meas = sorted(rng.choice(nq, int(rng.integers(1, nq - 1)), replace=False).tolist())
    for k, q in enumerate(meas):
        L += [f"{rng.choice(['M', 'MX', 'MY'])} {q}", f"DECISION({k}) rec[-1]"]
    L.append("OUTPUT_QUBITS out " + " ".join(str(q) for q in range(nq) if q not in meas))
    return "\n".join(L) + "\n"


def _fixtures():
    ch = open(os.path.join(ROOT, "tests", "data", "ch_cultivation.stim")).read()
    ch = "\n".join(line for line in ch.splitlines() if not line.startswith("PAULI_EXPECTATION")) + "\n"
    fx = [("bell_clean", _BELL.format(err="")), ("bell_XX", _BELL.format(err="X_ERROR(1) 0 1")),
          ("bell_X0", _BELL.format(err="X_ERROR(1) 0")), ("bell_dep", _BELL.format(err="DEPOLARIZE1(0.3) 1 2")),
          ("ch_noisy", ch.replace("0.001", "0.05") + "OUTPUT_QUBITS out 0\n"),
          ("ch_dep", ch.replace("0.001", "0.05") + "DEPOLARIZE1(0.2) 0\nOUTPUT_QUBITS out 0\n"),
          ("stab", _STAB)]
    rng = np.random.default_rng(11)
    nr = 0
    while nr < 8:                                    # random circuits that meet the strict contract
        txt = _random_text(rng, int(rng.integers(3, 6)))
        try:
            RecordFidelity(_compile(txt))
        except Exception:
            continue
        fx.append((f"rand{nr}", txt))
        nr += 1
    return fx


FIXTURES = _fixtures()


def test_fixture_corpus_vs_dense_register_oracle():
    """Every fixture row (half of them with a random correction on the output) vs the dense partial-trace oracle;
    the bare output is pure; non-vacuity per stratum."""
    tot = dict(rows=0, frac=0, route1=0, route2=0, kind2=0, born3b=0, rot=0, fired=0, corrected=0)
    worst = 0.0
    for name, txt in FIXTURES:
        s = _compile(txt)
        N = 40
        buf = s.sample_barrier(N, seed=3)
        ow = list(s.output_wires())
        rng = np.random.default_rng(len(name))
        corr = rng.integers(0, 2, (N, 2, len(ow))).astype(np.uint8)
        corr[: N // 2] = 0
        rf = RecordFidelity(s)
        F = np.asarray(rf.evaluate(buf, correction=corr))
        st = rf.stats()
        rhoB = bare_rho(txt, ow)
        assert abs(purity(rhoB) - 1.0) <= 1e-9, name               # never a false pure
        for i in range(N):
            worst = max(worst, abs(F[i] - register_F(rhoB, buf, i, ow, corr[i])))
        d = np.unpackbits(np.asarray(buf.dets()), axis=1, bitorder="little")[:, :s.num_detectors]
        tot["rows"] += N
        tot["frac"] += int(np.sum((F > 1e-6) & (F < 1 - 1e-6)))
        tot[f"route{st['route']}"] += N
        tot["kind2"] += sum(1 for i in range(N) if buf.plan_content(i)[0][:1] == b"\x02")
        tot["born3b"] += st["rn_misses"]
        tot["rot"] += st["rotation_misses"]
        tot["fired"] += int(d.any(1).sum())
        tot["corrected"] += int(corr.any((1, 2)).sum())
    assert worst <= TOL, worst
    for key, v in tot.items():
        assert v > 0, (key, tot)


def test_strict_contract_refusals_by_name():
    """Review (i)/(v): the B1 counterexample (an outcome-dependent output: fo = 0 but m_L = 1) is refused, naming its
    measurement; an open Bell frame (e = 1) and the kappa fixture (c = 4) too; no OUTPUT_QUBITS -> RF_NO_OUTPUT."""
    with pytest.raises(ValueError, match="RF_OUTPUT_NOT_STRICT") as ei:
        RecordFidelity(_compile(_B1))
    msg = str(ei.value)
    assert "c = 1" in msg and "measurements [0]" in msg, msg
    with pytest.raises(ValueError, match=r"RF_OUTPUT_NOT_STRICT.*e = 1"):
        RecordFidelity(_compile(_BELL_OPEN))
    with pytest.raises(ValueError, match=r"RF_OUTPUT_NOT_STRICT.*c = 4"):
        RecordFidelity(_compile(_kappa_text(2, 0.3, "MX")))
    with pytest.raises(ValueError, match="RF_NO_OUTPUT"):
        RecordFidelity(_compile("RX 0\nT 0\nMX 0\nDECISION(0) rec[-1]\n"))
    # the frame closed by Pauli feedback from the named measurement: served, F = 1 on the noiseless rows
    s = _compile(_B1.replace("DECISION(0) rec[-1]", "CX rec[-1] 0\nDECISION(0) rec[-1]"))
    assert np.allclose(RecordFidelity(s).evaluate(s.sample_barrier(32, seed=1)), 1.0, atol=TOL)


def test_ccz_factory_is_refused_by_name():
    """The CCZ factory (OUTPUT line, 162 qubits) entangles its output with unrecorded measurement reads: e = 72, c = 9."""
    s = _compile(private_fixture("ccz_factory_a_out.stim").read_text())
    with pytest.raises(ValueError, match=r"RF_OUTPUT_NOT_STRICT.*s_O = 81/162.*c = 9, e = 72"):
        RecordFidelity(s)


def test_bell_probe_with_closed_frame():
    """The fed-back Bell pair: a harmless XX fault scores 1 like the clean rows, the harmful X0 (a flipped feedback)
    scores 0; the unrelated detector never lowers F (its wires are traced out)."""
    want = {"bell_clean": 1.0, "bell_XX": 1.0, "bell_X0": 0.0}
    for name, txt in FIXTURES:
        if name not in want:
            continue
        s = _compile(txt)
        buf = s.sample_barrier(400, seed=5)
        F = np.asarray(RecordFidelity(s).evaluate(buf))
        assert np.max(np.abs(F - want[name])) <= TOL, (name, np.unique(F))
        assert np.asarray(buf.dets()).any()


def test_correction_is_part_of_the_memo_key():
    """Review (iii): one call, identical noiseless records, alternating corrections — a memo keyed without the
    correction would serve the first row's value to every row (MUT-RF-MEMO-NOC)."""
    for txt, route in ((_STAB.replace("DEPOLARIZE1(0.3) 0\n", ""), 1), (_BELL.format(err=""), 2)):
        s = _compile(xtim.Circuit(txt).without_noise().text)
        N = 16
        buf = s.sample_barrier(N, seed=1)
        corr = np.zeros((N, 2, len(s.output_wires())), np.uint8)
        corr[1::2, 0, 0] = 1                                   # X on the first output wire
        rf = RecordFidelity(s)
        F = np.asarray(rf.evaluate(buf, correction=corr))
        assert rf.stats()["route"] == route
        assert np.allclose(F[0::2], 1.0, atol=TOL) and np.all(F[1::2] < 1 - 1e-6), F


def test_color5_and_rp2_are_served():
    """color5 (a = 9) and rp2 (a = 1) meet the strict contract: route 3b; the noiseless rows score 1."""
    for name, a in (("magna_color5_fid.stim", 9), ("magna_rp2_fid.stim", 1)):
        txt = open(os.path.join(DATA, name)).read()
        s = _compile(xtim.Circuit(txt).without_noise().text)
        rf = RecordFidelity(s)
        F = np.asarray(rf.evaluate(s.sample_barrier(400, seed=2)))
        st = rf.stats()
        assert st["route"] == 2 and st["out_a"] == a, (name, st)
        assert np.allclose(F, 1.0, atol=1e-9), (name, np.unique(F))


def _color5():
    return _compile(open(os.path.join(DATA, "magna_color5_fid.stim")).read())


def test_contents_and_kernels_persist_until_clear():
    """C2: a warm evaluator compiles nothing; clear() drops contents and kernels."""
    s = _color5()
    buf = s.sample_barrier(2000, seed=12)
    rf = RecordFidelity(s)
    a = rf.evaluate(buf)
    st1 = rf.stats()
    assert st1["kernel_compiles"] > 0 and st1["memo_bytes"] == 0
    held = st1["held_bytes"]
    b = rf.evaluate(buf)
    st2 = rf.stats()
    assert np.array_equal(a, b)
    assert st2["kernel_compiles"] == st1["kernel_compiles"] and st2["held_bytes"] == held
    rf.clear()
    st3 = rf.stats()
    assert st3["held_bytes"] < held and st3["memo_bytes"] == 0
    c = rf.evaluate(buf)
    assert np.array_equal(a, c) and rf.stats()["kernel_compiles"] == 2 * st1["kernel_compiles"]


def test_max_bytes_budgets_what_is_held():
    """Q2: max_bytes budgets the register + memo + contents + kernels; above it the evaluation is refused by name."""
    s = _color5()
    buf = s.sample_barrier(2000, seed=12)
    rf = RecordFidelity(s)
    rf.evaluate(buf, keep_memo=True)
    held = rf.stats()["held_bytes"]
    assert held > 0
    with pytest.raises(RuntimeError, match="above the byte budget"):
        RecordFidelity(s, max_bytes=held // 2).evaluate(buf)
    F = RecordFidelity(s, max_bytes=2 * held).evaluate(buf, keep_memo=True)
    assert F.shape == (2000,)


def test_many_distinct_corrections_hold_no_memory():
    """Distinct corrections never grow what the evaluator holds across calls (the memo is per call)."""
    s = _color5()
    buf = s.sample_barrier(2000, seed=3)
    no = len(s.output_wires())
    rf = RecordFidelity(s)
    rf.evaluate(buf, correction=np.zeros((2000, 2, no), np.uint8))
    held, comp = rf.stats()["held_bytes"], rf.stats()["kernel_compiles"]
    rng = np.random.default_rng(0)
    for _ in range(4):
        rf.evaluate(buf, correction=rng.integers(0, 2, (2000, 2, no)).astype(np.uint8))
        assert rf.stats()["held_bytes"] == held and rf.stats()["kernel_compiles"] == comp


def _born_rest(nb, noise=""):
    """A magic output (route 3b) beside nb independent random-outcome measurements on the rest (final review B1)."""
    L = ["RX 0", "T 0"] + ([noise] if noise else [])
    for j in range(nb):
        L += [f"RX {j + 1}", f"M {j + 1}", f"DECISION({j}) rec[-1]"]
    return "\n".join(L + ["OUTPUT_QUBITS out 0"]) + "\n"


@pytest.mark.parametrize("nb", [40, 64, 130, 540, 900])
def test_many_born_records_on_the_rest(nb):
    """Every projection is normalised (a random-outcome record contributes an exact 1/2, carried as an integer
    exponent) and impossible patterns are decided in GF(2), so 40, 64 (two record words), 130, 540 and 900 independent
    Born records keep F exact — noiseless |T> scores 1; noisy rows with random corrections equal the dense oracle
    (XTIM_MUTATE_RF_NY_THRESH, the former 1e-12 threshold, and XTIM_MUTATE_RF_UNDERFLOW, the 2^-j factors
    accumulated as floats, score them 0)."""
    s = _compile(_born_rest(nb))
    rf = RecordFidelity(s)
    F = np.asarray(rf.evaluate(s.sample_barrier(50, seed=1)))
    assert rf.stats()["route"] == 2 and np.allclose(F, 1.0, atol=TOL), np.unique(F)
    txt = _born_rest(nb, "DEPOLARIZE1(0.3) 0")
    s = _compile(txt)
    N = 40 if nb < 500 else 12
    buf = s.sample_barrier(N, seed=2)
    ow = list(s.output_wires())
    rng = np.random.default_rng(nb)
    corr = rng.integers(0, 2, (N, 2, 1)).astype(np.uint8)
    F = np.asarray(RecordFidelity(s).evaluate(buf, correction=corr))
    rhoB = bare_rho(txt, ow)
    assert max(abs(F[i] - register_F(rhoB, buf, i, ow, corr[i])) for i in range(N)) <= TOL
    assert np.any((F > 1e-6) & (F < 1 - 1e-6))


def test_2048_born_records_on_the_rest():
    """2048 Born records (the dense oracle is not run at this width): noiseless |T> scores 1; with DEPOLARIZE1(0.3) on
    the output every row is an exact Pauli-fault value of |T> — 1 (none), 1/2 (X or Y), 0 (Z) — and 1 and 1/2 occur."""
    s = _compile(_born_rest(2048))
    F = np.asarray(RecordFidelity(s).evaluate(s.sample_barrier(8, seed=1)))
    assert np.allclose(F, 1.0, atol=TOL), np.unique(F)
    s = _compile(_born_rest(2048, "DEPOLARIZE1(0.3) 0"))
    F = np.asarray(RecordFidelity(s).evaluate(s.sample_barrier(24, seed=2)))
    d = np.min(np.abs(F[:, None] - np.array([0.0, 0.5, 1.0])[None, :]), axis=1)
    assert np.all(d <= TOL), np.unique(F)
    assert np.any(np.abs(F - 1) <= TOL) and np.any(np.abs(F - 0.5) <= TOL)


def test_correction_entries_outside_zero_one_are_refused():
    s = _compile(_BELL.format(err=""))
    buf = s.sample_barrier(4, seed=1)
    c = np.zeros((4, 2, 2), np.uint8)
    c[0, 0, 0] = 2
    with pytest.raises(ValueError, match="0 / 1"):
        RecordFidelity(s).evaluate(buf, correction=c)
