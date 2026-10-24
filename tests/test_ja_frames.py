"""J-a N4b: input_pauli frames on a FAM sampler with the frames class (plan §1.1 row 9, §3.6 gate 7; design §2.7, D4(a)).

Every frame component (X_i / Z_i on a port wire at t = 0) is an atom (P_F, S_F) of the term view, propagated like the
fault alternatives (TermResidual::port_atoms), and composed as the RIGHTMOST factor of the shot's residual:
R_total = R_noise · R_f. For a clean component (anchor image P' a Pauli) that composition IS the fold
P_rec = P·P''(S), P'' = rot(S) P' rot(S)^dag (dressed or not); for a poisoned one (a T in the port cone) its S_F joins S
and the law reweights (the [C-6] class). A framed shot takes the term route (every kind, clean shots included); the
static relabel, the dressing rows and frame_born_flip are off on a frames-class sampler.

Oracles: DENSE = tests/_record_oracle.emitted_law(text, frame=...) on the ORIGINAL text (the frame applied to |0..0>
before the first instruction), its columns REFERENCE-RELATIVE (J-b1 P10, Addendum A1 §2: a port circuit's columns
are relative to the frame-free noiseless reference; J-a N1's ABSOLUTE port convention is superseded); the
semantic oracle on every shot of the gate samplers (the term-view oracle applies each supplied frame atom, rot(S_F)
then P_F, by gates, then the noise residual).
"""
import io
import contextlib
import warnings

import numpy as np
import pytest

import _ja_fixtures as F
from _record_oracle import emitted_columns, emitted_law, reference_records
from _record_oracle_mixed import record_law_mixed
from test_ja_ppr_records import CH_NOISY, K0_PAIR_REFUSAL, assert_oracle_ran, biased_fuzz_records, max_z_pooled

import xtim

pytestmark = pytest.mark.filterwarnings("ignore")

N = 40_000

# [C-6]: the magic wire 1 carries T|+>; the port wire 0 controls an X onto it before a second T. A frame X0 (port
# |1>) is an X before that T: poisoned (its anchor image is not a Pauli). Frame-free MX 1: P(0) = 0.5 (S|+>);
# framed: P(0) = 1.0 (T X T|+> = e^{i pi/4} |+>). M 0 reads the port wire (a deterministic channel).
C6 = ("INPUT_QUBITS in 0\nRX 1\nT 1\nCX 0 1\nT 1\nMX 1\nM 0\n"
      "OBSERVABLE_INCLUDE(0) rec[-2]\nDETECTOR rec[-1]\n")
# the same with a fault between the two T's (term-routed alternatives under the frame) and every record class
C6_NOISY = ("INPUT_QUBITS in 0\nRX 1 2\nT 1\nX_ERROR(0.1) 1\nCX 0 1\nT 1\nX_ERROR(0.1) 2\nM 2\nMX 1\nM 0\n"
            "DETECTOR rec[-3]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\nDETECTOR rec[-1]\nDECISION(0) rec[-2]\n")
# adv-A l4 (tests/test_input_frame_mirror.L4): a clean frame X1 crossing a T...T_DAG mirror on Z0Z1, a certain fault
# between the terms: the static relabel gives P(obs) = 0.852, the truth is 0.146 (dressed: P''(S) != P').
L4 = ("INPUT_QUBITS in 1\nRX 0\nR 2\nT 0\nCX 0 1\nT 1\nCX 0 1\nX_ERROR(1) 0\nCX 0 1\nT_DAG 1\nCX 0 1\n"
      "DEPOLARIZE1(0.001) 2\nMX 0\nM 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n")
# adv-A strip.py: a frame X0 crossing a CCZ mirror twice, a 50% fault between: truth 0.5 on both reads
STRIP = ("INPUT_QUBITS in 0\nRX 1 2\nCCZ 0 1 2\nX_ERROR(0.5) 1\nCCZ 0 1 2\nDEPOLARIZE1(0.0001) 1 2\nMX 1 2\n"
         "DETECTOR rec[-2]\nDETECTOR rec[-1]\n")
# frames on the kind-2 route: ch_cultivation's H-eigenstate with a port wire 9 controlling an X onto the data wire
# before the CH Hadamard tests (frame X9: poisoned through the CH terms); the data read in X (a gauge detector and a
# Born observable on one record), the port read in Z (a channel)
CH_FR = ("INPUT_QUBITS in 9\n" + CH_NOISY.replace("# round 1\n", "CX 9 0\n# round 1\n") +
         "MX 0\nM 9\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\nDETECTOR rec[-1]\n")

FIXTURES = {"c6": C6, "c6_noisy": C6_NOISY, "l4": L4, "strip": STRIP, "ch_fr": CH_FR}
FRAMES = {"c6": ["X", "Z", "Y"], "c6_noisy": ["X", "Y"], "l4": ["X", "Z", "Y"], "strip": ["X", "Y"],
          "ch_fr": ["X", "Y"]}


def _port(text):
    return int(text.splitlines()[0].split()[2])


def family(text, **kw):
    return F.family(text, **kw)


def frame_rows(n, P):
    """input_pauli rows for one port wire: [x | z]."""
    fr = np.zeros((n, 2), np.uint8)
    if P in "XY":
        fr[:, 0] = 1
    if P in "ZY":
        fr[:, 1] = 1
    return fr


def columns(s, n, seed, P):
    d, o, dc = s.sample(n, seed=seed, input_pauli=frame_rows(n, P))
    return np.concatenate([F.unpack(d, s.num_detectors), F.unpack(o, s.num_observables), F.unpack(dc, s.num_decisions)], 1)


def law_rel(text, P):
    """DENSE's emitted law with the frame P on the port wire at t = 0: detector / observable columns relative to the
    frame-free noiseless reference (J-b1 P10, A1 §2), decisions raw — emitted_law's own convention."""
    return emitted_law(text, frame={_port(text): P} if P != "I" else None)


def frame_law_gate(name, text, P, n=N):
    """The law gate on an oracle-free sampler first (a law error is red through the law statistic), then the same
    draws replayed on a gate sampler with the semantic oracle on every shot (bitwise equal; every shot framed and
    term-routed, the term-view oracle on each)."""
    law = law_rel(text, P)
    s_law = family(text)
    draws = []

    def draw(k, seed):
        M = columns(s_law, k, seed, P)
        draws.append((k, seed, M))
        return M
    M = F.check_law(f"{name}-{P}", draw, law, n)
    s = family(text, oracle_all_shots=True)
    for k, seed, Mk in draws:
        assert np.array_equal(columns(s, k, seed, P), Mk), (name, P, "the oracle sampler's bits differ")
    shots = sum(k for k, _, _ in draws)
    assert_oracle_ran(s, shots, term=False)
    rs = s.route_stats()
    assert rs["shots_framed"] == shots and rs["oracle_term_shots"] == shots, rs
    return M, law, rs


def _quiet(fn, *a, **k):
    with contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


# ── the design values ────────────────────────────────────────────────────────────────────────────
def test_c6_dense_values():
    """[C-6] in DENSE: the poisoned frame X0 reweights MX 1 from P(0) = 0.5 to 1.0 (not a relabel)."""
    a, b = law_rel(C6, "I"), law_rel(C6, "X")
    assert abs(sum(p for k, p in a.items() if k[1] == 0) - 0.5) < 1e-12        # columns: (M 0 detector, obs)
    assert abs(sum(p for k, p in b.items() if k[1] == 0) - 1.0) < 1e-12


def test_l4_dense_value():
    l4 = law_rel(L4, "X")
    assert abs(sum(p for k, p in l4.items() if k[1] == 1) - 0.1464466) < 1e-6      # obs (absolute)


# ── G-7: the frames class vs DENSE, per frame component ──────────────────────────────────────────
@pytest.mark.parametrize("name,P", [(k, P) for k in FIXTURES for P in FRAMES[k]])
def test_g7_frames_class_vs_dense(name, P):
    M, law, rs = frame_law_gate(name, FIXTURES[name], P)
    if name == "ch_fr":
        assert rs["shots_kind2"] > 0, rs                  # frames on the kind-2 route
    if name == "c6" and P == "X":
        assert int(M[:, 1].sum()) == 0                    # framed MX 1 (the observable): P(0) = 1.0, exact


def test_g7_frame_free_shots_unchanged():
    """A frames-class sampler with a zero frame row: no shot is framed, and the bits equal the same sampler's
    frame-free call (input_pauli=None) at the same seed."""
    for text in (C6_NOISY, CH_FR):
        s = family(text)
        a = columns(s, 20_000, 3, "I")
        d, o, dc = s.sample(20_000, seed=3)
        b = np.concatenate([F.unpack(d, s.num_detectors), F.unpack(o, s.num_observables), F.unpack(dc, s.num_decisions)], 1)
        assert np.array_equal(a, b) and s.route_stats()["shots_framed"] == 0


def test_g7_per_shot_frames_equal_per_frame_runs():
    """Per-shot frames: a run whose frame row varies per shot equals, shot by shot, the constant-frame runs at the
    same seed (keyed per-shot streams; a frame consumes no randomness)."""
    for text in (C6_NOISY, L4, CH_FR):
        rng = np.random.default_rng(5)
        n = 8000
        which = rng.integers(0, 4, n)
        rows = np.stack([np.isin(which, (1, 3)), np.isin(which, (2, 3))], 1).astype(np.uint8)   # I X Z Y
        s = family(text)
        d, o, dc = s.sample(n, seed=9, input_pauli=rows)
        mixed = np.concatenate([F.unpack(d, s.num_detectors), F.unpack(o, s.num_observables), F.unpack(dc, s.num_decisions)], 1)
        for k, P in enumerate("IXZY"):
            ref = columns(family(text), n, 9, P)
            assert np.array_equal(mixed[which == k], ref[which == k]), (text[:30], P)


# ── G-7 on a random corpus (the lemma4.py class): the [A] fuzz circuits with wire 0 an input port ─────────────
def _port_fuzz(seed):
    """fuzz_records(seed) (biased X_ERROR noise) with wire 0 an INPUT port (not reset): the frame on wire 0 crosses the
    circuit's T layer and its mirror — clean, dressed or poisoned, as the seed has it."""
    t = biased_fuzz_records(seed)
    assert t.startswith("R 0 1 2 3\n")
    return "INPUT_QUBITS in 0\nR 1 2 3\n" + t[len("R 0 1 2 3\n"):]


def _rel_law_mixed(port_text, P):
    """DENSE (density matrices) of the closed circuit with the frame as a gate on |0> at t = 0, its detector /
    observable columns relative to the frame-free noiseless reference of the port circuit (J-b1 P10, A1 §2)."""
    closed = "R 0\n" + ("%s 0\n" % P if P != "I" else "") + port_text.split("\n", 1)[1]
    cols = emitted_columns(port_text)
    ref = reference_records(port_text)
    off = [sum(ref[i] for i in recs) & 1 if kind != "decision" else 0 for kind, _, recs in cols]
    out = {}
    for rec, p in record_law_mixed(closed, readout=True).items():
        k = tuple((sum(rec[i] for i in recs) + o) & 1 for (_, _, recs), o in zip(cols, off))
        out[k] = out.get(k, 0.0) + p
    return out


def test_g7_fuzz_frames_corpus_vs_dense():
    rng = np.random.default_rng(17)
    served, refused = 0, []
    for sd in range(30):
        text = _port_fuzz(sd)
        P = "XYZ"[int(rng.integers(0, 3))]
        law = _rel_law_mixed(text, P)
        s = family(text)

        def draw(k, seed, s=s, P=P):
            return columns(s, k, seed, P)
        try:
            M = draw(20_000, 11)
        except RuntimeError as e:
            assert K0_PAIR_REFUSAL in str(e), (sd, str(e))       # N4-D1, by name
            refused.append(sd)
            continue
        z, imp = max_z_pooled(M, law)
        assert imp == 0, (sd, P, "impossible outcomes", imp)
        if z >= 5:
            F._rerun_rule(f"g7-fuzz{sd}", z)
            z, imp = max_z_pooled(draw(40_000, 23), law)
            assert imp == 0 and z < 5, (sd, P, z, imp)
        served += 1
    assert served >= 20, (served, refused)


# ── refusals by name ─────────────────────────────────────────────────────────────────────────────
def test_production_serves_the_c6_poisoned_frame():
    """RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3 test_ja_frames.py:228-238;
    M-FRAME): the pre-flip sampler refused the poisoned component by name (HARD poison); production is the family
    sampler, whose frames class composes it — the public columns equal the FAM(all) columns."""
    text = C6.replace("OBSERVABLE_INCLUDE(0) rec[-2]\n", "")
    s = _quiet(xtim.compile_twirl_sampler, text)
    d, _o = s.sample(10, seed=1, input_pauli=frame_rows(10, "X"))
    fam = family(text)                                          # the frames class composes it
    assert columns(fam, 10, 1, "X").shape == (10, 1) and fam.route_stats()["shots_framed"] == 10
    assert s.record_report()["term_route"]["shots_framed"] == 10
