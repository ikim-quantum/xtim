"""Twirl sampler vs an EXACT record-law oracle — the 2026-10-09 sampling-audit findings.

Every expectation below is the exact joint law of the circuit's record parities,
enumerated on dense states by ``tests/_record_oracle.py`` (independent of xtim; validated
against the audit's density-matrix oracle).  The samplers are compared cell by cell at
|z| < 5 with N chosen so every defect is ≥ 20σ away.

* I / J — the BORN observable's deterministic record invert (``MY !q``, ``MPAD 1``, the
  H-elimination's folded sign) was folded into the clean-shot weight only, so every shot
  with a fired noise atom reported the observable FLIPPED: the (detector, observable)
  joint was wrong (I) and any noise anywhere leaked into an MPAD observable (J).
* M — a Born DECISION / retained state was computed after collapsing UNREAD certified
  generators (the κ kernel chain on the memo path, every generator on the exact path).
  An unread generator that anticommutes with the decision operator randomised it.
* SEG — the Born-decision memo conditioned on ``o_rec.sigma``, which the abelian fast path
  never writes: empty → null dereference (segfault), stale → the wrong record sector.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from _record_oracle import max_z, parities, parity_law  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def _unpack(a, k):
    a = np.asarray(a, np.uint8)
    if k == 0:
        return np.zeros((a.shape[0], 0), np.uint8)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :k]


def _det_obs_law(text, flips=()):
    dets, obs, _ = parities(text)
    cols = dets + [obs[k] for k in sorted(obs)]
    law = parity_law(text, cols)
    if any(flips):                       # observable-reference convention (audit K) agnostic:
        nd = len(dets)                   # a random observable may be reported XOR a constant
        law = {k[:nd] + tuple(b ^ f for b, f in zip(k[nd:], flips)): p for k, p in law.items()}
    return law


def _assert_det_obs(text, d, o):
    import itertools
    M = np.concatenate([d, o], 1)
    zs = [max_z(M, _det_obs_law(text, f)) for f in itertools.product((0, 1), repeat=o.shape[1])]
    best = min(zs)
    assert best[1] == 0 and best[0] < 5, (zs, {tuple(k): v for k, v in _det_obs_law(text).items()})


# ── I / J: the Born observable's deterministic invert ───────────────────────────────────
I_TXT = ("X_ERROR(0.2) 0\nM 0\nRX 1\nT 1\nH 0\nCY rec[-1] 0\nMX 0\nMY 1\n"
         "DETECTOR rec[-3] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]\n")
J_TXT = ("RX 0\nT 0\nMY 0\nX_ERROR(0.1) 0\nMPAD 1\nM 2\nDETECTOR rec[-1]\n"
         "OBSERVABLE_INCLUDE(0) rec[-2] rec[-3]\n")
# the invert on an explicitly inverted readout of the Born record, noise elsewhere
INV_TXT = ("RX 0\nT 0\nX_ERROR(0.3) 1\nM 1\nMX !0\nDETECTOR rec[-2]\n"
           "OBSERVABLE_INCLUDE(0) rec[-1]\n")


# audit campaign seed 8022 (minimized): the invert is eliminate_hadamards' fold of `X 0` into the
# first readout; observable 1 is the Born channel, observable 0 a deterministic channel.
S8022_TXT = ("H 0\nT 0\nH 0\nX_ERROR(0.1) 0\nX 0\nM 0\nM 0\nDETECTOR rec[-2] rec[-1]\n"
             "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-2]\n")


@pytest.mark.parametrize("text", [I_TXT, J_TXT, INV_TXT, S8022_TXT], ids=["I", "J", "inv", "8022"])
@pytest.mark.parametrize("selfcheck", [0, 2000])
@pytest.mark.parametrize("mode", ["sample", "barrier"])
def test_born_observable_joint_is_exact(text, selfcheck, mode):
    # barrier: the retention path never drew the observable on a diag shot (it re-emitted the
    # previous shot's bit) — invisible when obs and detectors are independent (I, J), wrong
    # joint on 8022 (3.1.9 included).
    s = xtim.compile_twirl_sampler(text, selfcheck=selfcheck)
    assert s.channel_report()["refused_observables"] == []
    if mode == "sample":
        d, o = s.sample(100_000, seed=7)
    else:
        b = s.sample_barrier(100_000, seed=7)
        d, o = b.dets(), b.obs()
    _assert_det_obs(text, _unpack(d, s.num_detectors), _unpack(o, s.num_observables))


@pytest.mark.parametrize("text", [I_TXT, J_TXT], ids=["I", "J"])
@pytest.mark.parametrize("engine", ["twirl", "auto"])
def test_born_observable_joint_detector_sampler(text, engine):
    c = xtim.Circuit(text).compile_detector_sampler(seed=3, engine=engine)
    d, o = c.sample(100_000, separate_observables=True)
    _assert_det_obs(text, np.asarray(d, np.uint8), np.asarray(o, np.uint8))


# ── M: unread generators must not collapse a Born DECISION ─────────────────────────────
M_TXT = ("RX 3\nT 3\nRX 4\nT_DAG 4\nX_ERROR({p}) 2\nCS_DAG 4 3\nCCZ 4 3 2\nMX 4\n"
         "DECISION(0) rec[-1]\n")


def _dec_law(text):
    _, _, decs = parities(text)
    return parity_law(text, [decs[k] for k in sorted(decs)])


@pytest.mark.parametrize("p", ["0.5", "1"])
@pytest.mark.parametrize("mode", ["sample", "barrier", "port"])
def test_decision_not_collapsed_by_unread_generator(p, mode):
    text = M_TXT.format(p=p)
    law = _dec_law(text)
    assert abs(law[(0,)] - (0.5 * (1 - float(p)) + float(p) * (2 + 2 ** 0.5) / 4)) < 1e-12
    N = 60_000
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    if mode == "sample":
        dec = s.sample(N, seed=3).decisions
    elif mode == "barrier":
        dec = s.sample_barrier(N, seed=3).decisions()
    else:
        from xtim.port import Consume, compile as port_compile
        r = port_compile(text, Consume(dets=True, decisions=True)).run(shots=N, seed=4)
        dec = np.packbits(np.asarray(r.decisions, np.uint8), axis=1, bitorder="little")
    z, imp = max_z(_unpack(dec, 1), law)
    assert imp == 0 and z < 5, (z, law, _unpack(dec, 1).mean())


# Audit seed 103021 (the M campaign case): THREE Born decisions on a κ > 0 plan. The exact
# per-shot path (sample() / xtim.port route κ > 0 shots there) computed every p_j on the
# UNcollapsed state — the decisions came out as independent marginals, the joint was wrong.
M3_TXT = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
          "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
          "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\nMX 3\nMX 4\nMX 2\n"
          "DECISION(0) rec[-3]\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\n")


# audit campaign seed 133 (minimized): MY 0 then Z0 read TWICE (the two records must agree), an
# X fault on the CCZ control after the reads puts the shot on the exact per-shot path.
S133_TXT = ("MY 0\nMPP Z0\nMPP Z0\nH 3\nT 3\nX_ERROR(0.5) 2\nCCZ 0 3 2\n"
            "DECISION(0) rec[-3]\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\n")


@pytest.mark.parametrize("text", [M3_TXT, S133_TXT], ids=["103021", "133"])
@pytest.mark.parametrize("mode", ["sample", "barrier", "port"])
def test_sequential_decisions_joint_on_exact_path(mode, text):
    law = _dec_law(text)
    N = 60_000
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    if mode == "sample":
        dec = s.sample(N, seed=3).decisions
    elif mode == "barrier":
        dec = s.sample_barrier(N, seed=3).decisions()
    else:
        from xtim.port import Consume, compile as port_compile
        r = port_compile(text, Consume(decisions=True)).run(shots=N, seed=4)
        dec = np.packbits(np.asarray(r.decisions, np.uint8), axis=1, bitorder="little")
    z, imp = max_z(_unpack(dec, 3), law)
    assert imp == 0 and z < 5, (z, law)


# ── SEG + stale σ: the Born-decision memo's record conditioning ────────────────────────
SEG_TXT = ("RX 0\nT_DAG 0\nPAULI_CHANNEL_1(0.01,0.25,0) 0\nT 0\nMY 2\nMX 0\n"
           "DECISION(0) rec[-2]\nDECISION(1) rec[-1]\n")
# X0X1 (a coin after the X fault crosses T0·T_DAG1) is read by a DETECTOR; the Born
# DECISION X0X1·X2 (X2 on T|+>) has P(+1) = (1 + s/√2)/2 in record sector s — so a memo
# entry conditioned on the WRONG sector (stale σ) is visible in the (det, dec) joint.
STALE_TXT = ("R 0 1 2\nH 0\nCX 0 1\nH 2\nT 2\nX_ERROR(0.5) 0\nT 0\nT_DAG 1\n"
             "R 3\nH 3\nCX 3 0\nCX 3 1\nH 3\nM 3\nDETECTOR rec[-1]\n"
             "R 4\nH 4\nCX 4 0\nCX 4 1\nCX 4 2\nH 4\nM 4\nDECISION(0) rec[-1]\n")


def test_born_memo_fast_path_does_not_crash():
    code = ("import os, xtim\nos.environ['XTIM_QUIET']='1'\n"
            f"t = {SEG_TXT!r}\n"
            "for seed in range(1, 13):\n"
            "    xtim.compile_twirl_sampler(t, selfcheck=0).sample(3000, seed=seed)\n"
            "from xtim.port import Consume, compile as pc\n"
            "for seed in range(1, 7):\n"
            "    pc(t, Consume(dets=True, decisions=True)).run(shots=3000, seed=seed)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       env=dict(os.environ, XTIM_QUIET="1"))
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])


@pytest.mark.parametrize("selfcheck", [0, 1, 2000])
def test_born_memo_conditions_on_the_shots_own_records(selfcheck):
    dets, _, decs = parities(STALE_TXT)
    law = parity_law(STALE_TXT, dets + [decs[0]])
    # non-vacuous: the decision's law really depends on the detector sector
    assert abs(law[(1, 0)] / (law[(1, 0)] + law[(1, 1)]) - law[(0, 0)] / (law[(0, 0)] + law[(0, 1)])) > 0.1
    s = xtim.compile_twirl_sampler(STALE_TXT, selfcheck=selfcheck)
    b = s.sample(80_000, seed=11)
    M = np.concatenate([_unpack(b.dets, 1), _unpack(b.decisions, 1)], 1)
    z, imp = max_z(M, law)
    assert imp == 0 and z < 5, (z, law)


def test_seg_decisions_law():
    law = _dec_law(SEG_TXT)
    s = xtim.compile_twirl_sampler(SEG_TXT, selfcheck=0)
    dec = _unpack(s.sample(60_000, seed=2).decisions, 2)
    z, imp = max_z(dec, law)
    assert imp == 0 and z < 5, (z, law)


# ── the selfcheck window's SEMANTICS ORACLE is live (non-vacuous) ───────────────────────
# Mutation record (scratchpad report): with the born invert dropped, the kernel collapse
# restored, or the exact-path collapse removed, compile_twirl_sampler(selfcheck=2000) raises
# on I/J/inv, M/M3 and M3 respectively (stderr "SEMANTICS ORACLE FAIL"). Stale σ (SEG) is a
# post-window defect the window cannot see by construction — the selfcheck=0 tests gate it.
@pytest.mark.parametrize("text", [I_TXT, M3_TXT, STALE_TXT], ids=["obs", "dec", "dec+det"])
@pytest.mark.parametrize("mode", ["sample", "barrier"])
def test_semantics_oracle_runs_in_the_window(text, mode):
    s = xtim.compile_twirl_sampler(text, selfcheck=2000)
    s.sample(3000, seed=5) if mode == "sample" else s.sample_barrier(3000, seed=5)
    assert s.channel_report()["semantic_oracle_shots"] == 64
    s.sample(3000, seed=6)                     # later runs: window spent, oracle off
    assert s.channel_report()["semantic_oracle_shots"] == 64
    s0 = xtim.compile_twirl_sampler(text, selfcheck=0)
    s0.sample(1000, seed=5)
    assert s0.channel_report()["semantic_oracle_shots"] == 0


# ── PPR exact-fallback shots of a Born-observable circuit (front-end audit lead) ─────────
# Two sandwiched CH Hadamard-test blocks (the qec_library fallback_ppr_kf2 benchmark): a shot
# firing BOTH X faults composes to kappa_fold = 2, the PPR plan falls back, and the shot is
# computed by exact_ppr_shot. That routine MEASURED the Born observable, but the emission passed
# a constant 0 (and skipped the obs statistics): every such shot reported observable 0.
KF2_TXT = ("H 1\nT 1\nH 0\nCH 1 0\nX_ERROR(0.35) 0\nCH 1 0\nH 3\nT 3\nH 2\nCH 3 2\n"
           "X_ERROR(0.35) 2\nCH 3 2\nH 4\nX_ERROR(0.05) 4\nMX 0\nMX 2\nMX 4\n"
           "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")


@pytest.mark.parametrize("read", ["MX 1", "MX 3", "M 1"])
def test_ppr_exact_fallback_emits_the_born_observable(read):
    text = KF2_TXT + read + "\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    d, o = s.sample(60_000, seed=3)
    assert s.channel_report()["exact_shots"] > 1000          # the fallback path is exercised
    _assert_det_obs(text, _unpack(d, s.num_detectors), _unpack(o, s.num_observables))


# ── no state shared across compiles (front-end audit lead on audit E) ───────────────────
# On 3.1.9 the front-end agent saw E's twirl classification depend on whether another circuit
# had been compiled earlier in the process. Differential check here: every engine's output for
# each circuit is byte-identical whether it is compiled FRESH (own process) or after the others
# (both orders). Re-measured on this branch and on 3.1.9 over 1,400 audit-generator circuits x 3
# orders + 240 fresh-process controls: no dependence found on either build.
E_TXT = ("MX 1\nCY rec[-1] 1\nMX 1\nMR 1\nMPP Z0*Z1\nDETECTOR rec[-1] rec[-2]\n"
         "PAULI_EXPECTATION(1) X0\n")
_XSTATE = [E_TXT, I_TXT, J_TXT, S8022_TXT, STALE_TXT, SEG_TXT, KF2_TXT + "MX 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"]
_XSTATE_CODE = r'''
import hashlib, json, os, sys, warnings
os.environ["XTIM_QUIET"] = "1"; warnings.simplefilter("ignore")
import numpy as np, xtim
texts = json.loads(sys.argv[1]); order = json.loads(sys.argv[2])
out = {}
for i in order:
    t = texts[i]; h = hashlib.sha256()
    for eng in ("twirl", "auto", "exact"):
        try:
            s = xtim.Circuit(t).compile_detector_sampler(seed=3, engine=eng)
            for a in s.sample(2000, separate_observables=True): h.update(np.asarray(a).tobytes())
            if eng == "auto": h.update(s.engine_report()["engine"].encode())
        except Exception as e:
            h.update(("ERR" + type(e).__name__).encode())
    try:
        r = xtim.compile_twirl_sampler(t, selfcheck=0).sample(2000, seed=5)
        for a in r:
            if a is not None: h.update(np.asarray(a).tobytes())
    except Exception as e:
        h.update(("ERR" + type(e).__name__).encode())
    out[i] = h.hexdigest()
print(json.dumps(out))
'''


def _xstate_run(order):
    import json
    r = subprocess.run([sys.executable, "-c", _XSTATE_CODE, json.dumps(_XSTATE), json.dumps(order)],
                       capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                       env=dict(os.environ, XTIM_QUIET="1"))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_outputs_do_not_depend_on_earlier_compiles():
    n = len(_XSTATE)
    fresh = {}
    for i in range(n):
        fresh.update(_xstate_run([i]))
    fwd = _xstate_run(list(range(n)))
    rev = _xstate_run(list(range(n))[::-1])
    assert fwd == fresh and rev == fresh


def test_E_detector_law_after_other_compiles():
    for t in _XSTATE[1:]:
        try:
            xtim.compile_twirl_sampler(t, selfcheck=0).sample(100, seed=1)
        except Exception:
            pass
    c = xtim.Circuit(E_TXT).compile_detector_sampler(seed=1, engine="auto")
    d = np.asarray(c.sample(40_000), np.uint8)
    dets, _, _ = parities(E_TXT)
    law = parity_law(E_TXT, dets)
    assert abs(law[(0,)] - 0.5) < 1e-12                 # the detector is a fair coin
    z, imp = max_z(d, law)
    assert imp == 0 and z < 5, (z, d.mean())


# ── a Born OBSERVABLE and Born DECISIONs are ONE joint law (campaign seeds 911643/911649) ──
# The observable was a σ-correlated biased coin and the decisions a separate memoized Born chain
# on a state that never saw the observable's outcome: an observable and a decision reading the
# SAME record came out independent (P(obs != dec) = 0.25 instead of 0). Found by the selfcheck
# semantics oracle on the twirl-targeted adversarial campaign; present on 3.1.9.
OBSDEC_TXT = ("RX 0\nT 0\nX_ERROR(0.2) 1\nM 1\nMY 0\nDETECTOR rec[-2]\n"
              "OBSERVABLE_INCLUDE(0) rec[-1]\nDECISION(0) rec[-1]\n")
S911649_TXT = ("RX 0\nR 1\nRX 2\nRX 3\nRX 4\nZ_ERROR(0.2) 2\nCCZ 4 3 2\nX_ERROR(0.05) 2\n"
               "CCZ 2 4 0\nR 5\nM 5\nDETECTOR rec[-1]\nMY !2\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
               "DECISION(0) rec[-1]\n")
S911643_TXT = ("RX 0\nRX 1\nR 2\nR 3\nZ_ERROR(0.2) 0\nT 0\nCCZ 1 2 0\nX_ERROR(0.5) 2\nCCZ 2 0 1\n"
               "S 3\nY_ERROR(0.1) 2\nCZ 2 1\nT_DAG 3\nZ_ERROR(0.05) 3\nT 3\nMX 3\nZ_ERROR(0.1) 3\n"
               "MX !2\nMX !1\nM 0\nR 4\nM 4\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
               "DECISION(0) rec[-5]\nDECISION(1) rec[-2]\nDECISION(2) rec[-4]\n")


@pytest.mark.parametrize("text", [OBSDEC_TXT, S911649_TXT, S911643_TXT], ids=["shared", "911649", "911643"])
@pytest.mark.parametrize("mode", ["sample", "barrier"])
@pytest.mark.parametrize("selfcheck", [0, 2000])
def test_born_observable_and_decisions_joint(text, mode, selfcheck):
    import itertools
    dets, obs, decs = parities(text)
    cols = dets + [obs[0]] + [decs[k] for k in sorted(decs)]
    law = parity_law(text, cols)
    nd = len(dets)
    s = xtim.compile_twirl_sampler(text, selfcheck=selfcheck)
    N = 60_000
    if mode == "sample":
        b = s.sample(N, seed=3)
        d, o, dec = b.dets, b.obs, b.decisions
    else:
        b = s.sample_barrier(N, seed=3)
        d, o, dec = b.dets(), b.obs(), b.decisions()
    M = np.concatenate([_unpack(d, nd), _unpack(o, 1), _unpack(dec, len(decs))], 1)
    zs = []
    for f in (0, 1):                                    # observable reference convention (K)
        lf = {k[:nd] + (k[nd] ^ f,) + k[nd + 1:]: p for k, p in law.items()}
        zs.append(max_z(M, lf))
    best = min(zs)
    assert best[1] == 0 and best[0] < 5, zs


# ── campaign seeds: obs ⊗ decisions memo key must carry ⟨W_obs, P⟩ (prefix sign) ───────────
# After the joint obs/decision fix the decision memo conditioned on W_obs, but its injective key
# carried the decisions' prefix signs only: two shots with the same plan/coins/obs outcome and a
# different ⟨W_obs, P⟩ shared a p_j. Found by the semantics oracle on the twirl campaign.
_CAMPAIGN = {
    900233: 'RX 0\nRX 1\nRX 2\nZ_ERROR(0.05) 1\nT_DAG 1\nX_ERROR(0.5) 2\nT 2\nZ 2\nY_ERROR(0.1) 2\nT_DAG 2\nY_ERROR(0.2) 2\nT 2\nS 0\nMY !2\nM 1\nZ_ERROR(0.1) 1\nMY !0\nMPAD 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-4] rec[-3]\nDECISION(0) rec[-2]\nDECISION(1) rec[-3]\n',
    911735: 'RX 0\nRX 1\nY_ERROR(0.5) 0\nCX 0 1\nX_ERROR(0.05) 1\nCS 1 0\nY_ERROR(0.05) 0\nT 0\nY_ERROR(0.2) 0\nCZ 0 1\nX_ERROR(0.5) 0\nT_DAG 0\nM 1\nMX 0\nMPAD 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1] rec[-3]\nDECISION(0) rec[-3]\n',
    901985: 'RX 0\nRX 1\nR 2\nRX 3\nRX 4\nX_ERROR(0.5) 4\nT_DAG 4\nY_ERROR(0.05) 1\nT 1\nZ_ERROR(0.05) 1\nCCZ 1 4 3\nM 0\nMY 4\nMX 1\nM 3\nZ_ERROR(0.1) 3\nMPAD 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-3] rec[-1]\nDECISION(0) rec[-2]\nDECISION(1) rec[-1]\nDECISION(2) rec[-5]\n',
    910913: 'RX 0\nRX 1\nRX 2\nX 1\nZ_ERROR(0.1) 1\nT 1\nX_ERROR(0.1) 2\nT 2\nX_ERROR(0.2) 0\nCCZ 2 1 0\nMX 1\nMY !0\nMY 2\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\nDECISION(0) rec[-1]\nDECISION(1) rec[-2]\n',
    910064: 'RX 0\nRX 1\nR 2\nRX 3\nRX 4\nCCZ 3 2 0\nCCZ 4 3 0\nZ_ERROR(0.2) 1\nCCZ 0 2 1\nMX 2\nMX 3\nMY 0\nM 4\nX_ERROR(0.05) 4\nMX !1\nMPAD 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-2]\nDECISION(0) rec[-3]\n',
}


def _ref_joint_law(text):
    """Exact (detectors, observable, decisions) law, detectors reference-relative (Stim)."""
    import re as _re
    dets, obs, decs = parities(text)
    cols = dets + [obs[k] for k in sorted(obs)] + [decs[k] for k in sorted(decs)]
    law = parity_law(text, cols)
    t0 = _re.sub(r"(X_ERROR|Y_ERROR|Z_ERROR)\(([0-9.]+)\)", r"\1(0)", text)
    law0 = parity_law(t0, dets) if dets else {(): 1.0}
    refs = []
    for i in range(len(dets)):
        v = {k[i] for k, p in law0.items() if p > 1e-12}
        refs.append(v.pop() if len(v) == 1 else 0)
    refs += [0] * (len(cols) - len(dets))
    return {tuple(b ^ c for b, c in zip(k, refs)): p for k, p in law.items()}, len(dets), len(obs), len(decs)


@pytest.mark.parametrize("seed", sorted(_CAMPAIGN))
@pytest.mark.parametrize("mode", ["sample", "barrier"])
def test_campaign_obs_decision_joint(seed, mode):
    text = _CAMPAIGN[seed]
    law, nd, no, nk = _ref_joint_law(text)
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    if mode == "sample":
        b = s.sample(60_000, seed=3); d, o, dec = b.dets, b.obs, b.decisions
    else:
        b = s.sample_barrier(60_000, seed=3); d, o, dec = b.dets(), b.obs(), b.decisions()
    M = np.concatenate([_unpack(d, nd), _unpack(o, no), _unpack(dec, nk)], 1)
    zs = [max_z(M, {k[:nd] + (k[nd] ^ f,) + k[nd + 1:]: p for k, p in law.items()}) for f in (0, 1)]
    best = min(zs)
    assert best[1] == 0 and best[0] < 5, zs
