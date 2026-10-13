"""Noise atoms follow the channel's support (fix/z-channel-atoms).

The propagation table used to build BOTH the X and the Z atom of every noise location, so a pure-Z
channel (Z_ERROR, PAULI_CHANNEL_1(0,0,p), a Z-only PAULI_CHANNEL_2, ...) before a T/CS/CCZ also
propagated its never-fired X atom, which could leave the simulable class and reject the whole
circuit. Now a location's atom set is exactly the Pauli factors its channel fires with nonzero
probability (a Y is its X atom then its Z atom); the forced-fault program alone keeps every atom,
since ANY Pauli may be forced at a site — a zero-probability one that leaves the class is refused
by name, never the circuit.

Oracles: tests/_dense_oracle.py (density matrix, DEM laws) and tests/_forced_oracle.py (forced
laws), no xtim code; and the exact identity "Z commutes with T": moving the channel past the T
must not change a single sampled byte.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

pytest.importorskip("stim")   # the dense oracles take Clifford unitaries from stim

import stim  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim  # noqa: E402
from _dense_oracle import Oracle, det_values  # noqa: E402
from _forced_oracle import aggregate, branches as forced_branches, law_shifts  # noqa: E402

ANCHOR = "R 9\nM 9\nDECISION(0) rec[-1]\nDETECTOR rec[-1]\n"
# Z-type channels: the X atom (or the first qubit's X atom) has probability zero.
Z_CHANNELS = ["Z_ERROR(0.1) 0", "PAULI_CHANNEL_1(0,0,0.1) 0",
              "PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0) 0 1",      # ZI only
              "PAULI_CHANNEL_2(0,0,0.04,0,0,0,0,0,0,0,0,0.05,0,0,0.02) 0 1"]  # IZ, ZI, ZZ


def _barrier(text, shots=4000, seed=5):
    T = _xtim.TwirlSampler(text, 1.0, "", 0, False, "")
    b = T.sample_barrier(shots, seed)
    ow = list(T.output_wires())
    reads = [([w], [], 0) for w in ow] + [([], [w], 0) for w in ow] + [([w], [w], 1) for w in ow]
    return np.asarray(b.decisions()).tobytes(), np.asarray(b.pauli_expectations_all(reads)).tobytes()


@pytest.mark.parametrize("ch", Z_CHANNELS)
def test_z_channel_before_t_barrier_equals_commuted(ch):
    """The review repro: RX; <Z-type noise>; T; H; OUTPUT_QUBITS was refused (state retention needs
    the diagonal class, and the never-fired X atom left it) while the same channel after the T was
    accepted. Z commutes with T, so both must now give the SAME bytes."""
    before = ANCHOR + "RX 0\nRX 1\n" + ch + "\nT 0\nH 0\nOUTPUT_QUBITS out 0 1\n"
    after = ANCHOR + "RX 0\nRX 1\nT 0\n" + ch + "\nH 0\nOUTPUT_QUBITS out 0 1\n"
    assert _barrier(before) == _barrier(after)


def _oracle_det_law(text):
    O = Oracle(text)
    O.run()
    rd = O.record_dist()
    On = Oracle(text, noiseless=True)
    On.run()
    rd0 = On.record_dist()
    r0 = max(rd0, key=rd0.get)
    assert rd0[r0] > 1 - 1e-12, "test circuit must have deterministic detectors"
    sets = O.detectors + [O.observables[k] for k in sorted(O.observables)]
    ref = det_values(r0, sets)
    law = collections.defaultdict(float)
    for r, p in rd.items():
        law[tuple(a ^ b for a, b in zip(det_values(r, sets), ref))] += p
    return law, len(O.detectors), len(sets)


def _dem_law(dem_text, nd, nt):
    law = {0: 1.0}
    for inst in stim.DetectorErrorModel(dem_text).flattened():
        if inst.type != "error":
            continue
        p, m = inst.args_copy()[0], 0
        for t in inst.targets_copy():
            m ^= (1 << t.val) if t.is_relative_detector_id() else (1 << (nd + t.val))
        nxt = collections.defaultdict(float)
        for k, q in law.items():
            nxt[k] += q * (1 - p)
            nxt[k ^ m] += q * p
        law = nxt
    return {tuple((k >> i) & 1 for i in range(nt)): q for k, q in law.items()}


CCZ_SANDWICH = """R 0 1 2
H 0 1 2
CCZ 0 1 2
{ch}
CCZ 0 1 2
H 0 1 2
M 0 1 2
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


# Random mirror circuits (seeded generator, 2026-10-12) whose DEM export a80dd33 REJECTED: a Z-type
# channel's never-fired X atom left the diagonal class. Every fired atom stays in it.
NEWLY_DEM = [
    "R 0 1 2 3\nH 0 1 2 3\nCCZ 0 1 3\nH 3\nPAULI_CHANNEL_2(0,0,0.033333,0.033333,0,0,0.033333,0,0,0,0,0,0,0,0) 3 0\n"
    "X 0\nH 0\nH 0\nPAULI_CHANNEL_1(0,0,0.1) 1\nS 2\nS_DAG 2\nH 0\nH 0\nX 0\nH 3\nCCZ 0 1 3\nH 0 1 2 3\n"
    "PAULI_CHANNEL_2(0,0,0,0,0,0,0,0.033333,0,0,0,0.033333,0,0,0) 0 2\nM 0 1 2 3\nDETECTOR rec[-4]\n"
    "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "R 0 1\nH 0 1\nPAULI_CHANNEL_2(0,0,0,0.016667,0,0,0,0.016667,0,0,0,0.016667,0,0,0) 1 0\nCZ 0 1\n"
    "Z_ERROR(0.05) 0\nT 1\nZ_ERROR(0.05) 1\nSWAP 1 0\nX 0\nSWAP 0 1\nX 1\nX 1\nSWAP 0 1\nX 0\n"
    "PAULI_CHANNEL_1(0,0,0.05) 1\nSWAP 1 0\nT_DAG 1\nCZ 0 1\nH 0 1\nZ_ERROR(0.05) 1\nM 0 1\n"
    "DETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "R 0 1\nH 0 1\nT 1\nX 1\nT 0\nCZ 0 1\nX 1\nX_ERROR(0) 0\nT 1\nCZ 1 0\nCZ 1 0\nT_DAG 1\nX 1\nCZ 0 1\n"
    "PAULI_CHANNEL_1(0,0,0.01) 1\nT_DAG 0\nX 1\nT_DAG 1\nH 0 1\nZ_ERROR(0.01) 1\nM 0 1\nDETECTOR rec[-2]\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
]


@pytest.mark.parametrize("text", NEWLY_DEM, ids=["ccz4", "t2_swaps", "t4_cz"])
def test_dem_newly_accepted_exact_vs_oracle(text):
    """DEM export (no PPR retry) rejected these; only the fired atoms are propagated now, and the
    exported DEM's detector/observable law equals the dense oracle's exactly."""
    law, nd, nt = _oracle_det_law(text)
    dem = xtim.Circuit(text).detector_error_model_text(include_expectations=False)
    dm = _dem_law(dem, nd, nt)
    assert max(abs(dm.get(k, 0.0) - law.get(k, 0.0)) for k in set(dm) | set(law)) < 1e-12


def test_samplers_z_channel_inside_ccz_vs_oracle():
    """Exact / auto detector samplers and the raw record sampler on a re-routed circuit (its only
    off-diagonal atom was the never-fired X atom): record laws vs the dense oracle."""
    text = CCZ_SANDWICH.format(ch="PAULI_CHANNEL_1(0,0,0.2) 1\nZ_ERROR(0.1) 0")
    law, nd, nt = _oracle_det_law(text)
    c = xtim.Circuit(text)
    shots = 20000
    for eng in ("exact", "auto"):
        d, o = c.compile_detector_sampler(seed=11, engine=eng).sample(shots, separate_observables=True)[:2]
        rows = np.concatenate([np.asarray(d, np.uint8), np.asarray(o, np.uint8)], axis=1)
        cnt = collections.Counter(map(tuple, rows.tolist()))
        for k in set(cnt) | set(law):
            p = law.get(k, 0.0)
            assert p > 1e-12 or cnt.get(k, 0) == 0, (eng, k)
            if 1e-12 < p < 1 - 1e-12:
                assert abs(cnt.get(k, 0) - shots * p) / np.sqrt(shots * p * (1 - p)) < 6, (eng, k)


def test_zero_probability_channels_sample():
    """X_ERROR(0) / Z_ERROR(0) / DEPOLARIZE1(0) / all-zero PAULI_CHANNEL_1 build no atom at all; every
    engine still samples (a zero-probability channel is never fired, never indexed)."""
    text = ("RX 0\nR 1\nX_ERROR(0) 0\nZ_ERROR(0) 1\nDEPOLARIZE1(0) 0\nPAULI_CHANNEL_1(0,0,0) 1\n"
            "T 0\nH 0\nH 0\nT_DAG 0\nMX 0\nDETECTOR rec[-1]\nM 1\nDETECTOR rec[-1]\n")
    c = xtim.Circuit(text)
    for eng in ("exact", "auto", "twirl"):
        d = np.asarray(c.compile_detector_sampler(seed=3, engine=eng).sample(200))
        assert not d.any(), eng
    assert "error(" not in c.detector_error_model_text()


def test_forced_any_pauli_contract_and_named_refusal():
    """Forced faults keep their documented contract (ANY Pauli at any site, the probability is
    irrelevant): a zero-probability Pauli whose atom stays simulable is answered — checked against
    the forced oracle — while one whose atom leaves the class is refused BY NAME instead of
    rejecting the circuit."""
    # Z_ERROR(0.1) 0 between the CCZs: its X atom (probability zero) stays PPR-simulable.
    text = CCZ_SANDWICH.format(ch="Z_ERROR(0.1) 1")
    sim = xtim.Circuit(text).compile_exact_branches()
    cands = None
    for p in "XYZ":
        law = sim.exact_branches([(0, 0, p)])
        cands = law_shifts(law, aggregate(forced_branches(text, 3, {(0, 0): p}, set())), cands)
        assert cands, p
    # A zero-probability X atom that leaves the class EVEN WITH the PPR retry (review repro): the
    # circuit compiles (the sampler never fires X there), Z is answered exactly, X and Y are
    # refused by name at the user's (site, target, Pauli) address.
    text = "Z_ERROR(0) 0\nCS_DAG 1 0\nSQRT_X_DAG 0\nT_DAG 0\nX 0\nT 0\nMX 0\nM 1\n"
    sim = xtim.Circuit(text).compile_exact_branches()
    sim.exact_branches([(0, 0, "Z")])
    for p in ("X", "Y"):
        with pytest.raises(ValueError) as ei:
            sim.exact_branches([(0, 0, p)])
        msg = str(ei.value)
        assert f"noise site 0 target 0 Pauli '{p}' has probability zero in its channel" in msg, msg
        assert "leaves the simulable class" in msg and "exact_branches: exact_branches" not in msg, msg
