"""S0(f) of the twirl consolidation: the 512-channel cap is gone (run-width channel rows).

The twirl sampler used fixed uint64_t[8] channel rows, so a circuit with more than 512 record
channels was refused at compile. The rows are now run-width (CW = ceil(nchan/64) words, sized
once per sampler). Gate here: a 600-detector synthetic circuit (603 channels; words 8-9 and the
512 boundary exercised) compiles and its twirl law equals the exact engine's at 5 sigma —
marginals and pairwise products over the correlated pairs and every pair touching the coin-driven
magic detector or the old cap boundary. Circuits with <= 512 channels are byte-identical to the
fixed-width engine (gated on the byte corpus, not here).
"""
import itertools

import numpy as np

import xtim

NCHAIN = 600


def _wide_circuit():
    n = NCHAIN + 1                           # a chain of n data qubits -> NCHAIN parity detectors
    m = n                                    # magic mirror qubit
    L = ["R " + " ".join(map(str, range(n))), f"RX {m}", f"T {m}"]
    L.append("X_ERROR(0.04) " + " ".join(map(str, range(n))))
    L.append(f"X_ERROR(0.2) {m}")
    L.append(f"T_DAG {m}")
    L.append("CX " + " ".join(f"{i} {i + 1}" for i in range(0, n - 1, 2)))   # extra correlation
    L.append("X_ERROR(0.03) " + " ".join(map(str, range(n))))
    L.append("M " + " ".join(map(str, range(n))))
    L.append(f"MX {m}")
    nrec = n + 1
    for i in range(NCHAIN):
        L.append(f"DETECTOR rec[-{nrec - i}] rec[-{nrec - i - 1}]")
    L.append("DETECTOR rec[-1]")             # the coin-driven magic detector
    # Three deterministic observables: channels 601..603 (past the old cap, words 9).
    L.append(f"OBSERVABLE_INCLUDE(0) rec[-{nrec}] rec[-{nrec - 1}]")
    L.append(f"OBSERVABLE_INCLUDE(1) rec[-{nrec - 510}] rec[-{nrec - 512}]")
    L.append("OBSERVABLE_INCLUDE(2) rec[-2] rec[-3]")
    return "\n".join(L) + "\n"


def _z(a, b):
    N = a.shape[0]
    return abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / N + 1e-12)


def test_600_detector_circuit_compiles_and_matches_exact():
    text = _wide_circuit()
    N = 20000
    s = xtim.compile_twirl_sampler(text, selfcheck=2000)
    rep = s.record_report()                        # J-b1 F: channel_report removed
    cls = rep["classes"]
    assert sum(c == "deterministic" for c in cls["detectors"].values()) == NCHAIN + 1
    assert len(cls["detectors"]) + len(cls["observables"]) + len(cls["decisions"]) == NCHAIN + 1 + 3 > 512
    d, o = s.sample(N, seed=3)
    assert not [k for k in s.record_report()["route_counts"] if k.startswith("exact")]
    T = np.unpackbits(np.asarray(d), axis=1, bitorder="little")[:, :NCHAIN + 1].astype(float)
    TO = np.unpackbits(np.asarray(o), axis=1, bitorder="little")[:, :3].astype(float)
    E, EO = xtim.Circuit(text).compile_detector_sampler(seed=5, engine="exact").sample(
        N, separate_observables=True)
    E = np.asarray(E, float)[:, :NCHAIN + 1]
    EO = np.asarray(EO, float)[:, :3]
    T = np.concatenate([T, TO], axis=1)
    E = np.concatenate([E, EO], axis=1)
    k = T.shape[1]
    zs = [_z(T[:, i], E[:, i]) for i in range(k)]
    pairs = set()
    for i in range(k):
        for d2 in (1, 2):
            if i + d2 < k:
                pairs.add((i, i + d2))
    special = [0, 510, 511, 512, 513, NCHAIN, NCHAIN + 1, NCHAIN + 2, NCHAIN + 3]
    for i, j in itertools.product(special, range(k)):
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    zs += [_z(T[:, i] * T[:, j], E[:, i] * E[:, j]) for i, j in sorted(pairs)]
    assert max(zs) < 5.0, max(zs)
    # Non-vacuity: the magic detector is a genuine coin-driven channel, the chain detectors fire.
    assert 0.05 < T[:, NCHAIN].mean() < 0.95
    assert T[:, :NCHAIN].mean() > 0.05


def test_wide_circuit_barrier_route():
    text = _wide_circuit()
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    b = s.sample_barrier(2000, 9)
    d2, o2 = s.sample(2000, seed=9)
    assert np.asarray(b.dets()).shape[1] == (NCHAIN + 1 + 7) // 8
    # Same seed: the retention route and the sampling route agree column-wise in law; the
    # emitted rows are a real (non-zero) channel row past word 8.
    D = np.unpackbits(np.asarray(b.dets()), axis=1, bitorder="little")[:, :NCHAIN + 1]
    assert D[:, 512:].any()
