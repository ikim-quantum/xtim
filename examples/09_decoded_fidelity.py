"""Decoded fidelity: a decoder's predicted observable flips become a Pauli correction on the output.

A 3-qubit repetition code holds |000> under bit-flip noise; one round of Z2-parity checks, then a noiseless copy of
each qubit's Z value onto a fresh ancilla gives two more detectors and three deterministic observables
(OBSERVABLE_INCLUDE(i) = qubit i's Z). A matching decoder (PyMatching, from xtim's detector error model) predicts
the observable flips; per observable the user names the output Pauli that undoes its flip (its "dual": X on qubit i);
`correction_from_observables` turns predictions into per-shot output corrections, and `RecordFidelity.evaluate(...,
correction=...)` scores the corrected output.

Decoding needs OBSERVABLE_INCLUDEs that are deterministic without noise (the detector error model refuses others).

Needs the [decode] extra (stim, pymatching).  Run:  python examples/09_decoded_fidelity.py
"""
import numpy as np
import pymatching

import xtim

TEXT = """R 0 1 2
X_ERROR(0.02) 0 1 2
R 3 4
CX 0 3 1 3 1 4 2 4
M 3 4
DETECTOR rec[-2]
DETECTOR rec[-1]
R 5 6 7
CX 0 5 1 6 2 7
M 5 6 7
DETECTOR rec[-3] rec[-2] rec[-5]
DETECTOR rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-3]
OBSERVABLE_INCLUDE(1) rec[-2]
OBSERVABLE_INCLUDE(2) rec[-1]
OUTPUT_QUBITS out 0 1 2
"""

if __name__ == "__main__":
    sampler = xtim.compile_twirl_sampler(TEXT)
    buf = sampler.sample_barrier(100_000, seed=1)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]

    dem = xtim.Circuit(TEXT).detector_error_model(approximate_disjoint_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    pred = matching.decode_batch(dets)[:, :sampler.num_observables]   # the circuit's observables come first

    duals = np.zeros((sampler.num_observables, 2, 3), np.uint8)      # [observable, (X row, Z row), output qubit]
    for i in range(3):
        duals[i, 0, i] = 1                                            # observable i flipped -> apply X on qubit i
    correction = xtim.correction_from_observables(sampler, pred, duals)

    rf = xtim.RecordFidelity(sampler)
    print(f"raw 1-F = {1 - rf.evaluate(buf).mean():.2e}   decoded 1-F = {1 - rf.evaluate(buf, correction=correction).mean():.2e}")

    s0 = xtim.compile_twirl_sampler(TEXT, p_factor=0.0, selfcheck=0)
    assert np.all(np.abs(xtim.RecordFidelity(s0).evaluate(s0.sample_barrier(500, seed=2)) - 1.0) < 1e-12)
    print("noiseless check: F = 1 on every shot")
