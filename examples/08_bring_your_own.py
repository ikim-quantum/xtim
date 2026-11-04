"""Bring your own protocol: declare the output, close the frame, score.

A T state is teleported through a Bell pair into a 3-qubit repetition code (qubits 2, 3, 4), with noise. Three steps:
  1. declare the output register: OUTPUT_QUBITS out 2 3 4;
  2. close the teleportation frame with Pauli feedback from the Bell measurement (as for a deterministic Stim
     detector), otherwise RecordFidelity refuses with RF_OUTPUT_NOT_STRICT and names what to fix;
  3. sample, post-select on the repetition-code checks, and score the output exactly.
How to read the refusal and pick each feedback (X byproduct -> CX, Z -> CZ; measurement m -> rec[-(M - m)]):
docs/usage.md, "Closing a frame".

Run:  python examples/08_bring_your_own.py
"""
import numpy as np

import xtim

PROTOCOL = """# a noisy T state on qubit 0
RX 0
T 0
DEPOLARIZE1(0.01) 0
# a Bell pair on qubits 1, 2; Bell measurement of qubits 0, 1
R 1 2
H 1
CX 1 2
DEPOLARIZE2(0.01) 1 2
CX 0 1
H 0
M 0 1
"""
CLOSE_FRAME = """# close the frame: X^m1 Z^m0 on the teleported qubit
CX rec[-1] 2
CZ rec[-2] 2
"""
ENCODE = """# encode into the repetition code (X_L = X2 X3 X4), noisy, then check Z2Z3 and Z3Z4 with ancillas 5, 6
R 3 4
CX 2 3 2 4
X_ERROR(0.01) 2 3 4
R 5 6
CX 2 5 3 5 3 6 4 6
M 5 6
DETECTOR rec[-2]
DETECTOR rec[-1]
OUTPUT_QUBITS out 2 3 4
"""

if __name__ == "__main__":
    try:
        xtim.RecordFidelity(xtim.compile_twirl_sampler(PROTOCOL + ENCODE, selfcheck=0))
    except xtim.XtimFidelityRefusal as e:
        print("without feedback:", e.code)
    sampler = xtim.compile_twirl_sampler(PROTOCOL + CLOSE_FRAME + ENCODE)
    buf = sampler.sample_barrier(100_000, seed=1)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    keep = np.flatnonzero(~dets.any(axis=1))
    F = xtim.RecordFidelity(sampler).evaluate(buf, shot_index=keep)
    print(f"with feedback: accepted {len(keep) / len(dets):.1%}, post-selected 1-F = {1 - F.mean():.2e}")
    s0 = xtim.compile_twirl_sampler(PROTOCOL + CLOSE_FRAME + ENCODE, p_factor=0.0, selfcheck=0)
    F0 = xtim.RecordFidelity(s0).evaluate(s0.sample_barrier(1000, seed=2))
    assert np.all(np.abs(F0 - 1.0) < 1e-12)
    print("noiseless check: F = 1 on every shot")
