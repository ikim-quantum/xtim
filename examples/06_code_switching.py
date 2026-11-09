"""Code switching from the [[15,1,3]] quantum Reed-Muller code to the Steane code.

`code_switching_faithful` applies a transversal T on the qRM block (noisy), switches the logical state onto a Steane
block (transversal CX, then X-measurement of the qRM block), and checks the Steane block noiselessly at the end.
The X-measurement outcomes of qRM qubits 0..6 decide a logical-Z byproduct on the Steane block; the circuit closes
that frame with Pauli feedback (`CZ rec[-k] q`) onto Z_L = Z15*Z16*Z17, so the Steane output is a fixed state.
`code_switching_rate` is the count-matched twin (Z-basis readout; per-qubit CX feedback).

The script reports the acceptance (no detector fired) and the exact output fidelity of the accepted shots and of
all shots.

Run:  python examples/06_code_switching.py
"""
import numpy as np

import xtim

SHOTS = 200_000


def run(name, shots=SHOTS, seed=7, p_factor=1.0):
    sampler = xtim.compile_twirl_sampler(xtim.load_example(name).text, p_factor=p_factor)
    buf = sampler.sample_barrier(shots, seed=seed)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    accepted = ~dets.any(axis=1)
    F = xtim.RecordFidelity(sampler).evaluate(buf)
    return accepted.mean(), 1.0 - F[accepted].mean(), 1.0 - F.mean()


if __name__ == "__main__":
    for name in ("code_switching_faithful", "code_switching_rate"):
        acc, kept, every = run(name)
        print(f"{name:24s} accepted {acc:6.1%}   1-F accepted {kept:.2e}   1-F all shots {every:.2e}")
    acc, kept, _ = run("code_switching_faithful", shots=1000, p_factor=0.0)
    assert acc == 1.0 and abs(kept) < 1e-12
    print("noiseless check: F = 1 on every shot")
