"""H-eigenstate cultivation: three controlled-Hadamard checks on a noisy |H> state.

`ch_cultivation` prepares |H> (the +1 eigenstate of the Hadamard) with T and Cliffords, then measures H three times
by Hadamard tests (an ancilla in |+>, CH onto the data, MX). Noise before each check is heralded by the checks;
post-selecting on all three reading +1 suppresses the infidelity of the output qubit.

Run:  python examples/04_h_cultivation.py
"""
import numpy as np

import xtim

SHOTS = 200_000


def run(p, shots=SHOTS, seed=3):
    text = xtim.scale_noise(xtim.load_example("ch_cultivation").text, p)
    sampler = xtim.compile_twirl_sampler(text)
    buf = sampler.sample_barrier(shots, seed=seed)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    accepted = ~dets.any(axis=1)
    F = xtim.RecordFidelity(sampler).evaluate(buf)
    return accepted.mean(), 1.0 - F[accepted].mean(), 1.0 - F.mean()


if __name__ == "__main__":
    for p in (1e-3, 1e-2, 5e-2):
        acc, kept, every = run(p)
        print(f"p = {p:5.0e}: accepted {acc:6.1%}   1-F accepted {kept:.2e}   1-F all shots {every:.2e}")
    acc, kept, _ = run(0.0, shots=1000)
    assert acc == 1.0 and abs(kept) < 1e-12
    print("noiseless check: F = 1 on every shot")
