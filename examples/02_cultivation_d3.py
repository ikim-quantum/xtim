"""Magic-state cultivation at distance 3: post-selection and the fidelity of the accepted output.

`cultivation_d3_faithful` is a d=3 cultivation circuit (T-state, colour code) at physical error p = 1e-3;
`cultivation_d3_rate` is its Clifford twin (S in place of T): the same circuit and noise with a stabilizer output.
Each declares its 7-qubit output register (OUTPUT_QUBITS). xtim reports the acceptance and the exact fidelity of the
accepted shots and of all shots (a fired detector by itself costs nothing; only the output is compared).

Run:  python examples/02_cultivation_d3.py
"""
import numpy as np

import xtim

SHOTS = 200_000


def run(name, shots=SHOTS, seed=11, p_factor=1.0):
    sampler = xtim.compile_twirl_sampler(xtim.load_example(name).text, p_factor=p_factor)
    buf = sampler.sample_barrier(shots, seed=seed)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    accepted = ~dets.any(axis=1)
    F = xtim.RecordFidelity(sampler).evaluate(buf)                 # float64 per shot
    return accepted.mean(), 1.0 - F[accepted].mean(), 1.0 - F.mean()


if __name__ == "__main__":
    for name in ("cultivation_d3_faithful", "cultivation_d3_rate"):
        acc, infid_kept, infid_all = run(name)
        print(f"{name:24s} accepted {acc:6.1%}   1-F accepted {infid_kept:.2e}   1-F all shots {infid_all:.2e}")
    acc, infid_kept, _ = run("cultivation_d3_faithful", shots=1000, p_factor=0.0)
    assert acc == 1.0 and abs(infid_kept) < 1e-12
    print("noiseless check: F = 1 on every shot")
