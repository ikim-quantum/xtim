"""[[8,3,2]] cube code: a post-selected transversal-CCZ magic state on three logical qubits.

`cube_ccz` prepares |+++>_L on the 8 cube vertices (Z-stabilizers measured, syndrome fed back) and applies the
logical CCZ as T on {0,4,5,6} and T_DAG on {1,2,3,7}: the output is CCZ|+++>_L. The bundled circuit is noiseless;
this script adds depolarizing noise after each physical T gate and one noiseless round of all five stabilizers as
detectors. The code has distance 2: post-selection heralds every single fault, so the accepted infidelity is O(p^2)
while the keep-all infidelity is O(p).

Run:  python examples/05_ccz_cube.py
"""
import numpy as np

import xtim

SHOTS = 100_000
STABS = ["X0*X1*X2*X3*X4*X5*X6*X7", "Z0*Z1*Z6*Z7", "Z0*Z2*Z3*Z6", "Z2*Z4*Z6*Z7", "Z0*Z2*Z5*Z7"]


def circuit(p):
    out = []
    for line in xtim.load_example("cube_ccz").text.splitlines():
        if line.startswith("OUTPUT_QUBITS"):                       # the syndrome round goes before the declaration
            out += [f"MPP {s}" for s in STABS] + [f"DETECTOR rec[{-k}]" for k in range(5, 0, -1)]
        out.append(line)
        if line.startswith(("T ", "T_DAG ")) and p > 0:            # noise on the physical T gates only
            out.append(f"DEPOLARIZE1({p}) " + line.split(maxsplit=1)[1])
    return "\n".join(out) + "\n"


def run(p, shots=SHOTS, seed=3):
    sampler = xtim.compile_twirl_sampler(circuit(p))
    buf = sampler.sample_barrier(shots, seed=seed)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    accepted = ~dets.any(axis=1)
    F = xtim.RecordFidelity(sampler).evaluate(buf)
    return accepted.mean(), 1.0 - F[accepted].mean(), 1.0 - F.mean()


if __name__ == "__main__":
    for p in (1e-3, 1e-2, 3e-2):
        acc, kept, every = run(p)
        print(f"p = {p:5.0e}: accepted {acc:6.1%}   1-F accepted {kept:.2e}   1-F all shots {every:.2e}")
    acc, kept, _ = run(0.0, shots=1000)
    assert acc == 1.0 and abs(kept) < 1e-12
    print("noiseless check: F = 1 on every shot")
