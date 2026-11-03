"""15-to-1 distillation: sample, post-select, and score the output state exactly.

The bundled `distillation_15_1_3` circuit prepares a [[15,1,3]] quantum Reed-Muller code state, applies a transversal
T with noisy physical T gates, and measures the X-stabilizers as detectors. Post-selecting on a trivial syndrome
distills: the infidelity of the accepted output falls from ~p to ~35 p^3.

For each p, xtim samples the circuit and scores every shot's 15-qubit output register against the noiseless output
(`RecordFidelity`), so the post-selected infidelity is resolved far below 1/shots. A mean 1-F of exactly 0 means no
scored shot had an output error; with N scored shots the rate is below about 3/N (95% confidence).

Run:  python examples/01_quickstart_distillation.py
"""
import numpy as np

import xtim

SHOTS = 200_000


def run(p, shots=SHOTS, seed=7):
    """(acceptance, post-selected 1-F) at physical T-gate error p."""
    text = xtim.scale_noise(xtim.load_example("distillation_15_1_3").text, p)
    sampler = xtim.compile_twirl_sampler(text)
    buf = sampler.sample_barrier(shots, seed=seed)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    keep = np.flatnonzero(~dets.any(axis=1))                    # post-select: no detector fired
    F = xtim.RecordFidelity(sampler).evaluate(buf, shot_index=keep)
    return len(keep) / shots, 1.0 - F.mean()


if __name__ == "__main__":
    print(f"{'p':>8} {'accepted':>9} {'1-F (post-selected)':>20} {'35 p^3':>9}")
    for p in (1e-3, 3e-3, 1e-2, 3e-2):
        acc, infid = run(p)
        print(f"{p:8.0e} {acc:9.1%} {infid:20.2e} {35 * p**3:9.2e}")
    acc, infid = run(0.0, shots=1000)                             # noiseless: every shot is the ideal output
    assert acc == 1.0 and abs(infid) < 1e-12
    print("noiseless check: F = 1 on every shot")
