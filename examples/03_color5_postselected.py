"""Distance-5 colour-code cultivation (SOFT's d=5 circuit, Apache-2.0): post-selected fidelity.

`color5_cultivation` grows a T state from d=3 to d=5 with double-checking, then moves the 19-qubit output onto fresh
wires (OUTPUT_QUBITS). SOFT's final logical-Y check is removed, but its noiseless transversal T-dagger layer is
kept: the noiseless output is therefore the logical |+i> (the cultivated T state rotated by that final layer), and the
fidelity reported equals the cultivated T state's fidelity. The output does have a deterministic logical
observable (Y_L); the circuit declares no OBSERVABLE_INCLUDE and is scored by post-selection plus the exact fidelity
of the accepted output. At p = 1e-3 most shots are discarded; the accepted output's fidelity is printed per noise
level.

Run:  python examples/03_color5_postselected.py
"""
import numpy as np

import xtim

SHOTS = 100_000


def run(p_factor, shots=SHOTS, seed=5):
    sampler = xtim.compile_twirl_sampler(xtim.load_example("color5_cultivation").text, p_factor=p_factor)
    buf = sampler.sample_barrier(shots, seed=seed)
    dets = np.unpackbits(buf.dets(), axis=1, bitorder="little")[:, :sampler.num_detectors]
    keep = np.flatnonzero(~dets.any(axis=1))
    F = xtim.RecordFidelity(sampler).evaluate(buf, shot_index=keep)
    return len(keep) / shots, (1.0 - F.mean()) if len(keep) else float("nan"), len(keep)


if __name__ == "__main__":
    for pf, label in ((1.0, "p = 1e-3"), (2.0, "p = 2e-3")):
        acc, infid, n = run(pf)
        print(f"{label}: accepted {acc:6.2%} ({n} shots)   post-selected 1-F = {infid:.2e}")
    acc, infid, _ = run(0.0, shots=500)
    assert acc == 1.0 and abs(infid) < 1e-12
    print("noiseless check: F = 1 on every shot")
