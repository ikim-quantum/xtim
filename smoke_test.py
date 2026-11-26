"""Post-build smoke test for an installed `xtim` wheel (self-contained: bundled examples only).

Run as `python smoke_test.py`. Checks import, compile, sample, sample_barrier + RecordFidelity on the accepted shots,
and the postselect= keyword, on the bundled cultivation_d3_faithful example.
"""
import numpy as np

import xtim


def main() -> None:
    # (1) Clifford detector path: prepare |1>, measure Z -> the reference-relative detector never fires.
    dets = xtim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]\n").compile_detector_sampler(seed=1).sample(256)
    assert dets.shape == (256, 1) and not bool(dets.any()), dets

    # (2) A bundled magic circuit through the twirl sampler.
    text = str(xtim.load_example("cultivation_d3_faithful"))
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    nd = s.num_detectors
    assert nd > 0
    shots = 400
    dets, obs = s.sample(shots, seed=3)[:2]
    assert np.asarray(dets).shape[0] == shots and np.asarray(obs).shape[0] == shots

    # (3) sample_barrier + RecordFidelity on the accepted shots (every detector 0).
    b = s.sample_barrier(shots, seed=3)
    assert b.num_shots == shots
    d = np.unpackbits(np.asarray(b.dets()), axis=1, bitorder="little")[:, :nd]
    accepted = np.flatnonzero(~d.any(1))
    assert 0 < len(accepted) < shots, len(accepted)
    F = np.asarray(xtim.RecordFidelity(s).evaluate(b, accepted))
    assert F.shape == (len(accepted),) and np.isfinite(F).all() and (F >= -1e-9).all() and (F <= 1 + 1e-9).all(), F

    # (4) postselect= on one call: the shots kept are exactly the all-detectors-zero ones, in shot order.
    s2 = xtim.compile_twirl_sampler(text, selfcheck=0)
    buf = s2.sample_barrier(shots, seed=3, postselect=list(range(nd)))
    c = buf.postselection
    assert (c["attempts"], c["passed"], c["discarded"]) == (shots, len(accepted), shots - len(accepted)), c
    assert buf.num_shots == len(accepted)
    assert not np.unpackbits(np.asarray(buf.dets()), axis=1, bitorder="little")[:, :nd].any()

    print("xtim smoke OK  |  engine", xtim.ENGINE_VERSION, "|  accepted", len(accepted), "of", shots,
          "|  mean fidelity", round(float(F.mean()), 6))


if __name__ == "__main__":
    main()
