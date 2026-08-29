"""Post-build smoke test for an installed `xtim` wheel.

Self-contained (no data files): run as `python smoke_test.py`. Used as the
cibuildwheel test-command so every wheel is exercised on its target platform.
"""
import numpy as np

import xtim


def main() -> None:
    # (1) Clifford sampling path: prepare |1>, measure Z -> always 1.
    c = xtim.Circuit("R 0\nX 0\nM 0\n")
    meas = c.compile_sampler(seed=1).sample(256)
    assert meas.shape == (256, 1) and bool(meas.all()), meas

    # (2) Magic + expectation path (exercises invisible reference compilation):
    # |T> = T|+> has <X> = cos(pi/4) = 1/sqrt(2), emitted as a RAW per-shot
    # expectation. Noiseless => deterministic, so every shot equals 1/sqrt(2).
    t = xtim.Circuit("RX 0\nT 0\nPAULI_EXPECTATION(0) X0\n")
    dets, obs, exps = t.compile_detector_sampler(seed=2).sample(
        4000, separate_observables=True, return_expectations=True)
    assert exps.shape == (4000, 1), exps.shape
    assert np.allclose(exps, 0.70710678, atol=1e-6), np.unique(exps)

    # (3) Decoder DEM + reject region (additive feature; needs stim, which the
    # minimal smoke env may lack -> guard on it).
    try:
        import stim  # noqa: F401
    except ImportError:
        pass
    else:
        rc = xtim.Circuit("R 0 1 2\nX_ERROR(0.05) 0 1 2\nM 0 1 2\n"
                          "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\n")
        res = rc.detector_error_model_with_reject()
        assert isinstance(res, xtim.DemWithReject), type(res)
        assert res.reject_detectors == sorted(res.reject_detectors)
        # a Pauli circuit has no not-Pauli-correctable faults to flag
        assert res.reject_detectors == [] and res.postselect_faults == []

    print("xtim smoke OK  |  engine", xtim.ENGINE_VERSION,
          "ref-format", xtim.REF_FORMAT_VERSION,
          "|  <X> on T|+> =", round(float(exps.mean()), 6), "(expect 0.707107)")


if __name__ == "__main__":
    main()
