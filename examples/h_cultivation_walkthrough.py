#!/usr/bin/env python3
"""Cultivating the Hadamard-eigenstate magic state by MEASURING the logical Hadamard.

This is the walkthrough for xtim's controlled-Hadamard (`CH`) support — the first
non-diagonal level-3 gate. The protocol is H-eigenstate cultivation:

    1. prepare the raw magic state (necessarily T-form — `T` is the only
       single-qubit non-Clifford in the gate set) and Clifford-ALIGN it onto the
       Hadamard axis:   |H> = (H S H) T H |0>,   Bloch (1/sqrt2, 0, 1/sqrt2);
    2. repeatedly MEASURE the logical Hadamard via a Hadamard test — ancilla |+>,
       `CH` onto the data, `MX` the ancilla — and post-select the +1 outcomes.

Because the data is an H-EIGENSTATE, the prep magic and the check magic commute
(xtim's simulable-class condition — docs/xtim_simulable_class.md), the ancillas
never entangle noiselessly (chi stays 2 no matter how many rounds), and every
check detector is deterministic: the perfect post-selection herald. Feed the
UNALIGNED T-state into the same check and the magic anticommutes — xtim rejects,
and that rejection IS the protocol's alignment requirement. Both facts are shown
below, using only the public surface: `load_example` + `diagnose()` + the sampler.

Run it:  `python examples/h_cultivation_walkthrough.py`
"""
import numpy as np

import xtim

TARGET = np.array([1 / np.sqrt(2.0), 1 / np.sqrt(2.0)])   # <X>, <Z> of |H>


def main(shots: int = 50_000, seed: int = 7) -> float:
    # ── Step 1: load the bundled protocol and ask the tool what it prepares ──────
    c = xtim.load_example("ch_cultivation")
    report = c.diagnose()
    print(report)
    # 3 check rounds -> 3 detectors, ALL deterministic (post-selectable), and the
    # payload columns are sign-constant at the |H> Bloch components.
    assert len(report.deterministic_detectors) == 3
    assert len(report.gauge_detectors) == 0
    for e, t in zip(report.expectations, TARGET):
        assert e.sign_constant and abs(e.signed_value - t) < 1e-9

    # ── Step 2: sample with noise, post-select the checks, score the state ───────
    dets, obs, exps = c.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True, return_expectations=True)
    keep = ~dets.any(axis=1)                       # every check must read +1
    acceptance = keep.mean()
    bloch = exps[keep].mean(axis=0)                # post-selected (<X>, <Z>)
    F = (1 + TARGET @ bloch) / 2                   # exact: <Y> = 0 for this target
    print(f"\n{shots} shots at the baked p=1e-3: acceptance = {acceptance:.4f}, "
          f"post-selected Bloch = ({bloch[0]:+.6f}, {bloch[1]:+.6f}), F = {F:.6f}")
    assert acceptance > 0.98                       # ~6 noise sites at p=1e-3
    assert F > 0.995                               # the heralded state IS |H>

    # ── Step 3: the physics lesson — alignment is the class condition ────────────
    # The same Hadamard check applied to the UNALIGNED raw T-state: the T's magic
    # axis anticommutes with the check's, so there is no single commuting magic
    # layer. xtim rejects — and the protocol, for the same reason, wouldn't work
    # (T|+> is not an H-eigenstate; the check would not be deterministic).
    unaligned = xtim.Circuit("H 0\nT 0\nH 1\nCH 1 0\nMX 1\nDETECTOR rec[-1]\n"
                             "PAULI_EXPECTATION(0) X0")
    try:
        unaligned.compile_detector_sampler(seed=seed).sample(16)
        raise AssertionError("the unaligned T-state->CH circuit must reject")
    except xtim.XtimRejectError as e:
        assert e.kind == "class"
        print("\nunaligned T-state into the H-check rejects, as the physics demands:")
        print(f"  hint: {e.hint[:150]}...")

    print("\nOK: cultivated the H-eigenstate; the class boundary enforced alignment.")
    return float(F)


if __name__ == "__main__":
    main()
