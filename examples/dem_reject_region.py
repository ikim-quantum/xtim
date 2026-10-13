#!/usr/bin/env python3
"""Decode + flag workflow when a fault GENUINELY changes a logical magnitude.

`Circuit.detector_error_model_with_reject()` returns:
  - res.dem               : a clean Stim DEM of the Pauli-CORRECTABLE faults (decode this);
  - res.postselect_faults : the not-Pauli-correctable faults to FLAG (each a detector
                            signature + probability) — YOU choose the policy:
                              * post-select  (sound, zero residual), or
                              * decode-and-budget (carry their total probability as a floor);
  - res.reject_detectors / res.keep_mask(dets) : the blunt CONSERVATIVE post-select
                            convenience (reject a shot if any flagged detector fired);
  - res.dropped_expectation_columns : PAULI_EXPECTATION columns with no DEM concept (their
                            noiseless per-shot value varies shot by shot); empty here.

A fault is flagged only when, judged PER SHOT (the value the sampler reports, conditioned on
every measurement outcome), it changes the magnitude of a declared expectation on some branch
and fires a detector. The circuit below has exactly one such fault family: an X/Y error between
T and T_DAG on a Bell half takes the per-shot <X0> (frame-corrected) from 1 to 0, and the Z0Z1
parity check catches it. (The bundled cultivation circuits have NO such fault: their former
"reject region" was an artifact of judging faults by read-averaged expectations, fixed in
3.1.13 — see the CHANGELOG.)

    python dem_reject_region.py

See docs/xtim_dem_reject.md.
"""
import numpy as np

import xtim

CIRCUIT = """\
RX 0
R 1
CX 0 1
T 0
DEPOLARIZE1(0.03) 0
T_DAG 0
MPP(0.01) Z0*Z1
MPP(0.01) Z0*Z1
DETECTOR rec[-2]
DETECTOR rec[-2] rec[-1]
MX(0.005) 1
PAULI_EXPECTATION(0) X0 rec[-1]
"""


def main() -> None:
    c = xtim.Circuit(CIRCUIT)

    res = c.detector_error_model_with_reject(include_expectations=True)
    print(res.dem)
    budget = sum(f.probability for f in res.postselect_faults)
    print(f"flagged not-correctable  : {[(f.detectors, round(f.probability, 6)) for f in res.postselect_faults]}"
          f"  (total prob {budget:.4g} <- the floor if you DON'T post-select them)")
    print(f"conservative reject union: detectors {res.reject_detectors}")
    print(f"dropped columns          : {res.dropped_expectation_columns}")

    # The flag is GENUINE: forcing the X fault, the exact engine reports <X0> = 0 on every
    # shot (noiseless: 1) and detector D0 fires on every shot.
    br = c.compile_exact_branches().exact_branches([(0, 0, "X")])
    print(f"\nforced X fault: <X0> = {sorted(set(float(v) for v in np.round(br.expectations[:, 0], 6)))}, "
          f"D0 fires with probability {float(br.probabilities @ br.detectors[:, 0]):.0%}")

    shots = 200_000
    dets, _obs, exps = c.compile_detector_sampler(seed=1).sample(
        shots, separate_observables=True, return_expectations=True)
    keep = res.keep_mask(dets)
    print(f"\nacceptance rate: {keep.mean():.4f}  ({int(keep.sum())}/{shots} kept)")
    print("<X0> post-selected (accepted shots):", round(float(exps[keep, 0].mean()), 4))
    print("<X0> no post-select (all shots):    ", round(float(exps[:, 0].mean()), 4))

    # Decode the accepted shots with the Pauli-correctable DEM: its L0 column is the frame
    # flip of the expectation column (Z faults, the MX readout flip); correct the sign.
    try:
        import pymatching
        matcher = pymatching.Matching.from_detector_error_model(res.dem)
        flips = matcher.decode_batch(dets[keep])
        corrected = np.where(flips[:, c.expectation_columns[0]], -exps[keep, 0], exps[keep, 0])
        print("<X0> post-selected + decoded:       ", round(float(corrected.mean()), 4),
              "(the residual is the undetectable Z / readout-flip mass, not a decoder failure)")
    except ImportError:
        print("\n(install `pymatching` to decode res.dem)")


if __name__ == "__main__":
    main()
