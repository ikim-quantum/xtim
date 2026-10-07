#!/usr/bin/env python3
"""Decode + flag workflow for a magic-state-prep protocol.

`Circuit.detector_error_model_with_reject()` returns:
  - res.dem               : a clean Stim DEM of the Pauli-CORRECTABLE faults (decode this);
  - res.postselect_faults : the not-Pauli-correctable faults to FLAG (each a detector
                            signature + probability) — YOU choose the policy:
                              * post-select  (sound, zero residual), or
                              * decode-and-budget (carry their total probability as a floor);
  - res.reject_detectors / res.keep_mask(dets) : the blunt CONSERVATIVE post-select
                            convenience (reject a shot if any flagged detector fired).

xtim does not force a policy. This example shows the conservative post-select + decode path
and prints the budget you'd carry instead.

Run from the examples/ directory (the bundled circuit lives alongside this file):

    python dem_reject_region.py

See docs/xtim_dem_reject.md.
"""
import os

import numpy as np

import xtim

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolve the circuit relative to this file so the script runs from anywhere. In
# the shipped export the circuit lives beside this script; in the monorepo it is
# under `benchmarks/`.
_CANDIDATES = [
    os.path.join(HERE, "cultivation_d3_faithful.stim"),
    os.path.join(HERE, os.pardir, os.pardir, os.pardir,
                 "benchmarks", "cultivation_d3_faithful.stim"),
]
CIRCUIT = next(p for p in _CANDIDATES if os.path.exists(p))


def main() -> None:
    c = xtim.Circuit.from_file(CIRCUIT)

    res = c.detector_error_model_with_reject()
    # The magic value must NOT be a DEM observable (it is the payload, read from the
    # expectation channel); the default include_expectations=False guarantees that.
    assert res.dem.num_observables == c.num_observables, \
        "magic should not be a DEM observable — read it from the expectation channel"
    n_edges = sum(line.startswith("error(") for line in str(res.dem).splitlines())
    budget = sum(f.probability for f in res.postselect_faults)
    print(f"decodable DEM mechanisms : {n_edges}")
    print(f"flagged not-correctable  : {len(res.postselect_faults)} faults "
          f"(total prob {budget:.4g}  <- the floor if you DON'T post-select them)")
    print(f"conservative reject union: {len(res.reject_detectors)} detectors")

    shots = 50_000
    dets, _obs, exps = c.compile_detector_sampler(seed=1).sample(
        shots, separate_observables=True, return_expectations=True)

    # Policy = conservative post-select: keep shots where no flagged detector fired.
    keep = res.keep_mask(dets)
    print(f"\nacceptance rate: {keep.mean():.4f}  ({int(keep.sum())}/{shots} kept)")

    if keep.any() and exps is not None and exps.shape[1] >= 1:
        # The two endpoints a decider weighs: post-select (sound, lower acceptance) vs
        # decode-and-budget (keep everything, carry the floor).
        print("magic value, post-selected (accepted shots):    ",
              np.round(exps[keep].mean(axis=0), 4), f"  @ {keep.mean():.0%} acceptance")
        # The budget above is an INFIDELITY (a probability ~0.19); the drop here in the
        # magic value is a different quantity — don't compare them directly.
        print("magic value, NO post-select (all shots):        ",
              np.round(exps.mean(axis=0), 4), "  @ 100% acceptance (carries the budget floor above)")

    # Decode the survivors. NOTE: this protocol has a SATURATED reject region — all
    # detectors are flagged, so survivors carry an all-zero syndrome and the decode is
    # a no-op here (post-selection already did the work). The DEM is a real decoder
    # model, not a vestige; to see it do nontrivial work, run it on a circuit with a
    # NON-empty keep region (e.g. the Clifford twin cultivation_d3_rate.stim, which has
    # an empty reject region — every shot kept, real corrections).
    try:
        import pymatching
        matcher = pymatching.Matching.from_detector_error_model(res.dem)
        if keep.any():
            corr = matcher.decode_batch(dets[keep])
            nontrivial = int(corr.any(axis=1).sum()) if corr.size else 0
            print(f"\ndecoded {corr.shape[0]} accepted shots "
                  f"({matcher.num_fault_ids} logical observable(s), {matcher.num_detectors} detectors); "
                  f"{nontrivial} got a nontrivial correction "
                  f"({'no-op — saturated reject' if nontrivial == 0 else 'real decode work'}).")
    except ImportError:
        print("\n(install `pymatching` to decode res.dem)")

    print("\nAlternatives you control: a finer per-signature post-select (accepts more shots, but VERIFY\n"
          "it stays sound against the noiseless magic value — on this saturated-reject circuit the blunt\n"
          "keep_mask is already minimal), or skip post-selection and carry the budget above as a floor.")


if __name__ == "__main__":
    main()
