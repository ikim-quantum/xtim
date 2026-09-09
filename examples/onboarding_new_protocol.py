#!/usr/bin/env python3
"""Onboarding: scoring a NEW magic-state-prep protocol with xtim.

You did not write `code_switching` or `cultivation_d3`. You wrote your OWN tiny
magic-state preparation and dropped it into a `.stim` file. None of the
hand-derived lore that ships with the demo protocols — which detectors to
post-select, the byproduct-frame sign rule, the |beta| magnitudes — exists for
*your* circuit yet. This script shows how `Circuit.diagnose()` hands you all of
it, so you can score your protocol without reverse-engineering anything.

The running circuit is `benchmarks/miniature_oracle.stim` (in the export it sits
next to this file as `miniature_oracle.stim`): a one-bit teleport of T|+> into a
3-data-qubit repetition code (5 qubits total, with ancillas), with two declared
expectations `PAULI_EXPECTATION(0) X0*X1*X2 rec[-3]` and
`PAULI_EXPECTATION(1) Y0*X1*X2 rec[-3]` — each declares its teleport byproduct frame
(the trailing `rec[-3]`), which the engine folds into the per-shot sign for you.
Pretend you just wrote it. It happens to be small (chi=2) and end-syndrome
post-selected — the simplest interesting case.

Everything below uses ONLY the public xtim surface: the circuit text and
`diagnose()`. No sidecar, no `benchmarks/targets/*.json`, no internal lore.

Run it:  `python examples/onboarding_new_protocol.py`
The full decoder-in-the-loop workflow (when you DO need pymatching + the DEM) is
the tour: ../docs/xtim_tour.md . The dialect your circuit is written in is
../docs/xtim_dialect.md .
"""
import os

import numpy as np

import xtim

# Resolve the circuit relative to this file so the script runs from anywhere. In
# the shipped export the circuit lives beside this script as
# `miniature_oracle.stim`; in the monorepo it is under `benchmarks/`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "miniature_oracle.stim"),
    os.path.join(_HERE, os.pardir, os.pardir, os.pardir,
                 "benchmarks", "miniature_oracle.stim"),
]
CIRCUIT_PATH = next(p for p in _CANDIDATES if os.path.exists(p))


def main() -> float:
    c = xtim.Circuit.from_file(CIRCUIT_PATH)

    # ── Step 1: ask the tool what you built ────────────────────────────────────
    # One noiseless run; diagnose() REPORTS facts, it never interprets them. Warm
    # the reference cache with a tiny sample first so the report can show chi (a
    # fresh circuit prints "chi: ? (deduced)" — same facts, just not cached yet).
    c.compile_detector_sampler(seed=0).sample(64, return_expectations=True)
    report = c.diagnose(shots=2000, seed=0)
    print(report)
    print()

    # Read — do NOT hand-derive — the three things you would otherwise reverse
    # engineer, straight off the report object:
    #
    #   * which detectors are deterministic  -> what you post-select on
    #   * |value| per expectation column      -> |beta_i| (the Bloch magnitudes)
    #   * the byproduct-frame hint            -> the per-shot sign rule
    keep_detectors = report.deterministic_detectors          # post-select on these
    beta_magnitude = np.array([e.abs_value for e in report.expectations])
    frame_records = [e.frame_hint for e in report.expectations]

    # Each column DECLARES its *byproduct frame* on the PAULI_EXPECTATION line (the
    # trailing `rec[-k]` records): the teleport's Z^m correction whose parity would
    # otherwise flip the sign ~50% of the time. Because it is declared, the engine
    # FOLDS it into the per-shot sign for you — so `exps` comes out already sign-
    # correct, and `frame_hint` just tells you WHICH records were folded (provenance,
    # not a step you re-apply). (When a protocol's frame is not a single parity, or the
    # DEM is expressible end to end, this is where pymatching + `c.detector_error_model()`
    # plug in — see ../docs/xtim_tour.md.)
    #
    # diagnose() also reports `dem_expressible=False` for this circuit: its
    # expectations cannot be DEM L-columns (a depolarize after the T rotates the
    # terminal read beyond a clean sign). So we score it by post-selection on the
    # declared-frame-corrected columns, never by feeding these columns to a DEM decoder.

    # ── Step 2: your target — and cross-check it against the circuit ────────────
    # YOU supply the target (what you intended to prepare). diagnose ALSO reports the
    # byproduct-frame-CORRECTED signed beta the *circuit* actually prepares — the
    # report's `target (signed beta)` line, or `e.signed_value` per column. They
    # should agree; if they don't, the circuit isn't making the state you think.
    # We aimed at T|+>, whose <X0X1X2> and <Y0X1X2> are both +1/sqrt2 -> signs +1.
    target_sign = np.array([+1.0, +1.0])
    prepared_signed = np.array([e.signed_value for e in report.expectations])
    assert np.allclose(np.sign(prepared_signed), target_sign), (
        "the circuit's prepared sign disagrees with your target — check the protocol")
    beta = target_sign * beta_magnitude

    # ── Step 3: sample, post-select, score ─────────────────────────────────────
    # The declared frame is folded by the engine, so `exps` is ALREADY sign-correct —
    # score it directly (no per-shot record-parity step to re-apply).
    N = 20_000
    cols = c.expectation_columns
    dets, exps, meas = c.compile_detector_sampler(seed=7).sample(
        N, return_expectations=True, return_measurements=True)

    keep = ~dets[:, keep_detectors].any(axis=1)              # diagnose's recipe
    value = exps[keep].mean(axis=0)                          # a-bar per channel (frame already folded)
    F = (1 + beta @ value) / 2                               # your target's F

    # What the declared frame bought you: if you (wrongly) RE-APPLIED the folded frame
    # parity yourself, you would double-correct and F would collapse to ~1/2. The engine
    # already did this for you — `frame_hint` is provenance, not a step you repeat.
    reflips = np.zeros((N, len(cols)), dtype=bool)
    for j, records in enumerate(frame_records):
        if records:
            reflips[:, j] = (meas[:, records].sum(axis=1) % 2).astype(bool)
    double = np.where(reflips, -exps, exps)
    collapsed_F = (1 + beta @ double[keep].mean(axis=0)) / 2

    print(f"kept {keep.sum()}/{N} shots")
    print(f"a-bar (declared-frame-corrected by the engine) = {value}")
    print(f"F = {F:.4f}   (if you WRONGLY re-apply the folded frame: {collapsed_F:.4f})")

    # Sanity: every number came from diagnose() + the target sign. The declared frame is
    # doing real work — the engine folded it so F is right out of the box; re-applying it
    # would collapse F to ~1/2, which is exactly the byproduct sign diagnose warned about.
    assert keep.sum() > 0.9 * N
    assert 0.9 < F < 1.0
    assert collapsed_F < 0.6
    return F


if __name__ == "__main__":
    main()
