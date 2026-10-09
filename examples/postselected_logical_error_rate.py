#!/usr/bin/env python
"""One-call post-selected logical-error-rate demo for xtim.

Runs ``Circuit.postselected_logical_error_rate`` — the single call that auto-applies
the deterministic-detector post-selection and auto-detects fidelity vs observable mode
— across a few bundled circuits and prints a physical-error-rate sweep.

    conda activate qec
    python examples/logical_error_rate.py

The point of the API: you never hand-build the post-selection mask or choose the
mode. One call gives the post-selected logical error rate (maximal heralding), not the
unsuppressed O(p) number you get if you forget the post-selection.
"""
from __future__ import annotations

import os
import sys
import warnings as _warnings

# Allow running straight from a source checkout (python examples/logical_error_rate.py):
# put the repo root (which contains the `xtim` package) on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xtim  # noqa: E402

# A few bundled circuits spanning both scoring modes. (name, kwargs)
CIRCUITS = [
    ("code_switching_faithful", {}),                 # fidelity (qRM <-> Steane switch)
    ("cultivation_d3_faithful", {}),                 # fidelity (d=3 cultivation)
    ("distillation_15_1_3", {}),                     # fidelity (15-to-1 distillation)
    ("cube_ccz", {}),                                # fidelity, INCOMPLETE support (warns)
    ("cultivation_d5", {}),                          # observable (d=5 cultivation)
]

P_SWEEP = [3e-3, 1e-3, 3e-4, 1e-4]
SHOTS = 200_000


def main() -> None:
    for name, kw in CIRCUITS:
        c = xtim.load_example(name)

        # Collect the entire p-sweep up front, suppressing the per-call warnings.warn
        # stderr emissions so they cannot interleave with the stdout table.
        # Advisories are preserved in r.warnings and printed in a notes section below.
        results: list[tuple[float, "xtim.PostselectedLER"]] = []
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            head = c.postselected_logical_error_rate(p=P_SWEEP[0], shots=1000, **kw)
            for p in P_SWEEP:
                r = c.postselected_logical_error_rate(p=p, shots=SHOTS, **kw)
                results.append((p, r))

        # --- print the formatted table (stdout, uninterrupted) ---
        print(f"\n=== {name}  (mode: {head.mode}"
              + (f", target_k={head.target_k}" if head.target_k else "") + ") ===")
        print(f"    {'p':>8} | {'LER':>12} | {'sem':>10} | {'upper':>10} | "
              f"{'accept':>8} | {'kept':>8}")
        print("    " + "-" * 70)
        for p, r in results:
            ler = max(0.0, r.value)  # guard the tiny-negative float near F=1
            ub = "n/a" if r.upper_bound is None else f"<{r.upper_bound:.1e}"
            print(f"    {p:>8.0e} | {ler:>12.3e} | {r.sem:>10.2e} | {ub:>10} | "
                  f"{r.acceptance:>8.4f} | {r.kept:>8d}")
        sys.stdout.flush()

        # --- notes section: unique advisories collected across all p values ---
        # Deduplicating by the first 120 chars avoids repeating the same advisory
        # (e.g. "no noise" or "partial support") once per p row.
        seen: set[str] = set()
        notes: list[str] = []
        for _p, r in results:
            for w in r.warnings:
                key = w[:120]
                if key not in seen:
                    seen.add(key)
                    notes.append(w)
        if notes:
            print("    Notes:")
            for w in notes:
                # Print the full advisory's first line — never truncate mid-word
                # (the advisory carries the actionable rule-of-three bound).
                print(f"      ! {w.splitlines()[0].strip()}")


if __name__ == "__main__":
    main()
