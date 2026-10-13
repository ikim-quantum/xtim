#!/usr/bin/env python3
"""Fidelity figure-of-merit: a 1-F vs p curve in a dozen lines (xtim >= 0.5.9).

The thing a magic-state paper actually wants is a plot of **infidelity 1-F**
against the **physical error rate p**, with error bars. `xtim.collect` builds it
for you: set `target_k` (the number of logical qubits in the target magic state)
on each `Task`, sweep `p`, and every result row comes back carrying `fidelity`,
`infidelity` (= 1 - F), and `infidelity_sem` — already correlation-aware. No
manual Bloch arithmetic, no per-column error propagation.

The protocol here is deliberately the simplest possible carrier of the idea: a
logical `|T>` magic state with a single depolarizing channel whose strength is
swept. `Task(p=...)` rescales the baked-in `DEPOLARIZE1(1e-3)` to each p, so one
circuit text gives the whole curve, and `target_k=1` turns the two declared
`PAULI_EXPECTATION` columns (<X>, <Y> — the nonzero Bloch components of |T>) into
a true-state fidelity. On a real post-selected protocol you would additionally
pass a `keep` mask (see ../docs/xtim_tour.md); the fidelity part is identical.

Run it:  `python examples/fidelity_curve.py`
With matplotlib installed it also writes `fidelity_curve.png` beside your cwd.
"""
import numpy as np

import xtim

# |T> = (|0> + e^{i pi/4}|1>)/sqrt(2): <X> = <Y> = 1/sqrt(2), <Z> = 0. The two
# declared columns are exactly its nonzero Bloch components -> a COMPLETE support,
# so target_k=1 gives a true-state fidelity. The DEPOLARIZE1(1e-3) is the knob
# Task(p=...) rescales across the sweep (its p0 default is 1e-3).
_T_STATE_NOISY = (
    "R 0\n"
    "H 0\n"
    "T 0\n"
    "DEPOLARIZE1(0.001) 0\n"
    "PAULI_EXPECTATION(0) X0\n"
    "PAULI_EXPECTATION(1) Y0\n"
)

# The physical error rates to sweep (DEPOLARIZE1 strength). Two orders of magnitude
# so the trend sits well outside the error bars.
_PS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def main(shots: int = 50_000, seed: int = 11) -> list[dict]:
    """Sweep p, return the collect rows. Each row already has 1-F +/- sem."""
    c = xtim.Circuit(_T_STATE_NOISY)
    tasks = [xtim.Task(circuit=c, p=p, shots=shots, seed=seed, target_k=1,
                       metadata={"p": p})
             for p in _PS]
    rows = xtim.collect(tasks, num_workers=min(4, len(_PS)))

    # A plot-ready table: 1-F +/- sem vs p (acceptance is 1.0 here — no
    # post-selection; on a real protocol it would track p alongside).
    print(f"{'p':>10} {'1 - F':>12} {'+/- sem':>10}  {'fidelity':>10}")
    for row in rows:
        # infidelity is unbiased/unclamped; guard for the log-scale plot only.
        inf = max(0.0, row["infidelity"])
        print(f"{row['p']:>10.1e} {inf:>12.3e} {row['infidelity_sem']:>10.1e}"
              f"  {row['fidelity']:>10.6f}")

    _maybe_plot(rows)

    # Self-checks so the worked example cannot silently rot:
    infids = np.array([r["infidelity"] for r in rows])
    sems = np.array([r["infidelity_sem"] for r in rows])
    assert (sems > 0).all(), "every swept point should have a real error bar"
    # the lowest-p infidelity is tiny; the highest-p one is clearly larger,
    # by far more than the combined error bars (a genuine, monotone trend).
    assert infids[0] < 1e-3, infids[0]
    assert infids[-1] - infids[0] > 10 * (sems[0] + sems[-1]), (infids, sems)
    print(f"\nOK: 1-F rises {infids[0]:.2e} -> {infids[-1]:.2e} across "
          f"p in [{_PS[0]:g}, {_PS[-1]:g}].")
    return rows


def _maybe_plot(rows: list[dict]) -> None:
    """Save a log-log 1-F vs p plot if matplotlib is available; else skip."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: write a file, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping the plot; "
              "`pip install xtim[tutorial]` to enable it)")
        return
    p = [r["p"] for r in rows]
    inf = [max(1e-12, r["infidelity"]) for r in rows]   # clamp for log axis only
    sem = [r["infidelity_sem"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(p, inf, yerr=sem, marker="o", capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel("infidelity  1 - F")
    ax.set_title("xtim fidelity figure-of-merit")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = "fidelity_curve.png"
    fig.savefig(out, dpi=120)
    print(f"(wrote {out})")


if __name__ == "__main__":
    main()
