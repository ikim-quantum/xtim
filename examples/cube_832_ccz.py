#!/usr/bin/env python3
"""[[8,3,2]] cube code: a post-selected transversal-CCZ magic state, scored to a
TRUE 3-qubit fidelity (xtim >= 0.5.9).

This is the worked, end-to-end version of the bundled `cube_ccz.stim` demo. The
[[8,3,2]] color code puts 3 logical qubits on the 8 vertices of a cube; a
transversal pattern of `T`/`T_dag` realizes a **logical CCZ**, so the output is the
genuine 3-logical-qubit magic state `CCZ|+++>_L` (stabilizer rank chi = 8).

Protocol (exactly the structure a fault-tolerant gadget would use):
  1. Prepare |+>^8, then MEASURE the four Z-stabilizers and apply classically-
     controlled Pauli (X) FEEDBACK to fold the random syndrome back to the
     codespace -> deterministically the logical |+++>_L.
  2. Apply the logical CCZ = `T` on {0,4,5,6}, `T_dag` on {1,2,3,7}. **Only the
     physical T gates are noisy** (a DEPOLARIZE1 on the eight T'd qubits); prep,
     feedback, and the syndrome round are exact.
  3. A single NOISELESS round of all five stabilizer measurements (the X-stabilizer
     catches Z/Y faults, the four Z-stabilizers catch X/Y faults). POST-SELECT on
     all five reading +1 — i.e. discard any shot whose T faults were detected.

Why a complete Pauli support matters: `cube_ccz.stim` declares only the three
logical-X marginals, which is an INCOMPLETE support — its "fidelity" maxes out at
0.219 even noiselessly (and xtim now warns about exactly that). Here we declare the
full logical Pauli group (63 operators, of which 28 are the nonzero CCZ support),
so `target_k=3` gives the real state fidelity: F = 1 with no noise.

The lesson the curve teaches: the [[8,3,2]] code has distance 2 — it DETECTS any
single fault but corrects none. Post-selecting the syndrome therefore heralds away
every single-fault error, suppressing the infidelity from O(p) (keep-all) to O(p^2)
(post-selected), at the cost of a falling acceptance rate.

Run it:  `python examples/cube_832_ccz.py`
With matplotlib installed it also writes `cube_832_ccz.png` beside your cwd.
"""
import itertools

import numpy as np

import xtim

# --- the [[8,3,2]] cube color code (physical qubits = cube vertices 0..7) -------
_Z_STABS = ["Z0*Z1*Z6*Z7", "Z0*Z2*Z3*Z6", "Z2*Z4*Z6*Z7", "Z0*Z2*Z5*Z7"]
_X_STAB = "X0*X1*X2*X3*X4*X5*X6*X7"
# Logical operators as physical supports. Xbar_i are the code's logical X; Zbar_i
# are weight-2 logical Z (solved to anticommute with Xbar_i and nothing else).
_XBAR = [[0, 1, 2, 4], [0, 1, 3, 5], [0, 2, 3, 6]]
_ZBAR = [[0, 3], [0, 2], [0, 1]]
_T_QUBITS, _TDAG_QUBITS = [0, 4, 5, 6], [1, 2, 3, 7]   # transversal CCZ pattern


def _logical_pauli_group():
    """The 63 non-identity logical Paulis as physical Pauli strings.

    Each is the product of the Xbar/Zbar generators it selects; a qubit carrying
    both an X part and a Z part becomes Y (global phase is irrelevant to a
    PAULI_EXPECTATION column). Declaring the FULL group is the foolproof complete
    support for any 3-qubit target; 28 of the 63 are nonzero here (the CCZ support)."""
    out = []
    for sel in itertools.product(range(4), repeat=3):   # 0=I,1=X,2=Y,3=Z per logical qubit
        if sel == (0, 0, 0):
            continue
        xp, zp = [0] * 8, [0] * 8
        for q, pa in enumerate(sel):
            if pa in (1, 2):                            # X or Y -> Xbar_q
                for k in _XBAR[q]:
                    xp[k] ^= 1
            if pa in (3, 2):                            # Z or Y -> Zbar_q
                for k in _ZBAR[q]:
                    zp[k] ^= 1
        out.append("*".join(f"{'IXZY'[xp[k] + 2 * zp[k]]}{k}"
                            for k in range(8) if xp[k] or zp[k]))
    return out


_LOGICAL_PAULIS = _logical_pauli_group()


def build_circuit(p: float) -> "xtim.Circuit":
    """The full protocol at T-gate noise strength p (only the T gates are noisy)."""
    lines = ["R 0 1 2 3 4 5 6 7", "H 0 1 2 3 4 5 6 7"]
    # 1. prep: measure the 4 Z-stabilizers, feedback-correct to |+++>_L.
    lines += [f"MPP {s}" for s in _Z_STABS]
    for rec, qubits in ((-4, [0, 1, 2, 3, 6]), (-3, [0, 1, 2, 3, 4]),
                        (-2, [4]), (-1, [5])):
        lines += [f"CX rec[{rec}] {q}" for q in qubits]
    # 2. logical CCZ = transversal T/T_dag; DEPOLARIZE1 ONLY on the T'd qubits.
    tq = " ".join(map(str, _T_QUBITS)); dq = " ".join(map(str, _TDAG_QUBITS))
    lines += [f"T {tq}", f"DEPOLARIZE1({p}) {tq}",
              f"T_DAG {dq}", f"DEPOLARIZE1({p}) {dq}"]
    # 3. noiseless syndrome round (all 5 stabilizers) -> 5 deterministic detectors.
    lines += [f"MPP {s}" for s in [_X_STAB, *_Z_STABS]]
    lines += [f"DETECTOR rec[{-k}]" for k in range(5, 0, -1)]
    # the true-state fidelity readout: the complete logical Pauli support.
    lines += [f"PAULI_EXPECTATION({i}) {s}" for i, s in enumerate(_LOGICAL_PAULIS)]
    return xtim.Circuit("\n".join(lines) + "\n")


def _keep_syndrome_zero(dets, meas):
    """Post-select: keep shots where every end-syndrome detector stayed 0 (+1)."""
    return ~dets.any(axis=1)


def main(shots: int = 60_000, seed: int = 3) -> list[dict]:
    """Sweep the T-gate noise; return rows with post-selected fidelity."""
    ps = [0.0, 1e-3, 3e-3, 1e-2, 3e-2]
    post, allshots = [], []
    for p in ps:
        c = build_circuit(p)
        post.append(xtim.collect([xtim.Task(circuit=c, p=None, shots=shots, seed=seed,
                                            target_k=3, keep=_keep_syndrome_zero)])[0])
        allshots.append(xtim.collect([xtim.Task(circuit=c, shots=shots, seed=seed,
                                               target_k=3)])[0])

    print(f"{'p (on T)':>10} {'accept':>8} {'1-F post-sel':>14} {'1-F keep-all':>14}")
    for p, rp, ra in zip(ps, post, allshots):
        print(f"{p:>10.0e} {rp['acceptance_rate']:>8.3f} "
              f"{max(0.0, rp['infidelity']):>14.2e} {max(0.0, ra['infidelity']):>14.2e}")

    _maybe_plot(ps, post, allshots)

    # Self-checks (so the example cannot rot):
    assert abs(post[0]["fidelity"] - 1.0) < 1e-9, post[0]["fidelity"]   # noiseless F = 1
    big = -1                                                            # largest p
    # post-selection genuinely helps, and by a widening margin (O(p^2) vs O(p)):
    assert post[big]["infidelity"] < allshots[big]["infidelity"] / 10
    assert post[1]["acceptance_rate"] > post[big]["acceptance_rate"]   # acceptance falls with p
    print(f"\nOK: noiseless F=1 on the COMPLETE support; post-selection suppresses "
          f"1-F from {allshots[big]['infidelity']:.1e} to {post[big]['infidelity']:.1e} "
          f"at p={ps[big]:g}.")
    return post


def _maybe_plot(ps, post, allshots):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping the plot; `pip install xtim[tutorial]`)")
        return
    pp = ps[1:]   # drop p=0 (log axis)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pp, [max(1e-12, r["infidelity"]) for r in post[1:]], "o-", label="post-selected ~ p²")
    ax.plot(pp, [max(1e-12, r["infidelity"]) for r in allshots[1:]], "s--", label="keep all ~ p")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical T-gate error rate  p"); ax.set_ylabel("infidelity  1 - F")
    ax.set_title("[[8,3,2]] transversal CCZ — syndrome post-selection")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("cube_832_ccz.png", dpi=120)
    print("(wrote cube_832_ccz.png)")


if __name__ == "__main__":
    main()
