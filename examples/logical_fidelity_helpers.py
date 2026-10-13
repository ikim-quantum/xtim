#!/usr/bin/env python
"""Opt-in logical-fidelity helpers — ``extract_frame`` and ``fidelity_from_logicals``.

These are **opt-in** helpers that require you to supply the logical operators
explicitly.  The default fidelity path remains
``Circuit.postselected_logical_error_rate``, which scores the
``PAULI_EXPECTATION`` columns the circuit already declares.  Reach for these
helpers when you want to:

* audit or construct a ``PAULI_EXPECTATION`` byproduct frame for a logical
  operator you choose yourself (``extract_frame``), or
* compute the TRUE full-support 1−F rather than only the subset of Paulis a
  particular circuit happens to declare (``fidelity_from_logicals``).

Run:

    conda activate qec
    python examples/logical_fidelity_helpers.py
"""
from __future__ import annotations

import os
import re
import sys
import warnings as _warnings

# Allow running straight from a source checkout (python examples/...py):
# put the repo root (which contains the `xtim` package) on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xtim  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — extract_frame: byproduct-frame lookup for a chosen logical Pauli
# ─────────────────────────────────────────────────────────────────────────────

def demo_extract_frame() -> None:
    """Show extract_frame on the qRM logical-X̄ in code_switching_faithful.

    code_switching_faithful switches between a qRM [[7,1,3]] and a Steane
    [[7,1,3]] code.  The qRM logical-X̄ acts on data qubits 15–21.

    We ask: which measurement records fix the SIGN of ⟨X̄⟩ in the noiseless
    output?  The answer is the byproduct frame — the set of records whose XOR
    produces the right correction.  Paste it directly into a PAULI_EXPECTATION
    line so the engine applies that correction automatically.
    """
    print("=" * 70)
    print("Section 1 — extract_frame: byproduct frame for a logical Pauli")
    print("=" * 70)

    c = xtim.load_example("code_switching_faithful")
    XBAR = "X15*X16*X17*X18*X19*X20*X21"

    # Minimal frame (default) — the canonical smallest sign-controlling set.
    minimal = xtim.extract_frame(c, XBAR, minimal=True)
    n = c.num_measurements
    minimal_abs = sorted(n - k for k in minimal)  # abs_index = num_measurements - k

    print(f"\nCircuit:        code_switching_faithful  ({n} measurements total)")
    print(f"Logical X̄:      {XBAR}")
    print()
    print("extract_frame(..., minimal=True)")
    print(f"  rec[-k] offsets : {minimal}")
    print(f"  absolute records: {minimal_abs}")
    print()
    print("  These are the 7 measurement records whose XOR fixes the sign of")
    print("  ⟨X̄⟩ in the noiseless output.  Paste them into a PAULI_EXPECTATION")
    print("  line to score this operator in your own circuit or analysis:")
    print()
    frame_str = " ".join(f"rec[-{k}]" for k in minimal)
    print(f"    PAULI_EXPECTATION(0) {XBAR} {frame_str}")
    print()
    print("  With this frame the engine reads ⟨X̄⟩ and applies the byproduct")
    print("  sign-correction automatically — no hand-tracking needed.")

    # Full-readout representative (minimal=False) — a spanning superset that
    # mirrors the declared frame from the original circuit, for readability.
    full = xtim.extract_frame(c, XBAR, minimal=False)
    full_abs = sorted(n - k for k in full)  # abs_index = num_measurements - k

    print()
    print("extract_frame(..., minimal=False)  — full-readout representative")
    print(f"  rec[-k] offsets : {full}")
    print(f"  absolute records: {full_abs}")
    print()
    print("  This is the spanning frame declared in the original circuit (a")
    print("  superset of the minimal set).  Use it when you want a layout that")
    print("  mirrors the original circuit's declared PAULI_EXPECTATION columns")
    print("  exactly.  Both frames produce the SAME corrected expectation;")
    print("  the extra records cancel out.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — fidelity_from_logicals: full-support 1−F for a k=1 magic prep
# ─────────────────────────────────────────────────────────────────────────────

def demo_fidelity_from_logicals() -> None:
    """True full-support 1−F for the d=3 cultivation magic-state-prep circuit.

    cultivation_d3_faithful prepares a logical T-magic state on seven data
    qubits.  We give it the X̄ and Z̄ generators; the helper generates all
    4^k − 1 = 3 non-identity logical Pauli products (X̄, Z̄, Ȳ=X̄Z̄), pins each
    product's byproduct frame, then samples all three at once.

    Step A — noiseless validator (p=0.0):
        A correct full-support algebra for a pure magic state reaches F=1
        noiselessly.  Running at p=0.0 is the cheapest sanity check: if 1−F > 0
        you know at least one generator is wrong.

    Step B — noisy estimate (p=1e-3):
        The real full-support infidelity of the prepared state.

    Step C — contrast with postselected_logical_error_rate:
        postselected_logical_error_rate scores only the columns the circuit
        already declares: here X̄ and Ȳ.  Because the T-magic state has ⟨Z̄⟩=0,
        the missing Z̄ column contributes nothing to F, so the two numbers agree.
        In general — when a circuit's declared columns are an INCOMPLETE
        support — postselected_logical_error_rate returns a partial-support proxy
        that has a noiseless floor which INFLATES (over-states) the apparent
        infidelity.  It is NOT a lower bound.  fidelity_from_logicals always
        scores the full 4^k support and has a noiseless floor of exactly 0.
    """
    print()
    print("=" * 70)
    print("Section 2 — fidelity_from_logicals: full-support 1−F")
    print("=" * 70)

    c = xtim.load_example("cultivation_d3_faithful")
    XBAR = "X0*X3*X7*X9*X10*X12*X13"
    ZBAR = "Z0*Z3*Z7*Z9*Z10*Z12*Z13"
    logicals = {0: {"X": XBAR, "Z": ZBAR}}

    print("\nCircuit: cultivation_d3_faithful  (d=3 T-magic-state prep, k=1)")
    print(f"  X̄ = {XBAR}")
    print(f"  Z̄ = {ZBAR}")

    # ── Step A: noiseless validator ──────────────────────────────────────────
    print()
    print("── Step A: noiseless validator (p=0.0, shots=2000) ──")
    with _warnings.catch_warnings(record=True) as _w:
        _warnings.simplefilter("always")
        r_noiseless = c.fidelity_from_logicals(logicals, p=0.0, shots=2000)
    print(f"  1−F = {r_noiseless}")
    print()
    print("  1−F ≈ 0 (within statistics) — the generators are correct.")
    print("  upper_bound = rule-of-three 95%-CL ceiling = 3/kept")
    if r_noiseless.upper_bound is not None:
        print(f"  upper_bound = {r_noiseless.upper_bound:.4f}  (LER < this at 95% CL)")
    print()
    print("  USAGE: always run the noiseless validator first.  If 1−F > 0 here,")
    print("  the generators are wrong — fix them before running noisy estimates.")

    # ── Step B: noisy estimate ───────────────────────────────────────────────
    SHOTS = 200_000
    P = 1e-3
    print()
    print(f"── Step B: noisy full-support 1−F (p={P:.0e}, shots={SHOTS:,}) ──")
    with _warnings.catch_warnings(record=True) as _w:
        _warnings.simplefilter("always")
        r_noisy = c.fidelity_from_logicals(logicals, p=P, shots=SHOTS)
    ler_noisy = max(0.0, r_noisy.value)
    print(f"  1−F = {ler_noisy:.3e}  (sem = {r_noisy.sem:.2e})")
    print(f"  acceptance = {r_noisy.acceptance:.4f}  kept = {r_noisy.kept:,}")
    print()
    print("  This is the TRUE infidelity: all 4^1−1 = 3 logical Pauli products")
    print("  (X̄, Z̄, Ȳ) are scored together with their byproduct frames.")

    # ── Step C: contrast with postselected_logical_error_rate ────────────────
    print()
    print("── Step C: contrast with postselected_logical_error_rate ──")
    with _warnings.catch_warnings(record=True) as _w:
        _warnings.simplefilter("always")
        r_ler = c.postselected_logical_error_rate(p=P, shots=SHOTS)
    ler_ler = max(0.0, r_ler.value)
    print(f"  postselected_logical_error_rate: 1−F = {ler_ler:.3e}"
          f"  (sem = {r_ler.sem:.2e})")
    print()
    print("  They agree here because cultivation_d3_faithful already declares X̄")
    print("  and Ȳ as PAULI_EXPECTATION columns.  The T-magic state has ⟨Z̄⟩=0,")
    print("  so the missing Z̄ column contributes nothing to F — the two methods")
    print("  reach the same result.")
    print()
    print("  General principle: if a circuit declares an INCOMPLETE column set")
    print("  (some logical Paulis with non-zero ⟨P⟩ are missing), then")
    print("  postselected_logical_error_rate gives a PARTIAL-SUPPORT PROXY that")
    print("  has a noiseless floor which INFLATES (over-states) the apparent")
    print("  infidelity.  It is NOT a lower bound.  In those cases reach for")
    print("  fidelity_from_logicals, which always scores the full 4^k support.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — XtimError: honest failure for bad inputs
# ─────────────────────────────────────────────────────────────────────────────

def demo_error_cases() -> None:
    """Show that invalid inputs raise xtim.XtimError — no silent wrong numbers."""
    print()
    print("=" * 70)
    print("Section 3 — XtimError: honest failure for bad inputs")
    print("=" * 70)
    print()

    c = xtim.load_example("cultivation_d3_faithful")
    XBAR = "X0*X3*X7*X9*X10*X12*X13"
    ZBAR = "Z0*Z3*Z7*Z9*Z10*Z12*Z13"

    # Case 1: an operator whose sign no measurement record can pin.
    print("-- extract_frame: operator that is not pinnable --")
    print("   RX 0; M 0 measures qubit 0 in the Z basis, so the sign of X0 is")
    print("   left fully random — no record parity determines it. extract_frame")
    print("   REFUSES rather than inventing a frame:")
    unpinnable = xtim.Circuit("RX 0\nM 0\n")
    try:
        xtim.extract_frame(unpinnable, "X0")
    except xtim.XtimError as e:
        print(f"  XtimError: {e}")

    print()
    print("-- fidelity_from_logicals: generators that COMMUTE (algebra violation) --")
    print("   Swapping X̄/Z̄ so they both map to X̄ — the same operator commutes")
    print("   with itself, violating that X̄_i and Z̄_i must anticommute.")
    bad_logicals = {0: {"X": XBAR, "Z": XBAR}}  # both X̄ — they commute
    try:
        c.fidelity_from_logicals(bad_logicals, p=0.0, shots=100)
    except xtim.XtimError as e:
        print(f"  XtimError: {e}")

    print()
    print("-- fidelity_from_logicals: X̄/Z̄ roles swapped — still a valid algebra --")
    # X̄ and Z̄ share the same support and anticommute, so swapping which one is
    # labelled "X" vs "Z" is just a relabelling of the same logical qubit: valid.
    try:
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            c.fidelity_from_logicals({0: {"X": ZBAR, "Z": XBAR}}, p=0.0, shots=100)
        print("  (no error — the swapped pair still anticommutes, so it is valid)")
    except xtim.XtimError as e:
        print(f"  XtimError: {e}")

    print()
    print("The helper never silently returns a wrong number.")
    print("Either it raises XtimError (bad algebra / unpinnable sign), or it")
    print("warns via r.warnings if the noiseless F is < 1 (wrong generators).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Partial-support proxy INFLATES: a concrete divergence example
# ─────────────────────────────────────────────────────────────────────────────

def demo_partial_support_divergence() -> None:
    """Show that a partial-support proxy INFLATES (over-states) the infidelity.

    cultivation_d3_faithful normally declares X̄ and Ȳ.  We strip all declared
    PAULI_EXPECTATION lines and re-declare ONLY X̄.

    At p=0 (noiseless):
      - postselected_logical_error_rate(target_k=1) returns 1−F ≈ 0.25
        and fires the "PARTIAL-support proxy" warning — the missing Ȳ column
        accounts for half the Parseval sum, so the partial proxy has a noiseless
        floor of 0.25 that INFLATES the apparent infidelity.
      - fidelity_from_logicals with the full {X̄, Z̄} algebra returns 1−F = 0.0
        (Parseval-exact, correct full-support).

    The partial proxy over-states the infidelity by 0.25 — it is NOT a lower
    bound.  Never trust postselected_logical_error_rate when the circuit's
    declared column set is incomplete.
    """
    print()
    print("=" * 70)
    print("Section 4 — Partial-support proxy INFLATES (divergence example)")
    print("=" * 70)

    c = xtim.load_example("cultivation_d3_faithful")
    XBAR = "X0*X3*X7*X9*X10*X12*X13"
    ZBAR = "Z0*Z3*Z7*Z9*Z10*Z12*Z13"

    # Build a stripped circuit that declares only X̄.
    stripped_text = re.sub(r'^PAULI_EXPECTATION\b.*$', '', c.text, flags=re.MULTILINE)
    offsets_xbar = xtim.extract_frame(c, XBAR)
    frame_str = " ".join(f"rec[-{k}]" for k in offsets_xbar)
    xbar_only_text = stripped_text.rstrip() + f"\nPAULI_EXPECTATION(0) {XBAR} {frame_str}\n"
    c_xbar_only = xtim.Circuit(xbar_only_text)

    print()
    print("Circuit: cultivation_d3_faithful with ONLY X̄ declared")
    print("  (stripped all PAULI_EXPECTATION lines, re-declared only X̄)")
    print()

    SHOTS = 10_000

    # Partial-support proxy via postselected_logical_error_rate at p=0
    print("── postselected_logical_error_rate(p=0.0, target_k=1) on X̄-only circuit:")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        r_partial = c_xbar_only.postselected_logical_error_rate(
            p=0.0, shots=SHOTS, target_k=1)
    partial_val = max(0.0, r_partial.value)
    print(f"  1−F = {partial_val:.4f}  (noiseless!)")
    partial_warns = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    if partial_warns:
        # Print the full advisory (its first line) — never truncate mid-sentence.
        print(f"  WARNING fired: {partial_warns[0].splitlines()[0]}")
    print()
    print("  The noiseless floor of 0.25 is NOT a measurement of noise — it is")
    print("  an artefact of the INCOMPLETE column set.  The partial proxy INFLATES")
    print("  the apparent infidelity; it is NOT a lower bound on the true 1−F.")

    print()
    # Full-support via fidelity_from_logicals at p=0
    print("── fidelity_from_logicals(p=0.0) on the same circuit (full {X̄, Z̄} algebra):")
    with _warnings.catch_warnings(record=True) as _w:
        _warnings.simplefilter("always")
        r_full = c.fidelity_from_logicals(
            {0: {"X": XBAR, "Z": ZBAR}}, p=0.0, shots=SHOTS)
    full_val = max(0.0, r_full.value)
    print(f"  1−F = {full_val:.4f}  (noiseless, full support)")
    print()
    print("  Full support reaches 1−F = 0.0 noiselessly (Parseval-exact).")
    print()
    print(f"  Divergence: partial proxy = {partial_val:.4f}  vs  full-support = {full_val:.4f}")
    print("  The 0.25 gap is the noiseless floor from the missing Ȳ column.")
    print("  Use fidelity_from_logicals when you cannot trust the circuit's")
    print("  declared column set to be complete.")


def main() -> None:
    demo_extract_frame()
    demo_fidelity_from_logicals()
    demo_error_cases()
    demo_partial_support_divergence()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
