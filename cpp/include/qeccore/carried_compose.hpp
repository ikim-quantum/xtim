#pragma once
// carried_compose — J-b1 P9b (XQ9 (A), ruling on fixplan-p9b-xq9 / review C1-C10): the bare state of a consumer whose
// non-Clifford gates may act on CARRIED wires, built on the carried free register exactly.
//
// Input: S_in = (carried state ⊗ |0…0⟩_fresh) as a FramedSuperposition on the combined register (frame U_in, eps,
// free rows F, amplitudes α over F) and the consumer's deferred circuit remapped to the combined wires.
// With U = gp·C·T (build_pauli_rotation_form; every even entry settled into C, equal axes merged), the bare is
//     |B⟩ = gp · C · T · U_in · Σ_x α_x |x ⊕ eps⟩.
// In frame coordinates (Q_j = U_in† P_j U_in) the input is a computational-basis superposition on the F rows with
// every other row a constant; T is a phase polynomial over F ∪ (created variables) EXACTLY when no combination of the
// merged axes has its X-part on F only (the span criterion). Then a frame gauge (Zrow_f ← Zrow_f · Π_{c∈S_f} Zrow_c,
// Xrow_c ← Xrow_c · Xrow_f, eps_f ← eps_f ⊕ Σ eps_c — the state and α unchanged) puts every axis's X-part on
// constant rows, and the diagonalizing Clifford C' acts on the constant rows only. The F rows enter the parity
// support as variables (|+⟩ on each, never a constant), FastTODD reduces the joint phase polynomial (a functional
// identity over all of F₂^d, so the weights ride along), and the assembler branches over span(odd magic columns ∪ F
// coordinates) with the per-coset coefficient (magic phase) · α(x_F) · 2^{|F|/2}.
// The frame of the result depends on (U_in, F, eps, the circuit) only, never on α (plans never read α); an α
// with zero entries keeps every free row (no α-dependent χ-drop).
//
// When the span criterion fails: refused by name (LAW_LIMITATION_CARRIED_AXIS; the caller picks the sub-class:
// carried_free_x at compile on the ideal state, carried_branch_x at add_branch on a branch's E rows).
#include <string>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

struct CarriedComposeResult {
    bool rejected = false;
    bool span_refusal = false;          // the span criterion failed (the carried-axis law limitation)
    std::string reason;                 // the refusal text (rotation-form rejection or the span criterion)
    std::vector<Pauli> reject_terms;    // rotation-form rejection: the anticommuting terms (deferred space)
    int reject_gate_index = -1;
    int span_row = -1;                  // span refusal: a carried free row the axes' X-parts reach
    Pauli span_term;                    // span refusal: a merged axis (deferred space) with an X-part on span_row
    FramedSuperposition state{0};
    int gauge_moves = 0;                // rows multiplied by the gauge
    int axes = 0;                       // merged odd axes
};

// The exact composed bare (see above). `S_in` and `deferred` are on the same n wires.
CarriedComposeResult compose_carried_bare(const FramedSuperposition& S_in, const Circuit& deferred);

}  // namespace qeccore
