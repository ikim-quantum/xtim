#pragma once
// Output-state helpers. GF(2) / Z8 algebra only: no amplitudes.
#include <cstdint>
#include <string>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/residue_projection.hpp"   // Row, Projector
#include "qeccore/stim_parse.hpp"

namespace qeccore {

// Lemma 1: the residue-space kernel Y of the odd magic columns (d-bit rows over the support's free variables). Empty
// iff every Pauli stabilizer of the state commutes with every odd column's Z-string. Empty without the quadratic test
// when the columns are independent.
std::vector<Row> magic_shift_kernel(const std::vector<Row>& odd_cols, int d);
// Merge equal columns (coeffs add mod 16) and drop the constant ones (zero direction, coeff 0 or 8), then apply the
// shift step until magic_shift_kernel is empty. The phase function is unchanged up to a constant. Returns the step
// count (<= the odd rank).
int stabilize_magic_columns(std::vector<Row>& cols, std::vector<int>& coeffs, int d);

// How each user qubit (index < num_user_qubits) ends: the kind of the last non-noise stream op ON THAT QUBIT. Measure
// on q -> Measured. Reset on q, and the basis gates its user instruction (stream_step) applies to q after it (RX, RY,
// MRX, ...) -> Reset. Gate or feedback target -> Gate. MPP-style parity readouts measure a desugar ancilla, so their
// data qubits end as Gate (C1/Q2). Needs parse_stim_circuit(text, true).
enum class QubitEnd : uint8_t { Untouched, Gate, Measured, Reset };
std::vector<QubitEnd> qubit_ends(const ParsedStim& ps);
std::vector<int> output_candidates(const ParsedStim& ps);           // the qubits whose end is Gate, ascending

int hermitian_sign(const Pauli& p);   // p Hermitian: +1 / -1 relative to i^{#Y} X^x Z^z; throws otherwise

enum class OutputVerdict { Deterministic, CrossStabilizer, CoupledMagic, CoupledStabilizer, Undecided, OutOfClass };
struct OutputAnalysis {
    OutputVerdict verdict = OutputVerdict::OutOfClass;
    std::vector<Pauli> witnesses;          // CrossStabilizer / CoupledStabilizer: cross-cut stabilizers; CoupledMagic /
                                           // Undecided: the straddling axes (full width)
    std::string reject_reason;             // OutOfClass: the parity form's reject fields
    std::vector<Pauli> reject_terms;
    std::vector<Pauli> rotations, pauli_stabilizers, clifford_paulis;   // Deterministic && want_state; local width
    int shift_steps = 0;
};
// PLAN.md "The decision procedure" and "The output state", on the deferred circuit.
OutputAnalysis analyze_output(const Circuit& deferred, const std::vector<int>& out_wires, bool want_state);

}  // namespace qeccore
