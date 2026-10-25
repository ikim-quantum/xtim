#pragma once
#include "qeccore/circuit_ir.hpp"   // GateKind (the IR owns the gate alphabet)
#include "qeccore/clifford_op.hpp"
namespace qeccore {
// Conjugate D in place to U·D·U†.  Returns true if the result stays a DiagPauliClifford;
// returns false (D left unspecified) if U takes it out of the diagonal+Pauli class.
// a,b,c are the gate's qubits (b,c = -1 when unused).
bool conjugate_by_gate(DiagPauliClifford& D, GateKind U, int a, int b = -1, int c = -1);
}
