#pragma once
#include "qeccore/pauli.hpp"
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/circuit_ir.hpp"
#include <vector>
#include <utility>

namespace qeccore {

// Pauli-Product-Rotation residual:
//   R = pauli * Π_j exp(i(π/4) e_j A_j)
// where the axes A_j mutually commute (invariant of the accept path), each stored
// axis is canonical (Hermitian, phase ∈ {0,1}; a −1 sign is folded into the
// exponent), and e_j ∈ 1..7 (mod 8).
// The materialised unitary applies rots first (in order), then the Pauli.
// The residual's overall SCALAR phase is not tracked — mirroring
// DiagPauliClifford::gamma, which the engine documents as unobservable and drops:
// a residual is applied to the state as one operator, so a global scalar cancels
// in every Born probability and expectation.
struct PprResidual {
    int n = 0;
    Pauli pauli;                                   // Hermitian Pauli part P
    std::vector<std::pair<Pauli,int>> rots;        // (axis A_j Hermitian canonical, exponent e_j ∈ 1..7 (mod 8))
    bool rejected = false;
    int reject_gate_index = -1;                    // set by the propagation driver

    // Propagate R ← U R U† for a single gate.
    // Returns false (sets rejected) iff a π/8 gate would rotate an existing axis off the Pauli group.
    bool propagate(GateKind g, const std::vector<int>& targets);

    // Recompile the commuting π/4 rotation list Π_j exp(i(π/4) e_j A_j) into a CliffordTableau.
    // The Pauli part is NOT included — it stays tracked separately on `pauli`.
    CliffordTableau clifford_tableau() const;

    // Recompile the full operator E = pauli · (Π rots Clifford) into a CliffordTableau.
    // Implementation: start from clifford_tableau() to get U (the rotation part), then
    //   left_x(q) for each qubit q with pauli.xbit(q) set,
    //   left_z(q) for each qubit q with pauli.zbit(q) set.
    // Global scalar phase is dropped (unobservable in conjugation, mirroring the engine's
    // treatment of DiagPauliClifford::gamma).
    CliffordTableau to_error_tableau() const;
};

// Seed a bare Pauli residual: X_q or Z_q.
PprResidual ppr_seed_x(int n, int q);
PprResidual ppr_seed_z(int n, int q);

}  // namespace qeccore
