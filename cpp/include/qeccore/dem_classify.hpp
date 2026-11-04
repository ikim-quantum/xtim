#pragma once
#include <cstdint>
#include <vector>

#include "qeccore/clifford_op.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

// Observable-magnitude classifier for magic-state-prep DEM export (xtim routing M-step).
//
// Given a propagated diagonal+Pauli error D_c = gamma · X^v · Delta(a, B) and the declared
// logical Paulis {P_r}, decide whether D_c acts on the bare state's logical content as a pure
// SIGN action: every logical-observable magnitude |<P_r>| is preserved (only signs may flip).
// Such a fault is "Pauli-correctable" (a Pauli frame update suffices); a fault that changes a
// magnitude is NOT — it must be post-selected away (handled by a later task).
//
// The decision is exact: conjugate each P_r through D_c to a Pauli Q_r = D_c† P_r D_c (the same
// diagonal conjugation algebra used by dem_export's alt_signature / dem_export.hpp:60-76), then
// evaluate <Q_r> on the bare state and compare its magnitude (and sign) to the precomputed
// baseline beta0[r] = <P_r>.
namespace qeccore {

// Conjugate the declared Pauli P through the diagonal+Pauli Clifford Dc, returning the Pauli
// Q = Dc† P Dc as a Hermitian operator (Q.phase encodes the overall ±1 sign in the
// framed_expectation convention i^phase X^x Z^z, phase ≡ #Y mod 4 for a +1 operator).
//   * Q.x = P.x  (the diagonal Dc fixes P's X-support);
//   * S-power a_q on q∈supp(x): factor i^{-a_q} and leftover Z_q^{a_q mod 2} (odd → X↔Y rotation);
//   * CZ-arm B_{qj}, q∈supp(x): leftover Z_j; a pair both in supp(x) additionally contributes −1;
//   * X^v vs P's Z-content: a −1 per qubit where v_q ∧ z_q.
Pauli conjugate_through(const DiagPauliClifford& Dc, const Pauli& P);

struct LogicalAction {
    bool magnitude_preserved = true;   // every |<P_r>| unchanged by Dc
    std::vector<uint8_t> sign_flip;    // per column r: sign(<Dc^dag P_r Dc>) != sign(beta0[r])
};

// Classify Dc's action on the declared logical Paulis expP against baseline beta0 = <expP>.
LogicalAction classify_logical_action(const FramedSuperposition& bare,
                                      const std::vector<Pauli>& expP,
                                      const std::vector<double>& beta0,
                                      const DiagPauliClifford& Dc);

}  // namespace qeccore
