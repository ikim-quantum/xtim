#pragma once
#include <string>
#include <vector>
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

/// Result of the port-separability rank test.
///
/// THE CONDITION: ok == (rank_B == n_B)
///   rank(S ∩ P_B) == |B|   <=>   |φ⟩ = |φ_A⟩ ⊗ |ψ_B⟩  with  |ψ_B⟩  a stabilizer state.
///
/// Signs are ignored — rank depends only on the symplectic part of the generators.
struct PortRanks {
    int rank_S = 0;  ///< rank(S)
    int rank_A = 0;  ///< rank(S ∩ P_A)  — generators supported only on port wires
    int rank_B = 0;  ///< rank(S ∩ P_B)  — generators supported only on non-port wires
    int n_B    = 0;  ///< |B| = n - |port_wires|

    bool ok                = false;  ///< rank_B == n_B  (THE CONDITION)
    bool clifford_factorizes = false; ///< rank_A + rank_B == rank_S  (diagnostic)

    /// Witness row indices from row_reduce(M) that have support in BOTH A-columns and
    /// B-columns.  Empty iff clifford_factorizes is true.
    std::vector<int> straddling;
};

/// Compute the port-separability verdict from a set of stabilizer generators.
///
/// @param gens        Stabilizer generators (vector<Pauli>); signs are ignored.
/// @param n           Number of qubits.
/// @param port_wires  Indices of port (A) wires; complement is B.
PortRanks port_ranks(const std::vector<Pauli>& gens, int n,
                     const std::vector<int>& port_wires);

// ── Task 2: PortVerdict — compile-time separability verdict on a FramedSuperposition ──────

/// A port qubit that is stabilized by a definite single-qubit Pauli operator.
/// axis: 0=X, 1=Y, 2=Z.  sign: 0=+1, 1=-1.
struct ProdQ { int q = 0; uint8_t axis = 0; uint8_t sign = 0; };

/// Full compile-time separability verdict.
struct PortVerdict {
    bool product = false;           ///< true iff rank(S ∩ P_B) == |B|  (the port is separable)
    int  k_port  = 0;               ///< number of free (magic) generators: (int)st.free.size()
    std::vector<ProdQ> stabilized;  ///< port wires with a definite single-qubit Pauli axis
    std::vector<int>   port_wires;  ///< the port wires (verbatim from the call argument)
    std::string witness;            ///< "" iff product; else a human-readable refusal reason
};

/// Compute the port-contract verdict for a compiled FramedSuperposition.
///
/// Common stabilizer generators = { st.U.Zrow[a] : a not in st.free }, rank n-k.
/// VACUOUS CASE: if port_wires is empty, returns {product:true, k_port:(int)st.free.size(),
/// port_wires:{}, stabilized:{}, witness:""} without calling port_ranks.
PortVerdict port_contract(const FramedSuperposition& st, const std::vector<int>& port_wires);

}  // namespace qeccore
