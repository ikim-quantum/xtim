#pragma once
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "qeccore/exact_phase.hpp"
#include "qeccore/stab_affine.hpp"   // SymPackedMat, AffineState

namespace qeccore {

// --- B-matrix helpers (B = symmetric GF(2), zero diagonal, reusing SymPackedMat) ---
SymPackedMat sym_zeros(int n);                                   // n×n all-zero
void sym_xor_into(SymPackedMat& dst, const SymPackedMat& src);   // dst ^= src (elementwise)
std::vector<uint8_t> sym_matvec(const SymPackedMat& B,
                                const std::vector<uint8_t>& v);  // (B·v) over GF(2)

// Diagonal+Pauli Clifford: U = gamma · X^v · diag(q),
//   q(y) = Σ_j a_j y_j + 2 Σ_{j<l} B_{jl} y_j y_l  (mod 4).
struct DiagPauliClifford {
    int n = 0;
    ExactPhase gamma = ExactPhase::one();
    std::vector<uint8_t> a;   // length n, ℤ₄ (0..3): S/Z layer
    SymPackedMat B;           // n×n symmetric GF(2), zero diagonal: CZ layer
    std::vector<uint8_t> v;   // length n, GF(2): X-translation (Pauli)

    static DiagPauliClifford identity(int n);
    static DiagPauliClifford S(int n, int q);
    static DiagPauliClifford Sdg(int n, int q);
    static DiagPauliClifford Z(int n, int q);
    static DiagPauliClifford X(int n, int q);
    static DiagPauliClifford Y(int n, int q);
    static DiagPauliClifford CZ(int n, int c, int t);

    // Precomputed operator-only structure (read by residual_normal_form).
    struct Cache {
        std::vector<int> active_a;                    // qubits j with a[j] != 0
        std::vector<std::pair<int,int>> b_pairs;      // (j,l) with j<l and B_{jl}=1
    };
    std::shared_ptr<const Cache> cache;               // built by finalize(); shareable read-only

    // Build the cache once. THREAD-SAFETY CONTRACT: call finalize() once, single-threaded; after
    // that the cache is immutable and shared safely via shared_ptr<const>. Idempotent.
    void finalize();

    DiagPauliClifford then(const DiagPauliClifford& next) const;  // next ∘ this
};

}  // namespace qeccore
