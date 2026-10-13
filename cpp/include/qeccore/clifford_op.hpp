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

// Reusable scratch for DiagPauliClifford::apply — pass one per worker thread to make the
// hot apply-to-many loop allocation-free. Buffers retain capacity across calls.
struct DiagApplyWorkspace {
    std::vector<long long> lin;
    std::vector<uint8_t>   qpair;
    std::vector<int>       sup_off;
    std::vector<int>       sup_idx;
    std::vector<int>                  active_a;   // local build when op not finalized
    std::vector<std::pair<int,int>>   b_pairs;
};

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

    // Precomputed operator-only structure shared read-only across an apply-to-many batch.
    struct Cache {
        std::vector<int> active_a;                    // qubits j with a[j] != 0
        std::vector<std::pair<int,int>> b_pairs;      // (j,l) with j<l and B_{jl}=1
    };
    std::shared_ptr<const Cache> cache;               // built by finalize(); shareable read-only

    // Build the cache once. THREAD-SAFETY CONTRACT: call finalize() once, single-threaded,
    // before applying the operator to many states (possibly in parallel). After that the cache
    // is immutable and shared safely via shared_ptr<const>. Idempotent (rebuild is fine).
    void finalize();

    void apply(AffineState& s) const;                       // wrapper (allocates a local ws)
    void apply(AffineState& s, DiagApplyWorkspace& ws) const;
    DiagPauliClifford then(const DiagPauliClifford& next) const;  // next ∘ this

   private:
    using PairList = std::vector<std::pair<int,int>>;
    // k>64 fallback: n-space per-pair flips (validated). apply() dispatches here for large k.
    void apply_generic(AffineState& s, DiagApplyWorkspace& ws,
                       const std::vector<int>& actives, const PairList& bpairs) const;
    // 4≤k≤64 runtime-k hot path (bit-mask supports, packed word M).
    void apply_kspace(AffineState& s, DiagApplyWorkspace& ws,
                      const std::vector<int>& actives, const PairList& bpairs) const;
    // compile-time-K specializations for k=0..3.
    template<int K> void apply_impl(AffineState& s, DiagApplyWorkspace& ws,
                                    const std::vector<int>& actives, const PairList& bpairs) const;
};

}  // namespace qeccore
