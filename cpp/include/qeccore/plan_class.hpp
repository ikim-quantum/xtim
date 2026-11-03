#pragma once
// plan_class — the K2' kernel of the diagonal plan key (plancache-design/plan.md §0-§1, owner-signed rev 4).
//
// Theorem T3 (plan §0). Two diagonal residuals C0 = S^a·CZ^B and C0' = C0·CZ^ΔB have the SAME record law on every
// bare state (magic included) when, for every record column c (channels AND extras: Born observables, decisions,
// family extras) with X-support x_c,
//   (i)  g_c = (−1)^{q_c(ΔB)} · Z^{ΔB·x_c} is a +1-signed element of the certified group G, with
//        ΔB·x_c = Σ_{(j,l)∈ΔB} (x_c(j) e_l ⊕ x_c(l) e_j) and q_c(ΔB) = Σ_{(j,l)∈ΔB} x_c(j) x_c(l)   (mod 2), and
//   (ii) β(c, c') = x_c^T ΔB x_c' = Σ_{(j,l)∈ΔB} (x_c(j) x_c'(l) ⊕ x_c(l) x_c'(j)) = 0 for every pair c < c'.
// Both are GF(2)-LINEAR in ΔB (the sign character of the pure-Z subgroup G_z is a homomorphism: G holds no −I), so
// K2' = {ΔB : (i) ∧ (ii)} is a subspace of the pair space, computed ONCE at construction by one GF(2) elimination.
// An a-bit on a wire with X-support in NO record column changes no pullback C0†M_cC0 at all, so the class key is
//   (a ∧ U ∧ ¬prod, Π(B)),  U = wires with X-support in any column, Π the projection with ker Π = K2'.
// Π(B) is the CANONICAL coset representative: B with every pivot bit of the reduced-echelon kernel basis cleared
// (B ⊕ Σ_i B[p_i]·K_i), so it is the same whichever key of the class arrives first, on every lane.
//
// The elimination streams the matrix columns in 2^16-bit blocks against a ≤ pairs²-bit kernel basis (memory ≤ 2 MiB
// basis + 32 MiB block at the pair cap). Deterministic cap (plan §1, review F3 / N3 / L3): Π := id when the kept pair
// count exceeds kMaxPairs or the word-work bound W = kept² · ceil(columns / 64) exceeds kMaxWork — a function of the
// circuit alone (never a clock, never a refusal); the a ∧ U strip is kept either way.
// O3 (plan §6): compile-time audit of every kernel basis vector on every column by the EXACT Pauli kernels
// (G.reduce for (i), the direct GF(2) sum for (ii)); a failure is an engine-bug throw.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/record_table.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

struct PlanClassifier {
    int n = 0, NW = 0, PW = 0, KW = 0;
    bool pi_identity = true;            // no pair is projected (no kernel, or the cap)
    std::vector<uint64_t> umask;        // NW words: U ∧ ¬prod — the a-part of the class key is key & umask
    std::vector<uint64_t> pkeep;        // PW words: the kept pairs (neither wire a product wire): a cache key's pairs
    std::vector<int> pivots;            // kernel basis pivot pair indices (ascending)
    std::vector<uint64_t> basis;        // dim() rows × PW words: reduced echelon (row i: bit pivots[i], no other pivot)
    // the compile's record (the PLAN_CLASS diag event, record_report)
    size_t pairs = 0, kept_pairs = 0, columns = 0;
    uint64_t work = 0;                  // W = kept_pairs² · ceil(columns / 64)
    std::string cap;                    // "" (the kernel is in force) or why Π = id ("pairs" / "work")
    size_t audited = 0;                 // O3: basis vectors audited on every column

    size_t dim() const { return pivots.size(); }
    // Π in place on a cache key (NW a-words, then PW pair words over the pair universe).
    void project(uint64_t* key) const {
        for (int w = 0; w < NW; ++w) key[w] &= umask[(size_t)w];
        if (pi_identity) return;
        const uint64_t* B = basis.data();
        for (size_t i = 0; i < pivots.size(); ++i, B += PW) {
            const int p = pivots[i];
            if ((key[NW + (p >> 6)] >> (p & 63)) & 1ULL)
                for (int w = 0; w < PW; ++w) key[NW + w] ^= B[w];
        }
    }
    // True iff Π(key) == key (the key is its own class representative).
    bool is_representative(const uint64_t* key) const {
        for (int w = 0; w < NW; ++w) if (key[w] & ~umask[(size_t)w]) return false;
        if (pi_identity) return true;
        for (int p : pivots) if ((key[NW + (p >> 6)] >> (p & 63)) & 1ULL) return false;
        return true;
    }
};

inline constexpr size_t kPlanClassMaxPairs = 4096;
inline constexpr uint64_t kPlanClassMaxWork = 500000000ULL;
inline constexpr size_t kPlanClassBlockBits = 65536;

// Compile the classifier of a sampler's key universe: the certified group G (its signed pure-Z subgroup), the record
// columns (storage order; every column counts), the sorted pair universe with its kept-pair mask (pairs with two
// non-product endpoints: the only pairs a key holds) and the product-wire mask. `audit` runs O3 on the result.
// identity_only: the null option (test lever): Π = id and the a ∧ U strip only.
PlanClassifier compile_plan_classifier(const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                                       const std::vector<std::pair<int, int>>& upairs,
                                       const std::vector<uint64_t>& upair_keep, const std::vector<uint64_t>& prodmask,
                                       int n, bool audit, bool identity_only = false);

// O3: audit every basis vector of `pc` against the exact Pauli kernels (throws std::logic_error on a violation).
void audit_plan_classifier(const PlanClassifier& pc, const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                           const std::vector<std::pair<int, int>>& upairs);

// Exact (i) ∧ (ii) of ONE pair set ΔB (bitset over the pair universe) — the reference the audit and the unit test use.
bool plan_class_kernel_member(const uint64_t* dB, const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                              const std::vector<std::pair<int, int>>& upairs);

}  // namespace qeccore
