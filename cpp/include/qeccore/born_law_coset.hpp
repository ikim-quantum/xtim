#pragma once
// born_law_coset — the coset-of-X-span Born law (explaw §1, owner-approved 2026-10-30).
//
// Objects. The bare free register ψ: a sparse key list on k coordinates (DenseAmplitudes::b_, support s). A law has
// d commuting Hermitian free-register directions λ_j = c_j X^{a_j} Z^{b_j} (FreeDir: (λψ)_x = c (−1)^{b·(x⊕a)} ψ_{x⊕a},
// c = i^phase (−1)^par) and optionally one column X0 = c_0 X^{a_0} Z^{b_0} commuting with every λ_j. Wanted for every
// z ∈ {0,1}^d (bit j = outcome of λ_j is −1):  P(z) = ‖Π_z ψ‖²,  N(z) = Re⟨ψ|X0 Π_z|ψ⟩,  h(z) = N(z)/P(z).
// BornTrie::enumerate evaluates the same objects by sequential projection, Θ(Σ_nodes χ_node) = Θ(2^d·s) on a dense
// hull; this file evaluates them in O(s·d + Σ_C min(m_C², d_X 2^{d_X}) + #classes·d_X 2^{d_X} + d·2^d):
//   A. a GF(2) re-basis of the directions: products λ'_i whose X-parts form a basis of A = span{a_j} (d_X of them,
//      "mixed") and pure-Z products (d_Z = d − d_X); outcome bits transform by an invertible M (XOR of factor bits);
//   B. the support splits into cosets C = x0 + A; every pure-Z direction is a CONSTANT ε_j(C) on a coset (its class
//      ε(C)); every mixed direction acts on the coset's d_X coordinates t as c'_i(C) X_i Z^{B_i}, B_il = ⟨b_i, a_l⟩
//      symmetric (commutation);
//   C. D = Π_{i<l} CZ_il^{B_il} Π_i S_i^{B_ii} and H^{⊗d_X} diagonalise them: P_C(z) = |Φ_C(z)|², Φ_C = WHT(D†ψ_C),
//      the eigen-label shifted by a per-coset sign vector σ(C) (c''_i(C) = ±1); dense per coset (one WHT) or, for a
//      small coset, pair sums e_C(S) accumulated per class and ONE WHT per class (the cheaper of the two, a pure
//      function of (m_C, d_X));
//   D. the column: a_0 ∈ A ⇒ diagonal in that basis (N_C = ω_C(z) P_C); a_0 ∉ A ⇒ a coset swap C ↔ C ⊕ a_0 (same
//      class) with N(z) = Σ_C ω_C(z) Re conj(Φ_C(z)) Φ_{C'}(z ⊕ r);
//   E. the leaf semantics of BornTrie::enumerate replayed from the table (coset_law_leaves): the trie's conditionals
//      are prefix ratios in the caller's direction order, a conditional within eps of 0 / 1 is deterministic exactly
//      as enumerate decides, the leaf weight is the product of the BRANCHING conditionals, h on every leaf. The law
//      differs from enumerate's by rounding only.
// Exactness margin: every per-coset sum (the WHT butterflies, the pair products, |Φ|²) is accumulated in
// double-double, so a truly empty eigen-component is ~1e-64 relative (never a spurious branch at kBranchEps) and h
// keeps ≥ 1e-12 relative precision on every leaf of weight ≥ 1e-20; the final tables are doubles.
// Not a sampling path: the DEM's Born laws (born_law) and the per-shot expectation-column walks (PeShotAnalyzer).
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/twirl_kernel.hpp"

namespace qeccore {

// The joint table in the caller's direction order, unnormalised (‖ψ‖² scale): P[z] = ‖Π_z ψ‖², N[z] = Re⟨ψ|X0 Π_z ψ⟩
// (N empty without a column). Census fields: the support s, the cosets of A on it, the classes, the per-pair routes.
struct CosetLawTable {
    int d = 0, dX = 0, dZ = 0;
    long support = 0, cosets = 0, classes = 0, dense_pairs = 0, sparse_pairs = 0;
    double work = 0.0;        // the operations performed: s·d + Σ_pairs (dense (d_X+2)·2^{d_X} per coset | pair products)
                              //   + sparse classes·d_X·2^{d_X} + d·2^d (the replay); the mechanism counter
    bool column = false;
    std::vector<double> P, N;
};

// The route of one content (read upfront in O(s·d): s, d, d_X, the coset census): the coset law or
// BornTrie::enumerate, by the measured cost model (ties and d ≤ 2 → enumerate).
struct BornRoute {
    bool coset = false;
    double cost_enum = 0.0, cost_coset = 0.0;   // model nanoseconds
    int d = 0, dX = 0;
    long s = 0, cosets = 0, classes = 0;
};

class CosetLaw {
  public:
    // Steps A–B (+ the route estimate). Throws std::logic_error on a non-Hermitian or non-commuting direction (an
    // engine bug: every caller's directions are commuting Hermitian Paulis).
    CosetLaw(const FramedSuperposition& bare, const std::vector<FreeDir>& lam);
    const BornRoute& route() const { return route_; }
    // Steps C–E's table (X: the column, or nullptr). X must commute with every direction (std::logic_error otherwise).
    void build(const FreeDir* X, CosetLawTable& out) const;

  private:
    struct Dir { uint64_t a = 0, b = 0; int q = 0; };   // i^q X^a Z^b
    struct Ent { uint64_t x0, t; uint32_t idx; };
    struct Cos { uint64_t x0; size_t begin, end; };
    static Dir dir_of(const FreeDir& d, int k);
    Dir product(uint64_t comb) const;
    uint64_t coords(uint64_t x, uint64_t& t) const;   // coset representative + coordinates
    uint64_t class_of(uint64_t x0) const;
    uint64_t sigma_of(uint64_t x0) const;
    inline uint8_t B(int i, int l) const { return B_[(size_t)i * (size_t)dX_ + (size_t)l]; }

    const FramedSuperposition& bare_;
    const DenseAmplitudes* da_ = nullptr;
    int k_ = 0, d_ = 0, dX_ = 0, dZ_ = 0;
    std::vector<Dir> dirs_, mixed_, pureZ_;
    std::vector<int> piv_;
    std::vector<uint64_t> alpha_, comb_;
    std::vector<uint8_t> B_;
    std::vector<Ent> ent_;
    std::vector<Cos> cos_;
    BornRoute route_;
};

// h(z) = N[z] / P[z] clamped to [−1, 1] (0 when P[z] = 0 or no column): the per-shot column value on leaf z.
double coset_law_value(const CosetLawTable& T, uint64_t z);

// Step E: BornTrie::enumerate's leaves from the table (+ outcome first, depth first, the same stop contract).
// `evals` (optional) accumulates the leaves visited.
void coset_law_leaves(const CosetLawTable& T, double eps, const std::function<bool(uint64_t, double, double)>& leaf,
                      long* evals = nullptr);

// THE Born-law routine: route by the cost model, then BornTrie::enumerate or the coset law with enumerate's leaf
// semantics. `evals` accumulates node + leaf evaluations (enumerate) or leaves + cosets (the coset law). table_out
// (optional) receives the coset law's table (null when enumerate ran); route_out (optional) the decision.
// A table passed IN through table_out (the same content, e.g. a copied Law) is replayed without a rebuild.
void born_law_enumerate(const FramedSuperposition& bare, const std::vector<FreeDir>& lam, const FreeDir* X, double eps,
                        const std::function<bool(uint64_t, double, double)>& leaf, long* evals = nullptr,
                        std::shared_ptr<const CosetLawTable>* table_out = nullptr, BornRoute* route_out = nullptr);

}  // namespace qeccore
