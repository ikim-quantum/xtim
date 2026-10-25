#pragma once
// J-a N5 (switched OFF; design §4 "signs()", adv-A r5check / r6check): the definite-sign rule of a
// retained record state. Built only by FAM samplers (test API); nothing in production calls it.
//
// The retained state of a record (C0, P_rec, y) is psi ∝ Π_c (1 + (−1)^{y_c} M_c) · P_rec · C0 |B⟩.
// Pulled back by P_rec·C0: psi_b ∝ Π_c (1 + (−1)^{y_c ⊕ t_c} Q_c) |B⟩, Q_c = C0† M_c C0 (exact phase),
// t_c = ⟨P_rec, M_c⟩ (anticommutation bit). Projecting a commuting family onto a stabilizer state keeps
// the commuting subgroup and adds the projected operators, so (up to the α register's own symmetries)
//     H = ⟨ {s ∈ ⟨G⟩ : s commutes with every Q_c},  {(−1)^{y_c ⊕ t_c} Q_c} ⟩,
// G = the certified +1 stabilizers of |B⟩ (exact phases). g is DEFINITE iff ±C0†gC0 ∈ H, and then
//     ⟨g⟩ = (−1)^{ eps ⊕ ⊕_{c∈U} (y_c ⊕ t_c) ⊕ ⟨P_rec, g⟩ },
// where C0†gC0 = (−1)^eps Π_{c∈U} Q_c · s (s ∈ the commuting subgroup). Otherwise signs() reports 0
// (never a sampled coin). Under-reporting (0 on a definite g) is possible when α has an uncertified
// Pauli stabilizer (design §4, OPEN with the zero-probability certification).
//
// The representation prefers RECORDS: the elimination basis takes the records first (storage column
// order), then the commuting subgroup's basis — so a g that is both a record product and a group
// element is valued through its records (that is where the t_c term lives, adv-A r6check).

#include <cstdint>
#include <vector>

#include "qeccore/pauli.hpp"

namespace qeccore {

struct SignRule {
    bool definite = false;
    uint8_t eps = 0;                 // the fixed sign bit
    std::vector<uint64_t> U;         // the records used (bit c of word c >> 6), over the record columns
};

class SignRuleBuilder {
public:
    // Q: per record column c, Q_c = C0† M_c C0 (exact phase); use[c] = 0 drops column c from H (it is
    // then never in U). G: the certified generators (exact phases, +1 on |B⟩). n qubits.
    SignRuleBuilder(int n, const std::vector<Pauli>& Q, const std::vector<uint8_t>& use, const std::vector<Pauli>& G);
    // The rule of g, given gp = C0† g C0 (exact phase). Throws std::logic_error if gp is ± a member of H
    // only up to a factor of ±i (a non-Hermitian input: an engine bug).
    SignRule rule(const Pauli& gp) const;
    int num_records() const { return m_; }
    int commuting_rank() const { return (int)S_.size(); }   // rank of the commuting subgroup's basis

private:
    int n_ = 0, m_ = 0, NW_ = 0, TW_ = 0;
    std::vector<Pauli> Q_;                       // the records (index = storage column)
    std::vector<Pauli> S_;                       // basis of {s ∈ ⟨G⟩ : [s, Q_c] = 0 ∀ used c}, exact phase
    // echelon basis over GF(2)^{2n}: symplectic vector (2·NW words), pivot bit, tag (TW words over m + |S|)
    std::vector<std::vector<uint64_t>> vec_, tag_;
    std::vector<int> piv_;
    void add_(std::vector<uint64_t> v, std::vector<uint64_t> t);
};

}  // namespace qeccore
