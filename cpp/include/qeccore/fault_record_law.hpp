#pragma once
// fault_record_law — THE stabilizer-twirl record law of one residual (twirl-routes plan §1, owner mandate
// 2026-10-30: one routine serves every law route). A residual R = P·C (DiagNormalForm nf: an exact Pauli prefix P
// and a diagonal S/CZ residual C) acting on the bare state ψ, read through record columns M_c (products of
// terminal reads, exact phase), has the joint record law of Theorem 1 (design.final.md §2.2):
//     y_c = A_c·u ⊕ B_c·z ⊕ β_c ⊕ ⟨P, M_c⟩                    (u fair coins, z the κ Born bits)
// — the sampler's own canonical record table (build_shot_law_core + the record-space B stage,
// record_space_b.hpp), P-free, plus Lemma 2's prefix flips. The exact law of E = P·Δ equals the law of its
// twirl over the certified group G (every M_c commutes with G): the coin part (K0, the closed form β, A) needs no
// Born sum; only the κ ≤ k kernel-logical directions (K1/K0) carry a free-register Born law, P(z) for
// z ∈ {0,1}^κ, evaluated by born_law below with 2^κ ≤ χ leaves. Nothing here is exponential in anything else.
// Consumers: the DEM per-fault law (dem_export.cpp), the per-shot expectation-column law (T2).
#include <cstdint>
#include <string>
#include <vector>

#include "qeccore/born_law_coset.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// The canonical record table of the residual nf (its prefix is ignored: the table is P-free, Lemma 2) over the
// record columns `cols` (channel columns: IN_GROUP with their generator mask and reference sign; extras:
// anything else). False iff the plan's law is a fallback (never for a diagonal nf).
bool fault_record_law(const CertifiedGroupPlanes& G, const FramedSuperposition& bare, const DiagNormalForm& nf,
                      const std::vector<RecColumn>& cols, RecTable& out);

// The joint Born law of κ commuting Hermitian free-register directions on the bare state: p[z] for every
// z ∈ {0,1}^κ (bit j of z = outcome of direction j, 1 ⇔ eigenvalue −1), by the ONE Born-law routine
// (born_law_enumerate, born_law_coset.hpp): BornTrie::enumerate — sequential projection on the free register, the
// sampler's own node operations — for κ ≤ 2 or when its cost model wins, else the coset-of-X-span law (explaw §1:
// O(s·κ + Σ_C min(m_C², κ 2^κ) + κ 2^κ) instead of Θ(2^κ·s)), both with enumerate's leaf semantics; a prefix of
// probability 0 is not expanded. `evals` (optional) accumulates the node + leaf evaluations (enumerate, ≤ 2^(κ+1) − 1)
// or the leaves + cosets (the coset law); `stats` (optional) counts the coset-law routes and their work. κ ≤ k always
// (the directions are independent commuting operators on k free coordinates), so the table has at most 2^k = χ-bounded
// entries.
struct BornLawStats { long coset_laws = 0; double coset_work = 0.0; };
std::vector<double> born_law(const FramedSuperposition& bare, const std::vector<FreeDir>& lam,
                             long* evals = nullptr, BornLawStats* stats = nullptr);

// The Born quotient of a record law f = c ⊕ Σ u_k W_k ⊕ Σ z_j B_j (u fair, z ~ pz; patterns on UW words): the classes
// of Σ z_j B_j mod span(W) with their weights Σ P(z) (support filter: P(z) > 1e-12 · max P). One class, or 2^d equal
// classes forming an affine set r0 + span(dirs), is a uniform COSET (centre shift r0, W extended by dirs); otherwise a
// Born MIXTURE (cls: class representative reduced mod span(W), weight; dirs: the classes' difference basis mod W).
struct BornQuotient {
    bool coset = true;
    std::vector<uint64_t> r0;
    std::vector<std::vector<uint64_t>> dirs;
    std::vector<std::pair<std::vector<uint64_t>, double>> cls;
};
BornQuotient born_quotient(const std::vector<std::vector<uint64_t>>& W, const std::vector<std::vector<uint64_t>>& B,
                           const std::vector<double>& pz, int UW);

// Record columns of the certified group G: an operator M (a product of commuting terminal reads, exact phase)
// becomes a CHANNEL column (IN_GROUP: the generator mask Π_{mask} g = ±M by GF(2) provenance, ref = the sign) or
// an EXTRA column (LOGICAL / ANTI). Built once per group (one provenance elimination over the generators).
class RecordColumns {
  public:
    explicit RecordColumns(const CertifiedGroupPlanes& G);
    RecColumn column(const Pauli& M, int order) const;

  private:
    const CertifiedGroupPlanes& G_;
    int NW_ = 0, GW_ = 0;
    struct Row { std::vector<uint64_t> s, c; int pc; };
    std::vector<Row> rows_;
};

}  // namespace qeccore
