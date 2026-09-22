#pragma once
// twirl_ppr — V2-T1: the diagonal twirl shot law transported to the AXIS-DRESSING space F2^t
// for commuting Pauli-Product-Rotation (PPR / CH-class) residuals.
//
//   E = prefix · Π_j exp(i·π·e_j/4 · A_j)   (t mutually-commuting Hermitian axes, e_j odd)
//
// Conjugating a certified generator gives h_i = E† g_i E = s_i · g_i · A_{col_i}, where
// col_i ∈ F2^t marks the axes anticommuting with g_i (the i-th column of the pattern matrix
// L[j][i] = ⟨A_j, g_i⟩). The V1 roles transport as
//   v_i → col_i,   G_z → Q = {w : A_w ∈ ±G},   N_z → ker L = {w : Σ_j w_j L_j = 0},
//   V = span{col_i},  Kz = V ∩ ker L,  D = Kz ∩ Q,   r = dim V − dim Kz,
//   coin_masks = rowspace(Π), Π_row_i = Σ_j col_i[j]·L_j  (ng-bit σ-flip masks).
//
// KERNEL SPLIT (the V2 reachability adjudication, executable spec:
// scripts/twirl_ppr_reference.py, 50/50 vs twirl_oracle.analytic_law at 1e-12):
//   • every Kz direction is FOLDABLE (it lies in V, so the preimage system is feasible);
//     logical Kz dirs (mod D) carry the V1 σ-fold machinery verbatim (preimage / δ-mask /
//     kernel_base), with the fold weighted by the magic Born ⟨Λ⟩ downstream.
//   • logical ker-L dirs OUTSIDE V are UNFOLDABLE: ker L = V^⊥ in F2^t, so such a dir never
//     appears as a generator dressing — the σ-marginal carries ZERO information about it.
//     They enter kernel_logicals with kernel_foldable=0 and an EMPTY mask (Born-only).
// This resolves the "full transversal S_L: t=7, rank L=3, sectors=1" collapse: there
// V = Kz (r=0) and the true logical direction (odd weight) is outside V.
//
// SCOPE (mirrors the diagonal κ-guards + the reference): ≥2 foldable logical dirs, or a
// foldable logical coupled with a foldable in-group (D) dir, or non-commuting axes, or
// t > 60 → law.fallback = true (loud, never a wrong σ).
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/pauli.hpp"
#include "qeccore/ppr_residual.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// Canonical commuting-PPR normal form: axes canonical Hermitian (phase ∈ {0,1}, a −1 sign
// folded into the exponent), DISTINCT, mutually commuting, e ∈ {1,3,5,7}, sorted by (x,z) key.
struct PprNormalForm {
    Pauli prefix;                                   // exact Pauli prefix (global phase dropped)
    std::vector<std::pair<Pauli, int>> rots;        // (axis, e) canonical odd rotations
    bool commuting_class = true;
    explicit PprNormalForm(int n = 0) : prefix(n) {}
};

// One fired atom for composition (views into caller-owned storage; E_atom = prefix · Π rots,
// rots applied first — the PprResidual convention).
struct PprAtom {
    const Pauli* prefix;
    const std::vector<std::pair<Pauli, int>>* rots;
};

// Exact composition of fired atoms in firing order (executable spec: compose_ppr in
// scripts/twirl_ppr_reference.py). Prefixes move left through rotation factors with the
// axis-sense negation P·rot(e,A) = rot(e·(−1)^{⟨P,A⟩}, A)·P; same-axis exponents add mod 8;
// even totals fold into the prefix (exp(i·π·e/4·A) = phase·A^{e/2}, global phase dropped);
// e ≡ 0 axes vanish. Mutually anticommuting survivors ⇒ commuting_class = false.
PprNormalForm compose_ppr(int n, const std::vector<PprAtom>& atoms);

// DiagNormalForm content as PPR rotation atoms: S^a → rot((−a) mod 8, Z_q) per a_q ∈ {1};
// CZ(j,l) → rot(7,Z_j)·rot(7,Z_l)·rot(1,Z_j Z_l); the prefix is carried unchanged.
// (Used to fold diagonal alts into a mixed diagonal+PPR shot composition.)
void ppr_rots_from_diag(const DiagNormalForm& nf, int n,
                        std::vector<std::pair<Pauli, int>>& rots_out);

// Axis-space shot law. Same ShotLaw contract as build_shot_law: σ-space masks, det_signs a
// valid base point (C_det-solved, inactive bits pinned to the prefix plane), kernel_foldable
// flags per the reachability split. kappa counts ALL logical kernel dirs (fold + unfold).
ShotLaw build_shot_law_ppr(const CertifiedGroupPlanes& G, const PprNormalForm& pnf);

// ── Content-keyed PPR plan memo ───────────────────────────────────────────────────────────
// Key = canonical rot list (axis x/z words + exponent, sorted) + group token; PREFIX-FREE
// (det_signs = base0 ⊕ prefix_plane1(G, prefix) — the same separability as the diagonal memo:
// h_i carries the (−1)^{⟨P,g_i⟩} factor and nothing else depends on P).
struct PprCachedPlan {
    int r = 0;
    int kappa = 0;
    bool fallback = false;
    std::vector<uint64_t> base0;                    // det_signs @ identity prefix
    std::vector<std::vector<uint64_t>> coin_masks;
    std::vector<std::vector<uint64_t>> kernel_masks;   // parallel to kernel_logicals ([] = unfoldable)
    std::vector<uint8_t> kernel_base;
    std::vector<uint8_t> kernel_foldable;
    std::vector<Pauli> kernel_logicals;
    std::vector<uint64_t> active;                   // tier1 active mask (prefix-free)
    // key content (bucket equality verification)
    std::vector<uint64_t> key_words;                // packed (x,z,e) per rot, sorted
    // Per-(plan, bare-state) fold Born weights, cached by the record consumer (mutable like the
    // diagonal plan's channel rows): p1[j] = P(outcome −1) = (1 − ⟨Λ_j⟩)/2 for FOLDABLE dirs.
    mutable std::vector<double> fold_p1;
    mutable uint64_t folds_built = 0;               // 1 once the weights are built (one bare state per consumer)
    // Channel-space rows (Tier B), lazily built per channel set — same contract as CachedPlan.
    mutable std::vector<uint64_t> ch_baseline;
    mutable std::vector<std::vector<uint64_t>> ch_coin_rows;
    mutable std::vector<std::vector<uint64_t>> ch_kernel_rows;
    mutable uint64_t ch_token = 0;
    // V3 Born-weighted observable channel data (same contract as CachedPlan's obs_*):
    mutable uint8_t obs_ready = 0;
    mutable uint8_t obs_guard = 0;
    mutable uint8_t obs_par_base = 0;
    mutable std::vector<uint8_t> obs_par_coin;
    mutable double obs_m = 0.0;
};

class PprPlanCache {
  public:
    // Get (or build + cache) the plan for pnf's ROT content (prefix ignored — prefix-free key).
    const PprCachedPlan& get_or_build(const CertifiedGroupPlanes& G, const PprNormalForm& pnf);
    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }
    size_t size() const { return n_plans_; }

  private:
    std::unordered_map<uint64_t, std::vector<std::unique_ptr<PprCachedPlan>>> store_;
    size_t hits_ = 0, misses_ = 0, n_plans_ = 0;
};

// V3: Born-weighted observable channel for a PPR plan (axis-space transport of
// classify_observable; executable spec twirl_ppr_reference.classify_observable_ppr,
// gated 16/16 vs analytic_law conditionals at 1e-12). Identity-prefix convention;
// reachability of the axis pattern u_W solved MOD kerL (the axis-space normalizer
// frame). Same ObsChannel contract as the diagonal classify.
ObsChannel classify_observable_ppr(const CertifiedGroupPlanes& G, const PprNormalForm& pnf,
                                   const Pauli& W);

// Pack a canonical rot list into key words (axis x-words, z-words, exponent per rot, in the
// canonical sorted order) — the memo identity. Exposed for tests and the record consumer.
void ppr_key_words(const PprNormalForm& pnf, int n, std::vector<uint64_t>& out);

}  // namespace qeccore
