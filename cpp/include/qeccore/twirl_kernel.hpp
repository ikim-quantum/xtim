#pragma once
// twirl_kernel — Task 6: residual normal form for the diagonal twirl kernel.
//
// A composed shot residual is a DiagPauliClifford  U = gamma · X^v · diag(q),
//   q(y) = Σ_j a_j y_j + 2 Σ_{j<l} B_{jl} y_j y_l   (a_j ∈ ℤ₄, B symmetric GF(2)).
//
// residual_normal_form splits U into an EXACT Pauli prefix P and a canonical
// diagonal residual (S-layer + CZ-layer) by the even-fold  a_j = s_j + 2 z_j:
//
//   U = [ gamma · X^v · Z^z ] · [ S^s · CZ(B) ]  =  P · C
//
//   * s_j = a_j & 1  ∈ {0,1}  — the residual S content         (DiagNormalForm::a)
//   * z_j = (a_j>>1) — the Z part folded S²=Z into the prefix   (P's Z support)
//   * v_j             — the X-translation folded into the prefix (P's X support)
//   * B pairs         — the CZ layer                             (DiagNormalForm::cz)
//   * gamma           — global phase folded into the prefix Pauli's i^phase.
//
// Since Z, S and CZ are all diagonal they commute freely, so moving Z^z left into
// the prefix and keeping S^s·CZ on the right is an EXACT operator identity (no extra
// phase: S²=Z=diag(1,−1), S³=Z·S=diag(1,−i)). The prefix carries the exact phase.
//
// The fold reproduces U bit-for-bit: reconstructing P·C as a DiagPauliClifford and
// applying it to any AffineState is byte-identical to applying U (the test oracle).

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/clifford_op.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

struct DiagNormalForm {
    Pauli prefix;                          // exact Pauli i^phase · X^v · Z^z (phase ∈ {0..3})
    std::vector<uint8_t> a;                // length n, {0,1} per qubit: residual S content (a_j & 1)
    std::vector<std::pair<int,int>> cz;    // CZ layer, each pair j<l, sorted ascending
    bool diagonal_class = false;           // always true here (see residual_normal_form)
};

// Extract the (P, a, cz) normal form from a composed diagonal+Pauli Clifford, folding
// the even S²=Z part and the X-translation into the exact Pauli prefix.
DiagNormalForm residual_normal_form(const DiagPauliClifford& composed);

// ── Task 11.5: canonicalise the residual modulo the certified group's Z-content. ─────────────
// The twirl law depends on R only through R|ψ⟩, and any certified stabiliser s satisfies
// s|ψ⟩ = |ψ⟩, so R·s produces the SAME state. Concretely, for a pure-Z stabiliser Z^u = ε·(+1 op)
// (ε∈{±1}) and an odd π/4 Z-rotation about axis v, the exact STATE identity
//        exp(iθ Z^v)|ψ⟩ = exp(iθ ε Z^{v'})|ψ⟩ ,   v' = v ⊕ u   (proof: g_op²=I, g_op|ψ⟩=|ψ⟩)
// lets us reduce every odd rotation axis v of C against the pure-Z RREF of G, flipping the
// rotation sense by the consumed rows' sign product ε (ε=−1 swaps S↔S†). C's diagonal content is
// the odd-axis set {e_q : a_q=1} together with each CZ(q,q')'s 3-axis decomposition
// {e_q, e_{q'}, e_q⊕e_{q'}}; reducing each axis and rebuilding the phase polynomial
//   Φ_lin[q] += −2·e·v'_q ,  Φ_quad[(q,q')] += 4·e·v'_q v'_{q'}   (mod 8)
// yields a NEW (a', cz') plus even folds (S²=Z) into the prefix — a weight-≥3 reduced axis v'
// contributes v'v'ᵀ (S on each qubit, CZ on each pair), which is legal diagonal-Clifford content.
// The global phase (each axis' constant ζ8 term) is discarded per the standing rule.
//
// The result reduces distinct residuals to a common canonical (a',cz') key (product wires strip
// their S/CZ legs entirely), cutting both the memo miss rate and the cold-build cost. It is a SOUND
// conjugation/rewrite: R'|ψ⟩ = R|ψ⟩, so every twirl distribution/conditional is invariant (gated by
// the reference M0 corpus + the field-by-field harness). Cheap per-shot: pivot-local reduction
// against G's precomputed pure-Z RREF (a product-wire fast mask short-circuits the weight-1 case).
DiagNormalForm canonicalize_mod_stabilizers(const DiagNormalForm& nf, const CertifiedGroupPlanes& G);
// In-place variant (hot path, 2026-07-15): the no-touch identity case — the overwhelming majority —
// returns false having done NOTHING (no copy of a/cz/prefix; the by-value API pays a full
// DiagNormalForm copy even for the identity). Returns true iff nf was rewritten.
bool canonicalize_mod_stabilizers_inplace(DiagNormalForm& nf, const CertifiedGroupPlanes& G);

// ── Task 7: Tier-1 sign planes + active mask (spec §5 step 2). ──────────────────────────
// For each certified generator g_i, conjugation by the residual R = γ·P·C gives
//   h_i = R† g_i R = s_i · g_i · Z^{M x_i},   M = diag(a) + cz-adjacency (GF(2)),
// with s_i ∈ {±1} EXACT. tier1_signs computes, over ALL generators at once (columnar,
// bit-sliced over the n_gens-bit planes):
//   * active_i = (M x_i ≠ 0)      — OR over M-rows of the row-pattern planes.
//   * sign_i   = s_i sign BIT     — VALID ONLY where active_i == 0 (the M x_i = 0 case;
//                for active gens the Task-8 base-point solve overrides). Three XORed planes:
//        plane1 (prefix anticommute ⟨P,g_i⟩): XOR zcol(q) over supp_X(P), XOR xcol(q) over supp_Z(P)
//        plane2 (linear-a): carry-save popcount mod 4 of {xcol(q): a_q=1}; sign bit = the 2s bit
//                (the 1s bit is 0 on ker M — proven: on M x_i=0, Σa_q x_{i,q} ≡ xᵀMx = 0 mod 2)
//        plane3 (CZ): for each (q,q')∈cz: sign ^= xcol(q) & xcol(q')
// NOTE: the prefix P's OWN phase (its ± component, prefix.phase>>1) applies GLOBALLY to the
// residual state/coefficient — it CANCELS in the conjugation R† g_i R and does NOT enter the
// per-generator sign plane (only P's anticommutation with g_i does, via plane1). The global ±
// is accounted downstream by Task 8/10 (state coefficient), not here.
struct Tier1Result {
    std::vector<uint64_t> sign;    // n_gens bits (words uint64s): 1 = generator sign flips
    std::vector<uint64_t> active;  // n_gens bits: 1 = M x_i != 0 (sign plane not valid there)
};
Tier1Result tier1_signs(const CertifiedGroupPlanes& G, const DiagNormalForm& nf);

// ── Task 8: joint lattice law (spec §5 steps 3–5). ──────────────────────────────────────────
// The full per-shot syndrome law for a diagonal residual R = γ·P·C, built by classifying the
// Z-dressing LATTICE V = span{v_i}, v_i = M x_i, against the certified group's Z-frames rather
// than per generator (Phase-A bug #1 — a logical/in-group dressing appears only as an XOR of
// anticommuting generator dressings). Splitting V by
//     G_z = { Σ c_i z_i : Σ c_i x_i = 0 }   (Z^v ∈ ±group)      D  = V ∩ G_z
//     N_z = { u : u·x_i = 0 ∀ i }           (Z^v commutes)      Kz = V ∩ N_z   (G_z ⊆ N_z ⇒ D ⊆ Kz)
// gives   κ = dim Kz − dim D  (kernel/logical directions, Born-measured)
//         r = dim V − dim Kz  (fair independent coins).
//
// Fields (and their downstream consumers):
//   tier1        — the three sign planes + active mask (Task 7). active = (M x_i ≠ 0); the sign
//                  plane is authoritative ONLY on inactive gens (cross-checked against det_signs).
//   r            — coin rank AFTER quotienting the in-group direction set D (F6 shows the drop).
//   kappa        — = kernel_logicals.size(); κ>0 ⇒ sector split is magic-Born, not fair coins.
//   det_signs    — n_gens-bit deterministic sign BASE POINT (Phase-A bug #2): the joint coset
//                  representative solving every deterministic-combination constraint C_det, NOT the
//                  per-generator Tier-1 formula (which is valid only on inactive gens). Consumer:
//                  Task 9 sampler / Task 10 memo. MEMO SEPARABILITY: det_signs = base0 ⊕ plane1,
//                  where base0 depends only on (a, cz)+group and plane1 = prefix anticommutation is
//                  the one prefix-dependent part; the sector support base0⊕span(coin_masks) is
//                  prefix-independent, so a shot only re-XORs the prefix plane (see twirl_law.cpp).
//   coin_masks   — r rows, each an n_gens-bit σ-flip mask = rowspace(Π), Π_ij = v_i·x_j. Each coin
//                  is a fair ±1 that XORs its mask into σ. rank(Π) == r is asserted loudly (this
//                  assert caught Phase-A bug #3).
//   kernel_logicals — κ reduced LOGICAL reps (exact Pauli phase) of the normaliser-not-group
//                  Z-dressing directions Kz/D. Consumer: Task 9 chain-rule logical sampler.
//   kernel_masks — κ rows, each an n_gens-bit σ-flip PREIMAGE (Task-10 review contract): a
//                  generator-subset c solving Σ_i c_i·v_i = v*_j over GF(2) (v*_j = the j-th kernel
//                  direction, the SAME vector kernel_logicals[j] is reduced from), chosen ⊥ C_det
//                  (orthogonal to every deterministic-combo constraint vector — so XORing it into σ
//                  preserves the base-point/deterministic parities; it is exactly the σ-delta of the
//                  Python reference sample_shot's full chain rule). CATEGORICALLY different from
//                  coin_masks: coin_masks are Π-rows (v_i·x_j), the kernel directions have ZERO
//                  Π-image (they lie in N_z), so a kernel flip is a generator-space preimage, not a
//                  Π-row. Pairing (Task 10 twirl_collapse): σ ⊕= kernel_masks[j] iff kernel outcome
//                  j == 1 (same 0=+1/1=−1 convention as the `coins` record). {coin_masks ∪
//                  kernel_masks} are jointly independent, rank r+κ = the full C_det-homogeneous space.
//   fallback     — nf not in the diagonal class ⇒ caller takes the exact fallback (law is empty).
struct ShotLaw {
    Tier1Result tier1;
    int r = 0;
    int kappa = 0;
    std::vector<uint64_t> det_signs;                 // n_gens bits (words uint64s)
    std::vector<std::vector<uint64_t>> coin_masks;   // r × (n_gens-bit) sign masks
    std::vector<std::vector<uint64_t>> kernel_masks; // κ × (n_gens-bit) kernel σ-flip preimages
    std::vector<uint8_t> kernel_base;                // κ base-sign bits (see below)
    std::vector<Pauli> kernel_logicals;              // κ reduced reps, exact phases
    // kernel_foldable — κ flags (parallel to kernel_logicals), 1 ⇒ the kernel direction folds σ.
    // V2 REACHABILITY split (controller adjudication a): a kernel logical is FOLDABLE iff its
    // dressing coordinate is reachable as a generator-subset h-product (Σ_{i∈c} col_i = α* feasible)
    // — then kernel_masks[j]/kernel_base[j] carry the V1 fold and this flag is 1. UNFOLDABLE dirs
    // (α* ∉ span of the active generator columns — e.g. PPR rot(1,X̄) where every pattern column is
    // zero) contribute NO σ weighting: kernel_masks[j] is empty, kernel_base[j]=0, flag=0; the
    // outcome is still Born-sampled into `coins` (it determines the collapsed amps) but never folds
    // σ. DIAGONAL V1 dirs are ALWAYS foldable ⇒ build_shot_law sets every flag to 1 (byte-identical
    // behaviour; the collapse fold guard is a no-op for all-foldable plans).
    std::vector<uint8_t> kernel_foldable;            // κ flags: 1 = foldable (folds σ), 0 = unfoldable
    bool fallback = false;
};
// kernel_base[j] — the base-sign bit that makes the assembled σ match the reference full chain.
// The operator identity Π_{i∈mask_j} g_i measured on R|ψ⟩ = s_j · (group) · L_j gives
// mask_j·σ = s_j ⊕ (L_j outcome); det_signs pins only the C_det parities (mask_j·det_signs is an
// unconstrained β_j), so twirl_collapse XORs mask_j iff (outcome_bit ⊕ kernel_base[j]) with
// kernel_base[j] = s_j ⊕ β_j. This keeps det_signs byte-identical to the Python reference
// build_shot_law (the 249/249 harness) while reproducing sample_shot's full-chain σ↔outcome
// correlation. Requires the kernel masks to be ORTHONORMAL under the GF(2) dot pairing
// (mask_j·mask_k = δ_jk) — true for the in-distribution κ≤1; build_shot_law falls back loudly
// otherwise (κ≥2 / even-popcount boundary), never shipping a wrong σ.
ShotLaw build_shot_law(const CertifiedGroupPlanes& G, const DiagNormalForm& nf);

// ── V3: Born-weighted OBSERVABLE channel (executable spec scripts/twirl_obs_reference.py,
// 58/58 vs the enumerated per-sector conditionals at 1e-12). For a LOGICAL observable W on
// a κ=0 diagonal plan: ⟨W⟩_σ = (−1)^{c_W·σ ⊕ ⟨P,W⟩-cancellation}·Re⟨Λ_W⟩ when the dressing
// v_W = M·x_W is reachable (∈ span{v_i} mod N_z), and EXACTLY 0 otherwise. classify runs at
// plan scope with the IDENTITY prefix (memo-separable: Λ_full = (−1)^{⟨P,O_W⟩}Λ_id, and the
// g-product parities cancel against c_W·plane1(P) inside c_W·σ, so the per-shot correction
// reduces to the single parity ⟨P,W⟩). Emission: bit_W ~ Bernoulli((1 − ⟨W⟩_σ)/2) — the
// (σ, W) joint is exact by construction (post-selected/decoded statistics exact).
struct ObsChannel {
    bool guard = false;                    // classification refused → exact-path routing
    bool reachable = false;                // false ⇒ conditional ≡ 0 (fair coin CORRECT)
    std::vector<uint64_t> mask;            // n_gens-bit preimage c_W (reachable only)
    Pauli lam;                             // exact-phased normalizer rep (identity prefix)
};
// nf must be the plan's IDENTITY-PREFIX content (a, cz); W the observable's combined
// terminal-read record operator (LOGICAL class — callers guard IN_GROUP/ANTI upstream).
ObsChannel classify_observable(const CertifiedGroupPlanes& G, const DiagNormalForm& nf,
                               const Pauli& W);

// prefix_plane1 — the n_gens-bit prefix-anticommutation plane ⟨P,g_i⟩ (spec §5 plane1), computed
// columnarly in O(|P|·n/64): XOR zcol(q) over supp_X(P), XOR xcol(q) over supp_Z(P). This is the
// ONLY prefix-dependent part of det_signs (det_signs = base0(a,cz)+group ⊕ plane1(prefix)); the
// content-keyed memo caches base0 and re-XORs this per shot. Exposed so the memo finalize and tests
// can consume the exact separation.
std::vector<uint64_t> prefix_plane1(const CertifiedGroupPlanes& G, const Pauli& prefix);

// ── Task 9: kernel chain-rule sampler + collapsed amplitudes (spec §5 step 5). ──────────────
// Rng: a source of uniform doubles in [0,1) — the SAME convention FramedSuperposition's
// measurement machinery (measure_pauli's `u`, batch_measure's `rng()`) consumes. The CALLER owns
// seeding; the reproducibility contract (same seed → identical sigma/coins/amps bytes) is a
// property of the supplied Rng being deterministic, not of twirl_collapse (which is a pure
// function of the Rng draw sequence). Draw order (reproducibility): the r fair coins first
// (one draw each), then the κ kernel chain steps (one draw each) — this is exactly the order the
// `coins` record is laid out.
using Rng = std::function<double()>;

// TwirlOutcome (spec §4): the per-shot record. `sigma` = certified-generator sign bits; `amps` =
// the collapsed magic-sector state with the residual's within-sector action applied; `coins` = the
// reproducibility record (r fair coins then κ chain outcomes, 0 ⇒ +1 / 1 ⇒ −1); `fallback` = a
// guard trip or out-of-diagonal-class law (caller must take the exact path — never silent).
// CONTRACT (mid-chain fallback): if a κ-chain guard trips (non-Hermitian rep / p±≠1) the shot is
// abandoned with fallback=true and `amps`/`coins`/`sigma` are UNDEFINED (partially-folded) — the
// caller MUST route the shot through the exact path and ignore these fields. An out-of-class law
// (law.fallback / !nf.diagonal_class) returns fallback=true with sigma=det_signs and amps=bare so
// the caller can still route exactly. Task 10: sigma = det ⊕ coins ⊕ kernel (fully assembled).
struct TwirlOutcome {
    std::vector<uint64_t> sigma;   // n_gens sign bits (words). Task 10: det ⊕ coins ⊕ kernel_masks.
    FramedSuperposition   amps;    // collapsed magic-sector state (chi' <= chi), residual applied.
    std::vector<uint8_t>  coins;   // r fair coins (0/1) then κ chain outcomes (0=+1, 1=−1).
    bool fallback = false;
    TwirlOutcome() : amps(0) {}
};

// twirl_collapse — sample one shot's syndrome + collapsed magic state for a DIAGONAL residual
// R = γ·P·C, per spec §5 step 5. Steps:
//   1. σ ← law.det_signs (the deterministic base point, Task 8).
//   2. r fair coins: draw u∈[0,1); bit = (u<0.5)?0:1; if bit, XOR coin_masks[k] into σ.
//   3. κ kernel chain: clone the bare magic sector ONCE; for j=0..κ-1: Born-measure
//      kernel_logicals[j] on the RUNNING state (framed_expectation for the guard, then measure_pauli
//      collapses + samples with the SAME draw), record the outcome. NO 2^κ table anywhere.
//   4. Apply the residual's within-sector action to the collapsed state: C (S^a, then CZ) then P
//      (X^v, Z^z), global phase dropped — the stored `amps`. (Validated against the Python oracle:
//      reading any logical column off `amps` reproduces analytic_law's per-sector conditional.)
//   5. Guards (always-on): each chain step's (p+,p−) sums to 1 within 1e-9, and every kernel rep is
//      Hermitian; a trip → fallback=true (loud, no silent-wrong).
// Task 10 σ-assembly: after the κ chain, σ ⊕= kernel_masks[j] for each outcome bit j that fired
// (== 1). kernel_masks[j] is the generator-subset preimage of the kernel direction v*_j (see
// ShotLaw), so the assembled σ equals the Python reference sample_shot's full-chain σ in
// distribution (gated on the K1 fixtures, σ-keyed). The `coins` record still lays out the r fair
// coins then the κ chain outcomes, so no information is lost.
//
// need_amps (profile-driven, 2026-07-15): the per-shot RECORD (σ, coins) is a pure function of the
// plan + the draws; `amps` is reconstructible from (plan, σ, coins) and only needs materialising
// when a consumer reads it. With need_amps=false, `amps` is UNSPECIFIED and must not be read
// (empty at κ=0 — the n-qubit state copy and residual application, ~244 µs/shot at n=298, are
// skipped entirely; at κ>0 the chain still runs on an internal working copy for the Born sampling,
// left WITHOUT the residual applied). σ, coins and fallback are identical either way — the amps
// construction never feeds back into them. Default true = prior behaviour at every call site.
TwirlOutcome twirl_collapse(const FramedSuperposition& bare, const CertifiedGroupPlanes& G,
                            const DiagNormalForm& nf, const ShotLaw& law, Rng& rng,
                            bool need_amps = true);

// (twirl_collapse_record — the CachedPlan-direct record fast path — is declared below the
// CachedPlan definition.)

// ── Task 10: content-keyed plan memo (M2 finding: fired-pattern hit-rate 6.8%, (a,cz)-content ~83%). ──
// The ShotLaw except the prefix plane depends only on the composed diagonal content (a-support, cz
// list) and the certified group. TwirlPlanCache keys a CachedPlan on that content (+ a group-identity
// token, so a plan is never served across different certified groups) and finalises per shot by
// XOR-ing the prefix plane1 (O(|P|·n/64)) into the cached base0 to recover det_signs. Everything else
// in the finalized ShotLaw (coin_masks, kernel_masks, kernel_logicals, r/κ/tier1) is prefix-free and
// copied straight from the cache. Two shots with the SAME (a,cz)+group but DIFFERENT prefixes hit the
// SAME cached plan and produce their own correct per-shot det_signs.
//
// base0 = det_signs computed with an IDENTITY prefix (the b0 base point). base0 ⊕ plane1(P) is a
// VALID base point for prefix P: it satisfies every C_det parity (C_det·(base0⊕plane1) = b0 ⊕
// C_det·plane1 = b_c(P)), so it lands on the correct affine sector coset. It may differ from
// build_shot_law(G,nf).det_signs by an element of span(coin_masks ∪ kernel_masks) — the SAME sector
// distribution (that span IS the full C_det-homogeneous space, dim r+κ), hence physically identical.
struct CachedPlan {
    int r = 0;
    int kappa = 0;
    bool fallback = false;                           // build-time fallback (e.g. κ≥2 non-orthonormal):
                                                     // MUST propagate — a cached fallback plan served
                                                     // as valid would be silent-wrong (2026-07-15 fix;
                                                     // previously finalize() hardcoded fallback=false)
    std::vector<uint64_t> base0;                     // det_signs @ identity prefix (prefix-free)
    std::vector<std::vector<uint64_t>> coin_masks;
    std::vector<std::vector<uint64_t>> kernel_masks;
    std::vector<uint8_t> kernel_base;                // κ base-sign bits (prefix-free: the prefix
                                                     // plane cancels between s_j and β_j — see .cpp)
    std::vector<uint8_t> kernel_foldable;            // κ flags (parallel to kernel_logicals): see ShotLaw
    std::vector<Pauli> kernel_logicals;
    Tier1Result tier1;                               // active mask (prefix-free); sign is NOT used here
    // Canonical content this plan was built from (hash-bucket equality verification — the cache
    // key is a 64-bit content hash; equality on these words makes a collision impossible to serve).
    std::vector<uint64_t> key_amask;                 // a-support packed words
    std::vector<std::pair<int, int>> key_cz;         // sorted cz list
    // Channel-space compilation (Tier B, lazily filled per (plan, channel-set) by TwirlRecordMode):
    // per-coin/kernel channel-flip rows + the baseline channel bits for base0. Kept here so the
    // per-key work is done once; row width = the channel-set word count.
    mutable std::vector<uint64_t> ch_baseline;       // parity(base0 & m_c) bits over channels
    mutable std::vector<std::vector<uint64_t>> ch_coin_rows;    // r rows
    mutable std::vector<std::vector<uint64_t>> ch_kernel_rows;  // κ rows
    mutable uint64_t ch_token = 0;                   // channel-set token the rows were built for
    // V3 Born-weighted observable channel (single logical observable; lazily classified per
    // plan by the record consumer — identity-prefix contract, see classify_observable):
    //   ⟨W⟩_σ = (−1)^{obs_par_base ⊕ Σ coins·obs_par_coin[k] ⊕ ⟨P,W⟩} · obs_m
    // (obs_guard ⇒ exact-path routing for the shot; obs_m = 0 when unreachable — fair coin
    // is then CORRECT). Consumer-owned like ch_* / fold_p1.
    mutable uint8_t obs_ready = 0;
    mutable uint8_t obs_guard = 0;
    mutable uint8_t obs_par_base = 0;
    mutable std::vector<uint8_t> obs_par_coin;       // r bits: parity(c_W & coin_masks[k])
    mutable double obs_m = 0.0;                      // Re⟨Λ_W⟩ on the bare state
    // Finalize to a per-shot ShotLaw: det_signs = base0 ⊕ prefix_plane1(G, nf.prefix).
    ShotLaw finalize(const CertifiedGroupPlanes& G, const DiagNormalForm& nf) const;
};

// Record fast path (2026-07-15 hot loop): σ/coins written straight from the CachedPlan into a
// caller-reused TwirlOutcome — no ShotLaw materialisation, no amps (record-only; amps are
// reconstructible from (plan, σ, coins)). Behaviour-equivalent to
// plan.finalize(...)+twirl_collapse(..., need_amps=false): same draw order, same fold rules,
// same guards. σ = base0 ⊕ prefix-plane (XORed in place, no temporary) ⊕ coins ⊕ kernel folds.
void twirl_collapse_record(const FramedSuperposition& bare, const CertifiedGroupPlanes& G,
                           const DiagNormalForm& nf, const CachedPlan& plan, Rng& rng,
                           TwirlOutcome& out);

class TwirlPlanCache {
  public:
    // enable_disk (default OFF for Task 10; Task 11/12 may enable): reserved hook for the optional
    // plan_cache blob layer. Env-free — the flag is a constructor argument, nothing is read from the
    // environment. When OFF the cache is a pure in-memory hash map.
    explicit TwirlPlanCache(bool enable_disk = false) : disk_(enable_disk) {}

    // Get (or build + cache) the CachedPlan for (nf content, G). Keyed on a 64-bit FNV over
    // (group token, packed a-support words, sorted cz list) with FULL content-equality verification
    // inside the bucket (hash collisions cannot serve a wrong plan — never-silent-wrong); NEVER
    // serves a plan across a different certified group (the token is in the hashed content and the
    // amask/cz words are compared exactly).
    const CachedPlan& get_or_build(const CertifiedGroupPlanes& G, const DiagNormalForm& nf);

    // Hot-loop lookup (2026-07-15): key directly on the CALLER-maintained packed content — no
    // DiagNormalForm materialisation on a hit. Returns nullptr on miss (caller then materialises nf
    // and calls get_or_build, which recounts nothing here — find() counts a hit only on success).
    const CachedPlan* find(const CertifiedGroupPlanes& G, const std::vector<uint64_t>& amask,
                           const std::vector<std::pair<int, int>>& cz);

    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }
    size_t size() const { return n_plans_; }

    // Disk layer (2026-07-15): serialize/load every cached plan via the plan_cache blob framing
    // (magic "TWPL", version 1, FNV-integrity, corrupt/skew → ok=false → caller rebuilds silently).
    // The store is NOISE-BLIND and prefix-free, so one file serves whole p-sweeps and reruns; the
    // group token is stored and must match on load (a file from a different certified group loads
    // nothing). Channel-space rows (Tier B) are NOT serialized — rebuilt lazily per channel set.
    bool load_file(const std::string& path, uint64_t group_token);
    bool save_file(const std::string& path, uint64_t group_token) const;

    // Visit every cached plan (e.g. to pre-build channel rows for a channel set at startup, off the
    // per-shot clock — rows are channel-set-dependent and not serialized).
    void for_each(const std::function<void(CachedPlan&)>& fn) {
        for (auto& kv : store_) for (auto& p : kv.second) fn(*p);
    }

  private:
    bool disk_ = false;
    size_t hits_ = 0, misses_ = 0, n_plans_ = 0;
    // hash -> plans whose content hashed there (bucket almost always size 1)
    std::unordered_map<uint64_t, std::vector<std::unique_ptr<CachedPlan>>> store_;
    std::vector<uint64_t> amask_scratch_;            // per-call packed a-support (no realloc)
};

}  // namespace qeccore
