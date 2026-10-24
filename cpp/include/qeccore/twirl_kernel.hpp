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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/clifford_op.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/keyed_rng.hpp"
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
//   kernel_logicals — κ reduced LOGICAL reps (exact Pauli phase) of the normaliser-not-group
//                  Z-dressing directions Kz/D. Consumer: Task 9 chain-rule logical sampler.
//   fallback     — nf not in the diagonal class ⇒ caller takes the exact fallback (law is empty).
struct ShotLaw {
    Tier1Result tier1;
    int r = 0;
    int kappa = 0;
    std::vector<uint64_t> det_signs;                 // n_gens bits (words uint64s)
    std::vector<Pauli> kernel_logicals;              // κ reduced reps, exact phases
    bool fallback = false;
};
ShotLaw build_shot_law(const CertifiedGroupPlanes& G, const DiagNormalForm& nf);
// (The dual-basis solver's test seam lives in cpp/src/twirl_law_internal.hpp, not here.)

// The PPR plan's Born-observable channel (classify_observable_ppr, twirl_ppr.hpp; Q2: the PPR Born
// observable stays on obs_m / OBS until J-b1 — a diagonal plan's observable is a record row).
struct ObsChannel {
    bool guard = false;                    // classification refused → exact-path routing
    bool reachable = false;                // false ⇒ conditional ≡ 0 (fair coin CORRECT)
    std::vector<uint64_t> mask;            // n_gens-bit preimage c_W (reachable only)
    Pauli lam;                             // exact-phased normalizer rep (identity prefix)
};

// prefix_plane1 — the n_gens-bit prefix-anticommutation plane ⟨P,g_i⟩ (spec §5 plane1), computed
// columnarly in O(|P|·n/64): XOR zcol(q) over supp_X(P), XOR xcol(q) over supp_Z(P). This is the
// ONLY prefix-dependent part of det_signs (det_signs = base0(a,cz)+group ⊕ plane1(prefix)); the
// content-keyed memo caches base0 and re-XORs this per shot. Exposed so the memo finalize and tests
// can consume the exact separation.
std::vector<uint64_t> prefix_plane1(const CertifiedGroupPlanes& G, const Pauli& prefix);
// The same plane into a caller-owned buffer (resized to G.words; reused across shots: no allocation).
void prefix_plane1_into(const CertifiedGroupPlanes& G, const Pauli& prefix, std::vector<uint64_t>& plane);

// ── S2a: the Born layer of a diagonal plan — compact directions + the Born trie ─────────────────
// FreeDir (owner decision 8): Born direction j of a plan is stored ONLY in FREE-REGISTER form,
// Q_j = U†Λ_jU of the bare frame (U, eps, free) restricted to the bare's k free coordinates:
//   a[d], b[d] = Q.xbit(free[d]), Q.zbit(free[d])   (d = 0..k−1, bare free order)
//   phase      = Q.phase & 3
//   par        = Σ_q Q.zbit(q)·eps[q] (mod 2) over ALL wires of the bare frame
// Q has no X off the free set (Λ_j is LOGICAL: it commutes with every certified row), so this is
// exactly the data measure_pauli's case (b) decodes per step — y = a, g = (−1)^b, c = i^phase·
// (−1)^par on the plan's fixed free register — decoded ONCE per plan instead of once per shot and
// step. `hermitian` is today's per-step Hermiticity
// guard of the n-qubit rep, evaluated once at conversion. The n-qubit Λ_j is DERIVED on demand
// (derive_kernel_logical) for exports only.
struct FreeDir {
    std::vector<uint8_t> a, b;
    uint8_t phase = 0, par = 0, hermitian = 1;
};
// The compact form of an n-qubit kernel logical on the bare frame (throws std::logic_error on an
// X outside the bare's free set: an engine bug, a kernel logical must commute with every certified
// row).
FreeDir free_dir_of(const FramedSuperposition& bare, const Pauli& L);
// The n-qubit Λ derived from its compact form: U·(L̂ ⊗ I)·U† with L̂ = i^phase·(−1)^par·X^a Z^b on
// the free coordinates, reduced by the SAME G.reduce the builder applies (twirl_law.cpp) — so it is
// the builder's coset representative, exact phase included (G.reduce is canonical on the signed
// coset: L̂ ⊗ I and the stored rep differ by a product of SIGNED certified rows).
// EXPORT ONLY: a consumer must never PROJECT a state onto an exported Λ — it is a coset
// representative of a Born DIRECTION, not a record operator (design §5.1 S2a, A2).
Pauli derive_kernel_logical(const FramedSuperposition& bare, const CertifiedGroupPlanes& G,
                            const FreeDir& d);

// One node of a plan's Born trie, keyed by the z-prefix (design §2.2 Cor. 1.3, ndrec §1.2):
// p0 = p(z_j = 0 | z_<j) — bit-for-bit measure_pauli's p_+ for direction j on the bare free
// register projected through directions 0..j−1 at z_<j (NaN ⇔ not yet evaluated); kid[z] = the
// child node index (0 ⇔ unvisited — node 0 is the root, never a child). Allocated on visit.
struct BornNode {
    double p0;
    uint32_t kid[2];
};


// The ShotLaw except the prefix plane depends only on the composed diagonal content (a-support, cz
// list) and the certified group. A CachedPlan finalises per shot by XOR-ing the prefix plane1
// (O(|P|·n/64)) into the cached base0 to recover det_signs; every other law field is prefix-free.
//
// base0 = det_signs computed with an IDENTITY prefix (the b0 base point). base0 ⊕ plane1(P) is a
// VALID base point for prefix P: it satisfies every C_det parity (C_det·(base0⊕plane1) = b0 ⊕
// C_det·plane1 = b_c(P)), so it lands on the correct affine sector coset. It may differ from
// build_shot_law(G,nf).det_signs by an element of span(coin_masks ∪ kernel_masks) — the SAME sector
// distribution (that span IS the full C_det-homogeneous space, dim r+κ), hence physically identical.
//
// IDENTITY (S0(c)). The plan KEY is ONE bitset of KW = NW + PW words stored ONCE, inline at the end
// of the header allocation: words [0, NW) = the a-support (a & 1) over the n deferred wires, words
// [NW, KW) = the CZ layer over the cache's sorted pair UNIVERSE (bit i = universe pair i). A header
// is created once per distinct key, receives a dense content id (1, 2, … in creation order), and is
// NEVER freed or moved for the cache's lifetime — so a header pointer, and its id, IS content
// identity for every holder (the flat index, the Born memo key, the barrier sink's per-buffer
// intern, the current shot's plan). Only the PAYLOAD is evictable: the law vectors (base0, coin /
// kernel masks, kernel reps, tier1), the channel rows and the observable data. Eviction happens
// only BETWEEN shots (TwirlRecordSampler's shot boundary), never of a plan the current shot holds;
// a later shot rebuilds the payload into fresh buffers from the same key (exactness-neutral: the
// build is a pure function of the key and the group).
struct RecTable;

struct CachedPlan {
    // ── HEADER (permanent) ──
    uint32_t id = 0;                                 // dense content id (≥ 1; 0 = clean residual)
    int r = 0;
    int kappa = 0;
    bool fallback = false;                           // build-time fallback (nf outside the diagonal
                                                     // class; κ is never a reason since the dual-basis
                                                     // fold, 2026-10-12): MUST propagate — a cached
                                                     // fallback plan served as valid would be
                                                     // silent-wrong (2026-07-15 fix)
    bool built = false;                              // r/kappa/fallback valid (a payload was built once)
    mutable bool resident = false;                   // payload present
    mutable size_t payload_bytes = 0;                // accounted payload heap bytes while resident
    const uint64_t* key() const { return reinterpret_cast<const uint64_t*>(this + 1); }
    uint64_t* key_mut() { return reinterpret_cast<uint64_t*>(this + 1); }
    // ── PAYLOAD (evictable) ──
    // S2b: the plan's RECORD TABLE (qeccore/record_table.hpp; the canonical record-order basis over
    // the sampler's whole record family) and its Born trie over the table's κ_rec directions. Built
    // lazily by the sampler (it owns the record family); payload: evicted and rebuilt with the rest
    // (every field is a deterministic function of the key — Lemma 6).
    mutable std::shared_ptr<RecTable> rec;
    // r9: the plan's EMISSION block (payload; charged with it, freed with it): the record rows split for
    // the hot loop, contiguous — channel words [base | coin 0..ρ-1 | born 0..κ-1] (CW words each), then
    // the extras words in the same order (XW each). The one storage of the emission rows (RecTable keeps
    // the full rows the readers use; ensure_rec_ asserts the two agree exactly at build).
    mutable std::vector<uint64_t> emit;
    mutable int emit_rho = 0, emit_kappa = 0;
    mutable bool emit_built = false;
    mutable std::vector<BornNode> rec_trie;
};

// Heap bytes held by a plan's payload (vector capacities; the accounting unit of the budget).
size_t plan_payload_bytes(const CachedPlan& p);
// Heap bytes of one record table (the plan payload's and the readers' accounting unit).
size_t rec_table_bytes(const RecTable& T);

// GF(2)-LINEAR key hash (+ constant): h(a ⊕ b) = h(a) ⊕ h(b) ⊕ C, C = 0xcbf29ce484222325. The
// abelian fast path XORs per-alt hash tokens (h(contrib) ⊕ C) to get a shot's key hash before the
// key words are assembled. The ONE hash of the plan index.
inline uint64_t plan_key_hash(const uint64_t* k, int KW) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int w = 0; w < KW; ++w) {
        const uint64_t x = k[w];
        h ^= ((x << ((7 * w + 3) & 63)) | (x >> ((64 - ((7 * w + 3) & 63)) & 63))) ^
             ((x << ((13 * w + 29) & 63)) | (x >> ((64 - ((13 * w + 29) & 63)) & 63))) ^ x;
    }
    return h;
}

// BornTrie — the per-sampler walker of the plans' Born tries (S2a: replaces the κ chain).
// A shot's κ Born outcomes are a walk of the plan's trie: at depth j, z_j = (u_j < p0) ? 0 : 1 with
// u_j = born.u(j) (the shot's keyed BORN word j) — the SAME decision measure_pauli made on the
// chain's running state wherever the chain stayed in case (b), since p0 is computed by the identical
// operation sequence on the same amplitudes. A missing node is evaluated on the FREE register only
// (never the n-qubit bare state): the working state ψ — ONE reusable buffer per sampler — is the
// bare free register, ALL k coordinates kept (never compacted), projected through directions 0..j−1
// at the shot's z_<j by the amplitude maps (1 ± Λ̂_i)/2, carried forward within the shot (a later
// miss of the same shot projects only the directions not yet applied: identical operations, so an
// incremental node equals its from-scratch definition bit for bit) and discarded at the shot's end.
// Nodes store only {p0, kid[2]}; no state is ever stored per node.
class BornTrie {
  public:
    explicit BornTrie(const FramedSuperposition& bare);
    // One shot's walk over (directions, trie) of `branch` (0 until J-b1; any other value is refused
    // by name) — S2b: the record table's canonical directions and the plan's record trie. Appends
    // the κ_rec outcome bits (0 ⇔ ε = +1) to `bits`. Returns −1 on success, else the index j of a
    // non-Hermitian direction (an engine bug for the caller). Grows the trie on visit; the caller
    // re-accounts the payload when nodes() moved.
    int walk(const std::vector<FreeDir>& lam, std::vector<BornNode>& trie, uint32_t branch,
             const krng::Stream& born, std::vector<uint8_t>& bits);
    // The node definition, evaluated FROM SCRATCH (bare free register, project 0..j−1 at z[0..j),
    // then p_+ of direction j). Test oracle for the incremental path (bit equality).
    double p0_from_scratch(const std::vector<FreeDir>& lam, const uint8_t* z, int j);
    // Exact counters (record_report()["born_trie"]): walks (shots), steps (Σ κ over walks; = κ per
    // shot exactly), nodes (allocated), node_evals, passes (amplitude-kernel passes actually run:
    // projections + evaluations), chi_j (Σ over passes of the support ψ had entering the pass — the
    // entries actually touched; a projection can grow the support up to its shift-union, so this is
    // the support REACHED, not the bare χ), max_support (largest support reached), max_depth
    // (deepest evaluated node + 1). S8-4: a root served by the ROOT memo (below) runs no pass, so node_evals,
    // passes, chi_j and max_support count only the evaluations actually run, not memo-served roots
    // (root_memo_hits counts those).
    long walks = 0, steps = 0, nodes = 0, node_evals = 0, chi_j = 0, passes = 0, max_support = 0;
    int max_depth = 0;
    // Decision 7 (large-χ INFO event): when armed, the NEXT node evaluation is wall-timed into
    // timed_eval_seconds and the arm is cleared (the sampler arms it once per run()). S8-4: a root served by the
    // ROOT memo is not an evaluation and leaves the arm set, so the timed node may be a deeper one (diagnostic only;
    // no output depends on it).
    bool time_next_eval = false;
    double timed_eval_seconds = 0.0;
    // S8-4 (plan r2.7): the ROOT memo. A root node (j = 0) is p_+ of its direction on the freshly reset bare free
    // register, a pure function of the direction's content (a, b, phase & 3, par & 1) on THIS trie's bare; the memo
    // maps that content to the root p0 (the same operations, the same double). Root only: a deeper node depends on
    // the projections above it. One memo per BornTrie (per bare: the sampler's own and each carried branch's trie,
    // rebuilt with the trie on a D6 re-supply). Capped at kRootMemoCap entries (cleared on overflow); the sampler
    // clears it under payload-budget pressure before evicting plans. root_memo_verify (the selfcheck window):
    // every hit is recomputed and compared bit for bit (an engine-bug throw).
    static constexpr size_t kRootMemoCap = 4096;
    long root_memo_hits = 0;
    bool root_memo_verify = false;
    size_t root_memo_bytes() const { return root_memo_bytes_; }   // charged: per entry key + double + node overhead
    void root_memo_clear() { root_memo_.clear(); root_memo_bytes_ = 0; }

  private:
    void reset_();                                   // ψ ← the bare free register
    void project_(const FreeDir& d, int eps, bool check_weight = false);   // (1 + εΛ̂)/2, normalized
    double p_plus_(const FreeDir& d);                // (1 + Re<ψ|Λ̂ψ>/<ψ|ψ>)/2
    void decode_(const FreeDir& d);                  // g, c of d (y = d.a)
    const FramedSuperposition& bare_;
    std::unique_ptr<Amplitudes> psi_, beta_;         // ψ always carries all k bare free coordinates
    std::vector<cd> g_;
    cd c_;
    AmpKey y_ = 0;                                   // the packed shift a of the decoded direction
  public:
    AmpIndex idx_;                                   // S2k: the walk's key-lookup scratch (owned here)
  private:
    int psi_depth_ = -1;                             // directions applied to ψ (−1: ψ stale)
#ifndef XTIM_MUTATE_BM_BRANCH
    std::unordered_map<std::string, double> root_memo_;
#else
    static inline std::unordered_map<std::string, double> root_memo_;   // mutation build only (MUT-BM-BRANCH): one memo
#endif                                                                   // shared by every trie (every bare)
    size_t root_memo_bytes_ = 0;
    static std::string root_key_(const FreeDir& d);
};

namespace twirl_law_detail { struct ShotLawCore; }   // J-b1 (fixplan-coldstart): the parked core (twirl_law_internal.hpp)

class TwirlPlanCache {
  public:
    TwirlPlanCache() = default;
    ~TwirlPlanCache();
    TwirlPlanCache(const TwirlPlanCache&) = delete;
    TwirlPlanCache& operator=(const TwirlPlanCache&) = delete;

    // S8-2a (plan r4.1 M3): the binding's one-thread-at-a-time flag for this cache — try-acquired (compare_exchange)
    // by every binding entry that reads or writes it (a run, the record readers, RecordFidelity.evaluate, the test
    // API's cache handle); a failed acquire refuses by name (XT_CACHE_BUSY), nothing waits. The engine never reads it.
    std::atomic<bool> binding_busy{false};

    // The key universe: n deferred wires and the sorted CZ pair universe (KW = ceil(n/64) +
    // ceil(|pairs|/64)) plus the certified group's identity token. Set once, by the sampler that owns
    // this cache (caches are per sampler: make() / the ctor build fresh ones and no path shares one);
    // a second call must present the identical universe — anything else throws (a plan is never
    // served across groups/universes).
    void set_universe(int n, const std::vector<std::pair<int, int>>& pairs, uint64_t group_token);
    int key_words() const { return KW_; }

    // Hot-path lookup on the flat (hash, header) index: nullptr on miss. Counts nothing.
    struct Slot { uint64_t h; CachedPlan* p; };
    const CachedPlan* find(const uint64_t* key, uint64_t h) const {
        size_t s = h & mask_;
        for (;;) {
            const Slot& sl = slots_[s];
            if (!sl.p) return nullptr;
            if (sl.h == h) {
                const uint64_t* k2 = sl.p->key();
                int w = 0;
                while (w < KW_ && k2[w] == key[w]) ++w;
                if (w == KW_) return sl.p;
            }
            s = (s + 1) & mask_;
        }
    }
    const void* slot_addr(uint64_t h) const { return &slots_[h & mask_]; }   // prefetch target

    // True iff `key` encodes exactly the content (nf.a & 1, nf.cz) — the full stored-key
    // equality a lookup on a closed-form key is verified against.
    bool key_encodes(const uint64_t* key, const DiagNormalForm& nf) const;

    // Get (or create + build) the plan for `key` (h = plan_key_hash(key)); `nf` is the shot's
    // canonical normal form. key_encodes(key, nf) — the check on the caller's key DERIVATION — runs
    // when a header is created and whenever `verify` is set (the caller's self-check window);
    // nullptr on a mismatch (engine bug). Collision safety does not depend on it: find() compares
    // the full stored key on every probe. A missing header is created (a MISS: the law is built); a
    // header whose payload was evicted (or that was loaded keys-only from TWPL) gets a fresh payload
    // built from nf with an identity prefix — the identical build (a PAYLOAD BUILD, not a miss).
    const CachedPlan* get_or_build(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                   const uint64_t* key, uint64_t h, const DiagNormalForm& nf,
                                   bool verify);

    // Decode a header's key (the a-support bit of wire q; the CZ pairs in universe order).
    bool key_a(const CachedPlan& p, int q) const { return (p.key()[q >> 6] >> (q & 63)) & 1ULL; }
    void key_cz(const CachedPlan& p, std::vector<std::pair<int, int>>& out) const;
    int n_wires() const { return n_; }

    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }
    size_t size() const { return headers_.size(); }
    const CachedPlan& header(size_t i) const { return *headers_[i]; }   // id − 1 → header
    // S0(a): misses (= law builds of NEW content) split by the built plan's class; sum = misses().
    size_t misses_kappa0() const { return misses_k0_; }
    size_t misses_kappa_pos() const { return misses_kpos_; }
    size_t misses_fallback() const { return misses_fb_; }

    // ── Payload memory (D6) ──
    size_t resident_bytes() const { return resident_bytes_; }
    size_t header_bytes() const;                     // permanent: headers + inline keys + index + FIFO
    size_t payload_builds() const { return payload_builds_; }   // rebuilds on an existing header
    size_t evictions() const { return evictions_; }
    // Re-account a plan's payload after a lazy fill (channel rows, observable data).
    void reaccount(const CachedPlan& p) {
        const size_t b = plan_payload_bytes(p);
        resident_bytes_ = resident_bytes_ - p.payload_bytes + b;
        p.payload_bytes = b;
    }
    // Evict the OLDEST resident payload (FIFO of installs). False when nothing is resident.
    // Only ever called between shots (TwirlRecordSampler's shot boundary).
    bool evict_oldest();
    // ── J-b1 P11a: the kind-2 / route-1 header ARENA (FAM caches only; defined in twirl_sampler_fam.cpp) ──
    // The permanent headers of a family sampler's (G) plans (kind 2, and kind 0 outside the pair universe:
    // route 1) are OWNED by this cache, outside headers_ / Slot / find(): one allocation per header with its
    // variable-length key (tag, reduced term set S) inline after it, an own index by (tag, key), creation order
    // kept for TWPL v6. Their payloads join this cache's install FIFO (install_arena_payload), so the budget,
    // reaccount() and evict_oldest() treat them exactly like the cache's own; the cache owns header and FIFO
    // pointer alike, so no FIFO pointer can outlive its header (replaces the sampler-owned header +
    // adopt_payload, deleted).
    enum : uint8_t { kArenaRoute1 = 1, kArenaKind2 = 2 };   // TWPL v6 class tags (0 = a diagonal header)
    CachedPlan* arena_header(uint8_t tag, const uint64_t* key, int words);   // find or create (cold)
    void install_arena_payload(CachedPlan& p);       // throws std::logic_error on a header this arena does not own
    size_t arena_size() const;
    // TWPL v6: the term-set token the file
    // must carry (digest of the merged pushed-axis list + term rows + ker Φ; 0 = no term view). A mismatch loads
    // nothing (the sampler's INFO disk event reports absent/rebuild).
    void set_term_token(uint64_t t) { term_token_ = t; }
    uint64_t term_token() const { return term_token_; }

    // ── J-b1 (fixplan-coldstart (A)): core mode for a FAM sampler's cache ──
    // build_payload_ runs build_shot_law_core ONCE (the header r, κ, fallback; J-b2 J7b: there is no σ law)
    // and parks {plan id, core, law} in ONE slot; the FAM sampler's
    // lazy record-space-B table build consumes it (parked_core) on the same miss — eviction happens only between
    // shots and clears the slot. A rebuild of an evicted payload checks the caller's nf against the stored key.
    // The parked core of plan p (nullptr unless the slot holds p's build); law_out receives its law.
    const twirl_law_detail::ShotLawCore* parked_core(const CachedPlan& p, const ShotLaw** law_out) const;
    size_t core_builds() const { return core_builds_; }   // core-mode law builds (one per payload build)

    // Disk layer (J-b1 P11a; J-b2 J7b: v6 only — a v5 file loads nothing, the caller rebuilds): one class tag per
    // plan, the diagonal keys as v5 and the arena's (tag, S) keys, the term-set token in the header; a v5 file
    // loads nothing there). TWPL v5 = KEYS ONLY (magic "TWPL", format version + ENGINE_VERSION, FNV
    // integrity; corrupt/skew → ok=false → caller rebuilds silently). Each key is persisted in
    // universe-free form (a-support words, sorted CZ pair list) and re-encoded on load; a key with
    // a pair outside this universe is unreachable here and skipped. Loaded headers carry no
    // payload: it is built on first use (the identical build). The group token is stored and
    // must match on load. Requires set_universe().
    bool load_file(const std::string& path, uint64_t group_token);
    bool save_file(const std::string& path, uint64_t group_token) const;

  private:
    struct HeaderDeleter { void operator()(CachedPlan* p) const; };
    CachedPlan* new_header_(const uint64_t* key, uint64_t h);
    void index_insert_(CachedPlan* p, uint64_t h);
    void build_payload_(CachedPlan& p, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                        const DiagNormalForm& nf);
    int n_ = -1, NW_ = 0, PW_ = 0, KW_ = 0;
    uint64_t token_ = 0;
    std::vector<std::pair<int, int>> pairs_;          // the sorted pair universe
    std::unordered_map<uint64_t, int> pidx_;          // pair → universe index (TWPL load only)
    std::vector<std::unique_ptr<CachedPlan, HeaderDeleter>> headers_;   // id-1 → header
    std::vector<Slot> slots_ = std::vector<Slot>(1, Slot{0, nullptr});  // never probe an empty table
    size_t mask_ = 0, used_slots_ = 0;
    std::vector<CachedPlan*> fifo_;                   // resident payloads, install order
    size_t fifo_head_ = 0;
    size_t hits_ = 0, misses_ = 0;
    size_t misses_k0_ = 0, misses_kpos_ = 0, misses_fb_ = 0;
    size_t resident_bytes_ = 0, payload_builds_ = 0, evictions_ = 0;
    // J-b1 (fixplan-coldstart): appended last (no existing member moves).
    size_t core_builds_ = 0;
    struct CoreSlot;
    CoreSlot* core_slot_ = nullptr;                   // owned (deleted by the out-of-line destructor)
    // J-b1 P11a: appended last (no existing member moves). The arena and its TWPL v6 I/O are defined in
    // twirl_sampler_fam.cpp (FAM only); the out-of-line destructor / header_bytes / save / load reach them.
    struct K2Arena;
    K2Arena* k2_ = nullptr;                           // owned (arena_free_ in the out-of-line destructor)
    uint64_t term_token_ = 0;
    static void arena_free_(K2Arena* a);
    size_t arena_bytes_() const;
    bool save_file_v6_(const std::string& path, uint64_t group_token) const;
    bool load_file_v6_(const std::string& path, uint64_t group_token);
};

}  // namespace qeccore
