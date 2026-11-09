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
#include "qeccore/free_dir.hpp"
#include "qeccore/plan_arena.hpp"

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
// In place (hot path, 2026-07-15): the no-touch identity case — the overwhelming majority —
// returns false having done NOTHING (no copy of a/cz/prefix). Returns true iff nf was rewritten.
// (bloat-cut B6: the by-value canonicalize_mod_stabilizers had no caller and is deleted.)
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
    long reduces = 0;                                // lean L4 (diagnostic): G.reduce calls on the build's production
                                                     // path (the κ kernel reps; the window's teeth are not counted)
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

// prefix_plane1_into — the n_gens-bit prefix-anticommutation plane ⟨P,g_i⟩ (spec §5 plane1), computed
// columnarly in O(|P|·n/64): XOR zcol(q) over supp_X(P), XOR xcol(q) over supp_Z(P). This is the
// ONLY prefix-dependent part of det_signs (det_signs = base0(a,cz)+group ⊕ plane1(prefix)); the
// content-keyed memo caches base0 and re-XORs this per shot. Into a caller-owned buffer (resized to
// G.words; reused across shots: no allocation).
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

// threads T2: merge a lane's copy-on-write trie `src` (a copy of `dst` that the lane grew) into `dst` by path: every
// node `src` allocated or evaluated is allocated / evaluated in `dst` (a node's value is a pure function of its
// z-prefix). A node evaluated in both must hold the same double bit for bit, else std::logic_error (engine bug).
void born_trie_merge(std::vector<BornNode>& dst, const std::vector<BornNode>& src);

// ── plancache-design (owner-signed rev 4, 2026-11-01): the TWO-LEVEL cache ────────────────────────────────────────
// A PlanClass is ONE record table + emission block + Born trie per CLASS key (a ∧ U ∧ ¬prod, Π(B)) — plan §0 T3:
// every key of a class has the same record law on the bare state, so one table serves them all. Its identity (dense
// class id, the inline class key) is permanent; its payload is evictable. A CachedPlan header (one per FULL key,
// permanent, id = content identity as before: the flat index, the Born memo key, the barrier intern, the record's
// content handle all keep the full key) links to its class and MIRRORS the per-shot fields the hot loop reads
// (resident, emit_built, emit*, the emission block pointer), copied at link and refreshed by the owning cache when the
// class is built, attached or evicted (sync_headers_: the class's intrusive header list — written by the owning cache
// only, never by a lane reading a home class). The per-shot loads are today's; κ > 0 walks reach the class through
// one dependent load (plan §1, pre-declared ≤ +3 Ir/shot on t_fast / t_retention).
class TwirlPlanCache;
struct PlanClassifier;
struct CachedPlan;

struct PlanClass {
    uint32_t id = 0;                                 // dense class id (≥ 1; arena classes: the arena's own series)
    int r = 0;
    int kappa = 0;
    bool fallback = false;
    bool built = false;                              // r / kappa / fallback valid (a payload was built once)
    bool arena = false;                              // a private class of an arena header (kind 2 / route 1): no class key, not indexed
    mutable bool resident = false;                   // payload present
    mutable bool emit_built = false;
    mutable int emit_rho = 0, emit_kappa = 0;
    mutable size_t payload_bytes = 0;                // accounted payload heap bytes while resident
    // ── PAYLOAD (evictable) ── built from the CANONICAL representative (class_nf): a pure function of the class key.
    // §3 B1: ONE blob — the record table (the readers' one table per class) and the emission block [base | coin
    // 0..ρ-1 | born 0..κ-1] (CW words each) then the extras words (XW each) — in the owning cache's arena (§4).
    mutable RecHandle rec;
    mutable std::vector<BornNode> rec_trie;          // the class's Born trie (shared by every header of the class)
    CachedPlan* headers = nullptr;                   // the headers linked here (intrusive: CachedPlan::next_in_class)
    const TwirlPlanCache* owner = nullptr;           // the cache holding this class (a lane overlay, or its home)
    uint8_t mark = 0;                                // retention (plan §2 R2): the second-chance hit mark (home classes)
    PlanClass* twin = nullptr;                       // adopt_from scratch: a lane class's home class
    uint64_t first_shot = 0;                         // review M1: the cumulative index of the shot that first needed
                                                     // the class (the barrier orders a super-block's new classes by it:
                                                     // class ids are the serial first-seen order at every T)
    // the class key (KW words, diagonal classes): inline after the object, or (speed fix 5) BORROWED — the key of the
    // permanent header that created it when that key IS the class key (its own representative: stored once; every
    // class of a Π = id circuit whose keys strip nothing); arena classes have none
    const uint64_t* kp = nullptr;
    const uint64_t* key() const { return kp; }
    uint64_t* key_mut() { return reinterpret_cast<uint64_t*>(this + 1); }   // (the inline storage)
};

struct CachedPlan {
    // ── HEADER (permanent) ──
    uint32_t id = 0;                                 // dense content id (≥ 1; 0 = clean residual)
    int r = 0;                                       // the CLASS representative's law (plan §1 moved fields)
    int kappa = 0;
    bool fallback = false;                           // build-time fallback (nf outside the diagonal class): MUST
                                                     // propagate — a cached fallback plan served as valid would be
                                                     // silent-wrong (2026-07-15 fix)
    bool built = false;                              // r/kappa/fallback valid (the class was built once)
    // ── MIRROR of the class (refreshed by the owning cache; the hot loop's loads) ──
    mutable bool resident = false;
    mutable bool emit_built = false;
    mutable int emit_rho = 0, emit_kappa = 0;
    mutable const uint64_t* emit_p = nullptr;        // cls->emit.data() while resident and built
    PlanClass* cls = nullptr;                        // the class (table, trie); never null once linked
    CachedPlan* next_in_class = nullptr;             // the class's header list
    const uint64_t* key() const { return reinterpret_cast<const uint64_t*>(this + 1); }
    uint64_t* key_mut() { return reinterpret_cast<uint64_t*>(this + 1); }
    void mirror() const {
        resident = cls->resident; emit_built = cls->emit_built;
        emit_rho = cls->emit_rho; emit_kappa = cls->emit_kappa;
        emit_p = cls->rec ? cls->rec->emit() : nullptr;
    }
};

// Heap bytes held by a class's payload (vector capacities; the accounting unit of the budget).
size_t plan_payload_bytes(const PlanClass& c);
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
    const FramedSuperposition& bare_state() const { return bare_; }   // threads T1: a lane's fresh walker
    // One shot's walk over (directions, trie) of `branch` (0 until J-b1; any other value is refused
    // by name) — S2b: the record table's canonical directions and the plan's record trie. Appends
    // the κ_rec outcome bits (0 ⇔ ε = +1) to `bits`. Returns −1 on success, else the index j of a
    // non-Hermitian direction (an engine bug for the caller). Grows the trie on visit; the caller
    // re-accounts the payload when nodes() moved.
    int walk(LamSpan lam, std::vector<BornNode>& trie, uint32_t branch,
             const krng::Stream& born, std::vector<uint8_t>& bits);
    // threads T2: the same walk on a trie it must not change (a lane reading the shared plan cache): when every node the
    // walk visits is evaluated and allocated, appends the same κ_rec outcomes as walk() and returns -1; returns -2 with
    // `bits` restored when the walk needs a node evaluated or allocated (the caller then walks a trie it may grow);
    // j >= 0 a non-Hermitian direction (as walk). Counts walks / steps only on success.
    int walk_ro(LamSpan lam, const std::vector<BornNode>& trie, const krng::Stream& born,
                std::vector<uint8_t>& bits);
    // The node definition, evaluated FROM SCRATCH (bare free register, project 0..j−1 at z[0..j),
    // then p_+ of direction j) by the REFERENCE kernels. Test oracle for the incremental, fused path (bit equality).
    double p0_from_scratch(LamSpan lam, const uint8_t* z, int j);
    // twirl-routes T2: the conditional expectation ⟨X̂⟩ = Re⟨ψ|X̂ψ⟩/⟨ψ|ψ⟩ of a Hermitian free-register operator X on
    // the bare free register projected through directions 0..j−1 at z (the same projections as p0_from_scratch):
    // the per-shot value of an expectation column on a Born leaf. Not a sampling path (the DEM's per-shot law).
    double value_from_scratch(LamSpan lam, const uint8_t* z, int j, const FreeDir& X);
    // twirl-routes T2: the depth-first enumeration of every Born leaf of `lam` on the bare free register, + outcome
    // first, the projections carried down the walk (one saved register per branching depth: O(2^κ·χ) work, the same
    // operations as the from-scratch nodes). A direction whose conditional p_+ (or 1 − p_+) is <= eps is taken as
    // deterministic (projected, not branched). leaf(z, P(z), ⟨X⟩ on the leaf — 0 when X is null) returns false to stop.
    // `evals` (optional) accumulates node + leaf evaluations. Not a sampling path (the DEM's Born laws).
    void enumerate(LamSpan lam, const FreeDir* X, double eps,
                   const std::function<bool(uint64_t, double, double)>& leaf, long* evals = nullptr);
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
    // The node kernels (perf-engine §1.1): the production pair runs the fused amplitude kernels; the *_ref_ pair is
    // the original β-materialising chain, kept as the ORACLE (p0_from_scratch, the self-check window's recompute
    // of every evaluated node) — bit-identical by construction, held together by cpp/tests/test_born_trie.cpp.
    void project_(const FreeDirView& d, int eps, bool check_weight = false);   // (1 + εΛ̂)/2, normalized
    double p_plus_(const FreeDirView& d);            // (1 + Re<ψ|Λ̂ψ>/<ψ|ψ>)/2
    void project_ref_(const FreeDirView& d, int eps, bool check_weight = false);
    double p_plus_ref_(const FreeDirView& d);
    void decode_(const FreeDirView& d);              // g, c of d (y = d.a)
    const FramedSuperposition& bare_;
    std::unique_ptr<Amplitudes> psi_, beta_;         // ψ always carries all k bare free coordinates
    std::vector<cd> g_;
    cd c_;
    AmpKey y_ = 0;                                   // the packed shift a of the decoded direction
  public:
    AmpIndex idx_;                                   // S2k: the walk's key-lookup scratch (owned here)
  private:
    int psi_depth_ = -1;                             // directions applied to ψ (−1: ψ stale)
    long oracle_passes_ = 0;                         // passes of the in-window fused-vs-reference node check (not public)
#ifndef XTIM_MUTATE_BM_BRANCH
    std::unordered_map<std::string, double> root_memo_;
#else
    static inline std::unordered_map<std::string, double> root_memo_;   // mutation build only (MUT-BM-BRANCH): one memo
#endif                                                                   // shared by every trie (every bare)
    size_t root_memo_bytes_ = 0;
    static std::string root_key_(const FreeDirView& d);
};

namespace twirl_law_detail { struct ShotLawCore; }   // J-b1 (fixplan-coldstart): the parked core (twirl_law_internal.hpp)

class TwirlPlanCache {
  public:
    TwirlPlanCache();
    ~TwirlPlanCache();
    TwirlPlanCache(const TwirlPlanCache&) = delete;
    TwirlPlanCache& operator=(const TwirlPlanCache&) = delete;

    // S8-2a (plan r4.1 M3): the binding's one-thread-at-a-time flag for this cache — try-acquired (compare_exchange)
    // by every binding entry that reads or writes it (a run, the record readers, RecordFidelity.evaluate, the test
    // API's cache handle); a failed acquire refuses by name (XT_CACHE_BUSY), nothing waits. The engine never reads it.
    std::atomic<bool> binding_busy{false};

    // The key universe: n deferred wires (KW = ceil(n/64): the a-mask; lean D1-S3 — no CZ pair universe: the rotation loop
    // serves every circuit with rotations, a diagonal-cache key is the a-mask of a pair-free content) plus the certified
    // group's identity token. Set once, by the sampler that owns
    // this cache (caches are per sampler: make() / the ctor build fresh ones and no path shares one);
    // a second call must present the identical universe — anything else throws (a plan is never
    // served across groups/universes).
    // reserve_identity (review M1): the all-zero class key — the identity plan's class — takes class id 1 at
    // construction (a class object without a header or payload; built when first needed), so its id does not depend
    // on WHEN the identity plan is first made (the threaded call makes it at a super-block start, the serial call at
    // the first shot that needs it). A lane overlay (set_overlay_of) passes false: its classes are the home's.
    void set_universe(int n, uint64_t group_token, bool reserve_identity = true);
    int key_words() const { return KW_; }
    // plancache-design: the class key map Π (qeccore/plan_class.hpp), compiled by the sampler after set_universe and
    // shared with its lanes' overlay caches (the same object: the same classes on every lane).
    void set_classifier(std::shared_ptr<const PlanClassifier> pc);
    const PlanClassifier& classifier() const;
    // §4: this cache's arena (headers, classes and class blobs) and the root every arena of the sampler hangs off.
    PlanArena& arena() const { return *arena_; }
    const std::shared_ptr<ArenaRoot>& arena_root() const { return root_; }
    void drain_arena() { arena_->drain(); }          // the owner's thread, between shots (payload_boundary_)
    // A lane whose super-block was dropped (a failed call): after clear_plans, every block it allocated is dropped and
    // its chunks (a lent one included) go back to the home (review H1). Review L3: no reader may hold a lane table —
    // assert_lane_tables_unheld() is the loud check the caller runs BEFORE clear_plans.
    void drop_lane_arena() { arena_->reset_dropped(); if (home_) home_->arena_->splice_from(*arena_); }
    void assert_lane_tables_unheld() const;
    // H1: the barrier lends `lane` this cache's arena chunk with the largest free tail (PlanArena::lend_to).
    void lend_arena_to(TwirlPlanCache& lane) { arena_->lend_to(*lane.arena_); }
    size_t arena_chunk_bytes() const { return arena_->chunk_bytes(); }
    size_t arena_free_bytes() const { return arena_->free_bytes(); }
    size_t arena_live_bytes() const { return arena_->live_bytes(); }

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
    // the full stored key on every probe. A missing header is created (a MISS) and linked to its CLASS (the class
    // index, probed here only): an existing resident class serves it at once; an absent class is created and built
    // from its canonical representative; a class whose payload was evicted is rebuilt from it — the identical build
    // (a PAYLOAD BUILD, not a miss). A lane overlay (set_overlay_of) links to a RESIDENT home class read-only and
    // builds every other class in its own index (adopted at the barrier).
    // created: a header was made (O1); shot: the cumulative index of the shot asking (a new class's first_shot, M1).
    const CachedPlan* get_or_build(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                   const uint64_t* key, uint64_t h, const DiagNormalForm& nf,
                                   bool verify, bool* created = nullptr, uint64_t shot = 0);

    // Decode a header's key (the a-support bit of wire q).
    bool key_a(const CachedPlan& p, int q) const { return (p.key()[q >> 6] >> (q & 63)) & 1ULL; }
    int n_wires() const { return n_; }
    // ── classes ──
    // The class key of a full key (Π; `out` KW words), and of a canonical normal form (false when it has a CZ part: such
    // a content is not a diagonal-cache key and is not classed — the readers key it by its full normal form).
    void class_key_of(const uint64_t* key, uint64_t* out) const;
    bool class_key_of_nf(const DiagNormalForm& nf, std::vector<uint64_t>& out) const;
    // This cache's class of a class key (nullptr: none). A lane overlay's own index only (the home's is read through
    // find_class on the home).
    const PlanClass* find_class(const uint64_t* ckey, uint64_t h) const;
    // The canonical representative's normal form of a diagonal class (identity prefix): what its payload is built from.
    void class_nf(const PlanClass& c, DiagNormalForm& nf) const { key_nf(c.key(), nf); }
    // The normal form (identity prefix) a key of this universe encodes (a full key, or a class key).
    void key_nf(const uint64_t* key, DiagNormalForm& nf) const;
    size_t classes() const { return classes_.size(); }
    // Test API (the key-density probe): the CZ-pair bits summed over every header key / every class key.
    const PlanClass& class_at(size_t i) const { return *classes_[i]; }   // id − 1 → class (diagonal classes)

    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }
    size_t size() const { return headers_.size(); }
    const CachedPlan& header(size_t i) const { return *headers_[i]; }   // id − 1 → header
    // S0(a): misses (= headers of NEW content) split by the CLASS representative's law; sum = misses().
    size_t misses_kappa0() const { return misses_k0_; }
    size_t misses_kappa_pos() const { return misses_kpos_; }
    size_t misses_fallback() const { return misses_fb_; }

    // ── Payload memory (D6) ──
    size_t resident_bytes() const { return resident_bytes_; }
    // permanent: headers + classes + keys + indexes + FIFO — an UPPER BOUND (re-review L-1): a class whose key is
    // borrowed from its creating header is counted with its key words, and the FIFO's capacity follows the eviction
    // history, so the value is identical at every thread count only when nothing is evicted
    size_t header_bytes() const;
    size_t payload_builds() const { return payload_builds_; }   // class rebuilds (an existing class, after eviction)
    size_t class_builds() const { return class_builds_; }       // every class payload build (first builds + rebuilds)
    size_t evictions() const { return evictions_; }
    size_t second_chances() const { return second_chances_; }   // §2 R2: FIFO entries re-queued once on a hit mark
    // Re-account a class's payload after a lazy fill (the record table / emission block, a grown trie).
    void reaccount(const PlanClass& c) {
        const size_t b = plan_payload_bytes(c);
        resident_bytes_ = resident_bytes_ - c.payload_bytes + b;
        c.payload_bytes = b;
    }
    // Refresh the mirrors of every header linked to `c` (after the owning cache built / attached / evicted its payload).
    static void sync_headers(const PlanClass& c) {
        for (const CachedPlan* p = c.headers; p; p = p->next_in_class) p->mirror();
    }
    // Evict the OLDEST resident class payload (FIFO of installs) — §2 R2 second chance: a class whose hit mark is set
    // (served since the last pass) has the mark cleared and goes to the back once, so a binding budget drops the
    // singletons first. False when nothing is resident. Only ever called between shots (the shot boundary).
    bool evict_oldest();
    void mark_class(size_t i) const { classes_[i]->mark = 1; }   // a lane's marks OR-folded at the barrier (id − 1)
    // Review M1: after a super-block's adoption (and its deferred run), the classes created since `from` are ordered
    // by first_shot (stable) and renumbered id = index + 1: the serial first-seen order, identical at every T.
    void order_new_classes(size_t from);
    // ── threads T2: lane overlays ──
    // A lane's private cache (the same universe and classifier) collects the plans the lane met that its home did not
    // serve; set_overlay_of names the home: during a super-block the lane links a new header to a RESIDENT home class
    // read-only (no list write) and builds every other class in its own index. At a barrier adopt_from merges it:
    // the lane's classes in creation order (a class key the home lacks MOVES in — pointer stable, the next class id;
    // a home class without its payload takes the lane's; a duplicate's trie is merged, the rest dropped: every payload
    // is a pure function of its key), then the lane's headers in creation order (a key the home lacks MOVES in with
    // its next id and is linked to its class here; a duplicate is dropped). The lane cache is left empty (universe
    // and classifier kept), its counters folded into this cache's. Never called inside a shot.
    void set_overlay_of(const TwirlPlanCache* home);   // (its arena joins the home's root)
    void adopt_from(TwirlPlanCache& lane);
    // scaling S4: the lane's half of its adoption, run ON THE LANE at the end of its super-block work (`home` is read
    // only then: nothing writes it during a super-block): per lane header / class its key hash and home's entry of
    // that key (or null). adopt_from then hashes nothing.
    void adopt_prepare(const TwirlPlanCache& home);
    // Empty a lane cache (every header, class and payload freed; the universe kept).
    void clear_plans();
    // A lane cache's headers get ids base + 1, base + 2, … (above every id of the cache it overlays: a plan id keys
    // per-plan state, e.g. the layout-3 record's content-handle cache); its classes likewise (set_class_base).
    void set_id_base(uint32_t base) { id_base_ = base; }
    void set_class_base(uint32_t base) { class_base_ = base; }
    // ── J-b1 P11a: the kind-2 / route-1 header ARENA (FAM caches only; defined in twirl_sampler_fam.cpp) ──
    // The permanent headers of a family sampler's (G) plans (kind 2, and kind 0 outside the pair universe:
    // route 1) are OWNED by this cache, outside headers_ / Slot / find(): one allocation per header with its
    // variable-length key (tag, reduced term set S) inline after it, an own index by (tag, key). Each has a PRIVATE
    // class (plancache-design: not classed, not in the class index) whose payload joins this cache's install FIFO
    // (install_arena_payload), so the budget, reaccount() and evict_oldest() treat it exactly like the cache's own.
    enum : uint8_t { kArenaKind2 = 2 };              // the arena class tag (part of the arena key)
    CachedPlan* arena_header(uint8_t tag, const uint64_t* key, int words);   // find or create (cold)
    void install_arena_payload(CachedPlan& p);       // throws std::logic_error on a header this arena does not own
    size_t arena_size() const;

    // ── J-b1 (fixplan-coldstart (A)): core mode for a FAM sampler's cache ──
    // build_class_ runs build_shot_law_core ONCE on the class representative (the class r, κ, fallback) and parks
    // {class, core, law} in ONE slot; the FAM sampler's lazy record-space-B table build consumes it (parked_core) on
    // the same miss — eviction happens only between shots and clears the slot.
    // The parked core of p's class (nullptr unless the slot holds its build); law_out receives its law.
    const twirl_law_detail::ShotLawCore* parked_core(const PlanClass& c, const ShotLaw** law_out) const;
    size_t core_builds() const { return core_builds_; }   // core-mode law builds (one per class payload build)

  private:
    struct HeaderDeleter { void operator()(CachedPlan* p) const; };   // ~CachedPlan, then the arena block released
    struct ClassDeleter { void operator()(PlanClass* c) const; };
    CachedPlan* new_header_(const uint64_t* key, uint64_t h);
    void index_insert_(CachedPlan* p, uint64_t h);
    PlanClass* new_class_(const uint64_t* ckey, uint64_t h, const uint64_t* borrow = nullptr);
    void class_index_insert_(PlanClass* c, uint64_t h);
    PlanClass* find_class_mut_(const uint64_t* ckey, uint64_t h) const { return const_cast<PlanClass*>(find_class(ckey, h)); }
    // link p to c: the mirrors; `list` pushes p on c's header list (never for a lane reading a home class)
    void link_(CachedPlan& p, PlanClass& c, bool list);
    void build_class_(PlanClass& c, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                      bool verify);                  // verify: the caller's self-check window (the law's teeth)
    void install_(PlanClass& c);                     // resident, accounted, on the FIFO
    void release_(PlanClass& c);                     // the payload freed (evict / clear)
    int n_ = -1, NW_ = 0, KW_ = 0;
    uint64_t token_ = 0;
    std::vector<std::unique_ptr<CachedPlan, HeaderDeleter>> headers_;   // id-1 → header
    std::vector<Slot> slots_ = std::vector<Slot>(1, Slot{0, nullptr});  // never probe an empty table
    size_t mask_ = 0, used_slots_ = 0;
    std::vector<PlanClass*> fifo_;                    // resident class payloads, install order
    size_t fifo_head_ = 0;
    size_t hits_ = 0, misses_ = 0;
    size_t misses_k0_ = 0, misses_kpos_ = 0, misses_fb_ = 0;
    size_t resident_bytes_ = 0, payload_builds_ = 0, evictions_ = 0, second_chances_ = 0;
    size_t core_builds_ = 0;
    struct CoreSlot;
    CoreSlot* core_slot_ = nullptr;                   // owned (deleted by the out-of-line destructor)
    DiagNormalForm nf_scratch_;                       // speed fix 5: build_class_'s representative decode (reused)
    std::vector<uint64_t> ck_scratch_;                // speed fix 5: get_or_build's class key when KW > 64
    struct K2Arena;
    K2Arena* k2_ = nullptr;                           // owned (arena_free_ in the out-of-line destructor)
    uint32_t id_base_ = 0;
    std::vector<std::pair<uint64_t, CachedPlan*>> adopt_pre_;   // scaling S4: adopt_prepare's (hash, home header)
    // plancache-design: the class index and storage
    std::shared_ptr<const PlanClassifier> pc_;
    std::vector<std::unique_ptr<PlanClass, ClassDeleter>> classes_;   // id-1 → class (diagonal classes)
    struct CSlot { uint64_t h; PlanClass* c; };
    std::vector<CSlot> cslots_ = std::vector<CSlot>(1, CSlot{0, nullptr});
    size_t cmask_ = 0, cused_ = 0;
    uint32_t class_base_ = 0;
    const TwirlPlanCache* home_ = nullptr;            // a lane overlay: its home (read only during a super-block)
    std::vector<std::pair<uint64_t, PlanClass*>> adopt_cpre_;   // adopt_prepare's (hash, home class)
    size_t class_builds_ = 0;
    std::shared_ptr<ArenaRoot> root_;                 // §4: the sampler's root (a lane shares its home's)
    PlanArena* arena_ = nullptr;                      // this cache's arena (owned by the root)
    static void arena_free_(K2Arena* a);
    size_t arena_bytes_() const;
};

}  // namespace qeccore
