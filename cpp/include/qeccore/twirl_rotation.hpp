#pragma once
// twirl_rotation — lean D1-S1 (2026-11-05): the ROTATION MODEL of a diagonal twirl sampler, compiled once
// (lean/D1/DESIGN.md §1 items 1–3; the proofs lean/P2/REPORT.md §1–2).
//
// The deferred circuit's non-Clifford content is t commuting π/8 rotations on FIXED signed Z-strings B_j (the term
// view's end-pushed axes, term_residual.hpp). A shot with fired alternatives f_1..f_r prepares P·D(s)|φ⟩ with
// P = ⊕ P_f (the Clifford-only end frames), s = ⊕ S_f (the sign vector over the t rotations), D(s) = Π_{j∈s} rot(B_j),
// rot(B) = exp(i(π/4) B), |φ⟩ the bare state. Every record column M_c is pulled back through D(s):
//     D(s)† M_c D(s) = M_c · Π_{j ∈ s ∧ ρ_c} (i B_j),       ρ_c[j] = ⟨M_c, B_j⟩   (a fixed t-bit row per column),
// so the record law of a key s is t-bit linear algebra on the words T_c = s ∧ ρ_c against two fixed subspaces:
//   * Γ_n = ker Φ, Φ(e_j)_i = ⟨g_i, B_j⟩ over the certified generators (the "cone" = the generators with a nonzero
//     row): T ∈ Γ_n ⇔ Z_T := Π_{j∈T}(iB_j) = i^{|T|}(−1)^{η(T)} Z^{φ(T)} commutes with the group;
//   * on a basis of Γ_n the free-register direction of Z^{φ(T)} (free_dir_of: X^a Z^b with its phase; the direction's
//     (a, b) is linear on Γ_n, its phase composes by fd_mul); η(T) = the parity of the −1 axes in T;
//     Γ_z = {T ∈ Γ_n : (a, b)(T) = 0} (in ±G).
//   * a record extra (Born observable / decision / family extra) has its own plane φ(M_e) reduced against the
//     rotation planes: T0_e (real bits < t) and a residual that no rotation reaches — a VIRTUAL coordinate (bits
//     [t, t + nvirt) of the extras' words; ANTI extras) only another extra can cancel; a logical extra (residual 0)
//     carries its base direction free_dir_of(M_e · Z_{T0_e}).
// Per alternative ONE compact record (u32 words; RotationRecords): its sign words S_f, the GF(2)-linear hash token
// plan_key_hash(S_f) ⊕ C, its extras words ⟨P_f, M_e⟩ and the nonzero channel words of t(P_f).
// S1 (this stage): the model and the records are BUILT and VERIFIED at construction; nothing sampled reads them yet.
// A circuit whose term view has an axis with X content (CH / PPR: the term route's kind 2) gets no model: `kind2`
// is set and S2 dispatches every shot of it to the existing (G) route (no new refusal on any circuit served today).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/free_dir.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/term_residual.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

struct RotationModel {
    bool built = false;                 // a model was compiled (Z-only axes; every sampler of the diagonal class)
    bool kind2 = false;                 // the term view has an axis with X content: the (G) route serves the circuit
    int t = 0, TW = 0;                  // rotations, ⌈t/64⌉ words (TW = 0 when t = 0)
    int k = 0, KW = 0;                  // the bare state's free-register width, ⌈k/64⌉ words (KW ≥ 1)
    int m = 0, nch = 0;                 // record columns, channels (extras = columns [nch, m))
    int ng = 0;                         // certified generators
    // ── per column ──
    std::vector<uint64_t> rho;          // m × TW: ρ_c
    std::vector<int> scan, rank_of;     // columns in record order; column → position in scan
    std::vector<int> cols_off, cols;    // CSR per rotation j: the columns with ρ_c[j] = 1, ascending (cols_of)
    std::vector<uint64_t> eta, live;    // TW: axis j has sign −1; OR of every ρ_c
    // ── the cone (certified generators meeting some axis) and Γ_n ──
    std::vector<int> cone_gen;          // generator indices with a nonzero row, ascending
    std::vector<uint64_t> cone_row;     // |cone| × TW: Φ(e_j)_i over the cone
    int phi_rank = 0;                   // rank Φ
    std::vector<uint64_t> gn_rows;      // dim Γ_n × TW: the fully reduced RREF of ker Φ, rows by ascending pivot
    std::vector<int> gn_piv;            // per row: its pivot (lowest bit)
    std::vector<int> gn_col_to_row;     // t: bit → row (−1: not a pivot)
    // per row the compact form of Z^{φ(row)} on the bare frame's free register (free_dir_of): i^phase (−1)^par X^a Z^b,
    // Hermitian. The bare frame can map a Z-string to X free content, so the direction is (a, b, phase, par), not a
    // Z direction + a sign bit (the P2 probe's pure-Z special case: a = 0, σ = (phase >> 1) ⊕ par).
    std::vector<uint64_t> gn_a, gn_b;   // dim Γ_n × KW each
    std::vector<uint8_t> gn_phase, gn_par;   // dim Γ_n each
    int gz_dim = 0;                     // dim Γ_z = dim Γ_n − rank of the (a | b) directions
    // ── extras ──
    int nx = 0, nvirt = 0, TXW = 1;     // extras, virtual coordinates, ⌈(t + nvirt)/64⌉ words (≥ 1)
    std::vector<uint64_t> x_T0;         // nx × TXW: T0_e (real bits < t) ⊕ the virtual coordinate bits [t, t + nvirt)
    std::vector<uint8_t> x_logical;     // nx: the residual plane is zero (a logical extra; else ANTI)
    std::vector<FreeDir> x_fd;          // nx: the base direction free_dir_of(M_e · Z_{T0_e}) (valid iff x_logical)
    int anti_extras = 0;
    size_t bytes() const;
};

// The per-alternative compact records (u32 words). Record a at off[a] .. off[a + 1]:
//   [0, 2·TW)        the sign words S_f (lo, hi per word)
//   [2·TW, 2·TW + 2) the hash token plan_key_hash(S_f, TW) ⊕ 0xcbf29ce484222325 (GF(2)-linear: a shot's hash is
//                    C ⊕ the XOR of its fired tokens, as the L2 blob's htok)
//   then XR extras words (lo, hi): ⟨P_f, M_e⟩ at bit e (0 words on a circuit without extras)
//   then nz, then nz entries (w, lo, hi): channel word w of t(P_f), ascending w, value ≠ 0.
struct RotationRecords {
    int TW = 0, XR = 0, CW = 0;
    std::vector<uint32_t> rec;
    std::vector<size_t> off;            // A + 1 (the last = rec.size()); never capped at a field width (owner rule)
    size_t bytes() const { return rec.capacity() * sizeof(uint32_t) + off.capacity() * sizeof(size_t); }
    int num_alts() const { return off.empty() ? 0 : (int)off.size() - 1; }
};

// The model of a Z-only term view over the record columns `cols` (channels first: [0, nch)), the certified group
// planes and the bare state. `colx_on_wire` (n × QCW words): per wire q the record columns whose operator has X on q
// (the sampler's qcol_z_), i.e. the columns a Z on q anticommutes with — ρ_c[j] = ⊕_{q ∈ supp B_j} bit c of it.
// Throws std::invalid_argument when an axis has X content (the caller flags kind 2 instead of calling) and
// std::logic_error on an engine bug (a Γ_n element anticommuting with the group, an odd axis phase, ...).
void build_rotation_model(const TermResidual& tr, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                          const std::vector<RecColumn>& cols, int nch, const std::vector<uint64_t>& colx_on_wire,
                          const std::vector<uint64_t>& colz_on_wire, int QCW, RotationModel& out);

// ══════════════════════ lean D1-S2: the plan of a key s — the builder and the plan store ══════════════════════
// A fully reduced GF(2) echelon over W-word rows (pivot = the row's lowest bit; a pivot bit occurs in its own row only),
// with a combination word of CW words per row (which inputs the row is the sum of). Widths are per instance: no cap.
struct RotEchelon {
    int W = 0, CW = 0;
    std::vector<uint64_t> rows, comb;     // |rows| × W, |rows| × CW
    std::vector<int> piv;
    std::vector<int> col_to_row;          // W·64
    void reset(int w, int cw);
    void reduce(uint64_t* v, uint64_t* c) const;   // v (and its combination c, CW words) reduced against the rows
    int insert(const uint64_t* v, const uint64_t* c);   // a reduced, nonzero v: its pivot
    int size() const { return (int)piv.size(); }
};

// The record table of ONE key s in the rotation picture (DESIGN §1 "per new key"; the P2 probe's build_rot generalised:
// multiword A / B / T, the (a, b, phase, par) directions of the Γ_n rows composed by fd_mul, the ANTI-extra products by
// free_dir_of on the literal product). Rows = the touched channels and every extra, in record order.
struct RotBuild {
    int TW = 0, TXW = 1;                  // the key's words; the T words (+ virtual coordinates)
    int ncoin = 0, ndir = 0;              // ρ, κ
    int AW = 0, BW = 0;                   // the A / B words the rows carry (≥ the final ⌈ρ/64⌉, ⌈κ/64⌉)
    std::vector<int> cols;                // the rows' columns (record order)
    std::vector<uint8_t> pivot, beta, flip;   // per row: a pivot; the raw base bit β (pivot / introducer: 0); flip = β ⊕ ref (the INTERNAL base: channels) / β (extras)
    std::vector<uint64_t> A, B;           // per row: AW / BW words (a pivot's A = e_slot; an introducer's B = e_j)
    std::vector<int> pivot_cols;          // by slot
    std::vector<uint64_t> pivot_T;        // slot × TXW: T'_p (T_p ⊕ T0_p, virtual bits included)
    std::vector<uint8_t> pivot_sg;        // slot: the extra pivot's own sign |T0| + |T0 ∧ T_p| (real bits; 0 for a channel)
    std::vector<FreeDir> dirs;            // the canonical directions (literal products, exact phase)
    std::vector<int> intro_cols;          // the introducer column of each direction
    long anti_products = 0;               // directions computed by the explicit Pauli product (ANTI extras)
    // scratch (reused across builds)
    RotEchelon pech, dech;
    std::vector<uint64_t> Tc, Tp, r, cb, TK, comb, dc, r2;
    std::vector<int> stamp;
    int epoch = 0;
    std::vector<int> tcols;
    FreeDir fdM, fdZ, Kd, prod;
};

class RotBuilder {
  public:
    RotBuilder(const RotationModel& M, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
               const std::vector<RecColumn>& cols, const std::vector<Pauli>& axes);
    // the table of key s (TW words); throws std::logic_error on an engine-bug invariant
    void build(const uint64_t* s, RotBuild& out);
    long anti_products() const { return anti_products_; }
    const std::vector<Pauli>& axes() const { return axes_; }
  private:
    // fd of Z_T's free content (T real bits ∈ Γ_n): fd_mul over the Γ_n rows of T's decomposition, phase += |T| + 2η(T)
    void fd_of_ZT_(const uint64_t* T, FreeDir& out) const;
    mutable std::vector<uint64_t> zt_a_, zt_b_, zt_acc_;   // FCS round 2 (c): fd_of_ZT_'s word scratch (no per-call alloc)
    FreeDir fp_scratch_;                                   // FCS round 2 (c): build's pivot-column direction scratch
    // fd of the literal product of the base operators Q_c0 · Π_p Q_p0 (an ANTI extra among them): memo by the ordered set
    const FreeDir& product_fd_(int c, const std::vector<int>& pivs);
    const RotationModel& M_;
    const CertifiedGroupPlanes& G_;
    const FramedSuperposition& bare_;
    const std::vector<RecColumn>& cols_;
    std::vector<Pauli> axes_;             // B_j (the term view's axes; the ANTI products' base operators)
    std::vector<uint64_t> rmask_;         // per key word: the REAL bits (< t) — the virtual coordinates never enter a sign
    std::unordered_map<std::string, FreeDir> prod_memo_;
    long anti_products_ = 0;
    std::vector<int> key_;
};

// The plan store (DESIGN §1 "per shot"): one arena of u64 words holding every plan, an open-addressing index keyed by the
// plan's hash (plan_key_hash(s, TW): the XOR of the fired records' tokens ⊕ C), and the TUPLES — one entry per distinct
// ORDERED direction tuple (content-keyed) with its directions and its Born trie (review finding 4). Plan layout (words):
//   [0, TW) s ‖ [TW] ne ‖ [TW+1] ρ | κ << 32 ‖ [TW+2] AW | BW << 32 ‖ [TW+3] tuple (kNoTuple: none; kHomeTuple bit: a lane
//   plan on a HOME tuple) ‖ [TW+4] ordinal ‖ ⌈ρ/2⌉ words pivot columns (u32 pairs) ‖ ⌈κ/2⌉ words introducer columns ‖ ne
//   entries, each [col (u32) | flip << 32 | nondet << 33] then AW words A then BW words B.
// Every field is a full word (or a u32 column index: the engine's columns are int): no width cap and no refusal at a storage
// width (review H3, owner rule). A lane holds an OVERLAY store (its own plans and tuples) and reads the home's (read only
// during a super-block); adopt_from moves the lane's plans / tuples into the home (content-keyed, so no value depends on an
// index or on creation order).
struct RotTuple {
    std::string key;                      // the content key of the ordered direction tuple
    std::vector<FreeDir> dirs;
    std::vector<BornNode> trie;
    bool evicted = false;                 // its trie was cleared under the payload budget (a regrowth is a payload build)
    size_t trie_bytes() const { return trie.capacity() * sizeof(BornNode); }
};
class RotPlanStore {
  public:
    static constexpr uint64_t kNoTuple = ~0ULL;
    static constexpr uint64_t kHomeTuple = 1ULL << 63;
    static constexpr uint64_t kAbsent = ~0ULL;
    static constexpr int kHdr = 5;        // the header words after the key
    void init(int TW);
    int TW() const { return TW_; }
    // the plan's arena offset + 1 (0: absent)
    size_t find(const uint64_t* s, uint64_t h) const {
        size_t i = (size_t)h & mask_;
        for (;;) {
            const Slot& sl = slots_[i];
            if (!sl.o) return 0;
            if (sl.h == h) {
                const uint64_t* k = &arena_[sl.o - 1];
                int w = 0;
                while (w < TW_ && k[w] == s[w]) ++w;
                if (w == TW_) return sl.o;
            }
            i = (i + 1) & mask_;
        }
    }
    const uint64_t* plan(size_t o) const { return &arena_[o - 1]; }
    const void* slot_addr(uint64_t h) const { return &slots_[(size_t)h & mask_]; }   // prefetch target
    uint64_t* plan_mut(size_t o) { return &arena_[o - 1]; }
    // append the plan of `b` (the tuple interned by content here, or found in `home` — then marked kHomeTuple); returns
    // its offset + 1. ordinal: the plan's ordinal (the home: its index; a lane: base + its local index).
    size_t insert(const uint64_t* s, uint64_t h, const RotBuild& b, uint64_t ordinal, const RotPlanStore* home);
    // decode helpers (a plan's words; every store of one sampler has the same TW)
    int ne(const uint64_t* p) const { return (int)p[TW_]; }
    int rho(const uint64_t* p) const { return (int)(uint32_t)p[TW_ + 1]; }
    int kappa(const uint64_t* p) const { return (int)(p[TW_ + 1] >> 32); }
    int AW(const uint64_t* p) const { return (int)(uint32_t)p[TW_ + 2]; }
    int BW(const uint64_t* p) const { return (int)(p[TW_ + 2] >> 32); }
    uint64_t tref(const uint64_t* p) const { return p[TW_ + 3]; }
    uint64_t ordinal(const uint64_t* p) const { return p[TW_ + 4]; }
    int pivots_at(const uint64_t* /*p*/) const { return TW_ + kHdr; }
    int intros_at(const uint64_t* p) const { return TW_ + kHdr + (rho(p) + 1) / 2; }
    int entries_at(const uint64_t* p) const { return TW_ + kHdr + (rho(p) + 1) / 2 + (kappa(p) + 1) / 2; }   // the first entry word
    size_t words_of(const uint64_t* p) const { return (size_t)entries_at(p) + (size_t)ne(p) * (size_t)(1 + AW(p) + BW(p)); }
    static int col_at(const uint64_t* p, int base, int i) { return (int)((p[base + i / 2] >> (32 * (i & 1))) & 0xFFFFFFFFu); }
    static int ent_col(uint64_t x) { return (int)(uint32_t)x; }
    static int ent_flip(uint64_t x) { return (int)((x >> 32) & 1); }
    static int ent_nondet(uint64_t x) { return (int)((x >> 33) & 1); }
    size_t plans() const { return plans_; }
    size_t tuples() const { return tuples_.size(); }
    const RotTuple& tuple(size_t i) const { return tuples_[i]; }
    RotTuple& tuple_mut(size_t i) { return tuples_[i]; }
    uint64_t tuple_index(const std::string& key) const {   // kAbsent: absent
        auto it = tuple_ix_.find(key);
        return it == tuple_ix_.end() ? kAbsent : it->second;
    }
    size_t misses_k0() const { return misses_k0_; }
    size_t misses_kpos() const { return misses_kpos_; }
    size_t plan_bytes() const;            // content-based (the plans' words + the index + the tuples' directions)
    size_t trie_bytes() const;            // the tuples' tries (the evictable payload)
    void clear_tries();                   // exact: a node is a pure function of its tuple and prefix
    // after a walk / merge grew tuple ix's trie: a regrowth after an eviction is one payload build
    void note_grown(size_t ix) {
        RotTuple& T = tuples_[ix];
        if (T.evicted && !T.trie.empty()) { T.evicted = false; ++rebuilds_; }
    }
    size_t evictions() const { return evictions_; }
    size_t rebuilds() const { return rebuilds_; }
    // threads: the lane's plans and tuples into this (the home) store, lane order; the lane left empty.
    void adopt_from(RotPlanStore& lane);
    void clear();
    const std::vector<uint64_t>& arena() const { return arena_; }
  private:
    struct Slot { uint64_t h; size_t o; };
    void grow_();
    void index_insert_(uint64_t h, size_t o);
    uint64_t tuple_intern_(const std::string& key, std::vector<FreeDir>&& dirs);
    int TW_ = 0;
    std::vector<uint64_t> arena_;
    std::vector<Slot> slots_ = std::vector<Slot>(1, Slot{0, 0});
    size_t mask_ = 0, used_ = 0, plans_ = 0;
    size_t misses_k0_ = 0, misses_kpos_ = 0;
    size_t evictions_ = 0, rebuilds_ = 0;
    std::vector<RotTuple> tuples_;
    std::unordered_map<std::string, uint64_t> tuple_ix_;
    std::vector<size_t> plan_off_;        // ordinal → offset + 1 (the home: its own ordinals)
  public:
    size_t offset_of_ordinal(size_t ordinal) const { return ordinal < plan_off_.size() ? plan_off_[ordinal] : 0; }
    static std::string tuple_key(const std::vector<FreeDir>& dirs);
};

// lean D1-S2: the GAUGE-MAP ORACLE's report (the selfcheck window's tooth; the test API's rot_gauge_of): the rotation plan
// of a shot's key against (D) — today's compose_fired → residual_normal_form → canonicalize_mod_stabilizers → record-space-B
// table — under the computed gauge (g from P ⊕ P_D, δ from the directions' phases), and against (G) on the raw s.
struct RotGaugeCheck {
    bool ok = false;
    std::string why;
    std::vector<uint64_t> s;              // the key (TW words)
    int rho = 0, kappa = 0;
    std::vector<int> piv, intro;          // the plan's pivot / introducer columns
    std::vector<uint8_t> g, delta;        // per pivot: ⟨P ⊕ P_D, M_p⟩; per direction: Λ^R = −Λ^D
    std::vector<uint64_t> Px, Pz, PDx, PDz;   // the Clifford-only frame P (= ⊕ P_f) and (D)'s canonical prefix
    std::vector<uint64_t> A, B;           // per column (m × AW / m × BW): the plan's rows
    int AW = 0, BW = 0;
    std::vector<uint8_t> flip;            // per column: the INTERNAL base flip (channels: β ⊕ ref; extras: β)
    std::vector<FreeDir> dirs;            // the plan's directions
    long columns = 0, nodes = 0;          // columns compared; walked nodes compared
    bool g_table_equal = false;           // (G)(raw s) == the plan
};

}  // namespace qeccore
