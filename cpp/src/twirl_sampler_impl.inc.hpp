// twirl_sampler_impl.inc.hpp — INTERNAL to the twirl sampler (J-a r2.1, JA-1). The definition of
// TwirlRecordSampler::Impl and the file-local helpers its members and run_w use, shared by the two
// translation units: twirl_sampler.cpp (the ctor and the shared members) and twirl_sampler_fam.cpp (the run_w
// instances, J-b2 J3 / J9). Include from those two TUs only.
#pragma once

#include "qeccore/twirl_sampler.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/branch_hint.hpp"
#include "qeccore/record_space_b.hpp"   // J-b2 J7a: the readers' record table (rec_table_of_)
#include "qeccore/plan_class.hpp"       // plancache-design: the class key map Π
#include "qeccore/twirl_rotation.hpp"   // lean D1-S1: the rotation model + records (Frozen)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "qeccore/clifford_op.hpp"
#include "qeccore/diag_error_sampler.hpp"
#include "qeccore/noise_chain.hpp"      // lean N1: the mechanism tables and the block chains (Frozen)
#include "qeccore/pauli.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/sampler.hpp"        // FiredPauli / compose_fired
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"


namespace qeccore {

namespace {
using clk = std::chrono::steady_clock;
inline uint64_t rotl64(uint64_t x, int r) { return r ? (x << r) | (x >> (64 - r)) : x; }

// PER TU, NOT PER PROCESS (J-a final review m2): this anonymous namespace is in a header that BOTH
// twirl_sampler.cpp (the ctor) and twirl_sampler_fam.cpp (the run_w instances) include, so the notice below has one
// static per TU, initialised at that TU's FIRST USE (not at load). selfcheck_notice_once: one SELFCHECK_WINDOW event
// per TU. Only the FAM TU's copy is armed by a run (run_w lives there since J-b2 J3): at most one notice per process
// per extension (pinned: test_jb2_absent's notice test). No environment is read here.
// S0d: true exactly once per TU (the first armed oracle window; see PER TU above); the caller hands the
// event to the Python layer (SELFCHECK_WINDOW -> XtimVerificationWarning), never to stderr.
inline bool selfcheck_notice_once() {
    static bool noticed = false;
    if (noticed) return false;
    noticed = true;
    return true;
}

}  // namespace


// threads (fixplan threads/fixplan.md, T1): the members a lane copy of the sampler (Impl's copy constructor) must not
// share or must not copy. A lane is a copy of the sampler that shares the construction state (Impl::Frozen) and owns
// its per-shot state; these wrappers give the few non-copyable members that copy's semantics. Named namespace (Impl is
// defined in two TUs: no anonymous-namespace type may appear in it).
namespace twirl_lane {
// A mutex whose copy is a fresh, unlocked mutex (the readers' memo lock: a lane never reads the memo).
struct FreshMutex : std::mutex {
    FreshMutex() = default;
    FreshMutex(const FreshMutex&) noexcept : std::mutex() {}
    FreshMutex& operator=(const FreshMutex&) = delete;
};
// An atomic whose copy takes the source's value (relaxed).
template <class T>
struct CopyAtomic : std::atomic<T> {
    CopyAtomic() noexcept : std::atomic<T>() {}
    constexpr CopyAtomic(T v) noexcept : std::atomic<T>(v) {}
    CopyAtomic(const CopyAtomic& o) noexcept : std::atomic<T>(o.load(std::memory_order_relaxed)) {}
    CopyAtomic& operator=(const CopyAtomic&) = delete;
};
// An owning pointer of the sampler itself (the noise stream): a copy holds nothing (a lane never draws noise).
// The Born-trie walker: a copy is a FRESH walker on the same bare state (a lane's own ψ registers, counters and root
// memo; the node values are pure functions of the plan, so no copied state is needed).
struct FreshTrie : BornTrie {
    explicit FreshTrie(const FramedSuperposition& bare) : BornTrie(bare) {}
    FreshTrie(const FreshTrie& o) : BornTrie(o.bare_state()) {}
    FreshTrie& operator=(const FreshTrie&) = delete;
};
// The semantic oracle's work state (bare-sized after an oracle shot): a copy is EMPTY (a lane never runs the oracle; a
// copy of a bare-sized state would allocate chi-scaled storage per lane).
struct FreshState : FramedSuperposition {
    FreshState() : FramedSuperposition(0) {}
    FreshState(const FreshState&) : FramedSuperposition(0) {}
    FreshState& operator=(const FreshState&) = delete;
    FreshState& operator=(const FramedSuperposition& o) { FramedSuperposition::operator=(o); return *this; }
};
template <class T>
struct HomeOnly : std::unique_ptr<T> {
    HomeOnly() = default;
    HomeOnly(const HomeOnly&) noexcept : std::unique_ptr<T>() {}
    HomeOnly& operator=(const HomeOnly&) = delete;
};
}  // namespace twirl_lane

// J-b1 F (F item 2): the deferred noise stream's channels and per-alternative fired lists (twirl_sampler_fam.cpp; the
// ctor's one copy, compiled with -Werror=switch over NoiseChannel).
void twirl_alternatives_(const Circuit& deferred, double factor, std::vector<ErrorChannel>& chans,
                         std::vector<std::vector<FiredPauli>>& sides);

struct TwirlRecordSampler::Impl {
    // ── Inputs / configuration ────────────────────────────────────────────────────────────
    const FramedSuperposition& bare;           // caller-owned; must outlive the sampler
    const int n;
    std::vector<TwirlRecordSampler::DiagEvent> diag_;   // S0d: pending diagnostics (take_diagnostics)
    const long selfcheck;
    int setup_error = 0;

    // threads T1: the construction state that is immutable after the constructor lives in ONE Frozen object (defined
    // at the end of Impl), shared by the sampler and its lanes (Impl copies: the reference members below bind to the
    // same Frozen). The constructor writes through the references; nothing writes them after it.
    struct Frozen;
    std::shared_ptr<Frozen> fz_ = std::make_shared<Frozen>();

    // ── Group / propagation substrate ─────────────────────────────────────────────────────
    std::vector<Pauli>& gens = fz_->gens;                  // Frozen
    int ng = 0;
    int GW = 0;
    CertifiedGroupPlanes& G = fz_->G;                 // Frozen (its lazy members are warmed by freeze_())
    PropagationTable& table = fz_->table;                 // Frozen
    bool ppr_tables = false;

    // ── Channels ──────────────────────────────────────────────────────────────────────────
    struct SChan { Pauli GS; std::vector<uint64_t> mask; };
    std::vector<SChan>& sch = fz_->sch;                 // Frozen
    // S2b: the record family's columns in STORAGE order — every channel (slot c = sch index c) then
    // the extras (the Born observable, then the Born decisions in declaration order). Circuit constant.
    std::vector<RecColumn>& rec_cols_ = fz_->rec_cols_;         // Frozen
    const RecColIndex& rec_col_index_ = fz_->rec_col_index_;    // Frozen (lean L4; built after rec_cols_ is final)
    static int rec_order_(int kind, int index) { return kind * (1 << 24) + index; }   // dets, obs, decs
    std::vector<int> chan_det;                    // det channel c → detector index
    std::vector<int> chan_obs;                    // obs channel k → observable index (V2-T4;
                                                  // channels chan_det.size().. are obs channels)
    std::vector<uint8_t> chan_ref;                // channel c → REPORTING reference bit
                                                  // (classify ⊕ rec_invert; for det/obs
                                                  // channels MINUS the stage text's own
                                                  // deterministic-Pauli flip — P0 fix,
                                                  // reference-relative Stim semantics)
    int nchan = 0;
    int CW = 0;                                   // channel-space word count
    std::vector<uint64_t> bits_;                  // S0(f): the per-shot channel row (CW words)
    int XR_ = 0;                                  // lean L3: the extras' row width (XW when nextra_ > 0, else 0)
    std::vector<uint64_t> pxrow, pzrow;           // per-qubit prefix channel rows (CW words)
    uint64_t chtok = 0;                           // channel-set token

    // ── Noise channels / per-alt substrate ────────────────────────────────────────────────
    std::vector<std::vector<FiredPauli>>& sides = fz_->sides;   // flat-alt-id → fired list (Frozen)
    const NoiseModel& noise = fz_->noise;         // lean N1: the NOISE stream's tables (Frozen; qeccore/noise_chain.hpp)

    std::vector<uint8_t>& alt_is_ppr = fz_->alt_is_ppr;             // Frozen

    int NW = 0;
    // lean D1-S3: the diagonal cache's key (KW = NW words — the pair universe is empty: no diagonal-class plan with a CZ
    // part exists since the rotation loop serves every circuit with rotations). The Clifford path's identity plan: its
    // zero key, the key's hash and its normal form (a = 0, no pairs, zero prefix: the plan is prefix-free), set at the ctor.
    int KW = 0;
    std::vector<uint64_t> cliff_key_;
    uint64_t cliff_h_ = 0;
    DiagNormalForm cliff_nf_;
    bool cliff_window_check_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot);   // the window's oracle
    std::vector<uint64_t> prodmask;               // product-wire word mask (the class key's a ∧ U ∧ ¬prod strip)
    // S0(c) / D6 plan-payload budget (0 = unbounded) + the boundary accounting it reports.
    size_t plan_budget_ = 0;
    size_t payload_peak_ = 0, payload_overshoot_ = 0;
    void payload_boundary_();   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    bool fam_has_branches_() const;   // termroute-a: a FAM sampler with carried branches (twirl_sampler_fam.cpp)
    size_t born_memo_bytes_() const;   // S8-4: the root memos of the sampler's trie and its carried branches' tries
    void born_memo_clear_();           //       (defined in twirl_sampler_fam.cpp, where the branches live)
    // ── lean D1-S1: the rotation model and the compact per-alternative rotation records (qeccore/twirl_rotation.hpp),
    // compiled once at the ctor's end from the armed term view (twirl_sampler_rot.cpp). Nothing sampled reads them at S1.
    RotationModel& rot = fz_->rot;                                // Frozen
    RotationRecords& rot_rec = fz_->rot_rec;                      // Frozen
    void rot_setup_();                                            // the model + the records (+ their one-time verification)
    void rot_row_of_(const uint64_t* px, const uint64_t* pz, uint64_t* chan, uint64_t* xw) const;   // t(P_f): CW channel + XR extras words
    void rot_record_of_(const uint64_t* s, const uint64_t* chan, const uint64_t* xw, std::vector<uint32_t>& out) const;
    bool rot_record_check_(int a, const uint64_t* s, const uint64_t* chan, const uint64_t* xw, std::string& why) const;   // decode == source rows, bounds checked
    TermResidual* fam_term_view_() const;                         // the FamRoute's term view (nullptr: none); twirl_sampler_fam.cpp
    // ── lean D1-S2 (OWNER-SIGNED stream change on T circuits): the ROTATION SHOT LOOP (lean/D1/S2/plan.md). rot_on_ =
    // rot.built && t > 0, decided per circuit at construction: every shot of such a circuit that is neither framed nor
    // term-recorded (rec_term) takes rot_shot_ (twirl_sampler_rotloop.inc.hpp) — keyed on s = ⊕ S_f, served by the plan
    // store (RotPlanStore: sparse plans + tuple tries) — and the framed / rec_term shots take the term route's (G) on the
    // RAW composed (P, S) (the same gauge). rot_on_ = false: every path is the pre-S2 code (t = 0 circuits byte-identical).
    bool rot_on_ = false;
    twirl_lane::HomeOnly<RotBuilder> rot_builder_;   // the miss builder (a lane gets its own: run_threads_)
    twirl_lane::HomeOnly<RotPlanStore> rstore_;      // the home's store / a lane's OVERLAY store (a copy holds none)
    const RotPlanStore* rot_home_ = nullptr;         // a lane: the home's store (read only during a super-block)
    RotBuild rot_build_;                             // the miss scratch
    std::vector<uint64_t> rot_s_, rot_u_, rot_z_;    // per shot: the key words, the coin / Born outcome words
    std::unordered_map<uint64_t, std::vector<BornNode>> rot_cow_;   // a lane: copy-on-write tries of HOME tuples (merged at the barrier)
    long rot_window_shots_ = 0;                      // the gauge-map oracle's shots (diagnostic)
    // the s = 0 plan (built at construction; the home store never drops a plan): a clean shot's plan with no key / hash /
    // probe (a lane finds it in the home store). rot_id_skip_: that plan has ne = ρ = κ = 0 — a clean shot outside the
    // window and the record writer emits nothing, so run_w skips rot_shot_ (coins are keyed per shot: stream-neutral).
    size_t rot_id_o_ = 0;
    bool rot_id_skip_ = false;
    // the sparse per-alternative P supports (P_f's x / z wires; recording runs, the window): built on demand on the home
    struct RotPList { bool built = false; std::vector<uint32_t> xq, zq; std::vector<size_t> xoff, zoff; };
    RotPList& rot_plist = fz_->rot_plist;            // Frozen (filled by the home before any lane starts)
    void ensure_rot_plists_();                       // twirl_sampler_rot.cpp
    void rot_P_of_events_(const int32_t* ev, int cnt, Pauli& P) const;   // P = ⊕ P_f (mod phase) into a sized buffer
    void rot_setup_store_();                         // the builder + the store (the ctor)
    void rot_lane_setup_(Impl& lane) const;          // a lane's own builder + overlay store (run_threads_)
    size_t rot_miss_(const uint64_t* s, uint64_t h, bool lane);        // build + insert (a lane: into its overlay)
    int rot_walk_(const uint64_t* P, const RotPlanStore* ps, bool home_plan, bool lane);   // the tuple trie walk into rec_uz_
    void rot_rec_write_(const uint64_t* P, const uint64_t* bits, const int32_t* ev, int cnt, bool home_plan);   // the layout-3 record
    bool rot_window_check_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot, const uint64_t* P,
                           const RotPlanStore* ps, uint64_t h);          // twirl_sampler_rotw.cpp (the gauge-map oracle)
    // the oracle itself: the plan (stored: P / ps; nullptr: a fresh build of the key) of the events `ev` against (D) and
    // (G); uz (ρ + κ outcomes; nullptr: no node check) = the walk the plan's trie took
    void rot_gauge_check_(const int32_t* ev, int cnt, const uint64_t* P, const RotPlanStore* ps, const uint8_t* uz,
                          uint64_t h, bool have_h, RotGaugeCheck& out);
    // 0 emitted, 1 rejected early, -1 oracle failure. sbk (P3 S-b): the shot's gathered key row (TW words, then the hash
    // h) from the shot block, with bits / xbits_ already holding its t(P) words (b's events are then not read); nullptr:
    // the shot's events b.ev[e0, e1).
    template <int FWE> int rot_shot_(uint64_t* bits, const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1,
                                     long done, long sc, bool famrec, const uint64_t* sbk);
    std::vector<int> rot_record_classes_() const;    // the s = 0 plan's column classes (record_report)
    size_t rot_identity_();                          // the s = 0 plan's offset (built on first use)
    // a fired alternative carries a PPR-retry atom: the shot is not of the diagonal class (today's term route serves it)
    // a fired list with a PPR-retry alternative (lean D1-S3: the window oracle's choice of base comparator only)
    bool shot_has_ppr_ev_(const int32_t* ev, int cnt) const {
        if (!ppr_tables) return false;
        for (int e = 0; e < cnt; ++e)
            if (alt_is_ppr[(size_t)ev[e]]) return true;
        return false;
    }

    // ── Plan caches ───────────────────────────────────────────────────────────────────────
    // Owned by this sampler; never shared (S2b R-m4). Payloads depend on the bare state (frame,
    // eps, free order, amplitudes) and on the Born observable, which no cache key covers. Both are
    // fixed per sampler, so sharing was never needed. Always non-null after construction.
    std::shared_ptr<TwirlPlanCache> cache;

    // ── Per-shot loop state (persists across run() calls) ────────────────────────────────
    // J-a (N1) / J-b2 J6b: THE emit seam — the FamilySink (the legacy barrier / dets-pack sinks are deleted).
    // J-b1 P3 (R1): always inlined into run_w<*> (its only caller), with FamilySink::emit.
    __attribute__((always_inline)) inline void emit_shot_fam_(const uint64_t* bits, int obs_bit,
                                                              const uint8_t* decision_bits) {
        if (family_sink_) family_sink_->emit(bits, obs_bit, decision_bits, xbits_.data(), rf_coins.data());
    }
    // S2.2 (decoder-feedback): retain each shot's collapsed post-barrier state. Off by default
    // (production path byte-identical). When on, each faulted shot's record (plan, P, outcomes, σ)
    // reaches the sink from the accumulator fast path (S5a) or, cold, the structural path.
    bool retain_state = false;
    // ── Task 2b.1 Born-measurement DECISION channels ─────────────────────────────────────
    // Each declared DECISION(k) parity is classified like a DETECTOR/OBSERVABLE (G.reduce):
    //   IN_GROUP → a deterministic σ-mask channel appended to `sch` (its bit = ref ⊕ mask·σ,
    //              the existing channel machinery); dec_chan_* map channel slot → decision k;
    //   LOGICAL/ANTI → born_dec: (decision k, record operator W_dec), Born-measured per shot
    //              on the retained collapsed state via measure_pauli, in DECLARATION order.
    // S2b: a Born decision is a record row (the extras columns); on the exact route it is measured on
    // the shot's exact state with EXACT word ng + 1 + j.
    int num_dec_ = 0;                   // declared decision count (max DECISION index + 1)
    bool has_decisions_ = false;        // ≥1 DECISION declared (any class)
    bool has_born_dec_ = false;         // ≥1 LOGICAL/ANTI decision (forces need_amps)
    std::vector<std::pair<int, Pauli>> born_dec;  // (decision index, W_dec), declaration order
    std::vector<uint8_t> born_dec_inv;  // per born decision: deterministic record invert (⊕ rec_invert)
    std::vector<std::vector<int>> born_dec_recs;  // per born decision: its record slots (for rf folding)
    std::vector<std::vector<uint64_t>> born_dec_rf_mask;  // per born decision: RFW-word noisy-slot mask
    std::vector<int> dec_chan_slot_;    // IN_GROUP decision → its channel index in `sch`
    std::vector<int> dec_chan_idx_;     // IN_GROUP decision → its decision index k
    std::vector<uint8_t> dec_chan_ref_; // IN_GROUP decision → ref bit (noiseless ⊕ rec_invert)
    std::vector<std::vector<int>> dec_chan_recs_; // IN_GROUP decision → its record slots (for rf)
    std::vector<uint8_t> dec_scratch_;  // per-shot decision output bits (len num_dec_)
    std::vector<uint8_t> dec_raw_;      // S2b: this shot's Born decisions' RAW outcomes (every route)
    // The shot's P (a recording run's P_rec / the rotation loop's record writer; a reused buffer).
    Pauli fast_prefix_;
    long fast_born_shots_ = 0;     // diagnostic counter (prof): born shots served by a resident diagonal-cache plan
    // Emit this shot's decision output bits.
    //   IN_GROUP decisions: bit = raw channel bit (parity σ&mask, in `bits`) ⊕ noiseless ref.
    //   Born (LOGICAL / ANTI) decisions:
    //     * record-row routes (S2b): the RAW outcome is the decision's extras column (rows ⊕ t(P));
    //   then the emitted bit = raw ⊕ record invert ⊕ input-frame relabel ⊕ readout-flip coins.
    //   Returns nullptr if no DECISION is declared (decision-free circuits).
    const uint8_t* emit_decisions(const uint64_t* bits);   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp

    // ── S2b (design §2.6): the FORCED-PROJECTION materialize ─────────────────────────────────
    // One shot's retained state from its record (plan, P, u, z), with no replay of any draw:
    //   1. copy bare;  2. apply C0 = S^a · CZ (tableau application);  3. apply P;
    //   4. force each pivot record's operator M at its value y, then each direction introducer's.
    // y = record_outcomes_(table, P, (u, z)) — the record's raw values (Theorem 1 rows ⊕ t(P)).
    // Every other record is then a function of these (Theorem 1), and nothing else is projected:
    // a check no record reads stays coherent. The force primitive is measure_pauli at u = 0 / 1⁻
    // and its returned sign MUST equal y — anything else is a named engine-bug throw (no
    // tolerance, no "already definite" branch).
    // The record table of `nf` (identity-prefix law; P enters only as t(P)). J-b2 J7a: by record-space B (the run
    // path's builder: build_shot_law_core + build_record_table_recB) — the canonical table is unique
    // (record_space_b.hpp), so every reader reads the same table the σ dual basis built.
    void rec_table_of_(const DiagNormalForm& nf, RecTable& T) const {
        if (!nf.diagonal_class || !build_record_table_recB(G, bare, nf, rec_cols_, T))
            throw std::logic_error("materialize: a fallback / out-of-class plan has no record table "
                                   "(its shot is an exact record) (engine bug)");
    }
    // The record's raw values y as WORDS (storage column order, T.CWX words): base ⊕ ⊕_{u_k} coin_row[k]
    // ⊕ ⊕_{z_j} born_row[j] ⊕ t(P), t_c(P) = ⟨P, M_c⟩ — Theorem 1's affine map, word-parallel (code
    // review MAJOR-3, r6.1). THE one derivation of y: every reader (materialize, record_outcomes,
    // frames, RecordFidelity) calls it. P is given as prefix words (x then z, pnw words each).
    // The P-free part (Theorem 1's rows at (u, z)); the σ-4 assembly reads C's values from it.
    static void record_rows_values_(const RecBlob& T, const uint8_t* uz, uint64_t* out) {
        const int W = T.CWX;
        for (int w = 0; w < W; ++w) out[w] = T.base()[(size_t)w];
        for (int k = 0; k < T.rho; ++k)
            if (uz[k]) { const uint64_t* r = T.coin_row(k); for (int w = 0; w < W; ++w) out[w] ^= r[w]; }
#ifdef XTIM_MUTATE_R61_ROW_SKIP
        for (int j = 0; j + 1 < T.kappa; ++j)     // mutation build only (R61): born_row[kappa-1] skipped
#else
        for (int j = 0; j < T.kappa; ++j)
#endif
            if (uz[T.rho + j]) { const uint64_t* r = T.born_row(j); for (int w = 0; w < W; ++w) out[w] ^= r[w]; }
    }
    void record_values_words_(const RecBlob& T, const uint64_t* px, const uint64_t* pz, int pnw,
                              const uint8_t* uz, int n_uz, uint64_t* out) const;   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // The byte form of record_values_words_ (one call's y; the per-shot accessors).
    void record_outcomes_(const RecBlob& T, const Pauli& P, const uint8_t* uz, int n_uz,
                          std::vector<uint8_t>& y) const {
        std::vector<uint64_t> w((size_t)T.CWX, 0);
        record_values_words_(T, P.x.data(), P.z.data(), (int)P.x.size(), uz, n_uz, w.data());
        y.assign((size_t)T.m, 0);
        for (int c = 0; c < T.m; ++c) y[(size_t)c] = (uint8_t)((w[(size_t)(c >> 6)] >> (c & 63)) & 1ULL);
    }
    void force_record_(FramedSuperposition& st, int c, uint8_t y, const char* what) const {
        const int sgn = st.measure_pauli(rec_cols_[(size_t)c].M, y ? std::nextafter(1.0, 0.0) : 0.0);
        if ((sgn < 0 ? 1 : 0) != (y & 1))
            throw std::logic_error(std::string("twirl_records: ") + what + " record " + std::to_string(c) +
                                   " forced to y=" + std::to_string((int)(y & 1)) +
                                   " has zero weight (engine bug)");
    }
    // The record table of a plan for the record READERS (materialize, record_outcomes, frames,
    // record_fidelity): memoized per plan content (a ‖ cz — the prefix enters only as t(P)); the
    // table is a pure function of it, so the memo changes no value.
    // r6.4 (code review MINOR-4): the memo is BOUNDED and CHARGED to the plan payload budget. Every
    // entry is charged its rec_table_bytes; the budget covers the sampler's payloads plus these.
    // EVICTION ORDER (review R6-2): reader tables first, sampler payloads only after — a reader never
    // evicts a sampler payload; run()'s shot boundary evicts the readers' UNHELD tables (LRU) before
    // any payload, and never a table a reader holds (use_count > 1). A reader insert trims the
    // unheld tables (LRU, never the one it is inserting) against the budget minus the sampler's
    // resident bytes as last published by run(). Like the payloads, an excess nothing can evict
    // (held tables, or one table larger than the budget) is REPORTED (reader_overshoot_bytes), never
    // refused — the payload budget's contract.
    struct ReaderEntry {
        RecHandle t;                                   // §3/§4: a heap blob (built on any thread), held by handle
        size_t bytes = 0;
        uint64_t tick = 0;
    };
    mutable twirl_lane::FreshMutex rec_memo_mu_;
    mutable std::unordered_map<std::string, ReaderEntry> rec_memo_;
    mutable uint64_t reader_tick_ = 0;                // under rec_memo_mu_
    mutable long reader_evictions_ = 0;               // under rec_memo_mu_
    mutable size_t reader_peak_ = 0, reader_overshoot_ = 0;   // under rec_memo_mu_
    mutable twirl_lane::CopyAtomic<size_t> reader_bytes_{0};
    twirl_lane::CopyAtomic<size_t> budget_pub_{0};               // plan_budget_, readable without run()'s state
    mutable twirl_lane::CopyAtomic<size_t> sampler_resident_pub_{0};   // published at run()'s shot boundaries
    mutable twirl_lane::CopyAtomic<long> reader_lookups_{0};
    // Evict unheld reader tables, least recently used first, until sampler + readers ≤ budget
    // (`keep` exempt). Caller holds rec_memo_mu_.
    void reader_trim_locked_(size_t sampler_bytes, const std::string* keep) const {
        const size_t B = budget_pub_.load(std::memory_order_relaxed);
        if (!B) return;
        while (sampler_bytes + reader_bytes_.load(std::memory_order_relaxed) > B) {
            auto victim = rec_memo_.end();
            for (auto it = rec_memo_.begin(); it != rec_memo_.end(); ++it) {
                if (keep && it->first == *keep) continue;
                if (it->second.t.use_count() != 1) continue;          // held by a reader
                if (victim == rec_memo_.end() || it->second.tick < victim->second.tick) victim = it;
            }
            if (victim == rec_memo_.end()) break;
            reader_bytes_.fetch_sub(victim->second.bytes, std::memory_order_relaxed);
            rec_memo_.erase(victim);
            ++reader_evictions_;
        }
        const size_t tot = sampler_bytes + reader_bytes_.load(std::memory_order_relaxed);
        if (tot > B && tot - B > reader_overshoot_) reader_overshoot_ = tot - B;
    }
    std::string memo_key_(const DiagNormalForm& nf) const {
        std::string key;
        key.reserve((size_t)n + 1 + 4 * nf.cz.size());
        for (int q = 0; q < n; ++q) key.push_back((char)(nf.a[(size_t)q] & 1));
        key.push_back((char)0xFF);
        for (const auto& pr : nf.cz) {
            key.push_back((char)(pr.first & 0xFF)); key.push_back((char)((pr.first >> 8) & 0xFF));
            key.push_back((char)(pr.second & 0xFF)); key.push_back((char)((pr.second >> 8) & 0xFF));
        }
        return key;
    }
    // ── plancache-design (review F2, the ONE choke point): every READER of a kind-0 content reads the table of its
    // CLASS — the run's own resident class table when it holds one, else the readers' memo keyed by the CLASS key and
    // built from the class REPRESENTATIVE (class_nf: the same pure function the run builds the class from). A content
    // with a CZ pair outside the pair universe is not classed (route 1): its full normal form keys the memo as before
    // (no a-strip). Consistency by construction: every consumer reads one table per class (plan §1).
    // The memo key of nf: the class key (tag 0x03 — no full-nf key starts with 0x03, a-bits are 0 / 1 — ‖ KW words)
    // or, unclassed, memo_key_(nf); `ck` receives the class key (empty when unclassed).
    std::string reader_key_(const DiagNormalForm& nf, std::vector<uint64_t>& ck) const {
        if (cache->class_key_of_nf(nf, ck)) {
            std::string k(1, '\x03');
            k.append(reinterpret_cast<const char*>(ck.data()), ck.size() * sizeof(uint64_t));
            return k;
        }
        ck.clear();
        return memo_key_(nf);
    }
    // The run's resident class table of a class key (nullptr: none / not built).
    RecHandle class_resident_table_(const std::vector<uint64_t>& ck) const {
        if (ck.empty()) return RecHandle();
        const PlanClass* c = cache->find_class(ck.data(), lhash(ck.data()));
        if (!c || !c->resident || !c->rec || !c->emit_built) return RecHandle();
        return c->rec;                                 // a handle: the blob outlives an eviction while held
    }
    // Build a reader's table: the class representative's (classed) or nf's own (unclassed).
    void reader_table_build_(const DiagNormalForm& nf, const std::vector<uint64_t>& ck, RecTable& T) const {
        if (ck.empty()) { rec_table_of_(nf, T); return; }
        DiagNormalForm rep;
        cache->key_nf(ck.data(), rep);
        rec_table_of_(rep, T);
    }
    // r8 (final review MAJOR-1): a table for a BUFFER to hold. It leaves the reader memo (if it was there)
    // and is never charged: buffer-held tables sit outside the plan payload budget (coordinator ruling),
    // so they can never make run() evict a sampler payload. The same pure function of the class key.
    RecHandle rec_table_held_(const DiagNormalForm& nf) const {
#ifdef XTIM_MUTATE_R81_HELD_CHARGED
        return rec_table_memo_(nf);               // mutation build only (R81): held tables charged (c8d1ff5)
#endif
        reader_lookups_.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint64_t> ck;
        const std::string key = reader_key_(nf, ck);
        if (RecHandle t = class_resident_table_(ck)) return t;
        {
            std::lock_guard<std::mutex> lk(rec_memo_mu_);
            auto it = rec_memo_.find(key);
            if (it != rec_memo_.end()) {
                RecHandle t = it->second.t;
                reader_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                rec_memo_.erase(it);
                return t;
            }
        }
        RecTable T;
        reader_table_build_(nf, ck, T);
        return make_heap_blob(T, 0, 0);                // (no emission block: readers never read one)
    }
    RecHandle rec_table_memo_(const DiagNormalForm& nf) const {
        reader_lookups_.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint64_t> ck;
        const std::string key = reader_key_(nf, ck);
        if (RecHandle t = class_resident_table_(ck)) return t;
        {
            std::lock_guard<std::mutex> lk(rec_memo_mu_);
            auto it = rec_memo_.find(key);
            if (it != rec_memo_.end()) { it->second.tick = ++reader_tick_; return it->second.t; }
        }
        RecTable T;
        reader_table_build_(nf, ck, T);
        RecHandle built = make_heap_blob(T, 0, 0);
        std::lock_guard<std::mutex> lk(rec_memo_mu_);
        auto ins = rec_memo_.emplace(key, ReaderEntry{});
        ReaderEntry& e = ins.first->second;
        if (ins.second) {                              // (another reader may have raced it in)
            e.t = std::move(built);
#ifdef XTIM_MUTATE_R64_NO_CHARGE
            e.bytes = 0;                               // mutation build only (R64): the table is not charged
#else
            e.bytes = e.t->bytes;
#endif
            reader_bytes_.fetch_add(e.bytes, std::memory_order_relaxed);
            const size_t rb = reader_bytes_.load(std::memory_order_relaxed);
            if (rb > reader_peak_) reader_peak_ = rb;
        }
        e.tick = ++reader_tick_;
        RecHandle out = e.t;                           // held while the trim runs: never its victim
        reader_trim_locked_(sampler_resident_pub_.load(std::memory_order_relaxed), &key);
        return out;
    }
    // plan_structure's table: the class table (a copy; uncharged, not memoized) and the class id (0: unclassed).
    void reader_table_fresh_(const DiagNormalForm& nf, RecTable& T, uint32_t& class_id) const {
        std::vector<uint64_t> ck;
        (void)reader_key_(nf, ck);
        class_id = 0;
        if (!ck.empty())
            if (const PlanClass* c = cache->find_class(ck.data(), lhash(ck.data()))) {
                class_id = c->id;
                if (c->resident && c->rec && c->emit_built) { c->rec->to_table(T); return; }
            }
        reader_table_build_(nf, ck, T);
    }
    FramedSuperposition materialize_(const DiagNormalForm& nf, const uint8_t* uz, int n_uz) const {
        const RecHandle Th = rec_table_memo_(nf);
        const RecBlob& T = *Th;
        std::vector<uint8_t> y;
        record_outcomes_(T, nf.prefix, uz, n_uz, y);
        FramedSuperposition st = bare;
        apply_residual_gates_(st, nf);
#ifdef XTIM_MUTATE_MS_SKIP_PIVOT
        for (int k = 1; k < T.rho; ++k)           // mutation build only (M-S b): pivot 0 skipped
#else
        for (int k = 0; k < T.rho; ++k)
#endif
            force_record_(st, T.pivot_rec()[k], y[(size_t)T.pivot_rec()[k]], "anchor (pivot)");
        for (int j = 0; j < T.kappa; ++j) {
            uint8_t yj = y[(size_t)T.born_rec()[j]];
#ifdef XTIM_MUTATE_MS_BORN_FLIP
            yj ^= 1;                              // mutation build only (M-S a / M-ANCH)
#endif
            force_record_(st, T.born_rec()[j], yj, "anchor (direction)");
        }
        return st;
    }
    // The barrier's record → state entry (plan bytes ‖ outcomes (u, z)): the state is a function
    // of the record.
    FramedSuperposition materialize_shot(const uint8_t* plan_key, int n_plan,
                                         const uint8_t* coins, int n_coins) const {
        const DiagNormalForm nf = parse_plan_key(plan_key, n_plan);
        return materialize_(nf, coins, n_coins);
    }

    // ── E3 (exact-residual arc): plan-key parser, shared format with materialize_shot ─────────
    // Byte-for-byte the SAME parse as materialize_shot's step 1 above (deliberately duplicated —
    // the established additive pattern — so the byte-pinned materialize_shot body stays
    // untouched). Format (serialized in run(), the "barrier2 Lever 2" block):
    //   prefix.x NW u64 words LE ‖ prefix.z NW words LE ‖ n a-mask bytes (&1) ‖ 0xFF ‖
    //   cz pairs (first u16-LE, second u16-LE)*          — global phase EXCLUDED.
    // Empty key (n_plan == 0) = clean/identity residual.
    DiagNormalForm parse_plan_key(const uint8_t* plan_key, int n_plan) const;   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // ── V3 Born-weighted observable channel state ────────────────────────────────────────
    bool born_obs = false;              // single logical observable, diagonal table
    int born_idx = -1;                  // its observable index
    Pauli born_W;                       // combined terminal-read record operator
    // Deterministic record invert of the born observable (parity of rec_invert over its records:
    // Stim `!`, MPAD 1, the H-elimination's folded sign). Every emission site — the record rows
    // (RAW), the PPR channel law (pl.obs_m is the invert-free ⟨Λ⟩) and the exact measure — XORs it
    // in (audit I/J, 2026-10-09).
    uint8_t born_inv_ = 0;
    int obs_bit_cur = 0;                // per-shot scratch consumed by the tail/sink
    // ── SEMANTICS ORACLE (2026-10-09 sampling audit; selfcheck window only) ─────────────────
    // The wiring oracles compare two implementations of the SAME law, so a law-level defect
    // (audit I/J: the born observable's invert; M: unread generators collapsing a decision)
    // passes them. This oracle recomputes, for the first min(selfcheck, 64) shots of the first
    // run() of a born-observable / born-decision circuit, the shot's physics from scratch:
    // R|bare⟩ with R = compose_fired's residual (the PRE-canonicalization normal form) applied
    // as gates, projected onto the EMITTED raw channel bits (a definite channel that disagrees
    // = impossible record → FAIL); then (S2b) every extra — the Born observable, then the Born
    // decisions — projected at its emitted RAW value, a definite one disagreeing → FAIL.
    // STREAM-NEUTRAL: no RNG draw, no output side effect (a trip makes run() return false).
    int obs_bit_cur_raw_ = 0;           // the Born observable's raw outcome this shot (pre rf/frame)
    std::vector<uint64_t> sem_raw_bits_;  // CW words: emitted channel bits BEFORE readout flips / frame
    long sem_budget_ = 0;               // oracle shots left in this run()
    long sem_checked_ = 0;              // diagnostic: shots the oracle verified
    twirl_lane::FreshState sem_work_;
    long obs_ones_ = 0, ps_acc_ = 0, ps_obs_ones_ = 0;
    std::vector<uint64_t> det_refw;     // CW words: det-channel ref bits (acceptance test)
    std::vector<uint64_t> det_maskw;    // CW words: 1s at det-channel positions
    // ── Readout-flip layer (Stim M(p)): a per-noisy-record Bernoulli(p) coin XORed into every
    // channel whose record set contains that record — a pure record-space Pauli flip.
    //
    // Fix A (byte-identical): transposed column form.
    //   chan_rf_mask[c][w] (row form) is replaced by rf_col[sl] (column form): for each noisy-
    //   record slot sl, rf_col[sl] is the CW-word bitvector of channels whose mask contained sl.
    //   apply_rf iterates SET bits of rf_coins via ctz and XORs the corresponding column — same
    //   GF(2) linearity, same result, but only O(fired) column XORs instead of O(nchan × RFW)
    //   popcount ops.  born_rf_mask is folded into a parallel per-slot bit rf_obs_bit[sl] so the
    //   born observable flip rides the same ctz loop.
    //   The row-form chan_rf_mask is kept at construction time (still needed for born_dec_rf_mask
    //   which is read in emit_decisions and has its own loop), but is no longer used in apply_rf.
    //
    // Per-shot rf coins (lean L5, owner-signed stream change): the slots of EQUAL p form a group (groups in the order
    //   of their first slot, slots ascending inside a group); per shot each group is ONE chain of inverse-CDF geometric
    //   gaps over its slots on the shot's sequential RF words (draw_rf) — the cost is the fired flips, not the slot
    //   count; a pure function of (seed, shot). rf_grp_off (G + 1) / rf_grp_slot (nrf: slot ids in group order) /
    //   rf_grp_invl (per group 1 / log1p(-p); 0 = p >= 1: every slot of the group fires without a draw).
    int nrf = 0;                        // distinct noisy records (prob>0), packed into RFW words
    int RFW = 0;                        // ceil(nrf/64)
    std::vector<double> rf_prob;        // nrf scaled probabilities (the groups are built from them)
    std::vector<int32_t> rf_grp_off, rf_grp_slot;
    std::vector<double> rf_grp_invl;
    // FCS fix 5: per group a threshold for its FIRST draw: u <= rf_grp_thr[g] ⇒ that draw's gap reaches past the group
    // (k = the group's slot count), so the chain ends with no log. thr = (1−p)^k · (1 − 2^−20) (0 when p >= 1); the
    // 2^−20 relative guard band dominates every rounding of log / the product / exp (draw_rf), and inside it the exact
    // computation decides, so the draws and the fired slots are unchanged.
    std::vector<double> rf_grp_thr;
    std::vector<std::vector<uint64_t>> chan_rf_mask;   // per channel: RFW-word noisy-slot mask (kept for born_dec path)
    std::vector<uint64_t> born_rf_mask; // born observable's noisy-slot mask (RFW words, kept for reference)
    std::vector<uint64_t> rf_col_data_; // FIX A: flat storage for transposed columns (nrf * CW words)
    std::vector<uint64_t*> rf_col;      // FIX A: rf_col[sl] -> CW-word channel bitvector for slot sl
    std::vector<uint8_t> rf_obs_bit;    // FIX A: rf_obs_bit[sl] = 1 if born_rf_mask bit sl is set
    std::vector<uint64_t> rf_coins;     // per-shot scratch (RFW words); nonzero only in the words rf_fw_ lists
    std::vector<int32_t> rf_fw_;        // lean L7: the words of rf_coins draw_rf set this shot (distinct, RFW capacity)
    int rf_nfw_ = 0;                    //          their count (0: no readout flip this shot)
    std::vector<FiredPauli> fired;
    std::vector<long long> chancnt;               // per-channel 1-bit counts
    bool count_channels_ = false;   // disabled by default; set in run() when profiling or QEC_TW_COUNT
    long clean = 0, diag = 0, checked = 0, used = 0;
    // ── S0(a): per-route shot counters (cumulative; reset by set_seed like every run counter) ──
    // Every shot takes exactly ONE route (record_report()["route_counts"]). The two hot routes —
    // the abelian streaming fast path and the κ=0 retention path — carry NO per-shot counter: run()
    // attributes the shots it did not count when it exits (RemainderGuard), so instrumentation adds
    // nothing to them. Every other route is a cold or slower path and pays one increment. The
    // exact_* routes are the per-shot exact-engine fallbacks, one key per trigger (their sum is fb);
    // each raises one XtimPerformanceWarning per run() in which it fired (S0d: after run()).
    enum Route : int {
        R_FAST_ACC = 0,          // accumulator fast path, κ_rec = 0 plan, not retained (S5a: that path counts
                                 // with the structural path's table: κ_rec > 0 / retained shots below)
        R_SLOW_K0,               // materialising path, κ=0 law (cold miss / selfcheck window)
        R_RECORD_KPOS,           // record path (accumulator fast or structural), κ>0 plan (Born-trie walk; non-retention)
        R_RETAIN_K0,             // retention (sample_barrier) record path (fast or structural), κ=0 plan
        R_RETAIN_KPOS,           // retention record path (fast or structural), κ>0 plan (Born-trie walk)
        R_COUNT
    };
    static const char* route_name_(int r) {
        static const char* nm[R_COUNT] = {
            "fast_accumulator", "slow_kappa0", "record_kappa_pos", "retain_kappa0",
            "retain_kappa_pos"};
        return nm[r];
    }
    long route_[R_COUNT] = {};
    long fast_derived_ = 0;              // abelian streaming fast path (attributed per run)
    // S2a: the plans' Born tries are walked by ONE per-sampler walker (its miss-path working
    // state is the sampler's single reusable free-register buffer). A walk that allocated nodes
    // grew the plan's payload: re-account it (the trie is payload, evictable between shots).
    twirl_lane::FreshTrie trie_;
    bool born_node_event_ = false;       // decision 7: this run's BORN_NODE event is queued
    // v2.7 selfcheck once-per-sampler: the oracle window is bound to the FIRST run() of a
    // sampler lifetime. This latch is set by run() (shots > 0) and deliberately NOT reset by
    // set_seed() — the in-place reseed zeroes the `checked`/`pprchecked` gating counters,
    // which would otherwise re-arm the window on every seeded call. rebuild() constructs a
    // fresh Impl, so the window re-arms there (fresh-construction semantics). STREAM-NEUTRAL:
    // a spent window behaves exactly like selfcheck=0, which is byte-identical to the
    // window-on path (ratified; oracle tests/test_selfcheck_once.py).
    bool selfcheck_spent_ = false;
    DiagErrorSampler::SparseBatch batch;
#ifdef XTIM_BENCH_REPLAY
    // ── BENCH-ONLY fast-path replay hook (design §5.0, "a fixed recorded event batch pushed
    // through the warm hot loop"). Compiled ONLY with -DXTIM_BENCH_REPLAY; the shipped extension
    // never sees this block (its preprocessed source is unchanged). RECORD appends every batch
    // the noise sampler hands the hot loop (shot-major alt ids + offsets); REPLAY makes the hot
    // loop read its batch from a recorded source instead of the noise chain's event list — the ONLY
    // difference. Everything downstream of `batch` (the per-shot loop, its keyed/coin/rf draws,
    // plans, sinks) is the production code. Replay shot i of a run() = recorded shot i.
    bool bench_rec_on_ = false;
    std::vector<int32_t> bench_rec_off_{0}, bench_rec_ev_;
    bool bench_src_ = false;
    std::vector<int32_t> bench_src_off_, bench_src_ev_;
    __attribute__((noinline)) void bench_fill_(long first, int S) {
        if (first + S + 1 > (long)bench_src_off_.size())
            throw std::runtime_error("bench replay: run() asks past the recorded shots");
        const int32_t b0 = bench_src_off_[(size_t)first], b1 = bench_src_off_[(size_t)(first + S)];
        batch.S = S;
        batch.shot_off.resize((size_t)S + 1);
        for (int i = 0; i <= S; ++i) batch.shot_off[(size_t)i] = bench_src_off_[(size_t)(first + i)] - b0;
        batch.ev.assign(bench_src_ev_.begin() + b0, bench_src_ev_.begin() + b1);
    }
    __attribute__((noinline)) void bench_append_() {
        const int32_t base = bench_rec_off_.back();
        for (int i = 1; i <= batch.S; ++i) bench_rec_off_.push_back(base + batch.shot_off[(size_t)i]);
        bench_rec_ev_.insert(bench_rec_ev_.end(), batch.ev.begin(), batch.ev.end());
    }
#endif
    // ── S0(b): keyed per-shot streams (krng, keyed_rng.hpp) — the determinism contract D10 ────
    // Every random draw of a shot is krng::draw(seed_, shot_, TAG, ctr): shot_ = the CUMULATIVE
    // shot index (fresh Impl / set_seed: 0; seed=None continues it across run() calls). A draw is
    // addressed, never consumed from a shared sequence, so the bytes of a shot do not depend on
    // the route it takes, on the plan-cache state, or on how calls split the shots. ONE coin
    // convention on every route: fair coin k = bit k of the COIN words (coins_), Born step j =
    // BORN word j (the record trie, then the σ-aux trie from kappa_rec on), readout flip slot
    // sl = RF word sl, gauge detector g = bit g of the GAUGE words, exact-fallback generator i =
    // EXACT word i, the exact route's Born observable = EXACT word ng and its Born decision j =
    // EXACT word ng + 1 + j (S2b: no DEC stream; OBS word 0 serves PPR plans only). A Born ±1 outcome is +1 (raw bit 0) iff u < p(+1) — measure_pauli's rule —
    // on every route. The noise stream is keyed per (equal-P group, aligned 4096-shot block) (lean N1, noise_chain.hpp).
    uint64_t seed_ = 0;
    // speed-kill T1: true when the S channels (sch) were drawn from mrng(seed + 4242) at
    // construction (!opt.circuit_channels) — the channel set itself is then seed-dependent
    // physics and set_seed() must refuse (the caller reconstructs instead).
    bool synthetic_channels_ = false;
    uint64_t shot_ = 0;                           // cumulative index of the shot being sampled
    // Cumulative shot counter (fresh Impl / set_seed(): 0; +1 per shot across run() calls, so an
    // unseeded call continues every per-shot stream instead of replaying it — review C4).
    uint64_t done_shots_ = 0;
    krng::CoinBits coins_;                        // COIN (positioned every shot)
    krng::Stream born_s_, rf_s_;   // positioned at first use per shot
    void set_keys_(uint64_t seed) {
        seed_ = seed;
        coins_.st.set_key(seed, krng::COIN);
        born_s_.set_key(seed, krng::BORN);
        rf_s_.set_key(seed, krng::RF);
    }
    // The shot's BORN stream (Born step j = word j), positioned at the current shot.
    const krng::Stream& born_at_shot_() {
        born_s_.at(shot_);
        return born_s_;
    }
    double t_draw = 0.0, wall = 0.0;

    Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
         const std::vector<std::pair<int, int>>& reads,
         const std::vector<std::vector<int>>& detectors,
         const std::vector<std::pair<int, std::vector<int>>>& observables,
         uint64_t seed, const TwirlRecordOptions& opt);
    // The caches' owner: this sampler (S2b R-m4: never shared). TwirlRecordSampler::make() creates
    // them before the sampler object (the bindings' historical allocation order).
    Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
         const std::vector<std::pair<int, int>>& reads,
         const std::vector<std::vector<int>>& detectors,
         const std::vector<std::pair<int, std::vector<int>>>& observables,
         uint64_t seed, const TwirlRecordOptions& opt,
         std::shared_ptr<TwirlPlanCache> plans,
         std::optional<PropagationTable>&& class_table = std::nullopt);
    // S0(f): run-width channel rows at no hot-path cost. The shot loop is ONE template body with
    // two width instances: FW = 8 keeps the per-shot channel row a local uint64_t[8] for CW ≤ 8 (≤ 512
    // channels — exactly the capped engine's code generation; see run_w), FW = 0 points it at the
    // run-width member storage bits_ (CW > 8). Both instances are compiled in twirl_sampler_fam.cpp (body:
    // twirl_sampler_runw.inc.hpp; J-b2 J9: the FAM parameter is dropped). run() calls the instance chosen at
    // construction (run_fn_): one indirect call per run() call, nothing per shot.
    // early-reject (OD-2026-10-25): with a postselection mask set, run() takes the ER instance (er_run_fn_,
    // twirl_sampler_er.cpp); without one, exactly the instance it always took.
    bool run(long shots) {
        if (XTIM_UNLIKELY(er_.p != nullptr)) return er_run_(shots);
        return (this->*run_fn_)(shots);
    }
    bool er_run_(long shots);   // twirl_sampler_er.cpp
    static constexpr int kErRun = 1 << 12;   // run_w<FW | kErRun>: the masked instance of width FW
    template <int FW> bool run_w(long shots);

    // ── speed-kill T1: in-place reseed ────────────────────────────────────────────────────
    // Resets EXACTLY the state a fresh Impl construction (re)derives from `seed`,
    // reconstructing nothing. The seed-dependent inventory (exhaustive — verified against
    // every `seed` use in the constructor plus every cumulative member run() mutates):
    //   * the keyed stream keys (set_keys_: COIN, BORN, OBS, DEC, RF, GAUGE, EXACT) — every
    //                           per-shot draw is a pure function of (seed, shot index, tag, ctr);
    //   * the noise chains     — lean N1: a pure function of (seed, block, table), keyed at generation (no state);
    //   * done_shots_         — cumulative shot index (fresh Impl: 0; cumulative otherwise);
    //   * the cumulative run() counters clean/diag/checked/used/fb, pprshots/pprchecked,
    //     obs_ones_/ps_acc_/ps_obs_ones_, chancnt, t_draw/wall — zeroed so the selfcheck
    //     oracle window (`checked`/`pprchecked` gating) and every reporting accessor behave
    //     exactly as on a fresh construction. Plan-cache hit/miss counters live in the SHARED
    //     caches, which survive rebuild() too — untouched, same as the rebuild path.
    // Seed-INDEPENDENT state deliberately kept warm (pure functions of the compiled circuit,
    // already shared across rebuild()s): propagation table, CertifiedGroupPlanes, channel
    // compilation, alt blobs/atoms, plan caches + fast-route index, lazily built fb_alt_tab,
    // In-place reseed: re-keys every per-shot stream (COIN / BORN / RF / GAUGE / EXACT, and OBS
    // for PPR plans) and zeroes the run counters. materialize_shot() is const and fully
    // record-driven — outstanding barrier buffers are unaffected by an in-place reseed.
    bool set_seed(uint64_t seed) {
        if (synthetic_channels_) return false;   // sch drawn from mrng(seed + 4242): rebuild required
        ++seed_epoch_;                           // scaling fix-4: a failed call that reseeded keeps the cached-work history
        set_keys_(seed);
        done_shots_ = 0;
        clean = diag = checked = used = 0;
        std::fill(route_, route_ + R_COUNT, 0L);
        fast_derived_ = 0;
        trie_.walks = trie_.steps = trie_.nodes = trie_.node_evals = trie_.chi_j = trie_.passes = 0;
        trie_.max_support = 0;
        trie_.max_depth = 0;
        rec_tables_ = rec_extras_ = rec_ops_plane_ = rec_ops_pivot_ = rec_ops_greduce_ = 0;
        rec_ops_echelon_ = rec_pivots_extras_ = rec_dirs_extras_ = 0;
        rec_kappa_hist_.clear();
        obs_draws_record_ = 0;
        structural_shots_ = 0;
        obs_ones_ = 0; ps_acc_ = 0; ps_obs_ones_ = 0;
        std::fill(chancnt.begin(), chancnt.end(), 0);
        t_draw = 0.0; wall = 0.0;
        return true;
    }

    // ── Small helpers (were block-scope lambdas in the bench mode) ───────────────────────
    int par(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const {
        int pc = 0;
        const size_t w = std::min(a.size(), b.size());
        for (size_t i = 0; i < w; ++i) pc += __builtin_popcountll(a[i] & b[i]);
        return pc & 1;
    }
    // The plan index hash (GF(2)-linear; shared with the cache — twirl_kernel.hpp).
    uint64_t lhash(const uint64_t* k) const { return plan_key_hash(k, KW); }
    static bool bany_words_(const std::vector<uint64_t>& w) {
        for (uint64_t x : w) if (x) return true;
        return false;
    }
    // ── S2b: emission from the record table ──────────────────────────────────────────────
    // Columns [0, nchan) are the σ channels (the existing CW-word channel rows, internal convention
    // raw ⊕ classify sign); columns [nchan, nchan + nextra_) are the extras (XW words, RAW). Every
    // diagonal route writes a shot's record bits as base ⊕ Σ u_k·coin_row[k] ⊕ Σ z_j·born_row[j]
    // (the plan's canonical table: coin k = COIN bit k, z_j = the trie walk on BORN word j) ⊕ t(P)
    // (the existing prefix rows for channels; ⟨P, W_e⟩ for the extras) — one definition.
    int nextra_ = 0;                     // extras after the channels
    int XW = 1;                          // extras word count (≥ 1, so .data() is never null)
    std::vector<uint64_t> xbits_;        // the current shot's extras bits (RAW ⊕ t(P))
    std::vector<uint64_t> refmask_ch_;   // CW words: the channels' classify signs
    int obs_col_ = -1;                   // extras column of the Born observable (−1: none)
    std::vector<int> dec_col_;           // Born decision j → extras column
    std::vector<uint64_t> qcol_x_, qcol_z_;   // r7.1: per-wire record-column rows (n × QCW_ words each)
    int QCW_ = 1;
    // lean L3: the extras' columns of qcol_x_ / qcol_z_ re-aligned to the extras' bit positions (n × XW words each): a prefix
    // X on wire q flips the extras in xrow_x_[q], a prefix Z those in xrow_z_[q] — t(P) over the extras, as pxrow / pzrow are
    // over the channels. The streaming route's per-alternative rows and pair term read these (empty when nextra_ == 0).
    std::vector<uint64_t> xrow_x_, xrow_z_;
    std::vector<uint8_t> rec_uz_;        // the current shot's recorded outcomes: u (ρ), then z (κ_rec)
    int rec_uz_width_ = 0;               // ρ + κ_rec of the plan record_rows_ last emitted
    bool rec_route_ = false;             // the current shot's extras came from the record rows
    const CachedPlan* plan0_ = nullptr;  // the identity plan (clean shots), built on first use
    bool xbit_(int e) const { return (xbits_[(size_t)(e >> 6)] >> (e & 63)) & 1ULL; }
    // Split a column row over m columns (CWX words) into the channel words (CW) and the extras (XW). S8-2: word at a
    // time (bits64_ reads 64 row bits from any offset, 0 past the row); the same bits as the per-column loop.
    static uint64_t bits64_(const std::vector<uint64_t>& row, int off) {
        const size_t w = (size_t)(off >> 6);
        const int sh = off & 63;
        const uint64_t lo = w < row.size() ? row[w] : 0;
        if (!sh) return lo;
        const uint64_t hi = w + 1 < row.size() ? row[w + 1] : 0;
        return (lo >> sh) | (hi << (64 - sh));
    }
    static uint64_t lowmask_(int nbits) { return nbits >= 64 ? ~0ULL : ((1ULL << nbits) - 1); }
    void split_row_(const std::vector<uint64_t>& row, std::vector<uint64_t>& ch, std::vector<uint64_t>& ex,
                    const uint64_t* chxor) const {
        ch.assign((size_t)std::max(CW, 1), 0);
        ex.assign((size_t)XW, 0);
        for (int w = 0; w * 64 < nchan; ++w) ch[(size_t)w] = bits64_(row, w * 64) & lowmask_(nchan - w * 64);
        if (chxor) for (int w = 0; w < CW; ++w) ch[(size_t)w] ^= chxor[w];
        for (int k = 0; k * 64 < nextra_; ++k) ex[(size_t)k] = bits64_(row, nchan + k * 64) & lowmask_(nextra_ - k * 64);
    }
    // r9 condition (a), word at a time (S8-2): the emission block's row r (cw channel words, XW extras words after the
    // nr channel rows) rejoined equals `row` over the m record columns.
    bool rejoin_ok_(const uint64_t* E, int nr, int cw, int m, int r, const std::vector<uint64_t>& row,
                    const uint64_t* chxor) const {
        const int mc = std::min(m, nchan);
        for (int w = 0; w * 64 < mc; ++w) {
            uint64_t v = E[(size_t)r * cw + (size_t)w];
            if (chxor) v ^= chxor[w];
            if ((v ^ bits64_(row, w * 64)) & lowmask_(mc - w * 64)) return false;
        }
        for (int k = 0; nchan + k * 64 < m; ++k)
            if ((E[(size_t)nr * cw + (size_t)r * XW + (size_t)k] ^ bits64_(row, nchan + k * 64)) &
                lowmask_(m - nchan - k * 64)) return false;
        return true;
    }
    // The plan's record table (cold; payload). Decodes the plan's (a, cz) from its stored key (the
    // identity-prefix content the payload was built from) and builds the canonical table over the
    // sampler's record family, then the emission split.
    // r7.3: the built-check is inline at every call site; only the cold build is out of line.
    void ensure_rec_(const CachedPlan& plan) const {
        if (XTIM_UNLIKELY(!plan.emit_built)) build_rec_cold_(plan);   // r9: the block's flag (hot line)
    }
    __attribute__((noinline, cold)) void build_rec_cold_(const CachedPlan& plan) const;   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // plancache-design: the CLASS's record table + emission block (from its representative; the parked core), then
    // the mirrors of its headers refreshed. build_rec_cold_ is this on the header's class.
    void build_class_rec_(const PlanClass& c) const;
    mutable RecTable rec_scratch_;                     // §3 B1: the builders' reusable scratch (capacities kept)
    mutable DiagNormalForm nf_scratch_;                // speed fix 5: build_class_rec_'s representative decode (reused)
    // O1 (plan §6, the selfcheck window): the arriving key's own table (build_record_table_recB of the full key)
    // against its class's table, bit for bit; a difference queues the CLASS_REPRESENTATION event (never a throw:
    // exactness holds by T3 — it is byte identity with the per-key table that moved) and counts class_rep_diffs_.
    void class_rep_check_(const CachedPlan& plan);
    mutable long class_rep_diffs_ = 0;
    // The identity plan's payload (clean shots: R = I). A plan like any other (ndrec §1.3).
    const CachedPlan& identity_plan_();   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // THE record emission (plan §1.2; ONE noinline definition, the t_row toggle target): XOR the
    // plan's record rows into this shot's bits / xbits_, drawing the coins (COIN bits) and walking
    // the record trie (BORN words); the shot's outcomes u then z land in rec_uz_ (reused). Every
    // diagonal route calls it — the clean shot, the streaming and accumulator fast paths, the slow
    // path (code review MINOR-2: the streaming path's inline copy is gone).
    // threads T2: LANE = a lane's variant (its Born-trie walk reads the shared plans' tries; lane_walk_).
    template <bool EX, bool KEEPU, bool LANE = false>
    __attribute__((noinline)) bool record_rows_(const CachedPlan& plan, uint64_t* bits);   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // r9: the record_rows_ variant of this run (set at the top of run(); one indirect call per shot).
    using RowsFn = bool (Impl::*)(const CachedPlan&, uint64_t*);
    RowsFn rows_fn_ = &Impl::record_rows_<true, true>;
    void select_rows_fn_() {
        const bool ex = nextra_ != 0, keep = retain_state;
        if (XTIM_UNLIKELY(lane_ != nullptr)) {         // threads T2: a lane's variants (T4: a recording run keeps u)
            const bool k = keep || lane_->keepu;
            rows_fn_ = ex ? (k ? &Impl::record_rows_<true, true, true> : &Impl::record_rows_<true, false, true>)
                          : (k ? &Impl::record_rows_<false, true, true> : &Impl::record_rows_<false, false, true>);
            return;
        }
        rows_fn_ = ex ? (keep ? &Impl::record_rows_<true, true> : &Impl::record_rows_<true, false>)
                      : (keep ? &Impl::record_rows_<false, true> : &Impl::record_rows_<false, false>);
    }
    bool record_rows_any_(const CachedPlan& plan, uint64_t* bits) { return (this->*rows_fn_)(plan, bits); }
    // record_report()["rec"]: the sampler's record-table builds (cold, once per payload) and the
    // Born-observable OBS draws by route (Q2: a diagonal record route never draws OBS).
    mutable long rec_tables_ = 0, rec_extras_ = 0, rec_ops_plane_ = 0, rec_ops_pivot_ = 0, rec_ops_greduce_ = 0;
    mutable long rec_ops_echelon_ = 0, rec_pivots_extras_ = 0, rec_dirs_extras_ = 0;
    mutable std::vector<long> rec_kappa_hist_;   // κ_rec of each built table
    long obs_draws_record_ = 0;
    // ⟨P, W_e⟩ for every extra (Lemma 2's t for the extras' columns).
    void ext_prefix_(const Pauli& P) {
        for (int e = 0; e < nextra_; ++e)
            if (Pauli::anticommute_bit(P, rec_cols_[(size_t)(nchan + e)].M))
                xbits_[(size_t)(e >> 6)] ^= 1ULL << (e & 63);
    }
    // Readout-flip layer (Stim M(p)).
    //
    // draw_rf (lean L5): independent Bernoulli(p_sl) per noisy record slot, skip-sampled — per group of equal p one
    // chain over its k slots: u = the next RF word as a uniform on (0, 1] (53 bits, log u finite), gap = floor(log u ·
    // inv_l) with inv_l = 1 / log1p(−p); the chain stops when the gap reaches past the group (compared as doubles
    // BEFORE any integer cast: no overflow for tiny p, a NaN stops), else the slot gap positions ahead fires and the
    // chain continues after it. P(gap = g) = (1−p)^g − (1−p)^(g+1) = p(1−p)^g: the geometric law of the non-firing
    // slots before the next firing one in i.i.d. Bernoulli(p) slots (memoryless), so the group's k slots are exactly
    // k independent Bernoulli(p) — the former gate-noise layer's method (lean N1: qeccore/noise_chain.hpp) over slots instead
    // of shots. A group's draws are its own (one keyed stream, sequential): the groups are independent. Draws per shot
    // ≤ nrf (every draw fires a slot or ends its group), so the < 2^20-draws refusal keeps its bound. The RF stream is
    // positioned at the CUMULATIVE shot (D10: route-, lane- and split-independent); the tag goes through the hash.
    //
    // apply_rf (transposed column form): iterate SET bits of the fired words via ctz; for each fired
    //   slot sl, XOR rf_col[sl] into bits and conditionally flip obs_bit_cur via rf_obs_bit[sl] —
    //   O(fired * CW) work. born_dec_rf_mask is read in emit_decisions per decision.
    // lean L7: draw_rf clears only the previous shot's fired words and lists this shot's (rf_fw_, first-set order);
    //   rf_coins stays the exact RFW-word coin vector for its full-array readers (the emit's family extras, the Born
    //   decisions), and apply_rf / er_rf_into_ / er_clean_ walk the listed words only (XOR / OR order is free).
    void rf_fire_(int32_t sl) {
        uint64_t& word = rf_coins[(size_t)(sl >> 6)];
        if (!word) rf_fw_[(size_t)rf_nfw_++] = sl >> 6;
        word |= 1ull << (sl & 63);
    }
    void draw_rf() {
        krng::Stream st = rf_s_;                       // local: the rf_coins stores cannot alias it
        st.at(shot_);
        for (int k = 0; k < rf_nfw_; ++k) rf_coins[(size_t)rf_fw_[(size_t)k]] = 0;
        rf_nfw_ = 0;
        const int32_t* slot = rf_grp_slot.data();
        const int G = (int)rf_grp_invl.size();
        for (int g = 0; g < G; ++g) {
            const int32_t e = rf_grp_off[(size_t)g + 1];
            int32_t pos = rf_grp_off[(size_t)g];
            const double il = rf_grp_invl[(size_t)g];
            if (il == 0.0) {                           // p >= 1: every slot of the group fires, no draw
                for (; pos < e; ++pos) rf_fire_(slot[pos]);
                continue;
            }
            {                                          // FCS fix 5: the group's first draw (k = e − pos slots left)
                const double u = (double)((st.next() >> 11) + 1) * 0x1.0p-53;
                if (u <= rf_grp_thr[(size_t)g]) continue;   // certainly !(gp < k): the chain ends, no log
                const double gp = std::log(u) * il;
                if (!(gp < (double)(e - pos))) continue;
                pos += (int32_t)gp;
                rf_fire_(slot[pos]);
                ++pos;
            }
            while (pos < e) {
                const double u = (double)((st.next() >> 11) + 1) * 0x1.0p-53;
                const double gp = std::log(u) * il;
                if (!(gp < (double)(e - pos))) break;  // the next firing slot lies beyond the group
                pos += (int32_t)gp;
                rf_fire_(slot[pos]);
                ++pos;
            }
        }
    }
    void apply_rf(uint64_t* bits) {
        // FIX A: column (transposed) form.  Iterate set bits of the fired words; for each fired slot sl
        // XOR rf_col[sl] (CW words) into bits and fold the born-obs bit via rf_obs_bit[sl].
        if (nrf == 0) return;
        int obs_par = 0;
        for (int k = 0; k < rf_nfw_; ++k) {
            const int w = rf_fw_[(size_t)k];
            uint64_t word = rf_coins[(size_t)w];
            while (word) {
                const int bit = __builtin_ctzll(word);
                word &= word - 1;
                const int sl = w * 64 + bit;
                const uint64_t* col = rf_col[(size_t)sl];
                for (int c = 0; c < CW; ++c) bits[(size_t)c] ^= col[(size_t)c];
                obs_par ^= rf_obs_bit[(size_t)sl];
            }
        }
        if (born_obs && obs_par) obs_bit_cur ^= 1;
    }
    // Input Pauli frames (Task 4 / J-a N4b): the port and the caller's per-shot frame rows (shots × 2·|port|,
    // X-part then Z-part); the term route composes a framed shot's frame atoms (fam_frame_take_).
    bool has_input_frame_ = false;
    std::vector<int> frame_port_qubits_;   // deferred-space qubit indices
    const uint8_t* frame_data_ = nullptr;
    int frame_stride_ = 0;
    // SEMANTICS ORACLE body (member block above). Returns false (with a stderr line) on a trip.
    bool semantic_oracle_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot);   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // The residual R = P·C applied as gates to a state (S^a, then CZ, then the prefix X^v/Z^z —
    // twirl_collapse's step-4 order; the global phase is unobservable).
    void apply_residual_gates_(FramedSuperposition& st, const DiagNormalForm& snf) const {
        for (int q = 0; q < n && q < (int)snf.a.size(); ++q)
            if (snf.a[q] & 1) st.apply_clifford(/*S*/ 1, q, 0);
        for (const std::pair<int, int>& e : snf.cz)
            st.apply_clifford(/*CZ*/ 7, e.first, e.second);
        for (int q = 0; q < n; ++q) {
            if (snf.prefix.xbit(q)) st.apply_clifford(/*X*/ 3, q, 0);
            if (snf.prefix.zbit(q)) st.apply_clifford(/*Z*/ 5, q, 0);
        }
    }
    // The Born observable's noiseless REFERENCE parity (Stim's reference_sample convention, the
    // exact engine's refobs — framed_reference_records on the same bare state). The engine emits
    // the ABSOLUTE Born bit; the reporting layer (xtim_py's sample / sample_barrier) XORs this
    // into the column after run(), exactly as it applies chan_ref to the deterministic channels,
    // so the column is reference-relative like engine="exact" (audit finding K: auto flipped the
    // observable's meaning with the engine). Kept OFF the per-shot emit seam on purpose: an XOR in
    // emit_shot cost 5-17% twirl µs/shot through codegen alone. 0 for a carried input port
    // (ABSOLUTE by contract, see NOISELESS-REFERENCE SEMANTICS) and when no Born observable is
    // active. obs_bit_cur and the selfcheck statistics stay ABSOLUTE (the absolute, inverted Born observable).
    // Declared LAST in Impl so no hot member's offset moves (run() stays instruction-identical).
    int born_ref_ = 0;
    // S5a: record_report()["structural_shots"] (reset by set_seed). Declared LAST (no hot member's
    // offset moves).
    long structural_shots_ = 0;
    // J-a (r2.1, J1): the run_w instance run() calls — FW by CW (8 iff CW ≤ 8). pick_run_() is the ONE rule; the
    // ctor re-applies it once CW is final. Declared LAST so no hot member's offset moves.
    using RunFn = bool (Impl::*)(long);
    RunFn pick_run_() const;
    RunFn run_fn_ = pick_run_();
    // ── J-a (N1): the FAM sampler's state (switched OFF: empty / null on every production sampler).
    // Declared LAST so no hot member's offset moves. FAM-only code lives in twirl_sampler_fam.cpp.
    //   famx_: the family extras — a gauge / ANTI detector or an extra observable classified as a record
    //          extra under its class bit (kind 0 / 1 / 2), storage columns nchan + famx_base_ + i (after
    //          the Born observable and the Born decisions); famx_off_ = inv ⊕ ref, famx_rf_ the RFW-word
    //          readout-flip slot masks (FamilySink).
    struct FamX { Pauli M; int order; int kind; int index; std::vector<int> recs; };
    std::vector<FamX> famx_;
    int famx_base_ = 0;
    std::vector<uint8_t> famx_kind_, famx_off_;
    std::vector<int> famx_out_;
    std::vector<uint64_t> famx_rf_;
    FamilySink* family_sink_ = nullptr;
    bool oracle_all_shots_ = false;
    // perf-engine §2.6: the current shot's self-check window (run_w sets it at the top of every shot and leaves it true
    // when it returns, so a build reached outside a run keeps the full checks). Gates the emission-block rejoin check
    // of the record-table builds (build_rec_cold_ / build_k2): outputs are identical either way, the check only throws.
    bool in_window_ = true;
    std::string family_refusal_;
    long sem_class_[6] = {0, 0, 0, 0, 0, 0};   // semantic-oracle checks per record class
    void fam_setup_(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                    const TwirlRecordOptions& opt, const std::vector<ErrorChannel>& chans);
    bool fam_run_guard_(long shots);
    void fam_refuse_(const std::string& m);
    // J-b1 P2 (twirl_sampler_fam.cpp, out of line and cold): a FAM plan that fell back (engine bug: the slow path
    // builds diagonal-class normal forms only).
    [[noreturn]] void fam_plan_fallback_bug_() const;
    void fam_add_(int kind, int index, const std::vector<int>& recs, const std::vector<std::pair<int, int>>& reads);
    void fam_cols_();
    bool semantic_oracle_fam_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot);
    bool rot_state_oracle_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot);   // lean D1-S3
    bool sem_project_(long shot, const char* route);
    // ── J-a (N2): the term route (qeccore/term_residual.hpp), built by fam_setup_; null on production
    // samplers. fam_term_route_ attaches the per-alternative (P_f, S_f) table on first use.
    // Owned through FamTermHold, whose destructor is OUT OF LINE (twirl_sampler_fam.cpp): the production
    // TU's ~Impl / ctor unwind paths gain one call, not an inlined shared_ptr release — an inlined release
    // grew twirl_sampler.cpp's inline-unit-growth accounting enough to flip a run_w<0, false> inlining
    // decision (push_back), i.e. production codegen.
    // N4a: the holder owns the FAM route state (FamRoute, defined in twirl_sampler_fam.cpp: the TermResidual
    // and the kind-2 route's dispatch table, plan headers and scratch) — still ONE pointer, so the layout is
    // unchanged.
    struct FamRoute;
    struct FamBranch;                    // J-b1 P9b: a registered carried branch (twirl_sampler_fam.cpp)
    // threads T1: a copy holds NO route (a lane gets its own small one: fam_lane_route_, twirl_sampler_fam.cpp).
    struct FamTermHold {
        FamRoute* p = nullptr;
        FamTermHold() = default;
        FamTermHold(const FamTermHold&) noexcept {}
        FamTermHold& operator=(const FamTermHold&) = delete;
        ~FamTermHold();
    };
    FamTermHold fam_term_;
    TermResidual& fam_term_route_();
    TermResidual& fam_term_axes_() const;   // lean D1-S2: the axes for the kind-2 content readers (any view)
    // ── J-a (N4a): the kind-2 route of a FAM sampler with the ppr_records class (run_w<*>, at the PPR-route
    // decision; twirl_sampler_fam.cpp). A PPR-class shot: compose_terms_ → reduce_key_ → kind_of_; kind 0 → the
    // shared normal form → the diagonal plan cache's kind-0 plan (builder D); kind 2 → the (G) table of the
    // reduced term set (a sampler-owned plan header, its payload charged to and evicted by the plan cache);
    // then the channel prefix rows of P, record_rows_ and t(P) for the extras. False: refused by name.
    bool fam_term_shot_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, uint64_t* bits);
    bool fam_term_oracle_(long shot);    // the term-view semantic oracle of a term-routed shot
    bool fam_project_extras_(long shot); // the oracle's per-class counts and family-extra projections
    // ── J-a (N4b): input frames on a FAM sampler with the frames class (twirl_sampler_fam.cpp). Every frame
    // component (X_i / Z_i on a port wire at t = 0) is an atom (P_F, S_F) of the term view, composed as the
    // RIGHTMOST factor of the shot's residual (R_total = R_noise · R_f, design §2.7 / D4(a)): a clean component's
    // fold P_rec = P·P″(S) and a poisoned component's terms (S_F into S) are that one composition. A framed shot
    // takes the term route (every kind); the static relabel, the dressing and frame_born_flip are off for the
    // whole run (fam_frame_take_ computes the shot's frame row).
    void fam_frame_take_(long done);      // J-b1 P9b: `done` = the shot's index in this run (input_branch)
    bool fam_frames_on_() const;          // FCS fix 4: the frames class (fam_frame_take_ is a no-op without it; per run)
    bool fam_term_all_() const;          // the shot takes the term route: framed (N4b) or an all-term recording run
    // J-b1 P9b: record_rows_ for a branch shot — the same emission block and coins, the BRANCH's Born trie
    // (its sparse (plan, branch) map). Cold relative to branch 0 (branch shots only).
    bool fam_record_rows_branch_(const CachedPlan& plan, uint64_t* bits, FamBranch& B);
    void fam_branch_payload_(FamBranch& B);   // D6: re-supply on a miss + LRU eviction under the branch budget
    // J-b1 P4: a recording run whose diagonal shots the (D) paths serve (set by fam_run_guard_ per run), and the
    // ONE (D)-side writer of a shot's layout-3 kind-0 record (plan content, P_rec = the canonical prefix, y).
    bool fam_rec_d_() const;
    void fam_rec_write_d_(const CachedPlan& plan, const Pauli& prefix);
    void fam_arm_terms_(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads);   // test API
    // J-b1 P5: the term view at construction (fam_setup_), on the ctor's alternatives (`sides` and the channels'
    // live flags; J-b1 F: twirl_alternatives_ is the one copy, built before fam_setup_): the
    // owner's non-Clifford rule (the static fixed-axis merge, then the commutation check; refused by name with
    // setup_error 7). Kept iff the term route can be armed (ppr_records on CH/PPR, frames with a port, the
    // all-term recording option), else only the verdict is kept (R9a).
    bool fam_terms_setup_(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                          const std::vector<ErrorChannel>& chans);
    // J-b1 P6: a FAM (D) plan's record table by record-space B (build_record_table_recB; J-b2 J7b: the only
    // builder, the σ dual basis is deleted). Defined in twirl_sampler_fam.cpp.
    void build_rec_table_fam_(const PlanClass& c, const DiagNormalForm& nf0, RecTable& out) const;
    // ── early-reject (OD-2026-10-25-early-reject-stage): the opt-in postselection mask. Declared LAST (no hot member's
    // offset moves); the state lives behind one pointer, null without a mask (the run then takes run_fn_ unchanged).
    // Per shot, before any plan lookup / build / Born draw: the shot's combined Clifford part C (the diagonal routes'
    // canonical plan key; the term route's reduced S, C = rot(S)) and its Pauli part P (the channel rows the engine
    // already XORs). A flagged deterministic detector channel c whose operator M_c shares no qubit with C has the
    // DETERMINISTIC emitted value ⟨P, M_c⟩ ⊕ rf_c ⊕ K_c (K = the clean shot's emitted value): if that is 1 the shot
    // is rejected at once. Sound and incomplete; every emitted shot then passes the exact late filter on its packed
    // detector row (gauge / ANTI detectors and C-touched channels included). shot_ advances for every shot, so the
    // accepted shots keep their counter-based streams (byte-identical to an unmasked run filtered afterwards).
    struct ErState {
        std::vector<uint64_t> F;        // CW words: the flagged deterministic detector channels the early test reads
        std::vector<uint64_t> K;        // CW words: their clean-shot emitted values (identity plan base ⊕ cref) & F
        std::vector<uint64_t> term;     // T × CW: flagged channels whose operator shares a qubit with term axis B_j
        // lean D1-S2 / FCS fix 1 (rot_on_): per flagged channel c the rotations that touch it, as a TW-word mask (row
        // ft_idx[c] of ft; -1: no rotation touches c). The column read rejects iff some fired flagged channel c has
        // ft[c] & s = 0 (no set rotation touches it), stopping at the first.
        std::vector<int> ft_idx;
        std::vector<uint64_t> ft;
        std::vector<uint64_t> tmp, hit; // CW scratch
        std::vector<int> dbyte;         // late filter: the packed detector bytes holding a flagged detector
        std::vector<uint8_t> dmask;     //              and their masks
        bool full = false;              // every flagged detector is an early-set channel (F)
        bool clean_ok = false;          // K & F = 0 (a clean shot without readout flips passes the early test)
        bool rejected = false;          // the term route rejected the current shot (read and cleared by run_w)
        bool anyF = false;              // F != 0: K and term were derived (perf-glue §4.2: the re-arm path)
        long early = 0, late = 0;       // cumulative since set_postselect
    };
    struct ErHold {
        ErState* p = nullptr;           // the armed mask's state (null: no mask); when set, p == memo
        // perf-glue §4.2: the last built state and its flags, kept across calls (the binding sets the mask per call):
        // set_postselect with the same flags re-arms it (counters reset) instead of rebuilding it.
        ErState* memo = nullptr;
        std::vector<uint8_t> memo_flags;
        long builds = 0;
        ErHold() = default;
        // threads T1: a copy owns a copy of the ARMED state (a lane's own scratch and counters); no memo.
        ErHold(const ErHold& o) : builds(o.builds) { if (o.p) { memo = new ErState(*o.p); p = memo; } }
        ErHold& operator=(const ErHold&) = delete;
        ~ErHold();                      // twirl_sampler_er.cpp (out of line: no inlined release in the other TUs)
    };
    ErHold er_;
    void er_set_(const std::vector<uint8_t>& det_flags);   // twirl_sampler_er.cpp (cold)
    bool er_term_reject_();             // twirl_sampler_er.cpp: the term route's test on (F.sh.P, F.sh.S)
    void fam_rec_rollback_();           // twirl_sampler_fam.cpp: drop the last layout-3 record row (late filter)
    void fam_er_terms_();               // twirl_sampler_fam.cpp: ErState::term from the armed term view (cold)
    // lean D1-S3: the early test on the Clifford path (t = 0): `pb` = the shot's t(P) channel words; C = I, so every flagged
    // deterministic channel's value is final. True: one of them fires.
    __attribute__((always_inline)) inline bool er_cliff_reject_(const uint64_t* pb) {
        ErState& R = *er_.p;
        uint64_t any = 0;
        if (!rf_nfw_) {                               // FCS fix 8: no readout flip: one fused pass, no scratch
            for (int w = 0; w < CW; ++w) any |= (pb[w] ^ R.K[(size_t)w]) & R.F[(size_t)w];
            return any != 0;
        }
        uint64_t* t = R.tmp.data();
        for (int w = 0; w < CW; ++w) t[w] = pb[w] ^ R.K[(size_t)w];
        er_rf_into_(t);
        for (int w = 0; w < CW; ++w) any |= t[w] & R.F[(size_t)w];
        return any != 0;
    }
    // The rotation loop's COLUMN READ (rot_shot_; FCS fix 8: also the shot-block pre-pass, run_w): `pb` = the shot's t(P)
    // channel words, `sw` = its key words. t = (pb ⊕ K ⊕ rf) & F; FCS fix 1: true at the first fired flagged channel that
    // no SET rotation touches (its emitted value is final: no plan entry reaches it).
    __attribute__((always_inline)) inline bool er_rot_reject_(const uint64_t* pb, const uint64_t* sw) {
        ErState& E = *er_.p;
        if (!E.anyF) return false;
        const int TW = rot.TW;
        uint64_t* t = E.tmp.data();
        uint64_t any = 0;
        for (int w = 0; w < CW; ++w) t[w] = pb[w] ^ E.K[(size_t)w];
        er_rf_into_(t);
        for (int w = 0; w < CW; ++w) { t[w] &= E.F[(size_t)w]; any |= t[w]; }
        if (!any) return false;
        for (int w = 0; w < CW; ++w)
            for (uint64_t word = t[w]; word; word &= word - 1) {
                const int ix = E.ft_idx[(size_t)(w * 64 + __builtin_ctzll(word))];
                if (ix < 0) return true;              // no rotation touches it: final
                const uint64_t* m = &E.ft[(size_t)ix * (size_t)TW];
                uint64_t hit = 0;
                for (int k = 0; k < TW; ++k) hit |= m[k] & sw[k];
                if (!hit) return true;                // no SET rotation touches it: final
            }
        return false;
    }
    // The early test on a clean shot (C = I, P = I): only the readout flips can fire a flagged channel. 1: rejected;
    // 0: provably accepted (every flagged detector is an early-set channel and none fires: no late filter needed);
    // 2: not rejected early (the late filter decides).
    __attribute__((always_inline)) inline int er_clean_() {
        ErState& R = *er_.p;
        if (R.clean_ok) {                   // K & F = 0: without a readout flip the shot passes the early test
            if (!rf_nfw_) return R.full ? 0 : 2;
        }
        uint64_t* t = R.tmp.data();
        for (int w = 0; w < CW; ++w) t[w] = R.K[(size_t)w];
        er_rf_into_(t);
        for (int w = 0; w < CW; ++w)
            if (t[w] & R.F[(size_t)w]) return 1;
        return R.full ? 0 : 2;
    }
    // This shot's readout-flip contribution to the channels (apply_rf's columns, read early).
    __attribute__((always_inline)) inline void er_rf_into_(uint64_t* t) const {
        for (int k = 0; k < rf_nfw_; ++k) {
            const int w = rf_fw_[(size_t)k];
            for (uint64_t word = rf_coins[(size_t)w]; word; word &= word - 1) {
                const uint64_t* col = rf_col[(size_t)(w * 64 + __builtin_ctzll(word))];
                for (int c = 0; c < CW; ++c) t[c] ^= col[c];
            }
        }
    }
    // The late filter (every emitted shot of a masked run): a flagged detector set in the emitted row → the row is
    // cleared and taken back (and its layout-3 record, on a recording run).
    __attribute__((always_inline)) inline void er_late_() {
        FamilySink* fs = family_sink_;
        ErState& R = *er_.p;
        if (!fs || R.dbyte.empty()) return;
        const long r = fs->row - 1;
        const uint8_t* dr = fs->dets + (size_t)r * (size_t)fs->DBB;
        uint8_t bad = 0;
        for (size_t i = 0; i < R.dbyte.size(); ++i) bad |= (uint8_t)(dr[R.dbyte[i]] & R.dmask[i]);
        if (!bad) return;
        er_late_drop_();
    }
    void er_late_drop_();               // twirl_sampler_er.cpp (out of line)

    // ── threads (fixplan threads/fixplan.md) ──────────────────────────────────────────────────────────────────
    // T1: a copy of the sampler (a LANE) shares Frozen and the caller-owned bare state, copies every per-shot member.
    Impl(const Impl&) = default;
    Impl& operator=(const Impl&) = delete;
    // T2: a threaded call (twirl_sampler_threads.cpp). The home (this sampler) draws the noise of a super-block — the
    // same stream a serial call draws, in the same order — and its lanes (copies of it: Frozen shared, per-shot state
    // their own) take disjoint blocks of it, each shot at its global index (the keyed streams address (seed, shot)),
    // each row at its global row. During a super-block the home's plan cache and its plans are read only: a lane serves
    // what it holds and builds the rest into its own overlay cache; a Born-trie node the shared trie lacks is evaluated
    // on a lane copy of that trie. At the barrier the home adopts the overlays (lane order) and merges the tries; every
    // value is a pure function of its key, so who built it moves no byte. threads = 1: run() (unchanged).
    static constexpr int kLaneRun = 1 << 13;      // run_w<FW | kLaneRun>: a lane's instance of width FW
    static constexpr int kDeferRun = 1 << 14;     // run_w<FW | kDeferRun>: the home's run of its lanes' deferred shots
    struct NoiseSpan;                              // lean N1 (below)
    struct LaneCtx {
        const TwirlPlanCache* shared = nullptr;    // the home's plan cache (read only during a super-block)
        std::unordered_map<const PlanClass*, std::vector<BornNode>> cow;   // grown copies of home classes' tries
        FamilySink sink;                           // the lane's copy of the home's sink (rows at their global rows)
        const NoiseSpan* span = nullptr;           // lean N1: the home's super-block of noise events (whole blocks)
        std::atomic<long>* next = nullptr;         // the next block of the super-block (one per call)
        const std::atomic<bool>* stop = nullptr;   // set by a lane that failed: no lane takes a new block
        long nblocks = 0, B = 0, S_sb = 0;         // blocks, block size, super-block shots
        uint64_t shot0 = 0;                        // cumulative shot index of super-block shot 0
        long done0 = 0, row0 = 0;                  // its index in the call, its sink row
        bool count_channels = false;               // the home run's channel counting (the window run counts)
        long cur_block = -1;                       // the block being run (the failure's block)
        // T3 / T4: every shot runs at its POSITIONAL row (row0 + its super-block index); acc[s] (one byte per
        // super-block shot, shared by the lanes of a call: disjoint indices) records whether it emitted a row
        // (kAccLane / kAccHome) or was deferred (kAccDefer); the home compacts the postselected rows in shot order
        // and remaps the lanes' record content handles at the barrier.
        uint8_t* acc = nullptr;
        uint8_t mark = 1;                          // the emitted mark this source writes (kAccLane / kAccHome)
        bool pend = false;                         // cur_s ran and its flag is not yet written
        long s0_cur = 0, cur_s = 0, cur_pos = 0;   // the block's first super-block shot, the shot, its row
        std::vector<long> blocks;                  // the blocks this source ran (the record remap's owner map)
        std::vector<long> deferred;                // a lane: the super-block shots it deferred (the term route)
        std::vector<long> rpend;                   // fid-routine B4: its record rows the premap left (shot indices)
        const std::vector<long>* defer_list = nullptr;   // the home's deferred run: the shots to run, in order
        TwirlRecordSampler::FamilyRecordSink* rs = nullptr;   // the run's record sink (positional rows), or null
        TwirlRecordSampler::FamilyRecordSink rsink;   // a lane's copy of the home's record sink
        bool keepu = false;                        // a recording run: record_rows_ keeps the outcomes
        std::vector<std::pair<long, size_t>> ev_marks;   // (block, diag_ size at its start)
        uint32_t home_plans = 0;                        // scaling F2: plan ids <= this are the home cache's
        // plancache-design §2 R2: the lane's hit marks over the HOME classes (id − 1), OR-folded at the barrier; a
        // lane's own (overlay) classes are marked on the class itself
        uint32_t home_classes = 0;
        std::vector<uint8_t> marks;                     // (all zero between super-blocks: the fold resets what it read)
        std::vector<uint32_t> touched;                  // speed fix 5: the marked indices (the fold reads these only;
                                                        //   reserved to home_classes at the setup: no hit-path allocation)
    };
    LaneCtx* lane_ = nullptr;                      // non-null: this Impl is a lane (owned by its home's LaneSet)
    LaneCtx* blk_ = nullptr;                       // the block source of a lane / the home's deferred run
    static constexpr uint8_t kAccNone = 0, kAccLane = 1, kAccDefer = 2, kAccHome = 3;
    // The shot's positional row; the previous shot's emitted flag (a row emitted iff the sink advanced past it).
    __attribute__((always_inline)) inline void lane_shot_begin_(int s) {
        LaneCtx& C = *blk_;
        if (C.pend) lane_shot_end_();
        C.cur_s = C.s0_cur + s;
        C.cur_pos = C.row0 + C.cur_s;
        if (family_sink_) family_sink_->row = C.cur_pos;
        if (C.rs) C.rs->row = C.cur_pos;
        C.pend = true;
    }
    __attribute__((always_inline)) inline void lane_shot_end_() {
        LaneCtx& C = *blk_;
        const long row = family_sink_ ? family_sink_->row : C.rs ? C.rs->row : -1;   // (no sink: no rows)
        C.acc[C.cur_s] = row == C.cur_pos + 1 ? C.mark : kAccNone;
        C.pend = false;
        if (XTIM_UNLIKELY(g_twirl_shot_end_test_hook != nullptr))
            g_twirl_shot_end_test_hook(C.shot0 + (uint64_t)C.cur_s, C.mark == kAccLane);
    }
    void lane_defer_() {                           // a lane: the current shot runs on the home after the barrier
        LaneCtx& C = *blk_;
        C.acc[C.cur_s] = kAccDefer;
        C.deferred.push_back(C.cur_s);
        C.pend = false;
    }
    struct LaneSet;                                // the home's lanes (twirl_sampler_threads.cpp)
    struct LaneSetHold {                           // a copy holds none
        LaneSet* p = nullptr;
        LaneSetHold() = default;
        LaneSetHold(const LaneSetHold&) noexcept {}
        LaneSetHold& operator=(const LaneSetHold&) = delete;
        ~LaneSetHold();
    };
    LaneSetHold lanes_;
    bool window_stop_ = false;                     // the serial window prefix of a threaded call (run_w)
    long last_run_done_ = 0;                       // the shots the last run_w call ran
    int threads_req_ = 1, threads_eff_ = 1;        // the last call's requested / effective thread count (D3)
    long threads_budget_fallbacks_ = 0;            // fix F2: super-blocks re-run serially after a lane's budget refusal
    int64_t lane_bytes_test_ = 0;                  // test API: the D3 per-lane estimate overridden (0: the estimate)
    long threads_deferred_ = 0;                    // the last call's shots its lanes deferred to the home (term route)
    std::string threads_limit_;                    // why the effective count is lower ("" = not lowered)
    bool run_threads_(long shots, int threads);
    // scaling fix-2 F1 / fix-3 (coordinator rulings): a FAILED call commits nothing. CallState is the sampler at a
    // call's entry, before any reseed: the seed (the keyed streams' keys; lean N1: the noise is addressed by it), the cumulative
    // shot index, the spent-selfcheck latch and every per-SHOT counter a report exposes (used / clean / diag / checked,
    // the route split, obs_ones / ps_acc / ps_obs_ones, the oracle counts, the channel 1-counts, the Born walks and
    // steps, the timings, the term route's shot counts). Not restored (they count work whose results stay cached and
    // are documented as including failed attempts): plan-cache and Born-trie build / node counters, record-table
    // build counters, the term route's plan / build / key counters (twirl_sampler_threads.cpp).
    struct FamShotStats {                          // the term route's per-shot counters (fam.cpp)
        long shots_kind0 = 0, shots_kind2 = 0, oracle_term_shots = 0, shots_framed = 0, shots_record_d = 0,
             shots_kind0_general = 0;
        std::vector<long> plan_shots;
        long branch_probes = 0;                    // fix-4: the carried branches' map probes and per-branch shots
        std::vector<long> branch_shots;
    };
    // fix-4 (B): the cached-work counters set_seed zeroes. A failed call that reseeded restores them to their value at
    // entry PLUS the work the call did (they only ever include failed attempts and never lose history).
    struct CachedWork {
        long nodes = 0, node_evals = 0, chi_j = 0, passes = 0, max_support = 0;
        int max_depth = 0;
        long rec_tables = 0, rec_extras = 0, rec_ops_plane = 0, rec_ops_pivot = 0, rec_ops_greduce = 0,
             rec_ops_echelon = 0, rec_pivots_extras = 0, rec_dirs_extras = 0;
        std::vector<long> kappa_hist;
    };
    uint64_t seed_epoch_ = 0;
    struct CallState {
        uint64_t seed = 0, done_shots = 0;
        bool selfcheck_spent = false;
        long clean = 0, diag = 0, checked = 0, used = 0, route[R_COUNT] = {}, fast_derived = 0, structural = 0,
             fast_born = 0, obs_ones = 0, ps_acc = 0, ps_obs_ones = 0, sem_checked = 0, sem_class[6] = {},
             trie_walks = 0, trie_steps = 0, obs_draws_record = 0;
        std::vector<long long> chancnt;
        double t_draw = 0.0, wall = 0.0;
        FamShotStats fam;
        uint64_t seed_epoch = 0;
        CachedWork cached;
        int threads_req = 1, threads_eff = 1;      // fix-4 (E): the last call's thread report
        long threads_deferred = 0, threads_budget_fallbacks = 0;
        std::string threads_limit;
    };
    void call_save_(CallState& c) const;
    void call_restore_(const CallState& c) noexcept;
    void fam_shot_stats_save_(FamShotStats& f) const;          // fam.cpp
    void fam_shot_stats_restore_(const FamShotStats& f) noexcept;
    bool run_committed_(long shots, int threads);  // run_threads_ under the call contract (no state on failure)
    twirl_lane::HomeOnly<CallState> entry_;
    void fam_term_reset_shot_() noexcept;          // fam.cpp: the term route's last-term-shot mark cleared
    const char* lanes_unsupported_() const;        // twirl_sampler_fam.cpp: a run the lanes do not serve (serial)
    void fam_lane_route_(Impl& lane) const;        // twirl_sampler_fam.cpp: the lane's own small FamRoute
    void fam_lane_merge_(Impl& lane);              // twirl_sampler_fam.cpp: the lane route's counters into the home's
    void fam_lane_call_(Impl& lane, LaneCtx& C) const;   // fam.cpp: a call's lane record sink / content table
    bool fam_lane_frame_defer_(long done) const;   // fam.cpp: a lane's shot that supplies a frame or a carried branch
    TwirlRecordSampler::FamilyRecordSink* fam_record_sink_() const;   // fam.cpp: the run's record sink (or null)
    // fam.cpp: the lanes' record rows of a super-block (acc == kAccLane), their lane-local content handles to the
    // home's, interned in shot order (the serial order of first sight); the lanes' content tables cleared.
    // fid-routine B4: `pend` non-null = the lanes premapped their rows (fam_lane_premap_): only the listed super-block
    // shots (ascending) are left, the rows of contents new to the home.
    void fam_lane_remap_(const uint8_t* acc, long row0, long S_sb, long B, const std::vector<int>& owner,
                         const std::vector<std::unique_ptr<Impl>>& lanes, const std::vector<long>* pend = nullptr);
    // fam.cpp (fid-routine B4): a lane, after fam_lane_resolve_, maps its own blocks' record rows whose home handle is
    // known (a home plan's row, or a content the home held at the resolve) — rows no other thread touches; the rest
    // (contents new to the home: interned in shot order by the remap) go to C.rpend.
    void fam_lane_premap_(const Impl& home, const uint8_t* acc, long row0, long S_sb, long B, LaneCtx& C);
    // fam.cpp (scaling F2): a lane, at the end of its super-block work, resolves its contents against the home's
    // content index (read only then): the home handle of each, or -1 (new to the home) — the remap then hashes only
    // the new ones, moving the lane's key and content in.
    void fam_lane_resolve_(const Impl& home);
    static void record_row_move_(TwirlRecordSampler::FamilyRecordSink& R, long dst, long src);
    static void record_row_clear_(TwirlRecordSampler::FamilyRecordSink& R, long r);
    bool lane_take_block_(int& S, long& done);     // a lane: the next block of the super-block (false: none left)
    int lane_walk_(const CachedPlan& plan, const RecBlob& T);    // a lane's Born-trie walk (twirl_sampler.cpp)
    // A lane's plan lookup: the home's resident plan (read only), else the lane's overlay (cache).
    __attribute__((always_inline)) inline const CachedPlan* lane_find_(const uint64_t* key, uint64_t h) const {
        const CachedPlan* p = lane_->shared->find(key, h);
        if (p && p->resident && (p->emit_built || p->fallback)) return p;
        return cache->find(key, h);
    }
    // §2 R2: the second-chance hit mark of a served plan's class — one byte store (the home: on the class; a lane: a
    // lane-local array over the home classes, OR-folded at the barrier; its own classes on the class).
    template <bool LANE>
    __attribute__((always_inline)) inline void hit_mark_(const CachedPlan& p) {
        PlanClass* c = p.cls;
        if constexpr (LANE) {
            const uint32_t i = c->id;
            if (i <= blk_->home_classes) {
                uint8_t& m = blk_->marks[(size_t)i - 1];
                if (!m) { m = 1; blk_->touched.push_back(i - 1); }
            } else {
                c->mark = 1;
            }
        } else {
            c->mark = 1;
        }
    }
    const CachedPlan* lane_get_or_build_(const uint64_t* key, uint64_t h, const DiagNormalForm& nf) {
        const CachedPlan* p = lane_->shared->find(key, h);
        if (p && p->resident && (p->emit_built || p->fallback)) return p;
        return cache->get_or_build(G, bare, key, h, nf, false, nullptr, shot_);
    }
    // T1: warm every lazily built read-only state a lane would read (the group planes' token / z-axis / G_z cache /
    // chunk table, the bare frame's dual tableau, the identity plan) and give every resident diagonal plan its
    // emission block, so a lane reads them without writing (twirl_sampler.cpp).
    void freeze_();

    // ── P3 S-b: the SHOT BLOCK front end of a serial run (twirl_sampler_runw.inc.hpp). lean N1: the run's noise block
    // (NoiseModel::walk on the run's table) and each event applied where it is drawn: its alternative's rotation record
    // XORed into the shot's rows — the channel row
    // (CW words), the key row (TW words, then the token hash h; FNV offset basis ⊕ the tokens, as rot_shot_) and the
    // extras row (XR words) — and the shot's event count. No event list, no shot sort, no per-shot event loop; per shot
    // run_w copies the rows and takes the same route (rotation tail on the key row / the Clifford path) and tail.
    // Serial runs outside the window, not recording, without frames or bench replay (sb_ok_); every other run (lanes,
    // deferred shots, the window, recording / term-recorded / framed runs) keeps the event list.
    std::vector<uint64_t> sb_rows_, sb_key_, sb_xb_;
    std::vector<int32_t> sb_nev_;
    int sb_B_ = 0;                                   // the rows' shot capacity (4096; fewer only past 8 MB of rows)
    // FCS fix 8: a masked shot-block run's pre-pass verdicts (sb_B_ rows): 0 the shot loop runs the shot; 1 rejected
    // early as a clean shot (readout flips), 2 rejected early by the column read / the Clifford early test (a diag shot)
    std::vector<uint8_t> sb_rej_;
    // FCS round 2 (b): channel rows [0, sb_clean_) are all zero. A shot consumes its channel row by moving it out (the
    // shot loop's copy into `bits` zeroes it; the pre-pass zeroes a row it rejects), so after a block's loop completes
    // every row it filled is zero again and sb_fill_ clears only the rows past sb_clean_ (any early exit leaves
    // sb_clean_ = 0: the next fill clears them all).
    int sb_clean_ = 0;
    // FCS round 2 (a): the construction's noiseless reference records (noiseless_reference_records(bare, deferred, reads,
    // rec_invert): a pure function of the ctor's arguments), computed at most ONCE — the Born observable's reference
    // (ctor) and the family extras' / port reference (fam_setup_) read the same vector; released after fam_setup_.
    std::vector<uint8_t> ctor_ref_;
    bool ctor_ref_ok_ = false;
    const std::vector<uint8_t>& ctor_ref_records_(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                                                  const std::vector<uint8_t>& rec_invert);
    int sb_zero_ = 0;                                // FCS fix 2: rows [0, sb_zero_) hold a no-event fill (a table with
                                                     // no mechanism leaves them so: sb_fill_ skips the refill)
    bool sb_off_ = false;                            // test API: serial runs take the event list (the oracle route)
    long sb_shots_ = 0;                              // test API: shots served from a shot block
    bool sb_ok_() const;
    void sb_size_();
    void sb_fill_(const NoiseTable& T, uint64_t c0, int S);   // the rows of cumulative shots [c0, c0 + S) (one block)

    // ── lean N1: the noise blocks (qeccore/noise_chain.hpp). The run's table: the FULL channel table on a run that reads
    // P_f itself (a recording run, a framed / term-recorded sampler), else the PRUNED mechanism table (decided per run;
    // the window's shots read the run's table). A serial run keeps its last generated block (chain order) for the next
    // call / the next route of the same block: a pure cache keyed by (seed, table, block).
    struct NoiseBlock {
        bool ok = false;
        uint64_t seed = 0, b = 0;
        const NoiseTable* T = nullptr;
        std::vector<uint64_t> ev;                    // (alt << 32 | s), chain order
    };
    twirl_lane::HomeOnly<NoiseBlock> nblk_;
    bool noise_full_test_ = false;                   // test API: every run draws the full table (the recording law)
    std::vector<int32_t> nb_cnt_;                    // shot_major's scratch
    const NoiseTable& run_table_() const;            // twirl_sampler_fam.cpp
    const std::vector<uint64_t>& nb_block_(const NoiseTable& T, uint64_t b);
    // the event list of cumulative shots [c0, c0 + S) (inside one block): shot-major, per shot ascending alternative
    void nb_events_(const NoiseTable& T, uint64_t c0, int S, DiagErrorSampler::SparseBatch& out);
    // threads: a super-block's noise, whole blocks shot-major (twirl_sampler_threads.cpp)
    struct NoiseBlockEv {
        uint64_t b = 0;
        std::vector<int32_t> off, ev;                // off: kB + 1 shot offsets; ev: alternatives
    };
    struct NoiseSpan {
        uint64_t c0 = 0;                             // cumulative index of the super-block's first shot
        uint64_t b0 = 0;                             // its first block
        std::vector<std::shared_ptr<NoiseBlockEv>> blk;   // blocks b0, b0 + 1, ...
        // shots [c, c + S) of the span (c cumulative) appended into out (shot_off rebased at out's end)
        void gather(uint64_t c, int S, DiagErrorSampler::SparseBatch& out) const;
    };

    // The construction state shared by the sampler and its lanes (immutable after the constructor, except rot_plist:
    // built on demand on the home thread before any lane starts).
    struct Frozen {
        std::vector<Pauli> gens;
        CertifiedGroupPlanes G;
        PropagationTable table;
        std::vector<SChan> sch;
        std::vector<RecColumn> rec_cols_;
        RecColIndex rec_col_index_;                         // lean L4: the record columns' transpose (record-space B)
        std::vector<std::vector<FiredPauli>> sides;
        std::vector<uint8_t> alt_is_ppr;                    // the alternatives touching a PPR-retry atom (the window's comparator)
        std::shared_ptr<const PlanClassifier> classifier;   // plancache-design: Π (shared with the lanes' overlays)
        RotationModel rot;                                  // lean D1-S1: the rotation model (compiled once)
        RotationRecords rot_rec;                            // lean D1-S1: the compact per-alternative rotation records
        RotPList rot_plist;                                 // lean D1-S2: the sparse P supports (on demand, home only)
        NoiseModel noise;                                   // lean N1: the pruned / full mechanism tables
    };
};

}  // namespace qeccore
