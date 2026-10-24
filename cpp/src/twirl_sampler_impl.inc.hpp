// twirl_sampler_impl.inc.hpp — INTERNAL to the twirl sampler (J-a r2.1, JA-1). The definition of
// TwirlRecordSampler::Impl and the file-local helpers its members and run_w use, shared by the two
// translation units: twirl_sampler.cpp (the ctor and the shared members) and twirl_sampler_fam.cpp (the run_w
// instances, J-b2 J3 / J9). Include from those two TUs only.
#pragma once

#include "qeccore/twirl_sampler.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/record_space_b.hpp"   // J-b2 J7a: the readers' record table (rec_table_of_)

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


// J-b1 F (F item 2): the deferred noise stream's channels and per-alternative fired lists (twirl_sampler_fam.cpp; the
// ctor's one copy, compiled with -Werror=switch over NoiseChannel).
void twirl_alternatives_(const Circuit& deferred, double factor, std::vector<ErrorChannel>& chans,
                         std::vector<std::vector<FiredPauli>>& sides);

struct TwirlRecordSampler::Impl {
    // ── Inputs / configuration ────────────────────────────────────────────────────────────
    const FramedSuperposition& bare;           // caller-owned; must outlive the sampler
    const int n;
    std::string disk_path;                     // non-const since V3-T3 R2: may be auto-derived
    std::vector<TwirlRecordSampler::DiagEvent> diag_;   // S0d: pending diagnostics (take_diagnostics)
    const long selfcheck;
    int setup_error = 0;

    // ── Group / propagation substrate ─────────────────────────────────────────────────────
    std::vector<Pauli> gens;
    int ng = 0;
    int GW = 0;
    CertifiedGroupPlanes G;
    PropagationTable table;
    bool ppr_tables = false;

    // ── Channels ──────────────────────────────────────────────────────────────────────────
    struct SChan { Pauli GS; std::vector<uint64_t> mask; };
    std::vector<SChan> sch;
    // S2b: the record family's columns in STORAGE order — every channel (slot c = sch index c) then
    // the extras (the Born observable, then the Born decisions in declaration order). Circuit constant.
    std::vector<RecColumn> rec_cols_;
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
    std::vector<uint64_t> oracle_abd_, oracle_nfd_, oracle_lin_;   // S0(f): selfcheck rows (CW words)
    std::vector<uint64_t> pxrow, pzrow;           // per-qubit prefix channel rows (CW words)
    uint64_t chtok = 0;                           // channel-set token

    // ── Noise channels / per-alt substrate ────────────────────────────────────────────────
    std::vector<std::vector<FiredPauli>> sides;   // flat-alt-id → fired list
    std::unique_ptr<DiagErrorSampler> smp;

    struct AltAtom {
        std::vector<std::pair<int, uint8_t>> av;   // (q, a mod 4)
        std::vector<std::pair<int, int>> czp;      // CZ pairs (j<l)
        std::vector<int> czidx;                    // union-pair indices (sorted lockstep w/ list)
        std::vector<int> vq;                       // X-translation support
    };
    std::vector<AltAtom> alt_atoms;
    std::vector<std::pair<int, int>> upairs;      // union CZ pair list, sorted
    // pair → upairs index (alt construction). unordered (2026-07-16): plan_local_key
    // does one lookup PER KEPT CZ PAIR per plan insert — the std::map walk dominated the
    // rows+index stage of the cold profile. Same lookups, same results.
    struct PairHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return ((uint64_t)(uint32_t)p.first << 32) ^ (uint32_t)p.second;
        }
    };
    std::unordered_map<std::pair<int, int>, int, PairHash> pidx;
    std::vector<uint8_t> alt_is_ppr;

    int PW = 0, NW = 0;
    std::vector<uint8_t> a_acc, v_acc, in_a, in_v;
    std::vector<uint64_t> czw;
    std::vector<uint8_t> in_w;
    std::vector<int> a_dirty, v_dirty, w_dirty;
    std::vector<uint64_t> amask_acc;              // maintained packed a&1 (the memo key part)
    std::vector<std::pair<int, int>> cz_s;        // this shot's cz list (reused)
    std::vector<uint64_t> prodmask;               // product-wire word mask
    const CertifiedGroupPlanes::ZAxisRREF* ZR = nullptr;

    struct AltMasks {
        std::vector<uint64_t> smask;   // NW: a&1 bits (full, unstripped)
        std::vector<uint64_t> vmask;   // NW: v bits
        std::vector<uint64_t> kmask;   // NW: a&1 & ~prodmask (KEY contribution)
        std::vector<uint64_t> czk;     // PW: kept (unstripped) pairs over upairs indexing
        std::vector<uint64_t> zxrow;   // CW: ⊕ pxrow over v support (x-part channel row)
        std::vector<uint64_t> zzrow;   // CW: ⊕ pzrow over z_j (a&2) support ⊕ own stripped −Z-pair flips
    };
    std::vector<AltMasks> alt_masks;

    std::vector<uint64_t> ab_kmask, ab_czk;
    std::vector<uint64_t> ab_T;                   // pair-term scratch
    std::vector<std::pair<int, int>> ab_cz;       // abelian key pair list (oracle only)
    int KW = 0;
    std::vector<uint64_t> ab_kw;                  // per-shot key scratch
    std::vector<uint64_t> ab_S, ab_V;             // running pair-term masks
    // S0(c): the plan index is the diagonal cache's own flat (hash, header) table — the key is
    // stored ONCE, inline in the header. upair_keep_: universe pairs whose endpoints are both
    // non-product wires (the product-wire strip keeps exactly these pairs), so a shot's canonical
    // key is (amask & ~prodmask ‖ czw & upair_keep_) — O(KW) words, no pair lookups.
    std::vector<uint64_t> upair_keep_;
    // S0(c) / D6 plan-payload budget (0 = unbounded) + the boundary accounting it reports.
    size_t plan_budget_ = 0;
    size_t payload_peak_ = 0, payload_overshoot_ = 0;
    void payload_boundary_();   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    bool fam_has_branches_() const;   // termroute-a: a FAM sampler with carried branches (twirl_sampler_fam.cpp)
    size_t born_memo_bytes_() const;   // S8-4: the root memos of the sampler's trie and its carried branches' tries
    void born_memo_clear_();           //       (defined in twirl_sampler_fam.cpp, where the branches live)
    std::vector<uint64_t> alt_blob;               // flat per-alt blob (see layout below)
    std::vector<uint32_t> alt_off;

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
    // The abelian accumulator route's residual prefix (X from v-support, Z from a&2 ⊕ zflip): the
    // extras' t(P) on that route (S2b).
    Pauli fast_prefix_;
    long fast_born_shots_ = 0;     // diagnostic counter (prof): born shots served by the abelian fast path
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
    static void record_rows_values_(const RecTable& T, const uint8_t* uz, uint64_t* out) {
        const int W = T.CWX;
        for (int w = 0; w < W; ++w) out[w] = T.base[(size_t)w];
        for (int k = 0; k < T.rho; ++k)
            if (uz[k]) { const uint64_t* r = T.coin_row[(size_t)k].data(); for (int w = 0; w < W; ++w) out[w] ^= r[w]; }
#ifdef XTIM_MUTATE_R61_ROW_SKIP
        for (int j = 0; j + 1 < T.kappa; ++j)     // mutation build only (R61): born_row[kappa-1] skipped
#else
        for (int j = 0; j < T.kappa; ++j)
#endif
            if (uz[T.rho + j]) { const uint64_t* r = T.born_row[(size_t)j].data(); for (int w = 0; w < W; ++w) out[w] ^= r[w]; }
    }
    void record_values_words_(const RecTable& T, const uint64_t* px, const uint64_t* pz, int pnw,
                              const uint8_t* uz, int n_uz, uint64_t* out) const;   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // The byte form of record_values_words_ (one call's y; the per-shot accessors).
    void record_outcomes_(const RecTable& T, const Pauli& P, const uint8_t* uz, int n_uz,
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
        std::shared_ptr<const RecTable> t;
        size_t bytes = 0;
        uint64_t tick = 0;
    };
    mutable std::mutex rec_memo_mu_;
    mutable std::unordered_map<std::string, ReaderEntry> rec_memo_;
    mutable uint64_t reader_tick_ = 0;                // under rec_memo_mu_
    mutable long reader_evictions_ = 0;               // under rec_memo_mu_
    mutable size_t reader_peak_ = 0, reader_overshoot_ = 0;   // under rec_memo_mu_
    mutable std::atomic<size_t> reader_bytes_{0};
    std::atomic<size_t> budget_pub_{0};               // plan_budget_, readable without run()'s state
    mutable std::atomic<size_t> sampler_resident_pub_{0};   // published at run()'s shot boundaries
    mutable std::atomic<long> reader_lookups_{0};
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
    // r8 (final review MAJOR-1): a table for a BUFFER to hold. It leaves the reader memo (if it was there)
    // and is never charged: buffer-held tables sit outside the plan payload budget (coordinator ruling),
    // so they can never make run() evict a sampler payload. The same pure function of (a, cz).
    std::shared_ptr<const RecTable> rec_table_held_(const DiagNormalForm& nf) const {
#ifdef XTIM_MUTATE_R81_HELD_CHARGED
        return rec_table_memo_(nf);               // mutation build only (R81): held tables charged (c8d1ff5)
#endif
        reader_lookups_.fetch_add(1, std::memory_order_relaxed);
        const std::string key = memo_key_(nf);
        {
            std::lock_guard<std::mutex> lk(rec_memo_mu_);
            auto it = rec_memo_.find(key);
            if (it != rec_memo_.end()) {
                std::shared_ptr<const RecTable> t = it->second.t;
                reader_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                rec_memo_.erase(it);
                return t;
            }
        }
        auto T = std::make_shared<RecTable>();
        rec_table_of_(nf, *T);
        return T;
    }
    std::shared_ptr<const RecTable> rec_table_memo_(const DiagNormalForm& nf) const {
        reader_lookups_.fetch_add(1, std::memory_order_relaxed);
        std::string key;
        key.reserve((size_t)n + 1 + 4 * nf.cz.size());
        for (int q = 0; q < n; ++q) key.push_back((char)(nf.a[(size_t)q] & 1));
        key.push_back((char)0xFF);
        for (const auto& pr : nf.cz) {
            key.push_back((char)(pr.first & 0xFF)); key.push_back((char)((pr.first >> 8) & 0xFF));
            key.push_back((char)(pr.second & 0xFF)); key.push_back((char)((pr.second >> 8) & 0xFF));
        }
        {
            std::lock_guard<std::mutex> lk(rec_memo_mu_);
            auto it = rec_memo_.find(key);
            if (it != rec_memo_.end()) { it->second.tick = ++reader_tick_; return it->second.t; }
        }
        auto T = std::make_shared<RecTable>();
        rec_table_of_(nf, *T);
        std::lock_guard<std::mutex> lk(rec_memo_mu_);
        auto ins = rec_memo_.emplace(key, ReaderEntry{});
        ReaderEntry& e = ins.first->second;
        if (ins.second) {                              // (another reader may have raced it in)
            e.t = std::move(T);
#ifdef XTIM_MUTATE_R64_NO_CHARGE
            e.bytes = 0;                               // mutation build only (R64): the table is not charged
#else
            e.bytes = rec_table_bytes(*e.t);
#endif
            reader_bytes_.fetch_add(e.bytes, std::memory_order_relaxed);
            const size_t rb = reader_bytes_.load(std::memory_order_relaxed);
            if (rb > reader_peak_) reader_peak_ = rb;
        }
        e.tick = ++reader_tick_;
        std::shared_ptr<const RecTable> out = e.t;     // held while the trim runs: never its victim
        reader_trim_locked_(sampler_resident_pub_.load(std::memory_order_relaxed), &key);
        return out;
    }
    FramedSuperposition materialize_(const DiagNormalForm& nf, const uint8_t* uz, int n_uz) const {
        const RecTable& T = *rec_table_memo_(nf);
        std::vector<uint8_t> y;
        record_outcomes_(T, nf.prefix, uz, n_uz, y);
        FramedSuperposition st = bare;
        apply_residual_gates_(st, nf);
#ifdef XTIM_MUTATE_MS_SKIP_PIVOT
        for (int k = 1; k < T.rho; ++k)           // mutation build only (M-S b): pivot 0 skipped
#else
        for (int k = 0; k < T.rho; ++k)
#endif
            force_record_(st, T.pivot_rec[(size_t)k], y[(size_t)T.pivot_rec[(size_t)k]], "anchor (pivot)");
        for (int j = 0; j < T.kappa; ++j) {
            uint8_t yj = y[(size_t)T.born_rec[(size_t)j]];
#ifdef XTIM_MUTATE_MS_BORN_FLIP
            yj ^= 1;                              // mutation build only (M-S a / M-ANCH)
#endif
            force_record_(st, T.born_rec[(size_t)j], yj, "anchor (direction)");
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
    FramedSuperposition sem_work_{0};
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
    // Fixed-point per-shot rf coins: slot sl fires iff RF word sl < rf_thresh[sl] =
    //   (uint64_t)(p_sl * 2^64) — one integer compare per slot; a pure function of (seed, shot).
    int nrf = 0;                        // distinct noisy records (prob>0), packed into RFW words
    int RFW = 0;                        // ceil(nrf/64)
    std::vector<double> rf_prob;        // nrf scaled probabilities (retained for reference/stats)
    std::vector<uint64_t> rf_thresh;    // FIX B: fixed-point thresholds: rf_thresh[sl] = (uint64_t)(p_sl * 2^64)
    std::vector<std::vector<uint64_t>> chan_rf_mask;   // per channel: RFW-word noisy-slot mask (kept for born_dec path)
    std::vector<uint64_t> born_rf_mask; // born observable's noisy-slot mask (RFW words, kept for reference)
    std::vector<uint64_t> rf_col_data_; // FIX A: flat storage for transposed columns (nrf * CW words)
    std::vector<uint64_t*> rf_col;      // FIX A: rf_col[sl] -> CW-word channel bitvector for slot sl
    std::vector<uint8_t> rf_obs_bit;    // FIX A: rf_obs_bit[sl] = 1 if born_rf_mask bit sl is set
    std::vector<uint64_t> rf_coins;     // per-shot scratch (RFW words)
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
    BornTrie trie_;
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
    // loop read its batch from a recorded source instead of smp->sample_events — the ONLY
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
    DiagNormalForm nf;                            // reused across shots
    std::vector<uint64_t> key_amask_s;            // canonical key (strip applied)
    std::vector<std::pair<int, int>> cz_key;
    std::vector<int> zflip;                       // −Z stripped-partner corrections
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
    // on every route. The noise stream is keyed per CHANNEL (DiagErrorSampler keyed mode).
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
         std::shared_ptr<TwirlPlanCache> plans);
    // S0(f): run-width channel rows at no hot-path cost. The shot loop is ONE template body with
    // two width instances: FW = 8 keeps the per-shot channel row a local uint64_t[8] for CW ≤ 8 (≤ 512
    // channels — exactly the capped engine's code generation; see run_w), FW = 0 points it at the
    // run-width member storage bits_ (CW > 8). Both instances are compiled in twirl_sampler_fam.cpp (body:
    // twirl_sampler_runw.inc.hpp; J-b2 J9: the FAM parameter is dropped). run() calls the instance chosen at
    // construction (run_fn_): one indirect call per run() call, nothing per shot.
    bool run(long shots) { return (this->*run_fn_)(shots); }
    template <int FW> bool run_w(long shots);

    // ── speed-kill T1: in-place reseed ────────────────────────────────────────────────────
    // Resets EXACTLY the state a fresh Impl construction (re)derives from `seed`,
    // reconstructing nothing. The seed-dependent inventory (exhaustive — verified against
    // every `seed` use in the constructor plus every cumulative member run() mutates):
    //   * the keyed stream keys (set_keys_: COIN, BORN, OBS, DEC, RF, GAUGE, EXACT) — every
    //                           per-shot draw is a pure function of (seed, shot index, tag, ctr);
    //   * smp                 — DiagErrorSampler keyed noise streams: reseed() re-derives every
    //                           channel's stream position AND its initial geometric skip exactly
    //                           as construction does (its ctor's last step IS reseed(seed));
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
        set_keys_(seed);
        smp->reseed(seed);
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
    bool rejoin_ok_(const std::vector<uint64_t>& E, int nr, int cw, int m, int r, const std::vector<uint64_t>& row,
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
        if (__builtin_expect(!plan.emit_built, 0)) build_rec_cold_(plan);   // r9: the block's flag (hot line)
    }
    __attribute__((noinline, cold)) void build_rec_cold_(const CachedPlan& plan) const;   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // The identity plan's payload (clean shots: R = I). A plan like any other (ndrec §1.3).
    const CachedPlan& identity_plan_();   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // THE record emission (plan §1.2; ONE noinline definition, the t_row toggle target): XOR the
    // plan's record rows into this shot's bits / xbits_, drawing the coins (COIN bits) and walking
    // the record trie (BORN words); the shot's outcomes u then z land in rec_uz_ (reused). Every
    // diagonal route calls it — the clean shot, the streaming and accumulator fast paths, the slow
    // path (code review MINOR-2: the streaming path's inline copy is gone).
    template <bool EX, bool KEEPU>
    __attribute__((noinline)) bool record_rows_(const CachedPlan& plan, uint64_t* bits);   // fixplan-coldstart (B): defined out of line in twirl_sampler.cpp
    // r9: the record_rows_ variant of this run (set at the top of run(); one indirect call per shot).
    using RowsFn = bool (Impl::*)(const CachedPlan&, uint64_t*);
    RowsFn rows_fn_ = &Impl::record_rows_<true, true>;
    void select_rows_fn_() {
        const bool ex = nextra_ != 0, keep = retain_state;
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
    // draw_rf: one Bernoulli(p_sl) coin per noisy record slot per shot — slot sl fires iff the
    // shot's keyed RF word sl is below the fixed-point threshold rf_thresh[sl] = (uint64_t)(p_sl·2^64)
    // (one integer compare per slot). The tag goes through the hash (krng), never XOR-linearly.
    //
    // apply_rf (transposed column form): iterate SET bits of rf_coins[w] via ctz; for each fired
    //   slot sl, XOR rf_col[sl] into bits and conditionally flip obs_bit_cur via rf_obs_bit[sl] —
    //   O(fired * CW) work. born_dec_rf_mask is read in emit_decisions per decision.
    void draw_rf() {
        krng::Stream st = rf_s_;                       // local: the rf_coins stores cannot alias it
        st.at(shot_);
        for (int w = 0; w < RFW; ++w) rf_coins[(size_t)w] = 0;
        for (int sl = 0; sl < nrf; ++sl)
            if (st.next() < rf_thresh[(size_t)sl])     // RF word sl (sequential = addressed)
                rf_coins[sl >> 6] |= 1ull << (sl & 63);
    }
    void apply_rf(uint64_t* bits) {
        // FIX A: column (transposed) form.  Iterate set bits of rf_coins; for each fired slot sl
        // XOR rf_col[sl] (CW words) into bits and fold the born-obs bit via rf_obs_bit[sl].
        if (nrf == 0) return;
        int obs_par = 0;
        for (int w = 0; w < RFW; ++w) {
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
    std::string family_refusal_;
    long sem_class_[6] = {0, 0, 0, 0, 0, 0};   // semantic-oracle checks per record class
    void fam_setup_(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                    const TwirlRecordOptions& opt, const std::vector<ErrorChannel>& chans);
    bool fam_run_guard_(long shots);
    void fam_refuse_(const std::string& m);
    // J-b1 P2 (twirl_sampler_fam.cpp, out of line and cold): a FAM plan that fell back (engine bug: the slow path
    // builds diagonal-class normal forms only), and the TWPL save notice as a DISK_SAVE event (no stderr).
    [[noreturn]] void fam_plan_fallback_bug_() const;
    void fam_disk_saved_();
    void fam_add_(int kind, int index, const std::vector<int>& recs, const std::vector<std::pair<int, int>>& reads);
    void fam_cols_();
    bool semantic_oracle_fam_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot);
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
    struct FamTermHold {
        FamRoute* p = nullptr;
        FamTermHold() = default;
        FamTermHold(const FamTermHold&) = delete;
        FamTermHold& operator=(const FamTermHold&) = delete;
        ~FamTermHold();
    };
    FamTermHold fam_term_;
    TermResidual& fam_term_route_();
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
    static uint64_t term_set_token_(const TermResidual& tr);   // J-b1 P11a: TWPL v6 (twirl_sampler_fam.cpp)
    // J-b1 P6: a FAM (D) plan's record table by record-space B (build_record_table_recB; J-b2 J7b: the only
    // builder, the σ dual basis is deleted). Defined in twirl_sampler_fam.cpp.
    void build_rec_table_fam_(const CachedPlan& plan, const DiagNormalForm& nf0, RecTable& out) const;
};

}  // namespace qeccore
