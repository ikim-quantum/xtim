// twirl_sampler — V2-T3: TwirlRecordSampler, the end-to-end twirl record sampler.
//
// MOVE-only extraction of the framed_bench `twirl_records` mode body (byte-equality gated
// at fixed seeds: identical RNG draw order, identical counters, identical oracles). The
// constructor is the mode's setup phase (channel setup, propagation table, per-alt sparse
// atoms + masks + flat blob, plan caches, disk load); run() is the shot loop. See the
// header and the original mode comments below for the algorithm.

#include "qeccore/twirl_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "qeccore/clifford_op.hpp"
#include "qeccore/diag_error_sampler.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/sampler.hpp"        // FiredPauli / compose_fired
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"
#include "qeccore/twirl_ppr.hpp"

namespace qeccore {

namespace {
using clk = std::chrono::steady_clock;
inline uint64_t rotl64(uint64_t x, int r) { return r ? (x << r) | (x >> (64 - r)) : x; }

// XTIM_QUIET: read once per process (static local). When set to a non-empty
// value other than "0", suppress informational [twirl_records] banners on
// stderr.  Error/fatal prints and prof-gated prints are unaffected.
// Per the engine coding playbook: env levers are read ONCE via function-local
// statics; tests toggling them need subprocesses.
inline bool xtim_quiet() {
    static const bool _quiet = [] {
        const char* v = std::getenv("XTIM_QUIET");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    return _quiet;
}

// V3-T3 R2: structural FNV-64 of the deferred circuit + terminal reads for the AUTO plan-cache
// file key. Noise probability VALUES are deliberately EXCLUDED (the plan cache is noise-blind:
// one auto file serves whole p-sweeps); everything shape-defining is mixed (kinds, gates,
// targets, channels, qubit lists, measure bases/inverts, observables, feedback offsets, reads).
// Mirrors the pybind deferred_signature content, hashed instead of concatenated.
uint64_t deferred_signature_fnv(const Circuit& deferred,
                                const std::vector<std::pair<int, int>>& reads) {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001B3ull; };
    mix((uint64_t)(uint32_t)deferred.n);
    for (const Instr& ins : deferred.stream) {
        mix(0x100u + (uint64_t)(int)ins.kind);
        switch (ins.kind) {
            case Instr::Kind::Gate:
                mix((uint64_t)(int)ins.gate);
                for (int t : ins.targets) mix((uint64_t)(uint32_t)t);
                break;
            case Instr::Kind::Noise:
                mix((uint64_t)(int)ins.channel);               // probs EXCLUDED (noise-blind)
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            case Instr::Kind::Measure:
                mix((uint64_t)(int)ins.basis);
                mix(ins.invert ? 0x21u : 0x2Eu);               // flip PROBABILITY excluded
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            case Instr::Kind::Reset:
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            case Instr::Kind::Observable:
                for (const PauliTerm& t : ins.obs) {
                    mix((uint64_t)(uint32_t)t.qubit);
                    mix(0x70u + (uint64_t)(int)t.p);
                }
                for (int r : ins.obs_frame) mix(0xF00u ^ (uint64_t)(uint32_t)r);
                break;
            case Instr::Kind::ControlledPauli:
                mix((uint64_t)(int)ins.basis);
                mix((uint64_t)(uint32_t)ins.control_rec_offset);
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            default: break;   // IfBegin/IfEnd: no shape-defining content
        }
    }
    mix(0x5EAD5ull);                                           // reads-section separator
    for (const auto& pr : reads) {
        mix((uint64_t)(uint32_t)pr.first);
        mix((uint64_t)(uint32_t)pr.second);
    }
    return h;
}
}  // namespace

struct TwirlRecordSampler::Impl {
    // ── Inputs / configuration ────────────────────────────────────────────────────────────
    const FramedSuperposition& bare;           // caller-owned; must outlive the sampler
    const int n;
    std::string disk_path;                     // non-const since V3-T3 R2: may be auto-derived
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
    std::vector<int> chan_det;                    // det channel c → detector index
    std::vector<int> chan_obs;                    // obs channel k → observable index (V2-T4;
                                                  // channels chan_det.size().. are obs channels)
    std::vector<uint8_t> chan_ref;                // channel c → REPORTING reference bit
                                                  // (classify ⊕ rec_invert; for det/obs
                                                  // channels MINUS the stage text's own
                                                  // deterministic-Pauli flip — P0 fix,
                                                  // reference-relative Stim semantics)
    std::vector<int> gauge_dets, anti_dets, refused_obs;
    int nchan = 0;
    int CW = 0;                                   // channel-space word count
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
    // pair → upairs index (kept: local-index keys). unordered (2026-07-16): plan_local_key
    // does one lookup PER KEPT CZ PAIR per plan insert — the std::map walk dominated the
    // rows+index stage of the cold profile. Same lookups, same results.
    struct PairHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return ((uint64_t)(uint32_t)p.first << 32) ^ (uint32_t)p.second;
        }
    };
    std::unordered_map<std::pair<int, int>, int, PairHash> pidx;
    std::vector<uint8_t> alt_is_ppr;
    std::vector<PprNormalForm> alt_pnf;

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
    std::vector<std::vector<uint64_t>> lkeys;     // stable key storage
    std::vector<const CachedPlan*> lplans;
    std::unordered_map<uint64_t, std::vector<int>> lidx;
    std::vector<uint64_t> lkey_scratch;
    std::vector<uint64_t> ftab;                   // flat open-addressed find table
    size_t fslots = 0, fcount = 0;
    // Shared fast-route index: if non-null, lkeys/lplans/lidx/ftab/fslots/fcount are adopted
    // from it at construction and written back at the end of run(). Warm rebuilds therefore find
    // the flat table pre-populated and skip materialize+canon on every shot outside the selfcheck
    // window. Soundness: coins are a pure function of (seed, shot_index, plan) after 4e51920c, so
    // serving a shot from the shared index is byte-identical to the cold call that built the entry.
    std::shared_ptr<SharedFastIndex> sfi_;
    std::vector<uint64_t> alt_blob;               // flat per-alt blob (see layout below)
    std::vector<uint32_t> alt_off;

    // ── Collapse RNG / plan caches ────────────────────────────────────────────────────────
    Rng coll_rng;
    // Shared across rebuilds of the same compiled circuit (perf fix: plans survive seed resets).
    // Always non-null after construction (created fresh or adopted from SharedPlanCaches).
    std::shared_ptr<TwirlPlanCache> cache;
    std::shared_ptr<PprPlanCache>   pcache;
    std::vector<PprAtom> ppr_views;
    long pprshots = 0, pprchecked = 0;

    // ── Per-shot loop state (persists across run() calls) ────────────────────────────────
    // V2-T4: optional per-shot record sink. Null by default; every call site is guarded
    // with `if (sink)` so the null-sink shot loop is byte-identical to the counting path.
    // Fix 2: devirtualised barrier sink (takes precedence over `sink` when set). Non-owning.
    BarrierSink* barrier_sink_ = nullptr;
    DetsPackSink* dets_pack_sink_ = nullptr;
    // Single per-shot emit seam: the barrier path uses the inline-able devirtualised sink; every
    // other consumer (sample()) keeps the std::function. The null-sink hot path is byte-identical.
    inline void emit_shot(const uint64_t* bits, int obs_bit, bool exact_fb,
                          const FramedSuperposition* amps, const uint64_t* sigma_gens,
                          const uint8_t* decision_bits, const uint8_t* coins, int n_coins,
                          const uint8_t* plan_key, int n_plan, const double* born_u, int n_born,
                          const CachedPlan* plan_ptr = nullptr, const uint64_t* pre_x = nullptr,
                          const uint64_t* pre_z = nullptr) {
        if (barrier_sink_)
            barrier_sink_->emit(bits, obs_bit, exact_fb, amps, sigma_gens, decision_bits,
                                coins, n_coins, plan_key, n_plan, born_u, n_born,
                                plan_ptr, pre_x, pre_z);
        else if (dets_pack_sink_)
            dets_pack_sink_->emit(bits, obs_bit, exact_fb, amps, sigma_gens, decision_bits,
                                  coins, n_coins, plan_key, n_plan, born_u, n_born,
                                  plan_ptr, pre_x, pre_z);
    }
    // S2.2 (decoder-feedback): retain each shot's collapsed post-barrier state. Off by default
    // (production path byte-identical). When on, run() forces the slow per-shot path and calls
    // twirl_collapse(need_amps=true) so o_rec.amps is the real collapsed state delivered to sink.
    bool retain_state = false;
    // ── Task 2b.1 Born-measurement DECISION channels ─────────────────────────────────────
    // Each declared DECISION(k) parity is classified like a DETECTOR/OBSERVABLE (G.reduce):
    //   IN_GROUP → a deterministic σ-mask channel appended to `sch` (its bit = ref ⊕ mask·σ,
    //              the existing channel machinery); dec_chan_* map channel slot → decision k;
    //   LOGICAL/ANTI → born_dec: (decision k, record operator W_dec), Born-measured per shot
    //              on the retained collapsed state via measure_pauli, in DECLARATION order.
    // has_decisions_ forces the need_amps slow path (a mutable per-shot state to measure on);
    // dec_gen is a DEDICATED stream so decision-free circuits stay byte-identical.
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
    // decoder-feedback perf: per-shot RAW born-decision outcomes (out<0 ? 1 : 0), in declaration
    // order, BEFORE invert/rf. These are the reproducibility record for the lazy state
    // materialization: appended to o_rec.coins (the coin record) so materialize() can force-replay
    // the exact per-shot born collapse. The reported decision bit (dec_scratch_) is derived from
    // the SAME measure_pauli outcome but ALSO carries invert/rf — decoupling the two is the
    // mutation the correlation gate catches. Filled by emit_decisions on the born (amps_mut) path.
    std::vector<uint8_t> born_raw_;     // len born_dec.size() on the born path; empty otherwise
    // decoder-feedback perf: per-shot born-decision COINS u∈[0,1). materialize() RE-MEASURES each
    // born operator on the independently-reconstructed collapsed state with THIS coin (a genuine
    // Born measurement, NOT a forced outcome) — so the retained state is byte-identical to the
    // record path's AND the correlation gate stays mutation-sensitive: decoupling the reported
    // decision bit from the measurement makes materialize's re-measured sign disagree with it.
    std::vector<double> born_u_;        // len born_dec.size() on the born path; empty otherwise
    FramedSuperposition dec_work_{0};   // per-shot mutable copy of bare for clean-shot Born reads
    std::mt19937_64 dec_gen;            // DEDICATED Born-decision rng (isolation)
    // decoder-feedback perf (fold_p1 pattern): MEMOIZED Born-decision probability, amortized per
    // distinct record. Key = record prefix bytes (σ ‖ coins ‖ plan) ‖ prior born outcomes
    // b_0..b_{j-1} (the conditional chain: p_j depends only on the pre-decision collapsed state,
    // which is a pure function of that key). Value = p_j = born_p1(W_j) on the state collapsed
    // through decisions <j. On a MISS the state is materialised ONCE (materialize_shot, the ~6.5µs
    // collapse) and p cached; on a HIT the per-shot cost is a hash lookup + one coin compare —
    // NO per-shot collapse. Cleared at run() start so it never grows unbounded across runs (p is a
    // pure function of the key, so re-derivation on a fresh run is identical). C++-side (not Python)
    // because the born draw + coin record live in this loop — a Python memo would add 20k round-trips.
    // decoder-feedback perf (barrier2, Lever 1): INTEGER-HASH open-addressed Born-probability memo.
    // Replaces the per-shot std::string record-key REBUILD (~10–45 push_backs) + the
    // unordered_map<std::string,double> re-hash of the full key every shot (the audit's #1 barrier
    // lever, 0.126 µs). Key components = (σ zero-extended to GW words, coins, plankey, born-outcome
    // prefix b_0..b_{j-1}) — the SAME sufficient statistic as the old string key. The hash is folded
    // incrementally from the LIVE upstream buffers (no concatenation). COLLISION-SAFE /
    // never-silent-wrong: a hash match is CONFIRMED by a full component-wise compare against the
    // slot's stored key bytes before p is served — a bare-hash accept could collide and serve a
    // stale p (the mandatory mutation proves this guard is load-bearing). Open-addressed, power-of-2,
    // ≤50% load, linear probe; cleared per run() (p is a pure function of the key ⇒ re-derivation is
    // exact, so a fresh run rebuilds identical entries).
    struct BornMemoEntry {
        uint64_t h = 0;
        bool used = false;
        std::vector<uint64_t> sig;      // σ zero-extended to GW words
        std::vector<uint8_t>  coins;    // o_rec.coins snapshot
        std::vector<uint8_t>  pk;       // plankey_ snapshot
        std::vector<uint8_t>  born;     // born-outcome prefix b_0..b_{j-1}
        double p = 0.0;
    };
    std::vector<BornMemoEntry> born_memo_;   // open-addressed table (size = power of 2, or empty)
    size_t born_memo_mask_ = 0;              // (table size − 1); valid only when table non-empty
    size_t born_memo_count_ = 0;             // occupied slots
    std::vector<uint64_t> sig_scratch_;      // σ zero-extended to GW words (reused, no per-shot alloc)
    static uint64_t fold_words_(uint64_t h, const uint64_t* p, size_t n) {
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
        return h;
    }
    static uint64_t fold_bytes_(uint64_t h, const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)p[i]; h *= 0x100000001b3ull; }
        return h;
    }
    void born_memo_grow_() {
        const size_t newsize = born_memo_.empty() ? 1024 : born_memo_.size() * 2;
        std::vector<BornMemoEntry> nt(newsize);
        const size_t nm = newsize - 1;
        for (auto& e : born_memo_)
            if (e.used) {
                size_t s = e.h & nm;
                while (nt[s].used) s = (s + 1) & nm;
                nt[s] = std::move(e);
            }
        born_memo_ = std::move(nt);
        born_memo_mask_ = nm;
    }
    // Hoisted (was constructed per call): std::uniform_real_distribution<double>{0,1} carries no
    // per-call state in libstdc++ (each operator() draws afresh from the generator), so reusing one
    // member instance is BYTE-IDENTICAL to the old per-call construction.
    std::uniform_real_distribution<double> dec_dist_{0.0, 1.0};
    double dec_u01() { return dec_dist_(dec_gen); }

    // ── decoder-feedback perf (injective u64 Born-memo, Fix 1) ────────────────────────────────
    // EXACT-INJECTIVE key that KILLS the byte-key collision guard structurally WITHOUT any per-shot
    // plankey hashing. Two facts drive it:
    //   (1) σ is REDUNDANT in the memo key. p_j is computed by `materialize_shot(plankey, coins,
    //       born_u, n_born=j)`, which DOES NOT CONSUME σ (see its signature); given the residual and
    //       coins, σ is fully determined. So σ can be dropped from the key — it merges nothing.
    //   (2) The residual = γ · P · C with P the prefix Pauli (X^v Z^z, incl. the S²→Z fold) applied
    //       OUTERMOST and C the Clifford (S^{a&1}, then CZ) captured by the CachedPlan (its cache key
    //       is exactly (a&1, cz)). For a Pauli P, P†W P = (−1)^{⟨W,P⟩} W, so the prefix enters the
    //       Born amplitude ⟨W_j⟩ (and every prior collapse of W_i) ONLY through the sign bit
    //       ⟨W_i, P⟩ — NOT the full prefix. Hence the collapsed pre-decision state (and p_j) is a
    //       pure function of (CachedPlan, coins, prior born OUTCOMES b_{<j}, signs ⟨W_i,P⟩, j).
    // The key packs a DENSE CachedPlan id (id 0 reserved for the clean/identity residual) into the
    // high 32 bits and (coins | born-prefix | prefix-signs | j) into the low field:
    //   key = (plan_id<<32) | (j<<(nc+2nborn)) | (signs<<(nc+nborn)) | (born<<nc) | coins.
    // Within a fixed plan_id the field widths (nc,nborn,jbits) are constant ⇒ collision-free. The
    // memo is a flat open-addressed u64→p table: a slot match compares the u64 KEY ITSELF (injective
    // ⇒ identity), NOT reconstructed byte components — the collision guard is gone AND no plankey is
    // hashed per shot (the previous byte-key path folded σ+coins+plankey EVERY shot). If the low
    // field (nc + 2·nborn + jbits) exceeds 32 bits, the shot FALLS BACK to the guarded byte-key memo
    // (loud, counted) — never truncated.
    const CachedPlan* cur_plan_ = nullptr;     // captured for emit_decisions (nullptr = clean/identity)
    const Pauli*      cur_prefix_ = nullptr;   // captured residual prefix (nullptr = identity ⇒ signs 0)
    std::unordered_map<const CachedPlan*, uint32_t> plan_ids_;   // plan ptr → dense id (bounded by #plans)
    uint32_t plan_dense_id_(const CachedPlan* p) {              // id 0 = clean/identity residual
        if (!p) return 0;
        auto it = plan_ids_.find(p);
        if (it != plan_ids_.end()) return it->second;
        const uint32_t id = (uint32_t)plan_ids_.size() + 1;
        plan_ids_.emplace(p, id);
        return id;
    }
    struct U64Entry { uint64_t k = 0; double p = 0.0; bool used = false; };
    std::vector<U64Entry> u64_memo_;
    size_t u64_memo_mask_ = 0, u64_memo_count_ = 0;
    long   u64_budget_fallbacks_ = 0;   // DIAGNOSTIC: shots that overflowed the 32-bit low field
    bool   u64_budget_warned_ = false;  // one-shot loud stderr note
    void u64_memo_grow_() {
        const size_t newsize = u64_memo_.empty() ? 1024 : u64_memo_.size() * 2;
        std::vector<U64Entry> nt(newsize);
        const size_t nm = newsize - 1;
        for (auto& e : u64_memo_)
            if (e.used) { size_t s = e.k & nm; while (nt[s].used) s = (s + 1) & nm; nt[s] = e; }
        u64_memo_ = std::move(nt);
        u64_memo_mask_ = nm;
    }
    // Emit this shot's decision output bits (once per shot; draws from dec_gen for Born reads).
    //   IN_GROUP decisions: bit = raw channel bit (parity σ&mask, in `bits`) ⊕ noiseless ref.
    //   LOGICAL/ANTI decisions: a MEMOIZED Born-probability coin (fold_p1 pattern). The +1 prob
    //   p_j = born_p1(W_dec_j) on the state collapsed through decisions <j is a pure function of the
    //   record prefix (σ ‖ coins ‖ plan) ‖ b_0..b_{j-1}; it is cached in born_memo_ (integer-hash,
    //   collision-guarded) so the collapse runs ~once per distinct record, not per shot. Per shot:
    //   an incremental-hash lookup + a guard compare + `u < p` on the SAME
    //   dec_gen stream measure_pauli drew from (raw = (u<p)?0:1 mirrors ε=+1 iff u<pp). On the rare
    //   fallback path (`exact_state` non-null) the born ops are measured DIRECTLY on the exact state.
    //   Returns nullptr if no DECISION is declared (decision-free circuits: byte-identical).
    const uint8_t* emit_decisions(const uint64_t* bits, FramedSuperposition* exact_state) {
        if (!has_decisions_) return nullptr;
        std::fill(dec_scratch_.begin(), dec_scratch_.end(), (uint8_t)0);
        born_raw_.clear();   // filled below only on the born path
        born_u_.clear();
        for (size_t j = 0; j < dec_chan_slot_.size(); ++j) {
            const int c = dec_chan_slot_[j];
            const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)dec_chan_ref_[j];
            dec_scratch_[(size_t)dec_chan_idx_[j]] = (uint8_t)b;
        }
        if (!has_born_dec_) return dec_scratch_.data();
        born_raw_.resize(born_dec.size());
        born_u_.resize(born_dec.size());

        // Build the base record-prefix key = σ (o_rec.sigma) ‖ coins (r fair ‖ κ chain) ‖ plan
        // (plankey_) — the SAME sufficient statistic the record buckets use. On the memo path the
        // conditional chain appends b_0..b_{j-1} to this base as decisions are decided.
        const bool memo_path = (exact_state == nullptr);
        // Fix 1: INJECTIVE u64 key setup (design note in the member block). The key is
        //   (plan_id, coins, born-prefix, prefix-signs ⟨W_i,P⟩, j)
        // — NO σ (redundant, materialize_shot never reads it) and NO per-shot plankey hashing (the
        // prefix compresses to nborn sign bits). Decide the low-field budget ONCE (constant within a
        // plan_id): nc coins + nborn born + nborn signs + jbits. Overflow ⇒ guarded byte-key fallback.
        uint32_t pid = 0, sign_bits = 0;
        const size_t nborn = born_dec.size();
        size_t nc = 0, jbits = 0;
        bool u64_ok = false;
        if (memo_path) {
            pid = plan_dense_id_(cur_plan_);
            nc = o_rec.coins.size();
            for (size_t t = nborn ? nborn - 1 : 0; t; t >>= 1) ++jbits;   // ceil(log2(nborn))
            u64_ok = (nc + 2 * nborn + jbits <= 32);
            // prefix-sign bits ⟨W_i, P⟩ (i<nborn): the ONLY way the prefix Pauli affects any p_j.
            // cur_prefix_ == nullptr ⇒ identity residual (clean shot) ⇒ all signs 0.
            if (u64_ok && cur_prefix_)
                for (size_t i = 0; i < nborn; ++i)
                    sign_bits |= (uint32_t)Pauli::anticommute_bit(born_dec[i].second, *cur_prefix_) << i;
            if (!u64_ok) {
                ++u64_budget_fallbacks_;
                if (!u64_budget_warned_) {
                    u64_budget_warned_ = true;
                    fprintf(stderr,
                        "[twirl_records] injective-u64 Born-memo budget exceeded "
                        "(coins=%zu + 2*born=%zu + jbits=%zu > 32); guarded byte-key fallback engaged\n",
                        nc, 2 * nborn, jbits);
                }
            }
        }
        // Packed coin bits (constant across the decision loop) for the u64 key.
        uint32_t coins_bits = 0;
        if (memo_path && u64_ok)
            for (size_t i = 0; i < nc; ++i)
                coins_bits |= (uint32_t)(o_rec.coins[i] & 1) << i;
        // Byte-key base hash — FALLBACK path only: fold σ (zero-extended to GW words), coins, plankey.
        uint64_t base_h = 0;
        if (memo_path && !u64_ok) {
            sig_scratch_.assign((size_t)GW, 0);
            for (size_t w = 0; w < o_rec.sigma.size() && w < (size_t)GW; ++w)
                sig_scratch_[w] = o_rec.sigma[w];
            base_h = 0xcbf29ce484222325ull;
            base_h = fold_words_(base_h, sig_scratch_.data(), (size_t)GW);
            base_h = fold_bytes_(base_h, o_rec.coins.data(), o_rec.coins.size());
            base_h = fold_bytes_(base_h, plankey_.data(), plankey_.size());
        }

        uint64_t h = base_h;         // byte-key conditional chain (fallback only)
        uint32_t born_bits = 0;      // packed born-prefix bits b_0..b_{j-1} (u64 key)
        for (size_t j = 0; j < born_dec.size(); ++j) {
            // p_j: the +1 Born probability of W_dec_j on the state collapsed through decisions <j.
            double p;
            if (!memo_path) {
                // Fallback (rare / out-of-class): measure directly on the exact per-shot state.
                p = exact_state->born_p1(born_dec[j].second);
            } else if (u64_ok) {
                // ── INJECTIVE u64 memo: NO full-key compare (the key is injective for the collapsed
                //    pre-decision state ⇒ for p_j). A slot match compares the u64 KEY only — that is
                //    open-addressing table-slot disambiguation, not a hash-collision guard.
                const uint64_t key = ((uint64_t)pid << 32)
                                   | ((uint64_t)j << (nc + 2 * nborn))
                                   | ((uint64_t)sign_bits << (nc + nborn))
                                   | ((uint64_t)born_bits << nc)
                                   | (uint64_t)coins_bits;
                if ((u64_memo_count_ + 1) * 2 > u64_memo_.size()) u64_memo_grow_();
                size_t s = key & u64_memo_mask_;
                for (;;) {
                    U64Entry& e = u64_memo_[s];
                    if (!e.used) {
                        // MISS: materialise the state collapsed THROUGH decisions <j (n_born=j
                        // replays the first j born ops with the coins drawn in prior iterations),
                        // then born_p1(W_j) = the exact pp measure_pauli would use. IDENTICAL
                        // materialise call + args as the byte-key path ⇒ bit-identical p.
                        FramedSuperposition st = materialize_shot(
                            plankey_.empty() ? nullptr : plankey_.data(), (int)plankey_.size(),
                            o_rec.coins.empty() ? nullptr : o_rec.coins.data(), (int)o_rec.coins.size(),
                            born_u_.empty() ? nullptr : born_u_.data(), (int)j);
                        p = st.born_p1(born_dec[j].second);
                        e.used = true; e.k = key; e.p = p; ++u64_memo_count_;
                        break;
                    }
                    if (e.k == key) { p = e.p; break; }   // HIT: injective key match ⇒ identity
                    s = (s + 1) & u64_memo_mask_;          // table-slot collision: keep probing
                }
            } else {
                // ── guarded byte-key FALLBACK memo (budget overflow): full-key collision guard.
                if ((born_memo_count_ + 1) * 2 > born_memo_.size()) born_memo_grow_();
                const uint64_t hh = h;
                size_t s = hh & born_memo_mask_;
                for (;;) {
                    BornMemoEntry& e = born_memo_[s];
                    if (!e.used) {
                        FramedSuperposition st = materialize_shot(
                            plankey_.empty() ? nullptr : plankey_.data(), (int)plankey_.size(),
                            o_rec.coins.empty() ? nullptr : o_rec.coins.data(), (int)o_rec.coins.size(),
                            born_u_.empty() ? nullptr : born_u_.data(), (int)j);
                        p = st.born_p1(born_dec[j].second);
                        e.used = true;  e.h = hh;  e.p = p;
                        e.sig = sig_scratch_;
                        e.coins = o_rec.coins;
                        e.pk = plankey_;
                        e.born.assign(born_raw_.begin(), born_raw_.begin() + (long)j);
                        ++born_memo_count_;
                        break;
                    }
                    const bool confirm =
                        (e.h == hh && e.sig == sig_scratch_ && e.coins == o_rec.coins &&
                         e.pk == plankey_ && e.born.size() == j &&
                         std::equal(e.born.begin(), e.born.end(), born_raw_.begin()));
                    if (confirm) { p = e.p; break; }
                    s = (s + 1) & born_memo_mask_;
                }
            }
            const double u = dec_u01();
            born_u_[j] = u;                              // COIN for materialize's genuine re-measure
            const uint8_t raw = (u < p) ? (uint8_t)0 : (uint8_t)1;   // ε=+1 (bit 0) iff u<pp
            born_raw_[j] = raw;                          // RAW outcome for the record-hash KEY
            uint8_t bit = raw;
            bit ^= born_dec_inv[j];                      // deterministic record invert (Stim `!` etc.)
            // input_pauli frame relabel (decoder-feedback engine fix): the frame P is a Pauli
            // applied at the port at t=0. For a Born DECISION operator W_j, P W_j P† =
            // (−1)^{sp(W_j,P)} W_j, so measuring W_j on P|ψ⟩ gives the frame-free outcome
            // XOR-flipped by sp(W_j,P) — the PROBABILITY p_j is unchanged (proven: the frame is a
            // deterministic bit relabel, not a reweight, even for magic states and mutually
            // anticommuting sequential decisions). This is the SAME Heisenberg relabel the born_obs
            // path applies at L804-810; the born_dec path previously dropped it entirely. Frame-free
            // shots (cur_frame_row_==nullptr) skip this ⇒ byte-identical.
            bit ^= frame_born_flip(j);
            if (!born_dec_rf_mask.empty() && !born_dec_rf_mask[j].empty()) {
                int par = 0;                             // M(p) readout-flip coins (drawn this shot)
                for (int w = 0; w < RFW; ++w)
                    par += __builtin_popcountll(rf_coins[(size_t)w] & born_dec_rf_mask[j][(size_t)w]);
                bit ^= (uint8_t)(par & 1);
            }
            dec_scratch_[(size_t)born_dec[j].first] = bit;
            if (memo_path) {
                if (u64_ok) born_bits |= (uint32_t)raw << j;   // extend the u64 born-prefix with b_j
                else        h = fold_bytes_(h, &raw, 1);       // extend the byte-key chain with b_j
            }
        }
        return dec_scratch_.data();
    }

    // ── decoder-feedback perf: LAZY per-shot state materialization ────────────────────────────
    // Reconstruct one shot's COLLAPSED post-barrier + post-decision FramedSuperposition from its
    // compact record (plan_key ‖ coins), WITHOUT having retained the full state. EXACT replay:
    //   1. parse plan_key -> the canonical residual normal form nf (identity when empty = clean);
    //   2. build the diagonal ShotLaw (kernel_logicals are prefix-free -> the amps are byte-
    //      identical to the record path's, which used the finalized law — the σ-only fields differ
    //      but never enter the amplitudes);
    //   3. twirl_collapse(need_amps=true) driven by a FORCED rng that replays the recorded coin
    //      record's first r+κ entries (the fair-coin/kernel-chain outcomes; u=0 forces +1/bit 0,
    //      u=nextafter(1,0) forces −1/bit 1 — valid because a recorded outcome had nonzero
    //      probability), reproducing the exact kernel collapse. The remaining coin entries are the
    //      born outcome BITS (record-hash key only) and are NOT consumed here;
    //   4. RE-MEASURE each born decision operator on the collapsed amps with its recorded COIN
    //      born_u[j] — a GENUINE Born measurement (not a forced outcome). In the record path this
    //      is the identical measure_pauli on the identical state, so the physical survivor collapses
    //      byte-identically; and because the reported decision bit is NOT consulted, decoupling it
    //      from this measurement (the mandatory mutation) makes the correlation gate disagree.
    // Called ~once per distinct record (bucket representative), NOT per shot — the O(1)-per-shot
    // hot path never touches it.
    FramedSuperposition materialize_shot(const uint8_t* plan_key, int n_plan,
                                         const uint8_t* coins, int n_coins,
                                         const double* born_u, int n_born) const {
        DiagNormalForm nf;
        nf.a.assign((size_t)n, 0);
        nf.prefix = Pauli(n);
        nf.diagonal_class = true;
        const int NW = (n + 63) / 64;
        if (plan_key && n_plan > 0) {
            int p = 0;
            auto rd_words = [&](std::vector<uint64_t>& dst) {
                for (int w = 0; w < NW; ++w) {
                    uint64_t v = 0;
                    for (int b = 0; b < 8; ++b)
                        if (p < n_plan) v |= ((uint64_t)plan_key[p++]) << (b * 8);
                    if (w < (int)dst.size()) dst[(size_t)w] = v;
                }
            };
            rd_words(nf.prefix.x);
            rd_words(nf.prefix.z);
            for (int q = 0; q < n; ++q)
                if (p < n_plan) nf.a[(size_t)q] = (uint8_t)(plan_key[p++] & 1);
            if (p < n_plan && plan_key[p] == 0xFF) ++p;   // separator
            while (p + 3 < n_plan) {
                int f = (int)plan_key[p] | ((int)plan_key[p + 1] << 8);
                int s = (int)plan_key[p + 2] | ((int)plan_key[p + 3] << 8);
                nf.cz.emplace_back(f, s);
                p += 4;
            }
        }
        ShotLaw law = build_shot_law(G, nf);
        int idx = 0;
        Rng frng = [&]() -> double {
            const uint8_t b = (idx < n_coins) ? coins[idx] : 0;
            ++idx;
            return b ? std::nextafter(1.0, 0.0) : 0.0;
        };
        TwirlOutcome o = twirl_collapse(bare, G, nf, law, frng, /*need_amps=*/true);
        // Re-measure the FIRST n_born born decisions on the collapsed amps in declaration order
        // with the recorded coins — a genuine Born measurement reproducing the record path's
        // collapse. The full-state caller passes n_born == born_dec.size() (all replayed, the
        // legacy behaviour). The memoized-coin builder passes n_born = j (0 ≤ j ≤ #born) to obtain
        // the state collapsed THROUGH decisions <j, on which born_p1(W_j) is the conditional p_j.
        const size_t jstop = std::min((size_t)std::max(0, n_born), born_dec.size());
        for (size_t j = 0; j < jstop; ++j) {
            const double u = born_u ? born_u[j] : 0.0;
            (void)o.amps.measure_pauli(born_dec[j].second, u);
        }
        return o.amps;
    }

    // ── E3 (exact-residual arc): plan-key parser, shared format with materialize_shot ─────────
    // Byte-for-byte the SAME parse as materialize_shot's step 1 above (deliberately duplicated —
    // the established additive pattern — so the byte-pinned materialize_shot body stays
    // untouched). Format (serialized in run(), the "barrier2 Lever 2" block):
    //   prefix.x NW u64 words LE ‖ prefix.z NW words LE ‖ n a-mask bytes (&1) ‖ 0xFF ‖
    //   cz pairs (first u16-LE, second u16-LE)*          — global phase EXCLUDED.
    // Empty key (n_plan == 0) = clean/identity residual.
    DiagNormalForm parse_plan_key(const uint8_t* plan_key, int n_plan) const {
        DiagNormalForm nf;
        nf.a.assign((size_t)n, 0);
        nf.prefix = Pauli(n);
        nf.diagonal_class = true;
        const int NW = (n + 63) / 64;
        if (plan_key && n_plan > 0) {
            int p = 0;
            auto rd_words = [&](std::vector<uint64_t>& dst) {
                for (int w = 0; w < NW; ++w) {
                    uint64_t v = 0;
                    for (int b = 0; b < 8; ++b)
                        if (p < n_plan) v |= ((uint64_t)plan_key[p++]) << (b * 8);
                    if (w < (int)dst.size()) dst[(size_t)w] = v;
                }
            };
            rd_words(nf.prefix.x);
            rd_words(nf.prefix.z);
            for (int q = 0; q < n; ++q)
                if (p < n_plan) nf.a[(size_t)q] = (uint8_t)(plan_key[p++] & 1);
            if (p < n_plan && plan_key[p] == 0xFF) ++p;   // separator
            while (p + 3 < n_plan) {
                int f = (int)plan_key[p] | ((int)plan_key[p + 1] << 8);
                int s = (int)plan_key[p + 2] | ((int)plan_key[p + 3] << 8);
                // T5 step 0 hardening (T3 review Minor): plan_structure accepts ARBITRARY caller
                // bytes (unlike materialize_shot, which only ever replays sampler-serialized
                // keys), so refuse an out-of-range cz wire before build_shot_law consumes it.
                if (f < 0 || s < 0 || f >= n || s >= n)
                    throw std::invalid_argument(
                        "plan_structure: cz pair (" + std::to_string(f) + ", " +
                        std::to_string(s) + ") out of range [0, " + std::to_string(n) +
                        ") — malformed plan key");
                nf.cz.emplace_back(f, s);
                p += 4;
            }
        }
        return nf;
    }

    // ── V3 Born-weighted observable channel state ────────────────────────────────────────
    bool born_obs = false;              // single logical observable, diagonal table
    int born_idx = -1;                  // its observable index
    Pauli born_W;                       // combined terminal-read record operator
    std::mt19937_64 obs_gen;            // DEDICATED rng: det-channel byte streams unchanged
    double m_clean = 0.0;               // ⟨W⟩ on the bare state (clean-shot weight)
    int obs_bit_cur = 0;                // per-shot scratch consumed by the tail/sink
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
    // Fix B (declared stream change): fixed-point per-shot rf coins.
    //   draw_rf previously consumed nrf uniform_real draws from rf_gen (a dedicated mt19937_64).
    //   The new scheme draws 64-bit words from a per-shot ShotRng-derived stream and compares
    //   each slot against a precomputed uint64_t threshold rf_thresh[sl] = (uint64_t)(p_sl * 2^64)
    //   so fire_sl = (u64 < rf_thresh[sl]).  Cost: one integer compare per slot (~86 ns/shot at
    //   nrf=86) versus one double draw + branch per slot (~300 ns).  rf_gen is removed from the
    //   hot path.  The rf coins are now a pure function of (seed, shot_index) — routing-
    //   independent and deterministic without a sequential generator.  The detector byte streams
    //   CHANGE (pre-approved declared stream change); all statistical and routing-independence
    //   gates are re-verified.
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
    TwirlOutcome o_rec;
    std::vector<uint8_t> plankey_;   // per-shot serialized canonical residual plan (record-hash key)
    std::vector<FiredPauli> fired;
    std::vector<long long> chancnt;               // per-channel 1-bit counts
    bool count_channels_ = false;   // disabled by default; set in run() when profiling or QEC_TW_COUNT
    long clean = 0, diag = 0, checked = 0, used = 0, fb = 0;
    // v2.7 selfcheck once-per-sampler: the oracle window is bound to the FIRST run() of a
    // sampler lifetime. This latch is set by run() (shots > 0) and deliberately NOT reset by
    // set_seed() — the in-place reseed zeroes the `checked`/`pprchecked` gating counters,
    // which would otherwise re-arm the window on every seeded call. rebuild() constructs a
    // fresh Impl, so the window re-arms there (fresh-construction semantics). STREAM-NEUTRAL:
    // a spent window behaves exactly like selfcheck=0, which is byte-identical to the
    // window-on path (ratified; oracle tests/test_selfcheck_once.py).
    bool selfcheck_spent_ = false;
    // ── V3-T3 R1: per-shot exact fallback state ───────────────────────────────────────────
    // A guard trip no longer drops the shot: the SAME channel bits (and obs bit) are computed
    // exactly — residual applied to a fresh bare copy, certified generators measured
    // sequentially (they commute: the sequential joint IS the true channel law), channel bit
    // c = parity(σ & mask_c), and in born mode the observable measured on the SAME collapsed
    // state after the generators (record convention). All exact-path draws come from the
    // DEDICATED fb_gen below, so the collapse/coin streams every non-fallback shot consumes
    // are byte-identical to the pre-R1 engine; `fb` stays as the diagnostic count.
    std::mt19937_64 fb_gen;                       // dedicated exact-fallback rng
    std::uniform_real_distribution<double> fbu{0.0, 1.0};
    FramedSuperposition fbwork{0};                // reusable working state (buffer-reusing =)
    std::vector<uint64_t> fb_sig;                 // GW-word σ scratch (measured generator signs)
    std::vector<uint64_t> retain_sig;             // S2.2: GW-word σ delivered to a retaining sink
    // Lazy per-alt composed error tableaus for the PPR exact fallback (E†·g·E read
    // conjugation, the production process_shot_general_ pattern). null = not built yet;
    // single stored-general atoms are shared, everything else composed on first use.
    std::vector<std::shared_ptr<const CliffordTableau>> fb_alt_tab;
    DiagErrorSampler::SparseBatch batch;
    DiagNormalForm nf;                            // reused across shots
    std::vector<uint64_t> key_amask_s;            // canonical key (strip applied)
    std::vector<std::pair<int, int>> cz_key;
    std::vector<int> zflip;                       // −Z stripped-partner corrections
    // Per-shot derived RNG: seeded from (seed_, done) so coin draws are a pure function
    // of (seed, shot_index, plan) — independent of which fast/slow path was taken.
    // Invariant: shot_rng is re-seeded O(1) at the top of every shot; coin_res/coin_left are reset
    // to 0 at the same point so the 64-way reservoir never carries state across shots.
    // seed_: the constructor seed, stored so run() can derive per-shot generators.
    uint64_t seed_ = 0;
    // speed-kill T1: true when the S channels (sch) were drawn from mrng(seed + 4242) at
    // construction (!opt.circuit_channels) — the channel set itself is then seed-dependent
    // physics and set_seed() must refuse (the caller reconstructs instead).
    bool synthetic_channels_ = false;
    // ShotRng: counter-based per-shot generator with O(1) seeding and ~ns draws.
    // splitmix64 step (Steele & Vigna): passes BigCrush; adequate for fair-coin draws.
    // Coins are a pure function of (seed_, shot_index, plan) — routing-independent.
    struct ShotRng {
        uint64_t s = 0;
        void seed(uint64_t base_seed, uint64_t shot_idx) noexcept {
            // Mix seed and shot index into a single counter value; the additive constant
            // 0x9E3779B97F4A7C15 (golden-ratio Fibonacci hash) keeps shot 0 distinct from
            // the raw seed while the shot-index mix via 0xBF58476D1CE4E5B9 avalanches all bits.
            s = (base_seed + 0x9E3779B97F4A7C15ULL) ^ (shot_idx * 0xBF58476D1CE4E5B9ULL);
        }
        uint64_t next() noexcept {            // splitmix64 step
            uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }
        // [0,1) double from top 53 bits (IEEE 754 mantissa trick)
        double next_double() noexcept {
            return (next() >> 11) * (1.0 / (1ULL << 53));
        }
    };
    ShotRng shot_rng;                             // per-shot generator; re-seeded O(1) per shot
    uint64_t coin_res = 0;
    int coin_left = 0;                            // 64-way coin reservoir (fast path; per-shot)
    double t_draw = 0.0, wall = 0.0;

    Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
         const std::vector<std::pair<int, int>>& reads,
         const std::vector<std::vector<int>>& detectors,
         const std::vector<std::pair<int, std::vector<int>>>& observables,
         uint64_t seed, const TwirlRecordOptions& opt);
    // Shared-cache overload: adopts the provided caches (creates fresh ones for null pointers).
    Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
         const std::vector<std::pair<int, int>>& reads,
         const std::vector<std::vector<int>>& detectors,
         const std::vector<std::pair<int, std::vector<int>>>& observables,
         uint64_t seed, const TwirlRecordOptions& opt,
         const SharedPlanCaches& sc);
    // Write-back the fast index to the shared owner (if any) so the next rebuild inherits it.
    ~Impl() {
        if (sfi_ && sfi_->KW == KW) {
            sfi_->lkeys  = std::move(lkeys);
            sfi_->lplans = std::move(lplans);
            sfi_->lidx   = std::move(lidx);
            sfi_->ftab   = std::move(ftab);
            sfi_->fslots = fslots;
            sfi_->fcount = fcount;
        }
    }
    bool run(long shots);

    // ── speed-kill T1: in-place reseed ────────────────────────────────────────────────────
    // Resets EXACTLY the state a fresh Impl construction (re)derives from `seed`,
    // reconstructing nothing. The seed-dependent inventory (exhaustive — verified against
    // every `seed` use in the constructor plus every cumulative member run() mutates):
    //   * seed_               — base of the per-shot counter ShotRng (shot_rng.seed(seed_,
    //                           done) at the top of every shot; rf coins from a local ShotRng
    //                           at (seed_ ^ 0x6F62E1D3C4B5A697, done_rf_)) — coins are a pure
    //                           function of (seed, shot_index, plan), nothing else to reset;
    //   * smp                 — DiagErrorSampler noise-event stream: reseed() re-derives the
    //                           xoshiro state AND the per-channel initial geometric skips
    //                           exactly as construction does (its ctor's last step IS
    //                           reseed(seed));
    //   * obs_gen             — born-observable stream (ctor seeds it only under born_obs;
    //                           mirrored — it is never drawn otherwise);
    //   * fb_gen / dec_gen    — dedicated exact-fallback / Born-decision streams (their
    //                           uniform distributions are reset for standard-conformance;
    //                           libstdc++'s carry no state);
    //   * done_rf_            — rf ShotRng shot counter (fresh Impl: 0; cumulative otherwise);
    //   * the cumulative run() counters clean/diag/checked/used/fb, pprshots/pprchecked,
    //     obs_ones_/ps_acc_/ps_obs_ones_, chancnt, t_draw/wall — zeroed so the selfcheck
    //     oracle window (`checked`/`pprchecked` gating) and every reporting accessor behave
    //     exactly as on a fresh construction. Plan-cache hit/miss counters live in the SHARED
    //     caches, which survive rebuild() too — untouched, same as the rebuild path.
    // Seed-INDEPENDENT state deliberately kept warm (pure functions of the compiled circuit,
    // already shared across rebuild()s): propagation table, CertifiedGroupPlanes, channel
    // compilation, alt blobs/atoms, plan caches + fast-route index, lazily built fb_alt_tab,
    // born-decision memos (cleared per run() anyway). materialize_shot() is const and fully
    // record-driven — outstanding barrier buffers are unaffected by an in-place reseed.
    bool set_seed(uint64_t seed) {
        if (synthetic_channels_) return false;   // sch drawn from mrng(seed + 4242): rebuild required
        seed_ = seed;
        smp->reseed(seed);
        if (born_obs) obs_gen.seed(seed ^ 0xA5A5A5A55A5A5A5Aull);
        fb_gen.seed(seed ^ 0xE4AC7F0DD15C0DE5ull);
        fbu.reset();
        dec_gen.seed(seed ^ 0x2B1DEC1510DEC0DEull);
        dec_dist_.reset();
        done_rf_ = 0;
        clean = diag = checked = used = fb = 0;
        pprshots = 0; pprchecked = 0;
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
    // GF(2)-LINEAR hash (+ constant): lhash(a ⊕ b) = lhash(a) ⊕ lhash(b) ⊕ C. This lets each
    // alt carry a precomputed hash token (blob header), so the shot hash is a pure XOR of
    // tokens — available BEFORE the key words are assembled (prefetch + slot overlap).
    uint64_t lhash(const uint64_t* k) const {
        uint64_t h = 0xcbf29ce484222325ull;                        // the constant C
        for (int w = 0; w < KW; ++w) {
            const uint64_t x = k[w];
            h ^= rotl64(x, (7 * w + 3) & 63) ^ rotl64(x, (13 * w + 29) & 63) ^ x;
        }
        return h;
    }
    // Flat open-addressed find table (hot path): slot = KW key words + plan ptr; empty ⇔ ptr
    // word == 0; linear probing, ≤50% load. One probe = one contiguous 26-word read.
    void finsert_raw(const uint64_t* key, const CachedPlan* p) {
        size_t s2 = lhash(key) & (fslots - 1);
        for (;;) {
            uint64_t* slot = &ftab[s2 * (size_t)(KW + 1)];
            if (slot[KW] == 0) {
                std::copy(key, key + KW, slot);
                slot[KW] = (uint64_t)(uintptr_t)p; ++fcount; return;
            }
            s2 = (s2 + 1) & (fslots - 1);
        }
    }
    void fgrow() {
        fslots = fslots ? fslots * 2 : 16384;
        ftab.assign(fslots * (size_t)(KW + 1), 0);
        fcount = 0;
        for (size_t i = 0; i < lkeys.size(); ++i) finsert_raw(lkeys[i].data(), lplans[i]);
    }
    const CachedPlan* ffind_h(const uint64_t* k, uint64_t h) const {
        size_t s2 = h & (fslots - 1);
        for (;;) {
            const uint64_t* slot = &ftab[s2 * (size_t)(KW + 1)];
            if (slot[KW] == 0) return nullptr;
            if (std::equal(k, k + KW, slot)) return (const CachedPlan*)(uintptr_t)slot[KW];
            s2 = (s2 + 1) & (fslots - 1);
        }
    }
    const CachedPlan* ffind(const uint64_t* k) const { return ffind_h(k, lhash(k)); }
    void lindex_insert(const std::vector<uint64_t>& key, const CachedPlan* p) {
        auto& v = lidx[lhash(key.data())];
        for (int i : v) if (lkeys[(size_t)i] == key) return;      // idempotent
        lkeys.push_back(key); lplans.push_back(p); v.push_back((int)lkeys.size() - 1);
        if ((fcount + 1) * 2 > fslots) fgrow();                   // grow re-inserts everything
        else finsert_raw(key.data(), p);
    }
    const CachedPlan* lindex_find(const uint64_t* k) const {
        auto it = lidx.find(lhash(k));
        if (it == lidx.end()) return nullptr;
        for (int i : it->second)
            if (std::equal(k, k + KW, lkeys[(size_t)i].data())) return lplans[(size_t)i];
        return nullptr;
    }
    bool plan_local_key(const CachedPlan& p, std::vector<uint64_t>& key) const {
        key.assign((size_t)KW, 0);
        for (size_t w = 0; w < p.key_amask.size() && w < (size_t)NW; ++w) key[w] = p.key_amask[w];
        for (const auto& pr : p.key_cz) {
            auto it = pidx.find(pr);
            if (it == pidx.end()) return false;
            key[(size_t)NW + (size_t)(it->second >> 6)] |= 1ull << (it->second & 63);
        }
        return true;
    }
    // Tier-B channel rows for one plan (CW words/row; channel-set token guarded). Rows are
    // channel-set-dependent and NOT serialized — build them off the per-shot clock.
    void build_ch_rows(const CachedPlan& plan) const {
        plan.ch_baseline.assign((size_t)CW, 0);
        for (int c = 0; c < nchan; ++c)
            if (par(plan.base0, sch[c].mask)) plan.ch_baseline[c >> 6] |= (1ull << (c & 63));
        plan.ch_coin_rows.assign(plan.coin_masks.size(), std::vector<uint64_t>((size_t)CW, 0));
        for (size_t k = 0; k < plan.coin_masks.size(); ++k)
            for (int c = 0; c < nchan; ++c)
                if (par(plan.coin_masks[k], sch[c].mask))
                    plan.ch_coin_rows[k][c >> 6] |= (1ull << (c & 63));
        plan.ch_kernel_rows.assign(plan.kernel_masks.size(), std::vector<uint64_t>((size_t)CW, 0));
        for (size_t k = 0; k < plan.kernel_masks.size(); ++k)
            for (int c = 0; c < nchan; ++c)
                if (par(plan.kernel_masks[k], sch[c].mask))
                    plan.ch_kernel_rows[k][c >> 6] |= (1ull << (c & 63));
        plan.ch_token = chtok;
    }
    // ── PPR plan family (V2-T1): memo + per-plan channel rows + cached fold Born weights.
    // PPR shots (≥1 fired alt with a PPR-retry atom) compose in the PPR algebra and emit
    // σ-channel bits as baseline ⊕ prefix rows ⊕ fair coins ⊕ Born-weighted foldable folds
    // (unfoldable kernel dirs never touch σ). Fold weights depend only on (plan, bare) —
    // cached on the plan like the channel rows.
    void build_ppr_rows(const PprCachedPlan& pl) const {
        pl.ch_baseline.assign((size_t)CW, 0);
        for (int c = 0; c < nchan; ++c)
            if (par(pl.base0, sch[c].mask)) pl.ch_baseline[c >> 6] |= (1ull << (c & 63));
        pl.ch_coin_rows.assign(pl.coin_masks.size(), std::vector<uint64_t>((size_t)CW, 0));
        for (size_t k = 0; k < pl.coin_masks.size(); ++k)
            for (int c = 0; c < nchan; ++c)
                if (par(pl.coin_masks[k], sch[c].mask))
                    pl.ch_coin_rows[k][c >> 6] |= (1ull << (c & 63));
        pl.ch_kernel_rows.assign(pl.kernel_masks.size(), std::vector<uint64_t>((size_t)CW, 0));
        for (size_t k = 0; k < pl.kernel_masks.size(); ++k) {
            if (pl.kernel_masks[k].empty()) continue;      // unfoldable: no σ row
            for (int c = 0; c < nchan; ++c)
                if (par(pl.kernel_masks[k], sch[c].mask))
                    pl.ch_kernel_rows[k][c >> 6] |= (1ull << (c & 63));
        }
        pl.ch_token = chtok;
    }
    double obs_u01() { return std::uniform_real_distribution<double>(0.0, 1.0)(obs_gen); }
    // Readout-flip layer (Stim M(p)).
    //
    // draw_rf (FIX B — declared stream change): one Bernoulli(p_sl) coin per noisy record slot,
    // per shot, drawn from a per-shot ShotRng-derived stream.  The old scheme called rf_u01(rf_gen)
    // (a sequential mt19937_64) once per slot.  The new scheme draws 64-bit words from shot_rng
    // (already reseeded at the top of each shot) using a separate counter domain (shots*2+1 to
    // avoid colliding with the coin draw domain at shots*2+0 which the plan path uses), and
    // compares each slot against precomputed uint64_t threshold rf_thresh[sl].  Cost: one integer
    // compare per slot (~1 ns) versus one double conversion + branch (~3-4 ns) — ~86 ns vs ~300 ns
    // at nrf=86.  The coins are now a pure function of (seed, shot_index) — routing-independent.
    // NOTE: the rf ShotRng derives from a SEPARATE per-shot generator seeded with a distinct
    // domain tag (done*2+1) to prevent any overlap with the plan-coin ShotRng (done*2+0 effectively,
    // since shot_rng is seeded from seed_+(done) and coin draws happen via shot_rng.next()).
    // We use a local ShotRng seeded from (seed_ ^ 0x6F62E1D3C4B5A697ull, done) to maintain
    // independence and byte-stability of the rf stream relative to any future plan-coin changes.
    //
    // apply_rf (FIX A — byte-identical): transposed column form.
    //   Old: for each channel c, popcount(rf_coins & chan_rf_mask[c]) => O(nchan * RFW) work.
    //   New: iterate SET bits of rf_coins[w] via ctz; for each fired slot sl, XOR rf_col[sl]
    //   into bits and conditionally flip obs_bit_cur via rf_obs_bit[sl].  Only O(fired * CW)
    //   work where fired ~ nrf * p_mean << nchan.  GF(2) linearity: same rf_coins in => same
    //   bits out — byte-identical to the old row-loop form.
    //   born_dec_rf_mask is NOT transposed (read in emit_decisions per decision, infrequent path).
    void draw_rf() {
        // FIX B: fixed-point per-shot Bernoulli draws from a dedicated per-shot ShotRng.
        // Domain tag 0x6F62E1D3C4B5A697ull (same as the old rf_gen seed xor) gives independence
        // from shot_rng (which is seeded from seed_ and used for plan coins).
        ShotRng rng;
        rng.seed(seed_ ^ 0x6F62E1D3C4B5A697ull, (uint64_t)done_rf_);
        for (int w = 0; w < RFW; ++w) rf_coins[(size_t)w] = 0;
        for (int sl = 0; sl < nrf; ++sl) {
            // One uint64 draw per slot; fire if draw < threshold (fixed-point Bernoulli).
            // rf_thresh[sl] = (uint64_t)(p_sl * 2^64), with p_sl=0 -> thresh=0 (never fires),
            // p_sl=1 -> thresh=2^64 which wraps to 0 but p_sl<=1 always holds in practice.
            if (rng.next() < rf_thresh[(size_t)sl])
                rf_coins[sl >> 6] |= 1ull << (sl & 63);
        }
    }
    long done_rf_ = 0;   // shot counter for the rf ShotRng; incremented by caller alongside done
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
    // Input Pauli frame relabel (Task 4 / decoder-feedback): per-shot sign flip on sigma bits.
    // Gated on has_input_frame_ (false for frame-free circuits => no-op, byte-identical).
    // cur_frame_row_ is set once per shot before the first exit point.
    //
    // ANCHOR-TIME CONJUGATION (decoder-feedback engine fix, 2026-07-22): the frame is a Pauli
    // P declared at t=0 (the stage port), but every relabel target (channel generator combos,
    // born_W, DECISION operators) lives at the engine ANCHOR = end of the deferred unitary
    // stream (the bare state has the full stream baked in; terminal reads / alt atoms are
    // end-time objects). Heisenberg identity: the anchor state is U P|ψ_in⟩ = (U P U†) U|ψ_in⟩
    // = P'|bare⟩ with P' = U P U† (U = the stage's full unitary stream), so every
    // anticommutation must be taken against P', NOT P. Each port component (X_i / Z_i) is
    // propagated ONCE at setup through the one propagate_atom kernel; sp(·,·) is bilinear, so
    // the per-component channel rows below are exact for any per-shot frame product. The phase
    // of P' never enters: P'ρP'† is phase-invariant and sp is support-only (derived, not
    // empirically tuned — no sign convention exists to get wrong). With an empty unitary
    // stream P' = P and the rows reduce EXACTLY to the old pxrow[q]/pzrow[q] lookups.
    // A component whose image leaves the Pauli class (non-Clifford gate in the port cone) is
    // POISONED: compile stays valid (frame-free use unaffected); run() loud-refuses only if a
    // shot actually supplies that component.
    bool has_input_frame_ = false;
    std::vector<int> frame_port_qubits_;   // deferred-space qubit indices
    const uint8_t* frame_data_ = nullptr;
    int frame_stride_ = 0;
    const uint8_t* cur_frame_row_ = nullptr;
    std::vector<uint64_t> frame_xrow_, frame_zrow_;  // pnq×CW channel rows of P'_{X_i}/P'_{Z_i}
    // 2·pnq per-component state: 0 = clean Pauli image (rows/bits active); 1 = HARD poison
    // (non-Pauli image whose support can touch a relabel target — run() refuses if supplied);
    // 2 = SOFT poison (non-Pauli image support-disjoint from EVERY certified generator and
    // Born operator — an EXACT no-op: the unitary P' commutes elementwise with every measured
    // operator, so ⟨bare|P'† f({Π}) P'|bare⟩ = ⟨bare|f({Π})|bare⟩; rows/bits stay zero, run()
    // allows it, and the frame's onward carry stays the orchestrator's job exactly as for
    // clean Pauli images — the engine never applies any frame to a state).
    std::vector<uint8_t> frame_poison_;
    std::vector<uint8_t> frame_obs_bits_;            // 2·pnq: sp(born_W, P'_comp)
    std::vector<std::vector<uint8_t>> frame_dec_bits_;  // per born_dec j: 2·pnq sp(W_j, P'_comp)

    void apply_input_frame(uint64_t* bits) {
        if (!has_input_frame_ || !cur_frame_row_) return;
        const int pnq = (int)frame_port_qubits_.size();
        for (int i = 0; i < pnq; ++i) {
            if (cur_frame_row_[i]) {                              // X on port qubit i at t=0
                const uint64_t* row = &frame_xrow_[(size_t)i * CW];
                for (int w = 0; w < CW; ++w) bits[w] ^= row[w];
            }
            if (cur_frame_row_[pnq + i]) {                        // Z on port qubit i at t=0
                const uint64_t* row = &frame_zrow_[(size_t)i * CW];
                for (int w = 0; w < CW; ++w) bits[w] ^= row[w];
            }
        }
        if (born_obs) {
            for (int i = 0; i < pnq; ++i) {
                if (cur_frame_row_[i]       && frame_obs_bits_[(size_t)i])         obs_bit_cur ^= 1;
                if (cur_frame_row_[pnq + i] && frame_obs_bits_[(size_t)(pnq + i)]) obs_bit_cur ^= 1;
            }
        }
    }
    // Born-DECISION frame relabel: sp(W_j, P') for the current shot's frame row (0 when
    // frame-free). P' is the ANCHOR-TIME image of the per-shot frame product (see the member
    // block comment); sp is bilinear, so the flip is the parity of the supplied frame bits
    // against the setup-precomputed per-component bits sp(W_j, P'_comp). Returns the flip bit;
    // measuring W_j on the framed state = frame-free outcome ⊕ this (the probability p_j is
    // unchanged — the frame is a deterministic bit relabel, not a reweight).
    uint8_t frame_born_flip(size_t j) const {
        if (!has_input_frame_ || !cur_frame_row_) return 0;
        const int pnq = (int)frame_port_qubits_.size();
        const uint8_t* fb = frame_dec_bits_[j].data();
        int par = 0;
        for (int i = 0; i < pnq; ++i) {
            if (cur_frame_row_[i]       && fb[i])       par ^= 1;
            if (cur_frame_row_[pnq + i] && fb[pnq + i]) par ^= 1;
        }
        return (uint8_t)par;
    }
    // V3: lazy per-plan observable channel data (identity-prefix classify_observable;
    // parities against the plan's base0/coin_masks; Born weight off the bare state).
    void ensure_obs(const CachedPlan& plan) const {
        if (plan.obs_ready) return;
        DiagNormalForm onf;
        onf.prefix = Pauli(n);
        onf.a.assign((size_t)n, 0);
        onf.diagonal_class = true;
        for (int q = 0; q < n; ++q)
            if ((plan.key_amask[q >> 6] >> (q & 63)) & 1) onf.a[q] = 1;
        onf.cz = plan.key_cz;
        ObsChannel oc = classify_observable(G, onf, born_W);
        plan.obs_guard = (uint8_t)((oc.guard || plan.kappa > 0) ? 1 : 0);
        if (std::getenv("QEC_TW_OBSDBG") && plan.obs_guard)
            fprintf(stderr, "[obsdbg] guard=%d reach=%d kappa=%d\n",
                    (int)oc.guard, (int)oc.reachable, plan.kappa);
        plan.obs_par_coin.assign(plan.coin_masks.size(), 0);
        plan.obs_m = 0.0;
        plan.obs_par_base = 0;
        if (!plan.obs_guard && oc.reachable) {
            const std::complex<double> ev = bare.pauli_expectation(oc.lam);
            if (std::abs(ev.imag()) > 1e-9)
                throw std::logic_error("ensure_obs: complex observable weight");
            plan.obs_m = ev.real();
            plan.obs_par_base = (uint8_t)par(oc.mask, plan.base0);
            for (size_t k = 0; k < plan.coin_masks.size(); ++k)
                plan.obs_par_coin[k] = (uint8_t)par(oc.mask, plan.coin_masks[k]);
        }
        plan.obs_ready = 1;
    }
    // V3: lazy PPR-plan observable channel data (classify_observable_ppr on the LIVE
    // composed rot content — inherently identity-prefix; guard = any FOLDABLE kernel dir,
    // mirroring the diagonal kappa>0 rule; unfoldable-only plans are in-scope per the
    // reference gate).
    void ensure_obs_ppr(const PprCachedPlan& pl, const PprNormalForm& pnf) const {
        if (pl.obs_ready) return;
        bool any_fold = false;
        for (uint8_t f : pl.kernel_foldable) if (f) { any_fold = true; break; }
        ObsChannel oc = classify_observable_ppr(G, pnf, born_W);
        pl.obs_guard = (uint8_t)((oc.guard || any_fold) ? 1 : 0);
        pl.obs_par_coin.assign(pl.coin_masks.size(), 0);
        pl.obs_m = 0.0;
        pl.obs_par_base = 0;
        if (!pl.obs_guard && oc.reachable) {
            const std::complex<double> ev = bare.pauli_expectation(oc.lam);
            if (std::abs(ev.imag()) > 1e-9)
                throw std::logic_error("ensure_obs_ppr: complex observable weight");
            pl.obs_m = ev.real();
            pl.obs_par_base = (uint8_t)par(oc.mask, pl.base0);
            for (size_t k = 0; k < pl.coin_masks.size(); ++k)
                pl.obs_par_coin[k] = (uint8_t)par(oc.mask, pl.coin_masks[k]);
        }
        pl.obs_ready = 1;
    }
    void build_ppr_folds(const PprCachedPlan& pl) const {
        pl.fold_p1.assign(pl.kernel_logicals.size(), 0.0);
        for (size_t j = 0; j < pl.kernel_logicals.size(); ++j) {
            if (!pl.kernel_foldable[j]) continue;
            const std::complex<double> ev = bare.pauli_expectation(pl.kernel_logicals[j]);
            if (std::abs(ev.imag()) > 1e-9)                     // Hermitian rep => real <Lambda>
                throw std::logic_error("build_ppr_folds: complex fold expectation");
            // fold fires iff (outcome ⊕ kernel_base) == 1; outcome=1 w.p. (1−⟨Λ⟩)/2
            const double p_out1 = (1.0 - ev.real()) / 2.0;
            pl.fold_p1[j] = pl.kernel_base[j] ? (1.0 - p_out1) : p_out1;
        }
        pl.folds_built = 1;
    }

    // ── V3-T3 R1: per-shot exact fallback ─────────────────────────────────────────────────
    double fb_u01() { return fbu(fb_gen); }
    // Measure every certified generator sequentially on the residual-applied working state
    // (a commuting family: the sequential joint IS the true channel law), σ_i = (outcome==−1),
    // channel bit c = parity(σ & mask_c) — the identical mask semantics the σ record path
    // uses. In born mode the observable is measured on the SAME collapsed state AFTER the
    // generators (the record convention), so the (σ, W) joint is the true channel's. Every
    // draw comes from fb_gen (dedicated stream): non-fallback shots stay byte-identical.
    void exact_sigma_bits(uint64_t* bits) {
        for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
        fb_sig.assign((size_t)GW, 0);
        for (int i = 0; i < ng; ++i)
            if (fbwork.measure_pauli(gens[(size_t)i], fb_u01()) < 0)
                fb_sig[(size_t)(i >> 6)] |= (1ull << (i & 63));
        for (int c = 0; c < nchan; ++c)
            if (par(fb_sig, sch[(size_t)c].mask)) bits[c >> 6] |= (1ull << (c & 63));
        if (born_obs)
            obs_bit_cur = (fbwork.measure_pauli(born_W, fb_u01()) < 0) ? 1 : 0;
    }
    // DIAGONAL exact shot: apply the residual R = P·C as gates to a fresh bare copy
    // (C = S^a then CZ, then the prefix X^v/Z^z — twirl_collapse's step-4 order; the global
    // phase is unobservable), then measure. Serves the record-path fallback (κ≥2 / mid-chain
    // guard), the born κ>0 shots, and the born obs_guard shots.
    void exact_diag_shot(const DiagNormalForm& snf, uint64_t* bits) {
        fbwork = bare;
        for (int q = 0; q < n && q < (int)snf.a.size(); ++q)
            if (snf.a[q] & 1) fbwork.apply_clifford(/*S*/ 1, q, 0);
        for (const std::pair<int, int>& e : snf.cz)
            fbwork.apply_clifford(/*CZ*/ 7, e.first, e.second);
        for (int q = 0; q < n; ++q) {
            if (snf.prefix.xbit(q)) fbwork.apply_clifford(/*X*/ 3, q, 0);
            if (snf.prefix.zbit(q)) fbwork.apply_clifford(/*Z*/ 5, q, 0);
        }
        exact_sigma_bits(bits);
    }
    // Lazy composed error tableau of one alt (framed_sampler's compose_alt_general_ pattern,
    // production-proven): forward rows folded through the alt's atoms in stream order (x_atom
    // before z_atom within a Y — compose_fired's order); diagonal atoms promoted by gate
    // decomposition (γ is unobservable in conjugation); a single stored-general atom is
    // shared as-is. Built on FIRST fallback use only — in-distribution shots never pay this.
    const CliffordTableau& fb_alt_tableau(size_t ai) {
        if (fb_alt_tab.empty()) fb_alt_tab.assign(sides.size(), nullptr);
        if (fb_alt_tab[ai]) return *fb_alt_tab[ai];
        std::vector<const PropResult*> atoms;
        for (const FiredPauli& f : sides[ai]) {
            const auto [loc, qi] = table.find_slot(f.location_index, f.qubit, "twirl_fb");
            if (f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) atoms.push_back(&loc->x_atom[qi]);
            if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) atoms.push_back(&loc->z_atom[qi]);
        }
        if (atoms.size() == 1 && atoms[0]->general) {
            fb_alt_tab[ai] = atoms[0]->general;
            return *fb_alt_tab[ai];
        }
        std::vector<CliffordTableau> temps;
        int nd = 0;
        for (const PropResult* a : atoms) if (!a->general) ++nd;
        temps.reserve((size_t)nd);                             // stable: all pushes precede tab()
        std::vector<int> tidx(atoms.size(), -1);
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (atoms[i]->general) continue;
            const DiagPauliClifford& D = atoms[i]->c_prop;
            CliffordTableau T(n);
            for (int q = 0; q < n; ++q)
                switch (D.a[q] & 3) {
                    case 1: T.left_s(q); break;
                    case 2: T.left_z(q); break;
                    case 3: T.left_sdg(q); break;
                    default: break;
                }
            for (int i2 = 0; i2 < n; ++i2)
                for (int j2 = i2 + 1; j2 < n; ++j2)
                    if (D.B.get(i2, j2)) T.left_cz(i2, j2);
            for (int q = 0; q < n; ++q)
                if (D.v[q]) T.left_x(q);
            tidx[i] = (int)temps.size();
            temps.push_back(std::move(T));
        }
        auto tab = [&](size_t i) -> const CliffordTableau& {
            return atoms[i]->general ? *atoms[i]->general : temps[(size_t)tidx[i]];
        };
        CliffordTableau E(n);
        for (int a = 0; a < n; ++a) {
            Pauli x(n); x.setx(a);
            Pauli z(n); z.setz(a);
            for (size_t i = 0; i < atoms.size(); ++i) {
                x = tab(i).forward_image(x);
                z = tab(i).forward_image(z);
            }
            E.Xrow[(size_t)a] = std::move(x);
            E.Zrow[(size_t)a] = std::move(z);
        }
        E.invalidate_dual();                       // rows set directly; dual_image reads forward rows
        fb_alt_tab[ai] = std::make_shared<CliffordTableau>(std::move(E));
        return *fb_alt_tab[ai];
    }
    // PPR (out-of-class) exact shot: measuring g on E|ψ⟩ equals measuring E†gE on |ψ⟩, so
    // each generator is pulled back through the fired alts' composed error tableaus
    // (last-fired conjugates first — the production process_shot_general_ pattern) and
    // measured on a fresh bare copy. Valid for ANY Clifford residual, including compositions
    // the PPR plan family refuses (non-commuting axes / plan fallback).
    void exact_ppr_shot(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1,
                        uint64_t* bits) {
        fbwork = bare;
        for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
        fb_sig.assign((size_t)GW, 0);
        for (int i = 0; i < ng; ++i) {
            Pauli M = gens[(size_t)i];
            for (int32_t e = e1 - 1; e >= e0; --e)
                M = fb_alt_tableau((size_t)b.ev[e]).dual_image(M);
            if ((M.phase & 1) != 0)                // Clifford conjugation preserves Hermiticity
                throw std::logic_error("exact_ppr_shot: conjugated generator not Hermitian");
            if (fbwork.measure_pauli(M, fb_u01()) < 0)
                fb_sig[(size_t)(i >> 6)] |= (1ull << (i & 63));
        }
        for (int c = 0; c < nchan; ++c)
            if (par(fb_sig, sch[(size_t)c].mask)) bits[c >> 6] |= (1ull << (c & 63));
        // V3 PPR-obs: born observable measured on the SAME collapsed state after the
        // generators (pulled back through the same tableau chain) — exact (σ, W) joint.
        if (born_obs) {
            Pauli MW = born_W;
            for (int32_t e = e1 - 1; e >= e0; --e)
                MW = fb_alt_tableau((size_t)b.ev[e]).dual_image(MW);
            if ((MW.phase & 1) != 0)
                throw std::logic_error("exact_ppr_shot: conjugated observable not Hermitian");
            obs_bit_cur = fbwork.measure_pauli(MW, fb_u01()) < 0 ? 1 : 0;
        }
    }
};

TwirlRecordSampler::Impl::Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
                               const std::vector<std::pair<int, int>>& reads,
                               const std::vector<std::vector<int>>& detectors,
                               const std::vector<std::pair<int, std::vector<int>>>& observables,
                               uint64_t seed, const TwirlRecordOptions& opt)
    : Impl(bare_in, deferred, reads, detectors, observables, seed, opt, SharedPlanCaches{}) {}

TwirlRecordSampler::Impl::Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
                               const std::vector<std::pair<int, int>>& reads,
                               const std::vector<std::vector<int>>& detectors,
                               const std::vector<std::pair<int, std::vector<int>>>& observables,
                               uint64_t seed, const TwirlRecordOptions& opt,
                               const SharedPlanCaches& sc)
    : bare(bare_in), n(bare_in.n()), disk_path(opt.disk_path), selfcheck(opt.selfcheck),
      cache(sc.plans ? sc.plans : std::make_shared<TwirlPlanCache>()),
      pcache(sc.ppr   ? sc.ppr   : std::make_shared<PprPlanCache>()) {
    int want_nchan = opt.want_nchan;           // --nchan N: channel-count scaling measurements
    const bool circ_chan = opt.circuit_channels;   // --circuit-channels: compile DETECTORs to σ-masks
    synthetic_channels_ = !circ_chan;              // speed-kill T1: mrng-drawn sch ⇒ set_seed refuses
    if (want_nchan < 1) want_nchan = 1;
    if (want_nchan > 512) want_nchan = 512;
    const double factor = opt.p_factor;
    gens = bare.certified_stabilizers();
    ng = (int)gens.size();
    GW = (ng + 63) / 64;
    G = CertifiedGroupPlanes::build(gens, n);
    table = build_propagation_table(deferred, /*ppr_retry=*/true);
    if (!table.all_in_class) { fprintf(stderr, "twirl_records: table not all-in-class\n"); setup_error = 3; return; }
    // V2-T1: CH-class (non-diagonal) atoms route through the PPR plan family
    // (compose_ppr + build_shot_law_ppr, oracle-certified 50/50 at 1e-12); shots whose
    // fired alts are all-diagonal keep the abelian fast path. Out-of-class compositions
    // and law fallbacks are counted in `fb` and EXCLUDED (loud, never silent-wrong).
    ppr_tables = table.has_general;

    // S channels: 16 singles + random products up to --nchan (default 32, matching the
    // twirl_collapse gate; --nchan 128 covers a full d5-scale detector set (108 channels) for
    // the channel-count scaling measurement the review demanded — per-channel cost is linear in
    // CHANNEL WORDS, and CW = ceil(nchan/64)).
    auto set_bit = [&](std::vector<uint64_t>& w, int i) { w[i >> 6] |= (1ULL << (i & 63)); };
    // Synthetic channels are dead weight under circuit_channels (the compiled channels
    // replace them below) — and the random-product loop must cap the subset size at ng:
    // drawing sz > ng DISTINCT indices never terminates (2026-07-16 fix: a 2-qubit gauge
    // circuit with ng=2 hung the compile). Skipping in circuit mode changes nothing
    // downstream (mrng is local); synthetic-mode draw sequences are unchanged whenever
    // ng > 5, i.e. on every existing benchmark.
    if (!circ_chan) {
        for (int i = 0; i < std::min(ng, 16) && (int)sch.size() < want_nchan; ++i) {
            SChan c; c.GS = gens[i]; c.mask.assign(GW, 0); set_bit(c.mask, i); sch.push_back(std::move(c));
        }
        std::mt19937_64 mrng(seed + 4242);
        while ((int)sch.size() < want_nchan && ng >= 2) {
            const int sz = std::min(ng, 2 + (int)(mrng() % 4));
            std::set<int> pick;
            while ((int)pick.size() < sz) pick.insert((int)(mrng() % ng));
            SChan c; c.mask.assign(GW, 0); Pauli P(n);
            for (int idx : pick) { P = Pauli::multiply(P, gens[idx]); set_bit(c.mask, idx); }
            c.GS = P; sch.push_back(std::move(c));
        }
    }
    // ── V2-T2: CIRCUIT channel compilation (--circuit-channels). Each DETECTOR's combined
    // record operator W = Π_k W_k (terminal reads) is classified against the certified group:
    //   IN_GROUP → σ-mask channel: bit_D = ref ⊕ (mask·σ); the mask is the generator
    //              PROVENANCE (the combination the RREF consumed), ref the exact reduction
    //              sign ((1−ε)/2 — the noiseless value);
    //   LOGICAL  → gauge channel (twirled semantics: DECLARED, excluded from the 5σ gate);
    //   ANTI     → refused (counted loudly; cannot be a deterministic record channel).
    // OBSERVABLEs with non-IN_GROUP operators are refused per the spec (workflow arc).
    if (circ_chan) {
        sch.clear();
        // provenance RREF over the generators' symplectic rows (combo-tracking)
        const int NW2 = (n + 63) / 64;
        struct PRow { std::vector<uint64_t> s; std::vector<uint64_t> c; int pc; };
        std::vector<PRow> prref;
        auto sympl = [&](const Pauli& P) {
            std::vector<uint64_t> s(2 * (size_t)NW2, 0);
            for (int w = 0; w < NW2; ++w) { s[w] = P.x[w]; s[NW2 + w] = P.z[w]; }
            return s;
        };
        auto sfirst = [&](const std::vector<uint64_t>& s) {
            for (size_t w = 0; w < s.size(); ++w)
                if (s[w]) return (int)w * 64 + __builtin_ctzll(s[w]);
            return -1;
        };
        for (int i = 0; i < ng; ++i) {
            PRow r; r.s = sympl(gens[i]); r.c.assign(GW, 0); set_bit(r.c, i);
            for (const PRow& e : prref)
                if ((r.s[e.pc >> 6] >> (e.pc & 63)) & 1) {
                    for (size_t w = 0; w < r.s.size(); ++w) r.s[w] ^= e.s[w];
                    for (int w = 0; w < GW; ++w) r.c[w] ^= e.c[w];
                }
            r.pc = sfirst(r.s);
            if (r.pc >= 0) prref.push_back(std::move(r));
        }
        auto record_pauli = [&](int k) {
            Pauli P(n);
            const int b2 = reads[(size_t)k].first, q = reads[(size_t)k].second;
            if (b2 == 0) P.setx(q);
            else if (b2 == 2) P.setz(q);
            else { P.setx(q); P.setz(q); P.phase = 1; }          // Y = i·XZ
            return P;
        };
        auto classify = [&](const std::vector<int>& recs, std::vector<uint64_t>& mask,
                            uint8_t& ref) {
            Pauli W(n);
            for (int k : recs) W = Pauli::multiply(W, record_pauli(k));
            Membership m = G.reduce(W);
            if (m.verdict != Membership::IN_GROUP) return m.verdict;
            std::vector<uint64_t> s = sympl(W);
            mask.assign(GW, 0);
            for (const PRow& e : prref)
                if ((s[e.pc >> 6] >> (e.pc & 63)) & 1) {
                    for (size_t w = 0; w < s.size(); ++w) s[w] ^= e.s[w];
                    for (int w = 0; w < GW; ++w) mask[w] ^= e.c[w];
                }
            for (uint64_t w : s)
                if (w) throw std::logic_error("provenance residual nonzero on IN_GROUP");
            if (m.rep.phase & 1) throw std::logic_error("non-Hermitian detector reduction");
            ref = (uint8_t)((m.rep.phase >> 1) & 1);
            return m.verdict;
        };
        auto record_product = [&](const std::vector<int>& recs) {
            Pauli Wp(n);
            for (int k : recs) Wp = Pauli::multiply(Wp, record_pauli(k));
            return Wp;
        };
        for (size_t di = 0; di < detectors.size(); ++di) {
            std::vector<uint64_t> mask;
            uint8_t ref = 0;
            const auto v = classify(detectors[di], mask, ref);
            if (v == Membership::IN_GROUP) {
                // >512 channels: LOUD compile refusal (final-review Important #1 — the old
                // truncation left detectors in NO classification list, i.e. silently-zero
                // Python columns; never-silent-wrong forbids that).
                if ((int)sch.size() >= 512) {
                    fprintf(stderr, "twirl_records: >512 deterministic detector channels — "
                                    "unsupported (raise the channel-word cap)\n");
                    setup_error = 2;
                    return;
                }
                SChan c; c.mask = std::move(mask); sch.push_back(std::move(c));
                chan_det.push_back((int)di);
                chan_ref.push_back(ref);
            } else if (v == Membership::LOGICAL) gauge_dets.push_back((int)di);
            else anti_dets.push_back((int)di);
        }
        // Task 2b.1: the empty gate relaxes — a barrier of only Born DECISIONs is valid (zero
        // syndrome = trivial decode). Fatal ONLY when there is no channel of ANY kind; when
        // DECISIONs are declared they are classified below and re-checked there. Decision-free
        // circuits keep the exact pre-2b.1 behavior (sch.empty() ⇒ error here).
        if (sch.empty() && opt.decisions.empty()) {
            fprintf(stderr, "twirl_records: no deterministic detector channels\n");
            setup_error = 2; return;
        }
        // V2-T4: deterministic (IN_GROUP) observables ARE record channels, compiled exactly
        // like detector channels and appended AFTER them (channel layout in the header). The
        // ch/diag benches' logical observables remain the refusal case (chan_obs empty there,
        // so the bench's byte-equality gate sees an unchanged channel set).
        for (const auto& kv : observables) {
            std::vector<uint64_t> mask;
            uint8_t ref = 0;
            const auto ov = classify(kv.second, mask, ref);
            if (ov == Membership::IN_GROUP) {
                // Same loud refusal as detectors (Important #1 / Minor #3: pushing a
                // truncated IN_GROUP observable into refused_obs misreported its class).
                if ((int)sch.size() >= 512) {
                    fprintf(stderr, "twirl_records: >512 channels (deterministic observables) — "
                                    "unsupported (raise the channel-word cap)\n");
                    setup_error = 2;
                    return;
                }
                SChan c; c.mask = std::move(mask); sch.push_back(std::move(c));
                chan_obs.push_back(kv.first);
                chan_ref.push_back(ref);
            } else if (ov == Membership::LOGICAL && born_idx < 0) {
                // V3: FIRST LOGICAL observable on a diagonal table → Born-weighted channel
                // (spec guards: a second logical observable falls to refusal below — the
                // biased coins would be conditionally independent, ⟨W₁W₂⟩ wrong; ANTI-class
                // observables — e.g. declared-byproduct-frame readouts whose raw record
                // product anticommutes with certified generators, code_switching — stay
                // refused: they need the frame fold, a recorded V3 open item).
                born_idx = kv.first;
                born_W = record_product(kv.second);
            } else {
                refused_obs.push_back(kv.first);
            }
        }
        if (born_idx >= 0) {
            if (std::getenv("QEC_TW_OBSDBG")) {
                Membership mb = G.reduce(born_W);
                int wx = 0, wz = 0;
                for (uint64_t w : born_W.x) wx += __builtin_popcountll(w);
                for (uint64_t w : born_W.z) wz += __builtin_popcountll(w);
                int bad = -1;
                for (int i = 0; i < ng && bad < 0; ++i)
                    if (Pauli::anticommute_bit(born_W, gens[i])) bad = i;
                fprintf(stderr, "[obsdbg-setup] born O%d W(x=%d,z=%d,n=%d) verdict=%d anti-gen=%d\n",
                        born_idx, wx, wz, born_W.n, (int)mb.verdict, bad);
            }
            born_obs = true;
            const std::complex<double> ev = bare.pauli_expectation(born_W);
            if (std::abs(ev.imag()) > 1e-9)
                throw std::logic_error("born observable: complex bare expectation");
            m_clean = ev.real();
            obs_gen.seed(seed ^ 0xA5A5A5A55A5A5A5Aull);
        }
        // ── Task 2b.1: DECISION classification ────────────────────────────────────────────
        // Each declared DECISION(k) parity is classified by the SAME G.reduce code path.
        //   IN_GROUP → deterministic decision: a σ-mask channel appended to `sch` (bit =
        //              ref ⊕ mask·σ); dec_chan_* map that channel back to decision index k.
        //   LOGICAL / ANTI → Born-measurement channel: store (k, W_dec); Born-sampled per shot.
        // (A DECISION may be EITHER class — a deterministic decision is a trivial coin; unlike a
        // DETECTOR, a nondeterministic DECISION is ALLOWED. Never inferred, always verified.)
        for (const auto& kv : opt.decisions) {
            const int k = kv.first;
            if (k + 1 > num_dec_) num_dec_ = k + 1;
            std::vector<uint64_t> mask;
            uint8_t ref = 0;
            const auto dv = classify(kv.second, mask, ref);
            if (dv == Membership::IN_GROUP) {
                if ((int)sch.size() >= 512) {
                    fprintf(stderr, "twirl_records: >512 channels (deterministic decisions) — "
                                    "unsupported (raise the channel-word cap)\n");
                    setup_error = 2; return;
                }
                SChan c; c.mask = std::move(mask); sch.push_back(std::move(c));
                dec_chan_slot_.push_back((int)sch.size() - 1);
                dec_chan_idx_.push_back(k);
                dec_chan_ref_.push_back(ref);
                // FIX: keep chan_ref.size() == sch.size() (nchan). Without this,
                // chan_ref is empty when there are only det-decision channels (no
                // regular detectors/observables), and the readout-flip layer's loop
                // `chan_ref[c] = ...` over c < nchan causes an OOB null-deref (segfault).
                chan_ref.push_back(ref);
                // Store records for the readout-flip layer (rf-mask + rec_invert accounting).
                dec_chan_recs_.push_back(kv.second);
            } else {
                // LOGICAL or ANTI: Born channel (the record operator, measured on the state).
                born_dec.push_back({k, record_product(kv.second)});
                // Deterministic record invert (Stim `!` + the H-elimination gadget's folded sign):
                // the recorded decision bit = measure_pauli(W_dec) ⊕ ⊕_{r∈recs} rec_invert[r].
                uint8_t inv = 0;
                for (int r : kv.second)
                    if (r >= 0 && r < (int)opt.rec_invert.size()) inv ^= (opt.rec_invert[(size_t)r] & 1);
                born_dec_inv.push_back(inv);
                born_dec_recs.push_back(kv.second);
            }
        }
        has_decisions_ = !opt.decisions.empty();
        has_born_dec_ = !born_dec.empty();
        dec_scratch_.assign((size_t)(num_dec_ > 0 ? num_dec_ : 0), 0);
        // Real emptiness gate (post-decision): fatal ONLY when NOTHING produces a channel/bit.
        if (sch.empty() && gauge_dets.empty() && born_idx < 0 && born_dec.empty()) {
            fprintf(stderr, "twirl_records: no channels of any kind (no detectors, observables, "
                            "or decisions)\n");
            setup_error = 2; return;
        }
        if (!xtim_quiet())
            fprintf(stderr, "[twirl_records] circuit channels: %zu deterministic (+%zu observable), "
                            "%zu gauge, %zu anti, %zu refused observables, %zu born-decisions "
                            "(+%zu det-decisions)\n",
                    chan_det.size(), chan_obs.size(), gauge_dets.size(), anti_dets.size(),
                    refused_obs.size(), born_dec.size(), dec_chan_idx_.size());
    }
    nchan = (int)sch.size();
    CW = (nchan + 63) / 64;                     // channel-space word count
    // ── Readout-flip layer build (circuit-channel mode). Assign a slot to each noisy record;
    // per channel record the noisy slots it contains; fold deterministic inverts into refs. ──
    if (circ_chan && !opt.rec_flip.empty()) {
        std::vector<int> rec_slot((size_t)opt.rec_flip.size(), -1);
        auto slot_of = [&](int rec) -> int {
            if (rec < 0 || rec >= (int)opt.rec_flip.size()) return -1;
            if (opt.rec_flip[(size_t)rec] <= 0.0) return -1;
            if (rec_slot[(size_t)rec] < 0) {
                rec_slot[(size_t)rec] = nrf++;
                rf_prob.push_back(opt.rec_flip[(size_t)rec] * factor);
            }
            return rec_slot[(size_t)rec];
        };
        auto inv_of = [&](int rec) -> int {
            return (rec >= 0 && rec < (int)opt.rec_invert.size()) ? (opt.rec_invert[(size_t)rec] & 1) : 0;
        };
        // detector channels [0, chan_det.size()), then observable channels
        std::vector<std::vector<int>> chan_records((size_t)nchan);
        for (size_t c = 0; c < chan_det.size(); ++c)
            chan_records[c] = detectors[(size_t)chan_det[c]];
        for (size_t k = 0; k < chan_obs.size(); ++k)
            for (const auto& kv : observables)
                if (kv.first == chan_obs[k]) { chan_records[chan_det.size() + k] = kv.second; break; }
        // Task 2b.1 FIX: fill chan_records for IN_GROUP decision channels so that:
        //   (a) slot_of() discovers their noisy records for rf-mask accounting, and
        //   (b) rec_invert is properly folded into chan_ref[c] (and synced to dec_chan_ref_[j]).
        for (size_t j = 0; j < dec_chan_slot_.size(); ++j)
            chan_records[(size_t)dec_chan_slot_[j]] = dec_chan_recs_[j];
        // first pass: discover all noisy slots (channels + born observable)
        for (int c = 0; c < nchan; ++c) for (int r : chan_records[(size_t)c]) slot_of(r);
        std::vector<int> born_records;
        if (born_idx >= 0)
            for (const auto& kv : observables)
                if (kv.first == born_idx) { born_records = kv.second; break; }
        for (int r : born_records) slot_of(r);
        // Task 2b.1: discover the Born decisions' noisy record slots (M(p) readout flips).
        for (const auto& recs : born_dec_recs) for (int r : recs) slot_of(r);
        RFW = (nrf + 63) / 64;
        rf_coins.assign((size_t)RFW, 0);
        chan_rf_mask.assign((size_t)nchan, std::vector<uint64_t>((size_t)RFW, 0));
        for (int c = 0; c < nchan; ++c) {
            int inv = 0;
            for (int r : chan_records[(size_t)c]) {
                const int sl = slot_of(r);
                if (sl >= 0) chan_rf_mask[(size_t)c][sl >> 6] |= 1ull << (sl & 63);
                inv ^= inv_of(r);
            }
            chan_ref[(size_t)c] = (uint8_t)(chan_ref[(size_t)c] ^ (inv & 1));  // deterministic invert
        }
        // Task 2b.1 FIX: sync dec_chan_ref_[j] from chan_ref[c] now that rec_invert has been
        // folded in. emit_decisions() reads dec_chan_ref_[j] for the decision reference bit —
        // without this sync, rec_invert (`!` in Stim) on a DECISION(k) record would be ignored.
        for (size_t j = 0; j < dec_chan_slot_.size(); ++j)
            dec_chan_ref_[j] = chan_ref[(size_t)dec_chan_slot_[j]];
        if (born_idx >= 0) {
            born_rf_mask.assign((size_t)RFW, 0);
            int inv = 0;
            for (int r : born_records) {
                const int sl = slot_of(r);
                if (sl >= 0) born_rf_mask[sl >> 6] |= 1ull << (sl & 63);
                inv ^= inv_of(r);
            }
            if (inv & 1) m_clean = -m_clean;    // deterministic observable invert folds into weight
        }
        // Task 2b.1: per-born-decision readout-flip mask (the deterministic invert is already in
        // born_dec_inv from opt.rec_invert). emit_decisions XORs the parity of the shot's rf coins
        // over this mask into the decision bit — inert when no record carries an M(p) probability.
        if (!born_dec.empty()) {
            born_dec_rf_mask.assign(born_dec.size(), std::vector<uint64_t>((size_t)RFW, 0));
            for (size_t j = 0; j < born_dec.size(); ++j)
                for (int r : born_dec_recs[j]) {
                    const int sl = slot_of(r);
                    if (sl >= 0) born_dec_rf_mask[j][(size_t)(sl >> 6)] |= 1ull << (sl & 63);
                }
        }
        // FIX B: precompute fixed-point thresholds rf_thresh[sl] = (uint64_t)(p_sl * 2^64).
        // p_sl is in (0,1] (slot_of only creates slots for p>0).  The cast saturates correctly:
        // p_sl=1.0 gives (uint64_t)(1.0 * 2^64) = 0 due to wrap, but p_sl=1 is only possible
        // in test circuits; real circuits have p<<1.  For safety we clamp: if p_sl >= 1.0 set
        // thresh = UINT64_MAX (always fire).
        rf_thresh.resize((size_t)nrf);
        for (int sl = 0; sl < nrf; ++sl) {
            const double p = rf_prob[(size_t)sl];
            rf_thresh[(size_t)sl] = (p >= 1.0) ? UINT64_MAX : (uint64_t)(p * 18446744073709551616.0); // 2^64
        }
        // FIX A: build transposed column layout rf_col[sl] = CW-word bitvector of channels
        // that have slot sl in their rf mask.  Also build rf_obs_bit[sl] from born_rf_mask.
        // rf_col_data_ is a flat nrf*CW allocation; rf_col[sl] points into it.
        rf_col_data_.assign((size_t)nrf * (size_t)CW, 0);
        rf_col.resize((size_t)nrf);
        for (int sl = 0; sl < nrf; ++sl)
            rf_col[(size_t)sl] = rf_col_data_.data() + (size_t)sl * (size_t)CW;
        // Transpose chan_rf_mask -> rf_col: for each channel c and each set bit sl in its mask,
        // set bit c in rf_col[sl].
        for (int c = 0; c < nchan; ++c) {
            for (int w = 0; w < RFW; ++w) {
                uint64_t word = chan_rf_mask[(size_t)c][(size_t)w];
                while (word) {
                    const int bit = __builtin_ctzll(word);
                    word &= word - 1;
                    const int sl = w * 64 + bit;
                    rf_col[(size_t)sl][c >> 6] |= 1ull << (c & 63);
                }
            }
        }
        // Build rf_obs_bit[sl] from born_rf_mask (1 if slot sl is set in the born obs mask).
        rf_obs_bit.assign((size_t)nrf, 0);
        if (!born_rf_mask.empty()) {
            for (int sl = 0; sl < nrf; ++sl)
                rf_obs_bit[(size_t)sl] = (uint8_t)((born_rf_mask[sl >> 6] >> (sl & 63)) & 1);
        }
        // FIX B: rf_gen removed from hot path.  done_rf_ tracks shot index for the rf ShotRng.
        done_rf_ = 0;
    }
    // ── NOISELESS-REFERENCE SEMANTICS (P0, adjudicated 2026-07-24, option (b)) ──────
    // Deterministic DETECTOR/OBSERVABLE channels are REFERENCE-RELATIVE — exactly Stim's
    // and the production sampler's semantics (sampler.cpp refdet/refobs: "the emitted bit
    // is parity(shot) XOR parity(noiseless reference)").  The engine's internal channel bit
    // is already the detection EVENT relative to THIS COMPILE'S BARE STATE (mask·σ ⊕ readout
    // coins), and the reporting layer (xtim_py sinks, framed_bench) XORs chan_ref into it.
    //
    // Stim DEFINES the reference as "the run with every noise channel frozen off".  That
    // definition splits into exactly two cases here, and BOTH are exact — no text
    // inspection, no propagation, no heuristic:
    //
    //   • NO INPUT PORT — a self-contained circuit (stage 0, any port-free consumer stage,
    //     and every existing pin/golden/bench).  The engine already HOLDS the noise-free
    //     run: the bare state IS it.  So the reference parity equals the bare parity, the
    //     internal bit is ALREADY the event, and  chan_ref = 0  EXACTLY.
    //
    //     This replaces (2026-07-24) the earlier attempt to reach the same number by
    //     conjugating the text's own deterministic X/Y/Z gates to circuit end with
    //     propagate_atom and subtracting their induced flip.  That was an APPROXIMATION with
    //     four documented failure modes (a Pauli leaving the Pauli class in propagation; a
    //     flip composed from non-Pauli text such as S;S = Z; a flip absorbed by
    //     eliminate_hadamards into a measurement invert; a Pauli in a PRODUCER stage before
    //     OUTPUT_QUBITS) — in each of which it silently fell back to the raw sign.  Zeroing
    //     is exact wherever the subtraction was merely trying to be.
    //
    //   • CARRIED INPUT PORT — a consumer stage compiled from_state.  The ideal (noise-free)
    //     CARRIED state is a PROTOCOL-level fact that no function of this compile's text can
    //     recover.  The SAME composed text
    //         X q (carried-state prep) ; INPUT_QUBITS ; X q (body) ; M q ; DETECTOR
    //     must report 0 when the carried |1⟩ is a Born branch (teleport-style feedback) and
    //     1 when it is an X_ERROR deviation — the two protocols are byte-identical here and
    //     demand OPPOSITE references, so no local rule can serve both.  The engine therefore
    //     does not guess: it reports the ABSOLUTE parity (chan_ref = the raw classify reduce
    //     sign, with rec_invert folded in above), and the layer that KNOWS the protocol —
    //     the noiseless-reference pass in xtim/feedback.py, which replays the stage chain at
    //     p_factor = 0 and XORs the resulting per-(stage, decision-pattern) reference — turns
    //     absolute parities into EVENTS before the decoder ever sees them.
    //
    //     CONTRACT: a from_state compile driven directly through the Python sampler API
    //     reports ABSOLUTE parity (tests/test_observable_reference.py
    //     ::test_carried_deviation_still_reported pins exactly this — a carried |1⟩ read in Z
    //     reports 1, and an X input frame corrects it to 0).
    //
    // DECISION channels are untouched by design: they are RAW recorded parity by contract
    // (they feed feedback/decoding) and occupy channel slots
    // [chan_det.size() + chan_obs.size(), nchan).
    if (circ_chan && opt.input_pauli_qubits.empty()) {
        const size_t ndo = chan_det.size() + chan_obs.size();
        for (size_t c = 0; c < ndo; ++c) chan_ref[c] = 0;
    }
    if (born_obs) {                             // acceptance test data (post-selected stats)
        det_refw.assign((size_t)CW, 0);
        det_maskw.assign((size_t)CW, 0);
        for (size_t c = 0; c < chan_det.size(); ++c) {
            det_maskw[c >> 6] |= 1ull << (c & 63);
            if (chan_ref[c]) det_refw[c >> 6] |= 1ull << (c & 63);
        }
    }
    // Per-circuit prefix qubit rows (CW words per qubit): bit c of pxrow[q] =
    // parity(zcol(q) & mask_c) (X-support contribution to plane1 lands on zcol), pzrow from xcol.
    pxrow.assign((size_t)n * CW, 0); pzrow.assign((size_t)n * CW, 0);
    {
        std::vector<uint64_t> col(GW);
        for (int q = 0; q < n; ++q) {
            const uint64_t* zc = G.col(q, false);
            const uint64_t* xc = G.col(q, true);
            for (int c = 0; c < nchan; ++c) {
                col.assign(zc, zc + GW);
                if (par(col, sch[c].mask)) pxrow[(size_t)q * CW + (c >> 6)] |= (1ull << (c & 63));
                col.assign(xc, xc + GW);
                if (par(col, sch[c].mask)) pzrow[(size_t)q * CW + (c >> 6)] |= (1ull << (c & 63));
            }
        }
    }
    chtok = 0x2545F4914F6CDD1Dull;               // channel-set token
    for (const SChan& c : sch) for (uint64_t w : c.mask) { chtok ^= w; chtok *= 0x100000001B3ull; }

    // DiagErrorSampler channels from the deferred noise stream (probs × factor); sides = the
    // per-alternative FiredPauli lists, flat-alt-id order == construction order.
    std::vector<ErrorChannel> chans;
    auto add_alt = [&](ErrorChannel& ch, double pp, std::vector<FiredPauli> fr) {
        ErrorChannel::Alt alt; alt.p = pp; ch.alts.push_back(std::move(alt));
        sides.push_back(std::move(fr)); };
    static const int kP2[15][2] = {{1,0},{2,0},{3,0},{0,1},{0,2},{0,3},{1,1},{1,2},{1,3},
                                   {2,1},{2,2},{2,3},{3,1},{3,2},{3,3}};
    for (int si = 0; si < (int)deferred.stream.size(); ++si) {
        const Instr& ins = deferred.stream[si];
        if (ins.kind != Instr::Kind::Noise) continue;
        switch (ins.channel) {
            case NoiseChannel::X_ERROR: case NoiseChannel::Y_ERROR: case NoiseChannel::Z_ERROR: {
                const PauliBasis b = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                                   : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z;
                for (int q : ins.qubits) { ErrorChannel ch;
                    add_alt(ch, ins.probs[0] * factor, {{si, q, b}});
                    chans.push_back(std::move(ch)); }
                break; }
            case NoiseChannel::DEPOLARIZE1: case NoiseChannel::PAULI_CHANNEL_1: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
                static const PauliBasis bs[3] = {PauliBasis::X, PauliBasis::Y, PauliBasis::Z};
                for (int q : ins.qubits) { ErrorChannel ch;
                    for (int i = 0; i < 3; ++i) {
                        const double pp = (dep ? ins.probs[0] / 3.0 : ins.probs[i]) * factor;
                        if (pp > 0.0) add_alt(ch, pp, {{si, q, bs[i]}}); }
                    if (!ch.alts.empty()) chans.push_back(std::move(ch)); }
                break; }
            case NoiseChannel::DEPOLARIZE2: case NoiseChannel::PAULI_CHANNEL_2: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE2;
                for (size_t pi = 0; pi + 1 < ins.qubits.size(); pi += 2) {
                    const int qa = ins.qubits[pi], qb = ins.qubits[pi + 1];
                    ErrorChannel ch;
                    for (int t = 0; t < 15; ++t) {
                        const double pp = (dep ? ins.probs[0] / 15.0 : ins.probs[t]) * factor;
                        if (pp <= 0.0) continue;
                        std::vector<FiredPauli> fr;
                        if (kP2[t][0]) fr.push_back({si, qa, (PauliBasis)(kP2[t][0] - 1)});
                        if (kP2[t][1]) fr.push_back({si, qb, (PauliBasis)(kP2[t][1] - 1)});
                        add_alt(ch, pp, std::move(fr)); }
                    if (!ch.alts.empty()) chans.push_back(std::move(ch)); }
                break; }
            default:
                fprintf(stderr, "twirl_records: unsupported noise channel\n"); setup_error = 3; return;
        }
    }
    smp.reset(new DiagErrorSampler(n, chans, seed));

    // ── Sparse per-alt atom extraction (Tier C substrate) ───────────────────────────────────
    // Per flat alternative: compose its (1–2 Pauli) fired list ONCE at setup via the exact
    // compose_fired, then store the sparse content. Per shot the residual is accumulated by the
    // SAME recurrence then() uses — r.a[j] = a1[j] + a2[j]·(1−2·v1[j]) + 2·(B2·v1)[j], v ^= v2,
    // B ^= B2 — but sparsely over the atom's support with membership-flag dirty lists (the
    // through-zero dedup lesson), so a shot costs O(Σ atom support), not fired × O(n²) B copies.
    // γ is never tracked (discarded at the normal form). Self-checked EXACTLY against the
    // compose_fired chain on the first --selfcheck diag shots.
    alt_atoms.assign(sides.size(), AltAtom());
    // ── PPR alts (has_general tables): an alt touching a PPR-retry atom cannot go through
    // the diagonal compose (c_prop is invalid there). Precompute a per-alt PprNormalForm for
    // EVERY alt instead (diagonal alts convert via residual_normal_form → ppr_rots_from_diag),
    // so a mixed shot composes uniformly in the PPR algebra. ENGINE ORDER: compose_fired
    // applies earlier atoms FIRST (rightmost operator factor), and compose_ppr builds
    // leftmost-first — so atom lists are fed REVERSED (within a Y: z_atom before x_atom).
    alt_is_ppr.assign(sides.size(), 0);
    if (ppr_tables) alt_pnf.assign(sides.size(), PprNormalForm(n));
    {
        std::set<std::pair<int, int>> ps;
        std::vector<DiagPauliClifford> altD;
        altD.reserve(sides.size());
        DiagPauliClifford Did = DiagPauliClifford::identity(n);
        for (size_t ai = 0; ai < sides.size(); ++ai) {
            const std::vector<FiredPauli>& fr = sides[ai];
            bool isppr = false;
            if (ppr_tables) {
                for (const FiredPauli& f : fr) {
                    const auto [loc, qi] = table.find_slot(f.location_index, f.qubit, "twirl_records");
                    if ((f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) && loc->x_atom[qi].ppr) isppr = true;
                    if ((f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) && loc->z_atom[qi].ppr) isppr = true;
                }
            }
            alt_is_ppr[ai] = isppr ? 1 : 0;
            if (isppr) {
                // exact atom sequence, REVERSED into compose_ppr's leftmost-first order
                std::vector<std::pair<Pauli, std::vector<std::pair<Pauli, int>>>> store;
                for (auto it = fr.rbegin(); it != fr.rend(); ++it) {
                    const auto [loc, qi] = table.find_slot(it->location_index, it->qubit, "twirl_records");
                    auto push_atom = [&](const PropResult& pr2) {
                        if (pr2.ppr) {
                            store.emplace_back(pr2.ppr->pauli, pr2.ppr->rots);
                        } else {
                            DiagNormalForm anf = residual_normal_form(pr2.c_prop);
                            std::vector<std::pair<Pauli, int>> rots;
                            ppr_rots_from_diag(anf, n, rots);
                            store.emplace_back(anf.prefix, std::move(rots));
                        }
                    };
                    if (it->pauli == PauliBasis::Z || it->pauli == PauliBasis::Y) push_atom(loc->z_atom[qi]);
                    if (it->pauli == PauliBasis::X || it->pauli == PauliBasis::Y) push_atom(loc->x_atom[qi]);
                }
                std::vector<PprAtom> views;
                views.reserve(store.size());
                for (const auto& s2 : store) views.push_back(PprAtom{&s2.first, &s2.second});
                alt_pnf[ai] = compose_ppr(n, views);
                altD.push_back(Did);               // placeholder (never fed to the diagonal path)
                continue;
            }
            DiagPauliClifford D = compose_fired(table, fr);
            D.finalize();
            for (const auto& pr : D.cache->b_pairs) ps.insert(pr);
            if (ppr_tables) {                       // diagonal alt as a single PPR atom
                DiagNormalForm anf = residual_normal_form(D);
                std::vector<std::pair<Pauli, int>> rots;
                ppr_rots_from_diag(anf, n, rots);
                std::vector<std::pair<Pauli, std::vector<std::pair<Pauli, int>>>> store;
                store.emplace_back(anf.prefix, std::move(rots));
                std::vector<PprAtom> views{PprAtom{&store[0].first, &store[0].second}};
                alt_pnf[ai] = compose_ppr(n, views);
            }
            altD.push_back(std::move(D));
        }
        upairs.assign(ps.begin(), ps.end());
        for (size_t i = 0; i < upairs.size(); ++i) pidx[upairs[i]] = (int)i;
        for (size_t ai = 0; ai < altD.size(); ++ai) {
            if (alt_is_ppr[ai]) continue;           // AltAtom stays empty (never composed diagonally)
            const DiagPauliClifford& D = altD[ai];
            AltAtom& A = alt_atoms[ai];
            for (int q : D.cache->active_a) A.av.push_back({q, (uint8_t)(D.a[q] & 3)});
            for (const auto& pr : D.cache->b_pairs) { A.czp.push_back(pr); A.czidx.push_back(pidx[pr]); }
            for (int q = 0; q < n; ++q) if (D.v[q]) A.vq.push_back(q);
        }
    }
    PW = (int)(upairs.size() + 63) / 64;
    NW = (n + 63) / 64;
    a_acc.assign((size_t)n, 0); v_acc.assign((size_t)n, 0);
    in_a.assign((size_t)n, 0); in_v.assign((size_t)n, 0);
    czw.assign((size_t)PW, 0);
    in_w.assign((size_t)PW, 0);
    amask_acc.assign((size_t)NW, 0);   // maintained packed a&1 (the memo key part)
    // Product-wire word mask (canon-touch test in O(NW) instead of an O(n) byte scan).
    prodmask.assign((size_t)NW, 0);
    {
        const CertifiedGroupPlanes::ZAxisRREF& Z = G.z_axis_rref();
        for (int q = 0; q < n; ++q)
            if (Z.prod_sign[q] != 0) prodmask[q >> 6] |= (1ULL << (q & 63));
    }
    ZR = &G.z_axis_rref();

    // ── ABELIAN FAST PATH substrate (2026-07-15, "why isn't warm just sampling?"). ─────────
    // The memo KEY's mod-2 content is an XOR-ADDITIVE function of the fired set (every
    // composition cross-term is even), so per shot: key = XOR of per-alt precomputed masks —
    // NO content materialisation on a warm hit. The composed prefix decomposes as
    //   x-part: v_tot = ⊕_j v_j                                             (abelian)
    //   z-part: z_tot = ⊕_j z_j ⊕ ⊕_{k<j}[ (s_j∧s_k) ⊕ (s_j∧v_k) ⊕ (B_j·v_k) ]
    // (the pairwise S-carry is EXACT for all coincidence counts: C(c,2) mod 2 == bit1(c mod 4)),
    // and channel bits are pxrow/pzrow XORs — LINEAR in these masks, so per-alt channel rows
    // XOR too; only the ordered-PAIR term needs per-shot work (~fired²/2 small word ops).
    // Strip interaction: the canonical key is (smask & ~prodmask, kept pairs) and the −Z-partner
    // zflips ride per-occurrence XOR (a cancelling pair cancels its flips) → per-alt zrow
    // includes the alt's own stripped-pair flips; product-wire z bits are inert (pzrow ≡ 0
    // there). ORACLE: selfcheck window compares abelian key == materialised+canon key and
    // abelian bits == σ-path bits EXACTLY, or aborts.
    alt_masks.assign(alt_atoms.size(), AltMasks());
    for (size_t ai = 0; ai < alt_atoms.size(); ++ai) {
        const AltAtom& A = alt_atoms[ai];
        AltMasks& M = alt_masks[ai];
        M.smask.assign((size_t)NW, 0); M.vmask.assign((size_t)NW, 0);
        M.kmask.assign((size_t)NW, 0); M.czk.assign((size_t)PW, 0);
        M.zxrow.assign((size_t)CW, 0); M.zzrow.assign((size_t)CW, 0);
        for (const auto& qa : A.av) {
            if (qa.second & 1) M.smask[qa.first >> 6] |= (1ULL << (qa.first & 63));
            if (qa.second & 2)
                for (int w2 = 0; w2 < CW; ++w2) M.zzrow[w2] ^= pzrow[(size_t)qa.first * CW + w2];
        }
        for (int w2 = 0; w2 < NW; ++w2) M.kmask[w2] = M.smask[w2] & ~prodmask[w2];
        for (int q : A.vq) {
            M.vmask[q >> 6] |= (1ULL << (q & 63));
            for (int w2 = 0; w2 < CW; ++w2) M.zxrow[w2] ^= pxrow[(size_t)q * CW + w2];
        }
        for (size_t i = 0; i < A.czp.size(); ++i) {
            const int a2 = A.czp[i].first, b2 = A.czp[i].second;
            const int s1 = ZR->prod_sign[a2], s2 = ZR->prod_sign[b2];
            if (s1 == 0 && s2 == 0) {
                const int pi2 = A.czidx[i];
                M.czk[pi2 >> 6] ^= (1ULL << (pi2 & 63));
            } else if (s1 != 0 && s2 == 0) {
                if (s1 < 0) for (int w2 = 0; w2 < CW; ++w2) M.zzrow[w2] ^= pzrow[(size_t)b2 * CW + w2];
            } else if (s1 == 0 && s2 != 0) {
                if (s2 < 0) for (int w2 = 0; w2 < CW; ++w2) M.zzrow[w2] ^= pzrow[(size_t)a2 * CW + w2];
            }                                              // both stripped: global phase
        }
    }
    ab_kmask.assign((size_t)NW, 0); ab_czk.assign((size_t)PW, 0);
    ab_T.assign((size_t)NW, 0);                       // pair-term scratch
    // ── Local bitmask index for the abelian fast path: key = (kmask NW ‖ czk PW) words over
    //    upairs indexing — no pair-list expansion, no pair-vector hashing on the hit path.
    //    Disk plans whose key_cz contains a pair outside upairs are unreachable under this alt
    //    set and are simply not indexed (a lookup miss falls through to the exact path).
    KW = NW + PW;
    ab_kw.assign((size_t)KW, 0);                      // per-shot key scratch
    ab_S.assign((size_t)NW, 0); ab_V.assign((size_t)NW, 0);     // running pair-term masks
    fgrow();                                                      // never probe an empty table
    // Shared fast-route index adoption: if a warm index is available with matching KW, swap its
    // contents in so ffind hits immediately on this rebuild's hot path. The disk-cache lindex_insert
    // calls below are idempotent on already-present entries. The write-back to sfi_ happens in
    // ~Impl(), so the next rebuild inherits everything this run() adds. Skipped (fresh fgrow kept)
    // when KW changed (shouldn't happen given the identity_token guard) or the index is empty.
    if (sc.fidx && sc.fidx->KW == KW && !sc.fidx->lkeys.empty()) {
        sfi_ = sc.fidx;
        lkeys  = std::move(sfi_->lkeys);
        lplans = std::move(sfi_->lplans);
        lidx   = std::move(sfi_->lidx);
        ftab   = std::move(sfi_->ftab);
        fslots = sfi_->fslots;
        fcount = sfi_->fcount;
    } else if (sc.fidx && sc.fidx->KW == 0) {
        // First-ever adoption: register this Impl with the shared owner but keep fresh index.
        sc.fidx->KW = KW;
        sfi_ = sc.fidx;
    } else if (sc.fidx && sc.fidx->KW == KW) {
        // Index exists but is empty (e.g. after a token-clear): just register.
        sfi_ = sc.fidx;
    }
    // ── Flat per-alt blob (2026-07-15): the AoS AltMasks walk was DRAM-latency-bound (~6
    //    scattered heap vectors per fired alt). One contiguous block per alt + prefetch makes
    //    the warm path stream. Layout (uint64 words):
    //      [0]            header: nc | np<<32  (czk entry count, czp pair count)
    //      [1]            htok: GF(2)-linear hash token of this alt's key contribution
    //      [2 .. +NW]     kmask (dense)        [.. +NW] smask   [.. +NW] vmask
    //      [.. +CW]       zrow = zxrow ^ zzrow (emission row; only ever used XORed together)
    //      [.. +2*nc]     czk entries (word-idx, value)
    //      [.. +np]       czp pairs (x | y<<32)
    alt_off.assign(alt_masks.size(), 0);
    {
        size_t tot = 0;
        for (size_t ai = 0; ai < alt_masks.size(); ++ai) {
            size_t nc = 0;
            for (int w = 0; w < PW; ++w) if (alt_masks[ai].czk[(size_t)w]) ++nc;
            tot += 2 + 3 * (size_t)NW + (size_t)CW + 2 * nc + alt_atoms[ai].czp.size();
        }
        alt_blob.reserve(tot);
        std::vector<uint64_t> contrib((size_t)KW);
        for (size_t ai = 0; ai < alt_masks.size(); ++ai) {
            const AltMasks& M = alt_masks[ai];
            const AltAtom& A = alt_atoms[ai];
            alt_off[ai] = (uint32_t)alt_blob.size();
            size_t nc = 0;
            for (int w = 0; w < PW; ++w) if (M.czk[(size_t)w]) ++nc;
            alt_blob.push_back((uint64_t)nc | ((uint64_t)A.czp.size() << 32));
            for (int w = 0; w < NW; ++w) contrib[(size_t)w] = M.kmask[(size_t)w];
            for (int w = 0; w < PW; ++w) contrib[(size_t)(NW + w)] = M.czk[(size_t)w];
            alt_blob.push_back(lhash(contrib.data()) ^ 0xcbf29ce484222325ull);  // htok (C removed)
            for (int w = 0; w < NW; ++w) alt_blob.push_back(M.kmask[(size_t)w]);
            for (int w = 0; w < NW; ++w) alt_blob.push_back(M.smask[(size_t)w]);
            for (int w = 0; w < NW; ++w) alt_blob.push_back(M.vmask[(size_t)w]);
            for (int w = 0; w < CW; ++w) alt_blob.push_back(M.zxrow[(size_t)w] ^ M.zzrow[(size_t)w]);
            for (int w = 0; w < PW; ++w)
                if (M.czk[(size_t)w]) { alt_blob.push_back((uint64_t)w); alt_blob.push_back(M.czk[(size_t)w]); }
            for (size_t i = 0; i < A.czp.size(); ++i)
                alt_blob.push_back((uint64_t)(uint32_t)A.czp[i].first |
                                   ((uint64_t)(uint32_t)A.czp[i].second << 32));
        }
        // One-time blob-vs-AoS verification (exact, every alt) — the blob is a pure re-encoding.
        for (size_t ai = 0; ai < alt_masks.size(); ++ai) {
            const AltMasks& M = alt_masks[ai];
            const AltAtom& A = alt_atoms[ai];
            const uint64_t* B = &alt_blob[alt_off[ai]];
            const uint32_t nc = (uint32_t)B[0], np = (uint32_t)(B[0] >> 32);
            for (int w = 0; w < NW; ++w) contrib[(size_t)w] = M.kmask[(size_t)w];
            for (int w = 0; w < PW; ++w) contrib[(size_t)(NW + w)] = M.czk[(size_t)w];
            if (B[1] != (lhash(contrib.data()) ^ 0xcbf29ce484222325ull)) {
                fprintf(stderr, "twirl_records BLOB VERIFY FAIL (htok) alt=%zu\n", ai);
                setup_error = 5; return;
            }
            const uint64_t* km = B + 2;
            const uint64_t* sm = km + NW;
            const uint64_t* vm = sm + NW;
            const uint64_t* zr = vm + NW;
            const uint64_t* ce = zr + CW;
            const uint64_t* pp = ce + 2 * nc;
            bool ok = np == A.czp.size();
            for (int w = 0; ok && w < NW; ++w)
                ok = km[w] == M.kmask[(size_t)w] && sm[w] == M.smask[(size_t)w] &&
                     vm[w] == M.vmask[(size_t)w];
            for (int w = 0; ok && w < CW; ++w)
                ok = zr[w] == (M.zxrow[(size_t)w] ^ M.zzrow[(size_t)w]);
            std::vector<uint64_t> cz2((size_t)PW, 0);
            for (uint32_t i = 0; i < nc; ++i) cz2[(size_t)ce[2 * i]] = ce[2 * i + 1];
            if (ok) ok = cz2 == M.czk;
            for (uint32_t i = 0; ok && i < np; ++i)
                ok = (uint32_t)pp[i] == (uint32_t)A.czp[i].first &&
                     (uint32_t)(pp[i] >> 32) == (uint32_t)A.czp[i].second;
            if (!ok) {
                fprintf(stderr, "twirl_records BLOB VERIFY FAIL alt=%zu\n", ai);
                setup_error = 5; return;
            }
        }
    }

    seed_ = seed;
    // Per-shot derived RNG: shot_rng is re-seeded O(1) at the top of each shot in run(); this
    // lambda captures `this` so coll_rng always delegates to the CURRENT shot_rng state.
    coll_rng = [this]() { return shot_rng.next_double(); };
    // V3-T3 R1: dedicated exact-fallback rng — its draws never touch coll_gen/obs_gen, so
    // every non-fallback shot's byte stream is identical to the pre-R1 engine.
    fb_gen.seed(seed ^ 0xE4AC7F0DD15C0DE5ull);
    // Task 2b.1: dedicated Born-decision rng — its draws never touch coll_gen/obs_gen/fb_gen, so
    // decision-free circuits (has_decisions_ == false, no draw) stay byte-identical.
    dec_gen.seed(seed ^ 0x2B1DEC1510DEC0DEull);
    // V3-T3 R2: automatic plan-cache path — <dir>/<fnv64(deferred-signature)>-<group-token>.twpl.
    // Noise-blind key (signature excludes noise values) + the group identity token; the TWPL
    // framing re-verifies the token on load, so stale/corrupt/skew files rebuild silently.
    if (disk_path.empty() && !opt.disk_auto_dir.empty()) {
        char key[64];
        snprintf(key, sizeof key, "/%016llx-%016llx.twpl",
                 (unsigned long long)deferred_signature_fnv(deferred, reads),
                 (unsigned long long)G.identity_token());
        disk_path = opt.disk_auto_dir + key;
    }
    if (!disk_path.empty()) {
        const bool ld = cache->load_file(disk_path, G.identity_token());
        auto tpb = clk::now();
        cache->for_each([&](CachedPlan& p) {
            if (p.ch_token != chtok) build_ch_rows(p);
            if (plan_local_key(p, lkey_scratch)) lindex_insert(lkey_scratch, &p);
        });
        if (!xtim_quiet())
            fprintf(stderr, "[twirl_records] disk '%s': %s (%zu plans, rows+index pre-built in %.1f ms)\n",
                    disk_path.c_str(), ld ? "LOADED" : "absent/rebuild", cache->size(),
                    1e3 * std::chrono::duration<double>(clk::now() - tpb).count());
    }
    chancnt.assign((size_t)nchan, 0);             // per-channel 1-bit counts
    nf.prefix = Pauli(n);
    // Task 4: input Pauli frame (gate on has_input_frame_; frame-free circuits byte-identical).
    // Initialized AFTER pxrow/pzrow so the anchor-time row build below can safely index them.
    has_input_frame_ = !opt.input_pauli_qubits.empty();
    if (has_input_frame_) {
        frame_port_qubits_ = opt.input_pauli_qubits;
        // ── Anchor-time conjugation of the port frame components (see the member block
        // comment for the Heisenberg derivation). For each port qubit q_i, propagate X_{q_i}
        // and Z_{q_i} through the FULL deferred unitary stream (from_index = -1) with the one
        // shared propagate_atom kernel. A Pauli image has an even S-layer (a[j] ∈ {0,2}) and
        // an empty CZ layer; anything else (PPR retry / general tableau / odd a / CZ) means a
        // non-Clifford gate sits in this component's cone — POISON it (loud refusal in run()
        // iff a shot supplies it; the compile itself stays valid for frame-free use).
        // Channel row of a Pauli P' (bilinearity of sp): ⊕_{j∈x(P')} pxrow[j] ⊕
        // ⊕_{j∈z(P')} pzrow[j] — a Y qubit (v=1, a=2) contributes both terms.
        const int pnq = (int)frame_port_qubits_.size();
        frame_xrow_.assign((size_t)pnq * (size_t)CW, 0);
        frame_zrow_.assign((size_t)pnq * (size_t)CW, 0);
        frame_poison_.assign((size_t)(2 * pnq), 0);
        frame_obs_bits_.assign((size_t)(2 * pnq), 0);
        // Relabel-target support mask. The frame relabels ONLY the stage's classical outputs:
        // the σ-channel bits (each channel operator is a ± product of the certified generators
        // its `mask` selects — support ⊆ the union of those members' supports) and the Born
        // operators (born_W + Born DECISIONs). The raw generator σ-record handed to the sink
        // is DELIBERATELY frame-free (the lazy-frame design: the orchestrator carries the
        // frame; the engine never applies it to states or σ), so generators outside every
        // channel mask are NOT relabel targets. A non-Pauli image support-DISJOINT from this
        // mask is an exact no-op (SOFT poison).
        const int NWf = (n + 63) / 64;
        std::vector<uint64_t> tgt((size_t)NWf, 0);
        auto fold_supp = [&](const Pauli& W) {
            for (int w = 0; w < NWf && w < (int)W.x.size(); ++w) tgt[(size_t)w] |= W.x[w];
            for (int w = 0; w < NWf && w < (int)W.z.size(); ++w) tgt[(size_t)w] |= W.z[w];
        };
        {
            std::vector<uint64_t> gm((size_t)GW, 0);       // union of channel combo masks
            for (const SChan& c : sch)
                for (int w = 0; w < GW && w < (int)c.mask.size(); ++w) gm[(size_t)w] |= c.mask[w];
            for (int i2 = 0; i2 < ng; ++i2)
                if ((gm[(size_t)(i2 >> 6)] >> (i2 & 63)) & 1) fold_supp(gens[(size_t)i2]);
        }
        if (born_obs) fold_supp(born_W);
        for (const auto& bd : born_dec) fold_supp(bd.second);
        std::vector<Pauli> conj((size_t)(2 * pnq), Pauli(n));
        for (int i = 0; i < pnq; ++i) {
            const int q = frame_port_qubits_[i];
            for (int comp = 0; comp < 2; ++comp) {                // 0: X_{q_i}, 1: Z_{q_i}
                const int slot = comp ? pnq + i : i;
                const PropResult pr = propagate_atom(
                    deferred, -1, comp ? DiagPauliClifford::Z(n, q)
                                       : DiagPauliClifford::X(n, q));
                bool pauli_ok = !pr.rejected && !pr.general;
                for (int j = 0; j < n && pauli_ok; ++j)
                    if (pr.c_prop.a[j] & 1) pauli_ok = false;     // odd S-layer ⇒ non-Pauli
                for (int j = 0; j < n && pauli_ok; ++j)
                    for (int l = j + 1; l < n && pauli_ok; ++l)
                        if (pr.c_prop.B.get(j, l)) pauli_ok = false;  // CZ layer ⇒ non-Pauli
                if (!pauli_ok) {
                    // SOFT vs HARD poison: SOFT iff the image is a KNOWN unitary (diagonal
                    // class, not rejected/general) whose full support {j : v_j ∨ a_j ∨ CZ_j}
                    // misses every relabel target — then the flip is exactly zero (see the
                    // member block derivation). Rejected/general images have no tractable
                    // support here: HARD (loud refusal in run() iff supplied).
                    bool soft = !pr.rejected && !pr.general;
                    if (soft)
                        for (int j = 0; j < n && soft; ++j) {
                            bool insupp = pr.c_prop.v[j] || (pr.c_prop.a[j] & 3);
                            if (!insupp)
                                for (int l = 0; l < n && !insupp; ++l)
                                    if (l != j && pr.c_prop.B.get(std::min(j, l), std::max(j, l)))
                                        insupp = true;
                            if (insupp && ((tgt[(size_t)(j >> 6)] >> (j & 63)) & 1))
                                soft = false;
                        }
                    frame_poison_[(size_t)slot] = soft ? 2 : 1;
                    continue;
                }
                Pauli& P = conj[(size_t)slot];
                uint64_t* dst = comp ? &frame_zrow_[(size_t)i * CW]
                                     : &frame_xrow_[(size_t)i * CW];
                for (int j = 0; j < n; ++j) {
                    if (pr.c_prop.v[j]) {                         // x-support at the anchor
                        P.setx(j);
                        const uint64_t* row = &pxrow[(size_t)j * CW];
                        for (int w = 0; w < CW; ++w) dst[w] ^= row[w];
                    }
                    if (pr.c_prop.a[j] & 2) {                     // z-support at the anchor
                        P.setz(j);
                        const uint64_t* row = &pzrow[(size_t)j * CW];
                        for (int w = 0; w < CW; ++w) dst[w] ^= row[w];
                    }
                }
                if (born_obs)
                    frame_obs_bits_[(size_t)slot] = Pauli::anticommute_bit(born_W, P);
            }
        }
        frame_dec_bits_.assign(born_dec.size(),
                               std::vector<uint8_t>((size_t)(2 * pnq), 0));
        for (size_t j = 0; j < born_dec.size(); ++j)
            for (int s2 = 0; s2 < 2 * pnq; ++s2)
                if (!frame_poison_[(size_t)s2])
                    frame_dec_bits_[j][(size_t)s2] =
                        Pauli::anticommute_bit(born_dec[j].second, conj[(size_t)s2]);
    }
}

bool TwirlRecordSampler::Impl::run(long shots) {
    // S2.2: state retention only supports the DIAGONAL residual class. A CH/PPR circuit routes
    // shots through the PPR fast path (which never materialises amps) — refuse loudly rather
    // than silently deliver un-retained states.
    // Task 2b.1: Born DECISIONs also need the materialised per-shot collapsed state (to
    // measure_pauli on) — they force the same need_amps slow path as retention. `need_state`
    // is the union; decision-free non-retaining circuits keep need_state == false (byte-identical).
    const bool need_state = retain_state || has_born_dec_;
    // Anchor-time frame conjugation: a POISONED component (its t=0 port Pauli does not
    // propagate to a Pauli at the anchor — a non-Clifford gate in the port cone) is an error
    // ONLY if a shot actually supplies it. Frame-free runs of the same compile are untouched.
    // Loud refusal, never silent-wrong (the pre-fix code silently mis-relabeled instead).
    if (has_input_frame_ && frame_data_) {
        const int pnq = (int)frame_port_qubits_.size();
        for (int s2 = 0; s2 < 2 * pnq; ++s2) {
            if (frame_poison_[(size_t)s2] != 1) continue;   // clean or SOFT (exact no-op): allowed
            for (long sh = 0; sh < shots; ++sh)
                if (frame_data_[(size_t)sh * (size_t)frame_stride_ + (size_t)s2]) {
                    std::fprintf(stderr,
                        "twirl_records: input_pauli frame %c on port qubit %d does not "
                        "propagate to a Pauli through this stage's unitaries (non-Clifford "
                        "gate in the port cone) — the sign relabel is undefined; refused\n",
                        s2 < pnq ? 'X' : 'Z',
                        frame_port_qubits_[(size_t)(s2 < pnq ? s2 : s2 - pnq)]);
                    return false;
                }
        }
    }
    // fresh per run() (p is a pure function of the key ⇒ re-derivation exact)
    born_memo_.clear(); born_memo_mask_ = 0; born_memo_count_ = 0;
    // Fix 1: injective u64 Born-memo + plan-id map — same lifetime as born_memo_.
    u64_memo_.clear(); u64_memo_mask_ = 0; u64_memo_count_ = 0;
    plan_ids_.clear(); cur_plan_ = nullptr; cur_prefix_ = nullptr;
    u64_budget_fallbacks_ = 0; u64_budget_warned_ = false;
    if (need_state && ppr_tables) {
        std::fprintf(stderr, "twirl_records: state retention / Born decisions unsupported for "
                             "CH/PPR circuits\n");
        return false;
    }
    uint64_t bits[8];                                     // CW ≤ 8 (nchan ≤ 512)
    long done = 0;
    double t_loop = 0.0;
    (void)t_loop;
    // Ephemeral profiling (QEC_TW_PROF=1): stage split of the abelian fast path.
    const bool prof = std::getenv("QEC_TW_PROF") != nullptr;
    // v2.7 selfcheck once-per-sampler (option c): `sc` is the EFFECTIVE window for this
    // run() — the configured `selfcheck` on the first run() of the sampler's lifetime, 0 on
    // every later run(). The latch is never reset by set_seed() (which zeroes the
    // `checked`/`pprchecked` gating counters and would otherwise re-arm the window on every
    // seeded call); rebuild() makes a fresh Impl → the window re-arms (fresh-construction
    // semantics). Zero-shot calls do not spend the window (nothing would have been checked).
    // STREAM-NEUTRAL: a spent window is exactly selfcheck==0, byte-identical to window-on.
    const long sc = selfcheck_spent_ ? 0 : selfcheck;
    if (shots > 0) selfcheck_spent_ = true;
    // count_channels_: enable channel-accumulation when profiling, QEC_TW_COUNT is set,
    // OR when the selfcheck oracle is active (the C++ test uses channel_one_counts() as
    // a correctness gate in the same window). Fast path (sc==0, incl. spent): disabled.
    count_channels_ = prof
                   || (std::getenv("QEC_TW_COUNT") != nullptr)
                   || (sc > 0);
    if (sc > 0 && !xtim_quiet()) {
        static bool _sc_noticed = false;
        if (!_sc_noticed) {
            _sc_noticed = true;
            fprintf(stderr, "[xtim] QEC_TW_SELFCHECK=%ld: oracle window active on the fast path (first run of this sampler only; slower; diagnostic)\n", sc);
        }
    }
    double p_key = 0.0, p_find = 0.0, p_emit = 0.0, p_pair = 0.0;
    double p_coins = 0.0, p_post = 0.0, p_sink = 0.0;
    double p_prolog = 0.0, p_rf = 0.0;
    double probe_ns = 0.0;
    if (prof) {                       // calibrate the probe cost itself so the ledger closes
        auto c0 = clk::now();
        for (int i = 0; i < 1000000; ++i) { auto x = clk::now(); (void)x; }
        probe_ns = std::chrono::duration<double>(clk::now() - c0).count() * 1e3;  // ns per clk::now()
    }
    // Cold-path (plan build) stage split, same flag: accumulate+materialize / canonicalize /
    // get_or_build (build_shot_law on a miss) / channel rows + local index.
    double c_mat = 0.0, c_canon = 0.0, c_build = 0.0, c_rows = 0.0;
    long slow_shots = 0;
    if (prof) fprintf(stderr, "[twirl_records] n=%d NW=%d PW=%d upairs=%zu alts=%zu CW=%d\n",
                      n, NW, PW, upairs.size(), alt_masks.size(), CW);
    auto t0 = clk::now();
    while (done < shots) {
        const int S = (int)std::min<long>(4096, shots - done);
        auto td = clk::now();
        smp->sample_events(S, batch);
        t_draw += std::chrono::duration<double>(clk::now() - td).count();
        auto tl = clk::now();
        (void)tl;
        for (int s = 0; s < S; ++s, ++done) {
            auto tp = prof ? clk::now() : clk::time_point();
            const int32_t e0 = batch.shot_off[s], e1 = batch.shot_off[s + 1];
            // Per-shot derived RNG: re-seed shot_rng from (seed_, done) in O(1) — no full
            // state initialization; every coin draw is a pure function of (seed, shot_index,
            // plan) — routing-independent. The reservoir is reset here so it never carries
            // state across shots (the invariant the test checks).
            shot_rng.seed(seed_, (uint64_t)done);
            coin_res = 0; coin_left = 0;
            for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
            // Task 4: set per-shot frame row pointer (nullptr => apply_input_frame is no-op)
            cur_frame_row_ = (has_input_frame_ && frame_data_)
                             ? frame_data_ + (size_t)done * (size_t)frame_stride_
                             : nullptr;
            if (nrf) { draw_rf(); ++done_rf_; }    // Stim M(p) readout coins (every shot); advance rf counter
            if (prof) p_prolog += std::chrono::duration<double>(clk::now() - tp).count();
            bool exact_cur = false;                // V3-T3 R1: shot took the exact fallback
            // S2.2: per-shot collapsed state + its raw generator syndrome delivered to the sink
            // under retention (nullptr on the production path — no state materialised).
            const FramedSuperposition* amps_out = nullptr;
            const uint64_t* sig_out = nullptr;
            // Record-hash bucketing: the canonical residual plan bytes (x/z support of the prefix +
            // a-mask + cz, EXCLUDING the global phase — approx_equal is phase-insensitive). Together
            // with sig_out + coins it is a SUFFICIENT statistic for the collapsed amps. nullptr on
            // clean/identity shots (empty plan) and fallback shots.
            const uint8_t* plan_out = nullptr;
            int plan_n = 0;
            // port-v3 T2 (compact record): the shot's plan identity in COMPACT form — the
            // interned CachedPlan pointer + the residual prefix support words. Set ONLY where
            // plan bytes would be non-empty (the retention non-fallback branch), so the sink's
            // compact identity is null exactly when the legacy plan component is empty.
            const CachedPlan* plan_ptr_out = nullptr;
            const uint64_t* pre_x_out = nullptr;
            const uint64_t* pre_z_out = nullptr;
            // decoder-feedback perf: the coin record is per-shot. Clear it up front (retention
            // sink only — the null/production paths never read o_rec.coins, kept byte-identical)
            // so a CLEAN shot never delivers a STALE coin record from a prior diagonal shot (the
            // born outcomes are appended in the tail; twirl_collapse overwrites it on diag shots).
            if (retain_state) o_rec.coins.clear();
            // Task 2b.1: the MUTABLE per-shot state Born decisions measure on (nullptr = none).
            FramedSuperposition* dec_state_ = nullptr;
            if (e0 == e1) {
                ++clean;
                if (born_obs)
                    obs_bit_cur = (obs_u01() < (1.0 - m_clean) / 2.0) ? 1 : 0;
                // S2.2: a no-error shot's collapsed post-barrier state IS the bare reference
                // (identity residual), with an all-zero certified-generator syndrome.
                if (retain_state) {
                    amps_out = &bare;
                    retain_sig.assign((size_t)GW, 0);
                    sig_out = retain_sig.data();
                }
                // Task 2b.1 / decoder-feedback perf: a clean shot's collapsed state is the BARE
                // reference (identity residual, empty coins/plan). Route the Born decision through
                // the MEMO path (dec_state_ = nullptr) exactly like a diagonal shot: p_j = born_p1(W_j)
                // on the bare is a CONSTANT (no per-shot noise), so it is computed ONCE on the first
                // clean shot (a memo miss → materialize_shot(identity)) and served from the memo for
                // every subsequent clean shot — no per-shot bare COPY, no per-shot born_p1 measure
                // (the pre-perf hot path did both ~70% of shots). The memo key components are the
                // clean-shot record: σ = 0 (GW words), empty coins, empty plan. dec_gen draw order is
                // UNCHANGED (dec_u01 is drawn per decision in both branches) ⇒ byte-identical outputs.
                if (has_born_dec_) {
                    o_rec.sigma.assign((size_t)GW, 0);
                    o_rec.coins.clear();
                    plankey_.clear();
                    dec_state_ = nullptr;
                    cur_plan_ = nullptr;    // identity residual ⇒ plan id 0, prefix signs 0
                    cur_prefix_ = nullptr;
                }
            }
            else {
                ++diag;
                // ── PPR SHOT (V2-T1): ≥1 fired alt carries a PPR-retry atom — compose in the
                // PPR algebra (engine order: last-fired leftmost), memoized law, σ-channel
                // emission. All-diagonal shots fall through to the abelian fast path below.
                if (ppr_tables) {
                    bool any_ppr = false;
                    for (int32_t e = e0; e < e1; ++e)
                        if (alt_is_ppr[(size_t)batch.ev[e]]) { any_ppr = true; break; }
                    if (any_ppr) {
                        ppr_views.clear();
                        bool ok_alt = true;
                        for (int32_t e = e1 - 1; e >= e0; --e) {
                            const PprNormalForm& ap2 = alt_pnf[(size_t)batch.ev[e]];
                            if (!ap2.commuting_class) { ok_alt = false; break; }
                            ppr_views.push_back(PprAtom{&ap2.prefix, &ap2.rots});
                        }
                        PprNormalForm pnf = ok_alt ? compose_ppr(n, ppr_views)
                                                   : PprNormalForm(n);
                        // V3-T3 R1: out-of-class compositions / plan fallbacks are computed
                        // per shot by the exact engine (generator pullback through the fired
                        // alts' error tableaus) and INCLUDED — no shot is ever dropped.
                        // Neither refused branch has consumed a coll_gen draw, so downstream
                        // shots keep their exact byte streams.
                        bool ppr_exact = !ok_alt || !pnf.commuting_class;
                        const PprCachedPlan* plp = nullptr;
                        if (!ppr_exact) {
                            plp = &pcache->get_or_build(G, pnf);
                            ppr_exact = plp->fallback;
                        }
                        if (ppr_exact) {
                            ++fb;
                            exact_ppr_shot(batch, e0, e1, bits);
                            apply_rf(bits);
                            apply_input_frame(bits);
                            ++used;
                            if (count_channels_) { for (int w2 = 0; w2 < CW; ++w2) { uint64_t word = bits[w2];
                                while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                    ++chancnt[c]; } } }
                            emit_shot(bits, 0, true, nullptr, nullptr, emit_decisions(bits, nullptr), nullptr, 0, nullptr, 0, nullptr, 0);
                            continue;
                        }
                        const PprCachedPlan& pl = *plp;
                        if (pl.ch_token != chtok) build_ppr_rows(pl);
                        if (pl.folds_built != 1) build_ppr_folds(pl);
                        if (born_obs) {
                            ensure_obs_ppr(pl, pnf);
                            if (pl.obs_guard) {            // foldable-kernel / classify guard →
                                exact_ppr_shot(batch, e0, e1, bits);   // whole shot exact
                                apply_rf(bits);
                                apply_input_frame(bits);
                                ++fb;
                                ++used; ++pprshots;
                                if (count_channels_) { for (int w2 = 0; w2 < CW; ++w2) { uint64_t word = bits[w2];
                                    while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                        ++chancnt[c]; } } }
                                obs_ones_ += obs_bit_cur;
                                bool acc0 = true;
                                for (int w2 = 0; w2 < CW; ++w2)
                                    if (bits[w2] & det_maskw[w2]) { acc0 = false; break; }
                                if (acc0) { ++ps_acc_; ps_obs_ones_ += obs_bit_cur; }
                                emit_shot(bits, obs_bit_cur, false, nullptr, nullptr, emit_decisions(bits, nullptr), nullptr, 0, nullptr, 0, nullptr, 0);
                                continue;
                            }
                        }
                        // PPR SELFCHECK (first --selfcheck ppr shots): rebuild the law
                        // DIRECTLY with the shot prefix and require every invariant field to
                        // match the memo route; base0 ⊕ plane1 may differ from the direct
                        // det_signs only by an element of span(coins ∪ fold masks).
                        if (pprchecked < sc) {
                            ++pprchecked;
                            ShotLaw dl = build_shot_law_ppr(G, pnf);
                            bool bad = dl.fallback || dl.r != pl.r || dl.kappa != pl.kappa ||
                                       dl.coin_masks != pl.coin_masks ||
                                       dl.kernel_masks != pl.kernel_masks ||
                                       dl.kernel_foldable != pl.kernel_foldable ||
                                       dl.kernel_base != pl.kernel_base;
                            if (!bad) {
                                std::vector<uint64_t> diff = dl.det_signs;
                                const std::vector<uint64_t> pl1 = prefix_plane1(G, pnf.prefix);
                                for (int w2 = 0; w2 < GW; ++w2) diff[w2] ^= pl.base0[w2] ^ pl1[w2];
                                // reduce diff against rref(coins ∪ fold masks)
                                std::vector<std::vector<uint64_t>> rows = pl.coin_masks;
                                for (const auto& km : pl.kernel_masks)
                                    if (!km.empty()) rows.push_back(km);
                                std::vector<std::vector<uint64_t>> rr;
                                std::vector<int> piv;
                                for (auto rrow : rows) {
                                    for (size_t k2 = 0; k2 < rr.size(); ++k2)
                                        if ((rrow[piv[k2] >> 6] >> (piv[k2] & 63)) & 1)
                                            for (int w2 = 0; w2 < GW; ++w2) rrow[w2] ^= rr[k2][w2];
                                    int pc = -1;
                                    for (int w2 = 0; w2 < GW && pc < 0; ++w2)
                                        if (rrow[w2]) pc = w2 * 64 + __builtin_ctzll(rrow[w2]);
                                    if (pc >= 0) { rr.push_back(rrow); piv.push_back(pc); }
                                }
                                for (size_t k2 = 0; k2 < rr.size(); ++k2)
                                    if ((diff[piv[k2] >> 6] >> (piv[k2] & 63)) & 1)
                                        for (int w2 = 0; w2 < GW; ++w2) diff[w2] ^= rr[k2][w2];
                                for (int w2 = 0; w2 < GW; ++w2) if (diff[w2]) bad = true;
                            }
                            if (bad) {
                                fprintf(stderr, "twirl_records PPR SELFCHECK FAIL shot=%ld\n", done);
                                return false;
                            }
                        }
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] = pl.ch_baseline[w2];
                        const Pauli& P2 = pnf.prefix;
                        for (size_t w = 0; w < P2.x.size(); ++w) { uint64_t word = P2.x[w];
                            while (word) { const int q = (int)(w * 64 + __builtin_ctzll(word)); word &= word - 1;
                                const uint64_t* row = &pxrow[(size_t)q * CW];
                                for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; } }
                        for (size_t w = 0; w < P2.z.size(); ++w) { uint64_t word = P2.z[w];
                            while (word) { const int q = (int)(w * 64 + __builtin_ctzll(word)); word &= word - 1;
                                const uint64_t* row = &pzrow[(size_t)q * CW];
                                for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; } }
                        int obs_par = born_obs ? (int)pl.obs_par_base : 0;
                        for (size_t k = 0; k < pl.ch_coin_rows.size(); ++k) {
                            if (coin_left == 0) { coin_res = shot_rng.next(); coin_left = 64; }
                            const bool cb = coin_res & 1; coin_res >>= 1; --coin_left;
                            if (cb) {
                                for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= pl.ch_coin_rows[k][w2];
                                if (born_obs) obs_par ^= pl.obs_par_coin[k];
                            }
                        }
                        for (size_t j = 0; j < pl.kernel_logicals.size(); ++j) {
                            if (!pl.kernel_foldable[j]) continue;
                            if (shot_rng.next_double() < pl.fold_p1[j])
                                for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= pl.ch_kernel_rows[j][w2];
                        }
                        if (born_obs) {
                            int pw = 0;                    // ⟨P, W⟩ — the only prefix correction
                            const Pauli& Pp = pnf.prefix;
                            for (int w2 = 0; w2 < NW; ++w2)
                                pw += __builtin_popcountll(Pp.x[w2] & born_W.z[w2]) +
                                      __builtin_popcountll(Pp.z[w2] & born_W.x[w2]);
                            obs_par ^= (pw & 1);
                            const double val = pl.obs_m * (obs_par ? -1.0 : 1.0);
                            obs_bit_cur = (obs_u01() < (1.0 - val) / 2.0) ? 1 : 0;
                        }
                        apply_rf(bits);
                        apply_input_frame(bits);
                        ++used; ++pprshots;
                        if (count_channels_) { for (int w2 = 0; w2 < CW; ++w2) { uint64_t word = bits[w2];
                            while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                ++chancnt[c]; } } }
                        if (born_obs) {
                            obs_ones_ += obs_bit_cur;
                            bool acc0 = true;
                            for (int w2 = 0; w2 < CW; ++w2)
                                if (bits[w2] & det_maskw[w2]) { acc0 = false; break; }
                            if (acc0) { ++ps_acc_; ps_obs_ones_ += obs_bit_cur; }
                        }
                        emit_shot(bits, obs_bit_cur, false, nullptr, nullptr, emit_decisions(bits, nullptr), nullptr, 0, nullptr, 0, nullptr, 0);
                        continue;
                    }
                }
                // ── ABELIAN FAST PATH: key by XOR of per-alt masks, emission by per-alt rows +
                // ordered-pair z-corrections. No accumulate, no content materialisation. Runs
                // outside the selfcheck window; misses fall through to the materialising path.
                // (born_obs: bypassed — the observable's ⟨P,W⟩ parity needs the materialised
                // prefix; diag shots take the slow path. cs profile: ~17.5% diag ⇒ still ≫ prod.)
                // Task 2b.1: need_state (retention / Born decisions) also forces the slow path —
                // this hashed shortcut never materialises amps.
                if (!born_obs && checked >= sc && !need_state) {
                    auto pk = prof ? clk::now() : clk::time_point();
                    // One streaming pass over the fired alts' blobs: key XOR + emission row +
                    // linearized pair term (S_acc/V_acc running masks) in a single block visit.
                    // Pre-pass: XOR the per-alt hash tokens (GF(2)-linear hash ⇒ shot hash is
                    // token-XOR). This touches each blob's first line — all DRAM misses issue
                    // in PARALLEL — and lets us prefetch the ftab slot before assembling the key.
                    uint64_t hh = 0xcbf29ce484222325ull;
                    for (int32_t e = e0; e < e1; ++e) {
                        const uint32_t off = alt_off[(size_t)batch.ev[e]];
                        const uint64_t* B = &alt_blob[off];
                        const uint32_t end = ((size_t)batch.ev[e] + 1 < alt_off.size())
                                                 ? alt_off[(size_t)batch.ev[e] + 1]
                                                 : (uint32_t)alt_blob.size();
                        for (uint32_t l = 8; l < end - off; l += 8) __builtin_prefetch(B + l);
                        hh ^= B[1];
                    }
                    __builtin_prefetch(&ftab[(hh & (fslots - 1)) * (size_t)(KW + 1)]);
                    for (int w2 = 0; w2 < KW; ++w2) ab_kw[w2] = 0;
                    const bool multi = (e1 - e0) > 1;
                    if (multi) for (int w2 = 0; w2 < NW; ++w2) { ab_S[w2] = 0; ab_V[w2] = 0; ab_T[w2] = 0; }
                    for (int32_t e = e0; e < e1; ++e) {
                        const uint64_t* B = &alt_blob[alt_off[(size_t)batch.ev[e]]];
                        const uint32_t nc = (uint32_t)B[0], np = (uint32_t)(B[0] >> 32);
                        const uint64_t* km = B + 2;
                        const uint64_t* sm = km + NW;
                        const uint64_t* vm = sm + NW;
                        const uint64_t* zr = vm + NW;
                        const uint64_t* ce = zr + CW;
                        for (int w2 = 0; w2 < NW; ++w2) ab_kw[w2] ^= km[w2];
                        for (uint32_t i = 0; i < nc; ++i) ab_kw[NW + (int)ce[2 * i]] ^= ce[2 * i + 1];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= zr[w2];
                        if (multi) {
                            if (e > e0) {
                                // T_{kj} = (s_j∧s_k)⊕(s_j∧v_k)⊕B_j·v_k is GF(2)-linear in the
                                // k-side, so Σ_{k<j} folds into the running S/V accumulators.
                                // (The selfcheck bits-oracle keeps the O(fired²) form.)
                                for (int w2 = 0; w2 < NW; ++w2)
                                    ab_T[w2] ^= sm[w2] & (ab_S[w2] ^ ab_V[w2]);
                                const uint64_t* pp = ce + 2 * nc;
                                for (uint32_t i = 0; i < np; ++i) {
                                    const int x = (int)(uint32_t)pp[i], y = (int)(uint32_t)(pp[i] >> 32);
                                    if ((ab_V[y >> 6] >> (y & 63)) & 1) ab_T[x >> 6] ^= (1ULL << (x & 63));
                                    if ((ab_V[x >> 6] >> (x & 63)) & 1) ab_T[y >> 6] ^= (1ULL << (y & 63));
                                }
                            }
                            for (int w2 = 0; w2 < NW; ++w2) { ab_S[w2] ^= sm[w2]; ab_V[w2] ^= vm[w2]; }
                        }
                    }
                    if (prof) { auto pf = clk::now(); p_key += std::chrono::duration<double>(pf - pk).count(); pk = pf; }
                    const CachedPlan* ap = ffind_h(ab_kw.data(), hh);
                    if (prof) { auto pf = clk::now(); p_find += std::chrono::duration<double>(pf - pk).count(); pk = pf; }
                    if (ap && !ap->fallback && ap->kernel_logicals.empty() &&
                        ap->ch_token == chtok) {
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= ap->ch_baseline[w2];
                        if (multi)
                            for (int w2 = 0; w2 < NW; ++w2) { uint64_t word = ab_T[w2];
                                while (word) { const int q = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                    const uint64_t* row = &pzrow[(size_t)q * CW];
                                    for (int w3 = 0; w3 < CW; ++w3) bits[w3] ^= row[w3]; } }
                        if (prof) { auto pf = clk::now(); p_pair += std::chrono::duration<double>(pf - pk).count(); pk = pf; }
                        for (size_t k = 0; k < ap->ch_coin_rows.size(); ++k) {
                            if (coin_left == 0) { coin_res = shot_rng.next(); coin_left = 64; }
                            const bool bit = coin_res & 1; coin_res >>= 1; --coin_left;
                            if (bit) for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= ap->ch_coin_rows[k][w2];
                        }
                        if (prof) { auto pf = clk::now(); p_coins += std::chrono::duration<double>(pf - pk).count(); pk = pf; }
                        apply_rf(bits);
                        if (prof) { auto pf = clk::now(); p_rf += std::chrono::duration<double>(pf - pk).count(); pk = pf; }
                        apply_input_frame(bits);
                        ++used;
                        if (count_channels_) { for (int w2 = 0; w2 < CW; ++w2) { uint64_t word = bits[w2];
                            while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                ++chancnt[c]; } } }
                        if (prof) { auto pf = clk::now(); p_post += std::chrono::duration<double>(pf - pk).count(); pk = pf; }
                        emit_shot(bits, 0, false, nullptr, nullptr, emit_decisions(bits, nullptr), nullptr, 0, nullptr, 0, nullptr, 0);
                        if (prof) { auto pf = clk::now(); p_sink += std::chrono::duration<double>(pf - pk).count(); }
                        continue;
                    }
                    // MISS: bits carries the streamed zr XORs — re-zero before the exact path
                    // (the record/fallback branch ORs into bits and relies on a clean start).
                    for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
                }
                // Sparse accumulate (then()'s exact recurrence; v1 = PRE-atom accumulated v).
                auto touch_a = [&](int q) { if (!in_a[q]) { in_a[q] = 1; a_dirty.push_back(q); } };
                auto touch_v = [&](int q) { if (!in_v[q]) { in_v[q] = 1; v_dirty.push_back(q); } };
                auto amask_upd = [&](int q) {
                    if (a_acc[q] & 1) amask_acc[q >> 6] |= (1ULL << (q & 63));
                    else              amask_acc[q >> 6] &= ~(1ULL << (q & 63)); };
                for (int32_t e = e0; e < e1; ++e) {
                    const AltAtom& A = alt_atoms[(size_t)batch.ev[e]];
                    for (const auto& qa : A.av) { const int q = qa.first; const uint8_t a2 = qa.second;
                        touch_a(q);
                        a_acc[q] = (uint8_t)((a_acc[q] + (v_acc[q] ? (uint8_t)((4 - a2) & 3) : a2)) & 3);
                        amask_upd(q); }
                    for (size_t i = 0; i < A.czp.size(); ++i) {
                        const int j = A.czp[i].first, l = A.czp[i].second;
                        if (v_acc[l]) { touch_a(j); a_acc[j] = (uint8_t)((a_acc[j] + 2) & 3); }
                        if (v_acc[j]) { touch_a(l); a_acc[l] = (uint8_t)((a_acc[l] + 2) & 3); }
                        const int pi2 = A.czidx[i];
                        if (!in_w[pi2 >> 6]) { in_w[pi2 >> 6] = 1; w_dirty.push_back(pi2 >> 6); }
                        czw[pi2 >> 6] ^= 1ull << (pi2 & 63);
                    }
                    for (int q : A.vq) { touch_v(q); v_acc[q] ^= 1; }
                }
                // This shot's cz list from the dirty words (sorted union list ⇒ sorted output).
                cz_s.clear();
                std::sort(w_dirty.begin(), w_dirty.end());
                for (int w : w_dirty) { uint64_t word = czw[w];
                    while (word) { const int idx = w * 64 + __builtin_ctzll(word); word &= word - 1;
                        cz_s.push_back(upairs[(size_t)idx]); } }
                // ── CANON IN ACCUMULATOR COORDINATES (closed form; final lever, 2026-07-15) ──
                // The product-wire strip maps the key to (amask & ~prodmask, kept pairs), and the
                // ONLY emission effect is prefix Z-flips on unstripped endpoints of pairs whose
                // stripped partner carries a −Z sign (sp_q = a_q + 2·neg_q mod 4 ⇒ a'_q = a_q,
                // flipz iff neg_q odd — XOR's self-inverse handles the parity per occurrence).
                // Stripped S-legs and both-stripped pairs are discarded global phases; stripped-
                // wire prefix-Z is inert (product wires have no X-support in any generator ⇒
                // pzrow ≡ 0 there). Derived from the canonicalize closed form; oracle'd below.
                key_amask_s.resize((size_t)NW);
                for (int w2 = 0; w2 < NW; ++w2) key_amask_s[w2] = amask_acc[w2] & ~prodmask[w2];
                cz_key.clear(); zflip.clear();
                for (const auto& e : cz_s) {
                    const int s1 = ZR->prod_sign[e.first], s2 = ZR->prod_sign[e.second];
                    if (s1 == 0 && s2 == 0) cz_key.push_back(e);
                    else if (s1 != 0 && s2 != 0) { /* both stripped: global phase */ }
                    else if (s1 != 0) { if (s1 < 0) zflip.push_back(e.second); }
                    else               { if (s2 < 0) zflip.push_back(e.first); }
                }
                // FAST PATH (no nf materialisation): canonical key + plan cached with rows
                // built + κ=0 + not fallback + outside the self-check window. Prefix supports
                // straight from the dirty lists (+ the strip's zflip corrections).
                // Plan-cache-sharing fix: use the Impl-local flat table (ffind) rather than
                // the shared TwirlPlanCache (cache->find). The flat table is populated
                // incrementally — identically to the pre-fix behaviour — so the fast/slow
                // branch decision is BYTE-IDENTICAL whether the plan cache is empty or
                // pre-populated from a previous rebuild. cache->find would always return a
                // hit for a warm cache, promoting slow-path shots to fast-path shots after
                // selfcheck and changing the coll_gen draw sequence.
                const CachedPlan* fplan = nullptr;
                if (!born_obs && checked >= sc && !need_state) {
                    // Compute the KW-word flat-table key from (key_amask_s, cz_key) — same
                    // encoding as plan_local_key / the LOCAL INDEX oracle's ab_kw.
                    ab_kw.assign((size_t)KW, 0);
                    for (int w2 = 0; w2 < NW; ++w2) ab_kw[w2] = key_amask_s[w2];
                    for (const auto& pr : cz_key) {
                        auto it = pidx.find(pr);
                        if (it != pidx.end())
                            ab_kw[(size_t)NW + (size_t)(it->second >> 6)] |= 1ull << (it->second & 63);
                    }
                    fplan = ffind(ab_kw.data());
                }
                if (fplan && !fplan->fallback && fplan->kernel_logicals.empty() &&
                    fplan->ch_token == chtok) {
                    for (int w2 = 0; w2 < CW; ++w2) bits[w2] = fplan->ch_baseline[w2];
                    for (int q : v_dirty) if (v_acc[q]) {
                        const uint64_t* row = &pxrow[(size_t)q * CW];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    for (int q : a_dirty) if (a_acc[q] & 2) {
                        const uint64_t* row = &pzrow[(size_t)q * CW];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    for (int q : zflip) {                        // strip's −Z-partner corrections
                        const uint64_t* row = &pzrow[(size_t)q * CW];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    // Coins from a 64-bit reservoir (fair iid bits; the σ-path draw-order
                    // convention only binds the self-check window, which runs the slow path).
                    for (size_t k = 0; k < fplan->ch_coin_rows.size(); ++k) {
                        if (coin_left == 0) { coin_res = shot_rng.next(); coin_left = 64; }
                        const bool bit = coin_res & 1; coin_res >>= 1; --coin_left;
                        if (bit) for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= fplan->ch_coin_rows[k][w2];
                    }
                    for (int q : a_dirty) { a_acc[q] = 0; in_a[q] = 0; }
                    std::fill(amask_acc.begin(), amask_acc.end(), 0ULL);
                    for (int q : v_dirty) { v_acc[q] = 0; in_v[q] = 0; }
                    for (int w : w_dirty) { czw[w] = 0; in_w[w] = 0; }
                    a_dirty.clear(); v_dirty.clear(); w_dirty.clear();
                    apply_rf(bits);
                    apply_input_frame(bits);
                    ++used;
                    if (count_channels_) {
                        for (int w2 = 0; w2 < CW; ++w2) {            // popcount channel accumulation
                            uint64_t word = bits[w2];
                            while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                ++chancnt[c]; }
                        }
                    }
                    emit_shot(bits, 0, false, nullptr, nullptr, emit_decisions(bits, nullptr), nullptr, 0, nullptr, 0, nullptr, 0);
                    continue;
                }
                // SLOW path: materialise the normal form (miss / canon-touched / self-check).
                auto ck = prof ? clk::now() : clk::time_point();
                ++slow_shots;
                std::fill(nf.a.begin(), nf.a.end(), (uint8_t)0);
                if ((int)nf.a.size() != n) nf.a.assign((size_t)n, 0);
                std::fill(nf.prefix.x.begin(), nf.prefix.x.end(), 0ULL);
                std::fill(nf.prefix.z.begin(), nf.prefix.z.end(), 0ULL);
                nf.prefix.phase = 0;
                nf.cz = cz_s;
                nf.diagonal_class = true;
                for (int q : a_dirty) { const int aq = a_acc[q] & 3;
                    if (aq & 1) nf.a[q] = 1;
                    if (aq & 2) nf.prefix.setz(q); }
                for (int q : v_dirty) if (v_acc[q]) nf.prefix.setx(q);
                // Reset the accumulators (dirty entries only; membership flags cleared).
                for (int q : a_dirty) { a_acc[q] = 0; in_a[q] = 0; }
                std::fill(amask_acc.begin(), amask_acc.end(), 0ULL);
                for (int q : v_dirty) { v_acc[q] = 0; in_v[q] = 0; }
                for (int w : w_dirty) { czw[w] = 0; in_w[w] = 0; }
                a_dirty.clear(); v_dirty.clear(); w_dirty.clear();
                // ORACLE (first --selfcheck diag shots): the accumulator's nf must equal the
                // exact compose_fired chain's nf EXACTLY (a, cz, prefix supports; phase is
                // discarded by contract) — both post-canonicalization.
                if (checked < sc) {
                    fired.clear();
                    for (int32_t e = e0; e < e1; ++e)
                        for (const FiredPauli& f : sides[(size_t)batch.ev[e]]) fired.push_back(f);
                    DiagPauliClifford D = compose_fired(table, fired);
                    D.finalize();
                    DiagNormalForm nfe = residual_normal_form(D);
                    canonicalize_mod_stabilizers_inplace(nfe, G);
                    DiagNormalForm nfa = nf;               // copy; canon applied below to nf too
                    canonicalize_mod_stabilizers_inplace(nfa, G);
                    if (nfa.a != nfe.a || nfa.cz != nfe.cz ||
                        nfa.prefix.x != nfe.prefix.x || nfa.prefix.z != nfe.prefix.z) {
                        fprintf(stderr, "twirl_records ACCUMULATOR ORACLE FAIL shot=%ld\n", done);
                        return false;
                    }
                    // KEY ORACLE: the accumulator-coordinates canonical key (strip closed form)
                    // must equal the post-canon nf's packed content EXACTLY — the wiring proof
                    // for the canon-in-accumulator fast path.
                    std::vector<uint64_t> ka((size_t)NW, 0);
                    for (int q = 0; q < n; ++q)
                        if (nfe.a[q] & 1) ka[q >> 6] |= (1ULL << (q & 63));
                    if (ka != key_amask_s || nfe.cz != cz_key) {
                        fprintf(stderr, "twirl_records KEY ORACLE FAIL shot=%ld\n", done);
                        return false;
                    }
                    // ABELIAN ORACLE (key): the XOR-of-per-alt-masks key must equal the
                    // materialised+canonicalised key EXACTLY (the abelian fast path's licence).
                    for (int w2 = 0; w2 < NW; ++w2) ab_kmask[w2] = 0;
                    for (int w2 = 0; w2 < PW; ++w2) ab_czk[w2] = 0;
                    for (int32_t e = e0; e < e1; ++e) {
                        const AltMasks& M = alt_masks[(size_t)batch.ev[e]];
                        for (int w2 = 0; w2 < NW; ++w2) ab_kmask[w2] ^= M.kmask[w2];
                        for (int w2 = 0; w2 < PW; ++w2) ab_czk[w2]  ^= M.czk[w2];
                    }
                    ab_cz.clear();
                    for (int w2 = 0; w2 < PW; ++w2) { uint64_t word = ab_czk[w2];
                        while (word) { const int idx = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                            ab_cz.push_back(upairs[(size_t)idx]); } }
                    if (ab_kmask != key_amask_s || ab_cz != cz_key) {
                        fprintf(stderr, "twirl_records ABELIAN KEY ORACLE FAIL shot=%ld\n", done);
                        return false;
                    }
                    // decoder-feedback perf: SELF-LIMIT the accumulator/key oracle on the need_state
                    // (barrier/decision) path. The non-need_state counting path increments `checked`
                    // in its own do_check branch (below), turning this verification off after the
                    // first `selfcheck` diag shots; the need_state branch never reaches that
                    // increment, so without this the (compose_fired + double-canonicalize) oracle ran
                    // for EVERY diag shot forever (~1.2 µs/shot — the dominant barrier cost). The
                    // oracle has NO output side effects (it aborts on a wiring mismatch, nothing
                    // else), so limiting it to the first `selfcheck` shots leaves every sampled record
                    // byte-identical — it only stops re-proving a property already verified 2000×.
                    if (need_state) ++checked;
                }
                if (prof) { auto cf = clk::now(); c_mat += std::chrono::duration<double>(cf - ck).count(); ck = cf; }
                canonicalize_mod_stabilizers_inplace(nf, G);
                if (prof) { auto cf = clk::now(); c_canon += std::chrono::duration<double>(cf - ck).count(); ck = cf; }
                const CachedPlan& plan = cache->get_or_build(G, nf);
                if (prof) { auto cf = clk::now(); c_build += std::chrono::duration<double>(cf - ck).count(); ck = cf; }
                // ── S2.2 RETENTION: materialise this shot's collapsed post-barrier state ──────
                // twirl_collapse(need_amps=true) runs the SAME diagonal law as the fast path but
                // ALSO returns o_rec.amps = the residual-applied, kernel-collapsed state, and
                // o_rec.sigma = the raw certified-generator syndrome. Both are genuinely per-shot
                // (they differ across shots whose fired errors give different residuals / Born
                // outcomes). Delivered to the sink; the fast-path channel machinery is bypassed.
                // Task 2b.1: need_state = retention OR Born decisions (both want the amps).
                if (need_state) {
                    // decoder-feedback perf (Item 1): produce σ+coins via twirl_collapse_record —
                    // it reads DIRECTLY from the CachedPlan (base0 ⊕ columnar prefix, then the r fair
                    // coins / κ chain) and NEVER materialises a per-shot ShotLaw. The old path called
                    // plan.finalize() every diag shot, which COPIES coin_masks/kernel_*/tier1/base0
                    // into a fresh ShotLaw (GW-word vectors × r) — pure per-shot heap+copy churn — and
                    // then twirl_collapse(need_amps=false,κ=0) only re-consumed det_signs + the r coins.
                    // twirl_collapse_record is the SAME σ/coins computation (already the counting path's
                    // κ>0 + self-check routine, lines below) with IDENTICAL coll_rng draw order (r fair
                    // coins then the κ chain) ⇒ σ, coins, and every downstream record byte are identical;
                    // only the ShotLaw copy is removed. amps are not read here (memoised born path).
                    twirl_collapse_record(bare, G, nf, plan, coll_rng, o_rec);
                    for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
                    if (o_rec.fallback) {
                        // Out-of-class / mid-chain guard trip: recompute the whole shot exactly
                        // (fresh bare + residual + generators measured -> fbwork, fb_sig).
                        ++fb; exact_cur = true;
                        exact_diag_shot(nf, bits);
                        amps_out = &fbwork;
                        sig_out = fb_sig.data();
                        dec_state_ = &fbwork;     // Born decisions measure on the exact state
                    } else {
                        for (int c = 0; c < nchan; ++c)
                            if (par(o_rec.sigma, sch[(size_t)c].mask))
                                bits[c >> 6] |= (1ull << (c & 63));
                        amps_out = nullptr;       // amps not materialised here (memoised born path)
                        retain_sig.assign(o_rec.sigma.begin(), o_rec.sigma.end());
                        if ((int)retain_sig.size() < GW) retain_sig.resize((size_t)GW, 0);
                        sig_out = retain_sig.data();
                        dec_state_ = nullptr;     // nullptr ⇒ emit_decisions takes the memo path
                        // Fix 1 (design B): capture the CachedPlan (→ dense id) and the residual
                        // prefix (→ ⟨W_i,prefix⟩ sign bits) for the injective born-memo key. The plan
                        // pointer is stable (interned in the cache); nf.prefix is not mutated between
                        // here and emit_decisions in this shot.
                        cur_plan_ = &plan;
                        cur_prefix_ = &nf.prefix;
                        // port-v3 T2 (compact record): hand the sink the plan identity in compact
                        // form. The (a,cz) tail is a pure function of the interned CachedPlan
                        // (cache key EXACTLY (a-mask&1, cz), full content equality — see
                        // get_or_build); the prefix words are the only per-shot part.
                        plan_ptr_out = &plan;
                        pre_x_out = nf.prefix.x.data();
                        pre_z_out = nf.prefix.z.data();
                        // L-E lever: with a COMPACT barrier sink and no Born decisions, the
                        // per-shot plan-byte serialization below is pure record-key work — the
                        // sink reconstructs the identical bytes per GROUP from the compact
                        // identity. Skip it (no RNG, no output side effects; the born memo
                        // path still needs plankey_ for materialize_shot, so born circuits
                        // keep serializing).
                        const bool skip_plankey =
                            barrier_sink_ && barrier_sink_->compact && !has_born_dec_;
                        if (skip_plankey) {
                            plankey_.clear();
                            plan_out = nullptr;
                            plan_n = 0;
                        } else {
                        // Serialize the canonical residual plan as the record key's plan component
                        // (prefix x/z support + a-mask + cz; the global phase is dropped — see above).
                        // barrier2 Lever 2: SOUND fast serialization. NOTE the plan-cache key is
                        // (a-mask, cz) — it EXCLUDES nf.prefix, which varies per shot (it carries the
                        // residual Pauli; canonicalize copies nf.prefix + flipz-folds S²=Z). So plankey
                        // is NOT a pure function of the CachedPlan and CANNOT be cached on the plan
                        // (that would serve a stale prefix). We take the audit's alternative branch:
                        // reserve once + bulk-copy the prefix WORDS (LE byte layout == the old nested
                        // 8-iteration bit loop) instead of ~2·NW·8 push_backs. Byte-identical output.
                        plankey_.clear();
                        plankey_.reserve(nf.prefix.x.size() * 8 + nf.prefix.z.size() * 8 +
                                         (size_t)n + 1 + nf.cz.size() * 4);
                        auto append_words = [&](const std::vector<uint64_t>& w) {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
                            const uint8_t* wp = reinterpret_cast<const uint8_t*>(w.data());
                            plankey_.insert(plankey_.end(), wp, wp + w.size() * 8);
#else
                            for (uint64_t word : w)
                                for (int b = 0; b < 8; ++b)
                                    plankey_.push_back((uint8_t)((word >> (b * 8)) & 0xFF));
#endif
                        };
                        append_words(nf.prefix.x);
                        append_words(nf.prefix.z);
                        for (int q = 0; q < n; ++q) plankey_.push_back((uint8_t)(nf.a[(size_t)q] & 1));
                        plankey_.push_back(0xFF);   // separator before the cz list
                        for (const auto& pr : nf.cz) {
                            plankey_.push_back((uint8_t)(pr.first & 0xFF));
                            plankey_.push_back((uint8_t)((pr.first >> 8) & 0xFF));
                            plankey_.push_back((uint8_t)(pr.second & 0xFF));
                            plankey_.push_back((uint8_t)((pr.second >> 8) & 0xFF));
                        }
                        plan_out = plankey_.data();
                        plan_n = (int)plankey_.size();
                        }  // !skip_plankey
                    }
                } else {
                // Lazy Tier-B rows for this plan (channel-set token guarded; CW words/row).
                if (plan.ch_token != chtok) {
                    // First encounter in this Impl: build channel rows and insert into local index.
                    build_ch_rows(plan);
                    if (plan_local_key(plan, lkey_scratch)) lindex_insert(lkey_scratch, &plan);
                    if (prof) { auto cf = clk::now(); c_rows += std::chrono::duration<double>(cf - ck).count(); ck = cf; }
                } else {
                    // Rows already built (from shared cache or previous encounter in this run).
                    // Ensure the plan is in THIS Impl's local index (the plan may come from a shared
                    // cache that was populated in a previous rebuild — lindex_insert is idempotent).
                    if (plan_local_key(plan, lkey_scratch)) lindex_insert(lkey_scratch, &plan);
                    if (checked <= sc) {
                        // LOCAL INDEX ORACLE (selfcheck window): the bitmask index must resolve this
                        // shot's XOR key to exactly the plan the exact path produced.
                        for (int w2 = 0; w2 < KW; ++w2) ab_kw[w2] = 0;
                        for (int32_t e = e0; e < e1; ++e) {
                            const AltMasks& M = alt_masks[(size_t)batch.ev[e]];
                            for (int w2 = 0; w2 < NW; ++w2) ab_kw[w2]      ^= M.kmask[w2];
                            for (int w2 = 0; w2 < PW; ++w2) ab_kw[NW + w2] ^= M.czk[w2];
                        }
                        const CachedPlan* lp = lindex_find(ab_kw.data());
                        const CachedPlan* fp2 = ffind(ab_kw.data());
                        if (lp != &plan || fp2 != &plan) {
                            fprintf(stderr, "twirl_records LOCAL INDEX ORACLE FAIL shot=%ld\n", done);
                            return false;
                        }
                    }
                }
                if (plan.fallback || !nf.diagonal_class || !plan.kernel_logicals.empty()) {
                    // κ>0 / fallback: rare (0 in-distribution) — use the record path for the
                    // kernel Born chain, then read channel bits off σ (exact).
                    twirl_collapse_record(bare, G, nf, plan, coll_rng, o_rec);
                    if (o_rec.fallback || born_obs) {
                        // V3-T3 R1 per-shot exact fallback, two classes through one routine:
                        //   * o_rec.fallback (κ≥2 plan fallback / mid-chain guard trip) — the
                        //     record σ is unusable by contract;
                        //   * born κ>0 (spec guard: kernel outcomes correlate with W; the σ-
                        //     channel machinery has no obs law here — pre-R1 this emitted a
                        //     STALE obs bit).
                        // The record call above already consumed exactly the draws it always
                        // did, so downstream shots keep their byte streams; the exact shot
                        // (residual as gates → measure generators → measure W) runs on the
                        // dedicated fb rng and REPLACES this shot's bits/obs wholesale (bits
                        // and obs must come from ONE sample — mixing record-σ bits with an
                        // exact obs bit would break the (σ, W) joint).
                        ++fb;
                        exact_cur = true;
                        exact_diag_shot(nf, bits);
                    } else {
                        for (int c = 0; c < nchan; ++c)
                            if (par(o_rec.sigma, sch[c].mask)) bits[c >> 6] |= (1ull << (c & 63));
                    }
                } else {
                    // Self-check (first N diag shots): run the σ path on a FORKED rng (same
                    // upcoming draws), then require bit_c == parity(σ & mask_c) EXACTLY for
                    // every channel — the wiring oracle for the row machinery.
                    const bool do_check = checked < sc;
                    if (do_check) {
                        // Fork the per-shot ShotRng (copy its counter state) and use a
                        // BIT-RESERVOIR Rng so the coin sequence matches the fast path exactly:
                        // both extract raw low bits from the same splitmix64 step, identical
                        // to the reservoir path (no double-draw discrepancy).
                        ShotRng fork = shot_rng;
                        uint64_t fork_res = 0; int fork_left = 0;
                        Rng rng2 = [&]() -> double {
                            if (fork_left == 0) { fork_res = fork.next(); fork_left = 64; }
                            const bool b = fork_res & 1; fork_res >>= 1; --fork_left;
                            return b ? std::nextafter(1.0, 0.0) : 0.0;
                        };
                        twirl_collapse_record(bare, G, nf, plan, rng2, o_rec);
                        ++checked;
                    }
                    // Tier-B fast path: channel space only (CW words).
                    for (int w2 = 0; w2 < CW; ++w2) bits[w2] = plan.ch_baseline[w2];
                    const Pauli& P = nf.prefix;
                    for (size_t w = 0; w < P.x.size(); ++w) {
                        uint64_t word = P.x[w];
                        while (word) { const int q = (int)(w * 64 + __builtin_ctzll(word)); word &= word - 1;
                            const uint64_t* row = &pxrow[(size_t)q * CW];
                            for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    }
                    for (size_t w = 0; w < P.z.size(); ++w) {
                        uint64_t word = P.z[w];
                        while (word) { const int q = (int)(w * 64 + __builtin_ctzll(word)); word &= word - 1;
                            const uint64_t* row = &pzrow[(size_t)q * CW];
                            for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    }
                    bool guard_exact = false;              // V3-T3: obs_guard → per-shot exact
                    int obs_par = 0;
                    if (born_obs) { ensure_obs(plan); obs_par = plan.obs_par_base; }
                    for (size_t k = 0; k < plan.ch_coin_rows.size(); ++k) {
                        // Use the same 64-bit reservoir as the fast path so slow-path
                        // (self-check window / cold miss) and warm-path coins are
                        // drawn identically from shot_rng — routing-independent.
                        if (coin_left == 0) { coin_res = shot_rng.next(); coin_left = 64; }
                        const bool bit = coin_res & 1; coin_res >>= 1; --coin_left;
                        if (bit) {
                            for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= plan.ch_coin_rows[k][w2];
                            if (born_obs) obs_par ^= plan.obs_par_coin[k];
                        }
                    }
                    if (born_obs) {
                        if (plan.obs_guard) {
                            // classify guard (ANTI-class Λ candidate): no closed-form obs law.
                            // V3-T3 R1: the whole shot is recomputed exactly AFTER the self-
                            // check oracle below (the coin draws above already happened, so
                            // downstream byte streams are untouched; the row-machinery bits
                            // are discarded — bits and obs must come from ONE exact sample).
                            guard_exact = true;
                        } else {
                        int pw = 0;                        // ⟨P, W⟩ — the ONLY prefix correction
                        for (int w2 = 0; w2 < NW; ++w2)
                            pw += __builtin_popcountll(P.x[w2] & born_W.z[w2]) +
                                  __builtin_popcountll(P.z[w2] & born_W.x[w2]);
                        obs_par ^= (pw & 1);
                        const double val = plan.obs_m * (obs_par ? -1.0 : 1.0);
                        obs_bit_cur = (obs_u01() < (1.0 - val) / 2.0) ? 1 : 0;
                        }
                    }
                    if (do_check) {
                        for (int c = 0; c < nchan; ++c) {
                            const uint64_t want = (uint64_t)par(o_rec.sigma, sch[c].mask);
                            if (((bits[c >> 6] >> (c & 63)) & 1) != want) {
                                fprintf(stderr, "twirl_records SELF-CHECK FAIL shot=%ld ch=%d\n",
                                        done, c);
                                return false;
                            }
                        }
                        // ABELIAN ORACLE (bits): the per-alt-row + ordered-pair-T decomposition
                        // of the DETERMINISTIC channel bits must equal the nf-based emission
                        // (baseline ⊕ prefix-support rows ⊕ strip zflips) EXACTLY.
                        uint64_t abd[8], nfd[8];
                        for (int w2 = 0; w2 < CW; ++w2) { abd[w2] = plan.ch_baseline[w2]; nfd[w2] = plan.ch_baseline[w2]; }
                        for (int32_t e = e0; e < e1; ++e) {
                            const AltMasks& M = alt_masks[(size_t)batch.ev[e]];
                            for (int w2 = 0; w2 < CW; ++w2) abd[w2] ^= M.zxrow[w2] ^ M.zzrow[w2];
                        }
                        for (int32_t ej = e0 + 1; ej < e1; ++ej) {
                            const AltMasks& Mj = alt_masks[(size_t)batch.ev[ej]];
                            const AltAtom& Aj = alt_atoms[(size_t)batch.ev[ej]];
                            for (int32_t ek = e0; ek < ej; ++ek) {
                                const AltMasks& Mk = alt_masks[(size_t)batch.ev[ek]];
                                for (int w2 = 0; w2 < NW; ++w2)
                                    ab_T[w2] = (Mj.smask[w2] & Mk.smask[w2]) ^ (Mj.smask[w2] & Mk.vmask[w2]);
                                for (size_t i = 0; i < Aj.czp.size(); ++i) {
                                    const int x = Aj.czp[i].first, y = Aj.czp[i].second;
                                    if ((Mk.vmask[y >> 6] >> (y & 63)) & 1) ab_T[x >> 6] ^= (1ULL << (x & 63));
                                    if ((Mk.vmask[x >> 6] >> (x & 63)) & 1) ab_T[y >> 6] ^= (1ULL << (y & 63));
                                }
                                for (int w2 = 0; w2 < NW; ++w2) { uint64_t word = ab_T[w2];
                                    while (word) { const int q = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                        const uint64_t* row = &pzrow[(size_t)q * CW];
                                        for (int w3 = 0; w3 < CW; ++w3) abd[w3] ^= row[w3]; } }
                            }
                        }
                        for (size_t w4 = 0; w4 < nf.prefix.x.size(); ++w4) {
                            uint64_t word = nf.prefix.x[w4];
                            while (word) { const int q = (int)(w4 * 64 + __builtin_ctzll(word)); word &= word - 1;
                                const uint64_t* row = &pxrow[(size_t)q * CW];
                                for (int w3 = 0; w3 < CW; ++w3) nfd[w3] ^= row[w3]; }
                        }
                        for (size_t w4 = 0; w4 < nf.prefix.z.size(); ++w4) {
                            uint64_t word = nf.prefix.z[w4];
                            while (word) { const int q = (int)(w4 * 64 + __builtin_ctzll(word)); word &= word - 1;
                                const uint64_t* row = &pzrow[(size_t)q * CW];
                                for (int w3 = 0; w3 < CW; ++w3) nfd[w3] ^= row[w3]; }
                        }
                        // (No zflip rows here: nf is POST-canon, so the strip's Z-flips are
                        // already inside nf.prefix.z; the abelian side reproduces them via the
                        // per-alt zzrow stripped-pair contributions.)
                        for (int w2 = 0; w2 < CW; ++w2)
                            if (abd[w2] != nfd[w2]) {
                                fprintf(stderr, "twirl_records ABELIAN BITS ORACLE FAIL shot=%ld\n", done);
                                return false;
                            }
                        // LINEARIZATION ORACLE: the fast path's O(fired) running-accumulator pair
                        // term must reproduce the O(fired²) T-emission above EXACTLY.
                        uint64_t lin[8];
                        for (int w2 = 0; w2 < CW; ++w2) { lin[w2] = plan.ch_baseline[w2]; }
                        for (int32_t e = e0; e < e1; ++e) {
                            const AltMasks& M = alt_masks[(size_t)batch.ev[e]];
                            for (int w2 = 0; w2 < CW; ++w2) lin[w2] ^= M.zxrow[w2] ^ M.zzrow[w2];
                        }
                        if (e1 - e0 > 1) {
                            for (int w2 = 0; w2 < NW; ++w2) { ab_S[w2] = 0; ab_V[w2] = 0; ab_T[w2] = 0; }
                            for (int32_t ej = e0; ej < e1; ++ej) {
                                const AltMasks& Mj = alt_masks[(size_t)batch.ev[ej]];
                                if (ej > e0) {
                                    const AltAtom& Aj2 = alt_atoms[(size_t)batch.ev[ej]];
                                    for (int w2 = 0; w2 < NW; ++w2)
                                        ab_T[w2] ^= Mj.smask[w2] & (ab_S[w2] ^ ab_V[w2]);
                                    for (size_t i = 0; i < Aj2.czp.size(); ++i) {
                                        const int x = Aj2.czp[i].first, y = Aj2.czp[i].second;
                                        if ((ab_V[y >> 6] >> (y & 63)) & 1) ab_T[x >> 6] ^= (1ULL << (x & 63));
                                        if ((ab_V[x >> 6] >> (x & 63)) & 1) ab_T[y >> 6] ^= (1ULL << (y & 63));
                                    }
                                }
                                for (int w2 = 0; w2 < NW; ++w2) {
                                    ab_S[w2] ^= Mj.smask[w2]; ab_V[w2] ^= Mj.vmask[w2];
                                }
                            }
                            for (int w2 = 0; w2 < NW; ++w2) { uint64_t word = ab_T[w2];
                                while (word) { const int q = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                    const uint64_t* row = &pzrow[(size_t)q * CW];
                                    for (int w3 = 0; w3 < CW; ++w3) lin[w3] ^= row[w3]; } }
                        }
                        for (int w2 = 0; w2 < CW; ++w2)
                            if (lin[w2] != abd[w2]) {
                                fprintf(stderr, "twirl_records LINEARIZATION ORACLE FAIL shot=%ld\n", done);
                                return false;
                            }
                    }
                    // V3-T3 R1: born obs_guard — exact recomputation of the WHOLE shot (bits
                    // + obs bit from one sample on the dedicated fb rng). Runs after the
                    // self-check oracle so the row-machinery wiring stays fully checked.
                    if (guard_exact) {
                        ++fb;
                        exact_cur = true;
                        exact_diag_shot(nf, bits);
                    }
                }
                }  // S2.2: close the `if (retain_state) {…} else {…}` slow-path guard
            }
            apply_rf(bits);
            apply_input_frame(bits);
            ++used;
            if (count_channels_) {
                for (int w2 = 0; w2 < CW; ++w2) {
                    uint64_t word = bits[w2];
                    while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                        ++chancnt[c]; }
                }
            }
            if (born_obs) {
                obs_ones_ += obs_bit_cur;
                // Acceptance = every deterministic detector at its NOISELESS value. The raw
                // twirl channel bit is parity(sigma & mask): noiselessly sigma = 0 => raw = 0
                // (the ref-XOR happens at REPORTING; final-review Important #1: XORing det_refw
                // here selected the complementary sector on ref=1 detectors — prod accepts
                // pbit == ref, which corresponds to raw == 0, not raw == ref).
                bool acc = true;
                for (int w2 = 0; w2 < CW; ++w2)
                    if (bits[w2] & det_maskw[w2]) { acc = false; break; }
                if (acc) { ++ps_acc_; ps_obs_ones_ += obs_bit_cur; }
            }
            // Task 2b.1: Born-measure the declared decisions on the retained collapsed state
            // (in declaration order) AFTER the residual/channel bits are final, then hand the
            // post-decision state to the sink — so the retained amps and the decision bits come
            // from the SAME Born draw (the correlation gate). dec_state_ is the mutable target.
            const uint8_t* dptr = emit_decisions(bits, dec_state_);
            if (has_born_dec_ && dec_state_) amps_out = dec_state_;
            // decoder-feedback perf: append the RAW born outcomes to the coin record so the
            // per-shot record-hash KEY (σ, coins=[r fair ‖ κ chain ‖ born], plan) distinguishes
            // shots that collapse to DIFFERENT states (the born outcome selects the OUTPUT patch's
            // eigenstate). materialize() uses coins[0:r+κ] for the twirl replay and the separate
            // born_u_ coins to RE-MEASURE the born operators (genuine, not forced) — so the full
            // state need NOT be materialised per shot. Only appended on the born path (born_raw_
            // empty on fast/decision-free shots); the null-sink loop is byte-identical.
            for (uint8_t b : born_raw_) o_rec.coins.push_back(b);
            emit_shot(bits, obs_bit_cur, exact_cur, amps_out, sig_out, dptr,
                          o_rec.coins.empty() ? nullptr : o_rec.coins.data(),
                          (int)o_rec.coins.size(), plan_out, plan_n,
                          born_u_.empty() ? nullptr : born_u_.data(), (int)born_u_.size(),
                          plan_ptr_out, pre_x_out, pre_z_out);
        }
    }
    auto t1 = clk::now();
    wall += std::chrono::duration<double>(t1 - t0).count();
    if (!disk_path.empty()) {
        if (cache->save_file(disk_path, G.identity_token()) && !xtim_quiet())
            fprintf(stderr, "[twirl_records] disk '%s': SAVED (%zu plans)\n", disk_path.c_str(), cache->size());
    }
    if (prof) fprintf(stderr, "[twirl_records PROF] key=%.3f find=%.3f emit=%.3f pair=%.3f coins=%.3f rf=%.3f post=%.3f sink=%.3f draw=%.3f prolog=%.3f | nrf=%d probe=%.1fns wall=%.3f us/shot\n",
                      p_key * 1e6 / shots, p_find * 1e6 / shots, p_emit * 1e6 / shots, p_pair * 1e6 / shots,
                      p_coins * 1e6 / shots, p_rf * 1e6 / shots, p_post * 1e6 / shots, p_sink * 1e6 / shots,
                      t_draw * 1e6 / shots, p_prolog * 1e6 / shots, nrf, probe_ns,
                      wall * 1e6 / shots);
    // plan_misses counts GENUINE LAW BUILDS (cache->get_or_build called build_shot_law).
    // Shared-cache hits via the canonical-key path count as HITs, not misses. After the
    // shared fast index fix, warm-pass slow_shots collapse to ~selfcheck shots (oracle window
    // only); law_build per-miss is the true plan-derive cost, not a shared-cache copy cost.
    if (prof && slow_shots)
        fprintf(stderr, "[twirl_records PROF cold] slow_shots=%ld law_misses=%zu | per-SLOW-shot: "
                        "materialize=%.1f canon=%.1f law_build=%.1f rows+index=%.1f us | "
                        "per-LAW-MISS law_build=%.1f us\n",
                slow_shots, cache->misses(),
                c_mat * 1e6 / slow_shots, c_canon * 1e6 / slow_shots,
                c_build * 1e6 / slow_shots, c_rows * 1e6 / slow_shots,
                cache->misses() ? c_build * 1e6 / cache->misses() : 0.0);
    return true;
}

// ── Public surface ────────────────────────────────────────────────────────────────────────
TwirlRecordSampler::TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                                       const std::vector<std::pair<int, int>>& reads,
                                       const std::vector<std::vector<int>>& detectors,
                                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                                       uint64_t seed, const TwirlRecordOptions& opt)
    : impl_(new Impl(bare, deferred, reads, detectors, observables, seed, opt)) {}

TwirlRecordSampler::TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                                       const std::vector<std::pair<int, int>>& reads,
                                       const std::vector<std::vector<int>>& detectors,
                                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                                       uint64_t seed, const TwirlRecordOptions& opt,
                                       const SharedPlanCaches& sc)
    : impl_(new Impl(bare, deferred, reads, detectors, observables, seed, opt, sc)) {}

TwirlRecordSampler::~TwirlRecordSampler() = default;

int TwirlRecordSampler::setup_error() const { return impl_->setup_error; }
bool TwirlRecordSampler::run(long shots) { return impl_->run(shots); }
bool TwirlRecordSampler::set_seed(uint64_t seed) { return impl_->set_seed(seed); }
void TwirlRecordSampler::set_barrier_sink(BarrierSink* sink) { impl_->barrier_sink_ = sink; }
void TwirlRecordSampler::set_dets_pack_sink(DetsPackSink* sink) { impl_->dets_pack_sink_ = sink; }
void TwirlRecordSampler::set_retain_state(bool on) { impl_->retain_state = on; }
void TwirlRecordSampler::set_input_frame(const uint8_t* data, int stride) {
    if (!impl_) return;
    impl_->frame_data_   = data;
    impl_->frame_stride_ = stride;
}
FramedSuperposition TwirlRecordSampler::materialize_shot(const uint8_t* plan_key, int n_plan,
                                                         const uint8_t* coins, int n_coins,
                                                         const double* born_u, int n_born) const {
    return impl_->materialize_shot(plan_key, n_plan, coins, n_coins, born_u, n_born);
}
TwirlRecordSampler::PlanStructure TwirlRecordSampler::plan_structure(const uint8_t* plan_key,
                                                                     int n_plan) const {
    PlanStructure out;
    out.nf = impl_->parse_plan_key(plan_key, n_plan);
    out.law = build_shot_law(impl_->G, out.nf);   // the same call materialize_shot's step 2 makes
    return out;
}
// T5 step 0 (exact-residual arc): pure-data forwarding of the Impl's born_dec vector — the
// Born-measured decision operators + record inverts, declaration order (see the header doc).
std::vector<TwirlRecordSampler::BornDecOp> TwirlRecordSampler::born_dec_ops() const {
    std::vector<BornDecOp> out;
    out.reserve(impl_->born_dec.size());
    for (size_t j = 0; j < impl_->born_dec.size(); ++j)
        out.push_back(BornDecOp{impl_->born_dec[j].first, impl_->born_dec[j].second,
                                impl_->born_dec_inv[j]});
    return out;
}

long TwirlRecordSampler::used() const { return impl_->used; }
int TwirlRecordSampler::born_obs_index() const { return impl_->born_idx; }
long TwirlRecordSampler::obs_ones() const { return impl_->obs_ones_; }
long TwirlRecordSampler::ps_accepted() const { return impl_->ps_acc_; }
long TwirlRecordSampler::ps_obs_ones() const { return impl_->ps_obs_ones_; }
long TwirlRecordSampler::fallbacks() const { return impl_->fb; }
long TwirlRecordSampler::exact_shots() const { return impl_->fb; }
const std::string& TwirlRecordSampler::disk_path() const { return impl_->disk_path; }
long TwirlRecordSampler::clean_shots() const { return impl_->clean; }
long TwirlRecordSampler::diag_shots() const { return impl_->diag; }
long TwirlRecordSampler::ppr_shots() const { return impl_->pprshots; }
size_t TwirlRecordSampler::ppr_plans() const { return impl_->pcache->size(); }
int TwirlRecordSampler::deferred_wires() const { return impl_->n; }
int TwirlRecordSampler::num_channels() const { return impl_->nchan; }
int TwirlRecordSampler::channel_words() const { return impl_->CW; }
int TwirlRecordSampler::num_decisions() const { return impl_->num_dec_; }
size_t TwirlRecordSampler::plan_hits() const { return impl_->cache->hits(); }
size_t TwirlRecordSampler::plan_misses() const { return impl_->cache->misses(); }
size_t TwirlRecordSampler::plans() const { return impl_->cache->size(); }
const std::vector<long long>& TwirlRecordSampler::channel_one_counts() const { return impl_->chancnt; }
double TwirlRecordSampler::draw_seconds() const { return impl_->t_draw; }
double TwirlRecordSampler::wall_seconds() const { return impl_->wall; }
const std::vector<int>& TwirlRecordSampler::channel_detectors() const { return impl_->chan_det; }
const std::vector<int>& TwirlRecordSampler::channel_observables() const { return impl_->chan_obs; }
const std::vector<uint8_t>& TwirlRecordSampler::channel_refs() const { return impl_->chan_ref; }
const std::vector<int>& TwirlRecordSampler::gauge_detectors() const { return impl_->gauge_dets; }
const std::vector<int>& TwirlRecordSampler::anti_detectors() const { return impl_->anti_dets; }
const std::vector<int>& TwirlRecordSampler::refused_observables() const { return impl_->refused_obs; }
uint64_t TwirlRecordSampler::group_token() const { return impl_->G.identity_token(); }

}  // namespace qeccore
