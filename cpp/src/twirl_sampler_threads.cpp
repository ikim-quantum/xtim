// twirl_sampler_threads — threads (fixplan threads/fixplan.md, T2): TwirlRecordSampler::run(shots, threads).
//
// The home sampler draws a super-block of noise events (lean N1: the run's table's aligned 4096-shot noise blocks that
// cover it — each a pure function of (seed, block, table), the same events a serial call draws) and its lanes take
// disjoint blocks of it. A lane is a copy of the home
// Impl: the construction state (Impl::Frozen) and the caller-owned bare state are shared, every per-shot member is its
// own. Every byte a shot emits is addressed by (seed, cumulative shot index) or is a pure function of plan content, so
// the shot at global index i gives the serial bytes, whichever lane runs it; its row lands at its global row.
// Scaling S1 / lean N1: the NEXT super-block's noise is generated while the lanes run the current one, by generation
// tasks (one per whole noise block, laid out shot-major: per shot ascending alternative, the serial event list's order)
// that the pool's jobs pull before their blocks; a lane gathers its block's shots when it takes it.
//
// Plans during a super-block: the home's cache and its plans are READ ONLY (no lock, no atomic on the shot path). A lane
// serves a resident home plan, else builds into its own overlay cache (TwirlPlanCache::get_or_build, the same build); a
// Born-trie node the shared trie lacks is evaluated on the lane's copy of that trie (lane_walk_). At the barrier the
// home adopts the overlays in lane order (TwirlPlanCache::adopt_from), merges the copied tries (born_trie_merge: node
// values are path-determined, a disagreement is an engine-bug throw), adds the lanes' counters, then runs the payload
// budget boundary. The selfcheck window's shots run serially first (run_w's window_stop_ prefix); an oracle_all_shots
// sampler runs serially (owner ruling D2). The effective thread count is capped to what the state budget holds
// (owner ruling D3: no new refusal) and reported (record_report).
#include "twirl_sampler_impl.inc.hpp"

#include <cfenv>
#include <climits>
#include <exception>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/state_budget.hpp"
#include "qeccore/thread_pool.hpp"

namespace qeccore {

void (*g_twirl_gen_task_test_hook)(int) = nullptr;

extern template bool TwirlRecordSampler::Impl::run_w<8 | TwirlRecordSampler::Impl::kLaneRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<0 | TwirlRecordSampler::Impl::kLaneRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<8 | TwirlRecordSampler::Impl::kErRun |
                                                     TwirlRecordSampler::Impl::kLaneRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<0 | TwirlRecordSampler::Impl::kErRun |
                                                     TwirlRecordSampler::Impl::kLaneRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<8 | TwirlRecordSampler::Impl::kDeferRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<0 | TwirlRecordSampler::Impl::kDeferRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<8 | TwirlRecordSampler::Impl::kErRun |
                                                     TwirlRecordSampler::Impl::kDeferRun>(long);
extern template bool TwirlRecordSampler::Impl::run_w<0 | TwirlRecordSampler::Impl::kErRun |
                                                     TwirlRecordSampler::Impl::kDeferRun>(long);

struct TwirlRecordSampler::Impl::LaneSet {
    std::vector<std::unique_ptr<Impl>> lanes;
    std::vector<std::unique_ptr<LaneCtx>> ctx;
    std::vector<std::shared_ptr<TwirlPlanCache>> overlay;
    NoiseSpan span[2];                                // the super-block's noise blocks, and the next one's (S1; lean N1)
    std::vector<size_t> gtask[2];                     // per buffer: the blocks its generation tasks fill (indices in blk)
    long B = 0;                                       // scaling S3: the block size the ramp reached (kept across calls)
    std::vector<uint8_t> acc;                         // per super-block shot: its row's state (kAcc*)
    std::vector<long> dlist;                          // the super-block's deferred shots, in shot order
    std::vector<long> rlist;                          // fid-routine B4: the lanes' premap leftovers, in shot order
    LaneCtx dctx;                                     // the home's block source for them
};

// A layout-3 record row moved (dst < src) / cleared (as a fresh buffer holds it: handle -1, the rest 0).
void TwirlRecordSampler::Impl::record_row_move_(TwirlRecordSampler::FamilyRecordSink& R, long dst, long src) {
    const size_t P2 = 2 * (size_t)R.PNW;
    R.handle[dst] = R.handle[src];
    R.route[dst] = R.route[src];
    std::memcpy(R.prec + (size_t)dst * P2, R.prec + (size_t)src * P2, P2 * sizeof(uint64_t));
    std::memcpy(R.y + (size_t)dst * (size_t)R.YW, R.y + (size_t)src * (size_t)R.YW, (size_t)R.YW * sizeof(uint64_t));
    std::memcpy(R.uz + (size_t)dst * (size_t)R.UZW, R.uz + (size_t)src * (size_t)R.UZW, (size_t)R.UZW);
    R.nuz[dst] = R.nuz[src];
    R.ff[dst] = R.ff[src];
    if (R.branch) R.branch[dst] = R.branch[src];
}
void TwirlRecordSampler::Impl::record_row_clear_(TwirlRecordSampler::FamilyRecordSink& R, long r) {
    const size_t P2 = 2 * (size_t)R.PNW;
    R.handle[r] = -1;
    R.route[r] = 0;
    std::memset(R.prec + (size_t)r * P2, 0, P2 * sizeof(uint64_t));
    std::memset(R.y + (size_t)r * (size_t)R.YW, 0, (size_t)R.YW * sizeof(uint64_t));
    std::memset(R.uz + (size_t)r * (size_t)R.UZW, 0, (size_t)R.UZW);
    R.nuz[r] = 0;
    R.ff[r] = 0;
    if (R.branch) R.branch[r] = 0;
}

TwirlRecordSampler::Impl::LaneSetHold::~LaneSetHold() { delete p; }

namespace {
// scaling S3: a super-block = bpl·T blocks of B shots, pulled dynamically. The block size B ramps 64 -> 1024 over a
// cold sampler's first super-blocks (fewer plans first met by two lanes at once) and is kept across calls (the ramp is
// about the plan cache, not the call); bpl ramps 4 -> 16 within every call (a call's first super-block is small: its
// noise draw is the one not overlapped with lane work), so a lane runs up to 16 blocks per barrier (a short barrier
// tail). A super-block that would leave a remainder below half its own size takes the remainder too.
constexpr long kBlockMax = 1024;
constexpr long kBlockMin = 64;
constexpr int kBplMin = 4;
constexpr int kBplMax = 16;
// lean N1: noise block b of table T laid out shot-major (per shot ascending alternative)
template <class BlockEv>
void gen_block_ev_(const NoiseModel& N, const NoiseTable& T, uint64_t seed, BlockEv& B) {
    std::vector<uint64_t> ev;
    std::vector<int32_t> cnt;
    DiagErrorSampler::SparseBatch sb;
    N.block_events(T, seed, B.b, ev);
    NoiseModel::shot_major(ev, 0, NoiseModel::kB, sb, cnt);
    B.off = std::move(sb.shot_off);
    B.ev = std::move(sb.ev);
}
}  // namespace

void TwirlRecordSampler::Impl::NoiseSpan::gather(uint64_t c, int S, DiagErrorSampler::SparseBatch& out) const {
    while (S > 0) {
        const NoiseBlockEv& B = *blk[(size_t)((c >> NoiseModel::kBShift) - b0)];
        const int lo = (int)(c & (NoiseModel::kB - 1)), n = std::min(S, NoiseModel::kB - lo);
        const int32_t e0 = B.off[(size_t)lo], base = (int32_t)out.ev.size();
        out.ev.insert(out.ev.end(), B.ev.begin() + e0, B.ev.begin() + B.off[(size_t)(lo + n)]);
        for (int i = 1; i <= n; ++i) out.shot_off.push_back(base + B.off[(size_t)(lo + i)] - e0);
        out.S += n;
        c += (uint64_t)n;
        S -= n;
    }
}

// A block source's next block (a lane: the next block of the super-block; the home's deferred run: the next deferred
// shot): its slice of the super-block's events (offsets rebased), its first shot's cumulative index (what the keyed
// streams address) and index in the call. The rows are positional (lane_shot_begin_).
bool TwirlRecordSampler::Impl::lane_take_block_(int& S, long& done) {
    LaneCtx& L = *blk_;
    if (L.pend) lane_shot_end_();
    if (L.stop && L.stop->load(std::memory_order_relaxed)) return false;
    long s0;
    const long k = L.next->fetch_add(1, std::memory_order_relaxed);
    if (L.defer_list) {
        if (k >= (long)L.defer_list->size()) return false;
        s0 = (*L.defer_list)[(size_t)k];
        S = 1;
    } else {
        if (k >= L.nblocks) return false;
        s0 = k * L.B;
        S = (int)std::min(L.B, L.S_sb - s0);
        L.blocks.push_back(k);
        lane_heap_headroom();                         // S2: the block's heap headroom (thread_pool.hpp)
    }
    batch.S = 0;                                      // lean N1: the block's shots gathered from the span's noise blocks
    batch.shot_off.assign(1, 0);
    batch.ev.clear();
    L.span->gather(L.shot0 + (uint64_t)s0, S, batch);
    done = L.done0 + s0;
    done_shots_ = L.shot0 + (uint64_t)s0;
    L.s0_cur = s0;
    L.cur_block = s0 / L.B;
    L.ev_marks.emplace_back(L.cur_block, diag_.size());
    return true;
}

namespace {
// The per-shot counters a lane accumulates in one super-block, added into the home's and zeroed in the lane
// (home == nullptr: zeroed only — a new lane, a copy of its home, starts from zero).
template <class T>
void mv_(T* h, T& l) {
    if (h) *h += l;
    l = 0;
}
template <class I>
void lane_counters_into(I* home, I& lane) {
#define XTIM_MV_(f) mv_(home ? &home->f : nullptr, lane.f)
    XTIM_MV_(clean); XTIM_MV_(diag); XTIM_MV_(checked); XTIM_MV_(used);
    for (int r = 0; r < I::R_COUNT; ++r) XTIM_MV_(route_[r]);
    XTIM_MV_(fast_derived_); XTIM_MV_(structural_shots_); XTIM_MV_(fast_born_shots_);
    XTIM_MV_(obs_ones_); XTIM_MV_(ps_acc_); XTIM_MV_(ps_obs_ones_);
    XTIM_MV_(sem_checked_);
    for (int k = 0; k < 6; ++k) XTIM_MV_(sem_class_[k]);
    for (size_t c = 0; c < lane.chancnt.size(); ++c) mv_(home && c < home->chancnt.size() ? &home->chancnt[c] : nullptr, lane.chancnt[c]);
    XTIM_MV_(trie_.walks); XTIM_MV_(trie_.steps); XTIM_MV_(trie_.nodes); XTIM_MV_(trie_.node_evals);
    XTIM_MV_(trie_.chi_j); XTIM_MV_(trie_.passes); XTIM_MV_(trie_.root_memo_hits);
    if (home) {
        home->trie_.max_support = std::max(home->trie_.max_support, lane.trie_.max_support);
        home->trie_.max_depth = std::max(home->trie_.max_depth, lane.trie_.max_depth);
    }
    lane.trie_.max_support = 0;
    lane.trie_.max_depth = 0;
    XTIM_MV_(rec_tables_); XTIM_MV_(rec_extras_); XTIM_MV_(rec_ops_plane_); XTIM_MV_(rec_ops_pivot_);
    XTIM_MV_(rec_ops_greduce_); XTIM_MV_(rec_ops_echelon_); XTIM_MV_(rec_pivots_extras_); XTIM_MV_(rec_dirs_extras_);
    XTIM_MV_(obs_draws_record_);
#undef XTIM_MV_
    if (home) {
        if (home->rec_kappa_hist_.size() < lane.rec_kappa_hist_.size())
            home->rec_kappa_hist_.resize(lane.rec_kappa_hist_.size(), 0);
        for (size_t k = 0; k < lane.rec_kappa_hist_.size(); ++k) home->rec_kappa_hist_[k] += lane.rec_kappa_hist_[k];
    }
    lane.rec_kappa_hist_.clear();
}
}  // namespace

bool TwirlRecordSampler::Impl::run_threads_(long shots, int threads) {
    threads_req_ = threads < 1 ? 1 : threads;
    threads_eff_ = 1;
    threads_deferred_ = 0;
    threads_budget_fallbacks_ = 0;
    threads_limit_.clear();
    if (threads <= 1 || shots <= 0) return run(shots);
    if (const char* why = lanes_unsupported_()) {
        threads_limit_ = why;
        return run(shots);
    }
    // D3: every lane needs its own chi-scaled Born register (and its builds' scratch): cap the lanes to what the state
    // budget holds (an estimate: 4 registers of the bare's chi per lane) — never a new refusal.
    int Te = threads;
    {
        const int64_t per = lane_bytes_test_ > 0 ? lane_bytes_test_
                          : 4 * (int64_t)bare.chi() * (int64_t)(sizeof(AmpKey) + sizeof(std::complex<double>)) +
                                (int64_t(64) << 10);
        const int64_t room = state_budget_bytes() - live_state_bytes();
        const int64_t fit = room > 0 ? room / per : 0;
        if (fit < Te) {
            Te = (int)std::max<int64_t>(1, fit);
            threads_limit_ = "the state budget holds " + std::to_string(Te) + " lane(s) at chi = " +
                             std::to_string(bare.chi());
        }
    }
    {   // fix F4: the pool's workers (a thread the OS refuses caps the fan-out: reported, never thrown)
        const int w = ThreadPool::instance().reserve(Te);
        if (w + 1 < Te) {
            Te = w + 1;
            threads_limit_ = "the OS gave the worker pool " + std::to_string(w) + " thread(s)";
        }
    }
    if (Te <= 1) return run(shots);
    // run_w's per-call refusals, before any effect (a lane checks per block only what its home checked here)
    {
        const uint64_t per_shot = std::max<uint64_t>({(uint64_t)nrf, (uint64_t)ng, (uint64_t)born_dec.size()});
        if (per_shot >= krng::kMaxCtr)
            throw std::invalid_argument("twirl sampler: a shot needs " + std::to_string(per_shot) +
                " draws of one keyed stream (noisy records / certified generators / Born decisions / "
                "gauge detectors); the keyed stream layout allows < 2^20 per shot");
        if (done_shots_ >= krng::kMaxShots || (uint64_t)shots > krng::kMaxShots - done_shots_)
            throw std::invalid_argument("twirl sampler: the cumulative shot index of this seed would "
                "reach 2^44 (keyed streams address shot*2^20 + draw); pass a new seed to continue");
    }
    long done = 0;
    // The selfcheck window (the first non-empty call of the sampler's life) runs serially: the same shots are checked.
    // That call counts its channels (count_channels_) on every shot, so its lanes do too.
    const bool window = !selfcheck_spent_ && selfcheck > 0;
    bool born_event = false;
    if (window) {
        window_stop_ = true;
        struct Off { bool& f; ~Off() { f = false; } } off{window_stop_};
        const size_t ev0 = diag_.size();
        if (!run(shots)) return false;
        done = last_run_done_;
        for (size_t e = ev0; e < diag_.size(); ++e)
            if (diag_[e].kind == TwirlRecordSampler::DiagEvent::BORN_NODE) born_event = true;
        if (done >= shots) {
            threads_limit_ = "the selfcheck window covered the call (serial)";
            return true;
        }
    }
    const bool count_ch = window;
    const auto t0 = clk::now();
    // the lanes (made on the first threaded call, kept): copies of this sampler after freeze_
    freeze_();
    if (!lanes_.p) lanes_.p = new LaneSet();
    LaneSet& LS = *lanes_.p;
    bool serial_rest = false;                         // fix F2: the rest of the call runs on the home
    while ((int)LS.lanes.size() < Te) {
        std::unique_ptr<Impl> L;
        try {
            L.reset(new Impl(*this));
        } catch (const BareStateCapacityError&) {     // fix F2: as many lanes as the state budget holds
            Te = (int)LS.lanes.size();
            threads_limit_ = "the state budget holds " + std::to_string(Te) + " lane copies";
            break;
        }
        auto ctx = std::make_unique<LaneCtx>();
        auto ov = std::make_shared<TwirlPlanCache>();
        ov->set_universe(n, G.identity_token(), /*reserve_identity=*/false);
        ov->set_classifier(fz_->classifier);      // plancache-design: the same Π; the home's classes read-only
        ov->set_overlay_of(cache.get());
        L->cache = ov;
        L->lane_ = ctx.get();
        L->rec_memo_.clear();                         // the readers' memo is the home's (a lane never reads it)
        L->reader_bytes_.store(0);
        L->diag_.clear();
        L->batch = DiagErrorSampler::SparseBatch{};
        fam_lane_route_(*L);
        rot_lane_setup_(*L);                          // lean D1-S2: the lane's own builder and overlay store
        lane_counters_into<Impl>(nullptr, *L);       // the copy's counters are the home's: start from zero
        ctx->shared = cache.get();
        LS.lanes.push_back(std::move(L));
        LS.ctx.push_back(std::move(ctx));
        LS.overlay.push_back(std::move(ov));
    }
    if (Te <= 1) serial_rest = true;
    // per call: the run's per-call setup (run_w's prologue: the record sink's shape, the term route's scratch, the
    // record_rows_ variant), then the lanes' streams keyed on this call's seed, the home's sinks, the mask
    if (!fam_run_guard_(shots - done)) return false;
    TwirlRecordSampler::FamilyRecordSink* const rs = fam_record_sink_();
    if (rs) ensure_rot_plists_();                    // lean D1-S2 / D1-S3: the lanes' record writers read the P supports
    for (int i = 0; i < Te; ++i) {
        Impl& L = *LS.lanes[(size_t)i];
        LaneCtx& C = *LS.ctx[(size_t)i];
        L.set_keys_(seed_);
        if (family_sink_) { C.sink = *family_sink_; L.family_sink_ = &C.sink; }
        else L.family_sink_ = nullptr;
        fam_lane_call_(L, C);                        // the recording run's lane sink and content table (T4)
        L.count_channels_ = count_ch;
        C.count_channels = count_ch;
        L.born_ref_ = born_ref_;
        L.in_window_ = false;
        L.blk_ = &C;
        // the call's postselection mask: the lane's own copy of the armed state (its scratch, its counters)
        delete L.er_.memo;
        L.er_.memo = er_.p ? new ErState(*er_.p) : nullptr;
        L.er_.p = L.er_.memo;
        if (L.er_.p) { L.er_.p->early = L.er_.p->late = 0; L.er_.p->rejected = false; }
    }
    std::atomic<long> next{0};
    std::atomic<bool> stop{false};
    std::fenv_t fe;
    std::fegetenv(&fe);
    using RunFn = bool (Impl::*)(long);
    const RunFn lane_fn = er_.p ? (CW <= 8 ? &Impl::run_w<8 | kErRun | kLaneRun> : &Impl::run_w<0 | kErRun | kLaneRun>)
                                : (CW <= 8 ? &Impl::run_w<8 | kLaneRun> : &Impl::run_w<0 | kLaneRun>);
    const RunFn defer_fn = er_.p ? (CW <= 8 ? &Impl::run_w<8 | kErRun | kDeferRun> : &Impl::run_w<0 | kErRun | kDeferRun>)
                                 : (CW <= 8 ? &Impl::run_w<8 | kDeferRun> : &Impl::run_w<0 | kDeferRun>);
    const bool budget_on = plan_budget_ != 0;
    std::vector<char> ok((size_t)Te, 1);
    std::vector<std::exception_ptr> err((size_t)Te);
    std::vector<int> owner;                           // block -> the lane that ran it (the record remap)
    int cur = 0;                                      // S1: LS.span[cur] = the current super-block's noise
    // The home runs `list` (super-block shot indices, ascending) itself: each shot at its global index and row.
    auto home_run = [&](const std::vector<long>& list, long nb, long B, long S_sb, uint64_t shot0, long done0,
                        long row0) -> bool {
        LaneCtx& D = LS.dctx;
        std::atomic<long> dnext{0};
        D.span = &LS.span[cur]; D.next = &dnext; D.stop = nullptr;
        D.nblocks = nb; D.B = B; D.S_sb = S_sb;
        D.shot0 = shot0; D.done0 = done0; D.row0 = row0;
        D.acc = LS.acc.data(); D.mark = kAccHome; D.pend = false;
        D.cur_block = -1;
        D.blocks.clear(); D.deferred.clear(); D.ev_marks.clear();
        D.defer_list = &list;
        D.rs = rs;
        D.count_channels = count_ch;
        born_node_event_ = born_event;
        blk_ = &D;
        struct Unset { LaneCtx*& b; ~Unset() { b = nullptr; } } unset{blk_};
        const bool r = (this->*defer_fn)((long)list.size());
        born_event = born_event || born_node_event_;
        return r;
    };
    // The rows a super-block's lanes wrote, cleared (a re-run writes them again into zeroed rows).
    auto clear_rows = [&](long r0, long n) {
        if (FamilySink* fs = family_sink_) {
            if (fs->DBB) std::memset(fs->dets + (size_t)r0 * fs->DBB, 0, (size_t)n * fs->DBB);
            if (fs->OBB) std::memset(fs->obs + (size_t)r0 * fs->OBB, 0, (size_t)n * fs->OBB);
            if (fs->DECB) std::memset(fs->dec + (size_t)r0 * fs->DECB, 0, (size_t)n * fs->DECB);
        }
        if (rs) for (long r = r0; r < r0 + n; ++r) record_row_clear_(*rs, r);
    };
    if (LS.B < kBlockMin) LS.B = kBlockMin;
    long B = LS.B;
    int bpl = kBplMin;
    const uint64_t call_c0 = done_shots_ - (uint64_t)done;   // the call's first cumulative shot index
    // the super-block starting at `at` with block size B_ and bpl_ blocks per lane
    auto sb_size = [&](long at, long B_, int bpl_) {
        const long rest = shots - at;
        long S = std::min<long>(rest, (long)bpl_ * (serial_rest ? 1 : Te) * B_);
        if (rest - S < S / 2) S = rest;               // S3: a short last super-block folds into this one
        if (S < rest) {                               // lean N1: end on a noise-block boundary if that keeps half of it
            const uint64_t at0 = call_c0 + (uint64_t)at, end = at0 + (uint64_t)S;
            const uint64_t al = end & ~(uint64_t)(NoiseModel::kB - 1);
            if (al > at0 + (uint64_t)(S / 2)) S = (long)(al - at0);
        }
        return S;
    };
    // S1 / lean N1: the noise of super-block k+1 is generated while the lanes run super-block k, into the other buffer:
    // the jobs of the pool first pull GENERATION TASKS (one per whole noise block of the run's table that the super-block
    // touches and the current one does not hold: each a pure function of (seed, block, table)), then blocks. A lane
    // gathers its block's shots from the span when it takes it. Serial (no lanes): the tasks run on the home.
    const NoiseTable& NT = run_table_();
    int state[2] = {0, 0};                            // per buffer: 0 empty, 1 generated
    auto prep = [&](int w, uint64_t c0, long S) {     // buffer w covers cumulative shots [c0, c0 + S)
        NoiseSpan& sp = LS.span[w];
        const NoiseSpan& pv = LS.span[w ^ 1];
        const bool pv_ok = state[w ^ 1] == 1 && !pv.blk.empty();
        sp.c0 = c0;
        sp.b0 = c0 >> NoiseModel::kBShift;
        const uint64_t b1 = (c0 + (uint64_t)S - 1) >> NoiseModel::kBShift;
        sp.blk.assign((size_t)(b1 - sp.b0 + 1), nullptr);
        LS.gtask[w].clear();
        for (uint64_t b = sp.b0; b <= b1; ++b) {
            std::shared_ptr<NoiseBlockEv>& slot = sp.blk[(size_t)(b - sp.b0)];
            if (pv_ok && b >= pv.b0 && b < pv.b0 + pv.blk.size()) { slot = pv.blk[(size_t)(b - pv.b0)]; continue; }
            slot = std::make_shared<NoiseBlockEv>();
            slot->b = b;
            LS.gtask[w].push_back((size_t)(b - sp.b0));
        }
    };
    std::atomic<int> gnext{0};
    std::vector<std::exception_ptr> gerr((size_t)std::max(Te, 1));
    std::vector<double> gsec((size_t)std::max(Te, 1), 0.0);
    // job j's share of buffer w's generation tasks (pulled; every task is pulled by a job that runs it to its end)
    auto gen_tasks = [&](int w, int j) {
        const auto td = clk::now();
        try {
            const int NGw = (int)LS.gtask[w].size();
            for (int g; (g = gnext.fetch_add(1, std::memory_order_relaxed)) < NGw;) {
                if (XTIM_UNLIKELY(g_twirl_gen_task_test_hook != nullptr)) g_twirl_gen_task_test_hook(g);
                gen_block_ev_(noise, NT, seed_, *LS.span[w].blk[LS.gtask[w][(size_t)g]]);
            }
        } catch (...) {
            if (!gerr[(size_t)j]) gerr[(size_t)j] = std::current_exception();
        }
        gsec[(size_t)j] += std::chrono::duration<double>(clk::now() - td).count();
    };
    // after buffer w's generation: its time and failures
    auto layout = [&](int w) {
        for (double& x : gsec) { t_draw += x; x = 0.0; }
        for (auto& e : gerr) if (e) { std::exception_ptr x = e; e = nullptr; std::rethrow_exception(x); }
        state[w] = 1;
    };
    auto draw = [&](int w, long S) {                  // serial: buffer w's tasks on the home
        prep(w, done_shots_, S);
        gnext.store(0);
        gen_tasks(w, 0);
        layout(w);
    };
    long S_sb = sb_size(done, B, bpl);
    if (!serial_rest) {                               // the call's first super-block: a generation-only fan-out
        prep(cur, done_shots_, S_sb);
        gnext.store(0);
        std::fenv_t fe0;
        std::fegetenv(&fe0);
        ThreadPool::instance().run(Te, [&](int j) {
            if (j) std::fesetenv(&fe0);               // the caller's floating-point environment (the skips' log)
            gen_tasks(cur, j);
        });
        layout(cur);
    }
    while (done < shots) {
        const long nb = (S_sb + B - 1) / B;
        if (state[cur] == 0) draw(cur, S_sb);
        const long B_next = std::min(kBlockMax, 2 * B);
        const int bpl_next = std::min(kBplMax, 2 * bpl);
        const long S_next = done + S_sb < shots ? sb_size(done + S_sb, B_next, bpl_next) : 0;
        const int nxt = cur ^ 1;
        state[nxt] = 0;
        const bool gen_next = !serial_rest && S_next > 0;
        if (gen_next) prep(nxt, done_shots_ + (uint64_t)S_sb, S_next);
        std::exception_ptr draw_err;
        if (rot_on_) (void)rot_identity_();          // lean D1-S2: the s = 0 plan in the home store (lanes read it)
        else if (nextra_ > 0 || rs) ensure_rec_(identity_plan_());   // resident and built after the last boundary
        LS.acc.assign((size_t)S_sb, kAccNone);
        const long row0 = family_sink_ ? family_sink_->row : rs ? rs->row : 0;
        const uint64_t shot0 = done_shots_;            // (the deferred run moves done_shots_: restored below)
        const size_t cls0 = cache->classes();          // M1: the home's classes before this super-block
        owner.assign((size_t)nb, -1);
        if (!serial_rest) {
        for (int i = 0; i < Te; ++i) {
            Impl& L = *LS.lanes[(size_t)i];
            LaneCtx& C = *LS.ctx[(size_t)i];
            C.span = &LS.span[cur]; C.next = &next; C.stop = &stop;
            C.home_plans = (uint32_t)cache->size();      // F2: the home plans (overlay ids are above)
            C.home_classes = (uint32_t)cache->classes(); // §2 R2: the lane's hit marks over the home classes
            // speed fix 5: the marks persist (all zero: the barrier's fold resets the touched ones) — grown, never
            // re-zeroed per super-block (that was O(classes × lanes) serial work at every barrier)
            for (uint32_t k : C.touched) C.marks[(size_t)k] = 0;   // (a failed super-block's marks: never folded)
            C.touched.clear();
            if (C.marks.size() < (size_t)C.home_classes) C.marks.resize((size_t)C.home_classes, 0);
            if (C.touched.capacity() < (size_t)C.home_classes)
                C.touched.reserve(std::max<size_t>((size_t)C.home_classes, 2 * C.touched.capacity()));
            C.nblocks = nb; C.B = B; C.S_sb = S_sb;
            C.shot0 = shot0; C.done0 = done; C.row0 = row0;
            C.acc = LS.acc.data(); C.mark = kAccLane; C.pend = false;
            C.cur_block = -1;
            C.blocks.clear();
            C.deferred.clear();
            C.rpend.clear();
            C.ev_marks.clear();
            L.plan0_ = plan0_;
            L.born_node_event_ = born_event;
            LS.overlay[(size_t)i]->set_id_base((uint32_t)cache->size());   // overlay ids never meet the home's
            LS.overlay[(size_t)i]->set_class_base((uint32_t)cache->classes());
            cache->lend_arena_to(*LS.overlay[(size_t)i]);   // review H1: the lane bumps from a home chunk's tail
            ok[(size_t)i] = 1;
            err[(size_t)i] = nullptr;
        }
        next.store(0);
        stop.store(false);
        gnext.store(0);
        const bool last = done + S_sb >= shots;
        ThreadPool::instance().run(Te, [&](int i) {
            if (i) std::fesetenv(&fe);                // the caller's floating-point environment on every lane
            if (gen_next) gen_tasks(nxt, i);          // the next super-block's generation tasks first (lean N1)
            ScratchCallScope scope;
            Impl& L = *LS.lanes[(size_t)i];
            try {
                if (!(L.*lane_fn)(LONG_MAX)) ok[(size_t)i] = 0;
                else {
                    LS.overlay[(size_t)i]->adopt_prepare(*cache);   // S4: the adoption's hashing and home probes
                    if (rs) {
                        L.fam_lane_resolve_(*this);         // F2: the lane's record contents against the home's
                        L.fam_lane_premap_(*this, LS.acc.data(), row0, S_sb, B, *LS.ctx[(size_t)i]);   // B4
                    }
                }
            } catch (...) {
                err[(size_t)i] = std::current_exception();
                ok[(size_t)i] = 0;
            }
            if (!ok[(size_t)i]) stop.store(true, std::memory_order_relaxed);
            // a worker's chi-scaled scratch dies with the call (fix F4: on a failure too)
            if (i && (last || !ok[(size_t)i] || stop.load(std::memory_order_relaxed))) release_thread_scratch_slots();
        });
        if (gen_next) {
            try { layout(nxt); } catch (...) { draw_err = std::current_exception(); }
        }
        // Fix F2: a lane's state-budget refusal (its own chi-scaled registers / scratch) is not a refusal of the call:
        // the super-block's lane work is dropped, its rows cleared, and the home re-runs it serially from the drawn
        // events (each shot at its index and row: the same bytes), then the rest of the call; the lanes are freed.
        bool cap = false;
        for (int i = 0; i < Te && !cap; ++i)
            if (err[(size_t)i]) {
                try { std::rethrow_exception(err[(size_t)i]); }
                catch (const BareStateCapacityError&) { cap = true; }
                catch (...) {}
            }
        if (cap) {
            for (int i = 0; i < Te; ++i) {            // the lanes' plans dropped; their chunks home (H1 / L3)
                LS.overlay[(size_t)i]->assert_lane_tables_unheld();
                LS.overlay[(size_t)i]->clear_plans();
                LS.overlay[(size_t)i]->drop_lane_arena();
                if (rot_on_) { LS.lanes[(size_t)i]->rstore_->clear(); LS.lanes[(size_t)i]->rot_cow_.clear(); }
            }
            LS.lanes.clear();
            LS.ctx.clear();
            LS.overlay.clear();
            clear_rows(row0, S_sb);
            LS.acc.assign((size_t)S_sb, kAccNone);
            serial_rest = true;
            ++threads_budget_fallbacks_;
            threads_limit_ = "the state budget refused a lane's allocation: the call continued on the calling thread";
        } else {
        // The earliest failed block decides (every earlier block was taken before it and ran to its end).
        int fl = -1;
        for (int i = 0; i < Te; ++i)
            if (!ok[(size_t)i] && (fl < 0 || LS.ctx[(size_t)i]->cur_block < LS.ctx[(size_t)fl]->cur_block)) fl = i;
        const long fb = fl >= 0 ? LS.ctx[(size_t)fl]->cur_block : LONG_MAX;
        if (fl >= 0) {                                // the call fails: the lanes' plans are dropped, not adopted
            // Fix F3: the deferred shots before the failed block's first shot run first (serially they come
            // before it): their failure, if any, is the call's.
            // (fix R2: and the failing lane's own deferred shots inside block fb — it deferred them before failing)
            std::vector<long> early;
            for (int i = 0; i < Te; ++i)
                for (long d : LS.ctx[(size_t)i]->deferred)
                    if (d < fb * B || (i == fl && d / B == fb)) early.push_back(d);
            std::sort(early.begin(), early.end());
            for (int i = 0; i < Te; ++i) {
                LS.overlay[(size_t)i]->assert_lane_tables_unheld();   // review L3: loud, before the blocks go
                LS.overlay[(size_t)i]->clear_plans();
                LS.overlay[(size_t)i]->drop_lane_arena();   // §4: nothing it allocated reached a reader
                LS.ctx[(size_t)i]->cow.clear();
                if (rot_on_) { LS.lanes[(size_t)i]->rstore_->clear(); LS.lanes[(size_t)i]->rot_cow_.clear(); }
                lane_counters_into(this, *LS.lanes[(size_t)i]);
                fam_lane_merge_(*LS.lanes[(size_t)i]);
            }
            if (!early.empty()) {
                const bool born0 = born_event;
                const size_t ev0 = diag_.size();
                bool okE = true;
                std::exception_ptr eE;
                try { okE = home_run(early, nb, B, S_sb, shot0, done, row0); }
                catch (...) { eE = std::current_exception(); okE = false; }
                if (!okE) {
                    // fix R3: the call's BORN_NODE event is the earliest one, a lane's from a block before the failed
                    // deferred shot's included (serially those shots ran first)
                    const long fbk = LS.dctx.cur_s / B;
                    const TwirlRecordSampler::DiagEvent* lb = nullptr;
                    long lbk = LONG_MAX;
                    if (!born0)
                        for (int i = 0; i < Te; ++i) {
                            const Impl& L = *LS.lanes[(size_t)i];
                            const auto& M = LS.ctx[(size_t)i]->ev_marks;
                            for (size_t m = 0; m < M.size(); ++m) {
                                if (M[m].first >= fbk || M[m].first >= lbk) continue;
                                const size_t e1 = m + 1 < M.size() ? M[m + 1].second : L.diag_.size();
                                for (size_t e = M[m].second; e < e1; ++e)
                                    if (L.diag_[e].kind == TwirlRecordSampler::DiagEvent::BORN_NODE) {
                                        lb = &L.diag_[e];
                                        lbk = M[m].first;
                                        break;
                                    }
                            }
                        }
                    if (lb) {
                        TwirlRecordSampler::DiagEvent keep = *lb;
                        for (size_t e = ev0; e < diag_.size();)
                            if (diag_[e].kind == TwirlRecordSampler::DiagEvent::BORN_NODE) diag_.erase(diag_.begin() + (long)e);
                            else ++e;
                        diag_.insert(diag_.begin() + (long)ev0, std::move(keep));
                    }
                    for (auto& L : LS.lanes) L->diag_.clear();
                    if (eE) std::rethrow_exception(eE);
                    return false;
                }
            }
        }
        // The lanes' events, in block order: the call's one BORN_NODE event (when the window prefix queued none) and
        // the failed block's ORACLE_FAILURE.
        {
            std::vector<std::pair<long, const TwirlRecordSampler::DiagEvent*>> evs;
            for (int i = 0; i < Te; ++i) {
                const Impl& L = *LS.lanes[(size_t)i];
                const auto& M = LS.ctx[(size_t)i]->ev_marks;
                for (size_t m = 0; m < M.size(); ++m) {
                    const size_t e1 = m + 1 < M.size() ? M[m + 1].second : L.diag_.size();
                    for (size_t e = M[m].second; e < e1; ++e) evs.emplace_back(M[m].first, &L.diag_[e]);
                }
            }
            std::stable_sort(evs.begin(), evs.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& ev : evs) {
                if (ev.first > fb) break;
                if (ev.second->kind == TwirlRecordSampler::DiagEvent::BORN_NODE) {
                    if (!born_event) { diag_.push_back(*ev.second); born_event = true; }
                } else if (ev.first == fb) {
                    diag_.push_back(*ev.second);
                }
            }
            for (auto& L : LS.lanes) L->diag_.clear();
        }
        if (fl >= 0) {
            if (err[(size_t)fl]) std::rethrow_exception(err[(size_t)fl]);
            return false;
        }
        if (er_.p)
            for (int i = 0; i < Te; ++i) {
                ErState& le = *LS.lanes[(size_t)i]->er_.p;
                er_.p->early += le.early;
                er_.p->late += le.late;
                le.early = le.late = 0;
            }
        // The barrier: the lanes' plans (lane order), their grown tries, their counters, their hit marks (§2 R2).
        for (int i = 0; i < Te; ++i) {
            LaneCtx& C = *LS.ctx[(size_t)i];          // speed fix 5: the touched indices only, reset as read
            for (uint32_t k : C.touched) { cache->mark_class((size_t)k); C.marks[(size_t)k] = 0; }
            C.touched.clear();
        }
        for (int i = 0; i < Te; ++i) cache->adopt_from(*LS.overlay[(size_t)i]);
        if (rot_on_)                                  // lean D1-S2: the lanes' rotation plans / tuples (content-keyed), their cow tries
            for (int i = 0; i < Te; ++i) {
                Impl& L = *LS.lanes[(size_t)i];
                rstore_->adopt_from(*L.rstore_);
                for (auto& kv : L.rot_cow_) { born_trie_merge(rstore_->tuple_mut(kv.first).trie, kv.second); rstore_->note_grown(kv.first); }
                L.rot_cow_.clear();
            }
        for (int i = 0; i < Te; ++i) {
            LaneCtx& C = *LS.ctx[(size_t)i];
            for (auto& kv : C.cow) {                  // the home classes' tries the lane grew (cow by class)
                born_trie_merge(kv.first->rec_trie, kv.second);
                cache->reaccount(*kv.first);
            }
            C.cow.clear();
            for (long b : C.blocks) owner[(size_t)b] = i;
            lane_counters_into(this, *LS.lanes[(size_t)i]);
            fam_lane_merge_(*LS.lanes[(size_t)i]);
        }
        for (long b = 0; b < nb; ++b)
            if (owner[(size_t)b] < 0) throw std::logic_error("threads: a super-block's block was not run (engine bug)");
        // T4: the shots the lanes deferred (the term route: its state is the home's), in shot order, on the home —
        // each at its global index and row.
        LS.dlist.clear();
        for (int i = 0; i < Te; ++i)
            LS.dlist.insert(LS.dlist.end(), LS.ctx[(size_t)i]->deferred.begin(), LS.ctx[(size_t)i]->deferred.end());
        if (!LS.dlist.empty()) {
            std::sort(LS.dlist.begin(), LS.dlist.end());
            threads_deferred_ += (long)LS.dlist.size();
            if (!home_run(LS.dlist, nb, B, S_sb, shot0, done, row0)) {
                cache->order_new_classes(cls0);       // re-review N-1: the failure path orders the adopted classes too
                return false;
            }
        }
        cache->order_new_classes(cls0);               // review M1: the super-block's new classes in first-shot order
        }
        }
        if (serial_rest) {                            // fix F2: the home runs the whole super-block
            LS.dlist.resize((size_t)S_sb);
            for (long i = 0; i < S_sb; ++i) LS.dlist[(size_t)i] = i;
            if (!home_run(LS.dlist, nb, B, S_sb, shot0, done, row0)) return false;
        }
        // Every shot emitted its row (or, under a postselection mask, was rejected): the lanes' record content
        // handles to the home's, in shot order (T4); the accepted rows down in shot order (T3).
        if (family_sink_ || rs) {
            for (long sidx = 0; sidx < S_sb; ++sidx) {
                const uint8_t a = LS.acc[(size_t)sidx];
                if (a == kAccDefer || (!er_.p && a == kAccNone))
                    throw std::logic_error("threads: a shot of the super-block emitted no row (engine bug)");
            }
            if (rs) {                                 // fid-routine B4: the lanes premapped theirs; the rest in shot order
                const std::vector<long>* pl = nullptr;
                if (!serial_rest) {
                    LS.rlist.clear();
                    for (int i = 0; i < Te; ++i)
                        LS.rlist.insert(LS.rlist.end(), LS.ctx[(size_t)i]->rpend.begin(), LS.ctx[(size_t)i]->rpend.end());
                    std::sort(LS.rlist.begin(), LS.rlist.end());
                    pl = &LS.rlist;
                }
                fam_lane_remap_(LS.acc.data(), row0, S_sb, B, owner, LS.lanes, pl);
            }
            FamilySink* fs = family_sink_;
            long dst = row0;
            if (er_.p) {
                for (long sidx = 0; sidx < S_sb; ++sidx) {
                    if (LS.acc[(size_t)sidx] == kAccNone) continue;
                    const long src = row0 + sidx;
                    if (src != dst) {
                        if (fs && fs->DBB) std::memcpy(fs->dets + (size_t)dst * fs->DBB, fs->dets + (size_t)src * fs->DBB, (size_t)fs->DBB);
                        if (fs && fs->OBB) std::memcpy(fs->obs + (size_t)dst * fs->OBB, fs->obs + (size_t)src * fs->OBB, (size_t)fs->OBB);
                        if (fs && fs->DECB) std::memcpy(fs->dec + (size_t)dst * fs->DECB, fs->dec + (size_t)src * fs->DECB, (size_t)fs->DECB);
                        if (rs) record_row_move_(*rs, dst, src);
                    }
                    ++dst;
                }
                const long end = row0 + S_sb;
                if (dst < end) {                      // the rows left behind: as a fresh buffer holds them
                    if (fs && fs->DBB) std::memset(fs->dets + (size_t)dst * fs->DBB, 0, (size_t)(end - dst) * fs->DBB);
                    if (fs && fs->OBB) std::memset(fs->obs + (size_t)dst * fs->OBB, 0, (size_t)(end - dst) * fs->OBB);
                    if (fs && fs->DECB) std::memset(fs->dec + (size_t)dst * fs->DECB, 0, (size_t)(end - dst) * fs->DECB);
                    if (rs) for (long r = dst; r < end; ++r) record_row_clear_(*rs, r);
                }
            } else {
                dst = row0 + S_sb;
            }
            if (fs) fs->row = dst;
            if (rs) rs->row = dst;
        }
        done_shots_ = shot0 + (uint64_t)S_sb;
        done += S_sb;
        if (budget_on) payload_boundary_();
        if (draw_err) std::rethrow_exception(draw_err);   // (the super-block itself completed)
        B = B_next;
        LS.B = B;
        bpl = bpl_next;
        S_sb = S_next;
        cur = nxt;
    }
    wall += std::chrono::duration<double>(clk::now() - t0).count();
    threads_eff_ = serial_rest ? 1 : Te;
    return true;
}

}  // namespace qeccore

namespace qeccore {
// Fix-2 F1 (coordinator ruling): a failed call commits nothing. Its shots are never returned, so on every failure path
// (serial or threaded: a refused or failed shot, a lane's failure, a noise-generation failure) the seed and the
// cumulative shot index go back to their values at the call's entry (lean N1: the noise is addressed by them — a pure
// function of (seed, block, table), no stream position), and a later seed=None call continues exactly as if the failed
// call had not been made — the same bytes at every thread count. The state is saved here, outside run(long) / run_w,
// and restored on the failure paths only. What else a
// failed call leaves: plan-cache entries, Born-trie nodes, record contents (pure caches / append-only, read by
// content), the counters (diagnostics) and the spent selfcheck window — none is addressed by shot index.
bool TwirlRecordSampler::Impl::run_committed_(long shots, int threads) {
    if (!entry_) entry_.reset(new CallState());
    call_save_(*entry_);
    bool ok;
    try {
        ok = run_threads_(shots, threads);
    } catch (...) {
        call_restore_(*entry_);
        throw;
    }
    if (!ok) call_restore_(*entry_);
    return ok;
}
bool TwirlRecordSampler::run(long shots, int threads) { return impl_->run_committed_(shots, threads); }
void TwirlRecordSampler::Impl::call_save_(CallState& c) const {
    c.seed = seed_;
    c.done_shots = done_shots_;
    c.selfcheck_spent = selfcheck_spent_;
    c.clean = clean; c.diag = diag; c.checked = checked; c.used = used;
    std::copy(route_, route_ + R_COUNT, c.route);
    c.fast_derived = fast_derived_; c.structural = structural_shots_; c.fast_born = fast_born_shots_;
    c.obs_ones = obs_ones_; c.ps_acc = ps_acc_; c.ps_obs_ones = ps_obs_ones_;
    c.sem_checked = sem_checked_;
    std::copy(sem_class_, sem_class_ + 6, c.sem_class);
    c.trie_walks = trie_.walks; c.trie_steps = trie_.steps; c.obs_draws_record = obs_draws_record_;
    c.chancnt = chancnt;
    c.t_draw = t_draw; c.wall = wall;
    fam_shot_stats_save_(c.fam);
    c.seed_epoch = seed_epoch_;
    CachedWork& w = c.cached;
    w.nodes = trie_.nodes; w.node_evals = trie_.node_evals; w.chi_j = trie_.chi_j; w.passes = trie_.passes;
    w.max_support = trie_.max_support; w.max_depth = trie_.max_depth;
    w.rec_tables = rec_tables_; w.rec_extras = rec_extras_; w.rec_ops_plane = rec_ops_plane_;
    w.rec_ops_pivot = rec_ops_pivot_; w.rec_ops_greduce = rec_ops_greduce_; w.rec_ops_echelon = rec_ops_echelon_;
    w.rec_pivots_extras = rec_pivots_extras_; w.rec_dirs_extras = rec_dirs_extras_;
    w.kappa_hist = rec_kappa_hist_;
    c.threads_req = threads_req_; c.threads_eff = threads_eff_;
    c.threads_deferred = threads_deferred_; c.threads_budget_fallbacks = threads_budget_fallbacks_;
    c.threads_limit = threads_limit_;
}
// (noexcept: equal-size copies into the sampler's own storage — the channel counts and the stream have a fixed shape,
// the term route's per-plan shot counts only ever grow, so a restore shrinks them — and the key derivation)
void TwirlRecordSampler::Impl::call_restore_(const CallState& c) noexcept {
    set_keys_(c.seed);
    done_shots_ = c.done_shots;
    selfcheck_spent_ = c.selfcheck_spent;
    clean = c.clean; diag = c.diag; checked = c.checked; used = c.used;
    std::copy(c.route, c.route + R_COUNT, route_);
    fast_derived_ = c.fast_derived; structural_shots_ = c.structural; fast_born_shots_ = c.fast_born;
    obs_ones_ = c.obs_ones; ps_acc_ = c.ps_acc; ps_obs_ones_ = c.ps_obs_ones;
    sem_checked_ = c.sem_checked;
    std::copy(c.sem_class, c.sem_class + 6, sem_class_);
    trie_.walks = c.trie_walks; trie_.steps = c.trie_steps; obs_draws_record_ = c.obs_draws_record;
    if (c.chancnt.size() == chancnt.size()) std::copy(c.chancnt.begin(), c.chancnt.end(), chancnt.begin());
    t_draw = c.t_draw; wall = c.wall;
    fam_shot_stats_restore_(c.fam);
    fam_term_reset_shot_();
    if (seed_epoch_ != c.seed_epoch) {               // fix-4 (B): the call reseeded (zeroed the cached-work counters)
        const CachedWork& w = c.cached;              // after it: its own work; restored: entry value + that work
        trie_.nodes += w.nodes; trie_.node_evals += w.node_evals; trie_.chi_j += w.chi_j; trie_.passes += w.passes;
        trie_.max_support = std::max(trie_.max_support, w.max_support);
        trie_.max_depth = std::max(trie_.max_depth, w.max_depth);
        rec_tables_ += w.rec_tables; rec_extras_ += w.rec_extras; rec_ops_plane_ += w.rec_ops_plane;
        rec_ops_pivot_ += w.rec_ops_pivot; rec_ops_greduce_ += w.rec_ops_greduce; rec_ops_echelon_ += w.rec_ops_echelon;
        rec_pivots_extras_ += w.rec_pivots_extras; rec_dirs_extras_ += w.rec_dirs_extras;
        try {                                        // (a histogram that grows may allocate: kept as is if it fails)
            if (rec_kappa_hist_.size() < w.kappa_hist.size()) rec_kappa_hist_.resize(w.kappa_hist.size(), 0);
            for (size_t k = 0; k < w.kappa_hist.size(); ++k) rec_kappa_hist_[k] += w.kappa_hist[k];
        } catch (...) {
        }
        seed_epoch_ = c.seed_epoch;
    }
    threads_req_ = c.threads_req; threads_eff_ = c.threads_eff;   // fix-4 (E): the thread report
    threads_deferred_ = c.threads_deferred; threads_budget_fallbacks_ = c.threads_budget_fallbacks;
    try { threads_limit_ = c.threads_limit; } catch (...) { threads_limit_.clear(); }
}
// fix-3: the binding's Python call (several engine calls: an input-branch run's E-class inner samplers) saves each
// sampler's state at its entry, before any reseed, and restores every one that ran on a failure.
struct TwirlRecordSampler::CallState {
    Impl::CallState st;
};
std::shared_ptr<TwirlRecordSampler::CallState> TwirlRecordSampler::save_call_state() const {
    auto c = std::make_shared<CallState>();
    impl_->call_save_(c->st);
    return c;
}
void TwirlRecordSampler::restore_call_state(const CallState& c) noexcept { impl_->call_restore_(c.st); }
int TwirlRecordSampler::threads_requested() const { return impl_->threads_req_; }
int TwirlRecordSampler::threads_effective() const { return impl_->threads_eff_; }
const std::string& TwirlRecordSampler::threads_limit() const { return impl_->threads_limit_; }
long TwirlRecordSampler::threads_deferred() const { return impl_->threads_deferred_; }
long TwirlRecordSampler::threads_budget_fallbacks() const { return impl_->threads_budget_fallbacks_; }
void TwirlRecordSampler::family_set_lane_bytes(int64_t bytes) { impl_->lane_bytes_test_ = bytes; }
uint64_t TwirlRecordSampler::family_shot_index() const { return impl_->done_shots_; }
}  // namespace qeccore
