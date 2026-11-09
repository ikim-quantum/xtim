// twirl_sampler_runw.inc.hpp — INTERNAL to the twirl sampler (J-a r2.1, JA-1): the shot loop,
// TwirlRecordSampler::Impl::run_w<FW>, included by twirl_sampler_fam.cpp only, which instantiates it (J-b2 J3/J4: the
// legacy instance and its source are gone; J9: the always-true FAM parameter is dropped). Include after
// twirl_sampler_impl.inc.hpp.
#pragma once

#include "twirl_sampler_impl.inc.hpp"

namespace qeccore {

// early-reject: a shot rejected before any plan work. Its cumulative index shot_ stays spent (assigned at the top of
// the shot), so every later shot keeps its counter-based streams.
#ifndef XTIM_MUTATE_ER_NO_SHOT_ADVANCE
#define XTIM_ER_REJECT_() do { ++er_.p->early; } while (0)
#else
#define XTIM_ER_REJECT_() do { ++er_.p->early; --done_shots_; } while (0)   // mutation build only: index reused
#endif

// J-a (N1): the emit seam — emit_shot_fam_ (the FamilySink; J-b2 J6b deleted the two legacy sinks).
#define XTIM_EMIT_SHOT(...) \
    do { emit_shot_fam_(__VA_ARGS__); } while (0)

}  // namespace qeccore
#include "twirl_sampler_rotloop.inc.hpp"   // lean D1-S2: rot_shot_<FWE> (the rotation shot loop)
namespace qeccore {

// ── lean N1: the noise blocks of a serial run (qeccore/noise_chain.hpp) ──
// The block of a (seed, table): generated once and kept (a pure cache: the next call of a seed=None continuation, or the
// shot-block route after the window's event-list shots, reads the same block).
const std::vector<uint64_t>& TwirlRecordSampler::Impl::nb_block_(const NoiseTable& T, uint64_t b) {
    if (!nblk_) nblk_.reset(new NoiseBlock());
    NoiseBlock& K = *nblk_;
    if (!(K.ok && K.seed == seed_ && K.T == &T && K.b == b)) {
        K.ok = false;
        noise.block_events(T, seed_, b, K.ev);
        K.seed = seed_; K.T = &T; K.b = b; K.ok = true;
    }
    return K.ev;
}
void TwirlRecordSampler::Impl::nb_events_(const NoiseTable& T, uint64_t c0, int S, DiagErrorSampler::SparseBatch& out) {
    const std::vector<uint64_t>& ev = nb_block_(T, c0 >> NoiseModel::kBShift);
    NoiseModel::shot_major(ev, (int)(c0 & (NoiseModel::kB - 1)), S, out, nb_cnt_);
}

// ── P3 S-b: the shot block (Impl members: twirl_sampler_impl.inc.hpp; sb_ok_: twirl_sampler_fam.cpp) ──
// The rows' widths are the record layout's (rot_shot_: rot.TW; the Clifford path: rot_rec.TW) and XR_. The rows hold one
// noise block (4096 shots) unless that would pass 8 MB (then fewer, ≥ 256: the block's chain is filtered per row range).
// The noise is a function of (seed, block, table), so the rows' capacity moves no byte.
void TwirlRecordSampler::Impl::sb_size_() {
    const int TW = rot_on_ ? rot.TW : rot_rec.TW, KR = TW + 1, XR = XR_;
    const size_t per = (size_t)CW + (size_t)KR + (size_t)XR + 1;
    int B = NoiseModel::kB;
    while (B > 256 && (size_t)B * per * 8 > ((size_t)8 << 20)) B >>= 1;
    if (B == sb_B_) return;
    sb_B_ = B;
    sb_zero_ = 0;
    sb_rows_.assign((size_t)B * (size_t)std::max(CW, 1), 0);
    sb_clean_ = B;
    sb_key_.assign((size_t)B * (size_t)KR, 0);
    sb_xb_.assign((size_t)B * (size_t)std::max(XR, 1), 0);
    sb_nev_.assign((size_t)B, 0);
    sb_rej_.assign((size_t)B, 0);
}
__attribute__((noinline)) void TwirlRecordSampler::Impl::sb_fill_(const NoiseTable& T, uint64_t c0, int S) {
    const int TW = rot_on_ ? rot.TW : rot_rec.TW, KR = TW + 1, XR = XR_, cw = CW;
    uint64_t* const rows = sb_rows_.data();
    uint64_t* const key = sb_key_.data();
    uint64_t* const xb = sb_xb_.data();
    int32_t* const nev = sb_nev_.data();
    // FCS fix 2: a table with no mechanism fires nothing — the rows of a previous no-event fill are this fill's
    if (T.M() == 0 && S <= sb_zero_) return;
    sb_zero_ = 0;
    if (S > sb_clean_)                                // FCS round 2 (b): rows [0, sb_clean_) are already zero
        std::memset(rows + (size_t)sb_clean_ * (size_t)cw, 0, (size_t)(S - sb_clean_) * (size_t)cw * 8);
    sb_clean_ = 0;                                    // the scatter below writes the rows
    std::memset(key, 0, (size_t)S * (size_t)KR * 8);
    if (XR) std::memset(xb, 0, (size_t)S * (size_t)XR * 8);
    std::memset(nev, 0, (size_t)S * 4);
    for (int s = 0; s < S; ++s) key[(size_t)s * KR + TW] = 0xcbf29ce484222325ull;   // rot_shot_'s h before the tokens
    const RotationRecords& R = rot_rec;
    const size_t hdr = (size_t)2 * TW + 2 + (size_t)2 * XR + 1;
    auto v64 = [](const uint32_t* q) { return (uint64_t)q[0] | ((uint64_t)q[1] << 32); };
    // FCS fix 7: the key / extras words of a record are 0-2 words on the workloads (TW = ⌈t/64⌉, XR): explicit short
    // cases instead of a general (vectorized, alias-checked) loop per event; the same XORs in the same order.
    auto xw = [&](uint64_t* d, const uint32_t* q, int n) {
        switch (n) {
            case 0: return;
            case 1: d[0] ^= v64(q); return;
            case 2: d[0] ^= v64(q); d[1] ^= v64(q + 2); return;
            default: for (int w = 0; w < n; ++w) d[w] ^= v64(q + 2 * w);
        }
    };
    auto f = [&](int32_t a, int s) {
        const uint32_t* Bp = R.rec.data() + R.off[(size_t)a];
        ++nev[s];
        uint64_t* k = key + (size_t)s * KR;
        xw(k, Bp, TW);
        k[TW] ^= v64(Bp + 2 * TW);
        xw(xb + (size_t)s * XR, Bp + 2 * TW + 2, XR);
        const uint32_t nz = Bp[hdr - 1];
        const uint32_t* p = Bp + hdr;
        uint64_t* row = rows + (size_t)s * cw;
        for (uint32_t i = 0; i < nz; ++i, p += 3) row[p[0]] ^= v64(p + 1);
    };
    const uint64_t b = c0 >> NoiseModel::kBShift;
    const int lo = (int)(c0 & (NoiseModel::kB - 1));
    if (T.M() == 0) { sb_zero_ = S; return; }         // FCS fix 2: no mechanism, no event (the rows stay a no-event fill)
    if (lo == 0 && S == NoiseModel::kB) {             // the whole block: applied where it is drawn (no list)
        noise.walk(T, seed_, b, f);
        return;
    }
    const uint32_t ulo = (uint32_t)lo, uS = (uint32_t)S;   // a part of the block: its kept chain, filtered
    for (uint64_t e : nb_block_(T, b)) {
        const uint32_t s = (uint32_t)e - ulo;
        if (s < uS) f((int32_t)(e >> 32), (int)s);
    }
}

template <int FWE>
bool TwirlRecordSampler::Impl::run_w(long shots) {
    // early-reject: FWE = FW, or FW | kErRun for the masked instances (twirl_sampler_er.cpp); one body.
    // threads T2: FW | kLaneRun (| kErRun) are a LANE's instances (twirl_sampler_threads.cpp): the shots of the blocks the
    // lane takes from its home's super-block, against the home's plan cache (read only) and the lane's overlay.
    // threads T4: FW | kDeferRun (| kErRun) are the HOME's instances for the shots its lanes deferred (the term route:
    // CH / PPR shots), run after the barrier in shot order, each shot at its global index and row (blk_).
    constexpr int FW = FWE & ~(kErRun | kLaneRun | kDeferRun);
    constexpr bool ER = (FWE & kErRun) != 0;
    constexpr bool LANE = (FWE & kLaneRun) != 0;
    constexpr bool DEFER = (FWE & kDeferRun) != 0;
    constexpr bool BLK = LANE || DEFER;           // the shots come from a block source (blk_), not the noise stream
    // S0(b): the keyed per-shot streams address draw (shot·2^20 + ctr) — injective (never a repeated
    // draw) only for shot < 2^44 per seed and < 2^20 draws per (shot, stream). Refused by name (CAPS-2 A4: why
    // neither is reachable, keyed_rng.hpp). The BORN words (one per Born step, κ <= 2k) need no check:
    static_assert(2 * kMaxBareRank < (int)krng::kMaxCtr, "Born steps per shot < 2^20");
    {
        const uint64_t per_shot = std::max<uint64_t>({(uint64_t)nrf, (uint64_t)ng, (uint64_t)born_dec.size()});
        if (per_shot >= krng::kMaxCtr)
            throw std::invalid_argument("twirl sampler: a shot needs " + std::to_string(per_shot) +
                " draws of one keyed stream (noisy records / certified generators / Born decisions / "
                "gauge detectors); the keyed stream layout allows < 2^20 per shot");
        if (!BLK && shots > 0 && (done_shots_ >= krng::kMaxShots || (uint64_t)shots > krng::kMaxShots - done_shots_))
            throw std::invalid_argument("twirl sampler: the cumulative shot index of this seed would "
                "reach 2^44 (keyed streams address shot*2^20 + draw); pass a new seed to continue");
    }
    // S2.2: state retention only supports the DIAGONAL residual class. A CH/PPR circuit routes
    // shots through the PPR fast path (which never materialises amps) — refuse loudly rather
    // than silently deliver un-retained states.
    // Born fast-path (2026-08-08): `need_state` is now retention ALONE. Born DECISIONs no longer
    // force the structural per-shot canonicalize/collapse detour — they run on the abelian σ fast
    // path and feed the EXISTING born-probability memo (materialize_shot on a memo miss). This is a
    // COMPILE-TIME routing decision (has_born_dec_ is set once at construction). S5a: a retained shot
    // on a warm resident plan takes the accumulator fast path too (σ-4 and the record from its prefix).
    // Class-(b) genuinely-magic-weighted decisions (m_out = (1∓1/√2)/2) stay correct via the memo's
    // materialize_shot; the win is the σ path + class-(a)/memoized shots no longer force the detour.
    // See the born-decision fast-path scope memo (adaptq docs). S0(b): every route reads the r fair
    // coins as the same keyed COIN bits, so the recorded rec_out_ (the memo key's coins_bits) are
    // route-independent.
    // J-b1 P2: the FAM instances retain no σ state (FAM retention is the layout-3 record; fam_run_guard_ refuses
    // retain_state by name). J-b2 J4: the legacy retention branches (need_state) are deleted with the legacy source.
    select_rows_fn_();                                     // r9: the record_rows_ variant of this run
    // S0(f): the per-shot channel row (formerly a local uint64_t[8], i.e. a 512-channel cap). Every
    // reader/writer loops over CW words. FW = 8: `bits` IS a local uint64_t[8] — the array type
    // matters, not only the storage: its bound caps every `w < CW` loop's trip count for the
    // optimizer (a pointer to the same local array costs +2.2% instructions on cult_d5's fast
    // path, measured). FW = 0: a pointer to the run-width member storage (sized ONCE at
    // construction).
    using Row = std::conditional_t<(FW > 0), uint64_t[(FW > 0) ? FW : 1], uint64_t*>;
    Row bits;
    if constexpr (FW == 0) bits = bits_.data();
    long done = 0;
    long route_at_start[R_COUNT];                         // S0(a): per-run deltas for the loud lines
    std::copy(route_, route_ + R_COUNT, route_at_start);
    if constexpr (!BLK) born_node_event_ = false;         // decision 7: one BORN_NODE event per run() (threads: per call)
    trie_.time_next_eval = true;
    // S0(a): the two hot routes carry no per-shot counter. Whatever this run's shots did not count
    // is attributed when run() exits (any exit): to the κ=0 retention route on a retention run (the
    // only uncounted route there), else to the abelian streaming fast path.
    struct RemainderGuard {
        Impl& I; const long* at; long used0, clean0; bool retain;
        ~RemainderGuard() {
            long counted = I.clean - clean0;
            for (int r = 0; r < R_COUNT; ++r) counted += I.route_[r] - at[r];
            const long rem = (I.used - used0) - counted;
            if (retain) I.route_[R_RETAIN_K0] += rem; else I.fast_derived_ += rem;
        }
    } remainder_guard{*this, route_at_start, used, clean, false};
    double t_loop = 0.0;
    (void)t_loop;
    // v2.7 selfcheck once-per-sampler (option c): `sc` is the EFFECTIVE window for this
    // run() — the configured `selfcheck` on the first run() of the sampler's lifetime, 0 on
    // every later run(). The latch is never reset by set_seed() (which zeroes the
    // `checked`/`pprchecked` gating counters and would otherwise re-arm the window on every
    // seeded call); rebuild() makes a fresh Impl → the window re-arms (fresh-construction
    // semantics). Zero-shot calls do not spend the window (nothing would have been checked).
    // STREAM-NEUTRAL: a spent window is exactly selfcheck==0, byte-identical to window-on.
    // J-a (Q7): a FAM sampler built with oracle_all_shots runs the (stream-neutral) window on EVERY shot.
    // threads T2: a lane never runs the window (its home runs the window's shots serially first, and an
    // oracle_all_shots run serially: owner ruling D2).
    const long sc = BLK ? 0 : oracle_all_shots_ ? (1L << 60) : (selfcheck_spent_ ? 0 : selfcheck);
    if (!BLK && shots > 0) selfcheck_spent_ = true;
    // perf-engine §2.6: in_window_ is the current shot's window flag (set at the top of every shot, below) and true
    // again when this run returns, so a record-table build reached outside a run keeps the full checks.
    struct WindowIdle { bool& f; ~WindowIdle() { f = true; } } window_idle_{in_window_};
    sem_budget_ = (sc > 0 && (born_obs || has_born_dec_) && !ppr_tables) ? std::min<long>(sc, 64) : 0;
    // J-a (N1): the named refusals of a FAM run (σ export / retention, frames on family extras), and
    // the semantic oracle over the family extras too (every shot under oracle_all_shots, Q7).
    if constexpr (!LANE) { if (!fam_run_guard_(shots)) return false; }   // (a lane: its home ran it)
    // N4a: the term-routed shots of a CH/PPR circuit have an oracle too (J-b1 P2: a FAM sampler on a CH/PPR
    // circuit that gets past fam_run_guard_ has the ppr_records class).
    if (sc > 0 && (oracle_all_shots_ || nextra_ > 0))
        sem_budget_ = oracle_all_shots_ ? sc : std::min<long>(sc, 64);
    long sem_left = sem_budget_;                             // register-resident window count
    // early-reject: the clean-shot verdict when it is a constant of the run (K & F = 0 and no readout flips): 0 passes
    // (no late filter), 1 passes the early test (late filter), 2 per shot (er_clean_).
    [[maybe_unused]] int er_cc = 2;
    if constexpr (ER) if (er_.p->clean_ok && RFW == 0) er_cc = er_.p->full ? 0 : 1;
    // J-b1 P4: a FAM recording run serves its unframed shots on the rotation loop / the Clifford path, which write the
    // layout-3 record; the term route keeps the framed / carried-branch shots. Per run, register-resident.
    const bool famrec = fam_rec_d_();
    // lean L3: the extras' row width (0 on a circuit without record extras: every extras loop below is then empty).
    // Hoisted: register-resident like famrec (P4: a condition recomputed per shot cost +58 Ir on cult d5).
    const int XR = XR_;
    // lean D1-S2 / D1-S3: the rotation shot loop serves every unframed, non-term-recorded shot of a circuit with rotations
    // (rot_on_, decided at construction: any commuting axes); a t = 0 circuit takes the Clifford path below.
    const bool ROT = rot_on_;
    // P3 S-a: a circuit without rotations, without extras, not recording (a serial run): the identity plan's record rows
    // are then ρ = κ = 0 with a zero base row, so a fired shot it serves is t(P) alone. `idz` is its header once served
    // resident with those rows; later shots skip the lookup and the (zero) rows and keep the second-chance mark. A
    // non-resident header (evicted under a budget) takes the lookup / re-supply path below; the window keeps its oracle.
    const CachedPlan* idz = nullptr;
    const bool idz_on = !LANE && !ROT && nextra_ == 0 && !famrec && !retain_state;
    auto idz_take = [&](const CachedPlan& p) {
        if (!idz_on || idz || p.fallback || !p.emit_built || p.emit_rho != 0 || p.emit_kappa != 0) return;
        for (int w = 0; w < CW; ++w) if (p.emit_p[w]) return;
        idz = &p;
    };
    if constexpr (!LANE) { if (sc > 0 || famrec) ensure_rot_plists_(); }   // the window / the record writer read P
    // count_channels_: enable channel-accumulation when profiling, QEC_TW_COUNT is set,
    // OR when the selfcheck oracle is active (the C++ test uses channel_one_counts() as
    // a correctness gate in the same window). Fast path (sc==0, incl. spent): disabled.
    count_channels_ = BLK ? blk_->count_channels : (sc > 0);   // a lane counts as its home's run does
    // J-b1 P2: the event is always queued (no environment read in this TU; the Python layer drains it).
    if (sc > 0 && selfcheck_notice_once())   // S0d: XtimVerificationWarning
        diag_.push_back({TwirlRecordSampler::DiagEvent::SELFCHECK_WINDOW, "", sc, 0, ""});
    // S0(c) / D6: payload eviction happens ONLY here, at the shot boundary (before a shot
    // references any plan), never inside a shot. No budget ⇒ one register test per shot.
    const bool budget_on = !BLK && plan_budget_ != 0;   // a lane never evicts (its home does, at barriers)
    // termroute-a: the budget is on by default (kDefaultCacheBudget), so the boundary runs only when what it reads
    // moved since the last one: between two boundaries the payload bytes, the readers' bytes and the Born root memo
    // only grow (evictions and memo clears happen inside the boundary), so an unchanged triple means the boundary
    // would decide exactly as last time. Carried branches (their tries' memos) keep the per-shot call.
    const bool pb_every = budget_on && fam_has_branches_();
    size_t pb_res = SIZE_MAX, pb_rdr = 0, pb_mem = 0;      // SIZE_MAX: the first shot of a call always runs it
    // P3 S-b: this run's shots come from shot blocks (a serial run outside the window, not recording; sb_ok_). The
    // window's shots, a recording run and every lane / deferred shot keep the event list (their oracles / record
    // writers read the shot's events).
    // M3: the route is decided per batch: the window's shots (checked < sc) come from the event list, every later batch
    // of the same call from shot blocks (lean N1: both read the same noise block of the run's table: no byte moves).
    bool SB = false, SB_ok = false;
    [[maybe_unused]] const NoiseTable* NT = nullptr;   // lean N1: this run's table (a lane / deferred run: its block source)
    if constexpr (!BLK) {
        NT = &run_table_();
        SB_ok = !famrec && sb_ok_();
        if (SB_ok) sb_size_();
    }
    const int KR = (ROT ? rot.TW : rot_rec.TW) + 1;
    // FCS fix 8: the pre-pass can reject only when the column read has flagged channels or a clean shot can fail
    [[maybe_unused]] bool er_pre = false;
    if constexpr (ER) er_pre = er_.p->anyF || er_cc == 2;
    // a speed policy only (the verdicts are the shot loop's either way): the pass runs on a block while the previous
    // block rejected at least half of its shots early (a kept shot redraws its readout flips and reruns the test)
    [[maybe_unused]] bool pre_on = true;
    [[maybe_unused]] long early_at = 0;
    // FCS fix 4: the frames class is a constant of the run (fam_frame_take_ returns at once without it): no per-shot call
    [[maybe_unused]] const bool frames_on = fam_frames_on_();
    auto t0 = clk::now();
    while (BLK || done < shots) {
        int S;
        if constexpr (BLK) {
            if (!lane_take_block_(S, done)) break;     // the next block: its batch slice, shot index, sink row
        } else {
        // threads T2: the serial window prefix of a threaded call stops at the first batch boundary after the window
        if (XTIM_UNLIKELY(window_stop_) && checked >= sc) break;
        auto td = clk::now();
        SB = SB_ok && checked >= sc;
        // lean N1: a batch never crosses a noise block (blocks aligned at multiples of 4096 of the cumulative shot index)
        const uint64_t c0 = done_shots_;
        S = (int)std::min<long>(NoiseModel::kB - (long)(c0 & (NoiseModel::kB - 1)), shots - done);
        if (SB) {
            S = std::min(S, sb_B_);
            sb_fill_(*NT, c0, S);
            sb_shots_ += S;
            if constexpr (ER) early_at = er_.p->early;
            if constexpr (ER) if (er_pre && pre_on) {
                // FCS fix 8: the block's early-reject verdicts in one tight pass (the same tests on the same rows, readout
                // flips and key words as the shot loop's: er_clean_ / er_rot_reject_ / er_cliff_reject_ at the shot's
                // cumulative index). A rejected shot touches nothing else (no plan, no build, no counter but early / diag),
                // so the shot loop below skips it at its top; a kept shot runs as before (its readout flips redrawn).
                for (int s = 0; s < S; ++s) {
                    shot_ = c0 + (uint64_t)s;
                    if (nrf) draw_rf();
                    uint64_t* row = &sb_rows_[(size_t)s * (size_t)CW];
                    uint8_t v = 0;
                    if (sb_nev_[(size_t)s] == 0) { if (er_cc == 2 && er_clean_() == 1) v = 1; }
                    else if (ROT ? er_rot_reject_(row, &sb_key_[(size_t)s * (size_t)KR]) : er_cliff_reject_(row)) v = 2;
                    sb_rej_[(size_t)s] = v;
                    if (v) for (int w2 = 0; w2 < CW; ++w2) row[w2] = 0;   // round 2 (b): consumed (the loop skips it)
                }
            }
        } else {
        if (SB_ok) S = (int)std::min<long>(S, std::max<long>(sc - checked, 256));   // M3: the window ends near a block end
#ifdef XTIM_BENCH_REPLAY
        if (bench_src_) bench_fill_(done, S); else
#endif
        nb_events_(*NT, c0, S, batch);
        }
        t_draw += std::chrono::duration<double>(clk::now() - td).count();
#ifdef XTIM_BENCH_REPLAY
        if (bench_rec_on_) bench_append_();
#endif
        }
        auto tl = clk::now();
        (void)tl;
        for (int s = 0; s < S; ++s, ++done) {
            if constexpr (ER) {                        // FCS fix 8: rejected by the block pre-pass (its index spent)
                if (SB && er_pre && pre_on && sb_rej_[(size_t)s]) {
                    ++done_shots_;
                    if (sb_rej_[(size_t)s] == 2) ++diag;
                    XTIM_ER_REJECT_();
                    continue;
                }
            }
            if constexpr (BLK) lane_shot_begin_(s);    // the shot's global row (and the last shot's emitted flag)
            if (XTIM_UNLIKELY(budget_on)) {
                const size_t r = cache->resident_bytes(), rb = reader_bytes_.load(std::memory_order_relaxed),
                             mb = trie_.root_memo_bytes();
#ifndef XTIM_MUTATE_CB_SKIP_BOUNDARY
                if (r != pb_res || rb != pb_rdr || mb != pb_mem || pb_every) {
#else
                if (pb_res == SIZE_MAX) {       // mutation build only (MUT-CB-SKIP-BOUNDARY): first shot only
#endif
                    payload_boundary_();
                    pb_res = cache->resident_bytes();
                    pb_rdr = reader_bytes_.load(std::memory_order_relaxed);
                    pb_mem = trie_.root_memo_bytes();
                }
            }
            // P3 S-b: a shot-block shot has e0 = 0, e1 = its event count (only e0 == e1 is read: the event list is not)
            const int32_t e0 = SB ? 0 : batch.shot_off[s], e1 = SB ? sb_nev_[(size_t)s] : batch.shot_off[s + 1];
            [[maybe_unused]] bool er_pass = false;   // early-reject: the shot provably passes (no late filter)
            // S0(b): this shot's keyed streams. shot_ is the CUMULATIVE index (seed=None continues
            // it across run() calls; set_seed() / a fresh Impl reset it to 0). The coin reservoir
            // is positioned here (it never carries bits across shots); the gauge words when gauge
            // detectors exist; every other stream is positioned at its first use in the shot.
            shot_ = done_shots_++;
            coins_.at(shot_);
#ifdef XTIM_MUTATE_Q7_COIN_DRAW
            if (oracle_all_shots_) (void)coins_.next();   // mutation build only (Q7)
#endif
            if (SB) {                                  // P3 S-b: the shot's rows (t(P) channel / extras words)
                uint64_t* row = &sb_rows_[(size_t)s * (size_t)CW];   // round 2 (b): moved out (the row left zero)
                for (int w2 = 0; w2 < CW; ++w2) { bits[w2] = row[w2]; row[w2] = 0; }
                if (nextra_) for (int w2 = 0; w2 < XW; ++w2) xbits_[(size_t)w2] = sb_xb_[(size_t)s * (size_t)XR + (size_t)w2];
            } else {
            for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
            if (nextra_) for (int w2 = 0; w2 < XW; ++w2) xbits_[(size_t)w2] = 0;   // S2b extras
            }
            rec_route_ = false;
            in_window_ = checked < sc;                 // perf-engine §2.6: this shot's window flag, before any build
            // J-a (N4b) / J-b1 P9b: the frame and the shot's branch. threads T5: a lane defers a framed or carried-branch
            // shot (the term route, the branch's payload and trie are the home's).
            if constexpr (LANE) { if (XTIM_UNLIKELY(fam_lane_frame_defer_(done))) { lane_defer_(); continue; } }
            else if (frames_on) fam_frame_take_(done);
            if (nrf) draw_rf();                    // Stim M(p) readout coins (every shot)
            // J-a (N4b): a framed shot of the frames class: the frame atoms and the fired alternatives
            // composed by the term route (any kind, clean shots included); the shot joins the ONE tail.
            // N5: on a recording run (a layout-3 record sink) EVERY shot takes the term route, which writes
            // the record.
            if (fam_term_all_()) {
                if (!fam_term_shot_(batch, e0, e1, bits)) return false;
                if constexpr (ER) if (er_.p->rejected) { er_.p->rejected = false; XTIM_ER_REJECT_(); continue; }
                goto shot_tail_;
            }
            if (ROT) {                                 // lean D1-S3: every shot of a circuit with rotations (any commuting axes)
                if (e0 == e1) {
                    if constexpr (ER) {                // early-reject: a clean shot (C = I, P = I): the readout flips alone
                        if (er_cc == 2) {
                            const int v = er_clean_();
                            if (v == 1) { XTIM_ER_REJECT_(); continue; }
                            er_pass = v == 0;
                        } else {
                            er_pass = er_cc == 0;
                        }
                    }
                    ++clean;
                    if (rot_id_skip_ && !famrec && !(checked < sc)) {   // the s = 0 plan emits nothing (ne = ρ = κ = 0)
                        rec_route_ = true;
                        rec_uz_width_ = 0;
                        rec_uz_.clear();
                        goto shot_tail_;
                    }
                } else {
                    ++diag;
                }
                const int rr = rot_shot_<FWE>(bits, batch, e0, e1, done, sc, famrec,
                                              SB ? &sb_key_[(size_t)s * (size_t)KR] : nullptr);
                if (rr < 0) return false;              // the gauge-map oracle tripped (ORACLE_FAILURE queued)
                if constexpr (ER) { if (rr > 0) { XTIM_ER_REJECT_(); continue; } }
                goto shot_tail_;
            }
            // lean D1-S3: a circuit WITHOUT rotations (t = 0, Clifford). Every fired alternative is its Pauli frame (S_f = ∅),
            // so a shot is P = ⊕ P_f on the bare state: the identity plan's record rows ⊕ t(P). t(P)'s channel / extras words
            // are the rotation records' (TW = 0); P itself (a recording run's P_rec, the window's oracle) is ⊕ of the sparse P
            // supports. The identity plan is the diagonal cache's zero key, looked up / built / evicted / counted exactly as
            // the deleted streaming / structural paths did on a Clifford circuit (their key was always zero there).
            if (e0 == e1) {
                if constexpr (ER) {                    // early-reject: a clean shot (C = I, P = I)
                    if (er_cc == 2) {
                        const int v = er_clean_();
                        if (v == 1) { XTIM_ER_REJECT_(); continue; }
                        er_pass = v == 0;
                    } else {
                        er_pass = er_cc == 0;
                    }
                }
                ++clean;
                // S2b: a clean shot is the identity plan's record rows (R = I, t(P) = 0). A circuit
                // without extras has an all-deterministic identity plan (every channel at its
                // noiseless value, internal bit 0) — nothing to draw, nothing to XOR.
                if (nextra_ > 0 || famrec) {
                    const CachedPlan& p0 = identity_plan_();
                    hit_mark_<LANE>(p0);               // §2 R2: served
                    record_rows_any_(p0, bits);
                    if (famrec) {                      // J-b1 P4: R = I, P_rec = I (the term route's S = ∅ record)
                        if (fast_prefix_.n != n) fast_prefix_ = Pauli(n);
                        else {
                            std::fill(fast_prefix_.x.begin(), fast_prefix_.x.end(), 0ULL);
                            std::fill(fast_prefix_.z.begin(), fast_prefix_.z.end(), 0ULL);
                            fast_prefix_.phase = 0;
                        }
                        fam_rec_write_d_(p0, fast_prefix_);
                    }
                }
            } else {
                ++diag;
                if (!SB) {   // t(P): the rotation records' channel and extras words (a shot-block shot has them in its rows)
                    const RotationRecords& RR = rot_rec;
                    const size_t xo = (size_t)2 * RR.TW + 2, hdr = xo + (size_t)2 * XR + 1;
                    for (int32_t e = e0; e < e1; ++e) {
                        const uint32_t* B = &RR.rec[RR.off[(size_t)batch.ev[e]]];
                        for (int w2 = 0; w2 < XR; ++w2)
                            xbits_[(size_t)w2] ^= (uint64_t)B[xo + 2 * w2] | ((uint64_t)B[xo + 2 * w2 + 1] << 32);
                        const uint32_t nz = B[hdr - 1];
                        const uint32_t* p = B + hdr;
                        for (uint32_t i = 0; i < nz; ++i, p += 3) bits[p[0]] ^= (uint64_t)p[1] | ((uint64_t)p[2] << 32);
                    }
                }
                // early-reject: C = I, so every flagged deterministic channel's value is final here
                if constexpr (ER) { if (er_cliff_reject_(bits)) { XTIM_ER_REJECT_(); continue; } }
                const bool in_window = checked < sc;
                const uint64_t* const k0 = cliff_key_.data();
                if (!in_window) {                      // a resident identity plan (the deleted streaming / accumulator hit)
                    if (idz && idz->resident && idz->emit_built) {   // P3 S-a: zero rows (record_rows_ with ρ = κ = 0)
                        hit_mark_<LANE>(*idz);
                        rec_uz_.clear();
                        rec_uz_width_ = -1;
                        rec_route_ = true;
                        goto shot_tail_;
                    }
                    const CachedPlan* ap = LANE ? lane_find_(k0, cliff_h_) : cache->find(k0, cliff_h_);
                    if (ap && !ap->fallback && ap->resident) ensure_rec_(*ap);   // cold: once per payload
                    if (ap && !ap->fallback && ap->resident) {
                        hit_mark_<LANE>(*ap);          // §2 R2: served
                        record_rows_any_(*ap, bits);
                        idz_take(*ap);
                        if (famrec) {                  // J-b1 P4: P_rec = P (the accumulator route's attribution)
                            rot_P_of_events_(batch.ev.data() + e0, (int)(e1 - e0), fast_prefix_);
                            fam_rec_write_d_(*ap, fast_prefix_);
                            ++route_[ap->emit_kappa > 0 ? R_RECORD_KPOS : R_FAST_ACC];
                        } else if (ap->emit_kappa > 0) {
                            ++route_[R_RECORD_KPOS];   // a κ_rec = 0 hit is the uncounted streaming remainder (`fast`)
                        }
                        if (has_born_dec_) ++fast_born_shots_;
                        goto shot_tail_;
                    }
                }
                // a miss (the payload not resident) or the selfcheck window: the identity plan built / re-supplied
                ++structural_shots_;
                trie_.root_memo_verify = in_window;     // S8-4: root-memo hits recomputed inside the window
                bool created = false;                  // plancache-design O1: a header made in the window is checked
                const CachedPlan* planp = LANE ? lane_get_or_build_(k0, cliff_h_, cliff_nf_)
                                               : cache->get_or_build(G, bare, k0, cliff_h_, cliff_nf_, in_window, &created, shot_);
                if (!planp) {      // S0d: ORACLE_FAILURE -> XtimCorrectnessWarning (Python layer)
                    diag_.push_back({TwirlRecordSampler::DiagEvent::ORACLE_FAILURE, "plan_key", done, 0,
                                     "twirl_records PLAN KEY FAIL shot=" + std::to_string(done) +
                                     ": the identity key disagrees with its normal form (engine bug)"});
                    return false;
                }
                const CachedPlan& plan = *planp;
                hit_mark_<LANE>(plan);                 // §2 R2: served (the cold path)
                if (XTIM_UNLIKELY(plan.fallback)) fam_plan_fallback_bug_();
                if (in_window) {                       // the CLIFFORD ORACLE (selfcheck window): t(P) against (D)
                    if (created) class_rep_check_(plan);   // O1: the key's own table
                    if (!cliff_window_check_(batch, e0, e1, done)) return false;
                    ++checked;
                }
                record_rows_any_(plan, bits);
                idz_take(plan);
                if (famrec) {
                    rot_P_of_events_(batch.ev.data() + e0, (int)(e1 - e0), fast_prefix_);
                    fam_rec_write_d_(plan, fast_prefix_);
                }
                ++route_[plan.emit_kappa > 0 ? R_RECORD_KPOS : R_SLOW_K0];
            }
            shot_tail_:   // S5a: the accumulator fast path joins here (the ONE shot tail)
            // S2b: on the record route the Born observable is its extras column (RAW ⊕ t(P)); the
            // record invert is applied here, the readout flip / frame below (as on every route).
            if (born_obs && rec_route_) obs_bit_cur = (int)(xbit_(obs_col_) ? 1 : 0) ^ born_inv_;
            if (XTIM_UNLIKELY(sem_left > 0)) {
                for (int w2 = 0; w2 < CW; ++w2) sem_raw_bits_[w2] = bits[w2];
                obs_bit_cur_raw_ = (obs_bit_cur ^ born_inv_) & 1;
            }
            apply_rf(bits);
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
            // Task 2b.1: the declared decisions' bits, AFTER the residual/channel bits are final.
            const uint8_t* dptr = has_decisions_ ? emit_decisions(bits) : nullptr;   // perf-glue §1.2: no call without decisions
            if (XTIM_UNLIKELY(sem_left > 0)) {
                --sem_left;
                if (!semantic_oracle_fam_(batch, e0, e1, done)) return false;
            }
            XTIM_EMIT_SHOT(bits, obs_bit_cur, dptr);
            if constexpr (ER) if (!er_pass) er_late_();
        }
        if constexpr (ER) if (SB && er_pre) { pre_on = (er_.p->early - early_at) * 2 >= (long)S; early_at = er_.p->early; }
        if (SB) sb_clean_ = std::max(sb_clean_, S);   // round 2 (b): every row of the block was consumed (moved out)
    }
    if (budget_on) payload_boundary_();            // between calls is a shot boundary too
    auto t1 = clk::now();
    wall += std::chrono::duration<double>(t1 - t0).count();
    last_run_done_ = done;                         // threads T2: the shots this call ran (the window prefix stops early)
    return true;
}

#undef XTIM_EMIT_SHOT
#undef XTIM_ER_REJECT_

}  // namespace qeccore
