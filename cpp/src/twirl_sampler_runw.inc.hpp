// twirl_sampler_runw.inc.hpp — INTERNAL to the twirl sampler (J-a r2.1, JA-1): the shot loop,
// TwirlRecordSampler::Impl::run_w<FW>, included by twirl_sampler_fam.cpp only, which instantiates it (J-b2 J3/J4: the
// legacy instance and its source are gone; J9: the always-true FAM parameter is dropped). Include after
// twirl_sampler_impl.inc.hpp.
#pragma once

#include "twirl_sampler_impl.inc.hpp"

namespace qeccore {

// J-a (N1): the emit seam — emit_shot_fam_ (the FamilySink; J-b2 J6b deleted the two legacy sinks).
#define XTIM_EMIT_SHOT(...) \
    do { emit_shot_fam_(__VA_ARGS__); } while (0)

template <int FW>
bool TwirlRecordSampler::Impl::run_w(long shots) {
    // S0(b): the keyed per-shot streams address draw (shot·2^20 + ctr) — injective (never a repeated
    // draw) only for shot < 2^44 per seed and < 2^20 draws per (shot, stream). Refused by name.
    {
        const uint64_t per_shot = std::max<uint64_t>({(uint64_t)nrf, (uint64_t)ng, (uint64_t)born_dec.size()});
        if (per_shot >= krng::kMaxCtr)
            throw std::invalid_argument("twirl sampler: a shot needs " + std::to_string(per_shot) +
                " draws of one keyed stream (noisy records / certified generators / Born decisions / "
                "gauge detectors); the keyed stream layout allows < 2^20 per shot");
        if (shots > 0 && (done_shots_ >= krng::kMaxShots || (uint64_t)shots > krng::kMaxShots - done_shots_))
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
    born_node_event_ = false;                             // decision 7: one BORN_NODE event per run()
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
    const long sc = oracle_all_shots_ ? (1L << 60) : (selfcheck_spent_ ? 0 : selfcheck);
    if (shots > 0) selfcheck_spent_ = true;
    sem_budget_ = (sc > 0 && (born_obs || has_born_dec_) && !ppr_tables) ? std::min<long>(sc, 64) : 0;
    // J-a (N1): the named refusals of a FAM run (σ export / retention, frames on family extras), and
    // the semantic oracle over the family extras too (every shot under oracle_all_shots, Q7).
    if (!fam_run_guard_(shots)) return false;
    // N4a: the term-routed shots of a CH/PPR circuit have an oracle too (J-b1 P2: a FAM sampler on a CH/PPR
    // circuit that gets past fam_run_guard_ has the ppr_records class).
    if (sc > 0 && (oracle_all_shots_ || nextra_ > 0))
        sem_budget_ = oracle_all_shots_ ? sc : std::min<long>(sc, 64);
    long sem_left = sem_budget_;                             // register-resident window count
    // J-b1 P4: a FAM recording run serves its diagonal shots on the (D) paths, which write the layout-3 record
    // (fam_rec_write_d_); the term route keeps the PPR-class, route-1 and framed shots. Per run, register-resident.
    const bool famrec = fam_rec_d_();
    // The streaming abelian path's FAM condition, hoisted: inline in the shot-loop condition (`nextra_ == 0 &&
    // !famrec`) it cost +58 Ir per shot on cultivation d5 sample mode (P4 forecast; register pressure in run_w).
    const bool fam_stream_ok = nextra_ == 0 && !famrec;
    // count_channels_: enable channel-accumulation when profiling, QEC_TW_COUNT is set,
    // OR when the selfcheck oracle is active (the C++ test uses channel_one_counts() as
    // a correctness gate in the same window). Fast path (sc==0, incl. spent): disabled.
    count_channels_ = (sc > 0);
    // J-b1 P2: the event is always queued (no environment read in this TU; the Python layer drains it).
    if (sc > 0 && selfcheck_notice_once())   // S0d: XtimVerificationWarning
        diag_.push_back({TwirlRecordSampler::DiagEvent::SELFCHECK_WINDOW, "", sc, 0, ""});
    // S0(c) / D6: payload eviction happens ONLY here, at the shot boundary (before a shot
    // references any plan), never inside a shot. No budget ⇒ one register test per shot.
    const bool budget_on = plan_budget_ != 0;
    // termroute-a: the budget is on by default (kDefaultCacheBudget), so the boundary runs only when what it reads
    // moved since the last one: between two boundaries the payload bytes, the readers' bytes and the Born root memo
    // only grow (evictions and memo clears happen inside the boundary), so an unchanged triple means the boundary
    // would decide exactly as last time. Carried branches (their tries' memos) keep the per-shot call.
    const bool pb_every = budget_on && fam_has_branches_();
    size_t pb_res = SIZE_MAX, pb_rdr = 0, pb_mem = 0;      // SIZE_MAX: the first shot of a call always runs it
    auto t0 = clk::now();
    while (done < shots) {
        const int S = (int)std::min<long>(4096, shots - done);
        auto td = clk::now();
#ifdef XTIM_BENCH_REPLAY
        if (bench_src_) bench_fill_(done, S); else
#endif
        smp->sample_events(S, batch);
        t_draw += std::chrono::duration<double>(clk::now() - td).count();
#ifdef XTIM_BENCH_REPLAY
        if (bench_rec_on_) bench_append_();
#endif
        auto tl = clk::now();
        (void)tl;
        for (int s = 0; s < S; ++s, ++done) {
            if (__builtin_expect(budget_on, 0)) {
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
            const int32_t e0 = batch.shot_off[s], e1 = batch.shot_off[s + 1];
            // S0(b): this shot's keyed streams. shot_ is the CUMULATIVE index (seed=None continues
            // it across run() calls; set_seed() / a fresh Impl reset it to 0). The coin reservoir
            // is positioned here (it never carries bits across shots); the gauge words when gauge
            // detectors exist; every other stream is positioned at its first use in the shot.
            shot_ = done_shots_++;
            coins_.at(shot_);
#ifdef XTIM_MUTATE_Q7_COIN_DRAW
            if (oracle_all_shots_) (void)coins_.next();   // mutation build only (Q7)
#endif
            for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
            if (nextra_) for (int w2 = 0; w2 < XW; ++w2) xbits_[(size_t)w2] = 0;   // S2b extras
            rec_route_ = false;
            fam_frame_take_(done);   // J-a (N4b) / J-b1 P9b: the frame and the shot's branch
            if (nrf) draw_rf();                    // Stim M(p) readout coins (every shot)
            // J-a (N4b): a framed shot of the frames class: the frame atoms and the fired alternatives
            // composed by the term route (any kind, clean shots included); the shot joins the ONE tail.
            // N5: on a recording run (a layout-3 record sink) EVERY shot takes the term route, which writes
            // the record.
            if (fam_term_all_()) {
                if (!fam_term_shot_(batch, e0, e1, bits)) return false;
                goto shot_tail_;
            }
            if (e0 == e1) {
                ++clean;
                // S2b: a clean shot is the identity plan's record rows (R = I, t(P) = 0). A circuit
                // without extras has an all-deterministic identity plan (every channel at its
                // noiseless value, internal bit 0) — nothing to draw, nothing to XOR.
                if (nextra_ > 0 || famrec) {
                    const CachedPlan& p0 = identity_plan_();
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
                        // J-a (N4a): the kind-2 route (ppr_records class): the term-indexed residual's plan (kind 0:
                        // the diagonal cache's; kind 2: (G)), its record rows ⊕ t(P); the shot joins the ONE shot
                        // tail. J-b1 P2: the ONLY route of a PPR shot (a running sampler on a CH/PPR circuit has the
                        // ppr_records class: fam_run_guard_). J-b2 J4: the PPR family's legacy route is deleted.
                        if (!fam_term_shot_(batch, e0, e1, bits)) return false;
                        goto shot_tail_;
                    }
                }
                // ── ABELIAN FAST PATH: key by XOR of per-alt masks, emission by per-alt rows +
                // ordered-pair z-corrections. No accumulate, no content materialisation. Runs
                // outside the selfcheck window; misses fall through to the materialising path.
                // Born observables, retention and Born decisions are EXCLUDED: their record needs the
                // residual prefix P (⟨P, W⟩, σ = rows ⊕ plane1(P), the record's prefix words), which
                // this shortcut never builds (P.z is not linear in the fired alts). S5a: they take the
                // ACCUMULATOR fast path below, which has P.
                // Born fast-path (2026-08-08): born decisions are EXCLUDED from this streaming-blob
                // shortcut (it reconstructs channel bits from per-alt rows and never materialises the
                // residual PREFIX, which the born memo key needs). Born diagonal shots fall through to
                // the ACCUMULATOR fast path (~2680) which builds key_amask_s + the residual prefix in
                // accumulator coordinates — the born memo inputs — while still skipping the structural
                // canonicalize/twirl_collapse_record detour.
                // J-a (N1): a FAM sampler with extras never takes this route (no residual prefix: no t(P)).
                // J-b1 P4: nor does a FAM recording run (its record needs P_rec, the prefix).
                if (!born_obs && checked >= sc && !has_born_dec_ && fam_stream_ok) {
                    // One streaming pass over the fired alts' blobs: key XOR + emission row +
                    // linearized pair term (S_acc/V_acc running masks) in a single block visit.
                    // Pre-pass: XOR the per-alt hash tokens (GF(2)-linear hash ⇒ shot hash is
                    // token-XOR). This touches each blob's first line — all DRAM misses issue
                    // in PARALLEL — and lets us prefetch the index slot before assembling the key.
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
                    __builtin_prefetch(cache->slot_addr(hh));
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
                    const CachedPlan* ap = cache->find(ab_kw.data(), hh);
                    if (ap && !ap->fallback && ap->resident) ensure_rec_(*ap);   // cold: once per payload
                    if (ap && !ap->fallback && ap->resident && ap->emit_kappa == 0) {
                        // The canonical coins (COIN bit k = pivot record k) through THE record
                        // emission. This route excludes Born observables and Born decisions, so the
                        // extras' words it XORs into xbits_ (reset per shot) are never read here.
                        record_rows_any_(*ap, bits);
                        // The ordered-pair z-corrections draw nothing: after the coins, byte-identical.
                        if (multi)
                            for (int w2 = 0; w2 < NW; ++w2) { uint64_t word = ab_T[w2];
                                while (word) { const int q = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                    const uint64_t* row = &pzrow[(size_t)q * CW];
                                    for (int w3 = 0; w3 < CW; ++w3) bits[w3] ^= row[w3]; } }
                        apply_rf(bits);
                        ++used;
                        if (count_channels_) { for (int w2 = 0; w2 < CW; ++w2) { uint64_t word = bits[w2];
                            while (word) { const int c = w2 * 64 + __builtin_ctzll(word); word &= word - 1;
                                ++chancnt[c]; } } }
                        XTIM_EMIT_SHOT(bits, 0, emit_decisions(bits));
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
                // The shot's canonical plan key (S0(c)): (key_amask_s ‖ czw & upair_keep_) — the
                // strip keeps exactly the pairs with two non-product endpoints, so this is cz_key's
                // bitset over upairs indexing (O(KW) words; the dirty words carry every set bit).
                for (int w2 = 0; w2 < NW; ++w2) ab_kw[w2] = key_amask_s[w2];
                for (int w2 = 0; w2 < PW; ++w2) ab_kw[(size_t)NW + w2] = 0;
                for (int w : w_dirty) ab_kw[(size_t)NW + w] = czw[w] & upair_keep_[w];
                const uint64_t key_h = lhash(ab_kw.data());
                // FAST PATH (no nf materialisation): canonical key + resident in-class plan, outside the
                // self-check window. Prefix supports straight from the dirty lists (+ the strip's zflip
                // corrections). S5a: it serves Born observables and the barrier record too (retained
                // shots: σ-4 and the record from the same prefix); only the oracle lever keeps a
                // retained shot on the exact engine. The shot then takes the shared tail below.
                // (S0(b): the route moves no byte — every route reads the same keyed draws.)
                const CachedPlan* fplan = nullptr;
                if (checked >= sc)
                    fplan = cache->find(ab_kw.data(), key_h);
                if (fplan && !fplan->fallback && fplan->resident) ensure_rec_(*fplan);   // cold: once
                // S2b: any in-class plan (κ_rec > 0 included: the ONE emission walks its record trie).
                if (fplan && !fplan->fallback && fplan->resident) {
                    for (int w2 = 0; w2 < CW; ++w2) bits[w2] = 0;
                    for (int q : v_dirty) if (v_acc[q]) {
                        const uint64_t* row = &pxrow[(size_t)q * CW];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    for (int q : a_dirty) if (a_acc[q] & 2) {
                        const uint64_t* row = &pzrow[(size_t)q * CW];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    for (int q : zflip) {                        // strip's −Z-partner corrections
                        const uint64_t* row = &pzrow[(size_t)q * CW];
                        for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2]; }
                    // S2b: the plan's record rows (canonical coins, the record trie walk) and,
                    // for the extras, t(P) = ⟨P, W_e⟩ from the residual prefix rebuilt in
                    // accumulator coordinates: prefix.x = v-support, prefix.z = (a&2) ⊕ zflip. This IS
                    // the structural path's canonicalized nf.prefix (canonicalize keeps x and flips z on
                    // an unstripped wire iff it has an odd number of −Z stripped partners: zflip), and
                    // (key_amask_s, cz_key) is the plan's canonical (a, cz) — the key find() matched.
                    // Gated by tests/test_s5a_fast_retention.py (fast ≡ structural on every record field).
                    record_rows_any_(*fplan, bits);
                    if (nextra_ || famrec) {
                        if (fast_prefix_.n != n) fast_prefix_ = Pauli(n);   // sized once, cleared in place
                        else {
                            std::fill(fast_prefix_.x.begin(), fast_prefix_.x.end(), 0ULL);
                            std::fill(fast_prefix_.z.begin(), fast_prefix_.z.end(), 0ULL);
                            fast_prefix_.phase = 0;
                        }
                        for (int q : v_dirty) if (v_acc[q]) fast_prefix_.setx(q);
                        for (int q : a_dirty) if (a_acc[q] & 2) fast_prefix_.flipz(q);
#ifndef XTIM_MUTATE_S5A_NO_ZFLIP
                        for (int q : zflip) fast_prefix_.flipz(q);   // (mutation S5A_NO_ZFLIP drops this line)
#endif
                        if (nextra_) ext_prefix_(fast_prefix_);
                        if (famrec) fam_rec_write_d_(*fplan, fast_prefix_);   // J-b1 P4
                    }
                    for (int q : a_dirty) { a_acc[q] = 0; in_a[q] = 0; }
                    std::fill(amask_acc.begin(), amask_acc.end(), 0ULL);
                    for (int q : v_dirty) { v_acc[q] = 0; in_v[q] = 0; }
                    for (int w : w_dirty) { czw[w] = 0; in_w[w] = 0; }
                    a_dirty.clear(); v_dirty.clear(); w_dirty.clear();
                    // Route attribution: the structural path's table (κ_rec = 0 retention: the remainder).
                    const bool kpos = fplan->emit_kappa > 0;
                    ++route_[kpos ? R_RECORD_KPOS : R_FAST_ACC];
                    if (has_born_dec_) ++fast_born_shots_;
                    goto shot_tail_;
                }
                // SLOW path: materialise the normal form (miss / canon-touched / self-check).
                ++structural_shots_;
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
                const bool in_window = checked < sc;    // S0(c): the key-derivation check window
                trie_.root_memo_verify = in_window;     // S8-4: root-memo hits recomputed inside the window
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
                }
                canonicalize_mod_stabilizers_inplace(nf, G);
                // S0(c): the closed-form key must encode EXACTLY the canonical normal form — an
                // engine-bug check on the key DERIVATION, run when a plan header is created and on
                // every shot of the self-check window (the same standard the abelian fast path's
                // key oracle uses). Collision safety is separate and unconditional: the lookup
                // compares the full stored key on every probe.
                const CachedPlan* planp = cache->get_or_build(G, bare, ab_kw.data(), key_h, nf, in_window);
                if (!planp) {      // S0d: ORACLE_FAILURE -> XtimCorrectnessWarning (Python layer)
                    diag_.push_back({TwirlRecordSampler::DiagEvent::ORACLE_FAILURE, "plan_key", done, 0,
                                     "twirl_records PLAN KEY FAIL shot=" + std::to_string(done) +
                                     ": the closed-form canonical key disagrees with canonicalize "
                                     "(engine bug)"});
                    return false;
                }
                const CachedPlan& plan = *planp;
                // ── S2b: ONE emission for every diagonal plan (retained or not): the plan's record rows
                // (canonical coins, the record trie walk) ⊕ t(P) (prefix rows for the channels,
                // ⟨P, W_e⟩ for the extras). Fallback / out-of-class plans (and the retained-route
                // oracle lever) take the per-shot exact engine.
                // J-b1 P2: a FAM plan never falls back (the slow path builds diagonal-class normal forms only, so
                // build_shot_law never sets fallback): an engine-bug invariant, out of line and cold.
                if (__builtin_expect(plan.fallback, 0)) fam_plan_fallback_bug_();
                {
                    // r7.4: the window flag computed before the increment (`checked <= sc` is true on EVERY
                    // shot at selfcheck = 0; in a55c1c0 this oracle was an else-branch on cold shots only)
                    if (in_window && plan.rec && plan.rec->built) {
                        // LOCAL INDEX ORACLE (selfcheck window): the bitmask index must resolve this
                        // shot's XOR key to exactly the plan the exact path produced.
                        for (int w2 = 0; w2 < KW; ++w2) ab_kw[w2] = 0;
                        for (int32_t e = e0; e < e1; ++e) {
                            const AltMasks& M = alt_masks[(size_t)batch.ev[e]];
                            for (int w2 = 0; w2 < NW; ++w2) ab_kw[w2]      ^= M.kmask[w2];
                            for (int w2 = 0; w2 < PW; ++w2) ab_kw[NW + w2] ^= M.czk[w2];
                        }
#ifdef XTIM_MUTATE_R74_INDEX_ORACLE_WRONG
                        if (cache->find(ab_kw.data(), lhash(ab_kw.data())) == &plan) {   // mutation build only (R74)
#else
                        if (cache->find(ab_kw.data(), lhash(ab_kw.data())) != &plan) {
#endif
                            diag_.push_back({TwirlRecordSampler::DiagEvent::ORACLE_FAILURE, "local_index",
                                             done, 0, "twirl_records LOCAL INDEX ORACLE FAIL shot=" +
                                             std::to_string(done)});   // S0d: XtimCorrectnessWarning
                            return false;
                        }
                    }
                    const bool do_check = checked < sc;
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
                    record_rows_any_(plan, bits);
                    if (nextra_) ext_prefix_(P);
                    if (famrec) fam_rec_write_d_(plan, P);   // J-b1 P4: P = the canonical nf.prefix
                    const bool kpos = plan.emit_kappa > 0;
                    ++route_[kpos ? R_RECORD_KPOS : R_SLOW_K0];
                    if (do_check) {
                        ++checked;
                        // ABELIAN ORACLE (bits): the per-alt-row + ordered-pair-T decomposition
                        // of the DETERMINISTIC channel bits must equal the nf-based emission
                        // (baseline ⊕ prefix-support rows ⊕ strip zflips) EXACTLY.
                        uint64_t* const abd = oracle_abd_.data();   // oracle window only (CW words)
                        uint64_t* const nfd = oracle_nfd_.data();
                        for (int w2 = 0; w2 < CW; ++w2) { abd[w2] = 0; nfd[w2] = 0; }
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
                        uint64_t* const lin = oracle_lin_.data();
                        for (int w2 = 0; w2 < CW; ++w2) { lin[w2] = 0; }
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
                }
            }
            shot_tail_:   // S5a: the accumulator fast path joins here (the ONE shot tail)
            // S2b: on the record route the Born observable is its extras column (RAW ⊕ t(P)); the
            // record invert is applied here, the readout flip / frame below (as on every route).
            if (born_obs && rec_route_) obs_bit_cur = (int)(xbit_(obs_col_) ? 1 : 0) ^ born_inv_;
            if (__builtin_expect(sem_left > 0, 0)) {
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
            const uint8_t* dptr = emit_decisions(bits);
            if (__builtin_expect(sem_left > 0, 0)) {
                --sem_left;
                if (!semantic_oracle_fam_(batch, e0, e1, done)) return false;
            }
            XTIM_EMIT_SHOT(bits, obs_bit_cur, dptr);
        }
    }
    if (budget_on) payload_boundary_();            // between calls is a shot boundary too
    auto t1 = clk::now();
    wall += std::chrono::duration<double>(t1 - t0).count();
    if (!disk_path.empty()) {
        if (cache->save_file(disk_path, G.identity_token())) fam_disk_saved_();   // J-b1 P2: an event, not stderr
    }
    return true;
}

#undef XTIM_EMIT_SHOT

}  // namespace qeccore
