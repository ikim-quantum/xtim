// twirl_sampler_rotloop.inc.hpp — INTERNAL to the twirl sampler (lean D1-S2, 2026-11-05): the ROTATION SHOT LOOP,
// TwirlRecordSampler::Impl::rot_shot_<FWE>, included by twirl_sampler_runw.inc.hpp (after its macros) and instantiated
// with run_w's instances. OWNER-SIGNED stream change on T circuits (lean/D1/S2/plan.md §1 "sample()"):
//   per shot (the fired alternatives in time order): s ^= S_f (the key: ⌈t/64⌉ words), h ^= token (plan_key_hash is
//   GF(2)-linear: h = plan_key_hash(s)), bits ^= the channel words of t(P_f), xbits ^= the extras word; ONE lookup on s
//   (a lane: the home's store read only, then its overlay); the postselection COLUMN READ (the flagged channels no set
//   rotation touches: a function of the shot alone — a reject needs no plan); a miss builds the key's sparse plan
//   (RotBuilder: the t-bit table over the touched columns, DESIGN §1 "per new key") into the store; coins = the keyed COIN
//   bits in pivot order; z = BornTrie::walk on the plan's TUPLE trie (keyed by the ordered direction tuple's content; the
//   root memo as today); emission = t(P) words ⊕ Σ entries (flip ⊕ A·u ⊕ B·z) into the channel words / the extras word;
//   a recording run writes the layout-3 record (content (2, s), P_rec = ⊕ P_f from the sparse P supports, y = the emission's
//   own values, u then z); the selfcheck window runs the gauge-map oracle (rot_window_check_). The shot then takes the ONE
//   shot tail of run_w (rf, the Born observable / decisions, emit). Returns 0 (emitted), 1 (rejected early), -1 (an oracle
//   failure: run_w returns false).
#pragma once

namespace qeccore {

template <int FWE>
__attribute__((noinline)) int TwirlRecordSampler::Impl::rot_shot_(uint64_t* bits, const DiagErrorSampler::SparseBatch& b,
                                                                 int32_t e0, int32_t e1, long done, long sc, bool famrec,
                                                                 const uint64_t* sbk) {
    constexpr bool ER = (FWE & kErRun) != 0;
    constexpr bool LANE = (FWE & kLaneRun) != 0;
    const int TW = rot.TW, XR = XR_;
    const RotationRecords& R = rot_rec;
    uint64_t* const sw = rot_s_.data();
    for (int w = 0; w < TW; ++w) sw[w] = 0;
    uint64_t hh = 0xcbf29ce484222325ull;
    const size_t hdr = (size_t)2 * TW + 2 + (size_t)2 * XR + 1;
    auto v64 = [](const uint32_t* q) { return (uint64_t)q[0] | ((uint64_t)q[1] << 32); };
    const RotPlanStore* HS = LANE ? rot_home_ : nullptr;
    size_t o = 0;
    bool home = false;
    if (e0 == e1) {                                   // a clean shot (s = 0, h = C, t(P) = 0): the s = 0 plan built at
        o = rot_id_o_;                                // construction (a lane: the home's); the column read is run_w's
        home = LANE;                                  // er_clean_ (no rotation set: the same test)
    } else if (sbk) {                                 // P3 S-b: the shot block's key row (bits / xbits_ are the shot's rows)
        for (int w = 0; w < TW; ++w) sw[w] = sbk[w];
        hh = sbk[TW];
        __builtin_prefetch(HS ? HS->slot_addr(hh) : rstore_->slot_addr(hh));
    } else {
        // pre-pass: the tokens (one line per record; the DRAM misses issue in parallel), then the index slot
        for (int32_t e = e0; e < e1; ++e) {
            const uint32_t* B = &R.rec[R.off[(size_t)b.ev[e]]];
            __builtin_prefetch(B + 16);
            hh ^= v64(B + 2 * TW);
        }
        __builtin_prefetch(HS ? HS->slot_addr(hh) : rstore_->slot_addr(hh));
        for (int32_t e = e0; e < e1; ++e) {
            const uint32_t* B = &R.rec[R.off[(size_t)b.ev[e]]];
            for (int w = 0; w < TW; ++w) sw[w] ^= v64(B + 2 * w);
            for (int w = 0; w < XR; ++w) xbits_[(size_t)w] ^= v64(B + 2 * TW + 2 + 2 * w);
            const uint32_t nz = B[hdr - 1];
            const uint32_t* p = B + hdr;
            for (uint32_t i = 0; i < nz; ++i, p += 3) bits[p[0]] ^= v64(p + 1);
        }
    }
    if (e0 != e1) {
        if constexpr (ER) {
            // early-reject (lean D1-S2; review M3): the COLUMN READ, a function of the shot alone (never of the store's state:
            // threads N == 1, warm == cold). t = (bits ⊕ K ⊕ rf) & F over the flagged deterministic channels that NO set rotation
            // touches (a touched channel's value needs the plan: the late filter decides it). An untouched channel has no plan
            // entry, so its emitted value is already final here; a remaining 1 rejects with no plan work. counts["early"] = the
            // shots rejected here (+ clean shots rejected by their readout flips in run_w).
            if (er_rot_reject_(bits, sw)) return 1;   // FCS fix 8: the one column read (also the block pre-pass's)
        }
        if (HS) { o = HS->find(sw, hh); home = o != 0; }
        if (!o) o = rstore_->find(sw, hh);
        if (!o) o = rot_miss_(sw, hh, LANE);              // a lane: into its overlay (home stays false)
    }
    const RotPlanStore* ps = home ? HS : rstore_.get();
    const uint64_t* P = ps->plan(o);
    const int ne = ps->ne(P), rho = ps->rho(P), kap = ps->kappa(P);
    const int AW = ps->AW(P), BW = ps->BW(P), es = 1 + AW + BW;
    // the coins: the keyed COIN bits in pivot order (the outcomes u kept in rec_uz_)
    uint64_t* const uw = rot_u_.data();
    for (int w = 0; w < AW; ++w) uw[w] = 0;
    rec_uz_.clear();
    for (int k = 0; k < rho; ++k) {
        const bool bit = coins_.next();
        rec_uz_.push_back((uint8_t)bit);
        if (bit) uw[k >> 6] |= 1ULL << (k & 63);
    }
    // the Born outcomes: the tuple trie's walk on the keyed BORN words
    uint64_t* const zw = rot_z_.data();
    for (int w = 0; w < BW; ++w) zw[w] = 0;
    if (kap > 0) {
        const int bad = rot_walk_(P, ps, home, LANE);
        if (bad >= 0) throw std::logic_error("rot_shot_: a non-Hermitian record direction (engine bug)");
        for (int j = 0; j < kap; ++j)
            if (rec_uz_[(size_t)rho + (size_t)j]) zw[j >> 6] |= 1ULL << (j & 63);
    }
    // the emission over the plan's entries
    {
        const uint64_t* En = P + ps->entries_at(P);
        for (int i = 0; i < ne; ++i, En += es) {
            const uint64_t x = *En;
            int v = RotPlanStore::ent_flip(x);
            const uint64_t* A = En + 1;
            for (int w = 0; w < AW; ++w) v ^= __builtin_popcountll(A[w] & uw[w]) & 1;
            const uint64_t* Bw = A + AW;
            for (int w = 0; w < BW; ++w) v ^= __builtin_popcountll(Bw[w] & zw[w]) & 1;
            if (!v) continue;
            const int c = RotPlanStore::ent_col(x);
            if (c < nchan) bits[c >> 6] ^= 1ULL << (c & 63);
            else { const int ex = c - nchan; xbits_[(size_t)(ex >> 6)] ^= 1ULL << (ex & 63); }
        }
    }
    rec_route_ = true;
    rec_uz_width_ = rho + kap;
    if (e1 > e0) {                                    // route attribution (a clean shot is counted as `clean` by run_w)
        if (kap > 0) ++route_[R_RECORD_KPOS];
        else if (famrec) ++route_[R_FAST_ACC];
        if (has_born_dec_) ++fast_born_shots_;
    }
    if (famrec) rot_rec_write_(P, bits, b.ev.data() + e0, (int)(e1 - e0), home);
    if (XTIM_UNLIKELY(checked < sc)) {
        if (!rot_window_check_(b, e0, e1, done, P, ps, hh)) return -1;
        ++checked;
    }
    return 0;
}

}  // namespace qeccore
