#include "qeccore/diag_error_sampler.hpp"
#include <cstddef>

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace qeccore {

// (A polynomial fast-log for the skips was tried and REVERTED: glibc's std::log measures ~2.7 ns
// on this target — the atanh-series version with its serialized division was ~6x slower.)

// ---- xoshiro256** ----------------------------------------------------------------------------
static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

void DiagErrorSampler::reseed(uint64_t seed) {
    if (snap_ok_ && snap_seed_ == seed) {          // perf-glue §4.4: the state this seed derived (exact copy)
        std::memcpy(rs_, snap_rs_, sizeof rs_);
        next_ = snap_next_;
        return;
    }
    {
        // splitmix64 expansion of the seed into the xoshiro state.
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ull;
            uint64_t t = z;
            t = (t ^ (t >> 30)) * 0xbf58476d1ce4e5b9ull;
            t = (t ^ (t >> 27)) * 0x94d049bb133111ebull;
            rs_[i] = t ^ (t >> 31);
        }
    }
    // Draw each error's first firing index (the once-per-stream initial skip). p = 0 -> "never"
    // (a huge sentinel; the per-batch decrement of S leaves it astronomically positive).
    next_.resize(K_);
    for (int k = 0; k < K_; ++k) {
        if (skipc_[k] == 0.0) { next_[k] = (int64_t)1 << 62; continue; }
        if (skipc_[k] > 0.0) { next_[k] = 0; continue; }            // forced channel (p >= 1)
        // Clamp before the cast: for p < ~4e-18 the exact skip can exceed INT64_MAX (UB on cast).
        // A clamped skip of 2^62 shots is indistinguishable from "never" at any feasible shot count.
        const double u = runi_open();
        const double gp = std::log(u) * skipc_[k];                  // >= 0
        next_[k] = (gp < 4.0e18) ? (int64_t)gp : ((int64_t)1 << 62);
    }
    if (last_ok_ && last_seed_ == seed) {           // perf-glue §4.4: a repeated seed — keep its state
        std::memcpy(snap_rs_, rs_, sizeof rs_);
        snap_next_ = next_;
        snap_seed_ = seed;
        snap_ok_ = true;
    }
    last_seed_ = seed;
    last_ok_ = true;
}

inline uint64_t DiagErrorSampler::rnext() {
    uint64_t* s = rs_;
    const uint64_t r = rotl64(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl64(s[3], 45);
    return r;
}
inline double DiagErrorSampler::runi_open() {        // U in (0, 1]: log(U) is finite
    return ((rnext() >> 11) + 1) * 0x1.0p-53;
}

// ---- construction ----------------------------------------------------------------------------
DiagErrorSampler::DiagErrorSampler(int n, const std::vector<ErrorChannel>& channels, uint64_t seed)
    : n_(n), K_((int)channels.size()) {
    a_off_.push_back(0); cz_off_.push_back(0);
    altbase_.push_back(0); cumoff_.push_back(0);
    skipc_.reserve(K_); ptot_.reserve(K_);
    std::vector<int32_t> pair_of((size_t)n * n, -1);
    auto pair_index = [&](int q, int r) {
        if (q == r || q < 0 || r < 0 || q >= n || r >= n) throw std::invalid_argument("DiagErrorSampler: bad cz pair");
        int lo = q < r ? q : r, hi = q < r ? r : q;
        int32_t& idx = pair_of[(size_t)lo * n + hi];
        if (idx < 0) { idx = P_++; pair_list_.push_back({lo, hi}); }
        return idx;
    };
    for (const ErrorChannel& ch : channels) {
        // No alternative-count cap: ids are int32 and the conditional draw is a cumulative scan,
        // so an E/ELSE chain of any length is one channel (the former 255 guard was not
        // structural — nothing stored an alternative index in a byte).
        if (ch.alts.empty())
            throw std::invalid_argument("DiagErrorSampler: need at least one alternative per channel");
        double pt = 0.0;
        for (const ErrorChannel::Alt& alt : ch.alts) {
            if (!(alt.p >= 0.0)) throw std::invalid_argument("DiagErrorSampler: need p >= 0");
            for (auto [q, c] : alt.a)
                if (q < 0 || q >= n || (c & 3) == 0) throw std::invalid_argument("DiagErrorSampler: bad a entry");
            Rec r{-1, -1, -1, 0, 0, -1};
            if (alt.a.size() <= 2 && alt.cz.size() <= 1) {              // record-complete fast shape
                if (alt.a.size() >= 1) { r.q1 = alt.a[0].first; r.c1 = (uint8_t)(alt.a[0].second & 3); }
                if (alt.a.size() == 2) { r.q2 = alt.a[1].first; r.c2 = (uint8_t)(alt.a[1].second & 3); }
                if (!alt.cz.empty()) r.cz = pair_index(alt.cz[0].first, alt.cz[0].second);
            } else {                                                    // general shape: CSR fallback
                const size_t nf = a_off_.size() - 1;
                if (nf > 32760) throw std::invalid_argument("DiagErrorSampler: too many large-support alternatives");
                r.csr = (int16_t)nf;
                for (auto [q, c] : alt.a) { a_q_.push_back(q); a_c_.push_back((uint8_t)(c & 3)); }
                a_off_.push_back((int32_t)a_q_.size());
                for (auto [q, rr] : alt.cz) cz_idx_.push_back(pair_index(q, rr));
                cz_off_.push_back((int32_t)cz_idx_.size());
            }
            recs_.push_back(r);
            flips_.push_back(alt.flips);
            alt_k_.push_back((int32_t)(altbase_.size() - 1));
            pt += alt.p;
            cum_.push_back(pt);                                         // cumulative within channel
        }
        if (!(pt < 1.0 + 1e-9))
            throw std::invalid_argument("DiagErrorSampler: need sum of alternative p <= 1");
        altbase_.push_back((int32_t)recs_.size());
        cumoff_.push_back((int32_t)cum_.size());
        ptot_.push_back(pt);
        // Valid geometric skip constants are NEGATIVE (1/log1p(-p) < 0), so 0 and +1 are free
        // sentinels: 0 = never fires (p == 0); +1 = FORCED channel (p >= 1 up to rounding of
        // explicit p=1 alternatives — fires every shot, skip 0; the categorical draw still
        // selects among alternatives).
        skipc_.push_back(pt >= 1.0 ? 1.0 : (pt > 0.0 ? 1.0 / std::log1p(-pt) : 0.0));
    }
    reseed(seed);
}

// ---- event generation (geometric skips) -------------------------------------------------------
// next_[k] persists ACROSS batches: it holds error k's next firing index relative to the current
// batch start (geometric inter-arrival gaps are memoryless, so continuing the same skip sequence
// across the batch boundary keeps the per-shot law exactly Bernoulli(p_k) i.i.d.). A batch then
// costs one sequential scan of next_ plus one (log, RNG) pair per FIRED event — the per-error
// initial skip is drawn once at construction/reseed, not once per batch.
void DiagErrorSampler::gen_events(int S) {
    ev_ks_.clear();
    const int64_t S64 = S;
    for (int k = 0; k < K_; ++k) {
        int64_t s = next_[k];
        if (s < S64) {
            const double sc = skipc_[k];
            const int32_t base = altbase_[k];
            const int nalt = altbase_[k + 1] - base;
            do {
                uint64_t aid = (uint64_t)base;
                if (nalt > 1) {
                    // Conditional categorical draw, paid per FIRED event only: v uniform on
                    // (0, p_total]; alternative t iff cum[t-1] < v <= cum[t] ⇒ P = p_t/p_total.
                    const double v = runi_open() * ptot_[k];
                    const double* c = cum_.data() + cumoff_[k];
                    int t = 0;
                    while (t + 1 < nalt && v > c[t]) ++t;
                    aid += (uint64_t)t;
                }
                ev_ks_.push_back((aid << 32) | (uint32_t)s);
                if (sc > 0.0) { s += 1; continue; }   // forced channel: fires every shot
                // log(U) <= 0 and sc < 0 ⇒ the product is >= 0: truncation == floor. Clamped
                // before the cast (p_total < ~4e-18 could otherwise overflow int64 — UB).
                const double gp = std::log(runi_open()) * sc;
                s += 1 + ((gp < 4.0e18) ? (int64_t)gp : ((int64_t)1 << 62));
            } while (s < S64);
        }
        next_[k] = s - S64;
    }
}

// counting sort by shot (events arrive k-major, i.e. unsorted in s): sample_events exposes the buckets.
void DiagErrorSampler::sort_events(int S) {
    const int M = (int)ev_ks_.size();
    sort_off_.assign((size_t)S + 1, 0);
    sort_ev_.resize(M);
    sort_s_.resize(M);
    for (int i = 0; i < M; ++i) ++sort_off_[(uint32_t)ev_ks_[i] + 1];
    for (int s = 0; s < S; ++s) sort_off_[s + 1] += sort_off_[s];
    counts_.assign(S, 0);
    for (int i = 0; i < M; ++i) {
        const uint64_t e = ev_ks_[i];
        const int s = (uint32_t)e;
        const int32_t pos = sort_off_[s] + counts_[s]++;
        sort_ev_[pos] = (int32_t)(e >> 32);
        sort_s_[pos] = s;
    }
}

// ---- event-list output -------------------------------------------------------------------------
void DiagErrorSampler::sample_events(int S, SparseBatch& out) {
    gen_events(S);
    sort_events(S);
    out.S = S;
    out.shot_off.swap(sort_off_);          // hand the buckets over (scratch re-grows next call)
    out.ev.swap(sort_ev_);
}

}  // namespace qeccore
