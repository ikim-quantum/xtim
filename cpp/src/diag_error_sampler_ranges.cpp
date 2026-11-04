// diag_error_sampler_ranges — scaling S1 / T8: the event stream drawn in chunks and generated in parallel over channel
// ranges (see diag_error_sampler.hpp). In its own translation unit so the serial draw's code (diag_error_sampler.cpp)
// is unchanged.
#include "qeccore/diag_error_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "qeccore/keyed_rng.hpp"
#include "diag_error_sampler_gen.inc.hpp"

namespace qeccore {

// scaling T8: the channels [k0, k1) only, appended to `ev` (a channel's skip sequence is its own, so disjoint ranges are
// independent work; keyed streams only — the sequential stream is shared by every channel): the serial draw's one
// generator body (diag_error_sampler_gen.inc.hpp).
void DiagErrorSampler::gen_range_events_(int S, int k0, int k1, std::vector<uint64_t>& ev) {
    constexpr bool Keyed = true;
    constexpr uint64_t kGammaNoise = krng::kGamma;   // (as diag_error_sampler.cpp: one counter step per draw)
    XTIM_DIAG_GEN_BODY(ev, k0, k1);
}
#undef XTIM_DIAG_GEN_BODY                          // (its last use in this file)

void DiagErrorSampler::save_pos(StreamPos& p) const {
    p.kpos = kpos_;
    p.next = next_;
    std::memcpy(p.rs, rs_, sizeof rs_);
}
void DiagErrorSampler::restore_pos(const StreamPos& p) noexcept {
    std::copy(p.kpos.begin(), p.kpos.end(), kpos_.begin());   // (sizes equal: pos_matches, at the save)
    std::copy(p.next.begin(), p.next.end(), next_.begin());
    std::memcpy(rs_, p.rs, sizeof rs_);
}

std::vector<int> DiagErrorSampler::channel_split(int G) const {
    // expected events per shot of channel k: p_total (1 for a forced channel), plus a per-channel constant for its scan
    std::vector<double> c((size_t)K_ + 1, 0.0);
    for (int k = 0; k < K_; ++k) c[(size_t)k + 1] = c[(size_t)k] + std::min(1.0, ptot_[(size_t)k]) + 1e-3;
    std::vector<int> b((size_t)G + 1, K_);
    b[0] = 0;
    int k = 0;
    for (int g = 1; g < G; ++g) {
        const double want = c[(size_t)K_] * g / G;
        while (k < K_ && c[(size_t)k] < want) ++k;
        b[(size_t)g] = k;
    }
    return b;
}

void DiagErrorSampler::gen_range(int S, int B, EventBuckets& out) {
    if (!keyed_) throw std::logic_error("DiagErrorSampler::gen_range: the keyed streams only (engine bug)");
    if (out.k0 < 0 || out.k1 > K_ || out.k0 > out.k1 || B < 1)
        throw std::logic_error("DiagErrorSampler::gen_range: a bad channel range or block size (engine bug)");
    out.tmp.clear();
    gen_range_events_(S, out.k0, out.k1, out.tmp);
    // stable counting sort by block (s / B): each block's events stay k-major
    const int nb = (S + B - 1) / B;
    out.off.assign((size_t)nb + 1, 0);
    for (uint64_t e : out.tmp) ++out.off[(size_t)((uint32_t)e / (uint32_t)B) + 1];
    for (int b = 0; b < nb; ++b) out.off[(size_t)b + 1] += out.off[(size_t)b];
    out.pos.assign(out.off.begin(), out.off.end() - 1);
    out.ev.resize(out.tmp.size());
    for (uint64_t e : out.tmp) out.ev[(size_t)out.pos[(size_t)((uint32_t)e / (uint32_t)B)]++] = e;
}

int DiagErrorSampler::merge_block(const EventBuckets* tasks, int G, int b, int B, int S, std::vector<int32_t>& cnt,
                                  int32_t base, int32_t* shot_end, int32_t* ev) {
    const int s0 = b * B, c = std::min(B, S - s0);
    cnt.assign((size_t)c + 1, 0);
    for (int g = 0; g < G; ++g) {
        const EventBuckets& t = tasks[g];
        for (int32_t i = t.off[(size_t)b]; i < t.off[(size_t)b + 1]; ++i)
            ++cnt[(size_t)((uint32_t)t.ev[(size_t)i] - (uint32_t)s0) + 1];
    }
    for (int i = 0; i < c; ++i) cnt[(size_t)i + 1] += cnt[(size_t)i];
    for (int i = 0; i < c; ++i) shot_end[i] = base + cnt[(size_t)i + 1];
    const int total = cnt[(size_t)c];
    // tasks in channel-range order, each k-major: per shot, ascending channel (sample_events' stable shot sort)
    for (int g = 0; g < G; ++g) {
        const EventBuckets& t = tasks[g];
        for (int32_t i = t.off[(size_t)b]; i < t.off[(size_t)b + 1]; ++i) {
            const uint64_t e = t.ev[(size_t)i];
            ev[cnt[(size_t)((uint32_t)e - (uint32_t)s0)]++] = (int32_t)(e >> 32);
        }
    }
    return total;
}

void DiagErrorSampler::sample_events_chunked(int S, int chunk, SparseBatch& out) {
    if (chunk < 1) throw std::invalid_argument("sample_events_chunked: chunk must be >= 1");
    out.S = S;
    out.shot_off.resize((size_t)S + 1);
    out.shot_off[0] = 0;
    out.ev.clear();
    for (int s0 = 0; s0 < S; s0 += chunk) {
        const int c = std::min(chunk, S - s0);
        gen_events(c);
        sort_events(c);
        const int32_t base = (int32_t)out.ev.size();
        for (int i = 1; i <= c; ++i) out.shot_off[(size_t)(s0 + i)] = base + sort_off_[(size_t)i];
        out.ev.insert(out.ev.end(), sort_ev_.begin(), sort_ev_.end());
    }
}

}  // namespace qeccore
