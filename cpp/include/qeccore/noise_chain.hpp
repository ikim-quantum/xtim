#pragma once
// noise_chain — the twirl sampler's NOISE stream (lean N1, OWNER-SIGNED 2026-11-07 "Signing b"; ENGINE_VERSION 17).
//
// THE MECHANISM TABLE (level 0). A channel k is categorical: it fires with Σ_a p_a and then realizes exactly one of its
// alternatives a (mutually exclusive). Two tables are built at construction from the channel table and the per-alternative
// rotation records (a record = what a shot on a record-only route reads of a fired alternative: its sign words, token,
// extras words and the channel words of t(P_f); the shot is the XOR of its fired records):
//   * PRUNED (the sample runs): per channel, the alternatives whose record is all zero (NULL: no record column sees them)
//     are dropped, and alternatives with the same record are one OUTCOME whose probability is the sum of theirs (in
//     alternative order); a channel with no outcome left is no mechanism. The record XOR of every shot has exactly the old
//     law (channels independent, outcomes of one channel exclusive). An outcome's representative alternative = the first
//     alternative of its class (the event-list routes hand it to the window's oracle: its record IS the outcome's).
//     NO cross-channel convolution.
//   * FULL (every run that reads P_f itself: recording runs (P_rec), framed / term-recorded runs): every channel with
//     Σ p > 0 is a mechanism, every alternative with p > 0 its own outcome — exact P_rec, including faults no record sees.
// Mechanisms keep channel order. The mechanisms of bitwise-equal P = min(1, Σ outcome p) form a GROUP; groups are in
// ascending P bits; inside a group, mechanisms in channel order. This order is part of the byte contract.
//
// THE CHAIN. Shots are cut into BLOCKS of kB = 4096 shots ALIGNED on the cumulative shot index (block b = shots
// [b·kB, (b+1)·kB)): a block's noise is a pure function of (seed, b, table), so a call split, a seed=None continuation,
// the route and the thread count cannot change it (a call ending mid-block keeps or regenerates its block; lanes take
// whole blocks). Per (seed, block b, group g) ONE chain on its OWN counter range of the tag's keyed sequence (lean N2,
// OWNER-SIGNED 2026-11-07; the per-shot streams' construction, keyed_rng.hpp):
//     draw c = mix(stream_key(seed, tag) + (b·R + r0_g + c + 1)·γ),   c < R_g = 2·M_g·kB + 1,
//     r0_g = Σ_{g' < g} R_g',   R = Σ_g R_g   (M_g = the group's mechanism count)
// (tag = krng::NOISE for the pruned table, krng::NOISE_FULL for the full table) walks the group's (mechanism × shot)
// grid MECHANISM-MAJOR (cell t = j·kB + s for the group's j-th mechanism and block shot s) with inverse-CDF geometric
// gaps: u = ((word >> 11) + 1)·2^-53 ∈ (0, 1], gap = floor(log(u) / log1p(−P)) (clamped at 2^62 before the cast); the
// first fired cell is gap, each next one the previous + 1 + gap; the walk ends past M_g·kB. A group with P ≥ 1 fires
// every cell without a gap word. A fired cell of a mechanism with n > 1 outcomes draws its outcome on the NEXT word of
// the same stream: x = u·n, i = min((int)x, n − 1), outcome = i if x − i < prob[i] else alias[i] (Vose alias table over
// the outcome probabilities, built in outcome order; a single-outcome mechanism draws no word). P(gap = g) = P(1−P)^g is
// the law of the non-firing cells before the next firing one of i.i.d. Bernoulli(P) cells, so every (mechanism, shot)
// cell is an independent Bernoulli(P) and the outcome categorical — exactly the channel's law (up to double rounding,
// the class of the former cumulative scan).
// DISJOINT RANGES. A chain draws at most one gap word per fired cell plus one, and at most one outcome word per fired cell
// (≤ M_g·kB fired cells): ≤ R_g words. The ranges [b·R + r0_g, b·R + r0_g + R_g) tile [b·R, (b + 1)·R) in group order,
// blocks abut, γ is odd and mix a bijection: no two chains of one (seed, tag) ever read the same word (before N2 every
// chain sat at a hashed base on the one γ-cycle, so two chains could share a stretch: a birthday event, quadratic in the
// shot count). Chains of DIFFERENT tags (NOISE / NOISE_FULL, the per-shot COIN / BORN / RF) are different keyed sequences,
// but NOT disjoint ones: every tag is a shift (its key) of the same γ-mix cycle, so a tag's span can overlap another's.
// That is a rare generator-quality event, not a law change (the NOISE span is one interval of length ≈ 2·M·N, so an
// overlap would be all-or-nothing: probability ≈ (2·M·N + N·2^20)/2^64 per tag, ≈ 7e-4 expected on the s25 circuit at
// 1e10 shots; the per-shot tags were always in this situation); a run draws one noise table. Block b needs (b + 1)·R ≤ 2^64: a block past that
// is REFUSED by name (std::invalid_argument; a new seed restarts the block index) — about 2·M·N ≥ 2^64 for N cumulative
// shots of one seed and M mechanisms (never reached under the 2^44 shot cap while R ≤ 2^32, i.e. about M < 2^19;
// e2e s25: M ≈ 1.1e5 pruned, 2.2e5 full).
//
// EVENT LISTS. The event-list routes read a block's events shot-major, per shot in ascending alternative id (= ascending
// channel: the former order), the pruned table handing representative alternatives.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "qeccore/diag_error_sampler.hpp"   // ErrorChannel, DiagErrorSampler::SparseBatch (the event-list shape)
#include "qeccore/keyed_rng.hpp"

namespace qeccore {

struct NoiseTable {
    uint64_t tag = 0;                       // the stream tag (krng::NOISE / krng::NOISE_FULL)
    std::vector<int32_t> mo0;               // M + 1: mechanism m's outcomes [mo0[m], mo0[m + 1])
    std::vector<int32_t> oalt;              // per outcome: its (representative) alternative id
    std::vector<double> ow;                 // per outcome: its probability
    std::vector<double> oprob;              // per outcome: the alias table's acceptance
    std::vector<int32_t> oalias;            // per outcome: the alias table's alias (an index inside the mechanism)
    std::vector<int32_t> mch;               // per mechanism: its channel
    std::vector<double> mP;                 // per mechanism: P = min(1, Σ outcome p in outcome order)
    struct Group {
        double P = 0.0, sc = 0.0;           // sc = 1 / log1p(−P) (< 0), or +1 for a forced group (P ≥ 1)
        int32_t m0 = 0, m1 = 0;             // the group's mechanisms gm[m0, m1)
        uint64_t r0 = 0, r = 0;             // the chain's counter range inside a block: [r0, r0 + r), r = 2·M_g·kB + 1
    };
    std::vector<Group> groups;              // ascending P bits
    uint64_t R = 0;                         // counters per block: Σ_g r (block b owns [b·R, (b + 1)·R))
    uint64_t blast = 0;                     // the last block whose range fits: (b + 1)·R ≤ 2^64
    std::vector<int32_t> gm;                // mechanism ids, group-major (channel order inside a group)
    double events = 0.0;                    // expected fired events per shot (Σ_m P_m)
    int M() const { return mo0.empty() ? 0 : (int)mo0.size() - 1; }
    size_t bytes() const;
};

class NoiseModel {
  public:
    static constexpr int kBShift = 12;
    static constexpr int kB = 1 << kBShift;   // the block (part of the byte contract)

    NoiseModel() = default;
    // channels: the alternatives' probabilities in flat-alternative-id order (channel-major); rec / off: the
    // alternatives' records (record a = rec[off[a], off[a + 1]); canonical: two alternatives have the same effect iff
    // their records are identical, and a record is NULL iff it is all zero). Throws std::invalid_argument on a channel
    // with no alternative, p < 0 or Σ p > 1 (+1e-9).
    NoiseModel(const std::vector<ErrorChannel>& channels, const std::vector<uint32_t>& rec, const std::vector<size_t>& off);

    const NoiseTable& pruned() const { return pruned_; }
    const NoiseTable& full() const { return full_; }
    int K() const { return K_; }                                  // channel count
    int alt_total() const { return (int)alt_k_.size(); }          // flat alternative count
    int32_t channel_of(int32_t a) const { return alt_k_[(size_t)a]; }
    double alt_p(int32_t a) const { return alt_p_[(size_t)a]; }
    size_t bytes() const;

    // Block b's events of table T under `seed`, handed to f(alt, s) (s = the shot inside the block, [0, kB)) in CHAIN
    // order: group by group, mechanism-major. The ONE generator (every route). Throws std::invalid_argument (by name)
    // for a block past T.blast.
    template <class F> void walk(const NoiseTable& T, uint64_t seed, uint64_t b, F&& f) const;
    // The position of chain (group g, block b): its draw c = mix(chain_pos + (c + 1)·γ) (the ONE layout; b <= T.blast).
    static uint64_t chain_pos(const NoiseTable& T, uint64_t key, size_t g, uint64_t b) noexcept {
        return krng::noise_pos(key, b * T.R + T.groups[g].r0);
    }
    // The same events appended to ev as (alt << 32 | s), chain order.
    void block_events(const NoiseTable& T, uint64_t seed, uint64_t b, std::vector<uint64_t>& ev) const;
    // Shots [lo, lo + S) of a block's chain-order events laid out shot-major (per shot ascending alternative id) into
    // out (out.S = S, shot_off from 0); cnt is scratch.
    static void shot_major(const std::vector<uint64_t>& ev, int lo, int S, DiagErrorSampler::SparseBatch& out,
                           std::vector<int32_t>& cnt);

  private:
    [[noreturn]] static void refuse_block_(const NoiseTable& T, uint64_t b);
    int K_ = 0;
    std::vector<int32_t> alt_k_;
    std::vector<double> alt_p_;
    NoiseTable pruned_, full_;
};

}  // namespace qeccore

#include <cmath>

namespace qeccore {

template <class F>
inline void NoiseModel::walk(const NoiseTable& T, uint64_t seed, uint64_t b, F&& f) const {
    const int32_t* const mo0 = T.mo0.data();
    const int32_t* const oalt = T.oalt.data();
    const double* const opr = T.oprob.data();
    const int32_t* const oal = T.oalias.data();
    const int32_t* const gm = T.gm.data();
    if (b > T.blast) refuse_block_(T, b);
    const uint64_t key = krng::stream_key(seed, T.tag);
    for (size_t g = 0; g < T.groups.size(); ++g) {
        const NoiseTable::Group& G = T.groups[g];
        uint64_t pos = chain_pos(T, key, g, b);                         // draw c = mix(pos + (c + 1)·γ)
        const int32_t* const gmm = gm + G.m0;
        auto emit = [&](int64_t t) {
            const int32_t m = gmm[t >> kBShift];
            const int32_t o0 = mo0[m], n = mo0[m + 1] - o0;
            int32_t o = o0;
            if (n > 1) {
                pos += krng::kGamma;
                const double x = ((krng::mix(pos) >> 11) + 1) * 0x1.0p-53 * n;
                int i = (int)x;
                if (i >= n) i = n - 1;
                o = (x - i < opr[o0 + i]) ? o0 + i : o0 + oal[o0 + i];
            }
            f(oalt[o], (int)(t & (kB - 1)));
        };
        const int64_t end = (int64_t)(G.m1 - G.m0) << kBShift;
        if (G.sc > 0.0) {                                   // forced (P >= 1): every cell, no gap word
            for (int64_t t = 0; t < end; ++t) emit(t);
            continue;
        }
        const double sc = G.sc;
        auto gap = [&]() -> int64_t {
            pos += krng::kGamma;
            const double gp = std::log(((krng::mix(pos) >> 11) + 1) * 0x1.0p-53) * sc;   // >= 0
            return (gp < 4.0e18) ? (int64_t)gp : ((int64_t)1 << 62);
        };
        for (int64_t t = gap(); t < end; t += 1 + gap()) emit(t);
    }
}

}  // namespace qeccore
