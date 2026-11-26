// test_keyed_rng — S0(b): the ONE keyed counter-based generator of the twirl sampler.
//
//  * known answers: mix(γ) is SplitMix64(seed 0)'s first output; draw() is the documented formula;
//  * Stream::next()/word()/u() and CoinBits are exactly draw(seed, shot, tag, ctr) / its bits (a
//    copy of a CoinBits is a stream-neutral fork);
//  * lean N1: the NOISE stream's tables (null pruning, intra-channel merging, groups) and chains: a block's noise is a
//    pure function of (seed, block, table); any shot range of a block reads the block's own event lists.
// Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qeccore/noise_chain.hpp"
#include "qeccore/keyed_rng.hpp"

using namespace qeccore;

static void known_answers() {
    CHECK(krng::mix(0) == 0ull);
    CHECK(krng::mix(krng::kGamma) == 0xE220A8397B1DCDAFull);   // SplitMix64(seed=0), output 1
    const uint64_t seed = 0x0123456789ABCDEFull;
    const uint64_t key = krng::mix(seed ^ krng::mix(krng::COIN));
    const uint64_t base = key + (7ull << 20) * krng::kGamma;          // shot 7: counter 7·2^20 + ctr
    CHECK(krng::draw(seed, 7, krng::COIN, 0) == krng::mix(base + krng::kGamma));
    CHECK(krng::draw(seed, 7, krng::COIN, 5) == krng::mix(base + 6 * krng::kGamma));
    // injective layout: the last draw of shot 6 and the first of shot 7 are adjacent counters
    CHECK(krng::draw(seed, 6, krng::COIN, krng::kMaxCtr - 1) == krng::mix(base));
    CHECK(krng::noise_pos(krng::mix(seed ^ krng::mix(krng::NOISE)), 12345) ==
          krng::mix(seed ^ krng::mix(krng::NOISE)) + 12345 * krng::kGamma);
    // distinct tags / slots give distinct streams (the tag goes through the hash)
    CHECK(krng::draw(seed, 7, krng::COIN, 0) != krng::draw(seed, 7, krng::BORN, 0));
    CHECK(krng::draw(seed, 7, krng::COIN, 0) != krng::draw(seed, 8, krng::COIN, 0));
}

static void stream_matches_reference() {
    const uint64_t seed = 977;
    for (uint64_t tag = krng::COIN; tag <= krng::EXACT; ++tag) {
        krng::Stream st;
        st.set_key(seed, tag);
        for (uint64_t slot : {0ull, 1ull, 12345ull, (1ull << 40) + 3}) {
            st.at(slot);
            for (uint64_t c = 0; c < 70; ++c) {
                CHECK(st.word(c) == krng::draw(seed, slot, tag, c));
                CHECK(st.u(c) == krng::u01(krng::draw(seed, slot, tag, c)));
            }
            for (uint64_t c = 0; c < 70; ++c) CHECK(st.next() == krng::draw(seed, slot, tag, c));
        }
    }
    // CoinBits: coin k = bit (k mod 64) of COIN word floor(k/64); a copy forks without consuming.
    krng::CoinBits cb;
    cb.st.set_key(seed, krng::COIN);
    cb.at(42);
    for (int k = 0; k < 64 + 17; ++k) {
        krng::CoinBits fork = cb;
        const bool f = fork.next();
        const bool b = cb.next();
        const bool want = (krng::draw(seed, 42, krng::COIN, (uint64_t)(k >> 6)) >> (k & 63)) & 1;
        CHECK(b == want);
        CHECK(f == b);
    }
    cb.at(43);   // re-positioning resets the reservoir
    CHECK(cb.next() == (bool)(krng::draw(seed, 43, krng::COIN, 0) & 1));
}

// lean N1: the NOISE stream (qeccore/noise_chain.hpp). A channel set with null and identical records, forced and p = 0
// alternatives: the tables (pruning, merging, groups), the chain's first draws by hand, and the event lists of any shot
// range of a block equal to the block's own (the noise of a block is a pure function of (seed, block, table)).
static void noise_chain_contract() {
    std::vector<ErrorChannel> ch;
    std::vector<uint32_t> rec;
    std::vector<size_t> off{0};
    auto alt = [&](ErrorChannel& c, double p, std::vector<uint32_t> r) {
        ErrorChannel::Alt a; a.p = p; c.alts.push_back(a);
        rec.insert(rec.end(), r.begin(), r.end());
        off.push_back(rec.size());
    };
    for (int k = 0; k < 30; ++k) {
        ErrorChannel c;
        const double p = (k % 5 == 0) ? 0.3 : (k % 3 == 0 ? 0.02 : 0.001);
        if (k % 4 == 0) {                                    // X / Y the same record, Z null
            alt(c, p / 3, {1u + (uint32_t)k, 0u});
            alt(c, p / 3, {1u + (uint32_t)k, 0u});
            alt(c, p / 3, {0u, 0u});
        } else if (k % 7 == 0) {                             // all null: no pruned mechanism
            alt(c, p, {0u, 0u});
        } else {
            alt(c, p, {(uint32_t)k, 7u});
            alt(c, p / 2, {(uint32_t)k, 8u});
        }
        ch.push_back(c);
    }
    { ErrorChannel f; alt(f, 1.0, {5u, 5u}); ch.push_back(f); }                      // forced
    { ErrorChannel z; alt(z, 0.0, {9u, 9u}); ch.push_back(z); }                      // p = 0
    const NoiseModel N(ch, rec, off);
    const NoiseTable& P = N.pruned();
    const NoiseTable& F = N.full();
    CHECK(P.tag == krng::NOISE && F.tag == krng::NOISE_FULL);
    CHECK(F.M() == 31 && P.M() == 31 - 3);                   // the p = 0 channel; k = 7, 14, 21 (all null) drop
    for (int m = 0; m < P.M(); ++m) {
        const int k = P.mch[(size_t)m];
        const int n = P.mo0[(size_t)m + 1] - P.mo0[(size_t)m];
        if (k == 30) CHECK(n == 1 && P.mP[(size_t)m] == 1.0);
        else if (k % 4 == 0) CHECK(n == 1 && P.ow[(size_t)P.mo0[(size_t)m]] == (ch[(size_t)k].alts[0].p + ch[(size_t)k].alts[1].p));
        else CHECK(n == 2);
    }
    for (size_t g = 1; g < P.groups.size(); ++g) CHECK(P.groups[g - 1].P < P.groups[g].P);
    // lean N2: the counter ranges r = 2·M_g·kB + 1 tile a block in group order; R = Σ r; blast = the last block that fits
    for (const NoiseTable* T : {&P, &F}) {
        uint64_t r0 = 0;
        for (const NoiseTable::Group& G : T->groups) {
            CHECK(G.r0 == r0 && G.r == 2 * (uint64_t)(G.m1 - G.m0) * NoiseModel::kB + 1);
            r0 += G.r;
        }
        CHECK(T->R == r0 && T->R > 0);
        const unsigned __int128 two64 = (unsigned __int128)1 << 64;
        CHECK((unsigned __int128)(T->blast + 1) * T->R <= two64 && (unsigned __int128)(T->blast + 2) * T->R > two64);
        bool threw = false;
        std::vector<uint64_t> ev;
        try { N.block_events(*T, 1, T->blast + 1, ev); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        N.block_events(*T, 1, T->blast, ev);                // the last block that fits is served
    }
    // the chain by hand: group 0's first gap word (chain (0, b) starts at counter b·R)
    {
        const NoiseTable::Group& g0 = P.groups[0];
        const uint64_t base = NoiseModel::chain_pos(P, krng::stream_key(77, krng::NOISE), 0, 3);
        CHECK(base == krng::mix(77 ^ krng::mix(krng::NOISE)) + 3 * P.R * krng::kGamma);
        const double u = ((krng::mix(base + krng::kGamma) >> 11) + 1) * 0x1.0p-53;
        const int64_t t0 = (int64_t)(std::log(u) * g0.sc);
        std::vector<uint64_t> ev;
        N.block_events(P, 77, 3, ev);
        bool found = false;
        for (uint64_t e : ev) {                              // the group's first event is at cell t0
            const int32_t a = (int32_t)(e >> 32);
            const int k = N.channel_of(a);
            int m = 0;
            while (P.mch[(size_t)m] != k) ++m;
            if (P.mP[(size_t)m] != g0.P) continue;
            int j = 0;
            while (P.gm[(size_t)(g0.m0 + j)] != m) ++j;
            CHECK((int64_t)j * NoiseModel::kB + (int64_t)(uint32_t)e == t0);
            found = true;
            break;
        }
        CHECK(found || t0 >= (int64_t)(g0.m1 - g0.m0) * NoiseModel::kB);
    }
    // any shot range of a block == the block's own lists; the forced channel fires every shot; seeds / blocks differ
    for (const NoiseTable* T : {&P, &F}) {
        std::vector<uint64_t> ev, ev2;
        N.block_events(*T, 2024, 5, ev);
        N.block_events(*T, 2024, 5, ev2);
        CHECK(ev == ev2);
        DiagErrorSampler::SparseBatch all, part;
        std::vector<int32_t> cnt;
        NoiseModel::shot_major(ev, 0, NoiseModel::kB, all, cnt);
        long forced = 0;
        for (int s = 0; s < NoiseModel::kB; ++s)
            for (int32_t i = all.shot_off[(size_t)s]; i < all.shot_off[(size_t)s + 1]; ++i) {
                if (N.channel_of(all.ev[(size_t)i]) == 30) ++forced;
                if (i > all.shot_off[(size_t)s]) CHECK(all.ev[(size_t)i - 1] < all.ev[(size_t)i]);
            }
        CHECK(forced == NoiseModel::kB);
        for (int lo : {0, 1, 1000, 4095}) for (int S : {1, 7, 96}) {
            if (lo + S > NoiseModel::kB) continue;
            NoiseModel::shot_major(ev, lo, S, part, cnt);
            CHECK(part.S == S && (int)part.shot_off.size() == S + 1 && part.shot_off[0] == 0);
            for (int s = 0; s < S; ++s) {
                const int32_t* p0 = all.ev.data() + all.shot_off[(size_t)(lo + s)];
                const int32_t c0 = all.shot_off[(size_t)(lo + s) + 1] - all.shot_off[(size_t)(lo + s)];
                CHECK(part.shot_off[(size_t)s + 1] - part.shot_off[(size_t)s] == c0);
                CHECK(std::equal(p0, p0 + c0, part.ev.data() + part.shot_off[(size_t)s]));
            }
        }
        N.block_events(*T, 2025, 5, ev2);
        CHECK(ev2 != ev);
        N.block_events(*T, 2024, 6, ev2);
        CHECK(ev2 != ev);
    }
}

// FCS fix 5: draw_rf's first-draw threshold. Every u <= geo_first_stop_thr(p, k) must END the chain exactly as the
// computed step does (!(log(u) · il < k), il = 1 / log1p(−p): draw_rf's expression); the scan walks the representable
// u = ((w >> 11) + 1)·2^-53 around the threshold and around the exact boundary (1−p)^k, plus random u below the
// threshold, over p from 1e-12 to 0.999 and k from 1 to 10^6.
static void rf_first_draw_threshold() {
    const double ps[] = {1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999};
    const size_t ks[] = {1, 2, 3, 7, 16, 63, 64, 100, 1000, 4096, 100000, 1000000};
    long viol = 0, taken = 0;
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (double p : ps)
        for (size_t k : ks) {
            const double il = 1.0 / std::log1p(-p), thr = geo_first_stop_thr(p, k);
            CHECK(thr >= 0.0 && thr <= 1.0);
            auto stops = [&](double u) { return !(std::log(u) * il < (double)k); };
            auto at = [&](double x, long d) {   // the representable u nearest x, then d steps of 2^-53
                if (!(x > 0.0)) return;
                const double q = std::floor(x * 0x1.0p53);
                for (long i = -d; i <= d; ++i) {
                    const double m = q + (double)i;
                    if (m < 1.0 || m > 0x1.0p53) continue;
                    const double u = m * 0x1.0p-53;
                    if (u <= thr) { ++taken; if (!stops(u)) ++viol; }
                }
            };
            at(thr, 4096);
            at(std::exp((double)k * std::log1p(-p)), 4096);   // the exact boundary: none of these may be taken wrongly
            for (int i = 0; i < 2000; ++i) {
                r = krng::mix(r + krng::kGamma);
                const double u = (double)((r >> 11) + 1) * 0x1.0p-53 * thr;
                if (u > 0.0 && u <= thr) { ++taken; if (!stops(u)) ++viol; }
            }
        }
    std::fprintf(stderr, "  rf threshold: %ld taken, %ld violations\n", taken, viol);
    CHECK(taken > 100000);
    CHECK_EQ(viol, 0);
}

int main() {
    RUN(known_answers);
    RUN(rf_first_draw_threshold);
    RUN(stream_matches_reference);
    RUN(noise_chain_contract);
    REPORT();
}
