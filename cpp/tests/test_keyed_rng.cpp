// test_keyed_rng — S0(b): the ONE keyed counter-based generator of the twirl sampler.
//
//  * known answers: mix(γ) is SplitMix64(seed 0)'s first output; draw() is the documented formula;
//  * Stream::next()/word()/u() and CoinBits are exactly draw(seed, shot, tag, ctr) / its bits (a
//    copy of a CoinBits is a stream-neutral fork);
//  * DiagErrorSampler keyed mode: every channel's event sequence is a pure function of (seed, k) —
//    the shot-bucketed event lists are IDENTICAL for every batch partition (the former sequential
//    xoshiro stream was channel-major within a batch, so its bytes depended on the partition);
//    reseed() restarts it exactly.
// Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

#include "qeccore/diag_error_sampler.hpp"
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
    CHECK(krng::channel_base(seed, 3) == krng::mix(krng::mix(seed ^ krng::mix(krng::NOISE)) + 3 * krng::kGamma));
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

// Shot-major event lists of N shots sampled in batches of the given sizes.
static std::vector<std::vector<int32_t>> events(DiagErrorSampler& d, const std::vector<int>& sizes) {
    std::vector<std::vector<int32_t>> out;
    DiagErrorSampler::SparseBatch b;
    for (int S : sizes) {
        d.sample_events(S, b);
        for (int s = 0; s < S; ++s)
            out.emplace_back(b.ev.begin() + b.shot_off[(size_t)s], b.ev.begin() + b.shot_off[(size_t)s + 1]);
    }
    return out;
}

static void keyed_noise_partition_independent() {
    const int n = 6;
    std::vector<ErrorChannel> ch;
    for (int k = 0; k < 40; ++k) {
        ErrorChannel c;
        const double p = (k % 5 == 0) ? 0.3 : (k % 3 == 0 ? 0.02 : 0.001);
        if (k % 4 == 0) {                                        // a 3-alternative channel
            for (int t = 0; t < 3; ++t) {
                ErrorChannel::Alt a; a.p = p / 3; a.a = {{(k + t) % n, (uint8_t)(1 + t)}};
                c.alts.push_back(a);
            }
        } else {
            ErrorChannel::Alt a; a.p = p; a.a = {{k % n, (uint8_t)2}};
            c.alts.push_back(a);
        }
        ch.push_back(c);
    }
    const int N = 9000;
    DiagErrorSampler ref(n, ch, 2024, /*keyed=*/true);
    const auto e0 = events(ref, {N});
    long fired = 0;
    for (const auto& v : e0) fired += (long)v.size();
    CHECK(fired > 1000);
    for (const std::vector<int>& sizes : std::vector<std::vector<int>>{
             {4096, 4096, 808}, {1, 1, 8998}, {7, 4093, 2, 4898}, {3000, 3000, 3000}}) {
        DiagErrorSampler d(n, ch, 2024, /*keyed=*/true);
        CHECK(events(d, sizes) == e0);
    }
    for (int chunk : {1, 7, 4096, 9000, 20000}) {               // scaling S1: the chunked draw, any chunk
        DiagErrorSampler d(n, ch, 2024, /*keyed=*/true);
        DiagErrorSampler::SparseBatch b;
        (void)events(d, {5});
        d.sample_events_chunked(N - 5, chunk, b);
        std::vector<std::vector<int32_t>> got(e0.begin(), e0.begin() + 5);
        CHECK(b.S == N - 5 && (long)b.shot_off.size() == N - 4 && b.shot_off[0] == 0);
        for (int s = 0; s < N - 5; ++s)
            got.emplace_back(b.ev.begin() + b.shot_off[(size_t)s], b.ev.begin() + b.shot_off[(size_t)s + 1]);
        CHECK(got == e0);
        CHECK(b.shot_off[(size_t)(N - 5)] == (int32_t)b.ev.size());
    }
    // scaling T8: generation tasks over channel ranges, bucketed by block, laid out block by block (any task count, any
    // block size, blocks in any order): the same events in the same per-shot order, and the streams continue as one.
    // R3: also with forced channels (p >= 1: one alternative, three alternatives) and a p = 0 channel among them.
    std::vector<ErrorChannel> ch2 = ch;
    {
        ErrorChannel f1; ErrorChannel::Alt a; a.p = 1.0; a.a = {{1, (uint8_t)1}}; f1.alts.push_back(a);
        ErrorChannel f3;
        for (int t = 0; t < 3; ++t) { ErrorChannel::Alt b; b.p = 1.0 / 3; b.a = {{(2 + t) % n, (uint8_t)(1 + t)}}; f3.alts.push_back(b); }
        ErrorChannel z; ErrorChannel::Alt c; c.p = 0.0; c.a = {{0, (uint8_t)2}}; z.alts.push_back(c);
        ch2.insert(ch2.begin() + 3, f1);
        ch2.insert(ch2.begin() + 17, z);
        ch2.insert(ch2.begin() + 30, f3);
    }
    for (const std::vector<ErrorChannel>* cs : {&ch, &ch2})
    for (int G : {1, 3, 7, 43}) for (int Bk : {1, 64, 1000, 9000}) {
        DiagErrorSampler dref(n, *cs, 2024, /*keyed=*/true);
        const auto e0 = events(dref, {N});
        DiagErrorSampler d(n, *cs, 2024, /*keyed=*/true);
        (void)events(d, {5});
        const int S = N - 5 - 100;
        const std::vector<int> kb = d.channel_split(G);
        CHECK((int)kb.size() == G + 1 && kb[0] == 0 && kb[(size_t)G] == d.K());
        std::vector<DiagErrorSampler::EventBuckets> t((size_t)G);
        for (int g = G - 1; g >= 0; --g) {            // tasks in any order (each advances its own channels only)
            t[(size_t)g].k0 = kb[(size_t)g];
            t[(size_t)g].k1 = kb[(size_t)g + 1];
            d.gen_range(S, Bk, t[(size_t)g]);
        }
        const int nb = (S + Bk - 1) / Bk;
        std::vector<int32_t> boff((size_t)nb + 1, 0), cnt;
        for (int b = 0; b < nb; ++b) {
            boff[(size_t)b + 1] = boff[(size_t)b];
            for (const auto& x : t) boff[(size_t)b + 1] += x.off[(size_t)b + 1] - x.off[(size_t)b];
        }
        std::vector<int32_t> off((size_t)S + 1, 0), ev((size_t)boff[(size_t)nb]);
        for (int b = nb - 1; b >= 0; --b) {           // blocks in any order
            const int m = DiagErrorSampler::merge_block(t.data(), G, b, Bk, S, cnt, boff[(size_t)b],
                                                        off.data() + (size_t)b * Bk + 1, ev.data() + boff[(size_t)b]);
            CHECK(m == boff[(size_t)b + 1] - boff[(size_t)b]);
        }
        std::vector<std::vector<int32_t>> got(e0.begin(), e0.begin() + 5);
        for (int s = 0; s < S; ++s) got.emplace_back(ev.begin() + off[(size_t)s], ev.begin() + off[(size_t)s + 1]);
        for (const auto& v : events(d, {100})) got.push_back(v);   // the streams continue from where the tasks left them
        CHECK(got == e0);
    }
    DiagErrorSampler d(n, ch, 5, /*keyed=*/true);
    (void)events(d, {123});
    d.reseed(2024);                                              // reseed == fresh construction
    CHECK(events(d, {N}) == e0);
    DiagErrorSampler other(n, ch, 2025, /*keyed=*/true);
    CHECK(events(other, {N}) != e0);
}

int main() {
    RUN(known_answers);
    RUN(stream_matches_reference);
    RUN(keyed_noise_partition_independent);
    REPORT();
}
