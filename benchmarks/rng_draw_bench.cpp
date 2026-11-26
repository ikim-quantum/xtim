// rng_draw_bench — the S0(b) generator micro-gate (D14 ledger component "ns/draw").
//
// Today's twirl streams (dev/next 9cf764e) against the keyed counter generator that replaces
// them (cpp/include/qeccore/keyed_rng.hpp), one row per draw kind:
//   uniform  : mt19937_64 + uniform_real_distribution (today: Born observable / decisions /
//              exact fallback)                      vs krng::Stream::next() -> u01
//   word     : the per-shot splitmix ShotRng step (today: coins, readout flips)
//                                                   vs krng::Stream::next()
//   noise    : xoshiro256** (today: noise skips, one shared state)
//                                                   vs krng per-channel stream (local position)
//   position : ShotRng::seed (XOR-linear)           vs krng::Stream::at (multiply-add)
// Method (design §5.0 hygiene): every timed block is preceded by a discarded warm-up block; the
// A/B blocks alternate within each round (order flipped every round); the result per row is the
// median over rounds of ns/draw, plus the median of the per-round B/A ratios.
//
// Build: g++ -O3 -funroll-loops -march=native -std=c++17 -Icpp/include benchmarks/rng_draw_bench.cpp
// Run:   taskset -c <core> ./a.out [rounds=31] [draws=20000000]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "qeccore/keyed_rng.hpp"

using namespace qeccore;
using clk = std::chrono::steady_clock;

namespace {
volatile uint64_t g_sink = 0;
volatile double g_dsink = 0.0;

// today's per-shot coin/rf generator (twirl_sampler.cpp ShotRng at 9cf764e), verbatim
struct ShotRng {
    uint64_t s = 0;
    void seed(uint64_t base_seed, uint64_t shot_idx) noexcept {
        s = (base_seed + 0x9E3779B97F4A7C15ULL) ^ (shot_idx * 0xBF58476D1CE4E5B9ULL);
    }
    uint64_t next() noexcept {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};
// today's noise generator (diag_error_sampler.cpp xoshiro256** at 9cf764e), verbatim
struct Xoshiro {
    uint64_t s[4];
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
};

template <class F> double time_ns(F&& f, long n) {
    auto t0 = clk::now();
    f(n);
    return std::chrono::duration<double, std::nano>(clk::now() - t0).count() / (double)n;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 31;
    const long N = argc > 2 ? std::atol(argv[2]) : 20000000L;

    // ---- the draw kinds (A = today, B = keyed) ----
    auto a_uniform = [](long n) {
        std::mt19937_64 g(12345);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        double acc = 0.0;
        for (long i = 0; i < n; ++i) acc += U(g);
        g_dsink = g_dsink + acc;
    };
    auto b_uniform = [](long n) {
        krng::Stream st; st.set_key(12345, krng::OBS); st.at(7);
        double acc = 0.0;
        for (long i = 0; i < n; ++i) acc += krng::u01(st.next());
        g_dsink = g_dsink + acc;
    };
    auto a_word = [](long n) {
        ShotRng r; r.seed(12345, 7);
        uint64_t acc = 0;
        for (long i = 0; i < n; ++i) acc ^= r.next();
        g_sink = g_sink ^ acc;
    };
    auto b_word = [](long n) {
        krng::Stream st; st.set_key(12345, krng::COIN); st.at(7);
        uint64_t acc = 0;
        for (long i = 0; i < n; ++i) acc ^= st.next();
        g_sink = g_sink ^ acc;
    };
    // noise: K channels, a few draws per channel per visit (the gen_events access pattern)
    constexpr int K = 4096;
    auto a_noise = [](long n) {
        Xoshiro x{{1, 2, 3, 4}};
        uint64_t acc = 0;
        for (long i = 0; i < n; i += 2) { acc ^= x.next(); acc ^= x.next(); }
        g_sink = g_sink ^ acc;
    };
    std::vector<uint64_t> kpos(K);
    for (int k = 0; k < K; ++k) kpos[(size_t)k] = krng::channel_base(12345, (uint64_t)k);
    auto b_noise = [&kpos](long n) {
        uint64_t acc = 0;
        int k = 0;
        for (long i = 0; i < n; i += 2) {
            uint64_t p = kpos[(size_t)k];
            p += krng::kGamma; acc ^= krng::mix(p);
            p += krng::kGamma; acc ^= krng::mix(p);
            kpos[(size_t)k] = p;
            k = (k + 1) & (K - 1);
        }
        g_sink = g_sink ^ acc;
    };
    auto a_pos = [](long n) {
        ShotRng r;
        uint64_t acc = 0;
        for (long i = 0; i < n; ++i) { r.seed(12345, (uint64_t)i); acc ^= r.s; }
        g_sink = g_sink ^ acc;
    };
    auto b_pos = [](long n) {
        krng::Stream st; st.set_key(12345, krng::COIN);
        uint64_t acc = 0;
        for (long i = 0; i < n; ++i) { st.at((uint64_t)i); acc ^= st.base; }
        g_sink = g_sink ^ acc;
    };

    struct Row { const char* name; void (*dummy)(); };
    const char* names[4] = {"uniform (mt19937_64+U vs keyed u01)", "word (ShotRng vs keyed)",
                            "noise (xoshiro vs keyed per-channel)", "position (ShotRng::seed vs at)"};
    std::vector<double> A[4], B[4], R[4];
    for (int rd = 0; rd < rounds; ++rd) {
        for (int row = 0; row < 4; ++row) {
            double ta = 0, tb = 0;
            auto run = [&](bool a) {
                switch (row) {
                    case 0: (a ? time_ns(a_uniform, N / 8) : time_ns(b_uniform, N / 8));
                            return a ? time_ns(a_uniform, N) : time_ns(b_uniform, N);
                    case 1: (a ? time_ns(a_word, N / 8) : time_ns(b_word, N / 8));
                            return a ? time_ns(a_word, N) : time_ns(b_word, N);
                    case 2: (a ? time_ns(a_noise, N / 8) : time_ns(b_noise, N / 8));
                            return a ? time_ns(a_noise, N) : time_ns(b_noise, N);
                    default: (a ? time_ns(a_pos, N / 8) : time_ns(b_pos, N / 8));
                             return a ? time_ns(a_pos, N) : time_ns(b_pos, N);
                }
            };
            if (rd & 1) { tb = run(false); ta = run(true); }
            else        { ta = run(true);  tb = run(false); }
            A[row].push_back(ta); B[row].push_back(tb); R[row].push_back(tb / ta);
        }
    }
    std::printf("rounds=%d draws/block=%ld (warm-up block discarded, A/B interleaved)\n", rounds, N);
    for (int row = 0; row < 4; ++row) {
        std::vector<double> r = R[row];
        std::sort(r.begin(), r.end());
        std::printf("%-40s A=%.3f ns  B=%.3f ns  B/A median=%.4f [q25 %.4f, q75 %.4f]\n", names[row],
                    median(A[row]), median(B[row]), r[r.size() / 2], r[r.size() / 4],
                    r[(3 * r.size()) / 4]);
    }
    return 0;
}
