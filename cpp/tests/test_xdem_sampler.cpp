// test_xdem_sampler — qeccore::xdem::XdemSampler (SPEC §1.2, §4.7).
//
//  * ordered composition (Review Focus 1): the ORDER file (lines 0 and 2 share p = 0.5, so one chain walked first;
//    line 1, p = 0.6, negates line 2's rotation): every shot that fired all three lines has D0 = 0 (the file-order
//    composite; the walk order 0, 2, 1 gives D0 = 1), over 2e4 shots on both paths; and the forced sets {0, 2} -> D0 = 1,
//    {0, 1, 2} in any order -> D0 = 0, {0, 0, 1, 2, 2} -> {1} -> D0 = 0 on every shot; sample() without fired lists
//    gives the fired call's bytes;
//  * stream identity: identical bytes (dets, obs, cspace, C table) at threads 1/2/8 and on a repeated call, both paths;
//    different bytes for seeds 1 and 2; sample(n) is a prefix of sample(m); first_block = b returns shots
//    [1024 b, ...) of the stream; fired = true gives the same bytes;
//  * lossy_frame_e4 (ERRATA E4): the optimal correction's frame on lossy observables, both law paths.
// argv[1] = tests/data/xdem. Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_usable_size (the padded-block check)
#endif
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_sampler.hpp"

using namespace qeccore::xdem;

static std::string g_dir;

static std::string slurp(const std::string& name) {
    std::ifstream f(g_dir + "/" + name);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static const char* kOrder =
    "xdem 1 detectors 1 observables 0 gauge 1 hidden 1 tc 3\n"
    "axis Q0 img D0 gauge g0\n"
    "axis Q1 img D0 gauge g0 hidden h0\n"
    "error(0.5) rot +Q1\n"
    "error(0.6) x Q0\n"
    "error(0.5) rot +Q0\n";

struct Out {
    std::vector<uint8_t> d, o;
    std::vector<int32_t> c;
    std::vector<std::vector<uint64_t>> table;
    XdemFired f;
};

static Out run(XdemSampler& S, size_t n, uint64_t seed, int threads, uint64_t fb = 0, bool fired = false) {
    Out r;
    r.d.resize(n * S.det_bytes());
    r.o.resize(n * S.obs_bytes());
    r.c.resize(n);
    S.sample(n, seed, threads, r.d.data(), r.o.data(), r.c.data(), fb, fired ? &r.f : nullptr);
    r.table = S.cspaces();
    return r;
}

static void ordered_composition() {
    for (bool general : {false, true}) {
        XdemSampler S(parse(kOrder), kDefaultCacheCap, general);
        CHECK(S.fast_path() == !general);
        const size_t n = 20000;
        const Out r = run(S, n, 1, 1, 0, true);
        size_t all3 = 0, bad = 0;
        for (size_t s = 0; s < n; ++s) {
            const int64_t a = r.f.off[s], b = r.f.off[s + 1];
            if (b - a == 3 && r.f.ids[a] == 0 && r.f.ids[a + 1] == 1 && r.f.ids[a + 2] == 2) {
                ++all3;
                bad += r.d[s] & 1u;
            }
        }
        std::fprintf(stderr, "  ORDER (%s): %zu of %zu shots fired lines 0, 1, 2; D0 = 1 on %zu of them\n",
                     general ? "general" : "fast", all3, n, bad);
        CHECK(all3 > 1000);
        CHECK_EQ(bad, (size_t)0);
        const Out plain = run(S, n, 1, 1);   // sample() (no fired lists: the walk-gather path) gives the same bytes
        CHECK(plain.d == r.d);
        const std::vector<std::pair<std::vector<int>, int>> fixed = {
            {{0, 2}, 1}, {{0, 1, 2}, 0}, {{2, 1, 0}, 0}, {{1, 2, 0}, 0}, {{0, 0, 1, 2, 2}, 0}, {{0, 2, 1, 1}, 1}};
        for (const auto& [ids, want] : fixed) {
            XdemSampler F(parse(kOrder), kDefaultCacheCap, general);
            std::vector<uint8_t> d(500), o(0);
            std::vector<int32_t> c(500);
            F.sample_fixed(ids, 500, 3, d.data(), o.data(), c.data());
            size_t wrong = 0;
            for (uint8_t x : d) wrong += (int)(x & 1u) != want;
            CHECK_EQ(wrong, (size_t)0);
        }
    }
}

static void stream_identity(const char* file, bool general) {
    const std::string text = slurp(file);
    const size_t n = 20000;
    Out ref;
    {
        XdemSampler S(parse(text), kDefaultCacheCap, general);
        CHECK(S.fast_path() == !general);
        ref = run(S, n, 1, 1);
    }
    for (int th : {2, 8}) {
        XdemSampler S(parse(text), general ? 3 : kDefaultCacheCap, general);   // general: a tiny cache, same bytes
        const Out r = run(S, n, 1, th);
        CHECK(r.d == ref.d && r.o == ref.o && r.c == ref.c && r.table == ref.table);
    }
    {
        XdemSampler S(parse(text), kDefaultCacheCap, general);
        const Out a = run(S, n, 1, 8);
        const Out b = run(S, n, 1, 1);   // repeated call, same seed
        CHECK(a.d == ref.d && b.d == ref.d && a.o == b.o && a.c == b.c);
        const Out c = run(S, n, 2, 8);
        CHECK(c.d != ref.d);
        const Out f = run(S, n, 1, 3, 0, true);   // fired: the same bytes
        CHECK(f.d == ref.d && f.o == ref.o && f.c == ref.c);
        CHECK_EQ(f.f.off.size(), n + 1);
    }
    {
        XdemSampler S(parse(text), kDefaultCacheCap, general);
        const Out p = run(S, 5000, 1, 3);   // prefix
        CHECK(std::memcmp(p.d.data(), ref.d.data(), p.d.size()) == 0);
        CHECK(std::memcmp(p.o.data(), ref.o.data(), p.o.size()) == 0);
        const Out q = run(S, 4096, 1, 4, 7);   // first_block = 7: shots [7168, 11264)
        const size_t s0 = 7 * 1024;
        CHECK(std::memcmp(q.d.data(), ref.d.data() + s0 * S.det_bytes(), q.d.size()) == 0);
        CHECK(std::memcmp(q.o.data(), ref.o.data() + s0 * S.obs_bytes(), q.o.size()) == 0);
    }
}

static void stream_identities() {
    for (const char* f : {"cultivation_d5.xdem", "ex422_obs.xdem", "synth/boundary_r64.xdem"})
        for (bool g : {false, true}) stream_identity(f, g);
}

// Task 12 fix round 1: after sampling at 8 threads (plain, then fired, then plain), no 64-byte line holds per-shot
// scratch of two different thread slots (the false sharing that cost +33-40 % at 8-16 threads), and every scratch
// block is 64-byte aligned. Time-free: addresses only.
static void slot_scratch_lines_are_private() {
    XdemSampler S(parse(slurp("cultivation_d3_faithful.xdem")));
    for (int fired = 0; fired < 3; ++fired) run(S, 40000, 5 + (uint64_t)fired, 8, 0, fired == 1);
    const auto sp = S.slot_scratch_spans();
    CHECK(sp.size() == 8);
    std::unordered_map<uintptr_t, size_t> owner;   // cache line -> slot
    bool shared = false, aligned = true;
    for (size_t t = 0; t < sp.size(); ++t)
        for (const auto& [a, n] : sp[t]) {
            aligned = aligned && a % 64 == 0;
            for (uintptr_t l = a / 64; l <= (a + n - 1) / 64; ++l) {
                auto it = owner.emplace(l, t).first;
                shared = shared || it->second != t;
            }
        }
    CHECK(aligned);
    CHECK(!shared);
#if defined(__GLIBC__)
    // the padded block itself (fr1 re-review m3): every scratch block owns whole lines, so no foreign allocation can
    // sit in the tail of a scratch line. slot_scratch_spans reports each block padded to whole lines; the heap block
    // behind it must be at least that large (a mutant that keeps the alignment but drops the round-up fails here).
    bool padded = true;
    for (const auto& v : sp)
        for (size_t i = 1; i < v.size(); ++i)   // [0] = the ThreadState (alignas(64): sizeof is whole lines)
            padded = padded && v[i].second % 64 == 0 && malloc_usable_size((void*)v[i].first) >= v[i].second;
    CHECK(padded);
    for (size_t n : {1, 2, 7, 8, 9, 16, 17}) {   // the allocator directly: a 64-byte aligned block of whole lines
        XdemLineAlloc<uint64_t> a;
        uint64_t* p = a.allocate(n);
        const size_t want = (n * 8 + 63) / 64 * 64;
        CHECK((uintptr_t)p % 64 == 0 && malloc_usable_size(p) >= want);
        a.deallocate(p, n);
    }
#endif
}

// Task 12 fix round 1: the general path computes its laws over kappa re-based to a lower-weight gauge basis. The
// weight never rises and falls on the canonical files whose basis is dense; on 300 random fired sets per file the
// general path's status (offset modulo the coins, coin span, C id content) equals the fast path's (file basis).
static std::vector<std::vector<uint64_t>> span_rref(std::vector<Bits> v) {   // fully reduced, pivot = highest bit
    std::vector<std::vector<uint64_t>> b;
    for (auto& x : v) {
        std::vector<uint64_t> y = x.w;
        for (const auto& r : b) {
            int p = -1;
            for (size_t w = r.size(); w-- > 0 && p < 0;)
                if (r[w]) p = (int)(w * 64 + 63 - (size_t)__builtin_clzll(r[w]));
            if ((y[(size_t)p >> 6] >> (p & 63)) & 1ull)
                for (size_t w = 0; w < y.size(); ++w) y[w] ^= r[w];
        }
        bool nz = false;
        for (uint64_t w : y) nz = nz || w;
        if (!nz) continue;
        int p = -1;
        for (size_t w = y.size(); w-- > 0 && p < 0;)
            if (y[w]) p = (int)(w * 64 + 63 - (size_t)__builtin_clzll(y[w]));
        for (auto& r : b)
            if ((r[(size_t)p >> 6] >> (p & 63)) & 1ull)
                for (size_t w = 0; w < y.size(); ++w) r[w] ^= y[w];
        b.push_back(y);
    }
    std::sort(b.begin(), b.end(), [](const auto& x, const auto& y) {
        return std::lexicographical_compare(x.rbegin(), x.rend(), y.rbegin(), y.rend());
    });
    return b;
}

static std::vector<uint64_t> reduce_by(std::vector<uint64_t> x, const std::vector<std::vector<uint64_t>>& b) {
    for (const auto& r : b) {
        int p = -1;
        for (size_t w = r.size(); w-- > 0 && p < 0;)
            if (r[w]) p = (int)(w * 64 + 63 - (size_t)__builtin_clzll(r[w]));
        if ((x[(size_t)p >> 6] >> (p & 63)) & 1ull)
            for (size_t w = 0; w < x.size(); ++w) x[w] ^= r[w];
    }
    return x;
}

static void general_path_rebasing() {
    long long fell = 0;
    uint64_t rng = 12345;
    auto next = [&] { return rng = rng * 6364136223846793005ull + 1442695040888963407ull; };
    for (const char* f : {"cultivation_d5.xdem", "color5_cultivation.xdem", "cultivation_d3_faithful.xdem",
                          "miniature_oracle.xdem", "ex422_obs.xdem"}) {
        const std::string text = slurp(f);
        XdemSampler G(parse(text), kDefaultCacheCap, true), Q(parse(text));
        CHECK(!G.fast_path() && Q.fast_path());
        const auto [wf, wl] = G.law_kappa_weights();
        std::fprintf(stderr, "  %s: kappa weight file %lld, re-based %lld\n", f, wf, wl);
        CHECK(wl <= wf);
        fell += wl < wf;
        CHECK(Q.law_kappa_weights() == std::make_pair(0LL, 0LL));
        size_t bad = 0;
        for (int t = 0; t < 300; ++t) {
            std::vector<int> ids;
            const int k = 1 + (int)(next() >> 60);
            for (int i = 0; i < k; ++i) ids.push_back((int)((next() >> 33) % G.num_lines()));
            Bits og, oq;
            std::vector<Bits> cg, cq;
            const int ig = G.status_of(ids, og, cg), iq = Q.status_of(ids, oq, cq);
            const auto bg = span_rref(cg), bq = span_rref(cq);
            bad += bg != bq || reduce_by(og.w, bg) != reduce_by(oq.w, bq) ||
                   G.cspaces_snapshot((size_t)ig, 1) != Q.cspaces_snapshot((size_t)iq, 1);
        }
        CHECK_EQ(bad, (size_t)0);
    }
    CHECK(fell > 0);
}

// fr1 re-review m4: r > 64 (multiword kappa, no fast path to compare with). On synth/boundary_r65 the forced general
// path's status_of (lookup) and law_of (the cache entry) equal the law over the FILE basis, law(F, compose(F, ids)):
// same coin span, same offset modulo the coins, and law_of's dim C = rank(coins) - rank(coins restricted to D).
static void general_path_rebasing_r65() {
    const std::string text = slurp("synth/boundary_r65.xdem");
    const File F = parse(text);
    CHECK(F.r > 64);
    XdemSampler G(parse(text), kDefaultCacheCap, true);
    CHECK(!G.fast_path());
    const auto [wf, wl] = G.law_kappa_weights();
    std::fprintf(stderr, "  boundary_r65 (r = %d): kappa weight file %lld, re-based %lld\n", F.r, wf, wl);
    CHECK(wl <= wf && wf > 0);
    uint64_t rng = 777;
    auto next = [&] { return rng = rng * 6364136223846793005ull + 1442695040888963407ull; };
    size_t bad = 0, nontrivial = 0;
    const size_t nw = words_for((long long)F.D + F.O);
    for (int t = 0; t < 2000; ++t) {
        std::vector<int> ids;
        const int k = 1 + (int)(next() >> 60);
        for (int i = 0; i < k; ++i) ids.push_back((int)((next() >> 33) % G.num_lines()));
        const Law ref = law(F, compose(F, ids));
        std::vector<Bits> rc = ref.coins;
        const auto br = span_rref(rc);
        nontrivial += !br.empty();
        Bits os, ol;
        std::vector<Bits> cs, cl;
        G.status_of(ids, os, cs);
        const int dimc = G.law_of(ids, ol, cl);
        const auto bs = span_rref(cs), bl = span_rref(cl);
        const auto ro = reduce_by(ref.offset.w, br);
        bad += bs != br || bl != br || reduce_by(os.w, bs) != ro || reduce_by(ol.w, bl) != ro;
        std::vector<Bits> cd = ref.coins;   // restricted to the detector bits
        for (auto& c : cd)
            for (size_t w = 0; w < nw; ++w) {
                const long long lo = 64LL * (long long)w;
                if (lo >= F.D) c.w[w] = 0;
                else if (lo + 64 > F.D) c.w[w] &= (1ull << (F.D - lo)) - 1;
            }
        bad += dimc != (int)br.size() - (int)span_rref(cd).size();
    }
    std::fprintf(stderr, "  boundary_r65: %zu of 2000 sets with coins\n", nontrivial);
    CHECK(nontrivial > 0);
    CHECK_EQ(bad, (size_t)0);
}

// ERRATA E4: the optimal correction's frame on lossy observables = the point of law([h, K])'s support restricted to
// them at d nearest h's flips (observable 0 first). kPair: L0 + L1 = d, both lossy; at d = 1 the frame is (0, 1) and
// the pass 1/2 (f_K = 0 and "none" score 0). kIndep: independent lossy observables, 1/4, frame = h's flips (1, 0).
static void lossy_frame_e4() {
    static const char* kPair =
        "xdem 1 detectors 1 observables 2 gauge 2 hidden 0 tc 1\n"
        "axis Q0 img D0 L0 gauge g0\n"
        "axis Q1 img D0 L1 gauge g1\n"
        "error(0.3) rot +Q0 +Q1\n";
    static const char* kIndep =
        "xdem 1 detectors 2 observables 2 gauge 4 hidden 0 tc 1\n"
        "axis Q0 img D0 L0 gauge g0\n"
        "axis Q1 img D0 gauge g1\n"
        "axis Q2 img D1 L1 gauge g2\n"
        "axis Q3 img D1 gauge g3\n"
        "error(0.3) L0 rot +Q0 +Q1 +Q2 +Q3\n";
    struct Case {
        const char* text;
        uint8_t d;
        double opt, none;
        uint8_t f0, f1;
    };
    const Case cases[] = {{kPair, 0, 0.5, 0.5, 0, 0}, {kPair, 1, 0.5, 0.0, 0, 1}, {kIndep, 0, 0.25, 0.25, 1, 0},
                          {kIndep, 3, 0.25, 0.25, 1, 0}};
    for (const Case& c : cases)
        for (bool general : {false, true}) {
            XdemSampler S(parse(c.text), 65536, general);
            const int64_t off[2] = {0, 1};
            const int32_t ids[1] = {0};
            const uint8_t dets[1] = {c.d};
            double p = -1;
            XdemLinesOptions o;
            S.score_lines(1, dets, off, ids, 1, off, ids, 1, &p, o);
            CHECK_EQ(p, c.opt);
            o.corr = Correction::None;
            S.score_lines(1, dets, off, ids, 1, off, ids, 1, &p, o);
            CHECK_EQ(p, c.none);
            XdemCorrectionOut x;
            S.correction(1, dets, off, ids, 1, Correction::Optimal, x);
            CHECK(x.lossy.size() == 2 && x.lossy[0] && x.lossy[1]);
            CHECK(x.frame.size() == 2 && x.frame[0] == c.f0 && x.frame[1] == c.f1);
            std::vector<KObs> k;
            const uint64_t sw[2] = {c.d, 0};
            S.k_optimal(ids, 1, sw, k);
            CHECK(k.size() == 2 && k[0].frame == c.f0 && k[1].frame == c.f1);
        }
    // E4 fix round 1 (SPEC §7 Q1): d outside law(h)'s detector support -> c = 0 on the lossy observables. kIndep plus
    // D2, flipped by line 1 only: h = {0}, d = D2 is outside (f_h = L0 would give frame (1, 0))
    static const char* kOut =
        "xdem 1 detectors 3 observables 2 gauge 4 hidden 0 tc 2\n"
        "axis Q0 img D0 L0 gauge g0\n"
        "axis Q1 img D0 gauge g1\n"
        "axis Q2 img D1 L1 gauge g2\n"
        "axis Q3 img D1 gauge g3\n"
        "error(0.3) L0 rot +Q0 +Q1 +Q2 +Q3\n"
        "error(0.1) D2\n";
    for (bool general : {false, true}) {
        XdemSampler S(parse(kOut), 65536, general);
        const int64_t off[2] = {0, 1};
        const int32_t ids[1] = {0};
        const uint8_t dets[1] = {4};
        XdemCorrectionOut x;
        S.correction(1, dets, off, ids, 1, Correction::Optimal, x);
        CHECK(x.frame.size() == 2 && x.frame[0] == 0 && x.frame[1] == 0);
        std::vector<KObs> k;
        const uint64_t sw[2] = {4, 0};
        S.k_optimal(ids, 1, sw, k);
        CHECK(k.size() == 2 && k[0].frame == 0 && k[1].frame == 0);
    }
}

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../tests/data/xdem";
    RUN(ordered_composition);
    RUN(stream_identities);
    RUN(slot_scratch_lines_are_private);
    RUN(general_path_rebasing);
    RUN(general_path_rebasing_r65);
    RUN(lossy_frame_e4);
    REPORT();
}
