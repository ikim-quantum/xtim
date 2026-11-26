// test_xdem_sampler_golden — byte references for qeccore::xdem::XdemSampler (xdem Task 6).
//
// FNV-1a hashes of (dets, obs, cspace, C table) of sample(5000, seed 7, threads 1 / 8) on a fresh sampler, and of the
// counters (shots, ext_shots, coin_shots, num_cspaces), for
//   * synth/surface_d5r5.dem — the conventional surface-code d = 5, 5-round DEM (stim generated, p = 1e-3): its hash
//     is the dc-DEM v1 sampler's (plain-DEM mode) at BASE 19f2f78 (computed with the then in-tree v1 code, unchanged
//     since BASE, by a scratch driver; recorded in the Task 6 report), so a conventional DEM samples as today (SPEC §6
//     T4);
//   * ex422_obs.xdem (SPEC §3.7), cultivation_d3_faithful.xdem (writer fixture), synth/boundary_r64.xdem (fast path at
//     its bound) and synth/boundary_r65.xdem (general path): new references pinned at the Task 6 commit (a declared
//     change: SPEC §6 T5 does not promise the dc v1 bytes), each on both paths (force_general) where they differ.
//     XFM: boundary_r65 is on the multi-word fast path, whose stream is the general path's: the Task 6 bytes on both.
// Threads 1 and 8 must give the same hash. argv[1] = tests/data/xdem; argv[2] = --print (regenerate the table).
// Must end main() with REPORT().
#include "check.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_sampler.hpp"

using namespace qeccore::xdem;

static std::string g_dir;
static bool g_print = false;

static std::string slurp(const std::string& name) {
    std::ifstream f(g_dir + "/" + name);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t call_hash(XdemSampler& S, size_t n, uint64_t seed, int threads) {
    std::vector<uint8_t> d(n * S.det_bytes()), o(n * S.obs_bytes());
    std::vector<int32_t> c(n);
    S.sample(n, seed, threads, d.data(), o.data(), c.data(), 0);
    uint64_t h = 1469598103934665603ull;
    h = fnv(d.data(), d.size(), h);
    h = fnv(o.data(), o.size(), h);
    h = fnv(c.data(), c.size() * 4, h);
    for (const auto& t : S.cspaces()) h = fnv(t.data(), t.size() * 8, h);
    return h;
}

static uint64_t counter_hash(const XdemSampler& S) {
    const auto k = S.counters();
    const uint64_t cv[] = {k.shots, k.ext_shots, k.coin_shots, k.num_cspaces};
    return fnv(cv, sizeof cv, 1469598103934665603ull);
}

struct Golden {
    const char* file;
    int general;
    bool fast;     // the expected fast_path()
    uint64_t h, ch;   // sample(5000, 7, threads) hash (threads 1 and 8), counters hash
};

// clang-format off
static const Golden kGolden[] = {
    {"synth/surface_d5r5.dem", 0, true, 0x29eebf71843c3097ull, 0x25803b188e1375f7ull},
    {"ex422_obs.xdem", 0, true, 0x5c736342b3e42bedull, 0x5bd01c5ae1bb9097ull},
    {"ex422_obs.xdem", 1, false, 0x38ed9b61230d3515ull, 0x5bd01c5ae1bb9097ull},
    {"cultivation_d3_faithful.xdem", 0, true, 0xaf7475ed0f9d72bdull, 0x0a7e167a85a7543dull},
    {"cultivation_d3_faithful.xdem", 1, false, 0x3408d254315c258eull, 0x0a7e167a85a7543dull},
    {"synth/boundary_r64.xdem", 0, true, 0xabb137fad12e0e64ull, 0xd7e11bcaca2db144ull},
    {"synth/boundary_r64.xdem", 1, false, 0xbc414345579fd543ull, 0xd7e11bcaca2db144ull},
    {"synth/boundary_r65.xdem", 0, true, 0x9bfe459f5e21ff0bull, 0xd7e11bcaca2db144ull},
    {"synth/boundary_r65.xdem", 1, false, 0x9bfe459f5e21ff0bull, 0xd7e11bcaca2db144ull},
};
// clang-format on

static void golden() {
    const size_t ng = sizeof(kGolden) / sizeof(kGolden[0]);
    for (size_t i = 0; i < ng; ++i) {
        const Golden& G = kGolden[i];
        uint64_t hs[2], cs[2];
        int k = 0;
        for (int th : {1, 8}) {
            XdemSampler S(parse(slurp(G.file)), kDefaultCacheCap, G.general != 0);
            CHECK(S.fast_path() == G.fast);
            hs[k] = call_hash(S, 5000, 7, th);
            cs[k] = counter_hash(S);
            ++k;
        }
        CHECK(hs[0] == hs[1] && cs[0] == cs[1]);
        if (g_print) {
            std::printf("    {\"%s\", %d, %s, 0x%016" PRIx64 "ull, 0x%016" PRIx64 "ull},\n", G.file, G.general,
                        G.fast ? "true" : "false", hs[0], cs[0]);
            continue;
        }
        const bool ok = G.h == hs[0] && G.ch == cs[0];
        if (!ok) std::fprintf(stderr, "  golden mismatch: %s general=%d\n", G.file, G.general);
        CHECK(ok);
    }
}

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../tests/data/xdem";
    g_print = argc > 2 && std::strcmp(argv[2], "--print") == 0;
    RUN(golden);
    REPORT();
}
