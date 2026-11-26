// test_xdem_speed_tripwire — order-of-magnitude speed guard for qeccore::xdem::XdemSampler (xdem Task 12, SPEC §4.7).
// Not a benchmark (that is an internal speed suite): each case times one path at 1 thread on a warm sampler,
// takes the BEST of kReps calls (load only ever adds time), and fails only if that best is above a generous cap: 10x
// the µs/shot this build measured (best of 3) on the in-repo fixtures. Machine noise and load cannot trip it; a
// regression of the "cache thrown away / O(n^2) per shot / accidental general path" kind (an order of magnitude) does.
//
// Cases, on ex422_obs.xdem and cultivation_d3_faithful.xdem: sample(), sample(fired), score_lines "r2b" and "optimal"
// with H = the fired lines. Also asserts that both files take the fast path (a fast-path file silently falling to the
// general path is itself a speed regression).
// argv[1] = tests/data/xdem. Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_sampler.hpp"

using namespace qeccore::xdem;

static std::string g_dir;
constexpr int kReps = 3;

static std::string slurp(const std::string& name) {
    std::ifstream f(g_dir + "/" + name);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Caps {
    double sample, fired, r2b, optimal;   // µs/shot
};

static void check_case(const char* file, size_t n, const Caps& cap) {
    XdemSampler S(parse(slurp(file)));
    CHECK(S.fast_path());
    CHECK(S.r2b_available().first);
    std::vector<uint8_t> d(n * S.det_bytes()), o(n * S.obs_bytes());
    std::vector<int32_t> c(n);
    std::vector<double> out(n);
    XdemFired f;
    S.sample(n, 7300, 1, d.data(), o.data(), c.data());               // warm (thread slot, buffers, C table)
    S.sample(n, 7301, 1, d.data(), o.data(), c.data(), 0, &f);
    double best_s = 1e300, best_f = 1e300, best_r = 1e300, best_o = 1e300;
    for (int r = 0; r < kReps; ++r) {
        double t0 = now_s();
        S.sample(n, 7310 + (uint64_t)r, 1, d.data(), o.data(), c.data());
        best_s = std::min(best_s, 1e6 * (now_s() - t0) / (double)n);
        t0 = now_s();
        S.sample(n, 7320 + (uint64_t)r, 1, d.data(), o.data(), c.data(), 0, &f);
        best_f = std::min(best_f, 1e6 * (now_s() - t0) / (double)n);
        for (Correction k : {Correction::R2b, Correction::Optimal}) {
            XdemLinesOptions opt;
            opt.corr = k;
            t0 = now_s();
            S.score_lines(n, d.data(), f.off.data(), f.ids.data(), f.ids.size(), f.off.data(), f.ids.data(),
                          f.ids.size(), out.data(), opt);
            double& best = k == Correction::R2b ? best_r : best_o;
            best = std::min(best, 1e6 * (now_s() - t0) / (double)n);
        }
    }
    std::printf("  %-28s best us/shot (cap) over %zu shots x %d: sample %.4f (%.2f) fired %.4f (%.2f) "
                "r2b %.4f (%.2f) optimal %.4f (%.2f)\n",
                file, n, kReps, best_s, cap.sample, best_f, cap.fired, best_r, cap.r2b, best_o, cap.optimal);
    CHECK(best_s < cap.sample);
    CHECK(best_f < cap.fired);
    CHECK(best_r < cap.r2b);
    CHECK(best_o < cap.optimal);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: test_xdem_speed_tripwire <tests/data/xdem>\n");
        return 2;
    }
    g_dir = argv[1];
    // caps >= 10x the worst of 6 runs of this binary at 6ac89f3 (best of 3 per run, 1 thread, load ~2.1 on 48 cores;
    // µs/shot, Task 12 report): ex422_obs sample 0.0045 fired 0.0058 r2b 0.0075 optimal 0.0075; cultivation_d3_faithful
    // sample 0.0378 fired 0.0400 r2b 0.0504 optimal 0.0545
    check_case("ex422_obs.xdem", 4000000, {0.05, 0.06, 0.08, 0.08});
    check_case("cultivation_d3_faithful.xdem", 400000, {0.38, 0.40, 0.51, 0.55});
    REPORT();
}
