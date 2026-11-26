// test_xdem_sampler_concurrency — qeccore::xdem::XdemSampler's C table read while another thread samples (ported in
// xdem Task 11 from the dc-DEM v1 sampler's race test, final review I1 of that route).
//
// The file is synthetic: O = 20 observables, each with its own axis (img L_a, gauge g_a) and its own rotating line
// (p = 0.05), so a shot's C is spanned by the observables of the odd axes it fired and the table grows on every call
// (thousands of distinct C, fast path). One thread samples with fresh seeds at 2 threads; the other reads the table
// in a loop through cspaces_snapshot() / num_cspaces() (the locked reads the Python binding uses). Every snapshot must
// be a prefix of the final table (the table is append-only and entry 0 is {0}), and sizes must never decrease. Under
// -fsanitize=thread (or address) this is the race test; -DXDEM_TEST_UNLOCKED_READ (test-of-test only) reads
// cspaces() unlocked instead, which TSan reports.
// Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_sampler.hpp"

using namespace qeccore::xdem;

static std::string many_c_text(int O) {
    const int L = O + 1;
    std::string t = "xdem 1 detectors 1 observables " + std::to_string(O) + " gauge " + std::to_string(O) +
                    " hidden 0 tc " + std::to_string(L) + "\n";
    for (int a = 0; a < O; ++a)
        t += "axis Q" + std::to_string(a) + " img L" + std::to_string(a) + " gauge g" + std::to_string(a) + "\n";
    for (int a = 0; a < O; ++a) t += "error(0.05) rot +Q" + std::to_string(a) + "\n";
    t += "error(0.01) D0\n";
    return t;
}

static void snapshot_while_sampling() {
    XdemSampler S(parse(many_c_text(20)));
    CHECK(S.fast_path());
    const size_t n = 20000;
    std::vector<uint8_t> d(n * S.det_bytes()), o(n * S.obs_bytes());
    std::vector<int32_t> c(n);
    std::atomic<bool> done{false};
    std::atomic<int> reads{0};
    std::vector<std::vector<std::vector<uint64_t>>> snaps;
    std::vector<size_t> sizes;
    std::thread reader([&] {
        while (!done.load()) {
#ifdef XDEM_TEST_UNLOCKED_READ
            std::vector<std::vector<uint64_t>> t = S.cspaces();
#else
            std::vector<std::vector<uint64_t>> t = S.cspaces_snapshot();
#endif
            sizes.push_back(S.num_cspaces());
            if (snaps.size() < 64) snaps.push_back(std::move(t));
            else snaps.back() = std::move(t);
            reads.fetch_add(1);
        }
    });
    for (int it = 0; it < 30; ++it) {
        // the mutex is not fair: let the reader in between calls (its reads still overlap the next call)
        const int r0 = reads.load();
        S.sample(n, 900 + (uint64_t)it, 2, d.data(), o.data(), c.data());
        while (reads.load() == r0) std::this_thread::yield();
    }
    done.store(true);
    reader.join();
    const auto& fin = S.cspaces();   // single-threaded from here
    int bad_prefix = 0, bad_order = 0;
    for (const auto& t : snaps) {
        bad_prefix += t.empty() || !t[0].empty() || t.size() > fin.size();
        for (size_t i = 0; i < t.size() && i < fin.size(); ++i) bad_prefix += t[i] != fin[i];
    }
    for (size_t i = 1; i < sizes.size(); ++i) bad_order += sizes[i] < sizes[i - 1];
    CHECK_EQ(bad_prefix, 0);
    CHECK_EQ(bad_order, 0);
    CHECK(fin.size() > 1000);                             // the table did grow while it was read
    CHECK(sizes.size() >= 30);
    CHECK(S.cspaces_snapshot(1, 3).size() == 3 && S.cspaces_snapshot(1, 3)[0] == fin[1]);
    CHECK(S.cspaces_snapshot(fin.size() + 5).empty());   // clipped
    std::printf("snapshot_while_sampling: %zu reads, table %zu entries at the end\n", sizes.size(), fin.size());
}

int main() {
    RUN(snapshot_while_sampling);
    REPORT();
}
