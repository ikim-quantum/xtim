// test_framed_kernels.cpp — S2k fix plan r2 (m5): function-level oracle for the framed kernels whose
// key loops were rewritten (set-bit iteration over key & low_mask(k); popcount parity for the
// Case-A classifiers). The pre-rewrite kernels (b3f2016, itself byte-identical to ab2ff72) are built
// in THIS TU from s2k_framed_ref.inc (verbatim, renamed *_ref); engine and oracle run on copies of
// the same random framed states and every output is compared bitwise: the whole post-state (frame
// rows x/z/phase, eps, free, k, entries' keys and coefficient bits), return values, Born pairs.
// States: random Clifford frames, random eps, k in {0,1,5,30,64} free coordinates, full or sparse
// random supports, random coefficients.
// S2b (owner 2026-10-16, no dead guards): the release-build low mask in the kernels' key loops is a
// Debug assert. Keys never carry bits at or above k (shift_key masks; every writer is clean), so:
//   * release: every post-state of every engine kernel here is checked to carry NO stray bit (an
//     exact counter, 0), on top of the bitwise oracle comparison on clean states;
//   * Debug (NDEBUG undefined): a state WITH stray bits makes each guarded kernel abort (death tests,
//     one per guarded loop family), and `--require-debug` fails a build where asserts are compiled out
//     (the Debug ctest config must run them; see cpp/CMakeLists.txt framed_kernels_debug_asserts).
#include "check.hpp"

#include <cmath>
#include <complex>
#include <csignal>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <functional>
#include <random>
#include <unordered_map>
#include <string>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli_kernels.hpp"

namespace qeccore {
namespace s2kref {
#include "s2k_framed_ref.inc"
}  // namespace s2kref
}  // namespace qeccore

using namespace qeccore;

namespace {

long g_cmp = 0;

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
long g_stray = 0;   // release: entries carrying a key bit at or above k after an engine kernel (must be 0)
void count_stray(const FramedSuperposition& A) {
    const AmpKey m = key_low_mask(A.k());
    for (const auto& e : A.entries()) g_stray += (e.first & ~m) != 0;
}
bool same_state(const FramedSuperposition& A, const FramedSuperposition& B) {
    count_stray(A);
    if (A.n() != B.n() || A.eps != B.eps || A.free != B.free || A.k() != B.k() || A.chi() != B.chi()) return false;
    if (A.alpha->k() != B.alpha->k()) return false;
    for (int a = 0; a < A.n(); ++a) {
        const Pauli *p[4] = {&A.U.Xrow[a], &B.U.Xrow[a], &A.U.Zrow[a], &B.U.Zrow[a]};
        for (int s = 0; s < 4; s += 2)
            if (p[s]->x != p[s + 1]->x || p[s]->z != p[s + 1]->z || (p[s]->phase & 3) != (p[s + 1]->phase & 3)) return false;
    }
    for (int i = 0; i < A.chi(); ++i) {
        if (A.entries()[i].first != B.entries()[i].first) return false;
        if (std::memcmp(&A.entries()[i].second, &B.entries()[i].second, sizeof(cd)) != 0) return false;
    }
    ++g_cmp;
    return true;
}

FramedSuperposition random_state(std::mt19937_64& rng, int n, int k, bool full) {
    FramedSuperposition f(n);
    std::uniform_int_distribution<int> qd(0, n - 1), kind(0, 7);
    for (int g = 0; g < 6 * n; ++g) {
        const int kd = kind(rng), a = qd(rng), b = qd(rng);
        if (kd >= 6 && a == b) continue;
        f.apply_clifford((uint8_t)kd, a, b);
    }
    for (int q = 0; q < n; ++q) f.eps[(size_t)q] = (uint8_t)(rng() & 1);
    std::vector<int> perm(n);
    for (int q = 0; q < n; ++q) perm[(size_t)q] = q;
    std::shuffle(perm.begin(), perm.end(), rng);
    f.free.assign(perm.begin(), perm.begin() + k);
    f.sync_alpha_k();
    std::normal_distribution<double> nd;
    const AmpKey mask = key_low_mask(k);
    if (k == 0) f.entries().emplace_back(AmpKey(0), cd(1, 0));
    else if (full && k <= 6) { for (AmpKey x = 0; x < (AmpKey(1) << k); ++x) f.entries().emplace_back(x, cd(nd(rng), nd(rng))); }
    else {
        std::unordered_map<AmpKey, int> seen;
        const int chi = 1 + (int)(rng() % 24);
        for (int i = 0; i < chi; ++i) { AmpKey x = rng() & mask; if (!seen.count(x)) { seen[x] = 1; f.entries().emplace_back(x, cd(nd(rng), nd(rng))); } }
    }
    f.alpha->normalize();
    return f;
}

Pauli random_pauli(std::mt19937_64& rng, int n, bool xs) {
    Pauli P(n);
    for (int q = 0; q < n; ++q) { if (xs && (rng() % 3 == 0)) P.setx(q); if (rng() % 3 == 0) P.setz(q); }
    P.phase = (int)(rng() & 3);
    return P;
}

// TLC: the engine kernels run on caller-owned scratch. mode 0: a fresh scratch per call; mode 1: ONE
// scratch reused by every call of every trial (warm from other shapes n, k, χ); mode 2: that reused
// scratch released before each trial. All three must equal the verbatim oracle bit for bit (every
// buffer is re-initialised before it is read, whatever the scratch's history).
void run_against_reference(int mode) {
    std::mt19937_64 rng(20261016);
    long trials = 0, anti = 0;
    FramedScratch warm;
    auto sc = [&](FramedScratch& fresh) -> FramedScratch& { return mode == 0 ? fresh : warm; };
    for (int trial = 0; trial < 600; ++trial) {
        if (mode == 2) warm.release();
        const int kk[5] = {0, 1, 5, 30, 64};
        const int k = kk[trial % 5];
        const int n = k == 64 ? 70 : (k == 30 ? 34 : 4 + (int)(rng() % 9));
        if (k > n) continue;
        const bool full = (trial / 5) % 2 == 0;
        FramedSuperposition base = random_state(rng, n, k, full);
        ++trials;
        // (1) canonicalise
        { FramedSuperposition A = base, B = base; FramedScratch f; framed_canonicalise(A, sc(f)); s2kref::framed_canonicalise_ref(B); CHECK(same_state(A, B)); }
        // (2) Born pairs on the frame (Case A: Z-only; Case B: with X)
        for (int r = 0; r < 4; ++r) {
            const Pauli Q = random_pauli(rng, n, r & 1);
            FramedScratch f;
            const auto a = framed_expectation_frame(base, Q, sc(f)), b = s2kref::framed_expectation_frame_ref(base, Q);
            CHECK(same_bits(a.first, b.first) && same_bits(a.second, b.second));
            ++g_cmp;
        }
        // (3) reference reads (Case-A split / anticommuting general + canonicalise), every qubit/axis/want
        for (int q = 0; q < n && q < 12; ++q)
            for (int pauli = 0; pauli < 3; ++pauli) {
                FramedSuperposition A = base, B = base;
                const int want = (int)(rng() & 1);
                FramedScratch f;
                const FramedRefRead ra = framed_reference_read_toward(A, pauli, q, want, sc(f));
                const FramedRefRead rb = s2kref::framed_reference_read_toward_ref(B, pauli, q, want);
                CHECK(ra.out == rb.out && same_bits(ra.pp, rb.pp) && same_bits(ra.pm, rb.pm) &&
                      ra.is_coin == rb.is_coin && ra.commuting == rb.commuting && ra.chi_pre == rb.chi_pre);
                CHECK(same_state(A, B));
                anti += !ra.commuting;
            }
        // (4) batch measurement (Case-A classifier caseA_eig + split, anticommuting reads, chi=1 pass)
        {
            std::vector<std::pair<int, int>> reads;
            for (int q = 0; q < n; ++q) reads.push_back({(int)(rng() % 3), q});
            FramedSuperposition A = base, B = base;
            std::mt19937_64 ra(trial), rb(trial);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            std::vector<int> oa, ob;
            FramedScratch f;
            batch_measure(A, reads, [&] { return U(ra); }, oa, sc(f));
            s2kref::batch_measure_ref(B, reads, [&] { return U(rb); }, ob);
            CHECK(oa == ob);
            CHECK(same_state(A, B));
        }
    }
    std::fprintf(stderr, "[framed_kernels] mode %d: %ld trials, %ld bitwise comparisons (%ld anticommuting reads): engine == verbatim pre-rewrite kernels; stray key bits in post-states: %ld\n",
                 mode, trials, g_cmp, anti, g_stray);
    CHECK(anti > 0);
    CHECK(g_stray == 0);
    if (mode == 1) CHECK(warm.charged_bytes() > 0);   // the warm scratch is charged while held ...
    const int64_t before = live_state_bytes();
    const int64_t held = warm.charged_bytes();
    warm.release();                                    // ... and release() credits exactly that charge
    CHECK(live_state_bytes() <= before - held);       // (the swapped entry buffer goes too)
    CHECK_EQ(warm.charged_bytes(), (int64_t)0);
}
void test_against_reference() { run_against_reference(0); }
void test_against_reference_warm() { run_against_reference(1); }
void test_against_reference_released() { run_against_reference(2); }

#ifndef NDEBUG
// Debug death tests: a stray key bit above k makes each guarded key loop's assert abort the process.
// One child per guarded loop family (canonicalise; the Born classifier of an X-carrying read; the
// anticommuting general measurement). Each child must die by SIGABRT from ITS loop's assert.
int g_deaths = 0;
// Runs fn in a child; true iff the child dies by SIGABRT from an assert in `where` (glibc's assert
// message names the function, so each death test pins ITS guarded loop, not an earlier one).
bool dies_by_abort(const std::function<void()>& fn, const char* where) {
    std::fflush(stderr);
    int fd[2];
    if (pipe(fd) != 0) return false;
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        dup2(fd[1], 2);
        fn();
        std::_Exit(0);                             // no abort: the guard is missing
    }
    close(fd[1]);
    std::string msg;
    char buf[512];
    for (ssize_t r; (r = read(fd[0], buf, sizeof buf)) > 0;) msg.append(buf, (size_t)r);
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT && msg.find(where) != std::string::npos &&
           msg.find("Assertion") != std::string::npos;
}
void test_stray_bits_abort_in_debug() {
    std::mt19937_64 rng(77);
    const int n = 9, k = 5;
    auto dirty = [&] {
        FramedSuperposition f = random_state(rng, n, k, true);
        for (auto& e : f.entries()) e.first |= AmpKey(1) << (k + (int)(rng() % 7));
        return f;
    };
    {
        FramedSuperposition f = dirty();
        CHECK(dies_by_abort([&] { framed_canonicalise(f); }, "framed_canonicalise"));
        ++g_deaths;
    }
    {
        FramedSuperposition f = dirty();
        Pauli Q(n);
        for (int q = 0; q < n; ++q) { Q.setx(q); Q.setz(q); }
        CHECK(dies_by_abort([&] { (void)framed_expectation_frame(f, Q); }, "framed_born_from_conjugated"));
        ++g_deaths;
    }
    {
        // The anticommuting collapse itself (its own key loops), entered directly so neither the
        // Born classifier nor canonicalise runs first: a lab read whose frame image has X support.
        FramedSuperposition f = dirty();
        int qa = -1, pa = -1;
        Pauli P(n), Q(n);
        for (int q = 0; q < n && qa < 0; ++q)
            for (int p = 0; p < 3 && qa < 0; ++p) {
                Pauli Pl(n);
                if (p != 2) Pl.setx(q);
                if (p != 0) Pl.setz(q);
                const Pauli Ql = f.U.dual_valid() ? f.U.conjugate(Pl) : f.U.dual_image(Pl);
                bool xs = false;
                for (int a = 0; a < n; ++a) xs = xs || Ql.xbit(a);
                if (xs && p == 2) { qa = q; pa = p; P = Pl; Q = Ql; }
            }
        CHECK(qa >= 0);
        std::vector<int> A;
        for (int a = 0; a < n; ++a) if (Q.xbit(a)) A.push_back(a);
        CHECK(dies_by_abort([&] { FramedScratch fs; (void)framed_measure_anticommuting_general(f, P, Q, A, 0.0, fs); },
                           "framed_measure_anticommuting_general"));
        ++g_deaths;
        (void)pa;
    }
    std::fprintf(stderr, "[framed_kernels] Debug death tests: %d (each guarded key loop aborts on a stray bit)\n", g_deaths);
    CHECK(g_deaths == 3);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const bool require_debug = argc > 1 && std::string(argv[1]) == "--require-debug";
#ifdef NDEBUG
    if (require_debug) {
        std::fprintf(stderr, "FAIL: --require-debug on a build with NDEBUG (asserts compiled out)\n");
        return 1;
    }
#else
    RUN(test_stray_bits_abort_in_debug);
#endif
    if (!require_debug) {
        RUN(test_against_reference);
        RUN(test_against_reference_warm);
        RUN(test_against_reference_released);
    }
    REPORT();
}
