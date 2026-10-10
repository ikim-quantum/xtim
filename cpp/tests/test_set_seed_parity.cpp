// test_set_seed_parity — speed-kill T2: ctor(seed=s) streams == ctor(1)+set_seed(s) streams.
//
// Pins seed-VALUE-dependent parity that the Python suite structurally cannot: the Python
// oracle (tests/test_set_seed_equivalence.py) always constructs with kDefaultSeed=1, then
// calls set_seed for the probe seeds. This test exercises the complementary direction at
// the C++ API level: ctor(seed=s) built from scratch at s vs ctor(seed=1) that receives
// set_seed(s) after warmup runs — for s ∈ {2, 7, 5_000_000_007} (the last value exceeds
// 2^32 = 4_294_967_296, pinning the full 64-bit seed path).
//
// Stream comparison: channel_one_counts() after run(shots). This is a deterministic
// aggregate that is a sufficient witness for identical shot streams: any divergence in the
// per-shot coin/noise draws (from seed_, smp, obs_gen, fb_gen, dec_gen, done_rf_) produces
// at least one channel bit difference across the shot population with overwhelming probability.
//
// Must end main() with REPORT() — without it, CHECK() failures are counted but not returned
// as a nonzero exit code and the test passes silently.  See check.hpp.
//
// Build: the cpp/CMakeLists.txt wires this test via qeccore_test.  Run from a configured
// build directory: cmake --build . && ctest -R seed_parity
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"

#include "check.hpp"

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/deferral.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_sampler.hpp"

using namespace qeccore;

// ── Minimal stim circuit with DETECTORs + noise ───────────────────────────────────────────────
// 3-qubit repetition code, one syndrome round, p=0.01.  Has 2 DETECTORs so the
// TwirlRecordSampler has something to accumulate in circuit_channels mode.  This circuit
// is known to be accepted by compile_twirl_sampler (rep-code class, χ=2).
static const char kSmallCircuit[] =
    "R 0 1 2 3 4\n"
    "X_ERROR(0.01) 0 1 2\n"
    "CX 0 3 1 3\n"
    "DEPOLARIZE2(0.01) 0 3 1 3\n"
    "CX 1 4 2 4\n"
    "DEPOLARIZE2(0.01) 1 4 2 4\n"
    "M 3 4\n"
    "DETECTOR rec[-2]\n"
    "DETECTOR rec[-1]\n"
    "X_ERROR(0.01) 0 1 2\n"
    "M 0 1 2\n"
    "OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]\n";

// ── Build a compiled sampler bundle ────────────────────────────────────────────────────────────
// Replicates the PyTwirlSampler circuit-only constructor's compile pipeline:
//   parse_stim_circuit → normalize → build_bare_state → FramedSuperposition::from_css
// Stores everything the TwirlRecordSampler constructor needs.
struct SamplerBundle {
    bool ok = false;
    std::string error;
    Circuit                                     deferred;
    std::vector<std::pair<int, int>>            reads;      // {(int)basis, qubit}
    std::vector<std::vector<int>>               detectors;
    std::vector<std::pair<int, std::vector<int>>> observables; // obs index -> abs meas idxs
    TwirlRecordOptions                          opt;
    std::unique_ptr<FramedSuperposition>        bare;
};

static SamplerBundle compile_bundle(const char* text) {
    SamplerBundle b;
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) {
        b.error = "parse failed";
        for (const auto& e : ps.errors)
            b.error += "\n  line " + std::to_string(e.line) + ": " + e.message;
        return b;
    }
    Circuit circ = ps.circuit;
    circ = coherentize_classically_controlled_feedback(circ);
    if (circuit_has_feedback(circ)) circ = strip_feedback(circ);
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Strip;
    NormalizeResult nr = normalize(circ, pol);
    b.deferred = std::move(nr.normalized);
    // Build reads: {(int)basis, qubit} — mirrors PyTwirlSampler's reads_ construction.
    for (const auto& r : nr.map.terminal_reads)
        b.reads.push_back({(int)r.second, r.first});
    b.detectors = std::move(ps.detectors);
    // Convert observables: map<int, vector<int>> → vector<pair<int, vector<int>>>
    for (const auto& kv : ps.observables)
        b.observables.push_back({kv.first, kv.second});
    // rec_flip / rec_invert from the coherent pre-strip circuit
    for (const auto& ins : nr.coherent.stream)
        if (ins.kind == Instr::Kind::Measure) {
            b.opt.rec_flip.push_back(ins.readout_flip_p);
            b.opt.rec_invert.push_back(ins.invert ? 1 : 0);
        }
    { size_t j = 0;
      for (const auto& ins : b.deferred.stream)
          if (ins.kind == Instr::Kind::Measure) {
              if (j < b.opt.rec_invert.size() && ins.invert) b.opt.rec_invert[j] ^= 1;
              ++j;
          } }
    // decisions (from ps.decisions; empty for our test circuit)
    b.opt.decisions = std::move(ps.decisions);
    b.opt.circuit_channels = true;
    b.opt.selfcheck = 0;
    b.opt.p_factor = 1.0;
    // Build bare state
    BareState bs = build_bare_state(b.deferred);
    if (bs.rejected) {
        b.error = "bare state rejected: " +
                  (bs.reject_reason.empty() ? std::string("unknown cause") : bs.reject_reason);
        return b;
    }
    b.bare = std::make_unique<FramedSuperposition>(FramedSuperposition::from_css(bs.state));
    b.ok = true;
    return b;
}

// ── Construct a TwirlRecordSampler at a given seed ─────────────────────────────────────────────
static std::unique_ptr<TwirlRecordSampler> make_sampler(
        const SamplerBundle& bun, uint64_t seed, const SharedPlanCaches& sc) {
    auto s = std::make_unique<TwirlRecordSampler>(
        *bun.bare, bun.deferred, bun.reads,
        bun.detectors, bun.observables, seed, bun.opt, sc);
    if (s->setup_error() != 0) {
        std::fprintf(stderr, "make_sampler: setup_error=%d (seed=%llu)\n",
                     s->setup_error(), (unsigned long long)seed);
        return nullptr;
    }
    return s;
}

// ── One seed-parity probe ──────────────────────────────────────────────────────────────────────
// Compare channel_one_counts of ctor(seed=s) vs ctor(seed=1).warmup(10).set_seed(s).run(shots).
static void check_seed_parity(const SamplerBundle& bun,
                               uint64_t seed, int shots, const char* label) {
    // Shared plan caches: allocated once, shared across both samplers (same circuit).
    SharedPlanCaches sc;
    sc.plans = std::make_shared<TwirlPlanCache>();
    sc.ppr   = std::make_shared<PprPlanCache>();

    // Sampler A: constructed directly at `seed`.
    auto sa = make_sampler(bun, seed, sc);
    CHECK(sa != nullptr);
    if (!sa) return;
    bool ok_a = sa->run(shots);
    CHECK(ok_a);
    if (!ok_a) return;
    const std::vector<long long> ca = sa->channel_one_counts();   // copy before B invalidates

    // Sampler B: constructed at seed=1, warmed up with 10 shots, then set_seed(seed).
    auto sb = make_sampler(bun, /*seed=*/1, sc);
    CHECK(sb != nullptr);
    if (!sb) return;
    sb->run(10);                       // dirty B with some history
    bool reseeded = sb->set_seed(seed);
    CHECK(reseeded);                   // must succeed (circuit_channels=true, setup ok)
    if (!reseeded) return;
    bool ok_b = sb->run(shots);
    CHECK(ok_b);
    if (!ok_b) return;
    const auto& cb = sb->channel_one_counts();

    // Compare channel_one_counts: must be identical.
    bool eq = (ca.size() == cb.size());
    for (size_t i = 0; eq && i < ca.size(); ++i)
        eq = (ca[i] == cb[i]);
    if (!eq) {
        std::fprintf(stderr,
            "FAIL seed_parity %s (seed=%llu shots=%d): "
            "channel_one_counts differ (|ca|=%zu |cb|=%zu) — "
            "set_seed is NOT stream-identical at the C++ level\n",
            label, (unsigned long long)seed, shots, ca.size(), cb.size());
        ++check::failures();
    }
}

static void test_seed_parity() {
    SamplerBundle bun = compile_bundle(kSmallCircuit);
    if (!bun.ok) {
        std::fprintf(stderr, "compile_bundle failed: %s\n", bun.error.c_str());
        CHECK(bun.ok);
        return;
    }
    // Three probe seeds: 2, 7, and one value >= 2^32 (5_000_000_007 > 4_294_967_296)
    check_seed_parity(bun, 2ULL,          300, "s2");
    check_seed_parity(bun, 7ULL,          300, "s7");
    check_seed_parity(bun, 5000000007ULL, 300, "s5G");
}

int main() {
    RUN(test_seed_parity);
    REPORT();
}
