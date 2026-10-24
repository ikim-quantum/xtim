// test_plan_cache_model — S2b R-m4 (null option): a TwirlRecordSampler's plan caches are its own.
//
// The plan-cache payloads depend on the bare state (frame, eps, free order, amplitudes: lam, the Born
// trie, the Born-observable weight) and on the Born observable — none of which a cache key covers. A
// T circuit and its T_DAG twin have the SAME certified group (identity token), wire count and CZ
// universe but different bare states: on 4c0c26b the C++ sharing overload accepted the T_DAG
// sampler on the T sampler's caches and served the T law (Born observable 29270 vs 170429 / 200k).
// The sharing API is deleted. This test pins:
//   (a) compile-absence: no public constructor takes an extra argument (caches or otherwise);
//   (b) two samplers of the T / T_DAG pair built in one process (A run first) give B exactly the
//       counts of a B built alone (equal tokens asserted for non-vacuity);
//   (c) two samplers of the SAME bundle: the second inherits no warm plan (its cache misses equal a
//       fresh sampler's).
// Must end main() with REPORT().
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"

#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
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

namespace {
// Converts to anything: an 8th constructor argument of ANY type would accept it.
struct AnyArg { template <class T> operator T() const; };
}  // namespace
static_assert(!std::is_constructible_v<TwirlRecordSampler, const FramedSuperposition&, const Circuit&,
                                       const std::vector<std::pair<int, int>>&,
                                       const std::vector<std::vector<int>>&,
                                       const std::vector<std::pair<int, std::vector<int>>>&, uint64_t,
                                       const TwirlRecordOptions&, AnyArg>,
              "S2b R-m4: no TwirlRecordSampler constructor may take external plan caches");
static_assert(std::is_constructible_v<TwirlRecordSampler, const FramedSuperposition&, const Circuit&,
                                      const std::vector<std::pair<int, int>>&,
                                      const std::vector<std::vector<int>>&,
                                      const std::vector<std::pair<int, std::vector<int>>>&, uint64_t,
                                      const TwirlRecordOptions&>,
              "the plain constructor exists (non-vacuity of the assertion above)");

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


static const char* kBase =
    "RX 0\n"
    "X_ERROR(1) 0\n"
    "%s 0\n"
    "S_DAG 0\n"
    "MX 0\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    "R 9\n"
    "X_ERROR(0.1) 9\n"
    "M 9\n"
    "DETECTOR rec[-1]\n";
static std::string circ(const char* g) { char b[512]; std::snprintf(b, sizeof b, kBase, g); return b; }

static std::unique_ptr<TwirlRecordSampler> make(const SamplerBundle& bun, uint64_t seed) {
    auto s = TwirlRecordSampler::make(*bun.bare, bun.deferred, bun.reads, bun.detectors,
                                      bun.observables, seed, bun.opt);
    CHECK(s && s->setup_error() == 0);
    return s;
}

static size_t misses(const TwirlRecordSampler& s) {
    const auto r = s.record_counters();
    return r.plan_misses_kappa0 + r.plan_misses_kappa_pos + r.plan_misses_fallback;
}

int main() {
    const SamplerBundle A = compile_bundle(circ("T").c_str());
    const SamplerBundle B = compile_bundle(circ("T_DAG").c_str());
    CHECK(A.ok && B.ok);
    if (!A.ok || !B.ok) { std::fprintf(stderr, "%s %s\n", A.error.c_str(), B.error.c_str()); REPORT(); }
    const long N = 200000;
    // The exact law of the observable bit (closed form): on T|+> the X fault before T is trivial
    // and MX after S_DAG reads Y, so P(1) = (1 - <Y>)/2 with <Y> = +1/sqrt2 for T and -1/sqrt2 for
    // T_DAG. A 5 sigma window on N shots.
    const double pa = (1.0 - std::sqrt(0.5)) / 2.0, pb = (1.0 + std::sqrt(0.5)) / 2.0;
    auto within = [&](long ones, double p) {
        const double sd = std::sqrt(p * (1.0 - p) / (double)N);
        return std::fabs((double)ones / (double)N - p) < 5.0 * sd;
    };
    // (b) A first, then B, in one process (the old sharing scenario), then B alone.
    long a_ones = 0, b_after = 0, b_alone = 0;
    uint64_t tok_a = 0, tok_b = 0;
    {
        auto a = make(A, 7);
        tok_a = a->group_token();
        CHECK(misses(*a) == 0);            // a new sampler starts with an empty cache of its own
        CHECK(a->run(N));
        a_ones = a->obs_ones();
        auto b = make(B, 7);
        tok_b = b->group_token();
        CHECK(misses(*b) == 0);            // ... even while another sampler is alive and warm
        CHECK(b->run(N));
        b_after = b->obs_ones();
    }
    {
        auto b = make(B, 7);
        CHECK(misses(*b) == 0);
        CHECK(b->run(N));
        b_alone = b->obs_ones();
    }
    CHECK(tok_a == tok_b);                 // non-vacuity: the token alone cannot tell A from B
    CHECK(within(a_ones, pa));             // each sampler samples ITS OWN law (exact oracle) ...
    CHECK(within(b_after, pb));
    CHECK(b_after == b_alone);             // ... and B's bytes do not depend on A having run
    std::fprintf(stderr, "[plan_cache_model] token=%016llx A=%ld (p=%.4f) B after A=%ld B alone=%ld "
                         "(p=%.4f) N=%ld\n",
                 (unsigned long long)tok_a, a_ones, pa, b_after, b_alone, pb, N);
    // (c) same bundle: the second sampler inherits no warm plan — it misses exactly as the first did.
    {
        auto first = make(A, 11);
        CHECK(first->run(5000));
        const size_t m1 = misses(*first);
        auto second = make(A, 11);
        CHECK(misses(*second) == 0);
        CHECK(second->run(5000));
        CHECK(m1 > 0 && misses(*second) == m1);
        CHECK(first->obs_ones() == second->obs_ones());
    }
    REPORT();
}
