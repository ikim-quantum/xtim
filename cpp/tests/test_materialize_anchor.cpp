// test_materialize_anchor.cpp — S2b MINOR-2 (departures review): the forced-projection
// materialize's named engine-bug throw is a live mechanism. a4_comb's identity plan has two record
// directions (the Born observable W = X2X3, then the Born decision d0 = Y2Y3); P(d0 = 1 | W = 1) = 0,
// so forcing that outcome pair is a ZERO-WEIGHT introducer: materialize_shot must throw
// "twirl_records: anchor (direction) record <c> forced to y=<b> has zero weight (engine bug)",
// and every positive-weight outcome pair must materialize.
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"

#include "check.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/deferral.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "../testapi/twirl_test_access.hpp"

using namespace qeccore;

static const char kA4Comb[] =
    "R 0\nRX 1 2 3\nT 2\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nCCZ 0 1 3\n"
    "MPP X2*X3\nOBSERVABLE_INCLUDE(0) rec[-1]\nMPP Y2*Y3\nDECISION(0) rec[-1]\nMX 1\nDECISION(1) rec[-1]\n"
    "R 9\nM 9\nDETECTOR rec[-1]\n";

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

void test_zero_weight_introducer_throws_by_name() {
    SamplerBundle bun = compile_bundle(kA4Comb);
    CHECK(bun.ok);
    if (!bun.ok) { std::fprintf(stderr, "%s\n", bun.error.c_str()); return; }
    auto s = TwirlRecordSampler::make(*bun.bare, bun.deferred, bun.reads, bun.detectors,
                                      bun.observables, 1, bun.opt);
    CHECK_EQ(s->setup_error(), 0);
    int ok = 0, threw = 0;
    for (int m = 0; m < 4; ++m) {
        const uint8_t uz[2] = {(uint8_t)(m & 1), (uint8_t)((m >> 1) & 1)};   // identity plan: rho 0, kappa_rec 2
        try {
            (void)s->materialize_shot(nullptr, 0, uz, 2);
            ++ok;
        } catch (const std::logic_error& e) {
            const std::string w = e.what();
            CHECK(w.find("twirl_records: anchor (direction) record") != std::string::npos);
            CHECK(w.find("has zero weight (engine bug)") != std::string::npos);
            ++threw;
        }
    }
    std::fprintf(stderr, "[materialize_anchor] positive-weight records %d, zero-weight records refused %d\n", ok, threw);
    CHECK_EQ(threw, 1);      // exactly the (W = 1, d0 = 1) pair has zero weight
    CHECK_EQ(ok, 3);
}

// S7 fold (#7): the ctor's one-time alt-blob verification failure is a NAMED refusal (setup_error 5 with the text in
// family_refusal(), which the bindings raise), not a stderr line. The test seam corrupts one blob word.
// Mutation: drop the family_refusal_ assignment -> the text checks go RED.
static int g_blob_hook_calls = 0;
void test_blob_verify_fail_is_a_named_refusal() {
    SamplerBundle bun = compile_bundle(kA4Comb);
    CHECK(bun.ok);
    if (!bun.ok) return;
    g_twirl_alt_blob_test_hook = [](std::vector<uint64_t>& blob) {
        ++g_blob_hook_calls;
        if (blob.size() > 2) blob[2] ^= 1;   // first alt's first kmask word
    };
    auto s = TwirlRecordSampler::make(*bun.bare, bun.deferred, bun.reads, bun.detectors, bun.observables, 1, bun.opt);
    g_twirl_alt_blob_test_hook = nullptr;
    CHECK_EQ(g_blob_hook_calls, 1);
    CHECK_EQ(s->setup_error(), 5);
    const std::string& w = TwirlSamplerTestAccess::family_refusal(*s);
    CHECK(w.find("twirl_records: BLOB VERIFY FAIL") != std::string::npos);
    CHECK(w.find("(engine bug)") != std::string::npos);
}

int main() {
    RUN(test_zero_weight_introducer_throws_by_name);
    RUN(test_blob_verify_fail_is_a_named_refusal);
    REPORT();
}
