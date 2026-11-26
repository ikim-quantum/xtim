// test_rotation_verify.cpp — lean D1-S1: the ctor's one-time verification of the compact rotation records (decode ==
// fresh recompute, bounds-checked) is a NAMED refusal (setup_error 5 with the text in family_refusal(), which the
// bindings raise), not a stderr line. The test seam g_twirl_rot_rec_test_hook corrupts one record word: a sign word
// (word 0), the token (word 2·TW), the entry count and a channel-entry index. On the same circuit an untouched
// construction serves (setup_error 0) and the records decode (family_rotation_check == (alts, 0)).
// Mutation: drop the family_refusal_ assignment / the check -> RED.
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"

#include "check.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include "qeccore/deferral.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_rotation.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "../testapi/twirl_test_access.hpp"

using namespace qeccore;

// T content (CS / T / CCZ), an ANTI extra (the Born observable), several alternatives with S_f != 0
static const char kPairCcz[] =
    "RX 0\nRX 1\nRX 2\nRX 3\nH 1\nX_ERROR(0.3) 3\nDEPOLARIZE1(0.3) 0\nCS 2 0\nX_ERROR(0.3) 2\nZ_ERROR(0.3) 1\nT 3\n"
    "X_ERROR(0.3) 1\nZ_ERROR(0.3) 0\nS 1\nX_ERROR(0.3) 0\nZ_ERROR(0.3) 3\nS 0\nS 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nMY 1\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n";

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

static int g_hook_calls = 0;

void test_rotation_records_verify_fail_is_a_named_refusal() {
    SamplerBundle bun = compile_bundle(kPairCcz);
    CHECK(bun.ok);
    if (!bun.ok) { std::fprintf(stderr, "compile: %s\n", bun.error.c_str()); return; }
    {   // untouched: served, the model built, every record decodes
        auto s = TwirlRecordSampler::make(*bun.bare, bun.deferred, bun.reads, bun.detectors, bun.observables, 1, bun.opt);
        CHECK_EQ(s->setup_error(), 0);
        const RotationModel& M = TwirlSamplerTestAccess::family_rotation_model(*s);
        CHECK(M.built);
        CHECK(!M.kind2);
        CHECK(M.t > 0);
        CHECK_EQ(M.nx, 1);
        const RotationRecords& R = TwirlSamplerTestAccess::family_rotation_records(*s);
        CHECK(R.num_alts() > 4);
        const auto chk = TwirlSamplerTestAccess::family_rotation_check(*s);
        CHECK_EQ(chk.first, (long)R.num_alts());
        CHECK_EQ(chk.second, 0L);
    }
    // the corruption targets: alt 0's first sign word (word 0) and its token (word 2·TW); the entry count and the
    // first channel-entry index of the first alternative whose frame touches a channel (nz >= 1)
    const auto s0 = TwirlRecordSampler::make(*bun.bare, bun.deferred, bun.reads, bun.detectors, bun.observables, 1, bun.opt);
    const RotationRecords& R0 = TwirlSamplerTestAccess::family_rotation_records(*s0);
    const size_t hdr = (size_t)2 * R0.TW + 2 + (size_t)2 * R0.XR + 1;
    int a1 = -1;
    for (int a = 0; a < R0.num_alts() && a1 < 0; ++a) if (R0.rec[(size_t)R0.off[(size_t)a] + hdr - 1] >= 1) a1 = a;
    CHECK(a1 >= 0);
    if (a1 < 0) return;
    const size_t o1 = R0.off[(size_t)a1];
    const std::pair<size_t, int> targets[4] = {{0, 0}, {(size_t)2 * R0.TW, 0}, {o1 + hdr - 1, a1}, {o1 + hdr, a1}};
    for (const auto& tg : targets) {
        static size_t s_word;
        s_word = tg.first;
        g_twirl_rot_rec_test_hook = [](std::vector<uint32_t>& rec, int chunk) {
            if (chunk != 0) return;                  // every target lives in the first chunk of alternatives
            ++g_hook_calls;
            if (rec.size() > s_word) rec[s_word] ^= 1;
        };
        auto s = TwirlRecordSampler::make(*bun.bare, bun.deferred, bun.reads, bun.detectors, bun.observables, 1, bun.opt);
        g_twirl_rot_rec_test_hook = nullptr;
        CHECK_EQ(s->setup_error(), 5);
        const std::string& w = TwirlSamplerTestAccess::family_refusal(*s);
        CHECK(w.find("twirl_records: ROT RECORD VERIFY FAIL at alt " + std::to_string(tg.second) + ":") != std::string::npos);
        CHECK(w.find("(engine bug)") != std::string::npos);
    }
    CHECK_EQ(g_hook_calls, 4);
}

int main() {
    RUN(test_rotation_records_verify_fail_is_a_named_refusal);
    REPORT();
}
