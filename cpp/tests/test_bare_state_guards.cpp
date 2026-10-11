// test_bare_state_guards.cpp — the bare-state refusal guards (3.1.8).
//
//   1. Branch-rank cap: a circuit whose reduced magic branch rank r exceeds kMaxBareRank throws
//      BareStateCapacityError naming 2^r and the limit — never forms `1 << r` (3.1.7: UB, wrapped
//      mod 32 into a truncated χ = 2^(r mod 32) state).
//   2. Norm guard on the REAL build path: corrupt the assembled state through the test seam
//      g_bare_state_test_hook and build_bare_state must throw BareStateInvariantError; with the
//      seam off the same circuit builds and passes verify_bare_state_norm.
#include "check.hpp"

#include <complex>
#include <string>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

Circuit deferred_of(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    return normalize(ps.circuit, pol).normalized;
}

std::string all_t(int n) {
    std::string qs;
    for (int q = 0; q < n; ++q) qs += " " + std::to_string(q);
    return "RX" + qs + "\nT" + qs + "\nMX" + qs + "\n";
}

void rank_cap_refuses() {
    for (int n : {31, 33, 40}) {
        bool threw = false;
        try {
            (void)build_bare_state(deferred_of(all_t(n)));
        } catch (const BareStateCapacityError& e) {
            threw = true;
            const std::string w = e.what();
            CHECK(w.find("2^" + std::to_string(n)) != std::string::npos);
            CHECK(w.find("2^30") != std::string::npos);
        }
        CHECK(threw);
    }
}

void rank_at_limit_minus_is_fine() {
    BareState bs = build_bare_state(deferred_of(all_t(8)));
    CHECK(!bs.rejected);
    CHECK_EQ(bs.chi, 256);
    verify_bare_state_norm(bs.state, bs.chi);   // must not throw
}

void norm_guard_fires_on_build_path() {
    const std::string text = all_t(5);
    g_bare_state_test_hook = [](CanonicalStabSum& s) { s.branches[0].c *= 0.5; };
    bool threw = false;
    try {
        (void)build_bare_state(deferred_of(text));
    } catch (const BareStateInvariantError& e) {
        threw = true;
        CHECK(std::string(e.what()).find("not normalised") != std::string::npos);
    }
    g_bare_state_test_hook = nullptr;
    CHECK(threw);
    // seam off: the same circuit builds cleanly
    BareState bs = build_bare_state(deferred_of(text));
    CHECK(!bs.rejected);
    CHECK_EQ(bs.chi, 32);
}

}  // namespace

int main() {
    RUN(rank_cap_refuses);
    RUN(rank_at_limit_minus_is_fine);
    RUN(norm_guard_fires_on_build_path);
    REPORT();
}
