// test_xdem_lean — LEAN-EXPORT-PLAN. Task 0: the lean and oracle routes return identical results on every
// _examples circuit. argv[1] = xtim/_examples, argv[2] = tests/data/xdem (set by CMake).
// Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/dem_export.hpp"
#include "qeccore/dem_preamble.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_lean.hpp"
#include "qeccore/xdem_writer.hpp"
#include "../testapi/xdem_lean_dense_oracle.hpp"
#include "../testapi/xdem_sweep_dense_oracle.hpp"

using namespace qeccore;

static std::string g_examples, g_xdem;
static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static DemExportResult xdem_route(const std::string& text, XdemRoute route, bool* used = nullptr,
                                  std::string* lean_exception = nullptr) {
    return export_dem_route(text, true, false, false, false, nullptr, nullptr, 0.0, kDemCliffordCutoffDefault,
                            nullptr, false, DemFormat::Xdem, false, route, used, lean_exception);
}
static void same_result(const DemExportResult& a, const DemExportResult& b, const std::string& what) {
    // final review M2: every DemExportResult field
    bool pe_eq = a.parse_errors.size() == b.parse_errors.size();
    for (std::size_t i = 0; pe_eq && i < a.parse_errors.size(); ++i)
        pe_eq = a.parse_errors[i].line == b.parse_errors[i].line &&
                a.parse_errors[i].message == b.parse_errors[i].message;
    const bool eq = a.ok == b.ok && a.dem == b.dem && a.error == b.error && a.rejected == b.rejected &&
                    a.reject_gate_index == b.reject_gate_index && pe_eq &&
                    a.dropped_observables == b.dropped_observables && a.reject_detectors == b.reject_detectors &&
                    a.postselect_faults == b.postselect_faults &&
                    a.dropped_expectation_columns == b.dropped_expectation_columns &&
                    a.clifford_faults == b.clifford_faults && a.clifford_dropped_mass == b.clifford_dropped_mass &&
                    a.nondeterministic_targets == b.nondeterministic_targets;
    if (!eq) std::fprintf(stderr, "  routes differ on %s\n", what.c_str());
    CHECK(eq);
}
static const std::vector<std::string> kExamples = {
    "cultivation_d3_faithful", "cultivation_d5", "color5_cultivation", "miniature_oracle", "distillation_15_1_3",
    "cube_ccz", "code_switching_faithful", "code_switching_rate", "cultivation_d3_rate", "ch_cultivation"};

static void test_routes_identical_on_examples() {
    int served = 0;
    for (const std::string& n : kExamples) {
        const std::string text = slurp(g_examples + "/" + n + ".stim");
        CHECK(!text.empty());
        bool used = false;
        std::string lexc = "unset";
        const DemExportResult lean = xdem_route(text, XdemRoute::Lean, &used, &lexc);
        // Task 1 re-review: the lean attempt did not throw (a swallowed throw would delegate and still compare equal),
        // the same check as the Python sweep's lean_exception == "".
        if (!lexc.empty()) std::fprintf(stderr, "  lean attempt threw on %s: %s\n", n.c_str(), lexc.c_str());
        CHECK(lexc.empty());
        if (used) ++served;
        same_result(xdem_route(text, XdemRoute::Oracle), lean, n);
    }
    CHECK(served >= 1);                         // non-vacuous: some example is served by the lean route
}

static void test_gate_rotation_counts() {
    CHECK_EQ(xdem::gate_rotation_count("T", 1), 1);
    CHECK_EQ(xdem::gate_rotation_count("T_DAG", 1), -1);
    CHECK_EQ(xdem::gate_rotation_count("CS", 1), 1);
    CHECK_EQ(xdem::gate_rotation_count("CS", 2), 1);
    CHECK_EQ(xdem::gate_rotation_count("CS", 3), -1);
    CHECK_EQ(xdem::gate_rotation_count("CS_DAG", 3), 1);
    for (uint32_t v = 1; v < 8; ++v)
        CHECK_EQ(xdem::gate_rotation_count("CCZ", v), (__builtin_popcount(v) & 1) ? 1 : -1);
}

// The exact engine's verdicts (dem_export.cpp check_deterministic) and the lean ones on one circuit.
static void verdicts(const std::string& text, std::vector<int>& lean, std::vector<int>& oracle, bool& usable) {
    DemPreamble pre;
    DemExportResult res;
    usable = false;
    if (!dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false)) return;
    xdem::WriterModel wm;
    try { wm = xdem::build_writer_model(pre); } catch (const xdem::WriterError&) { return; }
    if (!wm.diag_violations.empty() || wm.header.groups.empty()) return;
    std::vector<std::vector<int>> col_recs(wm.sk->targets.begin(), wm.sk->targets.end());
    const xdem::FaultColumns fc = xdem::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs);
    lean = xdem::nondeterministic_columns(wm, fc);
    if (!dem_build_bare(pre, res)) return;
    oracle.clear();
    for (int t = 0; t < (int)wm.sk->targets.size(); ++t) {
        Pauli P(pre.dn.deferred.n);
        int ny = 0;
        for (int j : wm.sk->targets[(size_t)t]) {
            const auto& [b, q] = pre.reads[(size_t)j];
            if (b == 0) P.setx(q); else if (b == 1) { P.setx(q); P.setz(q); ++ny; } else P.setz(q);
        }
        if (wm.sk->targets[(size_t)t].empty()) continue;
        P.phase = ny & 3;
        auto [pp, pm] = framed_expectation(pre.bare.state, P);
        if (!(std::max(pp, pm) > 1.0 - 1e-9)) oracle.push_back(t);
    }
    usable = true;
}

static void test_determinism_hand_circuits() {
    struct Case { const char* text; std::vector<int> nondet; };
    const std::vector<Case> cases = {
        {"RX 0\nX_ERROR(0.01) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n", {0}},                  // <X> = 1/sqrt2
        {"RX 0\nX_ERROR(0.01) 0\nT 0\nT 0\nMX 0\nDETECTOR rec[-1]\n", {0}},             // S|+>: <X> = 0
        {"RX 0\nX_ERROR(0.01) 0\nT 0\nT_DAG 0\nMX 0\nDETECTOR rec[-1]\n", {}},          // identity
        {"RX 0\nX_ERROR(0.01) 0\nT 0\nT 0\nT 0\nT 0\nMX 0\nDETECTOR rec[-1]\n", {}},    // Z|+>: <X> = -1
        // CCZ|+++>: <X_0> = E (-1)^{z1 z2} = 1/2. Every linear coefficient is 0 mod 4; only the pair coefficient
        // beta_12 (the odd axis Z0Z1Z2, |kappa| = 3) is nonzero: the case the beta update exists for.
        {"RX 0 1 2\nX_ERROR(0.01) 0\nCCZ 0 1 2\nMX 0\nDETECTOR rec[-1]\n", {0}},
    };
    for (const Case& c : cases) {
        std::vector<int> lean, oracle;
        bool usable = false;
        verdicts(c.text, lean, oracle, usable);
        CHECK(usable);
        CHECK(lean == c.nondet);
        CHECK(oracle == c.nondet);
    }
}

static void test_determinism_matches_bare_state_on_examples() {
    for (const std::string& n : kExamples) {
        std::vector<int> lean, oracle;
        bool usable = false;
        verdicts(slurp(g_examples + "/" + n + ".stim"), lean, oracle, usable);
        if (!usable) continue;
        if (lean != oracle) std::fprintf(stderr, "  verdicts differ on %s\n", n.c_str());
        CHECK(lean == oracle);
    }
}

// OD2 (Task 0 review): a lean attempt that THROWS is a non-success too; the oracle answers, byte for byte.
static void test_lean_throw_delegates_to_oracle() {
    std::vector<std::string> texts = {"RX 0\nX_ERROR(0.01) 0\nT 0\nT_DAG 0\nMX 0\nDETECTOR rec[-1]\n",
                                      "RX 0\nX_ERROR(0.01) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n"};
    for (const std::string& n : kExamples) texts.push_back(slurp(g_examples + "/" + n + ".stim"));
    int reached = 0;
    for (const std::string& text : texts) {
        const DemExportResult o = xdem_route(text, XdemRoute::Oracle);
        bool used_plain = false;
        (void)xdem_route(text, XdemRoute::Lean, &used_plain);
        xdem::set_lean_inject_throw_for_test(true);
        bool used = true;
        DemExportResult l;
        bool threw = false;
        std::string lexc;
        try {
            l = export_dem_route(text, true, false, false, false, nullptr, nullptr, 0.0, kDemCliffordCutoffDefault,
                                 nullptr, false, DemFormat::Xdem, false, XdemRoute::Lean, &used, &lexc);
        } catch (...) { threw = true; }
        xdem::set_lean_inject_throw_for_test(false);
        CHECK(!threw);
        CHECK(!used);
        same_result(o, l, "injected lean throw");
        if (used_plain) CHECK(lexc.find("injected (test)") != std::string::npos);   // I2: the throw is visible
        if (used_plain) ++reached;               // the injection point is on this circuit's lean path
    }
    CHECK(reached >= 2);
}

// nondet-message: gauss_z4 against the brute-force sum E_u i^{Phi(u)} on random forms (m <= 7).
static void test_gauss_z4_brute_force() {
    std::mt19937_64 rng(20261118);
    int zeros = 0, odd_h = 0;
    for (int it = 0; it < 3000; ++it) {
        const int m = 1 + (int)(rng() % 7);
        std::vector<uint8_t> alpha((size_t)m);
        std::vector<std::vector<uint64_t>> adj((size_t)m, std::vector<uint64_t>(1, 0));
        for (auto& a : alpha) a = (uint8_t)(rng() & 3);
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j)
                if (rng() & 1) { adj[(size_t)i][0] |= 1ull << j; adj[(size_t)j][0] |= 1ull << i; }
        double re = 0, im = 0;
        for (uint32_t u = 0; u < (1u << m); ++u) {
            int phi = 0;
            for (int i = 0; i < m; ++i)
                if ((u >> i) & 1u) {
                    phi += alpha[(size_t)i];
                    for (int j = i + 1; j < m; ++j) if (((u >> j) & 1u) && ((adj[(size_t)i][0] >> j) & 1ull)) phi += 2;
                }
            phi &= 3;
            re += phi == 0 ? 1 : phi == 2 ? -1 : 0;
            im += phi == 1 ? 1 : phi == 3 ? -1 : 0;
        }
        re /= (double)(1u << m); im /= (double)(1u << m);
        const xdem::GaussZ4 g = xdem::gauss_z4(alpha, adj);
        double gr = 0, gi = 0;
        if (!g.zero) {
            const double mag = std::pow(2.0, -g.h / 2.0);
            gr = mag * std::cos(M_PI * g.p8 / 4.0);
            gi = mag * std::sin(M_PI * g.p8 / 4.0);
        } else {
            ++zeros;
        }
        if (g.h & 1) ++odd_h;
        if (std::abs(gr - re) > 1e-12 || std::abs(gi - im) > 1e-12)
            std::fprintf(stderr, "  gauss_z4 differs at it %d (m %d): %g%+gi vs %g%+gi\n", it, m, gr, gi, re, im);
        CHECK(std::abs(gr - re) <= 1e-12 && std::abs(gi - im) <= 1e-12);
    }
    CHECK(zeros >= 100 && odd_h >= 100);                // both branches exercised
}

// nondet-message: column_expectation equals the bare state's <P> (framed_expectation, pp - pm) on every column.
static int expectations_agree(const std::string& text) {
    DemPreamble pre;
    DemExportResult res;
    if (!dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false, /*class_table=*/false)) return 0;
    xdem::WriterModel wm;
    try { wm = xdem::build_writer_model(pre); } catch (const xdem::WriterError&) { return 0; }
    if (!wm.diag_violations.empty() || wm.header.groups.empty() || wm.h_meets_location) return 0;
    if (!dem_build_bare(pre, res)) return 0;
    int n = 0;
    for (int t = 0; t < wm.header.D + wm.header.O; ++t) {
        if (wm.sk->targets[(size_t)t].empty()) continue;
        Pauli P(pre.dn.deferred.n);
        int ny = 0;
        for (int j : wm.sk->targets[(size_t)t]) {
            const auto& [b, q] = pre.reads[(size_t)j];
            if (b == 0) P.setx(q); else if (b == 1) { P.setx(q); P.setz(q); ++ny; } else P.setz(q);
        }
        P.phase = ny & 3;
        auto [pp, pm] = framed_expectation(pre.bare.state, P);
        const double lean = xdem::column_expectation(wm, t);
        if (std::abs(lean - (pp - pm)) > 1e-9) std::fprintf(stderr, "  <P_%d>: lean %.9f, bare %.9f\n", t, lean, pp - pm);
        CHECK(std::abs(lean - (pp - pm)) <= 1e-9);
        ++n;
    }
    return n;
}

static void test_column_expectation_matches_bare_state() {
    const std::vector<std::string> hand = {
        "RX 0\nX_ERROR(0.01) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n",                 // 1/sqrt2
        // deferred as T; MX with an inverted record (T_DAG = S_DAG T, and S_DAG folds into the read): the deferred
        // <X> is +1/sqrt2; the record parity (the refusal text) is <Y> = -1/sqrt2
        "RX 0\nX_ERROR(0.01) 0\nT_DAG 0\nMY 0\nDETECTOR rec[-1]\n",
        "RX 0\nX_ERROR(0.01) 0\nT 0\nT 0\nMY 0\nDETECTOR rec[-1]\n",            // S|+>: <Y> = 1
        "RX 0\nX_ERROR(0.01) 0\nT 0\nT 0\nT 0\nT 0\nMX 0\nDETECTOR rec[-1]\n",  // -1
        "RX 0 1 2\nX_ERROR(0.01) 0\nCCZ 0 1 2\nMX 0\nDETECTOR rec[-1]\n",       // 1/2
        "RX 0 1\nX_ERROR(0.01) 0\nCS 0 1\nMX 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-1] rec[-2]\n",
        "RX 0 1 2\nCX 0 1\nT 1\nX_ERROR(0.01) 2\nCCZ 2 1 0\nMX 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-3] rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n",
    };
    int n = 0;
    for (const std::string& t : hand) n += expectations_agree(t);
    for (const std::string& e : kExamples) n += expectations_agree(slurp(g_examples + "/" + e + ".stim"));
    CHECK(n >= 50);
}

// nondet-message: a non-deterministic circuit gets the oracle's determinism refusal from the lean route itself
// (lean_used, no delegation), field for field, with every non-deterministic target listed.
static void test_lean_determinism_refusal_equals_oracle() {
    const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
        {"RX 0\nX_ERROR(0.01) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n", {"D0"}},
        {"RX 0 1\nX_ERROR(0.01) 0 1\nT 0 1\nMX 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\n"
         "OBSERVABLE_INCLUDE(1) rec[-1]\n", {"D0", "D1", "L1"}},
        {"RX 0 1 2\nX_ERROR(0.01) 0\nCCZ 0 1 2\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n", {"L0"}},
    };
    for (const auto& [text, names] : cases) {
        bool used = false;
        std::string lexc = "unset";
        const DemExportResult o = xdem_route(text, XdemRoute::Oracle);
        const DemExportResult l = xdem_route(text, XdemRoute::Lean, &used, &lexc);
        CHECK(used);
        CHECK(lexc.empty());
        CHECK(!l.ok && l.error.rfind("non-deterministic ", 0) == 0);
        CHECK(l.nondeterministic_targets == names);
        same_result(o, l, "determinism refusal: " + text);
    }
}

// LEAN-EXPORT-PLAN Task 2 (F1/F2): the H-condition on the examples, and the ex422_obs counterexample.
// Task 2 review M3: the lean route's preamble configuration (class_table = false), with a floor on the examples checked.
static void test_h_condition_on_examples() {
    int checked = 0, checked_ch = 0;
    for (const std::string& n : kExamples) {
        DemPreamble pre;
        DemExportResult res;
        if (!dem_preamble(slurp(g_examples + "/" + n + ".stim"), false, nullptr, pre, res, /*build_bare=*/false,
                          /*class_table=*/false))
            continue;
        xdem::WriterModel wm;
        try { wm = xdem::build_writer_model(pre); } catch (const xdem::WriterError&) { continue; }
        if (!wm.diag_violations.empty() || wm.header.groups.empty()) continue;
        bool has_ch = false;
        for (const Instr& ins : pre.p.circuit.stream) has_ch = has_ch || (ins.kind == Instr::Kind::Gate &&
                                                                          ins.gate == GateKind::CH);
        if (has_ch) {                          // a CH's rewrite puts an H on its first rotation's wire: delegated
            CHECK(wm.h_meets_location);
            ++checked_ch;
            continue;
        }
        CHECK(!wm.h_meets_location);           // plan time: false on every CH-free example the writer accepts
        CHECK(build_propagation_table(pre.dn.deferred).all_in_class);
        ++checked;
    }
    CHECK(checked >= 6);                        // Task 2 review: 8 of the 10 examples are writer-OK with h false
    CHECK(checked_ch >= 1);                     // ch_cultivation (CH in the xdem export)
    DemPreamble pre;                            // ex422_obs: the H-condition holds (F1 counterexample)
    DemExportResult res;
    CHECK(dem_preamble(slurp(g_xdem + "/circuits/ex422_obs.stim"), false, nullptr, pre, res, /*build_bare=*/false,
                       /*class_table=*/false));
    CHECK(xdem::build_writer_model(pre).h_meets_location);
    // Task 3: so the lean route delegates it (no table on the lean route), with the oracle's bytes.
    const std::string text = slurp(g_xdem + "/circuits/ex422_obs.stim");
    bool used = true;
    std::string lexc = "unset";
    const DemExportResult lean = xdem_route(text, XdemRoute::Lean, &used, &lexc);
    CHECK(!used);
    CHECK(lexc.empty());
    same_result(xdem_route(text, XdemRoute::Oracle), lean, "ex422_obs");
}

// LEAN-EXPORT-PLAN Task 4: LeanLineAccumulator == LineAccumulator, bitwise (p by memcmp), on random keys and times.
// wide = false: the brief's instance family (A <= 6, single-bit images, some 0). wide = true (amendment: bit-word
// visibility and the zero-image prune): A up to 140 (three bit words), two-word images with several bits, a third 0.
static void lean_merge_vs_oracle(uint64_t seed, int instances, bool wide) {
    std::mt19937_64 rng(seed);
    auto U = [&](int n) { return (int)(rng() % (uint64_t)n); };
    int with_close = 0;                                                // instances where a visible swap closed a line
    int coarse = 0;                         // Task 4 review N2: instances where the zero-image prune merges oracle classes
    for (int inst = 0; inst < instances; ++inst) {
        const int A = wide ? 1 + U(140) : 1 + U(6), D = wide ? 90 : 3, O = 1;
        const size_t W = wide ? 2 : 1;
        std::vector<xdem::Axis> axes((size_t)A);
        for (auto& ax : axes) {
            ax.img.w.assign(W, 0);
            if (!wide) { if (U(3)) ax.img.w[0] = 1ull << U(D + O); continue; }
            if (U(3) == 0) continue;                                  // a zero image
            for (int b = 1 + U(3); b > 0; --b) { const int j = (U(2) ? 64 : 0) + U(3); ax.img.w[(size_t)j >> 6] ^= 1ull << (j & 63); }
        }
        const int pa = wide ? 12 : 3;                                  // an axis in a field with probability 1 / pa
        std::vector<xdem::LineKey> keys(1 + (size_t)U(wide ? 12 : 6));
        for (auto& k : keys) {
            k.side = U(4) == 0;
            for (int j = 0; j < D + O; ++j) if (U(wide ? 30 : 3) == 0) k.f.push_back(j);
            for (int a = 0; a < A; ++a) if (U(pa) == 0) k.tau.push_back(a);
            for (int a = 0; a < A; ++a) if (U(pa) == 0) k.lt.push_back({a, (uint8_t)(U(2) ? 1 : 3)});
            for (int a = 0; a < A; ++a) if (U(pa + 1) == 0) k.gt.push_back({a, (uint8_t)(U(2) ? 1 : 3)});
        }
        xdem::LineAccumulator ref;
        xdem::LeanLineAccumulator lean;
        ref.set_axes(axes);
        lean.set_axes(axes);
        const int M = 1 + U(wide ? 60 : 25);
        std::vector<char> used(keys.size(), 0);
        for (int i = 0; i < M; ++i) {
            const size_t ki = (size_t)U((int)keys.size());
            const xdem::LineKey& k = keys[ki];
            const double p = U(10) == 0 ? 0.0 : std::ldexp((double)(1 + U(1000)), -14);
            if (p > 0.0 && !k.empty()) used[ki] = 1;
            xdem::TimeKey t{k.side ? 10 + U(3) : U(3), U(4), i};   // ties in step / ordinal; mech breaks them
            ref.add(k, p, t);
            lean.add(xdem::LineKey(k), p, t);
        }
        CHECK(ref.any_rotation() == lean.any_rotation());
        xdem::Canon C;
        bool threw_ref = false, threw_lean = false;
        xdem::File a, b;
        try { a = ref.to_file(D, O, C, axes); } catch (const xdem::WriterError&) { threw_ref = true; }
        try { b = lean.to_file(D, O, C, axes); } catch (const xdem::WriterError&) { threw_lean = true; }
        CHECK(threw_ref == threw_lean);
        if (threw_ref) continue;
        CHECK(a.p.size() == b.p.size());
        CHECK(a.p.size() == 0 || std::memcmp(a.p.data(), b.p.data(), a.p.size() * sizeof(double)) == 0);
        CHECK(a.flip_off == b.flip_off && a.flip_cols == b.flip_cols && a.tau_off == b.tau_off &&
              a.tau_ax == b.tau_ax && a.rot_off == b.rot_off && a.late_off == b.late_off && a.tc == b.tc);
        CHECK(a.rot.size() == b.rot.size() && a.late.size() == b.late.size());
        for (size_t i = 0; i < a.rot.size() && i < b.rot.size(); ++i)
            CHECK(a.rot[i].axis == b.rot[i].axis && a.rot[i].e == b.rot[i].e);
        for (size_t i = 0; i < a.late.size() && i < b.late.size(); ++i)
            CHECK(a.late[i].axis == b.late[i].axis && a.late[i].e == b.late[i].e);
        CHECK(ref.num_lines() == lean.num_lines());
        if (ref.num_lines() > (size_t)std::count(used.begin(), used.end(), 1)) ++with_close;
        {   // oracle classes (tau, odd) vs lean classes (tau & Z, odd & Z) over the keys that made a line
            using Cls = std::pair<std::vector<int32_t>, std::vector<int32_t>>;
            std::set<Cls> oc, lc;
            auto inZ = [&](const std::vector<int32_t>& v) {
                std::vector<int32_t> o;
                for (int32_t x : v) {
                    bool nz = false;
                    for (uint64_t w : axes[(size_t)x].img.w) nz |= w != 0;
                    if (nz) o.push_back(x);
                }
                return o;
            };
            for (size_t ki = 0; ki < keys.size(); ++ki) {
                if (!used[ki]) continue;
                const std::vector<int32_t> odd = xdem::line_odd_axes(keys[ki]);
                oc.insert({keys[ki].tau, odd});
                lc.insert({inZ(keys[ki].tau), inZ(odd)});
            }
            if (lc.size() < oc.size()) ++coarse;     // two oracle classes share one lean class; bytes compared above
        }
    }
    std::fprintf(stderr, "  %s: %d of %d instances close a line, %d merge oracle classes in one lean class\n",
                 wide ? "wide" : "narrow", with_close, instances, coarse);
    CHECK(with_close >= instances / 10);                               // the closing rule is exercised
    CHECK(coarse >= 1);                                                // the zero-image prune is exercised (N2)
}

static void test_lean_merge_equals_oracle_random() { lean_merge_vs_oracle(20261117, 2000, false); }
static void test_lean_merge_equals_oracle_random_wide() { lean_merge_vs_oracle(20261118, 2000, true); }

// The E1 case of test_xdem_writer.cpp (w r w r: four lines; with img = 0: two), on the lean merge.
static void test_lean_merge_e1_cycle() {
    xdem::LineKey w;  w.tau = {0};
    xdem::LineKey r;  r.tau = {0}; r.lt = {{0, 3}};
    std::vector<xdem::Axis> vis(2), inv(2);
    vis[0].img.w = {0b01}; vis[1].img.w = {0b10};
    inv[0].img.w = {0}; inv[1].img.w = {0b10};
    auto cycle = [&](const std::vector<xdem::Axis>& ax) {
        xdem::LeanLineAccumulator c;
        c.set_axes(ax);
        c.add(xdem::LineKey(w), 0.2, {0, 0, 0}); c.add(xdem::LineKey(r), 0.1, {1, 1, 0});
        c.add(xdem::LineKey(w), 0.15, {2, 2, 0}); c.add(xdem::LineKey(r), 0.12, {3, 3, 0});
        return c.num_lines();
    };
    CHECK_EQ((int)cycle(vis), 4);
    CHECK_EQ((int)cycle(inv), 2);
}

// LEAN-EXPORT-PLAN Task 5: emit_fast == emit, byte for byte, on the fixtures and on the oracle writer's Files.
static void test_emit_fast_equals_emit() {
    const std::vector<std::string> fixtures = {
        "cultivation_d3_faithful.xdem", "cultivation_d5.xdem", "distillation_15_1_3_gatenoise.xdem",
        "color5_cultivation.xdem", "miniature_oracle.xdem", "code_switching_faithful_noobs.xdem",
        "code_switching_rate_noobs.xdem", "tdag_cycle.xdem", "ex422_obs.xdem", "synth/boundary_r64.xdem",
        "synth/boundary_r65.xdem", "synth/lossy.xdem"};
    for (const std::string& f : fixtures) {
        const xdem::File F = xdem::parse(slurp(g_xdem + "/" + f));
        CHECK(xdem::emit_fast(F) == xdem::emit(F));
    }
    int writer_files = 0;
    for (const std::string& n : kExamples) {                 // writer Files with sign -, hidden, late fields
        const DemExportResult r = xdem_route(slurp(g_examples + "/" + n + ".stim"), XdemRoute::Oracle);
        if (!r.ok || r.dem.rfind("xdem 1", 0) != 0) continue;
        const xdem::File F = xdem::parse(r.dem);
        CHECK(xdem::emit_fast(F) == r.dem);
        ++writer_files;
    }
    CHECK(writer_files >= 4);                                // non-vacuous: 4 of the examples export xdem
}

// LEAN-EXPORT-PLAN Task 6 (amendment): the lean route's sparse line key (LeanKeyBuilder) == the dense xdem_key on
// EVERY mechanism the lean route keys, compared field for field inside the export (set_lean_key_check_for_test):
// the in-repo circuits (every *.stim under tests/data, xtim/_examples, examples, benchmarks: the Python sweep's
// FILES), each at approximate_disjoint_errors 0 and 0.1, plus random circuits (a port of tests/_xdem_lean_fuzz.py's
// mirror / mid families: C C^dag with noise between, mid-circuit reads and rec feedback; plus a late family, the
// same with t_c first and the final reads as observables, so every gate is late: the E^> field and its folds).
static std::string fuzz_mirror(std::mt19937_64& rng, bool mid, bool late = false) {
    auto ri = [&](int lo, int hi) { return (int)std::uniform_int_distribution<int>(lo, hi)(rng); };
    auto rf = [&]() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };
    auto pick = [&](const std::vector<std::string>& v) { return v[(size_t)ri(0, (int)v.size() - 1)]; };
    const int n = mid ? ri(3, 4) : ri(3, 5);
    auto three = [&]() {
        std::vector<int> q(n);
        for (int i = 0; i < n; ++i) q[(size_t)i] = i;
        std::shuffle(q.begin(), q.end(), rng);
        return q;
    };
    auto inv = [](const std::string& g) {
        static const std::vector<std::pair<std::string, std::string>> m = {
            {"T", "T_DAG"}, {"T_DAG", "T"}, {"CS", "CS_DAG"}, {"CS_DAG", "CS"}, {"S", "S_DAG"}, {"S_DAG", "S"}};
        for (const auto& [a, b] : m) if (a == g) return b;
        return g;
    };
    std::string out;
    auto noise = [&]() {
        if (rf() < (mid ? 0.7 : 0.6))
            out += pick({"X_ERROR", "Z_ERROR", "DEPOLARIZE1"}) + "(0.01) " + std::to_string(ri(0, n - 1)) + "\n";
    };
    const bool xb = rf() < 0.5;
    std::string all;
    for (int i = 0; i < n; ++i) all += (i ? " " : "") + std::to_string(i);
    if (xb || mid) out += (xb ? "RX " : "R ") + all + "\n";
    if (late) out += "RX " + std::to_string(n) + "\nMX " + std::to_string(n) + "\nDETECTOR rec[-1]\n";   // t_c first
    const int segs = mid ? ri(1, 3) : 1;
    for (int s = 0; s < segs; ++s) {
        std::vector<std::pair<std::string, std::string>> b;   // (gate, targets)
        const int len = mid ? ri(2, 7) : ri(3, 10);
        for (int i = 0; i < len; ++i) {
            const double r = rf();
            const std::vector<int> q = three();
            const std::string q0 = std::to_string(q[0]), q01 = q0 + " " + std::to_string(q[1]);
            if (r < 0.3) b.push_back({pick({"T", "T_DAG"}), q0});
            else if (r < 0.4) b.push_back({pick({"CS", "CS_DAG"}), q01});
            else if (r < 0.45) b.push_back({"CCZ", q01 + " " + std::to_string(q[2])});
            else if (r < 0.67) b.push_back({pick({"H", "S", "S_DAG"}), q0});
            else b.push_back({pick({"CX", "CZ"}), q01});
        }
        for (const auto& [g, t] : b) { out += g + " " + t + "\n"; noise(); }
        for (size_t i = b.size(); i-- > 0;) { out += inv(b[i].first) + " " + b[i].second + "\n"; noise(); }
        if (mid) {
            out += (xb ? "MX " : "M ") + std::to_string(ri(0, n - 1)) + "\nDETECTOR rec[-1]\n";
            if (rf() < 0.3) out += "CX rec[-1] " + std::to_string(ri(0, n - 1)) + "\n";
        }
    }
    out += (xb ? "MX " : "M ") + all + "\n";
    if (late) {                                // every gate after t_c: the E^> field (and its folds)
        for (int k = 1; k <= n; ++k)
            out += "OBSERVABLE_INCLUDE(" + std::to_string(k - 1) + ") rec[-" + std::to_string(k) + "]\n";
        return out;
    }
    for (int k = 1; k <= n; ++k) out += "DETECTOR rec[-" + std::to_string(k) + "]\n";
    out += "OBSERVABLE_INCLUDE(0) rec[-1]\n";
    return out;
}

static void key_check_leg(const std::vector<std::string>& texts, double approx, long long& compared, int& served) {
    for (const std::string& text : texts) {
        bool used = false;
        std::string lexc;
        xdem::set_lean_key_check_for_test(true);
        (void)export_dem_route(text, true, false, false, false, nullptr, nullptr, approx, kDemCliffordCutoffDefault,
                               nullptr, false, DemFormat::Xdem, false, XdemRoute::Lean, &used, &lexc);
        const auto [c, m] = xdem::lean_key_check_counts();
        xdem::set_lean_key_check_for_test(false);
        if (m) std::fprintf(stderr, "  sparse key != dense key on %lld of %lld mechanisms\n", m, c);
        CHECK_EQ(m, 0LL);
        CHECK(lexc.empty());
        compared += c;
        served += used;
    }
}

static void test_sparse_key_equals_dense_key() {
    namespace fs = std::filesystem;
    const fs::path repo = fs::path(g_xdem) / ".." / ".." / "..";
    std::vector<std::string> inrepo;
    for (const char* d : {"tests/data", "xtim/_examples", "examples", "benchmarks"}) {
        if (!fs::exists(repo / d)) continue;
        std::vector<fs::path> ps;
        for (const auto& e : fs::recursive_directory_iterator(repo / d))
            if (e.is_regular_file() && e.path().extension() == ".stim") ps.push_back(e.path());
        std::sort(ps.begin(), ps.end());
        for (const fs::path& p : ps) inrepo.push_back(slurp(p.string()));
    }
    std::mt19937_64 rng(20261117);
    std::vector<std::string> fuzz = {   // hand: two late T_DAG on one axis after one fault, E^> = 2 (a late fold)
        "RX 0\nRX 1\nMX 1\nDETECTOR rec[-1]\nX_ERROR(0.01) 0\nT 0\nX_ERROR(0.01) 0\nT 0\nX_ERROR(0.01) 0\n"
        "T_DAG 0\nX_ERROR(0.01) 0\nT_DAG 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        // Task 6 review I1: one X fault, then two early T and two late T_DAG (or T) on one axis, and the CS analogue:
        // E^< = E^> = 2 on the same axis, the two folds cancel (xdem_lean.cpp, LeanKeyBuilder::build)
        "RX 0\nRX 1\nX_ERROR(0.01) 0\nT 0\nT 0\nMX 1\nDETECTOR rec[-1]\nT_DAG 0\nT_DAG 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "RX 0\nRX 1\nX_ERROR(0.01) 0\nT 0\nT 0\nMX 1\nDETECTOR rec[-1]\nT 0\nT 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "RX 0 1\nRX 2\nX_ERROR(0.01) 0\nCS 0 1\nCS 0 1\nMX 2\nDETECTOR rec[-1]\nCS_DAG 0 1\nCS_DAG 0 1\nMX 0 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2]\n"};
    {   // review I1: the both-fields-2 hand circuits are lean-keyed (non-vacuous)
        long long c = 0;
        int sv = 0;
        key_check_leg(std::vector<std::string>(fuzz.end() - 3, fuzz.end()), 0.0, c, sv);
        CHECK(c >= 3);
        CHECK_EQ(sv, 3);
    }
    for (int i = 0; i < 900; ++i) fuzz.push_back(fuzz_mirror(rng, i % 3 == 1, i % 3 == 2));   // mirror / mid / late
    long long c_in = 0, c_ap = 0, c_fz = 0;
    int s_in = 0, s_ap = 0, s_fz = 0;
    key_check_leg(inrepo, 0.0, c_in, s_in);
    key_check_leg(inrepo, 0.1, c_ap, s_ap);
    key_check_leg(fuzz, 0.0, c_fz, s_fz);
    std::fprintf(stderr, "  key check: in-repo %zu circuits, %d lean-served, %lld keys; approx %d, %lld; "
                 "fuzz %zu, %d, %lld\n", inrepo.size(), s_in, c_in, s_ap, c_ap, fuzz.size(), s_fz, c_fz);
    CHECK(inrepo.size() >= 70);                // non-vacuous floors (measured at Task 6: 81 circuits; 27 lean-served,
    CHECK(s_in >= 25);                         // 74542 keys; approx 43, 74955; fuzz 901 circuits, 663, 5481)
    CHECK(c_in >= 50000);
    CHECK(s_ap >= 40);
    CHECK(c_ap >= 50000);
    CHECK(s_fz >= 500);
    CHECK(c_fz >= 4500);
}

// LM-2: the sparse nondeterministic_columns == the dense oracle (verbatim 9fb442f), list and order, on the in-repo
// circuits, the mirror / mid / late fuzz, word-boundary chains with a noisy T (63..129 columns) and a wide-support
// family that the per-chunk bound splits into several chunks.
static int sparse_vs_dense(const std::string& text, xdem::NondetStats* out = nullptr) {
    DemPreamble pre;
    DemExportResult res;
    if (!dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false, /*class_table=*/false)) return 0;
    xdem::WriterModel wm;
    try { wm = xdem::build_writer_model(pre); } catch (const xdem::WriterError&) { return 0; }
    if (!wm.diag_violations.empty() || wm.header.groups.empty()) return 0;
    std::vector<std::vector<int>> col_recs(wm.sk->targets.begin(), wm.sk->targets.end());
    xdem::FaultColumns fc;
    try { fc = xdem::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs); } catch (const xdem::WriterError&) { return 0; }
    xdem::NondetStats st;
    const std::vector<int> s = xdem::nondeterministic_columns(wm, fc, &st);
    // LM-3: the oracle reads the dense x_start (the production sweep may end on its sparse lists)
    const xdem::FaultColumns fd = xdem::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs, xdem::SweepLayout::Dense);
    const std::vector<int> d = xdem::test_oracle::nondeterministic_columns_dense(wm, fd);
    if (s != d) std::fprintf(stderr, "  sparse != dense verdicts (%zu vs %zu columns)\n", s.size(), d.size());
    CHECK(s == d);
    if (out) *out = st;
    return 1;
}

static void test_sparse_nondet_equals_dense() {
    namespace fs = std::filesystem;
    const fs::path repo = fs::path(g_xdem) / ".." / ".." / "..";
    int n_in = 0, n_fz = 0, n_bd = 0;
    for (const char* d : {"tests/data", "xtim/_examples", "examples", "benchmarks"}) {
        if (!fs::exists(repo / d)) continue;
        std::vector<fs::path> ps;
        for (const auto& e : fs::recursive_directory_iterator(repo / d))
            if (e.is_regular_file() && e.path().extension() == ".stim") ps.push_back(e.path());
        std::sort(ps.begin(), ps.end());
        for (const fs::path& p : ps) n_in += sparse_vs_dense(slurp(p.string()));
    }
    std::mt19937_64 rng(20261123);
    for (int i = 0; i < 600; ++i) n_fz += sparse_vs_dense(fuzz_mirror(rng, i % 3 == 1, i % 3 == 2));
    for (int n : {63, 64, 65, 127, 128, 129}) {      // chain + noisy T: n + 1 columns around the word boundaries
        std::string t = "R " + std::to_string(n) + " " + std::to_string(n + 1) + "\nH " + std::to_string(n) +
                        "\nX_ERROR(0.001) " + std::to_string(n) + "\nT " + std::to_string(n) + "\nCX " +
                        std::to_string(n) + " " + std::to_string(n + 1) + "\n";
        for (int q = 0; q < n; ++q)
            t += "R " + std::to_string(q) + "\nX_ERROR(0.01) " + std::to_string(q) + "\nM " + std::to_string(q) +
                 "\nDETECTOR rec[-1]\n";
        t += "MX " + std::to_string(n) + "\nM " + std::to_string(n + 1) + "\nOBSERVABLE_INCLUDE(0) rec[-2]\n";
        n_bd += sparse_vs_dense(t);
    }
    int multi = 0;
    for (int n : {129, 200, 300}) {                  // wide: dense start supports, several chunks
        std::mt19937_64 wr((uint64_t)n);
        std::string allq;
        for (int q = 0; q < n; ++q) allq += " " + std::to_string(q);
        std::string t = "RX" + allq + "\nX_ERROR(0.01) 0\nT" + allq + "\n";   // n axes, every wire a pivot
        for (int k = 0; k < 6 * n; ++k) {
            const int a = (int)(wr() % (uint64_t)n);
            int b = (int)(wr() % (uint64_t)(n - 1));
            if (b >= a) ++b;
            t += "CX " + std::to_string(a) + " " + std::to_string(b) + "\n";
        }
        t += "M" + allq + "\n";                      // Z reads: dense X supports at the start
        for (int k = 1; k <= n; ++k) t += "DETECTOR rec[-" + std::to_string(k) + "]\n";
        xdem::NondetStats st;
        CHECK_EQ(sparse_vs_dense(t, &st), 1);
        multi += st.chunks >= 2;
    }
    std::fprintf(stderr, "  sparse vs dense: in-repo %d, fuzz %d, boundary %d, multi-chunk %d\n", n_in, n_fz, n_bd, multi);
    CHECK(n_in >= 40);
    CHECK(n_fz >= 300);
    CHECK_EQ(n_bd, 6);
    CHECK_EQ(multi, 3);
}

// LM-3: the production backward sweep under each storage layout (auto / sparse / dense) == the dense oracle
// (verbatim 2e89d05): the fault columns' noise slots and x ids, x_start (dense words, or the sparse per-column wire
// lists against the oracle's bits), the tau columns, and the lean determinism verdicts on those columns. Returns 1
// when the writer model is built (and sets *final_auto / *dense_from_auto to the auto fault sweep's history).
static int sweep_layouts_vs_oracle(const std::string& text, int* final_auto = nullptr, long long* dense_from_auto = nullptr) {
    DemPreamble pre;
    DemExportResult res;
    if (!dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false, /*class_table=*/false)) return 0;
    xdem::WriterModel wm;
    try { wm = xdem::build_writer_model(pre); } catch (const xdem::WriterError&) { return 0; }
    std::vector<std::vector<int>> col_recs(wm.sk->targets.begin(), wm.sk->targets.end());
    const bool admitted = wm.diag_violations.empty() && !wm.header.groups.empty();
    xdem::test_oracle::FaultColumnsDense of;
    xdem::test_oracle::SweepAtDense ot;
    try {
        of = xdem::test_oracle::build_fault_columns_dense(*wm.sk, pre.dn.deferred, col_recs);
        ot = xdem::test_oracle::build_tau_columns_dense(*wm.sk, pre.dn.deferred);
    } catch (const xdem::WriterError&) { return 0; }
    std::vector<int> nd_oracle;
    if (admitted) {
        xdem::FaultColumns ofc;
        ofc.ncols = of.ncols; ofc.xid = of.xid; ofc.at = of.at; ofc.x_start = of.x_start; ofc.cw = of.cw;
        nd_oracle = xdem::test_oracle::nondeterministic_columns_dense(wm, ofc);
    }
    for (xdem::SweepLayout L : {xdem::SweepLayout::Auto, xdem::SweepLayout::Sparse, xdem::SweepLayout::Dense}) {
        xdem::SweepStats fs, ts;
        const xdem::FaultColumns fc = xdem::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs, L, &fs);
        const xdem::TauColumns tc = xdem::build_tau_columns(*wm.sk, pre.dn.deferred, L, &ts);
        CHECK(fc.at == of.at);
        CHECK(fc.xid == of.xid);
        CHECK(tc.at == ot);
        if (fc.x_sparse()) {
            CHECK(fs.final_layout == 1);
            CHECK(fc.x_start.empty());
            CHECK_EQ(fc.xs_off.size(), (size_t)of.ncols + 1);
            bool same = fc.xs_off.back() == fc.xs_wire.size();
            size_t e = 0;
            for (int c = 0; same && c < of.ncols; ++c) {
                same = fc.xs_off[(size_t)c] == e;
                for (int w = 0; same && w < wm.sk->N; ++w)
                    if ((of.x_start[(size_t)w * of.cw + ((size_t)c >> 6)] >> (c & 63)) & 1ull)
                        same = e < fc.xs_wire.size() && fc.xs_wire[e++] == w;
            }
            CHECK(same && e == fc.xs_wire.size());
        } else {                                     // the planes, or (fix round 1) bit rows with dense X supports
            CHECK(fc.x_start == of.x_start);
            CHECK_EQ(fc.cw, of.cw);
        }
        if (L == xdem::SweepLayout::Sparse) CHECK(fs.final_layout == 1 && ts.final_layout == 1 && fs.dense_from == -1);
        if (L == xdem::SweepLayout::Dense) CHECK(fs.final_layout == 2 && ts.final_layout == 2 && fs.dense_from == 0);
        if (L == xdem::SweepLayout::Auto) {
            if (final_auto) *final_auto = fs.final_layout;
            if (dense_from_auto) *dense_from_auto = fs.dense_from;
        }
        if (admitted) CHECK(xdem::nondeterministic_columns(wm, fc) == nd_oracle);
    }
    return 1;
}

static void test_sparse_sweep_equals_dense() {
    namespace fs = std::filesystem;
    const fs::path repo = fs::path(g_xdem) / ".." / ".." / "..";
    int n_in = 0, n_fz = 0, n_bd = 0, n_sparse = 0, n_mid = 0;
    for (const char* d : {"tests/data", "xtim/_examples", "examples", "benchmarks"}) {
        if (!fs::exists(repo / d)) continue;
        std::vector<fs::path> ps;
        for (const auto& e : fs::recursive_directory_iterator(repo / d))
            if (e.is_regular_file() && e.path().extension() == ".stim") ps.push_back(e.path());
        std::sort(ps.begin(), ps.end());
        for (const fs::path& p : ps) n_in += sweep_layouts_vs_oracle(slurp(p.string()));
    }
    std::mt19937_64 rng(20261124);
    for (int i = 0; i < 400; ++i) n_fz += sweep_layouts_vs_oracle(fuzz_mirror(rng, i % 3 == 1, i % 3 == 2));
    for (int n : {63, 64, 65, 127, 128, 129, 3000}) {   // chain + noisy T around the word boundaries; 3000: auto sparse
        std::string t = "R " + std::to_string(n) + " " + std::to_string(n + 1) + "\nH " + std::to_string(n) +
                        "\nX_ERROR(0.001) " + std::to_string(n) + "\nT " + std::to_string(n) + "\nCX " +
                        std::to_string(n) + " " + std::to_string(n + 1) + "\n";
        for (int q = 0; q < n; ++q)
            t += "R " + std::to_string(q) + "\nX_ERROR(0.01) " + std::to_string(q) + "\nM " + std::to_string(q) +
                 "\nDETECTOR rec[-1]\n";
        t += "MX " + std::to_string(n) + "\nM " + std::to_string(n + 1) + "\nOBSERVABLE_INCLUDE(0) rec[-2]\n";
        int fin = 0;
        n_bd += sweep_layouts_vs_oracle(t, &fin);
        n_sparse += n == 3000 && fin == 1;
    }
    for (int n : {1600}) {                               // wide: dense supports fill the lists mid-sweep
        std::mt19937_64 wr((uint64_t)n);
        std::string allq;
        for (int q = 0; q < n; ++q) allq += " " + std::to_string(q);
        std::string t = "RX" + allq + "\nX_ERROR(0.01) 0\nT" + allq + "\n";
        for (int k = 0; k < 6 * n; ++k) {
            const int a = (int)(wr() % (uint64_t)n);
            int b = (int)(wr() % (uint64_t)(n - 1));
            if (b >= a) ++b;
            t += "CX " + std::to_string(a) + " " + std::to_string(b) + "\n";
        }
        t += "M" + allq + "\n";
        for (int k = 1; k <= n; ++k) t += "DETECTOR rec[-" + std::to_string(k) + "]\n";
        int fin = 0;
        long long from = -1;
        CHECK_EQ(sweep_layouts_vs_oracle(t, &fin, &from), 1);
        n_mid += fin == 2 && from > 0;
    }
    std::fprintf(stderr, "  sweep layouts vs oracle: in-repo %d, fuzz %d, boundary %d, auto sparse %d, mid-switch %d\n",
                 n_in, n_fz, n_bd, n_sparse, n_mid);
    CHECK(n_in >= 60);
    CHECK(n_fz >= 300);
    CHECK_EQ(n_bd, 7);
    CHECK_EQ(n_sparse, 1);
    CHECK_EQ(n_mid, 1);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_xdem_lean <xtim/_examples> <tests/data/xdem>\n"); return 2; }
    g_examples = argv[1];
    g_xdem = argv[2];
    RUN(test_routes_identical_on_examples);
    RUN(test_gate_rotation_counts);
    RUN(test_determinism_hand_circuits);
    RUN(test_determinism_matches_bare_state_on_examples);
    RUN(test_lean_throw_delegates_to_oracle);
    RUN(test_gauss_z4_brute_force);
    RUN(test_column_expectation_matches_bare_state);
    RUN(test_lean_determinism_refusal_equals_oracle);
    RUN(test_h_condition_on_examples);
    RUN(test_lean_merge_equals_oracle_random);
    RUN(test_lean_merge_equals_oracle_random_wide);
    RUN(test_lean_merge_e1_cycle);
    RUN(test_emit_fast_equals_emit);
    RUN(test_sparse_key_equals_dense_key);
    RUN(test_sparse_nondet_equals_dense);
    RUN(test_sparse_sweep_equals_dense);
    REPORT();
}
