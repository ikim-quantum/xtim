// test_dem_compose_cz — M2 (magic lane): the DEM export's sparse per-alternative composition (AltScratch, CZ layers
// included) against the dense chain it replaces (DiagPauliClifford::then over dense_c_prop, and alt_signature's full
// read scan), kept in the export as the oracle behind set_dem_compose_mode_for_test.
//   (1) mode 2 (check): every sparse composition is also composed by the dense chain and compared field for field
//       (n, gamma, a, v, B, the residual_normal_form cache, supp a superset of the support) — zero mismatches, and
//       non-vacuous: some compared alternatives carry a CZ layer.
//   (2) the export under mode 2 (production's composition) vs mode 1 (the dense chain): every DemExportResult
//       field byte for byte.
// Corpus: every *.stim with a non-Clifford gate under tests/data, xtim/_examples, examples, benchmarks, plus random
// magic circuits (T / CS / CCZ mirrors with noise, mid-circuit reads, rec feedback, PAULI_EXPECTATION columns);
// four export variants each. argv[1] = the repo root (set by CMake).
// Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/dem_export.hpp"

using namespace qeccore;

static std::string g_repo;
static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One export variant; a throw is folded into the result (its message as `error`) so both modes compare it.
static DemExportResult variant(const std::string& text, int v) {
    try {
        switch (v) {
            case 0: return export_dem(text, /*include_expectations=*/true, false, false, false, nullptr, nullptr, 0.0);
            case 1: return export_dem(text, /*include_expectations=*/false, false, false, false, nullptr, nullptr, 0.1);
            case 2:
                return export_dem_route(text, true, false, false, false, nullptr, nullptr, 0.0,
                                        kDemCliffordCutoffDefault, nullptr, false, DemFormat::Xdem, false,
                                        XdemRoute::Oracle, nullptr);
            default:
                return export_dem(text, true, false, false, false, nullptr, nullptr, 0.0, kDemCliffordCutoffDefault,
                                  nullptr, /*erasure=*/true);
        }
    } catch (const std::exception& e) {
        DemExportResult r;
        r.error = std::string("threw: ") + e.what();
        return r;
    }
}

static bool same_result(const DemExportResult& a, const DemExportResult& b) {
    bool pe_eq = a.parse_errors.size() == b.parse_errors.size();
    for (std::size_t i = 0; pe_eq && i < a.parse_errors.size(); ++i)
        pe_eq = a.parse_errors[i].line == b.parse_errors[i].line &&
                a.parse_errors[i].message == b.parse_errors[i].message;
    return a.ok == b.ok && a.dem == b.dem && a.error == b.error && a.rejected == b.rejected &&
           a.reject_gate_index == b.reject_gate_index && pe_eq && a.dropped_observables == b.dropped_observables &&
           a.reject_detectors == b.reject_detectors && a.postselect_faults == b.postselect_faults &&
           a.dropped_expectation_columns == b.dropped_expectation_columns && a.clifford_faults == b.clifford_faults &&
           a.clifford_dropped_mass == b.clifford_dropped_mass &&
           a.nondeterministic_targets == b.nondeterministic_targets;
}

static bool has_magic(const std::string& t) {
    for (const char* g : {"T ", "T_DAG", "CCZ", "CS ", "CS_DAG", "CH "})
        if (t.find(std::string("\n") + g) != std::string::npos || t.rfind(g, 0) == 0) return true;
    return false;
}

// Random magic circuits: C · noise · C^dag mirrors over T / CS / CCZ and Cliffords on 3..5 qubits (an X fault
// before a CS / CCZ leaves a CZ layer), optional mid-circuit reads with rec feedback, PAULI_EXPECTATION columns.
static std::string fuzz(std::mt19937_64& rng) {
    auto ri = [&](int lo, int hi) { return (int)std::uniform_int_distribution<int>(lo, hi)(rng); };
    auto rf = [&]() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };
    auto pick = [&](const std::vector<std::string>& v) { return v[(size_t)ri(0, (int)v.size() - 1)]; };
    const int n = ri(3, 5);
    const bool mid = rf() < 0.4, xb = rf() < 0.5;
    auto inv = [](const std::string& g) {
        static const std::vector<std::pair<std::string, std::string>> m = {
            {"T", "T_DAG"}, {"T_DAG", "T"}, {"CS", "CS_DAG"}, {"CS_DAG", "CS"}, {"S", "S_DAG"}, {"S_DAG", "S"}};
        for (const auto& [a, b] : m) if (a == g) return b;
        return g;
    };
    std::string out, all;
    for (int i = 0; i < n; ++i) all += (i ? " " : "") + std::to_string(i);
    auto noise = [&]() {
        if (rf() < 0.6) {
            if (rf() < 0.2)
                out += "DEPOLARIZE2(0.01) " + std::to_string(ri(0, 1)) + " " + std::to_string(ri(2, n - 1)) + "\n";
            else
                out += pick({"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}) + "(0.01) " +
                       std::to_string(ri(0, n - 1)) + "\n";
        }
    };
    out += (xb ? "RX " : "R ") + all + "\n";
    const int segs = mid ? ri(1, 3) : 1;
    for (int s = 0; s < segs; ++s) {
        std::vector<std::pair<std::string, std::string>> b;
        const int len = ri(3, 9);
        for (int i = 0; i < len; ++i) {
            std::vector<int> q(n);
            for (int k = 0; k < n; ++k) q[(size_t)k] = k;
            std::shuffle(q.begin(), q.end(), rng);
            const std::string q0 = std::to_string(q[0]), q01 = q0 + " " + std::to_string(q[1]);
            const double r = rf();
            if (r < 0.2) b.push_back({pick({"T", "T_DAG"}), q0});
            else if (r < 0.4) b.push_back({pick({"CS", "CS_DAG"}), q01});
            else if (r < 0.55) b.push_back({"CCZ", q01 + " " + std::to_string(q[2])});
            else if (r < 0.75) b.push_back({pick({"H", "S", "S_DAG"}), q0});
            else b.push_back({pick({"CX", "CZ"}), q01});
        }
        for (const auto& [g, t] : b) { out += g + " " + t + "\n"; noise(); }
        for (size_t i = b.size(); i-- > 0;) { out += inv(b[i].first) + " " + b[i].second + "\n"; noise(); }
        if (mid) {
            out += (xb ? "MX " : "M ") + std::to_string(ri(0, n - 1)) + "\nDETECTOR rec[-1]\n";
            if (rf() < 0.3) out += "CX rec[-1] " + std::to_string(ri(0, n - 1)) + "\n";
        }
    }
    if (rf() < 0.3)
        out += std::string("PAULI_EXPECTATION(0) ") + (xb ? "X" : "Z") + std::to_string(ri(0, n - 1)) + "\n";
    out += (xb ? "MX " : "M ") + all + "\n";
    for (int k = 1; k <= n; ++k) out += "DETECTOR rec[-" + std::to_string(k) + "]\n";
    out += "OBSERVABLE_INCLUDE(0) rec[-1]\n";
    return out;
}

static std::vector<std::string> corpus() {
    namespace fs = std::filesystem;
    const fs::path repo(g_repo);
    std::vector<std::string> texts;
    for (const char* d : {"tests/data", "xtim/_examples", "examples", "benchmarks"}) {
        if (!fs::exists(repo / d)) continue;
        std::vector<fs::path> ps;
        for (const auto& e : fs::recursive_directory_iterator(repo / d))
            if (e.is_regular_file() && e.path().extension() == ".stim") ps.push_back(e.path());
        std::sort(ps.begin(), ps.end());
        for (const fs::path& p : ps) {
            std::string t = slurp(p.string());
            if (has_magic(t)) texts.push_back(std::move(t));
        }
    }
    std::mt19937_64 rng(20261121);
    for (int i = 0; i < 400; ++i) texts.push_back(fuzz(rng));
    return texts;
}

static void test_sparse_composition_equals_dense_chain() {
    const std::vector<std::string> texts = corpus();
    long long compared = 0, mismatched = 0, with_cz = 0;
    int diff = 0, ok = 0;
    for (size_t i = 0; i < texts.size(); ++i) {
        for (int v = 0; v < 4; ++v) {
            set_dem_compose_mode_for_test(2);
            const DemExportResult prod = variant(texts[i], v);
            const DemComposeCheckCounts c = dem_compose_check_counts_for_test();
            set_dem_compose_mode_for_test(1);
            const DemExportResult dense = variant(texts[i], v);
            set_dem_compose_mode_for_test(0);
            compared += c.compared;
            mismatched += c.mismatched;
            with_cz += c.with_cz;
            ok += prod.ok;
            if (c.mismatched) std::fprintf(stderr, "  corpus item %zu variant %d: %lld composition mismatches\n", i, v,
                                           c.mismatched);
            if (!same_result(prod, dense)) {
                std::fprintf(stderr, "  corpus item %zu variant %d: sparse and dense-chain exports differ\n", i, v);
                ++diff;
            }
        }
    }
    std::fprintf(stderr, "  %zu circuits, %d ok exports, %lld compositions compared (%lld with a CZ layer)\n",
                 texts.size(), ok, compared, with_cz);
    CHECK_EQ(mismatched, 0LL);
    CHECK_EQ(diff, 0);
    CHECK(with_cz > 1000);           // non-vacuous: the CZ-layer path is exercised
    CHECK(compared > with_cz);       // and the no-CZ sparse path too
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_dem_compose_cz <repo root>\n"); return 2; }
    g_repo = argv[1];
    RUN(test_sparse_composition_equals_dense_chain);
    REPORT();
}
