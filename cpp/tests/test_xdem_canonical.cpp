// test_xdem_canonical — the xdem writer's canonical-form check (xdem_writer.hpp canonical_violation, docs/xdem.md
// "Canonical form") and the parser's opt-in source provenance it reads (ParsedStim::stream_line / stream_step).
// Moved from the dc-DEM v1 route's test (Task 11); the export leg now runs format = Xdem.
//  * Provenance is metadata only: the stream with track_source equals the default parse, instruction by instruction
//    (kind, gate, targets, qubits, probs); every instruction of one user line (REPEAT iteration) shares a step.
//  * C1' / C2' verdicts on hand circuits: canonical, observable with no record after / only in the same instruction
//    as the last detector record, frame records, noise and a noisy measurement before the last non-Clifford gate
//    (inside the dressing) vs after it, zero-probability noise (not noise), no detector, REPEAT.
//  * export_dem (format = Xdem, an extended output): require_canonical=false gives the same bytes on a
//    non-canonical circuit; true refuses by name; a canonical circuit gives the same bytes either way; the option is
//    refused with the Stim format.
// argv[1] = tests/data/xdem/circuits/ex422_obs.stim (set by CMake). Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <fstream>
#include <sstream>
#include <string>

#include "qeccore/dem_export.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/xdem_writer.hpp"

using namespace qeccore;

static std::string why(const std::string& text) {
    const ParsedStim p = parse_stim_circuit(text, true);
    CHECK(p.ok());
    return xdem::canonical_violation(p);
}
static bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

static void test_provenance_is_metadata() {
    const std::string t = "R 0 1 2\nH 0\nREPEAT 2 {\nCX 0 1\nDEPOLARIZE1(0.01) 0 1\nM 1\nDETECTOR rec[-1]\n}\n"
                          "MPP X0*X2 Z1*Z2\nMX(0.01) 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    const ParsedStim a = parse_stim_circuit(t), b = parse_stim_circuit(t, true);
    CHECK(a.stream_line.empty() && a.stream_step.empty());
    CHECK_EQ(a.circuit.stream.size(), b.circuit.stream.size());
    CHECK_EQ(b.stream_line.size(), b.circuit.stream.size());
    CHECK_EQ(b.stream_step.size(), b.circuit.stream.size());
    for (size_t i = 0; i < a.circuit.stream.size() && i < b.circuit.stream.size(); ++i) {
        const Instr &x = a.circuit.stream[i], &y = b.circuit.stream[i];
        CHECK(x.kind == y.kind && x.gate == y.gate && x.targets == y.targets && x.qubits == y.qubits &&
              x.probs == y.probs && x.readout_flip_p == y.readout_flip_p);
    }
    CHECK(a.detectors == b.detectors && a.observables == b.observables);
    for (size_t i = 1; i < b.stream_step.size(); ++i) {
        CHECK(b.stream_step[i] >= b.stream_step[i - 1]);
        if (b.stream_step[i] == b.stream_step[i - 1]) CHECK_EQ(b.stream_line[i], b.stream_line[i - 1]);
    }
    // the two REPEAT iterations' M 1 (line 6): same line, different steps
    std::vector<size_t> m6;
    for (size_t i = 0; i < b.circuit.stream.size(); ++i)
        if (b.circuit.stream[i].kind == Instr::Kind::Measure && b.stream_line[i] == 6) m6.push_back(i);
    CHECK_EQ(m6.size(), (size_t)2);
    if (m6.size() == 2) CHECK(b.stream_step[m6[0]] != b.stream_step[m6[1]]);
    const ParsedStim bad = parse_stim_circuit("H 0\nNOPE 1\n", true);
    CHECK(!bad.ok() && bad.stream_line.empty());
    bool threw = false;
    try { xdem::canonical_violation(a); } catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
}

static void test_verdicts() {
    // canonical: noisy protocol, detectors, then a noiseless dressed readout
    CHECK(why("R 0 1\nX_ERROR(0.1) 0 1\nM 1\nDETECTOR rec[-1]\nT 0\nMX 0\nT_DAG 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
              .empty());
    // C1 before; the reason names the observable, the record, the lines and the last detector
    std::string w = why("R 0 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 1\nDETECTOR rec[-1]\n");
    CHECK(has(w, "observable L0 reads no record measured after the last detector record (record #1 of D0, line 4): "
                 "its latest, record #0 (line 2), is measured before it"));
    // C1': earlier records are a classical frame; no record at all (cancelled) violates it
    CHECK(why("R 0 1 2\nM 0\nM 1\nDETECTOR rec[-1]\nM 2\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3]\n").empty());
    CHECK(has(why("R 0 1\nM 1\nDETECTOR rec[-1]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-1]\n"),
              "observable L0 reads no record (none measured after"));
    // C1 same instruction (M and MPP with several products)
    CHECK(has(why("R 0 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"), "by the same instruction"));
    CHECK(has(why("R 0 1 2\nMPP Z0*Z1 Z1*Z2\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"),
              "by the same instruction"));
    CHECK(why("R 0 1 2\nMPP Z0*Z1\nMPP Z1*Z2\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    // C1: a record listed twice is not read
    CHECK(why("R 0 1\nM 0\nM 1\nDETECTOR rec[-1]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3] rec[-3]\n").empty());
    // C2': noise / a noisy measurement before the last non-Clifford gate; after it (a readout flip) it is allowed
    w = why("R 0 1\nM 1\nDETECTOR rec[-1]\nDEPOLARIZE1(0.01) 0\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    CHECK(has(w, "noise DEPOLARIZE1 at line 4 acts between the last detector record (record #0 of D0, line 2) and the "
                 "last non-Clifford gate (T, line 5)"));
    CHECK(why("R 0 1\nM 1\nDETECTOR rec[-1]\nDEPOLARIZE1(0.01) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    w = why("R 0 1 2\nM 1\nDETECTOR rec[-1]\nMX(0.01) 0\nT 2\nM 2\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    CHECK(has(w, "noisy measurement at line 4 (record #1, readout flip p=0.01) acts between"));
    CHECK(why("R 0 1\nM 1\nDETECTOR rec[-1]\nMX(0.01) 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    // noise inside the dressing T; ..; T_DAG is refused (T_DAG named); readout noise after the dressing is allowed
    w = why("R 0 1\nM 1\nDETECTOR rec[-1]\nT 0\nZ_ERROR(0.1) 0\nT_DAG 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    CHECK(has(w, "noise Z_ERROR at line 5 acts between") && has(w, "(T_DAG, line 6)"));
    CHECK(why("R 0 1\nM 1\nDETECTOR rec[-1]\nT 0\nT_DAG 0\nZ_ERROR(0.1) 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
              .empty());
    CHECK(why("R 0 1\nM 1\nDETECTOR rec[-1]\nZ_ERROR(0) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    // the measurement noise of the t_c instruction itself is before t_c
    CHECK(why("R 0 1\nM(0.01) 1\nDETECTOR rec[-1]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    // no detector: t_c = start
    CHECK(why("R 0\nH 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    CHECK(has(why("R 0\nX_ERROR(0.1) 0\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"), "the start of the circuit"));
    CHECK(why("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    // REPEAT: detectors inside, observable after; noise inside a later REPEAT named by its line
    const std::string rep = "R 0 1\nREPEAT 3 {\nX_ERROR(0.1) 1\nMR 1\nDETECTOR rec[-1]\n}\n";
    CHECK(why(rep + "M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
    CHECK(has(why(rep + "REPEAT 2 {\nZ_ERROR(0.1) 0\n}\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"),
              "noise Z_ERROR at line 8 acts between the last detector record (record #2 of D2, line 4) and the last "
              "non-Clifford gate (T, line 10) (and 1 more"));
    // feedback after t_c is allowed (not noise)
    CHECK(why("R 0 1\nM 1\nDETECTOR rec[-1]\nCX rec[-1] 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").empty());
}

static DemExportResult xdem_export(const std::string& text, bool require_canonical) {
    return export_dem(text, true, false, false, false, nullptr, nullptr, 0.0, kDemCliffordCutoffDefault, nullptr,
                      false, DemFormat::Xdem, require_canonical);
}

// The SPEC §3.7 worked example (canonical, extended): tests/data/xdem/circuits/ex422_obs.stim (argv[1]).
static std::string g_ex422;

static void test_export_option() {
    // C2' violated: noise between t_c and the last non-Clifford gate; the fault between the two T gates rotates the
    // X readout (an extended output)
    const std::string nc = "RX 0\nR 1\nM 1\nDETECTOR rec[-1]\nT 0\nX_ERROR(0.1) 0\nT_DAG 0\nMX 0\n"
                           "OBSERVABLE_INCLUDE(0) rec[-1]\n";
    const DemExportResult a = export_dem(nc, true, false, false, false, nullptr, nullptr, 0.0,
                                         kDemCliffordCutoffDefault, nullptr, false, DemFormat::Xdem);
    const DemExportResult b = xdem_export(nc, false);
    CHECK(a.ok && b.ok && a.dem == b.dem);
    CHECK(a.dem.rfind("xdem 1", 0) == 0);
    const DemExportResult c = xdem_export(nc, true);
    CHECK(!c.ok && has(c.error, "require_canonical=True") && has(c.error, "noise X_ERROR at line 6 acts between"));
    // canonical and extended: the worked example gives the same bytes either way
    const DemExportResult d = xdem_export(g_ex422, true), e = xdem_export(g_ex422, false);
    CHECK(d.ok && e.ok && d.dem == e.dem);
    CHECK(d.dem.rfind("xdem 1", 0) == 0);
    const DemExportResult f = export_dem(nc, true, false, false, false, nullptr, nullptr, 0.0,
                                         kDemCliffordCutoffDefault, nullptr, false, DemFormat::Stim, true);
    CHECK(!f.ok && has(f.error, "format=\"xdem\" option"));
}

int main(int argc, char** argv) {
    {
        std::ifstream f(argc > 1 ? argv[1] : "tests/data/xdem/circuits/ex422_obs.stim");
        std::stringstream ss;
        ss << f.rdbuf();
        g_ex422 = ss.str();
    }
    CHECK(!g_ex422.empty());
    RUN(test_provenance_is_metadata);
    RUN(test_verdicts);
    RUN(test_export_option);
    REPORT();
}
