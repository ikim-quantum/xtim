// The per-shot analyzer's budget refusal text with the work bound LIFTED (pe_max_work = inf, the
// documented lift): the message must say "inf", never a (long) cast of infinity (undefined), and
// name the fixed branch cap as what binds. Built on a tiny T-ladder (3 T controls on one target,
// MX on the controls, column X on the target), deferred exactly as export_dem does it, with the
// leaf cap forced to 1 so the walk exceeds it on the first branch.
#include "check.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "qeccore/dem_pe_per_shot.hpp"
#include "qeccore/circuit_ir.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

void test_inf_work_message() {
    // 3 T controls on one target, MX on the controls, the column X on the target: normalized
    // with export_dem's own preset (feedback Reject, zero-probability noise transparent), the
    // column and the reads taken from the deferred stream exactly as export_dem builds them.
    const std::string text = "RX 0 1 2 3\nT 0 1 2\nCZ 0 3\nCZ 1 3\nCZ 2 3\nMX 0 1 2\n"
                             "PAULI_EXPECTATION(0) X3\n";
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    NormalizePolicy np;
    np.feedback = NormalizePolicy::Feedback::Reject;
    np.zero_noise_transparent = true;
    NormalizeResult nr = normalize(ps.circuit, np);
    CHECK(!nr.feedback_rejected);
    const Circuit& deferred = nr.normalized;
    FramedBareState b = build_bare_state_framed(deferred);
    CHECK(!b.rejected);
    std::vector<std::pair<int, int>> reads;
    Pauli col(deferred.n);
    bool have_col = false;
    for (const Instr& ins : deferred.stream) {
        if (ins.kind == Instr::Kind::Measure)
            reads.push_back({ins.basis == PauliBasis::X ? 0 : ins.basis == PauliBasis::Y ? 1 : 2,
                             ins.qubits[0]});
        if (ins.kind == Instr::Kind::Observable) {
            for (const PauliTerm& t : ins.obs) {
                if (t.p == PauliBasis::X) { col.setx(t.qubit); }
                else if (t.p == PauliBasis::Y) { col.setx(t.qubit); col.setz(t.qubit); col.phase += 1; }
                else { col.setz(t.qubit); }
            }
            col.phase &= 3;
            have_col = true;
        }
    }
    CHECK(reads.size() == 3);
    CHECK(have_col);
    const int n = b.state.n();
    auto read_pauli = [&](int j) { return single_pauli(reads[(size_t)j].first, reads[(size_t)j].second, n); };
    const double inf = std::numeric_limits<double>::infinity();
    // leaf cap 1, work unbounded: the column's walk has more than one support leaf
    PeShotAnalyzer pe(b.state, reads, /*leaf_budget=*/1, /*work_budget=*/inf);
    PeShotAnalyzer::Law L = pe.structure({}, read_pauli, col);
    CHECK(L.too_large.empty());
    CHECK(!L.zero);
    CHECK(L.d >= 1);
    CHECK(!pe.walk_h(L));
    CHECK(L.too_large.find("1 measurement branches / inf work units = pe_max_work") != std::string::npos);
    CHECK(L.too_large.find("the branch cap is fixed") != std::string::npos);
    CHECK(L.too_large.find("raise pe_max_work") == std::string::npos);
    // the same walk with the default caps completes (the ladder is tiny)
    PeShotAnalyzer pe2(b.state, reads);
    PeShotAnalyzer::Law L2 = pe2.structure({}, read_pauli, col);
    CHECK(pe2.walk_h(L2) && L2.walked && L2.too_large.empty());
    // joint law under a tiny FINITE work cap names pe_max_work with the integer count
    PeShotAnalyzer pe3(b.state, reads, PeShotAnalyzer::kLeafBudget, 1.0);
    std::vector<Pauli> B{read_pauli(0), read_pauli(1)};
    std::vector<PeShotAnalyzer::JointLeaf> out;
    std::string why;
    CHECK(!pe3.joint_law(B, {}, out, why));
    CHECK(why.find("raise pe_max_work (currently 1)") != std::string::npos);
}

}  // namespace

int main() {
    RUN(test_inf_work_message);
    REPORT();
}
