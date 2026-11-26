// test_trivial_completion.cpp — the trivial completion of a Clifford circuit's (empty) magic layer.
//
// build_parity_form (normal_form.cpp) skips diagonalize_commuting_layer when the rotation form has no term: the
// parity form's cprime is empty and restrict_to_support builds the support |0^n> with k == 0 free variables
// (Stage 1 T1). Before, the general routine completed the empty Lagrangian by {X_q} and the disentangler emitted
// H_q on every wire: the bare state was built on |+>^n (k = n) and rotated back by the H_q's in the output
// Clifford. That route is kept here as the ORACLE (`old_route`: the same restrict / FastTODD / assemble calls on
// C' = H^{⊗n}), and the production bare state must equal it byte for byte: frame rows and phases, inverse rows, eps,
// free, anchor, branch coefficients. diagonalize_commuting_layer itself keeps the general completion on an empty
// layer (its other caller, compose_carried_bare, depends on that layout): the test pins that its cprime on the empty
// layer IS the oracle's H list.
#include "check.hpp"

#include <complex>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/parity_assemble.hpp"
#include "qeccore/parity_support.hpp"
#include "qeccore/pauli_diagonalize.hpp"
#include "qeccore/pauli_rotation_form.hpp"
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

// The pre-T1 route for a Clifford circuit: C' = H_{n-1}···H_0 (cprime[0] first), support |+>^n, the H_q's in the
// output Clifford before the circuit's own Cliffords. Every stage after the completion is the production one.
ParityBareState old_route(const Circuit& deferred, const PauliRotationForm& f, const DiagResult& diag) {
    const int n = deferred.n;
    std::vector<CliffGate> cprime_old;                 // = reversed, daggered G with G = [H_0, ..., H_{n-1}]
    for (int q = n - 1; q >= 0; --q) cprime_old.push_back({GateKind::H, {q}});
    std::vector<CliffGate> cprime_dagger_old(cprime_old.rbegin(), cprime_old.rend());   // H self-adjoint
    ParitySupport ps = restrict_to_support(cprime_dagger_old, diag.tz, n);
    CHECK_EQ(ps.support.k_, n);
    CHECK(ps.columns.empty());                         // no term has a parity column: nothing for FastTODD to reduce
    std::vector<CliffGate> out_cliff = cprime_old;
    out_cliff.insert(out_cliff.end(), f.cliffords.begin(), f.cliffords.end());
    return assemble_from_parities(ps, out_cliff, n, AssemblePath::coset_family, nullptr);
}

std::string css_first_diff(const CanonicalStabSum& a, const CanonicalStabSum& b) {
    a.ensure_frame_current();
    b.ensure_frame_current();
    if (a.n() != b.n()) return "n";
    const int n = a.n();
    auto pauli_eq = [](const Pauli& x, const Pauli& y) {
        return x.n == y.n && x.phase == y.phase && x.x == y.x && x.z == y.z;
    };
    for (int r = 0; r < n; ++r) {
        if (!pauli_eq(a.U.dense().Zrow[(size_t)r], b.U.dense().Zrow[(size_t)r])) return "U.Zrow[" + std::to_string(r) + "]";
        if (!pauli_eq(a.U.dense().Xrow[(size_t)r], b.U.dense().Xrow[(size_t)r])) return "U.Xrow[" + std::to_string(r) + "]";
    }
    // The cached inverse rows, as the sampler's conjugate() reads them (both routes leave them valid).
    if (a.U.dense().dual_valid() != b.U.dense().dual_valid()) return "dual_valid";
    if (a.U.dense().dual_valid())
        for (int r = 0; r < n; ++r) {
            if (!pauli_eq(a.U.dense().zinv_row(r), b.U.dense().zinv_row(r))) return "Zinv[" + std::to_string(r) + "]";
            if (!pauli_eq(a.U.dense().xinv_row(r), b.U.dense().xinv_row(r))) return "Xinv[" + std::to_string(r) + "]";
        }
    if (a.eps != b.eps) return "eps";
    if (a.free != b.free) return "free";
    const AffineState &p = a.anchor_state(), &q = b.anchor_state();
    if (p.n_ != q.n_ || p.k_ != q.k_ || p.b != q.b || p.D != q.D) return "anchor (n/k/b/D)";
    if (!p.omega.exact_eq(q.omega)) return "anchor omega";
    for (int i = 0; i < p.n_; ++i)
        for (int j = 0; j < p.k_; ++j) if (p.R.get(i, j) != q.R.get(i, j)) return "anchor R";
    if (a.branches.size() != b.branches.size()) return "chi";
    for (size_t i = 0; i < a.branches.size(); ++i) {
        if (a.branches[i].sigma != b.branches[i].sigma) return "branch " + std::to_string(i) + " sigma";
        if (std::memcmp(&a.branches[i].c, &b.branches[i].c, sizeof(std::complex<double>)) != 0)
            return "branch " + std::to_string(i) + " coefficient bytes";
    }
    return "";
}

// A Clifford circuit: cprime empty, support k == 0, generators {Z_q}; the bare state equals the old route's.
void check_clifford(const std::string& text, const char* name) {
    Circuit deferred = deferred_of(text);
    const int n = deferred.n;
    PauliRotationForm f = build_pauli_rotation_form(deferred);
    CHECK(!f.rejected);
    CHECK(f.terms.empty());
    // The general routine on the empty layer: the old completion, H on every wire (cprime[0] first = H_{n-1}).
    DiagResult diag = diagonalize_commuting_layer(f.terms, n);
    CHECK(diag.tz.empty());
    CHECK_EQ((int)diag.cprime.size(), n);
    for (int i = 0; i < (int)diag.cprime.size(); ++i)
        CHECK(diag.cprime[(size_t)i].kind == GateKind::H && diag.cprime[(size_t)i].targets == std::vector<int>{n - 1 - i});

    ParityForm pf = build_parity_form(deferred);
    CHECK(!pf.rejected);
    CHECK(pf.cprime_dagger.empty());
    CHECK_EQ(pf.ps->support.k_, 0);
    CHECK(pf.ps->columns.empty());
    CHECK_EQ(pf.output_clifford.size(), f.cliffords.size());

    BareState bs = build_bare_state_parity(deferred, AssemblePath::coset_family);
    CHECK(!bs.rejected);
    CHECK_EQ(bs.chi, 1);
    ParityBareState old = old_route(deferred, f, diag);
    const std::string d = css_first_diff(bs.state, old.state);
    if (!d.empty()) std::fprintf(stderr, "  %s (n = %d): first difference at %s\n", name, n, d.c_str());
    CHECK(d.empty());
}

// Random Clifford circuits over R/RX/H/S/CX/CZ/M/MX/MR with noise, as the deferral sees them (fresh wires per
// measurement), n across word boundaries.
std::string random_clifford(std::mt19937_64& rng, int n, int depth) {
    std::string t;
    auto q = [&]() { return (int)(rng() % (uint64_t)n); };
    for (int i = 0; i < n; ++i) t += (rng() % 2 ? "R " : "RX ") + std::to_string(i) + "\n";
    for (int d = 0; d < depth; ++d) {
        const int a = q();
        int b = q();
        if (n > 1) while (b == a) b = q();
        switch (rng() % 9) {
            case 0: t += "H " + std::to_string(a) + "\n"; break;
            case 1: t += "S " + std::to_string(a) + "\n"; break;
            case 2: if (n > 1) t += "CX " + std::to_string(a) + " " + std::to_string(b) + "\n"; break;
            case 3: if (n > 1) t += "CZ " + std::to_string(a) + " " + std::to_string(b) + "\n"; break;
            case 4: t += "X_ERROR(0.01) " + std::to_string(a) + "\n"; break;
            case 5: t += "DEPOLARIZE1(0.01) " + std::to_string(a) + "\n"; break;
            case 6: t += "M " + std::to_string(a) + "\nDETECTOR rec[-1]\n"; break;
            case 7: t += "MR " + std::to_string(a) + "\n"; break;
            default: t += "MX " + std::to_string(a) + "\n"; break;
        }
    }
    for (int i = 0; i < n; ++i) t += "M " + std::to_string(i) + "\n";
    return t;
}

void clifford_circuits() {
    // The chain the Stage-1 benchmarks use, and a rotated surface-code memory round structure in miniature.
    std::string chain;
    for (int q = 0; q < 70; ++q)
        chain += "R " + std::to_string(q) + "\nX_ERROR(0.01) " + std::to_string(q) + "\nM " + std::to_string(q) +
                 "\nDETECTOR rec[-1]\n";
    check_clifford(chain, "chain70");
    check_clifford("R 0 1 2 3 4\nH 0 2 4\nCX 0 1 2 1 2 3 4 3\nDEPOLARIZE2(0.001) 0 1\nM 1 3\nDETECTOR rec[-1]\n"
                   "DETECTOR rec[-2]\nR 1 3\nCX 0 1 2 1 2 3 4 3\nM 1 3\nDETECTOR rec[-1] rec[-3]\n"
                   "DETECTOR rec[-2] rec[-4]\nMX 0 2 4\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]\n",
                   "rep5");
    std::mt19937_64 rng(0x7e1ULL);
    int cases = 0;
    for (int n : {1, 2, 3, 7, 20, 63, 64, 65}) {
        for (int rep = 0; rep < 4; ++rep) {
            check_clifford(random_clifford(rng, n, 3 * n + 4), "random");
            ++cases;
        }
    }
    CHECK(cases == 32);
}

// A magic circuit's layer is non-empty: it is diagonalized as before (cprime carries gates, k > 0).
void magic_circuit_unchanged() {
    Circuit deferred = deferred_of("RX 0 1 2\nT 0 1\nCX 0 2\nMX 0 1 2\n");
    PauliRotationForm f = build_pauli_rotation_form(deferred);
    CHECK(!f.rejected);
    CHECK(!f.terms.empty());
    DiagResult diag = diagonalize_commuting_layer(f.terms, deferred.n);
    CHECK(!diag.cprime.empty());
    ParityForm pf = build_parity_form(deferred);
    CHECK(!pf.rejected);
    CHECK(pf.ps->support.k_ > 0);
}

}  // namespace

int main() {
    RUN(clifford_circuits);
    RUN(magic_circuit_unchanged);
    REPORT();
}
