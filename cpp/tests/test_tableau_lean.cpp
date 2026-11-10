// test_tableau_lean.cpp — lean L6 (compile time): the frame build's new routines against the routines they replaced,
// row for row and phase for phase (every output is a unique Pauli, so the oracles are the former implementations):
//   1. CliffordTableau::left_batch (transposed replay) == left_rowwise (the left_* calls in order): forward rows,
//      inverse rows and dual validity, with the dual valid and stale on entry, lists below and above the row-wise cut.
//   2. CliffordTableau::reset_dual (column planes) == reset_dual_oracle (2n dual_images).
//   3. CliffordTableau::from_generators (sparse RREF + plane Gram–Schmidt) == from_generators_dense_oracle on identity-like,
//      sparse and dense commuting generator sets.
//   4. CanonicalStabSum's lazy anchor: reading the anchor after every gate and once at the end give the identical
//      AffineState (b, R, D, J, omega), and on a real bare-state build the lazily replayed anchor is the frame's state
//      (anchor_from_frame: |<anchor|reconstructed>| = 1) while the gates were pending until the read.
#include "check.hpp"

#include <complex>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/ref_io.hpp"
#include "qeccore/stab_affine.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

bool same_pauli(const Pauli& a, const Pauli& b) {
    return a.n == b.n && a.x == b.x && a.z == b.z && (((a.phase - b.phase) % 4) + 4) % 4 == 0;
}
bool same_rows(const std::vector<Pauli>& a, const std::vector<Pauli>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!same_pauli(a[i], b[i])) return false;
    return true;
}
bool same_frame(const CliffordTableau& a, const CliffordTableau& b) {
    if (a.n != b.n || !same_rows(a.Xrow, b.Xrow) || !same_rows(a.Zrow, b.Zrow)) return false;
    if (a.dual_valid() != b.dual_valid()) return false;
    if (!a.dual_valid()) return true;
    for (int q = 0; q < a.n; ++q)
        if (!same_pauli(a.xinv_row(q), b.xinv_row(q)) || !same_pauli(a.zinv_row(q), b.zinv_row(q))) return false;
    return true;
}

std::vector<CliffordTableau::LeftGate> random_gates(std::mt19937_64& rng, int n, size_t count) {
    std::vector<CliffordTableau::LeftGate> g;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t kind = (uint8_t)(rng() % 8);
        const int a = (int)(rng() % (uint64_t)n);
        int b = -1;
        if (kind >= 6) { if (n < 2) continue; do b = (int)(rng() % (uint64_t)n); while (b == a); }
        g.push_back({kind, a, b});
    }
    return g;
}

CliffordTableau random_frame(std::mt19937_64& rng, int n, size_t gates) {
    CliffordTableau U(n);
    U.left_rowwise(random_gates(rng, n, gates));   // the dual stays valid (patched per gate)
    return U;
}

const int NS[] = {1, 3, 17, 64, 65, 130, 301};

void test_left_batch() {
    std::mt19937_64 rng(0x1e6a);
    int checked = 0;
    for (int n : NS)
        for (size_t count : {size_t(1), size_t(31), size_t(32), size_t(33), size_t(200), size_t(1500)}) {
            CliffordTableau U0 = random_frame(rng, n, (size_t)3 * (size_t)n + 1);
            const std::vector<CliffordTableau::LeftGate> G = random_gates(rng, n, count);
            for (int stale = 0; stale < 2; ++stale) {
                CliffordTableau A = U0, B = U0;
                if (stale) { A.invalidate_dual(); B.invalidate_dual(); }
                A.left_rowwise(G);
                B.left_batch(G);
                CHECK(same_frame(A, B));
                CHECK(A.dual_valid() == !stale);
                ++checked;
            }
        }
    std::fprintf(stderr, "[left_batch] %d (frame, list) pairs equal\n", checked);
}

void test_reset_dual() {
    std::mt19937_64 rng(0x2e6a);
    int checked = 0;
    for (int n : NS)
        for (int trial = 0; trial < 3; ++trial) {
            CliffordTableau U = random_frame(rng, n, (size_t)(trial + 1) * (size_t)n + 1);
            CliffordTableau A = U, B = U;
            A.invalidate_dual(); B.invalidate_dual();
            A.reset_dual();
            B.reset_dual_oracle();
            CHECK(A.dual_valid() && B.dual_valid());
            CHECK(same_frame(A, B));
            ++checked;
        }
    std::fprintf(stderr, "[reset_dual] %d frames equal to the oracle\n", checked);
}

void test_from_generators() {
    std::mt19937_64 rng(0x3e6a);
    int checked = 0;
    for (int n : NS)
        for (int kind = 0; kind < 4; ++kind) {
            // kind 0: identity generators with random signs; 1: a few gates (sparse); 2/3: many gates (dense)
            const size_t gates = kind == 0 ? 0 : kind == 1 ? (size_t)n / 4 + 1 : (size_t)(kind + 1) * (size_t)n;
            CliffordTableau F = random_frame(rng, n, gates);
            std::vector<Pauli> gens = F.Zrow;
            for (Pauli& g : gens) if (rng() & 1) g.phase = (g.phase + 2) & 3;
            CliffordTableau A = CliffordTableau::from_generators(gens);
            CliffordTableau B = CliffordTableau::from_generators_dense_oracle(gens);
            CHECK(same_frame(A, B));
            CHECK(same_rows(A.Zrow, gens));
            ++checked;
        }
    std::fprintf(stderr, "[from_generators] %d generator sets equal to the dense oracle\n", checked);
}

bool same_affine(const AffineState& p, const AffineState& q) {
    if (p.n_ != q.n_ || p.k_ != q.k_ || p.b != q.b || p.D != q.D) return false;
    if (!p.omega.exact_eq(q.omega)) return false;
    for (int i = 0; i < p.n_; ++i)
        for (int j = 0; j < p.k_; ++j) if (p.R.get(i, j) != q.R.get(i, j)) return false;
    for (int i = 0; i < p.k_; ++i)
        for (int j = 0; j < p.k_; ++j) if (p.J.get(i, j) != q.J.get(i, j)) return false;
    return true;
}

void apply_gate(CanonicalStabSum& s, const CliffordTableau::LeftGate& g) {
    switch (g.kind) {
        case 0: s.apply_h(g.a); break;
        case 1: s.apply_s(g.a); break;
        case 2: s.apply_sdg(g.a); break;
        case 3: s.apply_x(g.a); break;
        case 4: s.apply_y(g.a); break;
        case 5: s.apply_z(g.a); break;
        case 6: s.apply_cx(g.a, g.b); break;
        default: s.apply_cz(g.a, g.b); break;
    }
}

void test_lazy_anchor_representation() {
    std::mt19937_64 rng(0x4e6a);
    int checked = 0;
    for (int n : {2, 5, 9, 70})
        for (int trial = 0; trial < 3; ++trial) {
            // a random stabilizer state as the one ray (chi = 1)
            auto psi = std::make_unique<AffineState>(n);
            for (const CliffordTableau::LeftGate& g : random_gates(rng, n, (size_t)4 * (size_t)n)) {
                switch (g.kind) {
                    case 0: psi->apply_h(g.a); break;
                    case 1: psi->apply_s(g.a); break;
                    case 2: psi->apply_sdg(g.a); break;
                    case 3: psi->apply_x(g.a); break;
                    case 4: psi->apply_y(g.a); break;
                    case 5: psi->apply_z(g.a); break;
                    case 6: psi->apply_cx(g.a, g.b); break;
                    default: psi->apply_cz(g.a, g.b); break;
                }
            }
            auto one_ray = [&]() {
                std::vector<std::unique_ptr<AffineState>> rays;
                rays.push_back(std::unique_ptr<AffineState>(static_cast<AffineState*>(psi->clone().release())));
                return CanonicalStabSum::from_rays(n, std::move(rays), {std::complex<double>(1.0, 0.0)});
            };
            CanonicalStabSum eager = one_ray(), lazy = one_ray();
            const std::vector<CliffordTableau::LeftGate> G = random_gates(rng, n, 100);
            for (const auto& g : G) { apply_gate(eager, g); (void)eager.anchor_state(); }   // read after every gate
            for (const auto& g : G) apply_gate(lazy, g);
            CHECK(lazy.anchor_pending() == G.size());
            CHECK(eager.anchor_pending() == 0);
            CHECK(same_affine(eager.anchor_state(), lazy.anchor_state()));
            CHECK(lazy.anchor_pending() == 0);
            lazy.ensure_frame_current(); eager.ensure_frame_current();
            CHECK(same_frame(eager.U, lazy.U));
            ++checked;
        }
    std::fprintf(stderr, "[lazy_anchor] %d (eager, lazy) pairs equal\n", checked);
}

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

void test_lazy_anchor_bare_state() {
    const char* texts[] = {
        "RX 0 1 2\nCX 0 1\nT 1\nCX 1 2\nH 2\nS 0\nM 0 1 2\nDETECTOR rec[-1] rec[-2]\n",
        "R 0 1 2 3\nH 0\nCX 0 1 1 2 2 3\nT 3\nCX 2 3\nH 1\nCZ 0 2\nMX 0\nR 0\nCX 1 0\nM 0 1 2 3\n",
        "RX 0 1 2 3 4\nCX 0 1 2 3\nT 0 1\nCZ 1 4\nH 3\nS 4\nCX 4 2\nM 0 1 2 3 4\n",
    };
    int built = 0;
    for (const char* t : texts) {
        BareState bs = build_bare_state(deferred_of(t));
        CHECK(!bs.rejected);
        if (bs.rejected) continue;
        CHECK(bs.state.anchor_pending() > 0);                 // the mechanism is live: nothing replayed yet
        FramedSuperposition L = FramedSuperposition::from_css(bs.state);   // reads U, eps, free, branches only
        CHECK(bs.state.anchor_pending() > 0);
        AffineState rec = anchor_from_frame(L);
        const AffineState& anc = bs.state.anchor_state();     // the lazy replay
        CHECK(bs.state.anchor_pending() == 0);
        ExactPhase ov = anc.inner_product(rec);
        CHECK(!ov.is_zero && ov.scale == 0);                  // the same ray
        ++built;
    }
    std::fprintf(stderr, "[lazy_anchor_bare] %d bare states: the replayed anchor is the frame's state\n", built);
}

}  // namespace

int main() {
    RUN(test_left_batch);
    RUN(test_reset_dual);
    RUN(test_from_generators);
    RUN(test_lazy_anchor_representation);
    RUN(test_lazy_anchor_bare_state);
    REPORT();
}
