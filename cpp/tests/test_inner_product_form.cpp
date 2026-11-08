// test_inner_product_form.cpp — perf-engine §3.1: AffineState::inner_product's closed-form linear coefficients and
// its structural memo against an INDEPENDENT reference overlap.
//
// Reference (this test's own, over the states' public (b, R, D, J, ω)): the pre-§3.1 algorithm verbatim in
// structure — the joint system [R_φ | R_ψ] y = b_φ ⊕ b_ψ by the byte gf2_solve, the ℤ₄ form E(z) = Q_ψ − Q_φ over the
// kernel SAMPLED (c0 = E(0), L[l] = E(e_l) − c0, K from E(e_a ⊕ e_b)), then gauss_sum — an exact ExactPhase, so the
// engine's value must match it field for field (is_zero / scale / z8). A second reference for n ≤ 10: the dense
// statevectors' inner product (1e-9).
// States: random Clifford circuits on |0^n⟩ (H-rich, so the affine rank k and the joint kernel r are large — the
// multi-word regime of the closed form and the memo's r ≥ 8 gate), probes = the state with random Paulis applied
// (same R / J / D mod 2: the memo's hit case, every call checked), with extra gates (a miss), and after
// Z-projections with both outcomes (the ray construction's variants). Disjoint-support pairs are checked to be 0.
#include "check.hpp"

#include <complex>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "qeccore/gf2_gauss.hpp"
#include "qeccore/stab_affine.hpp"

using namespace qeccore;

namespace {

// ── the reference overlap (sampled form extraction, byte solver) ──
int Qof(const AffineState& s, const std::vector<uint8_t>& y) {
    long q = 0;
    for (int j = 0; j < s.k_; ++j) if (y[(size_t)j]) q += s.D[(size_t)j];
    for (int a = 0; a < s.k_; ++a) if (y[(size_t)a])
        for (int c = a + 1; c < s.k_; ++c) if (y[(size_t)c] && s.J.get(a, c)) q += 2;
    return (int)((q % 4 + 4) % 4);
}

ExactPhase ref_inner(const AffineState& phi, const AffineState& psi) {
    const int n = phi.n_, kphi = phi.k_, kpsi = psi.k_, N = kphi + kpsi;
    std::vector<std::vector<uint8_t>> A((size_t)n, std::vector<uint8_t>((size_t)N, 0));
    std::vector<uint8_t> rhs((size_t)n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < kphi; ++j) A[(size_t)i][(size_t)j] = phi.R.get(i, j);
        for (int j = 0; j < kpsi; ++j) A[(size_t)i][(size_t)(kphi + j)] = psi.R.get(i, j);
        rhs[(size_t)i] = (uint8_t)((phi.b[(size_t)i] ^ psi.b[(size_t)i]) & 1);
    }
    std::vector<uint8_t> p0;
    std::vector<std::vector<uint8_t>> ker;
    if (!gf2_solve(A, rhs, N, p0, ker)) return ExactPhase::zero();
    const int r = (int)ker.size();
    auto E = [&](const std::vector<uint8_t>& z) {
        std::vector<uint8_t> x = p0;
        for (int l = 0; l < r; ++l) if (z[(size_t)l]) for (int j = 0; j < N; ++j) x[(size_t)j] ^= ker[(size_t)l][(size_t)j];
        std::vector<uint8_t> u(x.begin(), x.begin() + kphi), w(x.begin() + kphi, x.end());
        return ((Qof(psi, w) - Qof(phi, u)) % 4 + 4) % 4;
    };
    std::vector<uint8_t> z((size_t)r, 0);
    const int c0 = E(z);
    std::vector<int> L((size_t)r, 0);
    for (int l = 0; l < r; ++l) { z[(size_t)l] = 1; L[(size_t)l] = ((E(z) - c0) % 4 + 4) % 4; z[(size_t)l] = 0; }
    std::vector<std::vector<uint8_t>> K((size_t)r, std::vector<uint8_t>((size_t)r, 0));
    for (int a = 0; a < r; ++a)
        for (int b = a + 1; b < r; ++b) {
            z[(size_t)a] = 1; z[(size_t)b] = 1;
            const int val = ((E(z) - c0 - L[(size_t)a] - L[(size_t)b]) % 4 + 4) % 4;
            z[(size_t)a] = 0; z[(size_t)b] = 0;
            K[(size_t)a][(size_t)b] = K[(size_t)b][(size_t)a] = (uint8_t)((val / 2) & 1);
        }
    ExactPhase acc = gauss_sum(r, c0, std::move(L), std::move(K));
    ExactPhase res = phi.omega.conj().mul(psi.omega).mul(acc);
    if (res.is_zero) return ExactPhase::zero();
    res.scale -= (kphi + kpsi);
    return res;
}

bool same(const ExactPhase& a, const ExactPhase& b) {
    if (a.is_zero || b.is_zero) return a.is_zero == b.is_zero;
    return a.scale == b.scale && a.z8 == b.z8;
}

long checked = 0, nonzero = 0, big_r = 0;

void check_pair(const AffineState& phi, const AffineState& psi, bool dense) {
    const ExactPhase ref = ref_inner(phi, psi);
    const ExactPhase got = phi.inner_product(psi);
    CHECK(same(ref, got));
    if (!same(ref, got))
        std::fprintf(stderr, "  n=%d kphi=%d kpsi=%d ref(%d,%d,%d) got(%d,%d,%d)\n", phi.n_, phi.k_, psi.k_,
                     (int)ref.is_zero, ref.scale, ref.z8, (int)got.is_zero, got.scale, got.z8);
    ++checked;
    if (!ref.is_zero) ++nonzero;
    if (phi.k_ + psi.k_ > 63) ++big_r;
    if (dense) {
        const auto a = phi.to_statevector(), b = psi.to_statevector();
        std::complex<double> ip(0, 0);
        for (size_t i = 0; i < a.size(); ++i) ip += std::conj(a[i]) * b[i];
        const std::complex<double> g = got.is_zero ? std::complex<double>(0, 0) : got.to_complex();
        CHECK(std::abs(ip - g) < 1e-9);
    }
}

void random_clifford(std::mt19937_64& rng, AffineState& s, int gates) {
    const int n = s.n_;
    for (int t = 0; t < gates; ++t) {
        const int q = (int)(rng() % (uint64_t)n);
        switch (rng() % 6) {
            case 0: s.apply_h(q); break;
            case 1: s.apply_s(q); break;
            case 2: { int q2 = (int)(rng() % (uint64_t)n); if (q2 != q) s.apply_cx(q, q2); break; }
            case 3: { int q2 = (int)(rng() % (uint64_t)n); if (q2 != q) s.apply_cz(q, q2); break; }
            case 4: s.apply_h(q); break;
            default: s.apply_x(q); break;
        }
    }
}

void random_pauli(std::mt19937_64& rng, AffineState& s) {
    const int n = s.n_;
    for (int q = 0; q < n; ++q) {
        switch (rng() % 4) { case 1: s.apply_x(q); break; case 2: s.apply_y(q); break; case 3: s.apply_z(q); break; default: break; }
    }
}

void test_random() {
    std::mt19937_64 rng(31337);
    const int ns[] = {1, 2, 4, 7, 10, 20, 40, 70, 100, 130};
    for (int n : ns) {
        const bool dense = n <= 10;
        for (int trial = 0; trial < (n <= 10 ? 12 : 4); ++trial) {
            AffineState phi(n);
            random_clifford(rng, phi, 3 * n + 10);
            // (a) the memo's hit case: Pauli variants of both sides, repeatedly
            for (int v = 0; v < 6; ++v) {
                AffineState a(phi), b(phi);
                random_pauli(rng, a);
                random_pauli(rng, b);
                check_pair(a, b, dense);
                check_pair(b, a, dense);
            }
            // (b) self and a few more gates (a miss, mostly nonzero overlaps)
            check_pair(phi, phi, dense);
            AffineState psi(phi);
            random_clifford(rng, psi, 1 + (int)(rng() % 4));
            check_pair(phi, psi, dense);
            check_pair(psi, phi, dense);
            // (c) Z-projections with both outcomes (a ray construction's variants: same R, D even part differs)
            {
                AffineState r0(phi), r1(phi);
                const int q = (int)(rng() % (uint64_t)n);
                const ExactPhase f0 = r0.pauli_project(2, q, +1);
                const ExactPhase f1 = r1.pauli_project(2, q, -1);
                if (!f0.is_zero && !f1.is_zero) {
                    check_pair(r0, r1, dense);
                    check_pair(r0, r0, dense);
                    AffineState r0p(r0);
                    random_pauli(rng, r0p);
                    check_pair(r0, r0p, dense);
                }
            }
            // (d) an unrelated state (usually disjoint supports at larger n)
            AffineState other(n);
            random_clifford(rng, other, 3 * n + 10);
            check_pair(phi, other, dense);
        }
    }
    std::fprintf(stderr, "[inner_product_form] pairs %ld (nonzero %ld, multiword %ld)\n", checked, nonzero, big_r);
    CHECK(nonzero > 0);
    CHECK(big_r > 0);
}

// Cross-J pairs (review finding 1): the two sides' quadratic forms DIFFER, so a wrong self-quadratic term cannot cancel
// between ψ and φ as it does for Pauli variants / projections of one state. GHZ-type low-rank states against CZ-graph
// states (the triangle: S(g) = 3 with no odd-degree vertex), complete graphs, random CZ graphs with S layers, and a
// state against itself plus a few extra gates — every overlap nonzero by construction or mostly so.
void test_cross_J() {
    long before = nonzero;
    {   // |GHZ_3> vs the triangle graph state: the smallest counterexample of the mod-2 term
        AffineState phi(3), psi(3);
        phi.apply_h(0); phi.apply_cx(0, 1); phi.apply_cx(0, 2);
        for (int q = 0; q < 3; ++q) psi.apply_h(q);
        psi.apply_cz(0, 1); psi.apply_cz(0, 2); psi.apply_cz(1, 2);
        check_pair(phi, psi, true);
        check_pair(psi, phi, true);
    }
    for (int n = 3; n <= 9; ++n) {   // GHZ_n vs K_n
        AffineState phi(n), psi(n);
        phi.apply_h(0); for (int q = 1; q < n; ++q) phi.apply_cx(0, q);
        for (int q = 0; q < n; ++q) psi.apply_h(q);
        for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) psi.apply_cz(a, b);
        check_pair(phi, psi, true);
        check_pair(psi, phi, true);
    }
    std::mt19937_64 rng(424242);
    const int ns[] = {3, 4, 5, 6, 8, 10, 20, 40, 70, 100, 130};
    for (int n : ns) {
        const bool dense = n <= 10;
        for (int trial = 0; trial < (n <= 10 ? 60 : 8); ++trial) {
            // low-rank GHZ-type phi (a few H, many CX) vs a CZ-dense psi with S layers
            AffineState phi(n), psi(n);
            const int m = 1 + (int)(rng() % 3);
            for (int q = 0; q < m && q < n; ++q) phi.apply_h(q);
            for (int e = 0; e < 2 * n; ++e) { int a = (int)(rng() % (uint64_t)n), b = (int)(rng() % (uint64_t)n); if (a != b) phi.apply_cx(a, b); }
            for (int q = 0; q < n; ++q) psi.apply_h(q);
            for (int e = 0; e < 2 * n; ++e) { int a = (int)(rng() % (uint64_t)n), b = (int)(rng() % (uint64_t)n); if (a != b) psi.apply_cz(a, b); }
            for (int e = 0; e < n; ++e) if (rng() % 3 == 0) psi.apply_s((int)(rng() % (uint64_t)n));
            check_pair(phi, psi, dense);
            check_pair(psi, phi, dense);
            // the same state with extra CZ / S / CX gates: different J, (mostly) overlapping supports
            AffineState chi(psi);
            for (int e = 0; e < 1 + (int)(rng() % 6); ++e) {
                const int a = (int)(rng() % (uint64_t)n), b = (int)(rng() % (uint64_t)n);
                switch (rng() % 3) { case 0: if (a != b) chi.apply_cz(a, b); break; case 1: chi.apply_s(a); break; default: if (a != b) chi.apply_cx(a, b); break; }
            }
            check_pair(psi, chi, dense);
            check_pair(chi, psi, dense);
        }
    }
    std::fprintf(stderr, "[inner_product_form] cross-J pairs: %ld nonzero\n", nonzero - before);
    CHECK(nonzero - before > 100);
}

}  // namespace

int main() {
    RUN(test_random);
    RUN(test_cross_J);
    REPORT();
}
