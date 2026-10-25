// J-a N3 — the (G) builder's pullback (qeccore/record_general.hpp), against independent oracles:
//   * DENSE (2^n x 2^n complex matrices, n <= 4): term_pullback(B, S, M) == C0† M C0 EXACTLY (phase
//     included) for C0 = Π_{j∈S} rot(B_j), rot(B) = (I + iB)/√2, over random commuting signed axis sets
//     (Z-type and mixed), random S and random Hermitian M. The opposite dressing sign (−iB_j, design
//     §2.7's FORWARD-conjugation sign) is NOT the pullback: counted, must differ on some cases;
//   * the diagonal builder's own pullback: term_pullback over diag_plan_axes(nf) (all terms) ==
//     diag_pullback(nf, M) exactly (x, z, phase) — the pullback half of G-9a (D vs G field identity).
// The table half of G-9a / G-9b / G-9c and the null form's op counts run in tests/test_ja_dual_run.py
// (they need a sampler's G, bare state and record columns).
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "check.hpp"
#include "qeccore/record_general.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/term_residual.hpp"

using namespace qeccore;
using cd = std::complex<double>;

namespace {

struct Dense {
    int d = 0;
    std::vector<cd> m;
    explicit Dense(int n) : d(1 << n), m((size_t)d * d, 0.0) {}
    cd& at(int r, int c) { return m[(size_t)r * d + c]; }
    cd at(int r, int c) const { return m[(size_t)r * d + c]; }
};
Dense eye(int n) { Dense I(n); for (int i = 0; i < I.d; ++i) I.at(i, i) = 1.0; return I; }
Dense mul(const Dense& A, const Dense& B) {
    Dense C(0);
    C.d = A.d;
    C.m.assign((size_t)A.d * A.d, 0.0);
    for (int i = 0; i < A.d; ++i)
        for (int k = 0; k < A.d; ++k) {
            const cd a = A.at(i, k);
            if (a == 0.0) continue;
            for (int j = 0; j < A.d; ++j) C.at(i, j) += a * B.at(k, j);
        }
    return C;
}
Dense dag(const Dense& A) {
    Dense C(0);
    C.d = A.d;
    C.m.assign(A.m.size(), 0.0);
    for (int i = 0; i < A.d; ++i) for (int j = 0; j < A.d; ++j) C.at(i, j) = std::conj(A.at(j, i));
    return C;
}
// i^phase X^x Z^z: |b> -> i^phase (-1)^{z.b} |b ^ x>
Dense pauli_dense(int n, const Pauli& P) {
    Dense M(n);
    static const cd ip[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    uint64_t x = 0, z = 0;
    for (int q = 0; q < n; ++q) { if (P.xbit(q)) x |= 1ULL << q; if (P.zbit(q)) z |= 1ULL << q; }
    for (int b = 0; b < M.d; ++b)
        M.at((int)(b ^ x), b) = ip[P.phase & 3] * ((__builtin_popcountll(z & (uint64_t)b) & 1) ? -1.0 : 1.0);
    return M;
}
Dense rot_dense(int n, const Pauli& B) {            // exp(i(π/4)B) = (I + iB)/√2
    Dense R = pauli_dense(n, B), I = eye(n);
    for (size_t i = 0; i < R.m.size(); ++i) R.m[i] = (I.m[i] + cd(0, 1) * R.m[i]) / std::sqrt(2.0);
    return R;
}
bool eq_exact(const Dense& A, const Dense& B) {
    for (size_t i = 0; i < A.m.size(); ++i) if (std::abs(A.m[i] - B.m[i]) > 1e-12) return false;
    return true;
}
int ov(const Pauli& P) {
    int c = 0;
    for (int q = 0; q < P.n; ++q) c += (P.xbit(q) && P.zbit(q)) ? 1 : 0;
    return c;
}
Pauli random_hermitian(std::mt19937_64& rng, int n, bool ztype) {
    Pauli P(n);
    for (int q = 0; q < n; ++q) {
        const int c = (int)(rng() % 4);
        if (!ztype && (c == 1 || c == 2)) P.setx(q);
        if (c == 2 || c == 3) P.setz(q);
    }
    P.phase = (ov(P) + 2 * (int)(rng() % 2)) & 3;   // Hermitian: phase ≡ |x∧z| (mod 2); random sign
    return P;
}

void test_pullback_dense() {
    std::mt19937_64 rng(20261018);
    long cases = 0, dressed = 0, other_sign_differs = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const int n = 1 + (int)(rng() % 4);
        const bool ztype = trial % 3 == 0;
        std::vector<Pauli> B;
        const int want = 1 + (int)(rng() % 5);
        for (int tries = 0; tries < 50 && (int)B.size() < want; ++tries) {
            Pauli c = random_hermitian(rng, n, ztype);
            bool ok = false;
            for (int q = 0; q < n; ++q) ok |= c.xbit(q) || c.zbit(q);
            for (const Pauli& b : B) ok &= !Pauli::anticommute_bit(b, c);
            if (ok) B.push_back(c);
        }
        if (B.empty()) continue;
        const TermResidual tv(n, B);
        const int T = (int)B.size();
        std::vector<uint64_t> S(1, rng() & ((1ULL << T) - 1));
        Dense C0 = eye(n);
        for (int j = 0; j < T; ++j) if ((S[0] >> j) & 1) C0 = mul(C0, rot_dense(n, B[(size_t)j]));
        for (int k = 0; k < 4; ++k) {
            const Pauli M = random_hermitian(rng, n, false);
            const Pauli Q = term_pullback(tv, S, M);
            const Dense want_d = mul(dag(C0), mul(pauli_dense(n, M), C0));
            CHECK(eq_exact(pauli_dense(n, Q), want_d));
            ++cases;
            // the other dressing sign: M · Π(−iB_j) over the anticommuting j ∈ S
            Pauli Qm = M;
            int na = 0;
            for (int j = 0; j < T; ++j)
                if (((S[0] >> j) & 1) && Pauli::anticommute_bit(M, B[(size_t)j])) {
                    Pauli mB = B[(size_t)j];
                    mB.phase = (mB.phase + 3) & 3;
                    Qm = Pauli::multiply(Qm, mB);
                    ++na;
                }
            if (na) ++dressed;
            if (!eq_exact(pauli_dense(n, Qm), want_d)) ++other_sign_differs;
        }
    }
    std::fprintf(stderr, "  dense pullback: %ld cases, %ld dressed, -iB_j form differs on %ld\n", cases, dressed,
                 other_sign_differs);
    CHECK(cases > 1500);
    CHECK(dressed > 500);
    CHECK(other_sign_differs > 100);       // the sign is load-bearing (non-vacuity of MUT-G-PULL's oracle)
}

void test_diag_axes_equal_diag_pullback() {
    std::mt19937_64 rng(7);
    long cases = 0, dressed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int n = 1 + (int)(rng() % 6);
        DiagNormalForm nf;
        nf.prefix = Pauli(n);
        nf.a.assign((size_t)n, 0);
        nf.diagonal_class = true;
        for (int q = 0; q < n; ++q) nf.a[(size_t)q] = (uint8_t)(rng() % 4);   // a & 1 is the plan content
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b)
                if (rng() % 3 == 0) nf.cz.emplace_back(a, b);
        const std::vector<Pauli> axes = diag_plan_axes(nf, n);
        if (axes.empty()) continue;
        const TermResidual tv(n, axes);
        std::vector<uint64_t> S((size_t)tv.term_words(), 0);
        for (int j = 0; j < tv.num_terms(); ++j) S[(size_t)(j >> 6)] |= 1ULL << (j & 63);
        for (int k = 0; k < 8; ++k) {
            const Pauli M = random_hermitian(rng, n, false);
            const Pauli g = term_pullback(tv, S, M);
            const Pauli d = diag_pullback(nf, M);
            CHECK(g.x == d.x && g.z == d.z && (g.phase & 3) == (d.phase & 3));
            ++cases;
            dressed += (g.z != M.z) ? 1 : 0;
        }
    }
    std::fprintf(stderr, "  diag axes: %ld cases, %ld dressed\n", cases, dressed);
    CHECK(cases > 2000);
    CHECK(dressed > 500);
}

}  // namespace

int main() {
    RUN(test_pullback_dense);
    RUN(test_diag_axes_equal_diag_pullback);
    REPORT();
}
