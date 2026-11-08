// S8 RF2: the closed-form Pauli expansion of a kind-0 content D = S^a CZ^M (rf_pauli_expansion.hpp) against brute
// force — exhaustive over every (a, M) on n <= 4 wires and 200 random contents on n <= 7: the support of
// D̂(z) = 2^{−n} Σ_x i^{q(x)} (−1)^{z·x} is exactly {z(t)}, D̂(z(t)) = c · i^{Q_D(t)} with ONE c and |c| = 2^{−r/2};
// padding invariance (duplicated CZ pairs and a + 2 expand to the same object); the dressed image == diag_pullback;
// a malformed CZ pair is refused by name (RF_CONTENT_MALFORMED).
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/rf_pauli_expansion.hpp"
#include "qeccore/twirl_kernel.hpp"

using namespace qeccore;
using cd = std::complex<double>;

namespace {
const cd kI[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

int q_brute(const DiagNormalForm& nf, uint64_t x) {
    int q = 0;
    for (size_t i = 0; i < nf.a.size(); ++i) if (((x >> i) & 1) && (nf.a[i] & 1)) ++q;
    for (const auto& e : nf.cz) if (((x >> e.first) & 1) && ((x >> e.second) & 1)) q += 2;
    return q & 3;
}

long n_checked = 0, n_rank = 0;

void check_one(const DiagNormalForm& nf, int n) {
    const DiagExpansion E = expand_diag(nf, n);
    const uint64_t N = 1ULL << n;
    std::vector<cd> Dh(N, 0.0);
    for (uint64_t z = 0; z < N; ++z) {
        cd s = 0.0;
        for (uint64_t x = 0; x < N; ++x) s += kI[q_brute(nf, x)] * ((__builtin_popcountll(z & x) & 1) ? -1.0 : 1.0);
        Dh[z] = s / (double)N;
    }
    std::vector<int> hit(N, 0);
    cd c = 0.0;
    bool first = true;
    for (uint64_t t = 0; t < (1ULL << E.r); ++t) {
        uint64_t z = E.z0[0];
        int Q = 0;
        for (int i = 0; i < E.r; ++i)
            if ((t >> i) & 1) {
                z ^= E.cols[(size_t)i];
                Q += E.lin[(size_t)i];
                for (int j = i + 1; j < E.r; ++j) if (((t >> j) & 1) && ((E.beta[(size_t)i * E.WB] >> j) & 1)) Q += 2;
            }
        CHECK(hit[z] == 0);
        hit[z] = 1;
        const cd ratio = Dh[z] / kI[Q & 3];
        if (first) { c = ratio; first = false; }
        CHECK(std::abs(ratio - c) <= 1e-12);
    }
    CHECK(std::abs(std::abs(c) - std::pow(2.0, -0.5 * E.r)) <= 1e-12);
    for (uint64_t z = 0; z < N; ++z) if (!hit[z]) CHECK(std::abs(Dh[z]) <= 1e-12);
    ++n_checked;
    if (E.r > 0) ++n_rank;
}

DiagNormalForm make(int n, uint64_t amask, const std::vector<std::pair<int, int>>& cz) {
    DiagNormalForm nf;
    nf.prefix = Pauli(n);
    nf.a.assign((size_t)n, 0);
    for (int q = 0; q < n; ++q) nf.a[(size_t)q] = (amask >> q) & 1;
    nf.cz = cz;
    nf.diagonal_class = true;
    return nf;
}

void test_exhaustive_small() {
    for (int n = 1; n <= 4; ++n) {
        std::vector<std::pair<int, int>> pairs;
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) pairs.push_back({i, j});
        for (uint64_t am = 0; am < (1ULL << n); ++am)
            for (uint64_t pm = 0; pm < (1ULL << pairs.size()); ++pm) {
                std::vector<std::pair<int, int>> cz;
                for (size_t e = 0; e < pairs.size(); ++e) if ((pm >> e) & 1) cz.push_back(pairs[e]);
                check_one(make(n, am, cz), n);
            }
    }
    std::printf("exhaustive: %ld contents (%ld with r > 0)\n", n_checked, n_rank);
}

void test_random() {
    std::mt19937_64 R(2026);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 1 + (int)(R() % 7);
        std::vector<std::pair<int, int>> cz;
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) if (R() % 5 < 2) cz.push_back({i, j});
        check_one(make(n, R() & ((1ULL << n) - 1), cz), n);
    }
}

bool same(const DiagExpansion& A, const DiagExpansion& B) {
    return A.r == B.r && A.piv == B.piv && A.cols == B.cols && A.z0 == B.z0 && A.lin == B.lin && A.beta == B.beta;
}

void test_padding_invariance() {
    std::mt19937_64 R(7);
    int ok = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 2 + (int)(R() % 70);
        std::vector<std::pair<int, int>> cz;
        for (int e = 0; e < 3 * n; ++e) {
            const int i = (int)(R() % n), j = (int)(R() % n);
            if (i != j) cz.push_back({std::min(i, j), std::max(i, j)});
        }
        DiagNormalForm A = make(n, 0, cz);
        for (int q = 0; q < n; ++q) A.a[(size_t)q] = R() & 1;
        DiagNormalForm B = A;
        for (int q = 0; q < n; ++q) if (R() & 1) B.a[(size_t)q] += 2;    // S^2 = Z: folded elsewhere, a & 1 only
        for (const auto& e : cz) if (R() & 1) { B.cz.push_back(e); B.cz.push_back(e); }   // CZ^2 = I
        const int i = (int)(R() % n), j = (i + 1) % n;
        B.cz.push_back({std::min(i, j), std::max(i, j)});
        B.cz.push_back({std::min(i, j), std::max(i, j)});
        CHECK(same(expand_diag(A, n), expand_diag(B, n)));
        ++ok;
    }
    CHECK(ok == 200);
}

void test_dress() {
    std::mt19937_64 R(11);
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 1 + (int)(R() % 90);
        std::vector<std::pair<int, int>> cz;
        for (int e = 0; e < n; ++e) {
            const int i = (int)(R() % n), j = (int)(R() % n);
            if (i != j) cz.push_back({std::min(i, j), std::max(i, j)});
        }
        DiagNormalForm nf = make(n, 0, cz);
        for (int q = 0; q < n; ++q) nf.a[(size_t)q] = R() & 1;
        const DiagExpansion E = expand_diag(nf, n);
        Pauli M(n);
        for (int q = 0; q < n; ++q) { if (R() % 3 == 0) M.setx(q); if (R() % 3 == 0) M.setz(q); }
        M.phase = (int)(R() & 3);
        const Pauli a = dress_diag(E, M), b = diag_pullback(nf, M);
        CHECK(a.x == b.x && a.z == b.z && (a.phase & 3) == (b.phase & 3));
    }
}

void test_malformed() {
    for (const auto& bad : std::vector<std::pair<int, int>>{{1, 1}, {0, 5}, {-1, 2}}) {
        bool threw = false;
        try { expand_diag(make(4, 0, {bad}), 4); }
        catch (const std::invalid_argument& e) { threw = std::string(e.what()).find("RF_CONTENT_MALFORMED") != std::string::npos; }
        CHECK(threw);
    }
}
// S8 RF3: the exact Gauss sum of a random Z4 quadratic form over the non-keep variables (keep = parameters) vs brute
// force, every keep assignment: zero / the affine support A / 2^{h/2} ω^{p8} i^{Qf(free)}; Q0's polarization identity.
void test_gauss_sum() {
    std::mt19937_64 R(31);
    long nz = 0, ndep = 0, nkeep = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int V = 1 + (int)(R() % 11);
        QForm Q;
        Q.init(V);
        Q.c = (int)(R() & 3);
        for (int j = 0; j < V; ++j) Q.lin[(size_t)j] = (uint8_t)(R() & 3);
        for (int j = 0; j < V; ++j) for (int k = j + 1; k < V; ++k) if (R() % 3 == 0) Q.flip(j, k);
        for (int t = 0; t < 20; ++t) {                    // polarization
            uint64_t u = R() & ((1ULL << V) - 1), v = R() & ((1ULL << V) - 1), uv = u ^ v, z = 0;
            const int q0 = Q.eval(&z);
            CHECK(((Q.eval(&uv) - q0) - (Q.eval(&u) - q0) - (Q.eval(&v) - q0) - 2 * Q.polar(&u, &v)) % 4 == 0);
        }
        std::vector<uint8_t> keep((size_t)V, 0);
        for (int j = 0; j < V; ++j) keep[(size_t)j] = (R() % 3 == 0);
        std::vector<int> K, NK;
        for (int j = 0; j < V; ++j) (keep[(size_t)j] ? K : NK).push_back(j);
        GaussSum G;
        QForm Qw = Q;
        gauss_sum(Qw, keep, G);
        const cd unit = std::polar(std::pow(2.0, 0.5 * G.h), M_PI / 4 * G.p8);
        for (uint64_t a = 0; a < (1ULL << K.size()); ++a) {
            uint64_t base = 0;
            for (size_t i = 0; i < K.size(); ++i) if ((a >> i) & 1) base |= 1ULL << K[i];
            long re = 0, im = 0;
            for (uint64_t b = 0; b < (1ULL << NK.size()); ++b) {
                uint64_t u = base;
                for (size_t i = 0; i < NK.size(); ++i) if ((b >> i) & 1) u |= 1ULL << NK[i];
                const int q = Q.eval(&u);
                re += q == 0 ? 1 : q == 2 ? -1 : 0;
                im += q == 1 ? 1 : q == 3 ? -1 : 0;
            }
            if (G.zero) { CHECK(re == 0 && im == 0); continue; }
            bool inA = true;
            for (size_t d = 0; d < G.dep_var.size(); ++d)
                if ((int)((base >> G.dep_var[d]) & 1) != (G.dep_c[d] ^ (__builtin_popcountll(G.dep_mask[d] & base) & 1)))
                    inA = false;
            if (!inA) { CHECK(re == 0 && im == 0); continue; }
            uint64_t fr = 0;
            for (size_t i = 0; i < G.free_keep.size(); ++i) if ((base >> G.free_keep[i]) & 1) fr |= 1ULL << i;
            const cd want = unit * kI[G.f.eval(&fr) & 3];
            CHECK(std::abs(want - cd((double)re, (double)im)) <= 1e-9);
        }
        nz += G.zero;
        ndep += !G.dep_var.empty();
        nkeep += !K.empty();
    }
    std::printf("gauss_sum: 3000 forms (%ld zero, %ld with dependent keeps, %ld with keeps)\n", nz, ndep, nkeep);
    CHECK(nz > 50 && ndep > 50);
}
}  // namespace

int main() {
    RUN(test_gauss_sum);
    RUN(test_exhaustive_small);
    RUN(test_random);
    RUN(test_padding_invariance);
    RUN(test_dress);
    RUN(test_malformed);
    REPORT();
}
