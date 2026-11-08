// J-a N2 — the term-indexed residual (qeccore/term_residual.hpp), against an independent DENSE oracle
// (2^n x 2^n complex matrices, n <= 4):
//   G-TK2  the sequential kernel fold (adv-A kfold.py: 256/256 subsets, the two-vector basis; every
//          reduce_key_ result) and the mirror-pair counterexample (B1 = B2 = Z: no silent Z);
//   G-TK6  the permanent term-commutation assert (a named throw, reachable in a release build);
//   composition (compose_atom_) against the dense product; kind_of_ (exact diagonality: a Z-type but
//   non-diagonal action is kind 2); compose_terms_ performs no allocation.
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/term_residual.hpp"

using namespace qeccore;
using cd = std::complex<double>;

// ── allocation counter (compose_terms_ must not allocate) ──
static std::atomic<long> g_allocs{0};
void* operator new(size_t sz) {
    ++g_allocs;
    if (void* p = std::malloc(sz ? sz : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

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
Dense rot_dense(int n, const Pauli& B) {            // (I + iB)/sqrt(2)
    Dense R = pauli_dense(n, B), I = eye(n);
    for (size_t i = 0; i < R.m.size(); ++i) R.m[i] = (I.m[i] + cd(0, 1) * R.m[i]) / std::sqrt(2.0);
    return R;
}
Dense rotS_dense(int n, const std::vector<Pauli>& B, const std::vector<uint64_t>& S) {
    Dense R = eye(n);
    for (size_t j = 0; j < B.size(); ++j)
        if ((S[j >> 6] >> (j & 63)) & 1) R = mul(R, rot_dense(n, B[j]));
    return R;
}
bool eq_phase(const Dense& A, const Dense& B) {
    size_t im = 0;
    for (size_t i = 0; i < B.m.size(); ++i) if (std::abs(B.m[i]) > std::abs(B.m[im])) im = i;
    if (std::abs(B.m[im]) < 1e-12) return false;
    const cd ph = A.m[im] / B.m[im];
    if (std::abs(std::abs(ph) - 1.0) > 1e-9) return false;
    for (size_t i = 0; i < A.m.size(); ++i) if (std::abs(A.m[i] - ph * B.m[i]) > 1e-9) return false;
    return true;
}
Pauli zp(int n, std::initializer_list<int> qs) { Pauli P(n); for (int q : qs) P.setz(q); return P; }
std::vector<uint64_t> set_of(int T, unsigned mask) {
    std::vector<uint64_t> S((size_t)((T + 63) / 64), 0);
    for (int j = 0; j < T; ++j) if ((mask >> j) & 1) S[0] |= 1ULL << j;
    return S;
}

// adv-A kfold.py: n = 4; terms Z1234, the six pairs, and a duplicate Z12 (index 7 = index 1).
std::vector<Pauli> kfold_terms() {
    std::vector<Pauli> B;
    B.push_back(zp(4, {0, 1, 2, 3}));
    for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) B.push_back(zp(4, {a, b}));
    B.push_back(zp(4, {0, 1}));
    return B;
}

void test_gtk2_kfold_basis() {
    const std::vector<Pauli> B = kfold_terms();
    TermResidual tr(4, B);
    CHECK_EQ(tr.num_terms(), 8);
    CHECK_EQ(tr.kernel_dim(), 2);
    CHECK_EQ(tr.rank(), 6);
    // the index-order elimination reproduces kfold's basis: k1 = {0..6} (non-pivot 6), k2 = {1, 7}
    CHECK_EQ(tr.kernel()[0].j, 6);
    CHECK_EQ((long long)tr.kernel()[0].k[0], 0x7fLL);
    CHECK_EQ(tr.kernel()[1].j, 7);
    CHECK_EQ((long long)tr.kernel()[1].k[0], 0x82LL);
    // N_k = rot(k) as a Pauli, against the dense rot(k)
    for (const auto& kv : tr.kernel()) CHECK(eq_phase(rotS_dense(4, B, kv.k), pauli_dense(4, kv.N)));
    TermShot sh;
    tr.init_shot(sh);
    int seq_ok = 0, red_ok = 0, red_canon = 0;
    for (unsigned m = 0; m < 256; ++m) {
        const std::vector<uint64_t> S = set_of(8, m);
        const Dense truth = rotS_dense(4, B, S);
        // (a) the unconditional two-vector sequence k1 then k2 (kfold's S2 = S ^ k1 ^ k2)
        tr.clear_shot(sh);
        sh.S = S;
        tr.fold_sequence_({0, 1}, sh);
        if (sh.S[0] == (S[0] ^ 0x7fULL ^ 0x82ULL) &&
            eq_phase(truth, mul(pauli_dense(4, sh.P), rotS_dense(4, B, sh.S))))
            ++seq_ok;
        // (b) reduce_key_: exact, and pivot-supported
        tr.clear_shot(sh);
        sh.S = S;
        tr.reduce_key_(sh);
        if (eq_phase(truth, mul(pauli_dense(4, sh.P), rotS_dense(4, B, sh.S)))) ++red_ok;
        if ((sh.S[0] & ((1ULL << 6) | (1ULL << 7))) == 0) ++red_canon;
    }
    std::fprintf(stderr, "  G-TK2 kfold: sequential two-vector %d/256, reduce_key_ %d/256 (pivot-supported %d/256)\n",
                 seq_ok, red_ok, red_canon);
    CHECK_EQ(seq_ok, 256);
    CHECK_EQ(red_ok, 256);
    CHECK_EQ(red_canon, 256);
}

void test_gtk2_mirror_pair() {
    // B1 = B2 = Z: k = {0, 1}, N_k = rot(Z)^2 = iZ. S = {1} reduces to {0} with NO Pauli: rot(B2) = rot(B1).
    std::vector<Pauli> B = {zp(1, {0}), zp(1, {0})};
    TermResidual tr(1, B);
    CHECK_EQ(tr.kernel_dim(), 1);
    CHECK(tr.kernel()[0].N.zbit(0) && !tr.kernel()[0].N.xbit(0));
    TermShot sh;
    tr.init_shot(sh);
    tr.clear_shot(sh);
    sh.S = set_of(2, 0x2);
    tr.reduce_key_(sh);
    CHECK_EQ((long long)sh.S[0], 0x1LL);
    CHECK(!sh.P.zbit(0) && !sh.P.xbit(0));             // a constant fold would leave a silent Z here
    CHECK(eq_phase(rotS_dense(1, B, set_of(2, 0x2)), mul(pauli_dense(1, sh.P), rotS_dense(1, B, sh.S))));
}

void test_gtk6_commutation_assert() {
    Pauli X(1), Z(1);
    X.setx(0);
    Z.setz(0);
    bool threw = false;
    std::string msg;
    try {
        TermResidual tr(1, {X, Z});
    } catch (const TermClassRefusal& e) {
        threw = true;
        msg = e.what();
    }
    CHECK(threw);
    CHECK(msg.find("TERM CLASS REFUSAL") != std::string::npos);
    // a deferred circuit is checked the same way at construction: T 0 / H 0 / T 0 (terms Z and X)
    ParsedStim ps = parse_stim_circuit("T 0\nH 0\nT 0\nM 0\n");
    CHECK(ps.ok());
    threw = false;
    try {
        TermResidual tr(ps.circuit, {{2, 0}});
    } catch (const TermClassRefusal&) {
        threw = true;
    }
    CHECK(threw);
}

// A random commuting term set on n qubits: Z-strings conjugated by a random Clifford word.
std::vector<Pauli> random_commuting(int n, int T, std::mt19937_64& rng) {
    std::vector<Pauli> B;
    for (int j = 0; j < T; ++j) {
        Pauli P(n);
        for (int q = 0; q < n; ++q) if (rng() & 1) P.setz(q);
        if (rng() & 1) P.phase = 2;
        B.push_back(P);
    }
    for (int g = 0; g < 12; ++g) {
        const int k = (int)(rng() % 3), a = (int)(rng() % n);
        int b = (int)(rng() % n);
        if (b == a) b = (a + 1) % n;
        for (Pauli& P : B) {
            if (k == 0) CliffordTableau::conj_h(P, a);
            else if (k == 1) CliffordTableau::conj_s(P, a);
            else CliffordTableau::conj_cx(P, a, b);
        }
    }
    return B;
}

void test_compose_dense() {
    std::mt19937_64 rng(7);
    int checked = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const int n = 2 + (int)(rng() % 3), T = 2 + (int)(rng() % 5);
        const std::vector<Pauli> B = random_commuting(n, T, rng);
        TermResidual tr(n, B);
        TermShot sh;
        tr.init_shot(sh);
        tr.clear_shot(sh);
        Dense truth = eye(n);
        for (int f = 0; f < 4; ++f) {                   // R <- R_f R
            Pauli Pf(n);
            for (int q = 0; q < n; ++q) { if (rng() & 1) Pf.setx(q); if (rng() & 1) Pf.setz(q); }
            const std::vector<uint64_t> Sf = set_of(T, (unsigned)rng() & ((1u << T) - 1));
            truth = mul(mul(pauli_dense(n, Pf), rotS_dense(n, B, Sf)), truth);
            tr.compose_atom_(Pf.x.data(), Pf.z.data(), Sf.data(), sh);
            if (eq_phase(truth, mul(pauli_dense(n, sh.P), rotS_dense(n, B, sh.S)))) ++checked;
            else std::fprintf(stderr, "  compose mismatch trial %d atom %d\n", trial, f);
        }
        // the reduced form stays exact
        Dense before = mul(pauli_dense(n, sh.P), rotS_dense(n, B, sh.S));
        tr.reduce_key_(sh);
        CHECK(eq_phase(before, mul(pauli_dense(n, sh.P), rotS_dense(n, B, sh.S))));
    }
    CHECK_EQ(checked, 160);
}

void test_kind_exact_diagonality() {
    // B1 = X0, B2 = X0 Z1 commute; S = {0, 1}: Z0 -> Z0 Z1 (Z-type), Z1 -> Z1, but X1 -> X0 X1 Z1: NOT
    // diagonal. The exact test (Z_q -> ±Z_q) gives kind 2; "Z_q maps to a Z-type Pauli" alone would not.
    Pauli b1(2), b2(2);
    b1.setx(0);
    b2.setx(0);
    b2.setz(1);
    TermResidual tr(2, {b1, b2});
    std::vector<uint64_t> S = set_of(2, 0x3);
    TermShot sh;
    tr.init_shot(sh);
    tr.clear_shot(sh);
    sh.S = S;
    tr.reduce_key_(sh);
    Pauli z0(2);
    z0.setz(0);
    const Pauli im = tr.rot_image(sh.S, z0);
    bool ztype = true;
    for (int q = 0; q < 2; ++q) if (im.xbit(q)) ztype = false;
    CHECK(ztype);                                       // Z0's image is Z-type ...
    CHECK(im.zbit(1));                                  // ... but not ±Z0
    CHECK_EQ(tr.kind_of_(sh.S), 2);
    // Z-type terms: diagonal, kind 0
    TermResidual tz(2, {zp(2, {0}), zp(2, {0, 1})});
    CHECK_EQ(tz.kind_of_(set_of(2, 0x3)), 0);
    CHECK_EQ(tz.kind_of_(set_of(2, 0x0)), 0);
}

void test_term_view_and_zero_alloc() {
    // H 0 / T 0 / H 0 (term X0) ; noise ; T 1 (term Z1) ; noise ; CX 1 0 ; reads
    ParsedStim ps = parse_stim_circuit(
        "H 0\nT 0\nH 0\nDEPOLARIZE1(0.1) 0 1\nT 1\nX_ERROR(0.1) 1\nCX 1 0\nM 0 1\n");
    CHECK(ps.ok());
    TermResidual tr(ps.circuit, {{2, 0}, {2, 1}});
    CHECK_EQ(tr.num_terms(), 2);
    // B_0 = H Z H = X0 (CX 1 0 leaves it), B_1 = Z1 (CX 1 0 leaves it)
    CHECK(tr.axis(0).xbit(0) && !tr.axis(0).zbit(0) && !tr.axis(0).xbit(1) && !tr.axis(0).zbit(1));
    CHECK(tr.axis(1).zbit(1) && !tr.axis(1).xbit(1) && !tr.axis(1).xbit(0) && !tr.axis(1).zbit(0));
    // term_rows: the Z read of wire 0 anticommutes with B_0; nothing with B_1
    CHECK_EQ((int)tr.term_rows(0).size(), 1);
    CHECK_EQ((int)tr.term_rows(1).size(), 0);
    std::vector<std::vector<FiredPauli>> sides = {
        {{3, 0, PauliBasis::Z}}, {{3, 0, PauliBasis::X}}, {{3, 1, PauliBasis::X}}, {{5, 1, PauliBasis::X}}};
    tr.set_alts(sides);
    CHECK(tr.alts_ready());
    // alt 0: Z0 at the noise (after T 0's factor) commutes with T 1's Z1: S = {}
    CHECK_EQ((long long)tr.alt_s(0)[0], 0LL);
    // alt 2: X1 crosses T 1 (anticommutes with Z1): S = {1}, P_f = X1 -CX 1 0-> X0 X1; alt 3 is after it: S = {}
    CHECK_EQ((long long)tr.alt_s(2)[0], 2LL);
    CHECK_EQ((long long)tr.alt_px(2)[0], 3LL);
    CHECK_EQ((long long)tr.alt_s(3)[0], 0LL);
    TermShot sh;
    tr.init_shot(sh);
    const int32_t alts[4] = {0, 1, 2, 3};
    tr.compose_terms_(alts, 4, sh);                     // warm
    const long before = g_allocs.load();
    for (int rep = 0; rep < 1000; ++rep) {
        tr.compose_terms_(alts, 4, sh);
        tr.compose_terms_(alts + 1, 2, sh);
    }
    CHECK_EQ(g_allocs.load() - before, 0L);
}

}  // namespace

int main() {
    RUN(test_gtk2_kfold_basis);
    RUN(test_gtk2_mirror_pair);
    RUN(test_gtk6_commutation_assert);
    RUN(test_compose_dense);
    RUN(test_kind_exact_diagonality);
    RUN(test_term_view_and_zero_alloc);
    REPORT();
}
