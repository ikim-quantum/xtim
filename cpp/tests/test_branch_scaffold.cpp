// J-a N7 — the branch scaffolding (qeccore/branch_scaffold.hpp) against independent oracles:
//   * G-ES: ensure_es_ == the Lemma R2′ definition {a : ∃ j ∈ S, ⟨g_a, B_j⟩ = 1} computed here by a direct loop,
//     on random commuting term sets and random certified groups; the certified rows MOVED by Φ(S) (C0† g C0 ≠ ±g,
//     DENSE 2^n matrices) are a subset, reported. The r2.6 fixture es_four_terms: B = {Z1Z2, Z3Z4, Z1Z3, Z2Z4},
//     g = X1X4: g ∈ E_S (it anticommutes with all four), C0† g C0 = g EXACTLY (dense), |moved| < |E_S|.
//     Mutation XTIM_MUTATE_ES_SKIP (one j ∈ S skipped) → red.
//   * G-W (unit half): W_p (g_p on the LEFT, −iB_j) == C0 g_p C0† EXACTLY (dense, phase included), and g_p is
//     dual to the pivot pullbacks (⟨Q_p', g_p⟩ = δ_pp'). Mutation XTIM_MUTATE_W_SIGN (+iB_j) → red. The state half
//     (ψ_u = P·W(u)·ψ_{u=0} on retained shots) is tests/test_ja_branch.py.
//   * G-DIG: distinct keys, distinct digests (no collision over 200k keys); insert / resolve by FULL key; a forced
//     digest collision is the named refusal DigestCollision. Mutation XTIM_MUTATE_DIG_NOCMP (resolve compares the
//     digest only) → the forced collision silently resolves to the other key: red.
//   * G-LAZY (unit half): constructing a scaffold computes no E_S (counter 0); the first ensure_es_ computes it
//     once, later calls are cached. Mutation XTIM_MUTATE_ES_EAGER (E_S in the payload build) → red.
#include <cmath>
#include <complex>
#include <random>
#include <unordered_map>
#include <string>
#include <vector>

#include "check.hpp"
#include "qeccore/branch_scaffold.hpp"
#include "qeccore/twirl_planes.hpp"

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
bool eq_up_to_sign(const Dense& A, const Dense& B) {
    Dense nB = B;
    for (auto& v : nB.m) v = -v;
    return eq_exact(A, B) || eq_exact(A, nB);
}
int ov(const Pauli& P) {
    int c = 0;
    for (int q = 0; q < P.n; ++q) c += (P.xbit(q) && P.zbit(q)) ? 1 : 0;
    return c;
}
Pauli hermitian(int n, const std::string& s) {     // e.g. "XIIX", "+-" sign prefix optional
    Pauli P(n);
    int sign = 0;
    size_t o = 0;
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) { sign = s[0] == '-'; o = 1; }
    for (int q = 0; q < n; ++q) {
        const char c = s[o + (size_t)q];
        if (c == 'X' || c == 'Y') P.setx(q);
        if (c == 'Z' || c == 'Y') P.setz(q);
    }
    P.phase = (ov(P) + 2 * sign) & 3;
    return P;
}
Pauli random_hermitian(std::mt19937_64& rng, int n, bool ztype) {
    Pauli P(n);
    for (int q = 0; q < n; ++q) {
        const int c = (int)(rng() % 4);
        if (!ztype && (c == 1 || c == 2)) P.setx(q);
        if (c == 2 || c == 3) P.setz(q);
    }
    P.phase = (ov(P) + 2 * (int)(rng() % 2)) & 3;
    return P;
}
bool nontrivial(const Pauli& P) {
    for (int q = 0; q < P.n; ++q) if (P.xbit(q) || P.zbit(q)) return true;
    return false;
}
bool bit(const std::vector<uint64_t>& v, int i) { return (size_t)(i >> 6) < v.size() && ((v[(size_t)(i >> 6)] >> (i & 63)) & 1); }
Dense c0_dense(int n, const PlanTermView& v) {
    Dense C0 = eye(n);
    for (int j = 0; j < (int)v.axes.size(); ++j) if (bit(v.S, j)) C0 = mul(C0, rot_dense(n, v.axes[(size_t)j]));
    return C0;
}
// A random commuting set of Hermitian axes and a random commuting certified group (independent generators).
void random_case(std::mt19937_64& rng, int n, std::vector<Pauli>& axes, std::vector<Pauli>& gens) {
    axes.clear();
    gens.clear();
    const bool ztype = rng() % 2;
    const int want = 1 + (int)(rng() % 5);
    for (int t = 0; t < 60 && (int)axes.size() < want; ++t) {
        Pauli c = random_hermitian(rng, n, ztype);
        bool ok = nontrivial(c);
        for (const Pauli& b : axes) ok &= !Pauli::anticommute_bit(b, c);
        if (ok) axes.push_back(c);
    }
    const int wantg = 1 + (int)(rng() % (size_t)n);
    for (int t = 0; t < 80 && (int)gens.size() < wantg; ++t) {
        Pauli g = random_hermitian(rng, n, false);
        g.phase = (uint8_t)(ov(g) & 3);             // +1 stabilizers
        bool ok = nontrivial(g);
        for (const Pauli& h : gens) ok &= !Pauli::anticommute_bit(h, g);
        if (!ok) continue;
        std::vector<Pauli> trial = gens;
        trial.push_back(g);
        if (CertifiedGroupPlanes::build(trial, n).rank == (int)trial.size()) gens.push_back(g);
    }
}

// ── G-ES ─────────────────────────────────────────────────────────────────────────────────────────
void test_es_four_terms() {
    const int n = 4;
    PlanTermView v;
    v.kind = 2;
    v.key = "es_four_terms";
    v.axes = {hermitian(n, "ZZII"), hermitian(n, "IIZZ"), hermitian(n, "ZIZI"), hermitian(n, "IZIZ")};
    v.S = {0xF};
    const std::vector<Pauli> gens = {hermitian(n, "XIIX"), hermitian(n, "ZZZZ")};
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(gens, n);
    long cnt = 0;
    PlanScaffold sc(v, G, &cnt);
    const std::vector<uint64_t> es = sc.ensure_es_();
    CHECK(bit(es, 0));                               // X1X4 anticommutes with every B_j
    CHECK(!bit(es, 1));                              // ZZZZ commutes with every B_j
    const Dense C0 = c0_dense(n, v);
    const Dense g = pauli_dense(n, gens[0]);
    CHECK(eq_exact(mul(dag(C0), mul(g, C0)), g));    // C0† g C0 = g exactly: NOT moved
    const std::vector<uint64_t> mv = sc.moved_rows_();
    CHECK(!bit(mv, 0) && !bit(mv, 1));
    std::fprintf(stderr, "[branch_scaffold] es_four_terms: |E_S| = 1, |moved| = 0 (C0^dag g C0 = g, dense)\n");
}

void test_es_random_vs_definition_and_moved_subset() {
    std::mt19937_64 rng(20261018);
    long cases = 0, es_rows = 0, moved_rows = 0, strict = 0;
    for (int trial = 0; trial < 800; ++trial) {
        const int n = 1 + (int)(rng() % 4);
        PlanTermView v;
        std::vector<Pauli> gens;
        random_case(rng, n, v.axes, gens);
        if (v.axes.empty() || gens.empty()) continue;
        v.kind = 2;
        v.key = "r" + std::to_string(trial);
        v.S = {rng() & ((1ULL << v.axes.size()) - 1)};
        const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(gens, n);
        long cnt = 0;
        PlanScaffold sc(v, G, &cnt);
        const std::vector<uint64_t> es = sc.ensure_es_();
        const std::vector<uint64_t> mv = sc.moved_rows_();
        const Dense C0 = c0_dense(n, v);
        for (int a = 0; a < (int)G.gens.size(); ++a) {
            bool want = false;                      // the definition, directly
            for (int j = 0; j < (int)v.axes.size(); ++j)
                want |= bit(v.S, j) && Pauli::anticommute_bit(G.gens[(size_t)a], v.axes[(size_t)j]);
            CHECK_EQ(bit(es, a), want);
            const Dense g = pauli_dense(n, G.gens[(size_t)a]);
            const bool moved = !eq_up_to_sign(mul(dag(C0), mul(g, C0)), g);
            CHECK_EQ(bit(mv, a), moved);
            CHECK(!moved || want);                   // moved ⊆ E_S
            es_rows += want;
            moved_rows += moved;
            strict += want && !moved;
        }
        ++cases;
    }
    std::fprintf(stderr, "[branch_scaffold] G-ES random: %ld cases, |E_S| rows %ld, moved %ld (E_S but not moved: %ld)\n",
                 cases, es_rows, moved_rows, strict);
    CHECK(cases > 500);
    CHECK(strict > 0);
}

// ── G-W unit half ────────────────────────────────────────────────────────────────────────────────
void test_pivot_frames_dense() {
    std::mt19937_64 rng(7);
    long cases = 0, odd = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const int n = 2 + (int)(rng() % 3);
        PlanTermView v;
        std::vector<Pauli> gens;
        random_case(rng, n, v.axes, gens);
        if (v.axes.empty() || gens.size() < 2) continue;
        v.kind = 2;
        v.key = "w" + std::to_string(trial);
        v.S = {rng() & ((1ULL << v.axes.size()) - 1)};
        const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(gens, n);
        // pivot pullbacks: random Paulis whose planes over G are independent
        std::vector<Pauli> Q;
        std::vector<std::vector<uint64_t>> planes;
        for (int t = 0; t < 30 && Q.size() < 2; ++t) {
            const Pauli q = random_hermitian(rng, n, false);
            std::vector<uint64_t> ph(1, 0);
            for (int a = 0; a < (int)G.gens.size(); ++a)
                if (Pauli::anticommute_bit(q, G.gens[(size_t)a])) ph[0] |= 1ULL << a;
            if (!ph[0]) continue;
            bool dep = false;                       // two nonzero planes are dependent iff equal
            for (const auto& p : planes) dep |= p[0] == ph[0];
            if (dep) continue;
            Q.push_back(q);
            planes.push_back(ph);
        }
        if (Q.empty()) continue;
        PlanScaffold sc(v, G);
        const std::vector<Pauli> W = sc.pivot_frames_(Q);
        CHECK_EQ(W.size(), Q.size());
        const Dense C0 = c0_dense(n, v);
        for (size_t p = 0; p < W.size(); ++p) {
            // recover g_p = C0† W_p C0 and check duality, then W_p == C0 g_p C0† exactly
            const Dense Wd = pauli_dense(n, W[p]);
            const Dense gd = mul(dag(C0), mul(Wd, C0));
            bool found = false;
            for (uint64_t c = 1; c < (1ULL << G.gens.size()); ++c) {
                Pauli g(n);
                for (int a = 0; a < (int)G.gens.size(); ++a) if ((c >> a) & 1) g = Pauli::multiply(g, G.gens[(size_t)a]);
                if (!eq_exact(pauli_dense(n, g), gd)) continue;
                found = true;
                for (size_t p2 = 0; p2 < Q.size(); ++p2)
                    CHECK_EQ(Pauli::anticommute_bit(Q[p2], g), p2 == p);
                int na = 0;
                for (int j = 0; j < (int)v.axes.size(); ++j)
                    na += bit(v.S, j) && Pauli::anticommute_bit(g, v.axes[(size_t)j]);
                odd += na & 1;
            }
            CHECK(found);                            // W_p = C0 g_p C0† for a group element g_p (exact phase)
            ++cases;
        }
    }
    std::fprintf(stderr, "[branch_scaffold] G-W dense: %ld pivot frames (odd dressing count: %ld)\n", cases, odd);
    CHECK(cases > 200);
    CHECK(odd > 50);
}

// ── G-DIG ────────────────────────────────────────────────────────────────────────────────────────
void test_digest_and_resolve() {
    std::unordered_map<uint64_t, std::string> seen_lo;
    long coll64 = 0;
    std::vector<Digest128> ds;
    for (int i = 0; i < 200000; ++i) {
        std::string k = "key" + std::to_string(i);
        if (i % 3 == 0) k.push_back('\0');
        const Digest128 d = plan_key_digest128_(k);
        auto it = seen_lo.find(d.lo);
        if (it != seen_lo.end() && it->second != k) ++coll64;
        seen_lo[d.lo] = k;
    }
    CHECK_EQ(coll64, 0L);
    CHECK(plan_key_digest128_("abc") == plan_key_digest128_("abc"));
    CHECK(plan_key_digest128_("abc") != plan_key_digest128_("abd"));
    CHECK(plan_key_digest128_(std::string("a\0b", 3)) != plan_key_digest128_(std::string("a\0c", 3)));
    PlanTableSet T;
    const std::vector<int> none;
    const long a = T.insert(none, "plan A"), b = T.insert(none, "plan B");
    CHECK(a != b);
    CHECK_EQ(T.insert(none, "plan A"), a);
    CHECK_EQ(T.resolve(none, plan_key_digest128_("plan A"), "plan A"), a);
    CHECK_EQ(T.resolve(none, plan_key_digest128_("plan B"), "plan B"), b);
    CHECK_EQ(T.resolve(none, plan_key_digest128_("plan C"), "plan C"), -1L);
    const long e = T.insert({3, 7}, "plan A");      // another E_min: another table
    CHECK(e != a);
    CHECK_EQ(T.num_tables(), 2);
    // a forced collision: two keys, one digest -> named refusal, never a merge
    PlanTableSet F;
    F.force_digest_for_test(Digest128{1, 2});
    const long x = F.insert(none, "first");
    int refused = 0;
    try { (void)F.insert(none, "second"); } catch (const DigestCollision& ex) {
        ++refused;
        CHECK(std::string(ex.what()).find("digest collision") != std::string::npos);
    }
    try {
        const long r = F.resolve(none, Digest128{1, 2}, "second");
        CHECK(r != x);                               // must never resolve to the other key's plan
    } catch (const DigestCollision&) { ++refused; }
    CHECK_EQ(F.resolve(none, Digest128{1, 2}, "first"), x);
    CHECK_EQ(refused, 2);
    std::fprintf(stderr, "[branch_scaffold] G-DIG: 200000 keys, 0 64-bit collisions; forced collision refused %d/2\n", refused);
}

// ── G-LAZY unit half ─────────────────────────────────────────────────────────────────────────────
void test_lazy_es() {
    const int n = 4;
    PlanTermView v;
    v.axes = {hermitian(n, "ZZII"), hermitian(n, "IIZZ")};
    v.S = {0x3};
    v.key = "lazy";
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build({hermitian(n, "XXII"), hermitian(n, "IXXI")}, n);
    long cnt = 0;
    std::vector<PlanScaffold> plans;
    for (int i = 0; i < 10; ++i) plans.emplace_back(v, G, &cnt);
    CHECK_EQ(cnt, 0L);                               // the payload builds computed no E_S
    for (int i = 0; i < 10; ++i) CHECK(!plans[(size_t)i].es_built());
    (void)plans[3].ensure_es_();
    (void)plans[3].ensure_es_();
    CHECK_EQ(cnt, 1L);                               // once, then cached
    CHECK(plans[3].es_built());
}

}  // namespace

int main() {
    RUN(test_es_four_terms);
    RUN(test_es_random_vs_definition_and_moved_subset);
    RUN(test_pivot_frames_dense);
    RUN(test_digest_and_resolve);
    RUN(test_lazy_es);
    REPORT();
}
