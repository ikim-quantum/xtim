// test_kernel_dual_basis.cpp — the canonical dual-basis solver of build_shot_law's kernel section
// (twirl_law.cpp::kernel_dual_basis, reached through the kernel_dual_basis_for_test seam).
//
// Contract under test (derivation note 2026-10-12, Thm. C, owner-signed): given preimages c_1..c_κ
// that are independent as functionals on W = C_det^⊥, and a coin space inside W ∩ (∩_j c_j^⊥),
// the solver returns fold masks δ_1..δ_κ with
//   (W)   δ_k ∈ W                       (⊥ every C_det row),
//   (P)   c_j·δ_k = δ_jk                (dual basis),
//   (R)   rank(coins ∪ δ) = rank(coins) + κ   (Cor. D),
//   (G)   δ_k reduced mod the coin RREF (no coin pivot bit set) — the canonical representative,
//   (C)   δ depends on c_j only through the class c_j ⊕ C_det and on C_det only through its
//         rowspace (replacing a preimage by c_j ⊕ d, d ∈ C_det, or the C_det rows by another basis
//         of the same rowspace, returns identical bytes),
// and throws std::logic_error when the preimages are dependent on W (Lemma B says the production
// call site never is; the throw is the engine-bug assert).
// Random systems: ng ≤ 128 (several words), κ ≤ 4, r ≤ 12; all GF(2) helpers below are the
// test's own (oracle), not the engine's.
#include "check.hpp"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/twirl_law_internal.hpp"   // test seam (internal header)

using BV = std::vector<uint64_t>;

namespace {

int nw(int nbits) { return (nbits + 63) / 64; }
bool bit(const BV& v, int i) { return (v[(size_t)(i >> 6)] >> (i & 63)) & 1ULL; }
void setbit(BV& v, int i) { v[(size_t)(i >> 6)] |= 1ULL << (i & 63); }
void bxor(BV& a, const BV& b) { for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i]; }
bool any(const BV& v) { for (uint64_t w : v) if (w) return true; return false; }
int dot(const BV& a, const BV& b) {
    int acc = 0;
    for (size_t i = 0; i < a.size(); ++i) acc += __builtin_popcountll(a[i] & b[i]);
    return acc & 1;
}
int first(const BV& v) {
    for (size_t w = 0; w < v.size(); ++w) if (v[w]) return (int)w * 64 + __builtin_ctzll(v[w]);
    return -1;
}
BV randbv(std::mt19937_64& g, int ng) {
    BV v((size_t)nw(ng), 0);
    for (int i = 0; i < ng; ++i) if (g() & 1) setbit(v, i);
    return v;
}
// Oracle RREF (reduced): returns rows + pivot columns.
struct Rr { std::vector<BV> R; std::vector<int> piv; };
Rr rref(const std::vector<BV>& rows) {
    Rr o;
    for (const BV& r : rows) {
        BV v = r;
        for (size_t k = 0; k < o.R.size(); ++k) if (bit(v, o.piv[k])) bxor(v, o.R[k]);
        const int pc = first(v);
        if (pc < 0) continue;
        for (size_t k = 0; k < o.R.size(); ++k) if (bit(o.R[k], pc)) bxor(o.R[k], v);
        o.R.push_back(v);
        o.piv.push_back(pc);
    }
    return o;
}
int rank(const std::vector<BV>& rows) { return (int)rref(rows).R.size(); }
std::vector<BV> nullspace(const std::vector<BV>& rows, int n) {
    Rr b = rref(rows);
    std::vector<char> isp((size_t)n, 0);
    for (int p : b.piv) isp[(size_t)p] = 1;
    std::vector<BV> out;
    for (int f = 0; f < n; ++f) {
        if (isp[(size_t)f]) continue;
        BV x((size_t)nw(n), 0);
        setbit(x, f);
        for (size_t k = 0; k < b.R.size(); ++k) if (bit(b.R[k], f)) setbit(x, b.piv[k]);
        out.push_back(x);
    }
    return out;
}
bool in_span(const BV& v, const Rr& b) {
    BV x = v;
    for (size_t k = 0; k < b.R.size(); ++k) if (bit(x, b.piv[k])) bxor(x, b.R[k]);
    return !any(x);
}
BV combo(std::mt19937_64& g, const std::vector<BV>& basis, int ng) {
    BV v((size_t)nw(ng), 0);
    for (const BV& b : basis) if (g() & 1) bxor(v, b);
    return v;
}

struct Inst {
    int ng = 0, kappa = 0;
    std::vector<BV> cdet, pre, coins;
};

// A random valid instance: C_det a random subspace, preimages independent on W, coins inside
// (C_det ⊕ span pre)^⊥ (the structural fact Lemma A supplies at the production call site).
Inst random_instance(std::mt19937_64& g, int ng, int kappa, int r) {
    Inst I;
    I.ng = ng; I.kappa = kappa;
    for (;;) {
        I.cdet.clear(); I.pre.clear(); I.coins.clear();
        const int dimC = std::max(0, ng - kappa - r - (int)(g() % 3));
        for (int i = 0; i < dimC; ++i) I.cdet.push_back(randbv(g, ng));
        const std::vector<BV> W = nullspace(I.cdet, ng);
        if ((int)W.size() < kappa + r) continue;
        // restricted functionals A_j (bit m = c_j·w_m) must have rank κ
        for (int j = 0; j < kappa; ++j) I.pre.push_back(randbv(g, ng));
        std::vector<BV> A;
        for (const BV& c : I.pre) {
            BV row((size_t)nw((int)W.size()), 0);
            for (size_t m = 0; m < W.size(); ++m) if (dot(c, W[m])) setbit(row, (int)m);
            A.push_back(row);
        }
        if (rank(A) != kappa) continue;
        std::vector<BV> cp = I.cdet;
        for (const BV& c : I.pre) cp.push_back(c);
        const std::vector<BV> cspace = nullspace(cp, ng);   // = (C_det ⊕ span pre)^⊥
        if ((int)cspace.size() < r) continue;
        for (int k = 0; k < r; ++k) I.coins.push_back(combo(g, cspace, ng));
        return I;
    }
}

void check_contract(const Inst& I, const std::vector<BV>& delta, const char* tag) {
    const int K = I.kappa;
    CHECK_EQ((int)delta.size(), K);
    if ((int)delta.size() != K) { std::fprintf(stderr, "  [%s] size\n", tag); return; }
    const Rr coin = rref(I.coins);
    for (int k = 0; k < K; ++k) {
        for (const BV& d : I.cdet) CHECK(dot(delta[k], d) == 0);                      // (W)
        for (int j = 0; j < K; ++j) CHECK(dot(I.pre[j], delta[k]) == (j == k ? 1 : 0)); // (P)
        for (int p : coin.piv) CHECK(!bit(delta[k], p));                              // (G)
        CHECK(any(delta[k]));
    }
    std::vector<BV> all = coin.R;
    for (const BV& d : delta) all.push_back(d);
    CHECK_EQ(rank(all), (int)coin.R.size() + K);                                      // (R)
}

void random_systems() {
    std::mt19937_64 g(0x5EED2026101200ull);
    int n_inst = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int ng = 2 + (int)(g() % 127);          // 2..128 (straddles word boundaries)
        const int kappa = 1 + (int)(g() % 4);
        const int r = (int)(g() % 13);
        if (kappa + r > ng) continue;
        const Inst I = random_instance(g, ng, kappa, r);
        const std::vector<BV> delta = qeccore::twirl_law_detail::kernel_dual_basis_for_test(I.pre, I.cdet, I.coins, ng);
        check_contract(I, delta, "random");
        // (C) class invariance: shift every preimage by a C_det element, same bytes.
        if (!I.cdet.empty()) {
            Inst J = I;
            for (BV& c : J.pre) bxor(c, combo(g, I.cdet, ng));
            const std::vector<BV> d2 = qeccore::twirl_law_detail::kernel_dual_basis_for_test(J.pre, J.cdet, J.coins, ng);
            CHECK(d2 == delta);
            // another basis of the same C_det rowspace (random invertible recombination)
            std::vector<BV> cd2 = I.cdet;
            for (size_t i = 0; i < cd2.size(); ++i)
                for (size_t l = 0; l < cd2.size(); ++l)
                    if (l != i && (g() & 1)) bxor(cd2[i], I.cdet[l]);
            if (rank(cd2) == rank(I.cdet)) {
                const std::vector<BV> d3 = qeccore::twirl_law_detail::kernel_dual_basis_for_test(I.pre, cd2, I.coins, ng);
                CHECK(d3 == delta);
            }
        }
        // coin gauge: with NO coins the solution differs from the reduced one by coin vectors only
        // (Thm. C: the solution set of each δ_k is a coset of span(coins)).
        {
            const std::vector<BV> d0 = qeccore::twirl_law_detail::kernel_dual_basis_for_test(I.pre, I.cdet, {}, ng);
            const Rr coin = rref(I.coins);
            for (int k = 0; k < I.kappa; ++k) {
                BV diff = d0[(size_t)k];
                bxor(diff, delta[(size_t)k]);
                CHECK(in_span(diff, coin));
            }
        }
        ++n_inst;
    }
    CHECK(n_inst >= 300);
    std::fprintf(stderr, "  random systems checked: %d\n", n_inst);
}

void dependent_preimages_throw() {
    std::mt19937_64 g(7);
    int thrown = 0, total = 0;
    for (int trial = 0; trial < 50; ++trial) {
        const int ng = 8 + (int)(g() % 60);
        Inst I = random_instance(g, ng, 2 + (int)(g() % 3), (int)(g() % 5));
        // make c_2 ≡ c_1 modulo C_det (dependent on W)
        I.pre[1] = I.pre[0];
        if (!I.cdet.empty()) bxor(I.pre[1], combo(g, I.cdet, ng));
        ++total;
        try {
            (void)qeccore::twirl_law_detail::kernel_dual_basis_for_test(I.pre, I.cdet, I.coins, ng);
        } catch (const std::logic_error&) { ++thrown; }
    }
    CHECK_EQ(thrown, total);
}

// Prop. F instance (note §2.3): two generators, C_det = 0, preimages c(Z0Z1) = {g2} = (0,1),
// c(Z4Z5) = {g1,g2} = (1,1): |c_2| even, so no preimage can be its own fold; the dual basis is
// δ_1 = (1,1), δ_2 = (1,0) — pinned exactly. NOTE: the literal C_det_full = 0 instance is pinned
// at the SOLVER level only — it is not realisable through the engine's pipeline (every deferred
// ancilla wire contributes an inactive singleton to C_det_full); the engine-level fixtures of
// tests/test_twirl_kappa_dual.py cover the active-C_det = 0 analogue.
void prop_f_instance() {
    const int ng = 2;
    BV c1(1, 0), c2(1, 0);
    setbit(c1, 1);
    setbit(c2, 0); setbit(c2, 1);
    const std::vector<BV> delta = qeccore::twirl_law_detail::kernel_dual_basis_for_test({c1, c2}, {}, {}, ng);
    CHECK_EQ((int)delta.size(), 2);
    CHECK_EQ((long long)delta[0][0], 3LL);
    CHECK_EQ((long long)delta[1][0], 1LL);
    // the legacy pick δ := c fails (P): c_2·c_2 = 0, c_1·c_2 = 1
    CHECK_EQ(dot(c2, c2), 0);
    CHECK_EQ(dot(c1, c2), 1);
}

// κ=1 with a coin: the solution is reduced mod the coin — adding the coin would be a valid
// but non-canonical fold.
void kappa1_gauge() {
    const int ng = 3;
    BV c(1, 0); setbit(c, 0);                       // c = (1,0,0)
    BV coin(1, 0); setbit(coin, 1); setbit(coin, 2);   // coin = (0,1,1) ⊥ c
    const std::vector<BV> delta = qeccore::twirl_law_detail::kernel_dual_basis_for_test({c}, {}, {coin}, ng);
    CHECK_EQ((int)delta.size(), 1);
    CHECK_EQ(dot(c, delta[0]), 1);
    CHECK(!bit(delta[0], 1));                       // coin pivot (column 1) cleared
}

}  // namespace

int main() {
    RUN(random_systems);
    RUN(dependent_preimages_throw);
    RUN(prop_f_instance);
    RUN(kappa1_gauge);
    REPORT();
}
