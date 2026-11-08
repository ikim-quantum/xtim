// test_born_trie.cpp — S2a: the per-plan Born trie on general framed states.
//
// Random framed states (random Clifford frame U, random reference signs eps, k free coordinates
// with NON-uniform random amplitudes on a random support — the conditionals are non-trivial, unlike
// the in-class corpus where every kernel direction is a fair coin) and random commuting Born
// directions (frame Paulis with X/Z on the free coordinates and Z on non-free ones; a single-
// coordinate Z first so projections make coordinates constant).
// Oracles:
//   (D) an INDEPENDENT dense statevector on the 2^k free register (the test's own Pauli algebra:
//       (Λψ)_x = i^ph·(−1)^{b·eps}·(−1)^{b·(x⊕a)}·ψ_{x⊕a}, from the test's own frame bits — shares no
//       code with the engine): every node == the sequential Born conditional within 1e-12;
//   (C) the deleted κ chain's semantics on the n-qubit state (measure_pauli / born_p1 of the n-qubit
//       directions), which COMPACTS a coordinate made constant and then takes its case (a) — a fair
//       coin plus a frame rewrite — when a later direction has X on it. Secondary: bitwise on trials
//       where the chain never takes case (a); every chain case-(a) node of the trie is p0 == 0.5
//       BITWISE (the trie never compacts: the post-prune support and its shift are disjoint, the
//       overlap is an empty sum).
// Checked: (1) the free_dir_of / derive_kernel_logical round trip == G.reduce(Λ).rep (exact phase);
// (2) walk bits == the chain's outcomes on case-(b) trials, exactly κ steps per walk; (3) nodes vs
// (D), (C), and BornTrie::p0_from_scratch (bitwise); (4) nodes == distinct prefixes, a second pass
// evaluates none; (5) branch != 0 refused; (6) counters: case-(a) nodes counted, support reached.
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

using namespace qeccore;

namespace {

bool same(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Frame Pauli i^ph X^a Z^b (all n coordinates) mapped through U: i^ph ∏ Xrow^a ∏ Zrow^b.
Pauli image(const FramedSuperposition& f, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
            int ph) {
    Pauli P(f.n());
    P.phase = ph & 3;
    for (int q = 0; q < f.n(); ++q) if (a[(size_t)q]) P = Pauli::multiply(P, f.U.Xrow[(size_t)q]);
    for (int q = 0; q < f.n(); ++q) if (b[(size_t)q]) P = Pauli::multiply(P, f.U.Zrow[(size_t)q]);
    return P;
}

bool chain_case_a(const FramedSuperposition& w, const Pauli& L) {
    const Pauli Q = w.U.dual_valid() ? w.U.conjugate(L) : w.U.dual_image(L);
    std::vector<uint8_t> fr((size_t)w.n(), 0);
    for (int f : w.free) fr[(size_t)f] = 1;
    for (int q = 0; q < w.n(); ++q) if (Q.xbit(q) && !fr[(size_t)q]) return true;
    return false;
}

// (D) the dense free-register oracle.
struct Dense {
    int k = 0;
    std::vector<std::complex<double>> v;
};
struct DDir { int a = 0, b = 0; std::complex<double> c; };   // a, b as k-bit masks (free order)
std::vector<std::complex<double>> apply_dir(const Dense& s, const DDir& d) {
    std::vector<std::complex<double>> o(s.v.size());
    for (int x = 0; x < (int)s.v.size(); ++x) {
        const int src = x ^ d.a;
        const double sg = (__builtin_popcount(d.b & src) & 1) ? -1.0 : 1.0;
        o[(size_t)x] = d.c * sg * s.v[(size_t)src];
    }
    return o;
}
double dense_p(const Dense& s, const DDir& d) {
    const std::vector<std::complex<double>> w = apply_dir(s, d);
    std::complex<double> ip(0, 0);
    double n2 = 0;
    for (size_t x = 0; x < w.size(); ++x) { ip += std::conj(s.v[x]) * w[x]; n2 += std::norm(s.v[x]); }
    return 0.5 * (1.0 + ip.real() / n2);
}
void dense_project(Dense& s, const DDir& d, int eps) {
    const std::vector<std::complex<double>> w = apply_dir(s, d);
    double n2 = 0;
    for (size_t x = 0; x < w.size(); ++x) { s.v[x] += (double)eps * w[x]; n2 += std::norm(s.v[x]); }
    for (auto& z : s.v) z /= std::sqrt(n2);
}

struct Case {
    FramedSuperposition f{0};
    std::vector<Pauli> L;
    std::vector<DDir> D;          // the same directions, dense free-register form (test's own algebra)
    Dense psi0;
};

Case make_case(std::mt19937_64& rng, int n, int k, int kap) {
    std::uniform_int_distribution<int> bit(0, 1), qd(0, n - 1);
    std::normal_distribution<double> nd(0.0, 1.0);
    Case c;
    c.f = FramedSuperposition(n);
    for (int g = 0; g < 6 * n; ++g) {                       // random Clifford frame
        const int kind = std::uniform_int_distribution<int>(0, 7)(rng);
        int a = qd(rng), b = qd(rng);
        if (kind >= 6 && a == b) continue;
        c.f.apply_clifford((uint8_t)kind, a, b);
    }
    for (int q = 0; q < n; ++q) c.f.eps[(size_t)q] = (uint8_t)bit(rng);
    std::vector<int> perm(n);
    for (int q = 0; q < n; ++q) perm[(size_t)q] = q;
    std::shuffle(perm.begin(), perm.end(), rng);
    c.f.free.assign(perm.begin(), perm.begin() + k);
    auto* d = static_cast<DenseAmplitudes*>(c.f.alpha.get());
    d->k_ = k;
    for (int x = 0; x < (1 << k); ++x) {
        if (bit(rng) && x != 0) continue;                     // random support (always contains 0)
        d->b_.emplace_back((AmpKey)x, cd(nd(rng), nd(rng)));
    }
    d->normalize();
    c.psi0.k = k;
    c.psi0.v.assign((size_t)1 << k, 0.0);
    for (const auto& e : d->b_) {
        int x = 0;
        for (int i = 0; i < k; ++i) x |= (int)key_bit(e.first, i) << i;
        c.psi0.v[(size_t)x] = e.second;
    }
    // Directions: Q_0 = Z on one free coordinate (a χ-drop source); Q_j random commuting with all
    // earlier ones (X/Z on free coordinates, Z on non-free ones), Hermitian, random sign.
    std::vector<std::vector<uint8_t>> A, B;
    int tries = 0;
    while ((int)A.size() < kap && tries++ < 1000) {
        std::vector<uint8_t> a((size_t)n, 0), b((size_t)n, 0);
        if (A.empty()) {
            b[(size_t)c.f.free[0]] = 1;
        } else {
            for (int i = 0; i < k; ++i) { a[(size_t)c.f.free[(size_t)i]] = (uint8_t)bit(rng);
                                         b[(size_t)c.f.free[(size_t)i]] = (uint8_t)bit(rng); }
            if (A.size() == 1) a[(size_t)c.f.free[0]] = 0;    // keep commuting with Q_0, use its Z
        }
        for (int q = 0; q < n; ++q) {
            bool fr = false;
            for (int i = 0; i < k; ++i) fr |= c.f.free[(size_t)i] == q;
            if (!fr) b[(size_t)q] = (uint8_t)bit(rng);
        }
        bool ok = false;
        for (int q = 0; q < n; ++q) ok |= a[(size_t)q] || b[(size_t)q];
        for (size_t j = 0; j < A.size() && ok; ++j) {
            int s = 0;
            for (int q = 0; q < n; ++q) s ^= (a[(size_t)q] & B[j][(size_t)q]) ^ (b[(size_t)q] & A[j][(size_t)q]);
            ok = s == 0;
        }
        if (!ok) continue;
        int ov = 0;
        for (int q = 0; q < n; ++q) ov += a[(size_t)q] & b[(size_t)q];
        A.push_back(a);
        B.push_back(b);
        const int ph = (ov + 2 * bit(rng)) & 3;
        c.L.push_back(image(c.f, a, b, ph));
        DDir dd;
        int sg = 0;
        for (int q = 0; q < n; ++q) sg ^= b[(size_t)q] & c.f.eps[(size_t)q];
        for (int i = 0; i < k; ++i) {
            if (a[(size_t)c.f.free[(size_t)i]]) dd.a |= 1 << i;
            if (b[(size_t)c.f.free[(size_t)i]]) dd.b |= 1 << i;
        }
        static const std::complex<double> ip[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        dd.c = ip[ph] * (sg ? -1.0 : 1.0);
        c.D.push_back(dd);
    }
    return c;
}

void test_against_oracles(uint64_t seed, int trials, long* tot) {
    std::mt19937_64 rng(seed);
    for (int trial = 0; trial < trials; ++trial) {
        const int n = 5 + trial % 4, k = 3 + trial % 3, kap = 2 + trial % 3;
        Case c = make_case(rng, n, k, kap);
        const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(c.f.certified_stabilizers(), n);
        // S2b: the walker runs over an explicit (directions, trie) pair (the plan's record table)
        struct { std::vector<FreeDir> lam; std::vector<BornNode> trie; } plan;
        for (const Pauli& L : c.L) plan.lam.push_back(free_dir_of(c.f, L));
        // (1) derivation round trip
        for (size_t j = 0; j < c.L.size(); ++j) {
            const Pauli want = G.reduce(c.L[j]).rep, got = derive_kernel_logical(c.f, G, plan.lam[j]);
            CHECK(want.x == got.x && want.z == got.z && (want.phase & 3) == (got.phase & 3));
        }
        BornTrie trie(c.f);
        krng::Stream born;
        born.set_key(77 + (uint64_t)trial + 1000 * seed, krng::BORN);
        std::set<std::vector<uint8_t>> prefixes;
        // Chain replay per shot: outcomes + whether the chain took case (a) on this shot's path.
        const int shots = 400;
        bool trial_case_a = false;
        std::map<std::vector<uint8_t>, int> chain_a_node;   // prefix -> 1 iff the chain took case (a) there
        for (int s = 0; s < shots; ++s) {
            born.at((uint64_t)s);
            std::vector<uint8_t> bits;
            CHECK_EQ(trie.walk(plan.lam, plan.trie, 0, born, bits), -1);
            CHECK_EQ(bits.size(), c.L.size());
            FramedSuperposition w = c.f;
            std::vector<uint8_t> pre;
            bool path_a = false;
            for (size_t j = 0; j < c.L.size(); ++j) {
                const bool a = chain_case_a(w, c.L[j]);
                chain_a_node[pre] |= (int)a;
                path_a |= a;
                const int eps = w.measure_pauli(c.L[j], born.u((uint64_t)j));
                // (2) bit equality wherever the chain's path stayed in case (b); after a case-(a)
                // step the chain's later doubles differ in rounding only (counted, not asserted).
                if (!path_a) CHECK_EQ((int)bits[j], eps == +1 ? 0 : 1);
                else if ((int)bits[j] != (eps == +1 ? 0 : 1)) ++tot[5];
                pre.push_back(bits[j]);
            }
            trial_case_a |= path_a;
            for (size_t j = 0; j < c.L.size(); ++j)
                prefixes.insert(std::vector<uint8_t>(bits.begin(), bits.begin() + (long)j));
        }
        CHECK_EQ(trie.walks, (long)shots);
        CHECK_EQ(trie.steps, (long)shots * (long)c.L.size());
        CHECK_EQ(trie.node_evals, (long)prefixes.size());
        CHECK_EQ(trie.nodes, (long)prefixes.size());
        CHECK(trie.passes <= trie.node_evals * trie.max_depth);
        CHECK(trie.chi_j <= trie.passes * trie.max_support);
        CHECK(trie.max_support <= (1L << k));
        tot[0] += trie.node_evals;
        tot[1] += trial_case_a;
        // (3) every node vs (D) dense, vs from-scratch (bitwise), vs (C) the chain
        for (const std::vector<uint8_t>& z : prefixes) {
            uint32_t node = 0;
            for (uint8_t b : z) node = plan.trie[node].kid[b];
            const double stored = plan.trie[node].p0;
            // S2k mechanism (op counts): a from-scratch node at depth j runs j+1 passes over the
            // support reached (chi_j); each pass's key lookups go through the O(χ) index: slot reads
            // >= the entries indexed (χ_path per pass) and <= 16·χ_path (χ_this + χ_other <= 2χ_path,
            // <= 8 probes per key), table slots <= 8·χ_path + 2 — no Θ(χ²) term.
            const uint64_t pr0 = trie.idx_.probes, sl0 = trie.idx_.slots;
            const long cj0 = trie.chi_j, ps0 = trie.passes;
            CHECK(same(stored, trie.p0_from_scratch(plan.lam, z.data(), (int)z.size())));
            {
                const long dps = trie.passes - ps0, dcj = trie.chi_j - cj0;
                const long dpr = (long)(trie.idx_.probes - pr0), dsl = (long)(trie.idx_.slots - sl0);
                CHECK_EQ(dps, (long)z.size() + 1);
                CHECK(dcj <= dps * trie.max_support);
                CHECK(dpr >= dcj);
                CHECK(dpr <= 16 * dcj);
                CHECK(dsl <= 8 * dcj + 2 * dps);
                tot[6] += dpr; tot[7] += dcj;
            }
            Dense ds = c.psi0;
            for (size_t j = 0; j < z.size(); ++j) dense_project(ds, c.D[j], z[j] ? -1 : +1);
            const double pd = dense_p(ds, c.D[z.size()]);
            CHECK(std::abs(stored - pd) <= 1e-12);
            if (std::abs(stored - pd) > tot[4] * 1e-18) tot[4] = (long)(std::abs(stored - pd) * 1e18);
            const bool node_a = chain_a_node.count(z) && chain_a_node[z];
            if (node_a) {
                ++tot[2];
                CHECK(same(stored, 0.5));                     // case-(a) node: p0 == 0.5 bitwise
                CHECK(std::abs(pd - 0.5) <= 1e-12);
            }
            if (!trial_case_a) {
                FramedSuperposition w = c.f;
                for (size_t j = 0; j < z.size(); ++j)
                    (void)w.measure_pauli(c.L[j], z[j] ? std::nextafter(1.0, 0.0) : 0.0);
                CHECK(same(stored, w.born_p1(c.L[z.size()])));
            }
            if (!z.empty() && std::abs(stored - 0.5) > 1e-6) ++tot[3];
        }
        // (4) a second pass over the same shots evaluates no node (allocated on visit only)
        const long ev = trie.node_evals;
        for (int s = 0; s < shots; ++s) {
            born.at((uint64_t)s);
            std::vector<uint8_t> bits;
            (void)trie.walk(plan.lam, plan.trie, 0, born, bits);
        }
        CHECK_EQ(trie.node_evals, ev);
        // (5) carried branches arrive at J-b1
        bool threw = false;
        try { std::vector<uint8_t> bits; trie.walk(plan.lam, plan.trie, 1, born, bits); } catch (const std::logic_error&) { threw = true; }
        CHECK(threw);
    }
}

// S8-4: the root memo. Per bare, plan pairs whose root directions are L and −L (phase + 2: the same (a, b, par)):
// each root equals its from-scratch definition bit for bit (the memo key carries the phase; MUT-BM-PHASE red), a
// repeated direction is a memo hit, and the two roots are complementary (p(−L) = 1 − p(L) to rounding).
void test_root_memo() {
    std::mt19937_64 rng(4242);
    long hits = 0, nonfair = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const int n = 5 + trial % 4, k = 3 + trial % 3;
        Case c = make_case(rng, n, k, 1);
        if (c.L.empty()) continue;
        Pauli neg = c.L[0];
        neg.phase = (neg.phase + 2) & 3;
        std::vector<FreeDir> l1{free_dir_of(c.f, c.L[0])}, l2{free_dir_of(c.f, neg)};
        BornTrie trie(c.f);
        krng::Stream born;
        born.set_key(5 + (uint64_t)trial, krng::BORN);
        born.at(0);
        std::vector<BornNode> t1, t2, t3;
        std::vector<uint8_t> bits;
        CHECK_EQ(trie.walk(l1, t1, 0, born, bits), -1);
        CHECK_EQ(trie.walk(l2, t2, 0, born, bits), -1);
        CHECK_EQ(trie.walk(l1, t3, 0, born, bits), -1);   // a repeated root direction: served by the memo
        CHECK(same(t1[0].p0, t3[0].p0));
        const uint8_t none = 0;
        CHECK(same(t1[0].p0, trie.p0_from_scratch(l1, &none, 0)));
        CHECK(same(t2[0].p0, trie.p0_from_scratch(l2, &none, 0)));
        CHECK(std::abs(t1[0].p0 + t2[0].p0 - 1.0) <= 1e-12);
        if (std::abs(t1[0].p0 - 0.5) > 1e-9) ++nonfair;
        hits += trie.root_memo_hits;
    }
    std::fprintf(stderr, "[born_trie] root memo: hits=%ld non-fair roots=%ld\n", hits, nonfair);
    CHECK(hits >= 50 && nonfair > 0);
}

void test_random_framed_states() {
    long tot[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // nodes, case-(a) trials, case-(a) nodes, deep non-fair
                                        // nodes, max |p - dense| (1e-18 units), bit diffs after a case-(a)
                                        // step, from-scratch probes, from-scratch chi_j
    test_against_oracles(20261015, 120, tot);    // the S2a report's 120 trials (33 hit case (a))
    for (uint64_t sd = 1; sd <= 4; ++sd) test_against_oracles(sd, 120, tot);
    std::fprintf(stderr, "[born_trie] nodes=%ld case-(a) trials=%ld case-(a) nodes=%ld (all p0 == 0.5 "
                 "bitwise) deep-non-fair nodes=%ld max|p-dense|=%.3g bit-diffs-after-case-(a)=%ld\n",
                 tot[0], tot[1], tot[2], tot[3], tot[4] * 1e-18, tot[5]);
    std::fprintf(stderr, "[born_trie] S2k from-scratch key probes / chi_j = %.3f\n",
                 (double)tot[6] / (double)tot[7]);
    CHECK(tot[1] > 0 && tot[2] > 0);   // case (a) is exercised and served by value
    CHECK(tot[3] > 0);                 // non-fair conditionals at depth >= 1
}

}  // namespace

int main() {
    RUN(test_random_framed_states);
    RUN(test_root_memo);
    REPORT();
}
