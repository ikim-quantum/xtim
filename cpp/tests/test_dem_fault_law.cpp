// G-DEM-ORACLE (EXP-fix E1a): the owner's per-fault closed form (solve_coset_fault) and the joint
// FWHT solve (solve_mixture_fwht) against two independent references:
//   (1) the FORWARD model: the independent mechanisms the solver emits, XOR-composed over the
//       signature group by brute force, must reproduce the instance's categorical law
//       (1 - sum p_i) delta_0 + sum_i p_i law_i to 1e-10 (a direct check of the claim, no inverse
//       transform involved);
//   (2) solve_channel, the naive O(4^m) categorical -> independent solve the DEM used until EXP-fix:
//       same status, and every emitted q within 1e-9 relative (|q| < 1e-15 counts as absent).
// Random cosets (v <= 4, c' in W and not), random explicit supports, random Pauli signatures and
// mixtures of all three, at p from 1e-4 to 0.45 (Case B at p >= 0.5 over-mixes in both solvers).
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "qeccore/dem_decompose.hpp"

using namespace qeccore;

namespace {

std::mt19937_64 rng(20261021ull);
int rnd(int lo, int hi) { return lo + (int)(rng() % (uint64_t)(hi - lo + 1)); }
double unif() { return (double)(rng() >> 11) * 0x1.0p-53; }

Sig rand_sig(int U, double dens = 0.35) {
    Sig s;
    for (int t = 0; t < U; ++t) if (unif() < dens) s.push_back(t * 3 + 1);   // sparse ids
    return s;
}

// The fault's conditional law as a categorical map.
std::map<Sig, double> law_of(const FaultLaw& f) {
    std::map<Sig, double> L;
    if (f.kind == FaultLaw::Kind::Pauli) { L[f.c] += 1.0; return L; }
    if (f.kind == FaultLaw::Kind::Support) { for (auto& e : f.sup) L[e.first] += e.second; return L; }
    const int v = (int)f.W.size();
    for (uint32_t z = 0; z < (1u << v); ++z) {
        Sig s = f.c;
        for (int j = 0; j < v; ++j) if ((z >> j) & 1) s = sig_xor(s, f.W[(size_t)j]);
        L[s] += std::ldexp(1.0, -v);
    }
    return L;
}

std::map<Sig, double> instance_law(const std::vector<FaultLaw>& comps) {
    std::map<Sig, double> L;
    double tot = 0;
    for (auto& f : comps) { for (auto& [s, w] : law_of(f)) L[s] += f.p * w; tot += f.p; }
    L[Sig{}] += 1.0 - tot;
    return L;
}

// Forward model: compose independent mechanisms.
std::map<Sig, double> compose(const std::vector<Mech>& ms) {
    std::map<Sig, double> D;
    D[Sig{}] = 1.0;
    for (auto& m : ms) {
        std::map<Sig, double> N;
        for (auto& [s, w] : D) { N[s] += w * (1 - m.p); N[sig_xor(s, m.sig)] += w * m.p; }
        D.swap(N);
    }
    return D;
}

double law_dist(const std::map<Sig, double>& a, const std::map<Sig, double>& b) {
    std::set<Sig> keys;
    for (auto& e : a) keys.insert(e.first);
    for (auto& e : b) keys.insert(e.first);
    double d = 0;
    for (auto& k : keys) {
        auto ia = a.find(k), ib = b.find(k);
        d = std::max(d, std::abs((ia == a.end() ? 0 : ia->second) - (ib == b.end() ? 0 : ib->second)));
    }
    return d;
}

// Naive reference: solve_channel on the merged non-empty signatures.
bool naive(const std::vector<FaultLaw>& comps, std::vector<Mech>& out, bool& neg) {
    std::map<Sig, double> merged;
    for (auto& f : comps) for (auto& [s, w] : law_of(f)) if (!s.empty()) merged[s] += f.p * w;
    std::string err;
    neg = false;
    return solve_channel(merged, out, err, &neg);
}

int mech_mismatch(const std::vector<Mech>& a, const std::vector<Mech>& b, double rel) {
    std::map<Sig, double> A, B;
    for (auto& m : a) if (std::abs(m.p) >= 1e-15) A[m.sig] = m.p;
    for (auto& m : b) if (std::abs(m.p) >= 1e-15) B[m.sig] = m.p;
    int bad = 0;
    for (auto& [s, q] : A) { auto it = B.find(s); if (it == B.end() || std::abs(it->second - q) > rel * std::abs(q)) ++bad; }
    for (auto& [s, q] : B) if (!A.count(s)) ++bad;
    return bad;
}

FaultLaw rand_coset(int U, int v, bool want_in_W, double p) {
    for (;;) {
        FaultLaw f;
        f.kind = FaultLaw::Kind::Coset;
        f.p = p;
        for (int j = 0; j < v; ++j) f.W.push_back(rand_sig(U));
        // independence + c' membership via the brute-force law: |support| must be 2^v
        f.c = want_in_W ? Sig{} : rand_sig(U);
        if (want_in_W) for (int j = 0; j < v; ++j) if (rng() & 1) f.c = sig_xor(f.c, f.W[(size_t)j]);
        auto L = law_of(f);
        if ((int)L.size() != (1 << v)) continue;
        if (want_in_W != (L.count(Sig{}) > 0)) continue;
        return f;
    }
}

FaultLaw rand_support(int U, double p) {
    FaultLaw f;
    f.kind = FaultLaw::Kind::Support;
    f.p = p;
    const int n = rnd(1, 6);
    std::vector<double> w;
    double tot = 0;
    for (int i = 0; i < n; ++i) { w.push_back(0.1 + unif()); tot += w.back(); }
    std::map<Sig, double> L;
    for (int i = 0; i < n; ++i) L[rand_sig(U)] += w[(size_t)i] / tot;
    for (auto& e : L) f.sup.push_back(e);
    return f;
}

const double PS[] = {1e-4, 1e-3, 0.01, 0.05, 0.2, 0.45};

void test_single_coset() {
    int ok = 0, negc = 0, overc = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const int v = rnd(0, 4), U = rnd(v + 1, v + 5);
        const bool inW = (trial & 1) || v == 0 ? (trial & 1) : false;
        FaultLaw f = rand_coset(U, v, inW && true, PS[rnd(0, 5)]);
        std::vector<Mech> a, b, c;
        const SolveStatus sa = solve_coset_fault(f, a);
        int rk = -1;
        const SolveStatus sc = solve_mixture_fwht({f}, 20, c, &rk);
        bool neg = false;
        const bool okn = naive({f}, b, neg);
        const SolveStatus sb = okn ? SolveStatus::Ok : neg ? SolveStatus::Negative : SolveStatus::OverMixing;
        CHECK(sa == sb);
        CHECK(sc == sb);
        if (sa != SolveStatus::Ok) { (sa == SolveStatus::Negative ? negc : overc)++; continue; }
        ++ok;
        CHECK(mech_mismatch(a, b, 1e-9) == 0);
        CHECK(mech_mismatch(c, b, 1e-9) == 0);
        CHECK(law_dist(compose(a), instance_law({f})) < 1e-10);
    }
    std::printf("single coset: ok=%d negative=%d overmix=%d\n", ok, negc, overc);
    CHECK(ok > 200 && negc > 50);
}

void test_case_closed_values() {
    // Case A v=1: q = p/2; Case B v=1 at p: (1-2q_c)^2 = 1-2p.
    FaultLaw f;
    f.kind = FaultLaw::Kind::Coset;
    f.p = 0.01;
    f.W = {Sig{1}};
    f.c = Sig{};
    std::vector<Mech> a;
    CHECK(solve_coset_fault(f, a) == SolveStatus::Ok);
    CHECK(a.size() == 1 && std::abs(a[0].p - 0.005) < 1e-17);
    f.c = Sig{4};
    f.p = 1e-9;                         // Case B at tiny p: q_W ~ -p^2/4 > -1e-12 -> skipped
    a.clear();
    CHECK(solve_coset_fault(f, a) == SolveStatus::Ok);
    CHECK(a.size() == 2);
    for (auto& m : a) CHECK(std::abs(m.p - (5e-10 + 2.5e-19)) < 1e-24);   // (1-2q)^2 = 1-2p: q = p/2 + p^2/4 + O(p^3)
    f.p = 0.5;
    a.clear();
    CHECK(solve_coset_fault(f, a) == SolveStatus::OverMixing);
}

void test_mixtures() {
    int ok = 0, negc = 0, overc = 0;
    for (int trial = 0; trial < 2500; ++trial) {
        const int U = rnd(3, 8);
        const int nc = rnd(2, 4);
        std::vector<FaultLaw> comps;
        const double ptot = PS[rnd(0, 4)];
        for (int i = 0; i < nc; ++i) {
            const int kind = rnd(0, 2);
            const double p = ptot / nc;
            if (kind == 0) { FaultLaw f; f.kind = FaultLaw::Kind::Pauli; f.p = p; f.c = rand_sig(U); comps.push_back(f); }
            else if (kind == 1) comps.push_back(rand_coset(U, rnd(0, std::min(3, U - 1)), rng() & 1, p));
            else comps.push_back(rand_support(U, p));
        }
        std::vector<Mech> b, c;
        int rk = -1;
        const SolveStatus sc = solve_mixture_fwht(comps, 20, c, &rk);
        bool neg = false;
        const bool okn = naive(comps, b, neg);
        const SolveStatus sb = okn ? SolveStatus::Ok : neg ? SolveStatus::Negative : SolveStatus::OverMixing;
        CHECK(sc == sb);
        if (sc != SolveStatus::Ok) { (sc == SolveStatus::Negative ? negc : overc)++; continue; }
        ++ok;
        CHECK(mech_mismatch(c, b, 1e-9) == 0);
        CHECK(law_dist(compose(c), instance_law(comps)) < 1e-10);
    }
    std::printf("mixtures: ok=%d negative=%d overmix=%d\n", ok, negc, overc);
    CHECK(ok > 100 && negc > 100);
}

void test_rank_cap() {
    // 22 independent Pauli signatures: rank 22 > 20 -> RankTooLarge, nothing emitted.
    std::vector<FaultLaw> comps;
    for (int i = 0; i < 22; ++i) { FaultLaw f; f.p = 1e-3; f.c = Sig{i}; comps.push_back(f); }
    std::vector<Mech> out;
    int rk = -1;
    CHECK(solve_mixture_fwht(comps, 20, out, &rk) == SolveStatus::RankTooLarge);
    CHECK(rk == 22 && out.empty());
}

}  // namespace

int main() {
    RUN(test_case_closed_values);
    RUN(test_single_coset);
    RUN(test_mixtures);
    RUN(test_rank_cap);
    REPORT();
}
