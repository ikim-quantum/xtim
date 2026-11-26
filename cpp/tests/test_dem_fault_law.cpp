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
#include <string>
#include <vector>

#include "qeccore/dem_decompose.hpp"

using namespace qeccore;

namespace {

// twirl-routes T3: the two retired production solvers, kept HERE as independent oracles (one production solver
// remains: solve_mixture_fwht for Pauli instances, solve_mixture_classes / erasure_q for Clifford records).
// Exact conversion of one categorical channel (merged distinct signatures + probabilities)
// into independent error mechanisms, matching Stim: Walsh-Hadamard solve over the group the
// signatures generate. Returns false with a diagnostic when the channel is not exactly
// representable (over-mixing or a negative independent probability).
bool solve_channel(const std::map<Sig, double>& merged, std::vector<Mech>& out,
                   std::string& err, bool* needs_approximation, int max_rank, int* rank) {
    if (needs_approximation) *needs_approximation = false;
    if (rank) *rank = 0;
    std::vector<Sig> sigs;
    std::vector<double> probs;
    double total = 0.0;
    for (const auto& [s, p] : merged) {
        if (p <= 0.0) continue;
        sigs.push_back(s);
        probs.push_back(p);
        total += p;
    }
    if (sigs.empty()) return true;
    if (sigs.size() == 1) {                 // single mechanism: the merged mass directly
        if (rank) *rank = 1;
        out.push_back({sigs[0], probs[0]});  // (exact for any p in [0,1], like X_ERROR(0.9))
        return true;
    }
    // GF(2) reduction of the signatures (symmetric difference of sorted target lists) to a
    // basis; coords[i] = sig i in basis coordinates. export_dem calls this for all-Pauli instances with
    // max_rank = 4 (EXP-fix routing), so the naive 2^m x 2^m loops below stay tiny.
    auto sym_diff = [](const Sig& a, const Sig& b) {
        Sig r;
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                      std::back_inserter(r));
        return r;
    };
    // Basis vectors have pairwise-distinct pivots (first elements) by construction: each
    // candidate is pivot-reduced against the existing basis until its pivot is fresh. If
    // sig = residual XOR (XOR of basis[k] over used k's), its coordinates over the basis are
    // exactly the used-k bits (plus the residual's own new bit when it joins the basis).
    std::vector<Sig> basis;
    std::vector<uint32_t> coords(sigs.size(), 0);
    for (size_t i = 0; i < sigs.size(); ++i) {
        Sig cur = sigs[i];
        uint32_t c = 0;
        bool changed = true;
        while (changed && !cur.empty()) {
            changed = false;
            for (size_t k = 0; k < basis.size(); ++k)
                if (!cur.empty() && cur[0] == basis[k][0]) {
                    cur = sym_diff(cur, basis[k]);
                    c ^= 1u << k;
                    changed = true;
                }
        }
        if (!cur.empty()) {
            c |= 1u << basis.size();
            basis.push_back(std::move(cur));
        }
        coords[i] = c;
    }
    const int m = (int)basis.size();
    if (rank) *rank = m;
    if (m > max_rank) { err = "signature group rank above the caller's bound"; return false; }
    const uint32_t G = 1u << m;
    std::vector<double> mu(G, 0.0);
    for (size_t i = 0; i < sigs.size(); ++i) mu[coords[i]] += probs[i];
    mu[0] += 1.0 - total;
    // Walsh-Hadamard transform: muh[c] = sum_g (-1)^{popcount(c & g)} mu[g].
    std::vector<double> muh(G, 0.0);
    for (uint32_t c = 0; c < G; ++c)
        for (uint32_t g = 0; g < G; ++g)
            muh[c] += (__builtin_popcount(c & g) & 1) ? -mu[g] : mu[g];
    for (uint32_t c = 0; c < G; ++c)
        if (!(muh[c] > 1e-12)) {
            err = "over-mixing categorical channel (a Fourier coefficient of the signature "
                  "distribution is <= 0); not exactly DEM-convertible";
            return false;
        }
    for (uint32_t h = 1; h < G; ++h) {
        double L = 0.0;
        for (uint32_t c = 0; c < G; ++c)
            L += ((__builtin_popcount(c & h) & 1) ? -1.0 : 1.0) * std::log(muh[c]);
        L *= -2.0 / (double)G;
        double q = (1.0 - std::exp(L)) / 2.0;
        if (q < -1e-12) {
            err = "channel is not a product of independent record-flip mechanisms; exact "
                  "DEM conversion has a negative probability; refusing to approximate";
            if (needs_approximation) *needs_approximation = true;
            return false;
        }
        if (q <= 0.0) continue;
        Sig s;
        for (int k = 0; k < m; ++k)
            if ((h >> k) & 1) {
                Sig r;
                std::set_symmetric_difference(s.begin(), s.end(), basis[k].begin(),
                                              basis[k].end(), std::back_inserter(r));
                s = std::move(r);
            }
        if (s.empty()) continue;             // cannot happen (h != 0, basis independent)
        out.push_back({std::move(s), q});
    }
    return true;
}

SolveStatus solve_coset_fault(const FaultLaw& f, std::vector<Mech>& out) {
    const int v = (int)f.W.size();
    const double p = f.p;
    if (v == 0) {                                       // single signature: q = p (any p)
        if (!f.c.empty() && p > 0.0) out.push_back({f.c, p});
        return SolveStatus::Ok;
    }
    // c' in W? (rank of W with and without c', from the origin)
    std::vector<Sig> pw{Sig{}}, pwc{Sig{}};
    for (const Sig& w : f.W) { pw.push_back(w); pwc.push_back(w); }
    pwc.push_back(f.c);
    const bool c_in_W = sig_affine_rank(pw) == sig_affine_rank(pwc);
    auto walk = [&](const Sig& start, bool skip_start, double q) {   // start + every element of W
        Sig cur = start;
        if (!skip_start) out.push_back({cur, q});
        for (uint32_t g = 1; g < (1u << v); ++g) {
            cur = sig_xor(cur, f.W[(size_t)__builtin_ctz(g)]);
            out.push_back({cur, q});
        }
    };
    if (c_in_W) {
        if (!(1.0 - p > 1e-12)) return SolveStatus::OverMixing;
        const double q = -std::expm1(std::log1p(-p) * std::ldexp(1.0, 1 - v)) / 2.0;
        if (q > 0.0) walk(Sig{}, true, q);
        return SolveStatus::Ok;
    }
    if (!(1.0 - 2.0 * p > 1e-12)) return SolveStatus::OverMixing;
    const double l1 = std::log1p(-p), l2 = std::log1p(-2.0 * p);
    const double qW = -std::expm1((2.0 * l1 - l2) * std::ldexp(1.0, -v)) / 2.0;
    if (qW < -1e-12) return SolveStatus::Negative;
    const double qc = -std::expm1(l2 * std::ldexp(1.0, -v)) / 2.0;
    if (qc > 0.0) walk(f.c, false, qc);
    if (qW > 0.0) walk(Sig{}, true, qW);               // never (q_W <= 0); kept for exactness
    return SolveStatus::Ok;
}


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
    return solve_channel(merged, out, err, &neg, 20, nullptr);
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

// ── dem-erasure T1: the weight-cutoff enumerator against a brute-force 2^v oracle ───────────────
// Ids: detectors 0..ND-1 (D = ND), observables ND..ND+NO-1. A random coset: r det-bearing directions,
// kappa observable-only directions, an offset with 0..3 base detectors outside W's support (as the
// export's deterministic base), pure or offset.
struct ErasureCase { FaultLaw f; int D; };

ErasureCase rand_erasure(int v_target, int kappa, int nbase, bool pure) {
    for (;;) {
        const int ND = rnd(std::max(2, v_target - kappa), 14), NO = rnd(kappa > 0 ? kappa : 0, kappa + 3);
        FaultLaw f;
        f.kind = FaultLaw::Kind::Coset;
        f.p = PS[rnd(0, 4)];
        auto rnd_ids = [&](int lo, int hi, double dens) {
            Sig s;
            for (int t = lo; t < hi; ++t) if (unif() < dens) s.push_back(t);
            return s;
        };
        for (int j = 0; j < v_target - kappa; ++j) {
            Sig s = sig_xor(rnd_ids(0, ND, 0.3), rnd_ids(ND, ND + NO, 0.3));
            f.W.push_back(s);
        }
        for (int j = 0; j < kappa; ++j) f.W.push_back(rnd_ids(ND, ND + NO, 0.5));
        Sig c;
        if (pure) { for (const Sig& w : f.W) if (rng() & 1) c = sig_xor(c, w); }
        else c = sig_xor(rnd_ids(0, ND, 0.3), rnd_ids(ND, ND + NO, 0.3));
        // base detectors: ids above every W id (outside W's support), always offset
        if (nbase > 0 && !pure) for (int b = 0; b < nbase; ++b) c = sig_xor(c, Sig{ND + NO + b});
        f.c = c;
        const int D = nbase > 0 && !pure ? ND + NO + nbase : ND;   // base ids are detectors
        if (nbase > 0 && !pure) {
            // re-label: observables must sit at ids >= D; shift them above the base detectors
            auto relabel = [&](Sig s) {
                Sig o;
                for (int32_t t : s) o.push_back(t >= ND && t < ND + NO ? t + nbase + 100 : t);
                std::sort(o.begin(), o.end());
                return o;
            };
            for (Sig& w : f.W) w = relabel(w);
            f.c = relabel(f.c);
            // detectors are now 0..ND-1 and ND+NO..ND+NO+nbase-1; observables >= ND+NO+nbase+100
            auto L = law_of(f);
            if ((int)L.size() != (1 << v_target)) continue;
            if (L.count(Sig{}) > 0) continue;
            return {f, ND + NO + nbase};
        }
        auto L = law_of(f);
        if ((int)L.size() != (1 << v_target)) continue;          // W independent
        if (pure != (L.count(Sig{}) > 0)) continue;
        return {f, D};
    }
}

int det_w(const Sig& s, int D) { int c = 0; for (int32_t t : s) if (t < D) ++c; return c; }

void test_erasure_enumerator_oracle() {
    const int TS[] = {0, 1, 2, 3, -1};
    int trials = 0, kept_tot = 0, dropped_tot = 0, obs_only_kept = 0;
    for (int trial = 0; trial < 1500; ++trial) {
        const int v = rnd(1, 10), kappa = rnd(0, std::min(3, v)), nbase = rnd(0, 3);
        const bool pure = (trial % 3) == 0;
        ErasureCase ec = rand_erasure(v, kappa, pure ? 0 : nbase, pure);
        const int t = TS[rnd(0, 4)];
        // Oracle: the affine span by brute force, filtered by detector weight, the empty pattern excluded.
        std::set<Sig> want;
        int total = 0;
        for (auto& [s, w] : law_of(ec.f)) {
            if (s.empty()) continue;
            ++total;
            if (t < 0 || det_w(s, ec.D) <= t) want.insert(s);
        }
        std::vector<Sig> got;
        bool is_pure = false;
        int vv = -1;
        const long long n = enumerate_erasure(ec.f, ec.D, t, [&](const Sig& s) { got.push_back(s); }, &is_pure, &vv);
        std::set<Sig> gs(got.begin(), got.end());
        CHECK(gs.size() == got.size());                  // no duplicates
        CHECK(gs == want);
        CHECK(n == (long long)got.size());
        CHECK(is_pure == pure);
        CHECK(vv == v);
        CHECK(total == (1 << v) - (pure ? 1 : 0));
        dropped_tot += total - (int)n;
        kept_tot += (int)n;
        for (const Sig& s : got) if (det_w(s, ec.D) == 0) ++obs_only_kept;
        ++trials;
    }
    std::printf("erasure enumerator: trials=%d kept=%d dropped=%d obs_only_kept=%d\n", trials, kept_tot,
                dropped_tot, obs_only_kept);
    CHECK(dropped_tot > 1000 && obs_only_kept > 100);
}

void test_erasure_exact_matches_case_a() {
    // t = inf on a pure coset reproduces solve_coset_fault's Case A lines bit-for-bit (same sigs, same q).
    int n_cmp = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int v = rnd(1, 8);
        ErasureCase ec = rand_erasure(v, rnd(0, std::min(2, v)), 0, true);
        std::vector<Mech> a;
        CHECK(solve_coset_fault(ec.f, a) == SolveStatus::Ok);
        double q = -1;
        CHECK(erasure_q(ec.f.p, v, false, q) == SolveStatus::Ok);
        std::map<Sig, double> A, B;
        for (auto& m : a) A[m.sig] = m.p;
        enumerate_erasure(ec.f, ec.D, -1, [&](const Sig& s) { B[s] = q; });
        CHECK(A.size() == (size_t)((1 << v) - 1));
        CHECK(A == B);                                   // exact double equality
        ++n_cmp;
    }
    // Offset: c + W at q_c equals Case B's c + W lines (where q_W is skipped: tiny p).
    for (int trial = 0; trial < 200; ++trial) {
        const int v = rnd(1, 6);
        ErasureCase ec = rand_erasure(v, 0, 0, false);
        ec.f.p = 1e-9;
        std::vector<Mech> a;
        CHECK(solve_coset_fault(ec.f, a) == SolveStatus::Ok);
        double q = -1, qW = 1;
        CHECK(erasure_q(ec.f.p, v, true, q, &qW) == SolveStatus::Ok);
        CHECK(qW < 0.0);
        std::map<Sig, double> A, B;
        for (auto& m : a) A[m.sig] = m.p;
        enumerate_erasure(ec.f, ec.D, -1, [&](const Sig& s) { B[s] = q; });
        CHECK(B.size() == (size_t)(1 << v));
        CHECK(A == B);
        ++n_cmp;
    }
    // Over-mixing boundaries.
    double q = -1;
    CHECK(erasure_q(1.0, 2, false, q) == SolveStatus::OverMixing);
    CHECK(erasure_q(0.5, 2, true, q) == SolveStatus::OverMixing);
    std::printf("erasure exact == closed form: %d comparisons\n", n_cmp);
}

void test_erasure_candidates_budget() {
    // r det directions, kappa obs-only: candidates = sum_{j<=t} C(r,j) 2^kappa; 2^v at t >= r or t < 0.
    for (int r = 0; r <= 40; r += 5)
        for (int kappa = 0; kappa <= 3; ++kappa) {
            std::vector<std::vector<uint64_t>> W;
            std::vector<uint64_t> dm(1, r ? (r == 64 ? ~0ull : ((1ull << r) - 1)) : 0);
            for (int i = 0; i < r; ++i) W.push_back({1ull << i});
            for (int i = 0; i < kappa; ++i) W.push_back({1ull << (50 + i)});
            ErasureSplit s = erasure_split(W, dm);
            CHECK((int)s.g.size() == r && (int)s.h.size() == kappa);
            const double c2 = (1.0 + r + r * (r - 1) / 2.0) * std::ldexp(1.0, kappa);
            CHECK(r < 2 || erasure_candidates(s, 2) == c2);
            CHECK(erasure_candidates(s, -1) == std::ldexp(1.0, r + kappa));
            CHECK(erasure_candidates(s, 100) == std::ldexp(1.0, r + kappa));
        }
    // the default t = 2 admits r = 90 at kappa = 0 within 2^12 (1 + 90 + 4005 = 4096), not r = 91 (4187)
    ErasureSplit s;
    s.g.resize(90);
    CHECK(erasure_candidates(s, 2) == 4096.0);
    s.g.resize(91);
    CHECK(erasure_candidates(s, 2) == 4187.0);
}

}  // namespace

// twirl-routes T1b: the class-wise closed form (solve_mixture_classes) of a random Born MIXTURE law
// Σ_i w_i U(c_i + span(W)) (<= 16 classes = kappa <= 4 Born leaves, v <= 10) against solve_mixture_fwht on its explicit
// support (2^r patterns per class): same status, the same mechanism set and every q within 1e-14 relative of the
// largest (summation order only); at a finite cutoff, the same filtered lines and the dropped mass by brute force.
void test_mixture_classes_vs_fwht() {
    int cmp = 0, neg = 0;
    double worst = 0.0;
    for (int trial = 0; trial < 1500; ++trial) {
        const int U = rnd(3, 11);
        FaultLaw f;
        f.kind = FaultLaw::Kind::Mixture;
        f.p = (trial & 1) ? 0.15 : 1e-3;
        const int r = rnd(0, 4);
        for (int j = 0; j < r; ++j) f.W.push_back(rand_sig(U));
        const int ncls = rnd(1, 16);
        double tot = 0.0;
        for (int i = 0; i < ncls; ++i) {
            const double w = 0.1 + unif();
            f.sup.push_back({rand_sig(U, 0.3), w});
            tot += w;
        }
        for (auto& e : f.sup) e.second /= tot;
        // the explicit support law (duplicates merged), solved by the 2^rank FWHT
        std::map<Sig, double> L;
        for (auto& [c, w] : f.sup) {
            std::set<Sig> coset;
            const int v = (int)f.W.size();
            for (uint32_t z = 0; z < (1u << v); ++z) {
                Sig s = c;
                for (int j = 0; j < v; ++j) if ((z >> j) & 1) s = sig_xor(s, f.W[(size_t)j]);
                coset.insert(s);
            }
            for (const Sig& s : coset) L[s] += w / (double)coset.size();
        }
        FaultLaw g;
        g.kind = FaultLaw::Kind::Support;
        g.p = f.p;
        for (auto& e : L) g.sup.push_back(e);
        std::vector<Mech> a, b;
        int rank = 0;
        const SolveStatus sa = solve_mixture_fwht({g}, 20, a, &rank);
        double dropped = 0.0;
        int v = 0;
        Sig supp;
        const SolveStatus sb = solve_mixture_classes(f, 1000, -1, b, dropped, v, supp);
        CHECK(sa == sb);
        if (sa != sb) { std::fprintf(stderr, "  status %d vs %d\n", (int)sa, (int)sb); continue; }
        std::vector<Sig> pts;
        for (auto& e : L) pts.push_back(e.first);
        CHECK(v == sig_affine_rank(pts));
        if (sa == SolveStatus::Negative) {                 // the disjoint law: every nonempty support pattern at p w
            ++neg;
            std::map<Sig, double> dj;
            for (auto& m : b) dj[m.sig] += m.p;
            std::map<Sig, double> want;
            for (auto& [s2, w] : L) if (!s2.empty()) want[s2] = f.p * w;
            CHECK(dj.size() == want.size());
            for (auto& [s2, w] : want) CHECK(std::abs(dj[s2] - w) <= 1e-15);
            continue;
        }
        std::map<Sig, double> ma, mb;
        double qmax = 0.0;
        for (auto& m : a) { ma[m.sig] = m.p; qmax = std::max(qmax, m.p); }
        for (auto& m : b) { CHECK(!mb.count(m.sig)); mb[m.sig] = m.p; }
        std::set<Sig> keys;
        for (auto& e : ma) keys.insert(e.first);
        for (auto& e : mb) keys.insert(e.first);
        for (auto& k : keys) {
            const double x = ma.count(k) ? ma[k] : 0.0, y = mb.count(k) ? mb[k] : 0.0;
            worst = std::max(worst, std::abs(x - y) / qmax);
            if (!(std::abs(x - y) <= 1e-14 * qmax)) {
                std::fprintf(stderr, "  q %.17g vs %.17g (qmax %.3g)\n", x, y, qmax);
                CHECK(false);
                break;
            }
        }
        ++cmp;
        // a finite cutoff (ids < D = 13: the first 4 ids are detectors): the same lines, filtered, and the dropped mass
        const int D = 13, t = rnd(0, 3);
        auto dw = [&](const Sig& sg) { int c = 0; for (int32_t x : sg) if (x < D) ++c; return c; };
        std::vector<Mech> bt;
        double dropped_t = 0.0;
        CHECK(solve_mixture_classes(f, D, t, bt, dropped_t, v, supp) == sb);
        std::map<Sig, double> mt;
        for (auto& m : bt) mt[m.sig] = m.p;
        std::map<Sig, double> want;
        for (auto& [k, q] : mb) if (dw(k) <= t) want[k] = q;
        CHECK(mt == want);
        double dd = 0.0;
        for (auto& [s2, w] : L) if (dw(s2) > t) dd += f.p * w;
        CHECK(std::abs(dd - dropped_t) <= 1e-15);
    }
    std::printf("mixture classes: %d compared (%d negative), worst |dq|/qmax %.3g\n", cmp, neg, worst);
    CHECK(cmp >= 250 && neg >= 250);
}

int main() {
    RUN(test_erasure_enumerator_oracle);
    RUN(test_erasure_exact_matches_case_a);
    RUN(test_erasure_candidates_budget);
    RUN(test_case_closed_values);
    RUN(test_single_coset);
    RUN(test_mixtures);
    RUN(test_rank_cap);
    RUN(test_mixture_classes_vs_fwht);
    REPORT();
}
