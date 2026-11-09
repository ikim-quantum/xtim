// test_born_law_coset.cpp — explaw E1: the coset-of-X-span Born law against two independent references.
//
// Random free registers (k <= 9: random sparse / full / structured supports, non-uniform complex amplitudes, some
// made exact eigenstates of a direction so deterministic steps occur), random commuting independent Hermitian
// directions (d <= k, X and Z parts anywhere on the register) and a random commuting Hermitian column X0 (a_0 in the
// X-span A or not, or no column). Oracles:
//   (B) a dense 2^k brute force with the test's OWN Pauli algebra ((Λψ)_x = c (−1)^{b·(x⊕a)} ψ_{x⊕a}): P(z) = ‖Π_z ψ‖²
//       and N(z) = Re⟨ψ|X0 Π_z ψ⟩ for every z — the table (CosetLaw::build) within 1e-13;
//   (E) BornTrie::enumerate (the production walk, the small-d route): coset_law_leaves must visit the SAME leaf
//       sequence (order included) with weights and h within 1e-13, at eps = 1e-12 (walk_h) and eps = 0 (born_law).
// Mechanism: both per-coset routes (dense WHT / pair sums) occur over the run; a dense-hull d >= 3 content routes to
// the coset law and d <= 2 to enumerate; the routed routine (born_law_enumerate) == enumerate's leaves.
// Mutations (compile with -D): XTIM_MUTATE_EXPLAW_EPSSIGN (ε(C) sign dropped) -> (B) RED; XTIM_MUTATE_EXPLAW_SWAPPHASE
// (the coset swap's sign dropped for a_0 ∉ A) -> (B) RED; XTIM_MUTATE_EXPLAW_REPLAY (the kBranchEps rule dropped) ->
// (E) leaf-set RED.
// Report-only: enumerate vs the coset law wall time on full-support registers (the route's constants).
#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "qeccore/born_law_coset.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/twirl_kernel.hpp"

using namespace qeccore;

namespace {

struct TDir { uint64_t a = 0, b = 0; std::complex<double> c; FreeDir fd; };

inline int parity(uint64_t x) { return __builtin_parityll(x); }

std::vector<std::complex<double>> apply_dir(const std::vector<std::complex<double>>& v, const TDir& d) {
    std::vector<std::complex<double>> o(v.size());
    for (size_t x = 0; x < v.size(); ++x) {
        const size_t src = x ^ d.a;
        o[x] = d.c * (parity(d.b & src) ? -1.0 : 1.0) * v[src];
    }
    return o;
}

TDir make_dir(uint64_t a, uint64_t b, int k, std::mt19937_64& rng) {
    TDir d;
    d.a = a; d.b = b;
    const bool odd = __builtin_popcountll(a & b) & 1;
    const int sign = (int)(rng() & 1);
    d.fd.a.assign((size_t)k, 0); d.fd.b.assign((size_t)k, 0);
    for (int i = 0; i < k; ++i) { d.fd.a[(size_t)i] = (uint8_t)((a >> i) & 1); d.fd.b[(size_t)i] = (uint8_t)((b >> i) & 1); }
    d.fd.phase = (uint8_t)((odd ? 1 : 0) + 2 * sign);
    d.fd.par = (uint8_t)(rng() & 1);
    d.fd.hermitian = 1;
    static const std::complex<double> ip[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    d.c = ip[d.fd.phase & 3] * (d.fd.par ? -1.0 : 1.0);
    return d;
}

bool commute(const TDir& x, const TDir& y) { return (parity(x.a & y.b) ^ parity(y.a & x.b)) == 0; }

// GF(2) independence of the symplectic rows (a | b).
bool independent(const std::vector<TDir>& D, uint64_t a, uint64_t b, int k) {
    std::vector<unsigned __int128> rows;
    auto ins = [&](unsigned __int128 v) {
        for (unsigned __int128 r : rows) {
            const unsigned __int128 low = r & (~r + 1);
            if (v & low) v ^= r;
        }
        if (v) { rows.push_back(v); return true; }
        return false;
    };
    for (const TDir& d : D) ins(((unsigned __int128)d.a << k) | d.b);
    return ins(((unsigned __int128)a << k) | b);
}

struct Inst {
    int k = 0, d = 0;
    FramedSuperposition f{0};
    std::vector<TDir> L;
    TDir X;
    bool column = false;
    std::vector<std::complex<double>> v;    // dense ψ (key bit i = coordinate i)
    int kind = 0;
};

Inst make_inst(std::mt19937_64& rng, int k, int d, int kind) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Inst I;
    I.k = k; I.d = d; I.kind = kind;
    const size_t N = (size_t)1 << k;
    I.v.assign(N, 0.0);
    if (kind == 0) {                                    // random sparse support (contains 0)
        for (size_t x = 0; x < N; ++x) if (x == 0 || (rng() % 3) == 0) I.v[x] = {nd(rng), nd(rng)};
    } else if (kind == 1) {                             // full support
        for (size_t x = 0; x < N; ++x) I.v[x] = {nd(rng), nd(rng)};
    } else if (kind == 2) {                             // uniform ±1 / ±i on a random affine subspace (exact zeros)
        const uint64_t mask = rng() & (N - 1), off = rng() & (N - 1);
        static const std::complex<double> ip[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (size_t x = 0; x < N; ++x) if (((x ^ off) & ~mask & (N - 1)) == 0) I.v[x] = ip[rng() & 3];
    } else {                                            // a product of single-coordinate states (T-like, dense)
        I.v[0] = 1.0;
        for (int i = 0; i < k; ++i) {
            const std::complex<double> a(std::cos(0.3 + 0.37 * i), 0.0), b = std::polar(std::sin(0.3 + 0.37 * i), 0.1 + 0.785 * i);
            for (size_t x = 0; x < N; ++x) if ((x >> i) & 1) I.v[x] = I.v[x ^ (1ull << i)] * b;
            for (size_t x = 0; x < N; ++x) if (!((x >> i) & 1)) I.v[x] *= a;
        }
    }
    // directions
    for (int tries = 0; (int)I.L.size() < d && tries < 4000; ++tries) {
        const uint64_t a = rng() & (N - 1), b = rng() & (N - 1);
        if (!a && !b) continue;
        TDir cand = make_dir(a, b, k, rng);
        bool ok = true;
        for (const TDir& e : I.L) ok = ok && commute(cand, e);
        if (!ok || !independent(I.L, a, b, k)) continue;
        I.L.push_back(cand);
    }
    I.d = (int)I.L.size();
    // some instances: an exact eigenstate of direction 0 (and of direction 2 when present): deterministic steps
    if (I.d > 0 && (rng() % 3) == 0) {
        for (int j : {0, 2}) {
            if (j >= I.d) continue;
            const double s = (rng() & 1) ? 1.0 : -1.0;
            std::vector<std::complex<double>> w = apply_dir(I.v, I.L[(size_t)j]);
            double n2p = 0.0;
            for (size_t x = 0; x < N; ++x) n2p += std::norm(0.5 * (I.v[x] + s * w[x]));
            if (n2p < 1e-6) continue;                        // orthogonal to that eigenspace: keep ψ
            for (size_t x = 0; x < N; ++x) I.v[x] = 0.5 * (I.v[x] + s * w[x]);
        }
    }
    // the column: commuting with every direction (a_0 in A for ~half the instances), or none
    const int cm = (int)(rng() % 3);
    if (cm != 2 && I.d > 0) {
        for (int tries = 0; tries < 4000; ++tries) {
            uint64_t a = rng() & (N - 1), b = rng() & (N - 1);
            if (cm == 0) {                                   // a_0 in A: a random combination of the directions' a's
                a = 0;
                for (const TDir& e : I.L) if (rng() & 1) a ^= e.a;
            }
            if (!a && !b) continue;
            TDir cand = make_dir(a, b, k, rng);
            bool ok = true;
            for (const TDir& e : I.L) ok = ok && commute(cand, e);
            if (!ok) continue;
            I.X = cand; I.column = true;
            break;
        }
    }
    // the framed state: n = k, free = all coordinates, alpha = the dense vector's nonzero entries (random order)
    double n2 = 0.0;
    for (auto& z : I.v) n2 += std::norm(z);
    for (auto& z : I.v) z /= std::sqrt(n2);
    I.f = FramedSuperposition(k);
    I.f.free.clear();
    for (int i = 0; i < k; ++i) I.f.free.push_back(i);
    auto* da = static_cast<DenseAmplitudes*>(I.f.alpha.get());
    da->k_ = k;
    da->b_.clear();
    std::vector<size_t> order;
    for (size_t x = 0; x < N; ++x) if (std::abs(I.v[x]) > 0.0) order.push_back(x);
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t x : order) da->b_.emplace_back((AmpKey)x, I.v[x]);
    return I;
}

// (B) the dense brute force.
void brute(const Inst& I, std::vector<double>& P, std::vector<double>& Nn) {
    const size_t M = (size_t)1 << I.d;
    P.assign(M, 0.0); Nn.assign(M, 0.0);
    for (size_t z = 0; z < M; ++z) {
        std::vector<std::complex<double>> w = I.v;
        for (int j = 0; j < I.d; ++j) {
            const double s = ((z >> j) & 1) ? -1.0 : 1.0;
            std::vector<std::complex<double>> u = apply_dir(w, I.L[(size_t)j]);
            for (size_t x = 0; x < w.size(); ++x) w[x] = 0.5 * (w[x] + s * u[x]);
        }
        double p = 0.0;
        for (auto& c : w) p += std::norm(c);
        P[z] = p;
        if (I.column) {
            std::vector<std::complex<double>> xw = apply_dir(w, I.X);
            std::complex<double> acc(0, 0);
            for (size_t x = 0; x < w.size(); ++x) acc += std::conj(I.v[x]) * xw[x];
            Nn[z] = acc.real();
        }
    }
}

struct Leaf { uint64_t z; double p, h; };
std::vector<Leaf> leaves_enum(Inst& I, double eps, int stop_after = -1) {
    std::vector<FreeDir> lam;
    for (const TDir& d : I.L) lam.push_back(d.fd);
    BornTrie bt(I.f);
    std::vector<Leaf> out;
    bt.enumerate(lam, I.column ? &I.X.fd : nullptr, eps, [&](uint64_t z, double p, double h) {
        out.push_back({z, p, h});
        return stop_after < 0 || (int)out.size() < stop_after;
    });
    return out;
}
std::vector<Leaf> leaves_table(const CosetLawTable& T, double eps, int stop_after = -1) {
    std::vector<Leaf> out;
    coset_law_leaves(T, eps, [&](uint64_t z, double p, double h) {
        out.push_back({z, p, h});
        return stop_after < 0 || (int)out.size() < stop_after;
    });
    return out;
}

double worstB = 0.0, worstE = 0.0;
long dense_total = 0, sparse_total = 0, inst_total = 0, column_swap = 0, column_diag = 0, det_steps = 0;

void test_random_instances() {
    std::mt19937_64 rng(20261030);
    for (int trial = 0; trial < 600; ++trial) {
        const int k = 2 + (int)(rng() % 8);                 // 2..9
        const int d = 1 + (int)(rng() % (uint64_t)k);
        Inst I = make_inst(rng, k, d, (int)(rng() % 4));
        if (I.d == 0) continue;
        ++inst_total;
        std::vector<FreeDir> lam;
        for (const TDir& e : I.L) lam.push_back(e.fd);
        CosetLaw cl(I.f, lam);
        CosetLawTable T;
        cl.build(I.column ? &I.X.fd : nullptr, T);
        dense_total += T.dense_pairs; sparse_total += T.sparse_pairs;
        if (I.column) {
            uint64_t t; (void)t;
            // a_0 in A ⇔ the swap partner is the coset itself: counted through the census of the table
            bool inA = true;
            {
                std::vector<uint64_t> rows;
                auto ins = [&](uint64_t v) { for (uint64_t r : rows) if (v & (r & (~r + 1))) v ^= r; if (v) { rows.push_back(v); return true; } return false; };
                for (const TDir& e : I.L) ins(e.a);
                inA = !ins(I.X.a);
            }
            (inA ? column_diag : column_swap)++;
        }
        std::vector<double> P, Nn;
        brute(I, P, Nn);
        CHECK_EQ(T.P.size(), P.size());
        for (size_t z = 0; z < P.size(); ++z) {
            const double eP = std::abs(T.P[z] - P[z]);
            worstB = std::max(worstB, eP);
            if (!(eP <= 1e-13)) { std::fprintf(stderr, "trial %d k=%d d=%d kind=%d z=%zu P %.17g vs %.17g\n", trial, k, I.d, I.kind, z, T.P[z], P[z]); CHECK(eP <= 1e-13); break; }
            if (I.column) {
                const double eN = std::abs(T.N[z] - Nn[z]);
                worstB = std::max(worstB, eN);
                if (!(eN <= 1e-13)) { std::fprintf(stderr, "trial %d k=%d d=%d kind=%d z=%zu N %.17g vs %.17g (swap=%d)\n", trial, k, I.d, I.kind, z, T.N[z], Nn[z], (int)!column_diag); CHECK(eN <= 1e-13); break; }
            }
        }
        // (E) the leaf semantics: at eps = 1e-12 (walk_h) the SAME leaf sequence; at eps = 0 (born_law) enumerate's own
        // rounding may branch on a truly empty side (a 1e-17 leaf) where the double-double table is exactly 0, so the
        // aggregated law p[z] (born_law's output) is compared, plus every leaf of weight > 1e-13 one-to-one.
        for (double eps : {1e-12, 0.0}) {
            std::vector<Leaf> a = leaves_enum(I, eps), b = leaves_table(T, eps);
            if (eps > 0) {
                CHECK_EQ(a.size(), b.size());
                if (a.size() != b.size()) { std::fprintf(stderr, "trial %d k=%d d=%d kind=%d eps=%g: %zu vs %zu leaves\n", trial, k, I.d, I.kind, eps, a.size(), b.size()); continue; }
                if (a.size() < ((size_t)1 << I.d)) ++det_steps;
                for (size_t i = 0; i < a.size(); ++i) {
                    CHECK_EQ(a[i].z, b[i].z);
                    const double ep = std::abs(a[i].p - b[i].p), eh = std::abs(a[i].h - b[i].h);
                    worstE = std::max(worstE, std::max(ep, eh));
                    if (!(ep <= 1e-13 && eh <= 1e-13))
                        std::fprintf(stderr, "trial %d k=%d d=%d kind=%d eps=%g leaf %zu z=%llu: p %.17g vs %.17g h %.17g vs %.17g\n",
                                     trial, k, I.d, I.kind, eps, i, (unsigned long long)a[i].z, a[i].p, b[i].p, a[i].h, b[i].h);
                    CHECK(ep <= 1e-13);
                    CHECK(eh <= 1e-13);
                }
            } else {
                std::vector<double> pa((size_t)1 << I.d, 0.0), pb((size_t)1 << I.d, 0.0);
                for (const Leaf& l : a) pa[(size_t)l.z] += l.p;
                for (const Leaf& l : b) pb[(size_t)l.z] += l.p;
                for (size_t z = 0; z < pa.size(); ++z) { worstE = std::max(worstE, std::abs(pa[z] - pb[z])); CHECK(std::abs(pa[z] - pb[z]) <= 1e-13); }
                std::vector<Leaf> fa, fb;
                for (const Leaf& l : a) if (l.p > 1e-13) fa.push_back(l);
                for (const Leaf& l : b) if (l.p > 1e-13) fb.push_back(l);
                CHECK_EQ(fa.size(), fb.size());
                for (size_t i = 0; i < fa.size() && i < fb.size(); ++i) {
                    CHECK_EQ(fa[i].z, fb[i].z);
                    CHECK(std::abs(fa[i].p - fb[i].p) <= 1e-13);
                    CHECK(std::abs(fa[i].h - fb[i].h) <= 1e-13);
                }
            }
            if (a.size() > 2) {
                std::vector<Leaf> a2 = leaves_enum(I, eps, 2), b2 = leaves_table(T, eps, 2);
                CHECK_EQ(a2.size(), 2u);
                CHECK_EQ(b2.size(), 2u);
            }
        }
    }
    std::fprintf(stderr, "instances %ld: worst |table - brute| %.2e, worst |leaf - enumerate| %.2e; dense pairs %ld, sparse pairs %ld; "
                         "columns diag %ld swap %ld; walks with deterministic steps %ld\n",
                 inst_total, worstB, worstE, dense_total, sparse_total, column_diag, column_swap, det_steps);
    CHECK(dense_total > 0);
    CHECK(sparse_total > 0);
    CHECK(column_swap > 0);
    CHECK(column_diag > 0);
    CHECK(det_steps > 0);
}

// The routed routine: d <= 2 -> enumerate; a dense full-support content with d >= 3 -> the coset law; either way the
// leaves equal enumerate's.
void test_route() {
    std::mt19937_64 rng(7);
    int coset_routes = 0, enum_routes = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const int k = 3 + (int)(rng() % 7);
        const int d = 1 + (int)(rng() % (uint64_t)k);
        Inst I = make_inst(rng, k, d, 1);
        if (I.d == 0) continue;
        std::vector<FreeDir> lam;
        for (const TDir& e : I.L) lam.push_back(e.fd);
        std::vector<Leaf> a = leaves_enum(I, 1e-12), b;
        BornRoute r;
        std::shared_ptr<const CosetLawTable> T;
        long ev = 0;
        born_law_enumerate(I.f, lam, I.column ? &I.X.fd : nullptr, 1e-12, [&](uint64_t z, double p, double h) {
            b.push_back({z, p, h});
            return true;
        }, &ev, &T, &r);
        CHECK_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            CHECK_EQ(a[i].z, b[i].z);
            CHECK(std::abs(a[i].p - b[i].p) <= 1e-13);
            CHECK(std::abs(a[i].h - b[i].h) <= 1e-13);
        }
        if (I.d <= 2) { CHECK(!r.coset); CHECK(!T); }
        CHECK_EQ((int)r.coset, (int)(bool)T);
        if (r.coset) { ++coset_routes; CHECK(ev == (long)b.size() + T->cosets); }
        else ++enum_routes;
        if (I.d >= 5 && r.dX >= 4) CHECK(r.coset);        // dense hull, 2^dX / dX >= 4: the coset law wins
    }
    std::fprintf(stderr, "routes: coset %d, enumerate %d\n", coset_routes, enum_routes);
    CHECK(coset_routes > 0);
    CHECK(enum_routes > 0);
}

// Report-only: the two routes' wall times on full-support registers with d = k - 1 X-type directions (the ladder regime).
void report_timing() {
    std::mt19937_64 rng(11);
    for (int k : {9, 11, 12, 13}) {
        Inst I = make_inst(rng, k, 1, 3);
        I.L.clear();
        for (int i = 0; i < k - 1; ++i) I.L.push_back(make_dir(1ull << i, 0, k, rng));   // the ladder's X_i
        I.d = k - 1;
        I.column = true;
        I.X = make_dir(1ull << (k - 1), 0, k, rng);
        std::vector<FreeDir> lam;
        for (const TDir& e : I.L) lam.push_back(e.fd);
        const auto t0 = std::chrono::steady_clock::now();
        long n1 = 0;
        { BornTrie bt(I.f); bt.enumerate(lam, &I.X.fd, 1e-12, [&](uint64_t, double, double) { ++n1; return true; }); }
        const auto t1 = std::chrono::steady_clock::now();
        CosetLaw cl(I.f, lam);
        CosetLawTable T;
        cl.build(&I.X.fd, T);
        long n2 = 0;
        coset_law_leaves(T, 1e-12, [&](uint64_t, double, double) { ++n2; return true; });
        const auto t2 = std::chrono::steady_clock::now();
        const BornRoute& r = cl.route();
        std::fprintf(stderr, "k=%d d=%d leaves %ld/%ld: enumerate %.3f s, coset law %.4f s (model: enum %.3g ns, coset %.3g ns, route %s)\n",
                     k, k - 1, n1, n2, std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
                     r.cost_enum, r.cost_coset, r.coset ? "coset" : "enumerate");
    }
}

}  // namespace

int main() {
    RUN(test_random_instances);
    RUN(test_route);
    RUN(report_timing);
    REPORT();
}
