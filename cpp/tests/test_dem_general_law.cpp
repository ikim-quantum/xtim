// EXP-fix: the classified correlated-read law (dem_export.cpp classify_correlated_law + correlated_fault_law) on
// HAND-BUILT lean states, against a brute-force oracle. The circuit-reachable laws in the DEM class are uniform
// cosets (corpus census: no GENERAL law in 738,998 alternatives); this test drives the GENERAL path directly with
// states whose Born values are fractional, and the COSET paths (fast character test, re-detection) alike.
//
// Oracle: the targets' flip pattern f in GF(2)^m has Fourier transform <A_S> (A_S = product of the commuting
// Hermitian frame-coordinate Paulis Qf over S), so P(f) = 2^-m sum_S (-1)^{f.S} <A_S>, every <A_S> one direct
// framed_expectation_frame Born sum — no Fourier-support elimination, no memo, no character test. The law the
// engine builds (a Coset's uniform c + span(W), or a Support's weights) must equal it to 1e-12 (absolute), and its
// kind must match: COSET iff the oracle law is uniform on an affine subspace. Mutation MX-CHAR (snap a fractional
// basis value to +-1) makes a GENERAL law a coset: RED here.
#include "check.hpp"

#include <complex>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "qeccore/dem_decompose.hpp"
#include "qeccore/dem_export.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"

using namespace qeccore;

namespace {

std::mt19937_64 rng(20261022ull);
int rnd(int lo, int hi) { return lo + (int)(rng() % (uint64_t)(hi - lo + 1)); }

// Hand-built lean state: random eps, random distinct `free`, chi distinct sign patterns with
// random normalised coefficients. (The Born kernel reads eps/free/entries only.)
FramedSuperposition random_state(int n, int r, int chi) {
    FramedSuperposition L(n);
    for (int a = 0; a < n; ++a) L.eps[(size_t)a] = (uint8_t)(rng() & 1);
    std::vector<int> perm(n);
    for (int a = 0; a < n; ++a) perm[(size_t)a] = a;
    for (int a = n - 1; a > 0; --a) std::swap(perm[(size_t)a], perm[(size_t)rnd(0, a)]);
    L.free.assign(perm.begin(), perm.begin() + r);
    std::vector<std::vector<uint8_t>> sigs;
    while ((int)sigs.size() < chi) {
        std::vector<uint8_t> s((size_t)r, 0);
        for (int d = 0; d < r; ++d) s[(size_t)d] = (uint8_t)(rng() & 1);
        bool dup = false;
        for (auto& t : sigs) if (t == s) { dup = true; break; }
        if (!dup) sigs.push_back(s);
    }
    std::vector<std::complex<double>> co((size_t)chi);
    double nrm = 0.0;
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int i = 0; i < chi; ++i) { co[(size_t)i] = {U(rng), U(rng)}; nrm += std::norm(co[(size_t)i]); }
    nrm = std::sqrt(nrm);
    L.sync_alpha_k();
    for (int i = 0; i < chi; ++i) L.entries().push_back({pack_key(sigs[(size_t)i], "test"), co[(size_t)i] / nrm});
    return L;
}

// Random frame-coordinate Pauli with X support on the free generators only.
Pauli random_w_pauli(const FramedSuperposition& L, bool hermitian) {
    const int n = L.n();
    Pauli Q(n);
    for (int f : L.free) if (rng() & 1) Q.setx(f);
    for (int a = 0; a < n; ++a) if (rng() & 1) Q.setz(a);
    if (hermitian) {
        int xz = 0;
        for (size_t w = 0; w < Q.x.size(); ++w) xz += __builtin_popcountll(Q.x[w] & Q.z[w]);
        Q.phase = (xz & 1) + 2 * (int)(rng() & 1);          // i^phase X^x Z^z Hermitian
    } else {
        Q.phase = (int)(rng() & 3);
    }
    return Q;
}

std::vector<Pauli> random_w_basis(const FramedSuperposition& L, int wdim, bool commuting) {
    std::vector<Pauli> N;
    int tries = 0;
    while ((int)N.size() < wdim && tries < 100000) {
        ++tries;
        Pauli Q = random_w_pauli(L, commuting);
        if (commuting) {
            bool ok = true;
            for (const Pauli& P : N) if (!Pauli::commute(P, Q)) { ok = false; break; }
            if (!ok) continue;
        }
        N.push_back(Q);
    }
    return N;
}


std::vector<Pauli> commuting_targets(const FramedSuperposition& L, int m, bool nonfree_x) {
    std::vector<Pauli> Q;
    int tries = 0;
    while ((int)Q.size() < m && tries < 200000) {
        ++tries;
        Pauli P = random_w_pauli(L, true);
        if (nonfree_x && (rng() % 3) == 0) {          // an X bit on a non-free generator (outside W)
            int a = rnd(0, L.n() - 1);
            bool isfree = false;
            for (int f : L.free) if (f == a) isfree = true;
            if (!isfree) {
                P.setx(a);
                int xz = 0;
                for (size_t w = 0; w < P.x.size(); ++w) xz += __builtin_popcountll(P.x[w] & P.z[w]);
                P.phase = (xz & 1) + 2 * (int)(rng() & 1);
            }
        }
        bool ok = true;
        for (const Pauli& R : Q) if (!Pauli::commute(R, P)) { ok = false; break; }
        if (ok) Q.push_back(P);
    }
    return Q;
}

std::map<Sig, double> oracle_law(const FramedSuperposition& L, const std::vector<Pauli>& Qf) {
    const int m = (int)Qf.size();
    std::vector<double> mu((size_t)1 << m);
    for (uint32_t S = 0; S < (1u << m); ++S) {
        Pauli A(L.n());
        for (int i = 0; i < m; ++i) if ((S >> i) & 1) A = Pauli::multiply(A, Qf[(size_t)i]);
        auto [pp, pm] = framed_expectation_frame(L, A);
        mu[S] = pp - pm;
    }
    std::map<Sig, double> P;
    for (uint32_t f = 0; f < (1u << m); ++f) {
        double s = 0;
        for (uint32_t S = 0; S < (1u << m); ++S) s += (__builtin_popcount(f & S) & 1 ? -1.0 : 1.0) * mu[S];
        s /= (double)(1u << m);
        if (s > 1e-13) {
            Sig g;
            for (int i = 0; i < m; ++i) if ((f >> i) & 1) g.push_back(i);
            P[g] = s;
        }
    }
    return P;
}

std::map<Sig, double> engine_law(const FaultLaw& F) {
    std::map<Sig, double> P;
    if (F.kind == FaultLaw::Kind::Support) {
        for (auto& e : F.sup) P[e.first] += e.second;
        return P;
    }
    const int v = (int)F.W.size();
    for (uint32_t z = 0; z < (1u << v); ++z) {
        Sig s = F.c;
        for (int j = 0; j < v; ++j) if ((z >> j) & 1) s = sig_xor(s, F.W[(size_t)j]);
        P[s] += std::ldexp(1.0, -v);
    }
    return P;
}

bool uniform_affine(const std::map<Sig, double>& P) {
    std::vector<Sig> pts;
    for (auto& e : P) pts.push_back(e.first);
    const int d = sig_affine_rank(pts);
    if ((size_t)1 << d != pts.size()) return false;
    for (auto& e : P) if (std::abs(e.second - 1.0 / (double)pts.size()) > 1e-12) return false;
    return true;
}

int n_general = 0, n_fast = 0, n_redet = 0;

void check(const FramedSuperposition& L, const std::vector<Pauli>& Qf) {
    const int m = (int)Qf.size();
    CorrelatedLaw law;
    law.nullv = correlated_law_fourier_support(L, Qf);
    FreeRegisterBornMemo memo;
    classify_correlated_law(L, Qf, law, memo);
    std::vector<int> rt;
    for (int i = 0; i < m; ++i) rt.push_back(i);
    FaultLaw F = correlated_fault_law(law, m, rt, Sig{}, 1.0);
    std::map<Sig, double> want = oracle_law(L, Qf), got = engine_law(F);
    double worst = 0;
    std::map<Sig, double> all = want;
    for (auto& e : got) all[e.first] += 0;
    for (auto& e : all) {
        const double a = want.count(e.first) ? want[e.first] : 0.0, b = got.count(e.first) ? got[e.first] : 0.0;
        worst = std::max(worst, std::abs(a - b));
    }
    CHECK(worst < 1e-12);
    if (worst >= 1e-12) std::fprintf(stderr, "  m=%d kind=%s worst=%.3g\n", m, law.coset ? "coset" : "general", worst);
    CHECK(law.coset == uniform_affine(want));
    if (!law.coset) ++n_general; else if (law.redetected) ++n_redet; else ++n_fast;
}

void test_hand_built_laws() {
    for (int trial = 0; trial < 400; ++trial) {
        const int r = rnd(1, 4);
        const int n = rnd(std::max(4, r + 2), 12);
        const int chi = rnd(1, 1 << r);
        FramedSuperposition L = random_state(n, r, chi);
        std::vector<Pauli> Qf = commuting_targets(L, rnd(1, 6), trial % 2 == 1);
        check(L, Qf);
    }
    std::printf("laws: general %d, coset fast path %d, coset re-detected %d\n", n_general, n_fast, n_redet);
    CHECK(n_general >= 50 && n_fast >= 20);
}

}  // namespace

int main() {
    RUN(test_hand_built_laws);
    REPORT();
}
