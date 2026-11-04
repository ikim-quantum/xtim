// twirl-routes T1 (G2): THE stabilizer-twirl record law (fault_record_law + born_law, qeccore/fault_record_law.hpp)
// on HAND-BUILT lean states, against a brute-force oracle. The DEM's per-fault law and the sampler's plan law are
// this one routine; the circuit-reachable DEM laws are uniform cosets (κ = 0 or a uniform Born quotient), so this
// test drives the κ > 0 path with states whose free-register Born values are FRACTIONAL.
//
// Setting: a random lean state (random eps, random free coordinates, χ random coefficients) under a random
// Clifford frame (random H / S / CX / CZ applied through the framed surface), its certified group G, a random
// diagonal+Pauli residual Dc = γ X^v diag(S^a) CZ^B, and m commuting record columns M_c: products of certified
// generators (channel columns) and of generators times Z-type free logicals (extra columns).
// Oracle: the raw record pattern y ∈ GF(2)^m of the reads M_c on Dc|ψ⟩ has Fourier transform
// ⟨Π_{c∈S} A_c⟩, A_c = Dc† M_c Dc (exact phase; conjugate_through), so P(y) = 2^-m Σ_S (−1)^{y·S} ⟨A_S⟩, every
// ⟨A_S⟩ one direct Born sum on the bare state — no group, no table, no twirl.
// Engine: nf = residual_normal_form(Dc) (and its canonical form mod G), the table over the columns, the Born law of
// its κ directions: y = β ⊕ Σ u_k A_k ⊕ Σ z_j B_j ⊕ ⟨P, M_c⟩ (u fair, z ~ born_law). Equal to 1e-12 (absolute).
// Mutation G4 (treat κ > 0 as κ = 0, i.e. drop the Born rows): RED here.
#include "check.hpp"

#include <complex>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "qeccore/dem_classify.hpp"
#include "qeccore/dem_decompose.hpp"
#include "qeccore/dem_export.hpp"
#include "qeccore/fault_record_law.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"

using namespace qeccore;

namespace {

std::mt19937_64 rng(20261030ull);
int rnd(int lo, int hi) { return lo + (int)(rng() % (uint64_t)(hi - lo + 1)); }

// Hand-built lean state: random eps, random distinct `free`, chi distinct sign patterns with random normalised
// coefficients, then a random Clifford frame.
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
    const int gates = rnd(0, 3 * n);
    for (int g = 0; g < gates; ++g) {
        const int kind = rnd(0, 3);
        const int a = rnd(0, n - 1);
        int b = rnd(0, n - 1);
        if (b == a) b = (a + 1) % n;
        if (kind == 0) L.apply_clifford(0, a, 0);                  // H
        else if (kind == 1) L.apply_clifford(1, a, 0);             // S
        else if (kind == 2) L.apply_clifford(6, a, b);             // CX
        else L.apply_clifford(7, a, b);                            // CZ
    }
    return L;
}

DiagPauliClifford random_residual(int n) {
    DiagPauliClifford D = DiagPauliClifford::identity(n);
    const int steps = rnd(1, 2 * n);
    for (int s = 0; s < steps; ++s) {
        const int q = rnd(0, n - 1);
        int t = rnd(0, n - 1);
        if (t == q) t = (q + 1) % n;
        switch (rnd(0, 5)) {
            case 0: D = D.then(DiagPauliClifford::S(n, q)); break;
            case 1: D = D.then(DiagPauliClifford::Sdg(n, q)); break;
            case 2: D = D.then(DiagPauliClifford::CZ(n, q, t)); break;
            case 3: D = D.then(DiagPauliClifford::X(n, q)); break;
            case 4: D = D.then(DiagPauliClifford::Z(n, q)); break;
            default: D = D.then(DiagPauliClifford::Y(n, q)); break;
        }
    }
    return D;
}

std::map<uint32_t, double> oracle_law(const FramedSuperposition& L, const DiagPauliClifford& Dc,
                                      const std::vector<Pauli>& M) {
    const int m = (int)M.size();
    std::vector<Pauli> A;
    for (const Pauli& P : M) A.push_back(conjugate_through(Dc, P));
    std::vector<double> mu((size_t)1 << m);
    for (uint32_t S = 0; S < (1u << m); ++S) {
        Pauli AS(L.n());
        for (int i = 0; i < m; ++i) if ((S >> i) & 1) AS = Pauli::multiply(AS, A[(size_t)i]);
        mu[S] = L.expectation(AS);
    }
    std::map<uint32_t, double> P;
    for (uint32_t y = 0; y < (1u << m); ++y) {
        double s = 0;
        for (uint32_t S = 0; S < (1u << m); ++S) s += (__builtin_popcount(y & S) & 1 ? -1.0 : 1.0) * mu[S];
        s /= (double)(1u << m);
        if (std::abs(s) > 1e-14) P[y] = s;
    }
    return P;
}

std::map<uint32_t, double> engine_law(const FramedSuperposition& L, const RecTable& T, const DiagNormalForm& nf,
                                      const std::vector<Pauli>& M, bool drop_born) {
    const int m = (int)M.size();
    auto word = [&](const std::vector<uint64_t>& row) { return row.empty() ? 0u : (uint32_t)row[0]; };
    uint32_t c = word(T.base);
    for (int i = 0; i < m; ++i) if (Pauli::anticommute_bit(nf.prefix, M[(size_t)i])) c ^= 1u << i;
    const std::vector<double> pz = drop_born ? std::vector<double>{1.0} : born_law(L, T.lam);
    std::map<uint32_t, double> P;
    for (size_t z = 0; z < pz.size(); ++z) {
        if (pz[z] == 0.0) continue;
        uint32_t cz = c;
        for (int j = 0; j < T.kappa && !drop_born; ++j) if ((z >> j) & 1) cz ^= word(T.born_row[(size_t)j]);
        for (uint32_t u = 0; u < (1u << T.rho); ++u) {
            uint32_t y = cz;
            for (int k = 0; k < T.rho; ++k) if ((u >> k) & 1) y ^= word(T.coin_row[(size_t)k]);
            P[y] += pz[z] * std::ldexp(1.0, -T.rho);
        }
    }
    return P;
}

int n_kappa = 0, n_frac = 0, n_laws = 0, n_extra = 0, n_mix = 0, n_mix_ok = 0;

Sig bits_sig(uint32_t x) {
    Sig g;
    for (int i = 0; x; ++i, x >>= 1) if (x & 1) g.push_back(i);
    return g;
}

// Review F4: the DEM glue's Born quotient (born_quotient — the classes of Σ z_j B_j mod the coin span, the uniform test)
// and the mixture record it emits (solve_mixture_classes), end to end against the brute-force oracle law.
void check_quotient(const FramedSuperposition& L, const RecTable& T, const DiagNormalForm& nf, const std::vector<Pauli>& M,
                    const std::map<uint32_t, double>& want) {
    const int m = (int)M.size();
    auto word = [&](const std::vector<uint64_t>& row) { return row.empty() ? 0u : (uint32_t)row[0]; };
    uint32_t c = word(T.base);
    for (int i = 0; i < m; ++i) if (Pauli::anticommute_bit(nf.prefix, M[(size_t)i])) c ^= 1u << i;
    auto one = [](uint32_t x) { return std::vector<uint64_t>{(uint64_t)x}; };
    std::vector<std::vector<uint64_t>> W, B;
    for (const auto& r : T.coin_row) W.push_back(one(word(r)));
    for (const auto& r : T.born_row) B.push_back(one(word(r)));
    const BornQuotient q = born_quotient(W, B, born_law(L, T.lam), 1);
    std::map<uint32_t, double> got;
    auto add_coset = [&](uint32_t c0, const std::vector<uint32_t>& dir, double w) {
        for (uint32_t u = 0; u < (1u << dir.size()); ++u) {
            uint32_t y = c0;
            for (size_t k = 0; k < dir.size(); ++k) if ((u >> k) & 1) y ^= dir[k];
            got[y] += w * std::ldexp(1.0, -(int)dir.size());
        }
    };
    std::vector<uint32_t> wd;
    for (const auto& r : W) wd.push_back((uint32_t)r[0]);
    if (q.coset) {
        std::vector<uint32_t> d2 = wd;
        for (const auto& r : q.dirs) d2.push_back((uint32_t)r[0]);
        add_coset(c ^ (uint32_t)q.r0[0], d2, 1.0);
    } else {
        ++n_mix;
        for (const auto& [off, w] : q.cls) add_coset(c ^ (uint32_t)off[0], wd, w);
    }
    double worst = 0;
    std::map<uint32_t, double> all = want;
    for (auto& e : got) all[e.first] += 0;
    for (auto& e : all) {
        const double a = want.count(e.first) ? want.at(e.first) : 0.0, b = got.count(e.first) ? got.at(e.first) : 0.0;
        worst = std::max(worst, std::abs(a - b));
    }
    CHECK(worst < 1e-12);
    if (q.coset) return;
    // the emitted record: the Mixture FaultLaw the DEM builds, through its own solver, vs the forward model
    FaultLaw F;
    F.kind = FaultLaw::Kind::Mixture;
    F.p = 0.01;
    for (uint32_t w : wd) F.W.push_back(bits_sig(w));
    for (const auto& [off, w] : q.cls) F.sup.push_back({bits_sig(c ^ (uint32_t)off[0]), w});
    std::vector<Mech> ms;
    double dropped = 0.0;
    int v = 0;
    Sig supp;
    const SolveStatus st = solve_mixture_classes(F, 1000, -1, ms, dropped, v, supp);
    std::map<Sig, double> fwd, ref;
    for (auto& [y, w] : want) ref[bits_sig(y)] += F.p * w;
    ref[Sig{}] += 1.0 - F.p;
    if (st == SolveStatus::Ok) {
        ++n_mix_ok;
        fwd[Sig{}] = 1.0;
        for (const Mech& mm : ms) {
            std::map<Sig, double> N2;
            for (auto& [sg, w] : fwd) { N2[sg] += w * (1 - mm.p); N2[sig_xor(sg, mm.sig)] += w * mm.p; }
            fwd.swap(N2);
        }
    } else {
        CHECK(st == SolveStatus::Negative);
        fwd[Sig{}] = 1.0 - F.p;                           // the disjoint law (first order)
        for (const Mech& mm : ms) fwd[mm.sig] += mm.p;
        fwd[Sig{}] += ref.count(Sig{}) ? ref[Sig{}] - (1.0 - F.p) : 0.0;
    }
    double w2 = 0;
    std::map<Sig, double> keys = ref;
    for (auto& e : fwd) keys[e.first] += 0;
    for (auto& e : keys) w2 = std::max(w2, std::abs((fwd.count(e.first) ? fwd[e.first] : 0.0) -
                                                     (ref.count(e.first) ? ref[e.first] : 0.0)));
    CHECK(w2 < 1e-12);
}

void check_one(const FramedSuperposition& L, const CertifiedGroupPlanes& G, const RecordColumns& RC,
               const DiagPauliClifford& Dc, const std::vector<Pauli>& M, bool canon) {
    std::vector<RecColumn> cols;
    for (int i = 0; i < (int)M.size(); ++i) {
        cols.push_back(RC.column(M[(size_t)i], i));
        if (!cols.back().channel) ++n_extra;
    }
    DiagNormalForm nf = residual_normal_form(Dc);
    if (canon) canonicalize_mod_stabilizers_inplace(nf, G);
    RecTable T;
    CHECK(fault_record_law(G, L, nf, cols, T));
    const std::map<uint32_t, double> want = oracle_law(L, Dc, M);
#ifdef TWR_MUTATE_G4
    const std::map<uint32_t, double> got = engine_law(L, T, nf, M, true);   // mutation: κ > 0 treated as κ = 0
#else
    const std::map<uint32_t, double> got = engine_law(L, T, nf, M, false);
#endif
    double worst = 0;
    std::map<uint32_t, double> all = want;
    for (auto& e : got) all[e.first] += 0;
    for (auto& e : all) {
        const double a = want.count(e.first) ? want.at(e.first) : 0.0, b = got.count(e.first) ? got.at(e.first) : 0.0;
        worst = std::max(worst, std::abs(a - b));
    }
    CHECK(worst < 1e-12);
    if (worst >= 1e-12)
        std::fprintf(stderr, "  m=%zu rho=%d kappa=%d worst=%.3g canon=%d\n", M.size(), T.rho, T.kappa, worst, (int)canon);
    ++n_laws;
#ifndef TWR_MUTATE_G4
    check_quotient(L, T, nf, M, want);
#endif
    if (T.kappa > 0) {
        ++n_kappa;
        for (auto& e : want) if (e.second > 1e-9 && std::abs(e.second * (double)(1u << T.rho) * 4.0 -
                                                            std::round(e.second * (double)(1u << T.rho) * 4.0)) > 1e-6) {
            ++n_frac;
            break;
        }
    }
}

void test_hand_built_laws() {
    for (int trial = 0; trial < 3000; ++trial) {
        const int r = rnd(1, 3);
        const int n = rnd(r + 2, 7);
        const int chi = rnd(1, 1 << r);
        FramedSuperposition L = random_state(n, r, chi);
        const std::vector<Pauli> gens = L.certified_stabilizers();
        const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(gens, n);
        const RecordColumns RC(G);
        // columns: products of generators, some times a Z-type free logical (commuting extras)
        std::vector<Pauli> M;
        const int m = rnd(1, 6);
        for (int c = 0; c < m; ++c) {
            Pauli P(n);
            for (const Pauli& g : gens) if (rng() & 1) P = Pauli::multiply(P, g);
            if (trial % 3 == 0 && (rng() & 1))
                for (int f : L.free) if (rng() & 1) P = Pauli::multiply(P, L.U.Zrow[(size_t)f]);
            M.push_back(P);
        }
        const DiagPauliClifford Dc = random_residual(n);
        check_one(L, G, RC, Dc, M, false);
        check_one(L, G, RC, Dc, M, true);
    }
    std::printf("laws %d: kappa > 0 %d (fractional %d), extra columns %d, Born mixtures %d (%d split exactly)\n", n_laws,
                n_kappa, n_frac, n_extra, n_mix, n_mix_ok);
    CHECK(n_kappa >= 100 && n_frac >= 20 && n_extra >= 100 && n_mix >= 100 && n_mix_ok >= 10);
}

// The DEM consumes the routine's Born rows: a fault whose only randomization is a kappa = 1 direction with
// <Z_T> = 0 (review M1's T-type case) is a uniform v = 1 coset at p / 2, not a point law. Mutation G4 in the DEM
// glue (Born rows dropped / kappa > 0 treated as kappa = 0): RED here.
void test_dem_consumes_born_rows() {
    const std::string tz0 = "RX 0\nT 0\nRX 1\nCS 1 0\nX_ERROR(0.01) 1\nCS_DAG 1 0\nMX 1\nDETECTOR rec[-1]\n";
    DemLawCensus cen;
    const DemExportResult r = export_dem(tz0, true, false, false, false, nullptr, nullptr, 0.0,
                                         kDemCliffordCutoffExact, &cen);
    CHECK(r.ok);
    CHECK(r.dem.find("# clifford f0: p=0.01 v=1 offset=0 support=D0") != std::string::npos);
    CHECK(r.dem.find("error[clifford:f0](0.0050000000000000001) D0\n") != std::string::npos);
    CHECK(cen.inst.size() == 1 && cen.inst[0].comps.size() == 1 && cen.inst[0].comps[0].kappa == 1 &&
          cen.inst[0].comps[0].kind == 'C' && cen.born_sums == 3);   // 1 node + 2 leaves
}

}  // namespace

int main() {
    RUN(test_dem_consumes_born_rows);
    RUN(test_hand_built_laws);
    REPORT();
}
