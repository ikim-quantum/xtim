// Oracle test for the DEM's factored correlated-read law (dem_export.cpp, perf/dem-factored-law).
//
// The exact correlated-read law of a randomized-read fault needs the coset weights
//     nu = WHT(mu),   mu[c] = <A_{S_c}>,   A_{S_c} = Gray-code product of the W-basis Paulis N
// (frame coordinates, no non-free X part). Before the factoring (fd4f31c) every mu[c] was ONE Born sum
// (framed_expectation_frame per subset: up to 2^20 per law). correlated_law_coset_weights evaluates
// the same mu through the free-register projection of each A_S (its phase, its X/Z bits on the
// `free` generators, and the eps-sign of its non-free Z part carried by ONE representative
// non-free Z bit), memoised per distinct projection: at most 8 * 4^{r_free} Born sums per export.
// The ORACLE below is the pre-factoring walk (fd4f31c), verbatim. The contract is BIT identity of nu (memcmp), not
// closeness: the printed DEM probabilities must not move by an ulp.
//
// Corpus: hand-built lean states (r_free 0..4, chi 1..16, n 3..70, random eps/free/coefficients)
// and bare states of random Clifford+T circuits (build_bare_state_framed), with W bases of
// dimension 0..12 made of random frame-coordinate Paulis (half the trials: commuting Hermitian
// sets, as the engine produces; half: arbitrary phases, the kernel's full domain — these caught
// the first draft, which folded the sign into the phase: exact for Hermitian A_S only, since the
// kernel's no-X case reads an odd phase as +1 while still applying the eps sign). Mutation check
// (done by hand, 2026-10-12): dropping the non-free eps sign from the projection makes this test
// fail (see the report).
#include "check.hpp"

#include <complex>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qeccore/dem_export.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

std::mt19937_64 rng(20261012ull);
int rnd(int lo, int hi) { return lo + (int)(rng() % (uint64_t)(hi - lo + 1)); }

// ── The pre-factoring walk, verbatim (dem_export.cpp fd4f31c l.1406-1424). ─────────────────────────────
std::vector<double> oracle_walk(const FramedSuperposition& L, const std::vector<Pauli>& N) {
    const int wdim = (int)N.size();
    const uint32_t WG = 1u << wdim;
    std::vector<double> nu(WG, 0.0);
    {
        Pauli AS(L.n());
        for (uint32_t g = 0;; ++g) {
            const uint32_t c = g ^ (g >> 1);
            auto [pp, pm] = framed_expectation_frame(L, AS);
            nu[c] = pp - pm;
            if (g + 1 == WG) break;
            const int i = __builtin_ctz(g + 1);
            AS = Pauli::multiply(AS, N[(size_t)i]);
        }
    }
    for (int h = 1; h < (int)WG; h <<= 1)
        for (uint32_t i = 0; i < WG; i += 2 * h)
            for (uint32_t j = i; j < i + (uint32_t)h; ++j) {
                const double x = nu[j], y = nu[j + h];
                nu[j] = x + y; nu[j + h] = x - y;
            }
    return nu;
}

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

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

long trials_run = 0, born_sums_oracle = 0;

void check_state(const FramedSuperposition& L, int max_wdim, int reps) {
    FreeRegisterBornMemo shared;                 // one memo across every law of this state
    for (int rep = 0; rep < reps; ++rep) {
        const int wdim = rnd(0, max_wdim);
        const bool commuting = (rep & 1) == 0;
        std::vector<Pauli> N = random_w_basis(L, wdim, commuting);
        std::vector<double> want = oracle_walk(L, N);
        born_sums_oracle += (long)want.size();
        std::vector<double> got;
        correlated_law_coset_weights(L, N, got, shared);
        CHECK(same_bits(want, got));
        FreeRegisterBornMemo fresh;               // and with a cold memo
        std::vector<double> got2;
        correlated_law_coset_weights(L, N, got2, fresh);
        CHECK(same_bits(want, got2));
        ++trials_run;
    }
    // <= 8 * 4^{r_free} distinct projections (born_sums also counts unprojected non-free-X
    // evaluations; random_w_pauli never builds those, nor does the engine). Vacuous past r = 29.
    if (2 * L.k() < 60) CHECK(shared.born_sums <= 8L * (1L << (2 * L.k())));
}

void test_hand_built_states() {
    for (int trial = 0; trial < 60; ++trial) {
        const int r = rnd(0, 4);
        const int n = rnd(std::max(3, r), trial % 3 == 0 ? 70 : 12);
        const int chi = rnd(1, 1 << r);
        FramedSuperposition L = random_state(n, r, chi);
        check_state(L, trial % 5 == 0 ? 12 : 8, 6);
    }
}

// The kernel's hash-partner branch (chi >= partner_hash_chi_min() = 64) and the wide memo key
// (r_free > 30): r_free = 7 with chi 64 and 128, r_free = 33 with chi 1, 2 and 64.
void test_hash_partner_and_wide_key() {
    for (int chi : {64, 128}) {
        FramedSuperposition L = random_state(rnd(9, 40), 7, chi);
        check_state(L, 10, 4);
    }
    for (int chi : {1, 2, 64}) {
        FramedSuperposition L = random_state(rnd(33, 80), 33, chi);
        check_state(L, 10, 4);
    }
}

// Random Clifford+T circuits -> real bare states (frame, eps, free, branch coefficients).
std::string random_magic_circuit(int n, int nT) {
    static const char* one[] = {"H", "S", "S_DAG", "X", "Z", "SQRT_X", "SQRT_Y"};
    std::string s;
    for (int q = 0; q < n; ++q) s += ((rng() & 1) ? "RX " : "R ") + std::to_string(q) + "\n";
    auto cliff = [&](int m) {
        for (int i = 0; i < m; ++i) {
            if ((rng() & 1) || n < 2) s += std::string(one[rng() % 7]) + " " + std::to_string(rnd(0, n - 1)) + "\n";
            else {
                int a = rnd(0, n - 1), b = rnd(0, n - 2); if (b >= a) ++b;
                s += std::string((rng() & 1) ? "CX " : "CZ ") + std::to_string(a) + " " + std::to_string(b) + "\n";
            }
        }
    };
    cliff(rnd(1, 3 * n));
    std::vector<int> perm(n);
    for (int q = 0; q < n; ++q) perm[(size_t)q] = q;
    for (int a = n - 1; a > 0; --a) std::swap(perm[(size_t)a], perm[(size_t)rnd(0, a)]);
    for (int t = 0; t < nT && t < n; ++t) s += std::string((rng() & 1) ? "T " : "T_DAG ") + std::to_string(perm[(size_t)t]) + "\n";
    cliff(rnd(0, 2 * n));
    s += "M";
    for (int q = 0; q < n; ++q) s += " " + std::to_string(q);
    s += "\n";
    return s;
}

void test_circuit_states() {
    int built = 0;
    for (int trial = 0; trial < 40 && built < 24; ++trial) {
        const int n = rnd(3, 8), nT = rnd(1, 4);
        ParsedStim ps = parse_stim_circuit(random_magic_circuit(n, nT));
        if (!ps.ok()) continue;
        NormalizePolicy np;
        np.feedback = NormalizePolicy::Feedback::KeepCoherent;
        np.defer = true;
        NormalizeResult nr = normalize(ps.circuit, np);
        if (nr.feedback_rejected) continue;
        FramedBareState b = build_bare_state_framed(nr.normalized);
        if (b.rejected) continue;
        ++built;
        check_state(b.state, 10, 6);
    }
    CHECK(built >= 12);
}

}  // namespace

int main() {
    RUN(test_hand_built_states);
    RUN(test_hash_partner_and_wide_key);
    RUN(test_circuit_states);
    std::fprintf(stderr, "trials %ld (oracle Born sums %ld)\n", trials_run, born_sums_oracle);
    CHECK(trials_run >= 300);
    REPORT();
}
