// PeShotAnalyzer's per-alternative split (magic lane M3 / H5): alternative(J, read_pauli) does the column-independent
// half of structure() ONCE (the projection of W onto J, the J-row elimination, the free-sector basis W1); column(alt,
// X) adds only the column's coin / S0 / X0. Oracle: structure_reference, the pre-split structure() kept verbatim
// (every column recomputes the alternative). Every Law field must agree exactly on random lean states, random reads
// and random diagonal+Pauli faults, several columns per alternative, columns in random order.
#include "check.hpp"

#include <complex>
#include <random>
#include <vector>

#include "qeccore/dem_classify.hpp"
#include "qeccore/dem_pe_per_shot.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli_kernels.hpp"

using namespace qeccore;

namespace {

std::mt19937_64 rng(20261121ull);
int rnd(int lo, int hi) { return lo + (int)(rng() % (uint64_t)(hi - lo + 1)); }

// Hand-built lean state (as test_dem_general_law): random eps, random distinct `free`, chi distinct sign patterns with
// random normalised coefficients, then a random Clifford frame.
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

DiagPauliClifford random_residual(int n, int steps) {
    DiagPauliClifford D = DiagPauliClifford::identity(n);
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

bool same(const Pauli& a, const Pauli& b) { return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z; }

bool same_law(const PeShotAnalyzer::Law& a, const PeShotAnalyzer::Law& b) {
    if (a.zero != b.zero || a.coin != b.coin || a.S0 != b.S0 || a.d != b.d || a.T != b.T) return false;
    if (a.too_large != b.too_large || a.yP.size() != b.yP.size()) return false;
    if (!same(a.Xw, b.Xw) || !same(a.coinP, b.coinP) || !same(a.X0, b.X0)) return false;
    for (size_t i = 0; i < a.yP.size(); ++i) if (!same(a.yP[i], b.yP[i])) return false;
    return true;
}

int n_laws = 0, n_zero = 0, n_coin = 0, n_d = 0, n_alts_j = 0, n_multi = 0;

void run_case(int n, int r, int chi, int M, int steps) {
    const FramedSuperposition L = random_state(n, r, chi);
    std::vector<int> wires(n);
    for (int a = 0; a < n; ++a) wires[(size_t)a] = a;
    for (int a = n - 1; a > 0; --a) std::swap(wires[(size_t)a], wires[(size_t)rnd(0, a)]);
    std::vector<std::pair<int, int>> reads;
    for (int j = 0; j < M; ++j) reads.push_back({rnd(0, 2), wires[(size_t)j]});
    PeShotAnalyzer pe(L, reads);
    const DiagPauliClifford D = steps ? random_residual(n, steps) : DiagPauliClifford::identity(n);
    std::vector<Pauli> A;
    std::vector<int> J;
    for (int j = 0; j < M; ++j) {
        const Pauli R = single_pauli(reads[(size_t)j].first, reads[(size_t)j].second, n);
        A.push_back(conjugate_through(D, R));
        if (A.back().x != R.x || A.back().z != R.z) J.push_back(j);
    }
    if (!J.empty()) ++n_alts_j;
    const std::function<Pauli(int)> read_pauli = [&](int j) { return A[(size_t)j]; };
    // Columns: random Paulis restricted to I / the read basis on every read wire (so D^dag P D commutes with each A_j).
    const int ncol = rnd(1, 6);
    std::vector<Pauli> cols;
    for (int c = 0; c < ncol; ++c) {
        Pauli P(n);
        for (int q = 0; q < n; ++q) {
            int b = rnd(0, 3);                                     // 0 = I, else single_pauli's basis b - 1
            for (const auto& [rb, rq] : reads) if (rq == q && b) b = rnd(0, 1) ? rb + 1 : 0;
            if (b) pmul_into(P, single_pauli(b - 1, q, n));
        }
        // Hermitian: single-qubit factors on distinct wires commute, so the product of Hermitian Paulis is Hermitian.
        cols.push_back(conjugate_through(D, P));
    }
    PeShotAnalyzer::Alt alt = pe.alternative(J, read_pauli);
    if (ncol > 1) ++n_multi;
    for (int c = 0; c < ncol; ++c) {
        const Pauli& X = cols[(size_t)c];
        const PeShotAnalyzer::Law ref = pe.structure_reference(J, read_pauli, X);
        const PeShotAnalyzer::Law got = pe.column(alt, X);
        PeShotAnalyzer::Alt fresh = pe.alternative(J, read_pauli);           // a fresh Alt per column
        const PeShotAnalyzer::Law one = pe.column(fresh, X);
        CHECK(same_law(ref, got));
        CHECK(same_law(ref, one));
        ++n_laws;
        if (ref.zero) ++n_zero;
        if (ref.coin) ++n_coin;
        if (ref.d > 0) ++n_d;
    }
}

void test_split_matches_reference() {
    for (int t = 0; t < 2000; ++t) {
        const int n = rnd(3, 12);
        const int r = rnd(0, std::min(n, 4));
        const int chi = rnd(1, std::min(1 << r, 6));
        const int M = t % 2 ? rnd(1, n) : n;                       // all wires read: many dependent reads
        run_case(n, r, chi, M, t % 5 == 0 ? 0 : rnd(1, 2 * n));
    }
    // wide registers: rows span several 64-bit words
    for (int t = 0; t < 40; ++t) {
        const int n = rnd(65, 140);
        const int r = rnd(1, 5);
        const int chi = rnd(1, std::min(1 << r, 6));
        run_case(n, r, chi, rnd(60, n), rnd(1, 40));
    }
    std::fprintf(stderr, "laws %d (zero %d, coin %d, d>0 %d), alternatives with J %d, multi-column %d\n", n_laws,
                 n_zero, n_coin, n_d, n_alts_j, n_multi);
    CHECK(n_zero > 0);
    CHECK(n_coin > 0);
    CHECK(n_d > 0);
    CHECK(n_alts_j > 0);
    CHECK(n_multi > 0);
}

}  // namespace

int main() {
    RUN(test_split_matches_reference);
    REPORT();
}
