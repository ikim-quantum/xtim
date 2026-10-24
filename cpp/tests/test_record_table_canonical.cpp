// test_record_table_canonical.cpp — S2b unit gate: canonicalize_record_order preserves every
// record's value as a function of the sources, signs included (mutations M-BETA, M-SIGN, M-C).
//
// Random affine tables over ncoin coin sources and k commuting Hermitian directions with RANDOM
// signs (Z-type free operators (−1)^s Z^b, so every product commutes and its outcome bit is the XOR of
// the factors' plus the sign) and random betas — including introducing columns whose folded β' is 1,
// a case no circuit fixture reaches. For every source assignment (coins c, direction outcomes w):
//   input value    y_c = A_c·c ⊕ B_c·w ⊕ β_c,
//   canonical      u_k = y_{pivot_rec[k]}, z_j = outcome of Λ_j (its literal sign included),
//                  y_c = A'_c·u ⊕ B'_c·z ⊕ β'_c
// must agree on every column, and Lemma 4's structure (pivots / introducers in record order) holds.
#include "check.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "qeccore/record_table.hpp"

using namespace qeccore;

namespace {

bool bit(const std::vector<uint64_t>& v, int i) {
    return (size_t)(i >> 6) < v.size() && ((v[(size_t)(i >> 6)] >> (i & 63)) & 1ULL);
}
void setb(std::vector<uint64_t>& v, int i) {
    if ((size_t)(i >> 6) >= v.size()) v.resize((size_t)(i >> 6) + 1, 0);
    v[(size_t)(i >> 6)] |= 1ULL << (i & 63);
}

// Outcome bit of a Z-type effective operator i^ph (−1)^par Z^b on the computational outcome w (a
// Z-eigenbasis label of the free register: Z_i has eigenvalue (−1)^{w_i}).
int outcome(const FreeDir& d, const std::vector<int>& wbits) {
    int o = 0;
    for (size_t i = 0; i < d.b.size(); ++i) if (d.b[i]) o ^= wbits[i];
    const int ph = fd_eff_phase(d);
    if (ph & 1) return -1;                   // non-Hermitian: refused by the canonical pass
    return o ^ (ph == 2 ? 1 : 0);
}

void test_canonical_values_equal_input_values() {
    std::mt19937_64 rng(20261017);
    std::uniform_int_distribution<int> bitd(0, 1);
    long trials = 0, folded = 0, checked = 0, small_checked = 0;
    for (int it = 0; it < 4000; ++it) {
        const int m = 2 + it % 6, ncoin = it % 3, k = 1 + it % 4;
        AffineTable in;
        in.m = m;
        in.ncoin = ncoin;
        // k directions on a k-coordinate register: Z_i (independent), random signs
        for (int i = 0; i < k; ++i) {
            FreeDir d = fd_identity(k);
            d.b[(size_t)i] = 1;
            d.phase = bitd(rng) ? 2 : 0;
            in.dirs.push_back(d);
        }
        in.A.assign((size_t)m, {});
        in.B.assign((size_t)m, {});
        in.beta.assign((size_t)m, 0);
        for (int c = 0; c < m; ++c) {
            for (int s = 0; s < ncoin; ++s) if (bitd(rng)) setb(in.A[(size_t)c], s);
            for (int j = 0; j < k; ++j) if (bitd(rng)) setb(in.B[(size_t)c], j);
            in.beta[(size_t)c] = (uint8_t)bitd(rng);
        }
        std::vector<int> order((size_t)m);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        RecTable T;
        canonicalize_record_order(in, order, T);
        ++trials;
        {   // r8 (final review MINOR-2): the one-word form runs the SAME implementation — field-identical
            AffineTable s = in;
            s.small = true;
            s.A64.assign((size_t)m, 0);
            s.B64.assign((size_t)m, 0);
            for (int col = 0; col < m; ++col) {
                for (int q = 0; q < ncoin; ++q) if (bit(in.A[(size_t)col], q)) s.A64[(size_t)col] |= 1ULL << q;
                for (int j = 0; j < k; ++j) if (bit(in.B[(size_t)col], j)) s.B64[(size_t)col] |= 1ULL << j;
            }
            RecTable T2;
            canonicalize_record_order(s, order, T2);
            CHECK(T2.pivot_rec == T.pivot_rec && T2.born_rec == T.born_rec && T2.base == T.base &&
                  T2.coin_row == T.coin_row && T2.born_row == T.born_row && T2.folds == T.folds &&
                  T2.lam.size() == T.lam.size());
            for (size_t j = 0; j < T.lam.size() && j < T2.lam.size(); ++j)
                CHECK(T2.lam[j].a == T.lam[j].a && T2.lam[j].b == T.lam[j].b &&
                      T2.lam[j].phase == T.lam[j].phase && T2.lam[j].par == T.lam[j].par);
            ++small_checked;
        }
        // count introducers whose folded beta' is 1 (the M-BETA case)
        for (int j = 0; j < T.kappa; ++j) {
            const FreeDir& L = T.lam[(size_t)j];
            int prod_sign = 0;   // the sign of the plain literal product of the introducer's inputs
            FreeDir P = fd_identity(k);
            for (int i = 0; i < k; ++i) if (L.b[(size_t)i]) P = fd_mul(P, in.dirs[(size_t)i]);
            prod_sign = fd_eff_phase(P);
            if (((fd_eff_phase(L) - prod_sign) & 3) == 2) ++folded;
        }
        for (int cm = 0; cm < (1 << ncoin); ++cm)
            for (int wm = 0; wm < (1 << k); ++wm) {
                std::vector<int> c((size_t)ncoin), w((size_t)k);
                for (int s = 0; s < ncoin; ++s) c[(size_t)s] = (cm >> s) & 1;
                for (int i = 0; i < k; ++i) w[(size_t)i] = (wm >> i) & 1;
                // input values; dir outcome of input j = outcome(in.dirs[j], w)
                std::vector<int> yin((size_t)m);
                for (int col = 0; col < m; ++col) {
                    int v = in.beta[(size_t)col];
                    for (int s = 0; s < ncoin; ++s) if (bit(in.A[(size_t)col], s)) v ^= c[(size_t)s];
                    for (int j = 0; j < k; ++j) if (bit(in.B[(size_t)col], j)) v ^= outcome(in.dirs[(size_t)j], w);
                    yin[(size_t)col] = v;
                }
                std::vector<int> u((size_t)T.rho), z((size_t)T.kappa);
                for (int s = 0; s < T.rho; ++s) u[(size_t)s] = yin[(size_t)T.pivot_rec[(size_t)s]];
                for (int j = 0; j < T.kappa; ++j) {
                    z[(size_t)j] = outcome(T.lam[(size_t)j], w);
                    CHECK(z[(size_t)j] >= 0);
                }
                for (int col = 0; col < m; ++col) {
                    int v = bit(T.base, col) ? 1 : 0;
                    for (int s = 0; s < T.rho; ++s) if (bit(T.coin_row[(size_t)s], col)) v ^= u[(size_t)s];
                    for (int j = 0; j < T.kappa; ++j) if (bit(T.born_row[(size_t)j], col)) v ^= z[(size_t)j];
                    CHECK_EQ(v, yin[(size_t)col]);
                    ++checked;
                }
            }
    }
    std::fprintf(stderr, "[record_table_canonical] trials=%ld folded-sign introducers=%ld values checked=%ld\n",
                 trials, folded, checked);
    CHECK(folded > 500);   // non-vacuity: the folded-beta case is exercised
    CHECK(small_checked == trials);
}

}  // namespace

int main() {
    RUN(test_canonical_values_equal_input_values);
    REPORT();
}
