// test_fast_todd_wide.cpp — the 3.1.8 lift of FastTODD's r <= 30 limit.
//
//   1. fast_todd_reduce_wide (word-vector columns) returns EXACTLY fast_todd_reduce's columns, in
//      order, on random r <= 30 tables (the uint32 path is the pre-3.1.8 production path).
//   2. even_residue_symbolic (polynomial; production) equals even_residue_dense (the pre-3.1.8
//      dense 2^r tables + Möbius, kept as the ORACLE) monomial for monomial — including the first
//      violation when g is not even-graded — on random instances, and on FastTODD outputs where g
//      must be even-graded.
//   3. FastTODD at r > 30 (wide only): the reduction keeps the cubic signature (even_residue_symbolic
//      of the input vs the output is even-graded) and does not increase the T-count.
#include "check.hpp"

#include <cstdint>
#include <random>
#include <vector>

#include "qeccore/fast_todd.hpp"
#include "qeccore/residue_projection.hpp"

using namespace qeccore;

namespace {

WideCol widen(uint32_t m, int r) {
    WideCol w((size_t)((r + 63) / 64), 0);
    w[0] = m;
    return w;
}

std::vector<int> on_of(uint32_t m, int r) {
    std::vector<int> on;
    for (int i = 0; i < r; ++i) if ((m >> i) & 1u) on.push_back(i);
    return on;
}

std::vector<int> on_of(const WideCol& m, int r) {
    std::vector<int> on;
    for (int i = 0; i < r; ++i) if ((m[(size_t)(i >> 6)] >> (i & 63)) & 1ull) on.push_back(i);
    return on;
}

bool same(const EvenResidue& a, const EvenResidue& b) {
    if (a.ok != b.ok) return false;
    if (!a.ok) return a.bad_deg == b.bad_deg && a.bad_coeff == b.bad_coeff && a.bad_on == b.bad_on;
    if (a.mono.size() != b.mono.size()) return false;
    for (size_t i = 0; i < a.mono.size(); ++i)
        if (a.mono[i].hi != b.mono[i].hi || a.mono[i].lo != b.mono[i].lo || a.mono[i].a != b.mono[i].a)
            return false;
    return true;
}

std::vector<uint32_t> random_table(std::mt19937_64& rng, int r, int t, int maxw) {
    std::vector<uint32_t> tab;
    for (int i = 0; i < t; ++i) {
        uint32_t m = 0;
        const int w = 1 + (int)(rng() % (uint64_t)maxw);
        for (int k = 0; k < w; ++k) m |= 1u << (rng() % (uint64_t)r);
        tab.push_back(m);
    }
    return tab;
}

void wide_equals_uint32() {
    std::mt19937_64 rng(20261009);
    int cases = 0, reduced = 0;
    for (int it = 0; it < 300; ++it) {
        const int r = 2 + (int)(rng() % 29);                  // 2..30
        const int t = 1 + (int)(rng() % 40);
        std::vector<uint32_t> tab = random_table(rng, r, t, std::min(r, 6));
        std::vector<WideCol> wt;
        for (uint32_t m : tab) wt.push_back(widen(m, r));
        std::vector<uint32_t> a = fast_todd_reduce(tab, r);
        std::vector<WideCol> b = fast_todd_reduce_wide(wt, r);
        CHECK_EQ(a.size(), b.size());
        bool eq = a.size() == b.size();
        for (size_t i = 0; eq && i < a.size(); ++i) eq = (widen(a[i], r) == b[i]);
        CHECK(eq);
        ++cases;
        if (a.size() < tab.size()) ++reduced;
    }
    CHECK(reduced > 50);                                       // the cases actually exercise reduction
    std::fprintf(stderr, "  wide==uint32 on %d tables (%d reduced)\n", cases, reduced);
}

void symbolic_equals_dense() {
    std::mt19937_64 rng(77);
    int n_ok = 0, n_bad = 0;
    for (int it = 0; it < 600; ++it) {
        const int r = 1 + (int)(rng() % 14);                   // dense oracle: 2^r tables
        std::vector<ResidueColumn> f, fp;
        const int nf = (int)(rng() % 12), nfp = (int)(rng() % 12);
        for (int i = 0; i < nf; ++i) {
            uint32_t m = (uint32_t)(rng() & ((1ull << r) - 1)); if (!m) m = 1;
            f.push_back({on_of(m, r), (int)(rng() % 16) - 8});
        }
        for (int i = 0; i < nfp; ++i) {
            uint32_t m = (uint32_t)(rng() & ((1ull << r) - 1)); if (!m) m = 1;
            fp.push_back({on_of(m, r), 1});
        }
        EvenResidue s = even_residue_symbolic(r, f, fp), d = even_residue_dense(r, f, fp);
        CHECK(same(s, d));
        (s.ok ? n_ok : n_bad)++;
    }
    // Even-graded instances from FastTODD itself: inputs at odd ζ8 coefficients, red at +1.
    for (int it = 0; it < 200; ++it) {
        const int r = 3 + (int)(rng() % 12);
        std::vector<uint32_t> tab = random_table(rng, r, 4 + (int)(rng() % 30), r);
        std::vector<uint32_t> red = fast_todd_reduce(tab, r);
        // every input (repeats included) is a coeff-1 T: ζ8 coefficient +1 per column
        std::vector<ResidueColumn> f, fp;
        for (uint32_t m : tab) if (m) f.push_back({on_of(m, r), 1});
        for (uint32_t m : red) if (m) fp.push_back({on_of(m, r), 1});
        EvenResidue s = even_residue_symbolic(r, f, fp), d = even_residue_dense(r, f, fp);
        CHECK(same(s, d));
        CHECK(s.ok);                                            // FastTODD preserves the cubic part
        (s.ok ? n_ok : n_bad)++;
    }
    CHECK(n_ok > 150 && n_bad > 150);
    std::fprintf(stderr, "  symbolic==dense: %d even-graded, %d violations\n", n_ok, n_bad);
}

void wide_reduces_above_30() {
    std::mt19937_64 rng(31337);
    for (int it = 0; it < 12; ++it) {
        const int r = 31 + (int)(rng() % 60);                  // 31..90
        const int t = 10 + (int)(rng() % 60);
        std::vector<WideCol> tab;
        for (int i = 0; i < t; ++i) {
            WideCol m((size_t)((r + 63) / 64), 0);
            const int w = 1 + (int)(rng() % 5);
            for (int k = 0; k < w; ++k) { int q = (int)(rng() % (uint64_t)r); m[(size_t)(q >> 6)] |= 1ull << (q & 63); }
            tab.push_back(m);
        }
        // A planted reducible structure: the 15-to-1 style triple-overlap (a, b, c, a^b, ...) on
        // fresh bits makes sure the reduction has something to remove.
        {
            const int a = (int)(rng() % (uint64_t)r), b = (a + 1) % r, c = (a + 2) % r;
            const int words = (r + 63) / 64;
            for (int s = 1; s < 8; ++s) {
                WideCol m((size_t)words, 0);
                if (s & 1) m[(size_t)(a >> 6)] ^= 1ull << (a & 63);
                if (s & 2) m[(size_t)(b >> 6)] ^= 1ull << (b & 63);
                if (s & 4) m[(size_t)(c >> 6)] ^= 1ull << (c & 63);
                tab.push_back(m);
            }
        }
        std::vector<WideCol> red = fast_todd_reduce_wide(tab, r);
        CHECK(red.size() <= tab.size());
        std::vector<ResidueColumn> f, fp;
        for (const auto& m : tab) f.push_back({on_of(m, r), 1});
        for (const auto& m : red) fp.push_back({on_of(m, r), 1});
        CHECK(even_residue_symbolic(r, f, fp).ok);
    }
}

}  // namespace

int main() {
    RUN(wide_equals_uint32);
    RUN(symbolic_equals_dense);
    RUN(wide_reduces_above_30);
    REPORT();
}
