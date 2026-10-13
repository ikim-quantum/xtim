#include "qeccore/residue_projection.hpp"
#include <cstddef>
#include <algorithm>
#include <unordered_map>

namespace qeccore {

int popcount_row(const Row& r) {
    int c = 0;
    for (uint64_t w : r) c += __builtin_popcountll(w);
    return c;
}
bool row_zero(const Row& r) {
    for (uint64_t w : r) if (w) return false;
    return true;
}
Row row_xor(const Row& a, const Row& b) {
    Row r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] ^ b[i];
    return r;
}

void moebius(std::vector<uint8_t>& h, int r) {
    const size_t N = (size_t)1 << r;
    for (int b = 0; b < r; ++b) {
        const size_t bit = (size_t)1 << b;
        for (size_t x = 0; x < N; ++x)
            if (x & bit) h[x] = (uint8_t)((h[x] - h[x ^ bit]) & 7);
    }
}

EvenResidue even_residue_dense(int r, const std::vector<ResidueColumn>& f_cols,
                               const std::vector<ResidueColumn>& fp_cols) {
    const size_t N = (size_t)1 << r;
    auto mask_of = [](const std::vector<int>& on) { size_t m = 0; for (int i : on) m |= (size_t)1 << i; return m; };
    std::vector<uint8_t> g(N, 0);
    auto add = [&](const std::vector<ResidueColumn>& cols, int sign) {
        for (const auto& c : cols) {
            const size_t m = mask_of(c.on);
            const int c8 = ((sign * c.c8) % 8 + 8) % 8;
            if (!c8 || !m) continue;
            for (size_t x = 0; x < N; ++x)
                if (__builtin_popcountll(m & x) & 1) g[x] = (uint8_t)((g[x] + c8) & 7);
        }
    };
    add(f_cols, +1);
    add(fp_cols, -1);
    moebius(g, r);
    EvenResidue out;
    for (size_t s = 1; s < N; ++s) {
        const int deg = __builtin_popcountll(s), a = g[s] & 7;
        const bool ok = (deg == 1) ? !(a & 1) : (deg == 2) ? !(a & 3) : (a == 0);
        if (!ok) {
            out.ok = false; out.bad_deg = deg; out.bad_coeff = a;
            for (int i = 0; i < r; ++i) if ((s >> i) & 1) out.bad_on.push_back(i);
            return out;
        }
        if (!a) continue;
        const int lo = __builtin_ctzll(s);
        if (deg == 1) out.mono.push_back({lo, -1, a});
        else          out.mono.push_back({63 - __builtin_clzll(s), lo, a});
    }
    return out;
}

EvenResidue even_residue_symbolic(int r, const std::vector<ResidueColumn>& f_cols,
                                  const std::vector<ResidueColumn>& fp_cols) {
    // deg-1 / deg-2 coefficients in dense r and r×r arrays; deg-3 in a hash map keyed by the
    // packed triple (a < b < c).  Everything mod 8.
    std::vector<uint8_t> d1((size_t)r, 0), d2((size_t)r * (size_t)r, 0);
    std::unordered_map<uint64_t, uint8_t> d3;
    auto key3 = [](uint64_t a, uint64_t b, uint64_t c) { return a | (b << 21) | (c << 42); };
    auto add = [&](const std::vector<ResidueColumn>& cols, int sign) {
        for (const auto& col : cols) {
            const int c8 = ((sign * col.c8) % 8 + 8) % 8;
            if (!c8) continue;
            const int c2 = (8 - 2 * c8 % 8) % 8;              // (−2)·c  mod 8
            const int c3 = (4 * c8) % 8;                      // (+4)·c  mod 8
            const auto& on = col.on;
            const size_t w = on.size();
            for (size_t x = 0; x < w; ++x) {
                d1[(size_t)on[x]] = (uint8_t)((d1[(size_t)on[x]] + c8) & 7);
                if (c2 == 0 && c3 == 0) continue;
                for (size_t y = x + 1; y < w; ++y) {
                    uint8_t& e = d2[(size_t)on[x] * (size_t)r + (size_t)on[y]];
                    e = (uint8_t)((e + c2) & 7);
                    if (c3 == 0) continue;
                    for (size_t z = y + 1; z < w; ++z) {
                        uint8_t& t = d3[key3((uint64_t)on[x], (uint64_t)on[y], (uint64_t)on[z])];
                        t = (uint8_t)((t + c3) & 7);
                    }
                }
            }
        }
    };
    add(f_cols, +1);
    add(fp_cols, -1);

    // Violations are reported at the SMALLEST offending residue mask (the dense scan's order):
    // collect candidates, then pick the minimum by numeric mask (compare from the top bit down).
    EvenResidue out;
    struct Bad { std::vector<int> desc; int deg, a; };  // desc = bits high→low
    bool have_bad = false; Bad bad{};
    auto consider_bad = [&](std::vector<int> desc, int deg, int a) {
        if (!have_bad || desc < bad.desc) { bad = Bad{std::move(desc), deg, a}; have_bad = true; }
    };
    // Numeric order of masks == lexicographic order of their bits listed high→low, compared with
    // "absent" smaller than any index; std::vector's < does exactly that for equal prefixes
    // (a shorter vector — a sub-mask of the same top bits — sorts first).
    for (int i = 0; i < r; ++i) if (d1[(size_t)i] & 1) consider_bad({i}, 1, d1[(size_t)i]);
    for (int i = 0; i < r; ++i)
        for (int j = i + 1; j < r; ++j) {
            const int a = d2[(size_t)i * (size_t)r + (size_t)j];
            if (a & 3) consider_bad({j, i}, 2, a);
        }
    for (const auto& kv : d3)
        if (kv.second) {
            const int a = (int)(kv.first & 0x1fffff), b = (int)((kv.first >> 21) & 0x1fffff),
                      c = (int)(kv.first >> 42);
            consider_bad({c, b, a}, 3, kv.second);
        }
    if (have_bad) {
        out.ok = false; out.bad_deg = bad.deg; out.bad_coeff = bad.a;
        out.bad_on.assign(bad.desc.rbegin(), bad.desc.rend());
        return out;
    }
    // Emit nonzero deg-1 / deg-2 monomials in ascending mask order: by top bit, then (for the same
    // top bit) the singleton {hi} first (smaller mask), then pairs {lo < hi} by ascending lo.
    for (int hi = 0; hi < r; ++hi) {
        if (d1[(size_t)hi]) out.mono.push_back({hi, -1, d1[(size_t)hi]});
        for (int lo = 0; lo < hi; ++lo) {
            const int a = d2[(size_t)lo * (size_t)r + (size_t)hi];
            if (a) out.mono.push_back({hi, lo, a});
        }
    }
    return out;
}

}  // namespace qeccore
