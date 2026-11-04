#include "qeccore/dem_decompose.hpp"
#include <cstddef>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace qeccore {

namespace {

// GF(2) elimination over the DEM targets of an instance: every signature becomes a bitset over the
// instance's distinct target ids (bit order = id order, so a bitset converts back to a SORTED Sig),
// reduced against a basis whose rows have pairwise-distinct lowest-bit pivots. coord(s) = the rows
// whose XOR (with the residual, when it joins the basis) is s — solve_channel's convention.
struct SigSpace {
    std::vector<int32_t> ids;                  // sorted distinct target ids
    int UW = 0;
    std::vector<std::vector<uint64_t>> rows;   // basis rows, as inserted (reduced at insertion)
    std::vector<int> row_of_bit;               // pivot bit -> row index (-1: none)

    void add_ids(const Sig& s) { ids.insert(ids.end(), s.begin(), s.end()); }
    void freeze() {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        UW = (int)((ids.size() + 63) / 64);
        row_of_bit.assign((size_t)UW * 64, -1);
    }
    std::vector<uint64_t> bits(const Sig& s) const {
        std::vector<uint64_t> b((size_t)UW, 0);
        for (int32_t t : s) {
            const size_t i = (size_t)(std::lower_bound(ids.begin(), ids.end(), t) - ids.begin());
            b[i >> 6] ^= 1ull << (i & 63);
        }
        return b;
    }
    // Reduce s; a nonzero residual joins the basis. Returns the coordinate bits (rows < 64).
    uint64_t insert(const Sig& s) {
        std::vector<uint64_t> cur = bits(s);
        uint64_t c = 0;
        for (;;) {
            int lb = -1;
            for (int w = 0; w < UW; ++w)
                if (cur[(size_t)w]) { lb = w * 64 + __builtin_ctzll(cur[(size_t)w]); break; }
            if (lb < 0) return c;
            const int r = row_of_bit[(size_t)lb];
            if (r < 0) {
                row_of_bit[(size_t)lb] = (int)rows.size();
                if (rows.size() < 64) c |= 1ull << rows.size();
                rows.push_back(std::move(cur));
                return c;
            }
            for (int w = 0; w < UW; ++w) cur[(size_t)w] ^= rows[(size_t)r][(size_t)w];
            if (r < 64) c ^= 1ull << r;
        }
    }
    Sig to_sig(const std::vector<uint64_t>& b) const {
        Sig s;
        for (int w = 0; w < UW; ++w)
            for (uint64_t x = b[(size_t)w]; x; x &= x - 1)
                s.push_back(ids[(size_t)(w * 64 + __builtin_ctzll(x))]);
        return s;
    }
};

void fwht(std::vector<double>& a) {
    const size_t G = a.size();
    for (size_t h = 1; h < G; h <<= 1)
        for (size_t i = 0; i < G; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) {
                const double x = a[j], y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
}

}  // namespace

int sig_affine_rank(const std::vector<Sig>& pts) {
    if (pts.empty()) return 0;
    SigSpace sp;
    for (const Sig& s : pts) sp.add_ids(s);
    sp.freeze();
    for (const Sig& s : pts) sp.insert(sig_xor(s, pts[0]));
    return (int)sp.rows.size();
}

SolveStatus erasure_q(double p, int v, bool offset, double& q, double* qW) {
    if (!offset) {
        if (!(1.0 - p > 1e-12)) return SolveStatus::OverMixing;
        q = -std::expm1(std::log1p(-p) * std::ldexp(1.0, 1 - v)) / 2.0;
        return SolveStatus::Ok;
    }
    if (!(1.0 - 2.0 * p > 1e-12)) return SolveStatus::OverMixing;
    const double l1 = std::log1p(-p), l2 = std::log1p(-2.0 * p);
    if (qW) *qW = -std::expm1((2.0 * l1 - l2) * std::ldexp(1.0, -v)) / 2.0;
    q = -std::expm1(l2 * std::ldexp(1.0, -v)) / 2.0;
    return SolveStatus::Ok;
}

namespace {
inline int low_bit(const std::vector<uint64_t>& x) {
    for (size_t w = 0; w < x.size(); ++w)
        if (x[w]) return (int)(w * 64) + __builtin_ctzll(x[w]);
    return -1;
}
inline bool test_bit(const std::vector<uint64_t>& x, int b) { return (x[(size_t)(b >> 6)] >> (b & 63)) & 1; }
inline void xor_into(std::vector<uint64_t>& x, const std::vector<uint64_t>& y) {
    for (size_t w = 0; w < x.size(); ++w) x[w] ^= y[w];
}
}  // namespace

ErasureSplit erasure_split(const std::vector<std::vector<uint64_t>>& W, const std::vector<uint64_t>& dm) {
    ErasureSplit s;
    s.UW = (int)dm.size();
    s.dm = dm;
    std::vector<uint64_t> d((size_t)s.UW);
    for (const auto& w0 : W) {
        std::vector<uint64_t> x = w0;
        x.resize((size_t)s.UW, 0);
        for (size_t i = 0; i < s.g.size(); ++i)          // the g are fully reduced: one pass
            if (test_bit(x, s.piv[i])) xor_into(x, s.g[i]);
        for (int w = 0; w < s.UW; ++w) d[(size_t)w] = x[(size_t)w] & s.dm[(size_t)w];
        const int pd = low_bit(d);
        if (pd >= 0) {                                   // a new detector direction
            for (size_t i = 0; i < s.g.size(); ++i)
                if (test_bit(s.g[i], pd)) xor_into(s.g[i], x);
            s.g.push_back(std::move(x));
            s.piv.push_back(pd);
            continue;
        }
        for (size_t i = 0; i < s.h.size(); ++i)          // observable-only: independent of the h?
            if (test_bit(x, s.hpiv[i])) xor_into(x, s.h[i]);
        const int ph = low_bit(x);
        if (ph < 0) continue;                            // dependent row: dropped
        for (size_t i = 0; i < s.h.size(); ++i)
            if (test_bit(s.h[i], ph)) xor_into(s.h[i], x);
        s.h.push_back(std::move(x));
        s.hpiv.push_back(ph);
    }
    return s;
}

double erasure_candidates(const ErasureSplit& s, int t) {
    const int r = (int)s.g.size(), k = (int)s.h.size();
    if (t < 0 || t >= r) return std::ldexp(1.0, r + k);
    double sum = 0.0, binom = 1.0;                       // C(r, j), exact in double for r <= 64ish
    for (int j = 0; j <= t; ++j) {
        sum += binom;
        binom = binom * (double)(r - j) / (double)(j + 1);
    }
    return sum * std::ldexp(1.0, k);
}

bool erasure_pure(const ErasureSplit& s, const std::vector<uint64_t>& c) {
    std::vector<uint64_t> x = c;
    x.resize((size_t)s.UW, 0);
    for (size_t i = 0; i < s.g.size(); ++i) if (test_bit(x, s.piv[i])) xor_into(x, s.g[i]);
    for (size_t i = 0; i < s.h.size(); ++i) if (test_bit(x, s.hpiv[i])) xor_into(x, s.h[i]);
    return low_bit(x) < 0;
}

long long enumerate_erasure_split(const ErasureSplit& s, const std::vector<uint64_t>& c, int extra_w,
                                  bool pure, int t,
                                  const std::function<void(const std::vector<uint64_t>&)>& cb) {
    const int r = (int)s.g.size(), k = (int)s.h.size();
    const int UW = s.UW;
    // c reduced on the detector pivots: then the pivot bits of c + sum_S g are exactly S.
    std::vector<uint64_t> x = c;
    x.resize((size_t)UW, 0);
    for (int i = 0; i < r; ++i) if (test_bit(x, s.piv[(size_t)i])) xor_into(x, s.g[(size_t)i]);
    const int depth = (t < 0 || t > r) ? r : t;
    long long kept = 0;
    auto weight = [&](const std::vector<uint64_t>& y) {
        int w = extra_w;
        for (int i = 0; i < UW; ++i) w += __builtin_popcountll(y[(size_t)i] & s.dm[(size_t)i]);
        return w;
    };
    auto is_zero = [&](const std::vector<uint64_t>& y) {
        for (int i = 0; i < UW; ++i) if (y[(size_t)i]) return false;
        return true;
    };
    auto emit_lift = [&](std::vector<uint64_t>& y) {     // y + every subset of the h (Gray walk)
        if (!(pure && is_zero(y))) { cb(y); ++kept; }
        for (uint64_t gidx = 1; k > 0 && gidx < (1ull << k); ++gidx) {
            xor_into(y, s.h[(size_t)__builtin_ctzll(gidx)]);
            if (!(pure && is_zero(y))) { cb(y); ++kept; }
        }
        if (k > 0) xor_into(y, s.h[(size_t)(k - 1)]);    // the Gray walk ends at h_{k-1}: restore y
    };
    // Combination walk over S, |S| <= depth, indices increasing.
    std::function<void(int, int)> rec = [&](int start, int left) {
        if (t < 0 || weight(x) <= t) emit_lift(x);
        if (left == 0) return;
        for (int i = start; i < r; ++i) {
            xor_into(x, s.g[(size_t)i]);
            rec(i + 1, left - 1);
            xor_into(x, s.g[(size_t)i]);
        }
    };
    rec(0, depth);
    return kept;
}

long long enumerate_erasure(const FaultLaw& f, int D, int t, const std::function<void(const Sig&)>& cb,
                            bool* pure, int* v) {
    SigSpace sp;
    for (const Sig& w : f.W) sp.add_ids(w);
    sp.add_ids(f.c);
    sp.freeze();
    std::vector<uint64_t> dm((size_t)sp.UW, 0);
    for (size_t i = 0; i < sp.ids.size(); ++i)
        if (sp.ids[i] < D) dm[i >> 6] |= 1ull << (i & 63);
    std::vector<std::vector<uint64_t>> Wb;
    for (const Sig& w : f.W) Wb.push_back(sp.bits(w));
    const ErasureSplit s = erasure_split(Wb, dm);
    const std::vector<uint64_t> cb0 = sp.bits(f.c);
    const bool pu = erasure_pure(s, cb0);
    if (pure) *pure = pu;
    if (v) *v = s.v();
    return enumerate_erasure_split(s, cb0, 0, pu, t, [&](const std::vector<uint64_t>& x) { cb(sp.to_sig(x)); });
}

SolveStatus solve_mixture_fwht(const std::vector<FaultLaw>& comps, int max_rank,
                               std::vector<Mech>& out, int* rank) {
    SigSpace sp;
    for (const FaultLaw& f : comps) {
        sp.add_ids(f.c);
        for (const Sig& w : f.W) sp.add_ids(w);
        for (const auto& e : f.sup) sp.add_ids(e.first);
    }
    sp.freeze();
    // Coordinates of every point the scatter needs (inserting builds the basis as it goes; an
    // earlier point's coordinate stays valid because rows are only appended).
    std::vector<std::vector<uint64_t>> co(comps.size());
    for (size_t i = 0; i < comps.size(); ++i) {
        const FaultLaw& f = comps[i];
        if (f.kind == FaultLaw::Kind::Support) {
            for (const auto& e : f.sup) co[i].push_back(sp.insert(e.first));
        } else {
            co[i].push_back(sp.insert(f.c));
            for (const Sig& w : f.W) co[i].push_back(sp.insert(w));
        }
    }
    const int m = (int)sp.rows.size();
    if (rank) *rank = m;
    if (m > max_rank) return SolveStatus::RankTooLarge;
    if (m == 0) return SolveStatus::Ok;
    const size_t G = (size_t)1 << m;
    std::vector<double> a(G, 0.0);
    for (size_t i = 0; i < comps.size(); ++i) {
        const FaultLaw& f = comps[i];
        if (f.kind == FaultLaw::Kind::Support) {
            for (size_t j = 0; j < f.sup.size(); ++j) a[(size_t)co[i][j]] += f.p * f.sup[j].second;
        } else if (f.kind == FaultLaw::Kind::Pauli) {
            a[(size_t)co[i][0]] += f.p;
        } else {
            const int v = (int)f.W.size();
            const double w = std::ldexp(f.p, -v);
            uint64_t x = co[i][0];
            a[(size_t)x] += w;
            for (uint32_t g = 1; g < (1u << v); ++g) {
                x ^= co[i][1 + (size_t)__builtin_ctz(g)];
                a[(size_t)x] += w;
            }
        }
    }
    fwht(a);                                            // a = S(chi); 1 + delta = 1 + S(chi) - S(0)
    const double S0 = a[0];
    for (size_t c = 0; c < G; ++c) {
        const double d = a[c] - S0;
        if (!(1.0 + d > 1e-12)) return SolveStatus::OverMixing;
        a[c] = std::log1p(d);
    }
    fwht(a);
    const double scale = -2.0 / (double)G;
    std::vector<double> q(G, 0.0);
    for (size_t h = 1; h < G; ++h) {
        q[h] = -std::expm1(scale * a[h]) / 2.0;
        if (q[h] < -1e-12) return SolveStatus::Negative;
    }
    std::vector<uint64_t> cur((size_t)sp.UW, 0);        // Gray walk: cur = XOR of the rows in h
    for (size_t g = 1; g < G; ++g) {
        const int r = __builtin_ctzll(g);
        for (int w = 0; w < sp.UW; ++w) cur[(size_t)w] ^= sp.rows[(size_t)r][(size_t)w];
        const size_t h = g ^ (g >> 1);
        if (q[h] > 0.0) out.push_back({sp.to_sig(cur), q[h]});
    }
    return SolveStatus::Ok;
}

SolveStatus solve_mixture_classes(const FaultLaw& f, int D, int t, std::vector<Mech>& out, double& dropped, int& v,
                                  Sig& supp) {
    out.clear();
    dropped = 0.0;
    v = 0;
    supp.clear();
    SigSpace sp;
    for (const Sig& w : f.W) sp.add_ids(w);
    for (const auto& e : f.sup) sp.add_ids(e.first);
    sp.freeze();
    const int UW = std::max(1, sp.UW);
    auto bits = [&](const Sig& s) {
        std::vector<uint64_t> b = sp.bits(s);
        b.resize((size_t)UW, 0);
        return b;
    };
    std::vector<uint64_t> dm((size_t)UW, 0);
    for (size_t i = 0; i < sp.ids.size(); ++i)
        if (sp.ids[i] < D) dm[i >> 6] |= 1ull << (i & 63);
    // span(W): a fully reduced echelon (lowest-bit pivots; each row zero on the other rows' pivots)
    std::vector<std::vector<uint64_t>> Wr, Qr;
    std::vector<int> Wp, Qp;
    auto reduce = [](std::vector<uint64_t>& x, const std::vector<std::vector<uint64_t>>& R, const std::vector<int>& P) {
        for (size_t i = 0; i < R.size(); ++i) if (test_bit(x, P[i])) xor_into(x, R[i]);
    };
    auto insert = [&](std::vector<uint64_t> x, std::vector<std::vector<uint64_t>>& R, std::vector<int>& P) {
        const int pv = low_bit(x);
        if (pv < 0) return false;
        for (auto& r : R) if (test_bit(r, pv)) xor_into(r, x);
        R.push_back(std::move(x));
        P.push_back(pv);
        return true;
    };
    for (const Sig& w : f.W) {
        std::vector<uint64_t> x = bits(w);
        reduce(x, Wr, Wp);
        insert(std::move(x), Wr, Wp);
    }
    const int r = (int)Wr.size();
    // the classes: each centre reduced mod span(W), weights merged (identical classes are one coset)
    std::map<std::vector<uint64_t>, double> cls;
    for (const auto& e : f.sup) {
        std::vector<uint64_t> x = bits(e.first);
        reduce(x, Wr, Wp);
        cls[x] += e.second;
    }
    if (cls.empty()) return SolveStatus::Ok;
    // Q = H / span(W): a fully reduced echelon of the class representatives (zero on W's pivots by construction)
    for (const auto& [x, w] : cls) {
        std::vector<uint64_t> y = x;
        reduce(y, Qr, Qp);
        insert(std::move(y), Qr, Qp);
    }
    const int d = (int)Qr.size();
    if (d > 30) return SolveStatus::RankTooLarge;
    {   // v: the support's affine-hull dimension = r + rank of the class differences
        std::vector<std::vector<uint64_t>> Dr;
        std::vector<int> Dp;
        const std::vector<uint64_t>& x0 = cls.begin()->first;
        for (const auto& [x, w] : cls) {
            std::vector<uint64_t> y = x;
            xor_into(y, x0);
            reduce(y, Dr, Dp);
            insert(std::move(y), Dr, Dp);
        }
        v = r + (int)Dr.size();
    }
    {   // supp: every target some support pattern flips (the centres' bits and span(W)'s)
        std::vector<uint64_t> u((size_t)UW, 0);
        for (const auto& wr : Wr) for (int w = 0; w < UW; ++w) u[(size_t)w] |= wr[(size_t)w];
        for (const auto& e : f.sup) { const std::vector<uint64_t> b = bits(e.first); for (int w = 0; w < UW; ++w) u[(size_t)w] |= b[(size_t)w]; }
        supp = sp.to_sig(u);
    }
    // class coordinates over Q's basis (fully reduced: x = sum of the rows whose pivot bit x has)
    auto coord = [&](const std::vector<uint64_t>& x) {
        uint64_t c = 0;
        for (int l = 0; l < d; ++l) if (test_bit(x, Qp[(size_t)l])) c |= 1ull << l;
        return c;
    };
    auto centre = [&](uint64_t c) {
        std::vector<uint64_t> x((size_t)UW, 0);
        for (int l = 0; l < d; ++l) if ((c >> l) & 1) xor_into(x, Qr[(size_t)l]);
        return x;
    };
    const ErasureSplit split = erasure_split(Wr, dm);
    const double pr = std::ldexp(f.p, -r);
    auto emit_coset = [&](const std::vector<uint64_t>& c, bool pure, double q) {
        enumerate_erasure_split(split, c, 0, pure, t, [&](const std::vector<uint64_t>& x) { out.push_back({sp.to_sig(x), q}); });
    };
    if (t >= 0)                                          // the law's mass on patterns of detector weight > t
        for (const auto& [x, w] : cls) {
            const long long n = enumerate_erasure_split(split, x, 0, false, t, [](const std::vector<uint64_t>&) {});
            dropped += pr * w * (std::ldexp(1.0, r) - (double)n);
        }
    auto disjoint = [&]() {                              // every support pattern at its disjoint mass p w_i / 2^r
        out.clear();
        for (const auto& [x, w] : cls)
            if (pr * w > 0.0) emit_coset(x, true, pr * w);
    };
    const size_t G = (size_t)1 << d;
    std::vector<double> a(G, 0.0);
    double S0 = 0.0;
    for (const auto& [x, w] : cls) { a[(size_t)coord(x)] += w; S0 += w; }
    fwht(a);                                             // a = S(χ), χ in Q* (the characters on W^⊥)
    S0 = a[0];
    double G0 = 0.0;
    if (r > 0) {                                         // the characters off W^⊥: 1 - p S(0)
        if (!(1.0 - f.p * S0 > 1e-12)) { disjoint(); return SolveStatus::OverMixing; }
        G0 = std::log1p(-f.p * S0);
    }
    for (size_t c = 0; c < G; ++c) {
        const double dlt = f.p * (a[c] - S0);
        if (!(1.0 + dlt > 1e-12)) { disjoint(); return SolveStatus::OverMixing; }
        a[c] = std::log1p(dlt) - G0;
    }
    fwht(a);                                             // per class: |H| L / (-2)
    const double scale = -2.0 / std::ldexp(1.0, r + d);
    std::vector<double> q(G, 0.0);
    for (size_t c = 0; c < G; ++c) {
        if (c == 0 && r == 0) continue;                  // the zero class is the zero pattern only: no mechanism
        q[c] = -std::expm1(scale * a[c]) / 2.0;
        if (q[c] < -1e-12) { disjoint(); return SolveStatus::Negative; }
    }
    for (size_t c = 0; c < G; ++c)
        if (q[c] > 0.0) emit_coset(centre(c), c == 0, q[c]);
    return SolveStatus::Ok;
}

}  // namespace qeccore
