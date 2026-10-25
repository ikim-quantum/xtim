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

SolveStatus solve_coset_fault(const FaultLaw& f, std::vector<Mech>& out) {
    const int v = (int)f.W.size();
    const double p = f.p;
    if (v == 0) {                                       // single signature: q = p (any p)
        if (!f.c.empty() && p > 0.0) out.push_back({f.c, p});
        return SolveStatus::Ok;
    }
    // c' in W? (the elimination of W, then c')
    SigSpace sp;
    for (const Sig& w : f.W) sp.add_ids(w);
    sp.add_ids(f.c);
    sp.freeze();
    for (const Sig& w : f.W) sp.insert(w);
    const size_t rk = sp.rows.size();
    sp.insert(f.c);
    const bool c_in_W = sp.rows.size() == rk;
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

}  // namespace qeccore
