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
                   std::string& err) {
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
        out.push_back({sigs[0], probs[0]});  // (exact for any p in [0,1], like X_ERROR(0.9))
        return true;
    }
    // GF(2) reduction of the signatures (symmetric difference of sorted target lists) to a
    // basis; coords[i] = sig i in basis coordinates. Rank <= 4 (signatures are linear images
    // of a <= 2-qubit Pauli group), so the 2^m tables below are tiny.
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
    if (m > 20) { err = "signature group rank too large (internal guard)"; return false; }
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
            err = "channel is not a product of independent record-flip mechanisms (exact "
                  "DEM conversion has a negative probability; Stim would require "
                  "approximate_disjoint_errors); refusing to approximate";
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

}  // namespace qeccore
