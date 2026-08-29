#include "qeccore/dem_decompose.hpp"
#include <cstddef>
#include <algorithm>
#include <cstdio>
#include <functional>

namespace qeccore {

Sig sig_xor(const Sig& a, const Sig& b) {
    Sig out; size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) out.push_back(a[i++]);
        else if (b[j] < a[i]) out.push_back(b[j++]);
        else { ++i; ++j; }              // equal target cancels (GF(2))
    }
    while (i < a.size()) out.push_back(a[i++]);
    while (j < b.size()) out.push_back(b[j++]);
    return out;
}

static int det_count(const Sig& s, int D) {
    int c = 0; for (int32_t t : s) if (t < D) ++c; return c;
}

// Are all of e's DETECTORS present in remnant? (observables don't constrain the subset)
static bool det_subset(const Sig& e, const Sig& remnant, int D) {
    for (int32_t t : e) {
        if (t >= D) continue;
        if (!std::binary_search(remnant.begin(), remnant.end(), t)) return false;
    }
    return true;
}
// Are all of e's OBSERVABLES present in remnant? (used as a tie-break: prefer an edge that
// already carries the observable, so an observable isn't split across components unnecessarily)
static bool obs_subset(const Sig& e, const Sig& remnant, int D) {
    for (int32_t t : e) {
        if (t < D) continue;
        if (!std::binary_search(remnant.begin(), remnant.end(), t)) return false;
    }
    return true;
}

bool decompose_acc(const std::map<Sig, double>& acc, int D, bool ignore_failures,
                   std::vector<DecomposedMech>& out, std::string& err) {
    // Graphlike candidate edges (1-2 detectors), with their model probabilities. Built in acc's
    // key order (acc is a std::map<Sig,...>, so lex-sorted by target list); E[] is the ordered
    // edge list and Eidx is the membership map for "is this a real model edge?" queries.
    std::vector<Sig> E;
    std::vector<double> Ep;
    std::map<Sig, int> Eidx;
    for (const auto& kv : acc) {
        int dc = det_count(kv.first, D);
        if (dc >= 1 && dc <= 2) {
            Eidx.emplace(kv.first, (int)E.size());
            E.push_back(kv.first);
            Ep.push_back(kv.second);
        }
    }
    auto edge_exists = [&](const Sig& e) { return Eidx.count(e) != 0; };
    auto edge_index = [&](const Sig& e) {
        auto it = Eidx.find(e); return it == Eidx.end() ? -1 : it->second;
    };
    auto obs_prob = [&](const Sig& e) {
        // probability mass that THIS component carries the observable (0 if it has none)
        bool has_obs = false; for (int32_t t : e) if (t >= D) { has_obs = true; break; }
        if (!has_obs) return 0.0;
        int i = edge_index(e); return i < 0 ? 0.0 : Ep[(size_t)i];
    };

    for (const auto& kv : acc) {
        const Sig& s = kv.first; double p = kv.second;
        if (p <= 0.0) continue;
        if (det_count(s, D) <= 2) { out.push_back({{s}, p}); continue; }

        // Candidate edges whose detectors all lie inside s (pruned once up front).
        std::vector<int> cand;
        for (int i = 0; i < (int)E.size(); ++i)
            if (det_subset(E[i], s, D)) cand.push_back(i);

        // Enumerate decompositions of s into REAL model edges (every component, incl. the final
        // remnant, must be a 1-2-detector edge in E). Among all such, pick the one that places the
        // MOST observable probability mass on its observable-carrying components. xtim folds error
        // mechanisms into a canonical channel before decomposing, which loses stim's per-mechanism
        // observable placement; maximizing observable-mass on real edges reconstructs the per-edge
        // weight profile that makes PyMatching decode-equivalent to Stim (verified by LER on the
        // generated surface/repetition/color library). Components are required index-nondecreasing
        // to avoid enumerating the same multiset twice; depth is bounded (<=4 components).
        const int MAX_COMPONENTS = 4;  // surface/color hyperedges decompose into <=4 graphlike pieces in practice; raise if a new code needs deeper splits.
        std::vector<Sig> best_comps;
        double best_score = -1.0;
        std::vector<Sig> chosen;

        std::function<void(const Sig&, int)> rec =
            [&](const Sig& remnant, int minIdx) {
                if (det_count(remnant, D) <= 2) {
                    if (!edge_exists(remnant)) return;     // remnant is not a real edge: dead end
                    double score = obs_prob(remnant);
                    for (const Sig& c : chosen) score += obs_prob(c);
                    if (score > best_score) {
                        best_score = score;
                        best_comps = chosen;
                        best_comps.push_back(remnant);
                    }
                    return;
                }
                if ((int)chosen.size() >= MAX_COMPONENTS - 1) return;
                for (int ci : cand) {
                    if (ci < minIdx) continue;             // index-nondecreasing
                    const Sig& e = E[(size_t)ci];
                    if (!det_subset(e, remnant, D)) continue;
                    Sig rem2 = sig_xor(remnant, e);
                    if (det_count(rem2, D) >= det_count(remnant, D)) continue;  // must make progress
                    chosen.push_back(e);
                    rec(rem2, ci);
                    chosen.pop_back();
                }
            };
        rec(s, 0);

        if (best_score >= 0.0) { out.push_back({std::move(best_comps), p}); continue; }

        // Fallback: no all-real-edge decomposition found. Greedy-peel allowing a (possibly
        // non-model) <=2-detector remnant, matching the legacy behavior so we still emit graphlike
        // components when possible.
        std::vector<Sig> comps;
        Sig remnant = s;
        bool ok = true;
        while (det_count(remnant, D) > 2) {
            const Sig* best = nullptr; int best_dc = 0; bool best_obs = false; bool best_compl = false;
            for (const Sig& e : E) {
                int ec = det_count(e, D);
                if (!det_subset(e, remnant, D)) continue;
                Sig rem2 = sig_xor(remnant, e);
                bool compl_real = (det_count(rem2, D) <= 2) && edge_exists(rem2);
                bool eo = obs_subset(e, remnant, D);
                bool better;
                if (!best) better = true;
                else if (compl_real != best_compl) better = compl_real && !best_compl;
                else if (eo != best_obs) better = eo && !best_obs;
                else better = ec > best_dc;
                if (better) {
                    best = &e; best_dc = ec; best_obs = eo; best_compl = compl_real;
                    if (compl_real && eo && ec == 2) break;
                }
            }
            if (!best) { ok = false; break; }
            comps.push_back(*best);
            remnant = sig_xor(remnant, *best);
        }
        if (!ok) {
            if (ignore_failures) { out.push_back({{s}, p}); continue; }
            char buf[32]; std::snprintf(buf, sizeof buf, "%d", det_count(s, D));
            err = "failed to decompose error with " + std::string(buf) +
                  " detectors into graphlike components";
            return false;
        }
        comps.push_back(remnant);
        out.push_back({std::move(comps), p});
    }
    return true;
}

}  // namespace qeccore
