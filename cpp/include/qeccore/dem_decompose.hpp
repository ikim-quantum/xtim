#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qeccore {

// Sorted DEM targets; ids in [0,D) are detectors, [D,inf) are observables/expectation columns.
// D is a call-site parameter (see decompose_acc), not part of this type.
using Sig = std::vector<int32_t>;

struct DecomposedMech {
    std::vector<Sig> components;     // each component's sorted target list; XOR == original sig
    double p;
};

Sig sig_xor(const Sig& a, const Sig& b);   // XOR two sorted target lists (equal entries cancel)

// One independent error mechanism: a record-flip signature with its probability. Shared between
// the channel solver (dem_solve_channel.cpp) and export_dem (dem_export.cpp).
struct Mech { Sig sig; double p; };

// Exact conversion of one categorical channel (merged distinct signatures + probabilities) into
// independent error mechanisms, matching Stim: Walsh-Hadamard solve over the group the signatures
// generate. Returns false with a diagnostic in `err` when the channel is not exactly
// representable (over-mixing or a negative independent probability).
bool solve_channel(const std::map<Sig, double>& merged, std::vector<Mech>& out, std::string& err);

// Decompose every mechanism in `acc` into graphlike (<=2-detector) components. D = detector
// count (ids < D are detectors). <=2-detector mechs pass through as a single component; each
// hyperedge is decomposed into real model edges (preferring the split that best reconstructs
// the per-edge weight profile); a greedy peel is the fallback. Failure: if ignore_failures,
// emit undecomposed (single component); else return false with a message in `err`.
// `out` is APPENDED to (not cleared) — pass a fresh vector, or chain calls deliberately.
bool decompose_acc(const std::map<Sig, double>& acc, int D, bool ignore_failures,
                   std::vector<DecomposedMech>& out, std::string& err);

}  // namespace qeccore
