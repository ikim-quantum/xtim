#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <utility>
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
// representable (over-mixing or a negative independent probability). When `needs_approximation`
// is non-null it is set true iff the failure is the negative-probability case (Stim's
// "requires the `approximate_disjoint_errors` option" refusal).
// `max_rank`: when the signatures generate a group of rank > max_rank, return false BEFORE any 2^m table
// with *rank set (and *needs_approximation false); export_dem routes all-Pauli instances of rank <= 4 here.
bool solve_channel(const std::map<Sig, double>& merged, std::vector<Mech>& out, std::string& err,
                   bool* needs_approximation = nullptr, int max_rank = 20, int* rank = nullptr);

// ── One fault's DEM law and its exact independent-mechanism solve (EXP-fix, owner design
// 2026-10-21: docs/twirl_pauli_condition.tex, Theorem "Noise model"). A FaultLaw is what ONE
// alternative of a channel instance does to the DEM targets GIVEN that it fires (probability p):
//   * Pauli:   the single signature `c`;
//   * Coset:   uniform on the coset c + span(W) (W a basis, dim v = W.size()): the stabilizer-twirled
//              residual of a Clifford fault whose law is a flat indicator dressed by a linear character;
//   * Support: an explicit categorical law `sup` (signature, weight), weights summing to 1 — a law
//              that is not a uniform coset (the general correlated-read law, or the PK_RAND list law).
// A signature may be empty (the identity pattern).
struct FaultLaw {
    enum class Kind : uint8_t { Pauli, Coset, Support };
    Kind kind = Kind::Pauli;
    double p = 0.0;
    Sig c;
    std::vector<Sig> W;
    std::vector<std::pair<Sig, double>> sup;
};

// Outcome of an exact solve, with the same decisions as solve_channel: OverMixing = a Fourier
// coefficient of the instance's signature distribution is <= 1e-12; Negative = some independent
// mechanism has q < -1e-12 (Stim's `approximate_disjoint_errors` case); mechanisms with
// -1e-12 <= q <= 0 are skipped. RankTooLarge = the joint signature group's rank exceeds the caller's
// max_rank (no 2^rank array was allocated).
enum class SolveStatus : uint8_t { Ok, OverMixing, Negative, RankTooLarge };

// The owner's closed form for ONE fault uniform on c' + W (dim v), O(2^(v+1)) output:
//   c' in W:      (1-2q)^(2^(v-1)) = 1-p                 on every element of W \ {0};
//   c' not in W:  (1-2q_c)^(2^v)   = 1-2p                on c' + W,
//                 (1-2q_W)^(2^v)   = (1-p)^2/(1-2p)      on W \ {0} (q_W <= 0: skipped or Negative).
// v = 0 is the single-signature shortcut q = p (exact for any p, like X_ERROR(0.9)).
SolveStatus solve_coset_fault(const FaultLaw& f, std::vector<Mech>& out);

// Dimension of the affine hull of a set of signatures (rank of {s - s_0}); 0 for an empty set.
int sig_affine_rank(const std::vector<Sig>& pts);

// The exact joint solve of one channel instance (several alternatives of ONE fault, disjoint):
// sum_i p_i law_i scattered into the 2^m signature group its supports generate (m = `rank`), one
// fast Walsh-Hadamard transform (m 2^m) to the Fourier coefficients 1 + delta, and one more for
// L = -(2/G) WHT(log1p delta); q = -expm1(L)/2. m > max_rank: RankTooLarge before any allocation.
SolveStatus solve_mixture_fwht(const std::vector<FaultLaw>& comps, int max_rank,
                               std::vector<Mech>& out, int* rank);

// Decompose every mechanism in `acc` into graphlike (<=2-detector) components. D = detector
// count (ids < D are detectors). <=2-detector mechs pass through as a single component; each
// hyperedge is decomposed into real model edges (preferring the split that best reconstructs
// the per-edge weight profile); a greedy peel is the fallback. Failure: if ignore_failures,
// emit undecomposed (single component); else return false with a message in `err`.
// `out` is APPENDED to (not cleared) — pass a fresh vector, or chain calls deliberately.
bool decompose_acc(const std::map<Sig, double>& acc, int D, bool ignore_failures,
                   std::vector<DecomposedMech>& out, std::string& err);

}  // namespace qeccore
