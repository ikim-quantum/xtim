#pragma once
#include <cstdint>
#include <functional>
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

// ── One fault's DEM law and its exact independent-mechanism solve (EXP-fix, owner design
// 2026-10-21: docs/twirl_pauli_condition.tex, Theorem "Noise model"). A FaultLaw is what ONE
// alternative of a channel instance does to the DEM targets GIVEN that it fires (probability p):
//   * Pauli:   the single signature `c`;
//   * Coset:   uniform on the coset c + span(W) (W a basis, dim v = W.size()): the stabilizer-twirled
//              residual of a Clifford fault whose law is a flat indicator dressed by a linear character;
//   * Support: an explicit categorical law `sup` (signature, weight), weights summing to 1 — a law
//              that is not a uniform coset (the general correlated-read law, or the PK_RAND list law).
//   * Mixture: (twirl-routes T1, the ONE routine's law) Σ_i sup[i].second · U(sup[i].first + span(W)): the fair
//              coins' span W (a basis) and the Born leaves' centres with their weights (Support = Mixture, W empty).
// A signature may be empty (the identity pattern).
struct FaultLaw {
    enum class Kind : uint8_t { Pauli, Coset, Support, Mixture };
    Kind kind = Kind::Pauli;
    double p = 0.0;
    Sig c;
    std::vector<Sig> W;
    std::vector<std::pair<Sig, double>> sup;
};

// Outcome of an exact solve: OverMixing = a Fourier
// coefficient of the instance's signature distribution is <= 1e-12; Negative = some independent
// mechanism has q < -1e-12 (Stim's `approximate_disjoint_errors` case); mechanisms with
// -1e-12 <= q <= 0 are skipped. RankTooLarge = the joint signature group's rank exceeds the caller's
// max_rank (no 2^rank array was allocated).
enum class SolveStatus : uint8_t { Ok, OverMixing, Negative, RankTooLarge };

// ── Clifford faults as unheralded erasures (dem-erasure; design note signed 2026-10-30) ──────────
// A twirled Clifford fault uniform on c + span(W) (dim v >= 1) is emitted as its OWN independent
// mechanisms, one per pattern, all at the same probability (no enumeration needed for q):
//   pure   (c in W):     (1-2q)^(2^(v-1)) = 1-p  on each of the 2^v - 1 patterns of W \ {0} (exact);
//   offset (c not in W): (1-2q)^(2^v)     = 1-2p on each of the 2^v patterns of c + W. The exact split
//                        also needs q_W < 0 on W \ {0} (|q_W| ~ p^2 / 2^(v+1)), which is omitted: the
//                        emitted law is FIRST-ORDER exact. `qW` (optional) receives q_W.
// OverMixing when 1-p (pure) or 1-2p (offset) <= 1e-12 (no q is written). (The owner's closed form; the retired
// solve_coset_fault survives as an oracle in cpp/tests/test_dem_fault_law.cpp.)
SolveStatus erasure_q(double p, int v, bool offset, double& q, double* qW = nullptr);

// The weight-cutoff split of W over a local bitset frame (UW words; dm = the detector bits; the
// weight of a pattern is the number of DETECTORS it flips, observables never count):
//   g_1..g_r  rows with pairwise-distinct detector pivots piv[i], each zero on the other pivots;
//   h_1..h_k  observable-only directions (det part 0), r + k = v.
// Every element of c + span(W) is c' + sum_S g + sum_T h (c' = c reduced on the pivots), and its
// pivot bits are exactly S, so a pattern of weight <= t has |S| <= t. Dependent rows of W are dropped.
struct ErasureSplit {
    int UW = 0;
    std::vector<uint64_t> dm;
    std::vector<std::vector<uint64_t>> g;
    std::vector<int> piv;
    std::vector<std::vector<uint64_t>> h;
    std::vector<int> hpiv;                       // h in echelon form: lowest bit, zero on the others' pivots
    int v() const { return (int)(g.size() + h.size()); }
};
ErasureSplit erasure_split(const std::vector<std::vector<uint64_t>>& W, const std::vector<uint64_t>& dm);
// The number of patterns the walk visits at cutoff t (t < 0: unbounded): sum_{j <= min(t, r)} C(r, j)
// * 2^k (2^v when t >= r). As a double (saturates harmlessly far above any budget).
double erasure_candidates(const ErasureSplit& s, int t);
// Is c in span(W)? (c reduced by g and h is zero.)
bool erasure_pure(const ErasureSplit& s, const std::vector<uint64_t>& c);
// Every pattern x of c + span(W) with popcount(x & dm) + extra_w <= t (t < 0: every pattern), the
// zero pattern skipped when `pure`; cb(x) once per kept pattern, in walk order. Returns the count.
// Cost sum_{j <= t} C(r, j) * 2^k * O(UW); at t < 0 exactly the 2^v walk.
long long enumerate_erasure_split(const ErasureSplit& s, const std::vector<uint64_t>& c, int extra_w,
                                  bool pure, int t,
                                  const std::function<void(const std::vector<uint64_t>&)>& cb);
// The same on a Coset FaultLaw in Sig form (ids < D are detectors): cb(sig) per kept pattern (never
// the empty one). *pure / *v receive the law's class and dimension. Returns the kept count.
long long enumerate_erasure(const FaultLaw& f, int D, int t, const std::function<void(const Sig&)>& cb,
                            bool* pure = nullptr, int* v = nullptr);

// Dimension of the affine hull of a set of signatures (rank of {s - s_0}); 0 for an empty set.
int sig_affine_rank(const std::vector<Sig>& pts);

// twirl-routes T1b: the exact independent split of ONE fault's Mixture (or Support) law f by CLASSES. With
// H = span(W, centres) and Q = H / span(W) (dim d), the Fourier transform of (1 - p) δ_0 + p f is
//     1 - p S(0) + p 1[χ ⊥ W] S(χ),   S(χ) = Σ_i w_i (-1)^{χ·c_i}   (χ ∈ Q* on W^⊥),
// so L = -(2/|H|) WHT(log μ̂) is constant on each class of H mod W: the 2^d class values come from two d-point
// transforms (no 2^dim(H) table), and every pattern of the class's coset c + span(W) gets q = -expm1(L)/2 — the same
// mechanisms as solve_mixture_fwht on the explicit support (within rounding). Decisions as solve_mixture_fwht:
// OverMixing (a coefficient <= 1e-12), Negative (some q < -1e-12), mechanisms with -1e-12 <= q <= 0 skipped; on
// either failure `out` holds the fault's DISJOINT law instead (each support pattern at p w_i / 2^r: first-order
// exact — deviation D1 for Negative, stim's approximate_disjoint_errors law for OverMixing).
// Emission keeps the patterns of detector weight <= t (ids < D are detectors; t < 0: every pattern), each class
// walked by the erasure enumerator; `dropped` receives the law's mass p · Pr[weight > t]; `v` the dimension of the
// support's affine hull; `supp` every target the law can flip.
SolveStatus solve_mixture_classes(const FaultLaw& f, int D, int t, std::vector<Mech>& out, double& dropped, int& v,
                                  Sig& supp);

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
// `extra_edges` (dem-erasure): further mechanisms (the Clifford records' lines) whose 1-2-detector
// members join acc's in the edge model; nullptr = acc's own model (byte-identical to before).
bool decompose_acc(const std::map<Sig, double>& acc, int D, bool ignore_failures,
                   std::vector<DecomposedMech>& out, std::string& err,
                   const std::map<Sig, double>* extra_edges = nullptr);

// The edge model decompose_acc uses, built once and shared: the 1-2-detector members of acc (and of
// extra_edges, XOR-composed on a shared signature), in signature order.
struct DemEdgeModel {
    std::vector<Sig> E;
    std::vector<double> Ep;
    std::map<Sig, int> Eidx;
};
DemEdgeModel dem_edge_model(const std::map<Sig, double>& acc, int D,
                            const std::map<Sig, double>* extra_edges = nullptr);
// Decompose one mechanism against the model (decompose_acc's per-mechanism step); appends to out.
bool decompose_one(const DemEdgeModel& model, const Sig& s, double p, int D, bool ignore_failures,
                   std::vector<DecomposedMech>& out, std::string& err);

}  // namespace qeccore
