#pragma once
// record_table — S2b (twirl consolidation, design §5.1 S2b; ndrec §1.2-1.4): the per-plan RECORD
// TABLE of a DIAGONAL plan. Every record column (IN_GROUP channels, the Born observable, Born LOGICAL /
// ANTI decisions) is an affine function of the shot's fair coins u and Born outcomes z (Theorem 1):
//     y_c = A_c·u  ⊕  B_c·z  ⊕  β_c                                   (RAW, P-free; Lemma 2 adds t_c(P))
// in the RECORD-ORDERED CANONICAL BASIS (Lemma 4): columns are scanned in global record order
// (detectors, observables, decisions — the oracle's order); a column whose coin content is independent
// of every earlier column is a PIVOT (coin k = its own value: A = e_k, B = 0, β = 0); a later column
// whose residual direction content is independent INTRODUCES direction j (B = e_j, β = 0) with
//     Λ_j := (−1)^{β'} · Π_{i ∈ b'} Λ_i      (the LITERAL ordered product of the input directions'
//                                             effective free operators, exact phase; never
//                                             re-canonicalized — design r5.4 literal-product rule)
// and every other column is a known combination of earlier pivots / directions.
//
// Conventions (normative):
//   * A direction is a compact free-register operator in EFFECTIVE form i^phase · (−1)^par · X^a Z^b
//     (FreeDir; the Born trie projects with (1 ± Λ̂)/2). Its outcome bit is 0 ⇔ eigenvalue +1.
//   * The literal product of commuting Hermitian directions is Hermitian and its eigenvalue on a joint
//     eigenvector is the product of eigenvalues, so the outcome of Π_{i∈b} Λ_i is b·z EXACTLY (no sign).
//     The A4 ordered-product sign (Theorem 1's s_c) appears only when a record's own operator is
//     compared with that product, and it lands in β (emission), never in a projector.
//   * The oracle (cpp/testapi/record_law_oracle.cpp) stores Λ_j = the free part of K_{c_j} at K's exact
//     phase. Its conversion to FreeDir is (a, b, phase = ph & 3, par = parity of K's Z on eps = 1 rows,
//     hermitian = 1). K = Q_c·Π Q_p is a product of commuting Hermitian Paulis whose certified rows carry
//     no X in frame coordinates, so (phase − |a∧b|) is even on every reachable input: the conversion
//     asserts it (engine-bug throw) and never Hermitizes (design r3.3).
#include <cstdint>
#include <string>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// ── Compact free-register operators ───────────────────────────────────────────────────────────
inline int fd_eff_phase(const FreeDir& d) { return (d.phase + 2 * d.par) & 3; }
// (phase − |a∧b|) even ⇔ Hermitian.
bool fd_hermitian(const FreeDir& d);
// The literal product p·q of two effective operators (exact phase; par folded into phase).
FreeDir fd_mul(const FreeDir& p, const FreeDir& q);
// The identity on k coordinates.
FreeDir fd_identity(int k);
// Throws std::logic_error(what) unless d is Hermitian (the proven invariant above).
void fd_require_hermitian(const FreeDir& d, const char* what);

// ── Pullback and plane ────────────────────────────────────────────────────────────────────────
// C0† M C0 for the diagonal residual C0 = S^{a&1} · CZ of nf (identity prefix), exact phase.
Pauli diag_pullback(const DiagNormalForm& nf, const Pauli& M);
// phi(Q): bit a = <Q, g_a> over the certified generators (GW words).
void phi_of(const CertifiedGroupPlanes& G, const Pauli& Q, std::vector<uint64_t>& out);

// ── The record table ──────────────────────────────────────────────────────────────────────────
// A generic affine table over coin sources (uniform, independent of the directions) and directions
// (commuting Hermitian free-register operators with a known joint Born law). Column c is
// (A[c] over coins, B[c] over directions, beta[c]).
struct AffineTable {
    int m = 0;                                   // columns
    int ncoin = 0;                               // coin sources
    std::vector<FreeDir> dirs;                   // directions (effective free operators)
    std::vector<std::vector<uint64_t>> A, B;     // per column: bitmask over coins / directions
    std::vector<uint8_t> beta;                   // per column
    // r7.2: the one-word form (ncoin <= 64 and dirs <= 64): A64[c] / B64[c] replace A[c] / B[c]
    // (same bits); canonicalize_record_order then runs on words — the same algorithm, no per-column
    // vectors.
    bool small = false;
    std::vector<uint64_t> A64, B64;
};

// The canonical table of a plan (payload). Column order is the SAMPLER's storage order (channels in
// channel-slot order, then the extras); `order` gives each column's global record index.
struct RecTable {
    bool built = false;
    int m = 0, rho = 0, kappa = 0;               // columns, pivots (coins), record directions
    int CWX = 0;                                 // words per column row
    int folds = 0;                               // introducers built with the fold bit β' = 1 (review
                                                 // R6-4: the M-BETA reachability dump; fits the padding)
    std::vector<uint64_t> base;                  // β row (CWX words)
    std::vector<std::vector<uint64_t>> coin_row; // ρ rows (CWX words): coin k flips these columns
    std::vector<std::vector<uint64_t>> born_row; // κ rows (CWX words): direction j flips these columns
    std::vector<int> pivot_rec, born_rec;        // storage column of pivot k / introducer of direction j
    std::vector<FreeDir> lam;                    // κ canonical directions (literal products, exact phase)
    // add_extra operation counts of THIS table's build (mechanism test M-1).
    long ops_plane = 0, ops_pivot_reduce = 0, ops_greduce = 0, ops_echelon = 0, extras = 0;
};

// Canonicalize an affine table in record order. `order[c]` is column c's global record index (a
// permutation key; ties are not allowed). Fills `out` (canonical rows in STORAGE order, CWX words per
// row). Every canonical direction is the literal product of the input directions (with the folded β
// sign). Lemma-4 structure is asserted (engine-bug throws): A_c supported on pivots ≤ c, B_c on
// directions introduced by records ≤ c, B_{born_rec[j]} = e_j, every Λ Hermitian.
void canonicalize_record_order(const AffineTable& in, const std::vector<int>& order, RecTable& out);

// One record column of the sampler's family (circuit constant).
struct RecColumn {
    Pauli M;                  // the record operator (product of terminal reads, exact phase)
    int order = 0;            // global record index (detectors, observables, decisions)
    bool channel = false;     // IN_GROUP at R = I (σ-mask channel); else an extra (Born obs / decision)
    std::vector<uint64_t> mask;   // channel: certified-generator mask (σ parity)
    uint8_t ref = 0;          // channel: sign of M relative to Π_{mask} g (raw = ref ⊕ mask·σ)
};

}  // namespace qeccore
