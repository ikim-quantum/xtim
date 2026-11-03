#pragma once
// record_general — J-a (N3, switched OFF in production: built only through FAM samplers' test API). The
// builder (G) of design.final.md §5.1 J-a: the per-plan record table of ANY plan C0 = rot(S) of the
// term-indexed residual (diagonal or kind 2), in the NULL FORM (plan r2.0 Q8):
//
//   * every record column, in global record order, is ONE add_extra (record_table_internal.hpp — the
//     same elimination routine (D) applies to its extras: one plane, one pivot reduction, and for a
//     non-pivot one G.reduce and one free-register echelon) against a table that starts EMPTY, with
//     term_pullback in place of (D)'s diag_pullback;
//   * then the unchanged canonicalize_record_order.
//
// That is Theorem 1 verbatim. The output is the SAME RecTable (affine rows, pivots, introducers, FreeDir
// directions in the record-ordered canonical basis), so BornTrie::walk and the record readers apply to it
// unchanged, and on a diagonal plan it is field-identical to (D)'s table (the canonical basis is unique).
//
// The pullback (exact phase). rot(B) := exp(i(π/4) B) for a signed Hermitian axis B (term_residual.hpp,
// record_law_oracle.hpp). For {M, B} = 0: rot(B)† M rot(B) = M · (iB); for [M, B] = 0 it is M. The term
// axes commute pairwise, so for C0 = Π_{j∈S} rot(B_j)
//     C0† M C0 = M · Π_{j∈S, {M,B_j}=0} (i B_j)        (the product order is immaterial).
// NOTE on the sign: design §2.7's "f4check sign" −iB_j is the FORWARD conjugation C0 P C0† = P·Π(−iB_j)
// (a frame / fault image); the PULLBACK of a record operator carries +iB_j (record_law_oracle.cpp
// pulled_read; cpp/tests/test_record_general.cpp checks both against dense matrices). Mutation
// XTIM_MUTATE_G_PULL flips it to −iB_j.
#include <cstdint>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/term_residual.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// C0† M C0 for C0 = Π_{j∈S} rot(B_j), B_j = tv.axis(j) (exact sign), S as tv.term_words() words.
Pauli term_pullback(const TermResidual& tv, const std::vector<uint64_t>& S, const Pauli& M);

// The record table of the plan C0 = rot(S) over `cols` (storage order; RecColumn::M and ::order are
// read, the channel / mask / ref fields are not — every column's sign comes from its own pullback).
// G / bare: the sampler's certified group planes and bare state (R = I). Op counters (mechanism test,
// plan §3.10): ops_plane = ops_pivot_reduce = m, ops_greduce = ops_echelon = m − rho, extras = m.
// Throws std::logic_error on an engine-bug invariant violation (the shared routine's asserts).
// intro_b (optional; N3-D3): per canonical direction j, |b| = the number of the working table's input
// directions in its introducer's residual (canonicalize_record_order's bres: B_c ⊕ Σ_{pivots s of the
// column's A} B_{pivot s}) — the null form gives 1 for every j (and RecTable::folds = 0).
//
// memo (optional; termroute-a F2): the per-column constants of ONE fixed (G, bare, tv, cols), owned by the caller
// that owns those inputs (the FAM sampler's FamRoute) and filled lazily here. A column whose M_c commutes with
// every B_j, j ∈ S, is unaffected (its pullback is M_c): its plane φ(M_c) and, when φ(M_c) = 0, its direction
// free_dir_of(bare, M_c) come from the memo. Byte-identical to memo = nullptr (op counters included).
struct GeneralColumnMemo {
    bool ready = false;
    int MW = 0;                                  // ⌈m/64⌉
    std::vector<uint64_t> tcols;                 // T × MW: the columns c with {M_c, B_j} = 0, per term j
    std::vector<uint64_t> aff;                   // MW: scratch, the columns S affects
    std::vector<std::vector<uint64_t>> ph;       // per column: φ(M_c) (filled on first unaffected use)
    std::vector<uint8_t> ph_ready, phz;          // φ filled; φ(M_c) = 0
    std::vector<FreeDir> kd;                     // per column with φ(M_c) = 0: free_dir_of(bare, M_c)
};
void build_record_table_general(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                const TermResidual& tv, const std::vector<uint64_t>& S,
                                const std::vector<RecColumn>& cols, RecTable& out,
                                std::vector<int>* intro_b = nullptr, GeneralColumnMemo* memo = nullptr);

// The term form of a DIAGONAL plan's residual C0 = S^{a&1} · Π_{cz} CZ (diag_pullback's C0), up to a
// global phase: S_q = e^{iπ/4} rot(−Z_q), CZ_{fs} = e^{iπ/4} rot(−Z_f) rot(−Z_s) rot(Z_f Z_s). Every
// axis of the returned list is in the term set (S = all terms). (G)'s table of a diagonal plan.
std::vector<Pauli> diag_plan_axes(const DiagNormalForm& nf, int n);

}  // namespace qeccore
