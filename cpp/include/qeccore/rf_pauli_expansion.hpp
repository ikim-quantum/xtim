#pragma once
// rf_pauli_expansion — S8 RF2 (plan r4.1; rfref/fable-decoded-design.md §2.1): the closed-form Pauli expansion of a
// kind-0 content D = S^a CZ^M (a ∈ {0,1}^n after `& 1`, M the CZ adjacency), a pure function.
//
// With q(x) = Σ a_i x_i + 2 Σ_{i<j} M_ij x_i x_j (mod 4), D = Σ_x i^{q(x)} |x⟩⟨x| and Ã = M + diag(a) (symmetric):
//     q(u ⊕ v) = q(u) + q(v) + 2 uᵀÃv (mod 4),
//     D = c Σ_{t ∈ F2^r} i^{Q_D(t)} Z^{z(t)},   z(t) = z_0 ⊕ Σ_i t_i col_{p_i}(Ã),   |c| = 2^{−r/2},  r = rank Ã,
//     Q_D(t) = Σ_i t_i (−a_{p_i} + 2 z_0[p_i]) + 2 Σ_{i<j} t_i t_j M_{p_i p_j}   (mod 4),
// p_1 < … < p_r the pivot columns of Ã's row reduction over the touched wires (ascending), z_0 the particular
// solution of z_0 · κ = q(κ)/2 on ker Ã (q is {0,2}-valued and additive there) with zero non-pivot bits. The dressed
// record image is the exact-phase Lemma 1: D† (i^p X^x Z^z) D = i^{p − q(x)} X^x Z^{z ⊕ Ãx}.

#include <cstdint>
#include <vector>

#include "qeccore/pauli.hpp"

namespace qeccore {

struct DiagNormalForm;

struct DiagExpansion {
    int n = 0, W = 0;
    int r = 0;                              // rank Ã = the expansion's variable count (|c|² = 2^{−r})
    std::vector<int> piv;                   // p_1 < … < p_r (wires)
    std::vector<uint64_t> cols;             // r × W: col_{p_i}(Ã) (= row p_i)
    std::vector<uint64_t> z0;               // W words
    std::vector<uint8_t> lin;               // r: the Z4 linear coefficients of Q_D
    std::vector<uint64_t> beta;             // r × WB (WB = ⌈r/64⌉): β_ij = M_{p_i p_j}, symmetric, zero diagonal
    int WB = 0;
    // Ã by touched wire (for the dressed images): rowof[q] = index into arows (W words each), −1 if untouched
    std::vector<int> rowof;
    std::vector<uint64_t> arows;            // the M part only (diagonal cleared)
    std::vector<uint8_t> a1;                // a & 1 per wire
    int qform(const uint64_t* x) const;     // q(x) mod 4 for an n-bit x (W words)
};

// ── S8 RF3/RF4: a Z4 quadratic form over V binary variables and its exact Gauss sum. ─────────────────────────
// Q(u) = c + Σ_j lin_j u_j + 2 Σ_{j<j'} β_jj' u_j u_j' (mod 4), β symmetric with zero diagonal (rows of WV words).
// Q0 = Q − c satisfies Q0(u ⊕ v) = Q0(u) + Q0(v) + 2 B(u, v), B(u, v) = Σ_j (lin_j & 1) u_j v_j + Σ_{j≠j'} β_jj' u_j v_j'.
struct QForm {
    int V = 0, WV = 0, c = 0;
    std::vector<uint8_t> lin;
    std::vector<uint64_t> beta;
    void init(int V_);
    uint64_t* row(int j) { return beta.data() + (size_t)j * (size_t)WV; }
    const uint64_t* row(int j) const { return beta.data() + (size_t)j * (size_t)WV; }
    void flip(int j, int k) { row(j)[k >> 6] ^= 1ULL << (k & 63); row(k)[j >> 6] ^= 1ULL << (j & 63); }
    int eval(const uint64_t* u) const;                       // Q(u), Z4
    int polar(const uint64_t* u, const uint64_t* v) const;   // B(u, v)
};

// The exact Gauss sum of i^{Q} over the NON-keep variables, with the keep variables as parameters:
//     Σ_{u_nonkeep} i^{Q(u)} = 2^{h/2} ω^{p8} · [keep ∈ A] · i^{Qf(free)},   ω = e^{iπ/4},
// A an affine subspace of the keep assignments, parametrised by the free keep variables (`free_keep`, ascending
// original index); every other keep variable is dep_c[i] ⊕ parity(dep_mask[i] & u) over the ORIGINAL indices (only
// free keep bits set). zero: the sum vanishes identically (an even coefficient with no partner and odd target).
// Exact integer/GF(2) operations only (variable elimination in ascending index order; the even case substitutes the
// lowest non-keep partner, else the lowest keep partner, which turns that keep variable dependent).
struct GaussSum {
    bool zero = false;
    int h = 0, p8 = 0;
    std::vector<int> free_keep;
    QForm f;                                // over free_keep (index i ↔ free_keep[i]); constant 0 (folded into p8)
    std::vector<int> dep_var;
    std::vector<uint8_t> dep_c;
    std::vector<uint64_t> dep_mask;         // dep_var.size() × WV (original indices)
    std::vector<uint8_t> alive_;            // workspace (capacity kept across calls)
    std::vector<uint64_t> S_, N_;
};
// Q is consumed (modified in place); G's vectors are reused.
void gauss_sum(QForm& Q, const std::vector<uint8_t>& keep, GaussSum& G);

// RF_CONTENT_MALFORMED (std::invalid_argument) on a CZ pair out of range or on the diagonal.
DiagExpansion expand_diag(const DiagNormalForm& nf, int n);
// D† M D (exact phase), D = S^{a&1}·CZ^M — equal to diag_pullback(nf, M).
Pauli dress_diag(const DiagExpansion& E, const Pauli& M);

}  // namespace qeccore
