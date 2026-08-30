#pragma once
// CertifiedGroupPlanes — columnar bit-planes + RREF pivot table for a certified
// (real, Hermitian ±1) stabiliser group.
//
// The substrate for the diagonal-twirl kernel (Phase B, Task 5). Built ONCE from a
// vector of certified stabiliser Paulis (e.g. FramedSuperposition::certified_stabilizers(),
// exact-phase +1 stabilisers). Two views over the same group:
//
//   1. Columnar bit-planes (xcol/zcol): for each qubit q, an n_gens-bit column packing
//      bit i = gens[i].xbit(q) (resp. zbit). Tasks 7–10 XOR these columns to build sign
//      planes without touching the row-major Paulis.
//   2. A GF(2) reduced-row-echelon of the generators' (x|z) symplectic rows, with exact
//      phase tracked through every row combination (real ±1 group ⇒ signs are meaningful).
//      reduce(P) classifies an arbitrary probe Pauli against the group, pivot-locally and
//      allocation-light (reusable thread_local scratch, NO per-call heap churn).
//
// reduce() verdicts:
//   IN_GROUP — P's support (x|z) lies in the group; `sign` is the ±1 of THE group element
//              G with that support (P = i^k · G). rep = residual (identity support, exact phase).
//   LOGICAL  — P commutes with the whole group but its support is NOT in it; rep = the reduced
//              residual w = P·G with exact phase (a normaliser/logical representative).
//   ANTI     — P anticommutes with at least one generator (not in the normaliser).

#include <cstdint>
#include <vector>
#include "qeccore/pauli.hpp"

namespace qeccore {

struct Membership {
    enum Verdict { IN_GROUP, LOGICAL, ANTI };
    Verdict verdict = ANTI;
    int sign = 0;        // IN_GROUP: +1 or -1; otherwise 0
    Pauli rep;           // reduced residual w = P·G (IN_GROUP: identity support, exact phase)

    static Membership in_group(int s, Pauli r) { return {IN_GROUP, s, std::move(r)}; }
    static Membership logical(Pauli r)         { return {LOGICAL, 0, std::move(r)}; }
    static Membership anti(Pauli r)            { return {ANTI, 0, std::move(r)}; }
};

struct CertifiedGroupPlanes {
    int n_qubits = 0, n_gens = 0, words = 0;   // words = ceil(n_gens/64)
    std::vector<uint64_t> xcol, zcol;          // [n_qubits * words]; column q at offset q*words
    std::vector<Pauli>    gens;                 // row view (exact phases), as supplied

    // GF(2) RREF of the gens' (x|z) rows (columns 0..n-1 = X-block, n..2n-1 = Z-block),
    // exact phase carried on each row; built once.
    std::vector<Pauli> rref;                    // rank pivot rows, ascending pivot column
    std::vector<int>   pivcol;                  // pivcol[r] = symplectic pivot column of rref[r]
    std::vector<int>   col_to_row;              // size 2n; col_to_row[c] = pivot row of column c, else -1
    int rank = 0;

    mutable uint64_t token_cache_ = 0;          // memoised identity_token() (0 == not yet computed)
    mutable bool     token_ready_ = false;

    // ── Task 11.5: product-wire signs (canonicalize_mod_stabilizers substrate). ──
    // A "product wire" q is one whose ±Z_q is a WEIGHT-1 certified stabiliser: it appears as an rref
    // row with symplectic pivot in the Z-block (pivcol ≥ n ⇒ X-support empty) whose z-support is
    // exactly {q}. Such a wire commutes with the whole group (g_i.x[q]=0 ∀i), so stripping its S/CZ
    // legs is invisible to the dressing lattice {M·x_i} — the lattice-safe reduction. prod_sign[q]
    // holds ±Z_q's sign (Z_q = sign·(+1 op), sign = ((p>>1)&1)?−1:+1), or 0 for non-product wires.
    // (The GENERAL multi-qubit-stabiliser axis reduction — reducing a whole pure-Z RREF — is a sound
    // state rewrite but changes M·x_i / κ; deferred, since build_shot_law implements only κ≤1.)
    struct ZAxisRREF {
        std::vector<int8_t> prod_sign;              // size n: ±1 for product wires, else 0
    };
    mutable ZAxisRREF zaxis_;
    mutable bool      zaxis_ready_ = false;
    const ZAxisRREF& z_axis_rref() const;

    // ── Group-only law substrate (cold-build hoist, 2026-07-15): G_z and N_z depend ONLY on the
    // certified group, not on the shot's residual — build_shot_law used to recompute both per cold
    // key (an ng-scale left_deps + an ng-scale nullspace, ~40% of the 1.4 ms cold build at n=298).
    // Cached lazily here, byte-identical (same computation, hoisted). gz_rref/gz_piv hold the RREF
    // basis of G_z = { Σ c_i z_i : Σ c_i x_i = 0 } (pure-Z group content, n-bit rows); nz_basis the
    // nullspace basis of the generators' X-rows (the Z-normalizer directions, n-bit rows), in the
    // exact construction order the in-law computation produced. Defined in twirl_law.cpp (needs its
    // GF(2) helpers). (N_z is no longer materialised: Kz is computed in the V-basis α-space.)
    struct GroupZCache {
        std::vector<std::vector<uint64_t>> gz_rref;
        std::vector<int>                   gz_piv;
        std::vector<int>                   gz_col_to_row;   // qubit column → rref row (-1 = free);
                                                            // support-directed reduce (2026-07-16)
    };
    mutable GroupZCache gzc_;
    mutable bool        gzc_ready_ = false;
    const GroupZCache& group_z_cache() const;

    // N_z = { u ∈ F2^n : u·x_i = 0 ∀i } — the full normalizer-Z frame (G_z ⊂ N_z), RREF'd
    // with the same support-directed column map. Group-only; lazy (V3 observable channels:
    // reachability is solved MOD N_z — the executable spec's gate killed the mod-G_z rule).
    struct NzCache {
        std::vector<std::vector<uint64_t>> rref;
        std::vector<int>                   piv;
        std::vector<int>                   col_to_row;      // qubit column → rref row (-1 = free)
    };
    mutable NzCache nzc_;
    mutable bool    nzc_ready_ = false;
    const NzCache& nz_cache() const;

    static CertifiedGroupPlanes build(const std::vector<Pauli>& certified, int n_qubits);

    // pattern(q, xblock=true): the n_gens-bit column (words uint64) for the X- or Z-block of qubit q.
    const uint64_t* col(int q, bool xblock) const {
        return (xblock ? xcol.data() : zcol.data()) + (size_t)q * words;
    }

    // identity_token — FNV-1a over the group's OWN content (n_qubits, n_gens, each gen's x/z words +
    // phase), the group-identity component of the plan-cache key. Computed once and cached ON THIS
    // OBJECT (a pure function of the group's content — no aliasing hazard, unlike a map keyed on the
    // object's address). The plan memo used to recompute this O(n_gens·n) FNV on EVERY get_or_build
    // (Task 10 warm-cost floor: ~35µs/shot at n=298); this lazy cache retires that per-shot cost.
    // Mutable so it can memoise through a const CertifiedGroupPlanes& (the way the memo holds it).
    uint64_t identity_token() const;

    // Classify probe p against the group. Pivot-local, allocation-light.
    Membership reduce(const Pauli& p) const;

    // Verdict + exact residual i-phase WITHOUT materialising the rep (2026-07-16 plan-build
    // optimization: the b_c / teeth loops only consume (verdict, phase); the full reduce()
    // copies the rep Pauli — two vector allocations per call). Same reduction, same phase
    // convention ((phase>>1)&1 = the ± bit for IN_GROUP). phase_out is meaningful only for
    // IN_GROUP verdicts.
    Membership::Verdict reduce_cheap(const Pauli& p, int& phase_out) const;
};

}  // namespace qeccore
