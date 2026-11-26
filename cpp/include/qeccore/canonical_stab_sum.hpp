#pragma once
#include <complex>
#include <functional>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/frame.hpp"
#include "qeccore/stab_affine.hpp"
#include "qeccore/state_budget.hpp"

namespace qeccore {

// CanonicalStabSum (v2): a superposition Σ_{i<χ} c_i |φ_i⟩ of stabilizer branches that
// share a single stabilizer group up to signs, stored in the form of Definition 2 of
// an internal derivation note (not shipped):
//   * a Clifford frame U  — generators g_a = U Z_a U†, destabilisers d_a = U X_a U†;
//   * anchor signs eps[a]∈{0,1}  — g_a |φ0⟩ = (-1)^eps[a] |φ0⟩;
//   * an exact anchor state |φ0⟩ (an AffineState, global phase included);
//   * a MINIMAL set `free` of distinguishing generator indices, |free| = r;
//   * a list of branches, each a sign pattern σ (length r, over `free`) and coefficient c.
// Branch i is  |φ_i⟩ = (∏_{a∈free: σ_i[a]=1} d_a) |φ0⟩.
//
// This header declares the complete type: construction + invariants + χ-independent Clifford gates
// (apply_h / apply_s / … / apply_cz). (bloat-cut B1: the measurement / Born-probability / clone /
// statevector paths had no caller and are deleted.)
struct CanonicalStabSum {
    // frame: g_a = U Z_a U† (Zrow), d_a = U X_a U† (Xrow). S2-T3: a qeccore::Frame — from_coset_family builds it
    // COMPACT (the block U0 on the support of ray_0's non-isolated generators, the isolated Z_q rows as the index map π)
    // and the χ-independent gates are appended to its C; a reader that needs the dense rows calls U.dense()
    // (expands once: from_generators-equivalent rows with C replayed by left_batch — today's tableau byte for byte).
    Frame U;
    std::vector<uint8_t> eps;              // anchor signs: g_a|φ0⟩ = (-1)^eps[a]|φ0⟩ (length n)
    // |φ0⟩ held exactly (global phase). lean L6: read through anchor_state(), which first replays the Clifford gates
    // applied since the anchor was last read (apply_* records them; the frame U is replayed by flush_gates the same
    // way) — the same gate sequence on the same state as the former eager update, so every reader sees the identical
    // AffineState; a consumer that never reads the anchor (from_css: U, eps, free, branches) never pays for it.
    const AffineState& anchor_state() const { flush_anchor_(); return *anchor_; }
    AffineState& anchor_state() { flush_anchor_(); return *anchor_; }
    bool has_anchor() const { return anchor_ != nullptr; }
    size_t anchor_pending() const { return anchor_pending_.size(); }   // gates not yet applied to the anchor (tests)
    void set_anchor(std::unique_ptr<AffineState> a);      // a fresh anchor (no gate may be pending on the old one)
    std::vector<int> free;                 // distinguishing generator indices (into U), MINIMAL
    struct Branch {
        std::vector<uint8_t> sigma;        // sign pattern over `free` (length r)
        std::complex<double> c;            // coefficient
    };
    // χ entries; allocated through the process-wide state budget (state_budget.hpp).
    using BranchVec = std::vector<Branch, StateAllocator<Branch, kSigmaHeapBytes>>;
    BranchVec branches;

    // DEFERRED Clifford tableau conjugation. apply_* only RECORDS the gate, in the frame's gate list U.gates()
    // (S2-T3: Frame::C; formerly pending_gates) and in anchor_pending_ (lean L6: the anchor is deferred too, replayed
    // by anchor_state() on its first read); the frame conjugation is replayed into the dense tableau (left_batch) only
    // when a reader asks for U.dense(). Replay order MUST equal the apply order. (kind 0:h 1:s 2:sdg 3:x 4:y 5:z
    // 6:cx 7:cz; a = qubit/control, b = target for cx/cz, else -1.)
    using PendingGate = CliffordTableau::LeftGate;

    explicit CanonicalStabSum(int n) : U(Frame::expanded(CliffordTableau(n))), eps(n, 0) {}
    // The same empty sum on a given frame (taken over, never copied): a construction that computes its frame
    // first holds no identity placeholder tableau beside it.
    explicit CanonicalStabSum(CliffordTableau&& frame) : U(Frame::expanded(std::move(frame))), eps((size_t)U.n, 0) {}
    // S2-T3 (amendment F4): the same on a compact frame (from_coset_family's).
    explicit CanonicalStabSum(Frame&& frame) : U(std::move(frame)), eps((size_t)U.n, 0) {}

    int n() const { return U.n; }
    int chi() const { return (int)branches.size(); }

    // Build from explicit (normalized) rays + coefficients: anchor = rays[0], frame from its
    // stabilizers, branches from the per-ray sign syndromes (the overlap-based rebuild).
    // One-time construction path (bare-state assembly); O(chi^2 * n) — fine at construction chi.
    static CanonicalStabSum from_rays(int n, std::vector<std::unique_ptr<AffineState>> rays,
                                      std::vector<std::complex<double>> coeffs);

    // Build from a COSET FAMILY without materialising the χ rays (the bare-state hot path).
    // The family: ONE stabilizer state ψ and r commuting, independent Paulis zs[0..r-1] that split ψ
    // evenly (no nontrivial product of them lies in ±Stab(ψ)); make_ray(g), g ∈ {0..2^r-1}, returns
    // the unit ray 2^{r/2}·Π_g|ψ⟩ projected onto the joint eigenspace g — exactly what
    // assemble_from_parities' Z-string projections produce.
    // Result: BYTE-IDENTICAL to from_rays(n, {make_ray(g)}_{g<2^r}, coeffs) — same frame, eps,
    // anchor, free, branch order, sign patterns and coefficient doubles.
    //
    // 3.1.9: nothing but ray 0 (the anchor) and the spot-check probes is materialised.  ray_g =
    // M_g ray_0 with M_g = Π_i s_i^{g_i}, s_i ∈ Stab(ψ) anticommuting with zs[i] alone, so the unit-ray
    // syndromes are commutation bits of the frame generators with the s_i, and the weight ≤ 2 gauges
    // are frame expectations ⟨ray_0|M_g D_σ(g)|ray_0⟩ of Pauli products (derivation in the .cpp).
    // A deterministic spot check re-derives a few further branches the old way (projected ray +
    // exact inner product) and throws BareStateInvariantError on any disagreement.
    // `trace` (tests only) records the intermediate syndromes and gauge exponents.
    struct CosetTrace {
        std::vector<std::vector<uint8_t>> fs;   // initial-frame syndromes: [0] = ray 0, [1+i] = ray e_i
        std::vector<int> a1;                    // z8 of α(e_i)
        std::vector<int> a2;                    // z8 of α(e_i ⊕ e_j), i < j, row-major
        std::vector<uint8_t> b2;                // b_ij (r×r, upper triangle)
        std::vector<uint32_t> u;                // re-based linear syndrome columns (n)
    };
    static CanonicalStabSum from_coset_family(
        int n, int r, const AffineState& psi, const std::vector<Pauli>& zs,
        const std::function<std::unique_ptr<AffineState>(uint32_t)>& make_ray,
        const std::vector<std::complex<double>>& coeffs, CosetTrace* trace = nullptr);
    // TEST ORACLE (the 3.1.8 construction): the same object read off the materialised rays of weight
    // ≤ 2 (each disentangled for its syndrome, each gauge an exact inner product).
    static CanonicalStabSum from_coset_family_rays(
        int n, int r, const std::function<std::unique_ptr<AffineState>(uint32_t)>& make_ray,
        const std::vector<std::complex<double>>& coeffs, CosetTrace* trace = nullptr);

    // χ-independent Clifford gates (theory §8): conjugate the shared frame U (its generators g_a
    // and destabilisers d_a) and apply the SAME gate to the anchor |φ0⟩; eps/free/branches are
    // left untouched. Conjugating U transports the d_a, so |φ_i⟩ = (∏ d_a)|φ0⟩ evolves correctly.
    // Both are deferred (pending_gates, anchor_pending_): a call records the gate, independent of χ.
    void apply_h(int q);
    void apply_s(int q);
    void apply_sdg(int q);
    void apply_x(int q);
    void apply_y(int q);
    void apply_z(int q);
    void apply_cx(int c, int t);
    void apply_cz(int c, int t);

    // Re-select `free` as a minimal independent basis of the surviving sign-variation (theory §6 /
    // Lemma 2): GF(2) column-echelon of the χ×r matrix {σ_i ⊕ σ_0}; `free` ← pivot columns; restrict
    // every branch's σ to the new basis; drop coincidentally-fixed generators from `free`. Restores
    // Invariant (Minimality). Idempotent. O(χ·r).
    void canonicalise();

    // Distinctness (σ patterns pairwise distinct) AND Minimality (GF(2) rank of {σ_i⊕σ_0}
    // equals |free| and χ == 2^|free|). The distinctness scan is O(χ²) — for hot paths that only
    // need a current frame (not a full audit), call ensure_frame_current() instead.
    bool verify_invariants() const;

    // Make the dense frame current: U.dense() (expands a compact frame, flushes the pending gates) — the
    // side-effect verify_invariants() also performed before S2-T3, WITHOUT its O(χ²) distinctness / O(χ·r²)
    // minimality validation. For the dense readers (oracles: css_first_diff); from_css no longer calls it (it
    // moves the Frame in whatever shape it has).
    void ensure_frame_current() const;

  private:
    std::unique_ptr<AffineState> anchor_;                 // see anchor_state()
    mutable std::vector<PendingGate> anchor_pending_;     // the gates applied since the anchor was last read
    void flush_anchor_() const;
    // The dense tableau, current (U.dense(): expand + replay the pending gates by left_batch, in apply order).
    // Called by the methods that read dense rows (the oracle paths). const: the Frame's shape is mutable.
    void flush_gates() const;
    // d_a = U X_a U† on wires, current: the dense row when expanded, else the Frame's forward image (I3: the same
    // unique operator) — the compact readers (branch_gauge's spot check, canonicalise's fold).
    Pauli xrow_on_wires_(int a) const;

    // Overlap-based structure build: the one-time from_rays construction path.
    void rebuild_from_rays(std::vector<std::unique_ptr<AffineState>> rays,
                           std::vector<std::complex<double>> coeffs);

    // Per-ray generator-sign bits under the CURRENT frame U.Zrow (the disentangle read shared by
    // rebuild_from_rays and from_coset_family): fs[i][a] = sign bit of U.Zrow[a] on *rays[i].
    void ray_syndromes(const std::vector<const AffineState*>& rays,
                       std::vector<std::vector<uint8_t>>& fs) const;
    // The gauge rebuild_from_rays divides by: ⟨ray|D_σ anchor⟩ with D_σ = Π_{d: σ_d} Xrow[free[d]]
    // applied in ascending d (exact; a unit phase when ray and D_σ anchor are the same ray).
    ExactPhase branch_gauge(const AffineState& ray, const std::vector<uint8_t>& sigma) const;
};

}  // namespace qeccore
