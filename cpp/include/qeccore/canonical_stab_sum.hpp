#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/stab_affine.hpp"

namespace qeccore {

// CanonicalStabSum (v2): a superposition Σ_{i<χ} c_i |φ_i⟩ of stabilizer branches that
// share a single stabilizer group up to signs, stored in the form of Definition 2 of
// docs/v2-form-invariance.tex:
//   * a Clifford frame U  — generators g_a = U Z_a U†, destabilisers d_a = U X_a U†;
//   * anchor signs eps[a]∈{0,1}  — g_a |φ0⟩ = (-1)^eps[a] |φ0⟩;
//   * an exact anchor state |φ0⟩ (an AffineState, global phase included);
//   * a MINIMAL set `free` of distinguishing generator indices, |free| = r;
//   * a list of branches, each a sign pattern σ (length r, over `free`) and coefficient c.
// Branch i is  |φ_i⟩ = (∏_{a∈free: σ_i[a]=1} d_a) |φ0⟩.
//
// This header declares the complete type: construction + materialisation + invariants + clone
// + χ-independent Clifford gates (apply_h / apply_s / … / apply_cz) + single/batch Pauli
// measurement (measure_single / measure_batch) + Born probabilities (born_probabilities_single /
// born_probabilities).
struct CanonicalStabSum {
    CliffordTableau U;                       // frame: g_a = U Z_a U† (Zrow), d_a = U X_a U† (Xrow)
    std::vector<uint8_t> eps;              // anchor signs: g_a|φ0⟩ = (-1)^eps[a]|φ0⟩ (length n)
    std::unique_ptr<AffineState> anchor;   // |φ0⟩ held exactly (global phase)
    std::vector<int> free;                 // distinguishing generator indices (into U), MINIMAL
    struct Branch {
        std::vector<uint8_t> sigma;        // sign pattern over `free` (length r)
        std::complex<double> c;            // coefficient
    };
    std::vector<Branch> branches;          // χ entries

    // DEFERRED Clifford tableau conjugation. apply_* updates ONLY the anchor (eager, ~bare-affine
    // cost) and RECORDS the gate here; the O(n²)-per-gate frame conjugation (U.left_*) is replayed
    // lazily by flush_gates() the first time the frame U is actually read (measurement / materialise /
    // born / clone). Mirrors the lazy-dual (`ensure_dual`) idiom: `mutable` so const frame-readers can
    // materialise. Replay order MUST equal the apply order. (kind 0:h 1:s 2:sdg 3:x 4:y 5:z 6:cx 7:cz;
    // a = qubit/control, b = target for cx/cz, else -1.)
    struct PendingGate { uint8_t kind; int a; int b; };
    mutable std::vector<PendingGate> pending_gates;

    explicit CanonicalStabSum(int n) : U(n), eps(n, 0) {}

    int n() const { return U.n; }
    int chi() const { return (int)branches.size(); }

    // Mirror the legacy v1 from-recipe convention with identical args: a random Clifford base
    // state, its stabilizer generators, the first m as the distinguishing (free) set, and the
    // 2^m sign-pattern branches with `coeffs` (branch ordering is the historical v1 ordering).
    static CanonicalStabSum from_recipe(int n, int m,
            const std::vector<std::complex<double>>& coeffs, uint64_t seed);

    // Build from explicit (normalized) rays + coefficients: anchor = rays[0], frame from its
    // stabilizers, branches from the per-ray sign syndromes (the overlap-based rebuild).
    // One-time construction path (bare-state assembly); O(chi^2 * n) — fine at construction chi.
    static CanonicalStabSum from_rays(int n, std::vector<std::unique_ptr<AffineState>> rays,
                                      std::vector<std::complex<double>> coeffs);

    // χ-independent Clifford gates (theory §8): conjugate the shared frame U (its generators g_a
    // and destabilisers d_a) and apply the SAME gate to the anchor |φ0⟩; eps/free/branches are
    // left untouched. Conjugating U transports the d_a, so |φ_i⟩ = (∏ d_a)|φ0⟩ evolves correctly.
    // Cost is O(n²) per gate, independent of χ.
    void apply_h(int q);
    void apply_s(int q);
    void apply_sdg(int q);
    void apply_x(int q);
    void apply_y(int q);
    void apply_z(int q);
    void apply_cx(int c, int t);
    void apply_cz(int c, int t);

    // Single-qubit Pauli measurement (theory §§3–6). pauli ∈ {0:X, 1:Y, 2:Z}; q the qubit; u∈[0,1)
    // the Born random. Computes Q = U†·single_pauli(pauli,q)·U and A = supp(Q.x). Case A (A empty,
    // Q commutes with every generator): per-branch eigenvalue λ_i, deterministic or Born-sampled
    // collapse, then canonicalise(). Case B (A nonempty, anticommuting): delegates to
    // measure_single_anticommuting (pivot rotation + Born sample + closed-form recompute, theory §5).
    // Returns the outcome m ∈ {+1,-1}.
    int measure_single(int pauli, int q, double u);
    // Measure a SEQUENCE of single-qubit Paulis ((pauli, q) pairs, one uniform u each), returning
    // the ±1 outcomes. Statistically identical to calling measure_single in the given order, but
    // consecutive COMMUTING (Case A) measurements share one deferred collapse: each branch is a
    // simultaneous eigenstate of every commuting operator, so the run only tracks per-branch
    // eigenvalue bits and the chain-rule probabilities; branches are dropped, rescaled and
    // canonicalise()d ONCE per run instead of once per measurement. An anticommuting measurement
    // flushes the run and takes the full Case-B path, then batching resumes. (Outcome-for-outcome
    // equal to the sequential calls up to floating-point regrouping of the chain probabilities.)
    std::vector<int> measure_batch(const std::vector<std::pair<int, int>>& paulis,
                                   const std::vector<double>& us);

    // True iff Q = U†·single_pauli(pauli,q)·U commutes with EVERY generator (A = supp(Q.x) empty),
    // i.e. the measurement is Case A (commuting). When false it is Case B (anticommuting).
    bool is_commuting_single(int pauli, int q) const;

    // Born probabilities (p_+, p_-) for a single-qubit Pauli measurement WITHOUT collapsing.
    // Handles both commuting (Case A) and anticommuting (Case B) via born_from_conjugated;
    // does not collapse the state and does not throw.
    std::pair<double, double> born_probabilities_single(int pauli, int q) const;
    // Born probabilities (p₊, p₋) for an arbitrary Hermitian n-qubit Pauli P (i^ph X^x Z^z with
    // ph = #Y mod 4 — the historical v1 born-probability convention); ⟨P⟩ = p₊ − p₋. Closed
    // form from the frame algebra (no overlaps, no collapse, no clone): Q = U†PU via dual_image,
    // then the same commuting/anticommuting evaluation as the single-qubit path.
    std::pair<double, double> born_probabilities(const Pauli& P) const;

    // Re-select `free` as a minimal independent basis of the surviving sign-variation (theory §6 /
    // Lemma 2): GF(2) column-echelon of the χ×r matrix {σ_i ⊕ σ_0}; `free` ← pivot columns; restrict
    // every branch's σ to the new basis; drop coincidentally-fixed generators from `free`. Restores
    // Invariant (Minimality). Idempotent. O(χ·r).
    void canonicalise();

    // Dense materialisation: Σ_i c_i · D_{σ_i}|φ0⟩ (test/oracle path; O(χ·2ⁿ)).
    std::vector<std::complex<double>> to_statevector() const;

    // Distinctness (σ patterns pairwise distinct) AND Minimality (GF(2) rank of {σ_i⊕σ_0}
    // equals |free| and χ == 2^|free|). The distinctness scan is O(χ²) — for hot paths that only
    // need a current frame (not a full audit), call ensure_frame_current() instead.
    bool verify_invariants() const;

    // Flush pending Clifford gates so the frame U is current — the side-effect verify_invariants()
    // also performs, WITHOUT its O(χ²) distinctness / O(χ·r²) minimality validation. Use at public
    // entry points (e.g. materialize_rays) that read U and need it current but do not
    // need to re-audit the (already-trusted) branch set.
    void ensure_frame_current() const;

    CanonicalStabSum clone() const;
    // Copy *this into dst, REUSING dst's existing heap buffers wherever shapes match: after the
    // first call on a given dst, a steady-state per-shot loop (master.clone_into(work)) performs
    // no allocations — clone() by contrast pays ~8n fresh Pauli buffers per call. Observably
    // identical to `dst = clone()`.
    void clone_into(CanonicalStabSum& dst) const;

  private:
    // Replay every pending Clifford gate onto the frame U (U.left_*, in apply order) and clear the
    // list, making the tableau current. Called at the start of EVERY method that READS U (so the
    // frame is never stale when a branch d_a is materialised/projected). No-op when empty. `mutable`/
    // const so const frame-readers (to_statevector, born_probabilities_single, clone, ...) can flush.
    // U.left_* already lazy-rebuilds the inverse tableau on demand, so this composes with ensure_dual.
    void flush_gates() const;

    // Case B (anticommuting) of measure_single (theory §5). P is the single-qubit Pauli, A =
    // supp((U†PU).x) the nonempty anticommuting-generator set, u the Born random. Rotates the
    // anticommuters onto a pivot, samples the Born outcome (Eq. 6), fuses paired branches with
    // the 1/√2 unit-branch coefficient (Eq. 8), rebuilds the frame + anchor, then canonicalise().
    int measure_single_anticommuting(int pauli, int q, const Pauli& P,
                                     const std::vector<int>& A, double u);

    // clone()'s private copy path: copy-constructs U directly from o.U (the CliffordTableau copy
    // ctor, which skips a stale inverse tableau) instead of building a throwaway n-qubit identity
    // frame that the old `CanonicalStabSum out(n); out.U = U;` flow allocated and immediately
    // overwrote. Caller (clone) must flush_gates() first.
    struct CloneTag {};
    CanonicalStabSum(const CanonicalStabSum& o, CloneTag);

    // Shared Born evaluation from the frame-conjugated operator Q = U†PU (Case A commuting /
    // Case B anticommuting, clone-free). Used by born_probabilities{,_single}.
    std::pair<double, double> born_from_conjugated(const Pauli& Q) const;

    // CLOSED-FORM (overlap-free) structure recompute (docs §9). Each surviving group carries its
    // ORIGINAL representative destabiliser word `word` (bits over generator indices into the
    // ORIGINAL frame `origU`/`orig_anchor`, i.e. which d_a applied to |φ0⟩) and its closed-form
    // coefficient `coeff`. We project ONE representative per group (pauli_project on Π_m), read each
    // group's syndrome by SINGLE-STATE generator-sign reads, fix each group's gauge via a
    // single-state amplitude_at_bits ratio (NO pairwise ⟨φ_i|φ_j⟩ overlaps), then re-base dependent
    // distinguishing generators + canonicalise(). The post-measurement frame `U` must already be set
    // (Case B: rotated onto the pivot, dual reset; Case A: unchanged). `pauli`,`q`,`m` define Π_m.
    // `word` is WORD-PACKED: bit a (word a>>6, bit a&63) = destabiliser d_a applied, ceil(n/64)
    // words, trailing bits zero — the gauge/projection consumers copy or bit-scan it directly.
    struct SurvGroup { std::vector<uint64_t> word; std::complex<double> coeff; };
    // INCREMENTAL O(n²) recompute. The post-measurement frame must ALREADY be set in `this->U`
    // (Case A: unchanged; Case B: pivot rotated + pivot generator replaced by ±P + dual reset) — we
    // do NOT re-extract it. `fsynd` is the closed-form sign matrix of the generators on each
    // surviving group's projected representative (known from the rotation algebra, no projections),
    // WORD-PACKED row-major: bit a of row g (word g*W + (a>>6), bit a&63, W = ceil(n/64)) is the
    // sign of g_a = U.Zrow[a] on group g.
    // LEAD 2: we project ONLY group-0's representative (the single O(n²) pauli_project → new anchor);
    // every other ray is reachable as a Pauli bridge off it, so it needs no projection. LEAD 1: each
    // per-branch coefficient gauge γ_g is then a CLOSED FORM in two diagonal stabiliser expectations
    // on the ORIGINAL anchor (`oXrow`/`oZrow`/`old_eps`/`orig_anchor`), evaluated entirely in the old
    // frame's coordinates (χ-flat dual_image count) — NO per-branch projection, NO amplitude solve, NO
    // D_σ|anchor⟩ state rebuild. Then re-base dependents + canonicalise(). `oXrow`/`oZrow` are the
    // ORIGINAL frame's FORWARD rows (passed as plain vectors, not a full CliffordTableau, to avoid deep-
    // copying the inverse tableau); they materialise the ORIGINAL destabiliser words (docs §9 remark
    // iii) and supply the gauge's old-frame conjugation. `frame_patched` = true when the caller already
    // edited `this->U`'s forward rows (Case B pivot patch) and left its inverse tableau stale — recompute
    // then invalidates the dual (lazy rebuild). Case A passes false (U fully unchanged).
    // groups/fsynd are passed by reference so the caller can hand in per-thread reusable buffers (no
    // per-collapse heap churn). `groups` is read-only here; `fsynd` is mutated in place by the re-basis
    // echelon (the caller's buffer is scratch, regenerated each collapse, so in-place edits are safe).
    void recompute_from_words(const std::vector<Pauli>& oXrow, const std::vector<Pauli>& oZrow,
                              const AffineState& orig_anchor,
                              const std::vector<uint8_t>& old_eps,
                              const std::vector<SurvGroup>& groups,
                              std::vector<uint64_t>& fsynd,
                              int pauli, int q, int m, bool frame_patched,
                              bool dual_maintained = false);

    // Overlap-based structure recompute (the O(χ²) pairwise-overlap version). Debug oracle for
    // the closed-form measurement path (cross-checked under !NDEBUG, which never calls it in
    // release) AND the one-time from_rays construction path — always compiled.
    void rebuild_from_rays(std::vector<std::unique_ptr<AffineState>> rays,
                           std::vector<std::complex<double>> coeffs);
};

}  // namespace qeccore
