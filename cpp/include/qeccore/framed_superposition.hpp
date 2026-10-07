// FramedSuperposition — the unified sampler state (tableau frame + Pauli-coefficient
// amplitudes) and its anticommuting / batched measurement kernels.
//
// Production extraction of the validated spike (was cpp/tests/test_lean_anticommuting.cpp).
// Implements docs/superpowers/specs/2026-06-27-lean-tableau-sampling-rep-design.md §3.4 (+§2
// primitive) WITHOUT the affine `anchor` / pauli_project / dual_image gauge solve that
// CanonicalStabSum::measure_single_anticommuting carries. The affine path remains the EXACT
// oracle; this lean path was proven to reproduce it branch-for-branch on the d5 cultivation
// stream + randomized chi<=4 states (exact vs statevector) + statistical d5 active-block checks.
//
// THE LEAN STATE. Built by reading an existing chi-bounded CanonicalStabSum:
//   * reference |psi> = its signed tableau (frame U + eps); the AFFINE anchor is IGNORED.
//   * amplitudes    = its branches mapped to (sigma over `free`, coeff c) entries, stored in the
//                     swappable Amplitudes container. In frame coordinates entry i's Pauli
//                     P_i = prod_{a:sigma} d_{free[a]} is a pure X-word.
//
// PUBLIC API:
//   FramedSuperposition           -- the state type (frame + eps + free + alpha) + its framed
//                                    method surface (framed_superposition.cpp)
//   FramedSuperposition::from_css -- build one from a CanonicalStabSum (conversion point)
//   framed_canonicalise           -- re-select a minimal `free` basis (frame coords, tableau-only)
//   framed_measure_anticommuting  -- single-qubit anticommuting (Case B) collapse; mutates L
//   batch_chi1                    -- joint outcome of a COMMUTING read set on a chi==1 state
//   batch_measure                 -- general-chi batch: pass-1 collapses, pass-2 batch_chi1

#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "qeccore/amplitudes.hpp"
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

// FramedSuperposition: the shared-frame stabilizer superposition
//   |φ⟩ = Σ_x α_x (∏_{d: x_d=1} d_{free[d]}) |ψ⟩,
// where U.Zrow = signed stabilizers s_a, U.Xrow = destabilizers d_a, eps carries the reference-
// state signs s_a|ψ⟩ = (-1)^{eps[a]}|ψ⟩, `free` are the k distinguished destabiliser indices, and
// α is the coefficient amplitude over the free indices. Frame + eps + free + amplitudes; no anchor.
//
// ONE type since Stage 2/3 of the 2026-07-02 migration (docs/superpowers/specs/): the Stage-1
// alpha-backed lean struct carries the framed method surface (from_css / expectation /
// apply_clifford / measure_pauli / measure_pauli_batch, implemented in framed_superposition.cpp)
// under the FramedSuperposition name; the former lean struct name is gone (Stage-3 rename sweep).
//
// Amplitude storage (Stage 1): the former `branches` list lives behind the swappable
// `Amplitudes` seam. Backend: DenseAmplitudes — a sparse (sigma, coeff) key-list whose entry ORDER
// is the kernels' iteration order, carried by the container verbatim. NOTE the "entry i <-> dense
// x-index i" order invariant does NOT hold in the wild (cube_ccz's chi=8 bare state arrives in
// x-order 0 2 7 5 4 6 3 1 and canonicalises to 0 4 2 6 1 5 3 7), so the per-entry sigma keys are
// load-bearing: a truly dense 2^k backend would additionally need an iteration-order permutation.
struct FramedSuperposition {
    CliffordTableau U;               // frame: g_a = U Z_a U^dag (Zrow), d_a = U X_a U^dag (Xrow)
    std::vector<uint8_t> eps;        // g_a|psi> = (-1)^eps[a]|psi>
    std::vector<int> free;           // distinguishing generator indices into U
    // Amplitude entry: (sigma over `free`, coefficient) — DenseAmplitudes' storage element.
    using Entry = std::pair<std::vector<uint8_t>, std::complex<double>>;
    std::unique_ptr<Amplitudes> alpha;   // the amplitudes seam (owning; DenseAmplitudes backend)

    explicit FramedSuperposition(int n)
        : U(n), eps(n, 0),
          alpha(std::make_unique<DenseAmplitudes>(0)),
          dense_(static_cast<DenseAmplitudes*>(alpha.get())) {}
    FramedSuperposition(const FramedSuperposition& o)
        : U(o.U), eps(o.eps), free(o.free), alpha(o.alpha ? o.alpha->clone() : nullptr) {
        rebind_();
    }
    FramedSuperposition& operator=(const FramedSuperposition& o) {
        if (this == &o) return *this;
        U = o.U; eps = o.eps; free = o.free;
        if (alpha && o.alpha) {
            alpha->assign_from(*o.alpha);    // per-shot hot path: deep copy reusing our buffers
        } else {
            alpha = o.alpha ? o.alpha->clone() : nullptr;
            rebind_();
        }
        return *this;
    }
    FramedSuperposition(FramedSuperposition&& o) noexcept
        : U(std::move(o.U)), eps(std::move(o.eps)), free(std::move(o.free)),
          alpha(std::move(o.alpha)), dense_(o.dense_) {
        o.dense_ = nullptr;
    }
    FramedSuperposition& operator=(FramedSuperposition&& o) noexcept {
        U = std::move(o.U); eps = std::move(o.eps); free = std::move(o.free);
        alpha = std::move(o.alpha);
        dense_ = o.dense_; o.dense_ = nullptr;
        return *this;
    }

    int n() const { return U.n; }
    int k() const { return (int)free.size(); }
    int chi() const { return dense_ ? (int)dense_->b_.size() : (alpha ? alpha->support() : 0); }

    // ── The framed method surface (framed_superposition.cpp) ──────
    // Build from a chi-bounded CanonicalStabSum (the compile→sampling handoff, the conversion
    // point at bare-state load): frame made current, U + eps + free + branch (sigma, coeff)
    // entries copied verbatim, anchor dropped.
    static FramedSuperposition from_css(const CanonicalStabSum& s);
    double expectation(const Pauli& P) const;         // <P> = p+ - p-, real

    // ── Compile-time bare-state certification API (twirl fast path, Task 1) ──────────────────
    // certified_stabilizers(): Pauli generators g (phases included, g Hermitian) with g|psi> =
    // |psi> EXACTLY for this state. Derived FRAME-ONLY: for every reference generator g_a = U Z_a U†
    // (Zrow[a]) whose index a is NOT one of the distinguished `free` (logical) rows, g_a commutes
    // with every destabiliser word Xrow[free[d]] (symplectic frame ⇒ Z_a anticommutes only with
    // X_a), so it passes THROUGH the whole magic superposition |psi> = Σ_x α_x (∏_{d:x_d} Xrow[free[d]])|ref>
    // and inherits the reference eigenvalue (-1)^{eps[a]}. Returning (-1)^{eps[a]}·Zrow[a] thus gives an
    // EXACT +1 stabiliser of |psi>. For χ=1 (`free` empty) this is the full n-generator group; for
    // χ=2 it is the n−k anchor/code subgroup (the free rows are the magic-sector directions and are
    // NOT Pauli stabilisers — a proper subgroup, never a generator we cannot certify). This is the
    // SAME `Zrow[a] for a∉free` set expectation_frame_check uses as its fixed generators.
    std::vector<Pauli> certified_stabilizers() const;
    // pauli_expectation(P): the exact <psi|P|psi> for an arbitrary (possibly non-Hermitian) Pauli
    // P = i^{P.phase} X^x Z^z. WRAPS the existing exact facility framed_expectation (the born-probs
    // engine, bit-identical to CanonicalStabSum::born_probabilities): with H(x,z) = i^{x·z} X^x Z^z the
    // Hermitian canonical, <H> = p+ − p− is real and P = i^{(P.phase − x·z) mod 4}·H, so
    // <P> = i^{(P.phase − x·z) mod 4}·<H>.
    std::complex<double> pauli_expectation(const Pauli& P) const;
    void apply_clifford(uint8_t kind, int a, int b);  // 0:H 1:S 2:Sdg 3:X 4:Y 5:Z 6:CX 7:CZ
    int  measure_pauli(const Pauli& P, double u);     // general Pauli read; returns ±1
    // Born +1 probability p_+ = (1+⟨P⟩)/2 that measure_pauli would use for THIS read, WITHOUT
    // collapsing (const). Byte-identical to measure_pauli's internal pp for both the anticommuting-
    // active case (exact 0.5) and the commuting case (same exp_p arithmetic) — so a memoized coin
    // `u < born_p1(P)` reproduces measure_pauli(P,u)'s outcome bit exactly (decoder-feedback perf:
    // per-record memoized Born-decision coin; fold_p1 pattern).
    double born_p1(const Pauli& P) const;
    // Batched measurement of a set of COMMUTING single-qubit terminal reads (reads[k]=(pauli,qubit),
    // pauli 0:X 1:Y 2:Z, distinct qubits). Pass 1 collapses the ≤k reads that couple the distinguished
    // `free` generators via measure_pauli; Pass 2 batch-conjugates the branch-agreeing remainder in
    // ONE forward-row pass and reads them off with GF(2) coin bookkeeping (they are χ=1 reads — no
    // amplitude/container work). out[k] = ±1. Same joint distribution as per-read measure_pauli, at
    // ~O(reads·n) instead of O(reads·n²). `rng()` returns U[0,1).
    void measure_pauli_batch(const std::vector<std::pair<int, int>>& reads,
                             const std::function<double()>& rng, std::vector<int>& out);

    // ── Devirtualized fast path (cached DenseAmplitudes pointer; kernels compile to raw list
    //    ops, no virtual call per entry). entries()[i] = (sigma_i, c_i) in iteration order. ──
    std::vector<Entry>& entries() { return dense_->b_; }
    const std::vector<Entry>& entries() const { return dense_->b_; }
    std::vector<uint8_t>& sigma(int i) { return dense_->b_[i].first; }
    const std::vector<uint8_t>& sigma(int i) const { return dense_->b_[i].first; }
    uint8_t sigma_bit(int i, int d) const { return dense_->b_[i].first[d]; }
    std::complex<double>& coeff(int i) { return dense_->b_[i].second; }
    const std::complex<double>& coeff(int i) const { return dense_->b_[i].second; }
    // Keep the container's logical-index count in sync after `free` is edited directly.
    void sync_alpha_k() { if (dense_) dense_->k_ = (int)free.size(); }

  private:
    void rebind_() { dense_ = dynamic_cast<DenseAmplitudes*>(alpha.get()); }
    DenseAmplitudes* dense_ = nullptr;   // non-owning cache of alpha's dense backend (or null)
};

// Result of a single-qubit reference read (classify + collapse at u=0). See framed_reference_read.
struct FramedRefRead {
    int out;            // reference outcome +1/-1 (deterministic u=0 choice: +1 iff pp>1e-12)
    double pp, pm;      // pre-collapse Born probs P(+1),P(-1)
    bool is_coin;       // pp>1e-12 && pm>1e-12
    bool commuting;     // read commutes with the stabilisers (Case-A vs Case-B discriminator)
    int chi_pre;        // chi before collapse (caller needs it for the Case-A halving check)
};

// A coin fixed by an earlier read in a commuting batch (signed stabiliser of the post-state).
struct MeasCoin {
    Pauli Qc;
    int pivot;
    int o;
};

// ── Batched diagonal-Clifford conjugation of a frame (the per-bad-shot flush killer) ──────────
// Conjugate a CliffordTableau U by a diagonal Clifford D applied in the order S-powers, then CZ,
// then X^v:   U  ->  D · U   (forward rows M = U σ U† become D M D†), in ONE pass over the rows.
// This is BIT-IDENTICAL to applying the same error gate-by-gate via
//   U.left_s/left_z/left_sdg (per a_q∈{1,2,3}) ; U.left_cz (per cz pair) ; U.left_x (per v bit),
// the trusted deferred-gate replay path (flush_gates). A diagonal Clifford fixes the X-content of
// every row, so all the per-row z-flips and phase deltas are closed forms in the row's (fixed) X
// bits — one row scan replaces ~|a|+|cz|+|v| separate row scans. `a` is the S-power per qubit
// (size n, Z4: 1=S, 2=Z, 3=S†), `cz` the CZ pairs (qubit indices), `v` the X^v mask (size n).
// The inverse (dual) tableau is INVALIDATED (the lean reduction reads the forward rows only).
void conjugate_by_diag_clifford(CliffordTableau& U, const std::vector<uint8_t>& a,
                                const std::vector<std::pair<int, int>>& cz,
                                const std::vector<uint8_t>& v);

// Re-select `free` as a minimal independent basis of the surviving sign variation; fold constant
// columns into eps (no anchor). Mutates L in place. Allocation-free (thread_local scratch).
void framed_canonicalise(FramedSuperposition& L);

// Pauli expectation on a lean state: returns (p+, p-) with <P> = p+ - p-. This is
// CanonicalStabSum::born_probabilities re-expressed on the lean rep (Q = U†PU from the frame,
// then born_from_conjugated over eps/free/branches). Bit-identical to born_probabilities.
std::pair<double, double> framed_expectation(const FramedSuperposition& L, const Pauli& P);

// Single-qubit anticommuting (Case B) measurement. Returns outcome m in {+1,-1}; mutates L into
// the post-measurement lean state. `pauli`/`q` = lab single-qubit Pauli (0:X 1:Y 2:Z), `P` its
// standard Pauli form, `A` = supp((U^dag P U).x) (the anticommuters), `u` the Born random in [0,1).
int framed_measure_anticommuting(FramedSuperposition& L, int pauli, int q, const Pauli& P,
                               const std::vector<int>& A, double u);

// General-Pauli (multi-qubit) anticommuting collapse: identical algorithm, but the read is an
// arbitrary standard Pauli `P` with PRECOMPUTED conjugate `Q = U† P U` (and `A = supp(Q.x)`), so it
// does not assume a single-qubit lab read. The single-qubit overload above is exactly this with
// P = i^· X^x Z^z the lab Pauli and Q = conjugate_single(pauli,q). Used by the deferred-error path
// (D† Q_k D reads are multi-qubit). Same outcome m∈{+1,-1}; mutates L into the post-state.
int framed_measure_anticommuting_general(FramedSuperposition& L, const Pauli& P, const Pauli& Q,
                                       const std::vector<int>& A, double u);

// Classify + collapse a single-qubit read on L at the u=0 reference; MUTATES L to the reference
// post-state. Reference convention: out=+1 when pp>1e-12 (u=0), else out=-1. Reuses the existing
// lean collapse machinery (framed_measure_anticommuting_general for Case B; the shared Case-A split
// for a χ-discriminating commuting coin; no-op when deterministic). The lean mirror of the affine
// loop body born_probabilities + is_commuting_single + measure_single(...,0.0).
FramedRefRead framed_reference_read(FramedSuperposition& L, int pauli, int q);

// Does the single-qubit read (pauli 0:X 1:Y 2:Z, qubit q) couple the distinguished `free`
// generators? Column-form symplectic test (⟨read,R⟩ reads only column q of R: Z→R.x[q],
// X→R.z[q], Y→R.x[q]^R.z[q]; couples iff some free Zrow OR Xrow column bit is set). This is the
// predicate batch_measure's Pass-1 uses to pick the next collapse read — exported so the
// TreePlan extractor's forced-collapse order matches Pass-1 BY CONSTRUCTION (one code path).
bool framed_read_couples_free(const FramedSuperposition& L, int pauli, int q);

// chi==1 batch: sample the joint outcome of a COMMUTING single-qubit read set on a stabilizer
// state. `rng` returns U(0,1). Outcomes written to `out` (0:+1, 1:-1). No frame modification.
void batch_chi1(const FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                std::function<double()> rng, std::vector<int>& out);

// General-chi batch: pass-1 reduces chi->1 by collapsing only branch-coupling reads (Case A
// logical-Z split / Case B anticommuting collapse), pass-2 runs batch_chi1 on the branch-
// independent remainder. `rng` returns U(0,1). Outcomes in `out` (0:+1, 1:-1). Takes L BY
// REFERENCE and MUTATES it (the per-shot block is a fresh buffer, so the caller's copy is
// disposable). Callers that need L preserved must pass a copy.
void batch_measure(FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                   std::function<double()> rng, std::vector<int>& out);

}  // namespace qeccore
