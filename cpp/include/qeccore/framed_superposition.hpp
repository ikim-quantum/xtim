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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/amplitudes.hpp"
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/scratch.hpp"

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
// J-b1 P9a (design Lemma R1 gate): the count of anticommuting reads whose pivot row differs from the pattern-only rule
// (the first free row of the anticommuting set, else its first row); the post-barrier frame is a function of the circuit
// iff it stays 0 on the barrier reads.
long framed_pivot_nonpattern_count();

struct FramedSuperposition {
    CliffordTableau U;               // frame: g_a = U Z_a U^dag (Zrow), d_a = U X_a U^dag (Xrow)
    std::vector<uint8_t> eps;        // g_a|psi> = (-1)^eps[a]|psi>
    std::vector<int> free;           // distinguishing generator indices into U
    // Amplitude entry: (sigma over `free` as a packed key — bit d = sigma_d —, coefficient) —
    // DenseAmplitudes' storage element (the SAME type, state-budgeted at sizeof(Entry) per entry).
    using Entry = DenseAmplitudes::Entry;
    using EntryVec = DenseAmplitudes::EntryVec;
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
    // Same result, MOVING the branch sign patterns out of a CanonicalStabSum the caller discards
    // (the bare-state build sites): no per-branch vector copy, O(χ) pointer moves.
    static FramedSuperposition from_css(CanonicalStabSum&& s);
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
    EntryVec& entries() { return dense_->b_; }
    const EntryVec& entries() const { return dense_->b_; }
    AmpKey& key(int i) { return dense_->b_[i].first; }
    AmpKey key(int i) const { return dense_->b_[i].first; }
    uint8_t sigma_bit(int i, int d) const { return key_bit(dense_->b_[i].first, d); }
    std::complex<double>& coeff(int i) { return dense_->b_[i].second; }
    const std::complex<double>& coeff(int i) const { return dense_->b_[i].second; }
    // Keep the container's logical-index count in sync after `free` is edited directly (a free
    // register wider than one packed key word is refused by name).
    void sync_alpha_k() {
        check_amp_key_width((long long)free.size(), "FramedSuperposition");
        if (dense_) dense_->k_ = (int)free.size();
    }

  private:
    void rebind_() { dense_ = dynamic_cast<DenseAmplitudes*>(alpha.get()); }
    DenseAmplitudes* dense_ = nullptr;   // non-owning cache of alpha's dense backend (or null)
    // S2k: a state carries NO key-lookup scratch. Its kernels' callers (expectation, born_p1,
    // measure_pauli) own a per-call AmpIndexLocal on their stack frame (64 inline slots; heap slots
    // beyond are charged by its ticket for the call only), so a state's size, copies and retained
    // memory are unchanged by S2k.
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

struct FactoredBareState;

// ── FramedScratch: the framed kernels' caller-owned working storage (TLC) ──────────────────────
// Every kernel below (and the active-block / TreePlan kernels of factored_active_block.cpp and
// framed_sampler.cpp) keeps its working buffers in ONE part of a FramedScratch its caller owns and
// passes down — never in thread_local storage. One part per function: a kernel touches only its own
// part and hands the whole scratch to the kernels it calls (fmag → dscan, canon; bm → split(canon),
// fmag, chi1 → conj → dscan, chi1pre; reference read → born, canon, fmag; ablock → ainto | aset, diag;
// TreePlan build → bm, born, spc, dscan; cor → fmag), so no part is ever used re-entrantly (Debug
// builds assert it). Every buffer is (re)initialised by its kernel before it is read, so results do
// not depend on the scratch's history; the two parts that carry state between calls keep their own
// invariant (ablock's g2a: all −1 between calls; ablock's (&f, uid) axis cache).
//   Budget: the χ-scaled parts (canon, born, fmag, dscan) charge their high-water size BEFORE growing
//   (same estimates and names as the former thread scratch), released with the scratch / release().
//   Lifetime: owners hold it for one compile / run / call (the exact engine releases its scratch at the
//   end of every program entry, sampler.cpp), so at most one is grown per thread at a time. Copies
//   never carry scratch: copying builds an empty one and assignment keeps the target's own.
//   Not thread-safe: an owner (an exact sampler) is used by one thread at a time; nothing may keep a
//   pointer into a FramedScratch across a release().
struct FramedSurvivor { std::complex<double> coeff; int rep; };
struct FramedScratch {
    struct Canon {
        std::vector<std::vector<uint64_t>> col, red;
        std::vector<uint64_t> resid;
        std::vector<int> pivot_cols;
        std::vector<std::vector<uint8_t>> comp;
        std::vector<uint8_t> expand, is_pivot;
        ScratchCharge charge;
        bool busy = false;
    };
    struct Born {
        std::vector<uint64_t> synd_pk, eps_pk, tgt;
        std::unordered_map<std::string, int> sig_idx;
        ScratchCharge charge;
        bool busy = false;
    };
    struct Fmag {
        std::vector<uint64_t> synd_pk, post_pk, fsynd, fsyndT;
        std::vector<std::complex<double>> c0;
        std::vector<Pauli> oW, Wgs, Qfs;
        std::vector<const Pauli*> Wtg;
        std::vector<int> order;
        std::vector<FramedSurvivor> surv;
        std::vector<std::vector<uint64_t>> redcol;
        std::unordered_map<std::string, int> post_idx;
        DenseAmplitudes::EntryVec entries;   // swapped with the state's list (state-budgeted storage)
        std::vector<int> free_pos, basis, nfree;
        std::vector<uint64_t> eps_pk, amask, onesg, ecol, resid, varies, epsn_pk;
        std::vector<std::vector<uint8_t>> comp;
        std::vector<uint8_t> expand, isnf;
        ScratchCharge charge;
        bool busy = false;
    };
    struct Conj { std::vector<Pauli> Tg; std::vector<const Pauli*> Tp; bool busy = false; };
    struct Chi1Pre { std::vector<uint64_t> eps_pk; std::vector<MeasCoin> coins; bool busy = false; };
    struct Chi1 { std::vector<Pauli> Qpre; bool busy = false; };
    struct Bm {
        std::vector<int> A, remidx, ro;
        std::vector<std::pair<int, int>> rem;
        bool busy = false;
    };
    struct Diag {
        std::vector<uint64_t> vmask;
        std::vector<std::pair<int, uint8_t>> acols;
        std::vector<std::vector<uint64_t>> adj;
        bool busy = false;
    };
    struct ASet { std::vector<int> axisA; std::vector<uint8_t> active; bool busy = false; };
    struct AInto { std::vector<int> added; bool busy = false; };
    struct ABlock {
        std::vector<uint8_t> is_b, signA, pull_ax, pull_sg, a_col, v_col;
        std::vector<int> act_sorted, axisA, act, pulled, g2a;
        std::vector<std::pair<int, int>> cz_col;
        const FactoredBareState* cached_f = nullptr;   // axis cache key, with cached_uid
        uint64_t cached_uid = 0;
        bool busy = false;
    };
    struct Spc {
        std::vector<uint8_t> zsgn;
        std::vector<std::pair<uint64_t, std::complex<double>>> esA, esB;
        std::vector<int8_t> freeof;
        bool busy = false;
    };
    struct Words { std::vector<uint64_t> v; bool busy = false; };   // replay `avec`, alt-side `zmask`
    struct Ints { std::vector<int> v; bool busy = false; };          // condition_on_recorded `A`
    struct Sizes { std::vector<int64_t> v; bool busy = false; };     // expand_corrected_tree `sz`
    struct Parts {
        Canon canon; Born born; Fmag fmag; Conj conj; Chi1Pre chi1pre; Chi1 chi1; Bm bm; Diag diag;
        DualScanScratch dscan; ASet aset; AInto ainto; ABlock ablock; Spc spc;
        Words replay, alt; Ints cor; Sizes expand;
    };
    Parts p;

    FramedScratch() = default;
    FramedScratch(const FramedScratch&) {}
    FramedScratch(FramedScratch&&) noexcept {}
    FramedScratch& operator=(const FramedScratch&) { return *this; }
    FramedScratch& operator=(FramedScratch&&) noexcept { return *this; }
    // Drop every buffer and release every charge (an owner's compile / run is over).
    void release() { p = Parts(); }
    int64_t charged_bytes() const {
        return p.canon.charge.bytes() + p.born.charge.bytes() + p.fmag.charge.bytes() + p.dscan.charge.bytes();
    }
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
                                const std::vector<uint8_t>& v, FramedScratch& fs);

// Re-select `free` as a minimal independent basis of the surviving sign variation; fold constant
// columns into eps (no anchor). Mutates L in place. Allocation-free on a warm scratch.
void framed_canonicalise(FramedSuperposition& L, FramedScratch& fs);
void framed_canonicalise(FramedSuperposition& L);   // cold callers: a scratch for this call

// Pauli expectation on a lean state: returns (p+, p-) with <P> = p+ - p-. This is
// CanonicalStabSum::born_probabilities re-expressed on the lean rep (Q = U†PU from the frame,
// then born_from_conjugated over eps/free/branches). Bit-identical to born_probabilities.
std::pair<double, double> framed_expectation(const FramedSuperposition& L, const Pauli& P, FramedScratch& fs);
// The same for a Pauli ALREADY in frame coordinates (Q = U^dag P U): framed_expectation(L, P) ==
// framed_expectation_frame(L, U^dag P U) bit for bit. Lets a caller conjugate generators once and
// form products in frame coordinates (conjugation is an exact group homomorphism on Paulis).
std::pair<double, double> framed_expectation_frame(const FramedSuperposition& L, const Pauli& Q, FramedScratch& fs);
// Cold callers (DEM export, compile, twirl): the same with a scratch for this call only (its
// buffers are allocated per call). The exact engine passes its own scratch.
std::pair<double, double> framed_expectation(const FramedSuperposition& L, const Pauli& P);
std::pair<double, double> framed_expectation_frame(const FramedSuperposition& L, const Pauli& Q);

// Single-qubit anticommuting (Case B) measurement. Returns outcome m in {+1,-1}; mutates L into
// the post-measurement lean state. `pauli`/`q` = lab single-qubit Pauli (0:X 1:Y 2:Z), `P` its
// standard Pauli form, `A` = supp((U^dag P U).x) (the anticommuters), `u` the Born random in [0,1).
int framed_measure_anticommuting(FramedSuperposition& L, int pauli, int q, const Pauli& P,
                               const std::vector<int>& A, double u, FramedScratch& fs);

// General-Pauli (multi-qubit) anticommuting collapse: identical algorithm, but the read is an
// arbitrary standard Pauli `P` with PRECOMPUTED conjugate `Q = U† P U` (and `A = supp(Q.x)`), so it
// does not assume a single-qubit lab read. The single-qubit overload above is exactly this with
// P = i^· X^x Z^z the lab Pauli and Q = conjugate_single(pauli,q). Used by the deferred-error path
// (D† Q_k D reads are multi-qubit). Same outcome m∈{+1,-1}; mutates L into the post-state.
int framed_measure_anticommuting_general(FramedSuperposition& L, const Pauli& P, const Pauli& Q,
                                       const std::vector<int>& A, double u, FramedScratch& fs);

// Classify + collapse a single-qubit read on L at the u=0 reference; MUTATES L to the reference
// post-state. Reference convention: out=+1 when pp>1e-12 (u=0), else out=-1. Reuses the existing
// lean collapse machinery (framed_measure_anticommuting_general for Case B; the shared Case-A split
// for a χ-discriminating commuting coin; no-op when deterministic). The lean mirror of the affine
// loop body born_probabilities + is_commuting_single + measure_single(...,0.0).
FramedRefRead framed_reference_read(FramedSuperposition& L, int pauli, int q, FramedScratch& fs);

// As framed_reference_read, but the coin is biased toward outcome bit `want` (0:+1, 1:-1): out is
// `want` whenever that outcome has probability > 1e-12, else the other. want=0 IS
// framed_reference_read (same choices, same collapse calls — it delegates here).
FramedRefRead framed_reference_read_toward(FramedSuperposition& L, int pauli, int q, int want,
                                           FramedScratch& fs);

// THE noiseless reference sample every engine reports detector/observable channels relative to —
// Stim's definition (Circuit.reference_sample(): noise off, every collapse biased toward its +1
// eigenvalue, taken in circuit = record order), generalised to the framed (magic) state. The
// reads are collapsed SEQUENTIALLY in the given (record) order; read k takes raw outcome bit
// want(k, out) (`out` holds reads 0..k-1) whenever it has positive probability, else the other.
// The caller's `want` maps "user record bias +1" into this state's read frame (an
// eliminate_hadamards sign stamp, a feedback relabel). While chi > 1 every read goes through
// framed_reference_read_toward (mutating L); once chi == 1 the rest are one batched stabilizer
// pass with the same per-read bias (exact: batch_chi1 IS a sequential stabilizer simulation).
// Outcomes in `out` (0:+1, 1:-1). MUTATES L; pass a copy.
void framed_reference_records(FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                              const std::function<int(int, const std::vector<int>&)>& want,
                              std::vector<int>& out);

// Does the single-qubit read (pauli 0:X 1:Y 2:Z, qubit q) couple the distinguished `free`
// generators? Column-form symplectic test (⟨read,R⟩ reads only column q of R: Z→R.x[q],
// X→R.z[q], Y→R.x[q]^R.z[q]; couples iff some free Zrow OR Xrow column bit is set). This is the
// predicate batch_measure's Pass-1 uses to pick the next collapse read — exported so the
// TreePlan extractor's forced-collapse order matches Pass-1 BY CONSTRUCTION (one code path).
bool framed_read_couples_free(const FramedSuperposition& L, int pauli, int q);

// chi==1 batch: sample the joint outcome of a COMMUTING single-qubit read set on a stabilizer
// state. `rng` returns U(0,1). Outcomes written to `out` (0:+1, 1:-1). No frame modification.
void batch_chi1(const FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                std::function<double()> rng, std::vector<int>& out, FramedScratch& fs);

// General-chi batch: pass-1 reduces chi->1 by collapsing only branch-coupling reads (Case A
// logical-Z split / Case B anticommuting collapse), pass-2 runs batch_chi1 on the branch-
// independent remainder. `rng` returns U(0,1). Outcomes in `out` (0:+1, 1:-1). Takes L BY
// REFERENCE and MUTATES it (the per-shot block is a fresh buffer, so the caller's copy is
// disposable). Callers that need L preserved must pass a copy.
void batch_measure(FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                   std::function<double()> rng, std::vector<int>& out, FramedScratch& fs);

namespace framed_detail {
// First b < n with Q.xbit(b) and b ∉ free (−1 if none); y = Q's X bits on the free set (bit d ⇔
// Q.xbit(free[d])). Q is unchanged on return (the free bits are cleared for the scan and restored).
// The pivot search of expectation / born_p1 / measure_pauli (exported for its oracle test).
int first_x_off_free(Pauli& Q, int n, const std::vector<int>& free, AmpKey y);
}  // namespace framed_detail

// Free this thread's remaining thread scratch slots (the CanonicalStabSum oracle's; the framed
// kernels' scratch is caller-owned since TLC).  Diagnostics/tests: lets live_state_bytes() return to
// baseline.
void release_thread_state_scratch();

}  // namespace qeccore
