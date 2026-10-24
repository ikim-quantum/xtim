#pragma once
// record_fidelity — exact per-record REGISTER fidelity of a retained twirl output state (RF-ref, owner rulings
// 2026-10-22: the STRICT output contract; rfref/register-fidelity-investigation.md + -review.md, the review wins).
//
// Per shot the retained state is ψ = Π_rec(y)·P·C0|B⟩ (|B⟩ the bare state on the n deferred wires, P the record's
// prefix Pauli, C0 its content's Clifford, Π_rec(y) the projection onto every record operator at its raw value y).
// O = the OUTPUT_QUBITS wires, R = every other wire (ancillas and measurement read wires). With a decoder's Pauli
// correction P_c on O, the score is the register fidelity
//     F = ⟨φ| P_c Tr_R(ψψ†/⟨ψ|ψ⟩) P_c |φ⟩,   |B⟩ = |φ⟩_O ⊗ |χ⟩_R.
// STRICT contract (checked at construction, refused by name otherwise): the noiseless output is pure and fixed,
// independent of every outcome ⟺ (records live on R, asserted) the bare state is a product across O|R ⟺ Theorem 1
// on the unsigned bare stabilizer group S0 (no records): s_O = |O| or s_R = |R| (s_X = dim S0 ∩ P_X).
// Routes (review M3): a = 0 (s_O = |O|, φ a stabilizer state) → 3a, ONE rotation-free projector expectation
//     F·N = ⟨B|C0†P'' Π_rec(y) Π_{S*} P''C0|B⟩,  S* = S0 ∩ P_O signed on |B⟩, P'' = P_c·P (the correction a sign);
// a > 0 (s_R = |R|, χ a stabilizer state) → 3b, the rest-sector sum (S8, rfref/fable-decoded-design.md)
//     F·N = Σ_σ |A_σ|² / N_{y''},  A_σ = ⟨B| Π_{R_n}(y'') D_σ P'' C0 |B⟩,  y'' = y ⊕ ⟨D_σ, M⟩,
// D_σ the rest destabilizers of the sector, by the twirled Pauli expansion: C0 = c Σ_t i^{Q(t)} G_0 Π G_j^{t_j} (kind 0:
// the closed form of S^a CZ; a factor list: its direct expansion reduced to the support), Π_{R_n} over the independent
// Born records; a term survives only if its frame image has no X on a fixed row (a GF(2) solve per shot: INFEASIBLE
// rows are exactly 0); the surviving terms of a sector form a fiber whose stabilizer directions are summed EXACTLY
// (a Z4 Gauss sum) and whose logical directions are λ(u, v) = ⟨b|Z^v X^u|b⟩ lookups in a canonical order. The
// structure is compiled once per content; the memo keys on the solved shot data (the correction enters them). Exact
// or refuse by name (RF_SECTOR_RANK, RF_LOGICAL_RANK, RF_FIBER_RANK, RF_CONTENT_MALFORMED, RF_FACTOR_NOT_CLIFFORD).

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

struct DiagNormalForm;

struct RecordFidelityStats {
    long shots = 0;            // evaluate() calls
    long memo_hits = 0;        // served from the (plan, sign-key) memo
    long misses = 0;           // computed
    long plans = 0;            // distinct plan contents seen
    long max_dim_bits = 0;     // the bare's resident register, k bits (no other dense register exists)
    long rn_words = 0;               // distinct recorded R_n words y_{R_n} whose N_y = <B|Π_{R_n}(y)|B> was computed
    // register fidelity (strict contract)
    int route = 0;                   // 1 = 3a, 2 = 3b
    long out_a = 0;                  // a: the logical qubits wholly on the output
    // 3b (the twirled Pauli-expansion route)
    long kernel_compiles = 0;        // per-content expansion structures compiled
    long rn_misses = 0;              // misses of contents with independent Born records (R_n ≠ ∅)
    long rotation_misses = 0;        // misses of contents with a non-identity C0 (r > 0)
    long sectors = 0;                // rest sectors evaluated, summed over misses
    long max_sectors = 0;            // largest per-miss sector count 2^κ
    long sectors_nonzero = 0;        // sector terms with a nonzero (VALUE) Gauss sum
    long max_r = 0;                  // largest expansion rank r of a content (C0's expansion variables)
    long max_kappa_prime = 0;        // largest constraint-kernel dimension κ' of a content (sector + fiber directions)
    long cache_flushes = 0;          // termroute-a: trim_cache calls that dropped the held bytes
    std::vector<long> r_hist, kappa_prime_hist, dker_hist;   // compiled contents by r, κ', |dressed kernel|
};

class RecordFidelity {
  public:
    ~RecordFidelity();
    RecordFidelity(const RecordFidelity&) = delete;
    RecordFidelity& operator=(const RecordFidelity&) = delete;

    // The memo, the layout-3 contents (with their expansion structures) and the λ memo are pure speed devices
    // (values are pure functions of the key). clear() drops all of them; clear_memo() drops the memo alone (contents
    // persist across evaluate() calls). bytes_held() reports everything the evaluator holds; max_bytes budgets it
    // (memo + contents + structures + frame-image caches + λ; refused by name).
    void clear();
    void clear_memo();
    // termroute-a cache bound: when the memo + contents + caches hold more than `budget` bytes (0: no bound), drop
    // them (clear(): values are pure functions of their keys, so the output is unchanged). Call between evaluations
    // only (content ids are not stable across it). Returns whether it dropped.
    bool trim_cache(size_t budget);
    size_t bytes_held() const;
    size_t memo_bytes() const;
    // S8 test levers: the per-row zero class log (memo bypassed while on): 0 INFEASIBLE (the GF(2) constraint solve
    // fails), 1 BORN-ZERO (every sector's conditioned norm is zero, ref_ny), 2 GAUSS-ZERO (the exact Gauss sum vanishes
    // on every surviving sector), 3 VALUE (route 3a rows: 3), 255 a fiber beyond the enumeration oracle (F = NaN);
    // λ(u, v) = ⟨b|Z^v X^u|b⟩ through the memo (direct = false) or the formula; the dense table's byte cap (before the
    // first λ); the memo kind (0 none yet, 1 dense, 2 sparse); the fibers by exact Gaussian-integer enumeration (the
    // test-only oracle; fibers of dimension <= 12, else the row is logged 255; memo bypassed while on).
    void set_row_log_for_test(bool on);
    std::vector<uint8_t> take_row_log_for_test();
    std::complex<double> lambda_for_test(uint64_t u, uint64_t v, bool direct);
    void set_lambda_budget_for_test(size_t bytes);
    int lambda_mode_for_test() const;
    void set_hform_oracle_for_test(bool on);
    const RecordFidelityStats& stats() const;
    int n() const;

    // ── layout-3 records (J-b1 P7). `record_ops`: the sampler's record operators in storage column order (exact
    // phase). `out_wires`: the OUTPUT_QUBITS deferred wires (O). `read_meas`: per deferred wire, the measurement whose
    // read wire it is (−1 for none; may be empty) — the refusal diagnostic names measurements by it. Refused by name
    // (std::invalid_argument): RF_NO_OUTPUT (O empty), RF_RECORD_ON_OUTPUT (a record operator acts on O),
    // RF_OUTPUT_NOT_STRICT (Theorem 1 fails on S0: the diagnostic gives s_O, s_R, (a, b, c, e), the cross-cut stabilizer
    // directions on O and the measurements whose read wires carry their rest parts). `wire_user`: per deferred wire,
    // the user's qubit it carries (−1 for an internal wire; may be empty): the diagnostic names qubits by it.
    RecordFidelity(const FramedSuperposition& bare, const std::vector<Pauli>& record_ops,
                   const std::vector<int>& out_wires, const std::vector<int>& read_meas, size_t max_bytes,
                   const std::vector<int>& wire_user = {});
    // A content's Clifford C0: `pull(Q)` = C0† Q C0 (exact phase); `factors` = C0 as a product of factors, each
    // Σ_t c_t Q_t, applied in order: S / rot(B) = {c0 I, c1 Q} with c1/c0 = ±i, CZ = {I, A, B, −AB}/2 — any other
    // factor is refused by name, RF_FACTOR_NOT_CLIFFORD; a global phase is immaterial. Its Pauli expansion is the
    // factors' direct product reduced to its support (exact). Interned once per `key`; the pullbacks are computed here.
    using Factor = std::vector<std::pair<std::complex<double>, Pauli>>;
    int intern_content(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                       const std::vector<Factor>& factors);
    bool has_content(const std::string& key, int* id) const;
    // A kind-0 content C0 = S^{a&1}·CZ by its normal form (the closed-form expansion; the dressed images by its exact
    // Lemma 1); refused by name: RF_CONTENT_MALFORMED (a CZ pair out of range or on the diagonal).
    int intern_content_diag(const std::string& key, const DiagNormalForm& nf);
    // One layout-3 record: content id, P_rec (px, pz: ceil(n/64) words each), y (the raw record values,
    // ceil(m/64) words, storage column order), the correction P_c = X^qx Z^qz on O (deferred-wire words; nullptr =
    // none; refused by name if it acts off O).
    double evaluate_record(int content, const uint64_t* px, const uint64_t* pz, const uint64_t* y,
                           const uint64_t* qx = nullptr, const uint64_t* qz = nullptr);
    bool same_record_model(const FramedSuperposition& bare, const std::vector<Pauli>& record_ops) const;
    int route() const;            // 1 = 3a (a = 0), 2 = 3b (a > 0)

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qeccore
