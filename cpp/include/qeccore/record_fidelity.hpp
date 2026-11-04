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
// structure is compiled once per content in two stages (fid-orth): the constraint system's reduction on the content's
// first row (per row then only N^T b — orthogonal rows score 0 with nothing built or stored — and R b), the value
// structure on its first consistent row; the memo keys on the solved shot data of consistent rows (the correction
// enters them). The impossible-record checks and the content-level caps RF_SECTOR_RANK / RF_LOGICAL_RANK run only on
// consistent rows (a content that only ever sees orthogonal rows scores 0 exactly and is never refused by them).
// explaw T4(b) (owner-approved 2026-10-30): ksec = the rank of the C-syndrome map on the constraint kernel — the
// independent REST stabilizers, commuting with every record, whose sign the residual's expansion terms flip (a rest
// coin coupled to the output by a CZ residual; a coin on output generators alone has sigma = 0 and sits in the fiber).
// The sector sum grows as 2^ksec while the pure output |phi> has at most 4^a nonzero logical Pauli classes: with
// |phi><phi| = Pi_{S_O} 2^{-a} sum_c <phi|Pbar_c|phi> Pbar_c, F = 2^{-a} sum_{c nonzero} <phi|Pbar_c|phi> ·
// <psi|Pi_{S_O} Pbar_c|psi> / <psi|psi>, every term route 3a's projector expectation with ONE extra dressed Pauli
// C0† Pbar_c C0 (a Pauli stays a Pauli under the Clifford C0: no expansion, no sectors). Per content the cheaper route
// by a deterministic operation count of (a, k, ksec, |supp b|) decided before either route runs (ties: the sector sum;
// above ksec = 20 the class route is the only one; the classes need 2a <= 20 and are computed only for the class route);
// otherwise the sector sum (byte-identical to before; ksec = 0 on the whole corpus). RF_SECTOR_RANK applies to the
// sector route only, checked at the content build as before (a content over it refuses before RF_LOGICAL_RANK).
// Exact or refuse by name (RF_SECTOR_RANK, RF_LOGICAL_RANK, RF_FIBER_RANK, RF_CONTENT_MALFORMED, RF_FACTOR_NOT_CLIFFORD).

#include <array>
#include <complex>
#include <cstdint>
#include <exception>
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
    long cache_flushes = 0;          // plancache-design §5 F2: boundary passes that evicted, plus mid-call drops to
                                     //   held_rest (a would-refuse row's pass 2)
    long boundary_evictions = 0;     // §5 F2: contents evicted by the call-boundary pass (ranked coldest first)
    long midcall_evictions = 0;      // §5 N4: contents dropped by a would-refuse row's passes (max_bytes)
    // fid-orth (OD-2026-10-24-fid-orth): per content the constraint system is row-reduced once (stage 1); a row whose
    // right-hand side is outside its column space is orthogonal (F = 0: no value structure, no memo entry)
    long orthogonal_rows = 0;        // rows scored 0 by the consistency test alone (neither a memo hit nor a miss)
    long projection_skips = 0;       // rows whose right-hand side equalled the content's previous one (projection reused)
    long value_builds = 0;           // contents whose value structure (stage 2) was built (first consistent row)
    // explaw T4(b): the class route
    long max_ksec = 0;               // largest sector rank ksec of a content
    long class_contents = 0;         // contents routed to the class route (the cost rule, or ksec > 20)
    long class_rows = 0;             // misses evaluated by the class route
    long classes_nonzero = -1;       // nonzero logical classes of the output (−1: not computed — computed only when a
                                     //   content takes the class route)
    std::vector<long> r_hist, kappa_prime_hist, dker_hist;   // compiled contents by r, κ', |dressed kernel|
};

// plancache-design §5 (N4): the byte budget's refusal (max_bytes) — a named std::runtime_error the row transaction
// catches: pass 1 drops the memo and every content this call's rows have not read, pass 2 drops to held_rest (today's
// flush); the row is refused only when its own need, over the register and the rest structure, exceeds max_bytes.
struct RfBudgetRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
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
    // plancache-design §5 F2 (the call boundary, between evaluations only): the dead slots of the last call's
    // mid-call passes are recycled (their memo entries erased; ids are never renumbered); then, when the memo +
    // contents − the register's rest structure exceed `budget` (0: no bound), the conditioned-norm cache is dropped
    // and contents are evicted coldest first (rows served in the last two calls, ties by slot id), each with its memo
    // entries, until held <= budget − reserve − budget/8 (reserve: the previous call's new-content bytes, scaled to
    // `rows_next`, the coming call's rows, when given; saturating: budget 1 evicts everything = today's flush, which
    // also drops the memo). Every evicted value is a pure function of its key: the output is unchanged. Returns
    // whether it dropped anything (the memo, the norms or contents) — what cache_flushes counts (review N3).
    // threads (fid-routine B2): a whole-cache drop frees on up to that many pool jobs (deallocation only).
    bool trim_cache(size_t budget, long rows_next = -1, int threads = 1);
    // The largest heap reserve one pool lane takes (the B1 reserves are capped by it): one glibc arena heap, 64 MiB (a
    // larger reserve is trimmed straight back). Fixed per lane, independent of the cache budget (transient headroom).
    static constexpr size_t kLaneReserveCap = size_t(64) << 20;
    // §5 F2: the call's rows per content (the retention ranking's input; the counts are the row list's, identical at
    // every thread count), rolled at the end of every family_record_fidelity call.
    void note_rows(const int* content, long rows);
    bool has_byte_budget() const;                   // max_bytes != 0 (the row transaction, serial)
    // review L2 (test lever): {Σ live contents' charged bytes + the shared charges, held_l3} — equal by construction.
    std::pair<size_t, size_t> audit_bytes_for_test() const;
    // speed fix 4 (test lever): {Σ live contents' memo bytes, held_memo, stray memo entries} — equal, equal, 0.
    std::array<size_t, 3> audit_memo_for_test() const;
    // scaling S2: the held bytes per live content so far (the content builds' heap reserve), in [1 KiB, 64 KiB]
    size_t content_bytes_estimate() const;
    bool live(int id) const;                        // a content id that is not a tombstone
    // §5 N4: the row transaction's passes (max_bytes only). drop_content: the row's own staging (tombstoned; a later
    // row re-interns into a fresh id); drop_unreferenced: the whole memo and every content whose id is not in `keep`;
    // drop_to_rest: today's flush (clear(), counted as a cache flush). Each returns the contents dropped.
    long drop_content(int id);
    long drop_unreferenced(const std::vector<uint8_t>& keep);
    long drop_to_rest();
    void note_serial(int requested, const std::string& why);   // the thread report of a call run serially outside
    size_t bytes_held() const;
    size_t memo_bytes() const;
    // S8 test levers: the per-row zero class log (memo bypassed while on): 0 INFEASIBLE (the GF(2) constraint solve
    // fails), 1 BORN-ZERO (every sector's conditioned norm is zero, ref_ny), 2 GAUSS-ZERO (the exact Gauss sum vanishes
    // on every surviving sector), 3 VALUE (route 3a rows: 3), 255 a fiber beyond the enumeration oracle (F = NaN);
    // λ(u, v) = ⟨b|Z^v X^u|b⟩ through the memo (direct = false) or the formula; the dense table's byte cap (before the
    // first λ); the memo kind (0 none yet, 1 dense, 2 sparse); the fibers by exact Gaussian-integer enumeration (the
    // test-only oracle; fibers of dimension <= 12, else the row is logged 255; memo bypassed while on).
    void set_row_log_for_test(bool on);
    // fid-routine fix round 1 (test-only): the next threaded evaluate throws std::bad_alloc once at point 1 (between
    // its two waves: the call's failure, after the merge) or point 2 (the memo reserve: capacity only, not a failure);
    // fix round 2: point 3 = in the merge after lane 0 (the call fails with lanes 1.. unmerged), point 4 = at a merged
    // memo entry's note (the call fails; the entry is erased with its charge)
    void set_fault_for_test(int point);
    std::vector<uint8_t> take_row_log_for_test();
    std::complex<double> lambda_for_test(uint64_t u, uint64_t v, bool direct);
    void set_lambda_budget_for_test(size_t bytes);
    int lambda_mode_for_test() const;
    void set_hform_oracle_for_test(bool on);
    // explaw T4(b) test lever (review F4): 0 the cost rule (default), 1 the sector route always (its 2^20 boundary stays
    // testable), 2 the class route whenever its classes are computable. Shared by every lane; set it between calls.
    void set_class_route_for_test(int mode);
    const RecordFidelityStats& stats() const;
    int n() const;
    // explaw T4(b) mechanism probe (test API): per interned content its route-3b structure as built so far.
    struct ContentStructure {
        int id = 0, nM = 0, nG = 0, kprime = 0, ksec = 0, dC = 0;
        int fib_d = 0, fib_kap = 0;          // the sector fiber (σ = 0): stabilizer / logical directions (the sector route's per-sector work)
        bool full = false, class_route = false;
    };
    std::vector<ContentStructure> content_structures() const;

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
    // factors' direct product reduced to its support (exact). Interned once per `key`. Route 3a computes its pullbacks
    // here by `pull`; route 3b never calls `pull`: its record pull-backs come from `factors` themselves (C0 = F_V ⋯ F_1,
    // conjugation by F_V first), on the content's first consistent row, so `pull` must equal C0† · C0 exactly.
    using Factor = std::vector<std::pair<std::complex<double>, Pauli>>;
    int intern_content(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                       const std::vector<Factor>& factors);
    bool has_content(const std::string& key, int* id) const;
    // A kind-0 content C0 = S^{a&1}·CZ by its normal form (the closed-form expansion; the dressed images by its exact
    // Lemma 1); refused by name: RF_CONTENT_MALFORMED (a CZ pair out of range or on the diagonal).
    int intern_content_diag(const std::string& key, const DiagNormalForm& nf);
    // scaling F1: interning in two halves. prepare_content(_diag) builds a content's structure from the evaluator's
    // construction state alone — prepares of distinct contents may run concurrently (after prepare_begin, serially) —
    // and keeps the intern's refusal, if any; install_prepared then does what intern_content(_diag) does next with
    // the key (its id, the lazy record images, the charges, the refusal at the same step), serially. Installing in
    // first-row order gives the ids and every byte count of interning in that order. prepare_parallel_ok(): no byte
    // budget and no test lever (their refusal / logging order is the serial loop's).
    struct PreparedContent;
    void prepare_begin();
    bool prepare_parallel_ok() const;
    std::shared_ptr<PreparedContent> prepare_content_diag(const DiagNormalForm& nf) const;
    std::shared_ptr<PreparedContent> prepare_content(const std::function<Pauli(const Pauli&)>& pull,
                                                     const std::vector<Factor>& factors) const;
    int install_prepared(std::string&& key, size_t key_hash, PreparedContent& p);
    // fid-routine B5: room for n installs (the contents' slots and index) before an install loop — capacity only.
    void reserve_installs(size_t n);
    // scaling F2: a content key's hash (computed once, where the key is built) and the lookup that carries it
    // (-1: not interned); has_content = find_content(key, content_key_hash(key)).
    static size_t content_key_hash(const std::string& key);
    int find_content(const std::string& key, size_t key_hash) const;
    std::shared_ptr<PreparedContent> prepared_refused(std::exception_ptr e) const;   // refused before any effect
    // One layout-3 record: content id, P_rec (px, pz: ceil(n/64) words each), y (the raw record values,
    // ceil(m/64) words, storage column order), the correction P_c = X^qx Z^qz on O (deferred-wire words; nullptr =
    // none; refused by name if it acts off O).
    double evaluate_record(int content, const uint64_t* px, const uint64_t* pz, const uint64_t* y,
                           const uint64_t* qx = nullptr, const uint64_t* qz = nullptr);
    // threads T6: rows [0, rows) of layout-3 records — content ids (interned), P_rec rows (x then z, PNW words each), y
    // rows (YW words), the optional correction (uint8 rows × 2 × |out_wires|, X row then Z row, on the deferred wires
    // out_wires) — evaluated as evaluate_record per row in row order would, on up to `threads` threads: each content's
    // rows on one lane, in row order (a content's structure and its memo entries are built by that lane only); the
    // lanes read the evaluator's memo / caches and fill their own, merged in lane order afterwards. The earliest row's
    // refusal is thrown. Serial (threads_effective() 1, the reason in threads_limit()) under a byte budget (max_bytes:
    // its refusal is decided in row order), a test lever, or a single content.
    void evaluate_records(long rows, const int* content, const uint64_t* prec, int PNW, const uint64_t* y, int YW,
                          const uint8_t* corr, const std::vector<int>* out_wires, double* out, int threads);
    int threads_requested() const;
    int threads_effective() const;
    const std::string& threads_limit() const;
    bool same_record_model(const FramedSuperposition& bare, const std::vector<Pauli>& record_ops) const;
    int route() const;            // 1 = 3a (a = 0), 2 = 3b (a > 0)

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qeccore
