#pragma once
// CertifiedGroupPlanes — columnar bit-planes + RREF pivot table for a certified
// (real, Hermitian ±1) stabiliser group.
//
// The substrate for the diagonal-twirl kernel (Phase B, Task 5). Built ONCE from a
// vector of certified stabiliser Paulis (e.g. FramedSuperposition::certified_stabilizers(),
// exact-phase +1 stabilisers). Two views over the same group:
//
//   1. Columnar bit-planes: for each qubit q, an n_gens-bit column packing bit i = gens[i].xbit(q)
//      (resp. zbit). Tasks 7–10 XOR these columns to build sign planes without touching the row-major
//      Paulis. S2-T5: stored as a CSR index (wire -> generator indices), plus the dense planes when small against it
//      (the kDenseColsRatio rule below).
//   2. A GF(2) reduced-row-echelon of the generators' (x|z) symplectic rows, with exact
//      phase tracked through every row combination (real ±1 group ⇒ signs are meaningful).
//      reduce(P) classifies an arbitrary probe Pauli against the group, pivot-locally and
//      allocation-light (the one copy of P becomes the result rep; no scratch).
//
// reduce() verdicts:
//   IN_GROUP — P's support (x|z) lies in the group; `sign` is the ±1 of THE group element
//              G with that support (P = i^k · G). rep = residual (identity support, exact phase).
//   LOGICAL  — P commutes with the whole group but its support is NOT in it; rep = the reduced
//              residual w = P·G with exact phase (a normaliser/logical representative).
//   ANTI     — P anticommutes with at least one generator (not in the normaliser).

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "qeccore/pauli.hpp"
#include "qeccore/state_budget.hpp"

namespace qeccore {

// T6: the scan switch's cost ratio (pure performance, either side gives identical results): one row-list entry of
// CertifiedGroupPlanes::build (written, read back, re-checked) costs about as much as this many visits of the dense
// scans (a row's pivot-bit test). Measured on the corpus' dense-fill groups (sweeps 4…64 in the T6 report).
inline constexpr long kListEntryCost = 8;
// S2-T5: the planes' storage switches (pure performance, either side gives identical results), as ratios of the dense
// form's bytes to the sparse entries' (4 bytes per set bit; nnz = the generators' total weight). The RREF's working rows
// are dense words iff 16·n_gens·ceil(n/64) ≤ kDenseRowsRatio·4·nnz (transient: at most that multiple of the sparse
// group); the dense column planes are kept iff 16·n·words ≤ kDenseColsRatio·4·nnz (resident). Measured on the corpus
// and the surface / chain families (S2-T5 report): dense rows build the fill-heavy magic-circuit groups (ratios ≤ 20)
// ~2× faster, sparse rows win from ratio ~40–65 (surface d = 9…, chains earlier); the column readers break even at ~9.
// S2-FINALFIX F1: the rows' rule is applied again before each column of the elimination to the sparse rows' CURRENT
// weight (fill-in), switching them to packed words for the rest of it (random parity checks: 16 s -> 1.0 s at n' = 12000;
// surface, chains: no fill-in, never switched).
inline constexpr long kDenseRowsRatio = 32;
inline constexpr long kDenseColsRatio = 8;
// S2-T6 fix 4: the forward-form masks' bounds against the pre-S2-T6 prref route (CertifiedGroupPlanes, "THE BOUND").
// A forward unit is a step read, a row taken, a bitmap word scanned or a mask entry emitted (~2-4 ns); a column flip
// of the reduction (a byte toggle, a list push, a pivot lookup: ~2.5x that) counts kFwdFlipUnits. kFwdUnitWords prices
// a unit in the prref route's unit, one dense word op of its row XORs (~0.2-0.9 ns): the measured ratio is 3.3-11.6
// over 28 families, highest where prref's rows are cache-resident and its words cheapest (the S2-T6 fix-4 report's
// calibration table); the price sits above every measured ratio (S2-CLEAN, fix-4 review n1: 14, was 12, against the
// highest ratio found since, 11.8 on ghzl200 n=700; at 14 every reviewer family stays <= 1.07x eb55e1e's compile time
// and <= 1.02x its RSS, and the forward wins are kept; 16 lost qfx32's). kMaskSpendPct: the
// forward queries may spend at most this percent of the prref route's estimated cost beyond what prref would have
// spent on them. kFwdBytesPct: the forward form may hold at most this percent of prref's heap bytes.
inline constexpr long kFwdFlipUnits = 3;
inline constexpr long kFwdUnitWords = 14;
inline constexpr long kMaskSpendPct = 15;
inline constexpr long kFwdBytesPct = 50;

struct ProvenanceRREF;   // below: the dense-row groups' masks (S2-T6 fix 2)

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

    // S2-T5: every plane is stored sparse. The column view is a CSR index per block (wire q -> the ASCENDING indices of
    // the generators with X, resp. Z, on q: column q's set bits), always built. The dense column planes (column q at
    // offset q*words, bit i = gens[i].x/z(q)) are kept as well when they are small against it (kDenseColsRatio: small
    // groups, where the word XORs are the faster readers; a data-only switch, both give identical words). Readers go
    // through xor_col / col_begin / col_end / col_bit, never the storage.
    struct ColIndex {
        std::vector<int> off;                       // n_qubits + 1
        std::vector<int> gen;                       // generator indices, ascending per wire
    };
    using XColIndex = ColIndex;
    ColIndex xidx, zidx;
    bool dense_cols = false;
    std::vector<uint64_t> xcol, zcol;          // [n_qubits * words] iff dense_cols, else empty
    std::vector<SparsePauli> gens;              // row view (exact phases), as supplied

    // GF(2) RREF of the gens' (x|z) rows (columns 0..n-1 = X-block, n..2n-1 = Z-block),
    // exact phase carried on each row; built once.
    std::vector<SparsePauli> rref;              // rank pivot rows, ascending pivot column
    std::vector<int>   pivcol;                  // pivcol[r] = symplectic pivot column of rref[r]
    std::vector<int>   col_to_row;              // size 2n; col_to_row[c] = pivot row of column c, else -1
    int rank = 0;

    // S2-T6: the generator masks of the record columns. The elimination is kept in its FORWARD form: E_r = the pivot
    // row r when it was selected (reduced by the earlier pivots only; ech_col[ech_off[r] .. ech_off[r + 1]) = its
    // symplectic columns, ascending, X block then n + Z block), and its forward steps: E_r = g_{prov_row[r]} ⊕
    // Σ_{s ∈ steps(r)} E_s, steps(r) = the earlier pivot positions it was multiplied by, ascending (fix 2: one entry
    // per forward multiply — the elimination's own work — never a merged combination; the back-substitution that
    // turns E_r into rref[r] is not recorded; fix 4: stored per row as LEB128 varints of the position deltas,
    // prov_steps[prov_row[r]], ~1 byte a step on chains). A probe W in the group reduces against E by ascending
    // pivots (only the rows its decomposition uses); that sum is expanded through the steps by descending positions
    // with parity marks, giving the mask (Π_{mask} g = ±W). Unique when the generators are independent (rank ==
    // n_gens: the certified rows of a frame), where it is the decomposition the ProvenanceRREF below computes (its
    // oracle in the tests: the pre-T6 scan, test_planes_build).
    // THE BOUND (fix 4), against the pre-S2-T6 route (prref = the ProvenanceRREF below, built on the first mask after
    // the build, its rows 2·ceil(n/64) + words 8-byte words each):
    //   storage — the forward form's heap bytes, counted by capacity with every row's header (the step lists, the
    //     rows' last positions, E and its offsets), never exceed kFwdBytesPct % of prref's heap bytes (prref_route_bytes()):
    //     the build drops the form the moment it would, and the group takes the prref route (as eb55e1e did); E's and
    //     its offsets' reallocation transients (the old buffer beside the new one) count at the push that makes them
    //     (S2-CLEAN); a group whose time budget is spent frees the form at that query (below);
    //   time — the forward queries' units, priced at kFwdUnitWords word ops each, never exceed p = kMaskSpendPct % of
    //     the prref build's estimated word ops B = (n_gens + steps) · (2·ceil(n/64) + words) (it reduces every
    //     generator through about as many row XORs as the forward elimination took steps) plus (1 + p) times the
    //     estimated word ops C of prref's decomposes of the same queries ((rows + 1) · (2·ceil(n/64) + words) each, rows
    //     = the pivots of the query's decomposition), plus the one query that crosses it; the remaining queries take
    //     the prref route. Forward to the end costs <= p·B + (1 + p)·C; a switch after them, <= p·B + (1 + p)·C + B +
    //     the rest's decomposes. Either way the masks cost at most (1 + p) times the prref route's estimate (with the
    //     unit price calibrated on the cases where prref's words are cheapest, cache-resident rows: report S2-T6-fix4).
    // A group built on DENSE working rows (kDenseRowsRatio: small, fill-heavy groups, or switched to them part-way on
    // fill-in; ech_off empty) records nothing
    // here: its masks come from the ProvenanceRREF below, built once on the first mask (after the build's working rows
    // are freed, as before S2-T6) — tracking the combinations through the dense build cost +6-15% of compile on deep
    // dense Clifford circuits.
    // (prov_steps, prov_row, ech_off and ech_col are mutable: the first mask query that spends the time budget frees
    //  them — the remaining queries take the prref route, so F is not held beside P; S2-CLEAN, S2-T6 fix-4 review n3.)
    mutable std::vector<std::vector<uint8_t>> prov_steps;   // per generator (input row): its forward steps, varint deltas
    mutable std::vector<int32_t> prov_row;      // rank: the generator each forward row started as
    long fwd_steps = 0;                         // the forward steps recorded (Σ |steps(r)|)
    size_t fwd_bytes = 0;                       // the forward form's heap bytes after the build (the storage bound's count)
    mutable std::shared_ptr<const ProvenanceRREF> dense_prov_;   // the dense-row form (built on the first mask)
    // Drops the mask data once a holder has taken every mask it needs (the sampler, after its record columns): only
    // reduce_sparse's mask reads it.
    void release_provenance();
    bool prov_released_ = false;
    long build_multiplies = 0;                  // the elimination's row multiplies (tests: the work tripwire)
    // S2-T6 fix 3/4: the forward form's query units so far, the queries, the prref decomposes' estimated word ops on the
    // forward queries (C above), and whether its budget is spent (then the prref route serves the rest). Masks are
    // taken serially (the sampler's constructor, the DEM): not synchronised.
    mutable long mask_work_ = 0, mask_queries_ = 0, mask_credit_ = 0;
    mutable bool fwd_spent_ = false;
    // Test lever: >= 0 replaces the time budget by a query count (the forward form serves that many queries, then the
    // prref route; 0 = prref at once), so the oracle reaches both routes and every switch point. -1: the budget.
    mutable long fwd_query_limit_ = -1;
    // The heap bytes of the prref route's provenance on this group (its rows, headers and pivot tables; the storage
    // bound's yardstick) and its build's estimated word ops (B, the time bound's).
    size_t prref_route_bytes() const;
    long prref_build_words() const;
    mutable std::vector<int>     ech_off;       // rank + 1 (forward form; empty in the dense form)
    mutable std::vector<int32_t> ech_col;
    struct ProvScratch {                        // reused across calls (bits, pend and mk are left zeroed)
        std::vector<uint64_t> phi;
        std::vector<int> rows;
        std::vector<int32_t> touched;
        std::vector<uint8_t> bits;              // 2·n_qubits symplectic columns
        std::vector<uint64_t> pend, mk;         // rank bits: the reduction's pending pivots, the expansion's parity marks
        std::vector<uint64_t> gb;               // n_gens bits: the mask's generators (read ascending)
        long work = 0;                          // the forward units: flips, rows, steps read, words scanned
        std::vector<uint64_t> psv, pmask;       // the dense-row form's decompose scratch
        std::vector<int> pheap;
        SparsePauli w, t;
        Pauli wd;                               // reduce_sparse's dense accumulator once the residual fills in
    };
    // S2-T6: reduce_cheap's verdict and residual phase on a SPARSE probe (the record columns' classify): the same rows
    // (those whose pivot column W has set) multiplied in the same ascending order with pmul_sparse (bit-identical to
    // pmul_into), the ANTI test by the column readers on the residual's support — and, for IN_GROUP when mask is
    // given, the mask (W's reduction against the forward rows E, above). O(|W| + the applied rows' weight) instead of
    // O(n/64) per probe; a probe heavier than 2·ceil(n/64) entries takes the dense walk for the verdict and phase
    // (pure performance: identical results).
    Membership::Verdict reduce_sparse(const SparsePauli& W, int& phase_out, std::vector<int32_t>* mask,
                                      ProvScratch& sc) const;

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
    // Cached lazily here. The gz rows/gz_piv hold the RREF basis of G_z = { Σ c_i z_i : Σ c_i x_i = 0 } (pure-Z
    // group content), in the row order of the original in-law computation (its candidates' insertion order). (N_z is
    // no longer materialised: Kz is computed in the V-basis α-space.) S2-T10a: computed by one forward elimination of
    // the generators (twirl_law.cpp build_group_z_cache: the order and pivots; the rows read off the planes' own
    // RREF), identical to the pre-S2 dense computation, which is the oracle in cpp/tests/test_planes_sparse.cpp; the
    // rows are stored sparse (CSR): row r's Z-support, ascending wires, is gz_idx[gz_off[r] .. gz_off[r+1]).
    struct GroupZCache {
        std::vector<int>                   gz_off;          // rank + 1 (empty before the build)
        std::vector<int32_t>               gz_idx;
        std::vector<int>                   gz_piv;
        std::vector<int>                   gz_col_to_row;   // qubit column → rref row (-1 = free);
                                                            // support-directed reduce (2026-07-16)
        // lean L4: the sign of every RREF row as a group element — Z^{row} = (−1)^{gz_sign[r]} · g_r, g_r ∈ G.
        // S2-T10a: read off the planes' RREF row with that Z-support (pivot in the Z-block, no X): its exact phase is
        // 0 or 2, and the sign is its high bit (one group element per support: no −I in the group). build_shot_law's
        // C_det signs read them (reduce_vec_gz's row parity).
        std::vector<uint8_t>               gz_sign;
    };
    mutable GroupZCache gzc_;
    mutable bool        gzc_ready_ = false;
    // QW-D: set by a threaded call while its lanes may run (run_threads_): building the cache then would be an
    // unsynchronised write on the shared planes, so group_z_cache() refuses it as an engine bug (freeze_ builds it
    // first whenever a lane can read it).
    mutable bool        gzc_lanes_ = false;
    const GroupZCache& group_z_cache() const;

    // ── lean L4: the sparse view of the X-planes — per wire q the ASCENDING indices of the generators with X on q
    // (xcol(q)'s set bits; ~5 per wire on the full d5 circuit against GW = 101 words). The law's cone XORs these bits
    // one at a time where it XORed the whole column: identical words. S2-T5: the planes' own X index (built with them).
    const XColIndex& xcol_index() const { return xidx; }

    // ── perf-engine §2.1(b): the chunked pivot walk's table (twirl_planes.cpp). The reduction of a probe w against
    // the full RREF applies exactly the rows whose pivot column is set in w (a pivot column occurs in its own row
    // only, so no row changes another pivot bit), in ascending row order. The walk below reads those pivot bits in
    // one pass (pivmask per word, a parallel bit extract), then multiplies w by ONE precomputed product per 8-row
    // chunk — the ascending-order product of the chunk's selected rows (x|z words + exact phase). By associativity
    // the operator product is the same, and pmul_into computes the exact normal form of every partial product, so
    // (x, z, phase) are identical to the row-by-row walk (cpp/tests/test_planes_walk.cpp holds the two together).
    // Built lazily once per group: ceil(rank/8) chunks × 256 entries × (2·NW words + 1 byte) — ~32× the RREF. Bounded:
    // a table above kChunkTableCapBytes, or one the state budget refuses, is not built and the walk takes the
    // row-by-row path (walk_rows_, the same rows in the same order: identical by the same theorem). The table's bytes
    // are charged to the state budget for the group's lifetime (the shared ticket travels with copies of the planes).
    struct ChunkTable {
        int NW = 0, nchunks = 0;
        bool usable = false;                        // built and charged (else the row walk)
        std::vector<uint64_t> xz;                   // [chunk][byte][2*NW]: x words, then z words
        std::vector<uint8_t> ph;                    // [chunk][byte]: the product's phase (0..3)
        std::vector<uint64_t> pivmask;              // [2*NW]: pivot columns per word of (x ‖ z)
        std::shared_ptr<StateBudgetTicket> ticket;  // the charge (released with the last holder)
    };
    static constexpr int64_t kChunkTableCapBytes = int64_t(32) << 20;   // 32 MiB (factory-class groups: ~27 MiB)
    mutable ChunkTable chunk_;
    mutable bool chunk_ready_ = false;
    mutable bool chunk_off_ = false;                // oracle lever (tests): force the row walk
    const ChunkTable& chunk_table() const;

    // T6: build() == build_sparse(…, kListEntryCost): the pre-T6 build_dense's exact elimination (identical xcol, zcol,
    // rref, pivcol, col_to_row, rank; cpp/tests/test_planes_build.cpp) with a column's rows found from per-column row
    // lists (cost: total weight plus fill-in) instead of a scan of every row (2n·n_gens row visits). Adaptive: once
    // the lists have cost more (kListEntryCost row visits per entry) than the scans would have (small groups whose
    // eliminations fill up), the remaining columns use the scans — a pure-performance switch, checked at each column
    // and after each elimination. build_dense (DenseGroupPlanes, cpp/tests/planes_oracle.hpp) is kept as the test oracle.
    // S2-T5: the generators come in sparse (FramedSuperposition::certified_stabilizers_sparse) and become `gens` (moved:
    // one copy of the group). The elimination's working rows are sparse, or dense words when those are small
    // against the group (kDenseRowsRatio) — a data-only switch, the same eliminations either way; the rref rows are
    // stored sparse. Layout forces either switch (tests; the oracle lever XTIM_PLANES_LAYOUT=sparse|dense forces these two
    // in build() and, in the twirl sampler (twirl_sampler.cpp, S2-T7), the per-wire rows' and the readout-flip columns'
    // density switches too: one lever, every storage switch; read once per process).
    enum class Layout { Auto, Sparse, Dense };
    static CertifiedGroupPlanes build(std::vector<SparsePauli> certified, int n_qubits);
    static CertifiedGroupPlanes build(const std::vector<Pauli>& certified, int n_qubits);   // converts (tests, tools)
    // entry_cost: the switch's cost ratio (build: kListEntryCost); 0 = lists throughout (tests sweep it).
    // S2-FINALFIX F1: with rows == Auto, sparse working rows switch to packed words part-way once the elimination's
    // fill-in makes the words small against them (kDenseRowsRatio on the rows' current weight; never back; the forward
    // form is dropped there: the prref route) — the planes of random parity checks fill in ~25x. switch_at >= 0 (the
    // tests' lever): sparse rows, packed words from pivot position switch_at on. Identical results either way.
    static CertifiedGroupPlanes build_sparse(std::vector<SparsePauli> certified, int n_qubits, long entry_cost,
                                             Layout cols = Layout::Auto, Layout rows = Layout::Auto,
                                             long switch_at = -1);
    // The pivot position from which the working rows were packed words (0: throughout; -1: sparse throughout).
    int rows_words_from = -1;

    // Column readers (either storage, identical words). xor_col: acc[0..words) ^= column (q, block). col_gens: the
    // column's generator indices, ascending ([begin, end)). col_bit: bit i of the column.
    void xor_col(uint64_t* acc, int q, bool xblock) const {
        if (dense_cols) {
            const uint64_t* c = (xblock ? xcol.data() : zcol.data()) + (size_t)q * words;
            for (int w = 0; w < words; ++w) acc[w] ^= c[w];
            return;
        }
        const ColIndex& X = xblock ? xidx : zidx;
        for (int p = X.off[(size_t)q]; p < X.off[(size_t)q + 1]; ++p) {
            const int i = X.gen[(size_t)p];
            acc[i >> 6] ^= 1ULL << (i & 63);
        }
    }
    const int* col_begin(int q, bool xblock) const { const ColIndex& X = xblock ? xidx : zidx; return X.gen.data() + X.off[(size_t)q]; }
    const int* col_end(int q, bool xblock) const { const ColIndex& X = xblock ? xidx : zidx; return X.gen.data() + X.off[(size_t)q + 1]; }
    bool col_bit(int q, bool xblock, int i) const;
    // S2-T6 (review m1): the column index is exactly the transpose of the generators' supports (an engine-bug throw);
    // check_index_once runs it on the first call per group (the build_shot_law teeth), thread-safe.
    void check_index() const;
    void check_index_once() const;
    std::shared_ptr<std::once_flag> index_check_;

    // identity_token — FNV-1a over the group's OWN content (n_qubits, n_gens, each gen's x/z words +
    // phase), the group-identity component of the plan-cache key. Computed once and cached ON THIS
    // OBJECT (a pure function of the group's content — no aliasing hazard, unlike a map keyed on the
    // object's address). The plan memo used to recompute this O(n_gens·n) FNV on EVERY get_or_build
    // (Task 10 warm-cost floor: ~35µs/shot at n=298); this lazy cache retires that per-shot cost.
    // Mutable so it can memoise through a const CertifiedGroupPlanes& (the way the memo holds it). S2-T5: the same
    // word stream (each generator's dense words, materialised transiently inside this call only).
    uint64_t identity_token() const;

    // Classify probe p against the group. Pivot-local, allocation-light.
    Membership reduce(const Pauli& p) const;

    // Verdict + exact residual i-phase WITHOUT materialising the rep (2026-07-16 plan-build
    // optimization: the b_c / teeth loops only consume (verdict, phase); the full reduce()
    // copies the rep Pauli — two vector allocations per call). Same reduction, same phase
    // convention ((phase>>1)&1 = the ± bit for IN_GROUP). phase_out is meaningful only for
    // IN_GROUP verdicts. Reduces `w` IN PLACE (TLC: no thread-local scratch) — w is left as the
    // residual; callers pass a product they rebuild before every call.
    Membership::Verdict reduce_cheap(Pauli& w, int& phase_out) const;

    // The bytes of the planes' storage (footprint accounting): the column index (and dense planes), rref, tables.
    size_t storage_bytes() const;

private:
    void walk_(Pauli& w) const;              // the pivot walk (reduce, reduce_cheap): chunked, or walk_rows_
    void walk_rows_(Pauli& w) const;         // the row-by-row walk (no chunk table)
    bool anti_(const Pauli& w) const;        // φ(w) ≠ 0 by the column planes (termroute-a F3)
    void forward_mask_(const SparsePauli& W, std::vector<int32_t>& mask, ProvScratch& sc) const;
    size_t prref_bytes_() const;
    void mask_of_(const SparsePauli& W, std::vector<int32_t>& mask, ProvScratch& sc) const;
};

// (S2-T5's pre-S2 dense planes, DenseGroupPlanes::build_dense — the test oracle of the sparse build — live in
// cpp/tests/planes_oracle.hpp: S2-FINALFIX F2, test-only code out of the shipped engine.)

// One heap entry of ProvenanceRREF's reduction (a push and a pop) costs about as much as this many visits of the dense
// scan (a row's pivot-bit test) — the T6 switch (pure performance).
inline constexpr long kHeapEntryCost = 32;

// ── T6: the provenance RREF (S2-T6 fix 2: the masks of a DENSE-row group, built lazily by
// CertifiedGroupPlanes on the first mask, released with the provenance; the sparse-row groups use the forward form).
// Generators are inserted in index order; each is reduced against the rows so far in INSERTION order (a row is
// applied iff its pivot bit is set at that point; XOR on the symplectic bits, XOR on the generator combination) and
// kept, with pivot = its first set symplectic bit, when nonzero. decompose(W) runs the same reduction on W's
// symplectic bits and returns the combination: the channel's generator mask. Symplectic columns use the word layout
// of the original code: x word w → columns [64w, 64w+64), z word w → 64·NW + 64w + bit (NW = ceil(n/64)).
// The rows are the pre-T6 dense rows; the pre-T6 reduction tested every row's pivot bit (rows per reduction: ng·rank
// tests for the build, rank per detector), this one visits only the rows it applies: a row's pivot is never set in a
// later row, so applying the rows in ascending insertion order of the set pivot bits (a min-heap, fed with the pivot
// bits each applied row newly sets) applies exactly the scan's rows in the scan's order. When the starting pivot bits
// already cost more than a scan of all rows, or once the heap has cost more than the rows left (fill), the reduction
// uses the scan (pure performance; counts only). The pre-T6 scan is kept as the test oracle in
// cpp/tests/test_planes_build.cpp (dense_provenance, dense_decompose).
struct ProvenanceRREF {
    struct DenseRow { std::vector<uint64_t> s, c; int pc; };

    int n = 0, NW = 0, GW = 0;
    std::vector<DenseRow> rows;                 // insertion order
    std::vector<int> row_of_col;                // column → row with that pivot, else -1 (size 2·64·NW)
    std::vector<uint64_t> pivmask;              // the pivot columns (2·NW words)
    long entry_cost_ = kHeapEntryCost;          // the scan switch's cost ratio; 0 = the heap throughout (tests)

    static ProvenanceRREF build(const std::vector<SparsePauli>& gens, int n, long entry_cost = kHeapEntryCost);
    static ProvenanceRREF build(const std::vector<Pauli>& gens, int n, long entry_cost = kHeapEntryCost);   // converts
    // mask (GW words) := the combination reducing W; false iff W's symplectic residual is nonzero. The Scratch
    // overload reuses the caller's buffers across calls (the detector loop).
    struct Scratch { std::vector<uint64_t> sv; std::vector<int> heap; };
    bool decompose(const Pauli& W, std::vector<uint64_t>& mask) const;
    bool decompose(const Pauli& W, std::vector<uint64_t>& mask, Scratch& sc) const;

private:
    // Reduce sv (2·NW words) in place, XOR-ing the applied rows' combinations into cv (GW words).
    void reduce_(std::vector<uint64_t>& sv, std::vector<uint64_t>& cv, std::vector<int>& heap) const;
};

// QW-D: the number of group_z_cache builds in this process — the pin that a Clifford sampler never builds it (its
// only plan, the identity plan, reads no G_z; tests/test_qw_d_lazy.py through _xtim._group_z_cache_builds). Relaxed
// atomic; no effect on outputs.
long group_z_cache_builds();

}  // namespace qeccore
