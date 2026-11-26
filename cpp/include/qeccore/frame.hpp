#pragma once
// Frame (Stage 2, S2-T2): the frame Clifford U of a bare state kept by its parts instead of as a dense n'×n'
// tableau. It is the production frame of every FramedSuperposition and CanonicalStabSum (S2-T3, S2-T4b);
// cpp/tests/test_frame.cpp holds it to the dense CliffordTableau the pre-S2 path builds, query for query.
//
//   U = C · (U0[F_S → W_S] ⊕ π)
//
//   * F_S  the frame indices a whose rows are not identity pairs; W_S the wires those rows act on (|F_S| = |W_S|,
//          both ascending). U0 is today's CliffordTableau on |S| qubits in LOCAL coordinates: local row i is frame
//          index F_S[i], local column j is wire W_S[j].
//   * π    a bijection from the frame indices off F_S onto the wires off W_S: Zrow[a] = Z_{π(a)}, Xrow[a] = X_{π(a)}
//          there. On Clifford circuits π = id; on magic circuits it is the pivot-swap order of the stabilizer
//          generators (PLAN-stage2-amend F1), so "frame index a = wire a" does NOT hold off the support.
//   * C    the Clifford gates applied after U0 ⊕ π (today's CanonicalStabSum::pending_gates / Step-6 output
//          Clifford), in apply order, never replayed into a dense tableau while the frame stays compact.
//
// Invariants. (I1) U = C·(U0 ⊕ π) as an operator at all times. (I2) ensure_expanded().dense() is byte for byte the
// tableau today's path builds for the same circuit (from_generators on all generators is the direct sum of
// from_generators on the F_S rows and the identity on the isolated Z_q's — their destabilizer is X_q — and
// left_batch(C) + reset_dual produce unique rows). (I3) every query returns a uniquely defined operator
// (U†PU, UPU†, UZ_aU†) in its canonical (x, z, phase) form, so any correct implementation is bit-identical to
// today's conjugate / forward_image / Zrow[a]; the compact and the expanded shape give the same bytes.
//
// Queries. conjugate / forward_image are single-query scans over C: the gates in order (reverse order with the
// inverse gates for conjugate), a gate applied iff one of its targets is in the operator's touched support (the
// propagation_table.cpp Walk condition; a gate off the touched support is the identity on the operator). O(|C|)
// per query with a byte test per gate — measured 5–10× faster than a heap lightcone merge at every corpus size,
// because the cones are 6–30 % of C. sweep is the bit-sliced batch (≤ kChunk seeds per pass, a 2-plane row per
// touched wire, two phase bit-planes: left_batch's closed forms) — the transposed form for many seeds
// (classification of M record operators, the per-detector determinism gate, the rows cache). The rows cache
// (rows_z / rows_x, all UZ_aU† / UX_aU†) is one forward sweep of the 2n' base rows; Stage-1 T5's per-wire
// right-composition sweep over C (sweep_clifford_suffix) is a drop-in alternative (see build_rows_). The gate rules have ONE
// source: the plane rules mirror CliffordTableau::left_batch, the scans call CliffordTableau::conj_*.
//
// Shape. The compile builds the compact shape (owner D2). A product that edits rows calls ensure_expanded() once at
// its construction (owner D1(i)) and works on dense() exactly as today. Both are const (the shape is held in
// mutable members): the sampler's shared `bare` is a const& (amendment F2). S2-T4b: compile, DEM export and the
// default sampling path never expand it (they read it through inverse_image / inverse_images / zrow / xrow /
// certified_rows / sweep); freeze_() calls prepare_queries() before thread lanes start, and a dense-frame product
// (the semantic-oracle window, materialize) expands it once at its first use on the home — never inside a lane —
// like today's ensure_dual. The storage-size rule (the pre-approved contingency; kept in cpp/tests/test_frame.cpp) is
// not applied: the single conjugate() of a compact frame folds its inverse-image cache (below), which keeps the magic
// corpus at today's speed (S2-T4b measured the per-query scan alone at +15-360 % there).
//
// Laziness and threads. Everything lazy is const-mutable like CanonicalStabSum::flush_gates and
// CliffordTableau::ensure_dual today, and none of it is synchronised: the rows cache (rows_z / rows_x build it;
// certified_rows, row_z / row_x read it when built), the block's dual or the expanded frame's dense dual (conjugate
// and the backward sweep call ensure_dual), the inverse-image cache and its query count conj_queries_ (conjugate
// builds it after n/64 queries, S2-T4b), the expansion (ensure_expanded / dense) and a pending C (dense flushes it).
// prepare_queries() is THE contract for sharing a frame between threads: it builds the dual / the inverse-image cache
// (and flushes an expanded frame's pending gates), after which conjugate, inverse_image(s), zrow / xrow, row_z /
// row_x, certified_rows, forward_image and the forward sweep only read. Call it after the last change of the frame —
// an expansion leaves the dense dual stale (QW-B), so whoever expands a shared frame prepares it again (dense_bare_ in
// the twirl sampler) — and before the lanes start (freeze_). Not covered, so serial readers only: rows_z / rows_x
// (they build the rows cache) and a backward sweep of a compact frame with a block and pending gates (it reads the
// block's dual, built on first use). The scans use per-thread scratch.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {

// Work meter of the scans (gates applied), per thread, never reset (callers diff) — the Frame's own unit of query
// work, as dual_image_work is the dense engine's. Added once per scan (not per gate).
inline thread_local uint64_t frame_walk_work = 0;

// S2-T4b: the number of compact → expanded transitions (ensure_expanded on a compact frame) in this process — the pin
// that compile and DEM export never expand the frame (tests/test_frame_no_expand.py, through _xtim._frame_expansions).
// A flush of pending gates on an already expanded frame is not an expansion. Relaxed atomic; no effect on outputs.
long frame_expansions();
void reset_frame_expansions();

struct Frame {
    using LeftGate = CliffordTableau::LeftGate;
    static constexpr int kChunk = 1024;        // seeds per bit-sliced pass (rows of kChunk/64 words per touched wire)

    int n = 0;                                 // deferred wires = frame indices

    // ── construction ────────────────────────────────────────────────────────────────────────────────────────────
    Frame() = default;
    explicit Frame(int n_);                    // the identity frame on n wires: F_S = ∅, π = id, C = ∅
    // The compact shape from its parts. fidx = F_S, wires = W_S (ascending, |fidx| = |wires| = U0.n); pi[a] is the
    // wire of frame index a off F_S and -1 on F_S (a bijection onto the wires off W_S); U0 in local coordinates.
    static Frame compact(int n, std::vector<int> fidx, std::vector<int> wires, std::vector<int> pi,
                         CliffordTableau U0, std::vector<LeftGate> C);
    // The expanded shape around a dense tableau (F_S = all), with gates still pending in C.
    static Frame expanded(CliffordTableau U, std::vector<LeftGate> C = {});
    // (The decomposition of a dense frame by its identity pairs — the arc's test / oracle seam — is frame_split in
    // cpp/tests/test_frame.cpp, with the storage-size contingency rule: no production caller, S2-CLEAN.)

    // ── structure ───────────────────────────────────────────────────────────────────────────────────────────────
    // (S2-T3: the expanded shape stores no index maps — F_S = W_S = all, π empty — so a per-shot copy of an expanded
    // frame copies the tableau only; support_rows/support_wires build the "all" list on demand there.)
    const std::vector<int>& support_rows() const { return full_ ? all_indices_() : fidx_; }    // F_S
    const std::vector<int>& support_wires() const { return full_ ? all_indices_() : wires_; }  // W_S
    int row_pos(int a) const { return full_ ? a : fpos_[(size_t)a]; }    // local row of frame index a, -1 off F_S
    int wire_pos(int q) const { return full_ ? q : wpos_[(size_t)q]; }   // local column of wire q, -1 off W_S
    int pi(int a) const { return full_ ? -1 : pi_[(size_t)a]; }          // wire of frame index a off F_S, -1 on F_S
    int pi_inv(int q) const { return full_ ? -1 : pinv_[(size_t)q]; }    // frame index of wire q off W_S, -1 on W_S
    const std::vector<LeftGate>& gates() const { return C_; }
    // U0 in local coordinates. An in-place row edit through block() must be followed by invalidate_caches(); row
    // PRODUCTS (canonicalise's g_c ← g_c·g_p, d_p ← d_p·d_c) commute with conjugation by C, so they may be done on
    // U0 while C is non-empty.
    CliffordTableau& block() { return U0_; }
    const CliffordTableau& block() const { return U0_; }
    void append(const LeftGate& g);                                   // U ← g·U
    void append(const std::vector<LeftGate>& gs);
    void invalidate_caches() const;

    // ── queries (I3: each the unique operator, bit-identical to the dense tableau's) ─────────────────────────────
    Pauli conjugate(const Pauli& P) const;        // U† P U   (P on wires; result on frame indices)
    Pauli forward_image(const Pauli& P) const;    // U P U†   (P on frame indices; result on wires)
    SparsePauli row_z(int a) const;               // U Z_a U† (the cached row when the cache is built, else a scan)
    SparsePauli row_x(int a) const;               // U X_a U†
    const std::vector<SparsePauli>& rows_z() const;   // all rows, built once by one sweep (cached)
    const std::vector<SparsePauli>& rows_x() const;
    // rows_z()[a] for a ∉ free in ascending a, the (-1)^eps[a] sign folded into the phase when eps is given
    // (FramedSuperposition::certified_stabilizers' rows). eps, when given, must have n entries. Reads the rows cache
    // when built; otherwise sweeps the certified Z rows only and builds no cache (S2-T4b).
    std::vector<SparsePauli> certified_rows(const std::vector<int>& free, const std::vector<uint8_t>* eps = nullptr) const;
    // Batch: out[i] = U seeds[i] U† (forward; seeds on frame indices) or U† seeds[i] U (backward; seeds on wires).
    void sweep(const std::vector<SparsePauli>& seeds, bool forward, std::vector<SparsePauli>& out) const;

    // ── S2-T4b: the compile / DEM readers' forms (never expand) ─────────────────────────────────────────────────
    // U† P U as today's dense readers compute it on an expanded frame with no pending gate (the cached inverse rows
    // when the dual is valid, else dual_image — never a dense dual rebuild); otherwise the compact query conjugate().
    // Both are the unique operator (I3).
    Pauli inverse_image(const Pauli& P) const {
        if (expanded() && C_.empty()) return U0_.dual_valid() ? U0_.conjugate(P) : U0_.dual_image(P);
        return conjugate(P);
    }
    // out[i] = U† Ps[i] U for many operators: inverse_image per operator on an expanded frame with no pending gate,
    // else one backward sweep (bit-sliced, the transposed form of the per-query scan; equal to it by I3).
    void inverse_images(const std::vector<Pauli>& Ps, std::vector<Pauli>& out) const;
    void inverse_images(const std::vector<SparsePauli>& seeds, std::vector<Pauli>& out) const;   // seeds canonical
    // Zrow[a] / Xrow[a] (U Z_a U† / U X_a U†) as dense Paulis: the dense row on an expanded frame with no pending
    // gate, else to_dense(row_z / row_x) (cached row or a forward scan; read-only).
    Pauli zrow(int a) const { return expanded() && C_.empty() ? U0_.Zrow[(size_t)a] : to_dense(row_z(a)); }
    Pauli xrow(int a) const { return expanded() && C_.empty() ? U0_.Xrow[(size_t)a] : to_dense(row_x(a)); }
    // The threading contract (header, "Laziness and threads"): build everything the queries would otherwise build
    // lazily (unsynchronised) — on an expanded frame the flushed gates and the dense dual (today's ensure_dual); on a
    // compact frame the inverse-image cache (C ≠ ∅) or the block's dual (C = ∅). Afterwards the queries the header
    // lists read only, until the frame changes. The owner calls it before thread lanes share the frame (freeze_),
    // after expanding a shared frame (dense_bare_), and before the rotation model's free_dir_of rows. It never expands
    // a compact frame.
    void prepare_queries() const;
    // prepare_queries() has run since the last change (the threads' Debug assert).
    bool queries_prepared() const { return expanded() ? C_.empty() && U0_.dual_valid() : C_.empty() ? U0_.dual_valid() : d_.inv_ready_; }
    // S2-T10a (the S2-T2 lane rule): set by the owner while threads share the frame read-only (run_threads_, for the
    // call's lanes), cleared after. Frozen, an expansion or a flush of pending gates (ensure_expanded / dense) would be
    // an unsynchronised write: a Debug assert (no shipped cost). A copy of the frame is not frozen (it is private).
    void freeze(bool on) const { frozen_.on = on; }
    bool frozen() const { return frozen_.on; }

    // ── shape ───────────────────────────────────────────────────────────────────────────────────────────────────
    bool expanded() const { return full_; }
    // An in-place edit of dense() that changed the tableau's qubit count (grow_identity_qubits, extend_with_products,
    // assign_from of another size) is followed by this: the expanded frame takes the tableau's n (O(1)).
    void resync_dense_size() {
        if (!full_) throw std::logic_error("Frame::resync_dense_size: the frame is compact (engine bug)");
        n = U0_.n;
    }
    // F_S := all: U0 becomes the dense n-qubit tableau U0 ⊕ π with C replayed (left_batch); its dual is left stale
    // and built on first use (QW-B lazy dual; no stale dual storage is kept); C := ∅. Idempotent (an expanded frame
    // only flushes C, as left_batch does). const:
    // the owner of a shared const& frame expands it once before lanes start.
    void ensure_expanded() const;
    // The dense tableau: expands a compact frame (ensure_expanded) and flushes a pending C first, as
    // CanonicalStabSum::flush_gates does; afterwards expanded() holds.
    // Inline fast path: an expanded frame with no pending gate is the tableau itself (the per-shot readers' case). The
    // mutable form hands out the rows for in-place edits, so it drops a built rows cache first (slow path).
    CliffordTableau& dense() { if (expanded() && C_.empty() && !d_.rows_ready_ && !d_.inv_ready_) return U0_; invalidate_caches(); return dense_slow_(); }
    const CliffordTableau& dense() const { if (expanded() && C_.empty()) return U0_; return dense_slow_(); }
    // Storage accounting of the parts (U0 rows incl. a valid dual, the index maps, C, the rows cache and the inverse-image
    // cache when built).
    size_t compact_bytes() const;
    // S2-T10a: the parts alone (without the derived caches) — what a copy of the frame holds (a copy starts without
    // them), so the bare-state memo's entry bytes, which a copy is stored under, never depend on caches built later.
    size_t parts_bytes() const;

  private:
    // The shape is mutable so that ensure_expanded() / dense() can be const (see "Shape" above).
    mutable bool full_ = true;                 // the expanded shape (F_S = W_S = all; the index maps below are empty)
    mutable std::vector<int> fidx_, wires_, fpos_, wpos_, pi_, pinv_;
    mutable std::vector<int> all_;             // 0..n-1 for support_rows/support_wires of an expanded frame (on demand)
    const std::vector<int>& all_indices_() const;
    mutable CliffordTableau U0_{0};
    mutable std::vector<LeftGate> C_;
    // S2-T4b: the inverse images U† X_q U / U† Z_q U of every wire, so a single conjugate() is the fold of the query's
    // support exactly as CliffordTableau::conjugate folds the dual rows — instead of a scan over C. Sparse when the
    // base is a relabelling (F_S = ∅: the Clifford circuits; one backward sweep of the 2n wire Paulis, relabelled by
    // π⁻¹). With a nonempty block (the magic circuits' small frames, where the images fill the block) it is today's
    // dense dual: built as today's frame builds it (the forward rows of U in a transient tableau, reset_dual), the
    // forward rows then dropped — n²/2 bytes, as today; the frame keeps its compact shape. Built once the single
    // queries' scans would have cost about as much (ski rental: conj_queries_ ≥ n/64 — measured on the private
    // corpus, a scan costs ~1/50 of the build), or by prepare_queries() (before threads share the frame; the
    // rotation model's free_dir_of rows). Same unique operator either way (I3).
    // The derived query data (the rows cache and the inverse-image cache): a pure function of the frame, so a copy of a
    // Frame starts without it (S2-T4b: the per-call work copies of a bare state — the noiseless reference, the oracle
    // windows — rebuild only what they read; moves keep it).
    struct Derived {
        std::vector<SparsePauli> rows_z_, rows_x_;
        bool rows_ready_ = false;
        std::vector<SparsePauli> inv_x_, inv_z_;
        CliffordTableau inv_tab_{0};                 // dense: the dual rows only (forward rows dropped)
        bool inv_ready_ = false, inv_dense_ = false;
        uint64_t conj_queries_ = 0;
        Derived() = default;
        Derived(const Derived&) {}
        Derived(Derived&&) = default;
        Derived& operator=(const Derived&) { *this = Derived(); return *this; }
        Derived& operator=(Derived&&) = default;
    };
    mutable Derived d_;
    void build_inv_() const;
    CliffordTableau base_tableau_() const;          // U0 ⊕ π as a dense tableau (forward rows, dual stale)
    struct FreezeFlag {
        bool on = false;
        FreezeFlag() = default;
        FreezeFlag(const FreezeFlag&) {}
        FreezeFlag& operator=(const FreezeFlag&) { return *this; }
    };
    mutable FreezeFlag frozen_;

    CliffordTableau& dense_slow_() const;
    void set_support_(std::vector<int> fidx, std::vector<int> wires, std::vector<int> pi) const;
    void walk_(Pauli& M, bool forward) const;                       // M ← C M C† (forward) or C† M C (backward)
    Pauli base_forward_(const Pauli& P) const;                      // (U0 ⊕ π) P (U0 ⊕ π)†: frame indices → wires
    Pauli base_backward_(const Pauli& M) const;                     // (U0 ⊕ π)† M (U0 ⊕ π): wires → frame indices
    void sweep_gates_(const std::vector<SparsePauli>& seeds, bool forward, std::vector<SparsePauli>& out) const;
    void build_rows_() const;
    void base_row_(int a, bool isz, SparsePauli& out) const;       // (U0 ⊕ π) Z_a/X_a (U0 ⊕ π)† on wires
    bool base_is_identity_() const;                                 // F_S = ∅ and π = id (U = C)
    void relabel_(const SparsePauli& in, bool forward, SparsePauli& out) const;   // F_S = ∅: π (forward) / π⁻¹
};

}  // namespace qeccore
