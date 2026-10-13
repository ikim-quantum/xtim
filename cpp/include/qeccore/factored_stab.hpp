#pragma once
// A/B subsystem factorization of a bare state (factorize_framed: any chi; the affine
// factorize test-oracle twin remains chi<=2).
//
// For cultivation circuits ~91% of qubits sit in a single-qubit-stabilised PRODUCT state
// (the "A-group"); only a small entangled "B-group" carries the magic. factorize() splits
// the n qubits exactly into
//   * A: the qubits q that have a single-qubit Pauli (X_q / Y_q / Z_q) in the stabiliser
//        group — each stored as {q, axis, sign} (its product eigenstate), and
//   * block: a FramedSuperposition on the |B| remaining qubits equal to the state restricted
//        to B (the magic lives here),
// such that  s = (tensor_{A} product_eigenstates) (x) |block>_B  EXACTLY.
//
// The block is stored in the LEAN (anchor-free tableau) rep: factorize builds the exact
// |B|-qubit CanonicalStabSum restriction internally and projects it once via FramedSuperposition::from_css. At
// runtime only the lean frame/eps/free/branches are read (by framed_active_block), so the affine
// anchor is never needed past load.
//
// This header declares the full surface: the data structure + factorize (Phase 1) + the per-shot
// active-block reduction (active_set / ActiveSetWork / framed_active_block) + reduced reads (reduced_read).
#include <cstdint>
#include <vector>
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/framed_superposition.hpp"

namespace qeccore {

struct FactoredBareState {
    int n = 0;                       // global qubit count
    struct ProdQ {
        int q;                       // global qubit index
        uint8_t axis;                // 0 = X, 1 = Y, 2 = Z (matches single_pauli/measure)
        uint8_t sign;                // 0 = +, 1 = - (eigenvalue of the axis Pauli)
    };
    std::vector<ProdQ> A;            // the single-qubit-stabilised (product) qubits
    std::vector<int> cls;            // size n: global qubit -> -1 if in A, else its index in
                                     // block_to_global (i.e. its column in `block`)
    FramedSuperposition block;                 // the entangled core (lean rep), on |B| qubits (default 0-qubit)
    std::vector<int> block_to_global;// size |B|: block column -> global qubit index
    uint64_t uid = 0;                // unique per factorize_framed result: address reuse of throwaway
                                     // FactoredBareStates must not satisfy pointer-keyed caches

    FactoredBareState() : block(0) {}
};

// Greedily classify every qubit with a single-qubit stabiliser into A; the rest is B. Build
// `block` as the exact |B|-qubit restriction (stored in the lean rep). Throws std::runtime_error
// if s has chi > 2 (no ray-stabiliser basis) or if a structural invariant is violated.
FactoredBareState factorize(const CanonicalStabSum& s);

// Lean-frame version of factorize, GENERAL chi (Stage G1): same A/B split + same FactoredBareState
// result, but computed PURELY on the lean frame (FramedSuperposition bare = FramedSuperposition::from_css(s)),
// with NO affine anchor / no materialize_rays / no from_rays / no phase-fix. The A-qubit test is the
// joint commutant of the branch translates (sigma-difference span) with a common-sign read; Phase 3
// (B-restriction) is a phase-tracking GF(2) RREF on the frame Pauli rows. On chi<=2 inputs the
// result is BIT-IDENTICAL to the former chi=2 special case (and LOGICALLY EQUIVALENT to
// factorize(s): identical A set / block_to_global / cls; a B-block with the SAME measurement law —
// the frame gauge may differ). Block chi == bare chi (all free generators land in B). Throws
// std::runtime_error on a violated structural invariant (incl. a free generator with A-support).
FactoredBareState factorize_framed(const FramedSuperposition& bare);

// ── Phase 2: per-shot reduced bad-measurement ──────────────────────────────────────────────
//
// The propagated error a bad shot applies to the bare state is a DIAGONAL CLIFFORD (per-qubit
// S-power a_q in Z4 => S^{a_q}, and a CZ pattern) followed by a Pauli X^v. This matches the
// sampler's per-shot fallback gate form/order EXACTLY:
//   apply S^{a_q}/Z/Sdg per q (a_q = 1/2/3) ; apply CZ per pair ; apply X per v-bit ; measure.
struct FactoredDiagError {
    std::vector<uint8_t> a;                  // size n: S-power in Z4 per qubit (S^{a_q})
    std::vector<std::pair<int, int>> cz;     // CZ pairs (global qubit indices)
    std::vector<uint8_t> v;                  // size n: X^v Pauli (1 => X on that qubit)
};

// active_set(f, E): the qubits that must enter the reduced block = (C∩A)∪B with the
// transverse-axis rule applied (a Z-definite A-qubit whose only error action is an S-power or a
// CZ to another Z-definite qubit stays PRODUCT and is excluded). Returns the GLOBAL qubit
// indices of the active set, sorted ascending. `active_is_B` (size = result) flags which active
// qubits were already in B (vs pulled-in A-qubits). Safe superset: over-including is correct.
std::vector<int> active_set(const FactoredBareState& f, const FactoredDiagError& E,
                            std::vector<uint8_t>* active_is_B = nullptr);

// Precomputed static work for the fast active_set path. axisA/b_list/active depend ONLY on f, so
// build them ONCE per program (mirrors build_a_lookup). `active` is a PERSISTENT scratch buffer
// seeded B=1/A=0; the fast path toggles A-qubits on and restores them on exit, so it stays the
// B-seeded invariant between calls.
struct ActiveSetWork {
    std::vector<int> axisA;        // size n: A-qubit axis (0/1/2), -1 if in B
    std::vector<int> b_list;       // sorted GLOBAL indices of B qubits (cls[q] != -1) — always active
    std::vector<uint8_t> active;   // size n, PERSISTENT: seeded B=1 / A=0, toggled+restored each call
};
void build_active_set_work(const FactoredBareState& f, ActiveSetWork& w);

// Fast active_set: uses precomputed w (built from the SAME f). Output is byte-identical to
// active_set(f,E,is_b) — same ascending qubit list + same active_is_B. `w.active` MUST be the
// B-seeded invariant on entry; this call restores it on exit (clears only the A-qubits it set).
void active_set_into(const FactoredDiagError& E, ActiveSetWork& w, std::vector<int>& out,
                     std::vector<uint8_t>* active_is_B = nullptr);
// Returning convenience overload (allocates; used by the equivalence check, NOT the per-shot path).
std::vector<int> active_set(const FactoredDiagError& E, ActiveSetWork& w,
                            std::vector<uint8_t>* active_is_B = nullptr);

// reduced_read(f, E, basis, q): closed-form result for a read (basis in {0:X,1:Y,2:Z}) on a
// qubit q that stays in A (is NOT active). Returns {deterministic, value}: if deterministic is
// true the outcome is fixed at value (+1 or -1); if false the read is a fair independent coin
// (its eigen-axis after the error differs from `basis`). Precondition: q not in active_set.
struct ReadResult { bool deterministic; int value; };
ReadResult reduced_read(const FactoredBareState& f, const FactoredDiagError& E, int basis, int q);

// Fast overload: caller supplies a precomputed A-qubit lookup (global q -> ax1[q] = axis+1, or 0 if
// q is NOT an A-qubit; sgn[q] = its sign). Build it ONCE per shot stream via build_a_lookup(f) and
// reuse across the ~97 non-active A-reads/shot — the f.A-scanning overload is O(|A|) PER read (|A| can
// be ~271 on d5), an O(|A|·reads) per-shot cost. Bit-identical (same axis/sign, same closed form).
ReadResult reduced_read(const FactoredDiagError& E, int basis, int q,
                        const std::vector<uint8_t>& ax1, const std::vector<uint8_t>& sgn);
// Populate ax1/sgn from f.A (ax1[q]=axis+1, 0 if q not in A; sgn[q]=sign). Sized to f.n.
void build_a_lookup(const FactoredBareState& f, std::vector<uint8_t>& ax1, std::vector<uint8_t>& sgn);

// framed_active_block(f, E, out): build the active block's LEAN rep DIRECTLY (anchor-free), into the
// reusable `out`. Produces the same MEASUREMENT LAW as building the active block and extracting its
// lean rep, but WITHOUT the affine anchor and WITHOUT the gate-by-gate deferred-frame flush (the ~108-gate replay that
// dominates the per-bad-shot cost). Starts from f.block's lean rep (frame U + eps + free + branches;
// f.block is flushed once at factorize, no pending gates), extends the pulled-in A-qubits (product
// eigenstates), then conjugates the frame by the error RESTRICTED to the active set in ONE pass via
// conjugate_by_diag_clifford (S-powers + both-active CZs + classical-Z byproducts folded into the
// per-column S-power + X^v). `block_to_global_out` (if non-null) receives the active set's global
// indices (the column order of `out`, identical to active_block's). The lean rep's free/branches/eps
// come straight from f.block (+ a 0 eps per pulled qubit): the error is pure frame conjugation, so it
// never touches eps/free/branches — exactly as FramedSuperposition::from_css reads them off the flushed active block.
// `asw` (if non-null) routes the active-set computation through the fast precomputed-work path
// (active_set_into); else the f-scanning active_set(f,E,...). Byte-identical either way.
void framed_active_block(const FactoredBareState& f, const FactoredDiagError& E, FramedSuperposition& out,
                       std::vector<int>* block_to_global_out = nullptr,
                       ActiveSetWork* asw = nullptr);

}  // namespace qeccore
