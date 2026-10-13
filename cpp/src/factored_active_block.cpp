#include "qeccore/factored_stab.hpp"
#include <cstddef>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "qeccore/framed_superposition.hpp"   // conjugate_by_diag_clifford

namespace qeccore {

// ── Phase 2: per-shot reduced bad-measurement ──────────────────────────────────────────────
//
// The error is a DIAGONAL CLIFFORD (per-qubit S^{a_q}, a CZ pattern) followed by a Pauli X^v.
// We reduce its action on s = (tensor_A product eigenstates) (x) |block>_B to a small block on
// the ACTIVE qubits = (C∩A)∪B (C = qubits the diagonal Clifford touches), with the rest a
// deterministic Pauli/sign relabel.
//
// Active-set rule (the transverse-axis trap):
//   * Every B-qubit is active.
//   * A single-qubit gate (S^{a_q}) on an A-qubit NEVER entangles q -> not, by itself, active.
//   * CZ_{q,p}: entangling iff at least ONE endpoint is non-Z-definite. (On a Z-definite |b>
//     control, CZ acts as the CLASSICAL Pauli Z_partner^b -> no entanglement; on an X/Y-definite
//     endpoint CZ genuinely entangles.) If a CZ is entangling, BOTH its endpoints become active.
//   * CLOSURE: once an A-qubit is active, a CZ from it to ANY partner (even Z-definite) is
//     controlled by a superposed qubit -> that partner must join. So we grow the active set to
//     the CZ-connected closure of B ∪ {non-Z A-endpoints of any CZ}.
//   A CZ between two Z-definite, non-active A-qubits stays a deterministic (-1)^{b_i b_j} global
//   phase -> both stay product (excluded).  Over-including is SAFE; this rule is the tight one.

// Conjugate a SIGNED single-qubit Pauli stabiliser (-1)^sigma * axis by S^{a} (a in Z4), exactly.
// S X S^dag = Y ; S Y S^dag = -X ; S Z S^dag = Z. Z is fixed (sign too). X/Y cycle with sign:
//   +X -> +Y -> -X -> -Y -> +X (one S each). Returns the resulting (axis, sign).
static void conjugate_by_s(uint8_t& ax, uint8_t& sigma, uint8_t a) {
    if (ax == 2) return;                   // Z: invariant (axis and sign)
    a &= 3;
    for (uint8_t k = 0; k < a; ++k) {
        if (ax == 0) { ax = 1; }           // X -> Y (no sign change)
        else         { ax = 0; sigma ^= 1; }  // Y -> -X (sign flips)
    }
}

std::vector<int> active_set(const FactoredBareState& f, const FactoredDiagError& E,
                            std::vector<uint8_t>* active_is_B) {
    const int n = f.n;
    // axis lookup for A-qubits (else -1 => B/entangled). Reuse thread_local storage (re-filled every
    // call, NOT cached on &f — throwaway FactoredBareStates in tests reuse addresses) so the per-bad-
    // shot path doesn't malloc/free two O(n) vectors each call.
    static thread_local std::vector<int> axisA; axisA.assign(n, -1);
    for (const auto& a : f.A) axisA[a.q] = a.axis;

    static thread_local std::vector<uint8_t> active; active.assign(n, 0);
    for (int q = 0; q < n; ++q) if (f.cls[q] != -1) active[q] = 1;   // B always active

    // seed: any non-Z-definite A-qubit that is a CZ endpoint is active (its CZ entangles).
    for (const auto& pr : E.cz) {
        for (int q : {pr.first, pr.second}) {
            if (axisA[q] != -1 && axisA[q] != 2) active[q] = 1;   // X/Y-definite A endpoint
        }
    }
    // CZ-connected closure: propagate activeness across CZ edges (an active control makes its
    // partner active, regardless of the partner's axis).
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& pr : E.cz) {
            if (active[pr.first] && !active[pr.second]) { active[pr.second] = 1; changed = true; }
            if (active[pr.second] && !active[pr.first]) { active[pr.first] = 1; changed = true; }
        }
    }
    std::vector<int> out;
    for (int q = 0; q < n; ++q) if (active[q]) out.push_back(q);

    if (active_is_B) {
        active_is_B->assign(out.size(), 0);
        for (size_t i = 0; i < out.size(); ++i) (*active_is_B)[i] = (f.cls[out[i]] != -1) ? 1 : 0;
    }
    return out;
}

// ── Fast active_set: precompute the static (f-only) parts ONCE; per-shot do only the sparse closure ──
void build_active_set_work(const FactoredBareState& f, ActiveSetWork& w) {
    const int n = f.n;
    w.axisA.assign(n, -1);
    for (const auto& a : f.A) w.axisA[a.q] = a.axis;
    w.b_list.clear();
    w.active.assign(n, 0);
    for (int q = 0; q < n; ++q) if (f.cls[q] != -1) { w.active[q] = 1; w.b_list.push_back(q); }
    // b_list is ascending by construction (q scanned 0..n-1).
}

void active_set_into(const FactoredDiagError& E, ActiveSetWork& w, std::vector<int>& out,
                     std::vector<uint8_t>* active_is_B) {
    // Sparse work list of A-qubits we toggle ON (B qubits are already 1 in w.active and never added).
    static thread_local std::vector<int> added; added.clear();

    // seed: any non-Z-definite A-qubit that is a CZ endpoint is active (its CZ entangles).
    for (const auto& pr : E.cz) {
        for (int q : {pr.first, pr.second}) {
            if (w.axisA[q] != -1 && w.axisA[q] != 2 && !w.active[q]) { w.active[q] = 1; added.push_back(q); }
        }
    }
    // CZ-connected closure: an active endpoint makes its partner active (B endpoints stay 1; only
    // A-qubits flip 0->1, so we only ever record A-qubits into `added`).
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& pr : E.cz) {
            if (w.active[pr.first] && !w.active[pr.second]) { w.active[pr.second] = 1; added.push_back(pr.second); changed = true; }
            if (w.active[pr.second] && !w.active[pr.first]) { w.active[pr.first] = 1; added.push_back(pr.first); changed = true; }
        }
    }

    // out = sorted MERGE of b_list (ascending) ∪ added (sort ascending) == old 0..n scan order.
    std::sort(added.begin(), added.end());
    out.clear();
    out.reserve(w.b_list.size() + added.size());
    size_t bi = 0, ai = 0;
    const auto& B = w.b_list;
    const bool want_isb = (active_is_B != nullptr);
    if (want_isb) active_is_B->clear();
    while (bi < B.size() || ai < added.size()) {
        int q; uint8_t isb;
        if (ai >= added.size() || (bi < B.size() && B[bi] < added[ai])) { q = B[bi++]; isb = 1; }
        else                                                            { q = added[ai++]; isb = 0; }
        out.push_back(q);
        if (want_isb) active_is_B->push_back(isb);
    }

    // RESET: restore the B-seeded invariant (clear only the A-qubits we turned on).
    for (int q : added) w.active[q] = 0;
    added.clear();
}

std::vector<int> active_set(const FactoredDiagError& E, ActiveSetWork& w,
                            std::vector<uint8_t>* active_is_B) {
    std::vector<int> out;
    active_set_into(E, w, out, active_is_B);
    return out;
}

// Shared closed-form body: q is a NON-ACTIVE A-qubit with start axis ax0 / sign sigma. Its post-error
// stabiliser is single-qubit:
//   S^{a_q} rotates the AXIS (Z fixed; X<->Y by parity of a_q);
//   no CZ byproduct lands on a non-active q (proven: any CZ touching it has q Z-definite, and a
//   Z-eigenstate is invariant under CZ; a partner controlling q would make q active);
//   X^v_q conjugates: X commutes with X (no sign flip), anticommutes with Y,Z (sign flip).
// sign sigma also rotates with the axis under S only as a phase on the eigenstate, which does NOT
// change the +/- eigenvalue assignment of the rotated axis, so sigma carries through.
static inline ReadResult reduced_read_body(const FactoredDiagError& E, int basis, int q,
                                           int ax0, uint8_t sigma) {
    uint8_t newax = (uint8_t)ax0, newsign = sigma;
    conjugate_by_s(newax, newsign, E.a.empty() ? 0 : E.a[q]);
    if (!E.v.empty() && E.v[q]) {
        // X anticommutes with Y(1) and Z(2); commutes with X(0).
        if (newax != 0) newsign ^= 1;
    }
    if (basis == newax) return {true, newsign == 0 ? +1 : -1};
    return {false, 0};   // off-axis read on a product qubit: independent fair coin
}

ReadResult reduced_read(const FactoredBareState& f, const FactoredDiagError& E, int basis, int q) {
    int ax0 = -1; uint8_t sigma = 0;
    for (const auto& a : f.A) if (a.q == q) { ax0 = a.axis; sigma = a.sign; break; }
    if (ax0 < 0) throw std::runtime_error("reduced_read: q not an A-qubit");
    return reduced_read_body(E, basis, q, ax0, sigma);
}

void build_a_lookup(const FactoredBareState& f, std::vector<uint8_t>& ax1, std::vector<uint8_t>& sgn) {
    ax1.assign(f.n, 0);
    sgn.assign(f.n, 0);
    for (const auto& a : f.A) { ax1[a.q] = (uint8_t)(a.axis + 1); sgn[a.q] = a.sign; }
}

ReadResult reduced_read(const FactoredDiagError& E, int basis, int q,
                        const std::vector<uint8_t>& ax1, const std::vector<uint8_t>& sgn) {
    if (!ax1[q]) throw std::runtime_error("reduced_read: q not an A-qubit");
    return reduced_read_body(E, basis, q, (int)ax1[q] - 1, sgn[q]);
}

// ── framed_active_block: build the active block's LEAN rep directly (anchor-free, no gate flush) ──
// Mirrors the canonical active-block column order EXACTLY (B columns 0..nB-1 in f.block order, then
// pulled-in A-qubits ascending), so the result is the lean rep of the SAME active block — but the frame is
// f.block.U extended + conjugated by the error in ONE pass (conjugate_by_diag_clifford), not the
// ~108-gate deferred replay. eps/free/branches come straight from f.block (FramedSuperposition::from_css semantics:
// the error is pure frame conjugation, never touching eps/free/branches; extend pushes a 0 eps per
// pulled qubit and leaves free/branches alone — the new generators are deterministic, never free).
void framed_active_block(const FactoredBareState& f, const FactoredDiagError& E, FramedSuperposition& out,
                       std::vector<int>* block_to_global_out, ActiveSetWork* asw) {
    static thread_local std::vector<uint8_t> is_b; is_b.clear();
    static thread_local std::vector<int> act_sorted; act_sorted.clear();
    if (asw) active_set_into(E, *asw, act_sorted, &is_b);   // fast path: no per-call vector malloc
    else     act_sorted = active_set(f, E, &is_b);
    const int na = (int)act_sorted.size();
    const int nB = f.block.n();

    // axis/sign of A-qubits (global indexed). These depend ONLY on f (fixed for the whole shot
    // stream), so cache them keyed on &f and rebuild only when f changes — the per-bad-shot path
    // must NOT pay an O(n)+O(|A|) full re-scan (n=298, |A|~271 on d5) every shot.
    static thread_local const FactoredBareState* cached_f = nullptr;
    static thread_local uint64_t cached_uid = 0;
    static thread_local std::vector<int> axisA;
    static thread_local std::vector<uint8_t> signA;
    if (cached_f != &f || cached_uid != f.uid) {
        axisA.assign(f.n, -1);
        signA.assign(f.n, 0);
        for (const auto& a : f.A) { axisA[a.q] = a.axis; signA[a.q] = a.sign; }
        cached_f = &f;
        cached_uid = f.uid;
    }

    // COLUMN ORDER: B columns (f.block order), then pulled-in A-qubits (ascending). The canonical
    // active-block column order, so the measurement law matches column-for-column.
    static thread_local std::vector<int> act; act.clear(); act.reserve(na);
    for (int c = 0; c < nB; ++c) act.push_back(f.block_to_global[c]);
    static thread_local std::vector<int> pulled; pulled.clear();
    for (int i = 0; i < na; ++i) if (!is_b[i]) pulled.push_back(act_sorted[i]);
    for (int gq : pulled) act.push_back(gq);
    // global qubit -> active column. SPARSE set/clear (only the na active entries) to avoid an O(n)
    // full assign every shot; the buffer is sized to n once and left at -1 between calls (we clear
    // the entries we set at the end of the function).
    static thread_local std::vector<int> g2a;
    if ((int)g2a.size() != f.n) g2a.assign(f.n, -1);
    for (int c = 0; c < na; ++c) g2a[act[c]] = c;
    if (block_to_global_out) *block_to_global_out = act;

    // ── 1. base lean rep = f.block's frame/eps/free/branches (f.block has no pending gates). ──
    out.U.assign_from(f.block.U);       // buffer-reusing copy (no alloc when shapes already fit)
    // The lean reduction reads forward rows only (conjugate_single falls back to dual_image when the
    // dual is stale). Invalidate NOW so the extend grow_identity_qubit + rotation left_* below and the
    // batched conjugation skip ALL inverse-tableau maintenance (it would be thrown away anyway).
    out.U.invalidate_dual();
    out.eps = f.block.eps;
    out.free = f.block.free;
    out.entries() = f.block.entries();   // element-wise copy-assign reuses out's entry buffers
    out.sync_alpha_k();

    // ── 2. extend the pulled-in A-qubits as product eigenstates (frame grows, eps gains a 0). For
    //    each pulled qubit: grow U by an identity qubit, eps.push_back(0), then rotate the new column
    //    to its axis/sign with the U_A† convention (X if sign; H for X; H·S for Y). The k grows +
    //    rotations are BATCHED via CliffordTableau::extend_with_products: ONE pass over the existing
    //    rows instead of k·(grow + left_*) full-row
    //    scans (each left_* otherwise scans all ~2·n rows to rotate ONE isolated new column). The new
    //    column's rotation only touches its own 2 rows, so the batched form is bit-identical.
    //    free/branches are untouched (the new generators are deterministic, never distinguishing).
    static thread_local std::vector<uint8_t> pull_ax, pull_sg;
    pull_ax.clear(); pull_sg.clear();
    pull_ax.reserve(pulled.size()); pull_sg.reserve(pulled.size());
    for (int gq : pulled) { pull_ax.push_back((uint8_t)axisA[gq]); pull_sg.push_back(signA[gq]); }
    out.U.extend_with_products(pull_ax, pull_sg);
    for (size_t i = 0; i < pulled.size(); ++i) out.eps.push_back(0);

    // ── 3. apply the error RESTRICTED to the active set, in ONE pass over the frame rows. Build the
    //    per-column S-power array (E.a), CZ pairs (both endpoints active → local columns) and X^v
    //    mask. The classical-Z byproducts (CZ where one endpoint is a non-active Z-definite A-qubit
    //    in value b, the other active → Z_active^b) are PHASE-ONLY (conj_z: +2 iff x), identical to a
    //    +2 S-power on that column, so they fold into the per-column a array (a += 2). The S-powers,
    //    CZs and Z-byproducts all read only the FIXED x bits, so this folding/ordering is exact.
    static thread_local std::vector<uint8_t> a_col; a_col.assign(na, 0);
    static thread_local std::vector<uint8_t> v_col; v_col.assign(na, 0);
    static thread_local std::vector<std::pair<int, int>> cz_col; cz_col.clear();
    if (!E.a.empty()) for (int c = 0; c < na; ++c) a_col[c] = E.a[act[c]] & 3;
    if (!E.v.empty()) for (int c = 0; c < na; ++c) v_col[c] = E.v[act[c]] ? 1 : 0;
    for (const auto& pr : E.cz) {
        const int ci = g2a[pr.first], cj = g2a[pr.second];
        if (ci >= 0 && cj >= 0) { cz_col.push_back({ci, cj}); continue; }   // both-active CZ
        // classical-Z byproduct: non-active endpoint must be a Z-definite A-qubit (active rule);
        // if its value b = sign is 1, the active partner gains Z (a += 2).
        if (ci < 0 && cj >= 0 && axisA[pr.first] != -1 && signA[pr.first])
            a_col[cj] = (uint8_t)((a_col[cj] + 2) & 3);
        if (cj < 0 && ci >= 0 && axisA[pr.second] != -1 && signA[pr.second])
            a_col[ci] = (uint8_t)((a_col[ci] + 2) & 3);
    }
    conjugate_by_diag_clifford(out.U, a_col, cz_col, v_col);

    // restore g2a to all -1 (sparse clear: only the active entries were touched).
    for (int c = 0; c < na; ++c) g2a[act[c]] = -1;
}

}  // namespace qeccore
