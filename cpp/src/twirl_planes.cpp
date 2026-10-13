#include "qeccore/twirl_planes.hpp"
#include "qeccore/pauli_kernels.hpp"   // pmul_into

namespace qeccore {

// symplectic-column bit: c in [0,n) -> x_c, [n,2n) -> z_{c-n}
static inline bool sbit(const Pauli& P, int c, int n) { return c < n ? P.xbit(c) : P.zbit(c - n); }

CertifiedGroupPlanes CertifiedGroupPlanes::build(const std::vector<Pauli>& certified, int n_qubits) {
    CertifiedGroupPlanes G;
    G.n_qubits = n_qubits;
    G.n_gens   = (int)certified.size();
    G.words    = (G.n_gens + 63) / 64;
    G.gens     = certified;

    // ── Columnar bit-planes: for each qubit q, pack bit i = gens[i].x/z(q). ──
    // Build is O(n_gens · n) here (row-major scatter into columns); reduce() is the hot path,
    // build is not. n up to ~298, n_gens up to ~297 ⇒ trivially cheap.
    G.xcol.assign((size_t)n_qubits * G.words, 0);
    G.zcol.assign((size_t)n_qubits * G.words, 0);
    for (int i = 0; i < G.n_gens; ++i) {
        const Pauli& g = certified[i];
        const uint64_t bit = 1ULL << (i & 63);
        const int wo = i >> 6;
        for (int q = 0; q < n_qubits; ++q) {
            if (g.xbit(q)) G.xcol[(size_t)q * G.words + wo] |= bit;
            if (g.zbit(q)) G.zcol[(size_t)q * G.words + wo] |= bit;
        }
    }

    // ── GF(2) RREF of the (x|z) symplectic rows, exact phase carried per row. ──
    // Columns 0..n-1 = X-block, n..2n-1 = Z-block. Full reduced echelon (each pivot column has a
    // single 1, in its own row) so reduce() is a single pivot-local pass in ascending column order.
    const int two_n = 2 * n_qubits;
    std::vector<Pauli> work = certified;    // mutated in place (exact phases via pmul_into)
    G.col_to_row.assign(two_n, -1);
    int used = 0;
    for (int c = 0; c < two_n; ++c) {
        int sel = -1;
        for (int i = used; i < G.n_gens; ++i) if (sbit(work[i], c, n_qubits)) { sel = i; break; }
        if (sel < 0) continue;
        if (sel != used) std::swap(work[used], work[sel]);
        const Pauli piv = work[used];       // copy: pmul_into below mutates other rows
        for (int i = 0; i < G.n_gens; ++i) {
            if (i == used) continue;
            if (sbit(work[i], c, n_qubits)) pmul_into(work[i], piv);
        }
        G.col_to_row[c] = used;
        G.pivcol.push_back(c);
        ++used;
    }
    G.rank = used;
    G.rref.assign(work.begin(), work.begin() + used);
    return G;
}

Membership CertifiedGroupPlanes::reduce(const Pauli& p) const {
    // Reusable scratch: w starts as p, gets reduced against the RREF (w = p · ∏ used pivot rows).
    // vector copy-assign reuses w's buffer once sized ⇒ no per-call heap churn on the steady state.
    thread_local Pauli w;
    w = p;

    // Single pivot-local pass: pivot columns ascending; each pivot column occurs in exactly one
    // rref row, so clearing it never re-dirties an already-cleared earlier column.
    for (int r = 0; r < rank; ++r) {
        int c = pivcol[r];
        if (sbit(w, c, n_qubits)) pmul_into(w, rref[r]);
    }

    // Residual support?
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }

    if (zero) {
        // p's support is in the group. Residual w = p * G = i^{w.phase} I relates p to the group's
        // canonical element G of that support. The +-1 sign is the phase-2 component, EXACTLY the
        // reference convention `(residual.phase >> 1) & 1` in twirl_diag_reference.py
        // (::_reduce_to_logical / base-point read-off). Real group => w.phase is 0 or 2 for a real
        // probe; the >>1 also extracts the +-component for an off-Hermitian probe (== `(red.p//2)&1`).
        int sign = ((w.phase >> 1) & 1) ? -1 : +1;
        return Membership::in_group(sign, w);
    }

    // Non-zero residual: in the normaliser (LOGICAL) unless it anticommutes with a generator.
    // Anticommutation with the group ⟺ with some rref row (they generate the same group); and
    // anticommute(w, ·) == anticommute(p, ·) since the removed factor commutes with the group.
    for (int r = 0; r < rank; ++r)
        if (Pauli::anticommute_bit(w, rref[r])) return Membership::anti(w);

    return Membership::logical(w);
}

Membership::Verdict CertifiedGroupPlanes::reduce_cheap(const Pauli& p, int& phase_out) const {
    // Same reduction as reduce() (shared thread_local scratch semantics), rep never copied.
    // SUPPORT-DIRECTED pivot walk (2026-07-16 plan-build optimization): instead of testing all
    // `rank` pivot columns, scan w's set symplectic bits ascending and jump via col_to_row.
    // Sound and byte-identical: in the FULL reduced echelon each pivot column occurs only in
    // its own row, so a pmul can never set an earlier (or any other) pivot column — the rows
    // applied form the same set, in the same ascending pivot-column order, as reduce()'s blind
    // walk => identical residual and phase. Non-pivot bits toggled below the cursor are never
    // pivot-processed either way. Cost: O(|supp(w)| + #pmuls·row-width), not O(rank).
    thread_local Pauli w;
    w = p;
    const int NWq = (n_qubits + 63) / 64;
    for (int blk = 0; blk < 2; ++blk) {
        std::vector<uint64_t>& vec = blk ? w.z : w.x;
        for (int wq = 0; wq < NWq; ++wq) {
            uint64_t word = vec[wq];
            while (word) {
                const int b = __builtin_ctzll(word);
                const int q = wq * 64 + b;
                const int c = blk ? n_qubits + q : q;
                const int r = col_to_row[c];
                if (r >= 0) {
                    pmul_into(w, rref[r]);                    // clears bit c (pivot-exclusive)
                    word = (b == 63) ? 0 : (vec[wq] & (~0ULL << (b + 1)));
                } else {
                    word &= word - 1;                         // free column: skip
                }
            }
        }
    }
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }
    if (zero) {
        phase_out = w.phase;
        return Membership::IN_GROUP;
    }
    for (int r = 0; r < rank; ++r)
        if (Pauli::anticommute_bit(w, rref[r])) return Membership::ANTI;
    return Membership::LOGICAL;
}

// Product-wire signs (Task 11.5): scan the rref rows whose symplectic pivot lies in the Z-block
// (pure-Z stabilisers) for WEIGHT-1 rows — those are the product wires ±Z_q. Cached lazily.
const CertifiedGroupPlanes::ZAxisRREF& CertifiedGroupPlanes::z_axis_rref() const {
    if (zaxis_ready_) return zaxis_;
    zaxis_.prod_sign.assign(n_qubits, 0);
    for (int r = 0; r < rank; ++r) {
        if (pivcol[r] < n_qubits) continue;        // has X-support ⇒ not pure-Z
        const Pauli& row = rref[r];
        const int piv = pivcol[r] - n_qubits;      // z-column pivot
        int pop = 0, only = -1;
        for (size_t w = 0; w < row.z.size(); ++w) {
            uint64_t word = row.z[w];
            while (word) { only = (int)(w * 64 + __builtin_ctzll(word)); word &= word - 1; ++pop; }
        }
        // ±Z_q is a weight-1 stabiliser (row support == {q}) ⇒ product wire; carry its ±1 sign.
        if (pop == 1 && only == piv)
            zaxis_.prod_sign[piv] = ((row.phase >> 1) & 1) ? (int8_t)-1 : (int8_t)+1;
    }
    zaxis_ready_ = true;
    return zaxis_;
}

// FNV-1a over the group's own content (n_qubits, n_gens, each gen's x/z words + phase), memoised on
// the object (Task-11 warm-cost lever). Byte-for-byte identical to the free group_token() this
// replaced — the plan-cache key is unchanged, existing caches/refs stay valid.
uint64_t CertifiedGroupPlanes::identity_token() const {
    if (token_ready_) return token_cache_;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        for (int b = 0; b < 8; ++b) { h ^= (uint8_t)(v >> (8 * b)); h *= 1099511628211ull; }
    };
    mix((uint64_t)n_qubits);
    mix((uint64_t)n_gens);
    for (const Pauli& g : gens) {
        for (uint64_t w : g.x) mix(w);
        for (uint64_t w : g.z) mix(w);
        mix((uint64_t)g.phase);
    }
    token_cache_ = h;
    token_ready_ = true;
    return token_cache_;
}

}  // namespace qeccore
