#pragma once
#include <cstring>
#include <cstddef>
#include <vector>
#include "qeccore/pauli.hpp"
namespace qeccore {

struct CliffordTableau {
    int n = 0;
    std::vector<Pauli> Xrow;   // Xrow[a] = U X_a U†
    std::vector<Pauli> Zrow;   // Zrow[a] = U Z_a U†

    explicit CliffordTableau(int n_);

    // Copy skips the inverse tableau when it is STALE: every Xinv/Zinv content read is gated on
    // dual_valid_ (the gate patches return early, conjugate_single falls back to dual_image, and
    // reset_dual replaces both wholesale by move-assign without reading them), so stale contents
    // are unobservable — copying 2n Paulis of dead data was ~40% of a frame copy. Moves stay
    // defaulted (a user-declared copy ctor would otherwise suppress them).
    CliffordTableau(const CliffordTableau& o)
        : n(o.n), Xrow(o.Xrow), Zrow(o.Zrow), dual_valid_(o.dual_valid_) {
        if (o.dual_valid_) { Xinv = o.Xinv; Zinv = o.Zinv; }
    }
    CliffordTableau& operator=(const CliffordTableau& o) {
        if (this == &o) return *this;
        n = o.n; Xrow = o.Xrow; Zrow = o.Zrow; dual_valid_ = o.dual_valid_;
        if (o.dual_valid_) { Xinv = o.Xinv; Zinv = o.Zinv; }
        else { Xinv.clear(); Zinv.clear(); }
        return *this;
    }
    CliffordTableau(CliffordTableau&&) = default;
    CliffordTableau& operator=(CliffordTableau&&) = default;

    // Buffer-reusing copy (the clone_into hot path): when shapes match, Pauli contents are
    // memcpy'd straight into the destination's existing buffers — no vector bookkeeping, no
    // allocation. Falls back to operator= on any shape mismatch. Same observable result.
    void assign_from(const CliffordTableau& o) {
        if (n != o.n || Xrow.size() != o.Xrow.size() || Zrow.size() != o.Zrow.size()) {
            *this = o;
            return;
        }
        auto rows = [](std::vector<Pauli>& d, const std::vector<Pauli>& s) {
            for (size_t i = 0; i < s.size(); ++i) {
                Pauli& dp = d[i];
                const Pauli& sp = s[i];
                if (dp.x.size() != sp.x.size() || dp.z.size() != sp.z.size()) { dp = sp; continue; }
                std::memcpy(dp.x.data(), sp.x.data(), sp.x.size() * sizeof(uint64_t));
                std::memcpy(dp.z.data(), sp.z.data(), sp.z.size() * sizeof(uint64_t));
                dp.phase = sp.phase;
                dp.n = sp.n;
            }
        };
        rows(Xrow, o.Xrow);
        rows(Zrow, o.Zrow);
        dual_valid_ = o.dual_valid_;
        if (o.dual_valid_) {
            if (Xinv.size() == o.Xinv.size() && Zinv.size() == o.Zinv.size()) {
                rows(Xinv, o.Xinv);
                rows(Zinv, o.Zinv);
            } else {
                Xinv = o.Xinv;
                Zinv = o.Zinv;
            }
        } else {
            Xinv.clear();
            Zinv.clear();
        }
    }

    void left_h(int q);
    void left_s(int q);
    void left_sdg(int q);
    void left_x(int q);
    void left_y(int q);
    void left_z(int q);
    void left_cx(int c, int t);
    void left_cz(int c, int t);

    // U† P U (pullback), folding the INVERSE rows Xinv/Zinv — needs a valid dual (ensure_dual).
    // Semantically the same map as dual_image(); this one is O(weight) folds against the cached
    // inverse rows, dual_image() is an O(n²) symplectic decomposition needing forward rows only.
    Pauli conjugate(const Pauli& P) const;

    // U P U† (forward image), folding the FORWARD rows Xrow/Zrow — never needs the dual.
    // The forward counterpart of conjugate(); used e.g. to build a composed error tableau's
    // forward rows (framed_sampler's compose_alt_general_).
    Pauli forward_image(const Pauli& P) const;

    // U† T U for a STANDARD Pauli T (T = i^ph X^x Z^z), computed DIRECTLY from the forward rows
    // (Xrow/Zrow) WITHOUT a materialised inverse tableau — O(n²) (one symplectic decomposition of T
    // against the forward symplectic basis). Lets the single-qubit measurement path read Q = U†PU
    // after an in-place frame edit WITHOUT an O(n³) reset_dual; requires only that the forward rows
    // form a valid symplectic frame (same precondition as reset_dual). conjugate_single(pauli,q)
    // wraps it for the single-qubit lab Pauli used by measurement.
    Pauli dual_image(const Pauli& T) const;
    Pauli conjugate_single(int pauli, int q) const;

    // Lazy inverse-tableau maintenance. A single-qubit measurement edits the forward rows in place and
    // marks the inverse tableau STALE (invalidate_dual) instead of paying an O(n³) reset_dual — the
    // measurement path itself reads Q via dual_image/conjugate_single (forward rows only). The full
    // conjugate() needs a valid inverse tableau, so it calls ensure_dual() first (a no-op when valid);
    // the left_* gates patch the dual only when it is already valid, otherwise they leave it stale for
    // the next conjugate() to rebuild lazily. Consecutive measurements thus never rebuild the tableau
    // (O(n²)/measure); the rebuild is paid once, lazily, only when a multi-qubit conjugate next needs it.
    void invalidate_dual() { dual_valid_ = false; }
    void ensure_dual() const { if (!dual_valid_) { const_cast<CliffordTableau*>(this)->reset_dual(); } }
    bool dual_valid() const { return dual_valid_; }   // pick conjugate (O(w·n/64) cached rows) vs
                                                      // dual_image (O(n²)) without forcing a rebuild
    // Cached inverse row U†Z_aU / U†X_aU — caller must check dual_valid() first (contents undefined,
    // possibly empty, while the dual is stale).
    const Pauli& zinv_row(int a) const { return Zinv[a]; }
    const Pauli& xinv_row(int a) const { return Xinv[a]; }

    // ── Incremental inverse-tableau maintenance under U_new = U_old · g (right-composition). ──
    // The single-qubit measurement (Case B) edits the FORWARD rows in place as U_new = U_old·V with
    // V a product of self-inverse Cliffords (CX/CZ/H + a signed-Pauli phase correction). The inverse
    // rows transform as Xinv_new[b] = V† Xinv_old[b] V = ∏ (g† Xinv_old[b] g) — i.e. conjugate every
    // inverse row by each factor g of V (g self-inverse ⇒ g† M g = conj_g(M)), innermost factor last.
    // These right_* helpers apply ONE such factor to the WHOLE inverse tableau (every Xinv/Zinv row),
    // leaving the forward rows untouched (the measurement code owns those). O(n) per factor. The
    // overall i^k sign of V (from the m=±1 pivot phase and the Y-on-pivot phase) is applied to the
    // pivot inverse rows by right_pivot_phase. No-op when the dual is stale (nothing to maintain).
    void right_cx(int c, int t);   // V-factor CX(c,t)
    void right_cz(int c, int t);   // V-factor CZ(c,t)
    void right_h(int q);           // V-factor H_q
    // Pivot phase correction E (last V-factor of the Case-B collapse, U_new = U_unsigned · E). After
    // the unsigned C = (∏CZ)·H_p has been applied to the inverse via right_h/right_cz, this conjugates
    // every inverse row by E† (E acts only on pivot p, fixing X_p): supplying the m·i^{ph} scalar and,
    // when y_pivot (p∈q_z ⇔ ph odd), the Z_p→±Y_p √X-type correction. m∈{+1,-1}, ph = Qr.phase∈{0..3},
    // y_pivot = (p∈supp(Qr.z)). Exact gate per (m,ph,y_pivot) pinned by test_dual_maintenance.
    void right_pivot_E(int p, int m, int ph, bool y_pivot);

    static CliffordTableau from_generators(const std::vector<Pauli>& gens);
    Pauli destabilizer_product(const std::vector<uint64_t>& d) const;

    // Tensor a fresh identity qubit at column n: grow every forward row to n+1 qubits (new column
    // bit 0) and append the identity rows Xrow[n]=X_n, Zrow[n]=Z_n. The new frame is U ⊗ I_1. The
    // inverse tableau is INVALIDATED (the lazy-dual policy rebuilds it on demand). O(n) row grows.
    void grow_identity_qubit();

    // Batched grow: tensor `k` fresh identity qubits at columns n..n+k-1 in ONE pass over the existing
    // rows (single resize + n-set per row, instead of k separate full-row scans), then append the 2k
    // identity rows. Equivalent to grow_identity_qubit() called k times (same final frame, same column
    // order), but O(rows) once instead of O(k·rows). The inverse tableau is grown in step when valid
    // (mirrors grow_identity_qubit's U⊗I_k update), else left stale.
    void grow_identity_qubits(int k);

    // Tensor k product-eigenstate qubits (axes[j] in {0:X,1:Y,2:Z}, signs[j] in {0:+,1:-}) at columns
    // n..n+k-1. Equivalent to, for each j: grow_identity_qubit(); then the U_A† rotation of the new
    // column (X^sign, then H for X-axis / H·S for Y-axis). BUT only the new qubit's OWN two rows (X_q,
    // Z_q) carry a column-q bit, so the rotation is applied DIRECTLY to those 2 rows — O(k) instead of
    // the O(k·rows) cost of k·(grow + left_*) full-row scans. Bit-identical to the grow+left_* form.
    // Inverse tableau invalidated.
    void extend_with_products(const std::vector<uint8_t>& axes, const std::vector<uint8_t>& signs);

    // Recompute the dual tableau Xinv[a]=U†X_aU, Zinv[a]=U†Z_aU from the forward rows
    // (Xrow[a]=UX_aU†, Zrow[a]=UZ_aU†). Used after a measurement edits the forward tableau in
    // place (Case B). Requires the forward rows to form a valid symplectic frame: the Zrow mutually
    // commute, Xrow[a] anticommutes with Zrow[a] and commutes with Zrow[b],Xrow[b] (b≠a).
    // Phase-exact (all phases tracked via Pauli::multiply). O(n³).
    void reset_dual();

    // Forward single-Pauli conjugation rules: M ← G M G† (stateless; exact phases).
    // Public: the canonical single source of truth for conjugating a Pauli through a
    // single Clifford gate — also used by the PPR residual propagator (ppr_residual.cpp).
    static void conj_h(Pauli& M, int q);
    static void conj_s(Pauli& M, int q);    // S
    static void conj_sdg(Pauli& M, int q);  // S†
    static void conj_x(Pauli& M, int q);
    static void conj_z(Pauli& M, int q);
    static void conj_cx(Pauli& M, int c, int t);
    static void conj_cz(Pauli& M, int c, int t);

  private:
    // Dual tableau: Xinv[a] = U† X_a U, Zinv[a] = U† Z_a U.
    std::vector<Pauli> Xinv;
    std::vector<Pauli> Zinv;
    bool dual_valid_ = true;   // false after an in-place forward edit until reset_dual rebuilds it

    // Shared row-fold behind conjugate()/forward_image(): i^{P.phase} · Π over P's support of
    // xr[q] (set x-bits) and zr[q] (set z-bits), ascending q, X before Z at equal q.
    Pauli fold_rows_(const Pauli& P, const std::vector<Pauli>& xr,
                     const std::vector<Pauli>& zr) const;
};

}  // namespace qeccore
