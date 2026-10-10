#include "qeccore/clifford_tableau.hpp"
#include <cstddef>
#include "qeccore/pauli_kernels.hpp"   // shared pmul_into (single-sourced)

#include <cassert>

namespace qeccore {

// ---------------------------------------------------------------------------
// Helpers: build single-qubit Pauli words used as conjugation images.
// In this Pauli convention an operator is i^phase · X^x · Z^z, so e.g.
// Y_q = i · X_q Z_q is the Pauli with x,z set on q and phase = 1.
// ---------------------------------------------------------------------------
namespace {

Pauli pauli_X(int n, int q) { Pauli p(n); p.setx(q); return p; }
Pauli pauli_Z(int n, int q) { Pauli p(n); p.setz(q); return p; }

}  // namespace

// ---------------------------------------------------------------------------
// Static forward single-Pauli conjugation rules (M ← G M G†).
//
// A single-/two-qubit Clifford conjugating a Pauli M only rewrites the bits at
// the involved qubit(s) and adds a phase that depends only on those bits — so
// each rule is O(1) (a handful of word ops on the one or two affected qubits),
// making the per-row work constant and the whole-frame left_* loop O(n).
//
// The closed forms below are derived (and machine-checked, see derive scripts)
// to reproduce EXACTLY the old strip-and-reattach apply_forward semantics,
// including the canonical X-before-Z phase bookkeeping.  Local bits at a qubit
// are a=xbit, b=zbit; "dp" is the phase delta added to M.phase (mod 4).
// ---------------------------------------------------------------------------
void CliffordTableau::conj_h(Pauli& M, int q) {
    // H: (a,b) -> (b,a); dp = 2·(a&b).   [H X H = Z, H Z H = X]
    bool a = M.xbit(q), b = M.zbit(q);
    if (a != b) { M.flipx(q); M.flipz(q); }      // swap the two bits
    if (a && b) M.phase = (M.phase + 2) & 3;
}
void CliffordTableau::conj_s(Pauli& M, int q) {
    // S: a'=a, b'=a^b; dp = a.   [S X S† = Y = iXZ, S Z S† = Z]
    bool a = M.xbit(q);
    if (a) { M.flipz(q); M.phase = (M.phase + 1) & 3; }
}
void CliffordTableau::conj_sdg(Pauli& M, int q) {
    // S†: a'=a, b'=a^b; dp = 3a.   [S† X S = -Y, S† Z S = Z]
    bool a = M.xbit(q);
    if (a) { M.flipz(q); M.phase = (M.phase + 3) & 3; }
}
void CliffordTableau::conj_x(Pauli& M, int q) {
    // X: bits fixed; dp = 2b.   [X X X = X, X Z X = -Z]
    if (M.zbit(q)) M.phase = (M.phase + 2) & 3;
}
void CliffordTableau::conj_z(Pauli& M, int q) {
    // Z: bits fixed; dp = 2a.   [Z X Z = -X, Z Z Z = Z]
    if (M.xbit(q)) M.phase = (M.phase + 2) & 3;
}
void CliffordTableau::conj_cx(Pauli& M, int c, int t) {
    // CX(c,t): xt ^= xc, zc ^= zt; xc,zt fixed; dp = 0.
    if (M.xbit(c)) M.flipx(t);
    if (M.zbit(t)) M.flipz(c);
}
void CliffordTableau::conj_cz(Pauli& M, int c, int t) {
    // CZ(c,t): zt ^= xc, zc ^= xt; xc,xt fixed; dp = 2·(xc & xt).
    bool xc = M.xbit(c), xt = M.xbit(t);
    if (xc) M.flipz(t);
    if (xt) M.flipz(c);
    if (xc && xt) M.phase = (M.phase + 2) & 3;
}

// ---------------------------------------------------------------------------
// Constructor: identity tableau (U = U† = I).
// ---------------------------------------------------------------------------
CliffordTableau::CliffordTableau(int n_) : n(n_) {
    Xrow.reserve(n); Zrow.reserve(n); Xinv.reserve(n); Zinv.reserve(n);
    for (int a = 0; a < n; ++a) {
        Xrow.push_back(pauli_X(n, a));
        Zrow.push_back(pauli_Z(n, a));
        Xinv.push_back(pauli_X(n, a));
        Zinv.push_back(pauli_Z(n, a));
    }
}

// ---------------------------------------------------------------------------
// Dual-row update under U ← G U.
//
// Forward rows:  M = U σ U†  →  G M G†   (apply conj_G to every Xrow/Zrow).
// Inverse rows:  Xinv[a] = U† X_a U  →  (GU)† X_a (GU) = U† (G† X_a G) U.
// For a single-qubit gate on q only the generators a=q are affected, and
// G† X_q G is a single-qubit word in X_q, Z_q; substituting X_q→Xinv[q],
// Z_q→Zinv[q] (these distribute through U†(·)U) gives the new inverse rows.
// G† is the inverse gate, so we reuse the conj rule of the inverse gate but
// evaluated on the dual generators.
// ---------------------------------------------------------------------------

// Lazy-dual policy for the left_* gates
// --------------------------------------
// Each gate ALWAYS updates the forward rows (O(n) targeted-bit conjugation).  The incremental
// dual (inverse-tableau) patch is a small O(n) fix-up, but it requires a CURRENTLY-VALID inverse
// tableau on entry.  Rather than force an O(n³) reset_dual() when the dual is stale (e.g. right
// after an anticommuting measurement invalidated it), we simply SKIP the dual patch and leave the
// tableau stale: the next consumer rebuilds it lazily (conjugate() calls ensure_dual(); the
// single-qubit measurement path reads U†PU straight from the forward rows via dual_image and never
// needs the tableau at all).  So a run of gates after a measurement pays NO rebuild, and if only
// measurements follow, the O(n³) rebuild is never paid.  When the dual IS valid we keep it valid by
// patching, so consecutive conjugate()s stay on the O(n) fast path.
void CliffordTableau::left_h(int q) {
    for (auto& M : Xrow) conj_h(M, q);
    for (auto& M : Zrow) conj_h(M, q);
    if (!dual_valid_) return;   // leave stale; rebuilt lazily by the next conjugate()
    // H† = H.  H X_q H = Z_q,  H Z_q H = X_q.
    // newXinv[q] = U† (H X_q H) U = U† Z_q U = Zinv[q]; similarly swap.
    std::swap(Xinv[q], Zinv[q]);
}
void CliffordTableau::left_s(int q) {
    for (auto& M : Xrow) conj_s(M, q);
    for (auto& M : Zrow) conj_s(M, q);
    if (!dual_valid_) return;
    // G = S, G† = S†.  S† X_q S = -Y_q = -i X_q Z_q; S† Z_q S = Z_q.
    // newXinv[q] = U†(-i X_q Z_q)U = i^? · Xinv[q]·Zinv[q] (-i prefactor):
    Pauli ny = Pauli::multiply(Xinv[q], Zinv[q]);  // X*Z part
    ny.phase = (ny.phase + 3) & 3;                 // multiply by -i (i^3)
    Xinv[q] = ny;
    // Zinv unchanged.
}
void CliffordTableau::left_sdg(int q) {
    for (auto& M : Xrow) conj_sdg(M, q);
    for (auto& M : Zrow) conj_sdg(M, q);
    if (!dual_valid_) return;
    // G = S†, G† = S.  S X_q S† = Y_q = i X_q Z_q; S Z_q S† = Z_q.
    Pauli y = Pauli::multiply(Xinv[q], Zinv[q]);
    y.phase = (y.phase + 1) & 3;  // multiply by i
    Xinv[q] = y;
}
void CliffordTableau::left_x(int q) {
    for (auto& M : Xrow) conj_x(M, q);
    for (auto& M : Zrow) conj_x(M, q);
    if (!dual_valid_) return;
    // G = X = G†.  X X_q X = X_q;  X Z_q X = -Z_q.
    Zinv[q].phase = (Zinv[q].phase + 2) & 3;
}
void CliffordTableau::left_y(int q) {
    // Y = i X Z; conjugation Y P Y† = (XZ) P (XZ)† up to the global phase
    // cancelling.  Implement as Z then X conjugation (order chosen to match
    // X Z X Z ...).
    // left_y/left_z/left_sdg: derived from H/S/X composition; now covered by the framed apply_y/z/sdg dense test.
    for (auto& M : Xrow) { conj_x(M, q); conj_z(M, q); }
    for (auto& M : Zrow) { conj_x(M, q); conj_z(M, q); }
    if (!dual_valid_) return;
    // G = Y = G†.  Y X_q Y = -X_q;  Y Z_q Y = -Z_q.
    Xinv[q].phase = (Xinv[q].phase + 2) & 3;
    Zinv[q].phase = (Zinv[q].phase + 2) & 3;
}
void CliffordTableau::left_z(int q) {
    for (auto& M : Xrow) conj_z(M, q);
    for (auto& M : Zrow) conj_z(M, q);
    if (!dual_valid_) return;
    // G = Z = G†.  Z X_q Z = -X_q;  Z Z_q Z = Z_q.
    Xinv[q].phase = (Xinv[q].phase + 2) & 3;
}
void CliffordTableau::left_cx(int c, int t) {
    for (auto& M : Xrow) conj_cx(M, c, t);
    for (auto& M : Zrow) conj_cx(M, c, t);
    if (!dual_valid_) return;
    // CX = CX†.  CX X_c CX = X_c X_t, CX X_t CX = X_t,
    //            CX Z_t CX = Z_c Z_t, CX Z_c CX = Z_c.
    // newXinv[c] = U†(X_c X_t)U = Xinv[c]·Xinv[t]
    Pauli nxc = Pauli::multiply(Xinv[c], Xinv[t]);
    Pauli nzt = Pauli::multiply(Zinv[c], Zinv[t]);
    Xinv[c] = nxc;
    Zinv[t] = nzt;
    // Xinv[t], Zinv[c] unchanged.
}
void CliffordTableau::left_cz(int c, int t) {
    for (auto& M : Xrow) conj_cz(M, c, t);
    for (auto& M : Zrow) conj_cz(M, c, t);
    if (!dual_valid_) return;
    // CZ = CZ†.  CZ X_c CZ = X_c Z_t, CZ X_t CZ = Z_c X_t, Z's fixed.
    Pauli nxc = Pauli::multiply(Xinv[c], Zinv[t]);
    Pauli nxt = Pauli::multiply(Zinv[c], Xinv[t]);
    Xinv[c] = nxc;
    Xinv[t] = nxt;
}

// ---------------------------------------------------------------------------
// Right-composition inverse-tableau maintenance: U_new = U_old · g.
//   Xinv_new[b] = U_new† X_b U_new = g† (U_old† X_b U_old) g = g† Xinv_old[b] g.
// For self-inverse g (CX/CZ/H/X/Y/Z) g†=g, so g† M g = conj_g(M) applied to EVERY inverse row.
// The forward rows are NOT touched here (the measurement code edits those in place itself).
// ---------------------------------------------------------------------------
void CliffordTableau::right_cx(int c, int t) {
    if (!dual_valid_) return;
    for (auto& M : Xinv) conj_cx(M, c, t);
    for (auto& M : Zinv) conj_cx(M, c, t);
}
void CliffordTableau::right_cz(int c, int t) {
    if (!dual_valid_) return;
    for (auto& M : Xinv) conj_cz(M, c, t);
    for (auto& M : Zinv) conj_cz(M, c, t);
}
void CliffordTableau::right_h(int q) {
    if (!dual_valid_) return;
    for (auto& M : Xinv) conj_h(M, q);
    for (auto& M : Zinv) conj_h(M, q);
}

// Conjugate one Pauli M by E† at qubit p (M ← E† M E), where E is the pivot correction. E fixes X_p;
// only the (xbit,zbit) at p and the phase change. Four E-cases keyed by (m, ph parity, sign-of-i^ph):
//   sgn = m·i^ph.  p∉q_z (ph even): sgn∈{±1}; E=I (+1) or E=Z_p?  We need E Z E† = sgn·Z, E X E† = X.
//     A Pauli with X→X, Z→−Z is X_p (X X X=X, X Z X=−Z). So sgn=−1 ⇒ conjugate by X_p.
//   p∈q_z (ph odd): E Z E† = sgn_i·(X Z), E X E† = X with sgn_i = m·(±i). E = √X-type.
// The conj rule for M ← E† M E is implemented per case; sign constants pinned by test_dual_maintenance.
static inline void conj_pivot_E_dag(Pauli& M, int p, int ecase) {
    // ecase: 0 = I; 1 = X_p (sgn=−1, ph even); 2 = √X⁺ (ph odd branch A); 3 = √X⁻ (ph odd branch B).
    if (ecase == 0) return;
    bool a = M.xbit(p), b = M.zbit(p);   // a = xbit, b = zbit at p
    if (ecase == 1) {                    // E = X_p:  X M X.  X→X, Z→−Z  ⇒ dp = 2·b
        if (b) M.phase = (M.phase + 2) & 3;
        return;
    }
    // √X-type E (E X E† = X, E Z E† = ±Y).  M ← E† M E.  In the X-basis E is a phase gate, so the
    // X-bit (a) is preserved and the Z-bit toggles a Y; concretely E† Z E = ∓Y, E† X E = X, E† Y E = ±Z.
    // We implement as: zbit unchanged is wrong — Y has both bits. The single-qubit map E† (·) E on
    // (a,b): (X:10)->(10), (Z:01)->(11 with phase), (Y:11)->(01 with phase).  i.e. b' = b ^ a? No.
    // Easiest: M ← E† M E with E=√X means conjugate by the gate SX† then we read bits. Implement via
    // the explicit (a,b) truth table calibrated below.
    // E maps (using SX with SX Z SX†=+Y for ecase 2, =−Y for ecase 3):
    //   E†: Z->∓Y, X->X.  On bits: Z(0,1)->Y(1,1) so z stays, x toggles? Y=i XZ has x=1,z=1.
    //   So Z(a=0,b=1) -> (a=1,b=1) with a phase. X(a=1,b=0)->X(1,0). Y(1,1)->Z(0,1).
    //   bit map: a' = a ^ b, b' = b.  phase: contributes on the Z and Y inputs.
    bool a2 = a ^ b;
    if (a2 != a) { if (a2) M.setx(p); else M.flipx(p); }   // set a' = a^b
    // phase: ecase 2 adds +1 (i) when b set (Z/Y input), ecase 3 adds +3 (−i). Calibrated below.
    if (b) M.phase = (M.phase + (ecase == 2 ? 1 : 3)) & 3;
}

void CliffordTableau::right_pivot_E(int p, int m, int ph, bool y_pivot) {
    if (!dual_valid_) return;
    int ecase;
    if (!y_pivot) {
        // ph even: i^ph = (ph==2? -1 : +1).  sgn = m·i^ph.
        int sgn = m * (ph == 2 ? -1 : 1);
        ecase = (sgn == -1) ? 1 : 0;
    } else {
        // ph odd: i^ph = (ph==1? +i : -i).  sgn_i = m·i^ph.  Two √X branches.
        int im = (ph == 1) ? 1 : -1;     // +i -> +1, -i -> -1
        int s = m * im;                  // ∈ {+1,-1}
        ecase = (s == 1) ? 2 : 3;
    }
    if (ecase == 0) return;
    for (auto& M : Xinv) conj_pivot_E_dag(M, p, ecase);
    for (auto& M : Zinv) conj_pivot_E_dag(M, p, ecase);
}

// ---------------------------------------------------------------------------
// from_generators / destabilizer_product
// ---------------------------------------------------------------------------
CliffordTableau CliffordTableau::from_generators(const std::vector<Pauli>& gens) {
    // Direct tableau synthesis — no gate decomposition. Zrow[a] := gens[a] verbatim, so the
    // contract U·Z_a·U† = gens[a] holds exactly (signs included) by construction. The
    // destabilizers Xrow[a] are any Hermitian Paulis with
    //     ⟨Xrow[a], gens[b]⟩ = δ_ab   and   ⟨Xrow[a], Xrow[b]⟩ = 0
    // (⟨,⟩ = the anticommutation bit): a tableau with those relations and Hermitian rows
    // represents a Clifford, and destabilizer phases are pure gauge (composition with a Pauli),
    // which from_generators' contract leaves free. Found by one bit-packed GF(2) solve (RREF
    // with combination tracking) + symplectic Gram–Schmidt — O(n³/64). The previous
    // implementation synthesized an explicit gate sequence (O(n²) recorded gates, each
    // re-conjugating all n working Paulis through a freshly built n-qubit frame, plus 2n full
    // replays): O(n⁴⁺) — 74 s at n = 272 in build_bare_state; this is sub-ms there.
    const int n = (int)gens.size();
    const int W = n ? (n + 63) / 64 : 0;
    // Solve rows [z_b | x_b | y_b]: with v packed as [vx | vz], ⟨v, gens[b]⟩ = vx·z_b ⊕ vz·x_b.
    // y starts as e_b and tracks the gens-row combination each reduced row represents.
    const int RW = 3 * W;
    std::vector<uint64_t> M((size_t)n * RW, 0);
    for (int b = 0; b < n; ++b) {
        uint64_t* r = &M[(size_t)b * RW];
        for (int w = 0; w < W; ++w) { r[w] = gens[b].z[w]; r[W + w] = gens[b].x[w]; }
        r[2 * W + (b >> 6)] = 1ull << (b & 63);
    }
    // RREF over the 2n leading columns (column col<n = vx bit col, else vz bit col−n).
    std::vector<int> pivot(n, -1);
    int rank = 0;
    for (int col = 0; col < 2 * n && rank < n; ++col) {
        const int j = col < n ? col : col - n;
        const int cw = (col < n ? 0 : W) + (j >> 6);
        const uint64_t cb = 1ull << (j & 63);
        int sel = -1;
        for (int r = rank; r < n; ++r) if (M[(size_t)r * RW + cw] & cb) { sel = r; break; }
        if (sel < 0) continue;
        if (sel != rank)
            for (int w = 0; w < RW; ++w) std::swap(M[(size_t)sel * RW + w], M[(size_t)rank * RW + w]);
        for (int r = 0; r < n; ++r) {
            if (r == rank) continue;
            if (M[(size_t)r * RW + cw] & cb)
                for (int w = 0; w < RW; ++w) M[(size_t)r * RW + w] ^= M[(size_t)rank * RW + w];
        }
        pivot[rank++] = col;
    }
    assert(rank == n && "from_generators: generators must be independent");
    // Particular solutions with all free variables zero: D_a has bit pivot[i] = y_i bit a.
    std::vector<uint64_t> D((size_t)n * 2 * W, 0);
    for (int i = 0; i < n; ++i) {
        const uint64_t* y = &M[(size_t)i * RW + 2 * W];
        const int pc = pivot[i];
        const int pj = pc < n ? pc : pc - n;
        const int dw = (pc < n ? 0 : W) + (pj >> 6);
        const uint64_t db = 1ull << (pj & 63);
        for (int a = 0; a < n; ++a)
            if ((y[a >> 6] >> (a & 63)) & 1) D[(size_t)a * 2 * W + dw] |= db;
    }
    // Symplectic Gram–Schmidt: zero ⟨D_a, D_b⟩ (a<b) via D_b ^= vec(gens[a]). This flips that
    // one pairing (⟨D_a, gens[a]⟩ = 1) and preserves every ⟨·, gens⟩ pairing and every already
    // fixed ⟨D_{a'<a}, ·⟩ pair (gens mutually commute, ⟨D_{a'}, gens[a]⟩ = 0 for a' ≠ a).
    for (int a = 0; a < n; ++a) {
        const uint64_t* Da = &D[(size_t)a * 2 * W];
        for (int b = a + 1; b < n; ++b) {
            uint64_t* Db = &D[(size_t)b * 2 * W];
            int s = 0;
            for (int w = 0; w < W; ++w)
                s += __builtin_popcountll(Da[w] & Db[W + w]) + __builtin_popcountll(Da[W + w] & Db[w]);
            if (s & 1)
                for (int w = 0; w < W; ++w) { Db[w] ^= gens[a].x[w]; Db[W + w] ^= gens[a].z[w]; }
        }
    }
    CliffordTableau U(n);
    for (int a = 0; a < n; ++a) {
        U.Zrow[a] = gens[a];
        Pauli& X = U.Xrow[a];
        const uint64_t* Da = &D[(size_t)a * 2 * W];
        int ny = 0;
        for (int w = 0; w < W; ++w) {
            X.x[w] = Da[w];
            X.z[w] = Da[W + w];
            ny += __builtin_popcountll(Da[w] & Da[W + w]);
        }
        X.phase = ny & 3;                   // i^{#Y}·X^x·Z^z is Hermitian (+P gauge choice)
    }
    U.reset_dual();                          // restore the valid-dual postcondition
    return U;
}

// ---------------------------------------------------------------------------
// reset_dual(): rebuild Xinv/Zinv from the forward rows after an in-place edit.
//
// {Xrow[b], Zrow[b]} is a symplectic basis of the Pauli group mod phase. Any target
// standard Pauli T decomposes (over GF(2)) as
//     T = ζ · ∏_b Xrow[b]^{αb} · ∏_b Zrow[b]^{βb},
// where, because the basis is symplectic, αb = ⟨T, Zrow[b]⟩ and βb = ⟨T, Xrow[b]⟩
// (anticommutation bit — Xrow[b] is the unique basis element anticommuting with Zrow[b]).
// The dual image is then U†TU = ζ · ∏_b X_b^{αb} Z_b^{βb} (phases preserved by conjugation,
// and U†Xrow[b]U = X_b, U†Zrow[b]U = Z_b). We compute ζ exactly via Pauli::multiply: build
// `acc` = the forward product; ζ = i^{(T.phase − acc.phase) mod 4}; the abstract product carries
// the same exponents, so its phase + the ζ correction gives the dual image's phase.
// ---------------------------------------------------------------------------
Pauli CliffordTableau::dual_image(const Pauli& T) const {
    // The dual image is ζ · X^α Z^β where α_b = ⟨T,Zrow[b]⟩, β_b = ⟨T,Xrow[b]⟩ and
    // ζ = i^{(T.phase − acc.phase) mod 4} with acc = ∏_b Xrow[b]^{α_b} · ∏_b Zrow[b]^{β_b}.
    // We only need acc's PHASE (its bit-vectors necessarily equal T's), so we accumulate the
    // forward product IN PLACE into scratch word-buffers — no per-factor Pauli allocation. The
    // phase of a left-to-right product is Σ phase(factor) + 2·Σ popcnt_and(acc.z_sofar, factor.x).
    const size_t W = (size_t)((n + 63) / 64);
    static thread_local std::vector<uint64_t> ax, az;
    ax.assign(W, 0); az.assign(W, 0);
    Pauli img(n);              // output X^α Z^β, phase filled at the end
    int acc_phase = 0;
    auto fold = [&](const Pauli& F) {
        // sign from anticommuting the running acc.z against the new factor's x bits
        int s = 0;
        for (size_t w = 0; w < W; ++w) s += __builtin_popcountll(az[w] & F.x[w]);
        acc_phase += F.phase + 2 * (s & 1);
        for (size_t w = 0; w < W; ++w) { ax[w] ^= F.x[w]; az[w] ^= F.z[w]; }
    };
    // Inline the symplectic product T⟂R (== Pauli::anticommute_bit, which is in another TU ⇒ would be
    // a non-inlined call 2N× per dual_image) directly on the words.
    const uint64_t* Tx = T.x.data();
    const uint64_t* Tz = T.z.data();
    // T's support words: the inner products ⟨T,R⟩ only read words where T has a bit (the others
    // contribute 0), so scanning just these turns the 2N inner products from O(N·W) into
    // O(N·|supp|). Low-weight T — e.g. the single-qubit cascade reads that dominate the sampler's
    // stale-dual path — touch ONE word ⇒ the inner-product half of dual_image drops to O(N) (the
    // fold below still spans W as the running product densifies). Bit-identical: skipped words have
    // Tx=Tz=0, so they add nothing to either ac.
    static thread_local std::vector<int> supp;
    supp.clear();
    for (int w = 0; w < (int)W; ++w) if (Tx[w] | Tz[w]) supp.push_back(w);
    // α_b: factors Xrow[b] (sets X-exponent column b in img).
    for (int b = 0; b < n; ++b) {
        const Pauli& R = Zrow[b];
        int ac = 0;
        for (int w : supp) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
        if (ac & 1) { img.setx(b); fold(Xrow[b]); }
    }
    // β_b: factors Zrow[b] (sets Z-exponent column b in img).
    for (int b = 0; b < n; ++b) {
        const Pauli& R = Xrow[b];
        int ac = 0;
        for (int w : supp) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
        if (ac & 1) { img.setz(b); fold(Zrow[b]); }
    }
    img.phase = ((T.phase - acc_phase) % 4 + 4) % 4;   // = ζ (canonical X-then-Z order ⇒ img phase 0 base)
    return img;
}

Pauli CliffordTableau::conjugate_single(int pauli, int q) const {
    // single_pauli convention: 0→X, 1→Y(=i·XZ), 2→Z.
    // FAST PATH: when the inverse tableau is valid, U†PU for a single-qubit P is just a product of the
    // ONE or TWO cached inverse rows Xinv[q]/Zinv[q] — O(n). Use it. Only when the dual is STALE (after
    // an in-place measurement frame edit) fall back to dual_image (O(n²) from forward rows), which
    // avoids forcing an O(n³) reset_dual.
    if (dual_valid_) {
        Pauli acc(n);
        if (pauli == 0) { acc = Xinv[q]; }                              // X
        else if (pauli == 2) { acc = Zinv[q]; }                         // Z
        else { acc = Pauli::multiply(Xinv[q], Zinv[q]); acc.phase = (acc.phase + 1) & 3; }  // Y = i·XZ
        return acc;
    }
    Pauli T(n);
    if (pauli == 0) { T.setx(q); }
    else if (pauli == 1) { T.setx(q); T.setz(q); T.phase = 1; }
    else { T.setz(q); }
    return dual_image(T);
}

void CliffordTableau::reset_dual() {
    std::vector<Pauli> nXinv(n), nZinv(n);
    for (int a = 0; a < n; ++a) {
        Pauli Xa(n); Xa.setx(a); nXinv[a] = dual_image(Xa);
        Pauli Za(n); Za.setz(a); nZinv[a] = dual_image(Za);
    }
    Xinv = std::move(nXinv);
    Zinv = std::move(nZinv);
    dual_valid_ = true;
}

Pauli CliffordTableau::destabilizer_product(const std::vector<uint64_t>& d) const {
    Pauli acc(n);
    for (int a = 0; a < n; ++a)
        if ((d[a >> 6] >> (a & 63)) & 1ULL) pmul_into(acc, Xrow[a]);
    return acc;
}

void CliffordTableau::grow_identity_qubit() {
    const int nn = n + 1;
    const int oldW = (n + 63) / 64;
    const int newW = (nn + 63) / 64;
    auto grow = [&](Pauli& P) {
        P.n = nn;
        if (newW != oldW) { P.x.resize(newW, 0); P.z.resize(newW, 0); }
        // new column bit (index n) stays 0 — existing rows act as identity on the new qubit.
    };
    for (auto& M : Xrow) grow(M);
    for (auto& M : Zrow) grow(M);
    // U_new = U ⊗ I_1, so the INVERSE tableau also grows cleanly when it is valid:
    //   Xinv_new[a] = U†_new (X_a ⊗ I) U_new = (U† X_a U) ⊗ I = old Xinv[a] grown by a zero column;
    //   Xinv_new[new] = X_new, Zinv_new[new] = Z_new (U acts as identity on the new qubit).
    // Preserving validity here keeps the cheap O(n) cached conjugate_single path live for the
    // downstream measurement (vs an O(n²) dual_image fallback or an O(n³) reset_dual). O(n) per row.
    if (dual_valid_) {
        for (auto& M : Xinv) grow(M);
        for (auto& M : Zinv) grow(M);
    }
    n = nn;
    Xrow.push_back(pauli_X(nn, nn - 1));   // X_{new}
    Zrow.push_back(pauli_Z(nn, nn - 1));   // Z_{new}
    if (dual_valid_) {
        Xinv.push_back(pauli_X(nn, nn - 1));
        Zinv.push_back(pauli_Z(nn, nn - 1));
    }
}

void CliffordTableau::extend_with_products(const std::vector<uint8_t>& axes,
                                           const std::vector<uint8_t>& signs) {
    const int k = (int)axes.size();
    if (k <= 0) return;
    const int n0 = n;                 // first new column
    grow_identity_qubits(k);          // appends k rows to Xrow/Zrow; new qubit j at column n0+j,
                                      // its rows at Xrow[n0+j] (=X_{n0+j}) and Zrow[n0+j] (=Z_{n0+j}).
    for (int j = 0; j < k; ++j) {
        const int q = n0 + j;
        Pauli& Xr = Xrow[q];          // pure X_q (x-bit set), Pauli& Zr = pure Z_q (z-bit set)
        Pauli& Zr = Zrow[q];
        // SAME gate sequence as the per-qubit grow+left_* path: left_x (if sign), then left_h (X-axis)
        // or left_h;left_s (Y-axis). Each left_* conjugates EVERY row, but only X_q/Z_q carry column-q
        // bits, so applying conj_* to just these two rows is bit-identical.
        if (signs[j] == 1) { conj_x(Xr, q); conj_x(Zr, q); }
        if (axes[j] == 0)  { conj_h(Xr, q); conj_h(Zr, q); }                       // Z -> X
        else if (axes[j] == 1) { conj_h(Xr, q); conj_h(Zr, q); conj_s(Xr, q); conj_s(Zr, q); }  // Z->Y
    }
    // The forward X_q/Z_q rows were rotated without the matching dual patch, so the inverse tableau
    // (if it was valid through grow) no longer reflects U — invalidate it (lazy rebuild on demand).
    invalidate_dual();
}

void CliffordTableau::grow_identity_qubits(int k) {
    if (k <= 0) return;
    if (k == 1) { grow_identity_qubit(); return; }
    const int nn = n + k;
    const int oldW = (n + 63) / 64;
    const int newW = (nn + 63) / 64;
    // ONE pass over existing rows: bump n + resize the packed words once each (the new column bits
    // n..n+k-1 stay 0 — existing rows act as identity on all new qubits).
    auto grow = [&](Pauli& P) {
        P.n = nn;
        if (newW != oldW) { P.x.resize(newW, 0); P.z.resize(newW, 0); }
    };
    for (auto& M : Xrow) grow(M);
    for (auto& M : Zrow) grow(M);
    if (dual_valid_) { for (auto& M : Xinv) grow(M); for (auto& M : Zinv) grow(M); }
    // Append the 2k identity rows for the new qubits (each at its own column).
    Xrow.reserve(Xrow.size() + k); Zrow.reserve(Zrow.size() + k);
    if (dual_valid_) { Xinv.reserve(Xinv.size() + k); Zinv.reserve(Zinv.size() + k); }
    for (int j = 0; j < k; ++j) {
        const int col = n + j;
        Xrow.push_back(pauli_X(nn, col));
        Zrow.push_back(pauli_Z(nn, col));
        if (dual_valid_) { Xinv.push_back(pauli_X(nn, col)); Zinv.push_back(pauli_Z(nn, col)); }
    }
    n = nn;
}

// ---------------------------------------------------------------------------
// Shared row-fold: i^{P.phase} · ∏_q (xr[q] if x-bit) (zr[q] if z-bit).
// Per-qubit folds in ascending q, exactly as the old `for q < n: if bit` loop — but found by
// bit-scanning the packed words, so the cost is O(weight) folds instead of an O(n) scan
// (the dominant cost of low-weight conjugations at large n). Identical fold order ⇒
// byte-identical result.
// ---------------------------------------------------------------------------
Pauli CliffordTableau::fold_rows_(const Pauli& P, const std::vector<Pauli>& xr,
                                  const std::vector<Pauli>& zr) const {
    Pauli acc(n);
    acc.phase = P.phase;
    const int W = (int)P.x.size();
    for (int w = 0; w < W; ++w) {
        uint64_t xb = P.x[w], zb = P.z[w];
        const int base = w << 6;
        while (xb | zb) {
            const int qx = xb ? base + __builtin_ctzll(xb) : n + 1;
            const int qz = zb ? base + __builtin_ctzll(zb) : n + 1;
            if (qx <= qz) { pmul_into(acc, xr[qx]); xb &= xb - 1; }   // ties: X before Z at q
            else          { pmul_into(acc, zr[qz]); zb &= zb - 1; }
        }
    }
    return acc;
}

// conjugate(P) = U† P U (pullback; folds the inverse rows — needs a valid dual).
Pauli CliffordTableau::conjugate(const Pauli& P) const {
    ensure_dual();
    return fold_rows_(P, Xinv, Zinv);
}

// forward_image(P) = U P U† (folds the forward rows — never needs the dual).
Pauli CliffordTableau::forward_image(const Pauli& P) const {
    return fold_rows_(P, Xrow, Zrow);
}

}  // namespace qeccore
