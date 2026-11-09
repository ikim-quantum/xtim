#include "qeccore/clifford_tableau.hpp"
#include <cstddef>
#include "qeccore/pauli_kernels.hpp"   // shared pmul_into (single-sourced)

#include <algorithm>
#include <cassert>
#include <memory>

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
// from_generators
// ---------------------------------------------------------------------------
CliffordTableau CliffordTableau::from_generators_dense_oracle(const std::vector<Pauli>& gens) {
    // lean L6: the dense synthesis is kept as the TEST ORACLE of the sparse from_generators below (identical output).
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
    U.reset_dual_oracle();                   // restore the valid-dual postcondition (the oracle's own rebuild)
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
    // We only need acc's PHASE (its bit-vectors necessarily equal T's), so we accumulate only the
    // running Z-words of the forward product — no per-factor Pauli allocation. The phase of a
    // left-to-right product is Σ phase(factor) + 2·Σ popcnt_and(acc.z_sofar, factor.x).
    // TLC: the W running words live on the stack (heap only past 16 words, n > 1024, where this
    // kernel's ≥ 2n·W word operations dwarf one allocation) — no thread-local scratch.
    const size_t W = (size_t)((n + 63) / 64);
    constexpr size_t kInlineWords = 16;
    uint64_t az_inline[kInlineWords];
    std::unique_ptr<uint64_t[]> az_heap;
    uint64_t* az = az_inline;
    if (W > kInlineWords) { az_heap.reset(new uint64_t[W]); az = az_heap.get(); }
    std::fill(az, az + W, 0ull);
    Pauli img(n);              // output X^α Z^β, phase filled at the end
    int acc_phase = 0;
    auto fold = [&](const Pauli& F) {
        // sign from anticommuting the running acc.z against the new factor's x bits
        int s = 0;
        for (size_t w = 0; w < W; ++w) s += __builtin_popcountll(az[w] & F.x[w]);
        acc_phase += F.phase + 2 * (s & 1);
        for (size_t w = 0; w < W; ++w) az[w] ^= F.z[w];
    };
    // Inline the symplectic product T⟂R (== Pauli::anticommute_bit, which is in another TU ⇒ would be
    // a non-inlined call 2N× per dual_image) directly on the words.
    const uint64_t* Tx = T.x.data();
    const uint64_t* Tz = T.z.data();
    // T's support word range [wlo, whi]: the inner products ⟨T,R⟩ only read words where T has a bit
    // (the others contribute 0), so scanning just these turns the 2N inner products from O(N·W)
    // into O(N·(whi−wlo+1)). Low-weight T — e.g. the single-qubit cascade reads that dominate the
    // sampler's stale-dual path — touch ONE word ⇒ the inner-product half of dual_image drops to
    // O(N) (the fold below still spans W as the running product densifies). Bit-identical: words
    // outside T's support (incl. zero words inside the range) have Tx=Tz=0 and add nothing.
    int wlo = 0, whi = -1;
    for (int w = 0; w < (int)W; ++w)
        if (Tx[w] | Tz[w]) { if (whi < 0) wlo = w; whi = w; }
    // α_b: factors Xrow[b] (sets X-exponent column b in img).
    for (int b = 0; b < n; ++b) {
        const Pauli& R = Zrow[b];
        int ac = 0;
        for (int w = wlo; w <= whi; ++w) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
        if (ac & 1) { img.setx(b); fold(Xrow[b]); }
    }
    // β_b: factors Zrow[b] (sets Z-exponent column b in img).
    for (int b = 0; b < n; ++b) {
        const Pauli& R = Xrow[b];
        int ac = 0;
        for (int w = wlo; w <= whi; ++w) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
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

void CliffordTableau::reset_dual_oracle() {
    // lean L6: the 2n-dual_image rebuild, kept as the TEST ORACLE of reset_dual below.
    std::vector<Pauli> nXinv(n), nZinv(n);
    for (int a = 0; a < n; ++a) {
        Pauli Xa(n); Xa.setx(a); nXinv[a] = dual_image(Xa);
        Pauli Za(n); Za.setz(a); nZinv[a] = dual_image(Za);
    }
    Xinv = std::move(nXinv);
    Zinv = std::move(nZinv);
    dual_valid_ = true;
}

// ---------------------------------------------------------------------------
// lean L6: reset_dual through the TRANSPOSED forward rows. dual_image(T) for T = X_a reads
// α_b = ⟨X_a, Zrow[b]⟩ = Zrow[b].z bit a and β_b = ⟨X_a, Xrow[b]⟩ = Xrow[b].z bit a — i.e. column a of the
// z-planes of Zrow / Xrow — and T = Z_a reads the x-planes; img = X^α Z^β, so the four column planes ARE the
// inverse rows' bit content. The phase is dual_image's fold (Xrow[b] over α's set bits ascending, then
// Zrow[b] over β's, the running Z words against each factor's X words) over the set bits only. Identical
// output: the inverse rows are the unique Paulis U†X_aU / U†Z_aU, and the fold is the same sequence.
// Cost O(n²/64 + Σ_a (|α_a| + |β_a|)·W) instead of 2n × O(n·W) inner products.
// ---------------------------------------------------------------------------
void CliffordTableau::reset_dual() {
    const int W = (n + 63) / 64;
    std::vector<Pauli> nXinv, nZinv;
    nXinv.reserve((size_t)n); nZinv.reserve((size_t)n);
    // column planes over the row index b: ZZ[a] = {b : Zrow[b].z bit a}, XZ[a] = {b : Xrow[b].z bit a},
    // ZX[a] = {b : Zrow[b].x bit a}, XX[a] = {b : Xrow[b].x bit a}; each n × W words.
    std::vector<uint64_t> ZZ((size_t)n * W, 0), XZ((size_t)n * W, 0), ZX((size_t)n * W, 0), XX((size_t)n * W, 0);
    auto scatter = [&](const std::vector<Pauli>& rows, std::vector<uint64_t>& px, std::vector<uint64_t>& pz) {
        for (int b = 0; b < n; ++b) {
            const Pauli& M = rows[(size_t)b];
            const uint64_t bb = 1ull << (b & 63);
            const size_t bw = (size_t)(b >> 6);
            for (int w = 0; w < W; ++w) {
                for (uint64_t bits = M.x[(size_t)w]; bits; bits &= bits - 1)
                    px[(size_t)((w << 6) + __builtin_ctzll(bits)) * W + bw] |= bb;
                for (uint64_t bits = M.z[(size_t)w]; bits; bits &= bits - 1)
                    pz[(size_t)((w << 6) + __builtin_ctzll(bits)) * W + bw] |= bb;
            }
        }
    };
    scatter(Zrow, ZX, ZZ);
    scatter(Xrow, XX, XZ);
    std::vector<uint64_t> az((size_t)W, 0);
    // dual_image's fold: acc_phase += F.phase + 2·parity(az ∧ F.x); az ^= F.z.
    auto image = [&](const uint64_t* alpha, const uint64_t* beta, int tphase) {
        Pauli img(n);
        std::fill(az.begin(), az.end(), 0ull);
        int acc_phase = 0;
        auto fold = [&](const Pauli& F) {
            int sgn = 0;
            for (int w = 0; w < W; ++w) sgn += __builtin_popcountll(az[(size_t)w] & F.x[(size_t)w]);
            acc_phase += F.phase + 2 * (sgn & 1);
            for (int w = 0; w < W; ++w) az[(size_t)w] ^= F.z[(size_t)w];
        };
        for (int w = 0; w < W; ++w) {
            img.x[(size_t)w] = alpha[w];
            for (uint64_t bits = alpha[w]; bits; bits &= bits - 1) fold(Xrow[(size_t)((w << 6) + __builtin_ctzll(bits))]);
        }
        for (int w = 0; w < W; ++w) {
            img.z[(size_t)w] = beta[w];
            for (uint64_t bits = beta[w]; bits; bits &= bits - 1) fold(Zrow[(size_t)((w << 6) + __builtin_ctzll(bits))]);
        }
        img.phase = ((tphase - acc_phase) % 4 + 4) % 4;
        return img;
    };
    for (int a = 0; a < n; ++a) {
        nXinv.push_back(image(&ZZ[(size_t)a * W], &XZ[(size_t)a * W], 0));   // T = X_a (phase 0)
        nZinv.push_back(image(&ZX[(size_t)a * W], &XX[(size_t)a * W], 0));   // T = Z_a
    }
    Xinv = std::move(nXinv);
    Zinv = std::move(nZinv);
    dual_valid_ = true;
}

// ---------------------------------------------------------------------------
// lean L6: from_generators on SPARSE rows. The dense oracle above runs one full RREF with combination
// tracking over the rows [z_b | x_b | y_b] and the symplectic Gram–Schmidt; the RREF of a full-row-rank
// matrix is unique (and so is the combination y of each reduced row, the gens being a basis of the row
// space), the rows come out in ascending pivot-column order, and Gram–Schmidt's step a reads only the
// state at the start of step a (D_a is fixed, D_b ^= gens[a] touches D_b alone) — so a sparse-row RREF
// and a column-plane Gram–Schmidt give the identical D, rows and phases.
// ---------------------------------------------------------------------------
CliffordTableau CliffordTableau::from_generators(const std::vector<Pauli>& gens) {
    const int n = (int)gens.size();
    const int W = n ? (n + 63) / 64 : 0;
    // Sparse rows over 3n columns: [0, n) the vx coefficient (gens' z bits), [n, 2n) the vz coefficient (x bits),
    // [2n, 3n) the combination y (e_b at the start). Sorted column lists; per leading column the rows that
    // acquired a bit there (lazily deleted: a listed row is re-checked before use).
    std::vector<std::vector<int>> rows((size_t)n), colrows((size_t)2 * (size_t)n);
    for (int b = 0; b < n; ++b) {
        std::vector<int>& r = rows[(size_t)b];
        for (int w = 0; w < W; ++w)
            for (uint64_t bits = gens[(size_t)b].z[(size_t)w]; bits; bits &= bits - 1) r.push_back((w << 6) + __builtin_ctzll(bits));
        for (int w = 0; w < W; ++w)
            for (uint64_t bits = gens[(size_t)b].x[(size_t)w]; bits; bits &= bits - 1) r.push_back(n + (w << 6) + __builtin_ctzll(bits));
        r.push_back(2 * n + b);
        for (int c : r) if (c < 2 * n) colrows[(size_t)c].push_back(b);
    }
    auto has = [&](const std::vector<int>& r, int c) { return std::binary_search(r.begin(), r.end(), c); };
    std::vector<uint8_t> is_pivot((size_t)n, 0);
    std::vector<int> order;                       // pivot rows in ascending pivot-column order
    std::vector<int> pivot_col((size_t)n, -1);
    std::vector<int> merged;
    order.reserve((size_t)n);
    for (int col = 0; col < 2 * n && (int)order.size() < n; ++col) {
        int sel = -1;
        for (int r : colrows[(size_t)col])
            if (!is_pivot[(size_t)r] && has(rows[(size_t)r], col)) { sel = r; break; }
        if (sel < 0) continue;
        is_pivot[(size_t)sel] = 1;
        pivot_col[(size_t)sel] = col;
        order.push_back(sel);
        const std::vector<int>& pr = rows[(size_t)sel];
        for (int r : colrows[(size_t)col]) {
            if (r == sel || !has(rows[(size_t)r], col)) continue;
            std::vector<int>& rr = rows[(size_t)r];
            merged.clear();
            size_t i = 0, j = 0;                  // symmetric difference of two sorted lists
            while (i < rr.size() || j < pr.size()) {
                if (j == pr.size() || (i < rr.size() && rr[i] < pr[j])) merged.push_back(rr[i++]);
                else if (i == rr.size() || pr[j] < rr[i]) {
                    if (pr[j] < 2 * n) colrows[(size_t)pr[j]].push_back(r);   // newly set leading column
                    merged.push_back(pr[j++]);
                } else { ++i; ++j; }
            }
            rr.swap(merged);
        }
    }
    assert((int)order.size() == n && "from_generators: generators must be independent");
    // Particular solutions with all free variables zero: D_a has bit pivot[i] = y_i bit a.
    std::vector<uint64_t> D((size_t)n * 2 * W, 0);
    for (int i = 0; i < n; ++i) {
        const int pc = pivot_col[(size_t)order[(size_t)i]];
        const int pj = pc < n ? pc : pc - n;
        const int dw = (pc < n ? 0 : W) + (pj >> 6);
        const uint64_t db = 1ull << (pj & 63);
        for (int c : rows[(size_t)order[(size_t)i]])
            if (c >= 2 * n) D[(size_t)(c - 2 * n) * 2 * W + dw] |= db;
    }
    // Symplectic Gram–Schmidt on column planes: Cx[q] = {b : D_b.x bit q}, Cz[q] = {b : D_b.z bit q} (n × W
    // words each). Step a: the pairings ⟨D_a, D_b⟩ for every b at once are ⊕_{q ∈ x(D_a)} Cz[q] ⊕ ⊕_{q ∈ z(D_a)}
    // Cx[q]; every b > a with an odd pairing gets D_b ^= vec(gens[a]) (rows and planes).
    std::vector<uint64_t> Cx((size_t)n * W, 0), Cz((size_t)n * W, 0);
    for (int b = 0; b < n; ++b) {
        const uint64_t* Db = &D[(size_t)b * 2 * W];
        const uint64_t bb = 1ull << (b & 63);
        const size_t bw = (size_t)(b >> 6);
        for (int w = 0; w < W; ++w) {
            for (uint64_t bits = Db[w]; bits; bits &= bits - 1) Cx[(size_t)((w << 6) + __builtin_ctzll(bits)) * W + bw] |= bb;
            for (uint64_t bits = Db[W + w]; bits; bits &= bits - 1) Cz[(size_t)((w << 6) + __builtin_ctzll(bits)) * W + bw] |= bb;
        }
    }
    std::vector<uint64_t> P((size_t)W, 0);
    for (int a = 0; a < n; ++a) {
        const uint64_t* Da = &D[(size_t)a * 2 * W];
        std::fill(P.begin(), P.end(), 0ull);
        for (int w = 0; w < W; ++w) {
            for (uint64_t bits = Da[w]; bits; bits &= bits - 1) {
                const uint64_t* cz = &Cz[(size_t)((w << 6) + __builtin_ctzll(bits)) * W];
                for (int t = 0; t < W; ++t) P[(size_t)t] ^= cz[t];
            }
            for (uint64_t bits = Da[W + w]; bits; bits &= bits - 1) {
                const uint64_t* cx = &Cx[(size_t)((w << 6) + __builtin_ctzll(bits)) * W];
                for (int t = 0; t < W; ++t) P[(size_t)t] ^= cx[t];
            }
        }
        // b > a only
        P[(size_t)(a >> 6)] &= ~((2ull << (a & 63)) - 1ull);   // clear bits 0..a of that word
        for (int t = 0; t < (a >> 6); ++t) P[(size_t)t] = 0;
        const Pauli& g = gens[(size_t)a];
        for (int t = 0; t < W; ++t)
            for (uint64_t bits = P[(size_t)t]; bits; bits &= bits - 1) {
                const int b = (t << 6) + __builtin_ctzll(bits);
                uint64_t* Db = &D[(size_t)b * 2 * W];
                const uint64_t bb = 1ull << (b & 63);
                const size_t bw = (size_t)(b >> 6);
                for (int w = 0; w < W; ++w) {
                    Db[w] ^= g.x[(size_t)w];
                    Db[W + w] ^= g.z[(size_t)w];
                    for (uint64_t gb = g.x[(size_t)w]; gb; gb &= gb - 1) Cx[(size_t)((w << 6) + __builtin_ctzll(gb)) * W + bw] ^= bb;
                    for (uint64_t gb = g.z[(size_t)w]; gb; gb &= gb - 1) Cz[(size_t)((w << 6) + __builtin_ctzll(gb)) * W + bw] ^= bb;
                }
            }
    }
    CliffordTableau U(n);
    for (int a = 0; a < n; ++a) {
        U.Zrow[(size_t)a] = gens[(size_t)a];
        Pauli& X = U.Xrow[(size_t)a];
        const uint64_t* Da = &D[(size_t)a * 2 * W];
        int ny = 0;
        for (int w = 0; w < W; ++w) {
            X.x[(size_t)w] = Da[w];
            X.z[(size_t)w] = Da[W + w];
            ny += __builtin_popcountll(Da[w] & Da[W + w]);
        }
        X.phase = ny & 3;                   // i^{#Y}·X^x·Z^z is Hermitian (+P gauge choice)
    }
    U.reset_dual();                          // restore the valid-dual postcondition
    return U;
}

// ---------------------------------------------------------------------------
// lean L6: batched left-multiplication (CanonicalStabSum::flush_gates' replay).
// ---------------------------------------------------------------------------
void CliffordTableau::left_rowwise(const std::vector<LeftGate>& gates) {
    for (const LeftGate& g : gates) {
        switch (g.kind) {
            case 0: left_h(g.a);        break;
            case 1: left_s(g.a);        break;
            case 2: left_sdg(g.a);      break;
            case 3: left_x(g.a);        break;
            case 4: left_y(g.a);        break;
            case 5: left_z(g.a);        break;
            case 6: left_cx(g.a, g.b);  break;
            default: left_cz(g.a, g.b); break;
        }
    }
}

void CliffordTableau::left_batch(const std::vector<LeftGate>& gates) {
    // A short list is cheaper row by row (two O(n²/64)-bit transposes would dominate); the result is the same.
    if (gates.size() < 32) { left_rowwise(gates); return; }
    const bool had_dual = dual_valid_;
    const int W = (n + 63) / 64;
    const int R = 2 * n;                      // row r < n: Xrow[r]; r >= n: Zrow[r - n]
    const int RW = (R + 63) / 64;
    std::vector<uint64_t> X((size_t)n * RW, 0), Z((size_t)n * RW, 0), P0((size_t)RW, 0), P1((size_t)RW, 0);
    auto row_of = [&](int r) -> Pauli& { return r < n ? Xrow[(size_t)r] : Zrow[(size_t)(r - n)]; };
    for (int r = 0; r < R; ++r) {
        const Pauli& M = row_of(r);
        const uint64_t rb = 1ull << (r & 63);
        const size_t rw = (size_t)(r >> 6);
        for (int w = 0; w < W; ++w) {
            for (uint64_t bits = M.x[(size_t)w]; bits; bits &= bits - 1) X[(size_t)((w << 6) + __builtin_ctzll(bits)) * RW + rw] |= rb;
            for (uint64_t bits = M.z[(size_t)w]; bits; bits &= bits - 1) Z[(size_t)((w << 6) + __builtin_ctzll(bits)) * RW + rw] |= rb;
        }
        if (M.phase & 1) P0[rw] |= rb;
        if (M.phase & 2) P1[rw] |= rb;
    }
    // The conj_* closed forms on the planes (a = x bit, b = z bit at the wire; dp the phase delta mod 4 as two
    // bit-planes: +1 = add with carry, +2 = flip P1, +3 = subtract one with borrow).
    for (const LeftGate& g : gates) {
        uint64_t* Xa = &X[(size_t)g.a * RW];
        uint64_t* Za = &Z[(size_t)g.a * RW];
        switch (g.kind) {
            case 0:                                // H: (a,b) -> (b,a); dp = 2·(a&b)
                for (int w = 0; w < RW; ++w) { P1[(size_t)w] ^= Xa[w] & Za[w]; std::swap(Xa[w], Za[w]); }
                break;
            case 1:                                // S: b ^= a; dp = a
                for (int w = 0; w < RW; ++w) { Za[w] ^= Xa[w]; P1[(size_t)w] ^= P0[(size_t)w] & Xa[w]; P0[(size_t)w] ^= Xa[w]; }
                break;
            case 2:                                // S†: b ^= a; dp = 3a
                for (int w = 0; w < RW; ++w) { Za[w] ^= Xa[w]; P1[(size_t)w] ^= ~P0[(size_t)w] & Xa[w]; P0[(size_t)w] ^= Xa[w]; }
                break;
            case 3:                                // X: dp = 2b
                for (int w = 0; w < RW; ++w) P1[(size_t)w] ^= Za[w];
                break;
            case 4:                                // Y = X then Z: dp = 2b + 2a
                for (int w = 0; w < RW; ++w) P1[(size_t)w] ^= Za[w] ^ Xa[w];
                break;
            case 5:                                // Z: dp = 2a
                for (int w = 0; w < RW; ++w) P1[(size_t)w] ^= Xa[w];
                break;
            case 6: {                              // CX(c,t): xt ^= xc; zc ^= zt
                uint64_t* Xt = &X[(size_t)g.b * RW];
                uint64_t* Zt = &Z[(size_t)g.b * RW];
                for (int w = 0; w < RW; ++w) { Xt[w] ^= Xa[w]; Za[w] ^= Zt[w]; }
                break;
            }
            default: {                             // CZ(c,t): zt ^= xc; zc ^= xt; dp = 2·(xc & xt)
                uint64_t* Xt = &X[(size_t)g.b * RW];
                uint64_t* Zt = &Z[(size_t)g.b * RW];
                for (int w = 0; w < RW; ++w) { P1[(size_t)w] ^= Xa[w] & Xt[w]; Zt[w] ^= Xa[w]; Za[w] ^= Xt[w]; }
                break;
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        Pauli& M = row_of(r);
        std::fill(M.x.begin(), M.x.end(), 0ull);
        std::fill(M.z.begin(), M.z.end(), 0ull);
        M.phase = (int)((P0[(size_t)(r >> 6)] >> (r & 63)) & 1ull) | (int)(((P1[(size_t)(r >> 6)] >> (r & 63)) & 1ull) << 1);
    }
    for (int q = 0; q < n; ++q) {
        const uint64_t qb = 1ull << (q & 63);
        const size_t qw = (size_t)(q >> 6);
        const uint64_t* Xq = &X[(size_t)q * RW];
        const uint64_t* Zq = &Z[(size_t)q * RW];
        for (int w = 0; w < RW; ++w) {
            for (uint64_t bits = Xq[w]; bits; bits &= bits - 1) row_of((w << 6) + __builtin_ctzll(bits)).x[qw] |= qb;
            for (uint64_t bits = Zq[w]; bits; bits &= bits - 1) row_of((w << 6) + __builtin_ctzll(bits)).z[qw] |= qb;
        }
    }
    if (had_dual) reset_dual();               // the per-gate patches' result, rebuilt once (unique rows)
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
