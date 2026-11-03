#include "qeccore/pauli_diagonalize.hpp"
#include <cstddef>

#include <cassert>
#include <vector>

#include "qeccore/clifford_tableau.hpp"
#include "qeccore/stab_disentangle.hpp"

namespace qeccore {

// =============================================================================
// Simultaneous Clifford diagonalization of a commuting Pauli layer.
//
// The {P_k} mutually commute, so they generate an abelian (isotropic) subgroup.
// We:
//   1. pick a maximal INDEPENDENT commuting subset of {P_k} over GF(2) and EXTEND
//      it to a Lagrangian basis `gens` of n independent, mutually-commuting
//      Paulis;
//   2. obtain a disentangling Clifford G (disentangle_from_generators) with
//      G·gens[a]·G† = ±Z; G maps the whole commuting group into the Z-subgroup.
//      Set C' = G† (cprime = reversed, daggered G), so C'† = G.
//   3. for every original P_k, D_k = G·P_k·G† (conjugate P_k forward through G);
//      D_k is a pure-Z Pauli whose overall ± sign (phase ∈ {0,2}) is folded into
//      the coeff, leaving a SIGNLESS Z-string.
//
// Identity: rot(c,P_k) = G†·rot(c, G P_k G†)·G, hence
//   T = Π rot(c_k,P_k) = G†·(Π rot(c_k, D_k))·G = C'·(Π rot)·C'† .
//
// Materialization contract: C' = cprime.back()·…·cprime[0] (cprime[0] FIRST);
// materialize  C'·(Π_k rot(coeff_k, D_k))·C'†  to reproduce T.  cprime may contain
// H,S,SDG,X,Y,Z,CX,CZ (whatever the disentangler emitted, daggered).
// =============================================================================

namespace {

int sympl(const Pauli& a, const Pauli& b) { return Pauli::anticommute_bit(a, b); }

// Local single-Pauli conjugation rules M <- G M G† (the CliffordTableau::conj_*
// statics are private; these reproduce the same closed forms, see
// clifford_tableau.cpp). Phase convention: Pauli = i^phase X^x Z^z.
void cH(Pauli& M, int q) {  // H X H = Z, H Z H = X; dp = 2(a&b)
    bool a = M.xbit(q), b = M.zbit(q);
    if (a != b) { M.flipx(q); M.flipz(q); }
    if (a && b) M.phase = (M.phase + 2) & 3;
}
void cS(Pauli& M, int q) {  // S X S† = Y = iXZ; dp = a
    if (M.xbit(q)) { M.flipz(q); M.phase = (M.phase + 1) & 3; }
}
void cSdg(Pauli& M, int q) {  // S† X S = -Y; dp = 3a
    if (M.xbit(q)) { M.flipz(q); M.phase = (M.phase + 3) & 3; }
}
void cX(Pauli& M, int q) { if (M.zbit(q)) M.phase = (M.phase + 2) & 3; }  // X Z X = -Z
void cZ(Pauli& M, int q) { if (M.xbit(q)) M.phase = (M.phase + 2) & 3; }  // Z X Z = -X
void cY(Pauli& M, int q) { cX(M, q); cZ(M, q); }                          // Y = i XZ
void cCX(Pauli& M, int c, int t) {  // xt ^= xc, zc ^= zt; dp = 0
    if (M.xbit(c)) M.flipx(t);
    if (M.zbit(t)) M.flipz(c);
}
void cCZ(Pauli& M, int c, int t) {  // zt ^= xc, zc ^= xt; dp = 2(xc&xt)
    bool xc = M.xbit(c), xt = M.xbit(t);
    if (xc) M.flipz(t);
    if (xt) M.flipz(c);
    if (xc && xt) M.phase = (M.phase + 2) & 3;
}

// Conjugate M <- g M g† for a DisentangleGate op (0:H 1:S 2:SDG 3:X 4:Y 5:Z 6:CX 7:CZ).
void conj_op(Pauli& M, int op, int a, int b) {
    switch (op) {
        case 0: cH(M, a); break;
        case 1: cS(M, a); break;
        case 2: cSdg(M, a); break;
        case 3: cX(M, a); break;
        case 4: cY(M, a); break;
        case 5: cZ(M, a); break;
        case 6: cCX(M, a, b); break;
        case 7: cCZ(M, a, b); break;
        default: break;
    }
}

// A 2n-bit GF(2) vector packed as (x-words || z-words), for independence tests.
struct BitVec {
    std::vector<uint64_t> w;
    int W;
    explicit BitVec(int n) : w((size_t)2 * ((n + 63) / 64), 0), W((n + 63) / 64) {}
    static BitVec of(const Pauli& p, int n) {
        BitVec v(n);
        for (int i = 0; i < v.W; ++i) { v.w[i] = p.x[i]; v.w[v.W + i] = p.z[i]; }
        return v;
    }
    int first_set() const {
        for (size_t i = 0; i < w.size(); ++i)
            if (w[i]) return (int)(i * 64) + __builtin_ctzll(w[i]);
        return -1;
    }
    void xor_with(const BitVec& o) { for (size_t i = 0; i < w.size(); ++i) w[i] ^= o.w[i]; }
    bool is_zero() const { for (auto x : w) if (x) return false; return true; }
    bool bit(int b) const { return (w[b >> 6] >> (b & 63)) & 1ull; }
    void setbit(int b) { w[b >> 6] |= (1ull << (b & 63)); }
    // x-half is words [0,W), z-half is words [W,2W).
    Pauli to_pauli(int n) const {
        Pauli p(n);
        for (int i = 0; i < W; ++i) { p.x[i] = w[i]; p.z[i] = w[W + i]; }
        return p;  // phase 0; a signless commuting extender for the Lagrangian
    }
    // Symplectic dual: swap the x and z halves. <u,v>_sympl == dot(u, dual(v)).
    BitVec sympl_dual(int n) const {
        BitVec d(n);
        for (int i = 0; i < W; ++i) { d.w[i] = w[W + i]; d.w[W + i] = w[i]; }
        return d;
    }
};

struct GF2Basis {
    std::vector<BitVec> rows;
    std::vector<int> lead;
    bool add(BitVec v) {
        for (size_t i = 0; i < rows.size(); ++i) if (v.bit(lead[i])) v.xor_with(rows[i]);
        int p = v.first_set();
        if (p < 0) return false;
        for (auto& r : rows) if (r.bit(p)) r.xor_with(v);
        rows.push_back(v); lead.push_back(p);
        return true;
    }
    bool independent_of(BitVec v) const {
        for (size_t i = 0; i < rows.size(); ++i) if (v.bit(lead[i])) v.xor_with(rows[i]);
        return !v.is_zero();
    }
};

}  // namespace

DiagResult diagonalize_commuting_layer(
    const std::vector<PauliRotationForm::Entry>& terms, int n) {
    const int m = (int)terms.size();

    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j)
            assert(sympl(terms[i].pauli, terms[j].pauli) == 0 &&
                   "diagonalize_commuting_layer: input Paulis must mutually commute");

    DiagResult res;

    // ---------- Step 1a: maximal independent commuting subset ----------
    std::vector<Pauli> gens;
    GF2Basis basis;
    for (int k = 0; k < m; ++k)
        if (basis.add(BitVec::of(terms[k].pauli, n))) gens.push_back(terms[k].pauli);

    // ---------- Step 1b: extend to a Lagrangian of size n ----------
    // gens spans an isotropic subspace V of the 2n-dim symplectic GF(2) space.
    // We must extend it to a Lagrangian (maximal isotropic, dim n). A vector u may
    // join iff (i) it commutes with ALL current gens (lies in V^⊥) and (ii) it is
    // independent of span(gens). Since V ⊆ V^⊥ (isotropic) and dim V^⊥ = 2n - r ≥ r
    // (r = |gens| ≤ n), such a u ALWAYS exists until |gens| = n.
    //
    // The old code tried only a fixed ad-hoc candidate family (single-qubit Z/X/Y
    // and Z_aX_b). That family does NOT span the symplectic space, so for some
    // isotropic V every candidate either anticommuted with V or was dependent —
    // leaving gens short, which (a) tripped the size==n assert under -DNDEBUG-off
    // and (b) fed a malformed (incomplete) generator list to the disentangler,
    // surfacing as "row reduction failed" / heap issues. Fix: compute V^⊥ exactly
    // as the GF(2) null space of the symplectic-pairing constraints, then pull
    // independent basis vectors out of it.

    // V^⊥ = { u : dot(u, sympl_dual(g_i)) == 0 ∀ i }. Build the RREF basis of
    // span{sympl_dual(g_i)} =: D, then generate V^⊥ explicitly as the GF(2) null
    // space of D (orthogonal complement). For each non-pivot ("free") column f, a
    // basis vector of D^⊥ is e_f plus, for every pivot row whose leading column is
    // p and which has a 1 in column f, the bit e_p — i.e. set f, then cancel.
    GF2Basis dualspan;
    for (const auto& g : gens) dualspan.add(BitVec::of(g, n).sympl_dual(n));
    const int W = BitVec(n).W;
    const int DIM = 2 * W * 64;  // padded column count (== words*64)

    std::vector<char> is_pivot(DIM, 0);
    for (int p : dualspan.lead) is_pivot[p] = 1;
    // Only x-columns [0,n) and z-columns [W*64, W*64+n) are REAL Pauli bits; the
    // rest are word-padding and must never seed an extender.
    auto is_real_col = [&](int f) {
        return (f >= 0 && f < n) || (f >= W * 64 && f < W * 64 + n);
    };

    auto try_add_bv = [&](BitVec u) -> bool {
        if ((int)gens.size() >= n) return false;
        if (!basis.independent_of(u)) return false;  // independent_of copies u
        basis.add(u);
        gens.push_back(u.to_pauli(n));
        return true;
    };
    // Each free column yields one null-space (V^⊥) basis vector. These span V^⊥ ⊇ V,
    // so iterating them always finds the n - r independent extenders we need.
    for (int f = 0; f < DIM && (int)gens.size() < n; ++f) {
        if (is_pivot[f] || !is_real_col(f)) continue;
        BitVec u(n);
        u.setbit(f);
        for (size_t i = 0; i < dualspan.rows.size(); ++i)
            if (dualspan.rows[i].bit(f)) u.setbit(dualspan.lead[i]);
        // u is in V^⊥ by construction; only genuinely-new vectors are accepted.
        try_add_bv(u);
    }
    assert((int)gens.size() == n &&
           "diagonalize_commuting_layer: failed to complete Lagrangian basis");

    // ---------- Step 2: get a disentangling Clifford G with G gens[a] G† = ±Z ----------
    // disentangle_from_generators(rows) returns gates G (op-coded 0:H 1:S 2:SDG 3:X
    // 4:Y 5:Z 6:CX 7:CZ) with G|stab> = |0…0>, i.e. G·gens[a]·G† is a single-qubit
    // ±Z (after a permutation/GF(2) mixing of the generators).  G therefore maps the
    // WHOLE commuting group into the Z-subgroup: G·P_k·G† is pure Z for every P_k.
    //
    // We need C' with C'†·P_k·C' = D_k.  Set C' = G†  ⇒  C'† = G  ⇒  D_k = G·P_k·G†,
    // and the identity rot(c,P_k) = G†·rot(c, G P_k G†)·G gives
    //     T = Π rot(c_k,P_k) = G†·(Π rot(c_k, D_k))·G = C'·(Π rot)·C'† .  ✓
    std::vector<DisentangleGate> G = disentangle_from_generators(gens);

    // cprime = C' = G† : reverse G and dagger each gate.  Op-dagger: H,X,Y,Z,CX,CZ
    // self-inverse; S<->SDG.
    auto op_to_cliffgate = [](int op, int a, int b) -> CliffGate {
        switch (op) {
            case 0: return {GateKind::H, {a}};
            case 1: return {GateKind::SDG, {a}};  // S†
            case 2: return {GateKind::S, {a}};    // (SDG)† = S
            case 3: return {GateKind::X, {a}};
            case 4: return {GateKind::Y, {a}};
            case 5: return {GateKind::Z, {a}};
            case 6: return {GateKind::CX, {a, b}};
            case 7: return {GateKind::CZ, {a, b}};
        }
        return {GateKind::X, {a}};
    };
    res.cprime.clear();
    for (auto it = G.rbegin(); it != G.rend(); ++it)
        res.cprime.push_back(op_to_cliffgate(it->op, it->a, it->b));

    // ---------- Step 3: D_k = G·P_k·G†; fold sign into coeff ----------
    res.tz.clear();
    for (int k = 0; k < m; ++k) {
        Pauli D = terms[k].pauli;
        for (const auto& g : G) conj_op(D, g.op, g.a, g.b);  // forward through G
        for (int q = 0; q < n; ++q)
            assert(!D.xbit(q) && "diagonalization left residual X");
        int coeff = terms[k].coeff;
        if (D.phase == 2) { coeff = ((-coeff) % 16 + 16) % 16; D.phase = 0; }
        assert(D.phase == 0 && "pure-Z image must be signless after sign fold");
        res.tz.push_back({D, coeff});
    }

    return res;
}

}  // namespace qeccore
