#include "qeccore/clifford_op.hpp"
#include <cstddef>
#include <algorithm>

namespace qeccore {

SymPackedMat sym_zeros(int n) {
    // Lazy all-zero form (see SymPackedMat): dimension n, no row storage. Reads see zeros;
    // the first flip_sym materializes. Keeps identity()/then()/atom seeds O(n) instead of
    // O(n²) on circuits whose residuals never grow a CZ layer (any Clifford bulk).
    SymPackedMat B;
    B.k_ = n;
    return B;
}

void sym_xor_into(SymPackedMat& dst, const SymPackedMat& src) {
    // dst and src must have the same dimension. Their internal word-capacity (words_)
    // may differ if they were built via different append/drop paths; XOR only the common
    // words — bits beyond dim() are kept zero for a valid dim-k matrix, so this is exact.
    if (src.words() == 0) return;                     // src ≡ 0: dst unchanged
    if (dst.words() == 0) { dst = src; return; }      // dst ≡ 0: dst becomes a copy of src
    int k = dst.dim();
    int w = std::min(dst.words(), src.words());
    for (int i = 0; i < k; ++i) {
        uint64_t* d = dst.row(i);
        const uint64_t* s = src.row(i);
        for (int t = 0; t < w; ++t) d[t] ^= s[t];
    }
}

std::vector<uint8_t> sym_matvec(const SymPackedMat& B, const std::vector<uint8_t>& v) {
    int k = B.dim();
    std::vector<uint8_t> out(k, 0);
    // out[i] = XOR_{j: v[j]} B[i][j] = XOR_{j: v[j]} B[j][i]  (B symmetric). Accumulate the
    // word-packed XOR of rows j over the SUPPORT of v only (v is sparse — a handful of fired
    // error qubits — so this skips the all-decoupled columns that contribute identity), then
    // unpack the bits. Byte-identical to the dense double loop: same GF(2) parity per row i.
    const int w = B.words();
    static thread_local std::vector<uint64_t> acc;
    acc.assign(w, 0);
    bool any = false;
    for (int j = 0; j < k; ++j) if (v[j]) {
        const uint64_t* rj = B.row(j);            // row j == column j (symmetric)
        for (int t = 0; t < w; ++t) acc[t] ^= rj[t];
        any = true;
    }
    if (!any) return out;                          // v == 0 ⇒ all-zero output
    for (int t = 0; t < w; ++t) {
        uint64_t bits = acc[t];
        int base = t << 6;
        while (bits) {
            int i = base + __builtin_ctzll(bits);
            bits &= bits - 1;
            if (i < k) out[i] = 1;
        }
    }
    return out;
}

DiagPauliClifford DiagPauliClifford::identity(int n) {
    DiagPauliClifford d;
    d.n = n; d.gamma = ExactPhase::one();
    d.a.assign(n, 0); d.v.assign(n, 0); d.B = sym_zeros(n);
    return d;
}

DiagPauliClifford DiagPauliClifford::S(int n, int q)   { auto d = identity(n); d.a[q] = 1; return d; }
DiagPauliClifford DiagPauliClifford::Sdg(int n, int q) { auto d = identity(n); d.a[q] = 3; return d; }
DiagPauliClifford DiagPauliClifford::Z(int n, int q)   { auto d = identity(n); d.a[q] = 2; return d; }
DiagPauliClifford DiagPauliClifford::X(int n, int q)   { auto d = identity(n); d.v[q] = 1; return d; }
DiagPauliClifford DiagPauliClifford::Y(int n, int q) {
    auto d = identity(n); d.v[q] = 1; d.a[q] = 2; d.gamma = ExactPhase::zeta8(2); return d;  // Y=iXZ
}
DiagPauliClifford DiagPauliClifford::CZ(int n, int c, int t) {
    auto d = identity(n); d.B.flip_sym(c, t); return d;
}

// Bit-scan B's packed rows, emitting each unordered pair (j,l) with j<l and B_{jl}=1 exactly
// once. Shared by finalize() and the unfinalized ws build so both produce an identical pair list.
template <class Push>
static void scan_b_pairs(const SymPackedMat& B, int n, Push&& push) {
    const int Bwords = B.words();
    for (int j = 0; j < n; ++j) {
        const uint64_t* Brow = B.row(j);
        for (int wi = 0; wi < Bwords; ++wi) {
            uint64_t word = Brow[wi];
            while (word) {
                int l = wi * 64 + __builtin_ctzll(word);
                word &= word - 1;
                if (l <= j) continue;          // each unordered pair once (l > j)
                push(j, l);
            }
        }
    }
}

void DiagPauliClifford::finalize() {
    auto c = std::make_shared<Cache>();
    c->active_a.clear();
    for (int j = 0; j < n; ++j) if (a[j]) c->active_a.push_back(j);
    c->b_pairs.clear();
    scan_b_pairs(B, n, [&](int j, int l) { c->b_pairs.emplace_back(j, l); });
    cache = std::move(c);
}

DiagPauliClifford DiagPauliClifford::then(const DiagPauliClifford& next) const {
    const DiagPauliClifford& U1 = *this;   // applied first
    const DiagPauliClifford& U2 = next;
    DiagPauliClifford r = identity(n);
    for (int j = 0; j < n; ++j) r.v[j] = U1.v[j] ^ U2.v[j];
    r.B = U1.B; sym_xor_into(r.B, U2.B);
    std::vector<uint8_t> b2v1 = sym_matvec(U2.B, U1.v);
    // cst is the y-independent ℤ₄ constant of q2(y⊕v1), folded into gamma below as ζ8^{2·cst}.
    // supp(U1.v) collected once: v1 is sparse (the accumulated X-translation of a few fired error
    // qubits), and BOTH the linear cst term (a2_j·v1_j) and the quadratic cst term (over pairs
    // j<l with v1_j=v1_l=1) are nonzero only on this support — so they restrict to it exactly.
    static thread_local std::vector<int> v1supp;
    v1supp.clear();
    long long cst_lin = 0, cst_quad = 0;
    for (int j = 0; j < n; ++j) {
        long long aj = (long long)U1.a[j]
                     + (long long)U2.a[j] * (1 - 2 * (int)U1.v[j])
                     + 2 * (long long)b2v1[j];
        r.a[j] = (uint8_t)(((aj % 4) + 4) % 4);
        if (U1.v[j]) { cst_lin += (long long)U2.a[j]; v1supp.push_back(j); }   // linear: Σ a2_j·v1_j
    }
    // Quadratic: the term 2·B2_{jl}·y_j·y_l (note the leading 2) leaves shift-constant
    // 2·B2_{jl}·v1_j·v1_l, so each active pair (both in supp(v1)) adds 2 (not 1) to cst.
    // Restricted to ordered pairs within supp(v1) — identical pair set, same +2 per active pair.
    for (size_t a = 0; a < v1supp.size(); ++a)
        for (size_t b = a + 1; b < v1supp.size(); ++b)
            if (U2.B.get(v1supp[a], v1supp[b])) cst_quad += 2;
    long long cst = cst_lin + cst_quad;
    r.gamma = U1.gamma.mul(U2.gamma);
    r.gamma.add_z8((int)(((2 * cst) % 8 + 8) % 8));
    return r;
}


}  // namespace qeccore
