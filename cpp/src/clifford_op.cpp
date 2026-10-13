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

void DiagPauliClifford::apply(AffineState& s) const {
    DiagApplyWorkspace ws;
    apply(s, ws);
}

// k ≤ K_BITMASK_MAX ⇒ a qubit's k-bit support fits in one uint64_t word (R.row(j)[0]).
// k ≤ K_PACKED_MAX  ⇒ the k×k GF(2) matrix M fits in one uint64_t (k²≤64), so M is bit-packed.
// These thresholds are shared by apply() and the templated apply_impl<K> fast paths.
static constexpr int K_BITMASK_MAX = 64;
static constexpr int K_PACKED_MAX = 8;

// Dense-B quadratic layer via the RᵀBR contraction. Computes the SAME contributions
// as the per-pair `for (j,l) in bpairs` loop — linear D updates (into lin[]), the
// off-diagonal J flips (emitted through toggle(c,d), c<d), and the cst term — but in
// O(k·n²/64) B-passes instead of O(|bpairs|·k²) pair iterations. Used only when B is
// dense (gated at the call sites), so the sparse path is untouched. k ≤ 64 (one word
// per R-row). Byte-identical to the per-pair loop (verified vs a 200k-trial reference):
//   M_{cd}   = parity(R_c · B R_d)                 (off-diagonal, c<d)
//   lin[c]  += 2·(#edges of B inside supp(R_c) mod 2)   (p_j p_l diagonal)
//   lin[d]  += 2·parity(R_d · B b)                  (b_j p_l + b_l p_j cross terms)
//   cst     += 2·(#edges of B inside supp(b))       (b_j b_l constant)
// All quadratic-layer contributions carry the leading factor 2, so everything reduces
// to GF(2) bilinear forms — except the two edge counts, whose parity needs the integer
// degree sum Σ popcount(B_i & v) = 2·#edges.
template <class Toggle>
static void dense_quadratic_layer(const SymPackedMat& B, const PackedMat& R,
                                  const std::vector<uint8_t>& bvec, int n, int k,
                                  long long* lin, long long& cst, Toggle&& toggle) {
    const int nw = (n + 63) / 64;
    // Per-thread scratch: k column-vectors R_c | k matvecs B·R_c | B·b | packed b.
    thread_local std::vector<uint64_t> scratch;
    scratch.assign((size_t)(2 * k + 2) * nw, 0);
    uint64_t* Rc = scratch.data();
    uint64_t* BRc = Rc + (size_t)k * nw;
    uint64_t* Bb = BRc + (size_t)k * nw;
    uint64_t* bp = Bb + nw;
    auto col = [&](int c) -> uint64_t* { return Rc + (size_t)c * nw; };
    auto brc = [&](int c) -> uint64_t* { return BRc + (size_t)c * nw; };

    // Transpose R into k packed column-vectors, and pack b. (k ≤ 64 ⇒ one R word.)
    for (int j = 0; j < n; ++j) {
        if (bvec[j]) bp[j >> 6] |= (1ULL << (j & 63));
        uint64_t rj = R.row(j)[0];
        while (rj) { int c = __builtin_ctzll(rj); col(c)[j >> 6] |= (1ULL << (j & 63)); rj &= rj - 1; }
    }
    // GF(2) matvecs: BRc[c] = B·R_c, Bb = B·b.
    auto matvec = [&](const uint64_t* v, uint64_t* out) {
        for (int i = 0; i < n; ++i) {
            const uint64_t* Bi = B.row(i);
            uint64_t acc = 0;
            for (int w = 0; w < nw; ++w) acc ^= Bi[w] & v[w];
            if (__builtin_parityll(acc)) out[i >> 6] |= (1ULL << (i & 63));
        }
    };
    for (int c = 0; c < k; ++c) matvec(col(c), brc(c));
    matvec(bp, Bb);
    auto dotpar = [&](const uint64_t* a, const uint64_t* d) {
        uint64_t acc = 0;
        for (int w = 0; w < nw; ++w) acc ^= a[w] & d[w];
        return (int)__builtin_parityll(acc);
    };
    // Off-diagonal J: M_{cd} = parity(R_c · B R_d).
    for (int c = 0; c < k; ++c)
        for (int d = c + 1; d < k; ++d)
            if (dotpar(col(c), brc(d))) toggle(c, d);
    // b-cross linear: lin[d] += 2·parity(R_d · B b).
    for (int d = 0; d < k; ++d)
        if (dotpar(col(d), Bb)) lin[d] += 2;
    // Edge-count parity over supp(v): #edges = (Σ_{i∈supp(v)} popcount(B_i & v)) / 2.
    auto edges_odd = [&](const uint64_t* v) -> bool {
        long long S = 0;
        for (int i = 0; i < n; ++i)
            if ((v[i >> 6] >> (i & 63)) & 1ULL) {
                const uint64_t* Bi = B.row(i);
                for (int w = 0; w < nw; ++w) S += __builtin_popcountll(Bi[w] & v[w]);
            }
        return ((S / 2) & 1) != 0;
    };
    // Diagonal linear: lin[c] += 2·(#edges in supp(R_c) mod 2).
    for (int c = 0; c < k; ++c)
        if (edges_odd(col(c))) lin[c] += 2;
    // Constant: cst += 2·#edges in supp(b)  (== Σ_{i∈supp(b)} popcount(B_i & b)).
    {
        long long S = 0;
        for (int i = 0; i < n; ++i)
            if ((bp[i >> 6] >> (i & 63)) & 1ULL) {
                const uint64_t* Bi = B.row(i);
                for (int w = 0; w < nw; ++w) S += __builtin_popcountll(Bi[w] & bp[w]);
            }
        cst += S;   // S = 2·#edges = Σ 2·b_j b_l over pairs (matches the per-pair cst)
    }
}

// Density gate: use the RᵀBR contraction only when B is dense enough to win. The
// contraction cost is ~npairs-independent (≈(k+1) packed B-matvecs + (k+1) edge
// passes ≈ 2(k+1)·n·⌈n/64⌉ word-ops with popcount), while the per-pair loop costs
// ≈ npairs·(~2ns). Empirically (n=128) the crossover is ~1.1·(k+1)·n·⌈n/64⌉ pairs;
// we gate at 2× that so the sparse hot path is never pulled into the dense branch
// (verified: sparse B unchanged, dense B 3.6–5.3× faster).
static inline bool dense_B_worthwhile(size_t npairs, int n, int k) {
    const long long nw = (n + 63) / 64;
    return (long long)npairs > 2 * (long long)(k + 1) * n * nw;
}

// Runtime k-space hot path: 4 ≤ k ≤ 64.  Identical algorithm to the old apply() body.
void DiagPauliClifford::apply_kspace(AffineState& s, DiagApplyWorkspace& ws,
                                     const std::vector<int>& actives, const PairList& bpairs) const {
    const int k = s.k_;
    long long cst = 0;
    ws.lin.assign(k, 0); long long* lin = ws.lin.data();
    // M: k×k GF(2), packed bit (c*k+d) into one uint64_t when k≤8, else a k*k byte buffer.
    const bool m_packed = (k <= K_PACKED_MAX);
    uint64_t mword = 0;
    uint8_t* mbuf = nullptr;
    if (!m_packed) { ws.qpair.assign((size_t)k * k, 0); mbuf = ws.qpair.data(); }
    auto m_toggle = [&](int c, int d) {           // c≠d guaranteed by callers
        if (c > d) { int t = c; c = d; d = t; }
        if (m_packed) mword ^= (uint64_t)1 << (c * k + d);
        else mbuf[(size_t)c * k + d] ^= 1;
    };
    auto srow = [&](int j) -> uint64_t { return s.R.row(j)[0]; };
    // Linear layer Σ_j a_j y_j,  y_j = b_j ⊕ p_j,  p_j = ⊕_{c∈supp_j} x_c.
    for (int j : actives) {
        int aj = a[j];
        int bj = s.b[j];
        uint64_t sj = srow(j);
        cst += (long long)aj * bj;
        long long Lcoef = (long long)aj * (1 - 2 * bj);
        for (uint64_t w = sj; w; w &= w - 1) lin[__builtin_ctzll(w)] += Lcoef;
        if (aj & 1) {                              // a-odd S-pairs: c<d both in supp_j
            for (uint64_t wu = sj; wu; wu &= wu - 1) {
                int c = __builtin_ctzll(wu);
                for (uint64_t wd = wu & (wu - 1); wd; wd &= wd - 1)
                    m_toggle(c, __builtin_ctzll(wd));
            }
        }
    }
    // Quadratic layer 2 Σ_{j<l} B_{jl} y_j y_l. Dense B ⇒ RᵀBR contraction (k ≤ 64 ⇒ one
    // R word, helper precondition holds); else the per-pair loop.
    if (dense_B_worthwhile(bpairs.size(), s.n_, k)) {
        dense_quadratic_layer(B, s.R, s.b, s.n_, k, lin, cst, m_toggle);
    } else
    for (auto [j, l] : bpairs) {
        int bj = s.b[j], bl = s.b[l];
        uint64_t sj = srow(j), sl = srow(l);
        cst += 2 * bj * bl;
        if (bj) for (uint64_t w = sl; w; w &= w - 1) lin[__builtin_ctzll(w)] += 2;
        if (bl) for (uint64_t w = sj; w; w &= w - 1) lin[__builtin_ctzll(w)] += 2;
        for (uint64_t w = sj & sl; w; w &= w - 1) lin[__builtin_ctzll(w)] += 2;
        for (uint64_t wc = sj; wc; wc &= wc - 1) {
            int c = __builtin_ctzll(wc);
            for (uint64_t wd = sl; wd; wd &= wd - 1) {
                int d = __builtin_ctzll(wd);
                if (c != d) m_toggle(c, d);
            }
        }
    }
    for (int c = 0; c < k; ++c) s.D[c] = (int)(((s.D[c] + lin[c]) % 4 + 4) % 4);
    if (m_packed) {
        for (int c = 0; c < k; ++c) for (int d = c + 1; d < k; ++d)
            if ((mword >> (c * k + d)) & 1) s.J.flip_sym(c, d);
    } else {
        for (int c = 0; c < k; ++c) for (int d = c + 1; d < k; ++d)
            if (mbuf[(size_t)c * k + d]) s.J.flip_sym(c, d);
    }
    s.omega.add_z8((int)(((2 * cst) % 8 + 8) % 8));
    s.omega = s.omega.mul(gamma);
    for (int j = 0; j < s.n_; ++j) if (v[j]) s.b[j] ^= 1;
}

// Compile-time-K specialization: same algorithm, K fixed so loops unroll, M is always the packed
// uint64_t form (K≤3≤K_PACKED_MAX=8), and there is no runtime m_packed branch.
// K=0: no free variables → Dlin/M are conceptually empty; use a size-1 guard array but never
// index it (all support bitmasks are 0 since R has no columns).
template<int K>
void DiagPauliClifford::apply_impl(AffineState& s, DiagApplyWorkspace& ws,
                                   const std::vector<int>& actives, const PairList& bpairs) const {
    static_assert(K >= 0 && K <= 3, "apply_impl<K> only for K=0..3");
    (void)ws;  // stack-allocated Dlin; ws kept for API uniformity
    long long cst = 0;
    // Use a fixed-size stack array; for K=0 we guard the size at 1 to avoid ill-formed
    // zero-length arrays, but we never index it (all support masks are 0 when k=0).
    long long Dlin[K ? K : 1] = {};
    uint64_t mword = 0;  // packed K×K GF(2) upper-triangle; always fits (K≤3 ⇒ max bit c*K+d = 5)
    auto m_toggle = [&](int c, int d) {            // c≠d guaranteed by callers
        if (c > d) { int t = c; c = d; d = t; }
        mword ^= (uint64_t)1 << (c * K + d);
    };
    // For K=0 R has no columns; srow returns 0 so all inner loops are empty.
    auto srow = [&](int j) -> uint64_t { return K ? s.R.row(j)[0] : 0ULL; };
    // Linear layer
    for (int j : actives) {
        int aj = a[j];
        int bj = s.b[j];
        uint64_t sj = srow(j);
        cst += (long long)aj * bj;
        long long Lcoef = (long long)aj * (1 - 2 * bj);
        for (uint64_t w = sj; w; w &= w - 1) Dlin[__builtin_ctzll(w)] += Lcoef;
        if (aj & 1) {
            for (uint64_t wu = sj; wu; wu &= wu - 1) {
                int c = __builtin_ctzll(wu);
                for (uint64_t wd = wu & (wu - 1); wd; wd &= wd - 1)
                    m_toggle(c, __builtin_ctzll(wd));
            }
        }
    }
    // Quadratic layer 2 Σ_{j<l} B_{jl} y_j y_l. Dense B ⇒ RᵀBR contraction; else per-pair.
    if (K > 0 && dense_B_worthwhile(bpairs.size(), s.n_, K)) {
        dense_quadratic_layer(B, s.R, s.b, s.n_, K, Dlin, cst, m_toggle);
    } else
    for (auto [j, l] : bpairs) {
        int bj = s.b[j], bl = s.b[l];
        uint64_t sj = srow(j), sl = srow(l);
        cst += 2 * bj * bl;
        if (bj) for (uint64_t w = sl; w; w &= w - 1) Dlin[__builtin_ctzll(w)] += 2;
        if (bl) for (uint64_t w = sj; w; w &= w - 1) Dlin[__builtin_ctzll(w)] += 2;
        for (uint64_t w = sj & sl; w; w &= w - 1) Dlin[__builtin_ctzll(w)] += 2;
        for (uint64_t wc = sj; wc; wc &= wc - 1) {
            int c = __builtin_ctzll(wc);
            for (uint64_t wd = sl; wd; wd &= wd - 1) {
                int d = __builtin_ctzll(wd);
                if (c != d) m_toggle(c, d);
            }
        }
    }
    // Fold Dlin into s.D
    for (int c = 0; c < K; ++c) s.D[c] = (int)(((s.D[c] + Dlin[c]) % 4 + 4) % 4);
    // Fold mword into s.J
    for (int c = 0; c < K; ++c) for (int d = c + 1; d < K; ++d)
        if ((mword >> (c * K + d)) & 1) s.J.flip_sym(c, d);
    s.omega.add_z8((int)(((2 * cst) % 8 + 8) % 8));
    s.omega = s.omega.mul(gamma);
    for (int j = 0; j < s.n_; ++j) if (v[j]) s.b[j] ^= 1;
}

void DiagPauliClifford::apply(AffineState& s, DiagApplyWorkspace& ws) const {
    // Source the operator-only enumeration lists: from the immutable cache if finalized,
    // else built once into ws buffers (retaining capacity across calls).
    const std::vector<int>* actives;
    const PairList* bpairs;
    if (cache) {
        actives = &cache->active_a; bpairs = &cache->b_pairs;
    } else {
        ws.active_a.clear();
        for (int j = 0; j < n; ++j) if (a[j]) ws.active_a.push_back(j);
        ws.b_pairs.clear();
        scan_b_pairs(B, n, [&](int j, int l) { ws.b_pairs.emplace_back(j, l); });
        actives = &ws.active_a; bpairs = &ws.b_pairs;
    }
    switch (s.k_) {
        case 0: return apply_impl<0>(s, ws, *actives, *bpairs);
        case 1: return apply_impl<1>(s, ws, *actives, *bpairs);
        case 2: return apply_impl<2>(s, ws, *actives, *bpairs);
        case 3: return apply_impl<3>(s, ws, *actives, *bpairs);
        default:
            if (s.k_ > K_BITMASK_MAX) return apply_generic(s, ws, *actives, *bpairs);
            return apply_kspace(s, ws, *actives, *bpairs);   // 4 ≤ k ≤ 64 runtime-k path
    }
}

void DiagPauliClifford::apply_generic(AffineState& s, DiagApplyWorkspace& ws,
                                      const std::vector<int>& actives, const PairList& bpairs) const {
    const int k = s.k_;
    long long cst = 0;
    ws.lin.assign(k, 0); long long* lin = ws.lin.data();
    ws.qpair.assign((size_t)k * k, 0); uint8_t* qpair = ws.qpair.data();
    auto add_pair = [&](int s_, int t_) {
        if (s_ == t_) return;
        int s2 = s_, t2 = t_;
        if (s2 > t2) std::swap(s2, t2);
        qpair[(size_t)s2 * k + t2] ^= 1;
    };
    // Precompute each qubit's x-support once (supp_j = { l : R(j,l)=1 }); the quadratic layer
    // reuses these across all B pairs. Stored as a flat CSR buffer — sup_idx[sup_off[j]..
    // sup_off[j+1]) — so the whole precompute is two allocations regardless of n, not one
    // heap allocation per qubit (which dominated apply for small k).
    const int nn = s.n_;
    ws.sup_off.assign(nn + 1, 0); int* sup_off = ws.sup_off.data();
    for (int j = 0; j < nn; ++j) {
        int cnt = 0;
        for (int l = 0; l < k; ++l) if (s.R.get(j, l)) ++cnt;
        sup_off[j + 1] = sup_off[j] + cnt;
    }
    ws.sup_idx.resize(sup_off[nn]); int* sup_idx = ws.sup_idx.data();
    for (int j = 0; j < nn; ++j) {
        int p = sup_off[j];
        for (int l = 0; l < k; ++l) if (s.R.get(j, l)) sup_idx[p++] = l;
    }
    // .data() + offset (never &sup_idx[end]) keeps the one-past-end pointer well-defined for
    // empty supports.
    const int* sidx = sup_idx;
    // Linear layer Σ_j a_j y_j,  y_j = b_j ⊕ p_j,  p_j = ⊕_{l∈supp_j} x_l.
    for (int j : actives) {
        int aj = a[j];
        int bj = s.b[j];
        const int* sj = sidx + sup_off[j]; int sjl = sup_off[j + 1] - sup_off[j];
        cst += (long long)aj * bj;
        long long Lcoef = (long long)aj * (1 - 2 * bj);
        for (int u = 0; u < sjl; ++u) lin[sj[u]] += Lcoef;
        if (aj & 1) for (int u = 0; u < sjl; ++u)
            for (int w = u + 1; w < sjl; ++w) add_pair(sj[u], sj[w]);
    }
    // Quadratic layer 2 Σ_{j<l} B_{jl} y_j y_l ; y_j y_l = b_jb_l ⊕ b_jp_l ⊕ b_lp_j ⊕ p_jp_l.
    // Visit only B's nonzero pairs (j<l) from the shared enumeration list.
    for (auto [j, l] : bpairs) {
        int bj = s.b[j], bl = s.b[l];
        const int* sj = sidx + sup_off[j]; int sjl = sup_off[j + 1] - sup_off[j];
        const int* sl = sidx + sup_off[l]; int sll = sup_off[l + 1] - sup_off[l];
        cst += 2 * bj * bl;
        if (bj) for (int u = 0; u < sll; ++u) lin[sl[u]] += 2;
        if (bl) for (int u = 0; u < sjl; ++u) lin[sj[u]] += 2;
        for (int u = 0; u < sjl; ++u) for (int x = 0; x < sll; ++x) {
            int c = sj[u], d = sl[x];
            if (c == d) lin[c] += 2; else add_pair(c, d);
        }
    }
    for (int t = 0; t < k; ++t) s.D[t] = (int)(((s.D[t] + lin[t]) % 4 + 4) % 4);
    for (int si = 0; si < k; ++si) for (int t = si + 1; t < k; ++t)
        if (qpair[(size_t)si * k + t]) s.J.flip_sym(si, t);
    s.omega.add_z8((int)(((2 * cst) % 8 + 8) % 8));
    s.omega = s.omega.mul(gamma);
    for (int j = 0; j < s.n_; ++j) if (v[j]) s.b[j] ^= 1;     // X^v translation (v set from Task 3 on; no-op when v=0)
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
