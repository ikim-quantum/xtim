// record_table — S2b: the per-plan record table in the record-ordered canonical basis.
// See qeccore/record_table.hpp for the contract and the normative conventions.
#include "qeccore/record_table.hpp"
#include "record_table_internal.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace qeccore {

// ── Compact free-register operators ───────────────────────────────────────────────────────────
bool fd_hermitian(const FreeDir& d) {
    int ov = 0;
    for (size_t i = 0; i < d.a.size(); ++i) ov += (d.a[i] & d.b[i]) ? 1 : 0;
    return ((d.phase - ov) & 1) == 0;
}

FreeDir fd_identity(int k) {
    FreeDir d;
    d.a.assign((size_t)k, 0);
    d.b.assign((size_t)k, 0);
    d.phase = 0;
    d.par = 0;
    d.hermitian = 1;
    return d;
}

FreeDir fd_mul(const FreeDir& p, const FreeDir& q) {
    if (p.a.size() != q.a.size())
        throw std::logic_error("record_table: fd_mul of directions on different registers (engine bug)");
    FreeDir r;
    const size_t k = p.a.size();
    r.a.assign(k, 0);
    r.b.assign(k, 0);
    int cross = 0;                                   // |p.b ∧ q.a|: moving q's X past p's Z
    for (size_t i = 0; i < k; ++i) {
        r.a[i] = (uint8_t)(p.a[i] ^ q.a[i]);
        r.b[i] = (uint8_t)(p.b[i] ^ q.b[i]);
        cross += (p.b[i] & q.a[i]) ? 1 : 0;
    }
    r.phase = (uint8_t)((p.phase + q.phase + 2 * (cross & 1)) & 3);
    r.par = (uint8_t)((p.par ^ q.par) & 1);
    r.hermitian = fd_hermitian(r) ? 1 : 0;
    return r;
}

void fd_require_hermitian(const FreeDir& d, const char* what) {
    if (!fd_hermitian(d))
        throw std::logic_error(std::string("record_table: non-Hermitian direction (") + what +
                               "): a product of commuting Hermitian record operators must be "
                               "Hermitian (engine bug)");
}

// ── Pullback and plane ────────────────────────────────────────────────────────────────────────
Pauli diag_pullback(const DiagNormalForm& nf, const Pauli& M) {
    // C0† M C0 for C0 = S^{a&1}·CZ (the same h-free conjugate classify_observable uses): the X part is
    // unchanged; Z picks up the dressing v (a-part on X∧a, the CZ partner of each X endpoint); the
    // phase picks up −|x∧a| − 2·#{cz pairs with X on both ends} (mod 4).
    const int n = M.n;
    const int NW = (n + 63) / 64;
    Pauli h = M;
    int lin = 0;
    for (int q = 0; q < n && q < (int)nf.a.size(); ++q)
        if ((nf.a[(size_t)q] & 1) && M.xbit(q)) { h.flipz(q); ++lin; }
    int quad = 0;
    for (const auto& e : nf.cz) {
        const bool xf = M.xbit(e.first), xs = M.xbit(e.second);
        if (xs) h.flipz(e.first);
        if (xf) h.flipz(e.second);
        if (xf && xs) ++quad;
    }
    (void)NW;
    h.phase = (M.phase + ((-lin - 2 * quad) % 4 + 4) % 4) & 3;
    return h;
}

void phi_of(const CertifiedGroupPlanes& G, const Pauli& Q, std::vector<uint64_t>& out) {
    out.assign((size_t)G.words, 0);
    const int NW = (G.n_qubits + 63) / 64;
    for (int w = 0; w < NW; ++w) {
        uint64_t xs = Q.x[(size_t)w], zs = Q.z[(size_t)w];
        while (xs) {   // <X_q, g> = g.z_q
            const int q = w * 64 + __builtin_ctzll(xs); xs &= xs - 1;
            const uint64_t* c = G.col(q, false);
            for (int t = 0; t < G.words; ++t) out[(size_t)t] ^= c[t];
        }
        while (zs) {   // <Z_q, g> = g.x_q
            const int q = w * 64 + __builtin_ctzll(zs); zs &= zs - 1;
            const uint64_t* c = G.col(q, true);
            for (int t = 0; t < G.words; ++t) out[(size_t)t] ^= c[t];
        }
    }
}

// ── Canonicalization ──────────────────────────────────────────────────────────────────────────
namespace {
// The record-order scan: columns sorted by their global record index (std::sort skipped when the
// storage order is already the record order — the same permutation either way).
void record_scan_(const std::vector<int>& order, int m, std::vector<int>& scan) {
    scan.resize((size_t)m);
    std::iota(scan.begin(), scan.end(), 0);
#ifndef XTIM_MUTATE_MC_STORAGE_ORDER
    if (!std::is_sorted(order.begin(), order.end()))
        std::sort(scan.begin(), scan.end(), [&](int x, int y) { return order[(size_t)x] < order[(size_t)y]; });
#endif   // mutation build only (M-C): scan in STORAGE order
    for (int i = 1; i < m; ++i)
        if (order[(size_t)scan[(size_t)i]] == order[(size_t)scan[(size_t)i - 1]])
            throw std::logic_error("record_table: canonicalize: duplicate record order (engine bug)");
}

// THE canonicalization (one implementation; W = uint64_t for <= 64 coins / directions — the one-word
// form AffineTable::small — or Bits): pivots and introducers in record order, the β' fold, Lemma 4.
template <class W, class GetA, class GetB>
void canonicalize_impl_(const AffineTable& in, const std::vector<int>& order, RecTable& out,
                        GetA getA, GetB getB) {
    const int m = in.m;
    std::vector<int> scan;
    record_scan_(order, m, scan);
    const int k = in.dirs.empty() ? 0 : (int)in.dirs[0].a.size();

    EchelonT<W> coin_ech, dir_ech;               // labels: canonical pivot slots / canonical dirs
    std::vector<W> pivB;                         // per pivot slot: its full direction row (old basis)
    std::vector<uint8_t> pivBeta;
    std::vector<W> dirB;                         // per canonical dir: its introducer's residual (old)
    std::vector<uint8_t> dirBeta;
    std::vector<W> Acan((size_t)m), Bcan((size_t)m);
    std::vector<uint8_t> bcan((size_t)m, 0);
    out = RecTable();
    out.m = m;
    for (int c : scan) {
        W a = getA(c);
        W t = coin_ech.reduce(a);
        if (wany(a)) {
            const int slot = (int)out.pivot_rec.size();
            W comb = t;
            wflip(comb, slot);
            coin_ech.insert(a, comb);
            out.pivot_rec.push_back(c);
            pivB.push_back(getB(c));
            pivBeta.push_back(in.beta[(size_t)c]);
            wclear(Acan[(size_t)c]);
            wflip(Acan[(size_t)c], slot);
            continue;
        }
        // Dependent coin content: y_c = Σ_{k∈t} y_{p_k} ⊕ b'·z ⊕ β'.
        W bres = getB(c);
        uint8_t beta = in.beta[(size_t)c];
        for (int s2 = 0; s2 < (int)out.pivot_rec.size(); ++s2)
            if (wbit(t, s2)) { wxor(bres, pivB[(size_t)s2]); beta ^= pivBeta[(size_t)s2]; }
        Acan[(size_t)c] = t;
        W r = bres;
        W d = dir_ech.reduce(r);
        if (wany(r)) {
            const int j = (int)out.lam.size();
            FreeDir L = fd_identity(k);
            for (int i = 0; i < (int)in.dirs.size(); ++i)
                if (wbit(bres, i)) L = fd_mul(L, in.dirs[(size_t)i]);
            if (beta & 1) ++out.folds;                 // the reachability dump (every build)
#ifndef XTIM_MUTATE_MBETA_NO_FOLD
            L.phase = (uint8_t)((L.phase + 2 * (beta & 1)) & 3);   // fold β' into the sign
#endif   // mutation build only (M-BETA): the folded β' sign dropped
#ifdef XTIM_MUTATE_MP1B_FLIP_COMBINED
            {   // mutation build only (MP-1b): a sign error in a COMBINED direction (|b| >= 2)
                int nin = 0;
                for (int i = 0; i < (int)in.dirs.size(); ++i) nin += wbit(bres, i) ? 1 : 0;
                if (nin >= 2) L.phase = (uint8_t)((L.phase + 2) & 3);
            }
#endif
#ifdef XTIM_MUTATE_MH2_ANTIHERM
            L.phase = (uint8_t)((L.phase + 1) & 3);   // mutation build only (MH-2): anti-Hermitian
#endif
#ifdef XTIM_MUTATE_MP1_RECANON
            {   // mutation build only (MP-1): re-canonicalize — drop the literal product's sign
                int ov = 0;
                for (int i = 0; i < k; ++i) ov += (L.a[(size_t)i] & L.b[(size_t)i]) ? 1 : 0;
                L.phase = (uint8_t)(ov & 3);
                L.par = 0;
            }
#endif
            fd_require_hermitian(L, "canonical direction");
            L.hermitian = 1;
            out.lam.push_back(L);
            out.born_rec.push_back(c);
            W comb = d;
            wflip(comb, j);
            dir_ech.insert(r, comb);
            dirB.push_back(bres);
            dirBeta.push_back(beta);
            wclear(Bcan[(size_t)c]);
            wflip(Bcan[(size_t)c], j);
            bcan[(size_t)c] = 0;
        } else {
            uint8_t b2 = beta;
            W chk{};
            for (int j = 0; j < (int)out.lam.size(); ++j)
                if (wbit(d, j)) { b2 ^= dirBeta[(size_t)j]; wxor(chk, dirB[(size_t)j]); }
            if (!wequal(chk, bres))
                throw std::logic_error("record_table: canonicalize: direction combination mismatch (engine bug)");
            Bcan[(size_t)c] = d;
            bcan[(size_t)c] = b2;
        }
    }
    out.rho = (int)out.pivot_rec.size();
    out.kappa = (int)out.lam.size();
    out.CWX = std::max(1, (m + 63) / 64);
    out.base.assign((size_t)out.CWX, 0);
    out.coin_row.assign((size_t)out.rho, Bits((size_t)out.CWX, 0));
    out.born_row.assign((size_t)out.kappa, Bits((size_t)out.CWX, 0));
    for (int c = 0; c < m; ++c) {
        const uint64_t bit = 1ULL << (c & 63);
        const size_t w = (size_t)(c >> 6);
        if (bcan[(size_t)c]) out.base[w] |= bit;
        for (int s2 = 0; s2 < out.rho; ++s2)
            if (wbit(Acan[(size_t)c], s2)) out.coin_row[(size_t)s2][w] |= bit;
        for (int j = 0; j < out.kappa; ++j)
            if (wbit(Bcan[(size_t)c], j)) out.born_row[(size_t)j][w] |= bit;
    }
    // Lemma 4 structure (asserted): pivots and introducers in record order; a column only uses
    // pivots / directions introduced at or before its own record position.
    std::vector<int> pos((size_t)m);
    for (int i = 0; i < m; ++i) pos[(size_t)scan[(size_t)i]] = i;
    for (int c = 0; c < m; ++c) {
        for (int s2 = 0; s2 < out.rho; ++s2)
            if (wbit(Acan[(size_t)c], s2) && pos[(size_t)out.pivot_rec[(size_t)s2]] > pos[(size_t)c])
                throw std::logic_error("record_table: Lemma 4 violated (coin support) (engine bug)");
        for (int j = 0; j < out.kappa; ++j)
            if (wbit(Bcan[(size_t)c], j) && pos[(size_t)out.born_rec[(size_t)j]] > pos[(size_t)c])
                throw std::logic_error("record_table: Lemma 4 violated (direction support) (engine bug)");
    }
    out.built = true;
}
}  // namespace

void canonicalize_record_order(const AffineTable& in, const std::vector<int>& order, RecTable& out) {
    const int m = in.m;
    const bool shape = in.small ? ((int)in.A64.size() == m && (int)in.B64.size() == m)
                                : ((int)in.A.size() == m && (int)in.B.size() == m);
    if ((int)order.size() != m || !shape || (int)in.beta.size() != m)
        throw std::logic_error("record_table: canonicalize: table shape mismatch (engine bug)");
    if (in.small) {
        if (in.ncoin > 64 || (int)in.dirs.size() > 64)
            throw std::logic_error("record_table: canonicalize: one-word form over 64 labels (engine bug)");
        canonicalize_impl_<uint64_t>(in, order, out, [&](int c) { return in.A64[(size_t)c]; },
                                     [&](int c) { return in.B64[(size_t)c]; });
    } else {
        canonicalize_impl_<Bits>(in, order, out, [&](int c) { return in.A[(size_t)c]; },
                                 [&](int c) { return in.B[(size_t)c]; });
    }
}

}  // namespace qeccore
