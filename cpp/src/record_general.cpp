// record_general — J-a N3: the record-table builder (G), null form. See qeccore/record_general.hpp.
#include "qeccore/record_general.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "record_table_internal.hpp"

namespace qeccore {

Pauli term_pullback(const TermResidual& tv, const std::vector<uint64_t>& S, const Pauli& M) {
    Pauli Q = M;
#ifdef XTIM_MUTATE_G_DRESS
    bool skipped = false;                     // mutation build only (MUT-G-DRESS): one dressing j skipped
#endif
    const int T = tv.num_terms();
    const size_t NW = M.x.size();
    for (size_t w = 0; w < S.size(); ++w)
        for (uint64_t x = S[w]; x; x &= x - 1) {
            const int j = (int)(w * 64) + __builtin_ctzll(x);
            if (j >= T) throw std::logic_error("term_pullback: term index out of range (engine bug)");
            const Pauli& B = tv.axis(j);
            int an = 0;                           // termroute-a F3: ⟨M, B⟩ inline (word popcounts)
            for (size_t v = 0; v < NW; ++v) an += __builtin_popcountll(M.x[v] & B.z[v]) + __builtin_popcountll(M.z[v] & B.x[v]);
            if (!(an & 1)) continue;
#ifdef XTIM_MUTATE_G_DRESS
            if (!skipped) { skipped = true; continue; }
#endif
            // rot(B)† Q rot(B) = Q · (iB) for {Q, B} = 0, in place: Pauli::multiply(Q, iB)'s phase rule
#ifdef XTIM_MUTATE_G_PULL
            const int ib = (B.phase + 3) & 3;     // mutation build only (MUT-G-PULL): −iB_j
#else
            const int ib = (B.phase + 1) & 3;
#endif
            int sg = 0;
            for (size_t v = 0; v < NW; ++v) sg += __builtin_popcountll(Q.z[v] & B.x[v]);
            Q.phase = (Q.phase + ib + 2 * (sg & 1)) & 3;
            for (size_t v = 0; v < NW; ++v) { Q.x[v] ^= B.x[v]; Q.z[v] ^= B.z[v]; }
        }
    return Q;
}

void build_record_table_general(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                const TermResidual& tv, const std::vector<uint64_t>& S,
                                const std::vector<RecColumn>& cols, RecTable& out,
                                std::vector<int>* intro_b, GeneralColumnMemo* memo) {
    const int m = (int)cols.size();
    AffineTable W;
    W.m = m;
    W.ncoin = 0;
    W.A.assign((size_t)m, {});
    W.B.assign((size_t)m, {});
    W.beta.assign((size_t)m, 0);
    std::vector<int> order((size_t)m), scan((size_t)m);
    for (int c = 0; c < m; ++c) order[(size_t)c] = cols[(size_t)c].order;
    std::iota(scan.begin(), scan.end(), 0);
    std::sort(scan.begin(), scan.end(), [&](int x, int y) { return order[(size_t)x] < order[(size_t)y]; });
    // The null form: every column is an extra of the table built so far, in record order.
    ExtraCtx X{G, bare, W, {}, {}, {}};
    if (!memo) {
        for (int c : scan) add_extra(X, term_pullback(tv, S, cols[(size_t)c].M), c);
    } else {
        GeneralColumnMemo& Mm = *memo;
        const int T = tv.num_terms();
        if (!Mm.ready) {                          // cold, once per owner: the per-term column masks
            Mm.MW = (m + 63) / 64;
            Mm.tcols.assign((size_t)T * (size_t)Mm.MW, 0);
            for (int j = 0; j < T; ++j)
                for (int c = 0; c < m; ++c)
                    if (Pauli::anticommute_bit(cols[(size_t)c].M, tv.axis(j)))
                        Mm.tcols[(size_t)j * Mm.MW + (size_t)(c >> 6)] |= 1ULL << (c & 63);
            Mm.aff.assign((size_t)Mm.MW, 0);
            Mm.ph.assign((size_t)m, {});
            Mm.ph_ready.assign((size_t)m, 0);
            Mm.phz.assign((size_t)m, 0);
            Mm.kd.assign((size_t)m, FreeDir{});
            Mm.ready = true;
        }
        std::fill(Mm.aff.begin(), Mm.aff.end(), 0ULL);
#ifdef XTIM_MUTATE_F2_MASK
        bool dropped = false;                     // mutation build only (MUT-F2-MASK): one term of S left out
#endif
        for (size_t w = 0; w < S.size(); ++w)
            for (uint64_t x = S[w]; x; x &= x - 1) {
                const int j = (int)(w * 64) + __builtin_ctzll(x);
                if (j >= T) throw std::logic_error("build_record_table_general: term index out of range (engine bug)");
#ifdef XTIM_MUTATE_F2_MASK
                if (!dropped) { dropped = true; continue; }
#endif
                const uint64_t* tc = &Mm.tcols[(size_t)j * Mm.MW];
                for (int u = 0; u < Mm.MW; ++u) Mm.aff[(size_t)u] |= tc[u];
            }
        for (int c : scan) {
            const Pauli& M = cols[(size_t)c].M;
            if ((Mm.aff[(size_t)(c >> 6)] >> (c & 63)) & 1ULL) {
                add_extra(X, term_pullback(tv, S, M), c);
                continue;
            }
            if (!Mm.ph_ready[(size_t)c]) {        // cold per column: its plane, and its direction when φ = 0
                phi_of(G, M, Mm.ph[(size_t)c]);
                bool z = true;
                for (uint64_t v : Mm.ph[(size_t)c]) if (v) { z = false; break; }
                Mm.phz[(size_t)c] = z ? 1 : 0;
                if (z) Mm.kd[(size_t)c] = free_dir_of(bare, M);
                Mm.ph_ready[(size_t)c] = 1;
            }
            add_extra(X, M, c, &Mm.ph[(size_t)c], Mm.phz[(size_t)c] ? &Mm.kd[(size_t)c] : nullptr);
        }
    }
    canonicalize_record_order(W, order, out);
    if (intro_b) {                            // N3-D3: |b| of every canonical introducer (bres, as canonicalize)
        intro_b->clear();
        for (int j = 0; j < out.kappa; ++j) {
            const int c = out.born_rec[(size_t)j];
            Bits b = W.B[(size_t)c];
            for (int s2 = 0; s2 < out.rho; ++s2)
                if ((out.coin_row[(size_t)s2][(size_t)(c >> 6)] >> (c & 63)) & 1ULL)
                    bits_xor(b, W.B[(size_t)out.pivot_rec[(size_t)s2]]);
            int nb = 0;
            for (uint64_t w : b) nb += __builtin_popcountll(w);
            intro_b->push_back(nb);
        }
    }
    out.extras = m;
    out.ops_plane = X.ops_plane;
    out.ops_pivot_reduce = X.ops_pivot;
    out.ops_greduce = X.ops_greduce;
    out.ops_echelon = X.ops_ech;
}

std::vector<Pauli> diag_plan_axes(const DiagNormalForm& nf, int n) {
    auto z = [&](std::initializer_list<int> qs, int sign) {
        Pauli B(n);
        for (int q : qs) B.setz(q);
        B.phase = sign > 0 ? 0 : 2;
        return B;
    };
    std::vector<Pauli> axes;
    for (int q = 0; q < n && q < (int)nf.a.size(); ++q)
        if (nf.a[(size_t)q] & 1) axes.push_back(z({q}, -1));
    for (const auto& e : nf.cz) {
        axes.push_back(z({e.first}, -1));
        axes.push_back(z({e.second}, -1));
        axes.push_back(z({e.first, e.second}, +1));
    }
    return axes;
}

}  // namespace qeccore
