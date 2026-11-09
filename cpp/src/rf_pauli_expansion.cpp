// rf_pauli_expansion — see the header (S8 RF2; fable-decoded-design.md §2.1).
#include "qeccore/rf_pauli_expansion.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "qeccore/twirl_kernel.hpp"

namespace qeccore {

namespace {
inline bool gb(const uint64_t* v, int b) { return (v[b >> 6] >> (b & 63)) & 1ULL; }
inline void fb(uint64_t* v, int b) { v[b >> 6] ^= 1ULL << (b & 63); }
}  // namespace

int DiagExpansion::qform(const uint64_t* x) const {
    int q = 0;
    for (int w = 0; w < W; ++w)
        for (uint64_t m = x[w]; m; m &= m - 1) {
            const int i = w * 64 + __builtin_ctzll(m);
            q += a1[(size_t)i];
            const int ri = rowof[(size_t)i];
            if (ri < 0) continue;
            const uint64_t* row = &arows[(size_t)ri * (size_t)W];
            for (int v = 0; v < W; ++v) q += __builtin_popcountll(row[v] & x[v]);   // 2·#pairs, summed twice
        }
    return q & 3;
}

DiagExpansion expand_diag(const DiagNormalForm& nf, int n) {
    DiagExpansion E;
    E.n = n;
    E.W = (n + 63) / 64;
    const int W = E.W;
    E.a1.assign((size_t)n, 0);
    std::vector<int> order;
    {
        const int na = std::min(n, (int)nf.a.size());
        const uint8_t* ap = nf.a.data();
        int q = 0;
        for (; q + 8 <= na; q += 8) {                            // the S part is sparse: skip zero 8-byte runs
            uint64_t w8;
            std::memcpy(&w8, ap + q, 8);
            if (!(w8 & 0x0101010101010101ULL)) continue;
            for (int j = q; j < q + 8; ++j) E.a1[(size_t)j] = ap[j] & 1;
        }
        for (; q < na; ++q) E.a1[(size_t)q] = ap[q] & 1;
    }
    E.rowof.assign((size_t)n, -1);
    // the touched wires ascending (their index t is the compressed coordinate: compressed order = wire order)
    for (const auto& pr : nf.cz) {
        const int f = pr.first, g = pr.second;
#ifdef XTIM_MUTATE_RF_MALF
        if (false)                                               // mutation build only: the malformed check skipped
#else
        if (f < 0 || g < 0 || f >= n || g >= n || f == g)
#endif
            throw std::invalid_argument("RecordFidelity: RF_CONTENT_MALFORMED: a kind-0 content has the CZ pair (" +
                                        std::to_string(f) + ", " + std::to_string(g) + ") on " + std::to_string(n) +
                                        " wires — refused");
        if (E.rowof[(size_t)f] < 0) { E.rowof[(size_t)f] = 0; order.push_back(f); }
        if (E.rowof[(size_t)g] < 0) { E.rowof[(size_t)g] = 0; order.push_back(g); }
    }
    {
        const int na = std::min(n, (int)nf.a.size());
        const uint8_t* ap = nf.a.data();
        for (int q = 0; q < na; q += 8) {
            if (q + 8 <= na) {
                uint64_t w8;
                std::memcpy(&w8, ap + q, 8);
                if (!(w8 & 0x0101010101010101ULL)) continue;
            }
            for (int j = q; j < std::min(q + 8, na); ++j)
                if (E.a1[(size_t)j] && E.rowof[(size_t)j] < 0) { E.rowof[(size_t)j] = 0; order.push_back(j); }
        }
    }
    std::sort(order.begin(), order.end());
    const int m = (int)order.size(), WM = std::max(1, (m + 63) / 64);
    for (int t = 0; t < m; ++t) E.rowof[(size_t)order[(size_t)t]] = t;
    // M by touched wire: the compressed rows (Ã = M + diag(a), m bits), then the n-bit rows of M (the dressed images)
    std::vector<uint64_t> A((size_t)m * WM, 0);
    for (const auto& pr : nf.cz) {                               // M ⊕= e_f e_gᵀ + e_g e_fᵀ (duplicates cancel)
        const int tf = E.rowof[(size_t)pr.first], tg = E.rowof[(size_t)pr.second];
        fb(&A[(size_t)tf * WM], tg);
        fb(&A[(size_t)tg * WM], tf);
    }
    E.arows.assign((size_t)m * W, 0);
    for (int t = 0; t < m; ++t)
        for (int w = 0; w < WM; ++w)
            for (uint64_t mm = A[(size_t)t * WM + w]; mm; mm &= mm - 1)
                fb(&E.arows[(size_t)t * W], order[(size_t)(w * 64 + __builtin_ctzll(mm))]);
    for (int t = 0; t < m; ++t) if (E.a1[(size_t)order[(size_t)t]]) fb(&A[(size_t)t * WM], t);
    // row reduction over the columns (touched ascending): the first remaining row with the bit is the pivot
    // RREF by rows (each row reduced by the pivot rows, its lowest set bit its pivot, the pivot rows kept reduced):
    // the pivot columns are the RREF's leading columns (unique: independent of the row order)
    std::vector<uint64_t> R((size_t)m * WM);
    std::vector<int> pivt;
    int nr = 0;
    for (int o = 0; o < m; ++o) {
        uint64_t* v = &R[(size_t)nr * WM];
        for (int w = 0; w < WM; ++w) v[w] = A[(size_t)o * WM + w];
        for (int i = 0; i < nr; ++i) {                          // branch-free: the bit is data-dependent
            const int pi = pivt[(size_t)i];
            const uint64_t msk = 0ULL - ((v[pi >> 6] >> (pi & 63)) & 1ULL);
            const uint64_t* pr = &R[(size_t)i * WM];
            for (int w = 0; w < WM; ++w) v[w] ^= pr[w] & msk;
        }
        int p = -1;
        for (int w = 0; w < WM && p < 0; ++w) if (v[w]) p = w * 64 + __builtin_ctzll(v[w]);
        if (p < 0) continue;
        for (int i = 0; i < nr; ++i) {
            uint64_t* ri = &R[(size_t)i * WM];
            const uint64_t msk = 0ULL - ((ri[p >> 6] >> (p & 63)) & 1ULL);
            for (int w = 0; w < WM; ++w) ri[w] ^= v[w] & msk;
        }
        pivt.push_back(p);
        ++nr;
    }
    // ascending pivot columns (the rows follow)
    std::vector<int> ordp((size_t)nr);
    for (int i = 0; i < nr; ++i) ordp[(size_t)i] = i;
    std::sort(ordp.begin(), ordp.end(), [&](int x, int y) { return pivt[(size_t)x] < pivt[(size_t)y]; });
    std::vector<int> rref((size_t)nr);
    {
        std::vector<int> pv((size_t)nr);
        for (int i = 0; i < nr; ++i) { rref[(size_t)i] = ordp[(size_t)i]; pv[(size_t)i] = pivt[(size_t)ordp[(size_t)i]]; }
        pivt.swap(pv);
    }
    E.r = (int)pivt.size();
    for (int t : pivt) E.piv.push_back(order[(size_t)t]);
    // q on a compressed vector: Σ a_i x_i + Σ_{i ∈ x} |M_i ∧ x| (= 2·#pairs) mod 4
    auto qc = [&](const uint64_t* x) {
        int q = 0;
        for (int w = 0; w < WM; ++w)
            for (uint64_t mm = x[w]; mm; mm &= mm - 1) {
                const int t = w * 64 + __builtin_ctzll(mm);
                q += E.a1[(size_t)order[(size_t)t]];
                const uint64_t* row = &A[(size_t)t * WM];
                int pc = 0;
                for (int v = 0; v < WM; ++v) pc += __builtin_popcountll(row[v] & x[v]);
                q += pc - (gb(row, t) ? 1 : 0);                  // the diagonal a_t is not an M pair
            }
        return q & 3;
    };
    // kernel of Ã: per non-pivot column f, κ_f = e_f ⊕ Σ_{rref rows with bit f} e_{their pivot column} — its only
    // non-pivot coordinate is f, so z_0 supported off the pivots with z_0[f] = q(κ_f)/2 solves z_0·κ = q(κ)/2 on ker Ã
    std::vector<uint8_t> ispiv((size_t)m, 0);
    for (int c : pivt) ispiv[(size_t)c] = 1;
    std::vector<uint64_t> kv((size_t)WM);
    E.z0.assign((size_t)W, 0);
    for (int f = 0; f < m; ++f) {
        if (ispiv[(size_t)f]) continue;
        std::fill(kv.begin(), kv.end(), 0);
        fb(kv.data(), f);
        for (size_t j = 0; j < rref.size(); ++j) if (gb(&R[(size_t)rref[j] * WM], f)) fb(kv.data(), pivt[j]);
        const int qk = qc(kv.data());
        if (qk & 1) throw std::logic_error("RecordFidelity: q is odd on ker Ã (internal)");
#ifndef XTIM_MUTATE_RF_Z0
        if (qk >> 1) fb(E.z0.data(), order[(size_t)f]);
#endif                                                           // mutation build only: z_0 dropped
    }
    // the expansion's columns (n-bit rows of Ã), Q_D's coefficients
    const int r = E.r;
    std::vector<int> pidx((size_t)m, -1);                       // compressed column -> its pivot index
    for (int i = 0; i < r; ++i) pidx[(size_t)pivt[(size_t)i]] = i;
    E.WB = (r + 63) / 64;
    E.cols.assign((size_t)r * W, 0);
    E.lin.assign((size_t)r, 0);
    E.beta.assign((size_t)r * (size_t)std::max(E.WB, 1), 0);
    for (int i = 0; i < r; ++i) {
        const int t = pivt[(size_t)i], p = order[(size_t)t];
        for (int w = 0; w < W; ++w) E.cols[(size_t)i * W + w] = E.arows[(size_t)t * W + w];
        if (E.a1[(size_t)p]) fb(&E.cols[(size_t)i * W], p);
#ifdef XTIM_MUTATE_RF_QD
        E.lin[(size_t)i] = (uint8_t)((E.a1[(size_t)p] + 2 * (int)gb(E.z0.data(), p)) & 3);   // mutation: +a, not −a
#else
        E.lin[(size_t)i] = (uint8_t)((-(int)E.a1[(size_t)p] + 2 * (int)gb(E.z0.data(), p)) & 3);
#endif
#ifndef XTIM_MUTATE_RF_BETA
        const uint64_t* mrow = &A[(size_t)t * WM];
        for (int w = 0; w < WM; ++w)
            for (uint64_t mm = mrow[w]; mm; mm &= mm - 1) {
                const int c = w * 64 + __builtin_ctzll(mm), j = pidx[(size_t)c];
                if (j >= 0 && j != i) fb(&E.beta[(size_t)i * E.WB], j);
            }
#endif                                                           // mutation build only: β of Q_D dropped
    }
    return E;
}

void QForm::init(int V_) {
    V = V_;
    WV = std::max(1, (V + 63) / 64);
    c = 0;
    lin.assign((size_t)V, 0);
    beta.assign((size_t)V * (size_t)WV, 0);
}
int QForm::eval(const uint64_t* u) const {
    int q = c;
    for (int w = 0; w < WV; ++w)
        for (uint64_t m = u[w]; m; m &= m - 1) {
            const int j = w * 64 + __builtin_ctzll(m), b = j & 63;
            q += lin[(size_t)j];
            const uint64_t* r = row(j);
            int pc = b == 63 ? 0 : __builtin_popcountll(r[w] & u[w] & ~((2ULL << b) - 1));   // pairs j < j' only
            for (int v = w + 1; v < WV; ++v) pc += __builtin_popcountll(r[v] & u[v]);
            q += 2 * pc;
        }
    return q & 3;
}
int QForm::polar(const uint64_t* u, const uint64_t* v) const {
    int b = 0;
    for (int w = 0; w < WV; ++w)
        for (uint64_t m = u[w]; m; m &= m - 1) {
            const int j = w * 64 + __builtin_ctzll(m);
            if ((lin[(size_t)j] & 1) && ((v[w] >> (j & 63)) & 1)) b ^= 1;
            const uint64_t* r = row(j);
#ifdef XTIM_MUTATE_RF_MW
            for (int t = 0; t < 1; ++t) b ^= __builtin_popcountll(r[t] & v[t]) & 1;   // mutation: the second word dropped
#else
            for (int t = 0; t < WV; ++t) b ^= __builtin_popcountll(r[t] & v[t]) & 1;
#endif
        }
    return b & 1;
}

void gauss_sum(QForm& Q, const std::vector<uint8_t>& keep, GaussSum& G) {
    G.zero = false;
    G.h = G.p8 = 0;
    G.free_keep.clear();
    G.dep_var.clear();
    G.dep_c.clear();
    G.dep_mask.clear();
    const int V = Q.V, WV = Q.WV;
    std::vector<uint8_t>& alive = G.alive_;
    alive.assign((size_t)V, 1);
    std::vector<uint64_t>& S = G.S_;
    std::vector<uint64_t>& N = G.N_;
    S.assign((size_t)WV, 0);
    N.assign((size_t)WV, 0);
    auto bit = [](const uint64_t* v, int j) { return (v[j >> 6] >> (j & 63)) & 1ULL; };
    auto remove = [&](int j) {                       // drop variable j (its row and column)
        uint64_t* r = Q.row(j);
        for (int w = 0; w < WV; ++w)
            for (uint64_t m = r[w]; m; m &= m - 1) {
                const int k = w * 64 + __builtin_ctzll(m);
                Q.row(k)[j >> 6] &= ~(1ULL << (j & 63));
            }
        for (int w = 0; w < WV; ++w) r[w] = 0;
        Q.lin[(size_t)j] = 0;
        alive[(size_t)j] = 0;
    };
    // u_j := t ⊕ XOR_{m ∈ S} u_m (as a Z4 integer: t + Σ_S u − 2 t Σ_S u − 2 Σ_{m<m'∈S} u_m u_m')
    auto substitute = [&](int j, int t) {
        const int lj = Q.lin[(size_t)j];
        std::copy(Q.row(j), Q.row(j) + WV, N.begin());            // j's neighbours (S ∩ N possible)
        // lin_j · u_j
        Q.c = (Q.c + lj * t) & 3;
        for (int w = 0; w < WV; ++w)
            for (uint64_t m = S[(size_t)w]; m; m &= m - 1) {
                const int k = w * 64 + __builtin_ctzll(m);
                Q.lin[(size_t)k] = (uint8_t)((Q.lin[(size_t)k] + lj + 2 * ((lj * t) & 1)) & 3);
                if (lj & 1) {                                     // −2 lin_j Σ_{pairs in S}: flip every pair of S
                    uint64_t* rk = Q.row(k);
                    for (int v = 0; v < WV; ++v) rk[v] ^= S[(size_t)v];
                    rk[k >> 6] ^= 1ULL << (k & 63);              // (the diagonal was flipped too: undo)
                }
            }
        // 2 β_jk u_j u_k, k ∈ N: 2 u_k (t + Σ_S u_m)
        for (int w = 0; w < WV; ++w)
            for (uint64_t m = N[(size_t)w]; m; m &= m - 1) {
                const int k = w * 64 + __builtin_ctzll(m);
                if (k == j) continue;
                Q.lin[(size_t)k] = (uint8_t)((Q.lin[(size_t)k] + 2 * t) & 3);
                if (bit(S.data(), k)) Q.lin[(size_t)k] = (uint8_t)((Q.lin[(size_t)k] + 2) & 3);
                uint64_t* rk = Q.row(k);
                for (int v = 0; v < WV; ++v) rk[v] ^= S[(size_t)v];
            }
        for (int w = 0; w < WV; ++w)
            for (uint64_t m = S[(size_t)w]; m; m &= m - 1) {
                const int s2 = w * 64 + __builtin_ctzll(m);
                uint64_t* rs = Q.row(s2);
                for (int v = 0; v < WV; ++v) rs[v] ^= N[(size_t)v];
            }
        for (int k = 0; k < V; ++k) if (alive[(size_t)k]) Q.row(k)[k >> 6] &= ~(1ULL << (k & 63));   // diagonal stays clear
        // j now carries no information: clear j from every row (its old column), then drop it
        for (int k = 0; k < V; ++k) Q.row(k)[j >> 6] &= ~(1ULL << (j & 63));
        for (int w = 0; w < WV; ++w) Q.row(j)[w] = 0;
        Q.lin[(size_t)j] = 0;
        alive[(size_t)j] = 0;
        // recorded dependent keep variables that reference j
        for (size_t d = 0; d < G.dep_var.size(); ++d) {
            uint64_t* dm = &G.dep_mask[d * (size_t)WV];
            if (!bit(dm, j)) continue;
            G.dep_c[d] ^= (uint8_t)t;
            for (int w = 0; w < WV; ++w) dm[w] ^= S[(size_t)w];
            dm[j >> 6] ^= 1ULL << (j & 63);
        }
    };
    for (int v = 0; v < V; ++v) {
        if (!alive[(size_t)v] || keep[(size_t)v]) continue;
        const int l = Q.lin[(size_t)v];
        std::copy(Q.row(v), Q.row(v) + WV, S.begin());            // N(v)
        remove(v);
        if ((l & 1) == 0) {
            const int t = l >> 1;
            int j = -1;
            for (int w = 0; w < WV && j < 0; ++w)
                for (uint64_t m = S[(size_t)w]; m; m &= m - 1) {
                    const int k = w * 64 + __builtin_ctzll(m);
                    if (!keep[(size_t)k]) { j = k; break; }
                }
            if (j < 0)
                for (int w = 0; w < WV && j < 0; ++w) if (S[(size_t)w]) j = w * 64 + __builtin_ctzll(S[(size_t)w]);
            if (j < 0) {                                         // no partner: 1 + (−1)^t
                if (t) { G.zero = true; return; }
                G.h += 2;
                continue;
            }
            S[(size_t)(j >> 6)] ^= 1ULL << (j & 63);             // S = N(v) \ {j}
            G.h += 2;
            if (keep[(size_t)j]) {
                G.dep_var.push_back(j);
                G.dep_c.push_back((uint8_t)t);
                G.dep_mask.insert(G.dep_mask.end(), S.begin(), S.end());
            }
            substitute(j, t);
        } else {
            // 1 + i^l (−1)^{XOR_N u} = (1 ± i) · i^{∓ XOR_N u}  (l = 1: (1 + i) i^{−XOR}; l = 3: (1 − i) i^{+XOR})
            G.h += 1;
            G.p8 = (G.p8 + (l == 1 ? 1 : 7)) & 7;
            const int dl = l == 1 ? 3 : 1;
            for (int w = 0; w < WV; ++w)
                for (uint64_t m = S[(size_t)w]; m; m &= m - 1) {
                    const int k = w * 64 + __builtin_ctzll(m);
                    Q.lin[(size_t)k] = (uint8_t)((Q.lin[(size_t)k] + dl) & 3);
                    uint64_t* rk = Q.row(k);
                    for (int t2 = 0; t2 < WV; ++t2) rk[t2] ^= S[(size_t)t2];
                    rk[k >> 6] ^= 1ULL << (k & 63);
                }
        }
    }
    G.p8 = (G.p8 + 2 * Q.c) & 7;
    for (int j = 0; j < V; ++j) if (alive[(size_t)j] && keep[(size_t)j]) G.free_keep.push_back(j);
    const int F = (int)G.free_keep.size();
    G.f.init(F);
    for (int i = 0; i < F; ++i) {
        const int j = G.free_keep[(size_t)i];
        G.f.lin[(size_t)i] = Q.lin[(size_t)j];
        for (int i2 = 0; i2 < F; ++i2)
            if (i2 != i && bit(Q.row(j), G.free_keep[(size_t)i2])) G.f.row(i)[i2 >> 6] |= 1ULL << (i2 & 63);
    }
}

Pauli dress_diag(const DiagExpansion& E, const Pauli& M) {
    Pauli h = M;
    const int W = E.W;
    for (int w = 0; w < W; ++w)
        for (uint64_t mm = M.x[(size_t)w]; mm; mm &= mm - 1) {
            const int q = w * 64 + __builtin_ctzll(mm);
            if (E.a1[(size_t)q]) h.flipz(q);
            const int ri = E.rowof[(size_t)q];
            if (ri < 0) continue;
            const uint64_t* row = &E.arows[(size_t)ri * W];
            for (int v = 0; v < W; ++v) h.z[(size_t)v] ^= row[v];
        }
#ifdef XTIM_MUTATE_RF_DRESS
    h.phase = (M.phase - (E.qform(M.x.data()) & 1)) & 3;           // mutation: the CZ part of q(x) dropped
#else
    h.phase = (M.phase - E.qform(M.x.data())) & 3;
#endif
    return h;
}

}  // namespace qeccore
