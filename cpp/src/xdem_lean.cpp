// xdem_lean.cpp — the lean xdem route (LEAN-EXPORT-PLAN). Task 1: algebraic determinism (nondeterministic_columns).
// Task 4: the line merge (LeanLineAccumulator). Task 5: the emitter (emit_fast).
#include "qeccore/xdem_lean.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/xdem_file.hpp"

namespace qeccore::xdem {

namespace {
std::atomic<bool> g_inject_throw{false};   // set_lean_inject_throw_for_test
// LM-2: one column-word chunk of a row-major bit matrix transposed: for the columns [c0, c1) (c0 a multiple of 64),
// the rows holding each column's bit, ascending, as CSR (off: c1 - c0 + 1 offsets). row_word(i, w): row i's word w.
template <class RowWord>
void transpose_chunk(size_t nrows, int c0, int c1, const RowWord& row_word, std::vector<size_t>& off,
                     std::vector<size_t>& cur, std::vector<int32_t>& idx) {
    const size_t w0 = (size_t)c0 >> 6, w1 = ((size_t)c1 + 63) >> 6;
    const uint64_t last = (c1 & 63) ? (1ull << (c1 & 63)) - 1 : ~0ull;
    off.assign((size_t)(c1 - c0) + 1, 0);
    for (size_t i = 0; i < nrows; ++i)
        for (size_t w = w0; w < w1; ++w) {
            uint64_t b = row_word(i, w);
            if (w + 1 == w1) b &= last;
            for (; b; b &= b - 1) ++off[(w - w0) * 64 + (size_t)__builtin_ctzll(b) + 1];
        }
    for (size_t j = 1; j < off.size(); ++j) off[j] += off[j - 1];
    idx.resize(off.back());
    cur.assign(off.begin(), off.end() - 1);
    for (size_t i = 0; i < nrows; ++i)
        for (size_t w = w0; w < w1; ++w) {
            uint64_t b = row_word(i, w);
            if (w + 1 == w1) b &= last;
            for (; b; b &= b - 1) idx[cur[(w - w0) * 64 + (size_t)__builtin_ctzll(b)]++] = (int32_t)i;
        }
}
}  // namespace

void set_lean_inject_throw_for_test(bool on) { g_inject_throw.store(on, std::memory_order_relaxed); }

// <P> = E_u prod_{a in A_P} e^{i pi n_a q_a / 4} q_B, q_a = (-1)^{b_a + kappa_a.u} (derivation: see
// docs/xdem.md): deterministic iff the Z4 form Phi(u) is constant (see the header). LM-2: the sparse form (the
// header); the dense one it replaced is the test oracle (cpp/testapi/xdem_lean_dense_oracle.hpp).
std::vector<int> nondeterministic_columns(const WriterModel& wm, const FaultColumns& fc, NondetStats* stats) {
    if (g_inject_throw.load(std::memory_order_relaxed)) throw std::runtime_error("xdem nondeterministic_columns: injected (test)");
    const Skeleton& sk = *wm.sk;
    const Header& H = wm.header;
    const int A = (int)H.axes.size(), N = sk.N, r = H.r, T = H.D + H.O;
    const bool xsp = fc.x_sparse();                            // LM-3: x_start as per-column wire lists
    if ((int)sk.axis_p0.size() != A || fc.ncols < T ||
        (xsp ? (fc.xs_off.size() != (size_t)fc.ncols + 1 || fc.xs_off.back() != fc.xs_wire.size() ||
                !fc.x_start.empty())
             : fc.x_start.size() != (size_t)N * fc.cw))
        throw std::logic_error("xdem nondeterministic_columns: writer model / fault columns mismatch (engine bug)");
    auto getb = [](const std::vector<uint64_t>& w, size_t i) { return ((w[i >> 6] >> (i & 63)) & 1ull) != 0; };
    std::vector<int> n((size_t)A, 0);                          // n_a mod 8
    for (const Group& gr : H.groups)
        for (size_t vi = 0; vi < gr.v_axis.size(); ++vi) {
            const auto [a, s] = gr.v_axis[vi];
            if (a < 0) continue;                               // an identity axis: a global phase
            n[(size_t)a] = ((n[(size_t)a] + s * gate_rotation_count(gr.gate, (uint32_t)(vi + 1))) % 8 + 8) % 8;
        }
    std::vector<uint8_t> wa((size_t)A);                        // w_a = n_a (1 - 2 b_a) mod 4
    for (int a = 0; a < A; ++a) wa[(size_t)a] = (uint8_t)(((n[(size_t)a] * (H.axes[(size_t)a].b ? -1 : 1)) % 4 + 4) % 4);
    // Fully reduced echelon of X(U^dag Q_a U) (N bits), pivot = lowest set bit, with combinations (A bits). Fully
    // reduced: a pivot wire is set in its own row only, so reducing v applies exactly the rows whose pivot v holds
    // (pivot_row lookups on v's own bits; the dense form's sequential scan applies the same rows).
    const size_t NW = (size_t)(N + 63) / 64, AW = (size_t)A / 64 + 1, RW = std::max<size_t>(1, words_for(r));
    std::vector<std::vector<uint64_t>> rows, comb;
    std::vector<int32_t> pivot_row((size_t)N, -1);
    std::vector<int32_t> app;                                  // the rows a reduction applies
    for (int a = 0; a < A; ++a) {
        std::vector<uint64_t> v = sk.axis_p0[(size_t)a].x, c(AW, 0);
        v.resize(NW, 0);
        c[(size_t)a >> 6] |= 1ull << (a & 63);
        app.clear();
        for (size_t w = 0; w < NW; ++w)
            for (uint64_t b = v[w]; b; b &= b - 1) {
                const size_t q = w * 64 + (size_t)__builtin_ctzll(b);
                if (q < (size_t)N && pivot_row[q] >= 0) app.push_back(pivot_row[q]);
            }
        for (int32_t i : app) {
            for (size_t w = 0; w < NW; ++w) v[w] ^= rows[(size_t)i][w];
            for (size_t w = 0; w < AW; ++w) c[w] ^= comb[(size_t)i][w];
        }
        int h = -1;
        for (size_t w = 0; w < NW && h < 0; ++w) if (v[w]) h = (int)(w * 64 + (size_t)__builtin_ctzll(v[w]));
        if (h < 0) continue;
        for (size_t i = 0; i < rows.size(); ++i)
            if (getb(rows[i], (size_t)h)) {
                for (size_t w = 0; w < NW; ++w) rows[i][w] ^= v[w];
                for (size_t w = 0; w < AW; ++w) comb[i][w] ^= c[w];
            }
        pivot_row[(size_t)h] = (int32_t)rows.size();
        rows.push_back(std::move(v)); comb.push_back(std::move(c));
    }
    // Column-word chunks: a chunk's lists (x supports + axis images) hold at most 2 (N + A) TW int32 entries (the bytes
    // of the T columns' bitmaps), EXCEPT that a single column word is always admitted: one word's lists can hold up to
    // 64 (N + A) entries, 32x its bitmap; it exceeds the 2 (N + A) TW budget whenever TW < 32 and that word's supports
    // are dense. Memory stays linear in N + A.
    const size_t TW = words_for(T);
    std::vector<size_t> wnnz(TW, 0);
    auto x_word = [&](size_t w, size_t tw) { return fc.x_start[w * fc.cw + tw]; };
    auto img_word = [&](size_t a, size_t tw) {
        const std::vector<uint64_t>& iw = H.axes[a].img.w;
        return tw < iw.size() ? iw[tw] : 0ull;
    };
    const uint64_t tmask = (T & 63) ? (1ull << (T & 63)) - 1 : ~0ull;
    if (xsp) {                                                 // LM-3: the lists' counts per column word
        for (size_t tw = 0; tw < TW; ++tw)
            wnnz[tw] += fc.xs_off[std::min((size_t)T, (tw + 1) * 64)] - fc.xs_off[tw * 64];
    } else {
        for (size_t w = 0; w < (size_t)N; ++w)
            for (size_t tw = 0; tw < TW; ++tw)
                wnnz[tw] += (size_t)__builtin_popcountll(x_word(w, tw) & (tw + 1 == TW ? tmask : ~0ull));
    }
    for (size_t a = 0; a < (size_t)A; ++a)
        for (size_t tw = 0; tw < TW; ++tw) wnnz[tw] += (size_t)__builtin_popcountll(img_word(a, tw) & (tw + 1 == TW ? tmask : ~0ull));
    const size_t budget = 2 * ((size_t)N + (size_t)A) * TW;
    std::vector<int> bad;
    std::vector<uint64_t> x(NW, 0), c(AW, 0), kB(RW, 0);
    std::vector<uint8_t> alpha((size_t)r, 0), alpha_on((size_t)r, 0), brow_on((size_t)r, 0);
    std::vector<int32_t> alpha_list, brows;                    // the gauge bits / beta rows this column touched
    std::vector<std::vector<uint64_t>> beta((size_t)r);        // allocated on first touch; zero between columns
    std::vector<size_t> xoff, ioff, cur;
    std::vector<int32_t> xidx, iidx;
    auto touch_alpha = [&](size_t i) { if (!alpha_on[i]) { alpha_on[i] = 1; alpha_list.push_back((int32_t)i); } };
    for (size_t tw0 = 0; tw0 < TW;) {
        size_t tw1 = tw0 + 1, acc = wnnz[tw0];
        while (tw1 < TW && acc + wnnz[tw1] <= budget) acc += wnnz[tw1++];
        const int c0 = (int)(tw0 * 64), c1 = std::min(T, (int)(tw1 * 64));
        const size_t* xo;                                      // the chunk's column lists: xo[j] .. xo[j + 1] in xi
        const int32_t* xi;
        if (xsp) {                                             // LM-3: slices of the sweep's lists (no transpose)
            xo = fc.xs_off.data() + c0;
            xi = fc.xs_wire.data();
        } else {
            transpose_chunk((size_t)N, c0, c1, x_word, xoff, cur, xidx);
            xo = xoff.data();
            xi = xidx.data();
        }
        transpose_chunk((size_t)A, c0, c1, img_word, ioff, cur, iidx);
        if (stats) {
            ++stats->chunks;
            stats->x_nnz += (long long)(xo[c1 - c0] - xo[0]);
            stats->img_nnz += (long long)iidx.size();
        }
        for (int t = c0; t < c1; ++t) {
            const size_t j = (size_t)(t - c0);
            // reduce X(U^dag P U) (the column's start X support) by the echelon
            app.clear();
            for (size_t e = xo[j]; e < xo[j + 1]; ++e)
                if (pivot_row[(size_t)xi[e]] >= 0) app.push_back(pivot_row[(size_t)xi[e]]);
            if (app.empty() && xo[j + 1] > xo[j]) { bad.push_back(t); continue; }   // nothing cancels it: <P> = 0
            std::fill(kB.begin(), kB.end(), 0);
            if (!app.empty()) {
                for (size_t e = xo[j]; e < xo[j + 1]; ++e) x[(size_t)xi[e] >> 6] |= 1ull << (xi[e] & 63);
                for (int32_t i : app) {
                    for (size_t w = 0; w < NW; ++w) x[w] ^= rows[(size_t)i][w];
                    for (size_t w = 0; w < AW; ++w) c[w] ^= comb[(size_t)i][w];
                }
                bool nz = false;
                for (uint64_t w : x) nz |= w != 0;
                std::fill(x.begin(), x.end(), 0);
                if (nz) { std::fill(c.begin(), c.end(), 0); bad.push_back(t); continue; }   // not in the axes' span
                for (size_t w = 0; w < AW; ++w)                // kappa_B = sum of kappa_a over the solution set B
                    for (uint64_t b = c[w]; b; b &= b - 1) {
                        const std::vector<uint64_t>& ka = H.axes[w * 64 + (size_t)__builtin_ctzll(b)].k;
                        for (size_t u = 0; u < RW && u < ka.size(); ++u) kB[u] ^= ka[u];
                    }
                std::fill(c.begin(), c.end(), 0);
            }
            for (size_t w = 0; w < RW; ++w)
                for (uint64_t b = kB[w]; b; b &= b - 1) {
                    const size_t i = w * 64 + (size_t)__builtin_ctzll(b);
                    if (i >= (size_t)r) break;
                    alpha[i] = 2; touch_alpha(i);
                }
            for (size_t e = ioff[j]; e < ioff[j + 1]; ++e) {   // a in A_P
                const size_t a = (size_t)iidx[e];
                if (!wa[a]) continue;
                const std::vector<uint64_t>& ka = H.axes[a].k;
                for (size_t w = 0; w < ka.size(); ++w)
                    for (uint64_t b = ka[w]; b; b &= b - 1) {
                        const size_t i = w * 64 + (size_t)__builtin_ctzll(b);
                        if (i >= (size_t)r) break;
                        alpha[i] = (uint8_t)((alpha[i] + wa[a]) & 3); touch_alpha(i);
                        if (wa[a] & 1) {
                            std::vector<uint64_t>& row = beta[i];
                            if (row.empty()) row.assign(RW, 0);
                            if (!brow_on[i]) { brow_on[i] = 1; brows.push_back((int32_t)i); }
                            for (size_t u = 0; u < ka.size() && u < RW; ++u) row[u] ^= ka[u];
                        }
                    }
            }
            bool det = true;
            for (int32_t i : alpha_list) { if (alpha[(size_t)i] & 3) det = false; alpha[(size_t)i] = 0; alpha_on[(size_t)i] = 0; }
            alpha_list.clear();
            for (int32_t i : brows) {                          // only the touched rows can be nonzero
                std::vector<uint64_t>& row = beta[(size_t)i];
                row[(size_t)i >> 6] &= ~(1ull << (i & 63));    // the diagonal is not a pair coefficient
                for (uint64_t w : row) if (w) det = false;
                std::fill(row.begin(), row.end(), 0);          // back to zero for the next column
                brow_on[(size_t)i] = 0;
            }
            brows.clear();
            if (!det) bad.push_back(t);
        }
        tw0 = tw1;
    }
    return bad;
}

GaussZ4 gauss_z4(std::vector<uint8_t> alpha, std::vector<std::vector<uint64_t>> adj) {
    const int m = (int)alpha.size();
    if ((int)adj.size() != m) throw std::logic_error("xdem gauss_z4: alpha / adj size mismatch (engine bug)");
    const size_t MW = std::max<size_t>(1, words_for(m));
    for (auto& row : adj) row.resize(MW, 0);
    GaussZ4 g;
    int K = 0;                                                   // the constant term (mod 4)
    std::vector<uint8_t> alive((size_t)m, 1);
    auto nbrs = [&](int i) {
        std::vector<int> v;
        for (size_t w = 0; w < MW; ++w)
            for (uint64_t b = adj[(size_t)i][w]; b; b &= b - 1) v.push_back((int)(w * 64 + (size_t)__builtin_ctzll(b)));
        return v;
    };
    auto toggle = [&](int k, int l) {
        adj[(size_t)k][(size_t)l >> 6] ^= 1ull << (l & 63);
        adj[(size_t)l][(size_t)k >> 6] ^= 1ull << (k & 63);
    };
    auto remove = [&](int i) {
        for (int k : nbrs(i)) adj[(size_t)k][(size_t)i >> 6] &= ~(1ull << (i & 63));
        std::fill(adj[(size_t)i].begin(), adj[(size_t)i].end(), 0);
        alive[(size_t)i] = 0;
    };
    for (int i = 0; i < m; ++i) {
        if (!alive[(size_t)i]) continue;
        const int ai = alpha[(size_t)i] & 3;
        const std::vector<int> Ni = nbrs(i);
        remove(i);
        if (ai & 1) {                                            // (1 + i^{a}) / 2 = 2^{-1/2} e^{+-i pi/4}
            g.h += 1;
            g.p8 += ai == 1 ? 1 : 7;
            for (int k : Ni) alpha[(size_t)k] = (uint8_t)((alpha[(size_t)k] + 4 - ai) & 3);   // -a [L(u)]: linear
            for (size_t x = 0; x < Ni.size(); ++x)                                           // ... and 2a = 2 pairs
                for (size_t y = x + 1; y < Ni.size(); ++y) toggle(Ni[x], Ni[y]);
            continue;
        }
        if (Ni.empty()) {                                        // E (-1)^{(a/2) u_i}: 1 or 0
            if (ai == 2) { g.zero = true; return g; }
            continue;
        }
        // [L(u) = c]: u_j = c xor (xor_{S} u_k), a factor 1/2; substitute it into u_j's terms a_j u_j + 2 u_j M(u).
        const int c = ai >> 1, j = Ni[0];
        const std::vector<int> S(Ni.begin() + 1, Ni.end());
        const int aj = alpha[(size_t)j] & 3;
        const std::vector<int> M = nbrs(j);
        remove(j);
        g.h += 2;
        K = (K + aj * c) & 3;                                    // a_j [c xor y] = a_j c + a_j (1 - 2c) [y]
        for (int k : S) alpha[(size_t)k] = (uint8_t)((alpha[(size_t)k] + 4 + aj * (1 - 2 * c)) & 3);
        if (aj & 1)
            for (size_t x = 0; x < S.size(); ++x)
                for (size_t y = x + 1; y < S.size(); ++y) toggle(S[x], S[y]);
        for (int mm : M) alpha[(size_t)mm] = (uint8_t)((alpha[(size_t)mm] + 2 * c) & 3);   // 2 c M(u)
        for (int k : S)                                                                   // 2 y M(u), u_k^2 = u_k
            for (int mm : M) {
                if (k == mm) alpha[(size_t)k] = (uint8_t)((alpha[(size_t)k] + 2) & 3);
                else toggle(k, mm);
            }
    }
    g.p8 = (g.p8 + 2 * K) & 7;
    return g;
}

double column_expectation(const WriterModel& wm, int t) {
    const Skeleton& sk = *wm.sk;
    const Header& H = wm.header;
    const int A = (int)H.axes.size(), N = sk.N, r = H.r;
    if (t < 0 || t >= H.D + H.O || t >= (int)sk.targets.size() || (int)sk.axis_p0.size() != A)
        throw std::logic_error("xdem column_expectation: column out of range (engine bug)");
    auto getb = [](const std::vector<uint64_t>& w, size_t i) { return ((w[i >> 6] >> (i & 63)) & 1ull) != 0; };
    // P_t at the end of the deferred circuit (the oracle's construction), pulled back through the skeleton.
    Pauli P(N);
    int ny = 0;
    for (int j : sk.targets[(size_t)t]) {
        const auto& [b, q] = sk.reads[(size_t)j];
        if (b == 0) P.setx(q); else if (b == 1) { P.setx(q); P.setz(q); ++ny; } else P.setz(q);
    }
    if (sk.targets[(size_t)t].empty()) return 1.0;               // constant parity
    P.phase = ny & 3;
    for (size_t o = sk.ops.size(); o-- > 0;) conj_backward(P, sk.ops[o]);
    // B: X(U^dag P U) = sum_{a in B} X(U^dag Q_a U) (an echelon of the axes' X parts with combinations).
    const size_t NW = (size_t)(N + 63) / 64, AW = (size_t)A / 64 + 1;
    std::vector<std::vector<uint64_t>> rows, comb;
    std::vector<int> piv;
    auto reduce = [&](std::vector<uint64_t>& v, std::vector<uint64_t>& cb) {
        for (size_t i = 0; i < rows.size(); ++i)
            if (getb(v, (size_t)piv[i])) {
                for (size_t w = 0; w < NW; ++w) v[w] ^= rows[i][w];
                for (size_t w = 0; w < AW; ++w) cb[w] ^= comb[i][w];
            }
    };
    for (int a = 0; a < A; ++a) {
        std::vector<uint64_t> v = sk.axis_p0[(size_t)a].x, cb(AW, 0);
        v.resize(NW, 0);
        cb[(size_t)a >> 6] |= 1ull << (a & 63);
        reduce(v, cb);
        int h = -1;
        for (size_t w = 0; w < NW && h < 0; ++w) if (v[w]) h = (int)(w * 64 + (size_t)__builtin_ctzll(v[w]));
        if (h < 0) continue;
        rows.push_back(std::move(v)); comb.push_back(std::move(cb)); piv.push_back(h);
    }
    std::vector<uint64_t> x = P.x, cB(AW, 0);
    x.resize(NW, 0);
    reduce(x, cB);
    for (uint64_t w : x) if (w) return 0.0;                      // not in the span: every term vanishes, <P> = 0
    // W = (U^dag P U)(U^dag Q_B U) = c Z(z): c = i^{W.phase} (U^dag g U = +Z(z) stabilizes |0>).
    Pauli W = P;
    int bB = 0;
    std::vector<uint64_t> kB(std::max<size_t>(1, words_for(r)), 0);
    for (int a = 0; a < A; ++a)
        if (getb(cB, (size_t)a)) {
            W = Pauli::multiply(W, sk.axis_p0[(size_t)a]);
            bB ^= H.axes[(size_t)a].b;
            for (size_t w = 0; w < kB.size() && w < H.axes[(size_t)a].k.size(); ++w) kB[w] ^= H.axes[(size_t)a].k[w];
        }
    for (uint64_t w : W.x)
        if (w) throw std::logic_error("xdem column_expectation: P Q_B is not X-free (engine bug)");
    std::vector<int> n((size_t)A, 0);                            // n_a mod 8 (as nondeterministic_columns)
    for (const Group& gr : H.groups)
        for (size_t vi = 0; vi < gr.v_axis.size(); ++vi) {
            const auto [a, s] = gr.v_axis[vi];
            if (a < 0) continue;
            n[(size_t)a] = ((n[(size_t)a] + s * gate_rotation_count(gr.gate, (uint32_t)(vi + 1))) % 8 + 8) % 8;
        }
    int ph8 = 2 * (W.phase & 3) + 4 * bB, C0 = 0;                // the phase of <P>, in units of pi/4
    // Phi(u) = 2[kappa_B.u] + sum_{a in A_P} n_a [b_a xor kappa_a.u] = C0 + sum w_a [kappa_a.u] + 2[kappa_B.u].
    // Restricted to the variables it touches (local ids), then alpha_i and the pair matrix as nondeterministic_columns.
    std::vector<int> loc((size_t)r, -1), vars;
    auto lid = [&](int i) { if (loc[(size_t)i] < 0) { loc[(size_t)i] = (int)vars.size(); vars.push_back(i); } return loc[(size_t)i]; };
    std::vector<std::pair<int, int>> wa_axes;                    // (axis, w_a) for a in A_P with w_a != 0
    for (int a = 0; a < A; ++a) {
        if (!getb(H.axes[(size_t)a].img.w, (size_t)t)) continue;  // a not in A_P
        ph8 -= n[(size_t)a];                                     // e^{-i pi n_a / 4}
        C0 += n[(size_t)a] * H.axes[(size_t)a].b;
        const int wa = ((n[(size_t)a] * (H.axes[(size_t)a].b ? -1 : 1)) % 4 + 4) % 4;
        if (!wa) continue;
        wa_axes.push_back({a, wa});
        for (int i = 0; i < r; ++i) if (getb(H.axes[(size_t)a].k, (size_t)i)) lid(i);
    }
    for (int i = 0; i < r; ++i) if (getb(kB, (size_t)i)) lid(i);
    const int m = (int)vars.size();
    std::vector<uint8_t> alpha((size_t)m, 0);
    std::vector<std::vector<uint64_t>> adj((size_t)m, std::vector<uint64_t>(std::max<size_t>(1, words_for(m)), 0));
    for (int i = 0; i < r; ++i) if (getb(kB, (size_t)i)) alpha[(size_t)loc[(size_t)i]] = 2;
    for (const auto& [a, wa] : wa_axes) {
        std::vector<int> ka;
        for (int i = 0; i < r; ++i) if (getb(H.axes[(size_t)a].k, (size_t)i)) ka.push_back(loc[(size_t)i]);
        for (int li : ka) alpha[(size_t)li] = (uint8_t)((alpha[(size_t)li] + wa) & 3);
        if (wa & 1)                                              // [kappa.u] = sum u_i - 2 sum_{i<j} u_i u_j
            for (size_t x1 = 0; x1 < ka.size(); ++x1)
                for (size_t y1 = x1 + 1; y1 < ka.size(); ++y1) {
                    adj[(size_t)ka[x1]][(size_t)ka[y1] >> 6] ^= 1ull << (ka[y1] & 63);
                    adj[(size_t)ka[y1]][(size_t)ka[x1] >> 6] ^= 1ull << (ka[x1] & 63);
                }
    }
    const GaussZ4 G = gauss_z4(std::move(alpha), std::move(adj));
    if (G.zero) return 0.0;
    ph8 = ((ph8 + 2 * C0 + G.p8) % 8 + 8) % 8;
    if (ph8 != 0 && ph8 != 4)
        throw std::logic_error("xdem column_expectation: a non-real expectation (engine bug)");
    const double mag = (G.h & 1) ? std::ldexp(std::sqrt(0.5), -(G.h - 1) / 2) : std::ldexp(1.0, -G.h / 2);
    return ph8 == 0 ? mag : -mag;
}

// ── Task 4: LeanLineAccumulator ──────────────────────────────────────────────────────────────────────────────────
size_t LeanLineAccumulator::KeyHash::operator()(const LineKey& k) const noexcept {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t)k.side;
    auto mix = [&](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
    mix(k.f.size()); for (int32_t v : k.f) mix((uint64_t)(uint32_t)v);
    mix(k.tau.size()); for (int32_t v : k.tau) mix((uint64_t)(uint32_t)v);
    mix(k.lt.size()); for (const auto& [a, e] : k.lt) mix(((uint64_t)(uint32_t)a << 8) | e);
    mix(k.gt.size()); for (const auto& [a, e] : k.gt) mix(((uint64_t)(uint32_t)a << 8) | e);
    return (size_t)h;
}

void LeanLineAccumulator::set_axes(const std::vector<Axis>& axes) { axes_ = &axes; merged_ = false; }

void LeanLineAccumulator::add(LineKey&& k, double p, const TimeKey& t) {
    if (!(p > 0.0) || k.empty()) return;
    if (!k.lt.empty() || !k.gt.empty()) any_rot_ = true;   // a line's key is its members' key
    auto it = ids_.find(k);
    if (it == ids_.end()) {
        it = ids_.emplace(std::move(k), (int)key_.size()).first;
        key_.push_back(&it->first);
    }
    mechs_.push_back({it->second, p, t});
    merged_ = false;
}

bool LeanLineAccumulator::any_rotation() const { return any_rot_; }

size_t LeanLineAccumulator::num_lines() { merge(); return line_key_.size(); }

namespace {
struct AxisListHash {
    size_t operator()(const std::vector<int32_t>& v) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t)v.size();
        for (int32_t x : v) h ^= (uint64_t)(uint32_t)x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};
// A deliberate lean-only divergence (header: (i), plan M10).
[[noreturn]] void axis_beyond() {
    throw std::logic_error("xdem LeanLineAccumulator: an axis id beyond set_axes (engine bug)");
}
}  // namespace

// LineAccumulator::merge (xdem_writer.cpp), the same decisions on other containers. The oracle's visibility test for
// an open line of class c against the event's class k is
//     visible(c, k)  <=>  xor_{a in S} img_a != 0,   S = (tau_c & odd_k) xor (tau_k & odd_c)
// (its sorted-vector meets XOR an image per meet in both halves, so an axis in both cancels: exactly the set S).
//
// Zero-image axis prune (exact). Let Z = {a : img_a != 0}. Then xor_{a in S} img_a = xor_{a in S & Z} img_a, and
//     S & Z = ((tau_c & Z) & (odd_k & Z)) xor ((tau_k & Z) & (odd_c & Z)),
// so visible(c, k) depends on the classes only through (tau & Z, odd & Z). Hence:
//  (i)  classes are interned on (tau & Z, odd & Z). The oracle's classes (tau, odd) refine these; every oracle class
//       inside one lean class has the same visibility against every event, so the oracle closes all of them at the
//       same events, which is what closing the lean class does. For two oracle classes in one lean class, S & Z is
//       empty (the two halves coincide), so the oracle never closes one by the other: visible(kc, kc) is false
//       (s = (t & o) xor (t & o) = 0 word by word), so the event's own class is never closed, with no special case.
//  (ii) the per-axis indexes list classes under the axes of Z only. A class sharing no axis of Z with the event (tau_c
//       with odd_k, or odd_c with tau_k) has S & Z empty, so it is not visible: never collecting it drops only
//       non-closing candidates.
//  (iii) the remaining effect of collecting is the expiry prune (a class past its keys' last member leaves the
//       indexes, its lines left open). No mechanism after its expire joins one of its lines (expire = the last sorted
//       index of a member of any of its open lines' keys), and every other read of a line's openness is a later
//       mechanism of its key, so when (or whether) an expired class is pruned or closed never changes a decision.
// So the lines, their members and their p (accumulated in time order, as in the oracle) are the oracle's.
// The visibility predicate itself runs on bit words over the compacted Z axes: per word s = (tau_c & odd_k) xor
// (tau_k & odd_c), then xor img_a over the set bits of s, then image != 0: the same predicate (amendment 1).
void LeanLineAccumulator::merge() {
    if (merged_) return;
    static const std::vector<Axis> kNoAxes;
    const std::vector<Axis>& axes = axes_ ? *axes_ : kNoAxes;
    const size_t M = mechs_.size(), K = key_.size(), A = axes.size();
    std::vector<size_t> order(M);
    for (size_t i = 0; i < M; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return mechs_[a].t < mechs_[b].t; });
    std::vector<size_t> last_of(K, 0);
    for (size_t r = 0; r < M; ++r) last_of[(size_t)mechs_[order[r]].key] = r;

    // Z: the axes with a nonzero readout image, renumbered 0..nz-1 in ascending order (so sorted lists stay sorted).
    const size_t W = axes.empty() ? 0 : axes[0].img.w.size();
    std::vector<int32_t> zid(A, -1);
    std::vector<uint64_t> zimg;                             // nz x W: the images of Z, contiguous
    size_t nz = 0;
    for (size_t a = 0; a < A; ++a) {
        const auto& w = axes[a].img.w;
        // a deliberate lean-only divergence (header: (ii), Task 4 review M2): the oracle does not check
        if (w.size() != W) throw std::logic_error("xdem LeanLineAccumulator: axis images of unequal width (engine bug)");
        bool any = false;
        for (uint64_t x : w) any |= x != 0;
        if (!any) continue;
        zid[a] = (int32_t)nz++;
        zimg.insert(zimg.end(), w.begin(), w.end());
    }
    const size_t AW = (nz + 63) / 64;

    // classes (tau & Z, odd & Z), interned once per distinct key
    std::vector<int32_t> c_ax;                              // per class: tau & Z, then odd & Z (compact ids)
    std::vector<size_t> c_off{0}, c_mid;                    // class c: tau [c_off[c], c_mid[c]), odd [c_mid[c], c_off[c+1])
    std::vector<uint64_t> c_bits;                           // per class: AW words of tau & Z, then AW of odd & Z
    std::vector<int> cls_of_key(K);
    {
        std::unordered_map<std::vector<int32_t>, int, AxisListHash> intern;
        intern.reserve(K);
        std::vector<int32_t> u;
        for (size_t k = 0; k < K; ++k) {
            const LineKey& key = *key_[k];
            u.clear();
            for (int32_t a : key.tau) {
                if (a < 0 || (size_t)a >= A) axis_beyond();
                if (zid[(size_t)a] >= 0) u.push_back(zid[(size_t)a]);
            }
            const size_t ntau = u.size();
            u.push_back(-1);                                // separator
            for (int32_t a : line_odd_axes(key)) {
                if (a < 0 || (size_t)a >= A) axis_beyond();
                if (zid[(size_t)a] >= 0) u.push_back(zid[(size_t)a]);
            }
            auto it = intern.find(u);
            if (it == intern.end()) {
                const int c = (int)c_mid.size();
                c_ax.insert(c_ax.end(), u.begin(), u.begin() + (std::ptrdiff_t)ntau);
                c_mid.push_back(c_ax.size());
                c_ax.insert(c_ax.end(), u.begin() + (std::ptrdiff_t)ntau + 1, u.end());
                c_off.push_back(c_ax.size());
                c_bits.resize(c_bits.size() + 2 * AW, 0);
                uint64_t* tb = c_bits.data() + (size_t)c * 2 * AW;
                for (size_t i = c_off[(size_t)c]; i < c_mid[(size_t)c]; ++i) tb[(size_t)c_ax[i] >> 6] |= 1ull << (c_ax[i] & 63);
                for (size_t i = c_mid[(size_t)c]; i < c_off[(size_t)c + 1]; ++i)
                    tb[AW + ((size_t)c_ax[i] >> 6)] |= 1ull << (c_ax[i] & 63);
                it = intern.emplace(u, c).first;
            }
            cls_of_key[k] = it->second;
        }
    }
    const size_t NC = c_mid.size();
    struct Class {
        std::vector<int> lines;
        bool listed = false;
        int gen = 0;
        size_t expire = 0;
    };
    std::vector<Class> cls(NC);
    std::vector<int> open_line(K, -1), line_cls, cand, stamp(NC, -1);
    std::vector<char> is_open;
    std::vector<std::vector<std::pair<int, int>>> by_tau(nz), by_odd(nz);   // compact axis -> (class, generation)
    line_key_.clear(); line_p_.clear();
    int now = 0;
    std::vector<uint64_t> acc(W, 0);
    auto visible = [&](int c, int k) {                     // the oracle's predicate, on bit words (see above)
        const uint64_t* tc = c_bits.data() + (size_t)c * 2 * AW;
        const uint64_t* oc = tc + AW;
        const uint64_t* tk = c_bits.data() + (size_t)k * 2 * AW;
        const uint64_t* ok = tk + AW;
        bool any = false;
        for (size_t w = 0; w < AW; ++w) {
            for (uint64_t s = (tc[w] & ok[w]) ^ (tk[w] & oc[w]); s; s &= s - 1) {
                const uint64_t* im = zimg.data() + (w * 64 + (size_t)__builtin_ctzll(s)) * W;
                for (size_t q = 0; q < W; ++q) acc[q] ^= im[q];
                any = true;
            }
        }
        if (!any) return false;
        bool nzr = false;
        for (size_t q = 0; q < W; ++q) { nzr |= acc[q] != 0; acc[q] = 0; }
        return nzr;
    };
    auto list_class = [&](int c) {
        Class& C = cls[(size_t)c];
        if (C.listed) return;
        C.listed = true;
        ++C.gen;
        for (size_t i = c_off[(size_t)c]; i < c_mid[(size_t)c]; ++i) by_tau[(size_t)c_ax[i]].push_back({c, C.gen});
        for (size_t i = c_mid[(size_t)c]; i < c_off[(size_t)c + 1]; ++i) by_odd[(size_t)c_ax[i]].push_back({c, C.gen});
    };
    auto collect = [&](std::vector<std::vector<std::pair<int, int>>>& idx, size_t from, size_t to, size_t r) {
        for (size_t i = from; i < to; ++i) {
            auto& v = idx[(size_t)c_ax[i]];
            size_t keep = 0;
            for (const auto& [c, g] : v) {
                Class& C = cls[(size_t)c];
                if (!C.listed || C.gen != g) continue;       // emptied (or re-listed) since: pruned
                if (C.expire <= r) { C.lines.clear(); C.listed = false; continue; }   // expired: pruned for good
                v[keep++] = {c, g};
                if (stamp[(size_t)c] != now) { stamp[(size_t)c] = now; cand.push_back(c); }
            }
            v.resize(keep);
        }
    };
    for (size_t r = 0; r < M; ++r) {
        const Mech& m = mechs_[order[r]];
        int id = open_line[(size_t)m.key];
        if (id >= 0) {
            double& P = line_p_[(size_t)id];
            P = xor_compose(P, m.p);                       // the oracle's own function (bitwise equal)
        } else {
            id = (int)line_key_.size();
            line_key_.push_back(m.key);
            line_p_.push_back(m.p);
            is_open.push_back(1);
            open_line[(size_t)m.key] = id;
            const int c = cls_of_key[(size_t)m.key];
            line_cls.push_back(c);
            Class& C = cls[(size_t)c];
            const size_t last = last_of[(size_t)m.key];
            if (C.lines.empty() || last > C.expire) C.expire = last;
            C.lines.push_back(id);
            list_class(c);
        }
        // this event closes every open line whose swap with m's key is visible
        const int kc = line_cls[(size_t)id];
        ++now;
        cand.clear();
        collect(by_tau, c_mid[(size_t)kc], c_off[(size_t)kc + 1], r);   // tau_c meets odd_k
        collect(by_odd, c_off[(size_t)kc], c_mid[(size_t)kc], r);       // odd_c meets tau_k
        for (int c : cand) {
            if (!visible(c, kc)) continue;                 // own class (c == kc): S & Z is empty, never visible
            Class& C = cls[(size_t)c];
            for (int l : C.lines)
                if (is_open[(size_t)l]) { is_open[(size_t)l] = 0; open_line[(size_t)line_key_[(size_t)l]] = -1; }
            C.lines.clear();
            C.listed = false;                              // pruned from the indexes on their next scan
        }
    }
    merged_ = true;
}

File LeanLineAccumulator::to_file(int D, int O, const Canon& C, const std::vector<Axis>& axes) {
    merge();
    File F = file_begin(D, O, C, axes);              // shared with LineAccumulator::to_file (final review M3)
    for (size_t i = 0; i < line_key_.size(); ++i) file_add_line(F, *key_[(size_t)line_key_[i]], line_p_[i]);
    return F;
}

namespace {
inline void put_i(std::string& s, long long v) {
    char b[24];
    const auto r = std::to_chars(b, b + sizeof b, v);
    s.append(b, r.ptr);
}
}  // namespace

// xdem::emit (xdem_file.cpp) statement for statement, the same tokens in the same order; only the string building
// differs (to_chars into one reserved buffer instead of std::to_string temporaries).
std::string emit_fast(const File& F) {
    std::string out;
    out.reserve(64 + 48 * F.axes.size() +
                8 * (F.num_lines() * 4 + F.flip_cols.size() + F.tau_ax.size() + F.rot.size() + F.late.size()) +
                24 * F.num_lines());
    out += "xdem 1 detectors "; put_i(out, F.D);
    out += " observables "; put_i(out, F.O);
    out += " gauge "; put_i(out, F.r);
    out += " hidden "; put_i(out, F.m);
    out += " tc "; put_i(out, F.tc);
    out += '\n';
    auto target = [&](int j) {
        if (j < F.D) { out += 'D'; put_i(out, j); } else { out += 'L'; put_i(out, j - F.D); }
    };
    for (size_t a = 0; a < F.axes.size(); ++a) {
        const Axis& ax = F.axes[a];
        out += "axis Q"; put_i(out, (long long)a);
        if (ax.b) out += " sign -";
        const char* kw = " img";
        for (size_t w = 0; w < ax.img.w.size(); ++w)
            for (uint64_t x = ax.img.w[w]; x; x &= x - 1) {
                out += kw; kw = ""; out += ' '; target((int)(w * 64 + (size_t)__builtin_ctzll(x)));
            }
        kw = " gauge";
        for (size_t w = 0; w < ax.k.size(); ++w)
            for (uint64_t x = ax.k[w]; x; x &= x - 1) {
                out += kw; kw = ""; out += " g"; put_i(out, (long long)(w * 64 + (size_t)__builtin_ctzll(x)));
            }
        kw = " hidden";
        for (int32_t h : ax.hidden) { out += kw; kw = ""; out += " h"; put_i(out, h); }
        out += '\n';
    }
    char pb[40];
    for (size_t i = 0; i < F.num_lines(); ++i) {
        std::snprintf(pb, sizeof pb, "%.17g", F.p[i]);   // emit's probability spelling
        out += "error("; out += pb; out += ')';
        for (uint64_t t = F.flip_off[i]; t < F.flip_off[i + 1]; ++t) { out += ' '; target(F.flip_cols[t]); }
        if (F.tau_off[i] < F.tau_off[i + 1]) {
            out += " x";
            for (uint64_t t = F.tau_off[i]; t < F.tau_off[i + 1]; ++t) { out += " Q"; put_i(out, F.tau_ax[t]); }
        }
        auto rots = [&](const char* kw2, const std::vector<uint64_t>& off, const std::vector<Rot>& v) {
            if (off[i] == off[i + 1]) return;
            out += kw2;
            for (uint64_t t = off[i]; t < off[i + 1]; ++t) { out += v[t].e == 1 ? " +Q" : " -Q"; put_i(out, v[t].axis); }
        };
        rots(" rot", F.rot_off, F.rot);
        rots(" late", F.late_off, F.late);
        out += '\n';
    }
    return out;
}

// ── Task 6: the sparse line-key builder (== the export's dense xdem_key; see the header) ─────────────────────────────
// Why it is the same key. The dense builder computes, per mechanism: f = XOR of the flip ids (a bitset), Elt / Egt =
// sum over (g, x_g) and residue terms (v, e) of -(sign * e) mod 4 at axis a (fault_rotations), then for each field and
// each axis with exponent 2: f ^= img_a and the entry -> 0 (fold_field), then lt / gt = the nonzero entries by axis
// ascending and f's set bits ascending. Here the same sums are taken at the same axes (mod-4 addition and XOR commute,
// so the order of the terms is immaterial), the folds read img_a only where it is nonzero (XOR with a zero word is
// the identity), and the outputs are read off the touched axes / words sorted ascending, which are a superset of the
// nonzero ones (an untouched entry is zero). tau and x are copied from sig in its order, as xdem_key / xdem_x do.
void LeanKeyBuilder::init(const Header& H, const std::vector<Axis>& axes, const std::vector<std::pair<int, int>>& xid,
                          int ncols, int Gl) {
    H_ = &H;
    axes_ = &axes;
    xid_ = &xid;
    ncols_ = ncols;
    nx_ = (int)xid.size();
    Gl_ = Gl;
    const size_t W = words_for((long long)ncols), A = axes.size();
    img_nz_.assign(A, {});
    for (size_t a = 0; a < A; ++a) {
        if (axes[a].img.w.size() != W)
            throw std::logic_error("xdem LeanKeyBuilder: an axis image of width other than D + O (engine bug)");
        for (size_t w = 0; w < W; ++w)
            if (axes[a].img.w[w]) img_nz_[a].push_back((uint32_t)w);
    }
    elt_.assign(A, 0);
    egt_.assign(A, 0);
    axis_hit_.assign(A, 0);
    axes_list_.clear();
    f_.assign(W, 0);
    word_hit_.assign(W, 0);
    words_list_.clear();
    dirty_ = false;
}

void LeanKeyBuilder::reset_scratch() {
    std::fill(elt_.begin(), elt_.end(), 0);
    std::fill(egt_.begin(), egt_.end(), 0);
    std::fill(axis_hit_.begin(), axis_hit_.end(), 0);
    std::fill(f_.begin(), f_.end(), 0);
    std::fill(word_hit_.begin(), word_hit_.end(), 0);
    axes_list_.clear();
    words_list_.clear();
    dirty_ = false;
}

LineKey LeanKeyBuilder::build(const std::vector<int32_t>& sig, int side) {
    if (dirty_) reset_scratch();
    dirty_ = true;
    LineKey K;
    K.side = side;
    x_.clear();
    for (int32_t id : sig) {
        if (id < 0) throw std::logic_error("xdem LeanKeyBuilder: a negative converter id (engine bug)");
        if (id < ncols_) {
            const uint32_t w = (uint32_t)id >> 6;
            touch_word(w);
            f_[w] ^= 1ull << (id & 63);
        } else if (id < ncols_ + nx_) {
            const auto& [g, bit] = (*xid_)[(size_t)(id - ncols_)];
            if (x_.empty() || x_.back().first != g) x_.push_back({g, 0u});
            x_.back().second |= 1u << bit;
        } else {
            K.tau.push_back(id - ncols_ - nx_);
        }
    }
    const int32_t A = (int32_t)elt_.size();
    for (const auto& [g, xg] : x_) {                   // fault_rotations, at the axes hit
        const Group& gr = H_->groups[(size_t)g];
        std::vector<uint8_t>& E = g < Gl_ ? elt_ : egt_;
        for (const auto& [v, e] : residue_terms(gr.gate, xg)) {
            const auto& [a, sgn] = gr.v_axis[(size_t)(v - 1)];
            if (a < 0) continue;                       // an identity axis: a global phase
            if (a >= A) throw std::logic_error("xdem LeanKeyBuilder: an axis id beyond the axes (engine bug)");
            if (!axis_hit_[(size_t)a]) { axis_hit_[(size_t)a] = 1; axes_list_.push_back(a); }
            E[(size_t)a] = (uint8_t)((E[(size_t)a] + 4 - ((sgn * e) & 3)) & 3);   // -(sign * e) mod 4
        }
    }
    std::sort(axes_list_.begin(), axes_list_.end());
    const std::vector<Axis>& axes = *axes_;
    for (int32_t a : axes_list_) {                     // fold_field on both fields, then the entries (ascending)
        uint8_t& lt = elt_[(size_t)a];
        uint8_t& gt = egt_[(size_t)a];
        const int folds = (lt == 2) + (gt == 2);        // two folds of the same image cancel
        if (folds == 1)
            for (uint32_t w : img_nz_[(size_t)a]) { touch_word(w); f_[w] ^= axes[(size_t)a].img.w[w]; }
        if (lt == 2) lt = 0;
        if (gt == 2) gt = 0;
        if (lt) K.lt.push_back({a, lt});
        if (gt) K.gt.push_back({a, gt});
        lt = 0;
        gt = 0;
        axis_hit_[(size_t)a] = 0;
    }
    axes_list_.clear();
    std::sort(words_list_.begin(), words_list_.end());
    for (uint32_t w : words_list_) {
        for (uint64_t b = f_[w]; b; b &= b - 1) K.f.push_back((int32_t)(w * 64 + (uint32_t)__builtin_ctzll(b)));
        f_[w] = 0;
        word_hit_[w] = 0;
    }
    words_list_.clear();
    dirty_ = false;
    return K;
}

namespace {
std::atomic<bool> g_key_check{false};
std::atomic<long long> g_key_compared{0}, g_key_mismatched{0};
}  // namespace

void set_lean_key_check_for_test(bool on) {
    if (on) {
        g_key_compared.store(0, std::memory_order_relaxed);
        g_key_mismatched.store(0, std::memory_order_relaxed);
    }
    g_key_check.store(on, std::memory_order_relaxed);
}
bool lean_key_check_on() { return g_key_check.load(std::memory_order_relaxed); }
void lean_key_check_record(bool same) {
    g_key_compared.fetch_add(1, std::memory_order_relaxed);
    if (!same) g_key_mismatched.fetch_add(1, std::memory_order_relaxed);
}
std::pair<long long, long long> lean_key_check_counts() {
    return {g_key_compared.load(std::memory_order_relaxed), g_key_mismatched.load(std::memory_order_relaxed)};
}

}  // namespace qeccore::xdem
