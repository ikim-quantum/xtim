// xdem_sampler.cpp — file-only sampler + status + scorer for an xdem file or a conventional DEM
// (qeccore/xdem_sampler.hpp; SPEC §1.2, §1.3, §4.7). Started as a copy of the dc-DEM v1 sampler (19f2f78): the noise
// walk, stream, chains, postselection, fast path, law cache and C table are that code's; the shot row's X-pattern words
// and the residue lookup (accumulate_E) are replaced by the per-shot gather of the fired extended lines, their sort
// into file order and the cocycle composition with a sign register (compose_ext).
#include "qeccore/xdem_sampler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#include "qeccore/keyed_rng.hpp"
#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(__BMI2__)
#include <immintrin.h>   // _pext_u64 (the span test's pivot-bit gather)
#endif

namespace qeccore::xdem {

namespace {

inline uint64_t hash_words(const uint64_t* w, int n) noexcept {
    uint64_t h = 0x243F6A8885A308D3ull;
    for (int i = 0; i < n; ++i) h = krng::mix(h ^ (w[i] + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1)));
    return h;
}

// The C tables' hash of a canonical form (internal; never an output): two multiply-xorshift lanes over the words, one
// mix at the end (the serial mix per word of hash_words costs ~4x on the ~100-word forms of a large C)
inline uint64_t hash_c(const uint64_t* w, size_t n) noexcept {
    uint64_t a = 0x243F6A8885A308D3ull ^ (uint64_t)n, b = 0x13198A2E03707344ull;
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        a = (a ^ w[i]) * 0x9E3779B97F4A7C15ull;
        b = (b ^ w[i + 1]) * 0xC2B2AE3D27D4EB4Full;
        a ^= a >> 29;
        b ^= b >> 29;
    }
    if (i < n) {
        a = (a ^ w[i]) * 0x9E3779B97F4A7C15ull;
        a ^= a >> 29;
    }
    return krng::mix(a ^ (b * 0xD6E8FEB86659FD93ull));
}

inline int highest(const uint64_t* w, int n) {
    for (int i = n; i-- > 0;)
        if (w[i]) return i * 64 + 63 - __builtin_clzll(w[i]);
    return -1;
}

// rows[0, n) (cw words each; NW > 0: cw = NW) in place -> their fully reduced echelon (pivot = highest bit), the rows
// taken in order and the independent ones compacted to rows[0, k); pm (cw words) <- the pivot mask, pidx[p] = the row
// of pivot p. Returns k. A row is reduced by XOR-ing the row of each pivot bit it holds (the rows are fully reduced, so
// one XOR never changes another pivot bit).
template <int NW>
inline int rref_high(uint64_t* rows, int n, int cw_rt, uint64_t* pm, int* pidx) {
    const int cw = NW ? NW : cw_rt;
    for (int q = 0; q < cw; ++q) pm[q] = 0;
    int k = 0;
    for (int j = 0; j < n; ++j) {
        uint64_t* v = rows + (size_t)j * cw;
        for (int u = 0; u < cw; ++u)
            for (uint64_t y = v[u] & pm[u]; y; y &= y - 1) {
                const uint64_t* c = rows + (size_t)pidx[u * 64 + __builtin_ctzll(y)] * cw;
                for (int q = 0; q < cw; ++q) v[q] ^= c[q];
            }
        int p = -1;
        for (int u = cw; u-- > 0;)
            if (v[u]) {
                p = u * 64 + 63 - __builtin_clzll(v[u]);
                break;
            }
        if (p < 0) continue;
        const uint64_t pb = 1ull << (p & 63);
        for (int i = 0; i < k; ++i) {
            uint64_t* c = rows + (size_t)i * cw;
            if (c[p >> 6] & pb)
                for (int q = 0; q < cw; ++q) c[q] ^= v[q];
        }
        if (j != k) std::copy(v, v + cw, rows + (size_t)k * cw);
        pidx[p] = k;
        pm[p >> 6] |= pb;
        ++k;
    }
    return k;
}

// the rows of rref_high's echelon in descending pivot order (ord[0, k))
inline void rows_desc(const uint64_t* pm, int cw, const int* pidx, int* ord) {
    int n = 0;
    for (int u = cw; u-- > 0;)
        for (uint64_t y = pm[u]; y;) {
            const int b = 63 - __builtin_clzll(y);
            ord[n++] = pidx[u * 64 + b];
            y ^= 1ull << b;
        }
}

// The pivot-indexed reduction of v (cw words; NW > 0: cw = NW) by an echelon of independent vectors whose pivot is
// their LOWEST bit (pm: the pivot bits, pidx[p] = the vector of pivot p; vectors pv / combinations pc, cw words each),
// tracking the combination c: XOR the vector owning v's lowest pivot bit until none is left. Every vector's bits lie at
// or above its pivot, so the lowest pivot bit of v only rises. The result (v reduced to no pivot bit, c) is unique
// (the vectors restricted to the pivot bits are triangular), so it equals any other order's, e.g. the scan over every
// earlier pivot in insertion order that this replaces. pc = nullptr: no combination.
template <int NW>
inline void reduce_low(uint64_t* v, uint64_t* c, int cw_rt, const uint64_t* pm, const int* pidx, const uint64_t* pv,
                       const uint64_t* pc) {
    const int cw = NW ? NW : cw_rt;
    for (int u = 0; u < cw;) {
        const uint64_t h = v[u] & pm[u];
        if (!h) {
            ++u;
            continue;
        }
        const size_t q = (size_t)pidx[u * 64 + __builtin_ctzll(h)];
        const uint64_t* a = pv + q * (size_t)cw;
        for (int w = u; w < cw; ++w) v[w] ^= a[w];
        if (pc) {
            const uint64_t* b = pc + q * (size_t)cw;
            for (int w = 0; w < cw; ++w) c[w] ^= b[w];
        }
    }
}

// one-word reduce_low: pmask = the pivot bits, pidx[b] = the vector of pivot bit b
inline void reduce_low1(uint64_t& v, uint64_t& c, uint64_t pmask, const int* pidx, const uint64_t* pv,
                        const uint64_t* pc) {
    for (uint64_t h; (h = v & pmask) != 0;) {
        const int q = pidx[__builtin_ctzll(h)];
        v ^= pv[q];
        c ^= pc[q];
    }
}

// Fully reduced RREF (pivot = highest bit, rows by descending pivot) of `rows` (each nw words), in place.
void rref_desc(std::vector<std::vector<uint64_t>>& rows, int nw, std::vector<int>& piv) {
    std::vector<std::vector<uint64_t>> B;
    std::vector<int> P;
    for (auto& v : rows) {
        for (size_t i = 0; i < B.size(); ++i)
            if ((v[(size_t)P[i] >> 6] >> (P[i] & 63)) & 1ull)
                for (int w = 0; w < nw; ++w) v[(size_t)w] ^= B[i][(size_t)w];
        const int p = highest(v.data(), nw);
        if (p < 0) continue;
        for (size_t i = 0; i < B.size(); ++i)
            if ((B[i][(size_t)p >> 6] >> (p & 63)) & 1ull)
                for (int w = 0; w < nw; ++w) B[i][(size_t)w] ^= v[(size_t)w];
        B.push_back(v);
        P.push_back(p);
    }
    std::vector<size_t> ord(B.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return P[a] > P[b]; });
    rows.clear();
    piv.clear();
    for (size_t i : ord) {
        rows.push_back(std::move(B[i]));
        piv.push_back(P[i]);
    }
}

// A shot's fired ids (or extended-record offsets, increasing with the id) into ascending (file) order: the walk is
// chain-major, each chain's members ascending, so a list is a few ascending runs (insertion sort for short lists).
template <class V>
inline void sort_ids(V* v, uint32_t len) {
    if (len == 2) {
        const V x = v[0], y = v[1];
        v[0] = std::min(x, y);
        v[1] = std::max(x, y);
    } else if (len > 32) {
        std::sort(v, v + len);
    } else {
        for (uint32_t i = 1; i < len; ++i)
            for (uint32_t j = i; j > 0 && v[j - 1] > v[j]; --j) std::swap(v[j - 1], v[j]);
    }
}

// The scorer's two refusals, shared by its one-word and multiword loops (same exception, same text on both).
[[noreturn]] void refuse_score_id(int id, size_t s) {
    throw std::out_of_range("XdemSampler.score: cspace id " + std::to_string(id) + " at shot " + std::to_string(s) +
                            " is not in this sampler's table");
}
[[noreturn]] void refuse_score_flip() { throw std::invalid_argument("XdemSampler.score: flips must be 0 or 1"); }

// GSL: the kernel and the W phase of a law in graph-state form, shared by post_fast / post_multi (the per-shot law)
// and the optimal correction's law of [T, K]. The law matrix's columns col_j (j < R) hold, in their R low bits, row j
// of the symmetric G = sum_{items} t t^T (kappa in image coordinates), above them the completing gauge part; its
// kernel is the law's W. The columns are eliminated in a fixed order, each reduced by the pivot owning the lowest
// pivot bit it holds until it holds none (an XOR adds bits above that bit only): the residual and the combination are
// the unique ones of the sequential reduction, so the kernel basis sigma_q (Kt[j] bit q = sigma_q[j], depj[q] = its
// dependent column) is the old mask-select one. The W phase of sigma, |{items : t.sigma odd}| / 2 + lin.sigma (mod 2)
// with |{...}| = sum_{i in sigma} d_i + 2 sum_{i < j in sigma} G_ij (mod 4) (d_i = |{items : t bit i}|; d1 its 2-bit
// plane, G_ii = its 1-bit plane): X = 2 |sigma & d1| + sum_{i in sigma} |G_i & sigma|, f = X / 2 + lin.sigma, taken as
// each kernel vector appears; omega gets bit depj[q] when f(sigma_q) = 1.
struct Law1 {   // one word per vector
    uint64_t Kt[64], omega;
    int depj[64], npv, nker, npvD;   // npvD: the pivots among the columns below `split`
};
inline void law_kernel1(const uint64_t* col, int R, int split, uint64_t d1, uint64_t lin, Law1& L) {
    uint64_t pv_vec[64], pv_cmb[64], pmask = 0;
    int pidx[64];
    L.npv = L.nker = L.npvD = 0;
    L.omega = 0;
    for (int j = 0; j < R; ++j) L.Kt[j] = 0;
    for (int j = 0; j < R; ++j) {
        if (j == split) L.npvD = L.npv;
        uint64_t v = col[j], c = 1ull << j;
        for (uint64_t h; (h = v & pmask) != 0;) {
            const int q = pidx[__builtin_ctzll(h)];
            v ^= pv_vec[q];
            c ^= pv_cmb[q];
        }
        if (v) {
            pv_vec[L.npv] = v;
            pv_cmb[L.npv] = c;
            pidx[__builtin_ctzll(v)] = L.npv;
            pmask |= v & (0 - v);
            ++L.npv;
            continue;
        }
        for (uint64_t cc = c; cc; cc &= cc - 1) L.Kt[__builtin_ctzll(cc)] |= 1ull << L.nker;
        unsigned X = 2u * (unsigned)__builtin_popcountll(c & d1);
        for (uint64_t y = c; y; y &= y - 1) X += (unsigned)__builtin_popcountll(col[__builtin_ctzll(y)] & c);
        if (((X >> 1) ^ (unsigned)__builtin_parityll(c & lin)) & 1u) L.omega |= 1ull << j;
        L.depj[L.nker++] = j;
    }
}
// cw words per vector (NW > 0: cw = NW); columns R - 1 .. 0 when desc, else 0 .. R - 1; pcol[i] = pivot i's column.
// Scratch: tv, tc, pmk (cw words), pvv, pvc (R x cw), pidx (64 cw), sw (cw). Kt (R x cw) is cleared here; om (cw
// words) gets the phase bits XOR-ed in.
template <int NW>
inline void law_kernelw(int cw_rt, int R, bool desc, const uint64_t* col, const uint64_t* d1, const uint64_t* lin,
                        uint64_t* tv, uint64_t* tc, uint64_t* pmk, uint64_t* pvv, uint64_t* pvc, int* pidx, int* sw,
                        uint64_t* Kt, int* depj, int* pcol, int& npv, int& nker, uint64_t* om) {
    const int cw = NW ? NW : cw_rt;
    npv = nker = 0;
    std::fill(Kt, Kt + (size_t)R * cw, 0ull);
    for (int q = 0; q < cw; ++q) pmk[q] = 0;
    for (int s = 0; s < R; ++s) {
        const int j = desc ? R - 1 - s : s;
        std::copy(col + (size_t)j * cw, col + (size_t)(j + 1) * cw, tv);
        for (int q = 0; q < cw; ++q) tc[q] = 0;
        tc[j >> 6] = 1ull << (j & 63);
        for (int u = 0; u < cw;) {   // the lowest pivot bit tv holds (words below u hold none)
            const uint64_t h = tv[u] & pmk[u];
            if (!h) {
                ++u;
                continue;
            }
            const int q = pidx[u * 64 + __builtin_ctzll(h)];
            const uint64_t* pv = pvv + (size_t)q * cw;
            const uint64_t* pc = pvc + (size_t)q * cw;
            for (int w = 0; w < cw; ++w) {
                tv[w] ^= pv[w];
                tc[w] ^= pc[w];
            }
        }
        int lw = -1;
        for (int q = 0; q < cw && lw < 0; ++q)
            if (tv[q]) lw = q;
        if (lw >= 0) {
            std::copy(tv, tv + cw, pvv + (size_t)npv * cw);
            std::copy(tc, tc + cw, pvc + (size_t)npv * cw);
            pidx[lw * 64 + __builtin_ctzll(tv[lw])] = npv;
            pmk[lw] |= tv[lw] & (0 - tv[lw]);
            pcol[npv++] = j;
            continue;
        }
        unsigned X = 0, L = 0;
        int ns = 0;   // sw <- the words sigma (tc) occupies: |G_i & sigma| reads only those
        for (int u = 0; u < cw; ++u) {
            if (!tc[u]) continue;
            sw[ns++] = u;
            X += 2u * (unsigned)__builtin_popcountll(tc[u] & d1[u]);
            L += (unsigned)__builtin_popcountll(tc[u] & lin[u]);
        }
        for (int a = 0; a < ns; ++a) {
            const int u = sw[a];
            for (uint64_t y = tc[u]; y; y &= y - 1) {
                const size_t i = (size_t)(u * 64 + __builtin_ctzll(y));
                Kt[i * cw + (size_t)(nker >> 6)] |= 1ull << (nker & 63);
                const uint64_t* g = col + i * cw;
                for (int b = 0; b < ns; ++b) X += (unsigned)__builtin_popcountll(g[sw[b]] & tc[sw[b]]);
            }
        }
        if (((X >> 1) ^ L) & 1u) om[j >> 6] ^= 1ull << (j & 63);
        depj[nker++] = j;
    }
}

// GSL: the completing gauge coordinates S of the image coordinates (image_kappa, k_fast_rebase). Generators g in
// [0, n) of a space V of kappa vectors (kw words) with image coordinates t = L kappa over a basis beta (tb + g * bw, bw
// words); creator[j] (j < R) = a generator whose tb has highest bit j. Then kappa -> (t | kappa restricted to S) is
// injective on V iff S is injective on N = V & ker L. The creators are an echelon (pivot = highest bit) with no
// reduction; every other generator reduces to t = 0 on them, carrying kappa (kap(g, x): x ^= generator g's kappa),
// and leaves a residual in N; these span N. S (ascending) = the pivots (lowest bits) of a pivot-indexed echelon of the
// residuals (a nonzero vector of N holds the lowest pivot of the rows it combines); maxS >= dim N bounds it (the pass
// stops there).
template <class Kap>
std::vector<int> gauge_completion(int n, const uint64_t* tb, size_t bw, size_t kw, const std::vector<int>& creator,
                                  int maxS, Kap&& kap) {
    std::vector<int> S;
    if (maxS <= 0) return S;
    std::vector<uint8_t> made((size_t)n, 0);
    for (int g : creator) made[(size_t)g] = 1;
    std::vector<uint64_t> wb((size_t)maxS * kw + bw + 2 * kw, 0);
    uint64_t* e3 = wb.data();
    uint64_t* vt = e3 + (size_t)maxS * kw;
    uint64_t* x = vt + bw;
    uint64_t* m3 = x + kw;
    std::vector<int> p3(64 * kw);
    int n3 = 0;
    for (int g = 0; g < n && n3 < maxS; ++g) {
        if (made[(size_t)g]) continue;
        std::copy(tb + (size_t)g * bw, tb + (size_t)(g + 1) * bw, vt);
        std::fill(x, x + kw, 0ull);
        kap(g, x);
        for (size_t u = bw; u-- > 0;)   // t by the creators (the highest bit held; words above u hold none)
            while (vt[u]) {
                const int c = creator[u * 64 + 63 - (size_t)__builtin_clzll(vt[u])];
                const uint64_t* tc = tb + (size_t)c * bw;
                for (size_t q = 0; q <= u; ++q) vt[q] ^= tc[q];
                kap(c, x);
            }
        for (size_t q = 0; q < kw;) {   // x in N by the echelon (the lowest pivot bit held)
            const uint64_t h = x[q] & m3[q];
            if (!h) {
                ++q;
                continue;
            }
            const uint64_t* row = e3 + (size_t)p3[q * 64 + (size_t)__builtin_ctzll(h)] * kw;
            for (size_t z = 0; z < kw; ++z) x[z] ^= row[z];
        }
        size_t u = 0;
        while (u < kw && !x[u]) ++u;
        if (u == kw) continue;
        const int p = (int)(u * 64) + __builtin_ctzll(x[u]);
        p3[(size_t)p] = n3;
        m3[u] |= x[u] & (0 - x[u]);
        std::copy(x, x + kw, e3 + (size_t)n3++ * kw);
        S.push_back(p);
    }
    std::sort(S.begin(), S.end());
    return S;
}

inline uint64_t pext_bits(uint64_t v, uint64_t m) {
#if defined(__BMI2__)
    return _pext_u64(v, m);
#else
    uint64_t out = 0;
    int k = 0;
    while (m) {
        const int b = __builtin_ctzll(m);
        m &= m - 1;
        out |= ((v >> b) & 1ull) << k;
        ++k;
    }
    return out;
#endif
}

// kappa'' = (t | kappa restricted to S) for an ascending S, as a word gather: mask[u] = S's bits in kappa word u,
// off[u] = the output bit of its first one (R + the S members below word u)
struct GaugeSelect {
    std::vector<uint64_t> mask;
    std::vector<size_t> off;
    GaugeSelect(const std::vector<int>& S, size_t kw, int R) : mask(kw, 0), off(kw, 0) {
        for (int g : S) mask[(size_t)g >> 6] |= 1ull << (g & 63);
        size_t o = (size_t)R;
        for (size_t u = 0; u < kw; ++u) {
            off[u] = o;
            o += (size_t)__builtin_popcountll(mask[u]);
        }
    }
    // out (ow words) <- t (tw words), then the selected bits of kappa (kw words)
    void operator()(const uint64_t* t, size_t tw, const uint64_t* kappa, size_t kw, uint64_t* out, size_t ow) const {
        std::fill(out, out + ow, 0ull);
        std::copy(t, t + tw, out);
        for (size_t u = 0; u < kw && u < mask.size(); ++u) {
            if (!mask[u]) continue;
            const uint64_t g = pext_bits(kappa[u], mask[u]);
            const size_t b = off[u], w = b >> 6, sh = b & 63;
            out[w] |= g << sh;
            if (sh && (g >> (64 - sh))) out[w + 1] |= g >> (64 - sh);
        }
    }
};

}  // namespace

void* huge_alloc(size_t bytes) {
    constexpr size_t kAlign = size_t(1) << 21;
    if (bytes > SIZE_MAX - (kAlign - 1)) throw std::bad_alloc();
    const size_t r = (bytes + kAlign - 1) / kAlign * kAlign;
    void* p = xdem_aligned_malloc(kAlign, r);
    if (!p) throw std::bad_alloc();
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    ::madvise(p, r, MADV_HUGEPAGE);   // advice only: a refusal changes nothing but the fault count
#endif
    return p;
}

void huge_free(void* p) noexcept { xdem_aligned_free(p); }

size_t XdemSampler::VecHash::operator()(const std::vector<uint64_t>& v) const noexcept {
    return (size_t)hash_words(v.data(), (int)v.size());
}

struct alignas(64) XdemSampler::ThreadState {   // own cache lines (see XdemLineAlloc)
    struct Entry {
        size_t data_off;   // q (WF words) then rank coin rows (WF words each), internal layout
        int rank;
        int tuid;
    };
    std::vector<uint64_t> buf;
    std::vector<int32_t> table;
    uint64_t mask = 0;
    std::vector<uint64_t> keys;   // entries * WA (odd-axis set bitmasks)
    std::vector<uint64_t> data;
    std::vector<Entry> entries;
    // the slot's C table (SCF): local id -> its content (moved into the global table when merged; ctab[id] is read only
    // while remap[id] < 0), its canonical form (ckeys[ckoff[id], ckoff[id + 1]): the y-RREF on the fast path, the
    // content on the general path) and that form's hash; the index: open addressing over the ids (-1 empty, load <= 1/2)
    std::vector<std::vector<uint64_t>> ctab;
    std::vector<uint64_t> ckeys, chash;
    std::vector<size_t> ckoff;
    std::vector<int32_t> cslot;
    size_t cmask = 0;
    std::vector<int> remap;        // local id -> global id (-1 = not yet merged)
    std::vector<uint32_t> stamp;   // local id -> last block (stamp_ctr) it appeared in
    uint32_t stamp_ctr = 0;
    uint64_t shots = 0, ext_shots = 0, coin_shots = 0, hits = 0, misses = 0, clears = 0;
    // per-shot scratch of compose_ext / resolve / post_fast: the composite's eta mod 4 as two bit planes over the
    // axes (WA words each; eta_a = 2 phi[a] + plo[a]), the dirty-word bitmap of the planes (bit w: word w may be
    // nonzero), the sign register sgn (bit a = parity of the earlier lines' tau on a); all zero between shots.
    XdemLineWords plo, phi, pdirty, sgn;
    XdemLineWords keybuf;
    std::vector<int, XdemLineAlloc<int>> odd;            // fast path: odd axes of the shot
    XdemLineWords ckey;              // fast path: the shot's C in canonical form (y-RREF, yW_ words per row)
    // general path, per-miss scratch of lookup() (reused buffers; no state carried between misses)
    Composed lcmp;
    LawScratch lsc;
    std::vector<uint64_t> lR, lv;
    std::vector<int> lP;
    // postselection: the span test's scratch (XdemSpanTest::words()), shots of the current call it rejected early
    std::vector<uint64_t> ps_scratch;
    uint64_t ps_early = 0;
    // the block's fired extended lines: the walk's (shot << 32 | record offset) pairs, then their per-shot CSR (sort_ext:
    // exo, epos; walk order, each shot's list sorted into file order where it is composed)
    std::vector<uint64_t> epair;
    std::vector<uint32_t> epos, exo, ecum;
    // fired lines (KEEP): the walk's (shot << 32 | line) pairs of the current block, its per-shot CSR scratch, and
    // the call's output so far from this slot (the ids, ascending per shot, blocks in this slot's order)
    std::vector<uint64_t> fpair;
    std::vector<uint32_t> fpos, fmulti;
    std::vector<int32_t> fsort;
    std::vector<uint32_t> fcnt;     // PS: per survivor, its end relative to its block's first id
    std::vector<int32_t, XdemUninitAlloc<int32_t>> fids;
    int64_t* foff = nullptr;        // !PS: the current block's rows of the call's offsets (row s + 1 of the block);
                                    // nullptr outside a call
    // law_of (post_fast<LAW>): the kernel basis of N_O (lkt[j] bit q = sigma_q[j]), rank N_O and its pivots among the
    // beta_D coordinates; law_dim's general-path echelon (detector-pivot rows of WF words, C rows of WO words)
    uint64_t lkt[64];
    int lnpv = 0, lnpvD = 0;
    std::vector<uint64_t> ldr, lcr, lq;
    std::vector<int> ldp, lcp;
    // score_lines' word scratch (beta_d_coords: WD + 1 words; law_has: WO + 1 words)
    std::vector<uint64_t> ld, lx;
    // score_lines' fast-path law memo (law_list), content-keyed by the composite's eta planes (lo then hi, 2 WA words):
    // the law is offset = flips ^ g(eta) and the kernel basis Kt(eta). Per entry: the key, g (WF words), Kt (rho words),
    // npv, npvD, kind. Cleared when full (bounded memory); never changes an output. Bypassed for the rest of a call
    // whose first 4096 lookups hit less than 1 in 4 (diverse composites: its upkeep would cost).
    std::vector<int32_t> mtab;
    uint64_t mmask = 0;
    std::vector<uint64_t> mkey, mg, mkt, mtmp;
    std::vector<int32_t> mnpv, mnpvD;
    std::vector<int8_t> mkind;
    uint64_t mhits = 0, mmiss = 0;
    bool mbypass = false;
    // post_multi's per-shot scratch (CW_ >= 2; mw_words() words, 3 rho + 65 CW_ ints); no state between shots
    XdemLineWords mws;
    std::vector<int, XdemLineAlloc<int>> mwi;
    // the multi-word law (post_multi<LAW>, kind 3): lktW[j] (CW_ words) bit q = sigma_q's beta coordinate j (rho
    // vectors), dim C in lnpv (lnpvD = 0); lrw: the law readers' scratch (law_given / law_has / law_in_c)
    std::vector<uint64_t> lktW, lrw;
    std::vector<int> lrp;
    int lnk = 0;                          // the multi-word law's kernel dimension (lktW's bits per row)
    std::vector<uint64_t> mko;            // law_list memo, multi-word: entry e's Kt at mkt[mko[e]], mkw[e] words/row
    std::vector<int32_t> mkw;
};

// The law of [T, K] for one odd set (SPEC §4.3 step 6), in the internal row layout: offset q at flip 0 with every odd
// exponent taken as 1 (an exponent 3 is 1 plus the axis' image, added per shot), the coins (rank rows of WF words,
// fully reduced; piv = each row's pivot bit, its highest), and dim C (the observable coins given the detectors).
// fast (Task 8 fix round 1): the same law in the one-word coordinates z over beta' (KFast): the coin span is W^perp,
// W = ker N' with the private-pivot basis sigma_q (Kt[j] bit q = sigma_q[j]); an outcome v is in the law iff v - q has
// beta' coordinates z with XOR_{j : z_j} Kt[j] = 0; lv / llow: an echelon of the Kt[j] on the non-beta_D coordinates
// (j >= rho_D), so dim C = (rho' - rho_D) - its rank, and the detector support test is r0 in its span. Stored packed
// in fw (rho' + 2 nl words).
struct XdemSampler::XLaw {
    std::vector<uint64_t> q, coins;
    std::vector<int> piv;
    int kdim = 0;
    std::vector<uint64_t> eD;   // the coins' detector parts, fully reduced (pivot eP = lowest bit): the detector support
    std::vector<int> eP;
    bool fast = false;
    std::vector<uint64_t> fw;   // fast: Kt (rho' words), then lv (nl words), then llow (nl words)
    int nl = 0;
    // fast in cw words per vector (KFast::cw > 0): fw = Kt (rho' x cw words), then lv (nl x cw words); lp = lv's
    // pivots (bit index, lowest bit)
    int cw = 0;
    std::vector<int> lp;
    const uint64_t* Kt() const { return fw.data(); }
    const uint64_t* lv(int rho) const { return fw.data() + rho; }
    const uint64_t* llow(int rho) const { return fw.data() + rho + nl; }
};

// One memo entry of the optimal correction (SPEC §4.3, §4.7): everything K needs that depends on h alone. K depends
// on h only through its composite's eta and its tau on the rotated axes (S_k lies in Od), so the entry is keyed by
// that content (k_key: eta's two bit planes and tau, 3 WA words), a function of the sorted cancelled id list; h's
// flips f_h enter only through the eps term f.G, so the entry keeps e0 = eps's exponent without f.G; per shot
// frame_k = (d ^ f_h)|_D . P_k ^ f_h[L_k] ^ e0_k ^ y_k.
struct XdemSampler::KEntry {
    struct Row {
        uint8_t lossy = 0, y = 0, e0 = 0, ts = 0;   // ts = tau_h(S_k) (the same-composite sign of K's strings)
        uint32_t p0 = 0, p1 = 0, s0 = 0, s1 = 0, w0 = 0, w1 = 0;   // P_k = P[p0, p1), S_k = S[s0, s1), Sw[w0, w1)
    };
    std::vector<Row> rows;                          // one per observable
    std::vector<int32_t> P, S;                      // pivot detector ids, axis ids (ascending per row)
    std::vector<uint64_t> Sw;                       // S_k as (axis word, mask) pairs
    bool lossy = false;                             // some observable is lossy
    // K's strings as extra axes (step 6), nx of them: their images in the internal layout (nx * WF words), kappa (nx *
    // nwK), sigma (nx * nwH, hidden-syndrome qubit bits), delta (nx * WO: the observable bits of kappa' = [kappa |
    // delta], k_build step 4), exponents (1..3) and the observable whose tau_T(S_k) signs them (-1: Z_{L_k}, unsigned).
    // Derived on first use: File-layout axes (b = 0) for the generic law_into (xax; force_general and k_optimal), or
    // the beta' coordinates and kappa' words for the fast law (xT, xK: one word, or KFast::cw words, per string).
    std::vector<uint64_t> ximg, xkap, xhid, xdl;
    std::vector<uint8_t> xe;
    std::vector<int32_t> xk;
    bool xax_built = false, xfast_built = false;
    std::vector<Axis> xax;
    std::vector<uint64_t> xT, xK;
    // law(h)'s detector support at flip 0 (offset g_D, WD words, + span of eD, fully reduced, pivot eP = lowest bit);
    // built on first use (k_support) from h's eta: a shot whose T has h's composite reads it from the law of [T, K]
    std::vector<std::pair<int, int>> eta;
    bool sup_built = false;
    std::vector<uint64_t> gD, eD;
    std::vector<int> eP;
    // the law of [T, K] when T's extended lines are h's: built on first use; q includes the eta in {2, 3} images
    bool same_built = false;
    XLaw same;
    // ERRATA E4: the frame on the lossy observables is a point of the support of law([h, K]) restricted to them, given
    // d (k_lossy_setup / k_lossy_frames). lmask: the lossy observables (WO words). Built on first use from `same` (or,
    // in k_optimal, from a generic law of the same composite): lfq = its offset (WF words); its coins in a fully
    // reduced echelon (pivot = lowest bit), split into the rows with a detector pivot (lfD, WF words each, pivots
    // lfDP) and the rows with none (zero detector part), whose lossy parts form a second fully reduced echelon (lfL,
    // WO words each, pivots lfLP, lowest bit).
    std::vector<uint64_t> lmask;
    bool lf_built = false;
    std::vector<uint64_t> lfq, lfD, lfL;
    std::vector<int> lfDP, lfLP;
};

// k_build's reusable buffers (eta and colpos are all 0 / -1 between calls)
struct XdemSampler::KScratch {
    std::vector<uint8_t> eta;
    std::vector<int> colpos, od, two, cols, elead;
    std::vector<uint64_t> colv, ev, ec, v, cmb, Pm, phiI, phiK, phiH, Aimg, zl, tk;
    std::vector<int> lkoff, lk, lkpos;
    // the file axes' kappa (A * nwK words), sigma (A * nwH, as bits) and bhat, flat (built on first use)
    std::vector<uint64_t> axK, axH;
    std::vector<uint8_t> axB;
};

// The fast law's coordinates (Task 8 fix round 1): beta' = beta (beta_D, then beta_L; the sampler's fast-path basis)
// followed by nn new vectors, the observable unit vectors' residuals modulo span(beta) (an echelon, pivot = highest
// bit, each reduced by beta_L and the earlier ones in build order). ecoord[o] = e_{L_o}'s beta' coordinates. ok: the
// sampler is on its fast path (every file but force_general). cw = 0: one word per vector (one-word coordinates,
// rho' <= 64 and r + O <= 64: kappa' in one word); cw > 0 (XFM fix round 1): cw = ceil(max(rho', r + O) / 64) words,
// ecW (O x cw) the coordinates, TcB / KcB (A x cw) the file axes' beta' coordinates and kappa''. Same law either way.
// GSL: kappa' = [kappa | delta] is re-expressed once in image coordinates (k_fast_rebase): the law of [T, K] is in
// graph-state form too (law_kernel1 / law_kernelw).
struct XdemSampler::KFast {
    bool ok = false;
    bool built = false;
    int rhoX = 0, nn = 0;
    std::vector<uint64_t> ecoord, nv, nvI;   // O words; nn * WO (observable words); nn * WF (internal layout)
    std::vector<int> np;                     // pivot (observable bit) of new vector t
    std::vector<int> nobs;                   // the observable whose residual made new vector t
    int cw = 0;
    std::vector<uint64_t> ecW, TcB, KcB;
    // GSL (k_fast_rebase): kappa' in image coordinates, kappa'' = (t' | kappa' restricted to S), S over the r + O
    // bits of kappa' = [kappa | delta]; the file axes' kappa'' (KcB: cw words per axis; Kc1: one word)
    std::vector<int> S;
    std::vector<uint64_t> Kc1;
};

// The memo (cache_cap entries, cleared when full) and the extended-law cache, content-keyed by (entry id, the odd file-
// axis set of T's composite), cap cache_cap, cleared when full and with the memo. Both are only ever read under mu_.
struct XdemSampler::KMemo {
    std::unordered_map<std::vector<uint64_t>, int, VecHash> index;   // k_key content -> entry
    std::vector<KEntry> entries;
    std::unordered_map<std::vector<uint64_t>, int, VecHash> lawidx;
    std::vector<XLaw> laws;
    uint64_t clears = 0, law_clears = 0, fast_laws = 0;   // clear_all (memo + law cache); the law cache alone (full)
    KScratch ks;
    KFast kf;
    std::vector<uint64_t> item_T, item_K;   // the fast law's odd items (file axes, then K's strings)
    std::vector<uint8_t> item_b;
    // per-shot scratch: canonical lists, h's flips, the row, the frame bits, the law cache key, h's / T's composite
    // keys (k_key), an image accumulator, signs per observable
    std::vector<int32_t> hbuf, tbuf;
    std::vector<uint64_t> fh, row, fobs, lkey, hkey, tkey, acc;
    std::vector<uint8_t> sg;
    std::vector<uint64_t> lfu;   // k_lossy_frames' scratch (WF + WO words)
};

XdemSpanTest::XdemSpanTest(int D, const uint8_t* flags, const uint64_t* vecs, size_t nvec, size_t stride) {
    const size_t WD = ((size_t)D + 63) / 64;
    std::vector<uint64_t> m(WD, 0);
    int bits = 0;
    for (int d = 0; d < D; ++d)
        if (flags[d]) {
            m[(size_t)d >> 6] |= 1ull << (d & 63);
            ++bits;
        }
    for (size_t w = 0; w < WD; ++w)
        if (m[w]) {
            widx_.push_back((uint32_t)w);
            wmask_.push_back(m[w]);
        }
    const size_t nw = widx_.size();
    if (!nw) return;
    // fully reduced echelon basis of the restricted vectors (pivot = lowest bit; no other basis vector has it)
    std::vector<uint64_t> v(nw);
    for (size_t i = 0; i < nvec && rank_ < bits; ++i) {
        const uint64_t* src = vecs + i * stride;
        uint64_t any = 0;
        for (size_t k = 0; k < nw; ++k) any |= v[k] = src[widx_[k]] & wmask_[k];
        if (!any) continue;
        for (int b = 0; b < rank_; ++b)
            if (v[pk_[(size_t)b]] & pb_[(size_t)b])
                for (size_t k = 0; k < nw; ++k) v[k] ^= basis_[(size_t)b * nw + k];
        size_t kk = 0;
        while (kk < nw && !v[kk]) ++kk;
        if (kk == nw) continue;
        const uint64_t bit = v[kk] & (0 - v[kk]);
        for (int b = 0; b < rank_; ++b)
            if (basis_[(size_t)b * nw + kk] & bit)
                for (size_t k = 0; k < nw; ++k) basis_[(size_t)b * nw + k] ^= v[k];
        basis_.insert(basis_.end(), v.begin(), v.end());
        pk_.push_back((uint32_t)kk);
        pb_.push_back(bit);
        ps_.push_back((uint8_t)__builtin_ctzll(bit));
        ++rank_;
    }
    if (rank_ > 64 || nw > 4) return;
    // selector table path: basis sorted by pivot (word, bit), so the pivot bits of d gathered word by word (pext) are
    // the selector in basis order; tab_[t][v] = XOR of basis 8t + bits of v (nw words)
    std::vector<int> ord((size_t)rank_);
    for (int b = 0; b < rank_; ++b) ord[(size_t)b] = b;
    std::sort(ord.begin(), ord.end(), [&](int x, int y) {
        return pk_[(size_t)x] != pk_[(size_t)y] ? pk_[(size_t)x] < pk_[(size_t)y] : pb_[(size_t)x] < pb_[(size_t)y];
    });
    pm_.assign(nw, 0);
    for (int b = 0; b < rank_; ++b) pm_[pk_[(size_t)b]] |= pb_[(size_t)b];
    ntab_ = (rank_ + 7) / 8;
    tab_.assign((size_t)ntab_ * 256 * nw, 0);
    for (int t = 0; t < ntab_; ++t)
        for (int v = 1; v < 256; ++v) {
            const int j = t * 8 + __builtin_ctz((unsigned)v);
            uint64_t* dst = &tab_[((size_t)t * 256 + (size_t)v) * nw];
            const uint64_t* prev = &tab_[((size_t)t * 256 + (size_t)(v & (v - 1))) * nw];
            for (size_t k = 0; k < nw; ++k)
                dst[k] = prev[k] ^ (j < rank_ ? basis_[(size_t)ord[(size_t)j] * nw + k] : 0ull);
        }
}

namespace {
// the table path (rank <= 64, NW <= 4 words): selector = d's pivot bits in basis order, ceil(rank / 8) lookups
template <int NW>
inline bool span_table(const uint64_t* d, const uint32_t* widx, const uint64_t* wmask, const uint64_t* pm,
                       const uint64_t* tab, int ntab) {
    uint64_t v[NW], sel = 0;
    int off = 0;
    for (int k = 0; k < NW; ++k) {
        const uint64_t x = d[widx[k]];
        v[k] = x & wmask[k];
        sel |= pext_bits(x, pm[k]) << off;   // off <= 63 while a later word still has pivots (rank <= 64)
        off += __builtin_popcountll(pm[k]);
        if (off >= 64) off = 63;   // only reached when no pivot follows (its bits are then 0)
    }
    for (int t = 0; t < ntab; ++t) {
        const uint64_t* row = tab + ((size_t)t * 256 + (size_t)((sel >> (8 * t)) & 255u)) * NW;
        for (int k = 0; k < NW; ++k) v[k] ^= row[k];
    }
    uint64_t acc = 0;
    for (int k = 0; k < NW; ++k) acc |= v[k];
    return acc == 0;
}

// d_M in V_M for a row of NW compressed words (NW > 0: fixed; 0: nw at run time). Every basis vector's selector
// (d's bit at its pivot) depends on d alone, so the masked XORs are independent: no chain through the residual.
template <int NW>
inline bool span_reduce(const uint64_t* d, const uint32_t* widx, const uint64_t* wmask, size_t nw_rt, int rank,
                        const uint64_t* basis, const uint32_t* pk, const uint8_t* ps, uint64_t* r) {
    const size_t nw = NW ? (size_t)NW : nw_rt;
    uint64_t x[NW ? NW : 1];
    uint64_t* v = NW ? x : r;
    for (size_t k = 0; k < nw; ++k) v[k] = d[widx[k]] & wmask[k];
    for (int b = 0; b < rank; ++b) {
        const uint64_t msk = 0 - ((d[widx[pk[b]]] >> ps[b]) & 1ull);
        const uint64_t* bv = basis + (size_t)b * nw;
        for (size_t k = 0; k < nw; ++k) v[k] ^= bv[k] & msk;
    }
    uint64_t acc = 0;
    for (size_t k = 0; k < nw; ++k) acc |= v[k];
    return acc == 0;
}
}  // namespace

bool XdemSpanTest::in_span(const uint64_t* d, uint64_t* r) const {
    // the basis is fully reduced, so d_M is in V_M iff d_M = XOR of the basis vectors whose pivot bit d_M has (XOR-ing
    // one never changes another pivot bit): one masked XOR per basis vector, selected by the pivot bits of d itself
    if (!pm_.empty()) {
        const uint64_t *W = wmask_.data(), *P = pm_.data(), *T = tab_.data();
        switch (widx_.size()) {
            case 1: return span_table<1>(d, widx_.data(), W, P, T, ntab_);
            case 2: return span_table<2>(d, widx_.data(), W, P, T, ntab_);
            case 3: return span_table<3>(d, widx_.data(), W, P, T, ntab_);
            default: return span_table<4>(d, widx_.data(), W, P, T, ntab_);
        }
    }
    const uint64_t* B = basis_.data();
    switch (widx_.size()) {
        case 1: return span_reduce<1>(d, widx_.data(), wmask_.data(), 1, rank_, B, pk_.data(), ps_.data(), r);
        case 2: return span_reduce<2>(d, widx_.data(), wmask_.data(), 2, rank_, B, pk_.data(), ps_.data(), r);
        case 3: return span_reduce<3>(d, widx_.data(), wmask_.data(), 3, rank_, B, pk_.data(), ps_.data(), r);
        case 4: return span_reduce<4>(d, widx_.data(), wmask_.data(), 4, rank_, B, pk_.data(), ps_.data(), r);
        default:
            return span_reduce<0>(d, widx_.data(), wmask_.data(), widx_.size(), rank_, B, pk_.data(), ps_.data(), r);
    }
}

XdemSampler::XdemSampler(File F, size_t cache_cap, bool force_general) : F_(std::move(F)), cap_(cache_cap) {
    try {
        construct(force_general);
    } catch (const std::bad_alloc&) {   // sizes the file declares (no cap here: the failure is reported by name)
        throw XdemCapacityError("xdem: out of memory building the sampler (" + std::to_string(F_.D) + " detectors, " +
                                std::to_string(F_.O) + " observables, gauge " + std::to_string(F_.r) + ", hidden " +
                                std::to_string(F_.m) + ", " + std::to_string(F_.axes.size()) + " axes, " +
                                std::to_string(L_) + " error lines)");   // L_: set first; F_'s lines are freed
    }
}

void XdemSampler::construct(bool force_general) {
    if (cap_ < 1) throw std::invalid_argument("XdemSampler: cache_cap must be >= 1");
    D_ = F_.D;
    O_ = F_.O;
    // int arithmetic on D_ / O_ is safe: the reader bounds D + O by 2^31 - 1 (kMaxColumns), so 64 WF_ <= 2^32
    WD_ = (D_ + 63) / 64;
    WO_ = (O_ + 63) / 64;
    WF_ = WD_ + WO_;
    W_ = WF_;   // the shot row is the readout bits only (no X-pattern words)
    A_ = (int)F_.axes.size();
    WA_ = (A_ + 63) / 64;
    const size_t L = F_.num_lines();
    L_ = L;
    // line ids are int32 in the fired output
    if (L > (size_t)INT32_MAX)
        throw XdemFileError("xdem: " + std::to_string(L) + " error lines exceed the sampler's limit of 2^31 - 1 lines");
    img_.assign((size_t)A_ * (size_t)WF_, 0);
    nbD_ = (size_t)(D_ + 7) / 8;
    nbO_ = (size_t)(O_ + 7) / 8;
    // the rotated axes (some line's rot or late field holds them): tau elsewhere is never read
    std::vector<uint8_t> rotated((size_t)A_, 0);
    for (const Rot& x : F_.rot) rotated[(size_t)x.axis] = 1;
    for (const Rot& x : F_.late) rotated[(size_t)x.axis] = 1;
    xo_of_.assign(L, 0);
    xa_.push_back(0);   // offset 0 names no record
    std::vector<uint64_t> rec;
    for (size_t i = 0; i < L; ++i) {
        // total exponent per axis, (E^< + E^>) mod 4: two ascending lists merged; a 0 total is dropped, a 2 kept; the
        // nonzero totals as word groups (word, bits with E odd, bits with E & 2), words ascending
        uint64_t a0 = F_.rot_off[i], a1 = F_.rot_off[i + 1], b0 = F_.late_off[i], b1 = F_.late_off[i + 1];
        n_rot_ += (a1 > a0) || (b1 > b0);
        rec.assign(1, 0);
        uint64_t nE = 0, nT = 0;
        while (a0 < a1 || b0 < b1) {
            int ax;
            unsigned e = 0;
            if (b0 >= b1 || (a0 < a1 && F_.rot[a0].axis <= F_.late[b0].axis)) {
                ax = F_.rot[a0].axis;
                e += F_.rot[a0++].e;
                if (b0 < b1 && F_.late[b0].axis == ax) e += F_.late[b0++].e;
            } else {
                ax = F_.late[b0].axis;
                e += F_.late[b0++].e;
            }
            if (!(e & 3)) continue;
            if (!nE || rec[rec.size() - 3] != (uint64_t)(ax >> 6)) {
                rec.insert(rec.end(), {(uint64_t)(ax >> 6), 0, 0});
                ++nE;
            }
            rec[rec.size() - 2] |= (uint64_t)(e & 1) << (ax & 63);
            rec[rec.size() - 1] |= (uint64_t)((e >> 1) & 1) << (ax & 63);
        }
        for (uint64_t t = F_.tau_off[i]; t < F_.tau_off[i + 1]; ++t) {   // ascending axes: same word -> last entry
            const int a = F_.tau_ax[t];
            if (!rotated[(size_t)a]) continue;
            if (!nT || rec[rec.size() - 2] != (uint64_t)(a >> 6)) {
                rec.insert(rec.end(), {(uint64_t)(a >> 6), 0});
                ++nT;
            }
            rec.back() |= 1ull << (a & 63);
        }
        if (!nE && !nT) continue;   // not extended
        rec[0] = nE | nT << 32;
        if (xa_.size() + rec.size() > (size_t)UINT32_MAX)
            throw XdemFileError("xdem: the extended lines' exponent and tau entries exceed the sampler's limit of "
                                "2^32 - 1 words");
        xo_of_[i] = (uint32_t)xa_.size();
        xa_.insert(xa_.end(), rec.begin(), rec.end());
    }
    // the per-line flip arena: the File's sorted distinct columns mapped to the internal layout (monotone, so still
    // ascending and distinct)
    loff_.reserve(L + 1);
    loff_.push_back(0);
    const uint32_t padO = (uint32_t)(64LL * WD_ - D_);   // observable column D + o -> bit 64 WD_ + o
    std::map<uint64_t, int> chain_of;   // p's bit pattern -> chain (first appearance order kept by index)
    for (size_t i = 0; i < L; ++i) {
        const size_t first = lent_.size();
        for (uint64_t t = F_.flip_off[i]; t < F_.flip_off[i + 1]; ++t) {
            const uint32_t j = (uint32_t)F_.flip_cols[t];
            const uint32_t k = j < (uint32_t)D_ ? j : j + padO;   // bits arrive ascending: same word -> last entry
            if (lent_.size() > first && lent_.back().word == (k >> 6)) lent_.back().mask ^= 1ull << (k & 63);
            else lent_.push_back({1ull << (k & 63), (uint64_t)(k >> 6)});
        }
        loff_.push_back(lent_.size());
        const double p = F_.p[i];
        if (!(p >= 0.0 && p <= 1.0))
            throw XdemFileError("xdem: line " + std::to_string(F_.src_line.empty() ? 0 : F_.src_line[i]) +
                                ": error probability outside [0, 1]");
        if (p == 0.0) continue;
        mean_fired_ += p;
        if (xo_of_[i]) mean_ext_ += p;
        uint64_t pb;
        std::memcpy(&pb, &p, sizeof pb);
        auto it = chain_of.find(pb);
        int c;
        if (it == chain_of.end()) {
            c = (int)chains_.size();
            chain_of.emplace(pb, c);
            Chain ch;
            ch.p = p;
            ch.always = p == 1.0;
            ch.lp = ch.always ? 0.0 : std::log1p(-p);
            chains_.push_back(std::move(ch));
        } else {
            c = it->second;
        }
        chains_[(size_t)c].mem.push_back(Member{0, 0, xo_of_[i]});
        chains_[(size_t)c].lines.push_back((int)i);
    }
    lent_.resize(lent_.size() + kEntPad, LineEnt{0, 0});   // zero tail: an empty last line reads kEntPad of it
    for (Chain& ch : chains_)
        for (size_t m = 0; m < ch.mem.size(); ++m) {
            const size_t i = (size_t)ch.lines[m];
            ch.mem[m].beg = loff_[i];
            ch.mem[m].cnt = (uint32_t)(loff_[i + 1] - loff_[i]);
        }
    lent_.shrink_to_fit();
    // score_lines (Task 7): the file facts of SPEC §4.4, read while the lines are here; then each line's E^< entries,
    // E^> (as word groups, kl_) and full tau move to the scorer's arrays
    late_axes_ = qeccore::xdem::late_axes(F_);
    r2b_ = qeccore::xdem::r2b_available(F_);
    inv_ = qeccore::xdem::invisible_residues(F_);
    kl_of_.assign(L, 0);
    kl_.assign(1, 0);   // offset 0 names no record
    for (size_t i = 0; i < L; ++i) {   // E^> as word groups (w, bits with e odd, bits with e & 2); axes ascending
        const uint64_t b0 = F_.late_off[i], b1 = F_.late_off[i + 1];
        if (b0 == b1) continue;
        const size_t at = kl_.size();
        kl_.push_back(0);
        for (uint64_t k = b0; k < b1; ++k) {
            const int a = F_.late[k].axis, e = F_.late[k].e;
            if (kl_[at] == 0 || kl_[kl_.size() - 3] != (uint64_t)(a >> 6)) {
                kl_.insert(kl_.end(), {(uint64_t)(a >> 6), 0, 0});
                ++kl_[at];
            }
            kl_[kl_.size() - 2] |= (uint64_t)(e & 1) << (a & 63);
            kl_[kl_.size() - 1] |= (uint64_t)((e >> 1) & 1) << (a & 63);
        }
        if (at > (size_t)UINT32_MAX) throw XdemFileError("xdem: the late exponents exceed the scorer's limit of 2^32 - 1 words");
        kl_of_[i] = (uint32_t)at;
    }
    early_off_ = std::move(F_.rot_off);
    early_ = std::move(F_.rot);
    late_off_ = std::move(F_.late_off);
    tau_full_off_ = std::move(F_.tau_off);
    tau_full_ = std::move(F_.tau_ax);
    // the lines now live in the arenas only: release the File's copy (law_into reads the header alone)
    std::vector<double>().swap(F_.p);
    std::vector<uint64_t>().swap(F_.flip_off);
    std::vector<int32_t>().swap(F_.flip_cols);
    std::vector<uint64_t>().swap(F_.tau_off);
    std::vector<int32_t>().swap(F_.tau_ax);
    std::vector<uint64_t>().swap(F_.rot_off);
    std::vector<Rot>().swap(F_.rot);
    std::vector<uint64_t>().swap(F_.late_off);
    std::vector<Rot>().swap(F_.late);
    std::vector<int32_t>().swap(F_.src_line);
    for (int a = 0; a < A_; ++a) {
        const Bits& b = F_.axes[(size_t)a].img;
        uint64_t* out = &img_[(size_t)a * WF_];
        for (size_t wi = 0; wi < b.w.size(); ++wi)
            for (uint64_t x = b.w[wi]; x; x &= x - 1) {
                const long long j = 64LL * (long long)wi + __builtin_ctzll(x);
                const long long k = j < D_ ? j : 64LL * WD_ + (j - D_);
                out[k >> 6] ^= 1ull << (k & 63);
            }
    }
    build_beta(force_general);
    if (!fast_) build_law_axes();   // the general path's law code (force_general only)
    if (fast_) build_ycoords();
    cspaces_.push_back({});
    cdim_.push_back(0);
    cpiv_.push_back({});
    chash_.push_back(hash_c(nullptr, 0));
    cslot_.assign(1024, -1);
    cslot_mask_ = cslot_.size() - 1;
    cslot_[chash_[0] & cslot_mask_] = 0;
}

void XdemSampler::build_ycoords() {
    // E: the fully reduced echelon of the beta_L observable parts bLo_ (pivot = highest bit; the beta_L are independent
    // with zero detector part, so their observable parts are independent), rows by ascending pivot P_k; yA_[j] bit k =
    // bit P_k of bLo_[j] (bLo_[j] = sum_k yA_[j]_k E_k: E_k alone holds bit P_k)
    const int rL = rhoL_, WO = WO_;
    yW_ = (rL + 63) / 64;
    yA_.assign((size_t)rL * (size_t)yW_, 0);
    yE_.assign((size_t)rL * (size_t)WO, 0);
    if (!rL) return;
    std::vector<uint64_t> E(bLo_.begin(), bLo_.begin() + (long)rL * WO);
    std::vector<uint64_t> pm((size_t)WO, 0);
    std::vector<int> pidx((size_t)WO * 64, 0), ord((size_t)rL, 0);
    if (rref_high<0>(E.data(), rL, WO, pm.data(), pidx.data()) != rL)
        throw std::logic_error("xdem: dependent beta_L observable parts (internal)");
    rows_desc(pm.data(), WO, pidx.data(), ord.data());
    std::vector<int> P((size_t)rL);
    for (int k = 0; k < rL; ++k) {   // ascending pivot: k-th from the bottom of rows_desc's order
        const int i = ord[(size_t)(rL - 1 - k)];
        P[(size_t)k] = highest(&E[(size_t)i * WO], WO);
        std::copy(&E[(size_t)i * WO], &E[(size_t)(i + 1) * WO], &yE_[(size_t)k * WO]);
    }
    for (int j = 0; j < rL; ++j)
        for (int k = 0; k < rL; ++k)
            if ((bLo_[(size_t)j * WO + (size_t)(P[(size_t)k] >> 6)] >> (P[(size_t)k] & 63)) & 1ull)
                yA_[(size_t)j * yW_ + (size_t)(k >> 6)] |= 1ull << (k & 63);
}

void XdemSampler::build_beta(bool force_general) {
    // basis beta of span(img_a): first the vectors with independent detector parts (beta_D, pivot = lowest detector
    // bit), then a basis of the remaining zero-detector residuals (beta_L, pivot = highest bit); T_a = coordinates.
    fast_ = fast1_ = false;
    r_ = F_.r;
    if (force_general) return;
    if (A_ == 0) {   // no axes (a conventional DEM): no law to compute; post_fast is never reached
        fast_ = fast1_ = true;
        return;
    }
    if (r_ > 64) return build_multi();
    std::vector<std::vector<uint64_t>> bD, bL;
    std::vector<int> pD, pL;
    std::vector<uint64_t> cD((size_t)A_, 0), cL((size_t)A_, 0);
    std::vector<int> crD, crL;   // the axis that created each beta_D / beta_L vector (image_kappa)
    for (int a = 0; a < A_; ++a) {
        std::vector<uint64_t> v(img_.begin() + (long)a * WF_, img_.begin() + (long)(a + 1) * WF_);
        for (size_t i = 0; i < bD.size(); ++i)
            if ((v[(size_t)pD[i] >> 6] >> (pD[i] & 63)) & 1ull) {
                for (int w = 0; w < WF_; ++w) v[(size_t)w] ^= bD[i][(size_t)w];
                cD[(size_t)a] ^= 1ull << i;
            }
        int pd = -1;
        for (int w = 0; w < WD_ && pd < 0; ++w)
            if (v[(size_t)w]) pd = w * 64 + __builtin_ctzll(v[(size_t)w]);
        if (pd >= 0) {
            if (bD.size() + bL.size() >= 64) return build_multi();   // rho > 64: multi-word coordinates
            cD[(size_t)a] ^= 1ull << bD.size();
            crD.push_back(a);
            pD.push_back(pd);
            bD.push_back(std::move(v));
            continue;
        }
        for (size_t i = 0; i < bL.size(); ++i)
            if ((v[(size_t)pL[i] >> 6] >> (pL[i] & 63)) & 1ull) {
                for (int w = 0; w < WF_; ++w) v[(size_t)w] ^= bL[i][(size_t)w];
                cL[(size_t)a] ^= 1ull << i;
            }
        const int ph = highest(v.data(), WF_);
        if (ph >= 0) {
            if (bD.size() + bL.size() >= 64) return build_multi();
            cL[(size_t)a] ^= 1ull << bL.size();
            crL.push_back(a);
            pL.push_back(ph);
            bL.push_back(std::move(v));
        }
    }
    const int rD = (int)bD.size(), rL = (int)bL.size();
    rho_ = rD + rL;
    rhoL_ = rL;
    rmask_ = r_ >= 64 ? ~0ull : (r_ ? (1ull << r_) - 1 : 0ull);
    Tc_.assign((size_t)A_, 0);
    Kc_.assign((size_t)A_, 0);
    bb_.assign((size_t)A_, 0);
    for (int a = 0; a < A_; ++a) {
        Tc_[(size_t)a] = cD[(size_t)a] | (rL ? cL[(size_t)a] << rD : 0ull);   // rL > 0 implies rD < 64
        const Axis& ax = F_.axes[(size_t)a];
        Kc_[(size_t)a] = ax.k.empty() ? 0ull : ax.k[0];
        bb_[(size_t)a] = (uint8_t)ax.b;
    }
    Kg_.assign((size_t)A_, 0);
    crD.insert(crD.end(), crL.begin(), crL.end());
    creator_ = crD;
    image_kappa(Tc_.data(), 1, creator_, Tc_.data(), 1, Kg_.data(), 1);
    std::vector<std::vector<uint64_t>> beta = bD;
    for (auto& v : bL) beta.push_back(v);
    ntab_ = (rho_ + 7) / 8;
    btab_.assign((size_t)ntab_ * 256 * WF_, 0);
    for (int t = 0; t < ntab_; ++t)
        for (int v = 1; v < 256; ++v) {
            const int j = t * 8 + __builtin_ctz((unsigned)v);
            uint64_t* dst = &btab_[((size_t)t * 256 + (size_t)v) * WF_];
            const uint64_t* prev = &btab_[((size_t)t * 256 + (size_t)(v & (v - 1))) * WF_];
            for (int w = 0; w < WF_; ++w) dst[w] = prev[w] ^ (j < rho_ ? beta[(size_t)j][(size_t)w] : 0ull);
        }
    betaO_.assign((size_t)rho_ * WO_, 0);
    for (int j = 0; j < rho_; ++j)
        for (int w = 0; w < WO_; ++w) betaO_[(size_t)j * WO_ + w] = beta[(size_t)j][(size_t)(WD_ + w)];
    // score_lines' coordinates (law_given / law_has / law_in_c): beta_D's detector parts and pivots, beta_L's
    // observable parts and pivots (a beta_L vector has no detector bit, so its highest bit is an observable bit)
    bDd_.clear();
    bLo_.clear();
    for (int i = 0; i < rD; ++i) bDd_.insert(bDd_.end(), bD[(size_t)i].begin(), bD[(size_t)i].begin() + WD_);
    for (int i = 0; i < rL; ++i) bLo_.insert(bLo_.end(), bL[(size_t)i].begin() + WD_, bL[(size_t)i].end());
    pD_ = pD;
    pL_.clear();
    for (int p : pL) pL_.push_back(p - 64 * WD_);
    fast_ = fast1_ = true;
}

// GSL: the gauge basis in image coordinates, once at load. By the header condition §3.5(b) (checked by the reader) an
// axis' image coordinates t_a (rho_ bits: Tc_ over beta, or GcW_ over gamma) are L kappa_a for one linear L, so
// kappa'_a = (t_a | kappa_a restricted to S) is linear in kappa_a for any set S of gauge coordinates, and injective on
// V = span{kappa_a} for S = gauge_completion's (from the axes' beta coordinates tb and the beta creators). Then the law
// matrix N_O = sum_{a in O} kappa'_a t_a^T has the kernel, the dependent columns and the dependencies of the file's
// (its columns lie in V and are mapped injectively), and its first rho rows are G = sum_{a in O} t_a t_a^T: symmetric,
// the adjacency (loops on the diagonal) of the graph state of O's gauge graph. dim N = rank kappa - rho <= r - rho
// (r = rho: S empty, no work). out (ow words per axis): bits [0, rho) = t_a, bit rho + i = kappa_a's bit S[i].
void XdemSampler::image_kappa(const uint64_t* tb, size_t bw, const std::vector<int>& creator, const uint64_t* t,
                              size_t tw, uint64_t* out, size_t ow) const {
    const size_t rw = words_for(r_);
    const std::vector<int> S = gauge_completion(A_, tb, bw, rw, creator, r_ - rho_, [&](int a, uint64_t* x) {
        const std::vector<uint64_t>& k = F_.axes[(size_t)a].k;
        for (size_t u = 0; u < rw && u < k.size(); ++u) x[u] ^= k[u];
    });
    const GaugeSelect sel(S, rw, rho_);
    for (int a = 0; a < A_; ++a) {
        const std::vector<uint64_t>& k = F_.axes[(size_t)a].k;
        sel(t + (size_t)a * tw, tw, k.data(), std::min(k.size(), rw), out + (size_t)a * ow, ow);
    }
}

// General path (Task 12 fix round 1). The law of an odd set is unchanged by any change of the gauge-qubit basis
// (GL(r)): its coin span, its offset reduced modulo the coins and its W phases depend on the kappa vectors only through
// the gauge-qubit space they coordinatize (SPEC §3.5; the reader accepts any GL(r) kappa, Review Focus 3). law_into's
// cost grows with the kappa weights, and the writer's canonical gauge basis (SPEC §2) is dense (bt81: mean kappa
// weight 10.9 per axis, against 5.25 in the dc v1 file of the same circuit). So lookup() runs law_into over these
// axes: the file's, with kappa in a lower-weight basis. With K = the A x r matrix whose rows are the kappa vectors,
// the basis vectors are K's columns; column i is replaced by column i + column j while that lowers its weight, until
// no pair does (each step keeps the basis invertible; the total weight falls, so the loop ends). bt81: 10.9 -> 5.9.
// Only the re-based kappa words are kept (law_k_, A x ceil(r/64) words); lookup() reads img etc. from F_.axes.
// Bound: a pass (every ordered pair i != j once) costs O(r^2 A / 64); the passes are capped at r. Any stopping point
// is correct, since every accepted step is an invertible column operation and the law is basis-free (the outputs are
// byte-identical at any stop, Task 12 fr1 re-review); the cap bounds construction at O(r^3 A / 64). Measured (passes
// including the last, unchanged one): bt81 (r = 87) and bt27 (r = 33) 4, the in-repo fixtures <= 5 (boundary_r65).
void XdemSampler::build_law_axes() {
    const int r = F_.r;
    const size_t aw = ((size_t)A_ + 63) / 64;
    std::vector<uint64_t> col((size_t)r * aw, 0);
    for (int a = 0; a < A_; ++a) {
        const std::vector<uint64_t>& k = F_.axes[(size_t)a].k;
        for (size_t kw = 0; kw < k.size(); ++kw)
            for (uint64_t y = k[kw]; y; y &= y - 1)
                col[(kw * 64 + (size_t)__builtin_ctzll(y)) * aw + (size_t)a / 64] |= 1ull << (a & 63);
    }
    auto weight = [&](const uint64_t* x) {
        int n = 0;
        for (size_t w = 0; w < aw; ++w) n += __builtin_popcountll(x[w]);
        return n;
    };
    std::vector<int> wt((size_t)r);
    // (col.data() + offset, not &col[offset]: col is empty when the file has no axis)
    for (int i = 0; i < r; ++i) wt[(size_t)i] = weight(col.data() + (size_t)i * aw);
    int passes = 0;
    for (bool changed = true; changed && passes < r; ++passes) {
        changed = false;
        for (int i = 0; i < r; ++i) {
            uint64_t* ci = col.data() + (size_t)i * aw;
            for (int j = 0; j < r; ++j) {
                if (j == i) continue;
                const uint64_t* cj = col.data() + (size_t)j * aw;
                int n = 0;
                for (size_t w = 0; w < aw; ++w) n += __builtin_popcountll(ci[w] ^ cj[w]);
                if (n >= wt[(size_t)i]) continue;
                for (size_t w = 0; w < aw; ++w) ci[w] ^= cj[w];
                wt[(size_t)i] = n;
                changed = true;
            }
        }
    }
    law_k_.assign((size_t)A_, {});
    for (int a = 0; a < A_; ++a) {
        std::vector<uint64_t>& k = law_k_[(size_t)a];
        k.assign(F_.axes[(size_t)a].k.size(), 0ull);
        for (int j = 0; j < r; ++j)
            if ((col[(size_t)j * aw + (size_t)a / 64] >> (a & 63)) & 1ull) k[(size_t)j >> 6] |= 1ull << (j & 63);
    }
}

void XdemSampler::xor_beta(uint64_t* row, uint64_t z) const {
    for (int t = 0; z; ++t, z >>= 8) {
        const unsigned v = (unsigned)(z & 255u);
        if (!v) continue;
        const uint64_t* src = &btab_[((size_t)t * 256 + v) * WF_];
        for (int w = 0; w < WF_; ++w) row[w] ^= src[w];
    }
}

// The multi-word fast path's coordinates (r > 64 or rho > 64; build_beta's caller), CW_ = ceil(max(r, rho) / 64)
// words. gamma: the canonical basis of span(img_a), the fully reduced echelon with pivot = the highest bit, in
// ascending pivot order P_0 < P_1 < ...: a vector v of the span is sum_l v[P_l] gamma_l, so its gamma coordinates are
// its bits at the pivots, and the highest set coordinate of v is the one of its highest bit. law_into's coin RREF
// (fully reduced, pivot = highest bit, File order = internal order) and its offset reduced modulo the coins are
// therefore the same objects in gamma coordinates (post_multi). beta (build_beta's: beta_D, the independent detector
// parts, then beta_L, zero detector part) is kept for the status C: GbW_[j] = beta_j's gamma coordinates, betaO_ its
// observable part.
void XdemSampler::build_multi() {
    std::vector<std::vector<uint64_t>> bD, bL;
    std::vector<int> pD, pL;
    // each axis' coordinates over beta_D (dw words) and beta_L (lw words), and the axis that created each beta vector
    // (image_kappa): rank beta_D <= min(A, D), rank beta_L <= min(A, O)
    const size_t dw = words_for(std::min(A_, D_)), lw = words_for(std::min(A_, O_));
    std::vector<uint64_t> cD((size_t)A_ * dw, 0), cL((size_t)A_ * lw, 0);
    std::vector<int> crD, crL;
    for (int a = 0; a < A_; ++a) {
        std::vector<uint64_t> v(img_.begin() + (long)a * WF_, img_.begin() + (long)(a + 1) * WF_);
        uint64_t* ca = cD.data() + (size_t)a * dw;
        for (size_t i = 0; i < bD.size(); ++i)
            if ((v[(size_t)pD[i] >> 6] >> (pD[i] & 63)) & 1ull) {
                for (int w = 0; w < WF_; ++w) v[(size_t)w] ^= bD[i][(size_t)w];
                ca[i >> 6] ^= 1ull << (i & 63);
            }
        int pd = -1;
        for (int w = 0; w < WD_ && pd < 0; ++w)
            if (v[(size_t)w]) pd = w * 64 + __builtin_ctzll(v[(size_t)w]);
        if (pd >= 0) {
            ca[bD.size() >> 6] ^= 1ull << (bD.size() & 63);
            crD.push_back(a);
            pD.push_back(pd);
            bD.push_back(std::move(v));
            continue;
        }
        uint64_t* la = cL.data() + (size_t)a * lw;
        for (size_t i = 0; i < bL.size(); ++i)
            if ((v[(size_t)pL[i] >> 6] >> (pL[i] & 63)) & 1ull) {
                for (int w = 0; w < WF_; ++w) v[(size_t)w] ^= bL[i][(size_t)w];
                la[i >> 6] ^= 1ull << (i & 63);
            }
        const int ph = highest(v.data(), WF_);
        if (ph >= 0) {
            la[bL.size() >> 6] ^= 1ull << (bL.size() & 63);
            crL.push_back(a);
            pL.push_back(ph);
            bL.push_back(std::move(v));
        }
    }
    const int rD = (int)bD.size(), rL = (int)bL.size();
    rho_ = rD + rL;
    rhoL_ = rL;
    CW_ = (std::max(r_, rho_) + 63) / 64;
    if (CW_ < 2) throw std::logic_error("xdem: multi-word coordinates for a one-word file (internal)");
    if (CW_ == 5 || CW_ == 7) ++CW_;   // post_mw's instances: 2, 3, 4, 6, 8 words (zero padding), else run time
    const size_t CW = (size_t)CW_;
    GcW_.assign((size_t)A_ * CW, 0);
    bb_.assign((size_t)A_, 0);
    for (int a = 0; a < A_; ++a) bb_[(size_t)a] = (uint8_t)F_.axes[(size_t)a].b;
    std::vector<std::vector<uint64_t>> beta = bD;
    for (auto& v : bL) beta.push_back(v);
    // gamma: fully reduced echelon of beta (pivot = highest bit), then ascending pivot
    std::vector<std::vector<uint64_t>> g;
    std::vector<int> gp;
    for (const auto& b : beta) {
        std::vector<uint64_t> v = b;
        for (size_t i = 0; i < g.size(); ++i)
            if ((v[(size_t)gp[i] >> 6] >> (gp[i] & 63)) & 1ull)
                for (int w = 0; w < WF_; ++w) v[(size_t)w] ^= g[i][(size_t)w];
        const int p = highest(v.data(), WF_);
        if (p < 0) throw std::logic_error("xdem: dependent beta (internal)");
        for (size_t i = 0; i < g.size(); ++i)
            if ((g[i][(size_t)p >> 6] >> (p & 63)) & 1ull)
                for (int w = 0; w < WF_; ++w) g[i][(size_t)w] ^= v[(size_t)w];
        g.push_back(std::move(v));
        gp.push_back(p);
    }
    std::vector<int> go((size_t)rho_);
    for (int j = 0; j < rho_; ++j) go[(size_t)j] = j;
    std::sort(go.begin(), go.end(), [&](int x, int y) { return gp[(size_t)x] < gp[(size_t)y]; });
    auto gcoords = [&](const uint64_t* v, uint64_t* out) {   // gamma coordinates of v in span(img): its pivot bits
        for (int j = 0; j < rho_; ++j) {
            const int p = gp[(size_t)go[(size_t)j]];
            if ((v[(size_t)p >> 6] >> (p & 63)) & 1ull) out[(size_t)j >> 6] |= 1ull << (j & 63);
        }
    };
    gntab_ = (rho_ + 7) / 8;
    gtab_.assign((size_t)gntab_ * 256 * WF_, 0);
    for (int t = 0; t < gntab_; ++t)
        for (int v = 1; v < 256; ++v) {
            const int j = t * 8 + __builtin_ctz((unsigned)v);
            uint64_t* dst = &gtab_[((size_t)t * 256 + (size_t)v) * WF_];
            const uint64_t* prev = &gtab_[((size_t)t * 256 + (size_t)(v & (v - 1))) * WF_];
            for (int w = 0; w < WF_; ++w) dst[w] = prev[w] ^ (j < rho_ ? g[(size_t)go[(size_t)j]][(size_t)w] : 0ull);
        }
    std::vector<uint64_t> chk((size_t)WF_);
    for (int a = 0; a < A_; ++a) {   // img_a = gamma . (its gamma coordinates): the span property, checked once
        uint64_t* c = &GcW_[(size_t)a * CW];
        gcoords(&img_[(size_t)a * WF_], c);
        std::fill(chk.begin(), chk.end(), 0ull);
        xor_gamma(chk.data(), c);
        if (!std::equal(chk.begin(), chk.end(), img_.begin() + (long)a * WF_))
            throw std::logic_error("xdem: gamma coordinates (internal)");
    }
    {   // kappa in image coordinates (over gamma), S from the beta coordinates (beta_D then beta_L)
        const size_t bw = words_for(rho_);
        std::vector<uint64_t> tb((size_t)A_ * bw, 0);
        for (int a = 0; a < A_; ++a) {
            uint64_t* o = tb.data() + (size_t)a * bw;
            for (size_t u = 0; u < dw; ++u)
                for (uint64_t x = cD[(size_t)a * dw + u]; x; x &= x - 1) {
                    const size_t j = u * 64 + (size_t)__builtin_ctzll(x);
                    o[j >> 6] |= 1ull << (j & 63);
                }
            for (size_t u = 0; u < lw; ++u)
                for (uint64_t x = cL[(size_t)a * lw + u]; x; x &= x - 1) {
                    const size_t j = (size_t)rD + u * 64 + (size_t)__builtin_ctzll(x);
                    o[j >> 6] |= 1ull << (j & 63);
                }
        }
        crD.insert(crD.end(), crL.begin(), crL.end());
        creator_ = crD;
        KgW_.assign((size_t)A_ * CW, 0);
        image_kappa(tb.data(), bw, creator_, GcW_.data(), CW, KgW_.data(), CW);
    }
    GbW_.assign((size_t)rho_ * CW, 0);
    for (int j = 0; j < rho_; ++j) gcoords(beta[(size_t)j].data(), &GbW_[(size_t)j * CW]);
    betaO_.assign((size_t)rho_ * WO_, 0);
    for (int j = 0; j < rho_; ++j)
        for (int w = 0; w < WO_; ++w) betaO_[(size_t)j * WO_ + w] = beta[(size_t)j][(size_t)(WD_ + w)];
    // the law readers' beta data (build_beta's): beta_D's detector parts and pivots, beta_L's observable parts and
    // pivots (highest observable bit)
    bDd_.clear();
    bLo_.clear();
    for (int i = 0; i < rD; ++i) bDd_.insert(bDd_.end(), bD[(size_t)i].begin(), bD[(size_t)i].begin() + WD_);
    for (int i = 0; i < rL; ++i) bLo_.insert(bLo_.end(), bL[(size_t)i].begin() + WD_, bL[(size_t)i].end());
    pD_ = pD;
    pL_.clear();
    for (int p : pL) pL_.push_back(p - 64 * WD_);
    fast_ = true;
}

void XdemSampler::xor_gamma(uint64_t* row, const uint64_t* z) const {
    const int WF = WF_;
    for (int t = 0; t < gntab_; ++t) {
        const unsigned v = (unsigned)(z[t >> 3] >> (8 * (t & 7))) & 255u;
        if (!v) continue;
        const uint64_t* src = &gtab_[((size_t)t * 256 + v) * WF];
        for (int w = 0; w < WF; ++w) row[w] ^= src[w];
    }
}

size_t XdemSampler::mw_words() const {
    // post_multi's layout: 7 vectors, 4 * rho (the law matrix, the elimination's pivots and combinations, the kernel
    // basis)
    const size_t CW = (size_t)CW_, R = (size_t)std::max(rho_, 1);
    return 7 * CW + 4 * R * CW;
}

XdemSampler::~XdemSampler() = default;

int XdemSampler::cspace_dim(int id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (id < 0 || (size_t)id >= cdim_.size()) throw std::out_of_range("XdemSampler: cspace id out of range");
    return cdim_[(size_t)id];
}

std::vector<std::vector<uint64_t>> XdemSampler::cspaces_snapshot(size_t first, size_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t a = std::min(first, cspaces_.size());
    const size_t b = a + std::min(n, cspaces_.size() - a);
    return std::vector<std::vector<uint64_t>>(cspaces_.begin() + (long)a, cspaces_.begin() + (long)b);
}

size_t XdemSampler::num_cspaces() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cspaces_.size();
}

XdemSampler::ThreadState& XdemSampler::slot(int t) {
    while ((int)ts_.size() <= t) {
        auto T = std::make_unique<ThreadState>();
        if (CW_) {   // post_multi's scratch
            T->mws.assign(mw_words(), 0);
            T->mwi.assign((size_t)3 * (size_t)std::max(rho_, 1) + (size_t)65 * (size_t)CW_, 0);
            T->lktW.assign((size_t)std::max(rho_, 1) * (size_t)CW_, 0);
            T->lrw.assign(((size_t)2 * (size_t)std::max(rho_, 1) + 6) * (size_t)CW_ + (size_t)WF_ + 1, 0);
            T->lrp.assign((size_t)64 * (size_t)CW_, 0);   // law_given's pivot index (bit -> vector)
        }
        if (!fast_) {   // exact-law cache (table, keys, data, entries): used only by lookup(), never on the fast path
            size_t tc = 16;
            while (tc < 2 * cap_) tc <<= 1;
            T->table.assign(tc, -1);
            T->mask = tc - 1;
        }
        T->ctab.push_back({});   // local id 0 = {0}: the empty form
        T->ckoff.assign(2, 0);
        T->chash.push_back(hash_c(nullptr, 0));
        T->cslot.assign(64, -1);
        T->cmask = 63;
        T->cslot[T->chash[0] & T->cmask] = 0;
        T->remap.push_back(0);
        T->stamp.push_back(0);
        T->plo.assign((size_t)WA_ + 1, 0);
        T->phi.assign((size_t)WA_ + 1, 0);
        T->pdirty.assign((size_t)(WA_ + 63) / 64 + 1, 0);
        T->sgn.assign((size_t)WA_ + 1, 0);
        T->odd.assign((size_t)A_, 0);
        ts_.push_back(std::move(T));
    }
    return *ts_[(size_t)t];
}

std::vector<std::vector<std::pair<uintptr_t, size_t>>> XdemSampler::slot_scratch_spans() const {
    std::vector<std::vector<std::pair<uintptr_t, size_t>>> out;
    for (const auto& T : ts_) {
        std::vector<std::pair<uintptr_t, size_t>> v;
        v.emplace_back((uintptr_t)T.get(), sizeof(ThreadState));
        auto add = [&](const auto& vec) {
            if (vec.capacity())   // the block XdemLineAlloc hands out: padded to whole 64-byte lines
                v.emplace_back((uintptr_t)vec.data(), (vec.capacity() * sizeof(vec[0]) + 63) / 64 * 64);
        };
        add(T->plo);
        add(T->phi);
        add(T->pdirty);
        add(T->sgn);
        add(T->keybuf);
        add(T->odd);
        add(T->ckey);
        add(T->mws);
        add(T->mwi);
        out.push_back(std::move(v));
    }
    return out;
}

std::pair<long long, long long> XdemSampler::law_kappa_weights() const {
    long long f = 0, l = 0;
    for (size_t a = 0; a < law_k_.size(); ++a) {
        for (uint64_t w : F_.axes[a].k) f += __builtin_popcountll(w);
        for (uint64_t w : law_k_[a]) l += __builtin_popcountll(w);
    }
    return {f, l};
}

int XdemSampler::global_cspace(std::vector<uint64_t>& c, uint64_t h) {
    size_t i = h & cslot_mask_;
    for (int32_t x; (x = cslot_[i]) >= 0; i = (i + 1) & cslot_mask_)
        if (chash_[(size_t)x] == h && cspaces_[(size_t)x] == c) {
            std::vector<uint64_t>().swap(c);
            return x;
        }
    const int id = (int)cspaces_.size();
    const int dim = WO_ ? (int)(c.size() / (size_t)WO_) : 0;
    std::vector<int> piv((size_t)dim);
    for (int k = 0; k < dim; ++k) piv[(size_t)k] = highest(&c[(size_t)k * WO_], WO_);
    // every allocation first (a std::bad_alloc leaves the table and its index consistent), then no-throw appends
    auto room = [](auto& v) {
        if (v.size() == v.capacity()) v.reserve(2 * v.size() + 16);
    };
    room(cspaces_);
    room(cdim_);
    room(cpiv_);
    room(chash_);
    if (2 * ((size_t)id + 1) > cslot_.size()) {   // grow at load 1/2: rehash the ids by their stored hashes
        std::vector<int32_t> ns(cslot_.size() * 2, -1);
        const size_t m = ns.size() - 1;
        for (int32_t x : cslot_)
            if (x >= 0) {
                size_t j = chash_[(size_t)x] & m;
                while (ns[j] >= 0) j = (j + 1) & m;
                ns[j] = x;
            }
        cslot_.swap(ns);
        cslot_mask_ = m;
        i = h & m;
        while (cslot_[i] >= 0) i = (i + 1) & m;
    }
    cspaces_.push_back(std::move(c));
    std::vector<uint64_t>().swap(c);
    cdim_.push_back(dim);
    cpiv_.push_back(std::move(piv));
    chash_.push_back(h);
    cslot_[i] = id;
    return id;
}

void XdemSampler::compose_ext(ThreadState& T, const uint32_t* xo, uint32_t n) const {
    // SPEC §1.2: eta_a = sum_j (-1)^{tau_{<j}(a)} E_{i_j, a} (mod 4), lines in file order, bit-sliced one axis word at a
    // time: a line's group (w, e0 = its odd bits, e1 = its 2-bits) is negated where the sign register is set (-1 = 3,
    // -3 = 1 mod 4 flip the 2-bit of an odd exponent; -2 = 2), e1 ^= e0 & sgn[w], then added to the planes with a
    // carry. The register holds tau_{<j}: line j - 1's tau is applied just before line j (the last line's is read by
    // no later line).
    uint64_t* lo = T.plo.data();
    uint64_t* hi = T.phi.data();
    uint64_t* dm = T.pdirty.data();
    uint64_t* sg = T.sgn.data();
    const uint64_t* xa = xa_.data();
    const uint64_t* prev = nullptr;
    for (uint32_t j = 0; j < n; ++j) {
        const uint64_t* r = xa + xo[j];
        if (prev) {
            const uint64_t* t = prev + 1 + 3 * (prev[0] & 0xffffffffu);
            for (uint64_t k = 0; k < (prev[0] >> 32); ++k, t += 2) sg[t[0]] ^= t[1];
        }
        const uint64_t* g = r + 1;
        for (uint64_t k = 0; k < (r[0] & 0xffffffffu); ++k, g += 3) {
            const uint64_t w = g[0], e0 = g[1], e1 = g[2] ^ (g[1] & sg[w]);
            const uint64_t c = lo[w] & e0;
            lo[w] ^= e0;
            hi[w] ^= e1 ^ c;
            dm[w >> 6] |= 1ull << (w & 63);
        }
        prev = r;
    }
    // clear the register: every set bit lies in a word some applied tau entry names
    for (uint32_t j = 0; j + 1 < n; ++j) {
        const uint64_t* q = xa + xo[j];
        const uint64_t* t = q + 1 + 3 * (q[0] & 0xffffffffu);
        for (uint64_t k = 0; k < (q[0] >> 32); ++k, t += 2) sg[t[0]] = 0;
    }
}

void XdemSampler::sort_ext(ThreadState& T, size_t np, size_t n) const {
    // counting sort by shot (inclusive prefix, then a reverse scatter: fp[s] = shot s's first entry, fp[n] = the total;
    // stable, so each shot's list keeps the walk order). A shot's list is put into file order where it is composed
    // (sort_ids before compose_ext), so a shot that is never composed (postselection's early rejections) is not sorted.
    T.epos.assign(n + 1, 0);
    if (!np) return;
    if (T.exo.size() < np) T.exo.resize(np);
    uint32_t* fp = T.epos.data();
    const uint64_t* pr = T.epair.data();
    for (size_t i = 0; i < np; ++i) ++fp[pr[i] >> 32];
    uint32_t acc = 0;
    for (size_t s = 0; s < n; ++s) fp[s] = acc += fp[s];
    fp[n] = acc;
    uint32_t* kid = T.exo.data();
    for (size_t i = np; i-- > 0;) kid[--fp[pr[i] >> 32]] = (uint32_t)pr[i];
}

void XdemSampler::canonical_ids(const std::vector<int>& line_ids, const char* who, uint64_t* row,
                                std::vector<uint32_t>& ext) const {
    std::vector<int> ids(line_ids);
    for (int id : ids)
        if (id < 0 || (size_t)id >= L_)
            throw std::out_of_range(std::string(who) + ": line id " + std::to_string(id) + " out of range");
    std::sort(ids.begin(), ids.end());
    std::vector<int> order;
    for (int id : ids) {   // file order; a line listed twice cancels (M_i^2 = I, SPEC §1.2)
        if (!order.empty() && order.back() == id) order.pop_back();
        else order.push_back(id);
    }
    ext.clear();
    for (int id : order) {
        if (row)
            for (uint64_t t = loff_[(size_t)id]; t < loff_[(size_t)id + 1]; ++t) row[lent_[t].word] ^= lent_[t].mask;
        if (xo_of_[(size_t)id]) ext.push_back(xo_of_[(size_t)id]);
    }
}

int XdemSampler::resolve(ThreadState& T, uint64_t* row) {
    // general path: eta_a in {2, 3} adds img_a (eta = 3 is eta = 1 plus img_a modulo the coins); the odd-axis set (the
    // low plane) keys the law cache. The planes are drained (zeroed) here.
    T.keybuf.assign((size_t)WA_, 0ull);
    bool odd = false;
    uint64_t* dm = T.pdirty.data();
    for (int di = 0; di * 64 < WA_; ++di) {
        for (uint64_t x = dm[di]; x; x &= x - 1) {
            const int w = di * 64 + __builtin_ctzll(x);
            const uint64_t lw = T.plo[(size_t)w], hw = T.phi[(size_t)w];
            T.plo[(size_t)w] = T.phi[(size_t)w] = 0;
            for (uint64_t y = hw; y; y &= y - 1) {
                const uint64_t* im = &img_[(size_t)(w * 64 + __builtin_ctzll(y)) * WF_];
                for (int v = 0; v < WF_; ++v) row[v] ^= im[v];
            }
            T.keybuf[(size_t)w] = lw;
            odd |= lw != 0;
        }
        dm[di] = 0;
    }
    if (!odd) return -1;
    return lookup(T, T.keybuf.data());
}

int XdemSampler::local_cspace(ThreadState& T, const uint64_t* key, size_t n, uint64_t h) const {
    for (size_t i = h & T.cmask;; i = (i + 1) & T.cmask) {
        const int32_t x = T.cslot[i];
        if (x < 0) return -1;
        if (T.chash[(size_t)x] == h) {
            const size_t a = T.ckoff[(size_t)x], b = T.ckoff[(size_t)x + 1];
            if (b - a == n && std::equal(key, key + n, T.ckeys.data() + a)) return x;
        }
    }
}

int XdemSampler::local_insert(ThreadState& T, const uint64_t* key, size_t n, uint64_t h,
                              std::vector<uint64_t>&& content) {
    const int id = (int)T.ctab.size();
    size_t i = h & T.cmask;
    while (T.cslot[i] >= 0) i = (i + 1) & T.cmask;
    // every allocation first (a std::bad_alloc leaves the slot's table consistent), then no-throw appends
    auto room = [](auto& v, size_t k) {
        if (v.size() + k > v.capacity()) v.reserve(std::max(2 * v.capacity(), v.size() + k) + 16);
    };
    room(T.ctab, 1);
    room(T.remap, 1);
    room(T.stamp, 1);
    room(T.chash, 1);
    room(T.ckoff, 1);
    room(T.ckeys, n);
    if (2 * ((size_t)id + 1) > T.cslot.size()) {   // grow at load 1/2: rehash the ids by their stored hashes
        std::vector<int32_t> ns(T.cslot.size() * 2, -1);
        const size_t m = ns.size() - 1;
        for (int32_t x : T.cslot)
            if (x >= 0) {
                size_t j = T.chash[(size_t)x] & m;
                while (ns[j] >= 0) j = (j + 1) & m;
                ns[j] = x;
            }
        T.cslot.swap(ns);
        T.cmask = m;
        i = h & m;
        while (T.cslot[i] >= 0) i = (i + 1) & m;
    }
    T.ckeys.insert(T.ckeys.end(), key, key + n);   // (before the move: key may be the content itself)
    T.ckoff.push_back(T.ckeys.size());
    T.ctab.push_back(std::move(content));
    T.remap.push_back(-1);
    T.stamp.push_back(0);
    T.chash.push_back(h);
    T.cslot[i] = id;
    return id;
}

template <bool PS, bool LAW>
int XdemSampler::post_fast(ThreadState& T, uint64_t* row, uint64_t* cpos, std::vector<uint64_t>* coins_out,
                           const XdemSpanTest* zt) {
    // fast path (rho <= 64, r <= 64): everything in coordinates over the basis beta of span(img_a). sigma in
    // GF(2)^rho are functionals s -> (s.beta_j); W = coins^perp <-> ker N_O, N_O sigma = sum_j sigma_j col_j,
    // col_j = XOR_{a in O, T_a bit j} k'_a with kappa in image coordinates (Kg_, image_kappa: the same kernel), so the
    // first rho bits of col_j are row j of the symmetric G = sum_{a in O} T_a T_a^T (graph-state form). The W phase
    // f(sigma) = |{a in O : T_a.sigma = 1}| / 2 + sum_{a in O} b_a (T_a.sigma) (mod 2) is linear on ker N_O (§1.3),
    // and |{a in O : T_a.sigma odd}| = sum_a (T_a.sigma)^2 = sum_{i in sigma} d_i + 2 sum_{i < j in sigma} G_ij
    // (mod 4), d_i = |{a in O : T_a bit i}| mod 4, i.e. X = 2 |sigma & d1| + sum_{i in sigma} |G_i & sigma| (even on
    // ker G; law_kernel1); omega with omega.sigma_q = f(sigma_q) on the private-pivot kernel basis is the offset (beta
    // coordinates); coins in beta coordinates C_j = XOR_{a in O, k_a bit j} T_a (the file's kappa: the coin stream);
    // outcome = row ^ beta.(zhi ^ omega ^ sum_j t_j C_j). The composite's eta is in T's planes (compose_ext).
    uint64_t zhi = 0;
    int m = 0;
    {
        uint64_t* dm = T.pdirty.data();
        uint64_t* lo = T.plo.data();
        uint64_t* hi = T.phi.data();
        for (int di = 0; di * 64 < WA_; ++di) {   // drain the planes' dirty words
            for (uint64_t x = dm[di]; x; x &= x - 1) {
                const int w = di * 64 + __builtin_ctzll(x);
                for (uint64_t y = hi[w]; y; y &= y - 1) zhi ^= Tc_[(size_t)(w * 64 + __builtin_ctzll(y))];
                for (uint64_t y = lo[w]; y; y &= y - 1) T.odd[(size_t)m++] = w * 64 + __builtin_ctzll(y);
                lo[w] = hi[w] = 0;
            }
            dm[di] = 0;
        }
    }
    if (m == 0) {
        xor_beta(row, zhi);
        if (PS && !zt->zero(row)) return -1;
        return 0;
    }
    // the law matrix (col_j; its rho low bits: G's row j) and, alongside, d mod 4 (d0, d1) and lin = sum_a b_a T_a
    uint64_t col[64], d0 = 0, d1 = 0, lin = 0;
    for (int j = 0; j < rho_; ++j) col[j] = 0;
    for (int i = 0; i < m; ++i) {
        const int a = T.odd[(size_t)i];
        const uint64_t t = Tc_[(size_t)a], k = Kg_[(size_t)a];
        for (uint64_t tt = t; tt; tt &= tt - 1) col[__builtin_ctzll(tt)] ^= k;
        d1 ^= d0 & t;
        d0 ^= t;
        lin ^= (0 - (uint64_t)bb_[(size_t)a]) & t;
    }
    // the kernel of N_O (columns in ascending order; the beta_D coordinates come first) and the W phase on its basis
    // -> omega (no pass over the odd axes): law_kernel1
    Law1 Lw;
    law_kernel1(col, rho_, rho_ - rhoL_, d1, lin, Lw);
    const uint64_t* Kt = Lw.Kt;
    const int npv = Lw.npv, npvD = Lw.npvD;
    const uint64_t omega = Lw.omega;
    if constexpr (LAW) {   // law_of: the offset (no coin draw) and the kernel basis; no coins_out, no status C
        xor_beta(row, zhi ^ omega);
        std::copy(Kt, Kt + rho_, T.lkt);
        T.lnpv = npv;
        T.lnpvD = npvD;
        return npv ? 1 : 0;
    }
    // coins in beta coordinates; rank = npv (= rank of N_O)
    uint64_t tau = 0;
    if (npv) {
        uint64_t t = 0;
        if (cpos) {
            *cpos += krng::kGamma;
            t = krng::mix(*cpos) & rmask_;
            ++T.coin_shots;
        }
        for (int i = 0; i < m; ++i) {
            const int a = T.odd[(size_t)i];
            tau ^= Tc_[(size_t)a] & (0 - (uint64_t)__builtin_parityll(t & Kc_[(size_t)a]));
        }
    }
    xor_beta(row, zhi ^ omega ^ tau);
    if (coins_out) {
        coins_out->clear();
        for (int j = 0; j < r_; ++j) {
            uint64_t cj = 0;
            for (int i = 0; i < m; ++i)
                if ((Kc_[(size_t)T.odd[(size_t)i]] >> j) & 1ull) cj ^= Tc_[(size_t)T.odd[(size_t)i]];
            coins_out->push_back(cj);
        }
    }
    // postselected: a shot whose final detector row fails the mask is rejected here, before its status C
    if (PS && !zt->zero(row)) return -1;
    // dim C = rank(coins) - rank(coins restricted to the beta_D coordinates) = npv - npvD
    if (!rhoL_) return 0;
    if (npv - npvD == 0) return 0;
    // status C from the kernel basis: the coin span in beta coordinates is colspan(M) = W^perp, W = ker N_O with the
    // basis sigma_q above (Kt[j] bit q = sigma_q[j]). The beta_D have independent detector parts and the beta_L have
    // none, so C_beta = {sigma in W^perp : sigma_D = 0}: a sigma supported on the beta_L coordinates lies in C iff
    // XOR_{j in L, sigma_j = 1} Kt[j] = 0, i.e. C = the linear dependencies among the rho_L words Kt[j],
    // j in [rho - rho_L, rho) (pivot-indexed elimination, the combinations in y coordinates: column j's is yA_'s row
    // j - rho_D). Then the y-RREF (fast_cspace).
    uint64_t kern[64];
    int nk = 0;
    {
        uint64_t lv[64], lc[64], pmask = 0;
        int pidx[64];
        int nl = 0;
        const int rD = rho_ - rhoL_;
        for (int j = rD; j < rho_; ++j) {
            uint64_t v = Kt[j], c = yA_[(size_t)(j - rD)];
            reduce_low1(v, c, pmask, pidx, lv, lc);
            if (v) {
                lv[nl] = v;
                lc[nl] = c;
                pidx[__builtin_ctzll(v)] = nl;
                pmask |= v & (0 - v);
                ++nl;
            } else {
                kern[nk++] = c;
            }
        }
    }
    // live check of dim C = npv - npvD (one compare per status shot)
    if (nk != npv - npvD) throw std::logic_error("xdem: status C dimension mismatch on the fast path (internal)");
    if (!nk) return 0;
    // the fully reduced RREF in y coordinates (pivot = highest bit, descending)
    uint64_t pm;
    int pidx[64], ord[64];
    const int nr = rref_high<1>(kern, nk, 1, &pm, pidx);
    rows_desc(&pm, 1, pidx, ord);
    T.ckey.resize((size_t)nr);
    for (int i = 0; i < nr; ++i) T.ckey[(size_t)i] = kern[ord[i]];
    return fast_cspace(T, nr);
}

int XdemSampler::cspace_of_beta_key(const uint64_t* key, int nk) {
    if (nk == 0) return 0;
    if (!fast_ || A_ == 0)
        throw std::invalid_argument("XdemSampler.cspace_of_beta_key: the sampler is not on the fast path with axes");
    const int cw = coord_words();   // words per key vector: 1, or CW_ on a multi-word file
    if (nk < 0 || nk > rhoL_)
        throw std::invalid_argument("XdemSampler.cspace_of_beta_key: a key has at most rhoL = " +
                                    std::to_string(rhoL_) + " vectors");
    // beta_L coordinates are bits [rho - rhoL, rho); a fully reduced basis with distinct descending pivots (highest bit)
    const int rD = rho_ - rhoL_;
    std::vector<int> piv((size_t)nk);
    for (int i = 0; i < nk; ++i) {
        const uint64_t* v = key + (size_t)i * cw;
        int p = -1;
        for (int u = cw; u-- > 0 && p < 0;)
            if (v[u]) p = u * 64 + 63 - __builtin_clzll(v[u]);
        bool outside = false;
        for (int u = 0; u < cw; ++u) {
            const int lo = std::max(0, std::min(64, rD - 64 * u)), hi = std::max(0, std::min(64, rho_ - 64 * u));
            const uint64_t m = (hi >= 64 ? ~0ull : (1ull << hi) - 1) & ~(lo >= 64 ? ~0ull : (1ull << lo) - 1);
            outside = outside || (v[u] & ~m);
        }
        if (p < 0 || outside || (i && p >= piv[(size_t)i - 1]))
            throw std::invalid_argument("XdemSampler.cspace_of_beta_key: the key is not a fully reduced basis in "
                                        "beta_L coordinates (nonzero vectors, descending pivots)");
        piv[(size_t)i] = p;
    }
    for (int i = 0; i < nk; ++i)
        for (int k = 0; k < nk; ++k)
            if (k != i && ((key[(size_t)k * cw + (size_t)(piv[(size_t)i] >> 6)] >> (piv[(size_t)i] & 63)) & 1ull))
                throw std::invalid_argument("XdemSampler.cspace_of_beta_key: the key is not fully reduced");
    // upstream's canonical form (fast_cspace): the key's y coordinates (y(x) = sum_j x_j yA_[j - rho_D], linear and
    // injective on beta_L), their fully reduced RREF (pivot = highest bit, descending), hashed; the content row by row
    // from yE_. The same C content and the same hash as a thread slot's, so the id is the one sample() gives.
    const int yW = yW_, WO = WO_;
    std::vector<uint64_t> y((size_t)nk * (size_t)yW, 0ull);
    for (int i = 0; i < nk; ++i) {
        uint64_t* o = y.data() + (size_t)i * yW;
        for (int u = 0; u < cw; ++u)
            for (uint64_t x = key[(size_t)i * cw + u]; x; x &= x - 1) {
                const uint64_t* ya = yA_.data() + (size_t)(u * 64 + __builtin_ctzll(x) - rD) * yW;
                for (int q = 0; q < yW; ++q) o[q] ^= ya[q];
            }
    }
    std::vector<uint64_t> pm((size_t)yW, 0ull);
    std::vector<int> pidx((size_t)yW * 64, 0), ord((size_t)nk, 0);
    const int nr = rref_high<0>(y.data(), nk, yW, pm.data(), pidx.data());
    if (nr != nk) throw std::logic_error("xdem: cspace_of_beta_key: dependent y rows (internal)");
    rows_desc(pm.data(), yW, pidx.data(), ord.data());
    std::vector<uint64_t> yr((size_t)nr * (size_t)yW);
    for (int i = 0; i < nr; ++i)
        std::copy(y.begin() + (long)ord[(size_t)i] * yW, y.begin() + (long)(ord[(size_t)i] + 1) * yW,
                  yr.begin() + (long)i * yW);
    std::vector<uint64_t> c((size_t)nr * (size_t)WO, 0ull);
    for (int i = 0; i < nr; ++i) {
        uint64_t* o = c.data() + (size_t)i * WO;
        const uint64_t* yi = yr.data() + (size_t)i * yW;
        for (int u = 0; u < yW; ++u)
            for (uint64_t x = yi[u]; x; x &= x - 1) {
                const uint64_t* ek = yE_.data() + (size_t)(u * 64 + __builtin_ctzll(x)) * WO;
                for (int w = 0; w < WO; ++w) o[w] ^= ek[w];
            }
    }
    const uint64_t h = hash_c(yr.data(), yr.size());
    std::lock_guard<std::mutex> lk(mu_);
    return global_cspace(c, h);
}

XdemDeviceTables XdemSampler::device_tables() const {
    XdemDeviceTables t;
    t.D = D_;
    t.O = O_;
    t.W = W_;
    t.WD = WD_;
    t.WO = WO_;
    t.WF = WF_;
    t.A = A_;
    t.WA = WA_;
    // the one-word tables only when the one-word coordinates are in use (fast1_: btab_, Tc_, ... built); a multi-word
    // file (CW_ >= 2) builds none of them: btab_ is empty there, so reading rho_ beta vectors from it is out of range
    const bool one = fast1_, multi = fast_ && CW_ >= 2;
    t.rho = one || multi ? rho_ : 0;
    t.rhoL = one || multi ? rhoL_ : 0;
    t.r = r_;
    t.coord_words = coord_words();
    t.fast = one;
    t.rmask = one ? rmask_ : 0;
    t.mean_fired = mean_fired_;
    t.mean_ext = mean_ext_;
    t.chain_off.push_back(0);
    for (const Chain& ch : chains_) {
        t.chain_p.push_back(ch.p);
        for (size_t m = 0; m < ch.mem.size(); ++m) {
            t.mem_beg.push_back(ch.mem[m].beg);
            t.mem_cnt.push_back(ch.mem[m].cnt);
            t.mem_xo.push_back(ch.mem[m].xo);
            t.mem_line.push_back((int32_t)ch.lines[m]);
        }
        t.chain_off.push_back(t.mem_beg.size());
    }
    t.ent_mask.reserve(lent_.size());
    t.ent_word.reserve(lent_.size());
    for (const LineEnt& e : lent_) {
        t.ent_mask.push_back(e.mask);
        t.ent_word.push_back(e.word);
    }
    t.xa = xa_;
    if (multi && A_ > 0) {   // build_multi's tables; gamma_l is the single-bit entry 1 << (l % 8) of byte table l / 8
        // kappa_a in CW words (the file's kappa, r bits; the sampler keeps only its image-coordinate form KgW_)
        t.KcW.assign((size_t)A_ * (size_t)CW_, 0);
        for (int a = 0; a < A_; ++a) {
            const Axis& ax = F_.axes[(size_t)a];
            std::copy(ax.k.begin(), ax.k.begin() + (long)std::min(ax.k.size(), (size_t)CW_), &t.KcW[(size_t)a * CW_]);
        }
        t.GcW = GcW_;
        t.bb = bb_;
        t.GbW = GbW_;
        t.gamma.reserve((size_t)rho_ * WF_);
        for (int l = 0; l < rho_; ++l) {
            const uint64_t* src = &gtab_[((size_t)(l / 8) * 256 + (size_t)(1u << (l % 8))) * WF_];
            t.gamma.insert(t.gamma.end(), src, src + WF_);
        }
        t.betaO = betaO_;
        t.bLo = bLo_;
        t.pL.assign(pL_.begin(), pL_.end());
        return t;
    }
    if (!one || A_ == 0) return t;
    t.Tc = Tc_;
    t.Kc = Kc_;
    t.bb = bb_;
    // beta vector j is the single-bit entry 1 << (j % 8) of byte table j / 8
    t.beta.reserve((size_t)rho_ * WF_);
    for (int j = 0; j < rho_; ++j) {
        const uint64_t* src = &btab_[((size_t)(j / 8) * 256 + (size_t)(1u << (j % 8))) * WF_];
        t.beta.insert(t.beta.end(), src, src + WF_);
    }
    t.betaO = betaO_;
    t.bLo = bLo_;
    t.pL.assign(pL_.begin(), pL_.end());
    return t;
}

XdemSpanTables XdemSampler::span_tables(const std::vector<uint8_t>& flags) const {
    if (flags.size() != (size_t)D_)
        throw std::invalid_argument("XdemSampler.span_tables: the mask needs " + std::to_string(D_) +
                                    " entries (got " + std::to_string(flags.size()) + ")");
    std::vector<uint8_t> f(flags.size());   // sample_ps's 0/1 mask
    for (size_t d = 0; d < flags.size(); ++d) f[d] = flags[d] != 0;
    const XdemSpanTest zt(D_, f.data(), img_.data(), (size_t)A_, (size_t)WF_);
    XdemSpanTables t;
    t.widx.assign(zt.word_index(), zt.word_index() + zt.words());
    t.wmask.assign(zt.word_mask(), zt.word_mask() + zt.words());
    t.basis = zt.basis();
    t.piv_word = zt.pivot_word();
    t.piv_bit = zt.pivot_bit();
    t.rank = zt.rank();
    return t;
}

int XdemSampler::fast_cspace(ThreadState& T, int nr) {
    // the shot's C as its y-RREF (nr rows of yW_ words in T.ckey): the slot's id; a new C gets its content, row i =
    // sum_k y_ik E_k (build_ycoords: the observable parts' fully reduced RREF, descending pivot, with no second RREF)
    const size_t n = (size_t)nr * (size_t)yW_;
    const uint64_t* key = T.ckey.data();
    const uint64_t h = hash_c(key, n);
    const int id = local_cspace(T, key, n, h);
    if (id >= 0) return id;
    const int WO = WO_;
    std::vector<uint64_t> c((size_t)nr * (size_t)WO, 0ull);
    for (int i = 0; i < nr; ++i) {
        uint64_t* o = c.data() + (size_t)i * WO;
        const uint64_t* y = key + (size_t)i * yW_;
        for (int u = 0; u < yW_; ++u)
            for (uint64_t x = y[u]; x; x &= x - 1) {
                const uint64_t* e = yE_.data() + (size_t)(u * 64 + __builtin_ctzll(x)) * WO;
                for (int w = 0; w < WO; ++w) o[w] ^= e[w];
            }
    }
    return local_insert(T, key, n, h, std::move(c));
}

template <int NW, bool PS, bool LAW>
int XdemSampler::post_multi(ThreadState& T, uint64_t* row, uint64_t* cpos, std::vector<uint64_t>* coins_out,
                            const XdemSpanTest* zt) {
    // post_fast's construction in CW words per vector (kappa' and col_l: rank kappa <= r bits; coordinates and
    // combinations: rho bits; kernel words: nker <= rho bits), in gamma coordinates (build_multi; kappa' = KgW_ in
    // image coordinates over gamma, so col_l's rho low bits are row l of G, and the W phase is the quadratic form) and
    // with the columns eliminated in DESCENDING order (pivot-indexed, as post_fast). Then each kernel vector sigma_q is
    // e_{depj[q]} plus pivot columns above depj[q]: the kernel basis is W's fully reduced echelon with pivot = lowest
    // bit, so its annihilator, the coin span, has the fully reduced echelon (pivot = highest bit)
    // coin_p = e_p + sum_{q : sigma_q[p] = 1} e_{depj[q]}, one per pivot column p (Kt[p] bit q = sigma_q[p]); and omega
    // (the W phases f(sigma_q) on the depj[q] coordinates, zero on the pivot columns) is the offset reduced modulo it.
    // Those are law_into's coins and offset coordinate by coordinate; the coins are drawn in descending pivot order
    // with ceil(k / 64) words of *cpos: the general path's bytes (xdem_sampler.hpp, Stream).
    const int CW = NW ? NW : CW_;
    const int R = rho_, rD = rho_ - rhoL_;
    uint64_t* zg = T.mws.data();   // gamma coordinates of the correction
    uint64_t* tv = zg + CW;        // a vector being reduced, and its combination
    uint64_t* tc = tv + CW;
    uint64_t* d0 = tc + CW;        // d = |{a in O : t_a bit l}| mod 4 (two planes), lin = sum_a b_a t_a
    uint64_t* d1 = d0 + CW;
    uint64_t* lin = d1 + CW;
    uint64_t* pmk = lin + CW;      // a pivot mask (the elimination's lowest bits; rref_high)
    uint64_t* col = pmk + CW;                 // R x CW: col_l (G's row l in the rho low bits)
    uint64_t* pvv = col + (size_t)R * CW;     // R x CW: the elimination's pivot vectors
    uint64_t* pvc = pvv + (size_t)R * CW;     // R x CW: their combinations (rho bits)
    uint64_t* Kt = pvc + (size_t)R * CW;      // R x CW: Kt[l] bit q = sigma_q[l]
    int* depj = T.mwi.data();
    int* pcol = depj + R;                     // pivot columns, descending
    int* ord = pcol + R;
    int* pidx = ord + R;                      // CW * 64, then law_kernelw's sw (CW)
    for (int q = 0; q < CW; ++q) zg[q] = 0;
    int m = 0;
    {
        uint64_t* dm = T.pdirty.data();
        uint64_t* plo = T.plo.data();
        uint64_t* phi = T.phi.data();
        for (int di = 0; di * 64 < WA_; ++di) {   // drain the planes' dirty words
            for (uint64_t x = dm[di]; x; x &= x - 1) {
                const int w = di * 64 + __builtin_ctzll(x);
                for (uint64_t y = phi[w]; y; y &= y - 1) {
                    const uint64_t* g = &GcW_[(size_t)(w * 64 + __builtin_ctzll(y)) * CW];
                    for (int q = 0; q < CW; ++q) zg[q] ^= g[q];
                }
                for (uint64_t y = plo[w]; y; y &= y - 1) T.odd[(size_t)m++] = w * 64 + __builtin_ctzll(y);
                plo[w] = phi[w] = 0;
            }
            dm[di] = 0;
        }
    }
    if (m == 0) {
        xor_gamma(row, zg);
        if (PS && !zt->zero(row)) return -1;
        return 0;
    }
    // the law matrix and, alongside, d mod 4 and lin (post_fast's)
    std::fill(col, col + (size_t)R * CW, 0ull);
    for (int q = 0; q < CW; ++q) d0[q] = d1[q] = lin[q] = 0;
    for (int i = 0; i < m; ++i) {
        const int a = T.odd[(size_t)i];
        const uint64_t* k = &KgW_[(size_t)a * CW];
        const uint64_t* t = &GcW_[(size_t)a * CW];
        for (int u = 0; u < CW; ++u)
            for (uint64_t tt = t[u]; tt; tt &= tt - 1) {
                uint64_t* c = col + (size_t)(u * 64 + __builtin_ctzll(tt)) * CW;
                for (int q = 0; q < CW; ++q) c[q] ^= k[q];
            }
        const uint64_t bm = bb_[(size_t)a] ? ~0ull : 0ull;
        for (int q = 0; q < CW; ++q) {
            d1[q] ^= d0[q] & t[q];
            d0[q] ^= t[q];
            lin[q] ^= bm & t[q];
        }
    }
    // the kernel of N_O, columns l = R - 1 .. 0, and the W phase on its basis (law_kernelw; col keeps G) -> omega on
    // the depj coordinates of zg, already reduced modulo the coins
    int npv = 0, nker = 0;
    law_kernelw<NW>(CW, R, true, col, d1, lin, tv, tc, pmk, pvv, pvc, pidx, pidx + 64 * CW, Kt, depj, pcol, npv, nker,
                    zg);
    if (npv) {
        // the coins coin_i = e_{pcol[i]} + scatter(Kt[pcol[i]]) (descending pivot = law_into's order): a selection
        // sets its pivot bits and XORs its Kt words into t, whose bits q then set depj[q]
        if (cpos) {
            ++T.coin_shots;
            for (int q = 0; q < CW; ++q) tc[q] = 0;
            for (int c0 = 0; c0 < npv; c0 += 64) {
                *cpos += krng::kGamma;
                uint64_t bits = krng::mix(*cpos);
                const int rem = npv - c0;
                if (rem < 64) bits &= (1ull << rem) - 1;
                for (; bits; bits &= bits - 1) {
                    const int p = pcol[c0 + __builtin_ctzll(bits)];
                    zg[p >> 6] ^= 1ull << (p & 63);
                    const uint64_t* kr = Kt + (size_t)p * CW;
                    for (int q = 0; q < CW; ++q) tc[q] ^= kr[q];
                }
            }
            for (int u = 0; u < CW; ++u)
                for (uint64_t x = tc[u]; x; x &= x - 1) {
                    const int j = depj[u * 64 + __builtin_ctzll(x)];
                    zg[j >> 6] ^= 1ull << (j & 63);
                }
        }
        if (coins_out) {
            coins_out->assign((size_t)npv * CW, 0ull);
            for (int i = 0; i < npv; ++i) {
                uint64_t* c = coins_out->data() + (size_t)i * CW;
                const int p = pcol[i];
                c[p >> 6] ^= 1ull << (p & 63);
                const uint64_t* kr = Kt + (size_t)p * CW;
                for (int u = 0; u < CW; ++u)
                    for (uint64_t x = kr[u]; x; x &= x - 1) {
                        const int j = depj[u * 64 + __builtin_ctzll(x)];
                        c[j >> 6] ^= 1ull << (j & 63);
                    }
            }
        }
    }
    xor_gamma(row, zg);
    if constexpr (LAW) {
        // the law for the scorer (law_row, kind 3): row = the offset (no coin draw); the kernel basis in beta
        // coordinates, lktW[j] = XOR_{l : beta_j has gamma coordinate l} Kt[l] (a vector x of beta coordinates is in
        // the coin span iff XOR_{j : x_j} lktW[j] = 0, as post_fast's Kt), and dim C = rho_L - rank{lktW[j] : j >=
        // rho_D} (post_fast's status rule) in lnpv
        uint64_t* KB = T.lktW.data();
        std::fill(KB, KB + (size_t)R * CW, 0ull);
        for (int j = 0; j < R; ++j) {
            uint64_t* kb = KB + (size_t)j * CW;
            const uint64_t* gb = &GbW_[(size_t)j * CW];
            for (int u = 0; u < CW; ++u)
                for (uint64_t x = gb[u]; x; x &= x - 1) {
                    const uint64_t* kr = Kt + (size_t)(u * 64 + __builtin_ctzll(x)) * CW;
                    for (int q = 0; q < CW; ++q) kb[q] ^= kr[q];
                }
        }
        int dimc = 0;
        if (rhoL_ && npv) {   // pivot-indexed (pivot = lowest bit; reduce_low), the echelon in col
            int nl = 0;
            for (int q = 0; q < CW; ++q) pmk[q] = 0;
            for (int j = rD; j < R; ++j) {
                std::copy(KB + (size_t)j * CW, KB + (size_t)(j + 1) * CW, tv);
                reduce_low<NW>(tv, nullptr, CW, pmk, pidx, col, nullptr);
                int lw = -1;
                for (int q = 0; q < CW && lw < 0; ++q)
                    if (tv[q]) lw = q;
                if (lw < 0) {
                    ++dimc;
                    continue;
                }
                std::copy(tv, tv + CW, col + (size_t)nl * CW);
                pidx[lw * 64 + __builtin_ctzll(tv[lw])] = nl;
                pmk[lw] |= tv[lw] & (0 - tv[lw]);
                ++nl;
            }
        }
        T.lnpv = dimc;
        T.lnpvD = 0;
        T.lnk = nker;
        return npv ? 3 : 0;
    }
    // postselected: a shot whose final detector row fails the mask is rejected here, before its status C
    if (PS && !zt->zero(row)) return -1;
    // status C (post_fast's rule in beta coordinates): C_beta = {x supported on beta_L : x in W^perp} = the
    // dependencies among the words KtL[j] = XOR_{l : beta_j has gamma coordinate l} Kt[l] (sigma_q's beta
    // coordinates), j >= rho_D (pivot-indexed elimination, the combinations in y coordinates: column j's is yA_'s row
    // j - rho_D, yW_ <= CW words); then the y-RREF (fast_cspace)
    if (!rhoL_ || !npv) return 0;
    uint64_t* lv = col;    // (col, pvv, pvc are free now)
    uint64_t* lc = pvv;
    uint64_t* kern = pvc;
    int nl = 0, nk = 0;
    const int yW = yW_;
    for (int q = 0; q < CW; ++q) pmk[q] = 0;
    for (int j = rD; j < R; ++j) {
        for (int q = 0; q < CW; ++q) tv[q] = 0;
        const uint64_t* gb = &GbW_[(size_t)j * CW];
        for (int u = 0; u < CW; ++u)
            for (uint64_t x = gb[u]; x; x &= x - 1) {
                const uint64_t* kr = Kt + (size_t)(u * 64 + __builtin_ctzll(x)) * CW;
                for (int q = 0; q < CW; ++q) tv[q] ^= kr[q];
            }
        const uint64_t* ya = &yA_[(size_t)(j - rD) * yW];
        for (int q = 0; q < CW; ++q) tc[q] = q < yW ? ya[q] : 0ull;
        reduce_low<NW>(tv, tc, CW, pmk, pidx, lv, lc);
        int lw = -1;
        for (int q = 0; q < CW && lw < 0; ++q)
            if (tv[q]) lw = q;
        if (lw >= 0) {
            std::copy(tv, tv + CW, lv + (size_t)nl * CW);
            std::copy(tc, tc + CW, lc + (size_t)nl * CW);
            pidx[lw * 64 + __builtin_ctzll(tv[lw])] = nl;
            pmk[lw] |= tv[lw] & (0 - tv[lw]);
            ++nl;
        } else {
            std::copy(tc, tc + CW, kern + (size_t)nk * CW);
            ++nk;
        }
    }
    if (!nk) return 0;
    // the fully reduced RREF in y coordinates (pivot = highest bit), descending pivot, yW_ words per row
    const int nr = rref_high<NW>(kern, nk, CW, pmk, pidx);
    rows_desc(pmk, CW, pidx, ord);
    T.ckey.resize((size_t)nr * yW);
    for (int i = 0; i < nr; ++i)
        std::copy(kern + (size_t)ord[i] * CW, kern + (size_t)ord[i] * CW + yW, T.ckey.begin() + (long)i * yW);
    return fast_cspace(T, nr);
}

template <bool PS, bool LAW>
int XdemSampler::post_mw(ThreadState& T, uint64_t* row, uint64_t* cpos, std::vector<uint64_t>* coins_out,
                         const XdemSpanTest* zt) {
    switch (CW_) {
        case 2: return post_multi<2, PS, LAW>(T, row, cpos, coins_out, zt);
        case 3: return post_multi<3, PS, LAW>(T, row, cpos, coins_out, zt);
        case 4: return post_multi<4, PS, LAW>(T, row, cpos, coins_out, zt);
        case 6: return post_multi<6, PS, LAW>(T, row, cpos, coins_out, zt);
        case 8: return post_multi<8, PS, LAW>(T, row, cpos, coins_out, zt);
        default: return post_multi<0, PS, LAW>(T, row, cpos, coins_out, zt);
    }
}

int XdemSampler::lookup(ThreadState& T, const uint64_t* key) {
    if (fast_) throw std::logic_error("xdem: lookup on the fast path");   // slot() allocates no table on the fast path
    uint64_t h = hash_words(key, WA_) & T.mask;
    for (int32_t idx; (idx = T.table[h]) >= 0; h = (h + 1) & T.mask) {
        if (std::memcmp(&T.keys[(size_t)idx * WA_], key, sizeof(uint64_t) * (size_t)WA_) == 0) {
            ++T.hits;
            return idx;
        }
    }
    ++T.misses;
    if (T.entries.size() >= cap_) {   // bounded: clear the slot's cache (never changes an output byte)
        std::fill(T.table.begin(), T.table.end(), -1);
        T.entries.clear();
        T.keys.clear();
        T.data.clear();
        ++T.clears;
        h = hash_words(key, WA_) & T.mask;
    }
    // the exact law (qeccore::xdem::law_into) at flip = 0 with eta_a = 1 on the odd set (ascending)
    Composed& cmp = T.lcmp;
    cmp.flip.w.assign(words_for((long long)D_ + O_), 0);
    cmp.eta.clear();
    for (int w = 0; w < WA_; ++w)
        for (uint64_t x = key[w]; x; x &= x - 1) cmp.eta.emplace_back(w * 64 + __builtin_ctzll(x), 1);
    AxisView lav;   // the file's axes with kappa in the lower-weight gauge basis (build_law_axes): the same law
    lav.base = F_.axes.data();
    lav.nbase = F_.axes.size();
    lav.kbase = law_k_.data();
    law_into(F_.D, F_.O, F_.r, lav, cmp, T.lsc);
    const LawScratch& L = T.lsc;
    const size_t lnw = words_for((long long)D_ + O_);
    auto to_int = [&](const uint64_t* b, uint64_t* out) {
        std::fill(out, out + WF_, 0ull);
        for (size_t wi = 0; wi < lnw; ++wi)
            for (uint64_t x = b[wi]; x; x &= x - 1) {
                const long long j = 64LL * (long long)wi + __builtin_ctzll(x);
                const long long k = j < D_ ? j : 64LL * WD_ + (j - D_);
                out[k >> 6] ^= 1ull << (k & 63);
            }
    };
    ThreadState::Entry E;
    E.data_off = T.data.size();
    E.rank = L.rank;
    T.data.resize(E.data_off + (size_t)WF_ * (1 + (size_t)L.rank));
    to_int(L.off.data(), &T.data[E.data_off]);
    for (size_t i = 0; i < (size_t)L.rank; ++i)
        to_int(&L.coins[i * lnw], &T.data[E.data_off + (size_t)WF_ * (1 + i)]);
    // status C = {e_L : e in span(coins), e_D = 0}: eliminate the detector part, keep the kernel's observable parts,
    // canonical RREF over the O bits. Detector-pivot rows: one flat arena (lR, WF words each).
    std::vector<std::vector<uint64_t>> kern;
    if (WO_) {
        std::vector<uint64_t>& R = T.lR;
        std::vector<int>& P = T.lP;
        R.clear();
        P.clear();
        T.lv.resize((size_t)WF_);
        uint64_t* v = T.lv.data();
        for (size_t i = 0; i < (size_t)L.rank; ++i) {
            const uint64_t* src = &T.data[E.data_off + (size_t)WF_ * (1 + i)];
            std::copy(src, src + WF_, v);
            for (size_t r = 0; r < P.size(); ++r)
                if ((v[(size_t)P[r] >> 6] >> (P[r] & 63)) & 1ull)
                    for (int w = 0; w < WF_; ++w) v[(size_t)w] ^= R[r * (size_t)WF_ + (size_t)w];
            int pd = -1;
            for (int w = 0; w < WD_ && pd < 0; ++w)
                if (v[(size_t)w]) pd = w * 64 + __builtin_ctzll(v[(size_t)w]);
            if (pd >= 0) {
                R.insert(R.end(), v, v + WF_);
                P.push_back(pd);
            } else {
                kern.emplace_back(v + WD_, v + WF_);
            }
        }
    }
    std::vector<int> kp;
    rref_desc(kern, WO_, kp);
    std::vector<uint64_t> flat;
    for (const auto& r : kern) flat.insert(flat.end(), r.begin(), r.end());
    {   // the general path's canonical form is the content itself
        const uint64_t h = hash_c(flat.data(), flat.size());
        const int id = local_cspace(T, flat.data(), flat.size(), h);
        E.tuid = id >= 0 ? id : local_insert(T, flat.data(), flat.size(), h, std::move(flat));
    }
    const int idx = (int)T.entries.size();
    T.entries.push_back(E);
    T.keys.insert(T.keys.end(), key, key + WA_);
    T.table[h] = idx;
    return idx;
}

int XdemSampler::apply_entry(ThreadState& T, uint64_t* row, int idx, uint64_t& cpos) const {
    // the general path's correction of a composed row: the cached law's offset and its coin combination (ceil(rank /
    // 64) words of the coin stream at cpos); returns the entry's local C id
    const ThreadState::Entry& E = T.entries[(size_t)idx];
    const uint64_t* d = &T.data[E.data_off];
    const int WF = WF_;
    for (int w = 0; w < WF; ++w) row[w] ^= d[w];
    if (E.rank) {
        ++T.coin_shots;
        for (int c0 = 0; c0 < E.rank; c0 += 64) {
            cpos += krng::kGamma;
            uint64_t bits = krng::mix(cpos);
            const int rem = E.rank - c0;
            if (rem < 64) bits &= (1ull << rem) - 1;
            for (; bits; bits &= bits - 1) {
                const uint64_t* cr = d + (size_t)WF * (size_t)(1 + c0 + __builtin_ctzll(bits));
                for (int w = 0; w < WF; ++w) row[w] ^= cr[w];
            }
        }
    }
    return E.tuid;
}

template <class Fire>
void XdemSampler::walk_chains(uint64_t seed, uint64_t block, Fire&& fire) const {
    uint64_t pos = krng::mix(krng::stream_key(seed, kTagNoise) ^ krng::mix(block));
    auto u = [&]() {   // (0, 1]
        pos += krng::kGamma;
        return (double)((krng::mix(pos) >> 11) + 1) * 0x1.0p-53;
    };
    for (const Chain& ch : chains_) {
        const uint64_t S = (uint64_t)ch.mem.size() * (uint64_t)kBlockShots;
        if (ch.always) {
            for (uint64_t t = 0; t < S; ++t) fire(ch, (size_t)(t >> 10), (unsigned)(t & 1023));
            continue;
        }
        double g = std::floor(std::log(u()) / ch.lp);
        if (!(g < (double)S)) continue;
        uint64_t t = (uint64_t)g;
        for (;;) {
            fire(ch, (size_t)(t >> 10), (unsigned)(t & 1023));
            g = std::floor(std::log(u()) / ch.lp);
            if (!(g < (double)(S - t - 1))) break;
            t += 1 + (uint64_t)g;
        }
    }
}

namespace {
// output row copy without a libc call, as fixed-size copies (each one load/store): for nb >= 2 a first piece at byte
// 0 and a last one ending at byte nb (2, 4 or 8 bytes, overlapping when nb is not a multiple), plus whole 8-byte
// words in between when nb > 16. Every store lies inside [d, d + nb).
inline void put_row(uint8_t* d, const uint8_t* src, size_t nb) {
    if (nb < 4) {
        if (nb >= 2) {
            std::memcpy(d, src, 2);
            std::memcpy(d + nb - 2, src + nb - 2, 2);
        } else if (nb) {
            d[0] = src[0];
        }
    } else if (nb < 8) {
        std::memcpy(d, src, 4);
        std::memcpy(d + nb - 4, src + nb - 4, 4);
    } else if (nb <= 16) {
        std::memcpy(d, src, 8);
        std::memcpy(d + nb - 8, src + nb - 8, 8);
    } else {
        for (size_t i = 0; i + 8 < nb; i += 8) std::memcpy(d + i, src + i, 8);
        std::memcpy(d + nb - 8, src + nb - 8, 8);
    }
}
}  // namespace

template <bool PS, bool KEEP>
void XdemSampler::run_one_block(ThreadState& T, uint64_t seed, uint64_t block, size_t n, uint8_t* dets,
                                uint8_t* obs, int32_t* cspace, std::vector<int>& first,
                                const std::vector<int>* fixed, const XdemSpanTest* ps, size_t& kept) {
    const size_t W = (size_t)W_;
    // n shot rows (sized on demand), plus one spare word: with W = 0 (no readout bit) a line's no-op entries (mask 0,
    // word 0) still write inside the buffer
    if (T.buf.size() < n * W + 1) T.buf.resize(n * W + 1);
    uint64_t* buf = T.buf.data();
    std::memset(buf, 0, sizeof(uint64_t) * n * W);
    const uint64_t* lo = loff_.data();
    const LineEnt* le = lent_.data();
    // XOR a line (its entries p[0, c)) into row d: the first kEntPad entries branch-free (an entry k >= c is applied
    // with mask 0; the arena's zero tail keeps these reads in bounds), the rest (long lines only) in a loop.
    auto fire_line = [](uint64_t* d, const LineEnt* p, uint64_t c) {
        for (int k = 0; k < kEntPad; ++k) d[p[k].word] ^= p[k].mask & (0ull - (uint64_t)((uint64_t)k < c));
        for (uint64_t k = kEntPad; k < c; ++k) d[p[k].word] ^= p[k].mask;
    };
    // the extended lines' (shot << 32 | record offset) pairs: written always, the cursor advanced by offset != 0
    // (branch-free), grown when full; a file with no extended line (a conventional DEM) skips the gather, its sort
    // and the KEEP filter
    const bool has_ext = xa_.size() > 1;
    if (T.epair.size() < (size_t)kBlockShots) T.epair.resize((size_t)kBlockShots);
    uint64_t* ew = T.epair.data();
    uint64_t* ee = ew + T.epair.size();
    auto grow_ext = [&]() {
        const size_t k = T.epair.size();
        T.epair.resize(2 * k);
        ew = T.epair.data() + k;
        ee = T.epair.data() + 2 * k;
    };
    // KEEP: every fired line's (shot << 32 | line) pair, written through a local cursor (grown on demand)
    using Cursor = std::conditional_t<KEEP, uint64_t*, char>;   // no cursor state in the KEEP = false loop
    [[maybe_unused]] Cursor fw{}, fe{};
    if constexpr (KEEP) {
        if (T.fpair.size() < 2 * kBlockShots) T.fpair.resize(2 * kBlockShots);
        fw = T.fpair.data();
        fe = fw + T.fpair.size();
    }
    if (fixed) {
        std::vector<uint32_t> fx;
        canonical_ids(*fixed, "XdemSampler", nullptr, fx);
        for (size_t s = 0; s < n; ++s) {
            for (int id : *fixed) fire_line(buf + s * W, le + lo[id], lo[id + 1] - lo[id]);
            for (uint32_t xo : fx) {
                if (ew == ee) grow_ext();
                *ew++ = ((uint64_t)s << 32) | xo;
            }
        }
    } else if constexpr (KEEP) {
        walk_chains(seed, block, [&](const Chain& ch, size_t m, unsigned s) {
            if (s >= n) return;
            const Member& mb = ch.mem[m];
            fire_line(buf + (size_t)s * W, le + mb.beg, mb.cnt);
            if constexpr (PS) {   // !PS: the extended lists are filtered from the sorted fired lists below
                if (!has_ext) return;
                if (ew == ee) grow_ext();
                *ew = ((uint64_t)s << 32) | mb.xo;
                ew += mb.xo != 0;
            }
            if (fw == fe) {
                const size_t k = T.fpair.size();
                T.fpair.resize(2 * k);
                fw = T.fpair.data() + k;
                fe = T.fpair.data() + 2 * k;
            }
            *fw++ = ((uint64_t)s << 32) | (uint32_t)ch.lines[m];
        });
    } else if (has_ext) {
        walk_chains(seed, block, [&](const Chain& ch, size_t m, unsigned s) {
            if (s >= n) return;
            const Member& mb = ch.mem[m];
            fire_line(buf + (size_t)s * W, le + mb.beg, mb.cnt);
            if (ew == ee) grow_ext();
            *ew = ((uint64_t)s << 32) | mb.xo;
            ew += mb.xo != 0;
        });
    } else {   // no extended line in the file (a conventional DEM): the flips alone
        walk_chains(seed, block, [&](const Chain& ch, size_t m, unsigned s) {
            if (s >= n) return;
            const Member& mb = ch.mem[m];
            fire_line(buf + (size_t)s * W, le + mb.beg, mb.cnt);
        });
    }
    // KEEP: the block's fired lines as a per-shot CSR: a counting sort by shot (inclusive prefix, then a reverse
    // scatter leaves fp[s] = shot s's first id, fp[n] = the block's total); a shot with >= 2 lines is then sorted.
    // !PS: the ids go to T.fids directly, and each shot's end, as an index into T.fids, to its row of the call's
    // offsets (T.foff; run_blocks shifts it to the call's ids); PS: to T.fsort, the survivors' lists are copied out.
    [[maybe_unused]] int32_t* kid = nullptr;
    if constexpr (KEEP) {
        T.fpos.assign(n + 1, 0);
        if (T.fmulti.size() < n + 1) T.fmulti.resize(n + 1);   // + 1: the branch-free append writes one past
        uint32_t* fp = T.fpos.data();
        uint32_t* fm = T.fmulti.data();
        const uint64_t* pr = T.fpair.data();
        const size_t np = (size_t)(fw - pr);
        size_t nm = 0;   // the shots with >= 2 lines (appended branch-free when a shot's count reaches 2)
        for (size_t i = 0; i < np; ++i) {
            const uint32_t sh = (uint32_t)(pr[i] >> 32);
            fm[nm] = sh;
            nm += ++fp[sh] == 2;
        }
        size_t base = 0;
        if constexpr (PS) {
            T.fsort.resize(np);
            kid = T.fsort.data();
        } else {
            base = T.fids.size();
            T.fids.resize(base + np);
            kid = T.fids.data() + base;
        }
        uint32_t acc = 0;
        if constexpr (PS) {
            for (size_t s = 0; s < n; ++s) fp[s] = acc += fp[s];
        } else {
            int64_t* o = T.foff;
            for (size_t s = 0; s < n; ++s) {
                fp[s] = acc += fp[s];
                o[s] = (int64_t)(base + acc);
            }
        }
        fp[n] = acc;
        for (size_t i = np; i-- > 0;) kid[--fp[pr[i] >> 32]] = (int32_t)(uint32_t)pr[i];
        if constexpr (!PS) {
            for (size_t k = 0; k < nm; ++k) sort_ids(kid + fp[fm[k]], fp[fm[k] + 1] - fp[fm[k]]);
            // the extended lists, in file order, filtered from the sorted fired lists: one record-offset lookup per
            // fired line (branch-free: the slot is written always and kept when the line is extended), its running
            // count cum[k] = extended lines before fired entry k; shot s's list is [cum[fp[s]], cum[fp[s + 1]])
            if (has_ext) {   // (a file with no extended line has none to filter)
                if (T.exo.size() < np + 1) T.exo.resize(np + 1);
                if (T.ecum.size() < np + 1) T.ecum.resize(np + 1);
                uint32_t* ex = T.exo.data();
                uint32_t* cum = T.ecum.data();
                const uint32_t* xof = xo_of_.data();
                uint32_t c = 0;
                for (size_t k = 0; k < np; ++k) {
                    const uint32_t x = xof[kid[k]];
                    cum[k] = c;
                    ex[c] = x;
                    c += x != 0;
                }
                cum[np] = c;
            }
        }
    }
    if constexpr (!KEEP || PS)   // (KEEP never runs a fixed set; a file with no extended line has nothing to sort)
        if (has_ext || fixed) sort_ext(T, (size_t)(ew - T.epair.data()), n);
    // shot s's extended list: [ext_begin(s), ext_begin(s + 1)) of eid
    const uint32_t* ep = T.epos.data();
    uint32_t* eid = T.exo.data();
    [[maybe_unused]] const uint32_t* kfp = T.fpos.data();
    [[maybe_unused]] const uint32_t* kcum = T.ecum.data();
    [[maybe_unused]] auto ext_begin = [&](size_t s) -> uint32_t {
        if (!has_ext && !fixed) return 0u;   // no extended line: nothing gathered, every shot's list is empty
        if constexpr (KEEP && !PS) return kcum[kfp[s]];
        else return ep[s];
    };
    uint64_t cpos = krng::mix(krng::stream_key(seed, kTagCoin) ^ krng::mix(block));
    // copied out of the members: the byte stores below may alias *this, so members would be reloaded every shot
    const size_t nbD = nbD_, nbO = nbO_, offO = (size_t)WD_ * 8;
    const bool fast = fast_, multi = CW_ != 0;
    if (++T.stamp_ctr == 0) {
        std::fill(T.stamp.begin(), T.stamp.end(), 0u);
        T.stamp_ctr = 1;
    }
    // local id 0 is pre-mapped at slot creation (remap[0] = 0), so it never needs to be listed in `first`
    auto note = [&](int tuid) {
        if (tuid && T.stamp[(size_t)tuid] != T.stamp_ctr) {
            T.stamp[(size_t)tuid] = T.stamp_ctr;
            first.push_back(tuid);
        }
    };
    if constexpr (!PS) {   // sample()
        (void)ps;
        (void)kept;
        for (size_t s = 0; s < n; ++s) {
            uint64_t* row = buf + s * W;
            int tuid = 0;
            const uint32_t e0 = ext_begin(s), ne = ext_begin(s + 1) - e0;
            if (ne) {
                ++T.ext_shots;
                if (ne > 1) sort_ids(eid + e0, ne);   // file order (the walk is chain-major)
                compose_ext(T, eid + e0, ne);
                if (fast) {
                    tuid = multi ? post_mw(T, row, &cpos, nullptr) : post_fast(T, row, &cpos, nullptr);
                } else {
                    const int idx = resolve(T, row);
                    if (idx >= 0) tuid = apply_entry(T, row, idx, cpos);
                }
            }
            const uint8_t* rb = reinterpret_cast<const uint8_t*>(row);
            put_row(dets + s * nbD, rb, nbD);
            put_row(obs + s * nbO, rb + offO, nbO);
            cspace[s] = tuid;
            note(tuid);
        }
        T.shots += n;
        return;
    }
    // the shot's correction (an extended line fired), then the mask on its final row: the local C id of a survivor,
    // -1 for a rejected shot (its status C is never computed; the fast path stops before the C block)
    auto correct = [&](uint64_t* row) -> int {
        if (fast)
            return multi ? post_mw<true>(T, row, &cpos, nullptr, ps) : post_fast<true>(T, row, &cpos, nullptr, ps);
        const int idx = resolve(T, row);
        const int tuid = idx >= 0 ? apply_entry(T, row, idx, cpos) : 0;
        return ps->zero(row) ? tuid : -1;
    };
    {
        // postselected: survivors only, to consecutive rows. A shot with no extended line is final after the line XOR.
        // A shot with one whose detector row fails the span test is rejected with no further work (no coin word read,
        // no composition); otherwise it is composed and corrected (reading its coin words) and its final row decides;
        // only a survivor gets its status C.
        uint8_t* ad = dets;
        uint8_t* ao = obs;
        int32_t* ac = cspace;
        if (T.ps_scratch.size() < ps->words()) T.ps_scratch.resize(ps->words());
        uint64_t* scr = T.ps_scratch.data();
        // the mask's words as locals (the byte stores below may alias the span test's members)
        const uint32_t* zw = ps->word_index();
        const uint64_t* zm = ps->word_mask();
        const size_t znw = ps->words();
        const uint32_t zw0 = znw ? zw[0] : 0;
        const uint64_t zm0 = znw ? zm[0] : 0;
        auto zero = [&](const uint64_t* row) {
            if (znw <= 1) return (row[zw0] & zm0) == 0;   // M within one word (or empty: row[0] & 0)
            uint64_t acc = 0;
            for (size_t k = 0; k < znw; ++k) acc |= row[zw[k]] & zm[k];
            return acc == 0;
        };
        size_t k = 0;
        [[maybe_unused]] const size_t fid0 = KEEP ? T.fids.size() : 0;
        for (size_t s = 0; s < n; ++s) {
            uint64_t* row = buf + s * W;
            int tuid = 0;
            const uint32_t e0 = ext_begin(s), ne = ext_begin(s + 1) - e0;
            if (!ne) {
                if (!zero(row)) continue;   // final after the line XOR; no coins
            } else {
                ++T.ext_shots;
                if (!ps->in_span(row, scr)) {   // rejected for any correction: no coin word, no law, no status
                    ++T.ps_early;
                    continue;
                }
                if (ne > 1) sort_ids(eid + e0, ne);   // file order (the walk is chain-major)
                compose_ext(T, eid + e0, ne);
                tuid = correct(row);   // reads its coin words: the decision needs them
                if (tuid < 0) continue;
            }
            const uint8_t* rb = reinterpret_cast<const uint8_t*>(row);
            put_row(ad + k * nbD, rb, nbD);
            put_row(ao + k * nbO, rb + offO, nbO);
            ac[k] = tuid;
            ++k;
            if constexpr (KEEP) {   // block-relative ends over the survivors
                const uint32_t* fp = T.fpos.data();
                if (fp[s + 1] - fp[s] > 1) sort_ids(kid + fp[s], fp[s + 1] - fp[s]);
                T.fids.insert(T.fids.end(), kid + fp[s], kid + fp[s + 1]);
                T.fcnt.push_back((uint32_t)(T.fids.size() - fid0));
            }
            note(tuid);   // survivors only: the C table numbers the survivors' C in first appearance
        }
        kept = k;
        T.shots += n;
        return;
    }
}

void XdemSampler::run_blocks(size_t shots, uint64_t seed, int threads, uint64_t first_block, uint8_t* dets,
                             uint8_t* obs, int32_t* cspace, const std::vector<int>* fixed, const XdemSpanTest* ps,
                             XdemPsCounts* pc, XdemFired* fired) {
    if (threads < 1) throw std::invalid_argument("XdemSampler.sample: threads must be >= 1");
    if (fired && fixed) throw std::logic_error("XdemSampler: fired lines are kept for the production walk only");
    if (fired) {
        fired->off.assign(1, 0);
        fired->ids.clear();
    }
    if (fixed)
        for (int id : *fixed)
            if (id < 0 || (size_t)id >= L_)
                throw std::out_of_range("XdemSampler: line id " + std::to_string(id) + " out of range");
    const size_t nb = (shots + kBlockShots - 1) / kBlockShots;
    if (nb == 0) return;
    if (first_block > UINT64_MAX - nb) throw std::invalid_argument("XdemSampler.sample: first_block out of range");
    const int nt = (int)std::min<size_t>((size_t)threads, nb);
    for (int t = 0; t < nt; ++t) {
        ThreadState& T = slot(t);
        T.ps_early = 0;
        // the expected number of fired extended lines per block (sum of their p) with headroom: no regrowth on a
        // typical block
        const size_t want = (size_t)((double)kBlockShots * mean_ext_ * 1.25) + kBlockShots;
        if (T.epair.size() < want) T.epair.resize(want);
        if (fired) {
            T.fcnt.clear();
            T.fids.clear();
            // the expected number of fired lines (sum of p) with 5 % headroom: no regrowth on a typical call
            T.fids.reserve((size_t)((double)std::min(shots, (nb + (size_t)nt - 1) / (size_t)nt * kBlockShots) *
                                    mean_fired_ * 1.05) + kBlockShots);
        }
    }
    if (fired && !ps) {   // rows are fixed: every block writes its shots' ends in place (index into its slot's ids)
        fired->off.resize(shots + 1);
        fired->off[0] = 0;
    }
    // fired: block b's counts / ids start at these positions of its slot's fcnt / fids (and its ids end at fid1[b])
    std::vector<size_t> fc0(fired ? nb : 0), fid0(fired ? nb : 0), fid1(fired ? nb : 0);
    // postselected: block b writes its survivors from row dst[b], the running total when one thread takes the blocks
    // in order, else its own first row (compacted below)
    std::vector<size_t> kept(ps ? nb : 0, 0), dst(ps ? nb : 0, 0);
    size_t cursor = 0;
    std::vector<std::vector<int>> first(nb);
    std::vector<int> bt(nb, 0);
    std::atomic<size_t> next{0};
    // fired: the merge runs inside the workers, after a barrier every worker reaches (also after an error): the last
    // to arrive takes the per-block row / id prefixes (rb, ib) and sizes the output; then each worker moves only the
    // blocks it produced (bt[b] == t). One slot without postselection moves nothing (its ids are the output).
    std::vector<size_t> rb(fired ? nb + 1 : 0, 0), ib(fired ? nb + 1 : 0, 0);
    std::mutex bar_mu;
    std::condition_variable bar_cv;
    int bar_in = 0;
    bool bar_open = false, merge = false;
    std::exception_ptr merge_err;
    auto merge_prefix = [&] {   // the last worker to arrive, before the barrier opens
        for (size_t b = 0; b < nb; ++b) {
            rb[b + 1] = rb[b] + (ps ? kept[b] : std::min<size_t>(kBlockShots, shots - b * kBlockShots));
            ib[b + 1] = ib[b] + (fid1[b] - fid0[b]);
        }
        if (!ps && nt == 1) {
            fired->ids.swap(ts_[0]->fids);
            return false;
        }
        if (ps) {
            fired->off.resize(rb[nb] + 1);
            fired->off[0] = 0;
        }
        fired->ids.resize(ib[nb]);
        return true;
    };
    auto merge_own = [&](int t) {   // the blocks worker t produced
        const ThreadState& T = *ts_[(size_t)t];
        int64_t* off = fired->off.data();
        int32_t* ids = fired->ids.data();
        for (size_t b = 0; b < nb; ++b) {
            if (bt[b] != t) continue;
            int64_t* o = off + rb[b] + 1;
            const size_t m = rb[b + 1] - rb[b];
            if (ps) {
                const uint32_t* c = T.fcnt.data() + fc0[b];
                const int64_t base = (int64_t)ib[b];
                for (size_t i = 0; i < m; ++i) o[i] = base + (int64_t)c[i];
            } else {
                const int64_t delta = (int64_t)ib[b] - (int64_t)fid0[b];
                if (delta)
                    for (size_t i = 0; i < m; ++i) o[i] += delta;
            }
            if (ib[b + 1] > ib[b])
                std::memcpy(ids + ib[b], T.fids.data() + fid0[b], (ib[b + 1] - ib[b]) * sizeof(int32_t));
        }
    };
    // An error is raised for the LOWEST failing block, as at threads = 1: a failing worker records its block and
    // stops further claims; claimed blocks run to completion.
    std::vector<std::exception_ptr> errs((size_t)nt);
    std::vector<size_t> err_block((size_t)nt, SIZE_MAX);
    auto worker = [&](int t) {
        size_t b = SIZE_MAX;
        try {
            ThreadState& T = *ts_[(size_t)t];
            for (;;) {
                b = next.fetch_add(1, std::memory_order_relaxed);
                if (b >= nb) break;
                const size_t s0 = b * kBlockShots;
                const size_t n = std::min<size_t>(kBlockShots, shots - s0);
                bt[b] = t;
                if (fired) {
                    fc0[b] = T.fcnt.size();
                    fid0[b] = T.fids.size();
                }
                if (ps) {
                    const size_t r0 = nt == 1 ? cursor : s0;
                    dst[b] = r0;
                    if (fired)
                        run_one_block<true, true>(T, seed, first_block + b, n, dets + r0 * nbD_, obs + r0 * nbO_,
                                                  cspace + r0, first[b], fixed, ps, kept[b]);
                    else
                        run_one_block<true>(T, seed, first_block + b, n, dets + r0 * nbD_, obs + r0 * nbO_,
                                            cspace + r0, first[b], fixed, ps, kept[b]);
                    if (nt == 1) cursor += kept[b];   // read only at one thread (no shared write between workers)
                } else {
                    size_t unused = 0;
                    if (fired) T.foff = fired->off.data() + s0 + 1;
                    if (fired)
                        run_one_block<false, true>(T, seed, first_block + b, n, dets + s0 * nbD_, obs + s0 * nbO_,
                                                   cspace + s0, first[b], fixed, nullptr, unused);
                    else
                        run_one_block<false>(T, seed, first_block + b, n, dets + s0 * nbD_, obs + s0 * nbO_,
                                             cspace + s0, first[b], fixed, nullptr, unused);
                }
                if (fired) fid1[b] = T.fids.size();
            }
        } catch (...) {
            errs[(size_t)t] = std::current_exception();
            err_block[(size_t)t] = b;
            next.store(nb);
        }
        if (!fired) return;
        // the barrier (reached on every path above: the loop's errors are caught, the merge never throws past here)
        bool last;
        {
            std::unique_lock<std::mutex> lk(bar_mu);
            last = ++bar_in == nt;
            if (!last) bar_cv.wait(lk, [&] { return bar_open; });
        }
        if (last) {
            bool ok = true;
            for (int u = 0; u < nt; ++u) ok = ok && !errs[(size_t)u];
            bool m = false;
            if (ok) {
                try {
                    m = merge_prefix();
                } catch (...) {
                    merge_err = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> lk(bar_mu);
                merge = m;
                bar_open = true;
            }
            bar_cv.notify_all();
        }
        if (merge) merge_own(t);   // read after the barrier opened (under its mutex): no race
    };
    if (nt == 1) {
        worker(0);
    } else {
        std::vector<std::thread> th;
        th.reserve((size_t)nt);
        for (int t = 0; t < nt; ++t) th.emplace_back(worker, t);
        for (auto& x : th) x.join();
    }
    int fail = -1;
    for (int t = 0; t < nt; ++t)
        if (errs[(size_t)t] && (fail < 0 || err_block[(size_t)t] < err_block[(size_t)fail])) fail = t;
    if (fired)
        for (int t = 0; t < nt; ++t) ts_[(size_t)t]->foff = nullptr;
    if (fail >= 0) std::rethrow_exception(errs[(size_t)fail]);
    // deterministic C table: merge in block order (new contents get the next global id)
    for (size_t b = 0; b < nb; ++b) {
        ThreadState& T = *ts_[(size_t)bt[b]];
        for (int tu : first[b])
            if (T.remap[(size_t)tu] < 0)
                T.remap[(size_t)tu] = global_cspace(T.ctab[(size_t)tu], T.chash[(size_t)tu]);
    }
    if (ps) {   // survivors: compact in block order (dst[b] >= the running total, so memmove forward is safe), remap
        size_t at = 0;
        for (size_t b = 0; b < nb; ++b) {
            const size_t k = kept[b];
            if (k && dst[b] != at) {
                if (nbD_) std::memmove(dets + at * nbD_, dets + dst[b] * nbD_, k * nbD_);
                if (nbO_) std::memmove(obs + at * nbO_, obs + dst[b] * nbO_, k * nbO_);
                std::memmove(cspace + at, cspace + dst[b], k * sizeof(int32_t));
            }
            const ThreadState& T = *ts_[(size_t)bt[b]];
            for (size_t s = at; s < at + k; ++s) cspace[s] = T.remap[(size_t)cspace[s]];
            at += k;
        }
        pc->passed = at;
        for (int t = 0; t < nt; ++t) pc->early += ts_[(size_t)t]->ps_early;
    } else {
        for (size_t b = 0; b < nb; ++b) {
            const ThreadState& T = *ts_[(size_t)bt[b]];
            const size_t s0 = b * kBlockShots, s1 = std::min<size_t>(shots, s0 + kBlockShots);
            for (size_t s = s0; s < s1; ++s) cspace[s] = T.remap[(size_t)cspace[s]];
        }
    }
    if (merge_err) std::rethrow_exception(merge_err);
    shots_total_ += shots;
}

void XdemSampler::sample(size_t shots, uint64_t seed, int threads, uint8_t* dets, uint8_t* obs, int32_t* cspace,
                         uint64_t first_block, XdemFired* fired) {
    std::lock_guard<std::mutex> lk(mu_);
    run_blocks(shots, seed, threads, first_block, dets, obs, cspace, nullptr, nullptr, nullptr, fired);
}

void XdemSampler::sample_ps(size_t shots, uint64_t seed, int threads, const std::vector<uint8_t>& flags,
                            uint8_t* dets, uint8_t* obs, int32_t* cspace, XdemPsCounts& counts, uint64_t first_block,
                            XdemFired* fired) {
    if (flags.size() != (size_t)D_)
        throw std::invalid_argument("XdemSampler.sample_ps: the mask needs " + std::to_string(D_) +
                                    " entries (got " + std::to_string(flags.size()) + ")");
    std::lock_guard<std::mutex> lk(mu_);
    bool same = ps_built_;
    for (size_t d = 0; same && d < flags.size(); ++d) same = (flags[d] != 0) == (ps_flags_[d] != 0);
    if (!same) {   // once per mask: the echelon basis of span(img_a) restricted to the mask (detector words of img_)
        ps_built_ = false;
        ps_flags_.assign(flags.size(), 0);
        for (size_t d = 0; d < flags.size(); ++d) ps_flags_[d] = flags[d] != 0;
        ps_ = XdemSpanTest(D_, ps_flags_.data(), img_.data(), (size_t)A_, (size_t)WF_);
        ps_built_ = true;
    }
    counts = XdemPsCounts();
    counts.attempts = shots;
    run_blocks(shots, seed, threads, first_block, dets, obs, cspace, nullptr, &ps_, &counts, fired);
}

void XdemSampler::sample_fixed(const std::vector<int>& line_ids, size_t shots, uint64_t seed, uint8_t* dets,
                               uint8_t* obs, int32_t* cspace) {
    std::lock_guard<std::mutex> lk(mu_);
    run_blocks(shots, seed, 1, 0, dets, obs, cspace, &line_ids);
}

void XdemSampler::fire_counts(size_t shots, uint64_t seed, uint64_t first_block, std::vector<uint64_t>& counts,
                              const std::vector<std::pair<int, int>>& pairs, std::vector<uint64_t>& pair_counts) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t L = L_;
    counts.assign(L, 0);
    pair_counts.assign(pairs.size(), 0);
    std::vector<int> pidx(L, -1);
    std::vector<int> plines;
    for (const auto& [i, j] : pairs)
        for (int x : {i, j}) {
            if (x < 0 || (size_t)x >= L) throw std::out_of_range("fire_counts: line id out of range");
            if (pidx[(size_t)x] < 0) {
                pidx[(size_t)x] = (int)plines.size();
                plines.push_back(x);
            }
        }
    std::vector<uint8_t> mark(plines.size() * (size_t)kBlockShots);
    const size_t nb = (shots + kBlockShots - 1) / kBlockShots;
    for (size_t b = 0; b < nb; ++b) {
        const size_t n = std::min<size_t>(kBlockShots, shots - b * kBlockShots);
        std::fill(mark.begin(), mark.end(), 0);
        walk_chains(seed, first_block + b, [&](const Chain& ch, size_t m, unsigned s) {
            if (s >= n) return;
            const int line = ch.lines[m];
            ++counts[(size_t)line];
            const int k = pidx[(size_t)line];
            if (k >= 0) mark[(size_t)k * kBlockShots + s] = 1;
        });
        for (size_t q = 0; q < pairs.size(); ++q) {
            const uint8_t* a = &mark[(size_t)pidx[(size_t)pairs[q].first] * kBlockShots];
            const uint8_t* c = &mark[(size_t)pidx[(size_t)pairs[q].second] * kBlockShots];
            for (size_t s = 0; s < n; ++s) pair_counts[q] += a[s] & c[s];
        }
    }
}

namespace {
// internal row layout (detector d = bit d, observable o = bit 64 WD + o) -> File layout (observable o = bit D + o)
Bits to_file_bits(const uint64_t* in, int D, int O, int WD, int WF) {
    Bits b;
    b.w.assign(words_for((long long)D + O), 0);
    for (int wi = 0; wi < WF; ++wi)
        for (uint64_t x = in[wi]; x; x &= x - 1) {
            const long long k = 64LL * wi + __builtin_ctzll(x);
            const long long j = k < 64LL * WD ? k : D + (k - 64LL * WD);
            b.w[(size_t)(j >> 6)] ^= 1ull << (j & 63);
        }
    return b;
}
}  // namespace

int XdemSampler::status_of(const std::vector<int>& line_ids, Bits& offset, std::vector<Bits>& coins) {
    std::lock_guard<std::mutex> lk(mu_);
    ThreadState& T = slot(0);
    std::vector<uint64_t> row((size_t)W_ + 1, 0);
    std::vector<uint32_t> ext;
    canonical_ids(line_ids, "status_of", row.data(), ext);
    auto to_file = [&](const uint64_t* in) { return to_file_bits(in, D_, O_, WD_, WF_); };
    coins.clear();
    if (ext.empty()) {
        offset = to_file(row.data());
        return 0;
    }
    compose_ext(T, ext.data(), (uint32_t)ext.size());
    if (fast_ && CW_) {   // multi-word: the canonical offset and coins (gamma coordinates, CW_ words each)
        std::vector<uint64_t> cc;
        const int tu = post_mw(T, row.data(), nullptr, &cc);
        offset = to_file(row.data());
        std::vector<uint64_t> z((size_t)WF_ + 1, 0);
        for (size_t i = 0; i < cc.size(); i += (size_t)CW_) {
            std::fill(z.begin(), z.end(), 0ull);
            xor_gamma(z.data(), &cc[i]);
            coins.push_back(to_file(z.data()));
        }
        if (T.remap[(size_t)tu] < 0) T.remap[(size_t)tu] = global_cspace(T.ctab[(size_t)tu], T.chash[(size_t)tu]);
        return T.remap[(size_t)tu];
    }
    if (fast_) {
        std::vector<uint64_t> cc;
        const int tu = post_fast(T, row.data(), nullptr, &cc);
        offset = to_file(row.data());
        std::vector<uint64_t> z((size_t)WF_ + 1, 0);
        for (uint64_t c : cc) {
            if (!c) continue;
            std::fill(z.begin(), z.end(), 0ull);
            xor_beta(z.data(), c);
            coins.push_back(to_file(z.data()));
        }
        if (T.remap[(size_t)tu] < 0) T.remap[(size_t)tu] = global_cspace(T.ctab[(size_t)tu], T.chash[(size_t)tu]);
        return T.remap[(size_t)tu];
    }
    const int idx = resolve(T, row.data());
    if (idx < 0) {
        offset = to_file(row.data());
        return 0;
    }
    const ThreadState::Entry& E = T.entries[(size_t)idx];
    const uint64_t* d = &T.data[E.data_off];
    for (int w = 0; w < WF_; ++w) row[(size_t)w] ^= d[w];
    offset = to_file(row.data());
    for (int i = 0; i < E.rank; ++i) coins.push_back(to_file(d + (size_t)WF_ * (size_t)(1 + i)));
    if (T.remap[(size_t)E.tuid] < 0)
        T.remap[(size_t)E.tuid] = global_cspace(T.ctab[(size_t)E.tuid], T.chash[(size_t)E.tuid]);
    return T.remap[(size_t)E.tuid];
}

int XdemSampler::law_row(ThreadState& T, uint64_t* row, int* entry_out) {
    // the sampler's own law code with no coin draw: post_fast's LAW instantiation (offset + kernel basis), or resolve +
    // the cached entry's offset (its coin rows stay in the entry); multi-word coordinates: post_multi's LAW instance
    if (fast_) {
        const int kind =
            CW_ ? post_mw<false, true>(T, row, nullptr, nullptr) : post_fast<false, true>(T, row, nullptr, nullptr);
        if (kind == 0) T.lnpv = T.lnpvD = 0;   // m = 0 returns before the law scratch (no coins: a point law)
        return kind;
    }
    const int idx = resolve(T, row);
    if (idx < 0) return 0;   // no odd axis: the even images are in the row, a point law
    const uint64_t* d = &T.data[T.entries[(size_t)idx].data_off];
    for (int w = 0; w < WF_; ++w) row[w] ^= d[w];
    *entry_out = idx;
    return 2;
}

int XdemSampler::law_dim(ThreadState& T, int kind, int entry) {
    // dim C, C = {e_O : e in span(coins), e_D = 0}
    if (kind == 0) return 0;
    if (kind == 1) return rhoL_ ? T.lnpv - T.lnpvD : 0;   // post_fast's status rule (no beta_L: C = {0})
    if (kind == 3) return T.lnpv;                         // post_multi<LAW>'s
    // general path: the entry's rank coin rows (WF_ words): detector-pivot echelon, then the zero-detector rest's
    // observable parts (C, pivot = highest bit); each row reduced by the earlier ones
    const ThreadState::Entry& E = T.entries[(size_t)entry];
    const uint64_t* base = &T.data[E.data_off + (size_t)WF_];
    T.ldr.clear();
    T.ldp.clear();
    T.lcr.clear();
    T.lcp.clear();
    T.lq.resize((size_t)WF_ + 1);
    uint64_t* v = T.lq.data();
    for (int i = 0; i < E.rank; ++i) {
        std::copy(base + (size_t)i * WF_, base + (size_t)(i + 1) * WF_, v);
        for (size_t r = 0; r < T.ldp.size(); ++r)
            if ((v[(size_t)T.ldp[r] >> 6] >> (T.ldp[r] & 63)) & 1ull)
                for (int w = 0; w < WF_; ++w) v[w] ^= T.ldr[r * (size_t)WF_ + (size_t)w];
        int pd = -1;
        for (int w = 0; w < WD_ && pd < 0; ++w)
            if (v[w]) pd = w * 64 + __builtin_ctzll(v[w]);
        if (pd >= 0) {
            T.ldr.insert(T.ldr.end(), v, v + WF_);
            T.ldp.push_back(pd);
            continue;
        }
        uint64_t* vo = v + WD_;
        for (size_t r = 0; r < T.lcp.size(); ++r)
            if ((vo[(size_t)T.lcp[r] >> 6] >> (T.lcp[r] & 63)) & 1ull)
                for (int w = 0; w < WO_; ++w) vo[w] ^= T.lcr[r * (size_t)WO_ + (size_t)w];
        const int ph = highest(vo, WO_);
        if (ph < 0) continue;
        T.lcr.insert(T.lcr.end(), vo, vo + WO_);
        T.lcp.push_back(ph);
    }
    return (int)T.lcp.size();
}

int XdemSampler::law_coins(ThreadState& T, uint64_t* row, std::vector<uint64_t>& rows, int* entry_out) {
    // the law of the composite in T's eta planes (no coin draw, the sampler's own law code): row <- its offset (the
    // flips on entry), rows <- a basis of its coin span (WF_ words each, internal layout); returns the law kind
    // (law_row's; 3: multi-word), *entry_out its general-path entry (kind 2)
    rows.clear();
    std::vector<uint64_t> z((size_t)WF_ + 1, 0);
    if (fast_ && CW_) {   // multi-word: the canonical coins (gamma coordinates)
        std::vector<uint64_t> cc;
        const int kind = post_mw<false, true>(T, row, nullptr, &cc);
        if (kind == 0) T.lnpv = T.lnpvD = 0;
        for (size_t i = 0; i < cc.size(); i += (size_t)CW_) {
            std::fill(z.begin(), z.end(), 0ull);
            xor_gamma(z.data(), &cc[i]);
            rows.insert(rows.end(), z.begin(), z.begin() + WF_);
        }
        return kind;
    }
    const int kind = law_row(T, row, entry_out);
    if (kind == 1) {
        // the coin span in beta coordinates = {x : XOR_{j : x_j = 1} Kt[j] = 0}: a basis from the dependencies among
        // the Kt words (elimination with combinations), mapped through beta
        uint64_t pv[64], pc[64], pl[64];
        int np = 0;
        for (int j = 0; j < rho_; ++j) {
            uint64_t v = T.lkt[j], c = 1ull << j;
            for (int q = 0; q < np; ++q)
                if (v & pl[q]) {
                    v ^= pv[q];
                    c ^= pc[q];
                }
            if (v) {
                pv[np] = v;
                pc[np] = c;
                pl[np] = v & (0 - v);
                ++np;
                continue;
            }
            std::fill(z.begin(), z.end(), 0ull);
            xor_beta(z.data(), c);
            rows.insert(rows.end(), z.begin(), z.begin() + WF_);
        }
    } else if (kind == 2) {
        const ThreadState::Entry& E = T.entries[(size_t)*entry_out];
        const uint64_t* c0 = T.data.data() + E.data_off + (size_t)WF_;
        rows.insert(rows.end(), c0, c0 + (size_t)WF_ * (size_t)E.rank);
    }
    return kind;
}

int XdemSampler::law_of(const std::vector<int>& line_ids, Bits& offset, std::vector<Bits>& coins) {
    std::lock_guard<std::mutex> lk(mu_);
    ThreadState& T = slot(0);
    std::vector<uint64_t> row((size_t)W_ + 1, 0);
    std::vector<uint32_t> ext;
    canonical_ids(line_ids, "law_of", row.data(), ext);
    int entry = -1, kind = 0;
    std::vector<uint64_t> rows;
    if (!ext.empty()) {
        compose_ext(T, ext.data(), (uint32_t)ext.size());
        kind = law_coins(T, row.data(), rows, &entry);
    }
    offset = to_file_bits(row.data(), D_, O_, WD_, WF_);
    coins.clear();
    for (size_t i = 0; i < rows.size(); i += (size_t)WF_) coins.push_back(to_file_bits(&rows[i], D_, O_, WD_, WF_));
    return law_dim(T, kind, entry);
}

uint64_t XdemSampler::span_check(size_t shots, uint64_t seed, uint64_t* ext_shots) {
    std::lock_guard<std::mutex> lk(mu_);
    ThreadState& T = slot(0);
    const size_t W = (size_t)W_;
    std::vector<uint8_t> all((size_t)D_, 1);
    const XdemSpanTest span(D_, all.data(), img_.data(), (size_t)A_, (size_t)WF_);
    std::vector<uint64_t> buf((size_t)kBlockShots * W + 1), r1(W + 1), dd((size_t)WD_ + 1), scr(span.words() + 1);
    uint64_t bad = 0, xs = 0;
    const size_t nb = (shots + kBlockShots - 1) / kBlockShots;
    for (size_t b = 0; b < nb; ++b) {
        const size_t n = std::min<size_t>(kBlockShots, shots - b * kBlockShots);
        std::fill(buf.begin(), buf.end(), 0ull);
        size_t np = 0;
        walk_chains(seed, b, [&](const Chain& ch, size_t m, unsigned s) {
            if (s >= n) return;
            const Member& mb = ch.mem[m];
            for (uint64_t t = mb.beg; t < mb.beg + mb.cnt; ++t) buf[s * W + lent_[t].word] ^= lent_[t].mask;
            if (mb.xo) {
                if (np == T.epair.size()) T.epair.resize(std::max<size_t>(2 * np, kBlockShots));
                T.epair[np++] = ((uint64_t)s << 32) | mb.xo;
            }
        });
        sort_ext(T, np, n);
        uint64_t c1 = krng::mix(krng::stream_key(seed, kTagCoin) ^ krng::mix(b));
        for (size_t s = 0; s < n; ++s) {
            const uint64_t* row = &buf[s * W];
            const uint32_t e0 = T.epos[s], ne = T.epos[s + 1] - e0;
            if (!ne) continue;
            ++xs;
            std::copy(row, row + W, r1.begin());
            if (ne > 1) sort_ids(T.exo.data() + e0, ne);
            compose_ext(T, T.exo.data() + e0, ne);
            if (fast_) {
                if (CW_) post_mw(T, r1.data(), &c1, nullptr);
                else post_fast(T, r1.data(), &c1, nullptr);
            } else {
                const int idx = resolve(T, r1.data());
                if (idx >= 0) apply_entry(T, r1.data(), idx, c1);   // the production correction (run_one_block's)
            }
            for (int w = 0; w < WD_; ++w) dd[(size_t)w] = r1[(size_t)w] ^ row[w];
            bad += !span.in_span(dd.data(), scr.data());
        }
    }
    if (ext_shots) *ext_shots = xs;
    return bad;
}

void XdemSampler::score(size_t shots, const uint8_t* obs, const int32_t* cspace, const uint8_t* flips,
                        double* out) const {
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t last = (O_ & 63) ? (1ull << (O_ & 63)) - 1 : ~0ull;
    if (WO_ <= 1) {   // O <= 64: one word per shot; same checks in the same per-shot order as the general path
        const size_t nbO = nbO_;
        const int O = O_;
        for (size_t s = 0; s < shots; ++s) {
            const int id = cspace[s];
            if (id < 0 || (size_t)id >= cspaces_.size()) refuse_score_id(id, s);
            uint64_t x = 0;
            const uint8_t* ob = obs + s * nbO;
            for (size_t b = 0; b < nbO; ++b) x |= (uint64_t)ob[b] << (8 * b);
            x &= last;
            const uint8_t* f = flips + s * (size_t)O;
            unsigned bad = 0;
            for (int o = 0; o < O; ++o) {
                bad |= f[o];
                x ^= (uint64_t)(f[o] & 1u) << o;
            }
            if (bad > 1) refuse_score_flip();
            const auto& P = cpiv_[(size_t)id];
            if (P.empty()) {
                out[s] = x ? 0.0 : 1.0;
                continue;
            }
            const auto& C = cspaces_[(size_t)id];
            for (size_t i = 0; i < P.size(); ++i)
                if ((x >> P[i]) & 1ull) x ^= C[i];
            out[s] = x ? 0.0 : std::ldexp(1.0, -(int)P.size());
        }
        return;
    }
    std::vector<uint64_t> w((size_t)std::max(WO_, 1), 0);
    for (size_t s = 0; s < shots; ++s) {
        const int id = cspace[s];
        if (id < 0 || (size_t)id >= cspaces_.size()) refuse_score_id(id, s);
        std::fill(w.begin(), w.end(), 0ull);
        if (nbO_) std::memcpy(w.data(), obs + s * nbO_, nbO_);
        if (WO_) w[(size_t)WO_ - 1] &= last;
        const uint8_t* f = flips + s * (size_t)O_;
        for (int o = 0; o < O_; ++o) {
            if (f[o] > 1) refuse_score_flip();
            if (f[o]) w[(size_t)o >> 6] ^= 1ull << (o & 63);
        }
        const auto& C = cspaces_[(size_t)id];
        const auto& P = cpiv_[(size_t)id];
        for (size_t i = 0; i < P.size(); ++i)
            if ((w[(size_t)P[i] >> 6] >> (P[i] & 63)) & 1ull)
                for (int k = 0; k < WO_; ++k) w[(size_t)k] ^= C[i * (size_t)WO_ + k];
        uint64_t nz = 0;
        for (int k = 0; k < WO_; ++k) nz |= w[(size_t)k];
        out[s] = nz ? 0.0 : std::ldexp(1.0, -(int)P.size());
    }
}

// ---- score_lines (Task 7; SPEC §4.1, §4.4, §4.5). The law readers are the v1 sampler's, on the same law scratch.

uint64_t XdemSampler::beta_d_coords(ThreadState& T, const uint64_t* off, const uint64_t* d, bool& ok) const {
    // the beta_D coordinates of d ^ off_D (beta_D's detector parts are independent; beta_L has none, so every vector of
    // span(beta) has its beta_D coordinates fixed by its detector part); ok = d ^ off_D in span(beta_D)_D
    if (T.ld.size() < (size_t)WD_ + 1) T.ld.resize((size_t)WD_ + 1);
    uint64_t* dd = T.ld.data();
    for (int w = 0; w < WD_; ++w) dd[w] = off[w] ^ d[w];
    uint64_t sD = 0;
    const int rD = rho_ - rhoL_;
    for (int j = 0; j < rD; ++j) {
        const int p = pD_[(size_t)j];
        if ((dd[p >> 6] >> (p & 63)) & 1ull) {
            const uint64_t* b = bDd_.data() + (size_t)j * WD_;
            for (int w = 0; w < WD_; ++w) dd[w] ^= b[w];
            sD |= 1ull << j;
        }
    }
    uint64_t rest = 0;
    for (int w = 0; w < WD_; ++w) rest |= dd[w];
    ok = rest == 0;
    return sD;
}

bool XdemSampler::beta_d_coords_w(ThreadState& T, const uint64_t* off, const uint64_t* d, uint64_t* sD) const {
    // beta_d_coords in CW_ words (kind 3): sD <- the beta_D coordinates of d ^ off_D; true iff it lies in span(beta_D)_D
    const int CW = CW_, rD = rho_ - rhoL_;
    uint64_t* dd = T.lrw.data() + ((size_t)2 * (size_t)std::max(rho_, 1) + 6) * (size_t)CW;
    for (int w = 0; w < WD_; ++w) dd[w] = off[w] ^ d[w];
    std::fill(sD, sD + CW, 0ull);
    for (int j = 0; j < rD; ++j) {
        const int p = pD_[(size_t)j];
        if ((dd[p >> 6] >> (p & 63)) & 1ull) {
            const uint64_t* b = bDd_.data() + (size_t)j * WD_;
            for (int w = 0; w < WD_; ++w) dd[w] ^= b[w];
            sD[j >> 6] |= 1ull << (j & 63);
        }
    }
    uint64_t rest = 0;
    for (int w = 0; w < WD_; ++w) rest |= dd[w];
    return rest == 0;
}

bool XdemSampler::law_given(ThreadState& T, int kind, const uint64_t* off, const uint64_t* d, uint64_t* o) {
    // o <- an observable value of {o : (d, o) in off + span(coins)}; false when d is outside the detector support
    if (kind == 3) {
        // kind 1's rule in CW_ words: x_D = sD from d; solve XOR_{j in L, x_j} lktW[j] = r0 = XOR_{j in sD} lktW[j]
        // over the beta_L columns (private-pivot elimination with combinations); o = off_O + (beta x)_O
        const int CW = CW_, R = rho_, rD = rho_ - rhoL_;
        uint64_t* sD = T.lrw.data();
        uint64_t* r0 = sD + CW;
        uint64_t* sL = r0 + CW;
        uint64_t* pm = sL + CW;   // the echelon's pivot bits (each vector's lowest), pidx[p] = the vector of pivot p
        uint64_t* tv = pm + CW;
        uint64_t* tc = tv + CW;
        uint64_t* lv = tc + CW;
        uint64_t* lc = lv + (size_t)std::max(R, 1) * CW;
        int* pidx = T.lrp.data();
        if (!beta_d_coords_w(T, off, d, sD)) return false;
        const uint64_t* KB = T.lktW.data();
        std::fill(r0, r0 + CW, 0ull);
        std::fill(sL, sL + CW, 0ull);
        uint64_t any = 0;
        for (int u = 0; u < CW; ++u)
            for (uint64_t x = sD[u]; x; x &= x - 1) {
                const uint64_t* k = KB + (size_t)(u * 64 + __builtin_ctzll(x)) * CW;
                for (int q = 0; q < CW; ++q) r0[q] ^= k[q];
            }
        for (int q = 0; q < CW; ++q) any |= r0[q];
        if (any) {   // pivot-indexed (reduce_low)
            int np = 0;
            std::fill(pm, pm + CW, 0ull);
            for (int j = rD; j < R; ++j) {
                std::copy(KB + (size_t)j * CW, KB + (size_t)(j + 1) * CW, tv);
                std::fill(tc, tc + CW, 0ull);
                tc[j >> 6] = 1ull << (j & 63);
                reduce_low<0>(tv, tc, CW, pm, pidx, lv, lc);
                int p = -1;
                for (int u = 0; u < CW && p < 0; ++u)
                    if (tv[u]) p = u * 64 + __builtin_ctzll(tv[u]);
                if (p < 0) continue;
                std::copy(tv, tv + CW, lv + (size_t)np * CW);
                std::copy(tc, tc + CW, lc + (size_t)np * CW);
                pidx[p] = np++;
                pm[p >> 6] |= 1ull << (p & 63);
            }
            reduce_low<0>(r0, sL, CW, pm, pidx, lv, lc);
            any = 0;
            for (int q = 0; q < CW; ++q) any |= r0[q];
            if (any) return false;
        }
        std::copy(off + WD_, off + WF_, o);
        for (int u = 0; u < CW; ++u)
            for (uint64_t x = sD[u] | sL[u]; x; x &= x - 1) {
                const uint64_t* b = betaO_.data() + (size_t)(u * 64 + __builtin_ctzll(x)) * WO_;
                for (int w = 0; w < WO_; ++w) o[w] ^= b[w];
            }
        return true;
    }
    if (kind == 0) {
        bool ok = true;
        for (int w = 0; w < WD_; ++w) ok = ok && off[w] == d[w];
        std::copy(off + WD_, off + WF_, o);
        return ok;
    }
    if (kind == 1) {
        // coordinates x over beta: x is in the coin span (in beta coordinates, W^perp, W = ker N_O) iff
        // XOR_{j : x_j = 1} Kt[j] = 0. x_D = sD is fixed by d; solve XOR_{j in L, x_j} Kt[j] = r0 over the beta_L
        // columns (private-pivot elimination, combinations tracked); o = off_O + (beta x)_O
        bool ok;
        const uint64_t sD = beta_d_coords(T, off, d, ok);
        if (!ok) return false;
        uint64_t r0 = 0;
        for (uint64_t x = sD; x; x &= x - 1) r0 ^= T.lkt[__builtin_ctzll(x)];
        uint64_t sL = 0;
        if (r0) {   // pivot-indexed (reduce_low1)
            uint64_t pv[64], pc[64], pmask = 0;
            int pidx[64];
            int np = 0;
            for (int j = rho_ - rhoL_; j < rho_; ++j) {
                uint64_t v = T.lkt[j], c = 1ull << j;
                reduce_low1(v, c, pmask, pidx, pv, pc);
                if (!v) continue;
                pv[np] = v;
                pc[np] = c;
                pidx[__builtin_ctzll(v)] = np;
                pmask |= v & (0 - v);
                ++np;
            }
            reduce_low1(r0, sL, pmask, pidx, pv, pc);
            if (r0) return false;
        }
        std::copy(off + WD_, off + WF_, o);
        for (uint64_t x = sD | sL; x; x &= x - 1) {
            const uint64_t* b = betaO_.data() + (size_t)__builtin_ctzll(x) * WO_;
            for (int w = 0; w < WO_; ++w) o[w] ^= b[w];
        }
        return true;
    }
    // general path (after law_dim): reduce off ^ (d, 0) by the detector-pivot rows
    T.lq.resize((size_t)WF_ + 1);
    uint64_t* v = T.lq.data();
    std::copy(off, off + WF_, v);
    for (int w = 0; w < WD_; ++w) v[w] ^= d[w];
    for (size_t r = 0; r < T.ldp.size(); ++r)
        if ((v[(size_t)T.ldp[r] >> 6] >> (T.ldp[r] & 63)) & 1ull)
            for (int w = 0; w < WF_; ++w) v[w] ^= T.ldr[r * (size_t)WF_ + (size_t)w];
    uint64_t rest = 0;
    for (int w = 0; w < WD_; ++w) rest |= v[w];
    std::copy(v + WD_, v + WF_, o);
    return rest == 0;
}

bool XdemSampler::law_in_c(ThreadState& T, int kind, uint64_t* x) const {
    // x (WO_ words; destroyed) in C?
    uint64_t nz = 0;
    if (kind == 3) {   // kind 1's rule in CW_ words
        const int CW = CW_, rD = rho_ - rhoL_;
        uint64_t* s = T.lrw.data() + 3 * (size_t)CW;
        std::fill(s, s + CW, 0ull);
        for (int j = 0; j < rhoL_; ++j) {
            const int p = pL_[(size_t)j];
            if ((x[p >> 6] >> (p & 63)) & 1ull) {
                const uint64_t* b = bLo_.data() + (size_t)j * WO_;
                for (int w = 0; w < WO_; ++w) x[w] ^= b[w];
                s[(rD + j) >> 6] |= 1ull << ((rD + j) & 63);
            }
        }
        for (int w = 0; w < WO_; ++w) nz |= x[w];
        if (nz) return false;
        uint64_t* r = T.lrw.data() + 4 * (size_t)CW;
        std::fill(r, r + CW, 0ull);
        for (int u = 0; u < CW; ++u)
            for (uint64_t y = s[u]; y; y &= y - 1) {
                const uint64_t* k = &T.lktW[(size_t)(u * 64 + __builtin_ctzll(y)) * CW];
                for (int q = 0; q < CW; ++q) r[q] ^= k[q];
            }
        for (int q = 0; q < CW; ++q) nz |= r[q];
        return nz == 0;
    }
    if (kind == 1) {   // x = (beta s)_O with s supported on beta_L (unique: beta_L's observable parts are independent),
        const int rD = rho_ - rhoL_;   // and s in the coin span
        uint64_t s = 0;
        for (int j = 0; j < rhoL_; ++j) {
            const int p = pL_[(size_t)j];
            if ((x[p >> 6] >> (p & 63)) & 1ull) {
                const uint64_t* b = bLo_.data() + (size_t)j * WO_;
                for (int w = 0; w < WO_; ++w) x[w] ^= b[w];
                s |= 1ull << (rD + j);
            }
        }
        for (int w = 0; w < WO_; ++w) nz |= x[w];
        if (nz) return false;
        uint64_t r = 0;
        for (; s; s &= s - 1) r ^= T.lkt[__builtin_ctzll(s)];
        return r == 0;
    }
    if (kind == 2)
        for (size_t r = 0; r < T.lcp.size(); ++r)
            if ((x[(size_t)T.lcp[r] >> 6] >> (T.lcp[r] & 63)) & 1ull)
                for (int w = 0; w < WO_; ++w) x[w] ^= T.lcr[r * (size_t)WO_ + (size_t)w];
    for (int w = 0; w < WO_; ++w) nz |= x[w];
    return nz == 0;
}

bool XdemSampler::law_has(ThreadState& T, int kind, const uint64_t* off, const uint64_t* d, const uint64_t* c) {
    // (d, c) in off + span(coins)? (after law_dim)
    if (T.lx.size() < (size_t)WO_ + 1) T.lx.resize((size_t)WO_ + 1);
    uint64_t* x = T.lx.data();
    if (kind == 3) {   // kind 1's rule in CW_ words
        const int CW = CW_, rD = rho_ - rhoL_;
        uint64_t* sD = T.lrw.data();
        uint64_t* r = sD + CW;
        if (!beta_d_coords_w(T, off, d, sD)) return false;
        for (int w = 0; w < WO_; ++w) x[w] = c[w] ^ off[WD_ + w];
        std::fill(r, r + CW, 0ull);
        const uint64_t* KB = T.lktW.data();
        for (int u = 0; u < CW; ++u)
            for (uint64_t z = sD[u]; z; z &= z - 1) {
                const int j = u * 64 + __builtin_ctzll(z);
                const uint64_t* b = betaO_.data() + (size_t)j * WO_;
                for (int w = 0; w < WO_; ++w) x[w] ^= b[w];
                const uint64_t* k = KB + (size_t)j * CW;
                for (int q = 0; q < CW; ++q) r[q] ^= k[q];
            }
        for (int j = 0; j < rhoL_; ++j) {
            const int p = pL_[(size_t)j];
            if ((x[p >> 6] >> (p & 63)) & 1ull) {
                const uint64_t* b = bLo_.data() + (size_t)j * WO_;
                for (int w = 0; w < WO_; ++w) x[w] ^= b[w];
                const uint64_t* k = KB + (size_t)(rD + j) * CW;
                for (int q = 0; q < CW; ++q) r[q] ^= k[q];
            }
        }
        uint64_t nz = 0;
        for (int w = 0; w < WO_; ++w) nz |= x[w];
        for (int q = 0; q < CW; ++q) nz |= r[q];
        return nz == 0;
    }
    if (kind == 1) {   // the beta coordinates of (d, c) ^ off (sD from d, sL from the rest), then in the coin span
        bool ok;
        const uint64_t sD = beta_d_coords(T, off, d, ok);
        if (!ok) return false;
        for (int w = 0; w < WO_; ++w) x[w] = c[w] ^ off[WD_ + w];
        for (uint64_t z = sD; z; z &= z - 1) {
            const uint64_t* b = betaO_.data() + (size_t)__builtin_ctzll(z) * WO_;
            for (int w = 0; w < WO_; ++w) x[w] ^= b[w];
        }
        uint64_t r = 0;
        for (uint64_t z = sD; z; z &= z - 1) r ^= T.lkt[__builtin_ctzll(z)];
        const int rD = rho_ - rhoL_;
        for (int j = 0; j < rhoL_; ++j) {
            const int p = pL_[(size_t)j];
            if ((x[p >> 6] >> (p & 63)) & 1ull) {
                const uint64_t* b = bLo_.data() + (size_t)j * WO_;
                for (int w = 0; w < WO_; ++w) x[w] ^= b[w];
                r ^= T.lkt[rD + j];
            }
        }
        uint64_t nz = 0;
        for (int w = 0; w < WO_; ++w) nz |= x[w];
        return nz == 0 && r == 0;
    }
    if (!law_given(T, kind, off, d, x)) return false;
    for (int w = 0; w < WO_; ++w) x[w] ^= c[w];
    return law_in_c(T, kind, x);
}

// eta^> and tau_K of E^_< by the cocycle restricted to the late parts (SPEC §4.4 step 1). Under §4.4's availability
// condition no line >= tc rotates, so tau_K is read by no later line: it is computed for the export only.
// compose_ext's bit-sliced form: each line's late exponents as word groups (kl_), added to two planes (k_lo_, k_hi_)
// negated where the sign register (the earlier lines' tau on the rotated axes, from their xa_ records) is set; one step
// per axis word. The late axes are rotated, so the register holds every sign a late exponent reads.
void XdemSampler::r2b_k(const int32_t* h, size_t nh, std::vector<std::pair<int, int>>& eta_late,
                        std::vector<uint64_t>* tau_k, int mut) const {
    eta_late.clear();
    if (k_lo_.size() != (size_t)WA_) {
        k_lo_.assign((size_t)WA_, 0);
        k_hi_.assign((size_t)WA_, 0);
        k_sg_.assign((size_t)WA_, 0);
        k_dm_.assign((size_t)(WA_ + 63) / 64, 0);
    }
    uint64_t* lo = k_lo_.data();
    uint64_t* hi = k_hi_.data();
    uint64_t* sg = k_sg_.data();
    uint64_t* dm = k_dm_.data();
    const uint64_t* xa = xa_.data();
    auto add = [&](uint64_t w, uint64_t e0, uint64_t e1) {
        const uint64_t sgn = mut == kXdemMutNoCocycle ? 0ull : sg[w];   // MUTANT 4: no cocycle sign
        e1 ^= e0 & sgn;
        const uint64_t c = lo[w] & e0;
        lo[w] ^= e0;
        hi[w] ^= e1 ^ c;
        dm[w >> 6] |= 1ull << (w & 63);
    };
    const uint64_t* prev = nullptr;   // the last line's record, its tau not yet in the register
    for (size_t j = 0; j < nh; ++j) {
        const size_t id = (size_t)h[j];
        if (prev) {
            const uint64_t* q = prev + 1 + 3 * (prev[0] & 0xffffffffu);
            for (uint64_t k = 0; k < (prev[0] >> 32); ++k, q += 2) sg[q[0]] ^= q[1];
            prev = nullptr;
        }
        if (const uint32_t ko = kl_of_[id]) {
            const uint64_t* g = &kl_[ko];
            const uint64_t ng = *g++;
            for (uint64_t k = 0; k < ng; ++k, g += 3) add(g[0], g[1], g[2]);
        }
        if (mut == kXdemMutEarlyLate)   // MUTANT 1: E^< too
            for (uint64_t k = early_off_[id]; k < early_off_[id + 1]; ++k) {
                const int a = early_[k].axis, e = early_[k].e;
                add((uint64_t)(a >> 6), (uint64_t)(e & 1) << (a & 63), (uint64_t)((e >> 1) & 1) << (a & 63));
            }
        if (xo_of_[id]) prev = xa + xo_of_[id];
    }
    for (size_t j = 0; j + 1 < nh; ++j) {   // clear the register (the last line's tau was never applied)
        const uint32_t xo = xo_of_[(size_t)h[j]];
        if (!xo) continue;
        const uint64_t* q = xa + xo;
        const uint64_t* tt = q + 1 + 3 * (q[0] & 0xffffffffu);
        for (uint64_t k = 0; k < (q[0] >> 32); ++k, tt += 2) sg[tt[0]] = 0;
    }
    for (int di = 0; di * 64 < WA_; ++di) {   // drain the planes, axes ascending
        for (uint64_t z = dm[di]; z; z &= z - 1) {
            const int w = di * 64 + __builtin_ctzll(z);
            for (uint64_t y = lo[w] | hi[w]; y; y &= y - 1) {
                const int b = __builtin_ctzll(y);
                eta_late.emplace_back(w * 64 + b, (int)(((hi[w] >> b) & 1ull) << 1 | ((lo[w] >> b) & 1ull)));
            }
            lo[w] = hi[w] = 0;
        }
        dm[di] = 0;
    }
    if (tau_k) {   // the export: tau_K on the late axes, from the full tau
        std::vector<uint64_t> tau((size_t)WA_, 0);
        for (size_t j = 0; j < nh; ++j)
            for (uint64_t k = tau_full_off_[(size_t)h[j]]; k < tau_full_off_[(size_t)h[j] + 1]; ++k)
                tau[(size_t)tau_full_[k] >> 6] ^= 1ull << (tau_full_[k] & 63);
        tau_k->assign((size_t)WA_, 0);
        for (int a : late_axes_)
            if ((tau[(size_t)a >> 6] >> (a & 63)) & 1ull) (*tau_k)[(size_t)a >> 6] |= 1ull << (a & 63);
    }
}

namespace {
// law_list's fast-path memo: hash slots (a power of two), entries before it is cleared, and the bypass rule (after
// kLawMemoProbe lookups of a call, fewer than kLawMemoMinHits hits turn the memo off for the rest of the call)
constexpr size_t kLawMemoSlots = 8192;
constexpr size_t kLawMemoCap = 4096;
constexpr uint64_t kLawMemoProbe = 4096;
constexpr uint64_t kLawMemoMinHits = 1024;
static_assert(kLawMemoCap < kLawMemoSlots, "the open-addressing table needs a free slot");
}  // namespace

int XdemSampler::law_list(ThreadState& T, const int32_t* ids, size_t n, size_t at, const std::vector<uint64_t>& kE,
                          uint64_t* row, int* entry_out) {
    // compose_ext's cocycle (SPEC §1.2) over the extended lines of ids, with K (no tau) between ids[at - 1] and
    // ids[at]: K's rotation is negated where the register (tau of the lines before it) is set, as any later line's
    uint64_t* lo = T.plo.data();
    uint64_t* hi = T.phi.data();
    uint64_t* dm = T.pdirty.data();
    uint64_t* sg = T.sgn.data();
    const uint64_t* xa = xa_.data();
    const uint64_t* prev = nullptr;   // the last composed line, its tau not yet in the register
    bool any = false;
    auto add = [&](const uint64_t* g, uint64_t ng) {
        for (uint64_t k = 0; k < ng; ++k, g += 3) {
            const uint64_t w = g[0], e0 = g[1], e1 = g[2] ^ (g[1] & sg[w]);
            const uint64_t c = lo[w] & e0;
            lo[w] ^= e0;
            hi[w] ^= e1 ^ c;
            dm[w >> 6] |= 1ull << (w & 63);
        }
        any = true;
    };
    auto flush = [&]() {
        if (!prev) return;
        const uint64_t* t = prev + 1 + 3 * (prev[0] & 0xffffffffu);
        for (uint64_t k = 0; k < (prev[0] >> 32); ++k, t += 2) sg[t[0]] ^= t[1];
        prev = nullptr;
    };
    const uint64_t nK = kE.size() / 3;
    for (size_t j = 0; j <= n; ++j) {
        if (j == at && nK) {
            flush();
            add(kE.data(), nK);
        }
        if (j == n) break;
        const uint32_t xo = xo_of_[(size_t)ids[j]];
        if (!xo) continue;
        const uint64_t* r = xa + xo;
        flush();
        add(r + 1, r[0] & 0xffffffffu);
        prev = r;
    }
    // clear the register: every set bit lies in a word some extended line's tau entry names
    for (size_t j = 0; j < n; ++j) {
        const uint32_t xo = xo_of_[(size_t)ids[j]];
        if (!xo) continue;
        const uint64_t* q = xa + xo;
        const uint64_t* t = q + 1 + 3 * (q[0] & 0xffffffffu);
        for (uint64_t k = 0; k < (q[0] >> 32); ++k, t += 2) sg[t[0]] = 0;
    }
    if (!any) return 0;   // no rotation and no K: a point law (the row is the flips)
    if (!fast_ || T.mbypass) return law_row(T, row, entry_out);
    // fast path: the memo (offset = flips ^ g(eta), Kt(eta)); R words of Kt per entry: rho (one word each) or rho CW_
    // (the multi-word law's beta-coordinate kernel words)
    const int WA = WA_, WF = WF_, R = CW_ ? rho_ * CW_ : rho_, WK = 2 * WA_;
    uint64_t* lkt = CW_ ? T.lktW.data() : T.lkt;
    if (T.mhits + T.mmiss == kLawMemoProbe && T.mhits < kLawMemoMinHits) {
        T.mbypass = true;
        return law_row(T, row, entry_out);
    }
    if (T.mtab.empty()) {
        T.mtab.assign(kLawMemoSlots, -1);
        T.mmask = kLawMemoSlots - 1;
    }
    T.mtmp.resize((size_t)(WK + WF));
    uint64_t* key = T.mtmp.data();
    std::copy(lo, lo + WA, key);
    std::copy(hi, hi + WA, key + WA);
    uint64_t h = hash_words(key, WK) & T.mmask;
    for (int32_t e; (e = T.mtab[h]) >= 0; h = (h + 1) & T.mmask)
        if (std::memcmp(&T.mkey[(size_t)e * WK], key, sizeof(uint64_t) * (size_t)WK) == 0) {
            ++T.mhits;
            for (int di = 0; di * 64 < WA; ++di) {   // drain the planes (post_fast's contract)
                for (uint64_t z = dm[di]; z; z &= z - 1) {
                    const int w = di * 64 + __builtin_ctzll(z);
                    lo[w] = hi[w] = 0;
                }
                dm[di] = 0;
            }
            const uint64_t* g = T.mg.data() + (size_t)e * WF;
            for (int w = 0; w < WF; ++w) row[w] ^= g[w];
            if (!CW_) {
                std::copy(T.mkt.data() + (size_t)e * R, T.mkt.data() + (size_t)e * R + R, lkt);
            } else if (T.mkind[(size_t)e] == 3) {   // rows of kw words (the kernel's nker bits), the rest 0
                const size_t kw = (size_t)T.mkw[(size_t)e], CW = (size_t)CW_;
                const uint64_t* src = T.mkt.data() + T.mko[(size_t)e];
                for (int j = 0; j < rho_; ++j) {
                    uint64_t* dst = lkt + (size_t)j * CW;
                    std::copy(src + (size_t)j * kw, src + (size_t)(j + 1) * kw, dst);
                    std::fill(dst + kw, dst + CW, 0ull);
                }
            }
            T.lnpv = T.mnpv[(size_t)e];
            T.lnpvD = T.mnpvD[(size_t)e];
            return T.mkind[(size_t)e];
        }
    ++T.mmiss;
    if (T.mkind.size() >= kLawMemoCap) {   // full: clear (bounded memory)
        std::fill(T.mtab.begin(), T.mtab.end(), -1);
        T.mkey.clear();
        T.mg.clear();
        T.mkt.clear();
        T.mko.clear();
        T.mkw.clear();
        T.mnpv.clear();
        T.mnpvD.clear();
        T.mkind.clear();
        h = hash_words(key, WK) & T.mmask;
    }
    // post_fast can throw (an internal check): every memo array is appended only after it returns
    std::copy(row, row + WF, key + WK);   // the flips
    const int kind = law_row(T, row, entry_out);
    const int e = (int)T.mkind.size();
    T.mkey.insert(T.mkey.end(), key, key + WK);
    for (int w = 0; w < WF; ++w) T.mg.push_back(key[WK + w] ^ row[w]);   // g = offset ^ flips
    if (!CW_) {
        T.mkt.insert(T.mkt.end(), lkt, lkt + R);
    } else {   // multi-word: only the kernel's ceil(nker / 64) words per row (kind 3; kind 0 needs none)
        const size_t kw = kind == 3 ? ((size_t)T.lnk + 63) / 64 : 0, CW = (size_t)CW_;
        T.mko.push_back(T.mkt.size());
        T.mkw.push_back((int32_t)kw);
        for (int j = 0; j < rho_ && kw; ++j)
            T.mkt.insert(T.mkt.end(), lkt + (size_t)j * CW, lkt + (size_t)j * CW + kw);
    }
    T.mnpv.push_back(T.lnpv);
    T.mnpvD.push_back(T.lnpvD);
    T.mkind.push_back((int8_t)kind);
    T.mtab[h] = e;
    return kind;
}

namespace {
// a CSR (off[shots + 1], ids[n]) of line ids, checked: off[0] = 0, non-decreasing, off[shots] = n, ids in [0, L)
void check_lines_csr(const char* who, const char* what, size_t shots, const int64_t* off, const int32_t* ids, size_t n,
                     size_t L) {
    const std::string w = std::string(who) + ": " + what;
    if (!off) throw std::invalid_argument(w + " offsets missing");
    if (off[0] != 0) throw std::invalid_argument(w + " offsets must start at 0");
    bool dec = false;
    for (size_t s = 0; s < shots; ++s) dec |= off[s + 1] < off[s];
    if (dec)
        for (size_t s = 0; s < shots; ++s)
            if (off[s + 1] < off[s]) throw std::invalid_argument(w + " offsets decrease at shot " + std::to_string(s));
    if ((uint64_t)off[shots] != (uint64_t)n)
        throw std::invalid_argument(w + " offsets end at " + std::to_string(off[shots]) + " but there are " +
                                    std::to_string(n) + " ids");
    bool bad = false;
    for (size_t i = 0; i < n; ++i) bad |= (uint32_t)ids[i] >= (uint64_t)L;   // a negative id wraps above L
    if (!bad) return;
    for (size_t s = 0; s < shots; ++s)
        for (int64_t i = off[s]; i < off[s + 1]; ++i)
            if (ids[i] < 0 || (size_t)ids[i] >= L)
                throw std::out_of_range(w + " line id " + std::to_string(ids[i]) + " at shot " + std::to_string(s) +
                                        " is outside [0, num_lines = " + std::to_string(L) + ")");
}

// ids[0, n) in file order with pairs cancelled (SPEC §1.2): an ascending list (a fired list) is read in place, any
// other is sorted and cancelled into buf. Returns the list, its length in len.
const int32_t* canon_view(const int32_t* ids, int64_t n, std::vector<int32_t>& buf, size_t& len) {
    bool asc = true;
    for (int64_t i = 1; i < n; ++i) asc &= ids[i - 1] < ids[i];
    if (asc) {
        len = (size_t)n;
        return ids;
    }
    buf.assign(ids, ids + n);
    std::sort(buf.begin(), buf.end());
    size_t k = 0;
    for (int32_t id : buf) {
        if (k && buf[k - 1] == id) --k;
        else buf[k++] = id;
    }
    buf.resize(k);
    len = k;
    return buf.data();
}

// the extended lines of a and b (in order) are the same: their composites are equal
bool same_ext(const int32_t* a, size_t na, const int32_t* b, size_t nb, const std::vector<uint32_t>& xo_of) {
    size_t i = 0, j = 0;
    for (;;) {
        while (i < na && !xo_of[(size_t)a[i]]) ++i;
        while (j < nb && !xo_of[(size_t)b[j]]) ++j;
        if (i == na || j == nb) return i == na && j == nb;
        if (a[i] != b[j]) return false;
        ++i;
        ++j;
    }
}
}  // namespace

void XdemSampler::score_lines(size_t shots, const uint8_t* dets, const int64_t* foff, const int32_t* fids,
                              size_t nfids, const int64_t* hoff, const int32_t* hids, size_t nhids, double* out,
                              const XdemLinesOptions& opt, int8_t* action, uint8_t* cbits) {
    std::lock_guard<std::mutex> lk(mu_);
    score_lines_locked(shots, dets, foff, fids, nfids, hoff, hids, nhids, out, opt, action, cbits, true);
}

void XdemSampler::score_lines_locked(size_t shots, const uint8_t* dets, const int64_t* foff, const int32_t* fids,
                                     size_t nfids, const int64_t* hoff, const int32_t* hids, size_t nhids, double* out,
                                     const XdemLinesOptions& opt, int8_t* action, uint8_t* cbits, bool count) {
    const bool r2b = opt.corr == Correction::R2b;
    if (r2b && !r2b_.first)
        throw XdemFileError("score(lines=, correction=\"r2b\"): the file is not in canonical single-stage form at t_c: " +
                            r2b_.second + "; pass correction=\"optimal\" or \"none\"");
    check_lines_csr("XdemSampler.score_lines", "fired", shots, foff, fids, nfids, L_);
    check_lines_csr("XdemSampler.score_lines", "hypothesis", shots, hoff, hids, nhids, L_);
    if (opt.corr == Correction::Optimal) {
        score_optimal(shots, dets, foff, fids, hoff, hids, out, nullptr);
        return;
    }
    ThreadState& T = slot(0);
    T.mhits = T.mmiss = 0;
    T.mbypass = false;
    const int64_t tc = opt.mut == kXdemMutTcMinus1 ? std::max<int64_t>(F_.tc - 1, 0) : F_.tc;   // MUTANT 5
    const size_t W = (size_t)W_, nbD = nbD_;
    const int WD = WD_, WO = WO_;
    // rh: h's flips; r0 / r1: L0's / L1's rows (offset after the law); ra: the corrected shot's row; dw: d; c: the
    // prediction; x: law_given's output
    std::vector<uint64_t> rh(W + 1), r0(W + 1), r1(W + 1), ra(W + 1), dw((size_t)WD + 1, 0), c((size_t)WO + 1),
        x((size_t)WO + 1);
    std::vector<int32_t> hbuf, tbuf;
    std::vector<std::pair<int, int>> eta;
    std::vector<uint64_t> kE, none, sktW;   // sktW: L1's multi-word law scratch while L0 is computed
    const uint64_t dlast = (D_ & 63) ? (1ull << (D_ & 63)) - 1 : ~0ull;
    const uint8_t blast = (D_ & 7) ? (uint8_t)((1u << (D_ & 7)) - 1) : (uint8_t)0xff;
    auto flips = [&](const int32_t* ids, size_t n, uint64_t* row) {
        std::fill(row, row + W, 0ull);
        for (size_t j = 0; j < n; ++j)
            for (uint64_t t = loff_[(size_t)ids[j]]; t < loff_[(size_t)ids[j] + 1]; ++t) row[lent_[t].word] ^= lent_[t].mask;
    };
    auto below = [&](const int32_t* ids, size_t n) {   // the number of ids < tc (ids ascending)
        return (size_t)(std::lower_bound(ids, ids + n, (int64_t)tc, [](int32_t a, int64_t b) { return (int64_t)a < b; }) -
                        ids);
    };
    // T needs the law path when one of its lines is extended; h also when a line carries any rotation field (an
    // E^< + E^> = 0 line with no tau on a rotated axis is not extended, yet its late exponent enters eta^>)
    auto t_ext = [&](const int32_t* ids, size_t n) {
        for (size_t j = 0; j < n; ++j)
            if (xo_of_[(size_t)ids[j]]) return true;
        return false;
    };
    auto h_rot = [&](const int32_t* ids, size_t n) {
        for (size_t j = 0; j < n; ++j) {
            const size_t i = (size_t)ids[j];
            if (xo_of_[i] || late_off_[i + 1] > late_off_[i] || early_off_[i + 1] > early_off_[i]) return true;
        }
        return false;
    };
    auto put_c = [&](size_t s, const uint64_t* cv) {
        if (cbits)
            for (int o = 0; o < O_; ++o)
                cbits[s * (size_t)O_ + (size_t)o] = (uint8_t)((cv[(size_t)o >> 6] >> (o & 63)) & 1ull);
    };
    uint64_t n_app = 0, n_wh = 0, n_coin = 0, n_out = 0;
    for (size_t s = 0; s < shots; ++s) {
        if (hoff[s + 1] == hoff[s] && foff[s + 1] == foff[s]) {
            // nothing fired, nothing hypothesised: both laws are the point 0, c = 0, pass = [d = 0]
            const uint8_t* db = dets + s * nbD;
            uint8_t acc = 0;
            for (size_t b = 0; b + 1 < nbD; ++b) acc |= db[b];
            if (nbD) acc |= db[nbD - 1] & blast;
            out[s] = acc ? 0.0 : 1.0;
            n_out += acc != 0;
            if (action) action[s] = kLinesNone;
            if (cbits) std::fill(cbits + s * (size_t)O_, cbits + (s + 1) * (size_t)O_, (uint8_t)0);
            continue;
        }
        size_t nh, nt;
        const int32_t* hv = canon_view(hids + hoff[s], hoff[s + 1] - hoff[s], hbuf, nh);
        const int32_t* tv = canon_view(fids + foff[s], foff[s + 1] - foff[s], tbuf, nt);
        if (nbD) {
            std::memcpy(dw.data(), dets + s * nbD, nbD);
            dw[(size_t)WD - 1] &= dlast;
        }
        flips(hv, nh, rh.data());
        if (!h_rot(hv, nh) && !t_ext(tv, nt)) {
            // no rotation in h, no extended line in T: both laws are points (offsets f_h, f_T); no K. c = f_h|_O
            // (L0's definite value when d = f_h|_D, the fallback otherwise); pass = [(d, c) = f_T]
            flips(tv, nt, ra.data());
            bool inh = true, ok = true;
            for (int w = 0; w < WD; ++w) {
                inh = inh && rh[(size_t)w] == dw[(size_t)w];
                ok = ok && ra[(size_t)w] == dw[(size_t)w];
            }
            for (int w = 0; w < WO; ++w) ok = ok && ra[(size_t)(WD + w)] == rh[(size_t)(WD + w)];
            n_out += !inh;
            out[s] = ok ? 1.0 : 0.0;
            if (action) action[s] = kLinesNone;
            put_c(s, rh.data() + WD);
            continue;
        }
        // the laws: L0 = law(h), L1 = law(h with K); a law's prediction is its definite value at d (dim C = 0 and d in
        // its detector support), else f_h|_O
        auto fh = [&]() { std::copy(rh.begin() + WD, rh.begin() + WD + WO, c.begin()); };
        int kind0 = 0, e0 = -1, k0 = 0, kind1 = 0, e1 = -1, k1 = 0;
        bool in0 = false;
        auto law0 = [&]() {
            std::copy(rh.begin(), rh.end(), r0.begin());
            kind0 = law_list(T, hv, nh, nh, none, r0.data(), &e0);
            k0 = law_dim(T, kind0, e0);
            in0 = law_given(T, kind0, r0.data(), dw.data(), x.data());
        };
        size_t nlt = 0;
        eta.clear();
        if (r2b) {
            nlt = below(hv, nh);
            r2b_k(hv, nlt, eta, nullptr, opt.mut);
        }
        int8_t act = kLinesNone;
        int live = 0;   // the law whose scratch is live: 0 = L0, 1 = L1, -1 = neither
        if (eta.empty()) {   // "none", or no late exponent: L0 alone
            law0();
            n_out += !in0;
            if (k0 == 0 && in0 && opt.mut != kXdemMutXorFlips) std::copy(x.begin(), x.begin() + WO, c.begin());
            else fh();
        } else {
            kE.clear();
            for (const auto& [a, e] : eta) {   // K's exponent groups (w, odd bits, 2-bits), words ascending
                if (kE.empty() || kE[kE.size() - 3] != (uint64_t)(a >> 6)) kE.insert(kE.end(), {(uint64_t)(a >> 6), 0, 0});
                kE[kE.size() - 2] |= (uint64_t)(e & 1) << (a & 63);
                kE[kE.size() - 1] |= (uint64_t)((e >> 1) & 1) << (a & 63);
            }
            auto law1 = [&]() {
                std::copy(rh.begin(), rh.end(), r1.begin());
                kind1 = law_list(T, hv, nh, nlt, kE, r1.data(), &e1);
                k1 = law_dim(T, kind1, e1);
            };
            if (opt.mut == kXdemMutEarlyLate || opt.mut == kXdemMutTcMinus1) {
                // these levers put axes with detectors into K: SPEC §4.4's order, L0 then L1
                law0();
                n_out += !in0;
                if (k0 == 0 && in0 && opt.mut != kXdemMutXorFlips) std::copy(x.begin(), x.begin() + WO, c.begin());
                else fh();
                law1();
                live = 1;
                if (k1 <= k0) {
                    act = kLinesApplied;
                    const bool in1 = law_given(T, kind1, r1.data(), dw.data(), x.data());
                    if (k1 == 0 && in1) std::copy(x.begin(), x.begin() + WO, c.begin());
                    else fh();
                } else {
                    act = kLinesWithheld;
                }
            } else {
                // L1 first. K is a rotation about late axes, none of which has a detector in its img (§4.4's
                // availability), so K commutes with every detector parity's X: L0 and L1 have the same detector
                // marginal, and L1's detector support is L0's (lines_outside). k1 = 0 is applied whatever k0 is, so L0
                // is computed only when k1 > 0; then a prediction is definite only for a withheld K (k0 < k1 there).
                law1();
                const bool in1 = law_given(T, kind1, r1.data(), dw.data(), x.data());
                n_out += !in1;
                live = 1;
                if (k1 == 0) {
                    act = kLinesApplied;
                    if (in1 && opt.mut != kXdemMutXorFlips) std::copy(x.begin(), x.begin() + WO, c.begin());
                    else fh();
                } else {
                    uint64_t skt[64];
                    const int snpv = T.lnpv, snpvD = T.lnpvD;
                    if (kind1 == 1) std::copy(T.lkt, T.lkt + rho_, skt);
                    if (kind1 == 3) sktW.assign(T.lktW.begin(), T.lktW.end());
                    law0();
                    if (k1 <= k0 || opt.mut == kXdemMutNoGuard) {   // MUTANT 2: no guard
                        act = kLinesApplied;
                        fh();
                        // L1's scratch back (fast path); a general-path entry may be gone after L0's lookup
                        if (kind1 == 1 || kind1 == 3) {
                            if (kind1 == 1) std::copy(skt, skt + rho_, T.lkt);
                            else std::copy(sktW.begin(), sktW.end(), T.lktW.begin());
                            T.lnpv = snpv;
                            T.lnpvD = snpvD;
                        }
                        live = kind1 == 2 ? -1 : 1;
                    } else {
                        act = kLinesWithheld;
                        if (k0 == 0 && in0 && opt.mut != kXdemMutXorFlips) std::copy(x.begin(), x.begin() + WO, c.begin());
                        else fh();
                        live = 0;
                    }
                }
            }
        }
        // the corrected shot: T's lines with K (if applied); the live law serves when it is the same composite
        // (offset = the law's offset + the flip difference)
        const bool withK = act == kLinesApplied;
        flips(tv, nt, ra.data());
        int kinda, ea = -1, kc;
        if (live == (withK ? 1 : 0) && same_ext(tv, nt, hv, nh, xo_of_)) {
            const uint64_t* lr = live == 1 ? r1.data() : r0.data();
            for (size_t w = 0; w < W; ++w) ra[w] ^= lr[w] ^ rh[w];
            kinda = live == 1 ? kind1 : kind0;
            ea = live == 1 ? e1 : e0;
            kc = live == 1 ? k1 : k0;
        } else {
            kinda = law_list(T, tv, nt, withK ? below(tv, nt) : nt, withK ? kE : none, ra.data(), &ea);
            kc = law_dim(T, kinda, ea);
        }
        const bool in = law_has(T, kinda, ra.data(), dw.data(), c.data());
        out[s] = in ? (kc ? std::ldexp(1.0, -kc) : 1.0) : 0.0;
        n_app += act == kLinesApplied;
        n_wh += act == kLinesWithheld;
        n_coin += kc > 0;
        if (action) action[s] = act;
        put_c(s, c.data());
    }
    if (!count) return;
    lines_shots_ += shots;
    lines_applied_ += n_app;
    lines_withheld_ += n_wh;
    lines_coin_after_ += n_coin;
    lines_outside_ += n_out;
}

// ---- the optimal correction K (Task 8; SPEC §4.2, §4.3, §4.6, §4.7)

namespace {
// File layout (detector d = bit d, observable o = bit D + o; nin words) -> the internal layout (observable o = bit
// 64 WD + o; WF words)
void file_to_internal(const uint64_t* in, size_t nin, int D, int WD, uint64_t* out, int WF) {
    std::fill(out, out + WF, 0ull);
    for (size_t wi = 0; wi < nin; ++wi)
        for (uint64_t x = in[wi]; x; x &= x - 1) {
            const long long j = 64LL * (long long)wi + __builtin_ctzll(x);
            const long long k = j < D ? j : 64LL * WD + (j - D);
            out[k >> 6] |= 1ull << (k & 63);
        }
}
inline bool bit_at(const uint64_t* w, long long j) { return (w[j >> 6] >> (j & 63)) & 1ull; }
inline int parity_and(const uint64_t* a, const uint64_t* b, int n) {
    int p = 0;
    for (int i = 0; i < n; ++i) p ^= __builtin_parityll(a[i] & b[i]);
    return p;
}
inline bool any_word(const uint64_t* a, size_t n) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc |= a[i];
    return acc != 0;
}
// x (n words) into a fully reduced echelon (rows of n words, pivot = the row's lowest bit, unique to it): false when
// x is dependent. x is destroyed.
bool echelon_insert(std::vector<uint64_t>& rows, std::vector<int>& piv, uint64_t* x, int n) {
    for (size_t i = 0; i < piv.size(); ++i)
        if (bit_at(x, piv[i]))
            for (int w = 0; w < n; ++w) x[w] ^= rows[i * (size_t)n + (size_t)w];
    int p = -1;
    for (int w = 0; w < n && p < 0; ++w)
        if (x[w]) p = w * 64 + __builtin_ctzll(x[w]);
    if (p < 0) return false;
    for (size_t i = 0; i < piv.size(); ++i)
        if (bit_at(&rows[i * (size_t)n], p))
            for (int w = 0; w < n; ++w) rows[i * (size_t)n + (size_t)w] ^= x[w];
    rows.insert(rows.end(), x, x + n);
    piv.push_back(p);
    return true;
}
}  // namespace

void XdemSampler::k_key(const int32_t* ids, size_t n, uint64_t* key) const {
    // compose_ext's cocycle (SPEC §1.2) over the extended lines of ids (file order) into key = [eta's odd plane | its
    // 2-plane | tau on the rotated axes], WA words each; the tau part is the sign register as it goes
    const size_t WA = (size_t)WA_;
    std::fill(key, key + 3 * WA, 0ull);
    uint64_t *lo = key, *hi = key + WA, *tau = key + 2 * WA;
    for (size_t j = 0; j < n; ++j) {
        const uint32_t xo = xo_of_[(size_t)ids[j]];
        if (!xo) continue;
        const uint64_t* r = &xa_[xo];
        const uint64_t* g = r + 1;
        for (uint64_t k = 0; k < (r[0] & 0xffffffffu); ++k, g += 3) {
            const uint64_t w = g[0], e0 = g[1], e1 = g[2] ^ (g[1] & tau[w]);
            const uint64_t c = lo[w] & e0;
            lo[w] ^= e0;
            hi[w] ^= e1 ^ c;
        }
        for (uint64_t k = 0; k < (r[0] >> 32); ++k, g += 2) tau[g[0]] ^= g[1];
    }
}

void XdemSampler::k_build(const uint64_t* key, KEntry& E, KScratch& X) const {
    const int D = D_, O = O_, WD = WD_, WF = WF_;
    const size_t A = (size_t)A_, nwK = words_for(F_.r), nwH = words_for(F_.m), WA = (size_t)WA_;
    // scratch: eta and colpos stay all 0 / -1 between calls (reset where set)
    if (X.eta.size() != A) X.eta.assign(A, 0);
    if (X.colpos.size() != (size_t)D) X.colpos.assign((size_t)D, -1);
    if (X.axB.size() != A) {   // the file axes' kappa / sigma / bhat, flat
        X.axK.assign(A * nwK, 0);
        X.axH.assign(A * nwH, 0);
        X.axB.assign(A, 0);
        for (size_t a = 0; a < A; ++a) {
            const Axis& x = F_.axes[a];
            std::copy(x.k.begin(), x.k.begin() + (long)std::min(x.k.size(), nwK), X.axK.begin() + (long)(a * nwK));
            for (int32_t h : x.hidden) X.axH[a * nwH + ((size_t)h >> 6)] ^= 1ull << (h & 63);
            X.axB[a] = (uint8_t)x.b;
        }
    }
    X.v.assign((size_t)WD + 1, 0);
    X.cmb.assign((size_t)WD + 1, 0);
    X.Pm.assign((size_t)WD + 1, 0);
    uint8_t* eta = X.eta.data();
    // 1. h's composite from its key (k_key): eta mod 4 per axis, tau on the rotated axes
    const uint64_t* tau = key + 2 * WA;
    E.eta.clear();
    X.od.clear();
    X.two.clear();
    for (size_t w = 0; w < WA; ++w)
        for (uint64_t y = key[w] | key[WA + w]; y; y &= y - 1) {
            const int b = __builtin_ctzll(y);
            const int a = (int)(w * 64) + b;
            const int e = (int)((key[w] >> b) & 1ull) | (int)((key[WA + w] >> b) & 1ull) << 1;
            eta[a] = (uint8_t)e;
            E.eta.emplace_back(a, e);
            if (e & 1) X.od.push_back(a);
            else X.two.push_back(a);   // e = 2 (the key holds nonzero exponents only)
        }
    auto img = [&](int a) { return &img_[(size_t)a * (size_t)WF]; };
    // 2. Omega = sum_{a in Od} iota_a iota_a^T, iota_a = img_a on the detectors: column j = XOR of iota_a over the a
    // with j in iota_a. Greedy pivots: the columns in detector order into an echelon keyed by each row's highest bit,
    // tracking the combination over the pivot detectors.
    X.cols.clear();
    X.colv.clear();
    for (int a : X.od) {
        const uint64_t* io = img(a);
        for (int w = 0; w < WD; ++w)
            for (uint64_t y = io[w]; y; y &= y - 1) {
                const int j = w * 64 + __builtin_ctzll(y);
                if (X.colpos[(size_t)j] < 0) {
                    X.colpos[(size_t)j] = (int)X.cols.size();
                    X.cols.push_back(j);
                    X.colv.resize(X.colv.size() + (size_t)WD, 0);
                }
                uint64_t* c = &X.colv[(size_t)X.colpos[(size_t)j] * (size_t)WD];
                for (int u = 0; u < WD; ++u) c[u] ^= io[u];
            }
    }
    std::sort(X.cols.begin(), X.cols.end());
    X.ev.clear();
    X.ec.clear();
    X.elead.clear();
    uint64_t* v = X.v.data();
    uint64_t* cmb = X.cmb.data();
    auto reduce = [&]() -> bool {   // v (with its combination cmb) reduced; true when it reaches 0
        for (;;) {
            const int hb = highest(v, WD);
            if (hb < 0) return true;
            size_t i = 0;
            while (i < X.elead.size() && X.elead[i] != hb) ++i;
            if (i == X.elead.size()) return false;
            for (int w = 0; w < WD; ++w) {
                v[w] ^= X.ev[i * (size_t)WD + (size_t)w];
                cmb[w] ^= X.ec[i * (size_t)WD + (size_t)w];
            }
        }
    };
    for (int j : X.cols) {
        const uint64_t* c = &X.colv[(size_t)X.colpos[(size_t)j] * (size_t)WD];
        std::copy(c, c + WD, v);
        std::fill(cmb, cmb + WD, 0ull);
        cmb[(size_t)j >> 6] |= 1ull << (j & 63);
        X.colpos[(size_t)j] = -1;
        if (reduce()) continue;   // dependent: not a pivot
        X.ev.insert(X.ev.end(), v, v + WD);
        X.ec.insert(X.ec.end(), cmb, cmb + WD);
        X.elead.push_back(highest(v, WD));
    }
    // 3. per observable k: t_k = XOR of iota_a over a in Od with img_a[L_k]; t_k = Omega P_k for the pivot set P_k, or
    // the readout is lossy. G_k = {L_k} + P_k, S_k = {a in Od : img_a . G_k = 1}, Phi_{S_k}, y_k, and eps's exponent
    // without f.G (§4.2): e0 = n2 + bhat_S + tau(S) + q / 2, q = sum_S eta + y (mod 4, even).
    E.rows.assign((size_t)O, KEntry::Row());
    E.P.clear();
    E.S.clear();
    E.Sw.clear();
    E.lossy = false;
    E.lmask.clear();
    E.lf_built = false;
    X.phiI.assign((size_t)O * (size_t)WF, 0);
    X.phiK.assign((size_t)O * nwK, 0);
    X.phiH.assign((size_t)O * nwH, 0);
    uint64_t* Pm = X.Pm.data();
    // one pass over Od's observable bits: t_k (O * WD words) and, per k, the a in Od with img_a[L_k] = 1 (a CSR,
    // ascending a: S_k when P_k is empty)
    X.tk.assign((size_t)O * (size_t)WD + 1, 0);
    X.lkoff.assign((size_t)O + 1, 0);
    for (int a : X.od)
        for (int w = WD; w < WF; ++w)
            for (uint64_t y = img(a)[w]; y; y &= y - 1) ++X.lkoff[(size_t)((w - WD) * 64 + __builtin_ctzll(y)) + 1];
    for (int k = 0; k < O; ++k) X.lkoff[(size_t)k + 1] += X.lkoff[(size_t)k];
    X.lk.resize((size_t)X.lkoff[(size_t)O]);
    X.lkpos.assign(X.lkoff.begin(), X.lkoff.end() - 1);
    for (int a : X.od) {
        const uint64_t* ia = img(a);
        for (int w = WD; w < WF; ++w)
            for (uint64_t y = ia[w]; y; y &= y - 1) {
                const int k = (w - WD) * 64 + __builtin_ctzll(y);
                X.lk[(size_t)X.lkpos[(size_t)k]++] = a;
                uint64_t* t = &X.tk[(size_t)k * (size_t)WD];
                for (int u = 0; u < WD; ++u) t[u] ^= ia[u];
            }
    }
    E.S.reserve(X.lk.size() + X.od.size());
    for (int k = 0; k < O; ++k) {
        KEntry::Row& R = E.rows[(size_t)k];
        R.p0 = R.p1 = (uint32_t)E.P.size();
        R.s0 = R.s1 = (uint32_t)E.S.size();
        R.w0 = R.w1 = (uint32_t)E.Sw.size();
        const long long L = 64LL * WD + k;
        std::copy(&X.tk[(size_t)k * (size_t)WD], &X.tk[(size_t)k * (size_t)WD] + WD, v);
        std::fill(cmb, cmb + WD, 0ull);
        if (!reduce()) {
            R.lossy = 1;
            E.lossy = true;
            if (E.lmask.empty()) E.lmask.assign((size_t)WO_ + 1, 0);
            E.lmask[(size_t)k >> 6] |= 1ull << (k & 63);
            continue;
        }
        std::copy(cmb, cmb + WD, Pm);
        for (int w = 0; w < WD; ++w)
            for (uint64_t y = Pm[w]; y; y &= y - 1) E.P.push_back(w * 64 + __builtin_ctzll(y));
        R.p1 = (uint32_t)E.P.size();
        auto inG = [&](int a) { return (int)bit_at(img(a), L) ^ parity_and(img(a), Pm, WD); };
        uint64_t* PI = &X.phiI[(size_t)k * (size_t)WF];
        uint64_t* PK = X.phiK.data() + (size_t)k * nwK;   // (data() + offset: nwK or nwH may be 0)
        uint64_t* PH = X.phiH.data() + (size_t)k * nwH;
        int bS = 0, tS = 0, qs = 0;
        const bool noP = R.p1 == R.p0;   // P_k empty: S_k = the a in Od with img_a[L_k] = 1
        const int* sl = noP ? X.lk.data() + X.lkoff[(size_t)k] : X.od.data();
        const size_t sn = noP ? (size_t)(X.lkoff[(size_t)k + 1] - X.lkoff[(size_t)k]) : X.od.size();
        for (size_t si = 0; si < sn; ++si) {
            const int a = sl[si];
            if (!noP && !inG(a)) continue;
            E.S.push_back(a);
            if (E.Sw.size() == R.w0 || E.Sw[E.Sw.size() - 2] != (uint64_t)(a >> 6))
                E.Sw.insert(E.Sw.end(), {(uint64_t)(a >> 6), 0});
            E.Sw.back() |= 1ull << (a & 63);
            const uint64_t* ia = img(a);
            for (int w = 0; w < WF; ++w) PI[w] ^= ia[w];
            const uint64_t* ka = X.axK.data() + (size_t)a * nwK;
            for (size_t w = 0; w < nwK; ++w) PK[w] ^= ka[w];
            const uint64_t* ha = X.axH.data() + (size_t)a * nwH;
            for (size_t w = 0; w < nwH; ++w) PH[w] ^= ha[w];
            bS ^= X.axB[(size_t)a];
            tS ^= (int)bit_at(tau, a);
            qs += eta[a];
        }
        R.s1 = (uint32_t)E.S.size();
        R.w1 = (uint32_t)E.Sw.size();
        if (any_word(PI, (size_t)WD)) {
            for (const auto& [a, e] : E.eta) eta[a] = 0;
            throw std::logic_error(
                "xdem optimal: rotated readout with a Z on a detector in the non-lossy case (LOOP §4(ii))");
        }
        R.y = (uint8_t)bit_at(PI, L);
        int n2 = 0;
        for (int a : X.two) n2 ^= inG(a);
        const int q = (qs + R.y) & 3;
        if (q & 1) {
            for (const auto& [a, e] : E.eta) eta[a] = 0;
            throw std::logic_error("xdem optimal: internal: q = sum_S eta + y is odd (the rotated readout's sign is not "
                                   "real; docs/xdem.md §4)");
        }
        R.e0 = (uint8_t)((n2 ^ bS ^ tS ^ (q >> 1)) & 1);
        R.ts = (uint8_t)tS;
    }
    for (const auto& [a, e] : E.eta) eta[a] = 0;
    // 4. K_diag's strings (step 5/6), per non-lossy k in order: A_k = Phi_{S_k} minus L_k minus the observables L_k'
    // (k' < k, non-lossy) whose edge (k', k) is already placed; S_{L_k}^{y_k} prod_{q in A_k} CZ_{L_k q} as rotations
    // about Z_{L_k} (exponent 3 y + 3 [A_k != 0]), Z_{A_k} (3) and Z_{L_k} Z_{A_k} (1); b = 0.
    // Their gauge coordinates (Task 8 finding; SPEC §4.3 step 6 gives kappa only): a string's readout part is not the
    // image of its kappa under the file's header map (§3.5(b): img_a = Lm kappa_a for every file axis), so the §1.3
    // law over [T, K] is taken in the model whose gauge coordinates are kappa' = [kappa | delta], delta = img - Lm
    // kappa (observable columns only; a file axis has delta = 0), where §3.5(b) holds again. Lm kappa_S = img_S
    // (= Phi_{S_k}'s readout part), so delta = the observable bits of Phi_{S_k} that A_k leaves out (L_k, the edges
    // placed earlier) for Z_{A_k}, delta + e_{L_k} for Z_{L_k} Z_{A_k}, and e_{L_k} for Z_{L_k}. kappa only (the spec
    // text) is exact except where Phi_{S_k} holds a lossy observable (the CZ edge to it): measured against the stim
    // oracle, 0 / 13 739 wrong without a lossy observable, 51 wrong and 14 odd-sum refusals / 388 shots with one;
    // kappa' 0 / 14 127 (an internal acceptance script, 960 random models). This is
    // ERRATA E2. The strings carry bhat = 0 (ERRATA E3; SPEC §4.3 step 6's "bhat = the XOR of their members" would
    // make the rotation about the signed Z_S a CZ times Z_L, not step 5's physical CZ / S): both the generic axes
    // (ensure_xax) and the fast law's items set b = 0.
    E.ximg.clear();
    E.xkap.clear();
    E.xhid.clear();
    E.xdl.clear();
    E.xe.clear();
    E.xk.clear();
    E.xax_built = E.xfast_built = false;
    X.Aimg.assign((size_t)WF, 0);
    X.zl.assign((size_t)WF, 0);
    uint64_t* Aimg = X.Aimg.data();
    uint64_t* zl = X.zl.data();
    // lmk: Lm kappa of the string (internal layout; nullptr: 0); kap (nwK words), hid (nwH words): nullptr = none
    auto push = [&](const uint64_t* imgI, const uint64_t* lmk, const uint64_t* kap, const uint64_t* hid, int e, int k) {
        for (int w = 0; w < WD; ++w)
            if (imgI[w] ^ (lmk ? lmk[w] : 0ull))
                throw std::logic_error("xdem optimal: internal: a K string's readout part leaves its gauge image on a "
                                       "detector (LOOP §4(ii))");
        E.ximg.insert(E.ximg.end(), imgI, imgI + WF);
        for (int w = WD; w < WF; ++w) E.xdl.push_back(imgI[w] ^ (lmk ? lmk[w] : 0ull));
        if (kap) E.xkap.insert(E.xkap.end(), kap, kap + nwK);
        else E.xkap.resize(E.xkap.size() + nwK, 0);
        if (hid) E.xhid.insert(E.xhid.end(), hid, hid + nwH);
        else E.xhid.resize(E.xhid.size() + nwH, 0);
        E.xe.push_back((uint8_t)e);
        E.xk.push_back(k);
    };
    for (int k = 0; k < O; ++k) {
        const KEntry::Row& R = E.rows[(size_t)k];
        if (R.lossy) continue;
        const long long L = 64LL * WD + k;
        const uint64_t* PI = &X.phiI[(size_t)k * (size_t)WF];
        const uint64_t* PK = X.phiK.data() + (size_t)k * nwK;
        const uint64_t* PH = X.phiH.data() + (size_t)k * nwH;
        std::copy(PI, PI + WF, Aimg);
        Aimg[(size_t)(L >> 6)] &= ~(1ull << (L & 63));
        for (int w = WD; w < WF; ++w)   // the observables of Phi_{S_k}: the coupling graph is symmetric (LOOP §4(v))
            for (uint64_t y = PI[w]; y; y &= y - 1) {
                const int o = (w - WD) * 64 + __builtin_ctzll(y);
                if (o == k || E.rows[(size_t)o].lossy) continue;
                if (!bit_at(&X.phiI[(size_t)o * (size_t)WF], L))
                    throw std::logic_error("xdem optimal: internal: the observable coupling is not symmetric (L_" +
                                           std::to_string(o) + " in Phi_S of L_" + std::to_string(k) +
                                           " but not conversely) (LOOP §4(v))");
                if (o < k) Aimg[(size_t)w] &= ~(1ull << (o & 63));   // the edge (o, k) was placed at o
            }
        const bool nonempty = any_word(Aimg, (size_t)WF) || any_word(PK, nwK) || any_word(PH, nwH);
        const int eL = (3 * R.y + 3 * (int)nonempty) & 3;
        if (eL) {
            std::fill(zl, zl + WF, 0ull);
            zl[(size_t)(L >> 6)] = 1ull << (L & 63);
            push(zl, nullptr, nullptr, nullptr, eL, -1);
        }
        if (nonempty) {
            push(Aimg, PI, PK, PH, 3, k);
            Aimg[(size_t)(L >> 6)] ^= 1ull << (L & 63);
            push(Aimg, PI, PK, PH, 1, k);
        }
    }
}

void XdemSampler::k_support(KEntry& E) {
    // law(h)'s detector support at flip 0 (lines_outside: per shot, d ^ f_h|_D in g_D + span(eD)), from the sampler's
    // own law code (law_coins: the fast law, one-word or multi-word; the general path only under force_general): h's
    // eta into slot 0's planes (zero between shots), the offset's detector part, the coins' detector parts in a fully
    // reduced echelon. The support is a coset, so any offset representative and coin basis give the same sets.
    ThreadState& T = slot(0);
    const int WD = WD_;
    uint64_t* lo = T.plo.data();
    uint64_t* hi = T.phi.data();
    uint64_t* dm = T.pdirty.data();
    for (const auto& [a, e] : E.eta) {
        const int w = a >> 6;
        lo[w] |= (uint64_t)(e & 1) << (a & 63);
        hi[w] |= (uint64_t)((e >> 1) & 1) << (a & 63);
        dm[w >> 6] |= 1ull << (w & 63);
    }
    std::vector<uint64_t> row((size_t)W_ + 1, 0), rows;
    int entry = -1;
    law_coins(T, row.data(), rows, &entry);
    E.gD.assign((size_t)WD + 1, 0);
    std::copy(row.begin(), row.begin() + WD, E.gD.begin());
    E.eD.clear();
    E.eP.clear();
    std::vector<uint64_t> x((size_t)WD + 1, 0);
    for (size_t i = 0; i < rows.size(); i += (size_t)WF_) {
        std::copy(&rows[i], &rows[i] + WD, x.begin());
        echelon_insert(E.eD, E.eP, x.data(), WD);
    }
    E.sup_built = true;
}

void XdemSampler::k_frames(const KEntry& E, const uint64_t* fh, const uint64_t* d, uint64_t* fobs) const {
    // f_K,k = s . P_k xor [eps_k != (-1)^{y_k}], eps's exponent = f_h . G_k + e0_k (SPEC §4.2, §4.3 step 5)
    std::fill(fobs, fobs + WO_, 0ull);
    for (int k = 0; k < O_; ++k) {
        const KEntry::Row& R = E.rows[(size_t)k];
        if (R.lossy) continue;
        int f = R.e0 ^ R.y ^ (int)bit_at(fh, 64LL * WD_ + k);
        for (uint32_t t = R.p0; t < R.p1; ++t) f ^= (int)bit_at(d, E.P[t]) ^ (int)bit_at(fh, E.P[t]);
        if (f) fobs[(size_t)k >> 6] |= 1ull << (k & 63);
    }
}

void XdemSampler::k_hi_images(const uint64_t* key, uint64_t* row) const {
    // row ^= img_a for eta_a in {2, 3} of a key (eta = 3 is eta = 1 plus img_a: the law is built with every odd
    // exponent 1)
    const size_t WA = (size_t)WA_;
    for (size_t w = 0; w < WA; ++w)
        for (uint64_t y = key[WA + w]; y; y &= y - 1) {
            const uint64_t* im = &img_[(w * 64 + (size_t)__builtin_ctzll(y)) * (size_t)WF_];
            for (int u = 0; u < WF_; ++u) row[u] ^= im[u];
        }
}

void XdemSampler::k_strings(const KEntry& E, const uint8_t* sg, uint64_t* row) const {
    // K's strings with an exponent in {2, 3} after their sign (-1)^{tau_T(S_k)} (sg: per observable) add their image
    for (size_t x = 0; x < E.xe.size(); ++x) {
        const int e = E.xe[x], sgn = E.xk[x] >= 0 ? sg[(size_t)E.xk[x]] : 0;
        if (((sgn ? 4 - e : e) & 3) & 2) {
            const uint64_t* im = &E.ximg[x * (size_t)WF_];
            for (int u = 0; u < WF_; ++u) row[u] ^= im[u];
        }
    }
}

void XdemSampler::k_xax(KEntry& E) const {
    // K's strings as File-layout axes for the generic law_into (kappa' = [kappa | delta]: delta at gauge bit r + o)
    if (E.xax_built) return;
    const size_t nx = E.xe.size(), nwK = words_for(F_.r), nwH = words_for(F_.m);
    const size_t nwK2 = words_for((long long)F_.r + O_);
    E.xax.assign(nx, Axis());
    for (size_t x = 0; x < nx; ++x) {
        Axis& a = E.xax[x];
        a.img = to_file_bits(&E.ximg[x * (size_t)WF_], D_, O_, WD_, WF_);
        a.k.assign(nwK2, 0);
        std::copy(E.xkap.data() + x * nwK, E.xkap.data() + (x + 1) * nwK, a.k.begin());
        for (int w = 0; w < WO_; ++w)
            for (uint64_t y = E.xdl[x * (size_t)WO_ + (size_t)w]; y; y &= y - 1) {
                const long long t = (long long)F_.r + (long long)w * 64 + __builtin_ctzll(y);
                a.k[(size_t)(t >> 6)] |= 1ull << (t & 63);
            }
        for (size_t w = 0; w < nwH; ++w)
            for (uint64_t y = E.xhid[x * nwH + w]; y; y &= y - 1) a.hidden.push_back((int32_t)(w * 64 + __builtin_ctzll(y)));
        a.b = 0;
    }
    E.xax_built = true;
}

void XdemSampler::k_law_generic(KEntry& E, const uint64_t* odd, XLaw& X, LawScratch& LS) const {
    // the generic law of [T, K] for T's odd file-axis set odd: law_into over the file's axes and K's strings, every odd
    // exponent 1, at flip 0; to the internal layout; the coins' detector parts (echelon); dim C = rank - their rank
    k_xax(E);
    const size_t A = (size_t)A_, WA = (size_t)WA_, nwF = words_for((long long)D_ + O_);
    const int WD = WD_, WF = WF_;
    Composed c;
    c.flip.w.assign(nwF, 0);
    for (size_t w = 0; w < WA; ++w)
        for (uint64_t y = odd[w]; y; y &= y - 1) c.eta.emplace_back((int)(w * 64) + __builtin_ctzll(y), 1);
    for (size_t x = 0; x < E.xe.size(); ++x)
        if (E.xe[x] & 1) c.eta.emplace_back((int)(A + x), 1);
    AxisView av;
    av.base = F_.axes.data();
    av.nbase = A;
    av.extra = E.xax.data();
    av.nextra = E.xax.size();
    law_into(D_, O_, F_.r + O_, av, c, LS);   // kappa' = [kappa | delta] (k_build step 4)
    X.fast = false;
    X.q.assign((size_t)WF + 1, 0);
    file_to_internal(LS.off.data(), nwF, D_, WD, X.q.data(), WF);
    X.coins.assign((size_t)LS.rank * (size_t)WF, 0);
    X.piv.assign((size_t)LS.rank, 0);
    X.eD.clear();
    X.eP.clear();
    std::vector<uint64_t> dx((size_t)WD + 1, 0);
    for (int i = 0; i < LS.rank; ++i) {
        uint64_t* ci = &X.coins[(size_t)i * (size_t)WF];
        file_to_internal(&LS.coins[(size_t)i * nwF], nwF, D_, WD, ci, WF);
        X.piv[(size_t)i] = highest(ci, WF);
        std::copy(ci, ci + WD, dx.begin());
        echelon_insert(X.eD, X.eP, dx.data(), WD);
    }
    X.kdim = LS.rank - (int)X.eP.size();
}

void XdemSampler::k_lossy_setup(KEntry& E, const XLaw& X, const KFast& KF) const {
    // ERRATA E4: X = law([h, K]) (h's own composite with K's strings, q holding the eta in {2, 3} images). Its support
    // is {f_h + q + span(coins)}; the coins are explicit on the generic law and, on the fast law, beta'.z for z in the
    // kernel of z -> XOR_{j : z_j} Kt[j] (fast_in's membership test). They go into one fully reduced echelon with the
    // pivot = the lowest bit (internal layout: detectors below observables), so a row with an observable pivot has no
    // detector bit, and the support given d is o0 + span(those rows' observable parts) (k_lossy_frames).
    const int WD = WD_, WO = WO_, WF = WF_;
    E.lfq.assign(X.q.begin(), X.q.begin() + WF);
    E.lfq.push_back(0);
    std::vector<uint64_t> rows, v((size_t)WF + 1, 0);
    std::vector<int> piv;
    if (!X.fast) {
        for (size_t i = 0; i < X.piv.size(); ++i) {
            std::copy(&X.coins[i * (size_t)WF], &X.coins[i * (size_t)WF] + WF, v.begin());
            echelon_insert(rows, piv, v.data(), WF);
        }
    } else if (X.cw) {   // the same in cw words per vector
        const int R = KF.rhoX, C = X.cw;
        std::vector<uint64_t> pv, pc, a((size_t)C), c((size_t)C);
        std::vector<int> pl;
        for (int j = 0; j < R; ++j) {
            std::copy(&X.fw[(size_t)j * C], &X.fw[(size_t)(j + 1) * C], a.begin());
            std::fill(c.begin(), c.end(), 0ull);
            c[(size_t)j >> 6] = 1ull << (j & 63);
            for (size_t q = 0; q < pl.size(); ++q)
                if ((a[(size_t)pl[q] >> 6] >> (pl[q] & 63)) & 1ull)
                    for (int u = 0; u < C; ++u) {
                        a[(size_t)u] ^= pv[q * C + (size_t)u];
                        c[(size_t)u] ^= pc[q * C + (size_t)u];
                    }
            int p = -1;
            for (int u = 0; u < C && p < 0; ++u)
                if (a[(size_t)u]) p = u * 64 + __builtin_ctzll(a[(size_t)u]);
            if (p >= 0) {
                pv.insert(pv.end(), a.begin(), a.end());
                pc.insert(pc.end(), c.begin(), c.end());
                pl.push_back(p);
                continue;
            }
            std::fill(v.begin(), v.end(), 0ull);   // a kernel vector: its coin beta'.c
            k_xor_beta_x(KF, c.data(), v.data());
            echelon_insert(rows, piv, v.data(), WF);
        }
    } else {
        const int R = KF.rhoX;
        const uint64_t* Kt = X.Kt();
        uint64_t pv[64], pc[64], pl[64];
        int np = 0;
        for (int j = 0; j < R; ++j) {
            uint64_t a = Kt[j], c = 1ull << j;
            for (int q = 0; q < np; ++q)
                if (a & pl[q]) {
                    a ^= pv[q];
                    c ^= pc[q];
                }
            if (a) {
                pv[np] = a;
                pc[np] = c;
                pl[np] = a & (0 - a);
                ++np;
                continue;
            }
            std::fill(v.begin(), v.end(), 0ull);   // a kernel vector: its coin beta'.c
            xor_beta(v.data(), rho_ >= 64 ? c : c & ((1ull << rho_) - 1));
            for (int t = 0; t < KF.nn; ++t)
                if ((c >> (rho_ + t)) & 1ull)
                    for (int u = 0; u < WF; ++u) v[(size_t)u] ^= KF.nvI[(size_t)t * (size_t)WF + (size_t)u];
            echelon_insert(rows, piv, v.data(), WF);
        }
    }
    E.lfD.clear();
    E.lfDP.clear();
    E.lfL.clear();
    E.lfLP.clear();
    std::vector<uint64_t> o((size_t)WO + 1, 0);
    for (size_t i = 0; i < piv.size(); ++i) {
        const uint64_t* r = &rows[i * (size_t)WF];
        if (piv[i] < 64 * WD) {
            E.lfD.insert(E.lfD.end(), r, r + WF);
            E.lfDP.push_back(piv[i]);
            continue;
        }
        for (int w = 0; w < WO; ++w) o[(size_t)w] = r[WD + w] & E.lmask[(size_t)w];
        echelon_insert(E.lfL, E.lfLP, o.data(), WO);
    }
    E.lf_built = true;
}

void XdemSampler::k_lossy_frames(const KEntry& E, const uint64_t* fh, const uint64_t* d, uint64_t* fobs,
                                 uint64_t* u) const {
    // ERRATA E4 (SPEC §7 Q4 as amended): the frame on the lossy observables is a point c of the support of law([h, K])
    // restricted to them, given d. The rule (deterministic, basis-free): the support point whose difference from h's
    // flips f_h on the lossy observables is lexicographically least, observable 0 first -- so c = f_h there whenever
    // f_h is in the support (R2b step 4's c_xor, "none"'s prediction), and otherwise the nearest point in that order.
    // It is c = f_L + red(f_L + o0), red = reduction by the lowest-bit fully reduced echelon lfL (it clears the pivots,
    // the order's minimum), o0 = the observable part of (d, 0) + f_h + q reduced by the detector-pivot rows. When d is
    // outside law(h)'s detector support there is no point: c = 0 on the lossy observables (SPEC §7 Q1, owner ruling
    // 2026-11-17, E4 fix round 1). u: the caller's scratch, WF + WO words (no per-shot allocation).
    const int WD = WD_, WO = WO_, WF = WF_;
    uint64_t* y = u + WF;
    for (int w = 0; w < WF; ++w) u[w] = E.lfq[(size_t)w] ^ fh[w];
    for (int w = 0; w < WD; ++w) u[w] ^= d[w];
    for (size_t i = 0; i < E.lfDP.size(); ++i)
        if ((u[E.lfDP[i] >> 6] >> (E.lfDP[i] & 63)) & 1ull) {
            const uint64_t* r = &E.lfD[i * (size_t)WF];
            for (int w = 0; w < WF; ++w) u[w] ^= r[w];
        }
    if (any_word(u, (size_t)WD)) {   // outside: c = 0
        for (int w = 0; w < WO; ++w) fobs[w] &= ~E.lmask[(size_t)w];
        return;
    }
    for (int w = 0; w < WO; ++w) y[w] = (u[WD + w] ^ fh[WD + w]) & E.lmask[(size_t)w];
    for (size_t i = 0; i < E.lfLP.size(); ++i)
        if ((y[E.lfLP[i] >> 6] >> (E.lfLP[i] & 63)) & 1ull) {
            const uint64_t* r = &E.lfL[i * (size_t)WO];
            for (int w = 0; w < WO; ++w) y[w] ^= r[w];
        }
    for (int w = 0; w < WO; ++w) {
        const uint64_t m = E.lmask[(size_t)w];
        fobs[w] = (fobs[w] & ~m) | ((fh[WD + w] & m) ^ y[w]);
    }
}

void XdemSampler::k_fast_setup(KFast& F) const {
    F = KFast();
    F.built = true;
    if (!fast_ || A_ == 0) return;
    if (fast1_ && F_.r + O_ <= 64) {
        k_fast_setup1(F);   // one word per vector when rho' <= 64 too
        if (F.ok) return;
        F = KFast();
        F.built = true;
    }
    // CW words per vector: the new vectors as above (no bound); then each file axis' beta' coordinates (the detector
    // part over beta_D, the rest over beta_L; img_a lies in span(beta)) and kappa' = kappa (delta = 0 for a file axis)
    const int rD = rho_ - rhoL_, WO = WO_, WD = WD_, R = rho_;
    std::vector<uint64_t> v((size_t)WO + 1, 0);
    std::vector<std::vector<int>> eb((size_t)O_);   // e_{L_o}'s beta' coordinate bits
    for (int o = 0; o < O_; ++o) {
        std::fill(v.begin(), v.end(), 0ull);
        v[(size_t)o >> 6] = 1ull << (o & 63);
        for (int i = 0; i < rhoL_; ++i)
            if (bit_at(v.data(), pL_[(size_t)i])) {
                for (int w = 0; w < WO; ++w) v[(size_t)w] ^= bLo_[(size_t)i * WO + (size_t)w];
                eb[(size_t)o].push_back(rD + i);
            }
        for (int t = 0; t < F.nn; ++t)
            if (bit_at(v.data(), F.np[(size_t)t])) {
                for (int w = 0; w < WO; ++w) v[(size_t)w] ^= F.nv[(size_t)t * WO + (size_t)w];
                eb[(size_t)o].push_back(R + t);
            }
        if (any_word(v.data(), (size_t)WO)) {
            F.np.push_back(highest(v.data(), WO));
            F.nobs.push_back(o);
            F.nv.insert(F.nv.end(), v.begin(), v.begin() + WO);
            F.nvI.resize(F.nvI.size() + (size_t)WF_, 0);
            std::copy(v.begin(), v.begin() + WO, F.nvI.end() - WF_ + WD);
            eb[(size_t)o].push_back(R + F.nn);
            ++F.nn;
        }
    }
    F.rhoX = R + F.nn;
    const int cw = (std::max(F.rhoX, F_.r + O_) + 63) / 64;
    F.cw = std::max(cw, 1);
    const size_t C = (size_t)F.cw;
    F.ecW.assign((size_t)O_ * C, 0);
    for (int o = 0; o < O_; ++o)
        for (int j : eb[(size_t)o]) F.ecW[(size_t)o * C + ((size_t)j >> 6)] ^= 1ull << (j & 63);
    F.TcB.assign((size_t)A_ * C, 0);
    F.KcB.assign((size_t)A_ * C, 0);
    std::vector<uint64_t> dd((size_t)WD + 1, 0), ob((size_t)WO + 1, 0);
    for (int a = 0; a < A_; ++a) {
        const uint64_t* im = &img_[(size_t)a * WF_];
        uint64_t* t = &F.TcB[(size_t)a * C];
        std::copy(im, im + WD, dd.begin());
        std::copy(im + WD, im + WF_, ob.begin());
        for (int j = 0; j < rD; ++j)
            if (bit_at(dd.data(), pD_[(size_t)j])) {
                for (int w = 0; w < WD; ++w) dd[(size_t)w] ^= bDd_[(size_t)j * WD + (size_t)w];
                for (int w = 0; w < WO; ++w) ob[(size_t)w] ^= betaO_[(size_t)j * WO + (size_t)w];
                t[j >> 6] ^= 1ull << (j & 63);
            }
        for (int i = 0; i < rhoL_; ++i)
            if (bit_at(ob.data(), pL_[(size_t)i])) {
                for (int w = 0; w < WO; ++w) ob[(size_t)w] ^= bLo_[(size_t)i * WO + (size_t)w];
                t[(rD + i) >> 6] ^= 1ull << ((rD + i) & 63);
            }
        if (any_word(dd.data(), (size_t)WD) || any_word(ob.data(), (size_t)WO))
            throw std::logic_error("xdem optimal: internal: an axis image outside span(beta)");
    }
    k_fast_rebase(F, F.TcB.data(), F.ecW.data(), C);
    F.ok = true;
}

void XdemSampler::k_fast_setup1(KFast& F) const {
    // one word per vector (the Task 8 fix round 1 setup); F.ok = false when rho' > 64
    const int rD = rho_ - rhoL_, WO = WO_;
    F.ecoord.assign((size_t)O_, 0);
    std::vector<uint64_t> v((size_t)WO + 1, 0);
    for (int o = 0; o < O_; ++o) {
        std::fill(v.begin(), v.end(), 0ull);
        v[(size_t)o >> 6] = 1ull << (o & 63);
        uint64_t c = 0;
        for (int i = 0; i < rhoL_; ++i)   // span(beta) on zero detector parts = span(beta_L), in build order
            if (bit_at(v.data(), pL_[(size_t)i])) {
                for (int w = 0; w < WO; ++w) v[(size_t)w] ^= bLo_[(size_t)i * WO + (size_t)w];
                c |= 1ull << (rD + i);
            }
        for (int t = 0; t < F.nn; ++t)
            if (bit_at(v.data(), F.np[(size_t)t])) {
                for (int w = 0; w < WO; ++w) v[(size_t)w] ^= F.nv[(size_t)t * WO + (size_t)w];
                c |= 1ull << (rho_ + t);
            }
        if (any_word(v.data(), (size_t)WO)) {
            if (rho_ + F.nn >= 64) return;   // rho' > 64: k_fast_setup takes the cw-word setup
            F.np.push_back(highest(v.data(), WO));
            F.nobs.push_back(o);
            F.nv.insert(F.nv.end(), v.begin(), v.begin() + WO);
            F.nvI.resize(F.nvI.size() + (size_t)WF_, 0);
            std::copy(v.begin(), v.begin() + WO, F.nvI.end() - WF_ + WD_);
            c |= 1ull << (rho_ + F.nn);
            ++F.nn;
        }
        F.ecoord[(size_t)o] = c;
    }
    F.rhoX = rho_ + F.nn;
    k_fast_rebase(F, Tc_.data(), F.ecoord.data(), 1);
    F.ok = true;
}

// GSL: the law of [T, K] in graph-state form. Its items' beta' coordinates t' are L' kappa' for one linear L' on
// kappa' = [kappa | delta] (r + O bits): a file axis has t' = L kappa (§3.5(b), delta = 0); a K string's readout part
// is on the observables only (k_build refuses a detector Z there, LOOP §4(ii)) and is Lm kappa plus delta (k_build
// step 4), so t' = L kappa + E delta with E e_o = e_{L_o}'s coordinates. So the generators (the file axes: t' = their
// beta coordinates tb, kappa' = [kappa | 0]; the observable units: t' = e_{L_o}'s coordinates eo, kappa' = e_{r + o})
// span every item's kappa', and gauge_completion's S (the beta creators: creator_, then the observable that made each
// new vector) makes kappa'' = (t' | kappa' restricted to S) injective there: the law matrix keeps its kernel, and its
// rho' image rows are G' = sum t' t'^T. F.KcB / F.Kc1 <- the file axes' kappa''; a string's is the same GaugeSelect
// of (xT, xK) (ensure_xfast). w = the words per vector (F.cw or 1). Once per sampler, on its first "optimal" call.
void XdemSampler::k_fast_rebase(KFast& F, const uint64_t* tb, const uint64_t* eo, size_t w) const {
    const int R = F.rhoX, nk = F_.r + O_;
    const size_t kw = words_for(nk), rw = words_for(F_.r);
    std::vector<uint64_t> tg((size_t)(A_ + O_) * w);   // the generators' t'
    std::copy(tb, tb + (size_t)A_ * w, tg.begin());
    std::copy(eo, eo + (size_t)O_ * w, tg.begin() + (long)((size_t)A_ * w));
    std::vector<int> creator = creator_;
    for (int o : F.nobs) creator.push_back(A_ + o);
    F.S = gauge_completion(A_ + O_, tg.data(), w, kw, creator, nk - R, [&](int g, uint64_t* x) {
        if (g >= A_) {
            const size_t b = (size_t)(F_.r + (g - A_));
            x[b >> 6] ^= 1ull << (b & 63);
            return;
        }
        const std::vector<uint64_t>& k = F_.axes[(size_t)g].k;
        for (size_t u = 0; u < rw && u < k.size(); ++u) x[u] ^= k[u];
    });
    std::vector<uint64_t>& out = w > 1 || F.cw ? F.KcB : F.Kc1;
    out.assign((size_t)A_ * w, 0);
    const GaugeSelect sel(F.S, kw, R);
    for (int a = 0; a < A_; ++a) {
        const std::vector<uint64_t>& k = F_.axes[(size_t)a].k;
        sel(tb + (size_t)a * w, w, k.data(), std::min(k.size(), rw), out.data() + (size_t)a * w, w);
    }
}

void XdemSampler::k_xor_beta_x(const KFast& F, const uint64_t* z, uint64_t* row) const {
    // beta (coordinates j < rho): the one-word table, or via gamma (multi-word sampler: beta_j's gamma coordinates);
    // then the new vectors (coordinates rho + t)
    const int C = F.cw, R = rho_;
    if (CW_) {
        std::vector<uint64_t> g((size_t)CW_, 0);
        for (int j = 0; j < R; ++j)
            if ((z[j >> 6] >> (j & 63)) & 1ull)
                for (int q = 0; q < CW_; ++q) g[(size_t)q] ^= GbW_[(size_t)j * CW_ + q];
        xor_gamma(row, g.data());
    } else {
        xor_beta(row, R >= 64 ? z[0] : z[0] & ((1ull << R) - 1));
    }
    for (int t = 0; t < F.nn; ++t) {
        const int j = R + t;
        if (j < 64 * C && ((z[j >> 6] >> (j & 63)) & 1ull))
            for (int u = 0; u < WF_; ++u) row[u] ^= F.nvI[(size_t)t * (size_t)WF_ + (size_t)u];
    }
}

void XdemSampler::score_optimal(size_t shots, const uint8_t* dets, const int64_t* foff, const int32_t* fids,
                                const int64_t* hoff, const int32_t* hids, double* out, XdemCorrectionOut* cx) {
    if (!km_) km_ = std::make_unique<KMemo>();
    KMemo& M = *km_;
    const int WD = WD_, WO = WO_, WF = WF_, O = O_;
    const size_t WA = (size_t)WA_, nbD = nbD_;
    const uint64_t dlast = (D_ & 63) ? (1ull << (D_ & 63)) - 1 : ~0ull;
    const uint8_t blast = (D_ & 7) ? (uint8_t)((1u << (D_ & 7)) - 1) : (uint8_t)0xff;
    M.fh.assign((size_t)WF + 1, 0);
    M.row.assign((size_t)WF + 1, 0);
    M.acc.assign((size_t)WF + 1, 0);
    M.hkey.assign(3 * WA + 1, 0);
    M.tkey.assign(3 * WA + 1, 0);
    M.lkey.assign(WA + 1, 0);
    M.fobs.assign((size_t)WO + 1, 0);
    M.sg.assign((size_t)O + 1, 0);
    M.lfu.assign((size_t)WF + (size_t)WO + 1, 0);
    std::vector<uint64_t> dw((size_t)WD + 1, 0), vv((size_t)WD + 1, 0);
    LawScratch LS;
    auto flips = [&](const int32_t* ids, size_t n, uint64_t* row) {
        std::fill(row, row + WF, 0ull);
        for (size_t j = 0; j < n; ++j)
            for (uint64_t t = loff_[(size_t)ids[j]]; t < loff_[(size_t)ids[j] + 1]; ++t) row[lent_[t].word] ^= lent_[t].mask;
    };
    auto hi_images = [&](const uint64_t* key, uint64_t* row) { k_hi_images(key, row); };
    auto strings = [&](const KEntry& E, const uint8_t* sg, uint64_t* row) { k_strings(E, sg, row); };
    if (!M.kf.built) k_fast_setup(M.kf);
    const KFast& KF = M.kf;
    const int rD = rho_ - rhoL_;
    // K's strings in beta' coordinates (T') and their kappa'' words (kappa' = kappa | delta << r in image coordinates)
    std::vector<uint64_t> kraw((size_t)std::max(KF.cw, 1));
    const GaugeSelect ksel(KF.S, words_for(F_.r + O_), KF.rhoX);
    auto ensure_xfast = [&](KEntry& E) {
        if (E.xfast_built) return;
        const size_t nx = E.xe.size(), nwK = words_for(F_.r);
        if (KF.cw) {   // KF.cw words per string (x * cw)
            const size_t C = (size_t)KF.cw;
            E.xT.assign(nx * C, 0);
            E.xK.assign(nx * C, 0);
            for (size_t x = 0; x < nx; ++x) {
                uint64_t* t = &E.xT[x * C];
                uint64_t* k = &E.xK[x * C];
                std::copy(E.xkap.data() + x * nwK, E.xkap.data() + x * nwK + std::min(nwK, C), k);
                for (int w = 0; w < WO; ++w) {
                    for (uint64_t y = E.ximg[x * (size_t)WF + (size_t)(WD + w)]; y; y &= y - 1) {
                        const uint64_t* e = &KF.ecW[(size_t)(w * 64 + __builtin_ctzll(y)) * C];
                        for (size_t q = 0; q < C; ++q) t[q] ^= e[q];
                    }
                    for (uint64_t y = E.xdl[x * (size_t)WO + (size_t)w]; y; y &= y - 1) {
                        const size_t b = (size_t)F_.r + (size_t)w * 64 + (size_t)__builtin_ctzll(y);
                        k[b >> 6] |= 1ull << (b & 63);
                    }
                }
                std::copy(k, k + C, kraw.begin());   // kappa' -> kappa'' (k_fast_rebase)
                ksel(t, C, kraw.data(), C, k, C);
            }
            E.xfast_built = true;
            return;
        }
        E.xT.assign(nx, 0);
        E.xK.assign(nx, 0);
        for (size_t x = 0; x < nx; ++x) {
            uint64_t t = 0, k = nwK ? E.xkap[x * nwK] : 0ull;
            for (int w = 0; w < WO; ++w) {
                for (uint64_t y = E.ximg[x * (size_t)WF + (size_t)(WD + w)]; y; y &= y - 1)
                    t ^= KF.ecoord[(size_t)(w * 64 + __builtin_ctzll(y))];
                for (uint64_t y = E.xdl[x * (size_t)WO + (size_t)w]; y; y &= y - 1)
                    k |= 1ull << (F_.r + w * 64 + __builtin_ctzll(y));
            }
            E.xT[x] = t;
            ksel(&t, 1, &k, 1, &E.xK[x], 1);   // kappa' -> kappa'' (k_fast_rebase)
        }
        E.xfast_built = true;
    };
    auto build_generic = [&](KEntry& E, const uint64_t* odd, XLaw& X) { k_law_generic(E, odd, X, LS); };
    // the same law in beta' coordinates (post_fast's construction with beta extended by KF's vectors): odd items =
    // the odd file axes (T_a, kappa_a, bhat_a) and the odd K strings (T', kappa', 0); col_j = XOR of the items' kappa'
    // with T bit j; W = ker N' by private-pivot elimination; the W phase |{items : T.sigma = 1}| / 2 + sum bhat
    // (T.sigma) on the kernel basis gives omega; q = beta'.omega
    auto build_fast = [&](KEntry& E, const uint64_t* odd, XLaw& X) {
        ensure_xfast(E);
        ++M.fast_laws;
        M.item_T.clear();
        M.item_K.clear();
        M.item_b.clear();
        for (size_t w = 0; w < WA; ++w)
            for (uint64_t y = odd[w]; y; y &= y - 1) {
                const size_t a = w * 64 + (size_t)__builtin_ctzll(y);
                M.item_T.push_back(Tc_[a]);
                M.item_K.push_back(KF.Kc1[a]);
                M.item_b.push_back(bb_[a]);
            }
        for (size_t x = 0; x < E.xe.size(); ++x)
            if (E.xe[x] & 1) {
                M.item_T.push_back(E.xT[x]);
                M.item_K.push_back(E.xK[x]);
                M.item_b.push_back(0);
            }
        const int R = KF.rhoX;
        const size_t m = M.item_T.size();
        uint64_t col[64], lv[64], llow[64], d0 = 0, d1 = 0, lin = 0;
        for (int j = 0; j < R; ++j) col[j] = 0;
        for (size_t i = 0; i < m; ++i) {
            const uint64_t t = M.item_T[i];
            for (uint64_t tt = t; tt; tt &= tt - 1) col[__builtin_ctzll(tt)] ^= M.item_K[i];
            d1 ^= d0 & t;
            d0 ^= t;
            lin ^= (0 - (uint64_t)M.item_b[i]) & t;
        }
        Law1 Lw;
        law_kernel1(col, R, R, d1, lin, Lw);
        const uint64_t* Kt = Lw.Kt;
        const uint64_t omega = Lw.omega;
        X.fast = true;
        X.q.assign((size_t)WF + 1, 0);
        xor_beta(X.q.data(), rho_ >= 64 ? omega : omega & ((1ull << rho_) - 1));
        for (int t = 0; t < KF.nn; ++t)
            if ((omega >> (rho_ + t)) & 1ull)
                for (int u = 0; u < WF; ++u) X.q[(size_t)u] ^= KF.nvI[(size_t)t * (size_t)WF + (size_t)u];
        int nl = 0;
        uint64_t pmask = 0;
        int pidx[64];
        for (int j = rD; j < R; ++j) {   // pivot-indexed (reduce_low1; no combination tracked)
            uint64_t v = Kt[j];
            for (uint64_t h; (h = v & pmask) != 0;) v ^= lv[pidx[__builtin_ctzll(h)]];
            if (!v) continue;
            lv[nl] = v;
            llow[nl] = v & (0 - v);
            pidx[__builtin_ctzll(v)] = nl;
            pmask |= llow[nl];
            ++nl;
        }
        X.nl = nl;
        X.fw.assign(Kt, Kt + R);
        X.fw.insert(X.fw.end(), lv, lv + nl);
        X.fw.insert(X.fw.end(), llow, llow + nl);
        X.kdim = (R - rD) - nl;
    };
    // build_fast in KF.cw words per vector (XFM fix round 1): the same construction (items, col_j, private-pivot
    // elimination, W phase on the kernel basis, q = beta'.omega, lv over the non-beta_D coordinates)
    std::vector<const uint64_t*> itT, itK;
    std::vector<uint8_t> itB;
    std::vector<uint64_t> wcol, wpv, wpc, wKt, wv;
    std::vector<int> wdep, wpcol, wpidx;
    auto build_fast_w = [&](KEntry& E, const uint64_t* odd, XLaw& X) {
        ensure_xfast(E);
        ++M.fast_laws;
        const int C = KF.cw, R = KF.rhoX;
        const size_t RC = (size_t)R * (size_t)C;
        itT.clear();
        itK.clear();
        itB.clear();
        for (size_t w = 0; w < WA; ++w)
            for (uint64_t y = odd[w]; y; y &= y - 1) {
                const size_t a = w * 64 + (size_t)__builtin_ctzll(y);
                itT.push_back(&KF.TcB[a * (size_t)C]);
                itK.push_back(&KF.KcB[a * (size_t)C]);
                itB.push_back(bb_[a]);
            }
        for (size_t x = 0; x < E.xe.size(); ++x)
            if (E.xe[x] & 1) {
                itT.push_back(&E.xT[x * (size_t)C]);
                itK.push_back(&E.xK[x * (size_t)C]);
                itB.push_back(0);
            }
        const size_t m = itT.size();
        wcol.assign(RC, 0);
        wKt.resize(RC);
        wpv.resize(RC);
        wpc.resize(RC);
        wv.assign((size_t)7 * C, 0);
        wdep.resize((size_t)R + 1);
        wpcol.resize((size_t)R + 1);
        wpidx.resize((size_t)65 * C);
        uint64_t *col = wcol.data(), *Kt = wKt.data();
        uint64_t *tc = wv.data(), *d0 = tc + C, *d1 = d0 + C, *lin = d1 + C, *tv = lin + C, *pmk = tv + C;
        uint64_t* om = pmk + C;
        for (size_t i = 0; i < m; ++i) {
            const uint64_t* t = itT[i];
            for (int u = 0; u < C; ++u)
                for (uint64_t tt = t[u]; tt; tt &= tt - 1) {
                    uint64_t* c = col + (size_t)(u * 64 + __builtin_ctzll(tt)) * C;
                    for (int q = 0; q < C; ++q) c[q] ^= itK[i][q];
                }
            const uint64_t bm = itB[i] ? ~0ull : 0ull;
            for (int q = 0; q < C; ++q) {
                d1[q] ^= d0[q] & t[q];
                d0[q] ^= t[q];
                lin[q] ^= bm & t[q];
            }
        }
        int npv = 0, nker = 0;
        law_kernelw<0>(C, R, false, col, d1, lin, tv, tc, pmk, wpv.data(), wpc.data(), wpidx.data(),
                       wpidx.data() + 64 * C, Kt, wdep.data(), wpcol.data(), npv, nker, om);
        X.fast = true;
        X.cw = C;
        X.q.assign((size_t)WF + 1, 0);
        k_xor_beta_x(KF, om, X.q.data());
        X.fw.assign(Kt, Kt + RC);
        X.lp.clear();
        int nl = 0;
        std::fill(pmk, pmk + C, 0ull);   // (law_kernelw's pivot mask and index are free now)
        int* pidx = wpidx.data();
        for (int j = rD; j < R; ++j) {   // pivot-indexed (reduce_low; no combination tracked)
            std::copy(Kt + (size_t)j * C, Kt + (size_t)(j + 1) * C, tc);
            reduce_low<0>(tc, nullptr, C, pmk, pidx, X.fw.data() + RC, nullptr);
            int p = -1;
            for (int u = 0; u < C && p < 0; ++u)
                if (tc[u]) p = u * 64 + __builtin_ctzll(tc[u]);
            if (p < 0) continue;
            X.fw.insert(X.fw.end(), tc, tc + C);
            X.lp.push_back(p);
            pidx[p] = nl;
            pmk[p >> 6] |= 1ull << (p & 63);
            ++nl;
        }
        X.nl = nl;
        X.kdim = (R - rD) - nl;
    };
    auto build_law = [&](KEntry& E, const uint64_t* odd, XLaw& X) {
        if (KF.ok && KF.cw) build_fast_w(E, odd, X);
        else if (KF.ok) build_fast(E, odd, X);
        else build_generic(E, odd, X);
    };
    // fast laws: the beta' coordinates of the detector part of v (WD words; destroyed): beta_D's (false when v_D is
    // outside span(beta_D))
    auto coords_D = [&](uint64_t* v, uint64_t& sD) {
        sD = 0;
        for (int j = 0; j < rD; ++j)
            if (bit_at(v, pD_[(size_t)j])) {
                for (int w = 0; w < WD; ++w) v[w] ^= bDd_[(size_t)j * WD + (size_t)w];
                sD |= 1ull << j;
            }
        return !any_word(v, (size_t)WD);
    };
    // (fast) v (WF words, internal layout; destroyed) in the law q + beta'.W^perp, v already holding q
    std::vector<uint64_t> ow((size_t)WO + 1, 0);
    std::vector<uint64_t> zw((size_t)2 * std::max(KF.cw, 1));
    // the cw-word forms (X.cw > 0): z <- the beta_D coordinates of v_D (destroyed); false when outside span(beta_D)
    auto coords_D_w = [&](uint64_t* v, uint64_t* z) {
        std::fill(z, z + KF.cw, 0ull);
        for (int j = 0; j < rD; ++j)
            if (bit_at(v, pD_[(size_t)j])) {
                for (int w = 0; w < WD; ++w) v[w] ^= bDd_[(size_t)j * WD + (size_t)w];
                z[j >> 6] |= 1ull << (j & 63);
            }
        return !any_word(v, (size_t)WD);
    };
    auto kt_sum = [&](const XLaw& X, const uint64_t* z, uint64_t* r) {   // r <- XOR_{j : z_j} Kt[j]
        const int C = X.cw;
        std::fill(r, r + C, 0ull);
        for (int u = 0; u < C; ++u)
            for (uint64_t y = z[u]; y; y &= y - 1) {
                const uint64_t* k = &X.fw[(size_t)(u * 64 + __builtin_ctzll(y)) * C];
                for (int q = 0; q < C; ++q) r[q] ^= k[q];
            }
    };
    auto fast_in_w = [&](const XLaw& X, uint64_t* v) {
        uint64_t* z = zw.data();
        uint64_t* r = z + X.cw;
        if (!coords_D_w(v, z)) return false;
        uint64_t* o = ow.data();
        std::copy(v + WD, v + WF, o);
        for (int j = 0; j < rD; ++j)
            if ((z[j >> 6] >> (j & 63)) & 1ull) {
                const uint64_t* b = betaO_.data() + (size_t)j * WO;
                for (int w = 0; w < WO; ++w) o[w] ^= b[w];
            }
        for (int i = 0; i < rhoL_; ++i)
            if (bit_at(o, pL_[(size_t)i])) {
                for (int w = 0; w < WO; ++w) o[w] ^= bLo_[(size_t)i * WO + (size_t)w];
                z[(rD + i) >> 6] |= 1ull << ((rD + i) & 63);
            }
        for (int t = 0; t < KF.nn; ++t)
            if (bit_at(o, KF.np[(size_t)t])) {
                for (int w = 0; w < WO; ++w) o[w] ^= KF.nv[(size_t)t * WO + (size_t)w];
                z[(rho_ + t) >> 6] |= 1ull << ((rho_ + t) & 63);
            }
        if (any_word(o, (size_t)WO)) return false;
        kt_sum(X, z, r);
        return !any_word(r, (size_t)X.cw);
    };
    auto fast_det_in_w = [&](const XLaw& X, uint64_t* v) {
        uint64_t* z = zw.data();
        uint64_t* r = z + X.cw;
        if (!coords_D_w(v, z)) return false;
        kt_sum(X, z, r);
        const int C = X.cw;
        const size_t RC = (size_t)KF.rhoX * (size_t)C;
        for (int q = 0; q < X.nl; ++q)
            if ((r[X.lp[(size_t)q] >> 6] >> (X.lp[(size_t)q] & 63)) & 1ull)
                for (int u = 0; u < C; ++u) r[u] ^= X.fw[RC + (size_t)q * C + (size_t)u];
        return !any_word(r, (size_t)C);
    };
    auto fast_in = [&](const XLaw& X, uint64_t* v) {
        if (X.cw) return fast_in_w(X, v);
        uint64_t z;
        if (!coords_D(v, z)) return false;
        uint64_t* o = ow.data();
        std::copy(v + WD, v + WF, o);
        for (uint64_t y = z; y; y &= y - 1) {
            const uint64_t* b = betaO_.data() + (size_t)__builtin_ctzll(y) * WO;
            for (int w = 0; w < WO; ++w) o[w] ^= b[w];
        }
        for (int i = 0; i < rhoL_; ++i)
            if (bit_at(o, pL_[(size_t)i])) {
                for (int w = 0; w < WO; ++w) o[w] ^= bLo_[(size_t)i * WO + (size_t)w];
                z |= 1ull << (rD + i);
            }
        for (int t = 0; t < KF.nn; ++t)
            if (bit_at(o, KF.np[(size_t)t])) {
                for (int w = 0; w < WO; ++w) o[w] ^= KF.nv[(size_t)t * WO + (size_t)w];
                z |= 1ull << (rho_ + t);
            }
        if (any_word(o, (size_t)WO)) return false;
        uint64_t r = 0;
        const uint64_t* Kt = X.Kt();
        for (uint64_t y = z; y; y &= y - 1) r ^= Kt[__builtin_ctzll(y)];
        return r == 0;
    };
    // (fast) v_D (WD words; destroyed) in the law's detector support (v_D holding q_D)
    auto fast_det_in = [&](const XLaw& X, uint64_t* v) {
        if (X.cw) return fast_det_in_w(X, v);
        uint64_t sD;
        if (!coords_D(v, sD)) return false;
        uint64_t r = 0;
        const uint64_t *Kt = X.Kt(), *lv = X.lv(KF.rhoX), *llow = X.llow(KF.rhoX);
        for (uint64_t y = sD; y; y &= y - 1) r ^= Kt[__builtin_ctzll(y)];
        for (int q = 0; q < X.nl; ++q) r ^= lv[q] & (0 - (uint64_t)((r & llow[q]) != 0));
        return r == 0;
    };
    // v (WD words) in span(eD)? (fully reduced: one pass; v destroyed)
    auto in_det_span = [&](uint64_t* v, const std::vector<uint64_t>& eD, const std::vector<int>& eP) {
        for (size_t i = 0; i < eP.size(); ++i)
            if (bit_at(v, eP[i]))
                for (int w = 0; w < WD; ++w) v[w] ^= eD[i * (size_t)WD + (size_t)w];
        return !any_word(v, (size_t)WD);
    };
    auto clear_all = [&]() {
        M.index.clear();
        M.entries.clear();
        M.lawidx.clear();
        M.laws.clear();
        ++M.clears;
    };
    // the law of [h, K] (T = h): the entry's own law, built on first use (M.hkey = h's key)
    auto ensure_same = [&](KEntry& E) {
        if (E.same_built) return;
        std::fill(M.acc.begin(), M.acc.end(), 0ull);
        hi_images(M.hkey.data(), M.acc.data());
        for (int k = 0; k < O; ++k) M.sg[(size_t)k] = E.rows[(size_t)k].ts;
        strings(E, M.sg.data(), M.acc.data());
        build_law(E, M.hkey.data(), E.same);
        for (int u = 0; u < WF; ++u) E.same.q[(size_t)u] ^= M.acc[(size_t)u];
        E.same_built = true;
    };
    uint64_t n_out = 0, n_lossy = 0, n_coin = 0;
    for (size_t s = 0; s < shots; ++s) {
        const uint8_t* db = dets + s * nbD;
        if (!cx && hoff[s + 1] == hoff[s] && foff[s + 1] == foff[s]) {
            // nothing fired, nothing hypothesised: K = 1, both laws are the point 0, pass = [d = 0]
            uint8_t acc = 0;
            for (size_t b = 0; b + 1 < nbD; ++b) acc |= db[b];
            if (nbD) acc |= db[nbD - 1] & blast;
            out[s] = acc ? 0.0 : 1.0;
            n_out += acc != 0;
            continue;
        }
        size_t nh;
        const int32_t* hv = canon_view(hids + hoff[s], hoff[s + 1] - hoff[s], M.hbuf, nh);
        if (nbD) {
            std::memcpy(dw.data(), db, nbD);
            dw[(size_t)WD - 1] &= dlast;
        }
        // the memo entry of h's composite (content key)
        k_key(hv, nh, M.hkey.data());
        M.hkey.resize(3 * WA);
        int id;
        auto it = M.index.find(M.hkey);
        if (it == M.index.end()) {
            if (M.entries.size() >= cap_) clear_all();
            KEntry E;
            k_build(M.hkey.data(), E, M.ks);   // may throw (internal checks): the memo is unchanged then
            id = (int)M.entries.size();
            M.entries.push_back(std::move(E));
            M.index.emplace(M.hkey, id);
        } else {
            id = it->second;
        }
        KEntry& E = M.entries[(size_t)id];
        flips(hv, nh, M.fh.data());
        n_lossy += E.lossy;
        k_frames(E, M.fh.data(), dw.data(), M.fobs.data());
        if (E.lossy) {   // ERRATA E4: the lossy observables' frame c from law([h, K]) at d
            if (!E.lf_built) {
                ensure_same(E);
                k_lossy_setup(E, E.same, KF);
            }
            k_lossy_frames(E, M.fh.data(), dw.data(), M.fobs.data(), M.lfu.data());
        }
        if (cx)
            for (int k = 0; k < O; ++k) {
                const KEntry::Row& R = E.rows[(size_t)k];
                cx->lossy.push_back(R.lossy);
                cx->y.push_back(R.y);
                cx->frame.push_back((uint8_t)bit_at(M.fobs.data(), k));
                cx->axes.insert(cx->axes.end(), E.S.begin() + R.s0, E.S.begin() + R.s1);
                cx->piv.insert(cx->piv.end(), E.P.begin() + R.p0, E.P.begin() + R.p1);
                cx->axes_off.push_back((int64_t)cx->axes.size());
                cx->piv_off.push_back((int64_t)cx->piv.size());
            }
        if (!out) continue;
        // the corrected shot: T's lines, then K = Z[f_K] K_diag; its law at (d, c = 0)
        size_t nt;
        const int32_t* tv = canon_view(fids + foff[s], foff[s + 1] - foff[s], M.tbuf, nt);
        uint64_t* row = M.row.data();
        const bool same_list = nt == nh && (tv == hv || std::equal(tv, tv + nt, hv));
        if (same_list) std::copy(M.fh.begin(), M.fh.begin() + WF, row);
        else flips(tv, nt, row);
        if (!same_list) k_key(tv, nt, M.tkey.data());
        const XLaw* X;
        if (same_list || std::equal(M.tkey.begin(), M.tkey.begin() + (long)(3 * WA), M.hkey.begin())) {
            // T's composite is h's: the entry's own law; law(h)'s detector support is its (K commutes with every
            // detector parity's X and has no detector Z)
            ensure_same(E);
            X = &E.same;
            for (int w = 0; w < WD; ++w) vv[(size_t)w] = dw[(size_t)w] ^ M.fh[(size_t)w] ^ X->q[(size_t)w];
            n_out += X->fast ? !fast_det_in(*X, vv.data()) : !in_det_span(vv.data(), X->eD, X->eP);
        } else {
            if (!E.sup_built) k_support(E);
            for (int w = 0; w < WD; ++w) vv[(size_t)w] = dw[(size_t)w] ^ M.fh[(size_t)w] ^ E.gD[(size_t)w];
            n_out += !in_det_span(vv.data(), E.eD, E.eP);
            const uint64_t* tk = M.tkey.data();
            hi_images(tk, row);
            for (int k = 0; k < O; ++k) {   // tau_T(S_k): the sign of K's strings (step 6)
                const KEntry::Row& R = E.rows[(size_t)k];
                int p = 0;
                for (uint32_t t = R.w0; t < R.w1; t += 2) p ^= __builtin_parityll(tk[2 * WA + E.Sw[t]] & E.Sw[t + 1]);
                M.sg[(size_t)k] = (uint8_t)p;
            }
            strings(E, M.sg.data(), row);
            M.lkey[0] = (uint64_t)id;
            std::copy(tk, tk + WA, M.lkey.begin() + 1);
            auto li = M.lawidx.find(M.lkey);
            int lid;
            if (li == M.lawidx.end()) {
                if (M.laws.size() >= cap_) {
                    M.lawidx.clear();
                    M.laws.clear();
                    ++M.law_clears;
                }
                XLaw L;
                build_law(E, tk, L);
                lid = (int)M.laws.size();
                M.laws.push_back(std::move(L));
                M.lawidx.emplace(M.lkey, lid);
            } else {
                lid = li->second;
            }
            X = &M.laws[(size_t)lid];
        }
        for (int u = 0; u < WF; ++u) row[u] ^= X->q[(size_t)u];
        for (int w = 0; w < WO; ++w) row[WD + w] ^= M.fobs[(size_t)w];
        for (int w = 0; w < WD; ++w) row[w] ^= dw[(size_t)w];
        bool in;
        if (X->fast) {
            in = fast_in(*X, row);
        } else {
            const size_t rk = X->piv.size();   // (d, 0) - offset in span(coins)? (fully reduced: one pass)
            for (size_t i = 0; i < rk; ++i)
                if (bit_at(row, X->piv[i]))
                    for (int u = 0; u < WF; ++u) row[u] ^= X->coins[i * (size_t)WF + (size_t)u];
            in = !any_word(row, (size_t)WF);
        }
        out[s] = in ? (X->kdim ? std::ldexp(1.0, -X->kdim) : 1.0) : 0.0;
        n_coin += X->kdim > 0;
    }
    if (!out) return;
    lines_shots_ += shots;
    lines_outside_ += n_out;
    lines_lossy_ += n_lossy;
    lines_coin_after_ += n_coin;
}

void XdemSampler::k_optimal(const int32_t* h, size_t nh, const uint64_t* s, std::vector<KObs>& out) const {
    for (size_t j = 0; j < nh; ++j)
        if (h[j] < 0 || (size_t)h[j] >= L_)
            throw std::out_of_range("XdemSampler.k_optimal: line id " + std::to_string(h[j]) +
                                    " is outside [0, num_lines = " + std::to_string(L_) + ")");
    std::vector<int32_t> buf;
    size_t n;
    const int32_t* hv = canon_view(h, (int64_t)nh, buf, n);
    std::vector<uint64_t> key(3 * (size_t)WA_ + 1, 0);
    k_key(hv, n, key.data());
    KEntry E;
    KScratch X;
    k_build(key.data(), E, X);
    std::vector<uint64_t> fh((size_t)WF_ + 1, 0), d((size_t)WD_ + 1, 0), fobs((size_t)WO_ + 1, 0);
    for (size_t j = 0; j < n; ++j)
        for (uint64_t t = loff_[(size_t)hv[j]]; t < loff_[(size_t)hv[j] + 1]; ++t) fh[lent_[t].word] ^= lent_[t].mask;
    std::copy(s, s + WD_, d.begin());
    if (WD_ && (D_ & 63)) d[(size_t)WD_ - 1] &= (1ull << (D_ & 63)) - 1;
    k_frames(E, fh.data(), d.data(), fobs.data());
    if (E.lossy) {   // ERRATA E4: law([h, K]) by the generic law (score_optimal's own path is the fast one when set up)
        XLaw Xh;
        LawScratch LS;
        std::vector<uint64_t> acc((size_t)WF_ + 1, 0);
        std::vector<uint8_t> sg((size_t)O_ + 1, 0);
        k_hi_images(key.data(), acc.data());
        for (int k = 0; k < O_; ++k) sg[(size_t)k] = E.rows[(size_t)k].ts;
        k_strings(E, sg.data(), acc.data());
        k_law_generic(E, key.data(), Xh, LS);
        for (int u = 0; u < WF_; ++u) Xh.q[(size_t)u] ^= acc[(size_t)u];
        k_lossy_setup(E, Xh, KFast());
        std::vector<uint64_t> u((size_t)WF_ + (size_t)WO_ + 1, 0);
        k_lossy_frames(E, fh.data(), d.data(), fobs.data(), u.data());
    }
    out.assign((size_t)O_, KObs());
    for (int k = 0; k < O_; ++k) {
        const KEntry::Row& R = E.rows[(size_t)k];
        KObs& o = out[(size_t)k];
        o.lossy = R.lossy != 0;
        o.P.assign(E.P.begin() + R.p0, E.P.begin() + R.p1);
        o.S.assign(E.S.begin() + R.s0, E.S.begin() + R.s1);
        o.y = R.y;
        o.frame = (uint8_t)bit_at(fobs.data(), k);
    }
}

void XdemSampler::correction(size_t shots, const uint8_t* dets, const int64_t* hoff, const int32_t* hids, size_t nhids,
                             Correction corr, XdemCorrectionOut& out) {
    if (corr == Correction::None) throw std::invalid_argument("correction(): nothing to export for \"none\"");
    std::lock_guard<std::mutex> lk(mu_);
    if (corr == Correction::R2b && !r2b_.first)
        throw XdemFileError("correction(correction=\"r2b\"): the file is not in canonical single-stage form at t_c: " +
                            r2b_.second + "; pass correction=\"optimal\" or \"none\"");
    check_lines_csr("XdemSampler.correction", "hypothesis", shots, hoff, hids, nhids, L_);
    out = XdemCorrectionOut();
    if (corr == Correction::Optimal) {
        out.axes_off.assign(1, 0);
        out.piv_off.assign(1, 0);
        score_optimal(shots, dets, nullptr, nullptr, hoff, hids, nullptr, &out);
        return;
    }
    // "r2b": the guard's action and c from score_lines' own code (neither reads the fired lines: T = h here), then
    // eta^> and tau_K per shot (SPEC §4.4 step 1)
    std::vector<double> pass(shots);
    std::vector<int8_t> act(shots);
    out.c.assign(shots * (size_t)O_, 0);
    XdemLinesOptions opt;
    opt.corr = Correction::R2b;
    score_lines_locked(shots, dets, hoff, hids, nhids, hoff, hids, nhids, pass.data(), opt, act.data(), out.c.data(),
                       false);
    out.applied.resize(shots);
    out.late_off.assign(1, 0);
    out.tau_off.assign(1, 0);
    std::vector<int32_t> buf;
    std::vector<std::pair<int, int>> eta;
    std::vector<uint64_t> tauk;
    for (size_t s = 0; s < shots; ++s) {
        out.applied[s] = act[s] == kLinesApplied;
        size_t nh;
        const int32_t* hv = canon_view(hids + hoff[s], hoff[s + 1] - hoff[s], buf, nh);
        const size_t nlt = (size_t)(std::lower_bound(hv, hv + nh, F_.tc, [](int32_t a, int64_t b) {
                                        return (int64_t)a < b;
                                    }) - hv);
        r2b_k(hv, nlt, eta, &tauk);
        for (const auto& [a, e] : eta) {
            out.late_axis.push_back(a);
            out.late_eta.push_back((int8_t)e);
        }
        out.late_off.push_back((int64_t)out.late_axis.size());
        for (size_t w = 0; w < tauk.size(); ++w)
            for (uint64_t y = tauk[w]; y; y &= y - 1) out.tau_axis.push_back((int32_t)(w * 64 + __builtin_ctzll(y)));
        out.tau_off.push_back((int64_t)out.tau_axis.size());
    }
}

XdemCounters XdemSampler::counters() const {
    std::lock_guard<std::mutex> lk(mu_);
    XdemCounters c;
    for (const auto& T : ts_) {
        c.shots += T->shots;
        c.ext_shots += T->ext_shots;
        c.coin_shots += T->coin_shots;
        c.cache_hits += T->hits;
        c.cache_misses += T->misses;
        c.cache_clears += T->clears;
        c.cache_entries += T->entries.size();
    }
    c.cache_cap = cap_;
    c.lines_shots = lines_shots_;
    c.lines_applied = lines_applied_;
    c.lines_withheld = lines_withheld_;
    c.lines_coin_after = lines_coin_after_;
    c.lines_outside = lines_outside_;
    c.lines_lossy = lines_lossy_;
    c.kmemo_entries = km_ ? km_->entries.size() : 0;
    c.kmemo_clears = km_ ? km_->clears : 0;
    c.kmemo_law_clears = km_ ? km_->law_clears : 0;
    c.kfast_laws = km_ ? km_->fast_laws : 0;
    c.num_cspaces = cspaces_.size();
    c.thread_slots = ts_.size();
    return c;
}

void score_plain(size_t shots, int O, const uint8_t* obs, const uint8_t* flips, double* out) {
    const size_t nbO = (size_t)(O + 7) / 8;
    for (size_t s = 0; s < shots; ++s) {
        bool ok = true;
        const uint8_t* f = flips + s * (size_t)O;
        for (int o = 0; o < O; ++o) {
            if (f[o] > 1) throw std::invalid_argument("score_plain: flips must be 0 or 1");
            const int v = (obs[s * nbO + (size_t)(o >> 3)] >> (o & 7)) & 1;
            ok &= v == (int)f[o];
        }
        out[s] = ok ? 1.0 : 0.0;
    }
}

}  // namespace qeccore::xdem
