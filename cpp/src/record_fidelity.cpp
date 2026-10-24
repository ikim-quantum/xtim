// record_fidelity — see the header and the design note (scratchpad fid/design.md) for the
// derivation. Additive translation unit: no sampling path includes or calls it.

#include "qeccore/record_fidelity.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "qeccore/clifford_tableau.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/rf_pauli_expansion.hpp"
#include "qeccore/twirl_kernel.hpp"

namespace qeccore {

namespace {

inline int parw(const uint64_t* a, const uint64_t* b, int W) {
    int p = 0;
    for (int w = 0; w < W; ++w) p += __builtin_popcountll(a[w] & b[w]);
    return p & 1;
}

// Symplectic product bit <P, Q> for P given as raw words.
inline int sympl(const uint64_t* px, const uint64_t* pz, const Pauli& Q, int W) {
    int p = 0;
    for (int w = 0; w < W; ++w)
        p += __builtin_popcountll((px[w] & Q.z[(size_t)w]) ^ (pz[w] & Q.x[(size_t)w]));
    return p & 1;
}

inline Pauli with_sign(Pauli P, int neg) {
    if (neg) P.phase = (P.phase + 2) & 3;
    return P;
}

// Hermitian check for an exact-phase Pauli i^ph X^x Z^z: (i^ph X^x Z^z)^dag = (-i)^ph Z^z X^x
// = (-i)^ph (-1)^{x·z} X^x Z^z, equal iff i^{2ph} = (-1)^{x·z}, i.e. ph ≡ x·z (mod 2).
inline bool hermitian(const Pauli& P) { return ((P.phase ^ P.xz_overlap()) & 1) == 0; }

inline bool is_identity(const Pauli& P) {
    for (uint64_t v : P.x) if (v) return false;
    for (uint64_t v : P.z) if (v) return false;
    return true;
}

static const std::complex<double> kIPow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

// A signed permutation on the dense extended register: |idx> -> c·(-1)^{popc(idx&zm)} |idx^xm>.
struct FOp {
    uint64_t xm = 0, zm = 0;
    std::complex<double> c{1, 0};
};

inline double sgn(uint64_t v) { return (__builtin_popcountll(v) & 1) ? -1.0 : 1.0; }

// v <- (v + Q v)/2  (Q a commuting Hermitian involution on the register)
void apply_projector(std::vector<std::complex<double>>& v, const FOp& q) {
    const size_t N = v.size();
    if (q.xm == 0) {
        for (size_t i = 0; i < N; ++i)
            v[i] *= 0.5 * (1.0 + q.c * sgn(i & q.zm));
        return;
    }
    const uint64_t hb = 1ull << (63 - __builtin_clzll(q.xm));   // a bit set in xm: pair leader
    for (size_t i = 0; i < N; ++i) {
        if (i & hb) continue;
        const size_t j = i ^ q.xm;
        const std::complex<double> a = v[i], b = v[j];
        // (Qv)[j] = c·s(i)·v[i], (Qv)[i] = c·s(j)·v[j]
        v[i] = 0.5 * (a + q.c * sgn(j & q.zm) * b);
        v[j] = 0.5 * (b + q.c * sgn(i & q.zm) * a);
    }
}

// little-endian uint64 bit vectors (no index mask is narrower than the vector it indexes)
namespace gs {
inline bool getbit(const uint64_t* v, int b) { return (v[b >> 6] >> (b & 63)) & 1ULL; }
inline void flipbit(uint64_t* v, int b) { v[b >> 6] ^= 1ULL << (b & 63); }
}  // namespace gs

// RF-ref strict contract: GF(2) elimination of Pauli bit rows (x | z, 2W words, phases ignored) on a column mask; rows
// kept fully reduced on their pivots. `zero`: the inputs (as products) whose masked part reduces to 0, in insertion
// order; `pivot`: per add, whether the row was new on the mask.
struct MaskElim {
    int W = 0;
    std::vector<uint64_t> m;                          // 2W-word column mask
    struct Row { std::vector<uint64_t> v, f; int piv; };
    std::vector<Row> rows;
    std::vector<std::vector<uint64_t>> zero;          // full rows (2W words) reduced to 0 on the mask
    MaskElim(int W_, std::vector<uint64_t> m_) : W(W_), m(std::move(m_)) {}
    bool add(std::vector<uint64_t> f) {
        std::vector<uint64_t> v((size_t)2 * W);
        for (int w = 0; w < 2 * W; ++w) v[(size_t)w] = f[(size_t)w] & m[(size_t)w];
        for (const Row& r : rows)
            if ((v[(size_t)(r.piv >> 6)] >> (r.piv & 63)) & 1ULL)
                for (int w = 0; w < 2 * W; ++w) { v[(size_t)w] ^= r.v[(size_t)w]; f[(size_t)w] ^= r.f[(size_t)w]; }
        int piv = -1;
        for (int w = 0; w < 2 * W && piv < 0; ++w) if (v[(size_t)w]) piv = w * 64 + __builtin_ctzll(v[(size_t)w]);
        if (piv < 0) { zero.push_back(std::move(f)); return false; }
        for (Row& r : rows)
            if ((r.v[(size_t)(piv >> 6)] >> (piv & 63)) & 1ULL)
                for (int w = 0; w < 2 * W; ++w) { r.v[(size_t)w] ^= v[(size_t)w]; r.f[(size_t)w] ^= f[(size_t)w]; }
        rows.push_back(Row{std::move(v), std::move(f), piv});
        return true;
    }
};
inline std::vector<uint64_t> bits_of(const Pauli& P, int W) {
    std::vector<uint64_t> f((size_t)2 * W);
    for (int w = 0; w < W; ++w) { f[(size_t)w] = P.x[(size_t)w]; f[(size_t)(W + w)] = P.z[(size_t)w]; }
    return f;
}
inline Pauli pauli_of(const std::vector<uint64_t>& f, int n, int W) {   // the Hermitian i^{x·z} X^x Z^z
    Pauli P(n);
    for (int w = 0; w < W; ++w) { P.x[(size_t)w] = f[(size_t)w]; P.z[(size_t)w] = f[(size_t)(W + w)]; }
    P.phase = P.xz_overlap() & 3;
    return P;
}
// GF(2) rank of a square symmetric bit matrix (rows of WW words)
inline int gf2_rank(std::vector<std::vector<uint64_t>> M) {
    int r = 0;
    const int nr = (int)M.size();
    const int nc = nr ? (int)M[0].size() * 64 : 0;
    for (int c = 0; c < nc && r < nr; ++c) {
        int p = -1;
        for (int i = r; i < nr; ++i) if ((M[(size_t)i][(size_t)(c >> 6)] >> (c & 63)) & 1ULL) { p = i; break; }
        if (p < 0) continue;
        std::swap(M[(size_t)p], M[(size_t)r]);
        for (int i = 0; i < nr; ++i)
            if (i != r && ((M[(size_t)i][(size_t)(c >> 6)] >> (c & 63)) & 1ULL))
                for (size_t w = 0; w < M[(size_t)i].size(); ++w) M[(size_t)i][w] ^= M[(size_t)r][w];
        ++r;
    }
    return r;
}

}  // namespace

struct RecordFidelity::Impl {
    int n = 0, W = 0, k = 0;
    size_t max_bytes = 0;               // caller's byte budget for what the evaluator holds (0 = none)
    CliffordTableau U{0};
    std::vector<uint64_t> eps;          // n-bit reference signs
    std::vector<uint64_t> freemask;     // n-bit: free rows
    std::vector<int> freepos;           // row -> free index (or -1)
    std::vector<std::complex<double>> b;   // dense bare amplitudes over 2^k
    double bnorm2 = 0.0;
    RecordFidelityStats st;

    // ── S8 RF1: the logical expectation λ(u, v) = ⟨b| Z^v X^u |b⟩ on the free register, ONE formula:
    //     λ(u, v) = Σ_{y ∈ supp b, ascending, b_{y⊕u} ≠ 0} conj(b_y) · b_{y⊕u} · (−1)^{v·y},
    // a structural +0.0 when no support pair exists. The dense table (4^k entries, a not-computed bitmap) is a lazy
    // memo of exactly this function while it fits the λ budget (default 64 MiB) and the evaluator's byte budget;
    // otherwise the exact-key sparse memo is. Both hold the formula's bytes, so a value never depends on the path.
    std::vector<uint64_t> bsupp;                         // ascending support of b
    std::vector<std::complex<double>> lam_tab;           // dense memo (lam_mode 1): row u at [u << k, (u + 1) << k)
    std::vector<uint64_t> lam_done;                      //   its filled ROWS (one bit per u)
    std::vector<double> lam_re_, lam_im_;                //   row-fill scratch
    struct LamKey {
        uint64_t u, v;
        bool operator==(const LamKey& o) const { return u == o.u && v == o.v; }
    };
    struct LamHash {
        size_t operator()(const LamKey& K) const { return std::hash<uint64_t>()(K.u * 0x9E3779B97F4A7C15ULL ^ K.v); }
    };
    std::unordered_map<LamKey, std::complex<double>, LamHash> lam_sparse;   // sparse memo (lam_mode 2)
    size_t lam_budget = size_t(64) << 20;                // test lever: the dense table's byte cap
    int lam_mode = 0;                                    // 0 = undecided, 1 = dense, 2 = sparse
    // conj(b_y) · b_{y⊕u} with explicit fused multiply-adds: the bytes never depend on a compiler's contraction choice
    static inline void pair_prod(const std::complex<double>& by, const std::complex<double>& bu, double& tr, double& ti) {
        tr = std::fma(by.real(), bu.real(), by.imag() * bu.imag());
        ti = std::fma(by.real(), bu.imag(), -(by.imag() * bu.real()));
    }
    std::complex<double> lambda_direct(uint64_t u, uint64_t v) const {
        double re = 0.0, im = 0.0;
#ifdef XTIM_MUTATE_RF_LAMZ
        // mutation build only: the 4^k table by Walsh–Hadamard passes over f_u(y) = conj(b_y) b_{y⊕u} (FD §3.1's
        // first design; rounds differently and leaves interference noise where the formula is exact)
        {
            const size_t K = b.size();
            std::vector<std::complex<double>> f(K);
            for (size_t y = 0; y < K; ++y) f[y] = std::conj(b[y]) * b[y ^ (size_t)u];
            for (size_t h = 1; h < K; h <<= 1)
                for (size_t i = 0; i < K; i += 2 * h)
                    for (size_t j = i; j < i + h; ++j) { const auto x = f[j], z = f[j + h]; f[j] = x + z; f[j + h] = x - z; }
            return f[(size_t)v];
        }
#endif
        for (uint64_t y : bsupp) {
            const std::complex<double>& bu = b[(size_t)(y ^ u)];
            if (bu == 0.0) continue;
            double tr, ti;
            pair_prod(b[(size_t)y], bu, tr, ti);
#ifdef XTIM_MUTATE_RF_LAM
            if (__builtin_popcountll(v & (y ^ u)) & 1) { tr = -tr; ti = -ti; }   // mutation: (−1)^{v·(y⊕u)}
#else
            if (__builtin_popcountll(v & y) & 1) { tr = -tr; ti = -ti; }
#endif
            re += tr;
            im += ti;
        }
        return {re, im};
    }
    // a dense row: λ(u, v) for every v at once — per entry exactly the formula's operations in its order (y ascending,
    // re += (−1)^{v·y}·tr with (−1)^{v·y}·tr exact), many entries in parallel: the terms (tr, ti) per y once, then per
    // 8 blocks of kLane consecutive v register accumulators over y, (−1)^{v·y} = (−1)^{v_hi·y_hi} (−1)^{v_lo·y_lo} (the
    // block's sign an exact negation, the lane signs folded into the stored terms); small k: the plain loop
#ifdef __AVX512F__
    static constexpr int kLane = 8;                                     // the vector width (bytes are width-independent)
#else
    static constexpr int kLane = 4;
#endif
    typedef double vNd __attribute__((vector_size(8 * kLane)));         // element-wise IEEE (GCC / Clang vector extension)
    std::vector<vNd> lam_terms_;
    std::vector<uint64_t> lam_ys_;
    void lambda_row(uint64_t u) {
        const size_t K = (size_t)1 << k;
        constexpr int LB = kLane == 8 ? 3 : 2;                          // log2 lanes
        lam_re_.assign(K, 0.0);
        lam_im_.assign(K, 0.0);
        double* __restrict re = lam_re_.data();
        double* __restrict im = lam_im_.data();
        lam_ys_.clear();
        lam_terms_.clear();
        const bool blocked = k >= LB + 3;                               // 8 blocks of kLane lanes at least
        for (uint64_t y : bsupp) {
            const std::complex<double>& bu = b[(size_t)(y ^ u)];
            if (bu == 0.0) continue;
            double tr, ti;
            pair_prod(b[(size_t)y], bu, tr, ti);
#ifdef XTIM_MUTATE_RF_LAM
            const uint64_t ys = y ^ u;                                  // mutation: (−1)^{v·(y⊕u)}
#else
            const uint64_t ys = y;
#endif
            if (!blocked) {
                for (size_t v = 0; v < K; ++v) {
                    const double t = (__builtin_popcountll(v & ys) & 1) ? -1.0 : 1.0;
                    re[v] += t * tr;
                    im[v] += t * ti;
                }
                continue;
            }
            vNd a, c;
            for (int l = 0; l < kLane; ++l) {
                const double t = (__builtin_popcount((unsigned)(ys & (kLane - 1)) & (unsigned)l) & 1) ? -1.0 : 1.0;
                a[l] = t * tr;
                c[l] = t * ti;
            }
            lam_ys_.push_back(ys >> LB);
            lam_terms_.push_back(a);
            lam_terms_.push_back(c);
        }
        if (blocked) {
            const size_t B = K >> LB, ny = lam_ys_.size();
            const vNd* T = lam_terms_.data();
            const uint64_t* yh = lam_ys_.data();
            for (size_t b0 = 0; b0 < B; b0 += 8) {
                vNd r[8] = {}, i[8] = {};
                for (size_t t = 0; t < ny; ++t) {
                    const vNd a = T[2 * t], c = T[2 * t + 1];
                    for (int j = 0; j < 8; ++j) {
                        const bool neg = __builtin_popcountll((b0 + (size_t)j) & yh[t]) & 1;
                        r[j] += neg ? -a : a;
                        i[j] += neg ? -c : c;
                    }
                }
                for (int j = 0; j < 8; ++j) {
                    std::memcpy(re + (b0 + (size_t)j) * kLane, &r[j], sizeof(vNd));
                    std::memcpy(im + (b0 + (size_t)j) * kLane, &i[j], sizeof(vNd));
                }
            }
        }
        std::complex<double>* row = &lam_tab[(size_t)u << k];
        for (size_t v = 0; v < K; ++v) row[v] = {re[v], im[v]};
        lam_done[u >> 6] |= 1ULL << (u & 63);
    }
    // before a run of lookups: the dense row filled, the entry prefetched (no effect on any value)
    void lambda_touch(uint64_t u, uint64_t v) {
        if (lam_mode == 0) (void)lambda(u, v);
        if (lam_mode != 1) return;
        if (!((lam_done[u >> 6] >> (u & 63)) & 1ULL)) lambda_row(u);
        __builtin_prefetch(&lam_tab[((size_t)u << k) | (size_t)v]);
    }
    std::complex<double> lambda(uint64_t u, uint64_t v) {
        if (lam_mode == 0) {
            const size_t ent = k <= 30 ? (size_t)1 << (2 * k) : 0, tb = ent * 16 + ((size_t)1 << k) / 8;
            const size_t reg = 16 * b.size();
            if (ent && tb <= lam_budget && (!max_bytes || reg + held_memo + held_l3 + tb <= max_bytes)) {
                lam_tab.assign(ent, 0.0);
                lam_done.assign((((size_t)1 << k) + 63) / 64, 0);
                lam_mode = 1;
                charge(held_l3, tb, "the logical expectation table");
            } else {
                lam_mode = 2;
            }
        }
        if (lam_mode == 1) {
            if (!((lam_done[u >> 6] >> (u & 63)) & 1ULL)) lambda_row(u);
            return lam_tab[((size_t)u << k) | (size_t)v];
        }
        const LamKey K{u, v};
        auto it = lam_sparse.find(K);
        if (it != lam_sparse.end()) return it->second;
#ifdef XTIM_MUTATE_RF_LAMFB
        // mutation build only: the fallback sums the support in DESCENDING order (another formula's bytes)
        std::complex<double> r = 0.0;
        for (size_t i = bsupp.size(); i-- > 0;) {
            const uint64_t y = bsupp[i];
            if (b[(size_t)(y ^ u)] == 0.0) continue;
            std::complex<double> t = std::conj(b[(size_t)y]) * b[(size_t)(y ^ u)];
            if (__builtin_popcountll(v & y) & 1) t = -t;
            r += t;
        }
#else
        const std::complex<double> r = lambda_direct(u, v);
#endif
        lam_sparse.emplace(K, r);
        charge(held_l3, sizeof(LamKey) + 16 + 32, "a logical expectation");
        return r;
    }

    std::unordered_map<std::string, double> memo;
    std::string key_scratch;             // per-shot memo key / sign scratch (reused)
    Pauli p_scratch;
    std::vector<uint8_t> td_scratch;

    Pauli img(const Pauli& Q) const { return U.conjugate(Q); }

    // fixed part of a frame image's X support (free coordinates cleared)
    std::vector<uint64_t> fixed_part(const Pauli& Q) const {
        std::vector<uint64_t> f(Q.x.begin(), Q.x.end());
        for (int w = 0; w < W; ++w) f[(size_t)w] &= ~freemask[(size_t)w];
        return f;
    }

    // ── GF(2) elimination of commuting signed frame Paulis by their FIXED parts ──
    struct Row { std::vector<uint64_t> f; int piv; Pauli op; };
    struct Elim {
        std::vector<Row> rows;   // pivots (independent fixed parts), in insertion order
        std::vector<Pauli> kernel;
    };
    void reduce(const std::vector<Row>& rows, std::vector<uint64_t>& f, Pauli& op) const {
        for (const Row& r : rows)
            if ((f[(size_t)(r.piv >> 6)] >> (r.piv & 63)) & 1) {
                for (int w = 0; w < W; ++w) f[(size_t)w] ^= r.f[(size_t)w];
                op = Pauli::multiply(op, r.op);
            }
    }
    void insert(Elim& E, const Pauli& op0) const {
        std::vector<uint64_t> f = fixed_part(op0);
        Pauli op = op0;
        reduce(E.rows, f, op);
        int piv = -1;
        for (int w = 0; w < W && piv < 0; ++w)
            if (f[(size_t)w]) piv = w * 64 + __builtin_ctzll(f[(size_t)w]);
        if (piv < 0) { E.kernel.push_back(std::move(op)); return; }
        // keep rows fully reduced on their pivots (so reduce() order is irrelevant)
        for (Row& r : E.rows)
            if ((r.f[(size_t)(piv >> 6)] >> (piv & 63)) & 1) {
                for (int w = 0; w < W; ++w) r.f[(size_t)w] ^= f[(size_t)w];
                r.op = Pauli::multiply(r.op, op);
            }
        E.rows.push_back(Row{std::move(f), piv, std::move(op)});
    }

    // compile a frame Pauli with no X on a fixed row to a register op on the free register
    FOp compile(const Pauli& Q) const {
        FOp o;
        for (int d = 0; d < k; ++d) {
            const int a = T_free_[(size_t)d];
            if (Q.xbit(a)) o.xm |= 1ull << d;
            if (Q.zbit(a)) o.zm |= 1ull << d;
        }
        for (int w = 0; w < W; ++w)
            if (Q.x[(size_t)w] & ~freemask[(size_t)w])
                throw std::logic_error("RecordFidelity: operator leaves the tracked register (internal)");
        o.c = kIPow[Q.phase & 3] * (parw(Q.z.data(), eps.data(), W) ? -1.0 : 1.0);
        return o;
    }
    std::vector<int> T_free_;

    std::complex<double> inner_b(const std::vector<std::complex<double>>& v) const {
        std::complex<double> s = 0.0;
        for (size_t i = 0; i < b.size(); ++i) s += std::conj(b[i]) * v[i];
        return s;
    }

    // The resident bare register: 2^k amplitudes (k <= 62 is STRUCTURAL: uint64 index masks), with the work copy the
    // projector expectation takes, 16 B each; refused by name against the caller's byte budget (if any).
    size_t need_bytes(int bits) const {
        return 16u * ((size_t(2) << bits) + (size_t(1) << k));
    }
    void check_dim(int bits, const char* what) {
        if (bits > 62)
            throw std::runtime_error(std::string("RecordFidelity: ") + what + " needs a register of 2^" +
                                     std::to_string(bits) + " amplitudes; indices are 64-bit masks (k <= 62) — refused");
        if (max_bytes && need_bytes(bits) > max_bytes)
            throw std::runtime_error(std::string("RecordFidelity: ") + what + " needs " +
                                     std::to_string(need_bytes(bits)) + " bytes (16 B x (2·2^" + std::to_string(bits) +
                                     " + 2^" + std::to_string(k) + ")), above the byte budget " +
                                     std::to_string(max_bytes) + " — refused");
        st.max_dim_bits = std::max<long>(st.max_dim_bits, bits);
    }
    size_t bytes_held() const {
        size_t m = 0;
        for (const auto& kv : memo) m += memo_entry_bytes(kv.first);   // + node/bucket overhead
        return m + held_l3 + 16 * b.size();  // + contents, kernels, frame-image caches; the bare register
    }
    // clear_memo(): the memo alone (evaluate() without keep_memo). clear(): everything the evaluator holds — the memo,
    // the layout-3 contents with their kernels and the frame-image caches (ruling C2: contents and
    // kernels otherwise persist across evaluate() calls, so a warm evaluator recompiles nothing).
    void clear_memo() {
        memo.clear(); memo.rehash(0);
        held_memo = 0;
    }
    void clear() {
        clear_memo();
        l3c.clear(); l3c.shrink_to_fit();
        l3_ids.clear(); l3_ids.rehash(0);
        ny_cache.clear(); ny_cache.rehash(0);
        rec_lab_img.clear(); rec_lab_img.shrink_to_fit();
        lam_tab.clear(); lam_tab.shrink_to_fit(); lam_done.clear(); lam_done.shrink_to_fit();
        lam_sparse.clear(); lam_sparse.rehash(0); lam_mode = 0;
        held_l3 = held_rest;                 // termroute-a m4: the register's rest structure stays (and stays charged)
    }

    // <B| Π_K |B> · 2^{-rho} for the commuting signed set `ops` (all fixed parts handled by elim)
    // <B| Π_K |B> = m · 2^{-e}: the record scale 2^{-rows} is carried as the INTEGER exponent e (exact; never a float
    // that could underflow), the mantissa m = <b|Π_kernel|b> lives on the free register. Every ratio is formed on the
    // mantissas and the exponents are added back once (final review r2, B1').
    struct Scaled { double m = 0.0; int e = 0; bool zero = false; };   // zero: decided structurally (ref_ny)
    Scaled proj_expect(const std::vector<Pauli>& ops, Elim* keep) {
        if (ops.empty()) return Scaled{bnorm2, 0};   // no projector: <B|B> (no 2^k copy)
        Elim E;
        for (const Pauli& q : ops) insert(E, q);
                std::vector<std::complex<double>> v = b;
        for (const Pauli& K : E.kernel) apply_projector(v, compile(K));
        Scaled r{inner_b(v).real(), (int)E.rows.size()};
        if (keep) *keep = std::move(E);
        return r;
    }

    // ── J-b1 P7: layout-3 records ──
    std::vector<Pauli> recops;               // record operators (storage column order, exact phase)
    // the records' nonzero words (records are sparse): per record c, entries [rsw_off[c], rsw_off[c + 1]) of
    // (word, x word, z word) — <P, M_c> and the support tests read only these
    std::vector<int> rsw_off, rsw_w;
    std::vector<uint64_t> rsw_x, rsw_z;
    int rec_sympl(const uint64_t* px, const uint64_t* pz, int c) const {
        int p = 0;
        for (int i = rsw_off[(size_t)c]; i < rsw_off[(size_t)c + 1]; ++i) {
            const int w = rsw_w[(size_t)i];
            p += __builtin_popcountll((px[w] & rsw_z[(size_t)i]) ^ (pz[w] & rsw_x[(size_t)i]));
        }
        return p & 1;
    }
    // J-b1 P12 (fixplan-p12-fidelity ruling, X1): production's record reduction on layout-3 records. R_s = the records
    // whose operator stabilizes |B⟩ (sign rs_sign); every other record is R_n. Per content, a record of R_s whose
    // pull-back C0†M_cC0 equals M_c (UNDRESSED) is a δ / consistency check; only dressed R_s records and R_n enter
    // the register, the elimination and the memo key ("use"). Reference: ⟨B|Π_{R_s} = Π δ(y_c = s_c)⟨B| on the bra,
    // so the overlap carries R_n only.
    std::vector<Pauli> rec_lab_img;   // content-independent frame images (undressed operators)
    std::vector<uint8_t> rs;                 // record c ∈ R_s
    std::vector<uint8_t> rs_sign;            // its sign on |B⟩ (1 = eigenvalue −1)
    // ── register fidelity (strict contract; header). O = out wires (omask), R = the rest (rmask).
    int route = 0;                           // 1 = 3a (a = 0), 2 = 3b (a > 0)
    int MW = 0;                              // record words
    std::vector<uint64_t> omask, rmask;      // W words
    // 3a: S* = S0 ∩ P_O, |O| generators signed +1 on |B⟩, and their undressed frame images
    std::vector<Pauli> star, star_img0;
    // 3b: the rest stabilizer group S_χ = C ⊕ ⟨independent Born records⟩, C = the elements of S0 ∩ P_R commuting with
    // every Born record (dimension dC, the sector bits). Per C index i the rest destabilizer d_i (⟨d_i, g_j⟩ = δ_ij on
    // every S_χ generator): its frame image (2W words, x | z) and its record flips (MW words). Per deferred wire q the
    // C-syndromes of X_q / Z_q (WC words; zero off R).
    int dC = 0, WC = 0;
    std::vector<uint64_t> dimg, drec, colx, colz;
    std::vector<uint64_t> rsmask, rssignw;   // MW words: R_s records, their signs
    std::vector<uint64_t> pq_;               // per-shot scratch (the corrected prefix)
    // ══ S8 RF3/RF4: the twirled Pauli-expansion route (rfref/fable-decoded-design.md §2-§3; plan r4.1) ═══════════
    // A content's C0 as a NON-redundant Pauli expansion (lab): C0 = c Σ_t i^{Q(t)} G_0 · Π_j G_j^{t_j} (the product in
    // index order, left to right), |c|² = 2^{−ec}, the G_j independent. Kind 0: the closed form (expand_diag); a
    // factor list: the direct product expansion reduced by the exact Gauss sum over its redundancy (expand_factors).
    struct Expansion {
        Pauli G0;
        std::vector<Pauli> G;
        QForm q;                             // the coefficient form over t (constant 0)
        int ec = 0;
    };
    // A fiber: an affine family e = e_s ⊕ Σ_m c_m k_m of variable assignments (V bits) — the first kap directions are
    // logical (independent free-register parts L), the rest stabilizer directions (L = 0). Content-level: Q0(k_m),
    // B(k_m, k_m'), the logical L's.
    struct Fib {
        int d = 0, kap = 0, WD = 1;
        std::vector<uint64_t> vec;           // d × WV
        std::vector<uint8_t> q;              // Q0(k_m) (Z4)
        std::vector<uint64_t> B;             // d × WD
        std::vector<uint64_t> Lu, Lv;        // kap
        std::vector<uint64_t> supp;          // WV: the union support
        std::vector<uint64_t> Bk;            // d × WV: B(·, k_m) as a vector (B(u, k_m) = parity(u ∧ Bk_m))
        size_t bytes() const {
            return (vec.capacity() + B.capacity() + Lu.capacity() + Lv.capacity() + supp.capacity() + Bk.capacity()) * 8 +
                   q.capacity();
        }
    };
    // The route-3b structure of a content. Variables: the independent R_n records (M vars, ascending; the bra's
    // projector Π_{R_n}(y'') expanded, left of D_σ P' G_0) then the expansion's G_j (right of it).
    struct NewC {
        int nM = 0, nG = 0, V = 0, WV = 1, ec = 0;
        std::vector<int> mrec;               // M var → record column
        std::vector<uint64_t> I;             // V × 2W: frame image bits (x | z) per variable
        QForm Q;                             // the content form over V (Z^z X^x convention, eps, coefficients; c = 0)
        std::vector<uint64_t> I0, sig0;      // G_0's frame image bits (2W), its C-syndrome (WC)
        std::vector<uint64_t> gsig;          // nG × WC: σ_j = syn_C(G_j)
        int CW = 0;                          // constraint words: W (fixed X) + MW (R_s records)
        std::vector<uint64_t> prow, ppay;    // pivot rows (CW) with payloads (WV), fully reduced
        std::vector<int> pbit;
        int ksec = 0;                        // sector directions (σ independent), canonical: reduced on their highest
        std::vector<uint64_t> spay, ssig;    //   σ bit spiv, ascending; ksec × WV, ksec × WC
        std::vector<int> spiv;
        int kprime = 0;                      // dim of the constraint kernel (sector + fiber directions)
        Fib K0;                              // the fiber (σ = 0 directions)
        // ⟨ψ|ψ⟩: the used records' dressed images, kernel of their fixed X parts as a fiber over WU-word masks
        int nU = 0, WU = 1;
        QForm DQ;
        Fib KD;
        size_t bytes() const {
            return sizeof(NewC) + (I.capacity() + Q.beta.capacity() + I0.capacity() + sig0.capacity() + gsig.capacity() +
                                   prow.capacity() + ppay.capacity() + spay.capacity() + ssig.capacity() +
                                   DQ.beta.capacity()) * 8 + Q.lin.capacity() + DQ.lin.capacity() +
                   (pbit.capacity() + mrec.capacity()) * 4 + K0.bytes() + KD.bytes();
        }
    };
    struct L3Content {
        std::unique_ptr<DiagExpansion> dex;  // S8: kind 0 (intern_content_diag): the closed form's input
        std::vector<RecordFidelity::Factor> fac;   // S8: a factor-list content (its general expansion, lazily)
        std::unique_ptr<NewC> nc;            // S8: the route-3b structure (built on first use)
        std::vector<Pauli> rec_img;          // 3a: frame images of C0† M_c C0
        std::vector<uint64_t> uimg;          // 3b: per used record (C.use order) the frame image of C0† M_c C0 (2W)
        std::vector<uint8_t> uph;            //   its phase
        std::vector<int> use;                // records entering ⟨ψ|ψ⟩ and the key (R_n and the dressed R_s)
        std::vector<int> rn;                 // R_n records
        std::vector<int> undressed;          // undressed R_s records (consistency checks)
        std::vector<Pauli> star_img;         // 3a: frame images of C0† g C0, g ∈ S* (exact sign)
        size_t bytes() const {
            auto pb = [](const Pauli& P) { return (P.x.capacity() + P.z.capacity()) * 8 + sizeof(Pauli); };
            size_t m = sizeof(L3Content) + (use.capacity() + rn.capacity() + undressed.capacity()) * 4;
            for (const Pauli& P : rec_img) m += pb(P);
            m += uimg.capacity() * 8 + uph.capacity();
            for (const Pauli& P : star_img) m += pb(P);
            for (const auto& f : fac) for (const auto& t : f) m += pb(t.second) + 16;
            if (dex) m += sizeof(DiagExpansion) + (dex->piv.capacity() + dex->rowof.capacity()) * 4 + dex->lin.capacity() +
                          dex->a1.capacity() + (dex->cols.capacity() + dex->z0.capacity() + dex->beta.capacity() +
                                                dex->arows.capacity()) * 8;
            return m;
        }
    };
    std::unordered_map<std::string, int> l3_ids;
    std::vector<L3Content> l3c;
    int intern_(const std::string& key, const std::function<Pauli(const Pauli&)>& pull, const std::vector<uint64_t>& ux,
                const std::vector<uint64_t>& uz, std::vector<RecordFidelity::Factor>&& fac,
                std::unique_ptr<DiagExpansion>&& dex);
    std::vector<uint8_t> tdr_scratch, tdu_scratch;

    // ── RF-ref: lifecycle and budget (rulings C2 / Q2). Contents and their kernels persist across evaluate() calls
    // (values are pure functions of their keys); only clear() drops them, clear_memo() drops the memo alone. Every
    // byte the evaluator holds is counted (memo + contents + kernels + caches) and, with max_bytes, budgeted: a record
    // whose evaluation would take the held bytes above max_bytes is refused by name.
    size_t held_memo = 0, held_l3 = 0;
    size_t held_rest = 0;                    // the part of held_l3 built at construction (clear() keeps it)
    bool row_log_on = false;                 // test lever (S8 zero classes): per-row codes, memo bypassed
    std::vector<uint8_t> row_log;
    size_t memo_entry_bytes(const std::string& key) const { return key.capacity() + sizeof(double) + 64; }
    void charge(size_t& slot, size_t bytes, const char* what) {
        slot += bytes;
        const size_t reg = 16 * b.size();    // the resident bare register (budgeted at construction too)
        if (max_bytes && reg + held_memo + held_l3 > max_bytes)
            throw std::runtime_error(std::string("RecordFidelity: the evaluator holds ") +
                                     std::to_string(reg + held_memo + held_l3) + " bytes (bare register " +
                                     std::to_string(reg) + ", memo " + std::to_string(held_memo) +
                                     ", contents and kernels " + std::to_string(held_l3) + ") after " + what +
                                     ", above the byte budget " + std::to_string(max_bytes) +
                                     " — refused (clear() drops what it holds)");
    }
    void memo_put(const std::string& key, double F) {
        const auto r = memo.emplace(key, F);
        if (r.second) charge(held_memo, memo_entry_bytes(r.first->first), "a memo entry");
    }

    // N_y = ⟨B|Π_{R_n}(y)|B⟩ on the resident 2^k register (the lab frame images; R_n is every non-R_s record, so the
    // value is a function of the |R_n|-bit word y_{R_n}), cached per distinct word (charged to the held bytes).
    std::unordered_map<std::string, Scaled> ny_cache;
    std::string ny_key_;
    std::vector<Pauli> ny_ops_;
    Scaled ref_ny(const L3Content& C, const uint64_t* y, const std::vector<uint8_t>& td) {
#ifdef XTIM_MUTATE_RFREF_NOCOND
        (void)C; (void)y; (void)td;
        return Scaled{bnorm2, 0};                    // mutation build only: the unconditioned reference B
#endif
        if (C.rn.empty()) return Scaled{bnorm2, 0};
        auto bit = [&](int c) {
#ifdef XTIM_MUTATE_RFREF_NYTD
            (void)y;
            return (int)td[(size_t)c];               // mutation build only: the coin word td (= y ⊕ <P, M>), not y
#else
            (void)td;
            return (int)((y[c >> 6] >> (c & 63)) & 1ULL);
#endif
        };
        std::string& key = ny_key_;
        key.clear();
#ifdef XTIM_MUTATE_RFREF_NYDRESS
        const L3Content* pc = &C;                    // dressed images: per content
        key.append(reinterpret_cast<const char*>(&pc), sizeof(pc));
#endif
        for (size_t i = 0; i < C.rn.size(); i += 8) {
            uint8_t byte = 0;
            for (size_t j = i; j < C.rn.size() && j < i + 8; ++j) byte |= (uint8_t)(bit(C.rn[j]) << (j - i));
            key.push_back((char)byte);
        }
        auto it = ny_cache.find(key);
        if (it != ny_cache.end()) return it->second;
        ny_ops_.clear();
        for (int c : C.rn)
#ifdef XTIM_MUTATE_RFREF_NYDRESS
        {                                                               // mutation build only: C0-dressed images
            const size_t u = (size_t)(std::find(C.use.begin(), C.use.end(), c) - C.use.begin());
            Pauli R(n);
            for (int w = 0; w < W; ++w) { R.x[(size_t)w] = C.uimg[u * 2 * W + w]; R.z[(size_t)w] = C.uimg[u * 2 * W + W + w]; }
            R.phase = C.uph[u];
            ny_ops_.push_back(with_sign(R, bit(c)));
        }
#else
            ny_ops_.push_back(with_sign(rec_lab_img[(size_t)c], bit(c)));
#endif
        // Per-projection normalisation, decided in GF(2) (owner 2026-10-22): each independent random-outcome record
        // projects with probability exactly 1/2 (its 2^{-1} is the exponent e = rows, never a float); a dependent one is
        // a product of R_n records with no fixed X part — on route 3b an element of ±S(r) (r is a stabilizer state on
        // the rest), so its frame image is a SIGN on |B⟩ with no free part: the recorded pattern is possible iff every
        // such sign is +1. N_y'' = bnorm2 · 2^{-rows} or structurally 0; no float is compared.
        Scaled v{bnorm2, 0, false};
        {
            Elim E;
            for (const Pauli& q : ny_ops_) insert(E, q);
                        for (const Pauli& K : E.kernel) {
                const FOp o = compile(K);
                if (o.xm || o.zm || std::abs(o.c.imag()) > 0.5)
                    throw std::logic_error("RecordFidelity: a dependent Born-record product acts on the free register "
                                           "on route 3b (internal)");
                if (o.c.real() < 0) v.zero = true;    // the pattern contradicts a deterministic sign
            }
            v.e = (int)E.rows.size();
        }
        if (v.zero) v.m = 0.0;
        it = ny_cache.emplace(key, v).first;
        charge(held_l3, it->first.capacity() + sizeof(Scaled) + 64, "a conditioned-reference norm");
        ++st.rn_words;
        return v;
    }

    // C-syndrome (WC words) of the Pauli X^x Z^z (deferred-wire words): XOR over its R support of the per-wire columns
    void syndrome_of(const uint64_t* x, const uint64_t* z, uint64_t* s) const {
        for (int w = 0; w < WC; ++w) s[w] = 0;
        for (int w = 0; w < W; ++w) {
            for (uint64_t mm = x[w] & rmask[(size_t)w]; mm; mm &= mm - 1) {
                const uint64_t* c = &colx[(size_t)(w * 64 + __builtin_ctzll(mm)) * WC];
                for (int v = 0; v < WC; ++v) s[v] ^= c[v];
            }
            for (uint64_t mm = z[w] & rmask[(size_t)w]; mm; mm &= mm - 1) {
                const uint64_t* c = &colz[(size_t)(w * 64 + __builtin_ctzll(mm)) * WC];
                for (int v = 0; v < WC; ++v) s[v] ^= c[v];
            }
        }
    }
    // D_σ = Π_{i ∈ σ} d_i: XOR its frame image (2W words) and record flips (MW words) into img / rec
    void dest_product(const uint64_t* sig, uint64_t* im, uint64_t* rec) const {
        for (int w = 0; w < WC; ++w)
            for (uint64_t mm = sig[w]; mm; mm &= mm - 1) {
                const size_t i = (size_t)(w * 64 + __builtin_ctzll(mm));
                const uint64_t* di = &dimg[i * 2 * W];
                for (int v = 0; v < 2 * W; ++v) im[v] ^= di[v];
                const uint64_t* dr = &drec[i * MW];
                for (int v = 0; v < MW; ++v) rec[v] ^= dr[v];
            }
    }

    double evaluate_record(int id, const uint64_t* px, const uint64_t* pz, const uint64_t* y, const uint64_t* qx,
                           const uint64_t* qz) {
        try {
            return evaluate_record_(id, px, pz, y, qx, qz);
        } catch (const std::bad_alloc&) {
            throw std::runtime_error("RecordFidelity: allocation failed for this record — refused");
        }
    }
    double evaluate_record_(int id, const uint64_t* px0, const uint64_t* pz0, const uint64_t* y, const uint64_t* qx,
                            const uint64_t* qz) {
        if (id < 0 || id >= (int)l3c.size()) throw std::out_of_range("RecordFidelity: content index out of range");
        ++st.shots;
        // Pq = P_c · P: the correction (on O; it commutes with every record, which lives on R) is a Pauli factor of
        // the prefix — a sign in the kernel and part of the memo key (review (iii))
        const uint64_t* px = px0;
        const uint64_t* pz = pz0;
        if (qx || qz) {
            pq_.resize((size_t)2 * W);
            for (int w = 0; w < W; ++w) {
                const uint64_t cx = qx ? qx[w] : 0, cz = qz ? qz[w] : 0;
                if ((cx | cz) & ~omask[(size_t)w])
                    throw std::invalid_argument("RecordFidelity: RF_CORRECTION_OFF_OUTPUT: the correction acts off the "
                                                "OUTPUT_QUBITS wires — refused");
                pq_[(size_t)w] = px0[w] ^ cx;
                pq_[(size_t)(W + w)] = pz0[w] ^ cz;
            }
            px = pq_.data();
            pz = pq_.data() + W;
        }
#ifdef XTIM_MUTATE_RF_MEMO_NOC
        const uint64_t* kx = px0;                // mutation build only: the memo key without the correction
        const uint64_t* kz = pz0;
#else
        const uint64_t* kx = px;
        const uint64_t* kz = pz;
#endif
        L3Content& C = l3c[(size_t)id];
        const int m = (int)recops.size();
        std::string& key = key_scratch;
        key.assign(reinterpret_cast<const char*>(&id), sizeof(id));
        key.push_back(route == 1 ? '\x07' : '\x08');
        std::vector<uint8_t>& td = tdr_scratch;
        td.assign((size_t)m, 0);
        for (int c : C.use)
            td[(size_t)c] = (uint8_t)((int)((y[c >> 6] >> (c & 63)) & 1ULL) ^ rec_sympl(px, pz, c));
        std::vector<uint8_t>& tu = tdu_scratch;   // the key: the USED records' signs and the route's part of Pq
        tu.clear();
#ifdef XTIM_MUTATE_RF_MEMOKEY
        if (route == 1)                          // mutation build only: 3b's key without the records' signs
#endif
        for (int c : C.use) tu.push_back(td[(size_t)c]);
        if (route == 1)                          // 3a: Pq enters only through its S* parities
            for (const Pauli& g : star) tu.push_back((uint8_t)sympl(kx, kz, g, W));
        for (size_t i = 0; i < tu.size(); i += 8) {
            uint8_t byte = 0;
            for (size_t j = i; j < tu.size() && j < i + 8; ++j) byte |= (uint8_t)(tu[j] << (j - i));
            key.push_back((char)byte);
        }
        // 3b: solve the shot's constraints (the content's twirled Pauli expansion, below); the key is everything the
        // value reads beyond the content — t_0, the sector σ, the base L, the fiber's cross bits and the Born bits y''
        // (and, above, the used records' signs td: ⟨ψ|ψ⟩)
        NewC* Xp = nullptr;
        bool feasible = true;
        if (route == 2) {
            Xp = &new_content(C);
#ifdef XTIM_MUTATE_RF_MEMO_NOC
            feasible = solve_new(C, *Xp, kx, kz, y);            // mutation build only: the key without the correction
            key_new(*Xp, feasible, C, y);
            feasible = solve_new(C, *Xp, px, pz, y);
#else
            feasible = solve_new(C, *Xp, px, pz, y);
            key_new(*Xp, feasible, C, y);
#endif
        }
        // consistency of the undressed R_s records (each must be the identity projector on C0|B⟩ = |B⟩'s image).
        // Checked BEFORE the memo lookup (final review m1): the key carries only the used records' signs, so a memo hit
        // must not serve a row whose undressed record contradicts its definite value.
#ifdef XTIM_MUTATE_RF_UNDRESSED_AFTER_MEMO
        const bool undressed_first = false;      // mutation build only: the old order (a memo hit skips the check)
#else
        const bool undressed_first = true;
#endif
        auto check_undressed = [&] {
            for (int c : C.undressed)
                if (((int)((y[c >> 6] >> (c & 63)) & 1ULL) ^ rec_sympl(px, pz, c)) != rs_sign[(size_t)c])
                    throw std::runtime_error("RecordFidelity: record " + std::to_string(c) + " is definite on the "
                                             "retained state but its recorded value disagrees (impossible record)");
        };
        if (undressed_first) check_undressed();
        const bool use_memo = !row_log_on && !hform_oracle;
        if (use_memo) {
            auto mit = memo.find(key);
            if (mit != memo.end()) { ++st.memo_hits; return mit->second; }
        }
        ++st.misses;
        if (!undressed_first) check_undressed();
        double F = 0.0;
        int cls = kValue;
        if (route == 1) {
            std::vector<Pauli> chops;
            chops.reserve(C.use.size() + star.size());
            for (int c : C.use) chops.push_back(with_sign(C.rec_img[(size_t)c], td[(size_t)c]));
            const Scaled D = proj_expect(chops, nullptr);   // N = <ψ|ψ> (the correction commutes with every record)
            // the impossible-record refusal reads the mantissa (the exact 2^{-e} factor is never zero); D is α-dependent,
            // so this loud refusal is the one test that is not structural — an impossible record's projection is exactly 0
            if (!(D.m > 1e-300))
                throw std::runtime_error("RecordFidelity: <psi|psi> = 0 for this record (impossible record)");
            // 3a: F·N = <B| C0†Pq Π_rec(y) Π_{S*} Pq C0 |B>: Pq flips the S* signs by its parities, C0 dresses them
            for (size_t i = 0; i < star.size(); ++i) {
#ifdef XTIM_MUTATE_RF_STARSIGN
                chops.push_back(C.star_img[i]);      // mutation build only: the parities of Pq on S* dropped
#else
                chops.push_back(with_sign(C.star_img[i], sympl(px, pz, star[i], W)));
#endif
            }
            const Scaled Nm = proj_expect(chops, nullptr);
#ifdef XTIM_MUTATE_RF_UNDERFLOW
            F = std::ldexp(Nm.m, -Nm.e) / std::ldexp(D.m, -D.e);   // mutation build only: the scales as floats
#else
            F = std::ldexp(Nm.m / D.m, D.e - Nm.e);       // the exponents cancel exactly
#endif
        } else if (!feasible) {
            // an INFEASIBLE row scores exactly 0 — unless the record is impossible (⟨ψ|ψ⟩ = 0: refused by name)
            const Scaled D = norm_new(C, *Xp, td);
            if (D.zero || !(D.m > 1e-300))
                throw std::runtime_error("RecordFidelity: <psi|psi> = 0 for this record (impossible record)");
            cls = kInfeasible;
        } else {
            oracle_skip_ = false;
            F = value_new(C, *Xp, y, td, cls);
            if (oracle_skip_) { cls = 255; F = std::nan(""); }
        }
        if (row_log_on) row_log.push_back((uint8_t)cls);
        if (use_memo) memo_put(key, F);
        return F;
    }
    // ══ S8 RF3/RF4: the twirled Pauli-expansion evaluator (route 3b) ═════════════════════════════════════════════
    // F·⟨ψ|ψ⟩ = Σ_σ |A_σ|² / N_{y''},  A_σ = ⟨B| Π_{R_n}(y'') D_σ P' C0 |B⟩ (production's sector decomposition), with
    // Π_{R_n}(y'') = 2^{−m} Σ_T (−1)^{y''·T} M_T over the m independent Born records (the dependent ones are the GF(2)
    // consistency ref_ny already decides) and C0's expansion: every term W = M_T · D_σ P' G_0 · G(t) has
    // ⟨B|W|B⟩ = 0 unless its frame image has no X on a fixed row (a linear constraint on (T, t) together with the R_s
    // records' check, both affine in the shot); then ⟨B|W|B⟩ = i^{Φ} λ(x_free, z_free) with Φ a Z4 quadratic form
    // (the Z^z X^x order: Σ h_j e_j + 2 Σ_{j<j'} e_j e_j' x_j·z_j', eps and the coefficient forms included). The
    // surviving terms of a sector form a fiber; its stabilizer directions are summed EXACTLY (gauss_sum), its logical
    // directions by λ lookups in ascending (z_free, x_free) order. Only the final λ sum, |·|² and the ratio are floats.
#ifdef XTIM_MUTATE_RF_INTH
    int inth_ec_ = 0;
#endif
    static constexpr int kSectorRankCap = 20;        // RF_SECTOR_RANK: 2^20 sector terms per shot
    static constexpr int kLogicalRankCap = 20;       // RF_LOGICAL_RANK: 2^20 λ lookups per sector term
    static constexpr int kFiberRankCap = 256;        // RF_FIBER_RANK: κ' (declared; never reached on the corpus)
    enum { kInfeasible = 0, kBornZero = 1, kGaussZero = 2, kValue = 3 };
    std::vector<int> rn_ind;                         // independent Born records (S_χ generators beyond C), ascending
    bool hform_oracle = false;                       // test lever (S8 RF4): fibers by exact enumeration (κ' <= 12)
    bool oracle_skip_ = false;                       //   a fiber above 12: the row is logged as class 255, F = NaN

    // the frame image's bits of X^x Z^z (no phase): the XOR of the per-wire inverse rows' bits
    std::vector<uint64_t> invrows;                   // route 3b: per wire q, U†X_qU bits (2W) then U†Z_qU bits (2W)
    void img_bits(const uint64_t* x, const uint64_t* z, std::vector<uint64_t>& out) const {
        out.assign((size_t)2 * W, 0);
        uint64_t* o = out.data();
        for (int w = 0; w < W; ++w) {
            for (uint64_t mm = x[w]; mm; mm &= mm - 1) {
                const uint64_t* r = &invrows[(size_t)(w * 64 + __builtin_ctzll(mm)) * 4 * W];
                for (int t = 0; t < 2 * W; ++t) o[t] ^= r[t];
            }
            for (uint64_t mm = z[w]; mm; mm &= mm - 1) {
                const uint64_t* r = &invrows[(size_t)(w * 64 + __builtin_ctzll(mm)) * 4 * W + 2 * W];
                for (int t = 0; t < 2 * W; ++t) o[t] ^= r[t];
            }
        }
    }
    std::vector<uint8_t> invph;                      // route 3b: per wire q, the phases of U†X_qU and U†Z_qU
    // the exact frame image of i^ph X^x Z^z: the per-wire rows folded in fold_rows_' order (wires ascending, X before Z),
    // acc ← acc · row (phase += row's + 2 acc.z·row.x); into out (2W words), its phase into oph
    void img_full(const uint64_t* x, const uint64_t* z, int ph, uint64_t* out, int& oph) const {
        for (int t = 0; t < 2 * W; ++t) out[t] = 0;
        int p = ph;
        for (int w = 0; w < W; ++w) {
            uint64_t xb = x[w], zb = z[w];
            const int base = w << 6;
            while (xb | zb) {
                const int qx = xb ? base + __builtin_ctzll(xb) : n + 1;
                const int qz = zb ? base + __builtin_ctzll(zb) : n + 1;
                const uint64_t* row;
                if (qx <= qz) { row = &invrows[(size_t)qx * 4 * W]; p += invph[(size_t)qx * 2]; xb &= xb - 1; }
                else { row = &invrows[(size_t)qz * 4 * W + 2 * W]; p += invph[(size_t)qz * 2 + 1]; zb &= zb - 1; }
                p += 2 * parw(out + W, row, W);
                for (int t = 0; t < 2 * W; ++t) out[t] ^= row[t];
            }
        }
        oph = p & 3;
    }
    void lfree(const uint64_t* xz, uint64_t& u, uint64_t& v) const {
        u = v = 0;
        for (int d = 0; d < k; ++d) {
            const int a = T_free_[(size_t)d];
            u |= (uint64_t)gs::getbit(xz, a) << d;
            v |= (uint64_t)gs::getbit(xz + W, a) << d;
        }
    }
    // the (z_free, x_free) order key's highest bit (−1 for zero)
    int lhb(uint64_t u, uint64_t v) const {
        if (v) return k + 63 - __builtin_clzll(v);
        if (u) return 63 - __builtin_clzll(u);
        return -1;
    }
    bool lbit(uint64_t u, uint64_t v, int b) const { return b >= k ? (v >> (b - k)) & 1 : (u >> b) & 1; }
    void refuse_factor(const std::string& why) const {
        throw std::invalid_argument("RecordFidelity: RF_FACTOR_NOT_CLIFFORD: a C0 factor " + why +
                                    " (the reference overlap takes π/4 rotations: S, CZ, rot(B)) — refused");
    }

    // A factor list's expansion: the direct product (one variable per {c0 I, c1 Q}, two per CZ) reduced to its
    // non-redundant support by the exact Gauss sum over the kernel of t ↦ the Pauli's bits.
    // A factor list's shape, refused by name otherwise: per {c0 I, c1 Q} (c1/c0 = ±i) one variable, per CZ
    // {I, A, B, −AB}/2 two (A then B, β_AB = 1); the variables in product order (C0 = F_V ⋯ F_1: the last factor's
    // terms leftmost). Returns ec (|c|² = 2^{−ec}).
    int parse_factors(const std::vector<RecordFidelity::Factor>& facs, std::vector<Pauli>& P, std::vector<uint8_t>& cl,
                      std::vector<std::pair<int, int>>& cb) const {
        int ec = 0;
        const double r2 = 1.0 / std::sqrt(2.0);
        auto near = [](std::complex<double> a, std::complex<double> b) { return std::abs(a - b) < 1e-12; };
        for (size_t fi = facs.size(); fi-- > 0;) {           // C0 = F_V ⋯ F_1: the last factor's term leftmost
            const RecordFidelity::Factor& f = facs[fi];
            if (f.size() == 2 && is_identity(f[0].second) && near(std::abs(f[0].first), r2) &&
                near(std::abs(f[1].first), r2)) {
                if (!hermitian(f[1].second) || is_identity(f[1].second)) refuse_factor("has a non-Hermitian or identity axis");
                const std::complex<double> r = f[1].first / f[0].first;
                int l = -1;
                if (near(r, {0, 1})) l = 1;
                else if (near(r, {0, -1})) l = 3;
                else refuse_factor("{c0 I, c1 Q} has c1/c0 = (" + std::to_string(r.real()) + ", " + std::to_string(r.imag()) + "), not ±i");
                P.push_back(f[1].second);
                cl.push_back((uint8_t)l);
                ec += 1;
            } else if (f.size() == 4 && is_identity(f[0].second) && near(f[0].first, 0.5) && near(f[1].first, 0.5) &&
                       near(f[2].first, 0.5) && near(f[3].first, -0.5)) {
                const Pauli p12 = Pauli::multiply(f[1].second, f[2].second);
                if (!(p12.x == f[3].second.x && p12.z == f[3].second.z && ((p12.phase - f[3].second.phase) & 3) == 0) ||
                    Pauli::anticommute_bit(f[1].second, f[2].second))
                    refuse_factor("{I, A, B, −AB}/2 has AB != A·B or anticommuting A, B");
                const int a = (int)P.size();
                P.push_back(f[1].second);
                P.push_back(f[2].second);
                cl.push_back(0);
                cl.push_back(0);
                cb.push_back({a, a + 1});
                ec += 2;
            } else {
                refuse_factor("with " + std::to_string(f.size()) + " terms is neither {c0 I, c1 Q} nor a CZ");
            }
        }
        return ec;
    }
    Expansion expand_factors(const std::vector<RecordFidelity::Factor>& facs) const {
        std::vector<Pauli> P;
        std::vector<uint8_t> cl;
        std::vector<std::pair<int, int>> cb;
        const int ec = parse_factors(facs, P, cl, cb);
        const int Vr = (int)P.size();
        Expansion E;
        E.G0 = Pauli(n);
        if (Vr == 0) { E.q.init(0); return E; }
        QForm T;
        T.init(Vr);
        for (int j = 0; j < Vr; ++j) T.lin[(size_t)j] = (uint8_t)((cl[(size_t)j] + P[(size_t)j].phase) & 3);
#ifndef XTIM_MUTATE_RF_K2CROSS
        for (int j = 0; j < Vr; ++j)
            for (int j2 = j + 1; j2 < Vr; ++j2)
                if (parw(P[(size_t)j].z.data(), P[(size_t)j2].x.data(), W)) T.flip(j, j2);
#endif                                                           // mutation build only: 2 z_j·x_j' dropped
        for (const auto& e : cb) T.flip(e.first, e.second);
        // the image map t ↦ (x | z): pivot variables and kernel vectors
        const int WT = T.WV;
        std::vector<std::vector<uint64_t>> rv, rp, kern;
        std::vector<int> rpiv, pvar;
        for (int j = 0; j < Vr; ++j) {
            std::vector<uint64_t> v = bits_of(P[(size_t)j], W), pay((size_t)WT, 0);
            gs::flipbit(pay.data(), j);
            for (size_t r = 0; r < rv.size(); ++r)
                if (gs::getbit(v.data(), rpiv[r])) {
                    for (int w = 0; w < 2 * W; ++w) v[(size_t)w] ^= rv[r][(size_t)w];
                    for (int w = 0; w < WT; ++w) pay[(size_t)w] ^= rp[r][(size_t)w];
                }
            int pb = -1;
            for (int w = 0; w < 2 * W && pb < 0; ++w) if (v[(size_t)w]) pb = w * 64 + __builtin_ctzll(v[(size_t)w]);
            if (pb < 0) { kern.push_back(std::move(pay)); continue; }
            for (size_t r = 0; r < rv.size(); ++r)
                if (gs::getbit(rv[r].data(), pb)) {
                    for (int w = 0; w < 2 * W; ++w) rv[r][(size_t)w] ^= v[(size_t)w];
                    for (int w = 0; w < WT; ++w) rp[r][(size_t)w] ^= pay[(size_t)w];
                }
            rv.push_back(std::move(v));
            rp.push_back(std::move(pay));
            rpiv.push_back(pb);
            pvar.push_back(j);
        }
        // change of variables t = Σ_i a_i e_{pvar[i]} ⊕ Σ_m b_m kern_m; the a's are the parameters
        const int na = (int)pvar.size(), nb = (int)kern.size();
        std::vector<std::vector<uint64_t>> basis;
        for (int i = 0; i < na; ++i) {
            std::vector<uint64_t> e((size_t)WT, 0);
            gs::flipbit(e.data(), pvar[(size_t)i]);
            basis.push_back(std::move(e));
        }
        for (auto& kv : kern) basis.push_back(kv);
        QForm T2;
        T2.init(Vr);
        for (int i = 0; i < Vr; ++i) T2.lin[(size_t)i] = (uint8_t)T.eval(basis[(size_t)i].data());
        for (int i = 0; i < Vr; ++i)
            for (int i2 = i + 1; i2 < Vr; ++i2)
                if (T.polar(basis[(size_t)i].data(), basis[(size_t)i2].data())) T2.flip(i, i2);
        std::vector<uint8_t> keep((size_t)Vr, 0);
        for (int i = 0; i < na; ++i) keep[(size_t)i] = 1;
        (void)nb;
        GaussSum G;
        gauss_sum(T2, keep, G);
        if (G.zero) throw std::logic_error("RecordFidelity: a C0 factor list expands to zero (internal)");
        auto patt = [&](int i) { return bits_of(P[(size_t)pvar[(size_t)i]], W); };
        const int F = (int)G.free_keep.size();
        std::vector<uint64_t> w0((size_t)2 * W, 0);
        for (size_t d = 0; d < G.dep_var.size(); ++d)
            if (G.dep_c[d]) { const auto pw = patt(G.dep_var[d]); for (int w = 0; w < 2 * W; ++w) w0[(size_t)w] ^= pw[(size_t)w]; }
        std::vector<std::vector<uint64_t>> wf((size_t)F);
        for (int fi = 0; fi < F; ++fi) {
            const int i = G.free_keep[(size_t)fi];
            wf[(size_t)fi] = patt(i);
            for (size_t d = 0; d < G.dep_var.size(); ++d)
                if (gs::getbit(&G.dep_mask[d * (size_t)T2.WV], i)) {
                    const auto pw = patt(G.dep_var[d]);
                    for (int w = 0; w < 2 * W; ++w) wf[(size_t)fi][(size_t)w] ^= pw[(size_t)w];
                }
        }
        E.G0 = pauli_of(w0, n, W);
        E.G0.phase = 0;
        for (int fi = 0; fi < F; ++fi) { E.G.push_back(pauli_of(wf[(size_t)fi], n, W)); E.G.back().phase = 0; }
        // coefficient of the ORDERED product G0 · Π G_i^{f_i}: i^{Qf(f)} · i^{−π'(f)}, π' its phase vs the bare bits
        E.q = G.f;
        for (int fi = 0; fi < F; ++fi) {
            if (parw(w0.data() + W, wf[(size_t)fi].data(), W)) E.q.lin[(size_t)fi] = (uint8_t)((E.q.lin[(size_t)fi] + 2) & 3);
            for (int f2 = fi + 1; f2 < F; ++f2)
                if (parw(wf[(size_t)fi].data() + W, wf[(size_t)f2].data(), W)) E.q.flip(fi, f2);
        }
        E.ec = ec - G.h;
        if (E.ec != F) throw std::logic_error("RecordFidelity: a C0 factor list's expansion is not unitary (internal)");
        return E;
    }
    Expansion expansion_of(const L3Content& C) const {
        if (!C.dex) return expand_factors(C.fac);
        const DiagExpansion& D = *C.dex;
        Expansion E;
        E.G0 = Pauli(n);
        for (int w = 0; w < W; ++w) E.G0.z[(size_t)w] = D.z0[(size_t)w];
        for (int i = 0; i < D.r; ++i) {
            Pauli g(n);
            for (int w = 0; w < W; ++w) g.z[(size_t)w] = D.cols[(size_t)i * W + w];
            E.G.push_back(std::move(g));
        }
        E.q.init(D.r);
        for (int i = 0; i < D.r; ++i) {
            E.q.lin[(size_t)i] = D.lin[(size_t)i];
            for (int j = 0; j < D.r; ++j)
                if (j != i && ((D.beta[(size_t)i * D.WB + (j >> 6)] >> (j & 63)) & 1ULL)) E.q.row(i)[j >> 6] |= 1ULL << (j & 63);
        }
        E.ec = D.r;
        return E;
    }

    // A fiber from kernel vectors (V-bit, WV words) of a form over variables with frame-image bits Ixz (V × 2W).
    Fib make_fiber(const QForm& Q, const uint64_t* vecs, int nvec, const std::vector<uint64_t>& Ixz) const {
        const int WV = Q.WV;
        // split by L: rows (L reduced, its pivot, the vector) are the logical directions, zeros the stabilizer ones
        std::vector<uint64_t> Lu, Lv, lvec, svec, vec((size_t)WV), acc((size_t)2 * W);
        std::vector<int> Lp;
        for (int i = 0; i < nvec; ++i) {
            const uint64_t* v0 = vecs + (size_t)i * WV;
            std::fill(acc.begin(), acc.end(), 0);
            for (int w = 0; w < WV; ++w)
                for (uint64_t m = v0[w]; m; m &= m - 1) {
                    const uint64_t* I = &Ixz[(size_t)(w * 64 + __builtin_ctzll(m)) * 2 * W];
                    for (int t = 0; t < 2 * W; ++t) acc[(size_t)t] ^= I[t];
                }
            uint64_t u, v;
            lfree(acc.data(), u, v);
            for (int w = 0; w < WV; ++w) vec[(size_t)w] = v0[w];
            for (size_t r = 0; r < Lp.size(); ++r)
                if (lbit(u, v, Lp[r])) { u ^= Lu[r]; v ^= Lv[r]; for (int w = 0; w < WV; ++w) vec[(size_t)w] ^= lvec[r * WV + w]; }
            const int p = lhb(u, v);
            if (p < 0) { svec.insert(svec.end(), vec.begin(), vec.end()); continue; }
            for (size_t r = 0; r < Lp.size(); ++r)
                if (lbit(Lu[r], Lv[r], p)) { Lu[r] ^= u; Lv[r] ^= v; for (int w = 0; w < WV; ++w) lvec[r * WV + w] ^= vec[(size_t)w]; }
            Lu.push_back(u);
            Lv.push_back(v);
            Lp.push_back(p);
            lvec.insert(lvec.end(), vec.begin(), vec.end());
        }
        Fib F;
        F.kap = (int)Lp.size();
        F.d = F.kap + (int)(svec.size() / (size_t)WV);
        F.WD = std::max(1, (F.d + 63) / 64);
        F.vec = std::move(lvec);
        F.vec.insert(F.vec.end(), svec.begin(), svec.end());
        F.supp.assign((size_t)WV, 0);
        for (int m = 0; m < F.d; ++m) for (int w = 0; w < WV; ++w) F.supp[(size_t)w] |= F.vec[(size_t)m * WV + w];
        F.Lu = std::move(Lu);
        F.Lv = std::move(Lv);
        // Bk_m = B(·, k_m) as a V-bit vector (B(u, k_m) = parity(u ∧ Bk_m)): B on the fiber, and per shot B(e_s, k_m)
        F.q.assign((size_t)F.d, 0);
        F.B.assign((size_t)F.d * F.WD, 0);
        F.Bk.assign((size_t)F.d * WV, 0);
        for (int m = 0; m < F.d; ++m) {
            const uint64_t* km = &F.vec[(size_t)m * WV];
            F.q[(size_t)m] = (uint8_t)Q.eval(km);
            uint64_t* bk = &F.Bk[(size_t)m * WV];
            for (int j = 0; j < Q.V; ++j) {
                int b = (Q.lin[(size_t)j] & 1) && gs::getbit(km, j);
                const uint64_t* rj = Q.row(j);
                for (int w = 0; w < WV; ++w) b ^= __builtin_popcountll(rj[w] & km[w]) & 1;
                if (b) gs::flipbit(bk, j);
            }
        }
        for (int m = 0; m < F.d; ++m)
            for (int m2 = m + 1; m2 < F.d; ++m2) {
                int b = 0;
                for (int w = 0; w < WV; ++w) b ^= __builtin_popcountll(F.vec[(size_t)m2 * WV + w] & F.Bk[(size_t)m * WV + w]) & 1;
                if (b) { gs::flipbit(&F.B[(size_t)m * F.WD], m2); gs::flipbit(&F.B[(size_t)m2 * F.WD], m); }
            }
        return F;
    }

    // The route-3b structure of a content (built once; charged).
    // the expansion's generators as words: kind 0 straight from the closed form (no Pauli objects), a factor list by
    // its reduced expansion
    std::vector<uint64_t> ex_gx_, ex_gz_, ex_qb_, ex_g0_;
    std::vector<uint8_t> ex_gph_, ex_ql_;
    int ex_g0ph_ = 0, ex_WQ_ = 1;
    void expansion_words(const L3Content& C, int& nG, int& ec) {
        if (C.dex) {
            const DiagExpansion& D = *C.dex;
            nG = D.r;
            ec = D.r;
            ex_gx_.assign((size_t)nG * W, 0);
            ex_gz_.assign(D.cols.begin(), D.cols.begin() + (size_t)nG * W);
            ex_gph_.assign((size_t)nG, 0);
            ex_ql_.assign(D.lin.begin(), D.lin.begin() + nG);
            ex_WQ_ = std::max(1, D.WB);
            ex_qb_.assign((size_t)nG * ex_WQ_, 0);
            for (int i = 0; i < nG; ++i)
                for (int w = 0; w < D.WB; ++w) ex_qb_[(size_t)i * ex_WQ_ + w] = D.beta[(size_t)i * D.WB + w];
            ex_g0_.assign((size_t)2 * W, 0);
            for (int w = 0; w < W; ++w) ex_g0_[(size_t)(W + w)] = D.z0[(size_t)w];
            ex_g0ph_ = 0;
            return;
        }
        const Expansion E = expand_factors(C.fac);
        nG = (int)E.G.size();
        ec = E.ec;
        ex_gx_.assign((size_t)nG * W, 0);
        ex_gz_.assign((size_t)nG * W, 0);
        ex_gph_.assign((size_t)nG, 0);
        ex_ql_.assign(E.q.lin.begin(), E.q.lin.end());
        ex_WQ_ = std::max(1, E.q.WV);
        ex_qb_.assign(E.q.beta.begin(), E.q.beta.end());
        for (int i = 0; i < nG; ++i) {
            for (int w = 0; w < W; ++w) { ex_gx_[(size_t)i * W + w] = E.G[(size_t)i].x[(size_t)w]; ex_gz_[(size_t)i * W + w] = E.G[(size_t)i].z[(size_t)w]; }
            ex_gph_[(size_t)i] = (uint8_t)(E.G[(size_t)i].phase & 3);
        }
        ex_g0_.assign((size_t)2 * W, 0);
        for (int w = 0; w < W; ++w) { ex_g0_[(size_t)w] = E.G0.x[(size_t)w]; ex_g0_[(size_t)(W + w)] = E.G0.z[(size_t)w]; }
        ex_g0ph_ = E.G0.phase & 3;
    }

    // The route-3b structure of a content (built once; charged).
    NewC& new_content(L3Content& C) {
        if (C.nc) return *C.nc;
        auto Np = std::make_unique<NewC>();
        NewC& X = *Np;
        int nG = 0, ec = 0;
        expansion_words(C, nG, ec);
        X.nM = (int)rn_ind.size();
        X.nG = nG;
        X.V = X.nM + X.nG;
        X.ec = ec;
        X.mrec = rn_ind;
        X.Q.init(X.V);
        X.WV = X.Q.WV;
        X.I.assign((size_t)X.V * 2 * W, 0);
        for (int j = 0; j < X.V; ++j) {
            const bool isM = j < X.nM;
            const Pauli* Pl = isM ? &recops[(size_t)rn_ind[(size_t)j]] : nullptr;
            const uint64_t* lx = isM ? Pl->x.data() : &ex_gx_[(size_t)(j - X.nM) * W];
            const uint64_t* lz = isM ? Pl->z.data() : &ex_gz_[(size_t)(j - X.nM) * W];
            int ph = isM ? Pl->phase : ex_gph_[(size_t)(j - X.nM)];
            uint64_t* I = &X.I[(size_t)j * 2 * W];
            img_full(lx, lz, ph, I, ph);
#ifdef XTIM_MUTATE_RF_EPS
            const int ep = 0;                                    // mutation build only: the stabilizer character dropped
#else
            const int ep = parw(I + W, eps.data(), W);
#endif
            const int cl = isM ? 0 : ex_ql_[(size_t)(j - X.nM)];
#ifdef XTIM_MUTATE_RFREF_TSIGN
            if (isM) ph = 0;                                            // mutation build only: a Born record's phase dropped
#endif
            X.Q.lin[(size_t)j] = (uint8_t)((ph + 2 * parw(I, I + W, W) + 2 * ep + cl) & 3);
        }
        for (int j = 0; j < X.V; ++j)
            for (int j2 = j + 1; j2 < X.V; ++j2)
                if (parw(&X.I[(size_t)j * 2 * W], &X.I[(size_t)j2 * 2 * W + W], W)) X.Q.flip(j, j2);
        for (int a2 = 0; a2 < X.nG; ++a2)
            for (int b2 = a2 + 1; b2 < X.nG; ++b2)
                if ((ex_qb_[(size_t)a2 * ex_WQ_ + (b2 >> 6)] >> (b2 & 63)) & 1ULL) X.Q.flip(X.nM + a2, X.nM + b2);
        X.I0.assign((size_t)2 * W, 0);
        {
            int ph0 = 0;
            img_full(ex_g0_.data(), ex_g0_.data() + W, ex_g0ph_, X.I0.data(), ph0);
        }
        X.sig0.assign((size_t)WC, 0);
        syndrome_of(ex_g0_.data(), ex_g0_.data() + W, X.sig0.data());
        X.gsig.assign((size_t)X.nG * WC, 0);
        for (int a2 = 0; a2 < X.nG; ++a2)
            syndrome_of(&ex_gx_[(size_t)a2 * W], &ex_gz_[(size_t)a2 * W], &X.gsig[(size_t)a2 * WC]);
        // the constraint columns: the fixed X part of the variable's image with its sector's destabilizer | the R_s
        // record flips of that destabilizer
        X.CW = W + MW;
        std::vector<uint64_t> col((size_t)X.CW), pay((size_t)X.WV), im((size_t)2 * W), rc((size_t)MW);
        std::vector<uint64_t> kern;                              // the kernel payloads, flat (WV words each)
        for (int j = 0; j < X.V; ++j) {
            const uint64_t* I = &X.I[(size_t)j * 2 * W];
            std::fill(im.begin(), im.end(), 0);
            std::fill(rc.begin(), rc.end(), 0);
            if (j >= X.nM) dest_product(&X.gsig[(size_t)(j - X.nM) * WC], im.data(), rc.data());
            for (int w = 0; w < W; ++w) col[(size_t)w] = (I[w] ^ im[(size_t)w]) & ~freemask[(size_t)w];
#ifdef XTIM_MUTATE_RF_RHO
            for (int w = 0; w < MW; ++w) col[(size_t)(W + w)] = 0;      // mutation build only: ρ_i dropped
#else
            for (int w = 0; w < MW; ++w) col[(size_t)(W + w)] = rc[(size_t)w] & rsmask[(size_t)w];
#endif
            std::fill(pay.begin(), pay.end(), 0);
            gs::flipbit(pay.data(), j);
            const int npv = (int)X.pbit.size();
            for (int r = 0; r < npv; ++r)
                if (gs::getbit(col.data(), X.pbit[(size_t)r])) {
                    for (int w = 0; w < X.CW; ++w) col[(size_t)w] ^= X.prow[(size_t)r * X.CW + w];
                    for (int w = 0; w < X.WV; ++w) pay[(size_t)w] ^= X.ppay[(size_t)r * X.WV + w];
                }
            int pb = -1;
            for (int w = 0; w < X.CW && pb < 0; ++w) if (col[(size_t)w]) pb = w * 64 + __builtin_ctzll(col[(size_t)w]);
            if (pb < 0) { kern.insert(kern.end(), pay.begin(), pay.end()); continue; }
            for (int r = 0; r < npv; ++r)
                if (gs::getbit(&X.prow[(size_t)r * X.CW], pb)) {
                    for (int w = 0; w < X.CW; ++w) X.prow[(size_t)r * X.CW + w] ^= col[(size_t)w];
                    for (int w = 0; w < X.WV; ++w) X.ppay[(size_t)r * X.WV + w] ^= pay[(size_t)w];
                }
            X.prow.insert(X.prow.end(), col.begin(), col.end());
            X.ppay.insert(X.ppay.end(), pay.begin(), pay.end());
            X.pbit.push_back(pb);
        }
        X.kprime = (int)(kern.size() / (size_t)X.WV);
        if (X.kprime > kFiberRankCap)
            throw std::runtime_error("RecordFidelity: RF_FIBER_RANK: a plan content's surviving expansion terms span a fiber "
                                     "of dimension " + std::to_string(X.kprime) + " (cap " + std::to_string(kFiberRankCap) +
                                     ") — refused");
        // sector directions (σ independent) and the fiber (σ = 0)
        std::vector<uint64_t> K0;                                // the fiber's vectors, flat
        {
            std::vector<uint64_t> sg((size_t)WC);
            std::vector<int> spb;
            for (int ki = 0; ki < X.kprime; ++ki) {
                uint64_t* kv = &kern[(size_t)ki * X.WV];
                std::fill(sg.begin(), sg.end(), 0);
                for (int a = 0; a < X.nG; ++a)
                    if (gs::getbit(kv, X.nM + a)) for (int w = 0; w < WC; ++w) sg[(size_t)w] ^= X.gsig[(size_t)a * WC + w];
#ifdef XTIM_MUTATE_RF_FIBER
                std::fill(sg.begin(), sg.end(), 0);                     // mutation build only: every direction in the fiber
#endif
                for (int r = 0; r < X.ksec; ++r)
                    if (gs::getbit(sg.data(), spb[(size_t)r])) {
                        for (int w = 0; w < WC; ++w) sg[(size_t)w] ^= X.ssig[(size_t)r * WC + w];
                        for (int w = 0; w < X.WV; ++w) kv[(size_t)w] ^= X.spay[(size_t)r * X.WV + w];
                    }
                int pb = -1;
                for (int w = 0; w < WC && pb < 0; ++w) if (sg[(size_t)w]) pb = w * 64 + __builtin_ctzll(sg[(size_t)w]);
                if (pb < 0) { K0.insert(K0.end(), kv, kv + X.WV); continue; }
                for (int r = 0; r < X.ksec; ++r)
                    if (gs::getbit(&X.ssig[(size_t)r * WC], pb)) {
                        for (int w = 0; w < WC; ++w) X.ssig[(size_t)r * WC + w] ^= sg[(size_t)w];
                        for (int w = 0; w < X.WV; ++w) X.spay[(size_t)r * X.WV + w] ^= kv[(size_t)w];
                    }
                X.ssig.insert(X.ssig.end(), sg.begin(), sg.end());
                X.spay.insert(X.spay.end(), kv, kv + X.WV);
                spb.push_back(pb);
                ++X.ksec;
            }
        }
#ifdef XTIM_MUTATE_RF_RANKCAP
        if (X.ksec > kSectorRankCap + 1)                                 // mutation build only: the cap off by one
#else
        if (X.ksec > kSectorRankCap)
#endif
            throw std::runtime_error("RecordFidelity: RF_SECTOR_RANK: a plan content's rotations reach " +
                                     std::to_string(X.ksec) + " independent rest sectors that no record fixes (2^" +
                                     std::to_string(X.ksec) + " kernel terms per shot; cap 2^" +
                                     std::to_string(kSectorRankCap) + ") — refused");
        // the sector directions in canonical form: reduced on their HIGHEST σ bit, ascending pivots — a shot's coset
        // representative is reduced to the smallest σ of its coset, and the sectors are visited in ascending σ
        // (the per-sector terms are summed in an order independent of the expansion's basis)
        {
            std::vector<int> hp;
            auto hbit = [&](const uint64_t* v) {
                for (int w = WC - 1; w >= 0; --w) if (v[w]) return w * 64 + 63 - __builtin_clzll(v[w]);
                return -1;
            };
            std::vector<uint64_t> rs, rp;
            for (int r = 0; r < X.ksec; ++r) {
                std::vector<uint64_t> sg(X.ssig.begin() + (size_t)r * WC, X.ssig.begin() + (size_t)(r + 1) * WC);
                std::vector<uint64_t> py(X.spay.begin() + (size_t)r * X.WV, X.spay.begin() + (size_t)(r + 1) * X.WV);
                for (size_t q = 0; q < hp.size(); ++q)
                    if (gs::getbit(sg.data(), hp[q])) {
                        for (int w = 0; w < WC; ++w) sg[(size_t)w] ^= rs[q * WC + w];
                        for (int w = 0; w < X.WV; ++w) py[(size_t)w] ^= rp[q * X.WV + w];
                    }
                const int p = hbit(sg.data());
                if (p < 0) throw std::logic_error("RecordFidelity: dependent sector directions (internal)");
                for (size_t q = 0; q < hp.size(); ++q)
                    if (gs::getbit(&rs[q * WC], p)) {
                        for (int w = 0; w < WC; ++w) rs[q * WC + w] ^= sg[(size_t)w];
                        for (int w = 0; w < X.WV; ++w) rp[q * X.WV + w] ^= py[(size_t)w];
                    }
                rs.insert(rs.end(), sg.begin(), sg.end());
                rp.insert(rp.end(), py.begin(), py.end());
                hp.push_back(p);
            }
            std::vector<int> ord((size_t)X.ksec);
            for (int r = 0; r < X.ksec; ++r) ord[(size_t)r] = r;
            std::sort(ord.begin(), ord.end(), [&](int a, int b2) { return hp[(size_t)a] < hp[(size_t)b2]; });
            X.ssig.clear();
            X.spay.clear();
            X.spiv.clear();
            for (int r : ord) {
                X.ssig.insert(X.ssig.end(), rs.begin() + (size_t)r * WC, rs.begin() + (size_t)(r + 1) * WC);
                X.spay.insert(X.spay.end(), rp.begin() + (size_t)r * X.WV, rp.begin() + (size_t)(r + 1) * X.WV);
                X.spiv.push_back(hp[(size_t)r]);
            }
        }
        X.K0 = make_fiber(X.Q, K0.data(), (int)(K0.size() / (size_t)X.WV), X.I);
        // ⟨ψ|ψ⟩: the used records' dressed frame images (C.rec_img), ordered product ascending
        X.nU = (int)C.use.size();
        X.DQ.init(X.nU);
        X.WU = X.DQ.WV;
        std::vector<uint64_t> DI((size_t)X.nU * 2 * W, 0);
        for (int u = 0; u < X.nU; ++u) {
            uint64_t* I = &DI[(size_t)u * 2 * W];
#ifdef XTIM_MUTATE_RF_NORM
            const Pauli& Rl = rec_lab_img[(size_t)C.use[(size_t)u]];                 // mutation: <psi|psi> undressed
            for (int w = 0; w < W; ++w) { I[w] = Rl.x[(size_t)w]; I[W + w] = Rl.z[(size_t)w]; }
            const int rph = Rl.phase & 3;
#else
            for (int t = 0; t < 2 * W; ++t) I[t] = C.uimg[(size_t)u * 2 * W + t];
            const int rph = C.uph[(size_t)u];
#endif
#ifdef XTIM_MUTATE_RF_EPS
            const int ep = 0;
#else
            const int ep = parw(I + W, eps.data(), W);
#endif
            X.DQ.lin[(size_t)u] = (uint8_t)((rph + 2 * parw(I, I + W, W) + 2 * ep) & 3);
        }
        for (int u = 0; u < X.nU; ++u)
            for (int u2 = u + 1; u2 < X.nU; ++u2)
                if (parw(&DI[(size_t)u * 2 * W], &DI[(size_t)u2 * 2 * W + W], W)) X.DQ.flip(u, u2);
        {
            std::vector<uint64_t> rv, rp, dk, v((size_t)W), pu((size_t)X.WU);
            std::vector<int> rpb;
            for (int u = 0; u < X.nU; ++u) {
                for (int w = 0; w < W; ++w) v[(size_t)w] = DI[(size_t)u * 2 * W + w] & ~freemask[(size_t)w];
                std::fill(pu.begin(), pu.end(), 0);
                gs::flipbit(pu.data(), u);
                for (size_t r = 0; r < rpb.size(); ++r)
                    if (gs::getbit(v.data(), rpb[r])) {
                        for (int w = 0; w < W; ++w) v[(size_t)w] ^= rv[r * W + w];
                        for (int w = 0; w < X.WU; ++w) pu[(size_t)w] ^= rp[r * X.WU + w];
                    }
                int pb = -1;
                for (int w = 0; w < W && pb < 0; ++w) if (v[(size_t)w]) pb = w * 64 + __builtin_ctzll(v[(size_t)w]);
                if (pb < 0) { dk.insert(dk.end(), pu.begin(), pu.end()); continue; }
                for (size_t r = 0; r < rpb.size(); ++r)
                    if (gs::getbit(&rv[r * W], pb)) {
                        for (int w = 0; w < W; ++w) rv[r * W + w] ^= v[(size_t)w];
                        for (int w = 0; w < X.WU; ++w) rp[r * X.WU + w] ^= pu[(size_t)w];
                    }
                rv.insert(rv.end(), v.begin(), v.end());
                rp.insert(rp.end(), pu.begin(), pu.end());
                rpb.push_back(pb);
            }
            X.KD = make_fiber(X.DQ, dk.data(), (int)(dk.size() / (size_t)X.WU), DI);
        }
        if (X.KD.kap > kLogicalRankCap)
            throw std::runtime_error("RecordFidelity: RF_LOGICAL_RANK: a plan content's dressed records span " +
                                     std::to_string(X.KD.kap) + " logical directions (2^" + std::to_string(X.KD.kap) +
                                     " lookups; cap 2^" + std::to_string(kLogicalRankCap) + ") — refused");
        // the build-only parts go back to the allocator (the next content reuses them: no fresh pages)
        std::vector<uint64_t>().swap(X.Q.beta);
        std::vector<uint8_t>().swap(X.Q.lin);
        std::vector<uint64_t>().swap(X.DQ.beta);
        std::vector<uint8_t>().swap(X.DQ.lin);
#ifndef XTIM_MUTATE_RFREF_NYDRESS
        if (!hform_oracle) {                                    // (the enumeration oracle and NYDRESS read them)
            held_l3 -= C.uimg.capacity() * 8 + C.uph.capacity();   // charged with the content at intern
            std::vector<uint64_t>().swap(C.uimg);
            std::vector<uint8_t>().swap(C.uph);
        }
#endif
        ++st.kernel_compiles;
        st.max_r = std::max<long>(st.max_r, X.nG);
        st.max_kappa_prime = std::max<long>(st.max_kappa_prime, X.kprime);
        auto hist = [](std::vector<long>& h, int v) { if (h.size() <= (size_t)v) h.resize((size_t)v + 1, 0); ++h[(size_t)v]; };
        hist(st.r_hist, X.nG);
        hist(st.kappa_prime_hist, X.kprime);
        hist(st.dker_hist, X.KD.d);
        const size_t nb = X.bytes();
        C.nc = std::move(Np);
        charge(held_l3, nb, "a content's expansion structure");
        return *C.nc;
    }

    std::vector<uint64_t> cs_bu_, cs_bv_, cs_comb_, cs_L_;
    std::vector<uint8_t> cs_q_;
    std::vector<int> cs_bp_, cs_ord_;
    QForm cs_Q_;
    // Σ_{free logical assignments} i^{Qf} λ(L) in ascending (z_free, x_free) order; `cst` = Qf at the smallest L.
    std::complex<double> canonical_sum(const GaussSum& G, const Fib& F, uint64_t u0, uint64_t v0, int& cst) {
        const int nf = (int)G.free_keep.size();
        if (nf > kLogicalRankCap)
            throw std::runtime_error("RecordFidelity: RF_LOGICAL_RANK: a sector term reaches " + std::to_string(nf) +
                                     " free logical directions (2^" + std::to_string(nf) + " lookups; cap 2^" +
                                     std::to_string(kLogicalRankCap) + ") — refused");
        auto depL = [&](size_t d, uint64_t& u, uint64_t& v) { u ^= F.Lu[(size_t)G.dep_var[d]]; v ^= F.Lv[(size_t)G.dep_var[d]]; };
        const size_t WDm = G.dep_var.empty() ? 0 : G.dep_mask.size() / G.dep_var.size();
        std::vector<uint64_t>& bu = cs_bu_;
        std::vector<uint64_t>& bv = cs_bv_;
        std::vector<uint64_t>& comb = cs_comb_;
        std::vector<int>& bp = cs_bp_;
        bu.assign((size_t)nf, 0);
        bv.assign((size_t)nf, 0);
        comb.assign((size_t)nf, 0);
        bp.assign((size_t)nf, 0);
        for (int i = 0; i < nf; ++i) {
            const int m = G.free_keep[(size_t)i];
            uint64_t u = F.Lu[(size_t)m], v = F.Lv[(size_t)m];
            for (size_t d = 0; d < G.dep_var.size(); ++d)
                if ((G.dep_mask[d * WDm + (size_t)(m >> 6)] >> (m & 63)) & 1ULL) depL(d, u, v);
            uint64_t c = 1ULL << i;
            for (int r = 0; r < i; ++r) if (lbit(u, v, bp[(size_t)r])) { u ^= bu[(size_t)r]; v ^= bv[(size_t)r]; c ^= comb[(size_t)r]; }
            const int p = lhb(u, v);
            if (p < 0) throw std::logic_error("RecordFidelity: dependent logical directions in a fiber (internal)");
            for (int r = 0; r < i; ++r) if (lbit(bu[(size_t)r], bv[(size_t)r], p)) { bu[(size_t)r] ^= u; bv[(size_t)r] ^= v; comb[(size_t)r] ^= c; }
            bu[(size_t)i] = u; bv[(size_t)i] = v; comb[(size_t)i] = c; bp[(size_t)i] = p;
        }
        uint64_t ou = u0, ov = v0, f0 = 0;
#ifndef XTIM_MUTATE_RF_KAFF
        for (size_t d = 0; d < G.dep_var.size(); ++d) if (G.dep_c[d]) depL(d, ou, ov);
#endif                                                           // mutation build only: the affine offset dropped
        for (int r = 0; r < nf; ++r) if (lbit(ou, ov, bp[(size_t)r])) { ou ^= bu[(size_t)r]; ov ^= bv[(size_t)r]; f0 ^= comb[(size_t)r]; }
        std::vector<int>& ord = cs_ord_;
        ord.resize((size_t)nf);
        for (int i = 0; i < nf; ++i) ord[(size_t)i] = i;
#ifndef XTIM_MUTATE_RF_KBASIS
        std::sort(ord.begin(), ord.end(), [&](int a, int b2) { return bp[(size_t)a] < bp[(size_t)b2]; });
#endif                                                           // mutation build only: the canonical basis order
        // the form in c (f = f0 ⊕ Σ c_m comb_{ord m}); constant = Qf(f0)
        QForm& Qc = cs_Q_;
        Qc.init(nf);
        cst = G.f.eval(&f0);
        for (int m = 0; m < nf; ++m) {
            const uint64_t cm = comb[(size_t)ord[(size_t)m]];
            Qc.lin[(size_t)m] = (uint8_t)((G.f.eval(&cm) + 2 * G.f.polar(&f0, &cm)) & 3);
            for (int m2 = m + 1; m2 < nf; ++m2) {
                const uint64_t c2 = comb[(size_t)ord[(size_t)m2]];
                if (G.f.polar(&cm, &c2)) Qc.flip(m, m2);
            }
        }
        double re = 0.0, im = 0.0;
        const uint64_t total = 1ULL << nf;
        // c ascending; L and Q(c) carried incrementally: c → c + 1 flips the trailing bits, each flip j adding
        // lin_j + 2 B(c, e_j) to Q (exact Z4) and b_j to L
        uint64_t u = ou, v = ov, cc = 0;
        int q = 0;
#ifdef XTIM_MUTATE_RF_CANON
        for (uint64_t c = total; c-- > 0;) {                            // mutation build only: the order reversed
            u = ou; v = ov;
            for (uint64_t mm = c; mm; mm &= mm - 1) { const int m = __builtin_ctzll(mm); u ^= bu[(size_t)ord[(size_t)m]]; v ^= bv[(size_t)ord[(size_t)m]]; }
            const std::complex<double> l = lambda(u, v);
            switch (Qc.eval(&c) & 3) {
                case 0: re += l.real(); im += l.imag(); break;
                case 1: re -= l.imag(); im += l.real(); break;
                case 2: re -= l.real(); im -= l.imag(); break;
                default: re += l.imag(); im -= l.real(); break;
            }
        }
        return {re, im};
#endif
        // pass 1: the terms' (L, phase) in order (the λ rows filled, the table entries prefetched); pass 2: the sum
        cs_L_.resize((size_t)total * 2);
        cs_q_.resize((size_t)total);
        for (uint64_t c = 0; c < total; ++c) {
            if (c) {
                for (uint64_t fl = c ^ (c - 1); fl; fl &= fl - 1) {
                    const int m = __builtin_ctzll(fl);
                    const int bjm = ((Qc.lin[(size_t)m] & 1) & (int)((cc >> m) & 1)) ^
                                    (__builtin_popcountll(Qc.row(m)[0] & cc) & 1);
                    q += Qc.lin[(size_t)m] + 2 * bjm;
                    cc ^= 1ULL << m;
                    u ^= bu[(size_t)ord[(size_t)m]];
                    v ^= bv[(size_t)ord[(size_t)m]];
                }
            }
            cs_L_[(size_t)c * 2] = u;
            cs_L_[(size_t)c * 2 + 1] = v;
            cs_q_[(size_t)c] = (uint8_t)(q & 3);
            lambda_touch(u, v);
        }
        for (uint64_t c = 0; c < total; ++c) {
#ifdef XTIM_MUTATE_RF_INTH
            const std::complex<double> l = lambda(cs_L_[(size_t)c * 2], cs_L_[(size_t)c * 2 + 1]) * std::pow(2.0, -0.5 * inth_ec_);
#else
            const std::complex<double> l = lambda(cs_L_[(size_t)c * 2], cs_L_[(size_t)c * 2 + 1]);
#endif
            switch (cs_q_[(size_t)c]) {
                case 0: re += l.real(); im += l.imag(); break;
                case 1: re -= l.imag(); im += l.real(); break;
                case 2: re -= l.real(); im -= l.imag(); break;
                default: re += l.imag(); im -= l.real(); break;
            }
        }
        return {re, im};
    }
    // The fiber's value at e_s (nullptr: e = 0) with the shot's cross bits (WV words) and base L.
    struct HVal { bool zero = true; int h = 0, p = 0; std::complex<double> S{0.0, 0.0}; };
    QForm fv_R_;
    GaussSum fv_G_;
    std::vector<uint8_t> fv_keep_;
    HVal fiber_value(int WV, const Fib& F, const uint64_t* es, const uint64_t* cross, uint64_t u0, uint64_t v0) {
        HVal H;
        QForm& R = fv_R_;
        R.init(F.d);
        for (int m = 0; m < F.d; ++m) {
            const uint64_t* km = &F.vec[(size_t)m * WV];
            const uint64_t* bk = &F.Bk[(size_t)m * WV];
            int b = 0;
            if (es) for (int w = 0; w < WV; ++w) b ^= __builtin_popcountll(es[w] & bk[w]) & 1;   // B(e_s, k_m)
            for (int w = 0; w < WV; ++w) b ^= __builtin_popcountll(cross[w] & km[w]) & 1;
            R.lin[(size_t)m] = (uint8_t)((F.q[(size_t)m] + 2 * b) & 3);
            for (int w = 0; w < F.WD; ++w) R.row(m)[w] = F.B[(size_t)m * F.WD + w];
        }
        std::vector<uint8_t>& keep = fv_keep_;
        keep.assign((size_t)F.d, 0);
        for (int m = 0; m < F.kap; ++m) keep[(size_t)m] = 1;
        GaussSum& G = fv_G_;
        gauss_sum(R, keep, G);
        if (G.zero) return H;
        int cst = 0;
        H.S = canonical_sum(G, F, u0, v0, cst);
        H.zero = false;
        H.h = G.h;
        H.p = (G.p8 + 2 * cst) & 7;
        return H;
    }
    // Test-only oracle (S8 RF4, hform_oracle): a fiber's value by exact Gaussian-integer enumeration of its 2^d terms,
    // each term's phase from the EXPLICIT ordered product of its frame Paulis (independent of the form machinery), the
    // classes grouped by L; then the same canonical λ sum. A fiber above d = 12 marks the row skipped (class 255).
    using TermFn = std::function<void(const uint64_t* e, Pauli& Wp, int& coef)>;
    HVal fiber_oracle(const Fib& F, int WV, const uint64_t* es, const TermFn& term) {
        HVal H;
        if (F.d > 12) { oracle_skip_ = true; return H; }          // beyond the oracle: the row is logged as skipped
        std::map<std::pair<uint64_t, uint64_t>, std::pair<long, long>> Hm;   // (v, u) ascending
        std::vector<uint64_t> e((size_t)WV);
        Pauli Wp(n);
        for (uint64_t c = 0; c < (1ULL << F.d); ++c) {
            for (int w = 0; w < WV; ++w) e[(size_t)w] = es ? es[w] : 0;
            for (int m = 0; m < F.d; ++m)
                if ((c >> m) & 1) for (int w = 0; w < WV; ++w) e[(size_t)w] ^= F.vec[(size_t)m * WV + w];
            int coef = 0;
            term(e.data(), Wp, coef);
            for (int w = 0; w < W; ++w)
                if (Wp.x[(size_t)w] & ~freemask[(size_t)w]) throw std::logic_error("RecordFidelity: oracle term off the fiber (internal)");
            uint64_t xz[2];
            std::vector<uint64_t> bits((size_t)2 * W);
            for (int w = 0; w < W; ++w) { bits[(size_t)w] = Wp.x[(size_t)w]; bits[(size_t)(W + w)] = Wp.z[(size_t)w]; }
            lfree(bits.data(), xz[0], xz[1]);
            const int ph = (Wp.phase + 2 * parw(Wp.z.data(), eps.data(), W) + 2 * (__builtin_popcountll(xz[0] & xz[1]) & 1) + coef) & 3;
            auto& acc = Hm[{xz[1], xz[0]}];
            acc.first += ph == 0 ? 1 : ph == 2 ? -1 : 0;
            acc.second += ph == 1 ? 1 : ph == 3 ? -1 : 0;
        }
        long r0 = 0, i0 = 0, mag = 0;
        double re = 0.0, im = 0.0;
        for (const auto& kv : Hm) {
            const long hr = kv.second.first, hi = kv.second.second;
            if (hr == 0 && hi == 0) continue;
            if (mag == 0) { r0 = hr; i0 = hi; mag = hr * hr + hi * hi; H.zero = false; }
            if (hr * hr + hi * hi != mag) throw std::logic_error("RecordFidelity: oracle fiber of non-uniform magnitude (internal)");
            // unit u = H / H_first = H conj(H_first) / mag
            const long ur = hr * r0 + hi * i0, ui = hi * r0 - hr * i0;
            int q;
            if (ur == mag && ui == 0) q = 0; else if (ur == 0 && ui == mag) q = 1;
            else if (ur == -mag && ui == 0) q = 2; else if (ur == 0 && ui == -mag) q = 3;
            else throw std::logic_error("RecordFidelity: oracle fiber phase is not a unit (internal)");
            const std::complex<double> l = lambda(kv.first.second, kv.first.first);
            switch (q) {
                case 0: re += l.real(); im += l.imag(); break;
                case 1: re -= l.imag(); im += l.real(); break;
                case 2: re -= l.real(); im -= l.imag(); break;
                default: re += l.imag(); im -= l.real(); break;
            }
        }
        if (H.zero) return H;
        int h = 0;
        while ((1L << h) < mag) ++h;
        if ((1L << h) != mag) throw std::logic_error("RecordFidelity: oracle fiber magnitude is not a power of two (internal)");
        H.h = h;
        H.p = i0 == 0 ? (r0 > 0 ? 0 : 4) : r0 == 0 ? (i0 > 0 ? 2 : 6) :
              (r0 > 0 ? (i0 > 0 ? 1 : 7) : (i0 > 0 ? 3 : 5));
        H.S = {re, im};
        return H;
    }

    // ⟨ψ|ψ⟩ = 2^{−nU} Σ_{J ∈ ker} (−1)^{td·J} ⟨B|M'_J|B⟩ over the used records' dressed images (Scaled; zero: the record
    // is impossible — decided by the exact Gauss sum). In ascending L the first class is L = 0 (D ≠ 0 forces it nonzero:
    // the classes share one magnitude), so the canonical sum's unit is H(0)'s sign.
    std::vector<uint64_t> ncrossU_;
    Scaled norm_new(const L3Content& C, const NewC& X, const std::vector<uint8_t>& td) {
        ncrossU_.assign((size_t)X.WU, 0);
#ifndef XTIM_MUTATE_RFREF_TD
        for (int u = 0; u < X.nU; ++u) if (td[(size_t)C.use[(size_t)u]] & 1) gs::flipbit(ncrossU_.data(), u);
#endif                                                                  // mutation build only: the records' signs ignored
        HVal H;
        if (hform_oracle) {
            H = fiber_oracle(X.KD, X.WU, nullptr, [&](const uint64_t* e, Pauli& Wp, int& coef) {
                Wp = Pauli(n);
                coef = 0;
                for (int u = 0; u < X.nU; ++u)
                    if (gs::getbit(e, u)) {
                        Pauli R(n);
                        for (int w = 0; w < W; ++w) { R.x[(size_t)w] = C.uimg[(size_t)u * 2 * W + w]; R.z[(size_t)w] = C.uimg[(size_t)u * 2 * W + W + w]; }
                        R.phase = C.uph[(size_t)u];
                        Wp = Pauli::multiply(Wp, R);
                        coef += 2 * (td[(size_t)C.use[(size_t)u]] & 1);
                    }
            });
        }
        if (!hform_oracle || oracle_skip_) H = fiber_value(X.WU, X.KD, nullptr, ncrossU_.data(), 0, 0);   // (skipped row)
        Scaled D{0.0, 0, true};
        if (H.zero) return D;
        // the smallest class is L = 0 (the identity term J = 0 is in it): its stabilizer-direction sum is a sum of
        // ±1 (Hermitian products with no free part), a real integer — so the unit is ±1 (p ∈ {0, 4}), h even
        if ((H.p & 3) || (H.h & 1)) throw std::logic_error("RecordFidelity: <psi|psi>'s identity class is not real (internal)");
        D.m = H.p ? -H.S.real() : H.S.real();
        D.e = X.nU - H.h / 2;
        D.zero = false;
        return D;
    }

    // ── route 3b per shot. solve_new: the particular solution t_0 of the shot's constraints (false: INFEASIBLE, F = 0
    // exactly) and the coset representative's sector σ = σ_P0 ⊕ σ(t_0); key_new: the memo key's tail; value_new: F.
    std::vector<uint64_t> nsig_, nimD_, nrecD_, nrhs_, nt0_, nes_, nsg_, nim_, nrec_, ny_, ncross_, nacc_, npi_;
    bool solve_new(const L3Content& C, const NewC& X, const uint64_t* px, const uint64_t* pz, const uint64_t* y) {
        (void)C;
        img_bits(px, pz, npi_);                                         // P''s frame image (bits; its phase is global)
        nsig_.assign((size_t)WC, 0);
        syndrome_of(px, pz, nsig_.data());
        for (int w = 0; w < WC; ++w) nsig_[(size_t)w] ^= X.sig0[(size_t)w];
        nimD_.assign((size_t)2 * W, 0);
        nrecD_.assign((size_t)MW, 0);
        dest_product(nsig_.data(), nimD_.data(), nrecD_.data());
        nrhs_.assign((size_t)X.CW, 0);
        for (int w = 0; w < W; ++w)
            nrhs_[(size_t)w] = (npi_[(size_t)w] ^ X.I0[(size_t)w] ^ nimD_[(size_t)w]) & ~freemask[(size_t)w];
        for (int w = 0; w < MW; ++w) nrhs_[(size_t)(W + w)] = (y[w] ^ rssignw[(size_t)w] ^ nrecD_[(size_t)w]) & rsmask[(size_t)w];
        nt0_.assign((size_t)X.WV, 0);
        for (size_t r = 0; r < X.pbit.size(); ++r)
            if (gs::getbit(nrhs_.data(), X.pbit[r])) {
                for (int w = 0; w < X.CW; ++w) nrhs_[(size_t)w] ^= X.prow[r * X.CW + w];
                for (int w = 0; w < X.WV; ++w) nt0_[(size_t)w] ^= X.ppay[r * X.WV + w];
            }
        for (int w = 0; w < X.CW; ++w) if (nrhs_[(size_t)w]) return false;
        for (int a = 0; a < X.nG; ++a)
            if (gs::getbit(nt0_.data(), X.nM + a)) for (int w = 0; w < WC; ++w) nsig_[(size_t)w] ^= X.gsig[(size_t)a * WC + w];
        for (int r = 0; r < X.ksec; ++r)                                // the coset's smallest σ
            if (gs::getbit(nsig_.data(), X.spiv[(size_t)r])) {
                for (int w = 0; w < WC; ++w) nsig_[(size_t)w] ^= X.ssig[(size_t)r * WC + w];
                for (int w = 0; w < X.WV; ++w) nt0_[(size_t)w] ^= X.spay[(size_t)r * X.WV + w];
            }
        return true;
    }
    // A sector's frame quantities: σ (nsg_) → D_σ; Fm = D_σ · P' · G_0 (nim_, frame bits), y'' = y ⊕ rec(D_σ) (ny_), the
    // base L of e_s (u0, v0) and the fiber's cross bits (ncross_: x_Fm·z_j for a G variable, x_j·z_Fm ⊕ y''_c for a
    // Born record).
    void sector_frame(const NewC& X, const uint64_t* y, const uint64_t* es, uint64_t& u0, uint64_t& v0) {
        std::fill(nim_.begin(), nim_.end(), 0);
        std::fill(nrec_.begin(), nrec_.end(), 0);
        dest_product(nsg_.data(), nim_.data(), nrec_.data());
        for (int w = 0; w < MW; ++w)
#ifdef XTIM_MUTATE_RF_DFLIP
            ny_[(size_t)w] = y[w];                                      // mutation build only: D_σ's record flips dropped
#else
            ny_[(size_t)w] = y[w] ^ nrec_[(size_t)w];
#endif
        for (int w = 0; w < 2 * W; ++w) nim_[(size_t)w] ^= X.I0[(size_t)w];
        std::copy(nim_.begin(), nim_.end(), nacc_.begin());            // (Fm without P': the PSIGN mutation's cross)
        for (int w = 0; w < 2 * W; ++w) nim_[(size_t)w] ^= npi_[(size_t)w];
#ifdef XTIM_MUTATE_RFREF_PSIGN
        const uint64_t* fx = nacc_.data();                              // mutation build only: P''s sign action dropped
#else
        const uint64_t* fx = nim_.data();
#endif
        std::fill(ncross_.begin(), ncross_.end(), 0);
        for (int w = 0; w < X.WV; ++w)
            for (uint64_t mm = X.K0.supp[(size_t)w]; mm; mm &= mm - 1) {
                const int j = w * 64 + __builtin_ctzll(mm);
                const uint64_t* I = &X.I[(size_t)j * 2 * W];
                int b;
                if (j < X.nM) {
                    const int c = X.mrec[(size_t)j];
#ifdef XTIM_MUTATE_RF_BORNSIGN
                    b = parw(I, fx + W, W);                             // mutation build only: y'' dropped
                    (void)c;
#else
                    b = parw(I, fx + W, W) ^ (int)((ny_[(size_t)(c >> 6)] >> (c & 63)) & 1ULL);
#endif
                } else {
                    b = parw(fx, I + W, W);
                }
                if (b) gs::flipbit(ncross_.data(), j);
            }
        std::copy(nim_.begin(), nim_.end(), nacc_.begin());
        for (int w = 0; w < X.WV; ++w)
            for (uint64_t mm = es[w]; mm; mm &= mm - 1) {
                const uint64_t* I = &X.I[(size_t)(w * 64 + __builtin_ctzll(mm)) * 2 * W];
                for (int t = 0; t < 2 * W; ++t) nacc_[(size_t)t] ^= I[t];
            }
        lfree(nacc_.data(), u0, v0);
    }
    void sector_scratch(const NewC& X) {
        nes_.resize((size_t)X.WV);
        nsg_.resize((size_t)WC);
        nim_.resize((size_t)2 * W);
        nrec_.resize((size_t)MW);
        ny_.resize((size_t)MW);
        ncross_.resize((size_t)X.WV);
        nacc_.resize((size_t)2 * W);
    }
    // the memo key's tail (route 3b): feasible; t_0, σ, the sector-0 base L, cross bits and Born bits y'' on R_n
    void key_new(const NewC& X, bool feasible, const L3Content& C, const uint64_t* y) {
        std::string& key = key_scratch;
        key.push_back(feasible ? '\x01' : '\x00');
        if (!feasible) return;
        sector_scratch(X);
        std::copy(nsig_.begin(), nsig_.end(), nsg_.begin());
        uint64_t u0, v0;
        sector_frame(X, y, nt0_.data(), u0, v0);
#ifndef XTIM_MUTATE_RF_MEMOKEY
        key.append(reinterpret_cast<const char*>(nt0_.data()), nt0_.size() * 8);
        key.append(reinterpret_cast<const char*>(nsig_.data()), nsig_.size() * 8);
#endif                                                                  // mutation build only: t_0, σ (and td) dropped
        key.append(reinterpret_cast<const char*>(&u0), 8);
        key.append(reinterpret_cast<const char*>(&v0), 8);
        key.append(reinterpret_cast<const char*>(ncross_.data()), ncross_.size() * 8);
        for (size_t i = 0; i < C.rn.size(); i += 8) {
            uint8_t byte = 0;
            for (size_t j = i; j < C.rn.size() && j < i + 8; ++j) {
                const int c = C.rn[j];
                byte |= (uint8_t)(((ny_[(size_t)(c >> 6)] >> (c & 63)) & 1ULL) << (j - i));
            }
            key.push_back((char)byte);
        }
    }
    double value_new(L3Content& C, const NewC& X, const uint64_t* y, const std::vector<uint8_t>& td, int& cls) {
        const Scaled D = norm_new(C, X, td);
        if (D.zero) throw std::runtime_error("RecordFidelity: <psi|psi> = 0 for this record (impossible record)");
        // the impossible-record refusal: structural above (the exact Gauss sum); here the H-form mantissa (the one
        // non-structural test, as before: a nonzero exact value is never below 1e-300 at these scales)
        if (!(D.m > 1e-300)) throw std::runtime_error("RecordFidelity: <psi|psi> = 0 for this record (impossible record)");
        if (X.nM) ++st.rn_misses;
        if (X.nG) ++st.rotation_misses;
        double acc = 0.0;
        long born = 0, gauss = 0, val = 0;
#ifdef XTIM_MUTATE_RF_SECTOR0
        const uint64_t terms = 1;                                       // mutation build only: the kernel sectors dropped
#else
        const uint64_t terms = 1ULL << X.ksec;
#endif
        st.sectors += (long)terms;
        st.max_sectors = std::max<long>(st.max_sectors, (long)terms);
        sector_scratch(X);
        for (uint64_t step = 0; step < terms; ++step) {
            std::copy(nt0_.begin(), nt0_.end(), nes_.begin());
            std::copy(nsig_.begin(), nsig_.end(), nsg_.begin());
            for (int r = 0; r < X.ksec; ++r)
                if ((step >> r) & 1) {
                    for (int w = 0; w < X.WV; ++w) nes_[(size_t)w] ^= X.spay[(size_t)r * X.WV + w];
                    for (int w = 0; w < WC; ++w) nsg_[(size_t)w] ^= X.ssig[(size_t)r * WC + w];
                }
            uint64_t u0, v0;
            sector_frame(X, y, nes_.data(), u0, v0);
            const Scaled Ny = ref_ny(C, ny_.data(), td);
#ifdef XTIM_MUTATE_RF_NY_THRESH
            if (Ny.zero || std::ldexp(Ny.m, -Ny.e) <= 1e-12 * bnorm2) { ++born; continue; }   // mutation: a float threshold
#else
            if (Ny.zero) { ++born; continue; }                          // structural (GF(2) sign consistency, ref_ny)
#endif
            HVal H;
            if (hform_oracle) {
                const Expansion E = expansion_of(C);
                Pauli Fm(n);
                for (int w = 0; w < W; ++w) { Fm.x[(size_t)w] = nim_[(size_t)w]; Fm.z[(size_t)w] = nim_[(size_t)(W + w)]; }
                H = fiber_oracle(X.K0, X.WV, nes_.data(), [&](const uint64_t* e, Pauli& Wp, int& coef) {
                    Wp = Pauli(n);
                    coef = 0;
                    for (int j = 0; j < X.nM; ++j)
                        if (gs::getbit(e, j)) {
                            Wp = Pauli::multiply(Wp, img(recops[(size_t)X.mrec[(size_t)j]]));
                            const int c = X.mrec[(size_t)j];
                            coef += 2 * (int)((ny_[(size_t)(c >> 6)] >> (c & 63)) & 1ULL);
                        }
                    Wp = Pauli::multiply(Wp, Fm);
                    std::vector<uint64_t> t((size_t)std::max(1, (X.nG + 63) / 64), 0);
                    for (int a = 0; a < X.nG; ++a)
                        if (gs::getbit(e, X.nM + a)) {
                            Wp = Pauli::multiply(Wp, img(E.G[(size_t)a]));
                            gs::flipbit(t.data(), a);
                        }
                    coef += E.q.eval(t.data());
                });
            } else {
#ifdef XTIM_MUTATE_RF_INTH
                inth_ec_ = X.ec;
#endif
                H = fiber_value(X.WV, X.K0, nes_.data(), ncross_.data(), u0, v0);
#ifdef XTIM_MUTATE_RF_INTH
                inth_ec_ = 0;
#endif
            }
            if (H.zero) { ++gauss; continue; }
            ++val;
            ++st.sectors_nonzero;
#if defined(XTIM_MUTATE_RF_INTH)
            acc += std::ldexp(std::norm(H.S) / Ny.m / D.m, H.h - 2 * X.nM + Ny.e + D.e);   // (c applied in the λ sum)
#elif defined(XTIM_MUTATE_RF_UNDERFLOW)
            acc += std::ldexp(std::norm(H.S), H.h - X.ec - 2 * X.nM) / std::ldexp(Ny.m, -Ny.e) /
                   std::ldexp(D.m, -D.e);                               // mutation build only: the scales as floats
#else
            acc += std::ldexp(std::norm(H.S) / Ny.m / D.m, H.h - X.ec - 2 * X.nM + Ny.e + D.e);
#endif
        }
        cls = val ? kValue : (gauss ? kGaussZero : kBornZero);
        return acc;
    }
};

RecordFidelity::~RecordFidelity() = default;

RecordFidelity::RecordFidelity(const FramedSuperposition& bare, const std::vector<Pauli>& record_ops,
                               const std::vector<int>& out_wires, const std::vector<int>& read_meas, size_t max_bytes,
                               const std::vector<int>& wire_user)
    : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.n = bare.n();
    I.W = (I.n + 63) / 64;
    I.k = bare.k();
    I.max_bytes = max_bytes;
    I.check_dim(I.k, "the bare free register (k free rows)");
    I.U = bare.U;
    I.U.ensure_dual();
    I.eps.assign((size_t)I.W, 0);
    for (int a = 0; a < I.n; ++a) if (bare.eps[(size_t)a]) I.eps[(size_t)(a >> 6)] |= 1ull << (a & 63);
    I.freemask.assign((size_t)I.W, 0);
    I.freepos.assign((size_t)I.n, -1);
    I.T_free_ = bare.free;
    for (int d = 0; d < I.k; ++d) {
        const int a = bare.free[(size_t)d];
        I.freemask[(size_t)(a >> 6)] |= 1ull << (a & 63);
        I.freepos[(size_t)a] = d;
    }
    try {
        I.b.assign((size_t)1 << I.k, 0.0);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("RecordFidelity: allocation of the bare free register (" +
                                 std::to_string(16 * ((size_t)1 << I.k)) + " bytes, k = " + std::to_string(I.k) +
                                 ") failed — refused");
    }
    for (const auto& e : bare.entries()) {
        size_t idx = 0;
        for (int d = 0; d < I.k; ++d) if (key_bit(e.first, d)) idx |= (size_t)1 << d;
        I.b[idx] += e.second;
    }
    for (const auto& c : I.b) I.bnorm2 += std::norm(c);
    for (size_t y = 0; y < I.b.size(); ++y) if (I.b[y] != 0.0) I.bsupp.push_back((uint64_t)y);
    I.p_scratch = Pauli(I.n);
    const int n = I.n, W = I.W;
    for (const Pauli& M : record_ops) {
        if (M.n != n) throw std::invalid_argument("RecordFidelity: a record operator has the wrong width");
        if (!hermitian(M)) throw std::invalid_argument("RecordFidelity: a record operator is not Hermitian (internal)");
    }
    I.recops = record_ops;
    I.rsw_off.assign(1, 0);
    for (const Pauli& M : record_ops) {
        for (int w = 0; w < W; ++w)
            if (M.x[(size_t)w] | M.z[(size_t)w]) {
                I.rsw_w.push_back(w);
                I.rsw_x.push_back(M.x[(size_t)w]);
                I.rsw_z.push_back(M.z[(size_t)w]);
            }
        I.rsw_off.push_back((int)I.rsw_w.size());
    }
    const int m = (int)record_ops.size();
    I.MW = (m + 63) / 64;
    // J-b1 P12 (X1): R_s = the records whose frame image is Z-type on certified rows only (a stabilizer of |B⟩, up to
    // its sign i^ph (−1)^{b·eps}); the rest are R_n.
    I.rs.assign(record_ops.size(), 0);
    I.rs_sign.assign(record_ops.size(), 0);
    // A stabilizer of |B⟩ includes the α-certified ones (Z / X parts on the free register that the amplitudes make
    // definite): the test is <B|M|B> = ±1 exactly on the free register, with no X part on a certified row.
    for (size_t c = 0; c < record_ops.size(); ++c) {
        const Pauli Q = I.img(record_ops[c]);
        bool fixed_x = false;
        for (int w = 0; w < W && !fixed_x; ++w)
            if (Q.x[(size_t)w] & ~I.freemask[(size_t)w]) fixed_x = true;
        if (fixed_x) continue;
        const FOp o = I.compile(Q);
        std::complex<double> A = 0.0;
        for (size_t i = 0; i < I.b.size(); ++i)
            if (I.b[i] != 0.0) A += std::conj(I.b[i ^ o.xm]) * sgn(i & o.zm) * I.b[i];
        A *= o.c / I.bnorm2;
        if (std::abs(std::abs(A.real()) - 1.0) > 1e-12 || std::abs(A.imag()) > 1e-12) continue;
        I.rs[c] = 1;
        I.rs_sign[c] = (uint8_t)(A.real() < 0 ? 1 : 0);
    }
    I.rsmask.assign((size_t)I.MW, 0);
    I.rssignw.assign((size_t)I.MW, 0);
    for (int c = 0; c < m; ++c)
        if (I.rs[(size_t)c]) {
            I.rsmask[(size_t)(c >> 6)] |= 1ULL << (c & 63);
            if (I.rs_sign[(size_t)c]) I.rssignw[(size_t)(c >> 6)] |= 1ULL << (c & 63);
        }

    // ── the STRICT output contract (owner 2026-10-22; review (i), (iv)) ──
    auto refuse = [](const std::string& code, const std::string& why) {
        throw std::invalid_argument("RecordFidelity: " + code + ": " + why + " — refused");
    };
    if (out_wires.empty())
        refuse("RF_NO_OUTPUT", "the circuit declares no OUTPUT_QUBITS; the register fidelity scores the output register");
    I.omask.assign((size_t)W, 0);
    for (int q : out_wires) {
        if (q < 0 || q >= n) refuse("RF_NO_OUTPUT", "an output wire " + std::to_string(q) + " is out of range");
        if ((I.omask[(size_t)(q >> 6)] >> (q & 63)) & 1ULL)
            refuse("RF_NO_OUTPUT", "output wire " + std::to_string(q) + " is listed twice");
        I.omask[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    }
    I.rmask.assign((size_t)W, 0);
    for (int q = 0; q < n; ++q) if (!((I.omask[(size_t)(q >> 6)] >> (q & 63)) & 1ULL)) I.rmask[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    const int nO = (int)out_wires.size(), nR = n - nO;
    auto meas_of = [&](int q) { return q < (int)read_meas.size() ? read_meas[(size_t)q] : -1; };
    for (int c = 0; c < m; ++c) {                    // (iv): records live on the rest
        const Pauli& M = record_ops[(size_t)c];
        for (int w = 0; w < W; ++w)
            if ((M.x[(size_t)w] | M.z[(size_t)w]) & I.omask[(size_t)w]) {
                const int q = w * 64 + __builtin_ctzll((M.x[(size_t)w] | M.z[(size_t)w]) & I.omask[(size_t)w]);
                refuse("RF_RECORD_ON_OUTPUT", "record " + std::to_string(c) + " (storage column) acts on output " +
                       (q < (int)wire_user.size() && wire_user[(size_t)q] >= 0
                            ? "qubit " + std::to_string(wire_user[(size_t)q]) : "wire " + std::to_string(q)) +
                       "; records must live on the rest (ancillas and read wires)");
            }
    }
    // Theorem 1 on the unsigned bare stabilizer group S0 = ⟨U Z_a U† : a not free⟩ (no records, review (i)):
    // pure and outcome-independent ⟺ s_O = |O| or s_R = |R|, s_X = dim(S0 ∩ P_X).
    std::vector<std::vector<uint64_t>> G;
    for (int a = 0; a < n; ++a) if (I.freepos[(size_t)a] < 0) G.push_back(bits_of(I.U.Zrow[(size_t)a], W));
#ifdef XTIM_MUTATE_RF_STRICT_SY
    for (int c = 0; c < m; ++c) G.push_back(bits_of(record_ops[(size_t)c], W));   // mutation: Theorem 1 on S_y
#endif
    std::vector<uint64_t> Rm((size_t)2 * W), Om((size_t)2 * W);
    for (int w = 0; w < W; ++w) { Rm[(size_t)w] = Rm[(size_t)(W + w)] = I.rmask[(size_t)w]; Om[(size_t)w] = Om[(size_t)(W + w)] = I.omask[(size_t)w]; }
    MaskElim ER(W, Rm), EO(W, Om);
    for (const auto& g : G) { ER.add(g); EO.add(g); }
    const int sO = (int)ER.zero.size(), sR = (int)EO.zero.size();
    bool strict = sO == nO || sR == nR;
    // (a, b, c, e) (investigation §1): c = dim rad(S0|_O) − s_O, 2a + c = 2|O| − rank(S0|_O) − s_O, likewise b on R
    auto side = [&](const MaskElim& E, int s, int nside, int& c, int& ab) {
        const int dV = (int)E.rows.size();
        std::vector<std::vector<uint64_t>> gram((size_t)dV, std::vector<uint64_t>((size_t)((dV + 63) / 64), 0));
        for (int i = 0; i < dV; ++i)
            for (int j = 0; j < dV; ++j) {
                const auto& u = E.rows[(size_t)i].v; const auto& v = E.rows[(size_t)j].v;
                int p = 0;
                for (int w = 0; w < W; ++w) p ^= __builtin_popcountll((u[(size_t)w] & v[(size_t)(W + w)]) ^ (u[(size_t)(W + w)] & v[(size_t)w])) & 1;
                if (p) gram[(size_t)i][(size_t)(j >> 6)] |= 1ULL << (j & 63);
            }
        c = dV - gf2_rank(gram) - s;
        const int dimG = 2 * nside - dV - s;
        ab = (dimG - c) / 2;
    };
    int cO = 0, a = 0, cR = 0, bb = 0;
    side(EO, sO, nO, cO, a);       // EO.rows: S0 restricted to O
    side(ER, sR, nR, cR, bb);      // ER.rows: S0 restricted to R
    const int e = nO - sO - a - cO;
    bool route3a = sO == nO;
#ifdef XTIM_MUTATE_RF_STRICT_C
    if (!strict && e == 0 && (a == 0 || bb == 0)) strict = route3a = true;   // mutation: the c > 0 obstruction ignored
#endif
    if (!strict) {
        // the cross-cut generators: S0 modulo (S0 ∩ P_O) ⊕ (S0 ∩ P_R) — their O parts are the undetermined directions,
        // their R parts sit on the read wires of the measurements whose outcomes fix them
        std::vector<uint64_t> All((size_t)2 * W, ~0ULL);
        MaskElim EX(W, All);
        for (const auto& z : ER.zero) EX.add(z);
        for (const auto& z : EO.zero) EX.add(z);
        std::vector<std::vector<uint64_t>> cross;
        for (const auto& g : G) if (EX.add(g)) cross.push_back(g);
        // a deferred wire in the user's terms: its qubit (an output wire always carries one), else "internal wire d"
        auto user_of = [&](int q) { return q < (int)wire_user.size() ? wire_user[(size_t)q] : -1; };
        auto label = [&](int q) {
            return user_of(q) >= 0 ? std::to_string(user_of(q))
                                   : wire_user.empty() ? std::to_string(q) : "(internal wire " + std::to_string(q) + ")";
        };
        auto pstr = [&](const std::vector<uint64_t>& f, const std::vector<uint64_t>& mask) {
            std::vector<std::pair<long, std::string>> fs;   // sorted by the user's qubit (internal wires last)
            for (int q = 0; q < n; ++q) {
                if (!((mask[(size_t)(q >> 6)] >> (q & 63)) & 1ULL)) continue;
                const bool x = (f[(size_t)(q >> 6)] >> (q & 63)) & 1ULL, z = (f[(size_t)(W + (q >> 6))] >> (q & 63)) & 1ULL;
                if (!x && !z) continue;
                fs.push_back({user_of(q) >= 0 ? (long)user_of(q) : (1L << 40) + q,
                              std::string(x && z ? "Y" : x ? "X" : "Z") + label(q)});
            }
            std::sort(fs.begin(), fs.end());
            std::string s;
            for (const auto& t : fs) s += (s.empty() ? "" : "*") + t.second;
            return s;
        };
        std::vector<int> meas, bare_wires;
        std::vector<uint8_t> seen((size_t)n, 0);
        for (const auto& g : cross)
            for (int q = 0; q < n; ++q) {
                if (!((I.rmask[(size_t)(q >> 6)] >> (q & 63)) & 1ULL) || seen[(size_t)q]) continue;
                if (!(((g[(size_t)(q >> 6)] | g[(size_t)(W + (q >> 6))]) >> (q & 63)) & 1ULL)) continue;
                seen[(size_t)q] = 1;
                if (meas_of(q) >= 0) meas.push_back(meas_of(q)); else bare_wires.push_back(q);
            }
        std::sort(meas.begin(), meas.end());
        auto list = [](const std::vector<int>& v) {
            std::string s;
            for (size_t i = 0; i < v.size() && i < 24; ++i) s += (i ? ", " : "") + std::to_string(v[i]);
            if (v.size() > 24) s += ", ... (" + std::to_string(v.size()) + " in all)";
            return "[" + s + "]";
        };
        std::string dirs;
        for (size_t i = 0; i < cross.size() && i < 6; ++i) dirs += (i ? ", " : "") + pstr(cross[i], I.omask);
        if (cross.size() > 6) dirs += ", ... (" + std::to_string(cross.size()) + " in all)";
        std::string why = "the noiseless state of the OUTPUT_QUBITS register (" + std::to_string(nO) +
            (nO == 1 ? " qubit) is " : " qubits) is ") +
            "not pure and fixed independently of the measurement outcomes.";
        if (!cross.empty()) {
            std::string wires;
            for (size_t i = 0; i < bare_wires.size() && i < 24; ++i)
                wires += (i ? ", " : "") + (user_of(bare_wires[i]) >= 0 ? "q" + std::to_string(user_of(bare_wires[i]))
                                                                         : "internal wire " + std::to_string(bare_wires[i]));
            if (bare_wires.size() > 24) wires += ", ... (" + std::to_string(bare_wires.size()) + " in all)";
            why += " Undetermined output directions (Paulis on your qubits): " + dirs + ". They are fixed by the outcomes "
                   "of measurements " + list(meas) + " (0-based record indices, in circuit order)" +
                   (bare_wires.empty() ? std::string() : " and by the unmeasured qubits [" + wires + "]") +
                   ": close the frame with Pauli feedback from those records onto the output (CX / CZ / CY rec[-k] q), "
                   "as for a deterministic Stim detector, or measure the entangled qubits.";
        } else {
            why += " The output and the rest each carry logical qubits (a, b > 0): the output is pure only for special "
                   "logical states, which the strict contract does not certify.";
        }
        why += " (Diagnostic, Theorem 1 on the bare stabilizer group S0: s_O = " + std::to_string(sO) + "/" +
               std::to_string(nO) + ", s_R = " + std::to_string(sR) + "/" + std::to_string(nR) + "; a = " +
               std::to_string(a) + ", b = " + std::to_string(bb) + ", c = " + std::to_string(cO) + ", e = " +
               std::to_string(e) + ": e stabilizer directions entangled across the cut, c logical directions fixed only "
               "together with the rest, a / b logical qubits on the output / the rest)";
        refuse("RF_OUTPUT_NOT_STRICT", why);
    }
    I.st.out_a = a;
    if (route3a) {
        // 3a: S* = S0 ∩ P_O signed +1 on |B⟩ (its frame image is ±Z on certified rows: the sign i^ph (−1)^{z·eps})
        I.route = 1;
        for (const auto& z : ER.zero) {
            Pauli g = pauli_of(z, n, W);
            const Pauli Q = I.img(g);
            for (int w = 0; w < W; ++w)
                if (Q.x[(size_t)w] || (Q.z[(size_t)w] & I.freemask[(size_t)w]) || (Q.phase & 1))
                    throw std::logic_error("RecordFidelity: an output stabilizer is not certified on |B> (internal)");
            const int neg = ((Q.phase >> 1) & 1) ^ parw(Q.z.data(), I.eps.data(), W);
            if (neg) g.phase = (g.phase + 2) & 3;
            I.star_img0.push_back(neg ? with_sign(Q, 1) : Q);
            I.star.push_back(std::move(g));
        }
    } else {
        // 3b: |B⟩ = |φ⟩ ⊗ |r⟩, r the stabilizer state of S(r) = S0 ∩ P_R (|R| generators). S_χ = C ⊕ ⟨independent
        // R_n records⟩, C = {g ∈ S(r) : g commutes with every R_n record}; rest destabilizers of S_χ.
        I.route = 2;
        I.invrows.assign((size_t)n * 4 * W, 0);
        I.invph.assign((size_t)n * 2, 0);
        for (int q = 0; q < n; ++q) {
            Pauli X1(n), Z1(n);
            X1.setx(q);
            Z1.setz(q);
            const Pauli a = I.img(X1), c = I.img(Z1);
            I.invph[(size_t)q * 2] = (uint8_t)(a.phase & 3);
            I.invph[(size_t)q * 2 + 1] = (uint8_t)(c.phase & 3);
            uint64_t* r = &I.invrows[(size_t)q * 4 * W];
            for (int w = 0; w < W; ++w) { r[w] = a.x[(size_t)w]; r[W + w] = a.z[(size_t)w]; r[2 * W + w] = c.x[(size_t)w]; r[3 * W + w] = c.z[(size_t)w]; }
        }
        std::vector<int> rn;
        for (int c = 0; c < m; ++c) if (!I.rs[(size_t)c]) rn.push_back(c);
        const int nrn = (int)rn.size(), RNW = (nrn + 63) / 64;
        // C: the combinations of S(r) generators with zero commutation vector against R_n
        std::vector<std::vector<uint64_t>> Cb;
        {
            struct Rw { std::vector<uint64_t> v, f; int piv; };
            std::vector<Rw> rows;
            for (const auto& g : EO.zero) {
                std::vector<uint64_t> v((size_t)std::max(RNW, 1), 0), f = g;
                const Pauli gp = pauli_of(g, n, W);
                for (int i = 0; i < nrn; ++i)
                    if (sympl(record_ops[(size_t)rn[(size_t)i]].x.data(), record_ops[(size_t)rn[(size_t)i]].z.data(), gp, W))
                        v[(size_t)(i >> 6)] |= 1ULL << (i & 63);
                for (const Rw& r : rows)
                    if ((v[(size_t)(r.piv >> 6)] >> (r.piv & 63)) & 1ULL) {
                        for (size_t w = 0; w < v.size(); ++w) v[w] ^= r.v[w];
                        for (size_t w = 0; w < f.size(); ++w) f[w] ^= r.f[w];
                    }
                int p = -1;
                for (size_t w = 0; w < v.size() && p < 0; ++w) if (v[w]) p = (int)w * 64 + __builtin_ctzll(v[w]);
                if (p < 0) { Cb.push_back(std::move(f)); continue; }
                for (Rw& r : rows)
                    if ((r.v[(size_t)(p >> 6)] >> (p & 63)) & 1ULL) {
                        for (size_t w = 0; w < v.size(); ++w) r.v[w] ^= v[w];
                        for (size_t w = 0; w < f.size(); ++w) r.f[w] ^= f[w];
                    }
                rows.push_back(Rw{std::move(v), std::move(f), p});
            }
        }
        I.dC = (int)Cb.size();
        // S_χ generators: C, then the R_n records independent of C and of each other (earliest first)
        std::vector<std::vector<uint64_t>> gens = Cb;
        {
            std::vector<uint64_t> All((size_t)2 * W, ~0ULL);
            MaskElim EI(W, All);
            for (const auto& g : Cb) EI.add(g);
            for (int c : rn) {
                const auto f = bits_of(record_ops[(size_t)c], W);
                if (EI.add(f)) { gens.push_back(f); I.rn_ind.push_back(c); }
            }
        }
        if ((int)gens.size() != nR)
            throw std::logic_error("RecordFidelity: the rest stabilizer group has " + std::to_string(gens.size()) +
                                   " generators on " + std::to_string(nR) + " rest wires (internal)");
        // destabilizers: rows h_j = (g_j.z | g_j.x) so that <v, g_j> = v · h_j; [H | T] → RREF, d_i = Σ_j T_ji E_{p_j}
        const int TW = (nR + 63) / 64;
        std::vector<std::vector<uint64_t>> H((size_t)nR), T((size_t)nR, std::vector<uint64_t>((size_t)TW, 0));
        for (int j = 0; j < nR; ++j) {
            H[(size_t)j].assign((size_t)2 * W, 0);
            for (int w = 0; w < W; ++w) {
                H[(size_t)j][(size_t)w] = gens[(size_t)j][(size_t)(W + w)] & I.rmask[(size_t)w];
                H[(size_t)j][(size_t)(W + w)] = gens[(size_t)j][(size_t)w] & I.rmask[(size_t)w];
            }
            T[(size_t)j][(size_t)(j >> 6)] |= 1ULL << (j & 63);
        }
        std::vector<int> pc((size_t)nR, -1);
        {
            int r = 0;
            for (int col = 0; col < 2 * W * 64 && r < nR; ++col) {
                int p = -1;
                for (int i = r; i < nR; ++i) if ((H[(size_t)i][(size_t)(col >> 6)] >> (col & 63)) & 1ULL) { p = i; break; }
                if (p < 0) continue;
                std::swap(H[(size_t)p], H[(size_t)r]);
                std::swap(T[(size_t)p], T[(size_t)r]);
                for (int i = 0; i < nR; ++i)
                    if (i != r && ((H[(size_t)i][(size_t)(col >> 6)] >> (col & 63)) & 1ULL)) {
                        for (int w = 0; w < 2 * W; ++w) H[(size_t)i][(size_t)w] ^= H[(size_t)r][(size_t)w];
                        for (int w = 0; w < TW; ++w) T[(size_t)i][(size_t)w] ^= T[(size_t)r][(size_t)w];
                    }
                pc[(size_t)r] = col;
                ++r;
            }
            if (r != nR) throw std::logic_error("RecordFidelity: dependent rest stabilizers (internal)");
        }
        I.WC = std::max(1, (I.dC + 63) / 64);
        I.dimg.assign((size_t)I.dC * 2 * W, 0);
        I.drec.assign((size_t)I.dC * I.MW, 0);
        for (int i = 0; i < I.dC; ++i) {
            Pauli d(n);
            for (int j = 0; j < nR; ++j)
                if ((T[(size_t)j][(size_t)(i >> 6)] >> (i & 63)) & 1ULL) {
                    const int col = pc[(size_t)j];
                    if (col < 64 * W) d.setx(col); else d.setz(col - 64 * W);
                }
            const Pauli Q = I.img(d);
            for (int w = 0; w < W; ++w) {
                I.dimg[(size_t)i * 2 * W + w] = Q.x[(size_t)w];
                I.dimg[(size_t)i * 2 * W + W + w] = Q.z[(size_t)w];
            }
            for (int c = 0; c < m; ++c)
                if (sympl(d.x.data(), d.z.data(), record_ops[(size_t)c], W))
                    I.drec[(size_t)i * I.MW + (c >> 6)] |= 1ULL << (c & 63);
        }
        I.colx.assign((size_t)n * I.WC, 0);
        I.colz.assign((size_t)n * I.WC, 0);
        for (int i = 0; i < I.dC; ++i) {
            const auto& g = gens[(size_t)i];
            for (int q = 0; q < n; ++q) {
                if (!((I.rmask[(size_t)(q >> 6)] >> (q & 63)) & 1ULL)) continue;
                if ((g[(size_t)(W + (q >> 6))] >> (q & 63)) & 1ULL) I.colx[(size_t)q * I.WC + (i >> 6)] |= 1ULL << (i & 63);
                if ((g[(size_t)(q >> 6)] >> (q & 63)) & 1ULL) I.colz[(size_t)q * I.WC + (i >> 6)] |= 1ULL << (i & 63);
            }
        }
    }
    I.st.route = I.route;
    I.held_rest = (I.dimg.capacity() + I.drec.capacity() + I.colx.capacity() + I.colz.capacity() +
                   I.invrows.capacity()) * 8 +
                  (I.star.size() + I.star_img0.size()) * (2 * (size_t)W * 8 + sizeof(Pauli));
    I.charge(I.held_l3, I.held_rest, "the register's rest structure");
}
// The content's record images (dressed by C0 where it touches them) and its R_n / used / undressed split; ux / uz:
// the union X / Z support of C0's factor terms (an operator missing it commutes with C0: undressed, no pull-back).
int RecordFidelity::Impl::intern_(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                                  const std::vector<uint64_t>& ux, const std::vector<uint64_t>& uz,
                                  std::vector<RecordFidelity::Factor>&& fac, std::unique_ptr<DiagExpansion>&& dex) {
    Impl& I = *this;
    Impl::L3Content C;
    auto touches = [&](const Pauli& Q) {
        for (int w = 0; w < I.W; ++w)
            if ((Q.x[(size_t)w] & (uz[(size_t)w] | ux[(size_t)w])) || (Q.z[(size_t)w] & ux[(size_t)w])) return true;
        return false;
    };
    Pauli pb(I.n);
    auto dressed = [&](const Pauli& Q) {
        if (!touches(Q)) return false;
        pb = pull(Q);
        return !(((pb.phase ^ Q.phase) & 3) == 0 && pb.x == Q.x && pb.z == Q.z);
    };
    auto pbytes = [](const Pauli& P) { return (P.x.capacity() + P.z.capacity()) * 8 + sizeof(Pauli); };
    if (I.rec_lab_img.empty() && !I.recops.empty()) {
        size_t m = 0;
        for (const Pauli& M : I.recops) { I.rec_lab_img.push_back(I.img(M)); m += pbytes(I.rec_lab_img.back()); }
        I.charge(I.held_l3, m, "the record operators' frame images");
    }
    if (I.route == 2) {
        // 3b: the used records (R_n, and the R_s records C0 dresses) with their frame images, flat (C.uimg / C.uph) in
        // record order; kind 0 dresses by the exact Lemma 1 (D† M D = i^{p − q(x)} X^x Z^{z ⊕ Ãx}, dress_diag) and
        // images by the per-wire rows, a factor list by its pull-back
        std::vector<uint64_t> zz((size_t)I.W), im((size_t)2 * I.W);
        for (size_t c = 0; c < I.recops.size(); ++c) {
            const Pauli& M = I.recops[c];
            bool t = false;
            int ph = M.phase & 3;
            if (dex) {
                const DiagExpansion& E = *dex;
                for (int i = I.rsw_off[c]; i < I.rsw_off[c + 1] && !t; ++i)
                    if (I.rsw_x[(size_t)i] & uz[(size_t)I.rsw_w[(size_t)i]]) t = true;
                if (t) {
                    for (int w = 0; w < I.W; ++w) zz[(size_t)w] = M.z[(size_t)w];
                    for (int i = I.rsw_off[c]; i < I.rsw_off[c + 1]; ++i)
                        for (uint64_t mm = I.rsw_x[(size_t)i]; mm; mm &= mm - 1) {
                            const int q = I.rsw_w[(size_t)i] * 64 + __builtin_ctzll(mm);
                            if (E.a1[(size_t)q]) zz[(size_t)(q >> 6)] ^= 1ULL << (q & 63);
                            const int ri = E.rowof[(size_t)q];
                            if (ri < 0) continue;
                            const uint64_t* row = &E.arows[(size_t)ri * I.W];
                            for (int v = 0; v < I.W; ++v) zz[(size_t)v] ^= row[v];
                        }
                    ph = (M.phase - E.qform(M.x.data())) & 3;
                    t = ph != (M.phase & 3);
                    for (int w = 0; w < I.W && !t; ++w) if (zz[(size_t)w] != M.z[(size_t)w]) t = true;
                }
                if (t) I.img_full(M.x.data(), zz.data(), ph, im.data(), ph);
            } else if (dressed(M)) {
                t = true;
                const Pauli Q = I.img(pb);
                for (int w = 0; w < I.W; ++w) { im[(size_t)w] = Q.x[(size_t)w]; im[(size_t)(I.W + w)] = Q.z[(size_t)w]; }
                ph = Q.phase & 3;
            }
            const bool rn = !I.rs[c];
#ifdef XTIM_MUTATE_RF_RS_ALL_UNDRESSED
            const bool used = rn;                                      // mutation build only: dressed R_s as δ checks
#else
            const bool used = rn || t;
#endif
            if (!used) { C.undressed.push_back((int)c); continue; }
            C.use.push_back((int)c);
            if (rn) C.rn.push_back((int)c);
            if (t) {
                C.uimg.insert(C.uimg.end(), im.begin(), im.end());
                C.uph.push_back((uint8_t)ph);
            } else {
                const Pauli& L = I.rec_lab_img[c];
                C.uimg.insert(C.uimg.end(), L.x.begin(), L.x.end());
                C.uimg.insert(C.uimg.end(), L.z.begin(), L.z.end());
                C.uph.push_back((uint8_t)(L.phase & 3));
            }
        }
    } else {
        C.rec_img.resize(I.recops.size());
        std::vector<uint8_t> rec_dressed(I.recops.size(), 0);
        for (size_t c = 0; c < I.recops.size(); ++c) {
            rec_dressed[c] = dressed(I.recops[c]) ? 1 : 0;
            if (rec_dressed[c]) C.rec_img[c] = I.img(pb);
            else if (!I.rs[c]) C.rec_img[c] = I.rec_lab_img[c];
        }
        for (size_t c = 0; c < I.recops.size(); ++c) {       // J-b1 P12 (X1): used / R_n / undressed R_s
            if (!I.rs[c]) { C.use.push_back((int)c); C.rn.push_back((int)c); continue; }
#ifdef XTIM_MUTATE_RF_RS_ALL_UNDRESSED
            C.undressed.push_back((int)c);       // mutation build only: dressed R_s records treated as δ checks
#else
            if (!rec_dressed[c]) C.undressed.push_back((int)c);
            else C.use.push_back((int)c);
#endif
        }
    }
    if (I.route == 1)                                    // 3a: S*'s images under C0 (exact sign)
        for (size_t i = 0; i < I.star.size(); ++i)
            C.star_img.push_back(dressed(I.star[i]) ? I.img(pb) : I.star_img0[i]);
    if (I.route == 2) {                                  // the expansion's input (built on the first 3b row)
        C.fac = std::move(fac);
        C.dex = std::move(dex);
    }
    const int id = (int)I.l3c.size();
    const size_t cb = C.bytes() + key.capacity() + 64;
    I.l3c.push_back(std::move(C));
    I.l3_ids.emplace(key, id);
    ++I.st.plans;
    I.charge(I.held_l3, cb, "a content");
    return id;
}
int RecordFidelity::intern_content(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                                   const std::vector<Factor>& factors) {
    Impl& I = *impl_;
    auto it = I.l3_ids.find(key);
    if (it != I.l3_ids.end()) return it->second;
    std::vector<Pauli> P;                                // the factor list's shape (RF_FACTOR_NOT_CLIFFORD by name)
    std::vector<uint8_t> cl;
    std::vector<std::pair<int, int>> cb;
    I.parse_factors(factors, P, cl, cb);
    std::vector<uint64_t> ux((size_t)I.W, 0), uz((size_t)I.W, 0);
    for (const Factor& f : factors)
        for (const auto& t : f)
            for (int w = 0; w < I.W; ++w) { ux[(size_t)w] |= t.second.x[(size_t)w]; uz[(size_t)w] |= t.second.z[(size_t)w]; }
    return I.intern_(key, pull, ux, uz, std::vector<Factor>(factors), nullptr);
}
int RecordFidelity::intern_content_diag(const std::string& key, const DiagNormalForm& nf) {
    Impl& I = *impl_;
    auto it = I.l3_ids.find(key);
    if (it != I.l3_ids.end()) return it->second;
    auto dex = std::make_unique<DiagExpansion>(expand_diag(nf, I.n));   // RF_CONTENT_MALFORMED
    std::vector<uint64_t> ux((size_t)I.W, 0), uz((size_t)I.W, 0);       // C0 = S^{a&1}·CZ touches X on its wires
    for (int q = 0; q < I.n; ++q)
        if (dex->a1[(size_t)q] || dex->rowof[(size_t)q] >= 0) uz[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    const DiagExpansion& E = *dex;
    return I.intern_(key, [&](const Pauli& Q) { return dress_diag(E, Q); }, ux, uz, {}, std::move(dex));
}
bool RecordFidelity::has_content(const std::string& key, int* id) const {
    auto it = impl_->l3_ids.find(key);
    if (it == impl_->l3_ids.end()) return false;
    if (id) *id = it->second;
    return true;
}
double RecordFidelity::evaluate_record(int content, const uint64_t* px, const uint64_t* pz, const uint64_t* y,
                                       const uint64_t* qx, const uint64_t* qz) {
    return impl_->evaluate_record(content, px, pz, y, qx, qz);
}
int RecordFidelity::route() const { return impl_->route; }

void RecordFidelity::clear() { impl_->clear(); }
void RecordFidelity::clear_memo() { impl_->clear_memo(); }
bool RecordFidelity::trim_cache(size_t budget) {
    Impl& I = *impl_;
    // the droppable bytes only: the register's rest structure (held_rest) is never dropped, so it never triggers a flush
    if (!budget || I.held_memo + I.held_l3 - I.held_rest <= budget) return false;
#ifndef XTIM_MUTATE_CB_RF_NOCLEAR
    I.clear();
#endif   // mutation build only (MUT-CB-RF-NOCLEAR): counted, nothing dropped
    ++I.st.cache_flushes;
    return true;
}
std::complex<double> RecordFidelity::lambda_for_test(uint64_t u, uint64_t v, bool direct) {
    Impl& I = *impl_;
    if (I.k > 62 || (I.k < 64 && ((u >> I.k) || (v >> I.k))))
        throw std::invalid_argument("RecordFidelity: lambda index outside the free register");
    return direct ? I.lambda_direct(u, v) : I.lambda(u, v);
}
void RecordFidelity::set_lambda_budget_for_test(size_t bytes) {
    Impl& I = *impl_;
    if (I.lam_mode != 0) throw std::logic_error("RecordFidelity: the lambda memo is already built (set the budget first)");
    I.lam_budget = bytes;
}
int RecordFidelity::lambda_mode_for_test() const { return impl_->lam_mode; }
void RecordFidelity::set_hform_oracle_for_test(bool on) { impl_->hform_oracle = on; }
void RecordFidelity::set_row_log_for_test(bool on) { impl_->row_log_on = on; }
std::vector<uint8_t> RecordFidelity::take_row_log_for_test() {
    std::vector<uint8_t> out;
    out.swap(impl_->row_log);
    return out;
}
size_t RecordFidelity::bytes_held() const { return impl_->bytes_held(); }
size_t RecordFidelity::memo_bytes() const {
    size_t m = 0;
    for (const auto& kv : impl_->memo) m += impl_->memo_entry_bytes(kv.first);
    return m;
}
static bool eq_pauli(const Pauli& a, const Pauli& b) {
    return a.n == b.n && (a.phase & 3) == (b.phase & 3) && a.x == b.x && a.z == b.z;
}
bool RecordFidelity::same_record_model(const FramedSuperposition& bare, const std::vector<Pauli>& record_ops) const {
    // exact (bit-for-bit) equality of the state model: bare state (frame rows, signs, free rows, amplitudes) and the
    // record operators; a buffer is accepted only from a sampler with the same model
    const Impl& I = *impl_;
    if (record_ops.size() != I.recops.size()) return false;
    for (size_t c = 0; c < record_ops.size(); ++c) if (!eq_pauli(record_ops[c], I.recops[c])) return false;
    if (bare.n() != I.n || bare.k() != I.k || bare.free != I.T_free_) return false;
    for (int a = 0; a < I.n; ++a) {
        if (!eq_pauli(bare.U.Xrow[(size_t)a], I.U.Xrow[(size_t)a]) || !eq_pauli(bare.U.Zrow[(size_t)a], I.U.Zrow[(size_t)a]))
            return false;
        if ((bare.eps[(size_t)a] & 1) != (int)((I.eps[(size_t)(a >> 6)] >> (a & 63)) & 1)) return false;
    }
    std::vector<std::complex<double>> bb((size_t)1 << I.k, 0.0);
    for (const auto& e : bare.entries()) {
        size_t idx = 0;
        for (int d = 0; d < I.k; ++d) if (key_bit(e.first, d)) idx |= (size_t)1 << d;
        bb[idx] += e.second;
    }
    return bb == I.b;   // bit-exact amplitudes
}
const RecordFidelityStats& RecordFidelity::stats() const { return impl_->st; }
int RecordFidelity::n() const { return impl_->n; }
}  // namespace qeccore
