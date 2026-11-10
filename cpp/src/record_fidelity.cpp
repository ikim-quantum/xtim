// record_fidelity — see the header and the design note (scratchpad fid/design.md) for the
// derivation. Additive translation unit: no sampling path includes or calls it.

#include "qeccore/record_fidelity.hpp"
#include "qeccore/state_budget.hpp"
#include "qeccore/thread_pool.hpp"

#include <atomic>
#include <cfenv>
#include <exception>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <array>
#include <unordered_set>
#include <map>
#include <mutex>
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

namespace {
// scaling F2: a content key with its hash carried (computed once, by whoever builds the key — in parallel on the
// evaluate's fan-out): the index never rehashes a key string.
struct HKey {
    std::string k;
    size_t h;
    const std::string* probe = nullptr;      // a lookup's key (not copied); a stored key has none
    const std::string& key() const { return probe ? *probe : k; }
};
struct HKeyHash {
    size_t operator()(const HKey& x) const { return x.h; }
};
struct HKeyEq {
    bool operator()(const HKey& a, const HKey& b) const {
        return a.h == b.h && (&a.key() == &b.key() || a.key() == b.key());   // (an eviction probes with the stored key)
    }
};
inline size_t key_hash_(const std::string& k) { return std::hash<std::string>{}(k); }
// speed fix 4 (final round): a content's memo key list — two inline, the rest in a vector (most contents hold one or
// two entries: no allocation, and nothing to free at a flush)
struct KeyList {
    const std::string* in[2] = {nullptr, nullptr};
    uint32_t n = 0;
    std::vector<const std::string*> more;
    void clear() { n = 0; more.clear(); }
    void push_back(const std::string* k) { if (n < 2) in[n] = k; else more.push_back(k); ++n; }
    template <class F> void each(F&& f) const {
        for (uint32_t i = 0; i < n && i < 2; ++i) f(in[i]);
        for (const std::string* k : more) f(k);
    }
};
// §5 F2 the ranking's rows of a content slot: the two most recent calls that served it and their row counts (speed
// fix 2: note_rows touches only the contents of the call's rows; speed fix 4: kept apart from the content, so the
// boundary pass scans these records only)
struct Rank {
    uint64_t call_a = 0, call_b = 0;
    long rows_a = 0, rows_b = 0;
    uint64_t seq = 0;                           // its install order (the tie-break: oldest install first)
    bool live = false;
    long rows_last2(uint64_t now) const {       // rows served in the last two calls (now − 1, now − 2)
        return (call_a + 3 > now ? (long)rows_a : 0) + (call_b + 3 > now ? (long)rows_b : 0);
    }
};
// threads T6: a map whose copy is EMPTY (a lane's overlay over its home's map: reads go to the home first).
template <class T>
struct FreshOnCopy : T {
    using T::T;
    FreshOnCopy() = default;
    FreshOnCopy(const FreshOnCopy&) : T() {}
    FreshOnCopy& operator=(const FreshOnCopy&) = delete;
};
}  // namespace

struct RecordFidelity::Impl {
    // threads T6: the evaluator's construction state and its interned contents live in ONE Shared object (defined at
    // the end of Impl) that the evaluator and its lanes (Impl copies) share through the reference members below; the
    // memo / ny_cache / λ maps copy fresh (a lane's overlays). home_ != nullptr: this Impl is a lane of *home_.
    struct Shared;
    std::shared_ptr<Shared> sh_ = std::make_shared<Shared>();
    Impl* home_ = nullptr;
    int n = 0, W = 0, k = 0;
    size_t max_bytes = 0;               // caller's byte budget for what the evaluator holds (0 = none)
    CliffordTableau& U = sh_->U;
    std::vector<uint64_t>& eps = sh_->eps;          // n-bit reference signs
    std::vector<uint64_t>& freemask = sh_->freemask;     // n-bit: free rows
    std::vector<int>& freepos = sh_->freepos;           // row -> free index (or -1)
    std::vector<std::complex<double>>& b = sh_->b;   // dense bare amplitudes over 2^k
    double bnorm2 = 0.0;
    RecordFidelityStats st;

    // ── S8 RF1: the logical expectation λ(u, v) = ⟨b| Z^v X^u |b⟩ on the free register, ONE formula:
    //     λ(u, v) = Σ_{y ∈ supp b, ascending, b_{y⊕u} ≠ 0} conj(b_y) · b_{y⊕u} · (−1)^{v·y},
    // a structural +0.0 when no support pair exists. The dense table (4^k entries, a not-computed bitmap) is a lazy
    // memo of exactly this function while it fits the λ budget (default 64 MiB) and the evaluator's byte budget;
    // otherwise the exact-key sparse memo is. Both hold the formula's bytes, so a value never depends on the path.
    std::vector<uint64_t>& bsupp = sh_->bsupp;                         // ascending support of b
    std::vector<std::complex<double>>& lam_tab = sh_->lam_tab;           // dense memo (lam_mode 1): row u at [u << k, (u + 1) << k)
    std::vector<uint64_t>& lam_done = sh_->lam_done;                      //   its filled ROWS (one bit per u)
    std::vector<double> lam_re_, lam_im_;                //   row-fill scratch
    struct LamKey {
        uint64_t u, v;
        bool operator==(const LamKey& o) const { return u == o.u && v == o.v; }
    };
    struct LamHash {
        size_t operator()(const LamKey& K) const { return std::hash<uint64_t>()(K.u * 0x9E3779B97F4A7C15ULL ^ K.v); }
    };
    FreshOnCopy<std::unordered_map<LamKey, std::complex<double>, LamHash>> lam_sparse;   // sparse memo (lam_mode 2)
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
        if (home_) {                                  // threads T6: a lane never fills the shared table
            if (home_->lam_mode == 1 && ((lam_done[u >> 6] >> (u & 63)) & 1ULL))
                __builtin_prefetch(&lam_tab[((size_t)u << k) | (size_t)v]);
            return;
        }
        if (lam_mode == 0) (void)lambda(u, v);
        if (lam_mode != 1) return;
        if (!((lam_done[u >> 6] >> (u & 63)) & 1ULL)) lambda_row(u);
        __builtin_prefetch(&lam_tab[((size_t)u << k) | (size_t)v]);
    }
    // threads T6: a lane's λ: the home's filled dense row or its sparse memo, else the lane's own sparse memo of the
    // formula (lambda_direct: the same bytes as a dense row's entry; a lane never fills the shared table).
    std::complex<double> lambda_lane_(uint64_t u, uint64_t v) {
        if (home_->lam_mode == 1 && ((lam_done[u >> 6] >> (u & 63)) & 1ULL)) return lam_tab[((size_t)u << k) | (size_t)v];
        const LamKey K{u, v};
        if (home_->lam_mode == 2) {
            const auto h = home_->lam_sparse.find(K);
            if (h != home_->lam_sparse.end()) return h->second;
        }
        lam_used_ = true;                             // the home decides its λ memo at the merge if still undecided
        auto it = lam_sparse.find(K);
        if (it != lam_sparse.end()) return it->second;
        const std::complex<double> r = lambda_direct(u, v);
        lam_sparse.emplace(K, r);
        charge(held_ov_, sizeof(LamKey) + 16 + 32, "a logical expectation");   // re-charged by the home at the merge
        return r;
    }
    // the λ memo's kind, decided at the first λ (the dense table when it fits the λ and byte budgets)
    void lam_decide_() {
        const size_t ent = k <= 30 ? (size_t)1 << (2 * k) : 0, tb = ent * 16 + ((size_t)1 << k) / 8;
        const size_t reg = 16 * b.size();
        if (ent && tb <= lam_budget && (!max_bytes || reg + held_memo + held_l3 + tb <= max_bytes)) {
            lam_tab.assign(ent, 0.0);
            lam_done.assign((((size_t)1 << k) + 63) / 64, 0);
            lam_mode = 1;
            charge_s(tb, "the logical expectation table");
        } else {
            lam_mode = 2;
        }
    }
    std::complex<double> lambda(uint64_t u, uint64_t v) {
        if (home_) return lambda_lane_(u, v);
        if (lam_mode == 0) lam_decide_();
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
        charge_s(sizeof(LamKey) + 16 + 32, "a logical expectation");
        return r;
    }

    FreshOnCopy<std::unordered_map<std::string, double>> memo;
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
    std::vector<int>& T_free_ = sh_->T_free_;

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
        ++memo_epoch_;                       // every content's memo_b reads 0
    }
    // fid-routine B2: a whole-cache drop's frees (the contents' structures: millions of small blocks; the memo; the
    // content index) on up to `threads` pool jobs instead of the home alone. Pure deallocation: nothing it frees is read
    // again (the memo's and the index's key pointers held by the contents are not dereferenced by their destructors),
    // and no value, id or counter depends on it. Below a few slices of contents it is serial (clear()'s own frees).
    void release_parallel_(int threads) {
        constexpr size_t kSlice = 256;            // (>= 4 slices: >= 1024 contents; test_threads' color5 has 1375)
        const size_t nsl = (l3c.size() + kSlice - 1) / kSlice;
        if (threads <= 1 || nsl < 4) return;
        const int Te = std::min<int>(threads, ThreadPool::instance().reserve(threads) + 1);
        if (Te <= 1) return;
        // fix round 1: the memo and the index are detached inside the jobs (ThreadPool::run can throw only before any
        // job runs: then nothing was detached, and clear() frees everything serially from a consistent state)
        std::atomic<size_t> next{0};
        ThreadPool::instance().run(Te, [&](int) {
            for (size_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < nsl + 2;) {
                if (t == 0) { std::unordered_map<std::string, double> m; m.swap(memo); continue; }
                if (t == 1) { decltype(sh_->l3_ids) ids; ids.swap(l3_ids); continue; }
                const size_t a = (t - 2) * kSlice, b = std::min(l3c.size(), a + kSlice);
                for (size_t i = a; i < b; ++i) l3c[i] = L3Content();
            }
        });
    }
    // fid-routine B2: a partial boundary pass's evicted structures, freed on up to `threads` pool jobs (deallocation
    // only, as release_parallel_)
    template <class Grave>                   // (std::vector<L3Content>: declared below)
    void release_grave_(Grave& g, int threads) {
        constexpr size_t kSlice = 256;
        const size_t nsl = (g.size() + kSlice - 1) / kSlice;
        const int Te = threads > 1 && nsl >= 2 ? std::min<int>(threads, ThreadPool::instance().reserve(threads) + 1) : 1;
        if (Te > 1) {
            std::atomic<size_t> next{0};
            ThreadPool::instance().run(Te, [&](int) {
                for (size_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < nsl;)
                    for (size_t i = t * kSlice, e = std::min(g.size(), i + kSlice); i < e; ++i) g[i] = typename Grave::value_type();
            });
        }
        Grave().swap(g);
    }
    void clear(int threads = 1) {
        release_parallel_(threads);
        clear_memo();
        l3c.clear(); l3c.shrink_to_fit();
        dead_count_ = 0;
        dead_ids_.clear(); free_ids_.clear();
        rk.clear(); rk.shrink_to_fit();
        born_ids_.clear();
        l3_ids.clear(); l3_ids.rehash(0);
        ny_cache.clear(); ny_cache.rehash(0);
        rec_lab_img.clear(); rec_lab_img.shrink_to_fit();
        lam_tab.clear(); lam_tab.shrink_to_fit(); lam_done.clear(); lam_done.shrink_to_fit();
        lam_sparse.clear(); lam_sparse.rehash(0); lam_mode = 0;
        {                                    // explaw T4(b): the output's classes go too (their charge is dropped below)
            const std::lock_guard<std::mutex> lk(sh_->cls_mu);
            std::vector<Pauli>().swap(sh_->cls_ops);
            std::vector<double>().swap(sh_->cls_val);
            sh_->cls_state = 0;
        }
        st.classes_nonzero = -1;             // (recomputed by the next class-route content)
        held_l3 = held_rest;                 // termroute-a m4: the register's rest structure stays (and stays charged)
        held_shared_ = held_rest;
        reserve_ = 0;
        last_rows_ = 0;
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
    std::vector<Pauli>& recops = sh_->recops;               // record operators (storage column order, exact phase)
    // the records' nonzero words (records are sparse): per record c, entries [rsw_off[c], rsw_off[c + 1]) of
    // (word, x word, z word) — <P, M_c> and the support tests read only these
    std::vector<int>& rsw_off = sh_->rsw_off;
    std::vector<int>& rsw_w = sh_->rsw_w;
    std::vector<uint64_t>& rsw_x = sh_->rsw_x;
    std::vector<uint64_t>& rsw_z = sh_->rsw_z;
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
    std::vector<Pauli>& rec_lab_img = sh_->rec_lab_img;   // content-independent frame images (undressed operators)
    std::vector<uint8_t>& rs = sh_->rs;                 // record c ∈ R_s
    std::vector<uint8_t>& rs_sign = sh_->rs_sign;            // its sign on |B⟩ (1 = eigenvalue −1)
    // ── register fidelity (strict contract; header). O = out wires (omask), R = the rest (rmask).
    int route = 0;                           // 1 = 3a (a = 0), 2 = 3b (a > 0)
    int MW = 0;                              // record words
    std::vector<uint64_t>& omask = sh_->omask;      // W words
    std::vector<uint64_t>& rmask = sh_->rmask;
    // 3a: S* = S0 ∩ P_O, |O| generators signed +1 on |B⟩, and their undressed frame images
    std::vector<Pauli>& star = sh_->star;
    std::vector<Pauli>& star_img0 = sh_->star_img0;
    // 3b: the rest stabilizer group S_χ = C ⊕ ⟨independent Born records⟩, C = the elements of S0 ∩ P_R commuting with
    // every Born record (dimension dC, the sector bits). Per C index i the rest destabilizer d_i (⟨d_i, g_j⟩ = δ_ij on
    // every S_χ generator): its frame image (2W words, x | z) and its record flips (MW words). Per deferred wire q the
    // C-syndromes of X_q / Z_q (WC words; zero off R).
    int dC = 0, WC = 0;
    std::vector<uint64_t>& dimg = sh_->dimg;
    std::vector<uint64_t>& drec = sh_->drec;
    std::vector<uint64_t>& colx = sh_->colx;
    std::vector<uint64_t>& colz = sh_->colz;
    std::vector<uint64_t>& rsmask = sh_->rsmask;   // MW words: R_s records, their signs
    std::vector<uint64_t>& rssignw = sh_->rssignw;
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
    // Two stages (fid-orth, OD-2026-10-24-fid-orth). Stage 1 (the content's first row): the variables' frame images,
    // the constraint matrix A (columns = variables) row-reduced ONCE — pivot rows `prow` with their payloads `ppay`
    // (the reduction map R) and the kernel payloads `kern` — and the content's constant part `c0` of the right-hand
    // side. Per row, b = b(P', y) ⊕ c0; the left-kernel test N^T b = 0 is the residual of b after its pivot bits are
    // eliminated (rows fully reduced: each pivot bit lives in its own row only), R b the XOR of the payloads of b's
    // pivot bits. A row with N^T b ≠ 0 is orthogonal: F = 0 exactly, nothing else is built or stored. Stage 2 (the
    // first consistent row): the form's cross terms, the sector directions, the fiber and the dressed-record fiber.
    struct NewC {
        bool full = false;                   // stage 2 built
        // rowc = c0 (CW: the rhs constant, G_0's fixed X part ⊕ D_{σ_0}'s | D_{σ_0}'s R_s flips) ‖ lastb (CW: the
        // previous row's b) ‖ lastt (WV: its R b); last_ok −1: none yet
        std::vector<uint64_t> rowc;
        uint64_t* c0() { return rowc.data(); }
        uint64_t* lastb() { return rowc.data() + CW; }
        uint64_t* lastt() { return rowc.data() + 2 * CW; }
        int last_ok = -1;
        std::vector<uint64_t> kern;          // kprime × WV: the constraint kernel (stage 1 → stage 2, then dropped)
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
                                   DQ.beta.capacity() + rowc.capacity() + kern.capacity()) * 8 + Q.lin.capacity() + DQ.lin.capacity() +
                   (pbit.capacity() + mrec.capacity()) * 4 + K0.bytes() + KD.bytes();
        }
    };
    // FID fix 3: a factor list parsed ONCE (parse_factors' shape, at the intern / prepare that validates it): its
    // variables in product order (C0 = F_V ⋯ F_1: the last factor's terms first), flat. Per variable its Pauli bits
    // (x | z, 2W words), phase and class: 1 / 3 a rot factor {c0 I, c1 Q} (c1/c0 = i^cl), 0 a CZ leg (A, then B as
    // the next variable; cb lists the pairs). ec = Σ (1 per rot, 2 per CZ).
    struct FacList {
        int V = 0, ec = 0;
        std::vector<uint64_t> xz;
        std::vector<uint8_t> ph, cl;
        std::vector<std::pair<int, int>> cb;
        const uint64_t* bits(int j, int W) const { return xz.data() + (size_t)j * 2 * (size_t)W; }
        size_t bytes() const { return xz.capacity() * 8 + ph.capacity() + cl.capacity() + cb.capacity() * 8; }
    };
    struct L3Content {
        std::unique_ptr<DiagExpansion> dex;  // S8: kind 0 (intern_content_diag): the closed form's input
        FacList fac;                         // S8: a factor-list content (its general expansion, lazily; FID fix 3: parsed)
        std::unique_ptr<NewC> nc;            // S8: the route-3b structure (built on first use)
        size_t charged = 0;                  // review L2: every byte charged for this content, net of its releases
                                             //   (the ledger an eviction releases; Σ charged + held_shared_ == held_l3)
        std::vector<Pauli> rec_img;          // 3a: frame images of C0† M_c C0
        std::vector<uint64_t> uimg;          // 3b: per DRESSED used record the frame image of C0† M_c C0 (2W)
        std::vector<uint8_t> uph;            //   its phase
        std::vector<int> uix;                // 3b: per used record (C.use order) its uimg index, −1: undressed (the lab
                                             //   image rec_lab_img, shared)
        bool classified = true;              // 3b: use / rn / undressed / images deferred to stage 2 (false)
        std::vector<int> use;                // records entering ⟨ψ|ψ⟩ and the key (R_n and the dressed R_s)
        std::vector<int> rn;                 // R_n records
        std::vector<int> undressed;          // undressed R_s records (consistency checks)
        std::vector<Pauli> star_img;         // 3a: frame images of C0† g C0, g ∈ S* (exact sign)
        // explaw T4(b): the class route (3b, chosen by the cost rule): the used records' dressed images as Paulis, the
        // dressed S_O generators and the dressed nonzero classes C0† Pbar_c C0 (frame images, exact sign)
        bool cls_route = false;
        std::vector<Pauli> cls_rec, cls_so, cls_img;
        // plancache-design §5: retention and tombstoning
        bool dead = false;                   // evicted (its id stays; a later row re-interns into a fresh id)
        size_t key_cap = 0;                  // the charge's key part
        const std::string* keyp = nullptr;   // its l3_ids key: the index node's own string (node-stable; speed fix 4:
        size_t kh = 0;                       //   no per-content copy) and its hash (erased on eviction)
        // (§5 F2 the ranking's rows: in the slot's Rank record, Shared::rk — speed fix 4: the pass scans 32-byte
        // records, not the contents)
        uint64_t born_call = 0;              // the call that interned it (the reserve = that call's new bytes)
        size_t memo_b = 0;                   // speed fix 4: its memo entries' bytes and keys (the memo nodes' own
        KeyList memo_keys;                   //   strings: node-stable) — valid while memo_ep is the memo's
        uint64_t memo_ep = 0;                //   epoch (clear_memo() bumps it, so no per-content reset)
        size_t bytes() const {
            auto pb = [](const Pauli& P) { return (P.x.capacity() + P.z.capacity()) * 8 + sizeof(Pauli); };
            size_t m = sizeof(L3Content) + (use.capacity() + rn.capacity() + undressed.capacity()) * 4;
            for (const Pauli& P : rec_img) m += pb(P);
            m += uimg.capacity() * 8 + uph.capacity() + uix.capacity() * 4;
            for (const Pauli& P : star_img) m += pb(P);
            for (const Pauli& P : cls_rec) m += pb(P);
            for (const Pauli& P : cls_so) m += pb(P);
            for (const Pauli& P : cls_img) m += pb(P);
            m += fac.bytes();
            if (dex) m += sizeof(DiagExpansion) + (dex->piv.capacity() + dex->rowof.capacity()) * 4 + dex->lin.capacity() +
                          dex->a1.capacity() + (dex->cols.capacity() + dex->z0.capacity() + dex->beta.capacity() +
                                                dex->arows.capacity() + dex->pre.capacity()) * 8;
            return m;
        }
    };
    std::unordered_map<HKey, int, HKeyHash, HKeyEq>& l3_ids = sh_->l3_ids;
    int find_l3_(const std::string& key, size_t h) const {   // -1: absent
        const auto it = l3_ids.find(HKey{std::string(), h, &key});
        return it == l3_ids.end() ? -1 : it->second;
    }
    std::vector<L3Content>& l3c = sh_->l3c;
    std::vector<Rank>& rk = sh_->rk;
    std::vector<int>& dead_ids_ = sh_->dead_ids;
    std::vector<int>& free_ids_ = sh_->free_ids;
    // 3b: the frame image (2W words, x | z) and phase of the content's u-th used record (dressed or the lab image)
    void used_img(const L3Content& C, size_t u, uint64_t* I, int& ph) const;
    // kind 0: whether D = S^{a&1}·CZ dresses record c (D† M_c D ≠ M_c); if so its pulled-back Z bits (W words) and phase
    // by the exact Lemma 1 (dex_pull). uz: the wires D touches (W words)
    bool dex_dress(const DiagExpansion& E, const uint64_t* uz, int c, uint64_t* zz, int& ph) const {
        const Pauli& M = recops[(size_t)c];
        ph = M.phase & 3;
        bool t = false;
        for (int i = rsw_off[(size_t)c]; i < rsw_off[(size_t)c + 1] && !t; ++i)
            if (rsw_x[(size_t)i] & uz[(size_t)rsw_w[(size_t)i]]) t = true;
        if (!t) return false;
        dex_pull(E, c, zz, ph);
        t = ph != (M.phase & 3);
        for (int w = 0; w < W && !t; ++w) if (zz[w] != M.z[(size_t)w]) t = true;
        return t;
    }
    // D† M_c D = i^{ph} X^x Z^{zz} (Lemma 1): zz = z ⊕ Ã x, ph = p − q(x)
    void dex_pull(const DiagExpansion& E, int c, uint64_t* zz, int& ph) const {
        const Pauli& M = recops[(size_t)c];
        for (int w = 0; w < W; ++w) zz[w] = M.z[(size_t)w];
        for (int i = rsw_off[(size_t)c]; i < rsw_off[(size_t)c + 1]; ++i)
            for (uint64_t mm = rsw_x[(size_t)i]; mm; mm &= mm - 1) {
                const int q = rsw_w[(size_t)i] * 64 + __builtin_ctzll(mm);
                if (E.a1[(size_t)q]) zz[q >> 6] ^= 1ULL << (q & 63);
                const int ri = E.rowof[(size_t)q];
                if (ri < 0) continue;
                const uint64_t* row = &E.arows[(size_t)ri * W];
                for (int v = 0; v < W; ++v) zz[v] ^= row[v];
            }
        ph = (M.phase - E.qform(M.x.data())) & 3;
    }
    // 3b: a content's used / R_n / undressed split with the dressed records' images (C.uimg, freed at the end of stage
    // 2), built at stage 2 for both kinds (orthogonal rows never read it): kind 0 by the exact Lemma 1, a factor list by
    // its own pull-back from the stored factors (fac_pull), never the caller's.
    void classify_3b(L3Content& C);
    std::vector<uint64_t> cl_zz_, cl_im_, cl_ux_, cl_uz_;   // classify_3b scratch (capacity kept)
    Pauli cl_pb_;
    // C0† M C0 for a factor list C0 = F_V ⋯ F_1 (conjugation by F_V first; parse_factors' convention), in place. A
    // rot factor c0 (I + i l B) (c1/c0 = i l, l = ±1) anticommuting with Q maps Q ↦ Q · (i l B); a CZ factor
    // {I, A, B, −AB}/2 maps Q ↦ Q · A^{⟨Q,B⟩} B^{⟨Q,A⟩} (−1)^{⟨Q,A⟩⟨Q,B⟩}. The phase rule is term_pullback's
    // (Pauli::multiply's), so on the sampler's commuting term contents the result is term_pullback's bit for bit.
    // FID fix 3: on the parsed list (variables in product order = the factors from F_V down); a CZ's −AB term is
    // applied as A then −B (Q · A · (−B) = Q · (−AB) exactly: the product is associative, the phase rule multiply's).
    void fac_pull(const FacList& L, Pauli& Q) const {
        auto anti = [&](const uint64_t* B) {
            int an = 0;
            for (int v = 0; v < W; ++v)
                an += __builtin_popcountll(Q.x[(size_t)v] & B[W + v]) + __builtin_popcountll(Q.z[(size_t)v] & B[v]);
            return (an & 1) != 0;
        };
        auto mul = [&](const uint64_t* B, int bph, int extra) {   // Q ← Q · (i^extra B)
            int sg = 0;
            for (int v = 0; v < W; ++v) sg += __builtin_popcountll(Q.z[(size_t)v] & B[v]);
            Q.phase = (Q.phase + bph + extra + 2 * (sg & 1)) & 3;
            for (int v = 0; v < W; ++v) { Q.x[(size_t)v] ^= B[v]; Q.z[(size_t)v] ^= B[W + v]; }
        };
        for (int j = 0; j < L.V; ++j) {
            const uint64_t* B = L.bits(j, W);
            if (L.cl[(size_t)j] & 1) {
                if (!anti(B)) continue;
#ifdef XTIM_MUTATE_RF_FACPULL
                mul(B, L.ph[(size_t)j], 4 - L.cl[(size_t)j]);   // mutation build only: l sign flipped
#else
                mul(B, L.ph[(size_t)j], L.cl[(size_t)j]);       // c1/c0 = i^l (parse_factors: ±i)
#endif
            } else {                                              // {I, A, B, −AB}/2: A = j, B = j + 1
                const uint64_t* B2 = L.bits(j + 1, W);
                const bool a = anti(B), b = anti(B2);
                if (a && b) { mul(B, L.ph[(size_t)j], 0); mul(B2, L.ph[(size_t)j + 1], 2); }
                else if (b) mul(B, L.ph[(size_t)j], 0);
                else if (a) mul(B2, L.ph[(size_t)j + 1], 0);
                ++j;
            }
        }
    }
    int intern_(const std::string& key, const std::function<Pauli(const Pauli&)>& pull, const std::vector<uint64_t>& ux,
                const std::vector<uint64_t>& uz, FacList&& fac, std::unique_ptr<DiagExpansion>&& dex);
    // FID fix 1: a kind-0 content's prefix Δ (nf.prefix's bits; its phase is a global phase): kept only when nonzero
    void set_pre_(DiagExpansion& E, const Pauli& P) const {
        bool any = false;
        for (int w = 0; w < W && !any; ++w) any = P.x[(size_t)w] || P.z[(size_t)w];
        if (!any) return;
        E.pre.assign((size_t)2 * W, 0);
        for (int w = 0; w < W; ++w) { E.pre[(size_t)w] = P.x[(size_t)w]; E.pre[(size_t)(W + w)] = P.z[(size_t)w]; }
    }
    // a parsed factor list's union X / Z support (the identity terms add nothing; a CZ's AB lies inside A ∪ B)
    void fac_union_(const FacList& L, std::vector<uint64_t>& ux, std::vector<uint64_t>& uz) const {
        ux.assign((size_t)W, 0);
        uz.assign((size_t)W, 0);
        for (int j = 0; j < L.V; ++j) {
            const uint64_t* B = L.bits(j, W);
            for (int w = 0; w < W; ++w) { ux[(size_t)w] |= B[w]; uz[(size_t)w] |= B[W + w]; }
        }
    }
    std::shared_ptr<PreparedContent> prepare_fac_(const std::function<Pauli(const Pauli&)>& pull,
                                                  const std::function<FacList()>& parse) const;
    int intern_fac_(const std::string& key, const std::function<Pauli(const Pauli&)>& pull, FacList&& L) {
        std::vector<uint64_t> ux, uz;
        fac_union_(L, ux, uz);
        return intern_(key, pull, ux, uz, std::move(L), nullptr);
    }
    // scaling F1: intern_ split. build_l3_ makes a content's structure from the construction state alone (`lab`: the
    // record operators' lab images, rec_lab_img's values) — concurrent builds of distinct contents are safe;
    // install_l3_ gives it its id and charges it (the evaluator's mutable state; serial, in first-row order).
    L3Content build_l3_(const std::function<Pauli(const Pauli&)>& pull, const std::vector<uint64_t>& ux,
                        const std::vector<uint64_t>& uz, FacList&& fac, std::unique_ptr<DiagExpansion>&& dex,
                        const std::vector<Pauli>& lab) const;
    int install_l3_(size_t key_cap, std::string&& key, size_t h, L3Content&& C, size_t cbytes = SIZE_MAX);   // key_cap: the caller's key's
    void lab_init_();                        // rec_lab_img's lazy fill (charged): intern_'s first step
    std::vector<Pauli> lab_pre_;             // F1: rec_lab_img's values for builds before the first install
    std::vector<uint8_t> tdr_scratch, tdu_scratch;

    // ── RF-ref: lifecycle and budget (rulings C2 / Q2). Contents and their kernels persist across evaluate() calls
    // (values are pure functions of their keys); only clear() drops them, clear_memo() drops the memo alone. Every
    // byte the evaluator holds is counted (memo + contents + kernels + caches) and, with max_bytes, budgeted: a record
    // whose evaluation would take the held bytes above max_bytes is refused by name.
    // threads T6 (fix F1): on a lane, held_l3 is the lane's NET change to the shared contents' structures (charges and
    // releases, modulo 2^64: a lane may release what its home was charged) — folded into the home's held_l3 at the
    // merge; held_ov_ counts the lane's overlay entries (the home charges those it adopts).
    size_t held_memo = 0, held_l3 = 0, held_ov_ = 0;
    size_t ev_grow_ = 0;                     // fid-routine B1: the last threaded evaluate's largest lane growth (bytes)
    long ev_rows_ = 0;                       //   and its rows (the next call's lane heap reserve; speed only)
    bool lam_used_ = false;
    size_t held_rest = 0;                    // the part of held_l3 built at construction (clear() keeps it)
    bool row_log_on = false;                 // test lever (S8 zero classes): per-row codes, memo bypassed
    std::vector<uint8_t> row_log;
    size_t memo_entry_bytes(const std::string& key) const { return key.capacity() + sizeof(double) + 64; }
    // plancache-design §5 (N4): CHECK-THEN-ADD — a refused charge adds nothing (the row's staging is discarded whole)
    void charge(size_t& slot, size_t bytes, const char* what) {
        const size_t reg = 16 * b.size();    // the resident bare register (budgeted at construction too)
        if (max_bytes && reg + held_memo + held_l3 + bytes > max_bytes)
            throw RfBudgetRefusal(std::string("RecordFidelity: the evaluator would hold ") +
                                  std::to_string(reg + held_memo + held_l3 + bytes) + " bytes (bare register " +
                                  std::to_string(reg) + ", memo " + std::to_string(held_memo) +
                                  ", contents and kernels " + std::to_string(held_l3) + ") after " + what +
                                  ", above the byte budget " + std::to_string(max_bytes) +
                                  " — refused (clear() drops what it holds)");
        slot += bytes;
    }
    // review L2: the two ledgers of held_l3 — per content (C.charged: charged at install, grown and shrunk by its
    // structure builds, released whole at its eviction) and shared (held_shared_: the register's rest structure, the
    // record operators' frame images, the λ memo, the conditioned norms, the output's classes). On the home
    // Σ_live C.charged + held_shared_ == held_l3 (audit_bytes_for_test; a lane carries its deltas, merged with held_l3).
    size_t held_shared_ = 0;
    void charge_c(L3Content& C, size_t bytes, const char* what) { charge(held_l3, bytes, what); C.charged += bytes; }
    void release_c(L3Content& C, size_t bytes) { held_l3 -= bytes; C.charged -= bytes; }
    void charge_s(size_t bytes, const char* what) { charge(held_l3, bytes, what); held_shared_ += bytes; }
    uint64_t call_no_ = 1;                   // §5: the current call (born_call)
    size_t reserve_ = 0;                     // §5: the previous call's new-content bytes
    long last_rows_ = 0;                     //   and its rows (speed fix 4: the reserve scales to the next call's)
    uint64_t install_seq_ = 0;               // speed fix 4: installs so far (Rank::seq)
    std::vector<int> born_ids_;              // the contents installed in the current call (the reserve, O(new))
    size_t dead_count_ = 0;                  // dead slots in l3c (tombstoned or free)
    // grave (fid-routine B2): non-null = the content's structure is moved there (freed later on the pool) instead of
    // freed here; everything else (ledger, memo entries, index entry, flags) is done here, in order, as before.
    void tombstone_(int id, std::vector<L3Content>* grave = nullptr) {
        L3Content& C = l3c[(size_t)id];
        if (C.dead) return;
        held_l3 -= std::min(held_l3, C.charged);   // the ledger (review L2)
#ifndef XTIM_MUTATE_RF_RECYCLE_KEEPS_MEMO
        if (C.memo_ep == memo_epoch_)        // its memo entries go with it (a recycled slot must not inherit them)
#else
        if (false)                           // mutation build only (MUT-RF-RECYCLE-KEEPS-MEMO): the entries stay
#endif
            C.memo_keys.each([&](const std::string* k) {
                const auto mt = memo.find(*k);
                if (mt == memo.end()) throw std::logic_error("RecordFidelity: a content's memo entry is gone (engine bug)");
                held_memo -= std::min(held_memo, memo_entry_bytes(mt->first));
                memo.erase(mt);
            });
        const auto it = l3_ids.find(HKey{std::string(), C.kh, C.keyp});
        if (it == l3_ids.end()) throw std::logic_error("RecordFidelity: an evicted content is not indexed (engine bug)");
        l3_ids.erase(it);
        if (grave) grave->push_back(std::move(C));
        C = L3Content();
        C.dead = true;
        rk[(size_t)id].live = false;
        ++dead_count_;
        dead_ids_.push_back(id);
    }
    size_t ny_bytes_() const {
        size_t m = 0;
        for (const auto& kv : ny_cache) m += kv.first.capacity() + sizeof(Scaled) + 64;
        return m;
    }
    // speed fix 4 (replaces the compaction): the dead slots are recycled between calls — no content moves, no id is
    // renumbered, the index and the memo keep their keys. A content's memo entries are erased at its eviction
    // (tombstone_, through its key list: a memo key starts with its content's slot id, so a recycled slot must not
    // inherit them); a later install reuses the lowest free slot. Ids stay stable within a call (the recycle runs
    // between calls only).
    uint64_t memo_epoch_ = 1;
    size_t memo_b_of(const L3Content& C) const { return C.memo_ep == memo_epoch_ ? C.memo_b : 0; }
    void note_memo_(const std::string& key, size_t bytes) {    // (home only: a lane's entries are noted at the merge)
        int id = -1;
        std::memcpy(&id, key.data(), sizeof(int));
        if (id < 0 || (size_t)id >= l3c.size() || l3c[(size_t)id].dead)   // re-review L-2: loud, never a silent skip
            throw std::logic_error("RecordFidelity: a memo entry for an absent or evicted content (engine bug)");
        L3Content& C = l3c[(size_t)id];
        if (C.memo_ep != memo_epoch_) { C.memo_ep = memo_epoch_; C.memo_b = 0; C.memo_keys.clear(); }
        C.memo_keys.push_back(&key);              // (first: a failed push changes nothing — fix round 2)
        C.memo_b += bytes;
    }
    // fix round 2: a new memo entry's charge and note as one step — a throw in either erases the entry (and its charge),
    // so no entry its content does not track survives to be read under a recycled slot id
    void memo_track_(decltype(memo)::iterator it, bool note) {
        const size_t b = memo_entry_bytes(it->first);
        try { charge(held_memo, b, "a memo entry"); } catch (...) { memo.erase(it); throw; }
        try {
            if (fault_test_ == 4) { fault_test_ = 0; throw std::bad_alloc(); }   // test lever
            if (note) note_memo_(it->first, b);
        } catch (...) {
            held_memo -= b;
            memo.erase(it);
            throw;
        }
    }
    void recycle_dead_() {                   // between calls only (no content id is live)
        if (dead_ids_.empty()) return;
        born_ids_.clear();                   // (between calls: the reserve was taken at the call's end)
        free_ids_.insert(free_ids_.end(), dead_ids_.begin(), dead_ids_.end());
        dead_ids_.clear();
        std::sort(free_ids_.begin(), free_ids_.end(), std::greater<int>());
    }
    void memo_put(const std::string& key, double F) {
        const auto r = memo.emplace(key, F);
        if (r.second) memo_track_(r.first, !home_);
    }

    // N_y = ⟨B|Π_{R_n}(y)|B⟩ on the resident 2^k register (the lab frame images; R_n is every non-R_s record, so the
    // value is a function of the |R_n|-bit word y_{R_n}), cached per distinct word (charged to the held bytes).
    FreshOnCopy<std::unordered_map<std::string, Scaled>> ny_cache;
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
        if (home_) {                                  // threads T6: a lane reads its home's cache first
            const auto h = home_->ny_cache.find(key);
            if (h != home_->ny_cache.end()) return h->second;
        }
        auto it = ny_cache.find(key);
        if (it != ny_cache.end()) return it->second;
        ny_ops_.clear();
        for (int c : C.rn)
#ifdef XTIM_MUTATE_RFREF_NYDRESS
        {                                                               // mutation build only: C0-dressed images
            const size_t u = (size_t)(std::find(C.use.begin(), C.use.end(), c) - C.use.begin());
            Pauli R(n);
            {
                std::vector<uint64_t> I((size_t)2 * W);
                int ph = 0;
                used_img(C, u, I.data(), ph);
                for (int w = 0; w < W; ++w) { R.x[(size_t)w] = I[(size_t)w]; R.z[(size_t)w] = I[(size_t)(W + w)]; }
                R.phase = ph;
            }
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
        // (a lane's entry is an overlay: the home charges the entries it adopts at the merge)
        if (home_) charge(held_ov_, it->first.capacity() + sizeof(Scaled) + 64, "a conditioned-reference norm");
        else charge_s(it->first.capacity() + sizeof(Scaled) + 64, "a conditioned-reference norm");
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
        if (l3c[(size_t)id].dead)            // re-review L-2: a recycled slot must never be evaluated under its old id
            throw std::logic_error("RecordFidelity: a row reads an evicted content id (engine bug)");
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
        // FID fix 1: a kind-0 content with a prefix Δ (C0 = Δ·S^{a&1}·CZ): the row's state P·Δ·S^{a&1}·CZ|bare⟩ is the
        // prefix-free content's at P ⊕ Δ (a Pauli's phase is global) — Δ folded into the row's Pauli, memo key included
        if (const DiagExpansion* dx = l3c[(size_t)id].dex.get(); dx && !dx->pre.empty()) {
            pq_.resize((size_t)2 * W);
            for (int w = 0; w < W; ++w) {
                pq_[(size_t)w] = px[w] ^ dx->pre[(size_t)w];
                pq_[(size_t)(W + w)] = pz[w] ^ dx->pre[(size_t)(W + w)];
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
        // 3b: the content's constraint system first (stage 1, OD-2026-10-24-fid-orth): a row whose right-hand side is
        // outside the column space is ORTHOGONAL — F = 0 exactly; no value structure, no key, no memo entry, no record
        // check (the sampler's rows carry no impossible records; the consistent rows keep every check below)
        NewC* Xp = nullptr;
        if (route == 2) {
            Xp = &new_content(C);
            if (!consistent_new(*Xp, px, pz, y)) {
                ++st.orthogonal_rows;
                if (row_log_on) row_log.push_back((uint8_t)kInfeasible);
                return 0.0;
            }
            value_content(C, *Xp);
            Xp = C.nc.get();
        }
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
        // 3b: the key is everything the value reads beyond the content — t_0, the sector σ, the base L, the fiber's
        // cross bits and the Born bits y'' (and, above, the used records' signs td: ⟨ψ|ψ⟩)
        if (route == 2) {
#ifdef XTIM_MUTATE_RF_MEMO_NOC
            {                                                   // mutation build only: the key without the correction
                const bool kf = consistent_new(*Xp, kx, kz, y);
                if (kf) coset_new(*Xp);
                key_new(*Xp, kf, C, y);
                consistent_new(*Xp, px, pz, y);
                coset_new(*Xp);
            }
#else
            coset_new(*Xp);
            key_new(*Xp, true, C, y);
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
#ifndef XTIM_MUTATE_RF_UNDRESSED_SKIP                    // mutation build only: no undressed-record check at all
            for (int c : C.undressed)
                if (((int)((y[c >> 6] >> (c & 63)) & 1ULL) ^ rec_sympl(px, pz, c)) != rs_sign[(size_t)c])
                    throw std::runtime_error("RecordFidelity: record " + std::to_string(c) + " is definite on the "
                                             "retained state but its recorded value disagrees (impossible record)");
#endif
        };
        if (undressed_first) check_undressed();
        const bool use_memo = !row_log_on && !hform_oracle;
        if (use_memo) {
            if (home_) {                              // threads T6: a lane reads its home's memo first
                const auto hit = home_->memo.find(key);
                if (hit != home_->memo.end()) { ++st.memo_hits; return hit->second; }
            }
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
        } else if (C.cls_route) {
            F = value_classes(C, px, pz, td);              // explaw T4(b): the class route
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
    std::vector<int>& rn_ind = sh_->rn_ind;                         // independent Born records (S_χ generators beyond C), ascending
    // ── explaw T4(b): the output's logical classes (route 3b). S_O = S0 ∩ P_O (the output code, rank n_O − a); the
    // nonzero classes Pbar_c (Hermitian representatives of N(S_O)/S_O on O with <B|Pbar_c|B> ≠ 0) and their bare values,
    // computed once, lazily, by the first content the route rule below sends to the class route (2a <= kClassRankCap:
    // <= 2^20 classes).
    static constexpr int kClassRankCap = 20;
    // explaw T4(b) fix-1 M1: the two routes' operation counts for a content's first miss, in calibrated units of one
    // support-entry pass (~2.2 ns; e5/STATUS.md). Class route: the 4^a classes (the count before computing them), each
    // ~256 units of Pauli work plus one pass over the support |supp b| (the one-time bare value and the per-miss term
    // together), plus the miss's projections of the 2^k register. Sector route: 2^ksec sector terms at a LOWER bound
    // of 128 units each (measured 147-330 per term on the families in STATUS; the λ lookups per term are not known
    // before a term runs, so they are not counted): biased toward the sector route, the pre-T4(b) behaviour (on
    // the calibration families no class choice was slower than the sector sum). A pure function of (a, k, ksec, |supp b|): the same route on every lane and in every order.
    static constexpr double kSecOps = 128.0, kClsOps = 256.0, kProjOps = 4.0;
    double sector_route_cost(int ksec) const { return std::ldexp(kSecOps, ksec); }
    double class_route_cost() const {
        return std::ldexp(kClsOps + (double)bsupp.size(), 2 * (int)st.out_a) + std::ldexp(kProjOps, k);
    }
    // (threads: in Shared — computed once, under Shared::cls_mu, by whichever lane's content build needs them first;
    // a pure function of the bare state, so the classes are the same bytes whichever lane computes them)
    std::vector<int>& owires = sh_->owires;          // the output wires, ascending
    std::vector<Pauli>& so_gens = sh_->so_gens;      // route 3b: S_O generators (lab frame, Hermitian)
    std::vector<Pauli>& cls_ops = sh_->cls_ops;      // the nonzero classes (lab frame, Hermitian)
    std::vector<double>& cls_val = sh_->cls_val;     //   <B|Pbar_c|B> / <B|B>
    // Re <B|Q|B> / <B|B> for a lab Pauli Q (0 when its frame image has X on a fixed row: structurally zero)
    double bare_expect(const Pauli& Q) const {
        const Pauli I = img(Q);
        for (int w = 0; w < W; ++w) if (I.x[(size_t)w] & ~freemask[(size_t)w]) return 0.0;
        const FOp o = compile(I);
        std::complex<double> A = 0.0;
        for (uint64_t i : bsupp) A += std::conj(b[(size_t)(i ^ o.xm)]) * sgn(i & o.zm) * b[(size_t)i];
        A *= o.c / bnorm2;
        return A.real();
    }
    void compute_classes() {
        const std::lock_guard<std::mutex> lk(sh_->cls_mu);
        if (sh_->cls_state) return;
        std::vector<Pauli> cls_ops;                              // built locally, committed whole at the end (a
        std::vector<double> cls_val;                             //   failure mid-way leaves the classes uncomputed)
        const int nO = (int)owires.size(), sO = (int)so_gens.size(), a = (int)st.out_a;
        // the commutant of S_O inside P_O: the GF(2) nullspace of the sO × 2nO symplectic constraints (column j = X on
        // owires[j], column nO + j = Z on it: <v, g> = Σ_j v_x(j) g_z(j) + v_z(j) g_x(j))
        const int m = 2 * nO, MWd = std::max(1, (m + 63) / 64);
        std::vector<std::vector<uint64_t>> A((size_t)sO, std::vector<uint64_t>((size_t)MWd, 0));
        for (int r = 0; r < sO; ++r)
            for (int j = 0; j < nO; ++j) {
                const int q = owires[(size_t)j];
                if (so_gens[(size_t)r].zbit(q)) A[(size_t)r][(size_t)(j >> 6)] |= 1ULL << (j & 63);
                if (so_gens[(size_t)r].xbit(q)) A[(size_t)r][(size_t)((nO + j) >> 6)] |= 1ULL << ((nO + j) & 63);
            }
        std::vector<int> pivc;                                   // RREF: pivot column of row r
        {
            int r = 0;
            for (int c = 0; c < m && r < sO; ++c) {
                int p = -1;
                for (int i = r; i < sO; ++i) if ((A[(size_t)i][(size_t)(c >> 6)] >> (c & 63)) & 1ULL) { p = i; break; }
                if (p < 0) continue;
                std::swap(A[(size_t)p], A[(size_t)r]);
                for (int i = 0; i < sO; ++i)
                    if (i != r && ((A[(size_t)i][(size_t)(c >> 6)] >> (c & 63)) & 1ULL))
                        for (int w = 0; w < MWd; ++w) A[(size_t)i][(size_t)w] ^= A[(size_t)r][(size_t)w];
                pivc.push_back(c);
                ++r;
            }
            if (r != sO) throw std::logic_error("RecordFidelity: dependent output stabilizers (internal)");
        }
        std::vector<uint64_t> All((size_t)2 * W, ~0ULL);
        MaskElim EI(W, All);
        for (const Pauli& g : so_gens) EI.add(bits_of(g, W));
        std::vector<Pauli> L;                                    // the logical representatives (independent mod S_O)
        std::vector<uint8_t> ispiv((size_t)m, 0);
        for (int c : pivc) ispiv[(size_t)c] = 1;
        for (int f = 0; f < m; ++f) {
            if (ispiv[(size_t)f]) continue;
            Pauli v(n);                                          // v_f = 1, v_{pivc[r]} = A[r][f]
            auto setc = [&](int c) { if (c < nO) v.setx(owires[(size_t)c]); else v.setz(owires[(size_t)(c - nO)]); };
            setc(f);
            for (int r = 0; r < sO; ++r) if ((A[(size_t)r][(size_t)(f >> 6)] >> (f & 63)) & 1ULL) setc(pivc[(size_t)r]);
            if (EI.add(bits_of(v, W))) L.push_back(pauli_of(bits_of(v, W), n, W));
        }
        if ((int)L.size() != 2 * a)
            throw std::logic_error("RecordFidelity: the output code has " + std::to_string(L.size()) + " logical directions, expected " +
                                   std::to_string(2 * a) + " (internal)");
        for (uint64_t c = 0; c < (1ULL << (2 * a)); ++c) {
            Pauli Pc(n);
            for (int i = 0; i < 2 * a; ++i) if ((c >> i) & 1) Pc = Pauli::multiply(Pc, L[(size_t)i]);
            Pc = pauli_of(bits_of(Pc, W), n, W);                 // the Hermitian representative of the class
            const double v = bare_expect(Pc);
            if (std::abs(v) > 1e-12) { cls_ops.push_back(std::move(Pc)); cls_val.push_back(v); }
        }
#ifdef XTIM_MUTATE_EXPLAW_CLS_DROP
        if (cls_ops.size() > 1) { cls_ops.erase(cls_ops.begin() + 1); cls_val.erase(cls_val.begin() + 1); }   // mutation build only: a nonzero class dropped
#endif
        charge_s(cls_ops.size() * (2 * (size_t)W * 8 + sizeof(Pauli) + 8), "the output's logical classes");
        st.classes_nonzero = (long)cls_ops.size();
        sh_->cls_ops = std::move(cls_ops);
        sh_->cls_val = std::move(cls_val);
        sh_->cls_state = 1;
    }
    // C0† Q C0 on route 3b: kind 0 by the exact Lemma 1 (dress_diag), a factor list by its own pull-back (fac_pull)
    Pauli pull3b(const L3Content& C, const Pauli& Q) const {
        if (C.dex) return dress_diag(*C.dex, Q);
        Pauli pb = Q;
        fac_pull(C.fac, pb);
        return pb;
    }
    // The class route: F = 2^{-a} Σ_c <B|Pbar_c|B> <ψ|Π_{S_O} Pbar_c|ψ> / <ψ|ψ>, each term route 3a's projector expectation
    // (the used records at the shot's signs, the S_O generators signed by P''s parities) with the extra dressed Pauli
    // C0† Pbar_c C0 (signed by <P'', Pbar_c>): 2^{-rows} <b| Π_K Q̂'|b> with Q' the class reduced by the pivot rows (zero
    // when its fixed X part is outside their span).
    double value_classes(L3Content& C, const uint64_t* px, const uint64_t* pz, const std::vector<uint8_t>& td) {
        std::vector<Pauli> chops;
        chops.reserve(C.use.size() + so_gens.size());
        for (size_t u = 0; u < C.use.size(); ++u) chops.push_back(with_sign(C.cls_rec[u], td[(size_t)C.use[u]]));
        const Scaled D = proj_expect(chops, nullptr);
        if (!(D.m > 1e-300)) throw std::runtime_error("RecordFidelity: <psi|psi> = 0 for this record (impossible record)");
        for (size_t i = 0; i < so_gens.size(); ++i) chops.push_back(with_sign(C.cls_so[i], sympl(px, pz, so_gens[i], W)));
        Elim E;
        for (const Pauli& q : chops) insert(E, q);
        std::vector<std::complex<double>> v = b;
        for (const Pauli& K : E.kernel) apply_projector(v, compile(K));
        const int rows = (int)E.rows.size();
        double acc = 0.0;
        for (size_t c = 0; c < C.cls_img.size(); ++c) {
            Pauli q = with_sign(C.cls_img[c], sympl(px, pz, cls_ops[c], W));
            std::vector<uint64_t> f = fixed_part(q);
            reduce(E.rows, f, q);
            bool nz = false;
            for (int w = 0; w < W; ++w) if (f[(size_t)w]) nz = true;
            if (nz) continue;                                    // the class's fixed X part is not a record product: 0
            const FOp o = compile(q);
            std::complex<double> A = 0.0;
            for (uint64_t j : bsupp)                             // ascending: the order of the nonzero b[j] terms
                A += std::conj(b[(size_t)j]) * (o.c * sgn((j ^ o.xm) & o.zm)) * v[(size_t)(j ^ o.xm)];
            acc += cls_val[c] * std::ldexp(A.real() / D.m, D.e - rows);
        }
        ++st.class_rows;
        return std::ldexp(acc, -(int)st.out_a);
    }
    bool hform_oracle = false;                       // test lever (S8 RF4): fibers by exact enumeration (κ' <= 12)
    int fault_test_ = 0;                             // test lever (fix round 1): 1 / 2 = a bad_alloc between the
                                                     //   evaluate's waves / at the memo reserve, once; (fix round 2)
                                                     //   3 = in the merge after lane 0, 4 = at a merged entry's note
    int& class_lever = sh_->class_lever;              // test lever (explaw T4(b), review F4): 0 the cost rule, 1 the sector
                                                     //   route always, 2 the class route whenever computable (in Shared:
                                                     //   every lane, existing or not, reads the same lever)
    bool oracle_skip_ = false;                       //   a fiber above 12: the row is logged as class 255, F = NaN

    // the frame image's bits of X^x Z^z (no phase): the XOR of the per-wire inverse rows' bits
    std::vector<uint64_t>& invrows = sh_->invrows;                   // route 3b: per wire q, U†X_qU bits (2W) then U†Z_qU bits (2W)
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
    std::vector<uint8_t>& invph = sh_->invph;                      // route 3b: per wire q, the phases of U†X_qU and U†Z_qU
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
    // FID fix 3: the list parsed once into a FacList (refused by name, RF_FACTOR_NOT_CLIFFORD); the variables in
    // product order (C0 = F_V ⋯ F_1: the last factor's terms leftmost).
    void push_var_(FacList& L, const Pauli& Q, int cl) const {
        const size_t o = L.xz.size();
        L.xz.resize(o + (size_t)2 * W);
        std::copy(Q.x.begin(), Q.x.begin() + W, L.xz.begin() + (std::ptrdiff_t)o);
        std::copy(Q.z.begin(), Q.z.begin() + W, L.xz.begin() + (std::ptrdiff_t)(o + (size_t)W));
        L.ph.push_back((uint8_t)(Q.phase & 3));
        L.cl.push_back((uint8_t)cl);
        ++L.V;
    }
    FacList parse_factors(const std::vector<RecordFidelity::Factor>& facs) const {
        FacList L;
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
                push_var_(L, f[1].second, l);
                L.ec += 1;
            } else if (f.size() == 4 && is_identity(f[0].second) && near(f[0].first, 0.5) && near(f[1].first, 0.5) &&
                       near(f[2].first, 0.5) && near(f[3].first, -0.5)) {
                const Pauli p12 = Pauli::multiply(f[1].second, f[2].second);
                if (!(p12.x == f[3].second.x && p12.z == f[3].second.z && ((p12.phase - f[3].second.phase) & 3) == 0) ||
                    Pauli::anticommute_bit(f[1].second, f[2].second))
                    refuse_factor("{I, A, B, −AB}/2 has AB != A·B or anticommuting A, B");
                const int a = L.V;
                push_var_(L, f[1].second, 0);
                push_var_(L, f[2].second, 0);
                L.cb.push_back({a, a + 1});
                L.ec += 2;
            } else {
                refuse_factor("with " + std::to_string(f.size()) + " terms is neither {c0 I, c1 Q} nor a CZ");
            }
        }
        return L;
    }
    // FID fix 3: the rot list Π_j rot(B_j) = Π_j (I + i B_j)/√2 (factors in the given order), parsed without building
    // the factor terms — what parse_factors gives for {{1/√2, I}, {i/√2, B_j}} per axis.
    FacList parse_rots(const std::vector<const Pauli*>& axes) const {
        FacList L;
        L.xz.reserve(axes.size() * 2 * (size_t)W);
        for (size_t fi = axes.size(); fi-- > 0;) {
            const Pauli& B = *axes[fi];
            if (!hermitian(B) || is_identity(B)) refuse_factor("has a non-Hermitian or identity axis");
            push_var_(L, B, 1);
            L.ec += 1;
        }
        return L;
    }
    Expansion expand_factors(const FacList& L) const {
        const int ec = L.ec;
        const int Vr = L.V;
        const int W2 = 2 * W;
        Expansion E;
        E.G0 = Pauli(n);
        if (Vr == 0) { E.q.init(0); return E; }
        QForm T;
        T.init(Vr);
        for (int j = 0; j < Vr; ++j) T.lin[(size_t)j] = (uint8_t)((L.cl[(size_t)j] + L.ph[(size_t)j]) & 3);
#ifndef XTIM_MUTATE_RF_K2CROSS
        for (int j = 0; j < Vr; ++j)
            for (int j2 = j + 1; j2 < Vr; ++j2)
                if (parw(L.bits(j, W) + W, L.bits(j2, W), W)) T.flip(j, j2);
#endif                                                           // mutation build only: 2 z_j·x_j' dropped
        for (const auto& e : L.cb) T.flip(e.first, e.second);
        // the image map t ↦ (x | z): pivot variables and kernel vectors (flat: rows of 2W / WT words)
        const int WT = T.WV;
        std::vector<uint64_t> rv, rp, kern, v((size_t)W2), pay((size_t)WT);
        rv.reserve((size_t)Vr * W2);
        rp.reserve((size_t)Vr * WT);
        std::vector<int> rpiv, pvar;
        for (int j = 0; j < Vr; ++j) {
            std::copy(L.bits(j, W), L.bits(j, W) + W2, v.begin());
            std::fill(pay.begin(), pay.end(), 0);
            gs::flipbit(pay.data(), j);
            for (size_t r = 0; r < rpiv.size(); ++r)
                if (gs::getbit(v.data(), rpiv[r])) {
                    const uint64_t* a = &rv[r * W2];
                    const uint64_t* b = &rp[r * WT];
                    for (int w = 0; w < W2; ++w) v[(size_t)w] ^= a[w];
                    for (int w = 0; w < WT; ++w) pay[(size_t)w] ^= b[w];
                }
            int pb = -1;
            for (int w = 0; w < W2 && pb < 0; ++w) if (v[(size_t)w]) pb = w * 64 + __builtin_ctzll(v[(size_t)w]);
            if (pb < 0) { kern.insert(kern.end(), pay.begin(), pay.end()); continue; }
            for (size_t r = 0; r < rpiv.size(); ++r)
                if (gs::getbit(&rv[r * W2], pb)) {
                    uint64_t* a = &rv[r * W2];
                    uint64_t* b = &rp[r * WT];
                    for (int w = 0; w < W2; ++w) a[w] ^= v[(size_t)w];
                    for (int w = 0; w < WT; ++w) b[w] ^= pay[(size_t)w];
                }
            rv.insert(rv.end(), v.begin(), v.end());
            rp.insert(rp.end(), pay.begin(), pay.end());
            rpiv.push_back(pb);
            pvar.push_back(j);
        }
        // change of variables t = Σ_i a_i e_{pvar[i]} ⊕ Σ_m b_m kern_m; the a's are the parameters. On two unit
        // vectors the form is T's own entries (Q(e_a) = lin_a, B(e_a, e_b) = β_ab for a ≠ b): filled directly; a pair
        // with a kernel vector goes through polar / eval.
        const int na = (int)pvar.size();
        QForm T2;
        T2.init(Vr);
        for (int i = 0; i < na; ++i) T2.lin[(size_t)i] = (uint8_t)(T.lin[(size_t)pvar[(size_t)i]] & 3);
        for (int i = na; i < Vr; ++i) T2.lin[(size_t)i] = (uint8_t)T.eval(&kern[(size_t)(i - na) * WT]);
        for (int i = 0; i < na; ++i) {
            const uint64_t* ri = T.row(pvar[(size_t)i]);
            for (int i2 = i + 1; i2 < na; ++i2)
                if (gs::getbit(ri, pvar[(size_t)i2])) T2.flip(i, i2);
        }
        std::vector<uint64_t> e((size_t)WT, 0);
        for (int i = 0; i < Vr; ++i) {
            const uint64_t* bi = i < na ? e.data() : &kern[(size_t)(i - na) * WT];
            if (i < na) gs::flipbit(e.data(), pvar[(size_t)i]);
            for (int i2 = std::max(i + 1, na); i2 < Vr; ++i2)
                if (T.polar(bi, &kern[(size_t)(i2 - na) * WT])) T2.flip(i, i2);
            if (i < na) gs::flipbit(e.data(), pvar[(size_t)i]);
        }
        std::vector<uint8_t> keep((size_t)Vr, 0);
        for (int i = 0; i < na; ++i) keep[(size_t)i] = 1;
        GaussSum G;
        gauss_sum(T2, keep, G);
        if (G.zero) throw std::logic_error("RecordFidelity: a C0 factor list expands to zero (internal)");
        auto patt = [&](int i) { return L.bits(pvar[(size_t)i], W); };
        const int F = (int)G.free_keep.size();
        std::vector<uint64_t> w0((size_t)W2, 0), wf((size_t)F * W2, 0);
        for (size_t d = 0; d < G.dep_var.size(); ++d)
            if (G.dep_c[d]) { const uint64_t* pw = patt(G.dep_var[d]); for (int w = 0; w < W2; ++w) w0[(size_t)w] ^= pw[w]; }
        for (int fi = 0; fi < F; ++fi) {
            const int i = G.free_keep[(size_t)fi];
            uint64_t* f = &wf[(size_t)fi * W2];
            std::copy(patt(i), patt(i) + W2, f);
            for (size_t d = 0; d < G.dep_var.size(); ++d)
                if (gs::getbit(&G.dep_mask[d * (size_t)T2.WV], i)) {
                    const uint64_t* pw = patt(G.dep_var[d]);
                    for (int w = 0; w < W2; ++w) f[w] ^= pw[w];
                }
        }
        E.G0 = pauli_of(w0, n, W);
        E.G0.phase = 0;
        std::vector<uint64_t> one((size_t)W2);
        for (int fi = 0; fi < F; ++fi) {
            std::copy(&wf[(size_t)fi * W2], &wf[(size_t)fi * W2] + W2, one.begin());
            E.G.push_back(pauli_of(one, n, W));
            E.G.back().phase = 0;
        }
        // coefficient of the ORDERED product G0 · Π G_i^{f_i}: i^{Qf(f)} · i^{−π'(f)}, π' its phase vs the bare bits
        E.q = G.f;
        for (int fi = 0; fi < F; ++fi) {
            if (parw(w0.data() + W, &wf[(size_t)fi * W2], W)) E.q.lin[(size_t)fi] = (uint8_t)((E.q.lin[(size_t)fi] + 2) & 3);
            for (int f2 = fi + 1; f2 < F; ++f2)
                if (parw(&wf[(size_t)fi * W2] + W, &wf[(size_t)f2 * W2], W)) E.q.flip(fi, f2);
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

    // The route-3b structure of a content, stage 1 (built on its first row; charged): the variables, their images and
    // Z4 coefficients, the constraint matrix's reduction (prow / ppay / pbit / kern) and the rhs constant c0.
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
        // the expansion's own cross terms (the images' cross terms x_j·z_j' are added at stage 2; flips commute)
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
        X.kern = std::move(kern);
        // the rhs constant: G_0's fixed X part with its sector's destabilizer D_{σ_0} | D_{σ_0}'s R_s record flips (the
        // row part, from P' and y, is XORed per row: every term is linear in the bits)
        std::fill(im.begin(), im.end(), 0);
        std::fill(rc.begin(), rc.end(), 0);
        dest_product(X.sig0.data(), im.data(), rc.data());
        X.rowc.assign((size_t)(2 * X.CW + X.WV), 0);
        uint64_t* c0 = X.c0();
        for (int w = 0; w < W; ++w) c0[w] = (X.I0[(size_t)w] ^ im[(size_t)w]) & ~freemask[(size_t)w];
        for (int w = 0; w < MW; ++w) c0[W + w] = rc[(size_t)w] & rsmask[(size_t)w];
        ++st.kernel_compiles;
        st.max_r = std::max<long>(st.max_r, X.nG);
        st.max_kappa_prime = std::max<long>(st.max_kappa_prime, X.kprime);
        auto hist = [](std::vector<long>& h, int v) { if (h.size() <= (size_t)v) h.resize((size_t)v + 1, 0); ++h[(size_t)v]; };
        hist(st.r_hist, X.nG);
        hist(st.kappa_prime_hist, X.kprime);
        const size_t nb = X.bytes();
        C.nc = std::move(Np);
        charge_c(C, nb, "a content's constraint structure");
        return *C.nc;
    }

    // Stage 2 (the content's first consistent row; charged by its growth): the images' cross terms of the form, the
    // sector directions in canonical form, the fiber and the dressed-record fiber of ⟨ψ|ψ⟩.
    // A refusal (RF_SECTOR_RANK / RF_LOGICAL_RANK) or an allocation failure mid-build drops the content's structure
    // (stage 1 included, as before the split), so a later row rebuilds it from scratch.
    void value_content(L3Content& C, NewC& X) {
        if (X.full) return;
        const size_t nb0 = X.bytes();
        try {
            value_content_(C, X, nb0);
        } catch (...) {
            if (!X.full) {
                release_c(C, nb0);
                C.nc.reset();
                C.cls_route = false;                            // the class images were never charged: released
                std::vector<Pauli>().swap(C.cls_rec);
                std::vector<Pauli>().swap(C.cls_so);
                std::vector<Pauli>().swap(C.cls_img);
            }
            throw;
        }
    }
    void value_content_(L3Content& C, NewC& X, size_t nb0) {
        if (!C.classified) classify_3b(C);                      // the record split, on the first consistent row
        for (int j = 0; j < X.V; ++j)
            for (int j2 = j + 1; j2 < X.V; ++j2)
                if (parw(&X.I[(size_t)j * 2 * W], &X.I[(size_t)j2 * 2 * W + W], W)) X.Q.flip(j, j2);
        std::vector<uint64_t>& kern = X.kern;
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
        st.max_ksec = std::max<long>(st.max_ksec, X.ksec);
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
        C.cls_route = false;
        if (X.ksec > 0 && 2 * st.out_a <= kClassRankCap && class_lever != 1) {
            // the cheaper route by the operation counts above, decided before either route runs. Above the sector cap
            // the class route is the only one (whatever its count). Ties go to the sector route.
            const bool only = X.ksec > kSectorRankCap;
            if (only || class_lever == 2 || class_route_cost() < sector_route_cost(X.ksec)) C.cls_route = true;
            if (C.cls_route) compute_classes();
        }
        // the sector route's cap, at the content build as before T4(b) (review F3): a content over it refuses here,
        // before RF_LOGICAL_RANK and before its build finishes (the structure is dropped; value_builds does not count it)
#ifdef XTIM_MUTATE_RF_RANKCAP
        if (!C.cls_route && X.ksec > kSectorRankCap + 1)                 // mutation build only: the cap off by one
#else
        if (!C.cls_route && X.ksec > kSectorRankCap)
#endif
            throw std::runtime_error("RecordFidelity: RF_SECTOR_RANK: a plan content's rotations reach " +
                                     std::to_string(X.ksec) + " independent rest sectors that no record fixes (2^" +
                                     std::to_string(X.ksec) + " kernel terms per shot; cap 2^" +
                                     std::to_string(kSectorRankCap) + ") — refused");
        if (C.cls_route) {
            C.cls_rec.clear(); C.cls_so.clear(); C.cls_img.clear();
            std::vector<uint64_t> I((size_t)2 * W);
            for (size_t u = 0; u < C.use.size(); ++u) {
                int ph = 0;
                used_img(C, u, I.data(), ph);
                Pauli R(n);
                for (int w = 0; w < W; ++w) { R.x[(size_t)w] = I[(size_t)w]; R.z[(size_t)w] = I[(size_t)(W + w)]; }
                R.phase = ph & 3;
                C.cls_rec.push_back(std::move(R));
            }
            for (const Pauli& g : so_gens) C.cls_so.push_back(img(pull3b(C, g)));
#ifdef XTIM_MUTATE_EXPLAW_CLS_UNDRESSED
            for (const Pauli& Pc : cls_ops) C.cls_img.push_back(img(Pc));   // mutation build only: C0's dressing dropped
#else
            for (const Pauli& Pc : cls_ops) C.cls_img.push_back(img(pull3b(C, Pc)));
#endif
        }
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
            int rph = 0;
            used_img(C, (size_t)u, I, rph);
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
            release_c(C, C.uimg.capacity() * 8 + C.uph.capacity() + C.uix.capacity() * 4);   // charged by classify_3b
            std::vector<uint64_t>().swap(C.uimg);
            std::vector<uint8_t>().swap(C.uph);
            std::vector<int>().swap(C.uix);
        }
#endif
        std::vector<uint64_t>().swap(X.kern);
        X.full = true;
        ++st.value_builds;
        if (st.dker_hist.size() <= (size_t)X.KD.d) st.dker_hist.resize((size_t)X.KD.d + 1, 0);
        ++st.dker_hist[(size_t)X.KD.d];
        release_c(C, nb0);
        charge_c(C, X.bytes(), "a content's expansion structure");
        if (C.cls_route) {
            size_t m = 0;
            auto pb = [](const Pauli& Q) { return (Q.x.capacity() + Q.z.capacity()) * 8 + sizeof(Pauli); };
            for (const Pauli& Q : C.cls_rec) m += pb(Q);
            for (const Pauli& Q : C.cls_so) m += pb(Q);
            for (const Pauli& Q : C.cls_img) m += pb(Q);
            charge_c(C, m, "a content's class-route images");
            ++st.class_contents;                                // counted once the build has committed
        }
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
                        std::vector<uint64_t> Iu((size_t)2 * W);
                        int ph = 0;
                        used_img(C, (size_t)u, Iu.data(), ph);
                        for (int w = 0; w < W; ++w) { R.x[(size_t)w] = Iu[(size_t)w]; R.z[(size_t)w] = Iu[(size_t)(W + w)]; }
                        R.phase = ph;
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

    // ── route 3b per shot. consistent_new: the shot's constraints' particular solution t_0 (false: ORTHOGONAL, F = 0
    // exactly); coset_new: the coset representative's sector σ = σ_P0 ⊕ σ(t_0); key_new: the memo key's tail;
    // value_new: F.
    std::vector<uint64_t> nsig_, nimD_, nrecD_, nrhs_, nt0_, nes_, nsg_, nim_, nrec_, ny_, ncross_, nacc_, npi_;
    // stage-1 per row: b = b(P', y) ⊕ c0, then N^T b (consistency) and R b (nt0_); nsig_ = σ(P') ⊕ σ_0. A row whose b
    // equals the content's previous b reuses that projection (decoded rows repeat).
    // scaling F3: a lane running a chunk of a content's rows keeps the previous-row projection in its OWN cache (the
    // content's NewC copy is shared by the lanes of a split content); the projection is a pure function of b, so a
    // row's t_0 and consistency are the same whichever row came before (the cache only skips recomputing them).
    struct RowCache {
        std::vector<uint64_t> lb, lt;
        int last_ok = -1;
    };
    RowCache* rc_ = nullptr;
    bool consistent_new(NewC& X, const uint64_t* px, const uint64_t* pz, const uint64_t* y) {
        img_bits(px, pz, npi_);                                         // P''s frame image (bits; its phase is global)
        nsig_.assign((size_t)WC, 0);
        syndrome_of(px, pz, nsig_.data());
        nimD_.assign((size_t)2 * W, 0);
        nrecD_.assign((size_t)MW, 0);
        dest_product(nsig_.data(), nimD_.data(), nrecD_.data());
        for (int w = 0; w < WC; ++w) nsig_[(size_t)w] ^= X.sig0[(size_t)w];
        nrhs_.resize((size_t)X.CW);
        const uint64_t* c0 = X.c0();
        if (rc_ && rc_->lb.size() != (size_t)X.CW) {
            rc_->lb.assign((size_t)X.CW, 0);
            rc_->lt.assign((size_t)X.WV, 0);
            rc_->last_ok = -1;
        }
        uint64_t* lb = rc_ ? rc_->lb.data() : X.lastb();
        uint64_t* lt = rc_ ? rc_->lt.data() : X.lastt();
        int& last_ok = rc_ ? rc_->last_ok : X.last_ok;
        for (int w = 0; w < W; ++w)
            nrhs_[(size_t)w] = ((npi_[(size_t)w] ^ nimD_[(size_t)w]) & ~freemask[(size_t)w]) ^ c0[w];
        for (int w = 0; w < MW; ++w)
            nrhs_[(size_t)(W + w)] = ((y[w] ^ rssignw[(size_t)w] ^ nrecD_[(size_t)w]) & rsmask[(size_t)w]) ^ c0[W + w];
#ifdef XTIM_MUTATE_RF_LASTB_STALE
        const int cmpw = 1;                                             // mutation build only: a one-word compare
#else
        const int cmpw = X.CW;
#endif
        if (last_ok >= 0 && std::memcmp(nrhs_.data(), lb, (size_t)cmpw * 8) == 0) {
            ++st.projection_skips;
            nt0_.assign(lt, lt + X.WV);
            return last_ok != 0;
        }
        std::copy(nrhs_.begin(), nrhs_.end(), lb);
        nt0_.assign((size_t)X.WV, 0);
        for (size_t r = 0; r < X.pbit.size(); ++r)
            if (gs::getbit(lb, X.pbit[r])) {                            // b's pivot bits (rows fully reduced)
                for (int w = 0; w < X.CW; ++w) nrhs_[(size_t)w] ^= X.prow[r * X.CW + w];
                for (int w = 0; w < X.WV; ++w) nt0_[(size_t)w] ^= X.ppay[r * X.WV + w];
            }
        int ok = 1;
#ifndef XTIM_MUTATE_RF_ORTH_SKIP
        for (int w = 0; w < X.CW; ++w) if (nrhs_[(size_t)w]) { ok = 0; break; }
#endif                                                                  // mutation build only: every row consistent
        std::copy(nt0_.begin(), nt0_.end(), lt);
        last_ok = ok;
        return ok != 0;
    }
    // stage-2 per row (consistent rows): the coset representative's sector σ = σ_P0 ⊕ σ(t_0), reduced to the coset's
    // smallest σ (t_0 moved with it)
    void coset_new(const NewC& X) {
        for (int a = 0; a < X.nG; ++a)
            if (gs::getbit(nt0_.data(), X.nM + a)) for (int w = 0; w < WC; ++w) nsig_[(size_t)w] ^= X.gsig[(size_t)a * WC + w];
        for (int r = 0; r < X.ksec; ++r)                                // the coset's smallest σ
            if (gs::getbit(nsig_.data(), X.spiv[(size_t)r])) {
                for (int w = 0; w < WC; ++w) nsig_[(size_t)w] ^= X.ssig[(size_t)r * WC + w];
                for (int w = 0; w < X.WV; ++w) nt0_[(size_t)w] ^= X.spay[(size_t)r * X.WV + w];
            }
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

    // ── threads T6 ──
    Impl() = default;
    Impl(const Impl&) = default;                     // a lane: Shared shared, the maps fresh, every scratch its own
    Impl& operator=(const Impl&) = delete;
    struct LaneSet {
        std::vector<std::unique_ptr<Impl>> lanes;
    };
    struct LaneHold {                                // a copy holds no lanes
        std::unique_ptr<LaneSet> p;
        LaneHold() = default;
        LaneHold(const LaneHold&) noexcept {}
        LaneHold& operator=(const LaneHold&) = delete;
    };
    LaneHold lanes_;
    int threads_req_ = 1, threads_eff_ = 1;
    std::string threads_limit_;

    struct Shared {
        CliffordTableau U{0};
        std::vector<uint64_t> eps, freemask;
        std::vector<int> freepos;
        std::vector<std::complex<double>> b;
        std::vector<uint64_t> bsupp;
        std::vector<std::complex<double>> lam_tab;
        std::vector<uint64_t> lam_done;
        std::vector<int> T_free_;
        std::vector<Pauli> recops;
        std::vector<int> rsw_off, rsw_w;
        std::vector<uint64_t> rsw_x, rsw_z;
        std::vector<Pauli> rec_lab_img;
        std::vector<uint8_t> rs, rs_sign;
        std::vector<uint64_t> omask, rmask;
        std::vector<Pauli> star, star_img0;
        std::vector<uint64_t> dimg, drec, colx, colz, rsmask, rssignw;
        std::unordered_map<HKey, int, HKeyHash, HKeyEq> l3_ids;
        std::vector<L3Content> l3c;
        std::vector<Rank> rk;                        // per slot (parallel to l3c): the ranking's rows, live flag
        std::vector<int> dead_ids, free_ids;         // speed fix 4: tombstoned since the last recycle; recyclable slots
        std::vector<int> rn_ind;
        std::vector<uint64_t> invrows;
        std::vector<uint8_t> invph;
        std::vector<int> owires;                     // explaw T4(b)
        std::vector<Pauli> so_gens, cls_ops;
        std::vector<double> cls_val;
        int cls_state = 0;                           //   0 not computed, 1 computed (read and written under cls_mu)
        int class_lever = 0;                         //   test lever (set between calls only)
        std::mutex cls_mu;
    };
};

void RecordFidelity::Impl::used_img(const L3Content& C, size_t u, uint64_t* I, int& ph) const {
    const int j = C.uix[u];
    if (j >= 0) {
        for (int t = 0; t < 2 * W; ++t) I[t] = C.uimg[(size_t)j * 2 * W + t];
        ph = C.uph[(size_t)j];
        return;
    }
    const Pauli& L = rec_lab_img[(size_t)C.use[u]];
    for (int w = 0; w < W; ++w) { I[w] = L.x[(size_t)w]; I[W + w] = L.z[(size_t)w]; }
    ph = L.phase & 3;
}

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
        I.owires = out_wires;
        std::sort(I.owires.begin(), I.owires.end());
        for (const auto& z : ER.zero) I.so_gens.push_back(pauli_of(z, n, W));   // explaw T4(b): S_O = S0 ∩ P_O
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
    I.charge_s(I.held_rest, "the register's rest structure");
}
// The content's record images (dressed by C0 where it touches them) and its R_n / used / undressed split (stage 2).
// The used records (R_n, and the R_s records C0 dresses) with their frame images, flat (C.uimg / C.uph) in record
// order; kind 0 dresses by the exact Lemma 1 (D† M D = i^{p − q(x)} X^x Z^{z ⊕ Ãx}, dress_diag) and images by the
// per-wire rows, a factor list by its pull-back from the stored factors (fac_pull). ux / uz: the union X / Z support
// of C0's factor terms (an operator missing it commutes with C0: undressed, no pull-back).
void RecordFidelity::Impl::classify_3b(L3Content& C) {
    const size_t b0 = C.bytes();
    std::vector<uint64_t>& zz = cl_zz_;
    std::vector<uint64_t>& im = cl_im_;
    std::vector<uint64_t>& ux = cl_ux_;
    std::vector<uint64_t>& uz = cl_uz_;
    zz.resize((size_t)W);
    im.resize((size_t)2 * W);
    ux.assign((size_t)W, 0);
    uz.assign((size_t)W, 0);
    if (C.dex) {                                         // C0 = S^{a&1}·CZ touches X on its wires
        for (int q = 0; q < n; ++q)
            if (C.dex->a1[(size_t)q] || C.dex->rowof[(size_t)q] >= 0) uz[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    } else {
        for (int j = 0; j < C.fac.V; ++j) {
            const uint64_t* B = C.fac.bits(j, W);
            for (int w = 0; w < W; ++w) { ux[(size_t)w] |= B[w]; uz[(size_t)w] |= B[W + w]; }
        }
    }
    Pauli& pb = cl_pb_;
    for (size_t c = 0; c < recops.size(); ++c) {
        const Pauli& M = recops[c];
        bool t = false;
        int ph = M.phase & 3;
        if (C.dex) {
            t = dex_dress(*C.dex, uz.data(), (int)c, zz.data(), ph);
            if (t) img_full(M.x.data(), zz.data(), ph, im.data(), ph);
        } else {
            bool touch = false;
            for (int w = 0; w < W && !touch; ++w)
                if ((M.x[(size_t)w] & (uz[w] | ux[w])) || (M.z[(size_t)w] & ux[w])) touch = true;
            if (touch) {
                pb = M;                                         // (copy-assign: pb's capacity is reused)
                fac_pull(C.fac, pb);
                t = !(((pb.phase ^ M.phase) & 3) == 0 && pb.x == M.x && pb.z == M.z);
            }
            if (t) {
                const Pauli Q = img(pb);
                for (int w = 0; w < W; ++w) { im[(size_t)w] = Q.x[(size_t)w]; im[(size_t)(W + w)] = Q.z[(size_t)w]; }
                ph = Q.phase & 3;
            }
        }
        const bool rn = !rs[c];
#ifdef XTIM_MUTATE_RF_RS_ALL_UNDRESSED
        const bool used = rn;                                          // mutation build only: dressed R_s as δ checks
#else
        const bool used = rn || t;
#endif
        if (!used) { C.undressed.push_back((int)c); continue; }
        C.use.push_back((int)c);
        if (rn) C.rn.push_back((int)c);
        if (t) {
            C.uix.push_back((int)C.uph.size());
            C.uimg.insert(C.uimg.end(), im.begin(), im.end());
            C.uph.push_back((uint8_t)ph);
        } else {
            C.uix.push_back(-1);                                       // the shared lab image (rec_lab_img)
        }
    }
    C.classified = true;
    release_c(C, b0);
    charge_c(C, C.bytes(), "a content's record split");
}
void RecordFidelity::Impl::lab_init_() {
    Impl& I = *this;
    auto pbytes = [](const Pauli& P) { return (P.x.capacity() + P.z.capacity()) * 8 + sizeof(Pauli); };
    if (I.rec_lab_img.empty() && !I.recops.empty()) {
        size_t m = 0;
        if (I.lab_pre_.size() == I.recops.size()) {         // F1: the values a build already used (the same images)
            I.rec_lab_img = std::move(I.lab_pre_);
            I.lab_pre_.clear();
            for (const Pauli& P : I.rec_lab_img) m += pbytes(P);
        } else {
            for (const Pauli& M : I.recops) { I.rec_lab_img.push_back(I.img(M)); m += pbytes(I.rec_lab_img.back()); }
        }
        I.charge_s(m, "the record operators' frame images");
    }
}
int RecordFidelity::Impl::intern_(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                                  const std::vector<uint64_t>& ux, const std::vector<uint64_t>& uz,
                                  FacList&& fac, std::unique_ptr<DiagExpansion>&& dex) {
    lab_init_();
    L3Content C = build_l3_(pull, ux, uz, std::move(fac), std::move(dex), rec_lab_img);
    return install_l3_(key.capacity(), std::string(key), key_hash_(key), std::move(C));
}
RecordFidelity::Impl::L3Content RecordFidelity::Impl::build_l3_(const std::function<Pauli(const Pauli&)>& pull,
                                                                const std::vector<uint64_t>& ux,
                                                                const std::vector<uint64_t>& uz,
                                                                FacList&& fac,
                                                                std::unique_ptr<DiagExpansion>&& dex,
                                                                const std::vector<Pauli>& lab) const {
    const Impl& I = *this;
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
    if (I.route == 2) {
        // 3b: the used records' split and images, deferred to the content's first consistent row (stage 2)
        C.classified = false;
    } else {
        C.rec_img.resize(I.recops.size());
        std::vector<uint8_t> rec_dressed(I.recops.size(), 0);
        for (size_t c = 0; c < I.recops.size(); ++c) {
            rec_dressed[c] = dressed(I.recops[c]) ? 1 : 0;
            if (rec_dressed[c]) C.rec_img[c] = I.img(pb);
            else if (!I.rs[c]) C.rec_img[c] = lab[c];
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
    return C;
}
int RecordFidelity::Impl::install_l3_(size_t key_cap, std::string&& key, size_t h, L3Content&& C, size_t cbytes) {
    Impl& I = *this;
    const int id = I.free_ids_.empty() ? (int)I.l3c.size() : I.free_ids_.back();   // speed fix 4: a recycled slot first
    // fid-routine B5: C.bytes() as its prepare measured it (the same structure, moved), else measured here
    const size_t cb = (cbytes != SIZE_MAX ? cbytes : C.bytes()) + key_cap + 64;
    I.charge(I.held_l3, cb, "a content");        // §5: check-then-add — a refusal installs nothing
    C.charged = cb;                              // review L2: the content's ledger opens with its install charge
    C.key_cap = key_cap;
    C.born_call = I.call_no_;
    C.kh = h;
    if (id < (int)I.l3c.size()) {
        I.l3c[(size_t)id] = std::move(C);
        I.free_ids_.pop_back();
        --I.dead_count_;
    } else {
        I.l3c.push_back(std::move(C));
        I.rk.resize(I.l3c.size());
    }
    I.rk[(size_t)id] = Rank();
    I.rk[(size_t)id].live = true;
    I.rk[(size_t)id].seq = ++I.install_seq_;
    I.born_ids_.push_back(id);
    const auto r = I.l3_ids.emplace(HKey{std::move(key), h}, id);
    if (!r.second) throw std::logic_error("RecordFidelity: a content installed twice (engine bug)");
    I.l3c[(size_t)id].keyp = &r.first->first.k;      // (node-stable: the index never moves a stored key)
    ++I.st.plans;
    return id;
}
int RecordFidelity::intern_content(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                                   const std::vector<Factor>& factors) {
    Impl& I = *impl_;
    if (const int id = I.find_l3_(key, key_hash_(key)); id >= 0) return id;
    return I.intern_fac_(key, pull, I.parse_factors(factors));   // the factor list's shape (RF_FACTOR_NOT_CLIFFORD by name)
}
int RecordFidelity::intern_content_rot(const std::string& key, const std::function<Pauli(const Pauli&)>& pull,
                                       const std::vector<const Pauli*>& axes) {
    Impl& I = *impl_;
    if (const int id = I.find_l3_(key, key_hash_(key)); id >= 0) return id;
    return I.intern_fac_(key, pull, I.parse_rots(axes));
}
int RecordFidelity::intern_content_diag(const std::string& key, const DiagNormalForm& nf) {
    Impl& I = *impl_;
    if (const int id = I.find_l3_(key, key_hash_(key)); id >= 0) return id;
    auto dex = std::make_unique<DiagExpansion>(expand_diag(nf, I.n));   // RF_CONTENT_MALFORMED
    I.set_pre_(*dex, nf.prefix);
    std::vector<uint64_t> ux((size_t)I.W, 0), uz((size_t)I.W, 0);       // C0 = S^{a&1}·CZ touches X on its wires
    for (int q = 0; q < I.n; ++q)
        if (dex->a1[(size_t)q] || dex->rowof[(size_t)q] >= 0) uz[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    const DiagExpansion& E = *dex;
    return I.intern_(key, [&](const Pauli& Q) { return dress_diag(E, Q); }, ux, uz, {}, std::move(dex));
}
struct RecordFidelity::PreparedContent {
    std::exception_ptr pre;                  // a refusal before the intern's first effect (malformed, not Clifford)
    std::exception_ptr build;                // a refusal while building (after the record images' fill)
    Impl::L3Content C;
    size_t cbytes = SIZE_MAX;                // fid-routine B5: C.bytes(), measured on the build's thread
};
void RecordFidelity::prepare_begin() {
    Impl& I = *impl_;
    if (I.rec_lab_img.empty() && !I.recops.empty() && I.lab_pre_.size() != I.recops.size()) {
        I.lab_pre_.clear();
        for (const Pauli& M : I.recops) I.lab_pre_.push_back(I.img(M));
    }
}
bool RecordFidelity::prepare_parallel_ok() const {
    const Impl& I = *impl_;
    return !I.max_bytes && !I.row_log_on && !I.hform_oracle;
}
std::shared_ptr<RecordFidelity::PreparedContent> RecordFidelity::prepare_content_diag(const DiagNormalForm& nf) const {
    const Impl& I = *impl_;
    auto P = std::make_shared<PreparedContent>();
    std::unique_ptr<DiagExpansion> dex;
    try {
        dex = std::make_unique<DiagExpansion>(expand_diag(nf, I.n));   // RF_CONTENT_MALFORMED
        I.set_pre_(*dex, nf.prefix);
    } catch (...) {
        P->pre = std::current_exception();
        return P;
    }
    std::vector<uint64_t> ux((size_t)I.W, 0), uz((size_t)I.W, 0);       // as intern_content_diag
    for (int q = 0; q < I.n; ++q)
        if (dex->a1[(size_t)q] || dex->rowof[(size_t)q] >= 0) uz[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    const DiagExpansion& E = *dex;
    try {
        P->C = I.build_l3_([&](const Pauli& Q) { return dress_diag(E, Q); }, ux, uz, {}, std::move(dex),
                           I.rec_lab_img.empty() ? I.lab_pre_ : I.rec_lab_img);
        P->cbytes = P->C.bytes();
    } catch (...) {
        P->build = std::current_exception();
    }
    return P;
}
std::shared_ptr<RecordFidelity::PreparedContent> RecordFidelity::Impl::prepare_fac_(
        const std::function<Pauli(const Pauli&)>& pull, const std::function<FacList()>& parse) const {
    const Impl& I = *this;
    auto P = std::make_shared<PreparedContent>();
    FacList L;
    try {
        L = parse();                                 // the factor list's shape (RF_FACTOR_NOT_CLIFFORD by name)
    } catch (...) {
        P->pre = std::current_exception();
        return P;
    }
    std::vector<uint64_t> ux, uz;                    // as intern_content
    I.fac_union_(L, ux, uz);
    try {
        P->C = I.build_l3_(pull, ux, uz, std::move(L), nullptr, I.rec_lab_img.empty() ? I.lab_pre_ : I.rec_lab_img);
        P->cbytes = P->C.bytes();
    } catch (...) {
        P->build = std::current_exception();
    }
    return P;
}
std::shared_ptr<RecordFidelity::PreparedContent> RecordFidelity::prepare_content_rot(
        const std::function<Pauli(const Pauli&)>& pull, const std::vector<const Pauli*>& axes) const {
    const Impl& I = *impl_;
    return I.prepare_fac_(pull, [&]() { return I.parse_rots(axes); });
}
std::shared_ptr<RecordFidelity::PreparedContent> RecordFidelity::prepared_refused(std::exception_ptr e) const {
    auto P = std::make_shared<PreparedContent>();
    P->pre = std::move(e);
    return P;
}
int RecordFidelity::install_prepared(std::string&& key, size_t h, PreparedContent& p) {
    Impl& I = *impl_;
    const int id = I.find_l3_(key, h);
    if (id >= 0) return id;
    if (p.pre) std::rethrow_exception(p.pre);
    I.lab_init_();                                   // intern_'s first step (charged)
    if (p.build) std::rethrow_exception(p.build);
    const size_t cap = key.capacity();
    return I.install_l3_(cap, std::move(key), h, std::move(p.C), p.cbytes);
}
void RecordFidelity::reserve_installs(size_t n) {
    Impl& I = *impl_;                                // fid-routine B5: no regrowth / rehash inside the install loop
    const size_t free_n = I.free_ids_.size(), grow = n > free_n ? n - free_n : 0;
    // (geometric, as push_back's own growth: an exact reserve per call would move every slot each call it grows)
    if (I.l3c.size() + grow > I.l3c.capacity()) I.l3c.reserve(std::max(I.l3c.size() + grow, 2 * I.l3c.capacity()));
    if (I.rk.size() + grow > I.rk.capacity()) I.rk.reserve(std::max(I.rk.size() + grow, 2 * I.rk.capacity()));
    const size_t want = I.l3_ids.size() + n;        // (grow only: a reserve to another bucket count rehashes)
    if ((double)want > (double)I.l3_ids.bucket_count() * (double)I.l3_ids.max_load_factor()) I.l3_ids.reserve(want);
}
size_t RecordFidelity::content_key_hash(const std::string& key) { return key_hash_(key); }
int RecordFidelity::find_content(const std::string& key, size_t h) const { return impl_->find_l3_(key, h); }
bool RecordFidelity::has_content(const std::string& key, int* id) const {
    const int f = impl_->find_l3_(key, key_hash_(key));
    if (f < 0) return false;
    if (id) *id = f;
    return true;
}
double RecordFidelity::evaluate_record(int content, const uint64_t* px, const uint64_t* pz, const uint64_t* y,
                                       const uint64_t* qx, const uint64_t* qz) {
    return impl_->evaluate_record(content, px, pz, y, qx, qz);
}
int RecordFidelity::route() const { return impl_->route; }

namespace {
void add_stats_(RecordFidelityStats& a, RecordFidelityStats& b) {
    a.shots += b.shots; a.memo_hits += b.memo_hits; a.misses += b.misses; a.plans += b.plans;
    a.max_dim_bits = std::max(a.max_dim_bits, b.max_dim_bits);
    a.rn_words += b.rn_words; a.kernel_compiles += b.kernel_compiles; a.rn_misses += b.rn_misses;
    a.rotation_misses += b.rotation_misses; a.sectors += b.sectors; a.max_sectors = std::max(a.max_sectors, b.max_sectors);
    a.sectors_nonzero += b.sectors_nonzero; a.max_r = std::max(a.max_r, b.max_r);
    a.max_kappa_prime = std::max(a.max_kappa_prime, b.max_kappa_prime);
    a.orthogonal_rows += b.orthogonal_rows; a.projection_skips += b.projection_skips; a.value_builds += b.value_builds;
    a.max_ksec = std::max(a.max_ksec, b.max_ksec); a.class_contents += b.class_contents;   // explaw T4(b)
    a.class_rows += b.class_rows; a.classes_nonzero = std::max(a.classes_nonzero, b.classes_nonzero);
    auto addv = [](std::vector<long>& x, const std::vector<long>& y) {
        if (x.size() < y.size()) x.resize(y.size(), 0);
        for (size_t i = 0; i < y.size(); ++i) x[i] += y[i];
    };
    addv(a.r_hist, b.r_hist); addv(a.kappa_prime_hist, b.kappa_prime_hist); addv(a.dker_hist, b.dker_hist);
    const int route = b.route, out_a = b.out_a;
    b = RecordFidelityStats{};
    b.route = route;
    b.out_a = out_a;
}
}  // namespace

void RecordFidelity::evaluate_records(long rows, const int* content, const uint64_t* prec, int PNW, const uint64_t* y,
                                      int YW, const uint8_t* corr, const std::vector<int>* out_wires, double* out,
                                      int threads) {
    Impl& H = *impl_;
    H.threads_req_ = std::max(1, threads);
    H.threads_eff_ = 1;
    H.threads_limit_.clear();
    const int no = out_wires ? (int)out_wires->size() : 0;
    auto row_eval = [&](Impl& I, long i, std::vector<uint64_t>& cq) -> double {
        const uint64_t* pr = prec + (size_t)i * 2 * (size_t)PNW;
        const uint64_t* yr = y + (size_t)i * (size_t)YW;
        if (corr) {                                  // the decoder's correction P_c on OUTPUT_QUBITS
            const uint8_t* cr = corr + (size_t)i * 2 * (size_t)no;
            std::fill(cq.begin(), cq.end(), 0);
            for (int j = 0; j < no; ++j) {
                const int w = (*out_wires)[(size_t)j];
                if (cr[j] & 1) cq[(size_t)(w >> 6)] |= 1ULL << (w & 63);
                if (cr[no + j] & 1) cq[(size_t)PNW + (size_t)(w >> 6)] |= 1ULL << (w & 63);
            }
        }
        return I.evaluate_record(content[i], pr, pr + PNW, yr, corr ? cq.data() : nullptr,
                                 corr ? cq.data() + PNW : nullptr);
    };
    // the contents' rows, heaviest content first, each to the least loaded lane (deterministic)
    std::vector<long> cnt(H.l3c.size(), 0);
    for (long i = 0; i < rows; ++i) {
        if (content[i] < 0 || content[i] >= (int)cnt.size())
            throw std::out_of_range("RecordFidelity: content index out of range");
        ++cnt[(size_t)content[i]];
    }
    long distinct = 0;
    for (long c : cnt) distinct += c > 0;
    std::string why;
    if (threads > 1) {
        if (H.max_bytes) why = "max_bytes (the byte budget's refusal is decided in row order: serial)";
        else if (H.row_log_on || H.hform_oracle) why = "a test lever (serial)";
    }
    // scaling F3: the work items — a content's rows, or (a content of more than two chunks of rows: SPLIT) its rows in
    // chunks that run on different lanes
    const long chunk = std::max<long>(256, (rows + 8L * threads - 1) / (8L * std::max(threads, 1)));
    long nitems = 0;
    for (long c : cnt) if (c) nitems += c > 2 * chunk ? (c + chunk - 1) / chunk : 1;
    if (threads > 1 && why.empty() && nitems < 2)
        why = "one record content of few rows (its rows run in row order on one thread)";
    (void)distinct;
    int Te = (int)std::min<long>(threads, nitems);
    const bool serial_why = !why.empty();
    if (Te > 1 && !serial_why) {                     // fix F4: a thread the OS refuses caps the fan-out (reported)
        const int w = ThreadPool::instance().reserve(Te);
        if (w + 1 < Te) {
            Te = w + 1;
            why = "the OS gave the worker pool " + std::to_string(w) + " thread(s)";
        }
    }
    if (threads <= 1 || serial_why || Te <= 1) {
        std::vector<uint64_t> cq(corr ? (size_t)2 * PNW : 0);
        for (long i = 0; i < rows; ++i) out[i] = row_eval(H, i, cq);
        H.threads_limit_ = why;
        return;
    }
    // the work: per content its rows in row order (CSR), heaviest content first; a lane pulls the next item (a lane the
    // pool withdraws takes none: the others pull its share)
    std::vector<int> ord;
    for (size_t c = 0; c < cnt.size(); ++c) if (cnt[c]) ord.push_back((int)c);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[(size_t)a] > cnt[(size_t)b]; });
    std::vector<long> off(cnt.size() + 1, 0);
    for (size_t c = 0; c < cnt.size(); ++c) off[c + 1] = off[c] + cnt[c];
    std::vector<long> crow((size_t)rows);
    {
        std::vector<long> fill(off.begin(), off.end() - 1);
        for (long i = 0; i < rows; ++i) crow[(size_t)fill[(size_t)content[i]]++] = i;
    }
    // F3: a split content's lazy structure is completed FIRST, on the home, in row order (route 3b: stage 1 at its first
    // row, stage 2 at its first consistent row — both pure functions of the content; their refusal stands at its row);
    // its remaining rows then run in chunks on any lane, each lane with its own previous-row projection cache. Every
    // other per-row input is the row's own or a memo / cache keyed by what the value reads (pure functions of the key).
    struct Item { int c; long r0, r1; bool split; };
    std::vector<Item> items;
    long home_err_row = -1;
    std::exception_ptr home_err;
    // the lanes: copies of the evaluator (Shared shared; memo, caches, scratch and counters their own)
    if (!H.lanes_.p) H.lanes_.p.reset(new Impl::LaneSet());
    auto& LS = *H.lanes_.p;
    static_assert(std::is_copy_constructible<Impl>::value, "threads: a lane is a copy of the evaluator");
    while ((int)LS.lanes.size() < Te) {
        std::unique_ptr<Impl> L(new Impl(H));
        L->home_ = &H;
        L->row_log.clear();
        LS.lanes.push_back(std::move(L));
    }
    // fid-routine B1: each lane's heap headroom for its share of this call's builds (structures, memo, caches): the
    // largest lane growth of the previous threaded call, scaled to this call's rows (speed only: no value reads it)
    // (at most kLaneReserveCap per lane: one glibc arena heap; transient headroom, not retained cache)
    const size_t ev_reserve = H.ev_rows_ > 0
        ? (size_t)std::min<unsigned __int128>((unsigned __int128)H.ev_grow_ * (unsigned __int128)rows / (unsigned __int128)H.ev_rows_ * 5 / 4,
                                              (unsigned __int128)kLaneReserveCap)
        : 0;
    std::vector<char> ev_reserved((size_t)Te, 0);
    auto ev_headroom = [&](int l) {
        if (ev_reserve && !ev_reserved[(size_t)l]) { ev_reserved[(size_t)l] = 1; lane_heap_reserve(ev_reserve); }
    };
    for (int l = 0; l < Te; ++l) {
        Impl& L = *LS.lanes[(size_t)l];
        L.st = RecordFidelityStats{};
        L.st.route = H.st.route;
        L.st.out_a = H.st.out_a;
        L.held_memo = L.held_l3 = L.held_rest = L.held_ov_ = L.held_shared_ = 0;
        L.lam_used_ = false;
        L.max_bytes = 0;
        // fid-routine fix round 1: the lane's overlays start empty at every call (the merge empties them; a call that
        // threw before its merge must not leave a later call reading another content's entries under a recycled id)
        if (!L.memo.empty()) { L.memo.clear(); L.memo.rehash(0); }
        if (!L.ny_cache.empty()) { L.ny_cache.clear(); L.ny_cache.rehash(0); }
        if (!L.lam_sparse.empty()) { L.lam_sparse.clear(); L.lam_sparse.rehash(0); }
    }
    std::vector<long> err_row((size_t)Te, -1);
    std::vector<std::exception_ptr> err((size_t)Te);
    // F3: per lane, the counter deltas of each memo key it computed on a split content's rows (two lanes may compute
    // the same key: the merge keeps one entry and folds the counters back to one computation, as the serial loop's)
    struct Delta { long rn, rot, sectors, nonzero, cls; };
    std::vector<std::unordered_map<std::string, Delta>> dmaps((size_t)Te);
    std::fenv_t fe;
    std::fegetenv(&fe);
    std::vector<std::exception_ptr> lane_fail((size_t)Te);   // fix-2 N3: a lane's own failure outside a row
    // a computed memo key's counter deltas, on a split content's rows (false: allocation failed, the row's refusal)
    auto note_delta = [&](Impl& L, int l, const RecordFidelityStats& s0, long i) -> bool {
        try {
            dmaps[(size_t)l][L.key_scratch] = Delta{L.st.rn_misses - s0.rn_misses, L.st.rotation_misses - s0.rotation_misses,
                                                    L.st.sectors - s0.sectors, L.st.sectors_nonzero - s0.sectors_nonzero,
                                                    L.st.class_rows - s0.class_rows};
            return true;
        } catch (...) {
            if (err_row[(size_t)l] < 0 || i < err_row[(size_t)l]) {
                err_row[(size_t)l] = i;
                err[(size_t)l] = std::current_exception();
            }
            return false;
        }
    };
    // F3: a split content's lazy structure is completed FIRST, in row order (route 3b: stage 1 at its first row, stage 2
    // at its first consistent row — both pure functions of the content; their refusal stands at its row); its remaining
    // rows then run in chunks on any lane, each lane with its own previous-row projection cache. Every other per-row
    // input is the row's own or a memo / cache keyed by what the value reads (pure functions of the key).
    // fid-routine B5 (phase A): the split contents' completions run on the lanes, one content per item (the content's
    // own projection cache and structure, as the home's were: one thread per content), instead of serially on the home
    // (they grew with T: the chunk rows/(8T) splits more contents). Their memo keys carry their counter deltas, as a
    // split chunk's do (a chunk lane may compute the same key: one entry, one computation at the merge).
    std::vector<int> pre;
    for (int c : ord)
        if (cnt[(size_t)c] > 2 * chunk && H.route == 2) pre.push_back(c);
    std::vector<long> pre_end(pre.size());
    for (size_t t = 0; t < pre.size(); ++t) pre_end[t] = off[(size_t)pre[t]];
    bool wave_fail = false;
    if (!pre.empty()) {
        std::atomic<size_t> nexta{0};
        ThreadPool::instance().run(Te, [&](int l) {
            if (l) std::fesetenv(&fe);
            ScratchCallScope scope;
            Impl& L = *LS.lanes[(size_t)l];
            try {
                ev_headroom(l);
                std::vector<uint64_t> cq(corr ? (size_t)2 * PNW : 0);
                for (size_t t; (t = nexta.fetch_add(1, std::memory_order_relaxed)) < pre.size();) {
                    const int c = pre[t];
                    const Impl::L3Content& C = H.l3c[(size_t)c];
                    long r = off[(size_t)c];
                    while (r < off[(size_t)c + 1] && !(C.nc && C.nc->full)) {
                        const long i = crow[(size_t)r];
                        const RecordFidelityStats s0 = L.st;
                        try {
                            out[i] = row_eval(L, i, cq);
                        } catch (...) {
                            if (err_row[(size_t)l] < 0 || i < err_row[(size_t)l]) {
                                err_row[(size_t)l] = i;
                                err[(size_t)l] = std::current_exception();
                            }
                            r = off[(size_t)c + 1];
                            break;
                        }
                        if (L.st.misses != s0.misses && !note_delta(L, l, s0, i)) { r = off[(size_t)c + 1]; break; }
                        ++r;
                    }
                    pre_end[t] = r;
                }
            } catch (...) {                          // (an allocation outside a row: the call's failure)
                lane_fail[(size_t)l] = std::current_exception();
            }
            if (l) release_thread_scratch_slots();   // a worker's chi-scaled scratch dies with the call
        });
        for (auto& e : lane_fail) wave_fail = wave_fail || (bool)e;   // (its contents may be incomplete: no wave 2)
    }
    std::exception_ptr between_fail;                 // fix round 1: a throw between the waves (no wave 2; after the merge)
    try {
        if (H.fault_test_ == 1) { H.fault_test_ = 0; throw std::bad_alloc(); }   // test lever
        if (!pre.empty()) {
            bool lu = false;
            for (int l = 0; l < Te; ++l) lu = lu || LS.lanes[(size_t)l]->lam_used_;
            if (lu && H.lam_mode == 0) H.lam_decide_();  // as the home's first λ in phase A would
            // fix round 1: phase A's conditioned norms to the home before the chunks (as the home's phase A left them:
            // the chunk lanes read them there instead of recomputing); a norm two phase-A lanes computed is kept once
            // and counted once in rn_words (the home's phase A computed each new word once)
            for (int l = 0; l < Te; ++l) {
                Impl& L = *LS.lanes[(size_t)l];
                for (const auto& kv : L.ny_cache) {
                    const auto r = H.ny_cache.emplace(kv.first, kv.second);
                    if (r.second) H.charge_s(r.first->first.capacity() + sizeof(Impl::Scaled) + 64, "a conditioned-reference norm");
                    else --L.st.rn_words;
                }
                L.ny_cache.clear(); L.ny_cache.rehash(0);
            }
        }
    } catch (...) {
        between_fail = std::current_exception();
        wave_fail = true;
    }
    if (!wave_fail) {
        std::vector<Item> splits, whole;
        size_t pi = 0;
        for (int c : ord) {
            if (cnt[(size_t)c] <= 2 * chunk) { whole.push_back(Item{c, off[(size_t)c], off[(size_t)c + 1], false}); continue; }
            long r = off[(size_t)c];
            if (H.route == 2) r = pre_end[pi++];
            for (; r < off[(size_t)c + 1]; r += chunk) splits.push_back(Item{c, r, std::min(r + chunk, off[(size_t)c + 1]), true});
        }
        items = std::move(splits);
        items.insert(items.end(), whole.begin(), whole.end());
    }
    std::atomic<size_t> next{0};
    ThreadPool::instance().run(Te, [&](int l) {
        if (l) std::fesetenv(&fe);
        ScratchCallScope scope;
        Impl& L = *LS.lanes[(size_t)l];
        try {
        ev_headroom(l);
        std::vector<uint64_t> cq(corr ? (size_t)2 * PNW : 0);
        Impl::RowCache rc;
        // every item is run (a refused item stops at its refused row only): so every row before the earliest refusal is
        // evaluated, and that refusal is the serial loop's
        for (size_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < items.size();) {
            const Item& it = items[t];
            if (it.split) { rc.last_ok = -1; L.rc_ = &rc; }
            for (long r = it.r0; r < it.r1; ++r) {
                const long i = crow[(size_t)r];
                const RecordFidelityStats s0 = L.st;
                try {
                    out[i] = row_eval(L, i, cq);
                } catch (...) {
                    if (err_row[(size_t)l] < 0 || i < err_row[(size_t)l]) {
                        err_row[(size_t)l] = i;
                        err[(size_t)l] = std::current_exception();
                    }
                    break;
                }
                if (it.split && L.st.misses != s0.misses) {
                    try {                            // (an allocation failure: this row's, caught in the job)
                        dmaps[(size_t)l][L.key_scratch] = Delta{L.st.rn_misses - s0.rn_misses,
                                                                L.st.rotation_misses - s0.rotation_misses,
                                                                L.st.sectors - s0.sectors,
                                                                L.st.sectors_nonzero - s0.sectors_nonzero,
                                                                L.st.class_rows - s0.class_rows};
                    } catch (...) {
                        if (err_row[(size_t)l] < 0 || i < err_row[(size_t)l]) {
                            err_row[(size_t)l] = i;
                            err[(size_t)l] = std::current_exception();
                        }
                        break;
                    }
                }
            }
            L.rc_ = nullptr;
        }
        } catch (...) {                              // (an allocation outside a row: the call's failure)
            L.rc_ = nullptr;
            lane_fail[(size_t)l] = std::current_exception();
        }
        if (l) release_thread_scratch_slots();      // a worker's chi-scaled scratch dies with the call
    });
    // merge in lane order: the memo (content-disjoint: each content's entries come from its one lane), the conditioned
    // norms and λ entries (pure functions of their keys: a key two lanes computed is kept once), the counters
    bool lam_used = false;
    size_t grow = 0;                                 // fid-routine B1: the largest lane growth (charged bytes, net)
    for (int l = 0; l < Te; ++l) {
        const Impl& L = *LS.lanes[(size_t)l];
        const int64_t g = (int64_t)(L.held_l3 + L.held_memo + L.held_ov_);   // (modular net: a signed delta)
        if (g > 0) grow = std::max(grow, (size_t)g);
    }
    H.ev_grow_ = grow;
    H.ev_rows_ = rows;
    for (int l = 0; l < Te; ++l) {
        Impl& L = *LS.lanes[(size_t)l];
        H.held_l3 += L.held_l3;                     // F1: the lane's net structure bytes (modular: a signed delta)
        H.held_shared_ += L.held_shared_;           // (its shared part, the same way)
        lam_used = lam_used || L.lam_used_;
    }
    if (lam_used && H.lam_mode == 0) H.lam_decide_();   // as the serial loop's first λ would
    long dup = 0;
    Delta dd{0, 0, 0, 0, 0};
    {                                                // fid-routine B5: capacity only (no value reads the memo's order)
        size_t add = 0;
        for (int l = 0; l < Te; ++l) add += LS.lanes[(size_t)l]->memo.size();
        const size_t want = H.memo.size() + add;    // (grow only: a reserve to another bucket count rehashes)
        try {                                        // fix round 1: capacity only — a failed reserve is no failure
            if (H.fault_test_ == 2) { H.fault_test_ = 0; throw std::bad_alloc(); }   // test lever
            if (add && (double)want > (double)H.memo.bucket_count() * (double)H.memo.max_load_factor()) H.memo.reserve(want);
        } catch (const std::bad_alloc&) {
        }
    }
    for (int l = 0; l < Te; ++l) {
        if (l == 1 && H.fault_test_ == 3) { H.fault_test_ = 0; throw std::bad_alloc(); }   // test lever (fix round 2)
        Impl& L = *LS.lanes[(size_t)l];
        for (const auto& kv : L.memo) {
            const auto r = H.memo.emplace(kv.first, kv.second);
            if (r.second) {
                H.memo_track_(r.first, true);       // fix round 2: a throw erases the entry
                continue;
            }
            // F3: a key a lane before this one (or the home) holds: one entry, one computation in the counters
            const auto d = dmaps[(size_t)l].find(kv.first);
            if (d == dmaps[(size_t)l].end() ||
                std::memcmp(&r.first->second, &kv.second, sizeof(double)) != 0)
                throw std::logic_error("RecordFidelity: two lanes' memo entries disagree, or one of no split content "
                                       "(engine bug)");
            ++dup;
            dd.rn += d->second.rn; dd.rot += d->second.rot; dd.sectors += d->second.sectors; dd.nonzero += d->second.nonzero; dd.cls += d->second.cls;
        }
        for (const auto& kv : L.ny_cache) {
            const auto r = H.ny_cache.emplace(kv.first, kv.second);
            if (r.second) H.charge_s(r.first->first.capacity() + sizeof(Impl::Scaled) + 64, "a conditioned-reference norm");
        }
        if (H.lam_mode == 2)
            for (const auto& kv : L.lam_sparse) {
                const auto r = H.lam_sparse.emplace(kv.first, kv.second);
                if (r.second) H.charge_s(sizeof(Impl::LamKey) + 16 + 32, "a logical expectation");
            }
        L.memo.clear(); L.memo.rehash(0);
        L.ny_cache.clear(); L.ny_cache.rehash(0);
        L.lam_sparse.clear(); L.lam_sparse.rehash(0);
        add_stats_(H.st, L.st);
        L.held_memo = L.held_l3 = L.held_ov_ = L.held_shared_ = 0;
        L.lam_used_ = false;
    }
    H.st.misses -= dup;
    H.st.memo_hits += dup;
    H.st.rn_misses -= dd.rn;
    H.st.rotation_misses -= dd.rot;
    H.st.sectors -= dd.sectors;
    H.st.sectors_nonzero -= dd.nonzero;
    H.st.class_rows -= dd.cls;
    H.threads_eff_ = Te;
    H.threads_limit_ = why;
    for (auto& e : lane_fail) if (e) std::rethrow_exception(e);   // N3: a lane that could not run its share
    if (between_fail) std::rethrow_exception(between_fail);   // fix round 1: after the merge
    long fr = home_err_row;                          // the earliest row's refusal (the serial loop's)
    std::exception_ptr fe_ = home_err;
    for (int l = 0; l < Te; ++l)
        if (err_row[(size_t)l] >= 0 && (fr < 0 || err_row[(size_t)l] < fr)) { fr = err_row[(size_t)l]; fe_ = err[(size_t)l]; }
    if (fr >= 0) std::rethrow_exception(fe_);
}
int RecordFidelity::threads_requested() const { return impl_->threads_req_; }
int RecordFidelity::threads_effective() const { return impl_->threads_eff_; }
const std::string& RecordFidelity::threads_limit() const { return impl_->threads_limit_; }

void RecordFidelity::clear() { impl_->clear(); }
void RecordFidelity::clear_memo() { impl_->clear_memo(); }
bool RecordFidelity::trim_cache(size_t budget, long rows_next, int threads) {
    Impl& I = *impl_;
    I.recycle_dead_();                       // the previous call's tombstones (between calls only)
    // the droppable bytes only: the register's rest structure (held_rest) is never dropped, so it never triggers a pass
    if (!budget || I.held_memo + I.held_l3 - I.held_rest <= budget) return false;
    bool flushed = false;                    // review N3: counted when the pass dropped something (memo, norms, contents)
#ifndef XTIM_MUTATE_CB_RF_NOCLEAR
    // §5 F2: the conditioned-norm cache first (a pure function of its key), then the contents coldest first — rows
    // served in the last two calls, ties by id — each with its memo entries, until held <= budget − reserve
    // (saturating). Speed fix 4: a kept content keeps its memo entries (slots are recycled, never renumbered); the memo goes
    // whole only when everything goes (today's flush).
    flushed = !I.ny_cache.empty();
    {
        const size_t ny = I.ny_bytes_();
        I.held_l3 -= std::min(I.held_l3, ny);
        I.held_shared_ -= std::min(I.held_shared_, ny);
    }
    I.ny_cache.clear(); I.ny_cache.rehash(0);
    // speed fix 4: the reserve is the previous call's new-content bytes scaled to the coming call's rows (a 20k-row
    // call after a 400k-row one needs a twentieth of the room), and the pass evicts budget/8 further (hysteresis: a
    // full cache takes one pass every few calls instead of one per call, each with the pass's fixed costs)
    size_t reserve = I.reserve_;
    if (rows_next >= 0 && I.last_rows_ > 0)
        reserve = (size_t)((unsigned __int128)I.reserve_ * (unsigned __int128)rows_next / (unsigned __int128)I.last_rows_);
    const size_t room = reserve + budget / 8;
    const size_t target = budget > room ? budget - room : 0;
    const size_t held_c = I.held_memo + I.held_l3 - I.held_rest;
    const size_t need = held_c > target ? held_c - target : 0;
    // (the ledger: Σ live charged = held_l3 − held_shared_; the live count from the dead-slot count)
    const size_t charged_all = I.held_memo + (I.held_l3 - std::min(I.held_l3, I.held_shared_));
    const long live = (long)(I.l3c.size() - I.dead_count_);
    long n = 0;
    std::vector<int> order;
    if (need > 0 && need < charged_all) {    // a ranked prefix (speed fix 3: everything going needs no ranking)
        // FID fix 4 (owner 2026-11-08, replacing ruling (b)'s "most must go: one clear()"): always the ranked prefix —
        // the hot contents (served in the last two calls) survive every pass that does not need their bytes too, so
        // a full cache no longer rebuilds its hot contents after each flush (the 12 / 9 µs alternation of FID Table 1).
        // coldest first: rows served in the last two calls, ties by the last call that served it (least recently
        // used first), then by install order (the plan's "ties by content handle": handles were install-ordered
        // before slots were recycled)
        struct Key { long r; uint64_t last, seq; int id; };
        std::vector<Key> ks;
        ks.reserve((size_t)live);
        for (size_t i = 0; i < I.rk.size(); ++i) {
            const Rank& R = I.rk[i];
            if (R.live) ks.push_back(Key{R.rows_last2(I.call_no_), R.call_a, R.seq, (int)i});
        }
        std::sort(ks.begin(), ks.end(), [](const Key& a, const Key& b) {
#ifndef XTIM_MUTATE_RF_COLD_TAIL_KEPT
            if (a.r != b.r) return a.r < b.r;
#else
            if (a.r != b.r) return a.r > b.r;   // mutation build only (MUT-RF-COLD-TAIL-KEPT): the hot ones go first
#endif
            if (a.last != b.last) return a.last < b.last;
            return a.seq < b.seq;
        });
        size_t freed = 0;
        for (const Key& k : ks) {        // decide the prefix first, then act
            if (freed >= need) break;
            freed += I.l3c[(size_t)k.id].charged + I.memo_b_of(I.l3c[(size_t)k.id]);   // (with its memo entries)
            order.push_back(k.id);
        }
        n = (long)order.size();
    } else if (need > 0) {
        n = live;
    }
    if (n == live) {           // everything went: today's flush (ids reset, every cache dropped)
        flushed = flushed || n > 0 || !I.memo.empty();
        I.clear(threads);                    // fid-routine B2: its frees on the pool
    } else if (n > 0) {
        std::sort(order.begin(), order.end());   // (the evicted SET is the ranked prefix; slot order walks memory)
        // fid-routine B2: at threads > 1 the evicted structures are freed on the pool after the serial bookkeeping
        std::vector<Impl::L3Content> grave;
        if (threads > 1) grave.reserve(order.size());
        for (int id : order) I.tombstone_(id, threads > 1 ? &grave : nullptr);
        if (threads > 1) I.release_grave_(grave, threads);
        I.recycle_dead_();                   // (the evicted contents' memo entries go with them)
    }
    I.st.boundary_evictions += n;
    flushed = flushed || n > 0;
#else
    flushed = true;                          // mutation build only (MUT-CB-RF-NOCLEAR): counted, nothing dropped
#endif
    if (flushed) ++I.st.cache_flushes;
    return flushed;
}
void RecordFidelity::note_rows(const int* content, long rows) {
    Impl& I = *impl_;
    for (long i = 0; i < rows; ++i) {        // only the call's rows: a content's two most recent calls' counts
        if (content[i] < 0 || content[i] >= (int)I.l3c.size()) continue;
        Rank& C = I.rk[(size_t)content[i]];
        if (C.call_a != I.call_no_) { C.call_b = C.call_a; C.rows_b = C.rows_a; C.call_a = I.call_no_; C.rows_a = 0; }
        ++C.rows_a;
    }
    size_t nb = 0;                           // the reserve: the bytes of the contents this call installed (O(new))
    for (int id : I.born_ids_)
        if (id >= 0 && id < (int)I.l3c.size() && !I.l3c[(size_t)id].dead && I.l3c[(size_t)id].born_call == I.call_no_)
            nb += I.l3c[(size_t)id].charged;
    I.born_ids_.clear();
    I.reserve_ = nb;
    I.last_rows_ = rows;
    ++I.call_no_;
}
bool RecordFidelity::has_byte_budget() const { return impl_->max_bytes != 0; }
size_t RecordFidelity::content_bytes_estimate() const {
    const Impl& I = *impl_;
    const size_t live = I.l3c.size() - I.dead_count_;   // (the ledger: Σ live charged = held_l3 − held_shared_)
    const size_t charged = I.held_l3 - std::min(I.held_l3, I.held_shared_);
    const size_t e = live ? charged / live : 4096;
    return std::min<size_t>(64 << 10, std::max<size_t>(1 << 10, e));
}
std::array<size_t, 3> RecordFidelity::audit_memo_for_test() const {
    // speed fix 4: {Σ live contents' memo bytes (their key lists), held_memo, memo entries whose slot is dead or that no
    // live content lists} — equal / equal / 0 by construction (every memo entry is its content's, erased with it)
    const Impl& I = *impl_;
    size_t live_b = 0, stray = 0;
    std::unordered_set<const std::string*> listed;
    for (const Impl::L3Content& C : I.l3c)
        if (!C.dead && C.memo_ep == I.memo_epoch_) {
            live_b += C.memo_b;
            C.memo_keys.each([&](const std::string* k) { listed.insert(k); });
        }
    for (const auto& kv : I.memo) {
        int id = -1;
        std::memcpy(&id, kv.first.data(), sizeof(int));
        if (id < 0 || (size_t)id >= I.l3c.size() || I.l3c[(size_t)id].dead || !listed.count(&kv.first)) ++stray;
    }
    return {live_b, I.held_memo, stray};
}
std::pair<size_t, size_t> RecordFidelity::audit_bytes_for_test() const {
    const Impl& I = *impl_;
    size_t c = 0;
    for (const Impl::L3Content& C : I.l3c) if (!C.dead) c += C.charged;
    return {c + I.held_shared_, I.held_l3};
}
bool RecordFidelity::live(int id) const { return id >= 0 && id < (int)impl_->l3c.size() && !impl_->l3c[(size_t)id].dead; }
long RecordFidelity::drop_content(int id) {
    if (!live(id)) return 0;
    impl_->tombstone_(id);
    ++impl_->st.midcall_evictions;
    return 1;
}
long RecordFidelity::drop_unreferenced(const std::vector<uint8_t>& keep) {
    Impl& I = *impl_;
    I.clear_memo();
    long n = 0;
    for (size_t i = 0; i < I.l3c.size(); ++i)
        if (!I.l3c[i].dead && !(i < keep.size() && keep[i])) { I.tombstone_((int)i); ++n; }
    I.st.midcall_evictions += n;
    return n;
}
long RecordFidelity::drop_to_rest() {
    Impl& I = *impl_;
    long n = 0;
    for (const Impl::L3Content& C : I.l3c) n += !C.dead;
    I.clear();
    I.st.midcall_evictions += n;
    ++I.st.cache_flushes;
    return n;
}
void RecordFidelity::note_serial(int requested, const std::string& why) {
    impl_->threads_req_ = std::max(1, requested);
    impl_->threads_eff_ = 1;
    impl_->threads_limit_ = why;
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
void RecordFidelity::set_class_route_for_test(int mode) { impl_->class_lever = mode; }
void RecordFidelity::set_row_log_for_test(bool on) { impl_->row_log_on = on; }
void RecordFidelity::set_fault_for_test(int point) { impl_->fault_test_ = point; }
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
std::vector<RecordFidelity::ContentStructure> RecordFidelity::content_structures() const {
    std::vector<ContentStructure> out;
    const Impl& I = *impl_;
    for (size_t i = 0; i < I.l3c.size(); ++i) {
        ContentStructure c;
        if (I.l3c[i].dead) continue;         // §5: a tombstone
        c.id = (int)i;
        c.dC = I.dC;
        c.class_route = I.l3c[i].cls_route;
        if (I.l3c[i].nc) {
            const Impl::NewC& X = *I.l3c[i].nc;
            c.nM = X.nM; c.nG = X.nG; c.kprime = X.kprime; c.ksec = X.ksec; c.full = X.full;
            c.fib_d = X.K0.d; c.fib_kap = X.K0.kap;
        }
        out.push_back(c);
    }
    return out;
}
int RecordFidelity::n() const { return impl_->n; }
}  // namespace qeccore
