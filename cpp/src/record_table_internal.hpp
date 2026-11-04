#pragma once
// record_table_internal — the ONE elimination routine of the record table, shared by its two builders
// (J-a N3): (D) build_record_table (record_table.cpp: the σ-dual-basis channels, then add_extra per
// extra) and (G) build_record_table_general (record_general.cpp: the null form, add_extra over EVERY
// column). Moved here verbatim from record_table.cpp (S2b / J0); everything stays in an anonymous
// namespace, so each including TU keeps its own internal-linkage copy (record_table.cpp's code
// generation is unchanged) and every XTIM_MUTATE_* hook below reaches both builders.
// Internal: include only from cpp/src/record_table.cpp and cpp/src/record_general.cpp.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qeccore/record_table.hpp"

namespace qeccore {

namespace {

using Bits = std::vector<uint64_t>;

inline bool bit_get(const Bits& v, int i) {
    return (size_t)(i >> 6) < v.size() && ((v[(size_t)(i >> 6)] >> (i & 63)) & 1ULL);
}
inline void bit_flip(Bits& v, int i) {
    if ((size_t)(i >> 6) >= v.size()) v.resize((size_t)(i >> 6) + 1, 0);
    v[(size_t)(i >> 6)] ^= 1ULL << (i & 63);
}
inline void bits_xor(Bits& dst, const Bits& src) {
    if (dst.size() < src.size()) dst.resize(src.size(), 0);
    for (size_t w = 0; w < src.size(); ++w) dst[w] ^= src[w];
}
inline int bits_lead(const Bits& v) {
    for (size_t w = 0; w < v.size(); ++w)
        if (v[w]) return (int)(w * 64) + __builtin_ctzll(v[w]);
    return -1;
}
inline bool bits_any(const Bits& v) {
    for (uint64_t w : v) if (w) return true;
    return false;
}
inline int and_parity(const Bits& a, const Bits& b) {
    const size_t n = std::min(a.size(), b.size());
    int p = 0;
    for (size_t w = 0; w < n; ++w) p += __builtin_popcountll(a[w] & b[w]);
    return p & 1;
}
// The free-register coordinates of a FreeDir as one bitvector: a in [0, k), b in [k, 2k).
Bits fd_coords(const FreeDir& d) {
    const int k = (int)d.a.size();
    Bits v((size_t)((2 * k + 63) / 64), 0);
    for (int i = 0; i < k; ++i) {
        if (d.a[(size_t)i]) bit_flip(v, i);
        if (d.b[(size_t)i]) bit_flip(v, k + i);
    }
    return v;
}

// Word operations on the two row types (one implementation of the algorithms below over either):
// uint64_t (<= 64 labels) and Bits (any width).
inline bool wbit(const Bits& v, int i) { return bit_get(v, i); }
inline bool wbit(uint64_t v, int i) { return (v >> i) & 1ULL; }
inline void wflip(Bits& v, int i) { bit_flip(v, i); }
inline void wflip(uint64_t& v, int i) { v ^= 1ULL << i; }
inline void wxor(Bits& a, const Bits& b) { bits_xor(a, b); }
inline void wxor(uint64_t& a, uint64_t b) { a ^= b; }
inline bool wany(const Bits& v) { return bits_any(v); }
inline bool wany(uint64_t v) { return v != 0; }
inline int wlead(const Bits& v) { return bits_lead(v); }
inline int wlead(uint64_t v) { return v ? __builtin_ctzll(v) : -1; }
inline void wclear(Bits& v) { v.clear(); }
inline void wclear(uint64_t& v) { v = 0; }
inline bool wequal(const Bits& a, const Bits& b) { Bits d = a; bits_xor(d, b); return !bits_any(d); }
inline bool wequal(uint64_t a, uint64_t b) { return a == b; }

// Row echelon with combination tracking (rows reduced at every lead; leads tested in insertion
// order, like the oracle's eliminations). ONE implementation over the row word type.
template <class W>
struct EchelonT {
    struct Row { W v; int lead; W comb; };
    std::vector<Row> rows;
    // Reduce v; returns the combination (over the tracked labels) and leaves the residual in v.
    W reduce(W& v) const {
        W comb{};
        for (const Row& R : rows)
            if (wbit(v, R.lead)) { wxor(v, R.v); wxor(comb, R.comb); }
        return comb;
    }
    // Insert a NONZERO residual v whose combination (over labels) is comb.
    void insert(const W& v, const W& comb) {
        const int ld = wlead(v);
        if (ld < 0) throw std::logic_error("record_table: echelon insert of a zero row (engine bug)");
        for (Row& R : rows)
            if (wbit(R.v, ld)) { wxor(R.v, v); wxor(R.comb, comb); }
        rows.push_back(Row{v, ld, comb});
    }
};
using Echelon = EchelonT<Bits>;


// The extras: ONE extra record column c with pulled-back operator Q, against the working table and
// the elimination state so far (J0: the S2b extras loop body, extracted verbatim; the M-1 op counters
// move with it). A pivot extra takes a fresh fair coin; a non-pivot extra gets its comb, its
// direction (new or as a product of the working directions) and its sign.
struct ExtraCtx {
    const CertifiedGroupPlanes& G;
    const FramedSuperposition& bare;
    AffineTable& W;
    Echelon pech;                            // over generator bits; labels = working coin slots
    std::vector<Pauli> Qpiv;                 // the pivot records' pullbacks, by coin slot
    Echelon dech;                            // over free coords; labels = working directions
    long ops_plane = 0, ops_pivot = 0, ops_greduce = 0, ops_ech = 0;
};

// termroute-a F2: ph_in (optional) = φ(Q) precomputed; kd_in (optional, only with φ(Q) = 0, so comb is empty and
// K = Q) = free_dir_of(bare, Q) precomputed — G.reduce's verdict there is LOGICAL / IN_GROUP by φ = 0, so it is
// skipped (its op counter still counts). (G)'s per-column memo (record_general.cpp) supplies both.
[[maybe_unused]] void add_extra(ExtraCtx& X, const Pauli& Q, int c, const std::vector<uint64_t>* ph_in = nullptr,
                                const FreeDir* kd_in = nullptr) {   // J-b2 J7b: record_table.cpp no longer calls it
    std::vector<uint64_t> ph;
    const std::vector<uint64_t>* php = ph_in;
    if (!php) {
        phi_of(X.G, Q, ph);                // ONE plane
        php = &ph;
    }
    ++X.ops_plane;
#ifdef XTIM_MUTATE_M1B_PLANE_TWICE
    phi_of(X.G, Q, ph); ++X.ops_plane;       // mutation build only (M-1b)
#endif
    Bits v(php->begin(), php->end());
    Bits comb = X.pech.reduce(v);          // ONE pivot reduction
    ++X.ops_pivot;
    if (bits_any(v)) {                   // a PIVOT: a fresh fair coin
        const int slot = X.W.ncoin++;
        if (slot != (int)X.Qpiv.size())
            throw std::logic_error("build_record_table: coin slot bookkeeping (engine bug)");
        Bits cb = comb;
        bit_flip(cb, slot);
        X.pech.insert(v, cb);
        X.Qpiv.push_back(Q);
        bit_flip(X.W.A[(size_t)c], slot);
        return;
    }
    // K = Q · Π_{s ∈ comb, ascending} Q_{pivot s}: φ(K) = 0, its outcome is y_c ⊕ Σ y_p.
    Pauli K = Q;
    for (int s = 0; s < (int)X.Qpiv.size(); ++s)
        if (bit_get(comb, s)) K = Pauli::multiply(K, X.Qpiv[(size_t)s]);
    FreeDir Kd;
    if (kd_in) {
        ++X.ops_greduce;
        Kd = *kd_in;
    } else {
        const Membership mb = X.G.reduce(K);   // ONE G.reduce
        ++X.ops_greduce;
#ifdef XTIM_MUTATE_M1A_GREDUCE_TWICE
        (void)X.G.reduce(K); ++X.ops_greduce;    // mutation build only (M-1a)
#endif
        if (mb.verdict == Membership::ANTI)
            throw std::logic_error("build_record_table: K of a non-pivot extra is ANTI (engine bug)");
        Kd = free_dir_of(X.bare, K);
    }
    fd_require_hermitian(Kd, "extra K");
    X.W.A[(size_t)c] = comb;
    Bits r2 = fd_coords(Kd);
#ifdef XTIM_MUTATE_C2_NO_ECHELON
    Bits dc;                             // mutation build only (C-2): no reduction
#else
    Bits dc = X.dech.reduce(r2);           // ONE free-register echelon
#endif
    ++X.ops_ech;
    if (bits_any(r2)) {                  // a new direction, introduced by this extra
        const int j = (int)X.W.dirs.size();
        FreeDir L = Kd;
        L.hermitian = 1;
        X.W.dirs.push_back(L);
        Bits cb = dc;
        bit_flip(cb, j);
        X.dech.insert(r2, cb);
        bit_flip(X.W.B[(size_t)c], j);
    } else {
        FreeDir prod = fd_identity(X.bare.k());
        for (int j = 0; j < (int)X.W.dirs.size(); ++j)
            if (bit_get(dc, j)) prod = fd_mul(prod, X.W.dirs[(size_t)j]);
        const int dph = (fd_eff_phase(Kd) - fd_eff_phase(prod)) & 3;
        if (dph & 1)
            throw std::logic_error("build_record_table: non-Hermitian ratio of commuting record "
                                   "operators (engine bug)");
        X.W.B[(size_t)c] = dc;
#ifdef XTIM_MUTATE_MSIGN_DROP_DPH
        X.W.beta[(size_t)c] = 0;               // mutation build only (M-SIGN): s_c dropped
#else
        X.W.beta[(size_t)c] = (uint8_t)(dph == 2 ? 1 : 0);
#endif
#ifdef XTIM_MUTATE_ML2_BETA_FLIP
        X.W.beta[(size_t)c] ^= 1;              // mutation build only (M-L2)
#endif
    }
}
}  // namespace

}  // namespace qeccore
