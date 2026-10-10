// FastTODD (Vandaele, arXiv:2407.08695) — C++ port of src/t_opt.rs (proper/kernel/tohpe/fast_todd).
// Columns are nb_qubits-bit parities held as uint32 (nb_qubits <= 32); the quadratic-extension
// "matrix" and the "augmented" combination matrix use wide vector<uint64_t> bit-rows.  The port
// mirrors the reference logic; we keep only the reduced column DIRECTIONS (the via-TODD conditioning
// needs the span, not the mod-8 Clifford correction).
#include "qeccore/fast_todd.hpp"
#include <cstddef>

#include "qeccore/compiler_builtins.hpp"   // __builtin_ctzll / __builtin_popcountll shim on MSVC

#include <algorithm>
#include <unordered_map>

namespace qeccore {
namespace {

using Bits = std::vector<uint64_t>;

inline int  words_for(int nbits)        { return nbits / 64 + 1; }
inline bool bget(const Bits& b, int i)  { return (b[(size_t)(i >> 6)] >> (i & 63)) & 1ull; }
inline void bxorbit(Bits& b, int i)     { b[(size_t)(i >> 6)] ^= 1ull << (i & 63); }
inline void bxor(Bits& a, const Bits& o) { size_t k = std::min(a.size(), o.size());
                                           for (size_t i = 0; i < k; ++i) a[i] ^= o[i]; }
inline int  bfirst(const Bits& b) {  // first set bit, or 0 if none (matches Rust get_first_one)
    for (size_t i = 0; i < b.size(); ++i) if (b[i]) return (int)i * 64 + __builtin_ctzll(b[i]);
    return 0;
}
inline int bpopcount(const Bits& b) { int s = 0; for (uint64_t x : b) s += __builtin_popcountll(x); return s; }

// Column-type operations.  The algorithm below is written ONCE over a column type `Col`:
//   * uint32_t — the historical representation (r <= 30, the only domain before 3.1.8); every
//     operation is the identical builtin, so this instantiation is the old code verbatim;
//   * WideCol  — an r-bit word vector for r > 30 (3.1.8: the r > 30 guard used to skip FastTODD).
// Equality/hash/`less` are by VALUE (WideCol::less = numeric order, most-significant word first,
// matching uint32_t's `<` on the bits) so every tie-break is the same rule in both domains.
inline bool col_bit(uint32_t c, int i)            { return (c >> i) & 1u; }
inline bool col_zero(uint32_t c)                  { return c == 0; }
inline uint32_t col_zero_like(uint32_t)           { return 0u; }
inline void col_xor_into(uint32_t& a, uint32_t b) { a ^= b; }
inline uint32_t col_xor(uint32_t a, uint32_t b)   { return a ^ b; }
inline bool col_less(uint32_t a, uint32_t b)      { return a < b; }

struct WideColHash {
    size_t operator()(const WideCol& c) const {
        uint64_t h = 0x9e3779b97f4a7c15ull;
        for (uint64_t w : c) h = (h ^ w) * 0x100000001b3ull + (h >> 29);
        return (size_t)h;
    }
};
inline bool col_bit(const WideCol& c, int i)             { return (c[(size_t)(i >> 6)] >> (i & 63)) & 1ull; }
inline bool col_zero(const WideCol& c)                   { for (uint64_t w : c) if (w) return false; return true; }
inline WideCol col_zero_like(const WideCol& c)           { return WideCol(c.size(), 0); }
inline void col_xor_into(WideCol& a, const WideCol& b)   { for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i]; }
inline WideCol col_xor(WideCol a, const WideCol& b)      { col_xor_into(a, b); return a; }
inline bool col_less(const WideCol& a, const WideCol& b) {
    for (size_t i = a.size(); i-- > 0;) if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

template <class Col> struct ColHash { using type = std::hash<Col>; };
template <> struct ColHash<WideCol> { using type = WideColHash; };
template <class Col, class V> using ColMap = std::unordered_map<Col, V, typename ColHash<Col>::type>;

// Pair index l(a,b) (a > b) of the extension's C(r,2) block, in the enumeration order
// a = r-1..1, b = 0..a-1:  l = Σ_{aa=a+1}^{r-1} aa + b.
inline int pair_l(int r, int a, int b) { return (r - 1) * r / 2 - a * (a + 1) / 2 + b; }

// Extended column: [ r column bits ] ++ [ for a=r-1..1, for b=0..a-1: col_a AND col_b ]  (C(r,2) bits).
// (Bits are SET, so the enumeration order is immaterial: only the set bits' pairs are visited.)
template <class Col>
Bits extend_column(const Col& col, int r, int ext_words) {
    Bits e(ext_words, 0);
    std::vector<int> on;
    for (int q = 0; q < r; ++q) if (col_bit(col, q)) { bxorbit(e, q); on.push_back(q); }
    for (size_t x = 1; x < on.size(); ++x)
        for (size_t y = 0; y < x; ++y) bxorbit(e, r + pair_l(r, on[x], on[y]));
    return e;
}

// proper(): drop zero columns and cancel duplicate pairs (a parity of even multiplicity is Clifford).
template <class Col>
std::vector<Col> proper(std::vector<Col> table) {
    ColMap<Col, int> map;
    std::vector<char> keep(table.size(), 1);
    for (int i = 0; i < (int)table.size(); ++i) {
        const Col& col = table[i];
        if (col_zero(col)) { keep[i] = 0; }
        else if (map.count(col)) { keep[map[col]] = 0; keep[i] = 0; map.erase(col); }
        else map[col] = i;
    }
    std::vector<Col> out;
    for (int i = 0; i < (int)table.size(); ++i) if (keep[i]) out.push_back(table[i]);
    return out;
}

// Indices to remove so the multiset becomes proper (paired duplicates + zeros), WITHOUT removing yet
// (the caller fixes up its parallel matrices first). Mirrors to_remove() in the reference.
template <class Col>
std::vector<int> to_remove(const std::vector<Col>& table) {
    ColMap<Col, int> map;
    std::vector<int> rm;
    for (int i = 0; i < (int)table.size(); ++i) {
        const Col& col = table[i];
        if (col_zero(col)) rm.push_back(i);
        else if (map.count(col)) { rm.push_back(map[col]); rm.push_back(i); map.erase(col); }
        else map[col] = i;
    }
    return rm;
}

// Gaussian-elimination kernel finder (over the extended `matrix`), tracking the combination in
// `augmented`.  Returns the combination (a copy) when a dependent column is found, else empty.
// `pivots` maps matrix-row index -> pivot bit (persisted across calls, like the reference).
Bits kernel(std::vector<Bits>& matrix, std::vector<Bits>& augmented,
            std::unordered_map<int, int>& pivots, bool& found) {
    found = false;
    for (int i = 0; i < (int)matrix.size(); ++i) {
        if (pivots.count(i)) continue;
        for (auto& kv : pivots) {
            if (bget(matrix[i], kv.second)) { bxor(matrix[i], matrix[kv.first]); bxor(augmented[i], augmented[kv.first]); }
        }
        int index = bfirst(matrix[i]);
        if (bget(matrix[i], index)) {
            for (auto& kv : pivots) {
                if (bget(matrix[kv.first], index)) { bxor(matrix[kv.first], matrix[i]); bxor(augmented[kv.first], augmented[i]); }
            }
            pivots[i] = index;
        } else {
            found = true;
            return augmented[i];
        }
    }
    return Bits();
}

// TOHPE: iteratively eliminate via kernel relations.
template <class Col>
std::vector<Col> tohpe(std::vector<Col> table, int r, int ext_words) {
    auto aug_words = [](int ncols) { return words_for(ncols); };

    auto rebuild_aug = [&](int ncols) {
        std::vector<Bits> a(ncols, Bits(aug_words(ncols), 0));
        for (int i = 0; i < ncols; ++i) bxorbit(a[i], i);
        return a;
    };

    // clear_column(i): remove column i from the pivot/elimination structure (mirrors the reference).
    auto clear_column = [&](int i, std::vector<Bits>& matrix, std::vector<Bits>& augmented,
                            std::unordered_map<int, int>& pivots) {
        if (!pivots.count(i)) return;
        int val = pivots[i]; pivots.erase(i);
        if (!bget(augmented[i], i)) {
            for (int j = 0; j < (int)matrix.size(); ++j) {
                if (!bget(augmented[j], i)) continue;
                pivots[j] = val;
                std::swap(matrix[j], matrix[i]);
                std::swap(augmented[j], augmented[i]);
                break;
            }
        }
        Bits col = matrix[i], acol = augmented[i];
        for (int j = 0; j < (int)matrix.size(); ++j)
            if (j != i && bget(augmented[j], i)) { bxor(matrix[j], col); bxor(augmented[j], acol); }
    };

    std::vector<Bits> matrix(table.size());
    for (int i = 0; i < (int)table.size(); ++i) matrix[i] = extend_column(table[i], r, ext_words);
    std::unordered_map<int, int> pivots;
    std::vector<Bits> augmented = rebuild_aug((int)table.size());

    for (;;) {
        bool found = false;
        Bits y = kernel(matrix, augmented, pivots, found);
        if (!found) break;

        // Candidate z's: cancel as many existing columns as possible.
        ColMap<Col, int> score;
        bool parity = (bpopcount(y) & 1) == 1;
        for (int i = 0; i < (int)table.size(); ++i) {
            if (parity && !bget(y, i)) score[table[i]] = 1;
            else if (!parity && bget(y, i)) score[table[i]] = 1;
        }
        for (int i = 0; i < (int)table.size(); ++i) {
            if (!bget(y, i)) continue;
            for (int j = 0; j < (int)table.size(); ++j) {
                if (bget(y, j)) continue;
                score[col_xor(table[i], table[j])] += 2;
            }
        }
        int max_cost = 0; bool have = false; Col best = col_zero_like(table[0]);
        for (auto& kv : score)
            if (kv.second > max_cost || (kv.second == max_cost && have && col_less(kv.first, best))) {
                max_cost = kv.second; best = kv.first; have = true;
            }
        if (max_cost <= 0) break;
        const Col z = best;

        std::vector<char> to_update(table.size(), 0);
        for (int i = 0; i < (int)table.size(); ++i) to_update[i] = bget(y, i) ? 1 : 0;
        if (parity) {                                  // odd parity: append a fresh z slot
            table.push_back(col_zero_like(z));
            matrix.push_back(Bits(ext_words, 0));
            Bits bv(aug_words((int)table.size()), 0); bxorbit(bv, (int)augmented.size());
            // grow existing augmented rows to the new width
            int nw = aug_words((int)table.size());
            for (auto& a : augmented) if ((int)a.size() < nw) a.resize(nw, 0);
            augmented.push_back(bv);
            to_update.push_back(1);
        }
        for (int i = 0; i < (int)table.size(); ++i) if (to_update[i]) col_xor_into(table[i], z);

        std::vector<int> rm = to_remove(table);
        std::sort(rm.rbegin(), rm.rend());
        for (int i : rm) {
            clear_column(i, matrix, augmented, pivots);
            int last = (int)table.size() - 1;
            std::swap(table[i], table[last]);       table.pop_back();
            std::swap(matrix[i], matrix[last]);     matrix.pop_back();
            std::swap(augmented[i], augmented[last]); augmented.pop_back();
            std::swap(to_update[i], to_update[last]); to_update.pop_back();
            if (pivots.count(last)) { int tmp = pivots[last]; pivots.erase(last); pivots[i] = tmp; }
            for (int j = 0; j < (int)augmented.size(); ++j) {
                if (bget(augmented[j], i) != bget(augmented[j], last)) bxorbit(augmented[j], i);
                if (bget(augmented[j], last)) bxorbit(augmented[j], last);
            }
        }
        // trim augmented rows to the current column count's word width
        int sz = aug_words((int)table.size());
        for (auto& a : augmented) while ((int)a.size() > sz) a.pop_back();

        std::vector<int> upd;
        for (int i = 0; i < (int)table.size(); ++i) if (to_update[i]) upd.push_back(i);
        for (int i : upd) {
            clear_column(i, matrix, augmented, pivots);
            matrix[i] = extend_column(table[i], r, ext_words);
            Bits bv(aug_words((int)table.size()), 0); bxorbit(bv, i);
            augmented[i] = bv;
        }
    }
    return table;
}

// The FastTODD outer loop, generic over the column type (see col_* above).
template <class Col>
std::vector<Col> fast_todd_reduce_impl(std::vector<Col> table, int nb_qubits) {
    const int r = nb_qubits;
    const int ext_bits = r + (r * (r - 1)) / 2;
    const int ext_words = words_for(ext_bits);
    auto aug_words = [](int ncols) { return words_for(ncols); };

    table = proper(std::move(table));
    for (;;) {
        table = tohpe(std::move(table), r, ext_words);
        if (table.empty()) break;

        // Build extended matrix + augmented (identity), reduce to RREF, invert the pivot map.
        std::vector<Bits> matrix(table.size());
        for (int i = 0; i < (int)table.size(); ++i) matrix[i] = extend_column(table[i], r, ext_words);
        std::vector<Bits> augmented(table.size(), Bits(aug_words((int)table.size()), 0));
        for (int i = 0; i < (int)table.size(); ++i) bxorbit(augmented[i], i);
        std::unordered_map<int, int> pivots;
        { bool f; kernel(matrix, augmented, pivots, f); }
        std::unordered_map<int, int> piv_inv;   // pivot bit -> row index
        for (auto& kv : pivots) piv_inv[kv.second] = kv.first;

        ColMap<Col, int> colmap;
        for (int i = 0; i < (int)table.size(); ++i) colmap[table[i]] = i;

        // Fold extension bit `e` into (col, acol): the bit itself plus, if `e` is a pivot, the
        // pivot row's reduction.  Pure XOR accumulation — order-independent, so visiting only the
        // contributing (set) pairs gives the same bits as the reference's full (a,b) scan.
        auto fold = [&](Bits& col, Bits& acol, int e) {
            bxorbit(col, e);
            auto it = piv_inv.find(e);
            if (it != piv_inv.end()) { bxor(col, matrix[it->second]); bxor(acol, augmented[it->second]); }
        };
        std::vector<int> zon;
        int max_score = 0; bool have = false; Col max_z = col_zero_like(table[0]); Bits max_y;
        for (int i = 0; i < (int)table.size(); ++i)
            for (int j = i + 1; j < (int)table.size(); ++j) {
                const Col z = col_xor(table[i], table[j]);
                zon.clear();
                for (int q = 0; q < r; ++q) if (col_bit(z, q)) zon.push_back(q);
                // Build the residue matrix r_mat (one row per qubit k + one extra) of the quadratic
                // structure of z, reduced against the existing pivots.
                std::vector<Bits> r_mat, aug_r;
                for (int k = 0; k < r; ++k) {
                    Bits col(ext_words, 0), acol(aug_words((int)table.size()), 0);
                    // Off-diagonal contributions to row k: pair (a,b) with (a == k && z_b) or
                    // (b == k && z_a), i.e. l(k, b) for set b < k and l(a, k) for set a > k.
                    for (int q : zon) {
                        if (q < k) fold(col, acol, r + pair_l(r, k, q));
                        else if (q > k) fold(col, acol, r + pair_l(r, q, k));
                    }
                    r_mat.push_back(std::move(col));
                    aug_r.push_back(std::move(acol));
                }
                // the extra (diagonal) row: every set pair (a > b) of z, plus every set bit of z.
                {
                    Bits col(ext_words, 0), acol(aug_words((int)table.size()), 0);
                    for (size_t x = 1; x < zon.size(); ++x)
                        for (size_t y = 0; y < x; ++y) fold(col, acol, r + pair_l(r, zon[x], zon[y]));
                    for (int q : zon) fold(col, acol, q);
                    r_mat.push_back(std::move(col));
                    aug_r.push_back(std::move(acol));
                }
                // Reduce r_mat; for each dependent row whose augmented touches exactly one of i,j,
                // score the resulting y.
                for (int k = 0; k < (int)r_mat.size(); ++k) {
                    int index = bfirst(r_mat[k]);
                    if (bget(r_mat[k], index)) {
                        for (int l = k + 1; l < (int)r_mat.size(); ++l)
                            if (bget(r_mat[l], index)) { bxor(r_mat[l], r_mat[k]); bxor(aug_r[l], aug_r[k]); }
                    } else if (bget(aug_r[k], i) ^ bget(aug_r[k], j)) {
                        int sc = 0;
                        Bits y = aug_r[k];
                        for (int l = 0; l < (int)table.size(); ++l) {
                            if (bget(y, l)) {
                                auto it = colmap.find(col_xor(table[l], z));
                                if (it != colmap.end() && !bget(y, it->second)) sc += 2;
                            }
                        }
                        if (bpopcount(y) & 1) { sc += colmap.count(z) ? 1 : -1; }
                        if (sc > max_score) { max_score = sc; max_z = z; max_y = y; have = true; }
                    }
                }
            }
        if (max_score == 0 || !have) break;
        for (int l = 0; l < (int)table.size(); ++l) if (bget(max_y, l)) col_xor_into(table[l], max_z);
        if (bpopcount(max_y) & 1) table.push_back(max_z);
        table = proper(std::move(table));
    }
    return proper(std::move(table));
}

}  // namespace

std::vector<uint32_t> fast_todd_reduce(std::vector<uint32_t> table, int nb_qubits) {
    return fast_todd_reduce_impl<uint32_t>(std::move(table), nb_qubits);
}

std::vector<WideCol> fast_todd_reduce_wide(std::vector<WideCol> table, int nb_qubits) {
    return fast_todd_reduce_impl<WideCol>(std::move(table), nb_qubits);
}

}  // namespace qeccore
