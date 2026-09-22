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

// Extended column: [ r column bits ] ++ [ for a=r-1..1, for b=0..a-1: col_a AND col_b ]  (C(r,2) bits).
Bits extend_column(uint32_t col, int r, int ext_words) {
    Bits e(ext_words, 0);
    for (int q = 0; q < r; ++q) if ((col >> q) & 1u) bxorbit(e, q);
    int l = 0;
    for (int a = r - 1; a >= 1; --a)
        for (int b = 0; b < a; ++b) {
            if (((col >> a) & 1u) && ((col >> b) & 1u)) bxorbit(e, r + l);
            ++l;
        }
    return e;
}

// proper(): drop zero columns and cancel duplicate pairs (a parity of even multiplicity is Clifford).
std::vector<uint32_t> proper(std::vector<uint32_t> table) {
    std::unordered_map<uint32_t, int> map;
    std::vector<char> keep(table.size(), 1);
    for (int i = 0; i < (int)table.size(); ++i) {
        uint32_t col = table[i];
        if (col == 0) { keep[i] = 0; }
        else if (map.count(col)) { keep[map[col]] = 0; keep[i] = 0; map.erase(col); }
        else map[col] = i;
    }
    std::vector<uint32_t> out;
    for (int i = 0; i < (int)table.size(); ++i) if (keep[i]) out.push_back(table[i]);
    return out;
}

// Indices to remove so the multiset becomes proper (paired duplicates + zeros), WITHOUT removing yet
// (the caller fixes up its parallel matrices first). Mirrors to_remove() in the reference.
std::vector<int> to_remove(const std::vector<uint32_t>& table) {
    std::unordered_map<uint32_t, int> map;
    std::vector<int> rm;
    for (int i = 0; i < (int)table.size(); ++i) {
        uint32_t col = table[i];
        if (col == 0) rm.push_back(i);
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
std::vector<uint32_t> tohpe(std::vector<uint32_t> table, int r, int ext_words) {
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
        std::unordered_map<uint32_t, int> score;
        bool parity = (bpopcount(y) & 1) == 1;
        for (int i = 0; i < (int)table.size(); ++i) {
            if (parity && !bget(y, i)) score[table[i]] = 1;
            else if (!parity && bget(y, i)) score[table[i]] = 1;
        }
        for (int i = 0; i < (int)table.size(); ++i) {
            if (!bget(y, i)) continue;
            for (int j = 0; j < (int)table.size(); ++j) {
                if (bget(y, j)) continue;
                uint32_t z = table[i] ^ table[j];
                score[z] += 2;
            }
        }
        int max_cost = 0; bool have = false; uint32_t best = 0;
        for (auto& kv : score)
            if (kv.second > max_cost || (kv.second == max_cost && have && kv.first < best)) {
                max_cost = kv.second; best = kv.first; have = true;
            }
        if (max_cost <= 0) break;
        uint32_t z = best;

        std::vector<char> to_update(table.size(), 0);
        for (int i = 0; i < (int)table.size(); ++i) to_update[i] = bget(y, i) ? 1 : 0;
        if (parity) {                                  // odd parity: append a fresh z slot
            table.push_back(0);
            matrix.push_back(Bits(ext_words, 0));
            Bits bv(aug_words((int)table.size()), 0); bxorbit(bv, (int)augmented.size());
            // grow existing augmented rows to the new width
            int nw = aug_words((int)table.size());
            for (auto& a : augmented) if ((int)a.size() < nw) a.resize(nw, 0);
            augmented.push_back(bv);
            to_update.push_back(1);
        }
        for (int i = 0; i < (int)table.size(); ++i) if (to_update[i]) table[i] ^= z;

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

}  // namespace

std::vector<uint32_t> fast_todd_reduce(std::vector<uint32_t> table, int nb_qubits) {
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

        std::unordered_map<uint32_t, int> colmap;
        for (int i = 0; i < (int)table.size(); ++i) colmap[table[i]] = i;

        int max_score = 0; bool have = false; uint32_t max_z = 0; Bits max_y;
        for (int i = 0; i < (int)table.size(); ++i)
            for (int j = i + 1; j < (int)table.size(); ++j) {
                uint32_t z = table[i] ^ table[j];
                // Build the residue matrix r_mat (one row per qubit k + one extra) of the quadratic
                // structure of z, reduced against the existing pivots.
                std::vector<Bits> r_mat, aug_r;
                auto pair_index = [&](int a, int b) {   // l for pair (a>b), a from r-1..1, b 0..a-1
                    int l = 0;
                    for (int aa = r - 1; aa >= 1; --aa) for (int bb = 0; bb < aa; ++bb) {
                        if (aa == a && bb == b) return l; ++l;
                    }
                    return -1;
                };
                for (int k = 0; k < r; ++k) {
                    Bits col(ext_words, 0), acol(aug_words((int)table.size()), 0);
                    // Off-diagonal contributions to row k: pair (a,b), a=r-1..1, b=0..a-1 (matches
                    // the reference's (a in rev, b in 0..a) loop).
                    int l = 0;
                    for (int a = r - 1; a >= 1; --a) for (int bb = 0; bb < a; ++bb) {
                        bool zb = (z >> bb) & 1u, za = (z >> a) & 1u;
                        if ((a == k && zb) || (bb == k && za)) {
                            bxorbit(col, r + l);
                            if (piv_inv.count(r + l)) { int v = piv_inv[r + l]; bxor(col, matrix[v]); bxor(acol, augmented[v]); }
                        }
                        ++l;
                    }
                    r_mat.push_back(std::move(col));
                    aug_r.push_back(std::move(acol));
                }
                // the extra (diagonal) row
                {
                    Bits col(ext_words, 0), acol(aug_words((int)table.size()), 0);
                    int l = 0;
                    for (int a = r - 1; a >= 1; --a) {
                        for (int bb = 0; bb < a; ++bb) {
                            if (((z >> a) & 1u) && ((z >> bb) & 1u)) {
                                bxorbit(col, r + l);
                                if (piv_inv.count(r + l)) { int v = piv_inv[r + l]; bxor(col, matrix[v]); bxor(acol, augmented[v]); }
                            }
                            ++l;
                        }
                        if ((z >> a) & 1u) {
                            bxorbit(col, a);
                            if (piv_inv.count(a)) { int v = piv_inv[a]; bxor(col, matrix[v]); bxor(acol, augmented[v]); }
                        }
                    }
                    // a==0 diagonal term (the reference's loop includes a from rev down to 0 for the
                    // single-bit term; pair loop above stops at a>=1, so handle z bit 0 here)
                    if ((z >> 0) & 1u) {
                        bxorbit(col, 0);
                        if (piv_inv.count(0)) { int v = piv_inv[0]; bxor(col, matrix[v]); bxor(acol, augmented[v]); }
                    }
                    r_mat.push_back(std::move(col));
                    aug_r.push_back(std::move(acol));
                }
                (void)pair_index;
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
                                uint32_t t = table[l] ^ z;
                                auto it = colmap.find(t);
                                if (it != colmap.end() && !bget(y, it->second)) sc += 2;
                            }
                        }
                        if (bpopcount(y) & 1) { sc += colmap.count(z) ? 1 : -1; }
                        if (sc > max_score) { max_score = sc; max_z = z; max_y = y; have = true; }
                    }
                }
            }
        if (max_score == 0 || !have) break;
        for (int l = 0; l < (int)table.size(); ++l) if (bget(max_y, l)) table[l] ^= max_z;
        if (bpopcount(max_y) & 1) table.push_back(max_z);
        table = proper(std::move(table));
    }
    return proper(std::move(table));
}

}  // namespace qeccore
