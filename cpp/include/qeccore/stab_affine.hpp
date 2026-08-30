#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include "qeccore/stab_state.hpp"

namespace qeccore {

// Word-level in-place bit compaction: remove bit l from a packed row (valid bits
// [0, nbits) with everything at/above nbits invariant-zero), shifting bits (l, ...)
// down by one. Result is bit-identical to the per-bit shift loop it replaces —
// pure data movement, no semantic content.
inline void compact_bit_out(uint64_t* r, int words, int l) {
    const int lw = l >> 6, lb = l & 63;
    const uint64_t low = lb ? (r[lw] & ((1ULL << lb) - 1)) : 0;
    const uint64_t mid = (lb == 63) ? 0 : ((r[lw] >> (lb + 1)) << lb);
    const uint64_t carry = (lw + 1 < words) ? (r[lw + 1] & 1ULL) : 0;
    r[lw] = low | mid | (carry << 63);
    for (int w = lw + 1; w < words; ++w) {
        const uint64_t nxt = (w + 1 < words) ? (r[w + 1] & 1ULL) : 0;
        r[w] = (r[w] >> 1) | (nxt << 63);
    }
}

// Row-major bit-packed n×k matrix with dynamic column count.
// Row i occupies words [i*words_, (i+1)*words_); bit j is word (j>>6) bit (j&63),
// LSB-first. Trailing bits beyond k_ are kept zero. Capacity (words_) grows in
// 64-column blocks; append/drop column shift bits within each row.
struct PackedMat {
    int rows_ = 0;          // n
    int cols_ = 0;          // current k
    int words_ = 0;         // uint64 words per row (capacity = words_*64 columns)
    std::vector<uint64_t> data_;

    PackedMat() = default;
    explicit PackedMat(int rows) : rows_(rows), cols_(0), words_(0) {}

    int cols() const { return cols_; }
    int words() const { return words_; }
    uint64_t* row(int i) { return data_.data() + (size_t)i * words_; }
    const uint64_t* row(int i) const { return data_.data() + (size_t)i * words_; }

    bool get(int i, int j) const {
        return (row(i)[j >> 6] >> (j & 63)) & 1ULL;
    }
    void set(int i, int j, uint8_t v) {
        uint64_t* r = row(i);
        uint64_t m = 1ULL << (j & 63);
        if (v) r[j >> 6] |= m; else r[j >> 6] &= ~m;
    }
    void xor_bit(int i, int j, uint8_t v) {
        if (v) row(i)[j >> 6] ^= (1ULL << (j & 63));
    }

    void reserve_words(int w) {
        if (w <= words_) return;
        std::vector<uint64_t> nd((size_t)rows_ * w, 0);
        for (int i = 0; i < rows_; ++i)
            for (int t = 0; t < words_; ++t) nd[(size_t)i * w + t] = data_[(size_t)i * words_ + t];
        data_.swap(nd);
        words_ = w;
    }

    // Append a new column (index = old cols_) with given per-row bits (length rows_).
    void append_col(const std::vector<uint8_t>& colbits) {
        int j = cols_;
        if (((cols_ + 64) + 63) / 64 > words_) reserve_words((cols_ + 64) / 64 + 1);
        if ((j >> 6) >= words_) reserve_words((j >> 6) + 1);
        for (int i = 0; i < rows_; ++i) set(i, j, colbits[i]);
        cols_ += 1;
    }

    // Drop column l: shift columns >l down by one, decrement cols_.
    void drop_col(int l) {
        // bits [0,l) stay; bits (l,cols_) shift down by one. Word-level compaction
        // (compact_bit_out) — bit-identical to the per-bit shift, O(words) per row.
        // The vacated top position (cols_-1) receives old bit cols_, invariant-zero.
        for (int i = 0; i < rows_; ++i)
            compact_bit_out(row(i), words_, l);
        cols_ -= 1;
    }

    // Row op: row dst ^= row src (full word-wide XOR over all k columns).
    void xor_row(int dst, int src) {
        uint64_t* d = row(dst);
        const uint64_t* s = row(src);
        for (int t = 0; t < words_; ++t) d[t] ^= s[t];
    }
};

// Dynamic k×k symmetric GF(2) matrix with zero diagonal, bit-packed row-major.
// Full symmetric storage (BOTH (i,j) and (j,i) bits kept) so reads are simple and
// row XORs are word-wide. Row i occupies words [i*words_, (i+1)*words_); bit j is
// word (j>>6) bit (j&63), LSB-first. Grows/shrinks by one row+col (append_zero/drop)
// to track the affine quadratic-form J as H/drop_col change k.
struct SymPackedMat {
    int k_ = 0;             // dimension (rows == cols)
    int words_ = 0;         // uint64 words per row (capacity = words_*64)
    std::vector<uint64_t> data_;

    int dim() const { return k_; }
    int words() const { return words_; }
    uint64_t* row(int i) { return data_.data() + (size_t)i * words_; }
    const uint64_t* row(int i) const { return data_.data() + (size_t)i * words_; }

    // LAZY-ZERO invariant: words_ == 0 with k_ > 0 represents the k×k all-zero matrix
    // with NO row storage. Every word-bounded scan (`for t < words()`) is already a
    // correct no-op on it; get() reads zero; the first flip_sym() materializes rows.
    // Rationale: a DiagPauliClifford's CZ layer B is nonzero only for residuals that
    // cross a level-3 gate, yet it was allocated dense (n²) per identity()/then() —
    // the dominant compile cost on large Clifford circuits. sym_zeros() builds this
    // form; append_zero/reserve_words paths (AffineState J/R) are unaffected (they
    // always materialize as before).
    bool zero_unmaterialized() const { return words_ == 0; }
    void materialize_zero() {
        words_ = (k_ + 63) / 64;
        data_.assign((size_t)k_ * words_, 0);
    }

    bool get(int i, int j) const {
        if (words_ == 0) return false;      // lazy all-zero
        return (row(i)[j >> 6] >> (j & 63)) & 1ULL;
    }
    // Toggle the symmetric pair (i,j) and (j,i); no-op on the diagonal (i==j).
    void flip_sym(int i, int j) {
        if (i == j) return;
        if (words_ == 0) materialize_zero();   // first write to a lazy all-zero
        row(i)[j >> 6] ^= (1ULL << (j & 63));
        row(j)[i >> 6] ^= (1ULL << (i & 63));
    }
    // row dst ^= row src (word-wide over all k columns).
    void xor_row_into(int dst, int src) {
        uint64_t* d = row(dst);
        const uint64_t* s = row(src);
        for (int t = 0; t < words_; ++t) d[t] ^= s[t];
    }

    void reserve_words(int w) {
        if (w <= words_) return;
        std::vector<uint64_t> nd((size_t)k_ * w, 0);
        for (int i = 0; i < k_; ++i)
            for (int t = 0; t < words_; ++t) nd[(size_t)i * w + t] = data_[(size_t)i * words_ + t];
        data_.swap(nd);
        words_ = w;
    }

    // Grow to (k_+1)×(k_+1) with the new row/col all-zero (diagonal stays 0).
    void append_zero() {
        int nk = k_ + 1;
        int need_words = (nk + 63) / 64;
        if (need_words > words_) {
            // Reallocate to new word width AND add a row, copying existing rows.
            std::vector<uint64_t> nd((size_t)nk * need_words, 0);
            for (int i = 0; i < k_; ++i)
                for (int t = 0; t < words_; ++t)
                    nd[(size_t)i * need_words + t] = data_[(size_t)i * words_ + t];
            data_.swap(nd);
            words_ = need_words;
        } else {
            // Word width unchanged: just append one zero row.
            data_.resize((size_t)nk * words_, 0);
        }
        k_ = nk;   // new row k_-1 and new column k_-1 are already zero.
    }

    // Drop row/col l: remove row l, and bit l from every surviving row (shift the
    // higher bits down by one). Mirrors PackedMat::drop_col but on a square matrix.
    // Processed in ascending row order: surviving row i writes to dst di = i or i-1
    // (di <= i), reading src = row(i) word-by-word into the compacted dst within the
    // SAME row buffer first, then placing it — since di <= i and rows are visited
    // ascending, the source row i is never an already-overwritten destination.
    void drop(int l) {
        int nk = k_ - 1;
        if (words_ == 0) { k_ = nk; return; }    // lazy all-zero stays all-zero
        for (int i = 0; i < k_; ++i) {
            if (i == l) continue;
            const uint64_t* src = row(i);
            int di = (i < l) ? i : i - 1;     // destination row index after removal
            uint64_t* dst = data_.data() + (size_t)di * words_;
            // Move the surviving row down (no-op when i < l: dst == src), then
            // compact bit l out in place — word-level (compact_bit_out), bit-identical
            // to the per-bit rebuild it replaces. Ascending order keeps src rows
            // unclobbered (di <= i, and di == i-1 was consumed the previous step).
            if (dst != src) for (int t = 0; t < words_; ++t) dst[t] = src[t];
            compact_bit_out(dst, words_, l);
        }
        data_.resize((size_t)nk * words_, 0);
        k_ = nk;
    }
};

// |ψ⟩ = ω · 2^(-k/2) Σ_y i^{Q(y)} |b ⊕ R y⟩, y ∈ 𝔽₂^k.
struct AffineState : StabState {
    int n_ = 0;
    int k_ = 0;
    std::vector<uint8_t> b;              // length n
    PackedMat R;                         // n rows × k cols (row-major bit-packed):
                                         // generator j is column R(*,j); R.xor_row is
                                         // word-wide, column ops use get/set/xor_bit.
    std::vector<int>     D;              // length k, linear part of Q, entries mod 4
    SymPackedMat         J;              // k×k symmetric, zero diagonal, quadratic part of Q
                                         // (bit-packed; full symmetric storage, both halves)
    ExactPhase omega = ExactPhase::one();

    explicit AffineState(int n) : n_(n), b(n, 0), R(n) {}  // R: n empty rows (k=0)

    int n() const override { return n_; }
    ExactPhase phase() const override { return omega; }
    std::unique_ptr<StabState> clone() const override {
        return std::make_unique<AffineState>(*this);
    }

    void apply_h(int q) override;
    void apply_s(int q) override;
    void apply_sdg(int q) override;
    void apply_x(int q) override;
    void apply_y(int q) override;
    void apply_z(int q) override;
    void apply_cx(int c, int t) override;
    void apply_cz(int c, int t) override;
    ExactPhase inner_product(const StabState& other) const override;
    // Exact amplitude ⟨bits|ψ⟩ at one computational-basis point (`bits[i]` = qubit i; coset Gauss
    // sum over y with b⊕Ry = bits). Single-state read — does NOT mutate and does NOT touch any
    // other state (used by the closed-form measurement collapse to fix per-branch gauge phases
    // without pairwise overlaps). Works for n > 64 (no uint64_t index ceiling). Returns zero if
    // `bits` is off-support.
    ExactPhase amplitude_at_bits(const std::vector<uint8_t>& bits) const;
    ExactPhase pauli_project(int pauli, int q, int outcome) override;
    int single_pauli_expectation(int pauli, int q) const override;
    std::vector<std::complex<double>> to_statevector() const override;

    // Exact ⟨Z_q⟩ ∈ {+1,−1,0}: +1/−1 if qubit q is Z-deterministic (R-row q empty),
    // else 0 (in superposition). O(k/64). Public wrapper for the file-static diag_z helper.
    int z_expectation(int q) const;

    void canonicalize();  // restore full column rank of R after H
};

}  // namespace qeccore
