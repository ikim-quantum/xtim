#pragma once
// record_table — S2b (twirl consolidation, design §5.1 S2b; ndrec §1.2-1.4): the per-plan RECORD
// TABLE of a DIAGONAL plan. Every record column (IN_GROUP channels, the Born observable, Born LOGICAL /
// ANTI decisions) is an affine function of the shot's fair coins u and Born outcomes z (Theorem 1):
//     y_c = A_c·u  ⊕  B_c·z  ⊕  β_c                                   (RAW, P-free; Lemma 2 adds t_c(P))
// in the RECORD-ORDERED CANONICAL BASIS (Lemma 4): columns are scanned in global record order
// (detectors, observables, decisions — the oracle's order); a column whose coin content is independent
// of every earlier column is a PIVOT (coin k = its own value: A = e_k, B = 0, β = 0); a later column
// whose residual direction content is independent INTRODUCES direction j (B = e_j, β = 0) with
//     Λ_j := (−1)^{β'} · Π_{i ∈ b'} Λ_i      (the LITERAL ordered product of the input directions'
//                                             effective free operators, exact phase; never
//                                             re-canonicalized — design r5.4 literal-product rule)
// and every other column is a known combination of earlier pivots / directions.
//
// Conventions (normative):
//   * A direction is a compact free-register operator in EFFECTIVE form i^phase · (−1)^par · X^a Z^b
//     (FreeDir; the Born trie projects with (1 ± Λ̂)/2). Its outcome bit is 0 ⇔ eigenvalue +1.
//   * The literal product of commuting Hermitian directions is Hermitian and its eigenvalue on a joint
//     eigenvector is the product of eigenvalues, so the outcome of Π_{i∈b} Λ_i is b·z EXACTLY (no sign).
//     The A4 ordered-product sign (Theorem 1's s_c) appears only when a record's own operator is
//     compared with that product, and it lands in β (emission), never in a projector.
//   * The oracle (cpp/testapi/record_law_oracle.cpp) stores Λ_j = the free part of K_{c_j} at K's exact
//     phase. Its conversion to FreeDir is (a, b, phase = ph & 3, par = parity of K's Z on eps = 1 rows,
//     hermitian = 1). K = Q_c·Π Q_p is a product of commuting Hermitian Paulis whose certified rows carry
//     no X in frame coordinates, so (phase − |a∧b|) is even on every reachable input: the conversion
//     asserts it (engine-bug throw) and never Hermitizes (design r3.3).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// ── Compact free-register operators ───────────────────────────────────────────────────────────
inline int fd_eff_phase(const FreeDir& d) { return (d.phase + 2 * d.par) & 3; }
// (phase − |a∧b|) even ⇔ Hermitian.
bool fd_hermitian(const FreeDir& d);
// The literal product p·q of two effective operators (exact phase; par folded into phase).
FreeDir fd_mul(const FreeDir& p, const FreeDir& q);
// FCS round 2 (c): p ← p·q in place (exactly fd_mul's fields, hermitian included), no allocation once p is sized.
void fd_mul_into(FreeDir& p, const FreeDir& q);
// The identity on k coordinates.
FreeDir fd_identity(int k);
// Throws std::logic_error(what) unless d is Hermitian (the proven invariant above).
void fd_require_hermitian(const FreeDir& d, const char* what);

// ── Pullback and plane ────────────────────────────────────────────────────────────────────────
// C0† M C0 for the diagonal residual C0 = S^{a&1} · CZ of nf (identity prefix), exact phase.
Pauli diag_pullback(const DiagNormalForm& nf, const Pauli& M);
// phi(Q): bit a = <Q, g_a> over the certified generators (GW words).
void phi_of(const CertifiedGroupPlanes& G, const Pauli& Q, std::vector<uint64_t>& out);
void phi_of(const CertifiedGroupPlanes& G, const SparsePauli& Q, std::vector<uint64_t>& out);   // S2-T6: by support

// ── The record table ──────────────────────────────────────────────────────────────────────────
// A generic affine table over coin sources (uniform, independent of the directions) and directions
// (commuting Hermitian free-register operators with a known joint Born law). Column c is
// (A[c] over coins, B[c] over directions, beta[c]).
struct AffineTable {
    int m = 0;                                   // columns
    int ncoin = 0;                               // coin sources
    std::vector<FreeDir> dirs;                   // directions (effective free operators)
    std::vector<std::vector<uint64_t>> A, B;     // per column: bitmask over coins / directions
    std::vector<uint8_t> beta;                   // per column
    // r7.2: the one-word form (ncoin <= 64 and dirs <= 64): A64[c] / B64[c] replace A[c] / B[c]
    // (same bits); canonicalize_record_order then runs on words — the same algorithm, no per-column
    // vectors.
    bool small = false;
    std::vector<uint64_t> A64, B64;
};

// The canonical table of a plan (payload). Column order is the SAMPLER's storage order (channels in
// channel-slot order, then the extras); `order` gives each column's global record index.
struct RecTable {
    bool built = false;
    int m = 0, rho = 0, kappa = 0;               // columns, pivots (coins), record directions
    int CWX = 0;                                 // words per column row
    int folds = 0;                               // introducers built with the fold bit β' = 1 (review
                                                 // R6-4: the M-BETA reachability dump; fits the padding)
    std::vector<uint64_t> base;                  // β row (CWX words)
    std::vector<std::vector<uint64_t>> coin_row; // ρ rows (CWX words): coin k flips these columns
    std::vector<std::vector<uint64_t>> born_row; // κ rows (CWX words): direction j flips these columns
    std::vector<int> pivot_rec, born_rec;        // storage column of pivot k / introducer of direction j
    std::vector<FreeDir> lam;                    // κ canonical directions (literal products, exact phase)
    // add_extra operation counts of THIS table's build (mechanism test M-1).
    long ops_plane = 0, ops_pivot_reduce = 0, ops_greduce = 0, ops_echelon = 0, extras = 0;
};

// Canonicalize an affine table in record order. `order[c]` is column c's global record index (a
// permutation key; ties are not allowed). Fills `out` (canonical rows in STORAGE order, CWX words per
// row). Every canonical direction is the literal product of the input directions (with the folded β
// sign). Lemma-4 structure is asserted (engine-bug throws): A_c supported on pivots ≤ c, B_c on
// directions introduced by records ≤ c, B_{born_rec[j]} = e_j, every Λ Hermitian.
void canonicalize_record_order(const AffineTable& in, const std::vector<int>& order, RecTable& out);

// S2-T6 fix 4: a channel's generator mask — a set of certified-generator indices, read ascending — stored in the
// cheaper of two forms: a sorted list (4 bytes an entry) or a bitset trimmed after its highest set word (8 bytes a
// word; never more than the GW words the pre-S2-T6 mask held). The list is kept iff 4·|mask| <= 8·words, so a mask
// never costs more than min(list, bitset) of its own content (plus the two empty headers). Both forms read the same
// ascending indices (begin / end, size); the form is a pure function of the content, so equality is the content's.
class GenMask {
public:
    GenMask() = default;
    GenMask(std::vector<int32_t> list) {   // NOLINT: implicit, from the ascending generator list (the readers' form)
        n_ = (int32_t)list.size();
        // S2-CLEAN (fix-4 review n4): the bitset is sized from list.back(), so the list must be strictly ascending
        // and non-negative (every producer emits it so; Debug builds check it)
        assert(list.empty() || (list.front() >= 0 &&
                                std::adjacent_find(list.begin(), list.end(), std::greater_equal<int32_t>()) == list.end()));
        if (list.empty()) return;
        const size_t nw = (size_t)(list.back() >> 6) + 1;
        if (4 * list.size() <= 8 * nw) {
            if (list.capacity() > list.size()) list.shrink_to_fit();
            list_ = std::move(list);
            return;
        }
        words_.assign(nw, 0);
        for (int32_t i : list) words_[(size_t)(i >> 6)] |= 1ULL << (i & 63);
    }
    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = int32_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const int32_t*;
        using reference = int32_t;
        int32_t operator*() const { return w_ ? (int32_t)(wi_ * 64 + (size_t)__builtin_ctzll(bits_)) : *p_; }
        const_iterator& operator++() {
            if (!w_) { ++p_; return *this; }
            bits_ &= bits_ - 1;
            while (!bits_ && ++wi_ < wn_) bits_ = w_[wi_];
            return *this;
        }
        bool operator==(const const_iterator& o) const { return p_ == o.p_ && wi_ == o.wi_ && bits_ == o.bits_; }
        bool operator!=(const const_iterator& o) const { return !(*this == o); }
    private:
        friend class GenMask;
        const int32_t* p_ = nullptr;                    // list form
        const uint64_t* w_ = nullptr;                   // bitset form: words, the current word and its unread bits
        size_t wi_ = 0, wn_ = 0;
        uint64_t bits_ = 0;
    };
    const_iterator begin() const {
        const_iterator it;
        if (words_.empty()) { it.p_ = list_.data(); return it; }
        it.w_ = words_.data();
        it.wn_ = words_.size();
        it.bits_ = words_[0];
        while (!it.bits_ && ++it.wi_ < it.wn_) it.bits_ = words_[it.wi_];
        return it;
    }
    const_iterator end() const {
        const_iterator it;
        if (words_.empty()) { it.p_ = list_.data() + list_.size(); return it; }
        it.w_ = words_.data();
        it.wn_ = it.wi_ = words_.size();
        return it;
    }
    size_t size() const { return (size_t)n_; }
    bool empty() const { return n_ == 0; }
    bool is_bitset() const { return !words_.empty(); }
    size_t heap_bytes() const { return list_.capacity() * sizeof(int32_t) + words_.capacity() * sizeof(uint64_t); }
    std::vector<int32_t> list() const { return std::vector<int32_t>(begin(), end()); }
    bool operator==(const GenMask& o) const { return n_ == o.n_ && list_ == o.list_ && words_ == o.words_; }
    bool operator!=(const GenMask& o) const { return !(*this == o); }
private:
    std::vector<int32_t> list_;
    std::vector<uint64_t> words_;
    int32_t n_ = 0;
};

// One record column of the sampler's family (circuit constant). S2-T6: stored by support — M sparse (its weight is the
// number of reads), the mask by its generators (the dense forms were n'/64 and ng/64 words per column; fix 4: the
// cheaper of a list and a bitset, GenMask).
struct RecColumn {
    SparsePauli M;            // the record operator (product of terminal reads, exact phase)
    int order = 0;            // global record index (detectors, observables, decisions)
    bool channel = false;     // IN_GROUP at R = I (σ-mask channel); else an extra (Born obs / decision)
    GenMask mask;             // channel: the certified generators of the mask (σ parity), ascending
    uint8_t ref = 0;          // channel: sign of M relative to Π_{mask} g (raw = ref ⊕ mask·σ)
};

// S2-T6: a generator list's words (bit i = generator i, GW words) — the pre-S2-T6 mask form, for the digests, the
// channel-set token and the test APIs; and back.
inline std::vector<uint64_t> mask_words(const std::vector<int32_t>& mask, int GW) {
    std::vector<uint64_t> w((size_t)(GW > 0 ? GW : 0), 0);
    for (int32_t i : mask) w[(size_t)(i >> 6)] |= 1ULL << (i & 63);
    return w;
}
inline std::vector<uint64_t> mask_words(const GenMask& mask, int GW) {
    std::vector<uint64_t> w((size_t)(GW > 0 ? GW : 0), 0);
    for (int32_t i : mask) w[(size_t)(i >> 6)] |= 1ULL << (i & 63);
    return w;
}
inline std::vector<int32_t> mask_list(const std::vector<uint64_t>& words) {
    std::vector<int32_t> m;
    for (size_t w = 0; w < words.size(); ++w)
        for (uint64_t b = words[w]; b; b &= b - 1) m.push_back((int32_t)(w * 64 + (size_t)__builtin_ctzll(b)));
    return m;
}

// S2-T7: the record columns' per-wire rows — the transpose of the record operators — stored by support (CSR, column
// indexes ascending per wire). Wire q's X row lists the columns whose operator has Z on q (a prefix X on q flips them),
// its Z row the columns with X on q; t(P) = ⊕_{q ∈ x(P)} Xrow(q) ⊕ ⊕_{q ∈ z(P)} Zrow(q) over the columns. The
// channels (c < nchan) come first: a row's entries [off[q], mid[q]) are channels, [mid[q], off[q + 1]) extras. One
// entry per support bit of an operator: O(Σ |M_c| + n') storage. The pre-S2-T7 dense forms (n' × ⌈m/64⌉ words for
// all columns, n' × CW for the channels, n' × XR for the extras at their own bit positions) are this storage's words
// (`words`: the digest and the test API only).
struct WireRows {
    int n = 0, nchan = 0, m = 0;
    std::vector<int32_t> xoff, xmid, xcol;   // n + 1, n, Σ |M_c.z|
    std::vector<int32_t> zoff, zmid, zcol;   // n + 1, n, Σ |M_c.x|
    void build(const std::vector<RecColumn>& cols, int nchan_, int n_) {
        n = n_; nchan = nchan_; m = (int)cols.size();
        auto one = [&](bool xrow, std::vector<int32_t>& off, std::vector<int32_t>& mid, std::vector<int32_t>& col) {
            std::vector<int32_t> cnt((size_t)n, 0), cch((size_t)n, 0);
            for (int c = 0; c < m; ++c)
                for (int32_t q : xrow ? cols[(size_t)c].M.z : cols[(size_t)c].M.x) {
                    ++cnt[(size_t)q];
                    if (c < nchan) ++cch[(size_t)q];
                }
            off.assign((size_t)n + 1, 0);
            mid.assign((size_t)n, 0);
            for (int q = 0; q < n; ++q) {
                off[(size_t)q + 1] = off[(size_t)q] + cnt[(size_t)q];
                mid[(size_t)q] = off[(size_t)q] + cch[(size_t)q];
            }
            col.assign((size_t)off[(size_t)n], 0);
            std::vector<int32_t> pos(off.begin(), off.end() - 1);
            for (int c = 0; c < m; ++c)        // ascending c: each row fills in ascending order, channels first
                for (int32_t q : xrow ? cols[(size_t)c].M.z : cols[(size_t)c].M.x) col[(size_t)pos[(size_t)q]++] = c;
        };
        one(true, xoff, xmid, xcol);
        one(false, zoff, zmid, zcol);
    }
    // The row of wire q: [b, e) of col; channels only (e = mid), the extras only (b = mid), or every column.
    enum Part { CHANNELS, EXTRAS, ALL };
    // The dense words of one part over every wire: CHANNELS width CW (⌈nchan/64⌉), EXTRAS width XR at the extras' own
    // bit positions (c − nchan), ALL width QCW; `min_size` the old layout's minimum vector length.
    std::vector<uint64_t> words(bool xrow, Part part, int width, size_t min_size) const {
        std::vector<uint64_t> w(std::max((size_t)n * (size_t)width, min_size), 0);
        const std::vector<int32_t>& off = xrow ? xoff : zoff;
        const std::vector<int32_t>& mid = xrow ? xmid : zmid;
        const std::vector<int32_t>& col = xrow ? xcol : zcol;
        for (int q = 0; q < n; ++q) {
            const int32_t b = part == EXTRAS ? mid[(size_t)q] : off[(size_t)q];
            const int32_t e = part == CHANNELS ? mid[(size_t)q] : off[(size_t)q + 1];
            for (int32_t i = b; i < e; ++i) {
                const int c = col[(size_t)i] - (part == EXTRAS ? nchan : 0);
                w[(size_t)q * (size_t)width + (size_t)(c >> 6)] |= 1ULL << (c & 63);
            }
        }
        return w;
    }
    // The density switch (production storage choice; identical outputs both ways): when the full rows' dense words
    // (2 · n · QCW words) are at most kDenseRatio times the lists' bytes — small circuits, where a word XOR per wire
    // beats an index walk — they are kept as well (dx / dz, n × QCW each) and the full-row reader
    // (record_values_words_, per recorded shot) XORs them; otherwise dx / dz stay empty. `layout`: 0 auto, 1 always
    // lists only, 2 always the words too (XTIM_PLANES_LAYOUT=sparse / =dense: a test lever, read in twirl_sampler.cpp).
    static constexpr size_t kDenseRatio = 4;
    int QCW = 0;
    std::vector<uint64_t> dx, dz;
    void choose_layout(int layout) {
        QCW = std::max(1, (m + 63) / 64);
        const size_t dense = (size_t)2 * (size_t)n * (size_t)QCW * sizeof(uint64_t);
        const bool keep = layout == 2 || (layout == 0 && dense <= kDenseRatio * list_bytes());
        if (keep) { dx = words(true, ALL, QCW, 0); dz = words(false, ALL, QCW, 0); }
        else { std::vector<uint64_t>().swap(dx); std::vector<uint64_t>().swap(dz); }
    }
    size_t list_bytes() const {
        return (xoff.capacity() + xmid.capacity() + xcol.capacity() + zoff.capacity() + zmid.capacity() +
                zcol.capacity()) * sizeof(int32_t);
    }
    size_t bytes() const { return list_bytes() + (dx.capacity() + dz.capacity()) * sizeof(uint64_t); }
};

// S2-T7: toggle bit c of a word row (the per-wire rows' and the readout-flip columns' readers: one toggle per entry).
inline void toggle_bit(uint64_t* w, int32_t c) { w[(size_t)(c >> 6)] ^= 1ULL << (c & 63); }

}  // namespace qeccore
