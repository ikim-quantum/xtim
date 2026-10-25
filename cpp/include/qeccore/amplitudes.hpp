#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/state_budget.hpp"

namespace qeccore {

using cd = std::complex<double>;

// ---------------------------------------------------------------------------
// Packed free-register keys (S2k)
// ---------------------------------------------------------------------------
// An amplitude key x ∈ {0,1}^k is ONE uint64 word, bit d = x_d; bits ≥ k are always 0. The width is
// derived from the engine's branch-rank cap: every bare-state builder refuses r > kMaxBareRank, so a
// built state's free register fits one word. The paths whose k is NOT bounded by that cap (states
// assembled from user rows, tensor products, add_index) refuse k > kAmpKeyBits by name
// (check_amp_key_width) instead of growing a multi-word key.
using AmpKey = uint64_t;
constexpr int kAmpKeyBits = 64;
static_assert(kMaxBareRank <= kAmpKeyBits,
              "a bare state's free register (k <= kMaxBareRank) must fit one packed AmpKey word");

inline void check_amp_key_width(long long k, const char* where) {
    if (k < 0 || k > kAmpKeyBits)
        throw BareStateCapacityError(std::string(where) + ": refused: the free register has k = " +
                                     std::to_string(k) + " coordinates; an amplitude key holds at "
                                     "most " + std::to_string(kAmpKeyBits) + " (one 64-bit word)");
}
inline uint8_t key_bit(AmpKey key, int d) { return (uint8_t)((key >> d) & 1u); }
// Bits 0..k-1 set (k in 0..64; no 64-bit shift): masks a key to its register.
inline AmpKey key_low_mask(int k) { return k >= kAmpKeyBits ? ~AmpKey(0) : ((AmpKey(1) << k) - 1); }
// Pack a byte key (one byte per coordinate, each 0 or 1). Any other byte value is refused by name.
inline AmpKey pack_key(const std::vector<uint8_t>& x, const char* where) {
    check_amp_key_width((long long)x.size(), where);
    AmpKey key = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] > 1)
            throw std::invalid_argument(std::string(where) + ": sigma byte " + std::to_string(i) +
                                        " is " + std::to_string((int)x[i]) + " (must be 0 or 1)");
        key |= (AmpKey)x[i] << i;
    }
    return key;
}
inline std::vector<uint8_t> unpack_key(AmpKey key, int k) {
    std::vector<uint8_t> x((size_t)k);
    for (int i = 0; i < k; ++i) x[(size_t)i] = key_bit(key, i);
    return x;
}

// ---------------------------------------------------------------------------
// AmpIndex — the caller-owned O(χ) key → position lookup of inner / axpy (S2k)
// ---------------------------------------------------------------------------
// A TRANSIENT table, rebuilt from the target list at the start of every inner/axpy call (so it can
// never be stale after a prune / drop_index / add_index), insert-if-absent in list order (= the
// former scan's FIRST match). Layout by the minimum-memory rule, no tuned constant: direct
// addressing pos[x] (2^k slots) when 2^k ≤ next_pow2(2·maxkeys), else an open-addressing hash of
// next_pow2(2·maxkeys) slots (load ≤ ½, linear probing). Slots hold position+1 (0 = empty).
// The list is the order of record; the table is only a lookup and never reorders anything.
//
// Ownership: the CALLER owns it (a FramedSuperposition, the Born trie's working state, or a stack
// frame) and passes it down. Never copied: a copy (or copy-assignment target) starts / stays with
// its own table, so a state copy neither aliases nor re-charges a warm table. Heap slots (beyond a
// stack frame's inline buffer, AmpIndexLocal) are charged to the state budget by the owned RAII ticket (high-water, released on
// destruction). Debug builds guard against re-entrant use and cross-check lookups against a scan.
class AmpIndex {
  public:
    AmpIndex() noexcept = default;
    AmpIndex(const AmpIndex&) noexcept {}
    AmpIndex& operator=(const AmpIndex&) noexcept { return *this; }
    ~AmpIndex() = default;

    // Exact counters (mechanism tests): tables built, slots cleared, slot reads (probes).
    uint64_t builds = 0, slots = 0, probes = 0;
    int64_t charged_bytes() const { return ticket_.bytes(); }

  protected:
    // Optional caller-provided inline slots (AmpIndexLocal: a stack frame's index); an owned member
    // index has none, so it adds no inline buffer to its owner's size.
    uint32_t* inline_ = nullptr;
    size_t inline_cap_ = 0;

  private:
    friend struct DenseAmplitudes;
    std::unique_ptr<uint32_t[]> heap_;
    size_t heap_cap_ = 0;
    StateBudgetTicket ticket_;
    uint32_t* tab_ = nullptr;
    uint64_t mask_ = 0;
    int shift_ = 0;
    bool direct_ = false;
    bool busy_ = false;   // the debug re-entrancy guard's flag (always present: the layout must not depend on NDEBUG)

    void prepare_(int k, size_t maxkeys) {
        size_t hsize = 2;
        while (hsize < 2 * maxkeys) hsize <<= 1;
        direct_ = k < 63 && ((size_t)1 << k) <= hsize;
        const size_t size = direct_ ? ((size_t)1 << k) : hsize;
        if (size <= inline_cap_) {
            tab_ = inline_;
        } else {
            if (size > heap_cap_) {
                ticket_.charge((int64_t)(size - heap_cap_) * (int64_t)sizeof(uint32_t),
                               "amplitude key index");
                heap_.reset(new uint32_t[size]);
                heap_cap_ = size;
            }
            tab_ = heap_.get();
        }
        std::memset(tab_, 0, size * sizeof(uint32_t));
        mask_ = size - 1;
        int lg = 0;
        while (((size_t)1 << lg) < size) ++lg;
        shift_ = 64 - lg;   // hash only (size ≥ 2 ⇒ lg ≥ 1 ⇒ shift ≤ 63)
        ++builds;
        slots += size;
    }
    uint64_t home_(AmpKey key) const {
        return direct_ ? key : ((key * 0x9E3779B97F4A7C15ull) >> shift_);
    }
};

// A stack frame's index: 64 inline slots (no heap, no charge up to 64), heap + ticket beyond.
class AmpIndexLocal : public AmpIndex {
  public:
    AmpIndexLocal() noexcept { inline_ = buf_; inline_cap_ = kInline; }
    AmpIndexLocal(const AmpIndexLocal&) = delete;
    AmpIndexLocal& operator=(const AmpIndexLocal&) = delete;
  private:
    static constexpr size_t kInline = 64;
    uint32_t buf_[kInline];
};

// ---------------------------------------------------------------------------
// Abstract interface
// ---------------------------------------------------------------------------
struct Amplitudes {
    virtual ~Amplitudes() = default;

    virtual int k() const = 0;                                   // number of logical indices
    virtual std::unique_ptr<Amplitudes> clone() const = 0;

    // Deep-copy `other` into this instance IN PLACE (per-shot buffer reuse: no realloc when
    // the existing buffers already fit). Backends must match; throws on mismatch.
    virtual void assign_from(const Amplitudes& other) = 0;

    // Read α_x (returns 0 if x is absent from support).
    virtual cd at(const std::vector<uint8_t>& x) const = 0;

    // Number of stored nonzero entries (the "χ").
    virtual int support() const = 0;

    // shift: α_x  <-  α_{x XOR y}   (relabel keys); byte key and packed key forms
    virtual void shift(const std::vector<uint8_t>& y) = 0;
    virtual void shift_key(AmpKey y) = 0;

    // diagonal: α_x  <-  g0 * Π_i g[i]^{x_i} * α_x
    virtual void apply_diagonal(cd g0, const std::vector<cd>& g) = 0;

    // axpy: this  <-  this + c * other   (other must have the same k); `idx` is the caller's lookup
    // scratch (the overload without it uses a transient one on the stack)
    virtual void axpy(cd c, const Amplitudes& other, AmpIndex& idx) = 0;
    void axpy(cd c, const Amplitudes& other) { AmpIndexLocal idx; axpy(c, other, idx); }

    // inner product: Σ_x  conj(this_x) * other_x   (summed in OTHER's entry order)
    virtual cd inner(const Amplitudes& other, AmpIndex& idx) const = 0;
    cd inner(const Amplitudes& other) const { AmpIndexLocal idx; return inner(other, idx); }

    // ||this||^2 = inner(this).real()
    virtual double norm2() const = 0;

    // scale: α_x  <-  c * α_x
    virtual void scale(cd c) = 0;

    // normalize: divide by sqrt(norm2)
    virtual void normalize() = 0;

    // drop_index(i): remove index i (must be constant across all entries first)
    virtual void drop_index(int i) = 0;

    // add_index: append a fresh index; α_{x,0} = α_x, α_{x,1} = 0
    virtual void add_index() = 0;
};

// ---------------------------------------------------------------------------
// Dense (sparse key-list) backend
// ---------------------------------------------------------------------------
struct DenseAmplitudes final : Amplitudes {
    int k_;
    // Sparse list of (packed key, coefficient) pairs. Keys are unique (every writer merges); only
    // nonzero entries are stored; the list ORDER is the kernels' summation order (load-bearing).
    // χ-scaled: allocated through the process-wide state budget (state_budget.hpp) — every
    // allocation/release of the list is charged/credited exactly, sizeof(Entry) per entry (the
    // packed key has no heap block of its own).
    using Entry = std::pair<AmpKey, cd>;
    using EntryVec = std::vector<Entry, StateAllocator<Entry>>;
    EntryVec b_;

    explicit DenseAmplitudes(int k) : k_(k) {}

    int k() const override { return k_; }

    std::unique_ptr<Amplitudes> clone() const override {
        auto copy = std::make_unique<DenseAmplitudes>(k_);
        copy->b_ = b_;
        return copy;
    }

    void assign_from(const Amplitudes& other) override {
        const auto* od = dynamic_cast<const DenseAmplitudes*>(&other);
        if (!od) throw std::runtime_error("assign_from: backend mismatch");
        k_ = od->k_;
        b_ = od->b_;   // vector copy-assign: reuses this instance's buffers where they fit
    }

    cd at(const std::vector<uint8_t>& x) const override {
        // Byte-key boundary: a key of another width, or with a byte other than 0/1, matches no
        // stored key (the former vector equality), so it reads 0.
        if ((int)x.size() != k_) return cd(0, 0);
        AmpKey key = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i] > 1) return cd(0, 0);
            key |= (AmpKey)x[i] << i;
        }
        for (auto& [ekey, val] : b_) {
            if (ekey == key) return val;
        }
        return cd(0, 0);
    }

    int support() const override {
        return static_cast<int>(b_.size());
    }

    void shift(const std::vector<uint8_t>& y) override {
        // α_x <- α_{x XOR y}: relabel every key x -> x XOR y (the first k bytes of y)
        if ((int)y.size() < k_) throw std::invalid_argument("shift: y shorter than k");
        shift_key(pack_key(std::vector<uint8_t>(y.begin(), y.begin() + k_), "shift"));
    }

    void shift_key(AmpKey y) override {
        // Only the register's k bits shift (as the byte form's first-k-bytes truncation does);
        // a wider y is a caller bug, loud in debug builds.
        assert((y & ~key_low_mask(k_)) == 0);
        y &= key_low_mask(k_);
        for (auto& e : b_) e.first ^= y;
    }

    void apply_diagonal(cd g0, const std::vector<cd>& g) override {
        for (auto& [key, val] : b_) {
            cd factor = g0;
            // x_i = 1: multiply by g[i] — set bits in ASCENDING i (the former byte loop's order)
            for (AmpKey bits = key; bits; bits &= bits - 1) factor *= g[(size_t)__builtin_ctzll(bits)];
            val *= factor;
        }
        prune();
    }

    using Amplitudes::axpy;
    void axpy(cd c, const Amplitudes& other, AmpIndex& idx) override {
        // Merge other's entries, in other's order, into b_: an existing key accumulates at its
        // (first) position, a new key is appended AND indexed immediately; nothing is reordered.
        const auto* od = dynamic_cast<const DenseAmplitudes*>(&other);
        if (!od) throw std::runtime_error("axpy: backend mismatch");
        const size_t no = od->b_.size();
        IndexUse use(idx);
        build_index_(idx, b_.size() + no);
        uint64_t probes = 0;
        for (size_t t = 0; t < no; ++t) {
            const AmpKey okey = od->b_[t].first;
            cd contribution = c * od->b_[t].second;
            uint64_t s;
            const long pos = find_(idx, okey, s, probes, b_.size());
            if (pos >= 0) {
                b_[(size_t)pos].second += contribution;
            } else {
                idx.tab_[s] = (uint32_t)(b_.size() + 1);
                b_.emplace_back(okey, contribution);
            }
        }
        idx.probes += probes;
        prune();
    }

    using Amplitudes::inner;
    cd inner(const Amplitudes& other, AmpIndex& idx) const override {
        const auto* od = dynamic_cast<const DenseAmplitudes*>(&other);
        if (!od) throw std::runtime_error("inner: backend mismatch");
        IndexUse use(idx);
        build_index_(idx, b_.size());
        cd result(0, 0);
        uint64_t probes = 0;
        for (auto& [okey, oval] : od->b_) {
            uint64_t s;
            const long pos = find_(idx, okey, s, probes, b_.size());
            if (pos >= 0) result += std::conj(b_[(size_t)pos].second) * oval;
        }
        idx.probes += probes;
        return result;
    }

    double norm2() const override {
        double s = 0.0;
        for (auto& [key, val] : b_) {
            s += std::norm(val);   // std::norm(z) = |z|^2
        }
        return s;
    }

    void scale(cd c) override {
        if (std::abs(c) < 1e-15) {
            b_.clear();
            return;
        }
        for (auto& [key, val] : b_) {
            val *= c;
        }
        prune();
    }

    void normalize() override {
        double n2 = norm2();
        if (n2 < 1e-30) return;  // avoid dividing by ~zero
        double inv = 1.0 / std::sqrt(n2);
        for (auto& [key, val] : b_) {
            val *= inv;
        }
    }

    void drop_index(int i) override {
        // Check that bit i is the same across all entries (survives Release builds)
        if (!b_.empty()) {
            const AmpKey ref_bit = (b_[0].first >> i) & 1u;
            for (auto& entry : b_) {
                if (((entry.first >> i) & 1u) != ref_bit)
                    throw std::runtime_error("drop_index: index not constant across support");
            }
        }
        // Remove bit i from every key (the higher bits move down by one)
        const AmpKey low = (i == 0) ? 0 : (~AmpKey(0) >> (kAmpKeyBits - i));
        for (auto& e : b_) e.first = (e.first & low) | ((e.first >> i >> 1) << i);
        k_--;
    }

    void add_index() override {
        // Append a 0 bit to every key: the packed keys are unchanged.
        check_amp_key_width((long long)k_ + 1, "add_index");
        k_++;
    }

    // make_dense's merge: append each nonzero byte-key branch, accumulating a repeated key at its
    // first position; then the stable prune.
    void merge_bytes_(const std::vector<std::pair<std::vector<uint8_t>, cd>>& branches, AmpIndex& idx) {
        IndexUse use(idx);
        build_index_(idx, b_.size() + branches.size());
        uint64_t probes = 0;
        for (auto& [key, val] : branches) {
            if (std::abs(val) < 1e-15) continue;
            if ((int)key.size() != k_) throw std::invalid_argument("make_dense: key width != k");
            const AmpKey pk = pack_key(key, "make_dense");
            uint64_t s;
            const long pos = find_(idx, pk, s, probes, b_.size());
            if (pos >= 0) {
                b_[(size_t)pos].second += val;
            } else {
                idx.tab_[s] = (uint32_t)(b_.size() + 1);
                b_.emplace_back(pk, val);
            }
        }
        idx.probes += probes;
        prune();
    }

private:
    // Debug-build re-entrancy guard on the caller's scratch.
    struct IndexUse {
#ifndef NDEBUG
        AmpIndex& i;
        explicit IndexUse(AmpIndex& x) : i(x) { assert(!i.busy_ && "AmpIndex re-entered"); i.busy_ = true; }
        ~IndexUse() { i.busy_ = false; }
#else
        explicit IndexUse(AmpIndex&) {}
#endif
    };

    // Build the lookup over b_ (insert-if-absent in list order = the scan's first match).
    void build_index_(AmpIndex& idx, size_t maxkeys) const {
        idx.prepare_(k_, maxkeys);
        uint64_t probes = 0;
        for (size_t p = 0; p < b_.size(); ++p) {
            uint64_t s;
            if (find_(idx, b_[p].first, s, probes, p) < 0) idx.tab_[s] = (uint32_t)(p + 1);
        }
        idx.probes += probes;
    }

    // Position of `key` in b_ (−1 if absent; then `slot` is where it would be inserted). `indexed`:
    // how many leading entries the table holds (the debug cross-check scans exactly those).
    long find_(const AmpIndex& idx, AmpKey key, uint64_t& slot, uint64_t& probes, size_t indexed) const {
        assert(k_ >= 64 || (key >> k_) == 0);
        uint64_t s = idx.home_(key);
        long pos = -1;
        for (;;) {
            ++probes;
            const uint32_t v = idx.tab_[s];
            if (v == 0) break;
            if (b_[v - 1].first == key) { pos = (long)v - 1; break; }
            if (idx.direct_) break;   // unreachable: a direct slot holds only its own key
            s = (s + 1) & idx.mask_;
        }
        slot = s;
#ifndef NDEBUG
        if (idx.builds <= 4096) {   // index lookup == linear scan (first match)
            long scan = -1;
            for (size_t p = 0; p < indexed; ++p) if (b_[p].first == key) { scan = (long)p; break; }
            assert(scan == pos && "AmpIndex lookup disagrees with the linear scan");
        }
#endif
        return pos;
    }

    // Remove entries whose coefficient is effectively zero — stable (order-preserving).
    void prune() {
        b_.erase(std::remove_if(b_.begin(), b_.end(),
                                [](const Entry& e) { return std::abs(e.second) < 1e-15; }),
                 b_.end());
    }
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

inline std::unique_ptr<Amplitudes> make_dense(
    int k,
    const std::vector<std::pair<std::vector<uint8_t>, cd>>& branches)
{
    // Byte-key boundary: packs each key (width k, bytes 0/1, else refused by name), merges
    // duplicates at their first position (list order = input order of first appearance), then
    // prunes ~zero merged coefficients (order-preserving).
    check_amp_key_width(k, "make_dense");
    auto d = std::make_unique<DenseAmplitudes>(k);
    AmpIndexLocal idx;
    d->merge_bytes_(branches, idx);
    return d;
}

inline std::unique_ptr<Amplitudes> make_dense_basis0(int k) {
    check_amp_key_width(k, "make_dense_basis0");
    auto d = std::make_unique<DenseAmplitudes>(k);
    d->b_.emplace_back(AmpKey(0), cd(1.0, 0.0));
    return d;
}

}  // namespace qeccore
