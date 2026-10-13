#pragma once
// Process-wide budget on LIVE χ-scaled state storage (3.1.9).
//
// Every container whose size scales with the branch count χ = 2^r — the branch list of a
// CanonicalStabSum, the amplitude entries of a FramedSuperposition (DenseAmplitudes), a parsed
// RefFile's branches — allocates through StateAllocator, which charges one shared atomic counter on
// allocate() and releases EXACTLY the same amount on deallocate().  Copies, moves, grows, swaps,
// materialised BarrierBuffer states and Python-side GC are therefore all covered by construction:
// the counter follows the storage.  Transient χ-scaled buffers that are not such containers (the
// assembly's coefficient tables, the reference compile's text) are charged by an RAII
// StateBudgetTicket for their lifetime.
//
// An allocation that would push the live total over the budget is REFUSED BEFORE it happens with a
// BareStateCapacityError (Python: xtim.XtimCapacityError, a MemoryError) naming the requested
// bytes, the live total, the budget, χ/r/n where known and the override.  The budget is 2 GiB by
// default; XTIM_MAX_STATE_BYTES=<bytes> (suffixes K/M/G/T = powers of 1024 accepted) overrides it.
// It is read ONCE per process (the first χ-scaled allocation).
//
// Cost on the hot path: one relaxed atomic add per allocation (and one subtract per release); a
// buffer reused in place (assign_from, clear + push_back within capacity) costs nothing.
//
// What a charge counts: sizeof(element) per element plus `K` bytes for an element's own heap block
// (a branch's sign pattern σ: r ≤ kMaxBareRank = 30 bytes, one glibc chunk of 32 B for r ≤ 24 and
// 48 B for r = 25..40; charged as 48 B for every r since a stateless allocator does not know r — an
// over-count of 16 B per branch for r ≤ 24, never an under-count; 3.1.9 charged 32 B and so
// under-counted r = 25..30).  The absolute number is an accounting estimate (calibrated against peak
// RSS in the 3.1.9 report); the pairing is exact.
//
// Thread scratch (3.1.10).  The measurement / Born / canonicalise kernels keep χ-scaled per-thread
// scratch (packed per-branch sign rows, coefficient copies, per-branch Pauli images, partner hash
// tables, …) in capacity-retaining thread_local buffers.  Each such kernel owns one ScratchMark: it
// charges the kernel's χ-scaled scratch estimate (a per-branch byte formula at the call's χ, kept as
// a high-water mark) BEFORE the buffers grow, and its release() frees the buffers and credits the
// charge.  Scratch whose size does not scale with χ (O(n) words, O(n²) bit matrices, O(#reads)
// tables) is not charged.  release_thread_state_scratch() releases every slot of the calling thread
// (plus the anticommuting measurement's reusable entry list).
//
// Automatic release (3.1.10).  The Python bindings count "owners" (live Python-held samplers,
// states, barrier buffers, compiled programs) and module-level calls in flight on this thread; when
// an owner is dropped or a module call returns and no owner is left (and no call is in flight on
// this thread), the calling thread's scratch is released, so live_state_bytes() returns to its
// baseline without an explicit release.  Scratch of OTHER threads is released when those threads
// next reach that point, or exit.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <type_traits>

namespace qeccore {

// The malloc chunk of a branch's σ vector (≤ 30 payload bytes → at most a 48-byte glibc chunk).
constexpr std::size_t kSigmaHeapBytes = 48;
constexpr int64_t kDefaultStateBudgetBytes = int64_t(2) << 30;   // 2 GiB

namespace state_budget_detail {
extern std::atomic<int64_t> g_live;
int64_t read_budget();                                   // env parse (throws on a bad value)
[[noreturn]] void refuse(int64_t bytes, int64_t live, int64_t budget, const char* what);
}  // namespace state_budget_detail

// The budget in bytes (XTIM_MAX_STATE_BYTES or 2 GiB), read once.
inline int64_t state_budget_bytes() {
    static const int64_t b = state_budget_detail::read_budget();
    return b;
}
// Bytes of χ-scaled state storage currently live in this process.
inline int64_t live_state_bytes() {
    return state_budget_detail::g_live.load(std::memory_order_relaxed);
}

// Charge `bytes`; on refusal nothing stays charged and BareStateCapacityError is thrown.
inline void state_budget_charge(int64_t bytes, const char* what) {
    const int64_t budget = state_budget_bytes();
    const int64_t prev = state_budget_detail::g_live.fetch_add(bytes, std::memory_order_relaxed);
    if (__builtin_expect(prev + bytes > budget, 0)) {
        state_budget_detail::g_live.fetch_sub(bytes, std::memory_order_relaxed);
        state_budget_detail::refuse(bytes, prev, budget, what);
    }
}
inline void state_budget_release(int64_t bytes) noexcept {
    state_budget_detail::g_live.fetch_sub(bytes, std::memory_order_relaxed);
}
// Refuse up front (nothing charged) when `bytes` more would not fit — a pre-check that names the
// whole build before any of its allocations happen.
void state_budget_precheck(int64_t bytes, const char* what);

// χ / r / n of the build in progress on this thread, named by refusals raised under it.
struct StateBudgetContext {
    StateBudgetContext(long long chi, int r, int n);
    ~StateBudgetContext();
    StateBudgetContext(const StateBudgetContext&) = delete;
    StateBudgetContext& operator=(const StateBudgetContext&) = delete;
private:
    long long pchi_; int pr_, pn_;
};

// RAII charge for a transient χ-scaled buffer: charged on construction (refusing before the
// buffer is allocated), released exactly on destruction.
class StateBudgetTicket {
public:
    StateBudgetTicket() = default;
    StateBudgetTicket(int64_t bytes, const char* what) { charge(bytes, what); }
    ~StateBudgetTicket() { reset(); }
    StateBudgetTicket(const StateBudgetTicket&) = delete;
    StateBudgetTicket& operator=(const StateBudgetTicket&) = delete;
    StateBudgetTicket(StateBudgetTicket&& o) noexcept : b_(o.b_) { o.b_ = 0; }
    StateBudgetTicket& operator=(StateBudgetTicket&& o) noexcept {
        if (this != &o) { reset(); b_ = o.b_; o.b_ = 0; }
        return *this;
    }
    void charge(int64_t bytes, const char* what) {
        if (bytes <= 0) return;
        state_budget_charge(bytes, what);
        b_ += bytes;
    }
    void reset() noexcept { if (b_) { state_budget_release(b_); b_ = 0; } }
    int64_t bytes() const { return b_; }
private:
    int64_t b_ = 0;
};

// Stateless allocator charging n·(sizeof(T) + K) bytes per allocation.
template <class T, std::size_t K = 0>
struct StateAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    template <class U> struct rebind { using other = StateAllocator<U, K>; };

    StateAllocator() noexcept = default;
    template <class U> StateAllocator(const StateAllocator<U, K>&) noexcept {}

    static int64_t bytes_for(std::size_t n) { return (int64_t)n * (int64_t)(sizeof(T) + K); }

    T* allocate(std::size_t n) {
        const int64_t b = bytes_for(n);
        state_budget_charge(b, "chi-scaled state storage");
        try {
            return std::allocator<T>().allocate(n);
        } catch (...) {
            state_budget_release(b);
            throw;
        }
    }
    void deallocate(T* p, std::size_t n) noexcept {
        std::allocator<T>().deallocate(p, n);
        state_budget_release(bytes_for(n));
    }
    template <class U> bool operator==(const StateAllocator<U, K>&) const noexcept { return true; }
    template <class U> bool operator!=(const StateAllocator<U, K>&) const noexcept { return false; }
};

// ── thread scratch ─────────────────────────────────────────────────────────────────────────
// glibc malloc chunk of a `bytes`-byte block (0 for no block): the heap a vector of that payload
// really holds.  Used by the per-branch scratch formulas.
constexpr int64_t heap_chunk_bytes(int64_t bytes) {
    return bytes <= 0 ? 0 : (((bytes + 8 + 15) & ~int64_t(15)) < 32 ? 32 : ((bytes + 8 + 15) & ~int64_t(15)));
}

// One kernel's χ-scaled per-thread scratch: a high-water charge (a constant-initialised
// thread_local POD, so the per-call check is one TLS load and a compare) plus the function that
// frees the kernel's buffers.  In the kernel:
//     static thread_local ScratchMark slot_mark{0, 0};
//     static constexpr ScratchReleaser slot_rel = +[] { /* swap the buffers with empties */ };
//     scratch_need(slot_mark, slot_rel, bytes, "what");   // before the buffers grow
// The first growth registers the mark in the calling thread's registry (release / thread exit).
using ScratchReleaser = void (*)();
struct ScratchMark {
    int64_t charged;
    uint8_t registered;
};
// Slow path: charge the growth (refusing before the buffers grow; a refusal leaves the previous
// charge unchanged) and register the mark on first use.
void scratch_grow_slow(ScratchMark& m, ScratchReleaser rel, int64_t bytes, const char* what);
void scratch_register_slow(ScratchMark& m, ScratchReleaser rel);
inline void scratch_need(ScratchMark& m, ScratchReleaser rel, int64_t bytes, const char* what) {
    if (__builtin_expect(bytes > m.charged, 0)) scratch_grow_slow(m, rel, bytes, what);
}
// Register a buffer that is charged elsewhere (by its allocator) for the thread's release only.
inline void scratch_track(ScratchMark& m, ScratchReleaser rel) {
    if (__builtin_expect(!m.registered, 0)) scratch_register_slow(m, rel);
}

// Bytes currently charged by the calling thread's scratch slots.
int64_t thread_scratch_bytes();
// Release every scratch slot of the calling thread (and the reusable measurement entry list).
void release_thread_scratch_slots() noexcept;

// Owner / call accounting for the automatic release (bindings).  An owner is a Python-held object
// (sampler, state, buffer, compiled program); a call scope is a module-level call in flight on this
// thread.  When the last owner is dropped or the outermost call scope exits with no owner left,
// the calling thread's scratch is released.
// A move TRANSFERS the count (no atomic): the moved-from object stops counting.
class ScratchOwner {
public:
    ScratchOwner() noexcept;
    ScratchOwner(const ScratchOwner&) noexcept;
    ScratchOwner(ScratchOwner&& o) noexcept : active_(o.active_) { o.active_ = false; }
    ScratchOwner& operator=(const ScratchOwner&) noexcept { return *this; }
    ScratchOwner& operator=(ScratchOwner&&) noexcept { return *this; }
    ~ScratchOwner() { if (active_) drop_(); }
private:
    void drop_() noexcept;
    bool active_ = true;
};
struct ScratchCallScope {
    ScratchCallScope() noexcept;
    ~ScratchCallScope();
    ScratchCallScope(const ScratchCallScope&) = delete;
    ScratchCallScope& operator=(const ScratchCallScope&) = delete;
};
int64_t scratch_owner_count();

}  // namespace qeccore
