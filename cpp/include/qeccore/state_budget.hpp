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
// (a branch's sign pattern σ: r ≤ 24 bytes, one 32-byte malloc chunk).  The absolute number is an
// accounting estimate (calibrated against peak RSS in the 3.1.9 report); the pairing is exact.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <type_traits>

namespace qeccore {

// The malloc chunk of a branch's σ vector (≤ 24 payload bytes → 32-byte glibc chunk).
constexpr std::size_t kSigmaHeapBytes = 32;
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

}  // namespace qeccore
