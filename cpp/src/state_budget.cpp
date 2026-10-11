#include "qeccore/state_budget.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "qeccore/bare_state_errors.hpp"

namespace qeccore {

namespace state_budget_detail {

std::atomic<int64_t> g_live{0};

namespace {
thread_local long long t_chi = -1;
thread_local int t_r = -1, t_n = -1;

std::string gib(int64_t b) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.2f GiB", (double)b / (double)(int64_t(1) << 30));
    return buf;
}
}  // namespace

int64_t read_budget() {
    const char* v = std::getenv("XTIM_MAX_STATE_BYTES");
    if (!v || !*v) return kDefaultStateBudgetBytes;
    std::string s(v);
    size_t i = 0;
    while (i < s.size() && std::isdigit((unsigned char)s[i])) ++i;
    int shift = 0;
    bool ok = i > 0 && i + 1 >= s.size();
    if (ok && i < s.size()) {
        switch (std::toupper((unsigned char)s[i])) {
            case 'K': shift = 10; break;
            case 'M': shift = 20; break;
            case 'G': shift = 30; break;
            case 'T': shift = 40; break;
            default: ok = false;
        }
    }
    long double val = ok ? std::strtold(s.substr(0, i).c_str(), nullptr) : 0;
    if (ok) val *= (long double)(int64_t(1) << shift);
    if (!ok || val > 9.0e18L)
        throw BareStateCapacityError("XTIM_MAX_STATE_BYTES='" + s +
                                     "' is not a byte count (digits, optionally followed by one of "
                                     "K, M, G, T = powers of 1024)");
    return (int64_t)val;
}

void refuse(int64_t bytes, int64_t live, int64_t budget, const char* what) {
    std::string ctx;
    if (t_chi >= 0)
        ctx = " (bare state chi = " + std::to_string(t_chi) + " = 2^" + std::to_string(t_r) +
              ", r = " + std::to_string(t_r) + ", n = " + std::to_string(t_n) + ")";
    throw BareStateCapacityError(
        std::string("xtim state-memory budget exceeded: ") + what + ctx + " needs " +
        std::to_string(bytes) + " bytes (" + gib(bytes) + ") while " + std::to_string(live) +
        " bytes (" + gib(live) + ") of chi-scaled state storage are already live; the budget is " +
        std::to_string(budget) + " bytes (" + gib(budget) +
        "). Refused before allocating. Drop states you no longer need, or raise the budget with "
        "the environment variable XTIM_MAX_STATE_BYTES=<bytes> (default 2 GiB = " +
        std::to_string(kDefaultStateBudgetBytes) + "; read once per process).");
}

}  // namespace state_budget_detail

// ── thread scratch ─────────────────────────────────────────────────────────────────────────
namespace {
thread_local int64_t t_scratch_bytes = 0;
thread_local int t_call_depth = 0;
std::atomic<int64_t> g_owners{0};
struct ScratchEntry { ScratchMark* m; ScratchReleaser rel; };
// This thread's registered marks. At thread exit the buffers die with their own thread_local
// destructors; the registry credits whatever is still charged (the marks are PODs, still valid).
struct ScratchRegistry {
    std::vector<ScratchEntry> v;
    ~ScratchRegistry() {
        for (const ScratchEntry& e : v)
            if (e.m->charged) { state_budget_release(e.m->charged); e.m->charged = 0; }
    }
};
ScratchRegistry& thread_registry() {
    static thread_local ScratchRegistry reg;
    return reg;
}
}  // namespace

void scratch_register_slow(ScratchMark& m, ScratchReleaser rel) {
    if (m.registered) return;
    thread_registry().v.push_back({&m, rel});
    m.registered = 1;
}
void scratch_grow_slow(ScratchMark& m, ScratchReleaser rel, int64_t bytes, const char* what) {
    scratch_register_slow(m, rel);
    state_budget_charge(bytes - m.charged, what);            // refuses before the buffers grow
    t_scratch_bytes += bytes - m.charged;
    m.charged = bytes;
}

int64_t thread_scratch_bytes() { return t_scratch_bytes; }
void release_thread_scratch_slots() noexcept {
    for (const ScratchEntry& e : thread_registry().v) {
        if (e.rel) e.rel();
        if (e.m->charged) {
            state_budget_release(e.m->charged);
            t_scratch_bytes -= e.m->charged;
            e.m->charged = 0;
        }
    }
}

ScratchOwner::ScratchOwner() noexcept { g_owners.fetch_add(1, std::memory_order_relaxed); }
ScratchOwner::ScratchOwner(const ScratchOwner&) noexcept { g_owners.fetch_add(1, std::memory_order_relaxed); }
void ScratchOwner::drop_() noexcept {
    // The release frees only the CALLING thread's scratch, so the count needs no ordering.
    if (g_owners.fetch_sub(1, std::memory_order_relaxed) == 1 && t_call_depth == 0)
        release_thread_scratch_slots();
}
ScratchCallScope::ScratchCallScope() noexcept { ++t_call_depth; }
ScratchCallScope::~ScratchCallScope() {
    if (--t_call_depth == 0 && g_owners.load(std::memory_order_relaxed) == 0)
        release_thread_scratch_slots();
}
int64_t scratch_owner_count() { return g_owners.load(std::memory_order_relaxed); }

void state_budget_precheck(int64_t bytes, const char* what) {
    const int64_t budget = state_budget_bytes();
    const int64_t live = live_state_bytes();
    if (live + bytes > budget) state_budget_detail::refuse(bytes, live, budget, what);
}

StateBudgetContext::StateBudgetContext(long long chi, int r, int n)
    : pchi_(state_budget_detail::t_chi), pr_(state_budget_detail::t_r), pn_(state_budget_detail::t_n) {
    state_budget_detail::t_chi = chi;
    state_budget_detail::t_r = r;
    state_budget_detail::t_n = n;
}
StateBudgetContext::~StateBudgetContext() {
    state_budget_detail::t_chi = pchi_;
    state_budget_detail::t_r = pr_;
    state_budget_detail::t_n = pn_;
}

}  // namespace qeccore
