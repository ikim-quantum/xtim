// thread_pool — the process-wide worker pool (see qeccore/thread_pool.hpp).
#include "qeccore/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>   // _getpid (MSVC has no <unistd.h>)
#define getpid _getpid
using pid_t = int;
#else
#include <unistd.h>
#endif

namespace qeccore {

namespace {
constexpr size_t kLaneHeadroom = 60u << 10;
void* (*volatile g_lane_malloc)(size_t) = std::malloc;   // volatile: the compiler may not elide the pair
}  // namespace

namespace {
std::atomic<int> g_pool_push_fail_at{-1};            // fix round 2 test lever: the run() push that throws (-1: none)
}  // namespace
void thread_pool_fail_push_for_test(int i) { g_pool_push_fail_at.store(i, std::memory_order_relaxed); }

void lane_heap_headroom() { std::free(g_lane_malloc(kLaneHeadroom)); }

void lane_heap_reserve(size_t bytes) {
    // fid-routine B1: the whole request (it was capped at 64 blocks, 3.75 MiB, against ~76 MB built per RF build job on
    // the factory): the block list on the stack up to 64 blocks, else on the heap; at most kLaneReserveMax per call
    constexpr size_t kLaneReserveMax = size_t(1) << 30;
    const size_t k = (std::min(bytes, kLaneReserveMax) + kLaneHeadroom - 1) / kLaneHeadroom;
    void* stk[64];
    std::vector<void*> heap;
    void** blk = stk;
    if (k > 64) {
        try { heap.resize(k); } catch (...) { return; }   // (a reserve is a speed device: none on failure)
        blk = heap.data();
    }
    size_t got = 0;
    for (; got < k; ++got)
        if (!(blk[got] = g_lane_malloc(kLaneHeadroom))) break;
    for (size_t i = got; i-- > 0;) std::free(blk[i]);
}

namespace {
// One call's job slots. state: 0 = queued, 1 = running, 2 = finished or withdrawn. A slot is claimed (0 -> 1) and
// withdrawn (0 -> 2) only under the pool's queue lock, so no worker can hold a task of a call that has returned.
struct Call {
    const std::function<void(int)>* job = nullptr;
    std::unique_ptr<int[]> state;                    // under the pool's queue lock
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> running{0};                     // claimed, not finished (decremented under mu)
};
struct Task {
    Call* c;
    int i;
};
}  // namespace

struct ThreadPool::Impl {
    std::mutex mu;                                   // the queue and every call's slot states
    std::condition_variable cv;
    std::deque<Task> q;
    std::vector<std::thread> th;
    pid_t pid = getpid();

    void worker() {
        for (;;) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !q.empty(); });
                t = q.front();
                q.pop_front();
                if (t.c->state[(size_t)t.i] != 0) continue;   // withdrawn
                t.c->state[(size_t)t.i] = 1;
                t.c->running.fetch_add(1, std::memory_order_relaxed);
            }
            Call& c = *t.c;
            (*c.job)(t.i);
            std::lock_guard<std::mutex> lk(c.mu);    // the last touch of `c`: notify under its lock
            c.running.fetch_sub(1, std::memory_order_relaxed);
            c.cv.notify_all();
        }
    }
    void grow(size_t want) {   // under mu; a thread the OS refuses stops the growth (fix F4)
        while (th.size() < want) {
            try {
                th.emplace_back([this] { worker(); });
            } catch (const std::system_error&) {
                return;
            }
            th.back().detach();   // workers live for the process (never joined: no static-destruction order issue)
        }
    }
};

ThreadPool::ThreadPool() : p_(new Impl()) {}

ThreadPool& ThreadPool::instance() {
    static std::mutex m;
    static ThreadPool* pool = nullptr;               // leaked on purpose: workers outlive static destruction
    std::lock_guard<std::mutex> lk(m);
    if (!pool || pool->p_->pid != getpid()) pool = new ThreadPool();   // first use, or a forked child (no workers)
    return *pool;
}

int ThreadPool::reserve(int n) {
    std::lock_guard<std::mutex> lk(p_->mu);
    if (n > 1) p_->grow((size_t)n - 1);
    return (int)p_->th.size();
}

int ThreadPool::workers() const {
    std::lock_guard<std::mutex> lk(p_->mu);
    return (int)p_->th.size();
}
int ThreadPool::queued_for_test() const {
    std::lock_guard<std::mutex> lk(p_->mu);
    return (int)p_->q.size();
}

void ThreadPool::run(int n, const std::function<void(int)>& job) {
    if (n <= 1) {
        job(0);
        return;
    }
    Call c;
    c.job = &job;
    c.state.reset(new int[(size_t)n]());
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->grow((size_t)n - 1);
        try {                                         // fix round 2: a failed push leaves no task naming this frame's call
            for (int i = 1; i < n; ++i) {
                if (g_pool_push_fail_at.load(std::memory_order_relaxed) == i) {   // test lever
                    g_pool_push_fail_at.store(-1, std::memory_order_relaxed);
                    throw std::bad_alloc();
                }
                p_->q.push_back(Task{&c, i});
            }
        } catch (...) {                               // (under the queue lock: no worker claimed one)
            for (auto it = p_->q.begin(); it != p_->q.end();) it = (it->c == &c) ? p_->q.erase(it) : it + 1;
            throw;
        }
    }
    p_->cv.notify_all();
    job(0);
    {   // withdraw every job no worker has claimed, and drop this call's tasks from the queue
        std::lock_guard<std::mutex> lk(p_->mu);
        for (int i = 1; i < n; ++i)
            if (c.state[(size_t)i] == 0) c.state[(size_t)i] = 2;
        for (auto it = p_->q.begin(); it != p_->q.end();)
            it = (it->c == &c) ? p_->q.erase(it) : it + 1;
    }
    std::unique_lock<std::mutex> lk(c.mu);
    c.cv.wait(lk, [&] { return c.running.load(std::memory_order_relaxed) == 0; });
}

}  // namespace qeccore
