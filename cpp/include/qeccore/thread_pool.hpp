#pragma once
// thread_pool — threads (fixplan threads/fixplan.md, owner ruling D4): ONE process-wide pool of worker threads that
// every sampler's `threads=` call shares (E-class inner samplers included). It grows to the largest fan-out asked for
// and never shrinks. A call runs n jobs: job 0 on the CALLING thread, jobs 1..n-1 on workers; the call returns when
// every job has finished or was withdrawn. A job a worker has not started when the caller is done with its own work is
// withdrawn by the caller (it never runs): the caller never waits for a busy pool, so a call progresses even when other
// calls occupy every worker. Several calls may run at once (from different threads); their jobs share one queue.
//
// The job function must tolerate being withdrawn (jobs pull their work from a shared counter, so a withdrawn job's
// work is done by the jobs that run). Exceptions do not cross the pool: a job catches its own.
//
// Fork: a child process has none of the parent's threads; the first call in the child (getpid() changed) abandons the
// parent's pool object (its threads do not exist there) and starts a fresh one.
#include <cstddef>
#include <functional>

namespace qeccore {

// scaling S2: a lane's heap headroom. Work on a pool's lanes retains memory (record contents, the evaluator's content
// structures, the law builds' scratch), and glibc grows a thread's arena by the pages each allocation lacks: one
// mprotect per step, which takes the process's mmap lock EXCLUSIVELY — every thread's page faults wait.
// lane_heap_reserve(bytes) allocates `bytes` in 60 KiB blocks and releases them at once: the lane's arena is grown
// (its protected high-water mark raised) by up to `bytes` in 60 KiB steps, and the lane's next allocations up to that
// mark take no mprotect (a trim after the release lowers the arena's size, never that mark). lane_heap_headroom() =
// one block, at each work item a lane takes (a sampler block: about one block's consumption). The lane's own
// allocation pattern only: no allocator parameter changes (under another allocator a cheap no-op). Each block is
// below glibc's 64 KiB free-consolidation and 128 KiB mmap thresholds.
// plancache-design §4 deleted both for the plan cache's slab arena (which holds the plan headers, classes and
// tables); the RF content structures and the lanes' build scratch still live on glibc, and the stage's speed A/B
// (STATUS 2026-11-02: fid −31% at T=16/48, acc −11/−14% on the factory at T=16/48; profile: grow_heap/mprotect under
// the contents' vector fills) brought them back — fix round 2 (the coordinator's ruling 2b: "restore an equivalent
// of the deleted lanes' heap reserve").
void lane_heap_reserve(size_t bytes);   // (fid-routine B1: the whole request, up to 1 GiB)
void lane_heap_headroom();
// fix round 2 (test-only): the next run()'s i-th task push throws std::bad_alloc once (run() then leaves no queued task)
void thread_pool_fail_push_for_test(int i);

class ThreadPool {
  public:
    // The process-wide pool.
    static ThreadPool& instance();
    // Run job(i) for i in [0, n): i = 0 here, the rest on workers (or withdrawn, see above). n >= 1.
    // `job` must not throw.
    void run(int n, const std::function<void(int)>& job);
    // Workers started so far (test hook).
    int workers() const;
    // Tasks queued and not yet taken (test hook, fix round 2: none may outlive a run() that threw).
    int queued_for_test() const;
    // Grow to n − 1 workers if the OS allows; the workers available (fix F4: a thread the OS refuses —
    // std::system_error — caps the pool where it stands instead of throwing; the caller sizes its fan-out to
    // reserve(n) + 1 and reports it).
    int reserve(int n);

    struct Impl;

  private:
    ThreadPool();
    Impl* p_;
};

}  // namespace qeccore
