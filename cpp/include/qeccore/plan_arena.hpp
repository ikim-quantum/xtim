#pragma once
// plan_arena — plancache-design §3 B1 + §4 (owner-signed rev 4, 2026-11-01): engine-owned slab storage for the plan
// cache and the flat one-blob record table.
//
// §4 PlanArena: one per TwirlPlanCache (the home's and each lane overlay's). Chunks of 1 MiB doubling to 64 MiB, ONE
// operator new each (above glibc's mmap threshold: one mapping, one lock acquisition per chunk, instead of one
// mprotect per page of arena growth under the process mmap lock — the lanes' off-CPU share at T = 48, scaling-diag
// #2). Bump allocation for headers, classes and blobs; a released block goes to a 64 B size-class free list of its
// chunk's CURRENT OWNER and is reused before any bump; blocks never move.
// Ownership invariant (review F7 / N1 / L1): every chunk has exactly ONE owner arena at any time, written in the
// chunk header (std::atomic, acquire / release); a block is on at most one free list, its chunk's current owner's. A
// release from ANY thread (a reader with the GIL released, an RF lane, a Python object's destruction) loads the
// chunk's owner and pushes the block to THAT owner's MPSC deferred-free list; the owner's home thread drains it at
// payload_boundary_ (between shots): no lane ever frees or reuses a home block. Splice (the barrier, no sampler lane
// running): the lane's chunks move whole to the home (owners rewritten with store-release), the lane's free lists
// move with them, then the lane's deferred list is DRAINED into the home's; a straggler that loaded the owner before
// the rewrite lands on the lane's list and is forwarded at the next drain (chunks move only lane -> home: one hop).
// The lane restarts at 1 MiB (F10). Review H1 (2026-11-01): a lane's chunks are LENT back — at every super-block
// start the home lends each lane the chunk with the largest free tail (owner rewritten to the lane, the home's bump
// mark recorded as the chunk's lend mark); the lane bumps from that tail and maps a chunk of its own only when the
// tail is used up, so the mapped capacity is bounded by the live bytes plus one chunk per lane, not by the number of
// threaded calls. The splice returns the lent chunk; a dropped lane's reset restores the lend mark (the home's blocks
// below it are untouched) and the chunk goes home with it. Every arena of a sampler hangs off ONE ArenaRoot, which owns the chunk memory and
// the arena objects: a held table is a RecHandle {shared_ptr<ArenaRoot>, blob, atomic refcount}, so no arena, list
// or chunk dies under a handle (sampler destruction, clear_plans, lane reset: a retired arena's chunks are re-owned by
// the root's graveyard arena, whose list nobody drains; the memory goes with the root).
// Fragmentation (review low): 64 B classes never coalesce; free_bytes() is reported (arena_free_bytes).
// Blocks above kMaxPooled bytes are plain heap blocks (chunk == nullptr: operator delete on release); a blob the
// readers build outside a run (any thread) is a heap block too.
//
// §3 B1 RecBlob: ONE contiguous block per class payload — the canonical record table (the base row, the ρ coin rows,
// the κ Born rows, the pivot / introducer columns, the κ directions packed as bytes) and the emission block the hot
// loop reads — built in a reusable scratch RecTable and copied once (the 5 + 2ρ + 3κ allocations per build and the
// shared_ptr control block are gone). Readers see it through RecBlob's accessors (the same rows the RecTable held).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "qeccore/free_dir.hpp"

namespace qeccore {

struct RecTable;
class PlanArena;

struct ArenaChunk {
    std::atomic<PlanArena*> owner;                   // the ONE owner (plan §4)
    size_t cap = 0;                                  // payload capacity (bytes)
    size_t used = 0;                                 // bump mark (the owner's thread only)
    size_t lend_mark = 0;                            // while lent to a lane: the home's bump mark at the lend (H1)
    char* data() { return reinterpret_cast<char*>(this) + sizeof(ArenaChunk); }
};

// Every arena allocation: this header (32 B), then the payload; `bytes` (header + payload) is a multiple of 64.
struct ArenaBlock {
    ArenaChunk* chunk = nullptr;                     // nullptr: a heap block (operator new / delete)
    size_t bytes = 0;                                // CAPS-2 A8: size_t (a heap block can pass 4 GiB)
    uint32_t live = 0;                               // 1 while allocated (double-release = engine-bug throw)
    uint32_t pad_ = 0;
    ArenaBlock* next = nullptr;                      // free-list / deferred-list link
    void* payload() { return reinterpret_cast<char*>(this) + sizeof(ArenaBlock); }
    static ArenaBlock* of(void* payload) { return reinterpret_cast<ArenaBlock*>(static_cast<char*>(payload) - sizeof(ArenaBlock)); }
};
static_assert(sizeof(ArenaBlock) == 32, "ArenaBlock is 32 bytes");

inline constexpr size_t kArenaChunkMin = size_t(1) << 20;      // 1 MiB
inline constexpr size_t kArenaChunkMax = size_t(64) << 20;     // 64 MiB
inline constexpr size_t kArenaMaxPooled = size_t(256) << 10;   // blocks above this are heap blocks (not pooled)
inline constexpr size_t kArenaLendMin = size_t(16) << 10;      // H1: the smallest tail worth lending to a lane

struct ArenaRoot;

class PlanArena {
  public:
    // Allocate `bytes` (payload): the smallest 64 B class that fits, from the free list first, else bump (a new chunk
    // when the current one is full). Above kArenaMaxPooled: a heap block. The owner's thread only.
    void* alloc(size_t bytes);
    // Release a block from ANY thread: a heap block is deleted; an arena block goes to its chunk's current owner's
    // deferred list (never reused before that owner's drain).
    static void release(void* payload);
    // The owner's thread (between shots): the deferred list to the free lists; a block whose chunk this arena no
    // longer owns (a splice moved it) is forwarded to the current owner (one hop).
    void drain();
    // The barrier (no lane running): the lane's chunks become this arena's (owners rewritten, store-release), its free
    // lists join this arena's, then its deferred list is drained here; the lane restarts empty at 1 MiB.
    void splice_from(PlanArena& lane);
    // H1: lend `lane` this arena's chunk with the largest free tail (>= kArenaLendMin): the chunk leaves this arena
    // (owner rewritten to the lane, its lend mark = this arena's bump mark) and becomes the lane's current chunk; the
    // lane bumps from the tail. The barrier (no lane running). Nothing happens when no tail qualifies.
    void lend_to(PlanArena& lane);
    // A lane whose super-block was dropped (a failed call): every block the lane allocated is dropped (a lent chunk's
    // bump mark returns to its lend mark: the home's blocks below it are untouched; a home block the lane's drain put
    // on its free list goes back to the deferred list), the chunks stay with the lane for the splice that follows —
    // nothing it allocated was ever handed to a reader (the cache checks that before calling this).
    void reset_dropped();
    // The cache that owned this arena died: its chunks go to the root's graveyard (releases land on a list nobody
    // drains; the memory lives with the root). The arena object stays in the root (a release may still address it).
    void retire();
    size_t chunk_bytes() const { return chunk_bytes_; }        // mapped chunk capacity (this arena's chunks)
    size_t free_bytes() const { return free_bytes_; }          // blocks on the free lists (fragmentation gauge)
    size_t live_bytes() const { return live_bytes_; }          // allocated blocks (headers + classes + blobs)
    size_t chunks() const { return chunks_.size(); }
    bool owns(const void* payload) const;                      // test: the block's chunk's owner is this arena
    // TEST seam (ctest only): a straggler — a block pushed on THIS arena's deferred list although its chunk's owner
    // may be another arena (the release that loaded the owner before a splice's rewrite); drain must forward it.
    void push_deferred_for_test(void* payload);
    size_t deferred_count() const;                             // test: blocks on the deferred list (O(n))
    ArenaRoot* root() const { return root_; }

  private:
    friend struct ArenaRoot;
    explicit PlanArena(ArenaRoot* root) : root_(root) {}
    ArenaRoot* root_;
    bool graveyard_ = false, retired_ = false;
    ArenaChunk* cur_ = nullptr;
    std::vector<ArenaChunk*> chunks_;                // owned chunks (owner == this)
    std::vector<ArenaBlock*> free_;                  // size class (bytes / 64 − 1) → list head
    std::atomic<ArenaBlock*> deferred_{nullptr};     // MPSC: pushed from any thread, popped by the owner
    size_t next_chunk_ = kArenaChunkMin;
    size_t chunk_bytes_ = 0, free_bytes_ = 0, live_bytes_ = 0;
    void push_deferred_(ArenaBlock* b);
    void free_local_(ArenaBlock* b);                 // onto this arena's free list (this arena owns b's chunk)
    ArenaChunk* new_chunk_(size_t need);
};

// The root of a sampler's arenas: owns every chunk (freed with the root) and every arena object; the handles hold it.
struct ArenaRoot : std::enable_shared_from_this<ArenaRoot> {
    ArenaRoot();
    ~ArenaRoot();
    ArenaRoot(const ArenaRoot&) = delete;
    ArenaRoot& operator=(const ArenaRoot&) = delete;
    PlanArena* make_arena();                         // a new arena of this root (the caller never deletes it)
    PlanArena* graveyard() const { return graveyard_; }
    std::mutex mu;                                   // the chunk and arena vectors (chunks are made on lanes too)
    std::vector<ArenaChunk*> chunks;
    std::vector<std::unique_ptr<PlanArena>> arenas;
    size_t chunk_bytes() const;
  private:
    PlanArena* graveyard_ = nullptr;
};

// ── §3 B1: the flat record table ──────────────────────────────────────────────────────────────────────────────────
// One block: this header, then (64 B aligned) the rows [base | coin 0..ρ−1 | born 0..κ−1] (CWX words each), the pivot
// columns (ρ int32), the introducer columns (κ int32), the κ directions (lam_stride bytes each: phase, par, hermitian,
// a[k], b[k]) and the emission block [base | coin | born] (cw channel words each) then the extras words (XW each) —
// exactly the layout record_rows_ reads.
struct RecBlob {
    std::atomic<uint32_t> refs{0};                   // handles (RecHandle) + the class's own
    int32_t m = 0, rho = 0, kappa = 0, CWX = 0;
    int32_t k = 0;                                   // free coordinates of a direction
    int32_t cw = 0, XW = 0;                          // emission words: channel words per row, extras words per row
    // CAPS-2 A8: the size and the byte offsets are size_t (a blob can pass 4 GiB: (1+ρ+κ)·(CWX+cw+XW)·8 B; u32 fields
    // truncated silently -> out-of-bounds reads / writes). The header stays 64 B: the rows start at kOffRows, the
    // introducer columns right after the ρ pivot columns, and the direction stride is a function of k (lam_stride_of).
    size_t bytes = 0;                                // this blob's block payload bytes (the accounting unit)
    size_t off_piv = 0, off_lam = 0, off_emit = 0;   // byte offsets from `this`
    static constexpr size_t kOffRows = 64;
    static size_t lam_stride_of(int k) { return ((size_t)3 + 2 * (size_t)k + 7) & ~size_t(7); }
    const char* base_() const { return reinterpret_cast<const char*>(this); }
    const uint64_t* rows() const { return reinterpret_cast<const uint64_t*>(base_() + kOffRows); }
    const uint64_t* base() const { return rows(); }
    const uint64_t* coin_row(int s) const { return rows() + (size_t)(1 + s) * (size_t)CWX; }
    const uint64_t* born_row(int j) const { return rows() + (size_t)(1 + rho + j) * (size_t)CWX; }
    const int32_t* pivot_rec() const { return reinterpret_cast<const int32_t*>(base_() + off_piv); }
    const int32_t* born_rec() const { return pivot_rec() + rho; }
    LamSpan lam() const {
        LamSpan L;
        L.packed = reinterpret_cast<const uint8_t*>(base_() + off_lam);
        L.stride = lam_stride_of(k); L.k = k; L.n = kappa;
        return L;
    }
    const uint64_t* emit() const { return reinterpret_cast<const uint64_t*>(base_() + off_emit); }
    uint64_t* emit_mut() { return reinterpret_cast<uint64_t*>(const_cast<char*>(base_()) + off_emit); }
    size_t emit_words() const { return (size_t)(1 + rho + kappa) * (size_t)(cw + XW); }
    // The table as a RecTable (a copy: the cold readers / the test API).
    void to_table(RecTable& T) const;
    // The bytes a blob of this table needs (the block payload); the emission block is (1+ρ+κ)(cw + XW) words.
    static size_t bytes_for(const RecTable& T, int cw, int XW);
    // Lay a table out into `mem` (bytes_for bytes): the rows, columns, directions; the emission block is left zero for
    // the caller to fill (emit_mut). refs = 0.
    static RecBlob* place(void* mem, size_t bytes, const RecTable& T, int cw, int XW);
    // Equal canonical fields (O1's comparator): rows, pivots, introducers, directions.
    bool same_table(const RecTable& T) const;
};
static_assert(sizeof(RecBlob) == RecBlob::kOffRows, "RecBlob's header is 64 bytes (the rows start right after it)");

// A held blob: the blob's refcount and, for an arena blob, the root (its chunks and arenas outlive the handle).
class RecHandle {
  public:
    RecHandle() = default;
    RecHandle(RecBlob* b, std::shared_ptr<ArenaRoot> root);   // adopts one reference (refs incremented here)
    RecHandle(const RecHandle& o);
    RecHandle(RecHandle&& o) noexcept : b_(o.b_), root_(std::move(o.root_)) { o.b_ = nullptr; }
    RecHandle& operator=(const RecHandle& o);
    RecHandle& operator=(RecHandle&& o) noexcept;
    ~RecHandle() { reset(); }
    void reset();
    explicit operator bool() const { return b_ != nullptr; }
    const RecBlob* get() const { return b_; }
    const RecBlob& operator*() const { return *b_; }
    const RecBlob* operator->() const { return b_; }
    RecBlob* mut() const { return b_; }                        // the builder's write access (the owning thread)
    // TEST seam: give up this handle's reference WITHOUT releasing the blob (the test then releases it by hand through
    // PlanArena::release or a deferred-list seam: the straggler scenarios).
    RecBlob* detach() { RecBlob* b = b_; b_ = nullptr; root_.reset(); if (b) b->refs.fetch_sub(1, std::memory_order_acq_rel); return b; }
    uint32_t use_count() const { return b_ ? b_->refs.load(std::memory_order_acquire) : 0; }
  private:
    RecBlob* b_ = nullptr;
    std::shared_ptr<ArenaRoot> root_;
};

// A blob for the readers' memo (any thread): a heap block, no arena.
RecHandle make_heap_blob(const RecTable& T, int cw, int XW);
// A blob in an arena (the owner's thread).
RecHandle make_arena_blob(PlanArena& A, std::shared_ptr<ArenaRoot> root, const RecTable& T, int cw, int XW);

}  // namespace qeccore
