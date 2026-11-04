// plan_arena — the plan cache's slab storage and the flat record table (qeccore/plan_arena.hpp).
#include "qeccore/plan_arena.hpp"
#include "qeccore/record_table.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <unordered_map>

namespace qeccore {

namespace {
inline size_t round64(size_t b) { return (b + 63) & ~size_t(63); }
}  // namespace
#ifdef XTIM_MUTATE_ARENA_STALE_LIST
namespace {
// mutation build only (MUT-ARENA-STALE-LIST): the owner a chunk had when first seen by a release
std::unordered_map<const ArenaChunk*, PlanArena*>& stale_map_() { static std::unordered_map<const ArenaChunk*, PlanArena*> m; return m; }
PlanArena* g_arena_first_owner_(ArenaChunk* c) {
    auto& m = stale_map_();
    auto it = m.find(c);
    if (it == m.end()) it = m.emplace(c, c->owner.load(std::memory_order_acquire)).first;
    return it->second;
}
}  // namespace
#endif

// ── ArenaRoot ─────────────────────────────────────────────────────────────────────────────────────────────────────
ArenaRoot::ArenaRoot() {
    arenas.emplace_back(new PlanArena(this));
    graveyard_ = arenas.back().get();
    graveyard_->graveyard_ = true;
}
ArenaRoot::~ArenaRoot() {
    for (ArenaChunk* c : chunks) ::operator delete(static_cast<void*>(c));
}
PlanArena* ArenaRoot::make_arena() {
    const std::lock_guard<std::mutex> lk(mu);
    arenas.emplace_back(new PlanArena(this));
    return arenas.back().get();
}
size_t ArenaRoot::chunk_bytes() const {
    size_t b = 0;
    for (const ArenaChunk* c : chunks) b += sizeof(ArenaChunk) + c->cap;
    return b;
}

// ── PlanArena ─────────────────────────────────────────────────────────────────────────────────────────────────────
ArenaChunk* PlanArena::new_chunk_(size_t need) {
    size_t cap = std::max(need, next_chunk_);
    next_chunk_ = std::min(next_chunk_ * 2, kArenaChunkMax);
    void* mem = ::operator new(sizeof(ArenaChunk) + cap);        // one mapping above the mmap threshold
    ArenaChunk* c = new (mem) ArenaChunk();
    c->owner.store(this, std::memory_order_release);
    c->cap = cap;
    c->used = 0;
    {
        const std::lock_guard<std::mutex> lk(root_->mu);
        root_->chunks.push_back(c);
    }
    chunks_.push_back(c);
    chunk_bytes_ += cap;
    return c;
}

void* PlanArena::alloc(size_t bytes) {
    if (graveyard_ || retired_) throw std::logic_error("PlanArena::alloc on a retired arena (engine bug)");
    const size_t total = round64(bytes + sizeof(ArenaBlock));
    ArenaBlock* b = nullptr;
    if (total > kArenaMaxPooled) {                                 // a heap block: never pooled
        void* mem = ::operator new(total);
        b = new (mem) ArenaBlock();
        b->chunk = nullptr;
        b->bytes = (uint32_t)total;
        b->live = 1;
        return b->payload();
    }
    const size_t cls = total / 64 - 1;
    if (cls < free_.size() && free_[cls]) {                        // the free list first
        b = free_[cls];
        free_[cls] = b->next;
        free_bytes_ -= b->bytes;
        b->next = nullptr;
    } else {
        if (!cur_ || cur_->cap - cur_->used < total) {
            // a spliced chunk with a tail large enough (the largest), else a new chunk
            ArenaChunk* best = nullptr;
            for (ArenaChunk* c : chunks_)
                if (c != cur_ && c->cap - c->used >= total && (!best || c->cap - c->used > best->cap - best->used)) best = c;
            cur_ = best ? best : new_chunk_(total);
        }
        b = new (cur_->data() + cur_->used) ArenaBlock();
        cur_->used += total;
        b->chunk = cur_;
        b->bytes = (uint32_t)total;
    }
    b->live = 1;
    live_bytes_ += b->bytes;
    return b->payload();
}

void PlanArena::push_deferred_(ArenaBlock* b) {
    ArenaBlock* head = deferred_.load(std::memory_order_relaxed);
    do {
        b->next = head;
    } while (!deferred_.compare_exchange_weak(head, b, std::memory_order_release, std::memory_order_relaxed));
}

void PlanArena::release(void* payload) {
    ArenaBlock* b = ArenaBlock::of(payload);
    if (!b->live) throw std::logic_error("PlanArena::release: a block released twice (engine bug)");
    b->live = 0;
    if (!b->chunk) { ::operator delete(static_cast<void*>(b)); return; }
#ifndef XTIM_MUTATE_ARENA_STALE_LIST
    PlanArena* owner = b->chunk->owner.load(std::memory_order_acquire);   // the CURRENT owner's list
#else
    PlanArena* owner = g_arena_first_owner_(b->chunk);                     // mutation build only: the FIRST owner's list
#endif
    owner->push_deferred_(b);
}

void PlanArena::free_local_(ArenaBlock* b) {
    const size_t cls = b->bytes / 64 - 1;
    if (cls >= free_.size()) free_.resize(cls + 1, nullptr);
    b->next = free_[cls];
    free_[cls] = b;
    free_bytes_ += b->bytes;
    live_bytes_ -= b->bytes;
}

void PlanArena::drain() {
    ArenaBlock* list = deferred_.exchange(nullptr, std::memory_order_acquire);
    while (list) {
        ArenaBlock* b = list;
        list = b->next;
        PlanArena* owner = b->chunk->owner.load(std::memory_order_acquire);
#ifndef XTIM_MUTATE_ARENA_FOREIGN_REUSE
        if (owner == this) free_local_(b);
        else owner->push_deferred_(b);                             // a splice moved its chunk: forward (one hop)
#else
        (void)owner; free_local_(b);                               // mutation build only: a foreign block reused here
#endif
    }
}

void PlanArena::lend_to(PlanArena& lane) {
    if (&lane == this || lane.graveyard_ || lane.retired_ || graveyard_ || retired_) return;
    size_t bi = chunks_.size(), best = 0;
    for (size_t i = 0; i < chunks_.size(); ++i) {
        const size_t tail = chunks_[i]->cap - chunks_[i]->used;
        if (tail > best) { best = tail; bi = i; }
    }
    if (bi == chunks_.size() || best < kArenaLendMin) return;
    ArenaChunk* c = chunks_[bi];
    chunks_[bi] = chunks_.back();
    chunks_.pop_back();
    if (cur_ == c) cur_ = nullptr;
    chunk_bytes_ -= c->cap;
    c->lend_mark = c->used;
    c->owner.store(&lane, std::memory_order_release);
    lane.chunks_.push_back(c);
    lane.cur_ = c;
    lane.chunk_bytes_ += c->cap;
}

void PlanArena::splice_from(PlanArena& lane) {
    if (&lane == this) return;
    for (ArenaChunk* c : lane.chunks_) {
#ifndef XTIM_MUTATE_ARENA_NO_REWRITE
        c->owner.store(this, std::memory_order_release);
#endif                                                         // mutation build only: the owner stays the lane
        c->lend_mark = 0;                                      // (a lent chunk is home again)
        chunks_.push_back(c);
    }
    chunk_bytes_ += lane.chunk_bytes_;
    lane.chunks_.clear();
    lane.cur_ = nullptr;
    lane.chunk_bytes_ = 0;
    lane.next_chunk_ = kArenaChunkMin;                             // F10: the lane restarts at 1 MiB
    if (lane.free_.size() > free_.size()) free_.resize(lane.free_.size(), nullptr);
    for (size_t cls = 0; cls < lane.free_.size(); ++cls) {
        ArenaBlock* b = lane.free_[cls];
        if (!b) continue;
        ArenaBlock* last = b;
        while (last->next) last = last->next;
        last->next = free_[cls];
        free_[cls] = b;
        lane.free_[cls] = nullptr;
    }
    free_bytes_ += lane.free_bytes_;
    live_bytes_ += lane.live_bytes_;
    lane.free_bytes_ = lane.live_bytes_ = 0;
    // the lane's deferred list: every block's chunk is ours now (or another arena's: forwarded)
#ifndef XTIM_MUTATE_ARENA_NO_DRAIN
    ArenaBlock* list = lane.deferred_.exchange(nullptr, std::memory_order_acquire);
#else
    ArenaBlock* list = nullptr;                                    // mutation build only: the lane's list stays
#endif
    while (list) {
        ArenaBlock* b = list;
        list = b->next;
        PlanArena* owner = b->chunk->owner.load(std::memory_order_acquire);
        if (owner == this) free_local_(b);
        else owner->push_deferred_(b);
    }
}

void PlanArena::reset_dropped() {
    for (ArenaChunk* c : chunks_) c->used = c->lend_mark;      // H1: the home's blocks below the lend mark stay
    for (ArenaBlock*& head : free_) {                          // a home block (below its chunk's lend mark) the lane's
        ArenaBlock* b = head;                                  // drain freed locally: back to the deferred list (the
        head = nullptr;                                        // splice that follows frees it on the home's list)
        while (b) {
            ArenaBlock* nx = b->next;
            if (reinterpret_cast<char*>(b) < b->chunk->data() + b->chunk->lend_mark) push_deferred_(b);
            b = nx;
        }
    }
    free_bytes_ = live_bytes_ = 0;
    cur_ = chunks_.empty() ? nullptr : chunks_.front();
    ArenaBlock* list = deferred_.exchange(nullptr, std::memory_order_acquire);
    while (list) {                                                 // stragglers of a moved chunk: forwarded; ours: reset
        ArenaBlock* b = list;                                      // (a home block below a lent chunk's mark: kept)
        list = b->next;
        PlanArena* owner = b->chunk->owner.load(std::memory_order_acquire);
        if (owner != this) owner->push_deferred_(b);
        else if (reinterpret_cast<char*>(b) < b->chunk->data() + b->chunk->lend_mark) push_deferred_(b);
    }
}

void PlanArena::retire() {
    if (retired_) return;
    retired_ = true;
    PlanArena* g = root_->graveyard();
    for (ArenaChunk* c : chunks_) c->owner.store(g, std::memory_order_release);
    chunks_.clear();
    cur_ = nullptr;
    std::fill(free_.begin(), free_.end(), nullptr);
    chunk_bytes_ = free_bytes_ = live_bytes_ = 0;
    ArenaBlock* list = deferred_.exchange(nullptr, std::memory_order_acquire);
    while (list) { ArenaBlock* b = list; list = b->next; g->push_deferred_(b); }
}

void PlanArena::push_deferred_for_test(void* payload) { push_deferred_(ArenaBlock::of(payload)); }
size_t PlanArena::deferred_count() const {
    size_t n = 0;
    for (const ArenaBlock* b = deferred_.load(std::memory_order_acquire); b; b = b->next) ++n;
    return n;
}
bool PlanArena::owns(const void* payload) const {
    const ArenaBlock* b = ArenaBlock::of(const_cast<void*>(payload));
    return b->chunk && b->chunk->owner.load(std::memory_order_acquire) == this;
}

// ── RecBlob ───────────────────────────────────────────────────────────────────────────────────────────────────────
namespace {
struct BlobLayout { size_t rows, piv, born, lam, emit, total, stride; };
BlobLayout layout_(int m, int rho, int kappa, int CWX, int k, int cw, int XW) {
    (void)m;
    BlobLayout L;
    size_t off = round64(sizeof(RecBlob));
    L.rows = off; off += (size_t)(1 + rho + kappa) * (size_t)CWX * 8;
    off = (off + 7) & ~size_t(7);
    L.piv = off; off += (size_t)rho * 4;
    L.born = off; off += (size_t)kappa * 4;
    off = (off + 7) & ~size_t(7);
    L.stride = ((size_t)3 + 2 * (size_t)k + 7) & ~size_t(7);
    L.lam = off; off += (size_t)kappa * L.stride;
    off = (off + 7) & ~size_t(7);
    L.emit = off; off += (size_t)(1 + rho + kappa) * (size_t)(cw + XW) * 8;
    L.total = off;
    return L;
}
}  // namespace

size_t RecBlob::bytes_for(const RecTable& T, int cw, int XW) {
    const int k = T.lam.empty() ? 0 : (int)T.lam[0].a.size();
    return layout_(T.m, T.rho, T.kappa, T.CWX, k, cw, XW).total;
}

RecBlob* RecBlob::place(void* mem, size_t bytes, const RecTable& T, int cw, int XW) {
    const int k = T.lam.empty() ? 0 : (int)T.lam[0].a.size();
    const BlobLayout L = layout_(T.m, T.rho, T.kappa, T.CWX, k, cw, XW);
    if (bytes < L.total) throw std::logic_error("RecBlob::place: block too small (engine bug)");
    std::memset(mem, 0, L.total);
    RecBlob* B = new (mem) RecBlob();
    B->bytes = (uint32_t)bytes;
    B->m = T.m; B->rho = T.rho; B->kappa = T.kappa; B->CWX = T.CWX; B->k = k; B->cw = cw; B->XW = XW;
    B->off_rows = (uint32_t)L.rows; B->off_piv = (uint32_t)L.piv; B->off_born = (uint32_t)L.born;
    B->off_lam = (uint32_t)L.lam; B->off_emit = (uint32_t)L.emit; B->lam_stride = (uint32_t)L.stride;
    char* base = reinterpret_cast<char*>(mem);
    uint64_t* rows = reinterpret_cast<uint64_t*>(base + L.rows);
    const size_t W = (size_t)T.CWX;
    for (size_t w = 0; w < W && w < T.base.size(); ++w) rows[w] = T.base[w];
    for (int s = 0; s < T.rho; ++s)
        for (size_t w = 0; w < W && w < T.coin_row[(size_t)s].size(); ++w) rows[(size_t)(1 + s) * W + w] = T.coin_row[(size_t)s][w];
    for (int j = 0; j < T.kappa; ++j)
        for (size_t w = 0; w < W && w < T.born_row[(size_t)j].size(); ++w) rows[(size_t)(1 + T.rho + j) * W + w] = T.born_row[(size_t)j][w];
    int32_t* piv = reinterpret_cast<int32_t*>(base + L.piv);
    for (int s = 0; s < T.rho; ++s) piv[s] = T.pivot_rec[(size_t)s];
    int32_t* born = reinterpret_cast<int32_t*>(base + L.born);
    for (int j = 0; j < T.kappa; ++j) born[j] = T.born_rec[(size_t)j];
    uint8_t* lam = reinterpret_cast<uint8_t*>(base + L.lam);
    for (int j = 0; j < T.kappa; ++j) {
        const FreeDir& d = T.lam[(size_t)j];
        if ((int)d.a.size() != k || (int)d.b.size() != k)
            throw std::logic_error("RecBlob::place: directions of different widths (engine bug)");
        uint8_t* p = lam + (size_t)j * L.stride;
        p[0] = d.phase; p[1] = d.par; p[2] = d.hermitian;
        std::memcpy(p + 3, d.a.data(), (size_t)k);
        std::memcpy(p + 3 + k, d.b.data(), (size_t)k);
    }
    return B;
}

void RecBlob::to_table(RecTable& T) const {
    T = RecTable();
    T.built = true;
    T.m = m; T.rho = rho; T.kappa = kappa; T.CWX = CWX;
    const size_t W = (size_t)CWX;
    T.base.assign(base(), base() + W);
    T.coin_row.resize((size_t)rho);
    for (int s = 0; s < rho; ++s) T.coin_row[(size_t)s].assign(coin_row(s), coin_row(s) + W);
    T.born_row.resize((size_t)kappa);
    for (int j = 0; j < kappa; ++j) T.born_row[(size_t)j].assign(born_row(j), born_row(j) + W);
    T.pivot_rec.assign(pivot_rec(), pivot_rec() + rho);
    T.born_rec.assign(born_rec(), born_rec() + kappa);
    const LamSpan L = lam();
    T.lam.resize((size_t)kappa);
    for (int j = 0; j < kappa; ++j) T.lam[(size_t)j] = L.at(j).to_freedir();
}

bool RecBlob::same_table(const RecTable& T) const {
    if (!T.built || T.m != m || T.rho != rho || T.kappa != kappa || T.CWX != CWX) return false;
    const size_t W = (size_t)CWX;
    if (T.base.size() != W || std::memcmp(T.base.data(), base(), W * 8) != 0) return false;
    for (int s = 0; s < rho; ++s)
        if (T.coin_row[(size_t)s].size() != W || std::memcmp(T.coin_row[(size_t)s].data(), coin_row(s), W * 8) != 0) return false;
    for (int j = 0; j < kappa; ++j)
        if (T.born_row[(size_t)j].size() != W || std::memcmp(T.born_row[(size_t)j].data(), born_row(j), W * 8) != 0) return false;
    for (int s = 0; s < rho; ++s) if (T.pivot_rec[(size_t)s] != pivot_rec()[s]) return false;
    for (int j = 0; j < kappa; ++j) if (T.born_rec[(size_t)j] != born_rec()[j]) return false;
    if ((int)T.lam.size() != kappa) return false;
    const LamSpan L = lam();
    for (int j = 0; j < kappa; ++j) {
        const FreeDir& d = T.lam[(size_t)j];
        const FreeDirView v = L.at(j);
        if ((int)d.a.size() != v.k || d.phase != v.phase || d.par != v.par || d.hermitian != v.hermitian) return false;
        if (std::memcmp(d.a.data(), v.a, (size_t)v.k) != 0 || std::memcmp(d.b.data(), v.b, (size_t)v.k) != 0) return false;
    }
    return true;
}

// ── RecHandle ─────────────────────────────────────────────────────────────────────────────────────────────────────
RecHandle::RecHandle(RecBlob* b, std::shared_ptr<ArenaRoot> root) : b_(b), root_(std::move(root)) {
    if (b_) b_->refs.fetch_add(1, std::memory_order_acq_rel);
}
RecHandle::RecHandle(const RecHandle& o) : b_(o.b_), root_(o.root_) {
    if (b_) b_->refs.fetch_add(1, std::memory_order_acq_rel);
}
RecHandle& RecHandle::operator=(const RecHandle& o) {
    if (this == &o) return *this;
    RecHandle t(o);
    std::swap(b_, t.b_);
    std::swap(root_, t.root_);
    return *this;
}
RecHandle& RecHandle::operator=(RecHandle&& o) noexcept {
    if (this == &o) return *this;
    reset();
    b_ = o.b_; o.b_ = nullptr;
    root_ = std::move(o.root_);
    return *this;
}
void RecHandle::reset() {
    if (!b_) return;
    RecBlob* b = b_;
    b_ = nullptr;
    if (b->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) PlanArena::release(b);
    root_.reset();
}

RecHandle make_heap_blob(const RecTable& T, int cw, int XW) {
    const size_t bytes = RecBlob::bytes_for(T, cw, XW);
    const size_t total = round64(bytes + sizeof(ArenaBlock));
    void* mem = ::operator new(total);
    ArenaBlock* blk = new (mem) ArenaBlock();
    blk->chunk = nullptr;
    blk->bytes = (uint32_t)total;
    blk->live = 1;
    RecBlob* B = RecBlob::place(blk->payload(), total - sizeof(ArenaBlock), T, cw, XW);
    return RecHandle(B, nullptr);
}

RecHandle make_arena_blob(PlanArena& A, std::shared_ptr<ArenaRoot> root, const RecTable& T, int cw, int XW) {
    const size_t bytes = RecBlob::bytes_for(T, cw, XW);
    void* mem = A.alloc(bytes);
    const ArenaBlock* blk = ArenaBlock::of(mem);
    RecBlob* B = RecBlob::place(mem, (size_t)blk->bytes - sizeof(ArenaBlock), T, cw, XW);
    return RecHandle(B, std::move(root));
}

}  // namespace qeccore
