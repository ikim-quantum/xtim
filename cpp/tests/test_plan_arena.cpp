// test_plan_arena — plancache-design §4 (O5): the slab arena's ownership invariants and the flat record blob.
//
//  * one owner per chunk, written in the chunk header; a block is on at most one free list, its chunk's CURRENT
//    owner's; a release from any thread lands on that owner's deferred list and reaches its free list at the owner's
//    drain — never another arena's;
//  * splice (lane -> home): chunks move whole with their owner rewritten, the lane's free lists move with them, the
//    lane's pending deferred list is drained into the home; a lane handle released AFTER the splice lands on the
//    HOME's list (reuse-after-splice); the lane never hands out a home block afterwards (owner canary);
//  * a straggler on the wrong list (the owner loaded before the rewrite) is forwarded by the drain, never reused there;
//  * a double release is an engine-bug throw; a heap block (above the pooled size, or a reader's blob) is deleted;
//  * RecBlob round-trips a RecTable (same_table / to_table) and lays the emission block out where record_rows_ reads;
//  * handles keep a blob alive across its class's eviction (the blob returns to the owner only at the last release).
// Mutations (scratch builds, each RED here): XTIM_MUTATE_ARENA_STALE_LIST (release to the first owner's list),
// XTIM_MUTATE_ARENA_NO_REWRITE (splice keeps the lane as owner), XTIM_MUTATE_ARENA_NO_DRAIN (splice skips the lane's
// list), XTIM_MUTATE_ARENA_FOREIGN_REUSE (drain frees a foreign block locally).
// Must end main() with REPORT().
#include "check.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "qeccore/plan_arena.hpp"
#include "qeccore/record_table.hpp"

using namespace qeccore;

namespace {

RecTable small_table(int m, int rho, int kappa, int k) {
    RecTable T;
    T.built = true;
    T.m = m; T.rho = rho; T.kappa = kappa; T.CWX = (m + 63) / 64;
    T.base.assign((size_t)T.CWX, 0x5555ULL);
    T.coin_row.resize((size_t)rho);
    for (int s = 0; s < rho; ++s) { T.coin_row[(size_t)s].assign((size_t)T.CWX, 0); T.coin_row[(size_t)s][0] = 1ULL << s; T.pivot_rec.push_back(s); }
    T.born_row.resize((size_t)kappa);
    for (int j = 0; j < kappa; ++j) {
        T.born_row[(size_t)j].assign((size_t)T.CWX, 0); T.born_row[(size_t)j][0] = 1ULL << (rho + j);
        T.born_rec.push_back(rho + j);
        FreeDir d;
        d.a.assign((size_t)k, 0); d.b.assign((size_t)k, 0);
        d.a[(size_t)(j % std::max(1, k))] = 1; d.b[(size_t)((j + 1) % std::max(1, k))] = 1;
        d.phase = (uint8_t)(j & 3); d.par = (uint8_t)(j & 1); d.hermitian = 1;
        T.lam.push_back(d);
    }
    return T;
}

void test_blob_roundtrip() {
    const RecTable T = small_table(70, 3, 2, 5);
    RecHandle h = make_heap_blob(T, 2, 1);
    CHECK(h.use_count() == 1);
    CHECK(h->same_table(T));
    CHECK(h->m == 70 && h->rho == 3 && h->kappa == 2 && h->CWX == 2 && h->k == 5);
    CHECK(h->emit_words() == (size_t)(1 + 3 + 2) * 3);
    RecTable U;
    h->to_table(U);
    CHECK(h->same_table(U));
    CHECK(U.lam.size() == 2 && U.lam[1].a == T.lam[1].a && U.lam[1].b == T.lam[1].b && U.lam[1].phase == T.lam[1].phase);
    const LamSpan L = h->lam();
    CHECK(L.size() == 2 && L.at(1).phase == 1 && L.at(1).par == 1 && L.at(1).k == 5);
    RecTable V = T;
    V.base[0] ^= 1;
    CHECK(!h->same_table(V));
    RecHandle h2 = h;                                        // a second handle: the blob lives until both drop
    CHECK(h.use_count() == 2);
    h.reset();
    CHECK(h2.use_count() == 1 && h2->m == 70);
    h2.reset();
    CHECK(!h2);
}

void test_ownership_and_splice() {
    auto root = std::make_shared<ArenaRoot>();
    PlanArena* home = root->make_arena();
    PlanArena* lane = root->make_arena();
    const RecTable T = small_table(40, 2, 1, 4);
    // the home's own blobs: released -> deferred -> the home's free list at its drain; reused by the next alloc
    RecHandle a = make_arena_blob(*home, root, T, 1, 1);
    const size_t bytes_a = a->bytes;
    CHECK(home->owns(a.get()) && !lane->owns(a.get()));
    CHECK(home->live_bytes() > 0 && home->free_bytes() == 0);
    a.reset();
    CHECK(home->free_bytes() == 0);                          // deferred, not yet on a free list
    CHECK(home->deferred_count() == 1);
    home->drain();
    CHECK(home->free_bytes() > 0 && home->deferred_count() == 0);
    const size_t fb = home->free_bytes();
    RecHandle a2 = make_arena_blob(*home, root, T, 1, 1);    // the same size class: the free block comes back
    CHECK(home->free_bytes() == fb - (a2->bytes + sizeof(ArenaBlock)));
    CHECK(home->owns(a2.get()) && a2->bytes == bytes_a);
    // a lane builds two blobs; one is released BEFORE the splice (pending on the lane's list)
    RecHandle l1 = make_arena_blob(*lane, root, T, 1, 1);
    RecHandle l2 = make_arena_blob(*lane, root, T, 1, 1);
    const void* l1p = l1.get();
    CHECK(lane->owns(l1.get()) && !home->owns(l1.get()) && lane->chunks() == 1);
    l1.reset();
    CHECK(lane->deferred_count() == 1 && lane->free_bytes() == 0);
    const size_t home_free_before = home->free_bytes(), home_chunks = home->chunks();
    home->splice_from(*lane);
    // chunks moved whole, owners rewritten; the pending release drained into the home's free list
    CHECK(lane->chunks() == 0 && home->chunks() == home_chunks + 1);
    CHECK(home->owns(l2.get()) && !lane->owns(l2.get()));
    CHECK(lane->deferred_count() == 0);
    CHECK(home->free_bytes() > home_free_before);
    (void)l1p;
    // reuse-after-splice: a lane handle released AFTER the splice lands on the HOME's list, never the lane's
    l2.reset();
    CHECK(lane->deferred_count() == 0 && home->deferred_count() == 1);
    home->drain();
    CHECK(home->free_bytes() > home_free_before + 0);
    // the lane restarts in a fresh chunk: it never returns a home block (owner canary)
    RecHandle l3 = make_arena_blob(*lane, root, T, 1, 1);
    CHECK(lane->owns(l3.get()) && !home->owns(l3.get()) && lane->chunks() == 1);
    l3.reset();
    lane->drain();                                           // (l3's own block, the lane's: freed there)
    // a straggler: the release of a HOME block that loaded the owner before a rewrite lands on the LANE's list; the
    // lane's drain forwards it to the home (never frees it locally); the home's drain frees it
    home->drain();
    RecHandle s1 = make_arena_blob(*home, root, T, 1, 1);
    const size_t s1_bytes = s1->bytes + sizeof(ArenaBlock);
    RecBlob* s1b = s1.detach();                              // the reference dropped by hand: the block is dead
    const size_t lf = lane->free_bytes(), hf = home->free_bytes();
    lane->push_deferred_for_test(s1b);                       // (the straggler push)
    lane->drain();
    CHECK(lane->free_bytes() == lf);                         // not freed on the lane
    CHECK(home->deferred_count() == 1);                      // forwarded
    home->drain();
    CHECK(home->free_bytes() == hf + s1_bytes);
    // a double release is loud
    RecHandle d1 = make_arena_blob(*home, root, T, 1, 1);
    RecBlob* d1b = d1.detach();
    PlanArena::release(d1b);
    bool threw = false;
    try { PlanArena::release(d1b); } catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
    home->drain();
    // heap blocks: above the pooled size -> operator new / delete, no arena bookkeeping
    RecTable big = small_table(64 * 40000, 1, 0, 1);         // 40000 words per row: above the pooled size
    RecHandle hb = make_arena_blob(*home, root, big, 1, 1);
    CHECK(!home->owns(hb.get()));
    const size_t live = home->live_bytes();
    hb.reset();
    CHECK(home->live_bytes() == live);
    // retire: chunks go to the graveyard; a handle released after the retire does not touch the dead arena's lists
    RecHandle keep = make_arena_blob(*home, root, T, 1, 1);
    home->retire();
    CHECK(home->chunks() == 0 && !home->owns(keep.get()));
    CHECK(root->graveyard()->owns(keep.get()));
    keep.reset();                                            // lands on the graveyard's list (never drained)
    CHECK_EQ((long)root->graveyard()->deferred_count(), 1L);
    a2.reset();
}

// review H1: lending. The home lends a lane the chunk with the largest tail; the lane bumps from it (no chunk of its
// own); the splice returns it; a dropped lane's reset restores the lend mark (the home's blocks below it untouched,
// a home block the lane's drain freed locally goes back through the deferred list); mapped capacity stays bounded
// over many lend / splice rounds.
void test_lending() {
    auto root = std::make_shared<ArenaRoot>();
    PlanArena* home = root->make_arena();
    PlanArena* lane = root->make_arena();
    const RecTable T = small_table(40, 2, 1, 4);
    RecHandle h1 = make_arena_blob(*home, root, T, 1, 1);    // the home's chunk (1 MiB) with a big tail
    const RecBlob* h1p = h1.get();
    const size_t home_chunks = home->chunks(), home_cb = home->chunk_bytes();
    home->lend_to(*lane);
    CHECK(home->chunks() == home_chunks - 1 && lane->chunks() == 1);
    CHECK(home->chunk_bytes() == home_cb - lane->chunk_bytes());
    CHECK(lane->owns(h1p));                                  // the chunk (with the home's block) is the lane's for now
    RecHandle l1 = make_arena_blob(*lane, root, T, 1, 1);    // bumped from the lent tail: no new chunk
    CHECK(lane->chunks() == 1 && lane->owns(l1.get()));
    CHECK(root->chunks.size() == home_chunks);               // nothing mapped
    // a home block released while lent lands on the lane's list; the lane's drain frees it LOCALLY (its chunk's
    // owner is the lane) — a dropped lane must hand it back, a spliced lane moves it with the free lists
    h1.reset();
    CHECK(lane->deferred_count() == 1);
    lane->drain();
    CHECK(lane->free_bytes() > 0);
    // the lane fails: reset_dropped restores the lend mark; the home block on its free list goes to the deferred list
    l1.detach();                                             // (the lane's own block dies with the reset)
    lane->reset_dropped();
    CHECK(lane->free_bytes() == 0 && lane->live_bytes() == 0);
    CHECK(lane->deferred_count() == 1);                      // the home block, kept for the splice
    home->splice_from(*lane);
    CHECK(lane->chunks() == 0 && home->chunks() == home_chunks && home->chunk_bytes() == home_cb);
    CHECK(home->free_bytes() > 0);                           // h1's block is on the home's free list
    CHECK(home->owns(h1p));
    RecHandle h2 = make_arena_blob(*home, root, T, 1, 1);    // (the free block comes back first: h1's)
    CHECK(home->owns(h2.get()) && home->free_bytes() == 0);
    // many rounds: lend, allocate a little, splice — the mapped capacity never grows
    const size_t rc0 = root->chunks.size();
    std::vector<RecHandle> keep;
    for (int round = 0; round < 200; ++round) {
        home->lend_to(*lane);
        CHECK(lane->chunks() == 1);
        keep.push_back(make_arena_blob(*lane, root, T, 1, 1));
        home->splice_from(*lane);
        CHECK(lane->chunks() == 0);
    }
    CHECK(root->chunks.size() == rc0);
    CHECK(home->chunk_bytes() == home_cb);
    keep.clear();
    home->drain();
    // a lane whose lent tail runs out maps a chunk of its own; the splice brings it home as a new tail
    PlanArena* lane2 = root->make_arena();
    PlanArena* home2 = root->make_arena();
    RecTable big = small_table(64 * 4000, 1, 0, 1);          // ~32 KiB blobs
    std::vector<RecHandle> fill;
    fill.push_back(make_arena_blob(*home2, root, big, 1, 1));
    home2->lend_to(*lane2);
    CHECK(lane2->chunks() == 1);
    for (int i = 0; i < 40; ++i) fill.push_back(make_arena_blob(*lane2, root, big, 1, 1));   // > 1 MiB: lane chunks
    const size_t nl = lane2->chunks();
    CHECK(nl >= 2);                                          // the lent chunk, then the lane's own
    home2->splice_from(*lane2);
    CHECK(lane2->chunks() == 0 && home2->chunks() == nl);
    home2->lend_to(*lane2);                                  // the fuller chunk's tail is the largest now
    CHECK(lane2->chunks() == 1);
    home2->splice_from(*lane2);
    fill.clear();
    home2->drain();
    h2.reset();
    home->drain();
}

void test_concurrent_releases() {
    // releases from many threads into one owner's MPSC list: every block arrives exactly once
    auto root = std::make_shared<ArenaRoot>();
    PlanArena* home = root->make_arena();
    const RecTable T = small_table(16, 1, 0, 1);
    std::vector<RecHandle> hs;
    for (int i = 0; i < 2000; ++i) hs.push_back(make_arena_blob(*home, root, T, 1, 0));
    const size_t live = home->live_bytes();
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t)
        th.emplace_back([&, t]() { for (int i = t; i < 2000; i += 8) hs[(size_t)i].reset(); });
    for (auto& x : th) x.join();
    CHECK(home->deferred_count() == 2000);
    home->drain();
    CHECK(home->deferred_count() == 0 && home->live_bytes() == 0 && home->free_bytes() == live);
}

}  // namespace

int main() {
    RUN(test_blob_roundtrip);
    RUN(test_ownership_and_splice);
    RUN(test_lending);
    RUN(test_concurrent_releases);
    REPORT();
}
