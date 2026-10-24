// twirl_sampler_fam — J-a: the instances of the twirl shot loop, TwirlRecordSampler::Impl::run_w<8> and run_w<0>
// (body: twirl_sampler_runw.inc.hpp), in their own translation unit so twirl_sampler.cpp's code generation (its
// inlining budget) never sees the shot loop. J-b1 F made this family (FAM) sampler THE production sampler; J-b2
// deleted the legacy instances (J3) and the FAM template parameter (J9). N1: the family sampler's setup (extras'
// references, readout-flip masks, the named refusals), its run guard, its semantic oracle over the family extras,
// and its test-API surface.

#include "twirl_sampler_impl.inc.hpp"
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/branch_scaffold.hpp"
#include "qeccore/record_general.hpp"
#include "qeccore/record_signs.hpp"
#include "qeccore/term_residual.hpp"
#include "qeccore/record_space_b.hpp"   // J-b1 P6
#include "twirl_law_internal.hpp"   // J-b1 (fixplan-coldstart): ShotLawCore
#include "qeccore/record_fidelity.hpp"   // J-b1 P7
#include "qeccore/engine_version.hpp"    // J-b1 P11a: TWPL v6
#include "qeccore/plan_cache.hpp"        // J-b1 P11a: TWPL v6
#include "qeccore/state_budget.hpp"      // J-b1 F: C0̂ capacity refusal
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include <new>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "twirl_sampler_runw.inc.hpp"

namespace qeccore {
// The run_w instances (J-b2 J3: the only ones; the legacy run_w<*, false> is no longer instantiated).
template bool TwirlRecordSampler::Impl::run_w<8>(long);
template bool TwirlRecordSampler::Impl::run_w<0>(long);

// ── J-a (N2 / N4a): the FAM route state behind FamTermHold (one pointer in Impl) ───────────────────
// J-b1 P9b: one registered carried branch of a consumer (family_add_branch; E_min = ∅).
struct TwirlRecordSampler::Impl::FamBranch {
    int id = 0;
    FramedSuperposition st{0};                  // the branch's consumer bare: branch 0's frame, its own α
    std::unique_ptr<BornTrie> trie;             // over `st` (stable address: the branch is heap-owned)
    // the sparse (plan, branch) → trie map: one Born trie per plan header (permanent), reset when the plan's
    // record table is rebuilt (a pure function of the plan: same directions)
    std::unordered_map<const CachedPlan*, std::pair<const RecTable*, std::vector<BornNode>>> tries;
    Pauli dspec;                                // the eps delta on wires no consumer gate touches (the shot's P)
    std::vector<uint8_t> dport;                 // the eps delta on the port wires (a frames-class row)
    long shots = 0;
    // D6: the payload (st, trie, tries) is evictable; the header above is permanent
    bool resident = true;
    std::function<FramedSuperposition()> resupply;   // the pure rebuild add_branch validated (nullptr: never evicted)
    uint64_t last_use = 0;
    size_t bytes_() const {
        size_t b = (size_t)st.chi() * (sizeof(AmpKey) + sizeof(std::complex<double>)) +
                   (size_t)st.n() * (size_t)st.n() / 2 + 256;
        for (const auto& kv : tries) b += kv.second.second.size() * sizeof(BornNode) + 64;
        return b;
    }
};

struct TwirlRecordSampler::Impl::FamRoute {
    // The term view is built only when the term route is ARMED, at construction: the ppr_records class on a
    // CH/PPR circuit, the frames class with an input port, or the all-term recording option (J-b1 P4,
    // fam_record_route_term). A TermClassRefusal then propagates to fam_setup_. J-b1 P4 (R9a / R9b): no run path
    // builds it lazily — a recording run serves its diagonal shots on the (D) paths — so an unarmed sampler keeps
    // no construction inputs. The test API arms it on demand from ITS OWN copies (family_arm_terms).
    // The ctor and term_() are COLD and out of line (noinline, below): inlined, their copy / try-catch bodies spent
    // this TU's inline-unit-growth budget and moved run_w<8>'s code layout (bt81 FAM() +3% per shot).
    FamRoute() = default;
    std::optional<TermResidual> tr;                 // N2: the term view, per-alt (P_f, S_f), KerPhi, kind, plan ids
    // The armed term view (an unarmed one is an engine bug on a run path; the test API arms first).
    TermResidual& term_();
    bool rec_term = false;                          // J-b1 P4: fam_record_route_term (the all-term oracle route)
    bool armed_static = false;                      // J-b1 P5: the route is armed whatever the axes (fam_setup_)
    std::vector<int> port_qubits;                   // J-b1 P5: the frames class's port (fam_terms_setup_)
    std::vector<char> live;                         // J-b1 P5: alternative a has p > 0 (the rule's fault sites)
    TwirlRecordSampler::FamilyRefusalInfo refusal;  // J-b1 P13: the structured NONCOMMUTING_MAGIC refusal
    bool rec_d = false;                             // J-b1 P4: this run records its diagonal shots on (D)
    std::vector<int> d_rh;                          // J-b1 P4: (D) plan id -> layout-3 content handle (cold fill)
    // ── N4a: the kind-2 route (ppr_records class) ──
    // A kind-2 plan: its header (permanent; J-b1 P11a: OWNED by the plan cache's arena, never freed or moved while
    // the cache lives) and payload — the (G) record table and the emission block in CachedPlan's own fields, so
    // record_rows_ serves it unchanged; the payload joins the cache's FIFO (install_arena_payload) and is evicted
    // by it.
    struct K2Plan {
        CachedPlan* plan = nullptr;                 // J-b1 P11a: the header, OWNED by the plan cache's arena
        std::vector<uint64_t> S;                    // the reduced term set: the payload's build input
        bool k0general = false;                     // N4-D1: a kind-0 entry outside the pair universe (counters)
        size_t ix = 0;                              // termroute-a: its slot in k2plans (O(1) removal on eviction)
#ifdef XTIM_MUTATE_K2_OWN
        std::unique_ptr<CachedPlan> own;            // mutation build only (MUT-K2-OWN): the old sampler-owned header
#endif
    };
    // The dispatch entry of one reduced term set (cold: created on first visit, never removed).
    struct Entry {
        int kind = 2;
        long plan_id = -1;
        K2Plan* k2 = nullptr;                       // kind 2, or kind 0 outside the pair universe (N4-D1)
        const CachedPlan* k0 = nullptr;             // kind 0: the diagonal cache's header (permanent)
        std::vector<uint64_t> k0key;                // kind 0: its cache key (KW words) and hash
        uint64_t k0h = 0;
        DiagNormalForm nf0;                         // kind 0: the canonical normal form of rot(S) alone (P = I)
        int rh = -1;                                // N5: its layout-3 content handle (cold, first recorded shot)
        size_t bytes = 0;                           // termroute-a: charged to route_bytes
    };
    std::unordered_map<std::string, Entry> route;   // reduced S (bytes) → entry
    std::vector<std::unique_ptr<K2Plan>> k2plans;
    bool ready = false;                             // per-alt table + scratch sized (fam_run_guard_)
    TermShot sh;                                    // the shot's (P, S): composed, then reduced in place
    Pauli P0;                                       // the composed (P, S), for the term-view oracle
    std::vector<uint64_t> S0;
    std::string kbuf;                               // the lookup key buffer (reused: no per-shot allocation)
    Pauli pre;                                      // kind 0: the shot's prefix nf0.prefix ⊕ P (reused buffer)
    uint64_t term_shot = UINT64_MAX;                // cumulative index (shot_) of the last term-routed shot; none yet
    FamilyRouteStats stats;
    // ── N4b: the frames class ──
    bool frames = false;                            // input frames composed by the term route
    int pnq = 0;                                    // port wires; frame slot s2: X_{s2} (< pnq), Z_{s2 - pnq}
    std::vector<uint64_t> fpx, fpz, fs;             // per slot: the frame atom (P_F, S_F) (port_atoms)
    const uint8_t* frow = nullptr;                  // the shot's frame row (fam_frame_take_)
    bool framed = false;                            // the shot supplies a frame component
    // ── N5: retention records (layout 3) ──
    TwirlRecordSampler::FamilyRecordSink* rs = nullptr;   // non-null: a recording run (every shot term-routed)
    bool ff = false;                                // this run folds an input_pauli frame into P_rec
    std::vector<TwirlRecordSampler::FamilyRecordContent> contents;
    std::unordered_map<std::string, int> content_ix;
    // J-b1 P7 (m10): the kind-2 (G) tables of the readers live in the readers' memo (Impl::rec_memo_, tagged key,
    // charged to reader_bytes_, LRU-evictable, under rec_memo_mu_); the content headers are charged there too
    // (permanent, like plan headers: a record holds its handle).
    std::atomic<size_t>* reader_bytes = nullptr;
    std::vector<uint8_t> rec_inv;                   // G-8d: the measurement records' inverts and readout-flip
    std::vector<double> rec_flip;                   //       probabilities (the identity tuple's folds)
    // The record table of a content handle (materialize / signs; definition below the content parsers).
    std::shared_ptr<const RecTable> content_table_(const Impl& I, int h, DiagNormalForm* nf_out,
                                                   std::vector<uint64_t>* S_out);
    static std::shared_ptr<const RecTable> cached_rec_(const Impl& I, const DiagNormalForm& nf);
    // The content handle of a (kind, canonical content key) (cold: once per entry).
    int content_handle_(int kind, const std::string& key) {
        std::string tk = tagged_plan_key(kind, key);   // m3: (kind, key), so equal bytes under two kinds are two
        auto it = content_ix.find(tk);                 //     contents
        if (it != content_ix.end()) return it->second;
        TwirlRecordSampler::FamilyRecordContent c;
        c.kind = kind;
        c.content = key;
        const Digest128 d = plan_key_digest128_(tagged_plan_key(kind, key));
        c.digest_hi = d.hi;
        c.digest_lo = d.lo;
        const size_t charge = sizeof(c) + c.content.capacity() + tk.capacity() + 64;   // m10: charged, permanent
        contents.push_back(std::move(c));
        content_ix.emplace(std::move(tk), (int)contents.size() - 1);
        if (reader_bytes) reader_bytes->fetch_add(charge, std::memory_order_relaxed);
        return (int)contents.size() - 1;
    }

    void fr_setup(const std::vector<std::pair<int, PauliBasis>>& comps) {
        tr->port_atoms(comps, fpx, fpz, fs);
        pnq = (int)comps.size() / 2;
        frames = true;
    }
    // R ← R_f(slot) · R: one frame atom composed onto the shot.
    void compose_frame_(int s2) {
        const size_t NW = sh.P.x.size(), TW = (size_t)tr->term_words();
#ifdef XTIM_MUTATE_FR_POISON
        static const std::vector<uint64_t> zero(4096, 0);   // mutation build only (MUT-FR-POISON): S_F dropped
        tr->compose_atom_(fpx.data() + (size_t)s2 * NW, fpz.data() + (size_t)s2 * NW, zero.data(), sh);
#else
        tr->compose_atom_(fpx.data() + (size_t)s2 * NW, fpz.data() + (size_t)s2 * NW, fs.data() + (size_t)s2 * TW, sh);
#endif
    }

    // The (G) payload of a kind-2 plan (cold; again after an eviction): build_record_table_general over the
    // sampler's record columns, then the emission block exactly as build_rec_cold_ lays it out (channel words
    // with the base row XORed with the classify signs, then the extras words), rejoined and compared.
    void build_k2(Impl& I, K2Plan& k) {
        auto T = std::make_shared<RecTable>();
#ifdef XTIM_MUTATE_F2_STATIC
        // mutation build only (MUT-F2-STATIC): the prototype's pointer-keyed memo (a new sampler at a reused address
        // inherits the dead one's columns)
        static std::unordered_map<const void*, GeneralColumnMemo> stale;
        build_record_table_general(I.G, I.bare, *tr, k.S, I.rec_cols_, *T, nullptr, &stale[this]);
#else
        build_record_table_general(I.G, I.bare, *tr, k.S, I.rec_cols_, *T, nullptr, &colmemo);
#endif
        CachedPlan& P = *k.plan;
        const int nr = 1 + T->rho + T->kappa, cw = std::max(I.CW, 1), XW = I.XW;
        std::vector<uint64_t>& E = P.emit;
        E.assign((size_t)nr * (size_t)(cw + XW), 0);
        std::vector<uint64_t> ch, exw;
        auto put = [&](int r, const std::vector<uint64_t>& row, const uint64_t* chxor) {
            I.split_row_(row, ch, exw, chxor);
            std::copy(ch.begin(), ch.begin() + cw, E.begin() + (size_t)r * cw);
            std::copy(exw.begin(), exw.begin() + XW, E.begin() + (size_t)nr * cw + (size_t)r * XW);
        };
        put(0, T->base, I.refmask_ch_.data());
        for (int c = 0; c < T->rho; ++c) put(1 + c, T->coin_row[(size_t)c], nullptr);
        for (int j = 0; j < T->kappa; ++j) put(1 + T->rho + j, T->born_row[(size_t)j], nullptr);
        auto rejoin_ok = [&](int r, const std::vector<uint64_t>& row, const uint64_t* chxor) {
            return I.rejoin_ok_(E, nr, cw, T->m, r, row, chxor);
        };
        bool ok = T->m == (int)I.rec_cols_.size() && rejoin_ok(0, T->base, I.refmask_ch_.data());
        for (int c = 0; ok && c < T->rho; ++c) ok = rejoin_ok(1 + c, T->coin_row[(size_t)c], nullptr);
        for (int j = 0; ok && j < T->kappa; ++j) ok = rejoin_ok(1 + T->rho + j, T->born_row[(size_t)j], nullptr);
        if (!ok) throw std::logic_error("kind-2 route: the emission block differs from the (G) table (engine bug)");
        if (P.built && T->kappa != P.kappa)
            throw std::logic_error("kind-2 route: a payload rebuild disagrees with the first build (engine bug)");
        if (k.k0general) ++(P.built ? stats.payload_builds_kind0_general : stats.builds_kind0_general);
        else ++(P.built ? stats.payload_builds_kind2 : stats.builds_kind2);
        P.built = true;
        P.kappa = T->kappa;
        P.emit_rho = T->rho;
        P.emit_kappa = T->kappa;
        P.rec = std::move(T);
        P.rec_trie.clear();
        P.emit_built = true;
        I.cache->install_arena_payload(P);         // J-b1 P11a: the cache owns the header and the FIFO pointer
    }
    // J-b1 P11a: the entry's header — the plan cache's arena header of (tag, S) (a TWPL v6 load may have created
    // it, keys only: the payload is built on first use, as for a diagonal header).
    static void attach_header(Impl& I, K2Plan& k) {
#ifdef XTIM_MUTATE_K2_OWN
        k.own.reset(new CachedPlan());              // mutation build only (MUT-K2-OWN): sampler-owned, FIFO-adopted
        k.plan = k.own.get();
        (void)I;
#else
        k.plan = I.cache->arena_header(k.k0general ? TwirlPlanCache::kArenaRoute1 : TwirlPlanCache::kArenaKind2,
                                       k.S.data(), (int)k.S.size());
#endif
    }
    // J-b1 P10: the noiseless reference records of a port circuit with a Born observable (family_port_born_ref).
    // LAST, so no hot field's offset moves (run_w<*> codegen unchanged).
    std::vector<uint8_t> port_ref;
    std::unordered_map<int, std::vector<Pauli>> pivot_w;   // J-b1 P9b: content handle → W_p (family_pivot_frames)
    // ── J-b1 P9b: the consumer branch law (E_min = ∅ branches; family_add_branch). LAST (offsets unchanged). ──
    std::vector<std::unique_ptr<FamBranch>> branches;
    const int32_t* branch_ids = nullptr;            // per shot of the next run (nullptr: every shot is branch 0)
    long branch_n = 0;
    FamBranch* cur_branch = nullptr;                // the current shot's branch (nullptr: branch 0)
    std::vector<uint8_t> brow;                      // the branch shot's frame row (user frame ⊕ dport)
    long branch_probes = 0;                         // branch-map probes (branch ≠ 0 shots only)
    size_t branch_budget = SIZE_MAX;                // D6: resident branch payload bytes allowed
    long branch_evictions = 0, branch_resupplies = 0;
    uint64_t branch_tick = 0;
    // termroute-a F2: build_k2's per-column memo — a constant of (I.G, I.bare, tr, I.rec_cols_), all fixed for this
    // object's life, so it lives here (never keyed on an address). LAST (offsets unchanged).
    GeneralColumnMemo colmemo;
    // termroute-a cache bound (OD-2026-10-23-cache-bound): the dispatch entries (with their K2Plan and kind_of_ cache
    // entry) are charged here and dropped oldest-first under the sampler's plan budget, at a new key (no entry is
    // referenced then). Every value is a pure function of the key (the arena header, the plan id and the content
    // handle are permanent and found again), so a re-created entry serves identical bytes. Kind-0 entries outside
    // the pair universe (route 1, rare) stay: their plan counters are per entry.
    std::deque<const std::string*> route_fifo;
    size_t route_bytes = 0;
    long route_evictions = 0;
    K2Plan* k2_new_() {
        k2plans.emplace_back(new K2Plan());
        k2plans.back()->ix = k2plans.size() - 1;
        return k2plans.back().get();
    }
    size_t entry_bytes_(const std::string& key, const Entry& e) const {
        size_t b = key.capacity() + sizeof(Entry) + 2 * 64 + sizeof(void*) + e.k0key.capacity() * 8 + e.nf0.a.capacity() +
                   e.nf0.cz.capacity() * sizeof(std::pair<int, int>) +
                   (e.nf0.prefix.x.capacity() + e.nf0.prefix.z.capacity()) * 8 + tr->kind_bytes_(key);
        if (e.k2) b += sizeof(K2Plan) + sizeof(void*) + e.k2->S.capacity() * 8;
        return b;
    }
    void recharge_(const std::string& key, Entry& e) {
        route_bytes -= e.bytes;
        e.bytes = entry_bytes_(key, e);
        route_bytes += e.bytes;
    }
    void route_trim_(size_t budget) {
        while (route_bytes > budget && !route_fifo.empty()) {
            auto it = route.find(*route_fifo.front());
            route_fifo.pop_front();
#ifndef XTIM_MUTATE_CB_NOUNCHARGE
            route_bytes -= it->second.bytes;
#endif      // mutation build only (MUT-CB-NOUNCHARGE): the evicted entry stays charged
            tr->kind_forget_(it->first);
            if (K2Plan* k = it->second.k2) {
                const size_t i = k->ix;
                std::swap(k2plans[i], k2plans.back());
                k2plans[i]->ix = i;
                k2plans.pop_back();
            }
            route.erase(it);
            ++route_evictions;
        }
    }
};

__attribute__((noinline, cold)) TermResidual& TwirlRecordSampler::Impl::FamRoute::term_() {
    if (!tr) throw std::logic_error("term route: the term view of this family sampler is not armed (engine bug on a "
                                    "run path; test-API accessors arm it first)");
    return *tr;
}
// J-b1 P4: the test API's on-demand arming, from the caller's copy of the construction inputs (never a run path).
__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::fam_arm_terms_(
        const Circuit& deferred, const std::vector<std::pair<int, int>>& reads) {
    FamRoute* F = fam_term_.p;
    if (!F) throw std::logic_error("family_arm_terms: not a family sampler (J-a: test API only)");
    if (!F->tr) F->tr.emplace(deferred, reads, sides, F->live);   // the construction's merge; a refusal propagates
    cache->set_term_token(term_set_token_(*F->tr));   // J-b1 P11a: the armed term view's TWPL v6 token
}
void TwirlRecordSampler::family_arm_terms(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads) {
    impl_->fam_arm_terms_(deferred, reads);
}

// ── J-b1 P11a: the kind-2 / route-1 header arena of a FAM plan cache (r2.4 / r3.6) ──────────────────────────
// One allocation per header: CachedPlan, then its key words inline (CachedPlan::key()), so a header and its key are
// freed together by arena_free_ (the cache's out-of-line destructor). Index: tag ‖ key bytes → header. Creation
// order is TWPL v6's save order.
struct TwirlPlanCache::K2Arena {
    struct Rec { CachedPlan* p; uint8_t tag; int words; };
    std::vector<Rec> recs;
    std::unordered_map<std::string, size_t> ix;          // tag ‖ key bytes → recs index
    std::unordered_map<const CachedPlan*, size_t> own;   // header → recs index (install_arena_payload's check)
    static std::string key_(uint8_t tag, const uint64_t* key, int words) {
        std::string k(1, (char)tag);
        k.append(reinterpret_cast<const char*>(key), (size_t)words * sizeof(uint64_t));
        return k;
    }
};
void TwirlPlanCache::arena_free_(K2Arena* a) {
    for (const K2Arena::Rec& r : a->recs) {
        r.p->~CachedPlan();
        ::operator delete(r.p);
    }
    delete a;
}
__attribute__((noinline, cold)) CachedPlan* TwirlPlanCache::arena_header(uint8_t tag, const uint64_t* key, int words) {
    if (!k2_) k2_ = new K2Arena();
    std::string k = K2Arena::key_(tag, key, words);
    auto it = k2_->ix.find(k);
    if (it != k2_->ix.end()) return k2_->recs[it->second].p;
    void* mem = ::operator new(sizeof(CachedPlan) + (size_t)words * sizeof(uint64_t));
    CachedPlan* p = new (mem) CachedPlan();
    if (words) std::memcpy(p->key_mut(), key, (size_t)words * sizeof(uint64_t));
    k2_->recs.push_back({p, tag, words});
    p->id = (uint32_t)k2_->recs.size();               // arena-local (never a diagonal plan id; never exported)
    k2_->ix.emplace(std::move(k), k2_->recs.size() - 1);
    k2_->own.emplace(p, k2_->recs.size() - 1);
    return p;
}
void TwirlPlanCache::install_arena_payload(CachedPlan& p) {
#ifndef XTIM_MUTATE_K2_OWN
    if (!k2_ || !k2_->own.count(&p))
        throw std::logic_error("TwirlPlanCache::install_arena_payload: the header is not this cache's arena header "
                               "(engine bug)");
#endif
    if (p.resident) throw std::logic_error("TwirlPlanCache::install_arena_payload: the payload is already resident "
                                           "(engine bug)");
    p.resident = true;
    p.payload_bytes = plan_payload_bytes(p);
    resident_bytes_ += p.payload_bytes;
    fifo_.push_back(&p);
}
size_t TwirlPlanCache::arena_size() const { return k2_ ? k2_->recs.size() : 0; }
size_t TwirlPlanCache::arena_bytes_() const {
    size_t b = 0;
    for (const K2Arena::Rec& r : k2_->recs)
        b += sizeof(CachedPlan) + (size_t)r.words * sizeof(uint64_t) + sizeof(K2Arena::Rec) +
             2 * (1 + (size_t)r.words * sizeof(uint64_t) + 2 * sizeof(void*));   // the two index entries
    return b;
}

// ── J-b1 P11a: TWPL v6 (a FAM cache's disk layer) ───────────────────────────────────────────────────────────────
// [magic "TWPL"][u32 6][ENGINE_VERSION][u64 group token][u64 term-set token][u64 count] then per plan ONE class tag:
//   0  a diagonal header: its a-support words and sorted CZ pair list (v5's per-plan record);
//   1  a route-1 arena header (kind 0 outside the pair universe), 2  a kind-2 arena header: the reduced term set S.
// Keys only (no payload: built on first use, the identical build). Version / engine / group / term-set token
// mismatch → loads nothing (the caller rebuilds silently; the sampler's INFO disk event says absent/rebuild).
namespace {
constexpr char kTwpl6Magic[4] = {'T', 'W', 'P', 'L'};
constexpr uint32_t kTwplVersion6 = 6;
}  // namespace
__attribute__((noinline, cold)) bool TwirlPlanCache::save_file_v6_(const std::string& path,
                                                                    uint64_t group_token) const {
    plan_cache::Writer w = plan_cache::begin_blob(kTwpl6Magic, kTwplVersion6);
    w.str(std::string(kEngineVersion));
    w.pod(group_token);
    w.pod(term_token_);
    uint64_t live = 0;
    for (const auto& up : headers_) if (up) ++live;
    live += arena_size();
    w.pod(live);
    std::vector<uint64_t> am((size_t)NW_), key;
    std::vector<std::pair<int, int>> cz;
    for (const auto& up : headers_) {
        if (!up) continue;
        w.pod((uint8_t)0);
        std::memcpy(am.data(), up->key(), (size_t)NW_ * sizeof(uint64_t));
        key_cz(*up, cz);
        w.vpod(am);
        w.vpod(cz);
    }
    if (k2_)
        for (const K2Arena::Rec& r : k2_->recs) {
            w.pod(r.tag);
            key.assign(r.p->key(), r.p->key() + r.words);
            w.vpod(key);
        }
    const std::string blob = plan_cache::finish_blob(w);
#ifdef _WIN32
    const std::string tmp = path + ".tmp." + std::to_string((unsigned long)_getpid());
#else
    const std::string tmp = path + ".tmp." + std::to_string((unsigned long)::getpid());
#endif
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(blob.data(), 1, blob.size(), f) == blob.size();
    std::fclose(f);
    if (!ok) { std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}
__attribute__((noinline, cold)) bool TwirlPlanCache::load_file_v6_(const std::string& path, uint64_t group_token) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string blob;
    { std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
      if (sz <= 0) { std::fclose(f); return false; }
      blob.resize((size_t)sz);
      if (std::fread(&blob[0], 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; } }
    std::fclose(f);
    plan_cache::Reader r = plan_cache::open_blob(blob, kTwpl6Magic);
    if (!r.ok) return false;
    uint32_t ver = 0; r.pod(ver);
    if (!r.ok || ver != kTwplVersion6) return false;    // a v5 (production) file, or any other: loads nothing
    std::string eng; r.str(eng);
    if (!r.ok || eng != kEngineVersion) return false;
    uint64_t tok = 0, tt = 0, count = 0;
    r.pod(tok); r.pod(tt); r.pod(count);
    if (!r.ok || tok != group_token || tok != token_) return false;
#ifndef XTIM_MUTATE_TWPL6_NOTOKEN
    if (tt != term_token_) return false;                // another term view (merged axes / rows / ker Φ): nothing
#endif
    std::vector<uint64_t> am, key((size_t)KW_), k2;
    std::vector<std::pair<int, int>> cz;
    for (uint64_t i = 0; i < count; ++i) {
        uint8_t tag = 255;
        r.pod(tag);
        if (!r.ok) return false;
        if (tag == 0) {
            r.vpod(am);
            r.vpod(cz);
            if (!r.ok) return false;                    // a short read: stop (the loaded prefix stands, as v5)
            if ((int)am.size() != NW_) continue;
            std::fill(key.begin(), key.end(), 0ULL);
            std::memcpy(key.data(), am.data(), (size_t)NW_ * sizeof(uint64_t));
            bool ok = true;
            for (const auto& pr : cz) {
                auto it = pidx_.find(((uint64_t)(uint32_t)pr.first << 32) | (uint32_t)pr.second);
                if (it == pidx_.end()) { ok = false; break; }
                key[(size_t)NW_ + (size_t)(it->second >> 6)] |= 1ULL << (it->second & 63);
            }
            if (!ok) continue;
            const uint64_t h = plan_key_hash(key.data(), KW_);
            if (!find(key.data(), h)) new_header_(key.data(), h);
        } else if (tag == kArenaRoute1 || tag == kArenaKind2) {
            r.vpod(k2);
            if (!r.ok) return false;
            arena_header(tag, k2.data(), (int)k2.size());   // header only; payload on first use
        } else {
            return false;                               // an unknown class tag: stop
        }
    }
    return r.at_end();
}

__attribute__((noinline)) void TwirlRecordSampler::Impl::fam_refuse_(const std::string& m) {   // J-b1 P3: cold path
    family_refusal_ = m;                            // the bindings raise it (S7: no stderr copy)
}

// J-b1 P2: the slow path's only FAM invariant (no exact route exists in run_w<*>).
__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::fam_plan_fallback_bug_() const {
    throw std::logic_error("twirl sampler: a diagonal plan of a family sampler fell back (engine bug: the slow path "
                           "builds diagonal-class normal forms only)");
}
// J-b1 P2: the TWPL save notice of a FAM run, as an event (the Python layer logs it; no stderr).
__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::fam_disk_saved_() {
    diag_.push_back({TwirlRecordSampler::DiagEvent::DISK_SAVE, "SAVED", (long)cache->size(), 0, disk_path});
}

// A record classified as a family extra under its class bit (kind 0 gauge detector, 1 ANTI detector, 2
// extra observable): its record operator is the product of its terminal reads (Y = i·XZ), as the ctor's.
void TwirlRecordSampler::Impl::fam_add_(int kind, int index, const std::vector<int>& recs,
                                        const std::vector<std::pair<int, int>>& reads) {
    Pauli W(n);
    for (int k : recs) {
        Pauli P(n);
        const int b2 = reads[(size_t)k].first, q = reads[(size_t)k].second;
        if (b2 == 0) P.setx(q);
        else if (b2 == 2) P.setz(q);
        else { P.setx(q); P.setz(q); P.phase = 1; }
        W = Pauli::multiply(W, P);
    }
    famx_.push_back({W, rec_order_(kind == 2 ? 1 : 0, index), kind, index, recs});
}
// The family extras' record columns, after the Born observable and the Born decisions (storage order).
void TwirlRecordSampler::Impl::fam_cols_() {
    for (const FamX& fx : famx_) rec_cols_.push_back(RecColumn{fx.M, fx.order, false, {}, 0});
}

// J-b1 P11a: the TWPL v6 term-set token of a term view — FNV-1a over the merged pushed axes (x, z, phase), each
// term's rows and ker Φ (the kernel vectors and their non-pivot terms): every input a kind-2 key depends on.
// Never 0 (0 = no term view).
uint64_t TwirlRecordSampler::Impl::term_set_token_(const TermResidual& tr) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](const void* p, size_t nb) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < nb; ++i) { h ^= b[i]; h *= 0x100000001b3ULL; }
    };
    auto mixv = [&mix](const std::vector<uint64_t>& v) {
        const uint64_t sz = v.size();
        mix(&sz, sizeof sz);
        if (sz) mix(v.data(), v.size() * sizeof(uint64_t));
    };
    const int n0 = tr.n(), T = tr.num_terms();
    mix(&n0, sizeof n0);
    mix(&T, sizeof T);
    for (int j = 0; j < T; ++j) {
        const Pauli& B = tr.axis(j);
        mixv(B.x);
        mixv(B.z);
        mix(&B.phase, sizeof B.phase);
        const std::vector<int>& rows = tr.term_rows(j);
        const uint64_t nr = rows.size();
        mix(&nr, sizeof nr);
        if (nr) mix(rows.data(), rows.size() * sizeof(int));
    }
    for (const TermResidual::KernelVec& kv : tr.kernel()) {
        mix(&kv.j, sizeof kv.j);
        mixv(kv.k);
    }
    return h ? h : 1;
}

// The FAM sampler's construction (called at the end of the shared ctor's channel / readout-flip setup).
//   * named refusals: a Born DECISION on a CH/PPR circuit without the ppr_records class (here, at construction,
//     unchanged from J-a); every other CH/PPR circuit without the class is refused at run() (fam_run_guard_,
//     J-b1 P2: the FAM instances carry no PPR family; a sampler without the bit still builds, so the test-API
//     accessors serve the J-a dual-run gates); with it, the kind-2 route serves every PPR-class shot (N4a,
//     fam_term_shot_);
//   * per family extra: inv (the record inverts) ⊕ ref (the parity of noiseless_reference_records over
//     its records); the readout-flip slot masks were built by the ctor (famx_rf_);
//   * J-b1 P10 (Addendum A1 §2, owner O1): a circuit with an INPUT_QUBITS port — text, or an ideal-state compile —
//     is REFERENCE-RELATIVE on FAM, like every other circuit: the reference is the noiseless run from the ideal
//     state (the bare), frame-free. The shared ctor keeps production's ABSOLUTE port contract until F (TS:486,
//     :597), so the deterministic DETECTOR / OBSERVABLE channels' reporting bits are zeroed here, the extras take
//     ref as above, and the Born observable's reference parity is set by family_port_born_ref (its record list is
//     the binding's). DECISION channels stay raw. MUT-REF-ABS keeps the absolute convention;
//   * run_fn_ re-picked: run() runs run_w<FW> (J-b2 J9: the FAM parameter and the family flag are gone).
void TwirlRecordSampler::Impl::fam_setup_(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                                          const TwirlRecordOptions& opt, const std::vector<ErrorChannel>& chans) {
    oracle_all_shots_ = opt.oracle_all_shots;
    auto refuse = [&](const std::string& m) { fam_refuse_(m); setup_error = 7; };
    if (!opt.circuit_channels) { refuse("a family sampler needs circuit channels"); return; }

    famx_base_ = (born_idx >= 0 ? 1 : 0) + (int)born_dec.size();
    if ((size_t)(nchan + famx_base_) + famx_.size() != rec_cols_.size())
        throw std::logic_error("twirl_records: family extras / record columns mismatch (engine bug)");
#ifdef XTIM_MUTATE_REF_ABS
    const bool port_rel = false;                   // mutation build only (MUT-REF-ABS): port columns stay ABSOLUTE
#else
    const bool port_rel = !opt.input_pauli_qubits.empty();
#endif
    if (port_rel)                                  // J-b1 P10: the deterministic det / obs channels (decisions raw)
        for (size_t c = 0; c < chan_det.size() + chan_obs.size(); ++c) chan_ref[c] = 0;
    std::vector<uint8_t> R;
    if ((!famx_.empty() && (port_rel || opt.input_pauli_qubits.empty())) || (port_rel && born_idx >= 0))
        R = noiseless_reference_records(bare, deferred, reads, opt.rec_invert, nullptr);
    famx_kind_.clear(); famx_out_.clear(); famx_off_.clear();
    for (const FamX& fx : famx_) {
        int inv = 0, ref = 0;
        for (int r : fx.recs) {
            if (r >= 0 && r < (int)opt.rec_invert.size()) inv ^= opt.rec_invert[(size_t)r] & 1;
            if (r >= 0 && r < (int)R.size()) ref ^= R[(size_t)r] & 1;
        }
#ifdef XTIM_MUTATE_MUTREF_NO_REF
        ref = 0;                                   // mutation build only (MUT-REF): extras emitted raw ⊕ inv
#endif
        famx_kind_.push_back((uint8_t)fx.kind);
        famx_out_.push_back(fx.index);
        famx_off_.push_back((uint8_t)((inv ^ ref) & 1));
    }
    if (famx_rf_.size() != famx_.size() * (size_t)RFW) famx_rf_.assign(famx_.size() * (size_t)RFW, 0);
    // N2 / J-b1 P5: the term view (terms, B_j, term rows, ker Φ) is built here, on the ctor's alternatives
    // (fam_terms_setup_; J-b1 F: one copy): the static merge, then the ONE commutation check.
#ifndef XTIM_MUTATE_RET_TERM
    const bool rec_term = opt.fam_record_route_term;   // J-b1 P4: the all-term recording oracle route
#else
    const bool rec_term = true;                     // mutation build only (MUT-RET-TERM): every recording shot term-routed
#endif
    fam_term_.p = new FamRoute();                   // fam_setup_ runs once, at construction
    if (port_rel && born_idx >= 0) fam_term_.p->port_ref = std::move(R);   // J-b1 P10: family_port_born_ref's input
    fam_term_.p->reader_bytes = &reader_bytes_;   // J-b1 P7 (m10): content headers charged to the readers' bytes
    fam_term_.p->rec_inv = opt.rec_invert;
    fam_term_.p->rec_term = rec_term;
    fam_term_.p->rec_flip = opt.rec_flip;
    fam_term_.p->armed_static = ppr_tables || !opt.input_pauli_qubits.empty() || rec_term;
    fam_term_.p->port_qubits = opt.input_pauli_qubits;
    if (!fam_terms_setup_(deferred, reads, chans)) return;
    if (fam_term_.p->tr) cache->set_term_token(term_set_token_(*fam_term_.p->tr));   // J-b1 P11a: TWPL v6
    run_fn_ = pick_run_();
}

// The DiagErrorSampler channels of the deferred noise stream (probs × factor) and `sides`, the per-alternative
// FiredPauli lists (flat-alt-id order == construction order). J-b1 F (F items 1-2): the ONE copy — the ctor calls it
// (its inline copy, TS:689, is deleted) and fam_setup_ reuses the ctor's lists; family_static_rule builds its own.
// The switch is exhaustive over the eight NoiseChannel members with NO default: -Wswitch is an error on it (below),
// so a new channel that is not handled here fails the build instead of reaching a dead runtime refusal.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
__attribute__((noinline, cold)) void twirl_alternatives_(const Circuit& deferred, double factor, std::vector<ErrorChannel>& chans,
                                                       std::vector<std::vector<FiredPauli>>& sides) {
    auto add_alt = [&](ErrorChannel& ch, double pp, std::vector<FiredPauli> fr) {
        ErrorChannel::Alt alt; alt.p = pp; ch.alts.push_back(std::move(alt));
        sides.push_back(std::move(fr)); };
    for (int si = 0; si < (int)deferred.stream.size(); ++si) {
        const Instr& ins = deferred.stream[si];
        if (ins.kind != Instr::Kind::Noise) continue;
        switch (ins.channel) {
            case NoiseChannel::X_ERROR: case NoiseChannel::Y_ERROR: case NoiseChannel::Z_ERROR: {
                const PauliBasis b = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                                   : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z;
                if (!(ins.probs[0] > 0.0)) break;   // p = 0: never fires, no atom (as the cases below)
                for (int q : ins.qubits) { ErrorChannel ch;
                    add_alt(ch, ins.probs[0] * factor, {{si, q, b}});
                    chans.push_back(std::move(ch)); }
                break; }
            case NoiseChannel::DEPOLARIZE1: case NoiseChannel::PAULI_CHANNEL_1: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
                for (int q : ins.qubits) { ErrorChannel ch;
                    for (int i = 0; i < 3; ++i) {
                        const double pp = (dep ? ins.probs[0] / 3.0 : ins.probs[i]) * factor;
                        if (pp > 0.0) add_alt(ch, pp, {{si, q, pc1_slot_basis(i)}}); }
                    if (!ch.alts.empty()) chans.push_back(std::move(ch)); }
                break; }
            case NoiseChannel::DEPOLARIZE2: case NoiseChannel::PAULI_CHANNEL_2: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE2;
                for (size_t pi = 0; pi + 1 < ins.qubits.size(); pi += 2) {
                    const int qa = ins.qubits[pi], qb = ins.qubits[pi + 1];
                    ErrorChannel ch;
                    for (int t = 0; t < 15; ++t) {
                        const double pp = (dep ? ins.probs[0] / 15.0 : ins.probs[t]) * factor;
                        if (pp <= 0.0) continue;
                        std::vector<FiredPauli> fr;
                        // Stim slot order: the shared pc2_* definition (circuit_ir.hpp).
                        if (const int c = pc2_first_code(t))  fr.push_back({si, qa, pauli_code_basis(c)});
                        if (const int c = pc2_second_code(t)) fr.push_back({si, qb, pauli_code_basis(c)});
                        add_alt(ch, pp, std::move(fr)); }
                    if (!ch.alts.empty()) chans.push_back(std::move(ch)); }
                break; }
            case NoiseChannel::CORRELATED: {
                // E/ELSE chain = ONE categorical channel (framed_sampler's construction): term t at
                // q_t * factor, its product as the fired list; identity terms are not alternatives.
                const std::vector<double> q = corr_outcome_probs(ins);
                ErrorChannel ch;
                for (int t = 0; t < corr_terms(ins); ++t) {
                    const double pp = q[(size_t)t] * factor;
                    if (!(pp > 0.0)) continue;
                    std::vector<FiredPauli> fr;
                    for (size_t i = 0; i < ins.qubits.size(); ++i)
                        if (const int c = corr_code(ins, t, (int)i)) fr.push_back({si, ins.qubits[i], pauli_code_basis(c)});
                    if (fr.empty()) continue;
                    add_alt(ch, pp, std::move(fr)); }
                if (!ch.alts.empty()) chans.push_back(std::move(ch));
                break; }
        }
    }
}
#pragma GCC diagnostic pop


// J-b1 P5: the term view and the owner's non-Clifford rule, at construction (header). A refusal is by name
// (setup_error 7), with the law-limitation text of plan r3.1 item 7 (deferred-stream gate / noise indices; the
// typed XtimTwirlRefusal and user-qubit naming are P13's).
// The law-limitation text of a static-rule refusal (plan r3.1 item 7) from TermClassRefusal's text.
static std::string static_rule_message_(const std::string& why) {
    const size_t at = why.find("; terms ");
    if (at != std::string::npos)
        return "twirl sampler: law limitation: fault-dependent surviving axis/Clifford frame: " + why.substr(at + 2) +
               ", so the surviving term (or the Clifford frame) depends on the faults; the circuit is T-depth <= 1 per "
               "shot but outside the fixed-axis term law (a per-shot fold is not built)";
    return "twirl sampler: law limitation: non-commuting magic terms (" + why.substr(why.find("term axes")) +
           ")";
}

// J-b1 P13 rulings (fixplan-p5-c14 (A)): the static rule's verdict on a deferred circuit, outside any sampler (the
// bindings type a TS:119 propagation refusal by it: rule ACCEPT = LAW_LIMITATION_PROPAGATION, REFUSE =
// NONCOMMUTING_MAGIC). The same alternatives and TermResidual merge as fam_terms_setup_.
StaticRuleVerdict family_static_rule(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                                     double p_factor) {
    StaticRuleVerdict v;
    std::vector<ErrorChannel> chans;
    std::vector<std::vector<FiredPauli>> alts;
    twirl_alternatives_(deferred, p_factor, chans, alts);
    std::vector<char> live;
    for (const ErrorChannel& ch : chans)
        for (const ErrorChannel::Alt& a : ch.alts) live.push_back(a.p > 0.0 ? 1 : 0);
    try {
        TermResidual tr(deferred, reads, alts, live);
    } catch (const TermClassRefusal& e) {
        v.accept = false;
        v.message = static_rule_message_(e.what());
        v.axes = e.axes;
        v.gate_i = e.gate_i;
        v.gate_j = e.gate_j;
        v.site = e.site;
    }
    return v;
}

__attribute__((noinline, cold)) bool TwirlRecordSampler::Impl::fam_terms_setup_(
        const Circuit& deferred, const std::vector<std::pair<int, int>>& reads, const std::vector<ErrorChannel>& chans) {
    FamRoute& F = *fam_term_.p;
    const std::vector<std::vector<FiredPauli>>& alts = sides;   // J-b1 F: the ctor's alternatives (one copy)
    std::vector<char> live;                         // flat-alt-id order == the channels' construction order
    for (const ErrorChannel& ch : chans)
        for (const ErrorChannel::Alt& a : ch.alts) live.push_back(a.p > 0.0 ? 1 : 0);
    F.live = live;
    try {
        F.tr.emplace(deferred, reads, alts, live, /*z_screen=*/!F.armed_static);
    } catch (const TermClassRefusal& e) {
        F.refusal.cls = 1;
        F.refusal.axes = e.axes;
        F.refusal.gate_i = e.gate_i;
        F.refusal.gate_j = e.gate_j;
        F.refusal.site = e.site;
        fam_refuse_(static_rule_message_(e.what()));
        setup_error = 7;
        return false;
    }
    if (!F.armed_static) {                          // unarmed: only the verdict is kept (R9a; the route is never
        F.tr.reset();                               // reached: fam_run_guard_ arms it only for ppr_records, frames
        return true;                                // or the all-term recording option, all in armed_static)
    }
    // N4b: the frames class — every port component as a term atom (P_F, S_F) at t = 0 (the alternatives'
    // propagation), composed rightmost on every framed shot; the composition is exact for clean, dressed and
    // poisoned components alike (J-b2 J6b: the static relabel, the dressing and the poison refusal are deleted).
    if (!F.port_qubits.empty()) {
        std::vector<std::pair<int, PauliBasis>> comps;
        for (int q : F.port_qubits) comps.push_back({q, PauliBasis::X});
        for (int q : F.port_qubits) comps.push_back({q, PauliBasis::Z});
        F.fr_setup(comps);
    }
    return true;
}

void TwirlRecordSampler::Impl::fam_frame_take_(long done) {
    FamRoute* F = fam_term_.p;
    if (!F || !F->frames) return;
    // J-b2 J6b (r2.1 E14): the shot's frame row (formerly the run loop's cur_frame_row_, read only here)
    F->frow = (has_input_frame_ && frame_data_) ? frame_data_ + (size_t)done * (size_t)frame_stride_ : nullptr;
    F->framed = false;
    F->cur_branch = nullptr;
    if (__builtin_expect(F->branch_ids != nullptr, 0)) {   // J-b1 P9b: an input_branch run
        const int32_t b = F->branch_ids[done];
        if (b != 0) {                               // a branch shot: framed (the term route), its frame row and trie
            ++F->branch_probes;
            FamBranch& B = *F->branches[(size_t)(b - 1)];
            if (__builtin_expect(!B.resident || F->branch_budget != SIZE_MAX, 0)) fam_branch_payload_(B);
            const int nb = 2 * F->pnq;
            for (int s2 = 0; s2 < nb; ++s2) F->brow[(size_t)s2] = (F->frow ? F->frow[s2] : 0) ^ B.dport[(size_t)s2];
            F->frow = F->brow.data();
            F->cur_branch = &B;
            F->framed = true;
            ++B.shots;
            return;
        }
    }
    if (F->frow)
        for (int s2 = 0; s2 < 2 * F->pnq; ++s2)
            if (F->frow[s2]) { F->framed = true; break; }
}
bool TwirlRecordSampler::Impl::fam_term_all_() const {
    return fam_term_.p && (fam_term_.p->framed || (fam_term_.p->rs && fam_term_.p->rec_term));
}
bool TwirlRecordSampler::Impl::fam_rec_d_() const { return fam_term_.p && fam_term_.p->rs && fam_term_.p->rec_d; }

// J-b1 P4: the layout-3 record of a diagonal shot of a recording run, written by the (D) paths (clean, accumulator,
// structural): the content is the plan's canonical S^a·CZ view (memo_key_'s format, decoded from the plan key: the
// a-mask, then the kept pairs in upairs order), P_rec = the shot's canonical prefix, y = the emission's own values.
// Bitwise the term route's kind-0 record of the same shot (G-REC-EQ, tests/test_jb1_p4.py). The content handle is
// cached per dense plan id (cold: interned at the plan's first recorded shot); no per-shot allocation.
__attribute__((noinline)) void TwirlRecordSampler::Impl::fam_rec_write_d_(const CachedPlan& plan, const Pauli& P) {
    FamRoute& F = *fam_term_.p;
    TwirlRecordSampler::FamilyRecordSink& R = *F.rs;
    if (plan.id >= F.d_rh.size()) F.d_rh.resize((size_t)plan.id + 1, -1);   // cold: a new plan id
    int& h = F.d_rh[plan.id];
    if (h < 0) {                                    // cold: the plan's content, interned once
        std::string key;
        key.reserve((size_t)n + 1);
        const uint64_t* k = plan.key();
        for (int q = 0; q < n; ++q) key.push_back((char)((k[(size_t)(q >> 6)] >> (q & 63)) & 1ULL));
        key.push_back((char)0xFF);
        for (size_t i = 0; i < upairs.size(); ++i)
            if ((k[(size_t)NW + (i >> 6)] >> (i & 63)) & 1ULL) {
                const auto& pr = upairs[i];
                key.push_back((char)(pr.first & 0xFF)); key.push_back((char)((pr.first >> 8) & 0xFF));
                key.push_back((char)(pr.second & 0xFF)); key.push_back((char)((pr.second >> 8) & 0xFF));
            }
        h = F.content_handle_(0, key);
    }
    if (!plan.rec) throw std::logic_error("layout-3 record: the emitting plan has no record table (engine bug)");
    const long r = R.row;
    const size_t PNW = (size_t)R.PNW;
    std::copy(P.x.begin(), P.x.end(), R.prec + (size_t)r * 2 * PNW);
    std::copy(P.z.begin(), P.z.end(), R.prec + (size_t)r * 2 * PNW + PNW);
    record_values_words_(*plan.rec, P.x.data(), P.z.data(), (int)P.x.size(), rec_uz_.data(), (int)rec_uz_.size(),
                         R.y + (size_t)r * (size_t)R.YW);
    if ((int)rec_uz_.size() > R.UZW)
        throw std::logic_error("layout-3 record: more outcomes than record columns (engine bug)");
    std::copy(rec_uz_.begin(), rec_uz_.end(), R.uz + (size_t)r * (size_t)R.UZW);
    R.nuz[r] = (int32_t)rec_uz_.size();
    R.handle[r] = h;
    R.route[r] = 0;
    R.ff[r] = F.ff ? 1 : 0;
    if (R.branch) R.branch[r] = 0;                  // J-b1 P9b: the (D) paths serve branch 0 only
    R.row = r + 1;
    ++F.stats.shots_record_d;
}

// The named refusals of a FAM run (before any shot): σ export / state retention (Q3: σ is refused on
// FAM samplers, it retires at J-b1; retention of the family is N5), an input frame supplied to a FAM
// sampler with family extras (the frames class, N4), and (J-b1 P2) a CH/PPR circuit without the ppr_records
// class — family extras or not: the FAM instances carry no PPR family. A Born decision there is refused earlier,
// at construction (fam_setup_).
// J-b1 P3: out of line (once per run(); the FAM TU's inline budget belongs to run_w<*>). The P3 cold paths
// are pinned noinline only: adding `cold` too made GCC split run_w<*> (3.5k-instruction head, the hot
// callees draw_rf / find / ext_prefix_ out of line; measured with <S>/jb1/tools/famprobe.sh).
__attribute__((noinline)) bool TwirlRecordSampler::Impl::fam_run_guard_(long shots) {
    family_refusal_.clear();                        // a refusal names THIS run (a refused run leaves no stale text)
    if (retain_state) {
        fam_refuse_("sigma export is refused on a family sampler (J-a Q3: sigma retires at J-b1); the family's "
                    "retention is the layout-3 record (set_family_record_sink)");
        return false;
    }
    FamRoute* FR = fam_term_.p;
    const bool rec = FR && FR->rs;
    if (rec) {
        // N5: a recording run — every shot is term-routed and writes its layout-3 record.
        const TwirlRecordSampler::FamilyRecordSink& R = *FR->rs;
        if (R.PNW != (n + 63) / 64 || R.YW != QCW_ || R.UZW < (int)rec_cols_.size() || R.cap - R.row < shots)
            throw std::invalid_argument("family record sink: shape mismatch (PNW / YW / UZW / capacity)");
        FR->ff = FR->frames && has_input_frame_ && frame_data_;
        // the shot's outcomes (u, z) are kept by record_rows_ (its KEEPU variant)
        rows_fn_ = nextra_ != 0 ? &Impl::record_rows_<true, true> : &Impl::record_rows_<false, true>;
    }
    const bool k2 = ppr_tables;                     // a CH/PPR circuit: the kind-2 route serves its PPR-class shots
    const bool frames = fam_term_.p && fam_term_.p->frames;
    // J-b1 P4: a recording run's diagonal shots are recorded by the (D) paths (fam_rec_write_d_) unless the
    // all-term oracle option is set; the term route keeps the PPR-class, route-1 and framed shots.
    if (FR) FR->rec_d = rec && !FR->rec_term;
    if (k2 || frames || (rec && FR->rec_term)) {
        FamRoute& F = *fam_term_.p;
        if (!F.ready) {                             // cold, once: the per-alt table and the shot scratch
            TermResidual& tr = fam_term_route_();  // armed at construction (J-b1 P4: never lazily on a run)
            tr.init_shot(F.sh);
            F.P0 = Pauli(n);
            F.pre = Pauli(n);
            F.S0.assign((size_t)tr.term_words(), 0);
            F.kbuf.reserve((size_t)tr.term_words() * sizeof(uint64_t));
            F.ready = true;
        }
        if (!F.tr) throw std::logic_error("term route armed without its term view (engine bug)");
    }
    return true;
}

// ── J-a (N4a): the kind-2 route ─────────────────────────────────────────────────────────────────────
namespace {
std::string kind0_plan_key_(const DiagNormalForm& canon);
std::string kind2_plan_key_(const std::vector<uint64_t>& key);
}  // namespace

bool TwirlRecordSampler::Impl::fam_term_shot_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1,
                                              uint64_t* bits) {
    FamRoute& F = *fam_term_.p;
    TermResidual& tr = *F.tr;                      // armed: asserted by fam_run_guard_
    // R = R_{alt e1-1} ⋯ R_{alt e0} (the shot's events in time order; later leftmost), then reduced mod ker Φ.
    tr.compose_terms_(b.ev.data() + e0, (int)(e1 - e0), F.sh);
    std::copy(F.sh.P.x.begin(), F.sh.P.x.end(), F.P0.x.begin());     // (P0, S0): the noise residual alone
    std::copy(F.sh.P.z.begin(), F.sh.P.z.end(), F.P0.z.begin());
    std::copy(F.sh.S.begin(), F.sh.S.end(), F.S0.begin());
    if (F.framed) {
        // N4b: R_total = R_noise · R_f — the supplied frame atoms RIGHTMOST (composed first; t = 0), then the
        // fired alternatives in time order. A clean component's atom folds to P·P″(S); a poisoned one's S_F
        // joins S.
#ifndef XTIM_MUTATE_FR_FOLD
        tr.clear_shot(F.sh);
        for (int s2 = 0; s2 < 2 * F.pnq; ++s2)
            if (F.frow[s2]) F.compose_frame_(s2);
        for (int32_t e = e0; e < e1; ++e) {
            const int a = b.ev[(size_t)e];
            tr.compose_atom_(tr.alt_px(a), tr.alt_pz(a), tr.alt_s(a), F.sh);
        }
#else
        for (int s2 = 0; s2 < 2 * F.pnq; ++s2)       // mutation build only (MUT-FR-FOLD): the frame composed
            if (F.frow[s2]) F.compose_frame_(s2);     // LEFTMOST — a static relabel, no P″(S) dressing
#endif
        ++F.stats.shots_framed;
        if (F.cur_branch) {                         // J-b1 P9b: the branch's eps delta on the untouched wires
            const Pauli& D = F.cur_branch->dspec;
            for (size_t w = 0; w < F.sh.P.x.size(); ++w) { F.sh.P.x[w] ^= D.x[w]; F.sh.P.z[w] ^= D.z[w]; }
        }
    }
    tr.reduce_key_(F.sh);
    F.kbuf.assign(reinterpret_cast<const char*>(F.sh.S.data()), F.sh.S.size() * sizeof(uint64_t));
    auto it = F.route.find(F.kbuf);
    if (it == F.route.end()) {                      // cold: the key's kind (kind_of_, cached) and plan
        FamRoute::Entry e;
        e.kind = tr.kind_of_(F.sh.S);
#ifdef XTIM_MUTATE_G_KIND
        e.kind = 2;                                 // mutation build only (MUT-G-KIND): the kind is ignored
#endif
        if (e.kind != 0) {
            e.k2 = F.k2_new_();
            e.k2->S = F.sh.S;
            FamRoute::attach_header(*this, *e.k2);
            e.plan_id = tr.plan_id_(kind2_plan_key_(tr.key_of(F.sh.S)));
            // a kind-2 plan id is one reduced set (key_of is injective on reduced sets): counted once, at its first
            // entry (an entry re-created after an eviction finds its plan id already served)
            if ((size_t)e.plan_id >= F.stats.plan_kind.size() || F.stats.plan_kind[(size_t)e.plan_id] < 0)
                ++F.stats.plans_kind2;
        }
        if (plan_budget_) F.route_trim_(plan_budget_);
        it = F.route.emplace(F.kbuf, std::move(e)).first;
        F.recharge_(it->first, it->second);
        if (it->second.kind != 0) F.route_fifo.push_back(&it->first);
        ++F.stats.keys;
    }
    FamRoute::Entry& e = it->second;
    const CachedPlan* plan = nullptr;
    const Pauli* Pp = &F.sh.P;
    if (e.kind == 0 && !e.k2 && !e.k0) {
        // cold (first visit of the entry): the cache key of the canonical (a & 1, kept pairs). N4-D1
        // (fixplan-k0pairs r2): a kept pair outside the diagonal cache's pair universe `upairs` cannot be keyed
        // there; the entry is served by the kind-2 (G) path instead — a sampler-owned header whose payload is
        // the (G) table of C0 = rot(S) (build_k2: law and emission block in one step, never decoded from a key),
        // emitted with the reduced Pauli P as kind 2 is. The entry keeps kind 0 and its kind-0 plan id (JA-4).
        // termroute-a n1: the block re-runs only after a refused get_or_build (engine bug; e.k0 stays null): the entry
        // is then already queued (its plan id is set below), and a second push would let a trim free it while queued.
        const bool queued = e.plan_id >= 0;
        e.nf0 = tr.kind0_normal_form_(Pauli(n), F.sh.S, G);
        e.k0key.assign((size_t)KW, 0);
        for (int q = 0; q < n; ++q)
            if (e.nf0.a[(size_t)q] & 1) e.k0key[(size_t)(q >> 6)] |= 1ULL << (q & 63);
        bool in_universe = true;
        for (const auto& pr : e.nf0.cz) {
            const auto u = std::lower_bound(upairs.begin(), upairs.end(), pr);
            if (u == upairs.end() || *u != pr) {
#ifdef XTIM_MUTATE_K0_REFUSE
                fam_refuse_("kind-2 route: a kind-0 composite has a CZ pair (" + std::to_string(pr.first) + ", " +
                            std::to_string(pr.second) + ") outside the diagonal plan cache's pair universe; "
                            "refused");                // mutation build only (MUT-K0-REFUSE): the N4-D1 refusal
                return false;
#endif
                in_universe = false;
                break;
            }
            const size_t ui = (size_t)(u - upairs.begin());
            e.k0key[(size_t)NW + (ui >> 6)] |= 1ULL << (ui & 63);
        }
        e.plan_id = tr.plan_id_(kind0_plan_key_(e.nf0));
        if (in_universe) e.k0h = lhash(e.k0key.data());
        else {
            e.k2 = F.k2_new_();
            e.k2->S = F.sh.S;
            e.k2->k0general = true;
            FamRoute::attach_header(*this, *e.k2);
            e.k0key.clear();
            ++F.stats.plans_kind0_general;
        }
        F.recharge_(it->first, e);
        if (in_universe && !queued) F.route_fifo.push_back(&it->first);
    }
    if (e.kind == 0 && !e.k2) {
        // JA-4: P·rot(S) through THE diagonal route's normal form; served by that kind-0 plan (builder D).
        // N4a-2: a Pauli applied after a diagonal Clifford changes only the normal form's prefix, by itself:
        // DiagPauliClifford::then gives v ⊕ x, a + 2z (mod 4), B unchanged, so residual_normal_form keeps (a & 1,
        // cz) and XORs (x, z) into the prefix, and canonicalize_mod_stabilizers' rewrite reads (a, cz) only and
        // XORs Z's into the prefix. Hence nf(P, S) = nf(I, S) with prefix ⊕ P, up to the prefix phase, which no
        // consumer reads (residual_normal_form's standing rule). nf(I, S) is the key's constant (cold, nf0); per
        // shot only the prefix words are XORed into a reused buffer: no allocation, no normal-form call.
        const bool first = e.k0 == nullptr;
        for (size_t w = 0; w < F.pre.x.size(); ++w) {
            F.pre.x[w] = e.nf0.prefix.x[w] ^ F.sh.P.x[w];
#ifndef XTIM_MUTATE_K0_PREFIX
            F.pre.z[w] = e.nf0.prefix.z[w] ^ F.sh.P.z[w];
#else
            F.pre.z[w] = e.nf0.prefix.z[w];         // mutation build only (MUT-K0-PREFIX): the shot's Z part dropped
#endif
        }
        plan = cache->get_or_build(G, bare, e.k0key.data(), e.k0h, e.nf0, first);
        if (!plan) {
            fam_refuse_("kind-2 route: the kind-0 plan key disagrees with the canonical normal form (engine bug)");
            return false;
        }
        if (first && plan->fallback)                // cold: build_shot_law falls back only off the diagonal class
            throw std::logic_error("kind-2 route: a kind-0 plan fell back (engine bug: kind0_normal_form_ is "
                                   "diagonal)");
        e.k0 = plan;
        Pp = &F.pre;
        ++F.stats.shots_kind0;
    } else {
        // kind 2, or kind 0 outside the pair universe (N4-D1): the sampler-owned (G) plan, emitted with P
#ifdef XTIM_MUTATE_K2_ROUTE
        plan = &identity_plan_();                   // mutation build only (MUT-K2-ROUTE): the kind is ignored at
                                                    // dispatch — the stripped key's kind-0 plan (no Clifford part)
#else
        FamRoute::K2Plan& k = *e.k2;
        if (!k.plan->resident) F.build_k2(*this, k);
        plan = k.plan;
#endif
        if (e.kind == 0) { ++F.stats.shots_kind0; ++F.stats.shots_kind0_general; }
        else ++F.stats.shots_kind2;
    }
    // The ONE emission: t(P) for the channels (prefix rows), the plan's record rows, t(P) for the extras.
    const Pauli& P = *Pp;
    for (size_t w = 0; w < P.x.size(); ++w)
        for (uint64_t word = P.x[w]; word; word &= word - 1) {
            const uint64_t* row = &pxrow[(w * 64 + (size_t)__builtin_ctzll(word)) * (size_t)CW];
            for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2];
        }
    for (size_t w = 0; w < P.z.size(); ++w)
        for (uint64_t word = P.z[w]; word; word &= word - 1) {
            const uint64_t* row = &pzrow[(w * 64 + (size_t)__builtin_ctzll(word)) * (size_t)CW];
            for (int w2 = 0; w2 < CW; ++w2) bits[w2] ^= row[w2];
        }
    if (__builtin_expect(F.cur_branch != nullptr, 0)) fam_record_rows_branch_(*plan, bits, *F.cur_branch);
    else record_rows_any_(*plan, bits);
#ifdef XTIM_MUTATE_K2_TRIE_P0
    if (e.k2 && !plan->rec_trie.empty() && !std::isnan(plan->rec_trie[0].p0))   // mutation build only
        const_cast<CachedPlan*>(plan)->rec_trie[0].p0 = std::nextafter(plan->rec_trie[0].p0, 2.0);  // (MUT-K2-TRIE-P0)
#endif
    if (nextra_) ext_prefix_(P);
    if (F.rs) {
        // N5: the shot's layout-3 record. y = the emission's own derivation (its table, its P, its outcomes);
        // P_rec = the Pauli of the record's content: kind 0 (either route) = nf0.prefix ⊕ P with C0 = the
        // canonical S^a·CZ (N4a-2's identity); kind 2 = P with C0 = rot(S). Same state, P_rec·C0|bare⟩.
        TwirlRecordSampler::FamilyRecordSink& R = *F.rs;
        const uint8_t route = e.kind != 0 ? 2 : (e.k2 ? 1 : 0);
        if (e.rh < 0) {                             // cold: the entry's content handle
            if (e.kind == 0) {
                for (size_t q = 0; q < e.nf0.a.size(); ++q)
                    if (e.nf0.a[q] > 1)
                        throw std::logic_error("layout-3 record: a kind-0 normal form with a ≥ 2 (engine bug)");
                e.rh = F.content_handle_(0, memo_key_(e.nf0));
            } else {
                e.rh = F.content_handle_(2, F.kbuf);
            }
        }
        if (route == 1)
            for (size_t w = 0; w < F.pre.x.size(); ++w) {
                F.pre.x[w] = e.nf0.prefix.x[w] ^ F.sh.P.x[w];
                F.pre.z[w] = e.nf0.prefix.z[w] ^ F.sh.P.z[w];
            }
        const Pauli& Prec = route == 2 ? F.sh.P : F.pre;
        const long r = R.row;
        const size_t PNW = (size_t)R.PNW;
        std::copy(Prec.x.begin(), Prec.x.end(), R.prec + (size_t)r * 2 * PNW);
        std::copy(Prec.z.begin(), Prec.z.end(), R.prec + (size_t)r * 2 * PNW + PNW);
        if (!plan->rec)
            throw std::logic_error("layout-3 record: the emitting plan has no record table (engine bug)");
        record_values_words_(*plan->rec, P.x.data(), P.z.data(), (int)P.x.size(), rec_uz_.data(),
                             (int)rec_uz_.size(), R.y + (size_t)r * (size_t)R.YW);
        if ((int)rec_uz_.size() > R.UZW)
            throw std::logic_error("layout-3 record: more outcomes than record columns (engine bug)");
        std::copy(rec_uz_.begin(), rec_uz_.end(), R.uz + (size_t)r * (size_t)R.UZW);
        R.nuz[r] = (int32_t)rec_uz_.size();
        R.handle[r] = e.rh;
        R.route[r] = route;
        R.ff[r] = F.ff ? 1 : 0;
        if (R.branch) R.branch[r] = F.cur_branch ? F.cur_branch->id : 0;
        R.row = r + 1;
    }
    if (e.plan_id >= 0) {
        if ((size_t)e.plan_id >= F.stats.plan_shots.size()) {     // cold: a new plan id
            F.stats.plan_shots.resize((size_t)e.plan_id + 1, 0);
            F.stats.plan_kind.resize((size_t)e.plan_id + 1, -1);
        }
        ++F.stats.plan_shots[(size_t)e.plan_id];
        F.stats.plan_kind[(size_t)e.plan_id] = e.kind;
    }
    F.term_shot = shot_;
    return true;
}

// rot(B)|ψ⟩, rot(B) = exp(i(π/4) B), up to a global phase, by gates: a Clifford V with V B V† = ±Z_q (H / S on
// the support, then CX onto one wire; B tracked exactly by the tableau conjugations), rot(±Z_q) ∝ S† / S, V†.
namespace {
void apply_rot_(FramedSuperposition& st, Pauli B) {
    struct Gt { uint8_t k; int a, b; };
    std::vector<Gt> done;
    auto app = [&](uint8_t k, int a, int b) {
        st.apply_clifford(k, a, b);
        done.push_back({k, a, b});
        if (k == 0) CliffordTableau::conj_h(B, a);
        else if (k == 1) CliffordTableau::conj_s(B, a);
        else CliffordTableau::conj_cx(B, a, b);
    };
    const int n = B.n;
    for (int q = 0; q < n; ++q) {
        if (B.xbit(q) && B.zbit(q)) { app(1, q, 0); app(0, q, 0); }   // Y → ∓X → ∓Z
        else if (B.xbit(q)) app(0, q, 0);                             // X → Z
    }
    int q0 = -1;
    for (int q = 0; q < n; ++q)
        if (B.zbit(q)) { if (q0 < 0) q0 = q; else app(6, q, q0); }    // CX(q → q0): Z_q Z_q0 → Z_q0
    for (int q = 0; q < n; ++q)
        if (B.xbit(q) || (B.zbit(q) && q != q0))
            throw std::logic_error("term-view oracle: the axis did not reduce to one Z (engine bug)");
    if (q0 < 0 || (B.phase & 1)) throw std::logic_error("term-view oracle: a non-Hermitian / identity axis (engine bug)");
    st.apply_clifford(B.phase == 0 ? 2 : 1, q0, 0);                   // exp(+iπ/4 Z) ∝ S†, exp(−iπ/4 Z) ∝ S
    for (size_t i = done.size(); i-- > 0;)
        st.apply_clifford(done[i].k == 1 ? 2 : done[i].k, done[i].a, done[i].b);
}
}  // namespace

// The term-view semantic oracle of a term-routed shot: R|bare⟩ built from the COMPOSED (P0, S0) by gates —
// Π_{j∈S0} rot(B_j) (commuting: any order), then P0 — sharing no table, pullback or normal form with the
// route; then every record projected at its emitted raw value (a definite record must agree), as
// semantic_oracle_ / semantic_oracle_fam_ do on diagonal shots.
__attribute__((noinline)) bool TwirlRecordSampler::Impl::fam_term_oracle_(long shot) {   // J-b1 P3: oracle window
    FamRoute& F = *fam_term_.p;
    const TermResidual& tr = *F.tr;
    sem_work_ = bare;
    if (F.framed) {                                 // N4b: R_f first (t = 0): each supplied atom, rot(S_F) then P_F
        const size_t NW = F.P0.x.size(), TW = (size_t)tr.term_words();
        for (int s2 = 0; s2 < 2 * F.pnq; ++s2) {
            if (!F.frow[s2]) continue;
            for (int j = 0; j < tr.num_terms(); ++j)
                if ((F.fs[(size_t)s2 * TW + (size_t)(j >> 6)] >> (j & 63)) & 1ULL) apply_rot_(sem_work_, tr.axis(j));
            for (int q = 0; q < n; ++q) {
                if ((F.fpz[(size_t)s2 * NW + (size_t)(q >> 6)] >> (q & 63)) & 1ULL) sem_work_.apply_clifford(5, q, 0);
                if ((F.fpx[(size_t)s2 * NW + (size_t)(q >> 6)] >> (q & 63)) & 1ULL) sem_work_.apply_clifford(3, q, 0);
            }
        }
    }
    for (int j = 0; j < tr.num_terms(); ++j)
        if ((F.S0[(size_t)(j >> 6)] >> (j & 63)) & 1ULL) apply_rot_(sem_work_, tr.axis(j));
    for (int q = 0; q < n; ++q) {
        if (F.P0.zbit(q)) sem_work_.apply_clifford(/*Z*/ 5, q, 0);
        if (F.P0.xbit(q)) sem_work_.apply_clifford(/*X*/ 3, q, 0);
    }
    auto project_at = [&](const Pauli& W, int raw, const char* what, int idx) -> bool {
        const double p1 = sem_work_.born_p1(W);
        if (p1 > 1e-9 && p1 < 1.0 - 1e-9) {
            (void)sem_work_.measure_pauli(W, raw ? std::nextafter(1.0, 0.0) : 0.0);
        } else if ((p1 > 0.5) == (raw != 0)) {
            std::fprintf(stderr, "twirl_records SEMANTICS ORACLE FAIL shot=%ld (term route): %s %d emitted raw %d "
                                 "but is definite %+d on the record-conditioned state\n",
                         shot, what, idx, raw, p1 > 0.5 ? 1 : -1);
            fam_refuse_("semantic oracle failure on a term-routed shot (engine bug)");
            return false;
        }
        return true;
    };
    for (int c = 0; c < nchan; ++c) {
        Pauli Gc(n);
        const std::vector<uint64_t>& mask = sch[(size_t)c].mask;
        for (int w = 0; w < GW && w < (int)mask.size(); ++w)
            for (uint64_t word = mask[(size_t)w]; word; word &= word - 1)
                Gc = Pauli::multiply(Gc, gens[(size_t)(w * 64 + __builtin_ctzll(word))]);
        if (!project_at(Gc, (int)((sem_raw_bits_[(size_t)(c >> 6)] >> (c & 63)) & 1), "channel", c)) return false;
    }
    if (born_obs && !project_at(born_W, obs_bit_cur_raw_ & 1, "born observable", 0)) return false;
    for (size_t j = 0; j < born_dec.size(); ++j)
        if (!project_at(born_dec[j].second, dec_raw_[j] & 1, "born decision", (int)j)) return false;
    ++sem_checked_;
    ++F.stats.oracle_term_shots;
    return fam_project_extras_(shot);
}

// The semantic oracle of a FAM sampler: today's oracle (channels, the Born observable, the Born
// decisions projected on R|bare⟩ at their emitted raw values), then every family extra at its raw value
// y (record operators all commute: terminal reads) — a definite record must agree. Per-class counts.
__attribute__((noinline)) bool TwirlRecordSampler::Impl::semantic_oracle_fam_(   // J-b1 P3: oracle window
        const DiagErrorSampler::SparseBatch& b, int32_t e0,
                                                    int32_t e1, long shot) {
    if (fam_term_.p && fam_term_.p->term_shot == shot_) return fam_term_oracle_(shot);   // N4a: term-routed
    const long before = sem_checked_;
    if (!semantic_oracle_(b, e0, e1, shot)) return false;
    if (sem_checked_ == before) return true;      // out of the diagonal class: no oracle
    return fam_project_extras_(shot);
}

// Per-class counts (the channels, the Born observable and decisions were projected by the caller), then every
// family extra at its raw value y (record operators all commute: terminal reads) — a definite one must agree.
__attribute__((noinline)) bool TwirlRecordSampler::Impl::fam_project_extras_(long shot) {   // J-b1 P3: cold
    sem_class_[0] += nchan;
    if (born_obs) ++sem_class_[1];
    sem_class_[2] += (long)born_dec.size();
    for (size_t i = 0; i < famx_.size(); ++i) {
        const int e = famx_base_ + (int)i;
#ifdef XTIM_MUTATE_SEM_SKIP_EXTRA
        if (e > 0) continue;                       // mutation build only: extras beyond the first skipped
#endif
        const int raw = xbit_(e) ? 1 : 0;
        const Pauli& W = rec_cols_[(size_t)(nchan + e)].M;
        const double p1 = sem_work_.born_p1(W);
        if (p1 > 1e-9 && p1 < 1.0 - 1e-9) {
            (void)sem_work_.measure_pauli(W, raw ? std::nextafter(1.0, 0.0) : 0.0);
        } else if ((p1 > 0.5) == (raw != 0)) {
            std::fprintf(stderr, "twirl_records SEMANTICS ORACLE FAIL shot=%ld: family extra %zu (kind %d, "
                                 "index %d) emitted raw %d but is definite %+d on the record-conditioned "
                                 "state\n", shot, i, famx_[i].kind, famx_[i].index, raw, p1 > 0.5 ? 1 : -1);
            fam_refuse_("semantic oracle failure on a family extra (engine bug)");
            return false;
        }
        ++sem_class_[3 + famx_[i].kind];
    }
    return true;
}

// ── The FAM sampler's test-API surface ─────────────────────────────────────────────────────────
void TwirlRecordSampler::set_family_sink(FamilySink* sink) {
    impl_->family_sink_ = sink;
    if (!sink) return;
    // The channel metadata from the sampler's own vectors (one source of truth; no caller precondition), and
    // the detector word path (the former DetsPackSink's).
    const Impl& I = *impl_;
    sink->cdet = I.chan_det.data();
    sink->ndetc = (int)I.chan_det.size();
    sink->cobs = I.chan_obs.data();
    sink->ncobs = (int)I.chan_obs.size();
    sink->cref = I.chan_ref.data();
    sink->det_identity = true;
    for (int c = 0; c < sink->ndetc; ++c)
        if (I.chan_det[(size_t)c] != c) { sink->det_identity = false; break; }
    sink->det_words = (sink->ndetc + 63) / 64;
    sink->det_partial_bits = sink->ndetc % 64;
    sink->cref_det_packed.assign((size_t)sink->det_words, 0ULL);
    for (int c = 0; c < sink->ndetc; ++c)
        if (I.chan_ref[(size_t)c]) sink->cref_det_packed[(size_t)(c >> 6)] |= 1ULL << (c & 63);
    sink->xbase = impl_->famx_base_;
    sink->nfx = (int)impl_->famx_.size();
    sink->xkind = impl_->famx_kind_.data();
    sink->xout = impl_->famx_out_.data();
    sink->xoff = impl_->famx_off_.data();
    sink->xrf = impl_->famx_rf_.data();
    sink->RFW = impl_->RFW;
}
std::shared_ptr<TwirlPlanCache> TwirlRecordSampler::family_plan_cache() const {
    if (!impl_->fam_term_.p) throw std::logic_error("family_plan_cache: not a family sampler (test API only)");
    return impl_->cache;
}
void TwirlRecordSampler::family_port_born_ref(const std::vector<int>& born_records) {
    Impl& I = *impl_;
    if (!I.fam_term_.p) throw std::logic_error("family_port_born_ref: not a family sampler (test API only)");
    const std::vector<uint8_t>& R = I.fam_term_.p->port_ref;
    if (R.empty() || I.born_idx < 0) return;       // no port, or no Born observable: nothing to set
    int par = 0;
    for (int r : born_records) if (r >= 0 && r < (int)R.size()) par ^= R[(size_t)r] & 1;
    I.born_ref_ = par & 1;
}
std::vector<TwirlRecordSampler::FamilyExtra> TwirlRecordSampler::family_extras() const {
    std::vector<FamilyExtra> out;
    for (size_t i = 0; i < impl_->famx_.size(); ++i)
        out.push_back({impl_->famx_[i].kind, impl_->famx_[i].index, impl_->nchan + impl_->famx_base_ + (int)i,
                       i < impl_->famx_off_.size() ? impl_->famx_off_[i] : (uint8_t)0});
    return out;
}
const std::string& TwirlRecordSampler::family_refusal() const { return impl_->family_refusal_; }


// J-b1 P6: a FAM (D) plan's record table by record-space B (build_record_table_recB: the σ coin part is
// build_shot_law's cone-local core; the channel rows are computed in RECORD space), replacing the σ dual basis
// (build_record_table's kernel stage) on the FAM path. The canonical table is unique (record_space_b.hpp), so the
// emission is unchanged (G-B1). Legacy (D) keeps build_record_table.
__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::build_rec_table_fam_(
        const CachedPlan& plan, const DiagNormalForm& nf0, RecTable& out) const {
    FamRoute& F = *fam_term_.p;
    // fixplan-coldstart (A): the payload's ONE core law, parked by the plan cache on the same miss.
    const ShotLaw* law = nullptr;
    const twirl_law_detail::ShotLawCore* core = cache->parked_core(plan, &law);
#ifdef XTIM_MUTATE_1A_DOUBLE
    core = nullptr;                                 // mutation build only (MUT-1A-DOUBLE): the second law reinstated
#endif
    bool ok;
    if (core) {
        ok = build_record_table_recB_from_core(G, bare, nf0, *law, *core, rec_cols_, out);
        ++F.stats.tables_recb_parked;
    } else {                                        // the slot does not hold this plan (counted: expected 0)
        ok = build_record_table_recB(G, bare, nf0, rec_cols_, out);
        ++F.stats.recb_core_rebuilds;
    }
    if (!ok)
        throw std::logic_error("build_rec_table_fam_: record-space B reports a fallback on a non-fallback plan "
                               "(engine bug)");
    ++F.stats.tables_recb;
}

TwirlRecordSampler::FamilyRefusalInfo TwirlRecordSampler::family_refusal_info() const {
    return impl_->fam_term_.p ? impl_->fam_term_.p->refusal : FamilyRefusalInfo{};
}
std::vector<long> TwirlRecordSampler::semantic_oracle_class_checks() const {
    return std::vector<long>(impl_->sem_class_, impl_->sem_class_ + 6);
}


// ── J-a (N2): the term route's test-API surface ──────────────────────────────────────────────────
TwirlRecordSampler::Impl::FamTermHold::~FamTermHold() { delete p; }
TermResidual& TwirlRecordSampler::Impl::fam_term_route_() {
    if (!fam_term_.p) throw std::logic_error("term route: not a family sampler (J-a: test API only)");
    TermResidual& t = fam_term_.p->term_();
    if (!t.alts_ready()) t.set_alts(sides);
    return t;
}
// J-a final review m1: the bitwise Born-trie gates over the sampler-owned plans (the production walks,
// born_trie_check / born_trie_oracle in twirl_sampler.cpp, cover the diagonal cache's header table only, and are
// left untouched so production codegen stays frozen; J-b1 folds the two back into one walk).
__attribute__((cold)) std::vector<long> TwirlRecordSampler::family_born_trie_walk(bool oracle) {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_born_trie_walk: not a family sampler (J-a: test API only)");
    struct Restore {                                  // as born_trie_check: the walk moves no reported counter
        BornTrie& t;
        const long walks, steps, nodes, node_evals, chi_j, passes, max_support;
        const int max_depth;
        ~Restore() {
            t.walks = walks; t.steps = steps; t.nodes = nodes; t.node_evals = node_evals;
            t.chi_j = chi_j; t.passes = passes; t.max_support = max_support; t.max_depth = max_depth;
        }
    } restore{I.trie_, I.trie_.walks, I.trie_.steps, I.trie_.nodes, I.trie_.node_evals,
              I.trie_.chi_j, I.trie_.passes, I.trie_.max_support, I.trie_.max_depth};
    long checked[2] = {0, 0}, bad = 0;
    for (const auto& kp : F->k2plans) {
        const CachedPlan& p = *kp->plan;
        if (!p.resident || !p.rec || p.rec_trie.empty()) continue;
        const int r = kp->k0general ? 1 : 0;
        const std::vector<FreeDir>& lam = p.rec->lam;
        std::vector<Pauli> L;
        if (oracle)
            for (const FreeDir& d : lam) L.push_back(derive_kernel_logical(I.bare, I.G, d));
        std::vector<std::pair<uint32_t, std::vector<uint8_t>>> st{{0u, {}}};
        while (!st.empty()) {
            auto [node, pre] = st.back();
            st.pop_back();
            const BornNode nd = p.rec_trie[node];
            if (std::isnan(nd.p0)) continue;
            double q;
            if (oracle) {
                FramedSuperposition w = I.bare;
                for (size_t j = 0; j < pre.size(); ++j)
                    (void)w.measure_pauli(L[j], pre[j] ? std::nextafter(1.0, 0.0) : 0.0);
                q = w.born_p1(L[pre.size()]);
            } else {
                q = I.trie_.p0_from_scratch(lam, pre.data(), (int)pre.size());
            }
            ++checked[r];
            if (std::memcmp(&q, &nd.p0, sizeof(double)) != 0) ++bad;
            for (int b = 0; b < 2; ++b)
                if (nd.kid[b]) {
                    std::vector<uint8_t> z = pre;
                    z.push_back((uint8_t)b);
                    st.emplace_back(nd.kid[b], std::move(z));
                }
        }
    }
    return {checked[0], checked[1], bad};
}
__attribute__((cold)) std::vector<long> TwirlRecordSampler::family_trie_nodes() const {
    const Impl& I = *impl_;
    const Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_trie_nodes: not a family sampler (J-a: test API only)");
    auto count = [](const CachedPlan& p) {
        long c = 0;
        if (p.resident && p.rec)
            for (const BornNode& nd : p.rec_trie) c += std::isnan(nd.p0) ? 0 : 1;
        return c;
    };
    std::vector<long> out(3, 0);
    for (size_t i = 0; i < I.cache->size(); ++i) out[0] += count(I.cache->header(i));
    for (const auto& kp : F->k2plans) out[kp->k0general ? 2 : 1] += count(*kp->plan);
    return out;
}
TwirlRecordSampler::FamilyRouteStats TwirlRecordSampler::family_route_stats() const {
    if (!impl_->fam_term_.p) throw std::logic_error("family_route_stats: not a family sampler (J-a: test API only)");
    FamilyRouteStats st = impl_->fam_term_.p->stats;
    st.key_bytes = (long)impl_->fam_term_.p->route_bytes;            // termroute-a cache bound
    st.key_evictions = impl_->fam_term_.p->route_evictions;
    st.laws_payload = (long)impl_->cache->core_builds();   // J-b1 (fixplan-coldstart): core-mode payload law builds
    return st;
}
// The plan-identity keys (JA-4: composite Clifford mod Pauli, whatever route composed it). Kind 0: the
// stripped (a, cz) key of the canonical normal form — the diagonal plan cache's key content (a & 1 off the
// product wires, the kept pairs). Kind 2: the term key.
namespace {
std::string kind0_plan_key_(const DiagNormalForm& canon) {
    std::string k(1, '0');
    for (size_t q = 0; q < canon.a.size(); ++q) k.push_back((char)(canon.a[q] & 1));
    for (const auto& pr : canon.cz) {
        k.append(reinterpret_cast<const char*>(&pr.first), sizeof(int));
        k.append(reinterpret_cast<const char*>(&pr.second), sizeof(int));
    }
    return k;
}
std::string kind2_plan_key_(const std::vector<uint64_t>& key) {
    std::string k(1, '2');
    k.append(reinterpret_cast<const char*>(key.data()), key.size() * sizeof(uint64_t));
    return k;
}
}  // namespace
TermResidual& TwirlRecordSampler::family_terms() { return impl_->fam_term_route_(); }
const std::vector<std::vector<FiredPauli>>& TwirlRecordSampler::family_alt_faults() const { return impl_->sides; }
std::vector<uint8_t> TwirlRecordSampler::family_alt_is_ppr() const { return impl_->alt_is_ppr; }

std::vector<std::vector<int32_t>> TwirlRecordSampler::family_draw_events(long shots, uint64_t seed) {
    if (!impl_->set_seed(seed)) throw std::runtime_error("family_draw_events: set_seed refused");
    std::vector<std::vector<int32_t>> out;
    out.reserve((size_t)shots);
    DiagErrorSampler::SparseBatch batch;
    for (long done = 0; done < shots;) {
        const int S = (int)std::min<long>(4096, shots - done);
        impl_->smp->sample_events(S, batch);
        for (int s = 0; s < S; ++s)
            out.emplace_back(batch.ev.begin() + batch.shot_off[(size_t)s], batch.ev.begin() + batch.shot_off[(size_t)s + 1]);
        done += S;
    }
    return out;
}

void TwirlRecordSampler::family_term_shot(const std::vector<int32_t>& alts, TermRouteShot& out) {
    TermResidual& tr = impl_->fam_term_route_();
    for (int32_t a : alts)
        if (a < 0 || a >= tr.num_alts()) throw std::invalid_argument("family_term_shot: alternative id out of range");
    if (out.sh.P.n != tr.n() || (int)out.sh.S.size() != tr.term_words()) tr.init_shot(out.sh);
    tr.compose_terms_(alts.data(), (int)alts.size(), out.sh);
    out.P0 = out.sh.P;
    out.S0 = out.sh.S;
    tr.reduce_key_(out.sh);
    int kind = tr.kind_of_(out.sh.S);
#ifdef XTIM_MUTATE_G_KIND
    kind = 2;                                     // mutation build only (MUT-G-KIND): the kind is ignored
#endif
    out.kind = kind;
    if (kind == 0) {
        out.nf = tr.kind0_normal_form_(out.sh.P, out.sh.S, impl_->G, &out.nf_raw);
        out.plan_id = tr.plan_id_(kind0_plan_key_(out.nf));
    } else {
        out.nf = DiagNormalForm();
        out.nf_raw = DiagNormalForm();
        out.plan_id = tr.plan_id_(kind2_plan_key_(tr.key_of(out.sh.S)));
    }
}

bool TwirlRecordSampler::family_diag_shot(const std::vector<int32_t>& alts, DiagNormalForm& nf, long& plan_id) {
    TermResidual& tr = impl_->fam_term_route_();
    std::vector<FiredPauli> fired;
    for (int32_t a : alts) {
        if (a < 0 || a >= (int)impl_->sides.size()) throw std::invalid_argument("family_diag_shot: alternative id out of range");
        if (!impl_->alt_is_ppr.empty() && impl_->alt_is_ppr[(size_t)a]) return false;
        for (const FiredPauli& f : impl_->sides[(size_t)a]) fired.push_back(f);
    }
    DiagPauliClifford D = compose_fired(impl_->table, fired);
    D.finalize();
    nf = residual_normal_form(D);
    canonicalize_mod_stabilizers_inplace(nf, impl_->G);
    plan_id = tr.plan_id_(kind0_plan_key_(nf));
    return true;
}

std::vector<uint8_t> TwirlRecordSampler::family_relabel(const Pauli& P) const {
    const Impl& I = *impl_;
    if (P.n != I.n) throw std::invalid_argument("family_relabel: Pauli width differs from the sampler's");
    std::vector<uint64_t> bits((size_t)I.CW, 0);
    for (int q = 0; q < I.n; ++q) {
        if (P.xbit(q)) for (int w = 0; w < I.CW; ++w) bits[(size_t)w] ^= I.pxrow[(size_t)q * I.CW + w];
        if (P.zbit(q)) for (int w = 0; w < I.CW; ++w) bits[(size_t)w] ^= I.pzrow[(size_t)q * I.CW + w];
    }
    std::vector<uint8_t> t;
    for (int c = 0; c < I.nchan; ++c) t.push_back((uint8_t)((bits[(size_t)(c >> 6)] >> (c & 63)) & 1));
    for (size_t e = (size_t)I.nchan; e < I.rec_cols_.size(); ++e)
        t.push_back((uint8_t)Pauli::anticommute_bit(P, I.rec_cols_[e].M));
    return t;
}


// ── J-a (N3): the (G) builder and the dual-run surface ───────────────────────────────────────────
void TwirlRecordSampler::family_general_table(const std::vector<uint64_t>& S, RecTable& out,
                                              std::vector<int>* intro_b) {
    TermResidual& tr = impl_->fam_term_route_();
    if ((int)S.size() != tr.term_words())
        throw std::invalid_argument("family_general_table: S must have term_words() words");
    build_record_table_general(impl_->G, impl_->bare, tr, S, impl_->rec_cols_, out, intro_b);
}

void TwirlRecordSampler::family_general_table_diag(const uint8_t* plan_key, int n_plan, RecTable& out,
                                                   std::vector<int>* intro_b) const {
    const DiagNormalForm nf = impl_->parse_plan_key(plan_key, n_plan);
    const TermResidual tv(impl_->n, diag_plan_axes(nf, impl_->n));
    std::vector<uint64_t> S((size_t)tv.term_words(), 0);
    for (int j = 0; j < tv.num_terms(); ++j) S[(size_t)(j >> 6)] |= 1ULL << (j & 63);
    build_record_table_general(impl_->G, impl_->bare, tv, S, impl_->rec_cols_, out, intro_b);
}

std::vector<double> TwirlRecordSampler::family_born_law(const RecTable& T) const {
    if (T.kappa > 20) throw std::invalid_argument("family_born_law: kappa > 20");
    const size_t NZ = (size_t)1 << T.kappa;
    std::vector<double> p(NZ, 0.0);
    BornTrie bt(impl_->bare);
    std::vector<uint8_t> z((size_t)T.kappa, 0);
    // depth-first over z prefixes of nonzero weight: p(z) = Π_j p(z_j | z_<j), p(0 | z_<j) = p0_from_scratch
    std::function<void(int, double, size_t)> rec = [&](int j, double w, size_t idx) {
        if (j == T.kappa) { p[idx] += w; return; }
        const double p0 = bt.p0_from_scratch(T.lam, z.data(), j);
        if (p0 > 0.0) { z[(size_t)j] = 0; rec(j + 1, w * p0, idx); }
        if (p0 < 1.0) { z[(size_t)j] = 1; rec(j + 1, w * (1.0 - p0), idx | ((size_t)1 << j)); z[(size_t)j] = 0; }
    };
    rec(0, 1.0, 0);
    return p;
}


// ── J-a (N5): retention for the whole family — layout-3 records, materialize, signs(), record classes ──────
void TwirlRecordSampler::set_family_record_sink(FamilyRecordSink* rs) {
    if (!impl_->fam_term_.p) throw std::logic_error("set_family_record_sink: not a family sampler (J-a: test API only)");
    impl_->fam_term_.p->rs = rs;
}
const std::vector<TwirlRecordSampler::FamilyRecordContent>& TwirlRecordSampler::family_record_contents() const {
    if (!impl_->fam_term_.p) throw std::logic_error("family_record_contents: not a family sampler (J-a: test API only)");
    return impl_->fam_term_.p->contents;
}

int TwirlRecordSampler::family_intern_content(int kind, const std::string& content) {
    if (!impl_->fam_term_.p) throw std::logic_error("family_intern_content: not a family sampler (J-a: test API only)");
    if (kind != 0 && kind != 2) throw std::invalid_argument("family_intern_content: kind must be 0 or 2");
    return impl_->fam_term_.p->content_handle_(kind, content);
}

namespace {
// A record's content parsed: kind 0 → the S^a·CZ view (a ‖ 0xFF ‖ cz u16-LE pairs, memo_key_'s format);
// kind 2 → the reduced term bitset (TW words, little endian bytes).
DiagNormalForm parse_kind0_content_(const std::string& c, int n) {
    if ((int)c.size() < n + 1 || (unsigned char)c[(size_t)n] != 0xFF || ((c.size() - (size_t)n - 1) % 4) != 0)
        throw std::invalid_argument("layout-3 record: malformed kind-0 content");
    DiagNormalForm nf;
    nf.prefix = Pauli(n);
    nf.a.assign((size_t)n, 0);
    nf.diagonal_class = true;
    for (int q = 0; q < n; ++q) nf.a[(size_t)q] = (uint8_t)c[(size_t)q];
    for (size_t i = (size_t)n + 1; i + 3 < c.size(); i += 4) {
        const int f = (unsigned char)c[i] | ((unsigned char)c[i + 1] << 8);
        const int g = (unsigned char)c[i + 2] | ((unsigned char)c[i + 3] << 8);
        nf.cz.push_back({f, g});
    }
    return nf;
}
std::vector<uint64_t> parse_kind2_content_(const std::string& c, int TW) {
    if ((int)c.size() != TW * (int)sizeof(uint64_t)) throw std::invalid_argument("layout-3 record: malformed kind-2 content");
    std::vector<uint64_t> S((size_t)TW, 0);
    std::memcpy(S.data(), c.data(), c.size());
    return S;
}
Pauli prec_pauli_(const uint64_t* prec, int n) {
    Pauli P(n);
    const size_t NW = P.x.size();
    std::copy(prec, prec + NW, P.x.begin());
    std::copy(prec + NW, prec + 2 * NW, P.z.begin());
    return P;
}
inline bool bitw(const uint64_t* w, int i) { return (w[i >> 6] >> (i & 63)) & 1ULL; }
}  // namespace

// The record table of a content (pivot and introducer columns are the law's: any builder of the same state
// gives them; kind 0 through the readers' memo (D), kind 2 through (G), memoized per handle).
// J-b1 P12: the diagonal plan cache's resident record table of a kind-0 content (nullptr: not resident / not built).
std::shared_ptr<const RecTable> TwirlRecordSampler::Impl::FamRoute::cached_rec_(const Impl& I, const DiagNormalForm& nf) {
#ifdef XTIM_MUTATE_FR_NO_CACHE_REC
    (void)I; (void)nf;
    return nullptr;                                    // mutation build only: every reader rebuilds the shot law
#else
    std::vector<uint64_t> key((size_t)I.KW, 0);
    for (int q = 0; q < I.n && q < (int)nf.a.size(); ++q)
        if (nf.a[(size_t)q] & 1) key[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    for (const auto& pr : nf.cz) {
        const auto u = std::lower_bound(I.upairs.begin(), I.upairs.end(), pr);
        if (u == I.upairs.end() || *u != pr) return nullptr;   // outside the pair universe: not a diagonal-cache key
        const size_t ui = (size_t)(u - I.upairs.begin());
        key[(size_t)I.NW + (ui >> 6)] |= 1ULL << (ui & 63);
    }
    const CachedPlan* p = I.cache->find(key.data(), I.lhash(key.data()));
    if (!p || !p->resident || !p->rec || !p->rec->built) return nullptr;
    return p->rec;
#endif
}

std::shared_ptr<const RecTable> TwirlRecordSampler::Impl::FamRoute::content_table_(
        const Impl& I, int h, DiagNormalForm* nf_out, std::vector<uint64_t>* S_out) {
    if (h < 0 || h >= (int)contents.size()) throw std::invalid_argument("layout-3 record: content handle out of range");
    const TwirlRecordSampler::FamilyRecordContent& C = contents[(size_t)h];
    if (C.kind == 0) {
        DiagNormalForm nf = parse_kind0_content_(C.content, I.n);
        // J-b1 P12 (fixplan-p12-frames F3): the sampler's own resident plan first — its record table IS this
        // content's table (same (a, cz), P = I) — so a reader never rebuilds the shot law of a plan the run built.
        std::shared_ptr<const RecTable> T = cached_rec_(I, nf);
        if (!T) {
            ++stats.reader_table_rebuilds;           // evicted / never-run plan: the reader memo builds it (counted)
            T = I.rec_table_memo_(nf);
        }
        if (nf_out) *nf_out = std::move(nf);
        return T;
    }
    TermResidual& tr = term_();
    std::vector<uint64_t> S = parse_kind2_content_(C.content, tr.term_words());
    // m10: through the readers' memo. The tag byte 0x02 cannot open a kind-0 memo key (a kind-0 key starts with
    // an a-bit, 0 or 1).
    const std::string key = std::string(1, '\x02') + C.content;
    I.reader_lookups_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(I.rec_memo_mu_);
        auto it = I.rec_memo_.find(key);
        if (it != I.rec_memo_.end()) {
            it->second.tick = ++I.reader_tick_;
            if (S_out) *S_out = std::move(S);
            return it->second.t;
        }
    }
    auto Tn = std::make_shared<RecTable>();
    build_record_table_general(I.G, I.bare, tr, S, I.rec_cols_, *Tn);
    std::shared_ptr<const RecTable> out;
    {
        std::lock_guard<std::mutex> lk(I.rec_memo_mu_);
        auto ins = I.rec_memo_.emplace(key, Impl::ReaderEntry{});
        Impl::ReaderEntry& e = ins.first->second;
        if (ins.second) {
            e.t = std::move(Tn);
#ifdef XTIM_MUTATE_M10_NO_CHARGE
            e.bytes = 0;                               // mutation build only (m10): the kind-2 table is not charged
#else
            e.bytes = rec_table_bytes(*e.t);
#endif
            I.reader_bytes_.fetch_add(e.bytes, std::memory_order_relaxed);
            const size_t rb = I.reader_bytes_.load(std::memory_order_relaxed);
            if (rb > I.reader_peak_) I.reader_peak_ = rb;
        }
        e.tick = ++I.reader_tick_;
        out = e.t;                                     // held while the trim runs: never its victim
        I.reader_trim_locked_(I.sampler_resident_pub_.load(std::memory_order_relaxed), &key);
    }
    if (S_out) *S_out = std::move(S);
    return out;
}

namespace {
// J-b1 P8: C0̂ — the plan's C0 = Π_{j∈S} rot(B_j) applied to the bare state as an AMPLITUDE MAP on the register
// F₀ ∪ E_S (design §2.10; branch_scaffold.hpp): each term image B̂_j = U† B_j U acts on the frame coordinates as the
// signed permutation i^phase (−1)^{par + z·x} |x ⊕ a⟩ (its X part on certified rows lies in E_S), and rot(B) =
// (1 + iB)/√2. The frame (U, eps) is untouched; the E_S rows join the free register. Exact in the amplitudes (no
// tableau update); a register wider than one packed key word is refused by name.
FramedSuperposition c0hat_state_(const FramedSuperposition& bare, const CertifiedGroupPlanes& G, PlanTermView v) {
    PlanScaffold sc(std::move(v), G);
    const std::vector<uint64_t>& es = sc.ensure_es_();
    const std::vector<TermImage>& imgs = sc.term_images_(bare);
    const int N = bare.n(), kk = bare.k();
    std::vector<uint8_t> is_free((size_t)(N > 0 ? N : 1), 0);
    for (int d = 0; d < kk; ++d) is_free[(size_t)bare.free[(size_t)d]] = 1;
    std::vector<int> nonfree;
    for (int q = 0; q < N; ++q) if (!is_free[(size_t)q]) nonfree.push_back(q);
    std::vector<int> erows;
    for (int a = 0; a < (int)G.gens.size(); ++a)
        if ((es[(size_t)(a >> 6)] >> (a & 63)) & 1ULL) erows.push_back(nonfree[(size_t)a]);
    const int K = kk + (int)erows.size();
    if (K > kAmpKeyBits)
        throw std::runtime_error("materialize (C0 hat): the register F0 + E_S has " + std::to_string(K) +
                                 " coordinates, above one packed key word (" + std::to_string(kAmpKeyBits) +
                                 ") — refused");
    // J-b1 F (fixplan-F-materialize-c0hat r2 item 5): the NAMED capacity refusal, BEFORE any work. Each term image
    // B̂_j maps the support S to S ∪ (S ⊕ a_j) (a_j: its X-part on the register F₀ ∪ E_S), so the support stays
    // inside S₀ + span{a_j}: |support| <= c·2^d, d = rank_GF(2){a_j} (<= |E_S| + |F₀|), c <= χ the number of cosets
    // of the span that the bare's keys S₀ meet. That bound, at the RESULT's charge per entry (24 B, the
    // FramedSuperposition entry), must fit the state budget (XTIM_MAX_STATE_BYTES, default 2 GiB): state_budget_precheck
    // refuses otherwise (BareStateCapacityError -> xtim.XtimCapacityError). Reached by the frame-preserving branch
    // export (ideal_fs / add_branch / branch_handles) and carried-branch-row reads; the value readers on branch-0
    // rows take the tableau route and never come here.
    std::vector<std::pair<AmpKey, AmpKey>> masks;        // (X-part a_j, Z-part) of each image on the register
    masks.reserve(imgs.size());
    for (const TermImage& t : imgs) {
        AmpKey am = 0, zm = 0;
        for (int d = 0; d < K; ++d) {
            if (t.x[(size_t)d]) am |= AmpKey(1) << d;
            if (t.z[(size_t)d]) zm |= AmpKey(1) << d;
        }
        masks.emplace_back(am, zm);
    }
    {
        AmpKey basis[64] = {};                           // GF(2) echelon basis of the X-parts, by leading bit
        int rank = 0;
        for (const auto& m : masks) {
            AmpKey v = m.first;
            for (int bit = 63; bit >= 0 && v; --bit) {
                if (!((v >> bit) & 1ULL)) continue;
                if (!basis[bit]) { basis[bit] = v; ++rank; v = 0; break; }
                v ^= basis[bit];
            }
        }
        // the support lies in S₀ + V (V = span{a_j}): its size is (the number of cosets of V that the bare's keys
        // S₀ meet) · 2^d — exact for the set, an upper bound for the support (amplitudes may cancel)
        std::vector<AmpKey> cos;
        cos.reserve(bare.entries().size());
        for (const auto& e : bare.entries()) {
            AmpKey v = e.first;
            for (int bit = 63; bit >= 0 && v; --bit)
                if (((v >> bit) & 1ULL) && basis[bit]) v ^= basis[bit];
            cos.push_back(v);
        }
        std::sort(cos.begin(), cos.end());
        const size_t ncos = (size_t)(std::unique(cos.begin(), cos.end()) - cos.begin());
        // ruling on fixplan-F-c0hat-capacity (B): the RESULT is charged (the StateAllocator charge of a
        // FramedSuperposition entry, 24 B); the two transient maps stay uncharged, as before r2
        constexpr double kPer = (double)(sizeof(AmpKey) + sizeof(std::complex<double>));
        const double b = (double)ncos * std::ldexp(1.0, rank) * kPer;
        const int64_t bytes = b >= 4.0e18 ? (int64_t)4000000000000000000LL : (int64_t)b;
        const std::string what = "materialize (C0 hat, the frame-preserving amplitude map): support bound c * 2^d = " +
                                 std::to_string(ncos) + " * 2^" + std::to_string(rank) +
                                 " entries (d = rank of the term images' X-parts on F0 + E_S, c = the cosets of their "
                                 "span met by the bare's chi = " + std::to_string(bare.entries().size()) +
                                 " keys; |E_S| = " + std::to_string(erows.size()) + " promoted certified rows)";
        state_budget_precheck(bytes, what.c_str());
    }
    static const std::complex<double> ipow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const double r2 = 1.0 / std::sqrt(2.0);
    std::unordered_map<AmpKey, std::complex<double>> cur, nxt;
    for (const auto& e : bare.entries()) cur[e.first] += e.second;
    for (size_t ti = 0; ti < imgs.size(); ++ti) {
        const TermImage& t = imgs[ti];
        const AmpKey am = masks[ti].first, zm = masks[ti].second;
#ifdef XTIM_MUTATE_C0HAT
        const std::complex<double> c = ipow[1];                        // mutation build only: the image's sign dropped
#else
        const std::complex<double> c = ipow[(1 + t.phase) & 3] * (t.par ? -1.0 : 1.0);   // i · i^phase · (−1)^par
#endif
        nxt.clear();
        for (const auto& kv : cur) {
            nxt[kv.first] += kv.second * r2;
            const double s = (__builtin_popcountll(kv.first & zm) & 1) ? -1.0 : 1.0;
            nxt[kv.first ^ am] += kv.second * (r2 * s) * c;
        }
        cur.clear();
        for (const auto& kv : nxt) if (kv.second != 0.0) cur.emplace(kv.first, kv.second);
    }
    FramedSuperposition st(N);
    st.U = bare.U;
    st.eps = bare.eps;
    st.free = bare.free;
    st.free.insert(st.free.end(), erows.begin(), erows.end());
    st.sync_alpha_k();
    std::vector<std::pair<AmpKey, std::complex<double>>> ent(cur.begin(), cur.end());
    std::sort(ent.begin(), ent.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    auto& E = st.entries();
    E.clear();
    for (const auto& kv : ent) E.push_back({kv.first, kv.second});
    return st;
}
}  // namespace

// J-b1 P9b (fixplan-p9b-frame review C-R3): Lemma R1's gate on the forced-record path. At every forced record of a
// materialize whose C0̂ promoted E_S rows into the register, the pivot measure_pauli takes (the first X-bit of
// U†MU off the CURRENT free set, E_S rows included) is compared with the pattern-only rule (off the free set minus
// the promoted E_S rows, i.e. the identity's rule): a difference is a plan-dependent pivot, so the branch's frame
// leaves the ideal one. Cold path (test API materialize only); the per-shot loop never reaches it.
static std::atomic<long> g_forced_pivot_nonpattern_{0};
long family_forced_pivot_nonpattern_count() { return g_forced_pivot_nonpattern_.load(std::memory_order_relaxed); }
static void count_forced_pivot_(const FramedSuperposition& st, const std::vector<uint8_t>& es_row, const Pauli& M) {
    const Pauli Q = st.U.dual_valid() ? st.U.conjugate(M) : st.U.dual_image(M);
    std::vector<uint8_t> fr((size_t)(st.n() > 0 ? st.n() : 1), 0);
    for (int f : st.free) fr[(size_t)f] = 1;
    int p_cur = -1, p_pat = -1;
    for (int a = 0; a < st.n() && (p_cur < 0 || p_pat < 0); ++a) {
        if (!Q.xbit(a)) continue;
        if (p_cur < 0 && !fr[(size_t)a]) p_cur = a;
        if (p_pat < 0 && (!fr[(size_t)a] || es_row[(size_t)a])) p_pat = a;
    }
    if (p_cur != p_pat) g_forced_pivot_nonpattern_.fetch_add(1, std::memory_order_relaxed);
}

FramedSuperposition TwirlRecordSampler::family_materialize(int h, const uint64_t* prec, const uint64_t* y,
                                                           bool tableau, int branch) const {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_materialize: not a family sampler (J-a: test API only)");
    if (h < 0 || h >= (int)F->contents.size()) throw std::invalid_argument("family_materialize: content handle out of range");
    const FamilyRecordContent& C = F->contents[(size_t)h];
    // J-b1 F (fixplan-F-materialize-c0hat r2 item 4): the term view is read only by kind-2 content (its axes); a
    // kind-0 read never arms it (the arming, TermResidual's merge + ker Φ, was the reader's one-time first-call cost).
    auto TR = [&I]() -> TermResidual& { return I.fam_term_route_(); };
    DiagNormalForm nf;
    std::vector<uint64_t> S;
    const std::shared_ptr<const RecTable> Tp = F->content_table_(I, h, &nf, &S);
    const RecTable& T = *Tp;
    const Pauli Prec = prec_pauli_(prec, I.n);
    // J-b1 P9b: a branch row's state is its plan on the BRANCH's consumer bare (same frame, its own α)
    if (branch < 0 || branch > (int)F->branches.size())
        throw std::invalid_argument("family_materialize: branch id out of range");
    if (branch && tableau) throw std::invalid_argument("family_materialize: branch rows have no tableau route");
    if (branch && !F->branches[(size_t)(branch - 1)]->resident) I.fam_branch_payload_(*F->branches[(size_t)(branch - 1)]);
    const FramedSuperposition& base = branch ? F->branches[(size_t)(branch - 1)]->st : I.bare;
    FramedSuperposition st = base;
    std::vector<uint8_t> es_row;                         // J-b1 P9b: the promoted E_S rows (empty: none)
#ifdef XTIM_MUTATE_ANCH_LAM
    // mutation build only (MUT-ANCH-LAM, the A2 hazard): the anchors forced in the BARE frame, the pivots on
    // their pulled-back records but each Born direction on its Λ_j (a coset representative, NOT a record), then
    // C0 and P_rec applied.
    {
        auto pull = [&](const Pauli& M) { return C.kind == 0 ? diag_pullback(nf, M) : term_pullback(TR(), S, M); };
        for (int k = 0; k < T.rho; ++k) {
            const int c = T.pivot_rec[(size_t)k];
            const int v = (int)bitw(y, c) ^ Pauli::anticommute_bit(Prec, I.rec_cols_[(size_t)c].M);
            (void)st.measure_pauli(pull(I.rec_cols_[(size_t)c].M), v ? std::nextafter(1.0, 0.0) : 0.0);
        }
        for (int j = 0; j < T.kappa; ++j) {
            const int c = T.born_rec[(size_t)j];
            const int v = (int)bitw(y, c) ^ Pauli::anticommute_bit(Prec, I.rec_cols_[(size_t)c].M);
            // Λ_j as the FREE-REGISTER Pauli U·(i^phase (−1)^par X^a Z^b ⊗ I)·U† (derive_kernel_logical before its
            // G.reduce): it differs from the record by signed certified rows, which a pivot projection unfixes (A2)
            const FreeDir& d = T.lam[(size_t)j];
            Pauli L(I.n);
            int par_nf = d.par;
            for (int q = 0; q < I.bare.k(); ++q)
                if (d.b[(size_t)q] && I.bare.eps[(size_t)I.bare.free[(size_t)q]]) par_nf ^= 1;
            L.phase = (d.phase + 2 * par_nf) & 3;
            for (int q = 0; q < I.bare.k(); ++q)
                if (d.a[(size_t)q]) L = Pauli::multiply(L, I.bare.U.Xrow[(size_t)I.bare.free[(size_t)q]]);
            for (int q = 0; q < I.bare.k(); ++q)
                if (d.b[(size_t)q]) L = Pauli::multiply(L, I.bare.U.Zrow[(size_t)I.bare.free[(size_t)q]]);
            (void)st.measure_pauli(L, v ? std::nextafter(1.0, 0.0) : 0.0);
        }
        if (C.kind == 0) { nf.prefix = Prec; I.apply_residual_gates_(st, nf); }
        else {
            for (int j = 0; j < TR().num_terms(); ++j) if (bitw(S.data(), j)) apply_rot_(st, TR().axis(j));
            for (int q = 0; q < I.n; ++q) {
                if (Prec.xbit(q)) st.apply_clifford(/*X*/ 3, q, 0);
                if (Prec.zbit(q)) st.apply_clifford(/*Z*/ 5, q, 0);
            }
        }
        return st;
    }
#endif
    // 1. C0 then P_rec. J-b1 P8: C0̂ by default (the amplitude map on F₀ ∪ E_S, the frame untouched), then P_rec on the
    //    frame as Pauli gates. tableau: the gate-level route — kind 0 the S^a·CZ view and its prefix (S2b's
    //    application), kind 2 the term rotations Π_{j∈S} rot(B_j) on the frame tableau.
    if (!tableau) {
        PlanTermView v;
        v.kind = C.kind;
        v.key = C.content;
        if (C.kind == 0) {
            v.axes = diag_plan_axes(nf, I.n);
            v.S.assign((v.axes.size() + 63) / 64, 0);
            for (size_t j = 0; j < v.axes.size(); ++j) v.S[j >> 6] |= 1ULL << (j & 63);
        } else {
            for (int j = 0; j < TR().num_terms(); ++j) v.axes.push_back(TR().axis(j));
            v.S = S;
        }
        st = c0hat_state_(base, I.G, std::move(v));
        if (st.k() > I.bare.k()) {                       // the promoted E_S rows (the R1 forced-pivot counter)
            es_row.assign((size_t)I.n, 0);
            for (int f : st.free) es_row[(size_t)f] = 1;
            for (int f : I.bare.free) es_row[(size_t)f] = 0;
        }
        for (int q = 0; q < I.n; ++q) {
            if (Prec.xbit(q)) st.apply_clifford(/*X*/ 3, q, 0);
            if (Prec.zbit(q)) st.apply_clifford(/*Z*/ 5, q, 0);
        }
    } else if (C.kind == 0) {
        nf.prefix = Prec;
        I.apply_residual_gates_(st, nf);
    } else {
        for (int j = 0; j < TR().num_terms(); ++j)
            if (bitw(S.data(), j)) {
#ifdef XTIM_MUTATE_RET_ROT
                Pauli B = TR().axis(j);                    // mutation build only (MUT-RET-ROT): rot(B_j, −1)
                B.phase = (B.phase + 2) & 3;
                apply_rot_(st, B);
#else
                apply_rot_(st, TR().axis(j));
#endif
            }
        for (int q = 0; q < I.n; ++q) {
            if (Prec.xbit(q)) st.apply_clifford(/*X*/ 3, q, 0);
            if (Prec.zbit(q)) st.apply_clifford(/*Z*/ 5, q, 0);
        }
    }
    // 2. every pivot record at its y, then one introducer record per Born direction (never Λ_j: A2) — S2b's
    //    force primitive and named throw.
    for (int k = 0; k < T.rho; ++k) {
        if (!es_row.empty()) count_forced_pivot_(st, es_row, I.rec_cols_[(size_t)T.pivot_rec[(size_t)k]].M);
        I.force_record_(st, T.pivot_rec[(size_t)k], (uint8_t)bitw(y, T.pivot_rec[(size_t)k]), "anchor (pivot)");
    }
    for (int j = 0; j < T.kappa; ++j) {
        uint8_t yj = (uint8_t)bitw(y, T.born_rec[(size_t)j]);
#ifdef XTIM_MUTATE_MS_BORN_FLIP
        yj ^= 1;                                         // mutation build only (M-ANCH, reused for the family)
#endif
        if (!es_row.empty()) count_forced_pivot_(st, es_row, I.rec_cols_[(size_t)T.born_rec[(size_t)j]].M);
        I.force_record_(st, T.born_rec[(size_t)j], yj, "anchor (direction)");
    }
    return st;
}

__attribute__((noinline, cold)) TwirlRecordSampler::FamilyPlanView TwirlRecordSampler::family_plan_view(int h) const {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_plan_view: not a family sampler (test API only)");
    if (h < 0 || h >= (int)F->contents.size()) throw std::invalid_argument("family_plan_view: content handle out of range");
    const FamilyRecordContent& C = F->contents[(size_t)h];
    DiagNormalForm nf;
    std::vector<uint64_t> S;
    const std::shared_ptr<const RecTable> Tp = F->content_table_(I, h, &nf, &S);
    FamilyPlanView v;
    v.kind = C.kind;
    v.content = C.content;
    v.digest_hi = C.digest_hi;
    v.digest_lo = C.digest_lo;
    // J-b1 P12 (fixplan-p12-frames F3): no heap Paulis for the term view — the export builds its shared axis table
    // from (a, cz) for kind 0 and from the term view's axes for kind 2 (S here).
    if (C.kind == 0) {
        v.a.assign((size_t)I.n, 0);
        for (int q = 0; q < I.n && q < (int)nf.a.size(); ++q) v.a[(size_t)q] = nf.a[(size_t)q] & 1;
        v.cz = nf.cz;
    } else {
        v.S = S;
    }
    v.rho = Tp->rho;
    v.kappa = Tp->kappa;
    for (const FreeDir& d : Tp->lam) {               // U·(i^phase (−1)^par_nf X^a Z^b)·U† (derive_kernel_logical's L)
        Pauli L(I.n);
        int par_nf = d.par;
        for (int q = 0; q < I.bare.k(); ++q)
            if (d.b[(size_t)q] && I.bare.eps[(size_t)I.bare.free[(size_t)q]]) par_nf ^= 1;
        L.phase = (d.phase + 2 * par_nf) & 3;
        for (int q = 0; q < I.bare.k(); ++q)
            if (d.a[(size_t)q]) L = Pauli::multiply(L, I.bare.U.Xrow[(size_t)I.bare.free[(size_t)q]]);
        for (int q = 0; q < I.bare.k(); ++q)
            if (d.b[(size_t)q]) L = Pauli::multiply(L, I.bare.U.Zrow[(size_t)I.bare.free[(size_t)q]]);
        v.born.push_back(std::move(L));
    }
    return v;
}

// ── J-b1 P9b: the consumer branch law (E_min = ∅ branches) ─────────────────────────────────────────────────────
// record_rows_ for a branch shot: the plan's emission block and coins as branch 0's (the plans never read α), the
// Born directions walked on the BRANCH's trie over its own amplitudes (the sparse (plan, branch) map).
__attribute__((noinline)) bool TwirlRecordSampler::Impl::fam_record_rows_branch_(const CachedPlan& plan, uint64_t* bits,
                                                                              FamBranch& B) {
    ensure_rec_(plan);
    const bool EX = nextra_ != 0;
    const bool keep = fam_term_.p && fam_term_.p->rs;
    const int cw = std::max(CW, 1);
    const int rho = plan.emit_rho, kap = plan.emit_kappa, nr = 1 + rho + kap;
    const uint64_t* E = plan.emit.data();
    const uint64_t* EX0 = E + (size_t)nr * cw;
    for (int w = 0; w < CW; ++w) bits[w] ^= E[w];
    if (EX) for (int w = 0; w < XW; ++w) xbits_[(size_t)w] ^= EX0[w];
    rec_uz_.clear();
    for (int k = 0; k < rho; ++k) {
        const bool bit = coins_.next();
        if (keep) rec_uz_.push_back((uint8_t)bit);
        if (bit) {
            const uint64_t* rc = E + (size_t)(1 + k) * cw;
            for (int w = 0; w < CW; ++w) bits[w] ^= rc[w];
            if (EX) {
                const uint64_t* rx = EX0 + (size_t)(1 + k) * XW;
                for (int w = 0; w < XW; ++w) xbits_[(size_t)w] ^= rx[w];
            }
        }
    }
    if (kap > 0) {
        const RecTable& T = *plan.rec;
        auto& slot = B.tries[&plan];
        if (slot.first != plan.rec.get()) { slot.first = plan.rec.get(); slot.second.clear(); }
        const size_t z0 = rec_uz_.size();
#ifdef XTIM_MUTATE_BR_TRIE_SHARED
        const int bad = trie_.walk(T.lam, slot.second, 0, born_at_shot_(), rec_uz_);   // mutation build only
#else
        B.trie->root_memo_verify = trie_.root_memo_verify;   // S8-4: the same selfcheck window as branch 0's trie
        const int bad = B.trie->walk(T.lam, slot.second, 0, born_at_shot_(), rec_uz_);
#endif
        if (bad >= 0) throw std::logic_error("record_rows_ (branch): a non-Hermitian record direction (engine bug)");
        for (int j = 0; j < kap; ++j)
            if (rec_uz_[z0 + (size_t)j]) {
                const uint64_t* bc = E + (size_t)(1 + rho + j) * cw;
                for (int w = 0; w < CW; ++w) bits[w] ^= bc[w];
                if (EX) {
                    const uint64_t* bx = EX0 + (size_t)(1 + rho + j) * XW;
                    for (int w = 0; w < XW; ++w) xbits_[(size_t)w] ^= bx[w];
                }
            }
    }
    rec_uz_width_ = keep ? rho + kap : -1;
    rec_route_ = true;
    return true;
}

__attribute__((noinline, cold)) int TwirlRecordSampler::family_add_branch(const FramedSuperposition& st, const Pauli& dspec,
                                                                         const std::vector<uint8_t>& dport,
                                                                         std::function<FramedSuperposition()> resupply) {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_add_branch: not a family sampler (test API only)");
    if (!F->frames) throw std::invalid_argument("family_add_branch: the consumer branch law needs the frames class");
    // The branch's consumer bare must be on branch 0's frame EXACTLY (rows, signs, free rows): its plans are branch
    // 0's. A mismatch is a construction defect of the caller (an engine bug), never a silent law.
    const FramedSuperposition& b0 = I.bare;
    bool same = st.n() == b0.n() && st.free == b0.free && st.eps == b0.eps;
    for (int a = 0; same && a < b0.n(); ++a) {
        const Pauli &z0 = b0.U.Zrow[(size_t)a], &z1 = st.U.Zrow[(size_t)a];
        const Pauli &x0 = b0.U.Xrow[(size_t)a], &x1 = st.U.Xrow[(size_t)a];
        same = z0.x == z1.x && z0.z == z1.z && ((z0.phase - z1.phase) & 3) == 0 && x0.x == x1.x && x0.z == x1.z &&
               ((x0.phase - x1.phase) & 3) == 0;
    }
    if (!same) throw std::logic_error("family_add_branch: the branch's consumer bare is not on branch 0's frame (engine bug)");
    if (dspec.n != b0.n() || (int)dport.size() != 2 * F->pnq)
        throw std::invalid_argument("family_add_branch: eps-delta shapes (dspec width / dport length)");
    auto B = std::make_unique<Impl::FamBranch>();
    B->id = (int)F->branches.size() + 1;
    B->st = st;
    B->trie.reset(new BornTrie(B->st));
    B->dspec = dspec;
    B->dport = dport;
    B->resupply = std::move(resupply);
    F->branches.push_back(std::move(B));
    F->brow.assign((size_t)(2 * F->pnq), 0);
    return F->branches.back()->id;
}
// D6: make B's payload resident (re-supply on a miss: the pure function add_branch validated; inside run()), then
// evict the least recently used OTHER resident branches while the resident bytes exceed the budget.
__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::fam_branch_payload_(FamBranch& B) {
    FamRoute& F = *fam_term_.p;
    B.last_use = ++F.branch_tick;
    if (!B.resident) {
        if (!B.resupply) throw std::logic_error("branch payload evicted without a re-supply (engine bug)");
#ifdef XTIM_MUTATE_BR_RESUPPLY_STALE
        B.st = bare;                                // mutation build only (MUT-BR-RESUPPLY-STALE): branch 0's amplitudes
#else
        B.st = B.resupply();
#endif
        B.trie.reset(new BornTrie(B.st));
        B.resident = true;
        ++F.branch_resupplies;
    }
    for (;;) {
        size_t tot = 0;
        FamBranch* lru = nullptr;
        for (auto& up : F.branches) {
            if (!up->resident) continue;
            tot += up->bytes_();
            if (up.get() != &B && up->resupply && (!lru || up->last_use < lru->last_use)) lru = up.get();
        }
        if (tot <= F.branch_budget || !lru) break;
        lru->trie.reset();                          // the trie references st: dropped first
        lru->tries.clear();
        lru->st = FramedSuperposition(0);
        lru->resident = false;
        ++F.branch_evictions;
    }
}
void TwirlRecordSampler::family_set_branch_budget(size_t bytes) {
    Impl::FamRoute* F = impl_->fam_term_.p;
    if (!F) throw std::logic_error("family_set_branch_budget: not a family sampler (test API only)");
    F->branch_budget = bytes;
}
void TwirlRecordSampler::family_evict_branches() {
    Impl::FamRoute* F = impl_->fam_term_.p;
    if (!F) throw std::logic_error("family_evict_branches: not a family sampler (test API only)");
    for (auto& up : F->branches)
        if (up->resident && up->resupply) {
            up->trie.reset();
            up->tries.clear();
            up->st = FramedSuperposition(0);
            up->resident = false;
            ++F->branch_evictions;
        }
}
void TwirlRecordSampler::family_set_input_branch(const int32_t* ids, long n) {
    Impl::FamRoute* F = impl_->fam_term_.p;
    if (!F) throw std::logic_error("family_set_input_branch: not a family sampler (test API only)");
    if (ids) {
        if (!F->frames) throw std::invalid_argument("input_branch: the consumer branch law needs the frames class");
        for (long i = 0; i < n; ++i)
            if (ids[i] < 0 || ids[i] > (int32_t)F->branches.size())
                throw std::invalid_argument("input_branch: an engine branch id out of range");
    }
    F->branch_ids = ids;
    F->branch_n = ids ? n : 0;
}
TwirlRecordSampler::FamilyBranchCounters TwirlRecordSampler::family_branch_counters() const {
    FamilyBranchCounters c;
    const Impl::FamRoute* F = impl_->fam_term_.p;
    if (!F) return c;
    c.branches = (long)F->branches.size();
    c.map_probes = F->branch_probes;
    c.evictions = F->branch_evictions;
    c.resupplies = F->branch_resupplies;
    for (const auto& B : F->branches) {
        c.shots += B->shots;
        if (!B->resident) continue;
        ++c.resident;
        c.resident_bytes += B->bytes_();
        c.tries += (long)B->tries.size();
        c.trie_nodes += B->trie->nodes;
        c.trie_node_evals += B->trie->node_evals;
    }
    return c;
}

__attribute__((noinline, cold)) const std::vector<Pauli>& TwirlRecordSampler::family_pivot_frames(int h) const {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_pivot_frames: not a family sampler (test API only)");
    if (h < 0 || h >= (int)F->contents.size()) throw std::invalid_argument("family_pivot_frames: content handle out of range");
    auto it = F->pivot_w.find(h);
    if (it != F->pivot_w.end()) return it->second;
    const FamilyRecordContent& C = F->contents[(size_t)h];
    DiagNormalForm nf;
    std::vector<uint64_t> S;
    const std::shared_ptr<const RecTable> Tp = F->content_table_(I, h, &nf, &S);
    PlanTermView v;
    v.kind = C.kind;
    v.key = C.content;
    std::vector<Pauli> Q;
    if (C.kind == 0) {
        DiagNormalForm nf0 = nf;
        nf0.prefix = Pauli(I.n);                    // the content's C0 (P = I)
        v.axes = diag_plan_axes(nf0, I.n);
        v.S.assign((v.axes.size() + 63) / 64, 0);
        for (size_t j = 0; j < v.axes.size(); ++j) v.S[j >> 6] |= 1ULL << (j & 63);
        for (int c : Tp->pivot_rec) Q.push_back(diag_pullback(nf0, I.rec_cols_[(size_t)c].M));
    } else {
        TermResidual& tr = I.fam_term_route_();
        for (int j = 0; j < tr.num_terms(); ++j) v.axes.push_back(tr.axis(j));
        v.S = S;
        for (int c : Tp->pivot_rec) Q.push_back(term_pullback(tr, S, I.rec_cols_[(size_t)c].M));
    }
    PlanScaffold sc(std::move(v), I.G);
    return F->pivot_w.emplace(h, sc.pivot_frames_(Q)).first->second;
}

std::shared_ptr<const RecTable> TwirlRecordSampler::family_content_table(int h) const {
    Impl& I = *impl_;
    if (!I.fam_term_.p) throw std::logic_error("family_content_table: not a family sampler (test API only)");
    return I.fam_term_.p->content_table_(I, h, nullptr, nullptr);
}

std::vector<Pauli> TwirlRecordSampler::family_record_ops() const {
    std::vector<Pauli> out;
    for (const RecColumn& c : impl_->rec_cols_) out.push_back(c.M);
    return out;
}

// J-b1 P7: the layout-3 record fidelity (record_fidelity.hpp, record mode). The content's C0 per handle: kind 0 =
// S^{a&1}·CZ (diag_pullback, materialize's own pullback), kind 2 =
// rot(S) = Π_{j∈S} exp(iπ/4 B_j) through the term view (term_pullback).
void TwirlRecordSampler::family_record_fidelity(RecordFidelity& rf, const FamilyRecordSink& rs, long rows,
                                                double* out, const uint8_t* corr,
                                                const std::vector<int>* out_wires) {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_record_fidelity: not a family sampler (test API only)");
    if (rs.PNW != (I.n + 63) / 64 || rs.YW != I.QCW_)
        throw std::invalid_argument("family_record_fidelity: record shape mismatch");
    if (corr && !out_wires) throw std::logic_error("family_record_fidelity: a correction needs the port (internal)");
    const double r2 = 1.0 / std::sqrt(2.0);
    std::vector<int> id_of((size_t)F->contents.size(), -1);
    const int no = out_wires ? (int)out_wires->size() : 0;
    std::vector<uint64_t> cq(corr ? (size_t)2 * rs.PNW : 0);
    for (long i = 0; i < rows; ++i) {
        const int h = rs.handle[i];
        if (h < 0 || h >= (int)F->contents.size())
            throw std::invalid_argument("family_record_fidelity: content handle out of range");
        const uint64_t* pr = rs.prec + (size_t)i * 2 * (size_t)rs.PNW;
        const uint64_t* yr = rs.y + (size_t)i * (size_t)rs.YW;
        if (corr) {                          // the decoder's correction P_c on OUTPUT_QUBITS (a factor of the prefix)
            const uint8_t* cr = corr + (size_t)i * 2 * (size_t)no;
            std::fill(cq.begin(), cq.end(), 0);
#ifndef XTIM_MUTATE_RF_CORR_DROP
            for (int j = 0; j < no; ++j) {
                const int w = (*out_wires)[(size_t)j];
                if (cr[j] & 1) cq[(size_t)(w >> 6)] |= 1ULL << (w & 63);
                if (cr[no + j] & 1) cq[(size_t)rs.PNW + (size_t)(w >> 6)] |= 1ULL << (w & 63);
            }
#endif                                       // (mutation build only: the correction dropped)
        }
        if (id_of[(size_t)h] < 0) {
            const FamilyRecordContent& C = F->contents[(size_t)h];
            const std::string key = std::to_string(C.kind) + ":" + C.content;
            int id = -1;
            if (!rf.has_content(key, &id)) {
                std::vector<RecordFidelity::Factor> fac;
                auto I1 = [&]() { return Pauli(I.n); };
                if (C.kind == 0) {
                    id = rf.intern_content_diag(key, parse_kind0_content_(C.content, I.n));   // C0 = S^{a&1}·CZ
                } else {
                    TermResidual& tr = I.fam_term_route_();
                    const std::vector<uint64_t> S = parse_kind2_content_(C.content, tr.term_words());
                    auto pull = [&](const Pauli& Q) { return term_pullback(tr, S, Q); };
                    for (int j = 0; j < tr.num_terms(); ++j)
                        if ((S[(size_t)(j >> 6)] >> (j & 63)) & 1ULL)   // rot(B) = (1 + iB)/√2
                            fac.push_back({{r2, I1()}, {std::complex<double>(0, r2), tr.axis(j)}});
                    id = rf.intern_content(key, pull, fac);
                }
            }
            id_of[(size_t)h] = id;
        }
#ifdef XTIM_MUTATE_RF_L3
        out[i] = rf.evaluate_record(id_of[(size_t)h], pr + rs.PNW, pr, yr,   // mutation build only: x / z swapped
                                    corr ? cq.data() : nullptr, corr ? cq.data() + rs.PNW : nullptr);
#else
        out[i] = rf.evaluate_record(id_of[(size_t)h], pr, pr + rs.PNW, yr, corr ? cq.data() : nullptr,
                                    corr ? cq.data() + rs.PNW : nullptr);
#endif
    }
}

void TwirlRecordSampler::family_signs(const FamilyRecordSink& rs, long rows, const std::vector<Pauli>& g,
                                      int8_t* out) const {
    Impl& I = *impl_;
    Impl::FamRoute* F = I.fam_term_.p;
    if (!F) throw std::logic_error("family_signs: not a family sampler (J-a: test API only)");
    for (const Pauli& p : g) {
        if (p.n != I.n) throw std::invalid_argument("family_signs: a Pauli on the wrong number of qubits");
        int ov = 0;
        for (size_t w = 0; w < p.x.size(); ++w) ov += __builtin_popcountll(p.x[w] & p.z[w]);
        if (((p.phase - ov) & 1) != 0) throw std::invalid_argument("family_signs: a non-Hermitian Pauli");
    }
    if (rs.PNW != (I.n + 63) / 64 || rs.YW != I.QCW_) throw std::invalid_argument("family_signs: record shape mismatch");
    TermResidual& tr = I.fam_term_route_();
    const int m = (int)I.rec_cols_.size(), ng = (int)g.size();
    // H's record set: every record column (MUT-SG-NODEC: the decision records dropped).
    std::vector<uint8_t> use((size_t)m, 1);
#ifdef XTIM_MUTATE_SG_NODEC
    for (int c = 0; c < m; ++c) if ((I.rec_cols_[(size_t)c].order >> 24) == 2) use[(size_t)c] = 0;
#endif
    struct Rules { std::vector<SignRule> r; std::vector<uint8_t> undressed; };
    std::unordered_map<int, Rules> rules;               // per content handle (cold per distinct plan)
    std::vector<uint64_t> t((size_t)I.QCW_, 0);
    for (long i = 0; i < rows; ++i) {
        const int h = rs.handle[i];
        auto it = rules.find(h);
        if (it == rules.end()) {
            if (h < 0 || h >= (int)F->contents.size()) throw std::invalid_argument("family_signs: content handle out of range");
            const FamilyRecordContent& C = F->contents[(size_t)h];
            DiagNormalForm nf;
            std::vector<uint64_t> S;
            if (C.kind == 0) nf = parse_kind0_content_(C.content, I.n);
            else S = parse_kind2_content_(C.content, tr.term_words());
            auto pull = [&](const Pauli& M) { return C.kind == 0 ? diag_pullback(nf, M) : term_pullback(tr, S, M); };
            std::vector<Pauli> Q;
            Q.reserve((size_t)m);
            for (int c = 0; c < m; ++c) Q.push_back(pull(I.rec_cols_[(size_t)c].M));
            SignRuleBuilder B(I.n, Q, use, I.gens);
            Rules R;
            for (const Pauli& p : g) {
                const Pauli gp = pull(p);
                R.r.push_back(B.rule(gp));
                R.undressed.push_back(gp.x == p.x && gp.z == p.z ? 1 : 0);
            }
            it = rules.emplace(h, std::move(R)).first;
        }
        const Rules& R = it->second;
        const uint64_t* px = rs.prec + (size_t)i * 2 * (size_t)rs.PNW;
        const uint64_t* pz = px + rs.PNW;
        const uint64_t* yw = rs.y + (size_t)i * (size_t)rs.YW;
        // t_c = ⟨P_rec, M_c⟩ over the record columns (the per-wire column rows, record_values_words_'s t(P))
        std::fill(t.begin(), t.end(), 0);
#ifndef XTIM_MUTATE_SG_TC
        for (int q = 0; q < I.n; ++q) {
            if (bitw(px, q)) { const uint64_t* r = &I.qcol_x_[(size_t)q * I.QCW_]; for (int w = 0; w < I.QCW_; ++w) t[(size_t)w] ^= r[w]; }
            if (bitw(pz, q)) { const uint64_t* r = &I.qcol_z_[(size_t)q * I.QCW_]; for (int w = 0; w < I.QCW_; ++w) t[(size_t)w] ^= r[w]; }
        }
#endif
        for (int k = 0; k < ng; ++k) {
            const SignRule& sr = R.r[(size_t)k];
            int8_t v = 0;
#ifdef XTIM_MUTATE_SG_UNDRESSED
            const bool definite = R.undressed[(size_t)k] != 0;  // mutation build only: the withdrawn round-4 rule
#else
            const bool definite = sr.definite;
#endif
            if (definite) {
                int bit = 0;
                if (sr.definite) {
                    bit = sr.eps;
                    for (int w = 0; w < I.QCW_ && (size_t)w < sr.U.size(); ++w)
                        bit ^= __builtin_popcountll(sr.U[(size_t)w] & (yw[w] ^ t[(size_t)w])) & 1;
                }
                const Pauli& p = g[(size_t)k];
                int an = 0;
                for (int w = 0; w < rs.PNW; ++w)
                    an ^= __builtin_popcountll((px[w] & p.z[(size_t)w]) ^ (pz[w] & p.x[(size_t)w])) & 1;
                bit ^= an;
                v = bit ? -1 : 1;
            }
            out[(size_t)i * (size_t)ng + (size_t)k] = v;
        }
    }
}

std::vector<int> TwirlRecordSampler::family_record_classes() const {
    Impl& I = *impl_;
    if (!I.fam_term_.p) throw std::logic_error("family_record_classes: not a family sampler (J-a: test API only)");
    DiagNormalForm nf;                                  // R = I: the identity plan
    nf.prefix = Pauli(I.n);
    nf.a.assign((size_t)I.n, 0);
    nf.diagonal_class = true;
    const std::shared_ptr<const RecTable> T = I.rec_table_memo_(nf);
    std::vector<int> cls((size_t)T->m, 0);
    for (int c = 0; c < T->m; ++c) {
        bool born = false, coin = false;
        for (const auto& r : T->born_row) born = born || bitw(r.data(), c);
        for (const auto& r : T->coin_row) coin = coin || bitw(r.data(), c);
        cls[(size_t)c] = born ? 2 : coin ? 1 : 0;
    }
    return cls;
}

std::pair<long, long> TwirlRecordSampler::family_alt_atoms_oracle() const {
    const Impl& I = *impl_;
    long checked = 0, bad = 0;
    for (size_t ai = 0; ai < I.sides.size(); ++ai) {
        if (I.alt_is_ppr[ai]) continue;
        DiagPauliClifford D = compose_fired(I.table, I.sides[ai]);
        D.finalize();
        std::vector<std::pair<int, uint8_t>> av;
        for (int q : D.cache->active_a) av.push_back({q, (uint8_t)(D.a[(size_t)q] & 3)});
        std::vector<int> vq;
        for (int q = 0; q < I.n; ++q) if (D.v[(size_t)q]) vq.push_back(q);
        const auto& A = I.alt_atoms[ai];
        ++checked;
        if (av != A.av || D.cache->b_pairs != A.czp || vq != A.vq) ++bad;
    }
    return {checked, bad};
}

bool TwirlRecordSampler::Impl::fam_has_branches_() const { return fam_term_.p && !fam_term_.p->branches.empty(); }
size_t TwirlRecordSampler::Impl::born_memo_bytes_() const {
    size_t b = trie_.root_memo_bytes();
    if (fam_term_.p)
        for (const auto& B : fam_term_.p->branches) if (B->trie) b += B->trie->root_memo_bytes();
    return b;
}
void TwirlRecordSampler::Impl::born_memo_clear_() {
    trie_.root_memo_clear();
    if (fam_term_.p)
        for (auto& B : fam_term_.p->branches) if (B->trie) B->trie->root_memo_clear();
}

std::string TwirlRecordSampler::family_ctor_bytes() const {
    const Impl& I = *impl_;
    std::string k;
    auto put = [&](const void* p, size_t b) { k.append(reinterpret_cast<const char*>(p), b); };
    auto put_i = [&](int64_t v) { put(&v, sizeof v); };
    auto put_w = [&](const std::vector<uint64_t>& v) { put_i((int64_t)v.size()); put(v.data(), v.size() * 8); };
    auto put_b = [&](const std::vector<uint8_t>& v) { put_i((int64_t)v.size()); put(v.data(), v.size()); };
    auto put_pauli = [&](const Pauli& P) { put_w(P.x); put_w(P.z); put_i(P.phase & 3); };
    auto put_dpc = [&](const DiagPauliClifford& D) {
        put_i(D.n); put_i(D.gamma.is_zero); put_i(D.gamma.scale); put_i(D.gamma.z8);
        put_b(D.a); put_b(D.v); put_i(D.B.dim()); put_i(D.B.words()); put_w(D.B.data_);
    };
    const FramedSuperposition& B = I.bare;                    // the bare state (as family_model_key)
    put_i(I.n); put_i(B.n()); put_i(B.k());
    for (int a = 0; a < B.n(); ++a) { put_pauli(B.U.Zrow[(size_t)a]); put_pauli(B.U.Xrow[(size_t)a]); }
    for (size_t a = 0; a < B.eps.size(); ++a) put_i(B.eps[a] ? 1 : 0);
    for (int f : B.free) put_i(f);
    for (const auto& br : B.entries()) { put(&br.first, sizeof br.first); put(&br.second, sizeof br.second); }
    put_i((int64_t)I.gens.size());
    for (const Pauli& g : I.gens) put_pauli(g);
    const PropagationTable& T = I.table;
    put_i(T.n); put_i(T.all_in_class); put_i(T.has_general); put_i((int64_t)T.locations.size());
    for (const RejectInfo& r : T.rejects) { put_i(r.location_index); put_i(r.qubit); put_i((int)r.basis); put_i(r.reject_gate_index); }
    for (const LocationEntry& L : T.locations) {
        put_i(L.stream_index); put_i((int64_t)L.qubits.size());
        for (int q : L.qubits) put_i(q);
        for (const auto* atoms : {&L.x_atom, &L.z_atom})
            for (const PropAtom& A : *atoms) {
                put_i(A.present); put_i(A.rejected); put_i(A.reject_gate_index);
                put_dpc(A.c_prop);
                put_i(A.general != nullptr);
                if (A.general) for (int q = 0; q < A.general->n; ++q) { put_pauli(A.general->Xrow[(size_t)q]); put_pauli(A.general->Zrow[(size_t)q]); }
                put_i(A.ppr != nullptr);
                if (A.ppr) {
                    put_pauli(A.ppr->pauli);
                    for (const auto& r : A.ppr->rots) { put_pauli(r.first); put_i(r.second); }
                }
            }
    }
    put_i((int64_t)I.sides.size());
    for (const auto& fr : I.sides) {
        put_i((int64_t)fr.size());
        for (const FiredPauli& f : fr) { put_i(f.location_index); put_i(f.qubit); put_i((int)f.pauli); }
    }
    put_b(I.alt_is_ppr);
    for (const auto& A : I.alt_atoms) {
        put_i((int64_t)A.av.size()); for (const auto& e : A.av) { put_i(e.first); put_i(e.second); }
        put_i((int64_t)A.czp.size()); for (const auto& e : A.czp) { put_i(e.first); put_i(e.second); }
        put_i((int64_t)A.czidx.size()); for (int e : A.czidx) put_i(e);
        put_i((int64_t)A.vq.size()); for (int e : A.vq) put_i(e);
    }
    put_i((int64_t)I.upairs.size()); for (const auto& e : I.upairs) { put_i(e.first); put_i(e.second); }
    for (const auto& M : I.alt_masks) { put_w(M.smask); put_w(M.vmask); put_w(M.kmask); put_w(M.czk); put_w(M.zxrow); put_w(M.zzrow); }
    put_w(I.alt_blob);
    put_i((int64_t)I.alt_off.size()); put(I.alt_off.data(), I.alt_off.size() * sizeof(uint32_t));
    put_w(I.prodmask); put_w(I.upair_keep_); put_w(I.pxrow); put_w(I.pzrow); put_i((int64_t)I.chtok);
    put_i((int64_t)I.sch.size());
    for (const auto& c : I.sch) { put_pauli(c.GS); put_w(c.mask); }
    return k;
}

std::string TwirlRecordSampler::family_model_key() const {
    const Impl& I = *impl_;
    if (!I.fam_term_.p) throw std::logic_error("family_model_key: not a family sampler (J-a: test API only)");
    std::string k;
    auto put = [&](const void* p, size_t b) { k.append(reinterpret_cast<const char*>(p), b); };
    auto put_pauli = [&](const Pauli& P) {
        put(P.x.data(), P.x.size() * sizeof(uint64_t));
        put(P.z.data(), P.z.size() * sizeof(uint64_t));
        const int32_t ph = P.phase & 3;
        put(&ph, sizeof ph);
    };
    const int32_t n = I.n;
    put(&n, sizeof n);
    const FramedSuperposition& B = I.bare;
    for (int a = 0; a < B.n(); ++a) { put_pauli(B.U.Zrow[(size_t)a]); put_pauli(B.U.Xrow[(size_t)a]); }
    for (size_t a = 0; a < B.eps.size(); ++a) { const uint8_t e = B.eps[a] ? 1 : 0; put(&e, 1); }
    for (int f : B.free) put(&f, sizeof f);
    for (const auto& br : B.entries()) { put(&br.first, sizeof br.first); put(&br.second, sizeof br.second); }
    k.push_back('|');
    for (const Pauli& g : I.gens) put_pauli(g);
    k.push_back('|');
    for (const RecColumn& c : I.rec_cols_) { put_pauli(c.M); put(&c.order, sizeof c.order); }
    k.push_back('|');
    put(I.famx_kind_.data(), I.famx_kind_.size());
    put(I.famx_off_.data(), I.famx_off_.size());
    put(I.refmask_ch_.data(), I.refmask_ch_.size() * sizeof(uint64_t));
    const int32_t br = I.born_ref_, bi = I.born_inv_;
    put(&br, sizeof br);
    put(&bi, sizeof bi);
    k.push_back('|');
    const Impl::FamRoute& F = *I.fam_term_.p;
    put(F.rec_inv.data(), F.rec_inv.size());
    put(F.rec_flip.data(), F.rec_flip.size() * sizeof(double));
    return k;
}

}  // namespace qeccore
