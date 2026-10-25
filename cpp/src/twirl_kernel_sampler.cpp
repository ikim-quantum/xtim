#include "twirl_law_internal.hpp"   // J-b1 (fixplan-coldstart): build_shot_law_core, ShotLawCore
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/engine_version.hpp"
#ifdef _WIN32
#include <process.h>   // _getpid (MSVC has no <unistd.h>)
#else
#include <unistd.h>    // getpid
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/plan_cache.hpp"
#include "qeccore/record_table.hpp"

// The diagonal plan machinery: the compact Born directions (free_dir_of / derive_kernel_logical),
// the per-plan Born trie walker (S2a; S2b: over the record table's directions, plus the σ-4 aux
// walk), and the plan cache (permanent headers, evictable payloads). A shot's records are emitted
// from its plan's record table (record_table.hpp) by the sampler; nothing here samples σ.

namespace qeccore {

namespace {

// A Pauli i^p X^x Z^z is Hermitian ⟺ (p − x·z) is even (P† = i^{−p+2(x·z)} X^x Z^z ≡ P).
inline bool is_hermitian(const Pauli& P) { return ((P.phase - P.xz_overlap()) & 1) == 0; }


}  // namespace

// ── Task 10: content-keyed plan memo ────────────────────────────────────────────────────────────


// ── S2a: compact Born directions ─────────────────────────────────────────────────────────────

FreeDir free_dir_of(const FramedSuperposition& bare, const Pauli& L) {
    // Q = U†LU exactly as measure_pauli computes it (the same expression: dual cache when valid,
    // else the forward rows — both yield the unique Q).
    const Pauli Q = bare.U.dual_valid() ? bare.U.conjugate(L) : bare.U.dual_image(L);
    // termroute-a F3: word checks, no per-call vector. Q has no X off the free wires ⟺ the X weight on the
    // (distinct) free wires equals Q's whole X weight; the eps parity runs over Q's Z support only.
    const int kk = bare.k();
    FreeDir d;
    d.a.assign((size_t)kk, 0);
    d.b.assign((size_t)kk, 0);
    int xfree = 0, xall = 0;
    for (int i = 0; i < kk; ++i) {
        const int q = bare.free[(size_t)i];
        d.a[(size_t)i] = Q.xbit(q) ? 1 : 0;
        d.b[(size_t)i] = Q.zbit(q) ? 1 : 0;
        xfree += d.a[(size_t)i];
    }
    for (uint64_t w : Q.x) xall += __builtin_popcountll(w);
    if (xall != xfree)
        throw std::logic_error("free_dir_of: a kernel logical anticommutes with a certified row of "
                               "the bare frame (engine bug)");
    int par = 0;
    for (size_t w = 0; w < Q.z.size(); ++w)
        for (uint64_t zs = Q.z[w]; zs; zs &= zs - 1)
            if (bare.eps[w * 64 + (size_t)__builtin_ctzll(zs)]) par ^= 1;
    d.phase = (uint8_t)(Q.phase & 3);
    d.par = (uint8_t)par;
    d.hermitian = is_hermitian(L) ? 1 : 0;
    return d;
}

Pauli derive_kernel_logical(const FramedSuperposition& bare, const CertifiedGroupPlanes& G,
                            const FreeDir& d) {
    // Q = i^phase·X^a Z^b (free) · Z^{b_nf} (non-free) with Z^{b_nf} = (−1)^{par_nf}·(signed rows),
    // par_nf = par ⊕ Σ_d b[d]·eps[free[d]]. So U·(i^phase (−1)^{par_nf} X^a Z^b)·U† = Λ·(signed
    // certified rows)⁻¹ and G.reduce returns the builder's representative of Λ's coset.
    const int kk = bare.k();
    int par_nf = d.par;
    for (int i = 0; i < kk; ++i)
        if (d.b[(size_t)i] && bare.eps[(size_t)bare.free[(size_t)i]]) par_nf ^= 1;
    Pauli P(bare.n());
    P.phase = (d.phase + 2 * par_nf) & 3;
    for (int i = 0; i < kk; ++i)
        if (d.a[(size_t)i]) P = Pauli::multiply(P, bare.U.Xrow[(size_t)bare.free[(size_t)i]]);
    for (int i = 0; i < kk; ++i)
        if (d.b[(size_t)i]) P = Pauli::multiply(P, bare.U.Zrow[(size_t)bare.free[(size_t)i]]);
    return G.reduce(P).rep;
}

// ── S2a: the Born trie walker ────────────────────────────────────────────────────────────────

BornTrie::BornTrie(const FramedSuperposition& bare) : bare_(bare) {}

void BornTrie::reset_() {
    if (!psi_) { psi_ = bare_.alpha->clone(); beta_ = bare_.alpha->clone(); }
    else psi_->assign_from(*bare_.alpha);
    psi_depth_ = 0;
}

void BornTrie::decode_(const FreeDir& d) {
    // measure_pauli's case-(b) decode, on the plan's FIXED free register: ψ keeps all k bare free
    // coordinates for the whole walk (no χ-drop compaction, so no coordinate ever turns into a
    // certified row and there is no frame to carry). y = a, g = (−1)^b, c = i^phase·(−1)^par.
    g_.assign(d.b.size(), cd(1.0, 0.0));
    for (size_t i = 0; i < d.b.size(); ++i)
        if (d.b[i]) g_[i] = cd(-1.0, 0.0);
    y_ = pack_key(d.a, "BornTrie");
    static const cd ipow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    c_ = ipow[d.phase & 3];
    if (d.par & 1) c_ = -c_;
}

double BornTrie::p_plus_(const FreeDir& d) {
    // born_p1 / measure_pauli case (b), operation for operation: p = (1 + Re<ψ|Λ̂ψ>/<ψ|ψ>)/2.
    // When the (post-prune) support of ψ and its a-shift are disjoint — the configuration in which
    // the deleted chain, having compacted a constant coordinate, took measure_pauli's case (a) —
    // inner() is an EMPTY sum (literal 0) and p = 0.5 exactly, the correct conditional.
    decode_(d);
    chi_j += psi_->support();
    if (psi_->support() > max_support) max_support = psi_->support();
    beta_->assign_from(*psi_);
    beta_->apply_diagonal(cd(1.0, 0.0), g_);
    beta_->shift_key(y_);
    const double nrm2 = psi_->norm2();
    double exp_p = 0.0;
    if (nrm2 >= 1e-30) exp_p = (c_ * psi_->inner(*beta_, idx_)).real() / nrm2;
    if (exp_p < -1.0) exp_p = -1.0;
    if (exp_p > 1.0) exp_p = 1.0;
    ++passes;
    return 0.5 * (1.0 + exp_p);
}

void BornTrie::project_(const FreeDir& d, int eps, bool check_weight) {
    // measure_pauli case (b) at the known outcome ε: ψ ← ψ + ε·c·β, normalize — the amplitude map
    // (1 + εΛ̂)/2 on the fixed k-coordinate register (no compaction). Byte-identical to the
    // compacting chain on its case-(b) paths: dropping a constant coordinate only erased a key byte
    // (same entries, order and key comparisons), and its eps flip (a sign on c) equals the
    // apply_diagonal sign on every entry here — both exact IEEE negations (S2a report §10).
    decode_(d);
    chi_j += psi_->support();
    if (psi_->support() > max_support) max_support = psi_->support();
    beta_->assign_from(*psi_);
    beta_->apply_diagonal(cd(1.0, 0.0), g_);
    beta_->shift_key(y_);
    psi_->axpy(cd((double)eps, 0.0) * c_, *beta_, idx_);
    if (check_weight && !(psi_->norm2() >= 1e-30))
        throw std::logic_error("BornTrie: a forced σ-aux prefix (the shot's C values) has zero weight "
                               "(engine bug)");
    psi_->normalize();
    ++passes;
}

int BornTrie::walk(const std::vector<FreeDir>& lam, std::vector<BornNode>& trie, uint32_t branch,
                   const krng::Stream& born, std::vector<uint8_t>& bits) {
    if (branch != 0)
        throw std::logic_error("BornTrie::walk: carried branch " + std::to_string(branch) +
                               " — branches other than 0 arrive at J-b1");
    const int kap = (int)lam.size();
    ++walks;
    psi_depth_ = -1;                                  // ψ belongs to the previous shot
    const size_t b0 = bits.size();
    if (trie.empty()) { trie.push_back(BornNode{std::nan(""), {0, 0}}); ++nodes; }
    uint32_t node = 0;                                // branch 0's root (inline in the payload)
    for (int j = 0; j < kap; ++j) {
        const FreeDir& d = lam[(size_t)j];
        if (!d.hermitian) return j;
        if (std::isnan(trie[node].p0)) {
#ifndef XTIM_MUTATE_BM_DEEP
            const bool memo_ok = j == 0;                 // S8-4: the root memo, root only (a fresh bare register)
#else
            const bool memo_ok = true;                   // mutation build only (MUT-BM-DEEP): every depth memoized
#endif
            std::string key;
            bool hit = false;
            if (memo_ok) {
                key = root_key_(d);
                const auto it = root_memo_.find(key);
                if (it != root_memo_.end()) {
                    if (root_memo_verify) {              // the selfcheck window: recompute, compare bit for bit
                        reset_();
                        for (; psi_depth_ < j; ++psi_depth_)
                            project_(lam[(size_t)psi_depth_], bits[b0 + (size_t)psi_depth_] ? -1 : +1);
                        const double q = p_plus_(d);
                        if (std::memcmp(&q, &it->second, sizeof q) != 0)
                            throw std::logic_error("BornTrie: a root memo hit differs from its evaluation (engine bug)");
                    }
                    trie[node].p0 = it->second;
                    ++root_memo_hits;
                    hit = true;
                }
            }
            if (!hit) {
                // MISS: evaluate the node on the free register, carrying ψ forward within the shot.
                const auto t0 = time_next_eval ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
                if (psi_depth_ < 0) reset_();
                for (; psi_depth_ < j; ++psi_depth_)
                    project_(lam[(size_t)psi_depth_], bits[b0 + (size_t)psi_depth_] ? -1 : +1);
                trie[node].p0 = p_plus_(d);
                ++node_evals;
                if (j + 1 > max_depth) max_depth = j + 1;
                if (time_next_eval) {
                    timed_eval_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    time_next_eval = false;
                }
                if (memo_ok) {
                    if (root_memo_.size() >= kRootMemoCap) root_memo_clear();
                    root_memo_bytes_ += key.size() + sizeof(double) + 64;   // key bytes + value + node / bucket
                    root_memo_.emplace(std::move(key), trie[node].p0);
                }
            }
        }
        const uint8_t bit = (born.u((uint64_t)j) < trie[node].p0) ? 0 : 1;   // ε = +1 ⇔ bit 0
        bits.push_back(bit);
        ++steps;
        uint32_t kid = trie[node].kid[bit];
        if (kid == 0 && j + 1 < kap) {
            kid = (uint32_t)trie.size();
            trie.push_back(BornNode{std::nan(""), {0, 0}});
            trie[node].kid[bit] = kid;
            ++nodes;
        }
        node = kid;
    }
    return -1;
}


std::string BornTrie::root_key_(const FreeDir& d) {
    std::string k;
    k.reserve(d.a.size() + d.b.size() + 4);
    k.append(reinterpret_cast<const char*>(d.a.data()), d.a.size());
    k.push_back('\xff');
    k.append(reinterpret_cast<const char*>(d.b.data()), d.b.size());
#ifndef XTIM_MUTATE_BM_PHASE
    k.push_back((char)(d.phase & 3));
#endif
    k.push_back((char)(d.par & 1));
    return k;
}
double BornTrie::p0_from_scratch(const std::vector<FreeDir>& lam, const uint8_t* z, int j) {
    reset_();
    for (int i = 0; i < j; ++i) project_(lam[(size_t)i], z[i] ? -1 : +1);
    const double p = p_plus_(lam[(size_t)j]);
    psi_depth_ = -1;
    return p;
}

// ── S0(c): the plan cache — permanent headers (inline bitset key, dense id), evictable payloads ──

size_t rec_table_bytes(const RecTable& T) {
    auto vb = [](const auto& v) { return v.capacity() * sizeof(v[0]); };
    auto vvb = [&](const std::vector<std::vector<uint64_t>>& vv) {
        size_t b = vv.capacity() * sizeof(std::vector<uint64_t>);
        for (const auto& v : vv) b += vb(v);
        return b;
    };
    size_t b = sizeof(RecTable) + vb(T.base) + vvb(T.coin_row) + vvb(T.born_row) + vb(T.pivot_rec) +
               vb(T.born_rec) + T.lam.capacity() * sizeof(FreeDir);
    for (const FreeDir& d : T.lam) b += vb(d.a) + vb(d.b);
    return b;
}

size_t plan_payload_bytes(const CachedPlan& p) {
    auto vb = [](const auto& v) { return v.capacity() * sizeof(v[0]); };
    size_t b = vb(p.rec_trie);
    if (p.rec) b += rec_table_bytes(*p.rec);
    b += vb(p.emit);
    return b;
}

void TwirlPlanCache::HeaderDeleter::operator()(CachedPlan* p) const {
    p->~CachedPlan();
    ::operator delete(static_cast<void*>(p));
}

// J-b1 (fixplan-coldstart (A)): the one-slot core park (the FAM core-mode payload's law, consumed by the FAM
// sampler's lazy record-space-B table build of the same plan).
struct TwirlPlanCache::CoreSlot {
    const CachedPlan* plan = nullptr;
    twirl_law_detail::ShotLawCore core;
    ShotLaw law;
};

TwirlPlanCache::~TwirlPlanCache() {
    delete core_slot_;
    if (k2_) arena_free_(k2_);                        // J-b1 P11a: the cache-owned kind-2 / route-1 headers
}
#ifdef XTIM_MUTATE_1A_STALE
static void* g_stale_prev_ = nullptr;               // mutation build only (a TwirlPlanCache::CoreSlot)
#endif

const twirl_law_detail::ShotLawCore* TwirlPlanCache::parked_core(const CachedPlan& p, const ShotLaw** law_out) const {
#ifdef XTIM_MUTATE_1A_STALE
    if (g_stale_prev_ && static_cast<CoreSlot*>(g_stale_prev_)->plan) {   // mutation build only (MUT-1A-STALE):
        CoreSlot* pv = static_cast<CoreSlot*>(g_stale_prev_);                  // the PREVIOUS build's core
        if (law_out) *law_out = &pv->law;
        return &pv->core;
    }
#endif
    if (!core_slot_ || core_slot_->plan != &p) return nullptr;
    if (law_out) *law_out = &core_slot_->law;
    return &core_slot_->core;
}

void TwirlPlanCache::set_universe(int n, const std::vector<std::pair<int, int>>& pairs,
                                  uint64_t group_token) {
    if (n_ >= 0) {
        if (n != n_ || pairs != pairs_ || group_token != token_)
            throw std::logic_error("TwirlPlanCache: shared by samplers of different compiled "
                                   "circuits (key universe / certified group differ)");
        return;
    }
    n_ = n;
    pairs_ = pairs;
    token_ = group_token;
    NW_ = (n + 63) / 64;
    PW_ = (int)((pairs.size() + 63) / 64);
    KW_ = NW_ + PW_;
    for (size_t i = 0; i < pairs_.size(); ++i)
        pidx_[((uint64_t)(uint32_t)pairs_[i].first << 32) | (uint32_t)pairs_[i].second] = (int)i;
}

bool TwirlPlanCache::key_encodes(const uint64_t* key, const DiagNormalForm& nf) const {
    if ((int)nf.a.size() != n_) return false;
    const uint8_t* a = nf.a.data();
    // Pack a & 1 eight wires per step (gather the low bit of 8 bytes with one multiply), compare
    // word by word against the key's a-support.
    const int n8 = n_ & ~7;
    uint64_t word = 0;
    int q = 0;
    for (; q < n8; q += 8) {
        uint64_t x;
        std::memcpy(&x, a + q, 8);
        const uint64_t bits8 = ((x & 0x0101010101010101ULL) * 0x0102040810204080ULL) >> 56;
        word |= bits8 << (q & 63);
        if ((q & 63) == 56) { if (word != key[q >> 6]) return false; word = 0; }
    }
    for (; q < n_; ++q) word |= (uint64_t)(a[q] & 1) << (q & 63);
    if ((n_ & 63) && word != key[n_ >> 6]) return false;
    const std::pair<int, int>* cz = nf.cz.data();
    const size_t ncz = nf.cz.size();
    size_t i = 0;
    for (int w = 0; w < PW_; ++w) {
        uint64_t word = key[NW_ + w];
        while (word) {
            const size_t idx = (size_t)w * 64 + (size_t)__builtin_ctzll(word);
            word &= word - 1;
            if (i == ncz || cz[i] != pairs_[idx]) return false;
            ++i;
        }
    }
    return i == ncz;
}

void TwirlPlanCache::key_cz(const CachedPlan& p, std::vector<std::pair<int, int>>& out) const {
    out.clear();
    const uint64_t* k = p.key();
    for (int w = 0; w < PW_; ++w) {
        uint64_t word = k[NW_ + w];
        while (word) {
            const int idx = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            out.push_back(pairs_[(size_t)idx]);
        }
    }
}

size_t TwirlPlanCache::header_bytes() const {
    return headers_.size() * (sizeof(CachedPlan) + (size_t)KW_ * sizeof(uint64_t) +
                              sizeof(std::unique_ptr<CachedPlan, HeaderDeleter>)) +
           slots_.capacity() * sizeof(Slot) + fifo_.capacity() * sizeof(CachedPlan*) +
           (k2_ ? arena_bytes_() : 0);                // J-b1 P11a: the arena's permanent headers (FAM caches)
}

CachedPlan* TwirlPlanCache::new_header_(const uint64_t* key, uint64_t h) {
    void* mem = ::operator new(sizeof(CachedPlan) + (size_t)KW_ * sizeof(uint64_t));
    CachedPlan* p = new (mem) CachedPlan();
    std::memcpy(p->key_mut(), key, (size_t)KW_ * sizeof(uint64_t));
    headers_.emplace_back(p);
    p->id = (uint32_t)headers_.size();
    index_insert_(p, h);
    return p;
}

void TwirlPlanCache::index_insert_(CachedPlan* p, uint64_t h) {
    if ((used_slots_ + 1) * 2 > slots_.size()) {          // ≤ 50% load; grow re-inserts everything
        const size_t ns = std::max<size_t>(16384, slots_.size() * 2);
        std::vector<Slot> old(ns, Slot{0, nullptr});
        old.swap(slots_);
        mask_ = ns - 1;
        for (const Slot& sl : old)
            if (sl.p) {
                size_t s = sl.h & mask_;
                while (slots_[s].p) s = (s + 1) & mask_;
                slots_[s] = sl;
            }
    }
    size_t s = h & mask_;
    while (slots_[s].p) s = (s + 1) & mask_;
    slots_[s] = Slot{h, p};
    ++used_slots_;
}

void TwirlPlanCache::build_payload_(CachedPlan& p, const CertifiedGroupPlanes& G,
                                    const FramedSuperposition& bare, const DiagNormalForm& nf) {
    // The prefix-free law: build_shot_law with an IDENTITY prefix ⇒ det_signs = base0, and every
    // other field is prefix-independent (identical to the real-prefix law).
    DiagNormalForm nf0 = nf;
    nf0.prefix = Pauli(G.n_qubits);
    {                                  // J-b1 (fixplan-coldstart (A)) / J-b2 J7b: ONE core law per plan, no σ
        if (!core_slot_) core_slot_ = new CoreSlot();
#ifdef XTIM_MUTATE_1A_STALE
        if (core_slot_->plan) {
            if (!g_stale_prev_) g_stale_prev_ = new CoreSlot();
            *static_cast<CoreSlot*>(g_stale_prev_) = *core_slot_;
        }
#endif
        CoreSlot& sl = *core_slot_;
        sl.plan = nullptr;
        // S8-2 hygiene: the parked core's vectors keep their capacity across plans (every field cleared, as a fresh
        // ShotLawCore; the builder refills them)
        sl.core.n = 0; sl.core.ng = 0;
        sl.core.aidx.clear(); sl.core.v.clear(); sl.core.phi.clear(); sl.core.D.clear(); sl.core.U.clear();
        sl.law = twirl_law_detail::build_shot_law_core(G, nf0, sl.core);
        ++core_builds_;
        const ShotLaw& base = sl.law;
        if (p.built && (base.r != p.r || base.kappa != p.kappa || base.fallback != p.fallback))
            throw std::logic_error("TwirlPlanCache: a payload rebuild disagrees with the first build");
        if (!base.fallback && base.kappa != (int)base.kernel_logicals.size())
            throw std::logic_error("TwirlPlanCache: a built law's kappa disagrees with its kernel count (engine bug)");
        p.r = base.r;
        p.kappa = base.kappa;
        p.fallback = base.fallback;
        p.built = true;
        p.rec.reset();
        p.rec_trie.clear();
        sl.plan = &p;
        p.resident = true;
        p.payload_bytes = plan_payload_bytes(p);
        resident_bytes_ += p.payload_bytes;
        fifo_.push_back(&p);
    }
}

const CachedPlan* TwirlPlanCache::get_or_build(const CertifiedGroupPlanes& G,
                                               const FramedSuperposition& bare, const uint64_t* key,
                                               uint64_t h, const DiagNormalForm& nf, bool verify) {
    CachedPlan* p = const_cast<CachedPlan*>(find(key, h));
    if ((verify || !p) && !key_encodes(key, nf)) return nullptr;   // key-derivation check
    if (p && p->resident) { ++hits_; return p; }
    if (p && !verify && !key_encodes(key, nf)) return nullptr;   // J-b1: a rebuild, nf vs the stored key
    if (!p) {
        ++misses_;
        p = new_header_(key, h);
        build_payload_(*p, G, bare, nf);
        ++(p->fallback ? misses_fb_ : p->kappa > 0 ? misses_kpos_ : misses_k0_);
    } else {
        ++payload_builds_;
        build_payload_(*p, G, bare, nf);
    }
    return p;
}

bool TwirlPlanCache::evict_oldest() {
    while (fifo_head_ < fifo_.size()) {
        CachedPlan* p = fifo_[fifo_head_++];
        if (!p->resident) continue;
        resident_bytes_ -= p->payload_bytes;
        p->payload_bytes = 0;
        p->resident = false;
        if (core_slot_ && core_slot_->plan == p) core_slot_->plan = nullptr;   // J-b1: the park dies with the payload
        p->rec.reset();
        std::vector<uint64_t>().swap(p->emit);
        p->emit_built = false;
        std::vector<BornNode>().swap(p->rec_trie);
        ++evictions_;
        if (fifo_head_ * 2 > fifo_.size() && fifo_head_ > 1024) {   // compact the consumed prefix
            fifo_.erase(fifo_.begin(), fifo_.begin() + (long)fifo_head_);
            fifo_head_ = 0;
        }
        return true;
    }
    return false;
}

// ── Disk layer (J-b2 J7b): TWPL v6 only (twirl_sampler_fam.cpp); a v5 file is a version skew there: rebuild ──
bool TwirlPlanCache::save_file(const std::string& path, uint64_t group_token) const {
    if (n_ < 0) return false;
    return save_file_v6_(path, group_token);
}

bool TwirlPlanCache::load_file(const std::string& path, uint64_t group_token) {
    if (n_ < 0) return false;
    return load_file_v6_(path, group_token);
}

}  // namespace qeccore
