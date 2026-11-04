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

#include "qeccore/record_table.hpp"
#include "qeccore/plan_class.hpp"

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
    if (!psi_) psi_ = bare_.alpha->clone();
    else psi_->assign_from(*bare_.alpha);
    psi_depth_ = 0;
}

void BornTrie::decode_(const FreeDirView& d) {
    // measure_pauli's case-(b) decode, on the plan's FIXED free register: ψ keeps all k bare free
    // coordinates for the whole walk (no χ-drop compaction, so no coordinate ever turns into a
    // certified row and there is no frame to carry). y = a, g = (−1)^b, c = i^phase·(−1)^par.
    g_.assign((size_t)d.k, cd(1.0, 0.0));
    for (int i = 0; i < d.k; ++i)
        if (d.b[i]) g_[(size_t)i] = cd(-1.0, 0.0);
    y_ = pack_key(d.a, (size_t)d.k, "BornTrie");
    static const cd ipow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    c_ = ipow[d.phase & 3];
    if (d.par & 1) c_ = -c_;
}

double BornTrie::p_plus_ref_(const FreeDirView& d) {
    // born_p1 / measure_pauli case (b), operation for operation: p = (1 + Re<ψ|Λ̂ψ>/<ψ|ψ>)/2.
    // When the (post-prune) support of ψ and its a-shift are disjoint — the configuration in which
    // the deleted chain, having compacted a constant coordinate, took measure_pauli's case (a) —
    // inner() is an EMPTY sum (literal 0) and p = 0.5 exactly, the correct conditional.
    decode_(d);
    chi_j += psi_->support();
    if (psi_->support() > max_support) max_support = psi_->support();
    if (!beta_) beta_ = bare_.alpha->clone();
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

void BornTrie::project_ref_(const FreeDirView& d, int eps, bool check_weight) {
    // measure_pauli case (b) at the known outcome ε: ψ ← ψ + ε·c·β, normalize — the amplitude map
    // (1 + εΛ̂)/2 on the fixed k-coordinate register (no compaction). Byte-identical to the
    // compacting chain on its case-(b) paths: dropping a constant coordinate only erased a key byte
    // (same entries, order and key comparisons), and its eps flip (a sign on c) equals the
    // apply_diagonal sign on every entry here — both exact IEEE negations (S2a report §10).
    decode_(d);
    chi_j += psi_->support();
    if (psi_->support() > max_support) max_support = psi_->support();
    if (!beta_) beta_ = bare_.alpha->clone();
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

// perf-engine §1.1: the fused kernels (Amplitudes::born_*_fused) — the same operations without the β copy
// and its separate passes; see amplitudes.hpp for the operation-for-operation argument.
double BornTrie::p_plus_(const FreeDirView& d) {
    decode_(d);
    chi_j += psi_->support();
    if (psi_->support() > max_support) max_support = psi_->support();
    double nrm2 = 0.0;
    const cd ip = psi_->born_inner_fused(g_, y_, idx_, nrm2);
    double exp_p = 0.0;
    if (nrm2 >= 1e-30) exp_p = (c_ * ip).real() / nrm2;
    if (exp_p < -1.0) exp_p = -1.0;
    if (exp_p > 1.0) exp_p = 1.0;
    ++passes;
    return 0.5 * (1.0 + exp_p);
}

void BornTrie::project_(const FreeDirView& d, int eps, bool check_weight) {
    decode_(d);
    chi_j += psi_->support();
    if (psi_->support() > max_support) max_support = psi_->support();
    const double n2 = psi_->born_project_fused(g_, y_, cd((double)eps, 0.0) * c_, idx_);
    if (check_weight && !(n2 >= 1e-30))
        throw std::logic_error("BornTrie: a forced σ-aux prefix (the shot's C values) has zero weight "
                               "(engine bug)");
    ++passes;
}

int BornTrie::walk(LamSpan lam, std::vector<BornNode>& trie, uint32_t branch,
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
        const FreeDirView d = lam.at(j);
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
                    if (root_memo_verify) {              // the selfcheck window: recompute (reference kernels), compare bit for bit
                        const double q = p0_from_scratch(lam, bits.data() + b0, j);
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
                    project_(lam.at(psi_depth_), bits[b0 + (size_t)psi_depth_] ? -1 : +1);
                trie[node].p0 = p_plus_(d);
                ++node_evals;
                if (j + 1 > max_depth) max_depth = j + 1;
                if (root_memo_verify) {                  // the selfcheck window: the fused node == the reference chain
                    // (its passes are counted in oracle_passes_, not in the public counters: those equal a run
                    // without the check)
                    const long sp = passes, sc = chi_j, sm = max_support;
                    const double q = p0_from_scratch(lam, bits.data() + b0, j);
                    oracle_passes_ += passes - sp;
                    passes = sp; chi_j = sc; max_support = sm;
                    if (std::memcmp(&q, &trie[node].p0, sizeof q) != 0)
                        throw std::logic_error("BornTrie: a fused node evaluation differs from the reference chain (engine bug)");
                }
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


int BornTrie::walk_ro(LamSpan lam, const std::vector<BornNode>& trie, const krng::Stream& born,
                      std::vector<uint8_t>& bits) {
    const int kap = (int)lam.size();
    const size_t b0 = bits.size();
    if (trie.empty()) return -2;
    uint32_t node = 0;
    for (int j = 0; j < kap; ++j) {
        const FreeDirView d = lam.at(j);
        if (!d.hermitian) { bits.resize(b0); return j; }
        const double p0 = trie[node].p0;
        if (std::isnan(p0)) { bits.resize(b0); return -2; }
        const uint8_t bit = (born.u((uint64_t)j) < p0) ? 0 : 1;   // walk()'s decision, the same double
        bits.push_back(bit);
        const uint32_t kid = trie[node].kid[bit];
        if (kid == 0 && j + 1 < kap) { bits.resize(b0); return -2; }
        node = kid;
    }
    ++walks;
    steps += kap;
    return -1;
}

void born_trie_merge(std::vector<BornNode>& dst, const std::vector<BornNode>& src) {
    if (src.empty()) return;
    if (dst.empty()) dst.push_back(BornNode{std::nan(""), {0, 0}});
    std::vector<std::pair<uint32_t, uint32_t>> st{{0u, 0u}};   // (src node, dst node)
    while (!st.empty()) {
        const auto [si, di] = st.back();
        st.pop_back();
        const double v = src[si].p0;
        if (!std::isnan(v)) {
            if (std::isnan(dst[di].p0)) dst[di].p0 = v;
            else if (std::memcmp(&v, &dst[di].p0, sizeof v) != 0)
                throw std::logic_error("born_trie_merge: a node evaluated by two lanes differs (engine bug)");
        }
        for (int b = 0; b < 2; ++b) {
            const uint32_t sk = src[si].kid[b];
            if (!sk) continue;
            uint32_t dk = dst[di].kid[b];
            if (!dk) {
                dk = (uint32_t)dst.size();
                dst.push_back(BornNode{std::nan(""), {0, 0}});
                dst[di].kid[b] = dk;
            }
            st.emplace_back(sk, dk);
        }
    }
}

std::string BornTrie::root_key_(const FreeDirView& d) {
    std::string k;
    k.reserve(2 * (size_t)d.k + 4);
    k.append(reinterpret_cast<const char*>(d.a), (size_t)d.k);
    k.push_back('\xff');
    k.append(reinterpret_cast<const char*>(d.b), (size_t)d.k);
#ifndef XTIM_MUTATE_BM_PHASE
    k.push_back((char)(d.phase & 3));
#endif
    k.push_back((char)(d.par & 1));
    return k;
}
double BornTrie::p0_from_scratch(LamSpan lam, const uint8_t* z, int j) {
    reset_();
    for (int i = 0; i < j; ++i) project_ref_(lam.at(i), z[i] ? -1 : +1);
    const double p = p_plus_ref_(lam.at(j));
    // ψ now holds the reference chain's state at depth j — bit for bit the fused chain's (that is what the window
    // verifies), so the walk carries on from it exactly as a run without the recompute would (the same passes, the
    // same counters; a walk always starts from psi_depth_ = -1 anyway).
    psi_depth_ = j;
    return p;
}

double BornTrie::value_from_scratch(LamSpan lam, const uint8_t* z, int j, const FreeDir& X) {
    reset_();
    for (int i = 0; i < j; ++i) project_(lam.at(i), z[i] ? -1 : +1);
    const double v = 2.0 * p_plus_(X) - 1.0;
    psi_depth_ = -1;
    return v;
}

void BornTrie::enumerate(LamSpan lam, const FreeDir* X, double eps,
                         const std::function<bool(uint64_t, double, double)>& leaf, long* evals) {
    const int kap = (int)lam.size();
    if (kap > 63) throw std::logic_error("BornTrie::enumerate: more than 63 directions (engine bug)");
    reset_();
    std::vector<std::unique_ptr<Amplitudes>> saved((size_t)kap);
    bool stop = false;
    long ev = 0;
    std::function<void(int, uint64_t, double)> rec = [&](int j, uint64_t f, double p) {
        while (j < kap) {
            const double pp = p_plus_(lam.at(j));
            ++ev;
            const double pm = 1.0 - pp;
            if (pp > eps && pm > eps) {
                if (!saved[(size_t)j]) saved[(size_t)j] = psi_->clone();
                else saved[(size_t)j]->assign_from(*psi_);
                project_(lam.at(j), +1);
                rec(j + 1, f, p * pp);
                if (stop) return;
                psi_->assign_from(*saved[(size_t)j]);
                project_(lam.at(j), -1);
                rec(j + 1, f | (1ULL << j), p * pm);
                return;
            }
            const bool minus = !(pp > eps);
            project_(lam.at(j), minus ? -1 : +1);
            if (minus) f |= 1ULL << j;
            ++j;
        }
        const double v = X ? 2.0 * p_plus_(*X) - 1.0 : 0.0;
        ++ev;
        if (!leaf(f, p, v)) stop = true;
    };
    rec(0, 0, 1.0);
    psi_depth_ = -1;
    if (evals) *evals += ev;
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

size_t plan_payload_bytes(const PlanClass& c) {
    size_t b = c.rec_trie.capacity() * sizeof(BornNode);
    if (c.rec) b += c.rec->bytes;                     // §3 B1: the blob (table + emission block), one block
    return b;
}

void TwirlPlanCache::HeaderDeleter::operator()(CachedPlan* p) const {
    p->~CachedPlan();
    PlanArena::release(static_cast<void*>(p));
}
void TwirlPlanCache::ClassDeleter::operator()(PlanClass* c) const {
    c->~PlanClass();
    PlanArena::release(static_cast<void*>(c));
}

TwirlPlanCache::TwirlPlanCache() : root_(std::make_shared<ArenaRoot>()), arena_(root_->make_arena()) {}
void TwirlPlanCache::set_overlay_of(const TwirlPlanCache* home) {
    home_ = home;
    if (root_ != home->root_) {                      // the lane's arena joins the home's root (one root per sampler)
        if (!headers_.empty() || !classes_.empty())
            throw std::logic_error("TwirlPlanCache::set_overlay_of: a lane cache with plans (engine bug)");
        arena_->retire();
        root_ = home->root_;
        arena_ = root_->make_arena();
    }
}

// J-b1 (fixplan-coldstart (A)): the one-slot core park (the FAM core-mode payload's law, consumed by the FAM
// sampler's lazy record-space-B table build of the same class).
struct TwirlPlanCache::CoreSlot {
    const PlanClass* cls = nullptr;
    twirl_law_detail::ShotLawCore core;
    ShotLaw law;
};

TwirlPlanCache::~TwirlPlanCache() {
    delete core_slot_;
    if (k2_) arena_free_(k2_);                        // J-b1 P11a: the cache-owned kind-2 / route-1 headers
    headers_.clear();                                 // blocks released into this arena (then retired with it)
    classes_.clear();
    if (arena_) arena_->retire();                     // chunks to the root's graveyard: a handle may still read a blob
}
#ifdef XTIM_MUTATE_1A_STALE
static void* g_stale_prev_ = nullptr;               // mutation build only (a TwirlPlanCache::CoreSlot)
#endif

const twirl_law_detail::ShotLawCore* TwirlPlanCache::parked_core(const PlanClass& c, const ShotLaw** law_out) const {
#ifdef XTIM_MUTATE_1A_STALE
    if (g_stale_prev_ && static_cast<CoreSlot*>(g_stale_prev_)->cls) {   // mutation build only (MUT-1A-STALE):
        CoreSlot* pv = static_cast<CoreSlot*>(g_stale_prev_);                 // the PREVIOUS build's core
        if (law_out) *law_out = &pv->law;
        return &pv->core;
    }
#endif
    if (!core_slot_ || core_slot_->cls != &c) return nullptr;
    if (law_out) *law_out = &core_slot_->law;
    return &core_slot_->core;
}

void TwirlPlanCache::set_universe(int n, const std::vector<std::pair<int, int>>& pairs,
                                  uint64_t group_token, bool reserve_identity) {
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
    if (reserve_identity && classes_.empty()) {          // review M1: the identity class is class 1 at every T
        std::vector<uint64_t> z((size_t)KW_, 0);
        PlanClass* c = new_class_(z.data(), plan_key_hash(z.data(), KW_));
        c->first_shot = 0;
    }
}

void TwirlPlanCache::set_classifier(std::shared_ptr<const PlanClassifier> pc) {
    if (!pc || pc->NW != NW_ || pc->PW != PW_ || pc->n != n_)
        throw std::logic_error("TwirlPlanCache::set_classifier: a classifier of another key universe (engine bug)");
    if (pc_ && pc_ != pc) throw std::logic_error("TwirlPlanCache::set_classifier: the classifier is set once (engine bug)");
    pc_ = std::move(pc);
}
const PlanClassifier& TwirlPlanCache::classifier() const {
    if (!pc_) throw std::logic_error("TwirlPlanCache: no classifier (engine bug)");
    return *pc_;
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

void TwirlPlanCache::class_key_of(const uint64_t* key, uint64_t* out) const {
    std::memcpy(out, key, (size_t)KW_ * sizeof(uint64_t));
    classifier().project(out);
}

bool TwirlPlanCache::class_key_of_nf(const DiagNormalForm& nf, std::vector<uint64_t>& out) const {
    out.assign((size_t)KW_, 0);
    for (int q = 0; q < n_ && q < (int)nf.a.size(); ++q)
        if (nf.a[(size_t)q] & 1) out[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    for (const auto& pr : nf.cz) {
        const auto u = std::lower_bound(pairs_.begin(), pairs_.end(), pr);
        if (u == pairs_.end() || *u != pr) return false;   // outside the pair universe: not a diagonal-cache key
        const size_t ui = (size_t)(u - pairs_.begin());
        const auto& keep = classifier().pkeep;              // review N1: a pair on a product wire is not a key pair
        if ((ui >> 6) >= keep.size() || !((keep[ui >> 6] >> (ui & 63)) & 1ULL)) return false;
        out[(size_t)NW_ + (ui >> 6)] |= 1ULL << (ui & 63);
    }
    classifier().project(out.data());
    return true;
}

const PlanClass* TwirlPlanCache::find_class(const uint64_t* ckey, uint64_t h) const {
    size_t s = h & cmask_;
    for (;;) {
        const CSlot& sl = cslots_[s];
        if (!sl.c) return nullptr;
        if (sl.h == h) {
            const uint64_t* k2 = sl.c->key();
            int w = 0;
            while (w < KW_ && k2[w] == ckey[w]) ++w;
            if (w == KW_) return sl.c;
        }
        s = (s + 1) & cmask_;
    }
}

void TwirlPlanCache::key_nf(const uint64_t* k, DiagNormalForm& nf) const {
    if (nf.prefix.n == n_ && nf.prefix.x.size() == (size_t)((n_ + 63) / 64) && nf.prefix.z.size() == nf.prefix.x.size()) {
        std::fill(nf.prefix.x.begin(), nf.prefix.x.end(), 0ULL);   // speed fix 5: a reused scratch (no allocation)
        std::fill(nf.prefix.z.begin(), nf.prefix.z.end(), 0ULL);
        nf.prefix.phase = 0;
    } else {
        nf.prefix = Pauli(n_);
    }
    nf.a.assign((size_t)n_, 0);
    nf.cz.clear();
    nf.diagonal_class = true;
    for (int w = 0; w < NW_; ++w)                    // (speed fix 5: the set bits only)
        for (uint64_t word = k[w]; word; word &= word - 1) {
            const int q = w * 64 + __builtin_ctzll(word);
            if (q < n_) nf.a[(size_t)q] = 1;
        }
    for (int w = 0; w < PW_; ++w) {
        uint64_t word = k[NW_ + w];
        while (word) {
            const int idx = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            nf.cz.push_back(pairs_[(size_t)idx]);
        }
    }
}

size_t TwirlPlanCache::header_bytes() const {
    // review M2: the arena blocks as allocated (the 32 B block header, 64 B rounding) — exact permanent bytes
    auto block = [](size_t payload) { return (payload + sizeof(ArenaBlock) + 63) & ~size_t(63); };
    const size_t key = (size_t)KW_ * sizeof(uint64_t);
    return headers_.size() * (block(sizeof(CachedPlan) + key) + sizeof(std::unique_ptr<CachedPlan, HeaderDeleter>)) +
           // (speed fix 5: a class counts its key words whether inline or borrowed from its header — which one is the
           // arrival order's, so the gauge stays T-identical; the borrowed saving shows in arena_live_bytes)
           classes_.size() * (block(sizeof(PlanClass) + key) + sizeof(std::unique_ptr<PlanClass, ClassDeleter>)) +
           slots_.capacity() * sizeof(Slot) + cslots_.capacity() * sizeof(CSlot) +
           fifo_.capacity() * sizeof(PlanClass*) +
           (k2_ ? arena_bytes_() : 0);                // J-b1 P11a: the arena's permanent headers (FAM caches)
}

CachedPlan* TwirlPlanCache::new_header_(const uint64_t* key, uint64_t h) {
    void* mem = arena_->alloc(sizeof(CachedPlan) + (size_t)KW_ * sizeof(uint64_t));   // §4: bump-allocated
    CachedPlan* p = new (mem) CachedPlan();
    std::memcpy(p->key_mut(), key, (size_t)KW_ * sizeof(uint64_t));
    headers_.emplace_back(p);
    p->id = id_base_ + (uint32_t)headers_.size();
    index_insert_(p, h);
    return p;
}

void TwirlPlanCache::index_insert_(CachedPlan* p, uint64_t h) {
    if ((used_slots_ + 1) * 10 > slots_.size() * 7) {     // ≤ 70% load (R3); grow re-inserts everything
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

PlanClass* TwirlPlanCache::new_class_(const uint64_t* ckey, uint64_t h, const uint64_t* borrow) {
    // speed fix 5: `borrow` — a permanent header's key equal to ckey (the class key is then stored once)
    void* mem = arena_->alloc(sizeof(PlanClass) + (borrow ? 0 : (size_t)KW_ * sizeof(uint64_t)));
    PlanClass* c = new (mem) PlanClass();
    if (borrow) {
        c->kp = borrow;
    } else {
        std::memcpy(c->key_mut(), ckey, (size_t)KW_ * sizeof(uint64_t));
        c->kp = c->key_mut();
    }
    classes_.emplace_back(c);
    c->id = class_base_ + (uint32_t)classes_.size();
    c->owner = this;
    class_index_insert_(c, h);
    return c;
}

void TwirlPlanCache::class_index_insert_(PlanClass* c, uint64_t h) {
    if ((cused_ + 1) * 10 > cslots_.size() * 7) {         // ≤ 70% load (R3)
        const size_t ns = std::max<size_t>(16384, cslots_.size() * 2);
        std::vector<CSlot> old(ns, CSlot{0, nullptr});
        old.swap(cslots_);
        cmask_ = ns - 1;
        for (const CSlot& sl : old)
            if (sl.c) {
                size_t s = sl.h & cmask_;
                while (cslots_[s].c) s = (s + 1) & cmask_;
                cslots_[s] = sl;
            }
    }
    size_t s = h & cmask_;
    while (cslots_[s].c) s = (s + 1) & cmask_;
    cslots_[s] = CSlot{h, c};
    ++cused_;
}

void TwirlPlanCache::link_(CachedPlan& p, PlanClass& c, bool list) {
    p.cls = &c;
    p.r = c.r; p.kappa = c.kappa; p.fallback = c.fallback; p.built = c.built;
    p.mirror();
    if (list) { p.next_in_class = c.headers; c.headers = &p; }
}

void TwirlPlanCache::install_(PlanClass& c) {
    c.resident = true;
    c.payload_bytes = plan_payload_bytes(c);
    resident_bytes_ += c.payload_bytes;
    fifo_.push_back(&c);
}

void TwirlPlanCache::release_(PlanClass& c) {
    resident_bytes_ -= c.payload_bytes;
    c.payload_bytes = 0;
    c.resident = false;
    if (core_slot_ && core_slot_->cls == &c) core_slot_->cls = nullptr;   // J-b1: the park dies with the payload
    c.rec.reset();                                    // the blob: freed now, or when the last reader's handle drops
    c.emit_built = false;
    std::vector<BornNode>().swap(c.rec_trie);
    sync_headers(c);
}

void TwirlPlanCache::build_class_(PlanClass& c, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                  bool verify) {
    (void)bare;
    // The prefix-free law of the class REPRESENTATIVE (identity prefix ⇒ det_signs = base0, and every
    // other field is prefix-independent).
    DiagNormalForm& nf0 = nf_scratch_;   // speed fix 5: the decode's reusable scratch (capacities kept)
    class_nf(c, nf0);
    {                                  // J-b1 (fixplan-coldstart (A)) / J-b2 J7b: ONE core law per class, no σ
        if (!core_slot_) core_slot_ = new CoreSlot();
#ifdef XTIM_MUTATE_1A_STALE
        if (core_slot_->cls) {
            if (!g_stale_prev_) g_stale_prev_ = new CoreSlot();
            *static_cast<CoreSlot*>(g_stale_prev_) = *core_slot_;
        }
#endif
        CoreSlot& sl = *core_slot_;
        sl.cls = nullptr;
        // S8-2 hygiene: the parked core's vectors keep their capacity across classes (every field cleared, as a fresh
        // ShotLawCore; the builder refills them)
        sl.core.n = 0; sl.core.ng = 0;
        sl.core.aidx.clear(); sl.core.v.clear(); sl.core.phi.clear(); sl.core.D.clear(); sl.core.U.clear();
        sl.law = twirl_law_detail::build_shot_law_core(G, nf0, sl.core, /*teeth=*/verify);   // perf-engine §2.6
        ++core_builds_;
        ++class_builds_;
        const ShotLaw& base = sl.law;
        if (c.built && (base.r != c.r || base.kappa != c.kappa || base.fallback != c.fallback))
            throw std::logic_error("TwirlPlanCache: a class rebuild disagrees with the first build");
        if (!base.fallback && base.kappa != (int)base.kernel_logicals.size())
            throw std::logic_error("TwirlPlanCache: a built law's kappa disagrees with its kernel count (engine bug)");
        c.r = base.r;
        c.kappa = base.kappa;
        c.fallback = base.fallback;
        c.built = true;
        c.rec.reset();
        c.rec_trie.clear();
        c.emit_rho = c.emit_kappa = 0;
        c.emit_built = false;
        sl.cls = &c;
        install_(c);
        sync_headers(c);
    }
}

const CachedPlan* TwirlPlanCache::get_or_build(const CertifiedGroupPlanes& G,
                                               const FramedSuperposition& bare, const uint64_t* key,
                                               uint64_t h, const DiagNormalForm& nf, bool verify, bool* created,
                                               uint64_t shot) {
    if (created) *created = false;
    CachedPlan* p = const_cast<CachedPlan*>(find(key, h));
    if ((verify || !p) && !key_encodes(key, nf)) return nullptr;   // key-derivation check
    if (p && p->resident) { ++hits_; return p; }
    if (p && !verify && !key_encodes(key, nf)) return nullptr;   // J-b1: a rebuild, nf vs the stored key
    const bool fresh = p == nullptr;
    if (fresh) {
        ++misses_;
        if (created) *created = true;
        p = new_header_(key, h);
    }
    if (!p->cls) {
        // the class: a resident home class (a lane) read-only; else this cache's (created, or rebuilt when evicted).
        // (A header without a class: fresh, or an earlier link that failed mid-way — an allocation failure between
        // the header's creation and the link, scaling fix-2: the next shot links it.)
        uint64_t ck[64];
        uint64_t* ckp = ck;
        if (KW_ > 64) {                            // speed fix 5: the cache's scratch (no allocation per miss)
            if (ck_scratch_.size() < (size_t)KW_) ck_scratch_.resize((size_t)KW_);
            ckp = ck_scratch_.data();
        }
        class_key_of(key, ckp);
        const uint64_t ch = plan_key_hash(ckp, KW_);
        PlanClass* c = nullptr;
        bool home_class = false;
        if (home_) {                               // (as lane_find_: a home class a lane may read without writing)
            c = home_->find_class_mut_(ckp, ch);
            if (c && c->resident && (c->emit_built || c->fallback)) home_class = true;
            else c = nullptr;
        }
        if (!c) {
            c = find_class_mut_(ckp, ch);
            if (!c) {
                // speed fix 5: the creating header's key IS the class key (its own representative) — the class
                // borrows it (the header is permanent): the key is stored once. (Which member creates the class is
                // the arrival order's; nothing the run emits reads where the words live.)
                const bool borrow = std::memcmp(ckp, key, (size_t)KW_ * sizeof(uint64_t)) == 0;
                c = new_class_(ckp, ch, borrow ? p->key() : nullptr);
                c->first_shot = shot;
            }
            else if (!c->resident && c->built) ++payload_builds_;   // a rebuild (the reserved identity class's first
                                                                    // build is a first build)
            if (!c->resident) build_class_(*c, G, bare, verify);
        }
        link_(*p, *c, !home_class);
        if (fresh) ++(p->fallback ? misses_fb_ : p->kappa > 0 ? misses_kpos_ : misses_k0_);
    } else {
        PlanClass& c = *p->cls;
        if (c.owner != this)
            throw std::logic_error("TwirlPlanCache::get_or_build: a lane rebuilding a home class (engine bug)");
        if (!c.resident) {
            ++payload_builds_;
            build_class_(c, G, bare, verify);
        } else {
            p->mirror();                               // (a link that failed before the mirror: refreshed here)
        }
    }
    return p;
}

bool TwirlPlanCache::evict_oldest() {
    while (fifo_head_ < fifo_.size()) {
        PlanClass* c = fifo_[fifo_head_++];
        if (!c->resident) continue;
#ifndef XTIM_MUTATE_R2_NO_SECOND_CHANCE
        if (c->mark) {                                 // §2 R2: served since the last pass — one more lap
            c->mark = 0;
            fifo_.push_back(c);
            ++second_chances_;
            continue;
        }
#endif                                                 // mutation build only (MUT-R2-NO-SECOND-CHANCE): plain FIFO
        release_(*c);
        ++evictions_;
        if (fifo_head_ * 2 > fifo_.size() && fifo_head_ > 1024) {   // compact the consumed prefix
            fifo_.erase(fifo_.begin(), fifo_.begin() + (long)fifo_head_);
            fifo_head_ = 0;
        }
        return true;
    }
    return false;
}

void TwirlPlanCache::adopt_prepare(const TwirlPlanCache& home) {
    adopt_pre_.clear();
    adopt_pre_.reserve(headers_.size());
    for (const auto& up : headers_) {
        const uint64_t h = plan_key_hash(up->key(), KW_);
        adopt_pre_.emplace_back(h, const_cast<CachedPlan*>(home.find(up->key(), h)));
    }
    adopt_cpre_.clear();
    adopt_cpre_.reserve(classes_.size());
    for (const auto& uc : classes_) {
        const uint64_t h = plan_key_hash(uc->key(), KW_);
        adopt_cpre_.emplace_back(h, home.find_class_mut_(uc->key(), h));
    }
}

void TwirlPlanCache::adopt_from(TwirlPlanCache& o) {
    if (o.n_ != n_ || o.KW_ != KW_ || o.token_ != token_ || o.pairs_ != pairs_ || o.pc_ != pc_)
        throw std::logic_error("TwirlPlanCache::adopt_from: a lane cache of another universe (engine bug)");
    if (o.k2_ && o.arena_size())
        throw std::logic_error("TwirlPlanCache::adopt_from: a lane cache holds arena headers (engine bug)");
    if (o.adopt_pre_.size() != o.headers_.size() || o.adopt_cpre_.size() != o.classes_.size())
        o.adopt_prepare(*this);                            // (not prepared on its lane)
    // 1. the lane's classes, in creation order
    const size_t nc = o.classes_.size();
    std::vector<PlanClass*> cmap(nc, nullptr);
    for (size_t j = 0; j < nc; ++j) {
        auto& uc = o.classes_[j];
        PlanClass* lc = uc.get();
        const uint64_t h = o.adopt_cpre_[j].first;
        PlanClass* hc = o.adopt_cpre_[j].second;
        if (!hc) hc = find_class_mut_(lc->key(), h);   // an earlier lane of this barrier moved the key in
        if (!hc) {                                     // new class: the object moves in (pointer stable)
            PlanClass* c = uc.release();
            classes_.emplace_back(c);
            c->id = (uint32_t)classes_.size();
            c->owner = this;
            c->headers = nullptr;                      // its headers are relinked below (moved in or dropped)
            class_index_insert_(c, h);
            if (c->resident) { resident_bytes_ += c->payload_bytes; fifo_.push_back(c); }
            c->twin = c;
            cmap[j] = c;
            continue;
        }
        cmap[j] = hc;
        lc->twin = hc;
        if (!hc->resident && lc->resident) {           // the home class without its payload: the lane's, same id
            if (hc->built && (hc->r != lc->r || hc->kappa != lc->kappa || hc->fallback != lc->fallback))
                throw std::logic_error("TwirlPlanCache::adopt_from: a lane class disagrees with the first build");
            hc->r = lc->r; hc->kappa = lc->kappa; hc->fallback = lc->fallback; hc->built = true;
            hc->rec = std::move(lc->rec);               // the blob stays where it is (its chunk is spliced below)
            hc->emit_rho = lc->emit_rho; hc->emit_kappa = lc->emit_kappa; hc->emit_built = lc->emit_built;
            hc->rec_trie = std::move(lc->rec_trie);
            lc->resident = false; lc->payload_bytes = 0;
            install_(*hc);
            ++payload_builds_;
            sync_headers(*hc);
        } else if (hc->resident && lc->resident) {     // a duplicate (bytes equal by purity): its trie nodes join
            if (!lc->rec_trie.empty()) {
                born_trie_merge(hc->rec_trie, lc->rec_trie);
                reaccount(*hc);
            }
        }
        hc->mark |= lc->mark;                          // review N4: the lane's hit mark survives the merge
        if (lc->first_shot < hc->first_shot) hc->first_shot = lc->first_shot;   // M1: the earliest shot that needed it
    }
    // 2. the lane's headers, in creation order
    const size_t nh = o.headers_.size();
    for (size_t j = 0; j < nh; ++j) {
        if (j + 8 < nh) __builtin_prefetch(&slots_[o.adopt_pre_[j + 8].first & mask_]);
        auto& up = o.headers_[j];
        CachedPlan* lp = up.get();
        const uint64_t h = o.adopt_pre_[j].first;
        CachedPlan* sp = o.adopt_pre_[j].second;
        if (!sp) {                                         // absent from home at prepare: ONE probe walk
            size_t s = h & mask_;
            for (;;) {
                const Slot& sl = slots_[s];
                if (!sl.p) break;
                if (sl.h == h) {
                    const uint64_t* k2 = sl.p->key();
                    int w = 0;
                    while (w < KW_ && k2[w] == lp->key()[w]) ++w;
                    if (w == KW_) { sp = sl.p; break; }    // an earlier lane of this barrier moved the key in
                }
                s = (s + 1) & mask_;
            }
            if (!sp) {                                     // new content: the header moves in (pointer stable)
                CachedPlan* p = up.release();
                headers_.emplace_back(p);
                p->id = (uint32_t)headers_.size();
                if ((used_slots_ + 1) * 10 > slots_.size() * 7) {
                    index_insert_(p, h);                   // the index grows (as a serial insert would here)
                } else {
                    slots_[s] = Slot{h, p};
                    ++used_slots_;
                }
                // its class here: a home class it read, or the lane class's home twin
                PlanClass* c = p->cls;
                if (c->owner != this) {
                    c = c->twin;
                    if (!c || c->owner != this)
                        throw std::logic_error("TwirlPlanCache::adopt_from: a lane header's class is unknown (engine bug)");
                }
                p->next_in_class = nullptr;
                link_(*p, *c, true);
                ++misses_;
                ++(p->fallback ? misses_fb_ : p->kappa > 0 ? misses_kpos_ : misses_k0_);
                continue;
            }
        }
        // a duplicate key: dropped (its class was handled above). Speed fix 5: a moved-in class that borrows the
        // dropped header's key borrows the surviving header's (the same words) — the dropped one is freed below
        if (PlanClass* c = lp->cls ? lp->cls->twin : nullptr)
            if (c->owner == this && c->kp == lp->key()) c->kp = sp->key();
    }
    for (size_t j = 0; j < nc; ++j) if (cmap[j]) cmap[j]->twin = nullptr;
    core_builds_ += o.core_builds_;
    class_builds_ += o.class_builds_;
    o.clear_plans();                                   // the dropped duplicates: released (into the lane's lists)
    // §4 splice: the lane's chunks (every moved-in header, class and blob lives there) become this cache's; its free
    // lists and deferred list (the duplicates just released) join this arena's; the lane restarts at 1 MiB.
    arena_->splice_from(*o.arena_);
    arena_->drain();
}

void TwirlPlanCache::order_new_classes(size_t from) {
    if (from >= classes_.size()) return;
    std::stable_sort(classes_.begin() + (long)from, classes_.end(),
                     [](const auto& a, const auto& b) { return a->first_shot < b->first_shot; });
    for (size_t i = from; i < classes_.size(); ++i) classes_[i]->id = class_base_ + (uint32_t)i + 1;
}

void TwirlPlanCache::assert_lane_tables_unheld() const {
    for (const auto& uc : classes_)
        if (uc && uc->rec && uc->rec.use_count() > 1)
            throw std::logic_error("TwirlPlanCache: a reader holds a lane's record table at the lane's reset (engine bug)");
}

void TwirlPlanCache::clear_plans() {
    if (core_slot_) core_slot_->cls = nullptr;
    for (auto& uc : classes_) if (uc) uc->rec.reset();   // the blobs first (into this arena's deferred list)
    // speed fix 6: a lane overlay's indexes are cleared along their entries' probe runs (the adopt pass recorded every
    // entry's hash; linear probing without deletions keeps each entry inside the occupied run from its home slot, and
    // a cleared run is cleared through to its first empty slot) — not a full fill of the (≥ 16384-slot) arrays at
    // every barrier; a full fill otherwise
    // (every occupied slot cleared is counted: fewer than the index holds — an entry the record missed — falls back to
    // the full fill, so no stale slot can survive)
    bool full = true;
    if (adopt_pre_.size() == headers_.size() && used_slots_ * 8 < slots_.size()) {
        size_t cleared = 0;
        for (const auto& hp : adopt_pre_)
            for (size_t s = hp.first & mask_; slots_[s].p; s = (s + 1) & mask_) { slots_[s] = Slot{0, nullptr}; ++cleared; }
        full = cleared != used_slots_;
    }
    if (full) std::fill(slots_.begin(), slots_.end(), Slot{0, nullptr});
    full = true;
    if (adopt_cpre_.size() == classes_.size() && cused_ * 8 < cslots_.size()) {
        size_t cleared = 0;
        for (const auto& hc : adopt_cpre_)
            for (size_t s = hc.first & cmask_; cslots_[s].c; s = (s + 1) & cmask_) { cslots_[s] = CSlot{0, nullptr}; ++cleared; }
        full = cleared != cused_;
    }
    if (full) std::fill(cslots_.begin(), cslots_.end(), CSlot{0, nullptr});
    adopt_pre_.clear();
    adopt_cpre_.clear();
    headers_.clear();                                      // moved-out entries are null (nothing freed twice)
    classes_.clear();
    used_slots_ = 0;
    cused_ = 0;
    fifo_.clear();
    fifo_head_ = 0;
    hits_ = misses_ = misses_k0_ = misses_kpos_ = misses_fb_ = 0;
    resident_bytes_ = payload_builds_ = evictions_ = core_builds_ = class_builds_ = second_chances_ = 0;
}

}  // namespace qeccore
