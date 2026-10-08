#include "qeccore/twirl_kernel.hpp"
#ifdef _WIN32
#include <process.h>   // _getpid (MSVC has no <unistd.h>)
#else
#include <unistd.h>    // getpid
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/plan_cache.hpp"

// twirl_collapse — kernel chain-rule sampler + collapsed amplitudes (spec §5 step 5, Task 9).
//
// The structured per-shot law (Task 8's ShotLaw) is consumed here to produce the compact shot
// record WITHOUT measuring every generator (the O(n³) sequential twirl) and WITHOUT any 2^κ / 2^r
// outcome table. The magic-sector amplitude update and the syndrome sampling are ONE pass:
//   * det ⊕ r fair coins give the stabiliser-sector part of σ;
//   * the κ kernel logicals are Born-measured SEQUENTIALLY on a single working copy of the bare
//     state (chain rule — each conditioned on the collapse of the previous), and the residual's
//     within-sector action is applied to the final collapsed state to yield `amps`.
//
// Correctness note (validated against scripts/twirl_oracle.py::analytic_law, dense-sim prototype):
// reading a logical column Q off `amps` reproduces the per-sector conditional ⟨ψ|Π'_σ R†QR|ψ⟩/p_σ
// exactly. The r coins are logically trivial (fair, decorrelated from the magic sector — that is the
// V∩N_z classification), so measuring ONLY the κ kernel logicals (not the coins) on the working copy
// suffices for the magic content; applying R = P·C then gives amps = R·(kernel-collapsed |ψ⟩), whose
// logical reads equal the sector conditionals.

namespace qeccore {

namespace {

// A Pauli i^p X^x Z^z is Hermitian ⟺ (p − x·z) is even (P† = i^{−p+2(x·z)} X^x Z^z ≡ P).
inline bool is_hermitian(const Pauli& P) { return ((P.phase - P.xz_overlap()) & 1) == 0; }

// XOR an n_gens-bit mask into σ (both are word vectors of identical length).
inline void xor_into(std::vector<uint64_t>& sigma, const std::vector<uint64_t>& mask) {
    const size_t w = mask.size() < sigma.size() ? mask.size() : sigma.size();
    for (size_t i = 0; i < w; ++i) sigma[i] ^= mask[i];
}

}  // namespace

TwirlOutcome twirl_collapse(const FramedSuperposition& bare, const CertifiedGroupPlanes& /*G*/,
                            const DiagNormalForm& nf, const ShotLaw& law, Rng& rng,
                            bool need_amps) {
    TwirlOutcome out;

    // Out-of-diagonal-class law ⇒ caller must take the exact fallback (never silent).
    if (law.fallback || !nf.diagonal_class) {
        out.fallback = true;
        out.sigma = law.det_signs;
        out.amps = bare;   // valid (unmodified) state so the caller can route it exactly
        return out;
    }

    // Step 1 — deterministic base point.
    out.sigma = law.det_signs;

    // Step 2 — r independent fair coins (drawn FIRST; see the draw-order contract). Each coin XORs
    // its precomputed n_gens-bit sign mask into σ. Bit convention: 0 = no flip, 1 = flip.
    out.coins.reserve(law.coin_masks.size() + law.kernel_logicals.size());
    for (const std::vector<uint64_t>& mask : law.coin_masks) {
        const uint8_t bit = (rng() < 0.5) ? 0 : 1;
        out.coins.push_back(bit);
        if (bit) xor_into(out.sigma, mask);
    }

    // Steps 3+5 — κ kernel coordinates: SEQUENTIAL chain-rule Born on a single working copy of the
    // bare magic sector. No 2^κ table: the loop is linear in κ, each step reads the CURRENT state.
    // Task 10: each fired outcome (ε=−1, bit 1) XORs its precomputed kernel_masks[j] into σ — the
    // preimage of the kernel direction v*_j — so σ folds the κ logical outcomes (det ⊕ coins ⊕
    // kernel). Same 0=+1/1=−1 convention as the coins record.
    //
    // need_amps=false + κ=0 (the in-distribution common case): (σ, coins) are complete and amps is
    // unread by contract — skip the n-qubit state copy and the residual application entirely (the
    // ~244 µs/shot hot spot at n=298; header contract).
    if (!need_amps && law.kernel_logicals.empty()) return out;
    out.amps = bare;
    for (size_t j = 0; j < law.kernel_logicals.size(); ++j) {
        const Pauli& L = law.kernel_logicals[j];
        // Guard: the kernel rep must be Hermitian (a genuine ±1 logical observable).
        if (!is_hermitian(L)) {
            std::fprintf(stderr, "[twirl_collapse] guard: non-Hermitian kernel rep -> fallback\n");
            out.fallback = true;
            return out;
        }
        // Guard: Born probabilities must sum to 1 (p+ + p- == 1 within 1e-9).
        const std::pair<double, double> pr = framed_expectation(out.amps, L);
        if (!(std::abs(pr.first + pr.second - 1.0) < 1e-9)) {
            std::fprintf(stderr, "[twirl_collapse] guard: p+ + p- = %.12g != 1 -> fallback\n",
                         pr.first + pr.second);
            out.fallback = true;
            return out;
        }
        // Sample + collapse in ONE pass with the SAME draw. measure_pauli returns ε∈{+1,−1}
        // (ε=+1 iff u < p+), mutating out.amps into the post-measurement state.
        const int eps = out.amps.measure_pauli(L, rng());
        const uint8_t bit = (eps == +1) ? 0 : 1;
        out.coins.push_back(bit);
        // Fold into σ: XOR the kernel preimage iff (outcome ⊕ kernel_base[j]) — the base bit pins
        // mask_j·σ to the operator sign s_j so σ matches the reference full-chain σ↔outcome map.
        // V2 fold guard: unfoldable kernel dirs (kernel_foldable[j]==0) contribute NO σ weighting —
        // the outcome is still recorded in `coins` (and drives the collapsed amps) but never folds σ.
        // For diagonal V1 plans every flag is 1, so this is a no-op (byte-identical).
        const uint8_t fold_j = (j < law.kernel_foldable.size()) ? law.kernel_foldable[j] : 1;
        const uint8_t base_j = (j < law.kernel_base.size()) ? law.kernel_base[j] : 0;
        if (fold_j && (bit ^ base_j) && j < law.kernel_masks.size() && !law.kernel_masks[j].empty())
            xor_into(out.sigma, law.kernel_masks[j]);
    }

    // Step 4 — apply the residual's within-sector action R = P·C to the collapsed state. C is the
    // diagonal layer (S^a then CZ, all commuting), P the exact Pauli prefix (X^v, Z^z). Apply C then
    // P so the state becomes P·C·|collapsed⟩ = R·|collapsed⟩ (the residual's global phase, incl. the
    // discarded γ and prefix.phase, is unobservable and dropped).
    if (!need_amps) return out;   // κ>0 record-only: chain ran for σ/coins; amps left unspecified
    const int n = out.amps.n();
    for (int q = 0; q < n && q < (int)nf.a.size(); ++q)
        if (nf.a[q] & 1) out.amps.apply_clifford(/*S*/ 1, q, 0);
    for (const std::pair<int, int>& e : nf.cz)
        out.amps.apply_clifford(/*CZ*/ 7, e.first, e.second);
    for (int q = 0; q < n; ++q) {
        if (nf.prefix.xbit(q)) out.amps.apply_clifford(/*X*/ 3, q, 0);
        if (nf.prefix.zbit(q)) out.amps.apply_clifford(/*Z*/ 5, q, 0);
    }

    return out;
}

// ── Task 10: content-keyed plan memo ────────────────────────────────────────────────────────────

ShotLaw CachedPlan::finalize(const CertifiedGroupPlanes& G, const DiagNormalForm& nf) const {
    ShotLaw law;
    law.r = r;
    law.kappa = kappa;
    law.coin_masks = coin_masks;
    law.kernel_masks = kernel_masks;
    law.kernel_base = kernel_base;
    law.kernel_foldable = kernel_foldable;
    law.kernel_logicals = kernel_logicals;
    law.tier1 = tier1;                     // active is prefix-free (sign is base0-relative, unused here)
    law.fallback = fallback;               // 2026-07-15: propagate build-time fallback (was hardcoded
                                           // false — a cached κ≥2 fallback plan would have been
                                           // served as valid, silent-wrong; never hit in-distribution)
    // det_signs = base0 ⊕ plane1(P): a valid base point for prefix P (preserves every C_det parity).
    law.det_signs = base0;
    const std::vector<uint64_t> p1 = prefix_plane1(G, nf.prefix);
    xor_into(law.det_signs, p1);
    return law;
}

void twirl_collapse_record(const FramedSuperposition& bare, const CertifiedGroupPlanes& G,
                           const DiagNormalForm& nf, const CachedPlan& plan, Rng& rng,
                           TwirlOutcome& out) {
    out.coins.clear();
    if (plan.fallback || !nf.diagonal_class) {
        out.fallback = true;
        out.sigma = plan.base0;            // valid-but-unspecified; caller routes the exact path
        return;
    }
    out.fallback = false;
    // σ = base0 ⊕ plane1(P), XORed columnar in place (no temporary vector).
    out.sigma = plan.base0;
    for (int q = 0; q < G.n_qubits; ++q) {
        if (nf.prefix.xbit(q)) { const uint64_t* z = G.col(q, false);
            for (int w = 0; w < G.words; ++w) out.sigma[w] ^= z[w]; }
        if (nf.prefix.zbit(q)) { const uint64_t* x = G.col(q, true);
            for (int w = 0; w < G.words; ++w) out.sigma[w] ^= x[w]; }
    }
    // r fair coins (same draw order as twirl_collapse).
    for (const std::vector<uint64_t>& mask : plan.coin_masks) {
        const uint8_t bit = (rng() < 0.5) ? 0 : 1;
        out.coins.push_back(bit);
        if (bit) xor_into(out.sigma, mask);
    }
    // κ kernel chain (cold path in-distribution; identical rules to twirl_collapse need_amps=false).
    if (!plan.kernel_logicals.empty()) {
        out.amps = bare;
        for (size_t j = 0; j < plan.kernel_logicals.size(); ++j) {
            const Pauli& L = plan.kernel_logicals[j];
            if (!is_hermitian(L)) {
                std::fprintf(stderr, "[twirl_collapse_record] guard: non-Hermitian kernel rep -> fallback\n");
                out.fallback = true;
                return;
            }
            const std::pair<double, double> pr = framed_expectation(out.amps, L);
            if (!(std::abs(pr.first + pr.second - 1.0) < 1e-9)) {
                std::fprintf(stderr, "[twirl_collapse_record] guard: p+ + p- != 1 -> fallback\n");
                out.fallback = true;
                return;
            }
            const int eps = out.amps.measure_pauli(L, rng());
            const uint8_t bit = (eps == +1) ? 0 : 1;
            out.coins.push_back(bit);
            const uint8_t fold_j = (j < plan.kernel_foldable.size()) ? plan.kernel_foldable[j] : 1;
            const uint8_t base_j = (j < plan.kernel_base.size()) ? plan.kernel_base[j] : 0;
            if (fold_j && (bit ^ base_j) && j < plan.kernel_masks.size() && !plan.kernel_masks[j].empty())
                xor_into(out.sigma, plan.kernel_masks[j]);
        }
    }
}

namespace {
inline uint64_t content_hash(uint64_t token, const std::vector<uint64_t>& amask,
                             const std::vector<std::pair<int, int>>& cz) {
    uint64_t h = 0xcbf29ce484222325ull ^ token;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001B3ull; };
    for (uint64_t w : amask) mix(w);
    mix(((uint64_t)cz.size() << 32) ^ (uint64_t)amask.size());
    for (const auto& e : cz) mix(((uint64_t)e.first << 32) | (uint32_t)e.second);
    return h;
}
}  // namespace

const CachedPlan* TwirlPlanCache::find(const CertifiedGroupPlanes& G,
                                       const std::vector<uint64_t>& amask,
                                       const std::vector<std::pair<int, int>>& cz) {
    const uint64_t h = content_hash(G.identity_token(), amask, cz);
    auto it = store_.find(h);
    if (it == store_.end()) return nullptr;
    for (const std::unique_ptr<CachedPlan>& p : it->second)
        if (p->key_amask == amask && p->key_cz == cz) { ++hits_; return p.get(); }
    return nullptr;
}

const CachedPlan& TwirlPlanCache::get_or_build(const CertifiedGroupPlanes& G, const DiagNormalForm& nf) {
    // Content identity: (group token, packed a-support words, sorted cz list). Prefix and γ are
    // excluded — the cached plan is exactly the prefix-free part of the law (M2: the law-except-
    // plane1 depends only on (a,cz)+group). Hot-path (2026-07-15): the old byte-wise std::string
    // key + string-hash cost ~4 µs/shot at n=298; the key is now a 64-bit FNV over the PACKED
    // content (O(n/64 + |cz|) words) with full content equality verified inside the hash bucket —
    // a hash collision can never serve a wrong plan (never-silent-wrong).
    const int n = (int)nf.a.size();
    const int NW = (n + 63) / 64;
    amask_scratch_.assign((size_t)NW, 0);
    for (int q = 0; q < n; ++q)
        if (nf.a[q] & 1) amask_scratch_[q >> 6] |= (1ULL << (q & 63));
    // NOTE: content_hash mixes the amask WORD COUNT (not n) so the disk loader and find() re-derive
    // the identical hash from the serialized/maintained key content alone.
    const uint64_t h = content_hash(G.identity_token(), amask_scratch_, nf.cz);

    std::vector<std::unique_ptr<CachedPlan>>& bucket = store_[h];
    for (const std::unique_ptr<CachedPlan>& p : bucket)
        if (p->key_amask == amask_scratch_ && p->key_cz == nf.cz) { ++hits_; return *p; }
    ++misses_;
    // Build the prefix-free law: build_shot_law with an IDENTITY prefix ⇒ det_signs = base0, and
    // every other field is prefix-independent (identical to the real-prefix law).
    DiagNormalForm nf0 = nf;
    nf0.prefix = Pauli(G.n_qubits);
    ShotLaw base = build_shot_law(G, nf0);
    auto plan = std::make_unique<CachedPlan>();
    plan->r = base.r;
    plan->kappa = base.kappa;
    plan->fallback = base.fallback;        // 2026-07-15: never cache-serve a fallback law as valid
    plan->base0 = std::move(base.det_signs);
    plan->coin_masks = std::move(base.coin_masks);
    plan->kernel_masks = std::move(base.kernel_masks);
    plan->kernel_base = std::move(base.kernel_base);
    plan->kernel_foldable = std::move(base.kernel_foldable);
    plan->kernel_logicals = std::move(base.kernel_logicals);
    plan->tier1 = std::move(base.tier1);
    plan->key_amask = amask_scratch_;
    plan->key_cz = nf.cz;
    bucket.push_back(std::move(plan));
    ++n_plans_;
    return *bucket.back();
}

// ── Disk layer (2026-07-15): TWPL v1 blob = token ‖ n_plans ‖ per-plan payload. ────────────────
namespace {
constexpr char kTwplMagic[4] = {'T', 'W', 'P', 'L'};
constexpr uint32_t kTwplVersion = 2;   // v2: adds per-plan kernel_foldable flags (V2 reachability split)

void put_pauli(plan_cache::Writer& w, const Pauli& p) {
    w.pod((int32_t)p.n); w.pod((int32_t)p.phase); w.vpod(p.x); w.vpod(p.z);
}
bool get_pauli(plan_cache::Reader& r, Pauli& p) {
    int32_t n = 0, ph = 0;
    r.pod(n); r.pod(ph); r.vpod(p.x); r.vpod(p.z);
    p.n = n; p.phase = ph;
    return r.ok;
}
}  // namespace

bool TwirlPlanCache::save_file(const std::string& path, uint64_t group_token) const {
    plan_cache::Writer w = plan_cache::begin_blob(kTwplMagic, kTwplVersion);
    w.pod(group_token);
    w.pod((uint64_t)n_plans_);
    for (const auto& kv : store_) {
        for (const std::unique_ptr<CachedPlan>& p : kv.second) {
            w.vpod(p->key_amask);
            w.vpod(p->key_cz);
            w.pod((int32_t)p->r); w.pod((int32_t)p->kappa); w.pod((uint8_t)(p->fallback ? 1 : 0));
            w.vpod(p->base0);
            w.pod((uint64_t)p->coin_masks.size());
            for (const std::vector<uint64_t>& m : p->coin_masks) w.vpod(m);
            w.pod((uint64_t)p->kernel_masks.size());
            for (const std::vector<uint64_t>& m : p->kernel_masks) w.vpod(m);
            w.vpod(p->kernel_base);
            w.vpod(p->kernel_foldable);
            w.pod((uint64_t)p->kernel_logicals.size());
            for (const Pauli& L : p->kernel_logicals) put_pauli(w, L);
            w.vpod(p->tier1.sign); w.vpod(p->tier1.active);
        }
    }
    const std::string blob = plan_cache::finish_blob(w);
    // tmp + rename (final-review Minor #5): two processes sharing an auto cache file could
    // tear a direct write; the TWPL hash makes a torn READ a silent rebuild, but atomic
    // replacement avoids losing the cache. Same-directory tmp keeps rename() atomic.
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

bool TwirlPlanCache::load_file(const std::string& path, uint64_t group_token) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string blob;
    { std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
      if (sz <= 0) { std::fclose(f); return false; }
      blob.resize((size_t)sz);
      if (std::fread(&blob[0], 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; } }
    std::fclose(f);
    plan_cache::Reader r = plan_cache::open_blob(blob, kTwplMagic);
    if (!r.ok) return false;                          // bad magic / truncated / hash mismatch: rebuild
    uint32_t ver = 0; r.pod(ver);
    if (!r.ok || ver != kTwplVersion) return false;   // version skew: rebuild
    uint64_t tok = 0, count = 0;
    r.pod(tok); r.pod(count);
    if (!r.ok || tok != group_token) return false;    // different certified group: load nothing
    for (uint64_t i = 0; i < count; ++i) {
        auto p = std::make_unique<CachedPlan>();
        r.vpod(p->key_amask);
        r.vpod(p->key_cz);
        int32_t rr = 0, kk = 0; uint8_t fb = 0;
        r.pod(rr); r.pod(kk); r.pod(fb);
        p->r = rr; p->kappa = kk; p->fallback = fb != 0;
        r.vpod(p->base0);
        uint64_t nm = 0; r.pod(nm);
        p->coin_masks.resize((size_t)nm);
        for (auto& m : p->coin_masks) r.vpod(m);
        r.pod(nm);
        p->kernel_masks.resize((size_t)nm);
        for (auto& m : p->kernel_masks) r.vpod(m);
        r.vpod(p->kernel_base);
        r.vpod(p->kernel_foldable);
        r.pod(nm);
        p->kernel_logicals.resize((size_t)nm);
        for (Pauli& L : p->kernel_logicals) if (!get_pauli(r, L)) return false;
        r.vpod(p->tier1.sign); r.vpod(p->tier1.active);
        if (!r.ok) return false;                      // any short read: stop (loaded prefix stands)
        // Re-derive the content hash via the SHARED helper (one implementation for build/find/load).
        const uint64_t h = content_hash(group_token, p->key_amask, p->key_cz);
        store_[h].push_back(std::move(p));
        ++n_plans_;
    }
    return r.at_end();
}

}  // namespace qeccore
