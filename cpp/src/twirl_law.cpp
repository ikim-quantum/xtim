#include "qeccore/twirl_kernel.hpp"
#include "qeccore/pauli_kernels.hpp"   // reset_pauli_inplace
#include "twirl_law_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>

// build_shot_law — joint lattice law for the diagonal twirl kernel (spec §5 steps 3–5).
//
// FAITHFUL C++ port of scripts/twirl_diag_reference.py::build_shot_law (corpus-validated 41/41,
// census-validated 0-mismatch). Structure and every classification decision mirror the reference;
// the three Phase-A discoveries (lattice-subspace classification, joint base point, rank(Π)==r
// assert) are baked in exactly. See twirl_kernel.hpp for the field/consumer contract.

namespace qeccore {
namespace {

// ────────────────────────────────── GF(2) bit-vector helpers ────────────────────────────────
// A bit-vector is a std::vector<uint64_t> of ceil(nbits/64) words. All routines below take the
// bit width explicitly where it matters; word count is derived from the vectors themselves.
using BV = std::vector<uint64_t>;
inline int nwords(int nbits) { return (nbits + 63) / 64; }
inline bool bget(const BV& v, int i) { return (v[i >> 6] >> (i & 63)) & 1ULL; }
inline void bset(BV& v, int i) { v[i >> 6] |= (1ULL << (i & 63)); }
inline void bxor(BV& a, const BV& b) { for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i]; }
inline bool bany(const BV& v) { for (uint64_t w : v) if (w) return true; return false; }
inline int bfirst(const BV& v) {                       // lowest set bit index, -1 if none
    for (size_t w = 0; w < v.size(); ++w) if (v[w]) return (int)w * 64 + __builtin_ctzll(v[w]);
    return -1;
}

// Pooled row storage (perf-engine §2.2): the live rows are the prefix [0, n); rows beyond keep their capacity
// across builds, so a cleared pool re-fills without allocating. Every live row is fully assigned before use.
struct RowPool {
    std::vector<BV> rows;
    size_t n = 0;
    BV& push() { if (n == rows.size()) rows.emplace_back(); return rows[n++]; }
    void pop() { --n; }
    void clear() { n = 0; }
    size_t size() const { return n; }
    BV& operator[](size_t i) { return rows[i]; }
    const BV& operator[](size_t i) const { return rows[i]; }
};

// Reduced row-echelon (fully back-substituted) of GF(2) bit-rows. Each pivot column carries a 1
// in ITS OWN basis row only — matches the reference _rref (the 2026-07-15 back-substitution fix is
// load-bearing for the nullspace / intersect quantities). The live rows are R[0, piv.size()).
struct Rref { RowPool R; std::vector<int> piv; void clear() { R.clear(); piv.clear(); } };
// Insert `v` (ALREADY reduced against the basis, nonzero): back-substitute it into the existing rows and append —
// the per-row body of the sequential RREF, exposed for the incremental use below (perf-engine §2.5: an RREF built
// row by row is the same object whether the rows arrive at once or one at a time).
void rref_insert(Rref& out, const BV& v, int pc) {
    for (size_t k = 0; k < out.piv.size(); ++k) if (bget(out.R[k], pc)) bxor(out.R[k], v);
    BV& dst = out.R.push();
    dst.assign(v.begin(), v.end());
    out.piv.push_back(pc);
}
template <class Rows>
void rref_into(const Rows& rows, size_t nrows, Rref& out) {
    out.clear();
    for (size_t i = 0; i < nrows; ++i) {
        BV& v = out.R.push();                              // the candidate's slot (returned if it reduces to 0)
        const BV& r = *rows[i];
        v.assign(r.begin(), r.end());
        for (size_t k = 0; k < out.piv.size(); ++k) if (bget(v, out.piv[k])) bxor(v, out.R[k]);
        const int pc = bfirst(v);
        if (pc < 0) { out.R.pop(); continue; }
        for (size_t k = 0; k < out.piv.size(); ++k) if (bget(out.R[k], pc)) bxor(out.R[k], v);
        out.piv.push_back(pc);
    }
}
// Row accessors for rref_into: a vector of pointers, or a pool / vector of rows.
struct PtrRows { const std::vector<const BV*>& p; const BV* operator[](size_t i) const { return p[i]; } };
struct PoolRows { const RowPool& p; const BV* operator[](size_t i) const { return &p[i]; } };
struct VecRows { const std::vector<BV>& p; const BV* operator[](size_t i) const { return &p[i]; } };

void reduce_vec_inplace(BV& v, const Rref& b) {
    for (size_t k = 0; k < b.piv.size(); ++k) if (bget(v, b.piv[k])) bxor(v, b.R[k]);
}

// Support-directed reduction against the group's pure-Z RREF (2026-07-16): scan v's set bits
// ascending and jump via the column→row map instead of testing every rref row. Full reduced
// echelon: a pivot column occurs only in its own row, so no earlier pivot re-appears; XOR-only
// reduction => identical result to the row-walk overload.
void reduce_vec_gz_inplace(BV& v, const CertifiedGroupPlanes::GroupZCache& GZ) {
    const int NW = (int)v.size();
    for (int wq = 0; wq < NW; ++wq) {
        uint64_t word = v[wq];
        while (word) {
            const int b = __builtin_ctzll(word);
            const int q = wq * 64 + b;
            const int r = q < (int)GZ.gz_col_to_row.size() ? GZ.gz_col_to_row[q] : -1;
            if (r >= 0) {
                bxor(v, GZ.gz_rref[r]);
                word = (b == 63) ? 0 : (v[wq] & (~0ULL << (b + 1)));
            } else {
                word &= word - 1;
            }
        }
    }
}

// The elimination scratch of left_deps / solve_gf2 (pooled; one per thread, see BuildScratch).
struct ElimScratch { RowPool v, c; std::vector<int> pc, rhs; };

// Basis of the LEFT null space: combos c ∈ GF(2)^m with Σ_i c_i rows[i] = 0 (reference _left_deps).
// rows are n-bit; the combos are m-bit (m = the live row count). A zero row is its own dependency. The combos
// land in `deps` (cleared first), in the reference's order.
template <class Rows>
void left_deps_into(const Rows& rows, int m, RowPool& deps, ElimScratch& es) {
    deps.clear();
    es.v.clear(); es.c.clear(); es.pc.clear();
    for (int i = 0; i < m; ++i) {
        BV& v = es.v.push();
        const BV& r = *rows[(size_t)i];
        v.assign(r.begin(), r.end());
        BV& c = es.c.push();
        c.assign((size_t)nwords(m), 0);
        bset(c, i);
        for (size_t t = 0; t < es.pc.size(); ++t) if (bget(v, es.pc[t])) { bxor(v, es.v[t]); bxor(c, es.c[t]); }
        const int pc = bfirst(v);
        if (pc < 0) {
            BV& d = deps.push();
            d.assign(c.begin(), c.end());
            es.v.pop(); es.c.pop();
        } else {
            es.pc.push_back(pc);
        }
    }
}
// The value form (the once-per-group G_z build).
std::vector<BV> left_deps(const std::vector<BV>& rows, int /*n*/) {
    ElimScratch es;
    RowPool deps;
    left_deps_into(VecRows{rows}, (int)rows.size(), deps, es);
    std::vector<BV> out;
    for (size_t k = 0; k < deps.size(); ++k) out.push_back(deps[k]);
    return out;
}

// Particular solution x of A x = b over GF(2) (free vars = 0); reference _solve_gf2. A_rows are
// ncols-bit; b matches A_rows one bit each. The system is ASSERTED consistent (loud throw on an inconsistent
// row). x is written into `out` (assigned).
void solve_gf2_into(const RowPool& A_rows, const std::vector<int>& b, int ncols, BV& out, ElimScratch& es) {
    es.v.clear(); es.pc.clear(); es.rhs.clear();
    for (size_t i = 0; i < A_rows.size(); ++i) {
        BV& v = es.v.push();
        v.assign(A_rows[i].begin(), A_rows[i].end());
        int rr = b[i] & 1;
        for (size_t t = 0; t < es.pc.size(); ++t) if (bget(v, es.pc[t])) { bxor(v, es.v[t]); rr ^= es.rhs[t]; }
        const int pc = bfirst(v);
        if (pc < 0) {
            if (rr != 0) throw std::logic_error("build_shot_law: inconsistent base-point system");
            es.v.pop();
            continue;
        }
        for (size_t t = 0; t < es.pc.size(); ++t) if (bget(es.v[t], pc)) { bxor(es.v[t], v); es.rhs[t] ^= rr; }
        es.pc.push_back(pc);
        es.rhs.push_back(rr);
    }
    out.assign((size_t)nwords(ncols), 0);
    for (size_t t = 0; t < es.pc.size(); ++t) if (es.rhs[t]) bset(out, es.pc[t]);
}

// ─────────────────────────── residual-frame quantities on the group ─────────────────────────
// h_i = R† g_i R = (−1)^{⟨P,g⟩}·i^{c}·g_i·Z^{M x_i} as an exact Pauli, global phase dropped
// (reference conj_by_nf). c = (−Σ a_q x_q − 2 Σ_{(q,q')∈cz} x_q x_q') mod 4. Word-level over NW words
// (amask packs a_q; padding bits are 0 in both operands ⇒ identical to the per-qubit reference).
// In-place variant (2026-07-16 plan-build optimization): writes into a pooled Pauli whose
// x/z buffers keep their capacity across builds — the by-value version allocated two vectors
// per touched generator per build. Every field is fully overwritten: byte-identical result.
// h-FREE conjugation (2026-07-16, the h-free sign derivation): the conjugated generator is
//   h_i = i^{φ_i} · X^{x_i} Z^{z_i ⊕ v_i},   φ_i = g_i.phase + 2·α_i + c_i  (mod 4),
//   c_i  = (−lin_i − 2·quad_i) mod 4,
// and every phase ingredient is a COLUMNAR plane bit: α_i = ⟨P, g_i⟩ (prefix_plane1),
// lin_i mod 4 = ones_i + 2·twos_i (the same carry-save planes tier1 uses), quad_i mod 2 =
// the per-pair AND-XOR plane. So h_i is never materialised — products walk (g_i, v_i, φ_i)
// directly (identical word ops, identical phases: b.x = g.x, b.z = g.z ⊕ v_i, b.phase = φ_i)
// and the rare consumers that need h_i as a Pauli (teeth, κ>0 kernel products) assemble it
// on demand into scratch.

}  // namespace

// Group-only law substrate (see twirl_planes.hpp): the pure-Z group content G_z = { Σ c_i z_i :
// Σ c_i x_i = 0 }, RREF'd. Depends only on the certified group — computed lazily ONCE and cached on
// the planes object (the old build recomputed it per cold key). Defined here for the GF(2) helpers.
namespace {
// Free helper (not a member) so the anon-namespace GF(2) `rref` isn't shadowed by the planes'
// `rref` data member inside member-function scope.
void build_group_z_cache(const CertifiedGroupPlanes& P, CertifiedGroupPlanes::GroupZCache& out) {
    const int NW = nwords(P.n_qubits);
    std::vector<BV> xrows((size_t)P.n_gens);
    for (int i = 0; i < P.n_gens; ++i)
        xrows[i].assign(P.gens[i].x.begin(), P.gens[i].x.begin() + NW);
    std::vector<BV> Gz_gens;
    for (const BV& c : left_deps(xrows, P.n_qubits)) {
        BV zc(NW, 0);
        for (int i = 0; i < P.n_gens; ++i)
            if (bget(c, i)) for (int w = 0; w < NW; ++w) zc[w] ^= P.gens[i].z[w];
        if (bany(zc)) Gz_gens.push_back(std::move(zc));
    }
    Rref R;
    rref_into(VecRows{Gz_gens}, Gz_gens.size(), R);
    out.gz_rref.assign(R.R.rows.begin(), R.R.rows.begin() + (long)R.piv.size());
    out.gz_piv  = std::move(R.piv);
    out.gz_col_to_row.assign((size_t)P.n_qubits, -1);
    for (size_t r = 0; r < out.gz_piv.size(); ++r) out.gz_col_to_row[out.gz_piv[r]] = (int)r;
}
}  // namespace

const CertifiedGroupPlanes::GroupZCache& CertifiedGroupPlanes::group_z_cache() const {
    if (gzc_ready_) return gzc_;
    build_group_z_cache(*this, gzc_);
    gzc_ready_ = true;
    return gzc_;
}

// Ephemeral stage profiling (QEC_TW_PROF=1): per-build stage accumulators, dumped at exit.
// Timing only — no behavioral change; guarded so the default path costs one branch.
namespace {
struct LawProf {
    bool on = std::getenv("QEC_TW_PROF") != nullptr;
    double tier1 = 0, cone = 0, lattice = 0, coins = 0, cdet = 0, kernel = 0, base = 0;
    long builds = 0;
    // S2-pre: per-κ whole-build totals and the κ>0 W = C_det^⊥ (nullspace) sub-stage.
    long builds_k0 = 0, builds_k1 = 0;
    double total_k0 = 0, total_k1 = 0, nullsp = 0;
    long sumT = 0, sumcz = 0, sumA = 0, sumCdet = 0, suma = 0;
    ~LawProf() {
        if (on && builds)
            std::fprintf(stderr,
                "[build_shot_law PROF] builds=%ld | per-build us: tier1=%.1f cone/h=%.1f "
                "lattice=%.1f coins=%.1f cdet=%.1f kernel=%.1f base=%.1f | "
                "avg T=%.1f |cz|=%.1f |a|=%.1f A=%.1f |Cdet|=%.1f\n",
                builds, tier1 * 1e6 / builds, cone * 1e6 / builds, lattice * 1e6 / builds,
                coins * 1e6 / builds, cdet * 1e6 / builds, kernel * 1e6 / builds,
                base * 1e6 / builds,
                (double)sumT / builds, (double)sumcz / builds, (double)suma / builds,
                (double)sumA / builds, (double)sumCdet / builds);
        if (on && builds)
            std::fprintf(stderr,
                "[build_shot_law PROF] kappa=0 builds=%ld total=%.2f us/build | kappa>0 builds=%ld "
                "total=%.2f us/build nullspace=%.3f us/build (%.3f%% of a kappa>0 build)\n",
                builds_k0, builds_k0 ? total_k0 * 1e6 / builds_k0 : 0.0,
                builds_k1, builds_k1 ? total_k1 * 1e6 / builds_k1 : 0.0,
                builds_k1 ? nullsp * 1e6 / builds_k1 : 0.0,
                total_k1 > 0 ? 100.0 * nullsp / total_k1 : 0.0);
    }
};
LawProf g_lawprof;
using lawclk = std::chrono::steady_clock;

// Per-thread scratch pools (2026-07-16 plan-build optimization): the cone stage allocated
// ~3 vectors per touched generator per build (h's x/z + the dressing BV) — the measured
// bottleneck (24.4 of 55 µs/build on d5). Pooled buffers keep capacity across builds and
// every entry is fully overwritten before use — byte-identical results.
struct BuildScratch {
    std::vector<BV> v;                 // dressings (per touched generator)
    std::vector<uint8_t> phi;          // h_i phases (h-free derivation; see above)
    Pauli hk;                          // on-demand h assembly scratch (teeth / κ>0)
    Pauli sc;                          // b_c product scratch
    std::vector<int> ti;               // teeth touched-inactive candidates
    RowPool bred;                      // reduced dressing rows (cdet / lattice-D stages)
    std::vector<BV> pi;                // Π rows (coin-mask source rows)
    std::vector<int> tidx;
    std::vector<int> pos_of;
    std::vector<const BV*> row_ptrs;   // pointer rows for the rref overload
    std::vector<BV> vcol;              // per-qubit generator-bitmask dressing columns
    std::vector<uint8_t> vcol_used;    // dirty flags for vcol (cleared via ulist)
    std::vector<int> ulist;            // dirty qubit list
    // perf-engine §2.2: the per-build vectors below were fresh heap objects on every build (the measured alloc
    // churn, 12-15% of the d5 steady state). Every entry is fully assigned before use — byte-identical results.
    BV amask, touched, active, qplane, ones, twos, p1, kred;
    Rref vb, coin, det;                // V basis, the coin masks, the D ∪ kernel-direction basis
    RowPool prow, kz, dres, dpool, dsp, cdet, kdirs;
    ElimScratch es;
    std::vector<int> aidx, bvals;
    std::vector<const BV*> pi_ptrs;
    Pauli zu;                          // the kernel direction Z^u probe
};
thread_local BuildScratch g_bs;
// S8-2 hygiene: ONE TLS access per build. The scratch is reached through this out-of-line accessor and held in a local
// reference; with g_bs named directly GCC re-materialised the TLS address (a __tls_get_addr call in a -fPIC module) at
// each use (86 call sites in build_shot_law_).
__attribute__((noinline)) BuildScratch& build_scratch_() { return g_bs; }
}  // namespace

// J-a N6 (record-space B, switched OFF in production): ONE body for both entries. core == nullptr is
// build_shot_law (production, unchanged); core != nullptr is twirl_law_detail::build_shot_law_core: the same
// computation WITHOUT the κ>0 σ-fold stage (qrows, W, preimage solves, dual basis, its asserts) and without
// kernel_base, exporting the record-space inputs instead (twirl_law_internal.hpp, ShotLawCore).
// perf-engine §2.3 / §2.6: `teeth` = the caller's self-check window. Inside it the builder re-verifies two
// mechanisms that are otherwise pure overhead on the production path: the Tier-1 planes derived below
// against the standalone tier1_signs, and sign(h_i) == formula on up to 4 touched-inactive generators by
// exact reduction. Outside the window both checks are skipped; every output is identical (they only throw).
static ShotLaw build_shot_law_(const CertifiedGroupPlanes& G, const DiagNormalForm& nf,
                               twirl_law_detail::ShotLawCore* core, bool teeth) {
    ShotLaw law;
    BuildScratch& bs = build_scratch_();
    const int n = G.n_qubits;
    const int ng = G.n_gens;
    const int GW = nwords(ng);

    if (!nf.diagonal_class) { law.fallback = true; return law; }

    auto lp = g_lawprof.on ? lawclk::now() : lawclk::time_point();
    const auto lp_start = lp;
    auto lp_mark = [&](double& acc) {
        if (!g_lawprof.on) return;
        auto now = lawclk::now();
        acc += std::chrono::duration<double>(now - lp).count();
        lp = now;
    };
    if (g_lawprof.on) ++g_lawprof.builds;

    lp_mark(g_lawprof.tier1);

    // ── TOUCHED cone (cold-build restructure, 2026-07-15): only generators whose X-support meets
    // supp(M) = supp(a)∪cz-endpoints can carry a nonzero dressing v_i or a plane-2/3 sign; every
    // untouched generator has v_i = 0 and h_i = ±g_i with the plane-1 sign EXACTLY (conjugation by P
    // alone — no assumption, the C-layer commutes term-by-term). The old ng-scale loop materialised
    // v/h/xrows/zrows for all 297 generators per cold key; T = |touched| is lightcone-sized. ──
    const int NW = nwords(n);
    BV& amask = bs.amask;                             // amask[q]=1 ⇔ a_q=1 (word-level dressing S-part)
    amask.assign((size_t)NW, 0);
    for (int q = 0; q < n; ++q) if (nf.a[q]) bset(amask, q);
    // ── COLUMNAR dressing construction (2026-07-16): the old loop walked nf.cz once PER
    // TOUCHED GENERATOR (T×|cz| branchy tests — the measured cone bottleneck, ~20 µs/build
    // on d5 at T≈28, |cz|≈113). Build the dressing matrix column-wise instead:
    //   vcol[q] (generator bitmask) = a_q·xcol(q) ⊕ Σ_{(q,q')∈cz} xcol(q') ⊕ Σ_{(q'',q)∈cz} xcol(q'')
    // (|cz|·2·GW word XORs), then scatter the nonzeros to the per-generator rows
    // (Σ|supp v_i| bit-sets). Identical v_i bits, identical downstream everything.
    // Grind pass: the dirty list U = supp(a) ∪ cz-endpoints is exactly the touched-cone seed
    // set DEDUPED, so `touched` now ORs each xcol once (the old or_col walk paid ~2|cz| ORs
    // for ~|U| distinct qubits — identical mask, fewer word ops).
    auto& vcol = bs.vcol;
    auto& used = bs.vcol_used;
    auto& ulist = bs.ulist;
    if ((int)vcol.size() < n) vcol.resize(n);
    if ((int)used.size() < n) used.assign((size_t)n, 0);
    ulist.clear();
    {
        auto touch_q = [&](int q) {
            if (!used[q]) { used[q] = 1; ulist.push_back(q); vcol[q].assign(GW, 0); }
            return &vcol[q];
        };
        auto xor_col = [&](BV& dst, int q) {
            const uint64_t* xc = G.col(q, true);
            for (int t = 0; t < GW; ++t) dst[t] ^= xc[t];
        };
        for (int q = 0; q < n; ++q)
            if (nf.a[q]) xor_col(*touch_q(q), q);            // a-part: v_i[q] = a_q ∧ x_i[q]
        for (const auto& e : nf.cz) {
            xor_col(*touch_q(e.first), e.second);            // v_i[q] ^= x_i[q']
            xor_col(*touch_q(e.second), e.first);            // v_i[q'] ^= x_i[q]
        }
    }
    BV& touched = bs.touched;
    touched.assign((size_t)GW, 0);
    for (int q : ulist) {
        const uint64_t* xc = G.col(q, true);
        for (int t = 0; t < GW; ++t) touched[t] |= xc[t];
    }
    std::vector<int>& tidx = bs.tidx;               // touched generator indices, ascending
    tidx.clear();
    for (int w = 0; w < GW; ++w) { uint64_t bits = touched[w];
        while (bits) { tidx.push_back(w * 64 + __builtin_ctzll(bits)); bits &= bits - 1; } }
    const int T = (int)tidx.size();

    std::vector<BV>& v = bs.v;                      // dressings, indexed by touched position
    std::vector<uint8_t>& phi = bs.phi;             // h_i phase bits (h-free derivation)
    if ((int)v.size() < T) v.resize(T);
    phi.assign((size_t)T, 0);
    std::vector<int>& pos_of = bs.pos_of;           // generator index -> touched position
    pos_of.assign(ng, -1);
    // active_i = (v_i != 0) = OR over the dressing columns (perf-engine §2.3: this IS tier1_signs' active
    // mask — its M-row patterns pat_j are exactly vcol[j], OR-ed over the same dirty set — so the plane
    // is derived here once instead of rebuilt by tier1_signs; inside the self-check window the two are
    // compared bit for bit below).
    BV& active = bs.active;
    active.assign((size_t)GW, 0);
    for (int k = 0; k < T; ++k) pos_of[tidx[k]] = k;
    {
        for (int k = 0; k < T; ++k) v[k].assign(NW, 0);
        for (int q : ulist) {
            const BV& col = vcol[q];
            for (int w = 0; w < GW; ++w) {
                uint64_t word = col[w];
                active[w] |= word;
                while (word) {
                    const int i = w * 64 + __builtin_ctzll(word); word &= word - 1;
                    v[pos_of[i]][q >> 6] |= (1ULL << (q & 63));
                }
            }
            used[q] = 0;                                     // clear dirty flag for the next build
        }
        // quad-parity plane: bit i = (#{(q,q')∈cz : x_i[q]∧x_i[q']}) mod 2 — one AND-XOR per pair.
        BV& qplane = bs.qplane;
        qplane.assign((size_t)GW, 0);
        for (const auto& e : nf.cz) {
            const uint64_t* xa = G.col(e.first, true);
            const uint64_t* xb = G.col(e.second, true);
            for (int t = 0; t < GW; ++t) qplane[t] ^= xa[t] & xb[t];
        }
        // lin mod 4 planes (carry-save over supp(a), same construction as tier1 plane 2):
        // ones = bit0, twos = bit1 of Σ a_q x_{i,q}.
        BV& ones = bs.ones;
        BV& twos = bs.twos;
        ones.assign((size_t)GW, 0);
        twos.assign((size_t)GW, 0);
        for (int q = 0; q < n; ++q) {
            if (!nf.a[q]) continue;
            const uint64_t* c = G.col(q, true);
            for (int w = 0; w < GW; ++w) {
                const uint64_t carry = ones[w] & c[w];
                ones[w] ^= c[w];
                twos[w] ^= carry;
            }
        }
        // α_i bits: identically zero for the memo's identity prefix (the common case) —
        // skip the plane entirely there (prefix_plane1 of an empty support is all-zero).
        bool haspfx = false;
        for (size_t w2 = 0; w2 < nf.prefix.x.size(); ++w2)
            if (nf.prefix.x[w2] | nf.prefix.z[w2]) { haspfx = true; break; }
        BV& p1 = bs.p1;
        if (haspfx) prefix_plane1_into(G, nf.prefix, p1);
        for (int k = 0; k < T; ++k) {
            const int i = tidx[k];
            const int lin4 = (int)bget(ones, i) + 2 * (int)bget(twos, i);
            const int ci = ((-lin4 - 2 * (int)bget(qplane, i)) % 4 + 4) % 4;
            const int a1 = haspfx ? (int)bget(p1, i) : 0;
            phi[k] = (uint8_t)((G.gens[i].phase + 2 * a1 + ci) & 3);
        }
        // Tier-1 (spec §5 step 2) from the planes already built: sign = plane1 (prefix) ^ plane2 (the
        // carry-save 2s bit over supp(a)) ^ plane3 (the CZ AND-XOR plane) — the same three column
        // XOR sets tier1_signs forms, word for word; active = the dressing OR above.
        law.tier1.sign.assign((size_t)GW, 0);
        for (int w = 0; w < GW; ++w) law.tier1.sign[w] = (haspfx ? p1[w] : 0ULL) ^ twos[w] ^ qplane[w];
        law.tier1.active = active;
    }
    if (teeth) {                                      // self-check window: the standalone planes agree bit for bit
        const Tier1Result t1 = tier1_signs(G, nf);
        if (t1.sign != law.tier1.sign || t1.active != law.tier1.active)
            throw std::logic_error("build_shot_law: derived Tier-1 planes disagree with tier1_signs (engine bug)");
    }

    if (g_lawprof.on) {
        g_lawprof.sumT += T;
        g_lawprof.sumcz += (long)nf.cz.size();
        long na = 0;
        for (int q = 0; q < n; ++q) if (nf.a[q]) ++na;
        g_lawprof.suma += na;
    }
    lp_mark(g_lawprof.cone);
    // ── lattice classification (spec §5 step 4) ──
    std::vector<const BV*>& Vrows = bs.row_ptrs;
    Vrows.clear();
    for (int k = 0; k < T; ++k) if (bget(active, tidx[k])) Vrows.push_back(&v[k]);
    Rref& V_basis = bs.vb;
    rref_into(PtrRows{Vrows}, Vrows.size(), V_basis);
    const size_t nV = V_basis.piv.size();

    // G_z (pure-Z group content) is a GROUP-ONLY quantity — hoisted to a lazy per-group cache
    // (byte-identical computation, no longer rebuilt per cold key).
    const CertifiedGroupPlanes::GroupZCache& GZ = G.group_z_cache();

    // Kz = { u ∈ V : u·x_j = 0 ∀j } — the old 2n-wide Zassenhaus against an N_z basis is replaced by
    // the kernel of α ↦ Π-row(Σ α_k Vb_k) in the |V|-dim α-space, with Π-rows built columnar from the
    // group's X-plane. Same span always; the basis CHOICE can differ from the old construction only
    // when κ>0 ∧ |D|>0 (downstream: kernel reps canonicalise mod the group, masks/base re-solve per
    // direction — validated by the κ=1 gates) and κ=0 laws are byte-identical (Kz feeds only
    // span/dim quantities there).
    RowPool& Kz = bs.kz;
    Kz.clear();
    {
        RowPool& prow = bs.prow;
        prow.clear();
        for (size_t k = 0; k < nV; ++k) {
            BV& row = prow.push();
            row.assign((size_t)GW, 0);
            const BV& u = V_basis.R[k];
            for (int w = 0; w < NW; ++w) {
                uint64_t word = u[w];
                while (word) { const int q = w * 64 + __builtin_ctzll(word); word &= word - 1;
                    const uint64_t* xc = G.col(q, true);
                    for (int t = 0; t < GW; ++t) row[t] ^= xc[t]; }
            }
        }
        RowPool& deps = bs.dsp;
        left_deps_into(PoolRows{prow}, (int)nV, deps, bs.es);
        for (size_t i = 0; i < deps.size(); ++i) {
            const BV& c = deps[i];
            BV& u = Kz.push();
            u.assign((size_t)NW, 0);
            for (size_t k = 0; k < nV; ++k) if (bget(c, k)) bxor(u, V_basis.R[k]);
        }
    }
    // D = V ∩ G_z (= Kz ∩ G_z since G_z ⊆ N_z); only span(D)/|D| feed downstream.
    RowPool& D = bs.dpool;
    D.clear();
    {
        RowPool& Dres = bs.dres;
        Dres.clear();
        for (size_t k = 0; k < Kz.size(); ++k) { BV& d = Dres.push(); d.assign(Kz[k].begin(), Kz[k].end()); reduce_vec_gz_inplace(d, GZ); }
        RowPool& deps = bs.dsp;
        left_deps_into(PoolRows{Dres}, (int)Kz.size(), deps, bs.es);
        for (size_t i = 0; i < deps.size(); ++i) {
            const BV& c = deps[i];
            BV& d = D.push();
            d.assign((size_t)nwords(n), 0);
            for (size_t k = 0; k < Kz.size(); ++k) if (bget(c, k)) bxor(d, Kz[k]);
        }
    }
    lp_mark(g_lawprof.lattice);
    law.kappa = (int)Kz.size() - (int)D.size();
    law.r = (int)nV - (int)Kz.size();

    // ── coin masks: rowspace(Π), Π_ij = v_i·x_j; rank(Π) MUST equal r (Phase-A bug #3) ──
    // Columnar (spec §6): Π row i (bit j = v_i·x_j) = XOR over q∈supp(v_i) of xcol(q) (bit j = x_j[q]).
    // Replaces the O(ng²·n/64) row-wise popcount with O(Σ|supp v_i|·GW) column XORs into the group's
    // precomputed X-plane. Same rows, same order (only nonzero rows pushed) ⇒ byte-identical coin.R.
    std::vector<BV>& Pi = bs.pi;                    // pooled Π-row storage (2026-07-16)
    if (Pi.size() < (size_t)T) Pi.resize(T);          // pre-size: Pi_ptrs points into Pi, and a
                                                      // mid-loop grow would dangle those pointers
    std::vector<const BV*>& Pi_ptrs = bs.pi_ptrs;
    Pi_ptrs.clear();
    size_t npi = 0;
    for (int k = 0; k < T; ++k) {
        if (!bget(active, tidx[k])) continue;         // inactive rows were all-zero → skipped before too
        BV& row = Pi[npi];
        row.assign(GW, 0);
        for (int w = 0; w < NW; ++w) {
            uint64_t word = v[k][w];
            while (word) { const int q = w * 64 + __builtin_ctzll(word); word &= word - 1;
                const uint64_t* xc = G.col(q, true);
                for (int t = 0; t < GW; ++t) row[t] ^= xc[t];
            }
        }
        if (bany(row)) { Pi_ptrs.push_back(&row); ++npi; }
    }
    Rref& coin = bs.coin;
    rref_into(PtrRows{Pi_ptrs}, Pi_ptrs.size(), coin);
    if ((int)coin.piv.size() != law.r) throw std::logic_error("build_shot_law: coin rank != r (rank(Pi) assert)");

    lp_mark(g_lawprof.coins);
    // ── C_det = { c : v_c ∈ G_z } (combos whose product dressing is in-group): the constraints the
    // deterministic base point must satisfy AND the constraints every non-deterministic σ-flip mask
    // (coins AND kernel) must preserve. ──
    //
    // Restructure (2026-07-15, byte-identical): rowspace(C_det) = span{e_i : i inactive} ⊕ the
    // ACTIVE-supported dependencies (a zero dressing is its own singleton dependency, and left_deps
    // provenance never mixes a zero row into a nonzero-row combo). The old code ran the ng-scale
    // left_deps + an ng-row base solve per cold key; the solver output is a pure function of the
    // AUGMENTED ROWSPACE (unique RREF), so solving the small active system and pinning the inactive
    // singleton bits directly (their constraint value b(e_i) = sign(h_i) = the Tier-1 formula — the
    // invariant the old post-solve assert enforced on every shot to date; bounded-sample re-verified
    // below) reproduces the old base BIT FOR BIT. `Cdet` below holds the ACTIVE combos only; the κ>0
    // branch appends the inactive singletons where the full space is required (fold-mask ⊥ checks).
    std::vector<int>& aidx = bs.aidx;                 // active generator indices, ascending
    aidx.clear();
    for (int k = 0; k < T; ++k) if (bget(active, tidx[k])) aidx.push_back(tidx[k]);
    const int A = (int)aidx.size();
    RowPool& BredA = bs.bred;                         // pooled reduced-dressing rows
    BredA.clear();
    for (int a2 = 0; a2 < A; ++a2) {
        BV& b2 = BredA.push();
        b2.assign(v[pos_of[aidx[a2]]].begin(), v[pos_of[aidx[a2]]].end());
        reduce_vec_gz_inplace(b2, GZ);
    }
    RowPool& Cdet = bs.cdet;
    {
        RowPool& deps = bs.dsp;
        left_deps_into(PoolRows{BredA}, A, deps, bs.es);
        Cdet.clear();
        for (size_t i = 0; i < deps.size(); ++i) {    // combos in A-dim → scatter to ng-bit
            const BV& c = deps[i];
            BV& cs = Cdet.push();
            cs.assign((size_t)GW, 0);
            for (int a2 = 0; a2 < A; ++a2) if (bget(c, a2)) bset(cs, aidx[a2]);
        }
    }

    if (g_lawprof.on) { g_lawprof.sumA += (long)aidx.size(); g_lawprof.sumCdet += (long)Cdet.size(); }
    lp_mark(g_lawprof.cdet);
    // ── kernel logical reps: extend a basis of D to Kz; reduce each new Z^u to a logical ──
    // perf-engine §2.5: the basis of D ∪ {accepted directions} grows by ONE row per accepted u — the sequential
    // RREF's own per-row step (reduce, back-substitute, append) on the already-reduced residual, exactly the
    // state a from-scratch rref(D ∪ dirs) reaches after the same rows in the same order.
    RowPool& kernel_dirs = bs.kdirs;
    kernel_dirs.clear();
    {
        Rref& det = bs.det;
        rref_into(PoolRows{D}, D.size(), det);
        for (size_t i = 0; i < Kz.size(); ++i) {
            const BV& u = Kz[i];
            BV& red = bs.kred;
            red.assign(u.begin(), u.end());
            reduce_vec_inplace(red, det);
            const int pc = bfirst(red);
            if (pc >= 0) {
                BV& k2 = kernel_dirs.push();
                k2.assign(u.begin(), u.end());
                rref_insert(det, red, pc);
            }
        }
    }
    for (size_t i = 0; i < kernel_dirs.size(); ++i) {
        const BV& u = kernel_dirs[i];
        Pauli& zu = bs.zu;
        reset_pauli_inplace(zu, n);
        for (int q = 0; q < n; ++q) if (bget(u, q)) zu.setz(q);
        Membership m = G.reduce(zu);
        law.kernel_logicals.push_back(m.rep);       // LOGICAL: reduced normaliser rep, exact phase
    }


    lp_mark(g_lawprof.kernel);
    // ── valid base point (spec F7 / Phase-A bug #2): the Tier-1 formula is a per-generator sign,
    // not a joint coset rep. For c ∈ C_det, ∏ h_i ∈ ±G with a definite sign b_c (exact Pauli
    // reduction). Solve c·σ = b_c for one σ; assert σ ≡ formula on inactive.
    // MEMO NOTE: b_c = b0_c ⊕ ⟨P, g_c⟩ splits into an (a,cz)-only base0 plus the prefix plane1;
    // det_signs = base0 ⊕ plane1 (see prefix_plane1 / TwirlPlanCache). We compute the full b_c here
    // (faithful, byte-identical to the reference) — the split is consumed by the Task-10 memo.
    std::vector<int>& bvals = bs.bvals;
    bvals.assign(Cdet.size(), 0);
    Pauli& Sc = bs.sc;                              // pooled product scratch
    Sc.n = n;
    Sc.x.assign((size_t)NW, 0ULL);
    Sc.z.assign((size_t)NW, 0ULL);
    for (size_t ci = 0; ci < Cdet.size(); ++ci) {
        std::fill(Sc.x.begin(), Sc.x.end(), 0ULL);
        std::fill(Sc.z.begin(), Sc.z.end(), 0ULL);
        Sc.phase = 0;
        const BV& cc = Cdet[ci];                      // Sc ·= h_i over i∈c, in place (set-bits only)
        for (int cw = 0; cw < GW; ++cw) {
            uint64_t bits = cc[cw];
            while (bits) {
                const int i = cw * 64 + __builtin_ctzll(bits); bits &= bits - 1;
                const int k2 = pos_of[i];             // combo support ⊆ active ⊆ touched
                const Pauli& g = G.gens[i];           // h_i = i^{φ}·X^{g.x}Z^{g.z⊕v} (h-free)
                const BV& vi = v[k2];
                int sgn = 0;
                for (int w = 0; w < NW; ++w) sgn += __builtin_popcountll(Sc.z[w] & g.x[w]);
                for (int w = 0; w < NW; ++w) { Sc.x[w] ^= g.x[w]; Sc.z[w] ^= g.z[w] ^ vi[w]; }
                Sc.phase = (Sc.phase + phi[k2] + 2 * (sgn & 1)) & 3;
            }
        }
        int rphase = 0;
        if (G.reduce_cheap(Sc, rphase) != Membership::IN_GROUP)
            throw std::logic_error("build_shot_law: deterministic combo did not reduce into ±G");
        bvals[ci] = (rphase >> 1) & 1;
    }
    // Base assembly (byte-identical to the old ng-row solve, see the C_det restructure note): solve
    // the active system (free vars = 0, exactly the old RREF's behaviour on the active columns) and
    // pin every inactive bit to the Tier-1 formula — the old solve pinned them to sign(h_i) via the
    // singleton constraints, and sign(h_i) == formula is the invariant its own assert enforced.
    BV base;
    solve_gf2_into(Cdet, bvals, ng, base, bs.es);
    for (int w = 0; w < GW; ++w) base[w] |= law.tier1.sign[w] & ~active[w];
    // Teeth (bounded; perf-engine §2.6: self-check window only): re-verify sign(h_i) == formula by exact
    // reduction on up to 4 touched-INACTIVE generators — the population where planes 2/3 act nontrivially
    // (untouched generators carry the plane-1 sign by exact algebra). Review note: the sample is
    // CONTENT-SEEDED (deterministic per key — plan builds stay reproducible — but rotating across the key
    // population), so a systematic planes/conj_by_nf drift on ANY touched-inactive generator is caught
    // across a workload rather than only on the first four ascending indices.
    if (teeth) {
        std::vector<int>& ti = bs.ti;               // touched-inactive positions (pooled)
        ti.clear();
        for (int k = 0; k < T; ++k) if (!bget(active, tidx[k])) ti.push_back(k);
        uint64_t s = 0x9E3779B97F4A7C15ull;           // cheap content seed (deterministic per nf)
        for (uint64_t w : amask) s = (s ^ w) * 0x100000001B3ull;
        s = (s ^ (uint64_t)nf.cz.size()) * 0x100000001B3ull;
        const int NCHK = ti.empty() ? 0 : (int)std::min<size_t>(4, ti.size());
        for (int c2 = 0; c2 < NCHK; ++c2) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const int k = ti[(size_t)(s >> 33) % ti.size()];
            const int i = tidx[k];
            Pauli& hk = bs.hk;                      // on-demand h_i assembly (h-free derivation)
            hk.n = n;
            hk.x = G.gens[i].x;
            hk.z = G.gens[i].z;
            for (int w = 0; w < NW; ++w) hk.z[w] ^= v[k][w];
            hk.phase = phi[k];
            int rphase = 0;
            if (G.reduce_cheap(hk, rphase) != Membership::IN_GROUP)
                throw std::logic_error("build_shot_law: touched-inactive h_i did not reduce into ±G");
            if (((rphase >> 1) & 1) != (int)bget(law.tier1.sign, i))
                throw std::logic_error("build_shot_law: base/formula mismatch on inactive generator");
        }
    }
    lp_mark(g_lawprof.base);
    law.det_signs = std::move(base);

    if (core) {                                       // N6: the record-space inputs (cold; never in production)
        core->n = n;
        core->ng = ng;
        core->aidx = aidx;
        core->v.clear();
        core->phi.clear();
        for (int a2 = 0; a2 < A; ++a2) {
            core->v.push_back(v[pos_of[aidx[a2]]]);
            core->phi.push_back(phi[pos_of[aidx[a2]]]);
        }
        core->D.assign(D.rows.begin(), D.rows.begin() + (long)D.size());
        core->U.assign(kernel_dirs.rows.begin(), kernel_dirs.rows.begin() + (long)kernel_dirs.size());
    }

    if (g_lawprof.on) {
        const double dt = std::chrono::duration<double>(lawclk::now() - lp_start).count();
        if (law.kappa > 0) { ++g_lawprof.builds_k1; g_lawprof.total_k1 += dt; }
        else               { ++g_lawprof.builds_k0; g_lawprof.total_k0 += dt; }
    }
    return law;
}

ShotLaw build_shot_law(const CertifiedGroupPlanes& G, const DiagNormalForm& nf) {
    return build_shot_law_(G, nf, nullptr, /*teeth=*/true);   // the public / diagnostic entry: full checks
}

ShotLaw twirl_law_detail::build_shot_law_core(const CertifiedGroupPlanes& G, const DiagNormalForm& nf,
                                              ShotLawCore& core, bool teeth) {
    return build_shot_law_(G, nf, &core, teeth);
}

void prefix_plane1_into(const CertifiedGroupPlanes& G, const Pauli& prefix, std::vector<uint64_t>& plane) {
    plane.assign((size_t)G.words, 0);
    for (int q = 0; q < G.n_qubits; ++q) {
        if (prefix.xbit(q)) { const uint64_t* z = G.col(q, false); for (int w = 0; w < G.words; ++w) plane[w] ^= z[w]; }
        if (prefix.zbit(q)) { const uint64_t* x = G.col(q, true);  for (int w = 0; w < G.words; ++w) plane[w] ^= x[w]; }
    }
}

}  // namespace qeccore
