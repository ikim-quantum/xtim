#include "qeccore/twirl_kernel.hpp"

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

// Reduced row-echelon (fully back-substituted) of GF(2) bit-rows. Each pivot column carries a 1
// in ITS OWN basis row only — matches the reference _rref (the 2026-07-15 back-substitution fix is
// load-bearing for the nullspace / intersect quantities).
struct Rref { std::vector<BV> R; std::vector<int> piv; };
Rref rref(const std::vector<BV>& rows) {
    Rref out;
    for (const BV& r : rows) {
        BV v = r;
        for (size_t k = 0; k < out.R.size(); ++k) if (bget(v, out.piv[k])) bxor(v, out.R[k]);
        int pc = bfirst(v);
        if (pc < 0) continue;
        for (size_t k = 0; k < out.R.size(); ++k) if (bget(out.R[k], pc)) bxor(out.R[k], v);
        out.R.push_back(std::move(v));
        out.piv.push_back(pc);
    }
    return out;
}

// Pointer overload (2026-07-16 plan-build optimization): same algorithm, rows supplied by
// pointer so callers stop copying pooled scratch rows into a temporary vector first.
// Byte-identical outputs (the value overload also copies each row before reducing).
Rref rref(const std::vector<const BV*>& rows) {
    Rref out;
    for (const BV* r : rows) {
        BV v = *r;
        for (size_t k = 0; k < out.R.size(); ++k) if (bget(v, out.piv[k])) bxor(v, out.R[k]);
        int pc = bfirst(v);
        if (pc < 0) continue;
        for (size_t k = 0; k < out.R.size(); ++k) if (bget(out.R[k], pc)) bxor(out.R[k], v);
        out.R.push_back(std::move(v));
        out.piv.push_back(pc);
    }
    return out;
}

BV reduce_vec(BV v, const Rref& b) {
    for (size_t k = 0; k < b.R.size(); ++k) if (bget(v, b.piv[k])) bxor(v, b.R[k]);
    return v;
}
// Overload over raw (rows, pivots) — the hoisted per-group cache stores them without the Rref wrapper.
BV reduce_vec(BV v, const std::vector<BV>& R, const std::vector<int>& piv) {
    for (size_t k = 0; k < R.size(); ++k) if (bget(v, piv[k])) bxor(v, R[k]);
    return v;
}
bool in_span(const BV& v, const Rref& b) { return !bany(reduce_vec(v, b)); }

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

// Basis of { x ∈ GF(2)^n : row·x = 0 for every row } (reference _nullspace).
std::vector<BV> nullspace(const std::vector<BV>& rows, int n) {
    Rref b = rref(rows);
    std::vector<char> is_piv(n, 0);
    for (int pc : b.piv) is_piv[pc] = 1;
    std::vector<BV> basis;
    for (int f = 0; f < n; ++f) {
        if (is_piv[f]) continue;
        BV x(nwords(n), 0);
        bset(x, f);
        for (size_t k = 0; k < b.R.size(); ++k) if (bget(b.R[k], f)) bset(x, b.piv[k]);
        basis.push_back(std::move(x));
    }
    return basis;
}

// Basis of the LEFT null space: combos c ∈ GF(2)^m with Σ_i c_i rows[i] = 0 (reference _left_deps).
// rows are n-bit; the returned combos are m-bit (m = rows.size()). A zero row is its own dependency.
// nrows = -1: use all rows. A non-negative nrows restricts to the first nrows entries —
// pooled row storage can be larger than the live row count (grind pass, 2026-07-16).
std::vector<BV> left_deps(const std::vector<BV>& rows, int /*n*/, int nrows = -1) {
    const int m = nrows >= 0 ? nrows : (int)rows.size();
    struct Ent { BV v; BV c; int pc; };
    std::vector<Ent> red;
    std::vector<BV> deps;
    for (int i = 0; i < m; ++i) {
        BV v = rows[i];
        BV c(nwords(m), 0);
        bset(c, i);
        for (const Ent& e : red) if (bget(v, e.pc)) { bxor(v, e.v); bxor(c, e.c); }
        int pc = bfirst(v);
        if (pc < 0) deps.push_back(std::move(c));
        else        red.push_back(Ent{std::move(v), std::move(c), pc});
    }
    return deps;
}

// Particular solution x of A x = b over GF(2) (free vars = 0); reference _solve_gf2. A_rows are
// ncols-bit; b matches A_rows one bit each. If `feasible` is null the system is ASSERTED consistent
// (loud throw on an inconsistent row); if non-null the inconsistency is reported through it instead
// (used to PROBE whether a candidate preimage exists without aborting — the caller then routes to the
// alternate construction / fallback rather than crashing).
BV solve_gf2(const std::vector<BV>& A_rows, const std::vector<int>& b, int ncols,
             bool* feasible = nullptr) {
    if (feasible) *feasible = true;
    struct Ent { BV v; int rhs; int pc; };
    std::vector<Ent> red;
    for (size_t i = 0; i < A_rows.size(); ++i) {
        BV v = A_rows[i];
        int rr = b[i] & 1;
        for (const Ent& e : red) if (bget(v, e.pc)) { bxor(v, e.v); rr ^= e.rhs; }
        int pc = bfirst(v);
        if (pc < 0) {
            if (rr != 0) {
                if (feasible) { *feasible = false; return BV(nwords(ncols), 0); }
                throw std::logic_error("build_shot_law: inconsistent base-point system");
            }
            continue;
        }
        for (Ent& e : red) if (bget(e.v, pc)) { bxor(e.v, v); e.rhs ^= rr; }
        red.push_back(Ent{std::move(v), rr, pc});
    }
    BV x(nwords(ncols), 0);
    for (const Ent& e : red) if (e.rhs) bset(x, e.pc);
    return x;
}

// GF(2) dot product of two equal-width bit-vectors (parity of the AND).
inline int bvdot(const BV& a, const BV& b) {
    int acc = 0;
    const size_t w = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < w; ++i) acc += __builtin_popcountll(a[i] & b[i]);
    return acc & 1;
}

// (The old Zassenhaus `intersect` helper was retired 2026-07-15: Kz is now computed as the α-space
// kernel of the Π-row map over the V basis — same span, columnar, no 2n-wide stack — and D from the
// hoisted G_z RREF. See build_shot_law's lattice section.)

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
    Rref R = rref(Gz_gens);
    out.gz_rref = std::move(R.R);
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

namespace {
// Free helper (the planes' `rref` DATA MEMBER shadows the anon-namespace rref() inside
// member scope — the same footgun build_group_z_cache dodges).
void build_nz_cache(const CertifiedGroupPlanes& P, CertifiedGroupPlanes::NzCache& out) {
    const int NW = nwords(P.n_qubits);
    std::vector<BV> xrows((size_t)P.n_gens);
    for (int i = 0; i < P.n_gens; ++i)
        xrows[i].assign(P.gens[i].x.begin(), P.gens[i].x.begin() + NW);
    Rref R = rref(nullspace(xrows, P.n_qubits));
    out.rref = std::move(R.R);
    out.piv  = std::move(R.piv);
    out.col_to_row.assign((size_t)P.n_qubits, -1);
    for (size_t r = 0; r < out.piv.size(); ++r) out.col_to_row[out.piv[r]] = (int)r;
}
}  // namespace

const CertifiedGroupPlanes::NzCache& CertifiedGroupPlanes::nz_cache() const {
    if (nzc_ready_) return nzc_;
    build_nz_cache(*this, nzc_);
    nzc_ready_ = true;
    return nzc_;
}

// Support-directed reduction against N_z (same full-RREF exclusivity argument as
// reduce_vec_gz; XOR-only ⇒ unique result).
namespace {
void reduce_vec_nz(BV& v, const CertifiedGroupPlanes::NzCache& NZ) {
    const int NW = (int)v.size();
    for (int wq = 0; wq < NW; ++wq) {
        uint64_t word = v[wq];
        while (word) {
            const int b = __builtin_ctzll(word);
            const int q = wq * 64 + b;
            const int r = q < (int)NZ.col_to_row.size() ? NZ.col_to_row[q] : -1;
            if (r >= 0) {
                bxor(v, NZ.rref[r]);
                word = (b == 63) ? 0 : (v[wq] & (~0ULL << (b + 1)));
            } else {
                word &= word - 1;
            }
        }
    }
}
}  // namespace

// classify_observable — V3 observable channel (see the header contract and the executable
// spec scripts/twirl_obs_reference.py; identity-prefix convention, reachability MOD N_z).
ObsChannel classify_observable(const CertifiedGroupPlanes& G, const DiagNormalForm& nf,
                               const Pauli& W) {
    ObsChannel out;
    const int n = G.n_qubits;
    const int ng = G.n_gens;
    const int NW = nwords(n);
    const int GW = nwords(ng);
    const CertifiedGroupPlanes::NzCache& NZ = G.nz_cache();

    BV amask(NW, 0);
    for (int q = 0; q < n; ++q) if (q < (int)nf.a.size() && nf.a[q]) bset(amask, q);
    auto dressing = [&](const Pauli& p) {
        BV vv(NW, 0);
        for (int w = 0; w < NW; ++w) vv[w] = p.x[w] & amask[w];
        for (const auto& e : nf.cz) {
            if (p.xbit(e.second)) vv[e.first >> 6]  ^= (1ULL << (e.first & 63));
            if (p.xbit(e.first))  vv[e.second >> 6] ^= (1ULL << (e.second & 63));
        }
        return vv;
    };
    // h-free conjugate (identity prefix): i^{φ}·X^{p.x}Z^{p.z⊕v}, φ = p.phase + c(p).
    auto conj_id = [&](const Pauli& p, const BV& vv) {
        int lin = 0;
        for (int w = 0; w < NW; ++w) lin += __builtin_popcountll(p.x[w] & amask[w]);
        int quad = 0;
        for (const auto& e : nf.cz) if (p.xbit(e.first) && p.xbit(e.second)) ++quad;
        Pauli h = p;
        for (int w = 0; w < NW; ++w) h.z[w] ^= vv[w];
        h.phase = (p.phase + ((-lin - 2 * quad) % 4 + 4) % 4) & 3;
        return h;
    };

    BV vW = dressing(W);
    reduce_vec_nz(vW, NZ);                        // target mod N_z
    // eliminate against the generator dressings mod N_z, tracking ng-bit combos.
    struct Ent { BV v; BV c; int pc; };
    std::vector<Ent> red;
    for (int i = 0; i < ng; ++i) {
        BV vv = dressing(G.gens[i]);
        if (!bany(vv)) continue;                  // inactive: zero row, no pivot
        reduce_vec_nz(vv, NZ);
        BV cc(GW, 0);
        bset(cc, i);
        for (const Ent& e : red) if (bget(vv, e.pc)) { bxor(vv, e.v); bxor(cc, e.c); }
        int pc = bfirst(vv);
        if (pc >= 0) red.push_back(Ent{std::move(vv), std::move(cc), pc});
    }
    BV cW(GW, 0);
    for (const Ent& e : red) if (bget(vW, e.pc)) { bxor(vW, e.v); bxor(cW, e.c); }
    if (bany(vW)) {                               // v_W ∉ span{v_i} + N_z ⇒ conditional ≡ 0
        out.reachable = false;
        return out;
    }
    out.reachable = true;
    out.mask = cW;
    // Λ_W = reduce(W'·Π_{i∈c_W} h_i), exact phases, identity prefix.
    BV vXw = dressing(W);
    Pauli R = conj_id(W, vXw);
    for (int cw2 = 0; cw2 < GW; ++cw2) {
        uint64_t bits = cW[cw2];
        while (bits) {
            const int i = cw2 * 64 + __builtin_ctzll(bits); bits &= bits - 1;
            BV vi = dressing(G.gens[i]);
            R = Pauli::multiply(R, conj_id(G.gens[i], vi));
        }
    }
    Membership m = G.reduce(R);
    if (m.verdict == Membership::ANTI) {
        if (std::getenv("QEC_TW_OBSDBG")) {
            int cwpop = 0;
            for (uint64_t w : cW) cwpop += __builtin_popcountll(w);
            int rx = 0, rz = 0;
            for (int w = 0; w < NW; ++w) { rx += __builtin_popcountll(R.x[w]); rz += __builtin_popcountll(R.z[w]); }
            int bad = -1;
            for (int i = 0; i < ng && bad < 0; ++i)
                if (Pauli::anticommute_bit(R, G.gens[i])) bad = i;
            fprintf(stderr, "[obsdbg-classify] ANTI: |cW|=%d R(x=%d,z=%d) anti-with-gen=%d\n",
                    cwpop, rx, rz, bad);
        }
        out.guard = true;
        return out;
    }
    out.lam = m.rep;                              // LOGICAL rep (or ±I if W·Πh ∈ ±G)
    return out;
}

// Ephemeral stage profiling (QEC_TW_PROF=1): per-build stage accumulators, dumped at exit.
// Timing only — no behavioral change; guarded so the default path costs one branch.
namespace {
struct LawProf {
    bool on = std::getenv("QEC_TW_PROF") != nullptr;
    double tier1 = 0, cone = 0, lattice = 0, coins = 0, cdet = 0, kernel = 0, base = 0;
    long builds = 0;
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
    std::vector<BV> bred;              // reduced dressing rows (cdet / lattice-D stages)
    std::vector<BV> pi;                // Π rows (coin-mask source rows)
    std::vector<int> tidx;
    std::vector<int> pos_of;
    std::vector<const BV*> row_ptrs;   // pointer rows for the rref overload
    std::vector<BV> vcol;              // per-qubit generator-bitmask dressing columns
    std::vector<uint8_t> vcol_used;    // dirty flags for vcol (cleared via ulist)
    std::vector<int> ulist;            // dirty qubit list
};
thread_local BuildScratch g_bs;
}  // namespace

ShotLaw build_shot_law(const CertifiedGroupPlanes& G, const DiagNormalForm& nf) {
    ShotLaw law;
    const int n = G.n_qubits;
    const int ng = G.n_gens;
    const int GW = nwords(ng);

    if (!nf.diagonal_class) { law.fallback = true; return law; }

    auto lp = g_lawprof.on ? lawclk::now() : lawclk::time_point();
    auto lp_mark = [&](double& acc) {
        if (!g_lawprof.on) return;
        auto now = lawclk::now();
        acc += std::chrono::duration<double>(now - lp).count();
        lp = now;
    };
    if (g_lawprof.on) ++g_lawprof.builds;

    law.tier1 = tier1_signs(G, nf);
    lp_mark(g_lawprof.tier1);

    // ── TOUCHED cone (cold-build restructure, 2026-07-15): only generators whose X-support meets
    // supp(M) = supp(a)∪cz-endpoints can carry a nonzero dressing v_i or a plane-2/3 sign; every
    // untouched generator has v_i = 0 and h_i = ±g_i with the plane-1 sign EXACTLY (conjugation by P
    // alone — no assumption, the C-layer commutes term-by-term). The old ng-scale loop materialised
    // v/h/xrows/zrows for all 297 generators per cold key; T = |touched| is lightcone-sized. ──
    const int NW = nwords(n);
    BV amask(NW, 0);                                  // amask[q]=1 ⇔ a_q=1 (word-level dressing S-part)
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
    auto& vcol = g_bs.vcol;
    auto& used = g_bs.vcol_used;
    auto& ulist = g_bs.ulist;
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
    BV touched(GW, 0);
    for (int q : ulist) {
        const uint64_t* xc = G.col(q, true);
        for (int t = 0; t < GW; ++t) touched[t] |= xc[t];
    }
    std::vector<int>& tidx = g_bs.tidx;               // touched generator indices, ascending
    tidx.clear();
    for (int w = 0; w < GW; ++w) { uint64_t bits = touched[w];
        while (bits) { tidx.push_back(w * 64 + __builtin_ctzll(bits)); bits &= bits - 1; } }
    const int T = (int)tidx.size();

    std::vector<BV>& v = g_bs.v;                      // dressings, indexed by touched position
    std::vector<uint8_t>& phi = g_bs.phi;             // h_i phase bits (h-free derivation)
    if ((int)v.size() < T) v.resize(T);
    phi.assign((size_t)T, 0);
    std::vector<int>& pos_of = g_bs.pos_of;           // generator index -> touched position
    pos_of.assign(ng, -1);
    BV active(GW, 0);
    for (int k = 0; k < T; ++k) pos_of[tidx[k]] = k;
    {
        for (int k = 0; k < T; ++k) v[k].assign(NW, 0);
        for (int q : ulist) {
            const BV& col = vcol[q];
            for (int w = 0; w < GW; ++w) {
                uint64_t word = col[w];
                while (word) {
                    const int i = w * 64 + __builtin_ctzll(word); word &= word - 1;
                    v[pos_of[i]][q >> 6] |= (1ULL << (q & 63));
                }
            }
            used[q] = 0;                                     // clear dirty flag for the next build
        }
        // quad-parity plane: bit i = (#{(q,q')∈cz : x_i[q]∧x_i[q']}) mod 2 — one AND-XOR per pair.
        BV qplane(GW, 0);
        for (const auto& e : nf.cz) {
            const uint64_t* xa = G.col(e.first, true);
            const uint64_t* xb = G.col(e.second, true);
            for (int t = 0; t < GW; ++t) qplane[t] ^= xa[t] & xb[t];
        }
        // lin mod 4 planes (carry-save over supp(a), same construction as tier1 plane 2):
        // ones = bit0, twos = bit1 of Σ a_q x_{i,q}.
        BV ones(GW, 0), twos(GW, 0);
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
        const std::vector<uint64_t> p1 =
            haspfx ? prefix_plane1(G, nf.prefix) : std::vector<uint64_t>();
        for (int k = 0; k < T; ++k) {
            const int i = tidx[k];
            const int lin4 = (int)bget(ones, i) + 2 * (int)bget(twos, i);
            const int ci = ((-lin4 - 2 * (int)bget(qplane, i)) % 4 + 4) % 4;
            const int a1 = haspfx ? (int)bget(p1, i) : 0;
            phi[k] = (uint8_t)((G.gens[i].phase + 2 * a1 + ci) & 3);
            if (bany(v[k])) bset(active, i);
        }
    }
    // Cross-check vs Tier-1 keeps its full teeth: dressings are materialised independently (row-wise)
    // for every touched generator, and untouched generators are structurally inactive — so equality
    // below also certifies tier1.active is zero outside the touched cone.
    for (int w = 0; w < GW; ++w)
        if (active[w] != law.tier1.active[w]) throw std::logic_error("build_shot_law: active mask disagrees with tier1");

    if (g_lawprof.on) {
        g_lawprof.sumT += T;
        g_lawprof.sumcz += (long)nf.cz.size();
        long na = 0;
        for (int q = 0; q < n; ++q) if (nf.a[q]) ++na;
        g_lawprof.suma += na;
    }
    lp_mark(g_lawprof.cone);
    // ── lattice classification (spec §5 step 4) ──
    std::vector<const BV*>& Vrows = g_bs.row_ptrs;
    Vrows.clear();
    for (int k = 0; k < T; ++k) if (bget(active, tidx[k])) Vrows.push_back(&v[k]);
    Rref V_basis = rref(Vrows);

    // G_z (pure-Z group content) is a GROUP-ONLY quantity — hoisted to a lazy per-group cache
    // (byte-identical computation, no longer rebuilt per cold key).
    const CertifiedGroupPlanes::GroupZCache& GZ = G.group_z_cache();

    // Kz = { u ∈ V : u·x_j = 0 ∀j } — the old 2n-wide Zassenhaus against an N_z basis is replaced by
    // the kernel of α ↦ Π-row(Σ α_k Vb_k) in the |V|-dim α-space, with Π-rows built columnar from the
    // group's X-plane. Same span always; the basis CHOICE can differ from the old construction only
    // when κ>0 ∧ |D|>0 (downstream: kernel reps canonicalise mod the group, masks/base re-solve per
    // direction — validated by the κ=1 gates) and κ=0 laws are byte-identical (Kz feeds only
    // span/dim quantities there).
    std::vector<BV> Kz;
    {
        std::vector<BV> prow(V_basis.R.size());
        for (size_t k = 0; k < V_basis.R.size(); ++k) {
            BV row(GW, 0);
            const BV& u = V_basis.R[k];
            for (int w = 0; w < NW; ++w) {
                uint64_t word = u[w];
                while (word) { const int q = w * 64 + __builtin_ctzll(word); word &= word - 1;
                    const uint64_t* xc = G.col(q, true);
                    for (int t = 0; t < GW; ++t) row[t] ^= xc[t]; }
            }
            prow[k] = std::move(row);
        }
        for (const BV& c : left_deps(prow, ng)) {
            BV u(NW, 0);
            for (size_t k = 0; k < V_basis.R.size(); ++k) if (bget(c, k)) bxor(u, V_basis.R[k]);
            Kz.push_back(std::move(u));
        }
    }
    // D = V ∩ G_z (= Kz ∩ G_z since G_z ⊆ N_z); only span(D)/|D| feed downstream.
    std::vector<BV> D;
    {
        std::vector<BV> Dres(Kz.size());
        for (size_t k = 0; k < Kz.size(); ++k) { Dres[k] = Kz[k]; reduce_vec_gz_inplace(Dres[k], GZ); }
        for (const BV& c : left_deps(Dres, n)) {
            BV d(nwords(n), 0);
            for (size_t k = 0; k < Kz.size(); ++k) if (bget(c, k)) bxor(d, Kz[k]);
            D.push_back(std::move(d));
        }
    }
    lp_mark(g_lawprof.lattice);
    law.kappa = (int)Kz.size() - (int)D.size();
    law.r = (int)V_basis.R.size() - (int)Kz.size();

    // ── coin masks: rowspace(Π), Π_ij = v_i·x_j; rank(Π) MUST equal r (Phase-A bug #3) ──
    // Columnar (spec §6): Π row i (bit j = v_i·x_j) = XOR over q∈supp(v_i) of xcol(q) (bit j = x_j[q]).
    // Replaces the O(ng²·n/64) row-wise popcount with O(Σ|supp v_i|·GW) column XORs into the group's
    // precomputed X-plane. Same rows, same order (only nonzero rows pushed) ⇒ byte-identical coin.R.
    std::vector<BV>& Pi = g_bs.pi;                    // pooled Π-row storage (2026-07-16)
    if (Pi.size() < (size_t)T) Pi.resize(T);          // pre-size: Pi_ptrs points into Pi, and a
                                                      // mid-loop grow would dangle those pointers
    std::vector<const BV*> Pi_ptrs;
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
    Rref coin = rref(Pi_ptrs);
    if ((int)coin.R.size() != law.r) throw std::logic_error("build_shot_law: coin rank != r (rank(Pi) assert)");
    law.coin_masks = coin.R;

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
    std::vector<int> aidx;                            // active generator indices, ascending
    for (int k = 0; k < T; ++k) if (bget(active, tidx[k])) aidx.push_back(tidx[k]);
    const int A = (int)aidx.size();
    std::vector<BV>& BredA = g_bs.bred;               // pooled reduced-dressing rows
    if ((int)BredA.size() < A) BredA.resize(A);
    for (int a2 = 0; a2 < A; ++a2) {
        BredA[a2].assign(v[pos_of[aidx[a2]]].begin(), v[pos_of[aidx[a2]]].end());
        reduce_vec_gz_inplace(BredA[a2], GZ);
    }
    std::vector<BV> Cdet;
    for (const BV& c : left_deps(BredA, n, A)) {         // combos in A-dim → scatter to ng-bit
        BV cs(GW, 0);
        for (int a2 = 0; a2 < A; ++a2) if (bget(c, a2)) bset(cs, aidx[a2]);
        Cdet.push_back(std::move(cs));
    }

    if (g_lawprof.on) { g_lawprof.sumA += (long)aidx.size(); g_lawprof.sumCdet += (long)Cdet.size(); }
    lp_mark(g_lawprof.cdet);
    // ── kernel logical reps: extend a basis of D to Kz; reduce each new Z^u to a logical ──
    std::vector<BV> kernel_dirs;
    {
        std::vector<BV> acc = D;
        Rref det = rref(acc);
        for (const BV& u : Kz) {
            if (!in_span(u, det)) {
                kernel_dirs.push_back(u);
                acc.push_back(u);
                det = rref(acc);
            }
        }
    }
    for (const BV& u : kernel_dirs) {
        Pauli zu(n);
        for (int q = 0; q < n; ++q) if (bget(u, q)) zu.setz(q);
        Membership m = G.reduce(zu);
        law.kernel_logicals.push_back(m.rep);       // LOGICAL: reduced normaliser rep, exact phase
    }

    // ── kernel σ-fold masks + operator-identity preimages (Task-10 σ-assembly; §5 step 5).
    //
    // Two DISTINCT vectors per kernel direction u = v*_j (do not conflate them — the 2026-07-15 κ=1
    // geometry fix hinges on the split):
    //   • the PREIMAGE c_j solving Σ_i c_i v_i = u (a generator subset; always feasible since u ∈ V).
    //     Its σ-parity c_j·σ tracks the j-th logical outcome (Π_{i∈c_j} h_i = s_j·(group)·L_j), so it
    //     drives the operator-identity sign s_j and the base bit — NOT the σ flip itself.
    //   • the FOLD mask δ_j = the σ difference between the two logical cosets. XORing it must PRESERVE
    //     every deterministic parity, so δ_j ∈ C_det^⊥ (⊥ every C_det combo), and it must be
    //     independent of the coins. c_j itself is only a valid fold mask when it happens to be ⊥ C_det
    //     AND outside the coin span (the generic / fixture geometry) — then δ_j = c_j, byte-identical
    //     to the pre-fix build. When c_j is ⊥-C_det-infeasible or lands in the coin span (degenerate
    //     κ=1 geometries), δ_j is instead drawn from C_det^⊥ \ span(coins) with c_j·δ_j = 1. Both are
    //     exact — validated end-to-end against the reference sample_shot σ law on the κ=1 corpus.
    //
    // Legacy fast path (kept BYTE-IDENTICAL for every non-degenerate law incl. the κ=1 fixtures): first
    // PROBE the old joint system [qubit rows ‖ C_det rows]·c = [u ‖ 0]; if it is feasible and its
    // solution is outside the coin span, that single c serves as BOTH preimage and fold mask exactly
    // as before. Only when it is inconsistent or coin-dependent do we take the split construction.
    // The kernel σ-fold machinery (qrows selector, C_det^⊥ basis, per-dir GF(2) solves) is needed
    // ONLY when there is a kernel direction — i.e. κ>0. In-distribution κ=0 dominates (M2 census),
    // so building qrows / nullspace(Cdet) unconditionally is wasted work on the hot cold-build path;
    // skip it entirely when kernel_dirs is empty (kernel_pre / kernel_masks stay empty, exactly as
    // the loop would leave them).
    std::vector<BV> kernel_pre;                      // preimages c_j (operator-identity / sign)
    if (!kernel_dirs.empty()) {
    std::vector<BV> qrows;                           // qubit rows: bit i = v_i[q]  (Σ c_i v_i selector)
    qrows.reserve((size_t)n);
    for (int q = 0; q < n; ++q) {
        BV row(GW, 0);
        for (int a2 = 0; a2 < A; ++a2)               // v_i = 0 off the active set
            if (bget(v[pos_of[aidx[a2]]], q)) bset(row, aidx[a2]);
        qrows.push_back(std::move(row));
    }
    // The fold-mask machinery needs the FULL C_det space (a δ must preserve the inactive pins too):
    // active combos + the inactive singletons. Basis order is immaterial — Wbasis comes from the
    // (unique) RREF of the rowspace, and the probe solves are rowspace-determined.
    std::vector<BV> Cdet_full = Cdet;
    for (int i = 0; i < ng; ++i)
        if (!bget(active, i)) { BV e(GW, 0); bset(e, i); Cdet_full.push_back(std::move(e)); }
    std::vector<BV> Wbasis = nullspace(Cdet_full, ng);   // C_det^⊥ in σ-space (dim r+κ)
    kernel_pre.reserve(kernel_dirs.size());
    for (const BV& u : kernel_dirs) {
        // Legacy probe: [qubit ‖ C_det] c = [u ‖ 0]  (a ⊥-C_det preimage).
        std::vector<BV> mA = qrows;
        for (const BV& d : Cdet_full) mA.push_back(d);
        std::vector<int> mb(mA.size(), 0);
        for (int q = 0; q < n; ++q) mb[q] = bget(u, q) ? 1 : 0;   // C_det-row rhs stay 0
        bool legacy_ok = false;
        BV c_legacy = solve_gf2(mA, mb, ng, &legacy_ok);
        BV cj, delta;
        if (legacy_ok && !in_span(c_legacy, coin)) {
            cj = c_legacy;                           // non-degenerate: preimage == fold mask
            delta = c_legacy;
        } else {
            // Split construction. Plain preimage (qubit rows only) — feasible because u ∈ V.
            std::vector<int> pb(n, 0);
            for (int q = 0; q < n; ++q) pb[q] = bget(u, q) ? 1 : 0;
            bool pre_ok = false;
            cj = solve_gf2(qrows, pb, ng, &pre_ok);
            if (!pre_ok) throw std::logic_error("build_shot_law: kernel preimage system inconsistent (u not in V)");
            // The logical outcome must be independent of the fair coins (else the factored form
            // cannot separate them) — a representability condition ⇒ loud fallback, never wrong σ.
            for (const BV& cm : law.coin_masks) if (bvdot(cj, cm)) {
                std::fprintf(stderr, "[build_shot_law] kernel preimage not ⊥ coins -> fallback\n");
                law.fallback = true; break;
            }
            if (law.fallback) return law;
            // Fold direction δ ∈ C_det^⊥ \ span(coins) with c_j·δ = 1 (crosses the logical hyperplane).
            for (const BV& w : Wbasis) {
                BV wr = reduce_vec(w, coin);         // reduce mod the coin span
                if (bany(wr) && bvdot(cj, wr)) { delta = std::move(wr); break; }
            }
            if (delta.empty()) {
                std::fprintf(stderr, "[build_shot_law] no kernel fold direction in C_det^perp \\ coins -> fallback\n");
                law.fallback = true; return law;
            }
        }
        kernel_pre.push_back(std::move(cj));
        law.kernel_masks.push_back(std::move(delta));
    }
    }  // if (!kernel_dirs.empty())

    // Joint independence: {coin_masks ∪ kernel_masks} must span the full C_det-homogeneous space
    // (rank r+κ) — coins and the κ fold directions are distinct outcome types and no σ is doubly
    // reachable. Guaranteed by construction for κ≤1 (δ ∈ C_det^⊥ \ coins); stays a loud guard.
    {
        std::vector<BV> allm = law.coin_masks;
        for (const BV& k : law.kernel_masks) allm.push_back(k);
        if ((int)rref(allm).R.size() != law.r + law.kappa) {
            // κ≥2 degenerate fold geometry: the legacy δ_j = c_j picks can be linearly
            // DEPENDENT across kernel dirs (e.g. two dirs sharing one forced preimage) —
            // a representability boundary of the κ≥2 fold, not a bug. Route it to the
            // loud fallback like the pairing check below (V3-T3: the per-shot exact
            // engine computes those shots; pre-R1 this was an unreachable terminate).
            // κ≤1 keeps the hard throw: independence is guaranteed by construction
            // there (δ ∈ C_det^⊥ \ span(coins)), so a trip IS an engine bug.
            if (law.kappa >= 2) {
                std::fprintf(stderr, "[build_shot_law] coin+kernel masks rank-deficient "
                                     "(kappa=%d) -> fallback\n", law.kappa);
                law.fallback = true;
                return law;
            }
            throw std::logic_error("build_shot_law: coin_masks ∪ kernel_masks not rank r+kappa");
        }
    }

    lp_mark(g_lawprof.kernel);
    // ── valid base point (spec F7 / Phase-A bug #2): the Tier-1 formula is a per-generator sign,
    // not a joint coset rep. For c ∈ C_det, ∏ h_i ∈ ±G with a definite sign b_c (exact Pauli
    // reduction). Solve c·σ = b_c for one σ; assert σ ≡ formula on inactive.
    // MEMO NOTE: b_c = b0_c ⊕ ⟨P, g_c⟩ splits into an (a,cz)-only base0 plus the prefix plane1;
    // det_signs = base0 ⊕ plane1 (see prefix_plane1 / TwirlPlanCache). We compute the full b_c here
    // (faithful, byte-identical to the reference) — the split is consumed by the Task-10 memo.
    std::vector<int> bvals(Cdet.size(), 0);
    Pauli& Sc = g_bs.sc;                              // pooled product scratch
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
    BV base = solve_gf2(Cdet, bvals, ng);
    for (int w = 0; w < GW; ++w) base[w] |= law.tier1.sign[w] & ~active[w];
    // Teeth (bounded): re-verify sign(h_i) == formula by exact reduction on up to 4 touched-INACTIVE
    // generators — the population where planes 2/3 act nontrivially (untouched generators carry the
    // plane-1 sign by exact algebra). Review note: the sample is CONTENT-SEEDED (deterministic per
    // key — plan builds stay reproducible — but rotating across the key population), so a systematic
    // planes/conj_by_nf drift on ANY touched-inactive generator is caught across a workload rather
    // than only on the first four ascending indices.
    {
        std::vector<int>& ti = g_bs.ti;               // touched-inactive positions (pooled)
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
            Pauli& hk = g_bs.hk;                      // on-demand h_i assembly (h-free derivation)
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

    // ── kernel base-sign bits (Task-10 σ-assembly): s_j from the operator identity
    // Π_{i∈c_j} h_i = s_j·(group)·L_j on the PREIMAGE c_j (kernel_pre[j]), then
    // kernel_base[j] = s_j ⊕ (c_j·det_signs). The assembled σ = det ⊕ coins ⊕ (fold_j)·δ_j has
    // c_j·σ = c_j·det ⊕ fold_j (using c_j·δ_j = 1 and c_j ⊥ coins), so twirl_collapse folds δ_j iff
    // (outcome ⊕ kernel_base[j]). Cross-pairing requirement is c_j·δ_k = δ_jk (a fold of δ_k must
    // disturb no other logical read AND δ_j must flip its own): κ=1 satisfies it by construction,
    // κ≥2 falls back loudly (never ships a wrong σ). ──
    const int K = (int)law.kernel_masks.size();
    for (int j = 0; j < K && !law.fallback; ++j) {
        for (int k = 0; k < K; ++k) {
            const int want = (j == k) ? 1 : 0;
            if (bvdot(kernel_pre[j], law.kernel_masks[k]) != want) {
                std::fprintf(stderr, "[build_shot_law] kernel preimage/fold pairing != delta_jk (j=%d k=%d) -> fallback\n", j, k);
                law.fallback = true;
                break;
            }
        }
    }
    // Diagonal V1: every kernel direction is FOLDABLE (its dressing u ∈ V = span{v_i} by
    // construction, so the preimage system Σ c_i v_i = u is always feasible). Set the flags true so
    // the V2 collapse fold guard is a no-op here (byte-identical behaviour).
    law.kernel_foldable.assign(K, 1);
    if (!law.fallback) {
        law.kernel_base.assign(K, 0);
        for (int j = 0; j < K; ++j) {
            Pauli Hj(n);
            for (int i = 0; i < ng; ++i)
                if (bget(kernel_pre[j], i)) {
                    Pauli& hk = g_bs.hk;              // on-demand h_i assembly (h-free derivation)
                    hk.n = n;
                    hk.x = G.gens[i].x;
                    hk.z = G.gens[i].z;
                    for (int w = 0; w < NW; ++w) hk.z[w] ^= v[pos_of[i]][w];
                    hk.phase = phi[pos_of[i]];
                    Hj = Pauli::multiply(Hj, hk);
                }
            Membership m = G.reduce(Hj);
            const Pauli& L = law.kernel_logicals[j];
            if (m.rep.x != L.x || m.rep.z != L.z)
                throw std::logic_error("build_shot_law: kernel base reduction support != kernel logical");
            const int s_j = ((m.rep.phase ^ L.phase) >> 1) & 1;   // ± sign vs the kernel logical rep
            law.kernel_base[j] = (uint8_t)(s_j ^ bvdot(kernel_pre[j], law.det_signs));
        }
    }

    return law;
}

// prefix_plane1 — n_gens-bit ⟨P,g_i⟩ plane, columnar in O(|P|·n/64): XOR zcol(q) over supp_X(P),
// XOR xcol(q) over supp_Z(P). (⟨P,g⟩ = Σ_q P.x[q]·g.z[q] ⊕ P.z[q]·g.x[q]; zcol(q) packs g_i.z[q]
// over i, xcol(q) packs g_i.x[q].)
std::vector<uint64_t> prefix_plane1(const CertifiedGroupPlanes& G, const Pauli& prefix) {
    std::vector<uint64_t> plane(G.words, 0);
    for (int q = 0; q < G.n_qubits; ++q) {
        if (prefix.xbit(q)) { const uint64_t* z = G.col(q, false); for (int w = 0; w < G.words; ++w) plane[w] ^= z[w]; }
        if (prefix.zbit(q)) { const uint64_t* x = G.col(q, true);  for (int w = 0; w < G.words; ++w) plane[w] ^= x[w]; }
    }
    return plane;
}

}  // namespace qeccore
