#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/pauli_kernels.hpp"   // shared single_pauli/pmul_into/conj_cx_inplace/dual_image_rows_scan/partner_hash_chi_min
#include "qeccore/stab_generators.hpp"
#include "qeccore/stab_disentangle.hpp"
#include "qeccore/gf2_gauss.hpp"
#include "qeccore/bare_state_errors.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <string>

namespace qeccore {

// single_pauli now lives in qeccore/pauli_kernels.hpp (shared with framed_superposition / factored_stab).

// χ-independent Clifford gates (theory §8). DEFERRED: RECORD the gate twice — for the shared frame (U.left_*,
// which updates g_a = U Z_a U† and d_a = U X_a U†, replayed by flush_gates() via left_batch the first time U is
// read) and for the anchor |φ0⟩ (replayed by anchor_state() / flush_anchor_ before any read of it, so every
// reader sees the same gate sequence on the same state as an eager update). eps/free/branches are untouched.
void CanonicalStabSum::apply_h(int q)            { pending_gates.push_back({0, q, -1}); anchor_pending_.push_back({0, q, -1}); }
void CanonicalStabSum::apply_s(int q)            { pending_gates.push_back({1, q, -1}); anchor_pending_.push_back({1, q, -1}); }
void CanonicalStabSum::apply_sdg(int q)          { pending_gates.push_back({2, q, -1}); anchor_pending_.push_back({2, q, -1}); }
void CanonicalStabSum::apply_x(int q)            { pending_gates.push_back({3, q, -1}); anchor_pending_.push_back({3, q, -1}); }
void CanonicalStabSum::apply_y(int q)            { pending_gates.push_back({4, q, -1}); anchor_pending_.push_back({4, q, -1}); }
void CanonicalStabSum::apply_z(int q)            { pending_gates.push_back({5, q, -1}); anchor_pending_.push_back({5, q, -1}); }
void CanonicalStabSum::apply_cx(int c, int t)    { pending_gates.push_back({6, c, t});  anchor_pending_.push_back({6, c, t}); }
void CanonicalStabSum::apply_cz(int c, int t)    { pending_gates.push_back({7, c, t});  anchor_pending_.push_back({7, c, t}); }

// Replay the pending Clifford gates onto the frame U (in apply order), then clear. Mirrors the lazy
// `ensure_dual` pattern: const + mutable so const frame-readers materialise the tableau on demand.
void CanonicalStabSum::flush_gates() const {
    if (pending_gates.empty()) return;
    const_cast<CliffordTableau&>(U).left_batch(pending_gates);
    pending_gates.clear();
}

// Replay the gates applied since the anchor was last read, in apply order (the former eager update's sequence).
void CanonicalStabSum::flush_anchor_() const {
    if (anchor_pending_.empty()) return;
    if (!anchor_)
        throw std::logic_error("CanonicalStabSum: Clifford gates pending on a sum without an anchor (engine bug)");
    AffineState& A = *anchor_;
    for (const PendingGate& g : anchor_pending_) {
        switch (g.kind) {
            case 0: A.apply_h(g.a);        break;
            case 1: A.apply_s(g.a);        break;
            case 2: A.apply_sdg(g.a);      break;
            case 3: A.apply_x(g.a);        break;
            case 4: A.apply_y(g.a);        break;
            case 5: A.apply_z(g.a);        break;
            case 6: A.apply_cx(g.a, g.b);  break;
            default: A.apply_cz(g.a, g.b); break;
        }
    }
    anchor_pending_.clear();
}

void CanonicalStabSum::set_anchor(std::unique_ptr<AffineState> a) {
    if (!anchor_pending_.empty())
        throw std::logic_error("CanonicalStabSum: a new anchor while Clifford gates are pending on the old one (engine bug)");
    anchor_ = std::move(a);
}

// ── Canonicalisation (theory §6 / Lemma 2) ───────────────────────────────────────────────────
// Re-select `free` as a minimal independent basis of the surviving sign-variation. Form the χ×r
// matrix M with rows σ_i ⊕ σ_0 over the current `free`; GF(2) column-echelon; `free` ← pivot
// columns; re-coordinatise every branch's σ onto the pivot columns. Generators whose column is
// not a pivot (constant across survivors, or an F-combination of others — incl. coincidental
// fixing) leave `free` and become fixed. Distinctness must already hold (caller's responsibility).
void CanonicalStabSum::canonicalise() {
    flush_gates();                              // reads U.Xrow below — materialise the deferred frame
    int r = (int)free.size();
    int x = chi();
    if (r == 0 || x == 0) return;

    // Variation columns as branch-bitsets: col[d] bit i = σ_i[d] ⊕ σ_0[d].
    const int xw = (x + 63) / 64;
    // χ-scaled scratch: r difference columns + ≤ r reduced columns + one residual, xw words each.
    struct SlotScratch {
        std::vector<std::vector<uint64_t>> col;
        std::vector<std::vector<uint64_t>> red;
        std::vector<uint64_t> resid;
        ScratchMark mark{0, 0};
    };
    static thread_local SlotScratch sc;   // one TLS object: buffers + scratch charge
    static constexpr ScratchReleaser slot_rel = +[] {
        std::vector<std::vector<uint64_t>>().swap(sc.col);
        std::vector<std::vector<uint64_t>>().swap(sc.red);
        std::vector<uint64_t>().swap(sc.resid);
    };
    std::vector<std::vector<uint64_t>>& col = sc.col;
    std::vector<std::vector<uint64_t>>& red = sc.red;
    std::vector<uint64_t>& resid = sc.resid;
    scratch_need(sc.mark, slot_rel, (int64_t)(2 * r + 1) * (heap_chunk_bytes((int64_t)xw * 8) + 24),
              "bare-state canonicalise scratch");
    col.assign(r, std::vector<uint64_t>(xw, 0));
    for (int i = 1; i < x; ++i)
        for (int d = 0; d < r; ++d)
            if (branches[i].sigma[d] ^ branches[0].sigma[d]) col[d][i >> 6] |= 1ull << (i & 63);

    // Greedy column basis WITH expansion tracking (rebuild_from_rays' bookkeeping). A column that
    // is VARYING BUT DEPENDENT (nonzero, in the span of earlier pivots) arises whenever a Case-A
    // collapse drops branches at χ > 2 — the old code mis-folded it as a constant (the anchor got
    // branches[0]'s value while other survivors disagreed: wrong post-state, caught by the dense
    // oracle on cultivation-d3-faithful). The fix is the same frame rotation rebuild_from_rays
    // uses to re-base dependent generators onto the pivot set:
    //     g_c ← g_c·g_p (Zrow[f_c] *= Zrow[f_p]),  X_p ← X_p·X_c (the ORIGINAL X_c),
    //     σ_i[c] ^= σ_i[p],  eps[f_c] ^= eps[f_p]      for each pivot p in the dependency,
    // after which column c's variation is zero and the constant fold below is exact. (Validity:
    // X_c anticommutes with the new g_c and commutes with everything else it must; the rotated
    // pair keeps the symplectic frame relations — same algebra as rebuild_from_rays, which is
    // oracle-pinned.) When no dependent-varying columns exist, no rotation happens and the pivot
    // set equals the old row-echelon's (greedy column independence is order-identical).
    static thread_local std::vector<int> pivot_cols; pivot_cols.clear();
    red.clear();
    static thread_local std::vector<std::vector<uint8_t>> comp; comp.clear();
    static thread_local std::vector<uint8_t> expand;
    bool rows_edited = false;
    auto lead_bit = [&](const std::vector<uint64_t>& v) {
        for (int w = 0; w < xw; ++w) if (v[w]) return (w << 6) + __builtin_ctzll(v[w]);
        return -1;
    };
    for (int c = 0; c < r; ++c) {
        bool varying = false;
        for (int w = 0; w < xw; ++w) if (col[c][w]) { varying = true; break; }
        if (!varying) continue;                        // constant column: folded below
        resid = col[c];
        expand.assign(pivot_cols.size(), 0);
        for (size_t k = 0; k < pivot_cols.size(); ++k) {
            const int lb = lead_bit(red[k]);
            if (lb >= 0 && ((resid[lb >> 6] >> (lb & 63)) & 1)) {
                for (int w = 0; w < xw; ++w) resid[w] ^= red[k][w];
                for (size_t t = 0; t < comp[k].size(); ++t) expand[t] = (uint8_t)(expand[t] ^ comp[k][t]);
            }
        }
        bool zero = true;
        for (int w = 0; w < xw; ++w) if (resid[w]) { zero = false; break; }
        if (!zero) {                                   // new pivot
            std::vector<uint8_t> cc(pivot_cols.size() + 1, 0);
            for (size_t t = 0; t < expand.size(); ++t) cc[t] = expand[t];
            cc[pivot_cols.size()] = 1;
            for (auto& v : comp) v.push_back(0);
            pivot_cols.push_back(c);
            red.push_back(resid);
            comp.push_back(std::move(cc));
        } else {                                       // varying-but-dependent: re-base
            rows_edited = true;
            const Pauli Xc = U.Xrow[free[c]];          // snapshot before any row edit
            for (size_t k = 0; k < expand.size(); ++k) {
                if (!expand[k]) continue;
                const int p = pivot_cols[k];
                U.Zrow[free[c]] = Pauli::multiply(U.Zrow[free[c]], U.Zrow[free[p]]);
                U.Xrow[free[p]] = Pauli::multiply(U.Xrow[free[p]], Xc);
                eps[free[c]] = (uint8_t)(eps[free[c]] ^ eps[free[p]]);
                for (auto& br : branches) br.sigma[c] = (uint8_t)(br.sigma[c] ^ br.sigma[p]);
            }
        }
    }
    if (rows_edited) U.invalidate_dual();

    int rp = (int)pivot_cols.size();
    static thread_local std::vector<uint8_t> is_pivot; is_pivot.assign(r, 0);
    for (int c2 : pivot_cols) is_pivot[c2] = 1;

    // Every non-pivot column is now CONSTANT across survivors (by construction above). Drop each;
    // if its shared value is 1, fold the destabiliser into the anchor.
    for (int col2 = 0; col2 < r; ++col2) {
        if (is_pivot[col2]) continue;
        if (branches[0].sigma[col2]) { U.Xrow[free[col2]].apply_to(anchor_state()); eps[free[col2]] ^= 1; }
    }

    // New free generator indices: compact `free` onto the pivot columns IN PLACE (pivot_cols strictly
    // increasing ⇒ pivot_cols[d] ≥ d, so the read is never an already-overwritten slot).
    for (int d = 0; d < rp; ++d) free[d] = free[pivot_cols[d]];
    free.resize(rp);

    // Re-coordinatise each branch's σ onto the pivot columns, IN PLACE (same pivot_cols[d] ≥ d
    // argument), then shrink — no per-branch reallocation.
    for (auto& br : branches) {
        for (int d = 0; d < rp; ++d) br.sigma[d] = br.sigma[pivot_cols[d]];
        br.sigma.resize(rp);
    }
}

// ── (overlap-based) Rebuild (U, eps, anchor, free, branches) so Σ_g coeffs[g]·|rays[g]⟩ exactly ──
// (debug oracle for the closed-form measurement path + the one-time from_rays construction path;
//  always compiled — promoted out of #ifndef NDEBUG for build_bare_state.)
void CanonicalStabSum::rebuild_from_rays(std::vector<std::unique_ptr<AffineState>> rays,
                                         std::vector<std::complex<double>> coeffs) {
    const int N = n();
    const int xs = (int)rays.size();
    // anchor = rays[0]; frame from its stabilizer generators (g_a = Zrow[a]).
    std::vector<Pauli> gens = stabilizer_generators(*rays[0]);
    U = CliffordTableau::from_generators(gens);
    set_anchor(std::unique_ptr<AffineState>(static_cast<AffineState*>(rays[0]->clone().release())));

    // fs[g][a] = sign bit of the current frame generator U.Zrow[a] on ray_g.
    //
    // NEW (O(χ·n²)): disentangle each ray_g ONCE by the current frame U, reading ALL n syndrome bits
    // from the resulting computational-basis offset — replacing the old O(χ·n) × O(n²) amplitude reads.
    // The disentangling circuit is derived from U.Zrow (the CURRENT frame generators), so it is correct
    // for BOTH calls: call #1 uses the initial generators, call #2 uses the re-based U.Zrow. The key
    // invariant: U†|ray_g⟩ = |s_g⟩, a computational-basis state whose bit string IS the syndrome.
    // After disentangle, s.z_expectation(a) = (-1)^{s_g[a]}, so fs[g][a] = (z_expectation(a)==-1)?1:0.
    auto compute_fsynd = [&](std::vector<std::vector<uint8_t>>& fs) {
        std::vector<const AffineState*> rp((size_t)xs);
        for (int g = 0; g < xs; ++g) rp[(size_t)g] = rays[(size_t)g].get();
        ray_syndromes(rp, fs);
    };
    std::vector<std::vector<uint8_t>> fsynd; compute_fsynd(fsynd);

    // Re-base DEPENDENT distinguishing generators to fixed ones, so the varying syndrome lives on an
    // INDEPENDENT generator set (Minimality precondition for canonicalise). Greedy column-echelon of
    // the syndrome-difference matrix; for each VARYING generator g_a that is a GF(2) combination of
    // the chosen basis generators, replace g_a ← g_a·∏_{k∈dep} g_{basis[k]} (right-CX rotation:
    // Zrow[a]·=Zrow[basis], Xrow[basis]·=Xrow[a]) — zeroing its variation while keeping a valid frame.
    {
        std::vector<int> basis;
        std::vector<std::vector<uint8_t>> redcol;             // reduced difference column per basis (len xs)
        std::vector<std::vector<uint8_t>> comp;               // expansion of redcol[bi] in ORIGINAL basis
        for (int a = 0; a < N; ++a) {
            std::vector<uint8_t> col(xs);
            for (int g = 0; g < xs; ++g) col[g] = (uint8_t)(fsynd[g][a] ^ fsynd[0][a]);
            bool allzero = true; for (auto v : col) if (v) { allzero = false; break; }
            if (allzero) continue;                            // fixed generator — nothing to do
            std::vector<uint8_t> resid = col;
            std::vector<uint8_t> expand(basis.size(), 0);
            for (size_t bi = 0; bi < basis.size(); ++bi) {
                int pr = -1; for (int g = 0; g < xs; ++g) if (redcol[bi][g]) { pr = g; break; }
                if (pr >= 0 && resid[pr]) {
                    for (int g = 0; g < xs; ++g) resid[g] ^= redcol[bi][g];
                    for (size_t k = 0; k < comp[bi].size(); ++k) expand[k] ^= comp[bi][k];
                }
            }
            bool resid_zero = true; for (auto v : resid) if (v) { resid_zero = false; break; }
            if (resid_zero) {
                Pauli dofa = U.Xrow[a];                       // snapshot d_a before Zrow edits
                for (size_t k = 0; k < expand.size(); ++k) if (expand[k]) {
                    int abase = basis[k];
                    U.Zrow[a]     = Pauli::multiply(U.Zrow[a], U.Zrow[abase]);
                    U.Xrow[abase] = Pauli::multiply(U.Xrow[abase], dofa);
                    // Part A: O(χ·n) in-place fsynd update — sign of product g_a*g_abase equals
                    // XOR of individual signs, so fsynd[g][a] ^= fsynd[g][abase] for each ray.
                    // abase is a basis (independent) column: its fsynd rows are never modified
                    // by subsequent re-base steps (only dependent columns are updated), so this
                    // read of fsynd[g][abase] is always fresh.
                    for (int g = 0; g < xs; ++g) fsynd[g][a] ^= fsynd[g][abase];
                }
            } else {
                std::vector<uint8_t> cc(basis.size() + 1, 0); cc[basis.size()] = 1;
                for (size_t k = 0; k < expand.size(); ++k) cc[k] = expand[k];
                for (auto& v : comp) v.push_back(0);
                basis.push_back(a);
                redcol.push_back(resid);
                comp.push_back(cc);
            }
        }
        U.reset_dual();
        // Part A: fsynd is already up-to-date via the in-place XOR updates above (O(χ·n) total).
        // The full O(χ·n²) recompute is skipped.
    }

    eps.assign(N, 0);
    for (int a = 0; a < N; ++a) eps[a] = fsynd[0][a];         // anchor = rays[0] ⇒ its syndrome is eps
    std::vector<int> nfree;
    for (int a = 0; a < N; ++a) {
        bool vary = false;
        for (int g = 1; g < xs; ++g) if (fsynd[g][a] != fsynd[0][a]) { vary = true; break; }
        if (vary) nfree.push_back(a);
    }
    free = nfree;
    branches.clear();
    branches.reserve(xs);
    for (int g = 0; g < xs; ++g) {
        Branch br; br.sigma.resize(nfree.size());
        for (int d = 0; d < (int)nfree.size(); ++d)
            br.sigma[d] = (uint8_t)(fsynd[g][nfree[d]] ^ eps[nfree[d]]);
        // Gauge: built = D_σ|φ0'⟩ equals rays[g] up to a unit phase α (same ray — same syndrome by
        // construction); divide the coefficient by α so to_statevector renders coeffs[g]·rays[g].
        ExactPhase ov = branch_gauge(*rays[g], br.sigma);              // α = ⟨rays[g]|built⟩, |α|=1
        std::complex<double> alpha = ov.is_zero ? std::complex<double>(1, 0) : ov.to_complex();
        br.c = coeffs[g] / alpha;
        branches.push_back(std::move(br));
    }
    canonicalise();
}

void CanonicalStabSum::ray_syndromes(const std::vector<const AffineState*>& rays,
                                     std::vector<std::vector<uint8_t>>& fs) const {
    const int N = n();
    const int xs = (int)rays.size();
    fs.assign(xs, std::vector<uint8_t>(N, 0));
    // NEW (O(χ·n²)): disentangle each ray_g ONCE by the current frame U, reading ALL n syndrome bits
    // from the resulting computational-basis offset. The disentangling circuit is derived from the
    // CURRENT frame generators U.Zrow (rebuild_from_rays' call #2 runs after the re-base edited
    // them; rays[0]/anchor is NOT re-based, so a circuit derived from rays[0] alone would be stale).
    // U†|ray_g⟩ = |s_g⟩, a computational-basis state whose bit string IS the syndrome.
    //
    // perm[a]  = qubit index in the disentangled state for generator a.
    // trans[a] = GF(2) bitmask of original generators mixed into row a by row multiplies.
    // After applying the circuit, bit perm[a] of the disentangled state holds
    //   XOR_{j: trans[a][j]=1} fs[j]   (NOT just fs[a], due to row-multiply mixing);
    // recover_syndromes() undoes this mixing via forward substitution.
    std::vector<int> perm;
    std::vector<std::vector<uint64_t>> trans;
    std::vector<DisentangleGate> dis = disentangle_from_generators(U.Zrow, &perm, &trans);
    for (int g = 0; g < xs; ++g) {
        auto c_dis = rays[(size_t)g]->clone();
        AffineState& s = static_cast<AffineState&>(*c_dis);
        for (const auto& gate : dis) apply_disentangle_gate(s, gate);
        recover_syndromes(N, s, perm, trans, fs[(size_t)g]);
    }
}

ExactPhase CanonicalStabSum::branch_gauge(const AffineState& ray, const std::vector<uint8_t>& sigma) const {
    auto built = anchor_state().clone();
    for (int d = 0; d < (int)free.size(); ++d) if (sigma[(size_t)d]) U.Xrow[free[(size_t)d]].apply_to(*built);
    return ray.inner_product(*built);
}

// ── Coset-family construction from the materialised rays of weight ≤ 2 (3.1.8; now the TEST ORACLE
//    of from_coset_family below; byte-identical to from_rays) ─────────────────────────────────────
//
// Setting.  ψ is a stabilizer state, Z_1..Z_r commuting Paulis, and every joint-eigenspace coset g
// (Z_i = s_i·(−1)^{g_i}) carries weight 2^{-r} of ψ — i.e. NO nontrivial product of the Z_i is a
// stabilizer of ψ (up to sign).  ray_g := 2^{r/2}·Π_g|ψ⟩ (a unit vector; assemble_from_parities'
// projections return exactly this: each pauli_project factor is the real 2^{-1/2}).
//
// (A) Syndromes are linear in g.  Because no product of the Z_i is in ±Stab(ψ), the commutation map
// Stab(ψ) → F_2^r (s ↦ which Z_i anticommute with s) is onto: pick s_1..s_r ∈ Stab(ψ) with s_i
// anticommuting with Z_i alone.  Then s_i Π_h = Π_{h⊕e_i} s_i and s_i|ψ⟩ = |ψ⟩, so
//     ray_g = M_g ray_0,   M_g = Π_i s_i^{g_i}   (products of stabilizers of ψ: no phase).
// A frame generator G_a (a stabilizer of ray_0 up to sign) therefore has sign on ray_g equal to its
// sign on ray_0 times Π_i (−1)^{g_i·[G_a, s_i anticommute]}: fsynd[g][a] = fsynd[0][a] ⊕ u_a·g with
// u_a,i = fsynd[e_i][a] ⊕ fsynd[0][a].  Every GF(2) decision of rebuild_from_rays' re-base is a
// statement about the column space {u_a} (dependence of a column on earlier ones, and its UNIQUE
// expansion over an independent basis), so running it on the r-bit u_a reproduces it exactly.
//
// (B) The gauge is a quadratic character.  With σ(g) = L·g (the branch's sign pattern over `free`)
// the gauge operator is D_σ = Π_d X_d^{σ_d}; the destabilisers commute pairwise, so
// D_σ(g) = ±Π_i K_i^{g_i} (K_i = D_σ(e_i)) with a sign (−1)^{bilinear(g)} from X_d² = ±I.  Then
//     α(g) = ⟨ray_g|D_σ(g) ray_0⟩ = ⟨ray_0|M_g D_σ(g)|ray_0⟩,
// and reordering M_g D_σ(g) into Π_i (s_i K_i)^{g_i} costs another (−1)^{bilinear(g)}.  Each
// Y_i = s_i K_i maps ray_0 to α(e_i)·ray_0 (K_i: coset 0 → e_i, s_i: e_i → 0), so
//     α(g) = Π_i α(e_i)^{g_i} · (−1)^{Σ_{i<j} b_ij g_i g_j},
// fully determined by α at weight ≤ 2 (b_ij from α(e_i⊕e_j)/(α(e_i)α(e_j)) = ±1).  α is exact
// (ExactPhase, scale 0), so the coefficient double c = coeffs[g] / α.to_complex() is bit-identical.
CanonicalStabSum CanonicalStabSum::from_coset_family_rays(
        int n, int r, const std::function<std::unique_ptr<AffineState>(uint32_t)>& make_ray,
        const std::vector<std::complex<double>>& coeffs, CosetTrace* trace) {
    if (r < 0 || r > kMaxBareRank)
        throw BareStateCapacityError(bare_rank_refusal(r, "from_coset_family"));
    const uint32_t xs = 1u << r;
    if (coeffs.size() != (size_t)xs)
        throw std::invalid_argument("from_coset_family: coeffs must have 2^r entries");
    CanonicalStabSum s(n);
    const int N = n;
    std::unique_ptr<AffineState> ray0 = make_ray(0);
    // anchor = ray_0; frame from its stabilizer generators (g_a = Zrow[a]) — as rebuild_from_rays.
    std::vector<Pauli> gens = stabilizer_generators(*ray0);
    s.U = CliffordTableau::from_generators(gens);
    s.set_anchor(std::unique_ptr<AffineState>(static_cast<AffineState*>(ray0->clone().release())));

    // (A) syndromes of rays 0 and e_i under the initial frame; u[a] bit i = linear coefficient.
    std::vector<std::unique_ptr<AffineState>> unit((size_t)r);
    for (int i = 0; i < r; ++i) unit[(size_t)i] = make_ray(1u << i);
    std::vector<std::vector<uint8_t>> fs;
    {
        std::vector<const AffineState*> rp;
        rp.push_back(ray0.get());
        for (int i = 0; i < r; ++i) rp.push_back(unit[(size_t)i].get());
        s.ray_syndromes(rp, fs);
    }
    std::vector<uint8_t> fs0 = fs[0];
    std::vector<uint32_t> u((size_t)N, 0);
    for (int a = 0; a < N; ++a)
        for (int i = 0; i < r; ++i)
            if (fs[(size_t)i + 1][(size_t)a] ^ fs0[(size_t)a]) u[(size_t)a] |= 1u << i;
    if (trace) trace->fs = fs;

    // Re-base dependent varying generators (rebuild_from_rays' loop, on the r-bit columns u_a).
    // red[k]/comp[k]: reduced vector + its expansion over the basis indices (bit t = basis[t]).
    {
        std::vector<int> basis;
        std::vector<uint32_t> red;
        std::vector<uint32_t> comp;
        for (int a = 0; a < N; ++a) {
            if (!u[(size_t)a]) continue;                      // fixed generator
            uint32_t resid = u[(size_t)a], expand = 0;
            for (size_t k = 0; k < red.size(); ++k) {
                const uint32_t lb = red[k] & (0u - red[k]);    // pivot = lowest set bit
                if (resid & lb) { resid ^= red[k]; expand ^= comp[k]; }
            }
            if (!resid) {
                Pauli dofa = s.U.Xrow[(size_t)a];             // snapshot d_a before Zrow edits
                for (size_t k = 0; k < basis.size(); ++k) if ((expand >> k) & 1u) {
                    const int abase = basis[k];
                    s.U.Zrow[(size_t)a]     = Pauli::multiply(s.U.Zrow[(size_t)a], s.U.Zrow[(size_t)abase]);
                    s.U.Xrow[(size_t)abase] = Pauli::multiply(s.U.Xrow[(size_t)abase], dofa);
                    u[(size_t)a] ^= u[(size_t)abase];
                    fs0[(size_t)a] ^= fs0[(size_t)abase];
                }
            } else {
                // reduce the existing reduced vectors' pivots out of nothing — keep pivots distinct:
                // resid's lowest bit is not a pivot of any earlier red (each was XOR-cleared above).
                red.push_back(resid);
                comp.push_back(expand | (1u << basis.size()));
                basis.push_back(a);
            }
        }
        s.U.reset_dual();
    }
    s.eps.assign((size_t)N, 0);
    for (int a = 0; a < N; ++a) s.eps[(size_t)a] = fs0[(size_t)a];
    std::vector<int> nfree;
    for (int a = 0; a < N; ++a) if (u[(size_t)a]) nfree.push_back(a);
    s.free = nfree;
    const int rf = (int)nfree.size();
    std::vector<uint32_t> L((size_t)rf);                       // σ_d(g) = parity(L[d] & g)
    for (int d = 0; d < rf; ++d) L[(size_t)d] = u[(size_t)nfree[(size_t)d]];
    auto sigma_of = [&](uint32_t g, std::vector<uint8_t>& sg) {
        sg.resize((size_t)rf);
        for (int d = 0; d < rf; ++d) sg[(size_t)d] = (uint8_t)(__builtin_popcount(L[(size_t)d] & g) & 1);
    };

    // (B) gauge at weight ≤ 2, exact; then the quadratic closed form.
    auto bad = [](const std::string& why) {
        throw BareStateInvariantError("from_coset_family: " + why +
                                      " — the coset-family structure does not hold; refusing");
    };
    std::vector<uint8_t> sg;
    std::vector<int> a1((size_t)r);                            // z8 of α(e_i)
    for (int i = 0; i < r; ++i) {
        sigma_of(1u << i, sg);
        ExactPhase ov = s.branch_gauge(*unit[(size_t)i], sg);
        if (ov.is_zero || ov.scale != 0) bad("unit-ray gauge is not a unit phase");
        a1[(size_t)i] = ov.z8;
    }
    unit.clear();
    if (trace) { trace->a1 = a1; trace->a2.clear(); trace->u = u; }
    std::vector<uint8_t> b2((size_t)r * (size_t)r, 0);         // b_ij ∈ {0,1}: (−1)^{b_ij g_i g_j}
    for (int i = 0; i < r; ++i)
        for (int j = i + 1; j < r; ++j) {
            const uint32_t g = (1u << i) | (1u << j);
            std::unique_ptr<AffineState> ray = make_ray(g);
            sigma_of(g, sg);
            ExactPhase ov = s.branch_gauge(*ray, sg);
            if (ov.is_zero || ov.scale != 0) bad("weight-2 gauge is not a unit phase");
            if (trace) trace->a2.push_back(ov.z8);
            const int dz = ((ov.z8 - a1[(size_t)i] - a1[(size_t)j]) % 8 + 8) % 8;
            if (dz != 0 && dz != 4) bad("weight-2 gauge is not a quadratic character");
            b2[(size_t)i * (size_t)r + (size_t)j] = (uint8_t)(dz == 4);
        }
    if (trace) trace->b2 = b2;
    auto alpha_z8 = [&](uint32_t g) {                          // direct O(r^2) evaluation
        int z = 0;
        for (int i = 0; i < r; ++i) if ((g >> i) & 1u) {
            z += a1[(size_t)i];
            for (int j = i + 1; j < r; ++j)
                if (((g >> j) & 1u) && b2[(size_t)i * (size_t)r + (size_t)j]) z += 4;
        }
        return ((z % 8) + 8) % 8;
    };
    // The same form tabulated in O(1) per branch: with i the lowest set bit of g and g' = g ^ e_i
    // (every other set bit of g lies above i), z(g) = z(g') + a1[i] + 4·parity(brow[i] & g').
    std::vector<uint32_t> brow((size_t)r, 0);
    for (int i = 0; i < r; ++i)
        for (int j = i + 1; j < r; ++j) if (b2[(size_t)i * (size_t)r + (size_t)j]) brow[(size_t)i] |= 1u << j;
    std::vector<uint8_t> ztab((size_t)xs, 0);
    for (uint32_t g = 1; g < xs; ++g) {
        const int i = __builtin_ctz(g);
        const uint32_t gp = g & (g - 1);
        ztab[g] = (uint8_t)((ztab[gp] + a1[(size_t)i] + 4 * (__builtin_popcount(brow[(size_t)i] & gp) & 1)) & 7);
    }

    // Branches, in index order g = 0..χ-1 (rebuild_from_rays' order).  α(g)'s complex value comes
    // from the same ExactPhase::to_complex the old per-branch inner product would have produced.
    std::complex<double> unit_alpha[8];
    for (int z = 0; z < 8; ++z) unit_alpha[z] = ExactPhase{false, 0, z}.to_complex();
    s.branches.clear();
    s.branches.reserve(xs);
    for (uint32_t g = 0; g < xs; ++g) {
        Branch br;
        sigma_of(g, br.sigma);
        br.c = coeffs[g] / unit_alpha[ztab[g]];            // α = ExactPhase{false, 0, z}: never zero
        s.branches.push_back(std::move(br));
    }

    // Spot check (a mechanism, not a comment): re-derive a few further branches the OLD way — the
    // per-ray disentangle syndrome and the exact inner-product gauge — and refuse on any mismatch.
    // Probes are branches of weight >= 2 (weight 0/1 are the measured inputs): all-ones (every pair
    // term of the gauge present) plus up to four pseudo-random ones; one disentangling circuit is
    // derived for all of them.
    {
        std::vector<uint32_t> probe;
        auto add_probe = [&](uint32_t g) {
            if (__builtin_popcount(g) < 2) return;
            for (uint32_t q : probe) if (q == g) return;
            probe.push_back(g);
        };
        if (xs > 1) add_probe(xs - 1);
        uint64_t h = 0x9e3779b97f4a7c15ull;
        for (int t = 0; t < 4 && xs > 4; ++t) { h = h * 6364136223846793005ull + 1442695040888963407ull;
                                                add_probe((uint32_t)(h >> 33) & (xs - 1)); }
        std::vector<std::unique_ptr<AffineState>> prays;
        std::vector<const AffineState*> pp;
        for (uint32_t g : probe) { prays.push_back(make_ray(g)); pp.push_back(prays.back().get()); }
        std::vector<std::vector<uint8_t>> pfs;
        if (!pp.empty()) s.ray_syndromes(pp, pfs);
        for (size_t t = 0; t < probe.size(); ++t) {
            const uint32_t g = probe[t];
            for (int a = 0; a < N; ++a) {
                const uint8_t want = (uint8_t)(s.eps[(size_t)a] ^ (__builtin_popcount(u[(size_t)a] & g) & 1));
                if (pfs[t][(size_t)a] != want) bad("syndrome of spot-check branch disagrees");
            }
            ExactPhase ov = s.branch_gauge(*prays[t], s.branches[g].sigma);
            if (ov.is_zero || ov.scale != 0 || ov.z8 != alpha_z8(g) || ov.z8 != ztab[g])
                bad("gauge of spot-check branch disagrees");
        }
    }
    // canonicalise() (what from_rays ends with) is the identity here: the branches realise EVERY
    // σ(g) = L·g with the rf = r rows of L independent (they are the re-based basis columns), so
    // its column echelon makes every column a pivot — no re-base, no constant fold, `free` and all
    // σ unchanged.  Skip it in exactly that case; anything else takes the general path.
    if (rf != r) s.canonicalise();
    return s;
}

// ── Coset-family construction from the Pauli algebra (3.1.9; production) ────────────────────────
//
// Same object as from_coset_family_rays (the 3.1.8 construction, kept as the oracle), but the unit
// rays and the weight-2 rays are never materialised.  With the setting and (A)/(B) above:
//
// (A') The s_i are found directly.  Let h_1..h_n be +1 stabilizer generators of ψ
// (stabilizer_generators pins every phase to +1) and C the n×r matrix C_ki = [h_k anticommutes with
// zs_i].  "No nontrivial product of the zs_i lies in ±Stab(ψ)" ⟺ rank C = r (a Pauli commuting with
// all of a maximal stabilizer group is in it up to phase) — checked, refusal otherwise.  Gauss–Jordan
// on C's rows gives combinations c^(i) with c^(i)·C = e_i; s_i := Π_k h_k^{c^(i)_k} (ascending, exact
// products of commuting +1 stabilizers, so s_i ψ = ψ).  Any other choice differs by an element of
// Stab(ψ) commuting with every zs_j, i.e. a +1 stabilizer of ray_0 — M_g ray_0 is choice-free.
// Then, for the initial frame (Zrow[a] = stabilizer_generators(ray_0)[a] verbatim, each pinned +1
// on ray_0, so the anchor syndrome fs[0] ≡ 0):
//     fs[e_i][a] = fs[0][a] ⊕ [Zrow[a] anticommutes with s_i]
// because Zrow[a] s_i ray_0 = (−1)^{[Zrow[a], s_i]} s_i Zrow[a] ray_0.  The re-base then runs
// unchanged on these columns.
//
// (B') The gauge at weight ≤ 2 is a frame expectation.  α(g) = ⟨ray_g|D_σ(g) ray_0⟩ (branch_gauge:
// D_σ applied in ascending d, i.e. the operator X_{d_m}⋯X_{d_1}) and ⟨ray_g| = ⟨ray_0|M_g (M_g is a
// product of commuting Hermitian Paulis), so α(g) = ⟨ray_0|P|ray_0⟩ with P = M_g·D_σ(g) an exact
// Pauli product.  A unit-modulus α means P ∝ an element of the (re-based) anchor stabilizer group
// ⟨Zrow⟩; its coordinates are c_a = [P anticommutes with Xrow[a]] (the tableau pairing
// ⟨Xrow[a], Zrow[b]⟩ = δ_ab), Q := Π_{a: c_a} Zrow[a] (ascending) has the same X/Z bits as P —
// checked — and P = i^{P.phase − Q.phase}·Q, Zrow[a] ray_0 = (−1)^{eps_a} ray_0, so
//     α(g) = i^{P.phase − Q.phase} · (−1)^{Σ_a c_a eps_a}      (an exact z8: 2·Δphase + 4·sign).
// b_ij then follows exactly as in (B).  Cost: O(r·n²/64) words for the s_i, O(n²/64) per gauge —
// no AffineState is disentangled (3.1.8 ran an O(n³) canonicalising H-circuit per unit ray).
//
// Spot check (mechanism): every probe branch is materialised by projection (make_ray, which also
// re-checks the σ-independent projection amplitude) and its gauge is the exact inner product
// branch_gauge — the 3.1.7/3.1.8 definition.  Unit modulus of ⟨ray_g|D_σ(g) anchor⟩ already proves
// ray_g ∝ D_σ(g)·anchor, whose frame syndrome is eps ⊕ σ(g) on `free` = the predicted eps ⊕ u·g, so
// the 3.1.8 per-probe disentangled syndrome is implied and is no longer recomputed.
CanonicalStabSum CanonicalStabSum::from_coset_family(
        int n, int r, const AffineState& psi, const std::vector<Pauli>& zs,
        const std::function<std::unique_ptr<AffineState>(uint32_t)>& make_ray,
        const std::vector<std::complex<double>>& coeffs, CosetTrace* trace) {
    if (r < 0 || r > kMaxBareRank)
        throw BareStateCapacityError(bare_rank_refusal(r, "from_coset_family"));
    const uint32_t xs = 1u << r;
    if (coeffs.size() != (size_t)xs)
        throw std::invalid_argument("from_coset_family: coeffs must have 2^r entries");
    if (zs.size() != (size_t)r || psi.n_ != n)
        throw std::invalid_argument("from_coset_family: need r projected Paulis on the n-qubit psi");
    auto bad = [](const std::string& why) {
        throw BareStateInvariantError("from_coset_family: " + why +
                                      " — the coset-family structure does not hold; refusing");
    };
    CanonicalStabSum s(n);
    const int N = n;
    std::unique_ptr<AffineState> ray0 = make_ray(0);
    // anchor = ray_0; frame from its stabilizer generators (g_a = Zrow[a]) — as rebuild_from_rays.
    std::vector<Pauli> gens = stabilizer_generators(*ray0);
    s.U = CliffordTableau::from_generators(gens);
    s.set_anchor(std::unique_ptr<AffineState>(static_cast<AffineState*>(ray0->clone().release())));

    // (A') s_i ∈ Stab(ψ) with [s_i, zs_j] = δ_ij.
    std::vector<Pauli> sv;
    if (r > 0) {
        std::vector<Pauli> hg = stabilizer_generators(psi);
        const int H = (int)hg.size();
        const int SW = (H + 63) / 64;
        std::vector<uint32_t> red;                             // reduced commutation rows
        std::vector<std::vector<uint64_t>> sel;                // their generator combinations
        for (int k = 0; k < H && (int)red.size() < r; ++k) {
            uint32_t row = 0;
            for (int i = 0; i < r; ++i)
                if (Pauli::anticommute_bit(hg[(size_t)k], zs[(size_t)i])) row |= 1u << i;
            if (!row) continue;
            std::vector<uint64_t> sk((size_t)SW, 0);
            sk[(size_t)(k >> 6)] |= 1ull << (k & 63);
            for (size_t t = 0; t < red.size(); ++t) {
                const uint32_t lb = red[t] & (0u - red[t]);    // pivot = lowest set bit
                if (row & lb) { row ^= red[t]; for (int w = 0; w < SW; ++w) sk[(size_t)w] ^= sel[t][(size_t)w]; }
            }
            if (row) { red.push_back(row); sel.push_back(std::move(sk)); }
        }
        if ((int)red.size() != r) bad("a product of the projected Paulis stabilizes psi (rank " +
                                      std::to_string(red.size()) + " < " + std::to_string(r) + ")");
        for (size_t t = 0; t < red.size(); ++t) {              // Gauss–Jordan: every row a unit vector
            const uint32_t lb = red[t] & (0u - red[t]);
            for (size_t t2 = 0; t2 < red.size(); ++t2)
                if (t2 != t && (red[t2] & lb)) {
                    red[t2] ^= red[t];
                    for (int w = 0; w < SW; ++w) sel[t2][(size_t)w] ^= sel[t][(size_t)w];
                }
        }
        sv.assign((size_t)r, Pauli(N));
        std::vector<uint8_t> have((size_t)r, 0);
        for (size_t t = 0; t < red.size(); ++t) {
            if (red[t] & (red[t] - 1)) bad("commutation matrix did not reduce to the identity");
            const int i = __builtin_ctz(red[t]);
            have[(size_t)i] = 1;
            for (int k = 0; k < H; ++k)
                if ((sel[t][(size_t)(k >> 6)] >> (k & 63)) & 1ull) pmul_into(sv[(size_t)i], hg[(size_t)k]);
        }
        for (int i = 0; i < r; ++i) if (!have[(size_t)i]) bad("commutation matrix is singular");
    }

    // Initial-frame syndromes: fs[0] ≡ 0 (Zrow = gens pinned +1 on ray_0); u[a] bit i = [Zrow[a], s_i].
    std::vector<uint8_t> fs0((size_t)N, 0);
    std::vector<uint32_t> u((size_t)N, 0);
    for (int a = 0; a < N; ++a)
        for (int i = 0; i < r; ++i)
            if (Pauli::anticommute_bit(s.U.Zrow[(size_t)a], sv[(size_t)i])) u[(size_t)a] |= 1u << i;
    if (trace) {
        trace->fs.assign((size_t)r + 1, fs0);
        for (int i = 0; i < r; ++i)
            for (int a = 0; a < N; ++a) trace->fs[(size_t)i + 1][(size_t)a] = (uint8_t)(fs0[(size_t)a] ^ ((u[(size_t)a] >> i) & 1u));
    }

    // Re-base dependent varying generators (rebuild_from_rays' loop, on the r-bit columns u_a).
    {
        std::vector<int> basis;
        std::vector<uint32_t> red;
        std::vector<uint32_t> comp;
        for (int a = 0; a < N; ++a) {
            if (!u[(size_t)a]) continue;                      // fixed generator
            uint32_t resid = u[(size_t)a], expand = 0;
            for (size_t k = 0; k < red.size(); ++k) {
                const uint32_t lb = red[k] & (0u - red[k]);    // pivot = lowest set bit
                if (resid & lb) { resid ^= red[k]; expand ^= comp[k]; }
            }
            if (!resid) {
                Pauli dofa = s.U.Xrow[(size_t)a];             // snapshot d_a before Zrow edits
                for (size_t k = 0; k < basis.size(); ++k) if ((expand >> k) & 1u) {
                    const int abase = basis[k];
                    s.U.Zrow[(size_t)a]     = Pauli::multiply(s.U.Zrow[(size_t)a], s.U.Zrow[(size_t)abase]);
                    s.U.Xrow[(size_t)abase] = Pauli::multiply(s.U.Xrow[(size_t)abase], dofa);
                    u[(size_t)a] ^= u[(size_t)abase];
                    fs0[(size_t)a] ^= fs0[(size_t)abase];
                }
            } else {
                red.push_back(resid);
                comp.push_back(expand | (1u << basis.size()));
                basis.push_back(a);
            }
        }
        s.U.reset_dual();
    }
    s.eps.assign((size_t)N, 0);
    for (int a = 0; a < N; ++a) s.eps[(size_t)a] = fs0[(size_t)a];
    std::vector<int> nfree;
    for (int a = 0; a < N; ++a) if (u[(size_t)a]) nfree.push_back(a);
    s.free = nfree;
    const int rf = (int)nfree.size();
    std::vector<uint32_t> L((size_t)rf);                       // σ_d(g) = parity(L[d] & g)
    for (int d = 0; d < rf; ++d) L[(size_t)d] = u[(size_t)nfree[(size_t)d]];
    auto sigma_of = [&](uint32_t g, std::vector<uint8_t>& sg) {
        sg.resize((size_t)rf);
        for (int d = 0; d < rf; ++d) sg[(size_t)d] = (uint8_t)(__builtin_popcount(L[(size_t)d] & g) & 1);
    };

    // (B') α(g) at weight ≤ 2 as the anchor expectation of P = M_g·D_σ(g).
    std::vector<uint8_t> sg;
    Pauli D(N), P(N), Q(N), tmp(N);
    auto gauge_z8 = [&](uint32_t g) -> int {
        sigma_of(g, sg);
        reset_pauli_inplace(D, N);
        for (int d = 0; d < rf; ++d) if (sg[(size_t)d]) {     // D ← Xrow[free[d]]·D (applied last = leftmost)
            tmp = s.U.Xrow[(size_t)nfree[(size_t)d]];
            pmul_into(tmp, D);
            std::swap(D, tmp);
        }
        reset_pauli_inplace(P, N);
        for (int i = 0; i < r; ++i) if ((g >> i) & 1u) pmul_into(P, sv[(size_t)i]);   // M_g
        pmul_into(P, D);                                                              // P = M_g·D
        reset_pauli_inplace(Q, N);
        int sgn = 0;
        for (int a = 0; a < N; ++a)
            if (Pauli::anticommute_bit(P, s.U.Xrow[(size_t)a])) {
                pmul_into(Q, s.U.Zrow[(size_t)a]);
                sgn ^= s.eps[(size_t)a];
            }
        if (Q.x != P.x || Q.z != P.z) bad("gauge Pauli is not in the anchor's stabilizer group");
        return (2 * ((P.phase - Q.phase) & 3) + 4 * sgn) & 7;
    };
    std::vector<int> a1((size_t)r);                            // z8 of α(e_i)
    for (int i = 0; i < r; ++i) a1[(size_t)i] = gauge_z8(1u << i);
    if (trace) { trace->a1 = a1; trace->a2.clear(); trace->u = u; }
    std::vector<uint8_t> b2((size_t)r * (size_t)r, 0);         // b_ij ∈ {0,1}: (−1)^{b_ij g_i g_j}
    for (int i = 0; i < r; ++i)
        for (int j = i + 1; j < r; ++j) {
            const int z = gauge_z8((1u << i) | (1u << j));
            if (trace) trace->a2.push_back(z);
            const int dz = ((z - a1[(size_t)i] - a1[(size_t)j]) % 8 + 8) % 8;
            if (dz != 0 && dz != 4) bad("weight-2 gauge is not a quadratic character");
            b2[(size_t)i * (size_t)r + (size_t)j] = (uint8_t)(dz == 4);
        }
    if (trace) trace->b2 = b2;
    auto alpha_z8 = [&](uint32_t g) {                          // direct O(r^2) evaluation
        int z = 0;
        for (int i = 0; i < r; ++i) if ((g >> i) & 1u) {
            z += a1[(size_t)i];
            for (int j = i + 1; j < r; ++j)
                if (((g >> j) & 1u) && b2[(size_t)i * (size_t)r + (size_t)j]) z += 4;
        }
        return ((z % 8) + 8) % 8;
    };
    // Tabulated in O(1) per branch: with i the lowest set bit of g and g' = g ^ e_i,
    // z(g) = z(g') + a1[i] + 4·parity(brow[i] & g').
    std::vector<uint32_t> brow((size_t)r, 0);
    for (int i = 0; i < r; ++i)
        for (int j = i + 1; j < r; ++j) if (b2[(size_t)i * (size_t)r + (size_t)j]) brow[(size_t)i] |= 1u << j;
    std::vector<uint8_t> ztab((size_t)xs, 0);
    for (uint32_t g = 1; g < xs; ++g) {
        const int i = __builtin_ctz(g);
        const uint32_t gp = g & (g - 1);
        ztab[g] = (uint8_t)((ztab[gp] + a1[(size_t)i] + 4 * (__builtin_popcount(brow[(size_t)i] & gp) & 1)) & 7);
    }

    // Branches, in index order g = 0..χ-1 (rebuild_from_rays' order).
    std::complex<double> unit_alpha[8];
    for (int z = 0; z < 8; ++z) unit_alpha[z] = ExactPhase{false, 0, z}.to_complex();
    s.branches.clear();
    s.branches.reserve(xs);
    for (uint32_t g = 0; g < xs; ++g) {
        Branch br;
        sigma_of(g, br.sigma);
        br.c = coeffs[g] / unit_alpha[ztab[g]];            // α = ExactPhase{false, 0, z}: never zero
        s.branches.push_back(std::move(br));
    }

    // Spot check: probes of weight >= 2 (all-ones: every pair term present, plus up to four
    // pseudo-random ones — the 3.1.8 selection), each materialised by projection and gauged by the
    // exact inner product.  Unit modulus ⇒ the probe's syndrome is the predicted one (see above).
    {
        std::vector<uint32_t> probe;
        auto add_probe = [&](uint32_t g) {
            if (__builtin_popcount(g) < 2) return;
            for (uint32_t q : probe) if (q == g) return;
            probe.push_back(g);
        };
        if (xs > 1) add_probe(xs - 1);
        uint64_t h = 0x9e3779b97f4a7c15ull;
        for (int t = 0; t < 4 && xs > 4; ++t) { h = h * 6364136223846793005ull + 1442695040888963407ull;
                                                add_probe((uint32_t)(h >> 33) & (xs - 1)); }
        for (uint32_t g : probe) {
            std::unique_ptr<AffineState> ray = make_ray(g);
            ExactPhase ov = s.branch_gauge(*ray, s.branches[g].sigma);
            if (ov.is_zero || ov.scale != 0) bad("spot-check branch is not the predicted ray");
            if (ov.z8 != alpha_z8(g) || ov.z8 != ztab[g]) bad("gauge of spot-check branch disagrees");
        }
    }
    // canonicalise() is the identity when rf == r (see from_coset_family_rays); else the general path.
    if (rf != r) s.canonicalise();
    return s;
}

CanonicalStabSum CanonicalStabSum::from_rays(int n, std::vector<std::unique_ptr<AffineState>> rays,
                                             std::vector<std::complex<double>> coeffs) {
    CanonicalStabSum s(n);
    s.rebuild_from_rays(std::move(rays), std::move(coeffs));
    return s;
}

void CanonicalStabSum::ensure_frame_current() const { flush_gates(); }

bool CanonicalStabSum::verify_invariants() const {
    flush_gates();                              // keep the frame current at every public entry point
    int r = (int)free.size();
    int x = chi();
    // Distinctness: σ patterns pairwise distinct.
    for (int i = 0; i < x; ++i)
        for (int j = i + 1; j < x; ++j)
            if (branches[i].sigma == branches[j].sigma) return false;
    // Minimality: GF(2) rank of {σ_i ⊕ σ_0} == r (every free generator genuinely varies / is
    // independent — no over-complete distinguishing generator). We do NOT require χ == 2^r: a
    // single-qubit measurement can mix merge and promotion across pairs, yielding an irreducible
    // branch set that is a valid but NON-full affine subspace (χ < 2^r). Such states are physical
    // and minimal; only redundancy in `free` (rank < r) is the bug to reject.
    if (r > 60) return false;                 // sanity bound for the rank loop
    // Row-reduce the difference matrix {σ_i ⊕ σ_0} (x rows × r cols) over GF(2); rank == r.
    std::vector<std::vector<uint8_t>> M;
    M.reserve(x);
    for (int i = 0; i < x; ++i) {
        std::vector<uint8_t> row(r);
        for (int a = 0; a < r; ++a) row[a] = branches[i].sigma[a] ^ branches[0].sigma[a];
        M.push_back(std::move(row));
    }
    int rank = 0;
    for (int col = 0; col < r; ++col) {
        int piv = -1;
        for (int row = rank; row < x; ++row) if (M[row][col]) { piv = row; break; }
        if (piv < 0) continue;
        std::swap(M[rank], M[piv]);
        for (int row = 0; row < x; ++row)
            if (row != rank && M[row][col])
                for (int a = 0; a < r; ++a) M[row][a] ^= M[rank][a];
        ++rank;
    }
    return rank == r;
}

}  // namespace qeccore
