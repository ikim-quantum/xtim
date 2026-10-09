#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/pauli_kernels.hpp"   // shared single_pauli/pmul_into/conj_cx_inplace/dual_image_rows_scan/partner_hash_chi_min
#include "qeccore/stab_generators.hpp"
#include "qeccore/stab_disentangle.hpp"
#include "qeccore/gf2_gauss.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <string>

namespace qeccore {

// single_pauli now lives in qeccore/pauli_kernels.hpp (shared with framed_superposition / factored_stab).

CanonicalStabSum CanonicalStabSum::from_recipe(int n, int m,
        const std::vector<std::complex<double>>& coeffs, uint64_t seed) {
    if (m < 1 || m > n) throw std::invalid_argument("from_recipe: need 1<=m<=n");
    if ((int)coeffs.size() != (1 << m)) throw std::invalid_argument("from_recipe: coeffs size must be 2^m");

    // Same random Clifford base circuit as the legacy v1 from-recipe convention (identical RNG
    // draws) so the anchor |φ0⟩ and the generators are bit-identical to the v1 branch 0.
    std::mt19937_64 rng(seed);
    auto base = std::make_unique<AffineState>(n);
    for (int d = 0; d < 4 * n; ++d) {
        int q = rng() % n;
        switch (rng() % 6) {
            case 0: base->apply_h(q);   break;
            case 1: base->apply_s(q);   break;
            case 2: base->apply_sdg(q); break;
            case 3: base->apply_x(q);   break;
            case 4: base->apply_y(q);   break;
            default: base->apply_z(q);  break;
        }
        if (n >= 2 && rng() % 3 == 0) {
            int c = rng() % n, t = rng() % n; while (t == c) t = rng() % n;
            (rng() & 1) ? base->apply_cx(c, t) : base->apply_cz(c, t);
        }
    }

    std::vector<Pauli> gens = stabilizer_generators(*base);
    CanonicalStabSum S(n);
    S.U = CliffordTableau::from_generators(gens);

    // Anchor signs: g_a |φ0⟩ = (-1)^eps[a] |φ0⟩ (g_a = gens[a]; +1 since gens extracted from base).
    for (int a = 0; a < n; ++a) {
        auto c = base->clone();
        gens[a].apply_to(*c);
        ExactPhase ip = base->inner_product(*c);
        S.eps[a] = (!ip.is_zero && ip.to_complex().real() < 0) ? 1 : 0;
    }

    // free = first m generators (the distinguishing set); anchor = the base state.
    S.free.resize(m);
    for (int d = 0; d < m; ++d) S.free[d] = d;
    S.anchor = std::move(base);

    // Normalize coefficients (match v1, which normalizes the orthonormal branch sum).
    double nrm2 = 0.0; for (auto& z : coeffs) nrm2 += std::norm(z);
    double inv = (nrm2 > 0) ? 1.0 / std::sqrt(nrm2) : 1.0;

    // 2^m branches: σ_i = bits of i over `free`; |φ_i⟩ = ∏_{a∈free:σ_i[a]} d_a |φ0⟩.
    S.branches.resize(1 << m);
    for (int p = 0; p < (1 << m); ++p) {
        Branch& br = S.branches[p];
        br.sigma.resize(m);
        for (int d = 0; d < m; ++d) br.sigma[d] = (uint8_t)((p >> d) & 1);
        br.c = coeffs[p] * inv;
    }
    return S;
}

// χ-independent Clifford gates (theory §8). DEFERRED frame conjugation: update ONLY the anchor |φ0⟩
// (eager, ~bare-affine cost) and RECORD the gate; the O(n²) shared-frame conjugation (U.left_*, which
// updates g_a = U Z_a U† and d_a = U X_a U†) is replayed lazily by flush_gates() the first time U is
// read. eps/free/branches are untouched. The anchor must stay current (its global phase + the b/y form
// feed every branch materialisation), so it is updated eagerly here; only the tableau is deferred.
void CanonicalStabSum::apply_h(int q)            { pending_gates.push_back({0, q, -1}); anchor->apply_h(q); }
void CanonicalStabSum::apply_s(int q)            { pending_gates.push_back({1, q, -1}); anchor->apply_s(q); }
void CanonicalStabSum::apply_sdg(int q)          { pending_gates.push_back({2, q, -1}); anchor->apply_sdg(q); }
void CanonicalStabSum::apply_x(int q)            { pending_gates.push_back({3, q, -1}); anchor->apply_x(q); }
void CanonicalStabSum::apply_y(int q)            { pending_gates.push_back({4, q, -1}); anchor->apply_y(q); }
void CanonicalStabSum::apply_z(int q)            { pending_gates.push_back({5, q, -1}); anchor->apply_z(q); }
void CanonicalStabSum::apply_cx(int c, int t)    { pending_gates.push_back({6, c, t});  anchor->apply_cx(c, t); }
void CanonicalStabSum::apply_cz(int c, int t)    { pending_gates.push_back({7, c, t});  anchor->apply_cz(c, t); }

// Replay the pending Clifford gates onto the frame U (in apply order), then clear. Mirrors the lazy
// `ensure_dual` pattern: const + mutable so const frame-readers materialise the tableau on demand.
void CanonicalStabSum::flush_gates() const {
    if (pending_gates.empty()) return;
    CliffordTableau& F = const_cast<CliffordTableau&>(U);
    for (const PendingGate& g : pending_gates) {
        switch (g.kind) {
            case 0: F.left_h(g.a);        break;
            case 1: F.left_s(g.a);        break;
            case 2: F.left_sdg(g.a);      break;
            case 3: F.left_x(g.a);        break;
            case 4: F.left_y(g.a);        break;
            case 5: F.left_z(g.a);        break;
            case 6: F.left_cx(g.a, g.b);  break;
            default: F.left_cz(g.a, g.b); break;
        }
    }
    pending_gates.clear();
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
    static thread_local std::vector<std::vector<uint64_t>> col;
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
    static thread_local std::vector<std::vector<uint64_t>> red; red.clear();
    static thread_local std::vector<std::vector<uint8_t>> comp; comp.clear();
    static thread_local std::vector<uint64_t> resid;
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
        if (branches[0].sigma[col2]) { U.Xrow[free[col2]].apply_to(*anchor); eps[free[col2]] ^= 1; }
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

bool CanonicalStabSum::is_commuting_single(int pauli, int q) const {
    flush_gates();                              // materialise the deferred frame before any U read
    Pauli Q = U.conjugate_single(pauli, q);     // U† P U from forward rows (no inverse tableau needed)
    for (int a = 0; a < n(); ++a) if (Q.xbit(a)) return false;
    return true;
}

// ── Born probabilities ────────────────────────────────────────────────────────────────────────
// Commuting (Case A): p_m from the per-branch eigenvalues. Anticommuting (Case B): the off-diagonal
// Born formula Eq. (6) — no collapse. Both are closed form from the frame algebra (O(n²)+O(χ²)).
// conj_cx_inplace / pmul_into / dual_image_rows_scan / partner_hash_chi_min are shared kernels from
// qeccore/pauli_kernels.hpp.

std::pair<double, double> CanonicalStabSum::born_probabilities_single(int pauli, int q) const {
    flush_gates();                              // materialise the deferred frame before any U read
    // Q = U† P U from the cached inverse rows when the dual is valid (O(n)), else dual_image.
    return born_from_conjugated(U.conjugate_single(pauli, q));
}

// Multi-qubit generalization: P is any Hermitian n-qubit Pauli (i^ph X^x Z^z with ph = #Y mod 4 —
// the historical v1 born-probability convention). Q = U†PU comes from dual_image (forward
// rows, O(n²), valid with a stale dual); everything downstream only ever uses Q, so Case A / Case B
// are shared verbatim with the single-qubit path. <P> = p₊ − p₋.
std::pair<double, double> CanonicalStabSum::born_probabilities(const Pauli& P) const {
    if (P.n != n()) throw std::logic_error("born_probabilities: Pauli width != n");
    flush_gates();                              // materialise the deferred frame before any U read
    // Q = U†PU: from the cached inverse rows when the dual tableau is valid (conjugate,
    // O(weight) row folds), else from the forward rows (dual_image, O(n²)) — never force the
    // O(n³) rebuild. Both produce the same unique exact Pauli.
    return born_from_conjugated(U.dual_valid() ? U.conjugate(P) : U.dual_image(P));
}

// Shared Born evaluation from the frame-conjugated operator Q = U†PU = i^{ph} X^{qx} Z^{qz}.
//
// DIRECT pairing (no pivot rotation — user-identified structure): in the conjugated picture every
// branch is a computational basis state labelled by its sign vector, U†|φ_i⟩ = X^{w_i}(U†|φ0⟩) =
// e^{iθ}|synd_i⟩ with a COMMON phase θ (the destabiliser words are phase-free pure-X operators on
// basis states, so the canonical branch construction pins all relative phases). Hence
//   ⟨φ_i|P|φ_j⟩ = ⟨synd_i|Q|synd_j⟩ = i^{ph}·(−1)^{qz·synd_j}·[synd_i = synd_j ⊕ qx].
// Diagonal: nonzero only when qx = 0 (Case A; ±1 read off the tableau signs). Off-diagonal:
// branch j pairs with the unique i whose signs are j's flipped EXACTLY on supp(qx) — a single
// XOR-and-compare per pair against the ORIGINAL sign vectors. This replaces the former Step-1
// pivot rotation (pivot choice + synd mask pass + |A\p| CX conjugations of Q — the whole
// weight-dependent cost, inherited from the measurement path where the rotation IS needed for
// the collapse). Each term is i^{(ph + 2·parity) mod 4} from one pow-computed table — the same
// exact matrix element the rotated formula produced; the float regrouping can move pp by ~1e-16
// (within the statevector-oracle fuzz tolerance; all suite comparisons unaffected).

// partner_hash_chi_min() lives in qeccore/pauli_kernels.hpp (shared with the lean path).

std::pair<double, double> CanonicalStabSum::born_from_conjugated(const Pauli& Q) const {
    const int N = n();
    const int W = (N + 63) / 64;
    bool anyx = false;
    for (int w = 0; w < W; ++w) if (Q.x[w]) { anyx = true; break; }

    if (!anyx) {
        // Case A. Q = i^{Q.phase} Z^{q_z}; s0 = i^{Q.phase}·∏_{a:q_z[a]}(-1)^{eps[a]} ∈ {±1}.
        int s0 = ((Q.phase & 3) == 2) ? -1 : 1;
        for (int a = 0; a < N; ++a) if (Q.zbit(a) && eps[a]) s0 = -s0;
        double pp = 0.0;
        for (const auto& br : branches) {
            int dot = 0;
            for (int d = 0; d < (int)free.size(); ++d)
                if (Q.zbit(free[d]) && br.sigma[d]) dot ^= 1;
            int lam = dot ? -s0 : s0;
            if (lam == +1) pp += std::norm(br.c);
        }
        if (pp < 0) pp = 0; if (pp > 1) pp = 1;
        return {pp, 1.0 - pp};
    }

    // Case B: build the packed per-branch sign vectors (bit a of row i = synd[i][a]).
    const int x = chi();
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    static thread_local std::vector<uint64_t> synd_pk; synd_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = synd_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) row[w] = eps_pk[w];
        for (int d = 0; d < (int)free.size(); ++d)
            if (branches[i].sigma[d]) row[free[d] >> 6] ^= 1ull << (free[d] & 63);
    }
    // expt = Σ_{i,j: synd_i⊕synd_j=qx} conj(c_i)·c_j·i^{(ph + 2·parity(qz&synd_j)) mod 4}.
    // (i = j never matches: qx ≠ 0. If supp(qx) leaves the free set, nothing matches and
    // ⟨P⟩ = 0 — branch signs only vary on free coordinates.)
    static const std::complex<double> Iunit(0, 1);
    static const std::complex<double> ipow[4] = {
        std::pow(Iunit, 0.0), std::pow(Iunit, 1.0), std::pow(Iunit, 2.0), std::pow(Iunit, 3.0)};
    static thread_local std::vector<uint64_t> tgt; tgt.resize(W);
    std::complex<double> expt(0, 0);
    // Each i pairs with the UNIQUE j whose signature sj = si ^ qx (signatures are distinct: synd_j
    // = eps ^ (sigma_j restricted to `free`), an injection of the distinct branch masks). CHI-GATED:
    // below the partner-hash crossover the O(chi^2) scan beats the hash (its map build + string
    // keying have fixed per-entry overhead — the small-chi per-shot path); above it the O(chi) hash
    // wins (large-chi cascades). BOTH accumulate matched pairs in i-ascending order, so `expt` is
    // bit-for-bit identical between the two AND to the pre-optimization scan.
    if (x < partner_hash_chi_min()) {
        for (int i = 0; i < x; ++i) {
            const uint64_t* si = synd_pk.data() + (size_t)i * W;
            for (int w = 0; w < W; ++w) tgt[w] = si[w] ^ Q.x[w];      // partner signature of i
            for (int j = 0; j < x; ++j) {
                const uint64_t* sj = synd_pk.data() + (size_t)j * W;
                bool match = true;
                for (int w = 0; w < W; ++w) if (sj[w] != tgt[w]) { match = false; break; }
                if (!match) continue;
                int par = 0;
                for (int w = 0; w < W; ++w) par += __builtin_popcountll(Q.z[w] & sj[w]);
                expt += std::conj(branches[i].c) * branches[j].c
                        * ipow[(Q.phase + 2 * (par & 1)) & 3];
            }
        }
    } else {
        static thread_local std::unordered_map<std::string, int> sig_idx;
        sig_idx.clear();
        sig_idx.reserve((size_t)x * 2);
        for (int j = 0; j < x; ++j) {
            const uint64_t* sj = synd_pk.data() + (size_t)j * W;
            // INVARIANT (defended): signatures are distinct on a canonical state (distinct sigma,
            // enforced by verify_invariants), so each key is unique and the hash sums exactly the
            // pairs the scan would. A duplicate would make the hash drop terms the scan keeps.
            auto ins = sig_idx.emplace(
                std::string(reinterpret_cast<const char*>(sj), (size_t)W * 8), j);
            assert(ins.second && "born hash: duplicate branch signature (non-canonical state)");
            (void)ins;
        }
        for (int i = 0; i < x; ++i) {
            const uint64_t* si = synd_pk.data() + (size_t)i * W;
            for (int w = 0; w < W; ++w) tgt[w] = si[w] ^ Q.x[w];      // partner signature of i
            auto it = sig_idx.find(
                std::string(reinterpret_cast<const char*>(tgt.data()), (size_t)W * 8));
            if (it == sig_idx.end()) continue;                        // no partner: term vanishes
            const int j = it->second;
            const uint64_t* sj = synd_pk.data() + (size_t)j * W;
            int par = 0;
            for (int w = 0; w < W; ++w) par += __builtin_popcountll(Q.z[w] & sj[w]);
            expt += std::conj(branches[i].c) * branches[j].c
                    * ipow[(Q.phase + 2 * (par & 1)) & 3];
        }
    }
    double pp = 0.5 * (1.0 + std::real(expt));
    if (pp < 0) pp = 0; if (pp > 1) pp = 1;
    return {pp, 1.0 - pp};
}

// ── Overlap-free single-state helpers (docs §9) ──────────────────────────────────────────────
namespace {
// Sign of a stabiliser Pauli g on a stabilizer state |ψ⟩: returns 0 if g|ψ⟩=+|ψ⟩, 1 if =−|ψ⟩.
// g must stabilise |ψ⟩ (±1). Read overlap-free: g|ψ⟩ has the SAME support as |ψ⟩, so the ratio
// of a single shared computational-basis amplitude (at idx in support) is ±1. NO inner_product.
[[maybe_unused]] int gen_sign(const AffineState& psi, const Pauli& g) {
    // On-support index = ψ's affine offset b (y=0). Read via amplitude_at_bits (n>64 safe).
    AffineState gp = psi;
    g.apply_to(gp);                                  // |ψ'⟩ = g|ψ⟩ = ±|ψ⟩
    std::complex<double> a0 = psi.amplitude_at_bits(psi.b).to_complex();
    std::complex<double> a1 = gp.amplitude_at_bits(psi.b).to_complex();
    // a1/a0 = ±1 (g stabilises ψ). Robust to which idx (any shared-support index works).
    return (std::real(a1 * std::conj(a0)) < 0) ? 1 : 0;
}
// Unit-modulus gauge γ with |built⟩ = γ·|target⟩ (same support, same stabiliser signs). Computed
// overlap-free from one shared amplitude. NO inner_product.
[[maybe_unused]] std::complex<double> gauge_ratio(const AffineState& target, const AffineState& built) {
    // On-support index = built's affine offset b (y=0). Read via amplitude_at_bits (n>64 safe).
    std::complex<double> at = target.amplitude_at_bits(built.b).to_complex();
    std::complex<double> ab = built.amplitude_at_bits(built.b).to_complex();
    if (std::abs(ab) < 1e-300) return std::complex<double>(1, 0);
    std::complex<double> g = ab / at;                // built = g·target
    double mg = std::abs(g);
    return (mg > 1e-300) ? g / mg : std::complex<double>(1, 0);
}

// pmul_into / conj_cx_inplace / reset_pauli_inplace / dual_image_rows_scan are shared kernels from
// qeccore/pauli_kernels.hpp. transpose64/transpose_bits live in qeccore/gf2_gauss.hpp (shared with
// the PreparedAffine prepare/overlap fast paths).
using qeccore::transpose64;
using qeccore::transpose_bits;

// BATCHED dual-image, dispatching flavour. The image's x/z bits are the target's anticommute
// pattern against the Zrow/Xrow tableau halves — a GF(2) matrix-vector product — so for cnt >= 3
// the four tableau bit-matrices are TRANSPOSED once (blockwise 64x64) and each target's whole
// pattern is read as |support(T)| word-XORs of transposed columns instead of 2N branchy row scans.
// Only the per-hit phase chain (the order-dependent part) still touches the original rows, walking
// the pattern's set bits in ascending b — the SAME factor order and accumulation as the row scan,
// so the outputs are byte-identical (cross-checked under -DCB_BATCH_XCHECK over the full fuzz).
// For cnt < 3 the transpose isn't amortized and the row-scan flavour runs directly.
void dual_image_rows_batch(const std::vector<Pauli>& Xrow, const std::vector<Pauli>& Zrow,
                           int N, const Pauli* const* T, int cnt, Pauli* out) {
    if (cnt < 3) { dual_image_rows_scan(Xrow, Zrow, N, T, cnt, out); return; }
    const int W = (N + 63) / 64;
    // Gather each tableau half (row b = Zrow[b].z, Zrow[b].x, Xrow[b].z, Xrow[b].x) and transpose:
    // row j of the transpose = column j over b.
    thread_local std::vector<uint64_t> G, ZzT, ZxT, XzT, XxT, pat, azt;
    G.resize((size_t)N * W);
    auto gather_T = [&](std::vector<uint64_t> Pauli::* member, const std::vector<Pauli>& rows,
                        std::vector<uint64_t>& MT) {
        for (int b = 0; b < N; ++b) {
            const uint64_t* src = (rows[b].*member).data();
            for (int w = 0; w < W; ++w) G[(size_t)b * W + w] = src[w];
        }
        MT.resize((size_t)N * W);
        transpose_bits(G.data(), N, N, MT.data());
    };
    gather_T(&Pauli::z, Zrow, ZzT);
    gather_T(&Pauli::x, Zrow, ZxT);
    gather_T(&Pauli::z, Xrow, XzT);
    gather_T(&Pauli::x, Xrow, XxT);
    pat.resize(W);
    azt.resize(W);
    // pattern bit b = popcount(Tx & row_b(zhalf)) ^ popcount(Tz & row_b(xhalf)) — accumulated as
    // column XORs: ⊕_{j∈supp(Tx)} colT_j(zhalf) ⊕ ⊕_{j∈supp(Tz)} colT_j(xhalf).
    auto pattern = [&](const Pauli& Tt, const std::vector<uint64_t>& zT,
                       const std::vector<uint64_t>& xT) {
        for (int w = 0; w < W; ++w) pat[w] = 0;
        for (int w = 0; w < W; ++w) {
            const int base = w << 6;
            uint64_t bits = Tt.x[w];
            while (bits) {
                const uint64_t* row = zT.data() + (size_t)(base + __builtin_ctzll(bits)) * W;
                bits &= bits - 1;
                for (int ww = 0; ww < W; ++ww) pat[ww] ^= row[ww];
            }
            bits = Tt.z[w];
            while (bits) {
                const uint64_t* row = xT.data() + (size_t)(base + __builtin_ctzll(bits)) * W;
                bits &= bits - 1;
                for (int ww = 0; ww < W; ++ww) pat[ww] ^= row[ww];
            }
        }
    };
    for (int t = 0; t < cnt; ++t) {
        reset_pauli_inplace(out[t], N);
        for (int w = 0; w < W; ++w) azt[w] = 0;
        int acc_phase = 0;
        // Zrow pass: pattern bits = image X-bits; phase factors are the matching Xrow rows.
        pattern(*T[t], ZzT, ZxT);
        for (int w = 0; w < W; ++w) out[t].x[w] = pat[w];
        for (int w = 0; w < W; ++w) {
            uint64_t bits = pat[w];
            const int base = w << 6;
            while (bits) {
                const Pauli& P = Xrow[base + __builtin_ctzll(bits)];
                bits &= bits - 1;
                int sign = 0; for (int ww = 0; ww < W; ++ww) sign += __builtin_popcountll(azt[ww] & P.x[ww]);
                acc_phase = (acc_phase + P.phase + 2 * (sign & 1)) & 3;
                for (int ww = 0; ww < W; ++ww) azt[ww] ^= P.z[ww];
            }
        }
        // Xrow pass: pattern bits = image Z-bits; phase factors are the matching Zrow rows.
        pattern(*T[t], XzT, XxT);
        for (int w = 0; w < W; ++w) out[t].z[w] = pat[w];
        for (int w = 0; w < W; ++w) {
            uint64_t bits = pat[w];
            const int base = w << 6;
            while (bits) {
                const Pauli& P = Zrow[base + __builtin_ctzll(bits)];
                bits &= bits - 1;
                int sign = 0; for (int ww = 0; ww < W; ++ww) sign += __builtin_popcountll(azt[ww] & P.x[ww]);
                acc_phase = (acc_phase + P.phase + 2 * (sign & 1)) & 3;
                for (int ww = 0; ww < W; ++ww) azt[ww] ^= P.z[ww];
            }
        }
        out[t].phase = ((T[t]->phase - acc_phase) % 4 + 4) % 4;
    }
#ifdef CB_BATCH_XCHECK
    {   // TEMP differential check (build-flag only): row-scan reference must match bit-for-bit.
        std::vector<Pauli> ref(cnt, Pauli(N));
        dual_image_rows_scan(Xrow, Zrow, N, T, cnt, ref.data());
        for (int t = 0; t < cnt; ++t)
            if (ref[t].x != out[t].x || ref[t].z != out[t].z || ref[t].phase != out[t].phase) {
                fprintf(stderr, "CB_BATCH_XCHECK: mismatch at target %d\n", t);
                abort();
            }
    }
#endif
}

// As above but the word `Wf` is ALREADY in frame coordinates (= U† W U). E(W) = ⟨φ0|W|φ0⟩ on the
// anchor with generator signs eps. O(n), NO dual_image. (Wf = i^{ph} X^{xf} Z^{zf}; X-part ⇒ E = 0;
// else W = i^{ph}∏_{a:zf[a]} g_a and E = i^{ph}·∏_{a:zf[a]}(−1)^{eps[a]}.)
// eps is WORD-PACKED by the caller: the sign product is the parity popcount of Wf.z & eps_pk (an
// even number of e=-e flips cancels exactly, so the parity fold is bitwise identical to the old
// sequential per-bit flips). i^{ph} comes from a table computed ONCE via the same
// std::pow(Iunit, (double)ph) the old per-call code evaluated — same inputs, same bit-exact values
// (an exact {1,i,-1,-i} table would NOT be byte-exact: std::pow leaves ~1e-17 residues that flow
// into the branch coefficients).
std::complex<double> frame_diag_E_pk(const uint64_t* eps_pk, const Pauli& Wf, int W) {
    for (int w = 0; w < W; ++w) if (Wf.x[w]) return std::complex<double>(0, 0);   // X-part ⇒ E = 0
    static const std::complex<double> Iunit(0, 1);
    static const std::complex<double> ipow[4] = {
        std::pow(Iunit, 0.0), std::pow(Iunit, 1.0), std::pow(Iunit, 2.0), std::pow(Iunit, 3.0)};
    int s = 0;
    for (int w = 0; w < W; ++w) s += __builtin_popcountll(Wf.z[w] & eps_pk[w]);
    const std::complex<double> e = ipow[Wf.phase & 3];
    return (s & 1) ? -e : e;
}
}  // namespace

// ── Case B: anticommuting single-qubit measurement (theory §5, closed form §9) ────────────────
// Step 1 rotates the anticommuting generators onto a pivot p (frame + full-n sign vectors `synd`).
// Step 2 computes the Born probability in CLOSED FORM (Eq. born): pairs of branches differing in
// the pivot bit, matrix element μ(σ)=i^{ph}(−1)^{q_z·synd}. Step 3 forms the surviving COEFFICIENTS
// in CLOSED FORM (Eq. mergecoef / promotion) via the PARTNER LOOKUP (w1=w0⊕e_p present ⇒ merge,
// else promotion) — NO pairwise state overlaps. Step 4 hands the surviving groups (original rep
// destabiliser words + closed-form coeffs) to recompute_from_words, which reads the post-structure
// from single-state projections + sign reads (also overlap-free).
int CanonicalStabSum::measure_single_anticommuting(int pauli, int q, const Pauli& P,
                                                   const std::vector<int>& A, double u) {
    const int N = n();
    const int x = chi();

    // Whether the inverse tableau is valid on entry (before ANY frame edit). When true we maintain
    // it INCREMENTALLY through the in-place frame edits below (Step-1 CX rotation W and the Step-4
    // collapse patch C), so the post-measurement frame keeps a valid dual and the next single-qubit
    // conjugate_single stays on the O(n) cached path instead of the O(n²) dual_image fallback. When
    // false we leave it stale (forcing an O(n³) rebuild here would be a net loss). U_new = U_old·V,
    // V = W·C, so Xinv_new[b] = V† Xinv_old[b] V (conjugate every inverse row by each factor of V).
    const bool dual_was_valid = U.dual_valid();

    // Absolute sign of g_a on branch i:  synd[i][a] = eps[a] ⊕ (free-pattern bit, if a∈free).
    // (Eq. 5: g_a|φ_i⟩ = (-1)^{eps[a]}(-1)^{[a∈free]σ_i[a]}|φ_i⟩.)
    // The per-branch sign vectors are WORD-PACKED (bit a of row i = synd[i][a], W words per row,
    // flat x×W layout): each row is eps's packed words with the branch's free-pattern bits flipped —
    // the same bits the old byte matrix held, but the Born pairing / partner lookup / qz_par below
    // become word compares and popcounts instead of O(N) byte scans. Per-thread reusable scratch
    // (one measurement in flight per thread; storage persists across calls).
    const int W = (N + 63) / 64;
    static thread_local std::vector<int> free_pos; free_pos.assign(N, -1);   // gen a → free coord, or -1
    for (int d = 0; d < (int)free.size(); ++d) free_pos[free[d]] = d;
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    static thread_local std::vector<uint64_t> synd_pk; synd_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = synd_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) row[w] = eps_pk[w];
        for (int d = 0; d < (int)free.size(); ++d)
            if (branches[i].sigma[d]) row[free[d] >> 6] ^= 1ull << (free[d] & 63);
    }
    auto synd_bit = [&](int i, int a) -> uint64_t {
        return (synd_pk[(size_t)i * W + (a >> 6)] >> (a & 63)) & 1ull;
    };

    // Capture the ORIGINAL frame's FORWARD rows + anchor so recompute_from_words can materialise the
    // post-state's representatives from the ORIGINAL destabiliser words (docs §9 remark iii) before any
    // frame edit. We copy ONLY the forward rows (Xrow/Zrow = 2n Paulis) — NOT the full CliffordTableau
    // (which would also deep-copy the 2n-Pauli inverse tableau, doubling the copy + its allocator
    // churn). Every origU read here is dual_image (forward rows) or Xrow[a] access; Q is taken from the
    // pristine U below before any mutation, so the inverse tableau is never needed for the old frame.
    //
    // PERF: the dominant cost here was the 2n fresh per-Pauli heap allocations (each Pauli owns two
    // std::vector<uint64_t>). We reuse capacity-retaining thread_local scratch vectors: assigning into a
    // vector<Pauli> that already holds n right-sized Paulis copies the packed words in place (no
    // allocator churn), cutting the forward-row capture ~16µs→~2.6µs at n=128 (measured) with NO change
    // to any downstream interface (oXrow/oZrow stay vector<Pauli> as dual_image_rows / apply_to expect).
    // Single-threaded by construction (one measurement in flight per thread); thread_local keeps it safe
    // under multi-threaded use and frees the scratch only at thread exit.
    // synd/eps GF(2) sign-vector + free_pos build so far: GF(2) frame-syndrome structure ⇒ TABLEAU.
    static thread_local std::vector<Pauli> oXrow_scratch, oZrow_scratch;
    oXrow_scratch = U.Xrow;                        // capacity-retaining copy (no per-Pauli realloc)
    oZrow_scratch = U.Zrow;
    std::vector<Pauli>& oXrow = oXrow_scratch;     // original destabilisers d_a = U X_a U†
    std::vector<Pauli>& oZrow = oZrow_scratch;     // original generators   g_a = U Z_a U†
    static thread_local std::vector<uint8_t> old_eps; old_eps = eps;   // ORIGINAL generator signs
    auto orig_anchor_ptr = anchor->clone();       // |φ0⟩ (original)
    AffineState& orig_anchor = static_cast<AffineState&>(*orig_anchor_ptr);
    // ORIGINAL destabiliser word per branch (which d_a applied to |φ0⟩): branch i uses
    // d_{free[d]} iff branches[i].sigma[d]. Captured BEFORE Step-1 rotation mutates `synd`.
    static thread_local std::vector<uint64_t> orig_word; orig_word.assign((size_t)x * W, 0);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = orig_word.data() + (size_t)i * W;
        for (int d = 0; d < (int)free.size(); ++d)
            if (branches[i].sigma[d]) row[free[d] >> 6] |= 1ull << (free[d] & 63);
    }
    static thread_local std::vector<std::complex<double>> c0; c0.resize(x);   // orig branch coefficients
    for (int i = 0; i < x; ++i) c0[i] = branches[i].c;

    // orig_word GF(2) + c0 branch-coeff copy: classified TABLEAU (the c0 copy is a tiny memcpy folded
    // into the GF(2) word build; both are dominated by the orig_word bit construction).
    // ── Step 1: rotate other anticommuters onto a pivot p∈A. Prefer a FREE pivot (so a free
    // coordinate is consumed); among those, prefer one that actually VARIES across branches.
    int p = -1;
    for (int a : A) if (free_pos[a] >= 0) {       // varying free generator
        bool vary = false;
        for (int i = 1; i < x; ++i) if (synd_bit(i, a) != synd_bit(0, a)) { vary = true; break; }
        if (vary) { p = a; break; }
    }
    if (p < 0) for (int a : A) if (free_pos[a] >= 0) { p = a; break; }  // any free
    if (p < 0) p = A[0];                                                // else any (promotion)

    // The Step-1 rotation right-composes the frame by the abstract Clifford W = ∏_{a∈A\p} CX(p,a),
    // since CX(p,a) sends Z_a→Z_aZ_p and X_p→X_pX_a — exactly the row ops below. We therefore get the
    // post-rotation Qr = U_new† P U_new = W†(U_old† P U_old)W = W† Q W in CLOSED FORM (O(|A|·n)) from
    // the already-computed original Q, AVOIDING an O(n³) reset_dual here (the inverse tableau is left
    // STALE by the in-place frame patch and rebuilt LAZILY only when the next gate / multi-qubit
    // conjugate needs it — a consecutive single-qubit measurement reads Q via conjugate_single).
    Pauli Q = U.conjugate_single(pauli, q);      // = U_old† P U_old (U still pristine — dual valid, O(n))
    // The Step-1 rotation right-composes the frame by W = ∏_{a∈A\p} CX(p,a) (the CX(p,a) row ops on U
    // below). Qr = U_new† P U_new = W† Q W. All these CX share control p ⇒ commute ⇒ W†=W; apply each
    // CX(p,a) conjugation DIRECTLY to a copy of Q (O(|A|·n)) — NO fresh CliffordTableau + inverse tableau.
    Pauli Qr = Q;
    {   // synd: bit a ^= bit p for every a∈A\p — ONE mask XOR per branch row with the pivot bit
        // set (mask excludes p ⇒ p-bit stable; identical bits to the per-a updates).
        static thread_local std::vector<uint64_t> amask; amask.assign(W, 0);
        for (int a : A) if (a != p) amask[a >> 6] |= 1ull << (a & 63);
        for (int i = 0; i < x; ++i) {
            uint64_t* row = synd_pk.data() + (size_t)i * W;
            if ((row[p >> 6] >> (p & 63)) & 1ull)
                for (int w = 0; w < W; ++w) row[w] ^= amask[w];
        }
    }
    for (int a : A) {
        if (a == p) continue;
        pmul_into(U.Zrow[a], U.Zrow[p]);   // g_a ← g_a·g_p   (p≠a ⇒ no aliasing; g_p stable in loop)
        pmul_into(U.Xrow[p], U.Xrow[a]);   // dual d_p ← d_p·d_a
        conj_cx_inplace(Qr, p, a);                           // Qr ← CX(p,a) Qr CX(p,a)
        U.right_cx(p, a);                  // W-factor: maintain the inverse tableau (no-op if stale)
    }
    // Post-rotation Qr = i^{ph} X_p Z^{q_z} (anticommutes with g_p alone). Matrix element
    // μ(σ_j) = ⟨φ_i|Qr|φ_j⟩ = i^{ph}(−1)^{q_z·synd[j]}·[i = pivot-partner(j)] (X_p=d_p flips the pivot
    // bit); Iph is the FULL complex i^{ph} (Y-pivots have odd ph). (Eq. matelt.)
    static const std::complex<double> Iunit(0, 1);
    std::complex<double> Iph = std::pow(Iunit, (double)(Qr.phase & 3));
    auto qz_par = [&](int j) {   // parity(Qr.z & synd[j]) — both packed, trailing bits ≥ N zero in both
        const uint64_t* row = synd_pk.data() + (size_t)j * W;
        int s = 0;
        for (int w = 0; w < W; ++w) s += __builtin_popcountll(Qr.z[w] & row[w]);
        return s & 1;
    };
    // μ(σ_j): the matrix element ⟨partner|Q|φ_j⟩ (the coefficient with which Q maps φ_j to its partner).
    auto mu = [&](int j) -> std::complex<double> { return Iph * (double)(qz_par(j) ? -1 : 1); };

    // post-syndrome key (pivot column zeroed) + pivot bit, per branch (packed, flat x×W).
    static thread_local std::vector<uint64_t> post_pk; post_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        const uint64_t* src = synd_pk.data() + (size_t)i * W;
        uint64_t* dst = post_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) dst[w] = src[w];
        dst[p >> 6] &= ~(1ull << (p & 63));
    }
    auto post_eq = [&](int i, int j) {
        const uint64_t* a = post_pk.data() + (size_t)i * W;
        const uint64_t* b = post_pk.data() + (size_t)j * W;
        for (int w = 0; w < W; ++w) if (a[w] != b[w]) return false;
        return true;
    };

    // ── Step 2: Born probability (Eq. born). ⟨Q⟩ = Re Σ_j conj(c_{partner(j)})·c_j·μ(j) over pairs
    // that differ exactly in the pivot bit (same post-key). p_± = ½(1±⟨Q⟩).
    // Branch i's partner is the UNIQUE j with the same post-key (pivot column zeroed) and the
    // OPPOSITE pivot bit. CHI-GATED (see partner_hash_chi_min): small chi keeps the O(chi^2) scan
    // (the per-shot small-chi measurement path — hash map build loses there); large chi uses the
    // O(chi) (post-key, pivot-bit)->index hash. BOTH keep the outer i ascending with exactly one
    // partner term each, so `expt` is bit-for-bit identical between them and to the original scan.
    std::complex<double> expt(0, 0);
    if (x < partner_hash_chi_min()) {
        for (int i = 0; i < x; ++i)
            for (int j = 0; j < x; ++j) {
                if (!post_eq(i, j)) continue;                      // same group
                if (synd_bit(i, p) == synd_bit(j, p)) continue;    // must differ in the pivot bit
                expt += std::conj(branches[i].c) * branches[j].c * mu(j);
            }
    } else {
        static thread_local std::unordered_map<std::string, int> post_idx;
        post_idx.clear();
        post_idx.reserve((size_t)x * 2);
        for (int j = 0; j < x; ++j) {
            std::string key(reinterpret_cast<const char*>(post_pk.data() + (size_t)j * W),
                            (size_t)W * 8);
            key.push_back((char)synd_bit(j, p));           // distinguish the two pivot-bit members
            // INVARIANT (defended): on a canonical state each (post-key, pivot-bit) keys at most one
            // branch, so the hash sums exactly the pairs the scan keeps (a duplicate would drop one).
            auto ins = post_idx.emplace(std::move(key), j);
            assert(ins.second && "measure Step2 hash: duplicate (post-key,pivot) (non-canonical state)");
            (void)ins;
        }
        for (int i = 0; i < x; ++i) {
            std::string key(reinterpret_cast<const char*>(post_pk.data() + (size_t)i * W),
                            (size_t)W * 8);
            key.push_back((char)(1 - synd_bit(i, p)));     // the partner has the opposite pivot bit
            auto it = post_idx.find(key);
            if (it == post_idx.end()) continue;            // no partner: branch i is unpaired here
            const int j = it->second;
            expt += std::conj(branches[i].c) * branches[j].c * mu(j);
        }
    }
    double pp = 0.5 * (1.0 + std::real(expt));
    if (pp < 0) pp = 0; if (pp > 1) pp = 1;
    int m = (u < pp) ? +1 : -1;
    double pm = (m == +1) ? pp : (1.0 - pp);
    double inv = (pm > 0) ? 1.0 / std::sqrt(pm) : 0.0;

    // ── Step 3: surviving coefficients in CLOSED FORM (Eqs. mergecoef / promotion) via the §9
    // PARTNER LOOKUP (no overlaps). In POST-ROTATION coordinates Q anticommutes with the pivot
    // ALONE, so branch i's partner is the branch whose post-rotation syndrome differs from i's by
    // exactly e_p — i.e. same post-key (pivot column zeroed) and opposite pivot bit. (This is the
    // SAME pairing the Born loop above uses.)
    //   * partner present ⇒ MERGE: the pivot-0 member w0 represents the pair, ν =
    //     (1/√2)(c_{w0} + m·conj(μ(w0))·c_{w1})·(1/√p_m);
    //   * partner absent ⇒ PROMOTION: the word survives orthogonally, ν = (1/√2)·c_w·(1/√p_m).
    // Key the partner lookup on the FULL packed post-syndrome (all W words — a single-uint64 key
    // would alias for N>64). The old std::map<vector<uint8_t>,…> is replaced by an index sort over
    // the packed keys in BYTE-LEXICOGRAPHIC order — generator a ascending, i.e. the LOWEST differing
    // bit decides and the key with that bit 0 sorts first — so the survivor groups are visited in
    // EXACTLY the order the map iteration produced (keys are unique per run: two branches share a key
    // only as pivot-partners, with opposite pivot bits, so ties never reach the comparator's `false`).
    static thread_local std::vector<int> order; order.resize(x);
    for (int i = 0; i < x; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        const uint64_t* a = post_pk.data() + (size_t)i * W;
        const uint64_t* b = post_pk.data() + (size_t)j * W;
        for (int w = 0; w < W; ++w) if (a[w] != b[w]) {
            const int t = __builtin_ctzll(a[w] ^ b[w]);        // lowest differing generator index
            return ((a[w] >> t) & 1ull) == 0;
        }
        return false;
    });
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    // Per-thread reusable survivor pool: `survivors` keeps its SurvGroup slots (and each slot's `word`
    // vector) alive across collapses; we append by reusing slot `nsurv` (assign into its existing word
    // buffer — capacity-retaining) instead of constructing a fresh std::vector<uint64_t> per survivor.
    // surv_rep is a trivial vector (clear() keeps capacity). At the end resize(nsurv) so groups.size()
    // is the logical survivor count (kept slots' word storage is retained; only the tail is shed).
    static thread_local std::vector<SurvGroup> survivors;
    static thread_local std::vector<int> surv_rep;   // representative branch index per survivor (for fsynd)
    surv_rep.clear();
    size_t nsurv = 0;
    auto push_surv = [&](const uint64_t* row, std::complex<double> nu, int rep) {
        if (nsurv >= survivors.size()) survivors.emplace_back();
        SurvGroup& sg = survivors[nsurv];
        sg.word.assign(row, row + W);                 // reuse existing capacity (no fresh alloc)
        sg.coeff = nu;
        surv_rep.push_back(rep);
        ++nsurv;
    };
    for (int beg = 0; beg < x; ) {
        int end = beg + 1;
        while (end < x && post_eq(order[beg], order[end])) ++end;
        int w0 = -1, w1 = -1;
        for (int t = beg; t < end; ++t) {
            const int i = order[t];
            if (synd_bit(i, p)) w1 = i; else w0 = i;
        }
        beg = end;
        if (w0 >= 0 && w1 >= 0) {
            // MERGE pair: w0 has pivot syndrome bit 0, w1 has 1 (μ(w0) is the matrix element on w0).
            std::complex<double> nu = inv_sqrt2 *
                (c0[w0] + (double)m * std::conj(mu(w0)) * c0[w1]) * inv;
            if (std::abs(nu) < 1e-12) continue;
            push_surv(orig_word.data() + (size_t)w0 * W, nu, w0);
        } else {
            // PROMOTION (partner absent): the single member survives as its own orthogonal branch.
            int w = (w0 >= 0) ? w0 : w1;
            std::complex<double> nu = inv_sqrt2 * c0[w] * inv;
            if (std::abs(nu) < 1e-12) continue;
            push_surv(orig_word.data() + (size_t)w * W, nu, w);
        }
    }
    survivors.resize(nsurv);   // logical survivor count (kept slots retain word capacity)

    // ── Step 4: PATCH the frame in place (no re-extraction). Post-measurement the pivot generator
    // g_p becomes the LAB operator ±P (m=+1: +P; m=−1: −P), and its destabiliser d_p becomes the OLD
    // g_p (= current U.Zrow[p], which anticommutes with P and commutes with every other generator —
    // it was a generator — and with every other destabiliser). All other g_a (a≠p) are unchanged.
    //
    // The rotated destabilisers d_a (a≠p) may now anticommute with the NEW pivot generator ±P (they
    // were built dual to the OLD g_p). Fix each such d_a by d_a ← d_a·d_p (d_p = old g_p): old g_p
    // also anticommutes with P so the product commutes with ±P, and since old g_p commutes with every
    // OTHER generator/destabiliser the rest of the symplectic relations are preserved. O(|broken|·n).
    Pauli newpiv = P;
    if (m == -1) newpiv.phase = (newpiv.phase + 2) & 3;
    Pauli new_dp = U.Zrow[p];         // d_p = old generator g_p (anticommutes with ±P)
    U.Zrow[p] = newpiv;               // new generator g_p = ±P (lab operator, docs §9 remark i)
    // d_a ← d_a · d_p for each d_a anticommuting with ±P. Inline the symplectic product (no cross-TU
    // anticommute_bit call) and fold via pmul_into (no per-row Pauli::multiply allocation).
    {
        const int Wp = (N + 63) / 64;
        const uint64_t* npx = newpiv.x.data();
        const uint64_t* npz = newpiv.z.data();
        for (int a = 0; a < N; ++a) {
            if (a == p) continue;
            const Pauli& Xa = U.Xrow[a];
            int ac = 0;
            for (int w = 0; w < Wp; ++w) ac += __builtin_popcountll(Xa.x[w] & npz[w]) + __builtin_popcountll(Xa.z[w] & npx[w]);
            if (ac & 1) pmul_into(U.Xrow[a], new_dp);
        }
    }
    U.Xrow[p] = new_dp;

    // ── Step-4 inverse-tableau maintenance (the C factor of V = W·C). ──────────────────────────
    // The forward C-patch above is, in abstract Z_a/X_a generator language, conjugation by
    //     C = E · (∏_{b∈q_z\p} CZ(p,b)) · H_p,
    // where q_z = supp(Qr.z) is the Z-support of Qr = i^{ph} X_p Z^{q_z} (post-Step-1), and E is a
    // single-qubit signed-Pauli correction on the pivot p that supplies (i) the m·i^{ph} scalar and
    // (ii) the extra Z_p factor when p∈q_z (Y-pivot). Verified against reset_dual below:
    //   X_p →_C Z_p,  Z_p →_C ±i^{ph} X_p ∏_{b∈q_z} Z_b,  Z_a →_C Z_a (a≠p),
    //   X_a →_C X_a Z_p^{[a∈q_z]} (a≠p)  — exactly the row edits performed just above.
    // Inverse rows: Xinv_new[b] = C† Xinv_old[b] C = conjugate every inverse row by each factor of C
    // (innermost H_p first via right_h; then each CZ; then the pivot phase correction E).
    if (dual_was_valid) {
        // C = E·(∏_{b∈q_z\p} CZ(p,b))·H_p ⇒ C† M C peels outermost-first: conjugate by E first, then
        // ∏CZ, then H_p (the innermost forward factor is conjugated LAST). For the inverse rows we
        // apply, in this order, the ∏CZ patch, then the H_p patch — and the E correction is folded in
        // last via right_pivot_E (it commutes onto the pivot column).  [order pinned by the oracle]
        for (int b = 0; b < N; ++b)
            if (b != p && Qr.zbit(b)) U.right_cz(p, b);
        U.right_h(p);
        // Pivot phase correction E: supplies the m·i^{ph} scalar (+ √X-type Y-pivot fix). Conjugates
        // every inverse row by E† (E acts on p only, fixing X_p). Calibrated against reset_dual.
        U.right_pivot_E(p, m, Qr.phase & 3, Qr.zbit(p));
    }
    // NB: when the dual was STALE on entry it stays stale here (rebuilt lazily). When it was valid
    // it is now maintained for frame U_2; recompute_from_words continues the maintenance through the
    // re-basis edit and keeps it valid at the end.

    // Closed-form generator signs on each survivor's projected representative: g_a (a≠p) is unchanged
    // by the projection (commutes with P), so its sign is synd[rep][a]; the pivot generator ±P fixes
    // the projected ray with eigenvalue +1 (the m-sign is baked into newpiv), so column p is 0.
    // Per-thread reusable fsynd scratch (capacity-retaining). Every used word is fully overwritten
    // below, so resize without re-zeroing is byte-identical to the old (size*W, 0) construction.
    static thread_local std::vector<uint64_t> fsynd;
    fsynd.resize((size_t)survivors.size() * W);
    for (int g = 0; g < (int)survivors.size(); ++g) {
        const uint64_t* src = synd_pk.data() + (size_t)surv_rep[g] * W;
        uint64_t* dst = fsynd.data() + (size_t)g * W;
        for (int w = 0; w < W; ++w) dst[w] = src[w];
        dst[p >> 6] &= ~(1ull << (p & 63));
    }

    // ── Step 5: read the post-measurement structure from the surviving groups (overlap-free).
    // frame_patched=true: the pivot patch above left U's inverse tableau stale; recompute owns the
    // single reset_dual.
    recompute_from_words(oXrow, oZrow, orig_anchor, old_eps, survivors, fsynd,
                         pauli, q, m, /*frame_patched=*/true, /*dual_maintained=*/dual_was_valid);

#ifndef NDEBUG
    // Cross-check against the overlap-based (O(χ²)) oracle: same statevector. (Debug only.)
    // to_statevector() is an O(2^n) oracle, valid only for small n; the n>64 path can't be
    // materialised (1<<n would be UB) and is correctness-checked via overlaps in the test itself.
    if (N <= 20) {
        CanonicalStabSum dbg(N);
        dbg.anchor = std::unique_ptr<AffineState>(static_cast<AffineState*>(orig_anchor.clone().release()));
        // Reproduce the rays + coeffs the overlap path would have built, then compare.
        std::vector<std::unique_ptr<AffineState>> tgts;
        std::vector<std::complex<double>> coeffs;
        std::vector<std::unique_ptr<AffineState>> projected(x);
        std::vector<ExactPhase> projamp(x);
        for (int j = 0; j < x; ++j) {
            auto st = orig_anchor.clone();
            for (int a = 0; a < N; ++a) if ((orig_word[(size_t)j * W + (a >> 6)] >> (a & 63)) & 1) oXrow[a].apply_to(*st);
            ExactPhase a = static_cast<AffineState&>(*st).pauli_project(pauli, q, m);
            projamp[j] = a;
            if (!a.is_zero) projected[j] = std::unique_ptr<AffineState>(static_cast<AffineState*>(st.release()));
        }
        for (int j = 0; j < x; ++j) {
            if (!projected[j]) continue;
            int ray = -1;
            for (int t = 0; t < (int)tgts.size(); ++t) {
                ExactPhase ov = tgts[t]->inner_product(*projected[j]);
                if (!ov.is_zero && std::abs(std::abs(ov.to_complex()) - 1.0) < 1e-9) { ray = t; break; }
            }
            if (ray < 0) {
                tgts.push_back(std::unique_ptr<AffineState>(static_cast<AffineState*>(projected[j]->clone().release())));
                coeffs.push_back(std::complex<double>(0, 0));
                ray = (int)tgts.size() - 1;
            }
            ExactPhase al = tgts[ray]->inner_product(*projected[j]);
            std::complex<double> phase = al.is_zero ? std::complex<double>(0, 0) : al.to_complex();
            coeffs[ray] += c0[j] * projamp[j].to_complex() * phase;
        }
        for (auto& c : coeffs) c *= inv;
        std::vector<std::unique_ptr<AffineState>> kt; std::vector<std::complex<double>> kc;
        for (size_t t = 0; t < tgts.size(); ++t) if (std::abs(coeffs[t]) >= 1e-12) {
            kt.push_back(std::move(tgts[t])); kc.push_back(coeffs[t]);
        }
        dbg.rebuild_from_rays(std::move(kt), std::move(kc));
        auto sv_cf = to_statevector();
        auto sv_ov = dbg.to_statevector();
        double md = 0; for (size_t t = 0; t < sv_cf.size(); ++t) md = std::max(md, std::abs(sv_cf[t] - sv_ov[t]));
        assert(md < 1e-9 && "closed-form Case-B collapse diverged from overlap oracle");
    }
#endif
    return m;
}

// ── INCREMENTAL O(n²) structure recompute (docs §9): read (eps,anchor,free,branches) from the
// surviving groups using the ALREADY-PATCHED post-measurement frame `this->U` (no re-extraction) and
// the closed-form generator signs `fsynd`. Only ONE projection per group is done — and ONLY to fix
// that group's coefficient gauge (the same per-branch projection v1 performs); the structure itself
// (signs, free set, σ) is read algebraically. No stabilizer_generators / from_generators.
void CanonicalStabSum::recompute_from_words(const std::vector<Pauli>& oXrow, const std::vector<Pauli>& oZrow,
                                            const AffineState& orig_anchor,
                                            const std::vector<uint8_t>& old_eps,
                                            const std::vector<SurvGroup>& groups,
                                            std::vector<uint64_t>& fsynd,
                                            int pauli, int q, int m, bool frame_patched,
                                            bool dual_maintained) {
    const int N = n();
    const int xs = (int)groups.size();
    if (xs == 0) { branches.clear(); free.clear(); eps.assign(N, 0); return; }
    // fsynd is word-packed row-major (bit a of row g = fsynd[g*W + (a>>6)] bit (a&63));
    // groups[g].word likewise (W words, bit a = destabiliser d_a applied).
    const int W = (N + 63) / 64;
    auto fbit = [&](int g, int a) -> uint64_t {
        return (fsynd[(size_t)g * W + (a >> 6)] >> (a & 63)) & 1ull;
    };

    // LEAD 2 — ONE projection. Materialise ONLY the representative group-0 ray (the single O(n²)
    // pauli_project); it becomes the new anchor |φ0'⟩. Every other group's projected ray is reachable
    // WITHOUT a projection as proj[g] = V_g·|φ0'⟩, V_g = ∏(post-frame destabilisers distinguishing g
    // from rep 0) — and the per-branch coefficient gauge γ_g is read in CLOSED FORM (LEAD 1, below),
    // so in a release build we never materialise proj[g] for g>0 at all. The bridge rays are built
    // (after the frame patch — they MUST come from the POST-measurement U.Xrow, which commute the new
    // pivot ±P; original-frame bridges anticommute P ~37% and land in the wrong eigenspace) only for
    // the debug cross-checks.
    {
        auto st = orig_anchor.clone();
        for (int w = 0; w < W; ++w) {        // bit-scan ascending = same apply order as the byte loop
            uint64_t bits = groups[0].word[w];
            const int base = w << 6;
            while (bits) { oXrow[base + __builtin_ctzll(bits)].apply_to(*st); bits &= bits - 1; }
        }
        static_cast<AffineState&>(*st).pauli_project(pauli, q, m);     // collapse onto Π_m (unit ray)
        anchor = std::unique_ptr<AffineState>(static_cast<AffineState*>(st.release()));
    }
    // LEAD-2 anchor (clone + destabiliser apply_to + pauli_project) is AFFINE state work. The tiny
    // fbit/W setup before it is folded in here (negligible).

    // The lab pivot Pauli P (operator measured) — used by the LEAD-1 closed-form gauge below.
    Pauli labP = single_pauli(pauli, q, N);

    // Re-base DEPENDENT distinguishing generators onto fixed ones so the varying syndrome lives on an
    // INDEPENDENT generator set (Minimality precondition for canonicalise). Greedy column-echelon of
    // the syndrome-difference matrix; for each VARYING g_a that is a GF(2) combination of the chosen
    // basis generators, replace g_a ← g_a·∏ g_{basis[k]} (Zrow[a]·=Zrow[basis], Xrow[basis]·=Xrow[a]).
    // The sign array fsynd is updated ALGEBRAICALLY (fsynd[g][a] ^= fsynd[g][basis]) — no re-reads.
    bool frame_edited = false;
    {
        // Columns of the syndrome-difference matrix are packed over the BRANCH index g (gw words):
        // same greedy echelon, same pivot choice (pr = lowest set g, found by ctz scan = the byte
        // version's first-hit scan), same reduction algebra — only the column container changed.
        // fsynd (xs×N) is TRANSPOSED once so column a is a single gw-word row read; the difference
        // column is that row XOR a broadcast of its g=0 bit (g=0 flips to 0 under its own
        // broadcast, exactly fbit(g,a)^fbit(0,a)). The echelon's fsynd sign updates touch ONLY the
        // current column a — already consumed — never a later one, so the transposed copy never
        // goes stale for the reads that remain.
        const int gw = (xs + 63) >> 6;
        static thread_local std::vector<uint64_t> fsyndT;
        fsyndT.resize((size_t)N * gw);
        transpose_bits(fsynd.data(), xs, N, fsyndT.data());
        // Per-collapse echelon scratch is per-thread reusable (capacity-retaining): clear()/resize()
        // keep the backing storage so the re-basis no longer reallocates these every collapse. basis /
        // redcol / comp track a logical size `nb` over POOLED inner vectors (redcol[bi]/comp[bi] are
        // re-used by resize/fill, never destroyed) so the inner difference-column / expansion vectors
        // also keep their capacity across calls. Byte-identical: same algebra, same growth order.
        static thread_local std::vector<uint64_t> onesg; onesg.assign(gw, ~0ull);
        if (xs & 63) onesg[gw - 1] = (1ull << (xs & 63)) - 1;
        static thread_local std::vector<int> basis; basis.clear();
        static thread_local std::vector<std::vector<uint64_t>> redcol;   // reduced diff col per basis (pool)
        static thread_local std::vector<std::vector<uint8_t>> comp;      // expansion in ORIGINAL basis (pool)
        size_t nb = 0;                                        // logical #basis (pool may hold more)
        static thread_local std::vector<uint64_t> col, resid;
        col.resize(gw); resid.resize(gw);
        static thread_local std::vector<uint8_t> expand;     // hoisted: reused per generator
        for (int a = 0; a < N; ++a) {
            const uint64_t* ta = fsyndT.data() + (size_t)a * gw;
            const uint64_t b0 = ta[0] & 1ull;                 // fsynd[0][a]
            for (int w = 0; w < gw; ++w) col[w] = ta[w] ^ (b0 ? onesg[w] : 0ull);
            bool allzero = true; for (int w = 0; w < gw; ++w) if (col[w]) { allzero = false; break; }
            if (allzero) continue;                            // fixed generator — nothing to do
            resid = col;
            expand.assign(nb, 0);
            for (size_t bi = 0; bi < nb; ++bi) {
                const std::vector<uint64_t>& rc = redcol[bi];
                int pr = -1;
                for (int w = 0; w < gw; ++w) if (rc[w]) { pr = (w << 6) + __builtin_ctzll(rc[w]); break; }
                if (pr >= 0 && ((resid[pr >> 6] >> (pr & 63)) & 1ull)) {
                    for (int w = 0; w < gw; ++w) resid[w] ^= rc[w];
                    for (size_t k = 0; k < comp[bi].size(); ++k) expand[k] ^= comp[bi][k];
                }
            }
            bool resid_zero = true; for (int w = 0; w < gw; ++w) if (resid[w]) { resid_zero = false; break; }
            if (resid_zero) {
                Pauli dofa = U.Xrow[a];                       // snapshot d_a before Zrow edits
                for (size_t k = 0; k < expand.size(); ++k) if (expand[k]) {
                    int abase = basis[k];
                    pmul_into(U.Zrow[a], U.Zrow[abase]);
                    pmul_into(U.Xrow[abase], dofa);
                    // Re-basis is the abstract CX(abase,a) (Z_a→Z_aZ_abase, X_abase→X_abaseX_a). Keep
                    // the inverse tableau valid through it when the measurement maintained it (else it
                    // is already stale and right_cx is a no-op). All these CX use the ORIGINAL d_a
                    // (dofa), matching ∏_k CX(abase_k,a) which leaves X_a fixed across k.
                    if (dual_maintained) U.right_cx(abase, a);
                    for (int g = 0; g < xs; ++g) {            // fsynd[g][a] ^= fsynd[g][abase]
                        uint64_t* row = fsynd.data() + (size_t)g * W;
                        row[a >> 6] ^= ((row[abase >> 6] >> (abase & 63)) & 1ull) << (a & 63);
                    }
                }
                frame_edited = true;
            } else {
                // Grow every existing comp[bi] by one trailing zero (the new basis column), exactly as
                // the original `for (auto& v : comp) v.push_back(0)` did — but only over the LOGICAL
                // basis [0,nb) so pooled-but-retired slots aren't touched.
                for (size_t bi = 0; bi < nb; ++bi) comp[bi].push_back(0);
                // Append slot nb (reuse pooled storage when present, else grow the pool by one).
                if (nb >= redcol.size()) { redcol.emplace_back(); comp.emplace_back(); }
                std::vector<uint64_t>& rc = redcol[nb];
                rc = resid;                                   // reduced column (capacity-retaining copy)
                std::vector<uint8_t>& cc = comp[nb];          // expansion: size nb+1, bit nb = 1, rest = expand
                cc.assign(nb + 1, 0); cc[nb] = 1;
                for (size_t k = 0; k < expand.size(); ++k) cc[k] = expand[k];
                basis.push_back(a);
                ++nb;
            }
        }
        // When the caller did NOT maintain the inverse tableau (dual stale on measurement entry), the
        // pivot patch (frame_patched) and any re-basis edit leave it STALE. Do NOT rebuild here (O(n³)):
        // mark invalid and let it rebuild LAZILY on the next gate / multi-qubit conjugate (the collapse
        // below reads only forward rows; a consecutive single-qubit measurement reads Q via
        // conjugate_single — neither needs the dual). When dual_maintained, the inverse tableau was
        // kept VALID incrementally through the pivot patch AND the re-basis CX above, so leave it valid.
        if (!dual_maintained && (frame_edited || frame_patched)) U.invalidate_dual();
    }
    // Re-basis column-echelon (transpose + pmul_into on U.Zrow/U.Xrow + right_cx dual maintenance +
    // fsynd GF(2) algebra) plus the labP construction above are TABLEAU (symplectic frame + dual).

    eps.assign(N, 0);
    for (int a = 0; a < N; ++a) eps[a] = (uint8_t)fbit(0, a); // anchor = group 0 ⇒ its syndrome is eps
    // varies = OR over g≥1 of (row_g ^ row_0): bit a set iff some group's sign differs from group
    // 0's. Ascending set-bit scan = the byte version's ascending-a nfree order.
    static thread_local std::vector<uint64_t> varies; varies.assign(W, 0);
    for (int g = 1; g < xs; ++g) {
        const uint64_t* rg = fsynd.data() + (size_t)g * W;
        for (int w = 0; w < W; ++w) varies[w] |= rg[w] ^ fsynd[w];
    }
    static thread_local std::vector<int> nfree; nfree.clear();   // reusable (capacity-retaining)
    for (int w = 0; w < W; ++w) {
        uint64_t bits = varies[w];
        const int base = w << 6;
        while (bits) { nfree.push_back(base + __builtin_ctzll(bits)); bits &= bits - 1; }
    }
    free = nfree;

    // LEAD-1 gauge in OLD-FRAME COORDINATES (the key to O(n)/branch, χ-flat dual_image count). The
    // gauge E-reads ⟨φ0|·|φ0⟩ live on the ORIGINAL anchor whose frame is origU, so we express EVERY
    // word factor in origU's coordinates ONCE and then only multiply/XOR (O(n)) per branch:
    //   * a destabiliser product ∏ origU.Xrow[a] maps to the pure X-word ∏ X_a (origU†(origU X_a origU†)origU = X_a),
    //     so D_w and D_r become trivial X-bit words with NO dual_image at all;
    //   * the NEW-frame destabilisers U.Xrow[nfree[d]] and the lab pivot P are the only operators not
    //     already in origU coords — conjugate each into origU coords ONCE (|nfree|+1 dual_image calls,
    //     INDEPENDENT of χ) and cache them. E is then read in O(n) by frame_diag_E (no dual_image).
    Pauli Dr_f(N);                                            // rep-0 word in origU coords (pure X)
    for (int w = 0; w < W; ++w) Dr_f.x[w] = groups[0].word[w];   // direct word copy (bits ≥ N zero)
    // All |nfree|+1 conjugations into origU coords in ONE batched pass over the frame rows
    // (outputs identical to per-target dual_image_rows calls; see dual_image_rows_batch).
    // origU coords of each new destabiliser, +P. Per-thread reusable Pauli pool: resize() keeps the
    // existing Pauli objects (and their packed-word vectors) alive so no fresh per-Pauli heap alloc per
    // collapse. Slot nfree.size() holds the lab pivot's image (Pf below) — we KEEP it in the pool
    // (swap into a reusable Pf instead of move+pop_back, so the pool capacity persists across calls).
    static thread_local std::vector<Pauli> Simg;
    Simg.resize(nfree.size() + 1);
    {
        static thread_local std::vector<const Pauli*> tgts;
        tgts.resize(nfree.size() + 1);
        for (int d = 0; d < (int)nfree.size(); ++d) tgts[d] = &U.Xrow[nfree[d]];
        tgts[nfree.size()] = &labP;
        dual_image_rows_batch(oXrow, oZrow, N, tgts.data(), (int)tgts.size(), Simg.data());
    }
    static thread_local Pauli Pf;
    std::swap(Pf, Simg[nfree.size()]);                      // origU coords of the lab pivot P (pool slot reused)
    // Pack old_eps once for the popcount-parity frame_diag_E_pk reads in the gauge loop below.
    static thread_local std::vector<uint64_t> oeps_pk; oeps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (old_eps[a]) oeps_pk[a >> 6] |= 1ull << (a & 63);

    // Reusable per-thread Pauli scratch for the allocation-free gauge below (no fresh Pauli per
    // product over the χ groups). Cleared/resized per use; storage persists across calls.
    static thread_local Pauli gw_Dwf, gw_Sf, gw_M, gw_acc;
    auto gw_reset = [&](Pauli& p) {
        if ((int)p.x.size() != (N + 63) / 64) { p = Pauli(N); }
        else { std::fill(p.x.begin(), p.x.end(), 0); std::fill(p.z.begin(), p.z.end(), 0);
               p.phase = 0; p.n = N; }
    };

    branches.clear();
    branches.reserve(xs);
    for (int g = 0; g < xs; ++g) {
        Branch br; br.sigma.resize(nfree.size());
        for (int d = 0; d < (int)nfree.size(); ++d)
            br.sigma[d] = (uint8_t)(fbit(g, nfree[d]) ^ (uint64_t)eps[nfree[d]]);
        // Group 0 IS the anchor (proj[0]); D_∅|φ0'⟩ = anchor with γ=1 — skip its gauge solve.
        bool all_zero = true; for (uint8_t s : br.sigma) if (s) { all_zero = false; break; }
        if (g == 0 && all_zero) { br.c = groups[g].coeff; branches.push_back(std::move(br)); continue; }

        // LEAD 1 — O(n) closed-form gauge γ_g relating the reconstructed branch S·|φ0'⟩ to the TRUE
        // projected ray Π_m D_w|φ0⟩ (S·|φ0'⟩ = γ_g·true_ray ⇒ br.c = coeff/γ_g, |γ_g|=1):
        //   γ_g = ½·E(D_w† S D_r) + ½·m·E(D_w† S P D_r),
        // D_w = group-g word, D_r = rep-0 word, S = ∏_{d:σ_g[d]} U.Xrow[nfree[d]] (commutes ±P), P =
        // lab pivot, E = ⟨φ0|·|φ0⟩ on the ORIGINAL anchor (Π_m=(I+mP)/2 carries S back into its
        // eigenspace). Evaluated entirely in origU coordinates (cached above) — O(n)/branch, χ-flat.
        // Accumulate D_w† S D_r and D_w† S P D_r into reusable scratch via pmul_into (same operand
        // order ⇒ byte-identical to the Pauli::multiply chains). D_w and D_r are pure-X words (phase 0),
        // so D_w† == D_w (pauli_dagger of a pure-X phase-0 Pauli is itself).
        gw_reset(gw_Dwf);
        for (int w = 0; w < W; ++w) gw_Dwf.x[w] = groups[g].word[w];   // D_w (= D_w†): direct word copy
        gw_reset(gw_Sf);
        for (int d = 0; d < (int)nfree.size(); ++d) if (br.sigma[d]) pmul_into(gw_Sf, Simg[d]);
        gw_M = gw_Dwf; pmul_into(gw_M, gw_Sf);                 // M = D_w† · S
        gw_acc = gw_M; pmul_into(gw_acc, Dr_f);               // base = M · D_r
        std::complex<double> Eb = frame_diag_E_pk(oeps_pk.data(), gw_acc, W);
        gw_acc = gw_M; pmul_into(gw_acc, Pf); pmul_into(gw_acc, Dr_f);   // withP = M · P · D_r
        std::complex<double> Ew = frame_diag_E_pk(oeps_pk.data(), gw_acc, W);
        std::complex<double> gamma = 0.5 * Eb + 0.5 * (double)m * Ew;
        double mg = std::abs(gamma);
        if (mg > 1e-300) gamma /= mg;                                    // enforce |γ|=1 (numeric guard)
        else gamma = std::complex<double>(1, 0);
#ifndef NDEBUG
        {   // Cross-check (a) the LEAD-2 post-patch bridge ray V_g·|φ0'⟩ reproduces the closed-form
            //   generator signs fsynd[g] (so a release build could use it for proj[g]), and
            // (b) the LEAD-1 closed-form γ_g matches the amplitude gauge against the TRUE projected ray
            //   Π_m D_w|φ0⟩ (the ground truth the overlap oracle uses).
            auto bridge_ptr = anchor->clone();
            AffineState& bridge = static_cast<AffineState&>(*bridge_ptr);
            for (int a = 0; a < N; ++a) if (fbit(g, a) ^ fbit(0, a)) U.Xrow[a].apply_to(bridge);
            for (int a = 0; a < N; ++a)
                assert(gen_sign(bridge, U.Zrow[a]) == (int)fbit(g, a)
                       && "post-patch bridge ray disagrees with closed-form fsynd");
            // TRUE projected ray (independent projection — the oracle's ground truth).
            auto true_ptr = orig_anchor.clone();
            for (int a = 0; a < N; ++a) if ((groups[g].word[a >> 6] >> (a & 63)) & 1) oXrow[a].apply_to(*true_ptr);
            static_cast<AffineState&>(*true_ptr).pauli_project(pauli, q, m);
            AffineState& true_ray = static_cast<AffineState&>(*true_ptr);
            auto built_ptr = anchor->clone();
            AffineState& built = static_cast<AffineState&>(*built_ptr);
            for (int d = 0; d < (int)nfree.size(); ++d) if (br.sigma[d]) U.Xrow[nfree[d]].apply_to(built);
            ExactPhase ov = true_ray.inner_product(built);
            double mod = ov.is_zero ? 0.0 : std::abs(ov.to_complex());
            assert(std::abs(mod - 1.0) <= 1e-7 && "built D_sigma|anchor> not parallel to true ray");
            std::complex<double> gamma_ov = gauge_ratio(true_ray, built);
            assert(std::abs(gamma - gamma_ov) < 1e-7 && "closed-form gauge diverged from amplitude gauge");
        }
#endif
        br.c = groups[g].coeff / gamma;
        branches.push_back(std::move(br));
    }
    // eps/varies/nfree GF(2) reads + LEAD-1 closed-form gauge (dual_image_rows_batch into origU coords +
    // frame_diag_E sign reads + per-branch coefficient/γ arithmetic) reconstruct the SIGN-SENSITIVE
    // branch state ⇒ AFFINE. (The eps/varies/nfree GF(2) part is borderline-tableau but tightly coupled
    // to the branch reconstruction; classified affine and noted in the report.)
    canonicalise();
    // canonicalise() is a MIXED phase (frame column re-basis = tableau + branch coeff/sign fold =
    // affine). Reported as its own bucket rather than force-split into the two totals.
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
    anchor = std::unique_ptr<AffineState>(static_cast<AffineState*>(rays[0]->clone().release()));

    // fs[g][a] = sign bit of the current frame generator U.Zrow[a] on ray_g.
    //
    // NEW (O(χ·n²)): disentangle each ray_g ONCE by the current frame U, reading ALL n syndrome bits
    // from the resulting computational-basis offset — replacing the old O(χ·n) × O(n²) amplitude reads.
    // The disentangling circuit is derived from U.Zrow (the CURRENT frame generators), so it is correct
    // for BOTH calls: call #1 uses the initial generators, call #2 uses the re-based U.Zrow. The key
    // invariant: U†|ray_g⟩ = |s_g⟩, a computational-basis state whose bit string IS the syndrome.
    // After disentangle, s.z_expectation(a) = (-1)^{s_g[a]}, so fs[g][a] = (z_expectation(a)==-1)?1:0.
    //
    // FSYND_XCHECK env guard: when set, re-runs the OLD amplitude method alongside the new and aborts
    // on any mismatch — the durable bit-identical oracle for this subtle numeric path.
    const bool _xchk = std::getenv("FSYND_XCHECK") != nullptr;
    auto compute_fsynd = [&](std::vector<std::vector<uint8_t>>& fs) {
        fs.assign(xs, std::vector<uint8_t>(N, 0));
        // Derive the disentangling circuit from the CURRENT frame generators U.Zrow.
        // This correctly tracks the re-based frame on call #2 (U.Zrow is modified by re-base;
        // rays[0]/anchor is NOT, so a circuit derived from rays[0] alone would be stale on call #2).
        //
        // perm[a]  = qubit index in the disentangled state for generator a.
        // trans[a] = GF(2) bitmask of original generators mixed into row a by row multiplies.
        //
        // After applying the circuit, bit perm[a] of the disentangled state holds:
        //   XOR_{j: trans[a][j]=1} fs[j]   (NOT just fs[a], due to row-multiply mixing).
        // recover_syndromes() undoes this mixing via forward substitution.
        std::vector<int> perm;
        std::vector<std::vector<uint64_t>> trans;
        std::vector<DisentangleGate> dis = disentangle_from_generators(U.Zrow, &perm, &trans);
        for (int g = 0; g < xs; ++g) {
            // Clone ray_g and apply the disentangling circuit.
            auto c_dis = rays[g]->clone();
            AffineState& s = static_cast<AffineState&>(*c_dis);
            for (const auto& gate : dis) apply_disentangle_gate(s, gate);
            // Recover individual syndromes via forward substitution (undoes GF(2) row-multiply mixing).
            recover_syndromes(N, s, perm, trans, fs[g]);

            if (_xchk) {
                // Cross-check: recompute via the OLD amplitude method and abort on any mismatch.
                const std::vector<uint8_t>& x = rays[g]->b;
                const ExactPhase ar = rays[g]->amplitude_at_bits(x);
                for (int a = 0; a < N; ++a) {
                    auto cc = rays[g]->clone(); U.Zrow[a].apply_to(*cc);
                    const ExactPhase ac =
                        static_cast<AffineState*>(cc.get())->amplitude_at_bits(x);
                    uint8_t old = (!ac.is_zero && (((ac.z8 - ar.z8) % 8 + 8) % 8) == 4) ? 1 : 0;
                    if (old != fs[g][a]) {
                        std::fprintf(stderr, "[FSYND_XCHECK] MISMATCH g=%d a=%d old=%d new=%d\n",
                                     g, a, old, fs[g][a]);
                        std::abort();
                    }
                }
            }
        }
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
        // The full O(χ·n²) recompute is skipped. FSYND_XCHECK (env-guarded) re-runs the old
        // method inside compute_fsynd and aborts on any mismatch — the durable equivalence oracle.
        if (_xchk) compute_fsynd(fsynd);
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
        auto built = anchor->clone();
        for (int d = 0; d < (int)nfree.size(); ++d) if (br.sigma[d]) U.Xrow[nfree[d]].apply_to(*built);
        ExactPhase ov = rays[g]->inner_product(*built);                // α = ⟨rays[g]|built⟩, |α|=1
        std::complex<double> alpha = ov.is_zero ? std::complex<double>(1, 0) : ov.to_complex();
        br.c = coeffs[g] / alpha;
        branches.push_back(std::move(br));
    }
    canonicalise();
}

CanonicalStabSum CanonicalStabSum::from_rays(int n, std::vector<std::unique_ptr<AffineState>> rays,
                                             std::vector<std::complex<double>> coeffs) {
    CanonicalStabSum s(n);
    s.rebuild_from_rays(std::move(rays), std::move(coeffs));
    return s;
}

// ── Single-qubit measurement: Case A (commuting) + Case B (anticommuting, theory §5) ──────────
int CanonicalStabSum::measure_single(int pauli, int q, double u) {
    flush_gates();                              // materialise the deferred frame before any U read
    Pauli P = single_pauli(pauli, q, n());
    Pauli Q = U.conjugate_single(pauli, q);     // Q = U† P U from forward rows (no inverse tableau;
                                                // a prior measurement may have left the dual stale)

    // A = supp(Q.x). Nonempty ⇒ Q anticommutes with a generator ⇒ Case B (theory §5). Word-scan for
    // emptiness first so the common Case-A path never builds the A vector.
    {
        bool anti = false;
        for (int w = 0, nw = (n() + 63) / 64; w < nw; ++w) if (Q.x[w]) { anti = true; break; }
        if (anti) {
            static thread_local std::vector<int> A; A.clear();   // reusable (capacity-retaining)
            for (int a = 0; a < n(); ++a) if (Q.xbit(a)) A.push_back(a);
            return measure_single_anticommuting(pauli, q, P, A, u);
        }
    }

    // Case A. Q = i^{Q.phase} Z^{q_z}; s0 = i^{Q.phase}·∏_{a:q_z[a]}(-1)^{eps[a]} ∈ {±1}.
    int s0 = ((Q.phase & 3) == 2) ? -1 : 1;
    for (int a = 0; a < n(); ++a) if (Q.zbit(a) && eps[a]) s0 = -s0;

    // Per-branch eigenvalue λ_i = s0·(-1)^{q_z·σ̃_i}.
    int x = chi();
    std::vector<int> lam(x);
    for (int i = 0; i < x; ++i) {
        int dot = 0;
        for (int d = 0; d < (int)free.size(); ++d)
            if (Q.zbit(free[d]) && branches[i].sigma[d]) dot ^= 1;
        lam[i] = dot ? -s0 : s0;
    }

    // p_+ = Σ_{λ_i=+1}|c_i|².
    double pp = 0.0;
    for (int i = 0; i < x; ++i) if (lam[i] == +1) pp += std::norm(branches[i].c);
    if (pp < 0) pp = 0;
    if (pp > 1) pp = 1;

    int m = (u < pp) ? +1 : -1;
    double pm = (m == +1) ? pp : (1.0 - pp);
    double inv = (pm > 0) ? 1.0 / std::sqrt(pm) : 0.0;

    // ── PURE IN-PLACE COLLAPSE (Case A is structurally trivial — no projection, no gauge, no frame
    // edit). Commuting Q leaves every survivor's ray UNCHANGED: Π_m|φ_i⟩ = |φ_i⟩ for λ_i = m, so the
    // anchor |φ0⟩, the frame U (generators g_a AND destabilisers d_a) and every surviving σ pattern are
    // exactly as before. We simply DROP the λ_i ≠ m branches, rescale survivors by 1/√p_m, and re-run
    // canonicalise() (which folds any now-constant free generator into the anchor and re-minimises the
    // free set). O(χ·n) — matches v1's fast-diagonal Case A. (If the OLD anchor branch σ=0 was dropped
    // canonicalise's reference row 0 just becomes the first survivor; |φ0⟩ stays a valid +1 reference
    // since g_a|φ0⟩ signs eps are untouched.)
    std::vector<Branch> kept;
    kept.reserve(x);
    for (int i = 0; i < x; ++i)
        if (lam[i] == m) { Branch br = branches[i]; br.c *= inv; kept.push_back(std::move(br)); }
    if (kept.empty()) { branches.clear(); free.clear(); return m; }  // zero-probability outcome
    branches = std::move(kept);
    canonicalise();
    return m;
}

// Batched measurement with deferred Case-A collapse. While a run of commuting measurements is
// open, the frame, eps, free and the branch list are all UNTOUCHED (Case A never edits them) —
// only an `alive` mask and the running outcome-probability product `scale` evolve. Each commuting
// measurement j costs one conjugate_single + an O(alive·|free|) eigenvalue column; the chain-rule
// probability is p_j = (surviving λ=+1 weight)/scale, exactly the sequential value (sequential
// rescales coefficients by 1/sqrt(p) each step; dividing the original weights by the running
// product is the same quantity, regrouped). The single flush drops dead branches, rescales by
// 1/sqrt(scale) and canonicalise()s once. A Case-B measurement flushes the run (its collapse and
// frame edits need the materialised state), runs measure_single, and a fresh run resumes.
std::vector<int> CanonicalStabSum::measure_batch(const std::vector<std::pair<int, int>>& paulis,
                                                 const std::vector<double>& us) {
    if (paulis.size() != us.size())
        throw std::logic_error("measure_batch: paulis/us length mismatch");
    const int M = (int)paulis.size();
    std::vector<int> out(M);
    flush_gates();                              // materialise the deferred frame before any U read

    std::vector<uint8_t> alive, lamb;           // run state: surviving mask, per-branch λ=+1 bits
    double scale = 1.0;                         // Π p_l over the open run
    bool run_open = false;
    auto flush_run = [&]() {
        if (!run_open) return;
        run_open = false;
        const double inv = (scale > 0) ? 1.0 / std::sqrt(scale) : 0.0;
        std::vector<Branch> kept;
        kept.reserve(branches.size());
        for (size_t i = 0; i < branches.size(); ++i)
            if (alive[i]) { Branch br = std::move(branches[i]); br.c *= inv; kept.push_back(std::move(br)); }
        if (kept.empty()) { branches.clear(); free.clear(); return; }   // zero-probability tail
        branches = std::move(kept);
        canonicalise();
    };

    // Conjugation strategy: with a VALID dual tableau, conjugate_single is an O(n) cached-row
    // read per measurement. With a STALE dual (the usual state after any Case-B measurement,
    // under the lazy-dual policy) the per-measurement fallback is an O(n²) dual_image — so we
    // batch-conjugate ALL remaining measurements in ONE dual_image_rows_batch pass (the
    // transposed-tableau form: tableau transposed once, each single-qubit target costs a couple
    // of column XORs + its phase chain). A Case-B measurement edits the frame, so the prefetched
    // batch is discarded and rebuilt from the next index. Outputs are the same unique exact
    // Pauli U†PU either way.
    std::vector<Pauli> qbuf;
    int qb_base = -1;
    auto rebatch = [&](int from) {
        std::vector<Pauli> tg;
        tg.reserve(M - from);
        std::vector<const Pauli*> tp(M - from);
        for (int t = from; t < M; ++t) {
            tg.push_back(single_pauli(paulis[t].first, paulis[t].second, n()));
            tp[t - from] = &tg[t - from];
        }
        qbuf.assign(M - from, Pauli(n()));
        dual_image_rows_batch(U.Xrow, U.Zrow, n(), tp.data(), M - from, qbuf.data());
        qb_base = from;
    };

    const int nw = (n() + 63) / 64;
    Pauli Qtmp(n());
    for (int j = 0; j < M; ++j) {
        const int pauli = paulis[j].first, q = paulis[j].second;
        const Pauli* Qp;
        if (U.dual_valid()) {
            Qtmp = U.conjugate_single(pauli, q);
            Qp = &Qtmp;
        } else {
            if (qb_base < 0 || j < qb_base) rebatch(j);
            Qp = &qbuf[j - qb_base];
        }
        const Pauli& Q = *Qp;
        bool anti = false;
        for (int w = 0; w < nw; ++w) if (Q.x[w]) { anti = true; break; }
        if (anti) {
            flush_run();                        // pending Case-A outcomes must land first
            out[j] = measure_single(pauli, q, us[j]);
            qb_base = -1;                       // frame edited: prefetched conjugations are stale
            continue;
        }
        if (!run_open) { alive.assign(chi(), 1); scale = 1.0; run_open = true; }
        // s0 and per-branch eigenvalues — identical arithmetic to measure_single's Case A
        // (eps/free/branches are stable while the run is open).
        int s0 = ((Q.phase & 3) == 2) ? -1 : 1;
        for (int a = 0; a < n(); ++a) if (Q.zbit(a) && eps[a]) s0 = -s0;
        const int x = chi();
        lamb.assign(x, 0);
        double wplus = 0.0;
        for (int i = 0; i < x; ++i) {
            if (!alive[i]) continue;
            int dot = 0;
            for (int d = 0; d < (int)free.size(); ++d)
                if (Q.zbit(free[d]) && branches[i].sigma[d]) dot ^= 1;
            const int lam = dot ? -s0 : s0;
            lamb[i] = (uint8_t)(lam == +1);
            if (lam == +1) wplus += std::norm(branches[i].c);
        }
        double pp = (scale > 0) ? wplus / scale : 0.0;
        if (pp < 0) pp = 0; if (pp > 1) pp = 1;
        const int m = (us[j] < pp) ? +1 : -1;
        out[j] = m;
        const uint8_t want = (uint8_t)(m == +1);
        for (int i = 0; i < x; ++i) if (alive[i] && lamb[i] != want) alive[i] = 0;
        scale *= (m == +1) ? pp : (1.0 - pp);
    }
    flush_run();
    return out;
}

std::vector<std::complex<double>> CanonicalStabSum::to_statevector() const {
    assert(n() < 31 && "to_statevector: 2^n statevector infeasible for n>=31 (and 1<<n is UB for n>=64)");
    flush_gates();                              // materialise the deferred frame before any U read
    size_t dim = (size_t)1 << n();
    std::vector<std::complex<double>> out(dim, std::complex<double>(0, 0));
    for (const auto& br : branches) {
        auto st = anchor->clone();
        // Apply the destabiliser product D_σ = ∏_{a∈free:σ[a]} d_a, with d_a = U X_a U† = Xrow[free[a]].
        for (int a = 0; a < (int)free.size(); ++a)
            if (br.sigma[a]) U.Xrow[free[a]].apply_to(*st);
        auto sv = st->to_statevector();
        for (size_t x = 0; x < dim; ++x) out[x] += br.c * sv[x];
    }
    return out;
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

CanonicalStabSum::CanonicalStabSum(const CanonicalStabSum& o, CloneTag)
    : U(o.U), eps(o.eps), free(o.free), branches(o.branches) {
    // (pending_gates is empty — clone() flushed before constructing.)
    if (o.anchor) {
        // anchor->clone() returns std::unique_ptr<StabState>; the concrete type is AffineState.
        auto* ap = static_cast<AffineState*>(o.anchor->clone().release());
        anchor = std::unique_ptr<AffineState>(ap);
    }
}

void CanonicalStabSum::clone_into(CanonicalStabSum& dst) const {
    flush_gates();                              // materialise the deferred frame so the copy of U is current
    dst.U.assign_from(U);                       // memcpy fast path into dst's existing frame buffers
    dst.eps = eps;
    dst.free = free;
    dst.branches = branches;
    dst.pending_gates.clear();                  // source flushed; dst must not replay stale gates
    if (anchor) {
        if (dst.anchor) *dst.anchor = static_cast<const AffineState&>(*anchor);
        else dst.anchor.reset(static_cast<AffineState*>(anchor->clone().release()));
    } else {
        dst.anchor.reset();
    }
}

CanonicalStabSum CanonicalStabSum::clone() const {
    flush_gates();                              // materialise the deferred frame so the copy of U is current
    return CanonicalStabSum(*this, CloneTag{});
}

}  // namespace qeccore
