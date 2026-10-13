#include "qeccore/twirl_kernel.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qeccore {

DiagNormalForm residual_normal_form(const DiagPauliClifford& composed) {
    const int n = composed.n;
    DiagNormalForm nf;
    nf.prefix = Pauli(n);
    nf.a.assign(n, 0);

    // Linear layer split  a_j = s_j + 2 z_j:
    //   s_j = a_j & 1  -> residual S content (kept in nf.a, ∈ {0,1})
    //   z_j = a_j & 2  -> Z part (S²=Z), folded into the prefix Pauli's Z support
    // X-translation v_j folds into the prefix Pauli's X support.
    // Hot-path (2026-07-15): a finalized composed op carries cache->active_a (qubits with a≠0), so
    // the a-layer loop visits only the support instead of all n wires. The v scan stays O(n) bytes
    // (no support list exists for it; it is a cheap single pass). Identical output either way.
    if (composed.cache) {
        for (int q : composed.cache->active_a) {
            const int aq = composed.a[q] & 3;
            if (aq & 1) nf.a[q] = 1;
            if (aq & 2) nf.prefix.setz(q);
        }
        for (int q = 0; q < n; ++q)
            if (composed.v[q] & 1) nf.prefix.setx(q);
    } else
    for (int q = 0; q < n; ++q) {
        const int aq = composed.a[q] & 3;
        if (aq & 1) nf.a[q] = 1;
        if (aq & 2) nf.prefix.setz(q);
        if (composed.v[q] & 1) nf.prefix.setx(q);
    }

    // Global phase gamma = ζ8^{z8} (z8 ∈ [0,8)) folds into the prefix Pauli's i^phase.
    //
    // gamma is GLOBALLY UNOBSERVABLE in the twirl kernel: it cancels in every per-generator
    // sign R†g_iR (gamma·gamma^{-1}), in all sector probabilities, and appears in a collapsed
    // state only as an overall global phase — shots never interfere, so no gamma is ever
    // measured. The Python reference embodies this too (compose_diag drops the global phase;
    // build_shot_law never reads P.p). Consumers must therefore NEVER branch on prefix.phase,
    // with the single exception of the Task-6 reconstruction oracle, whose factory inputs
    // (S/Sdg/Z/X/Y/CZ compositions) are always even and for which the fold is EXACT.
    //
    //  - scale != 0 is impossible for a real (odd γ) residual and stays a HARD throw.
    //  - Even z8 (Task-6 factory inputs): keep the exact fold phase = z8/2 (mod 4); the
    //    reconstruct oracle rebuilds gamma = ζ8(2·phase) and compares omega exactly, so any
    //    dropped i/−1 there is caught.
    //  - Odd z8: an odd ζ8 residual arises only from X-crossing-T propagation in composed
    //    real residuals (clifford_conjugate T rule add_z8(1)). Fold the even part (z8>>1) and
    //    DISCARD the leftover ζ8 factor — it is globally unobservable (argument above). This
    //    replaces the former throw, which fired on genuine cultivation_d5 shots.
    if (composed.gamma.scale != 0)
        throw std::logic_error("residual_normal_form: gamma.scale != 0 — diagonal-class invariant violated");
    nf.prefix.phase = ((composed.gamma.z8 >> 1) % 4 + 4) % 4;

    // CZ layer: every unordered pair (j<l) with B_{jl}=1, once. Handles the lazy
    // all-zero B form (words()==0 -> no rows, no pairs).
    // Hot-path (2026-07-15): a finalized op's cache->b_pairs IS this list (j<l, built by the same
    // row scan in finalize()); consume it directly instead of rescanning n×Bwords words per shot.
    if (composed.cache) {
        nf.cz = composed.cache->b_pairs;           // already each-pair-once with j<l
    } else {
    const int Bwords = composed.B.words();
    for (int j = 0; j < n; ++j) {
        const uint64_t* Brow = composed.B.row(j);
        for (int wi = 0; wi < Bwords; ++wi) {
            uint64_t word = Brow[wi];
            while (word) {
                const int l = wi * 64 + __builtin_ctzll(word);
                word &= word - 1;
                if (l <= j) continue;              // each unordered pair once (l > j)
                nf.cz.emplace_back(j, l);
            }
        }
    }
    }
    std::sort(nf.cz.begin(), nf.cz.end());

    // A DiagPauliClifford is diagonal-and-Pauli by construction: it can only ever
    // represent gamma·X^v·diag(q), so non-diagonal content cannot reach this function
    // (the census flags PPR/general atoms BEFORE they compose into a DiagPauliClifford
    // and takes the exact fallback for them). The diagonal_class=false branch is thus
    // unreachable given the type system, and is reported (not coded) as such.
    nf.diagonal_class = true;
    return nf;
}

// ── Task 11.5: residual canonicalisation modulo the certified group's Z-content. ─────────────────
//
// LATTICE-SAFE SCOPE — we reduce ONLY against the PRODUCT-WIRE pure-Z stabilisers (wires q whose
// ±Z_q is a weight-1 generator; prod_sign[q]≠0). This is provably invisible to the twirl law's
// dressing lattice: a product wire q commutes with the whole group, so g_i.x[q]=0 for every
// generator, hence every dressing v_i = M·x_i is independent of M's q-th row/column. Stripping a
// product-wire axis leg therefore leaves {v_i}, r, κ, coin_masks and the active mask UNCHANGED — it
// only collapses the (a,cz) MEMO KEY (residuals differing solely on product-wire content share a
// plan) and shifts a −Z control's CZ leg into a prefix Z. So the cached plan is exactly correct and
// build_shot_law never sees a new lattice (κ stays 0 on d5). The GENERAL multi-qubit-stabiliser axis
// relocation (a weight-3 reduced axis) is a SOUND state rewrite too, but it changes M·x_i and thus κ
// — build_shot_law only implements κ≤1, so that path is DEFERRED (see task report). ε (the stripped
// stabilisers' sign product) still flips S↔S† / turns a −Z-controlled CZ into a target Z, exactly.
DiagNormalForm canonicalize_mod_stabilizers(const DiagNormalForm& nf, const CertifiedGroupPlanes& G) {
    DiagNormalForm out = nf;
    canonicalize_mod_stabilizers_inplace(out, G);
    return out;
}

bool canonicalize_mod_stabilizers_inplace(DiagNormalForm& nfio, const CertifiedGroupPlanes& G) {
    const DiagNormalForm& nf = nfio;                      // read-only view until the rewrite commits
    if (!nf.diagonal_class) return false;                 // out-of-class ⇒ exact fallback, untouched
    const int n = (int)nf.a.size();
    const CertifiedGroupPlanes::ZAxisRREF& Z = G.z_axis_rref();

    // Fast identity path (profile-driven, 2026-07-15): the only implemented reduction is the
    // product-wire strip, so when no S/CZ leg touches a product wire the function is exactly the
    // identity. One O(n + |cz|) scan, zero allocations, ZERO copies, decides it.
    {
        bool touches = false;
        for (int q = 0; q < n && !touches; ++q)
            if ((nf.a[q] & 1) && Z.prod_sign[q] != 0) touches = true;
        for (size_t i = 0; i < nf.cz.size() && !touches; ++i)
            if (Z.prod_sign[nf.cz[i].first] != 0 || Z.prod_sign[nf.cz[i].second] != 0) touches = true;
        if (!touches) return false;
    }

    // Closed-form rebuild (profile-driven rewrite, 2026-07-15 — replaces the std::map phase-poly
    // accumulation, ~46 µs/shot of allocation churn at n=298, byte-identical output).
    // After the product-wire strip every reduced axis is a UNIT or a PAIR (the strip only removes
    // bits: pair→unit/empty, unit→empty), and pair axes cannot merge (cz pairs are unique), so the
    // generic accumulation collapses to a closed form:
    //   kept pair (q,q') — both endpoints unstripped — survives verbatim (Φ_quad = 4);
    //   unit-axis coefficient at unstripped q:
    //     L_q = −a_q − deg_q + Σ_{(q,x)∈cz : x stripped} s_x
    //     (each incident cz contributes a −1 unit leg regardless of its partner; a pair whose
    //      partner was stripped leaves a +s_x remnant on q, ε = the stripped wire's ±Z sign);
    //   Φ_lin[q] = −2·(L_q + kept_deg_q)  ⇒  sp_q = (−L_q − kept_deg_q) mod 4;  a'_q = sp_q & 1;
    //   sp_q ≥ 2 folds S² = Z into the prefix.  Stripped q: no residual content (a fully-stripped
    //   axis is a discarded global phase — the old add_axis empty-return).
    //   Identity check (nothing stripped): L_q = −a_q − deg_q, kept_deg_q = deg_q ⇒ sp_q = a_q,
    //   cz' = cz — matches the fast path above.
    DiagNormalForm out;
    out.prefix = nf.prefix;                    // X, phase unchanged (phase is global/unobservable)
    out.a.assign(n, 0);
    out.diagonal_class = true;
    std::vector<int16_t> L(n, 0), kept_deg(n, 0);
    for (int q = 0; q < n; ++q) if (nf.a[q] & 1) L[q] -= 1;
    out.cz.reserve(nf.cz.size());
    for (const auto& e : nf.cz) {
        const int s1 = Z.prod_sign[e.first], s2 = Z.prod_sign[e.second];
        if (s1 == 0) L[e.first]  -= 1;         // unit leg of the CZ decomposition, kept
        if (s2 == 0) L[e.second] -= 1;
        if (s1 == 0 && s2 == 0) {              // pair axis kept verbatim
            out.cz.push_back(e); ++kept_deg[e.first]; ++kept_deg[e.second];
        } else if (s1 == 0) {                  // partner stripped: pair → unit remnant ε·(+1)
            L[e.first] += s2;
        } else if (s2 == 0) {
            L[e.second] += s1;
        }                                       // both stripped: discarded global phase
    }
    for (int q = 0; q < n; ++q) {
        if (Z.prod_sign[q] != 0) continue;      // stripped wire: no residual content survives
        const int sp = ((-(int)L[q] - (int)kept_deg[q]) % 4 + 4) % 4;
        out.a[q] = (uint8_t)(sp & 1);
        if (sp >= 2) out.prefix.flipz(q);       // S² = Z folded into the prefix
    }
    nfio = std::move(out);                      // cz kept in input (sorted) order ⇒ still sorted
    return true;
}

}  // namespace qeccore
