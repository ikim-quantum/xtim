#include "qeccore/normal_form.hpp"
#include <cstddef>
#include "qeccore/pauli_rotation_form.hpp"   // build_pauli_rotation_form (U = C·T)
#include "qeccore/pauli_diagonalize.hpp"     // diagonalize_commuting_layer
#include "qeccore/parity_support.hpp"        // restrict_to_support
#include "qeccore/parity_assemble.hpp"       // assemble_from_parities
#include "qeccore/fast_todd.hpp"             // fast_todd_reduce
#include "qeccore/residue_projection.hpp"    // Projector / moebius
#include <cstdint>
#include <map>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace qeccore {

// TEST-ONLY: count of phase-faithful recoveries that emitted a non-trivial even (Clifford) residue.
long g_even_residue_recovered_count = 0;
// TEST-ONLY: count of recoveries whose even residue carried a DEG-2 (CZ) monomial (the deg-2
// even-residue reconstruction branch — 4·t_j·t_k split into three even parity columns — fired).
long g_even_residue_deg2_count = 0;

// Dagger a cprime gate list: reverse order, invert each gate (S<->SDG; H,X,Y,Z,CX,CZ self-inverse).
static std::vector<CliffGate> dagger_cliff_list(const std::vector<CliffGate>& gs) {
    std::vector<CliffGate> out;
    out.reserve(gs.size());
    for (auto it = gs.rbegin(); it != gs.rend(); ++it) {
        CliffGate g = *it;
        if (g.kind == GateKind::S) g.kind = GateKind::SDG;
        else if (g.kind == GateKind::SDG) g.kind = GateKind::S;
        out.push_back(g);
    }
    return out;
}

// FastTODD residue-project -> reduce -> map-back on the genuine-magic (odd-coeff) parity columns.
// Operates over the ParitySupport columns (the GradedPoly todd_condition_block path it once mirrored
// was deleted in C4-Int 6):
//   (1) combine equal columns (coeffs mod 16), keep the genuine magic (odd, non-constant, c≠8) — the
//       assembler's notion of magic — and build a Projector over those distinct directions, so the
//       residue rank r is the TRUE magic rank (not the raw column count that would blow up FastTODD);
//   (2) if r > 30 (uint32 residue-mask guard: Projector::project packs into uint32, truncating at
//       r > 31) skip — columns AS-IS; else project the magic directions to r-bit masks and
//       fast_todd_reduce them, then Projector::reconstruct each reduced mask to a d-wide Row;
//   (3) REPLACE the magic columns with the reduced set — this brings χ down from the raw column rank
//       to 2^(true rank) and makes χ FEASIBLE (e.g. bgk_1513_x4 20-dim raw → rank 4; cube_ccz 4 → 3).
// Phase: a reduced direction that MATCHES an input direction (residue mask among the projections)
// carries that input's exact combined coeff; a FastTODD-INTRODUCED direction is an odd T-parity at
// coeff +1.  FastTODD's mod-8 Clifford bookkeeping is intentionally dropped, so an introduced
// direction whose true sign/grade is set by a CUBIC reduction (deep-triorthogonal cultivation_d5
// family) is NOT recovered from the degree-1 columns — those cases are caught by the differential
// firewall (overlap < 1, χ still correct) and reported as the cutover blocker, never silently wrong.
// Transversal-T, CS, CCZ (incl. cube_ccz) and the H-on-magic class all reduce phase-faithfully here.
static void fast_todd_reduce_columns(ParitySupport& ps) {
    const int d = ps.support.k_;
    const int dwords = (d + 63) / 64;
    if (d == 0 || ps.columns.empty()) return;

    // Combine equal columns (coeffs add mod 16) and keep only the GENUINE-MAGIC residue — the odd
    // survivors after dropping coeff 0/8 (constant on every coset) and even (Clifford) columns. This
    // is exactly the assembler's notion of magic; building the span over these distinct directions
    // keeps the residue rank r == the true magic rank (NOT the raw, possibly larger, column count —
    // which would blow up fast_todd_reduce's O(2^r) residue work).
    std::map<std::vector<uint64_t>, int> comb;
    for (size_t k = 0; k < ps.columns.size(); ++k) {
        std::vector<uint64_t> col = ps.columns[k];
        if ((int)col.size() < dwords) col.resize(dwords, 0);
        comb[col] += ps.coeffs[k];
    }
    Projector P; P.words = dwords;
    std::map<std::vector<uint64_t>, int> magic_coeff;   // distinct magic direction -> its combined coeff
    for (auto& kv : comb) {
        int c = ((kv.second % 16) + 16) % 16;
        if (c == 0 || c == 8 || (c % 2 == 0)) continue;   // identity / global / Clifford
        bool zero = true; for (auto w : kv.first) if (w) { zero = false; break; }
        if (zero) continue;                          // constant column ⇒ global phase, not magic
        Row rr = kv.first;
        P.extend(rr);
        magic_coeff[kv.first] = c;
    }
    const int r = P.r();
    if (r == 0 || r > 30) return;                   // no magic, or uint32 residue-mask guard ⇒ AS-IS

    // Project the distinct magic directions into the r-bit residue space and run FastTODD.  The
    // reduced set `red` spans the SAME r-dimensional residue (true magic rank) but with the MINIMAL
    // T-count — this is what brings χ down from the raw column rank to 2^(true rank) and makes χ
    // feasible (e.g. bgk_1513_x4 20→? , cube_ccz 4→3).
    std::vector<uint32_t> proj;
    proj.reserve(magic_coeff.size());
    for (auto& kv : magic_coeff) proj.push_back(P.project(kv.first));
    std::vector<uint32_t> red = fast_todd_reduce(proj, r);

    // ── PHASE-FAITHFUL rebuild: reduced ODD T-set + recovered EVEN (Clifford) residue ──────────────
    // FastTODD reduces the magic DIRECTIONS only: it sees each input as a coeff-1 T (its proper()/
    // tohpe() work mod-2 on the masks), so `red` spans the SAME residue as `proj` with the minimal
    // T-count, but FastTODD DROPS the even-graded (Clifford) remainder of the reduction — including
    // (a) the mod-8 Clifford residue of a genuinely CUBIC reduction (cultivation_d5) and (b) the
    // even part of any input whose coeff ≠ 1 (mod 8).  We RECOVER that remainder densely over the
    // rank-r residue and fold it back as even (Clifford) parity columns, so the assembled state's
    // total phase is f' + g = f EXACTLY (phase-faithful), at the reduced χ = 2^r.
    //
    // Convention (all in the column relphase: a column (mask m, coeff c mod 16) contributes the
    // per-residue-point phase ζ16^{c·(-1)^{m·x}}, whose RELATIVE-phase ζ8 content is (-c)·p_m(x)
    // with p_m(x)=popcount(m&x)&1 — matching dense_eval's parity convention):
    //   • f(x)  = Σ_input (-c_in)·p_{m_in}(x)            (the TRUE magic phase poly over the residue)
    //   • f'(x) = Σ_{m∈red} (+1)·p_m(x)                  (each reduced direction as a coeff-1 T)
    //   • g(x)  = f(x) − f'(x)  (mod 8)                  (the Clifford remainder — PROVABLY even-graded:
    //        f = Σ_in(+1)p_m + Σ_in(-c_in−1)p_m, the 2nd sum is even-coeff⇒Clifford, and
    //        Σ_in(+1)p_m = Σ_proj(+1)p_m ≡ Σ_red(+1)p_m (mod Clifford) by FastTODD's span theorem).
    // A reduced direction representing a coeff-1 T is the column (m, 7): -7 ≡ 1 (mod 8) ⇒ ζ8 +1·p.
    auto p_par = [](uint32_t m, uint32_t x) -> int { return __builtin_popcount(m & x) & 1; };

    // f over the residue (dense ζ8 table over 2^r): Σ_input (-c_in mod 8)·p_{m_in}.
    std::vector<uint8_t> f_dense((size_t)1 << r, 0);
    for (auto& kv : magic_coeff) {
        uint32_t m = P.project(kv.first);
        int c8 = ((-(kv.second)) % 8 + 8) % 8;
        if (!c8) continue;
        for (uint32_t x = 0; x < ((uint32_t)1 << r); ++x)
            if (p_par(m, x)) f_dense[x] = (uint8_t)((f_dense[x] + c8) & 7);
    }
    // f' over the residue: Σ_{m∈red} (+1)·p_m.
    std::vector<uint8_t> fp_dense((size_t)1 << r, 0);
    for (uint32_t m : red) {
        if (!m) continue;
        for (uint32_t x = 0; x < ((uint32_t)1 << r); ++x)
            if (p_par(m, x)) fp_dense[x] = (uint8_t)((fp_dense[x] + 1) & 7);
    }
    // g = f − f' (mod 8), dense.
    std::vector<uint8_t> g_dense((size_t)1 << r, 0);
    for (uint32_t x = 0; x < ((uint32_t)1 << r); ++x)
        g_dense[x] = (uint8_t)((f_dense[x] - fp_dense[x]) & 7);

    // g MUST be even-graded (its odd/T part is zero) — otherwise the FastTODD reduction or this
    // reconstruction is wrong.  Möbius to the monomial basis and check deg-1 even, deg-2 ≡0 mod4,
    // deg≥3 == 0.  Fail-LOUD (abort) rather than silently mis-build — the firewall would catch it too.
    std::vector<uint8_t> g_mob(g_dense);
    moebius(g_mob, r);
    for (size_t s = 1; s < g_mob.size(); ++s) {
        int deg = __builtin_popcount((unsigned)s), a = g_mob[s] & 7;
        bool ok2 = (deg == 1) ? !(a & 1) : (deg == 2) ? !(a & 3) : (a == 0);
        if (!ok2) {
            std::fprintf(stderr,
                "fast_todd_reduce_columns: recovered residue g is NOT even-graded "
                "(monomial 0x%zx deg=%d coeff=%d) — reduction/recovery model is wrong\n", s, deg, a);
            std::abort();
        }
    }

    {   // TEST-ONLY: record that a non-trivial even (Clifford) residue was recovered here.
        bool nontrivial = false;
        for (size_t s = 1; s < g_mob.size(); ++s) if (g_mob[s] & 7) { nontrivial = true; break; }
        if (nontrivial) ++g_even_residue_recovered_count;
    }

    std::vector<std::vector<uint64_t>> new_cols;
    std::vector<int> new_coeffs;
    // (1) reduced ODD columns: each a coeff-1 T ⇒ column coeff 7 (ζ8 +1·p).
    for (uint32_t m : red) {
        if (!m) continue;
        std::vector<uint64_t> dir = P.reconstruct(m);
        bool nz = false; for (auto w : dir) if (w) { nz = true; break; }
        if (!nz) continue;
        new_cols.push_back(std::move(dir));
        new_coeffs.push_back(7);
    }
    // (2) recovered EVEN (Clifford) residue g, emitted as even parity columns the assembler folds.
    //   • deg-1 monomial {j}, coeff a (∈{2,4,6}): one even column (mask e_j, coeff c) with -c ≡ a
    //     (mod 8) ⇒ c = (8−a) mod 8 (even). Its ζ8 relphase a·t_j matches.
    //   • deg-2 monomial {j,k}, coeff 4 (= CZ = ζ8^{4 t_j t_k}): use 4·t_j·t_k = 2 t_j + 2 t_k −
    //     2(t_j⊕t_k) (mod 8) ⇒ THREE even columns on residue masks e_j, e_k, e_j⊕e_k.
    auto emit_even = [&](uint32_t resmask, int a8) {        // even ζ8 relphase a8·p_resmask
        a8 = ((a8 % 8) + 8) % 8;
        if (a8 == 0) return;
        int c16 = ((8 - a8) % 8 + 8) % 8;                  // column coeff: -c16 ≡ a8 (mod 8)
        std::vector<uint64_t> dir = P.reconstruct(resmask);
        bool nz = false; for (auto w : dir) if (w) { nz = true; break; }
        if (!nz) return;                                   // constant ⇒ pure global phase, drop
        new_cols.push_back(std::move(dir));
        new_coeffs.push_back(c16);
    };
    for (size_t s = 1; s < g_mob.size(); ++s) {
        int deg = __builtin_popcount((unsigned)s), a = g_mob[s] & 7;
        if (!a) continue;
        if (deg == 1) {
            emit_even((uint32_t)s, a);                     // a ∈ {2,4,6}
        } else {                                           // deg == 2, a == 4 (CZ); guarded above
            ++g_even_residue_deg2_count;                   // TEST-ONLY: deg-2 (CZ) residue branch fired
            int j = __builtin_ctz((unsigned)s);
            int k = __builtin_ctz((unsigned)s & ((unsigned)s - 1));
            emit_even((uint32_t)1 << j, 2);
            emit_even((uint32_t)1 << k, 2);
            emit_even(((uint32_t)1 << j) ^ ((uint32_t)1 << k), 6);   // −2 ≡ 6 (mod 8)
        }
    }
    // Carry over the EVEN (Clifford) / constant columns of the ORIGINAL support (untouched by the
    // magic reduction): assemble_from_parities folds them exactly.
    for (auto& kv : comb) {
        int c = ((kv.second % 16) + 16) % 16;
        if (c == 0) continue;                          // identity
        bool zero = true; for (auto w : kv.first) if (w) { zero = false; break; }
        if (c % 2 == 1 && !zero && c != 8) continue;   // genuine magic — replaced via the reduced set
        new_cols.push_back(kv.first);
        new_coeffs.push_back(c);
    }
    ps.columns = std::move(new_cols);
    ps.coeffs = std::move(new_coeffs);
}

BareState build_bare_state_parity(const Circuit& deferred) {
    // the parity-native assembler builds at the TRUE magic rank chi; the propagation-class
    // check (build_pauli_rotation_form) is the SOLE rejection.
    const int n = deferred.n;
    BareState out;   // default-constructed = rejected

    // Step 1: U = C·T (single commuting magic layer). Out-of-class or non-commuting magic ⇒ reject.
    PauliRotationForm f = build_pauli_rotation_form(deferred);
    if (f.rejected) {
        out.rejected = true;
        out.reject_gate_index = f.reject_gate_index;   // >=0 out-of-class; -1 non-commuting magic
        out.reject_reason = f.reject_reason;
        return out;
    }

    // Step 2: diagonalize the commuting magic layer (odd AND even commute when the form accepts —
    // verified across the engine fixtures; even-coeff terms carry Clifford content that restrict/
    // assemble fold, so they must NOT be dropped).  C' (cprime), Z-strings tz.
    //
    // Guard: build_pauli_rotation_form's settlement test verifies odd-odd commutation AFTER
    // settling even entries, but even-coeff entries in f.terms may still anticommute with adjacent
    // entries in rare CH+Clifford sequences. Catch this gap here and reject gracefully instead of
    // letting diagonalize_commuting_layer's assert fire.
    for (int _i = 0; _i < (int)f.terms.size(); ++_i)
        for (int _j = _i + 1; _j < (int)f.terms.size(); ++_j)
            if (!Pauli::commute(f.terms[_i].pauli, f.terms[_j].pauli)) {
                out.rejected = true;
                out.reject_gate_index = -1;
                out.reject_reason = "rotation-form terms anticommute (even-Clifford ladder gap)";
                return out;
            }
    DiagResult diag = diagonalize_commuting_layer(f.terms, n);

    // Step 3: restrict to the support |psi> = C'†|0>; magic parity columns over d free vars + const phase.
    std::vector<CliffGate> cprime_dagger = dagger_cliff_list(diag.cprime);
    ParitySupport ps = restrict_to_support(cprime_dagger, diag.tz, n);

    // Step 4: FastTODD residue-project -> reduce -> map-back over the magic columns (T-count min;
    // no-op for transversal-T; r>30 skips). Span/chi preserved.
    fast_todd_reduce_columns(ps);

    // Step 5: output Clifford = C · C' (C' applied first, then C). As an output list [0]=FIRST:
    // [cprime gates ..., f.cliffords ...] — both already in [0]-first order.
    std::vector<CliffGate> output_clifford = diag.cprime;
    output_clifford.insert(output_clifford.end(), f.cliffords.begin(), f.cliffords.end());

    ParityBareState bs = assemble_from_parities(ps, output_clifford, n);

    // Inject the form's tracked global phase (e^{iπ/8} accumulators from T/CS/CCZ). The parity
    // assembler already folds ps.global_phase_mult8; f.global_phase is the SEPARATE ζ16 scalar of
    // U = global_phase·C·T. It multiplies every branch coefficient equally (an exact global phase).
    {
        std::complex<double> gp = f.global_phase;
        double mag = std::abs(gp);
        if (mag > 1e-300) gp /= mag;             // normalize to unit modulus (guard fp drift)
        for (auto& br : bs.state.branches) br.c *= gp;
    }

    out.state = std::move(bs.state);
    out.chi = bs.chi;
    out.rejected = false;
    return out;
}

BareState build_bare_state(const Circuit& circ) {
    // The PARITY-NATIVE build is the SOLE bare-state path (the legacy GradedPoly sweep/substitute/
    // assemble path + its QEC_LEGACY_BARE_STATE opt-out were deleted in C4-Int 6).
    return build_bare_state_parity(circ);
}

FramedBareState build_bare_state_framed(const Circuit& circ) {
    // Source the bare state through the SAME (CSS) builder, then project to a FramedSuperposition at the
    // shared conversion point. build_bare_state is UNCHANGED — dem_export / ref_compile keep
    // consuming its CanonicalStabSum return. The projection is the only added step, so the lean
    // state is byte-identical to FramedSuperposition::from_css(build_bare_state(circ).state).
    BareState bs = build_bare_state(circ);
    FramedBareState out;
    out.rejected = bs.rejected;
    out.reject_gate_index = bs.reject_gate_index;
    out.reject_reason = bs.reject_reason;
    out.chi = bs.chi;
    if (!bs.rejected) out.state = FramedSuperposition::from_css(bs.state);
    return out;
}

}  // namespace qeccore
