#include "qeccore/factored_stab.hpp"
#include <cstddef>
#include <atomic>

#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>

#include "qeccore/exact_phase.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/pauli_kernels.hpp"    // shared single_pauli
#include "qeccore/ray_stab_bits.hpp"
#include "qeccore/ref_io.hpp"            // materialize_rays
#include "qeccore/stab_affine.hpp"
#include "qeccore/stab_disentangle.hpp"  // disentangle_from_generators, apply_disentangle_gate
#include "qeccore/stab_generators.hpp"   // stabilizer_generators

namespace qeccore {

namespace { uint64_t next_fbs_uid() {
    static std::atomic<uint64_t> c{1};
    return c.fetch_add(1, std::memory_order_relaxed);
} }

// single_pauli (i^phase·X^x·Z^z, axis: 0=X,1=Y,2=Z) is the shared kernel from
// qeccore/pauli_kernels.hpp (was a file-local copy here).

// ── factorize ────────────────────────────────────────────────────────────────────────────
//
// 1. Classify A-qubits: a qubit q is in A iff single_pauli(axis,q) reduces to 0 against the
//    ray-stabiliser RREF basis (i.e. the single-qubit Pauli is in the stabiliser group). The
//    sign is read from born_probabilities of that axis Pauli (p+~1 -> +, p-~1 -> -).
// 2. Rotate every A-qubit to the +Z axis on a CLONE (single-qubit Cliffords; exact,
//    chi-independent, phase-preserving): X->H; Y->Sdg,H. Then read <Z_q> on the clone and
//    flip with X if it is -Z. After this the clone equals |0...0>_A (x) |block>_B EXACTLY.
// 3. Restrict each materialised ray to B (clear A-support via the +Z_q stabilisers, drop A
//    columns, rebuild a |B|-qubit AffineState from the B-only generators), and PHASE-FIX each
//    restricted ray to the original ray's amplitude at the matching point so from_rays
//    reconstructs the exact branch coefficients. block = from_rays(|B|, restricted, coeffs).
FactoredBareState factorize(const CanonicalStabSum& s) {
    const int n = s.n();
    const int W = (n + 63) / 64;
    if (s.chi() > 2)
        throw std::runtime_error("factorize: chi > 2 (no ray-stabiliser basis)");

    FactoredBareState f;
    f.uid = next_fbs_uid();
    f.n = n;
    f.cls.assign(n, -1);

    // ── 1. classify A ──
    // A qubit q tensor-factors out of the whole chi<=2 sum iff a single-qubit Pauli P_q fixes
    // BOTH branch rays with the SAME sign (a "branch-preserving common-sign stabiliser"). Two
    // subtleties make the plain ray_stabiliser_bits test WRONG here:
    //   (i)  ray_stabiliser_bits includes the SWAP coset element (a Pauli that SWAPS the two
    //        branches). A P_q reachable only with the swap element does NOT fix each ray, so we
    //        must test membership against the BRANCH-PRESERVING subgroup ONLY (no swap).
    //   (ii) Even a branch-preserving common-sign stabiliser P_q can coincide on q while the
    //        BRANCH TRANSLATE D = prod_{a: sigma0!=sigma1} d_a touches q (opposite Z signs while
    //        +Y is common) — then the rays are different single-qubit states on q. So also
    //        require D trivial on q.
    // Both rejections only push qubits into B (safe; correctness over tightness).
    s.ensure_frame_current();
    const int chi = s.chi();
    // branch translate D (chi==2); identity for chi==1
    Pauli Dtrans(n);
    if (chi == 2) {
        const auto& s0 = s.branches[0].sigma;
        const auto& s1 = s.branches[1].sigma;
        for (size_t d = 0; d < s.free.size(); ++d) {
            const uint8_t b0 = d < s0.size() ? s0[d] : 0, b1 = d < s1.size() ? s1[d] : 0;
            if (b0 ^ b1) Dtrans = Pauli::multiply(Dtrans, s.U.Xrow[s.free[d]]);
        }
    }
    // Branch-preserving membership basis (NO swap coset): the bits of the generators g_a that
    // commute with D; pairs of D-anticommuters are combined so their product commutes. This is
    // exactly ray_stabiliser_bits' first loop, WITHOUT the trailing swap-element add.
    BitRref basis;
    basis.W2 = 2 * W;
    if (chi == 1) {
        for (int a = 0; a < n; ++a) basis.add(pauli_bits(s.U.Zrow[a], W));
    } else {
        int m1 = -1;
        for (int a = 0; a < n; ++a) {
            if (Pauli::anticommute_bit(s.U.Zrow[a], Dtrans)) {
                if (m1 < 0) { m1 = a; continue; }
                basis.add(pauli_bits(Pauli::multiply(s.U.Zrow[a], s.U.Zrow[m1]), W));
            } else {
                basis.add(pauli_bits(s.U.Zrow[a], W));
            }
        }
    }
    // D's single-qubit X/Z bits on q (its local action). q factors out iff D's local action on
    // q COMMUTES with the chosen axis P_q: a commuting D-action (e.g. D=Z on a Z-product q) only
    // rephases the branch and leaves q's single-qubit STATE identical across rays; an
    // anticommuting D-action (e.g. D=X on a Z-product q) flips q between the rays -> not a
    // product. (This is the relaxation that keeps Z-definite product qubits with a Z-only D in
    // A — matching the cultivation A-group size.)
    for (int q = 0; q < n; ++q) {
        const bool Dx = Dtrans.xbit(q), Dz = Dtrans.zbit(q);
        int found_axis = -1;
        for (int axis = 2; axis >= 0; --axis) {   // try Z, then X, then Y (cosmetic order)
            // P_q's local (x,z) bits: X=(1,0) Y=(1,1) Z=(0,1). D|_q anticommutes with P_q iff
            // (Px & Dz) ^ (Pz & Dx) == 1.
            const int Px = (axis == 0 || axis == 1) ? 1 : 0;
            const int Pz = (axis == 1 || axis == 2) ? 1 : 0;
            if (((Px && Dz) ^ (Pz && Dx)))     // D|_q anticommutes with this axis -> not product
                continue;
            Pauli P = single_pauli(axis, q, n);
            std::vector<uint64_t> v = pauli_bits(P, W);
            basis.reduce(v);
            bool in_group = true;
            for (uint64_t w : v) if (w) { in_group = false; break; }
            if (in_group) { found_axis = axis; break; }
        }
        if (found_axis < 0) continue;             // q is entangled -> B
        // sign of the axis Pauli on s
        Pauli P = single_pauli(found_axis, q, n);
        auto pp = s.born_probabilities(P);
        uint8_t sign = (pp.first > pp.second) ? 0 : 1;   // p+ dominant -> +, else -
        f.A.push_back({q, (uint8_t)found_axis, sign});
    }

    // block_to_global / cls
    for (int q = 0; q < n; ++q) {
        bool inA = false;
        for (const auto& a : f.A) if (a.q == q) { inA = true; break; }
        if (!inA) { f.cls[q] = (int)f.block_to_global.size(); f.block_to_global.push_back(q); }
    }
    const int nB = (int)f.block_to_global.size();

    // ── 2. rotate A-qubits to +Z on a clone ──
    CanonicalStabSum s2 = s.clone();
    for (auto& a : f.A) {
        if (a.axis == 0) {            // X -> Z
            s2.apply_h(a.q);
        } else if (a.axis == 1) {     // Y -> Z  (Sdg then H, matches pauli_project's Y form)
            s2.apply_sdg(a.q);
            s2.apply_h(a.q);
        }
        // read the post-rotation sign and flip to +Z if needed
        Pauli Zq = single_pauli(2, a.q, n);
        auto pp = s2.born_probabilities(Zq);
        if (pp.second > pp.first) s2.apply_x(a.q);   // -Z -> +Z
    }

    // ── 3. restrict each ray to B with exact phase ──
    auto rays = materialize_rays(s2);
    std::vector<std::unique_ptr<AffineState>> bRays;
    bRays.reserve(chi);
    std::vector<std::complex<double>> coeffs;
    coeffs.reserve(chi);
    s2.ensure_frame_current();

    // A reference "A=0" bit pad (the rotated A-qubits are +Z => value 0 in every ray).
    for (int i = 0; i < chi; ++i) {
        const AffineState& ray = *rays[i];

        // (a) B-only stabiliser generators of this ray. Every A-qubit is +Z in s2, so no ray
        //     generator can carry X/Y on an A-qubit (that would anticommute with +Z_q in the
        //     abelian group). Z-on-A acts as +1 on |0>_A and contributes no eigenvalue, so the
        //     exact B-restriction of a generator is "drop the A columns" (phase preserved). The
        //     n restricted generators span a rank-|B| group (the |A| A-stabilisers map to
        //     identity); extract a |B|-element independent set via a phase-tracking RREF.
        std::vector<Pauli> gens = stabilizer_generators(ray);
        std::vector<Pauli> restricted;            // n B-Paulis (A columns dropped)
        restricted.reserve(gens.size());
        for (auto& g : gens) {
            for (int q = 0; q < n; ++q)
                if (f.cls[q] == -1 && g.xbit(q))
                    throw std::runtime_error("factorize: X/Y on A-qubit in a generator "
                                             "(A not actually decoupled)");
            Pauli gb(nB);
            for (int q = 0; q < n; ++q) {
                int c = f.cls[q];
                if (c == -1) continue;            // A column dropped (Z-on-A acts as +1)
                if (g.xbit(q)) gb.setx(c);
                if (g.zbit(q)) gb.setz(c);
            }
            // g is Hermitian with phase = (#Y + 2*[sign-]) mod 4; A carries no Y (no X on A)
            // and Z-on-A acts as +1 on |0>_A, so #Y and the eigenvalue sign are unchanged ->
            // the B-restricted Pauli keeps g.phase verbatim.
            gb.phase = g.phase;
            restricted.push_back(std::move(gb));
        }
        // phase-tracking GF(2) RREF over (x|z) bits of the B-Paulis to pull out nB independent
        // generators (Pauli::multiply keeps the eigenvalue sign exact).
        std::vector<Pauli> bgens;
        std::vector<int> pivot_bit;             // unified bit: [0,nB)=x_q, [nB,2nB)=z_q
        auto low_bit = [&](const Pauli& P) -> int {
            for (int q = 0; q < nB; ++q) if (P.xbit(q)) return q;
            for (int q = 0; q < nB; ++q) if (P.zbit(q)) return nB + q;
            return -1;
        };
        auto bit_set = [&](const Pauli& P, int b) -> bool {
            return (b < nB) ? P.xbit(b) : P.zbit(b - nB);
        };
        for (auto& P : restricted) {
            Pauli cur = P;
            for (size_t r = 0; r < bgens.size(); ++r) {
                if (bit_set(cur, pivot_bit[r])) cur = Pauli::multiply(cur, bgens[r]);
            }
            int lb = low_bit(cur);
            if (lb < 0) continue;             // reduced to identity (an A-stabiliser image)
            for (size_t r = 0; r < bgens.size(); ++r)   // back-eliminate to keep RREF
                if (bit_set(bgens[r], lb)) bgens[r] = Pauli::multiply(bgens[r], cur);
            bgens.push_back(cur);
            pivot_bit.push_back(lb);
        }
        if ((int)bgens.size() != nB)
            throw std::runtime_error("factorize: B-generator rank " +
                                     std::to_string(bgens.size()) + " != |B| " +
                                     std::to_string(nB));

        // (b) build the |B|-qubit AffineState from bgens (disentangle -> invert -> apply to |0>)
        auto bray = std::make_unique<AffineState>(nB);   // |0...0>_B
        if (nB > 0) {
            std::vector<DisentangleGate> dis = disentangle_from_generators(bgens);
            for (auto it = dis.rbegin(); it != dis.rend(); ++it) {
                DisentangleGate g = *it;
                if (g.op == 1) g.op = 2;            // S -> SDG (invert)
                else if (g.op == 2) g.op = 1;       // SDG -> S
                apply_disentangle_gate(*bray, g);
            }
        }

        // (c) phase-fix: match this restricted ray's amplitude to the original ray's amplitude
        //     at the SAME B point (with A=0). Pick an on-support B point of the restricted ray
        //     (its affine offset b), embed into n qubits (A=0), compare exact amplitudes.
        std::vector<uint8_t> bpt(nB, 0);
        for (int c = 0; c < nB; ++c) bpt[c] = bray->b[c];   // on-support: y=0 -> idx=b
        ExactPhase cur = bray->amplitude_at_bits(bpt);
        std::vector<uint8_t> npt(n, 0);
        for (int c = 0; c < nB; ++c) npt[f.block_to_global[c]] = bpt[c];   // A stays 0
        ExactPhase tgt = ray.amplitude_at_bits(npt);
        if (cur.is_zero || tgt.is_zero)
            throw std::runtime_error("factorize: phase-fix reference point off support");
        // both are unit-magnitude eigenstate amplitudes scaled equally; correction is a pure
        // zeta8 phase z8(tgt) - z8(cur).
        int dz8 = ((tgt.z8 - cur.z8) % 8 + 8) % 8;
        // sanity: magnitudes (scale) must match for a pure-phase correction.
        if (tgt.scale != cur.scale)
            throw std::runtime_error("factorize: phase-fix magnitude mismatch (scale)");
        bray->omega.add_z8(dz8);

        bRays.push_back(std::move(bray));
        coeffs.push_back(s2.branches[i].c);
    }

    // Build the exact |B|-qubit restriction (affine), then project ONCE to the lean rep — the frame
    // is BIT-IDENTICAL; only the affine anchor (unused past load) is dropped. f.block stores lean.
    CanonicalStabSum css = CanonicalStabSum::from_rays(nB, std::move(bRays), std::move(coeffs));
    css.ensure_frame_current();
    f.block = FramedSuperposition::from_css(css);
    return f;
}

// ── factorize_framed ────────────────────────────────────────────────────────────────────────
//
// The lean-frame twin of factorize(), general chi. Works purely on the FramedSuperposition frame
// (U/eps/free/branches):
//   Phase 1 (classify A): a qubit q factors out iff a single-qubit Pauli P_q fixes EVERY branch
//     ray with the SAME eigenvalue — i.e. P_q is a common-sign member of the BRANCH-PRESERVING
//     stabiliser subgroup: the joint commutant of the branch TRANSLATES inside the generator
//     span. The translates are D^x = ∏_{d:x_d} d_{free[d]} for x in the GF(2) span of the
//     sigma-DIFFERENCES {σ_i ⊕ σ_0} (rank r' ≤ r; r'=1 with a single Dtrans is the old chi=2
//     case, r'=0 the chi=1 case). Sign read via framed_expectation.
//   Phase 2 (rotate A→+Z): single-qubit Clifford FRAME conjugation on a FramedSuperposition clone (left_h /
//     left_sdg+left_h / left_x on L2.U; eps/free/branches are untouched by a Clifford — flush_gates
//     only conjugates the frame rows), reading the post-rotation sign with framed_expectation.
//   Phase 3 (restrict to B): the A-qubits are now +Z product (generator +Z_q, destabiliser X_q), so
//     drop the A columns from the frame generators (a phase-tracking GF(2) RREF mirroring affine
//     Step 3a), carrying eps through the row ops; the free generators are B-supported (A is product)
//     and become the trailing block generators (branches' sigma copied VERBATIM, no phase-fix).
//     Phase 3's assembly and the γ_d coefficient phase-fix were already chi-general.
FactoredBareState factorize_framed(const FramedSuperposition& bare) {
    const int n = bare.n();
    const int W = (n + 63) / 64;

    FactoredBareState f;
    f.uid = next_fbs_uid();
    f.n = n;
    f.cls.assign(n, -1);

    // ── 1. classify A ──
    // A qubit q tensor-factors out of the sum iff a single-qubit Pauli P_q fixes EVERY branch
    // ray with the SAME sign (a "branch-preserving common-sign stabiliser"). Two subtleties make
    // the plain ray_stabiliser_bits test WRONG here (inherited verbatim from the chi=2 case):
    //   (i)  a P_q reachable only WITH a branch-swapping coset element (a Pauli that permutes the
    //        branch rays) does NOT fix each ray, so membership must be tested against the
    //        BRANCH-PRESERVING subgroup ONLY: the joint commutant of every branch translate.
    //        (g commutes with translate T ⇒ g T|ψ0⟩ = T g|ψ0⟩ ⇒ the SAME eigenvalue on every
    //        branch; an anticommuting g flips sign between branches.)
    //   (ii) P_q must also act consistently on q across the rays: each translate's LOCAL action
    //        on q must COMMUTE with the chosen axis. A commuting local action (e.g. D=Z on a
    //        Z-product q) only rephases the branch and leaves q's single-qubit STATE identical
    //        across rays; an anticommuting local action (e.g. D=X on a Z-product q) flips q
    //        between the rays -> not a product. (Since P_q is supported ONLY on q, this is
    //        implied by joint-commutant membership — kept as the cheap pre-filter it always was;
    //        the relaxation from "D trivial on q" keeps Z-definite product qubits with a Z-only
    //        D in A, matching the cultivation A-group size.)
    // Both rejections only push qubits into B (safe; correctness over tightness).
    const int chi = bare.chi();
    const int r = (int)bare.free.size();
    // Branch translates: one Pauli per GF(2)-basis vector of span{σ_i ⊕ σ_0}. Using the
    // sigma-differences (not all r free generators) keeps the test exact for present branches
    // only, and reduces bit-identically to the single Dtrans = D^{σ0⊕σ1} at chi==2 (canonical
    // states have full diff-rank r, so the two coincide there).
    std::vector<Pauli> trans;
    if (chi >= 2) {
        std::vector<std::vector<uint8_t>> diff_basis;   // forward-eliminated, leading-bit pivots
        std::vector<int> pivots;
        const auto& s0 = bare.sigma(0);
        for (int i = 1; i < chi; ++i) {
            const auto& si = bare.sigma(i);
            std::vector<uint8_t> d(r, 0);
            for (int dd = 0; dd < r; ++dd) {
                const uint8_t b0 = dd < (int)s0.size() ? s0[dd] : 0;
                const uint8_t bi = dd < (int)si.size() ? si[dd] : 0;
                d[dd] = b0 ^ bi;
            }
            for (size_t k = 0; k < diff_basis.size(); ++k)
                if (d[pivots[k]])
                    for (int dd = 0; dd < r; ++dd) d[dd] ^= diff_basis[k][dd];
            int lb = -1;
            for (int dd = 0; dd < r; ++dd) if (d[dd]) { lb = dd; break; }
            if (lb < 0) continue;                       // dependent difference
            diff_basis.push_back(std::move(d));
            pivots.push_back(lb);
        }
        trans.reserve(diff_basis.size());
        for (const auto& b : diff_basis) {
            Pauli T(n);
            for (int dd = 0; dd < r; ++dd)
                if (b[dd]) T = Pauli::multiply(T, bare.U.Xrow[bare.free[dd]]);
            trans.push_back(std::move(T));
        }
    }
    // Branch-preserving membership basis (NO swap coset): the joint commutant of every translate
    // inside the generator span, built by ONE symplectic pairing pass PER translate (pairs of
    // T-anticommuters are combined so their product commutes; each pass preserves commutation
    // with the earlier translates since it only multiplies survivors). r'=1 is exactly the old
    // chi=2 pairing; r'=0 (chi=1) keeps all n generators.
    std::vector<Pauli> work;
    work.reserve(n);
    for (int a = 0; a < n; ++a) work.push_back(bare.U.Zrow[a]);
    for (const Pauli& T : trans) {
        std::vector<Pauli> next;
        next.reserve(work.size());
        int m1 = -1;
        for (int a = 0; a < (int)work.size(); ++a) {
            if (Pauli::anticommute_bit(work[a], T)) {
                if (m1 < 0) { m1 = a; continue; }
                next.push_back(Pauli::multiply(work[a], work[m1]));
            } else {
                next.push_back(work[a]);
            }
        }
        work.swap(next);
    }
    BitRref basis;
    basis.W2 = 2 * W;
    for (const auto& g : work) basis.add(pauli_bits(g, W));
    for (int q = 0; q < n; ++q) {
        int found_axis = -1;
        for (int axis = 2; axis >= 0; --axis) {   // try Z, then X, then Y (cosmetic order)
            const int Px = (axis == 0 || axis == 1) ? 1 : 0;
            const int Pz = (axis == 1 || axis == 2) ? 1 : 0;
            // pre-filter (ii): EVERY translate's local action on q must commute with this axis.
            bool anti = false;
            for (const Pauli& T : trans) {
                const bool Dx = T.xbit(q), Dz = T.zbit(q);
                if ((Px && Dz) ^ (Pz && Dx)) { anti = true; break; }
            }
            if (anti) continue;                // some D|_q anticommutes with this axis -> not product
            Pauli P = single_pauli(axis, q, n);
            std::vector<uint64_t> v = pauli_bits(P, W);
            basis.reduce(v);
            bool in_group = true;
            for (uint64_t w : v) if (w) { in_group = false; break; }
            if (in_group) { found_axis = axis; break; }
        }
        if (found_axis < 0) continue;             // q is entangled -> B
        // sign of the axis Pauli on the lean state (P is a stabiliser => e ≈ ±1).
        Pauli P = single_pauli(found_axis, q, n);
        auto pp = framed_expectation(bare, P);
        double e = pp.first - pp.second;
        uint8_t sign = (e > 0) ? 0 : 1;
        f.A.push_back({q, (uint8_t)found_axis, sign});
    }

    // block_to_global / cls (identical to factorize).
    for (int q = 0; q < n; ++q) {
        bool inA = false;
        for (const auto& a : f.A) if (a.q == q) { inA = true; break; }
        if (!inA) { f.cls[q] = (int)f.block_to_global.size(); f.block_to_global.push_back(q); }
    }
    const int nB = (int)f.block_to_global.size();

    // ── 2. rotate A-qubits to +Z on a FramedSuperposition clone (frame conjugation; eps/free/branches stay) ──
    FramedSuperposition L2 = bare;
    for (auto& a : f.A) {
        if (a.axis == 0) {            // X -> Z
            L2.U.left_h(a.q);
        } else if (a.axis == 1) {     // Y -> Z  (Sdg then H, matching factorize)
            L2.U.left_sdg(a.q);
            L2.U.left_h(a.q);
        }
        // post-rotation sign; flip to +Z with X if needed.
        Pauli Zq = single_pauli(2, a.q, n);
        auto pp = framed_expectation(L2, Zq);
        if (pp.second > pp.first) L2.U.left_x(a.q);   // -Z -> +Z
    }

    // ── 3. restrict the frame to B (drop A columns; phase-tracking RREF carrying eps) ──
    // B-restrict a single +-form frame generator g (with sign e): assert no X/Y on A (a generator
    // commutes with +Z_q for every q∈A — the abelian group — so it carries NO X/Y on A; an X/Y here
    // means A is not actually decoupled), then drop the A columns. Z-on-A acts as +1 on |0>_A (every
    // A-qubit is +Z after Phase 2), so dropping it is a +1 no-op — it carries no eigenvalue and no Y,
    // so the +-form phase and the sign e are preserved. (A free generator may carry Z-on-A; it folds
    // to +1 identically, so the B-restriction is gauge-clean WITHOUT any A-support throw.)
    auto restrict_to_B = [&](const Pauli& g, uint8_t e, Pauli& gb, uint8_t& eb) {
        gb = Pauli(nB);
        for (int q = 0; q < n; ++q) {
            const int c = f.cls[q];
            if (c == -1) {                       // A column
                if (g.xbit(q))
                    throw std::runtime_error("factorize_framed: X/Y on A-qubit in a generator "
                                             "(A not actually decoupled)");
                continue;                        // drop A column (Z-on-A => +1 on |0>_A)
            }
            if (g.xbit(q)) gb.setx(c);
            if (g.zbit(q)) gb.setz(c);
        }
        gb.phase = g.phase;                      // #Y unchanged (no Y on A) => +-form preserved
        eb = e;
    };

    // free generator -> B-block index (in bare.free order). bare.free entries are distinct frame
    // generator indices; map global free-position d to the B-restricted generator + eps.
    // (r = bare.free.size() from Phase 1.)
    static thread_local std::vector<uint8_t> is_free; is_free.assign(n, 0);
    for (int d = 0; d < r; ++d) is_free[bare.free[d]] = 1;

    // (a) common (non-free) generators: B-restrict + phase-tracking RREF -> (nB - r) generators.
    std::vector<Pauli> common_gens;
    std::vector<uint8_t> common_eps;
    std::vector<int> pivot_bit;                 // unified bit: [0,nB)=x_c, [nB,2nB)=z_c
    auto low_bit = [&](const Pauli& P) -> int {
        for (int c = 0; c < nB; ++c) if (P.xbit(c)) return c;
        for (int c = 0; c < nB; ++c) if (P.zbit(c)) return nB + c;
        return -1;
    };
    auto bit_set = [&](const Pauli& P, int b) -> bool {
        return (b < nB) ? P.xbit(b) : P.zbit(b - nB);
    };
    for (int a = 0; a < n; ++a) {
        if (is_free[a]) continue;               // free generators handled in (b)
        Pauli gb; uint8_t eb;
        restrict_to_B(L2.U.Zrow[a], L2.eps[a], gb, eb);
        // reduce against the running RREF basis (carrying eps via XOR).
        Pauli cur = gb; uint8_t cur_e = eb;
        for (size_t k = 0; k < common_gens.size(); ++k) {
            if (bit_set(cur, pivot_bit[k])) {
                cur = Pauli::multiply(cur, common_gens[k]);
                cur_e ^= common_eps[k];
            }
        }
        int lb = low_bit(cur);
        if (lb < 0) continue;                   // reduced to identity (an A-stabiliser image)
        for (size_t k = 0; k < common_gens.size(); ++k)   // back-eliminate to keep RREF
            if (bit_set(common_gens[k], lb)) {
                common_gens[k] = Pauli::multiply(common_gens[k], cur);
                common_eps[k] ^= cur_e;
            }
        common_gens.push_back(std::move(cur));
        common_eps.push_back(cur_e);
        pivot_bit.push_back(lb);
    }
    if ((int)common_gens.size() != nB - r)
        throw std::runtime_error("factorize_framed: common B-generator rank " +
                                 std::to_string(common_gens.size()) + " != nB-r " +
                                 std::to_string(nB - r));

    // (b) free generators: B-restrict in bare.free order. These are the DISTINGUISHING generators;
    //     their eigenvalue PATTERN across branches (sigma) is gauge-free, so it copies VERBATIM.
    std::vector<Pauli> free_gens(r);
    std::vector<uint8_t> free_eps(r);
    for (int d = 0; d < r; ++d) {
        Pauli gb; uint8_t eb;
        restrict_to_B(L2.U.Zrow[bare.free[d]], L2.eps[bare.free[d]], gb, eb);
        free_gens[d] = std::move(gb);
        free_eps[d] = eb;
    }

    // (c) assemble the |B|-qubit lean block: Zrow = [common..., free...] so block.free is the
    //     trailing r indices and branches' sigma (indexed by free-position d) is copied VERBATIM.
    FramedSuperposition block(nB);
    if (nB > 0) {
        std::vector<Pauli> all_gens;
        all_gens.reserve(nB);
        for (auto& g : common_gens) all_gens.push_back(g);
        for (auto& g : free_gens) all_gens.push_back(g);
        block.U = CliffordTableau::from_generators(all_gens);   // asserts rank == nB (independence)
        block.eps.assign(nB, 0);
        for (int i = 0; i < nB - r; ++i) block.eps[i] = common_eps[i];
        for (int d = 0; d < r; ++d) block.eps[(nB - r) + d] = free_eps[d];
        block.free.resize(r);
        for (int d = 0; d < r; ++d) block.free[d] = (nB - r) + d;
        auto& be = block.entries();
        be.resize(chi);
        for (int i = 0; i < chi; ++i) {
            be[i].first = bare.sigma(i);     // VERBATIM (indexes free-position d)
            be[i].second = bare.coeff(i);    // bare-gauge amplitude (phase-fixed below)
        }
        block.sync_alpha_k();
        // ── branch coefficient PHASE-FIX (the only residual gauge freedom) ──
        // The generators/eps/free/sigma are EXACT, so the block frame fixes the reference state and
        // the branch eigenvalue patterns. The branch RAYS are reached by the free DESTABILISERS
        //   d_d^block = block.U.Xrow[(nB-r)+d]   (synthesized by from_generators)
        // whereas `bare`'s branch coeffs are expressed in `bare`'s destabiliser gauge
        //   d_d^bare = restrict_to_B(L2.U.Xrow[bare.free[d]])  (the original translate).
        // For branch i with pattern x = σ_i, both PRODUCTS D^x = ∏_{d:x_d} d_d translate |ref> to
        // the SAME physical branch ray (the destabiliser rows commute pairwise on either side, and
        // restriction preserves commutation — A columns are Z-only), so they are PROPORTIONAL:
        //   D_block^x |ref> = Γ_x · D_bare^x |ref>,   Γ_x = <ref| D_bare^x · D_block^x |ref> ∈ {1,i,-1,-i}.
        // N_x = D_bare^x · D_block^x is a stabiliser-group element, so Γ_x = its eigenvalue on
        // |ref> = i^{N.phase} · ∏_{generators in N's expansion} (-1)^eps. To keep block representing
        // the SAME state, c_i^block = c_i^bare · conj(Γ_{σ_i}).
        // GENERAL-chi NOTE: Γ_x is NOT ∏_{d:x_d} γ_d in general — the per-d gauge mismatches N_d
        // need not commute with the OTHER bare destabilisers in x, so the product picks up
        // reordering signs. Invisible at chi<=2 (|x| <= 1, where the per-branch read reduces to the
        // former per-d γ_d bit-identically); computed per BRANCH here.
        // (No anchor / no materialize_rays — this is a closed-form frame computation, NOT the affine
        // per-ray amplitude phase-fix.)
        std::vector<Pauli> bare_tr(r);                             // restricted bare translates
        for (int d = 0; d < r; ++d) {
            uint8_t de;
            restrict_to_B(L2.U.Xrow[bare.free[d]], 0, bare_tr[d], de);
        }
        for (int i = 0; i < chi; ++i) {
            // D_bare^x and D_block^x for this branch's pattern (ascending d; order-free, the rows
            // commute pairwise).
            Pauli Drx(nB), Dbx(nB);
            for (int d = 0; d < r; ++d) {
                if (!block.sigma_bit(i, d)) continue;
                Drx = Pauli::multiply(Drx, bare_tr[d]);
                Dbx = Pauli::multiply(Dbx, block.U.Xrow[(nB - r) + d]);
            }
            // N = D_bare^x · D_block^x; read its reference eigenvalue Γ_x by expressing N over the
            // block generators g_a = Zrow[a] (sign (-1)^eps[a]): bit a is set iff N anticommutes
            // with Xrow[a]  (⟨N, d_a⟩ = coefficient of g_a in N's stabiliser expansion).
            Pauli N = Pauli::multiply(Drx, Dbx);
            std::complex<double> g(0, 0);
            {
                static const std::complex<double> ip[4] = {{1,0},{0,1},{-1,0},{0,-1}};
                int sgn = 0;
                Pauli acc(nB);                       // rebuild ∏ g_a to verify N == ± that product
                for (int a = 0; a < nB; ++a) {
                    if (Pauli::anticommute_bit(N, block.U.Xrow[a])) {
                        acc = Pauli::multiply(acc, block.U.Zrow[a]);
                        if (block.eps[a]) sgn ^= 1;
                    }
                }
                // N and acc must be the SAME Pauli up to a phase (N maps |ref> to itself up to a
                // phase => it is a stabiliser-group element with this exact generator expansion). If the
                // bits disagree the proportionality assumption is violated (a structural bug) — fail loud.
                for (int w = 0; w < (int)N.x.size(); ++w)
                    if (N.x[w] != acc.x[w] || N.z[w] != acc.z[w])
                        throw std::runtime_error("factorize_framed: branch translate not proportional "
                                                 "(free destabiliser gauge inconsistency)");
                // their ratio is i^{N.phase-acc.phase}.
                int dph = ((N.phase - acc.phase) % 4 + 4) % 4;
                g = ip[dph] * (sgn ? -1.0 : 1.0);
            }
            block.coeff(i) *= std::conj(g);
        }
        // Re-select a minimal independent `free` basis (folds away any redundant distinguishing gens).
        framed_canonicalise(block);
    } else {
        // nB == 0: a fully product state. chi must be 1 (no B to carry branches).
        block.eps.clear();
        block.free.clear();
        auto& be = block.entries();
        be.resize(chi);
        for (int i = 0; i < chi; ++i) { be[i].first.clear(); be[i].second = bare.coeff(i); }
        block.sync_alpha_k();
    }
    f.block = std::move(block);
    return f;
}

// ── Phase 2: per-shot reduced bad-measurement ──────────────────────────────────────────────
// The active-set + reduced-read + framed_active_block machinery (active_set / active_set_into /
// build_active_set_work / reduced_read / build_a_lookup / framed_active_block and the file-local
// conjugate_by_s / reduced_read_body helpers) lives in factored_active_block.cpp. All declarations
// stay in factored_stab.hpp so callers are unaffected.

}  // namespace qeccore
