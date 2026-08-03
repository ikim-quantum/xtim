#include "qeccore/stab_generators.hpp"

#include <complex>
#include <stdexcept>

#include "qeccore/stab_affine.hpp"
#include "qeccore/stab_overlap.hpp"

namespace qeccore {

namespace {

// Pin the constant phase of a candidate generator EMPIRICALLY.
// g already carries the correct X-part and Z-part (phase 0 on entry). We
// measure ip0 = <ψ|g|ψ> on a clone; if the X/Z parts are right this is a
// unit-modulus phase i^p (p∈{0,1,2,3}). Set g.phase = (4−p)%4 so that the
// generator i^{phase}·X^x·Z^z is a +1 stabilizer: i^{phase}·ip0 = 1.
void pin_phase(Pauli& g, const AffineState& psi) {
    // ⟨ψ|g|ψ⟩ read OVERLAP-FREE as a single shared-amplitude ratio: g|ψ⟩ = i^p|ψ⟩ has the same
    // support as |ψ⟩, so (g|ψ⟩)(idx)/ψ(idx) at an on-support idx is the unit phase i^p. (Avoids
    // a pairwise inner_product so the closed-form measurement collapse stays overlap-free.)
    AffineState gp = psi;
    g.apply_to(gp);
    // On-support index = psi's affine offset b (the y=0 basis point). Read via the wide-index
    // amplitude_at_bits so this works for n > 64 (no uint64_t index ceiling).
    //
    // The ratio is taken EXACTLY in the ExactPhase representation (is_zero, √2-scale, ζ8
    // exponent) — never materialized as a double. A double would silently underflow the
    // common modulus 2^{-k/2} to 0.0 for k ≳ 2100 free variables (deferred circuits beyond
    // ~2100 qubits, e.g. a stock d=11, r≥7 surface-code memory) and mis-report a support
    // error; the exact form is scale-independent.
    const ExactPhase a0 = psi.amplitude_at_bits(psi.b);
    const ExactPhase a1 = gp.amplitude_at_bits(psi.b);
    // |c| == 1  ⟺  both amplitudes non-zero with equal √2-scale; otherwise the X/Z
    // bit-pattern is wrong (fix that, not the phase).
    if (a0.is_zero || a1.is_zero || a0.scale != a1.scale)
        throw std::logic_error(
            "stabilizer_generators: candidate has wrong X/Z support "
            "(|<ψ|g|ψ>| != 1): n=" + std::to_string(psi.n_) +
            " k=" + std::to_string(psi.k_) +
            " a0=(zero=" + std::to_string(a0.is_zero) +
            ",scale=" + std::to_string(a0.scale) +
            ") a1=(zero=" + std::to_string(a1.is_zero) +
            ",scale=" + std::to_string(a1.scale) + ")");
    // c = ζ8^{Δ}; a stabilizer eigenphase is a 4th root, so Δ must be even: c = i^{Δ/2}.
    const int dz = ((a1.z8 - a0.z8) % 8 + 8) % 8;
    if (dz & 1)
        throw std::logic_error(
            "stabilizer_generators: candidate eigenphase is not a 4th root "
            "(zeta8 exponent " + std::to_string(dz) + ")");
    g.phase = (4 - dz / 2) % 4;
}

// ---- Prepared phase pin -----------------------------------------------------
// pin_phase above is the amplitude-ratio ORACLE: exact and self-contained, but each
// call re-derives a fresh solve of R y = rhs (a dense byte-matrix Gaussian) — O(n·k)
// materialization + solve PER GENERATOR, the dominant reference-compile cost on wide
// deferred circuits (d=11 memory: n = k = 2316, 2n pins ≈ 47 s). stabilizer_generators
// pins every generator against the SAME state psi and has already paid for its
// PreparedAffine: G_θ with G_θ·R = I_k and H_θ with im(R) = ker(H_θ). That factors
// each pin into one packed solve + one ℤ₄ form evaluation:
//   a0 = ⟨b|ψ⟩    = ω_ψ · 2^{−k/2}                       (rhs = 0 ⇒ y = 0, Q(0) = 0)
//   a1 = ⟨b|g·ψ⟩  = ω_gψ · i^{Q'(y)} · 2^{−k/2},   y = G_θ·x the unique sol of R y = x
// where x = b ⊕ b' is the Pauli's X-translation. prepare() succeeding proves R has
// full column rank, so amplitude_at_bits' kernel is empty and its gauss_sum reduces
// to ζ8^{2·Q'(y)} — the formulas above are EXACTLY that path specialized to r = 0
// (test_stab_generators_pin cross-checks the +1-stabilizer property against
// amplitude_at_bits itself). a0 is computed ONCE via the oracle (it is g-independent).
struct PinPrepared {
    const AffineState& psi;
    const PreparedAffine& pa;
    AffineState scratch;           // = psi; per-pin restore of the Pauli-touched fields
    ExactPhase a0;
    std::vector<uint64_t> xw, yw;  // packed rhs (n bits) / solution (k bits)
    std::vector<int> ysupp;

    PinPrepared(const AffineState& s, const PreparedAffine& p)
        : psi(s), pa(p), scratch(s), a0(s.amplitude_at_bits(s.b)),
          xw((size_t)p.mw_, 0), yw(((size_t)s.k_ + 63) / 64, 0) {}

    void pin(Pauli& g) {
        // gψ = g|ψ⟩: a Pauli apply touches ONLY {b, D, omega} (apply_x / apply_z /
        // phase-in-omega) — restore those three and reuse scratch's R and J, which are
        // byte-owned copies of psi's that no pin ever mutates.
        scratch.b = psi.b;
        scratch.D = psi.D;
        scratch.omega = psi.omega;
        g.apply_to(scratch);
        // rhs x = b ⊕ b', packed. On-support ⟺ H_θ·x = 0 (im(R) = ker(H_θ)) — the
        // exact condition under which the oracle's gf2_solve finds a solution.
        std::fill(xw.begin(), xw.end(), 0);
        for (int q = 0; q < psi.n_; ++q)
            if (psi.b[q] ^ scratch.b[q]) xw[q >> 6] |= 1ull << (q & 63);
        for (int i = 0; i < pa.m_; ++i) {
            const uint64_t* hr = pa.Hrow(i);
            uint64_t acc = 0;
            for (int t = 0; t < pa.mw_; ++t) acc ^= hr[t] & xw[t];
            if (__builtin_parityll(acc))
                throw std::logic_error(
                    "stabilizer_generators: candidate has wrong X/Z support "
                    "(rhs off the state's affine support): n=" + std::to_string(psi.n_) +
                    " k=" + std::to_string(psi.k_) + " check_row=" + std::to_string(i));
        }
        // y = G_θ·x (the unique solution — full column rank).
        std::fill(yw.begin(), yw.end(), 0);
        ysupp.clear();
        for (int j = 0; j < psi.k_; ++j) {
            const uint64_t* gr = pa.Grow(j);
            uint64_t acc = 0;
            for (int t = 0; t < pa.mw_; ++t) acc ^= gr[t] & xw[t];
            if (__builtin_parityll(acc)) { yw[j >> 6] |= 1ull << (j & 63); ysupp.push_back(j); }
        }
        // c0 = Q'(y) = Σ_{j∈supp(y)} D'[j] + 2·#{J-edges inside supp(y)}   (mod 4).
        // Σ_a popcount(J_a & y) over the support counts every edge twice = the 2·#edges
        // term directly. J bits beyond dim are zero, so truncating to yw's words is exact.
        long long q4 = 0;
        for (int j : ysupp) q4 += scratch.D[j];
        const int tw = std::min(psi.J.words(), (int)yw.size());
        for (int a : ysupp) {
            const uint64_t* jr = psi.J.row(a);
            for (int t = 0; t < tw; ++t) q4 += __builtin_popcountll(jr[t] & yw[t]);
        }
        const int c0 = (int)(((q4 % 4) + 4) % 4);
        // a1 = ω_gψ · ζ8^{2c0}, scale −k — amplitude_at_bits' r = 0 result verbatim.
        ExactPhase a1 = scratch.omega;
        a1.add_z8((2 * c0) % 8);
        a1.scale -= psi.k_;
        // Identical accept/reject arithmetic to the oracle.
        if (a0.is_zero || a1.is_zero || a0.scale != a1.scale)
            throw std::logic_error(
                "stabilizer_generators: candidate has wrong X/Z support "
                "(|<psi|g|psi>| != 1): n=" + std::to_string(psi.n_) +
                " k=" + std::to_string(psi.k_) +
                " a0=(zero=" + std::to_string(a0.is_zero) +
                ",scale=" + std::to_string(a0.scale) +
                ") a1=(zero=" + std::to_string(a1.is_zero) +
                ",scale=" + std::to_string(a1.scale) + ")");
        const int dz = ((a1.z8 - a0.z8) % 8 + 8) % 8;
        if (dz & 1)
            throw std::logic_error(
                "stabilizer_generators: candidate eigenphase is not a 4th root "
                "(zeta8 exponent " + std::to_string(dz) + ")");
        g.phase = (4 - dz / 2) % 4;
    }
};

}  // namespace

std::vector<Pauli> stabilizer_generators(const StabState& s) {
    const AffineState* aff = dynamic_cast<const AffineState*>(&s);
    if (!aff)
        throw std::logic_error(
            "stabilizer_generators: only the affine representation is "
            "supported");

    const int n = aff->n_;
    const int k = aff->k_;
    PreparedAffine pa = PreparedAffine::prepare(*aff);
    PinPrepared pin(*aff, pa);   // shares pa + the state across all n pins

    std::vector<Pauli> gens;
    gens.reserve(n);

    // ---- Z-type generators: the n−k parity-check (left null-space) rows of R.
    // Each Hth row is an n-bit Z-support; X-part is 0.
    for (int i = 0; i < pa.m_; ++i) {
        Pauli g(n);
        const uint64_t* hr = pa.Hrow(i);
        for (int q = 0; q < n; ++q)
            if ((hr[q >> 6] >> (q & 63)) & 1ULL) g.setz(q);
        pin.pin(g);
        gens.push_back(std::move(g));
    }

    // ---- X-type generators: one per free column l of R.
    // Applying X^{a_l} (a_l = column l of R) shifts y → y⊕e_l, changing the
    // amplitude phase by  Q(y⊕e_l) − Q(y) ≡ D_l + 2·(Σ_j J(l,j) y_j − D_l y_l)
    // (mod 4).  The constant D_l is a global phase (pinned empirically). The
    // y-dependent part is 2·c·y, c = J_l ⊕ (D_l mod 2)·e_l, which a Z-operator
    // Z^{z_l} cancels iff (R^T z_l)·y ≡ c·y, i.e. R^T z_l = c.  Since
    // G_θ R = I_k ⇒ R^T G_θ^T = I_k, a solution is z_l = G_θ^T·c, i.e.
    //   z_l = (XOR over j with J(l,j)=1 of Grow(j)) ⊕ (D_l odd ? Grow(l) : 0).
    auto xor_grow_into = [&](Pauli& g, int j) {
        const uint64_t* gr = pa.Grow(j);
        for (int q = 0; q < n; ++q)
            if ((gr[q >> 6] >> (q & 63)) & 1ULL) g.flipz(q);
    };
    for (int l = 0; l < k; ++l) {
        Pauli g(n);
        // X-part = column l of R.
        for (int q = 0; q < n; ++q)
            if (aff->R.get(q, l)) g.setx(q);
        // Z-part = G_θ^T · (J row l ⊕ (D_l mod 2)·e_l).
        for (int j = 0; j < k; ++j)
            if (aff->J.get(l, j)) xor_grow_into(g, j);
        if (aff->D[l] & 1) xor_grow_into(g, l);
        pin.pin(g);
        gens.push_back(std::move(g));
    }

    return gens;
}

}  // namespace qeccore
