#include "qeccore/parity_support.hpp"

#include <cassert>

namespace qeccore {

// Apply one CliffGate to an AffineState (cprime_dagger gate semantics: the gate
// list builds |ψ⟩ = C'†|0⟩ when applied in order, [0] first).
static void apply_gate(AffineState& st, const CliffGate& g) {
    switch (g.kind) {
        case GateKind::H:   st.apply_h(g.targets[0]); break;
        case GateKind::S:   st.apply_s(g.targets[0]); break;
        case GateKind::SDG: st.apply_sdg(g.targets[0]); break;
        case GateKind::X:   st.apply_x(g.targets[0]); break;
        case GateKind::Y:   st.apply_y(g.targets[0]); break;
        case GateKind::Z:   st.apply_z(g.targets[0]); break;
        case GateKind::CX:  st.apply_cx(g.targets[0], g.targets[1]); break;
        case GateKind::CZ:  st.apply_cz(g.targets[0], g.targets[1]); break;
        default: assert(false && "restrict_to_support: unexpected gate kind"); break;
    }
}

ParitySupport restrict_to_support(const std::vector<CliffGate>& cprime_dagger,
                                  const std::vector<DiagResult::ZTerm>& tz, int n) {
    ParitySupport ps(n);

    // ---------- Step 1: build |ψ⟩ = C'†|0⟩ ----------
    for (const auto& g : cprime_dagger) apply_gate(ps.support, g);

    const AffineState& psi = ps.support;
    const int d = psi.k_;               // free-var count
    const int dwords = (d + 63) / 64;   // words per parity column over d free vars

    // ---------- Step 2: restrict every Z-term ----------
    // Word-parallel: R is row-major (row q = qubit q, bit j = free var j), so the
    // parity column m·R is the XOR of the R-rows of the qubits in the Z-mask m, and
    // m·b is the parity of b over m. Visiting only the set bits of m (ctz walk) makes
    // each term O(|m|·dwords) instead of the scalar O(n·d) bit loop it replaces;
    // the output is bit-identical (the scalar loop is kept as the oracle in
    // cpp/tests/test_restrict_to_support.cpp).
    const int nwords = (n + 63) / 64;
    const uint64_t last_mask = (d & 63) ? ((1ull << (d & 63)) - 1) : ~0ull;
    int gphase = 0;  // accumulator mod 16
    for (const auto& zt : tz) {
        const Pauli& D = zt.zstring;
        // (defensive, debug-only: assert() is compiled out under -DNDEBUG, which both the
        // setup.py and the CMake Release builds define) Z-strings are pure Z: no x-bits.
        for (int w = 0; w < nwords; ++w) {
            const uint64_t xm = (w == nwords - 1 && (n & 63)) ? ((1ull << (n & 63)) - 1) : ~0ull;
            assert(!(D.x[w] & xm) && "restrict_to_support: Z-string has x-bit");
            (void)xm;
        }

        // parity column = m·R : XOR of rows R(q,·) over qubits q in mask m.
        // constant bit = m·b.
        std::vector<uint64_t> col(dwords, 0);
        int cbit = 0;  // m·b
        for (int w = 0; w < nwords; ++w) {
            uint64_t zm = D.z[w];
            if (w == nwords - 1 && (n & 63)) zm &= (1ull << (n & 63)) - 1;
            while (zm) {
                const int q = (w << 6) | __builtin_ctzll(zm);
                zm &= zm - 1;
                cbit ^= (psi.b[q] != 0);
                if (dwords) {
                    const uint64_t* r = psi.R.row(q);
                    for (int t = 0; t < dwords; ++t) col[t] ^= r[t];
                }
            }
        }
        bool any = false;
        if (dwords) {
            col[dwords - 1] &= last_mask;  // bits >= d are invariant-zero in R; mask anyway
            for (int t = 0; t < dwords; ++t) any |= (col[t] != 0);
        }

        if (!any) {
            // CONSTANT on support: phase exp(i(π/8)·coeff·(-1)^{m·b}).
            int contrib = cbit ? -zt.coeff : zt.coeff;
            gphase = ((gphase + contrib) % 16 + 16) % 16;
        } else {
            // GENUINE magic: the (-1)^{m·b} sign is folded into the coeff so the
            // per-point phase is ζ16^{coeff·(-1)^{m·R·t}} with NO extra constant
            // (cbit flips the overall sign of the column's ±1 contribution).
            int c = cbit ? -zt.coeff : zt.coeff;
            c = ((c % 16) + 16) % 16;
            ps.columns.push_back(std::move(col));
            ps.coeffs.push_back(c);
        }
    }

    ps.global_phase_mult8 = gphase;
    return ps;
}

}  // namespace qeccore
