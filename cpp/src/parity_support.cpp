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
    int gphase = 0;  // accumulator mod 16
    for (const auto& zt : tz) {
        const Pauli& D = zt.zstring;
        // (defensive) Z-strings are pure Z: no x-bits.
        for (int q = 0; q < n; ++q) assert(!D.xbit(q) && "restrict_to_support: Z-string has x-bit");

        // parity column = m·R : for each free var j, bit = XOR over qubits q in
        // mask m of R(q,j).  constant bit = m·b.
        std::vector<uint64_t> col(dwords, 0);
        bool any = false;
        for (int j = 0; j < d; ++j) {
            int bit = 0;
            for (int q = 0; q < n; ++q)
                if (D.zbit(q) && psi.R.get(q, j)) bit ^= 1;
            if (bit) { col[j >> 6] |= (1ull << (j & 63)); any = true; }
        }
        int cbit = 0;  // m·b
        for (int q = 0; q < n; ++q)
            if (D.zbit(q) && psi.b[q]) cbit ^= 1;

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
