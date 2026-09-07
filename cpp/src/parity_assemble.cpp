#include "qeccore/parity_assemble.hpp"
#include <cstddef>
#include "qeccore/pauli_kernels.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <vector>

namespace qeccore {

namespace {

using cd = std::complex<double>;

// ζ16^e = e^{iπe/8}.
static cd zeta16(int e) {
    e = ((e % 16) + 16) % 16;
    double th = M_PI * e / 8.0;
    return cd(std::cos(th), std::sin(th));
}

// d-bit packed column helpers (column[w] holds free vars 64w..64w+63).
static bool colbit(const std::vector<uint64_t>& c, int j) {
    return (c[j >> 6] >> (j & 63)) & 1ull;
}
static bool is_zero_col(const std::vector<uint64_t>& c) {
    for (auto w : c) if (w) return false;
    return true;
}

// Solve  m·R = col  over GF(2):  express the d-bit row `col` as a GF(2) sum of R's
// ROWS (R is n×d, full column rank ⇒ its rows span F2^d ⇒ always solvable).  Returns
// the qubit set m (length-n indicator) as a Z-string mask.  `dwords` = ceil(d/64).
static std::vector<uint8_t> solve_row_combination(const AffineState& psi,
                                                  const std::vector<uint64_t>& col,
                                                  int dwords) {
    const int n = psi.n_, d = psi.k_;
    // Augmented rows: [ R[q,:] | e_q ] reduced; we want the combination of R-rows = col.
    // Work matrix: each entry is (d-bit R-row, n-bit selector). RREF on the R-row part.
    struct Row { std::vector<uint64_t> rbits; std::vector<uint64_t> sel; };
    const int nwords = (n + 63) / 64;
    std::vector<Row> rows;
    rows.reserve(n);
    for (int q = 0; q < n; ++q) {
        Row r;
        r.rbits.assign(dwords, 0);
        for (int j = 0; j < d; ++j)
            if (psi.R.get(q, j)) r.rbits[j >> 6] |= 1ull << (j & 63);
        r.sel.assign(nwords, 0);
        r.sel[q >> 6] |= 1ull << (q & 63);
        rows.push_back(std::move(r));
    }
    // Gaussian elimination over the d-bit R-row part, tracking selector.
    std::vector<int> piv_row(d, -1);
    int rank = 0;
    for (int j = 0; j < d && rank < (int)rows.size(); ++j) {
        int sel = -1;
        for (int r = rank; r < (int)rows.size(); ++r)
            if ((rows[r].rbits[j >> 6] >> (j & 63)) & 1ull) { sel = r; break; }
        if (sel < 0) continue;
        std::swap(rows[rank], rows[sel]);
        for (int r = 0; r < (int)rows.size(); ++r) {
            if (r == rank) continue;
            if ((rows[r].rbits[j >> 6] >> (j & 63)) & 1ull) {
                for (int w = 0; w < dwords; ++w) rows[r].rbits[w] ^= rows[rank].rbits[w];
                for (int w = 0; w < nwords; ++w) rows[r].sel[w] ^= rows[rank].sel[w];
            }
        }
        piv_row[j] = rank;
        ++rank;
    }
    // Back-substitute: target = col, accumulate selector of pivot rows.
    std::vector<uint8_t> m(n, 0);
    std::vector<uint64_t> work = col;
    for (int j = 0; j < d; ++j) {
        if (!((work[j >> 6] >> (j & 63)) & 1ull)) continue;
        int pr = piv_row[j];
        assert(pr >= 0 && "solve_row_combination: col not in row space of R (R must be full rank)");
        // add pivot row pr (its rbits has leading bit j after RREF; xor it out)
        for (int w = 0; w < dwords; ++w) work[w] ^= rows[pr].rbits[w];
        for (int q = 0; q < n; ++q)
            if ((rows[pr].sel[q >> 6] >> (q & 63)) & 1ull) m[q] ^= 1;
    }
    for (auto w : work) { (void)w; assert(w == 0 && "solve_row_combination: residual after back-subst"); }
    return m;
}

// Apply the diagonal Clifford  i^{c'·(m·y)}  (m a Z-string over qubits) to the
// AffineState exactly: CX-fan every qubit of m into the first, S^{c'} on it, then
// CX-uncompute.  No ancilla; the compute/uncompute sandwich is identity on the
// computational basis so the net effect is the diagonal i^{c'·parity}.
static void apply_parity_s(AffineState& st, const std::vector<uint8_t>& m, int cprime) {
    cprime = ((cprime % 4) + 4) % 4;
    if (cprime == 0) return;
    std::vector<int> qs;
    for (int q = 0; q < st.n_; ++q) if (m[q]) qs.push_back(q);
    if (qs.empty()) return;                 // identity Z-string ⇒ pure (already-handled) constant
    int head = qs[0];
    for (size_t i = 1; i < qs.size(); ++i) st.apply_cx(qs[i], head);  // y_head ^= y_qs[i]
    for (int r = 0; r < cprime; ++r) st.apply_s(head);                // i^{cprime·y_head}
    for (size_t i = qs.size(); i-- > 1;) st.apply_cx(qs[i], head);    // uncompute
}

// Project the AffineState onto the Z-string coset  m·y = bit  (bit∈{0,1}), returning
// the amplitude factor (so factor · projected.to_statevector() = the restriction of
// the ORIGINAL state to the coset, un-renormalized).  Same CX-fan-in trick: fan the
// parity into `head`, single-qubit Z-project, fan back.
static ExactPhase project_parity(AffineState& st, const std::vector<uint8_t>& m, int bit) {
    std::vector<int> qs;
    for (int q = 0; q < st.n_; ++q) if (m[q]) qs.push_back(q);
    assert(!qs.empty() && "project_parity: empty Z-string");
    int head = qs[0];
    for (size_t i = 1; i < qs.size(); ++i) st.apply_cx(qs[i], head);
    ExactPhase f = st.pauli_project(2, head, bit ? -1 : +1);   // Z outcome (-1)^bit
    for (size_t i = qs.size(); i-- > 1;) st.apply_cx(qs[i], head);
    return f;
}

}  // namespace

ParityBareState assemble_from_parities(const ParitySupport& ps,
                                       const std::vector<CliffGate>& output_clifford,
                                       int n) {
    const int d = ps.support.k_;
    const int dwords = (d + 63) / 64;

    // ── Step 1: combine equal columns; fold zero/coeff-8 constants ───────────────────────────────
    int gp = ((ps.global_phase_mult8 % 16) + 16) % 16;
    std::map<std::vector<uint64_t>, int> comb;
    for (size_t k = 0; k < ps.columns.size(); ++k) {
        std::vector<uint64_t> col = ps.columns[k];
        if ((int)col.size() < dwords) col.resize(dwords, 0);
        comb[col] = (comb.count(col) ? comb[col] : 0) + ps.coeffs[k];
    }

    std::vector<std::pair<std::vector<uint64_t>, int>> odd_cols;   // genuine magic
    struct EvenCol { std::vector<uint64_t> col; int coeff; };  // FULL even coeff c
    std::vector<EvenCol> even_cols;                                // Clifford (S-type)

    for (auto& kv : comb) {
        int c = ((kv.second % 16) + 16) % 16;
        if (c == 0) continue;                       // identity
        if (is_zero_col(kv.first)) {
            // constant column: ζ16^{c·(+1)} on every point ⇒ global phase.
            gp = ((gp + c) % 16 + 16) % 16;
            continue;
        }
        if (c == 8) {                               // ζ16^{8·(±1)} = −1 on every coset ⇒ global.
            gp = ((gp + 8) % 16 + 16) % 16;
            continue;
        }
        if (c % 2 == 1) odd_cols.push_back({kv.first, c});
        else            even_cols.push_back({kv.first, c});  // ζ16^{c·(-1)^{m·y}}, c even
    }

    // ── Step 2: working support ψ; fold even (Clifford) columns into it ──────────────────────────
    // The column phase is ζ16^{c·(-1)^{col·t}} (c even).  The Z-string m (m·R = col) reads
    // m·y = (m·b) ⊕ (col·t) = cbit ⊕ col·t, so (-1)^{col·t} = (-1)^{cbit}·(-1)^{m·y}.  With
    // c2 = (-1)^{cbit}·c the phase is ζ16^{c2·(-1)^{m·y}} = ζ16^{c2} · i^{(-c2/2)·(m·y)}, i.e.
    // S^{(-c2/2) mod 4} along m plus the constant ζ16^{c2} into the global phase.  (b≠0 makes
    // cbit nonzero; the earlier b=0 path missed this sign.)
    AffineState psi = ps.support;   // value copy (b,R,D,J,omega)
    for (const auto& e : even_cols) {
        std::vector<uint8_t> m = solve_row_combination(psi, e.col, dwords);
        int cbit = 0;
        for (int q = 0; q < n; ++q) if (m[q] && psi.b[q]) cbit ^= 1;
        int c2 = cbit ? ((-e.coeff) % 16 + 16) % 16 : e.coeff;
        int spow = ((-(c2 / 2)) % 4 + 4) % 4;
        apply_parity_s(psi, m, spow);
        gp = ((gp + c2) % 16 + 16) % 16;               // global ζ16^{c2}
    }

    // ── Step 3: independent GF(2) basis of the odd columns (magic rank r) ────────────────────────
    std::vector<std::vector<uint64_t>> basis;       // RREF rows
    std::vector<int> piv;                            // pivot bit per basis row
    auto reduce = [&](std::vector<uint64_t> v) -> std::vector<uint64_t> {
        for (size_t i = 0; i < basis.size(); ++i)
            if (colbit(v, piv[i])) xor_into(v, basis[i]);
        return v;
    };
    for (auto& oc : odd_cols) {
        std::vector<uint64_t> rr = reduce(oc.first);
        int p = -1;
        for (int j = 0; j < d; ++j) if (colbit(rr, j)) { p = j; break; }
        if (p < 0) continue;                         // dependent on existing basis
        basis.push_back(rr);
        piv.push_back(p);
    }
    const int r = (int)basis.size();
    const int chi = 1 << r;

    // Express each odd column in the basis (which basis rows XOR to it): coeff for the per-coset
    // phase.  For column col, find subset S⊆basis with XOR = col; its parity on coset σ is
    // XOR_{i∈S} σ_i.
    std::vector<std::vector<uint8_t>> odd_in_basis;  // per odd column: r-bit membership
    std::vector<int> odd_coeff;
    for (auto& oc : odd_cols) {
        std::vector<uint64_t> v = oc.first;
        std::vector<uint8_t> mem(r, 0);
        for (int i = 0; i < r; ++i)
            if (colbit(v, piv[i])) { mem[i] = 1; xor_into(v, basis[i]); }
        assert(is_zero_col(v) && "odd column not spanned by basis");
        odd_in_basis.push_back(std::move(mem));
        odd_coeff.push_back(oc.second);
    }

    // Z-string for each basis column (for coset projection), with its constant offset cbit = m·b.
    // The Z-string m satisfies m·R = col, so on a support point m·y = (m·b) ⊕ (col·t) = cbit ⊕ col·t.
    // To force col·t = σ_i we therefore project onto the physical outcome m·y = σ_i ⊕ cbit_i.
    std::vector<std::vector<uint8_t>> basis_m(r);
    std::vector<int> basis_cbit(r, 0);
    for (int i = 0; i < r; ++i) {
        basis_m[i] = solve_row_combination(psi, basis[i], dwords);
        int cb = 0;
        for (int q = 0; q < n; ++q) if (basis_m[i][q] && psi.b[q]) cb ^= 1;
        basis_cbit[i] = cb;
    }

    // ── Step 4: build the χ branch rays ─────────────────────────────────────────────────────────
    std::vector<std::unique_ptr<AffineState>> rays;
    std::vector<cd> coeffs;
    rays.reserve(chi);
    coeffs.reserve(chi);
    for (int sigma = 0; sigma < chi; ++sigma) {
        AffineState ray = psi;                       // even-folded support
        ExactPhase fac = ExactPhase::one();          // amplitude factors from projection
        for (int i = 0; i < r; ++i) {
            int sig_i = (sigma >> i) & 1;                 // desired col_i·t value
            int phys = sig_i ^ basis_cbit[i];             // physical Z outcome m·y = col·t ⊕ cbit
            ExactPhase f = project_parity(ray, basis_m[i], phys);
            fac = fac.mul(f);
        }
        // `from_rays` expects UNIT-NORM rays with all amplitude scaling in the coefficient.
        // `ray` (post pauli_project) is already the unit-norm coset stabilizer state; the
        // projection amplitude factor `fac` (= 2^{-r/2} · projection phase) carries ψ's amplitude
        // on this coset relative to that unit-norm state, so it belongs in the coefficient.

        // per-coset magic coefficient = ζ16^{gp} · ∏_{odd k} ζ16^{coeff_k·(-1)^{(col_k in basis)·σ}}
        cd w = zeta16(gp) * fac.to_complex();
        for (size_t k = 0; k < odd_in_basis.size(); ++k) {
            int par = 0;
            for (int i = 0; i < r; ++i) if (odd_in_basis[k][i] && ((sigma >> i) & 1)) par ^= 1;
            int e = par ? -odd_coeff[k] : odd_coeff[k];
            w *= zeta16(e);
        }
        rays.push_back(std::make_unique<AffineState>(std::move(ray)));
        coeffs.push_back(w);
    }

    // ── Step 5: assemble the CanonicalStabSum from rays+coeffs ───────────────────────────────────
    ParityBareState out(n);
    out.state = CanonicalStabSum::from_rays(n, std::move(rays), std::move(coeffs));
    out.chi = chi;

    // ── Step 6: apply the output Clifford (C·C') through the deferred-Clifford path ──────────────
    for (const CliffGate& g : output_clifford) {
        switch (g.kind) {
            case GateKind::H:   out.state.apply_h(g.targets[0]); break;
            case GateKind::S:   out.state.apply_s(g.targets[0]); break;
            case GateKind::SDG: out.state.apply_sdg(g.targets[0]); break;
            case GateKind::X:   out.state.apply_x(g.targets[0]); break;
            case GateKind::Y:   out.state.apply_y(g.targets[0]); break;
            case GateKind::Z:   out.state.apply_z(g.targets[0]); break;
            case GateKind::CX:  out.state.apply_cx(g.targets[0], g.targets[1]); break;
            case GateKind::CZ:  out.state.apply_cz(g.targets[0], g.targets[1]); break;
            default: assert(false && "assemble_from_parities: unexpected output gate"); break;
        }
    }

    return out;
}

}  // namespace qeccore
