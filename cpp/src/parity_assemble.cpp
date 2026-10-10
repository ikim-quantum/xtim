#include "qeccore/parity_assemble.hpp"
#include <cstddef>
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/bare_state_errors.hpp"
#include "qeccore/framed_superposition.hpp"   // sizeof(Entry): the build-peak estimate
#include "qeccore/state_budget.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <new>
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

// The ONE complex multiply of the per-branch coefficient chain.  Out of line on purpose: the bytes
// of every coefficient are a contract, and an inlined `w *= z` lets the compiler pick a different
// FMA contraction / vectorisation per call site (a prefix-shared loop and a per-branch loop then
// disagree in the last bit).  Every chain multiply goes through here, so all sites agree.
__attribute__((noinline)) static cd chain_mul(cd w, cd z) {
    w *= z;
    return w;
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
// ROWS (R is n×d, full column rank ⇒ its rows span F2^d ⇒ always solvable).  solve() returns
// the qubit set m (length-n indicator) as a Z-string mask.  `dwords` = ceil(d/64).
//
// The elimination depends on R only, the back-substitution on col: prepare() runs the elimination
// once and records the R it ran on; for() re-prepares only when R's CONTENT differs (a content key,
// never an address), so every solve returns exactly what a from-scratch solve on the current R would
// (the same elimination, the same particular solution).  assemble_from_parities solves for every
// even column and every basis column against the same R (apply_parity_s' CX sandwich restores it).
struct RowCombinationSolver {
    struct Row { std::vector<uint64_t> rbits; std::vector<uint64_t> sel; };
    int n = -1, d = -1, dwords = 0;
    std::vector<uint64_t> key;       // R rows (d bits each, dwords words) the elimination ran on
    std::vector<Row> rows;
    std::vector<int> piv_row;

    static std::vector<uint64_t> key_of(const AffineState& psi, int dwords) {
        // Bits 0..k-1 of each row (the packed row may carry capacity words / stale bits beyond k).
        std::vector<uint64_t> k((size_t)psi.n_ * (size_t)dwords, 0);
        const int kk = psi.k_;
        const int cw = std::min(dwords, (kk + 63) / 64);
        for (int q = 0; q < psi.n_; ++q) {
            const uint64_t* rr = psi.R.row(q);
            uint64_t* out = &k[(size_t)q * (size_t)dwords];
            for (int w = 0; w < cw; ++w) out[w] = rr[w];
            if (cw > 0 && (kk & 63)) out[cw - 1] &= (1ull << (kk & 63)) - 1;
        }
        return k;
    }

    // Re-prepare iff psi's (n, d, R) differ from the prepared ones.
    void for_state(const AffineState& psi, int dw) {
        std::vector<uint64_t> k = key_of(psi, dw);
        if (psi.n_ == n && psi.k_ == d && dw == dwords && k == key) return;
        n = psi.n_; d = psi.k_; dwords = dw; key.swap(k);
        // Augmented rows: [ R[q,:] | e_q ] reduced; we want the combination of R-rows = col.
        // Work matrix: each entry is (d-bit R-row, n-bit selector). RREF on the R-row part.
        const int nwords = (n + 63) / 64;
        rows.clear();
        rows.reserve(n);
        for (int q = 0; q < n; ++q) {
            Row r;
            r.rbits.assign(key.begin() + (ptrdiff_t)q * dwords, key.begin() + (ptrdiff_t)(q + 1) * dwords);
            r.sel.assign(nwords, 0);
            r.sel[q >> 6] |= 1ull << (q & 63);
            rows.push_back(std::move(r));
        }
        // Gaussian elimination over the d-bit R-row part, tracking selector.
        piv_row.assign(d, -1);
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
    }

    std::vector<uint8_t> solve(const std::vector<uint64_t>& col) const {
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
};

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
                                       int n, AssemblePath path,
                                       CanonicalStabSum::CosetTrace* trace) {
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
    RowCombinationSolver rcs;       // prepared once per distinct R (content-keyed)
    for (const auto& e : even_cols) {
        rcs.for_state(psi, dwords);
        std::vector<uint8_t> m = rcs.solve(e.col);
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
    // χ = 2^r must be representable: refuse LOUDLY above the cap.  (3.1.7 computed `1 << r`
    // unchecked — UB for r >= 31, wrapping mod 32 on x86 into a truncated, non-normalised state.)
    if (r > kMaxBareRank) throw BareStateCapacityError(bare_rank_refusal(r, "assemble_from_parities"));
    const int chi = 1 << r;                                   // r <= kMaxBareRank: no overflow

    // Process-wide state budget (state_budget.hpp): refuse BEFORE any χ-sized allocation when the
    // build's peak χ-scaled footprint would not fit next to what is already live.  The peak holds
    // the CanonicalStabSum branch list AND the FramedSuperposition entry list it is moved into
    // (from_css: the source list's capacity lives until the sum is dropped), each element plus its
    // σ chunk, and this build's per-branch tables: coefficient (16 B), pattern (4 B), gauge
    // exponent (1 B) and the prefix-shared coefficient levels (≤ 2·χ·16 B).  The tables are charged
    // for the build's lifetime by a ticket; the lists by StateAllocator as they are allocated.
    StateBudgetContext budget_ctx(chi, r, n);
    const int64_t kTableBytesPerBranch = 16 + 4 + 1 + 32;
    state_budget_precheck(
        (int64_t)chi * ((int64_t)(sizeof(CanonicalStabSum::Branch) + kSigmaHeapBytes) +
                        (int64_t)(sizeof(FramedSuperposition::Entry) + kSigmaHeapBytes) +
                        kTableBytesPerBranch),
        "building the bare state");
    StateBudgetTicket build_tables((int64_t)chi * kTableBytesPerBranch, "the bare-state build tables");

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
        rcs.for_state(psi, dwords);
        basis_m[i] = rcs.solve(basis[i]);
        int cb = 0;
        for (int q = 0; q < n; ++q) if (basis_m[i][q] && psi.b[q]) cb ^= 1;
        basis_cbit[i] = cb;
    }

    // ── Step 4: the χ branches ──────────────────────────────────────────────────────────────────
    // Branch σ ∈ {0..χ-1} is |ψ⟩ restricted to the coset {t : basis·t = σ}: the unit ray ray(σ)
    // (r Z-string projections of ψ) times the coefficient
    //     w(σ) = ζ16^{gp} · fac · ∏_{odd k} ζ16^{coeff_k·(-1)^{(col_k in basis)·σ}},
    // where fac = 2^{-r/2} is the projection amplitude (every projection splits ψ evenly: the basis
    // columns are independent, so each pauli_project factor is the real 2^{-1/2}, for EVERY σ).
    auto make_ray = [&](uint32_t sigma, ExactPhase* fac_out) {
        auto ray = std::make_unique<AffineState>(psi);   // even-folded support
        ExactPhase fac = ExactPhase::one();              // amplitude factors from projection
        for (int i = 0; i < r; ++i) {
            int sig_i = (int)((sigma >> i) & 1u);             // desired col_i·t value
            int phys = sig_i ^ basis_cbit[i];                 // physical Z outcome m·y = col·t ⊕ cbit
            ExactPhase f = project_parity(*ray, basis_m[i], phys);
            fac = fac.mul(f);
        }
        if (fac_out) *fac_out = fac;
        return ray;
    };
    // ζ16 per odd column at ±coeff (the same zeta16() doubles, looked up instead of recomputed),
    // and each odd column's basis membership as an r-bit mask (parity = popcount(mask & σ)).
    std::vector<cd> zplus(odd_in_basis.size()), zminus(odd_in_basis.size());
    std::vector<uint32_t> odd_mask(odd_in_basis.size(), 0);
    for (size_t k = 0; k < odd_in_basis.size(); ++k) {
        zplus[k] = zeta16(odd_coeff[k]);
        zminus[k] = zeta16(-odd_coeff[k]);
        for (int i = 0; i < r; ++i) if (odd_in_basis[k][i]) odd_mask[k] |= 1u << i;
    }
    auto coeff_of = [&](uint32_t sigma, const ExactPhase& fac) {
        cd w = zeta16(gp) * fac.to_complex();
        for (size_t k = 0; k < odd_mask.size(); ++k)
            w = chain_mul(w, (__builtin_popcount(odd_mask[k] & sigma) & 1) ? zminus[k] : zplus[k]);
        return w;
    };

    // Allocation failure at a representable-but-huge χ is a capacity refusal too (named, loud).
    ParityBareState out(n);
    try {
    if (path == AssemblePath::ray_oracle) {
        // ORACLE (3.1.7 construction, tests only): materialise all χ rays, then from_rays.
        std::vector<std::unique_ptr<AffineState>> rays;
        std::vector<cd> coeffs;
        rays.reserve(chi);
        coeffs.reserve(chi);
        for (int sigma = 0; sigma < chi; ++sigma) {
            ExactPhase fac;
            rays.push_back(make_ray((uint32_t)sigma, &fac));
            coeffs.push_back(coeff_of((uint32_t)sigma, fac));
        }
        out.state = CanonicalStabSum::from_rays(n, std::move(rays), std::move(coeffs));
    } else {
        // Production: the coset-family construction reads the frame off ray 0 and the syndromes
        // and gauges off the Pauli algebra of psi and the Z-strings (CanonicalStabSum::
        // from_coset_family; byte-identical to the oracles).  fac is σ-independent (see above) —
        // every ray it builds (ray 0, the spot-check probes; the 3.1.8 oracle: every weight <= 2
        // ray) is checked against fac(0).
        ExactPhase fac0;
        { auto r0 = make_ray(0, &fac0); }
        auto checked_ray = [&](uint32_t sigma) {
            ExactPhase f;
            auto ray = make_ray(sigma, &f);
            if (!f.exact_eq(fac0))
                throw BareStateInvariantError("assemble_from_parities: coset projection amplitude "
                                              "depends on the branch — refusing");
            return ray;
        };
        std::vector<cd> coeffs((size_t)chi);
        const size_t K = odd_mask.size();
        if (K <= (size_t)r) {
            // w(σ) depends on σ only through the K parity bits p_k(σ) = parity(odd_mask[k] & σ), and
            // the product chain multiplies them in k order.  Share the chain's PREFIXES: level k holds
            // one value per (p_0..p_{k-1}) pattern, each the SAME left-to-right product coeff_of forms,
            // so every coefficient is bit-identical — 2^{K+1} complex multiplies instead of K·χ.
            std::vector<cd> lvl(1, zeta16(gp) * fac0.to_complex()), nxt;
            for (size_t k = 0; k < K; ++k) {
                nxt.resize(lvl.size() * 2);
                for (size_t p = 0; p < lvl.size(); ++p) {
                    nxt[p] = chain_mul(lvl[p], zplus[k]);              // p_k = 0
                    nxt[p | lvl.size()] = chain_mul(lvl[p], zminus[k]);  // p_k = 1
                }
                lvl.swap(nxt);
            }
            // pattern(σ), O(1) per σ: with i the lowest set bit, pattern(σ) = pattern(σ ^ e_i) ^ colpat[i].
            std::vector<uint32_t> colpat((size_t)r, 0), pat((size_t)chi, 0);
            for (size_t k = 0; k < K; ++k)
                for (int i = 0; i < r; ++i) if ((odd_mask[k] >> i) & 1u) colpat[(size_t)i] |= 1u << k;
            coeffs[0] = lvl[0];
            for (uint32_t g = 1; g < (uint32_t)chi; ++g) {
                pat[g] = pat[g & (g - 1)] ^ colpat[(size_t)__builtin_ctz(g)];
                coeffs[g] = lvl[pat[g]];
            }
        } else {
            for (int sigma = 0; sigma < chi; ++sigma) coeffs[(size_t)sigma] = coeff_of((uint32_t)sigma, fac0);
        }
        if (path == AssemblePath::coset_rays) {
            out.state = CanonicalStabSum::from_coset_family_rays(n, r, checked_ray, coeffs, trace);
        } else {
            std::vector<Pauli> zs((size_t)r, Pauli(n));        // the projected Z-strings Z^{m_i}
            for (int i = 0; i < r; ++i)
                for (int q = 0; q < n; ++q) if (basis_m[(size_t)i][(size_t)q]) zs[(size_t)i].setz(q);
            out.state = CanonicalStabSum::from_coset_family(n, r, psi, zs, checked_ray, coeffs, trace);
        }
    }
    out.chi = chi;
    } catch (const std::bad_alloc&) {
        throw BareStateCapacityError(
            "assemble_from_parities: allocation failed building the bare state at chi = 2^" +
            std::to_string(r) + " branches on n = " + std::to_string(n) +
            " qubits — the state does not fit in memory");
    }

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
