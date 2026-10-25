// carried_compose — J-b1 P9b (XQ9 (A)): see cpp/include/qeccore/carried_compose.hpp.
#include "qeccore/carried_compose.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/fast_todd.hpp"
#include "qeccore/parity_support.hpp"
#include "qeccore/pauli_diagonalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/pauli_rotation_form.hpp"
#include "qeccore/residue_projection.hpp"
#include "qeccore/state_budget.hpp"

namespace qeccore {
namespace {

// ── verbatim copies (normal_form.cpp, parity_assemble.cpp): file-local, so those TUs' codegen is untouched ──
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
    if (r == 0) return;                             // no magic

    // Project the distinct magic directions into the r-bit residue space and run FastTODD.  The
    // reduced set `red` has the MINIMAL T-count; its span can be SMALLER than r (BH: rank(red) = k
    // = r - 3 — the lost directions are carried by the even residue g below), and χ = 2^rank(red).
    // Residue masks are uint32 for r <= 30 (the historical path, unchanged) and word vectors above
    // (3.1.8: the old r > 30 guard skipped the reduction and left χ at the raw rank — bt27: 2^33).
    // Each column is handed on as its residue SUPPORT (sorted set-bit indices).
    std::vector<ResidueColumn> f_cols, fp_cols;        // f = Σ_in (-c_in)·p, f' = Σ_red (+1)·p
    if (r <= 30) {
        std::vector<uint32_t> proj;
        proj.reserve(magic_coeff.size());
        for (auto& kv : magic_coeff) proj.push_back(P.project(kv.first));
        std::vector<uint32_t> red = fast_todd_reduce(proj, r);
        auto on_of = [r](uint32_t m) { std::vector<int> on; for (int i = 0; i < r; ++i) if ((m >> i) & 1u) on.push_back(i); return on; };
        size_t t = 0;
        for (auto& kv : magic_coeff) f_cols.push_back({on_of(proj[t++]), -kv.second});
        for (uint32_t m : red) if (m) fp_cols.push_back({on_of(m), 1});
    } else {
        std::vector<WideCol> proj;
        proj.reserve(magic_coeff.size());
        for (auto& kv : magic_coeff) proj.push_back(P.project_wide(kv.first));
        std::vector<WideCol> red = fast_todd_reduce_wide(proj, r);
        auto on_of = [r](const WideCol& m) { std::vector<int> on; for (int i = 0; i < r; ++i) if ((m[(size_t)(i >> 6)] >> (i & 63)) & 1ull) on.push_back(i); return on; };
        size_t t = 0;
        for (auto& kv : magic_coeff) f_cols.push_back({on_of(proj[t++]), -kv.second});
        for (const WideCol& m : red) { std::vector<int> on = on_of(m); if (!on.empty()) fp_cols.push_back({std::move(on), 1}); }
    }

    // ── PHASE-FAITHFUL rebuild: reduced ODD T-set + recovered EVEN (Clifford) residue ──────────────
    // FastTODD reduces the magic DIRECTIONS only: it sees each input as a coeff-1 T (its proper()/
    // tohpe() work mod-2 on the masks), and DROPS the even-graded (Clifford) remainder of the
    // reduction — including (a) the mod-8 Clifford residue of a genuinely CUBIC reduction
    // (cultivation_d5) and (b) the even part of any input whose coeff ≠ 1 (mod 8).  We RECOVER that
    // remainder over the rank-r residue and fold it back as even (Clifford) parity columns, so the
    // assembled state's total phase is f' + g = f EXACTLY (phase-faithful).
    //
    // Convention (all in the column relphase: a column (mask m, coeff c mod 16) contributes the
    // per-residue-point phase ζ16^{c·(-1)^{m·x}}, whose RELATIVE-phase ζ8 content is (-c)·p_m(x)
    // with p_m(x)=popcount(m&x)&1):
    //   • f(x)  = Σ_input (-c_in)·p_{m_in}(x)            (the TRUE magic phase poly over the residue)
    //   • f'(x) = Σ_{m∈red} (+1)·p_m(x)                  (each reduced direction as a coeff-1 T)
    //   • g(x)  = f(x) − f'(x)  (mod 8)                  (the Clifford remainder — PROVABLY even-graded:
    //        f = Σ_in(+1)p_m + Σ_in(-c_in−1)p_m, the 2nd sum is even-coeff⇒Clifford, and
    //        Σ_in(+1)p_m = Σ_proj(+1)p_m ≡ Σ_red(+1)p_m (mod Clifford) by FastTODD's span theorem).
    // A reduced direction representing a coeff-1 T is the column (m, 7): -7 ≡ 1 (mod 8) ⇒ ζ8 +1·p.
    //
    // g's monomial expansion is computed SYMBOLICALLY (even_residue_symbolic: the exact mod-8
    // identity p_m = Σ_S (−2)^{|S|−1} x^S, |S| <= 3), polynomial in r.  The pre-3.1.8 dense 2^r
    // tables + Möbius transform (exponential in r: 3·2^r bytes) survive as the test ORACLE
    // even_residue_dense; test_fast_todd_wide pins symbolic == dense, monomial for monomial.
    EvenResidue g = even_residue_symbolic(r, f_cols, fp_cols);
    // g MUST be even-graded (its odd/T part is zero) — otherwise the FastTODD reduction or this
    // reconstruction is wrong.  Fail-LOUD (std::logic_error) rather than silently mis-build.
    if (!g.ok) {
        size_t s = 0;
        for (int i : g.bad_on) if (i < 64) s |= (size_t)1 << i;
        char buf[96];
        std::snprintf(buf, sizeof buf, "(monomial 0x%zx deg=%d coeff=%d)", s, g.bad_deg, g.bad_coeff);
        throw std::logic_error(std::string("fast_todd_reduce_columns: recovered residue g is NOT even-graded ") +
                               buf + " — reduction/recovery model is wrong (engine bug)");
    }
    

    std::vector<std::vector<uint64_t>> new_cols;
    std::vector<int> new_coeffs;
    // (1) reduced ODD columns: each a coeff-1 T ⇒ column coeff 7 (ζ8 +1·p).
    for (const ResidueColumn& m : fp_cols) {
        std::vector<uint64_t> dir = P.reconstruct_bits(m.on);
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
    auto emit_even = [&](const std::vector<int>& on, int a8) {   // even ζ8 relphase a8·p_on
        a8 = ((a8 % 8) + 8) % 8;
        if (a8 == 0) return;
        int c16 = ((8 - a8) % 8 + 8) % 8;                  // column coeff: -c16 ≡ a8 (mod 8)
        std::vector<uint64_t> dir = P.reconstruct_bits(on);
        bool nz = false; for (auto w : dir) if (w) { nz = true; break; }
        if (!nz) return;                                   // constant ⇒ pure global phase, drop
        new_cols.push_back(std::move(dir));
        new_coeffs.push_back(c16);
    };
    for (const EvenMonomial& mo : g.mono) {               // ascending residue-mask order
        if (mo.lo < 0) {
            emit_even({mo.hi}, mo.a);                      // a ∈ {2,4,6}
        } else {                                           // deg == 2, a == 4 (CZ); guarded above
            
            const int j = mo.lo, k = mo.hi;                // j = lower bit, k = higher bit
            emit_even({j}, 2);
            emit_even({k}, 2);
            emit_even({j, k}, 6);                          // −2 ≡ 6 (mod 8)
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



// The Hermitian representative of a signed Pauli (pauli_rotation_form's convention: phase = #Y mod 2) and its sign.
std::pair<Pauli, int> split_sign_(const Pauli& Qs) {
    Pauli Q = Qs;
    Q.phase = Q.xz_overlap() & 1;
    const int delta = ((Qs.phase - Q.phase) % 4 + 4) % 4;
    if (delta != 0 && delta != 2) throw std::logic_error("compose_carried_bare: a non-Hermitian axis (engine bug)");
    return {Q, delta == 0 ? 1 : -1};
}
int mod16_(int c) { return ((c % 16) + 16) % 16; }
uint8_t clif_code_(GateKind g) {
    switch (g) {
        case GateKind::H: return 0;
        case GateKind::S: return 1;
        case GateKind::SDG: return 2;
        case GateKind::X: return 3;
        case GateKind::Y: return 4;
        case GateKind::Z: return 5;
        case GateKind::CX: return 6;
        case GateKind::CZ: return 7;
        default: throw std::logic_error("compose_carried_bare: a non-Clifford gate in a Clifford layer (engine bug)");
    }
}
void apply_gate_css_(CanonicalStabSum& s, const CliffGate& g) {
    switch (g.kind) {
        case GateKind::H:   s.apply_h(g.targets[0]); break;
        case GateKind::S:   s.apply_s(g.targets[0]); break;
        case GateKind::SDG: s.apply_sdg(g.targets[0]); break;
        case GateKind::X:   s.apply_x(g.targets[0]); break;
        case GateKind::Y:   s.apply_y(g.targets[0]); break;
        case GateKind::Z:   s.apply_z(g.targets[0]); break;
        case GateKind::CX:  s.apply_cx(g.targets[0], g.targets[1]); break;
        case GateKind::CZ:  s.apply_cz(g.targets[0], g.targets[1]); break;
        default: throw std::logic_error("compose_carried_bare: unexpected output gate (engine bug)");
    }
}

}  // namespace

CarriedComposeResult compose_carried_bare(const FramedSuperposition& S_in, const Circuit& deferred) {
    CarriedComposeResult out;
    const int n = S_in.n();
    if (deferred.n != n) throw std::invalid_argument("compose_carried_bare: the circuit and the state differ in width");

    // ── 1. U = gp·C·T, every even entry settled into C, equal axes merged (the MERGED odd term list) ────────────
    PauliRotationForm f = build_pauli_rotation_form(deferred);
    if (f.rejected) {
        out.rejected = true;
        out.reason = f.reject_reason;
        out.reject_terms = f.reject_terms;
        out.reject_gate_index = f.reject_gate_index;
        return out;
    }
    for (int round = 0;; ++round) {
        bool even = false;
        for (const auto& e : f.terms) if (e.coeff % 2 == 0) { even = true; break; }
        if (even) settle_even_entries(f);
        std::map<std::string, size_t> at;
        std::vector<PauliRotationForm::Entry> merged;
        for (const auto& e : f.terms) {
            auto sp = split_sign_(e.pauli);
            const int c = sp.second > 0 ? e.coeff : -e.coeff;
            const std::string k = pauli_key(sp.first);
            auto it = at.find(k);
            if (it == at.end()) { at.emplace(k, merged.size()); merged.push_back({sp.first, mod16_(c)}); }
            else merged[it->second].coeff = mod16_(merged[it->second].coeff + c);
        }
        f.terms.clear();
        bool left = false;
        for (auto& e : merged)
            if (e.coeff != 0) { left |= (e.coeff % 2 == 0); f.terms.push_back(std::move(e)); }
        if (!left) break;
        if (round > 8) throw std::logic_error("compose_carried_bare: even entries do not settle (engine bug)");
    }
    for (size_t i = 0; i < f.terms.size(); ++i)
        for (size_t j = i + 1; j < f.terms.size(); ++j)
            if (!Pauli::commute(f.terms[i].pauli, f.terms[j].pauli)) {
                out.rejected = true;
                out.reason = "rotation-form terms anticommute";
                out.reject_terms = {f.terms[i].pauli, f.terms[j].pauli};
                return out;
            }
    const int m = (int)f.terms.size();
    out.axes = m;

    // ── 2. frame coordinates; the span criterion and the gauge ──────────────────────────────────────────────
    FramedSuperposition S = S_in;
    const std::vector<int> F = S.free;
    const int kF = (int)F.size();
    std::vector<uint8_t> isF((size_t)(n > 0 ? n : 1), 0);
    for (int r : F) isF[(size_t)r] = 1;
    S.U.ensure_dual();
    std::vector<Pauli> Q((size_t)m);
    for (int j = 0; j < m; ++j) Q[(size_t)j] = S.U.conjugate(f.terms[(size_t)j].pauli);
    const int MW = (m + 63) / 64, NWn = (n + 63) / 64;
    auto colx = [&](int a) {
        std::vector<uint64_t> c((size_t)(MW > 0 ? MW : 1), 0);
        for (int j = 0; j < m; ++j) if (Q[(size_t)j].xbit(a)) c[(size_t)(j >> 6)] |= 1ULL << (j & 63);
        return c;
    };
    struct BRow { std::vector<uint64_t> v; int piv; std::vector<uint64_t> comb; };
    std::vector<BRow> basis;
    auto reduce = [&](std::vector<uint64_t>& v, std::vector<uint64_t>& comb) {
        for (const BRow& b : basis)
            if ((v[(size_t)(b.piv >> 6)] >> (b.piv & 63)) & 1ULL) {
                for (size_t w = 0; w < v.size(); ++w) v[w] ^= b.v[w];
                for (size_t w = 0; w < comb.size(); ++w) comb[w] ^= b.comb[w];
            }
    };
    auto lowbit = [](const std::vector<uint64_t>& v) {
        for (size_t w = 0; w < v.size(); ++w) if (v[w]) return (int)(w * 64 + (size_t)__builtin_ctzll(v[w]));
        return -1;
    };
    if (m > 0) {
        for (int a = 0; a < n; ++a) {
            if (isF[(size_t)a]) continue;
            std::vector<uint64_t> v = colx(a);
            if (lowbit(v) < 0) continue;
            std::vector<uint64_t> comb((size_t)NWn, 0);
            comb[(size_t)(a >> 6)] |= 1ULL << (a & 63);
            reduce(v, comb);
            const int p = lowbit(v);
            if (p >= 0) basis.push_back({std::move(v), p, std::move(comb)});
        }
    }
    std::vector<std::vector<uint64_t>> sF((size_t)kF);
    for (int d = 0; d < kF && m > 0; ++d) {
        std::vector<uint64_t> v = colx(F[(size_t)d]);
        const int j0 = lowbit(v);
        std::vector<uint64_t> comb((size_t)NWn, 0);
        reduce(v, comb);
        if (lowbit(v) >= 0) {                        // some combination of the axes has its X-part on F only
            out.rejected = true;
            out.span_refusal = true;
            out.span_row = F[(size_t)d];
            out.span_term = f.terms[(size_t)j0].pauli;
            out.reason = "a consumer non-Clifford axis flips a carried amplitude coordinate (carried free row " +
                         std::to_string(F[(size_t)d]) + "): no gauge puts every merged axis's X-part on constant rows";
            return out;
        }
        sF[(size_t)d] = std::move(comb);
    }
    for (int d = 0; d < kF; ++d) {
        if (sF[(size_t)d].empty()) continue;
        const int fr = F[(size_t)d];
        for (int c = 0; c < n; ++c) {
            if (!((sF[(size_t)d][(size_t)(c >> 6)] >> (c & 63)) & 1ULL)) continue;
            S.U.Zrow[(size_t)fr] = Pauli::multiply(S.U.Zrow[(size_t)fr], S.U.Zrow[(size_t)c]);
            S.eps[(size_t)fr] ^= S.eps[(size_t)c];
            S.U.Xrow[(size_t)c] = Pauli::multiply(S.U.Xrow[(size_t)c], S.U.Xrow[(size_t)fr]);
            ++out.gauge_moves;
        }
    }
    if (out.gauge_moves) {
        S.U.invalidate_dual();
        S.U.ensure_dual();
        for (int j = 0; j < m; ++j) Q[(size_t)j] = S.U.conjugate(f.terms[(size_t)j].pauli);
    }
    for (int j = 0; j < m; ++j)
        for (int r : F)
            if (Q[(size_t)j].xbit(r)) throw std::logic_error("compose_carried_bare: the gauge left an X-part on a free row (engine bug)");

    // ── 3. the frame-coordinate layer X^eps·T̂ : axes with the X^eps sign, split into (rest, Z on F) ─────────────
    std::vector<PauliRotationForm::Entry> rest;
    std::vector<Pauli> zF;
    rest.reserve((size_t)m);
    for (int j = 0; j < m; ++j) {
        auto sp = split_sign_(Q[(size_t)j]);
        int c = sp.second > 0 ? f.terms[(size_t)j].coeff : -f.terms[(size_t)j].coeff;
        int par = 0;
        for (int a = 0; a < n; ++a) if (sp.first.zbit(a) && S.eps[(size_t)a]) par ^= 1;
        if (par) c = -c;
        Pauli R = sp.first, Zf(n);
        for (int r : F) if (R.zbit(r)) { R.flipz(r); Zf.setz(r); }
        rest.push_back({R, mod16_(c)});
        zF.push_back(std::move(Zf));
    }
    // C' acts on the constant rows only: diagonalize on the compacted register of the non-F rows, then map back.
    std::vector<int> cidx;                            // compact index -> row
    std::vector<int> crow((size_t)(n > 0 ? n : 1), -1);
    for (int a = 0; a < n; ++a) if (!isF[(size_t)a]) { crow[(size_t)a] = (int)cidx.size(); cidx.push_back(a); }
    const int nr = (int)cidx.size();
    std::vector<PauliRotationForm::Entry> rest_c;
    rest_c.reserve(rest.size());
    for (const auto& e : rest) {
        Pauli c(nr);
        c.phase = e.pauli.phase;
        for (int t = 0; t < nr; ++t) {
            if (e.pauli.xbit(cidx[(size_t)t])) c.setx(t);
            if (e.pauli.zbit(cidx[(size_t)t])) c.setz(t);
        }
        rest_c.push_back({std::move(c), e.coeff});
    }
    DiagResult dgc = diagonalize_commuting_layer(rest_c, nr);
    if ((int)dgc.tz.size() != m) throw std::logic_error("compose_carried_bare: the diagonalizer changed the term count");
    DiagResult dg;
    for (const CliffGate& g : dgc.cprime) {
        CliffGate h = g;
        for (int& q : h.targets) q = cidx[(size_t)q];
        dg.cprime.push_back(std::move(h));
    }
    for (int j = 0; j < m; ++j) {
        DiagResult::ZTerm zt;
        zt.coeff = dgc.tz[(size_t)j].coeff;
        zt.zstring = Pauli(n);
        zt.zstring.phase = dgc.tz[(size_t)j].zstring.phase;
        for (int t = 0; t < nr; ++t) if (dgc.tz[(size_t)j].zstring.zbit(t)) zt.zstring.setz(cidx[(size_t)t]);
        for (size_t w = 0; w < zt.zstring.z.size(); ++w) zt.zstring.z[w] ^= zF[(size_t)j].z[w];
        dg.tz.push_back(std::move(zt));
    }
    (void)crow;

    // ── 4. the parity support: |+⟩ on every free row (a variable, never a constant), then C'† ───────────────────
    std::vector<CliffGate> gates;
#ifndef XTIM_MUTATE_F_CONST
    for (int r : F) gates.push_back(CliffGate{GateKind::H, {r}});
#endif
    {
        std::vector<CliffGate> cd_ = dagger_cliff_list(dg.cprime);
        gates.insert(gates.end(), cd_.begin(), cd_.end());
    }
    ParitySupport ps = restrict_to_support(gates, dg.tz, n);
    fast_todd_reduce_columns(ps);

    // ── 5. the weighted assembly: branch over span(odd columns ∪ F coordinates) ───────────────────────────────
    const int d = ps.support.k_;
    const int dwords = (d + 63) / 64;
    int gp = mod16_(ps.global_phase_mult8);
    std::map<std::vector<uint64_t>, int> comb;
    for (size_t k = 0; k < ps.columns.size(); ++k) {
        std::vector<uint64_t> col = ps.columns[k];
        if ((int)col.size() < dwords) col.resize((size_t)dwords, 0);
        comb[col] = (comb.count(col) ? comb[col] : 0) + ps.coeffs[k];
    }
    std::vector<std::pair<std::vector<uint64_t>, int>> odd_cols;
    struct EvenCol { std::vector<uint64_t> col; int coeff; };
    std::vector<EvenCol> even_cols;
    for (auto& kv : comb) {
        const int c = mod16_(kv.second);
        if (c == 0) continue;
        if (is_zero_col(kv.first)) { gp = mod16_(gp + c); continue; }
        if (c == 8) { gp = mod16_(gp + 8); continue; }
        if (c % 2 == 1) odd_cols.push_back({kv.first, c});
        else even_cols.push_back({kv.first, c});
    }
    AffineState psi = ps.support;
    RowCombinationSolver rcs;
    for (const auto& e : even_cols) {
        rcs.for_state(psi, dwords);
        std::vector<uint8_t> mm = rcs.solve(e.col);
        int cbit = 0;
        for (int q = 0; q < n; ++q) if (mm[(size_t)q] && psi.b[(size_t)q]) cbit ^= 1;
        const int c2 = cbit ? mod16_(-e.coeff) : e.coeff;
        const int spow = ((-(c2 / 2)) % 4 + 4) % 4;
        apply_parity_s(psi, mm, spow);
        gp = mod16_(gp + c2);
    }
    std::vector<std::vector<uint64_t>> bas;
    std::vector<int> piv;
    auto red = [&](std::vector<uint64_t> v) {
        for (size_t i = 0; i < bas.size(); ++i) if (colbit(v, piv[i])) xor_into(v, bas[i]);
        return v;
    };
    auto add_basis = [&](const std::vector<uint64_t>& col) {
        std::vector<uint64_t> rr = red(col);
        int p = -1;
        for (int j = 0; j < d; ++j) if (colbit(rr, j)) { p = j; break; }
        if (p < 0) return;
        bas.push_back(rr);
        piv.push_back(p);
    };
    for (auto& oc : odd_cols) add_basis(oc.first);
    const int r_odd = (int)bas.size();
    // the F coordinates as functionals of the support variables: x_f = b_f ⊕ λ_f · t
    std::vector<std::vector<uint64_t>> lam((size_t)kF);
    std::vector<int> boff((size_t)kF, 0);
#ifndef XTIM_MUTATE_F_CONST
    for (int t = 0; t < kF; ++t) {
        const int r = F[(size_t)t];
        lam[(size_t)t].assign((size_t)(dwords > 0 ? dwords : 1), 0);
        const uint64_t* row = psi.R.row(r);
        for (int w = 0; w < dwords; ++w) lam[(size_t)t][(size_t)w] = row[w];
        if (dwords > 0 && (d & 63)) lam[(size_t)t][(size_t)(dwords - 1)] &= (1ULL << (d & 63)) - 1;
        boff[(size_t)t] = psi.b[(size_t)r];
        add_basis(lam[(size_t)t]);
    }
#endif
    const int r = (int)bas.size();
    (void)r_odd;
    if (r > kMaxBareRank) throw BareStateCapacityError(bare_rank_refusal(r, "compose_carried_bare"));
    const int chi = 1 << r;
    StateBudgetContext budget_ctx(chi, r, n);
    const int64_t kTableBytesPerBranch = 16 + 4 + 1 + 32;
    state_budget_precheck(
        (int64_t)chi * ((int64_t)(sizeof(CanonicalStabSum::Branch) + kSigmaHeapBytes) +
                        (int64_t)sizeof(FramedSuperposition::Entry) + kTableBytesPerBranch),
        "composing the carried bare state");
    StateBudgetTicket build_tables((int64_t)chi * kTableBytesPerBranch, "the carried bare-state build tables");
    auto in_basis = [&](std::vector<uint64_t> v) {
        uint32_t mask = 0;
        for (int i = 0; i < r; ++i)
            if (colbit(v, piv[(size_t)i])) { mask |= 1u << i; xor_into(v, bas[(size_t)i]); }
        if (!is_zero_col(v)) throw std::logic_error("compose_carried_bare: a column outside the branch basis (engine bug)");
        return mask;
    };
    std::vector<uint32_t> odd_mask;
    std::vector<cd> zplus, zminus;
    for (auto& oc : odd_cols) {
        odd_mask.push_back(in_basis(oc.first));
        zplus.push_back(zeta16(oc.second));
        zminus.push_back(zeta16(-oc.second));
    }
    std::vector<uint32_t> lmask((size_t)kF, 0);
#ifndef XTIM_MUTATE_F_CONST
    for (int t = 0; t < kF; ++t) lmask[(size_t)t] = in_basis(lam[(size_t)t]);
#endif
    std::vector<std::vector<uint8_t>> basis_m((size_t)r);
    std::vector<int> basis_cbit((size_t)r, 0);
    for (int i = 0; i < r; ++i) {
        rcs.for_state(psi, dwords);
        basis_m[(size_t)i] = rcs.solve(bas[(size_t)i]);
        int cb = 0;
        for (int q = 0; q < n; ++q) if (basis_m[(size_t)i][(size_t)q] && psi.b[(size_t)q]) cb ^= 1;
        basis_cbit[(size_t)i] = cb;
    }
    auto make_ray = [&](uint32_t sigma, ExactPhase* fac_out) {
        auto ray = std::make_unique<AffineState>(psi);
        ExactPhase fac = ExactPhase::one();
        for (int i = 0; i < r; ++i) {
            const int phys = (int)((sigma >> i) & 1u) ^ basis_cbit[(size_t)i];
            fac = fac.mul(project_parity(*ray, basis_m[(size_t)i], phys));
        }
        if (fac_out) *fac_out = fac;
        return ray;
    };
    ExactPhase fac0;
    { auto r0 = make_ray(0, &fac0); }
    auto checked_ray = [&](uint32_t sigma) {
        ExactPhase fx;
        auto ray = make_ray(sigma, &fx);
        if (!fx.exact_eq(fac0))
            throw BareStateInvariantError("compose_carried_bare: coset projection amplitude depends on the branch");
        return ray;
    };
    std::unordered_map<AmpKey, cd> alpha;
    for (const auto& e : S.entries()) alpha[e.first] += e.second;
    const double wF = std::pow(std::sqrt(2.0), kF);
    std::vector<cd> coeffs((size_t)chi);
    const cd base = zeta16(gp) * fac0.to_complex();
    for (uint32_t sg = 0; sg < (uint32_t)chi; ++sg) {
        cd w = base;
        for (size_t k = 0; k < odd_mask.size(); ++k)
            w = chain_mul(w, (__builtin_popcount(odd_mask[k] & sg) & 1) ? zminus[k] : zplus[k]);
#ifndef XTIM_MUTATE_F_CONST
        AmpKey key = 0;
        for (int t = 0; t < kF; ++t)
            if ((boff[(size_t)t] ^ (__builtin_popcount(lmask[(size_t)t] & sg) & 1)) & 1) key |= AmpKey(1) << t;
        auto it = alpha.find(key);
        w = it == alpha.end() ? cd(0.0, 0.0) : w * it->second * wF;
#endif
        coeffs[(size_t)sg] = w;
    }
    std::vector<Pauli> zs((size_t)r, Pauli(n));
    for (int i = 0; i < r; ++i)
        for (int q = 0; q < n; ++q) if (basis_m[(size_t)i][(size_t)q]) zs[(size_t)i].setz(q);
    CanonicalStabSum css = CanonicalStabSum::from_coset_family(n, r, psi, zs, checked_ray, coeffs, nullptr);
    for (const CliffGate& g : dg.cprime) apply_gate_css_(css, g);
    for (int a = 0; a < n; ++a) if (S.eps[(size_t)a]) css.apply_x(a);

    // ── 6. back to the physical frame: U_in (gauged) composed, then C, then the global phase ────────────────────
    FramedSuperposition B = FramedSuperposition::from_css(std::move(css));
    {
        auto& E = B.entries();
        E.erase(std::remove_if(E.begin(), E.end(), [](const auto& e) { return e.second == cd(0.0, 0.0); }), E.end());
    }
    for (int a = 0; a < n; ++a) {
        B.U.Zrow[(size_t)a] = S.U.forward_image(B.U.Zrow[(size_t)a]);
        B.U.Xrow[(size_t)a] = S.U.forward_image(B.U.Xrow[(size_t)a]);
    }
    B.U.invalidate_dual();
    for (const CliffGate& g : f.cliffords)
        B.apply_clifford(clif_code_(g.kind), g.targets[0], g.targets.size() > 1 ? g.targets[1] : 0);
    {
        cd gph = f.global_phase;
        const double mag = std::abs(gph);
        if (mag > 1e-300) gph /= mag;
        for (auto& e : B.entries()) e.second *= gph;
    }
    out.state = std::move(B);
    return out;
}

}  // namespace qeccore
