// output_state.cpp — the output-state helpers' engine.
#include "qeccore/output_state.hpp"

#include "qeccore/clifford_tableau.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/parity_support.hpp"
#include "qeccore/stab_affine.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace qeccore {
namespace {
inline int mod(int v, int m) { return ((v % m) + m) % m; }
inline bool bit(const Row& r, int i) { return (r[(size_t)(i >> 6)] >> (i & 63)) & 1ULL; }
inline void flip(Row& r, int i) { r[(size_t)(i >> 6)] ^= 1ULL << (i & 63); }
inline void xor_into(Row& a, const Row& b) { for (size_t w = 0; w < a.size() && w < b.size(); ++w) a[w] ^= b[w]; }
inline bool zero_row(const Row& r) { for (uint64_t w : r) if (w) return false; return true; }
inline int first_bit(const Row& r) {
    for (size_t w = 0; w < r.size(); ++w) if (r[w]) return (int)(w * 64) + __builtin_ctzll(r[w]);
    return -1;
}
inline bool dot(const Row& a, const Row& b) {
    uint64_t s = 0;
    for (size_t w = 0; w < a.size() && w < b.size(); ++w) s ^= a[w] & b[w];
    return __builtin_popcountll(s) & 1;
}
inline int64_t sym_index(int i, int j, int r) { return (int64_t)i * r - (int64_t)i * (i - 1) / 2 + (j - i); }   // i <= j

void merge_columns(std::vector<Row>& cols, std::vector<int>& coeffs, int dw) {
    std::map<Row, int> comb;
    std::vector<Row> order;
    for (size_t k = 0; k < cols.size(); ++k) {
        Row c = cols[k];
        c.resize((size_t)dw, 0);
        if (zero_row(c)) continue;
        auto it = comb.find(c);
        if (it == comb.end()) { comb.emplace(c, mod(coeffs[k], 16)); order.push_back(std::move(c)); }
        else it->second = mod(it->second + coeffs[k], 16);
    }
    cols.clear();
    coeffs.clear();
    for (Row& c : order) {
        const int v = comb[c];
        if (v == 0 || v == 8) continue;
        coeffs.push_back(v);
        cols.push_back(std::move(c));
    }
}
// The kernel in the residue coordinates of P, which the caller builds from `odd` (in order) and keeps: the shift step
// reads w and lambda off the same P, so the coordinates cannot drift between two independently built projectors (M2).
std::vector<Row> shift_kernel_in(const std::vector<Row>& odd, int d, Projector& P) {
    P = Projector{};
    P.words = std::max(1, (d + 63) / 64);
    for (const Row& c : odd) { Row cc = c; cc.resize((size_t)P.words, 0); P.extend(cc); }
    const int r = P.r();
    if ((int)odd.size() == r) return {};
    const int64_t nsym = (int64_t)r * (r + 1) / 2;
    const size_t sw = (size_t)((nsym + 63) / 64);
    std::vector<Row> acol((size_t)r, Row(sw, 0));
    std::vector<int> on;
    for (const Row& c : odd) {
        Row cc = c;
        cc.resize((size_t)P.words, 0);
        const Row h = P.project_wide(cc);
        on.clear();
        for (int i = 0; i < r; ++i) if (bit(h, i)) on.push_back(i);
        Row s(sw, 0);
        for (size_t a = 0; a < on.size(); ++a)
            for (size_t b = a; b < on.size(); ++b) {
                const int64_t ix = sym_index(on[a], on[b], r);
                s[(size_t)(ix >> 6)] ^= 1ULL << (ix & 63);
            }
        for (int i : on) xor_into(acol[(size_t)i], s);
    }
    struct E { Row v, comb; int piv; };
    std::vector<E> basis;
    std::vector<Row> ker;
    const size_t rw = (size_t)((r + 63) / 64);
    for (int i = 0; i < r; ++i) {
        Row v = acol[(size_t)i], comb(rw, 0);
        flip(comb, i);
        for (const E& e : basis) if (bit(v, e.piv)) { xor_into(v, e.v); xor_into(comb, e.comb); }
        const int p = first_bit(v);
        if (p < 0) { ker.push_back(std::move(comb)); continue; }
        basis.push_back(E{std::move(v), std::move(comb), p});
    }
    return ker;
}
inline bool xz_zero(const Pauli& p) { for (uint64_t w : p.x) if (w) return false; for (uint64_t w : p.z) if (w) return false; return true; }
inline Pauli unsigned_of(Pauli p) { p.phase = mod(p.xz_overlap(), 4); return p; }
inline bool sbit(const Pauli& p, int i) {
    const int nx = (int)p.x.size() * 64;
    return i < nx ? ((p.x[(size_t)(i >> 6)] >> (i & 63)) & 1ULL) : ((p.z[(size_t)((i - nx) >> 6)] >> ((i - nx) & 63)) & 1ULL);
}
inline int first_sbit(const Pauli& p) {
    for (size_t w = 0; w < p.x.size(); ++w) if (p.x[w]) return (int)(w * 64) + __builtin_ctzll(p.x[w]);
    for (size_t w = 0; w < p.z.size(); ++w) if (p.z[w]) return (int)((p.x.size() + w) * 64) + __builtin_ctzll(p.z[w]);
    return -1;
}
struct SignedSpan {                     // RREF-by-insertion span of commuting signed Paulis; reduce() multiplies on the right
    struct E { Pauli p; int piv; };
    std::vector<E> rows;
    Pauli reduce(Pauli p) const { for (const E& e : rows) if (sbit(p, e.piv)) p = Pauli::multiply(p, e.p); return p; }
    bool add(Pauli p) { p = reduce(std::move(p)); const int v = first_sbit(p); if (v < 0) return false; rows.push_back(E{std::move(p), v}); return true; }
};
template <class Key>
std::vector<Pauli> kernel_products(const std::vector<Pauli>& gens, Key key) {   // basis of {g in <gens> : key(g) = 0}
    struct E { Row v; Pauli p; int piv; };
    std::vector<E> basis;
    std::vector<Pauli> ker;
    for (const Pauli& g : gens) {
        Row v = key(g);
        Pauli p = g;
        for (const E& e : basis) if (bit(v, e.piv)) { xor_into(v, e.v); p = Pauli::multiply(p, e.p); }
        const int piv = first_bit(v);
        if (piv < 0) { if (!xz_zero(p)) ker.push_back(std::move(p)); continue; }
        basis.push_back(E{std::move(v), std::move(p), piv});
    }
    return ker;
}
void fold_even(std::vector<Pauli>& gens, const Pauli& A, int e) {    // Rule R2: exp(-i e pi/8 A), e even
    e = mod(e, 8);
    if (e == 0) return;
    for (Pauli& g : gens) {
        if (Pauli::commute(g, A)) continue;
        if (e == 4) { g.phase = mod(g.phase + 2, 4); continue; }
        Pauli ag = Pauli::multiply(A, g);
        ag.phase = mod(ag.phase + (e == 2 ? 3 : 1), 4);
        g = std::move(ag);
    }
}
void left_apply(CliffordTableau& T, const CliffGate& g) {
    switch (g.kind) {
        case GateKind::H: T.left_h(g.a()); break;   case GateKind::S: T.left_s(g.a()); break;
        case GateKind::SDG: T.left_sdg(g.a()); break; case GateKind::X: T.left_x(g.a()); break;
        case GateKind::Y: T.left_y(g.a()); break;   case GateKind::Z: T.left_z(g.a()); break;
        case GateKind::CX: T.left_cx(g.a(), g.b()); break; case GateKind::CZ: T.left_cz(g.a(), g.b()); break;
        default: throw std::logic_error("output_state: a non-Clifford gate in a Clifford list (engine bug)");
    }
}
Pauli to_local(const Pauli& p, const std::vector<int>& wires) {
    Pauli l((int)wires.size());
    for (size_t i = 0; i < wires.size(); ++i) { if (p.xbit(wires[i])) l.setx((int)i); if (p.zbit(wires[i])) l.setz((int)i); }
    l.phase = p.phase;
    return l;
}
struct ZSolver {                        // Z(m) with m.R = col (pivot qubits of the support's R), and the offset bit m.b
    const AffineState& s;
    int n, d, dw, nw;
    std::vector<Row> val, mask;
    std::vector<int> piv;
    explicit ZSolver(const AffineState& st) : s(st), n(st.n_), d(st.k_), dw(std::max(1, (st.k_ + 63) / 64)), nw((st.n_ + 63) / 64) {
        for (int q = 0; q < n && (int)piv.size() < d; ++q) {
            Row v((size_t)dw, 0);
            const uint64_t* r = st.R.row(q);
            for (int w = 0; w < (d + 63) / 64; ++w) v[(size_t)w] = r[w];
            Row m((size_t)nw, 0);
            flip(m, q);
            for (size_t i = 0; i < piv.size(); ++i) if (bit(v, piv[i])) { xor_into(v, val[i]); xor_into(m, mask[i]); }
            const int p = first_bit(v);
            if (p < 0) continue;
            val.push_back(std::move(v)); mask.push_back(std::move(m)); piv.push_back(p);
        }
    }
    Pauli zstring(const Row& col) const {
        Row v = col; v.resize((size_t)dw, 0);
        Row m((size_t)nw, 0);
        for (size_t i = 0; i < piv.size(); ++i) if (bit(v, piv[i])) { xor_into(v, val[i]); xor_into(m, mask[i]); }
        if (!zero_row(v)) throw std::logic_error("output_state: a column outside the support's span (engine bug)");
        Pauli z(n); z.z = m; return z;
    }
    bool offset(const Pauli& z) const { int b = 0; for (int q = 0; q < n; ++q) if (z.zbit(q)) b ^= s.b[(size_t)q] != 0; return b != 0; }
};
// Final review N4: reduced row echelon form of commuting generators on the symplectic X|Z matrix, columns X_0..X_{n-1}
// then Z_0..Z_{n-1} (ascending local index = ascending user qubit). Each pivot column has exactly one 1; rows come out
// in pivot order. A CSS group comes back as pure X-checks then pure Z-checks. Row operations are exact signed products
// (the generators commute, so every product stays Hermitian); the signed group is unchanged.
void rref_generators(std::vector<Pauli>& g, int n) {
    size_t r = 0;
    for (int col = 0; col < 2 * n && r < g.size(); ++col) {
        const bool isx = col < n;
        const int q = isx ? col : col - n;
        const auto has = [&](const Pauli& p) { return isx ? p.xbit(q) : p.zbit(q); };
        size_t piv = r;
        while (piv < g.size() && !has(g[piv])) ++piv;
        if (piv == g.size()) continue;
        std::swap(g[r], g[piv]);
        for (size_t i = 0; i < g.size(); ++i) if (i != r && has(g[i])) g[i] = Pauli::multiply(g[i], g[r]);
        ++r;
    }
    if (r != g.size()) throw std::logic_error("output_state: dependent Pauli stabilizer generators (engine bug)");
}
}  // namespace

std::vector<Row> magic_shift_kernel(const std::vector<Row>& odd, int d) {
    Projector P;
    return shift_kernel_in(odd, d, P);
}

int stabilize_magic_columns(std::vector<Row>& cols, std::vector<int>& coeffs, int d) {
    const int dw = std::max(1, (d + 63) / 64);
    for (int steps = 0;; ++steps) {
        merge_columns(cols, coeffs, dw);
        std::vector<Row> odd, ev;
        std::vector<int> oc, ec;
        for (size_t k = 0; k < cols.size(); ++k) {
            if (coeffs[k] & 1) { odd.push_back(cols[k]); oc.push_back(coeffs[k]); }
            else { ev.push_back(cols[k]); ec.push_back(coeffs[k]); }
        }
        Projector P;
        const std::vector<Row> Y = shift_kernel_in(odd, d, P);
        if (Y.empty()) return steps;
        const Row& ys = Y.front();
        Row w((size_t)dw, 0);
        int j = -1;
        for (int i = 0; i < P.r(); ++i)
            if (bit(ys, i)) { flip(w, P.pivot[(size_t)i]); if (j < 0) j = i; }
        const Row lam = P.basis[(size_t)j];
        int kappa = 0;
        std::vector<int> acc((size_t)d, 0);
        for (size_t k = 0; k < odd.size(); ++k) {
            if (!dot(odd[k], w)) continue;
            const int a = mod(-oc[k], 8);
            kappa = mod(kappa + a, 8);
            for (int t = 0; t < d; ++t) if (bit(odd[k], t)) acc[(size_t)t] = mod(acc[(size_t)t] + a, 4);
        }
        Row mu((size_t)dw, 0);
        for (int t = 0; t < d; ++t) {
            if (acc[(size_t)t] & 1)
                throw std::logic_error("stabilize_magic_columns: an odd linear coefficient on the kernel (engine bug)");
            if (acc[(size_t)t] == 2) flip(mu, t);
        }
        const int beta = dot(mu, w) ? 1 : 0;
        for (size_t k = 0; k < odd.size(); ++k) if (dot(odd[k], w)) xor_into(odd[k], lam);
        auto emit = [&](Row dir, int a8) {
            a8 = mod(a8, 8);
            if (a8 == 0 || zero_row(dir)) return;
            ev.push_back(std::move(dir));
            ec.push_back(mod(8 - a8, 8));
        };
        Row lm = lam;
        xor_into(lm, mu);
        emit(lam, kappa + 4 * beta + 2);
        emit(mu, 2);
        emit(lm, 6);
        cols = std::move(odd);
        coeffs = std::move(oc);
        cols.insert(cols.end(), ev.begin(), ev.end());
        coeffs.insert(coeffs.end(), ec.begin(), ec.end());
    }
}

std::vector<QubitEnd> qubit_ends(const ParsedStim& ps) {
    const std::vector<Instr>& st = ps.circuit.stream;
    if (ps.stream_step.size() != st.size()) throw std::logic_error("qubit_ends: parsed without track_source (internal)");
    const int nu = std::max(ps.num_user_qubits, 0);
    std::vector<QubitEnd> end((size_t)nu, QubitEnd::Untouched);
    std::vector<int> reset_step((size_t)nu, -1);
    for (size_t i = 0; i < st.size(); ++i) {
        const Instr& ins = st[i];
        const std::vector<int>* qs = ins.kind == Instr::Kind::Gate ? &ins.targets
                                   : (ins.kind == Instr::Kind::Measure || ins.kind == Instr::Kind::Reset ||
                                      ins.kind == Instr::Kind::ControlledPauli) ? &ins.qubits : nullptr;
        if (!qs) continue;
        for (int q : *qs) {
            if (q < 0 || q >= nu) continue;
            switch (ins.kind) {
                case Instr::Kind::Measure: end[(size_t)q] = QubitEnd::Measured; break;
                case Instr::Kind::Reset: end[(size_t)q] = QubitEnd::Reset; reset_step[(size_t)q] = ps.stream_step[i]; break;
                default: end[(size_t)q] = reset_step[(size_t)q] == ps.stream_step[i] ? QubitEnd::Reset : QubitEnd::Gate;
            }
        }
    }
    return end;
}

std::vector<int> output_candidates(const ParsedStim& ps) {
    const std::vector<QubitEnd> e = qubit_ends(ps);
    std::vector<int> out;
    for (size_t q = 0; q < e.size(); ++q) if (e[q] == QubitEnd::Gate) out.push_back((int)q);
    return out;
}

int hermitian_sign(const Pauli& p) {
    const int d = mod(p.phase - p.xz_overlap(), 4);
    if (d & 1) throw std::logic_error("output_state: a non-Hermitian Pauli (engine bug)");
    return d == 0 ? 1 : -1;
}

OutputAnalysis analyze_output(const Circuit& deferred, const std::vector<int>& out_wires, bool want_state) {
    OutputAnalysis res;
    {                                                     // Task 2 review m1: a malformed list is refused by name
        std::vector<char> seen((size_t)std::max(deferred.n, 0), 0);
        for (int w : out_wires) {
            if (w < 0 || w >= deferred.n)
                throw std::invalid_argument("malformed qubit list: wire " + std::to_string(w) +
                                            " is out of range (the circuit has " + std::to_string(deferred.n) +
                                            " wires)");
            if (seen[(size_t)w]++)
                throw std::invalid_argument("malformed qubit list: wire " + std::to_string(w) +
                                            " is listed twice");
        }
    }
    ParityForm pf = build_parity_form(deferred);   // the ladder settle is inside (normal_form.hpp)
    if (pf.rejected) { res.reject_reason = pf.reject_reason; res.reject_terms = pf.reject_terms; return res; }   // OutOfClass
    ParitySupport& ps = *pf.ps;
    const int n = deferred.n, W = (n + 63) / 64, nO = (int)out_wires.size();
    res.shift_steps = stabilize_magic_columns(ps.columns, ps.coeffs, ps.support.k_);            // Lemma 1 precondition

    // ── 3. |S> (even columns folded in the psi frame), the odd axes, the end frame ──
    CliffordTableau Tpsi(n);
    for (const CliffGate& g : pf.cprime_dagger) left_apply(Tpsi, g);
    std::vector<Pauli> S = Tpsi.Zrow;
    const ZSolver zs(ps.support);
    std::vector<Pauli> zodd;
    std::vector<int> eodd;
    std::vector<Row> oddcols;
    for (size_t k = 0; k < ps.columns.size(); ++k) {
        Pauli zm = zs.zstring(ps.columns[k]);
        const int e = mod(zs.offset(zm) ? ps.coeffs[k] : -ps.coeffs[k], 8);                  // zeta16^{c(-1)^.} = exp(-i e pi/8 Z(m))
        if (e % 2 == 0) fold_even(S, zm, e);
        else { zodd.push_back(std::move(zm)); eodd.push_back(e); oddcols.push_back(ps.columns[k]); }
    }
    CliffordTableau TV(n);
    for (const CliffGate& g : pf.output_clifford) left_apply(TV, g);
    for (Pauli& g : S) g = TV.forward_image(g);
    std::vector<Pauli> axes;
    std::vector<int> cnt;
    for (size_t k = 0; k < zodd.size(); ++k) {
        const Pauli Q = TV.forward_image(zodd[k]);
        axes.push_back(unsigned_of(Q));
        cnt.push_back(mod(hermitian_sign(Q) * eodd[k], 8));
    }
    std::vector<uint64_t> om((size_t)W, 0);
    for (int w : out_wires) om[(size_t)(w >> 6)] |= 1ULL << (w & 63);
    const auto side = [&](const Pauli& P, bool rest) {
        Pauli r(n);
        for (int w = 0; w < W; ++w) { const uint64_t m = rest ? ~om[(size_t)w] : om[(size_t)w]; r.x[(size_t)w] = P.x[(size_t)w] & m; r.z[(size_t)w] = P.z[(size_t)w] & m; }
        return r;
    };
    const auto vanish_on = [&](bool rest) {           // key: the part on that side (it must vanish)
        return [&, rest](const Pauli& g) { const Pauli r = side(g, rest); Row v(r.x); v.insert(v.end(), r.z.begin(), r.z.end()); return v; };
    };
    const auto cross = [&](const std::vector<Pauli>& all, const std::vector<Pauli>& a, const std::vector<Pauli>& b) {
        SignedSpan sp;
        for (const Pauli& g : a) sp.add(g);
        for (const Pauli& g : b) sp.add(g);
        std::vector<Pauli> out;
        for (const Pauli& g : all) if (sp.add(g)) out.push_back(g);
        return out;
    };

    // ── 4. G = stab(S) ∩ C(axes) (Lemma 1); does it split (Lemma 5)? ──
    const auto comm = [&](const Pauli& g) {
        Row v(axes.size() / 64 + 1, 0);
        for (size_t k = 0; k < axes.size(); ++k) if (!Pauli::commute(g, axes[k])) flip(v, (int)k);
        return v;
    };
    const std::vector<Pauli> G = kernel_products(S, comm);
    const std::vector<Pauli> GO = kernel_products(G, vanish_on(true)), GR = kernel_products(G, vanish_on(false));
    if (GO.size() + GR.size() < G.size()) {
        res.verdict = OutputVerdict::CrossStabilizer;
        res.witnesses = cross(G, GO, GR);
        return res;
    }

    // ── 5. classify the axes: local-O, local-R, straddling ──
    SignedSpan sGO, sGR;
    for (const Pauli& g : GO) sGO.add(g);
    for (const Pauli& g : GR) sGR.add(g);
    std::vector<int> straddle, local_o;
    std::vector<int> qphase(axes.size(), 0);                 // local-O: A_R = i^q h, h in G_R
    for (size_t k = 0; k < axes.size(); ++k) {
        const Pauli rr = sGR.reduce(side(axes[k], true));
        if (xz_zero(rr)) { local_o.push_back((int)k); qphase[k] = rr.phase; continue; }
        if (xz_zero(sGO.reduce(side(axes[k], false)))) continue;                                  // local-R
        straddle.push_back((int)k);
    }
    if (!straddle.empty()) {
        const int dw = std::max(1, (ps.support.k_ + 63) / 64);
        for (int k : straddle) {                              // Lemma 4a: an independent straddling column
            Projector P;
            P.words = dw;
            for (size_t j = 0; j < oddcols.size(); ++j) if ((int)j != k) { Row c = oddcols[j]; c.resize((size_t)dw, 0); P.extend(c); }
            Row ck = oddcols[(size_t)k];
            ck.resize((size_t)dw, 0);
            if (P.extend(ck)) { res.verdict = OutputVerdict::CoupledMagic; res.witnesses = {axes[(size_t)k]}; return res; }
        }
        res.verdict = OutputVerdict::Undecided;               // A3
        for (int k : straddle) res.witnesses.push_back(axes[(size_t)k]);
        return res;
    }

    // ── 6. every axis local: deterministic iff |S> splits (Lemma 7) ──
    const std::vector<Pauli> SOg = kernel_products(S, vanish_on(true)), SRg = kernel_products(S, vanish_on(false));
    if ((int)(SOg.size() + SRg.size()) != n) {
        res.verdict = OutputVerdict::CoupledStabilizer;
        res.witnesses = cross(S, SOg, SRg);
        return res;
    }
    res.verdict = OutputVerdict::Deterministic;
    for (const Pauli& g : GO) res.pauli_stabilizers.push_back(to_local(g, out_wires));
    if (!want_state) return res;
    rref_generators(res.pauli_stabilizers, nO);                                                  // N4; checked below

    // ── the output state: S_O, the localized local-O axes, one mod-4 pass, clifford_paulis ──
    std::vector<Pauli> SO;
    for (const Pauli& g : SOg) SO.push_back(to_local(g, out_wires));
    for (int k : local_o) {
        Pauli AO = side(axes[(size_t)k], false);
        AO.phase = mod(axes[(size_t)k].phase + qphase[(size_t)k], 4);                             // A acts as i^{p+q} A_O
        const Pauli loc = to_local(AO, out_wires);
        const Pauli au = unsigned_of(loc);
        const int e = mod(hermitian_sign(loc) * cnt[(size_t)k], 8);
        const int s = (e % 4 == 1) ? 1 : -1;
        fold_even(SO, au, mod(e - s, 8));                                                        // a Pauli remainder (0 or 4)
        Pauli out = au;
        if (s < 0) out.phase = mod(out.phase + 2, 4);
        res.rotations.push_back(std::move(out));
    }
    // invariant: G_O = stab(S_O) ∩ C(rotations), signs included
    std::vector<Pauli> rax;
    for (const Pauli& r : res.rotations) rax.push_back(unsigned_of(r));
    const auto rcomm = [&](const Pauli& g) { Row v(rax.size() / 64 + 1, 0); for (size_t k = 0; k < rax.size(); ++k) if (!Pauli::commute(g, rax[k])) flip(v, (int)k); return v; };
    SignedSpan want, got;
    for (const Pauli& g : res.pauli_stabilizers) want.add(g);
    const std::vector<Pauli> Gs = kernel_products(SO, rcomm);
    for (const Pauli& g : Gs) got.add(g);
    for (const Pauli& g : res.pauli_stabilizers) { const Pauli r = got.reduce(g); if (!xz_zero(r) || mod(r.phase, 4)) throw std::logic_error("output_state: G_O is not in stab(S_O) ∩ C(axes) (engine bug)"); }
    for (const Pauli& g : Gs) { const Pauli r = want.reduce(g); if (!xz_zero(r) || mod(r.phase, 4)) throw std::logic_error("output_state: stab(S_O) ∩ C(axes) exceeds G_O (engine bug)"); }
    SignedSpan PS;
    for (const Pauli& g : res.pauli_stabilizers) PS.add(g);
    for (const Pauli& g : SO) { Pauli r = PS.reduce(g); if (xz_zero(r)) continue; PS.add(r); res.clifford_paulis.push_back(std::move(r)); }
    if ((int)(res.pauli_stabilizers.size() + res.clifford_paulis.size()) != nO)
        throw std::logic_error("output_state: the stabilizer count does not match the output size (engine bug)");
    return res;
}

}  // namespace qeccore
