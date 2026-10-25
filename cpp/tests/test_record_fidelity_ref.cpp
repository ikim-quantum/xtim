// RF-ref: the layout-3 REGISTER RecordFidelity (strict contract; routes 3a / 3b on the Gauss-sum kernel) against an
// INDEPENDENT dense statevector partial-trace oracle (test-only). Every contraction mode (the cost rule, forced direct,
// forced Walsh–Hadamard) and the padded register are checked. Refusals by name: RF_OUTPUT_NOT_STRICT (incl. the
// review's B1 counterexample), RF_RECORD_ON_OUTPUT, RF_NO_OUTPUT, RF_CORRECTION_OFF_OUTPUT, RF_FACTOR_NOT_CLIFFORD.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/record_fidelity.hpp"
#include "qeccore/twirl_kernel.hpp"

using namespace qeccore;
using cd = std::complex<double>;
using Vec = std::vector<cd>;

namespace {

const cd kI[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

Vec apply_pauli(const Vec& v, const Pauli& P) {   // i^ph X^x Z^z
    const int n = P.n;
    size_t xm = 0, zm = 0;
    for (int q = 0; q < n; ++q) { if (P.xbit(q)) xm |= (size_t)1 << q; if (P.zbit(q)) zm |= (size_t)1 << q; }
    Vec o(v.size(), 0.0);
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == 0.0) continue;
        const double s = (__builtin_popcountll(i & zm) & 1) ? -1.0 : 1.0;
        o[i ^ xm] += kI[P.phase & 3] * s * v[i];
    }
    return o;
}
cd inner(const Vec& a, const Vec& b) { cd s = 0.0; for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i]; return s; }
void axpy(Vec& y, cd a, const Vec& x) { for (size_t i = 0; i < y.size(); ++i) y[i] += a * x[i]; }

struct Gate { int kind, a, b; };   // 0 = H, 1 = S, 2 = CX
void sv_gate(Vec& v, const Gate& g) {
    const size_t N = v.size();
    if (g.kind == 0) {
        const size_t m = (size_t)1 << g.a;
        const double r = 1.0 / std::sqrt(2.0);
        for (size_t i = 0; i < N; ++i) if (!(i & m)) { const cd x = v[i], y = v[i | m]; v[i] = r * (x + y); v[i | m] = r * (x - y); }
    } else if (g.kind == 1) {
        const size_t m = (size_t)1 << g.a;
        for (size_t i = 0; i < N; ++i) if (i & m) v[i] *= cd(0, 1);
    } else {
        const size_t c = (size_t)1 << g.a, t = (size_t)1 << g.b;
        for (size_t i = 0; i < N; ++i) if ((i & c) && !(i & t)) std::swap(v[i], v[i | t]);
    }
}
void tab_gate(CliffordTableau& U, const Gate& g) {
    if (g.kind == 0) U.left_h(g.a); else if (g.kind == 1) U.left_s(g.a); else U.left_cx(g.a, g.b);
}
std::vector<Gate> random_gates(std::mt19937_64& R, int n, int L, const std::vector<int>& qs) {
    std::vector<Gate> gs;
    if (qs.empty()) return gs;
    for (int i = 0; i < L; ++i) {
        const int kind = (int)(R() % 3);
        const int a = qs[R() % qs.size()];
        if (kind == 2) {
            if (qs.size() < 2) continue;
            int b = a;
            while (b == a) b = qs[R() % qs.size()];
            gs.push_back({2, a, b});
        } else gs.push_back({kind, a, 0});
    }
    (void)n;
    return gs;
}
Pauli random_pauli(std::mt19937_64& R, int n) {
    Pauli P(n);
    for (int q = 0; q < n; ++q) { if (R() & 1) P.setx(q); if (R() & 1) P.setz(q); }
    return P;
}
Pauli hermitian_of(Pauli P) { P.phase = P.xz_overlap() & 1; return P; }   // i^{x·z} X^x Z^z is Hermitian
bool is_id(const Pauli& P) { for (auto w : P.x) if (w) return false; for (auto w : P.z) if (w) return false; return true; }

// Register fidelity (strict contract): random small instances with an output cut O | R. The bare frame U = U_O ⊗ U_R
// (sometimes with cross gates), the free wires mostly on one side (else the contract fails), the records random
// commuting operators U·W·Z_q·W†·U† for q ∈ R and W a Clifford on R (on O too when U has cross gates: a record on the
// output, refused by name), a random C0 of S / CZ / rot(B) factors on all wires, a random prefix P, a random
// correction P_c on O and y from the retained state's own Born law (sometimes flipped). The INDEPENDENT dense oracle
// builds ψ_c = P_c Π_rec(y) P C0 |B⟩ and the partial traces ρ_O(B), ρ_O(ψ_c) on the statevector: an accepted instance
// must be pure (Tr ρ_O(B)² = 1, never a false pure), fixed (Π_rec(y)|B⟩ reduces to the same output state) and
// F = Tr[ρ_O(B) ρ_O(ψ_c)]; a refused instance with dense amplitudes must be impure (Theorem 1 complete for generic α).
struct Instance {
    int n = 0, k = 0, nO = 0;
    std::vector<int> outw;           // O
    std::vector<uint8_t> isout;
    std::vector<Gate> ugates;
    std::vector<uint8_t> eps;
    std::vector<int> free;
    std::vector<cd> alpha;           // over 2^k
    bool dense = false, cross = false;
    std::vector<Pauli> recs;
    std::vector<RecordFidelity::Factor> factors;
    std::vector<std::pair<Pauli, int>> rots;   // C0 = R_V ⋯ R_1 (for the pull-back)
};

long n_gen_cz = 0;                            // general (non-Z) CZ factors drawn (NOTE-A)
Instance random_instance(std::mt19937_64& R) {
    Instance I;
    I.n = 2 + (int)(R() % 6);                 // 2..7
    std::vector<int> all(I.n);
    for (int q = 0; q < I.n; ++q) all[q] = q;
    std::vector<int> perm = all;
    std::shuffle(perm.begin(), perm.end(), R);
    I.nO = 1 + (int)(R() % (I.n - 1));        // 1 .. n-1 output wires
    I.outw.assign(perm.begin(), perm.begin() + I.nO);
    I.isout.assign(I.n, 0);
    for (int q : I.outw) I.isout[q] = 1;
    std::vector<int> O, Rs;
    for (int q = 0; q < I.n; ++q) (I.isout[q] ? O : Rs).push_back(q);
    I.ugates = random_gates(R, I.n, 3 * (int)O.size(), O);
    for (const Gate& g : random_gates(R, I.n, 3 * (int)Rs.size(), Rs)) I.ugates.push_back(g);
    I.cross = (R() % 5) == 0;
    if (I.cross) I.ugates.push_back({2, O[R() % O.size()], Rs[R() % Rs.size()]});
    I.eps.assign(I.n, 0);
    for (int q = 0; q < I.n; ++q) I.eps[q] = R() & 1;
    // free wires: one side (strict) mostly, both sides sometimes
    const int mode = (int)(R() % 4);
    std::vector<int> pool = mode == 0 ? O : mode == 1 ? Rs : all;
    if (mode == 2) pool = O;
    std::shuffle(pool.begin(), pool.end(), R);
    I.k = (int)(R() % (std::min((int)pool.size(), 3) + 1));
    I.free.assign(pool.begin(), pool.begin() + I.k);
    I.alpha.assign((size_t)1 << I.k, 0.0);
    std::normal_distribution<double> nd;
    I.dense = (R() % 3) != 0;
    for (auto& a : I.alpha) if (I.dense || (R() & 1)) a = cd(nd(R), nd(R));
    if (std::abs(I.alpha[0]) == 0.0) I.alpha[0] = 1.0;
    // records on R: rows of (U·W), W a random Clifford on a subset of R
    CliffordTableau T2(I.n);
    std::vector<int> sub;
    for (int q : Rs) if (R() % 2) sub.push_back(q);
    for (const Gate& g : random_gates(R, I.n, 2 * (int)sub.size(), sub)) tab_gate(T2, g);
    for (const Gate& g : I.ugates) tab_gate(T2, g);
    for (int q : Rs)
        if (R() % 3) {
            Pauli M = T2.Zrow[q];
            if (R() & 1) M.phase = (M.phase + 2) & 3;
            I.recs.push_back(M);
        }
    // C0 factors on all wires
    const double r2 = 1.0 / std::sqrt(2.0);
    const int V = (int)(R() % 5);
    for (int f = 0; f < V; ++f) {
        const int kind = (int)(R() % 3);
        Pauli Id(I.n);
        if (kind == 0) {                       // S_q ∝ (1 − iZ)/√2
            Pauli Z(I.n); Z.setz((int)(R() % I.n));
            I.factors.push_back({{r2, Id}, {cd(0, -r2), Z}});
            I.rots.push_back({Z, -1});
        } else if (kind == 1 && I.n >= 2) {    // CZ = (1 + Zi + Zj − ZiZj)/2 — or, half the time, a GENERAL CZ factor
            const int a = (int)(R() % I.n);    // {I, A, B, −AB}/2 on commuting Hermitian A = W Z_a W†, B = W Z_b W†
            int b = a;                         // (W a random Clifford on all wires, random signs: X / Y / mixed types)
            while (b == a) b = (int)(R() % I.n);
            Pauli Zi(I.n), Zj(I.n);
            Zi.setz(a); Zj.setz(b);
            if (R() & 1) {
                CliffordTableau TW(I.n);
                for (const Gate& g : random_gates(R, I.n, 3 * I.n, all)) tab_gate(TW, g);
                Zi = TW.Zrow[a];
                Zj = TW.Zrow[b];
                if (R() & 1) Zi.phase = (Zi.phase + 2) & 3;
                if (R() & 1) Zj.phase = (Zj.phase + 2) & 3;
                ++n_gen_cz;
            }
            const Pauli Zij = Pauli::multiply(Zi, Zj);
            I.factors.push_back({{0.5, Id}, {0.5, Zi}, {0.5, Zj}, {-0.5, Zij}});
            I.rots.push_back({Zi, -1}); I.rots.push_back({Zj, -1}); I.rots.push_back({Zij, +1});
        } else {                               // rot(B) = (1 + iB)/√2, B a random Hermitian Pauli
            Pauli B = hermitian_of(random_pauli(R, I.n));
            if (is_id(B)) B.setz(0), B = hermitian_of(B);
            I.factors.push_back({{r2, Id}, {cd(0, r2), B}});
            I.rots.push_back({B, +1});
        }
    }
    return I;
}

// C0† Q C0, C0 = R_V ⋯ R_1, R = e^{i s π/4 A}: R† Q R = Q if [Q, A] = 0, else −i s A Q
Pauli pull(const Instance& I, Pauli Q) {
    for (int j = (int)I.rots.size() - 1; j >= 0; --j) {
        const Pauli& A = I.rots[(size_t)j].first;
        if (!Pauli::anticommute_bit(Q, A)) continue;
        Pauli AQ = Pauli::multiply(A, Q);
        AQ.phase = (AQ.phase + (I.rots[(size_t)j].second > 0 ? 3 : 1)) & 3;   // −i s
        Q = AQ;
    }
    return Q;
}

FramedSuperposition bare_of(const Instance& I) {
    FramedSuperposition B(I.n);
    for (const Gate& g : I.ugates) tab_gate(B.U, g);
    B.eps = I.eps;
    B.free = I.free;
    B.sync_alpha_k();
    for (size_t x = 0; x < I.alpha.size(); ++x)
        if (I.alpha[x] != 0.0) B.entries().push_back({(AmpKey)x, I.alpha[x]});
    return B;
}

Vec bare_vec(const Instance& I) {
    Vec v((size_t)1 << I.n, 0.0);
    for (size_t x = 0; x < I.alpha.size(); ++x) {
        size_t idx = 0;
        for (int q = 0; q < I.n; ++q) if (I.eps[q]) idx |= (size_t)1 << q;
        for (int d = 0; d < I.k; ++d) if ((x >> d) & 1) idx ^= (size_t)1 << I.free[d];
        v[idx] += I.alpha[x];
    }
    for (const Gate& g : I.ugates) sv_gate(v, g);
    return v;
}

Vec project(const Vec& v, const Pauli& M, int y) {   // (1 + (−1)^y M)/2
    Vec o(v.size(), 0.0);
    axpy(o, 0.5, v);
    axpy(o, y ? -0.5 : 0.5, apply_pauli(v, M));
    return o;
}

// ρ_O = Tr_R |v><v| / <v|v> (row-major 2^|O| x 2^|O|)
std::vector<cd> reduced(const Instance& I, const Vec& v) {
    const size_t dO = (size_t)1 << I.nO, dR = (size_t)1 << (I.n - I.nO);
    std::vector<int> Rw;
    for (int q = 0; q < I.n; ++q) if (!I.isout[q]) Rw.push_back(q);
    auto idx = [&](size_t o, size_t r) {
        size_t i = 0;
        for (int j = 0; j < I.nO; ++j) if ((o >> j) & 1) i |= (size_t)1 << I.outw[j];
        for (size_t j = 0; j < Rw.size(); ++j) if ((r >> j) & 1) i |= (size_t)1 << Rw[j];
        return i;
    };
    std::vector<cd> rho(dO * dO, 0.0);
    double nn = 0.0;
    for (size_t r = 0; r < dR; ++r)
        for (size_t a = 0; a < dO; ++a) {
            const cd va = v[idx(a, r)];
            nn += std::norm(va);
            if (va == 0.0) continue;
            for (size_t b = 0; b < dO; ++b) rho[a * dO + b] += va * std::conj(v[idx(b, r)]);
        }
    for (auto& z : rho) z /= nn;
    return rho;
}
double tr_prod(const std::vector<cd>& A, const std::vector<cd>& B, size_t d) {   // Re Tr(A B)
    cd s = 0.0;
    for (size_t i = 0; i < d; ++i) for (size_t j = 0; j < d; ++j) s += A[i * d + j] * B[j * d + i];
    return s.real();
}

long n_cls[4] = {0, 0, 0, 0};
long n_checked = 0, n_frac = 0, n_rot = 0, n_up = 0, n_fired = 0, n_corr = 0, n_impossible = 0, n_impossible_orth = 0, n_refused = 0,
     n_ref_impure = 0, n_rec_on_o = 0, n_3a = 0, n_3b = 0, n_sect = 0, n_facpull = 0;

void one_trial(std::mt19937_64& R) {
    const Instance I = random_instance(R);
    const FramedSuperposition B = bare_of(I);
    const Vec bv = bare_vec(I);
    const size_t dO = (size_t)1 << I.nO;
    const std::vector<cd> rhoB = reduced(I, bv);
    const double purity = tr_prod(rhoB, rhoB, dO);
    bool rec_on_o = false;
    for (const Pauli& M : I.recs)
        for (int q : I.outw) if (M.xbit(q) || M.zbit(q)) rec_on_o = true;
    // the engine's construction: refusal by name, or a route
    std::unique_ptr<RecordFidelity> rf0;
    try {
        rf0 = std::make_unique<RecordFidelity>(B, I.recs, I.outw, std::vector<int>{}, 0);
    } catch (const std::invalid_argument& e) {
        const std::string w = e.what();
        if (rec_on_o) { CHECK(w.find("RF_RECORD_ON_OUTPUT") != std::string::npos); ++n_rec_on_o; return; }
        CHECK(w.find("RF_OUTPUT_NOT_STRICT") != std::string::npos);
        ++n_refused;
        if (purity < 1 - 1e-9) ++n_ref_impure;
        else if (I.dense && !(purity < 1 - 1e-9)) {
            std::fprintf(stderr, "refused a pure output with generic amplitudes (n=%d k=%d |O|=%d)\n", I.n, I.k, I.nO);
            CHECK(false);
        }
        return;
    }
    CHECK(!rec_on_o);
    if (!(purity > 1 - 1e-9)) {                      // never a false pure
        std::fprintf(stderr, "FALSE PURE n=%d k=%d |O|=%d purity=%.12g\n", I.n, I.k, I.nO, purity);
        CHECK(false);
        return;
    }
    // the retained state before the records: P C0 |B>
    Vec c0b = bv;
    for (const auto& f : I.factors) {
        Vec t(c0b.size(), 0.0);
        for (const auto& term : f) axpy(t, term.first, apply_pauli(c0b, term.second));
        c0b = t;
    }
    const Pauli P = random_pauli(R, I.n);
    Pauli Pc(I.n);
    if (R() % 2) for (int q : I.outw) { if (R() & 1) Pc.setx(q); if (R() & 1) Pc.setz(q); }
    const Vec pc = apply_pauli(c0b, P);
    std::vector<int> y(I.recs.size(), 0);
    Vec psi = pc;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    for (size_t c = 0; c < I.recs.size(); ++c) {
        const Vec p0 = project(psi, I.recs[c], 0);
        const double q0 = inner(p0, p0).real() / inner(psi, psi).real();
        y[c] = U01(R) < q0 ? 0 : 1;
        psi = y[c] ? project(psi, I.recs[c], 1) : p0;
    }
    if (!I.recs.empty() && R() % 5 == 0) y[R() % I.recs.size()] ^= 1;
    psi = pc;
    for (size_t c = 0; c < I.recs.size(); ++c) psi = project(psi, I.recs[c], y[c]);
    const double pn = inner(psi, psi).real();
    bool fired = false;                              // an R_s record off its sign on |B> (a fired detector)
    const double bn = inner(bv, bv).real();
    for (size_t c = 0; c < I.recs.size(); ++c) {
        const cd e = inner(bv, apply_pauli(bv, I.recs[c])) / bn;
        if (std::abs(std::abs(e.real()) - 1.0) < 1e-9 && std::abs(e.imag()) < 1e-9)
            if ((e.real() < 0) != (y[c] == 1)) fired = true;
    }
    // fixedness: the outcome-conditioned ideal reduces to the same output state
    {
        Vec Rv = bv;
        for (size_t c = 0; c < I.recs.size(); ++c) Rv = project(Rv, I.recs[c], y[c]);
        if (inner(Rv, Rv).real() > 1e-9 * bn) CHECK(std::abs(tr_prod(rhoB, reduced(I, Rv), dO) - 1.0) < 1e-9);
    }
    const int W = (I.n + 63) / 64;
    std::vector<uint64_t> px(W, 0), pz(W, 0), qx(W, 0), qz(W, 0), yw((I.recs.size() + 63) / 64 + 1, 0);
    for (int q = 0; q < I.n; ++q) {
        if (P.xbit(q)) px[q >> 6] |= 1ULL << (q & 63);
        if (P.zbit(q)) pz[q >> 6] |= 1ULL << (q & 63);
        if (Pc.xbit(q)) qx[q >> 6] |= 1ULL << (q & 63);
        if (Pc.zbit(q)) qz[q >> 6] |= 1ULL << (q & 63);
    }
    for (size_t c = 0; c < y.size(); ++c) if (y[c]) yw[c >> 6] |= 1ULL << (c & 63);
    double Fo = -1.0;
    if (pn > 1e-18) Fo = tr_prod(rhoB, reduced(I, apply_pauli(psi, Pc)), dO);
    // the register fidelity (the twirled Pauli-expansion route, its fibers by the exact Gauss sum) == the statevector;
    // the same with the fibers by the exact enumeration oracle, byte for byte; the content padded with cancelling
    // pairs S·S† (fresh factors on every wire: the expansion reduces them away), byte for byte; INFEASIBLE rows are 0.
    // Variant 3 (route 3b only, fid-orth fix round): `pull` deliberately WRONG (the identity) — 3b's record pull-backs
    // come from the factor list itself (fac_pull), so F is unchanged byte for byte and still the statevector's (the
    // independent `pull` above is fac_pull's oracle, general non-commuting factors and CZ included)
    double Fg = 0.0;
    for (int variant = 0; variant < 4; ++variant) {
        RecordFidelity rf(B, I.recs, I.outw, std::vector<int>{}, 0);
        if (variant == 3 && rf.route() != 2) continue;
        rf.set_row_log_for_test(true);
        if (variant == 1) rf.set_hform_oracle_for_test(true);
        std::vector<RecordFidelity::Factor> fac = I.factors;
        if (variant == 2) {
            const double r2 = 1.0 / std::sqrt(2.0);
            for (int q = 0; q < I.n; ++q) {
                Pauli Z(I.n);
                Z.setz(q);
                fac.push_back({{r2, Pauli(I.n)}, {cd(0, -r2), Z}});
                fac.push_back({{r2, Pauli(I.n)}, {cd(0, r2), Z}});
            }
        }
        const int id = variant == 3 ? rf.intern_content("c", [](const Pauli& Q) { return Q; }, fac)
                                    : rf.intern_content("c", [&](const Pauli& Q) { return pull(I, Q); }, fac);
        if (variant == 3 && !I.factors.empty() && !I.recs.empty()) ++n_facpull;
        double F = -1.0;
        bool threw = false;
        try {
            F = rf.evaluate_record(id, px.data(), pz.data(), yw.data(), qx.data(), qz.data());
        } catch (const std::runtime_error& e) {
            threw = true;
            if (!(pn < 1e-18 && std::string(e.what()).find("impossible record") != std::string::npos)) {
                std::fprintf(stderr, "unexpected throw: %s (pn=%g)\n", e.what(), pn);
                CHECK(false);
            }
        }
        if (pn < 1e-18) {
            // an impossible record is refused by name on a consistent row; an ORTHOGONAL row (route 3b, the constraint
            // test fails) scores 0 without the record checks (OD-2026-10-24-fid-orth, D1(b): the sampler's rows carry
            // no impossible records)
            if (!threw) {
                const std::vector<uint8_t> lg0 = rf.take_row_log_for_test();
                CHECK(rf.route() == 2 && lg0.size() == 1 && !lg0.empty() && lg0[0] == 0 && F == 0.0);
                if (variant == 0) ++n_impossible_orth;
            } else if (variant == 0) {
                ++n_impossible;
            }
            continue;
        }
        const std::vector<uint8_t> lg = rf.take_row_log_for_test();
        CHECK(lg.size() == 1);
        const int cls = lg.empty() ? -1 : lg[0];
        if (cls == 255) { CHECK(variant == 1); continue; }    // a fiber beyond the oracle (dimension > 12)
        if (!(std::abs(F - Fo) <= 1e-10)) {
            std::fprintf(stderr, "MISMATCH n=%d k=%d |O|=%d V=%zu recs=%zu route=%d variant=%d cls=%d F=%.17g oracle=%.17g\n",
                         I.n, I.k, I.nO, I.factors.size(), I.recs.size(), rf.route(), variant, cls, F, Fo);
            CHECK(false);
        }
        if (cls == 0) CHECK(F == 0.0);
        if (variant) { CHECK(std::memcmp(&Fg, &F, sizeof(double)) == 0); continue; }
        Fg = F;
        ++n_cls[cls];
        ++n_checked;
        if (Fo > 1e-6 && Fo < 1 - 1e-6) ++n_frac;
        if (!I.factors.empty()) ++n_rot;
        if (fired) ++n_fired;
        if (!is_id(Pc)) ++n_corr;
        (rf.route() == 1 ? n_3a : n_3b) += 1;
        if (rf.stats().max_sectors > 1) ++n_sect;
        if (rf.stats().max_kappa_prime > 0) ++n_up;
    }
}

void test_random_vs_statevector() {
    std::mt19937_64 R(20261022);
    for (int t = 0; t < 6000; ++t) one_trial(R);
    std::fprintf(stderr, "checked %ld (fractional %ld, with C0 %ld, kappa'>0 %ld, fired R_s %ld, corrected %ld, 3a %ld, 3b %ld, "
                 "3b with >1 sector %ld), impossible records: refused %ld, on orthogonal rows scored 0 %ld; refused %ld (impure %ld), record on output %ld\n", n_checked,
                 n_frac, n_rot, n_up, n_fired, n_corr, n_3a, n_3b, n_sect, n_impossible, n_impossible_orth, n_refused, n_ref_impure,
                 n_rec_on_o);
    std::fprintf(stderr, "route 3b with C0 and records, pull = identity (fac_pull serves): %ld; general (non-Z) CZ factors drawn: %ld\n",
                 n_facpull, n_gen_cz);
    std::fprintf(stderr, "classes: INFEASIBLE %ld BORN-ZERO %ld GAUSS-ZERO %ld VALUE %ld\n", n_cls[0], n_cls[1], n_cls[2],
                 n_cls[3]);
    CHECK(n_cls[0] > 50 && n_cls[3] > 500);
    CHECK(n_checked > 1500 && n_frac > 300 && n_rot > 500 && n_up > 100 && n_fired > 50 && n_corr > 300 && n_3a > 300 &&
          n_3b > 300 && n_sect > 50 && n_impossible > 5 && n_facpull > 300 && n_gen_cz > 300 && n_refused > 300 && n_ref_impure > 200 && n_rec_on_o > 20);
}

// Owner counterexample (review B1): |T>-ish output entangled with a measured read wire (no feedback) — refused by name,
// naming the measurement; with the frame closed (the read wire's flip fed back), served.
void test_b1_counterexample() {
    // H 0; T 0; CX 0 1; M 1 with wire 1 the measurement's read wire: B = (|00> + e^{iπ/4}|11>)/√2, record Z_1
    Instance I;
    I.n = 2; I.nO = 1; I.outw = {0}; I.isout = {1, 0}; I.k = 1; I.free = {0};
    I.eps = {0, 0};
    I.ugates = {{0, 0, 0}, {2, 0, 1}};               // the frame H 0, CX 0 1 on the free wire 0
    I.alpha = {1.0, std::polar(1.0, M_PI / 4)};
    const FramedSuperposition B = bare_of(I);
    Pauli M(2); M.setz(1);
    bool named = false;
    try {
        RecordFidelity rf(B, {M}, {0}, {-1, 0}, 0);
    } catch (const std::invalid_argument& e) {
        const std::string w = e.what();
        named = w.find("RF_OUTPUT_NOT_STRICT") != std::string::npos && w.find("c = 1") != std::string::npos &&
                w.find("measurements [0]") != std::string::npos;
        if (!named) std::fprintf(stderr, "refusal text: %s\n", e.what());
    }
    CHECK(named);
}

void test_correction_off_output_refused() {
    std::mt19937_64 R(3);
    for (int t = 0; t < 200; ++t) {
        const Instance I = random_instance(R);
        if (I.cross || I.k == 0) continue;
        const FramedSuperposition B = bare_of(I);
        std::unique_ptr<RecordFidelity> rf;
        try { rf = std::make_unique<RecordFidelity>(B, I.recs, I.outw, std::vector<int>{}, 0); }
        catch (const std::invalid_argument&) { continue; }
        const int id = rf->intern_content("c", [&](const Pauli& Q) { return pull(I, Q); }, I.factors);
        std::vector<uint64_t> z(1, 0), q(1, 0), yw(2, 0);
        for (int w = 0; w < I.n; ++w) if (!I.isout[w]) { q[0] |= 1ULL << w; break; }
        bool named = false;
        try { rf->evaluate_record(id, z.data(), z.data(), yw.data(), q.data(), z.data()); }
        catch (const std::invalid_argument& e) { named = std::string(e.what()).find("RF_CORRECTION_OFF_OUTPUT") != std::string::npos; }
        CHECK(named);
        return;
    }
    CHECK(false);
}

void test_no_output_refused() {
    std::mt19937_64 R(5);
    const Instance I = random_instance(R);
    const FramedSuperposition B = bare_of(I);
    bool named = false;
    try { RecordFidelity rf(B, I.recs, std::vector<int>{}, std::vector<int>{}, 0); }
    catch (const std::invalid_argument& e) { named = std::string(e.what()).find("RF_NO_OUTPUT") != std::string::npos; }
    CHECK(named);
}

void test_non_clifford_factor_refused() {
    std::mt19937_64 R(7);
    Instance I = random_instance(R);
    while (I.cross || I.k > 0) I = random_instance(R);   // a strict instance (k = 0: any cut)
    const FramedSuperposition B = bare_of(I);
    Pauli Id(I.n), Z(I.n), X(I.n);
    Z.setz(0); X.setx(0);
    const double c = std::cos(0.3), s = std::sin(0.3);
    const std::vector<std::vector<RecordFidelity::Factor>> bad = {
        {{{c, Id}, {cd(0, s), Z}}},                          // e^{i 0.3 Z}: not a π/4 rotation
        {{{0.5, Id}, {0.5, X}}},                             // (1 + X)/2: not unitary
        {{{0.5, Id}, {0.5, Z}, {0.5, X}, {-0.5, Pauli::multiply(Z, X)}}},   // "CZ" of anticommuting operators
        {{{1.0, Z}}},                                        // a support-only factor
    };
    for (const auto& f : bad) {
        RecordFidelity rf(B, I.recs, I.outw, std::vector<int>{}, 0);
        bool named = false;
        try {
            rf.intern_content("bad", [](const Pauli& Q) { return Q; }, f);
        } catch (const std::invalid_argument& e) {
            named = std::string(e.what()).find("RF_FACTOR_NOT_CLIFFORD") != std::string::npos;
            if (!named) std::fprintf(stderr, "refusal text: %s\n", e.what());
        }
        CHECK(named);
    }
}

}  // namespace

// S8 RF1: λ(u, v) = ⟨b|Z^v X^u|b⟩ through the memo (dense, and sparse with the table budget 0) == the formula, byte
// for byte; == the statevector (−1)^{v·eps} ⟨B| U Z^v X^u U† |B⟩ (U Z_a U† = Zrow[a], U X_a U† = Xrow[a]) to 1e-12; a pair-free
// (u, v) — no y with b_y, b_{y⊕u} both nonzero — is exactly +0.0 (sparse-support instances).
long n_lam = 0, n_lam_zero = 0;
void test_lambda() {
    std::mt19937_64 R(77);
    for (int trial = 0; trial < 400; ++trial) {
        Instance I = random_instance(R);
        if (I.k == 0) continue;
        const FramedSuperposition B = bare_of(I);
        const Vec bv = bare_vec(I);
        std::unique_ptr<RecordFidelity> rd, rs;
        try {
            rd = std::make_unique<RecordFidelity>(B, I.recs, I.outw, std::vector<int>{}, 0);
            rs = std::make_unique<RecordFidelity>(B, I.recs, I.outw, std::vector<int>{}, 0);
        } catch (const std::invalid_argument&) { continue; }
        rs->set_lambda_budget_for_test(0);
        for (uint64_t u = 0; u < (1ULL << I.k); ++u)
            for (uint64_t v = 0; v < (1ULL << I.k); ++v) {
                const cd d = rd->lambda_for_test(u, v, true), m1 = rd->lambda_for_test(u, v, false),
                         m2 = rs->lambda_for_test(u, v, false);
                CHECK(std::memcmp(&d, &m1, sizeof(cd)) == 0 && std::memcmp(&d, &m2, sizeof(cd)) == 0);
                Pauli P(I.n);
                for (int t = 0; t < I.k; ++t) if ((v >> t) & 1) P = Pauli::multiply(P, B.U.Zrow[(size_t)I.free[(size_t)t]]);
                for (int t = 0; t < I.k; ++t) if ((u >> t) & 1) P = Pauli::multiply(P, B.U.Xrow[(size_t)I.free[(size_t)t]]);
                int es = 0;                          // the free register is |eps ⊕ x>: Z_a there carries (−1)^{eps_a}
                for (int t = 0; t < I.k; ++t) if ((v >> t) & 1) es ^= I.eps[(size_t)I.free[(size_t)t]];
                const cd o = inner(bv, apply_pauli(bv, P)) * (es ? -1.0 : 1.0);
                CHECK(std::abs(o - d) <= 1e-12);
                bool pair = false;
                for (size_t y = 0; y < I.alpha.size() && !pair; ++y)
                    if (I.alpha[y] != 0.0 && I.alpha[y ^ (size_t)u] != 0.0) pair = true;
                if (!pair) {
                    const cd z = 0.0;
                    CHECK(std::memcmp(&d, &z, sizeof(cd)) == 0);
                    ++n_lam_zero;
                }
                ++n_lam;
            }
        CHECK(rd->lambda_mode_for_test() == 1 && rs->lambda_mode_for_test() == 2);
    }
    std::printf("lambda: %ld entries (%ld structural zeros)\n", n_lam, n_lam_zero);
    CHECK(n_lam > 1000 && n_lam_zero > 50);
}

// S8 RF3: the named refusals of the new route. RF_SECTOR_RANK at the boundary on BOTH routes: a |T> output beside m
// rest wires in |+> and C0 = Π_q S_q (output and rest) reaches m independent rest sectors — m = 20 accepted (the new route ==
// production), 21 refused by name. RF_FIBER_RANK: S on the output and on 256 rest wires in |0> (κ' = 257 > 256) refused by
// the new route, 255 accepted (== production). RF_CONTENT_MALFORMED: a CZ pair on the diagonal.
FramedSuperposition rest_bare(int m, bool plus) {
    FramedSuperposition B(1 + m);
    if (plus) for (int q = 1; q <= m; ++q) B.U.left_h(q);
    B.eps.assign((size_t)(1 + m), 0);
    B.free = {0};
    B.sync_alpha_k();
    B.entries().push_back({(AmpKey)0, cd(1.0, 0.0)});
    B.entries().push_back({(AmpKey)1, std::polar(1.0, M_PI / 4)});
    return B;
}
void test_new_route_refusals() {
    for (int m : {20, 21}) {
        const FramedSuperposition B = rest_bare(m, true);
        RecordFidelity rf(B, {}, {0}, std::vector<int>{}, 0);
        std::vector<RecordFidelity::Factor> fac;
        const double r2 = 1.0 / std::sqrt(2.0);
        for (int q = 0; q <= m; ++q) {                           // S on the output and on every rest wire
            Pauli Zq(1 + m);
            Zq.setz(q);
            fac.push_back({{r2, Pauli(1 + m)}, {cd(0, -r2), Zq}});
        }
        const int id = rf.intern_content("s", [](const Pauli& Q) { return Q; }, fac);
        const std::vector<uint64_t> z((size_t)((1 + m + 63) / 64), 0), yw(1, 0);
        bool refused = false;
        double F = -1.0;
        try { F = rf.evaluate_record(id, z.data(), z.data(), yw.data()); }
        catch (const std::runtime_error& e) { refused = std::string(e.what()).find("RF_SECTOR_RANK") != std::string::npos; }
        std::fprintf(stderr, "sector boundary m=%d refused=%d F=%.17g max_sectors=%ld\n", m, (int)refused, F, rf.stats().max_sectors);
        if (m == 21) { CHECK(refused); continue; }
        CHECK(!refused && std::abs(F - 0.5) <= 1e-12 && rf.stats().max_sectors == (1L << 20));
    }
    for (int m : {255, 256}) {                                   // κ' = m + 1 (the output's S is a logical direction)
        const FramedSuperposition B = rest_bare(m, false);
        RecordFidelity rf(B, {}, {0}, std::vector<int>{}, 0);
        std::vector<RecordFidelity::Factor> fac;
        const double r2 = 1.0 / std::sqrt(2.0);
        for (int q = 1; q <= m; ++q) {
            Pauli Zq(1 + m);
            Zq.setz(q);
            fac.push_back({{r2, Pauli(1 + m)}, {cd(0, -r2), Zq}});
        }
        Pauli Z0(1 + m);
        Z0.setz(0);
        fac.push_back({{r2, Pauli(1 + m)}, {cd(0, -r2), Z0}});      // and S on the output: a nontrivial value
        const int id = rf.intern_content("s", [](const Pauli& Q) { return Q; }, fac);
        const std::vector<uint64_t> z((size_t)((1 + m + 63) / 64), 0), yw(1, 0);
        bool refused = false;
        double F = -1.0;
        try { F = rf.evaluate_record(id, z.data(), z.data(), yw.data()); }
        catch (const std::runtime_error& e) { refused = std::string(e.what()).find("RF_FIBER_RANK") != std::string::npos; }
        std::fprintf(stderr, "fiber boundary m=%d refused=%d F=%.17g\n", m, (int)refused, F);
        if (m == 256) { CHECK(refused); continue; }
        CHECK(!refused && std::abs(F - 0.5) <= 1e-12);
    }
    {
        const FramedSuperposition B = rest_bare(3, true);
        RecordFidelity rf(B, {}, {0}, std::vector<int>{}, 0);
        DiagNormalForm nf;
        nf.prefix = Pauli(4);
        nf.a.assign(4, 0);
        nf.cz = {{2, 2}};
        bool refused = false;
        try { rf.intern_content_diag("bad", nf); }
        catch (const std::invalid_argument& e) { refused = std::string(e.what()).find("RF_CONTENT_MALFORMED") != std::string::npos; }
        CHECK(refused);
    }
}

// fid-orth (review MINOR-1, declared): the content-level caps RF_SECTOR_RANK / RF_LOGICAL_RANK are stage-2 cost caps,
// so a row the consistency test finds ORTHOGONAL scores 0 exactly without them; a consistent row of the same content
// is still refused. |T> output wire 0, |0> output wire 1, m = 21 |+> rest wires; C0 = S on wire 0 and every rest wire
// (2^21 sectors > the cap). P' = X_1 is orthogonal (|0> -> |1> on an output wire): F = 0; P' = I: RF_SECTOR_RANK.
void test_sector_cap_orthogonal_row() {
    const int m = 21, n = 2 + m;
    FramedSuperposition B(n);
    for (int q = 2; q < n; ++q) B.U.left_h(q);
    B.eps.assign((size_t)n, 0);
    B.free = {0};
    B.sync_alpha_k();
    B.entries().push_back({(AmpKey)0, cd(1.0, 0.0)});
    B.entries().push_back({(AmpKey)1, std::polar(1.0, M_PI / 4)});
    RecordFidelity rf(B, {}, {0, 1}, std::vector<int>{}, 0);
    std::vector<RecordFidelity::Factor> fac;
    const double r2 = 1.0 / std::sqrt(2.0);
    for (int q = 0; q < n; ++q) {
        if (q == 1) continue;
        Pauli Zq(n);
        Zq.setz(q);
        fac.push_back({{r2, Pauli(n)}, {cd(0, -r2), Zq}});
    }
    const int id = rf.intern_content("s", [](const Pauli& Q) { return Q; }, fac);
    CHECK(rf.route() == 2);
    const std::vector<uint64_t> z(1, 0), yw(1, 0), x1(1, 2);
    double F = -1.0;
    bool refused = false;
    try { F = rf.evaluate_record(id, x1.data(), z.data(), yw.data()); }
    catch (const std::runtime_error&) { refused = true; }
    CHECK(!refused && F == 0.0 && rf.stats().orthogonal_rows == 1 && rf.stats().value_builds == 0);
    refused = false;
    try { rf.evaluate_record(id, z.data(), z.data(), yw.data()); }
    catch (const std::runtime_error& e) { refused = std::string(e.what()).find("RF_SECTOR_RANK") != std::string::npos; }
    CHECK(refused && rf.stats().value_builds == 0);
    F = -1.0;                                                    // the content's stage 1 survives: orthogonal again
    try { F = rf.evaluate_record(id, x1.data(), z.data(), yw.data()); }
    catch (const std::runtime_error&) {}
    CHECK(F == 0.0 && rf.stats().orthogonal_rows == 2);
    std::fprintf(stderr, "sector cap on an orthogonal row: F=0, consistent row refused=%d\n", (int)refused);
}

// S8 RF4: a two-word fiber (κ' >= 70): a |T> output beside m = 100 rest wires in |0>, C0 = S^a CZ^M with a CZ chain and
// random extra pairs over output and rest (kind 0, the closed form: r ~ 100 > 64 variables, β across words; the rest's
// Z directions are all in the fiber) == the same C0 as a factor list (its direct expansion reduced to the support).
void test_two_word_fiber() {
    std::mt19937_64 R(4040);
    long served = 0;
    for (int trial = 0; trial < 6; ++trial) {
        const int m = 100, n = 1 + m;
        const FramedSuperposition B = rest_bare(m, false);
        RecordFidelity rf(B, {}, {0}, std::vector<int>{}, 0);
        DiagNormalForm nf;
        nf.prefix = Pauli(n);
        nf.a.assign((size_t)n, 0);
        for (int q = 0; q < n; ++q) nf.a[(size_t)q] = R() & 1;
        for (int q = 0; q + 1 < n; ++q) nf.cz.push_back({q, q + 1});
        for (int e = 0; e < 60; ++e) {
            const int i = (int)(R() % n), j = (int)(R() % n);
            if (i != j) nf.cz.push_back({std::min(i, j), std::max(i, j)});
        }
        const int id = rf.intern_content_diag("d", nf);
        const std::vector<uint64_t> z((size_t)((n + 63) / 64), 0), yw(1, 0);
        const double F = rf.evaluate_record(id, z.data(), z.data(), yw.data());
        CHECK(rf.stats().max_kappa_prime >= 70 && rf.stats().max_r > 64);
        // the same C0 as a factor list (the direct expansion reduced by the Gauss sum: an independent path)
        RecordFidelity rg(B, {}, {0}, std::vector<int>{}, 0);
        std::vector<RecordFidelity::Factor> fac;
        const double r2 = 1.0 / std::sqrt(2.0);
        for (int q = 0; q < n; ++q)
            if (nf.a[(size_t)q] & 1) { Pauli Zq(n); Zq.setz(q); fac.push_back({{r2, Pauli(n)}, {cd(0, -r2), Zq}}); }
        for (const auto& pr : nf.cz) {
            Pauli Zi(n), Zj(n);
            Zi.setz(pr.first);
            Zj.setz(pr.second);
            fac.push_back({{0.5, Pauli(n)}, {0.5, Zi}, {0.5, Zj}, {-0.5, Pauli::multiply(Zi, Zj)}});
        }
        const int ig = rg.intern_content("f", [](const Pauli& Q) { return Q; }, fac);
        const double G = rg.evaluate_record(ig, z.data(), z.data(), yw.data());
        CHECK(std::abs(G - F) <= 1e-12 * std::max(1e-300, std::abs(F)) && F > 0.0);
        ++served;
    }
    CHECK(served == 6);
}

int main() {
    RUN(test_random_vs_statevector);
    RUN(test_b1_counterexample);
    RUN(test_correction_off_output_refused);
    RUN(test_no_output_refused);
    RUN(test_non_clifford_factor_refused);
    RUN(test_lambda);
    RUN(test_new_route_refusals);
    RUN(test_two_word_fiber);
    RUN(test_sector_cap_orthogonal_row);
    REPORT();
}
