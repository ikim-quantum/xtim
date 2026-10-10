// test_exact_engine_audit.cpp — exact oracles for the 2026-10-09 sampling-audit findings in the
// exact engine (fix/audit-exact).
//
//   1. PprResidual::clifford_tableau / to_error_tableau vs a DENSE matrix oracle. A stored axis
//      A = i^{phase} X^x Z^z (canonical phase ∈ {0,1}) equals −(letter string) whenever its Y count
//      is ≡ 2,3 (mod 4) (phase 0 on X0Z0X1Z1 is −Y0Y1). The diagonalizer W maps the LETTER string
//      to +Z, so the exponent must be negated for those axes; 3.1.9 dropped the sign and applied
//      an extra A factor (findings B and H: every read anticommuting with A flipped on a fired
//      off-diagonal noise atom). Checked: every axis on n = 3 (all x/z patterns), every exponent
//      1..7, rows U·P·U† against exp(iπ/4·e·A) P exp(−iπ/4·e·A) to 1e-12, and the full error
//      tableau with a Pauli part.
//   2. framed_measure_anticommuting (the batch_measure / TreePlan case-B collapse) vs the
//      amplitude-container measure_pauli on the SAME state with the SAME forced outcome: the two
//      post-states must agree on every Pauli expectation over the circuit's wires (≤ 8 wires; else
//      every product of the terminal read Paulis) — exact, every outcome string. 3.1.9
//      re-referenced the frame to survivor 0 but kept the branch labels relative to the projected
//      σ=0 state — a relative Pauli frame error whenever survivor 0 was not the σ=0 branch
//      (finding G, and the exact-expectation sign finding D). Circuits: the audit's G and D (labels)
//      plus two seeded multi-magic circuits whose branch sign parity z_g·x0 varies (the sign half).
#include "check.hpp"

#include <cmath>
#include <complex>
#include <functional>
#include <string>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/ppr_residual.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;
using cd = std::complex<double>;
using Mat = std::vector<cd>;   // row-major d×d

namespace {

// ── dense helpers (qubit 0 = least-significant bit of the basis index) ────────────────────────
Mat matmul(const Mat& A, const Mat& B, int d) {
    Mat C((size_t)d * d, 0.0);
    for (int i = 0; i < d; ++i)
        for (int k = 0; k < d; ++k) {
            const cd a = A[(size_t)i * d + k];
            if (a == 0.0) continue;
            for (int j = 0; j < d; ++j) C[(size_t)i * d + j] += a * B[(size_t)k * d + j];
        }
    return C;
}
Mat dagger(const Mat& A, int d) {
    Mat B((size_t)d * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) B[(size_t)j * d + i] = std::conj(A[(size_t)i * d + j]);
    return B;
}
// Dense i^{phase} X^x Z^z (Z applied first, then X — the engine's operator convention).
Mat dense_pauli(const Pauli& P) {
    const int n = P.n, d = 1 << n;
    static const cd ip[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    Mat M((size_t)d * d, 0.0);
    int xm = 0, zm = 0;
    for (int q = 0; q < n; ++q) {
        if (P.xbit(q)) xm |= 1 << q;
        if (P.zbit(q)) zm |= 1 << q;
    }
    for (int col = 0; col < d; ++col) {
        const int row = col ^ xm;
        const double s = (__builtin_popcount(col & zm) & 1) ? -1.0 : 1.0;
        M[(size_t)row * d + col] = ip[P.phase & 3] * s;
    }
    return M;
}
double maxdiff(const Mat& A, const Mat& B) {
    double m = 0;
    for (size_t i = 0; i < A.size(); ++i) m = std::max(m, std::abs(A[i] - B[i]));
    return m;
}
// exp(i·(π/4)·e·A) for a Hermitian Pauli A (A² = I): cos θ·I + i sin θ·A.
Mat rot(const Pauli& A, int e) {
    const int d = 1 << A.n;
    const double th = M_PI / 4 * e;
    Mat R = dense_pauli(A);
    for (auto& v : R) v *= cd(0, std::sin(th));
    for (int i = 0; i < d; ++i) R[(size_t)i * d + i] += std::cos(th);
    return R;
}
// The tableau's forward rows must equal V·P·V† for P = X_q, Z_q.
double tableau_vs_dense(const CliffordTableau& T, const Mat& V, int n) {
    const int d = 1 << n;
    const Mat Vd = dagger(V, d);
    double worst = 0;
    for (int q = 0; q < n; ++q)
        for (int xz = 0; xz < 2; ++xz) {
            Pauli P(n);
            if (xz) P.setz(q); else P.setx(q);
            const Mat want = matmul(matmul(V, dense_pauli(P), d), Vd, d);
            const Mat got = dense_pauli(xz ? T.Zrow[(size_t)q] : T.Xrow[(size_t)q]);
            worst = std::max(worst, maxdiff(want, got));
        }
    return worst;
}

// ── 1. PprResidual tableau vs dense ─────────────────────────────────────────────────────────────
void test_ppr_clifford_tableau_every_axis() {
    const int n = 3, d = 1 << n;
    int checked = 0, neg_axes = 0;
    for (int code = 1; code < (1 << (2 * n)); ++code) {      // every non-identity x/z pattern
        Pauli A(n);
        for (int q = 0; q < n; ++q) {
            if ((code >> q) & 1) A.setx(q);
            if ((code >> (n + q)) & 1) A.setz(q);
        }
        A.phase = A.xz_overlap() & 1;                          // the stored canonical phase ∈ {0,1}
        if (((A.phase - A.xz_overlap()) & 3) == 2) ++neg_axes; // = −(letter string)
        for (int e = 1; e <= 7; ++e) {
            PprResidual R;
            R.n = n;
            R.pauli = Pauli(n);
            R.rots.push_back({A, e});
            const double err = tableau_vs_dense(R.clifford_tableau(), rot(A, e), n);
            if (err > 1e-12)
                std::fprintf(stderr, "  axis code %d (#Y=%d) e=%d: max row error %.3g\n", code,
                             A.xz_overlap(), e, err);
            CHECK(err < 1e-12);
            ++checked;
        }
    }
    CHECK(neg_axes > 0);   // the sign-carrying axes (#Y ≡ 2,3 mod 4) are in the sweep
    CHECK(checked == 63 * 7);
    (void)d;
}

void test_ppr_error_tableau_two_axes_with_pauli() {
    // E = pauli · exp(iπ/4 e1 A1) · exp(iπ/4 e2 A2) with commuting A1 (two Y's: the finding-B axis
    // −Y0Y1 is phase-0 X0Z0X1Z1) and A2 = Y1, Pauli part Y0X1 (finding B's X-atom residual).
    const int n = 2, d = 4;
    Pauli A1(n); A1.setx(0); A1.setz(0); A1.setx(1); A1.setz(1); A1.phase = 0;
    Pauli A2(n); A2.setx(1); A2.setz(1); A2.phase = 1;
    Pauli Pp(n); Pp.setx(0); Pp.setz(0); Pp.setx(1); Pp.phase = 1;   // i·X0Z0·X1 = Y0X1
    for (int e1 = 1; e1 <= 7; ++e1)
        for (int e2 = 1; e2 <= 7; ++e2) {
            PprResidual R;
            R.n = n;
            R.pauli = Pp;
            R.rots.push_back({A1, e1});
            R.rots.push_back({A2, e2});
            const Mat V = matmul(dense_pauli(Pp), matmul(rot(A1, e1), rot(A2, e2), d), d);
            CHECK(tableau_vs_dense(R.to_error_tableau(), V, n) < 1e-12);
        }
}

// ── 2. case-B collapse vs measure_pauli ─────────────────────────────────────────────────────────
struct Prepared {
    FramedSuperposition bare{0};
    std::vector<std::pair<int, int>> reads;   // (0:X 1:Y 2:Z, wire)
    std::vector<int> wires;                   // wires any gate/read touches
};

Prepared prepare(const std::string& text) {
    Prepared out;
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    NormalizePolicy np;
    np.feedback = NormalizePolicy::Feedback::KeepCoherent;
    np.defer = true;
    NormalizeResult nr = normalize(ps.circuit, np);
    std::vector<int> touched(nr.normalized.n, 0);
    for (const Instr& i : nr.normalized.stream) {
        if (i.kind == Instr::Kind::Measure)
            for (int q : i.qubits) {
                out.reads.push_back({i.basis == PauliBasis::X ? 0 : i.basis == PauliBasis::Y ? 1 : 2, q});
                touched[q] = 1;
            }
        if (i.kind == Instr::Kind::Gate)
            for (int q : i.targets) touched[q] = 1;
    }
    for (int q = 0; q < nr.normalized.n; ++q) if (touched[q]) out.wires.push_back(q);
    FramedBareState b = build_bare_state_framed(nr.normalized);
    CHECK(!b.rejected);
    out.bare = std::move(b.state);
    out.bare.U.ensure_dual();
    return out;
}

// Max |<P>_a − <P>_b| over every Pauli on the circuit's wires (4^|wires| − 1 strings) when
// |wires| ≤ 8, else over every product of the terminal read Paulis (the observables that fix the
// record law).
double max_expectation_gap(const FramedSuperposition& a, const FramedSuperposition& b,
                           const Prepared& pr) {
    const int N = a.n();
    double worst = 0;
    auto gap = [&](Pauli& P) {
        P.phase = P.xz_overlap() & 3;   // Hermitian letter string
        worst = std::max(worst, std::abs(a.expectation(P) - b.expectation(P)));
    };
    const std::vector<int>& wires = pr.wires;
    if (wires.size() <= 8) {
        const int k = (int)wires.size();
        long tot = 1;
        for (int i = 0; i < k; ++i) tot *= 4;
        for (long code = 1; code < tot; ++code) {
            Pauli P(N);
            long c = code;
            for (int i = 0; i < k; ++i, c /= 4) {
                const int l = (int)(c % 4);
                if (l == 1 || l == 2) P.setx(wires[(size_t)i]);
                if (l == 2 || l == 3) P.setz(wires[(size_t)i]);
            }
            gap(P);
        }
    } else {
        const size_t r = pr.reads.size();
        for (unsigned long m = 1; m < (1ul << r); ++m) {
            Pauli P(N);
            for (size_t k = 0; k < r; ++k)
                if ((m >> k) & 1) {
                    const int p = pr.reads[k].first, q = pr.reads[k].second;
                    if (p == 0 || p == 1) P.flipx(q);
                    if (p == 1 || p == 2) P.flipz(q);
                }
            gap(P);
        }
    }
    return worst;
}

// Walk the circuit's terminal reads in batch_measure's pick order (first free-coupling read), with
// a prescribed outcome string. Case-B picks collapse through framed_measure_anticommuting on one
// copy and measure_pauli on the other; the two post-states must agree EXACTLY. Case-A picks (and
// the tail) use measure_pauli on both. Returns the number of case-B collapses compared.
int compare_collapses(const Prepared& pr, unsigned outcomes) {
    FramedSuperposition L = pr.bare, S = pr.bare;
    std::vector<int> done(pr.reads.size(), 0);
    int caseb = 0;
    const int N = L.n(), W = (N + 63) / 64;
    while (L.chi() > 1) {
        int pick = -1;
        for (size_t k = 0; k < pr.reads.size(); ++k)
            if (!done[k] && framed_read_couples_free(L, pr.reads[k].first, pr.reads[k].second)) {
                pick = (int)k;
                break;
            }
        if (pick < 0) break;
        const Pauli P = single_pauli(pr.reads[(size_t)pick].first, pr.reads[(size_t)pick].second, N);
        const int want = (outcomes >> pick) & 1;                 // 0: +1, 1: −1
        const double u = want ? 1.0 - 1e-15 : 0.0;
        // Skip an impossible branch (prescribed outcome has zero Born weight).
        const double p1 = S.born_p1(P);
        if ((want == 0 && p1 < 1e-12) || (want == 1 && p1 > 1.0 - 1e-12)) return caseb;
        const Pauli Qf = L.U.conjugate_single(pr.reads[(size_t)pick].first, pr.reads[(size_t)pick].second);
        bool caseB = false;
        for (int w = 0; w < W; ++w) if (Qf.x[w]) caseB = true;
        int mL;
        if (caseB) {
            std::vector<int> A;
            for (int a = 0; a < N; ++a) if (Qf.xbit(a)) A.push_back(a);
            mL = framed_measure_anticommuting(L, pr.reads[(size_t)pick].first,
                                              pr.reads[(size_t)pick].second, P, A, u);
            ++caseb;
        } else {
            mL = L.measure_pauli(P, u);
        }
        const int mS = S.measure_pauli(P, u);
        CHECK(mL == mS);
        done[(size_t)pick] = 1;
        if (!caseB) continue;   // case A runs measure_pauli on both copies
        const double gap = max_expectation_gap(L, S, pr);
        if (gap > 1e-9)
            std::fprintf(stderr, "  outcomes %x: read %d (case %s) post-state gap %.3g\n", outcomes,
                         pick, caseB ? "B" : "A", gap);
        CHECK(gap < 1e-9);
    }
    return caseb;
}

// Finding G (audit mins/G.stim): noiseless multi-magic; the case-B collapse of the X1 read after a
// −1 on MPP Z2*Z0 landed survivor 0 on a σ≠0 branch.
const char* kG =
    "RX 0\nRY 1\nH 2\nT 2\nMPP Z2*Z0\nRY 2\nCCZ 0 2 1\nCS 2 1\nMPP X1\nMPP X1\nMR 0\nMRY 1\n"
    "R 2\nMX 1\n";
// Finding D (audit mins/D.stim): the exact PAULI_EXPECTATION sign followed r2⊕r3 instead of r3.
const char* kD =
    "RX 0\nMPP Y0*Y3\nSWAP 1 3\nRY 5\nT 1\nMX 4\nRX 4\nCCZ 4 0 5\nMX 0\nMY 4\n";

void test_caseb_collapse_matches_measure_pauli(const char* text, int min_caseb) {
    Prepared pr = prepare(text);
    CHECK(pr.reads.size() <= 14);
    int total = 0;
    for (unsigned o = 0; o < (1u << pr.reads.size()); ++o) total += compare_collapses(pr, o);
    std::fprintf(stderr, "  %d case-B collapses compared\n", total);
    CHECK(total >= min_caseb);
}

// Seeded multi-magic circuits whose collapses land survivor 0 off the σ=0 branch with a
// branch-VARYING sign parity z_g·x0 (the sign half of the fix; G and D only exercise the labels;
// vary1 is the one that fails with the eps ^= x0 fold removed).
const char* kVary1 =
    "RX 0\nRX 1\nR 2\nRY 3\nRX 4\nCX 4 3\nMX 2\nCCZ 3 4 0\nM 2\nMPP X2*Y0\nCCZ 0 3 4\nCS 1 2\n"
    "MPP Z1*Y4\nMPP Y4*Y2\nCX 2 1\nCS_DAG 2 4\nSWAP 2 4\nCCZ 4 2 0\nCCZ 2 4 1\nRY 1\nRY 0\n"
    "MPP Y1*X4\nMPP X0*X3\n";
const char* kVary3 =
    "RY 0\nRX 1\nRX 2\nRY 3\nRY 4\nCCZ 4 1 2\nCCZ 2 0 3\nCX 4 2\nMPP Z3*Y2\nRX 1\nCCZ 1 3 4\n"
    "MPP Z1*X3\nCX 2 4\nCS_DAG 0 1\nM 4\nT 2\nMY 3\nCS_DAG 2 3\nMX 1\nCCZ 1 2 3\nT 1\nRY 4\n"
    "CCZ 0 4 2\n";

void test_caseb_G() { test_caseb_collapse_matches_measure_pauli(kG, 8); }
void test_caseb_vary1() { test_caseb_collapse_matches_measure_pauli(kVary1, 1); }
void test_caseb_vary3() { test_caseb_collapse_matches_measure_pauli(kVary3, 1); }
void test_caseb_D() { test_caseb_collapse_matches_measure_pauli(kD, 1); }

}  // namespace

int main() {
    RUN(test_ppr_clifford_tableau_every_axis);
    RUN(test_ppr_error_tableau_two_axes_with_pauli);
    RUN(test_caseb_G);
    RUN(test_caseb_D);
    RUN(test_caseb_vary1);
    RUN(test_caseb_vary3);
    REPORT();
}
