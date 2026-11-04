// test_restrict_to_support.cpp — byte-identity of the word-parallel restrict_to_support.
//
// The production restrict_to_support (parity_support.cpp) computes each Z-term's parity column
// m·R as an XOR of R-rows over the set bits of m. The pre-3.1.12 scalar double loop (bit j =
// XOR_q m_q R(q,j)) is kept here as the ORACLE: on random C'† gate lists and random Z-terms
// (sparse, dense, empty, single-qubit, constant-on-support, n and d straddling word boundaries),
// the columns, their order, the coeffs and the global phase must be EXACTLY the oracle's.
//
// Also: AffineState::apply_h skips canonicalize() when the old R-row q is zero. That is exact
// only under the invariant "R has full column rank after every public op"; the second test
// checks the invariant with an independent rank computation after every gate/projection, and
// that an extra canonicalize() after EVERY op (H, CX, other gates, projections incl. zero-
// probability branches) is a no-op (so the skip cannot drop work). The ctest build defines
// QECCORE_CHECK_INVARIANTS, which makes the skip path itself throw on a violation;
// invariant_check_fires() proves that mechanism is compiled in and live.
#include "check.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "qeccore/parity_support.hpp"
#include "src/stab_affine_internal.hpp"   // kernel_vec_for_test (internal seam)

#include <stdexcept>

using namespace qeccore;

namespace {

struct OracleOut {
    std::vector<std::vector<uint64_t>> columns;
    std::vector<int> coeffs;
    int gphase = 0;
};

// The pre-word-parallel scalar loop, verbatim in its arithmetic (oracle).
OracleOut scalar_oracle(const AffineState& psi, const std::vector<DiagResult::ZTerm>& tz, int n) {
    OracleOut o;
    const int d = psi.k_;
    const int dwords = (d + 63) / 64;
    int gphase = 0;
    for (const auto& zt : tz) {
        const Pauli& D = zt.zstring;
        std::vector<uint64_t> col(dwords, 0);
        bool any = false;
        for (int j = 0; j < d; ++j) {
            int bit = 0;
            for (int q = 0; q < n; ++q)
                if (D.zbit(q) && psi.R.get(q, j)) bit ^= 1;
            if (bit) { col[j >> 6] |= (1ull << (j & 63)); any = true; }
        }
        int cbit = 0;
        for (int q = 0; q < n; ++q)
            if (D.zbit(q) && psi.b[q]) cbit ^= 1;
        if (!any) {
            int contrib = cbit ? -zt.coeff : zt.coeff;
            gphase = ((gphase + contrib) % 16 + 16) % 16;
        } else {
            int c = cbit ? -zt.coeff : zt.coeff;
            c = ((c % 16) + 16) % 16;
            o.columns.push_back(std::move(col));
            o.coeffs.push_back(c);
        }
    }
    o.gphase = gphase;
    return o;
}

std::vector<CliffGate> random_gates(std::mt19937_64& rng, int n, int ngates, double hfrac) {
    std::vector<CliffGate> g;
    std::uniform_int_distribution<int> q(0, n - 1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < ngates; ++i) {
        const double r = u(rng);
        const int a = q(rng);
        if (r < hfrac) { g.push_back({GateKind::H, {a}}); continue; }
        const int kind = (int)(rng() % 8);
        int b = q(rng);
        if (n > 1) while (b == a) b = q(rng);
        switch (kind) {
            case 0: g.push_back({GateKind::S, {a}}); break;
            case 1: g.push_back({GateKind::SDG, {a}}); break;
            case 2: g.push_back({GateKind::X, {a}}); break;
            case 3: g.push_back({GateKind::Y, {a}}); break;
            case 4: g.push_back({GateKind::Z, {a}}); break;
            case 5: if (n > 1) g.push_back({GateKind::CX, {a, b}}); break;
            case 6: if (n > 1) g.push_back({GateKind::CZ, {a, b}}); break;
            default: g.push_back({GateKind::H, {a}}); break;
        }
    }
    return g;
}

std::vector<DiagResult::ZTerm> random_terms(std::mt19937_64& rng, int n, int nterms) {
    std::vector<DiagResult::ZTerm> tz;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < nterms; ++i) {
        DiagResult::ZTerm zt{Pauli(n), (int)(rng() % 31) - 15};
        const int style = (int)(rng() % 5);
        if (style == 0) {                      // empty
        } else if (style == 1) {               // single qubit
            zt.zstring.setz((int)(rng() % n));
        } else if (style == 2) {               // sparse
            const int w = 1 + (int)(rng() % 4);
            for (int t = 0; t < w; ++t) zt.zstring.setz((int)(rng() % n));
        } else {                               // dense-ish
            const double p = style == 3 ? 0.5 : 0.1;
            for (int q = 0; q < n; ++q) if (u(rng) < p) zt.zstring.setz(q);
        }
        if (rng() % 7 == 0) zt.zstring.phase = 2;
        tz.push_back(std::move(zt));
    }
    return tz;
}

void fuzz_new_equals_scalar() {
    std::mt19937_64 rng(0x5eedULL);
    const int ns[] = {1, 2, 5, 63, 64, 65, 127, 128, 129, 200, 330};
    long checked_cols = 0, checked_const = 0, multiword = 0;
    for (int n : ns) {
        for (int rep = 0; rep < 12; ++rep) {
            const double hfrac = (rep % 3 == 0) ? 0.6 : (rep % 3 == 1 ? 0.3 : 0.05);
            const int ngates = (int)(n * (1 + rep % 4)) + 3;
            auto gates = random_gates(rng, n, ngates, hfrac);
            auto tz = random_terms(rng, n, 40);
            ParitySupport ps = restrict_to_support(gates, tz, n);
            OracleOut o = scalar_oracle(ps.support, tz, n);
            CHECK_EQ(ps.columns.size(), o.columns.size());
            CHECK_EQ(ps.coeffs.size(), o.coeffs.size());
            CHECK_EQ(ps.global_phase_mult8, o.gphase);
            if (ps.columns.size() == o.columns.size())
                for (size_t i = 0; i < o.columns.size(); ++i) {
                    CHECK(ps.columns[i] == o.columns[i]);
                    CHECK_EQ(ps.coeffs[i], o.coeffs[i]);
                }
            checked_cols += (long)o.columns.size();
            checked_const += (long)tz.size() - (long)o.columns.size();
            if (ps.support.k_ > 64) ++multiword;
        }
    }
    // the fuzz must exercise both branches and multi-word columns
    CHECK(checked_cols > 1000);
    CHECK(checked_const > 100);
    CHECK(multiword > 5);
    std::fprintf(stderr, "  magic columns %ld, constant terms %ld, multi-word cases %ld\n",
                 checked_cols, checked_const, multiword);
}

int col_rank(const AffineState& s) {
    // independent GF(2) rank of the n×k matrix R (scalar elimination on transposed columns)
    const int n = s.n_, k = s.k_;
    std::vector<std::vector<uint8_t>> cols(k, std::vector<uint8_t>(n, 0));
    for (int j = 0; j < k; ++j) for (int i = 0; i < n; ++i) cols[j][i] = s.R.get(i, j) ? 1 : 0;
    int r = 0;
    for (int i = 0; i < n && r < k; ++i) {
        int p = -1;
        for (int j = r; j < k; ++j) if (cols[j][i]) { p = j; break; }
        if (p < 0) continue;
        std::swap(cols[p], cols[r]);
        for (int j = 0; j < k; ++j)
            if (j != r && cols[j][i]) for (int t = 0; t < n; ++t) cols[j][t] ^= cols[r][t];
        ++r;
    }
    return r;
}

bool same_state(const AffineState& a, const AffineState& b) {
    if (a.k_ != b.k_ || a.b != b.b || a.D != b.D) return false;
    if (!a.omega.exact_eq(b.omega)) return false;
    for (int i = 0; i < a.n_; ++i) for (int j = 0; j < a.k_; ++j) if (a.R.get(i, j) != b.R.get(i, j)) return false;
    for (int i = 0; i < a.k_; ++i) for (int j = 0; j < a.k_; ++j) if (a.J.get(i, j) != b.J.get(i, j)) return false;
    return true;
}

void fuzz_full_rank_invariant() {
    std::mt19937_64 rng(0xabcdULL);
    long hs = 0, zero_row_hs = 0, projections = 0, zero_projections = 0, cxs = 0;
    const int ns[] = {1, 3, 8, 40, 70, 130};
    for (int n : ns) {
        for (int rep = 0; rep < 6; ++rep) {
            AffineState s(n);
            auto gates = random_gates(rng, n, 30 * n + 10, rep % 2 ? 0.5 : 0.2);
            for (size_t gi = 0; gi < gates.size(); ++gi) {
                const auto& g = gates[gi];
                const int q = g.targets[0];
                if (g.kind == GateKind::H) {
                    const bool zero_row = [&] { for (int j = 0; j < s.k_; ++j) if (s.R.get(q, j)) return false; return true; }();
                    s.apply_h(q);
                    ++hs; zero_row_hs += zero_row;
                } else if (g.kind == GateKind::S) s.apply_s(q);
                else if (g.kind == GateKind::SDG) s.apply_sdg(q);
                else if (g.kind == GateKind::X) s.apply_x(q);
                else if (g.kind == GateKind::Y) s.apply_y(q);
                else if (g.kind == GateKind::Z) s.apply_z(q);
                else if (g.kind == GateKind::CX) { s.apply_cx(q, g.targets[1]); ++cxs; }
                else if (g.kind == GateKind::CZ) s.apply_cz(q, g.targets[1]);
                {   // extra canonicalize() after EVERY gate is a no-op
                    AffineState c = s;
                    c.canonicalize();
                    CHECK(same_state(c, s));
                }
                if (rng() % 25 == 0) {   // Pauli projection (X/Y/Z) on both branches; keep a nonzero one
                    const int pauli = (int)(rng() % 3);
                    AffineState c = s;
                    ExactPhase a = c.pauli_project(pauli, q, +1);
                    AffineState c2 = s;
                    ExactPhase a2 = c2.pauli_project(pauli, q, -1);
                    for (AffineState* br : {&c, &c2}) {   // incl. the zero-probability branch
                        CHECK_EQ(col_rank(*br), br->k_);
                        AffineState cc = *br;
                        cc.canonicalize();
                        CHECK(same_state(cc, *br));
                    }
                    if (a.is_zero || a2.is_zero) ++zero_projections;
                    if (!a.is_zero) { s = c; ++projections; }
                    else if (!a2.is_zero) { s = c2; ++projections; }
                }
                CHECK_EQ(col_rank(s), s.k_);
            }
        }
    }
    CHECK(zero_row_hs > 100);
    CHECK(hs - zero_row_hs > 100);
    CHECK(projections > 50);
    CHECK(zero_projections > 20);
    CHECK(cxs > 1000);
    std::fprintf(stderr, "  H %ld (zero-row fast exit %ld), CX %ld, projections %ld (zero-prob branch %ld)\n",
                 hs, zero_row_hs, cxs, projections, zero_projections);
}

// canonicalize()'s kernel-vector routine: the rank-(k-1) back-substitution must return exactly
// the augmented Gauss–Jordan oracle's vector (and the same full-rank / deficient verdict) on
// random R with nullity 0, 1 and >= 2, k on both sides of 64.
void fuzz_kernel_vec_fast_equals_augmented() {
    std::mt19937_64 rng(0x4b65ULL);
    long cases = 0, null1 = 0, null2 = 0, full = 0, null1_nkm1 = 0, null1_wide_full_top = 0;
    const int ks[] = {1, 2, 7, 63, 64, 65, 100, 127, 128, 129, 130, 191, 192, 193, 200, 256};
    for (int k : ks) {
        for (int rep = 0; rep < 30; ++rep) {
            // n = k-1 (the reachable post-H shape when every qubit was superposed: k_old = n),
            // n = k, n >> k, and n < k-1 (nullity >= 2 forced)
            const int mode = rep % 4;
            const int n = mode == 0 ? std::max(1, k - 1)
                        : mode == 1 ? k
                        : mode == 2 ? k + 1 + (int)(rng() % (2 * k + 5))
                        : std::max(1, k / 2);
            const int nullity = (int)(rng() % 3);            // 0, 1, 2 dependent columns
            const double dens = (rep % 3 == 0) ? 0.03 : 0.5;
            std::uniform_real_distribution<double> u(0.0, 1.0);
            std::vector<std::vector<uint8_t>> cols;
            for (int j = 0; j < k; ++j) {
                std::vector<uint8_t> c(n, 0);
                if (j >= k - nullity && j >= 1) {             // dependent: XOR of random earlier columns
                    for (int i = 0; i < j; ++i) if (rng() & 1) for (int r = 0; r < n; ++r) c[r] ^= cols[i][r];
                } else {
                    for (int r = 0; r < n; ++r) c[r] = u(rng) < dens;
                }
                cols.push_back(c);
            }
            // shuffle column order so dependent columns are not always last
            for (int j = k - 1; j > 0; --j) std::swap(cols[j], cols[(size_t)(rng() % (j + 1))]);
            PackedMat R(n);
            for (auto& c : cols) R.append_col(c);
            std::vector<uint8_t> vf, va;
            const bool df = affine_detail::kernel_vec_for_test(R, n, k, vf, false);
            const bool da = affine_detail::kernel_vec_for_test(R, n, k, va, true);
            CHECK_EQ(df, da);
            if (df && da) {
                CHECK(vf == va);
                // and it IS a kernel vector
                for (int r = 0; r < n; ++r) {
                    int s = 0;
                    for (int j = 0; j < k; ++j) s ^= (vf[j] & cols[j][r]);
                    CHECK_EQ(s, 0);
                }
            }
            const int rk = [&] {
                std::vector<std::vector<uint8_t>> m = cols; int r = 0;
                for (int i = 0; i < n && r < k; ++i) {
                    int p = -1; for (int j = r; j < k; ++j) if (m[j][i]) { p = j; break; }
                    if (p < 0) continue;
                    std::swap(m[p], m[r]);
                    for (int j = 0; j < k; ++j) if (j != r && m[j][i]) for (int t2 = 0; t2 < n; ++t2) m[j][t2] ^= m[r][t2];
                    ++r;
                }
                return r; }();
            ++cases; if (rk == k) ++full; else if (rk == k - 1) ++null1; else ++null2;
            if (rk == k - 1 && n == k - 1) ++null1_nkm1;
            if (rk == k - 1 && k > 64 && k % 64 == 0) ++null1_wide_full_top;
        }
    }
    CHECK(null1 > 40);
    CHECK(null2 > 20);
    CHECK(full > 40);
    CHECK(null1_nkm1 > 20);
    CHECK(null1_wide_full_top > 10);
    std::fprintf(stderr, "  cases %ld (full %ld, nullity-1 %ld [n=k-1: %ld, k in {128,192,256}: %ld], nullity>=2 %ld)\n",
                 cases, full, null1, null1_nkm1, null1_wide_full_top, null2);
}

// restrict_to_support with the free count d forced to 0, 64, 128, 192 (last_mask = ~0) and to
// 65 / 129: H on exactly d of n qubits, then CX scrambling (rank-preserving, d unchanged).
void restrict_word_boundaries() {
    std::mt19937_64 rng(0x77ULL);
    long cases = 0;
    for (int d : {0, 64, 128, 192, 65, 129}) {
        for (int extra : {0, 1, 37}) {
            const int n = std::max(1, d + extra);
            std::vector<CliffGate> g;
            for (int q = 0; q < d; ++q) g.push_back({GateKind::H, {q}});
            if (n > 1) for (int r = 0; r < 3 * n; ++r) {
                const int a = (int)(rng() % n), b = (int)(rng() % n);
                if (a != b) g.push_back({GateKind::CX, {a, b}});
            }
            auto tz = random_terms(rng, n, 60);
            ParitySupport ps = restrict_to_support(g, tz, n);
            CHECK_EQ(ps.support.k_, d);
            OracleOut o = scalar_oracle(ps.support, tz, n);
            CHECK_EQ(ps.columns.size(), o.columns.size());
            CHECK_EQ(ps.global_phase_mult8, o.gphase);
            if (ps.columns.size() == o.columns.size())
                for (size_t i = 0; i < o.columns.size(); ++i) {
                    CHECK(ps.columns[i] == o.columns[i]);
                    CHECK_EQ(ps.coeffs[i], o.coeffs[i]);
                }
            if (d > 0) CHECK(!o.columns.empty());
            ++cases;
        }
    }
    std::fprintf(stderr, "  cases %ld (d in {0,64,128,192,65,129})\n", cases);
}

// The QECCORE_CHECK_INVARIANTS mechanism: an H on a zero R-row of a state whose R is NOT full
// column rank (built by writing the public fields) must throw in the test build.
void invariant_check_fires() {
#ifndef QECCORE_CHECK_INVARIANTS
    CHECK(false && "ctest build must define QECCORE_CHECK_INVARIANTS (cpp/CMakeLists.txt)");
#else
    AffineState s(3);
    s.apply_h(0);
    s.apply_h(1);                       // k = 2, R columns e0, e1; row 2 zero
    CHECK_EQ(s.k_, 2);
    s.R.set(0, 1, 1);
    s.R.set(1, 1, 0);                   // corrupt: column 1 := e0 = column 0 (rank 1 < k)
    bool threw = false;
    try { s.apply_h(2); } catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
    AffineState ok(3);                  // control: the same H on a valid state does not throw
    ok.apply_h(0); ok.apply_h(1);
    bool threw_ok = false;
    try { ok.apply_h(2); } catch (const std::logic_error&) { threw_ok = true; }
    CHECK(!threw_ok);
#endif
}

}  // namespace

int main() {
    RUN(invariant_check_fires);
    RUN(restrict_word_boundaries);
    RUN(fuzz_kernel_vec_fast_equals_augmented);
    RUN(fuzz_new_equals_scalar);
    RUN(fuzz_full_rank_invariant);
    REPORT();
}
