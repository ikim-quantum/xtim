// test_frame_cosets.cpp — the compiled sector table on hand-built references.
// The bit-for-bit oracle (numpy path == compiled path on the real u2 plans, the fixtures and a
// random sweep) lives in tests/test_frame_cosets_cpp.py; this file checks the C++ core alone:
// the one-branch stabilizer reference (every table is a single frame of weight 1), a chi = 2
// magic reference on which the |supp a| = 1 plan splits into two sectors with the Born weights
// of S|T> = (|0> + e^{i3π/4}|1>)/√2 ... measured on the S-layer, determinism across calls, and
// the two refusal paths.
#include "check.hpp"
#include "qeccore/frame_cosets.hpp"

#include <cmath>
#include <cstring>
#include <vector>

using namespace qeccore::frame_cosets;

namespace {

// Single-qubit |+>: generator X (phase 0), destabilizer Z, no free row, chi = 1.
FrameRef plus_state() {
    const int n = 1;
    uint8_t sx[1] = {1}, sz[1] = {0}, dx[1] = {0}, dz[1] = {1};
    int8_t sp[1] = {0};
    uint8_t bx[1] = {0}, bz[1] = {0};
    cd co[1] = {cd(1.0, 0.0)};
    return FrameRef::from_bits(n, sx, sz, sp, dx, dz, nullptr, 0, nullptr, co, 1, bx, bz);
}

// Single-qubit |T> in the frame of |0>: generator Z (free row, i.e. NOT a stabilizer of |T>),
// destabilizer X; |T> = c0 |0> + c1 X|0> with c0 = 1/√2, c1 = e^{iπ/4}/√2 — chi = 2, k = 1.
FrameRef t_state() {
    const int n = 1;
    uint8_t sx[1] = {0}, sz[1] = {1}, dx[1] = {1}, dz[1] = {0};
    int8_t sp[1] = {0};
    int fr[1] = {0};
    uint8_t sig[2] = {0, 1};
    uint8_t bx[2] = {0, 1}, bz[2] = {0, 0};
    const double r = 1.0 / std::sqrt(2.0);
    cd co[2] = {cd(r, 0.0), cd(r * std::cos(M_PI / 4), r * std::sin(M_PI / 4))};
    return FrameRef::from_bits(n, sx, sz, sp, dx, dz, fr, 1, sig, co, 2, bx, bz);
}

void test_stabilizer_reference_single_frame() {
    FrameRef f = plus_state();
    uint8_t a[1] = {1};
    auto t = plan_frame_cosets(f, a);
    // S|+> = |+i>: Z^v for v ∈ {∅, {0}} — Z anticommutes with X so the two v are DIFFERENT
    // cosets, each of weight 1/2, each a frame copy (chi = 1: the trivial logical image).
    CHECK_EQ(t.size(), (size_t)2);
    for (const Sector& s : t) {
        CHECK(s.frame_copy);
        CHECK(s.has_flip);
        CHECK(std::fabs(s.weight - 0.5) < 1e-15);
        CHECK_EQ(s.fx[0], 0);
        CHECK_EQ(s.fz[0], s.vz[0]);
    }
    CHECK_EQ(t[0].vz[0], 0);   // first encounter: v = ∅ first
    CHECK_EQ(t[1].vz[0], 1);
    uint8_t a0[1] = {0};
    auto t0 = plan_frame_cosets(f, a0);
    CHECK_EQ(t0.size(), (size_t)1);
    CHECK(std::fabs(t0[0].weight - 1.0) < 1e-15);
}

void test_magic_reference_definite_plan_with_logical_flip() {
    FrameRef f = t_state();
    uint8_t a[1] = {1};
    // Z commutes with the (free) generator Z: v = ∅ and v = {0} are the SAME coset, so
    // S|T> is ONE sector, (1 − iZ)|T> = (1−i)c0|0> + (1+i)c1|1> ∝ |0> + e^{i3π/4}|1> — and
    // that is Y|T> up to phase (Y|T> ∝ |0> − e^{−iπ/4}|1>): a definite plan WITH a logical
    // flip, the "one entry = a definite plan (possibly with a logical flip)" case of the
    // docstring.  Frame F = D^x G^z = X·Z over the one free row: fx = destab X, fz = vz ⊕ Z.
    auto t = plan_frame_cosets(f, a);
    CHECK_EQ(t.size(), (size_t)1);
    CHECK(std::fabs(t[0].weight - 1.0) < 1e-15);
    CHECK(t[0].frame_copy);
    CHECK(t[0].has_flip);
    CHECK_EQ(t[0].xb[0], 1);
    CHECK_EQ(t[0].zb[0], 1);
    CHECK_EQ(t[0].vz[0], 0);
    CHECK_EQ(t[0].fx[0], 1);
    CHECK_EQ(t[0].fz[0], 1);
    // the empty plan is the identity: one sector, a frame copy with the trivial flip
    uint8_t a0[1] = {0};
    auto t0 = plan_frame_cosets(f, a0);
    CHECK_EQ(t0.size(), (size_t)1);
    CHECK(t0[0].frame_copy);
    CHECK_EQ(t0[0].xb.size(), (size_t)1);
    CHECK_EQ(t0[0].xb[0], 0);
    CHECK_EQ(t0[0].zb[0], 0);
}

void test_two_qubit_magic_splits_into_sectors() {
    // |T> ⊗ |+>  (n = 2): generators Z_0 (free), X_1; destabs X_0, Z_1.  Plan a = {1}:
    // S_1|+> = |+i>: Z_1 anticommutes with X_1 → two cosets, weight 1/2 each, both frame copies
    // with the trivial logical flip (the magic qubit is untouched).
    const int n = 2;
    uint8_t sx[4] = {0, 0, 0, 1}, sz[4] = {1, 0, 0, 0}, dx[4] = {1, 0, 0, 0}, dz[4] = {0, 0, 0, 1};
    int8_t sp[2] = {0, 0};
    int fr[1] = {0};
    uint8_t sig[2] = {0, 1};
    uint8_t bx[4] = {0, 0, 1, 0}, bz[4] = {0, 0, 0, 0};
    const double r = 1.0 / std::sqrt(2.0);
    cd co[2] = {cd(r, 0.0), cd(r * std::cos(M_PI / 4), r * std::sin(M_PI / 4))};
    FrameRef f = FrameRef::from_bits(n, sx, sz, sp, dx, dz, fr, 1, sig, co, 2, bx, bz);
    uint8_t a[2] = {0, 1};
    auto t = plan_frame_cosets(f, a);
    CHECK_EQ(t.size(), (size_t)2);
    double tot = 0.0;
    for (const Sector& s : t) {
        CHECK(s.frame_copy);
        CHECK_EQ(s.xb[0], 0); CHECK_EQ(s.zb[0], 0);
        CHECK(std::fabs(s.weight - 0.5) < 1e-15);
        tot += s.weight;
        CHECK_EQ(s.fx[0], 0); CHECK_EQ(s.fx[1], 0); CHECK_EQ(s.fz[0], 0);
        CHECK_EQ(s.fz[1], s.vz[1]);
    }
    CHECK(std::fabs(tot - 1.0) < 1e-15);
    // determinism: a second call is byte-identical
    auto t2 = plan_frame_cosets(f, a);
    CHECK_EQ(t2.size(), t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        CHECK(t2[i].vz == t[i].vz); CHECK(t2[i].fx == t[i].fx); CHECK(t2[i].fz == t[i].fz);
        CHECK(std::memcmp(&t2[i].weight, &t[i].weight, sizeof(double)) == 0);
        CHECK_EQ(t2[i].frame_copy, t[i].frame_copy);
    }
}

void test_refusals() {
    // (1) lambda_functional: a null-space basis read that is not ±1 → invalid_argument.
    // |+> with the generator's phase corrupted to i·X: <r|Z^∅|r> stays 1 but a plan on a wire
    // whose Z COMMUTES with every generator reads the generator product's phase.  Build a
    // 1-qubit reference whose only generator is i·Z (phase 1) with destabilizer X — then Z^{0}
    // (v = {0}) commutes with the generator, is the generator up to i, and the read is −i.
    {
        uint8_t sx[1] = {0}, sz[1] = {1}, dx[1] = {1}, dz[1] = {0};
        int8_t sp[1] = {1};
        uint8_t bx[1] = {0}, bz[1] = {0};
        cd co[1] = {cd(1.0, 0.0)};
        FrameRef f = FrameRef::from_bits(1, sx, sz, sp, dx, dz, nullptr, 0, nullptr, co, 1, bx, bz);
        uint8_t a[1] = {1};
        bool threw = false;
        try { plan_frame_cosets(f, a); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    // (2) ref_expectation: P commutes with every generator but is not their product → FrameNotFull.
    // Two qubits with generators Z_0, Z_0 (a degenerate "frame": Z_1 commutes with both, no
    // generator anticommutes with the destabilizers' pattern to produce it).
    {
        uint8_t sx[4] = {0, 0, 0, 0}, sz[4] = {1, 0, 1, 0}, dx[4] = {1, 0, 1, 0}, dz[4] = {0, 0, 0, 0};
        int8_t sp[2] = {0, 0};
        uint8_t bx[2] = {0, 0}, bz[2] = {0, 0};
        cd co[1] = {cd(1.0, 0.0)};
        FrameRef f = FrameRef::from_bits(2, sx, sz, sp, dx, dz, nullptr, 0, nullptr, co, 1, bx, bz);
        uint8_t a[2] = {0, 1};
        bool threw = false;
        try { plan_frame_cosets(f, a); } catch (const FrameNotFull&) { threw = true; }
        CHECK(threw);
    }
}

void test_sector_weight_is_the_portable_sequence() {
    // Non-exact inputs; the pinned value is (re·re + im·im) per branch, accumulated left to
    // right from 0.0, with NO contraction — the literals are the Python bodies' results for
    // these inputs (python3: w=0.0; for re,im in ...: w += re*re + im*im), authored here.
    const std::complex<double> A[3] = {{0.1, 0.7}, {-0.3, 0.2}, {0.55, -0.05}};
    const double w3 = detail::sector_weight(A, 3);
    CHECK_EQ(w3, 0.9349999999999999);          // 0x1.deb851eb851ebp-1 (CPython 3.12.13)
    double seq = 0.0;
    for (int i = 0; i < 3; ++i) { const double re = A[i].real(), im = A[i].imag(); seq += re * re + im * im; }
    CHECK_EQ(w3, seq);
    CHECK_EQ(detail::sector_weight(A, 0), 0.0);
    CHECK_EQ(detail::sector_weight(A, 1), 0.49999999999999994);   // 0x1.fffffffffffffp-2: 0.1·0.1 + 0.7·0.7 unfused
}

}  // namespace

int main() {
    RUN(test_stabilizer_reference_single_frame);
    RUN(test_magic_reference_definite_plan_with_logical_flip);
    RUN(test_two_qubit_magic_splits_into_sectors);
    RUN(test_refusals);
    RUN(test_sector_weight_is_the_portable_sequence);
    REPORT();
}
