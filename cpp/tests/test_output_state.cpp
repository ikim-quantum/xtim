// Output-state helpers (engine tests).
#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/output_state.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

static bool rbit(const Row& r, int i) { return (r[(size_t)(i >> 6)] >> (i & 63)) & 1ULL; }
static Row row_of(unsigned v) { Row r(1, 0); r[0] = v; return r; }
static int m8(int v) { return ((v % 8) + 8) % 8; }

// The unique Z8 multilinear coefficients (degree 1..3; the constant dropped) of sum_k a_k [col_k . t]:
// [l.t] = sum over nonempty T ⊆ supp(l), |T| <= 3, of (-2)^{|T|-1} t^T (mod 8). Equal maps <=> equal functions up to a
// constant. Symbolic: no evaluation at points.
using Mono = std::map<std::vector<int>, int>;
static void addm(Mono& m, std::vector<int> k, int v) { int& x = m[k]; x = m8(x + v); if (!x) m.erase(k); }
static Mono multilinear(const std::vector<Row>& cols, const std::vector<int>& a, int d) {
    Mono m;
    for (size_t k = 0; k < cols.size(); ++k) {
        std::vector<int> s;
        for (int i = 0; i < d; ++i) if (rbit(cols[k], i)) s.push_back(i);
        for (size_t x = 0; x < s.size(); ++x) {
            addm(m, {s[x]}, a[k]);
            for (size_t y = x + 1; y < s.size(); ++y) {
                addm(m, {s[x], s[y]}, -2 * a[k]);
                for (size_t z = y + 1; z < s.size(); ++z) addm(m, {s[x], s[y], s[z]}, 4 * a[k]);
            }
        }
    }
    return m;
}
static std::vector<int> rel(const std::vector<int>& c) { std::vector<int> a; for (int v : c) a.push_back(m8(-v)); return a; }
static Mono fn(const std::vector<Row>& cols, const std::vector<int>& c, int d) { return multilinear(cols, rel(c), d); }
// The independent oracle for Lemma 1 (I5(b)): w is a stabilizer X-part iff D_w f = f(t+w) - f(t) has linear
// coefficients in {0, 4} and no quadratic or cubic part (mod 8). Computed symbolically: [l.(t+w)] = (l.w) ? 1 - [l.t] : [l.t].
static bool stabilizer_shift(const std::vector<Row>& cols, const std::vector<int>& c, int d, unsigned w) {
    std::vector<int> a = rel(c), a2 = a;
    for (size_t k = 0; k < cols.size(); ++k) if (__builtin_popcountll(cols[k][0] & w) & 1) a2[k] = m8(-a[k]);
    Mono f0 = multilinear(cols, a, d), f1 = multilinear(cols, a2, d), D = f1;
    for (const auto& kv : f0) addm(D, kv.first, -kv.second);
    for (const auto& kv : D) if (kv.first.size() > 1 || (kv.second != 4)) return false;
    return true;
}
static bool kernel_matches_brute_force(const std::vector<Row>& cols, const std::vector<int>& c, int d) {
    std::vector<Row> odd;
    for (size_t k = 0; k < cols.size(); ++k) if (c[k] & 1) odd.push_back(cols[k]);
    const bool clean = magic_shift_kernel(odd, d).empty();
    bool equal = true;                                        // W_true == L^perp  <=>  clean
    for (unsigned w = 1; w < (1u << d); ++w) {                // candidate X-parts (d <= 6), each checked symbolically
        bool perp = true;
        for (const Row& l : odd) perp &= !(__builtin_popcountll(l[0] & w) & 1);
        if (stabilizer_shift(cols, c, d, w) != perp) equal = false;
    }
    return clean == equal;
}

// The documented postconditions of stabilize_magic_columns (M3): no constant column survives (zero direction, or
// coefficient 0 or 8 mod 16), and the step count is at most the odd rank of the merged input.
static int odd_rank(const std::vector<Row>& cols, const std::vector<int>& c, int d) {
    std::map<Row, int> m;
    for (size_t k = 0; k < cols.size(); ++k) { int& v = m[cols[k]]; v = ((v + c[k]) % 16 + 16) % 16; }
    Projector P;
    P.words = std::max(1, (d + 63) / 64);
    for (const auto& kv : m) if ((kv.second & 1) && kv.first[0]) P.extend(kv.first);
    return P.r();
}
static bool no_constant_columns(const std::vector<Row>& cols, const std::vector<int>& c) {
    for (size_t k = 0; k < cols.size(); ++k)
        if (cols[k][0] == 0 || c[k] % 16 == 0 || c[k] % 16 == 8) return false;
    return true;
}
static bool postconditions(const std::vector<Row>& cols, const std::vector<int>& c, int d) {
    std::vector<Row> c2 = cols; std::vector<int> k2 = c;
    const int steps = stabilize_magic_columns(c2, k2, d);
    return steps <= odd_rank(cols, c, d) && no_constant_columns(c2, k2);
}

static void coefficient_8_columns_dropped() {
    std::vector<Row> cols = {row_of(1), row_of(2), row_of(3), row_of(3), row_of(0)};
    std::vector<int> c = {8, 7, 5, 11, 3};                    // an 8, a pair summing to 16, a zero direction
    std::vector<Row> c2 = cols; std::vector<int> k2 = c;
    CHECK_EQ(stabilize_magic_columns(c2, k2, 2), 0);
    CHECK_EQ(c2.size(), 1u);
    CHECK(c2[0] == row_of(2)); CHECK_EQ(k2[0], 7);
    std::vector<Row> c3 = {row_of(1), row_of(1)}; std::vector<int> k3 = {3, 5};   // 3 + 5 = 8: a global phase
    CHECK_EQ(stabilize_magic_columns(c3, k3, 1), 0);
    CHECK(c3.empty());
}

static void shift_kernel_nonzero_example() {
    const int d = 4;                                          // the 8 directions with bit 0 set, each a T
    std::vector<Row> cols; std::vector<int> c;
    for (unsigned v = 0; v < 16; ++v) if (v & 1u) { cols.push_back(row_of(v)); c.push_back(7); }
    CHECK(!magic_shift_kernel(cols, d).empty());
    CHECK(kernel_matches_brute_force(cols, c, d));
    std::vector<Row> c2 = cols; std::vector<int> k2 = c;
    CHECK(stabilize_magic_columns(c2, k2, d) >= 1);
    CHECK(postconditions(cols, c, d));
    CHECK(no_constant_columns(c2, k2));
    CHECK(fn(cols, c, d) == fn(c2, k2, d));
    CHECK(kernel_matches_brute_force(c2, k2, d));
    std::vector<Row> o2; for (size_t k = 0; k < c2.size(); ++k) if (k2[k] & 1) o2.push_back(c2[k]);
    CHECK(magic_shift_kernel(o2, d).empty());
}

static void independent_columns_untouched() {
    std::vector<Row> cols = {row_of(1), row_of(2), row_of(4)}; std::vector<int> c = {7, 7, 1};
    std::vector<Row> c2 = cols; std::vector<int> k2 = c;
    CHECK_EQ(stabilize_magic_columns(c2, k2, 3), 0);
    CHECK(fn(cols, c, 3) == fn(c2, k2, 3));
}

// I6: unions of affine blocks {v in U : v.a = 1} (a random subspace U, a not ⊥ U) with one common odd coefficient,
// plus random extra columns — the configurations that make the shift step fire.
static void affine_block(std::mt19937& rng, int d, std::vector<Row>& cols, std::vector<int>& c) {
    const int dimU = 2 + (int)(rng() % (unsigned)(d - 1));
    std::vector<unsigned> basis;
    while ((int)basis.size() < dimU) basis.push_back(1u + (unsigned)(rng() % ((1u << d) - 1)));
    unsigned a = 1u + (unsigned)(rng() % ((1u << d) - 1));
    const int coeff = 1 + 2 * (int)(rng() % 8);
    std::vector<unsigned> span = {0};
    for (unsigned b : basis) { const size_t s = span.size(); for (size_t i = 0; i < s; ++i) span.push_back(span[i] ^ b); }
    std::sort(span.begin(), span.end()); span.erase(std::unique(span.begin(), span.end()), span.end());
    for (unsigned v : span) if (__builtin_popcount(v & a) & 1) { cols.push_back(row_of(v)); c.push_back(coeff); }
}

static void random_sets_preserve_function_and_match_brute_force() {
    std::mt19937 rng(20261117);
    int stepped = 0;
    for (int it = 0; it < 400; ++it) {
        const int d = 3 + (int)(rng() % 4);                   // 3..6
        std::vector<Row> cols; std::vector<int> c;
        const int blocks = (int)(rng() % 3);                  // 0, 1 or 2 affine blocks
        for (int b = 0; b < blocks; ++b) affine_block(rng, d, cols, c);
        for (int k = 0; k < (int)(rng() % 5); ++k) { cols.push_back(row_of(1u + (unsigned)(rng() % ((1u << d) - 1)))); c.push_back(1 + (int)(rng() % 15)); }
        if (cols.empty()) continue;
        CHECK(kernel_matches_brute_force(cols, c, d));
        std::vector<Row> c2 = cols; std::vector<int> k2 = c;
        const int steps = stabilize_magic_columns(c2, k2, d);
        stepped += steps > 0;
        CHECK(steps <= odd_rank(cols, c, d));
        CHECK(no_constant_columns(c2, k2));
        CHECK(fn(cols, c, d) == fn(c2, k2, d));
        CHECK(kernel_matches_brute_force(c2, k2, d));
    }
    std::fprintf(stderr, "[info] sets that took a shift step: %d / 400\n", stepped);
    CHECK(stepped >= 20);                                     // measured 67 / 400 (16.75%) with this generator
}

// The twirl sampler's normalization preset (cpp/bindings/twirl_front.hpp:69-80), repeated for the test.
static NormalizeResult norm_twirl(const Circuit& c) {
    NormalizePolicy pol;
    pol.coherentize = true; pol.coherentize_all = true; pol.defer = true; pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject; pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    return normalize(c, pol);
}
static OutputAnalysis run(const std::string& text, const std::vector<int>& qubits) {
    ParsedStim ps = parse_stim_circuit(text, true);
    CHECK(ps.ok());                                           // Task 2 review m2: a typo in a toy fails loudly
    if (!ps.ok()) return OutputAnalysis{};
    NormalizeResult nr = norm_twirl(ps.circuit);
    std::vector<int> ow;
    for (int q : qubits) {
        CHECK(q >= 0 && q < (int)nr.map.final_wire.size());
        if (q < 0 || q >= (int)nr.map.final_wire.size()) return OutputAnalysis{};
        ow.push_back(nr.map.final_wire[(size_t)q]);
    }
    return analyze_output(nr.normalized, ow, true);
}
static Pauli lp(int n, const std::string& s, int sign) {
    Pauli p(n);
    int y = 0;
    for (int i = 0; i < n; ++i) {
        if (s[(size_t)i] == 'X' || s[(size_t)i] == 'Y') p.setx(i);
        if (s[(size_t)i] == 'Z' || s[(size_t)i] == 'Y') p.setz(i);
        y += s[(size_t)i] == 'Y';
    }
    p.phase = ((y + (sign < 0 ? 2 : 0)) % 4 + 4) % 4;
    return p;
}
static bool eq(const Pauli& a, const Pauli& b) { return a.x == b.x && a.z == b.z && ((a.phase - b.phase) % 4 + 4) % 4 == 0; }
#define DET(r) CHECK((r).verdict == OutputVerdict::Deterministic)

static void qubit_end_classification() {          // C1 / Q2
    auto ends = [](const std::string& t) { return qubit_ends(parse_stim_circuit(t, true)); };
    CHECK(ends("H 0 1\nMPP X0*X1\n")[0] == QubitEnd::Gate);         // MPP measures an ancilla
    CHECK(ends("H 0\nMX 0\n")[0] == QubitEnd::Measured);
    CHECK(ends("H 0\nMRX 0\n")[0] == QubitEnd::Reset);
    CHECK(ends("H 0\nRX 0\n")[0] == QubitEnd::Reset);
    CHECK(ends("H 0\nM 0\nH 0\n")[0] == QubitEnd::Gate);
    CHECK(ends("H 0\nM 0\nCX rec[-1] 1\n")[1] == QubitEnd::Gate);
    CHECK(ends("X_ERROR(0.1) 0\nH 1\n")[0] == QubitEnd::Untouched);
}

static void deterministic_toys() {
    { const OutputAnalysis r = run("H 0\nT 0\n", {0}); DET(r);
      CHECK_EQ(r.rotations.size(), 1u); CHECK(eq(r.rotations[0], lp(1, "Z", +1)));
      CHECK(r.pauli_stabilizers.empty()); CHECK_EQ(r.clifford_paulis.size(), 1u); CHECK(eq(r.clifford_paulis[0], lp(1, "X", +1))); }
    { const OutputAnalysis r = run("H 0\nT_DAG 0\n", {0}); DET(r);                       // M6
      // The parser writes T_DAG as `T; S_DAG`, so the parity form sees a count-1 column with the S_DAG in the Clifford:
      // exp(-i pi/8 Z)|-i> is the same state as exp(+i pi/8 Z)|+> (decision 2: the completing Pauli may be X or Y).
      // Both exact representations of T_DAG|+> pass; the two sign errors, (Z,+1)|+> = T|+> and (Z,-1)|-i>, fail.
      CHECK_EQ(r.rotations.size(), 1u); CHECK_EQ(r.clifford_paulis.size(), 1u); CHECK(r.pauli_stabilizers.empty());
      CHECK((eq(r.rotations[0], lp(1, "Z", -1)) && eq(r.clifford_paulis[0], lp(1, "X", +1))) ||
            (eq(r.rotations[0], lp(1, "Z", +1)) && eq(r.clifford_paulis[0], lp(1, "Y", -1)))); }
    { const OutputAnalysis r = run("RX 0\nT 0\nR 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nM 0 1\nCX rec[-1] 2\nCZ rec[-2] 2\n", {2}); DET(r);
      CHECK_EQ(r.rotations.size(), 1u); CHECK(eq(r.rotations[0], lp(1, "Z", +1))); CHECK(eq(r.clifford_paulis[0], lp(1, "X", +1))); }
    { const OutputAnalysis r = run("H 0\nCX 0 1\n", {0, 1}); DET(r);
      CHECK(r.rotations.empty()); CHECK(r.clifford_paulis.empty()); CHECK_EQ(r.pauli_stabilizers.size(), 2u); }
    { const OutputAnalysis r = run("H 0\nT 0\nM 0\nH 1\n", {1}); DET(r);                 // magic on the rest only
      CHECK(r.rotations.empty()); CHECK_EQ(r.pauli_stabilizers.size(), 1u); CHECK(eq(r.pauli_stabilizers[0], lp(1, "X", +1))); }
    { const OutputAnalysis r = run("H 0 1\nT 0 1\nM 1\n", {0}); DET(r);                  // I2: magic on both sides
      CHECK_EQ(r.rotations.size(), 1u); CHECK(eq(r.rotations[0], lp(1, "Z", +1))); }
    { const OutputAnalysis r = run("H 0 1\nT 0 1\n", {0}); DET(r); }
    { const OutputAnalysis r = run("H 0 1 2\nCCZ 0 1 2\n", {0, 1, 2}); DET(r);
      CHECK_EQ(r.rotations.size(), 7u); CHECK(r.pauli_stabilizers.empty()); CHECK_EQ(r.clifford_paulis.size(), 3u); }
}

static void not_deterministic_toys() {
    CHECK(run("H 0\nCX 0 1\n", {0}).verdict == OutputVerdict::CrossStabilizer);                      // entangled leftover
    CHECK(run("RX 0\nT 0\nR 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nM 0 1\nCX rec[-1] 2\n", {2}).verdict    // missing CZ feedback
          == OutputVerdict::CrossStabilizer);
    CHECK(run("H 0 1\nCX 0 1\nT 1\nCX 0 1\n", {0}).verdict == OutputVerdict::CrossStabilizer);      // Z0Z1 rotation on |++>
    CHECK(run("H 0\nCX 0 1\nT 0\nH 1\nT 1\nH 1\n", {0}).verdict == OutputVerdict::CoupledStabilizer); // local axes, Bell S
    // Lemma 4a: X0 and X0X1 rotations on |00> (independent columns): the straddling X0X1 decides "no"
    const OutputAnalysis r = run("H 0\nT 0\nH 0\nH 0 1\nCX 0 1\nT 1\nCX 0 1\nH 0 1\n", {0});
    CHECK(r.verdict == OutputVerdict::CoupledMagic);
    CHECK_EQ(r.witnesses.size(), 1u);
}

static void undecided_toys() {                    // A3: straddling, all columns dependent
    const OutputAnalysis c = run("H 0 1 2\nCCZ 0 1 2\n", {0});                         // one qubit of CCZ|+++>
    CHECK(c.verdict == OutputVerdict::Undecided);
    CHECK(c.witnesses.size() >= 1);
    const std::string t = "H 1\nS 1\n"                                                  // XX, ZZ, YY rotations on |0,+i>
                          "H 0 1\nCX 0 1\nT 1\nCX 0 1\nH 0 1\n"
                          "CX 0 1\nT 1\nCX 0 1\n"
                          "H_YZ 0 1\nCX 0 1\nT 1\nCX 0 1\nH_YZ 0 1\n";
    CHECK(run(t, {0}).verdict == OutputVerdict::Undecided);
}

static void malformed_qubit_lists_refused() {        // Task 2 review m1: repeated, negative, out-of-range wires
    ParsedStim ps = parse_stim_circuit("H 0\nT 0\nH 1\n", true);
    NormalizeResult nr = norm_twirl(ps.circuit);
    const int w0 = nr.map.final_wire[0], n = nr.normalized.n;
    const auto refused = [&](const std::vector<int>& ow) {
        try { analyze_output(nr.normalized, ow, true); } catch (const std::invalid_argument& e) {
            return std::string(e.what()).find("malformed qubit list") != std::string::npos;
        }
        return false;
    };
    CHECK(refused({w0, w0}));
    CHECK(refused({-1}));
    CHECK(refused({n}));
    CHECK(!refused({w0}));
}

int main() {
    RUN(coefficient_8_columns_dropped);
    RUN(shift_kernel_nonzero_example);
    RUN(independent_columns_untouched);
    RUN(random_sets_preserve_function_and_match_brute_force);
    RUN(qubit_end_classification);
    RUN(deterministic_toys);
    RUN(not_deterministic_toys);
    RUN(undecided_toys);
    RUN(malformed_qubit_lists_refused);
    REPORT();
}
