// test_xdem_writer — the xdem writer model (qeccore/xdem_writer.hpp, SPEC §2) on the §3.7 worked example (ex422_obs):
//  * canonical coordinates: r = 3, m = 2, the gauge / hidden-syndrome lists and bhat of the worked file;
//  * bhat = beta on K agrees with the writer model's raw b (which reproduces beta on ker kappa) on every c_a;
//  * strict tau invariant: for every fault before a group g and every (g, v) with axis a, tau bit a = parity(x_g & v)
//    (x_g from build_fault_columns' location columns); and for every fault and EVERY axis, tau bit a = the
//    anticommutation of the fault pushed forward to the end with axis_q[a] (on cultivation_d3_faithful some of
//    those bits are 1 on axes with no group ahead: strict, not dc-style); ex422's tau sets {2,3}, {2,3}, {};
//    on ex422 and on cultivation_d3_faithful (argv[1]);
//  * fault_rotations with Gl = 0 routes every exponent to E^>;
//  * the E sign convention (SPEC §1.1, E = -E'): ex422's three faults give the worked file's three lines after
//    folding (`rot -Q2 +Q3` twice, the Y line without rotation);
//  * fold_field: an exponent 2 becomes the flip img_a and is dropped; exponents 1 / 3 stay.
//  * LineAccumulator (Task 4): equal keys merge with p1 (1 - p2) + p2 (1 - p1) at the earliest TimeKey; the side
//    splits otherwise equal keys; lines in time order, tc = the side-0 lines; an empty key and p = 0 are dropped;
//    any_rotation; atom_lines resolves by (instance, key) (-1: none); the merge erratum (fix round 1): a line is
//    closed by a later event whose swap with it has a nonzero readout image (w r w r: four lines; with img = 0: two).
//  * Task 11 (moved from the dc-DEM v1 ctests): the diagonality check names both locations; an inverted record fed
//    back before a T gives b = 1; the residue table equals the reference's 15 rows.
// argv[1] = xtim/_examples (set by CMake). Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/dem_preamble.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/xdem_writer.hpp"

using namespace qeccore;

static const std::string kEx422 = R"(R 0 1 2 3
CX 2 0 0 2 2 0
H 0 2
CX 0 2 0 3 2 1 1 2 2 1 1 2 1 3 2 3
H 2 3
S 2 2 3 3
H 2 3
S 0 0 1 1
RX 4 5 6
S_DAG 0
H 0
T_DAG 0
H 0
S 0
CZ 4 0
S_DAG 0
H 0
T 0
H 0
S 0
X_ERROR(0.001) 1
Z_ERROR(0.001) 1
Y_ERROR(0.001) 1
S_DAG 1
H 1
T_DAG 1
H 1
S 1
CZ 4 1
S_DAG 1
H 1
T 1
H 1
S 1
S_DAG 2
H 2
T_DAG 2
H 2
S 2
CZ 4 2
S_DAG 2
H 2
T 2
H 2
S 2
S_DAG 3
H 3
T_DAG 3
H 3
S 3
CZ 4 3
S_DAG 3
H 3
T 3
H 3
S 3
CZ 5 0
CZ 5 1
CZ 5 2
CZ 5 3
CX 6 0
CX 6 1
CX 6 2
CX 6 3
MX 4 5 6
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
MPP X1*X2 Z1*Z2
OBSERVABLE_INCLUDE(0) rec[-2]
OBSERVABLE_INCLUDE(1) rec[-1]
)";

static std::vector<int> bits(const std::vector<uint64_t>& w) {
    std::vector<int> out;
    for (size_t i = 0; i < w.size(); ++i)
        for (uint64_t x = w[i]; x; x &= x - 1) out.push_back((int)(i * 64 + (size_t)__builtin_ctzll(x)));
    return out;
}

struct Model {
    DemPreamble pre;
    xdem::WriterModel W;
};
static bool build(const std::string& text, Model& M) {
    DemExportResult res;
    if (!dem_preamble(text, false, nullptr, M.pre, res, /*build_bare=*/false)) return false;
    M.W = xdem::build_writer_model(M.pre);
    return true;
}

// x patterns of one fault (noise ordinal k, position qi, basis 0 X / 1 Z / 2 Y) and its flipped columns.
static void fault_signature(const xdem::FaultColumns& fc, int k, size_t qi, int basis, std::vector<int>& flips,
                            std::map<int, uint32_t>& x) {
    std::vector<int32_t> sig;
    std::vector<std::pair<int, PauliBasis>> f;
    f.push_back({(int)qi, basis == 0 ? PauliBasis::X : basis == 1 ? PauliBasis::Z : PauliBasis::Y});
    fc.signature(k, f, sig);
    flips.clear();
    x.clear();
    for (int32_t c : sig) {
        if (c < fc.ncols) { flips.push_back(c); continue; }
        const auto& [g, b] = fc.xid[(size_t)(c - fc.ncols)];
        x[g] ^= 1u << b;
    }
}

struct TauCounts {
    long group_checks = 0;      // (fault, (g, v)) pairs checked against parity(x_g & v)
    long strict_checks = 0;     // (fault, axis) pairs checked against the forward-propagated end Pauli
    long strict_ones = 0;       // ... of those, tau bit 1 on an axis with no group ahead of the fault
};

// For every X / Z fault (noise ordinal k, position qi):
//  (i) tau bit a == parity(x_g & v) for every group g after the fault and every (g, v) with an axis a;
//  (ii) strict tau: tau bit a == [the fault pushed forward through the skeleton to the end anticommutes with
//       axis_q[a]], for EVERY axis, including the axes no group after the fault rotates about.
static TauCounts tau_invariant(const Model& M) {
    const auto& sk = *M.W.sk;
    const Circuit& def = M.pre.dn.deferred;
    const xdem::FaultColumns fc = xdem::build_fault_columns(sk, def, sk.targets);
    const xdem::TauColumns tc = xdem::build_tau_columns(sk, def);
    CHECK_EQ(tc.at.size(), fc.at.size());
    std::vector<int> stream_of;                      // noise ordinal -> deferred stream index
    for (size_t i = 0; i < def.stream.size(); ++i)
        if (def.stream[i].kind == Instr::Kind::Noise) stream_of.push_back((int)i);
    const auto& H = M.W.header;
    const size_t A = H.axes.size();
    TauCounts n;
    for (size_t k = 0; k < tc.at.size(); ++k) {
        std::vector<char> ahead(A, 0);               // axes some group after the fault rotates about
        for (size_t g = 0; g < H.groups.size(); ++g)
            if (sk.group_stream[g] > stream_of[k])
                for (const auto& [a, sg] : H.groups[g].v_axis) if (a >= 0) ahead[(size_t)a] = 1;
        const Instr& ins = def.stream[(size_t)stream_of[k]];
        for (size_t qi = 0; qi < tc.at[k].size(); ++qi)
            for (int basis = 0; basis < 2; ++basis) {
                std::vector<int> flips;
                std::map<int, uint32_t> x;
                fault_signature(fc, (int)k, qi, basis, flips, x);
                std::vector<char> tau(A, 0);
                for (int32_t a : tc.at[k][qi][(size_t)basis]) tau[(size_t)a] = 1;
                for (size_t g = 0; g < H.groups.size(); ++g) {
                    if (sk.group_stream[g] <= stream_of[k]) continue;   // the fault is not before g
                    const uint32_t xg = x.count((int)g) ? x[(int)g] : 0u;
                    for (size_t v = 1; v <= H.groups[g].v_axis.size(); ++v) {
                        const int a = H.groups[g].v_axis[v - 1].first;
                        if (a < 0) continue;
                        CHECK_EQ((int)tau[(size_t)a], __builtin_popcount(xg & (uint32_t)v) & 1);
                        ++n.group_checks;
                    }
                }
                Pauli P(sk.N);                       // the fault, pushed forward to the end of the skeleton
                if (basis == 0) P.setx(ins.qubits[qi]); else P.setz(ins.qubits[qi]);
                for (int oi = sk.op_of_stream[(size_t)stream_of[k] + 1]; oi < (int)sk.ops.size(); ++oi)
                    xdem::conj_forward(P, sk.ops[(size_t)oi]);
                for (size_t a = 0; a < A; ++a) {
                    const int want = Pauli::anticommute_bit(P, sk.axis_q[a]);
                    CHECK_EQ((int)tau[a], want);
                    ++n.strict_checks;
                    if (!ahead[a] && tau[a]) ++n.strict_ones;
                }
            }
    }
    return n;
}

static void test_ex422_header() {
    Model M;
    CHECK(build(kEx422, M));
    const xdem::Canon C = xdem::canonical_coordinates(M.W);
    CHECK_EQ(C.r, 3);
    CHECK_EQ(C.m, 2);
    const std::vector<std::vector<int>> gauge = {{0}, {1}, {2}, {0, 1, 2}, {2}, {0, 1, 2}, {0}, {1}};
    const std::vector<std::vector<int32_t>> hidden = {{}, {}, {}, {}, {0}, {0}, {1}, {1}};
    const std::vector<uint8_t> bhat = {0, 0, 0, 0, 1, 1, 1, 1};
    CHECK_EQ(C.kappa.size(), gauge.size());
    for (size_t a = 0; a < gauge.size() && a < C.kappa.size(); ++a) {
        CHECK(bits(C.kappa[a]) == gauge[a]);
        CHECK(C.hidden[a] == hidden[a]);
        CHECK_EQ((int)C.bhat[a], (int)bhat[a]);
        CHECK_EQ((int)C.pivot[a], a < 3 ? 1 : 0);
    }
    // bhat = beta on K: the raw b reproduces beta on ker kappa, so its parity over c_a is bhat_a
    CHECK(M.W.diag_violations.empty());
    std::vector<int> piv;
    for (size_t a = 0; a < C.pivot.size(); ++a) if (C.pivot[a]) piv.push_back((int)a);
    for (size_t a = 0; a < C.pivot.size(); ++a) {
        if (C.pivot[a]) continue;
        int par = M.W.header.axes[a].b;
        for (int t : bits(C.kappa[a])) par ^= M.W.header.axes[(size_t)piv[(size_t)t]].b;
        CHECK_EQ(par, (int)C.bhat[a]);
    }
}

static void test_tau_invariant() {
    Model M;
    CHECK(build(kEx422, M));
    const TauCounts n = tau_invariant(M);
    CHECK(n.group_checks > 0);
    CHECK(n.strict_checks > 0);
    // the exact tau sets of the three faults (X_ERROR 1, Z_ERROR 1, Y_ERROR 1): the worked file's x fields
    const xdem::TauColumns tc = xdem::build_tau_columns(*M.W.sk, M.pre.dn.deferred);
    CHECK_EQ(tc.at.size(), (size_t)3);
    if (tc.at.size() == 3) {
        CHECK(tc.at[0][0][0] == std::vector<int32_t>({2, 3}));            // X
        CHECK(tc.at[1][0][1] == std::vector<int32_t>({2, 3}));            // Z
        std::vector<char> y(M.W.header.axes.size(), 0);                   // Y = X Z: the symmetric difference
        for (int32_t a : tc.at[2][0][0]) y[(size_t)a] ^= 1;
        for (int32_t a : tc.at[2][0][1]) y[(size_t)a] ^= 1;
        CHECK(std::count(y.begin(), y.end(), 1) == 0);
    }
}

static std::string g_ex;
static void test_tau_invariant_cultivation() {
    std::ifstream f(g_ex + "/cultivation_d3_faithful.stim");
    CHECK(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    Model M;
    CHECK(build(ss.str(), M));
    const TauCounts n = tau_invariant(M);
    CHECK(n.group_checks > 1000);
    CHECK(n.strict_checks > 1000);
    CHECK(n.strict_ones > 0);       // strict tau is not the dc-style tau (zero on the axes with no group ahead)
}

// The worked file's lines (SPEC §3.7): flips, then E^< as (axis, e).
static void test_ex422_rotations() {
    Model M;
    CHECK(build(kEx422, M));
    const auto& sk = *M.W.sk;
    const auto& H = M.W.header;
    const Circuit& def = M.pre.dn.deferred;
    const xdem::FaultColumns fc = xdem::build_fault_columns(sk, def, sk.targets);
    const int Gl = xdem::late_group_count(parse_stim_circuit(kEx422, true));
    CHECK_EQ(Gl, (int)H.groups.size());              // every T before t_c: tc 3 = all lines early
    std::vector<xdem::Axis> axes(H.axes.size());
    for (size_t a = 0; a < axes.size(); ++a) axes[a].img.w = H.axes[a].img.w;
    CHECK_EQ(fc.at.size(), (size_t)3);               // X_ERROR 1, Z_ERROR 1, Y_ERROR 1
    using Line = std::pair<std::vector<int>, std::vector<std::pair<int, int>>>;
    std::vector<Line> got;
    for (int k = 0; k < 3 && k < (int)fc.at.size(); ++k) {
        std::vector<int> flips;
        std::map<int, uint32_t> x;
        fault_signature(fc, k, 0, k == 0 ? 0 : k == 1 ? 1 : 2, flips, x);
        std::vector<std::pair<int, uint32_t>> xv;
        for (const auto& [g, p] : x) if (p) xv.push_back({g, p});
        std::vector<uint8_t> Elt(H.axes.size(), 0), Egt(H.axes.size(), 0);
        xdem::fault_rotations(H, xv, Gl, Elt, Egt);
        for (uint8_t e : Egt) CHECK_EQ((int)e, 0);
        std::vector<uint8_t> Elt0(H.axes.size(), 0), Egt0(H.axes.size(), 0);   // Gl = 0: every group late
        xdem::fault_rotations(H, xv, 0, Elt0, Egt0);
        CHECK(Egt0 == Elt);
        for (uint8_t e : Elt0) CHECK_EQ((int)e, 0);
        xdem::Bits f;
        f.w.assign(xdem::words_for(H.D + H.O), 0);
        for (int c : flips) f.w[(size_t)c >> 6] ^= 1ull << (c & 63);
        xdem::fold_field(axes, Elt, f);
        Line L;
        L.first = bits(f.w);
        for (size_t a = 0; a < Elt.size(); ++a) if (Elt[a]) L.second.push_back({(int)a, (int)Elt[a]});
        got.push_back(L);
    }
    // error(0.001) D0 D1 L1 x Q2 Q3 rot -Q2 +Q3 / D2 L0 x Q2 Q3 rot -Q2 +Q3 / D0 D1 D2 L0 L1 (L0 = column 3)
    std::vector<Line> want = {{{0, 1, 4}, {{2, 3}, {3, 1}}}, {{2, 3}, {{2, 3}, {3, 1}}}, {{0, 1, 2, 3, 4}, {}}};
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    CHECK(got == want);
}

static void test_fold_field() {
    std::vector<xdem::Axis> axes(3);
    axes[0].img.w = {0b0011};
    axes[1].img.w = {0b0110};
    axes[2].img.w = {0b1100};
    std::vector<uint8_t> E = {2, 1, 2};
    xdem::Bits f;
    f.w = {0b0001};
    xdem::fold_field(axes, E, f);
    CHECK_EQ(f.w[0], (uint64_t)(0b0001 ^ 0b0011 ^ 0b1100));
    CHECK(E == std::vector<uint8_t>({0, 1, 0}));
    std::vector<uint8_t> E2 = {3, 0, 1};
    xdem::fold_field(axes, E2, f);
    CHECK_EQ(f.w[0], (uint64_t)(0b0001 ^ 0b0011 ^ 0b1100));
    CHECK(E2 == std::vector<uint8_t>({3, 0, 1}));
}

static void test_line_accumulator() {
    using xdem::LineKey;
    xdem::LineAccumulator acc;
    {
        std::vector<xdem::Axis> ax0(2);
        ax0[0].img.w = {0b01}; ax0[1].img.w = {0b10};
        acc.set_axes(ax0);
    }
    LineKey a;  a.f = {0};
    LineKey b;  b.f = {1}; b.tau = {0}; b.lt = {{0, 3}};
    LineKey b1 = b; b1.side = 1;                       // b after t_c: a separate line
    LineKey e;                                          // an empty key: dropped
    LineKey z;  z.tau = {1};                            // tau only: kept, and commutes with b
    acc.add(b, 0.1, {5, 2, 0}, 0);
    acc.add(a, 0.2, {3, 1, 1}, 1);
    acc.add(b, 0.3, {2, 9, 0}, 2);                      // earlier: b sits at step 2
    acc.add(b1, 0.05, {9, 0, 0}, 3);
    acc.add(e, 0.4, {0, 0, 0}, 5);
    acc.add(z, 0.01, {4, 0, 0}, 4);
    LineKey nil; nil.f = {2};
    acc.add(nil, 0.0, {1, 0, 0}, 6);                    // p = 0: dropped
    acc.note_atom(a, 1); acc.note_atom(b1, 3); acc.note_atom(e, 5);
    acc.note_atom(nil, 6); acc.note_atom(b, 0); acc.note_atom(a, 7);
    CHECK(acc.any_rotation());
    xdem::Canon C; C.r = 1; C.m = 0;
    std::vector<xdem::Axis> axes(2);
    axes[0].img.w = {0b01}; axes[0].k = {1};
    axes[1].img.w = {0b10}; axes[1].k = {1};
    const xdem::File F = acc.to_file(2, 0, C, axes);
    CHECK_EQ((int)F.num_lines(), 4);                    // b (step 2, 5), a (3), z (4), b1 (9)
    CHECK_EQ((int)F.tc, 3);
    CHECK(std::abs(F.p[0] - (0.3 + 0.1 - 2 * 0.3 * 0.1)) < 1e-15);
    CHECK(acc.atom_lines() == std::vector<int>({1, 3, -1, -1, 0, -1}));
    CHECK(xdem::emit(F) == "xdem 1 detectors 2 observables 0 gauge 1 hidden 0 tc 3\n"
                           "axis Q0 img D0 gauge g0\naxis Q1 img D1 gauge g0\n"
                           "error(0.34000000000000002) D1 x Q0 rot -Q0\nerror(0.20000000000000001) D0\n"
                           "error(0.01) x Q1\nerror(0.050000000000000003) D1 x Q0 rot -Q0\n");
    xdem::LineAccumulator flat;
    flat.set_axes(axes);
    flat.add(a, 0.1, {0, 0, 0});
    flat.add(z, 0.1, {1, 0, 0});
    CHECK(!flat.any_rotation());
    // the merge erratum (tdag_cycle): w = x Q0, r = x Q0 rot -Q0, in time order w r w r. The swap set of w and r is
    // {0}; with img_0 != 0 each swap is visible, so every event between two equal keys closes the line: four lines.
    // With img_0 = 0 every swap is invisible and the equal keys merge: two lines.
    LineKey w;  w.tau = {0};
    LineKey r;  r.tau = {0}; r.lt = {{0, 3}};
    std::vector<xdem::Axis> vis(2), inv(2);
    vis[0].img.w = {0b01}; vis[1].img.w = {0b10};
    inv[0].img.w = {0}; inv[1].img.w = {0b10};
    CHECK(xdem::swap_set(w, r) == std::vector<int32_t>({0}));
    CHECK(xdem::swap_visible(w, r, vis));
    CHECK(!xdem::swap_visible(w, r, inv));
    CHECK(!xdem::swap_visible(w, w, vis));
    CHECK(xdem::swap_set(r, r).empty());               // (tau & odd) xor itself
    auto cycle = [&](const std::vector<xdem::Axis>& ax) {
        xdem::LineAccumulator c;
        c.set_axes(ax);
        c.add(w, 0.2, {0, 0, 0}); c.add(r, 0.1, {1, 1, 0}); c.add(w, 0.15, {2, 2, 0}); c.add(r, 0.12, {3, 3, 0});
        return c.num_lines();
    };
    CHECK_EQ((int)cycle(vis), 4);
    CHECK_EQ((int)cycle(inv), 2);
    xdem::LineAccumulator pair;                         // r r w w: r merges, w merges
    pair.set_axes(vis);
    pair.add(r, 0.1, {0, 0, 0}); pair.add(r, 0.12, {1, 0, 0}); pair.add(w, 0.2, {2, 0, 0}); pair.add(w, 0.15, {3, 0, 0});
    CHECK_EQ((int)pair.num_lines(), 2);
    xdem::LineAccumulator mid;                          // r, an odd E^> on axis 0, r: swap set {0}, visible: closes
    LineKey q;  q.f = {1}; q.gt = {{0, 1}};
    mid.set_axes(vis);
    mid.add(r, 0.1, {0, 0, 0}); mid.add(q, 0.1, {1, 0, 0}); mid.add(r, 0.1, {2, 0, 0});
    CHECK_EQ((int)mid.num_lines(), 3);
    xdem::LineAccumulator two;                          // S = {0, 1} with img_0 = img_1: invisible, merges
    std::vector<xdem::Axis> same(2);
    same[0].img.w = {0b11}; same[1].img.w = {0b11};
    LineKey r2; r2.tau = {0, 1}; r2.lt = {{0, 1}, {1, 1}};   // tau = odd = {0, 1}
    LineKey w2; w2.tau = {0, 1};
    CHECK(xdem::swap_set(w2, r2) == std::vector<int32_t>({0, 1}));
    CHECK(!xdem::swap_visible(w2, r2, same));
    two.set_axes(same);
    two.add(r2, 0.1, {0, 0, 0}); two.add(w2, 0.1, {1, 0, 0}); two.add(r2, 0.1, {2, 0, 0});
    CHECK_EQ((int)two.num_lines(), 2);
    // fix round 2: one instruction (step 0, ordinal 0) flushed per qubit, both qubits giving key r, split by a visible
    // w between them (flush 0: r, w; flush 1: r). The atoms resolve by flush: lines 0 and 2 (by (step, ordinal, key)
    // alone both would read line 0).
    xdem::LineAccumulator q2;
    q2.set_axes(vis);
    q2.add(r, 0.1, {0, 0, 0}, 0); q2.add(w, 0.1, {0, 0, 1}, 0); q2.add(r, 0.1, {0, 0, 2}, 1);
    q2.note_atom(r, 0); q2.note_atom(w, 0); q2.note_atom(r, 1); q2.note_atom(w, 1);
    CHECK_EQ((int)q2.num_lines(), 3);
    CHECK(q2.atom_lines() == std::vector<int>({0, 1, 2, -1}));
    // the expiry bound: a line no later mechanism can join is not consulted, and the decisions are unchanged
    // (w r w r with an extra unrelated r-class key that expires early)
    xdem::LineAccumulator ex;
    ex.set_axes(vis);
    LineKey r3 = r; r3.f = {1};                         // same (tau, odd) class as r, its own key
    ex.add(r3, 0.1, {0, 0, 0}); ex.add(w, 0.2, {1, 0, 0}); ex.add(r, 0.1, {2, 0, 0}); ex.add(w, 0.15, {3, 0, 0});
    ex.add(r, 0.12, {4, 0, 0});
    CHECK_EQ((int)ex.num_lines(), 5);
}

// Ported in Task 11 from the dc-DEM v1 route's ctests (test_dc_writer_axes, test_dc_file at 6738e40).
// A circuit whose first T's Z reaches the second T as X: diag_violations names both locations, G0 and G1.
static void test_diag_violation() {
    Model M;
    CHECK(build("R 0\nH 0\nT 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n", M));
    CHECK_EQ(M.W.diag_violations.size(), (size_t)1);
    if (M.W.diag_violations.empty()) return;
    const std::string& s = M.W.diag_violations[0];
    CHECK(s.find("G0") != std::string::npos);
    CHECK(s.find("G1") != std::string::npos);
}

// Inverted record feeding a feedback before a T (`M !0; CX rec[-1] 1; T 1`): the record is 1 deterministically, so
// the X fires, qubit 1 is |1> at the T and the axis Z_1 has <Z_1> = -1 on the skeleton state: kappa = 0 (r = 0), the
// axis is ker kappa, b = 1 (sign +). Without the `!` the X never fires: b = 0. (Hand-derived.)
static void test_inverted_feedback() {
    for (int inv = 0; inv < 2; ++inv) {
        Model M;
        CHECK(build(std::string("R 0 1\nM ") + (inv ? "!" : "") +
                    "0\nCX rec[-1] 1\nT 1\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n", M));
        const auto& H = M.W.header;
        CHECK_EQ(H.axes.size(), (size_t)1);
        CHECK_EQ(H.r, 0);
        CHECK(M.W.diag_violations.empty());
        if (H.axes.size() != 1) continue;
        CHECK_EQ(H.groups[0].v_axis[0].first, 0);
        CHECK_EQ(H.groups[0].v_axis[0].second, 1);
        CHECK_EQ(H.axes[0].b, inv);
    }
}

// residue_terms for every gate and every nonzero x equals the reference model's table (the tie-break convention:
// the minimal-weight Z4 solution, first in lexicographic order of e). Constants from the dc-DEM v1 reference
// (tests/_dc_reference.py residue_terms at 6738e40); tests/_xdem_engine.py residue_terms is its Python port.
static void test_residue_table() {
    using Terms = std::vector<std::pair<int, int>>;
    struct Row { const char* g; uint32_t x; Terms t; };
    const std::vector<Row> rows = {
        {"T", 1, {{1, 3}}},
        {"T_DAG", 1, {{1, 1}}},
        {"CS", 1, {{1, 3}, {3, 1}}},
        {"CS", 2, {{2, 3}, {3, 1}}},
        {"CS", 3, {{1, 3}, {2, 3}}},
        {"CS_DAG", 1, {{1, 1}, {3, 3}}},
        {"CS_DAG", 2, {{2, 1}, {3, 3}}},
        {"CS_DAG", 3, {{1, 1}, {2, 1}}},
        {"CCZ", 1, {{2, 1}, {4, 1}, {6, 3}}},
        {"CCZ", 2, {{1, 1}, {4, 1}, {5, 3}}},
        {"CCZ", 3, {{3, 1}, {4, 3}, {7, 3}}},
        {"CCZ", 4, {{1, 1}, {2, 1}, {3, 3}}},
        {"CCZ", 5, {{2, 1}, {5, 3}, {7, 1}}},
        {"CCZ", 6, {{1, 1}, {6, 3}, {7, 1}}},
        {"CCZ", 7, {{3, 1}, {5, 1}, {6, 1}}},
    };
    CHECK_EQ(rows.size(), (size_t)15);
    for (const Row& r : rows) {
        const Terms& got = xdem::residue_terms(r.g, r.x);
        if (got != r.t) std::fprintf(stderr, "  residue mismatch %s x=%u\n", r.g, r.x);
        CHECK(got == r.t);
    }
    bool threw = false;                               // x out of range, unknown gate
    try { xdem::residue_terms("T", 2); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { xdem::residue_terms("CH", 1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

int main(int argc, char** argv) {
    g_ex = argc > 1 ? argv[1] : "xtim/_examples";
    RUN(test_ex422_header);
    RUN(test_tau_invariant);
    RUN(test_tau_invariant_cultivation);
    RUN(test_ex422_rotations);
    RUN(test_fold_field);
    RUN(test_line_accumulator);
    RUN(test_diag_violation);
    RUN(test_inverted_feedback);
    RUN(test_residue_table);
    REPORT();
}
