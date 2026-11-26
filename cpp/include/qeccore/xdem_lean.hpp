// xdem_lean.hpp — the lean xdem route's own pieces (LEAN-EXPORT-PLAN). Nothing here is called by the oracle route.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/xdem_writer.hpp"

namespace qeccore::xdem {
// The lean route's "not served here" result (export_dem_route reruns the oracle route; this text never surfaces).
inline constexpr const char* kXdemLeanDelegate = "xdem lean route: delegated to the oracle route";

// Algebraic determinism (Task 1): the readout columns 0..D+O-1 whose noiseless parity is NOT deterministic (+-1),
// ascending; empty = all deterministic. Decided from the writer model alone (no bare state): column c (record parity
// P) is deterministic iff X(U^dag P U) lies in the span of the axes' X(U^dag Q_a U) (solution set B), and the Z4
// form Phi(u) = 2[kappa_B.u] + sum_{a in A_P} w_a [kappa_a.u], w_a = n_a (1 - 2 b_a) mod 4, is constant: every
// linear coefficient = 0 mod 4 and every off-diagonal pair coefficient = 0 mod 2.
// LM-2: sparse. The work follows the supports: the readout columns' start X bits (fc.x_start) and the axes' images are
// transposed into per-column ascending lists (wires; axes), a column-word chunk at a time. A chunk's lists take at most
// max(the bitmap bytes of x_start's and the images' T columns, one column word's lists <= 64 (N + A) int32): a single
// word is always admitted, so a dense word exceeds the 2 (N + A) TW budget whenever TW < 32 (up to 32x the bitmap at
// TW = 1); memory stays
// linear in N + A either way. The echelon is fully reduced, so
// a column applies exactly the rows whose pivot wire it holds (a pivot lookup per support wire), and the Z4 form is
// accumulated only at the gauge bits the column's axes touch. The verdicts equal the dense form's (kept as the test
// oracle, cpp/testapi/xdem_lean_dense_oracle.hpp), list and order. stats (optional, tests): chunk count and list sizes,
// ADDED to the struct's current values (not reset on entry). LM-3: when the sweep ended sparse, fc's x_start is
// already per-column wire lists (FaultColumns::xs_off / xs_wire); a chunk then reads slices of them (no transpose, no
// N x T bit scan), the chunking and the verdicts unchanged.
struct NondetStats { int chunks = 0; long long x_nnz = 0, img_nnz = 0; };
std::vector<int> nondeterministic_columns(const WriterModel& wm, const FaultColumns& fc, NondetStats* stats = nullptr);

// The exact noiseless expectation <P_t> of readout column t (0 <= t < D + O), from the writer model alone (no bare
// state), for the determinism refusal's text (the oracle prints its framed_expectation value). P_t is the oracle's
// parity Pauli (dem_export.cpp check_deterministic: X/Y/Z on each read wire, i^{#Y}). With U^dag P_t U = c g Q_B
// (g in the skeleton state's stabilizer group, c a unit; derivation: see docs/xdem.md),
// <P_t> = c (-1)^{b_B} e^{-i pi sum_{A_P} n_a / 4} i^{sum_{A_P} n_a b_a} E_u i^{Phi(u)}, Phi the reduced Z4 form of
// nondeterministic_columns; E_u i^Phi is evaluated in closed form (gauss_z4). The value is 0 or +-2^{-h/2}. Throws
// std::logic_error on an internal inconsistency (W not X-free, a non-real value): a lean throw delegates (OD2).
double column_expectation(const WriterModel& wm, int t);

// E_u i^{Phi(u)}, u uniform on F_2^m, Phi(u) = sum_i alpha_i u_i + 2 sum_{i<j} adj_ij u_i u_j (mod 4); adj is a
// symmetric 0-diagonal bit matrix (m rows of words_for(m) words). Exact: zero, or 2^{-h/2} e^{i pi p8 / 4}. Variable
// elimination: an odd alpha_i sums to (1 + i^{alpha_i})/2 i^{-alpha_i [L_i(u)]}; an even one to the indicator
// [L_i(u) = alpha_i / 2], which substitutes one neighbour (a factor 1/2), or to 1 / 0 with no neighbour.
struct GaussZ4 { bool zero = false; int h = 0; int p8 = 0; };
GaussZ4 gauss_z4(std::vector<uint8_t> alpha, std::vector<std::vector<uint64_t>> adj);

// The lean route's line merge (Task 4): the same strict merge as LineAccumulator (SPEC §2, the E1-refined rule
// verbatim: a line is closed by a later event whose swap set S = (tau_c & odd_k) xor (tau_k & odd_c) has
// xor_{a in S} img_a != 0), the same time order, the same p composition P + p - 2 P p, in the same order; only the
// containers differ (keys interned to ids by a hash map, open / last_of as vectors by key id, classes interned once
// per merge, per-axis class lists as vectors, each class's axis sets as bit words for the visibility predicate, and
// the exact zero-image axis prune; see xdem_lean.cpp). to_file / any_rotation / num_lines equal LineAccumulator's
// on the same adds. Holds a pointer to set_axes' vector: it must outlive the last merge (to_file / num_lines).
// Deliberate lean-only divergences (engine-bug inputs the export never produces; a lean throw delegates to the oracle
// route, OD2, so neither reaches the output): merge() throws std::logic_error (i) on an axis id < 0 or >= the
// set_axes count in a key's tau / lt / gt (plan M10; the oracle would index out of range), and (ii) on axis images of
// unequal width (Task 4 review M2; the oracle has no check and would read out of range; the export's images all have
// width D + O).
class LeanLineAccumulator {
  public:
    void set_axes(const std::vector<Axis>& axes);
    void add(LineKey&& k, double p, const TimeKey& t);   // p <= 0 or an empty key: no line, no event (as add)
    bool any_rotation() const;                            // == LineAccumulator::any_rotation on the same adds
    File to_file(int D, int O, const Canon& C, const std::vector<Axis>& axes);   // == LineAccumulator::to_file
    size_t num_lines();

  private:
    struct KeyHash { size_t operator()(const LineKey& k) const noexcept; };
    struct KeyEq {
        bool operator()(const LineKey& a, const LineKey& b) const noexcept {
            return a.side == b.side && a.f == b.f && a.tau == b.tau && a.lt == b.lt && a.gt == b.gt;
        }
    };
    std::unordered_map<LineKey, int, KeyHash, KeyEq> ids_;
    std::vector<const LineKey*> key_;                     // id -> key (unordered_map nodes are address-stable)
    struct Mech { int key; double p; TimeKey t; };
    std::vector<Mech> mechs_;
    const std::vector<Axis>* axes_ = nullptr;
    bool any_rot_ = false, merged_ = false;
    std::vector<int> line_key_;                           // per line (time order of its first member): its key id
    std::vector<double> line_p_;
    void merge();
};

// The lean route's emitter (Task 5): byte-identical to xdem::emit (xdem_file.cpp) for every extended File; integers
// by std::to_chars into one reserved string, p by snprintf("%.17g") as emit (GCC 9 has no floating to_chars).
std::string emit_fast(const File& F);

// The lean route's line-key builder (Task 6, plan amendment): build(sig, side) == the export's dense xdem_key (sig,
// step) (dem_export.cpp) with side = (step > t_c's step), field for field, on every converter key: flips = the ids
// < ncols, XOR-reduced, then every exponent-2 rotation folded (f ^= img_a); x = the ids in [ncols, ncols + nx) by
// group (xdem_x's grouping); tau = the ids >= ncols + nx, in sig order; E^< / E^> = fault_rotations split at Gl, the
// exponent-2 entries dropped, ascending. Only the containers differ: the A-sized exponent scratch and the D+O flip
// words are reused across calls and touched (and cleared) only at the axes / words a key hits; each axis image is
// read only at its nonzero words (listed once, in init). Holds pointers to init's arguments (they must outlive the
// last build). Lean-only divergences (engine-bug inputs; a lean throw delegates, OD2): build throws
// std::logic_error on a negative id or an axis id >= the axes count, init on an axis image whose width is not
// words_for(ncols) (the dense fold would read out of range).
class LeanKeyBuilder {
  public:
    void init(const Header& H, const std::vector<Axis>& axes, const std::vector<std::pair<int, int>>& xid,
              int ncols, int Gl);
    LineKey build(const std::vector<int32_t>& sig, int side);

  private:
    const Header* H_ = nullptr;
    const std::vector<Axis>* axes_ = nullptr;
    const std::vector<std::pair<int, int>>* xid_ = nullptr;
    int ncols_ = 0, nx_ = 0, Gl_ = 0;
    std::vector<std::vector<uint32_t>> img_nz_;          // per axis: the nonzero word indices of its image
    std::vector<uint8_t> elt_, egt_, axis_hit_;           // A-sized, zero outside axes_list_
    std::vector<int32_t> axes_list_;
    std::vector<uint64_t> f_;                             // words_for(ncols), zero outside words_list_
    std::vector<uint8_t> word_hit_;
    std::vector<uint32_t> words_list_;
    std::vector<std::pair<int, uint32_t>> x_;
    bool dirty_ = false;                                  // a build threw mid-way: clear the scratch fully first
    void touch_word(uint32_t w) { if (!word_hit_[w]) { word_hit_[w] = 1; words_list_.push_back(w); } }
    void reset_scratch();
};

// TEST ONLY (cpp/tests/test_xdem_lean.cpp, Task 6): while on, every lean-route line key is also built by the dense
// xdem_key and compared field for field; the counts are read by lean_key_check_counts (compared, mismatched).
// Production never turns it on (atomic flag, off). PROCESS-GLOBAL: while on it affects every concurrent export in the
// process (bytes stay correct; each lean attempt also builds the dense key); for tests only, not while other exports run.
void set_lean_key_check_for_test(bool on);
bool lean_key_check_on();
void lean_key_check_record(bool same);
std::pair<long long, long long> lean_key_check_counts();   // (compared, mismatched) since the last set(true)

// TEST ONLY (cpp/tests/test_xdem_lean.cpp, the OD2 throw fallback): while on, nondeterministic_columns throws
// std::runtime_error, standing in for any exception of a lean piece. Production never calls it (atomic flag, off).
// PROCESS-GLOBAL: while on, every lean attempt in the process throws and delegates to the oracle (bytes stay correct);
// for tests only, not while other exports run.
void set_lean_inject_throw_for_test(bool on);
}  // namespace qeccore::xdem
