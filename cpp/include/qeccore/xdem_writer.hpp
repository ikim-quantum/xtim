// xdem_writer.hpp — the xdem writer (SPEC §2). Built from the DEM export's own preamble (dem_preamble.hpp): xtim's
// deferred circuit (every terminal read on its own wire, coherentized feedback already gates, residual feedback folded
// into the record sets).
//   * the writer model (build_writer_model): the Clifford skeleton S, the location groups, the raw axes (signed end
//     Paulis + readout images), their raw kappa and sign bits b, and the diagonality check;
//   * the fault columns (build_fault_columns) and the strict tau columns (build_tau_columns), both from one backward
//     sweep of S;
//   * t_c, the late-group count and the canonical-form check, on the parsed stream;
//   * canonical gauge and hidden-syndrome coordinates and the axis signs bhat (canonical_coordinates);
//   * a fault's rotations E, split at t_c into E^< (groups g < Gl) and E^> (g >= Gl), in the xdem convention
//     (fault_rotations), and the folding of an exponent-2 entry into the flips (fold_field);
//   * the strict merge of the lines (LineAccumulator).
//
// Writer model construction (re-derived from an internal Python reference model of the
// writer):
//   * Groups: every deferred non-Clifford gate instance (one per target group, deferred stream order == user circuit
//     order), named by the USER gate: the parser writes the adjoint of a diagonal gate as the gate followed by its
//     Clifford correction (adjacent, same targets, the suffix tagged Instr::Origin::DagSuffix), so a group named for
//     an adjoint carries that Clifford right after its location in the skeleton (the deferred stream's own copy,
//     wherever eliminate_hadamards placed it, stays in the skeleton). Group qubits: the deferred gate's targets in
//     order (two- and three-qubit gates: sorted by the parser). A CH(c,t) is two groups on {t} (Group::gate T_DAG
//     then T, which only selects the residue table; the file names no gate), both at the CH's stream index:
//     CH = (W T W^dag) CZ(c,t) (W T_DAG W^dag) exactly (no global phase), W = S H on t (W Z W^dag = Y), i.e. the
//     skeleton gets S_DAG H LOC H S, CZ, S_DAG H LOC H S on t (the bare state's fold, two pi/8 rotations about Y_t
//     around a CZ). Whether its axes commute with the rest is the diag check, as for any group.
//   * Skeleton S: the deferred unitary with every location = identity (deferred wires start in |0>; a measurement is
//     the deferral's copy/terminal read; coherentized feedback is in the gate stream). A read's `invert` (user `!`,
//     or eliminate_hadamards' sign stamp) is a final Pauli: it conjugates every signed axis and the final state
//     together, so no header quantity's physics depends on it; it is not applied.
//   * Q_{g,v} = U_after Z(v) U_after^dag (exact phase, forward pass); unsigned axis = Q with phase #Y (Hermitian +),
//     ids in first-appearance order over (g, v = 1 .. 2^m - 1); identity -> I.
//   * img_a: readout qubits t (detectors 0..D-1, observables D..D+O-1) whose raw record set's read product
//     anticommutes with Q_a.
//   * raw kappa: the stabilizer generators of S's final state taken as U Z_w U^dag (w = every deferred wire), so
//     k_a[w] = X-part of U^dag Q_a U = U_before^dag Z(v) U_before at w (a backward pass); stored on the information
//     set J = pivots (highest-bit RREF) of the k_a (r = |J|).
//   * b: on ker kappa, sigma(c) = sign of <prod_{a in c} Q_a> on S's final state = phase of prod U^dag Q_a U on |0>;
//     b = the unique RREF solution (pivot = highest axis id; b = 0 off the pivots). Not written to the file (the
//     canonical bhat is); kept as the theory check the canonical coordinates are tested against.
//   * diag check: a location's Z forwarded to a strictly later location with X/Y on that location's qubits.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/dem_preamble.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/xdem_file.hpp"

namespace qeccore::xdem {

// Every writer refusal (export_dem returns its message as the export's error; Python raises xtim.XtimDemError). The
// message starts "xdem writer: " and names the cause.
struct WriterError : std::runtime_error {
    explicit WriterError(const std::string& m) : std::runtime_error(m) {}
};

struct Group {
    std::string gate;                            // the user gate's name
    std::vector<std::pair<int, int>> v_axis;     // index v-1 -> (axis, sign) or (-1, 0) = I
};

struct RawAxis {
    Bits img;                                    // readout image (D + O bits)
    std::vector<uint64_t> k;                     // raw kappa on the information set J (r bits)
    int b = 0;                                   // the sign bit on ker kappa (see above)
};

struct Header {
    int D = 0, O = 0, r = 0;
    std::vector<Group> groups;
    std::vector<RawAxis> axes;
};

// The Clifford skeleton S and its location map.
struct Skeleton {
    enum Kind : uint8_t { H, S, SDG, X, Y, Z, CX, CZ, LOC };
    struct Op { uint8_t kind; int a; int b; };   // LOC: a = group id, b = -1
    int N = 0;                                   // deferred wires
    std::vector<Op> ops;
    std::vector<int> op_of_stream;               // deferred stream index i -> #ops emitted before instruction i
                                                 // (size stream + 1)
    std::vector<int> group_op;                   // group -> its LOC op index
    std::vector<int> group_stream;               // group -> its deferred stream index
    std::vector<std::vector<int>> group_qubits;  // group -> deferred wires (gate target order; bit i of v / x)
    std::vector<std::pair<int, int>> reads;      // record order: (0 X / 1 Y / 2 Z, wire)
    std::vector<std::vector<int>> targets;       // D + O raw record sets (sorted, parity-reduced)
    std::vector<int> gv_offset;                  // group -> index of (g, v=1) in gv_q
    std::vector<Pauli> gv_q;                     // signed end Pauli Q_{g,v}
    std::vector<Pauli> axis_q;                   // unsigned end Pauli per axis
    std::vector<Pauli> axis_p0;  // per axis: U^dag axis_q[a] U (unsigned), the lean determinism check's input
};

// Sort v[from, end) and keep each id that occurs an odd number of times there (mod-2 reduction; v[0, from) untouched).
void parity_reduce_sorted(std::vector<int32_t>& v, size_t from = 0);

// The conjugation of one skeleton op (LOC = identity): forward M <- G M G^dag, backward M <- G^dag M G.
void conj_forward(Pauli& M, const Skeleton::Op& o);
void conj_backward(Pauli& M, const Skeleton::Op& o);

struct WriterModel {
    Header header;
    std::unique_ptr<Skeleton> sk;
    std::vector<std::string> diag_violations;    // non-empty: the residue does not stay diagonal (b left 0)
    // Some skeleton H acts on a wire where an earlier location's forwarded Z has X or Z support. False implies the
    // default diagonal-class table of this deferred circuit accepts every atom (conjugate_layer rejects only an H on a
    // wire where the residue is non-Pauli, or a CH; a residue is a product of these forwarded Z's): LEAN-EXPORT-PLAN
    // F2 / Task 2. A CH's expansion puts an H right after its first rotation's location on that location's wire, so
    // a circuit with a CH always sets it (the lean route delegates CH circuits to the oracle route).
    bool h_meets_location = false;
};

// Throws WriterError (IF blocks, an internal pipeline mismatch, a theory check). A CH is expanded (see above).
WriterModel build_writer_model(const DemPreamble& pre);

// The gate's exact residue D_x = prod S^e(Z(v)) for 0 < x < 2^arity: the minimal-weight Z4 phase-polynomial
// solution, ties broken as the reference (`residue_terms`, first in lexicographic order of e). (v, e) pairs, v
// ascending, e in 1..3. Computed once per (gate, x) and cached (static table). Throws std::invalid_argument for an
// unknown gate name or x out of range.
const std::vector<std::pair<int, int>>& residue_terms(const std::string& gate, uint32_t x);

// The net number of pi/8 rotations of `gate` about Z(v) (the gate = prod_v exp(-i n_v pi/8 Z(v)) up to a phase):
// n_v = -2^{1-m} sum_z PHASE(z) (-1)^{v.z}. T: 1; T_DAG: -1; CS: 1, 1, -1; CCZ: (-1)^{|v|+1}.
int gate_rotation_count(const std::string& gate, uint32_t v);

// ── Fault columns: each channel alternative is a Pauli fault propagated through S. ─────────────────────────────────
// Its signature is linear in the fault, so it is read off ONE backward sweep (stim's DEM method): every column is a
// Pauli conjugated from its activation point back through S (commutation only; signs are irrelevant to flips):
//   * column t < ncols (a detector / observable / DECISION column): the product of its raw record set's read Paulis,
//     activated at each read's Measure instruction;
//   * column ncols + l (location qubit l = (group g, tuple position i), groups in order): Z on that wire, activated at
//     g's location — a fault before g anticommutes with it iff its X-part just before g has X/Y there (x bit i of g).
// A fault P_q at noise instruction k anticommutes with column c iff P has X where c has Z, or Z where c has X.
// The extended signature (sorted ids: flipped columns, then x ids >= ncols) is the converter's opaque key.
//
// LM-3: the sweep's storage (pure performance: every layout gives identical noise-slot lists, x_start contents and tau
// columns; a CW = ceil(C/64)-word bit row costs CW words per gate).
//   * Sparse: per wire and Pauli part a row, held as the ascending column ids while it has at most
//     max(kSweepRowMinList, CW / kSweepRowListDiv) entries (a gate costs the lists it touches, so at most about a
//     dense row's word cost), and as its own CW-word bit row above that (fix round 1: a long row costs CW words per gate,
//     not its length; it returns to a list at a quarter of the bound). A list's capacity is kept within 2 x its size
//     + 16 entries, so the store's bytes are its sizes', not its history's.
//   * Dense: the wires x columns bit planes (16 N CW bytes).
//   * Auto starts sparse and moves to the dense planes, once and for good, as soon as the planes' bytes are at most
//     kSweepDenseRatio times the sparse store's actual bytes (list capacities, row bits, headers); the check runs after
//     every deferred instruction. The rows' fixed bytes (a list header and a slot id, 28 bytes) alone pass the test
//     when CW <= 28, so Auto is dense from the start there (the margin grows with the stored entries per wire); this
//     is a byte comparison, not a column-count cutoff.
// The dense planes it replaced (verbatim 2e89d05) are the test oracle (cpp/testapi/xdem_sweep_dense_oracle.hpp).
// XTIM_SWEEP_LAYOUT=sparse|dense (a TEST lever, read once per process) forces an Auto request to one side; an explicit
// layout argument wins over it.
enum class SweepLayout : int { Auto = 0, Sparse = 1, Dense = 2 };
inline constexpr long kSweepDenseRatio = 8;
inline constexpr long kSweepRowListDiv = 4;
inline constexpr long kSweepRowMinList = 16;
// The sweep's storage history (tests): final_layout 1 sparse / 2 dense; dense_from = the number of deferred
// instructions swept (from the end) before the dense planes took over (0: dense from the start; -1: never); peak_nnz
// the lists' largest total entry count, peak_bytes the sparse store's largest byte count (capacities, bit rows,
// headers; the planes' 16 N CW when dense from the start), peak_bit_rows the most rows held as bit rows at once, and
// bit_rows_made how many times a list became a bit row (all while sparse); work the gates' total work (list entries
// merged or moved, plus bit-row and plane words touched; the dense planes cost CW per row a gate touches).
struct SweepStats {
    int final_layout = 0;
    long long dense_from = -1, peak_nnz = 0, peak_bytes = 0, peak_bit_rows = 0, bit_rows_made = 0, work = 0;
};
struct FaultColumns {
    int ncols = 0;                                   // D + O (+ DECISION columns)
    std::vector<std::pair<int, int>> xid;            // x id - ncols -> (group, bit)
    // per deferred noise ordinal (Noise instructions in stream order) -> per qubit position of the instruction:
    // [0] the columns an X fault there flips, [1] the columns a Z fault flips (sorted ids)
    std::vector<std::vector<std::array<std::vector<int32_t>, 2>>> at;
    // The extended signature of one alternative (factors: (qubit position, Pauli) as the DEM walk enumerates them).
    void signature(int noise_ordinal, const std::vector<std::pair<int, PauliBasis>>& factors,
                   std::vector<int32_t>& sig) const;
    // X bits of the readout columns (< ncols) at the start of the stream; read only by nondeterministic_columns (the
    // lean route); export_dem releases it right after that check (release_x_start). LM-3: one of two forms, by the
    // layout the sweep ended on. Dense: x_start, wire-major, cw words per wire (the readout words; xs_off empty).
    // Sparse: xs_off (ncols + 1 offsets) / xs_wire, per readout column the wires holding its X bit, ascending
    // (x_start empty, cw = 0).
    std::vector<uint64_t> x_start; size_t cw = 0;
    std::vector<size_t> xs_off; std::vector<int32_t> xs_wire;
    bool x_sparse() const { return !xs_off.empty(); }
    void release_x_start() {
        std::vector<uint64_t>().swap(x_start);
        std::vector<size_t>().swap(xs_off);
        std::vector<int32_t>().swap(xs_wire);
    }
};

// col_recs: each column's raw record set (indices into sk.reads; parity, duplicates cancel). layout / stats: LM-3
// (above); production passes neither.
FaultColumns build_fault_columns(const Skeleton& sk, const Circuit& deferred,
                                 const std::vector<std::vector<int>>& col_recs, SweepLayout layout = SweepLayout::Auto,
                                 SweepStats* stats = nullptr);

// ── t_c and the canonical single-stage form (docs/xdem.md "Canonical form"; export_dem's opt-in require_canonical) ─
// Checked on the PARSED stream (flat, REPEAT unrolled, desugared; parse_stim_circuit(text, /*track_source=*/true)):
// the deferred stream moves every read to the end, so it no longer says when a record was measured. t_c = the point
// right after the user instruction that measures the last detector record (the start of the circuit when no detector
// reads a record). Canonical iff
//   (C1') every observable reads at least one record measured by a user instruction after t_c (records before t_c
//         are a classical frame, XOR; an observable with no record violates it), and
//   (C2') no noise acts between t_c and the last non-Clifford gate after t_c: a noise channel with a nonzero
//         probability, or a measurement with a nonzero readout-flip argument (MX(p), MPP(p), ...). Noise after that
//         gate (or anywhere after t_c when no non-Clifford gate follows it) is a classical readout flip and allowed.
//         Heralded ops are refused by the parser. Gates of any kind, feedback and noiseless measurements are allowed.
// Record sets are the DECLARED ones, parity-reduced (a record listed twice is not read). Returns "" iff canonical,
// else the reason naming the first violation (observable / detector, record and source lines) and how many more
// there are. Throws std::logic_error when p was parsed without track_source.
std::string canonical_violation(const ParsedStim& p);

// t_c on the PARSED stream (track_source), the one definition shared by canonical_violation and late_group_count:
// the user instruction (stream_step) that measures the last detector record (the highest record index any
// detector's parity-reduced record set reads), that record and its detector; all -1 when no detector reads a record
// (t_c = the start of the circuit). Throws std::logic_error when p was parsed without track_source.
struct TcPoint {
    int step = -1, record = -1, detector = -1;
};
TcPoint last_detector_point(const ParsedStim& p);

// Gl: the number of non-Clifford gate instances (one per target group, the writer's location groups in circuit
// order; a CH counts 2, its two rotations) whose user instruction precedes t_c (stream_step < t_c's step). The
// writer's groups are these instances in the same order (build_writer_model matches them one to one, a CH to its
// two), so the late groups are g >= Gl. The deferred stream
// cannot place t_c itself: deferral moves every read to its end (an abandoned wire's read emits no gate where it
// was measured), so t_c is located on the parsed stream and carried to the groups by their order. Throws
// std::logic_error when p was parsed without track_source.
int late_group_count(const ParsedStim& p);

// The canonical coordinates of SPEC §2 from the writer model (raw kappa, axis Paulis, skeleton).
//   * gauge: walk the axes in file order; axis a is a pivot iff its raw kappa is not in the span of the earlier
//     ones; the t-th pivot owns gauge qubit g_t alone; a non-pivot axis's kappa = the unit vectors of the pivots its
//     raw kappa depends on.
//   * hidden-syndrome: each non-pivot axis a closes c_a = {a} + its pivots (kappa_{c_a} = 0, so Q_{c_a} is +- a
//     stabilizer of the skeleton's final state). Walking the non-pivot axes in order, c_a is either in K_I + the
//     span of the earlier generators (sigma_a = its coordinates in those generators) or the next generator
//     (sigma_a = {h_t}); K_I = the c with Q_c = +-I. m = the number of generators = dim K - dim K_I.
//   * bhat: 0 on pivots; beta(c_a) on non-pivots, beta(c) = 1 iff <Q_c> = -1 on the skeleton's final state.
struct Canon {
    int r = 0, m = 0;
    std::vector<char> pivot;                           // per axis
    std::vector<std::vector<uint64_t>> kappa;          // per axis, r bits: g_t of the t-th pivot; non-pivot: its pivots
    std::vector<std::vector<int32_t>> hidden;          // per axis: sigma (hidden-syndrome ids)
    std::vector<uint8_t> bhat;                         // per axis: 0 on pivots; beta(c_a) on non-pivots
};
Canon canonical_coordinates(const WriterModel& wm);

// beta of an end Pauli P (a product of axis Paulis, exact phase): 1 iff <P> = -1 on the skeleton's final state
// (the phase of U^dag P U on |0>, U = the skeleton). Throws WriterError, naming `what` (the axes), when P is not
// +- a stabilizer of that state.
int stabilizer_sign(const Skeleton& sk, const Pauli& P, const std::string& what = "");

// Strict tau columns: each axis's end Pauli swept back through the whole skeleton (the fault columns' backward sweep,
// every column activated at the end); for each deferred noise ordinal and qubit position, the axes an X / a Z fault
// there anticommutes with (sorted axis ids).
struct TauColumns { std::vector<std::vector<std::array<std::vector<int32_t>, 2>>> at; };
TauColumns build_tau_columns(const Skeleton& sk, const Circuit& deferred, SweepLayout layout = SweepLayout::Auto,
                             SweepStats* stats = nullptr);

// A fault's rotations: E^< (groups g < Gl) and E^> (g >= Gl), per axis mod 4, in the xdem convention (SPEC §1.1:
// E = -E' where residue_terms is in the owner's S^e = i^{e[Z=-1]} form). x: (group, x-pattern), the groups at or
// after the fault; Elt / Egt are A-sized and accumulated into.
void fault_rotations(const Header& H, const std::vector<std::pair<int, uint32_t>>& x, int Gl,
                     std::vector<uint8_t>& Elt, std::vector<uint8_t>& Egt);
// Fold an exponent-2 entry of one field into the flips (SPEC §2): f ^= img_a, the entry dropped.
void fold_field(const std::vector<Axis>& axes, std::vector<uint8_t>& E, Bits& f);

// ── Lines (SPEC §2: strict merge, time order, t_c) ──────────────────────────────────────────────────────────────
// A mechanism's time: the user step of its noise site (a readout flip: the step of its Measure), then the deferred
// noise ordinal (readout flips: num_noise + record), then the channel converter's mechanism index, counted across
// the instruction's per-qubit flushes (so every mechanism has its own TimeKey).
struct TimeKey {
    int step = 0, ordinal = 0, mech = 0;
    bool operator<(const TimeKey& o) const;
};
// The line key of a mechanism: (f, tau, E^<, E^>), each field folded, and the side of t_c.
struct LineKey {
    int side = 0;                                       // 0: before t_c (step <= t_c's step), 1: after
    std::vector<int32_t> f, tau;                        // readout columns; axes (ascending)
    std::vector<std::pair<int32_t, uint8_t>> lt, gt;    // E^<, E^>: (axis, 1 | 3), ascending, after folding
    bool empty() const { return f.empty() && tau.empty() && lt.empty() && gt.empty(); }
    bool operator<(const LineKey& o) const;             // lexicographic over (side, f, tau, lt, gt)
};
// The swap set of two keys (axes, ascending): (tau_a & odd(E_b)) xor (tau_b & odd(E_a)), odd = E^< + E^> per axis.
// Swapping two adjacent lines changes their product only by prod_{a in S} Zhat_a (each sign flip of a rotation is a
// pi/2 rotation, a Pauli); its gauge / hidden-syndrome Z-parts are invisible (as for the E = 2 folding), so the swap
// is exact iff the readout image xor_{a in S} img_a is zero (plusmodel/REPORT.md §3(c)).
std::vector<int32_t> swap_set(const LineKey& a, const LineKey& b);
bool swap_visible(const LineKey& a, const LineKey& b, const std::vector<Axis>& axes);   // img of the swap set != 0
// The axes with an odd total exponent E^< + E^> of a key (ascending): the odd(E) of swap_set.
std::vector<int32_t> line_odd_axes(const LineKey& k);
// The default export's XOR composition of two independent flip probabilities, P + p - 2 P p. ONE out-of-line
// definition (xdem_writer.cpp, [[gnu::noinline]]) called by both line merges (LineAccumulator and the lean
// LeanLineAccumulator), so the two run the same machine code and agree bitwise whatever FP contraction or inlining
// a compiler applies at either call site (Task 4 review M1).
double xor_compose(double P, double p);
// The extended File of a line list, shared by both line merges' to_file (LineAccumulator and the lean
// LeanLineAccumulator; final review M3): file_begin, then file_add_line per line in time order. F.tc counts the side-0
// lines; a side-0 line after a side-1 line is a WriterError.
File file_begin(int D, int O, const Canon& C, const std::vector<Axis>& axes);
void file_add_line(File& F, const LineKey& k, double p);

// The strict merge (SPEC §2 with the controller's erratum of 2026-11-16, fix round 1 of Task 4). Mechanisms are
// taken in time order (TimeKey). A mechanism joins the open line of its key, else it opens a new line with that key.
// A line is open until a later event (a line opened or joined) whose swap with it is visible (swap_visible); then it
// is closed for good, and a later mechanism with the same key opens a new line. Joining moves a fault back in time to
// its line's position past the lines opened or joined since the line's latest member; each of those swaps is checked
// on its own (lines fire independently), so the merge preserves the law and the operators modulo invisible factors.
// p combined as p1 (1 - p2) + p2 (1 - p1); a line sits at its first member. Cost: an event evaluates only the open
// lines whose swap set with it is nonempty (per-axis indexes of tau and odd(E)) and whose key has a member still to
// come: a line no later mechanism can join is never consulted again (closing it could change nothing), so it leaves
// the indexes at its key's last member (fix round 2).
class LineAccumulator {
  public:
    // the canonical axes (their readout images); required before the first merge when any key has tau or E
    void set_axes(const std::vector<Axis>& axes) { axes_ = axes; merged_ = false; }
    // flush: the converter call the mechanism came from (one per channel instance and qubit / pair, one per readout
    // flip; -1: none); p <= 0 or an empty key: no line, no event
    void add(const LineKey& k, double p, const TimeKey& t, int flush = -1);
    // A channel alternative or readout flip of the census (the test API's atom map): its line is the line a
    // mechanism of its own flush with its own key joined (unique: one flush's equal keys share a line), -1 if none.
    void note_atom(const LineKey& k, int flush) { atoms_.push_back({k, flush}); }
    // The lines in time order; tc = the number of side-0 lines (they precede the side-1 lines).
    File to_file(int D, int O, const Canon& C, const std::vector<Axis>& axes) const;
    bool any_rotation() const;                          // some line has a nonempty E^< or E^>
    std::vector<int> atom_lines() const;                // per note_atom call: its line, -1: none
    size_t num_lines() const { merge(); return lines_.size(); }

  private:
    struct Mech { LineKey k; double p; TimeKey t; int flush; };
    struct Line { const LineKey* k; double p; };
    std::vector<Mech> mechs_;
    std::vector<std::pair<LineKey, int>> atoms_;
    std::vector<Axis> axes_;
    mutable bool merged_ = false;
    mutable std::vector<Line> lines_;                   // in time order of their first member
    mutable std::vector<int> mech_line_;                // per sorted mechanism: its line
    mutable std::vector<size_t> order_;                 // mechs_ indices in time order
    void merge() const;
};

}  // namespace qeccore::xdem
