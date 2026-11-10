#pragma once
// record_law_oracle — the EXACT joint record law of one shot (design note "Consolidated twirl
// sampler", §2.2 Theorem 1 + Lemma 2, §2.5 term-indexed residual, §2.7 frames). Stage S1: TEST /
// ORACLE infrastructure only. No production path calls anything here; the twirl engine's streams,
// plans and counters are untouched by this translation unit.
//
// What it computes. The circuit is compiled the way the TwirlSampler family compiles it (parse ->
// normalize with the twirl-sampler preset -> deferred circuit -> bare state |B> = U|phi> -> terminal
// reads). The deferred circuit is then read as a sequence of Clifford gates and T-TYPE pi/8 FACTORS
// exp(-i(pi/8) A) (T, and the Z-string expansions of CS / CCZ / CH): factor j is TERM j, with
// end-of-circuit signed axis B_j. A Pauli fault P at time t, pushed to the end, is
//     R_f = P_end * prod_{j in S_f} rot(B_j),   rot(B) := exp(i(pi/4) B),
// where S_f = the terms whose factor P anticommutes with at the moment it crosses it (every T-type
// factor sends an anticommuting P to P*rot(A), never rot(A)^-1 — no sign corrections), and P_end is
// P conjugated by the Clifford gates only. A whole shot (frame first, then the faults in stream
// order) is ONE forward pass: R = P * C0, C0 = prod_{j in S} rot(B_j) (the terms commute in class —
// asserted at compile). S is the TERM-INDEXED residual (the plan content), P its Pauli part.
//
// The law (Theorem 1). Record operators M_c (DETECTOR / OBSERVABLE / DECISION columns, each a
// product of terminal reads) are pulled back record by record, SUPPORT-SPARSELY:
//     Q_r = C0^dag B_r C0 = B_r * prod_{j in S, {B_r,B_j}=0} (i B_j)       (one read, |S| terms max)
// and Q_c = prod_{r in c} Q_r. In the bare frame (U^dag Q U) the syndrome phi(Q) is the X-part on
// the fixed (certified) rows (the builder's frame state is sum_x alpha_x |emb x>: eps = 0, checked).
// Pivots / A / K_c / Lambda_j / B / beta / P0 follow the theorem
// verbatim; P0 is evaluated on the free register from the bare amplitudes. Lemma 2 adds the Pauli
// part: t_c = <P, M_c>. No per-alternative tableau and no dense state are ever formed.
//
// Conventions. y_c = the RAW parity of the deferred reads' outcomes (bit 0 <-> eigenvalue +1).
// The engine's RECORDED column is y_c ^ (XOR of the records' rec_invert) ^ (readout-flip coins);
// detector/observable columns are EMITTED relative to the noiseless reference record
// (noiseless_reference_records), decisions raw. Program::emission_offset carries both constants.
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

namespace qeccore {
namespace record_oracle {

// A record column: kind 0 = DETECTOR, 1 = OBSERVABLE, 2 = DECISION; `recs` = the measurement
// records it XORs (duplicates cancelled), `index` = the detector / observable / decision index.
struct Column {
    int kind = 0;
    int index = 0;
    std::vector<int> recs;
};

// One elementary op of the deferred circuit: a Clifford gate or a T-type pi/8 factor (a term).
struct Op {
    uint8_t kind = 0;          // 0 Clifford, 1 term
    GateKind g = GateKind::X;  // Clifford: X Y Z S SDG H CX CZ
    int a = -1, b = -1;        // Clifford targets
    int term = -1;             // term: index into Program::terms / local_axes
};

// One noise site of the deferred circuit: a categorical law over Pauli outcomes on `qubits`
// (codes 0 I, 1 X, 2 Y, 3 Z per qubit), identity outcome included; zero-probability outcomes are
// dropped. `op_pos` = the fault's insertion point (before ops[op_pos]).
struct NoiseSite {
    int stream_index = -1;
    int op_pos = 0;
    std::vector<int> qubits;
    std::vector<std::vector<uint8_t>> outcomes;
    std::vector<double> probs;
};

// A fault (or a frame component): a Pauli on the deferred wires inserted before ops[op_pos].
struct Fault {
    int op_pos = 0;
    Pauli P;
};

// The term-indexed residual R = P * prod_{j in S} rot(B_j).
struct TermResidual {
    Pauli P;
    std::vector<uint8_t> S;   // size = #terms
};

// Theorem 1 for one plan (one C0 = rot(S)), P-free (Lemma 2: add t_c afterwards).
//   y_p = u_p                                   (p = pivots[i])
//   y_c = A_c . u  ^  B_c . z  ^  beta_c        (c not a pivot)
// u uniform on F2^rho, z ~ p0 (index z = sum_j z_j 2^j), independent.
struct PlanLaw {
    int m = 0, rho = 0, kappa = 0;
    std::vector<int> pivots;                    // column indices (rho), increasing
    std::vector<int> born;                      // column c_j per Born direction (kappa)
    std::vector<std::vector<uint64_t>> A;       // per column: bitmask over pivot slots
    std::vector<std::vector<uint64_t>> B;       // per column: bitmask over Born slots
    std::vector<uint8_t> beta;                  // per column
    std::vector<double> p0;                     // 2^kappa
    std::vector<uint8_t> cls;                   // per column marginal: 0 det, 1 coin, 2 born
    // Diagnostic (the A4 case): per non-basis Born column, the sign the ORDERED product of its
    // Born factors' Hermitian forms picks up (s_c of Theorem 1; folded into beta).
    std::vector<uint8_t> s_sign;
    // S2b (L-3): the Born directions Lambda_j (free part of K_{c_j} at K's exact phase), as k-bit
    // masks a / b over the free coordinates and the phase (eps = 0 frames: the effective phase).
    std::vector<uint64_t> lam_a, lam_b;
    std::vector<int> lam_ph;
};

struct Program {
    int n = 0;                                  // deferred wires
    int k = 0;                                  // free (magic) rows of the bare state
    Circuit deferred;
    FramedSuperposition bare{0};
    std::vector<Op> ops;
    std::vector<Pauli> local_axes;              // term j: signed axis A_j at its factor
    std::vector<Pauli> terms;                   // term j: end-of-circuit signed axis B_j
    std::vector<int> term_stream_index;         // term j: deferred-stream index of its gate
    std::vector<std::vector<int>> wire_terms;   // wire w: terms with support on w
    std::vector<std::pair<int, int>> reads;     // per record: (basis 0 X/1 Y/2 Z, deferred wire)
    std::vector<uint8_t> rec_invert;            // per record (user '!' ^ eliminate_hadamards stamp)
    std::vector<double> rec_flip;               // per record readout-flip probability (M(p))
    std::vector<Column> columns;
    // Filled by ensure_reference (on demand: the noiseless reference is a compile-time cost of its
    // own — about 12 s on the f3 factory — and only the emitted-column convention needs it).
    bool reference_ready = false;
    std::vector<uint8_t> reference;             // per record: noiseless reference RECORDED bit
    std::vector<uint8_t> emission_offset;       // per column: invert parity ^ (det/obs: ref parity)
    std::vector<NoiseSite> sites;
    std::vector<int> input_qubits;              // INPUT_QUBITS port (deferred wire = user qubit)
    std::vector<uint64_t> fixed_rows;           // n-bit mask of the certified (non-free) rows
    // Free-register amplitudes: |phi> = sum_x alpha_x |x>, x bit d <-> free row free[d]
    // (support only; chi entries).
    std::vector<std::pair<uint64_t, std::complex<double>>> alpha;
};

// Compile `text` (refuses IF blocks, out-of-class circuits and k > 64 by name).
Program compile(const std::string& text);

// The engine's noiseless reference (noiseless_reference_records) and the per-column emission
// offsets; idempotent.
void ensure_reference(Program& prog);

// The shot residual of `faults` (any order; sorted by op_pos, ties keep the given order) with an
// optional frame (a Pauli on the deferred wires applied at t = 0, before every fault). One pass.
TermResidual residual(const Program& prog, const std::vector<Fault>& faults, const Pauli* frame);

// Theorem 1 for C0 = rot(S).
PlanLaw plan_law(const Program& prog, const std::vector<uint8_t>& S);

// Lemma 2: t_c = <P, M_c> per column.
std::vector<uint8_t> relabel(const Program& prog, const Pauli& P);

// Section 2.7 per-term relabel of a Pauli-image frame component P' on a shot with term set S:
// t_c = <P', M_c> ^ XOR_{j in S} <P', B_j><M_c, B_j>   (the frame's share of t; XOR <P, M_c> for P).
std::vector<uint8_t> frame_relabel(const Program& prog, const Pauli& Pprime,
                                   const std::vector<uint8_t>& S);

// rot(S) up to phase, when it is a Pauli N (S in ker Phi): returns true and sets N; false otherwise.
// A frame component's anchor image is P' = P_f * N for its residual (P_f, S_f).
bool term_set_pauli(const Program& prog, const std::vector<uint8_t>& S, Pauli& N);

// The exact law of the RAW column vector y (bit c of the key = y_c), t folded in. Requires m <= 63
// and rho + kappa <= 30.
std::vector<std::pair<uint64_t, double>> enumerate_law(const PlanLaw& law,
                                                       const std::vector<uint8_t>& t);

}  // namespace record_oracle
}  // namespace qeccore
