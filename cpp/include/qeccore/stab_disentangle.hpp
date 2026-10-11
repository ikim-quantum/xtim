#pragma once
// Disentangling circuit extraction: given a set of n commuting ±1 Pauli generators
// that form a complete stabilizer group (one per qubit), return the gate list that
// maps the stabilized state to |0...0> (the disentangling direction, NOT the prep).
//
// Ops: 0:H  1:S  2:SDG  3:X  4:Y  5:Z  6:CX  7:CZ  (same numbering as RefFile::PrepGate).
// b = -1 for single-qubit gates.
//
// Used by:
//   - canonical_stab_sum.cpp (compute_fsynd): disentangle each ray by the CURRENT frame
//     U.Zrow to read its syndrome as a computational-basis offset.
//   - ref_io.cpp (synthesize_prep): the existing prep synthesis now calls this helper and
//     inverts (same observable result, one definition instead of two).
#include <vector>
#include <cstdint>
#include "qeccore/pauli.hpp"
#include "qeccore/stab_affine.hpp"

namespace qeccore {

struct DisentangleGate {
    int op;   // 0:H 1:S 2:SDG 3:X 4:Y 5:Z 6:CX 7:CZ
    int a;    // qubit / control
    int b;    // target for CX/CZ; -1 for single-qubit gates
};

// Return the gate list G such that G |stabilized_state> = e^{i theta}|0...0>.
// `rows` must be n Pauli stabilizers forming a complete independent group (one per qubit).
// They are consumed (modified) by the reduction. Throws std::runtime_error on bad input.
// O(n^2) gates, O(n^2) work.
//
// After applying G to the stabilized state, the b offset of the disentangled state encodes
// COMBINATIONS of original syndromes (not individual ones) due to row-multiply steps in
// Gaussian elimination. Use `recover_syndromes()` below to reconstruct individual syndromes.
//
// perm_out  (if non-null): perm[a] = the qubit index that original generator a maps to.
// trans_out (if non-null): trans[a] is a word-packed bitmask (ceil(n/64) uint64_t words)
//           of original generator indices combined in the row assigned to generator a's qubit.
//           Bit j of trans[a] is set iff original generator j contributes to perm[a]'s bit.
//           Used by recover_syndromes() to undo GF(2) mixing from row multiplies.
std::vector<DisentangleGate> disentangle_from_generators(
    std::vector<Pauli> rows,
    std::vector<int>* perm_out = nullptr,
    std::vector<std::vector<uint64_t>>* trans_out = nullptr);

// Recover per-generator syndrome bits from a disentangled ray.
// dis_ray  : AffineState after all DisentangleGates have been applied.
// perm     : from disentangle_from_generators (perm[a] = qubit for generator a).
// trans    : from disentangle_from_generators (GF(2) mixing matrix, n rows of ceil(n/64) words).
// fs       : output, length n; fs[a] = 0 or 1 (syndrome of original generator a).
// O(n^2/64) work. Fills fs via forward substitution in reduction order.
void recover_syndromes(int n, const AffineState& dis_ray,
                       const std::vector<int>& perm,
                       const std::vector<std::vector<uint64_t>>& trans,
                       std::vector<uint8_t>& fs);

// Apply a single DisentangleGate to an AffineState (using its public gate API).
void apply_disentangle_gate(AffineState& s, const DisentangleGate& g);

}  // namespace qeccore
