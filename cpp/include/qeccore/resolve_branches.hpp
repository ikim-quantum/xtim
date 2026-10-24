// Task 5 (decoder-feedback): IR-level IF-branch resolver.
//
// `resolve_branches` converts a Circuit that may contain IfBegin/IfEnd markers
// into a concrete, branch-free Circuit by evaluating each IF condition against
// a supplied bit_values array and either inlining the span body (condition met)
// or dropping it entirely (condition not met).  Nesting is handled via a
// keep/drop stack.  A circuit with no IfBegin instructions returns a
// byte-identical copy.
//
// Supporting helpers:
//   if_driving_bits          — (port_idx, bit_idx) pairs that drive any IF condition
//   if_driving_global_offsets — global bit offsets of those bits (for partition grouping)
//   bits_total_width          — total width of all Bits input ports
//   emit_resolved_text        — re-emit a resolved Circuit as parseable Stim-superset text
#pragma once
#include "qeccore/circuit_ir.hpp"
#include "qeccore/stim_parse.hpp"
#include <string>
#include <utility>
#include <vector>

namespace qeccore {

// Resolve all IF spans in `c` using `bit_values` (indexed by global decision-bit offset).
// Global offset of a port = sum of widths of prior Bits ports in c.inputs order.
// A span [IfBegin..matching IfEnd] is inlined (body kept, markers removed) when
//   bit_values[offset(cond_port) + cond_bit] == cond_value,
// otherwise the entire span is dropped.  Nested IFs are resolved recursively via
// a stack.  The returned Circuit:
//   - has stream with no IfBegin / IfEnd instructions
//   - has inputs = Qubits ports only (Bits ports consumed)
//   - has c.n unchanged
// A circuit with no IfBegin instructions returns a byte-identical copy.
Circuit resolve_branches(const Circuit& c, const std::vector<uint8_t>& bit_values);

// Return the list of (port_idx, bit_idx) pairs for decision bits that appear in
// at least one IF condition in `c`.  port_idx is the index in c.inputs; bit_idx
// is the bit within that port.  Duplicate appearances are deduplicated.
std::vector<std::pair<int,int>> if_driving_bits(const Circuit& c);

// Return the global bit offsets for IF-driving bits in `c`.
// Global offset = sum of widths of prior Bits ports (in c.inputs order) + bit_idx.
// Duplicate appearances are deduplicated; order matches if_driving_bits().
std::vector<int> if_driving_global_offsets(const Circuit& c);

// Total width of all Bits input ports in `c`
// (= the required length of bit_values for resolve_branches).
int bits_total_width(const Circuit& c);

// Re-emit a resolved Circuit (no IfBegin/IfEnd) as a Stim-superset text string.
// Uses `original` for DETECTOR and OBSERVABLE_INCLUDE annotations (placed at the
// end of the emitted text with rec[-k] offsets relative to the final meas count).
// PAULI_EXPECTATION instructions carry their label from original.expectation_labels.
// The returned text is parseable by parse_stim_circuit and usable as input to
// the PyTwirlSampler constructor.
//
// Task 6 (bit_values non-empty): when `bit_values` is the same pattern supplied to
// resolve_branches(), the function correctly handles measurement-bearing branches:
//   - Detectors inside dropped branches are OMITTED (their superset slot is absent).
//   - rec[-k] references for detectors/observables outside dropped branches are
//     REMAPPED to compensate for the reduced measurement count (the mapping is built
//     by re-simulating the branch resolution on original.circuit.stream).
//   - If a live detector/observable references a measurement from a dropped branch
//     (circuit bug), a std::runtime_error is thrown — LOUD REFUSE, never silent-wrong.
// When `bit_values` is empty (default / backward-compat, branch-free circuits):
//   the function behaves identically to the pre-Task-6 implementation.
std::string emit_resolved_text(const Circuit& resolved, const ParsedStim& original,
                                const std::vector<uint8_t>& bit_values = {});

// Task 6: return the IF conditions for each detector in `ps.detectors` (parallel order).
// Each element is a list of (global_bit_offset, cond_value) pairs describing all IF
// conditions that must hold for that detector to be active.  An empty list means the
// detector is outside all IF blocks (always active).  Used by the Python partition layer
// to build the superset live-mask without re-parsing the circuit text.
std::vector<std::vector<std::pair<int,bool>>>
    detector_branch_spans(const ParsedStim& ps);

}  // namespace qeccore
