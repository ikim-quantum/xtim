// Task 5 (decoder-feedback): IF-branch resolver — see resolve_branches.hpp.
#include "qeccore/resolve_branches.hpp"
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace qeccore {

// ── helpers ────────────────────────────────────────────────────────────────────

// Compute the global bit offset for each port in `inputs`.
// Bits ports are numbered sequentially; Pauli ports contribute 0 width.
static std::vector<int> compute_port_offsets(const std::vector<InputPort>& inputs) {
    std::vector<int> offsets(inputs.size(), 0);
    int offset = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        offsets[i] = offset;
        if (inputs[i].kind == InputPort::Kind::Bits)
            offset += inputs[i].width;
    }
    return offsets;
}

// ── resolve_branches ──────────────────────────────────────────────────────────

Circuit resolve_branches(const Circuit& c, const std::vector<uint8_t>& bit_values) {
    // Fast path: no IF blocks.
    if (!circuit_has_branches(c)) {
        Circuit out = c;
        // Still remove Bits ports (spec: inputs keeps only Pauli ports).
        out.inputs.clear();
        for (const InputPort& p : c.inputs)
            if (p.kind == InputPort::Kind::Qubits)
                out.inputs.push_back(p);
        return out;
    }

    // Compute global offset for each port.
    std::vector<int> port_offsets = compute_port_offsets(c.inputs);

    Circuit out;
    out.n = c.n;
    out.outputs = c.outputs;  // Fix 1: preserve OUTPUT_QUBITS declarations
    for (const InputPort& p : c.inputs)
        if (p.kind == InputPort::Kind::Qubits)
            out.inputs.push_back(p);

    // Walk stream; maintain keep/drop stack (one frame per nested IF level).
    std::vector<bool> keep_stack;

    auto all_keep = [&]() -> bool {
        for (bool k : keep_stack)
            if (!k) return false;
        return true;
    };

    for (const Instr& ins : c.stream) {
        if (ins.kind == Instr::Kind::IfBegin) {
            if (ins.cond_port >= 0 && ins.cond_port < (int)c.inputs.size()) {
                int global_bit = port_offsets[ins.cond_port] + ins.cond_bit;
                bool bit_val = (global_bit >= 0 && global_bit < (int)bit_values.size())
                               ? (bit_values[(size_t)global_bit] != 0) : false;
                bool cond_met = (bit_val == ins.cond_value);
                keep_stack.push_back(cond_met);
            } else {
                keep_stack.push_back(false);
            }
        } else if (ins.kind == Instr::Kind::IfEnd) {
            if (!keep_stack.empty()) keep_stack.pop_back();
        } else {
            if (all_keep())
                out.stream.push_back(ins);
        }
    }

    return out;
}

// ── if_driving_bits ───────────────────────────────────────────────────────────

std::vector<std::pair<int,int>> if_driving_bits(const Circuit& c) {
    std::vector<std::pair<int,int>> result;
    std::set<std::pair<int,int>> seen;
    for (const Instr& ins : c.stream) {
        if (ins.kind == Instr::Kind::IfBegin) {
            auto key = std::make_pair(ins.cond_port, ins.cond_bit);
            if (seen.insert(key).second)
                result.push_back(key);
        }
    }
    return result;
}

// ── if_driving_global_offsets ─────────────────────────────────────────────────

std::vector<int> if_driving_global_offsets(const Circuit& c) {
    std::vector<int> port_offsets = compute_port_offsets(c.inputs);

    std::vector<int> result;
    std::set<int> seen;
    for (const Instr& ins : c.stream) {
        if (ins.kind == Instr::Kind::IfBegin) {
            if (ins.cond_port >= 0 && ins.cond_port < (int)c.inputs.size()) {
                int global = port_offsets[ins.cond_port] + ins.cond_bit;
                if (seen.insert(global).second)
                    result.push_back(global);
            }
        }
    }
    return result;
}

// ── bits_total_width ──────────────────────────────────────────────────────────

int bits_total_width(const Circuit& c) {
    int total = 0;
    for (const InputPort& p : c.inputs)
        if (p.kind == InputPort::Kind::Bits)
            total += p.width;
    return total;
}

// ── emit_resolved_text ────────────────────────────────────────────────────────

static const char* gate_name_for_emit(GateKind g) {
    switch (g) {
        case GateKind::X:   return "X";
        case GateKind::Y:   return "Y";
        case GateKind::Z:   return "Z";
        case GateKind::S:   return "S";
        case GateKind::SDG: return "S_DAG";
        case GateKind::H:   return "H";
        case GateKind::CX:  return "CX";
        case GateKind::CZ:  return "CZ";
        case GateKind::T:   return "T";
        case GateKind::CS:  return "CS";
        case GateKind::CCZ: return "CCZ";
        case GateKind::CH:  return "CH";
        default:            return "X";  // fallback; should not happen
    }
}

static const char* noise_name_for_emit(NoiseChannel ch) {
    switch (ch) {
        case NoiseChannel::X_ERROR:        return "X_ERROR";
        case NoiseChannel::Y_ERROR:        return "Y_ERROR";
        case NoiseChannel::Z_ERROR:        return "Z_ERROR";
        case NoiseChannel::DEPOLARIZE1:    return "DEPOLARIZE1";
        case NoiseChannel::DEPOLARIZE2:    return "DEPOLARIZE2";
        case NoiseChannel::PAULI_CHANNEL_1: return "PAULI_CHANNEL_1";
        case NoiseChannel::PAULI_CHANNEL_2: return "PAULI_CHANNEL_2";
        default:                           return "DEPOLARIZE1";
    }
}

static const char* pauli_char_for_emit(PauliBasis p) {
    switch (p) {
        case PauliBasis::X: return "X";
        case PauliBasis::Y: return "Y";
        default:            return "Z";
    }
}

// ── emit_resolved_text ────────────────────────────────────────────────────────

// Build orig_to_res_meas: for each measurement in original.circuit.stream (in stream order),
// store the resolved-circuit absolute index if that measurement is in a kept branch, or -1
// if it was dropped.  Uses the same keep_stack logic as resolve_branches().
// Returns empty vector when bit_values is empty (caller falls back to identity mapping).
static std::vector<int> build_meas_remap(const ParsedStim& original,
                                          const std::vector<uint8_t>& bit_values) {
    if (bit_values.empty()) return {};

    std::vector<int> port_offsets = compute_port_offsets(original.circuit.inputs);
    std::vector<bool> keep_stack;
    auto all_keep = [&]() -> bool {
        for (bool k : keep_stack) if (!k) return false;
        return true;
    };

    std::vector<int> mapping;
    int res_idx = 0;
    for (const Instr& ins : original.circuit.stream) {
        if (ins.kind == Instr::Kind::IfBegin) {
            if (ins.cond_port >= 0 && ins.cond_port < (int)original.circuit.inputs.size()) {
                int gb = port_offsets[ins.cond_port] + ins.cond_bit;
                bool bv = gb >= 0 && gb < (int)bit_values.size() ? (bit_values[gb] != 0) : false;
                keep_stack.push_back(bv == ins.cond_value);
            } else {
                keep_stack.push_back(false);
            }
        } else if (ins.kind == Instr::Kind::IfEnd) {
            if (!keep_stack.empty()) keep_stack.pop_back();
        } else if (ins.kind == Instr::Kind::Measure) {
            mapping.push_back(all_keep() ? res_idx++ : -1);
        }
    }
    return mapping;
}

// Check whether all guards in `guards` are satisfied by `bit_values`.
// An empty guard list means "always active" (vacuously true).
static bool guards_satisfied(const std::vector<IfGuard>& guards,
                              const std::vector<uint8_t>& bit_values,
                              const std::vector<int>& port_offsets) {
    for (const IfGuard& g : guards) {
        if (g.cond_port < 0 || g.cond_port >= (int)port_offsets.size()) return false;
        int gb = port_offsets[g.cond_port] + g.cond_bit;
        bool bv = gb >= 0 && gb < (int)bit_values.size() ? (bit_values[gb] != 0) : false;
        if (bv != g.cond_value) return false;
    }
    return true;
}

std::string emit_resolved_text(const Circuit& resolved, const ParsedStim& original,
                                const std::vector<uint8_t>& bit_values) {
    std::ostringstream out;

    // 1. INPUT_QUBITS headers (Bits ports were consumed by resolve_branches).
    for (const InputPort& p : resolved.inputs)
        if (p.kind == InputPort::Kind::Qubits) {
            out << "INPUT_QUBITS " << p.name;
            for (int q : p.qubits) out << ' ' << q;
            out << '\n';
        }

    // 1b. OUTPUT_QUBITS headers (Fix 2: emit output port declarations).
    for (const OutputPort& p : resolved.outputs) {
        out << "OUTPUT_QUBITS " << p.name;
        for (int q : p.qubits) out << ' ' << q;
        out << '\n';
    }

    // Build orig_to_res BEFORE the stream loop so it is available for
    // ControlledPauli and PAULI_EXPECTATION obs_frame remapping inside the loop.
    // Empty when bit_values is empty (backward-compat: identity mapping).
    std::vector<int> orig_to_res = build_meas_remap(original, bit_values);
    // Pre-compute port offsets for keep_stack evaluation when bit_values non-empty.
    std::vector<int> port_offsets;
    if (!bit_values.empty())
        port_offsets = compute_port_offsets(original.circuit.inputs);

    // Helper: remap original absolute measurement index to resolved absolute index.
    // Returns -1 if the measurement lived in a dropped branch.
    // When orig_to_res is empty (no bit_values), returns abs_idx unchanged (identity).
    auto remap = [&](int abs_idx) -> int {
        if (orig_to_res.empty()) return abs_idx;
        if (abs_idx < 0 || abs_idx >= (int)orig_to_res.size()) return -1;
        return orig_to_res[abs_idx];
    };

    // 2. Stream instructions: walk original.circuit.stream with the same keep_stack
    // logic as resolve_branches() / build_meas_remap().  Walking the original stream
    // (rather than the pre-filtered resolved.stream) gives us orig_meas_count — the
    // measurement count in the ORIGINAL stream — which is needed to correctly remap
    // ControlledPauli's control_rec_offset (a relative-k into the original stream) and
    // PAULI_EXPECTATION's obs_frame (absolute original indices) through orig_to_res.
    // LOUD REFUSE for both when the referenced measurement was in a dropped branch.
    int meas_count      = 0;   // measurements emitted in resolved stream so far
    int orig_meas_count = 0;   // measurements encountered in original stream (kept + dropped)
    int exp_idx         = 0;   // index into original.expectation_labels (all Observable instrs)

    std::vector<bool> keep_stack;
    auto all_keep_emit = [&]() -> bool {
        for (bool k : keep_stack) if (!k) return false;
        return true;
    };

    for (const Instr& ins : original.circuit.stream) {
        // IfBegin / IfEnd: update keep_stack when bit_values are given; skip emission.
        if (ins.kind == Instr::Kind::IfBegin) {
            if (!bit_values.empty()) {
                if (ins.cond_port >= 0 && ins.cond_port < (int)original.circuit.inputs.size()) {
                    int gb = port_offsets[ins.cond_port] + ins.cond_bit;
                    bool bv = (gb >= 0 && gb < (int)bit_values.size())
                              ? (bit_values[gb] != 0) : false;
                    keep_stack.push_back(bv == ins.cond_value);
                } else {
                    keep_stack.push_back(false);
                }
            }
            // When bit_values is empty: no keep_stack push → all_keep_emit() stays true.
            continue;
        }
        if (ins.kind == Instr::Kind::IfEnd) {
            if (!bit_values.empty() && !keep_stack.empty()) keep_stack.pop_back();
            continue;
        }

        bool keep = all_keep_emit();

        switch (ins.kind) {
            case Instr::Kind::Gate:
                if (keep) {
                    out << gate_name_for_emit(ins.gate);
                    for (int t : ins.targets) out << ' ' << t;
                    out << '\n';
                }
                break;

            case Instr::Kind::Measure:
                if (keep) {
                    const char* nm = ins.basis == PauliBasis::X ? "MX"
                                   : ins.basis == PauliBasis::Y ? "MY" : "M";
                    out << nm;
                    if (ins.readout_flip_p > 0.0)
                        out << '(' << ins.readout_flip_p << ')';
                    for (int q : ins.qubits)
                        out << (ins.invert ? " !" : " ") << q;
                    out << '\n';
                    ++meas_count;
                }
                ++orig_meas_count;  // always count: kept OR dropped
                break;

            case Instr::Kind::Reset:
                if (keep) {
                    out << 'R';
                    for (int q : ins.qubits) out << ' ' << q;
                    out << '\n';
                }
                break;

            case Instr::Kind::Noise:
                if (keep) {
                    out << noise_name_for_emit(ins.channel) << '(';
                    for (size_t i = 0; i < ins.probs.size(); ++i) {
                        if (i) out << ',';
                        out << ins.probs[i];
                    }
                    out << ')';
                    for (int q : ins.qubits) out << ' ' << q;
                    out << '\n';
                }
                break;

            case Instr::Kind::Observable: {
                // PAULI_EXPECTATION <label> P0*P1*... [rec[-k] ...]
                // exp_idx increments for EVERY Observable instruction (kept or dropped) so
                // the label-index mapping stays aligned with original.expectation_labels.
                int label = (exp_idx < (int)original.expectation_labels.size())
                            ? original.expectation_labels[exp_idx] : 0;
                if (keep) {
                    out << "PAULI_EXPECTATION(" << label << ")";
                    bool first = true;
                    for (const PauliTerm& t : ins.obs) {
                        out << (first ? ' ' : '*');
                        first = false;
                        out << pauli_char_for_emit(t.p) << t.qubit;
                    }
                    // obs_frame: absolute original measurement indices.
                    // Remap through orig_to_res; LOUD REFUSE if any ref was dropped.
                    for (int abs_idx : ins.obs_frame) {
                        int res_abs = remap(abs_idx);
                        if (res_abs < 0) {
                            std::string msg =
                                "emit_resolved_text: rec[-k] remapping error — "
                                "PAULI_EXPECTATION label " + std::to_string(label) +
                                " obs_frame references original measurement index " +
                                std::to_string(abs_idx) +
                                " which belongs to a dropped IF branch. "
                                "A PAULI_EXPECTATION may not reference a measurement "
                                "inside a dropped branch.";
                            throw std::runtime_error(msg);
                        }
                        out << " rec[-" << (meas_count - res_abs) << ']';
                    }
                    out << '\n';
                }
                ++exp_idx;
                break;
            }

            case Instr::Kind::ControlledPauli: {
                // control_rec_offset = k from rec[-k] (relative to original stream meas count).
                // Absolute original index = orig_meas_count - k.
                // Remap through orig_to_res; LOUD REFUSE if the measurement was dropped.
                if (keep) {
                    int orig_abs = orig_meas_count - ins.control_rec_offset;
                    int res_abs  = remap(orig_abs);
                    if (res_abs < 0) {
                        const char* g = ins.basis == PauliBasis::X ? "CX"
                                      : ins.basis == PauliBasis::Y ? "CY" : "CZ";
                        std::string msg =
                            "emit_resolved_text: rec[-k] remapping error — "
                            "ControlledPauli (" + std::string(g) + " rec[-" +
                            std::to_string(ins.control_rec_offset) + "] " +
                            std::to_string(ins.qubits.empty() ? -1 : ins.qubits[0]) +
                            ") references original measurement index " +
                            std::to_string(orig_abs) +
                            " which belongs to a dropped IF branch. "
                            "A ControlledPauli may not reference a measurement "
                            "inside a dropped branch.";
                        throw std::runtime_error(msg);
                    }
                    const char* g = ins.basis == PauliBasis::X ? "CX"
                                  : ins.basis == PauliBasis::Y ? "CY" : "CZ";
                    out << g << " rec[-" << (meas_count - res_abs) << "] "
                        << ins.qubits[0] << '\n';
                }
                break;
            }

            case Instr::Kind::IfBegin:
            case Instr::Kind::IfEnd:
                // Already handled above; unreachable here.
                break;
        }
    }

    // 3. DETECTOR annotations: skip dropped-branch detectors; remap rec[-k] for live ones.
    for (size_t i = 0; i < original.detectors.size(); ++i) {
        // Task 6: if bit_values given, check whether this detector's IF guards are satisfied.
        if (!bit_values.empty()) {
            const std::vector<IfGuard>& guards =
                (i < original.detector_if_guards.size())
                ? original.detector_if_guards[i]
                : std::vector<IfGuard>{};
            if (!guards_satisfied(guards, bit_values, port_offsets))
                continue;  // detector is inside a dropped branch → absent from resolved text
        }

        const auto& det = original.detectors[i];
        out << "DETECTOR";
        for (int abs_idx : det) {
            int res_idx = remap(abs_idx);
            if (res_idx < 0) {
                // LOUD REFUSE: a live detector references a measurement that was dropped.
                // This indicates a malformed circuit (a detector outside a branch that refs a
                // measurement inside a DIFFERENT dropped branch). Never emit a wrong rec[-k].
                std::string msg =
                    "emit_resolved_text: rec[-k] remapping error — detector " +
                    std::to_string(i) + " references original measurement index " +
                    std::to_string(abs_idx) + " which belongs to a dropped IF branch. "
                    "A detector outside a dropped branch may not reference a measurement "
                    "inside that branch.";
                throw std::runtime_error(msg);
            }
            out << " rec[-" << (meas_count - res_idx) << ']';
        }
        out << '\n';
    }

    // 4. OBSERVABLE_INCLUDE annotations (sorted by observable index via std::map iteration).
    // Remap abs_idx for each ref; LOUD REFUSE if any ref was dropped.
    for (const auto& kv : original.observables) {
        out << "OBSERVABLE_INCLUDE(" << kv.first << ')';
        for (int abs_idx : kv.second) {
            int res_idx = remap(abs_idx);
            if (res_idx < 0) {
                std::string msg =
                    "emit_resolved_text: rec[-k] remapping error — observable " +
                    std::to_string(kv.first) + " references original measurement index " +
                    std::to_string(abs_idx) + " which belongs to a dropped IF branch. "
                    "An observable outside a dropped branch may not reference a measurement "
                    "inside that branch.";
                throw std::runtime_error(msg);
            }
            out << " rec[-" << (meas_count - res_idx) << ']';
        }
        out << '\n';
    }

    // 5. DECISION annotations (same remap treatment as OBSERVABLE_INCLUDE; one line per entry).
    for (const auto& kv : original.decisions) {
        out << "DECISION(" << kv.first << ')';
        for (int abs_idx : kv.second) {
            int res_idx = remap(abs_idx);
            if (res_idx < 0) {
                std::string msg =
                    "emit_resolved_text: rec[-k] remapping error — decision " +
                    std::to_string(kv.first) + " references original measurement index " +
                    std::to_string(abs_idx) + " which belongs to a dropped IF branch. "
                    "A DECISION may not reference a measurement inside a dropped branch.";
                throw std::runtime_error(msg);
            }
            out << " rec[-" << (meas_count - res_idx) << ']';
        }
        out << '\n';
    }

    return out.str();
}

// ── detector_branch_spans ─────────────────────────────────────────────────────

std::vector<std::vector<std::pair<int,bool>>>
    detector_branch_spans(const ParsedStim& ps) {
    std::vector<int> port_offsets = compute_port_offsets(ps.circuit.inputs);
    std::vector<std::vector<std::pair<int,bool>>> result;
    result.reserve(ps.detector_if_guards.size());
    for (const auto& guards : ps.detector_if_guards) {
        std::vector<std::pair<int,bool>> conds;
        conds.reserve(guards.size());
        for (const IfGuard& g : guards) {
            int global_off = (g.cond_port >= 0 && g.cond_port < (int)port_offsets.size())
                             ? port_offsets[g.cond_port] + g.cond_bit : -1;
            conds.push_back({global_off, g.cond_value});
        }
        result.push_back(std::move(conds));
    }
    return result;
}

}  // namespace qeccore
