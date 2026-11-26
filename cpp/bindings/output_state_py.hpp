#pragma once
// Output-state helpers' binding: parse, normalize (the twirl sampler's
// preset), the wire maps, analyze_output, and the by-name messages (decision 6). No bare state, no RecordFidelity.
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/output_state.hpp"

namespace {

struct OutputModel {
    qeccore::ParsedStim ps;
    qeccore::NormalizeResult nr;
    std::vector<int> qubits, out_wires, read_meas, wire_user;
};

inline qeccore::ParsedStim output_parse(const std::string& text) {
    qeccore::ParsedStim ps = qeccore::parse_stim_circuit(text, /*track_source=*/true);
    if (!ps.ok()) {
        std::string err = "parse failed";
        for (const auto& e : ps.errors) err += "\n  line " + std::to_string(e.line) + ": " + e.message;
        throw std::invalid_argument(err);
    }
    return ps;
}

inline void output_model(const std::string& text, const std::vector<int>& qubits, OutputModel& M) {
    using namespace qeccore;
    M.ps = output_parse(text);
    for (const InputPort& p : M.ps.circuit.inputs)
        if (p.kind == InputPort::Kind::Qubits)
            throw std::invalid_argument("the circuit declares INPUT_QUBITS '" + p.name + "': its output "
                                        "depends on the carried input state; compose the stages first "
                                        "(compile_twirl_sampler_from_ideal_state)");
    // Messages carry no surface prefix: xtim/output_helpers.py names the helper that raised (final review m3).
    if (qubits.empty()) throw std::invalid_argument("malformed qubit list: qubits is empty");
    const std::vector<QubitEnd> ends = qubit_ends(M.ps);
    std::set<int> seen;
    for (int q : qubits) {
        if (q < 0 || q >= M.ps.num_user_qubits)
            throw std::invalid_argument("malformed qubit list: qubit " + std::to_string(q) + " is not a qubit of the circuit " +
                                        (M.ps.num_user_qubits > 0 ? "(it uses qubits 0.." + std::to_string(M.ps.num_user_qubits - 1) + ")"
                                                                  : std::string("(the circuit has no qubits)")));
        if (!seen.insert(q).second) throw std::invalid_argument("malformed qubit list: qubit " + std::to_string(q) +
                                                                " is listed twice");
        if (ends[(size_t)q] == QubitEnd::Measured)
            throw std::invalid_argument("RF_OUTPUT_MEASURED: qubit " + std::to_string(q) + " is measured after its last "
                                        "gate, so its end state is a classical record. Pass qubits that are not "
                                        "measured after their last gate (live qubits from xtim.output_candidates, or "
                                        "qubits left in their reset state), or move the state onto a fresh wire "
                                        "(noiseless R + SWAP) before measuring the old one.");
    }
    M.qubits = qubits;                                 // M7: one canonical order, so every field is the same for
    std::sort(M.qubits.begin(), M.qubits.end());       // any order of the caller's list
    M.nr = normalize_for_twirl_sampler(M.ps.circuit);
    const int n = M.nr.normalized.n;
    const DeferralMap& map = M.nr.map;
    for (int q : M.qubits) M.out_wires.push_back(map.final_wire[(size_t)q]);
    M.read_meas.assign((size_t)n, -1);                 // family_sampler.hpp:521-525
    for (size_t r = 0; r < map.terminal_reads.size(); ++r) {
        const int w = map.terminal_reads[r].first;
        if (w >= 0 && w < n) M.read_meas[(size_t)w] = (int)r;
    }
    M.wire_user.assign((size_t)n, -1);                 // family_sampler.hpp:530-536
    for (size_t d = 0; d < map.wire_origin.size() && d < (size_t)n; ++d)
        if (map.wire_origin[d].wire >= 0 && map.wire_origin[d].wire < M.ps.num_user_qubits) M.wire_user[d] = map.wire_origin[d].wire;
    for (size_t q = 0; q < map.final_wire.size(); ++q)
        if (map.final_wire[q] >= 0 && map.final_wire[q] < n) M.wire_user[(size_t)map.final_wire[q]] = (int)q;
}

inline std::string out_part(const qeccore::Pauli& P, const OutputModel& M) {
    std::vector<std::pair<int, char>> f;
    for (size_t i = 0; i < M.out_wires.size(); ++i) {
        const int w = M.out_wires[i];
        const bool x = P.xbit(w), z = P.zbit(w);
        if (x || z) f.push_back({M.qubits[i], x && z ? 'Y' : x ? 'X' : 'Z'});
    }
    std::sort(f.begin(), f.end());
    std::string s;
    for (const auto& t : f) s += (s.empty() ? "" : "*") + std::string(1, t.second) + std::to_string(t.first);
    return s.empty() ? "I" : s;
}
inline void rest_part(const qeccore::Pauli& P, const OutputModel& M, std::set<int>& meas, std::set<std::string>& wires) {
    const std::set<int> outw(M.out_wires.begin(), M.out_wires.end());
    for (int w = 0; w < P.n; ++w) {
        if (outw.count(w) || !(P.xbit(w) || P.zbit(w))) continue;
        if (M.read_meas[(size_t)w] >= 0) meas.insert(M.read_meas[(size_t)w]);
        else wires.insert(M.wire_user[(size_t)w] >= 0 ? "q" + std::to_string(M.wire_user[(size_t)w]) : "internal wire " + std::to_string(w));
    }
}
template <class C> std::string listing(const C& v) {
    std::string s;
    size_t i = 0;
    for (const auto& x : v) {
        if (i == 24) { s += ", ... (" + std::to_string(v.size()) + " in all)"; break; }
        std::ostringstream o;
        o << x;
        s += (i++ ? ", " : "") + o.str();
    }
    return "[" + s + "]";
}

// "" when deterministic; else the RF_OUTPUT_NOT_DETERMINISTIC message. Undecided and out-of-class throw by name.
inline std::string verdict_message(const qeccore::OutputAnalysis& a, const OutputModel& M) {
    using V = qeccore::OutputVerdict;
    if (a.verdict == V::Deterministic) return "";
    if (a.verdict == V::OutOfClass)
        throw std::invalid_argument("bare state rejected: " + a.reject_reason + user_terms_text(a.reject_terms, M.nr, M.ps) +
                                    "; the circuit is outside xtim's model (its non-Clifford gates do not fold into one "
                                    "commuting layer of pi/8 rotations)");
    std::set<int> meas;
    std::set<std::string> wires;
    std::vector<std::string> dirs;                    // final review m1: each direction once, in witness order
    for (const auto& w : a.witnesses) {
        const std::string d = out_part(w, M);
        if (std::find(dirs.begin(), dirs.end(), d) == dirs.end()) dirs.push_back(d);
        rest_part(w, M, meas, wires);
    }
    // the rest's side, naming only the non-empty parts (no "measurements []")
    std::string fixers;
    if (!meas.empty()) fixers = "measurements " + listing(meas) + " (0-based record indices, in circuit order)";
    if (!wires.empty()) fixers += (fixers.empty() ? "" : " and ") + std::string("the unmeasured qubits ") + listing(wires);
    if (fixers.empty()) fixers = "no qubit outside the output";
    if (a.verdict == V::Undecided)
        throw std::invalid_argument(
            "RF_OUTPUT_UNDECIDED: xtim cannot decide whether the noiseless state of qubits " + listing(M.qubits) +
            " is pure and independent of the measurement outcomes: pi/8 rotations straddle the output and the rest "
            "(on your qubits " + listing(dirs) + "; on the rest " + fixers + "). This usually means the output is "
            "entangled with the rest through the magic: include those qubits in the output, or apply the non-Clifford "
            "gates to the output only. (xtim refuses rather than guess.)");
    const std::string head = "the noiseless state of qubits " + listing(M.qubits) +
                             " is not pure and fixed independently of the measurement outcomes.";
    if (a.verdict == V::CrossStabilizer) {
        const std::string why = "RF_OUTPUT_NOT_DETERMINISTIC: " + head + " Undetermined output directions (Paulis on your "
                                "qubits): " + listing(dirs) + ".";
        if (meas.empty())                             // no records to feed back: only the unmeasured qubits remain
            return why + " The output is entangled with " + fixers + " (there are no measurement records to feed back): "
                   "include them in the output, or measure them and feed the outcomes back onto the output (CX rec[-k] "
                   "q, CZ rec[-k] q or CY rec[-k] q).";
        return why + " They are fixed by the outcomes of " + fixers + ": close the frame with Pauli feedback from "
               "those records onto the output (CX rec[-k] q, CZ rec[-k] q or CY rec[-k] q), as for a deterministic Stim "
               "detector, or measure the entangled qubits.";
    }
    if (a.verdict == V::CoupledMagic)
        return "RF_OUTPUT_NOT_DETERMINISTIC: " + head + " A non-Clifford (pi/8) rotation couples the output to the "
               "rest: on your qubits its axis acts as " + listing(dirs) + ", and on the rest it touches " + fixers + ".";
    return "RF_OUTPUT_NOT_DETERMINISTIC: " + head + " The output is entangled with the rest through its stabilizer "
           "(Clifford) part, in the directions " + listing(dirs) + " on your qubits (rest: " + fixers + "). Disentangle "
           "it by measurement with Pauli feedback, or include the entangled qubits in the output.";
}

inline std::vector<int> py_output_candidates(const std::string& text) {
    std::vector<int> out;
    { py::gil_scoped_release rel; out = qeccore::output_candidates(output_parse(text)); }
    return out;
}

inline py::tuple py_output_state_check(const std::string& text, const std::vector<int>& qubits) {
    std::string why;
    {
        py::gil_scoped_release rel;
        OutputModel M;
        output_model(text, qubits, M);
        why = verdict_message(qeccore::analyze_output(M.nr.normalized, M.out_wires, false), M);
    }
    return py::make_tuple(why.empty(), why);
}

inline py::tuple signed_user_pauli(const qeccore::Pauli& p, const std::vector<int>& qubits) {
    std::vector<std::pair<int, char>> f;
    for (int i = 0; i < p.n; ++i) { const bool x = p.xbit(i), z = p.zbit(i); if (x || z) f.push_back({qubits[(size_t)i], x && z ? 'Y' : x ? 'X' : 'Z'}); }
    std::sort(f.begin(), f.end());
    std::string s;
    for (const auto& t : f) s += (s.empty() ? "" : "*") + std::string(1, t.second) + std::to_string(t.first);
    return py::make_tuple(s, qeccore::hermitian_sign(p));
}

inline py::dict py_output_state(const std::string& text, const std::vector<int>& qubits) {
    OutputModel M;
    qeccore::OutputAnalysis a;
    {
        py::gil_scoped_release rel;
        output_model(text, qubits, M);
        a = qeccore::analyze_output(M.nr.normalized, M.out_wires, true);
        const std::string why = verdict_message(a, M);
        if (!why.empty()) throw std::invalid_argument(why);
    }
    const auto lst = [&](const std::vector<qeccore::Pauli>& v) { py::list l; for (const auto& p : v) l.append(signed_user_pauli(p, M.qubits)); return l; };
    py::dict d;
    d["rotations"] = lst(a.rotations);
    d["pauli_stabilizers"] = lst(a.pauli_stabilizers);
    d["clifford_paulis"] = lst(a.clifford_paulis);
    return d;
}

}  // namespace
