// refusal_text.hpp — J-b1 P13: the user-qubit rendering of a twirl refusal's two anticommuting axes (deferred
// wires -> user qubits through DeferralMap::wire_origin, coherentizer ancillas -> the control they copy), then the
// parsed circuit's non-Clifford gates on those qubits. The SAME text as cpp/bindings/xtim_py.cpp's
// bare_reject_message (whose body this is, generalized over the term list); the test API (the twirl
// refusal twin) uses it pre-flip, and at F the production binding includes it and its copy is deleted.
#pragma once

#include <set>
#include <string>
#include <vector>

#include "qeccore/normalize.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/stim_parse.hpp"

namespace qeccore {

// "; in user qubits: <A> and <B>[; the non-Clifford gates on these qubits, as parsed (...): ...]" ("" when the
// map is absent). `extra_wires`: deferred wires whose user qubits also join the gate listing (a noise site's).
inline std::string user_terms_text(const std::vector<Pauli>& terms, const NormalizeResult& nr,
                                   const ParsedStim& ps, std::vector<int> extra_wires = {}) {
    const Circuit& user = ps.circuit;
    const int user_n = ps.num_user_qubits;    // wires at/above it are MPP/SPP desugar ancillas
    std::string msg;
    if (terms.size() != 2 || nr.map.wire_origin.empty()) return msg;
    std::set<int> user_qubits;
    auto coherent_label = [&](int c) -> std::string {
        if (c >= 0 && c < user_n) { user_qubits.insert(c); return "q" + std::to_string(c); }
        if (c >= 0 && c < user.n) return "desugar ancilla wire " + std::to_string(c) + " (MPP/SPP gadget)";
        // A coherentizer ancilla (allocated above the parsed circuit's n). Its first instruction
        // tells which: a held readout-flip instance starts with the fold's X_ERROR(p) on the
        // ancilla (origin ReadoutFold, index = the measurement record) and is then applied to the
        // control by CX/CZ(anc, control); a copy ancilla's first touch is the copy's two-qubit gate
        // (CX(control, anc) for Z, H anc; CX(anc, control) for X, the bracketed Z-copy for Y).
        int flip_record = -1;
        for (const Instr& ins : nr.coherent.stream) {
            bool touches = false;
            if (ins.kind == Instr::Kind::Gate) { for (int t : ins.targets) if (t == c) touches = true; }
            else { for (int q : ins.qubits) if (q == c) touches = true; }
            if (!touches) continue;
            if (ins.kind == Instr::Kind::Noise) {
                if (ins.origin == Instr::Origin::ReadoutFold) flip_record = ins.origin_index;
                continue;
            }
            if (ins.kind != Instr::Kind::Gate || ins.targets.size() != 2) continue;   // H anc of an X-copy
            const int other = ins.targets[0] == c ? ins.targets[1] : ins.targets[0];
            if (other < 0 || other >= user_n) break;
            user_qubits.insert(other);
            if (flip_record >= 0)
                return "readout-flip ancilla of q" + std::to_string(other) + "'s measurement #" +
                       std::to_string(flip_record) + " (coherentized M(p))";
            return "copy of q" + std::to_string(other) + " (coherentized-feedback ancilla)";
        }
        return "internal wire " + std::to_string(c);
    };
    auto wire_label = [&](int d) -> std::string {
        if (d < 0 || d >= (int)nr.map.wire_origin.size()) return "deferred wire " + std::to_string(d);
        const WireOrigin& o = nr.map.wire_origin[(size_t)d];
        std::string s = coherent_label(o.wire);
        switch (o.kind) {
            case WireOrigin::Kind::Original: break;
            case WireOrigin::Kind::MeasureCopy:
                s += " [copy at its measurement #" + std::to_string(o.record) + "]"; break;
            case WireOrigin::Kind::Retired:
                s += " [after its measurement #" + std::to_string(o.record) + ", retired]"; break;
            case WireOrigin::Kind::Reset:
                s += " [after its reset]"; break;
        }
        return s + " (deferred wire " + std::to_string(d) + ")";
    };
    auto term_label = [&](const Pauli& p) -> std::string {
        std::string s;
        for (int q = 0; q < p.n; ++q) {
            const bool xb = p.xbit(q), zb = p.zbit(q);
            if (!xb && !zb) continue;
            if (!s.empty()) s += " * ";
            s += std::string(xb && zb ? "Y" : xb ? "X" : "Z") + " on " + wire_label(q);
        }
        return s.empty() ? std::string("I") : s;
    };
    const std::string t0 = term_label(terms[0]);
    const std::string t1 = term_label(terms[1]);
    for (int w : extra_wires) (void)wire_label(w);   // the noise site's wires join the gate listing
    msg += "; in user qubits: " + t0 + " and " + t1;
    // The parsed circuit's non-Clifford gates on those user qubits (what introduced the magic).
    std::string gates;
    for (size_t i = 0; i < user.stream.size(); ++i) {
        const Instr& ins = user.stream[i];
        if (ins.kind != Instr::Kind::Gate) continue;
        const char* name = nullptr;
        switch (ins.gate) {
            case GateKind::T:   name = "T";   break;
            case GateKind::CS:  name = "CS";  break;
            case GateKind::CCZ: name = "CCZ"; break;
            case GateKind::CH:  name = "CH";  break;
            default: break;
        }
        if (!name) continue;
        bool hit = false;
        for (int t : ins.targets) if (user_qubits.count(t)) hit = true;
        if (!hit) continue;
        if (!gates.empty()) gates += ", ";
        gates += name;
        for (int t : ins.targets) gates += " " + std::to_string(t);
        gates += " (#" + std::to_string(i) + ")";
    }
    if (!gates.empty())
        msg += "; the non-Clifford gates on these qubits, as parsed (flat instruction index, "
               "REPEAT unrolled; T_DAG parses as T up to Cliffords): " + gates;
    return msg;
}

// " (deferred wire d = <user label>)" for one deferred wire (a propagation refusal's fault wire).
inline std::string user_wire_text(int d, const NormalizeResult& nr, const ParsedStim& ps) {
    if (d < 0 || d >= (int)nr.map.wire_origin.size()) return std::string();
    const int w = nr.map.wire_origin[(size_t)d].wire;
    if (w >= 0 && w < ps.num_user_qubits) return " (deferred wire " + std::to_string(d) + " = q" + std::to_string(w) + ")";
    return " (deferred wire " + std::to_string(d) + ": an internal wire)";
}

}  // namespace qeccore
