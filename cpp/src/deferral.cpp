#include "qeccore/deferral.hpp"
#include <cstddef>
#include "qeccore/clifford_frames.hpp"
#include <cassert>
#include <stdexcept>

namespace qeccore {

static Instr mk_gate(GateKind g, std::vector<int> targets) {
    Instr i; i.kind = Instr::Kind::Gate; i.gate = g; i.targets = std::move(targets); return i;
}

Circuit defer_measurements(const Circuit& in) { return defer_measurements(in, nullptr); }

Circuit defer_measurements(const Circuit& in, DeferralMap* map) {
    Circuit out;
    std::vector<int> w2q(in.n);
    for (int i = 0; i < in.n; ++i) w2q[i] = i;
    int next_fresh = in.n;
    // (physical qubit, basis) in record order. Abandoned-wire measurements are the wire
    // itself with its tagged basis; live-wire measurements are the CNOT-copy ancilla in Z.
    std::vector<std::pair<int, PauliBasis>> end_meas;

    // Liveness: the Measure at input index k on wire w is ABANDONED iff no Gate/Measure/
    // Observable uses w before the next Reset of w (or stream end). Noise on w in that
    // window is NOT a use — it hits a discarded post-measurement state (unobservable);
    // retiring the wire at the Measure redirects such noise to a dead fresh wire so it
    // cannot corrupt the pending terminal measurement.
    auto abandoned_after = [&](size_t k, int w) {
        for (size_t j = k + 1; j < in.stream.size(); ++j) {
            const Instr& x = in.stream[j];
            switch (x.kind) {
                case Instr::Kind::Gate:
                    for (int t : x.targets) if (t == w) return false;
                    break;
                case Instr::Kind::Measure:
                    assert(!x.qubits.empty() && "Measure requires a qubit");
                    if (x.qubits[0] == w) return false;
                    break;
                case Instr::Kind::Observable:
                    for (const PauliTerm& pt : x.obs) if (pt.qubit == w) return false;
                    break;
                case Instr::Kind::Reset:
                    assert(!x.qubits.empty() && "Reset requires a qubit");
                    if (x.qubits[0] == w) return true;
                    break;
                case Instr::Kind::Noise:
                    break;
                case Instr::Kind::ControlledPauli:
                    // Feedback is split out before deferral (Task 2); it must never reach here.
                    // HARD throw (active in Release too) so feedback can never be silently mishandled.
                    throw std::runtime_error(
                        "defer_measurements: classically-controlled Pauli feedback "
                        "(CX/CY/CZ rec[-k] q) must be coherentized/stripped before deferral; "
                        "reaching here is a pipeline bug");
            }
        }
        return true;
    };

    auto remap = [&](std::vector<int> qs) {
        for (int& q : qs) { assert(q >= 0 && q < (int)w2q.size() && "wire index out of range"); q = w2q[q]; }
        return qs;
    };

    for (size_t k = 0; k < in.stream.size(); ++k) {
        const Instr& ins = in.stream[k];
        switch (ins.kind) {
            case Instr::Kind::Gate: {
                Instr g = ins; g.targets = remap(ins.targets); out.stream.push_back(std::move(g));
                break;
            }
            case Instr::Kind::Noise: {
                Instr nz = ins; nz.qubits = remap(ins.qubits); out.stream.push_back(std::move(nz));
                break;
            }
            case Instr::Kind::Observable: {
                Instr ob = ins;
                for (PauliTerm& pt : ob.obs) {
                    assert(pt.qubit >= 0 && pt.qubit < (int)w2q.size() && "observable wire out of range");
                    pt.qubit = w2q[pt.qubit];
                }
                out.stream.push_back(std::move(ob));
                break;
            }
            case Instr::Kind::Measure: {
                assert(!ins.qubits.empty() && "Measure requires a qubit");
                assert(ins.qubits[0] >= 0 && ins.qubits[0] < (int)w2q.size() && "Measure wire out of range");
                const int wire = ins.qubits[0];
                const int q = w2q[wire];
                if (abandoned_after(k, wire)) {
                    end_meas.push_back({q, ins.basis});
                    w2q[wire] = next_fresh++;              // retire (virtual reset)
                } else {
                    const int a = next_fresh++;
                    // Basis change so a (measured in Z) records q's basis-eigenvalue; q stays coherent.
                    // Shared frame primitives (clifford_frames): rotate_to_z then CNOT then rotate_from_z.
                    for (Instr& gi : rotate_to_z(ins.basis, q)) out.stream.push_back(std::move(gi));
                    out.stream.push_back(mk_gate(GateKind::CX, {q, a}));      // CNOT(q->a)
                    for (Instr& gi : rotate_from_z(ins.basis, q)) out.stream.push_back(std::move(gi));
                    end_meas.push_back({a, PauliBasis::Z});
                }
                break;
            }
            case Instr::Kind::Reset: {
                assert(!ins.qubits.empty() && "Reset requires a qubit");
                assert(ins.qubits[0] >= 0 && ins.qubits[0] < (int)w2q.size() && "Reset wire out of range");
                w2q[ins.qubits[0]] = next_fresh++;                     // fresh |0>
                break;
            }
            case Instr::Kind::ControlledPauli:
                // Classically-controlled Pauli feedback is split out before deferral (Task 2);
                // it must never reach the deferral pass. HARD throw (active in Release too): the
                // worst case for any consumer is a loud error, never a silent drop of feedback.
                throw std::runtime_error(
                    "defer_measurements: classically-controlled Pauli feedback "
                    "(CX/CY/CZ rec[-k] q) must be coherentized/stripped before deferral; "
                    "reaching here is a pipeline bug");
        }
    }
    for (auto& [q, b] : end_meas) {
        Instr m; m.kind = Instr::Kind::Measure; m.basis = b; m.qubits = {q};
        out.stream.push_back(std::move(m));
    }
    out.n = next_fresh;
    if (map) { map->final_wire = w2q; map->terminal_reads = end_meas; }
    return out;
}

}  // namespace qeccore
