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
    return defer_measurements(in, map, false);
}

Circuit defer_measurements(const Circuit& in, DeferralMap* map, bool diagonal_copy) {
    Circuit out;
    std::vector<int> w2q(in.n);
    for (int i = 0; i < in.n; ++i) w2q[i] = i;
    int next_fresh = in.n;
    // Provenance of every deferred qubit (DeferralMap::wire_origin; messages only). Filled in
    // lockstep with next_fresh so index == deferred qubit.
    std::vector<WireOrigin> origin((size_t)in.n);
    for (int i = 0; i < in.n; ++i) origin[(size_t)i] = {WireOrigin::Kind::Original, i, -1};
    auto fresh = [&](WireOrigin::Kind kind, int wire, int record) {
        origin.push_back({kind, wire, record});
        return next_fresh++;
    };
    // A Reset of a wire nothing has touched yet (the usual `R q` / `RX q` prologue) allocates the
    // qubit's FIRST incarnation: Original, not "after its reset" (messages name it plainly).
    std::vector<char> used((size_t)in.n, 0);
    auto touch = [&](int w) { if (w >= 0 && w < in.n) used[(size_t)w] = 1; };
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
        for (int& q : qs) { assert(q >= 0 && q < (int)w2q.size() && "wire index out of range"); touch(q); q = w2q[q]; }
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
                    touch(pt.qubit);
                    pt.qubit = w2q[pt.qubit];
                }
                out.stream.push_back(std::move(ob));
                break;
            }
            case Instr::Kind::Measure: {
                assert(!ins.qubits.empty() && "Measure requires a qubit");
                assert(ins.qubits[0] >= 0 && ins.qubits[0] < (int)w2q.size() && "Measure wire out of range");
                const int wire = ins.qubits[0];
                touch(wire);
                const int q = w2q[wire];
                if (abandoned_after(k, wire)) {
                    const int record = (int)end_meas.size();
                    end_meas.push_back({q, ins.basis});
                    w2q[wire] = fresh(WireOrigin::Kind::Retired, wire, record);  // retire (virtual reset)
                } else if (diagonal_copy && ins.basis != PauliBasis::Z) {
                    // Diagonal-copy deferral (DEM export fallback, see deferral.hpp): copy the
                    // X/Y eigenvalue of q onto a FRESH |+> ancilla by a controlled-P(a -> q), and
                    // read the ancilla terminally in X. Heisenberg: X_a -> X_a P_q with X_a = +1 on
                    // |+>, so the record is P_q's eigenvalue and q is left exactly as the H-copy
                    // leaves it (the same measurement channel). No basis change ever touches q, so
                    // a diagonal-plus-Pauli error residual on q stays in that class through the
                    // copy (CX/S/SDG are in-class); the only H acts on the untouched fresh wire.
                    const int a = fresh(WireOrigin::Kind::MeasureCopy, wire, (int)end_meas.size());
                    out.stream.push_back(mk_gate(GateKind::H, {a}));          // |0> -> |+>
                    if (ins.basis == PauliBasis::Y)
                        out.stream.push_back(mk_gate(GateKind::SDG, {q}));    // CY = S_q CX S_q^dag
                    out.stream.push_back(mk_gate(GateKind::CX, {a, q}));      // CNOT(a->q)
                    if (ins.basis == PauliBasis::Y)
                        out.stream.push_back(mk_gate(GateKind::S, {q}));
                    end_meas.push_back({a, PauliBasis::X});
                } else {
                    const int a = fresh(WireOrigin::Kind::MeasureCopy, wire, (int)end_meas.size());
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
                {
                    const int w = ins.qubits[0];
                    const bool first = !used[(size_t)w];     // the qubit's first incarnation
                    touch(w);
                    w2q[w] = fresh(first ? WireOrigin::Kind::Original : WireOrigin::Kind::Reset, w, -1);  // fresh |0>
                }
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
    if (map) { map->final_wire = w2q; map->terminal_reads = end_meas; map->wire_origin = origin; }
    return out;
}

}  // namespace qeccore
