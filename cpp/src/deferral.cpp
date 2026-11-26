#include "qeccore/deferral.hpp"
#include <cstddef>
#include "qeccore/clifford_frames.hpp"
#include <cassert>
#include <stdexcept>
#include <string>

namespace qeccore {

static Instr mk_gate(GateKind g, std::vector<int> targets) {
    Instr i; i.kind = Instr::Kind::Gate; i.gate = g; i.targets = std::move(targets); return i;
}

// The by-name refusal of a circuit with IF blocks (owner rule: exact or refuse by name). An IF
// body runs for a shot iff its INPUT_BITS bit matches; deferral (and every surface built on it:
// DEM export, twirl state construction, frame extraction, ...) receives no bit values, so it
// cannot evaluate the branch. std::invalid_argument surfaces as Python ValueError.
[[noreturn]] static void refuse_if_blocks(const Circuit& in) {
    std::string conds;
    for (const Instr& i : in.stream) {
        if (i.kind != Instr::Kind::IfBegin) continue;
        std::string c = i.cond_value ? "IF " : "IF !";
        if (i.cond_port >= 0 && i.cond_port < (int)in.inputs.size())
            c += in.inputs[(size_t)i.cond_port].name;
        else
            c += "<INPUT_BITS port #" + std::to_string(i.cond_port) + ">";
        c += "[" + std::to_string(i.cond_bit) + "]";
        if (conds.find(c) == std::string::npos) conds += (conds.empty() ? "" : ", ") + c;
    }
    throw std::invalid_argument(
        "the circuit has IF blocks on INPUT_BITS (" + (conds.empty() ? std::string("IF") : conds) +
        ") and no bit values are supplied here; an IF body runs for a shot iff its bit matches, "
        "which this surface cannot evaluate. Resolve the branches first "
        "(xtim._xtim.resolve_branches_text(text, bits) -> one concrete circuit per bit pattern) "
        "or use xtim.twirl.PartitionedTwirlSampler.sample(decision_bits=...)");
}

Circuit defer_measurements(const Circuit& in) { return defer_measurements(in, nullptr); }

Circuit defer_measurements(const Circuit& in, DeferralMap* map) {
    return defer_measurements(in, map, false);
}

// Liveness in one backward pass, O(L + n): abandoned[k] (k a Measure on wire w) iff the first
// Gate/Measure/Observable use or Reset of w after k is a Reset, or there is neither. The same
// predicate as the forward scan below (`abandoned_after`). The scan also stops at the first
// ControlledPauli / IF marker after k and throws; this pass ignores them: the main loop throws the
// same error (same type and message) at the stream's FIRST such marker, which is the one every
// scan reaches first, and nothing is returned or written before that throw.
static std::vector<char> abandoned_measurements(const Circuit& in) {
    enum : char { None = 0, Use = 1, Reset = 2 };
    std::vector<char> next((size_t)in.n, None);     // first event on each wire after the cursor
    std::vector<char> abandoned(in.stream.size(), 0);
    auto use = [&](int q) { if (q >= 0 && q < in.n) next[(size_t)q] = Use; };
    for (size_t k = in.stream.size(); k-- > 0;) {
        const Instr& x = in.stream[k];
        switch (x.kind) {
            case Instr::Kind::Gate:
                for (int t : x.targets) use(t);
                break;
            case Instr::Kind::Measure:
                if (x.qubits.empty()) break;
                if (x.qubits[0] >= 0 && x.qubits[0] < in.n)
                    abandoned[k] = next[(size_t)x.qubits[0]] != Use;
                use(x.qubits[0]);
                break;
            case Instr::Kind::Observable:
                for (const PauliTerm& pt : x.obs) use(pt.qubit);
                break;
            case Instr::Kind::Reset:
                if (!x.qubits.empty() && x.qubits[0] >= 0 && x.qubits[0] < in.n) next[(size_t)x.qubits[0]] = Reset;
                break;
            case Instr::Kind::Noise:
            case Instr::Kind::ControlledPauli:
            case Instr::Kind::IfBegin:
            case Instr::Kind::IfEnd:
                break;
        }
    }
    return abandoned;
}

static Circuit defer_impl(const Circuit& in, DeferralMap* map, bool diagonal_copy, bool bruteforce_liveness) {
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

    const std::vector<char> abandoned = bruteforce_liveness ? std::vector<char>() : abandoned_measurements(in);
    // Liveness (the TEST-ORACLE forward scan; production uses abandoned_measurements): the
    // Measure at input index k on wire w is ABANDONED iff no Gate/Measure/
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
                case Instr::Kind::IfBegin:
                case Instr::Kind::IfEnd:
                    // Conditional-branch marker: the liveness of a wire depends on the shot's
                    // branch bits, which deferral does not receive. Refuse by name.
                    refuse_if_blocks(in);
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
                if (bruteforce_liveness ? abandoned_after(k, wire) : abandoned[k] != 0) {
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
            case Instr::Kind::IfBegin:
            case Instr::Kind::IfEnd:
                // Conditional-branch marker: deferral cannot defer measurements/resets inside a
                // shot-dependent branch (no bit values here). Refuse by name.
                refuse_if_blocks(in);
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

Circuit defer_measurements(const Circuit& in, DeferralMap* map, bool diagonal_copy) {
    return defer_impl(in, map, diagonal_copy, false);
}

Circuit defer_measurements_bruteforce_liveness(const Circuit& in, DeferralMap* map, bool diagonal_copy) {
    return defer_impl(in, map, diagonal_copy, true);
}

}  // namespace qeccore
