#include "qeccore/feedback.hpp"
#include <cstddef>
#include "qeccore/deferral.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/pauli.hpp"
#include <cassert>

namespace qeccore {

Circuit strip_feedback(const Circuit& user) {
    Circuit out;
    out.n = user.n;
    out.stream.reserve(user.stream.size());
    for (const Instr& ins : user.stream)
        if (ins.kind != Instr::Kind::ControlledPauli) out.stream.push_back(ins);
    return out;
}

namespace {

// The probe circuit: each ControlledPauli is replaced by a deterministic single-qubit Pauli
// noise location (X/Y/Z_ERROR p=1) on its target wire, with a recorded order index. Every other
// instruction is copied verbatim. A ControlledPauli is a liveness-"use"-free instruction (a
// Gate/Measure/Observable triggers liveness, a Noise does not), so the probe deferral has the
// SAME abandoned/live decisions, fresh-wire allocation, and terminal-read order as the bare
// (feedback-removed) circuit — the resolved record indices line up 1:1.
// Each probe is identified by its ORDINAL among all Noise instructions in the probe circuit.
// Deferral copies Noise instructions through in their original relative order (it only moves
// Measures to the end and inserts measurement gadgets — it never reorders Noise among itself),
// so build_propagation_table's locations (one per deferred Noise, in deferred order) line up
// 1:1 with the probe circuit's Noise ordinals. We carry each probe's noise-ordinal to index
// table.locations directly (deferred stream indices shift, so an index match is unreliable).
struct Probe { int noise_ordinal; PauliBasis pauli; };

Circuit make_probe_circuit(const Circuit& user, std::vector<Probe>& probes) {
    Circuit out;
    out.n = user.n;
    out.stream.reserve(user.stream.size());
    int noise_ord = 0;
    for (const Instr& ins : user.stream) {
        if (ins.kind == Instr::Kind::Noise) { out.stream.push_back(ins); ++noise_ord; continue; }
        if (ins.kind != Instr::Kind::ControlledPauli) { out.stream.push_back(ins); continue; }
        Instr nz;
        nz.kind = Instr::Kind::Noise;
        nz.channel = ins.basis == PauliBasis::X ? NoiseChannel::X_ERROR
                   : ins.basis == PauliBasis::Y ? NoiseChannel::Y_ERROR
                   :                              NoiseChannel::Z_ERROR;
        nz.probs = {1.0};
        nz.qubits = ins.qubits;                 // the target qubit (wire-remapped by deferral)
        probes.push_back({noise_ord++, ins.basis});
        out.stream.push_back(std::move(nz));
    }
    return out;
}

}  // namespace

FeedbackPlan build_feedback_plan(const Circuit& user, int num_records, int num_observables) {
    FeedbackPlan plan;
    plan.MW = (num_records + 63) / 64;
    if (!circuit_has_feedback(user)) return plan;        // ok=true, has_feedback=false: dead pass
    // Only NON-crossing feedback (a Pauli byproduct) survives coherentization as a ControlledPauli;
    // coherentize_classically_controlled_feedback rewrites every feedback whose byproduct crosses a
    // non-Clifford gate into coherent gates upstream. The sign-relabel below thus only ever sees
    // Pauli-byproduct feedback (the non_pauli_byproduct guard is a defensive safety, not expected to
    // fire). The primary circuit-rejection criterion remains the !all_in_class propagation check.
    plan.has_feedback = true;

    // 1. Resolve each controlled-Pauli's rec[-k] control to an ABSOLUTE record index (the user
    //    circuit preserves measurement record order through deferral, so the absolute index = the
    //    packed-record index). Done on the USER stream by counting measurements.
    std::vector<int> control_record;     // one per ControlledPauli, in stream order
    {
        int meas = 0;
        for (const Instr& ins : user.stream) {
            if (ins.kind == Instr::Kind::Measure) { ++meas; continue; }
            if (ins.kind != Instr::Kind::ControlledPauli) continue;
            const int abs = meas - ins.control_rec_offset;   // rec[-k] = k-th record before here
            control_record.push_back(abs);
        }
    }

    // 2. Build the probe circuit, defer it, and propagate each probe Pauli to end-of-circuit.
    std::vector<Probe> probes;
    Circuit probe = make_probe_circuit(user, probes);
    Circuit deferred = defer_measurements(probe);
    PropagationTable table = build_propagation_table(deferred);   // ppr_retry OFF (feedback plan: diagonal class only; sampler opts in separately)
    if (!table.all_in_class) {
        plan.ok = false;
        plan.reject_gate_index = table.rejects.empty() ? -1 : table.rejects[0].reject_gate_index;
        plan.error = "classically-controlled Pauli leaves the simulable (diagonal+Pauli) class "
                     "after propagation";
        return plan;
    }

    // 3. The deferred terminal reads (record order) and the observable Paulis: the targets of the
    //    relabel. Terminal reads are the trailing Measure instructions (record index == position).
    struct TRead { int pauli; int qubit; };      // pauli 0:X 1:Y 2:Z
    std::vector<TRead> ms;
    std::vector<Pauli> obsP;
    for (const Instr& ins : deferred.stream) {
        if (ins.kind == Instr::Kind::Measure) {
            const int pidx = ins.basis == PauliBasis::X ? 0 : ins.basis == PauliBasis::Y ? 1 : 2;
            ms.push_back({pidx, ins.qubits[0]});
        } else if (ins.kind == Instr::Kind::Observable) {
            Pauli P(deferred.n);
            int ny = 0;
            for (const PauliTerm& t : ins.obs) {
                if (t.p == PauliBasis::X) P.setx(t.qubit);
                else if (t.p == PauliBasis::Y) { P.setx(t.qubit); P.setz(t.qubit); ++ny; }
                else P.setz(t.qubit);
            }
            P.phase = ny & 3;
            obsP.push_back(std::move(P));
        }
    }
    // Record count must match the bare pipeline's (1:1 alignment is the whole point).
    assert((int)ms.size() == num_records && "feedback: probe record count != bare record count");
    assert((int)obsP.size() == num_observables && "feedback: probe observable count mismatch");
    (void)num_observables;

    // 4. Per probe: the end-of-circuit error D (its atom in the table). The X^V part flips a
    //    terminal read iff that read has Z content (Y/Z basis) on a V-qubit (an X read commutes
    //    with X); the S-power part rotates a non-Z read's basis (X<->Y) which also flips that
    //    read's outcome when the rotation carries a sign. `esign` bit r = D's X^V anticommutes
    //    with observable r's Pauli (parity of V over r's Z support) — the same εᵢ mechanism the
    //    sampler already uses for noise. (D's diagonal commutes with every observable's read,
    //    contributing no expectation sign; for a single-qubit seed Pauli the propagated D's
    //    interaction with the terminal reads is captured by V plus the per-wire S-power.)
    for (size_t i = 0; i < probes.size(); ++i) {
        const Probe& pr = probes[i];
        assert(pr.noise_ordinal >= 0 && pr.noise_ordinal < (int)table.locations.size() &&
               "probe noise ordinal out of range of the propagation table");
        const LocationEntry* loc = &table.locations[pr.noise_ordinal];
        assert(loc->qubits.size() == 1 && "probe location not single-qubit");
        DiagPauliClifford D = DiagPauliClifford::identity(deferred.n);
        if (pr.pauli == PauliBasis::X || pr.pauli == PauliBasis::Y) D = D.then(loc->x_atom[0].c_prop);
        if (pr.pauli == PauliBasis::Z || pr.pauli == PauliBasis::Y) D = D.then(loc->z_atom[0].c_prop);
        // A pure-Pauli seed crossing only Clifford gates propagates to a pure Pauli P (no CZ layer):
        // x[q] = v[q], z[q] = (a[q]>>1)&1 (a∈{0,2}; a=2 -> Z, with v -> Y). But a controlled-Pauli
        // that crosses a NON-Clifford gate (e.g. a transversal T) propagates to a non-Pauli Clifford:
        // an ODD S-power (a[q] odd) and/or a CZ arm (B nonempty). Such feedback is COHERENTIZED
        // upstream (feedback_crosses_nonclifford flags it), so it never reaches this relabel — the
        // guard below is a defensive LOUD reject, NOT expected to fire on a well-formed circuit. The
        // sign-flip model (mflips / esign) assumes a Pauli byproduct; a non-Pauli byproduct would
        // ROTATE <X̄> toward <Ȳ> rather than flip a sign, so refusing loudly is correct.
        bool non_pauli_byproduct = false;
        for (int q = 0; q < deferred.n && !non_pauli_byproduct; ++q)
            if (D.a[q] & 1) non_pauli_byproduct = true;  // odd S-power
        for (int bi = 0; bi < D.B.dim() && !non_pauli_byproduct; ++bi) {
            const uint64_t* brow = D.B.row(bi);
            for (int bw = 0; bw < D.B.words(); ++bw)
                if (brow[bw]) { non_pauli_byproduct = true; break; }  // CZ arm
        }
        if (non_pauli_byproduct) {
            plan.ok = false;
            plan.reject_gate_index = loc->stream_index;
            plan.error = "classically-controlled Pauli feedback crosses a non-Clifford gate but "
                         "reached the sign-relabel: the byproduct propagates to a non-Pauli Clifford "
                         "(odd S-power or CZ arm). This should have been coherentized upstream; "
                         "refusing rather than silently approximating.";
            return plan;
        }
        Pauli P(deferred.n);
        for (int q = 0; q < deferred.n; ++q) {
            if (D.v[q]) P.setx(q);
            if ((D.a[q] >> 1) & 1) P.setz(q);  // a is even here (guarded above): a=2 -> Z
        }

        FeedbackOp op;
        op.control_record = control_record[i];
        op.mflips.assign(plan.MW, 0);
        // P flips a terminal read iff P ANTICOMMUTES with that read's Pauli (symplectic form):
        //   X read: P.z;  Y read: P.x ^ P.z;  Z read: P.x.
        for (int j = 0; j < (int)ms.size(); ++j) {
            const int q = ms[j].qubit;
            const int px = P.xbit(q) ? 1 : 0, pz = P.zbit(q) ? 1 : 0;
            const int anti = ms[j].pauli == 0 ? pz : ms[j].pauli == 1 ? (px ^ pz) : px;
            if (anti) op.mflips[j >> 6] ^= 1ull << (j & 63);
        }
        // P contributes a SIGN to expectation r iff P anticommutes with observable r's Pauli A:
        //   symp(P, A) = parity over q of (P.x & A.z) ^ (P.z & A.x).
        for (size_t r = 0; r < obsP.size(); ++r) {
            int par = 0;
            for (int q = 0; q < deferred.n; ++q) {
                const int ax = obsP[r].xbit(q) ? 1 : 0, az = obsP[r].zbit(q) ? 1 : 0;
                const int px = P.xbit(q) ? 1 : 0, pz = P.zbit(q) ? 1 : 0;
                par ^= (px & az) ^ (pz & ax);
            }
            if (par) op.esign |= 1ull << r;
        }
        // Causality / triangularity guard: the control record must PRECEDE every record this op
        // flips (a Pauli can only flip a LATER measurement). A violation is an acausal/cyclic
        // feedback (malformed) — reject loudly rather than relabel incorrectly.
        for (int j = 0; j <= op.control_record && j < (int)ms.size(); ++j)
            if ((op.mflips[j >> 6] >> (j & 63)) & 1) {
                plan.ok = false;
                plan.reject_gate_index = loc->stream_index;
                plan.error = "acausal classically-controlled Pauli feedback: a controlled-Pauli "
                             "flips a record at or before its own control record";
                return plan;
            }
        plan.ops.push_back(std::move(op));
    }
    return plan;
}

}  // namespace qeccore
