#include "qeccore/propagation_table.hpp"
#include "qeccore/clifford_conjugate.hpp"
#include "qeccore/ppr_residual.hpp"
#include <cassert>
#include <stdexcept>
#include <string>

namespace qeccore {

std::pair<const LocationEntry*, int>
PropagationTable::find_slot(int stream_index, int qubit, const char* who) const {
    const LocationEntry* loc = nullptr;
    for (const LocationEntry& e : locations)
        if (e.stream_index == stream_index) { loc = &e; break; }
    if (!loc)                                   // always-on: public API, null-deref otherwise
        throw std::logic_error(std::string(who) + ": fired location not in the propagation table");
    int qi = -1;
    for (int i = 0; i < (int)loc->qubits.size(); ++i) if (loc->qubits[i] == qubit) { qi = i; break; }
    if (qi < 0)
        throw std::logic_error(std::string(who) + ": fired qubit not in the table's location entry");
    return {loc, qi};
}

PropResult propagate_atom(const Circuit& circ, int from_index, const DiagPauliClifford& seed) {
    PropResult r;
    r.c_prop = seed;
    for (int k = from_index + 1; k < (int)circ.stream.size(); ++k) {
        const Instr& ins = circ.stream[k];
        if (ins.kind != Instr::Kind::Gate) continue;        // Noise/Measure/Reset/Observable transparent
        assert(!ins.targets.empty() && "canonical IR: a Gate instruction must carry arity-sized targets");
        int a = ins.targets[0];
        int b = ins.targets.size() > 1 ? ins.targets[1] : -1;
        int c = ins.targets.size() > 2 ? ins.targets[2] : -1;
        if (!conjugate_by_gate(r.c_prop, ins.gate, a, b, c)) {
            r.rejected = true;
            r.reject_gate_index = k;
            return r;
        }
    }
    return r;
}

// Helper: attempt PPR retry for one atom (seedX selects X vs Z seed).
// If the PPR residual propagates the entire tail without rejection, sets rx.rejected=false,
// rx.reject_gate_index=-1, and rx.general to the full error tableau.
// If the PPR residual also rejects, updates rx.reject_gate_index to the PPR reject position.
static void ppr_retry_atom(PropResult& rx, const Circuit& circ, int from_index,
                           bool seedX, int q, PropagationTable& tbl) {
    PprResidual Rpr = seedX ? ppr_seed_x(circ.n, q) : ppr_seed_z(circ.n, q);
    bool ppr_ok = true;
    int ppr_reject_idx = -1;
    for (int k = from_index + 1; k < (int)circ.stream.size() && ppr_ok; ++k) {
        const Instr& ins2 = circ.stream[k];
        if (ins2.kind != Instr::Kind::Gate) continue;
        ppr_ok = Rpr.propagate(ins2.gate, ins2.targets);
        if (!ppr_ok) ppr_reject_idx = k;
    }
    if (ppr_ok) {
        rx.rejected = false;
        rx.reject_gate_index = -1;
        rx.general = std::make_shared<CliffordTableau>(Rpr.to_error_tableau());
        rx.ppr = std::make_shared<PprResidual>(std::move(Rpr));  // V2: keep the axis list for the twirl law
        tbl.has_general = true;
    } else {
        rx.reject_gate_index = ppr_reject_idx;
        // rx.rejected stays true
    }
}

PropagationTable build_propagation_table(const Circuit& circ, bool ppr_retry) {
    PropagationTable tbl;
    tbl.n = circ.n;
    for (int p = 0; p < (int)circ.stream.size(); ++p) {
        const Instr& ins = circ.stream[p];
        if (ins.kind != Instr::Kind::Noise) continue;
        LocationEntry entry;
        entry.stream_index = p;
        entry.qubits = ins.qubits;
        for (int q : ins.qubits) {
            PropResult rx = propagate_atom(circ, p, DiagPauliClifford::X(circ.n, q));
            if (rx.rejected && ppr_retry)
                ppr_retry_atom(rx, circ, p, true, q, tbl);
            entry.x_atom.push_back(rx);
            if (rx.rejected) {
                tbl.all_in_class = false;
                tbl.rejects.push_back({p, q, PauliBasis::X, rx.reject_gate_index});
            }
            PropResult rz = propagate_atom(circ, p, DiagPauliClifford::Z(circ.n, q));
            if (rz.rejected && ppr_retry)
                ppr_retry_atom(rz, circ, p, false, q, tbl);
            entry.z_atom.push_back(rz);
            if (rz.rejected) {
                tbl.all_in_class = false;
                tbl.rejects.push_back({p, q, PauliBasis::Z, rz.reject_gate_index});
            }
        }
        tbl.locations.push_back(std::move(entry));
    }
    // has_general is only meaningful when the circuit is fully in-class: a partially-accepted
    // table is rejected by the caller, so general residuals in it are unused. Clear the flag
    // when all_in_class is false to avoid misleading callers.
    if (!tbl.all_in_class) tbl.has_general = false;
    return tbl;
}

}  // namespace qeccore
