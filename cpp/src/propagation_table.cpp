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

const PropAtom& LocationEntry::atom(int qi, PauliBasis basis, const char* who) const {
    if (basis == PauliBasis::Y)                 // always-on: a Y is its X atom then its Z atom
        throw std::logic_error(std::string(who) + ": LocationEntry::atom takes X or Z (a Y factor is "
                               "read as its X atom then its Z atom)");
    const PropAtom& a = basis == PauliBasis::X ? x_atom[(size_t)qi] : z_atom[(size_t)qi];
    if (!a.present)
        throw std::invalid_argument(
            std::string(who) + ": a " + (basis == PauliBasis::X ? "X/Y" : "Z/Y") + " fault on deferred qubit " +
            std::to_string(qubits[(size_t)qi]) + " at deferred noise stream index " + std::to_string(stream_index) +
            " has probability zero in its channel" +
            (a.rejected ? " and its propagated error leaves the simulable class at deferred-stream gate index " +
                              std::to_string(a.reject_gate_index) + " (no atom for it)"
                        : " (no atom is built for it)"));
    return a;
}

// The channel's support per qubit slot: need_x[i] / need_z[i] iff some Pauli with nonzero
// probability has an X/Y (resp. Z/Y) factor on qubits[i]. Unioned over every slot holding the
// same qubit, so find_slot's first-match slot always carries everything that qubit can fire.
static void channel_support(const Instr& ins, std::vector<char>& need_x, std::vector<char>& need_z) {
    const size_t nq = ins.qubits.size();
    need_x.assign(nq, 0);
    need_z.assign(nq, 0);
    auto mark = [&](size_t i, PauliBasis b) {
        if (b == PauliBasis::X || b == PauliBasis::Y) need_x[i] = 1;
        if (b == PauliBasis::Z || b == PauliBasis::Y) need_z[i] = 1;
    };
    switch (ins.channel) {
        case NoiseChannel::X_ERROR:
        case NoiseChannel::Y_ERROR:
        case NoiseChannel::Z_ERROR: {
            const PauliBasis b = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                               : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z;
            if (ins.probs.at(0) > 0.0) for (size_t i = 0; i < nq; ++i) mark(i, b);
            break;
        }
        case NoiseChannel::DEPOLARIZE1:
        case NoiseChannel::PAULI_CHANNEL_1: {
            const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
            for (int t = 0; t < 3; ++t)
                if ((dep ? ins.probs.at(0) : ins.probs.at((size_t)t)) > 0.0)
                    for (size_t i = 0; i < nq; ++i) mark(i, pc1_slot_basis(t));
            break;
        }
        case NoiseChannel::DEPOLARIZE2:
        case NoiseChannel::PAULI_CHANNEL_2: {
            const bool dep = ins.channel == NoiseChannel::DEPOLARIZE2;
            for (int t = 0; t < 15; ++t) {
                if (!((dep ? ins.probs.at(0) : ins.probs.at((size_t)t)) > 0.0)) continue;
                for (size_t i = 0; i + 1 < nq; i += 2) {
                    if (const int c = pc2_first_code(t))  mark(i, pauli_code_basis(c));
                    if (const int c = pc2_second_code(t)) mark(i + 1, pauli_code_basis(c));
                }
            }
            break;
        }
        case NoiseChannel::CORRELATED: {            // E/ELSE chain: term t's product, if it can fire
            const std::vector<double> q = corr_outcome_probs(ins);
            for (int t = 0; t < corr_terms(ins); ++t) {
                if (!(q[(size_t)t] > 0.0)) continue;
                for (size_t i = 0; i < nq; ++i)
                    if (const int c = corr_code(ins, t, (int)i)) mark(i, pauli_code_basis(c));
            }
            break;
        }
        default:                                    // a new channel must declare its support here
            throw std::invalid_argument("propagation table: noise channel " +
                                        std::to_string((int)ins.channel) +
                                        " has no declared Pauli support (unsupported channel)");
    }
    for (size_t i = 0; i < nq; ++i)
        for (size_t j = 0; j < nq; ++j)
            if (ins.qubits[j] == ins.qubits[i]) { need_x[i] |= need_x[j]; need_z[i] |= need_z[j]; }
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

PropagationTable build_propagation_table(const Circuit& circ, bool ppr_retry, AtomSet atoms) {
    PropagationTable tbl;
    tbl.n = circ.n;
    for (int p = 0; p < (int)circ.stream.size(); ++p) {
        const Instr& ins = circ.stream[p];
        if (ins.kind != Instr::Kind::Noise) continue;
        LocationEntry entry;
        entry.stream_index = p;
        entry.qubits = ins.qubits;
        std::vector<char> need_x, need_z;
        channel_support(ins, need_x, need_z);
        // One atom per (qubit, basis), X before Z as always. A factor the channel cannot fire is
        // absent: never propagated (ChannelSupport), or propagated but never a reject (AnyPauli).
        auto build = [&](bool seedX, bool needed, int q) {
            PropResult r;
            if (!needed && atoms == AtomSet::ChannelSupport) { r.present = false; return r; }
            r = propagate_atom(circ, p, seedX ? DiagPauliClifford::X(circ.n, q) : DiagPauliClifford::Z(circ.n, q));
            if (r.rejected && ppr_retry)
                ppr_retry_atom(r, circ, p, seedX, q, tbl);
            if (r.rejected && !needed) { r.present = false; return r; }
            if (r.rejected) {
                tbl.all_in_class = false;
                tbl.rejects.push_back({p, q, seedX ? PauliBasis::X : PauliBasis::Z, r.reject_gate_index});
            }
            return r;
        };
        for (size_t i = 0; i < ins.qubits.size(); ++i) {
            entry.x_atom.push_back(build(true, need_x[i], ins.qubits[i]));
            entry.z_atom.push_back(build(false, need_z[i], ins.qubits[i]));
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
