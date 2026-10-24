#pragma once
#include <memory>
#include <vector>
#include "qeccore/clifford_op.hpp"        // DiagPauliClifford
#include "qeccore/clifford_tableau.hpp"   // CliffordTableau (for PropResult::general)
#include "qeccore/circuit_ir.hpp"         // Circuit, Instr, PauliBasis
#include "qeccore/ppr_residual.hpp"       // PprResidual (V2: PropResult::ppr axis list)

namespace qeccore {

// Result of pushing one seed error to the end of the circuit.
struct PropResult {
    DiagPauliClifford c_prop;   // valid iff !rejected (the end-of-circuit error)
    bool rejected = false;
    int  reject_gate_index = -1;  // stream index of the gate that left the class
    // Non-null iff a PPR retry accepted this atom when the standard path rejected:
    // holds the full error tableau E = pauli · (Π rots Clifford) as a CliffordTableau.
    std::shared_ptr<const CliffordTableau> general;
    // V2 (2026-07-15, additive): the accepted PPR retry's residual itself — the AXIS LIST the
    // twirl sampler's PPR plan family consumes (the tableau above erases the axes). Non-null
    // exactly when `general` is.
    std::shared_ptr<const PprResidual> ppr;
    // False iff this atom is absent: its seed Pauli factor has probability ZERO in the location's
    // channel (e.g. the X atom of a Z_ERROR), so no shot can fire it, and the table either never
    // built it (AtomSet::ChannelSupport) or built it and it left the class (AtomSet::AnyPauli;
    // then `rejected` is also set). An absent atom never rejects the circuit and must never be
    // read — LocationEntry::atom() refuses it by name.
    bool present = true;
};

// Propagate `seed` through every Gate instruction at stream index > from_index, in order,
// via conjugate_by_gate. Non-Gate instructions are transparent. Stops and reports reject the
// moment a gate takes the object out of the diagonal+Pauli class.
PropResult propagate_atom(const Circuit& circ, int from_index, const DiagPauliClifford& seed);

// A propagated basis error stored in the table. Identical in content to PropResult (a stored
// propagation result), so it aliases it; kept as a distinct name for call-site readability.
using PropAtom = PropResult;

// All atoms for one Noise location.
// ATOM SET = the channel's support: the X atom of a qubit is built iff some Pauli with nonzero
// probability in the channel has an X or Y factor on that qubit, the Z atom iff some has a Z or Y
// factor (a Y is composed as X then Z). Unbuilt slots hold a placeholder with present == false.
struct LocationEntry {
    int stream_index = -1;        // index of the Noise instruction in the stream
    std::vector<int> qubits;      // qubits the channel touches
    std::vector<PropAtom> x_atom; // one per qubit (parallel to `qubits`); read via atom()
    std::vector<PropAtom> z_atom; // one per qubit (parallel to `qubits`); read via atom()

    // The X (basis X) or Z (basis Z) atom of qubit slot `qi`. Throws std::invalid_argument
    // prefixed with `who` if that atom is absent (its factor has probability zero here).
    const PropAtom& atom(int qi, PauliBasis basis, const char* who) const;
};

// A single rejecting atom, for the whole-circuit reject diagnostic.
struct RejectInfo {
    int location_index = -1;      // stream index of the Noise instruction
    int qubit = -1;
    PauliBasis basis = PauliBasis::X;
    int reject_gate_index = -1;   // gate that left the class
};

// The per-location propagation table. all_in_class is false iff any atom rejected,
// in which case the WHOLE circuit is rejected and `rejects` lists every offending atom.
// has_general is true iff at least one atom was accepted via the PPR retry path and
// carries a non-null PropResult::general tableau.
struct PropagationTable {
    int n = 0;
    std::vector<LocationEntry> locations;
    bool all_in_class = true;
    std::vector<RejectInfo> rejects;
    bool has_general = false;

    // Locate the (LocationEntry, qubit-slot) a fired (stream_index, qubit) references; throws
    // std::logic_error prefixed with `who` if absent. Shared by compose_fired (sampler.cpp) and
    // the framed sampler's gather_atoms_ — one lookup, one error contract.
    std::pair<const LocationEntry*, int> find_slot(int stream_index, int qubit,
                                                   const char* who) const;
};

// Which atoms a table builds.
//   ChannelSupport (the samplers, DEM export, feedback plan): exactly the channel's support —
//     a factor with probability zero is never fired, so its atom is absent and can never reject
//     the circuit.
//   AnyPauli (the forced-fault program: ANY Pauli may be forced at a site, its probability is
//     irrelevant — docs/xtim_forced_faults.md): every X/Z atom is built, as before; a
//     zero-probability atom that leaves the class (after the PPR retry, when on) does not reject
//     the circuit but is marked absent (rejected = true), so forcing it is refused by name.
enum class AtomSet { ChannelSupport, AnyPauli };

// Build the table: for each Noise instruction and each touched qubit, propagate X_q and Z_q per
// `atoms` (above).
// If ppr_retry is true, any atom rejected by the standard DiagPauliClifford path is
// re-run with PprResidual; if the PPR residual propagates through the whole tail without
// rejection the atom is accepted and its full error tableau stored in PropResult::general.
PropagationTable build_propagation_table(const Circuit& circ, bool ppr_retry = false,
                                         AtomSet atoms = AtomSet::ChannelSupport);

}  // namespace qeccore
