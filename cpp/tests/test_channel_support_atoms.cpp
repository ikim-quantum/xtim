// test_channel_support_atoms.cpp — the propagation table's atom set (fix/z-channel-atoms).
//
//   ChannelSupport: a location builds exactly the X/Z atoms its channel fires with nonzero
//     probability (a Y needs both); an unbuilt atom is absent, never a reject, and reading it
//     through LocationEntry::atom() throws a named std::invalid_argument.
//   AnyPauli (forced faults): every atom is built; a zero-probability atom that leaves the class is
//     absent (rejected = true) instead of rejecting the circuit; a nonzero one still rejects.
//   Every atom built in both modes is identical (the fix changes no built atom).
#include "check.hpp"

#include <stdexcept>
#include <string>

#include "qeccore/propagation_table.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

Circuit circ(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    return ps.circuit;
}

bool throws_named(const LocationEntry& e, int qi, PauliBasis b) {
    try {
        (void)e.atom(qi, b, "t");
    } catch (const std::invalid_argument& ex) {
        return std::string(ex.what()).find("probability zero") != std::string::npos;
    }
    return false;
}

void same_atom(const PropAtom& a, const PropAtom& b) {
    CHECK(a.present && b.present);
    CHECK_EQ(a.rejected, b.rejected);
    CHECK_EQ(a.reject_gate_index, b.reject_gate_index);
    CHECK_EQ((bool)a.general, (bool)b.general);
    if (a.rejected || a.general) return;
    CHECK(a.c_prop.a == b.c_prop.a);
    CHECK(a.c_prop.v == b.c_prop.v);
    DiagPauliClifford x = a.c_prop, y = b.c_prop;
    x.finalize();
    y.finalize();
    CHECK(x.cache->b_pairs == y.cache->b_pairs);
}

// The review repro: Z_ERROR before T then H. Its X atom (probability 0) leaves the diagonal class.
void pure_z_before_t() {
    const Circuit c = circ("RX 0\nZ_ERROR(0.1) 0\nT 0\nH 0\nM 0\n");
    PropagationTable t = build_propagation_table(c, false, AtomSet::ChannelSupport);
    CHECK(t.all_in_class);
    CHECK(t.rejects.empty());
    CHECK_EQ((int)t.locations.size(), 1);
    const LocationEntry& e = t.locations[0];
    CHECK(!e.x_atom[0].present);
    CHECK(e.z_atom[0].present && !e.z_atom[0].rejected);
    CHECK(throws_named(e, 0, PauliBasis::X));
    CHECK(!throws_named(e, 0, PauliBasis::Z));

    // AnyPauli: the X atom is built, leaves the class, and becomes absent — not a circuit reject.
    PropagationTable a = build_propagation_table(c, false, AtomSet::AnyPauli);
    CHECK(a.all_in_class);
    CHECK(!a.locations[0].x_atom[0].present && a.locations[0].x_atom[0].rejected);
    CHECK(throws_named(a.locations[0], 0, PauliBasis::X));
    same_atom(a.locations[0].z_atom[0], e.z_atom[0]);

    // A NONZERO X there still rejects the whole circuit, in both modes.
    const Circuit cx = circ("RX 0\nX_ERROR(0.1) 0\nT 0\nH 0\nM 0\n");
    CHECK(!build_propagation_table(cx, false, AtomSet::ChannelSupport).all_in_class);
    CHECK(!build_propagation_table(cx, false, AtomSet::AnyPauli).all_in_class);
}

// Support per channel and slot, incl. PAULI_CHANNEL_2's first/second-qubit order and a qubit
// repeated across pairs of one instruction (unioned, so find_slot's first match carries it all).
void channel_supports() {
    struct Case { const char* line; const char* want; };   // want: per slot "xz" flags, '-' none
    const Case cases[] = {
        {"X_ERROR(0.1) 0", "x"},
        {"Z_ERROR(0.1) 0", "z"},
        {"Y_ERROR(0.1) 0", "b"},
        {"X_ERROR(0) 0", "-"},
        {"DEPOLARIZE1(0) 0", "-"},
        {"DEPOLARIZE1(0.1) 0", "b"},
        {"PAULI_CHANNEL_1(0,0,0.1) 0", "z"},
        {"PAULI_CHANNEL_1(0.1,0,0) 0", "x"},
        {"PAULI_CHANNEL_1(0,0.1,0) 0", "b"},
        {"PAULI_CHANNEL_1(0,0,0) 0", "-"},
        {"DEPOLARIZE2(0.1) 0 1", "bb"},
        {"DEPOLARIZE2(0) 0 1", "--"},
        // slot order IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ
        {"PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0) 0 1", "z-"},      // ZI
        {"PAULI_CHANNEL_2(0,0,0.1,0,0,0,0,0,0,0,0,0,0,0,0) 0 1", "-z"},      // IZ
        {"PAULI_CHANNEL_2(0.1,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0) 0 1", "bx"},    // IX, YI
        {"PAULI_CHANNEL_2(0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0) 0 1", "xz"},      // XZ
        {"PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0) 0 1 1 2", "zzz-"},  // ZI; q1 unioned
        // E/ELSE chains: ONE location whose slots are the chain's support in first-appearance
        // order; a term's factors mark its slots iff the term's CATEGORICAL probability is > 0.
        {"E(0.1) X0", "x"},
        {"E(0.1) Z2 X0\nELSE_CORRELATED_ERROR(0.2) Y1", "zxb"},               // support (2, 0, 1)
        {"E(0) X0\nELSE_CORRELATED_ERROR(0.2) Z1", "-z"},                     // p=0 term: no atom
        {"E(1) X0\nELSE_CORRELATED_ERROR(0.2) Z1", "x-"},                     // dead term after p=1
        {"E(0.1)\nELSE_CORRELATED_ERROR(0.2) Y2", "b"},                       // empty first term
        {"E(0.3) X0 X1\nELSE_CORRELATED_ERROR(0.3) Z1 Z2\nELSE_CORRELATED_ERROR(0) Y0 Y2", "xbz"},
    };
    for (const Case& k : cases) {
        const Circuit c = circ(std::string("R 0 1 2\n") + k.line + "\nM 0 1 2\n");
        const PropagationTable t = build_propagation_table(c, true, AtomSet::ChannelSupport);
        const PropagationTable a = build_propagation_table(c, true, AtomSet::AnyPauli);
        CHECK(t.all_in_class && a.all_in_class);
        const LocationEntry& e = t.locations[0];
        const std::string want = k.want;
        CHECK_EQ(want.size(), e.qubits.size());
        for (size_t i = 0; i < want.size(); ++i) {
            const bool wx = want[i] == 'x' || want[i] == 'b';
            const bool wz = want[i] == 'z' || want[i] == 'b';
            if (e.x_atom[i].present != wx || e.z_atom[i].present != wz)
                std::fprintf(stderr, "  support mismatch: %s slot %zu\n", k.line, i);
            CHECK_EQ(e.x_atom[i].present, wx);
            CHECK_EQ(e.z_atom[i].present, wz);
            if (wx) same_atom(e.x_atom[i], a.locations[0].x_atom[i]);
            if (wz) same_atom(e.z_atom[i], a.locations[0].z_atom[i]);
            CHECK(a.locations[0].x_atom[i].present && a.locations[0].z_atom[i].present);
        }
    }
}

}  // namespace

int main() {
    RUN(pure_z_before_t);
    RUN(channel_supports);
    REPORT();
}
