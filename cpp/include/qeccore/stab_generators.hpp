#pragma once
#include <vector>
#include "qeccore/pauli.hpp"
#include "qeccore/stab_state.hpp"
namespace qeccore {
std::vector<Pauli> stabilizer_generators(const StabState& s);
// S2-T3: the same generators (bits, phases, order) stored by support — the compact bare-state build reads the
// isolated Z_q rows and the support split off them without an n-word row per generator.
std::vector<SparsePauli> stabilizer_generators_sparse(const StabState& s);
}
