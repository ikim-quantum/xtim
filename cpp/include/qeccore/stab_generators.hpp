#pragma once
#include <vector>
#include "qeccore/pauli.hpp"
#include "qeccore/stab_state.hpp"
namespace qeccore {
std::vector<Pauli> stabilizer_generators(const StabState& s);
}
