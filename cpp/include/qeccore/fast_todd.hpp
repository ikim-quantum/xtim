#pragma once
// FastTODD T-count reduction (Vivien Vandaele, "Lower T-count with faster algorithms",
// Quantum 9, 1860 (2025); arXiv:2407.08695). A faithful C++ port of the reference Rust
// implementation (github.com/VivienVandaele/quantum_circuit_optimization, src/t_opt.rs):
// proper + kernel + tohpe + fast_todd, over the multiset-of-parity-columns phase-polynomial
// representation. Deterministic, polynomial time — replaces the randomized greedy TODD descent
// in the via-TODD conditioning path.
//
// Input/output: a multiset of parity COLUMNS, each a `nb_qubits`-bit mask (uint32; nb_qubits <= 32),
// one per T gate.  fast_todd_reduce returns an equivalent (same cubic phase polynomial up to
// Clifford) but smaller multiset.  For the via-TODD conditioning we need only the column DIRECTIONS
// (their span is the conditioning subspace), so the mod-8 Clifford bookkeeping is intentionally not
// reconstructed here.
#include <cstdint>
#include <vector>

namespace qeccore {

std::vector<uint32_t> fast_todd_reduce(std::vector<uint32_t> columns, int nb_qubits);

}  // namespace qeccore
