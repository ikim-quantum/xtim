#pragma once
#include <complex>
#include <memory>
#include <vector>
#include "qeccore/exact_phase.hpp"

namespace qeccore {

struct Pauli;

// Audit counter: total number of AffineState::inner_product (pairwise state-overlap)
// evaluations. The closed-form (overlap-free) measurement collapse must NOT increment
// this in the release path; the test snapshots it around a measurement and asserts 0.
extern unsigned long long g_inner_product_calls;

struct StabState {
    virtual ~StabState() = default;
    virtual int n() const = 0;
    virtual void apply_h(int q)   = 0;
    virtual void apply_s(int q)   = 0;
    virtual void apply_sdg(int q) = 0;
    virtual void apply_x(int q)   = 0;
    virtual void apply_y(int q)   = 0;
    virtual void apply_z(int q)   = 0;
    virtual void apply_cx(int c, int t) = 0;
    virtual void apply_cz(int c, int t) = 0;
    virtual ExactPhase phase() const = 0;
    virtual std::unique_ptr<StabState> clone() const = 0;
    virtual ExactPhase inner_product(const StabState& other) const = 0;
    // Project onto the `outcome`-eigenspace of single-qubit Pauli `pauli`
    // (0=X,1=Y,2=Z) on qubit q. Mutates *this to the collapsed stabilizer state and
    // returns the amplitude α (ExactPhase): zero ⇒ this term has no weight in that
    // eigenspace (caller drops it), else 1 or 2^(−1/2).
    virtual ExactPhase pauli_project(int pauli, int q, int outcome) = 0;
    // Exact single-state diagonal expectation ⟨φ|P|φ⟩ ∈ {+1, −1, 0}.
    // pauli: 0=X, 1=Y, 2=Z. Does NOT mutate *this.
    virtual int single_pauli_expectation(int pauli, int q) const = 0;
    virtual std::vector<std::complex<double>> to_statevector() const = 0;
};

}  // namespace qeccore
