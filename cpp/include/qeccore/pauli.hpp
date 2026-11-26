#pragma once
#include <cstdint>
#include <vector>
#include "qeccore/stab_state.hpp"
namespace qeccore {

// Pauli operator on n qubits: i^phase · X^x · Z^z (phase ∈ {0,1,2,3}).
struct Pauli {
    int n = 0;
    int phase = 0;
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;

    Pauli() = default;
    explicit Pauli(int n_) : n(n_), x((n_+63)/64, 0), z((n_+63)/64, 0) {}

    bool xbit(int q) const { return (x[q>>6] >> (q&63)) & 1ULL; }
    bool zbit(int q) const { return (z[q>>6] >> (q&63)) & 1ULL; }
    void setx(int q) { x[q>>6] |= (1ULL << (q&63)); }
    void setz(int q) { z[q>>6] |= (1ULL << (q&63)); }
    void flipx(int q){ x[q>>6] ^= (1ULL << (q&63)); }
    void flipz(int q){ z[q>>6] ^= (1ULL << (q&63)); }

    static int anticommute_bit(const Pauli& a, const Pauli& b);
    static bool commute(const Pauli& a, const Pauli& b) { return anticommute_bit(a,b)==0; }
    static Pauli multiply(const Pauli& a, const Pauli& b);
    int xz_overlap() const;
    void apply_to(StabState& s) const;
};

// The same operator i^phase · X^x · Z^z stored by support: x, z are the ascending, duplicate-free qubit
// indices (< n) of the set bits. Same phase convention as Pauli (stored verbatim, like Pauli's). For rows
// whose weight is far below n (frame rows, certified generators, record columns); the kernels against
// the dense Pauli live in pauli_kernels.hpp, each pinned to the dense routine (cpp/tests/test_sparse_pauli.cpp).
struct SparsePauli {
    int n = 0;
    int phase = 0;
    std::vector<int32_t> x;
    std::vector<int32_t> z;
};

SparsePauli from_dense(const Pauli& P);
Pauli to_dense(const SparsePauli& s);

}  // namespace qeccore
