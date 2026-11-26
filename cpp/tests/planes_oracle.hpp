#pragma once
// planes_oracle.hpp — S2-T5: CertifiedGroupPlanes (sparse storage) against the pre-S2 dense planes kept as the oracle
// (DenseGroupPlanes::build_dense), field by field: dims, the column planes (the CSR index's words, the dense planes when
// kept), gens, rref (x, z, phase; canonical sparse form), pivcol, col_to_row, rank. Shared by test_planes_build.cpp
// and test_planes_sparse.cpp (include after check.hpp).
#include <utility>
#include <vector>

#include "qeccore/pauli_kernels.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// S2-T5: the pre-S2 dense planes, kept as the TEST ORACLE of the sparse build and of every reader built on it
// (cpp/tests/test_planes_build.cpp, test_planes_sparse.cpp): build_dense is the pre-T6 build verbatim (a dense column
// scatter and an RREF that scans every row per column), with the dense column planes and rows. (S2-FINALFIX F2: moved
// here from the engine, which never called it.)
struct DenseGroupPlanes {
    int n_qubits = 0, n_gens = 0, words = 0;
    std::vector<uint64_t> xcol, zcol;          // [n_qubits * words]; column q at offset q*words
    std::vector<Pauli>    gens;
    std::vector<Pauli>    rref;
    std::vector<int>      pivcol, col_to_row;
    int rank = 0;
    static DenseGroupPlanes build_dense(const std::vector<Pauli>& certified, int n_qubits);
    const uint64_t* col(int q, bool xblock) const { return (xblock ? xcol.data() : zcol.data()) + (size_t)q * words; }
};

// The pre-T6 build, verbatim: a dense column scatter and an RREF that scans every row per column — O(n_gens · n) each.
inline DenseGroupPlanes DenseGroupPlanes::build_dense(const std::vector<Pauli>& certified, int n_qubits) {
    // symplectic-column bit: c in [0,n) -> x_c, [n,2n) -> z_{c-n}
    auto sbit = [](const Pauli& P, int c, int n) { return c < n ? P.xbit(c) : P.zbit(c - n); };
    DenseGroupPlanes G;
    G.n_qubits = n_qubits;
    G.n_gens   = (int)certified.size();
    G.words    = (G.n_gens + 63) / 64;
    G.gens     = certified;

    // ── Columnar bit-planes: for each qubit q, pack bit i = gens[i].x/z(q). ──
    G.xcol.assign((size_t)n_qubits * G.words, 0);
    G.zcol.assign((size_t)n_qubits * G.words, 0);
    for (int i = 0; i < G.n_gens; ++i) {
        const Pauli& g = certified[i];
        const uint64_t bit = 1ULL << (i & 63);
        const int wo = i >> 6;
        for (int q = 0; q < n_qubits; ++q) {
            if (g.xbit(q)) G.xcol[(size_t)q * G.words + wo] |= bit;
            if (g.zbit(q)) G.zcol[(size_t)q * G.words + wo] |= bit;
        }
    }

    // ── GF(2) RREF of the (x|z) symplectic rows, exact phase carried per row. ──
    // Columns 0..n-1 = X-block, n..2n-1 = Z-block. Full reduced echelon (each pivot column has a
    // single 1, in its own row) so reduce() is a single pivot-local pass in ascending column order.
    const int two_n = 2 * n_qubits;
    std::vector<Pauli> work = certified;    // mutated in place (exact phases via pmul_into)
    G.col_to_row.assign(two_n, -1);
    int used = 0;
    for (int c = 0; c < two_n; ++c) {
        int sel = -1;
        for (int i = used; i < G.n_gens; ++i) if (sbit(work[i], c, n_qubits)) { sel = i; break; }
        if (sel < 0) continue;
        if (sel != used) std::swap(work[used], work[sel]);
        const Pauli piv = work[used];       // copy: pmul_into below mutates other rows
        for (int i = 0; i < G.n_gens; ++i) {
            if (i == used) continue;
            if (sbit(work[i], c, n_qubits)) pmul_into(work[i], piv);
        }
        G.col_to_row[c] = used;
        G.pivcol.push_back(c);
        ++used;
    }
    G.rank = used;
    G.rref.assign(work.begin(), work.begin() + used);
    return G;
}

}  // namespace qeccore

namespace planes_oracle {
using namespace qeccore;

inline bool same_pauli(const Pauli& a, const Pauli& b) { return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z; }

inline std::vector<SparsePauli> sparse_rows(const std::vector<Pauli>& v) {
    std::vector<SparsePauli> s;
    for (const Pauli& P : v) s.push_back(from_dense(P));
    return s;
}

inline void same_planes(const CertifiedGroupPlanes& A, const DenseGroupPlanes& B) {
    CHECK(A.n_qubits == B.n_qubits && A.n_gens == B.n_gens && A.words == B.words);
    if (!(A.n_qubits == B.n_qubits && A.n_gens == B.n_gens && A.words == B.words)) return;
    const int n = A.n_qubits, W = A.words;
    // The column planes: the index's words (ascending entries) == the oracle's planes; dense planes iff dense_cols.
    std::vector<uint64_t> col((size_t)W);
    for (int blk = 0; blk < 2; ++blk) {
        const CertifiedGroupPlanes::ColIndex& X = blk ? A.zidx : A.xidx;
        CHECK((int)X.off.size() == n + 1);
        if ((int)X.off.size() != n + 1) return;
        for (int q = 0; q < n; ++q) {
            std::fill(col.begin(), col.end(), 0);
            for (int p = X.off[(size_t)q]; p < X.off[(size_t)q + 1]; ++p) {
                if (p > X.off[(size_t)q]) CHECK(X.gen[(size_t)p - 1] < X.gen[(size_t)p]);
                col[(size_t)(X.gen[(size_t)p] >> 6)] |= 1ULL << (X.gen[(size_t)p] & 63);
            }
            const uint64_t* b = B.col(q, blk == 0);
            for (int w = 0; w < W; ++w) CHECK(col[(size_t)w] == b[w]);
            std::fill(col.begin(), col.end(), 0);
            A.xor_col(col.data(), q, blk == 0);
            for (int w = 0; w < W; ++w) CHECK(col[(size_t)w] == b[w]);
        }
    }
    if (A.dense_cols) { CHECK(A.xcol == B.xcol); CHECK(A.zcol == B.zcol); }
    else { CHECK(A.xcol.empty()); CHECK(A.zcol.empty()); }
    CHECK(A.gens.size() == B.gens.size());
    for (size_t i = 0; i < A.gens.size() && i < B.gens.size(); ++i) {
        CHECK(sparse_canonical(A.gens[i]));
        CHECK(same_pauli(to_dense(A.gens[i]), B.gens[i]));
    }
    CHECK(A.rank == B.rank);
    CHECK(A.pivcol == B.pivcol);
    CHECK(A.col_to_row == B.col_to_row);
    CHECK(A.rref.size() == B.rref.size());
    for (size_t r = 0; r < A.rref.size() && r < B.rref.size(); ++r) {
        CHECK(sparse_canonical(A.rref[r]));
        CHECK(same_pauli(to_dense(A.rref[r]), B.rref[r]));
    }
}

}  // namespace planes_oracle
