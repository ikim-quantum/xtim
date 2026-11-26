#pragma once
// free_dir — the compact Born direction (FreeDir; S2a, owner decision 8) and its views (plancache-design §3 B1): a
// direction read from a RecTable's vector or from a flat RecBlob's packed bytes. Shared by the Born trie and the arena.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qeccore {

// FreeDir (owner decision 8): Born direction j of a plan is stored ONLY in FREE-REGISTER form,
// Q_j = U†Λ_jU of the bare frame (U, eps, free) restricted to the bare's k free coordinates:
//   a[d], b[d] = Q.xbit(free[d]), Q.zbit(free[d])   (d = 0..k−1, bare free order)
//   phase      = Q.phase & 3
//   par        = Σ_q Q.zbit(q)·eps[q] (mod 2) over ALL wires of the bare frame
// Q has no X off the free set (Λ_j is LOGICAL: it commutes with every certified row), so this is
// exactly the data measure_pauli's case (b) decodes per step — y = a, g = (−1)^b, c = i^phase·
// (−1)^par on the plan's fixed free register — decoded ONCE per plan instead of once per shot and
// step. `hermitian` is today's per-step Hermiticity guard of the n-qubit rep, evaluated once at conversion.
struct FreeDir {
    std::vector<uint8_t> a, b;
    uint8_t phase = 0, par = 0, hermitian = 1;
};

// A direction by pointers (a RecTable's FreeDir or a RecBlob's packed bytes): what the trie kernels decode.
struct FreeDirView {
    const uint8_t* a = nullptr;
    const uint8_t* b = nullptr;
    int k = 0;
    uint8_t phase = 0, par = 0, hermitian = 1;
    FreeDirView() = default;
    FreeDirView(const FreeDir& f)   // NOLINT (implicit by design)
        : a(f.a.data()), b(f.b.data()), k((int)f.a.size()), phase(f.phase), par(f.par), hermitian(f.hermitian) {}
    FreeDir to_freedir() const {
        FreeDir d;
        d.a.assign(a, a + k);
        d.b.assign(b, b + k);
        d.phase = phase; d.par = par; d.hermitian = hermitian;
        return d;
    }
};
// The directions of a table: a RecTable's vector, or a blob's packed bytes (stride bytes each: phase, par,
// hermitian, a[k], b[k]). BornTrie takes this.
struct LamSpan {
    const FreeDir* v = nullptr;                      // a vector's data (nullptr: packed)
    const uint8_t* packed = nullptr;
    size_t stride = 0;
    int k = 0;
    int n = 0;
    LamSpan() = default;
    LamSpan(const std::vector<FreeDir>& L) : v(L.data()), n((int)L.size()) {}   // NOLINT (implicit by design)
    int size() const { return n; }
    FreeDirView at(int j) const {
        if (v) return FreeDirView(v[(size_t)j]);
        FreeDirView d;
        const uint8_t* p = packed + (size_t)j * stride;
        d.phase = p[0]; d.par = p[1]; d.hermitian = p[2];
        d.a = p + 3; d.b = p + 3 + k; d.k = k;
        return d;
    }
};

}  // namespace qeccore
