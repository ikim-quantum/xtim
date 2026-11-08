#include "qeccore/stab_disentangle.hpp"
#include <cstddef>
#include "qeccore/pauli_kernels.hpp"
#include <stdexcept>
#include <cassert>

namespace qeccore {

namespace {

constexpr int DG_H = 0, DG_S = 1, DG_SDG = 2, DG_X = 3, DG_Y = 4, DG_Z = 5,
              DG_CX = 6, DG_CZ = 7;

// Single-Pauli conjugation rules M <- G M G†  (same as ref_io.cpp's static helpers)
static void conj_h(Pauli& M, int q) {
    bool a = M.xbit(q), b = M.zbit(q);
    if (a != b) { M.flipx(q); M.flipz(q); }
    if (a && b) M.phase = (M.phase + 2) & 3;
}
static void conj_s(Pauli& M, int q) {
    if (M.xbit(q)) { M.flipz(q); M.phase = (M.phase + 1) & 3; }
}
static void conj_x(Pauli& M, int q) {
    if (M.zbit(q)) M.phase = (M.phase + 2) & 3;
}
static void conj_cx(Pauli& M, int c, int t) {
    if (M.xbit(c)) M.flipx(t);
    if (M.zbit(t)) M.flipz(c);
}
static void conj_cz(Pauli& M, int c, int t) {
    bool xc = M.xbit(c), xt = M.xbit(t);
    if (xc) M.flipz(t);
    if (xt) M.flipz(c);
    if (xc && xt) M.phase = (M.phase + 2) & 3;
}

static void conj_gate(Pauli& M, const DisentangleGate& g) {
    switch (g.op) {
        case DG_H:   conj_h(M, g.a);        break;
        case DG_S:   conj_s(M, g.a);        break;
        case DG_SDG:  // S†: a'=a, b'=a^b; dp = 3a
            if (M.xbit(g.a)) { M.flipz(g.a); M.phase = (M.phase + 3) & 3; }
            break;
        case DG_X:   conj_x(M, g.a);        break;
        case DG_Y:   // Y = XZ: X then Z conjugation
            if (M.zbit(g.a)) M.phase = (M.phase + 2) & 3;
            if (M.xbit(g.a)) M.phase = (M.phase + 2) & 3;
            break;
        case DG_Z:   // Z: dp = 2a
            if (M.xbit(g.a)) M.phase = (M.phase + 2) & 3;
            break;
        case DG_CX:  conj_cx(M, g.a, g.b); break;
        case DG_CZ:  conj_cz(M, g.a, g.b); break;
        default: throw std::logic_error("conj_gate(DisentangleGate): bad op");
    }
}

// GF(2) row vector over n bits, stored in ceil(n/64) uint64_t words.
// Word-packed: bit j lives in word j/64, bit j%64.  (Row XOR: qeccore::xor_into.)
static void set_bit(std::vector<uint64_t>& row, int j) {
    row[(size_t)j / 64] |= (uint64_t)1 << (j % 64);
}

static int get_bit(const std::vector<uint64_t>& row, int j) {
    return (int)((row[(size_t)j / 64] >> (j % 64)) & 1);
}

}  // namespace

// Reduce n Pauli stabilizer generators to {+Z_0, ..., +Z_{n-1}} by conjugating with
// H/S/X/CX/CZ; the recorded gate list G maps |stabilized_state> -> e^{i theta}|0...0>.
// Step q fixes row q to +Z_q and clears column q from the remaining rows.
//
// Row permutation tracking: `row_of` is the ORIGINAL input row index that is now at
// position i after swaps. After reduction, the original generator `a` (= row_of[q] at
// the moment step q's pivot was row a) is encoded at qubit q. The returned `perm_out`
// (if non-null) satisfies: syndrome bit for original generator `a` = f(disentangled, perm[a])
// where perm[a] = the qubit that original row a was reduced to = q_of[a].
//
// GF(2) transformation tracking (trans_out): row multiply steps mix original generators.
// trans[a][j] = 1 iff original generator j was included in the row assigned to generator a.
// After applying all gates, bit perm[a] of the disentangled b vector equals
//   XOR_{j: trans[a][j]=1} fs[j]    (a XOR of original syndrome bits).
// Use recover_syndromes() to invert this and obtain individual fs[a] values.
std::vector<DisentangleGate> disentangle_from_generators(
    std::vector<Pauli> rows,
    std::vector<int>* perm_out,
    std::vector<std::vector<uint64_t>>* trans_out) {
    const int n = (int)rows.size();
    if (n == 0) {
        if (perm_out) perm_out->clear();
        if (trans_out) trans_out->clear();
        return {};
    }

    const int W = (n + 63) / 64;  // words per row

    // row_of[i] = original index of the generator currently at position i
    std::vector<int> row_of(n);
    for (int i = 0; i < n; ++i) row_of[i] = i;
    // perm[a] = qubit that original generator a was reduced to (filled as we go)
    std::vector<int> perm(n, -1);

    // combo[i] = GF(2) bitmask: which original generators have been combined into current rows[i].
    // Initially each row is just its own generator.
    std::vector<std::vector<uint64_t>> combo(n, std::vector<uint64_t>(W, 0));
    for (int i = 0; i < n; ++i) set_bit(combo[i], i);

    // trans[a] = combo[q] at the moment perm[a] = q was assigned.
    // Indexed by original generator a (not by qubit q).
    // Built up as each generator's qubit is assigned.
    std::vector<std::vector<uint64_t>> trans(n, std::vector<uint64_t>(W, 0));

    std::vector<DisentangleGate> gates;
    auto emit = [&](int op, int a, int b = -1) {
        DisentangleGate g{op, a, b};
        for (Pauli& M : rows) conj_gate(M, g);
        gates.push_back(g);
    };

    for (int q = 0; q < n; ++q) {
        // Fast path: a remaining row already EXACTLY +-Z_q
        {
            int zrow = -1;
            for (int i = q; i < n && zrow < 0; ++i) {
                const Pauli& M = rows[i];
                if (!M.zbit(q) || M.xbit(q) || (M.phase & 1)) continue;
                bool lone = true;
                for (int r = 0; lone && r < n; ++r) {
                    if (M.xbit(r)) lone = false;
                    if (r != q && M.zbit(r)) lone = false;
                }
                if (lone) zrow = i;
            }
            if (zrow >= 0) {
                if (zrow != q) {
                    std::swap(rows[q], rows[zrow]);
                    std::swap(row_of[q], row_of[zrow]);
                    std::swap(combo[q], combo[zrow]);
                }
                // Assign: original generator row_of[q] maps to qubit q.
                perm[row_of[q]] = q;
                trans[row_of[q]] = combo[q];
                if (rows[q].phase == 2) emit(DG_X, q);
                for (int i = q + 1; i < n; ++i) {
                    if (rows[i].zbit(q)) {
                        rows[i] = Pauli::multiply(rows[i], rows[q]);
                        xor_into(combo[i], combo[q]);  // track GF(2) mixing
                    }
                }
                continue;
            }
        }
        // Pivot: a remaining row with X (else Z, after an H) content at qubit q
        int piv = -1;
        for (int i = q; i < n; ++i)
            if (rows[i].xbit(q)) { piv = i; break; }
        if (piv < 0) {
            for (int i = q; i < n; ++i)
                if (rows[i].zbit(q)) { piv = i; break; }
            if (piv < 0)
                throw std::runtime_error("disentangle_from_generators: no generator touches qubit");
            emit(DG_H, q);   // Z at q -> X at q in the pivot row
        }
        if (piv != q) {
            std::swap(rows[q], rows[piv]);
            std::swap(row_of[q], row_of[piv]);
            std::swap(combo[q], combo[piv]);
        }
        // Assign: original generator row_of[q] maps to qubit q.
        perm[row_of[q]] = q;
        trans[row_of[q]] = combo[q];

        for (int r = 0; r < n; ++r)  // clear X off-pivot
            if (r != q && rows[q].xbit(r)) emit(DG_CX, q, r);
        if (rows[q].zbit(q)) emit(DG_S, q);  // Y_q -> +-X_q
        for (int r = 0; r < n; ++r)  // clear Z off-pivot
            if (r != q && rows[q].zbit(r)) emit(DG_CZ, q, r);
        emit(DG_H, q);   // +-X_q -> +-Z_q
        if (rows[q].phase == 2) emit(DG_X, q);  // -Z_q -> +Z_q

        // Defensive check: row q must now be EXACTLY +Z_q
        bool ok = rows[q].phase == 0 && rows[q].zbit(q);
        for (int r = 0; ok && r < n; ++r) {
            if (rows[q].xbit(r)) ok = false;
            if (r != q && rows[q].zbit(r)) ok = false;
        }
        if (!ok) throw std::runtime_error("disentangle_from_generators: row reduction failed");

        for (int i = q + 1; i < n; ++i) {  // multiply out Z_q from remaining rows
            if (rows[i].zbit(q)) {
                rows[i] = Pauli::multiply(rows[i], rows[q]);
                xor_into(combo[i], combo[q]);  // track GF(2) mixing
            }
        }
    }
    if (perm_out) *perm_out = perm;
    if (trans_out) *trans_out = trans;
    return gates;
}

// Recover individual syndrome bits from the disentangled state.
//
// After applying the disentangle circuit, the b-offset at qubit perm[a] holds
//   b[perm[a]] = XOR_{j: trans[a][j]=1} fs[j]
// where fs[j] is the syndrome of original generator j.
//
// Because the row multiplies only affect LATER rows (i > q at step q), generator a is
// combined only with generators that were assigned to qubits BEFORE perm[a] in step order.
// This means we can recover individual syndromes by forward substitution in step order
// (processing generators in the order they were assigned, i.e., sorted by perm[a]).
//
// Step-order recovery:
//   For each generator a in ascending perm[a] order:
//     raw[a] = b[perm[a]]  (the disentangled bit)
//     fs[a]  = raw[a] XOR (XOR_{j != a: trans[a][j]=1} fs[j])
//            = raw[a] XOR (XOR_{j: trans[a][j]=1 AND j was processed before a} fs[j])
// The second form is valid because trans[a] only has bits set for generators j whose
// perm[j] < perm[a] (they were processed at earlier steps q' < q).
void recover_syndromes(int n, const AffineState& dis_ray,
                       const std::vector<int>& perm,
                       const std::vector<std::vector<uint64_t>>& trans,
                       std::vector<uint8_t>& fs) {
    fs.assign(n, 0);
    if (n == 0) return;

    // Build step_order[q] = the generator a s.t. perm[a] = q, for q = 0..n-1.
    std::vector<int> step_order(n, -1);
    for (int a = 0; a < n; ++a) {
        if (perm[a] >= 0 && perm[a] < n) step_order[perm[a]] = a;
    }

    // Forward substitution: process in order q = 0, 1, ..., n-1.
    for (int q = 0; q < n; ++q) {
        int a = step_order[q];
        if (a < 0) continue;  // shouldn't happen for a valid disentangle

        // Raw bit at qubit q in the disentangled state.
        // AffineState b-offset: b[q] is the q-th bit of the base point.
        // For a k=0 stabilizer state post-disentangle, this is deterministic.
        // Use z_expectation or directly read b.
        // z_expectation returns +1 (bit=0) or -1 (bit=1) if Z-deterministic.
        int ze = dis_ray.z_expectation(q);
        assert(ze != 0 && "recover_syndromes: non-deterministic qubit post-disentangle (invalid generator set)");
        uint8_t raw = (ze == -1) ? 1 : 0;

        // Undo contributions from generators already solved (earlier steps).
        // trans[a][j] = 1 means original gen j was mixed into this row.
        // Those j have perm[j] < q (processed earlier), so fs[j] is known.
        uint8_t xor_acc = 0;
        const std::vector<uint64_t>& row = trans[a];
        for (int j = 0; j < n; ++j) {
            if (j == a) continue;
            if (get_bit(row, j)) xor_acc ^= fs[j];
        }
        fs[a] = raw ^ xor_acc;
    }
}

// Apply one DisentangleGate to an AffineState (using its public gate API).
void apply_disentangle_gate(AffineState& s, const DisentangleGate& g) {
    switch (g.op) {
        case DG_H:   s.apply_h(g.a);          break;
        case DG_S:   s.apply_s(g.a);          break;
        case DG_SDG: s.apply_sdg(g.a);        break;
        case DG_X:   s.apply_x(g.a);          break;
        case DG_Y:   s.apply_y(g.a);          break;
        case DG_Z:   s.apply_z(g.a);          break;
        case DG_CX:  s.apply_cx(g.a, g.b);   break;
        case DG_CZ:  s.apply_cz(g.a, g.b);   break;
        default: throw std::logic_error("apply_disentangle_gate: bad op");
    }
}

}  // namespace qeccore
