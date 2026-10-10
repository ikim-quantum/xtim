#include "qeccore/port_contract.hpp"
#include "qeccore/bitmat.hpp"
#include <algorithm>
#include <unordered_set>

namespace qeccore {

// Build a GF(2) submatrix of M restricted to a given set of column indices.
// M has shape (m x 2n); the output has shape (m x 2*|subset|).
// Column layout of M: columns 0..n-1 = X-part, columns n..2n-1 = Z-part.
// We pass wire indices in [0, n); they map to columns {wire, n+wire} in M.
static GF2Mat restrict_to_wires(const GF2Mat& M, int n,
                                 const std::vector<int>& wires) {
    int m   = M.rows;
    int ncols = 2 * (int)wires.size();
    GF2Mat R(m, ncols);
    for (int i = 0; i < m; ++i) {
        for (int c = 0; c < (int)wires.size(); ++c) {
            int wire = wires[c];
            // X part: column wire of M -> column c of R
            if (M.get(i, wire))       R.set1(i, c);
            // Z part: column n+wire of M -> column |wires|+c of R
            if (M.get(i, n + wire))   R.set1(i, (int)wires.size() + c);
        }
    }
    return R;
}

PortRanks port_ranks(const std::vector<Pauli>& gens, int n,
                     const std::vector<int>& port_wires) {
    // Validate port_wires: every wire must be in [0, n) and unique.
    // Catching this here stops an unguarded OOB read in restrict_to_wires
    // (M.get(i, wire) / M.get(i, n+wire)) which in practice returns garbage
    // ranks and produces a silent-wrong verdict (product=false for a wire typo).
    // This guard also kills the reserve() size_t underflow (|wires|>n impossible
    // once wires are unique and in-range) and duplicate entries in `stabilized`.
    {
        std::unordered_set<int> seen;
        for (int w : port_wires) {
            if (w < 0 || w >= n)
                throw std::invalid_argument(
                    "port_ranks: port wire " + std::to_string(w) +
                    " out of range [0, " + std::to_string(n) + ")");
            if (!seen.insert(w).second)
                throw std::invalid_argument(
                    "port_ranks: duplicate port wire " + std::to_string(w));
        }
    }

    int m = (int)gens.size();

    // Build the m x 2n GF(2) check matrix. Ignore phases entirely.
    GF2Mat M(m, 2 * n);
    for (int i = 0; i < m; ++i) {
        const Pauli& g = gens[i];
        for (int q = 0; q < n; ++q) {
            if (g.xbit(q)) M.set1(i, q);
            if (g.zbit(q)) M.set1(i, n + q);
        }
    }

    // Build complement (B) wire list.
    std::unordered_set<int> port_set(port_wires.begin(), port_wires.end());
    std::vector<int> b_wires;
    b_wires.reserve(n - (int)port_wires.size());
    for (int q = 0; q < n; ++q) {
        if (!port_set.count(q)) b_wires.push_back(q);
    }

    // Three rank computations.
    int rank_S = rank(M);

    // rank(S ∩ P_B) = rank_S - rank(A-block)
    GF2Mat A_block = restrict_to_wires(M, n, port_wires);
    int rank_A_block = rank(A_block);
    int rank_B = rank_S - rank_A_block;

    // rank(S ∩ P_A) = rank_S - rank(B-block)
    GF2Mat B_block = restrict_to_wires(M, n, b_wires);
    int rank_B_block = rank(B_block);
    int rank_A = rank_S - rank_B_block;

    int n_B = (int)b_wires.size();

    PortRanks result;
    result.rank_S = rank_S;
    result.rank_A = rank_A;
    result.rank_B = rank_B;
    result.n_B    = n_B;
    result.ok     = (rank_B == n_B);
    result.clifford_factorizes = (rank_A + rank_B == rank_S);

    // Straddling witnesses: rows of row_reduce(M) that have support in BOTH
    // A-columns and B-columns. Only populated when !clifford_factorizes.
    if (!result.clifford_factorizes) {
        // Determine which columns are A-type and which are B-type.
        std::unordered_set<int> a_cols;
        for (int wire : port_wires) {
            a_cols.insert(wire);        // X-part column
            a_cols.insert(n + wire);    // Z-part column
        }

        GF2Mat R = row_reduce(M);
        for (int i = 0; i < R.rows; ++i) {
            bool has_a = false, has_b = false;
            for (int c = 0; c < 2 * n; ++c) {
                if (!R.get(i, c)) continue;
                if (a_cols.count(c)) has_a = true;
                else                  has_b = true;
                if (has_a && has_b) break;
            }
            if (has_a && has_b) result.straddling.push_back(i);
        }
    }

    return result;
}

// ── Task 2: port_contract() ────────────────────────────────────────────────────────────────

PortVerdict port_contract(const FramedSuperposition& st, const std::vector<int>& port_wires) {
    PortVerdict v;
    v.port_wires = port_wires;
    v.k_port     = (int)st.free.size();

    // VACUOUS CASE: empty port — the contract does not apply.
    if (port_wires.empty()) {
        v.product = true;
        v.witness = "";
        return v;
    }

    // Build common stabilizer generators: { st.U.Zrow[a] : a not in st.free }.
    std::unordered_set<int> free_set(st.free.begin(), st.free.end());
    int n = st.n();
    std::vector<Pauli> gens;
    gens.reserve(n - v.k_port);
    for (int a = 0; a < n; ++a) {
        if (!free_set.count(a)) {
            // Reconstruct the Pauli with the correct sign from eps.
            // eps[a]=1 means g_a|psi> = -|psi>, i.e. g_a has sign -1 => add phase 2 (i^2=-1).
            Pauli g = st.U.Zrow[a];
            if (st.eps[a]) g.phase = (g.phase + 2) & 3;
            gens.push_back(g);
        }
    }

    PortRanks pr = port_ranks(gens, n, port_wires);
    v.product = pr.ok;

    // Build the stabilized list: for each port wire, read exact single-qubit expectations.
    if (v.product) {
        const double kTol = 1e-9;
        for (int q : port_wires) {
            // X
            Pauli Px(n); Px.setx(q);
            double ex = st.pauli_expectation(Px).real();
            if (std::abs(std::abs(ex) - 1.0) < kTol) {
                v.stabilized.push_back({q, 0, ex < 0 ? uint8_t(1) : uint8_t(0)});
                continue;
            }
            // Y = i·XZ
            Pauli Py(n); Py.setx(q); Py.setz(q); Py.phase = 1;
            double ey = st.pauli_expectation(Py).real();
            if (std::abs(std::abs(ey) - 1.0) < kTol) {
                v.stabilized.push_back({q, 1, ey < 0 ? uint8_t(1) : uint8_t(0)});
                continue;
            }
            // Z
            Pauli Pz(n); Pz.setz(q);
            double ez = st.pauli_expectation(Pz).real();
            if (std::abs(std::abs(ez) - 1.0) < kTol) {
                v.stabilized.push_back({q, 2, ez < 0 ? uint8_t(1) : uint8_t(0)});
                continue;
            }
            // No definite axis (magic qubit): emit nothing.
        }
    }

    // Build witness string for refusal cases.
    if (!v.product) {
        if (!pr.clifford_factorizes) {
            // Entanglement straddles the cut.
            std::string wlist = "{";
            for (size_t i = 0; i < port_wires.size(); ++i) {
                if (i) wlist += ",";
                wlist += std::to_string(port_wires[i]);
            }
            wlist += "}";
            int n_strad = (int)pr.straddling.size();
            v.witness = "port wires " + wlist + " are entangled with the rest (" +
                        std::to_string(n_strad) + " stabilizer generator" +
                        (n_strad == 1 ? "" : "s") + " straddle the cut)";
        } else {
            // B is not a stabilizer state.
            int k_B = pr.n_B - pr.rank_B;
            v.witness = "the rest is not a stabilizer state (k_B = " +
                        std::to_string(k_B) + "); measure those wires in a product basis "
                        "(twirled) before the boundary";
        }
    }

    return v;
}

}  // namespace qeccore
