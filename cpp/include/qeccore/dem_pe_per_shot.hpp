#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

// Per-shot law of a PAULI_EXPECTATION column (DEM export, 3.1.13).
//
// The sampler reports a column PER SHOT: <P'> on the state conditioned on EVERY measurement
// outcome (P' = the frame-folded declared Pauli, see dem_export.cpp). DEM export used to judge a
// fault's action on a column from the READ-AVERAGED <D^dag P' D> on the bare state, which disagrees
// with the per-shot value whenever a read that is neither a detector nor a declared frame record is
// correlated with the column. This header computes the exact per-shot law instead.
//
// Setting (deferred space). The bare state |phi> (frame U, reference signs eps, `free` magic
// generators); the terminal reads R_1..R_M (single-qubit, distinct wires, commuting); a world in
// which read j is measured as A_j (A_j = D^dag R_j D for a fault D; A_j = ±R_j for every read the
// fault acts on as a Pauli, i.e. every read outside J); the column X (X = D^dag P' D) commuting
// with every A_j. With Pi_m = prod_j (1 + (-1)^{m_j} A_j)/2 the per-shot value is
//     v(m) = <X Pi_m> / <Pi_m> = sum_S (-1)^{m.S} <X A_S> / sum_S (-1)^{m.S} <A_S>.
// In frame coordinates the state is |chi>_free (x) |eps>_rest, so <A_S> != 0 only on the subspace
// W = {S : the non-free X-part of A_S vanishes}; W = W0 (+) W1 where A_S (S in W0) acts as a
// constant +-1 on the state (a deterministic parity) and W1 (dim d <= 2k) carries the free-sector
// part. Exactly:
//   * if xn(X) (the non-free X-part of X) is not in the span of the xn(A_j): v(m) = 0 for every m;
//   * else pick S0 with xn(X A_S0) = 0; then
//       v(m) = (-1)^{m.S0} h(y),  y_i = m.T_i  (T_1..T_d a basis of W1 mod W0),
//       h(y) = sum_c (-1)^{y.c} <X0 A_T(c)> / sum_c (-1)^{y.c} <A_T(c)>,  X0 = X A_S0,
//     and when xn(X) != 0 the parity m.S0 is an independent fair coin given y (every character
//     containing S0 has a vanishing expectation): a read-dependent SIGN coin.
// Nothing is sampled or approximated. h is evaluated on the SUPPORT of y only, by the engine's own
// exact branch walk on the magic-sector parities (FramedSuperposition::born_p1 / measure_pauli,
// the engine's coin threshold 1e-12): measure the d generator Paulis A_T_i in turn, branching on every outcome of
// positive probability, and read <X0> on each leaf. Cost = (#support leaves) x d measurements —
// polynomial whenever the support is (no 2^d Fourier table); a walk that would exceed the leaf
// cap (kLeafBudget, fixed) or the deterministic work bound (kWorkBudget ~0.5 s by default; the DEM
// export's pe_max_work argument) -- both DETERMINISTIC counts, never wall-clock -- stops after
// bounded work and the caller refuses by name (include_expectations=false: reports the column).
//
// Efficiency: the noiseless reads' X-parts are eliminated ONCE (echelon + nullspace W). A fault
// changes only the rows of the reads in J; the faulted span/nullspace is obtained from the
// noiseless elimination by a |J|-dimensional linear system (projection of W onto J), never by a
// re-elimination of all M reads. The structure (coin, S0, T, d) costs no Born sum at all.
namespace qeccore {

class PeShotAnalyzer {
  public:
    static constexpr long kLeafBudget = 4096;   // default support leaves per walk before refusing by name
    // Default deterministic work bound per walk: each born_p1 / measure_pauli / leaf read is
    // charged (chi + 1) * n * (k + 1) units (calibrated ~3-4e7 units/s on the k=11-13 T-ladder,
    // so ~0.5 s). The DEM export exposes it as pe_max_work; the leaf cap stays fixed (it is not the
    // binding limit on the known cases: a column over d magic-correlated reads has <= 2^d leaves).
    static constexpr double kWorkBudget = 1.5e7;

    PeShotAnalyzer(const FramedSuperposition& bare, const std::vector<std::pair<int, int>>& reads,
                   long leaf_budget = kLeafBudget, double work_budget = kWorkBudget);
    long leaf_budget() const { return leaf_budget_; }
    double work_budget() const { return work_budget_; }

    struct Leaf { uint64_t y = 0; double pr = 0.0; double h = 0.0; };
    struct Law {
        bool zero = false;           // v(m) = 0 on every branch
        bool coin = false;           // v carries the independent fair sign (-1)^{m.S0}
        std::vector<int> S0;         // coin parity read set (ascending; empty when !coin)
        int d = 0;                   // number of y bits
        std::vector<std::vector<int>> T;   // y-bit read sets (ascending)
        Pauli Xw, coinP, X0;         // the world column X, A_S0 and X A_S0 (original coordinates)
        std::vector<Pauli> yP;       // A_T_i (original coordinates)
        std::vector<Leaf> leaves;    // the support of y: Pr[y] > 1e-12 and h(y) (filled by walk_h)
        bool walked = false;         // leaves complete (false after an early exit)
        std::string too_large;       // non-empty: exceeds the caps (caller refuses / reports)
    };

    // Structure only (no Born sum). J: ascending read indices whose world image is not +-R_j.
    // read_pauli(j): the world's A_j (exact, phase included) for ANY read j. X: the world's column
    // (Hermitian, commuting with every A_j — the caller guarantees it).
    Law structure(const std::vector<int>& J, const std::function<Pauli(int)>& read_pauli,
                  const Pauli& X);
    // Fill law.leaves by the exact branch walk. stop(leaves) is consulted after each leaf; true
    // stops the walk early (law.walked stays false). Returns false (too_large set; it names
    // pe_max_work when the work bound binds) when the support exceeds the leaf cap or the walk
    // exceeds the work bound.
    bool walk_h(Law& law, const std::function<bool(const std::vector<Leaf>&)>& stop = nullptr);

    // Exact joint law of the +-1 outcomes of commuting Hermitian Paulis B_1..B_t (original
    // coordinates) on the bare state, by the same branch walk; on every support leaf also the
    // conditional expectation of each Paulis in V (commuting with every B_i). Leaf: (pattern bit
    // i = outcome of B_i is -1, probability, values). false + `why` beyond this analyzer's caps
    // (`why` names pe_max_work when the work bound binds) / t > 63.
    struct JointLeaf { uint64_t f = 0; double pr = 0.0; std::vector<double> v; };
    bool joint_law(const std::vector<Pauli>& B, const std::vector<Pauli>& V,
                   std::vector<JointLeaf>& out, std::string& why);
    // The same walk with caller-chosen DETERMINISTIC caps (support leaves / work units, the
    // walk's count-based cost proxy) — the forced-fault exact law (sample_program_exact_law)
    // passes its max_rows / max_work here. Memoized like joint_law.
    // over_kind (optional): 1 = the leaf cap refused, 2 = the work cap refused. chi2_cost: charge
    // each step max((chi+1) n (k+1), (chi+1)^2) units (the forced law's model; the DEM keeps the
    // linear model its kWorkBudget was calibrated on).
    bool joint_law(const std::vector<Pauli>& B, const std::vector<Pauli>& V,
                   std::vector<JointLeaf>& out, std::string& why, long leaf_budget,
                   double work_budget, int* over_kind = nullptr, bool chi2_cost = false);

    // Frame coordinates of a Pauli on the bare state (exact, linear in P up to phase):
    // xn = its X-part on the NON-free frame positions (nonfree_words() words; bit a set iff P
    // anticommutes with Zrow[a], a not free) and phi = its free-sector part (free_words() words;
    // bit 2d = X on free[d], bit 2d+1 = Z on free[d]). A product of commuting Paulis with xn = 0
    // has a nonzero expectation only through phi; xn = 0 AND phi = 0 => a deterministic +-1.
    void frame_parts(const Pauli& P, std::vector<uint64_t>& xn, std::vector<uint64_t>& phi) const;
    int nonfree_words() const { return NW_; }
    int free_words() const { return FW_; }
    const FramedSuperposition& bare() const { return L_; }

    int num_reads() const { return M_; }

  private:
    const FramedSuperposition& L_;
    std::vector<std::pair<int, int>> reads_;
    long leaf_budget_ = kLeafBudget;     // this analyzer's caps (walk_h and the 4-arg joint_law)
    double work_budget_ = kWorkBudget;
    int n_ = 0, M_ = 0, k_ = 0, NW_ = 0, MW_ = 0, FW_ = 0;
    std::vector<uint64_t> nfm_;                       // non-free frame positions mask (NW_)
    std::vector<std::vector<uint64_t>> zt_x_, zt_z_;  // per wire w: {a : Zrow[a].x_w}, {a : Zrow[a].z_w}
    // noiseless elimination of the reads' non-free X-parts
    struct Row { std::vector<uint64_t> v, comb, phi; int piv; };
    std::vector<Row> ech_;
    struct Null { std::vector<uint64_t> comb, phi; };
    std::vector<Null> null_;
    // exact-content memo: the leaves plus the work the walk charged, so a hit refuses exactly as
    // a fresh walk would under the caller's caps (answers never depend on query history)
    struct MemoEntry { std::vector<JointLeaf> leaves; double work = 0.0; };
    std::unordered_map<std::string, MemoEntry> joint_memo_;
    static constexpr size_t kMemoCells = (size_t)1 << 22;   // stored values cap (~32 MB)
    size_t memo_cells_ = 0;

    void xv(const Pauli& P, std::vector<uint64_t>& out) const;     // all-frame-position X-part
    void phi_of(const Pauli& P, const std::vector<uint64_t>& xvP, std::vector<uint64_t>& out) const;
    void reduce(std::vector<uint64_t>& v, std::vector<uint64_t>& comb, std::vector<uint64_t>& phi) const;
};

}  // namespace qeccore
