#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/born_law_coset.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"

// Per-shot law of a PAULI_EXPECTATION column (DEM export, 3.1.12).
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
// Nothing is sampled or approximated. twirl-routes T2: the y bits are Born directions on the bare FREE register (A_T_i
// has no non-free X part, so it is a free-register operator: FreeDir), and h is evaluated on the support of y by the
// ONE routine's Born law (born_law_enumerate: BornTrie's sequential projection, 2^d <= 2^k = chi-bounded leaves, or
// the coset-of-X-span law of born_law_coset.hpp when its cost model wins — explaw) and the leaf value ⟨X0⟩ on the
// projected register — no coin is ever branched on (coins are the GF(2) part: S0) and there is no work or leaf cap.
//
// Efficiency: the noiseless reads' X-parts are eliminated ONCE (echelon + nullspace W). A fault
// changes only the rows of the reads in J; the faulted span/nullspace is obtained from the
// noiseless elimination by a |J|-dimensional linear system (projection of W onto J), never by a
// re-elimination of all M reads. The structure (coin, S0, T, d) costs no Born sum at all.
namespace qeccore {

class PeShotAnalyzer {
  public:
    PeShotAnalyzer(const FramedSuperposition& bare, const std::vector<std::pair<int, int>>& reads);

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
        std::shared_ptr<const CosetLawTable> table;   // the coset law's table when that route ran (value_at reads it)
        std::string too_large;       // non-empty: > 63 y bits (structure; caller refuses / reports)
    };

    // Structure only (no Born sum). J: ascending read indices whose world image is not +-R_j.
    // read_pauli(j): the world's A_j (exact, phase included) for ANY read j. X: the world's column
    // (Hermitian, commuting with every A_j — the caller guarantees it).
    Law structure(const std::vector<int>& J, const std::function<Pauli(int)>& read_pauli,
                  const Pauli& X);
    // Fill law.leaves: every y of positive probability (the Born walk on the free register, + outcome first; a step
    // whose conditional probability is within 1e-12 of 0 / 1 is taken as deterministic, the engine's coin threshold)
    // with h(y). stop(leaves) is consulted after each leaf; true stops early (law.walked stays false). Always true
    // (no cap: 2^d <= chi leaves).
    bool walk_h(Law& law, const std::function<bool(const std::vector<Leaf>&)>& stop = nullptr);
    // h on one y (bit i = outcome of yP_i is -1): the same projections as walk_h's leaf, from scratch.
    double value_at(const Law& law, uint64_t y);
    long born_evals() const { return born_evals_; }   // Born-law node + leaf-value evaluations (mechanism counter)
    long coset_walks() const { return coset_walks_; }   // explaw: column walks routed to the coset law
    double coset_work() const { return coset_work_; }  //   and their work (CosetLawTable::work)

    // EXACT ENGINE ONLY (the forced-fault exact law, sample_program_exact_law; test-API oracle surface): the exact
    // joint law of the +-1 outcomes of commuting Hermitian Paulis B_1..B_t (original coordinates) on the bare state
    // by a branch walk on the full state, with caller-chosen DETERMINISTIC caps (support leaves / work units, the
    // walk's count-based cost proxy); on every support leaf also the conditional expectation of each Pauli in V.
    // Leaf: (pattern bit i = outcome of B_i is -1, probability, values). The DEM export never calls it.
    struct JointLeaf { uint64_t f = 0; double pr = 0.0; std::vector<double> v; };
    // over_kind (optional): 1 = the leaf cap refused, 2 = the work cap refused. chi2_cost: charge
    // each step max((chi+1) n (k+1), (chi+1)^2) units (the forced law's model; the DEM keeps the
    // linear model).
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
    long born_evals_ = 0;
    long coset_walks_ = 0;
    double coset_work_ = 0.0;
    int n_ = 0, M_ = 0, k_ = 0, NW_ = 0, MW_ = 0, FW_ = 0;
    std::vector<uint64_t> nfm_;                       // non-free frame positions mask (NW_)
    std::vector<std::vector<uint64_t>> zt_x_, zt_z_;  // per wire w: {a : Zrow[a].x_w}, {a : Zrow[a].z_w}
    // noiseless elimination of the reads' non-free X-parts
    struct Row { std::vector<uint64_t> v, comb, phi; int piv; };
    std::vector<Row> ech_;
    struct Null { std::vector<uint64_t> comb, phi; };
    std::vector<Null> null_;
    // structure()'s bit rows over the changed reads J (flat, JW words per row): scratch kept across calls
    struct JScratch { std::vector<uint64_t> piv_proj, kern_used, jr_pi, jr_used, er_pi, er_u, nullu, p, used, pi, u, s_used; };
    JScratch js_;
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
