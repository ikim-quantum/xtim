#pragma once
// term_residual — J-a (N2, switched OFF in production: built only by FAM samplers). The TERM-INDEXED
// shot residual of design.final.md §2.5:
//
//   * TermView. The deferred circuit's T-type π/8 factors exp(-i(π/8) A_j) — T (A = +Z), CS (Za, Zb,
//     −ZaZb), CCZ (the seven Z-strings), CH (V† ... V around a CZ, one −Z and one +Z factor) — are the
//     TERMS, in stream order; B_j = A_j pushed to the end of the circuit through the later Clifford
//     gates (exact sign). The residual covers COMMUTING term families only (design §2.5): construction
//     checks the B_j pairwise, at ONE site, and throws the typed class-boundary error TermClassRefusal
//     when two anticommute. Such circuits are NOT refused upstream (the bare-state builder and today's
//     twirl sampler accept them, e.g. two CH Hadamard tests sharing a wire); the FAM sampler catches
//     exactly that type and refuses by name. term_rows(j) = the records whose terminal read anticommutes
//     with B_j.
//   * Atoms. A Pauli fault P at time t, pushed to the end, is R_f = P_f · Π_{j∈S_f} rot(B_j),
//     rot(B) = exp(i(π/4) B): S_f = the terms whose factor P anticommutes with when it crosses it, P_f =
//     P conjugated by the Clifford gates only. Per flat alternative (FiredPauli list) the table holds
//     (P_f, S_f); every phase is global (mod-phase Paulis throughout).
//   * Composition (compose_terms_): R ← R_f · R, i.e. later faults LEFTMOST:
//       P ← P_f · P · Π_{j∈S_f, {B_j,P}=0} B_j · Π_{j∈S_f∩S} B_j,   S ← S ⊕ S_f.
//   * KerPhi. Φ: S ↦ symplectic action of rot(S) is linear over F₂ (v ↦ v + Σ_{j∈S}⟨v,b_j⟩ b_j).
//     Its kernel basis k_i (one per non-pivot term, terms eliminated in index order) and N_{k_i} =
//     rot(k_i) (a Pauli). reduce_key_ folds SEQUENTIALLY (adv-A kfold.py): S_i = S_{i−1} △ k_i with
//     rot(S_{i−1}) = rot(S_i) · N_{k_i} · Π_{j∈S_{i−1}∩k_i} B_j (mod phase), the right factor moved
//     left into P. The reduced S is supported on the pivot terms: the KEY (rank Φ bits) is canonical,
//     equal Clifford mod Pauli ⇔ equal key.
//   * kind_of_. EXACT diagonality of Φ(S): rot(S) fixes every Z_q up to sign (Z_q ↦ ±Z_q). Cold,
//     cached per distinct reduced S. Kind 0 (diagonal): rot(S)·P is handed, as a DiagPauliClifford, to
//     THE diagonal route's normal form — residual_normal_form then canonicalize_mod_stabilizers_inplace
//     (twirl_normal_form.cpp, one function, no copy) — giving (P′, a, cz), the stripped kind-0 key.
//     Otherwise kind 2.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/clifford_op.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/sampler.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// The term-class boundary: two term axes anticommute — the circuit is outside the term-indexed
// residual's class (commuting term families, design §2.5). NOT an engine bug: circuits the bare-state
// builder and today's twirl sampler accept reach it; the FAM sampler (fam_setup_) catches exactly this
// type and refuses by name.
struct TermClassRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
    // J-b1 P13: the structured refusal (the bindings render it in user qubits): the two anticommuting pushed axes,
    // and, when a fault blocked a merge, that merge's two deferred gate indices and the deferred noise site.
    std::vector<Pauli> axes;
    int gate_i = -1, gate_j = -1, site = -1;
};

// Per-shot scratch for the term route (sized once by TermResidual::init_shot; reused, no allocation).
struct TermShot {
    Pauli P;                       // mod-phase Pauli part (phase always 0)
    std::vector<uint64_t> S;       // TW words: the term set
    std::vector<uint64_t> C;       // TW words: scratch (correction set / original S)
    Pauli N;                       // scratch: the kernel factor N′
};

// One shot through the term route (FAM test API): the composed (P0, S0), the reduced (sh.P, sh.S), the
// kind and the plan id; kind 0 also carries the canonical normal form (P′ = nf.prefix).
struct TermRouteShot {
    TermShot sh;
    Pauli P0;
    std::vector<uint64_t> S0;
    int kind = -1;
    long plan_id = -1;
    DiagNormalForm nf;             // kind 0: canonical (after canonicalize_mod_stabilizers_inplace)
    DiagNormalForm nf_raw;         // kind 0: residual_normal_form's output, before the canonicalization
};

class TermResidual {
public:
    // The term view of a deferred circuit (twirl-sampler normalized stream) with its terminal reads
    // (basis 0 X / 1 Y / 2 Z, deferred wire). Both constructors throw TermClassRefusal when two term
    // axes anticommute (check_commuting_, the one check).
    TermResidual(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads);
    // J-b1 P5 (plan r3.1 + r3.1-fix, the owner's non-Clifford rule): with the sampler's alternatives (p > 0 only),
    // the STATIC fixed-axis merge runs first whenever two pushed axes anticommute — two terms with the same
    // pushed axis A (up to sign) merge iff no unmerged term between them anticommutes with A and no alternative
    // located between them flips the later one; equal signs fold the S-type Clifford exp(-iπ/4 A_i) into the op
    // list at the earlier term's slot (convention (A): every axis and alternative is recomputed by the end-push,
    // the earlier ones change), opposite signs cancel; passes in time order to a fixpoint. The commutation check
    // then runs on the merged set (TermClassRefusal names the merge a fault blocks). The alternatives are then
    // attached (alts_ready()). `live` (size = sides.size(), or empty = all live): alternative a is a fault site of
    // the rule iff live[a] (its probability after p_factor is > 0; r3.1-fix R3-4 — a p = 0 component is not a
    // fault site; the sampler keeps zero-probability X/Y/Z_ERROR alternatives under p_factor = 0).
    // `z_screen` (plan r3.1 item 2): when every pushed axis is Z-type the set commutes and no merge can run; the
    // construction then stops after the axis pass (z_only() true; no rows, kernel or alternatives: the verdict only,
    // for an unarmed sampler, R9a).
    TermResidual(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                 const std::vector<std::vector<FiredPauli>>& sides, const std::vector<char>& live = {},
                 bool z_screen = false);
    bool z_only() const { return z_only_; }
    // J-b1 P5: the merge pass's counters (merges, folds, passes; 0 when no axis pair anticommuted).
    struct MergeStats { int merges = 0, folds = 0, cancels = 0, passes = 0, terms_before = 0; };
    const MergeStats& merge_stats() const { return mstats_; }
    // A hand-built term set (signed Hermitian axes on n wires): tests. No ops, no alternatives.
    TermResidual(int n, const std::vector<Pauli>& axes);

    int n() const { return n_; }
    int num_terms() const { return T_; }
    int term_words() const { return TW_; }
    const Pauli& axis(int j) const { return B_[(size_t)j]; }
    const std::vector<int>& term_rows(int j) const { return term_rows_[(size_t)j]; }

    // ── per-alternative (P_f, S_f) ──
    void set_alts(const std::vector<std::vector<FiredPauli>>& sides);
    bool alts_ready() const { return alts_ready_; }
    int num_alts() const { return A_; }
    const uint64_t* alt_px(int a) const { return &apx_[(size_t)a * NW_]; }
    const uint64_t* alt_pz(int a) const { return &apz_[(size_t)a * NW_]; }
    const uint64_t* alt_s(int a) const { return &as_[(size_t)a * TW_]; }
    // J-a (N4b): input frame components (port wire, X / Z / Y) at t = 0 as atoms (P_F, S_F), the alternatives'
    // propagation; output (comps × NW) X / Z words and (comps × TW) term words.
    void port_atoms(const std::vector<std::pair<int, PauliBasis>>& comps, std::vector<uint64_t>& px,
                    std::vector<uint64_t>& pz, std::vector<uint64_t>& s) const;

    // ── composition ──
    void init_shot(TermShot& sh) const;                         // size the buffers (cold)
    void clear_shot(TermShot& sh) const;                        // R = I
    // R ← (P_f, S_f) · R (one atom, on the left).
    void compose_atom_(const uint64_t* px, const uint64_t* pz, const uint64_t* s, TermShot& sh) const;
    // R = R_{alts[count-1]} ⋯ R_{alts[0]} (alternatives in time order; later leftmost).
    void compose_terms_(const int32_t* alts, int count, TermShot& sh) const;

    // ── KerPhi ──
    int rank() const { return (int)pivot_terms_.size(); }
    int kernel_dim() const { return (int)kern_.size(); }
    const std::vector<int>& pivot_terms() const { return pivot_terms_; }
    struct KernelVec {
        int j;                          // its non-pivot term
        std::vector<uint64_t> k;        // TW words
        Pauli N;                        // rot(k) up to phase
    };
    const std::vector<KernelVec>& kernel() const { return kern_; }
    // One sequential fold step with kernel vector i: S ← S △ k_i, the factor N_{k_i}·Π_{S∩k_i} B_j taken
    // from the CURRENT S and moved into P.
    void fold_kernel_(int i, TermShot& sh) const;
    // An unconditional fold sequence (kernel vectors idx[0], idx[1], ... in order; adv-A kfold.py).
    void fold_sequence_(const std::vector<int>& idx, TermShot& sh) const;
    // S → its canonical representative (pivot-supported), every kernel factor folded into P.
    void reduce_key_(TermShot& sh) const;
    // The key: the reduced S on the pivot terms (rank() bits, (rank()+63)/64 words).
    std::vector<uint64_t> key_of(const std::vector<uint64_t>& Sred) const;

    // ── kind ──
    // 0 = Φ(S) diagonal, 2 otherwise. Sred must be reduced (reduce_key_). Cold, cached per Sred.
    int kind_of_(const std::vector<uint64_t>& Sred);
    // termroute-a cache bound: the kind-cache entry of a reduced set given as its raw word bytes (a FAM route key):
    // its bytes (0 when absent) and its removal (kind_of_ is a pure function of Sred: a later call recomputes it).
    size_t kind_bytes_(const std::string& sbytes) const;
    void kind_forget_(const std::string& sbytes);
    // Kind 0 only: rot(Sred)·P through the diagonal route's normal form (residual_normal_form, then
    // canonicalize_mod_stabilizers_inplace against G); `raw` (optional) receives the normal form before
    // the canonicalization.
    DiagNormalForm kind0_normal_form_(const Pauli& P, const std::vector<uint64_t>& Sred,
                                      const CertifiedGroupPlanes& G, DiagNormalForm* raw = nullptr);
    // The exact image rot(S) V rot(S)† (exact phase).
    Pauli rot_image(const std::vector<uint64_t>& S, const Pauli& V) const;

    // ── plan identity (composite Clifford mod Pauli; JA-4): one intern table for every route; the caller
    // builds the key bytes (kind 0: the stripped (a, cz) key of the canonical normal form; kind 2: the
    // term key).
    long plan_id_(const std::string& key);
    long num_plan_ids() const { return (long)plan_ids_.size(); }

private:
    struct Op {
        uint8_t kind = 0;              // 0 Clifford, 1 term, 2 a folded S-type Clifford (J-b1 P5), 3 removed
        GateKind g = GateKind::X;
        int a = -1, b = -1;
        int term = -1;                 // kind 1: term index; kind 2: folds_ index
    };
    struct Fold { Pauli A; int k = 2; };   // exp(-i k π/8 A), k ∈ {2, 6}, A the time-frame (Z-string) axis
    void build_ops_(const Circuit& deferred);
    void push_axes_();                     // B_j: local axes end-pushed through ops kinds 0 and 2 (exact sign)
    void finish_(const std::vector<std::pair<int, int>>& reads);
    void merge_(const std::vector<std::vector<FiredPauli>>& sides, const std::vector<char>& live);
    std::vector<Fold> folds_;
    std::vector<int> term_si_;             // term → deferred stream index of its gate (messages)
    MergeStats mstats_;
    std::string blocked_;                  // the first fault-blocked merge (refusal text)
    int blocked_gi_ = -1, blocked_gj_ = -1, blocked_site_ = -1;   // J-b1 P13: the same, structured
    struct KindEntry {
        int kind = 2;
        DiagPauliClifford D;           // kind 0: rot(Sred) exactly, mod global phase
    };
    void build_kernel_();
    void check_commuting_() const;   // throws TermClassRefusal
    struct Inj { int pos, alt, q; PauliBasis b; };   // an injected Pauli: before ops_[pos], atom `alt`
    void propagate_(std::vector<Inj> inj, int A, std::vector<uint64_t>& px, std::vector<uint64_t>& pz,
                    std::vector<uint64_t>& s) const;

    int n_ = 0, NW_ = 0, T_ = 0, TW_ = 0;
    std::vector<Op> ops_;
    std::vector<Pauli> local_;                 // A_j at its factor
    std::vector<Pauli> B_;                     // B_j at the end (signed)
    std::vector<int> loc_op_;                  // deferred stream index → op position (Noise: -1 else)
    std::vector<std::vector<int>> term_rows_;
    // alternatives
    bool alts_ready_ = false;
    bool z_only_ = false;                           // J-b1 P5: stopped after the axis pass (every axis Z-type)
    int A_ = 0;
    std::vector<uint64_t> apx_, apz_, as_;
    // kernel
    std::vector<int> pivot_terms_;
    std::vector<KernelVec> kern_;
    // kind cache / plan ids
    std::unordered_map<std::string, KindEntry> kind_cache_;
    std::vector<uint64_t> kscr_;               // termroute-a F4: kind_of_'s per-q (x | z) accumulators (n × 2NW, cold)
    std::vector<uint8_t> kmark_;
    std::vector<int> ktouch_;
    std::unordered_map<std::string, long> plan_ids_;
};

}  // namespace qeccore
