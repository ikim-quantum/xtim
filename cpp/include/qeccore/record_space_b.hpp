#pragma once
// record_space_b — J-a N6 (switched OFF in production; reached only through the test API): the canonical record
// table of a DIAGONAL plan with the read directions' rows computed in RECORD space (design §5.1 J-a, owner decision
// 2; plan §1.1 row 12), replacing the σ dual basis of build_record_table (record_table.hpp).
//
// The σ coin part is build_shot_law's own (twirl_law_detail::build_shot_law_core: the same body without the κ>0
// σ-fold stage and without kernel_base). Then, with v_a the dressings of the active generators, D = V ∩ G_z and
// U = {u_j} the kernel directions (L_j = G.reduce(Z^{u_j}) the κ_σ directions, as (D)):
//   1. restricted preimages: u_j = Σ_{a ∈ c_j} v_a over the ACTIVE generators (one A-row elimination);
//   2. kb_j = s_j ⊕ c_j·det, s_j the sign of Π_{a∈c_j} h_a against L_j (kernel_base's operator identity; it is
//      independent of the preimage representative and prefix-free, twirl_kernel.hpp);
//   3. the channels in record order (Theorem 1 in record space): a channel's dressing vv = Σ_{a ∈ mask ∩ active} v_a
//      is reduced against D ∪ U ∪ (the earlier pivots' dressings). A nonzero residual is a PIVOT (A = e_slot,
//      B = 0, β = 0; within V, "dressing independent mod Kz" is "plane independent": Π is symmetric). Otherwise the
//      reduction gives the pivot combination t and the U-coefficients b: A = t, B = b over the κ_σ directions and
//      β = par(det & c') ⊕ ref_c ⊕ Σ_{p∈t} ref_p ⊕ par(kb & b), c' = mask_c ⊕ Σ_{p∈t} mask_p (c' ≡ Σ b_j c_j mod
//      C_det, so c'·σ = c'·det ⊕ b·kb ⊕ b·z). A channel whose mask meets no active generator is constant: β = par(det &
//      mask) ⊕ ref, no elimination;
//   4. the extras by add_extra (record_table_internal.hpp, the ONE routine (D) and (G) share) against that table;
//      then canonicalize_record_order over every column in global record order.
// The canonical table is unique, so the output is FIELD-identical to build_record_table's on every plan (gate
// G-B1, plan §3.9) in the canonical fields: pivot_rec, born_rec, the coin / born rows, β, and Λ_j as (a, b, EFFECTIVE
// phase) — plan §5 Q12: Λ_j's split into (phase, par) may differ where it is a product with a different input path,
// and `folds` / the op counters are build diagnostics (B runs one canonicalization, (D) two), not compared.
// The B STAGE (after build_shot_law_core) never visits an inactive generator (gate G-B2: its operation counts are
// independent of inactive padding); the shared core (tier1, the touched cone, the lattice) is build_shot_law's own.
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

// Operation counts of the B stage of ONE build (the mechanism gate G-B2).
struct RecBOps {
    long touched = 0;       // channels whose mask meets an active generator
    long gen_visits = 0;    // generator dressings XORed into a channel (active generators only)
    long elim_rows = 0;     // row XORs of the channel elimination (against D ∪ U ∪ pivots)
    long pre_rows = 0;      // row XORs of the restricted preimages
    long pivots = 0;        // channel pivots
    long kb_read = 0;       // directions with kb_j = 1 that some channel reads (test coverage of the β sign path)
    long mask_words = 0;    // lean L4: channel-mask words scanned densely (0: the record-column index serves the
                            // active generators' channels; the MUT-B-PAD mutation walks every mask)
    long law_reduces = 0;   // lean L4: G.reduce calls on the production path of the law build (build_record_table_recB
                            // only: the κ kernel representatives; the C_det signs are a closed form)
};

// lean L4: the circuit-constant transpose of the record columns, built ONCE (the sampler's ctor; the tests per call)
// and read by every record-space B build: per certified generator the channels whose σ-mask holds it (CSR, ascending
// channel index), the channel / extra columns in global record order, each column's record index and reference bit.
// A build's per-channel data (β's deterministic part par(det & mask_c), the active generators a channel's mask meets)
// is then gathered from the det / active generators' set bits instead of scanning every mask over every generator word.
struct RecColIndex {
    int m = 0, n_gens = 0;
    std::vector<int> order;             // column c → global record index
    std::vector<uint8_t> ref;           // column c → ref & 1 (channels), 0 for extras
    std::vector<int> ch, ex;            // channel / extra columns, each in record order
    std::vector<int> goff;              // generator i → [goff[i], goff[i+1]) into gcol (n_gens + 1 entries)
    std::vector<int> gcol;              // the channels whose mask holds generator i, ascending
    RecColIndex() = default;
    RecColIndex(const std::vector<RecColumn>& cols, int n_gens);
};

// The canonical record table of the diagonal plan nf (its prefix is ignored: the table is P-free, Lemma 2).
// Returns false (out unbuilt) iff the plan's law is a fallback. Throws std::logic_error on an engine-bug invariant.
// (The cold entry: the window's class_rep_check_, the record readers, the test API; it builds the law with its teeth
// and a RecColIndex of `cols` for the call.)
bool build_record_table_recB(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                             const DiagNormalForm& nf, const std::vector<RecColumn>& cols, RecTable& out,
                             RecBOps* ops = nullptr);
// J-b1 (fixplan-coldstart (A)): the same table from a core law already built (build_shot_law_core at the identity
// prefix of the same plan): the B stage only. build_record_table_recB = core build + this. `idx` is RecColIndex(cols,
// G.n_gens); `teeth` (lean L4: the caller's self-check window) re-derives the index-gathered per-channel data densely
// and compares it (throws by name) — outputs are identical either way.
namespace twirl_law_detail { struct ShotLawCore; }
bool build_record_table_recB_from_core(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                       const DiagNormalForm& nf, const ShotLaw& law,
                                       const twirl_law_detail::ShotLawCore& core,
                                       const std::vector<RecColumn>& cols, const RecColIndex& idx, RecTable& out,
                                       bool teeth, RecBOps* ops = nullptr);

}  // namespace qeccore
