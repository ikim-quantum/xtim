#pragma once
#include <memory>
#include <optional>
#include <vector>
#include "qeccore/clifford_op.hpp"        // DiagPauliClifford
#include "qeccore/clifford_tableau.hpp"   // CliffordTableau (for PropResult::general)
#include "qeccore/circuit_ir.hpp"         // Circuit, Instr, PauliBasis
#include "qeccore/ppr_residual.hpp"       // PprResidual (V2: PropResult::ppr axis list)

namespace qeccore {

// Result of pushing one seed error to the end of the circuit.
struct PropResult;
void sparsify_cz(PropResult& a);
void sparsify_av(PropResult& a);
DiagPauliClifford dense_c_prop(const PropResult& a);

struct PropResult {
    // The end-of-circuit error (valid iff !rejected). czrows F2: read through c_prop() / c_prop_mut(), which THROW
    // std::logic_error on an atom in the sparse CZ form (sparsify_cz; the twirl sampler's table), so a reader that
    // forgets the form fails loudly instead of reading a valid-looking all-zero CZ layer. A sparse atom is read
    // through dense_c_prop(atom) (the dense operator, rebuilt bit for bit), prop_a() / prop_v() (its S/Z and X
    // parts; they throw in the sparse a/v form below) and cz_pairs() (its CZ layer).
    const DiagPauliClifford& c_prop() const { if (cz_sparse_ || av_sparse_) sparse_read_(); return c_prop_; }
    DiagPauliClifford& c_prop_mut() { if (cz_sparse_ || av_sparse_) sparse_read_(); return c_prop_; }
    const std::vector<uint8_t>& prop_a() const { if (av_sparse_) sparse_av_read_(); return c_prop_.a; }
    const std::vector<uint8_t>& prop_v() const { if (av_sparse_) sparse_av_read_(); return c_prop_.v; }
    // perf-engine §3.2: a SUPERSET of the qubits c_prop acts on (a / v / B rows outside it are zero), as a bitset
    // over the wires (ceil(n/64) words; bit q = wire q): the table's walk marks the seed qubit and the targets of
    // every gate it applied (the walk's own support, propagation_table.cpp); propagate_atom (no walk) marks every
    // wire. Consumers extracting c_prop's sparse content walk its set bits (ascending) instead of all n wires.
    // lean L1: freed in the sparse a/v form (the lists below are the exact content); reading it there throws.
    const std::vector<uint64_t>& support() const { if (av_sparse_) sparse_av_read_(); return support_; }
    std::vector<uint64_t>& support_mut() { if (av_sparse_) sparse_av_read_(); return support_; }
    // lean L1 (2026-11-04): the SPARSE a/v form (sparsify_av; the twirl sampler's table, whose atoms are read after
    // construction only by the selfcheck window's compose_fired, the digest and the test APIs — all through
    // dense_c_prop). When av_sparse(): sparse_a() = (q, a_q) for every a_q != 0 (a_q in 1..3; ascending q),
    // sparse_v() = every q with v_q = 1 (ascending); the dense a / v vectors and `support` are freed. dense_c_prop
    // rebuilds a / v bit for bit. On a DENSE atom the two list readers throw (never a valid-looking empty list).
    bool av_sparse() const { return av_sparse_; }
    const std::vector<std::pair<int, uint8_t>>& sparse_a() const { if (!av_sparse_) dense_av_read_(); return a_sparse_; }
    const std::vector<int>& sparse_v() const { if (!av_sparse_) dense_av_read_(); return v_sparse_; }
    bool rejected = false;
    int  reject_gate_index = -1;  // stream index of the gate that left the class
    // Non-null iff a PPR retry accepted this atom when the standard path rejected AND the table
    // was built with build_general (the default): the full error tableau
    // E = pauli · (Π rots Clifford) as a CliffordTableau (= ppr->to_error_tableau()). The twirl
    // sampler's table does not build it (it reads `ppr` only).
    std::shared_ptr<const CliffordTableau> general;
    // V2 (2026-07-15, additive): the accepted PPR retry's residual itself — the AXIS LIST the
    // twirl sampler's PPR plan family consumes (the tableau above erases the axes). Non-null
    // exactly when a PPR retry accepted this atom.
    std::shared_ptr<const PprResidual> ppr;
    // False iff this atom is absent: its seed Pauli factor has probability ZERO in the location's
    // channel (e.g. the X atom of a Z_ERROR), so no shot can fire it, and the table either never
    // built it (AtomSet::ChannelSupport) or built it and it left the class (AtomSet::AnyPauli;
    // then `rejected` is also set). An absent atom never rejects the circuit and must never be
    // read — LocationEntry::atom() refuses it by name.
    bool present = true;
    // czrows (2026-10-30): the SPARSE CZ form (sparsify_cz). When cz_sparse(), the CZ layer B is held as its pair list
    // instead of n dense rows: cz_pairs() = B's (j, l) pairs with j < l, ascending (j, then l), cz_words() = B.words()
    // as built (0 = the lazy all-zero form); the dense rows are freed (cz_rows_freed()).
    bool cz_sparse() const { return cz_sparse_; }
    int cz_words() const { return cz_words_; }
    // threads L1 (czrows review): a DENSE atom has no pair list — reading it would return a valid-looking empty CZ
    // layer, so it throws std::logic_error like c_prop() does on a sparse atom.
    const std::vector<std::pair<int, int>>& cz_pairs() const { if (!cz_sparse_) dense_pairs_read_(); return cz_pairs_; }
    bool cz_rows_freed() const { return c_prop_.B.words() == 0 && c_prop_.B.data_.capacity() == 0; }

   private:
    DiagPauliClifford c_prop_;
    std::vector<uint64_t> support_;
    bool cz_sparse_ = false;
    int  cz_words_ = 0;
    std::vector<std::pair<int, int>> cz_pairs_;
    bool av_sparse_ = false;
    std::vector<std::pair<int, uint8_t>> a_sparse_;
    std::vector<int> v_sparse_;
    [[noreturn]] static void sparse_read_();
    [[noreturn]] static void dense_pairs_read_();
    [[noreturn]] static void sparse_av_read_();
    [[noreturn]] static void dense_av_read_();
    friend void sparsify_cz(PropResult& a);
    friend void sparsify_av(PropResult& a);
    friend DiagPauliClifford dense_c_prop(const PropResult& a);
    friend struct WalkAccess;                                   // lean L6: the table's walk writes the atom directly
};

// Propagate `seed` through every Gate instruction at stream index > from_index, in order,
// via conjugate_by_gate. Non-Gate instructions are transparent. Stops and reports reject the
// moment a gate takes the object out of the diagonal+Pauli class.
PropResult propagate_atom(const Circuit& circ, int from_index, const DiagPauliClifford& seed);

// czrows: put an atom's CZ layer in the sparse form (PropResult::cz_sparse): scan every dense row of c_prop's B (all n,
// independent of `support`), keep the j < l pairs, free the rows. Lossless: B is symmetric with a zero diagonal and
// zero padding (it is written through SymPackedMat::flip_sym only) — and VERIFIED before the rows are freed (F1): the
// rows rebuilt from the pairs must equal the dense rows word for word, else std::logic_error (engine bug). Idempotent.
// dense_c_prop: the atom's c_prop with its dense CZ layer (a copy; for a sparse atom B is rebuilt bit for bit: same
// dim, same words(), same rows).
// lean L1 — sparsify_av: put an atom's S/Z and X layers in the sparse a/v form (PropResult::av_sparse): the (q, a_q)
// list of the nonzero a bytes and the q list of the v bits, the dense vectors and the walk support freed. Lossless
// under the content contract (a_q in Z4, v_q in GF(2) — all conjugate_by_gate writes) and VERIFIED before the
// vectors are freed (F1): the dense vectors rebuilt from the lists must equal the originals byte for byte, else
// std::logic_error (engine bug). Idempotent. dense_c_prop rebuilds a / v (then B) for a sparse atom.

// A propagated basis error stored in the table. Identical in content to PropResult (a stored
// propagation result), so it aliases it; kept as a distinct name for call-site readability.
using PropAtom = PropResult;

// All atoms for one Noise location.
// ATOM SET = the channel's support: the X atom of a qubit is built iff some Pauli with nonzero
// probability in the channel has an X or Y factor on that qubit, the Z atom iff some has a Z or Y
// factor (a Y is composed as X then Z). Unbuilt slots hold a placeholder with present == false.
struct LocationEntry {
    int stream_index = -1;        // index of the Noise instruction in the stream
    std::vector<int> qubits;      // qubits the channel touches
    std::vector<PropAtom> x_atom; // one per qubit (parallel to `qubits`); read via atom()
    std::vector<PropAtom> z_atom; // one per qubit (parallel to `qubits`); read via atom()

    // The X (basis X) or Z (basis Z) atom of qubit slot `qi`. Throws std::invalid_argument
    // prefixed with `who` if that atom is absent (its factor has probability zero here).
    const PropAtom& atom(int qi, PauliBasis basis, const char* who) const;
};

// A single rejecting atom, for the whole-circuit reject diagnostic.
struct RejectInfo {
    int location_index = -1;      // stream index of the Noise instruction
    int qubit = -1;
    PauliBasis basis = PauliBasis::X;
    int reject_gate_index = -1;   // gate that left the class
};

// The per-location propagation table. all_in_class is false iff any atom rejected,
// in which case the WHOLE circuit is rejected and `rejects` lists every offending atom.
// has_general is true iff at least one atom was accepted via the PPR retry path: it carries a
// non-null PropResult::ppr residual, and its PropResult::general tableau only when the table was
// built with build_general.
struct PropagationTable {
    int n = 0;
    std::vector<LocationEntry> locations;
    bool all_in_class = true;
    std::vector<RejectInfo> rejects;
    bool has_general = false;

    // Locate the (LocationEntry, qubit-slot) a fired (stream_index, qubit) references; throws
    // std::logic_error prefixed with `who` if absent. Shared by compose_fired (sampler.cpp) and
    // the framed sampler's gather_atoms_ — one lookup, one error contract.
    std::pair<const LocationEntry*, int> find_slot(int stream_index, int qubit,
                                                   const char* who) const;
};

// Which atoms a table builds.
//   ChannelSupport (the samplers, DEM export, feedback plan): exactly the channel's support —
//     a factor with probability zero is never fired, so its atom is absent and can never reject
//     the circuit.
//   AnyPauli (the forced-fault program: ANY Pauli may be forced at a site, its probability is
//     irrelevant — docs/xtim_forced_faults.md): every X/Z atom is built, as before; a
//     zero-probability atom that leaves the class (after the PPR retry, when on) does not reject
//     the circuit but is marked absent (rejected = true), so forcing it is refused by name.
enum class AtomSet { ChannelSupport, AnyPauli };

// Build the table: for each Noise instruction and each touched qubit, propagate X_q and Z_q per
// `atoms` (above).
// If ppr_retry is true, any atom rejected by the standard DiagPauliClifford path is
// re-run with PprResidual; if the PPR residual propagates through the whole tail without
// rejection the atom is accepted, its residual stored in PropResult::ppr and — iff build_general
// — its full error tableau in PropResult::general (a dense n x n tableau per such atom: the twirl
// sampler, which never reads it, passes false).
// Each atom's end-of-circuit operator is propagate_atom's (the walk skips only gates that act
// trivially on it; see propagation_table.cpp).
// If sparse_cz, every atom is put in the sparse CZ form (sparsify_cz) the moment it is built, so at most one dense CZ
// layer is alive during the build (the twirl sampler's table: its readers go through dense_c_prop / cz_pairs).
// If sparse_av (lean L1), likewise the sparse a/v form (sparsify_av): at most one dense a / v pair is alive during
// the build; the twirl sampler passes both.
PropagationTable build_propagation_table(const Circuit& circ, bool ppr_retry = false,
                                         AtomSet atoms = AtomSet::ChannelSupport,
                                         bool build_general = true, bool sparse_cz = false, bool sparse_av = false);

// A diagonal atom's DiagPauliClifford (γ·X^v·diag(S^a)·∏CZ) promoted to a CliffordTableau by gate decomposition:
// diagonal layer (S/Z/S†) first, then the CZ layer, then X^v. The global scalar γ is unobservable in conjugation
// (dual_image is γ-insensitive), so the tableau is exact for E†·M·E. Dense: n² CZ scan.
CliffordTableau tableau_from_diag(const DiagPauliClifford& D);

// The composed end-of-circuit error tableau E = atom_last ∘ ... ∘ atom_first of one alternative whose atoms (in
// compose_fired's order) include at least one PPR-retry atom (PropResult::ppr non-null). Forward rows
// Xrow[a] = E·X_a·E† are built by chaining each atom's FORWARD action (CliffordTableau::forward_image); a diagonal atom
// is promoted by tableau_from_diag, a PPR atom uses its stored `general` tableau or, when the table was built without
// build_general, materializes ppr->to_error_tableau() here (only for the alternatives that need it). A single atom
// with a stored tableau is returned as is (no build). Shared by the framed sampler and the DEM export.
std::shared_ptr<const CliffordTableau> compose_error_tableau(const std::vector<const PropResult*>& atoms, int n);

// The class check: build_propagation_table(circ) (no PPR retry, AtomSet::ChannelSupport) if every
// atom stays in the diagonal + Pauli class, else nullopt — the build stops at the first atom that
// leaves it (the caller needs only the yes/no then).
std::optional<PropagationTable> build_propagation_table_if_in_class(const Circuit& circ, bool sparse = false);   // lean L6: sparse = the sampler's sparse CZ + a/v form

}  // namespace qeccore
