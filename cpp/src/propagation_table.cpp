#include "qeccore/propagation_table.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_map>
#include "qeccore/clifford_conjugate.hpp"
#include "qeccore/pauli_kernels.hpp"   // pmul_into (the sweep's exact row products)
#include "qeccore/ppr_residual.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace qeccore {

std::pair<const LocationEntry*, int>
PropagationTable::find_slot(int stream_index, int qubit, const char* who) const {
    const LocationEntry* loc = nullptr;
    for (const LocationEntry& e : locations)
        if (e.stream_index == stream_index) { loc = &e; break; }
    if (!loc)                                   // always-on: public API, null-deref otherwise
        throw std::logic_error(std::string(who) + ": fired location not in the propagation table");
    int qi = -1;
    for (int i = 0; i < (int)loc->qubits.size(); ++i) if (loc->qubits[i] == qubit) { qi = i; break; }
    if (qi < 0)
        throw std::logic_error(std::string(who) + ": fired qubit not in the table's location entry");
    return {loc, qi};
}

const PropAtom& LocationEntry::atom(int qi, PauliBasis basis, const char* who) const {
    if (basis == PauliBasis::Y)                 // always-on: a Y is its X atom then its Z atom
        throw std::logic_error(std::string(who) + ": LocationEntry::atom takes X or Z (a Y factor is "
                               "read as its X atom then its Z atom)");
    const PropAtom& a = basis == PauliBasis::X ? x_atom[(size_t)qi] : z_atom[(size_t)qi];
    if (!a.present)
        throw std::invalid_argument(
            std::string(who) + ": a " + (basis == PauliBasis::X ? "X/Y" : "Z/Y") + " fault on deferred qubit " +
            std::to_string(qubits[(size_t)qi]) + " at deferred noise stream index " + std::to_string(stream_index) +
            " has probability zero in its channel" +
            (a.rejected ? " and its propagated error leaves the simulable class at deferred-stream gate index " +
                              std::to_string(a.reject_gate_index) + " (no atom for it)"
                        : " (no atom is built for it)"));
    return a;
}

// The channel's support per qubit slot: need_x[i] / need_z[i] iff some Pauli with nonzero
// probability has an X/Y (resp. Z/Y) factor on qubits[i]. Unioned over every slot holding the
// same qubit, so find_slot's first-match slot always carries everything that qubit can fire.
static void channel_support(const Instr& ins, std::vector<char>& need_x, std::vector<char>& need_z) {
    const size_t nq = ins.qubits.size();
    need_x.assign(nq, 0);
    need_z.assign(nq, 0);
    auto mark = [&](size_t i, PauliBasis b) {
        if (b == PauliBasis::X || b == PauliBasis::Y) need_x[i] = 1;
        if (b == PauliBasis::Z || b == PauliBasis::Y) need_z[i] = 1;
    };
    switch (ins.channel) {
        case NoiseChannel::X_ERROR:
        case NoiseChannel::Y_ERROR:
        case NoiseChannel::Z_ERROR: {
            const PauliBasis b = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                               : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z;
            if (ins.probs.at(0) > 0.0) for (size_t i = 0; i < nq; ++i) mark(i, b);
            break;
        }
        case NoiseChannel::DEPOLARIZE1:
        case NoiseChannel::PAULI_CHANNEL_1: {
            const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
            for (int t = 0; t < 3; ++t)
                if ((dep ? ins.probs.at(0) : ins.probs.at((size_t)t)) > 0.0)
                    for (size_t i = 0; i < nq; ++i) mark(i, pc1_slot_basis(t));
            break;
        }
        case NoiseChannel::DEPOLARIZE2:
        case NoiseChannel::PAULI_CHANNEL_2: {
            const bool dep = ins.channel == NoiseChannel::DEPOLARIZE2;
            for (int t = 0; t < 15; ++t) {
                if (!((dep ? ins.probs.at(0) : ins.probs.at((size_t)t)) > 0.0)) continue;
                for (size_t i = 0; i + 1 < nq; i += 2) {
                    if (const int c = pc2_first_code(t))  mark(i, pauli_code_basis(c));
                    if (const int c = pc2_second_code(t)) mark(i + 1, pauli_code_basis(c));
                }
            }
            break;
        }
        case NoiseChannel::CORRELATED: {            // E/ELSE chain: term t's product, if it can fire
            const std::vector<double> q = corr_outcome_probs(ins);
            for (int t = 0; t < corr_terms(ins); ++t) {
                if (!(q[(size_t)t] > 0.0)) continue;
                for (size_t i = 0; i < nq; ++i)
                    if (const int c = corr_code(ins, t, (int)i)) mark(i, pauli_code_basis(c));
            }
            break;
        }
        default:                                    // a new channel must declare its support here
            throw std::invalid_argument("propagation table: noise channel " +
                                        std::to_string((int)ins.channel) +
                                        " has no declared Pauli support (unsupported channel)");
    }
    for (size_t i = 0; i < nq; ++i)
        for (size_t j = 0; j < nq; ++j)
            if (ins.qubits[j] == ins.qubits[i]) { need_x[i] |= need_x[j]; need_z[i] |= need_z[j]; }
}

namespace {
// The dense rows (k rows of w words, zero-initialised in `out`) of the symmetric zero-diagonal B whose j < l pairs are
// `pairs`: the one rebuild behind dense_c_prop and sparsify_cz's check.
void cz_rows_of(int k, int w, const std::vector<std::pair<int, int>>& pairs, std::vector<uint64_t>& out) {
    out.assign((size_t)k * (size_t)w, 0ULL);
    for (const auto& pr : pairs) {
        out[(size_t)pr.first * (size_t)w + (size_t)(pr.second >> 6)] ^= 1ULL << (pr.second & 63);
        out[(size_t)pr.second * (size_t)w + (size_t)(pr.first >> 6)] ^= 1ULL << (pr.first & 63);
    }
}
}  // namespace

void sparsify_cz(PropResult& a) {
    if (a.cz_sparse_) return;
    SymPackedMat& B = a.c_prop_.B;
    const int k = B.dim(), w = B.words();
    a.cz_words_ = w;
    a.cz_pairs_.clear();
    for (int j = 0; j < k && w; ++j) {
        const uint64_t* row = B.row(j);
        for (int t = (j + 1) >> 6; t < w; ++t)
            for (uint64_t word = row[t]; word; word &= word - 1) {
                const int l = t * 64 + __builtin_ctzll(word);
                if (l >= k)                          // a padding bit: no pair can carry it (F1 below)
                    throw std::logic_error("sparsify_cz: a CZ-layer bit beyond the dimension (engine bug)");
                if (l > j) a.cz_pairs_.emplace_back(j, l);
            }
    }
    a.cz_pairs_.shrink_to_fit();
    // F1 (review 2026-10-30): the pairs must account for every dense bit before the rows are freed — the rows rebuilt
    // from them equal B word for word (a lower-only, diagonal or padding bit, or a lossy extraction, fails here). This
    // is the mechanism independent of the pairs' consumers (the alt-atom extraction and the window's compose both read
    // cz_pairs).
    if (w) {
        static thread_local std::vector<uint64_t> rb;
        cz_rows_of(k, w, a.cz_pairs_, rb);
        if (rb != B.data_)
            throw std::logic_error("sparsify_cz: the CZ pair list does not reproduce the dense CZ layer (engine bug)");
    }
    B.words_ = 0;                                    // the storage-free lazy zero (never read in the sparse form)
    std::vector<uint64_t>().swap(B.data_);
    a.cz_sparse_ = true;
}

void PropResult::sparse_read_() {
    throw std::logic_error(
        "PropResult::c_prop: the atom is in the sparse CZ form (read dense_c_prop / prop_a / prop_v / cz_pairs)");
}

void PropResult::dense_pairs_read_() {
    throw std::logic_error(
        "PropResult::cz_pairs: the atom is in the dense CZ form (read c_prop; cz_pairs exists in the sparse form only)");
}

namespace {
// The dense a / v vectors (n bytes each, zero-initialised in `a` / `v`) of the sparse a/v lists: the one rebuild
// behind dense_c_prop and sparsify_av's check.
void av_dense_of(size_t n, const std::vector<std::pair<int, uint8_t>>& al, const std::vector<int>& vl,
                 std::vector<uint8_t>& a, std::vector<uint8_t>& v) {
    a.assign(n, 0);
    v.assign(n, 0);
    for (const auto& e : al) a[(size_t)e.first] = e.second;
    for (int q : vl) v[(size_t)q] = 1;
}
}  // namespace

void sparsify_av(PropResult& a) {
    if (a.av_sparse_) return;
    const std::vector<uint8_t>& A = a.c_prop_.a;
    const std::vector<uint8_t>& V = a.c_prop_.v;
    if (A.size() != V.size() || (int)A.size() != a.c_prop_.n)
        throw std::logic_error("sparsify_av: the atom's a / v vectors are not n long (engine bug)");
    a.a_sparse_.clear();
    a.v_sparse_.clear();
    for (size_t q = 0; q < A.size(); ++q)
        if (A[q]) a.a_sparse_.emplace_back((int)q, (uint8_t)(A[q] & 3));   // the Z4 contract: a byte outside it fails F1
    for (size_t q = 0; q < V.size(); ++q)
        if (V[q]) a.v_sparse_.push_back((int)q);                            // the GF(2) contract: v rebuilt as 1
    a.a_sparse_.shrink_to_fit();
    a.v_sparse_.shrink_to_fit();
    // F1 (lean L1): the lists must reproduce the dense vectors byte for byte before the vectors are freed — the
    // mechanism independent of the lists' consumers (the sampler's alt extraction, compose_fired via dense_c_prop).
    {
        static thread_local std::vector<uint8_t> ra, rv;
        av_dense_of(A.size(), a.a_sparse_, a.v_sparse_, ra, rv);
        if (ra != A || rv != V)
            throw std::logic_error("sparsify_av: the sparse a/v lists do not reproduce the dense a / v vectors (engine bug)");
    }
    std::vector<uint8_t>().swap(a.c_prop_.a);
    std::vector<uint8_t>().swap(a.c_prop_.v);
    std::vector<uint64_t>().swap(a.support_);
    a.av_sparse_ = true;
}

void PropResult::sparse_av_read_() {
    throw std::logic_error(
        "PropResult::prop_a / prop_v / support: the atom is in the sparse a/v form (read dense_c_prop / sparse_a / sparse_v)");
}

void PropResult::dense_av_read_() {
    throw std::logic_error(
        "PropResult::sparse_a / sparse_v: the atom is in the dense a/v form (read prop_a / prop_v; the lists exist in the sparse form only)");
}

DiagPauliClifford dense_c_prop(const PropResult& a) {
    DiagPauliClifford D = a.c_prop_;
    if (a.av_sparse_) av_dense_of((size_t)D.n, a.a_sparse_, a.v_sparse_, D.a, D.v);
    if (!a.cz_sparse_ || a.cz_words_ == 0) return D;
    D.B.words_ = a.cz_words_;
    cz_rows_of(D.B.dim(), a.cz_words_, a.cz_pairs_, D.B.data_);
    return D;
}

PropResult propagate_atom(const Circuit& circ, int from_index, const DiagPauliClifford& seed) {
    PropResult r;
    r.c_prop_mut() = seed;
    std::vector<uint64_t>& sup = r.support_mut();
    sup.assign((size_t)(circ.n + 63) / 64, 0);                 // no walk: every wire
    for (int q = 0; q < circ.n; ++q) sup[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    for (int k = from_index + 1; k < (int)circ.stream.size(); ++k) {
        const Instr& ins = circ.stream[k];
        if (ins.kind != Instr::Kind::Gate) continue;        // Noise/Measure/Reset/Observable transparent
        assert(!ins.targets.empty() && "canonical IR: a Gate instruction must carry arity-sized targets");
        int a = ins.targets[0];
        int b = ins.targets.size() > 1 ? ins.targets[1] : -1;
        int c = ins.targets.size() > 2 ? ins.targets[2] : -1;
        if (!conjugate_by_gate(r.c_prop_mut(), ins.gate, a, b, c)) {
            r.rejected = true;
            r.reject_gate_index = k;
            return r;
        }
    }
    return r;
}

// ── The table's propagation walk (F-F, compile stage 2026-10-24) ─────────────────────────────────
// propagate_atom (above) walks every later instruction of the stream; the table's walk below gives
// the identical result over a compact list of the Gate instructions, skipping a gate when every one
// of its targets lies outside the atom's SUPPORT. Exactness:
//   * support := the seed qubit ∪ the targets of every gate the walk has applied. It is a superset of
//     the qubits where the operator acts: a gate writes its own targets only, and the couplings it
//     follows (conjugate_by_gate's B-row scans for X/Y/CX; PprResidual's axes) end on qubits already
//     touched.
//   * every case of conjugate_by_gate except CH is the identity on an operator with no X part, no
//     diagonal phase and no coupling on the gate's targets (each case reads only v / a / B at its
//     targets); likewise PprResidual::propagate on a residual whose Pauli and axes miss the targets.
//   * CH (it always leaves the diagonal class, wherever it acts) and any gate kind not listed in
//     skippable() are always applied, exactly as before.
// tests/test_compile_speed.py holds this walk to propagate_atom (and to the unskipped PPR walk)
// atom for atom over the corpus and random circuits.
namespace {

struct GateRef { int k; GateKind g; int a, b, c; };   // a Gate instruction: stream index, kind, targets

std::vector<GateRef> compact_gates(const Circuit& circ) {
    std::vector<GateRef> out;
    for (int k = 0; k < (int)circ.stream.size(); ++k) {
        const Instr& ins = circ.stream[k];
        if (ins.kind != Instr::Kind::Gate) continue;
        assert(!ins.targets.empty() && "canonical IR: a Gate instruction must carry arity-sized targets");
        out.push_back({k, ins.gate, ins.targets[0], ins.targets.size() > 1 ? ins.targets[1] : -1,
                       ins.targets.size() > 2 ? ins.targets[2] : -1});
    }
    return out;
}

bool skippable(GateKind g) {
    switch (g) {
        case GateKind::X: case GateKind::Y: case GateKind::Z: case GateKind::S: case GateKind::SDG:
        case GateKind::CZ: case GateKind::CX: case GateKind::H: case GateKind::T: case GateKind::CS:
        case GateKind::CCZ:
            return true;
        case GateKind::CH:
            return false;
    }
    return false;
}

// The walk state of one table build: the compact gate list and the support marks (a qubit is in the
// current atom's support iff mark[q] == stamp; a new stamp per atom clears it in O(1)).
//
// lean L6: the operator is ONE scratch DiagPauliClifford per build (a / v over all n wires, zero between atoms):
// the seed is written, the walk runs, the result is read off the support's wires and the scratch is cleared over
// them — and then CHECKED all-zero (the walk writes its targets only; a stray byte is an engine bug, thrown by
// name). The CZ layer B is the atom's own (lazy: it materialises on the first flip exactly as before, so
// cz_words() is unchanged). No per-atom n-byte seed, no per-atom n-byte sparsify scan.
//
// lean L6, the gate selection: the linear scan visited every later gate and applied the skippable ones whose targets
// met the support. Here every wire holds a bitset over the gate positions of the skippable gates touching it, the
// non-skippable ones (CH) form one more bitset, and an atom's PENDING positions are the OR of the bitsets of its
// marked wires (ORed from the current position on when the wire is marked) plus the always-bitset: the next gate to
// apply is the next set bit. A position is pending iff a wire marked before it is reached has the gate, or the gate is
// non-skippable — exactly the scan's condition — so the applied gates, their order and the marks are identical;
// the cost is O(|support| · gates/64) words per atom instead of a test per later gate.
// lean L6: the walk's CZ layer, support-local. A row (W words over the GLOBAL wires, as the dense layer's) exists only
// for a wire a flip touched; untouched wires read the shared zero row. words() reports the dense width once the first
// flip has happened and 0 before (the dense layer's lazy-zero form: the walk's cz_words() is unchanged). The rule
// (conjugate_layer) is the dense one, instantiated on this type; the pairs are read off the touched rows ascending
// (the dense sparsify_cz's order) and the rows are zeroed and CHECKED zero between atoms.
struct WalkB {
    int n = 0, W = 0;
    bool touched = false;
    std::vector<int> row_of;                         // wire -> row index, -1 = none
    std::vector<int> used;                           // the wires with a row (in creation order)
    std::vector<uint64_t> rows;                      // used.size() rows of W words
    std::vector<uint64_t> zero;                      // W words of zero
    void init(int n_) { n = n_; W = (n_ + 63) / 64; row_of.assign((size_t)n_, -1); zero.assign((size_t)W, 0); }
    int words() const { return touched ? W : 0; }
    int dim() const { return n; }
    const uint64_t* row(int i) const { const int r = row_of[(size_t)i]; return r < 0 ? zero.data() : rows.data() + (size_t)r * W; }
    bool get(int i, int j) const { return (row(i)[j >> 6] >> (j & 63)) & 1ULL; }
    uint64_t* ensure(int i) {
        int r = row_of[(size_t)i];
        if (r < 0) { r = (int)used.size(); row_of[(size_t)i] = r; used.push_back(i); rows.resize(rows.size() + (size_t)W, 0); }
        return rows.data() + (size_t)r * W;
    }
    void flip_sym(int i, int j) {
        if (i == j) return;
        touched = true;
        ensure(i)[j >> 6] ^= 1ULL << (j & 63);
        ensure(j)[i >> 6] ^= 1ULL << (i & 63);
    }
    // The (j, l) pairs with j < l, ascending (j, then l) — sparsify_cz's extraction over the touched rows.
    void pairs_into(std::vector<std::pair<int, int>>& out) const {
        out.clear();
        std::vector<int> js(used);
        std::sort(js.begin(), js.end());
        for (int j : js) {
            const uint64_t* r = row(j);
            for (int t = (j + 1) >> 6; t < W; ++t)
                for (uint64_t word = r[t]; word; word &= word - 1) {
                    const int l = t * 64 + __builtin_ctzll(word);
                    if (l >= n) throw std::logic_error("propagation table: a CZ-layer bit beyond the dimension (engine bug)");
                    if (l > j) out.emplace_back(j, l);
                }
        }
    }
    // Forget the touched rows (the storage is reused; a new row is zero-filled on creation).
    void reset() {
        for (int q : used) row_of[(size_t)q] = -1;
        used.clear();
        rows.clear();
        touched = false;
    }
};

struct Walk {
    std::vector<GateRef> gates;
    std::vector<uint32_t> mark;
    std::vector<uint64_t> sup;                       // the current atom's marked qubits (PropResult::support)
    uint32_t stamp = 0;
    std::vector<int> touched;                        // the support as a list
    DiagPauliClifford op;                            // the scratch operator (a, v zero between atoms)
    WalkB B;                                         // the scratch CZ layer (support-local)
    int PW = 0;                                      // words of a gate-position bitset
    std::vector<uint64_t> wire_bits;                 // per wire: the positions of the skippable gates touching it
    std::vector<uint64_t> always_bits;               // the positions of the non-skippable gates
    std::vector<uint64_t> pending;                   // the current atom's pending positions
    size_t cur = 0;                                  // the next position to consider

    void init(const Circuit& circ) {
        gates = compact_gates(circ);
        mark.assign((size_t)circ.n, 0);
        sup.assign((size_t)(circ.n + 63) / 64, 0);
        op = DiagPauliClifford::identity(circ.n);
        B.init(circ.n);
        PW = (int)((gates.size() + 63) / 64);
    }
    // The gate-position bitsets (n x PW words): built only when some atom is walked (lean L6 review F3: a sparse build
    // whose atoms the Clifford-suffix sweep serves all never reads them).
    void index_positions() {
        const size_t n = mark.size();
        wire_bits.assign(n * (size_t)PW, 0);
        always_bits.assign((size_t)PW, 0);
        pending.assign((size_t)PW, 0);
        for (size_t i = 0; i < gates.size(); ++i) {
            const GateRef& G = gates[i];
            const uint64_t bit = 1ULL << (i & 63);
            const size_t w = i >> 6;
            if (!skippable(G.g)) { always_bits[w] |= bit; continue; }
            wire_bits[(size_t)G.a * PW + w] |= bit;
            if (G.b >= 0) wire_bits[(size_t)G.b * PW + w] |= bit;
            if (G.c >= 0) wire_bits[(size_t)G.c * PW + w] |= bit;
        }
    }
    void or_wire(int q) {                            // the wire's positions from the current one on
        const uint64_t* wb = &wire_bits[(size_t)q * PW];
        for (size_t w = cur >> 6; w < (size_t)PW; ++w) pending[w] |= wb[w];
    }
    void begin(int q, size_t gi) {
        if (pending.size() != (size_t)PW) throw std::logic_error("propagation table: a walk before index_positions (engine bug)");
        ++stamp;
        std::fill(sup.begin(), sup.end(), 0ULL);
        touched.clear();
        cur = gi;
        for (size_t w = cur >> 6; w < (size_t)PW; ++w) pending[w] = always_bits[w];
        mark1(q);
    }
    // The next gate to apply (its index into `gates`), or -1 when none is left.
    int next() {
        size_t w = cur >> 6;
        if (w >= (size_t)PW) return -1;
        uint64_t bits = pending[w] & (~0ULL << (cur & 63));
        while (!bits) { if (++w >= (size_t)PW) return -1; bits = pending[w]; }
        const size_t i = (w << 6) + (size_t)__builtin_ctzll(bits);
        cur = i + 1;
        return (int)i;
    }
    void mark1(int q) {
        if (mark[(size_t)q] == stamp) return;
        mark[(size_t)q] = stamp; sup[(size_t)(q >> 6)] |= 1ULL << (q & 63); touched.push_back(q);
        or_wire(q);
    }
    void touch(const GateRef& G) {
        mark1(G.a);
        if (G.b >= 0) mark1(G.b);
        if (G.c >= 0) mark1(G.c);
    }
    void support_into(std::vector<uint64_t>& out) const { out = sup; }
};

}  // namespace

// The walk's writes into the atom (a friend of PropResult).
struct WalkAccess {
// Read the walked operator off the scratch over the support (ascending wires) and clear the scratch there; then the
// scratch must be all-zero. sparse: the a/v lists (PropResult's sparse form; support_ stays until the build is done
// with it, see drop_support); dense: the a/v vectors copied out.
static void take_av(Walk& W, PropResult& r, bool sparse) {
    std::sort(W.touched.begin(), W.touched.end());
    // The Z4 / GF(2) contract (lean L6 review F2: sparsify_av's L1-review check on a dense atom): the walk writes the
    // lists directly, so a byte outside it throws here, over the support wires only.
    for (int q : W.touched)
        if (W.op.a[(size_t)q] > 3 || W.op.v[(size_t)q] > 1)
            throw std::logic_error("propagation table: take_av: the walked a byte is outside Z4 or the v byte outside GF(2) (engine bug)");
    if (sparse) {
        r.a_sparse_.clear();
        r.v_sparse_.clear();
        for (int q : W.touched) {
            if (W.op.a[(size_t)q]) r.a_sparse_.emplace_back(q, W.op.a[(size_t)q]);
            if (W.op.v[(size_t)q]) r.v_sparse_.push_back(q);
        }
        r.a_sparse_.shrink_to_fit();
        r.v_sparse_.shrink_to_fit();
        std::vector<uint8_t>().swap(r.c_prop_.a);
        std::vector<uint8_t>().swap(r.c_prop_.v);
        r.av_sparse_ = true;
    } else {
        r.c_prop_.a = W.op.a;
        r.c_prop_.v = W.op.v;
    }
    for (int q : W.touched) { W.op.a[(size_t)q] = 0; W.op.v[(size_t)q] = 0; }
    const uint8_t* pa = W.op.a.data();
    const uint8_t* pv = W.op.v.data();
    const size_t n = W.op.a.size();
    uint64_t acc = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t wa, wv; std::memcpy(&wa, pa + i, 8); std::memcpy(&wv, pv + i, 8); acc |= wa | wv; }
    for (; i < n; ++i) acc |= pa[i] | pv[i];
    if (acc)
        throw std::logic_error("propagation table: the walk wrote a / v outside the atom's support (engine bug)");
}
// The operator's n / gamma / CZ layer from the scratch. sparse_cz: the pair list straight off the touched rows (the
// sparse form sparsify_cz leaves: cz_words = the dense width iff a flip happened, the lazy all-zero dense layer);
// dense: the dense layer rebuilt from the rows (materialised iff a flip happened, as before).
static void take_core(Walk& W, PropResult& r, bool sparse_cz) {
    r.c_prop_.n = W.op.n;
    r.c_prop_.gamma = W.op.gamma;
    r.c_prop_.B = sym_zeros(W.op.n);
    if (sparse_cz) {
        r.cz_words_ = W.B.words();
        W.B.pairs_into(r.cz_pairs_);
        r.cz_pairs_.shrink_to_fit();
        r.cz_sparse_ = true;
    } else if (W.B.touched) {
        r.c_prop_.B.materialize_zero();
        for (int q : W.B.used) std::memcpy(r.c_prop_.B.row(q), W.B.row(q), (size_t)W.B.W * sizeof(uint64_t));
    }
    W.B.reset();
    W.support_into(r.support_);
}
static void set_support(Walk& W, PropResult& r) { W.support_into(r.support_); }   // the PPR retry's walk
static void drop_support(PropResult& r) { if (r.av_sparse_) std::vector<uint64_t>().swap(r.support_); }
// lean L6: an atom served by the Clifford-suffix sweep (below): the Pauli image P = U_suffix · seed · U_suffix† in
// the sparse form the walk + sparsify_cz / sparsify_av would have left — v = P's x bits, a_q = 2 on P's z bits,
// gamma = zeta8^{2·phase} (Y = i·X·Z: the operator is gamma · X^v · Z^{a/2}), B the lazy all-zero layer (no flip
// ever happens on a Pauli), no support (freed in the sparse form), present, never rejected.
static void from_pauli(const Pauli& P, int n, PropResult& r) {
    r.c_prop_.n = n;
    r.c_prop_.gamma = ExactPhase::one();
    r.c_prop_.gamma.add_z8(2 * (P.phase & 3));
    r.c_prop_.B = sym_zeros(n);
    const int W = (int)P.x.size();
    r.v_sparse_.clear();
    r.a_sparse_.clear();
    for (int w = 0; w < W; ++w)
        for (uint64_t bits = P.x[(size_t)w]; bits; bits &= bits - 1) r.v_sparse_.push_back((w << 6) + __builtin_ctzll(bits));
    for (int w = 0; w < W; ++w)
        for (uint64_t bits = P.z[(size_t)w]; bits; bits &= bits - 1) r.a_sparse_.emplace_back((w << 6) + __builtin_ctzll(bits), (uint8_t)2);
    r.v_sparse_.shrink_to_fit();
    r.a_sparse_.shrink_to_fit();
    r.av_sparse_ = true;
    sparsify_cz(r);
}
};

namespace {

// propagate_atom(circ, from, seed on qubit q), over the gates from position `gi` on (the first gate
// after `from`).
PropResult walk_atom(Walk& W, size_t gi, int q, bool seedX, bool sparse_cz, bool sparse_av) {
    PropResult r;
    W.op.gamma = ExactPhase::one();
    if (seedX) W.op.v[(size_t)q] = 1; else W.op.a[(size_t)q] = 2;
    W.begin(q, gi);
    for (int i = W.next(); i >= 0; i = W.next()) {
        const GateRef& G = W.gates[(size_t)i];
        if (!conjugate_layer(W.op.a, W.B, W.op.v, W.op.gamma, G.g, G.a, G.b, G.c)) {
            r.rejected = true;
            r.reject_gate_index = G.k;
            break;
        }
        W.touch(G);
    }
    WalkAccess::take_core(W, r, sparse_cz);
    WalkAccess::take_av(W, r, sparse_av);
    return r;
}

// ── lean L6: the Clifford-suffix sweep ────────────────────────────────────────────────────────────
// For an atom whose remaining circuit (the gates from its first position on) is Clifford-only, the walk is the
// exact conjugation of a Pauli seed through those gates: a Pauli P = U·seed·U† with U the suffix Clifford, which
// the walk's rules compute gate by gate (every case of conjugate_by_gate on a Pauli operator is that gate's
// conjugation rule, phases included; a skipped gate acts as the identity; no H can reject a Pauli). One backward
// sweep over the gates keeps the images U X_q U† / U Z_q U† of every wire (U grows by right-composition: the gate's
// own conjugation rule on the one or two touched rows, exact Pauli products), and serves every noise location it
// passes: the X atom of wire q IS the current image of X_q, the Z atom that of Z_q — O(n/64) per atom instead of a
// scan of every later gate. Atoms before the last non-Clifford gate (T, CS, CCZ, CH) keep the walk. Only the sparse
// a/v build takes it (a dense table also carries the walk's support, which the sweep does not compute).
// tests/test_lean_l1.py (`_sparse_av_mismatches`: the sparse table vs the dense one, atom for atom) and the digest
// pins (tests/test_lean_l6.py) hold the sweep to the walk.
bool clifford_gate(GateKind g) {
    switch (g) {
        case GateKind::X: case GateKind::Y: case GateKind::Z: case GateKind::S: case GateKind::SDG:
        case GateKind::CZ: case GateKind::CX: case GateKind::H:
            return true;
        case GateKind::T: case GateKind::CS: case GateKind::CCZ: case GateKind::CH:
            return false;
    }
    return false;
}

struct SweptLocation { std::vector<PropResult> x_atom, z_atom; };   // per served noise location (parallel to qubits)

// The sweep: results for every noise location whose first gate index gi >= first_cliff (the index after the last
// non-Clifford gate), keyed by the location's stream index. need(p, i, basisX) says whether that atom is built.
std::unordered_map<int, SweptLocation> sweep_clifford_suffix(
        const Circuit& circ, const std::vector<GateRef>& gates, size_t first_cliff,
        const std::vector<std::pair<int, size_t>>& noise_gi,          // (stream index p, gi) in stream order
        const std::function<bool(int, size_t, bool)>& need) {
    std::unordered_map<int, SweptLocation> out;
    const int n = circ.n;
    std::vector<Pauli> X((size_t)n, Pauli(n)), Z((size_t)n, Pauli(n));
    for (int q = 0; q < n; ++q) { X[(size_t)q].setx(q); Z[(size_t)q].setz(q); }
    // the locations by gi, served when the suffix is exactly gates[gi..)
    std::vector<std::vector<int>> by_gi(gates.size() + 1);
    for (const auto& pg : noise_gi) if (pg.second >= first_cliff) by_gi[pg.second].push_back(pg.first);
    auto serve = [&](size_t gi) {
        for (int p : by_gi[gi]) {
            const Instr& ins = circ.stream[(size_t)p];
            SweptLocation L;
            for (size_t i = 0; i < ins.qubits.size(); ++i) {
                const int q = ins.qubits[i];
                PropResult rx, rz;                   // an absent atom is the sparse default, as the walk's build leaves it
                if (need(p, i, true)) WalkAccess::from_pauli(X[(size_t)q], n, rx); else { rx.present = false; sparsify_cz(rx); sparsify_av(rx); }
                if (need(p, i, false)) WalkAccess::from_pauli(Z[(size_t)q], n, rz); else { rz.present = false; sparsify_cz(rz); sparsify_av(rz); }
                L.x_atom.push_back(std::move(rx));
                L.z_atom.push_back(std::move(rz));
            }
            out.emplace(p, std::move(L));
        }
    };
    Pauli t(n);
    serve(gates.size());
    for (size_t i = gates.size(); i-- > first_cliff;) {
        const GateRef& G = gates[i];
        switch (G.g) {                       // U <- U·g: the touched rows become U(g σ g†)U†, exact products
            case GateKind::H:   std::swap(X[(size_t)G.a], Z[(size_t)G.a]); break;
            case GateKind::S:   pmul_into(X[(size_t)G.a], Z[(size_t)G.a]); X[(size_t)G.a].phase = (X[(size_t)G.a].phase + 1) & 3; break;   // S X S† = i·X·Z
            case GateKind::SDG: pmul_into(X[(size_t)G.a], Z[(size_t)G.a]); X[(size_t)G.a].phase = (X[(size_t)G.a].phase + 3) & 3; break;   // S† X S = −i·X·Z
            case GateKind::X:   Z[(size_t)G.a].phase = (Z[(size_t)G.a].phase + 2) & 3; break;                                               // X Z X = −Z
            case GateKind::Y:   X[(size_t)G.a].phase = (X[(size_t)G.a].phase + 2) & 3; Z[(size_t)G.a].phase = (Z[(size_t)G.a].phase + 2) & 3; break;
            case GateKind::Z:   X[(size_t)G.a].phase = (X[(size_t)G.a].phase + 2) & 3; break;                                               // Z X Z = −X
            case GateKind::CX:                                                                                                              // X_c → X_c X_t, Z_t → Z_c Z_t
                pmul_into(X[(size_t)G.a], X[(size_t)G.b]);
                t = Z[(size_t)G.a]; pmul_into(t, Z[(size_t)G.b]); Z[(size_t)G.b] = t;
                break;
            case GateKind::CZ:                                                                                                              // X_c → X_c Z_t, X_t → X_t Z_c
                t = X[(size_t)G.a]; pmul_into(t, Z[(size_t)G.b]);
                pmul_into(X[(size_t)G.b], Z[(size_t)G.a]);
                X[(size_t)G.a] = t;
                break;
            default:
                throw std::logic_error("propagation table: a non-Clifford gate inside the Clifford suffix (engine bug)");
        }
        serve(i);
    }
    return out;
}

// Helper: attempt PPR retry for one atom (seedX selects X vs Z seed).
// If the PPR residual propagates the entire tail without rejection, sets rx.rejected=false,
// rx.reject_gate_index=-1, rx.ppr to the residual and — iff build_general — rx.general to its full
// error tableau. If the PPR residual also rejects, updates rx.reject_gate_index to the PPR reject
// position.
void ppr_retry_atom(PropResult& rx, const Circuit& circ, Walk& W, size_t gi, bool seedX, int q,
                    bool build_general, PropagationTable& tbl) {
    PprResidual Rpr = seedX ? ppr_seed_x(circ.n, q) : ppr_seed_z(circ.n, q);
    bool ppr_ok = true;
    int ppr_reject_idx = -1;
    W.begin(q, gi);
    for (int i = W.next(); i >= 0 && ppr_ok; i = W.next()) {
        const GateRef& G = W.gates[(size_t)i];
        ppr_ok = Rpr.propagate(G.g, circ.stream[(size_t)G.k].targets);
        if (!ppr_ok) ppr_reject_idx = G.k;
        else W.touch(G);
    }
    if (ppr_ok) {
        WalkAccess::set_support(W, rx);             // the accepted PPR walk's support (a rejected atom keeps the first walk's)
        rx.rejected = false;
        rx.reject_gate_index = -1;
        if (build_general) rx.general = std::make_shared<CliffordTableau>(Rpr.to_error_tableau());
        rx.ppr = std::make_shared<PprResidual>(std::move(Rpr));  // V2: keep the axis list for the twirl law
        tbl.has_general = true;
    } else {
        rx.reject_gate_index = ppr_reject_idx;
        // rx.rejected stays true
    }
}

// The one table builder. stop_at_reject (the class check): return at the first atom that rejects
// the circuit, the table then partial with all_in_class == false.
PropagationTable build_table(const Circuit& circ, bool ppr_retry, AtomSet atoms, bool build_general,
                             bool stop_at_reject, bool sparse_cz, bool sparse_av) {
    PropagationTable tbl;
    tbl.n = circ.n;
    Walk W;
    W.init(circ);
    // lean L6: the Clifford-suffix sweep serves every atom after the last non-Clifford gate (sparse a/v builds).
    std::unordered_map<int, SweptLocation> swept;
    bool walked = !sparse_av;                       // some noise location is left to the walk
    if (sparse_av) {
        size_t first_cliff = 0;
        for (size_t i = 0; i < W.gates.size(); ++i) if (!clifford_gate(W.gates[i].g)) first_cliff = i + 1;
        std::vector<std::pair<int, size_t>> noise_gi;
        std::unordered_map<int, std::pair<std::vector<char>, std::vector<char>>> needs;
        size_t g2 = 0;
        for (int p = 0; p < (int)circ.stream.size(); ++p) {
            const Instr& ins = circ.stream[(size_t)p];
            if (ins.kind != Instr::Kind::Noise) continue;
            while (g2 < W.gates.size() && W.gates[g2].k <= p) ++g2;
            if (g2 < first_cliff) { walked = true; continue; }
            noise_gi.emplace_back(p, g2);
            auto& nd = needs[p];
            channel_support(ins, nd.first, nd.second);
        }
        swept = sweep_clifford_suffix(circ, W.gates, first_cliff, noise_gi,
                                      [&](int p, size_t i, bool bx) {
                                          const auto& nd = needs.at(p);
                                          return atoms != AtomSet::ChannelSupport || (bx ? nd.first[i] : nd.second[i]);
                                      });
    }
    if (walked) W.index_positions();
    size_t gi = 0;                                  // first gate after the current Noise instruction
    for (int p = 0; p < (int)circ.stream.size(); ++p) {
        const Instr& ins = circ.stream[p];
        if (ins.kind != Instr::Kind::Noise) continue;
        while (gi < W.gates.size() && W.gates[gi].k <= p) ++gi;
        LocationEntry entry;
        entry.stream_index = p;
        entry.qubits = ins.qubits;
        std::vector<char> need_x, need_z;
        channel_support(ins, need_x, need_z);
        auto sw = swept.find(p);
        if (sw != swept.end()) {                    // served by the sweep: in class, nothing to retry
            entry.x_atom = std::move(sw->second.x_atom);
            entry.z_atom = std::move(sw->second.z_atom);
            tbl.locations.push_back(std::move(entry));
            continue;
        }
        // One atom per (qubit, basis), X before Z as always. A factor the channel cannot fire is
        // absent: never propagated (ChannelSupport), or propagated but never a reject (AnyPauli).
        auto build_dense = [&](bool seedX, bool needed, int q) {
            PropResult r;
            if (!needed && atoms == AtomSet::ChannelSupport) { r.present = false; return r; }
            r = walk_atom(W, gi, q, seedX, sparse_cz, sparse_av);
            if (r.rejected && ppr_retry)
                ppr_retry_atom(r, circ, W, gi, seedX, q, build_general, tbl);
            if (r.rejected && !needed) { r.present = false; return r; }
            if (r.rejected) {
                tbl.all_in_class = false;
                tbl.rejects.push_back({p, q, seedX ? PauliBasis::X : PauliBasis::Z, r.reject_gate_index});
            }
            return r;
        };
        auto build = [&](bool seedX, bool needed, int q) {   // czrows: every atom of a sparse_cz table (absent too)
            PropResult r = build_dense(seedX, needed, q);
            if (sparse_cz) sparsify_cz(r);
            if (sparse_av) { sparsify_av(r); WalkAccess::drop_support(r); }   // lean L1 / L6: a walked atom is already sparse
            return r;
        };
        for (size_t i = 0; i < ins.qubits.size(); ++i) {
            entry.x_atom.push_back(build(true, need_x[i], ins.qubits[i]));
            if (stop_at_reject && !tbl.all_in_class) break;
            entry.z_atom.push_back(build(false, need_z[i], ins.qubits[i]));
            if (stop_at_reject && !tbl.all_in_class) break;
        }
        tbl.locations.push_back(std::move(entry));
        if (stop_at_reject && !tbl.all_in_class) break;
    }
    // has_general is only meaningful when the circuit is fully in-class: a partially-accepted
    // table is rejected by the caller, so general residuals in it are unused. Clear the flag
    // when all_in_class is false to avoid misleading callers.
    if (!tbl.all_in_class) tbl.has_general = false;
    return tbl;
}

}  // namespace

PropagationTable build_propagation_table(const Circuit& circ, bool ppr_retry, AtomSet atoms,
                                         bool build_general, bool sparse_cz, bool sparse_av) {
    return build_table(circ, ppr_retry, atoms, build_general, /*stop_at_reject=*/false, sparse_cz, sparse_av);
}

std::optional<PropagationTable> build_propagation_table_if_in_class(const Circuit& circ, bool sparse) {
    // lean L6: `sparse` = the twirl sampler's form (sparse CZ AND sparse a/v: the sampler is its only consumer and
    // puts the table in that form anyway) — the sparse a/v build is the one the Clifford-suffix sweep serves.
    PropagationTable t = build_table(circ, /*ppr_retry=*/false, AtomSet::ChannelSupport,
                                     /*build_general=*/true, /*stop_at_reject=*/true, sparse, /*sparse_av=*/sparse);
    if (!t.all_in_class) return std::nullopt;
    return t;
}

CliffordTableau tableau_from_diag(const DiagPauliClifford& D) {
    CliffordTableau T(D.n);
    for (int q = 0; q < D.n; ++q)
        switch (D.a[q] & 3) {
            case 1: T.left_s(q); break;
            case 2: T.left_z(q); break;
            case 3: T.left_sdg(q); break;
            default: break;
        }
    for (int i = 0; i < D.n; ++i)
        for (int j = i + 1; j < D.n; ++j)
            if (D.B.get(i, j)) T.left_cz(i, j);
    for (int q = 0; q < D.n; ++q)
        if (D.v[q]) T.left_x(q);
    return T;
}

// Bug note (framed sampler history): using conjugate() (= pullback T†·P·T) here would store dual rows in Xrow and make
// a later dual_image(M) return E·M·E† instead of E†·M·E, flipping outcome signs for non-Pauli composed residuals.
std::shared_ptr<const CliffordTableau> compose_error_tableau(const std::vector<const PropResult*>& atoms, int n) {
    if (atoms.size() == 1 && atoms[0]->general) return atoms[0]->general;
    if (atoms.size() == 1 && atoms[0]->ppr) return std::make_shared<CliffordTableau>(atoms[0]->ppr->to_error_tableau());
    int nt = 0;
    for (const PropResult* a : atoms) if (!a->general) ++nt;
    std::vector<CliffordTableau> temps;
    temps.reserve((size_t)nt);                                 // stable: all pushes precede tab()
    std::vector<int> tidx(atoms.size(), -1);
    for (size_t i = 0; i < atoms.size(); ++i) {
        if (atoms[i]->general) continue;
        tidx[i] = (int)temps.size();
        temps.push_back(atoms[i]->ppr ? atoms[i]->ppr->to_error_tableau() : tableau_from_diag(atoms[i]->c_prop()));
    }
    auto tab = [&](size_t i) -> const CliffordTableau& {
        return atoms[i]->general ? *atoms[i]->general : temps[(size_t)tidx[i]];
    };
    CliffordTableau E(n);
    for (int a = 0; a < n; ++a) {
        Pauli x(n); x.setx(a);
        Pauli z(n); z.setz(a);
        for (size_t i = 0; i < atoms.size(); ++i) { x = tab(i).forward_image(x); z = tab(i).forward_image(z); }
        E.Xrow[a] = std::move(x);
        E.Zrow[a] = std::move(z);
    }
    E.invalidate_dual();                                       // rows set directly; dual_image reads forward rows
    return std::make_shared<CliffordTableau>(std::move(E));
}

}  // namespace qeccore
