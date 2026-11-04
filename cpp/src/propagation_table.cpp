#include "qeccore/propagation_table.hpp"

#include <algorithm>
#include "qeccore/clifford_conjugate.hpp"
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

DiagPauliClifford dense_c_prop(const PropResult& a) {
    DiagPauliClifford D = a.c_prop_;
    if (!a.cz_sparse_ || a.cz_words_ == 0) return D;
    D.B.words_ = a.cz_words_;
    cz_rows_of(D.B.dim(), a.cz_words_, a.cz_pairs_, D.B.data_);
    return D;
}

PropResult propagate_atom(const Circuit& circ, int from_index, const DiagPauliClifford& seed) {
    PropResult r;
    r.c_prop_mut() = seed;
    r.support.assign((size_t)(circ.n + 63) / 64, 0);           // no walk: every wire
    for (int q = 0; q < circ.n; ++q) r.support[(size_t)(q >> 6)] |= 1ULL << (q & 63);
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
struct Walk {
    std::vector<GateRef> gates;
    std::vector<uint32_t> mark;
    std::vector<uint64_t> sup;                       // the current atom's marked qubits (PropResult::support)
    uint32_t stamp = 0;

    bool outside(const GateRef& G) const {
        return mark[(size_t)G.a] != stamp && (G.b < 0 || mark[(size_t)G.b] != stamp) &&
               (G.c < 0 || mark[(size_t)G.c] != stamp);
    }
    void begin(int q) { ++stamp; std::fill(sup.begin(), sup.end(), 0ULL); mark1(q); }
    void mark1(int q) { mark[(size_t)q] = stamp; sup[(size_t)(q >> 6)] |= 1ULL << (q & 63); }
    void touch(const GateRef& G) {
        mark1(G.a);
        if (G.b >= 0) mark1(G.b);
        if (G.c >= 0) mark1(G.c);
    }
    void support_into(std::vector<uint64_t>& out) const { out = sup; }
};

// propagate_atom(circ, from, seed on qubit q), over the gates from position `gi` on (the first gate
// after `from`).
PropResult walk_atom(Walk& W, size_t gi, int q, const DiagPauliClifford& seed) {
    PropResult r;
    r.c_prop_mut() = seed;
    W.begin(q);
    for (size_t i = gi; i < W.gates.size(); ++i) {
        const GateRef& G = W.gates[i];
        if (skippable(G.g) && W.outside(G)) continue;
        if (!conjugate_by_gate(r.c_prop_mut(), G.g, G.a, G.b, G.c)) {
            r.rejected = true;
            r.reject_gate_index = G.k;
            W.support_into(r.support);
            return r;
        }
        W.touch(G);
    }
    W.support_into(r.support);
    return r;
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
    W.begin(q);
    for (size_t i = gi; i < W.gates.size() && ppr_ok; ++i) {
        const GateRef& G = W.gates[i];
        if (skippable(G.g) && W.outside(G)) continue;
        ppr_ok = Rpr.propagate(G.g, circ.stream[(size_t)G.k].targets);
        if (!ppr_ok) ppr_reject_idx = G.k;
        else W.touch(G);
    }
    if (ppr_ok) {
        W.support_into(rx.support);                 // the accepted PPR walk's support (a rejected atom keeps the first walk's)
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
                             bool stop_at_reject, bool sparse_cz) {
    PropagationTable tbl;
    tbl.n = circ.n;
    Walk W;
    W.gates = compact_gates(circ);
    W.mark.assign((size_t)circ.n, 0);
    W.sup.assign((size_t)(circ.n + 63) / 64, 0);
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
        // One atom per (qubit, basis), X before Z as always. A factor the channel cannot fire is
        // absent: never propagated (ChannelSupport), or propagated but never a reject (AnyPauli).
        auto build_dense = [&](bool seedX, bool needed, int q) {
            PropResult r;
            if (!needed && atoms == AtomSet::ChannelSupport) { r.present = false; return r; }
            r = walk_atom(W, gi, q, seedX ? DiagPauliClifford::X(circ.n, q) : DiagPauliClifford::Z(circ.n, q));
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
                                         bool build_general, bool sparse_cz) {
    return build_table(circ, ppr_retry, atoms, build_general, /*stop_at_reject=*/false, sparse_cz);
}

std::optional<PropagationTable> build_propagation_table_if_in_class(const Circuit& circ, bool sparse_cz) {
    PropagationTable t = build_table(circ, /*ppr_retry=*/false, AtomSet::ChannelSupport,
                                     /*build_general=*/true, /*stop_at_reject=*/true, sparse_cz);
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
