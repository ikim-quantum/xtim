// Compile stage (2026-10-24) test API (not a public interface): the oracles behind the construction-speed
// changes. tests/test_compile_speed.py drives them.
//   _push_table_oracle_mismatches(): the frame-push table (inverted search) vs try_push's brute force, all 1152
//     inputs (F-G).
//   _propagation_walk_oracle(text, normalized): the propagation table's support-skipping walk vs the unskipped
//     walks (propagate_atom; the PPR retry's gate-by-gate loop), atom for atom (F-F); the early-stopping class
//     check vs the full table (F-D); the general-free table vs the default one (F-A); the sparse-CZ tables vs the
//     dense ones (czrows).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "qeccore/clifford_frames.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/ppr_residual.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/stim_parse.hpp"

namespace py = pybind11;
using namespace qeccore;

namespace {

bool same_dpc(const DiagPauliClifford& x, const DiagPauliClifford& y) {
    return x.n == y.n && x.gamma.is_zero == y.gamma.is_zero && x.gamma.scale == y.gamma.scale &&
           x.gamma.z8 == y.gamma.z8 && x.a == y.a && x.v == y.v && x.B.dim() == y.B.dim() &&
           x.B.words() == y.B.words() && x.B.data_ == y.B.data_;
}
bool same_pauli(const Pauli& x, const Pauli& y) { return x.x == y.x && x.z == y.z && x.phase == y.phase; }
bool same_tableau(const CliffordTableau& x, const CliffordTableau& y) {
    if (x.n != y.n) return false;
    for (int q = 0; q < x.n; ++q)
        if (!same_pauli(x.Xrow[(size_t)q], y.Xrow[(size_t)q]) || !same_pauli(x.Zrow[(size_t)q], y.Zrow[(size_t)q]))
            return false;
    return true;
}
bool same_ppr(const PprResidual& x, const PprResidual& y) {
    if (x.n != y.n || !same_pauli(x.pauli, y.pauli) || x.rots.size() != y.rots.size()) return false;
    for (size_t i = 0; i < x.rots.size(); ++i)
        if (!same_pauli(x.rots[i].first, y.rots[i].first) || x.rots[i].second != y.rots[i].second) return false;
    return true;
}
bool same_atom(const PropAtom& x, const PropAtom& y) {
    if (x.present != y.present || x.rejected != y.rejected || x.reject_gate_index != y.reject_gate_index ||
        !same_dpc(x.c_prop(), y.c_prop()) || (x.general == nullptr) != (y.general == nullptr) ||
        (x.ppr == nullptr) != (y.ppr == nullptr))
        return false;
    if (x.general && !same_tableau(*x.general, *y.general)) return false;
    if (x.ppr && !same_ppr(*x.ppr, *y.ppr)) return false;
    return true;
}
bool same_table(const PropagationTable& x, const PropagationTable& y) {
    if (x.n != y.n || x.all_in_class != y.all_in_class || x.has_general != y.has_general ||
        x.locations.size() != y.locations.size() || x.rejects.size() != y.rejects.size())
        return false;
    for (size_t i = 0; i < x.rejects.size(); ++i) {
        const RejectInfo &a = x.rejects[i], &b = y.rejects[i];
        if (a.location_index != b.location_index || a.qubit != b.qubit || a.basis != b.basis ||
            a.reject_gate_index != b.reject_gate_index)
            return false;
    }
    for (size_t i = 0; i < x.locations.size(); ++i) {
        const LocationEntry &a = x.locations[i], &b = y.locations[i];
        if (a.stream_index != b.stream_index || a.qubits != b.qubits || a.x_atom.size() != b.x_atom.size() ||
            a.z_atom.size() != b.z_atom.size())
            return false;
        for (size_t j = 0; j < a.x_atom.size(); ++j) if (!same_atom(a.x_atom[j], b.x_atom[j])) return false;
        for (size_t j = 0; j < a.z_atom.size(); ++j) if (!same_atom(a.z_atom[j], b.z_atom[j])) return false;
    }
    return true;
}

// czrows: a sparse-CZ table (sparse_cz) against the dense one built the same way: every atom is in the sparse form
// with its dense rows freed, and rehydrates (dense_c_prop) to the dense atom bit for bit; the rest of the table is
// unchanged, and reading c_prop() of a sparse atom throws (F2). Returns the number of mismatching atoms (a table-level mismatch counts as one).
long sparse_mismatches(const PropagationTable& sp, const PropagationTable& dn) {
    if (sp.n != dn.n || sp.all_in_class != dn.all_in_class || sp.has_general != dn.has_general ||
        sp.locations.size() != dn.locations.size() || sp.rejects.size() != dn.rejects.size())
        return 1;
    long bad = 0;
    for (size_t i = 0; i < sp.locations.size(); ++i) {
        const LocationEntry &a = sp.locations[i], &b = dn.locations[i];
        if (a.stream_index != b.stream_index || a.qubits != b.qubits || a.x_atom.size() != b.x_atom.size() ||
            a.z_atom.size() != b.z_atom.size()) { ++bad; continue; }
        for (int bx = 0; bx < 2; ++bx)
            for (size_t j = 0; j < (bx ? a.x_atom : a.z_atom).size(); ++j) {
                const PropAtom& s = (bx ? a.x_atom : a.z_atom)[j];
                const PropAtom& d = (bx ? b.x_atom : b.z_atom)[j];
                PropAtom r = d;                       // the dense atom with the sparse one's rehydrated c_prop
                r.c_prop_mut() = dense_c_prop(s);
                if (!s.cz_sparse() || !s.cz_rows_freed() || s.cz_words() != d.c_prop().B.words() || !same_atom(r, d) || s.present != d.present ||
                    (s.ppr == nullptr) != (d.ppr == nullptr) || (s.general == nullptr) != (d.general == nullptr))
                    ++bad;
                bool loud = false;                    // F2: a direct read of a sparse atom's operator throws
                try { (void)s.c_prop(); } catch (const std::logic_error&) { loud = true; }
                if (!loud) ++bad;
            }
    }
    return bad;
}

// The PPR retry's walk as 3.2.0 ran it: every Gate instruction after `from`, in order.
PprResidual ppr_unskipped(const Circuit& circ, int from, bool seedX, int q, bool& ok, int& reject_k) {
    PprResidual R = seedX ? ppr_seed_x(circ.n, q) : ppr_seed_z(circ.n, q);
    ok = true;
    reject_k = -1;
    for (int k = from + 1; k < (int)circ.stream.size() && ok; ++k) {
        const Instr& ins = circ.stream[(size_t)k];
        if (ins.kind != Instr::Kind::Gate) continue;
        ok = R.propagate(ins.gate, ins.targets);
        if (!ok) reject_k = k;
    }
    return R;
}

}  // namespace

// czrows F1 teeth: sparsify_cz's check against corrupted dense layers. For up to `limit` atoms of the dense table with
// a materialized CZ layer: the clean copy must sparsify (and rehydrate exactly); a lower-triangle-only bit, a diagonal
// bit and (when n % 64 != 0) a padding bit must each make sparsify_cz throw before freeing the rows.
py::dict sparsify_teeth(const Circuit& c, int limit) {
    const PropagationTable T = build_propagation_table(c, true, AtomSet::AnyPauli);
    long atoms = 0, cases = 0, thrown = 0, clean_ok = 0;
    for (const LocationEntry& L : T.locations)
        for (const auto* v : {&L.x_atom, &L.z_atom})
            for (const PropAtom& A : *v) {
                if (atoms >= limit || A.c_prop().B.words() == 0) continue;
                ++atoms;
                const int k = A.c_prop().B.dim(), w = A.c_prop().B.words();
                PropAtom clean = A;
                sparsify_cz(clean);
                if (same_dpc(dense_c_prop(clean), A.c_prop())) ++clean_ok;
                auto expect_throw = [&](int row, int col) {
                    PropAtom bad = A;
                    bad.c_prop_mut().B.data_[(size_t)row * (size_t)w + (size_t)(col >> 6)] ^= 1ULL << (col & 63);
                    ++cases;
                    try { sparsify_cz(bad); } catch (const std::logic_error&) { ++thrown; }
                };
                if (k >= 2) expect_throw(k - 1, 0);           // lower-triangle only (row k-1, col 0)
                expect_throw(0, 0);                            // diagonal
                if (k % 64) expect_throw(0, w * 64 - 1);       // padding
            }
    // threads L1: cz_pairs() on a DENSE atom throws std::logic_error (never a silent empty pair list); the sparse
    // copy of the same atom reads its pairs.
    long dense_reads = 0, dense_thrown = 0;
    for (const LocationEntry& L : T.locations)
        for (const auto* v : {&L.x_atom, &L.z_atom})
            for (const PropAtom& A : *v) {
                if (dense_reads >= limit) break;
                ++dense_reads;
                try { (void)A.cz_pairs(); } catch (const std::logic_error&) { ++dense_thrown; }
            }
    return py::dict(py::arg("atoms") = atoms, py::arg("cases") = cases, py::arg("thrown") = thrown,
                    py::arg("clean_ok") = clean_ok, py::arg("dense_reads") = dense_reads,
                    py::arg("dense_thrown") = dense_thrown);
}

void register_compile(py::module_& m) {
    m.def("_sparsify_cz_teeth",
          [](const std::string& text, int limit) {
              ParsedStim ps = parse_stim_circuit(text);
              if (!ps.ok()) throw std::invalid_argument("_sparsify_cz_teeth: parse failed");
              return sparsify_teeth(ps.circuit, limit);
          },
          py::arg("text"), py::arg("limit") = 50,
          "czrows TEST API (F1): sparsify_cz's dense-row check against corrupted CZ layers.");
    m.def("_push_table_oracle_mismatches", []() { return push_table_oracle_mismatches(); },
          "Compile stage TEST API (F-G): frame-push table entries differing from the brute-force search.");
    m.def("_propagation_walk_oracle",
          [](const std::string& text, bool normalized) {
              ParsedStim ps = parse_stim_circuit(text);
              if (!ps.ok()) throw std::invalid_argument("_propagation_walk_oracle: parse failed");
              Circuit c = ps.circuit;
              if (normalized) {                     // the twirl sampler preset (twirl_front.hpp)
                  NormalizePolicy pol;
                  pol.coherentize = true;
                  pol.coherentize_all = true;
                  pol.defer = true;
                  pol.want_map = true;
                  pol.feedback = NormalizePolicy::Feedback::Reject;
                  pol.hoist_feedback_on_class_exit = true;
                  pol.zero_noise_transparent = true;
                  NormalizeResult nr = normalize(ps.circuit, pol);
                  if (nr.feedback_rejected) throw std::invalid_argument("_propagation_walk_oracle: feedback rejected");
                  c = std::move(nr.normalized);
              }
              long atoms = 0, atom_bad = 0, ppr_atoms = 0, ppr_bad = 0, rejected = 0;
              // F-F: every atom (AnyPauli builds them all) vs propagate_atom; PPR retries vs the unskipped loop.
              const PropagationTable T = build_propagation_table(c, false, AtomSet::AnyPauli);
              const PropagationTable P = build_propagation_table(c, true, AtomSet::AnyPauli);
              const PropagationTable Pn = build_propagation_table(c, true, AtomSet::AnyPauli, /*build_general=*/false);
              for (size_t li = 0; li < T.locations.size(); ++li) {
                  const LocationEntry& L = T.locations[li];
                  for (size_t qi = 0; qi < L.qubits.size(); ++qi)
                      for (int bx = 1; bx >= 0; --bx) {
                          const int q = L.qubits[qi];
                          const PropAtom& t = bx ? L.x_atom[qi] : L.z_atom[qi];
                          const PropResult o = propagate_atom(
                              c, L.stream_index, bx ? DiagPauliClifford::X(c.n, q) : DiagPauliClifford::Z(c.n, q));
                          ++atoms;
                          if (t.rejected != o.rejected || t.reject_gate_index != o.reject_gate_index ||
                              !same_dpc(t.c_prop(), o.c_prop()))
                              ++atom_bad;
                          if (!o.rejected) continue;
                          ++rejected;
                          bool ok = false;
                          int rk = -1;
                          const PprResidual R = ppr_unskipped(c, L.stream_index, bx == 1, q, ok, rk);
                          const PropAtom& pa = bx ? P.locations[li].x_atom[qi] : P.locations[li].z_atom[qi];
                          const PropAtom& pn = bx ? Pn.locations[li].x_atom[qi] : Pn.locations[li].z_atom[qi];
                          ++ppr_atoms;
                          bool good = pa.rejected == !ok && pa.reject_gate_index == (ok ? -1 : rk) &&
                                      same_dpc(pa.c_prop(), o.c_prop());
                          if (good && ok)
                              good = pa.ppr && same_ppr(*pa.ppr, R) && pa.general &&
                                     same_tableau(*pa.general, R.to_error_tableau());
                          if (good && !ok) good = !pa.ppr && !pa.general;
                          // F-A: the general-free table is the default one minus the tableaux
                          good = good && pn.general == nullptr && pn.rejected == pa.rejected &&
                                 pn.reject_gate_index == pa.reject_gate_index && (pn.ppr == nullptr) == (pa.ppr == nullptr) &&
                                 (!pn.ppr || same_ppr(*pn.ppr, *pa.ppr));
                          if (!good) ++ppr_bad;
                      }
              }
              // F-D: the early-stopping class check vs the full table (ChannelSupport, no PPR retry).
              const PropagationTable F = build_propagation_table(c);
              const std::optional<PropagationTable> E = build_propagation_table_if_in_class(c);
              const bool class_ok = E.has_value() == F.all_in_class && (!E || same_table(*E, F));
              // czrows: the sparse-CZ builds (the twirl sampler's table; normalize's class_table) vs the dense ones.
              const PropagationTable Ps = build_propagation_table(c, true, AtomSet::AnyPauli, /*build_general=*/true,
                                                                  /*sparse_cz=*/true);
              long sparse_bad = sparse_mismatches(Ps, P), sparse_pairs = 0;
              for (const LocationEntry& L : Ps.locations)
                  for (const auto* atoms : {&L.x_atom, &L.z_atom})
                      for (const PropAtom& A : *atoms) sparse_pairs += (long)A.cz_pairs().size();
              sparse_bad += sparse_mismatches(
                  build_propagation_table(c, true, AtomSet::ChannelSupport, /*build_general=*/false, /*sparse_cz=*/true),
                  build_propagation_table(c, true, AtomSet::ChannelSupport, /*build_general=*/false));
              const std::optional<PropagationTable> Es = build_propagation_table_if_in_class(c, /*sparse_cz=*/true);
              if (Es.has_value() != E.has_value()) ++sparse_bad;
              else if (E) sparse_bad += sparse_mismatches(*Es, *E);
              return py::dict(py::arg("n") = c.n, py::arg("atoms") = atoms, py::arg("atom_mismatches") = atom_bad,
                              py::arg("rejected_atoms") = rejected, py::arg("ppr_atoms") = ppr_atoms,
                              py::arg("ppr_mismatches") = ppr_bad, py::arg("in_class") = F.all_in_class,
                              py::arg("class_check_agrees") = class_ok, py::arg("sparse_mismatches") = sparse_bad,
                              py::arg("sparse_pairs") = sparse_pairs,
                              py::arg("has_general") = P.has_general && Pn.has_general == P.has_general);
          },
          py::arg("text"), py::arg("normalized") = true,
          "Compile stage TEST API (F-F/F-D/F-A): the propagation table's walk vs the unskipped walks.");
}
