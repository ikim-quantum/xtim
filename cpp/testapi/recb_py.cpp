// J-a-b (N6, switched OFF in production): the TEST API of record-space B (qeccore/record_space_b.hpp), on a FAM
// sampler of xtim._xtim_testapi.compile_family_sampler (its TwirlRecordSampler via FamilySampler._engine()).
//   recb_plan_structure(fam, key) — the plan's record-space-B table as plain fields (the G-9 convention of
//       tests/test_ja_dual_run.py), plus B's op counts (J-b2 J7b: the σ dual basis "d" side is deleted);
//   recb_replay(fam, keys, passes, mode) — the B-banking replay: every key built `passes` times by one builder; mode
//       "recb" (build_record_table_recB) or "plan_structure" (the ledger's t_build call itself). Returns a digest of
//       the tables. The measured function is the noinline recb_build_one_ (callgrind --toggle-collect).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>
#include <vector>

#include "qeccore/record_space_b.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/twirl_sampler.hpp"

namespace py = pybind11;
using namespace qeccore;

namespace {

TwirlRecordSampler& engine_of(const py::object& fam) {
    py::capsule c = fam.attr("_engine")();
    if (std::strcmp(c.name(), "xtim.TwirlRecordSampler") != 0)
        throw std::invalid_argument("recb: expected a FamilySampler (compile_family_sampler)");
    return *static_cast<TwirlRecordSampler*>(c.get_pointer());
}

// The sampler's record columns (storage order) from its public accessors.
std::vector<RecColumn> columns_of(const TwirlRecordSampler& s) {
    const std::vector<Pauli> ops = s.record_operators();
    const std::vector<int> ord = s.record_column_order();
    const std::vector<std::vector<uint64_t>> masks = s.record_channel_masks();
    const std::vector<uint8_t> refs = s.record_channel_refs();
    const int nch = s.num_channels();
    std::vector<RecColumn> cols(ops.size());
    for (size_t c = 0; c < ops.size(); ++c) {
        cols[c].M = ops[c];
        cols[c].order = ord[c];
        cols[c].channel = (int)c < nch;
        if ((int)c < nch) {
            cols[c].mask = masks[c];
            cols[c].ref = refs[c];
        }
    }
    return cols;
}

py::dict table_dict(const RecTable& T) {
    py::dict res;
    res["rec_built"] = T.built;
    res["m"] = T.m;
    res["rho"] = T.rho;
    res["kappa_rec"] = T.kappa;
    res["pivot_rec"] = T.pivot_rec;
    res["born_rec"] = T.born_rec;
    py::array_t<uint8_t> A({(py::ssize_t)T.m, (py::ssize_t)T.rho});
    py::array_t<uint8_t> B({(py::ssize_t)T.m, (py::ssize_t)T.kappa});
    py::array_t<uint8_t> be((py::ssize_t)T.m);
    for (int c = 0; c < T.m; ++c) {
        for (int k = 0; k < T.rho; ++k)
            A.mutable_data()[(size_t)c * T.rho + k] = (uint8_t)((T.coin_row[(size_t)k][(size_t)(c >> 6)] >> (c & 63)) & 1);
        for (int j = 0; j < T.kappa; ++j)
            B.mutable_data()[(size_t)c * T.kappa + j] = (uint8_t)((T.born_row[(size_t)j][(size_t)(c >> 6)] >> (c & 63)) & 1);
        be.mutable_data()[c] = (uint8_t)((T.base.empty() ? 0 : (T.base[(size_t)(c >> 6)] >> (c & 63))) & 1);
    }
    res["rec_A"] = A;
    res["rec_B"] = B;
    res["rec_beta"] = be;
    const int kk = T.lam.empty() ? 0 : (int)T.lam[0].a.size();
    py::array_t<uint8_t> lam({(py::ssize_t)T.kappa, (py::ssize_t)(2 * kk)});
    py::array_t<uint8_t> lph((py::ssize_t)T.kappa);
    for (int j = 0; j < T.kappa; ++j) {
        for (int i = 0; i < kk; ++i) {
            lam.mutable_data()[(size_t)j * 2 * kk + i] = T.lam[(size_t)j].a[(size_t)i];
            lam.mutable_data()[(size_t)j * 2 * kk + kk + i] = T.lam[(size_t)j].b[(size_t)i];
        }
        lph.mutable_data()[j] = (uint8_t)fd_eff_phase(T.lam[(size_t)j]);
    }
    res["rec_lam"] = lam;
    res["rec_lam_phase"] = lph;
    return res;
}

// A content digest of a table's canonical fields (FNV-1a), Λ by (a, b, effective phase).
uint64_t table_digest(const RecTable& T, uint64_t h) {
    auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001B3ull; };
    mix((uint64_t)T.m); mix((uint64_t)T.rho); mix((uint64_t)T.kappa);
    for (int c : T.pivot_rec) mix((uint64_t)c);
    for (int c : T.born_rec) mix((uint64_t)c);
    for (uint64_t w : T.base) mix(w);
    for (const auto& r : T.coin_row) for (uint64_t w : r) mix(w);
    for (const auto& r : T.born_row) for (uint64_t w : r) mix(w);
    for (const FreeDir& d : T.lam) {
        for (uint8_t v : d.a) mix(v);
        for (uint8_t v : d.b) mix(v);
        mix((uint64_t)fd_eff_phase(d));
    }
    return h;
}

DiagNormalForm prefix_free(const DiagNormalForm& nf, int n) {
    DiagNormalForm nf0 = nf;
    nf0.prefix = Pauli(n);
    return nf0;
}

// The two measured builders (B-banking). Same inputs: the certified planes, the bare state, the parsed plan (identity
// prefix) and the record columns.
__attribute__((noinline)) void recb_build_one_(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                               const DiagNormalForm& nf, const std::vector<RecColumn>& cols,
                                               RecTable& T) {
    build_record_table_recB(G, bare, nf, cols, T);
}

}  // namespace

void register_branch(py::module_& m);   // branch_py.cpp (J-a-b N7)

void register_recb(py::module_& m) {
    register_branch(m);
    m.def("recb_plan_structure",
          [](const py::object& fam, py::bytes key) {
              TwirlRecordSampler& s = engine_of(fam);
              const std::string k = key.cast<std::string>();
              const TwirlRecordSampler::PlanStructure ps =
                  s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
              py::dict out;
              out["fallback"] = ps.law.fallback || !ps.rec.built;
              out["kappa"] = ps.law.kappa;
              out["nchan"] = s.num_channels();
              if (ps.law.fallback || !ps.rec.built) return out;
              const int n = s.deferred_wires();
              const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
              RecTable T;
              RecBOps ops;
              build_record_table_recB(G, s.bare_state(), ps.nf, columns_of(s), T, &ops);
              out["b"] = table_dict(T);
              py::dict o;
              o["touched"] = ops.touched;
              o["gen_visits"] = ops.gen_visits;
              o["elim_rows"] = ops.elim_rows;
              o["pre_rows"] = ops.pre_rows;
              o["pivots"] = ops.pivots;
              o["kb_read"] = ops.kb_read;
              out["ops"] = o;
              return out;
          },
          py::arg("fam"), py::arg("key"),
          "J-a-b TEST API (N6): record-space B's table of one plan key (plan_structure's format) and its op counts.");
    m.def("recb_replay",
          [](const py::object& fam, const std::vector<py::bytes>& keys, int passes, const std::string& mode) {
              TwirlRecordSampler& s = engine_of(fam);
              if (mode != "recb" && mode != "plan_structure")
                  throw std::invalid_argument("recb_replay: mode must be 'recb' or 'plan_structure'");
              std::vector<std::string> ks;
              for (const py::bytes& b : keys) ks.push_back(b.cast<std::string>());
              uint64_t h = 0xcbf29ce484222325ull;
              long built = 0;
              {
                  py::gil_scoped_release rel;           // pure C++ below; Python objects only after this scope
                  if (mode == "plan_structure") {
                      for (int p = 0; p < passes; ++p)
                          for (const std::string& k : ks) {
                              const TwirlRecordSampler::PlanStructure ps =
                                  s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
                              if (p == 0) { h = table_digest(ps.rec, h); built += ps.rec.built; }
                          }
                  } else {
                      const int n = s.deferred_wires();
                      const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
                      const std::vector<RecColumn> cols = columns_of(s);
                      std::vector<DiagNormalForm> nfs;
                      for (const std::string& k : ks)
                          nfs.push_back(prefix_free(
                              s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size()).nf, n));
                      RecTable T;
                      for (int p = 0; p < passes; ++p)
                          for (const DiagNormalForm& nf : nfs) {
                              recb_build_one_(G, s.bare_state(), nf, cols, T);
                              if (p == 0) { h = table_digest(T, h); built += T.built; }
                          }
                  }
              }
              return py::make_tuple(h, built);
          },
          py::arg("fam"), py::arg("keys"), py::arg("passes") = 1, py::arg("mode") = "recb",
          "J-a-b TEST API (N6, B-banking): build every plan key `passes` times by one builder; returns "
          "(table digest, tables built). Releases the GIL while building: the sampler must not be used from another "
          "thread meanwhile (benchmark use).");
}
