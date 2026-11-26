// EXP-fix test API (not a public interface): the per-instance DEM law census of export_dem — every channel
// instance that reached the solve, with its route (pauli_old / joint_fwht / erasure), status, joint rank and
// components (kind, p, kappa, rho, v) — plus the export's Born-law node count (born_sums). The G-DEM-* gates and the E0 census use it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qeccore/dem_export.hpp"

namespace py = pybind11;

void register_dem_census(py::module_& m) {
    m.def("dem_law_census",
          [](const std::string& text, bool include_expectations, double approximate_disjoint_errors,
             int clifford_cutoff) {
              qeccore::DemLawCensus c;
              qeccore::DemExportResult r = qeccore::export_dem(
                  text, include_expectations, false, false, false, nullptr, nullptr,
                  approximate_disjoint_errors, clifford_cutoff, &c);
              py::list inst;
              for (const auto& in : c.inst) {
                  py::list comps;
                  for (const auto& cm : in.comps)
                      comps.append(py::dict(py::arg("kind") = std::string(1, cm.kind), py::arg("p") = cm.p,
                                            py::arg("kappa") = cm.kappa, py::arg("rho") = cm.rho, py::arg("v") = cm.v,
                                            py::arg("off") = cm.off, py::arg("W") = cm.W, py::arg("sup") = cm.sup,
                                            py::arg("fac") = cm.fac));
                  inst.append(py::dict(py::arg("k") = in.k, py::arg("li") = in.li, py::arg("site") = in.site, py::arg("channel") = in.channel,
                                       py::arg("route") = in.route, py::arg("status") = in.status,
                                       py::arg("rank") = in.rank, py::arg("comps") = comps,
                                       py::arg("kept") = in.kept, py::arg("dropped") = in.dropped));
              }
              std::string err = r.error;
              if (!r.parse_errors.empty()) err = "parse error";
              if (r.rejected) err = "rejected";
              return py::dict(py::arg("ok") = r.ok, py::arg("error") = err, py::arg("dem") = r.dem,
                              py::arg("born_sums") = c.born_sums, py::arg("coset_laws") = c.coset_laws,
                              py::arg("coset_work") = c.coset_work, py::arg("instances") = inst,
                              py::arg("clifford_faults") = r.clifford_faults,
                              py::arg("clifford_dropped_mass") = r.clifford_dropped_mass,
                              py::arg("fb_ops") = c.fb_ops, py::arg("fb_q_corrections") = c.fb_q_corrections,
                              py::arg("fb_q_pairs") = c.fb_q_pairs, py::arg("fb_touched") = c.fb_touched);
          },
          py::arg("text"), py::arg("include_expectations") = true, py::arg("approximate_disjoint_errors") = 0.0,
          py::arg("clifford_cutoff") = qeccore::kDemCliffordCutoffDefault,
          "EXP-fix TEST API: export_dem with the per-instance DEM law census.");
}
