// EXP-fix test API (not a public interface): the per-instance DEM law census of export_dem — every channel
// instance that reached the solve, with its route (pauli_old / coset_closed / joint_fwht), status, joint rank and
// components (kind, p, v0, wdim, v) — plus the export's Born-sum count. The G-DEM-* gates and the E0 census use it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qeccore/dem_export.hpp"

namespace py = pybind11;

void register_dem_census(py::module_& m) {
    m.def("dem_law_census",
          [](const std::string& text, bool include_expectations, double approximate_disjoint_errors) {
              qeccore::DemLawCensus c;
              qeccore::DemExportResult r = qeccore::export_dem(
                  text, include_expectations, false, false, false, nullptr, nullptr,
                  approximate_disjoint_errors, qeccore::kPeMaxWorkDefault, &c);
              py::list inst;
              for (const auto& in : c.inst) {
                  py::list comps;
                  for (const auto& cm : in.comps)
                      comps.append(py::dict(py::arg("kind") = std::string(1, cm.kind), py::arg("p") = cm.p,
                                            py::arg("v0") = cm.v0, py::arg("wdim") = cm.wdim, py::arg("v") = cm.v));
                  inst.append(py::dict(py::arg("k") = in.k, py::arg("channel") = in.channel,
                                       py::arg("route") = in.route, py::arg("status") = in.status,
                                       py::arg("rank") = in.rank, py::arg("comps") = comps));
              }
              std::string err = r.error;
              if (!r.parse_errors.empty()) err = "parse error";
              if (r.rejected) err = "rejected";
              return py::dict(py::arg("ok") = r.ok, py::arg("error") = err, py::arg("dem") = r.dem,
                              py::arg("born_sums") = c.born_sums, py::arg("instances") = inst);
          },
          py::arg("text"), py::arg("include_expectations") = true, py::arg("approximate_disjoint_errors") = 0.0,
          "EXP-fix TEST API: export_dem with the per-instance DEM law census.");
}
