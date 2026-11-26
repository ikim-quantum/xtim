// xdem Task 2 test API (not a public interface): the xdem reader, the canonical emitter, the composite and the law
// (qeccore/xdem_file.hpp) for tests/test_xdem_file.py. Registered in record_law_oracle_py.cpp (PYBIND11_MODULE
// _xtim_testapi). Reader refusals (qeccore::xdem::XdemFileError) surface as xtim.XtimXdemError (the exception
// translator _xtim registers is process-wide).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "qeccore/xdem_file.hpp"

namespace py = pybind11;
namespace xd = qeccore::xdem;

namespace {

py::list bit_list(const std::vector<uint64_t>& w) {
    py::list out;
    for (size_t i = 0; i < w.size(); ++i)
        for (uint64_t x = w[i]; x; x &= x - 1) out.append(i * 64 + (size_t)__builtin_ctzll(x));
    return out;
}

py::list rots(const std::vector<uint64_t>& off, const std::vector<xd::Rot>& v, size_t i) {
    py::list out;
    for (uint64_t t = off[i]; t < off[i + 1]; ++t) out.append(py::make_tuple(v[t].axis, (int)v[t].e));
    return out;
}

}  // namespace

void register_xdem_file(py::module_& m) {
    m.def("xdem_parse", [](const std::string& text) {
        const xd::File F = xd::parse(text);
        py::dict d;
        d["extended"] = F.extended;
        d["D"] = F.D;
        d["O"] = F.O;
        d["r"] = F.r;
        d["m"] = F.m;
        d["tc"] = F.tc;
        d["A"] = F.axes.size();
        d["L"] = F.num_lines();
        py::list axes;
        for (const xd::Axis& a : F.axes) axes.append(py::make_tuple(a.b, bit_list(a.img.w), bit_list(a.k), a.hidden));
        d["axes"] = axes;
        py::list lines;
        for (size_t i = 0; i < F.num_lines(); ++i) {
            py::list tg, tau;
            for (uint64_t t = F.flip_off[i]; t < F.flip_off[i + 1]; ++t) tg.append(F.flip_cols[t]);
            for (uint64_t t = F.tau_off[i]; t < F.tau_off[i + 1]; ++t) tau.append(F.tau_ax[t]);
            lines.append(py::make_tuple(F.p[i], tg, tau, rots(F.rot_off, F.rot, i), rots(F.late_off, F.late, i)));
        }
        d["lines"] = lines;
        return d;
    }, py::arg("text"),
       "TEST API: parse an xdem (or conventional DEM) text -> {extended, D, O, r, m, tc, A, L, axes: [(b, img "
       "columns, gauge qubits, hidden-syndrome qubits)], lines: [(p, target columns, tau axes, rot [(a, e)], late "
       "[(a, e)])]}; refusals raise XtimXdemError.");

    m.def("xdem_emit", [](const std::string& text) { return xd::emit(xd::parse(text)); }, py::arg("text"),
          "TEST API: the canonical text of the parsed file (emit(parse(text))).");

    m.def("xdem_laws", [](const std::string& text, const std::vector<std::vector<int>>& configs) {
        const xd::File F = xd::parse(text);
        py::list out;
        for (const auto& ids : configs) {
            const xd::Law L = xd::law(F, xd::compose(F, ids));
            py::list coins;
            for (const auto& c : L.coins) coins.append(xd::to_hex(c));
            out.append(py::make_tuple(xd::to_hex(L.offset), coins));
        }
        return out;
    }, py::arg("text"), py::arg("configs"),
       "TEST API: per configuration (line ids, any order, repeats cancel): (offset hex, [coin hex]) of its law "
       "(offset reduced modulo the coins; coins RREF, descending pivot; detector d = bit d, observable o = bit D+o).");

    m.def("xdem_compose", [](const std::string& text, const std::vector<int>& ids) {
        const xd::File F = xd::parse(text);
        const xd::Composed c = xd::compose(F, ids);
        py::list eta;
        for (const auto& [a, e] : c.eta) eta.append(py::make_tuple(a, e));
        return py::make_tuple(xd::to_hex(c.flip), bit_list(c.tau), eta);
    }, py::arg("text"), py::arg("ids"),
       "TEST API: the composite (SPEC §1.2) of the lines ids -> (flip hex, tau axes, eta [(a, e mod 4)]).");

    m.def("xdem_facts", [](const std::string& text) {
        const xd::File F = xd::parse(text);
        py::dict d;
        d["late_axes"] = xd::late_axes(F);
        const auto r2b = xd::r2b_available(F);
        d["r2b_available"] = r2b.first;
        d["r2b_reason"] = r2b.second;
        d["invisible_residues"] = xd::invisible_residues(F);
        return d;
    }, py::arg("text"),
       "TEST API: the file-level facts {late_axes, r2b_available, r2b_reason, invisible_residues} (SPEC §4.4, §4.6).");
}
