// xdem Task 3 test API (not a public interface): the writer model's canonical header (qeccore/xdem_writer.hpp) for
// tests/test_xdem_writer.py; Task 4: the writer's atom map (xdem_atom_lines, from the export's census) for
// tests/test_xdem_export.py; Task 10: the export's deferred circuit (xdem_deferred_debug) for the engine oracles of
// tests/_xdem_engine.py. Registered in record_law_oracle_py.cpp (PYBIND11_MODULE _xtim_testapi). Refusals raise
// xtim.XtimDemError.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/dem_export.hpp"
#include "qeccore/dem_preamble.hpp"
#include "qeccore/exact_scope.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_rotation_form.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/xdem_lean.hpp"
#include "qeccore/xdem_writer.hpp"
#include "xdem_lean_dense_oracle.hpp"
#include "xdem_sweep_dense_oracle.hpp"

namespace py = pybind11;
namespace xd = qeccore::xdem;

namespace {

// A stim Pauli string of a Hermitian Pauli (sign, then _XYZ per wire).
std::string stim_str(const qeccore::Pauli& P) {
    const int d = (P.phase - P.xz_overlap()) & 3;    // 0: +, 2: -
    std::string s(1, d == 2 ? '-' : '+');
    for (int q = 0; q < P.n; ++q) {
        const bool x = P.xbit(q), z = P.zbit(q);
        s.push_back(x ? (z ? 'Y' : 'X') : (z ? 'Z' : '_'));
    }
    return s;
}
std::vector<int> bits(const std::vector<uint64_t>& w) {
    std::vector<int> out;
    for (size_t i = 0; i < w.size(); ++i)
        for (uint64_t x = w[i]; x; x &= x - 1) out.push_back((int)(i * 64 + (size_t)__builtin_ctzll(x)));
    return out;
}
[[noreturn]] void raise_dem(const std::string& what) {
    py::object c = py::module_::import("xtim.errors").attr("XtimDemError");
    PyErr_SetString(c.ptr(), what.c_str());
    throw py::error_already_set();
}

// xdem Task 10: one deferred stream instruction as a stim-like line (gates by name; "NOISE" / "M" / "MX" / "MY" /
// "OBS" placeholders that tests/_xdem_engine.py replaces or drops).
std::string deferred_line(const qeccore::Instr& ins) {
    using K = qeccore::Instr::Kind;
    using G = qeccore::GateKind;
    auto join = [](const char* nm, const std::vector<int>& t) {
        std::string s = nm;
        for (int q : t) s += " " + std::to_string(q);
        return s;
    };
    switch (ins.kind) {
        case K::Gate: {
            const char* nm = nullptr;
            switch (ins.gate) {
                case G::X: nm = "X"; break;
                case G::Y: nm = "Y"; break;
                case G::Z: nm = "Z"; break;
                case G::S: nm = "S"; break;
                case G::SDG: nm = "S_DAG"; break;
                case G::CZ: nm = "CZ"; break;
                case G::CX: nm = "CX"; break;
                case G::H: nm = "H"; break;
                case G::T: nm = "T"; break;
                case G::CS: nm = "CS"; break;
                case G::CCZ: nm = "CCZ"; break;
                case G::CH: nm = "CH"; break;
            }
            return join(nm, ins.targets);
        }
        case K::Noise: return join("NOISE", ins.qubits);
        case K::Measure:
            return join(ins.basis == qeccore::PauliBasis::X ? "MX" : ins.basis == qeccore::PauliBasis::Y ? "MY" : "M",
                        ins.qubits);
        case K::Observable: return "OBS";
        default: break;
    }
    raise_dem("xdem_deferred_debug: the deferred stream holds a reset, feedback or IF block (unsupported)");
}

}  // namespace

void register_xdem_writer(py::module_& m) {
    m.def("xdem_determinism_verdicts", [](const std::string& text) {
        qeccore::ExplicitExactScope scope;                   // review I1: an explicit exact entry point
        qeccore::DemPreamble pre;
        qeccore::DemExportResult res;
        // Task 3 review M2: the lean route's preamble (default normalization only, no propagation table)
        if (!qeccore::dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false, /*class_table=*/false)) {
            if (!res.parse_errors.empty()) raise_dem("xdem_determinism_verdicts: parse error: " + res.parse_errors[0].message);
            if (res.rejected) raise_dem("xdem_determinism_verdicts: preamble rejected at gate " +
                                        std::to_string(res.reject_gate_index));
            raise_dem("xdem_determinism_verdicts: preamble refusal: " + res.error);
        }
        xd::WriterModel wm;
        try {
            wm = xd::build_writer_model(pre);
        } catch (const xd::WriterError& e) {
            raise_dem(std::string("xdem_determinism_verdicts: writer refusal: ") + e.what());
        }
        if (!wm.diag_violations.empty()) raise_dem("xdem_determinism_verdicts: diagonality violation: " + wm.diag_violations[0]);
        if (wm.header.groups.empty()) raise_dem("xdem_determinism_verdicts: no non-Clifford groups");
        const std::vector<std::vector<int>> col_recs(wm.sk->targets.begin(), wm.sk->targets.end());
        std::vector<int> lean;
        try {
            const xd::FaultColumns fc = xd::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs);
            lean = xd::nondeterministic_columns(wm, fc);
        } catch (const xd::WriterError& e) {
            raise_dem(std::string("xdem_determinism_verdicts: fault columns: ") + e.what());
        }
        const bool rot_rejected = qeccore::build_pauli_rotation_form(pre.dn.deferred).rejected;
        bool capacity = false, bare_rejected = false;
        try {
            bare_rejected = !qeccore::dem_build_bare(pre, res);
        } catch (const qeccore::BareStateCapacityError&) {
            capacity = true;
        }
        int ncols = 0;
        std::vector<int> oracle, ev_cols;
        std::vector<double> lean_ev, oracle_ev;       // nondet-message: column_expectation against the bare state
        for (int t = 0; t < (int)wm.sk->targets.size(); ++t) {
            if (wm.sk->targets[(size_t)t].empty()) continue;
            ++ncols;
            if (capacity || bare_rejected) continue;
            if (t < wm.header.D + wm.header.O) { ev_cols.push_back(t); lean_ev.push_back(xd::column_expectation(wm, t)); }
            qeccore::Pauli P(pre.dn.deferred.n);
            int ny = 0;
            for (int j : wm.sk->targets[(size_t)t]) {
                const auto& [b, q] = pre.reads[(size_t)j];
                if (b == 0) P.setx(q); else if (b == 1) { P.setx(q); P.setz(q); ++ny; } else P.setz(q);
            }
            P.phase = ny & 3;
            auto [pp, pm] = qeccore::framed_expectation(pre.bare.state, P);
            if (!(std::max(pp, pm) > 1.0 - 1e-9)) oracle.push_back(t);
            if (t < wm.header.D + wm.header.O) oracle_ev.push_back(pp - pm);
        }
        py::dict d;
        d["lean"] = lean;
        d["ev_cols"] = ev_cols; d["lean_ev"] = lean_ev; d["oracle_ev"] = oracle_ev;
        if (capacity || bare_rejected) d["oracle"] = py::none(); else d["oracle"] = oracle;
        d["ncols"] = ncols;
        d["rotation_form_rejected"] = rot_rejected;
        d["bare_rejected"] = bare_rejected;
        d["capacity"] = capacity;
        return d;
    }, py::arg("text"),
       "LEAN-EXPORT-PLAN TEST API (Task 1): the lean algebraic determinism verdicts (nondeterministic_columns) and "
       "the exact engine's (the bare state's |<P>| = 1 test) on the D + O readout columns of an admitted xdem "
       "circuit: {lean, oracle (None when the bare state is refused or over capacity), ncols (columns with a "
       "record), rotation_form_rejected, bare_rejected, capacity, ev_cols / lean_ev / oracle_ev (nondet-message: per "
       "column with a record, column_expectation and the bare state's <P>; empty when there is no oracle)}. Raises "
       "XtimDemError when not admitted.");
    m.def("dem_lazy_bare_counts", []() {
        const qeccore::DemLazyBareCounts c = qeccore::dem_lazy_bare_counts_for_test();
        return py::make_tuple(c.certified, c.lazy_builds);
    }, "LM-1 TEST API: (certified no-rotation exports, lazy bare-state builds among them), monotone. The counters are "
       "this test-API module's own engine copy only: public _xtim calls (xtim.Circuit(...).detector_error_model_text) "
       "never move them. Process-wide and relaxed: a delta taken around one call is valid only while exports run one "
       "at a time.");
    m.def("set_dem_cert_gate_off", [](bool on) { qeccore::set_dem_cert_gate_off_for_test(on); }, py::arg("on"),
          "LM-1 TEST API (process-wide, this module's engine copy only): run the bare-state determinism gate on "
          "certified no-rotation exports anyway (forces a lazy bare build).");
    m.def("xdem_nondet_sparse_dense", [](const std::string& text) {
        qeccore::ExplicitExactScope scope;
        qeccore::DemPreamble pre;
        qeccore::DemExportResult res;
        if (!qeccore::dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false, /*class_table=*/false))
            raise_dem("xdem_nondet_sparse_dense: preamble refusal");
        xd::WriterModel wm;
        try {
            wm = xd::build_writer_model(pre);
        } catch (const xd::WriterError& e) {
            raise_dem(std::string("xdem_nondet_sparse_dense: writer refusal: ") + e.what());
        }
        if (!wm.diag_violations.empty() || wm.header.groups.empty())
            raise_dem("xdem_nondet_sparse_dense: not admitted (diagonality / no groups)");
        const std::vector<std::vector<int>> col_recs(wm.sk->targets.begin(), wm.sk->targets.end());
        std::vector<int> sparse, dense;
        xd::NondetStats st;
        std::string werr;
        {
            py::gil_scoped_release rel;
            try {
                const xd::FaultColumns fc = xd::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs);
                sparse = xd::nondeterministic_columns(wm, fc, &st);
                // LM-3: the oracle reads the dense x_start (the production sweep may end on its sparse lists)
                const xd::FaultColumns fd =
                    xd::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs, xd::SweepLayout::Dense);
                dense = xd::test_oracle::nondeterministic_columns_dense(wm, fd);
            } catch (const xd::WriterError& e) {
                werr = e.what();
            }
        }
        if (!werr.empty()) raise_dem("xdem_nondet_sparse_dense: fault columns: " + werr);
        py::dict d;
        d["sparse"] = sparse; d["dense"] = dense;
        d["chunks"] = st.chunks; d["x_nnz"] = st.x_nnz; d["img_nnz"] = st.img_nnz;
        d["N"] = wm.sk->N; d["T"] = wm.header.D + wm.header.O; d["A"] = (int)wm.header.axes.size(); d["r"] = wm.header.r;
        return d;
    }, py::arg("text"),
       "LM-2 TEST API: the lean determinism verdicts by the production (sparse) nondeterministic_columns and by the "
       "dense oracle (cpp/testapi/xdem_lean_dense_oracle.hpp) on the same writer model and fault columns: {sparse, "
       "dense, chunks, x_nnz, img_nnz (the sparse form's NondetStats), N, T (= D + O), A, r}. Raises XtimDemError when "
       "the lean route would not reach the check (preamble / writer refusal, diagonality, no groups).");
    m.def("xdem_sweep_layouts", [](const std::string& text) {
        qeccore::ExplicitExactScope scope;
        qeccore::DemPreamble pre;
        qeccore::DemExportResult res;
        if (!qeccore::dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false, /*class_table=*/false))
            raise_dem("xdem_sweep_layouts: preamble refusal");
        xd::WriterModel wm;
        try {
            wm = xd::build_writer_model(pre);
        } catch (const xd::WriterError& e) {
            raise_dem(std::string("xdem_sweep_layouts: writer refusal: ") + e.what());
        }
        const std::vector<std::vector<int>> col_recs(wm.sk->targets.begin(), wm.sk->targets.end());
        const bool admitted = wm.diag_violations.empty() && !wm.header.groups.empty();
        const int T = wm.header.D + wm.header.O, N = wm.sk->N;
        py::dict d;
        std::string werr;
        {
            py::gil_scoped_release rel;
            try {
                const xd::test_oracle::FaultColumnsDense of =
                    xd::test_oracle::build_fault_columns_dense(*wm.sk, pre.dn.deferred, col_recs);
                const xd::test_oracle::SweepAtDense ot = xd::test_oracle::build_tau_columns_dense(*wm.sk, pre.dn.deferred);
                // the oracle's readout X supports as per-column wire lists (columns < ncols, wires ascending)
                std::vector<std::vector<int32_t>> ocol((size_t)of.ncols);
                long long x_nnz = 0;
                for (int w = 0; w < N; ++w)
                    for (int c = 0; c < of.ncols; ++c)
                        if ((of.x_start[(size_t)w * of.cw + ((size_t)c >> 6)] >> (c & 63)) & 1ull) {
                            ocol[(size_t)c].push_back(w);
                            ++x_nnz;
                        }
                std::vector<int> nd_oracle;
                if (admitted) {
                    xd::FaultColumns ofc;              // the oracle's columns in the production struct (dense form)
                    ofc.ncols = of.ncols; ofc.xid = of.xid; ofc.at = of.at; ofc.x_start = of.x_start; ofc.cw = of.cw;
                    nd_oracle = xd::test_oracle::nondeterministic_columns_dense(wm, ofc);
                }
                struct Out {
                    bool at = false, x = false, tau = false, xid = false, nd = false;
                    xd::SweepStats fs, ts;
                    std::vector<int> nondet;
                };
                std::vector<std::pair<std::string, Out>> outs;
                for (auto [name, L] : {std::make_pair("auto", xd::SweepLayout::Auto),
                                       std::make_pair("sparse", xd::SweepLayout::Sparse),
                                       std::make_pair("dense", xd::SweepLayout::Dense)}) {
                    Out o;
                    const xd::FaultColumns fc = xd::build_fault_columns(*wm.sk, pre.dn.deferred, col_recs, L, &o.fs);
                    const xd::TauColumns tc = xd::build_tau_columns(*wm.sk, pre.dn.deferred, L, &o.ts);
                    o.at = fc.at == of.at && fc.ncols == of.ncols;
                    o.xid = fc.xid == of.xid;
                    o.tau = tc.at == ot;
                    if (fc.x_sparse()) {
                        bool same = fc.x_start.empty() && fc.cw == 0 && fc.xs_off.size() == (size_t)of.ncols + 1 &&
                                    fc.xs_off.back() == fc.xs_wire.size();
                        for (int c = 0; same && c < of.ncols; ++c)
                            same = std::vector<int32_t>(fc.xs_wire.begin() + (std::ptrdiff_t)fc.xs_off[(size_t)c],
                                                        fc.xs_wire.begin() + (std::ptrdiff_t)fc.xs_off[(size_t)c + 1]) ==
                                   ocol[(size_t)c];
                        o.x = same;
                    } else {
                        o.x = fc.x_start == of.x_start && fc.cw == of.cw;
                    }
                    if (admitted) {
                        o.nondet = xd::nondeterministic_columns(wm, fc);
                        o.nd = o.nondet == nd_oracle;
                    }
                    outs.emplace_back(name, std::move(o));
                }
                py::gil_scoped_acquire acq;
                d["N"] = N; d["T"] = T; d["A"] = (int)wm.header.axes.size(); d["ncols"] = of.ncols;
                d["C"] = of.ncols + (int)of.xid.size(); d["x_nnz"] = x_nnz; d["admitted"] = admitted;
                d["nondet"] = nd_oracle;
                for (const auto& [name, o] : outs) {
                    d[(name + "_at").c_str()] = o.at; d[(name + "_x").c_str()] = o.x; d[(name + "_tau").c_str()] = o.tau;
                    d[(name + "_xid").c_str()] = o.xid;
                    d[(name + "_nondet").c_str()] = admitted ? py::object(py::bool_(o.nd)) : py::object(py::none());
                    d[(name + "_final").c_str()] = o.fs.final_layout == 1 ? "sparse" : "dense";
                    d[(name + "_dense_from").c_str()] = o.fs.dense_from;
                    d[(name + "_peak_nnz").c_str()] = o.fs.peak_nnz;
                    d[(name + "_peak_bytes").c_str()] = o.fs.peak_bytes;          // fix round 1
                    d[(name + "_peak_bit_rows").c_str()] = o.fs.peak_bit_rows;
                    d[(name + "_bit_rows_made").c_str()] = o.fs.bit_rows_made;
                    d[(name + "_work").c_str()] = o.fs.work;
                    d[(name + "_tau_bit_rows_made").c_str()] = o.ts.bit_rows_made;
                    d[(name + "_tau_final").c_str()] = o.ts.final_layout == 1 ? "sparse" : "dense";
                    d[(name + "_tau_dense_from").c_str()] = o.ts.dense_from;
                }
            } catch (const xd::WriterError& e) {
                werr = e.what();
            }
        }
        if (!werr.empty()) raise_dem("xdem_sweep_layouts: sweep: " + werr);
        return d;
    }, py::arg("text"),
       "LM-3 TEST API: the production backward sweep under each storage layout (auto / sparse / dense) against the "
       "dense oracle (cpp/testapi/xdem_sweep_dense_oracle.hpp, verbatim 2e89d05) on one writer model: per layout L, "
       "L_at / L_xid (the fault columns' noise slots, x ids), L_x (x_start: the dense words, or the sparse per-column "
       "wire lists against the oracle's bits), L_tau (the tau columns), L_nondet (the lean determinism verdicts on "
       "those columns == the dense oracle's on the oracle's columns; None when not admitted), and the storage history "
       "L_final / L_dense_from / L_peak_nnz / L_peak_bytes / L_peak_bit_rows / L_bit_rows_made / L_work (fault sweep), "
       "L_tau_final / L_tau_dense_from / L_tau_bit_rows_made; plus N, T, A, ncols, C "
       "(columns incl. locations), x_nnz (the readout X supports' total), admitted, nondet (the oracle verdicts). "
       "Raises XtimDemError on a preamble / writer refusal.");
    m.def("xdem_class_probe", [](const std::string& text) {
        qeccore::ExplicitExactScope scope;                // review I1
        qeccore::ParsedStim p = qeccore::parse_stim_circuit(text);
        if (!p.ok()) raise_dem("parse error: " + p.errors[0].message);
        int site = 0;
        for (qeccore::Instr& ins : p.circuit.stream)
            if (ins.kind == qeccore::Instr::Kind::Noise) {
                ins.origin = qeccore::Instr::Origin::UserSite; ins.origin_index = site++;
            }
        qeccore::NormalizePolicy pol;                     // dem_preamble's preset + dem_normalize's transparency
        pol.feedback = qeccore::NormalizePolicy::Feedback::KeepCoherent;
        pol.defer = true;
        pol.zero_noise_transparent = true;
        qeccore::NormalizeResult nr = qeccore::normalize(p.circuit, pol);
        qeccore::NormalizePolicy alt = pol;
        alt.diagonal_copy = true;
        qeccore::NormalizeResult nr2 = qeccore::normalize(p.circuit, alt);
        py::dict d;
        d["tdef"] = qeccore::build_propagation_table(nr.normalized).all_in_class;
        d["tdc"] = qeccore::build_propagation_table(nr2.normalized).all_in_class;
        d["t3"] = qeccore::build_propagation_table(nr.normalized, true, qeccore::AtomSet::ChannelSupport, false).all_in_class;
        d["feedback"] = qeccore::circuit_has_feedback(nr.coherent);
        // the writer as the export's lean route sees it: dem_preamble without tables (feedback fold, same-wire check)
        qeccore::DemPreamble pre;
        qeccore::DemExportResult pres;
        const bool pre_ok = qeccore::dem_preamble(text, false, nullptr, pre, pres, /*build_bare=*/false,
                                                  /*class_table=*/false);
        std::string w = pre_ok ? "ok" : "preamble_refused";
        bool h = false;
        if (pre_ok) try {
            const xd::WriterModel wm = xd::build_writer_model(pre);
            if (!wm.diag_violations.empty()) w = "diag";
            else if (wm.header.groups.empty()) w = "no_groups";
            h = wm.h_meets_location;
        } catch (const xd::WriterError&) {
            w = "writer_error";
        }
        d["writer"] = w;
        d["h_meets_location"] = h;
        return d;
    }, py::arg("text"), "LEAN-EXPORT-PLAN Task 2 TEST API: the three class tables of dem_normalize, feedback, and "
                        "the writer model on the default normalization.");
    m.def("xdem_export_route", [](const std::string& text, const std::string& route, double approximate_disjoint_errors,
                                  bool require_canonical) {
        if (route != "lean" && route != "oracle") throw py::value_error("route must be \"lean\" or \"oracle\"");
        qeccore::DemExportResult r;
        bool lean_used = false;
        std::string exc, msg, lean_exc;
        {
            py::gil_scoped_release rel;
            qeccore::ExplicitExactScope scope;               // review I1: an explicit exact entry point
            try {
                r = qeccore::export_dem_route(text, true, false, false, false, nullptr, nullptr,
                                              approximate_disjoint_errors, qeccore::kDemCliffordCutoffDefault, nullptr,
                                              false, qeccore::DemFormat::Xdem, require_canonical,
                                              route == "lean" ? qeccore::XdemRoute::Lean : qeccore::XdemRoute::Oracle,
                                              &lean_used, &lean_exc);
            } catch (const qeccore::BareStateCapacityError& e) {
                exc = "BareStateCapacityError"; msg = e.what();
            } catch (const std::exception& e) {
                exc = "std::exception"; msg = e.what();
            }
        }
        py::dict d;
        d["ok"] = r.ok; d["dem"] = r.dem; d["error"] = r.error; d["rejected"] = r.rejected;
        d["reject_gate_index"] = r.reject_gate_index;
        py::list pe, pel;
        for (const auto& e : r.parse_errors) { pe.append(e.message); pel.append(e.line); }
        d["parse_errors"] = pe;
        d["parse_error_lines"] = pel;                     // final review M2: every result field, for the sweep
        d["dropped_observables"] = r.dropped_observables;
        d["reject_detectors"] = r.reject_detectors;
        d["nondeterministic_targets"] = r.nondeterministic_targets;
        d["postselect_faults"] = r.postselect_faults;
        d["dropped_expectation_columns"] = r.dropped_expectation_columns;
        d["clifford_faults"] = r.clifford_faults;
        d["clifford_dropped_mass"] = r.clifford_dropped_mass;
        d["exception"] = exc; d["exception_message"] = msg; d["lean_used"] = lean_used;
        d["lean_exception"] = lean_exc;
        return d;
    }, py::arg("text"), py::arg("route"), py::arg("approximate_disjoint_errors") = 0.0,
       py::arg("require_canonical") = false,
       "LEAN-EXPORT-PLAN TEST API: format=\"xdem\" with the public defaults (include_expectations=True) through the "
       "given route (\"lean\" or \"oracle\"); a dict of the result fields (exceptions caught and named), plus lean_used and lean_exception (the what() of an "
       "exception the lean attempt threw and the route swallowed, OD2; \"\" when none).");
    m.def("xdem_atom_lines", [](const std::string& text, double approximate_disjoint_errors) {
        qeccore::DemLawCensus c;
        qeccore::DemExportResult r;
        {
            py::gil_scoped_release rel;
            r = qeccore::export_dem(text, true, false, false, false, nullptr, nullptr, approximate_disjoint_errors,
                                    qeccore::kDemCliffordCutoffDefault, &c, false, qeccore::DemFormat::Xdem);
        }
        if (!r.parse_errors.empty()) raise_dem("parse error: " + r.parse_errors[0].message);
        if (r.rejected) raise_dem("rejected (propagation class) at gate " + std::to_string(r.reject_gate_index));
        if (!r.ok) raise_dem(r.error);
        py::list out;
        for (const auto& a : c.atoms) {
            if (a.site == -2) raise_dem("xdem_atom_lines: a deferred noise instruction has no provenance");
            py::dict d;
            if (a.record >= 0) {
                d["site"] = -1;
                d["addr"] = py::make_tuple("record", a.record);
            } else {
                d["site"] = a.site;
                d["addr"] = py::make_tuple(a.target, a.word);
            }
            d["line"] = a.line;
            d["p"] = a.p;
            d["step"] = a.step;
            d["x"] = a.x;
            d["flips"] = a.flips;
            out.append(d);
        }
        return out;
    }, py::arg("text"), py::arg("approximate_disjoint_errors") = 0.0,
       "xdem Task 4 TEST API: the writer's atom map of `text` (format=\"xdem\", the public export's defaults): one "
       "dict(site, addr, line, p, step, x, flips) per channel alternative with p > 0 and per readout flip. site / addr are "
       "the exact engine's forced-fault address: (site, (target, word)), or (-1, (\"record\", j)) for a readout flip; "
       "line = the index of the xdem error line carrying the alternative's own line key (-1: none, or the output is "
       "the conventional DEM); step = the user step of its noise site (a readout flip: of its Measure); x = its "
       "x-patterns [(group, pattern)] by group (a readout flip: []); flips = its readout flips (the converter key's "
       "flip ids; a readout flip: its targets). A circuit with no non-Clifford gate gives []. "
       "Refusals raise xtim.XtimDemError.");

    m.def("xdem_header_debug", [](const std::string& text) {
        qeccore::DemPreamble pre;
        qeccore::DemExportResult res;
        if (!qeccore::dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false)) {
            std::string e = res.error;
            if (!res.parse_errors.empty()) e = "parse error: " + res.parse_errors[0].message;
            else if (res.rejected) e = "rejected (propagation class) at gate " + std::to_string(res.reject_gate_index);
            raise_dem(e);
        }
        xd::WriterModel W;
        xd::Canon C;
        std::vector<std::vector<int>> K, KI;           // axis sets
        std::vector<int> betaK;
        try {
            W = xd::build_writer_model(pre);
            C = xd::canonical_coordinates(W);
            const auto& sk = *W.sk;
            const size_t A = W.header.axes.size();
            std::vector<int> piv, gen;                 // piv[t]: the t-th pivot axis; gen[t]: the axis of h_t
            for (size_t a = 0; a < A; ++a) if (C.pivot[a]) piv.push_back((int)a);
            auto c_of = [&](size_t a) {                // c_a as an axis bitmap
                std::vector<char> c(A, 0);
                c[a] = 1;
                for (int t : bits(C.kappa[a])) c[(size_t)piv[(size_t)t]] ^= 1;
                return c;
            };
            auto as_set = [&](const std::vector<char>& c) {
                std::vector<int> s;
                for (size_t a = 0; a < A; ++a) if (c[a]) s.push_back((int)a);
                return s;
            };
            for (size_t a = 0; a < A; ++a) {
                if (C.pivot[a]) continue;
                const std::vector<char> c = c_of(a);
                K.push_back(as_set(c));
                qeccore::Pauli P(sk.N);
                for (size_t b = 0; b < A; ++b) if (c[b]) P = qeccore::Pauli::multiply(P, sk.axis_q[b]);
                betaK.push_back(xd::stabilizer_sign(sk, P));
                const bool is_gen = C.hidden[a].size() == 1 && C.hidden[a][0] == (int32_t)gen.size();
                if (is_gen) { gen.push_back((int)a); continue; }
                std::vector<char> ci = c;              // c_a + its generators: an element of K_I
                for (int32_t t : C.hidden[a]) {
                    const std::vector<char> cg = c_of((size_t)gen[(size_t)t]);
                    for (size_t b = 0; b < A; ++b) ci[b] ^= cg[b];
                }
                KI.push_back(as_set(ci));
            }
        } catch (const xd::WriterError& e) {
            raise_dem(e.what());
        }
        const auto& sk = *W.sk;
        const auto& H = W.header;
        py::list axis_paulis, kappa_raw, pivot, gauge, hidden, bhat;
        for (size_t a = 0; a < H.axes.size(); ++a) {
            axis_paulis.append(stim_str(sk.axis_q[a]).substr(1));   // unsigned (Hermitian +)
            kappa_raw.append(xd::to_hex(xd::Bits{H.axes[a].k}));
            pivot.append((int)C.pivot[a]);
            gauge.append(bits(C.kappa[a]));
            hidden.append(C.hidden[a]);
            bhat.append((int)C.bhat[a]);
        }
        py::list stabs;                                // U Z_w U^dag, w = every deferred wire
        for (int w = 0; w < sk.N; ++w) {
            qeccore::Pauli Z(sk.N);
            Z.setz(w);
            for (const auto& o : sk.ops) xd::conj_forward(Z, o);   // M <- G M G^dag
            stabs.append(stim_str(Z));
        }
        const qeccore::ParsedStim ps = qeccore::parse_stim_circuit(text, true);
        py::dict d;
        d["axis_paulis"] = axis_paulis;
        d["kappa_raw"] = kappa_raw;
        d["pivot"] = pivot;
        d["gauge"] = gauge;
        d["hidden"] = hidden;
        d["bhat"] = bhat;
        d["K_basis"] = K;
        d["KI_basis"] = KI;
        d["beta_K"] = betaK;
        d["skeleton_stabilizers"] = stabs;
        py::list groups;                               // per group: the user gate and (axis, sign) per v = 1 ..
        for (const auto& g : H.groups) groups.append(py::make_tuple(g.gate, g.v_axis));
        d["groups"] = groups;
        d["Gl"] = xd::late_group_count(ps);
        d["tc_step"] = xd::last_detector_point(ps).step;
        d["r"] = C.r;
        d["m"] = C.m;
        return d;
    }, py::arg("text"),
       "xdem Task 3 TEST API: the writer's canonical header of `text` (SPEC §2): axis_paulis (unsigned end Paulis over "
       "the deferred wires), kappa_raw (hex, information-set basis), pivot, gauge, hidden, bhat (per axis), K_basis "
       "({c_a} over the non-pivot axes), KI_basis (c_a + its generators, for the c_a in K_I + span), beta_K (beta of "
       "each K_basis element by the skeleton), skeleton_stabilizers (the N generators U Z_w U^dag, signed), groups "
       "((user gate, [(axis, sign) per v = 1 .. 2^arity - 1; (-1, 0) = identity]) per group), Gl, tc_step, r, m. "
       "Refusals raise xtim.XtimDemError.");

    m.def("xdem_deferred_debug", [](const std::string& text) {
        // The xdem export's deferred circuit (dem_preamble: parse, site stamp, DEM normalization), what the engine
        // oracles of tests/_xdem_engine.py build their engine texts from (as idealcorr/ilib.Circ reads dccore's dfr /
        // M). t_c is located by a MARKER: a zero-probability noise instruction on a fresh qubit (index n, touched by
        // nothing else), inserted into a copy of the parsed stream right before the first instruction whose user step
        // is after t_c's (xdem::last_detector_point). The DEM normalization lets a zero-probability noise pass every
        // pending Clifford frame (zero_noise_transparent) and the fresh qubit interacts with nothing, so the copy's
        // stream minus the marker must equal the export's stream up to a relabelling of the deferred wires (the fresh
        // qubit's wire shifts the deferral's later fresh wires); this is checked instruction by instruction, with one
        // consistent bijection of wires (refused by name otherwise).
        qeccore::DemPreamble pre;
        qeccore::DemExportResult res;
        if (!qeccore::dem_preamble(text, false, nullptr, pre, res, /*build_bare=*/false)) {
            std::string e = res.error;
            if (!res.parse_errors.empty()) e = "parse error: " + res.parse_errors[0].message;
            else if (res.rejected) e = "rejected (propagation class) at gate " + std::to_string(res.reject_gate_index);
            raise_dem(e);
        }
        const qeccore::Circuit& def = pre.dn.deferred;
        // the marker copy: the preamble's site stamp, then the marker
        qeccore::ParsedStim ps = qeccore::parse_stim_circuit(text, true);
        if (!ps.ok()) raise_dem("xdem_deferred_debug: parse error");
        constexpr int kMarker = -7;
        {
            int site = 0;
            for (qeccore::Instr& ins : ps.circuit.stream)
                if (ins.kind == qeccore::Instr::Kind::Noise) {
                    ins.origin = qeccore::Instr::Origin::UserSite;
                    ins.origin_index = site++;
                }
        }
        const xd::TcPoint tcp = xd::last_detector_point(ps);
        size_t at = 0;
        int early_reads = 0;                               // records measured at or before t_c's user step
        while (at < ps.circuit.stream.size() && ps.stream_step[at] <= tcp.step)
            early_reads += ps.circuit.stream[at++].kind == qeccore::Instr::Kind::Measure;
        qeccore::Instr mk;
        mk.kind = qeccore::Instr::Kind::Noise;
        mk.channel = qeccore::NoiseChannel::X_ERROR;
        mk.probs = {0.0};
        mk.qubits = {ps.circuit.n};
        mk.origin = qeccore::Instr::Origin::UserSite;
        mk.origin_index = kMarker;
        ps.circuit.n += 1;
        ps.circuit.stream.insert(ps.circuit.stream.begin() + (std::ptrdiff_t)at, mk);
        qeccore::NormalizePolicy pol;
        pol.feedback = qeccore::NormalizePolicy::Feedback::KeepCoherent;
        pol.defer = true;
        const qeccore::DemNormalized dm = qeccore::dem_normalize(ps.circuit, pol);
        int tc_index = -1;
        {
            std::map<int, int> fwd, bwd;                   // marked-stream wire <-> export wire
            auto wire_ok = [&](int a, int b) {
                const auto f = fwd.emplace(a, b).first, g = bwd.emplace(b, a).first;
                return f->second == b && g->second == a;
            };
            auto same_mapped = [&](const qeccore::Instr& a, const qeccore::Instr& b) {
                if (!(a.kind == b.kind && a.gate == b.gate && a.basis == b.basis && a.channel == b.channel &&
                      a.probs == b.probs && a.origin == b.origin && a.origin_index == b.origin_index &&
                      a.targets.size() == b.targets.size() && a.qubits.size() == b.qubits.size() &&
                      a.obs.size() == b.obs.size()))
                    return false;
                for (size_t t = 0; t < a.targets.size(); ++t) if (!wire_ok(a.targets[t], b.targets[t])) return false;
                for (size_t t = 0; t < a.qubits.size(); ++t) if (!wire_ok(a.qubits[t], b.qubits[t])) return false;
                for (size_t t = 0; t < a.obs.size(); ++t)
                    if (a.obs[t].p != b.obs[t].p || !wire_ok(a.obs[t].qubit, b.obs[t].qubit)) return false;
                return true;
            };
            size_t k = 0;                                  // index into the export's stream
            for (size_t i = 0; i < dm.deferred.stream.size(); ++i) {
                const qeccore::Instr& ins = dm.deferred.stream[i];
                if (ins.kind == qeccore::Instr::Kind::Noise && ins.origin == qeccore::Instr::Origin::UserSite &&
                    ins.origin_index == kMarker) {
                    if (tc_index >= 0) raise_dem("xdem_deferred_debug: the t_c marker was split");
                    tc_index = (int)k;
                    continue;
                }
                if (k >= def.stream.size() || !same_mapped(ins, def.stream[k]))
                    raise_dem("xdem_deferred_debug: the marked stream differs from the export's at deferred "
                              "instruction " + std::to_string(k) + " (" + deferred_line(ins) + " vs " +
                              (k < def.stream.size() ? deferred_line(def.stream[k]) : std::string("end")) + ")");
                ++k;
            }
            if (k != def.stream.size()) raise_dem("xdem_deferred_debug: the marked stream is shorter");
            if (tc_index < 0) raise_dem("xdem_deferred_debug: the t_c marker did not survive normalization");
        }
        py::list lines;
        py::dict sites, folds;
        for (size_t i = 0; i < def.stream.size(); ++i) {
            const qeccore::Instr& ins = def.stream[i];
            lines.append(deferred_line(ins));
            if (ins.kind != qeccore::Instr::Kind::Noise) continue;
            const qeccore::NoiseFrameProvenance* fp = ins.frame_prov.get();
            if (ins.origin == qeccore::Instr::Origin::UserSite) {
                py::list part;                             // (user position, wire, deferred letter per user X/Y/Z)
                for (size_t j = 0; j < ins.qubits.size(); ++j) {
                    py::list lt;
                    for (int c = 1; c <= 3; ++c) lt.append(fp ? (int)fp->letter[3 * j + (size_t)c - 1] : c);
                    part.append(py::make_tuple(fp ? fp->pos[j] : (int)j, ins.qubits[j], lt));
                }
                const py::int_ key(ins.origin_index);
                if (!sites.contains(key)) sites[key] = py::list();
                py::list(sites[key]).append(py::make_tuple((int)i, part));
            } else if (ins.origin == qeccore::Instr::Origin::ReadoutFold) {
                using NC = qeccore::NoiseChannel;
                if ((ins.channel != NC::X_ERROR && ins.channel != NC::Y_ERROR && ins.channel != NC::Z_ERROR) ||
                    ins.qubits.size() != 1)
                    raise_dem("xdem_deferred_debug: a folded readout flip is not one X/Y/Z_ERROR target");
                const char* L = ins.channel == NC::X_ERROR ? "X" : ins.channel == NC::Y_ERROR ? "Y" : "Z";
                folds[py::int_(ins.origin_index)] = py::make_tuple((int)i, ins.qubits[0], std::string(L));
            } else {
                raise_dem("xdem_deferred_debug: a deferred noise instruction has no provenance");
            }
        }
        py::list reads;
        for (const auto& [b, w] : pre.reads) reads.append(py::make_tuple(std::string(1, "XYZ"[b]), w));
        py::list drecs, orecs;
        for (const auto& d : pre.p.detectors) {
            std::vector<int> s = d;
            std::sort(s.begin(), s.end());
            std::vector<int> o;
            for (size_t a = 0; a < s.size();) {
                size_t b = a;
                while (b < s.size() && s[b] == s[a]) ++b;
                if ((b - a) & 1) o.push_back(s[a]);
                a = b;
            }
            drecs.append(o);
        }
        int maxo = -1;
        for (const auto& kv : pre.p.observables) maxo = std::max(maxo, kv.first);
        for (int k = 0; k <= maxo; ++k) {
            std::vector<int> s;
            auto it = pre.p.observables.find(k);
            if (it != pre.p.observables.end()) s = it->second;
            std::sort(s.begin(), s.end());
            std::vector<int> o;
            for (size_t a = 0; a < s.size();) {
                size_t b = a;
                while (b < s.size() && s[b] == s[a]) ++b;
                if ((b - a) & 1) o.push_back(s[a]);
                a = b;
            }
            orecs.append(o);
        }
        py::dict d;
        d["deferred_text"] = lines;
        d["num_wires"] = def.n;
        d["reads"] = reads;
        d["detector_recs"] = drecs;
        d["observable_recs"] = orecs;
        d["sites"] = sites;
        d["readout_folds"] = folds;
        d["tc_index"] = tc_index;
        d["early_reads"] = early_reads;
        return d;
    }, py::arg("text"),
       "xdem Task 10 TEST API: the xdem export's deferred circuit. deferred_text: one line per deferred stream "
       "instruction (gates by name, non-Clifford gates kept; NOISE / M / MX / MY / OBS placeholders); num_wires; "
       "reads: (basis, wire) per record; detector_recs / observable_recs: raw (feedback-folded) record sets; sites: "
       "{user noise site: [(deferred instruction index, [(user qubit position, wire, [deferred letter of user X, Y, "
       "Z])])]}; readout_folds: {record: (deferred instruction index, wire, letter)} (a folded readout flip fires that "
       "letter there); "
       "tc_index: t_c as the deferred instruction index it precedes; early_reads: the number of records measured at or "
       "before t_c (reads[:early_reads]). Refusals raise xtim.XtimDemError.");
}
