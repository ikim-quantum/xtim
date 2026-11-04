// pybind11 bindings for the xtim Python package (extension module xtim._xtim).
//
// Exposes the engine entry points of the public package:
//   * parse_info        — parse a strict-Stim-superset circuit, line-numbered errors;
//   * export_dem_text   — the Stim-compatible DEM text (run_stim_main --dem semantics);
//   * the twirl record sampler (TwirlSampler), the bare / input / ideal state API and the port contract.
// (exact-out T5: the exact engine's entry points — CompiledProgram, ExactBranchProgram / ExactLawExpander, the
//  .ref surface ref_info / ref_verify_cheap / ref_compile_text / deferred_signature, the TreePlan memo diagnostics —
//  moved to the test API, cpp/testapi/exact_py.cpp. The exact engine itself is the test oracle.)
//
// Contract notes:
//   * Every function returns a plain dict with an "ok" flag and structured error info;
//     the Python layer (xtim/) converts these into the typed exceptions. No exception
//     types are defined here.
//   * The GIL is released around every engine call; results are computed into plain C++
//     structs first, then converted to Python objects.
#include "qeccore/exact_scope.hpp"

#include "qeccore/engine_version.hpp"
#include <cstddef>
#include "qeccore/bare_state_errors.hpp"
#include "qeccore/state_budget.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <string>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <vector>

#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/deferral.hpp"
#include "qeccore/dem_export.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/bare_state_memo.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_rotation_form.hpp"
#include "qeccore/port_contract.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/record_fidelity.hpp"
#include "qeccore/ref_io.hpp"
#include "qeccore/resolve_branches.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_sampler.hpp"

#include "injection.hpp"   // J-b1 P9a: the state-level injection helpers (shared with the test API)
#include "common_py.hpp"   // J-b1 F: the binding helpers (shared with the test API)
#include "framed_py.hpp"   // J-b1 F: PyFramedSuperposition (module_local; shared with the test API)
#include "twirl_front.hpp"   // J-b1 F: normalization presets + refusal text (shared with the legacy sampler)
#include "family_sampler.hpp"   // J-b1 F: THE twirl sampler (the family record law; shared with the test API)

namespace py = pybind11;
using namespace qeccore;

namespace {

// kEngineVersion (and its history) lives in qeccore/engine_version.hpp.
// engine-9: PAULI_EXPECTATION declared byproduct frames migrated onto the benchmark circuits (code_switching_faithful {16..30} = the full logical MX readout, miniature_oracle {2}); the engine folds each declared frame into the per-shot sign so the noiseless channel is sign-CONSTANT, and diagnose sources the sign from the DECLARED frame (statistical _discover_frame retired) + refuses a frame that does not SPAN the sign-controlling generators via expectation_frames (span verify accepts any determinizing set, incl. supersets).
// engine-8: one engine — the framed circuit-level engine (FramedSuperposition) carries ALL modes (memory R=0 + expectation R>0); the legacy shot loop and LeanState are deleted, chi>2 fully supported, unified TreePlan, .ref format v4 (no anchor block).
// engine-7: parity-native bare-state rebuild (build_frontier/GradedPoly removed; FastTODD-only; chi cap removed -> propagation-class rejection is the sole reject) + lean sampling path + compile-cache.
                                                         // engine-6: ref-format v3 direct-construction
                                                         // gauge (structure-preserving .ref; direct-built
                                                         // CanonicalStabSum by default; gauge-independent
                                                         // verification: NN-1 from_rays ORACLE + NN-2
                                                         // distribution-equivalence gate (e) + NN-3
                                                         // per-pin stream localization). .ref bytes changed
                                                         // (v3 format), invalidating stale engine-5 v2 refs.
                                                         // engine-5: ref-format v2 shared-anchor prep
                                                         // (anchor block + per-branch Pauli suffix) — the
                                                         // .ref byte layout changed, invalidating stale
                                                         // engine-4 v1-format/full-prep cached refs.
                                                         // engine-4: a chi-reduction conditioning pass in
                                                         // build_bare_state lowers the COMPILED reference
                                                         // chi (e.g. cube_ccz 16->8). This is a reference-
                                                         // COMPILE-algorithm change that does NOT change the
                                                         // deferred circuit, so the deferred-signature cache
                                                         // key is unchanged — this version bump is the
                                                         // backstop that invalidates stale engine-3 chi=16 refs.
                                                         // engine-3: Hadamard-minimal MPP desugar (Z-ancilla).
                                                         // engine-2: deduced build_bare_state gauge.

// ── helpers ──────────────────────────────────────────────────────────────────────────


py::list errors_to_py(const std::vector<ParseError>& es) {
    py::list out;
    for (const auto& e : es) out.append(py::make_tuple(e.line, e.message));
    return out;
}

// ── bound functions ──────────────────────────────────────────────────────────────────

py::dict py_parse_info(const std::string& text) {
    ParsedStim p;
    {
        py::gil_scoped_release rel;
        p = parse_stim_circuit(text);
    }
    py::dict d;
    d["ok"] = p.ok();
    d["errors"] = errors_to_py(p.errors);
    if (p.ok()) {
        int meas = 0;
        for (const auto& ins : p.circuit.stream)
            if (ins.kind == Instr::Kind::Measure) ++meas;
        int max_obs = -1;
        for (const auto& kv : p.observables)
            if (kv.first > max_obs) max_obs = kv.first;
        d["n"] = p.circuit.n;
        d["num_measurements"] = meas;
        d["num_detectors"] = (int)p.detectors.size();
        d["num_observables"] = max_obs + 1;
        d["num_expectations"] = (int)p.expectation_labels.size();
        // M-B coordinate surface (shift-resolved; mirrors Stim's get_final_qubit_coordinates /
        // get_detector_coordinates). qubit_coords: dict qubit->list[float]; detector_coords: a
        // list parallel to detectors (declaration order), [] for a coordinate-free DETECTOR.
        py::dict qc;
        for (const auto& kv : p.qubit_coords) {
            py::list cs;
            for (double v : kv.second) cs.append(v);
            qc[py::int_(kv.first)] = std::move(cs);
        }
        d["qubit_coords"] = std::move(qc);
        py::list dc;
        for (const auto& coords : p.detector_coords) {
            py::list cs;
            for (double v : coords) cs.append(v);
            dc.append(std::move(cs));
        }
        d["detector_coords"] = std::move(dc);
    }
    return d;
}


py::dict py_export_dem_text(const std::string& text,
                            bool include_expectations,
                            bool decompose_errors,
                            bool ignore_decomposition_failures,
                            bool drop_gauge_observables,
                            const std::vector<int>& trusted_detectors,
                            const std::vector<int>& trusted_observables,
                            double approximate_disjoint_errors,
                            int clifford_cutoff,
                            bool erasure) {
    if (clifford_cutoff < qeccore::kDemCliffordCutoffExact)
        throw std::invalid_argument("export_dem_text: clifford_cutoff must be >= 0 (or -1 = exact), got " +
                                    std::to_string(clifford_cutoff));
    DemExportResult dr;
    {
        py::gil_scoped_release rel;
        dr = export_dem(text, include_expectations,
                        decompose_errors, ignore_decomposition_failures,
                        drop_gauge_observables,
                        trusted_detectors.empty() ? nullptr : &trusted_detectors,
                        trusted_observables.empty() ? nullptr : &trusted_observables,
                        approximate_disjoint_errors, clifford_cutoff, nullptr, erasure);
    }
    py::dict d;
    d["ok"] = dr.ok;
    d["dem"] = dr.dem;
    d["errors"] = errors_to_py(dr.parse_errors);
    d["rejected"] = dr.rejected;
    d["reject_gate_index"] = dr.reject_gate_index;
    d["error"] = dr.error;
    d["dropped_observables"] = dr.dropped_observables;
    d["reject_detectors"] = dr.reject_detectors;
    d["postselect_faults"] = dr.postselect_faults;
    d["dropped_expectation_columns"] = dr.dropped_expectation_columns;
    d["clifford_faults"] = dr.clifford_faults;
    d["clifford_dropped_mass"] = dr.clifford_dropped_mass;
    return d;
}

// Exact per-detector/observable determinism partition (parity-Pauli Born check on the
// noiseless bare state). Independent of DEM expressibility — answers even when the full DEM
// refuses. Shot-free; the source-of-truth determinism verdict (ler's keep mask; adaptq).
py::dict py_detector_determinism(const std::string& text) {
    DeterminismResult r;
    {
        py::gil_scoped_release rel;
        r = detector_determinism(text);
    }
    py::dict d;
    d["ok"] = r.ok;
    d["errors"] = errors_to_py(r.parse_errors);
    d["rejected"] = r.rejected;
    d["reject_gate_index"] = r.reject_gate_index;
    d["error"] = r.error;
    d["num_detectors"] = r.num_detectors;
    d["num_observables"] = r.num_observables;
    d["deterministic_detectors"] = r.deterministic_detectors;
    d["gauge_detectors"] = r.gauge_detectors;
    d["deterministic_observables"] = r.deterministic_observables;
    d["gauge_observables"] = r.gauge_observables;
    d["detector_parity"] = r.detector_parity;
    d["observable_parity"] = r.observable_parity;
    return d;
}

// Exact verify of each declared PAULI_EXPECTATION byproduct frame (Task 4). Sampling-free;
// answers whether a declared `rec[-k]` frame equals the true sign-controlling record set (a
// property of the bare state's FIXED stabilizer frame). Sibling of detector_determinism.
py::dict py_expectation_frames(const std::string& text) {
    FrameCheckResult r;
    {
        py::gil_scoped_release rel;
        r = expectation_frame_check(text);
    }
    py::dict d;
    d["ok"] = r.ok;
    d["errors"] = errors_to_py(r.parse_errors);
    d["rejected"] = r.rejected;
    d["reject_gate_index"] = r.reject_gate_index;
    d["error"] = r.error;
    d["num_columns"] = r.num_columns;
    d["required"] = r.required;
    d["declared"] = r.declared;
    d["solvable"] = r.solvable;
    d["column_ok"] = r.column_ok;
    d["all_ok"] = r.all_ok;
    d["refusal"] = r.refusal;
    return d;
}

// DEM column layout — structural query for the module-system observable columns.
// Returns a dict with: ok, errors, rejected, reject_gate_index, error, n_obs (O+R),
// logical_flip [(decision_k, col), ...].
py::dict py_dem_column_layout(const std::string& text) {
    DemColumnLayout r;
    {
        py::gil_scoped_release rel;
        r = dem_column_layout(text);
    }
    py::dict d;
    d["ok"] = r.ok;
    d["errors"] = errors_to_py(r.parse_errors);
    d["rejected"] = r.rejected;
    d["reject_gate_index"] = r.reject_gate_index;
    d["error"] = r.error;
    d["n_obs"] = r.n_obs;
    // logical_flip: list of (decision_k, col)
    py::list lf_list;
    for (auto& [k2, c] : r.logical_flip)
        lf_list.append(py::make_tuple(k2, c));
    d["logical_flip"] = lf_list;
    return d;
}

// ── PyFramedSuperposition — Python handle for a FramedSuperposition (S2.1/S2.2) ──────────
// S2.1: opaque handle; created by bare_state_of and passed to TwirlSampler(state,…).
// S2.2 additions: approx_equal (exact_sum_overlap-based, global-phase insensitive) +
//   apply_clifford (for the mutation check: mutate a copy, verify approx_equal fails).
// PyFramedSuperposition: cpp/bindings/framed_py.hpp (J-b1 F; shared with the test API)

// ── S2.1: bare_state_of — build the FramedSuperposition bare state of a circuit ─────────
// Replicates the normalization + build_bare_state path from PyTwirlSampler's circuit-only
// constructor and returns the result as a PyFramedSuperposition. Used by
// compile_twirl_sampler_from_ideal_state (with input_state_of) to inject a pre-built ideal state instead of rebuilding
// it from the deferred circuit each time.
// TEST ORACLE (3.1.8): build the bare state of `text` twice — the production coset-family
// construction and the 3.1.7 per-branch construction (AssemblePath::ray_oracle) — and compare the
// two CanonicalStabSums field by field, coefficients BITWISE. Returns "" when byte-identical, else a
// description of the first difference. Also returns chi.
static std::string css_first_diff(const CanonicalStabSum& a, const CanonicalStabSum& b) {
    a.ensure_frame_current();
    b.ensure_frame_current();
    if (a.n() != b.n()) return "n";
    const int n = a.n();
    auto pauli_eq = [](const Pauli& x, const Pauli& y) {
        return x.n == y.n && (((x.phase - y.phase) % 4 + 4) % 4) == 0 && x.x == y.x && x.z == y.z;
    };
    for (int r = 0; r < n; ++r) {
        if (!pauli_eq(a.U.Zrow[(size_t)r], b.U.Zrow[(size_t)r])) return "U.Zrow[" + std::to_string(r) + "]";
        if (!pauli_eq(a.U.Xrow[(size_t)r], b.U.Xrow[(size_t)r])) return "U.Xrow[" + std::to_string(r) + "]";
    }
    if (a.eps != b.eps) return "eps";
    if (a.free != b.free) return "free";
    const AffineState &p = *a.anchor, &q = *b.anchor;
    if (p.n_ != q.n_ || p.k_ != q.k_ || p.b != q.b || p.D != q.D) return "anchor (n/k/b/D)";
    if (!p.omega.exact_eq(q.omega)) return "anchor omega";
    for (int i = 0; i < p.n_; ++i)
        for (int j = 0; j < p.k_; ++j) if (p.R.get(i, j) != q.R.get(i, j)) return "anchor R";
    for (int i = 0; i < p.k_; ++i)
        for (int j = 0; j < p.k_; ++j) if (p.J.get(i, j) != q.J.get(i, j)) return "anchor J";
    if (a.branches.size() != b.branches.size()) return "chi";
    for (size_t i = 0; i < a.branches.size(); ++i) {
        if (a.branches[i].sigma != b.branches[i].sigma) return "branch " + std::to_string(i) + " sigma";
        if (std::memcmp(&a.branches[i].c, &b.branches[i].c, sizeof(std::complex<double>)) != 0)
            return "branch " + std::to_string(i) + " coefficient bytes";
    }
    return "";
}

py::tuple py_bare_state_oracle_diff(const std::string& text) {
    std::string diff, err;
    int chi = 0;
    {
        py::gil_scoped_release rel;
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            err = "parse failed";
        } else {
            NormalizeResult nr = normalize_for_twirl(ps.circuit);
            BareState fast = build_bare_state_parity(nr.normalized, AssemblePath::coset_family);
            BareState slow = build_bare_state_parity(nr.normalized, AssemblePath::ray_oracle);
            if (fast.rejected != slow.rejected) diff = "rejected flag";
            else if (!fast.rejected) {
                chi = fast.chi;
                diff = fast.chi != slow.chi ? std::string("chi") : css_first_diff(fast.state, slow.state);
            }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    return py::make_tuple(diff, chi);
}

// TEST ORACLE (3.1.9): the production coset-family construction (unit-ray syndromes and weight <= 2
// gauges from the Pauli algebra) vs the 3.1.8 one (the same quantities read off materialised,
// disentangled, inner-producted rays; AssemblePath::coset_rays).  Compares every unit-ray syndrome
// (initial frame), the re-based linear syndrome columns, every weight-1/2 gauge exponent, b_ij, and
// then the whole CanonicalStabSum field by field with coefficient BYTES.  Returns (diff, chi, r).
static std::string coset_trace_first_diff(const CanonicalStabSum::CosetTrace& a,
                                          const CanonicalStabSum::CosetTrace& b) {
    if (a.fs.size() != b.fs.size()) return "unit-ray count";
    for (size_t i = 0; i < a.fs.size(); ++i)
        if (a.fs[i] != b.fs[i]) return i == 0 ? std::string("ray-0 syndrome")
                                              : "unit-ray syndrome e_" + std::to_string(i - 1);
    if (a.u != b.u) return "re-based syndrome columns u";
    for (size_t i = 0; i < std::max(a.a1.size(), b.a1.size()); ++i)
        if (i >= a.a1.size() || i >= b.a1.size() || a.a1[i] != b.a1[i])
            return "weight-1 gauge z8 at e_" + std::to_string(i);
    for (size_t i = 0; i < std::max(a.a2.size(), b.a2.size()); ++i)
        if (i >= a.a2.size() || i >= b.a2.size() || a.a2[i] != b.a2[i])
            return "weight-2 gauge z8 at pair #" + std::to_string(i);
    if (a.b2 != b.b2) return "b_ij";
    return "";
}

py::tuple py_ray_syndrome_oracle_diff(const std::string& text) {
    std::string diff, err;
    int chi = 0, r = -1;
    {
        py::gil_scoped_release rel;
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            err = "parse failed";
        } else {
            NormalizeResult nr = normalize_for_twirl(ps.circuit);
            CanonicalStabSum::CosetTrace ta, tb;
            ta.a1.assign(1, -1); tb.a1.assign(1, -2);          // "never filled" differs on purpose
            BareState fast = build_bare_state_parity(nr.normalized, AssemblePath::coset_family, &ta);
            BareState slow = build_bare_state_parity(nr.normalized, AssemblePath::coset_rays, &tb);
            if (fast.rejected != slow.rejected) diff = "rejected flag";
            else if (!fast.rejected) {
                chi = fast.chi;
                r = (int)ta.fs.size() - 1;
                diff = coset_trace_first_diff(ta, tb);
                if (diff.empty())
                    diff = fast.chi != slow.chi ? std::string("chi") : css_first_diff(fast.state, slow.state);
            }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    return py::make_tuple(diff, chi, r);
}

// TEST ORACLE for build_pauli_rotation_form: the production conjugation into the |0> frame vs the
// 3.1.8 per-gate backward walk (kept as the oracle). Two checks: (1) every magic generator's image,
// term by term, bits AND phase (cross-check mode); (2) the whole form built on the production path
// vs the whole form built on the walk path — terms (Pauli bits, phase, coeff), the Clifford list,
// global-phase BYTES and the reject fields. `normalize` = run normalize_for_twirl first (the
// production input); False feeds the parsed gate stream directly (every Clifford kind, CH, CS).
static std::string rotation_form_first_diff(const PauliRotationForm& a, const PauliRotationForm& b) {
    if (a.n != b.n) return "n";
    if (a.rejected != b.rejected) return "rejected";
    if (a.reject_gate_index != b.reject_gate_index) return "reject_gate_index";
    if (a.reject_reason != b.reject_reason) return "reject_reason";
    if (std::memcmp(&a.global_phase, &b.global_phase, sizeof(a.global_phase)) != 0) return "global_phase bytes";
    if (a.cliffords.size() != b.cliffords.size()) return "cliffords size";
    for (size_t i = 0; i < a.cliffords.size(); ++i)
        if (a.cliffords[i].kind != b.cliffords[i].kind || a.cliffords[i].targets != b.cliffords[i].targets)
            return "clifford " + std::to_string(i);
    if (a.terms.size() != b.terms.size()) return "terms size";
    for (size_t i = 0; i < a.terms.size(); ++i) {
        const Pauli &p = a.terms[i].pauli, &q = b.terms[i].pauli;
        if (p.n != q.n || p.phase != q.phase || p.x != q.x || p.z != q.z) return "term " + std::to_string(i) + " pauli";
        if (a.terms[i].coeff != b.terms[i].coeff) return "term " + std::to_string(i) + " coeff";
    }
    return "";
}

py::tuple py_rotation_form_oracle_diff(const std::string& text, bool normalize) {
    std::string diff, err;
    RotationFormCheck chk;
    size_t nterms = 0;
    {
        py::gil_scoped_release rel;
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            err = "parse failed";
        } else {
            Circuit c = normalize ? normalize_for_twirl(ps.circuit).normalized : ps.circuit;
            PauliRotationForm prod = build_pauli_rotation_form(c);
            PauliRotationForm walk = build_pauli_rotation_form(c, /*walk_oracle=*/true);
            PauliRotationForm cross = build_pauli_rotation_form(c, /*walk_oracle=*/false, &chk);
            nterms = prod.terms.size();
            if (!chk.first_diff.empty()) diff = chk.first_diff;
            else {
                diff = rotation_form_first_diff(prod, walk);
                if (diff.empty()) {
                    std::string d2 = rotation_form_first_diff(prod, cross);
                    if (!d2.empty()) diff = "cross-check build differs: " + d2;
                }
            }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    return py::make_tuple(diff, chk.terms_checked, chk.frame_terms, nterms);
}

PyFramedSuperposition py_bare_state_of(const std::string& text) {
    std::string err;
    FramedSuperposition fs(0);
    {
        py::gil_scoped_release rel;
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            err = "parse failed";
            for (const auto& e : ps.errors)
                err += "\n  line " + std::to_string(e.line) + ": " + e.message;
        } else {
            NormalizeResult nr = normalize_for_twirl_sampler(ps.circuit);
            BareState bs = build_bare_state(nr.normalized);
            if (bs.rejected) {
                err = bare_reject_message(bs, nr, ps);
            } else {
                fs = FramedSuperposition::from_css(std::move(bs.state));
            }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    return PyFramedSuperposition(fs);
}

// ── Task 2b.2: input_state_of — physical carried state + its output-wire map ────────────
// Builds the frame-explicit bare of `text` (same path as bare_state_of) AND returns the
// deferred-space wire indices of the declared OUTPUT_QUBITS (final_wire[orig], declaration
// order). The ⊗-composition uses this to locate WHERE the carried qubits live in the provided
// state (its content sits at final_wire coords, not the original indices — 2b.2-pre). A text
// with NO trailing measurement on an output qubit keeps that qubit's prep Clifford BAKED into
// the frame, so the returned state carries the physical carried state (e.g. |+>).
py::tuple py_input_state_of(const std::string& text) {
    std::string err;
    FramedSuperposition fs(0);
    std::vector<int> owires;
    {
        py::gil_scoped_release rel;
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            err = "parse failed";
            for (const auto& e : ps.errors)
                err += "\n  line " + std::to_string(e.line) + ": " + e.message;
        } else {
            NormalizeResult nr = normalize_for_twirl_sampler(ps.circuit);
            BareState bs = build_bare_state(nr.normalized);
            if (bs.rejected) {
                err = bare_reject_message(bs, nr, ps);
            } else {
                fs = FramedSuperposition::from_css(std::move(bs.state));
                const std::vector<int>& fw = nr.map.final_wire;
                for (const OutputPort& p : ps.circuit.outputs)
                    for (int q : p.qubits) {
                        if (q < 0 || q >= (int)fw.size()) {
                            err = "output qubit " + std::to_string(q) + " out of final_wire range";
                            break;
                        }
                        owires.push_back(fw[(size_t)q]);
                    }
            }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    // Task 4: port-contract validation — the OUTPUT_QUBITS wires must form a separable,
    // stabilizer-backed partition.  This is a compile-time check (rank test on the symplectic
    // frame), so there is no per-shot cost.  Fires for BOTH the public and underscore name.
    {
        PortVerdict v = qeccore::port_contract(fs, owires);
        if (!v.product) {
            const std::string escape_note =
                "\n\nIf this circuit's port only becomes separable AFTER measurement"
                " collapse, this\nstate is the wrong one to validate: build the"
                " post-collapse carried state via\nsample_barrier(...).materialize(i)"
                " and check it with\nport_contract_of_state(state, wires) instead.";
            throw py::value_error(v.witness + escape_note);
        }
    }
    return py::make_tuple(PyFramedSuperposition(fs), owires);
}

// ── Task 2b.2: input_qubits — the flattened INPUT_QUBITS port qubit indices (declaration order) ─
std::vector<int> py_input_qubits(const std::string& text) {
    ParsedStim ps;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
    }
    if (!ps.ok()) {
        std::string msg = "parse failed";
        for (const auto& e : ps.errors)
            msg += "\n  line " + std::to_string(e.line) + ": " + e.message;
        throw py::value_error(msg);
    }
    std::vector<int> out;
    for (const InputPort& p : ps.circuit.inputs)
        if (p.kind == InputPort::Kind::Qubits)
            for (int q : p.qubits) out.push_back(q);
    return out;
}

// Task 4 helper: _output_wires_of(text) — deferred wire indices of OUTPUT_QUBITS without
// building the bare state and WITHOUT running the port-contract validation.  Used by tests
// that need the wire map for entangled circuits (the differential oracle test).
std::vector<int> py_output_wires_of(const std::string& text) {
    std::string err;
    std::vector<int> owires;
    {
        py::gil_scoped_release rel;
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            err = "parse failed";
            for (const auto& e : ps.errors)
                err += "\n  line " + std::to_string(e.line) + ": " + e.message;
        } else {
            NormalizeResult nr = normalize_for_twirl(ps.circuit);
            const std::vector<int>& fw = nr.map.final_wire;
            for (const OutputPort& p : ps.circuit.outputs)
                for (int q : p.qubits) {
                    if (q < 0 || q >= (int)fw.size()) {
                        err = "output qubit " + std::to_string(q) + " out of final_wire range";
                        break;
                    }
                    owires.push_back(fw[(size_t)q]);
                }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    return owires;
}

// Task 2 free function: port_contract_of_state(state, wires) -> dict.
// Takes a PyFramedSuperposition and a list of wire indices; returns the port verdict dict.
// Public (the _port_contract_of_state alias was deleted at bloat-cut D). Used by the Task 3 invariance gate
// and by differential tests.
py::dict py_port_contract_of_state(const PyFramedSuperposition& st,
                                   const std::vector<int>& wires) {
    PortVerdict v = qeccore::port_contract(st.state, wires);
    return port_verdict_to_dict(v);
}

// ── Task 5: resolve_branches Python bindings ─────────────────────────────────

// Return list of (port_index, bit_index) tuples for bits driving any IF condition.
py::list py_if_driving_bits(const std::string& text) {
    ParsedStim ps;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
    }
    if (!ps.ok()) throw std::invalid_argument("parse failed");
    auto bits = if_driving_bits(ps.circuit);
    py::list out;
    for (const auto& b : bits)
        out.append(py::make_tuple(b.first, b.second));
    return out;
}

// Return list of global bit offsets for IF-driving bits.
py::list py_if_driving_global_offsets(const std::string& text) {
    ParsedStim ps;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
    }
    if (!ps.ok()) throw std::invalid_argument("parse failed");
    auto offsets = if_driving_global_offsets(ps.circuit);
    py::list out;
    for (int o : offsets) out.append(o);
    return out;
}

// Return total width of all Bits input ports.
int py_bits_total_width(const std::string& text) {
    ParsedStim ps;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
    }
    if (!ps.ok()) throw std::invalid_argument("parse failed");
    return bits_total_width(ps.circuit);
}

// Resolve IF branches for a given bit_values pattern and re-emit as text.
// pattern_bytes: a Python bytes / list[int] of length bits_total_width(text).
// Task 6: passes pattern_bytes to emit_resolved_text so rec[-k] refs are correctly
// remapped when a measurement-bearing branch is dropped.
std::string py_resolve_branches_text(const std::string& text,
                                     const std::vector<uint8_t>& pattern_bytes) {
    ParsedStim ps;
    std::string result;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
        if (!ps.ok()) {
            // Exit GIL release scope before throwing to re-acquire the GIL.
        } else {
            Circuit resolved = resolve_branches(ps.circuit, pattern_bytes);
            result = emit_resolved_text(resolved, ps, pattern_bytes);
        }
    }
    // Now that the GIL release scope is closed, raise if parse failed.
    if (!ps.ok()) {
        std::string msg = "parse failed";
        for (const auto& e : ps.errors)
            msg += "\n  line " + std::to_string(e.line) + ": " + e.message;
        throw py::value_error(msg);
    }
    return result;
}

// Task 6: return the IF guard conditions for each detector in the circuit.
// Returns a list (one element per detector) of lists of (global_bit_offset, cond_value).
// An empty inner list means the detector is outside all IF blocks (always active).
// Used by the Python partition layer to build the superset live-mask.
py::list py_detector_branch_spans(const std::string& text) {
    ParsedStim ps;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
    }
    if (!ps.ok()) {
        std::string msg = "parse failed";
        for (const auto& e : ps.errors)
            msg += "\n  line " + std::to_string(e.line) + ": " + e.message;
        throw py::value_error(msg);
    }
    auto spans = detector_branch_spans(ps);
    py::list result;
    for (const auto& conds : spans) {
        py::list det_conds;
        for (const auto& c : conds)
            det_conds.append(py::make_tuple(c.first, (int)c.second));
        result.append(det_conds);
    }
    return result;
}

}  // namespace

// Raise a typed xtim.errors exception (looked up at raise time; xtim.errors is pure Python and
// always importable once the package is). Falls back to `fallback` if the lookup itself fails.
static void raise_xtim_error(const char* cls, PyObject* fallback, const char* what) {
    try {
        py::object c = py::module_::import("xtim.errors").attr(cls);
        PyErr_SetString(c.ptr(), what);
    } catch (...) {
        PyErr_SetString(fallback, what);
    }
}

PYBIND11_MODULE(_xtim, m) {
    // Record-key byte layout of BarrierBuffer.record_keys()/record_groups(): 3 (J-b1 F, design §4) = [kind u8][len
    // u32 LE] plan content ‖ P_rec words ‖ y words (sigma retired). Readers check it (xtim.port).
    m.attr("RECORD_KEY_LAYOUT") = xtim_api::kRecordKeyLayout;
    // Bare-state refusals surface as TYPED errors that engine="auto" does not swallow (it only
    // reroutes ValueError/RuntimeError to the exact engine — which cannot represent the state
    // either): capacity -> xtim.XtimCapacityError (MemoryError), invariant -> XtimInvariantError.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const qeccore::BareStateCapacityError& e) {
            raise_xtim_error("XtimCapacityError", PyExc_MemoryError, e.what());
        } catch (const qeccore::BareStateInvariantError& e) {
            raise_xtim_error("XtimInvariantError", PyExc_AssertionError, e.what());
        }
    });
    m.doc() = "xtim engine bindings: parse / twirl record sampling / DEM export / state API.";
    m.attr("ENGINE_VERSION") = kEngineVersion;
    // J-b1 P13: the runtime exact-construction counter and the explicit-exact scope (xtim/_exact_scope.py).
    m.def("_exact_constructions", []() { return qeccore::exact_constructions(); });
    m.def("_reset_exact_constructions", []() { qeccore::reset_exact_constructions(); });
    m.def("_explicit_exact_enter", []() { qeccore::explicit_exact_enter(); });
    m.def("_explicit_exact_exit", []() { qeccore::explicit_exact_exit(); });
    // plancache-design null option (PRIVATE test lever; a process global read once per construction): samplers built
    // while on compile their plan classifier with Pi = id (the a ∧ U strip only).
    m.def("_plan_class_identity", [](bool on) { qeccore::g_twirl_plan_class_identity_test = on; }, py::arg("on"));

    mdef(m, "live_state_bytes", []() { return (long long)qeccore::live_state_bytes(); },
          "Bytes of chi-scaled state storage (bare-state branches, FramedSuperposition amplitude "
          "entries, parsed reference branches, build transients, the measurement kernels' "
          "per-thread scratch, live branch_frames() arrays) currently live in this process — "
          "the quantity the state-memory budget bounds.");
    mdef(m, "state_budget_bytes", []() { return (long long)qeccore::state_budget_bytes(); },
          "The process-wide state-memory budget in bytes: XTIM_MAX_STATE_BYTES (read once) or 2 GiB. "
          "An allocation that would push live_state_bytes() over it raises XtimCapacityError first.");
    mdef(m, "_release_thread_state_scratch", []() { qeccore::release_thread_state_scratch(); },
          "Free this thread's chi-scaled measurement scratch and reusable entry list now "
          "(diagnostics; 3.1.10 releases it automatically when no Python-held xtim object is left).");
    mdef(m, "_thread_scratch_bytes", []() { return (long long)qeccore::thread_scratch_bytes(); },
          "Bytes of chi-scaled measurement scratch charged by THIS thread (part of live_state_bytes).");
    mdef(m, "_scratch_owner_count", []() { return (long long)qeccore::scratch_owner_count(); },
          "Live Python-held xtim owners (samplers, states, buffers, compiled programs).");
    mdef(m, "parse_info", &py_parse_info, py::arg("text"),
          "Parse a circuit; returns {ok, errors=[(line, message)...], counts...}.");
    mdef(m, "export_dem_text", &py_export_dem_text, py::arg("text"),
          py::arg("include_expectations") = true,
          py::arg("decompose_errors") = false,
          py::arg("ignore_decomposition_failures") = false,
          py::arg("drop_gauge_observables") = false,
          py::arg("trusted_detectors") = std::vector<int>{},
          py::arg("trusted_observables") = std::vector<int>{},
          py::arg("approximate_disjoint_errors") = 0.0,
          py::arg("clifford_cutoff") = qeccore::kDemCliffordCutoffDefault,
          py::arg("erasure") = false,
          "Stim-compatible DEM text export (run_stim_main --dem semantics). "
          "erasure=True (research option, not stim/decoder-ingestible): each coset Clifford record is ONE "
          "`erasure[clifford:f<id>](p) <offset> | <gen 1> | ... | <gen v>` line (no cutoff, no enumeration). "
          "clifford_cutoff: each twirled Clifford fault is its own record (`error[clifford:f<id>]` "
          "lines) keeping its patterns that flip <= clifford_cutoff detectors (default 2 = graphlike, can lower "
          "the suppression order, the dropped mass is reported; -1 = every pattern, the exact split); result keys "
          "clifford_faults / clifford_dropped_mass. "
          "approximate_disjoint_errors: 0 = refuse a channel with no exact conversion into "
          "independent mechanisms (default); t in (0, 1] = stim's approximate_disjoint_errors "
          "(True = 1): such a channel is approximated (each merged signature an independent "
          "mechanism with its disjoint probability) if every probability argument is <= t. "
          "include_expectations: PAULI_EXPECTATION L-columns after the observables "
          "(default ON; False reproduces the observable/detector-only export). "
          "trusted_detectors/trusted_observables: ids whose determinism is certified "
          "externally against the TRUE composed bare state (from_state channel_report) — "
          "the text-only determinism gate is skipped for exactly those ids; mechanisms "
          "are state-free and unchanged.");
    mdef(m, "detector_determinism", &py_detector_determinism, py::arg("text"),
          "Exact per-detector/observable determinism partition of the noiseless circuit "
          "(parity-Pauli Born check on the bare state). Shot-free, and answers even when the "
          "full DEM refuses. Keys: ok, deterministic_detectors, gauge_detectors, "
          "deterministic_observables, gauge_observables, detector_parity, observable_parity, "
          "num_detectors, num_observables, rejected, reject_gate_index, error, errors.");
    mdef(m, "expectation_frames", &py_expectation_frames, py::arg("text"),
          "Exact verify of each declared PAULI_EXPECTATION byproduct frame. Sampling-free: "
          "decides whether the declared rec[-k] frame equals the true sign-controlling record "
          "set (a property of the bare state's fixed stabilizer frame; free/logical rows "
          "excluded). Keys: ok, num_columns, required, declared, solvable, column_ok, all_ok, "
          "refusal, rejected, reject_gate_index, error, errors. all_ok False => refusal names "
          "the mismatched columns and their missing/spurious records.");
    mdef(m, "dem_column_layout", &py_dem_column_layout, py::arg("text"),
          "DEM column layout for a module-system circuit (structural query, no DEM export). "
          "Keys: ok, errors, rejected, reject_gate_index, error, n_obs (O+R, existing L-columns), "
          "logical_flip [(decision_k, col), ...] (DECISION logical-flip columns). "
          "OUTPUT_QUBITS adds no DEM column.");
    mdef(m, "bare_state_memo_stats", []() {
            qeccore::BareStateMemoStats s = qeccore::bare_state_memo_stats();
            py::dict d;
            d["hits"] = s.hits; d["misses"] = s.misses; d["stores"] = s.stores;
            d["skipped"] = s.skipped; d["evictions"] = s.evictions; d["bypassed"] = s.bypassed;
            d["entries"] = s.entries; d["bytes"] = s.bytes;
            d["cap_entries"] = s.cap_entries; d["cap_bytes"] = s.cap_bytes;
            d["cap_entry_bytes"] = s.cap_entry_bytes;
            d["enabled"] = s.enabled; d["verify"] = s.verify;
            return d;
          },
          "Counters and bounds of the process-level framed bare-state memo (bare_state_memo.hpp): "
          "hits/misses/stores/skipped/evictions/bypassed, entries, bytes, caps (entries, bytes, "
          "bytes per entry), enabled, verify.");
    mdef(m, "bare_state_memo_reset_stats", []() { qeccore::bare_state_memo_reset_stats(); },
          "Zero the bare-state memo counters (entries kept).");
    mdef(m, "bare_state_memo_set_entry_cap", [](size_t max_entry_bytes) {
            qeccore::bare_state_memo_set_entry_cap(max_entry_bytes); },
          py::arg("max_entry_bytes"),
          "Per-entry byte cap of the bare-state memo: a larger state is never captured.");
    bind_framed_superposition(m);   // J-b1 F: cpp/bindings/framed_py.hpp
    // J-b1 F (fixplan-F-materialize-c0hat r2 item 1): the layout-3 BarrierBuffer returns THIS extension's handle.
    xtim_api::wrap_state_hook() = [](FramedSuperposition&& st) -> py::object {
        return py::cast(PyFramedSuperposition(std::move(st)));
    };
    // ── State-in / port-contract API naming convention ──────────────────────────
    // Public names only (bare_state_of, input_state_of, port_contract_of_state); bloat-cut D deleted the
    // _underscore back-compat aliases. _output_wires_of is INTENTIONALLY private with no public twin: it is
    // a non-validating wire-lookup for test harnesses that must inspect states the validating API refuses.

    // S2.1: bare_state_of.
    const char* bare_state_of_doc =
        "Build and return the FramedSuperposition bare state of `circuit_text` "
        "(the same build_bare_state(deferred) path as compile_twirl_sampler). "
        "(compile_twirl_sampler_from_ideal_state takes an ideal state from input_state_of / ideal_state()). "
        "Raises ValueError on parse failure or bare-state rejection.";
    mdef(m, "framed_from_rows",
          [](int n, const std::vector<std::vector<uint8_t>>& zx, const std::vector<std::vector<uint8_t>>& zz,
             const std::vector<int>& zph, const std::vector<std::vector<uint8_t>>& xx,
             const std::vector<std::vector<uint8_t>>& xz, const std::vector<int>& xph,
             const std::vector<uint8_t>& eps, const std::vector<int>& free,
             const std::vector<std::pair<std::vector<uint8_t>, std::complex<double>>>& entries) {
              return PyFramedSuperposition(framed_from_rows(n, zx, zz, zph, xx, xz, xph, eps, free, entries));
          },
          py::arg("n"), py::arg("zx"), py::arg("zz"), py::arg("zph"), py::arg("xx"), py::arg("xz"),
          py::arg("xph"), py::arg("eps"), py::arg("free"), py::arg("entries"),
          "Harness continuation (2026-10-07): a FramedSuperposition from explicit frame rows "
          "(Zrow[a] = i^zph X^zx Z^zz, Xrow[a] likewise), eps, free row indices and amplitude "
          "entries (sigma over free, coeff). Validated symplectic frame, Hermitian rows, unit norm. "
          "adaptq's port restriction (the carried state rebuilt on the port wires only).");
    mdef(m, "bare_state_of",  &py_bare_state_of, py::arg("circuit_text"), bare_state_of_doc);
    mdef(m, "_rotation_form_oracle_diff", &py_rotation_form_oracle_diff, py::arg("circuit_text"),
          py::arg("normalize") = true,
          "TEST ORACLE: (diff, terms_checked, frame_terms, n_terms) — diff is '' iff the production "
          "rotation form equals the per-gate-walk oracle term by term (bits AND phase) and as a whole "
          "form (terms, coeffs, Clifford list, global-phase bytes, reject fields).");
    mdef(m, "_bare_state_oracle_diff", &py_bare_state_oracle_diff, py::arg("circuit_text"),
          "TEST ORACLE: (diff, chi) — diff is '' iff the production (coset-family) bare state is "
          "byte-identical to the 3.1.7 per-branch construction (frame, eps, free, anchor, branch "
          "sigma, coefficient bytes).");
    mdef(m, "_ray_syndrome_oracle_diff", &py_ray_syndrome_oracle_diff, py::arg("circuit_text"),
          "TEST ORACLE: production coset-family construction (Pauli algebra) vs the 3.1.8 materialised "
          "weight<=2 rays — unit-ray syndromes, gauge exponents, b_ij, then the whole state BYTES. "
          "Returns (first_diff or '', chi, r).");

    // ── J-b1 F: THE twirl sampler — the family record law with every record class (plan §2.2), the layout-3
    //    BarrierBuffer and the layout-3 RecordFidelity (cpp/bindings/record_buffer.hpp). The binding source is the
    //    test API's (cpp/bindings/family_sampler.hpp), so F-B1a compares one code path. ──
    xtim_api::bind_record_buffer(m);
    py::class_<PyIdealState>(m, "IdealState", py::module_local(),
        "An ideal (carried) state: the frame-explicit state of a producer's OUTPUT_QUBITS (the post-barrier ideal "
        "state of TwirlSampler.ideal_state(), or a rows-only state from a FramedSuperposition), its wires and its "
        "frame token. Consumers compile against it (TwirlSampler.from_ideal_state); it is branch 0 of their branch "
        "table, and add_branch registers the producer's other record handles.")
        .def_property_readonly("token", [](const PyIdealState& i) { return PyFamilySampler::dbytes_(i.tok.d); })
        .def_property_readonly("provenance", [](const PyIdealState& i) { return i.tok.producer; })
        .def_property_readonly("wires", [](const PyIdealState& i) { return i.wires; });
    m.def("ideal_state_of_state", [](const PyFramedSuperposition& st, const std::vector<int>& wires) {
              for (int w : wires)
                  if (w < 0 || w >= st.state.n())
                      throw std::invalid_argument("ideal_state_of_state: a carried wire is out of range");
              PyIdealState i;
              i.fs = st.state;
              i.wires = wires;
              i.tok = frame_token_rows(i.fs);
              i.producer = py::none();
              return i;
          }, py::arg("state"), py::arg("wires"),
          "A rows-only ideal state (no producer provenance: branch 0 only) from a FramedSuperposition and its "
          "carried wires (xtim.input_state_of(text) gives both).");
    auto twirl_ctor = [](const std::string& text, double p_factor, long selfcheck, bool decisions,
                         const PyIdealState* ideal) {
        std::unique_ptr<PyFamilySampler> s;
        {
            qeccore::ScratchCallScope scope;
            s = std::make_unique<PyFamilySampler>(text, p_factor, selfcheck, /*oracle_all_shots=*/false, decisions,
                                                  /*seed=*/1, /*record_route_term=*/false, ideal);
        }
        if (ideal) {
            s->btab_ = std::make_shared<BranchTable>(ideal->tok);
            s->producer_ = ideal->producer;
        }
        return s;
    };
    // scaling fix-3: an opaque saved sampler state (TwirlSampler._save_call_state / _restore_call_state)
    py::class_<PyFamilySampler::CallStateHandle>(m, "_CallState", py::module_local());
    py::class_<PyFamilySampler>(m, "TwirlSampler", py::module_local(),
          "The twirl record sampler (qeccore::TwirlRecordSampler, the family record law; design "
          "docs/twirl_record_sampler.md). Every DETECTOR / OBSERVABLE / DECISION column is a record of the law: "
          "deterministic channels exact, gauge (LOGICAL) detectors exact Born records, ANTI detectors and second "
          "LOGICAL / ANTI observables served, Born decisions and retention on CH / PPR circuits served. Circuits "
          "outside the twirl class raise xtim.errors.XtimTwirlRefusal by name at construction. Thread safety: one "
          "call at a time per sampler (a concurrent call raises XT_SAMPLER_BUSY); threads= parallelises inside a call "
          "(byte-identical for every count); separate samplers are independent.")
        .def(py::init([twirl_ctor](const std::string& text, double p_factor, long selfcheck, bool decisions) {
                 return twirl_ctor(text, p_factor, selfcheck, decisions, nullptr);
             }),
             py::arg("circuit_text"), py::arg("p_factor") = 1.0, py::arg("selfcheck") = 2000,
             py::arg("decisions") = true)
        .def_static("from_ideal_state",
             [twirl_ctor](const PyIdealState& ideal_state, const std::string& circuit_text, double p_factor,
                          long selfcheck, bool decisions) {
                 return twirl_ctor(circuit_text, p_factor, selfcheck, decisions, &ideal_state);
             },
             py::kw_only(), py::arg("ideal_state"), py::arg("circuit_text"), py::arg("p_factor") = 1.0,
             py::arg("selfcheck") = 2000, py::arg("decisions") = true,
             py::keep_alive<0, 1>(),
             "A consumer sampler whose INPUT_QUBITS carry `ideal_state` (branch 0, state-level injection; every "
             "carried state, single-qubit stabilizer states included). DETECTOR / OBSERVABLE columns are "
             "reference-relative (the noiseless run from the ideal state); add_branch registers producer branches.")
        .def("sample", [](PyFamilySampler& s, long shots, py::object seed, py::object input_pauli,
                          py::object input_branch, py::object postselect, py::object threads) -> py::tuple {
                 if (shots > (long)INT32_MAX) throw std::invalid_argument("shots exceeds INT32_MAX (b8 array shape is int)");
                 const int nt = threads_arg(threads);
                 if (!postselect.is_none()) {       // early-reject: only the accepted shots, then the counts
                     const std::vector<uint8_t> f = postselect_flags(postselect, s.num_det_);
                     PyFamilySampler::PsCounts pc;
                     // scaling fix-4 (C): the result (with its counts) is built before the call commits
                     const std::function<void(py::tuple&)> fin = [&](py::tuple& t) {
                         if (s.num_dec_ > 0) t = py::make_tuple(t[0], t[1], t[2], ps_counts_dict(pc));
                         else t = py::make_tuple(t[0], t[1], ps_counts_dict(pc));
                     };
                     return budget_poison_on_refusal(s, [&] {
                         qeccore::ScratchCallScope scope;
                         return s.sample_ps(shots, seed, input_pauli, input_branch, f, pc, nt, &fin);
                     });
                 }
                 // Task 2b.1: (dets, obs, decisions); decision-free: (dets, obs) — built before the call commits
                 const std::function<void(py::tuple&)> fin = [&](py::tuple& t) {
                     if (s.num_dec_ == 0) t = py::make_tuple(t[0], t[1]);
                 };
                 return budget_poison_on_refusal(s, [&] {
                     qeccore::ScratchCallScope scope;
                     return s.sample(shots, seed, false, input_pauli, input_branch, nt, &fin);
                 });
             },
             py::arg("shots"), py::arg("seed") = py::none(), py::arg("input_pauli") = py::none(),
             py::arg("input_branch") = py::none(), py::kw_only(), py::arg("postselect") = py::none(),
             py::arg("threads") = 1,
             "Sample -> (detectors, observables[, decisions]), Stim-b8-packed uint8 arrays (shots, ceil(K/8)). "
             "seed=None continues the stream; an explicit seed reseeds in place (same seed -> same bytes). "
             "input_pauli: optional uint8 (shots, 2*support) per-shot Pauli frame on the INPUT_QUBITS port (X part "
             "then Z part). input_branch: optional int32 (shots,) per-shot carried branch (0 = the ideal state; "
             "from_ideal_state samplers). postselect (keyword): a boolean (dtype=bool) mask of length num_detectors, "
             "or a sequence of distinct detector indices (any non-bool array is read as indices; duplicate or "
             "out-of-range indices are refused, so pass a 0/1 mask as bool; empty: no detector flagged); returns only the shots whose flagged detectors are all 0 (in shot order, "
             "byte-identical to filtering an unmasked call) and appends the counts {attempts, passed, discarded, "
             "early} (early: shots rejected before any plan work). threads (keyword): an int >= 1 or \"auto\" (the "
             "CPUs this process may use); the call's shots run on up to that many threads with byte-identical output "
             "for every count (one call at a time per sampler: threads parallelise inside it). record_report()"
             "['threads'] gives the call's effective count and, when it is lower, why (runs the lanes do not serve run "
             "serially; the state budget caps the lanes).")
        .def("sample_barrier", [](py::object self, long shots, py::object seed, py::object input_pauli, bool compact,
                                  py::object input_branch, py::object postselect, py::object threads) {
                 PyFamilySampler& s = self.cast<PyFamilySampler&>();
                 const int nt = threads_arg(threads);
                 std::vector<uint8_t> f;
                 if (!postselect.is_none()) f = postselect_flags(postselect, s.num_det_);
                 return budget_poison_on_refusal(s, [&] {
                     qeccore::ScratchCallScope scope;
                     return family_sample_barrier_ps(self, shots, seed, input_pauli, compact, input_branch,
                                                     postselect.is_none() ? nullptr : &f, nt);
                 });
             },
             py::arg("shots"), py::arg("seed") = py::none(), py::arg("input_pauli") = py::none(),
             py::arg("compact") = true, py::arg("input_branch") = py::none(), py::kw_only(),
             py::arg("postselect") = py::none(), py::arg("threads") = 1, py::keep_alive<0, 1>(),
             "Sample shots keeping every shot's layout-3 record -> BarrierBuffer (dets / obs / decisions as sample(); "
             "per shot the record key [kind][len] content ‖ P_rec ‖ y and the outcomes u ‖ z; materialize(i) rebuilds "
             "the retained state from the record). A layout-3 buffer is always compact (compact=False is refused). "
             "postselect (keyword, as sample()): the buffer holds only the accepted shots; buffer.postselection "
             "carries the counts. threads (keyword, as sample()): byte-identical records for every count.")
        .def("_save_call_state", &PyFamilySampler::save_call_state_,
             "scaling fix-3: the sampler's state (stream, shot index, seed, per-shot counters) for a multi-call rollback")
        .def("_restore_call_state", &PyFamilySampler::restore_call_state_, py::arg("state"),
             "PRIVATE (scaling fix-3/4): put this sampler back to a state its own _save_call_state returned, for a "
             "Python-level multi-call operation that failed (PartitionedTwirlSampler.sample) — the failed call commits "
             "nothing. A state saved from another sampler is refused by name (ValueError). Not a public replay API: "
             "only the stream position, shot index, seed and per-shot counters are restored, never cached plans.")
        .def("record_fidelity", &family_record_fidelity_l3, py::arg("max_bytes") = (size_t)0,
             py::arg("cache_budget") = py::none(),
             "The layout-3 reference RecordFidelity evaluator (use xtim.RecordFidelity).")
        .def("record_report", [](const PyFamilySampler& s) {
                 const BusyHold h = s.hold_("record_report");
                 return BudgetGuard<&PyFamilySampler::record_report>::call(s); },
             "Record classes and run counters (exact at a fixed seed; cumulative since construction / the last "
             "seeded call): classes {detectors, observables, decisions: {index: deterministic | coin | born}}, "
             "nondeterministic_detectors, nondeterministic_observables, route_counts {route: shots}, counters, used, "
             "plans, plan_misses {kappa0, kappa_pos, fallback}, born_trie, structural_shots, rec, plan_memory, "
             "feedback_hoisted, branches (branch counters), branch_table_digest. born_trie's node_evals / "
             "passes / chi_j / max_support count the evaluations run; roots served by the root memo are counted "
             "in root_memo_hits only.")
        .def("set_plan_payload_budget", [](PyFamilySampler& s, py::object nbytes) {
                 const BusyHold h = s.hold_("set_plan_payload_budget"), hc = s.hold_cache_("set_plan_payload_budget");
                 const size_t b = cache_budget_arg(nbytes, "plan_payload_budget");
                 s.opt_.plan_payload_budget = b;
                 s.s_->set_plan_payload_budget(b);
             }, py::arg("nbytes"),
             "S0(c) / D6: caller byte budget for plan PAYLOADS and the term route's dispatch entries (None = the "
             "default, 512 MiB; math.inf = unbounded; 0 refused). Evicted entries are rebuilt on demand: byte-identical output for every "
             "budget.")
        .def("set_branch_budget", [](PyFamilySampler& s, py::object nbytes) {
                 const BusyHold h = s.hold_("set_branch_budget");
                 std::vector<std::unique_ptr<BusyHold>> hin;      // the E-class samplers' own flags
                 for (auto& ec : s.eclass_)
                     hin.push_back(std::make_unique<BusyHold>(ec.s->busy_, "XT_SAMPLER_BUSY", "sampler", "set_branch_budget"));
                 const size_t b = nbytes.is_none() ? SIZE_MAX : nbytes.cast<size_t>();
                 BA_::family_set_branch_budget(*s.s_, b);
                 for (auto& ec : s.eclass_) BA_::family_set_branch_budget(*ec.s->s_, b);
             }, py::arg("nbytes"),
             "D6: byte budget for carried-branch payloads (None = unbounded, the default). "
             "An evicted branch is re-supplied inside run() by the same pure construction: byte-identical output.")
        .def("channel_one_counts", [](const PyFamilySampler& s) {
                 const BusyHold h = s.hold_("channel_one_counts");
                 py::list out;
                 for (auto x : s.s_->channel_one_counts()) out.append(x);
                 return out;
             },
             "Per-channel fired-count list, accumulated only while the selfcheck window runs (the window's observable; "
             "all-zero at selfcheck=0 and once the window is spent). J-b1 F: kept — the family run accumulates it "
             "in its window too (jb2 A3 had listed it as legacy-only).")
        .def_property_readonly("num_detectors", [](const PyFamilySampler& s) { return s.num_det_; })
        .def_property_readonly("num_observables", [](const PyFamilySampler& s) { return s.num_obs_; })
        .def_property_readonly("num_decisions", [](const PyFamilySampler& s) { return s.num_dec_; },
             "Number of declared DECISION output bits (max DECISION index + 1).")
        .def("output_wires", &PyFamilySampler::output_wires,
             "Deferred-space wire indices of the declared OUTPUT_QUBITS (flattened, declaration order).")
        .def("output_ports", &PyFamilySampler::output_ports,
             "[(name, circuit qubits, deferred wires)] of the declared OUTPUT_QUBITS.")
        .def("record_operators", &PyFamilySampler::record_operators,
             "The record operators M_c (storage column order) as (x sites, z sites, phase).")
        .def("record_reads", &PyFamilySampler::record_reads,
             "Per measurement record: deferred wire, basis (0 X / 1 Y / 2 Z), invert, readout-flip probability.")
        .def("record_layout", [](const PyFamilySampler& s) {
                 const int n = s.bare_->n();
                 return xtim_api::record_layout_dict(n, (int)s.s_->record_operators().size(), (n + 63) / 64,
                                                     s.s_->record_value_words());
             }, "The layout-3 record key and outcome order (static).")
        .def("plan_structure", &Busy<&PyFamilySampler::plan_structure_of_content>::call, py::arg("plan_content"),
             "plan_structure() (design §4 fields) of a layout-3 plan content (BarrierBuffer.plan_content(i)[0]); "
             "no buffer needed.")
        .def("born_dec", [](const PyFamilySampler& s) { return xtim_api::born_dec_dict(s.s_->born_dec_ops()); },
             "The Born-measured DECISION operators (identical dict to BarrierBuffer.born_dec).")
        // J-b1 final review m6: the Born-trie test oracles (_born_trie_check / _born_trie_oracle) are test-API only
        // (xtim._xtim_testapi.compile_family_sampler(...)._born_trie_check()).
        .def("ideal_state", [](py::object self, bool provenance) {
                 const PyFamilySampler& s = self.cast<const PyFamilySampler&>();
                 const BusyHold h = s.hold_("ideal_state");
                 PyIdealState i;
                 i.fs = s.ideal_fs();
                 i.wires = s.out_wires_;
                 if (provenance) { i.tok = s.own_token_(); i.producer = self; }
                 else { i.tok = frame_token_rows(i.fs); i.producer = py::none(); }
                 return i;
             }, py::arg("provenance") = true,
             "The post-barrier ideal state of this producer's OUTPUT_QUBITS (provenance=True: a producer token, "
             "this sampler kept alive for branch rebuilds; False: a rows-only token, branch 0 only).")
        .def("branch_handles", [](const PyFamilySampler& s, const py::object& buf) {
                 const BusyHold h = s.hold_("branch_handles");
                 return s.branch_handles(*records_arg(buf)); }, py::arg("buffer"),
             "Per record of a BarrierBuffer: its branch handle {token, digest, z} (add_branch's argument).")
        .def("branches", [](const PyFamilySampler& s, const py::object& buf) {
                 const BusyHold h = s.hold_("branches");
                 return s.branches(*records_arg(buf)); }, py::arg("buffer"),
             "(handles, frames uint8[shots, 2, n_out]): per record its branch handle and the shot's Pauli frame "
             "P_rec·W(u) on the OUTPUT_QUBITS (X row, Z row) — the consumer's input_pauli.")
        .def("add_branch", &Busy<&PyFamilySampler::add_branch>::call, py::arg("handle"), py::arg("vector") = py::none(),
             "Register a producer branch handle on this consumer (from_ideal_state samplers); returns its branch id "
             "(input_branch value). Branches outside the carried-branch law are refused by name "
             "(XtimTwirlRefusal LAW_LIMITATION_CARRIED_AXIS).")
        .def("branch_counters", &Busy<&PyFamilySampler::branch_counters>::call)
        .def("branch_table_digest", [](const PyFamilySampler& s) {
                 if (!s.btab_) throw std::invalid_argument("branch_table_digest: not an ideal-state sampler");
                 return PyFamilySampler::dbytes_(s.btab_->digest()); })
#ifdef XTIM_BENCH_REPLAY
        // BENCH-ONLY (-DXTIM_BENCH_REPLAY): the ledger's replay hooks (record a run's fired events, replay them).
        .def("_bench_record", [](PyFamilySampler& s, bool on) { s.s_->bench_record(on); }, py::arg("on"))
        .def("_bench_recorded", [](const PyFamilySampler& s) {
                 const auto& off = s.s_->bench_rec_off();
                 const auto& ev = s.s_->bench_rec_ev();
                 py::array_t<int32_t> o((py::ssize_t)off.size()), e((py::ssize_t)ev.size());
                 std::copy(off.begin(), off.end(), o.mutable_data());
                 std::copy(ev.begin(), ev.end(), e.mutable_data());
                 return py::make_tuple(o, e); })
        .def("_bench_replay", [](PyFamilySampler& s, py::object off, py::object ev) {
                 if (off.is_none()) { s.s_->bench_replay({}, {}); return; }
                 auto o = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(off);
                 auto e = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(ev);
                 s.s_->bench_replay(std::vector<int32_t>(o.data(), o.data() + o.size()),
                                    std::vector<int32_t>(e.data(), e.data() + e.size()));
             }, py::arg("off"), py::arg("ev"))
        .def("_bench_times", [](const PyFamilySampler& s) {
                 return py::make_tuple(s.s_->wall_seconds(), s.s_->draw_seconds()); })
        .def("_bench_fingerprint", [](const PyFamilySampler& s) { return s.s_->bench_fingerprint(); })
        .def("_bench_build_plans", [](PyFamilySampler& s, const py::list& keys) {   // S8-0b t_build (redefined)
                 std::vector<std::string> k;
                 for (const auto& o : keys) k.push_back(o.cast<std::string>());
                 return s.s_->bench_build_plans(k); }, py::arg("keys"))
        .def("_bench_noise_only", [](PyFamilySampler& s, long shots) {
                 long ev = 0; double sec;
                 { py::gil_scoped_release rel; sec = s.s_->bench_noise_only(shots, &ev); }
                 return py::make_tuple(sec, ev); }, py::arg("shots"))
#endif
        ;

    // Task 2b.2 / Task 4: physical carried input state + its OUTPUT_QUBITS wire map (for ⊗-composition).
    // Validation runs in C++ inside py_input_state_of.
    const char* input_state_of_doc =
        "Build the frame-explicit bare state of `circuit_text` AND return the deferred-space "
        "wire indices of its declared OUTPUT_QUBITS (final_wire[orig], declaration order). "
        "Returns (FramedSuperposition, list[int]). The state carries the PHYSICAL carried "
        "qubits when the prep has no trailing measurement on them (frame-baked). Used by "
        "compile_twirl_sampler_from_ideal_state(..., ideal_state=input_state_of(prep)) to locate the carried "
        "qubits in the provided state. "
        "Raises ValueError on parse failure, bare-state rejection, or an entangled / "
        "non-stabilizer-rest port.";
    mdef(m, "input_state_of",  &py_input_state_of, py::arg("circuit_text"), input_state_of_doc);
    mdef(m, "input_qubits", &py_input_qubits, py::arg("text"),
          "Return the flattened INPUT_QUBITS port qubit indices (declaration order). "
          "Empty for a port-free circuit. Raises ValueError on parse failure.");
    mdef(m, "_output_wires_of", &py_output_wires_of, py::arg("text"),
          "Return the deferred-space wire indices of the OUTPUT_QUBITS in `text` WITHOUT "
          "building the bare state and WITHOUT running port-contract validation. "
          "Used by tests that need the wire map for entangled circuits.");

    // port_contract_of_state — public free function for computing the port-verdict.
    const char* _port_contract_docstring =
        "Compute the port-contract verdict for a (state, wires) pair.\n\n"
        "The verdict is a property of a (state, wires) pair, not of a compiled program.\n"
        "For a circuit containing measurements the BARE state is not the carried state,\n"
        "so callers must pass a post-collapse state (e.g. BarrierBuffer.materialize(i)),\n"
        "and the wires are deferred-space indices as returned by TwirlSampler.output_wires().\n\n"
        "Returns a dict with keys: product (bool), k_port (int), "
        "stabilized (list of (q, axis, sign) int tuples; axis 0=X/1=Y/2=Z, sign 0=+/1=-), "
        "port_wires (list of int; deferred-space wire indices), witness (str; empty iff product).";
    mdef(m, "port_contract_of_state", &py_port_contract_of_state,
          py::arg("state"), py::arg("wires"),
          _port_contract_docstring);

    // Task 5: IF branch resolver helpers
    mdef(m, "if_driving_bits", &py_if_driving_bits, py::arg("text"),
          "Return list of (port_index, bit_index) tuples for decision bits that "
          "appear in any IF condition in the circuit. Duplicates deduplicated; "
          "raises ValueError on parse failure.");
    mdef(m, "if_driving_global_offsets", &py_if_driving_global_offsets, py::arg("text"),
          "Return list of global bit offsets (sum of prior Bits port widths + bit_idx) "
          "for IF-driving bits. Order matches if_driving_bits(). Raises ValueError on parse failure.");
    mdef(m, "bits_total_width", &py_bits_total_width, py::arg("text"),
          "Return total width (sum of widths) of all INPUT_BITS ports in the circuit. "
          "This is the required column count for decision_bits arrays. "
          "Raises ValueError on parse failure.");
    mdef(m, "resolve_branches_text", &py_resolve_branches_text,
          py::arg("text"), py::arg("pattern_bytes"),
          "Resolve all IF spans in `text` using `pattern_bytes` (list[int] or bytes, "
          "indexed by global decision-bit offset) and return the branch-free circuit "
          "as a Stim-superset text string suitable for compile_twirl_sampler. "
          "Task 6: rec[-k] references are correctly remapped when measurement-bearing "
          "branches are dropped (pattern_bytes forwarded to emit_resolved_text). "
          "Raises ValueError on parse failure.");
    mdef(m, "detector_branch_spans", &py_detector_branch_spans, py::arg("text"),
          "Task 6: return the IF conditions for each detector in the circuit. "
          "Returns list[list[tuple[int,int]]]: for each detector (in declaration order), "
          "a list of (global_bit_offset, cond_value) pairs that must ALL hold for that "
          "detector to be active.  An empty inner list means always active (not inside any IF). "
          "Used by the Python partition layer to build the superset live-mask. "
          "Raises ValueError on parse failure.");
}
