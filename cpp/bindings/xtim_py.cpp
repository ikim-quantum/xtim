// pybind11 bindings for the xtim Python package (extension module xtim._xtim).
//
// Exposes the MSP simulator's CLI-equivalent entry points (M1 of the xtim Python UX
// plan, docs/superpowers/plans/2026-06-12-xtim-python-ux.md):
//   * parse_info        — parse a strict-Stim-superset circuit, line-numbered errors;
//   * CompiledProgram   — compile a circuit ONCE, sample() it repeatedly -> all four
//                         packed record buffers (Stim b8 layout uint8 buffers + row-major
//                         float64 expectations) as numpy arrays, plus chi / reject info;
//                         optional supplied .ref text;
//   * export_dem_text   — the Stim-compatible DEM text (run_stim_main --dem semantics);
//   * ref_info / ref_verify_cheap / ref_compile_text — the stored-reference surface:
//                         header info, the cheap load+invariants verify, and the FULL
//                         ref_compile gate battery (compile mode) returning .ref text.
//
// Contract notes:
//   * Every function returns a plain dict with an "ok" flag and structured error info;
//     the Python layer (xtim/) converts these into the typed exceptions. No exception
//     types are defined here.
//   * The GIL is released around every engine call (parse/sample/DEM/battery); results
//     are computed into plain C++ structs first, then converted to Python objects.
//   * ref_compile_text runs the SHARED gate battery (refcompile::run_compile_battery,
//     cpp/apps/ref_compile.cpp included with REF_COMPILE_NO_MAIN — the exact machinery
//     the ref_compile app's compile mode self-verifies with, single source of truth).
//   * ENGINE_VERSION participates in the Python refcache key: bump it whenever an
//     engine change could alter the constructed reference or the byte stream.
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"
#include <cstddef>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <memory>
#include <random>
#include <stdexcept>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/deferral.hpp"
#include "qeccore/dem_export.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/port_contract.hpp"
#include "qeccore/ref_io.hpp"
#include "qeccore/resolve_branches.hpp"
#include "qeccore/run_stim.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_ppr.hpp"
#include "qeccore/twirl_sampler.hpp"

namespace py = pybind11;
using namespace qeccore;
using namespace refcompile;  // the shared ref_compile gate battery (REF_COMPILE_NO_MAIN)

namespace {

constexpr const char* kEngineVersion = "xtim-engine-9";
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
constexpr int kRefFormatVersion = 4;  // the .ref on-disk format the engine WRITES (ref_from_state_v4 ->
                                      // "ref-format 4": written FROM the framed state, anchor block
                                      // retired, coefficients as-is — Phase B3). Feeds the reference
                                      // cache key + the ref_format_version metadata; the LOADER
                                      // dispatches on each file's own `version` field, so older v2/v3
                                      // refs still load. NOTE older xtim versions cannot read v4 files.

// ── helpers ──────────────────────────────────────────────────────────────────────────

py::list errors_to_py(const std::vector<ParseError>& es) {
    py::list out;
    for (const auto& e : es) out.append(py::make_tuple(e.line, e.message));
    return out;
}

py::array_t<uint8_t> packed_to_numpy(const std::vector<uint8_t>& v, int shots, int bytes) {
    py::array_t<uint8_t> a({shots, bytes});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size());
    return a;
}

py::array_t<double> exps_to_numpy(const std::vector<double>& v, int shots, int r) {
    py::array_t<double> a({shots, r});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
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

// Forward declaration: the shared per-channel marshaller (defined below).
py::dict records_to_dict(const PackedRecords& rec, bool want_meas, bool want_det,
                         bool want_obs, bool want_exp);

// Marshal a sampled PackedRecords into the dict. The scalar metadata
// (num_measurements/.../shots/ok/rejected) is always populated (cheap, needed by the
// Python layer); each of the four record-buffer channels is materialized into a numpy array
// ONLY when its want_* flag is set. An unrequested channel's key is omitted — the Python
// callers only read a channel when they also requested it (verified in xtim/circuit.py), so
// the RETURNED arrays are byte-identical to building all four. Defaults are all-true, so any
// caller that doesn't pass flags gets the original all-four-channel dict.
py::dict records_to_dict(const PackedRecords& rec, bool want_meas = true, bool want_det = true,
                         bool want_obs = true, bool want_exp = true) {
    py::dict d;
    d["rejected"] = rec.rejected;
    d["reject_gate_index"] = rec.reject_gate_index;
    if (rec.rejected) return d;
    d["shots"] = rec.shots;
    d["num_measurements"] = rec.num_measurements;
    d["num_detectors"] = rec.num_detectors;
    d["num_observables"] = rec.num_observables;
    d["num_expectations"] = rec.num_expectations;
    if (want_meas)
        d["measurements"] = packed_to_numpy(rec.measurements, rec.shots,
                                            PackedRecords::bytes_per_shot(rec.num_measurements));
    if (want_det)
        d["detectors"] = packed_to_numpy(rec.detectors, rec.shots,
                                         PackedRecords::bytes_per_shot(rec.num_detectors));
    if (want_obs)
        d["observables"] = packed_to_numpy(rec.observables, rec.shots,
                                           PackedRecords::bytes_per_shot(rec.num_observables));
    if (want_exp)
        d["expectations"] = exps_to_numpy(rec.expectations, rec.shots, rec.num_expectations);
    return d;
}

// Stateful compiled program: parse the circuit text + build the SamplerProgram ONCE in the
// constructor, then sample() it repeatedly with NO recompile (the ~870ms d5 / ~66ms d3 compile is
// paid once). Holds the compiled program by raw pointer + the engine's out-of-line deleter
// (SamplerProgram is an opaque sampler-internal type).
class CompiledProgram {
public:
    CompiledProgram(const std::string& text, const std::string& ref_text) {
        {
            py::gil_scoped_release rel;
            FramedSuperposition lref(0);
            const FramedSuperposition* reference = nullptr;
            if (!ref_text.empty()) {
                RefParseResult rp = parse_ref_text(ref_text);
                if (!rp.ok) { ref_error_ = rp.error; }
                else {
                    // Validated FRAMED load (framed_from_ref_validated: v3/v4 direct behind the
                    // structural + norm gates; v2 via the fully-validating CSS loader inside the
                    // helper). For v3 text the state is byte-identical to the historical
                    // FramedSuperposition::from_css(ref_to_state(...)) sourcing.
                    LoadedFramedRef lfr = framed_from_ref_validated(rp.file);
                    if (!lfr.ok) ref_error_ = lfr.error;
                    else { lref = std::move(lfr.state); reference = &lref; }
                }
            }
            if (ref_error_.empty()) {
                CompiledStimProgram cp = compile_stim_program(text, reference);
                if (!cp.errors.empty()) {
                    errors_ = std::move(cp.errors);
                } else {
                    prog_.reset(cp.program);   // bare lean state was CLONED from `reference`; lref may drop
                }
            }
        }
    }

    py::dict sample(int shots, uint64_t seed, bool want_meas, bool want_det,
                    bool want_obs, bool want_exp) {
        py::dict d;
        d["ref_error"] = ref_error_;
        if (!ref_error_.empty()) { d["ok"] = false; d["errors"] = py::list(); return d; }
        d["ok"] = errors_.empty();
        d["errors"] = errors_to_py(errors_);
        if (!errors_.empty()) return d;
        PackedRecords rec;
        {
            py::gil_scoped_release rel;
            rec = sample_program(*prog_, shots, seed);
        }
        py::dict r = records_to_dict(rec, want_meas, want_det, want_obs, want_exp);
        for (auto item : r) d[item.first] = item.second;
        return d;
    }

    // attributes computed once at compile time.
    bool ok() const { return ref_error_.empty() && errors_.empty(); }
    std::string ref_error() const { return ref_error_; }
    py::list errors() const { return errors_to_py(errors_); }

private:
    struct ProgDeleter {
        void operator()(SamplerProgram* p) const { if (p) delete_sampler_program(p); }
    };
    std::unique_ptr<SamplerProgram, ProgDeleter> prog_;
    std::vector<ParseError> errors_;
    std::string ref_error_;
};

py::dict py_export_dem_text(const std::string& text,
                            bool include_expectations,
                            bool decompose_errors,
                            bool ignore_decomposition_failures,
                            bool drop_gauge_observables,
                            const std::vector<int>& trusted_detectors,
                            const std::vector<int>& trusted_observables) {
    DemExportResult dr;
    {
        py::gil_scoped_release rel;
        dr = export_dem(text, include_expectations,
                        decompose_errors, ignore_decomposition_failures,
                        drop_gauge_observables,
                        trusted_detectors.empty() ? nullptr : &trusted_detectors,
                        trusted_observables.empty() ? nullptr : &trusted_observables);
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
    return d;
}

// Exact per-detector/observable determinism partition (parity-Pauli Born check on the
// noiseless bare state). Independent of DEM expressibility — answers even when the full DEM
// refuses. Shot-free; the source-of-truth determinism verdict for xtim.diagnose.
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
// frame [(orig_qubit, col_x, col_z), ...], logical_flip [(decision_k, col), ...].
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
    // frame: list of (orig_qubit, col_x, col_z)
    py::list frame_list;
    for (auto& [q, cx, cz] : r.frame)
        frame_list.append(py::make_tuple(q, cx, cz));
    d["frame"] = frame_list;
    // logical_flip: list of (decision_k, col)
    py::list lf_list;
    for (auto& [k2, c] : r.logical_flip)
        lf_list.append(py::make_tuple(k2, c));
    d["logical_flip"] = lf_list;
    return d;
}

py::dict py_ref_info(const std::string& ref_text) {
    RefParseResult rp;
    {
        py::gil_scoped_release rel;
        rp = parse_ref_text(ref_text);
    }
    py::dict d;
    d["ok"] = rp.ok;
    d["error"] = rp.error;
    if (rp.ok) {
        d["version"] = rp.file.version;
        d["source"] = rp.file.source;
        d["n"] = rp.file.n;
        d["chi"] = rp.file.chi;
    }
    return d;
}

// The cache-hit verify: parse + load the ref (fail-loud), n-match vs the circuit's deferred
// space, then the noiseless invariants (gate (d)) at a reduced shot count. The exact
// PHYSICAL-STATE overlap gate (|<ref|reconstruct>| == 1) runs CONDITIONALLY — only when the
// noiseless invariants are VACUOUS.
// SOUNDNESS + PERF: the noiseless-invariants check only inspects DETECTORS + PAULI_EXPECTATION
// magnitudes. A circuit with 0 deterministic detectors AND 0 PAULI_EXPECTATIONs passes it
// VACUOUSLY (nothing to check), so without an extra gate a physically-WRONG cached ref
// (raw-record-corrupting; the MPP record-flip bug) slips through. For THAT vacuous case we
// reconstruct the bare state from the circuit (build_bare_state) and pin |overlap|==1. When the
// circuit DOES carry detectors/expectations the cheap invariants verify it non-vacuously (the
// original accepted design), so we SKIP the costly reconstruction — restoring the reference
// cache's setup-time saving for those circuits. This is reference RESOLUTION (one-time,
// memoized per Circuit), NOT the sampling hot loop.

// Structural signature of the NORMALIZED+DEFERRED circuit — the frame-determining
// content: instruction kinds/gates/targets/bases/inverts, deferred n, and the
// terminal-read map. Noise probability VALUES are deliberately EXCLUDED (matching the
// noise-value-masked reference cache key: every p-point of a sweep shares one
// signature). Python hashes the returned string and stores it beside each cached
// .ref; on a cache hit the signature recomputed from the CALLER's text must match —
// this catches the whole frame-divergence class (two texts whose normalization
// lands in different terminal frames) at normalize cost, without rebuilding the
// bare state per hit.
std::string py_deferred_signature(const std::string& circuit_text) {
    py::gil_scoped_release rel;
    protoref::LoadedBench lb = bench_from_text(circuit_text);
    if (!lb.ok) throw std::runtime_error("deferred_signature: " + lb.error);
    std::string s;
    s.reserve(64 * lb.deferred.stream.size() + 64);
    s += "n=" + std::to_string(lb.deferred.n) + ";";
    for (const Instr& ins : lb.deferred.stream) {
        s += std::to_string((int)ins.kind);
        s += ":";
        switch (ins.kind) {
            case Instr::Kind::Gate:
                s += std::to_string((int)ins.gate);
                for (int t : ins.targets) { s += ","; s += std::to_string(t); }
                break;
            case Instr::Kind::Noise:
                s += std::to_string((int)ins.channel);   // channel + qubits; probs EXCLUDED
                for (int q : ins.qubits) { s += ","; s += std::to_string(q); }
                break;
            case Instr::Kind::Measure:
                s += std::to_string((int)ins.basis);
                s += ins.invert ? "!" : ".";             // flip PROBABILITY excluded
                for (int q : ins.qubits) { s += ","; s += std::to_string(q); }
                break;
            case Instr::Kind::Reset:
                for (int q : ins.qubits) { s += ","; s += std::to_string(q); }
                break;
            case Instr::Kind::Observable:
                for (const PauliTerm& t : ins.obs) {
                    s += ","; s += std::to_string(t.qubit);
                    s += "p"; s += std::to_string((int)t.p);
                }
                for (int r : ins.obs_frame) { s += "f"; s += std::to_string(r); }
                break;
            case Instr::Kind::ControlledPauli:
                s += std::to_string((int)ins.basis);
                s += "r" + std::to_string(ins.control_rec_offset);
                for (int q : ins.qubits) { s += ","; s += std::to_string(q); }
                break;
        }
        s += ";";
    }
    s += "reads=";
    for (const auto& pr : lb.map.terminal_reads) {
        s += std::to_string(pr.first);
        s += "b" + std::to_string((int)pr.second);
        s += ",";
    }
    return s;
}

py::dict py_ref_verify_cheap(const std::string& circuit_text, const std::string& ref_text,
                             uint64_t seed, int shots) {
    bool ok = false;
    std::string error;
    int chi = 0, n = 0;
    {
        py::gil_scoped_release rel;
        do {
            protoref::LoadedBench lb = bench_from_text(circuit_text);
            if (!lb.ok) { error = lb.error; break; }
            RefParseResult rp = parse_ref_text(ref_text);
            if (!rp.ok) { error = "ref parse failed: " + rp.error; break; }
            // The verified state is sourced via the FRAMED route (the state the sampler
            // actually consumes) — framed_from_ref_validated: v3/v4 direct behind the
            // structural + norm gates; v2 keeps the validating CSS loader + from_css
            // fallback inside the same helper.
            LoadedFramedRef lfr = framed_from_ref_validated(rp.file);
            if (!lfr.ok) { error = "ref load failed: " + lfr.error; break; }
            FramedSuperposition lref = std::move(lfr.state);
            chi = lref.chi();
            n = lref.n();
            if (lref.n() != lb.deferred.n) {
                error = "n mismatch: ref n=" + std::to_string(lref.n()) +
                        " != deferred n=" + std::to_string(lb.deferred.n);
                break;
            }
            // PHYSICAL-STATE gate (representation-independent), run ONLY when the cheap
            // noiseless-invariants check below would be VACUOUS. That check only inspects
            // DETECTORS + PAULI_EXPECTATION magnitudes (OBSERVABLE_INCLUDE is NOT a determinism
            // check, so it does NOT count toward non-vacuity); a circuit with no deterministic
            // detectors and no PAULI_EXPECTATIONs passes it vacuously, so a physically-WRONG
            // cached reference (raw-record-corrupting) would slip through and be supplied to the
            // sampler. For THAT case, pin the state itself: exact overlap == 1 vs an INDEPENDENT
            // reconstruction from the circuit alone (the same frontier constructor ref_compile
            // --verify uses). On failure the caller falls back to the deduced bare state (the
            // silent-correct path). When detectors/expectations exist we SKIP this costly
            // reconstruction and rely on the cheap invariants (the original design), restoring
            // the cache's setup-time saving.
            //   Counts off the already-parsed LoadedBench: detectors = lb.detectors.size();
            //   PAULI_EXPECTATIONs are the Kind::Observable instructions in the stream (the only
            //   producer of them — OBSERVABLE_INCLUDE goes to lb.observables, not the stream),
            //   matching the sampler's num_expectations (R) exactly (sampler.cpp ++R per
            //   Kind::Observable).
            size_t num_detectors = lb.detectors.size();
            size_t num_expectations = 0;
            for (const auto& ins : lb.circuit.stream)
                if (ins.kind == Instr::Kind::Observable) ++num_expectations;
            const bool invariants_vacuous = (num_detectors == 0 && num_expectations == 0);
            if (invariants_vacuous) {
                // Independent reconstruction via the FRAMED bare-state build (Phase B2);
                // overlap on the framed pair (magnitude — the reconstructed anchors carry an
                // arbitrary global phase each).
                FramedBareState rec = build_bare_state_framed(lb.deferred);
                if (rec.rejected) {
                    error = "reconstruction failed: reconstruction rejected: " +
                            (rec.reject_reason.empty() ? std::string("unknown cause")
                                                       : rec.reject_reason);
                    break;
                }
                double ov = 0.0;
                try {
                    ov = std::abs(exact_sum_overlap(lref, rec.state));
                } catch (const std::exception& e) {
                    error = std::string("state overlap failed: ") + e.what();
                    break;
                }
                if (!(std::abs(ov - 1.0) < 1e-9)) {
                    error = "state overlap != 1 (|overlap|=" + std::to_string(ov) +
                            "): reference does not match the circuit";
                    break;
                }
            }
            InvariantsResult ir = noiseless_invariants(lb, &lref, seed, shots);
            if (ir.rejected) { error = "noiseless invariants run rejected"; break; }
            if (!ir.ok) { error = "noiseless invariants failed"; break; }
            ok = true;
        } while (false);
    }
    py::dict d;
    d["ok"] = ok;
    d["error"] = error;
    d["chi"] = chi;
    d["n"] = n;
    return d;
}

py::dict py_ref_compile_text(const std::string& circuit_text, const std::string& source,
                             int oracle_chi, uint64_t seed, int shots) {
    CompileBatteryResult r;
    {
        py::gil_scoped_release rel;
        r = run_compile_battery(circuit_text, source, oracle_chi, seed, shots);
    }
    py::dict d;
    d["ok"] = r.ok;
    d["error"] = r.error;
    d["ref_text"] = r.ref_text;
    d["chi"] = r.chi;
    d["n"] = r.n;
    py::dict gates;
    for (const auto& g : r.gates) gates[py::str(g.first)] = g.second;
    d["gates"] = gates;
    return d;
}

// ── PyFramedSuperposition — Python handle for a FramedSuperposition (S2.1/S2.2) ──────────
// S2.1: opaque handle; created by _bare_state_of and passed to TwirlSampler(state,…).
// S2.2 additions: approx_equal (exact_sum_overlap-based, global-phase insensitive) +
//   apply_clifford (for the mutation check: mutate a copy, verify approx_equal fails).
struct PyFramedSuperposition {
    FramedSuperposition state;
    explicit PyFramedSuperposition(const FramedSuperposition& s) : state(s) {}
    explicit PyFramedSuperposition(FramedSuperposition&& s) : state(std::move(s)) {}

    int n() const { return state.n(); }
    int k() const { return state.k(); }

    // |<self|other>| ≈ 1  ⟺  states equal up to global phase.
    bool approx_equal(const PyFramedSuperposition& other, double tol = 1e-12) const {
        auto ov = exact_sum_overlap(state, other.state);
        return std::abs(std::abs(ov) - 1.0) < tol;
    }

    // In-place Clifford gate for mutation testing. kind: 0=H 1=S 2=Sdg 3=X 4=Y 5=Z
    // 6=CX 7=CZ. For single-qubit gates b is ignored (pass 0).
    void apply_clifford_py(uint8_t kind, int a, int b) {
        state.apply_clifford(kind, a, b);
    }

    // Task 2b.1: single-qubit Pauli expectations <X_q> / <Z_q> (real). Wraps the exact
    // FramedSuperposition::pauli_expectation (framed_expectation engine). Used by the Born-
    // decision correlation gate: the retained collapsed state must satisfy <X_0> = ±1.
    double pauli_expectation_x(int q) const {
        Pauli P(state.n()); P.setx(q);
        return state.pauli_expectation(P).real();
    }
    double pauli_expectation_z(int q) const {
        Pauli P(state.n()); P.setz(q);
        return state.pauli_expectation(P).real();
    }
    // Task 2b.2: <Y_q> (real). Y = i·XZ; matches measure_pauli's Y encoding. Used by the
    // ⊗-composition to classify a carried qubit's single-qubit stabilizer state (X/Y/Z axis).
    double pauli_expectation_y(int q) const {
        Pauli P(state.n()); P.setx(q); P.setz(q); P.phase = 1;   // Y = i·XZ
        return state.pauli_expectation(P).real();
    }
    // Diagnostic: <Π_i X_{xs[i]} · Π_j Z_{zs[j]}> (real part). For frame-checking.
    double pauli_expectation_xz(const std::vector<int>& xs, const std::vector<int>& zs) const {
        Pauli P(state.n());
        for (int q : xs) P.setx(q);
        for (int q : zs) P.setz(q);
        return state.pauli_expectation(P).real();
    }
    // decoder-feedback perf (BUCKET FOLD guard, audit §1/§4 "ρ_ports check", COMPLETE / magic-aware):
    // a CANONICAL fingerprint of this state's reduced density matrix on the carried port `wires`.
    // Two states return EQUAL bytes IFF ρ_port is identical (so injecting either representative gives
    // identical stage-k+1 physics — stage k+1 measures ONLY the port). run_protocol folds all record
    // buckets that share one port_signature into ONE compiled law; the fingerprint being COMPLETE is
    // what makes that sound (audit's stabilizer-only sketch was INSUFFICIENT — the Steane T-teleport
    // carries a MAGIC state whose logical frame ⟨Ȳ⟩=±1/√2 varies within a decision pattern; a
    // stabilizer-sign-only guard silently mis-folds it and fails the bp_osd ML-oracle gate at 7σ).
    //
    // The fingerprint has TWO parts:
    //  (1) STABILIZER part — the port-restricted +1 Hermitian stabilizer subgroup with signs, in a
    //      canonical RREF. certified_stabilizers() gives the n−k exact stabilizer generators; a
    //      symplectic Gauss-Jordan over the 2·|non-port| coordinates isolates the port subgroup, and
    //      a second Gauss-Jordan over the 2·|port| coordinates canonicalizes it (all phase algebra via
    //      Pauli::multiply, so each surviving generator's Hermitian sign s∈{±1} is exact).
    //  (2) MAGIC/LOGICAL part — the reduced logical subsystem (k_port = |port|−#stab generators). For
    //      k_port==0 the port is a pure stabilizer state → part (1) is already complete. For k_port==1
    //      the logical qubit's Bloch vector (⟨L_X⟩,⟨L_Z⟩,⟨L_X·L_Z⟩ for a CANONICAL logical pair
    //      derived from the shared stabilizer subgroup — the values are stabilizer-multiple invariant)
    //      pins the pure/mixed logical state exactly. For k_port≥2 the guard cannot cheaply certify a
    //      complete fingerprint, so it emits a FALLBACK flag (leading byte) that run_protocol treats
    //      as "un-foldable" → per-record path (never a silent fold of an un-foldable case).
    py::bytes port_signature(const std::vector<int>& wires) const {
        const int N = state.n();
        std::vector<uint8_t> in_port((size_t)(N > 0 ? N : 1), 0);
        for (int w : wires)
            if (w >= 0 && w < N) in_port[(size_t)w] = 1;

        std::vector<Pauli> rows = state.certified_stabilizers();

        // Gauss-Jordan eliminate a coordinate (qubit q, which: 0=x-bit, 1=z-bit) using rows
        // [pivot_start..): pick a pivot row with that bit set, clear it from every OTHER row via
        // Pauli::multiply (XOR of x/z bits + exact phase), swap the pivot to `pivot_start`, and
        // advance. Returns the new pivot_start (unchanged if no pivot found).
        auto eliminate = [&](int q, int which, int pivot_start) -> int {
            int piv = -1;
            for (int i = pivot_start; i < (int)rows.size(); ++i) {
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) { piv = i; break; }
            }
            if (piv < 0) return pivot_start;
            std::swap(rows[pivot_start], rows[piv]);
            for (int i = 0; i < (int)rows.size(); ++i) {
                if (i == pivot_start) continue;
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) rows[i] = Pauli::multiply(rows[i], rows[pivot_start]);
            }
            return pivot_start + 1;
        };

        // Phase 1: eliminate all non-port coordinates → rows[r..] are supported only on `wires`.
        int r = 0;
        for (int q = 0; q < N; ++q) {
            if (in_port[(size_t)q]) continue;
            r = eliminate(q, 0, r);
            r = eliminate(q, 1, r);
        }
        std::vector<Pauli> port(rows.begin() + r, rows.end());

        // Phase 2: canonical RREF of the port subgroup over the 2·|port| port coordinates.
        rows = std::move(port);
        int pr = 0;
        for (int w : wires) {
            if (w < 0 || w >= N) continue;
            pr = eliminate(w, 0, pr);
            pr = eliminate(w, 1, pr);
        }
        // Keep only the independent (non-identity) port generators.
        std::vector<Pauli> gens(rows.begin(), rows.begin() + pr);
        // Deterministic order: by (leading port coordinate). RREF from the fixed wire scan already
        // yields a unique echelon; sort by the packed port support to remove residual row order.
        auto packed = [&](const Pauli& g) {
            std::vector<uint8_t> key;
            key.reserve(2 * wires.size() + 1);
            for (int w : wires) key.push_back(g.xbit(w) ? 1 : 0);
            for (int w : wires) key.push_back(g.zbit(w) ? 1 : 0);
            // Hermitian sign s = i^{(phase − x·z) mod 4} ∈ {+1,−1} → 0/1.
            int d = ((g.phase - g.xz_overlap()) % 4 + 4) % 4;   // 0 or 2 for a +1 stabilizer
            key.push_back(d == 0 ? 0 : 1);
            return key;
        };
        std::vector<std::vector<uint8_t>> keys;
        keys.reserve(gens.size());
        for (const auto& g : gens) keys.push_back(packed(g));
        std::sort(keys.begin(), keys.end());

        const int W = (int)wires.size();
        const int ns = (int)gens.size();          // # independent port stabilizer generators
        const int k_port = W - ns;                // logical DOF carried on the port

        std::string out;
        out.reserve(keys.size() * (2 * W + 1) + 64);
        // Leading FLAG byte: 0/1 = complete fingerprint (foldable); ≥2 = k_port≥2 fallback marker
        // (run_protocol refuses to fold on it). Then port width + stabilizer-generator count so
        // different-width / different-rank states never collide.
        out.push_back((char)(k_port >= 2 ? 2 : k_port));
        // W header is a single byte.  W>64 already short-circuits to flag=2 above
        // so W>255 is unreachable today; if the fallback threshold is ever raised
        // past 255 this header must widen or port_signature_struct will misalign.
        out.push_back((char)(W & 0xFF));
        out.push_back((char)(ns & 0xFF));
        for (const auto& k : keys)
            out.append(reinterpret_cast<const char*>(k.data()), k.size());
        if (k_port == 0)
            return py::bytes(out);                 // port fully stabilized — stabilizer part is complete
        if (k_port >= 2 || W > 64)
            return py::bytes(out);                 // fallback: stabilizer part only + flag=2 (unfoldable)

        // ── MAGIC/LOGICAL part (k_port == 1): append the carried logical qubit's Bloch vector ──
        // A valid logical is a port Pauli in the NORMALIZER (commuting with every stabilizer) but not
        // in the stabilizer group.  We compute the normalizer as the GF(2) null space of the
        // symplectic-constraint matrix and pick the two independent (anticommuting) logicals.  ⟨L⟩ is
        // invariant under multiplying L by any +1 stabilizer, so the Bloch coordinates are canonical
        // given the (shared) stabilizer subgroup — comparable across buckets without a canonical L.
        // Symplectic coordinate: 2W bits, low W = x over wires, high W = z over wires.
        using u128 = unsigned __int128;
        auto to_sv = [&](const Pauli& g) -> u128 {         // Pauli → port symplectic vector
            u128 v = 0;
            for (int i = 0; i < W; ++i) {
                if (g.xbit(wires[i])) v |= (u128)1 << i;
                if (g.zbit(wires[i])) v |= (u128)1 << (W + i);
            }
            return v;
        };
        auto sp = [&](u128 a, u128 b) -> int {             // symplectic product a_x·b_z + a_z·b_x
            u128 axbz = (a & (((u128)1 << W) - 1)) & (b >> W);
            u128 azbx = (a >> W) & (b & (((u128)1 << W) - 1));
            u128 t = axbz ^ azbx;
            return (__builtin_popcountll((uint64_t)t) + __builtin_popcountll((uint64_t)(t >> 64))) & 1;
        };
        std::vector<u128> S;                               // stabilizer symplectic vectors
        S.reserve(gens.size());
        for (const auto& g : gens) S.push_back(to_sv(g));
        // Constraint rows: A_i · v = sp(v, S_i).  A_i = (S_i.z ‖ S_i.x) (swap halves) so that dotting
        // with v = (v_x ‖ v_z) gives v_x·S_i.z + v_z·S_i.x = sp(v,S_i).
        std::vector<u128> A;
        A.reserve(S.size());
        for (u128 s : S) {
            u128 lo = s & (((u128)1 << W) - 1), hi = s >> W;
            A.push_back(hi | (lo << W));
        }
        // Row-reduce A to echelon; record pivot columns.
        std::vector<int> pivcol;
        std::vector<u128> pivrow;
        for (u128 row : A) {
            for (size_t p = 0; p < pivrow.size(); ++p)
                if ((row >> pivcol[p]) & 1) row ^= pivrow[p];
            if (row == 0) continue;
            int lead = 0;
            for (int c = 0; c < 2 * W; ++c) if ((row >> c) & 1) { lead = c; break; }
            // clear this column from previous pivot rows (reduced echelon)
            for (size_t p = 0; p < pivrow.size(); ++p)
                if ((pivrow[p] >> lead) & 1) pivrow[p] ^= row;
            pivrow.push_back(row);
            pivcol.push_back(lead);
        }
        std::vector<char> is_piv(2 * W, 0);
        for (int c : pivcol) is_piv[c] = 1;
        // Null-space basis: one vector per free column (bit set there + back-substituted pivots).
        std::vector<u128> nullb;
        for (int fc = 0; fc < 2 * W; ++fc) {
            if (is_piv[fc]) continue;
            u128 v = (u128)1 << fc;
            for (size_t p = 0; p < pivrow.size(); ++p)
                if ((pivrow[p] >> fc) & 1) v |= (u128)1 << pivcol[p];
            nullb.push_back(v);
        }
        // Logical operators = null-space vectors NOT in the stabilizer span.  Reduce each by S; keep
        // the nonzero residues; take the first two independent (anticommuting) as L_a, L_b.
        // Build a reduced basis of S (for span membership tests).
        std::vector<u128> Sred, Spiv;
        for (u128 s : S) {
            int lead = -1; u128 r = s;
            for (size_t p = 0; p < Sred.size(); ++p) {
                int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Spiv[p] >> c) & 1) { lp = c; break; }
                if ((r >> lp) & 1) r ^= Sred[p];
            }
            if (r) { Sred.push_back(r); Spiv.push_back(r); }
        }
        auto in_span_S = [&](u128 v) -> bool {
            for (size_t p = 0; p < Sred.size(); ++p) {
                int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Sred[p] >> c) & 1) { lp = c; break; }
                if ((v >> lp) & 1) v ^= Sred[p];
            }
            return v == 0;
        };
        u128 La = 0, Lb = 0; bool haveA = false, haveB = false;
        for (u128 v : nullb) {
            if (in_span_S(v)) continue;
            if (!haveA) { La = v; haveA = true; }
            else if (sp(v, La)) { Lb = v; haveB = true; break; }
        }
        auto sv_to_pauli = [&](u128 v) -> Pauli {
            Pauli P(N);
            for (int i = 0; i < W; ++i) {
                if ((v >> i) & 1) P.setx(wires[i]);
                if ((v >> (W + i)) & 1) P.setz(wires[i]);
            }
            return P;
        };
        auto herm_exp = [&](u128 v) -> long long {
            // Hermitian expectation ⟨i^{x·z} X^x Z^z⟩ ∈ [−1,1]; quantize to 1e-9 (exact-value stable,
            // distinguishes ±1/√2 cleanly, never merges genuinely-different reduced states).
            Pauli H = sv_to_pauli(v);
            H.phase = H.xz_overlap() & 3;
            double e = state.pauli_expectation(H).real();
            return llround(e * 1e9);
        };
        auto append_i64 = [&](long long v) {
            for (int b = 0; b < 8; ++b) out.push_back((char)((v >> (8 * b)) & 0xFF));
        };
        if (haveA && haveB) {
            append_i64(herm_exp(La));
            append_i64(herm_exp(Lb));
            // The Bloch third coordinate: ⟨L_a·L_b⟩ (Hermitian) — its GF(2) support is La^Lb.
            append_i64(herm_exp(La ^ Lb));
        } else {
            out[0] = (char)2;                      // degenerate — be conservative (unfoldable)
        }
        return py::bytes(out);
    }

    // Task 6: structured accessor — one owner of the wire format.
    // Calls port_signature(wires) to get the canonical byte string and decodes it using
    // EXACTLY the same logic as fold.py's _parse_port_signature.  No second implementation
    // of Gauss-Jordan or Bloch derivation — the byte string is the single source of truth.
    //
    // Layout mirror of _parse_port_signature(b):
    //   b[0] = flag, b[1] = W, b[2] = ns
    //   then ns entries each (2*W + 1) bytes: first 2*W = x/z support; last 1 = Hermitian sign
    //   then remainder = little-endian signed int64s (Bloch coords)
    //     each v mapped to (abs(v), 1 if v < 0 else 0)
    //
    // Returns dict { flag:int, W:int, ns:int,
    //                keys:dict{bytes->int},
    //                blochs:list[list[int,int]] }
    py::dict port_signature_struct(const std::vector<int>& wires) const {
        // Delegate to the single authoritative emitter.
        py::bytes raw = port_signature(wires);
        std::string s = raw.cast<std::string>();
        const uint8_t* b = reinterpret_cast<const uint8_t*>(s.data());
        const size_t blen = s.size();

        int flag = (blen > 0) ? (int)b[0] : 0;
        int W    = (blen > 1) ? (int)b[1] : 0;
        int ns   = (blen > 2) ? (int)b[2] : 0;

        const int ksz = 2 * W + 1;
        size_t off = 3;

        // keys: dict mapping 2W-byte support bytes -> Hermitian sign int
        py::dict keys;
        for (int i = 0; i < ns; ++i) {
            if (off + (size_t)ksz > blen) break;
            py::bytes support(reinterpret_cast<const char*>(b + off), 2 * W);
            int sign = (int)b[off + 2 * W];
            keys[support] = sign;
            off += (size_t)ksz;
        }

        // blochs: list of [magnitude, sign] pairs (signed int64, little-endian)
        py::list blochs;
        const size_t tail_len = (blen > off) ? (blen - off) : 0;
        const size_t n_i64 = tail_len / 8;
        for (size_t j = 0; j < n_i64; ++j) {
            int64_t v = 0;
            for (int byte = 0; byte < 8; ++byte)
                v |= (int64_t)b[off + 8 * j + (size_t)byte] << (8 * byte);
            int64_t mag  = (v < 0) ? -v : v;
            int     sgn  = (v < 0) ? 1 : 0;
            py::list pair;
            pair.append((long long)mag);
            pair.append(sgn);
            blochs.append(pair);
        }

        py::dict res;
        res["flag"]   = flag;
        res["W"]      = W;
        res["ns"]     = ns;
        res["keys"]   = keys;
        res["blochs"] = blochs;
        return res;
    }

    // decoder-feedback perf (PAULI-ORBIT FOLD companion to port_signature): the CANONICAL port
    // symplectic OPERATORS whose signs the fingerprint records — the port stabilizer generators (in
    // the SAME sorted order as port_signature's stabilizer supports) plus the carried logical pair
    // (La, Lb) in the SAME order as the Bloch coordinates (⟨La⟩, ⟨Lb⟩, ⟨La·Lb⟩).  These are a PURE
    // FUNCTION of the stabilizer SUPPORT structure (signs never enter the canonicalization or the
    // null-space logical selection), so they are IDENTICAL for every state in one sign-stripped ORBIT
    // class — the constant GF(2) solve matrix run_protocol uses to recover, per shot, the residual
    // port-Pauli P_i whose anticommutation reproduces that shot's sign vector vs the orbit rep.  A
    // Pauli P anticommutes with (and hence flips the sign of) operator O iff sp(P,O)=1, so
    //     M · P = Δsigns   with M's rows = [stab_1 … stab_ns, La, Lb] symplectic vectors
    // and columns the 2W port-Pauli coordinates (low W = X over wires, high W = Z over wires) — the
    // SAME coordinate convention as input_pauli's frame slice.  Returns ``ok=False`` for the
    // un-foldable / degenerate cases (flag≥2, W>64) so the caller falls back to the per-record path.
    py::dict port_orbit_operators(const std::vector<int>& wires) const {
        py::dict res;
        const int N = state.n();
        const int W = (int)wires.size();
        res["W"] = W;
        if (W > 64) { res["ok"] = false; return res; }
        std::vector<uint8_t> in_port((size_t)(N > 0 ? N : 1), 0);
        for (int w : wires)
            if (w >= 0 && w < N) in_port[(size_t)w] = 1;

        std::vector<Pauli> rows = state.certified_stabilizers();
        auto eliminate = [&](int q, int which, int pivot_start) -> int {
            int piv = -1;
            for (int i = pivot_start; i < (int)rows.size(); ++i) {
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) { piv = i; break; }
            }
            if (piv < 0) return pivot_start;
            std::swap(rows[pivot_start], rows[piv]);
            for (int i = 0; i < (int)rows.size(); ++i) {
                if (i == pivot_start) continue;
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) rows[i] = Pauli::multiply(rows[i], rows[pivot_start]);
            }
            return pivot_start + 1;
        };
        int r = 0;
        for (int q = 0; q < N; ++q) {
            if (in_port[(size_t)q]) continue;
            r = eliminate(q, 0, r);
            r = eliminate(q, 1, r);
        }
        std::vector<Pauli> port(rows.begin() + r, rows.end());
        rows = std::move(port);
        int pr = 0;
        for (int w : wires) {
            if (w < 0 || w >= N) continue;
            pr = eliminate(w, 0, pr);
            pr = eliminate(w, 1, pr);
        }
        std::vector<Pauli> gens(rows.begin(), rows.begin() + pr);
        // Emit the stabilizer symplectic vectors in the SAME sorted key order port_signature uses
        // (key = [x over wires ‖ z over wires ‖ sign]; sort strips residual row order).  We sort by
        // the (support-only) 2W-bit key — distinct stabilizers have distinct support so the sign byte
        // is never a tiebreaker (matches port_signature's std::sort on the full key).
        auto sv_bits = [&](const Pauli& g) {
            std::vector<uint8_t> v; v.reserve(2 * W);
            for (int w : wires) v.push_back(g.xbit(w) ? 1 : 0);
            for (int w : wires) v.push_back(g.zbit(w) ? 1 : 0);
            return v;
        };
        std::vector<std::vector<uint8_t>> stab_sv;
        stab_sv.reserve(gens.size());
        for (const auto& g : gens) stab_sv.push_back(sv_bits(g));
        std::sort(stab_sv.begin(), stab_sv.end());

        const int ns = (int)gens.size();
        const int k_port = W - ns;
        res["ns"] = ns;
        res["k_port"] = (k_port >= 2 ? 2 : k_port);
        if (k_port >= 2) { res["ok"] = false; return res; }

        py::list stab_list;
        for (auto& v : stab_sv) {
            py::array_t<uint8_t> a((py::ssize_t)v.size());
            std::memcpy(a.mutable_data(), v.data(), v.size());
            stab_list.append(a);
        }
        res["stab"] = stab_list;

        py::list log_list;
        if (k_port == 1) {
            // Recompute (La, Lb) EXACTLY as port_signature does — a pure function of the supports.
            using u128 = unsigned __int128;
            auto to_sv = [&](const Pauli& g) -> u128 {
                u128 v = 0;
                for (int i = 0; i < W; ++i) {
                    if (g.xbit(wires[i])) v |= (u128)1 << i;
                    if (g.zbit(wires[i])) v |= (u128)1 << (W + i);
                }
                return v;
            };
            auto sp = [&](u128 a, u128 b) -> int {
                u128 axbz = (a & (((u128)1 << W) - 1)) & (b >> W);
                u128 azbx = (a >> W) & (b & (((u128)1 << W) - 1));
                u128 t = axbz ^ azbx;
                return (__builtin_popcountll((uint64_t)t) + __builtin_popcountll((uint64_t)(t >> 64))) & 1;
            };
            std::vector<u128> S;
            for (const auto& g : gens) S.push_back(to_sv(g));
            std::vector<u128> A;
            for (u128 s : S) {
                u128 lo = s & (((u128)1 << W) - 1), hi = s >> W;
                A.push_back(hi | (lo << W));
            }
            std::vector<int> pivcol; std::vector<u128> pivrow;
            for (u128 row : A) {
                for (size_t p = 0; p < pivrow.size(); ++p)
                    if ((row >> pivcol[p]) & 1) row ^= pivrow[p];
                if (row == 0) continue;
                int lead = 0;
                for (int c = 0; c < 2 * W; ++c) if ((row >> c) & 1) { lead = c; break; }
                for (size_t p = 0; p < pivrow.size(); ++p)
                    if ((pivrow[p] >> lead) & 1) pivrow[p] ^= row;
                pivrow.push_back(row); pivcol.push_back(lead);
            }
            std::vector<char> is_piv(2 * W, 0);
            for (int c : pivcol) is_piv[c] = 1;
            std::vector<u128> nullb;
            for (int fc = 0; fc < 2 * W; ++fc) {
                if (is_piv[fc]) continue;
                u128 v = (u128)1 << fc;
                for (size_t p = 0; p < pivrow.size(); ++p)
                    if ((pivrow[p] >> fc) & 1) v |= (u128)1 << pivcol[p];
                nullb.push_back(v);
            }
            std::vector<u128> Sred, Spiv;
            for (u128 s : S) {
                u128 rr = s;
                for (size_t p = 0; p < Sred.size(); ++p) {
                    int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Spiv[p] >> c) & 1) { lp = c; break; }
                    if ((rr >> lp) & 1) rr ^= Sred[p];
                }
                if (rr) { Sred.push_back(rr); Spiv.push_back(rr); }
            }
            auto in_span_S = [&](u128 v) -> bool {
                for (size_t p = 0; p < Sred.size(); ++p) {
                    int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Sred[p] >> c) & 1) { lp = c; break; }
                    if ((v >> lp) & 1) v ^= Sred[p];
                }
                return v == 0;
            };
            u128 La = 0, Lb = 0; bool haveA = false, haveB = false;
            for (u128 v : nullb) {
                if (in_span_S(v)) continue;
                if (!haveA) { La = v; haveA = true; }
                else if (sp(v, La)) { Lb = v; haveB = true; break; }
            }
            if (!(haveA && haveB)) { res["ok"] = false; return res; }
            auto emit = [&](u128 v) {
                py::array_t<uint8_t> a((py::ssize_t)(2 * W));
                uint8_t* d = a.mutable_data();
                for (int i = 0; i < 2 * W; ++i) d[i] = (uint8_t)((v >> i) & 1);
                log_list.append(a);
            };
            emit(La); emit(Lb);
        }
        res["logicals"] = log_list;
        res["ok"] = true;
        return res;
    }

    // E2 (exact-residual arc, Task 2): COMBINATION-TRACKED port decomposition — the σ-LAW
    // companion to port_signature / port_orbit_operators.  Runs the SAME two-phase symplectic
    // Gauss-Jordan over certified_stabilizers() as those methods (deliberately duplicated so the
    // existing methods stay byte-untouched), but ALSO tracks, for every row, its GF(2) combination
    // over the ORIGINAL certified generators: combo[i] starts as e_i and is XORed alongside every
    // Pauli::multiply — pure bookkeeping, no new math.  After the two phases, row j's combination
    // c_j satisfies   port_stab_j = i-phase · Π_{b : c_j[b]=1} certified_stabilizers()[b],
    // so on any collapsed shot with certified-generator syndrome σ (BarrierBuffer.sigmas() row):
    //     sign_j(shot) = ref_signs[j] ⊕ parity(c_j & σ)          (exact, GF(2))
    // — the i-phases of the product are shot-independent and are folded into ref_signs[j]
    // (the Hermitian sign of the tracked product on THIS state; 0 on a bare state whose
    // certified generators all have +1 eigenvalue).  Returns dict:
    //   W, ngens, gw       — port width, #certified generators, ceil(ngens/64)
    //   ns, k_port         — #surviving port stabilizer generators, W − ns (UNCAPPED, unlike
    //                        port_orbit_operators' clamped field)
    //   stab      uint8[ns, 2W]   — port symplectic supports [x over wires ‖ z over wires], in
    //                        the IDENTICAL sorted order port_signature uses (supports are
    //                        distinct so the sign byte never tiebreaks — same argument as
    //                        port_orbit_operators)
    //   comb      uint64[ns, gw]  — c_j rows, packed in the σ WORD LAYOUT: bit b (word b>>6,
    //                        bit b&63) ↔ certified_stabilizers()[b] ↔ σ bit b of sigmas()
    //   ref_signs uint8[ns]       — Hermitian sign bit of port stab j on THIS state (0 = +1)
    //   logicals  list            — the carried logical pair [La, Lb] (2W-uint8 symplectic rows)
    //                        for k_port==1, delegated to port_orbit_operators (the single owner
    //                        of the M·P-law operator selection, see the doc at its definition):
    //                        the σ-law defines NO combination over certified generators for them
    //                        (La, Lb are out of span(certified) by construction — their signs are
    //                        per-plan data, E3/Task-3 scope).  Empty for k_port==0 / unavailable.
    //   ok        bool            — logical-part completeness flag, identical to
    //                        port_orbit_operators' (False for W>64 / k_port≥2 / degenerate pair).
    //                        The stabilizer part (stab/comb/ref_signs) is valid regardless.
    py::dict port_sigma_law(const std::vector<int>& wires) const {
        py::dict res;
        const int N = state.n();
        const int W = (int)wires.size();
        res["W"] = W;
        std::vector<uint8_t> in_port((size_t)(N > 0 ? N : 1), 0);
        for (int w : wires)
            if (w >= 0 && w < N) in_port[(size_t)w] = 1;

        std::vector<Pauli> rows = state.certified_stabilizers();
        const int ngens = (int)rows.size();
        const int GW = (ngens + 63) / 64;            // words per combination row (σ word layout)
        res["ngens"] = ngens;
        res["gw"] = GW;

        // Combination tracking: combo[i] = e_i, packed uint64 words, bit b ↔ certified gen b.
        std::vector<std::vector<uint64_t>> combo(
            (size_t)ngens, std::vector<uint64_t>((size_t)(GW > 0 ? GW : 1), 0));
        for (int i = 0; i < ngens; ++i)
            combo[(size_t)i][(size_t)(i >> 6)] = (uint64_t)1 << (i & 63);

        // SAME eliminate as port_signature/port_orbit_operators, plus the parallel combo XOR/swap.
        auto eliminate = [&](int q, int which, int pivot_start) -> int {
            int piv = -1;
            for (int i = pivot_start; i < (int)rows.size(); ++i) {
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) { piv = i; break; }
            }
            if (piv < 0) return pivot_start;
            std::swap(rows[pivot_start], rows[piv]);
            std::swap(combo[pivot_start], combo[piv]);
            for (int i = 0; i < (int)rows.size(); ++i) {
                if (i == pivot_start) continue;
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) {
                    rows[i] = Pauli::multiply(rows[i], rows[pivot_start]);
                    for (int wd = 0; wd < GW; ++wd) combo[i][wd] ^= combo[pivot_start][wd];
                }
            }
            return pivot_start + 1;
        };
        int r = 0;
        for (int q = 0; q < N; ++q) {
            if (in_port[(size_t)q]) continue;
            r = eliminate(q, 0, r);
            r = eliminate(q, 1, r);
        }
        // Carry rows AND their combinations into Phase 2 together.
        std::vector<Pauli> port(rows.begin() + r, rows.end());
        std::vector<std::vector<uint64_t>> pcombo(combo.begin() + r, combo.end());
        rows = std::move(port);
        combo = std::move(pcombo);
        int pr = 0;
        for (int w : wires) {
            if (w < 0 || w >= N) continue;
            pr = eliminate(w, 0, pr);
            pr = eliminate(w, 1, pr);
        }
        const int ns = pr;
        const int k_port = W - ns;
        res["ns"] = ns;
        res["k_port"] = k_port;

        // Sort by the packed port support — the IDENTICAL order port_signature's full-key sort
        // yields (supports of independent RREF rows are distinct, so the trailing sign byte of
        // port_signature's key never acts as a tiebreaker).
        auto sv_bits = [&](const Pauli& g) {
            std::vector<uint8_t> v; v.reserve(2 * W);
            for (int w : wires) v.push_back(g.xbit(w) ? 1 : 0);
            for (int w : wires) v.push_back(g.zbit(w) ? 1 : 0);
            return v;
        };
        std::vector<std::vector<uint8_t>> keys;
        keys.reserve((size_t)ns);
        for (int j = 0; j < ns; ++j) keys.push_back(sv_bits(rows[j]));
        std::vector<int> order((size_t)ns);
        for (int j = 0; j < ns; ++j) order[(size_t)j] = j;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return keys[(size_t)a] < keys[(size_t)b]; });

        py::array_t<uint8_t> stab_arr({(py::ssize_t)ns, (py::ssize_t)(2 * W)});
        py::array_t<uint64_t> comb_arr({(py::ssize_t)ns, (py::ssize_t)GW});
        py::array_t<uint8_t> ref_arr((py::ssize_t)ns);
        {
            uint8_t* sd = stab_arr.mutable_data();
            uint64_t* cd = comb_arr.mutable_data();
            uint8_t* rd = ref_arr.mutable_data();
            for (int j = 0; j < ns; ++j) {
                const int src = order[(size_t)j];
                if (W > 0)
                    std::memcpy(sd + (size_t)j * (size_t)(2 * W), keys[(size_t)src].data(),
                                (size_t)(2 * W));
                if (GW > 0)
                    std::memcpy(cd + (size_t)j * (size_t)GW, combo[(size_t)src].data(),
                                (size_t)GW * sizeof(uint64_t));
                // Hermitian sign s = i^{(phase − x·z) mod 4} ∈ {+1,−1} → 0/1 (same formula as
                // port_signature's packed key sign byte).
                const Pauli& g = rows[(size_t)src];
                int d = ((g.phase - g.xz_overlap()) % 4 + 4) % 4;
                rd[j] = (uint8_t)(d == 0 ? 0 : 1);
            }
        }
        res["stab"] = stab_arr;
        res["comb"] = comb_arr;
        res["ref_signs"] = ref_arr;

        // Logical pair rows per the M·P law: delegate to port_orbit_operators — the single owner
        // of the (La, Lb) selection — so the emitted rows are identical by construction.
        py::dict orbit = port_orbit_operators(wires);
        if (orbit.contains("ns") && orbit["ns"].cast<int>() != ns)
            throw std::runtime_error("port_sigma_law: internal GJ disagreement with "
                                     "port_orbit_operators (ns mismatch)");
        res["logicals"] = orbit.contains("logicals") ? orbit["logicals"]
                                                     : py::object(py::list());
        res["ok"] = orbit["ok"];
        return res;
    }

    // E3 (exact-residual arc, Task 3): PURE-DATA exposure of the state's certified stabilizer
    // generators as full n-qubit symplectic rows + exact phases.  Row b is
    // certified_stabilizers()[b] — the IDENTICAL order as the σ bit order of
    // BarrierBuffer.sigmas() (twirl_sampler.hpp "parallel to bare.certified_stabilizers()") and
    // as port_sigma_law's comb bit order.  Coordinate convention matches port_orbit_operators /
    // port_sigma_law rows, widened to all n qubits: low n = x bits, high n = z bits.  The
    // generator OPERATOR is  i^phase · X^x · Z^z  (Pauli rep, phase ∈ {0..3}); on the bare state
    // every row has +1 eigenvalue.  Consumers: the adaptq per-(check, plan) indefiniteness
    // classifier (GF(2) span basis + exact Pauli-product phase bookkeeping).  Returns dict:
    //   n:int, ngens:int, xz: uint8[ngens, 2n], phase: uint8[ngens]
    py::dict certified_symplectic() const {
        const int N = state.n();
        const std::vector<Pauli> gens = state.certified_stabilizers();
        const int ngens = (int)gens.size();
        py::array_t<uint8_t> xz({(py::ssize_t)ngens, (py::ssize_t)(2 * N)});
        py::array_t<uint8_t> ph((py::ssize_t)ngens);
        uint8_t* xd = xz.mutable_data();
        uint8_t* pd = ph.mutable_data();
        for (int b = 0; b < ngens; ++b) {
            const Pauli& g = gens[(size_t)b];
            uint8_t* row = xd + (size_t)b * (size_t)(2 * N);
            for (int q = 0; q < N; ++q) {
                row[q] = g.xbit(q) ? 1 : 0;
                row[N + q] = g.zbit(q) ? 1 : 0;
            }
            pd[b] = (uint8_t)(((g.phase % 4) + 4) % 4);
        }
        py::dict res;
        res["n"] = N;
        res["ngens"] = ngens;
        res["xz"] = xz;
        res["phase"] = ph;
        return res;
    }

    // Task 2b.2-pre: Born-measure a single-qubit Pauli (basis 0:X 1:Y 2:Z) on qubit q with
    // random u∈[0,1); returns ±1 and COLLAPSES the state (mutates in place). This is the exact
    // per-shot decision-operator read the engine performs internally (emit_decisions) exposed
    // for the frame-explicit acceptance oracle: on the physical bare, measuring the decision
    // operator collapses the survivors physically.
    int measure_pauli(int basis, int q, double u) {
        Pauli P(state.n());
        if (basis == 0) P.setx(q);
        else if (basis == 2) P.setz(q);
        else { P.setx(q); P.setz(q); P.phase = 1; }   // Y = i·XZ
        return state.measure_pauli(P, u);
    }
};

// ── port-v3 T2: shared dict builders for the σ-law/plan-structure read model ───────────────
// One owner of the plan_structure / born_dec dict formats: PyBarrierBuffer (the historical
// surface) and PyTwirlSampler (the port's Segment-level read model, no buffer required)
// delegate to these. PURE refactor of the former PyBarrierBuffer method bodies — the emitted
// dicts are byte-identical.
static py::dict plan_structure_dict(const TwirlRecordSampler& sampler, int ngens,
                                    const std::string& pk) {
    const TwirlRecordSampler::PlanStructure ps = sampler.plan_structure(
        pk.empty() ? nullptr : reinterpret_cast<const uint8_t*>(pk.data()), (int)pk.size());
    const int N = (int)ps.nf.a.size();
    const ShotLaw& law = ps.law;
    const int GW = (int)law.det_signs.size();
    const int r = law.r, kappa = law.kappa;

    py::dict res;
    res["n"] = N;
    res["ngens"] = ngens;
    res["gw"] = GW;
    res["r"] = r;
    res["kappa"] = kappa;
    res["fallback"] = law.fallback;

    py::array_t<uint8_t> pxz((py::ssize_t)(2 * N));
    py::array_t<uint8_t> amask((py::ssize_t)N);
    {
        uint8_t* pd = pxz.mutable_data();
        uint8_t* ad = amask.mutable_data();
        for (int q = 0; q < N; ++q) {
            pd[q] = ps.nf.prefix.xbit(q) ? 1 : 0;
            pd[N + q] = ps.nf.prefix.zbit(q) ? 1 : 0;
            ad[q] = ps.nf.a[(size_t)q] ? 1 : 0;
        }
    }
    res["prefix_xz"] = pxz;
    res["a"] = amask;
    py::array_t<int32_t> cz({(py::ssize_t)ps.nf.cz.size(), (py::ssize_t)2});
    {
        int32_t* cd = cz.mutable_data();
        for (size_t i = 0; i < ps.nf.cz.size(); ++i) {
            cd[2 * i] = (int32_t)ps.nf.cz[i].first;
            cd[2 * i + 1] = (int32_t)ps.nf.cz[i].second;
        }
    }
    res["cz"] = cz;

    auto words_arr = [&](const std::vector<uint64_t>& w) {
        py::array_t<uint64_t> a((py::ssize_t)w.size());
        if (!w.empty())
            std::memcpy(a.mutable_data(), w.data(), w.size() * sizeof(uint64_t));
        return a;
    };
    auto words_mat = [&](const std::vector<std::vector<uint64_t>>& m) {
        py::array_t<uint64_t> a({(py::ssize_t)m.size(), (py::ssize_t)GW});
        uint64_t* d = a.mutable_data();
        for (size_t i = 0; i < m.size(); ++i)
            for (int w = 0; w < GW; ++w)
                d[i * (size_t)GW + (size_t)w] =
                    (w < (int)m[i].size()) ? m[i][(size_t)w] : 0;
        return a;
    };
    auto bytes_arr = [&](const std::vector<uint8_t>& v) {
        py::array_t<uint8_t> a((py::ssize_t)v.size());
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size());
        return a;
    };
    res["det_signs"] = words_arr(law.det_signs);
    res["coin_masks"] = words_mat(law.coin_masks);
    // NOTE (T3 review Minor): an UNFOLDABLE kernel direction has an EMPTY engine-side mask,
    // zero-padded here to a gw-word row — consumers must consult kernel_foldable[j] before
    // treating row j as a valid (all-zero) σ-flip mask.
    res["kernel_masks"] = words_mat(law.kernel_masks);
    res["kernel_base"] = bytes_arr(law.kernel_base);
    res["kernel_foldable"] = bytes_arr(law.kernel_foldable);
    py::array_t<uint8_t> kxz({(py::ssize_t)kappa, (py::ssize_t)(2 * N)});
    py::array_t<uint8_t> kph((py::ssize_t)kappa);
    {
        uint8_t* kd = kxz.mutable_data();
        uint8_t* pd = kph.mutable_data();
        for (int j = 0; j < kappa; ++j) {
            const Pauli& L = law.kernel_logicals[(size_t)j];
            uint8_t* row = kd + (size_t)j * (size_t)(2 * N);
            for (int q = 0; q < N; ++q) {
                row[q] = L.xbit(q) ? 1 : 0;
                row[N + q] = L.zbit(q) ? 1 : 0;
            }
            pd[j] = (uint8_t)(((L.phase % 4) + 4) % 4);
        }
    }
    res["kernel_xz"] = kxz;
    res["kernel_phase"] = kph;
    return res;
}

static py::dict born_dec_dict(const TwirlRecordSampler& sampler) {
    const std::vector<TwirlRecordSampler::BornDecOp> ops = sampler.born_dec_ops();
    const int nb = (int)ops.size();
    const int N = nb ? ops[0].op.n : 0;
    py::array_t<int32_t> idx((py::ssize_t)nb);
    py::array_t<uint8_t> xz({(py::ssize_t)nb, (py::ssize_t)(2 * N)});
    py::array_t<uint8_t> ph((py::ssize_t)nb);
    py::array_t<uint8_t> inv((py::ssize_t)nb);
    int32_t* id = idx.mutable_data();
    uint8_t* xd = xz.mutable_data();
    uint8_t* pd = ph.mutable_data();
    uint8_t* vd = inv.mutable_data();
    for (int j = 0; j < nb; ++j) {
        const TwirlRecordSampler::BornDecOp& b = ops[(size_t)j];
        id[j] = (int32_t)b.dec_index;
        uint8_t* row = xd + (size_t)j * (size_t)(2 * N);
        for (int q = 0; q < N; ++q) {
            row[q] = b.op.xbit(q) ? 1 : 0;
            row[N + q] = b.op.zbit(q) ? 1 : 0;
        }
        pd[j] = (uint8_t)(((b.op.phase % 4) + 4) % 4);
        vd[j] = b.inv;
    }
    py::dict res;
    res["nborn"] = nb;
    res["n"] = N;
    res["dec_index"] = idx;
    res["xz"] = xz;
    res["phase"] = ph;
    res["inv"] = inv;
    return res;
}

// ── PyBarrierBuffer — per-shot post-barrier state + channel bits container (S2.2/S2.4) ─────
// Holds one COLLAPSED post-barrier FramedSuperposition per shot (the residual-applied,
// kernel-collapsed state that twirl_collapse(need_amps=true) returns — genuinely per-shot,
// NOT a bare copy) plus that shot's raw certified-generator syndrome σ AND the packed
// detector/observable bits — all captured from the SAME sink invocation. Using a single
// pass guarantees that state(i) and dets()/obs() row i correspond to the exact same noise
// draw (no dual-sample misalignment even when gauge_rng_ or the retain-path RNG differs
// from the fast abelian path).
struct PyBarrierBuffer {
    // decoder-feedback perf (lazy materialization): the FULL per-shot collapsed state is NO
    // LONGER retained. Instead each shot's COMPACT record (σ, coins=[r fair‖κ chain‖born], plan)
    // is stored, and state(i)/materialize(i) reconstruct the state ON DEMAND from that record via
    // the sampler's exact replay (twirl_kernel.hpp: "amps reconstructible from (plan, σ, coins)").
    // Callers materialize only the ~O(#distinct-records) bucket representatives, not every shot.
    std::shared_ptr<TwirlRecordSampler> sampler_;    // shared: keeps the source sampler alive so
                                                     // materialize() is valid for the buffer's whole
                                                     // lifetime (even after the owner is re-sampled)
    int nshots_ = 0;                                 // shot count (states are not retained)
    // decoder-feedback perf (flat records, brief §4): the four per-shot variable-width record fields
    // are stored as FLAT CSR buffers — one contiguous `*_data_` blob plus an `*_off_` offset array of
    // length nshots+1 (row i is data[off[i] : off[i+1]]) — instead of a vector-of-vectors. The sink
    // APPENDS to the flat blob (amortized O(1), geometric growth) with NO per-shot heap allocation
    // (the pre-perf sink emplace_back'd 4 fresh std::vectors per shot ≈ 0.22 µs/shot). The offset
    // array IS the record "stride"; an off-by-one corrupts every downstream slice (the mutation gate).
    // barrier2 Lever 3 (low-risk σ word-store): σ is FIXED-WIDTH (ngens bits every shot — the sink
    // always writes exactly ngens, zero on clean/fallback). So instead of a per-shot 34-iteration
    // word→byte expansion into a CSR byte blob, store σ as a fixed stride of SGW_=ceil(ngens/64)
    // 64-bit WORDS (a bulk memcpy per shot) and expand words→bits at READ time (record_keys/sigma —
    // O(1) calls, not per shot). Output bytes are IDENTICAL (same bit i = (word[i>>6]>>(i&63))&1).
    std::vector<uint64_t> sig_words_;                    // nshots_ × SGW_ words (fixed stride)
    int SGW_ = 0;                                        // σ words per shot
    int sig_bits_ = 0;                                   // σ bit width (= ngens); byte width at read
    std::vector<uint8_t> coin_data_; std::vector<uint32_t> coin_off_{0};  // coins [r fair‖κ‖born]/shot
    std::vector<uint8_t> pk_data_;   std::vector<uint32_t> pk_off_{0};    // canonical residual plan/shot
    std::vector<double>  bu_data_;   std::vector<uint32_t> bu_off_{0};    // born-decision coins/shot
    // Slice helpers (row i of each CSR field): pointer + length, no copy.
    const uint64_t* sig_word_ptr(int i) const { return sig_words_.data() + (size_t)i * (size_t)SGW_; }
    int sig_len(int i) const { return sig_bits_; }       // fixed width every shot
    const uint8_t* coin_ptr(int i) const { return coin_off_[i]==coin_off_[i+1]?nullptr:coin_data_.data()+coin_off_[i]; }
    int coin_len(int i) const { return (int)(coin_off_[i+1]-coin_off_[i]); }
    const uint8_t* pk_ptr(int i) const { return pk_off_[i]==pk_off_[i+1]?nullptr:pk_data_.data()+pk_off_[i]; }
    int pk_len(int i) const { return (int)(pk_off_[i+1]-pk_off_[i]); }
    const double* bu_ptr(int i) const { return bu_off_[i]==bu_off_[i+1]?nullptr:bu_data_.data()+bu_off_[i]; }
    int bu_len(int i) const { return (int)(bu_off_[i+1]-bu_off_[i]); }
    // ── port-v3 T2 (L-E compact-record internal) ──────────────────────────────────────────
    // When compact_ is true (sample_barrier(..., compact=True) — the xtim.port record route),
    // the per-shot plan BYTES are NOT stored. Instead the sink records the compact plan
    // identity: a dense interned plan id (0 = empty plan) + the per-shot prefix support words,
    // plus a per-DISTINCT-plan (a‖0xFF‖cz) suffix blob. The legacy plan bytes
    // (prefix.x words LE ‖ prefix.z words LE ‖ a ‖ 0xFF ‖ cz) are reconstructed on demand —
    // per GROUP for record_groups(), per shot only in the exactness-test accessors. This is a
    // BIJECTIVE re-encoding (plan cache interns on exactly (a&1, cz) with full content
    // equality), so the group partition and every reconstructed byte are identical to the
    // legacy store. The legacy (compact_=false) code paths below are byte-for-byte untouched.
    bool compact_ = false;
    std::vector<uint32_t> plan_id_;       // per-shot dense plan id (0 = empty plan)
    std::vector<uint64_t> prefix_words_;  // per-shot 2*PNW_ words (x part then z part)
    int PNW_ = 0;                         // prefix words per part = ceil(n/64)
    std::vector<uint8_t>  suffix_data_;   // per-plan (a‖0xFF‖cz) bytes, CSR by id
    std::vector<uint32_t> suffix_off_{0}; // suffix of plan id g is data[off[g-1] : off[g]]
    // Compact-mode plan-byte length of shot i (== the legacy pk_len(i)).
    int plan_len_c(int i) const {
        const uint32_t id = plan_id_[(size_t)i];
        if (!id) return 0;
        return 16 * PNW_ + (int)(suffix_off_[id] - suffix_off_[id - 1]);
    }
    // Reconstruct shot i's legacy plan bytes into `out` (byte-identical to the engine's
    // plankey_ serialization: prefix x words LE ‖ prefix z words LE ‖ a ‖ 0xFF ‖ cz).
    void build_plan_bytes(int i, std::vector<uint8_t>& out) const {
        out.clear();
        const uint32_t id = plan_id_[(size_t)i];
        if (!id) return;
        out.reserve((size_t)plan_len_c(i));
        const uint64_t* pw = prefix_words_.data() + (size_t)i * (size_t)(2 * PNW_);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(pw);
        out.insert(out.end(), pb, pb + (size_t)(16 * PNW_));
#else
        for (int w = 0; w < 2 * PNW_; ++w)
            for (int b = 0; b < 8; ++b)
                out.push_back((uint8_t)((pw[w] >> (b * 8)) & 0xFF));
#endif
        out.insert(out.end(), suffix_data_.begin() + suffix_off_[id - 1],
                   suffix_data_.begin() + suffix_off_[id]);
    }
    std::vector<uint8_t> dets_packed_;               // packed dets: shots × DBB_ bytes (b8 LE)
    std::vector<uint8_t> obs_packed_;                // packed obs:  shots × OBB_ bytes (b8 LE)
    std::vector<uint8_t> dec_packed_;                // Task 2b.1: packed decisions (b8 LE)
    int DBB_ = 0;                                    // bytes per shot for dets
    int OBB_ = 0;                                    // bytes per shot for obs
    int DECB_ = 0;                                   // bytes per shot for decisions

    int num_shots() const { return nshots_; }

    // Lazily reconstruct shot i's EXACT collapsed post-barrier + post-decision state from its
    // compact record. Called once per bucket representative, NOT per shot.
    PyFramedSuperposition materialize(int i) const {
        if (i < 0 || i >= nshots_)
            throw std::out_of_range(
                "BarrierBuffer.materialize(" + std::to_string(i) +
                "): index out of range [0, " + std::to_string(nshots_) + ")");
        if (!sampler_)
            throw std::runtime_error(
                "BarrierBuffer.materialize: the source sampler is no longer available "
                "(materialize the states before re-sampling / destroying the sampler)");
        if (compact_) {
            // Compact store: reconstruct this shot's legacy plan bytes (per representative
            // call, not per shot) and replay through the identical engine path.
            std::vector<uint8_t> pk;
            build_plan_bytes(i, pk);
            return PyFramedSuperposition(sampler_->materialize_shot(
                pk.empty() ? nullptr : pk.data(), (int)pk.size(),
                coin_ptr(i), coin_len(i), bu_ptr(i), bu_len(i)));
        }
        return PyFramedSuperposition(sampler_->materialize_shot(
            pk_ptr(i), pk_len(i), coin_ptr(i), coin_len(i), bu_ptr(i), bu_len(i)));
    }

    // Back-compat alias: state(i) IS the lazily materialized state.
    PyFramedSuperposition state(int i) const { return materialize(i); }

    std::vector<uint8_t> sigma(int i) const {
        if (i < 0 || i >= nshots_)
            throw std::out_of_range(
                "BarrierBuffer.sigma(" + std::to_string(i) +
                "): index out of range [0, " + std::to_string(nshots_) + ")");
        std::vector<uint8_t> out((size_t)sig_bits_);     // expand SGW_ words → ngens bytes
        const uint64_t* sw = sig_word_ptr(i);
        for (int b = 0; b < sig_bits_; ++b) out[(size_t)b] = (uint8_t)((sw[b >> 6] >> (b & 63)) & 1ULL);
        return out;
    }

    // Per-shot coin record (r fair coins then κ chain outcomes, 0/1). Length varies per shot with
    // the residual plan. Together with sigma(i) this is a SUFFICIENT classical statistic for the
    // collapsed state(i) — the record-hash bucketing key (see xtim/feedback.py run_protocol).
    std::vector<uint8_t> coins(int i) const {
        if (i < 0 || i >= nshots_)
            throw std::out_of_range(
                "BarrierBuffer.coins(" + std::to_string(i) +
                "): index out of range [0, " + std::to_string(nshots_) + ")");
        return std::vector<uint8_t>(coin_data_.begin() + coin_off_[i], coin_data_.begin() + coin_off_[i + 1]);
    }

    // Per-shot canonical residual plan bytes (prefix x/z support + a-mask + cz, global phase
    // excluded). The third component of the record-hash key (plan, σ, coins). Empty on clean /
    // fallback shots.
    std::vector<uint8_t> plan_key(int i) const {
        if (i < 0 || i >= nshots_)
            throw std::out_of_range(
                "BarrierBuffer.plan_key(" + std::to_string(i) +
                "): index out of range [0, " + std::to_string(nshots_) + ")");
        if (compact_) {
            std::vector<uint8_t> out;
            build_plan_bytes(i, out);
            return out;
        }
        return std::vector<uint8_t>(pk_data_.begin() + pk_off_[i], pk_data_.begin() + pk_off_[i + 1]);
    }

    // E3 (exact-residual arc, Task 3): STRUCTURED plan accessor — parse a plan key's bytes into
    // the residual normal form + the plan's ShotLaw, as numpy data.  PURE DATA EXPOSURE of the
    // structures the engine already derives per shot (TwirlRecordSampler::plan_structure →
    // parse_plan_key + the existing build_shot_law, the identical call materialize(i) makes);
    // no new C++ computation.  Cold per DISTINCT plan (memoize on the plan_key bytes) — the
    // per-(check, plan) indefiniteness classifier composes conjugation + span test in Python.
    //
    // Input: the plan key bytes exactly as plan_key(i) returns them (b"" = clean/identity shot).
    // Returns dict (n = deferred wire count, gw = σ words = sigmas() columns):
    //   n, ngens, gw, r, kappa : int      fallback : bool
    //   prefix_xz : uint8[2n]        residual prefix Pauli P support (low n = x, high n = z;
    //                                 P's global phase is EXCLUDED from the key — it cancels in
    //                                 every conjugation R†OR and never enters per-check signs)
    //   a         : uint8[n]         residual S-layer mask (S^a)
    //   cz        : int32[ncz, 2]    residual CZ pairs (j < l, sorted)
    //   det_signs : uint64[gw]       σ deterministic base point for THIS plan (prefix folded in)
    //   coin_masks   : uint64[r, gw]     per fair coin k: σ-flip mask (coins[k] == 1 ⇒ σ ^= row)
    //   kernel_masks : uint64[kappa, gw] per kernel outcome j: σ-flip preimage (see ShotLaw doc)
    //   kernel_base  : uint8[kappa]      base-sign bits pairing masks with outcomes
    //   kernel_foldable : uint8[kappa]   1 = the direction folds σ (V2 reachability split)
    //   kernel_xz    : uint8[kappa, 2n]  Born-measured kernel logical reps (bare frame, measured
    //                                    with outcome bit coins(i)[r + j], 0 = +1)
    //   kernel_phase : uint8[kappa]      exact i^phase of each kernel rep (operator = i^phase·X^x·Z^z)
    py::dict plan_structure(py::bytes plan_key_bytes) const {
        if (!sampler_)
            throw std::runtime_error(
                "BarrierBuffer.plan_structure: the source sampler is no longer available "
                "(read the plan structures before re-sampling / destroying the sampler)");
        // port-v3 T2: delegates to the shared dict builder (one owner of the format;
        // PyTwirlSampler.plan_structure emits the identical dict without a buffer).
        return plan_structure_dict(*sampler_, sig_bits_,
                                   plan_key_bytes.cast<std::string>());
    }

    // T5 step 0 (exact-residual arc): PURE-DATA exposure of the sampler's Born-measured
    // decision operators (LOGICAL/ANTI-class DECISIONs, declaration order) — see
    // TwirlRecordSampler::born_dec_ops for the full semantics.  Returns dict:
    //   nborn : int                # of Born-class decisions (0 for det-only circuits)
    //   n     : int                deferred wire count of the ops (0 when nborn == 0)
    //   dec_index : int32[nborn]   DECISION index each operator decides
    //   xz    : uint8[nborn, 2n]   operator support (low n = x, high n = z; op = i^phase·X^x·Z^z)
    //   phase : uint8[nborn]       exact i^phase of each operator
    //   inv   : uint8[nborn]       deterministic record invert (emitted bit = raw ⊕ inv ⊕ …)
    // The RAW outcome bit of born decision j on shot i (0 = +1 eigenvalue of the operator on
    // the collapsed state) is coins(i)[r + kappa + j] (r, kappa from plan_structure of that
    // shot's plan).  Coordinate/phase conventions identical to certified_symplectic /
    // plan_structure's kernel_xz.
    py::dict born_dec() const {
        if (!sampler_)
            throw std::runtime_error(
                "BarrierBuffer.born_dec: the source sampler is no longer available "
                "(read the born-decision operators before re-sampling / destroying the sampler)");
        // port-v3 T2: delegates to the shared dict builder (one owner of the format).
        return born_dec_dict(*sampler_);
    }

    // Return the packed detector bits as a (shots, DBB_) uint8 numpy array (Stim b8 layout).
    // Populated by sample_barrier from the SAME sink pass that captures the state — no
    // dual-sample misalignment.
    py::array_t<uint8_t> dets() const {
        return packed_to_numpy(dets_packed_, nshots_, DBB_);
    }

    // Return the packed observable bits as a (shots, OBB_) uint8 numpy array (Stim b8 layout).
    py::array_t<uint8_t> obs() const {
        return packed_to_numpy(obs_packed_, nshots_, OBB_);
    }

    // Task 2b.1: packed decision bits (shots, DECB_) uint8 Stim-b8 (LE) — one bit per declared
    // DECISION index, captured from the SAME sink pass as state(i) (no dual-sample misalignment).
    py::array_t<uint8_t> decisions() const {
        return packed_to_numpy(dec_packed_, nshots_, DECB_);
    }

    // E1 (exact-residual arc, Task 1): bulk certified-generator sign syndrome matrix.
    // Returns a COPY of sig_words_ reshaped as (nshots, SGW_) uint64 array.
    // This is the zero-transform view of the word store: bit b of shot i is
    //   (sigmas()[i, b>>6] >> (b&63)) & 1   — identical to sigma(i)[b].
    // Tail bits (positions sig_bits_ .. SGW_*64-1) are ALWAYS ZERO by construction: the
    // source sigma vectors are initialized to GW zero words with only bits 0..ngens-1 ever
    // written, and the entire store is pre-zeroed before sampling (the sink memcpys full
    // rows; it does NOT zero per-row). No masking is required; consumers may nonetheless
    // mask with sig_bits for clarity.
    // Returns shape (nshots_, 0) when SGW_==0 (empty circuit; SGW_==0 iff ngens==0),
    // and shape (0, SGW_) when nshots_==0.
    py::array_t<uint64_t> sigmas() const {
        const int cols = (SGW_ > 0) ? SGW_ : 0;
        py::array_t<uint64_t> a({nshots_, cols});
        if (nshots_ > 0 && cols > 0)
            std::memcpy(a.mutable_data(), sig_words_.data(),
                        (size_t)nshots_ * (size_t)cols * sizeof(uint64_t));
        return a;
    }

    // decoder-feedback perf (vectorized bucketing): the per-shot record (σ ‖ coins ‖ plan) as a
    // single FIXED-WIDTH (shots × W) uint8 matrix — one pybind call, O(shots) fill. run_protocol
    // np.unique's a void-view of this matrix to partition shots by record in O(shots) with NO
    // per-shot pybind round-trip. Layout per row (all lengths are ≤ 65535 here):
    //   σ bytes (SBW) ‖ [len_coins : 2 LE] ‖ coins (padded to max_coins) ‖
    //                    [len_plan  : 2 LE] ‖ plan  (padded to max_plan)
    // The explicit length prefixes make padding-zeros unambiguous, so distinct records always map
    // to distinct rows (an EXACT partition — over-splitting is impossible, under-merging is
    // impossible). The record is a SUFFICIENT statistic for the collapsed state, so shots sharing a
    // row share a state (verified by the correlation + record-bucketing gates).
    // ── port-v3 T2: compact-store grouping core ───────────────────────────────────────────
    // The compact per-shot identity is (σ words, coins, plan_id, prefix words) — a bijective
    // re-encoding of the legacy padded key row (σ bytes ‖ [len_c]‖coins‖pad ‖ [len_p]‖plan‖pad):
    //   σ words ↔ σ bytes (tail bits provably zero); coins verbatim; (plan_id, prefix) ↔ plan
    //   bytes (plan_id ↔ (a,cz) is the plan cache's own content-interning; serialization is
    //   injective at fixed n). Hence the partition, the first-occurrence order, and every
    //   reconstructed key byte are IDENTICAL to the legacy path — oracle-gated in
    //   tests/test_port_groups.py (exhaustive byte compare + corruption probe).
    // The walk hashes ~(SGW + 2*PNW + 1) words + the coin bytes per shot (~145 B on
    // cultivation d5, vs the legacy 2349-B padded-row memset+hash+map-string-insert) and
    // compares candidates component-wise against the group REPRESENTATIVE shot — full
    // equality, never hash-trust (never-silent-wrong).
    void groups_walk_compact(int32_t* gid, std::vector<int32_t>& first_occ) const {
        const size_t sgw = (size_t)SGW_;
        const size_t pwn = (size_t)(2 * PNW_);
        size_t cap = 16;
        while (cap < (size_t)nshots_ * 2 + 4) cap <<= 1;
        std::vector<int32_t> slot_rep(cap, -1), slot_gid(cap, 0);
        std::vector<uint64_t> slot_hash(cap, 0);
        auto shot_hash = [&](int i) -> uint64_t {
            uint64_t h = 0xcbf29ce484222325ull;
            auto mixw = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ull; };
            const uint64_t* sw = sig_words_.data() + (size_t)i * sgw;
            for (size_t w = 0; w < sgw; ++w) mixw(sw[w]);
            const int cl = coin_len(i);
            mixw((uint64_t)cl);
            const uint8_t* cn = coin_ptr(i);
            for (int b = 0; b < cl; ++b) mixw((uint64_t)cn[b]);
            mixw((uint64_t)plan_id_[(size_t)i]);
            const uint64_t* pw = prefix_words_.data() + (size_t)i * pwn;
            for (size_t w = 0; w < pwn; ++w) mixw(pw[w]);
            return h;
        };
        auto same = [&](int i, int j) -> bool {
            if (plan_id_[(size_t)i] != plan_id_[(size_t)j]) return false;
            const int cl = coin_len(i);
            if (cl != coin_len(j)) return false;
            if (sgw && std::memcmp(sig_words_.data() + (size_t)i * sgw,
                                   sig_words_.data() + (size_t)j * sgw,
                                   sgw * sizeof(uint64_t))) return false;
            if (cl && std::memcmp(coin_ptr(i), coin_ptr(j), (size_t)cl)) return false;
            if (pwn && std::memcmp(prefix_words_.data() + (size_t)i * pwn,
                                   prefix_words_.data() + (size_t)j * pwn,
                                   pwn * sizeof(uint64_t))) return false;
            return true;
        };
        first_occ.clear();
        const size_t mask = cap - 1;
        for (int i = 0; i < nshots_; ++i) {
            const uint64_t h = shot_hash(i);
            size_t s = h & mask;
            for (;;) {
                if (slot_rep[s] < 0) {
                    slot_rep[s] = i;
                    slot_hash[s] = h;
                    slot_gid[s] = (int32_t)first_occ.size();
                    gid[i] = slot_gid[s];
                    first_occ.push_back((int32_t)i);
                    break;
                }
                if (slot_hash[s] == h && same(i, slot_rep[s])) { gid[i] = slot_gid[s]; break; }
                s = (s + 1) & mask;
            }
        }
    }

    // Build shot i's legacy padded key row (row pre-zeroed, width W = SBW+2+maxc+2+maxp)
    // from the COMPACT store — byte-identical to the legacy per-shot row construction.
    void build_key_row_compact(int i, uint8_t* row, int SBW, int maxc) const {
        const uint64_t* sw = sig_word_ptr(i);
        for (int b = 0; b < sig_bits_; ++b)
            row[b] = (uint8_t)((sw[b >> 6] >> (b & 63)) & 1ULL);
        int p = SBW;
        const int cl = coin_len(i);
        const uint8_t* cn = coin_ptr(i);
        row[p] = (uint8_t)(cl & 0xFF);
        row[p + 1] = (uint8_t)((cl >> 8) & 0xFF);
        p += 2;
        for (int b = 0; b < cl; ++b) row[p + b] = cn[b];
        p = SBW + 2 + maxc;
        const int pl = plan_len_c(i);
        row[p] = (uint8_t)(pl & 0xFF);
        row[p + 1] = (uint8_t)((pl >> 8) & 0xFF);
        p += 2;
        if (pl) {
            const uint32_t id = plan_id_[(size_t)i];
            const uint64_t* pw = prefix_words_.data() + (size_t)i * (size_t)(2 * PNW_);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
            std::memcpy(row + p, pw, (size_t)(16 * PNW_));
            p += 16 * PNW_;
#else
            for (int w = 0; w < 2 * PNW_; ++w)
                for (int b = 0; b < 8; ++b)
                    row[p++] = (uint8_t)((pw[w] >> (b * 8)) & 0xFF);
#endif
            const uint32_t s0 = suffix_off_[id - 1], s1 = suffix_off_[id];
            std::memcpy(row + p, suffix_data_.data() + s0, (size_t)(s1 - s0));
        }
    }

    // Compact-mode legacy-key dims: SBW/maxc/maxp/W identical to the legacy computation.
    void key_dims_compact(int& SBW, int& maxc, int& maxp, int& W) const {
        SBW = nshots_ > 0 ? sig_bits_ : 0;
        maxc = 0;
        maxp = 0;
        for (int i = 0; i < nshots_; ++i) {
            maxc = std::max(maxc, coin_len(i));
            maxp = std::max(maxp, plan_len_c(i));
        }
        W = SBW + 2 + maxc + 2 + maxp;
    }

    py::array_t<uint8_t> record_keys_compact() const {
        int SBW, maxc, maxp, W;
        key_dims_compact(SBW, maxc, maxp, W);
        py::array_t<uint8_t> a({nshots_, W});
        uint8_t* out = a.mutable_data();
        std::memset(out, 0, (size_t)nshots_ * (size_t)W);
        for (int i = 0; i < nshots_; ++i)
            build_key_row_compact(i, out + (size_t)i * (size_t)W, SBW, maxc);
        return a;
    }

    py::tuple record_groups_compact() const {
        int SBW, maxc, maxp, W;
        key_dims_compact(SBW, maxc, maxp, W);
        py::array_t<int32_t> gids_arr(nshots_);
        int32_t* gid = gids_arr.mutable_data();
        std::vector<int32_t> first_occ_vec;
        {
            py::gil_scoped_release rel;
            groups_walk_compact(gid, first_occ_vec);
        }
        const int32_t n_groups = (int32_t)first_occ_vec.size();
        py::array_t<int32_t> first_occ_arr(n_groups);
        if (n_groups > 0)
            std::memcpy(first_occ_arr.mutable_data(), first_occ_vec.data(),
                        (size_t)n_groups * sizeof(int32_t));
        py::array_t<uint8_t> dk_arr({(int)n_groups, W});
        uint8_t* dk = dk_arr.mutable_data();
        std::memset(dk, 0, (size_t)n_groups * (size_t)W);
        // Legacy key bytes materialized ONCE PER GROUP (the L-E lever's read side).
        for (int32_t g = 0; g < n_groups; ++g)
            build_key_row_compact(first_occ_vec[(size_t)g], dk + (size_t)g * (size_t)W,
                                  SBW, maxc);
        return py::make_tuple(gids_arr, first_occ_arr, dk_arr);
    }

    // Mutation probe (port-v3 T2 oracle): corrupt ONE shot's COMPACT identity (prefix word 0).
    // The group partition / reconstructed keys must change — proving the compact fields are
    // load-bearing for the partition (never a decorative copy). Test-only; compact mode only.
    void _debug_corrupt_compact(int i, const std::string& field) {
        if (!compact_)
            throw std::runtime_error(
                "_debug_corrupt_compact: buffer is not in compact mode");
        if (i < 0 || i >= nshots_)
            throw std::out_of_range("_debug_corrupt_compact: shot index out of range");
        if (field == "prefix") {
            prefix_words_[(size_t)i * (size_t)(2 * PNW_)] ^= 1ULL;
        } else if (field == "plan_id") {
            // port-v3 T3 fold-in: corrupt the INTERNED plan id itself (set the
            // shot's plan to the empty plan).  Proves the plan_id component of
            // the compact identity is load-bearing for the partition AND for
            // the reconstructed legacy plan bytes (not just the prefix words).
            if (plan_id_[(size_t)i] == 0)
                throw std::runtime_error(
                    "_debug_corrupt_compact: shot has no plan (plan_id == 0) — "
                    "pick a shot with a nonzero plan for the plan_id probe");
            plan_id_[(size_t)i] = 0;
        } else {
            throw std::runtime_error(
                "_debug_corrupt_compact: unknown field '" + field +
                "' (expected 'prefix' or 'plan_id')");
        }
    }

    py::array_t<uint8_t> record_keys() const {
        if (compact_) return record_keys_compact();
        int SBW = 0, maxc = 0, maxp = 0;
        for (int i = 0; i < nshots_; ++i) {
            SBW  = std::max(SBW, sig_len(i));
            maxc = std::max(maxc, coin_len(i));
            maxp = std::max(maxp, pk_len(i));
        }
        const int W = SBW + 2 + maxc + 2 + maxp;
        py::array_t<uint8_t> a({nshots_, W});
        uint8_t* out = a.mutable_data();
        std::memset(out, 0, (size_t)nshots_ * (size_t)W);
        for (int i = 0; i < nshots_; ++i) {
            uint8_t* row = out + (size_t)i * (size_t)W;
            int p = 0;
            const int sl = sig_len(i);  const uint64_t* sw = sig_word_ptr(i);
            for (int b = 0; b < sl; ++b) row[p + b] = (uint8_t)((sw[b >> 6] >> (b & 63)) & 1ULL);
            p = SBW;
            const int cl = coin_len(i); const uint8_t* cn = coin_ptr(i);
            row[p] = (uint8_t)(cl & 0xFF); row[p + 1] = (uint8_t)((cl >> 8) & 0xFF);
            p += 2;
            for (int b = 0; b < cl; ++b) row[p + b] = cn[b];
            p = SBW + 2 + maxc;
            const int pl = pk_len(i);   const uint8_t* pk = pk_ptr(i);
            row[p] = (uint8_t)(pl & 0xFF); row[p + 1] = (uint8_t)((pl >> 8) & 0xFF);
            p += 2;
            for (int b = 0; b < pl; ++b) row[p + b] = pk[b];
        }
        return a;
    }

    // decoder-feedback perf (vectorized bucketing, part 2): FACTORIZE the per-shot record into a
    // dense int32 first-occurrence group id, in ONE pybind call, WITHOUT ever materializing the
    // wide (shots × W) key matrix in Python. Returns a length-`nshots_` int32 array `gid` where
    // gid[i] is the index (0-based, in first-occurrence order) of shot i's DISTINCT record.
    //
    // Correctness: the hash-map key is byte-for-byte the SAME fixed-width padded row `record_keys`
    // emits (σ ‖ [len_c] ‖ coins‖pad ‖ [len_p] ‖ plan‖pad), so the partition is IDENTICAL to the
    // Python `|S{W}` view of record_keys() — but produced by C-level bytes hashing (no wide-key
    // sort, no per-shot Python). run_protocol combines this record id with the (small) driving
    // decision-pattern code at numpy speed to form the final (dp, record) groups.
    py::array_t<int32_t> record_group_ids() const {
        if (compact_) {
            py::array_t<int32_t> out(nshots_);
            int32_t* gid = out.mutable_data();
            std::vector<int32_t> first_occ_vec;
            {
                py::gil_scoped_release rel;
                groups_walk_compact(gid, first_occ_vec);
            }
            return out;
        }
        int SBW = 0, maxc = 0, maxp = 0;
        for (int i = 0; i < nshots_; ++i) {
            SBW  = std::max(SBW, sig_len(i));
            maxc = std::max(maxc, coin_len(i));
            maxp = std::max(maxp, pk_len(i));
        }
        const int W = SBW + 2 + maxc + 2 + maxp;
        py::array_t<int32_t> out(nshots_);
        int32_t* gid = out.mutable_data();
        {
            py::gil_scoped_release rel;   // pure C++ work — release the GIL
            std::unordered_map<std::string, int32_t> ids;
            ids.reserve((size_t)nshots_ * 2 + 16);
            std::string row((size_t)W, '\0');
            int32_t next = 0;
            for (int i = 0; i < nshots_; ++i) {
                std::memset(&row[0], 0, (size_t)W);
                int p = 0;
                const int sl = sig_len(i);  const uint64_t* sw = sig_word_ptr(i);
                for (int b = 0; b < sl; ++b) row[(size_t)(p + b)] = (char)((sw[b >> 6] >> (b & 63)) & 1ULL);
                p = SBW;
                const int cl = coin_len(i); const uint8_t* cn = coin_ptr(i);
                row[(size_t)p] = (char)(cl & 0xFF); row[(size_t)(p + 1)] = (char)((cl >> 8) & 0xFF);
                p += 2;
                for (int b = 0; b < cl; ++b) row[(size_t)(p + b)] = (char)cn[b];
                p = SBW + 2 + maxc;
                const int pl = pk_len(i);   const uint8_t* pk = pk_ptr(i);
                row[(size_t)p] = (char)(pl & 0xFF); row[(size_t)(p + 1)] = (char)((pl >> 8) & 0xFF);
                p += 2;
                for (int b = 0; b < pl; ++b) row[(size_t)(p + b)] = (char)pk[b];
                auto it = ids.find(row);
                if (it == ids.end()) { ids.emplace(row, next); gid[i] = next; ++next; }
                else                 { gid[i] = it->second; }
            }
        }
        return out;
    }

    // decoder-feedback perf (FUSED single-walk accessor): replaces the two-pass pattern
    // ``record_keys() + record_group_ids()`` with one hash-map walk. Returns a 3-tuple:
    //   (a) gids          — (nshots,) int32: per-shot first-occurrence group id. IDENTICAL
    //                        to record_group_ids() — first-occurrence order is load-bearing
    //                        for reproducibility (same group seeds → same ProtocolResult).
    //   (b) first_occ     — (n_groups,) int32: shot index of the FIRST occurrence of each
    //                        group (in first-occurrence order). Lets callers index the
    //                        bucket representative without scanning `gids`.
    //   (c) distinct_keys — (n_groups, W) uint8: the fixed-width padded key row for each
    //                        distinct record, gathered during the SAME hash-map walk.
    //                        Byte-identical to record_keys()[first_occ[g]] for each group g
    //                        (row g == the first-occurrence row from record_keys()).
    //
    // Per-shot only the group id is needed by run_protocol; the FULL per-shot key matrix
    // (record_keys) is only needed for the ~O(n_groups) distinct records.  distinct_keys is
    // ~100× smaller than the full per-shot matrix, so reconstructing ``record_keys`` from
    // ``distinct_keys[gids]`` (numpy fancy indexing) is both correct and cheaper than
    // computing it from scratch.  record_keys() is retained for exactness tests only.
    py::tuple record_groups() const {
        if (compact_) return record_groups_compact();
        int SBW = 0, maxc = 0, maxp = 0;
        for (int i = 0; i < nshots_; ++i) {
            SBW  = std::max(SBW, sig_len(i));
            maxc = std::max(maxc, coin_len(i));
            maxp = std::max(maxp, pk_len(i));
        }
        const int W = SBW + 2 + maxc + 2 + maxp;
        py::array_t<int32_t> gids_arr(nshots_);
        int32_t* gid = gids_arr.mutable_data();
        std::vector<int32_t> first_occ_vec;
        std::vector<uint8_t> dk_flat;           // n_groups × W bytes (grows by W per new group)
        {
            py::gil_scoped_release rel;
            std::unordered_map<std::string, int32_t> ids;
            ids.reserve((size_t)nshots_ * 2 + 16);
            std::string row((size_t)W, '\0');
            int32_t next = 0;
            for (int i = 0; i < nshots_; ++i) {
                std::memset(&row[0], 0, (size_t)W);
                int p = 0;
                const int sl = sig_len(i);  const uint64_t* sw = sig_word_ptr(i);
                for (int b = 0; b < sl; ++b) row[(size_t)(p + b)] = (char)((sw[b >> 6] >> (b & 63)) & 1ULL);
                p = SBW;
                const int cl = coin_len(i); const uint8_t* cn = coin_ptr(i);
                row[(size_t)p] = (char)(cl & 0xFF); row[(size_t)(p + 1)] = (char)((cl >> 8) & 0xFF);
                p += 2;
                for (int b = 0; b < cl; ++b) row[(size_t)(p + b)] = (char)cn[b];
                p = SBW + 2 + maxc;
                const int pl = pk_len(i);   const uint8_t* pk = pk_ptr(i);
                row[(size_t)p] = (char)(pl & 0xFF); row[(size_t)(p + 1)] = (char)((pl >> 8) & 0xFF);
                p += 2;
                for (int b = 0; b < pl; ++b) row[(size_t)(p + b)] = (char)pk[b];
                auto it = ids.find(row);
                if (it == ids.end()) {
                    ids.emplace(row, next);
                    gid[i] = next;
                    first_occ_vec.push_back((int32_t)i);
                    // Gather this distinct row into the flat key buffer (exactly W bytes).
                    dk_flat.insert(dk_flat.end(),
                                   reinterpret_cast<const uint8_t*>(row.data()),
                                   reinterpret_cast<const uint8_t*>(row.data()) + W);
                    ++next;
                } else {
                    gid[i] = it->second;
                }
            }
        }
        const int32_t n_groups = (int32_t)first_occ_vec.size();
        py::array_t<int32_t> first_occ_arr(n_groups);
        if (n_groups > 0)
            std::memcpy(first_occ_arr.mutable_data(), first_occ_vec.data(),
                        (size_t)n_groups * sizeof(int32_t));
        py::array_t<uint8_t> dk_arr({n_groups, W});
        if (!dk_flat.empty())
            std::memcpy(dk_arr.mutable_data(), dk_flat.data(), dk_flat.size());
        return py::make_tuple(gids_arr, first_occ_arr, dk_arr);
    }
};

// ── Task 3.0: multi-qubit carried-port state-level injection helpers ──────────────────────
// Grow a FramedSuperposition by `k` fresh |0> qubits at columns n..n+k-1 (tensor with I on the
// tableau, +Z_j stabilisers ⇒ |0>, eps 0). free/amplitudes untouched (the new qubits are not
// logical; sigma keys are over `free`), so a magic (chi>=2) state grows verbatim.
static void grow_framed(FramedSuperposition& fs, int k) {
    if (k <= 0) return;
    const int old_n = fs.n();
    fs.U.grow_identity_qubits(k);
    fs.eps.resize((size_t)old_n + k, 0);   // new stabilisers g_{n+j}=Z_{n+j}: g|0>=+|0>
}

// D2 — the n-qubit computational-basis |0…0> as a FramedSuperposition (χ=1, one amplitude entry
// with empty sigma and coeff 1).  A default-constructed FramedSuperposition has an EMPTY container
// (χ=0); this seeds the single branch, matching FramedSuperposition::from_css(|0>).
static FramedSuperposition zero_state(int n) {
    FramedSuperposition s(n);                 // frame = identity (Z-stabilised |0>), free empty
    s.entries().clear();
    s.entries().emplace_back(std::vector<uint8_t>{}, std::complex<double>(1.0, 0.0));
    s.sync_alpha_k();
    return s;
}

// D2 — TENSOR two FramedSuperpositions:  out = A ⊗ B  (A on wires [0,A.n), B on wires
// [A.n, A.n+B.n)).  The frame is the DIRECT SUM of the two symplectic frames (block-diagonal
// tableau — B's rows shifted by A.n in column space); eps concatenates; the distinguished
// (magic/logical) `free` rows concatenate (B's shifted by A.n); the amplitude container is the
// OUTER PRODUCT (χ multiplies) — sigma = (A's sigma over A.free) ++ (B's sigma over B.free),
// coeff = c_A · c_B.  This is a genuine general engine primitive: it composes a carried patch of
// ANY χ with a freshly-built magic block of ANY χ.  (Verified byte-identical to build_bare_state
// on the all-|0> instance at the injection site — see the composition guard.)
static FramedSuperposition tensor_framed(const FramedSuperposition& A,
                                         const FramedSuperposition& B) {
    const int nA = A.n(), nB = B.n();
    FramedSuperposition out(A);              // A verbatim (frame + eps + free + amplitudes)
    grow_framed(out, nB);                    // append nB fresh |0> columns (identity rows N_A+q)
    // Splice B's frame rows into columns [nA, nA+nB): row (nA+q) becomes B.Xrow[q]/Zrow[q] with
    // every column bit shifted up by nA (B is on its own qubits; no overlap with A's columns).
    auto shift_into = [&](const Pauli& src, Pauli& dst) {
        dst.phase = src.phase;
        for (int q = 0; q < nB; ++q) {
            if (src.xbit(q)) dst.setx(nA + q);
            if (src.zbit(q)) dst.setz(nA + q);
        }
    };
    for (int q = 0; q < nB; ++q) {
        Pauli xr(nA + nB), zr(nA + nB);
        shift_into(B.U.Xrow[(size_t)q], xr);
        shift_into(B.U.Zrow[(size_t)q], zr);
        out.U.Xrow[(size_t)(nA + q)] = xr;
        out.U.Zrow[(size_t)(nA + q)] = zr;
        out.eps[(size_t)(nA + q)] = B.eps[(size_t)q];
    }
    out.U.invalidate_dual();
    // free: A.free (indices < nA, unchanged) ++ (B.free shifted by nA).
    const int kA = (int)A.free.size();
    for (int f : B.free) out.free.push_back(nA + f);
    out.sync_alpha_k();
    // amplitudes: outer product.  out entry m*chiB + i has sigma = A[m].sigma ++ B[i].sigma
    // (ordered by out.free = A.free ++ shifted B.free) and coeff = A[m].c · B[i].c.
    const auto& ea = A.entries();
    const auto& eb = B.entries();
    const int kB = (int)B.free.size();
    std::vector<FramedSuperposition::Entry> prod;
    prod.reserve(ea.size() * eb.size());
    for (const auto& a : ea)
        for (const auto& b : eb) {
            std::vector<uint8_t> sig((size_t)(kA + kB), 0);
            for (int d = 0; d < kA; ++d) sig[(size_t)d] = a.first[(size_t)d];
            for (int d = 0; d < kB; ++d) sig[(size_t)(kA + d)] = b.first[(size_t)d];
            prod.emplace_back(std::move(sig), a.second * b.second);
        }
    out.entries() = std::move(prod);
    out.sync_alpha_k();
    return out;
}

// Apply the Clifford Gate instructions of a (measurement/noise-inclusive) deferred circuit to a
// FramedSuperposition, in stream order — turning an INITIAL state into the PRE-MEASUREMENT state
// the twirl sampler's bare contract requires (build_bare_state does this from |0>; injection does
// it from the carried input). Returns "" on success, or a loud message on an unsupported op:
//   * a non-Clifford gate (T/CS/CCZ/CH) — fresh-magic guard already refuses these upstream;
//   * a mid-stream Reset (build_bare_state's provenance would differ) — out of scope.
static std::string apply_stream_cliffords(FramedSuperposition& fs, const Circuit& deferred) {
    for (const Instr& ins : deferred.stream) {
        if (ins.kind == Instr::Kind::Reset) {
            // A reset re-initialises to |0>. Our fresh wires arrive |0>; a reset on them is a
            // no-op. A reset on any other wire (a carried/derived wire) is a genuine mid-stream
            // reinitialisation we do not model here — refuse loudly.
            for (int q : ins.qubits) {
                Pauli zq(fs.n());
                if (q >= 0 && q < fs.n()) zq.z[(size_t)(q >> 6)] |= (1ull << (q & 63));
                if (std::abs(fs.expectation(zq) - 1.0) > 1e-9)
                    return "state-level injection: mid-stream RESET on a non-|0> wire is out of "
                           "scope (Clifford consumer stages only)";
            }
            continue;
        }
        if (ins.kind != Instr::Kind::Gate) continue;   // Noise/Measure/Observable/feedback: not state ops
        uint8_t kind;
        switch (ins.gate) {
            case GateKind::H:   kind = 0; break;
            case GateKind::S:   kind = 1; break;
            case GateKind::SDG: kind = 2; break;
            case GateKind::X:   kind = 3; break;
            case GateKind::Y:   kind = 4; break;
            case GateKind::Z:   kind = 5; break;
            case GateKind::CX:  kind = 6; break;
            case GateKind::CZ:  kind = 7; break;
            default:
                return "state-level injection: non-Clifford gate on a consumer stage is out of "
                       "scope (only Clifford consumer stages carry patches)";
        }
        const int a = ins.targets.empty() ? -1 : ins.targets[0];
        const int b = ins.targets.size() > 1 ? ins.targets[1] : -1;
        fs.apply_clifford(kind, a, b);
    }
    return "";
}

// D2 — like apply_stream_cliffords, but ONLY bakes gates that TOUCH a port wire (`is_port[w]`).
// Gates supported entirely on FRESH (non-port) wires are SKIPPED — they are already baked into the
// fresh block (built by build_bare_state on the fresh-only sub-circuit).  A non-Clifford gate that
// touches a port wire is the CROSSING case (front-pushed magic on a carried wire) — returned as a
// precise loud message.  A non-Clifford gate on fresh wires only is skipped (already in the block).
// Wire indices here are in the COMBINED space (post-remap): port wires = src_wires, fresh = the
// appended |0> columns.  `is_port` is indexed over the combined width.
static std::string apply_stream_cliffords_ports(FramedSuperposition& fs, const Circuit& deferred,
                                                const std::vector<uint8_t>& is_port) {
    auto touches_port = [&](const Instr& ins) {
        for (int q : ins.targets)
            if (q >= 0 && q < (int)is_port.size() && is_port[(size_t)q]) return true;
        return false;
    };
    for (const Instr& ins : deferred.stream) {
        if (ins.kind == Instr::Kind::Reset) {
            if (!touches_port(ins)) continue;   // fresh-wire reset: fresh block arrives |0>
            for (int q : ins.qubits) {
                Pauli zq(fs.n());
                if (q >= 0 && q < fs.n()) zq.z[(size_t)(q >> 6)] |= (1ull << (q & 63));
                if (std::abs(fs.expectation(zq) - 1.0) > 1e-9)
                    return "state-level injection: mid-stream RESET on a non-|0> carried wire is "
                           "out of scope (Clifford consumer stages only)";
            }
            continue;
        }
        if (ins.kind != Instr::Kind::Gate) continue;
        if (!touches_port(ins)) continue;       // fresh-only gate: already baked into the block
        uint8_t kind;
        switch (ins.gate) {
            case GateKind::H:   kind = 0; break;
            case GateKind::S:   kind = 1; break;
            case GateKind::SDG: kind = 2; break;
            case GateKind::X:   kind = 3; break;
            case GateKind::Y:   kind = 4; break;
            case GateKind::Z:   kind = 5; break;
            case GateKind::CX:  kind = 6; break;
            case GateKind::CZ:  kind = 7; break;
            default:
                return "state-level injection: front-pushed magic supported on carried wires — "
                       "a non-Clifford gate entangles a fresh magic qubit with a carried port "
                       "qubit; prepare the magic state upstream of the carried-fresh entanglement "
                       "(prep-then-teleport), then this composes";
        }
        const int a = ins.targets.empty() ? -1 : ins.targets[0];
        const int b = ins.targets.size() > 1 ? ins.targets[1] : -1;
        fs.apply_clifford(kind, a, b);
    }
    return "";
}

// D2 — build the FRESH-ONLY sub-circuit: the consumer deferred stream restricted to gates whose
// support lies entirely on FRESH (non-port) wires, with a Reset on each fresh wire prepended so
// build_bare_state sees them as |0>.  Wire indices are the ORIGINAL consumer indices (0..Nc-1);
// build_bare_state consumes width `Nc` and reports χ = 2^k for the k fresh magic states.  Port
// wires appear in NO instruction (their gates are the crossing/port gates, excluded here and baked
// later), so the fresh block is a genuine tensor factor on the fresh wires.
static Circuit fresh_only_subcircuit(const Circuit& deferred, const std::vector<uint8_t>& is_port,
                                     int Nc) {
    Circuit out;
    out.n = Nc;
    for (int w = 0; w < Nc; ++w)
        if (w < (int)is_port.size() && !is_port[(size_t)w]) {
            Instr r; r.kind = Instr::Kind::Reset; r.qubits = {w};
            out.stream.push_back(r);
        }
    auto touches_port = [&](const Instr& ins) {
        for (int q : ins.targets)
            if (q >= 0 && q < (int)is_port.size() && is_port[(size_t)q]) return true;
        return false;
    };
    for (const Instr& ins : deferred.stream) {
        if (ins.kind != Instr::Kind::Gate) continue;   // magic + Cliffords are Gate instrs
        if (touches_port(ins)) continue;                // cross/port gates baked later
        out.stream.push_back(ins);
    }
    return out;
}

// D2 — STATE-BLIND crossing check (the §6.1 narrow refusal, made general).  A fresh magic state must
// be prepared BEFORE it entangles with any carried port qubit (prep-then-teleport).  We propagate a
// "port-tainted" mark along the stream: port wires start tainted; any multi-qubit gate spreads taint
// to all its wires if ANY is tainted (they become correlated with the carried state).  A non-Clifford
// gate acting on a tainted wire means its front-pushed magic support reaches the carried state — the
// injection cannot factor it out.  Returns the precise loud message on that case, else "".  This is
// conservative (state-blind), so it never depends on the carried state collapsing the magic away
// (the all-|0> exactness oracle alone would MISS a control-in-superposition crossing).
static std::string check_no_magic_on_carried(const Circuit& deferred,
                                             const std::vector<uint8_t>& is_port_orig, int Nc) {
    std::vector<uint8_t> tainted((size_t)(Nc > 0 ? Nc : 1), 0);
    for (int w = 0; w < Nc; ++w)
        if (w < (int)is_port_orig.size() && is_port_orig[(size_t)w]) tainted[(size_t)w] = 1;
    auto is_nonclifford = [](GateKind g) {
        return !(g == GateKind::H || g == GateKind::S || g == GateKind::SDG ||
                 g == GateKind::X || g == GateKind::Y || g == GateKind::Z ||
                 g == GateKind::CX || g == GateKind::CZ);
    };
    for (const Instr& ins : deferred.stream) {
        if (ins.kind != Instr::Kind::Gate) continue;
        bool any_tainted = false;
        for (int q : ins.targets)
            if (q >= 0 && q < Nc && tainted[(size_t)q]) { any_tainted = true; break; }
        if (is_nonclifford(ins.gate) && any_tainted)
            return "state-level injection: front-pushed magic supported on carried wires — a "
                   "non-Clifford gate acts on a qubit already entangled with a carried port; "
                   "prepare the magic state upstream of the carried-fresh entanglement "
                   "(prep-then-teleport), then this composes";
        if (any_tainted)                               // a multi-qubit gate spreads the taint
            for (int q : ins.targets) if (q >= 0 && q < Nc) tainted[(size_t)q] = 1;
    }
    return "";
}

// D2 — relabel a FramedSuperposition's qubit wires by the INJECTIVE map `map` (source wire w ->
// map[w] in a width-`new_n` space).  Frame row w moves to row map[w]; each row's column bits move
// w -> map[w]; eps and `free` follow.  Wires of the new space not hit by `map` are |0> (identity
// Z-stabiliser).  Used ONLY by the compile-time composition exactness guard (compare the mapped
// all-|0> oracle against the composed-with-|0> instance).
static FramedSuperposition remap_state_wires(const FramedSuperposition& s,
                                             const std::vector<int>& map, int new_n) {
    const int old_n = s.n();
    FramedSuperposition out(new_n);   // all-|0>: Xrow[a]=X_a, Zrow[a]=Z_a, eps 0, free empty
    auto move_pauli = [&](const Pauli& src) {
        Pauli dst(new_n);
        dst.phase = src.phase;
        for (int q = 0; q < old_n; ++q) {
            const int nq = (q < (int)map.size()) ? map[(size_t)q] : q;
            if (src.xbit(q)) dst.setx(nq);
            if (src.zbit(q)) dst.setz(nq);
        }
        return dst;
    };
    for (int a = 0; a < old_n; ++a) {
        const int na = (a < (int)map.size()) ? map[(size_t)a] : a;
        out.U.Xrow[(size_t)na] = move_pauli(s.U.Xrow[(size_t)a]);
        out.U.Zrow[(size_t)na] = move_pauli(s.U.Zrow[(size_t)a]);
        out.eps[(size_t)na] = s.eps[(size_t)a];
    }
    out.U.invalidate_dual();
    for (int f : s.free) out.free.push_back((f < (int)map.size()) ? map[(size_t)f] : f);
    out.sync_alpha_k();
    // amplitudes: sigma order follows s.free order, which maps 1:1 to out.free order (same
    // sequence, relabelled), so the (sigma, coeff) entries copy verbatim.
    out.entries() = s.entries();
    out.sync_alpha_k();
    return out;
}

// Remap every qubit index in a Circuit's instruction stream through `map` (map[w] = new index),
// and set the new width. Record indices (Measure order) are UNCHANGED — only wire indices move.
static void remap_circuit_wires(Circuit& c, const std::vector<int>& map, int new_n) {
    auto mp = [&](int q) -> int { return (q >= 0 && q < (int)map.size()) ? map[(size_t)q] : q; };
    for (Instr& ins : c.stream) {
        for (int& q : ins.targets) q = mp(q);
        for (int& q : ins.qubits) q = mp(q);
        for (PauliTerm& t : ins.obs) t.qubit = mp(t.qubit);
    }
    c.n = new_n;
}

// ── Twirl record sampler (V2-T4) ─────────────────────────────────────────────────────────
// Python-facing holder around qeccore::TwirlRecordSampler. The engine class holds a
// const& to the bare FramedSuperposition, so THIS object owns the bare state (plus the
// deferred circuit / read / record lists the constructor consumes) alongside the sampler.
//
// Semantics contract (docs/twirl_record_sampler.md):
//   * DETERMINISTIC detector/observable channels are EXACT (twirl law; 5σ-gated vs the
//     production sampler on all benchmarks incl. CH-class).
//   * GAUGE detector columns are DECLARED fair coins: independent Bernoulli(1/2). Gauge-
//     record marginals/correlations are TWIRLED in the V1 record product — this is the
//     spec's declared semantics, not an approximation claim.
//   * ANTI detectors refuse at compile (ValueError).
//   * OBSERVABLEs whose operator is not IN_GROUP have no exact record channel: sample()
//     raises RuntimeError naming them unless skip_refused_observables=True (then their
//     columns are all-zero and channel_report() lists them under "refused_observables").
//   * The plan-cache disk path is noise-blind: one cache file serves p-sweeps.

// Task 2: convert a PortVerdict to a Python dict with keys:
//   product (bool), k_port (int), stabilized (list of (q,axis,sign) tuples),
//   port_wires (list of int), witness (str).
static py::dict port_verdict_to_dict(const PortVerdict& v) {
    py::dict d;
    d["product"]    = v.product;
    d["k_port"]     = v.k_port;
    py::list stab;
    for (const ProdQ& pq : v.stabilized)
        stab.append(py::make_tuple((int)pq.q, (int)pq.axis, (int)pq.sign));
    d["stabilized"] = stab;
    py::list pw;
    for (int w : v.port_wires) pw.append(w);
    d["port_wires"] = pw;
    d["witness"]    = v.witness;
    return d;
}

class PyTwirlSampler {
public:
    static constexpr uint64_t kDefaultSeed = 1;

    PyTwirlSampler(const std::string& text, double p_factor, const std::string& disk_cache,
                   long selfcheck, bool skip_refused_observables,
                   const std::string& disk_cache_auto_dir)
        : skip_refused_(skip_refused_observables) {
        std::string err;
        {
            py::gil_scoped_release rel;
            // EXACTLY the bench driver's setup recipe (framed_bench.cpp main()):
            // parse → normalize{coherentize, defer, want_map, feedback=Strip} →
            // build_bare_state → FramedSuperposition::from_css → terminal reads.
            ParsedStim ps = parse_stim_circuit(text);
            if (!ps.ok()) {
                err = "parse failed";
                for (const auto& e : ps.errors)
                    err += "\n  line " + std::to_string(e.line) + ": " + e.message;
            } else {
                // Task 4: extract Pauli port qubits BEFORE normalize (indices preserved into deferred)
                for (const InputPort& p : ps.circuit.inputs)
                    if (p.kind == InputPort::Kind::Qubits) {
                        input_port_qubits_ = p.qubits;
                        break;
                    }
                NormalizePolicy pol;
                pol.coherentize = true;
                pol.defer = true;
                pol.want_map = true;
                pol.feedback = NormalizePolicy::Feedback::Strip;
                NormalizeResult nr = normalize(ps.circuit, pol);
                // V3: per-record readout flip/invert (deferral drops M(p)/invert; recover from
                // the coherent pre-strip circuit + XOR eliminate_hadamards' deterministic invert
                // from the deferred Measures — record order == terminal-read order).
                for (const Instr& ins : nr.coherent.stream)
                    if (ins.kind == Instr::Kind::Measure) {
                        opt_.rec_flip.push_back(ins.readout_flip_p);
                        opt_.rec_invert.push_back(ins.invert ? 1 : 0);
                    }
                { size_t j = 0;
                  for (const Instr& ins : nr.normalized.stream)
                      if (ins.kind == Instr::Kind::Measure) {
                          if (j < opt_.rec_invert.size() && ins.invert) opt_.rec_invert[j] ^= 1;
                          ++j;
                      } }
                deferred_ = std::move(nr.normalized);
                for (const auto& r : nr.map.terminal_reads)
                    reads_.push_back({(int)r.second, r.first});
                final_wire_ = nr.map.final_wire;              // Task 2b.2-pre
                outputs_ = ps.circuit.outputs;               // Task 2b.2-pre
                detectors_ = std::move(ps.detectors);
                obs_list_.assign(ps.observables.begin(), ps.observables.end());
                num_det_ = (int)detectors_.size();
                // Task 2b.1: forward declared DECISION parities to the engine (classified there).
                opt_.decisions = std::move(ps.decisions);
                num_dec_ = 0;
                for (const auto& kv : opt_.decisions)
                    if (kv.first + 1 > num_dec_) num_dec_ = kv.first + 1;
                int mx = -1;
                for (const auto& kv : obs_list_) if (kv.first > mx) mx = kv.first;
                num_obs_ = mx + 1;
                BareState bs = build_bare_state(deferred_);
                if (bs.rejected) {
                    err = "bare state rejected: " +
                          (bs.reject_reason.empty() ? std::string("unknown cause")
                                                    : bs.reject_reason);
                } else {
                    bare_.reset(new FramedSuperposition(FramedSuperposition::from_css(bs.state)));
                    opt_.p_factor = p_factor;
                    opt_.circuit_channels = true;
                    opt_.selfcheck = selfcheck;
                    opt_.disk_path = disk_cache;
                    // V3-T3 R2: automatic cache dir (xtim.twirl resolved env + mkdir); the
                    // engine derives <dir>/<signature-fnv>-<group-token>.twpl itself.
                    opt_.disk_auto_dir = disk_cache_auto_dir;
                    opt_.input_pauli_qubits = input_port_qubits_;  // Task 4: port qubit indices for frame relabel
                    err = rebuild(kDefaultSeed);
                }
            }
        }
        if (err.empty() && !sampler_->anti_detectors().empty()) {
            err = "ANTI detector(s)";
            for (int di : sampler_->anti_detectors()) err += " " + std::to_string(di);
            err += ": record operator anticommutes with the certified group — not a "
                   "well-formed detector; refused at compile";
        }
        if (!err.empty()) throw std::invalid_argument(err);   // -> ValueError
    }

    // S2.1 — construct from an EXTERNALLY PROVIDED bare FramedSuperposition.
    // Parses and normalizes `text` exactly as the circuit-only constructor (deferred_,
    // reads_, detectors_, obs_list_, rec_flip/invert are built the same way), but uses
    // `provided_state.state` verbatim as bare_ instead of calling build_bare_state.
    // Oracle: compile_from_state(_bare_state_of(C), C) is byte-identical to
    // compile_twirl_sampler(C) at the same seed (same bare + same deferred → same engine).
    PyTwirlSampler(const PyFramedSuperposition& provided_state,
                   const std::string& text, double p_factor,
                   const std::string& disk_cache, long selfcheck,
                   bool skip_refused_observables,
                   const std::string& disk_cache_auto_dir)
        : skip_refused_(skip_refused_observables) {
        std::string err;
        {
            py::gil_scoped_release rel;
            ParsedStim ps = parse_stim_circuit(text);
            if (!ps.ok()) {
                err = "parse failed";
                for (const auto& e : ps.errors)
                    err += "\n  line " + std::to_string(e.line) + ": " + e.message;
            } else {
                // Task 4: extract Pauli port qubits BEFORE normalize
                for (const InputPort& p : ps.circuit.inputs)
                    if (p.kind == InputPort::Kind::Qubits) {
                        input_port_qubits_ = p.qubits;
                        break;
                    }
                NormalizePolicy pol;
                pol.coherentize = true;
                pol.defer = true;
                pol.want_map = true;
                pol.feedback = NormalizePolicy::Feedback::Strip;
                NormalizeResult nr = normalize(ps.circuit, pol);
                // V3: per-record readout flip/invert (identical recovery to circuit-only ctor)
                for (const Instr& ins : nr.coherent.stream)
                    if (ins.kind == Instr::Kind::Measure) {
                        opt_.rec_flip.push_back(ins.readout_flip_p);
                        opt_.rec_invert.push_back(ins.invert ? 1 : 0);
                    }
                { size_t j = 0;
                  for (const Instr& ins : nr.normalized.stream)
                      if (ins.kind == Instr::Kind::Measure) {
                          if (j < opt_.rec_invert.size() && ins.invert) opt_.rec_invert[j] ^= 1;
                          ++j;
                      } }
                deferred_ = std::move(nr.normalized);
                for (const auto& r : nr.map.terminal_reads)
                    reads_.push_back({(int)r.second, r.first});
                final_wire_ = nr.map.final_wire;              // Task 2b.2-pre
                outputs_ = ps.circuit.outputs;               // Task 2b.2-pre
                detectors_ = std::move(ps.detectors);
                obs_list_.assign(ps.observables.begin(), ps.observables.end());
                num_det_ = (int)detectors_.size();
                // Task 2b.1: forward declared DECISION parities to the engine (classified there).
                opt_.decisions = std::move(ps.decisions);
                num_dec_ = 0;
                for (const auto& kv : opt_.decisions)
                    if (kv.first + 1 > num_dec_) num_dec_ = kv.first + 1;
                int mx = -1;
                for (const auto& kv : obs_list_) if (kv.first > mx) mx = kv.first;
                num_obs_ = mx + 1;
                // Use the PROVIDED state (copy) instead of build_bare_state.
                bare_.reset(new FramedSuperposition(provided_state.state));
                opt_.p_factor = p_factor;
                opt_.circuit_channels = true;
                opt_.selfcheck = selfcheck;
                opt_.disk_path = disk_cache;
                opt_.disk_auto_dir = disk_cache_auto_dir;
                opt_.input_pauli_qubits = input_port_qubits_;
                err = rebuild(kDefaultSeed);
            }
        }
        if (err.empty() && !sampler_->anti_detectors().empty()) {
            err = "ANTI detector(s)";
            for (int di : sampler_->anti_detectors()) err += " " + std::to_string(di);
            err += ": record operator anticommutes with the certified group — not a "
                   "well-formed detector; refused at compile";
        }
        if (!err.empty()) throw std::invalid_argument(err);   // -> ValueError
    }

    // Task 3.0 — MULTI-QUBIT CARRIED PORT via STATE-LEVEL INJECTION.
    // Build a sampler whose bare is the provided carried FramedSuperposition (magic and all)
    // tensor-extended with the consumer stage's fresh |0> qubits. Unlike the S2.1 verbatim
    // path (which needs the provided state to already BE the consumer's full bare), this path
    // carries an ENTANGLED / chi>=2 patch: the carried state stays VERBATIM (its amplitude
    // container is never surgically split), and the consumer's deferred circuit is relabeled so
    // its INPUT_QUBITS port wires reference the carried qubits (src_wires) and its fresh wires
    // reference freshly-grown |0> columns.
    //   carried     — the producer's per-shot collapsed FramedSuperposition (N_in wires).
    //   src_wires   — the carried qubits' wire indices in `carried` (its output_wires order),
    //                 mapping 1:1 to the consumer's INPUT_QUBITS declaration order.
    //   text        — the consumer stage text (declares INPUT_QUBITS on |src_wires| qubits).
    // Loud-refuses (ValueError): a consumer whose OWN bare has front-pushed FRESH magic
    // (k>0) — the fresh qubits must be |0> at bare time (spec §1: fresh qubits get |0> wires);
    // a port-size mismatch; a carried state too small for the referenced src_wires.
    PyTwirlSampler(const PyFramedSuperposition& carried, const std::vector<int>& src_wires,
                   const std::string& text, double p_factor,
                   const std::string& disk_cache, long selfcheck,
                   bool skip_refused_observables,
                   const std::string& disk_cache_auto_dir)
        : skip_refused_(skip_refused_observables) {
        std::string err;
        {
            py::gil_scoped_release rel;
            ParsedStim ps = parse_stim_circuit(text);
            if (!ps.ok()) {
                err = "parse failed";
                for (const auto& e : ps.errors)
                    err += "\n  line " + std::to_string(e.line) + ": " + e.message;
            } else {
                for (const InputPort& p : ps.circuit.inputs)
                    if (p.kind == InputPort::Kind::Qubits) {
                        input_port_qubits_ = p.qubits;
                        break;
                    }
                NormalizePolicy pol;
                pol.coherentize = true;
                pol.defer = true;
                pol.want_map = true;
                pol.feedback = NormalizePolicy::Feedback::Strip;
                NormalizeResult nr = normalize(ps.circuit, pol);
                for (const Instr& ins : nr.coherent.stream)
                    if (ins.kind == Instr::Kind::Measure) {
                        opt_.rec_flip.push_back(ins.readout_flip_p);
                        opt_.rec_invert.push_back(ins.invert ? 1 : 0);
                    }
                { size_t j = 0;
                  for (const Instr& ins : nr.normalized.stream)
                      if (ins.kind == Instr::Kind::Measure) {
                          if (j < opt_.rec_invert.size() && ins.invert) opt_.rec_invert[j] ^= 1;
                          ++j;
                      } }
                deferred_ = std::move(nr.normalized);
                for (const auto& r : nr.map.terminal_reads)
                    reads_.push_back({(int)r.second, r.first});
                final_wire_ = nr.map.final_wire;
                outputs_ = ps.circuit.outputs;
                detectors_ = std::move(ps.detectors);
                obs_list_.assign(ps.observables.begin(), ps.observables.end());
                num_det_ = (int)detectors_.size();
                opt_.decisions = std::move(ps.decisions);
                num_dec_ = 0;
                for (const auto& kv : opt_.decisions)
                    if (kv.first + 1 > num_dec_) num_dec_ = kv.first + 1;
                int mx = -1;
                for (const auto& kv : obs_list_) if (kv.first > mx) mx = kv.first;
                num_obs_ = mx + 1;

                // ── The consumer's OWN bare (all wires |0>) — the composition oracle. ──
                // build_bare_state sees the port qubits as |0>; χ>1 here is FRESH-qubit magic,
                // which D2 COMPOSES: composed bare = (carried on ports) ⊗ (fresh magic block),
                // then the port-touching Cliffords baked (spec §6.1 fresh-magic-in-consumer). The
                // all-|0> bare `cbs` is the exact oracle the composition is verified against below.
                BareState cbs = build_bare_state(deferred_);
                if (cbs.rejected) {
                    err = "bare state rejected: " +
                          (cbs.reject_reason.empty() ? std::string("unknown cause")
                                                     : cbs.reject_reason);
                } else if ((int)input_port_qubits_.size() != (int)src_wires.size()) {
                    err = "state-level injection: consumer INPUT_QUBITS declares " +
                          std::to_string(input_port_qubits_.size()) + " port qubit(s) but the "
                          "carried state provides " + std::to_string(src_wires.size()) +
                          " (src_wires) — a Level-2 interface mismatch";
                } else {
                    const int N_in = carried.state.n();
                    bool src_ok = true;
                    for (int w : src_wires) if (w < 0 || w >= N_in) src_ok = false;
                    if (!src_ok) {
                        err = "state-level injection: a src_wire is out of range for the carried "
                              "state (n=" + std::to_string(N_in) + ")";
                    } else {
                        // Build the wire remap: consumer deferred wire -> combined-space wire.
                        //   port wire input_port_qubits_[j] -> src_wires[j]        (into [0, N_in))
                        //   every other (FRESH) deferred wire w -> N_in + w         (its column in
                        //     the tensored fresh block; the block is width Nc, its own port columns
                        //     ride along as dead |0> spectators — never referenced post-remap).
                        const int Nc = deferred_.n;
                        std::vector<uint8_t> is_port_orig((size_t)(Nc > 0 ? Nc : 1), 0);
                        std::vector<int> map((size_t)Nc, -1);
                        for (size_t j = 0; j < input_port_qubits_.size(); ++j) {
                            const int pw = input_port_qubits_[j];
                            if (pw < 0 || pw >= Nc) { err = "port qubit out of deferred range"; break; }
                            map[(size_t)pw] = src_wires[j];
                            is_port_orig[(size_t)pw] = 1;
                        }
                        for (int w = 0; w < Nc && err.empty(); ++w)
                            if (map[(size_t)w] < 0) map[(size_t)w] = N_in + w;   // fresh column
                        // §6.1 narrow refusal (state-blind): no front-pushed magic on carried wires.
                        if (err.empty())
                            err = check_no_magic_on_carried(deferred_, is_port_orig, Nc);
                        if (err.empty()) {
                            const int combined_n = N_in + Nc;
                            // ── Build the FRESH MAGIC BLOCK (χ = 2^k) via the ONE unmodified
                            //    build_bare_state path, on the fresh-only sub-circuit (port wires
                            //    and every port-touching gate removed; fresh wires reset to |0>). ──
                            Circuit fresh_sub = fresh_only_subcircuit(deferred_, is_port_orig, Nc);
                            BareState fbs = build_bare_state(fresh_sub);
                            if (fbs.rejected) {
                                err = "state-level injection: fresh magic block rejected: " +
                                      (fbs.reject_reason.empty() ? std::string("unknown cause")
                                                                 : fbs.reject_reason);
                            } else {
                                FramedSuperposition freshblock =
                                    FramedSuperposition::from_css(fbs.state);
                                // composed initial state = carried ⊗ freshblock (χ multiplies).
                                bare_.reset(new FramedSuperposition(
                                    tensor_framed(carried.state, freshblock)));
                                remap_circuit_wires(deferred_, map, combined_n);
                                // Build the combined-space port mask for the port-only baker.
                                std::vector<uint8_t> is_port_comb((size_t)combined_n, 0);
                                for (int sw : src_wires)
                                    if (sw >= 0 && sw < combined_n) is_port_comb[(size_t)sw] = 1;
                                // Bake ONLY the port-touching Cliffords into the composed bare
                                // (fresh-only Cliffords + the magic are already in freshblock).
                                err = apply_stream_cliffords_ports(*bare_, deferred_, is_port_comb);

                                // ── Composition exactness guard (compile-time, per pattern). ──
                                // Re-run the SAME tensor+bake with the carried subsystem replaced by
                                // |0>^N_in; the result must be byte-identical to the all-|0> oracle
                                // `cbs` (build_bare_state on the full consumer, port wires |0>),
                                // restricted to the live wires.  This certifies the fresh/port SPLIT
                                // + bake ORDER is exact for THIS circuit; the carried state then
                                // rides through the identical transformation.  Any unsound
                                // gate-ordering (a fresh-only Clifford sequenced after a port gate)
                                // fails here and refuses loudly — never silent-wrong.
                                if (err.empty()) {
                                    FramedSuperposition chk =
                                        tensor_framed(zero_state(N_in), freshblock);
                                    std::string cerr =
                                        apply_stream_cliffords_ports(chk, deferred_, is_port_comb);
                                    if (!cerr.empty()) {
                                        err = cerr;
                                    } else {
                                        // `chk` = composed bare with carried = |0>^N_in, on
                                        // combined_n wires (consumer wire w lives at map[w]).  The
                                        // oracle `cbs` = build_bare_state on the full consumer
                                        // (all |0>), on Nc wires.  Map the oracle through the SAME
                                        // `map` into the combined space and require byte-equality:
                                        // this certifies the split + bake ORDER for THIS circuit.
                                        FramedSuperposition oracle =
                                            FramedSuperposition::from_css(cbs.state);
                                        FramedSuperposition oracle_mapped =
                                            remap_state_wires(oracle, map, combined_n);
                                        auto ov = exact_sum_overlap(chk, oracle_mapped);
                                        if (std::abs(std::abs(ov) - 1.0) > 1e-9) {
                                            err = "state-level injection: fresh-magic composition "
                                                  "failed its exactness guard (the fresh/port split "
                                                  "or bake order is unsound for this stage — a "
                                                  "fresh-only Clifford is sequenced after a "
                                                  "port-touching gate); refused";
                                        }
                                    }
                                }
                            }
                            if (err.empty()) {
                            for (auto& r : reads_) r.second = map[(size_t)r.second];
                            for (int& fw : final_wire_)
                                if (fw >= 0 && fw < Nc) fw = map[(size_t)fw];
                            std::vector<int> remapped_ports;
                            for (int pw : input_port_qubits_) remapped_ports.push_back(map[(size_t)pw]);
                            input_port_qubits_ = remapped_ports;   // now src_wires (frame relabel site)

                            opt_.p_factor = p_factor;
                            opt_.circuit_channels = true;
                            opt_.selfcheck = selfcheck;
                            opt_.disk_path = disk_cache;
                            opt_.disk_auto_dir = disk_cache_auto_dir;
                            opt_.input_pauli_qubits = input_port_qubits_;
                            err = rebuild(kDefaultSeed);
                            }
                        }
                    }
                }
            }
        }
        if (err.empty() && !sampler_->anti_detectors().empty()) {
            err = "ANTI detector(s)";
            for (int di : sampler_->anti_detectors()) err += " " + std::to_string(di);
            err += ": record operator anticommutes with the certified group — not a "
                   "well-formed detector; refused at compile";
        }
        if (!err.empty()) throw std::invalid_argument(err);   // -> ValueError
    }

    // sample(shots, seed=None) -> (detectors, observables), both Stim-b8-packed uint8
    // numpy arrays of shape (shots, ceil(K/8)). seed=None continues the sampler's current
    // stream; an explicit seed deterministically RESEEDS the sampler IN PLACE (speed-kill T1:
    // byte-identical to the historical rebuild, without the ~180 ms reconstruction).
    // BYTE-DETERMINISM CONTRACT (final-review Important #2): same seed -> same bytes holds
    // at FIXED plan-cache state. With disk_cache set, run() enriches the cache file, and a
    // later rebuild pre-warms plans whose shots previously took the slow path — the collapse
    // RNG is consumed differently on the fast path (64-bit coin reservoir) than on the slow
    // path (one draw per coin), so bytes may differ across calls while every DISTRIBUTION is
    // identical. Without disk_cache, repeated sample(N, seed=s) on one object is byte-stable
    // (each rebuild starts from an empty in-memory cache).
    // NOT THREAD-SAFE: the GIL is released during run(); do not call sample()/channel_report()
    // from another thread concurrently.
    py::tuple sample(long shots, py::object seed_obj, py::object input_pauli_obj = py::none()) {
        if (shots < 0) throw std::invalid_argument("shots must be >= 0");
        if (shots > (long)INT32_MAX)
            throw std::invalid_argument("shots exceeds INT32_MAX (b8 array shape is int)");
        if (sampler_->setup_error() != 0)
            throw std::runtime_error("twirl sampler in a failed-setup state (code " +
                                     std::to_string(sampler_->setup_error()) + ")");
        if (!skip_refused_ && !sampler_->refused_observables().empty()) {
            std::string msg = "observable index(es)";
            for (int oi : sampler_->refused_observables()) msg += " " + std::to_string(oi);
            msg += ": operator is not IN_GROUP (logical) — no exact record channel exists "
                   "in the twirl product; pass skip_refused_observables=True to "
                   "compile_twirl_sampler to emit all-zero columns for them";
            throw std::runtime_error(msg);
        }
        const bool reseed = !seed_obj.is_none();
        const uint64_t seed = reseed ? py::cast<uint64_t>(seed_obj) : 0;
        const int DBB = PackedRecords::bytes_per_shot(num_det_);
        const int OBB = PackedRecords::bytes_per_shot(num_obs_);
        const int DECB = PackedRecords::bytes_per_shot(num_dec_);   // Task 2b.1
        std::vector<uint8_t> dbuf, obuf, decbuf;
        std::string err;
        bool oracle_ok = true;
        // Task 4: Input frame validation + setup.
        // frame_arr must be declared HERE (outside GIL release) to keep the numpy buffer alive
        // during run(). The raw pointer is extracted before GIL release, then set_input_frame
        // is called AFTER rebuild (rebuild creates a new sampler — setting before rebuild is lost).
        py::array_t<uint8_t> frame_arr;
        const uint8_t* frame_ptr = nullptr;
        int frame_stride = 0;
        const bool has_frame = prepare_input_frame(
            input_pauli_obj, shots, frame_arr, frame_ptr, frame_stride);
        {
            py::gil_scoped_release rel;
            if (reseed) err = reseed_in_place(seed);   // speed-kill T1: no reconstruction
            if (err.empty()) {
                // Task 4: set frame AFTER the reseed (a rebuild fallback creates a fresh sampler).
                if (has_frame) sampler_->set_input_frame(frame_ptr, frame_stride);
                dbuf.assign((size_t)shots * DBB, 0);
                obuf.assign((size_t)shots * OBB, 0);
                decbuf.assign((size_t)shots * DECB, 0);
                const std::vector<int>& cdet = sampler_->channel_detectors();
                const std::vector<int>& cobs = sampler_->channel_observables();
                const std::vector<uint8_t>& cref = sampler_->channel_refs();
                const std::vector<int>& gdet = sampler_->gauge_detectors();
                const int ndetc = (int)cdet.size();
                const int ndec = num_dec_;
                const int born_oi = sampler_->born_obs_index();
                // Build DetsPackSink: pre-compute identity check + packed cref
                DetsPackSink dpk;
                dpk.dets = dbuf.data(); dpk.DBB = DBB;
                dpk.obs  = obuf.data(); dpk.OBB = OBB;
                dpk.dec  = decbuf.data(); dpk.DECB = DECB;
                dpk.cdet = cdet.data(); dpk.ndetc = ndetc;
                dpk.cobs = cobs.data(); dpk.ncobs = (int)cobs.size();
                dpk.cref = cref.data();
                dpk.gdet = gdet.data(); dpk.ngdet = (int)gdet.size();
                dpk.ndec = ndec;
                dpk.born_oi = born_oi;
                dpk.gauge_rng = &gauge_rng_;
                dpk.row = 0;
                // Check identity-contiguous: cdet[c] == c for all c
                dpk.det_identity = true;
                for (int c = 0; c < ndetc; ++c)
                    if (cdet[(size_t)c] != c) { dpk.det_identity = false; break; }
                // Pack cref into uint64 words for the fast path
                dpk.det_words = (ndetc + 63) / 64;
                dpk.det_partial_bits = ndetc % 64;   // 0 = last word fully used (all 64 valid)
                dpk.cref_det_packed.assign((size_t)dpk.det_words, 0ULL);
                for (int c = 0; c < ndetc; ++c)
                    if (cref[(size_t)c])
                        dpk.cref_det_packed[(size_t)(c >> 6)] |= (1ULL << (c & 63));
                last_sink_kind_ = "dets_pack";
                sampler_->set_dets_pack_sink(&dpk);
                struct DpkGuard {
                    TwirlRecordSampler* s;
                    ~DpkGuard() { s->set_dets_pack_sink(nullptr); }
                } dg{sampler_.get()};
                struct FrameGuard {                    // clear frame even if run() throws
                    TwirlRecordSampler* s;
                    ~FrameGuard() { s->set_input_frame(nullptr, 0); }
                } fg{sampler_.get()};
                oracle_ok = sampler_->run(shots);
            }
        }
        if (!err.empty()) throw std::runtime_error(err);
        if (!oracle_ok)
            throw std::runtime_error("twirl sampler oracle abort (details on stderr)");
        // Task 2b.1: circuits WITH DECISIONs return a 3-tuple (dets, obs, decisions); decision-
        // free circuits keep the exact (dets, obs) 2-tuple (byte-identical to the pre-2b.1 API).
        if (num_dec_ > 0)
            return py::make_tuple(packed_to_numpy(dbuf, (int)shots, DBB),
                                  packed_to_numpy(obuf, (int)shots, OBB),
                                  packed_to_numpy(decbuf, (int)shots, DECB));
        return py::make_tuple(packed_to_numpy(dbuf, (int)shots, DBB),
                              packed_to_numpy(obuf, (int)shots, OBB));
    }

    // S2.2/S2.4: sample_barrier — run shots, retaining each shot's REAL collapsed
    // post-barrier state + σ AND the packed detector/observable bits, ALL from the
    // SAME single run() pass.
    //
    // Retention (set_retain_state(true)) forces the slow per-shot path: run() calls
    // twirl_collapse(need_amps=true), so the sink receives amps = the residual-applied,
    // kernel-collapsed FramedSuperposition (NOT *bare_ — the fake-green trap). The state is
    // genuinely per-shot: shots whose fired errors give different residuals (or different
    // Born outcomes) collapse to DIFFERENT states. We COPY amps (the pointer aliases a reused
    // working state) and the σ bit-list into the buffer.
    //
    // S2.4 single-pass: the sink ALSO packs the detector/observable bits using the SAME
    // `bits` word the state is derived from, with the same gauge_rng_ draw order as
    // sample().  This eliminates the dual-sample misalignment that arises from calling
    // sample_barrier + sample separately (different RNG paths for retain vs fast-abelian).
    // Only the DIAGONAL residual class is supported (run() refuses CH/PPR circuits under
    // retention).
    // port-v3 T2: `compact` (default False — old callers byte- and cost-identical) switches the
    // record store to the COMPACT plan identity (plan_id + prefix words + per-plan suffix blob)
    // instead of per-shot serialized plan bytes. The sampled STREAMS (dets/obs/decisions/σ/coins)
    // are byte-identical either way (the compact path draws nothing and skips only record-key
    // serialization work); record_groups()/record_keys()/plan_key() reconstruct the identical
    // legacy bytes. Used by the xtim.port record route.
    PyBarrierBuffer sample_barrier(long shots, py::object seed_obj,
                                   py::object input_pauli_obj = py::none(),
                                   bool compact = false) {
        if (shots < 0) throw std::invalid_argument("shots must be >= 0");
        if (shots > (long)INT32_MAX)
            throw std::invalid_argument("shots exceeds INT32_MAX");
        if (sampler_->setup_error() != 0)
            throw std::runtime_error("twirl sampler in a failed-setup state (code " +
                                     std::to_string(sampler_->setup_error()) + ")");

        const bool reseed = !seed_obj.is_none();
        const uint64_t seed = reseed ? py::cast<uint64_t>(seed_obj) : 0;
        // Task 2b.4 (Important 1): input Pauli frame relabel on the RETENTION path — same
        // validation + machinery as sample(). frame_arr must outlive the GIL release / run().
        // The relabel flips the packed detector/observable/decision RECORD bits (apply_input_frame
        // is called on the slow retention path); the retained collapsed state itself is NOT frame-
        // mutated (the orchestrator carries the frame forward classically).
        py::array_t<uint8_t> frame_arr;
        const uint8_t* frame_ptr = nullptr;
        int frame_stride = 0;
        const bool has_frame = prepare_input_frame(
            input_pauli_obj, shots, frame_arr, frame_ptr, frame_stride);
        PyBarrierBuffer buf;
        std::string err;
        bool oracle_ok = true;
        // n_gens (σ bit width) = number of certified generators of the bare state.
        const int ngens = (int)bare_->certified_stabilizers().size();
        // Channel packing dimensions — mirror sample() exactly.
        const int DBB = PackedRecords::bytes_per_shot(num_det_);
        const int OBB = PackedRecords::bytes_per_shot(num_obs_);
        const int DECB = PackedRecords::bytes_per_shot(num_dec_);   // Task 2b.1
        buf.DBB_ = DBB;
        buf.OBB_ = OBB;
        buf.DECB_ = DECB;

        {
            py::gil_scoped_release rel;
            if (reseed) err = reseed_in_place(seed);   // speed-kill T1: no reconstruction
            if (err.empty()) {
                // Task 2b.4: set the input frame AFTER the reseed (a rebuild fallback makes a
                // fresh sampler).
                if (has_frame) sampler_->set_input_frame(frame_ptr, frame_stride);
                buf.sampler_ = sampler_;   // shared_ptr copy — outlives per-group sampler scope
                buf.nshots_ = (int)shots;
                // barrier2 Lever 3: σ is a FIXED stride of SGW words/shot (bulk memcpy, no per-shot
                // byte expansion). Size it once here so the sink never reallocates.
                const int SGW = (ngens + 63) / 64;
                buf.SGW_ = SGW;
                buf.sig_bits_ = ngens;
                buf.sig_words_.assign((size_t)shots * (size_t)SGW, 0);
                // Flat CSR: reserve the offset arrays (nshots+1) and a rough data estimate so the
                // append path never reallocates in the hot loop.
                buf.coin_off_.reserve((size_t)shots + 1);
                buf.pk_off_.reserve((size_t)shots + 1);
                buf.bu_off_.reserve((size_t)shots + 1);
                buf.dets_packed_.assign((size_t)shots * DBB, 0);
                buf.obs_packed_.assign((size_t)shots * OBB, 0);
                buf.dec_packed_.assign((size_t)shots * DECB, 0);   // Task 2b.1
                // Cache channel info for bit packing (same references as sample()).
                const std::vector<int>& cdet = sampler_->channel_detectors();
                const std::vector<int>& cobs = sampler_->channel_observables();
                const std::vector<uint8_t>& cref = sampler_->channel_refs();
                const std::vector<int>& gdet = sampler_->gauge_detectors();
                const int ndec = num_dec_;
                const int born_oi = sampler_->born_obs_index();
                sampler_->set_retain_state(true);
                // Fix 2: DEVIRTUALISED sink. Populate a POD BarrierSink with raw pointers to the
                // (pre-sized, non-reallocating) fixed buffers, POINTERS TO the growable CSR blob
                // vectors (reallocation-safe), borrowed channel metadata, and a REFERENCE to the
                // gauge coin generator (drawn in the SAME order as sample()). Byte-for-byte port of
                // the old std::function sink body (now in BarrierSink::emit). `bs` must outlive
                // run() — it lives to the end of this block, after run() returns.
                BarrierSink bs;
                bs.dets = DBB ? buf.dets_packed_.data() : nullptr;  bs.DBB = DBB;
                bs.obs  = OBB ? buf.obs_packed_.data()  : nullptr;  bs.OBB = OBB;
                bs.dec  = DECB ? buf.dec_packed_.data() : nullptr;  bs.DECB = DECB;
                bs.sig_words = buf.sig_words_.data();               bs.SGW = SGW;
                bs.coin_data = &buf.coin_data_;  bs.coin_off = &buf.coin_off_;
                bs.pk_data   = &buf.pk_data_;    bs.pk_off   = &buf.pk_off_;
                bs.bu_data   = &buf.bu_data_;    bs.bu_off   = &buf.bu_off_;
                bs.cdet = cdet.data();  bs.ndetc = (int)cdet.size();
                bs.cobs = cobs.data();  bs.ncobs = (int)cobs.size();
                bs.cref = cref.data();
                bs.gdet = gdet.data();  bs.ngdet = (int)gdet.size();
                bs.ndec = ndec;         bs.born_oi = born_oi;
                bs.gauge_rng = &gauge_rng_;
                bs.row = 0;
                // port-v3 T2: compact plan-identity store (pre-sized fixed-stride arrays; the
                // suffix blob grows once per DISTINCT plan). Legacy mode leaves all of this
                // untouched — the sink's pk CSR path is byte-for-byte the historical one.
                if (compact) {
                    const int nW = sampler_->deferred_wires();
                    const int PNW = (nW + 63) / 64;
                    buf.compact_ = true;
                    buf.PNW_ = PNW;
                    buf.plan_id_.assign((size_t)shots, 0);
                    buf.prefix_words_.assign((size_t)shots * (size_t)(2 * PNW), 0);
                    bs.compact = true;
                    bs.plan_id = buf.plan_id_.data();
                    bs.prefix_words = buf.prefix_words_.data();
                    bs.PNW = PNW;
                    bs.n_wires = nW;
                    bs.suffix_data = &buf.suffix_data_;
                    bs.suffix_off = &buf.suffix_off_;
                }
                last_sink_kind_ = "barrier";
                sampler_->set_barrier_sink(&bs);
                struct SinkGuard {
                    TwirlRecordSampler* s;
                    ~SinkGuard() { s->set_barrier_sink(nullptr);
                                   s->set_retain_state(false); }
                } sg{sampler_.get()};
                struct FrameGuard {                    // clear frame even if run() throws
                    TwirlRecordSampler* s;
                    ~FrameGuard() { s->set_input_frame(nullptr, 0); }
                } fg{sampler_.get()};
                oracle_ok = sampler_->run(shots);
            }
        }

        if (!err.empty()) throw std::invalid_argument(err);
        if (!oracle_ok)
            throw std::runtime_error(
                "twirl sampler oracle abort during sample_barrier (details on stderr; "
                "state retention requires a diagonal-class circuit)");
        return buf;
    }

    py::dict channel_report() const {
        py::dict d;
        d["deterministic_detectors"] = sampler_->channel_detectors();
        d["gauge_detectors"] = sampler_->gauge_detectors();
        d["anti_detectors"] = sampler_->anti_detectors();
        d["refused_observables"] = sampler_->refused_observables();
        d["deterministic_observables"] = sampler_->channel_observables();
        d["num_channels"] = sampler_->num_channels();
        d["plans"] = sampler_->plans();
        d["plan_hits"] = sampler_->plan_hits();
        d["plan_misses"] = sampler_->plan_misses();
        d["ppr_plans"] = sampler_->ppr_plans();
        // V3-T3 R1: exact_shots = shots computed by the per-shot exact engine (INCLUDED in
        // used/rows — a default sampler never drops a shot). fallback_shots kept as an alias
        // of the same diagnostic count for report compatibility.
        d["fallback_shots"] = sampler_->exact_shots();
        d["exact_shots"] = sampler_->exact_shots();
        d["used"] = sampler_->used();
        d["disk_path"] = sampler_->disk_path();
        d["sink_kind"] = last_sink_kind_;   // I3: "dets_pack", "barrier", or "null" (before first run)
        return d;
    }

    // C1: channel_one_counts — per-channel fired-count vector (nonzero only when
    // QEC_TW_COUNT=1 or selfcheck>0; all-zero on the fast path). Gate: the default
    // (selfcheck=0) path must produce all-zero counts.
    py::list channel_one_counts() const {
        const auto& v = sampler_->channel_one_counts();
        py::list out;
        for (auto x : v) out.append(x);
        return out;
    }

    int num_detectors() const { return num_det_; }
    int num_observables() const { return num_obs_; }
    int num_decisions() const { return num_dec_; }   // Task 2b.1

    // Task 2b.2-pre: deferred-space wire indices of the declared OUTPUT_QUBITS, flattened in
    // declaration order. The retained (per-shot collapsed) barrier state is indexed in
    // deferred wire space; a declared output qubit `orig` lives at final_wire_[orig]. Reading
    // pauli_expectation_x/z at these indices on buf.state(i) returns the PHYSICAL frame value
    // (the bare bakes each surviving qubit's pushed Clifford frame in at its final wire).
    std::vector<int> output_wires() const {
        std::vector<int> out;
        for (const OutputPort& p : outputs_)
            for (int q : p.qubits) {
                if (q < 0 || q >= (int)final_wire_.size())
                    throw std::runtime_error("output qubit " + std::to_string(q) +
                                             " out of final_wire range");
                out.push_back(final_wire_[(size_t)q]);
            }
        return out;
    }

    // ── port-v3 T2: Segment-level σ-law/plan-structure read model (no buffer required) ─────
    // The plan structure and the Born-decision operators are pure functions of the COMPILED
    // circuit (BarrierBuffer merely forwarded them to its source sampler). Exposing them here
    // lets xtim.port build its Segment read model at compile time; the emitted dicts are
    // byte-identical to the BarrierBuffer accessors (same shared builders).
    py::dict plan_structure(py::bytes plan_key_bytes) const {
        return plan_structure_dict(*sampler_,
                                   (int)bare_->certified_stabilizers().size(),
                                   plan_key_bytes.cast<std::string>());
    }
    py::dict born_dec() const { return born_dec_dict(*sampler_); }
    int sig_bits() const { return (int)bare_->certified_stabilizers().size(); }

private:
    // Task 2b.4 (Important 2): validate + set up a per-shot input Pauli frame. Shared by
    // sample() and sample_barrier() so the loud-refuse + shape-check are identical.
    //   * input_pauli_obj None            -> returns false (no frame; byte-identical hot path).
    //   * supplied but NO INPUT_QUBITS port -> ValueError (never a silent no-op).
    //   * supplied with a port             -> shape-checked (shots, 2*support); returns true,
    //     with frame_ptr/frame_stride set. `frame_arr` (kept alive by the caller) owns the buffer.
    bool prepare_input_frame(py::object input_pauli_obj, long shots,
                             py::array_t<uint8_t>& frame_arr,
                             const uint8_t*& frame_ptr, int& frame_stride) {
        if (input_pauli_obj.is_none()) return false;
        if (input_port_qubits_.empty())
            throw std::invalid_argument(
                "input_pauli supplied but circuit declares no INPUT_QUBITS port");
        frame_arr = input_pauli_obj.cast<py::array_t<uint8_t>>();
        auto buf = frame_arr.request();
        const int expected_cols = (int)input_port_qubits_.size() * 2;
        if (buf.ndim != 2 || buf.shape[0] != shots || buf.shape[1] != expected_cols)
            throw std::runtime_error(
                "input_pauli: expected shape (" + std::to_string(shots) + ", " +
                std::to_string(expected_cols) + "), got (" +
                std::to_string(buf.ndim == 2 ? (long long)buf.shape[0] : -1LL) + ", " +
                std::to_string(buf.ndim == 2 ? (long long)buf.shape[1] : -1LL) + ")");
        frame_ptr = static_cast<const uint8_t*>(buf.ptr);
        frame_stride = (int)(buf.strides[0] / buf.itemsize);  // Task 4 R1: byte stride, not col count
        return true;
    }

    // speed-kill T1: the seeded sample()/sample_barrier() path — reseed the EXISTING engine
    // IN PLACE (TwirlRecordSampler::set_seed, ~µs) instead of reconstructing it. The rebuild
    // was a ~180 ms circuit-compile-scale fixed cost per seeded call on cultivation d5
    // (propagation table + planes group + channel classification — all pure functions of the
    // circuit, not the seed). Byte contract: set_seed resets exactly the seed-derived engine
    // state a fresh construction initializes, and gauge_rng_ is re-tied to the seed exactly
    // as rebuild() does — every emitted stream is byte-identical to a fresh rebuild at
    // `seed` (oracle: tests/test_set_seed_equivalence.py; zero tolerance). rebuild() is
    // RETAINED for construction and as the fallback when the engine refuses an in-place
    // reseed (synthetic-channel mode — unreachable from these bindings, which always compile
    // circuit channels).
    std::string reseed_in_place(uint64_t seed) {
        if (sampler_ && sampler_->setup_error() == 0 && sampler_->set_seed(seed)) {
            gauge_rng_.seed(seed ^ 0xD1B54A32D192ED03ull);   // same seed-tie as rebuild()
            return "";
        }
        // set_seed refused (synthetic-channel mode — unreachable from the public Python
        // constructors, which always set circuit_channels=true). This path is unexpected
        // in production; emit a loud note so it surfaces during debugging.
        std::fprintf(stderr,
            "[xtim] reseed_in_place: set_seed refused (synthetic-channel mode?); "
            "falling back to full rebuild (seed=%llu) — this should not happen on "
            "the circuit-channel path\n",
            (unsigned long long)seed);
        return rebuild(seed);
    }

    // (Re)construct the engine sampler at `seed`. Returns "" on success. The engine holds
    // a const& to *bare_ — destroy the old sampler before any rebuild, never move bare_.
    // Perf fix: shared plan caches are created ONCE on first rebuild and reused thereafter.
    // Safety guard: if the new sampler's G.identity_token() differs from the token the
    // caches were first populated under, the caches are CLEARED before reuse (loud-safe:
    // a different token means a different compiled circuit — stale plans must not be served).
    std::string rebuild(uint64_t seed) {
        // Lazy cache creation: allocate once (not at construction, since the circuit may not
        // be fully set up yet when the constructors call rebuild the first time).
        if (!shared_caches_.plans) {
            shared_caches_.plans = std::make_shared<TwirlPlanCache>();
            shared_caches_.ppr   = std::make_shared<PprPlanCache>();
            // Shared fast-route index: same lifetime as the plan caches. Warm rebuilds adopt its
            // pre-populated lkeys/lplans/lidx/ftab and skip materialize+canon on every shot
            // outside the selfcheck window. plan_misses counts only genuine law builds.
            shared_caches_.fidx  = std::make_shared<SharedFastIndex>();
        }
        // sampler_.reset() destroys the old Impl which writes lkeys/lplans/lidx/ftab back into
        // shared_caches_.fidx (via ~Impl); the new Impl then adopts them below.
        sampler_.reset();
        sampler_.reset(new TwirlRecordSampler(*bare_, deferred_, reads_, detectors_,
                                              obs_list_, seed, opt_, shared_caches_));
        gauge_rng_.seed(seed ^ 0xD1B54A32D192ED03ull);   // gauge-coin stream, seed-tied
        if (sampler_->setup_error() != 0) {
            // Reset shared caches on setup failure so a later successful rebuild starts fresh.
            shared_caches_.plans = std::make_shared<TwirlPlanCache>();
            shared_caches_.ppr   = std::make_shared<PprPlanCache>();
            shared_caches_.fidx  = std::make_shared<SharedFastIndex>();
            cache_initialized_ = false;
            switch (sampler_->setup_error()) {
                case 2: return "twirl sampler setup failed: no deterministic detector channels";
                case 3: return "twirl sampler setup failed: propagation table not all-in-class "
                               "or unsupported noise channel";
                default:
                    return "twirl sampler setup failed (code " +
                           std::to_string(sampler_->setup_error()) + ", details on stderr)";
            }
        }
        // Token guard: ensure the shared caches belong to THIS circuit's group.
        const uint64_t tok = sampler_->group_token();
        if (!cache_initialized_) {
            cache_token_ = tok;
            cache_initialized_ = true;
        } else if (tok != cache_token_) {
            // Different circuit: clear and re-adopt fresh caches so no stale plan is served.
            shared_caches_.plans = std::make_shared<TwirlPlanCache>();
            shared_caches_.ppr   = std::make_shared<PprPlanCache>();
            shared_caches_.fidx  = std::make_shared<SharedFastIndex>();
            cache_token_ = tok;
            // Rebuild with the fresh caches so this sampler uses them too.
            sampler_.reset();
            sampler_.reset(new TwirlRecordSampler(*bare_, deferred_, reads_, detectors_,
                                                  obs_list_, seed, opt_, shared_caches_));
            gauge_rng_.seed(seed ^ 0xD1B54A32D192ED03ull);
        }
        return "";
    }

    // Owned storage the engine class references / consumed at construction.
    Circuit deferred_;
    std::vector<std::pair<int, int>> reads_;
    std::vector<std::vector<int>> detectors_;
    std::vector<std::pair<int, std::vector<int>>> obs_list_;
    std::unique_ptr<FramedSuperposition> bare_;
    TwirlRecordOptions opt_;
    std::shared_ptr<TwirlRecordSampler> sampler_;
    std::mt19937_64 gauge_rng_;
    int num_det_ = 0, num_obs_ = 0, num_dec_ = 0;
    bool skip_refused_ = false;
    std::vector<int> input_port_qubits_;   // Task 4: Pauli input port qubit support (original indices)
    // Task 2b.2-pre: end-of-circuit wire placement (final_wire[orig] = deferred wire holding
    // original wire `orig`) + declared OUTPUT_QUBITS, so a consumer can read a declared output
    // qubit's PHYSICAL X/Z on the retained collapsed state at its correct deferred index.
    std::vector<int> final_wire_;
    std::vector<OutputPort> outputs_;
    // ── Perf fix: shared plan caches survive seed-driven rebuild() calls ──────────────────────
    // PyTwirlSampler is per-compiled-circuit; rebuild() is called on every sample() to replay
    // the deterministic seed stream. Without sharing, each rebuild starts with empty caches and
    // re-derives every plan. With sharing, pass-0 builds plans into the shared cache; pass 1+
    // find them with zero plan-build cost (warm path: ~0.5-1.0 us/shot vs ~5.9 us/shot before).
    // Safety: reuse is valid for the SAME compiled circuit. PyTwirlSampler is per-circuit, so
    // this holds structurally. An extra guard: if rebuild() sees a different G.identity_token()
    // (can only happen if someone makes rebuild() recompile a different circuit), the cache is
    // CLEARED rather than reused — loud-safe beats latent corruption.
    SharedPlanCaches shared_caches_;       // constructed ONCE at first rebuild(); reused thereafter
    uint64_t cache_token_ = 0;            // G.identity_token() the cache was first populated under
    bool cache_initialized_ = false;      // true after the first successful rebuild
    std::string last_sink_kind_ = "null"; // I3: "dets_pack", "barrier", or "null" (before first run)
};

// ── S2.1: _bare_state_of — build the FramedSuperposition bare state of a circuit ─────────
// Replicates the normalization + build_bare_state path from PyTwirlSampler's circuit-only
// constructor and returns the result as a PyFramedSuperposition. Used by
// compile_twirl_sampler_from_state to let callers inject a pre-built (possibly collapsed)
// state instead of rebuilding it from the deferred circuit each time.
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
            NormalizePolicy pol;
            pol.coherentize = true;
            pol.defer = true;
            pol.want_map = true;
            pol.feedback = NormalizePolicy::Feedback::Strip;
            NormalizeResult nr = normalize(ps.circuit, pol);
            BareState bs = build_bare_state(nr.normalized);
            if (bs.rejected) {
                err = "bare state rejected: " +
                      (bs.reject_reason.empty() ? std::string("unknown cause")
                                                : bs.reject_reason);
            } else {
                fs = FramedSuperposition::from_css(bs.state);
            }
        }
    }
    if (!err.empty()) throw std::invalid_argument(err);
    return PyFramedSuperposition(fs);
}

// ── Task 2b.2: _input_state_of — physical carried state + its output-wire map ────────────
// Builds the frame-explicit bare of `text` (same path as _bare_state_of) AND returns the
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
            NormalizePolicy pol;
            pol.coherentize = true;
            pol.defer = true;
            pol.want_map = true;
            pol.feedback = NormalizePolicy::Feedback::Strip;
            NormalizeResult nr = normalize(ps.circuit, pol);
            BareState bs = build_bare_state(nr.normalized);
            if (bs.rejected) {
                err = "bare state rejected: " +
                      (bs.reject_reason.empty() ? std::string("unknown cause")
                                                : bs.reject_reason);
            } else {
                fs = FramedSuperposition::from_css(bs.state);
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
            NormalizePolicy pol;
            pol.coherentize = true;
            pol.defer = true;
            pol.want_map = true;
            pol.feedback = NormalizePolicy::Feedback::Strip;
            NormalizeResult nr = normalize(ps.circuit, pol);
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

// Task 2 free function: _port_contract_of_state(state, wires) -> dict.
// Takes a PyFramedSuperposition and a list of wire indices; returns the port verdict dict.
// Private (underscore prefix). Used by the Task 3 invariance gate and by differential tests.
py::dict py_port_contract_of_state(const PyFramedSuperposition& st,
                                   const std::vector<int>& wires) {
    PortVerdict v = qeccore::port_contract(st.state, wires);
    return port_verdict_to_dict(v);
}

// ── S2.2: _project_bare_onto_syndrome — independent exact oracle ─────────────────────────
// For each shot's recorded syndrome σ (a shared fact from sample_barrier), INDEPENDENTLY
// reconstruct the collapsed post-barrier state by projecting a FRESH bare onto σ. The method
// is the frame's DESTABILISERS: applying d_i = U.Xrow[i] (the destabiliser that anticommutes
// with generator g_i = U.Zrow[i] and commutes with every other generator/destabiliser) flips
// exactly g_i's eigenvalue. Applying the product { d_i : σ_i = 1 } to the bare produces the
// unique state with syndrome σ. This is genuinely independent of twirl_collapse: it uses the
// general FramedSuperposition Clifford-apply engine on the frame's destabilisers, NOT the
// DiagNormalForm residual (S^a·CZ·X^v·Z^z) that the sampler applies to build amps.
//
// Uniqueness requires k()==0 (no logical/free DOF): otherwise σ does not determine the state
// (destabiliser projection fixes the logical to the reference, but the sampler's residual may
// carry a logical component). We REFUSE k>0 loudly rather than compare ill-defined states.
py::list py_project_bare_onto_syndrome(const std::string& text,
                                       const std::vector<std::vector<uint8_t>>& sigmas) {
    PyFramedSuperposition base = py_bare_state_of(text);   // fresh independent bare
    const FramedSuperposition& b = base.state;
    if (b.k() != 0)
        throw std::invalid_argument(
            "_project_bare_onto_syndrome: bare state has k=" + std::to_string(b.k()) +
            " logical DOF; the recorded syndrome does not uniquely determine the collapsed "
            "state. Use a k=0 (fully-stabilised) barrier circuit for this oracle.");
    const int n = b.n();
    const std::vector<Pauli> gens = b.certified_stabilizers();  // order == σ bit order
    const int ng = (int)gens.size();                            // == n for k=0
    py::list result;
    for (const std::vector<uint8_t>& sig : sigmas) {
        FramedSuperposition st = b;                             // fresh copy of the bare
        for (int i = 0; i < ng && i < (int)sig.size(); ++i) {
            if (!sig[(size_t)i]) continue;
            const Pauli& d = b.U.Xrow[(size_t)i];               // destabiliser of g_i
            for (int q = 0; q < n; ++q) {
                if (d.xbit(q)) st.apply_clifford(/*X*/ 3, q, 0);
                if (d.zbit(q)) st.apply_clifford(/*Z*/ 5, q, 0);
            }
        }
        result.append(PyFramedSuperposition(std::move(st)));
    }
    return result;
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

// ── D1: carried_symplectic_action — the stage's Clifford action on carried wires ─────────────
// Compute the GF(2) symplectic map M: (x‖z) frame bits on the INPUT_QUBITS carried wires (at stage
// entry, t=0) → (x‖z) frame bits on the OUTPUT_QUBITS carried wires (at stage exit).  Built by
// conjugating each basis Pauli (X_i, Z_i on each carried input wire) through the stage's Clifford
// content C and reading the RESTRICTION of C·P·C† to the output wires.  Frames are projective, so
// the Pauli SIGN/phase is dropped (only the x/z SUPPORT bits matter).  Support that C pushes onto
// measured/discarded wires is intentionally omitted: it is accounted by the input-frame channel
// relabel (input_pauli applied at t=0 rides the same C through the sampler and relabels THIS stage's
// records), so the carried part for the NEXT stage is exactly the output-wire restriction.
//
// Returns {ok, error, in_wires:[...], out_wires:[...], M: flat row-major (2*n_in)·(2*n_out) uint8}.
// Row r = the image of input basis Pauli:  r < n_in  → X_{in_wires[r]};  r >= n_in → Z_{in_wires[r-n_in]}.
// Column layout per row: [x-part over out_wires | z-part over out_wires] (width 2*n_out).
// out_wires order == flattened circuit.outputs (== dem_column_layout 'frame' order == output_wires()).
// Measure/Reset/Noise/Observable/Feedback instructions are SKIPPED (only Kind::Gate conjugates the
// frame; the carried wires are OUTPUT_QUBITS = never measured, so no reset touches them).
py::dict py_carried_symplectic_action(const std::string& text) {
    py::dict out;
    ParsedStim ps;
    {
        py::gil_scoped_release rel;
        ps = parse_stim_circuit(text);
    }
    if (!ps.ok()) {
        out["ok"] = false;
        std::string msg = "parse failed";
        for (const auto& e : ps.errors)
            msg += "\n  line " + std::to_string(e.line) + ": " + e.message;
        out["error"] = msg;
        return out;
    }
    const Circuit& c = ps.circuit;

    // Flatten the carried input / output wires in declaration order.
    std::vector<int> in_wires, out_wires;
    for (const InputPort& p : c.inputs)
        if (p.kind == InputPort::Kind::Qubits)
            for (int q : p.qubits) in_wires.push_back(q);
    for (const OutputPort& op : c.outputs)
        for (int q : op.qubits) out_wires.push_back(q);

    const int n_in = (int)in_wires.size();
    const int n_out = (int)out_wires.size();

    // CONE-TRACKED forward propagation (2026-07-23, replaces the whole-stage tableau + global
    // non-Clifford refusal).  Only the 2·n_in port basis-Pauli IMAGES are tracked, gate by gate
    // in time order, so a non-Clifford gate refuses ONLY when the frame cone actually reaches it
    // — the same Heisenberg-propagation principle as the engine's input frame relabel:
    //   * Clifford gates conjugate the images (support-only; signs are projective for frames).
    //   * DIAGONAL non-Clifford (T/CS/CCZ) commutes with Z-support: refuse iff any image carries
    //     an X-bit on a target wire (the image would leave the Pauli group).
    //   * Non-diagonal non-Clifford (CH): refuse on ANY image support at its targets.
    //   * Measure/Reset CLEAR image support on their wires: a measured wire's flip is consumed by
    //     the input_pauli record relabel, a reset wire's state is re-prepared — either way the
    //     support does not propagate further (also makes wire-reuse after M/R correct, which the
    //     old skip silently mishandled).
    //   * Noise / Observable / ControlledPauli (a classically-controlled PAULI commutes with the
    //     tracked Paulis up to phase) do not alter support.
    const int NW = c.n;
    std::vector<std::vector<uint8_t>> ix((size_t)(2 * n_in), std::vector<uint8_t>((size_t)NW, 0));
    std::vector<std::vector<uint8_t>> iz((size_t)(2 * n_in), std::vector<uint8_t>((size_t)NW, 0));
    for (int i = 0; i < n_in; ++i) {
        if (in_wires[i] < 0 || in_wires[i] >= NW) {
            out["ok"] = false;
            out["error"] = "carried_symplectic_action: INPUT_QUBITS wire out of range";
            return out;
        }
        ix[(size_t)i][(size_t)in_wires[i]] = 1;               // image of X_{in_wires[i]}
        iz[(size_t)(n_in + i)][(size_t)in_wires[i]] = 1;      // image of Z_{in_wires[i]}
    }
    auto any_x_on = [&](const std::vector<int>& ws) {
        for (int r = 0; r < 2 * n_in; ++r)
            for (int w : ws)
                if (w >= 0 && w < NW && ix[(size_t)r][(size_t)w]) return true;
        return false;
    };
    auto any_support_on = [&](const std::vector<int>& ws) {
        for (int r = 0; r < 2 * n_in; ++r)
            for (int w : ws)
                if (w >= 0 && w < NW &&
                    (ix[(size_t)r][(size_t)w] || iz[(size_t)r][(size_t)w])) return true;
        return false;
    };
    auto clear_on = [&](const std::vector<int>& ws) {
        for (int r = 0; r < 2 * n_in; ++r)
            for (int w : ws)
                if (w >= 0 && w < NW) { ix[(size_t)r][(size_t)w] = 0; iz[(size_t)r][(size_t)w] = 0; }
    };
    for (const Instr& ins : c.stream) {
        if (ins.kind == Instr::Kind::Measure || ins.kind == Instr::Kind::Reset) {
            clear_on(ins.qubits);
            continue;
        }
        if (ins.kind != Instr::Kind::Gate) continue;
        const std::vector<int>& t = ins.targets;
        switch (ins.gate) {
            case GateKind::H:
                for (int r = 0; r < 2 * n_in; ++r)
                    std::swap(ix[(size_t)r][(size_t)t[0]], iz[(size_t)r][(size_t)t[0]]);
                break;
            case GateKind::S:
            case GateKind::SDG:
                for (int r = 0; r < 2 * n_in; ++r)
                    iz[(size_t)r][(size_t)t[0]] ^= ix[(size_t)r][(size_t)t[0]];
                break;
            case GateKind::X: case GateKind::Y: case GateKind::Z:
                break;                                        // Paulis: support unchanged
            case GateKind::CX:
                for (int r = 0; r < 2 * n_in; ++r) {
                    ix[(size_t)r][(size_t)t[1]] ^= ix[(size_t)r][(size_t)t[0]];
                    iz[(size_t)r][(size_t)t[0]] ^= iz[(size_t)r][(size_t)t[1]];
                }
                break;
            case GateKind::CZ:
                for (int r = 0; r < 2 * n_in; ++r) {
                    iz[(size_t)r][(size_t)t[1]] ^= ix[(size_t)r][(size_t)t[0]];
                    iz[(size_t)r][(size_t)t[0]] ^= ix[(size_t)r][(size_t)t[1]];
                }
                break;
            case GateKind::T: case GateKind::CS: case GateKind::CCZ:
                if (any_x_on(t)) {
                    out["ok"] = false;
                    out["error"] = std::string(
                        "carried_symplectic_action: the carried-frame cone reaches a DIAGONAL "
                        "non-Clifford gate with X-support (frame cannot be conjugated through it)");
                    return out;
                }
                break;                                        // Z-support commutes with diagonals
            default:                                          // CH and anything unclassified
                if (any_support_on(t)) {
                    out["ok"] = false;
                    out["error"] = std::string(
                        "carried_symplectic_action: the carried-frame cone reaches a "
                        "non-diagonal non-Clifford gate (frame cannot be conjugated through it)");
                    return out;
                }
                break;
        }
    }

    // M is (2*n_in) rows × (2*n_out) cols, row-major uint8: each image restricted to out_wires.
    std::vector<uint8_t> M((size_t)(2 * n_in) * (size_t)(2 * n_out), 0);
    for (int r = 0; r < 2 * n_in; ++r) {
        uint8_t* row = M.data() + (size_t)r * (size_t)(2 * n_out);
        for (int j = 0; j < n_out; ++j) {
            int w = out_wires[j];
            if (w >= 0 && w < NW) {
                if (ix[(size_t)r][(size_t)w]) row[j] = 1;         // x-part
                if (iz[(size_t)r][(size_t)w]) row[n_out + j] = 1; // z-part
            }
        }
    }

    out["ok"] = true;
    out["in_wires"] = in_wires;
    out["out_wires"] = out_wires;
    py::array_t<uint8_t> Marr({2 * n_in, 2 * n_out});
    std::memcpy(Marr.mutable_data(), M.data(), M.size());
    out["M"] = Marr;
    return out;
}

}  // namespace

PYBIND11_MODULE(_xtim, m) {
    m.doc() = "xtim engine bindings: parse / packed sampling / DEM export / .ref surface.";
    m.attr("ENGINE_VERSION") = kEngineVersion;
    m.attr("REF_FORMAT_VERSION") = kRefFormatVersion;

    m.def("parse_info", &py_parse_info, py::arg("text"),
          "Parse a circuit; returns {ok, errors=[(line, message)...], counts...}.");
    py::class_<CompiledProgram>(m, "CompiledProgram",
          "Stateful compiled circuit: parse + compile ONCE in the constructor, then sample() "
          "repeatedly with no recompile -> packed b8 record buffers + float64 expectations.")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("text"), py::arg("ref_text") = std::string())
        .def("sample", &CompiledProgram::sample, py::arg("shots"), py::arg("seed"),
             py::arg("want_meas") = true, py::arg("want_det") = true,
             py::arg("want_obs") = true, py::arg("want_exp") = true,
             "Sample the compiled program -> packed b8 buffers + expectations; skips the "
             "per-call compile. want_meas/want_det/want_obs/want_exp "
             "(default all True): build only the requested record channels as numpy arrays; an "
             "unrequested channel's dict key is omitted (scalar metadata always present).")
        .def_property_readonly("ok", &CompiledProgram::ok)
        .def_property_readonly("ref_error", &CompiledProgram::ref_error)
        .def_property_readonly("errors", &CompiledProgram::errors);
    m.def("export_dem_text", &py_export_dem_text, py::arg("text"),
          py::arg("include_expectations") = true,
          py::arg("decompose_errors") = false,
          py::arg("ignore_decomposition_failures") = false,
          py::arg("drop_gauge_observables") = false,
          py::arg("trusted_detectors") = std::vector<int>{},
          py::arg("trusted_observables") = std::vector<int>{},
          "Stim-compatible DEM text export (run_stim_main --dem semantics). "
          "include_expectations: PAULI_EXPECTATION L-columns after the observables "
          "(default ON; False reproduces the observable/detector-only export). "
          "trusted_detectors/trusted_observables: ids whose determinism is certified "
          "externally against the TRUE composed bare state (from_state channel_report) — "
          "the text-only determinism gate is skipped for exactly those ids; mechanisms "
          "are state-free and unchanged.");
    m.def("detector_determinism", &py_detector_determinism, py::arg("text"),
          "Exact per-detector/observable determinism partition of the noiseless circuit "
          "(parity-Pauli Born check on the bare state). Shot-free, and answers even when the "
          "full DEM refuses. Keys: ok, deterministic_detectors, gauge_detectors, "
          "deterministic_observables, gauge_observables, detector_parity, observable_parity, "
          "num_detectors, num_observables, rejected, reject_gate_index, error, errors.");
    m.def("expectation_frames", &py_expectation_frames, py::arg("text"),
          "Exact verify of each declared PAULI_EXPECTATION byproduct frame. Sampling-free: "
          "decides whether the declared rec[-k] frame equals the true sign-controlling record "
          "set (a property of the bare state's fixed stabilizer frame; free/logical rows "
          "excluded). Keys: ok, num_columns, required, declared, solvable, column_ok, all_ok, "
          "refusal, rejected, reject_gate_index, error, errors. all_ok False => refusal names "
          "the mismatched columns and their missing/spurious records.");
    m.def("carried_symplectic_action", &py_carried_symplectic_action, py::arg("text"),
          "D1: GF(2) symplectic map of the stage's Clifford action on carried wires — maps "
          "(x‖z) frame bits on INPUT_QUBITS wires (t=0) to (x‖z) frame bits on OUTPUT_QUBITS "
          "wires (t=exit), by conjugating each basis Pauli through the stage's Clifford content "
          "(sign dropped; frames projective). Support pushed onto measured/discarded wires is "
          "omitted (accounted by the input-frame channel relabel). Returns {ok, error, in_wires, "
          "out_wires, M} where M is a (2*n_in)x(2*n_out) uint8 array: row r<n_in = image of "
          "X_{in_wires[r]}, row r>=n_in = image of Z_{in_wires[r-n_in]}; columns [x over out_wires "
          "| z over out_wires]. A non-Clifford gate in the body -> ok=False (loud).");
    m.def("dem_column_layout", &py_dem_column_layout, py::arg("text"),
          "DEM column layout for a module-system circuit (structural query, no DEM export). "
          "Keys: ok, errors, rejected, reject_gate_index, error, n_obs (O+R, existing L-columns), "
          "frame [(orig_qubit, col_x, col_z), ...] (OUTPUT_QUBITS frame columns), "
          "logical_flip [(decision_k, col), ...] (DECISION logical-flip columns). "
          "Column convention: col_x fires when the error mechanism has X-support on the "
          "qubit (an X correction is needed); col_z fires when Z-support (Z correction needed). "
          "The orchestrator maps frame col_x -> X-part of input_pauli, col_z -> Z-part.");
    m.def("ref_info", &py_ref_info, py::arg("ref_text"),
          "Header info of a .ref text (no state construction).");
    m.def("deferred_signature", &py_deferred_signature, py::arg("circuit_text"),
          "Structural signature of the normalized+deferred circuit (noise values excluded); "
          "used by the reference cache to reject frame-divergent hits.");
    m.def("ref_verify_cheap", &py_ref_verify_cheap, py::arg("circuit_text"),
          py::arg("ref_text"), py::arg("seed") = 12345, py::arg("shots") = 200,
          "Cheap cache-hit verify: load + n-match + noiseless invariants; the exact "
          "state-overlap gate runs only when the invariants are vacuous (0 det + 0 exp).");
    m.def("ref_compile_text", &py_ref_compile_text, py::arg("circuit_text"),
          py::arg("source") = std::string("xtim-circuit"), py::arg("oracle_chi") = 256,
          py::arg("seed") = 12345, py::arg("shots") = 2000,
          "The FULL ref_compile gate battery (compile mode); returns .ref text on success.");
    // S2.1: FramedSuperposition handle (created by _bare_state_of).
    // S2.2: adds approx_equal (exact state comparison) + apply_clifford (mutation testing).
    py::class_<PyFramedSuperposition>(m, "FramedSuperposition",
          "Handle for a qeccore::FramedSuperposition (a collapsed bare state). "
          "Create via _xtim._bare_state_of(text); pass to _xtim.TwirlSampler(state, text, …) "
          "or xtim.compile_twirl_sampler_from_state(state, text, …). "
          "S2.2: approx_equal compares states up to global phase; apply_clifford mutates in-place.")
        .def("approx_equal", &PyFramedSuperposition::approx_equal,
             py::arg("other"), py::arg("tol") = 1e-12,
             "Return True iff |<self|other>| >= 1 - tol (equal up to global phase). "
             "Uses exact_sum_overlap (framed ray materialisation + inner product). "
             "S2.2 correctness oracle: compare sample_barrier states against the "
             "independent _project_bare_onto_syndrome path.")
        .def("apply_clifford", &PyFramedSuperposition::apply_clifford_py,
             py::arg("kind"), py::arg("a"), py::arg("b") = 0,
             "Apply a Clifford gate to the state in-place. "
             "kind: 0=H 1=S 2=Sdg 3=X 4=Y 5=Z 6=CX 7=CZ. "
             "For single-qubit gates (0-5) b is unused (pass 0). "
             "S2.2 mutation check: apply Z (kind=5) to qubit 0, verify approx_equal fails.")
        .def("pauli_expectation_x", &PyFramedSuperposition::pauli_expectation_x, py::arg("q"),
             "Task 2b.1: exact real single-qubit <X_q> on the collapsed state. Used by the "
             "Born-decision correlation gate (retained state must satisfy <X_0> = ±1).")
        .def("pauli_expectation_z", &PyFramedSuperposition::pauli_expectation_z, py::arg("q"),
             "Task 2b.1: exact real single-qubit <Z_q> on the collapsed state.")
        .def("pauli_expectation_y", &PyFramedSuperposition::pauli_expectation_y, py::arg("q"),
             "Task 2b.2: exact real single-qubit <Y_q> on the collapsed state (Y = i·XZ). "
             "Used by the ⊗-composition to classify a carried qubit's stabilizer axis.")
        .def("pauli_expectation_xz", &PyFramedSuperposition::pauli_expectation_xz,
             py::arg("xs"), py::arg("zs"), "Diagnostic: <ΠX_xs·ΠZ_zs> (real).")
        .def("port_signature", &PyFramedSuperposition::port_signature,
             py::arg("wires"),
             "Canonical bytes fingerprint of the reduced stabilizer state on the carried port "
             "`wires` (port-restricted reduced tableau + signs). Equal bytes ⟺ identical reduced "
             "density matrix on `wires`. The BUCKET-FOLD guard (run_protocol): fold record buckets "
             "of a decision pattern iff they share one port_signature.")
        .def("port_signature_struct", &PyFramedSuperposition::port_signature_struct,
             py::arg("wires"),
             "Task 6: structured accessor for the port_signature byte format. Decodes the raw "
             "bytes (via port_signature) into a dict with keys flag, W, ns, keys, blochs — "
             "same fields and semantics as fold.py's _parse_port_signature. One owner of the "
             "wire format: changing port_signature automatically updates this accessor.")
        .def("port_orbit_operators", &PyFramedSuperposition::port_orbit_operators,
             py::arg("wires"),
             "PAULI-ORBIT FOLD companion to port_signature: the canonical port symplectic OPERATORS "
             "whose signs the fingerprint records — dict{ok, W, ns, k_port, stab:[2W-uint8 …], "
             "logicals:[La,Lb]} — in the SAME order as port_signature (stab supports sorted; La,Lb "
             "match the Bloch coords). A pure function of the stabilizer SUPPORTS ⇒ identical across "
             "one sign-stripped orbit class. Coordinate layout matches input_pauli (low W = X, high "
             "W = Z over wires). ok=False for k_port≥2 / W>64 (un-foldable).")
        .def("port_sigma_law", &PyFramedSuperposition::port_sigma_law,
             py::arg("wires"),
             "E2 (exact-residual arc): COMBINATION-TRACKED port decomposition — the σ-law "
             "companion to port_signature. Same two-phase Gauss-Jordan, plus per-row GF(2) "
             "combination tracking over the certified generators. Returns dict{W, ngens, gw, "
             "ns, k_port, stab:(ns,2W) uint8, comb:(ns,gw) uint64 (σ WORD layout: bit b ↔ "
             "certified_stabilizers()[b] ↔ sigmas() bit b), ref_signs:(ns,) uint8, "
             "logicals:[La,Lb], ok}. Exact law for every port stabilizer check j: "
             "sign_j(shot) = ref_signs[j] ^ parity(comb[j] & σ_shot); stab rows are in the "
             "IDENTICAL sorted order port_signature uses. La/Lb (k_port==1, via "
             "port_orbit_operators) carry NO combination — out of span(certified), per-plan "
             "scope. ADDITIVE: existing methods and byte outputs untouched.")
        .def("certified_symplectic", &PyFramedSuperposition::certified_symplectic,
             "E3 (exact-residual arc): the state's certified stabilizer generators as full "
             "n-qubit symplectic rows + exact phases (PURE DATA). Returns dict{n, ngens, "
             "xz:(ngens,2n) uint8 (low n = x bits, high n = z bits), phase:(ngens,) uint8}. "
             "Row b == certified_stabilizers()[b] == σ bit b of BarrierBuffer.sigmas() == "
             "port_sigma_law comb bit b. Generator operator = i^phase·X^x·Z^z; every row has "
             "+1 eigenvalue on the bare state.")
        .def("measure_pauli", &PyFramedSuperposition::measure_pauli,
             py::arg("basis"), py::arg("q"), py::arg("u"),
             "Task 2b.2-pre: Born-measure single-qubit Pauli (basis 0:X 1:Y 2:Z) on qubit q "
             "with random u∈[0,1); returns ±1 and COLLAPSES the state in place. The exact "
             "per-shot decision read the engine performs internally, exposed for the oracle.")
        .def_property_readonly("n", &PyFramedSuperposition::n, "Number of qubits.")
        .def_property_readonly("k", &PyFramedSuperposition::k,
             "Number of logical (free) DOF; k==0 ⇒ syndrome fully determines the state.");
    // ── State-in / port-contract API naming convention ──────────────────────────
    // Public names (bare_state_of, input_state_of, port_contract_of_state) are
    // canonical; the _underscore twins are backward-compat aliases bound to the
    // SAME function (identical behaviour, including validation). Prefer the public
    // name in new code. _output_wires_of is INTENTIONALLY private with no public
    // twin: it is a non-validating wire-lookup for test harnesses that must
    // inspect states the validating API refuses.

    // S2.1: bare_state_of — public name; _bare_state_of is a backward-compat alias, see convention note above.
    const char* _bare_state_of_doc =
        "Build and return the FramedSuperposition bare state of `circuit_text` "
        "(the same build_bare_state(deferred) path as compile_twirl_sampler). "
        "Used by compile_twirl_sampler_from_state to allow external state injection. "
        "Raises ValueError on parse failure or bare-state rejection.";
    m.def("bare_state_of",  &py_bare_state_of, py::arg("circuit_text"), _bare_state_of_doc);
    m.def("_bare_state_of", &py_bare_state_of, py::arg("circuit_text"), _bare_state_of_doc); // public + back-compat alias, see convention note above

    // S2.2/S2.4: BarrierBuffer — per-shot collapsed post-barrier state + syndrome +
    //            packed channel bits (all from the same single run() pass).
    py::class_<PyBarrierBuffer>(m, "BarrierBuffer",
          "Per-shot collapsed post-barrier container (S2.2/S2.4). buffer.state(i) returns an "
          "independent copy of shot i's REAL collapsed state (residual-applied, kernel-"
          "collapsed — genuinely per-shot, not a bare copy); buffer.sigma(i) returns that "
          "shot's raw certified-generator syndrome (bit list over bare.certified_stabilizers()). "
          "buffer.dets() / buffer.obs() return the packed Stim-b8 detector/observable arrays "
          "captured from the SAME single sink pass — no dual-sample misalignment.")
        .def("state", &PyBarrierBuffer::state, py::arg("i"),
             "Return shot i's collapsed post-barrier FramedSuperposition, reconstructed ON DEMAND "
             "from its compact record (alias of materialize(i) — the full state is no longer "
             "retained per shot).")
        .def("materialize", &PyBarrierBuffer::materialize, py::arg("i"),
             "Lazily reconstruct shot i's EXACT collapsed post-barrier + post-decision state from "
             "its record (plan, σ, coins). Call once per bucket representative, not per shot.")
        .def("record_keys", &PyBarrierBuffer::record_keys,
             "Per-shot record (σ ‖ coins ‖ plan) as a fixed-width (shots × W) uint8 matrix for "
             "O(shots) vectorized bucketing (np.unique on a void view). Exact record partition.")
        .def("record_group_ids", &PyBarrierBuffer::record_group_ids,
             "Per-shot int32 first-occurrence group id of the DISTINCT record (σ ‖ coins ‖ plan), "
             "factorized by C-level bytes hashing in one pybind call. Same partition as a "
             "record_keys() |S{W} view, without the wide-key sort or per-shot Python.")
        .def("record_groups", &PyBarrierBuffer::record_groups,
             "Fused single-walk accessor. Returns a 3-tuple (gids, first_occ, distinct_keys):\n"
             "  gids          — (nshots,) int32: per-shot group id, IDENTICAL to record_group_ids().\n"
             "  first_occ     — (n_groups,) int32: first-occurrence shot index for each group.\n"
             "  distinct_keys — (n_groups, W) uint8: key row for each distinct record; row g ==\n"
             "                  record_keys()[first_occ[g]] (byte-identical). Replaces the two-pass\n"
             "                  record_group_ids() + record_keys() pattern with ONE hash-map walk.")
        .def("coins", &PyBarrierBuffer::coins, py::arg("i"),
             "Per-shot coin record (r fair coins then κ chain outcomes, 0/1); length varies per "
             "shot with the residual plan. With sigma(i) and plan_key(i) it is a sufficient "
             "classical statistic for the collapsed state(i) — the record-hash bucketing key.")
        .def("plan_key", &PyBarrierBuffer::plan_key, py::arg("i"),
             "Per-shot canonical residual plan bytes (prefix x/z support + a-mask + cz, global "
             "phase excluded); the third component of the record-hash key (plan, σ, coins).")
        .def("plan_structure", &PyBarrierBuffer::plan_structure, py::arg("plan_key_bytes"),
             "E3 (exact-residual arc): parse a plan key's bytes (as plan_key(i) returns them; "
             "b'' = clean/identity) into the residual normal form + the plan's ShotLaw, as "
             "numpy data — PURE DATA EXPOSURE of the per-plan structure the engine already "
             "derives (parse + the identical build_shot_law call materialize(i) makes). "
             "Returns dict{n, ngens, gw, r, kappa, fallback, prefix_xz:uint8[2n], a:uint8[n], "
             "cz:int32[ncz,2], det_signs:uint64[gw], coin_masks:uint64[r,gw], "
             "kernel_masks:uint64[kappa,gw], kernel_base:uint8[kappa], "
             "kernel_foldable:uint8[kappa], kernel_xz:uint8[kappa,2n], "
             "kernel_phase:uint8[kappa]}. Kernel rep j (operator i^phase·X^x·Z^z, bare frame) "
             "is Born-measured with outcome bit coins(i)[r+j] (0 = +1). Cold per DISTINCT "
             "plan — memoize on the plan_key bytes.")
        .def("born_dec", &PyBarrierBuffer::born_dec,
             "T5 step 0 (exact-residual arc): the sampler's Born-measured decision operators "
             "(LOGICAL/ANTI-class DECISIONs, declaration order) — PURE DATA. Returns "
             "dict{nborn, n, dec_index:int32[nborn], xz:uint8[nborn,2n], phase:uint8[nborn], "
             "inv:uint8[nborn]}; operator j = i^phase·X^x·Z^z over the n deferred wires, "
             "measured per shot on the residual-applied collapsed state. Its RAW outcome bit "
             "(0 = +1) is coins(i)[r+kappa+j]; the EMITTED decision bit is raw XOR inv (XOR "
             "readout-flip coin parity / input-frame relabel — record-only effects). Consumers "
             "extend the per-(check, plan) span basis with the plan-CONJUGATED ops.")
        .def("sigma", &PyBarrierBuffer::sigma, py::arg("i"),
             "Return shot i's certified-generator syndrome as a list of 0/1 bits.")
        .def("sigmas", &PyBarrierBuffer::sigmas,
             "Bulk certified-generator sign syndrome matrix, shape (nshots, SGW) uint64 "
             "(a COPY of the internal word store). Bit b of shot i is "
             "``(row[b>>6] >> (b&63)) & 1``, identical to ``sigma(i)[b]``. "
             "Tail bits beyond sig_bits (positions sig_bits..SGW*64-1) are ALWAYS ZERO "
             "by construction: source sigma vectors are initialized to GW zero words with "
             "only bits 0..ngens-1 ever written, and the entire store is pre-zeroed before "
             "sampling (the sink does not zero per-row). "
             "Use ``buf.sig_bits`` to mask the tail word if needed.")
        .def_property_readonly("sig_bits", [](const PyBarrierBuffer& b) { return b.sig_bits_; },
             "σ bit width = number of certified generators (ngens); equals the number of "
             "meaningful bits in each row of sigmas(). Use to mask the tail word of sigmas() "
             "or to know the length of sigma(i).")
        .def("dets", &PyBarrierBuffer::dets,
             "Return packed detector bits, shape (shots, DBB), uint8 Stim-b8 layout. "
             "Captured from the same sink call as state(i) — single-pass, no misalignment.")
        .def("obs", &PyBarrierBuffer::obs,
             "Return packed observable bits, shape (shots, OBB), uint8 Stim-b8 layout. "
             "Captured from the same sink call as state(i) — single-pass, no misalignment.")
        .def("decisions", &PyBarrierBuffer::decisions,
             "Task 2b.1: packed Born-decision bits, shape (shots, ceil(nD/8)), uint8 Stim-b8 "
             "(LE) layout — one bit per declared DECISION index, captured from the SAME sink "
             "pass as state(i). The decision and the retained state come from one Born draw.")
        .def_property_readonly("num_shots", &PyBarrierBuffer::num_shots,
             "Number of shots stored in this buffer.")
        .def("_debug_corrupt_compact", &PyBarrierBuffer::_debug_corrupt_compact,
             py::arg("i"), py::arg("field") = "prefix",
             "TEST-ONLY mutation probe (port-v3 T2; field variants T3): corrupt ONE shot's "
             "stored compact identity — field='prefix' XORs bit 0 of the shot's prefix word, "
             "field='plan_id' resets the shot's interned plan id to the empty plan — so "
             "record_groups() must produce a DIFFERENT partition/keys (proves each compact "
             "field is load-bearing). Raises unless the buffer was sampled with compact=True.");

    py::class_<PyTwirlSampler>(m, "TwirlSampler",
          "End-to-end twirl record sampler (qeccore::TwirlRecordSampler). Deterministic "
          "detector/observable channels are EXACT; gauge detector columns are DECLARED "
          "fair coins; refused observables raise unless skip_refused_observables. See "
          "docs/twirl_record_sampler.md for the semantics contract.")
        .def(py::init<const std::string&, double, const std::string&, long, bool,
                      const std::string&>(),
             py::arg("circuit_text"), py::arg("p_factor") = 1.0,
             py::arg("disk_cache") = std::string(), py::arg("selfcheck") = 2000,
             py::arg("skip_refused_observables") = false,
             py::arg("disk_cache_auto_dir") = std::string())
        // S2.1: construct from a provided FramedSuperposition bare state (compile_from_state).
        .def(py::init<const PyFramedSuperposition&, const std::string&, double,
                      const std::string&, long, bool, const std::string&>(),
             py::arg("state"), py::arg("circuit_text"), py::arg("p_factor") = 1.0,
             py::arg("disk_cache") = std::string(), py::arg("selfcheck") = 2000,
             py::arg("skip_refused_observables") = false,
             py::arg("disk_cache_auto_dir") = std::string())
        // Task 3.0: multi-qubit carried port via state-level injection (carried, src_wires, text).
        .def(py::init<const PyFramedSuperposition&, const std::vector<int>&, const std::string&,
                      double, const std::string&, long, bool, const std::string&>(),
             py::arg("carried"), py::arg("src_wires"), py::arg("circuit_text"),
             py::arg("p_factor") = 1.0,
             py::arg("disk_cache") = std::string(), py::arg("selfcheck") = 2000,
             py::arg("skip_refused_observables") = false,
             py::arg("disk_cache_auto_dir") = std::string())
        .def("sample", &PyTwirlSampler::sample, py::arg("shots"),
             py::arg("seed") = py::none(),
             py::arg("input_pauli") = py::none(),
             "Sample -> (detectors, observables), Stim-b8-packed uint8 arrays of shape "
             "(shots, ceil(K/8)). seed=None continues the current stream; an explicit "
             "seed deterministically reseeds the sampler in place (same seed -> same "
             "bytes; no reconstruction). "
             "input_pauli: optional uint8 array of shape (shots, 2*support) encoding a "
             "per-shot Pauli frame (X-part then Z-part); None => byte-identical no-op "
             "(Task 4 / decoder-feedback sign relabel).")
        // S2.2: sample_barrier — run shots and retain per-shot post-barrier states.
        .def("sample_barrier", &PyTwirlSampler::sample_barrier,
             py::arg("shots"), py::arg("seed") = py::none(),
             py::arg("input_pauli") = py::none(),
             py::arg("compact") = false,
             // keep_alive<0,1>: the returned BarrierBuffer keeps its owning sampler (and its bare
             // state, which materialize() replays from) alive for the buffer's whole lifetime —
             // so lazy state materialization is valid even after `s` goes out of the caller's scope.
             py::keep_alive<0, 1>(),
             "Sample shots, retaining per-shot post-barrier FramedSuperpositions. "
             "Returns a BarrierBuffer; buffer.state(i) gives the state for shot i. "
             "Stage 1: correct for p_factor=0 (all-clean shots, state == bare state). "
             "seed=None continues the current RNG stream; explicit seed reseeds in place. "
             "input_pauli: optional (shots, 2*support) uint8 per-shot Pauli frame "
             "(X-part then Z-part), sign-relabeling the packed record bits exactly like "
             "sample(); supplied without an INPUT_QUBITS port raises ValueError. "
             "compact (port-v3 T2, default False): store the COMPACT plan identity "
             "(interned plan id + prefix words + per-plan suffix blob) instead of per-shot "
             "serialized plan bytes — sampled streams byte-identical, record_groups()/"
             "record_keys()/plan_key() reconstruct the identical legacy bytes (per GROUP, "
             "not per shot). The xtim.port record route uses compact=True.")
        .def("channel_report", &PyTwirlSampler::channel_report,
             "Channel classification + run counters: deterministic_detectors, "
             "gauge_detectors, anti_detectors, refused_observables, "
             "deterministic_observables, num_channels, plans, plan_hits, plan_misses, "
             "ppr_plans, fallback_shots, used, sink_kind.")
        .def("channel_one_counts", &PyTwirlSampler::channel_one_counts,
             "C1: per-channel fired-count list. All-zero on the default selfcheck=0 path; "
             "nonzero when QEC_TW_COUNT=1 or selfcheck>0.")
        .def_property_readonly("num_detectors", &PyTwirlSampler::num_detectors)
        .def_property_readonly("num_observables", &PyTwirlSampler::num_observables)
        .def_property_readonly("num_decisions", &PyTwirlSampler::num_decisions,
             "Task 2b.1: number of declared DECISION output bits (max DECISION index + 1).")
        .def("output_wires", &PyTwirlSampler::output_wires,
             "Task 2b.2-pre: deferred-space wire indices of the declared OUTPUT_QUBITS "
             "(flattened, declaration order). The retained barrier state is indexed in deferred "
             "wire space; read pauli_expectation_x/z at these indices on buf.state(i) for a "
             "declared output qubit's PHYSICAL frame value.")
        .def("plan_structure", &PyTwirlSampler::plan_structure, py::arg("plan_key_bytes"),
             "port-v3 T2: Segment-level plan-structure accessor — identical dict to "
             "BarrierBuffer.plan_structure (same shared builder), available WITHOUT a buffer "
             "(the plan structure is a pure function of the compiled circuit + plan bytes).")
        .def("born_dec", &PyTwirlSampler::born_dec,
             "port-v3 T2: Segment-level Born-decision-operator accessor — identical dict to "
             "BarrierBuffer.born_dec (same shared builder), available WITHOUT a buffer.")
        .def_property_readonly("sig_bits", &PyTwirlSampler::sig_bits,
             "port-v3 T2: σ bit width = number of certified generators of the sampler's bare "
             "state — equals BarrierBuffer.sig_bits of every buffer this sampler produces.");

    // Task 2b.2 / Task 4: physical carried input state + its OUTPUT_QUBITS wire map (for ⊗-composition).
    // input_state_of is the public name; _input_state_of is a backward-compat alias, see convention note above.
    // Validation runs in C++ inside py_input_state_of; both names validate identically.
    const char* _input_state_of_doc =
        "Build the frame-explicit bare state of `circuit_text` AND return the deferred-space "
        "wire indices of its declared OUTPUT_QUBITS (final_wire[orig], declaration order). "
        "Returns (FramedSuperposition, list[int]). The state carries the PHYSICAL carried "
        "qubits when the prep has no trailing measurement on them (frame-baked). Used by "
        "compile_twirl_sampler_from_state to locate the carried qubits in a provided state. "
        "Raises ValueError on parse failure, bare-state rejection, or an entangled / "
        "non-stabilizer-rest port (both names validate).";
    m.def("input_state_of",  &py_input_state_of, py::arg("circuit_text"), _input_state_of_doc);
    m.def("_input_state_of", &py_input_state_of, py::arg("circuit_text"), _input_state_of_doc); // public + back-compat alias, see convention note above
    m.def("input_qubits", &py_input_qubits, py::arg("text"),
          "Return the flattened INPUT_QUBITS port qubit indices (declaration order). "
          "Empty for a port-free circuit. Raises ValueError on parse failure.");
    m.def("_output_wires_of", &py_output_wires_of, py::arg("text"),
          "Return the deferred-space wire indices of the OUTPUT_QUBITS in `text` WITHOUT "
          "building the bare state and WITHOUT running port-contract validation. "
          "Used by tests that need the wire map for entangled circuits.");

    // port_contract_of_state — public free function for computing the port-verdict.
    // _port_contract_of_state is a backward-compat alias, see convention note above.
    const char* _port_contract_docstring =
        "Compute the port-contract verdict for a (state, wires) pair.\n\n"
        "The verdict is a property of a (state, wires) pair, not of a compiled program.\n"
        "For a circuit containing measurements the BARE state is not the carried state,\n"
        "so callers must pass a post-collapse state (e.g. BarrierBuffer.materialize(i)),\n"
        "and the wires are deferred-space indices as returned by TwirlSampler.output_wires().\n\n"
        "Returns a dict with keys: product (bool), k_port (int), "
        "stabilized (list of (q, axis, sign) int tuples; axis 0=X/1=Y/2=Z, sign 0=+/1=-), "
        "port_wires (list of int; deferred-space wire indices), witness (str; empty iff product).";
    m.def("port_contract_of_state", &py_port_contract_of_state,
          py::arg("state"), py::arg("wires"),
          _port_contract_docstring);
    m.def("_port_contract_of_state", &py_port_contract_of_state,
          py::arg("state"), py::arg("wires"),
          _port_contract_docstring); // public + back-compat alias, see convention note above

    // S2.2: independent exact oracle — reconstruct each collapsed state from its syndrome σ.
    m.def("_project_bare_onto_syndrome", &py_project_bare_onto_syndrome,
          py::arg("circuit_text"), py::arg("sigmas"),
          "Independent post-barrier oracle (S2.2). For each shot's syndrome σ (a list of 0/1 "
          "bits over bare.certified_stabilizers()), reconstruct the collapsed state by applying "
          "the frame's destabilisers { d_i : σ_i=1 } to a FRESH build_bare_state bare — the "
          "unique k=0 state with syndrome σ. Independent of twirl_collapse (general Clifford "
          "apply on destabilisers, not the DiagNormalForm residual). Raises ValueError if k>0.");

    // Task 5: IF branch resolver helpers
    m.def("if_driving_bits", &py_if_driving_bits, py::arg("text"),
          "Return list of (port_index, bit_index) tuples for decision bits that "
          "appear in any IF condition in the circuit. Duplicates deduplicated; "
          "raises ValueError on parse failure.");
    m.def("if_driving_global_offsets", &py_if_driving_global_offsets, py::arg("text"),
          "Return list of global bit offsets (sum of prior Bits port widths + bit_idx) "
          "for IF-driving bits. Order matches if_driving_bits(). Raises ValueError on parse failure.");
    m.def("bits_total_width", &py_bits_total_width, py::arg("text"),
          "Return total width (sum of widths) of all INPUT_BITS ports in the circuit. "
          "This is the required column count for decision_bits arrays. "
          "Raises ValueError on parse failure.");
    m.def("resolve_branches_text", &py_resolve_branches_text,
          py::arg("text"), py::arg("pattern_bytes"),
          "Resolve all IF spans in `text` using `pattern_bytes` (list[int] or bytes, "
          "indexed by global decision-bit offset) and return the branch-free circuit "
          "as a Stim-superset text string suitable for compile_twirl_sampler. "
          "Task 6: rec[-k] references are correctly remapped when measurement-bearing "
          "branches are dropped (pattern_bytes forwarded to emit_resolved_text). "
          "Raises ValueError on parse failure.");
    m.def("detector_branch_spans", &py_detector_branch_spans, py::arg("text"),
          "Task 6: return the IF conditions for each detector in the circuit. "
          "Returns list[list[tuple[int,int]]]: for each detector (in declaration order), "
          "a list of (global_bit_offset, cond_value) pairs that must ALL hold for that "
          "detector to be active.  An empty inner list means always active (not inside any IF). "
          "Used by the Python partition layer to build the superset live-mask. "
          "Raises ValueError on parse failure.");
}
