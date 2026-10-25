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
#include "qeccore/exact_scope.hpp"
#include "qeccore/framed_sampler.hpp"   // tree_fail_memo_stats / _clear (diagnostics)
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
#include "qeccore/frame_cosets.hpp"
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
#include "qeccore/run_stim.hpp"
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
using namespace refcompile;  // the shared ref_compile gate battery (REF_COMPILE_NO_MAIN)

namespace {

// kEngineVersion (and its history) lives in qeccore/engine_version.hpp — the TWPL plan-cache
// header stamps it too (S0(b), D13(a)).
// The compiled sector table's ABI (3.1.3): exported as _xtim.FRAME_COSETS_ABI and compared by
// xtim.frames against its own FRAME_COSETS_ABI at import — a stale .so (missing symbol or
// another value) is REFUSED by plan_frame_cosets_fast, never silently replaced by the
// pure-Python path.  Bump both together when the binding's signature or output changes.
constexpr int kFrameCosetsAbi = 2;   // 2 = EXP-fix #5: weight within 1e-12 of the oracle (was bit-for-bit); |supp a| <= 64
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
    qeccore::ScratchOwner scratch_owner_;   // Python-held owner (automatic scratch release)
    std::string budget_invalid_;            // set by a mid-operation budget refusal
    static constexpr const char* kBudgetWhat = "CompiledProgram (exact sampler)";
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

    // How the zero-error TreePlan and the R>0 expectation channel came out (framed_sampler.hpp
    // ZeroPlan / ChannelPlan). "skipped: zero-error plan failed" is the NAMED outcome of
    // perf/ref-compile-battery: the channel build is not attempted when the zero-error extraction
    // of the same block failed (it would fail identically); expectations take the exact live
    // block. (Before, that build ran and could refuse on a tight state budget.)
    py::dict plan_report() {
        py::dict d;
        d["zero_error_plan"] = "none";
        d["expectation_channel"] = "none";
        if (!prog_) return d;
        FramedCircuitShotSampler* f = sampler_program_framed(*prog_);
        if (!f) return d;
        using ZP = FramedPauliShotSampler::ZeroPlan;
        using CP = FramedPauliShotSampler::ChannelPlan;
        const auto& sp = f->shot_sampler();
        const ZP z = sp.zero_plan_status();
        d["zero_error_plan"] = z == ZP::Built ? "built"
                             : z == ZP::Failed ? "failed (live block)"
                                               : "failed, recorded earlier in this process (live block)";
        const CP c = sp.channel_status();
        d["expectation_channel"] = c == CP::None ? "none"
                                 : c == CP::Built ? "built"
                                 : c == CP::Failed ? "failed (live block)"
                                                   : "skipped: zero-error plan failed (live block)";
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

// The forced-fault law's EXPANDED view: every (row, coin subset) — B * 2^u rows, row-major
// (row, subset), the subset's bit j = direction j; probability = row probability / 2^u.
static py::tuple expand_exact_law_core(py::ssize_t B, py::ssize_t u, const double* prob,
                                       const bool* det, py::ssize_t D, const bool* obs, py::ssize_t O,
                                       const bool* dec, py::ssize_t K, const double* ex, py::ssize_t R,
                                       const bool* udet, const bool* uobs, const bool* udec,
                                       const bool* usign) {
    // Overflow guard: B * 2^u must fit (callers check their row cap first; this is the last line).
    if (u < 0 || u > 62 || B < 0 || B > (std::numeric_limits<py::ssize_t>::max() >> u))
        throw std::invalid_argument("expand: B * 2^u rows overflow (u = " + std::to_string(u) + ")");
    const py::ssize_t S = (py::ssize_t)1 << u, N = B * S;
    auto mix = [&](const bool* r, const bool* g, py::ssize_t w) {
        py::array_t<bool> out({N, w});
        bool* o = out.mutable_data();
        for (py::ssize_t i = 0; i < B; ++i)
            for (py::ssize_t sub = 0; sub < S; ++sub) {
                bool* dst = o + (i * S + sub) * w;
                if (w) std::memcpy(dst, r + i * w, (size_t)w);
                for (py::ssize_t j = 0; j < u; ++j)
                    if ((sub >> j) & 1)
                        for (py::ssize_t t = 0; t < w; ++t) dst[t] ^= g[j * w + t];
            }
        return out;
    };
    py::array_t<double> p(N);
    double* pp = p.mutable_data();
    for (py::ssize_t i = 0; i < B; ++i)
        for (py::ssize_t sub = 0; sub < S; ++sub) pp[i * S + sub] = prob[i] / (double)S;
    py::array_t<double> e({N, R});
    double* ee = e.mutable_data();
    for (py::ssize_t i = 0; i < B; ++i)
        for (py::ssize_t sub = 0; sub < S; ++sub)
            for (py::ssize_t r = 0; r < R; ++r) {
                int par = 0;
                for (py::ssize_t j = 0; j < u; ++j) par ^= (int)(((sub >> j) & 1) && usign[j * R + r]);
                const double v = ex[i * R + r];
                ee[(i * S + sub) * R + r] = par ? -v : v;
            }
    return py::make_tuple(p, mix(det, udet, D), mix(obs, uobs, O), mix(dec, udec, K), e);
}
// The lazily expanded view of one law: built only when an expanded field is accessed. Refuses
// (ValueError) beyond max_rows without ever forming B * 2^u in a signed shift.
struct PyExactLawExpander {
    std::shared_ptr<qeccore::ExactLawResult> r;
    py::tuple expand(long max_rows) const {
        const py::ssize_t B = (py::ssize_t)r->prob.size(), u = r->u;
        if (u > 62 || max_rows < 0 || B > ((py::ssize_t)max_rows >> u))
            throw std::invalid_argument("expand: more than max_rows rows");
        return expand_exact_law_core(
            B, u, r->prob.data(), reinterpret_cast<const bool*>(r->det.data()), r->D,
            reinterpret_cast<const bool*>(r->obs.data()), r->O,
            reinterpret_cast<const bool*>(r->dec.data()), r->K, r->exps.data(), r->R,
            reinterpret_cast<const bool*>(r->udet.data()), reinterpret_cast<const bool*>(r->uobs.data()),
            reinterpret_cast<const bool*>(r->udec.data()), reinterpret_cast<const bool*>(r->usign.data()));
    }
};
// ── Forced faults (3.1.11): compile once, every measurement-outcome branch per fault set ─────
class PyExactBranchProgram {
public:
    qeccore::ScratchOwner scratch_owner_;
    std::string budget_invalid_;
    static constexpr const char* kBudgetWhat = "ExactBranchProgram";
    explicit PyExactBranchProgram(const std::string& text) {
        py::gil_scoped_release rel;
        prog_.reset(compile_exact_branch_program(text));
    }
    bool ok() const { return prog_->errors.empty() && prog_->program && !sampler_program_rejected(*prog_->program); }
    py::list errors() const { return errors_to_py(prog_->errors); }
    int reject_gate_index() const {
        return prog_->program ? sampler_program_reject_gate_index(*prog_->program) : -1;
    }
    std::string noise_refusal() const { return prog_->noise_refusal; }
    std::string record_refusal() const { return prog_->record_refusal; }
    py::dict refused_records() const {
        py::dict d;
        for (const auto& kv : prog_->refused_records) d[py::int_(kv.first)] = kv.second;
        return d;
    }
    std::vector<int> folded_records() const {
        std::vector<int> v;
        for (const auto& kv : prog_->folds) v.push_back(kv.first);
        return v;
    }
    py::list noise_sites() const {
        static const char* names[] = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
                                      "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR"};
        py::list out;
        for (const NoiseSite& s : prog_->sites) {
            py::dict d;
            d["channel"] = names[(int)s.channel];
            d["qubits"] = s.qubits;
            d["probs"] = s.probs;
            if (s.channel == NoiseChannel::CORRELATED) {
                // The chain's terms as Pauli words over `qubits` (one letter each), the address
                // alphabet of a forced fault on this site (target 0, a word over the same qubits).
                py::list terms;
                const size_t w = s.qubits.size();
                for (size_t t = 0; t < s.probs.size(); ++t) {
                    std::string word(w, 'I');
                    for (size_t i = 0; i < w; ++i) word[i] = "IXYZ"[s.corr[t * w + i]];
                    terms.append(word);
                }
                d["paulis"] = terms;
            }
            out.append(d);
        }
        return out;
    }
    // faults: list of (kind, index, target, pauli) with kind 0 = noise site, 1 = record flip
    py::dict exact_law(const std::vector<std::tuple<int, int, int, std::string>>& faults,
                       const std::vector<int>& exp_cols, long max_rows, double max_work,
                       bool no_columns) const {
        if (!ok()) throw std::invalid_argument("exact_branches: the circuit was not compiled (parse error or rejected)");
        std::vector<ForcedFault> ff;
        for (const auto& t : faults) {
            ForcedFault f;
            f.record = std::get<0>(t) == 1;
            f.index = std::get<1>(t);
            f.target = std::get<2>(t);
            f.pauli = std::get<3>(t);
            ff.push_back(std::move(f));
        }
        qeccore::ExactLawResult r;
        {
            py::gil_scoped_release rel;
            r = qeccore::exact_law(*prog_, ff, exp_cols, max_rows, max_work, no_columns);
        }
        if (!r.error.empty()) {
            if (r.error.rfind("internal:", 0) == 0) throw std::runtime_error("exact_branches: " + r.error);
            throw std::invalid_argument("exact_branches: " + r.error);
        }
        auto u8 = [&](const std::vector<uint8_t>& v, py::ssize_t rows, int w) {
            py::array_t<bool> a({rows, (py::ssize_t)w});
            auto* p = a.mutable_data();
            for (size_t i = 0; i < v.size(); ++i) p[i] = v[i] != 0;
            return a;
        };
        const py::ssize_t B = (py::ssize_t)r.prob.size(), U = (py::ssize_t)r.u;
        py::array_t<double> prob(B);
        std::memcpy(prob.mutable_data(), r.prob.data(), sizeof(double) * r.prob.size());
        py::array_t<double> ex({B, (py::ssize_t)r.R});
        if (!r.exps.empty()) std::memcpy(ex.mutable_data(), r.exps.data(), sizeof(double) * r.exps.size());
        py::dict d;
        d["probabilities"] = prob;
        d["detectors"] = u8(r.det, B, r.D);
        d["observables"] = u8(r.obs, B, r.O);
        d["decisions"] = u8(r.dec, B, r.K);
        d["expectations"] = ex;
        d["uniform_detectors"] = u8(r.udet, U, r.D);
        d["uniform_observables"] = u8(r.uobs, U, r.O);
        d["uniform_decisions"] = u8(r.udec, U, r.K);
        d["uniform_expectation_signs"] = u8(r.usign, U, r.R);
        d["magic_parities"] = r.d;
        d["expander"] = PyExactLawExpander{std::make_shared<qeccore::ExactLawResult>(std::move(r))};
        return d;
    }
private:
    struct Del { void operator()(ExactBranchProgram* p) const { delete_exact_branch_program(p); } };
    std::unique_ptr<ExactBranchProgram, Del> prog_;
};


py::dict py_export_dem_text(const std::string& text,
                            bool include_expectations,
                            bool decompose_errors,
                            bool ignore_decomposition_failures,
                            bool drop_gauge_observables,
                            const std::vector<int>& trusted_detectors,
                            const std::vector<int>& trusted_observables,
                            double approximate_disjoint_errors,
                            double pe_max_work) {
    if (!(pe_max_work > 0.0) || std::isnan(pe_max_work))
        throw std::invalid_argument("export_dem_text: pe_max_work must be a positive number (work units; "
                                    "float('inf') lifts the bound), got " + std::to_string(pe_max_work));
    DemExportResult dr;
    {
        py::gil_scoped_release rel;
        dr = export_dem(text, include_expectations,
                        decompose_errors, ignore_decomposition_failures,
                        drop_gauge_observables,
                        trusted_detectors.empty() ? nullptr : &trusted_detectors,
                        trusted_observables.empty() ? nullptr : &trusted_observables,
                        approximate_disjoint_errors, pe_max_work);
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
                // An E/ELSE chain's term SUPPORTS are structure (which Paulis can fire), not noise
                // values: included, as a PAULI_CHANNEL's slot order is implied by its name.
                if (ins.channel == NoiseChannel::CORRELATED) {
                    s += "c";
                    for (unsigned char c : ins.corr) s += (char)('0' + c);
                }
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
    int inv_shots = 0, inv_sized = 0;
    uint64_t inv_work = 0, inv_cwork = 0;
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
                } catch (const qeccore::BareStateCapacityError&) {
                    throw;
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
            // Work-sized (ref_compile.cpp noiseless_invariants): `shots` is the CEILING — run in
            // full wherever a noiseless shot is cheap (w1 = 0), otherwise clamp(W_compile/w1, 1,
            // shots): verification never costs more than the program compile.
            InvariantsResult ir = noiseless_invariants(lb, &lref, seed, shots);
            inv_shots = ir.shots;
            inv_sized = ir.sized_shots;
            inv_work = ir.shot_work;
            inv_cwork = ir.compile_work;
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
    d["invariants_shots"] = inv_shots;
    d["invariants_ceiling"] = shots < 0 ? 0 : shots;   // the battery's full count
    d["invariants_sized_shots"] = inv_sized;
    d["invariants_shot_work"] = inv_work;
    d["invariants_compile_work"] = inv_cwork;
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
    d["invariants_shots"] = r.invariants_shots;
    d["invariants_ceiling"] = r.invariants_ceiling;
    d["invariants_sized_shots"] = r.invariants_sized_shots;
    d["invariants_shot_work"] = r.invariants_shot_work;
    d["invariants_compile_work"] = r.invariants_compile_work;
    py::dict gates;
    for (const auto& g : r.gates) gates[py::str(g.first)] = g.second;
    d["gates"] = gates;
    return d;
}

// ── PyFramedSuperposition — Python handle for a FramedSuperposition (S2.1/S2.2) ──────────
// S2.1: opaque handle; created by _bare_state_of and passed to TwirlSampler(state,…).
// S2.2 additions: approx_equal (exact_sum_overlap-based, global-phase insensitive) +
//   apply_clifford (for the mutation check: mutate a copy, verify approx_equal fails).
// PyFramedSuperposition: cpp/bindings/framed_py.hpp (J-b1 F; shared with the test API)

// ── S2.1: _bare_state_of — build the FramedSuperposition bare state of a circuit ─────────
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

// Task 2 free function: _port_contract_of_state(state, wires) -> dict.
// Takes a PyFramedSuperposition and a list of wire indices; returns the port verdict dict.
// Private (underscore prefix). Used by the Task 3 invariance gate and by differential tests.
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

// 3.1.3 (docs/frame_cosets_cpp.md): the compiled Born SECTOR table of an S-layer plan — the
// `cpp` backend of xtim.frames.plan_frame_cosets_fast.  Inputs are the branch_frames() export's
// arrays (unpacked uint8 bit rows, the numpy layout) and the plan's a-mask; the output is the
// numpy path's list of dicts, bit-for-bit (same sectors, same first-encounter order, same
// fields / dtypes, same `weight` bits).  Refusals map to the numpy path's: ValueError for a
// null-space basis read that is not ±1, AssertionError for an export that is not a full frame.
// Read-only, no engine state; the sampling hot path does not touch this code.
using _u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
static py::list py_plan_frame_cosets(
    _u8arr stab_x, _u8arr stab_z, py::array_t<int8_t, py::array::c_style | py::array::forcecast> stab_phase,
    _u8arr destab_x, _u8arr destab_z, py::array_t<int32_t, py::array::c_style | py::array::forcecast> free_rows,
    _u8arr sigma, py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeff,
    _u8arr branch_x, _u8arr branch_z, _u8arr a_mask) {
    using namespace qeccore::frame_cosets;
    if (stab_x.ndim() != 2 || stab_x.shape(0) != stab_x.shape(1))
        throw py::value_error("plan_frame_cosets: stab_x must be (n, n)");
    const int n = (int)stab_x.shape(0);
    auto need = [&](const py::array& a, int r, int c, const char* name) {
        if (a.ndim() != 2 || a.shape(0) != r || a.shape(1) != c)
            throw py::value_error(std::string("plan_frame_cosets: ") + name + " has the wrong shape");
    };
    need(stab_z, n, n, "stab_z"); need(destab_x, n, n, "destab_x"); need(destab_z, n, n, "destab_z");
    if (stab_phase.ndim() != 1 || stab_phase.shape(0) != n) throw py::value_error("plan_frame_cosets: stab_phase must be (n,)");
    if (free_rows.ndim() != 1) throw py::value_error("plan_frame_cosets: free must be (k,)");
    const int k = (int)free_rows.shape(0);
    if (coeff.ndim() != 1) throw py::value_error("plan_frame_cosets: coeff must be (chi,)");
    const int chi = (int)coeff.shape(0);
    if (sigma.ndim() != 2 || sigma.shape(0) != chi || sigma.shape(1) != k)
        throw py::value_error("plan_frame_cosets: sigma must be (chi, k)");
    need(branch_x, chi, n, "branch_x"); need(branch_z, chi, n, "branch_z");
    if (a_mask.ndim() != 1 || a_mask.shape(0) != n) throw py::value_error("plan_frame_cosets: a_mask must be (n,)");
    std::vector<int> fr(k);
    for (int d = 0; d < k; ++d) {
        fr[d] = (int)free_rows.data()[d];
        if (fr[d] < 0 || fr[d] >= n) throw py::value_error("plan_frame_cosets: free row out of range");
    }
    // χ-scaled transient (branch Pauli rows + sigma + coeff): charged to the state budget first.
    qeccore::StateBudgetTicket frame_ticket(
        (int64_t)chi * (int64_t)(2 * 8 * FrameRef::words(n) + k + 16), "plan_frame_cosets' packed branch table");
    const FrameRef f = FrameRef::from_bits(n, stab_x.data(), stab_z.data(), stab_phase.data(),
                                           destab_x.data(), destab_z.data(), fr.data(), k, sigma.data(),
                                           coeff.data(), chi, branch_x.data(), branch_z.data());
    std::vector<Sector> tab;
    try {
        tab = plan_frame_cosets(f, a_mask.data());
    } catch (const FrameNotFull& e) {
        PyErr_SetString(PyExc_AssertionError, e.what());
        throw py::error_already_set();
    }
    py::list out;
    auto bits = [n](const std::vector<uint8_t>& v) {
        py::array_t<uint8_t> a(n);
        std::memcpy(a.mutable_data(), v.data(), (size_t)n);
        return a;
    };
    for (const Sector& s : tab) {
        py::dict d;
        d["vz"] = bits(s.vz);
        d["fx"] = bits(s.fx);
        d["fz"] = bits(s.fz);
        if (s.has_flip) d["logical_flip"] = py::make_tuple(py::cast(s.xb), py::cast(s.zb));
        else d["logical_flip"] = py::none();
        d["weight"] = py::float_(s.weight);
        d["frame_copy"] = py::bool_(s.frame_copy);
        out.append(std::move(d));
    }
    return out;
}

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
    mdef(m, "plan_frame_cosets", &py_plan_frame_cosets,
          py::arg("stab_x"), py::arg("stab_z"), py::arg("stab_phase"), py::arg("destab_x"),
          py::arg("destab_z"), py::arg("free"), py::arg("sigma"), py::arg("coeff"),
          py::arg("branch_x"), py::arg("branch_z"), py::arg("a_mask"),
          "3.1.3: the Born SECTOR table of an S-layer plan on a branch_frames() reference "
          "(xtim.frames.plan_frame_cosets_fast's compiled backend; bit-for-bit the numpy "
          "path's list of dicts). docs/frame_cosets_cpp.md.");
    m.doc() = "xtim engine bindings: parse / packed sampling / DEM export / .ref surface.";
    m.attr("ENGINE_VERSION") = kEngineVersion;
    // J-b1 P13: the runtime exact-construction counter and the explicit-exact scope (xtim/_exact_scope.py).
    m.def("_exact_constructions", []() { return qeccore::exact_constructions(); });
    m.def("_reset_exact_constructions", []() { qeccore::reset_exact_constructions(); });
    m.def("_explicit_exact_enter", []() { qeccore::explicit_exact_enter(); });
    m.def("_explicit_exact_exit", []() { qeccore::explicit_exact_exit(); });
    m.attr("EXACT_LAW_WORK_DEFAULT") = qeccore::kExactLawWorkDefault;
    m.attr("PE_MAX_WORK_DEFAULT") = qeccore::kPeMaxWorkDefault;   // DEM export pe_max_work default
    m.attr("REF_FORMAT_VERSION") = kRefFormatVersion;
    m.attr("FRAME_COSETS_ABI") = kFrameCosetsAbi;

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
    py::class_<CompiledProgram>(m, "CompiledProgram",
          "Stateful compiled circuit: parse + compile ONCE in the constructor, then sample() "
          "repeatedly with no recompile -> packed b8 record buffers + float64 expectations.")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("text"), py::arg("ref_text") = std::string())
        .def("sample", &BudgetGuard<&CompiledProgram::sample>::call, py::arg("shots"), py::arg("seed"),
             py::arg("want_meas") = true, py::arg("want_det") = true,
             py::arg("want_obs") = true, py::arg("want_exp") = true,
             "Sample the compiled program -> packed b8 buffers + expectations; skips the "
             "per-call compile. want_meas/want_det/want_obs/want_exp "
             "(default all True): build only the requested record channels as numpy arrays; an "
             "unrequested channel's dict key is omitted (scalar metadata always present).")
        .def("plan_report", &CompiledProgram::plan_report,
             "How the zero-error TreePlan and the R>0 expectation channel came out.")
        .def_property_readonly("ok", &CompiledProgram::ok)
        .def_property_readonly("ref_error", &CompiledProgram::ref_error)
        .def_property_readonly("errors", &CompiledProgram::errors);
    py::class_<PyExactBranchProgram>(m, "ExactBranchProgram",
          "Forced faults, exact branches (3.1.11): parse + compile ONCE (the exact sampler's program), "
          "then exact_law(faults) returns the exact FACTORED law of (detectors, observables, decisions): "
          "structured rows (probability, targets, conditional expectations) plus independent "
          "fair-coin directions. Read-only: never touches any RNG or sampled stream.")
        .def(py::init<const std::string&>(), py::arg("text"))
        .def_property_readonly("ok", &PyExactBranchProgram::ok)
        .def_property_readonly("errors", &PyExactBranchProgram::errors)
        .def_property_readonly("reject_gate_index", &PyExactBranchProgram::reject_gate_index)
        .def_property_readonly("noise_refusal", &PyExactBranchProgram::noise_refusal)
        .def_property_readonly("record_refusal", &PyExactBranchProgram::record_refusal)
        .def_property_readonly("refused_records", &PyExactBranchProgram::refused_records)
        .def_property_readonly("folded_records", &PyExactBranchProgram::folded_records)
        .def("noise_sites", &BudgetGuard<&PyExactBranchProgram::noise_sites>::call)
        .def("exact_law", &BudgetGuard<&PyExactBranchProgram::exact_law>::call,
             py::arg("faults"), py::arg("exp_cols") = std::vector<int>{},
             py::arg("max_rows") = 1L << 20, py::arg("max_work") = qeccore::kExactLawWorkDefault,
             py::arg("no_columns") = false);
    py::class_<PyExactLawExpander>(m, "ExactLawExpander",
          "Holds one factored forced-fault law; expand(max_rows) builds the expanded view.")
        .def("expand", &PyExactLawExpander::expand, py::arg("max_rows"));
    mdef(m, "export_dem_text", &py_export_dem_text, py::arg("text"),
          py::arg("include_expectations") = true,
          py::arg("decompose_errors") = false,
          py::arg("ignore_decomposition_failures") = false,
          py::arg("drop_gauge_observables") = false,
          py::arg("trusted_detectors") = std::vector<int>{},
          py::arg("trusted_observables") = std::vector<int>{},
          py::arg("approximate_disjoint_errors") = 0.0,
          py::arg("pe_max_work") = qeccore::kPeMaxWorkDefault,
          "Stim-compatible DEM text export (run_stim_main --dem semantics). "
          "pe_max_work: deterministic work cap of the exact per-shot analysis of a "
          "PAULI_EXPECTATION column (work units per branch walk, default 1.5e7 ~0.5 s; a count, "
          "never wall-clock). Past it, include_expectations=True refuses by name (the message "
          "names the argument) and include_expectations=False reports the column in "
          "dropped_expectation_columns. "
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
    mdef(m, "carried_symplectic_action", &py_carried_symplectic_action, py::arg("text"),
          "D1: GF(2) symplectic map of the stage's Clifford action on carried wires — maps "
          "(x‖z) frame bits on INPUT_QUBITS wires (t=0) to (x‖z) frame bits on OUTPUT_QUBITS "
          "wires (t=exit), by conjugating each basis Pauli through the stage's Clifford content "
          "(sign dropped; frames projective). Support pushed onto measured/discarded wires is "
          "omitted (accounted by the input-frame channel relabel). Returns {ok, error, in_wires, "
          "out_wires, M} where M is a (2*n_in)x(2*n_out) uint8 array: row r<n_in = image of "
          "X_{in_wires[r]}, row r>=n_in = image of Z_{in_wires[r-n_in]}; columns [x over out_wires "
          "| z over out_wires]. A non-Clifford gate in the body -> ok=False (loud).");
    mdef(m, "dem_column_layout", &py_dem_column_layout, py::arg("text"),
          "DEM column layout for a module-system circuit (structural query, no DEM export). "
          "Keys: ok, errors, rejected, reject_gate_index, error, n_obs (O+R, existing L-columns), "
          "frame [(orig_qubit, col_x, col_z), ...] (OUTPUT_QUBITS frame columns), "
          "logical_flip [(decision_k, col), ...] (DECISION logical-flip columns). "
          "Column convention: col_x fires when the error mechanism has X-support on the "
          "qubit (an X correction is needed); col_z fires when Z-support (Z correction needed). "
          "The orchestrator maps frame col_x -> X-part of input_pauli, col_z -> Z-part.");
    mdef(m, "ref_info", &py_ref_info, py::arg("ref_text"),
          "Header info of a .ref text (no state construction).");
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
    mdef(m, "bare_state_memo_clear", []() { qeccore::bare_state_memo_clear(); },
          "Drop every entry of the bare-state memo (counters kept).");
    mdef(m, "bare_state_memo_reset_stats", []() { qeccore::bare_state_memo_reset_stats(); },
          "Zero the bare-state memo counters (entries kept).");
    mdef(m, "bare_state_memo_set_caps", [](size_t max_entries, size_t max_bytes) {
            qeccore::bare_state_memo_set_caps(max_entries, max_bytes); },
          py::arg("max_entries"), py::arg("max_bytes"),
          "Re-bound the bare-state memo (entries over either cap are evicted now).");
    mdef(m, "bare_state_memo_set_entry_cap", [](size_t max_entry_bytes) {
            qeccore::bare_state_memo_set_entry_cap(max_entry_bytes); },
          py::arg("max_entry_bytes"),
          "Per-entry byte cap of the bare-state memo: a larger state is never captured.");
    mdef(m, "deferred_signature", &py_deferred_signature, py::arg("circuit_text"),
          "Structural signature of the normalized+deferred circuit (noise values excluded); "
          "used by the reference cache to reject frame-divergent hits.");
    mdef(m, "_tree_fail_memo_stats", [] {
              const qeccore::TreeFailMemoStats st = qeccore::tree_fail_memo_stats();
              py::dict d;
              d["hits"] = st.hits;
              d["misses"] = st.misses;
              d["stores"] = st.stores;
              return d;
          },
          "Diagnostics: the failed-TreePlan-extraction memo's counters (framed_sampler.cpp).");
    mdef(m, "_tree_fail_memo_clear", [] { qeccore::tree_fail_memo_clear(); },
          "Diagnostics/tests: empty the failed-TreePlan-extraction memo and zero its counters.");
    mdef(m, "ref_verify_cheap", &py_ref_verify_cheap, py::arg("circuit_text"),
          py::arg("ref_text"), py::arg("seed") = 12345, py::arg("shots") = 200,
          "Cheap cache-hit verify: load + n-match + noiseless invariants (work-sized: `shots` "
          "is the ceiling, run in full where a noiseless shot is cheap, otherwise "
          "clamp(compile work / shot work, 1, shots); result['invariants_shots'] is the count "
          "run, result['invariants_ceiling'] the full count); the exact "
          "state-overlap gate runs only when the invariants are vacuous (0 det + 0 exp).");
    mdef(m, "ref_compile_text", &py_ref_compile_text, py::arg("circuit_text"),
          py::arg("source") = std::string("xtim-circuit"), py::arg("oracle_chi") = 256,
          py::arg("seed") = 12345, py::arg("shots") = 2000,
          "The FULL ref_compile gate battery (compile mode); returns .ref text on success. "
          "The noiseless-invariants gate is work-sized (500 shots wherever a noiseless shot is "
          "cheap, otherwise clamp(compile work / shot work, 1, 500): verification never costs "
          "more than compilation; result['invariants_shots'] is the count run, "
          "result['invariants_ceiling'] the full count).");
    bind_framed_superposition(m);   // J-b1 F: cpp/bindings/framed_py.hpp
    // J-b1 F (fixplan-F-materialize-c0hat r2 item 1): the layout-3 BarrierBuffer returns THIS extension's handle.
    xtim_api::wrap_state_hook() = [](FramedSuperposition&& st) -> py::object {
        return py::cast(PyFramedSuperposition(std::move(st)));
    };
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
    mdef(m, "bare_state_of",  &py_bare_state_of, py::arg("circuit_text"), _bare_state_of_doc);
    mdef(m, "_bare_state_of", &py_bare_state_of, py::arg("circuit_text"), _bare_state_of_doc); // public + back-compat alias, see convention note above
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
    auto twirl_ctor = [](const std::string& text, double p_factor, const std::string& disk_cache, long selfcheck,
                         const std::string& disk_cache_auto_dir, bool decisions, const PyIdealState* ideal) {
        std::unique_ptr<PyFamilySampler> s;
        {
            qeccore::ScratchCallScope scope;
            s = std::make_unique<PyFamilySampler>(text, p_factor, selfcheck, /*oracle_all_shots=*/false, decisions,
                                                  /*seed=*/1, /*record_route_term=*/false, ideal, disk_cache,
                                                  disk_cache_auto_dir);
        }
        if (ideal) {
            s->btab_ = std::make_shared<BranchTable>(ideal->tok);
            s->producer_ = ideal->producer;
        }
        return s;
    };
    py::class_<PyFamilySampler>(m, "TwirlSampler", py::module_local(),
          "The twirl record sampler (qeccore::TwirlRecordSampler, the family record law; design "
          "docs/twirl_record_sampler.md). Every DETECTOR / OBSERVABLE / DECISION column is a record of the law: "
          "deterministic channels exact, gauge (LOGICAL) detectors exact Born records, ANTI detectors and second "
          "LOGICAL / ANTI observables served, Born decisions and retention on CH / PPR circuits served. Circuits "
          "outside the twirl class raise xtim.errors.XtimTwirlRefusal by name at construction. Thread safety: one "
          "sampler object must not be used from two threads at once; separate samplers are independent.")
        .def(py::init([twirl_ctor](const std::string& text, double p_factor, const std::string& disk_cache,
                                   long selfcheck, const std::string& disk_cache_auto_dir, bool decisions) {
                 return twirl_ctor(text, p_factor, disk_cache, selfcheck, disk_cache_auto_dir, decisions, nullptr);
             }),
             py::arg("circuit_text"), py::arg("p_factor") = 1.0, py::arg("disk_cache") = std::string(),
             py::arg("selfcheck") = 2000, py::arg("disk_cache_auto_dir") = std::string(),
             py::arg("decisions") = true)
        .def_static("from_ideal_state",
             [twirl_ctor](const PyIdealState& ideal_state, const std::string& circuit_text, double p_factor,
                          const std::string& disk_cache, long selfcheck, const std::string& disk_cache_auto_dir,
                          bool decisions) {
                 return twirl_ctor(circuit_text, p_factor, disk_cache, selfcheck, disk_cache_auto_dir, decisions,
                                   &ideal_state);
             },
             py::kw_only(), py::arg("ideal_state"), py::arg("circuit_text"), py::arg("p_factor") = 1.0,
             py::arg("disk_cache") = std::string(), py::arg("selfcheck") = 2000,
             py::arg("disk_cache_auto_dir") = std::string(), py::arg("decisions") = true,
             py::keep_alive<0, 1>(),
             "A consumer sampler whose INPUT_QUBITS carry `ideal_state` (branch 0, state-level injection; every "
             "carried state, single-qubit stabilizer states included). DETECTOR / OBSERVABLE columns are "
             "reference-relative (the noiseless run from the ideal state); add_branch registers producer branches.")
        .def("sample", [](PyFamilySampler& s, long shots, py::object seed, py::object input_pauli,
                          py::object input_branch) -> py::tuple {
                 if (shots > (long)INT32_MAX) throw std::invalid_argument("shots exceeds INT32_MAX (b8 array shape is int)");
                 py::tuple t = budget_poison_on_refusal(s, [&] {
                     qeccore::ScratchCallScope scope;
                     return s.sample(shots, seed, false, input_pauli, input_branch);
                 });
                 if (s.num_dec_ > 0) return t;      // Task 2b.1: (dets, obs, decisions); decision-free: (dets, obs)
                 return py::make_tuple(t[0], t[1]);
             },
             py::arg("shots"), py::arg("seed") = py::none(), py::arg("input_pauli") = py::none(),
             py::arg("input_branch") = py::none(),
             "Sample -> (detectors, observables[, decisions]), Stim-b8-packed uint8 arrays (shots, ceil(K/8)). "
             "seed=None continues the stream; an explicit seed reseeds in place (same seed -> same bytes). "
             "input_pauli: optional uint8 (shots, 2*support) per-shot Pauli frame on the INPUT_QUBITS port (X part "
             "then Z part). input_branch: optional int32 (shots,) per-shot carried branch (0 = the ideal state; "
             "from_ideal_state samplers).")
        .def("sample_barrier", [](py::object self, long shots, py::object seed, py::object input_pauli, bool compact,
                                  py::object input_branch) {
                 PyFamilySampler& s = self.cast<PyFamilySampler&>();
                 return budget_poison_on_refusal(s, [&] {
                     qeccore::ScratchCallScope scope;
                     return family_sample_barrier(self, shots, seed, input_pauli, compact, input_branch);
                 });
             },
             py::arg("shots"), py::arg("seed") = py::none(), py::arg("input_pauli") = py::none(),
             py::arg("compact") = true, py::arg("input_branch") = py::none(), py::keep_alive<0, 1>(),
             "Sample shots keeping every shot's layout-3 record -> BarrierBuffer (dets / obs / decisions as sample(); "
             "per shot the record key [kind][len] content ‖ P_rec ‖ y and the outcomes u ‖ z; materialize(i) rebuilds "
             "the retained state from the record). A layout-3 buffer is always compact (compact=False is refused).")
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
             "disk_path, feedback_hoisted, branches (branch counters), branch_table_digest. born_trie's node_evals / "
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
    // input_state_of is the public name; _input_state_of is a backward-compat alias, see convention note above.
    // Validation runs in C++ inside py_input_state_of; both names validate identically.
    const char* _input_state_of_doc =
        "Build the frame-explicit bare state of `circuit_text` AND return the deferred-space "
        "wire indices of its declared OUTPUT_QUBITS (final_wire[orig], declaration order). "
        "Returns (FramedSuperposition, list[int]). The state carries the PHYSICAL carried "
        "qubits when the prep has no trailing measurement on them (frame-baked). Used by "
        "compile_twirl_sampler_from_ideal_state(..., ideal_state=input_state_of(prep)) to locate the carried "
        "qubits in the provided state. "
        "Raises ValueError on parse failure, bare-state rejection, or an entangled / "
        "non-stabilizer-rest port (both names validate).";
    mdef(m, "input_state_of",  &py_input_state_of, py::arg("circuit_text"), _input_state_of_doc);
    mdef(m, "_input_state_of", &py_input_state_of, py::arg("circuit_text"), _input_state_of_doc); // public + back-compat alias, see convention note above
    mdef(m, "input_qubits", &py_input_qubits, py::arg("text"),
          "Return the flattened INPUT_QUBITS port qubit indices (declaration order). "
          "Empty for a port-free circuit. Raises ValueError on parse failure.");
    mdef(m, "_output_wires_of", &py_output_wires_of, py::arg("text"),
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
    mdef(m, "port_contract_of_state", &py_port_contract_of_state,
          py::arg("state"), py::arg("wires"),
          _port_contract_docstring);
    mdef(m, "_port_contract_of_state", &py_port_contract_of_state,
          py::arg("state"), py::arg("wires"),
          _port_contract_docstring); // public + back-compat alias, see convention note above

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
