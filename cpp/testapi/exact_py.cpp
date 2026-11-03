// exact-out T5 (owner ruling 2026-10-30) — TEST API (not a public interface): the EXACT engine's entry points.
// The exact engine (TreeBuilder sampler, forced-fault law, reference battery) is the test oracle; its bindings moved
// here from cpp/bindings/xtim_py.cpp verbatim:
//   * CompiledProgram   — compile a circuit ONCE, sample() it repeatedly -> all four packed record buffers (Stim b8
//                         uint8 buffers + row-major float64 expectations), optional supplied .ref text;
//   * ExactBranchProgram / ExactLawExpander — forced faults, exact factored law (3.1.11);
//   * ref_info / ref_verify_cheap / ref_compile_text / deferred_signature — the stored-reference surface: header
//                         info, the cheap load+invariants verify, the FULL ref_compile gate battery (compile mode,
//                         refcompile::run_compile_battery from cpp/apps/ref_compile.cpp with REF_COMPILE_NO_MAIN),
//                         and the structural deferred-circuit signature of the reference cache;
//   * _tree_fail_memo_stats / _clear — the exact sampler's failed-TreePlan memo diagnostics;
//   * this extension's copies of the process diagnostics the exact-engine tests read (its engine copy has its own
//     state budget, bare-state memo and scratch).
// The Python layer is tests/_exact_engine.py (+ tests/_exact_forced.py, tests/_exact_reference.py).
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"
#include "qeccore/bare_state_memo.hpp"
#include "qeccore/engine_version.hpp"
#include "qeccore/exact_scope.hpp"
#include "qeccore/framed_sampler.hpp"
#include "qeccore/run_stim.hpp"
#include "qeccore/state_budget.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../bindings/common_py.hpp"

namespace py = pybind11;
using namespace qeccore;
using namespace refcompile;  // the shared ref_compile gate battery (REF_COMPILE_NO_MAIN)

namespace {

constexpr int kRefFormatVersion = 4;  // the .ref on-disk format the engine WRITES (ref_from_state_v4 ->
                                      // "ref-format 4": written FROM the framed state, anchor block
                                      // retired, coefficients as-is — Phase B3). Feeds the reference
                                      // cache key + the ref_format_version metadata; the LOADER
                                      // dispatches on each file's own `version` field, so older v2/v3
                                      // refs still load. NOTE older xtim versions cannot read v4 files.

py::list errors_to_py(const std::vector<ParseError>& es) {
    py::list out;
    for (const auto& e : es) out.append(py::make_tuple(e.line, e.message));
    return out;
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

}  // namespace

void register_exact(py::module_& m) {
    m.attr("ENGINE_VERSION") = kEngineVersion;
    m.attr("REF_FORMAT_VERSION") = kRefFormatVersion;
    m.attr("EXACT_LAW_WORK_DEFAULT") = qeccore::kExactLawWorkDefault;
    py::class_<CompiledProgram>(m, "CompiledProgram",
          "Stateful compiled circuit on the EXACT engine: parse + compile ONCE in the constructor, then sample() "
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
    mdef(m, "ref_info", &py_ref_info, py::arg("ref_text"),
          "Header info of a .ref text (no state construction).");
    mdef(m, "deferred_signature", &py_deferred_signature, py::arg("circuit_text"),
          "Structural signature of the normalized+deferred circuit (noise values excluded); "
          "used by the reference cache to reject frame-divergent hits.");
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
    // This extension's engine copy: its own state budget, bare-state memo and scratch (the exact-engine tests read
    // these; the production twins are xtim._xtim's).
    mdef(m, "live_state_bytes", []() { return (long long)qeccore::live_state_bytes(); },
          "Bytes of chi-scaled state storage live in THIS extension's engine copy.");
    mdef(m, "_thread_scratch_bytes", []() { return (long long)qeccore::thread_scratch_bytes(); },
          "Bytes of chi-scaled measurement scratch charged by this thread (this extension's engine copy).");
    mdef(m, "_scratch_owner_count", []() { return (long long)qeccore::scratch_owner_count(); },
          "Live Python-held owners of this extension's engine copy.");
    mdef(m, "_release_thread_state_scratch", []() { qeccore::release_thread_state_scratch(); },
          "Free this thread's chi-scaled measurement scratch of this extension's engine copy now (diagnostics).");
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
          "Counters of this extension's bare-state memo (the production twin: xtim._xtim.bare_state_memo_stats).");
    mdef(m, "bare_state_memo_clear", []() { qeccore::bare_state_memo_clear(); },
          "Drop every entry of this extension's bare-state memo (counters kept).");
    mdef(m, "bare_state_memo_reset_stats", []() { qeccore::bare_state_memo_reset_stats(); },
          "Zero this extension's bare-state memo counters (entries kept).");
}
