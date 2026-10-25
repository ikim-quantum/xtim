#include "qeccore/exact_scope.hpp"   // J-b1 P13: the exact-construction counter
#include "qeccore/run_stim.hpp"
#include <cstddef>
#include "qeccore/sampler.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/coherentize_feedback.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace qeccore {

StimPackedResult run_stim_packed(const std::string& text, int shots, uint64_t seed,
                                 const FramedSuperposition* reference) {
    StimPackedResult out;
    ParsedStim p = parse_stim_circuit(text);
    if (!p.ok()) { out.errors = std::move(p.errors); return out; }
    // Classically-controlled Pauli feedback (CX/CY/CZ rec[-k] q) is handled by the sampler as an
    // exact post-sampling triangular record/expectation relabel (Task 2: strip → propagate →
    // relabel, off the state-evolution hot loop). It flows through run_shots_packed unchanged.
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    // index-dense observable channel (gaps = empty parity lists -> constant 0, Stim convention)
    std::vector<std::vector<int>> obs((size_t)(max_obs + 1));
    for (auto& kv : p.observables) obs[kv.first] = kv.second;
    out.records = run_shots_packed(p.circuit, shots, seed, p.detectors, obs,
                                   reference);
    return out;
}

CompiledStimProgram compile_stim_program(const std::string& text,
                                         const FramedSuperposition* reference) {
    CompiledStimProgram out;
    ParsedStim p = parse_stim_circuit(text);
    if (!p.ok()) { out.errors = std::move(p.errors); return out; }
    // index-dense observable channel (gaps = empty parity lists -> constant 0, Stim convention) —
    // identical marshalling to run_stim_packed, so the compiled program matches run_shots_packed.
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    std::vector<std::vector<int>> obs((size_t)(max_obs + 1));
    for (auto& kv : p.observables) obs[kv.first] = kv.second;
    out.program = compile_sampler_program(p.circuit, p.detectors, obs, reference);
    return out;
}

ExactBranchProgram* compile_exact_branch_program(const std::string& text) {
    note_exact_construction();                   // J-b1 P13
    auto* out = new ExactBranchProgram();
    ParsedStim p = parse_stim_circuit(text);
    if (!p.ok()) { out->errors = std::move(p.errors); return out; }
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    std::vector<std::vector<int>> obs((size_t)(max_obs + 1));
    for (auto& kv : p.observables) obs[kv.first] = kv.second;
    // PROVENANCE: tag every user noise instruction with its site index (metadata only — no pass
    // reads it, so the compiled program is the sampler's, byte for byte) and record each Measure's
    // record ordinal for the coherentized-source scan below.
    std::vector<int> record_of_stream(p.circuit.stream.size(), -1);
    {
        int site = 0, rec = 0;
        for (size_t i = 0; i < p.circuit.stream.size(); ++i) {
            Instr& ins = p.circuit.stream[i];
            if (ins.kind == Instr::Kind::Noise) {
                ins.origin = Instr::Origin::UserSite;
                ins.origin_index = site++;
                out->sites.push_back({ins.channel, ins.qubits, ins.probs, -1, {}, ins.corr});
            } else if (ins.kind == Instr::Kind::Measure) {
                record_of_stream[i] = rec++;
            }
        }
    }
    // AnyPauli: a forced fault may name ANY Pauli at a site (docs/xtim_forced_faults.md).
    out->program = compile_sampler_program(p.circuit, p.detectors, obs, nullptr, AtomSet::AnyPauli);
    out->law_cache = new_exact_law_cache(*out->program);
    for (auto& d : p.decisions) out->decisions.push_back(d.second);
    if (sampler_program_rejected(*out->program)) return out;
    const Circuit& def = sampler_program_deferred(*out->program);
    // Map by tag: user site i -> the deferred noise instruction tagged UserSite i (exactly one,
    // same channel / arity / probabilities); record m -> its ReadoutFold. Anything else (an
    // untagged or doubly-tagged noise instruction) is a provenance mismatch: refuse by name.
    for (int si = 0; si < (int)def.stream.size() && out->noise_refusal.empty(); ++si) {
        const Instr& d = def.stream[(size_t)si];
        if (d.kind != Instr::Kind::Noise) continue;
        if (d.origin == Instr::Origin::UserSite && d.origin_index >= 0 &&
            d.origin_index < (int)out->sites.size()) {
            // One user site -> one or more deferred parts (a frame commuted through the site may
            // relabel and split it): every user qubit position must be covered exactly once, by a
            // part of the same arity class.
            NoiseSite& s = out->sites[(size_t)d.origin_index];
            const size_t nq = s.qubits.size();
            if (s.deferred_index_of.empty()) {
                s.deferred_index_of.assign(nq, -1);
                s.deferred_qubits.assign(nq, -1);
                s.deferred_letter.assign(3 * nq, 0);
            }
            auto two_q = [](NoiseChannel c) {
                return c == NoiseChannel::DEPOLARIZE2 || c == NoiseChannel::PAULI_CHANNEL_2;
            };
            const NoiseFrameProvenance* fp = d.frame_prov.get();
            bool bad = two_q(d.channel) != two_q(s.channel) ||
                       (fp && (fp->pos.size() != d.qubits.size() ||
                               fp->letter.size() != 3 * d.qubits.size())) ||
                       (!fp && (d.qubits.size() != nq || d.channel != s.channel ||
                                d.probs != s.probs || d.corr != s.corr)) ||
                       (fp && (d.channel == NoiseChannel::CORRELATED) != (s.channel == NoiseChannel::CORRELATED));
            for (size_t j = 0; !bad && j < d.qubits.size(); ++j) {
                const int pos = fp ? fp->pos[j] : (int)j;
                if (pos < 0 || (size_t)pos >= nq || s.deferred_index_of[(size_t)pos] >= 0) { bad = true; break; }
                s.deferred_index_of[(size_t)pos] = si;
                s.deferred_qubits[(size_t)pos] = d.qubits[j];
                for (int c = 1; c <= 3; ++c)
                    s.deferred_letter[3 * (size_t)pos + (size_t)c - 1] =
                        fp ? fp->letter[3 * j + (size_t)c - 1] : (unsigned char)c;
            }
            if (bad) {
                out->noise_refusal = "noise site " + std::to_string(d.origin_index) +
                    " does not correspond to its normalized noise instructions";
                break;
            }
            if (s.deferred_stream_index < 0) s.deferred_stream_index = si;
        } else if (d.origin == Instr::Origin::ReadoutFold && d.qubits.size() == 1 &&
                   (d.channel == NoiseChannel::X_ERROR || d.channel == NoiseChannel::Y_ERROR ||
                    d.channel == NoiseChannel::Z_ERROR) &&
                   !out->folds.count(d.origin_index)) {
            // The fold fires its OWN (possibly frame-relabelled) channel letter.
            out->folds[d.origin_index] = {si, d.qubits[0],
                                          d.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                                          : d.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z};
        } else {
            out->noise_refusal = "the normalized circuit's noise instruction " + std::to_string(si) +
                " has no recognised provenance (neither a user noise site nor one readout-flip fold)"
                " — noise sites cannot be addressed";
        }
    }
    if (out->noise_refusal.empty())
        for (size_t i = 0; i < out->sites.size(); ++i)
            if (out->sites[i].deferred_stream_index < 0 ||
                std::find(out->sites[i].deferred_index_of.begin(), out->sites[i].deferred_index_of.end(), -1) !=
                    out->sites[i].deferred_index_of.end()) {
                out->noise_refusal = "noise site " + std::to_string(i) +
                    " has no normalized noise instruction — noise sites cannot be addressed";
                break;
            }
    // Records read by COHERENTIZED feedback (the sampler's own decision: the byproduct crosses a
    // non-Clifford gate). A folded one is forced through its fold; a readout-noiseless one has no
    // fold channel, so a record-tail flip would not reach its coherent control: refuse it by name.
    bool any_fb = false;
    for (const Instr& ins : p.circuit.stream) if (ins.kind == Instr::Kind::ControlledPauli) { any_fb = true; break; }
    if (any_fb) {
        const std::vector<bool> crosses = feedback_crosses_nonclifford(p.circuit);
        int ord = 0;
        for (int i = 0; i < (int)p.circuit.stream.size(); ++i) {
            if (p.circuit.stream[(size_t)i].kind != Instr::Kind::ControlledPauli) continue;
            if (!crosses[(size_t)ord++]) continue;
            const FeedbackSource src = resolve_feedback_source(p.circuit, i);
            const int rec = src.resolved ? record_of_stream[(size_t)src.measure_stream_index] : -1;
            if (rec < 0) {
                out->record_refusal = "a coherentized feedback's source record could not be resolved";
                continue;
            }
            if (src.readout_flip_p > 0.0) {
                if (!out->folds.count(rec))
                    out->refused_records[rec] = "record " + std::to_string(rec) +
                        " feeds coherentized feedback but its readout-flip fold was not found";
            } else {
                out->refused_records[rec] = "record " + std::to_string(rec) +
                    " is read by feedback coherentized across a non-Clifford gate and its measurement "
                    "is readout-noiseless (no M(p) fold channel to force): a forced flip would not "
                    "reach the coherent control";
            }
        }
    }
    return out;
}

void delete_exact_branch_program(ExactBranchProgram* p) {
    if (!p) return;
    if (p->law_cache) delete_exact_law_cache(p->law_cache);   // references the program's bare state
    if (p->program) delete_sampler_program(p->program);
    delete p;
}

static int pauli_letter(char c) {   // 0 I, 1 X, 2 Y, 3 Z, -1 bad
    return c == 'I' ? 0 : c == 'X' ? 1 : c == 'Y' ? 2 : c == 'Z' ? 3 : -1;
}

// Address validation shared by both forced-fault entry points: the fired atoms + record flips.
static void forced_fired(const ExactBranchProgram& p, const std::vector<ForcedFault>& faults,
                         std::vector<FiredPauli>& fired, std::vector<int>& flips) {
    if (!p.program) throw std::invalid_argument("exact_branches: the circuit did not parse");
    std::vector<int> seen_records;
    std::vector<std::pair<int, int>> seen_noise;
    for (size_t fi = 0; fi < faults.size(); ++fi) {
        const ForcedFault& f = faults[fi];
        const std::string who = "fault " + std::to_string(fi) + ": ";
        if (f.record) {
            if (!p.record_refusal.empty()) throw std::invalid_argument(who + p.record_refusal);
            for (int j : seen_records)
                if (j == f.index) throw std::invalid_argument(who + "record " + std::to_string(j) + " flipped twice");
            seen_records.push_back(f.index);
            auto rr = p.refused_records.find(f.index);
            if (rr != p.refused_records.end()) throw std::invalid_argument(who + rr->second);
            auto fo = p.folds.find(f.index);
            if (fo != p.folds.end()) {   // fire the fold: record and coherent control flip together
                fired.push_back({fo->second.deferred_stream_index, fo->second.deferred_qubit, fo->second.pauli});
                continue;
            }
            flips.push_back(f.index);    // range-checked by the engine
            continue;
        }
        if (!p.noise_refusal.empty()) throw std::invalid_argument(who + p.noise_refusal);
        if (f.index < 0 || f.index >= (int)p.sites.size())
            throw std::invalid_argument(who + "noise site " + std::to_string(f.index) + " out of range (" +
                                        std::to_string(p.sites.size()) + " sites)");
        const NoiseSite& s = p.sites[(size_t)f.index];
        // A site's target GROUPS: qubits (1q channels), pairs (DEPOLARIZE2 / PAULI_CHANNEL_2), or the
        // whole chain support (CORRELATED: one group of arity = the chain's qubit count, its Pauli
        // written as one letter per chain qubit in `noise_sites()` order — any non-identity Pauli
        // on the support, not only the chain's own terms).
        const bool two = s.channel == NoiseChannel::DEPOLARIZE2 || s.channel == NoiseChannel::PAULI_CHANNEL_2;
        const bool chain = s.channel == NoiseChannel::CORRELATED;
        const size_t want = chain ? s.qubits.size() : (two ? 2 : 1);   // letters per target
        const int ntg = chain ? 1 : (two ? (int)s.qubits.size() / 2 : (int)s.qubits.size());
        if (f.target < 0 || f.target >= ntg)
            throw std::invalid_argument(who + "target " + std::to_string(f.target) + " out of range (site " +
                                        std::to_string(f.index) + " has " + std::to_string(ntg) + ")");
        for (const auto& st : seen_noise)
            if (st.first == f.index && st.second == f.target)
                throw std::invalid_argument(who + "site " + std::to_string(f.index) + " target " +
                                            std::to_string(f.target) + " carries two faults");
        seen_noise.push_back({f.index, f.target});
        bool ok = f.pauli.size() == want;
        std::vector<int> codes(want, 0);
        bool any = false;
        for (size_t c = 0; ok && c < want; ++c) {
            codes[c] = pauli_letter(f.pauli[c]);
            if (codes[c] < 0 || (want == 1 && codes[c] == 0)) ok = false;
            any |= codes[c] > 0;
        }
        if (ok && !any) ok = false;
        if (!ok)
            throw std::invalid_argument(who + "Pauli '" + f.pauli + "' is not a non-identity " +
                                        (chain ? std::to_string(want) + "-qubit Pauli (" + std::to_string(want) +
                                                     " letters over IXYZ, one per qubit of the chain's support)"
                                         : two ? std::string("two-qubit Pauli (two letters over IXYZ)")
                                               : std::string("single-qubit Pauli (X, Y or Z)")));
        for (size_t c = 0; c < want; ++c) {
            if (!codes[c]) continue;
            const size_t pos = chain ? c : (size_t)(two ? 2 * f.target + (int)c : f.target);
            const int dq = s.deferred_qubits[pos];
            const FiredPauli fp{s.deferred_index_of[pos], dq,
                                pauli_code_basis(s.deferred_letter[3 * pos + (size_t)codes[c] - 1])};
            if (s.deferred_stream_index >= 0 && !sampler_program_rejected(*p.program)) {
                const std::string why = sampler_program_fault_refusal(*p.program, fp);
                if (!why.empty())
                    throw std::invalid_argument(who + "noise site " + std::to_string(f.index) + " target " +
                                                std::to_string(f.target) + " Pauli '" + f.pauli + "' " + why);
            }
            fired.push_back(fp);
        }
    }
}

ExactLawResult exact_law(const ExactBranchProgram& p, const std::vector<ForcedFault>& faults,
                         const std::vector<int>& exp_cols, long max_rows, double max_work,
                         bool no_columns) {
    std::vector<FiredPauli> fired;
    std::vector<int> flips;
    forced_fired(p, faults, fired, flips);
    if (!p.law_cache) throw std::invalid_argument("exact_branches: the circuit was not compiled");
    return sample_program_exact_law(*p.program, *p.law_cache, std::move(fired), flips, p.decisions,
                                    exp_cols, max_rows, max_work, no_columns);
}

StimRunResult run_stim(const std::string& text, int shots, uint64_t seed,
                       const FramedSuperposition* reference) {
    // Per-shot view over the packed records (the contract is StimPackedResult/PackedRecords).
    StimRunResult out;
    StimPackedResult pk = run_stim_packed(text, shots, seed, reference);
    if (!pk.errors.empty()) { out.errors = std::move(pk.errors); return out; }
    const PackedRecords& r = pk.records;
    out.num_observables = r.num_observables;
    if (r.rejected) { out.reject_gate_index = r.reject_gate_index; return out; }
    out.chi = r.chi;
    out.shots.resize(r.shots);
    for (int s = 0; s < r.shots; ++s) {
        StimShot& sh = out.shots[s];
        sh.bits.resize(r.num_measurements);
        for (int k = 0; k < r.num_measurements; ++k) sh.bits[k] = (uint8_t)r.meas_bit(s, k);
        sh.detector_bits.resize(r.num_detectors);
        for (int k = 0; k < r.num_detectors; ++k) sh.detector_bits[k] = (uint8_t)r.det_bit(s, k);
        sh.observable_bits.resize(r.num_observables);
        for (int k = 0; k < r.num_observables; ++k) sh.observable_bits[k] = (uint8_t)r.obs_bit(s, k);
        sh.expectations.assign(r.expectations.begin() + (size_t)s * r.num_expectations,
                               r.expectations.begin() + (size_t)(s + 1) * r.num_expectations);
    }
    out.rejected = false;
    return out;
}

}  // namespace qeccore
