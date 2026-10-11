#include "qeccore/run_stim.hpp"
#include <cstddef>
#include "qeccore/sampler.hpp"
#include "qeccore/normalize.hpp"
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
    auto* out = new ExactBranchProgram();
    ParsedStim p = parse_stim_circuit(text);
    if (!p.ok()) { out->errors = std::move(p.errors); return out; }
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    std::vector<std::vector<int>> obs((size_t)(max_obs + 1));
    for (auto& kv : p.observables) obs[kv.first] = kv.second;
    out->program = compile_sampler_program(p.circuit, p.detectors, obs, nullptr);
    for (auto& d : p.decisions) out->decisions.push_back(d.second);
    for (const Instr& ins : p.circuit.stream)
        if (ins.kind == Instr::Kind::Noise) out->sites.push_back({ins.channel, ins.qubits, ins.probs, -1, {}});
    if (sampler_program_rejected(*out->program)) return out;
    const Circuit& def = sampler_program_deferred(*out->program);
    std::vector<int> dn;
    for (int si = 0; si < (int)def.stream.size(); ++si)
        if (def.stream[(size_t)si].kind == Instr::Kind::Noise) dn.push_back(si);
    if (dn.size() != out->sites.size()) {
        out->noise_refusal = "the normalized circuit has " + std::to_string(dn.size()) +
            " noise instructions for the circuit's " + std::to_string(out->sites.size()) +
            " (coherentized feedback inserted readout-flip noise) — noise sites cannot be addressed";
    } else {
        for (size_t i = 0; i < dn.size(); ++i) {
            const Instr& d = def.stream[(size_t)dn[i]];
            NoiseSite& s = out->sites[i];
            if (d.channel != s.channel || d.qubits.size() != s.qubits.size() || d.probs != s.probs) {
                out->noise_refusal = "noise site " + std::to_string(i) +
                    " does not correspond to the normalized circuit's noise instruction";
                break;
            }
            s.deferred_stream_index = dn[i];
            s.deferred_qubits = d.qubits;
        }
    }
    int user_fb = 0;
    for (const Instr& ins : p.circuit.stream) if (ins.kind == Instr::Kind::ControlledPauli) ++user_fb;
    if (user_fb > 0) {
        NormalizePolicy npol;
        npol.feedback = NormalizePolicy::Feedback::KeepCoherent;
        npol.defer = true;
        NormalizeResult nr = normalize(p.circuit, npol);
        int kept = 0;
        for (const Instr& ins : nr.coherent.stream) if (ins.kind == Instr::Kind::ControlledPauli) ++kept;
        if (kept != user_fb)
            out->record_refusal = "feedback reading a record was coherentized (it crosses a non-Clifford "
                                  "gate): a forced record flip would not reach its coherent control";
    }
    return out;
}

void delete_exact_branch_program(ExactBranchProgram* p) {
    if (!p) return;
    if (p->program) delete_sampler_program(p->program);
    delete p;
}

static int pauli_letter(char c) {   // 0 I, 1 X, 2 Y, 3 Z, -1 bad
    return c == 'I' ? 0 : c == 'X' ? 1 : c == 'Y' ? 2 : c == 'Z' ? 3 : -1;
}

ExactBranchResult exact_branches(const ExactBranchProgram& p, const std::vector<ForcedFault>& faults,
                                 const std::vector<int>& exp_cols, long max_branches) {
    if (!p.program) throw std::invalid_argument("exact_branches: the circuit did not parse");
    std::vector<FiredPauli> fired;
    std::vector<int> flips;
    std::vector<std::pair<int, int>> seen_noise;
    for (size_t fi = 0; fi < faults.size(); ++fi) {
        const ForcedFault& f = faults[fi];
        const std::string who = "fault " + std::to_string(fi) + ": ";
        if (f.record) {
            if (!p.record_refusal.empty()) throw std::invalid_argument(who + p.record_refusal);
            for (int j : flips)
                if (j == f.index) throw std::invalid_argument(who + "record " + std::to_string(j) + " flipped twice");
            flips.push_back(f.index);    // range-checked by the engine
            continue;
        }
        if (!p.noise_refusal.empty()) throw std::invalid_argument(who + p.noise_refusal);
        if (f.index < 0 || f.index >= (int)p.sites.size())
            throw std::invalid_argument(who + "noise site " + std::to_string(f.index) + " out of range (" +
                                        std::to_string(p.sites.size()) + " sites)");
        const NoiseSite& s = p.sites[(size_t)f.index];
        const bool two = s.channel == NoiseChannel::DEPOLARIZE2 || s.channel == NoiseChannel::PAULI_CHANNEL_2;
        const int ntg = two ? (int)s.qubits.size() / 2 : (int)s.qubits.size();
        if (f.target < 0 || f.target >= ntg)
            throw std::invalid_argument(who + "target " + std::to_string(f.target) + " out of range (site " +
                                        std::to_string(f.index) + " has " + std::to_string(ntg) + ")");
        for (const auto& st : seen_noise)
            if (st.first == f.index && st.second == f.target)
                throw std::invalid_argument(who + "site " + std::to_string(f.index) + " target " +
                                            std::to_string(f.target) + " carries two faults");
        seen_noise.push_back({f.index, f.target});
        const size_t want = two ? 2 : 1;
        bool ok = f.pauli.size() == want;
        int codes[2] = {0, 0};
        for (size_t c = 0; ok && c < want; ++c) {
            codes[c] = pauli_letter(f.pauli[c]);
            if (codes[c] < 0 || (!two && codes[c] == 0)) ok = false;
        }
        if (ok && two && codes[0] == 0 && codes[1] == 0) ok = false;
        if (!ok)
            throw std::invalid_argument(who + "Pauli '" + f.pauli + "' is not a non-identity " +
                                        (two ? "two-qubit Pauli (two letters over IXYZ)" : "single-qubit Pauli (X, Y or Z)"));
        for (size_t c = 0; c < want; ++c) {
            if (!codes[c]) continue;
            const int dq = s.deferred_qubits[(size_t)(two ? 2 * f.target + (int)c : f.target)];
            fired.push_back({s.deferred_stream_index, dq, pauli_code_basis(codes[c])});
        }
    }
    return sample_program_exact_branches(*p.program, std::move(fired), flips, p.decisions, exp_cols,
                                         max_branches);
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
