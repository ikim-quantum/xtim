#include "qeccore/run_stim.hpp"
#include <cstddef>
#include "qeccore/sampler.hpp"

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
