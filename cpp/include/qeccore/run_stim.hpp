#pragma once
#include <string>
#include <vector>
#include "qeccore/sampler.hpp"
#include "qeccore/stim_parse.hpp"

namespace qeccore {

// Per-shot output: Stim's channels — raw measurement records, detector EVENTS and observable
// FLIPS (reference-relative: bit = parity ⊕ noiseless parity, exactly Stim's detector-sampler
// semantics; postselection is downstream, per Stim convention there is NO accept bit) plus the
// channel Stim cannot have: real-valued PAULI_EXPECTATION values, in declaration
// (= expectation_labels) order. [2026-06-11: detector/observable channels changed from raw
// parities to reference-relative events to match Stim's format byte-for-byte.]
struct StimShot {
    std::vector<uint8_t> bits, detector_bits, observable_bits;
    std::vector<double> expectations;
};

struct StimRunResult {
    std::vector<StimShot> shots;
    bool rejected = true;                    // parse errors OR pipeline reject (sweep/table)
    int  reject_gate_index = -1;
    std::vector<ParseError> errors;          // parse errors when present
    int  chi = 0;
    int  num_observables = 0;                // = max declared index + 1 (Stim convention)
};

// `reference`: optional SUPPLIED bare/reference state on the DEFERRED qubit space, passed
// through to run_shots_packed (same contract as sampler.hpp's: TRUSTED, qubit-count checked,
// reject_gate_index == -1 on mismatch; nullptr = deduced path, bit-exact).
StimRunResult run_stim(const std::string& text, int shots, uint64_t seed,
                       const FramedSuperposition* reference = nullptr);

// The packed level-2 entry point (the agreed record contract): parse + sample straight into
// Stim-b8 buffers (PackedRecords in sampler.hpp) with detector/observable channels resolved
// from the circuit's DETECTOR / OBSERVABLE_INCLUDE declarations. On parse failure `errors` is
// non-empty and records.rejected stays true. run_stim/StimShot is a per-shot view over this.
struct StimPackedResult {
    PackedRecords records;
    std::vector<ParseError> errors;
};
StimPackedResult run_stim_packed(const std::string& text, int shots, uint64_t seed,
                                 const FramedSuperposition* reference = nullptr);

// ── Compile-once from circuit TEXT ───────────────────────────────────────────────────────────
// run_stim_packed's per-call work = parse the text + marshal the observable channel + build the
// SamplerProgram (deferral/propagation/bare-state/cascade). compile_stim_program does the first two
// (text-dependent, seed/shots-independent) ONCE and returns the compiled SamplerProgram, so a
// stateful caller (the xtim CompiledProgram binding) can sample it repeatedly with NO recompile.
// On parse failure `errors` is non-empty and `program` is null. The program (when non-null) is
// owned by the caller and must be released with delete_sampler_program.
struct CompiledStimProgram {
    SamplerProgram* program = nullptr;       // owned; release with delete_sampler_program
    std::vector<ParseError> errors;          // parse errors when present (program == nullptr)
};
CompiledStimProgram compile_stim_program(const std::string& text,
                                         const FramedSuperposition* reference = nullptr);

}  // namespace qeccore
