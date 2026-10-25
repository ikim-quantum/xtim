#pragma once
#include <map>
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

// ── Forced faults (3.1.11): compile once, query every branch exactly per fault set ───────────
// A NOISE SITE is one noise instruction of the PARSED circuit (REPEAT unrolled), numbered 0, 1, …
// in stream order; its targets are its qubits (1-qubit channels), qubit pairs (DEPOLARIZE2 /
// PAULI_CHANNEL_2) or, for an E/ELSE chain (CORRELATED), the ONE group of its whole support,
// numbered within the instruction. A forced fault names (site, target, Pauli);
// a record flip names an absolute measurement record.
// Addressing is by PROVENANCE, not by order: the compiler tags each user noise instruction with
// its site index (Instr::origin, metadata only) and the coherentizer tags the readout-flip fold
// it inserts for an `M(p)` read by non-Clifford-crossing feedback with that measurement's
// record; normalize copies the tags through. Site i -> the deferred instruction tagged UserSite i.
// A record flip of a FOLDED measurement fires the fold's Pauli (record AND coherent control flip
// together; a held fold's undo restores the state — exactly the fold's model of `M(p)`); any
// other record flip flips the recorded bit in the record tail, as before. A record read by
// coherentized feedback whose measurement has NO fold (readout-noiseless) is refused by name in
// `refused_records`. `noise_refusal` / `record_refusal` (whole-circuit) remain for a provenance
// mismatch the compiler cannot resolve (none is known).
struct NoiseSite {
    NoiseChannel channel;
    std::vector<int> qubits;           // user qubits
    std::vector<double> probs;         // CORRELATED: the chain's sequential probabilities
    int deferred_stream_index = -1;    // the (first) deferred instruction tagged with this site
    std::vector<int> deferred_qubits;  // parallel to qubits
    std::vector<unsigned char> corr;   // CORRELATED: the terms' factor codes (Instr::corr)
    // Parallel to qubits: the deferred instruction carrying that position (a frame commuted
    // through the site may split it) and the Pauli code there for user code c (index 3*j+c-1;
    // the frame relabels letters — Instr::frame_prov).
    std::vector<int> deferred_index_of;
    std::vector<unsigned char> deferred_letter;
};
struct ReadoutFoldSite {               // the deferred noise instruction folding one record's M(p)
    int deferred_stream_index = -1;
    int deferred_qubit = -1;
    PauliBasis pauli = PauliBasis::X;  // the fold channel's Pauli (X/Y/Z_ERROR -> X/Y/Z; a frame may relabel it)
};
struct ExactBranchProgram {
    SamplerProgram* program = nullptr;           // owned; release with delete_exact_branch_program
    ExactLawCache* law_cache = nullptr;          // owned; the factored law's per-program forms
    std::vector<ParseError> errors;
    std::vector<NoiseSite> sites;
    std::vector<std::vector<int>> decisions;     // DECISION record lists, declaration order
    std::string noise_refusal, record_refusal;   // non-empty => that fault kind is refused
    std::map<int, ReadoutFoldSite> folds;        // record -> its readout-flip fold
    std::map<int, std::string> refused_records;  // record -> why a forced flip of it is refused
};
ExactBranchProgram* compile_exact_branch_program(const std::string& text);
void delete_exact_branch_program(ExactBranchProgram* p);
struct ForcedFault {
    bool record = false;      // true: a record flip of record `index`
    int index = -1;           // noise site / record
    int target = -1;          // qubit or pair index within the site
    std::string pauli;        // "X"/"Y"/"Z" (1q), two letters over I/X/Y/Z (2q, not "II"), or one
                              // letter per chain qubit (CORRELATED, not all "I")
};
// Validates the addresses (std::invalid_argument naming the fault; a repeated location is an
// error), then the exact FACTORED law (sample_program_exact_law).
ExactLawResult exact_law(const ExactBranchProgram& p, const std::vector<ForcedFault>& faults,
                         const std::vector<int>& exp_cols, long max_rows,
                         double max_work = kExactLawWorkDefault, bool no_columns = false);

}  // namespace qeccore
