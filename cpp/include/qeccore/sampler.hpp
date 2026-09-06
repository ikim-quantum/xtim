#pragma once
#include <cstdint>
#include <cstddef>
#include <random>
#include <vector>
#include <functional>
#include "qeccore/circuit_ir.hpp"
#include "qeccore/normal_form.hpp"        // BareState
#include "qeccore/propagation_table.hpp"  // PropagationTable
#include "qeccore/canonical_stab_sum.hpp"

namespace qeccore {

// One fired single-qubit Pauli error at a noise location.
struct FiredPauli {
    int location_index;   // stream index of the Noise instruction
    int qubit;            // physical qubit it fired on
    PauliBasis pauli;     // X, Y, or Z
};

// Sample every Noise instruction's outcome for one shot. Draw order (determinism contract):
// Noise instructions in stream order; within an instruction, one uniform draw per qubit group
// (1q channels: per qubit; DEPOLARIZE2/PAULI_CHANNEL_2: one draw per qubit PAIR).
// Channels: X/Y/Z_ERROR(p): that Pauli w.p. p per qubit. DEPOLARIZE1(p): X/Y/Z w.p. p/3 each.
// PAULI_CHANNEL_1(px,py,pz). DEPOLARIZE2(p): the 15 non-identity 2q Paulis w.p. p/15 each.
// PAULI_CHANNEL_2(p1..p15, same Stim order IX,IY,IZ,XI,XX,...,ZZ): per-qubit factors recorded
// as separate FiredPauli entries (identity factors skipped).
std::vector<FiredPauli> sample_noise(const Circuit& circ, std::mt19937_64& rng);

// Compose the fired errors' end-of-circuit C_props in location order (earlier locations first:
// for l1 < l2 the noisy state is C2*C1*U|0> — push the earlier error out first), X-atom then
// Z-atom per fired Pauli (Y = both; the XZ = -iY phase is global and dropped downstream).
// Precondition: table.all_in_class and every fired (location, qubit) exists in the table.
DiagPauliClifford compose_fired(const PropagationTable& table,
                                const std::vector<FiredPauli>& fired);

// NOTE: the dense-oracle scaffolding apply_diag_pauli / sample_measurements / the free
// pauli_expectation(CanonicalStabSum, vector<PauliTerm>) had ZERO non-test callers and was
// relocated to cpp/tests/sampler_dense_oracle.hpp (included only by test_sampler.cpp).

struct ShotRecord {
    std::vector<uint8_t> bits;          // measurement record, original-circuit order
    std::vector<double> expectations;   // one per Observable instruction, declaration order
};

struct SampleResult {
    std::vector<ShotRecord> shots;      // size = requested shots iff !rejected
    bool rejected = true;               // table not in-class or sweep reject
    int  reject_gate_index = -1;
    int  chi = 0;
};

// Level-2 shot records, Stim's packed layout (the agreed record contract): every bit channel
// is row-major `b8` — shot s's row is bytes_per_shot(K) bytes, bit k at byte k>>3, bit k&7
// (little-endian within the byte) — byte-identical to Stim's `b8` / numpy bit_packed=True
// outputs, so Stim tooling and decoders consume the buffers directly. Channel semantics are
// Stim's too: measurements are RAW outcomes; detectors/observables are REFERENCE-RELATIVE
// (detection events / logical flips: bit = parity ⊕ noiseless-reference parity). The one
// channel Stim cannot carry, PAULI_EXPECTATION, rides alongside as a row-major float64 array.
struct PackedRecords {
    int shots = 0;
    int num_measurements = 0;
    int num_detectors = 0;
    int num_observables = 0;            // = max declared index + 1 (Stim convention)
    int num_expectations = 0;
    static int bytes_per_shot(int k) { return (k + 7) >> 3; }
    std::vector<uint8_t> measurements;  // shots × bytes_per_shot(num_measurements)
    std::vector<uint8_t> detectors;     // shots × bytes_per_shot(num_detectors)
    std::vector<uint8_t> observables;   // shots × bytes_per_shot(num_observables)
    std::vector<double>  expectations;  // shots × num_expectations
    bool rejected = true;
    int  reject_gate_index = -1;
    int  chi = 0;
    bool bit(const std::vector<uint8_t>& ch, int K, int s, int k) const {
        return (ch[(size_t)s * bytes_per_shot(K) + (k >> 3)] >> (k & 7)) & 1;
    }
    bool meas_bit(int s, int k) const { return bit(measurements, num_measurements, s, k); }
    bool det_bit(int s, int k) const { return bit(detectors, num_detectors, s, k); }
    bool obs_bit(int s, int k) const { return bit(observables, num_observables, s, k); }
};

// The packed-native sampler (the hot loop writes these buffers directly; run_shots is a
// per-shot view over it). detectors[d] / observables = lists of ABSOLUTE measurement-record
// indices whose parity defines detector d / observable o (Stim semantics); pass empty lists
// when only the measurement channel is needed.
// `reference` (design log §4.10): optional SUPPLIED bare/reference FramedSuperposition on the DEFERRED
// qubit space (qubit placement per DeferralMap). The engine TRUSTS it — verification that it
// is the circuit's noiseless final state is the supplier's job; the only check is qubit
// count (== deferred.n), rejecting with reject_gate_index == -1. The propagation-class check
// is the sole rejection criterion.
// nullptr = today's deduced path (build_bare_state_framed), bit-exact.
PackedRecords run_shots_packed(const Circuit& user_circuit, int shots, uint64_t seed,
                               const std::vector<std::vector<int>>& detectors = {},
                               const std::vector<std::vector<int>>& observables = {},
                               const FramedSuperposition* reference = nullptr);

// ── Compile-once / sample-many split of run_shots_packed ─────────────────────────────────────
// run_shots_packed's body factors cleanly into a seed/shots-INDEPENDENT COMPILE phase (deferral,
// propagation classification, bare state, feedback plan, detector/observable layout, the lean
// bare state, and the terminal-cascade plan(s)) and a seed/shots-DEPENDENT SAMPLE phase (the shot
// loop). `SamplerProgram` holds the compiled artifact; building it ONCE and reusing it across
// many sample_program() calls skips the recompile a stateless run_shots_packed pays every call.
//
// The split is PURELY STRUCTURAL + caching: sample_program() reproduces run_shots_packed's RNG
// path and per-shot scratch verbatim, so for a fixed (program, shots, seed) the produced
// PackedRecords is BYTE-IDENTICAL to the single-call run_shots_packed. SamplerProgram is an opaque
// type (its members reference sampler-internal plan structs); hold it via a pointer.
struct SamplerProgram;
void delete_sampler_program(SamplerProgram* p);   // out-of-line deleter for the opaque type

// Build the compiled program (no RNG consumed; seed/shots-independent). `detectors`/`observables`
// are the absolute-record parity lists (Stim semantics, same as run_shots_packed). On a pipeline
// reject the returned program has `rejected()==true` and `reject_gate_index()` set; sample_program
// then returns an empty rejected PackedRecords.
SamplerProgram* compile_sampler_program(const Circuit& user_circuit,
                                        const std::vector<std::vector<int>>& detectors = {},
                                        const std::vector<std::vector<int>>& observables = {},
                                        const FramedSuperposition* reference = nullptr);

// Sample the compiled program (the only seed/shots-dependent phase). Byte-identical to a
// run_shots_packed call with the same (circuit, detectors, observables, reference) at
// (shots, seed). The program's pattern-plan memo is mutable and persists across calls (it is
// seed-independent and consumes no RNG), so later samples may reuse earlier patterns.
PackedRecords sample_program(SamplerProgram& prog, int shots, uint64_t seed);

// ── Generic seams for alternative shot sources (used by research/stratified.hpp) ─────────────
// sample_program_with_source: the framed record-packing tail (record-flip / feedback / detector /
// observable packing, verbatim the framed branch of sample_program) over CALLER-SUPPLIED shots:
// shot_source(rng, out) must fill out[j] = ±1 per raw measurement read (the framed engine's
// next_shot / shot_from_events contract). Record channels only (no expectation rows); returns a
// rejected PackedRecords on a rejected program. Does NOT touch the program's pattern memo or
// reseed anything — callers own begin_run.
class FramedCircuitShotSampler;
PackedRecords sample_program_with_source(
    SamplerProgram& prog, int shots, uint64_t seed,
    const std::function<void(const std::function<double()>&, std::vector<int>&)>& shot_source);
// The compiled program's framed engine — THE engine (E2b: the legacy loop is gone; R >= 0 all
// route here). Null only for a rejected program.
FramedCircuitShotSampler* sampler_program_framed(SamplerProgram& prog);

// Per-shot view over run_shots_packed (kept for tests and small interactive runs). Pipeline:
// normalize/defer -> build_propagation_table (whole-circuit reject if any atom leaves the
// diagonal+Pauli class) -> bare state -> the framed circuit-level engine
// (FramedCircuitShotSampler: tier0 TreePlan replay / pattern memo / A-B error shots, with the
// R>0 expectation channels since E2b) -> the shared record-flip / feedback / packing tail.
// `reference`: optional supplied bare state, passed through to run_shots_packed (see there).
SampleResult run_shots(const Circuit& user_circuit, int shots, uint64_t seed,
                       const FramedSuperposition* reference = nullptr);

}  // namespace qeccore
