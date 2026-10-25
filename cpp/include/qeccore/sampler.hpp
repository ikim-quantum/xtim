#pragma once
#include <cstdint>
#include <cstddef>
#include <random>
#include <string>
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
// then returns an empty rejected PackedRecords. `atoms`: the propagation table's atom set
// (propagation_table.hpp) — AnyPauli only for the forced-fault program, which may fire any Pauli.
SamplerProgram* compile_sampler_program(const Circuit& user_circuit,
                                        const std::vector<std::vector<int>>& detectors = {},
                                        const std::vector<std::vector<int>>& observables = {},
                                        const FramedSuperposition* reference = nullptr,
                                        AtomSet atoms = AtomSet::ChannelSupport);

// Sample the compiled program (the only seed/shots-dependent phase). Byte-identical to a
// run_shots_packed call with the same (circuit, detectors, observables, reference) at
// (shots, seed). The program's pattern-plan memo is mutable and persists across calls (it is
// seed-independent and consumes no RNG), so later samples may reuse earlier patterns.
PackedRecords sample_program(SamplerProgram& prog, int shots, uint64_t seed);

// sample_program with a per-shot hook: after shot `s` is packed into `rec` (rows [0, s] valid),
// hook(s, rec) returning false stops the run there — the result holds exactly shots [0, s], a
// PREFIX of the `shots`-shot run (byte-identical rows; shot s never depends on the shot count).
// A null hook is sample_program. Used by the work-sized noiseless-invariants battery
// (ref_compile.cpp) to grow one shot sequence instead of restarting it.
using ShotHook = std::function<bool(int shot, const PackedRecords& rec)>;
PackedRecords sample_program_streamed(SamplerProgram& prog, int shots, uint64_t seed,
                                      const ShotHook* hook);

// ── Generic seams for alternative shot sources (used by research/stratified.hpp) ─────────────
// sample_program_with_source: the framed record-packing tail (record-flip / feedback / detector /
// observable packing, verbatim the framed branch of sample_program) over CALLER-SUPPLIED shots:
// shot_source(rng, out) must fill out[j] = ±1 per raw measurement read (the framed engine's
// next_shot / shot_from_events contract). Record channels only (no expectation rows); returns a
// rejected PackedRecords on a rejected program. Does NOT touch the program's pattern memo or
// reseed anything — callers own begin_run.
class FramedCircuitShotSampler;

struct FramedSuperposition;
struct FeedbackPlan;
// THE noiseless reference record every engine reports detector/observable channels relative to —
// Stim's Circuit.reference_sample() convention (noise off, every collapse biased toward its +1
// eigenvalue = the user's record before `!`, in record order, feedback applied), computed on the
// bare state with framed_reference_records. Returns the RECORDED reference bit per terminal read
// (`!` invert included; M(p) coins excluded). `deferred` supplies eliminate_hadamards' per-read
// sign stamps (Measure.invert, read order), `rec_invert` the full per-record invert (user `!` ^
// stamp; empty = none), `fb` the feedback relabel plan for a KeepCoherent (stripped-feedback) bare
// state, or nullptr when feedback is coherent in the state (twirl) or absent.
std::vector<uint8_t> noiseless_reference_records(const FramedSuperposition& bare,
                                                 const Circuit& deferred,
                                                 const std::vector<std::pair<int, int>>& reads,
                                                 const std::vector<uint8_t>& rec_invert,
                                                 const FeedbackPlan* fb);
PackedRecords sample_program_with_source(
    SamplerProgram& prog, int shots, uint64_t seed,
    const std::function<void(const std::function<double()>&, std::vector<int>&)>& shot_source);
// The compiled program's framed engine — THE engine (E2b: the legacy loop is gone; R >= 0 all
// route here). Null only for a rejected program.
FramedCircuitShotSampler* sampler_program_framed(SamplerProgram& prog);

// ── Forced faults: the exact FACTORED law (3.1.11; factored form Unreleased) ─────────────────
// The compiled program's law CONDITIONED on one fixed fired set: `fired` (deferred stream index of
// a Noise instruction, the deferred qubit it names, the Pauli — the FiredPauli contract of
// sample_noise/compose_fired; any order: it is sorted into stream order here) and `record_flips`
// (absolute record indices whose RECORDED bit is flipped — the M(p) coin forced to fire; every other
// coin and every other channel is forced NOT to fire). No RNG is touched and nothing of the program
// is mutated: the sampled streams are unaffected by construction. The fired atoms' end-of-circuit
// errors (propagation table; diagonal AND CH-class general atoms) compose into one E; every
// terminal read M_k and declared PAULI_EXPECTATION P_r is conjugated to E^dag·M·E
// (process_shot_general_'s convention). Channels: detectors / observables reference-relative
// (Stim semantics), `decisions` = record lists whose RAW recorded parity is reported (DECISION
// lines), `exp_cols` = expectation columns to evaluate (empty = all). No read-by-read
// enumeration (the 2^r branch walk is retired; raw records are not returned). Every target —
// detector / observable (reference-relative), decision (raw recorded parity) — is an AFFINE GF(2)
// functional f = c + S m of the raw read vector m (`!` inverts, forced flips and the feedback
// relabel compose into it), and each expectation column's sign is (-1)^{e + u.m}. The reads
// A_k = E^dag M_k E commute; in the bare state's frame a read set's product has a nonzero
// expectation only when its non-free X-part vanishes (the stabilizer twirl). Hence:
//   * uniform directions: S applied to the span of the reads' non-free X-parts — every such
//     target direction is an INDEPENDENT FAIR COIN (never enumerated); u = their number;
//   * structured rows: the remaining target parities are deterministic (+-1 on the bare state)
//     or functions of d <= k magic-sector parities whose exact joint law is walked on its
//     SUPPORT only (PeShotAnalyzer::joint_law). Each row = the coset representative with every
//     coin 0 (zero on the directions' pivot targets), its total probability, and per column the
//     CONDITIONAL expectation given the row's targets (0 for a column anticommuting with a read
//     or outside the reads' X-span);
//   * law = rows XOR any subset of directions, each subset with probability (row prob) / 2^u;
//     direction j flips the sign of the columns in usign[j].
// Exact or refuse by name (`error`): more than `max_rows` structured rows, more than `max_work`
// walk work units (a deterministic count, never a clock), > 63 magic parities. `no_columns`:
// evaluate no expectation column (an empty `exp_cols` otherwise means all). `cache` (new_exact_law_cache) holds the
// program's linear record forms and the bare-state analyzer; queries on one cache serialize.
// Default cap on the magic-parity walk's work, in the walk's deterministic cost units (each Born
// probability / collapse / leaf read charged max((chi+1) n (k+1), (chi+1)^2)): about 1 s of work
// on the reference host (k |T> states read in X: k <= 8 answers, k >= 9 refuses by name).
constexpr double kExactLawWorkDefault = 4.0e8;
struct ExactLawCache;
ExactLawCache* new_exact_law_cache(const SamplerProgram& prog);
void delete_exact_law_cache(ExactLawCache* c);
struct ExactLawResult {
    int D = 0, O = 0, K = 0, R = 0;            // detectors, observables, decisions, columns
    int u = 0;                                 // independent fair-coin directions
    int d = 0;                                 // magic-sector parities walked
    std::vector<double> prob;                  // per structured row (sums to 1)
    std::vector<uint8_t> det, obs, dec;        // per row, row-major 0/1 bytes
    std::vector<double> exps;                  // per row (R wide): conditional expectations
    std::vector<uint8_t> udet, uobs, udec;     // per direction, row-major 0/1 bytes
    std::vector<uint8_t> usign;                // per direction (R wide): 1 = flips that column
    std::string error;                         // non-empty => no result
};
ExactLawResult sample_program_exact_law(const SamplerProgram& prog, ExactLawCache& cache,
                                        std::vector<FiredPauli> fired,
                                        const std::vector<int>& record_flips,
                                        const std::vector<std::vector<int>>& decisions,
                                        const std::vector<int>& exp_cols,
                                        long max_rows = 1L << 20,
                                        double max_work = kExactLawWorkDefault,
                                        bool no_columns = false);
// The deferred circuit a compiled program samples (the address space of FiredPauli above).
const Circuit& sampler_program_deferred(const SamplerProgram& prog);
bool sampler_program_rejected(const SamplerProgram& prog);
// "" if the program's table holds the atom(s) a fired Pauli needs, else why not (the Pauli has
// probability zero in its channel and its atom is absent). For naming refusals at the caller's
// address (forced faults); never throws for a fired location in the table.
std::string sampler_program_fault_refusal(const SamplerProgram& prog, const FiredPauli& f);
int sampler_program_reject_gate_index(const SamplerProgram& prog);

// Per-shot view over run_shots_packed (kept for tests and small interactive runs). Pipeline:
// normalize/defer -> build_propagation_table (whole-circuit reject if any atom leaves the
// diagonal+Pauli class) -> bare state -> the framed circuit-level engine
// (FramedCircuitShotSampler: tier0 TreePlan replay / pattern memo / A-B error shots, with the
// R>0 expectation channels since E2b) -> the shared record-flip / feedback / packing tail.
// `reference`: optional supplied bare state, passed through to run_shots_packed (see there).
SampleResult run_shots(const Circuit& user_circuit, int shots, uint64_t seed,
                       const FramedSuperposition* reference = nullptr);

}  // namespace qeccore
