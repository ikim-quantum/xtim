#pragma once
// twirl_sampler — V2-T3: the END-TO-END twirl record sampler as an engine class.
//
// Extracted MOVE-only from the framed_bench `twirl_records` mode (byte-equality gated):
// per shot, ALL costs included: production-grade sparse noise draw (DiagErrorSampler
// skip-sampling) → exact compose_fired → normal form → canonicalize → content-keyed plan
// memo → channel-space emission: bits = baseline(plan) ⊕ Σ coin·coin_row ⊕ Σ prefix qubit
// rows ⊕ kernel folds. σ is NEVER materialised on the fast path (Tier B: every consumer-
// visible bit lives in channel space, 1 word for ≤64 channels). CH-class (non-diagonal)
// atoms route through the PPR plan family (V2-T1); --circuit-channels compiles DETECTORs
// into σ-mask channels with provenance masks + noiseless reference bits (V2-T2).
//
// The class runs every oracle the bench mode ran (accumulator / KEY / BITS / LINEARIZATION /
// LOCAL INDEX / PPR selfchecks on the first `selfcheck` diag shots) — a trip aborts run()
// loudly (returns false), never silent-wrong. The 5σ production gate stays with the caller
// (it needs run_shots_packed; this class is free of that dependency).

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/framed_superposition.hpp"

#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_ppr.hpp"

namespace qeccore {

// ── decoder-feedback perf (Fix 2): DEVIRTUALISED barrier record sink ──────────────────────────
// The per-shot record emission for sample_barrier used to cross a 12-arg `std::function` ShotSink
// (type-erased indirect call + argument marshaling every shot). This POD struct replaces it with a
// plain, inline-able `emit()` that the sampler calls directly. It holds:
//   * raw pointers to the caller's FIXED-STRIDE output buffers (pre-sized, never reallocated), and
//     POINTERS TO the growable CSR blob VECTORS (so a reallocation on push_back never dangles);
//   * BORROWED channel metadata (cdet/cobs/cref/gdet) — stable for the run;
//   * a REFERENCED (never copied) gauge-coin generator: the sampler draws the gauge coins directly,
//     preserving the EXACT draw order sample()/the old sink used (one draw per gauge detector, in
//     gdet order, AFTER the det/obs packing) so gauge-detector byte-identity is unchanged.
// The body is a byte-for-byte port of the old barrier sink lambda (xtim_py.cpp).
struct BarrierSink {
    // Fixed-stride output buffers (raw data pointers; buffers are pre-sized and never grow).
    uint8_t*  dets = nullptr;  int DBB = 0;
    uint8_t*  obs  = nullptr;  int OBB = 0;
    uint8_t*  dec  = nullptr;  int DECB = 0;
    uint64_t* sig_words = nullptr; int SGW = 0;
    // Growable CSR blobs — pointers to the VECTORS (reallocation-safe).
    std::vector<uint8_t>*  coin_data = nullptr;  std::vector<uint32_t>* coin_off = nullptr;
    std::vector<uint8_t>*  pk_data   = nullptr;  std::vector<uint32_t>* pk_off   = nullptr;
    std::vector<double>*   bu_data   = nullptr;  std::vector<uint32_t>* bu_off   = nullptr;
    // Borrowed channel metadata (owned by the sampler; stable across the run).
    const int*     cdet = nullptr;  int ndetc = 0;
    const int*     cobs = nullptr;  int ncobs = 0;
    const uint8_t* cref = nullptr;
    const int*     gdet = nullptr;  int ngdet = 0;
    int ndec = 0;
    int born_oi = -1;
    // Gauge-coin stream: REFERENCED (not copied) so the draw sequence matches sample() exactly.
    std::mt19937_64* gauge_rng = nullptr;
    long row = 0;
    // ── port-v3 T2 (L-E compact-record internal) ──────────────────────────────────────────
    // When `compact` is true the sink does NOT append the per-shot serialized plan bytes to
    // the pk CSR blob. Instead it stores the COMPACT plan identity per shot:
    //   * plan_id[row]      — dense id of the shot's interned CachedPlan (0 = no plan bytes,
    //                         i.e. clean / fallback shots whose legacy plan component is empty);
    //   * prefix_words row  — the residual prefix Pauli support, x words then z words (PNW
    //                         words each), copied straight from nf.prefix (the ONLY per-shot
    //                         part of the plan bytes — the (a-mask, cz) tail is a pure function
    //                         of the CachedPlan, interned once per DISTINCT plan into the
    //                         suffix blob from plan->key_amask / plan->key_cz).
    // This is a bijective re-encoding of the legacy plan bytes (prefix.x‖prefix.z‖a‖0xFF‖cz):
    // the plan cache interns on EXACTLY (a-mask packed bits = a&1, cz) with full content
    // equality (twirl_kernel_sampler.cpp get_or_build), so plan pointer ↔ (a,cz) ↔ suffix
    // bytes is one-to-one, and the legacy bytes are reconstructed per GROUP at read time
    // (PyBarrierBuffer) — byte-identical group keys, no per-shot serialization/copy.
    bool compact = false;
    uint32_t* plan_id = nullptr;          // per-shot dense plan id (pre-zeroed by caller)
    uint64_t* prefix_words = nullptr;     // per-shot 2*PNW words (pre-zeroed by caller)
    int PNW = 0;                          // prefix words per part (x or z) = ceil(n/64)
    int n_wires = 0;                      // deferred wire count (a-mask byte length)
    std::vector<uint8_t>*  suffix_data = nullptr;   // per-plan (a‖0xFF‖cz) bytes, CSR
    std::vector<uint32_t>* suffix_off  = nullptr;   // suffix_off[id-1]..suffix_off[id] = plan id
    std::unordered_map<const void*, uint32_t> plan_intern;  // CachedPlan* → dense id (≥1)

    inline void emit(const uint64_t* bits, int obs_bit, bool /*exact_fb*/,
                     const FramedSuperposition* /*amps*/, const uint64_t* sigma_gens,
                     const uint8_t* decision_bits, const uint8_t* coins, int n_coins,
                     const uint8_t* plan_key, int n_plan, const double* born_u, int n_born,
                     const CachedPlan* plan_ptr = nullptr, const uint64_t* pre_x = nullptr,
                     const uint64_t* pre_z = nullptr) {
        // ── record (σ, coins, plan) CSR appends — SAME order as the old sink ──
        if (coins && n_coins) coin_data->insert(coin_data->end(), coins, coins + n_coins);
        coin_off->push_back((uint32_t)coin_data->size());
        if (!compact) {
            if (plan_key && n_plan) pk_data->insert(pk_data->end(), plan_key, plan_key + n_plan);
            pk_off->push_back((uint32_t)pk_data->size());
        } else if (plan_ptr) {
            // Compact path: intern the plan (once per DISTINCT plan) + store the per-shot
            // prefix words. plan_key bytes (if any were serialized for the born memo) are
            // IGNORED here — the compact identity carries the same information.
            auto ins = plan_intern.try_emplace(plan_ptr, (uint32_t)plan_intern.size() + 1);
            if (ins.second) {
                // First encounter: serialize the (a-mask ‖ 0xFF ‖ cz) suffix from the plan's
                // OWN canonical content — byte-identical to the legacy plankey_ tail
                // (a byte per wire = key_amask bit = nf.a[q]&1; cz pairs u16-LE).
                for (int q = 0; q < n_wires; ++q)
                    suffix_data->push_back(
                        (uint8_t)((plan_ptr->key_amask[(size_t)(q >> 6)] >> (q & 63)) & 1ULL));
                suffix_data->push_back(0xFF);
                for (const auto& pr : plan_ptr->key_cz) {
                    suffix_data->push_back((uint8_t)(pr.first & 0xFF));
                    suffix_data->push_back((uint8_t)((pr.first >> 8) & 0xFF));
                    suffix_data->push_back((uint8_t)(pr.second & 0xFF));
                    suffix_data->push_back((uint8_t)((pr.second >> 8) & 0xFF));
                }
                suffix_off->push_back((uint32_t)suffix_data->size());
            }
            plan_id[row] = ins.first->second;
            uint64_t* pw = prefix_words + (size_t)row * (size_t)(2 * PNW);
            std::memcpy(pw, pre_x, (size_t)PNW * sizeof(uint64_t));
            std::memcpy(pw + PNW, pre_z, (size_t)PNW * sizeof(uint64_t));
        }
        // (compact && !plan_ptr: plan_id stays 0, prefix row stays zero — the empty-plan shot.)
        if (born_u && n_born) bu_data->insert(bu_data->end(), born_u, born_u + n_born);
        bu_off->push_back((uint32_t)bu_data->size());
        // ── decision bits (b8 LE) ──
        if (decision_bits && DECB) {
            uint8_t* decr = dec + (size_t)row * DECB;
            for (int j = 0; j < ndec; ++j)
                if (decision_bits[j]) decr[j >> 3] |= (uint8_t)(1u << (j & 7));
        }
        // ── σ words (fixed stride memcpy) ──
        {
            uint64_t* sdst = sig_words + (size_t)row * (size_t)SGW;
            if (sigma_gens) std::memcpy(sdst, sigma_gens, (size_t)SGW * sizeof(uint64_t));
        }
        // ── dets + obs from the SAME sink call ──
        uint8_t* dr = DBB ? dets + (size_t)row * DBB : nullptr;
        if (born_oi >= 0 && obs_bit && OBB)
            obs[(size_t)row * OBB + (born_oi >> 3)] |= (uint8_t)(1u << (born_oi & 7));
        for (int c = 0; c < ndetc; ++c) {
            const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)cref[(size_t)c];
            if (b && dr) { const int di = cdet[(size_t)c]; dr[di >> 3] |= (uint8_t)(1u << (di & 7)); }
        }
        uint8_t* orow = OBB ? obs + (size_t)row * OBB : nullptr;
        for (int k = 0; k < ncobs; ++k) {
            const int c = ndetc + k;
            const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)cref[(size_t)c];
            if (b && orow) { const int oi = cobs[(size_t)k]; orow[oi >> 3] |= (uint8_t)(1u << (oi & 7)); }
        }
        // ── gauge detector fair coins: one draw per gdet, in gdet order (matches sample()) ──
        for (int gi = 0; gi < ngdet; ++gi) {
            const int di = gdet[gi];
            if (((*gauge_rng)() & 1) && dr) dr[di >> 3] |= (uint8_t)(1u << (di & 7));
        }
        ++row;
    }
};

// ── decoder-feedback perf (Fix 1): DEVIRTUALISED word-packing dets/obs sink for sample() ───
// Parallel to BarrierSink but for the non-retention sample() path: no CSR blobs, no σ words.
// When det_identity=true (cdet[c]==c for all c, typical for circuit-channel compiles),
// the detector row is a word-level XOR copy of `bits ⊕ cref_det_packed` into the output
// buffer — no per-bit branch, just memcpy. Falls back to per-bit loop when non-identity.
// Observable handling: per-bit loop (obs channels start at a non-aligned bit offset in
// general; word-packing obs requires ndetc%64==0, not guaranteed).
// Gauge coins: one draw per gdet entry, in gdet order, AFTER det/obs packing — same as
// the old std::function sink. The caller passes &gauge_rng_ (referenced, not copied).
// Decision bits: byte-identical port of the std::function lambda's decision block.
// Output buffers are pre-zeroed by the caller; we OR bits in for set-1-only correctness.
struct DetsPackSink {
    // Output buffers (raw pointers; pre-sized by caller, never reallocated)
    uint8_t*  dets = nullptr;  int DBB = 0;
    uint8_t*  obs  = nullptr;  int OBB = 0;
    uint8_t*  dec  = nullptr;  int DECB = 0;
    // Channel metadata (borrowed; stable for the run)
    const int*     cdet = nullptr;  int ndetc = 0;
    const int*     cobs = nullptr;  int ncobs = 0;
    const uint8_t* cref = nullptr;
    const int*     gdet = nullptr;  int ngdet = 0;
    int ndec = 0;
    int born_oi = -1;
    // Gauge-coin stream (referenced; draw order matches the old lambda exactly)
    std::mt19937_64* gauge_rng = nullptr;
    long row = 0;
    // Word-pack fast path (det_identity means cdet[c]==c for all c)
    bool det_identity = false;
    std::vector<uint64_t> cref_det_packed; // ndetc bits as uint64 words (bit j of word w = cref[w*64+j])
    int det_words = 0;          // ceil(ndetc / 64)
    int det_partial_bits = 0;   // ndetc % 64; 0 means last word is fully used (all 64 bits valid)

    inline void emit(const uint64_t* bits, int obs_bit, bool /*exact_fb*/,
                     const FramedSuperposition* /*amps*/, const uint64_t* /*sigma_gens*/,
                     const uint8_t* decision_bits,
                     const uint8_t* /*coins*/, int /*n_coins*/,
                     const uint8_t* /*plan_key*/, int /*n_plan*/,
                     const double* /*born_u*/, int /*n_born*/,
                     const CachedPlan* /*plan_ptr*/ = nullptr,
                     const uint64_t* /*pre_x*/ = nullptr,
                     const uint64_t* /*pre_z*/ = nullptr) {
        // Decision bits (b8 LE)
        if (decision_bits && DECB) {
            uint8_t* decr = dec + (size_t)row * DECB;
            for (int j = 0; j < ndec; ++j)
                if (decision_bits[j]) decr[j >> 3] |= (uint8_t)(1u << (j & 7));
        }
        uint8_t* dr = DBB ? dets + (size_t)row * DBB : nullptr;
        // Born observable bit
        if (born_oi >= 0 && obs_bit && OBB)
            obs[(size_t)row * OBB + (born_oi >> 3)] |= (uint8_t)(1u << (born_oi & 7));
        // Detector packing
        if (dr) {
            if (det_identity) {
                // Word-level: XOR bits with packed cref, memcpy into output.
                // The last word may contain obs-channel bits beyond ndetc — mask them
                // out before writing so they don't bleed into the det output.
                for (int w = 0; w < det_words; ++w) {
                    uint64_t word = bits[w] ^ cref_det_packed[(size_t)w];
                    // Mask last word to only the valid detector bits
                    if (w == det_words - 1 && det_partial_bits)
                        word &= (1ULL << det_partial_bits) - 1ULL;
                    int byte_off = w * 8;
                    int nbytes = std::min(8, DBB - byte_off);
                    std::memcpy(dr + byte_off, &word, (size_t)nbytes);
                }
            } else {
                // Per-bit fallback
                for (int c = 0; c < ndetc; ++c) {
                    const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)cref[(size_t)c];
                    if (b) { const int di = cdet[(size_t)c]; dr[di >> 3] |= (uint8_t)(1u << (di & 7)); }
                }
            }
        }
        // Observable packing (per-bit; obs channels start at ndetc, may not be word-aligned)
        uint8_t* orow = OBB ? obs + (size_t)row * OBB : nullptr;
        if (orow) {
            for (int k = 0; k < ncobs; ++k) {
                const int c = ndetc + k;
                const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)cref[(size_t)c];
                if (b) { const int oi = cobs[(size_t)k]; orow[oi >> 3] |= (uint8_t)(1u << (oi & 7)); }
            }
        }
        // Gauge coins: one draw per gdet, in gdet order (matches old lambda)
        for (int gi = 0; gi < ngdet; ++gi) {
            const int di = gdet[gi];
            if (((*gauge_rng)() & 1) && dr) dr[di >> 3] |= (uint8_t)(1u << (di & 7));
        }
        ++row;
    }
};

// ── Plan-cache sharing across TwirlRecordSampler rebuilds (perf fix) ──────────────────────────
// TwirlPlanCache and PprPlanCache are pure functions of the compiled circuit content (keyed on
// the group identity token + normalised noise-blind content). PyTwirlSampler passes a
// SharedPlanCaches instance on every rebuild() so plan derivation work survives the seed-driven
// sampler reconstructions. Default (null pointers): the sampler creates fresh caches — identical
// to the pre-fix behaviour. A non-null pointer is shared and mutated by the sampler; the caller
// must ensure lifetime exceeds all samplers that share it.

// Shared fast-route index: the flat open-addressed table (ftab) + the stable key/plan vectors
// that back it (lkeys, lplans, lidx). These are impl-local in a single-rebuild world; hoisting
// them here lets warm rebuild() calls skip materialize+canon entirely on every shot outside the
// selfcheck window. Soundness: after commit 4e51920c, each shot's coins are a pure function of
// (seed, shot_index, plan) — routing cannot affect any coin stream — so serving a shot from the
// shared fast index on a warm call is byte-identical to the cold call that built the entry.
// KW must match Impl's KW at adoption time; the identity_token guard in PyTwirlSampler::rebuild()
// guarantees this structurally for the same compiled circuit.
struct SharedFastIndex {
    int KW = 0;  // key-word count; verified at adoption (mismatch → ignore, use fresh)
    std::vector<std::vector<uint64_t>> lkeys;     // stable per-plan key storage
    std::vector<const CachedPlan*>     lplans;    // parallel plan pointers
    std::unordered_map<uint64_t, std::vector<int>> lidx;  // hash → bucket of lkeys indices
    std::vector<uint64_t> ftab;    // flat open-addressed table (KW key words + ptr, 0=empty)
    size_t fslots = 0;             // power-of-2 slot count
    size_t fcount = 0;             // occupied slots
};

struct SharedPlanCaches {
    std::shared_ptr<TwirlPlanCache>  plans;  // diagonal twirl plan cache
    std::shared_ptr<PprPlanCache>    ppr;    // PPR (CH-class) plan cache
    std::shared_ptr<SharedFastIndex> fidx;   // fast-route flat index (warm rebuilds skip materialize+canon)
};

struct TwirlRecordOptions {
    // Per-terminal-record readout metadata (record order == terminal-read order). Stim's
    // M(p) flips the RECORD BIT with probability p (a record-space Pauli flip — it never
    // touches the state, unlike a pre-measurement X_ERROR); `invert` is the deterministic
    // record flip. Deferral DROPS both from the terminal Measures, so the CALLER must pass
    // them (read off the coherent pre-strip circuit's Measures, in order). rec_flip is scaled
    // by p_factor inside the sampler; consumed only under circuit_channels.
    std::vector<double> rec_flip;   // per record: readout flip probability (0 = none)
    std::vector<uint8_t> rec_invert;// per record: deterministic record flip
    double p_factor = 1.0;          // noise scale (p / 0.001)
    int want_nchan = 32;            // synthetic channel count (ignored w/ circuit_channels)
    bool circuit_channels = false;  // compile DETECTORs to σ-mask channels
    long selfcheck = 2000;          // oracle window: first N diag shots also run the σ path.
                                    // v2.7: runs ONCE per sampler lifetime — the first (non-
                                    // zero-shot) run() after construction; set_seed() never
                                    // re-arms it (rebuild/reconstruction does). Stream-neutral
                                    // overlay: sampled bytes identical with window on or off.
    std::string disk_path;          // TWPL plan-cache path ("" = off)
    // V3-T3 R2 (cache lifecycle): automatic plan-cache path. When disk_path is empty and this
    // directory is non-empty, the sampler derives disk_path itself as
    //   <disk_auto_dir>/<fnv64(deferred-signature)>-<group-token>.twpl
    // The deferred signature is STRUCTURAL (noise probability values excluded — the plan cache
    // is noise-blind, so one auto file serves whole p-sweeps); the group token is the certified
    // group's identity token, so a file is never shared across groups by name — and the TWPL
    // framing stores + verifies the token again on load (stale/corrupt/skew files load nothing
    // and rebuild silently). The caller owns directory creation / env resolution (xtim.twirl).
    std::string disk_auto_dir;
    std::vector<int> input_pauli_qubits;  // Task 4: deferred-space port qubit indices (X-then-Z sign relabel)
    // Task 2b.1 (module system): declared DECISION(k) record parities (decision_index, absolute
    // meas indices), one per DECISION line. Each parity is classified by G.reduce like a
    // DETECTOR/OBSERVABLE: IN_GROUP → deterministic σ-mask decision channel; LOGICAL/ANTI → the
    // Born-measurement channel (measure_pauli on the collapsed per-shot state, declaration order).
    // Empty (the default) ⇒ decision-free, the sampler is byte-identical to the pre-2b.1 engine.
    std::vector<std::pair<int, std::vector<int>>> decisions;
};

class TwirlRecordSampler {
  public:
    // bare: the certified in-class state (must outlive this object — run() reads it);
    // deferred: the deferred-noise circuit; reads: terminal read list (basis, qubit);
    // detectors/observables: record lists (absolute measurement indices).
    // deferred/reads/detectors/observables are consumed at construction only.
    TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                       const std::vector<std::pair<int, int>>& reads,
                       const std::vector<std::vector<int>>& detectors,
                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                       uint64_t seed, const TwirlRecordOptions& opt);
    // Plan-cache-sharing overload: functionally identical to the above but uses the supplied
    // shared caches (if non-null) instead of constructing fresh ones. Null pointers in `sc`
    // fall back to fresh caches, so partial sharing and the no-arg form are both safe.
    // Callers that share caches across rebuilds MUST check that G.identity_token() matches
    // the token the cache was first populated under (PyTwirlSampler does this guard).
    TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                       const std::vector<std::pair<int, int>>& reads,
                       const std::vector<std::vector<int>>& detectors,
                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                       uint64_t seed, const TwirlRecordOptions& opt,
                       const SharedPlanCaches& sc);
    ~TwirlRecordSampler();
    TwirlRecordSampler(const TwirlRecordSampler&) = delete;
    TwirlRecordSampler& operator=(const TwirlRecordSampler&) = delete;

    // 0 = OK. Nonzero = construction failed loudly (message already on stderr); the value is
    // the pre-extraction bench exit code (2 = no deterministic detector channels, 3 = table
    // not all-in-class / unsupported noise channel, 5 = blob verify fail). run() must not be
    // called after a setup error.
    int setup_error() const;

    // Sample `shots`; per-channel 1-counts accumulate internally (ref-XOR is applied at
    // reporting by the caller). Returns false on an oracle abort (message already on stderr).
    bool run(long shots);

    // ── speed-kill T1: in-place reseed ─────────────────────────────────────────────────────
    // Reset EXACTLY the seed-dependent state to what a fresh construction at `seed` would
    // hold — the noise-event stream (DiagErrorSampler::reseed: xoshiro state + initial
    // skips), the per-shot counter-RNG base seed, the dedicated obs/fallback/decision
    // streams, the rf shot counter, and the cumulative run() counters/stats (so every
    // reporting accessor behaves as freshly built) — reconstructing NOTHING. v2.7: the
    // selfcheck oracle window does NOT re-arm — it runs once per sampler lifetime (first
    // run() after construction); the window is a stream-neutral overlay, so seeded streams
    // are byte-identical either way. The circuit-derived material (propagation table, certified
    // planes, channel compilation, alt blobs) and the (shareable) plan caches are pure
    // functions of the compiled circuit; seed never enters them. Streams emitted after
    // set_seed(s) are byte-identical to a freshly constructed sampler at s (oracle:
    // xtim tests/test_set_seed_equivalence.py; zero tolerance).
    // Returns false WITHOUT mutating anything when the sampler cannot be reseeded in place:
    // synthetic-channel mode (!opt.circuit_channels), where the channel set itself is drawn
    // from an mrng(seed + 4242) at construction — seed-dependent physics. The caller must
    // then reconstruct (PyTwirlSampler falls back to rebuild()). Must not be called after a
    // setup error.
    bool set_seed(uint64_t seed);

    // ── Optional per-shot record sink (V2-T4, ADDITIVE) ─────────────────────────────────────
    // When set, the sink is invoked exactly once per shot, in shot order, inside run().
    // `bits` is ALWAYS a valid channel-bit row (channel_words() uint64 words; bit c is
    // EXACTLY the value accumulated into channel_one_counts() — the reporting ref-XOR is
    // NOT applied). V3-T3 R1: a default sampler never drops a shot — shots that trip a twirl
    // guard are computed per-shot by the EXACT engine (residual applied to a fresh bare copy,
    // certified generators measured sequentially, observable measured on the same collapsed
    // state) and delivered like any other shot; `exact_fallback` marks them (diagnostic only —
    // the row is a true sample of the same channel law). The exact computation draws from a
    // DEDICATED rng, so the collapse/coin streams of non-fallback shots are byte-identical to
    // a build without the fallback path. The sink itself draws NOTHING: with a null sink (the
    // default) the shot loop is byte-identical to the counting-only path.
    // obs_bit: the V3 Born-weighted observable bit for this shot (0 when no born-observable
    // mode is active — see born_obs_index()). Signature extended additively in V3.
    // S2.2 (decoder-feedback): the sink ALSO receives `amps` — a pointer to this shot's
    // COLLAPSED post-barrier FramedSuperposition (the residual-applied, kernel-collapsed state)
    // when state retention is active (set_retain_state(true)); otherwise `amps` is nullptr and
    // the null-sink hot path is byte-identical (no state is ever materialised). The pointer is
    // valid ONLY for the duration of the sink call (it aliases a reused per-shot working state);
    // a retaining consumer must COPY it. sigma_gens (n_gens sign bits, GW words) is the raw
    // certified-generator syndrome for this shot (parallel to bare.certified_stabilizers()),
    // nullptr when retention is off — the independent oracle projects a fresh bare onto it.
    // Task 2b.1: the sink ALSO receives `decision_bits` — a pointer to this shot's per-decision
    // output bits (byte per declared decision, length num_decisions(); index = the DECISION(k)
    // index). IN_GROUP decisions carry their deterministic σ-channel value; LOGICAL/ANTI decisions
    // carry the Born outcome (measure_pauli on the retained collapsed `amps`, declaration order).
    // nullptr when no DECISION is declared (decision-free circuits: byte-identical hot path).
    // Perf (decoder-feedback record-hash bucketing): the sink ALSO receives `coins` — a pointer to
    // this shot's TwirlOutcome coin record (r fair coins then κ chain outcomes, 0=+1/1=−1; length
    // n_coins), the reproducibility record that — together with sigma_gens — is a SUFFICIENT
    // classical statistic for the collapsed `amps` (twirl_kernel.hpp: "amps reconstructible from
    // (plan, σ, coins)"). Non-empty only on the diagonal RETENTION path (need_state); nullptr /
    // n_coins==0 on the fast counting path and on fallback shots. Additive; the null-sink hot path
    // is byte-identical.
    // `plan_key` (n_plan bytes) — the canonical residual plan (prefix x/z support + a-mask + cz,
    // global phase EXCLUDED). It is the third component of the sufficient statistic (plan, σ, coins)
    // for `amps`; nullptr on clean/identity and fallback shots. Additive; null-sink path unchanged.
    // `born_u` (n_born doubles) — decoder-feedback: this shot's per-DECISION Born COINS u∈[0,1),
    // in declaration order. materialize() RE-MEASURES the born operators with these on the
    // reconstructed state (genuine measurement, keeps the correlation gate mutation-sensitive).
    // Non-null only on the born retention path; nullptr / n_born==0 otherwise. Additive.
    // Fix 2: DEVIRTUALISED barrier sink. When non-null, run() emits each shot's record directly
    // through `BarrierSink::emit` (a plain inline call) INSTEAD of the `std::function` ShotSink —
    // no type-erased dispatch, no 12-arg marshaling, and the gauge coins are drawn from the sink's
    // referenced generator in the exact same order. The historical std::function ShotSink was fully removed 2026-07-27 (zero production consumers) when both
    // are set. The pointee must outlive run(); nullptr restores the std::function path. Only the
    // sample_barrier (retention) consumer uses it; sample() keeps the std::function sink.
    void set_barrier_sink(BarrierSink* sink);
    // Fix 1: DEVIRTUALISED dets-pack sink for sample(). When non-null, run() emits each
    // shot's record directly through DetsPackSink::emit instead of the std::function sink.
    // Takes precedence over the dets-pack sink (but NOT over set_barrier_sink). The pointee must
    // outlive run(); nullptr restores the std::function path.
    void set_dets_pack_sink(DetsPackSink* sink);
    // S2.2: when true, run() materialises each shot's collapsed post-barrier state (the slow
    // per-shot twirl_collapse(need_amps=true) path) and delivers it + its raw syndrome to the
    // sink. Off by default — the production sampling path is untouched (byte-identical). Only
    // the DIAGONAL residual class is supported under retention (a CH/PPR circuit throws in run()).
    void set_retain_state(bool on);
    // Task 4: set per-shot input Pauli frame (data layout: shots × 2*support, X-part then Z-part).
    // data=nullptr disables; safe to call before/after run().
    void set_input_frame(const uint8_t* data, int stride);

    // decoder-feedback perf (lazy state materialization): reconstruct one shot's COLLAPSED
    // post-barrier + post-decision state EXACTLY from its compact record (plan_key ‖ coins),
    // without ever having retained the full per-shot state. plan_key is the canonical residual
    // plan bytes (empty = clean/identity shot); coins is the reproducibility record [r fair ‖ κ
    // chain ‖ born outcomes]. Called once per bucket representative (NOT per shot).
    FramedSuperposition materialize_shot(const uint8_t* plan_key, int n_plan,
                                         const uint8_t* coins, int n_coins,
                                         const double* born_u, int n_born) const;

    // ── E3 (exact-residual arc, Task 3): PURE-DATA plan-structure exposure ────────────────────
    // Parse a plan key (the SAME canonical residual bytes materialize_shot consumes: prefix x/z
    // support words LE ‖ a-mask bytes ‖ 0xFF ‖ cz pairs u16-LE; empty = clean/identity shot;
    // global phase excluded) into its DiagNormalForm, and build that plan's ShotLaw via the
    // EXISTING build_shot_law — the identical call materialize_shot makes in step 2 of its
    // replay. No new computation is introduced: this is an accessor for the per-plan structure
    // (kernel_logicals, coin/kernel masks, det_signs, r/κ) the engine already derives per shot
    // but never exposed (T0 audit "no accessor for CachedPlan/ShotLaw internals"). Consumers:
    // the adaptq per-(check, plan) indefiniteness classifier (plan-conjugation + span test are
    // composed in Python from this data — cold per distinct plan, memoizable).
    struct PlanStructure {
        DiagNormalForm nf;   // parsed prefix (phase 0 — excluded from the key), a-mask, cz list
        ShotLaw law;         // build_shot_law(G, nf): det_signs, coin/kernel masks,
                             // kernel_base/foldable, kernel_logicals (exact phases), r, κ, fallback
    };
    PlanStructure plan_structure(const uint8_t* plan_key, int n_plan) const;

    // ── T5 step 0 (exact-residual arc): PURE-DATA born-decision-operator exposure ─────────────
    // The Born-measured decision operators (LOGICAL/ANTI-class DECISIONs), in declaration
    // order — the operators emit_decisions() measures per shot AFTER the twirl collapse, on the
    // residual-applied collapsed state.  op is the record-product Pauli over the n deferred
    // wires (operator = i^phase·X^x·Z^z, exact phase); the RAW outcome bit of decision j
    // (0 = +1 eigenvalue) is appended to the shot's coin record at coins[r + κ + j] (the
    // record-hash key component).  inv is the deterministic record invert: the EMITTED decision
    // bit is raw ⊕ inv (⊕ readout-flip coin parity ⊕ input-frame relabel, when present — both
    // affect the RECORD only, never the collapsed state).  Empty when the circuit declares no
    // Born-class decision.  Consumer: the adaptq per-(check, plan) classifier extends its span
    // basis with the plan-CONJUGATED ops (basis eigenvalue = the raw coin bit).
    struct BornDecOp {
        int dec_index;   // the DECISION index this operator decides
        Pauli op;        // record-product operator (n deferred wires, exact phase)
        uint8_t inv;     // deterministic record invert bit
    };
    std::vector<BornDecOp> born_dec_ops() const;

    // ── Reporting accessors (the bench line + 5σ gate are reproducible from these) ──────────
    long used() const;                 // shots entering the accumulation (== shots since V3-T3)
    long fallbacks() const;            // DIAGNOSTIC: shots computed by the per-shot exact
                                       // engine (INCLUDED in used/counts since V3-T3 R1)
    long exact_shots() const;          // alias of fallbacks() (the V3-T3 name)
    long clean_shots() const;          // no-fire shots
    long diag_shots() const;           // fired shots
    long ppr_shots() const;            // shots routed through the PPR plan family
    size_t ppr_plans() const;          // PPR plan cache size
    int num_channels() const;          // nchan
    int channel_words() const;         // CW = ceil(nchan/64)
    int deferred_wires() const;        // n = deferred wire count (port-v3 T2: the compact
                                       // record sink needs the a-mask byte length / prefix
                                       // word count = ceil(n/64) to configure its storage)
    int num_decisions() const;         // Task 2b.1: declared decision count (max DECISION index+1)
    size_t plan_hits() const;          // diagonal plan cache hits
    size_t plan_misses() const;        // diagonal plan cache misses
    size_t plans() const;              // diagonal plan cache size
    const std::vector<long long>& channel_one_counts() const;   // per-channel 1-bit counts
    double draw_seconds() const;       // accumulated noise-draw wall time
    double wall_seconds() const;       // accumulated end-to-end run() wall time
    // Circuit-channel compilation results (populated iff opt.circuit_channels). Channel
    // layout: channels [0, channel_detectors().size()) are DETECTOR channels; channels
    // [channel_detectors().size(), num_channels()) are deterministic-OBSERVABLE channels
    // (V2-T4; IN_GROUP observables compiled exactly like detector channels). channel_refs()
    // is parallel to the FULL channel list (detector channels first).
    const std::vector<int>& channel_detectors() const;      // det channel c → detector index
    const std::vector<int>& channel_observables() const;    // obs channel k → observable index
    const std::vector<uint8_t>& channel_refs() const;       // channel c → reporting ref bit
                                                            // (det/obs: the stage text's own
                                                            // deterministic-Pauli flips cancel
                                                            // — reference-relative, Stim
                                                            // semantics; a from_state class
                                                            // compile's carried-input
                                                            // deviation is kept)
    const std::vector<int>& gauge_detectors() const;        // LOGICAL-classified detectors
    const std::vector<int>& anti_detectors() const;         // ANTI-classified detectors
    const std::vector<int>& refused_observables() const;    // non-IN_GROUP observable indices
    // ── V3 Born-weighted observable channel (single logical observable, diagonal tables) ──
    // born_obs_index() >= 0 ⇒ that observable is emitted per shot as a σ-correlated biased
    // coin (exact joint with the deterministic detectors; executable spec
    // scripts/twirl_obs_reference.py). Accessors for the gate battery:
    int born_obs_index() const;        // observable index, -1 = mode off
    long obs_ones() const;             // Σ obs bits over used shots
    long ps_accepted() const;          // shots with EVERY deterministic detector at its ref
    long ps_obs_ones() const;          // Σ obs bits over accepted shots
    // The resolved plan-cache path ("" = disk layer off): opt.disk_path verbatim, or the
    // auto-derived <disk_auto_dir>/<signature>-<group-token>.twpl (V3-T3 R2).
    const std::string& disk_path() const;
    // The certified group's identity token (FNV-1a over group content). Used by PyTwirlSampler
    // to guard plan-cache reuse across rebuilds: if the token changes, the cache is cleared.
    uint64_t group_token() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qeccore
