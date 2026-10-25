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
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/propagation_table.hpp"

#include "qeccore/twirl_kernel.hpp"
#include "qeccore/record_table.hpp"

namespace qeccore {

class TermResidual;        // J-a (N2): qeccore/term_residual.hpp (FAM samplers only)
struct TermRouteShot;
struct FiredPauli;

// termroute-a (OD-2026-10-23-cache-bound): the default byte budget of each growing cache object — a sampler's plan
// payloads (+ its readers' tables, its term-route dispatch entries) and a RecordFidelity's held contents and memo.
// Eviction is byte-neutral (every evicted value is a pure function of its key); a constructor argument changes it.
inline constexpr size_t kDefaultCacheBudget = (size_t)512 << 20;

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
    // S0(c) / D6: caller byte budget for plan PAYLOADS (0 = no budget: unbounded; the Python surface
    // refuses an explicit 0 by name). Headers are never freed. termroute-a (OD-2026-10-23-cache-bound):
    // the default is kDefaultCacheBudget; the same budget bounds the term route's dispatch entries.
    size_t plan_payload_budget = kDefaultCacheBudget;
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
    // ── The record-family (FAM) sampler: J-a built it switched OFF; J-b1 F made it THE sampler; J-b2 J3 made it the
    // only one (the legacy instantiation and the `family` switch are deleted). The defaults are the production
    // configuration (J-b2 J6: every record class is served; the class bits are deleted):
    //   oracle_all_shots — Q7: the semantic oracle on EVERY shot (stream-neutral).
    bool oracle_all_shots = false;
    // J-b1 P4 (test API only, an oracle construction option; KEPT by J-b2, C4 — not a class switch): a FAM
    // recording run term-routes EVERY shot (the J-a route) instead of serving diagonal shots on the (D) fast /
    // accumulator / structural paths; the term view is armed at construction. The G-REC-EQ comparison object.
    bool fam_record_route_term = false;
};

// ── J-a (N1): THE record sink (J-b2 J6b: the legacy BarrierSink / DetsPackSink are deleted). It emits the
// detector / observable channels with their reporting refs, the Born observable (made reference-relative here:
// born_ref), the decisions (raw) — plus the FAMILY EXTRAS: a gauge /
// ANTI detector or an extra observable served from the plan's record table, reported reference-
// relative (gauge-plan §1.4):
//     emitted = y ⊕ rf ⊕ inv ⊕ ref,
// y the extra's raw record value (record rows ⊕ t(P)), rf the parity of this shot's readout-flip coins
// over the extra's noisy records (a feedback-DRIVING noisy record carries no rf slot: its flip is a
// residual fault of the deferred normal form, MINOR-2), inv ⊕ ref = xoff (the record invert and the
// parity of noiseless_reference_records over the extra's records, the born_ref_ convention). The
// family-extras metadata (xbase .. RFW), the channel metadata (cdet .. ngdet) and the detector word path
// (det_identity .. det_partial_bits) are filled by TwirlRecordSampler::set_family_sink from the sampler's own
// vectors (one source of truth); the caller fills the output buffers, ndec and the Born fields.
// Detector packing (J-a final review M1): the former DetsPackSink's word path — when det_identity (cdet[c] == c),
// the detector row is bits ⊕ cref_det_packed copied word by word (the last word masked to ndetc), else the
// per-bit loop. ORDER IS LOAD-BEARING: the copy overwrites whole bytes (bits ≥ ndetc of the last word are 0), and
// under det_identity every family-extra detector has index ≥ ndetc, so the extras are ORed in AFTER it.
struct FamilySink {
    uint8_t*  dets = nullptr;  int DBB = 0;
    uint8_t*  obs  = nullptr;  int OBB = 0;
    uint8_t*  dec  = nullptr;  int DECB = 0;
    const int*     cdet = nullptr;  int ndetc = 0;
    const int*     cobs = nullptr;  int ncobs = 0;
    const uint8_t* cref = nullptr;
    int ndec = 0;
    int born_oi = -1;
    int born_ref = 0;
    // the family extras (set by set_family_sink): extras column xbase + i, i < nfx
    int xbase = 0, nfx = 0;
    const uint8_t*  xkind = nullptr;   // 0 gauge detector, 1 ANTI detector, 2 extra observable
    const int*      xout  = nullptr;   // the detector / observable index the column reports
    const uint8_t*  xoff  = nullptr;   // inv ⊕ ref
    const uint64_t* xrf   = nullptr;   // nfx × RFW readout-flip slot masks
    int RFW = 0;
    long row = 0;
    // the detector word path (DetsPackSink's; filled by set_family_sink)
    bool det_identity = false;
    std::vector<uint64_t> cref_det_packed;     // ndetc bits as words (bit j of word w = cref[w*64+j])
    int det_words = 0;                         // ceil(ndetc / 64)
    int det_partial_bits = 0;                  // ndetc % 64; 0: the last word is fully used

    // J-b1 P3 (R1): ALWAYS inlined into run_w<*> (the FAM instances' only caller): out of line it cost
    // 134 / 170 / 326 Ir per shot on d3b / d5s / bt81b against production's inlined DetsPackSink (P1 profile).
    __attribute__((always_inline)) inline void emit(const uint64_t* bits, int obs_bit, const uint8_t* decision_bits,
                     const uint64_t* xbits, const uint64_t* rf_coins) {
        if (decision_bits && DECB) {
            uint8_t* decr = dec + (size_t)row * DECB;
            for (int j = 0; j < ndec; ++j)
                if (decision_bits[j]) decr[j >> 3] |= (uint8_t)(1u << (j & 7));
        }
        uint8_t* dr = DBB ? dets + (size_t)row * DBB : nullptr;
        uint8_t* orow = OBB ? obs + (size_t)row * OBB : nullptr;
        if (born_oi >= 0 && orow && ((obs_bit ^ born_ref) & 1))
            orow[born_oi >> 3] |= (uint8_t)(1u << (born_oi & 7));
#ifdef XTIM_MUTATE_FS_ORDER
        emit_extras_(dr, orow, xbits, rf_coins);   // mutation build only (MUT-FS-ORDER): extras before the copy
#endif
        if (dr) {
            if (det_identity) {
                for (int w = 0; w < det_words; ++w) {
                    uint64_t word = bits[w] ^ cref_det_packed[(size_t)w];
#ifndef XTIM_MUTATE_FS_MASK
                    if (w == det_words - 1 && det_partial_bits)
                        word &= (1ULL << det_partial_bits) - 1ULL;
#endif                                         // mutation build only (MUT-FS-MASK): the last word unmasked
                    const int byte_off = w * 8;
                    std::memcpy(dr + byte_off, &word, (size_t)std::min(8, DBB - byte_off));
                }
            } else {
                for (int c = 0; c < ndetc; ++c) {
                    const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)cref[(size_t)c];
                    if (b) { const int di = cdet[(size_t)c]; dr[di >> 3] |= (uint8_t)(1u << (di & 7)); }
                }
            }
        }
        if (orow)
            for (int k = 0; k < ncobs; ++k) {
                const int c = ndetc + k;
                const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)cref[(size_t)c];
                if (b) { const int oi = cobs[(size_t)k]; orow[oi >> 3] |= (uint8_t)(1u << (oi & 7)); }
            }
#ifndef XTIM_MUTATE_FS_ORDER
        emit_extras_(dr, orow, xbits, rf_coins);
#endif
        ++row;
    }
    // the family extras: ORed in after the detector copy (see the ORDER note above)
    __attribute__((always_inline)) inline void emit_extras_(uint8_t* dr, uint8_t* orow, const uint64_t* xbits,
                                                            const uint64_t* rf_coins) const {
        for (int i = 0; i < nfx; ++i) {
            const int e = xbase + i;
            int b = (int)((xbits[e >> 6] >> (e & 63)) & 1) ^ (int)xoff[i];
            const uint64_t* m = xrf + (size_t)i * (size_t)RFW;
            for (int w = 0; w < RFW; ++w) b ^= __builtin_popcountll(rf_coins[w] & m[w]) & 1;
            if (!b) continue;
            const int oi = xout[i];
            uint8_t* r = (xkind[i] == 2) ? orow : dr;
            if (r) r[oi >> 3] |= (uint8_t)(1u << (oi & 7));
        }
    }
};

// TEST seam (ctest only; nullptr in production): called on the packed alt blob just before the ctor's one-time
// blob-vs-masks verification, so a test can corrupt it and see the named setup_error-5 refusal.
extern void (*g_twirl_alt_blob_test_hook)(std::vector<uint64_t>&);

class RecordFidelity;            // J-b1 P7 (record_fidelity.hpp)
struct TwirlSamplerTestAccess;   // J-b1 P3 (m6): defined in cpp/testapi/twirl_test_access.hpp

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
    // The bindings' constructor. Identical to the constructor above, except for one thing: the two
    // plan caches are created BEFORE the sampler object, which keeps the binding's historical heap
    // allocation order (caches, then the sampler, then its Impl).
    // The plan caches are owned by this sampler and are never shared (S2b R-m4: cache payloads
    // depend on the bare state — its frame, eps, free order and amplitudes — and on the Born
    // observable, which no key covers, and no caller ever shared them).
    // class_table (compile stage, F-C): the class check's table of `deferred` when the caller has
    // one (NormalizeResult::class_table: every atom in class, hence equal to the table this setup
    // builds); empty = build it here. The caller passes it only for the `deferred` it came from.
    static std::unique_ptr<TwirlRecordSampler> make(
        const FramedSuperposition& bare, const Circuit& deferred,
        const std::vector<std::pair<int, int>>& reads,
        const std::vector<std::vector<int>>& detectors,
        const std::vector<std::pair<int, std::vector<int>>>& observables,
        uint64_t seed, const TwirlRecordOptions& opt,
        std::optional<PropagationTable> class_table = std::nullopt);
    ~TwirlRecordSampler();
    TwirlRecordSampler(const TwirlRecordSampler&) = delete;
    TwirlRecordSampler& operator=(const TwirlRecordSampler&) = delete;

    // 0 = OK. Nonzero = construction failed; the bindings raise the refusal (its text is family_refusal() for
    // 5 and 7, worded by the bindings for 3). The value is the pre-extraction bench exit code (2 = no deterministic detector channels, 3 = table
    // not all-in-class / unsupported noise channel, 5 = blob verify fail). run() must not be
    // called after a setup error.
    int setup_error() const;

    // Sample `shots`; per-channel 1-counts accumulate internally (ref-XOR is applied at
    // reporting by the caller). Returns false on an oracle abort (the message is an ORACLE_FAILURE
    // event in take_diagnostics() for the S0 checks, on stderr for the older oracles — S7).
    bool run(long shots);

    // ── early-reject (OD-2026-10-25): the opt-in postselection mask (twirl_sampler_er.cpp). det_flags[d] != 0 flags
    // detector d; a run then emits ONLY the shots whose flagged detectors are all 0 (the sink's rows / the record
    // sink's rows are the accepted shots, in shot order). Shots a flagged deterministic detector rejects for certain
    // are dropped before any plan work; every emitted shot passes the exact filter on its detector row. Accepted shots
    // are byte-identical to an unmasked run filtered afterwards. Empty / all-zero flags: no mask (byte-identical to
    // a sampler that never had one). The counters are cumulative since the last set_postselect.
    void set_postselect(const std::vector<uint8_t>& det_flags);
    bool postselect_active() const;
    long postselect_early() const;     // shots rejected before any plan work
    long postselect_late() const;      // emitted shots the exact filter took back

    // ── speed-kill T1: in-place reseed ─────────────────────────────────────────────────────
    // Reset EXACTLY the seed-dependent state to what a fresh construction at `seed` would
    // hold — the keyed noise streams (DiagErrorSampler::reseed: per-channel stream positions +
    // initial skips), the keys of every per-shot keyed stream (krng: COIN, BORN, OBS, DEC, RF,
    // GAUGE, EXACT), the cumulative shot index, and the cumulative run() counters/stats (so every
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

    // ── J-a (N1, switched OFF): the FAM sampler surface (test API only; defined in twirl_sampler_fam.cpp).
    // set_family_sink: the record sink of a FAM sampler (J-b2 J6b: the only sink at
    // the emit seam); fills the sink's family-extras metadata. Throws on a non-FAM sampler.
    void set_family_sink(FamilySink* sink);
    // J-b1 P3 (m6): the FAM sampler's surface below (through family_model_key) is private, reached only
    // through TwirlSamplerTestAccess (cpp/testapi/twirl_test_access.hpp: the test API) and, from J-b1 F, through
    // TwirlSamplerBindingAccess (cpp/bindings/twirl_binding_access.hpp: the members the PUBLIC bindings read —
    // records, materialize, signs, fidelity, branches). Declarations only: no layout change.
    friend struct TwirlSamplerTestAccess;
    friend struct TwirlSamplerBindingAccess;

  private:
    struct FamilyExtra {
        int kind;                  // 0 gauge detector, 1 ANTI detector, 2 extra observable
        int index;                 // detector / observable index
        int column;                // storage column of its record (channels, then extras)
        uint8_t off;               // inv ⊕ ref (the emitted bit is y ⊕ rf ⊕ off)
    };
    std::vector<FamilyExtra> family_extras() const;
    // J-b1 P10 (A1 §2): on a port circuit (INPUT_QUBITS / ideal-state compile) the Born observable's reporting
    // reference = the parity of the noiseless reference records over its records (the binding's record list).
    // No-op without a port or a Born observable.
    void family_port_born_ref(const std::vector<int>& born_records);
    // J-b1 P11a (test API): the sampler's diagonal plan cache (shared ownership: the destroy-then-evict test keeps
    // it alive past the sampler; the arena makes that safe).
    std::shared_ptr<TwirlPlanCache> family_plan_cache() const;
    // The named construction refusal ("" = none; setup_error() is then 7, or 5 for the blob verify).
    const std::string& family_refusal() const;
    // J-b1 P13: the structured NONCOMMUTING_MAGIC refusal of a FAM sampler (cls 1; 0 = not this class): the two
    // anticommuting pushed axes (deferred wires) and the fault-blocked merge (deferred gate indices, noise site).
    struct FamilyRefusalInfo {
        int cls = 0;
        std::vector<Pauli> axes;
        int gate_i = -1, gate_j = -1, site = -1;
    };
    FamilyRefusalInfo family_refusal_info() const;
    // Semantic-oracle checks per record class (FAM samplers): channels, Born observable, Born decisions,
    // gauge detectors, ANTI detectors, extra observables (records projected, cumulative).
    std::vector<long> semantic_oracle_class_checks() const;
    // ── J-a (N2, switched OFF): the term route's test-API surface (FAM samplers; twirl_sampler_fam.cpp).
    // family_terms: the sampler's TermResidual (built at construction; the per-alternative (P_f, S_f)
    // table attached on first use). Throws on a non-FAM sampler.
    TermResidual& family_terms();
    const std::vector<std::vector<FiredPauli>>& family_alt_faults() const;   // flat alt id → fired list
    std::vector<uint8_t> family_alt_is_ppr() const;
    // The fired alternative ids of `shots` shots exactly as run() draws them after set_seed(seed) (the
    // noise stream is left advanced: reseed before sampling).
    std::vector<std::vector<int32_t>> family_draw_events(long shots, uint64_t seed);
    // The term route of an alternative list (time order): compose → reduce_key_ → kind_of_ → (kind 0)
    // the diagonal route's normal form → plan id.
    void family_term_shot(const std::vector<int32_t>& alts, TermRouteShot& out);
    // The diagonal route of an all-diagonal alternative list (compose_fired → residual_normal_form →
    // canonicalize_mod_stabilizers_inplace, the self-check window's chain) and its plan id; false when an
    // alternative is a PPR atom.
    bool family_diag_shot(const std::vector<int32_t>& alts, DiagNormalForm& nf, long& plan_id);
    // t(P) per storage column: the channels' prefix rows (pxrow / pzrow) and ⟨P, M_e⟩ for the extras.
    std::vector<uint8_t> family_relabel(const Pauli& P) const;
    // ── J-a (N3, switched OFF): the (G) builder and the dual-run surface (FAM samplers; twirl_sampler_fam.cpp).
    // family_general_table: build_record_table_general over this sampler's record columns for C0 = rot(S),
    // S a term set of family_terms() (term_words() words); the sampler's G and bare state.
    void family_general_table(const std::vector<uint64_t>& S, RecTable& out, std::vector<int>* intro_b = nullptr);
    // (G)'s table of a DIAGONAL plan key (plan_structure's format; prefix ignored): C0 = S^{a&1}·CZ written
    // as terms (diag_plan_axes) — the same C0 (D) builds plan_structure's table from.
    void family_general_table_diag(const uint8_t* plan_key, int n_plan, RecTable& out,
                                   std::vector<int>* intro_b = nullptr) const;
    // The joint Born law p(z) (index Σ_j z_j 2^j, 2^kappa entries) of a table's directions on the bare free
    // register, by the unchanged BornTrie node definition (p0_from_scratch), chained.
    std::vector<double> family_born_law(const RecTable& T) const;
    // ── J-a (N4a, switched OFF): the kind-2 route's counters (FAM samplers with the ppr_records class;
    // cumulative over run() calls). shots_kind0 / shots_kind2: PPR-class shots served by a kind-0 plan
    // (the diagonal cache) / a kind-2 (G) plan; keys: distinct reduced term sets dispatched; plans_kind2:
    // kind-2 plan headers; builds_kind2 / payload_builds_kind2: (G) table builds (first / after an eviction);
    // oracle_term_shots: term-routed shots the term-view semantic oracle verified; plan_shots: per plan id
    // (the N2 intern table, one for kind 0 and kind 2) the term-routed shots it served; plan_kind: its kind.
    // N4-D1 (fixplan-k0pairs r2): a kind-0 composite with a CZ pair outside the diagonal cache's pair universe
    // is served by the kind-2 (G) path on a sampler-owned header: plans_ / builds_ / payload_builds_ /
    // shots_kind0_general count those (shots_kind0_general is a sub-count of shots_kind0).
    struct FamilyRouteStats {
        long shots_kind0 = 0, shots_kind2 = 0, keys = 0, plans_kind2 = 0, builds_kind2 = 0,
             payload_builds_kind2 = 0, oracle_term_shots = 0, shots_framed = 0;
        long shots_record_d = 0;     // J-b1 P4: recording-run shots whose layout-3 record the (D) paths wrote
        // J-b1 P6: (D) record tables built by record-space B on this FAM sampler (J-b2 J7b: the only builder).
        long tables_recb = 0;
        // J-b1 (fixplan-coldstart): law builds. laws_payload = the cache's core-mode payload law builds (one per
        // diagonal payload build: a miss or a rebuild after eviction; kind-2 / route-1 plans build none, reader-memo
        // and plan_structure builds are excluded); tables_recb_parked = record-space-B tables built from that parked
        // law; recb_core_rebuilds = tables whose plan's law was NOT parked (a second law: expected 0).
        long laws_payload = 0, tables_recb_parked = 0, recb_core_rebuilds = 0;
        long plans_kind0_general = 0, builds_kind0_general = 0, payload_builds_kind0_general = 0,
             shots_kind0_general = 0;
        std::vector<long> plan_shots;
        std::vector<int> plan_kind;
        long reader_table_rebuilds = 0;   // J-b1 P12: kind-0 reader tables built because the plan was not resident
        long key_bytes = 0, key_evictions = 0;   // termroute-a: the route's dispatch entries, charged / dropped
    };
    FamilyRouteStats family_route_stats() const;
    // ── J-a (N5, switched OFF): retention for the whole family (test API; twirl_sampler_fam.cpp). With a
    // record sink set, a FAM run routes EVERY shot through the term route and writes its layout-3 record:
    // the content handle (kind + canonical content key: kind 0 = the S^a·CZ view a ‖ 0xFF ‖ cz u16 pairs,
    // kind 2 = the reduced term bitset), P_rec (x then z, PNW words each; the frame folded on the
    // input_pauli route), y (the raw record values, record_value_words() words, storage column order),
    // the emission's outcomes (u then z) and the frame_folded flag. The state of a record is
    // P_rec·C0|bare⟩ projected onto every record at y (materialize: every pivot and introducer forced).
    struct FamilyRecordSink {
        long cap = 0, row = 0;          // rows available / written (the run appends at `row`)
        int PNW = 0, YW = 0, UZW = 0;   // P_rec words per half, y words, outcome bytes per row
        int32_t* handle = nullptr;      // family_record_contents() index
        uint8_t* route = nullptr;       // 0: (D) kind 0; 1: (G) kind 0 outside the pair universe; 2: (G) kind 2
        uint64_t* prec = nullptr;       // 2·PNW words per row
        uint64_t* y = nullptr;          // YW words per row
        uint8_t* uz = nullptr;          // UZW bytes per row (the first nuz[row] are the outcomes)
        int32_t* nuz = nullptr;
        uint8_t* ff = nullptr;          // frame_folded
        int32_t* branch = nullptr;      // J-b1 P9b: the row's carried branch (engine id; 0 = the ideal state)
    };
    void set_family_record_sink(FamilyRecordSink* rs);
    struct FamilyRecordContent {
        int kind = 0;
        std::string content;            // the canonical content key
        uint64_t digest_hi = 0, digest_lo = 0;   // the 128-bit digest of (kind, content)
    };
    const std::vector<FamilyRecordContent>& family_record_contents() const;
    // J-b1 P11b / P12: a content's plan as the record API exports it: kind 0 its S^a·CZ view (a, cz), kind 2 its
    // reduced term set S (the axes are the term view's: family_terms), the table's ρ / κ_rec, and the Born
    // directions Λ_j as physical Paulis (U·i^phase(−1)^par X^a Z^b·U†, before the certified-group reduction).
    // `axes` stays empty: the export builds one shared axis table (no per-plan heap Paulis).
    struct FamilyPlanView {
        int kind = 0;
        std::string content;
        uint64_t digest_hi = 0, digest_lo = 0;
        std::vector<Pauli> axes;
        std::vector<uint64_t> S;
        int rho = 0, kappa = 0;
        std::vector<Pauli> born;
        std::vector<uint8_t> a;                       // kind 0 only (n entries, a & 1)
        std::vector<std::pair<int, int>> cz;          // kind 0 only
    };
    FamilyPlanView family_plan_view(int h) const;
    // J-b1 P9b (P9a-2; design §2.10 "Canonical branch vector and W(u)"): the per-pivot frames W_p of a content
    // (PlanScaffold::pivot_frames_ on the pivots' pullbacks Q_p = C0† M_p C0), cached per content; the shot's Pauli
    // frame is P_rec · Π_p W_p^{u_p} (u = the record's first ρ outcomes).
    const std::vector<Pauli>& family_pivot_frames(int h) const;
    // This sampler's handle of a (kind, content) key — records are read by their CONTENT (another sampler's
    // handles are local to it): interned on first sight, a key under two kinds is refused.
    int family_intern_content(int kind, const std::string& content);
    // The retained state of a record (content handle, P_rec words, y words) by FORCED PROJECTION.
    // J-b1 P8: C0 is applied as the C0̂ amplitude map on the register F₀ ∪ E_S (the plan's term images, branch
    // scaffold) by default; tableau = true is the gate-level route on the frame tableau (the 1e-12 comparator; deleted
    // at J-b2).
    FramedSuperposition family_materialize(int handle, const uint64_t* prec, const uint64_t* y,
                                           bool tableau = false, int branch = 0) const;
    // J-b1 P9b: the consumer branch law (design §2.10 Lemma R3; E_min = ∅ branches). A carried producer branch is
    // registered with (i) its consumer bare `st` — the SAME frame (U, F₀, eps) as this sampler's bare, its own
    // amplitudes (the plans never read α: every plan is shared with branch 0), (ii) the eps delta of the branch as a
    // Pauli: `dspec` on the full register (wires no consumer gate touches: applied as the shot's P) and `dport` the
    // frames-class row (X then Z per INPUT_QUBITS wire: composed at t = 0 like an input_pauli frame). Returns the
    // engine branch id (>= 1). Per shot, family_set_input_branch's id selects the branch: a branch shot takes the term
    // route (framed) and walks ITS Born trie (one per (plan, branch): the sparse map, branch 0 never probes it).
    // D6 (design §2.10 "Memory"): the branch's consumer bare and its tries are an EVICTABLE payload (the header —
    // id, eps delta — is permanent); `resupply` rebuilds the bare by the SAME pure function add_branch ran (validated
    // there: no refusal is reachable inside run()), called inside run() on a miss. Budget: family_set_branch_budget
    // (bytes over the resident branch payloads; SIZE_MAX = never evict).
    int family_add_branch(const FramedSuperposition& st, const Pauli& dspec, const std::vector<uint8_t>& dport,
                          std::function<FramedSuperposition()> resupply = nullptr);
    void family_set_input_branch(const int32_t* ids, long n);   // nullptr: every shot is branch 0
    void family_set_branch_budget(size_t bytes);
    void family_evict_branches();                 // D6 test knob: evict every re-suppliable branch payload now
    struct FamilyBranchCounters {
        long branches = 0, shots = 0, map_probes = 0, tries = 0, trie_nodes = 0, trie_node_evals = 0;
        long resident = 0, evictions = 0, resupplies = 0;
        size_t resident_bytes = 0;
    };
    FamilyBranchCounters family_branch_counters() const;
    // signs() core (design §4): per row, per Pauli g (Hermitian, exact phase): +1 / −1 when g is definite on
    // the row's retained state by the H-membership rule, else 0. out: rows × g.size(), row major.
    void family_signs(const FamilyRecordSink& rs, long rows, const std::vector<Pauli>& g, int8_t* out) const;
    // J-b1 P7 (test API): the record operators (storage column order, exact phase) and the layout-3 record
    // fidelity of `rows` records through a record-mode RecordFidelity built on this sampler's bare state and
    // record operators (the caller checks the model identity): each record's content handle is resolved to its
    // C0 (kind 0: S^a·CZ; kind 2: rot(S) through the term view), interned once per content.
    std::vector<Pauli> family_record_ops() const;
    // J-b1 P9a (test API): a content handle's record table (the readers' memo / the (G) builder).
    std::shared_ptr<const RecTable> family_content_table(int handle) const;
    // corr: uint8[rows, 2, |out_wires|] output-Pauli corrections (X row, Z row) on the
    // deferred wires out_wires (RecordFidelity::evaluate_record's qx / qz); nullptr = none.
    void family_record_fidelity(RecordFidelity& rf, const FamilyRecordSink& rs, long rows, double* out,
                                const uint8_t* corr = nullptr,
                                const std::vector<int>* out_wires = nullptr);
    // record_report() classes: per record column (storage order), the NOISELESS (R = I) class
    // 0 deterministic / 1 coin / 2 born (the identity plan's record table).
    std::vector<int> family_record_classes() const;
    // G-8d (gauge-plan G-3): the state-model identity of this sampler's records — the bare state (frame rows,
    // signs, free rows, amplitudes), the certified generators and EVERY record operator with its global order,
    // its class and its inv ⊕ ref fold, the channel classify signs, and every measurement record's invert and
    // readout-flip probability (the folds). Records are read (materialize /
    // signs) only by a sampler with the same key (a gauge-only difference refuses).
    std::string family_model_key() const;
    // S8-0b (plan r2.3, gate G-PREP): every construction product, serialized — the bare state (frame rows, signs,
    // free order, amplitudes), the certified generators, the propagation table (every atom's c_prop / PPR residual / general tableau), the
    // per-alternative fired lists, AltAtom contents, the union pair list, AltMasks, the flat alt blob, the channel
    // rows and masks. The test API hashes it (_ctor_digest); no run path reads it.
    std::string family_ctor_bytes() const;
    // S8-1' (5-a) oracle: recomputes every diagonal alternative's content by compose_fired + finalize (the dense
    // chain the sparse composition replaced) and returns {alternatives checked, mismatches}. Test API only.
    std::pair<long, long> family_alt_atoms_oracle() const;
    // J-b1 P4: arms the term view of an unarmed sampler from the caller's copy of the construction inputs (the
    // test API's accessors; no run path builds it). A TermClassRefusal propagates.
    void family_arm_terms(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads);

  public:
    // S2.2: when true, run() routes every shot through the retention path and delivers its record
    // (plan, prefix, outcomes) + its σ (σ-4) to the
    // sink. Off by default — the production sampling path is untouched (byte-identical). Only
    // the DIAGONAL residual class is supported under retention (a CH/PPR circuit throws in run()).
    void set_retain_state(bool on);
    // S0(c) / D6: caller byte budget for plan PAYLOADS (0 = unbounded). Enforced only BETWEEN
    // shots: at each shot boundary the oldest resident payloads are evicted until the total is
    // within budget (the shot that just ran no longer holds any; the next shot rebuilds what it
    // needs, overshooting transiently — reported, never refused). Headers are never freed.
    void set_plan_payload_budget(size_t bytes);
    // Task 4: set per-shot input Pauli frame (data layout: shots × 2*support, X-part then Z-part).
    // data=nullptr disables; safe to call before/after run().
    void set_input_frame(const uint8_t* data, int stride);

    // S2b (design §2.6): one record-route shot's retained post-barrier + post-decision state from its
    // record — plan_key (the canonical residual plan bytes; empty = clean/identity shot) and its
    // outcomes (u, z) = coins — by FORCED PROJECTION: bare → C0 = S^a·CZ → P → every pivot record and
    // every direction introducer forced at its raw value y (record_outcomes). Nothing else is
    // projected (a check no record reads stays coherent). Called once per bucket representative.
    FramedSuperposition materialize_shot(const uint8_t* plan_key, int n_plan,
                                         const uint8_t* coins, int n_coins) const;

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
        RecTable rec;        // S2b: the plan's record table (canonical record-order basis), built
                             // iff the law is diagonal and not a fallback — the same build the
                             // production payload runs
    };
    // S2b: the record family's global record order per storage column (channels, then extras).
    std::vector<int> record_column_order() const;
    // S2b: the record operators M_c (storage column order: channels, then the extras), exact phase.
    std::vector<Pauli> record_operators() const;
    // S2b: the record channels' classify signs (channel c's internal bit = raw y_c ⊕ ref_c).
    std::vector<uint8_t> record_channel_refs() const;
    int record_value_words() const;
    PlanStructure plan_structure(const uint8_t* plan_key, int n_plan) const;
    // S2a test oracle: re-evaluate EVERY evaluated node of every resident plan's Born trie from
    // scratch (BornTrie::p0_from_scratch) and compare with the stored double bit for bit.
    // Returns {nodes checked, mismatches}. Off the shot loop; mutates no stored node.
    std::pair<long, long> born_trie_check();
    // S2a independent oracle: every evaluated node against the DELETED chain's semantics on the
    // n-qubit state — a bare copy, measure_pauli of the derived n-qubit Λ_0..Λ_{j−1} at the node's
    // z-prefix (forced outcomes), then FramedSuperposition::born_p1(Λ_j) — compared bit for bit.
    // Shares only derive_kernel_logical with the trie path. Returns {nodes checked, mismatches}.
    std::pair<long, long> born_trie_oracle() const;
    // J-a final review m1 (test API; twirl_sampler_fam.cpp): the same bitwise walks over the FAM route's
    // sampler-owned plans (kind 2 and route 1: FamRoute::k2plans, outside the cache's header table) —
    // {checked kind 2, checked route 1, bad}; oracle = false: BornTrie::p0_from_scratch (counters restored), true:
    // measure_pauli / born_p1 semantics. family_trie_nodes: the non-NaN nodes of every resident record trie by a
    // linear scan, {diagonal cache, kind 2, route 1} (the walks' exact expected counts).
  private:   // J-b1 P3 (m6): test surface (TwirlSamplerTestAccess)
    std::vector<long> family_born_trie_walk(bool oracle);
    std::vector<long> family_trie_nodes() const;

  public:

    // ── T5 step 0 (exact-residual arc): PURE-DATA born-decision-operator exposure ─────────────
    // The Born-measured decision operators (LOGICAL/ANTI-class DECISIONs), in declaration
    // order.  op is the record-product Pauli over the n deferred wires (operator =
    // i^phase·X^x·Z^z, exact phase); S2b: the RAW outcome bit of decision j (0 = +1 eigenvalue)
    // is its record column (a row of the plan's record table; record_outcomes).  inv is the
    // deterministic record invert: the EMITTED decision
    // bit is raw ⊕ inv (⊕ readout-flip coin parity ⊕ input-frame relabel, when present — both
    // affect the RECORD only, never the collapsed state).  Empty when the circuit declares no
    // Born-class decision.  Consumer: the adaptq per-(check, plan) classifier extends its span
    // basis with the plan-CONJUGATED ops (basis eigenvalue = the raw record value).
    struct BornDecOp {
        int dec_index;   // the DECISION index this operator decides
        Pauli op;        // record-product operator (n deferred wires, exact phase)
        uint8_t inv;     // deterministic record invert bit
    };
    std::vector<BornDecOp> born_dec_ops() const;

    // ── record fidelity (feat/record-fidelity): READ-ONLY exposure of the data the per-shot
    // state model is built from (design note §1.1). Pure accessors — nothing on run() changes.
    //   bare_state()            — the bare state this sampler samples against (caller-owned).
    //   certified_generators()  — the certified +1 stabilizers g_i, in σ bit order (exact phases).
    //   record_channel_masks()  — per record channel c (DETECTOR / in-group OBSERVABLE / in-group
    //                             DECISION — the channels materialize_shot projects onto), the
    //                             generator-subset mask (σ words): G_c = Π_{i in mask} g_i.
    const FramedSuperposition& bare_state() const;
    const std::vector<Pauli>& certified_generators() const;
    std::vector<std::vector<uint64_t>> record_channel_masks() const;

    // ── Reporting accessors (the bench line + 5σ gate are reproducible from these) ──────────
    long used() const;                 // shots entering the accumulation (== shots since V3-T3)
    // Shots the selfcheck window's SEMANTICS ORACLE verified (born observable / born decisions;
    // first run only, ≤ 64). Diagnostic — 0 when selfcheck == 0 or no born channel exists.
    long semantic_oracle_shots() const;
    long clean_shots() const;          // no-fire shots
    long diag_shots() const;           // fired shots
    int num_channels() const;          // nchan
    int channel_words() const;         // CW = ceil(nchan/64)
    int deferred_wires() const;        // n = deferred wire count (port-v3 T2: the compact
                                       // record sink needs the a-mask byte length / prefix
                                       // word count = ceil(n/64) to configure its storage)
    int num_decisions() const;         // Task 2b.1: declared decision count (max DECISION index+1)
    size_t plans() const;              // diagonal plan cache size
    const std::vector<long long>& channel_one_counts() const;   // per-channel 1-bit counts
    // ── S0(a): route + memo/miss counters (record_report()). Exact at a fixed seed (no timing
    // enters them); cumulative like used() (reset by set_seed). `routes` lists EVERY
    // route in a fixed order ("fast" = the abelian streaming hot loop first); the routes sum to
    // used(). The two hot routes ("fast",
    // "retain_kappa0") are attributed per run, not counted per shot. Plan-miss splits are read off
    // the (possibly shared) plan caches, whose counters live with the caches.
    struct RecordCounters {
        std::vector<std::pair<std::string, long>> routes;
        // S2a: the Born trie (BornTrie's exact counters; see twirl_kernel.hpp). The κ chain and its
        // kappa_chain counters are deleted (walks/steps are the trie's shots/outcomes).
        long trie_walks = 0, trie_steps = 0, trie_nodes = 0, trie_node_evals = 0, trie_chi_j = 0;
        long trie_passes = 0, trie_max_support = 0;
        long trie_root_memo_hits = 0;   // S8-4: root nodes served by the root memo (not evaluated)
        int trie_max_depth = 0;
        // S2b record tables built by the sampler (cold), their add_extra op counts (M-1), and the
        // Born observable's OBS draws by route (Q2: record = 0 on every diagonal plan).
        long rec_tables = 0, rec_extras = 0, rec_ops_plane = 0, rec_ops_pivot_reduce = 0;
        long rec_ops_greduce = 0, rec_ops_echelon = 0, rec_pivots_from_extras = 0, rec_dirs_from_extras = 0;
        std::vector<long> rec_kappa_hist;
        long obs_draws_record = 0;
        // S5a: shots that entered the structural slow path (nf materialised + canonicalized; cold misses,
        // the selfcheck window, non-resident payloads, fallback / out-of-class plans).
        long structural_shots = 0;
        size_t plan_misses_kappa0 = 0, plan_misses_kappa_pos = 0, plan_misses_fallback = 0;
        // S0(c) plan memory (the diagonal cache; bytes are heap accounting):
        size_t payload_budget = 0;            // 0 = unbounded
        size_t payload_resident_bytes = 0, payload_peak_bytes = 0, payload_overshoot_bytes = 0;
        size_t header_bytes = 0, payload_builds = 0, payload_evictions = 0;
        // S2b r6.1 / r8: record-table resolutions by the record READERS: once per distinct plan per BUFFER on
        // a compact buffer (the buffer then holds the table), once per call for the per-shot accessors
        // (record_outcomes / materialize); never for an exact-fallback shot (its y is its stored record).
        long reader_lookups = 0;
        // r6.4: the readers' bounded record-table memo, charged to the payload budget (evicted first).
        long reader_tables = 0, reader_evictions = 0;
        size_t reader_bytes = 0, reader_peak_bytes = 0, reader_overshoot_bytes = 0;
    };
    RecordCounters record_counters() const;
    // ── S0d (diagnostics policy, design §5.0): the engine writes NO stderr line for these events.
    // Construction and run() append them here; the Python layer drains them with
    // take_diagnostics() AFTER run() / construction returns (GIL held, never mid-loop) and raises
    // the warnings / logging records (xtim/_diagnostics.py). The hot loop only counts: an
    // EXACT_FALLBACK event is the per-run delta of record_counters()'s exact_* route counters,
    // appended once after the shot loop; the other kinds carry their first-occurrence payload.
    //   EXACT_FALLBACK   name = exact_* route, count = shots of this run on it, shots = run size
    //   SELFCHECK_WINDOW count = the oracle window size (first window of the process only)
    //   DISK_LOAD        name = "LOADED" | "absent/rebuild", count = plans, detail = cache path
    //   ORACLE_FAILURE   name = "plan_key" | "local_index", count = shot index, detail = message
    //                    (run() then returns false: the oracle abort)
    //   BORN_NODE        S2a, decision 7 (large-χ): queued at the FIRST Born-trie node evaluation
    //                    of a run() (at most one per call; none when no node is evaluated):
    //                    count = χ (the bare free-register support), shots = κ_rec of that plan,
    //                    detail = that node's evaluation wall time in µs. No threshold, no cap.
    struct DiagEvent {
        enum Kind : int { EXACT_FALLBACK = 0, SELFCHECK_WINDOW, DISK_LOAD, ORACLE_FAILURE, BORN_NODE, DISK_SAVE,
                          CHANNELS };   // S7: the ctor's circuit-channels line (detail = text)
        Kind kind;
        std::string name;
        long count = 0;
        long shots = 0;
        std::string detail;
        double value = 0.0;   // BORN_NODE: the first node's evaluation time in SECONDS (raw; the
                              // Python layer formats it — no formatting inside run()); 0 otherwise
    };
    std::vector<DiagEvent> take_diagnostics();   // moves the pending events out (oldest first)
    double draw_seconds() const;       // accumulated noise-draw wall time
    double wall_seconds() const;       // accumulated end-to-end run() wall time
#ifdef XTIM_BENCH_REPLAY
    // BENCH-ONLY (compiled with -DXTIM_BENCH_REPLAY; absent from the shipped extension): the
    // fast-path replay micro-gate's event source. bench_record(true) records every noise batch
    // run() consumes (cleared on each call); bench_replay(off, ev) makes run() read shot i of
    // each call from the recorded source (empty off = back to the live sampler).
    void bench_record(bool on);
    const std::vector<int32_t>& bench_rec_off() const;
    const std::vector<int32_t>& bench_rec_ev() const;
    void bench_replay(std::vector<int32_t> off, std::vector<int32_t> ev);
    uint64_t bench_fingerprint() const;
    double bench_noise_only(long shots, long* events);
    // S8-0b t_build (redefined, plan r2.5): the PRODUCTION cold build of each recorded kind-0 plan key (plan_structure
    // byte format) — the canonical cache key words, TwirlPlanCache::get_or_build (payload + law) and the record table
    // with its emission block (ensure_rec_) — on this sampler's cache. Returns the plans built; a key with a pair
    // outside this alt set's pair universe is skipped (unreachable).
    long bench_build_plans(const std::vector<std::string>& keys);
#endif
    // Circuit-channel compilation results (populated iff opt.circuit_channels). Channel
    // layout: channels [0, channel_detectors().size()) are DETECTOR channels; channels
    // [channel_detectors().size(), num_channels()) are deterministic-OBSERVABLE channels
    // (V2-T4; IN_GROUP observables compiled exactly like detector channels).
    const std::vector<int>& channel_detectors() const;      // det channel c → detector index
    const std::vector<int>& channel_observables() const;    // obs channel k → observable index
    // ── V3 Born-weighted observable channel (single logical observable, diagonal tables) ──
    // born_obs_index() >= 0 ⇒ that observable is emitted per shot as a σ-correlated biased
    // coin (exact joint with the deterministic detectors; executable spec
    // scripts/twirl_obs_reference.py). Accessors for the gate battery:
    int born_obs_index() const;        // observable index, -1 = mode off
    // The Born observable's noiseless REFERENCE parity (0/1). The engine (sinks, obs_ones) carries
    // the ABSOLUTE Born bit; a reporting consumer XORs born_obs_ref() into column born_obs_index()
    // to report it reference-relative (Stim's reference_sample convention = engine="exact"), as it
    // XORs the channels' reporting refs into the deterministic channels. 0 when no Born observable / input port.
    int born_obs_ref() const;
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
    struct OwnCaches;   // the plan caches make() creates before the sampler (private: never shared)
    TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                       const std::vector<std::pair<int, int>>& reads,
                       const std::vector<std::vector<int>>& detectors,
                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                       uint64_t seed, const TwirlRecordOptions& opt, OwnCaches&& caches,
                       std::optional<PropagationTable>&& class_table);
    std::unique_ptr<Impl> impl_;
};

// J-b1 P13 rulings: the owner's static fixed-axis rule on a deferred circuit, outside any sampler (twirl_sampler_fam.cpp).
struct StaticRuleVerdict {
    bool accept = true;
    std::string message;          // the law-limitation text when refused
    std::vector<Pauli> axes;      // the two anticommuting pushed axes (deferred wires)
    int gate_i = -1, gate_j = -1, site = -1;
};
StaticRuleVerdict family_static_rule(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                                     double p_factor);

// J-b1 P9b (fixplan-p9b-frame review C-R3): forced records (family_materialize, test API) whose pivot differs from
// the pattern-only rule because C0̂ promoted E_S rows into the register (Lemma R1's gate on the materialize path).
long family_forced_pivot_nonpattern_count();

}  // namespace qeccore
