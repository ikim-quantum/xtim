#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/diag_error_sampler.hpp"
#include "qeccore/factored_stab.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/sampler.hpp"     // FiredPauli / compose_fired

namespace qeccore {

// FramedPauliShotSampler — compile-once / sample-many shot driver for a FramedSuperposition bare
// state under PAULI noise, at production-sampler speed. It reuses the sampler's validated A/B
// machinery on the shared frame structure:
//   compile:  FramedSuperposition (dense backend) -> factorize_framed = A (product qubits) (x)
//             B (entangled block); precompute the active-set work + A-qubit axis lookup.
//   per shot: fold the fired Paulis into a FactoredDiagError (X->v, Z->S^2, Y->both; no CZ, so the
//             active set is exactly B plus nothing), framed_active_block builds the |B|-qubit block
//             with the error conjugated in, batch_measure collapses ONLY the block reads, and every
//             A-qubit read is answered in closed form by reduced_read (deterministic or fair coin).
// This is the same shot shape as sampler.cpp's run_ab_reduced_shot, restricted to the Pauli error
// class (no S/CZ-dressed propagated errors, no read-basis rotation patterns). Distribution-exact:
// validated statistically against per-read FramedSuperposition::measure_pauli.
//
// FAST PATH (the cascade analogue). For Pauli noise the active set is ALWAYS exactly B (no CZ ever
// pulls an A-qubit), so the read partition is fixed, and any shot whose fired errors avoid B leaves
// the block's joint read distribution identical to the ZERO-ERROR one. That structure is extracted
// ONCE at compile time as a TreePlan (Stage T1, 2026-07-03 spec; correction-CHAIN construction
// since the same date): a conditional-coin DAG built by forced collapse — each internal node
// resolves the next free-coupling read in batch_measure's Pass-1 order as a biased coin
// (u < p_plus, the measure_pauli convention), each leaf is the chi=1 GF(2)-affine remainder
// (base bit XOR coin-parity mask — the exact symbolic mirror of batch_chi1's coin bookkeeping).
// At each biased coin the two forced posts are generically Pauli-related (always at chi<=2);
// the relating Pauli is solved off the shared frame (solve_pauli_correction) and stored on the
// node instead of recursing into the -1 subtree, so the build is LINEAR in depth for the
// generic case. Only genuinely non-Pauli-related posts (possible at skewed chi>2) branch for
// real, up to a node cap (default 4096, QEC_TREEPLAN_CAP), over cap => extraction fails and the
// caller keeps live-block sampling. chi=1 is the r=0 flat instance; chi=2 the depth-1 instance
// (root coin == the former p_plus — bit-identical streams). Per shot the plan replays in
// O(depth + reads) bit-ops. Shots whose errors touch B (~|B|·p of them) take the full
// framed_active_block + batch_measure path.
//
// Over-cap / out-of-scope plans (Stage G3 fallback, kept): the zero-error base outcomes come from
// LIVE-BLOCK SAMPLING instead — a per-shot block clone + batch_measure over the same fixed read
// partition (batch_measure and the G1-general factorize are chi-uniform). O(block) at any chi.
//
// Scope: dense Amplitudes backend only (throws otherwise); chi-general.
struct FramedPauliShotSampler {
    // reads[k] = (pauli 0:X 1:Y 2:Z, qubit), distinct qubits (terminal reads). [Stage 2: the
    // former FramedSuperposition-taking bridge ctor collapsed into this one — one state type now.]
    FramedPauliShotSampler(const FramedSuperposition& bare, std::vector<std::pair<int, int>> reads);

    // fired[j] = (qubit, pauli 0:X 1:Y 2:Z). out[k] = ±1 per read. `rng()` returns U[0,1).
    void sample_shot(const std::vector<std::pair<int, int>>& fired,
                     const std::function<double()>& rng, std::vector<int>& out);

    int n() const { return n_; }
    int block_qubits() const { return f_.block.n(); }
    bool plan_active() const { return plan_ok_; }
    const std::vector<std::pair<int, int>>& reads() const { return reads_; }
    const FactoredBareState& factored() const { return f_; }
    const std::vector<uint8_t>& a_axis1() const { return ax1_; }   // wire -> A axis+1 (0: not in A)
    const std::vector<uint8_t>& a_sign() const { return sgn_; }    // wire -> A eigenvalue sign

    // Zero-error base outcomes: replay the precomputed plan when plan_active(), else draw them by
    // live-block sampling (chi>2 / out-of-scope plans). Used by the circuit-level driver as the
    // cascade reference for tier0 shots.
    void base_outcomes(const std::function<double()>& rng, std::vector<int>& out);

    // General diagonal-Clifford error shot (S-powers + CZ pairs + X^v): the full A/B path with a
    // PER-SHOT read partition (an entangling CZ can pull A-qubits into the active block, so the
    // block column set is not fixed). E must be the folded normal form (S, CZ, X order).
    void error_shot(const FactoredDiagError& E, const std::function<double()>& rng,
                    std::vector<int>& out);

    // Pass-2 structure of a chi=1 block measurement: per read, outcome bit = base XOR
    // parity(coin bits selected by cmask). Fresh-coin reads carry their own coin id in cmask.
    // cmask is stored FLAT (read j's row at [j*cw, (j+1)*cw)) so build-time consumers (leaf
    // hash-cons key, E1 coin cross-check) walk one contiguous array. The per-shot replay uses the
    // TRANSPOSED packed form: outcome-bit words = basew XOR the tog rows of the set coins (read j
    // = bit j), so a shot costs O(set coins + reads) word/bit ops instead of a per-read
    // mask-row popcount scatter.
    struct BlockPlan {
        int n_coins = 0;
        int cw = 0;                                  // coin words per row = (n_coins + 63) / 64
        int rw = 0;                                  // read words = (base.size() + 63) / 64
        std::vector<uint8_t> base;                   // per covered block read
        std::vector<uint64_t> cmask;                 // flat coin-parity masks (base.size() * cw)
        std::vector<uint64_t> basew;                 // base bits packed over reads (rw words)
        std::vector<uint64_t> tog;                   // transposed: coin c's read-toggle row at
                                                     // [c*rw, (c+1)*rw) (n_coins * rw words)
        std::vector<int32_t> coin_src;               // coin id -> the covered-read index that
                                                     // created it (its fresh-coin read)
    };

    // TreePlan — the conditional-coin DAG (one formalism for all chi). Internal node = the next
    // free-coupling read (Pass-1 order) as a biased coin over its two forced-collapse children;
    // leaf = the chi=1 remainder's BlockPlan over `pos` (the still-unresolved block-read indices,
    // in read order). Nodes are HASH-CONSED bottom-up on (read_pos, p_plus bits, child ids) /
    // the leaf's full plan content — structurally identical futures share one node, so factorized
    // amplitude tensors collapse to linear-size DAGs. child[v] = -1 marks an impossible outcome
    // (p < 1e-12; never drawn). Replay: walk from root drawing u < p_plus per node (+1 -> child 0),
    // then the leaf's GF(2) plan on the shared fair-bit pool.
    struct TreePlan {
        struct Node {
            // ── replay-hot fields, kept on the leading cache line (the walk reads read_pos /
            //    p_plus / child every node and has_corr/osig/rflip0 on corrected trees) ──
            int read_pos = -1;               // block-read index resolved here; -1 = leaf
            // CORRECTED biased coin marker (the generic chain-build shape, any R): the two
            // forced-collapse posts are Pauli-related, |post_-> ∝ corr·|post_+>, so only the +1
            // subtree is stored (child[1] == child[0]) and an outcome of -1 (in the ORIGINAL
            // path frame) fires the correction instead: downstream read outcomes/probabilities
            // toggle by `rflip` (packed per-block-read anticommutation row of corr, single-word
            // mirror rflip0) and — on R>0 channel trees — each observable's sign by the
            // channel-filled masks below.
            bool has_corr = false;
            double p_plus = 1.0;             // P(read = +1 | path)
            int32_t child[2] = {-1, -1};     // node id per outcome (+1 -> 0, -1 -> 1)
            // Observable sign word (channel build fills it): bit r (r < 64) = corr anticommutes
            // with observable r's block restriction. Replay XORs it into the shot's sign
            // accumulator on fire — the zero-event fast path then needs no per-node loop.
            uint64_t osig = 0;
            uint64_t rflip0 = 0;             // rflip word 0 (block reads <= 64: the norm)
            double p_sel[2] = {1.0, 0.0};    // {pr+/tot, pr-/tot} — toggled-draw table (p_sel[1]
                                             // is the -1-frame Born value to the BIT, not
                                             // 1 - p_plus; see the builder note)
            // ── cold payloads ──
            BlockPlan plan;                  // leaf payload
            std::vector<int> pos;            // leaf payload: covered block-read indices
            Pauli corr{0};                   // correction Pauli (channel build / touched eval)
            std::vector<uint64_t> rflip;     // full per-read anticommutation row of corr
            // Expanded-leaf payloads (expand_corrected_tree): the base leaf whose channel data
            // (val0/sig/coinC/post) this path variant shares, and the path's fired corrected
            // nodes (ORIGINAL node ids — their corr/nsig entries stay live). osig on an expanded
            // leaf holds the path's baked observable sign word; cxor the path's baked coin-space
            // toggle (XORed into the drawn coin words: outcome rows via the tog loop AND the
            // channel's implied-coin export — see the coin-gauge note at the replay).
            int32_t base_leaf = -1;
            std::vector<int32_t> fpath;
            std::vector<uint64_t> cxor;
        };
        std::vector<Node> nodes;
        int32_t root = -1;
        bool any_corr = false;               // any corrected node (replay toggle path engaged)
        int n_corr = 0;                      // corrected node count (fired-buffer bound)
        bool expanded = false;               // partially evaluated (plain replay, baked variants)
        int node_count() const { return (int)nodes.size(); }
    };

    // Plan introspection (tests/diagnostics): node count of the active zero-error tree plan
    // (0 when extraction failed / fell to live-block sampling).
    int plan_nodes() const { return plan_ok_ ? tree_.node_count() : 0; }

    // ── R>0 expectation-channel hooks (E1/E2b) ─────────────────────────────────────────────────
    // Rebuild the zero-error TreePlan in CHANNEL mode (correction-chain construction): at each
    // biased node the builder solves for the Pauli relating the two forced-collapse posts
    // (|post_-> ∝ C·|post_+>, C = the eps/amplitude-difference syndrome read off the shared
    // frame's dual structure) and, when found, stores C + its read-anticommutation row instead
    // of recursing into the -1 subtree — the leaf-coin correction rule extended to the whole
    // tree. The build is then LINEAR in depth where the old path-unique (uncons) build exploded
    // exponentially, leaf ids determine the reference leaf state (observable-sound), and a -1
    // outcome per corrected node enters the channel as one more sign coin. Nodes where no such
    // C exists (genuinely non-Pauli-related posts, possible at chi>2) keep both subtrees —
    // path-dependence the tree expresses exactly. Captures each leaf's forced-collapse block
    // state once (the channel's leaf posts). Returns the new plan_active(); on failure (node
    // cap) the live-block base path holds and expectations use the live collapsed block.
    bool rebuild_plan_channel(std::vector<std::pair<int32_t, FramedSuperposition>>& leaf_states);
    int32_t last_leaf() const { return last_leaf_; }               // leaf id of the last plan_shot_
    const std::vector<uint64_t>& last_coin_words() const { return coin_words_; }
    // Corrected-node coins fired by the last plan replay (node ids, path order) — the R>0
    // channel's per-node sign coins (accumulated but unconsumed at R=0) — and their
    // accumulated observable sign word (XOR of the fired nodes' osig).
    const int32_t* last_fired_nodes() const { return fired_ptr_; }
    int last_nfired() const { return n_fired_; }
    uint64_t last_osign() const { return osign_; }
    TreePlan& tree_mut() { return tree_; }         // channel build annotates node osig words
    // Zero-error base outcomes FORCED onto the live-block path (R>0 when the plan/leaf channel is
    // unavailable): collapses a block clone into block_work() — the shot's zero-error block post-
    // state the expectation channel reads — and CONDITIONS it on the recorded outcomes of the
    // Pass-2 fresh-coin reads (the cascade expectation convention, E2b: batch_measure draws those
    // coins symbolically without collapsing the state; the physical post-state the expectations
    // live on is the outcome-projected one — dense-oracle-arbitrated in test_framed_r).
    // (base_outcomes only routes here when !plan_active().)
    void live_base_outcomes(const std::function<double()>& rng, std::vector<int>& out);
    FramedSuperposition& block_work() { return *block_work_; }     // post-block of the last live shot
    const std::vector<std::pair<int, int>>& block_read_list() const { return block_reads_; }
    const TreePlan& tree() const { return tree_; }                 // the active zero-error plan
    // R>0 bad-shot path: error_shot + the tensor-split expectation channel evaluated on the
    // collapsed active block — the legacy run_ab_obs_shot body (G2), hoisted into the framed
    // engine. `out` = final ±1 outcomes; exp[r] = <P_r> on the shot's post-state:
    //   s_pre · ∏(A-qubit closed-form factors) · <block_post| P|_active |block_post>.
    // The block post is conditioned on the recorded fresh-coin outcomes (cascade convention, E2b).
    void error_shot_obs(const FactoredDiagError& E, const std::function<double()>& rng,
                        std::vector<int>& out, const std::vector<Pauli>& obsP, double* exp);
    // Zero-error plan internals consumed by the zero-event expectation fast path (E2b).
    const std::vector<int>& a_coin_positions() const { return a_coin_pos_; }
    const std::vector<int>& zero_outs() const { return zero_out_; }
    const std::vector<uint64_t>& a_coin_words() const { return acw_; }

  private:
    void init_(const FramedSuperposition& bare);
    void build_plan_();
    void plan_shot_(const std::function<double()>& rng, std::vector<int>& out);
    void live_base_shot_(const std::function<double()>& rng, std::vector<int>& out);

    int n_ = 0;
    std::vector<std::pair<int, int>> reads_;   // (pauli, qubit)
    FactoredBareState f_;
    ActiveSetWork asw_;
    std::vector<uint8_t> ax1_, sgn_;           // A-qubit axis+1 / sign lookup (build_a_lookup)

    // fixed read partition (Pauli noise: active set == B always)
    std::vector<std::pair<int, int>> block_reads_;  // (pauli, block column)
    std::vector<int> block_pos_;                    // -> index in reads_
    std::vector<int> a_pos_;                        // indices in reads_ of A-qubit reads
    std::vector<uint8_t> a_kind_;                   // zero-error A-read: 0 det(+1), 1 det(-1), 2 coin
    std::vector<int> zero_out_;                     // baked zero-error out (+1 / det A-read values)
    std::vector<int> a_coin_pos_;                   // read indices of the kind-2 (coin) A-reads
    std::vector<uint64_t> acw_;                     // bulk A-read coin scratch

    // zero-error plan
    bool plan_ok_ = false;
    TreePlan tree_;                            // the conditional-coin DAG (all chi)

    // per-shot scratch (sized once; sparse-reset after each shot)
    FactoredDiagError E_;
    std::vector<int> touched_;
    std::unique_ptr<FramedSuperposition> block_work_;
    std::vector<int> b2g_;                     // block column -> global qubit
    std::vector<int> g2a_;                     // global qubit -> active column (else -1); error_shot
    std::vector<std::pair<int, int>> active_reads_;
    std::vector<int> active_pos_;
    std::vector<int> bits_;
    std::vector<uint64_t> coin_words_;
    std::vector<uint64_t> ozw_;                // packed leaf-outcome scratch (tree_replay_block)
    std::vector<uint64_t> togg_;               // corrected-coin read-toggle scratch (channel trees)
    std::vector<int32_t> fired_nodes_;         // chain-replay fired-coin scratch buffer
    const int32_t* fired_ptr_ = nullptr;       // fired corrected-node ids of the last replay
    int n_fired_ = 0;                          // how many
    uint64_t osign_ = 0;                       // their accumulated observable sign word
    int32_t last_leaf_ = -1;                   // leaf reached by the last plan replay (obs channel)
    std::vector<int> w2r_;                     // wire -> read index (else -1); error_shot_obs
    Pauli pb_scratch_{0};                      // error_shot_obs block-restriction scratch
};

// ── R>0 per-plan expectation channel ───────────────────────────────────────────────────────────
// The compile-time artifact that lets a tier0/pattern shot produce its PAULI_EXPECTATION row
// without touching the full state (the legacy refpost/obs_pp0/esign machinery, re-expressed on
// the A/B tensor split — 2026-07-03 framed-R spec §"What R>0 needs"). Per observable P (full
// space), the shot's value factorizes over |φ> = (⊗ A-qubit posts) ⊗ |block post>:
//   <P> = (-1)^{V·z(P)} · s_pre(Pc) · ∏(A-letter factors) · <block leaf post| Pc|_B |block leaf post>
//         · (-1)^{Σ_c coin_c · anticomm(C_c, Pc|_B)}
// where Pc = Δ†PΔ (the shot's folded diagonal error, kind-2 CZ pairs excluded — they live in the
// pattern state), a READ A-letter contributes (letter == this plan's read basis) ? BASE outcome : 0,
// an UNREAD A-letter its zero-error axis factor (reduced_read; off-axis = exact 0), and the block
// factor is the per-LEAF reference post-state value corrected by the leaf coins' Pauli corrections
// C_c (the cascade-theorem sign rule — legacy CascadeCoin::esign0 semantics). Untouched shots
// (error misses the observable) use the precomputed val0/sig masks; touched shots build Pc and do
// one live framed_expectation on the stored leaf post.
struct FramedObsChannel {
    bool split_ok = false;              // static per-observable split built
    bool leaves_ok = false;             // every TreePlan leaf carries a valid channel
    struct Split {                      // static tensor-split of one observable at this plan's reads
        double s_pre = 1.0;             // i^{phase - #Y} of the full P (Hermitian => ±1)
        double a_const = 1.0;           // product of the unread-A deterministic factors
        bool zero = false;              // structurally 0 (basis-mismatched read / off-axis unread)
        std::vector<int> a_reads;       // read indices whose BASE outcome multiplies in
        Pauli Pb{0};                    // block restriction (this plan's block columns)
        bool block_any = false;
        // Zero-event fast path (zero-error plan only, E2b): on a shot with NO fired events the
        // value collapses to fast_const · leaf.val0 with sign = parity(leaf.sig & leaf coins)
        // XOR parity(acmask & A-read coin words) — fast_const folds s_pre · a_const · the
        // deterministic A-read outcomes, acmask selects the fair-coin A-reads in a_reads
        // (bit i = position in the plan's a_coin list). Exactly the untouched general path.
        double fast_const = 1.0;
        std::vector<uint64_t> acmask;
    };
    std::vector<Split> split;           // per observable
    struct Leaf {                       // per TreePlan leaf, indexed by node id
        bool ok = false;
        FramedSuperposition post{0};    // reference (coins = 0) block post-state, renormalised
        std::vector<Pauli> coinC;       // per leaf-plan coin: correction Pauli (block space)
        std::vector<double> val0;       // per obs: <post|Pb|post> (1.0 when !block_any)
        std::vector<std::vector<uint64_t>> sig;   // per obs: coin mask (bit c: C_c anticommutes Pb)
    };
    std::vector<Leaf> leaves;           // size = tree node count; only leaf ids populated
    // Per corrected tree node (indexed by node id): observable sign mask — bit r set when the
    // node's correction Pauli anticommutes with observable r's block restriction Pb. A fired
    // node coin XORs this into the shot's sign parities (untouched path); touched shots
    // recompute the anticommutation against the error-conjugated Pb from the node's stored corr.
    std::vector<std::vector<uint64_t>> nsig;
    std::vector<int> g2a;               // global qubit -> this plan's block column (else -1)
    std::vector<int8_t> rbasis;         // per read index: this plan's read basis (patterns swap X<->Y)
    int nb = 0;                         // block qubit count
};

// FramedCircuitShotSampler — CIRCUIT-LEVEL-noise shot driver: the production sampler's workload
// (every circuit noise location propagated to its end-of-circuit S/CZ-dressed diagonal error) on
// the FramedSuperposition structure. Compile once:
//   * build_propagation_table + compose_fired give every noise alternative's end-of-circuit
//     (a: S-powers, cz pairs, v: X mask) — one AltSide per flat alternative;
//   * a CZ pair universe with the tier0 classification (0: unobservable, 1: relabel — the non-Z
//     read's outcome XORs its Z-read partner's, 2: entangling/bad) and per-alternative read-flip
//     masks (X^v flips Z- and Y-reads);
//   * the FramedPauliShotSampler compile artifacts (factorize_framed A/B split + zero-error plan).
// Per shot: DiagErrorSampler draws the fired alternatives (geometric skip, O(fired)); the fired
// sides fold sparsely into the (a, cz, v) normal form with the X-crossing corrections; classify:
//   tier0 (no odd S-power on an X/Y-read wire, no kind-2 CZ): outcomes = zero-error plan replay,
//         then relabels (on base values), X^v flips, and S²-on-X/Y-read flips — the cascade
//         theorem: a diagonal error that rotates no read basis and entangles no read only
//         relabels the terminal joint law, REGARDLESS of which qubits (A or B) it touches;
//   else: full A/B path (error_shot) — framed_active_block with the general E + batched block
//         measurement + closed-form product reads.
// Scope: THE production engine, R >= 0 (measurement records always; with enable_expectations the
// per-shot PAULI_EXPECTATION row too); channels X/Y/Z_ERROR, DEPOLARIZE1/2, PAULI_CHANNEL_1/2;
// chi-general (tier0 base outcomes and pattern plans replay the TreePlan DAG at any chi; over
// the node cap they keep G3's live-block sampling / error_shot fallback — see
// FramedPauliShotSampler above). Distribution-validated against (i) the gate-applied
// FramedSuperposition reference, (ii) the retained full-state fallback (QEC_FORCE_FALLBACK=1),
// and (iii) a dense statevector oracle at chi<=16 (test_ab_general_chi, test_treeplan,
// test_framed_r — the R>0 channels are additionally record-conditioned EXACT vs the dense law).
struct FramedCircuitShotSampler {
    // `deferred` = the normalized+deferred circuit (feedback coherentized/stripped) whose terminal
    // reads are `reads[k] = (pauli 0:X 1:Y 2:Z, wire)`; `bare` = its bare state. `table`, when
    // non-null, is REUSED (must be the deferred circuit's table) instead of rebuilding it —
    // compile_sampler_program already owns one. [Stage 2: the former FramedSuperposition-taking
    // bridge ctor collapsed into this one — one state type now.]
    FramedCircuitShotSampler(const FramedSuperposition& bare, const Circuit& deferred,
                             std::vector<std::pair<int, int>> reads, uint64_t noise_seed,
                             const PropagationTable* table = nullptr);

    // Reset the run state for a fresh sampling call: reseed the noise stream and CLEAR the
    // pattern memo. Per-call memo semantics are load-bearing for the same-seed ⇒ same-bytes
    // contract — a memo persisting across calls would make which shots take the memoized vs
    // full path (different RNG consumption) depend on prior call history.
    void begin_run(uint64_t noise_seed);

    // Sample one shot (noise drawn internally from the compiled channels). out[k] = ±1 per read.
    void next_shot(const std::function<double()>& rng, std::vector<int>& out);
    int last_nev = -1;   // DIAGNOSTIC: fired-event count of the last shot (next_shot / shot_from_events)
    std::vector<int32_t> last_ev;  // DIAGNOSTIC: flat alt ids of the last shot

    // Event-injection entry: run one shot from a CALLER-SUPPLIED fired-alternative list (flat alt
    // ids in channel-ascending order — the same encoding next_shot's internal event batches use).
    // The seam for alternative event sources (stratified/rare-event sampling in research/, forced-
    // pattern tests): everything downstream (fold/classify/replay/A-B dispatch) is the validated
    // shot path.
    void shot_from_events(const int32_t* ev, int nev, const std::function<double()>& rng,
                          std::vector<int>& out);
    // Per-channel alternative probabilities, in flat-alt-id order (channel-major). The channel
    // fire probability is the row sum; alternative selection within a fired channel is the
    // normalized row. Feeds external event sources.
    std::vector<std::vector<double>> channel_alt_probs() const {
        std::vector<std::vector<double>> r;
        for (const auto& ch : channels_) { r.push_back({}); for (const auto& a : ch.alts) r.back().push_back(a.p); }
        return r;
    }

    // ── R>0 expectation channels (production since E2b) ────────────────────────────────────────
    // Enable the PAULI_EXPECTATION channel: obsP = the declared observable Paulis (deferred
    // space, plan.obsP order), bare = the full bare state (kept for the full-state fallback).
    // After this, every shot also fills last_expectations() (size R). Returns false only on a
    // structural failure (the caller then rejects the program — there is no other engine).
    bool enable_expectations(const std::vector<Pauli>& obsP, const FramedSuperposition& bare);
    bool expectations_enabled() const { return exp_on_; }
    const std::vector<double>& last_expectations() const { return exp_; }
    // Full-state fallback shot counter. The engine retains ONE full-state general path (clone
    // the bare state + fold the complete error + tagged lean collapse + Born reads — the former
    // run_lean_fallback_shot body); QEC_FORCE_FALLBACK=1 (read at construction) forces EVERY
    // shot onto it, at R==0 too — the engine-neutral A/B oracle lever for the fast paths.
    long fallback_full_count() const { return fallback_full_; }

    long tier0_count() const { return tier0_; }
    long full_count() const { return full_; }
    int block_qubits() const { return sp_.block_qubits(); }
    double mean_fired() const { return fired_ ? (double)fired_ / (tier0_ + full_) : 0.0; }
    // Zero-error TreePlan introspection (tests): active? / DAG node count (0 = live fallback).
    bool zero_plan_active() const { return sp_.plan_active(); }
    int zero_plan_nodes() const { return sp_.plan_nodes(); }

  private:
    struct AltSide {                                  // one flat alternative's folded atoms
        std::vector<std::pair<int, uint8_t>> a;       // (wire, S-power Z4)
        std::vector<int32_t> cz;                      // pair ids
        std::vector<int32_t> v;                       // X wires
        std::vector<uint64_t> mflip;                  // read-flip mask of the X part (MW words)
    };
    struct PairCls { uint8_t kind = 0; int target = -1, source = -1; };

    // Memoized plan for one BAD pattern (rot wires + entangling pairs), the pattern-cascade
    // analogue: the block dressed with the pattern's CZs, rot reads basis-swapped, TreePlan
    // extracted once (any chi, subject to the node cap); per shot it replays in
    // O(depth + reads) with the usual relabel layers on top. ok=false => error_shot fallback.
    struct PatternPlan {
        bool ok = false;                              // extraction succeeded (else full path)
        std::vector<std::pair<int, int>> block_reads; // (basis — swapped on rot wires, dressed col)
        std::vector<int> block_pos;                   // -> read index
        std::vector<int> a_pos;                       // non-active reads
        std::vector<uint8_t> a_kind;                  // their zero-error kind at the swapped basis
        std::vector<int> zero_out;                    // baked out (+1 / det A-read values)
        std::vector<int> a_coin_pos;                  // read indices of the kind-2 (coin) A-reads
        FramedPauliShotSampler::TreePlan tree;
        FramedObsChannel chan;                        // E1: expectation channel (exp_on_ only)
    };

    int pair_id_(int a, int b);
    void add_alt_(ErrorChannel& ch, double p, const std::vector<FiredPauli>& fired);
    void build_(const Circuit& deferred, uint64_t noise_seed, const PropagationTable* table);
    void process_shot_(const int32_t* ev, int nev, const std::function<double()>& rng,
                       std::vector<int>& out);
    // ── general (off-diagonal residual) shot path ───────────────────────────────────────────────
    // CH-class circuits: a noise atom whose end-of-circuit error is a general Clifford (PPR retry,
    // PropResult::general). Such an alt cannot fold into an AltSide (diagonal-only), so the shot's
    // terminal reads are instead CONJUGATED through the fired residuals (last-fired first — the
    // compose_fired order) and measured on a fresh bare clone: out[k] = <clone| E†M_k E |clone>.
    // Gathers a fired list's atoms (parallel to compose_fired's lookup) for the build-time general
    // detection / tableau compose; a general alt's composed error tableau (single atom = the stored
    // one, else the ordered composition, diagonal atoms promoted via left_ gates); the per-alt
    // diagonal read conjugation E†·M·E via the canonical diag_conjugate rules.
    std::vector<const PropResult*> gather_atoms_(const std::vector<FiredPauli>& fired) const;
    std::shared_ptr<const CliffordTableau>
        compose_alt_general_(const std::vector<const PropResult*>& atoms) const;
    void conj_by_diag_altside_(Pauli& M, const AltSide& sd) const;
    void process_shot_general_(const int32_t* ev, int nev, const std::function<double()>& rng,
                               std::vector<int>& out);
    PatternPlan build_pattern_(const std::vector<int32_t>& key) const;
    void replay_pattern_(const PatternPlan& P, const std::function<double()>& rng,
                         std::vector<int>& out);
    // R>0 helpers: per-shot channel evaluation (tier0/pattern; live != nullptr routes the block
    // factor through the live collapsed block instead of a leaf post), the zero-event leaf fast
    // path (fast_const/acmask popcounts — bit-equal to the general untouched path), and the
    // full-state fallback.
    void eval_channel_shot_(const FramedObsChannel& ch,
                            const FramedPauliShotSampler::TreePlan* tree,
                            const int32_t* fired_nodes, int n_fired, uint64_t osign, int32_t leaf,
                            const std::vector<uint64_t>& coins, const FramedSuperposition* live,
                            const std::vector<int>& out);
    void eval_zero_fast_(int32_t leaf, uint64_t osign);
    void full_fallback_shot_(const std::function<double()>& rng, std::vector<int>& out);

    FramedPauliShotSampler sp_;
    int n_ = 0, M_ = 0, MW_ = 0;
    std::vector<std::pair<int, int>> rd_;             // (pauli, wire)
    std::vector<int> rid_;                            // wire -> read index (else -1)
    std::vector<uint8_t> rbase_;                      // wire -> read basis (3 = unread)
    PropagationTable pt_;
    bool has_general_ = false;                         // pt_.has_general: any off-diagonal residual
    std::vector<AltSide> sides_;                      // flat alternative id -> side
    // Parallel to sides_: a general alt's composed end-of-circuit error tableau (null for diagonal
    // alts, whose action lives in sides_). A shot firing any general alt takes process_shot_general_.
    std::vector<std::shared_ptr<const CliffordTableau>> alt_tab_;
    std::vector<std::pair<int, int>> pair_list_;      // pair id -> (wire, wire)
    std::vector<PairCls> paircls_;
    std::unique_ptr<DiagErrorSampler> smp_;
    std::vector<ErrorChannel> channels_;              // kept for probability bookkeeping
    // per-shot fold scratch (sparse-reset)
    std::vector<uint8_t> acc_a_, in_dq_;
    std::vector<uint64_t> czacc_, vmask_, mflip_acc_;
    std::vector<uint8_t> in_dw_;
    std::vector<int32_t> dirty_q_, dirty_w_, vdirty_;
    std::vector<std::pair<int, int>> relabels_;
    std::vector<int> flips_;
    struct RotInfo { int read; uint8_t base; uint8_t a; };
    std::vector<RotInfo> rots_;                       // odd-S-power X/Y reads this shot
    std::vector<int32_t> badp_, key_;                 // entangling pair ids / memo key scratch
    std::map<std::vector<int32_t>, PatternPlan> memo_;
    FactoredDiagError E_;
    DiagErrorSampler::SparseBatch batch_;
    int cursor_ = 0;
    long tier0_ = 0, full_ = 0, fired_ = 0;

    // ── R>0 expectation channel state ──
    bool exp_on_ = false;
    bool force_fb_ = false;                           // QEC_FORCE_FALLBACK=1: every shot full-state
    std::vector<Pauli> obsP_;                         // declared observable Paulis (full space)
    int R_ = 0;
    std::vector<double> exp_;                         // per-shot output row
    std::unique_ptr<FramedSuperposition> bare_full_;  // full bare state (full-state fallback)
    FramedObsChannel zero_chan_;                      // the zero-error plan's channel
    FactoredDiagError Ez_;                            // all-zero error (unread-A closed forms)
    std::vector<uint64_t> zmask_;                     // per-obs conjugation scratch (n-bit words)
    Pauli pc_scratch_{0};                             // touched-path conjugated observable scratch
    Pauli pb_scratch_{0};                             // touched-path block-restriction scratch
    std::vector<uint64_t> no_coins_;                  // empty coin words (live route)
    std::vector<uint64_t> pat_coins_;                 // pattern replay coin words
    std::vector<uint64_t> pat_oz_;                    // pattern replay packed-outcome scratch
    std::vector<uint64_t> pat_acw_;                   // pattern replay bulk A-read coin scratch
    std::vector<uint64_t> pat_togg_;                  // pattern replay corrected-coin toggle scratch
    std::vector<int32_t> pat_fired_;                  // pattern replay fired-coin scratch buffer
    const int32_t* pat_fptr_ = nullptr;               // fired corrected-node ids (last replay)
    int pat_nfired_ = 0;                              // how many
    uint64_t pat_osign_ = 0;                          // their accumulated observable sign word
    int32_t pat_leaf_ = -1;                           // pattern replay leaf id
    std::vector<uint8_t> fb_a_, fb_v_;                // full-fallback fold scratch
    std::vector<std::pair<int, int>> fb_cz_;
    std::vector<int> fb_bits_;
    long fallback_full_ = 0;

  public:
    size_t memo_size() const { return memo_.size(); }
};

}  // namespace qeccore
