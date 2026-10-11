#include <memory>
#include <cstddef>
#include "qeccore/framed_sampler.hpp"   // FramedCircuitShotSampler — THE shot engine (E2b)
#include "qeccore/sampler.hpp"
#include "qeccore/bare_state_errors.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/diag_conjugate.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace qeccore {

// Pauli-channel slot order: pc1_slot_basis / pc2_{first,second}_code (circuit_ir.hpp, THE shared
// Stim-order definition). A factor code is 0 = identity, else pauli_code_basis(code).
static void fire_factor(std::vector<FiredPauli>& out, int loc, int qubit, int code) {
    if (code == 0) return;
    out.push_back({loc, qubit, pauli_code_basis(code)});
}

std::vector<FiredPauli> sample_noise(const Circuit& circ, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<FiredPauli> fired;
    for (int k = 0; k < (int)circ.stream.size(); ++k) {
        const Instr& ins = circ.stream[k];
        if (ins.kind != Instr::Kind::Noise) continue;
        switch (ins.channel) {
            case NoiseChannel::X_ERROR:
            case NoiseChannel::Y_ERROR:
            case NoiseChannel::Z_ERROR: {
                assert(ins.probs.size() == 1);
                PauliBasis p = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                             : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z;
                for (int q : ins.qubits)
                    if (uni(rng) < ins.probs[0]) fired.push_back({k, q, p});
                break;
            }
            case NoiseChannel::DEPOLARIZE1: {
                assert(ins.probs.size() == 1);
                for (int q : ins.qubits) {
                    double u = uni(rng);
                    if (u < ins.probs[0]) {
                        int idx = (int)(u / (ins.probs[0] / 3.0));
                        if (idx > 2) idx = 2;                      // guard the u ~ p edge
                        fired.push_back({k, q, (PauliBasis)idx});
                    }
                }
                break;
            }
            case NoiseChannel::PAULI_CHANNEL_1: {
                assert(ins.probs.size() == 3);
                for (int q : ins.qubits) {
                    double u = uni(rng), acc = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        acc += ins.probs[i];
                        if (u < acc) { fired.push_back({k, q, pc1_slot_basis(i)}); break; }
                    }
                }
                break;
            }
            case NoiseChannel::DEPOLARIZE2: {
                assert(ins.probs.size() == 1 && ins.qubits.size() % 2 == 0);
                for (size_t i = 0; i + 1 < ins.qubits.size(); i += 2) {
                    double u = uni(rng);
                    if (u < ins.probs[0]) {
                        int idx = (int)(u / (ins.probs[0] / 15.0));
                        if (idx > 14) idx = 14;                    // guard the u ~ p edge
                        fire_factor(fired, k, ins.qubits[i],     pc2_first_code(idx));
                        fire_factor(fired, k, ins.qubits[i + 1], pc2_second_code(idx));
                    }
                }
                break;
            }
            case NoiseChannel::PAULI_CHANNEL_2: {
                assert(ins.probs.size() == 15 && ins.qubits.size() % 2 == 0);
                for (size_t i = 0; i + 1 < ins.qubits.size(); i += 2) {
                    double u = uni(rng), acc = 0.0;
                    for (int idx = 0; idx < 15; ++idx) {
                        acc += ins.probs[idx];
                        if (u < acc) {
                            fire_factor(fired, k, ins.qubits[i],     pc2_first_code(idx));
                            fire_factor(fired, k, ins.qubits[i + 1], pc2_second_code(idx));
                            break;
                        }
                    }
                }
                break;
            }
        }
    }
    return fired;
}

DiagPauliClifford compose_fired(const PropagationTable& table,
                                const std::vector<FiredPauli>& fired) {
    DiagPauliClifford c = DiagPauliClifford::identity(table.n);
    // `fired` must already be in stream order (sample_noise emits it that way); a wrong order
    // would compose non-commuting C_props silently wrong, so it is asserted, not assumed.
    int prev_loc = -1;
    (void)prev_loc;                                 // only read in the assert; silence Release -Wunused-but-set-variable
    for (const FiredPauli& f : fired) {
        assert(f.location_index >= prev_loc && "fired list must be in stream (location) order");
        prev_loc = f.location_index;
        const auto [loc, qi] = table.find_slot(f.location_index, f.qubit, "compose_fired");
        if (f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) {
            assert(!loc->x_atom[qi].rejected);
            c = c.then(loc->x_atom[qi].c_prop);
        }
        if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) {
            assert(!loc->z_atom[qi].rejected);
            c = c.then(loc->z_atom[qi].c_prop);
        }
    }
    return c;
}

// Order-of-composition caution: then semantics are a.then(b) = b∘a (b applied after a). The
// compose_fired loop composes c = c.then(atom) walking `fired` in stream order — earlier
// locations enter earlier, so they end up applied FIRST: matches the C2·C1·U|0> derivation.
// The dense oracle (which inserts the Paulis at their true mid-circuit positions) is the
// arbiter: if the order convention is wrong, 100 random trials fail loudly.
//
// NOTE: the dense-oracle scaffolding apply_diag_pauli / sample_measurements / the free
// pauli_expectation(CanonicalStabSum, vector<PauliTerm>) lived here but had ZERO non-test
// callers; they now live in cpp/tests/sampler_dense_oracle.hpp (test_sampler.cpp only).

namespace {

// Shared sampler prologue: the ONE place the user circuit is normalized + propagation-classified.
// run_shots_packed calls this so its `deferred` circuit (and thus the
// propagation result) is byte-identical. The sampler-preset normalize() runs coherentize ->
// strip-for-consumer -> defer -> eliminate_hadamards (always last). `coherent` (pre-strip) is the
// feedback-plan input; `deferred` (Hadamard-free) drives the propagation table / bare state.
// The diagonal-Clifford+Pauli propagation-class check is the SOLE circuit-rejection criterion;
// `in_class` carries its verdict and `reject_gate_index` the offending deferred-stream index.
struct SamplerPrologue {
    NormalizeResult nr;          // owns `coherent` and `normalized`
    Circuit deferred;            // == std::move(nr.normalized)
    PropagationTable table;
    bool has_fb = false;
    bool in_class = false;
    int reject_gate_index = -1;
};

SamplerPrologue build_sampler_prologue(const Circuit& user_circuit) {
    SamplerPrologue p;
    NormalizePolicy npol;
    npol.feedback = NormalizePolicy::Feedback::KeepCoherent;
    npol.defer = true;
    npol.want_map = false;
    p.nr = normalize(user_circuit, npol);
    p.has_fb = circuit_has_feedback(p.nr.coherent);
    p.deferred = std::move(p.nr.normalized);
    p.table = build_propagation_table(p.deferred, /*ppr_retry=*/true);
    p.in_class = p.table.all_in_class;
    if (!p.in_class)
        p.reject_gate_index = p.table.rejects.empty() ? -1 : p.table.rejects[0].reject_gate_index;
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// FastPlan — the compile-time read/observable layout the program and its emit tail consume:
// the terminal reads in record order, the declared observable Paulis, and the measurement word
// count. (The legacy shot loop's per-alternative side data, CZ pair universe, cascade plan and
// rotation-pattern machinery lived here until E2b; the framed engine — FramedCircuitShotSampler
// — owns all of that per-shot machinery now, one engine for R >= 0.)
namespace {

struct FastPlan {
    std::vector<std::pair<int, int>> ms;      // terminal reads, record order: (pauli idx 0:X 1:Y 2:Z, qubit)
    std::vector<Pauli> obsP;                  // observable base Paulis (i^{#Y} X^x Z^z)
    int MW = 0;                               // measurement words
};

// Build the observable's Pauli exactly as pauli_expectation does.
Pauli obs_pauli(int n, const std::vector<PauliTerm>& obs) {
    Pauli P(n);
    int ny = 0;
    for (const PauliTerm& t : obs) {
        if (t.p == PauliBasis::X) { P.setx(t.qubit); }
        else if (t.p == PauliBasis::Y) { P.setx(t.qubit); P.setz(t.qubit); ++ny; }
        else { P.setz(t.qubit); }
    }
    P.phase = ny & 3;
    return P;
}

FastPlan build_fast_plan(const Circuit& deferred) {
    FastPlan plan;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure) {
            const int pidx = ins.basis == PauliBasis::X ? 0 : ins.basis == PauliBasis::Y ? 1 : 2;
            plan.ms.push_back({pidx, ins.qubits[0]});
        }
        else if (ins.kind == Instr::Kind::Observable)
            plan.obsP.push_back(obs_pauli(deferred.n, ins.obs));
    plan.MW = ((int)plan.ms.size() + 63) / 64;
    return plan;
}

}  // namespace

// ── SamplerProgram: the compiled (seed/shots-independent) artifact of run_shots_packed ──────────
// Holds everything sample_program reads that the COMPILE phase computes. Built ONCE by
// compile_sampler_program; sample_program runs the seed/shots-dependent shot loop over it:
// the deferred circuit + propagation table, the bare state, the FastPlan layout, the feedback
// plan, the detector/observable layout (masks + CSR + reference parities), record-flip metadata,
// and the framed circuit-level engine — THE shot engine (E2b: one engine, all modes, R >= 0;
// the legacy shot loop, its cascade/pattern machinery and the QEC_FRAMED / QEC_AB_REDUCE /
// QEC_AB_OBS levers are gone).
struct SamplerProgram {
    bool rejected = true;
    int  reject_gate_index = -1;
    int  chi = 0;
    int  n = 0;

    // deferral / propagation / feedback inputs
    Circuit deferred;
    PropagationTable table;
    bool has_fb = false;
    FeedbackPlan fb;

    // bare state (the framed engine compiles from it; kept for supplied-reference introspection)
    FramedSuperposition bare{0};

    // read/observable layout + sizes
    FastPlan plan;
    int M = 0, R = 0, MWp = 0;

    // record-flip metadata (record order)
    bool has_rf = false;
    std::vector<uint8_t> rec_invert;
    std::vector<double>  rec_flip;

    // detector / observable channels
    std::vector<std::vector<int>> detectors, observables;
    int D = 0, O = 0;
    std::vector<uint64_t> dmask, omask;
    std::vector<int> dword_off, dword_idx, oword_off, oword_idx;
    std::vector<uint8_t> refdet, refobs;
    // Per-expectation-column declared byproduct-frame mask (R rows × MWp words, same layout as
    // omask). Row r is a record bitmask: bit j set iff abs record index j is in the frame for
    // column r. Built from each Observable instr's obs_frame in declaration order. Empty iff R==0.
    std::vector<uint64_t> emask;

    // The framed circuit-level shot engine: tier0 TreePlan replay + pattern memo + A/B error
    // shots + the R>0 expectation channels (enable_expectations at compile when R > 0). Non-null
    // for every accepted program. Its pattern memo resets per sample_program call (begin_run) —
    // the same-seed => same-bytes contract.
    std::unique_ptr<FramedCircuitShotSampler> framed;
};

void delete_sampler_program(SamplerProgram* p) { delete p; }

SamplerProgram* compile_sampler_program(const Circuit& user_circuit,
                                        const std::vector<std::vector<int>>& detectors,
                                        const std::vector<std::vector<int>>& observables,
                                        const FramedSuperposition* reference) {
    SamplerProgram* progp = new SamplerProgram();
    SamplerProgram& prog = *progp;
    prog.detectors = detectors;
    prog.observables = observables;
    // Classically-controlled Pauli feedback is handled OFF the state-evolution hot loop as an
    // exact post-sampling triangular record/expectation relabel (see feedback.hpp). The bare
    // circuit = the user circuit with the controlled-Paulis REMOVED; the existing pipeline runs
    // on it UNCHANGED (deferral / propagation / bare state / engine compile — all feedback-free,
    // full speed). For a feedback-free circuit strip_feedback is a copy and feedback.has_feedback
    // is false ⇒ the relabel pass is dead code ⇒ byte-identical streams, same speed.
    // Sampler-preset normalize() + propagation classification run via the SHARED prologue (see
    // build_sampler_prologue), so the `deferred` circuit / propagation result is byte-identical
    // across sampler entry points. `coherent` (pre-strip) is the feedback-plan input; `deferred`
    // (Hadamard-free) drives the propagation table / bare state / framed engine.
    // eliminate_hadamards is physics-preserving; it runs once during this setup.
    SamplerPrologue prologue = build_sampler_prologue(user_circuit);
    const Circuit& coherent = prologue.nr.coherent;
    prog.has_fb = prologue.has_fb;
    const bool has_user_rf = circuit_has_record_flips(user_circuit);
    const Circuit& src = coherent;   // pre-deferral Measure-field source (see record-flip note below)
    prog.deferred = std::move(prologue.deferred);
    prog.table = std::move(prologue.table);
    Circuit& deferred = prog.deferred;
    if (!prologue.in_class) {
        prog.reject_gate_index = prologue.reject_gate_index;
        return progp;
    }
    if (reference) {
        // §4.10 supplied mode: the engine TRUSTS the supplied deferred-space FramedSuperposition.
        if (reference->n() != deferred.n) { prog.reject_gate_index = -1; return progp; }
        // The propagation-class check is the sole rejection criterion, so a class-passing circuit
        // with a trusted reference always builds.
        prog.bare = *reference;             // clone the supplied FramedSuperposition
        prog.bare.U.ensure_dual();          // clones inherit a valid dual (loop recipe)
        prog.chi = reference->chi();
    } else {
        FramedBareState lb = build_bare_state_framed(deferred);
        if (lb.rejected) { prog.reject_gate_index = lb.reject_gate_index; return progp; }
        prog.bare = std::move(lb.state);
        prog.bare.U.ensure_dual();          // clones inherit a valid dual (loop recipe)
        prog.chi = lb.chi;
    }
    prog.n = deferred.n;
    prog.plan = build_fast_plan(deferred);
    FastPlan& plan = prog.plan;
    const int M = (int)plan.ms.size();
    const int R = (int)plan.obsP.size();
    prog.M = M;
    prog.R = R;
    // Per-record flip metadata, in record order. Two independent sources, XOR'd:
    //   (a) the USER's invert/readout_flip_p, which deferral rebuilds terminal Measures from a
    //       (qubit, basis) list and DROPS — read back off `src` (= the coherent, pre-strip circuit;
    //       coherentize/strip_feedback copy Measures verbatim and in order). Deferral never reorders
    //       Measures among themselves, so src's Measure order == terminal-read (== plan.ms == mw bit) order.
    //   (b) the deterministic `invert` that eliminate_hadamards STAMPS onto a terminal Measure when it
    //       absorbs a single-qubit Clifford frame whose sign flips the read (X;M(Z)->M(Z)!, H;M(Z)->M(X)
    //       with a sign). (b) lives ON the deferred Measure (== plan.ms order); (a) does not. Both must
    //       be applied ⇒ the flip pass runs whenever EITHER is present (not just user record flips).
    bool has_elim_inv = false;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure && ins.invert) { has_elim_inv = true; break; }
    const bool has_rf = has_user_rf || has_elim_inv;
    prog.has_rf = has_rf;
    // Off the hot loop: populated/consulted ONLY when has_rf, so a flip-free, Hadamard-free circuit
    // allocates these (trivially) but never reads them, drawing no coin.
    std::vector<uint8_t>& rec_invert = prog.rec_invert;
    std::vector<double>&  rec_flip = prog.rec_flip;
    rec_invert.assign(M, 0);
    rec_flip.assign(M, 0.0);
    if (has_rf) {
        int j = 0;
        for (const Instr& ins : src.stream)
            if (ins.kind == Instr::Kind::Measure) {
                assert(j < M && "record-flip metadata: more user Measures than terminal reads");
                rec_invert[j] = ins.invert ? 1 : 0;
                rec_flip[j] = ins.readout_flip_p;
                ++j;
            }
        assert(j == M && "record-flip metadata: Measure count mismatch with plan.ms");
        // XOR in the eliminate_hadamards-stamped invert (deferred Measure order == plan.ms order).
        j = 0;
        for (const Instr& ins : deferred.stream)
            if (ins.kind == Instr::Kind::Measure) { if (ins.invert) rec_invert[j] ^= 1; ++j; }
    }
    const int D = (int)detectors.size();
    const int O = (int)observables.size();
    prog.D = D;
    prog.O = O;
    // Feedback relabel plan (built once; records/observables align 1:1 with the bare pipeline).
    prog.fb = build_feedback_plan(coherent, M, R);
    FeedbackPlan& fb = prog.fb;
    if (!fb.ok) { prog.reject_gate_index = fb.reject_gate_index; return progp; }
    // detector / observable parity masks over the measurement words (parity = popcount(AND))
    const int MWp = plan.MW + 1;
    prog.MWp = MWp;
    auto build_masks = [&](const std::vector<std::vector<int>>& defs) {
        std::vector<uint64_t> m((size_t)defs.size() * MWp, 0);
        for (size_t d = 0; d < defs.size(); ++d)
            for (int idx : defs[d]) m[(size_t)d * MWp + (idx >> 6)] ^= 1ull << (idx & 63);
        return m;
    };
    prog.dmask = build_masks(detectors);
    prog.omask = build_masks(observables);
    // Build emask: one row per PAULI_EXPECTATION column (declaration order = obsP order), using
    // each Observable instr's obs_frame (absolute measurement indices). No reference subtraction —
    // the fold is a pure declared XOR; the engine already outcome-conditions <P>.
    {
        std::vector<std::vector<int>> eframes;
        bool any_frame = false;
        for (const Instr& ins : deferred.stream)
            if (ins.kind == Instr::Kind::Observable) {
                eframes.push_back(ins.obs_frame);
                if (!ins.obs_frame.empty()) any_frame = true;
            }
        // Only build (and keep) emask when at least one column declares a frame.
        // For fully-frameless circuits emask stays empty → the per-shot fold is skipped.
        if (any_frame)
            prog.emask = build_masks(eframes);
        // else prog.emask remains empty (default-constructed); the guard !prog.emask.empty()
        // in pack_shot_records will be false and the fold loop is never entered.
    }
    const std::vector<uint64_t>& dmask = prog.dmask;
    const std::vector<uint64_t>& omask = prog.omask;
    // Per-def CSR list of NONZERO mask words (built once). Detector/observable masks are sparse
    // (a detector usually spans a handful of measurements ⇒ 1–2 nonzero words even when MW is
    // large), so the per-shot parity scans only the populated words instead of all MW+1. Zero
    // words contribute 0 to popcount ⇒ byte-identical parity.
    auto build_word_csr = [&](const std::vector<uint64_t>& mask, size_t count,
                              std::vector<int>& off, std::vector<int>& idx) {
        off.assign(count + 1, 0);
        for (size_t d = 0; d < count; ++d) {
            int c = 0;
            for (int w = 0; w <= plan.MW; ++w) if (mask[d * MWp + w]) ++c;
            off[d + 1] = off[d] + c;
        }
        idx.assign((size_t)off[count], 0);
        for (size_t d = 0; d < count; ++d) {
            int p = off[d];
            for (int w = 0; w <= plan.MW; ++w) if (mask[d * MWp + w]) idx[p++] = w;
        }
    };
    build_word_csr(dmask, detectors.size(), prog.dword_off, prog.dword_idx);
    build_word_csr(omask, observables.size(), prog.oword_off, prog.oword_idx);
    // The bare lean state (prog.bare) is sourced directly above — from the supplied reference
    // FramedSuperposition, or from build_bare_state_framed — with its dual already ensured.
    const FramedSuperposition& bare = prog.bare;
    // Detector/observable channels are REFERENCE-RELATIVE, exactly Stim's semantics: the
    // emitted bit is parity(shot) XOR parity(noiseless reference) — a detection EVENT / logical
    // FLIP, not the raw parity. For a deterministic channel any noiseless sample gives the same
    // parity; for a RANDOM one (a Born observable, a gauge detector) the reference is a
    // CONVENTION, and it is Stim's: Circuit.reference_sample() — noise off, every collapse biased
    // toward its +1 eigenvalue (the user's record before `!`), in record order, feedback applied.
    // framed_reference_records implements exactly that on the bare state (shared with the twirl
    // engine's Born observable, so engine="auto" never changes the observable bit's meaning);
    // the bias is mapped into the deferred read frame here:
    //   raw user record_k = b_k ^ elim_k ^ flip_k     (b_k: deferred read bit; elim_k: the sign
    //   eliminate_hadamards stamped on read k; flip_k: the feedback relabel's flip of record k
    //   given the reference's earlier records)  =>  want(b_k) = elim_k ^ flip_k.
    // (Through 3.1.9 the reference was batch_measure at u=0: deferred read bit 0, in batch order,
    // so a stamped sign — e.g. `RX 0; MY 0` — or a magic-coupled read collapsed out of order gave
    // a random channel the opposite constant to Stim's. Audit finding K.)
    // The recorded reference bit is R_k = b_k ^ rec_invert_k ^ flip_k: the deterministic `!`
    // invert is part of every shot's record (and the reference's), so a detector/observable with
    // an even count of inverted records cancels it, exactly as Stim. The probabilistic M(p) coin is
    // per-shot, NOT in the reference. The feedback relabel reads the RECORDED control bit (invert
    // included) — same order as pack_shot_records (invert first, feedback second).
    std::vector<uint64_t> refmw((size_t)MWp, 0);
    if (M > 0) {
        const std::vector<uint8_t> R = noiseless_reference_records(
            bare, deferred, plan.ms, has_rf ? rec_invert : std::vector<uint8_t>{},
            prog.has_fb ? &fb : nullptr);
        for (int j = 0; j < M; ++j)
            if (R[(size_t)j]) refmw[(size_t)(j >> 6)] |= 1ull << (j & 63);
    }
    auto ref_parity = [&](const std::vector<uint64_t>& masks, int idx) {
        int par = 0;
        for (int w = 0; w < MWp; ++w)
            par += __builtin_popcountll(masks[(size_t)idx * MWp + w] & refmw[w]);
        return par & 1;
    };
    prog.refdet.assign(D, 0);
    prog.refobs.assign(O, 0);
    for (int d = 0; d < D; ++d) prog.refdet[d] = (uint8_t)ref_parity(dmask, d);
    for (int o = 0; o < O; ++o) prog.refobs[o] = (uint8_t)ref_parity(omask, o);

    // ── THE shot engine (E2b): the framed circuit-level sampler, unconditional, R >= 0 ────────
    // Since Stage G3 the engine is chi-GENERAL (tier0 base outcomes replay the TreePlan DAG at
    // any chi, over the node cap they come from live-block sampling; pattern misses take the A/B
    // error shot). R > 0 enables the per-plan expectation channels (leaf posts + coin corrections
    // + tensor-split observables — see FramedObsChannel); every shot then fills
    // last_expectations(). The try/catch is the out-of-domain guard: construction throws on
    // factorize structural invariants — with no legacy loop left, that is a LOUD program
    // rejection (reject_gate_index = -1), not a silent alternative.
    try {
        prog.framed = std::make_unique<FramedCircuitShotSampler>(
            prog.bare, prog.deferred, plan.ms, /*noise_seed=*/0, &prog.table);
        if (R > 0 && !prog.framed->enable_expectations(plan.obsP, prog.bare))
            throw std::runtime_error("expectation channel refused");
    } catch (const BareStateCapacityError&) {
        throw;                         // a state-budget refusal is NOT an out-of-class rejection
    } catch (const std::exception&) {
        prog.framed.reset();
        prog.reject_gate_index = -1;
        return progp;
    }

    prog.rejected = false;
    return progp;
}

// Shared mw->records packing tail (sample_program + sample_program_with_source). Takes the
// shot's populated measurement words `mw` and produces every downstream channel: record-flip
// pass (deterministic `!` XOR + the M(p) coin from flip_rng — one flip_uni draw per
// rec_flip[j] > 0, Stim ordering: BEFORE feedback so feed-forward reads the noisy record),
// classically-controlled Pauli feedback in record/stream order (mflips XOR; esign expectation
// sign flips only when `exp_row` is non-null — sample_program passes the shot's expectation row
// at R > 0, the with_source call site passes nullptr), then the packed-row output (measurements
// memcpy verbatim, detector/observable channels as mask parities over the measurement words).
static void pack_shot_records(
        const SamplerProgram& prog, uint64_t* mw, int gs, PackedRecords& out,
        std::mt19937_64& flip_rng, std::uniform_real_distribution<double>& flip_uni,
        double* exp_row) {
    const int M = prog.M, MWp = prog.MWp, D = prog.D, O = prog.O, R = prog.R;
    const int MB = PackedRecords::bytes_per_shot(M);
    const int DB = PackedRecords::bytes_per_shot(D);
    const int OB = PackedRecords::bytes_per_shot(O);
    if (prog.has_rf) {
        for (int j = 0; j < M; ++j) {
            uint8_t flip = prog.rec_invert[j];
            if (prog.rec_flip[j] > 0.0 && flip_uni(flip_rng) < prog.rec_flip[j]) flip ^= 1;
            if (flip) mw[j >> 6] ^= 1ull << (j & 63);
        }
    }
    if (prog.has_fb) {
        for (const FeedbackOp& op : prog.fb.ops) {
            const int c = op.control_record;
            if (!((mw[c >> 6] >> (c & 63)) & 1)) continue;
            for (int w = 0; w < prog.fb.MW; ++w) mw[w] ^= op.mflips[w];
            if (exp_row && op.esign)
                for (int r = 0; r < R; ++r)
                    if ((op.esign >> r) & 1) exp_row[r] = -exp_row[r];
        }
    }
    // Declared PAULI_EXPECTATION frame: XOR the parity of each column's declared record mask
    // into that column's sign. Cheap (one popcount-parity per column), reusing mw already in
    // hand. No reference subtraction — the frame is a pure declared XOR of the record parity.
    if (exp_row && !prog.emask.empty()) {
        const int EW = MWp;                         // words per column mask == record words
        for (int r = 0; r < R; ++r) {
            int par = 0;
            for (int w = 0; w < EW; ++w)
                par ^= __builtin_popcountll(prog.emask[(size_t)r * EW + w] & mw[w]) & 1;
            if (par) exp_row[r] = -exp_row[r];
        }
    }
    std::memcpy(out.measurements.data() + (size_t)gs * MB, mw, MB);
    uint8_t* drow = out.detectors.data() + (size_t)gs * DB;
    for (int d = 0; d < D; ++d) {
        int par = 0;
        const uint64_t* dm = prog.dmask.data() + (size_t)d * MWp;
        for (int p = prog.dword_off[d]; p < prog.dword_off[d + 1]; ++p) {
            int w = prog.dword_idx[p]; par += __builtin_popcountll(dm[w] & mw[w]);
        }
        drow[d >> 3] = (uint8_t)(drow[d >> 3] | (((par ^ prog.refdet[d]) & 1) << (d & 7)));
    }
    uint8_t* orow = out.observables.data() + (size_t)gs * OB;
    for (int o = 0; o < O; ++o) {
        int par = 0;
        const uint64_t* om = prog.omask.data() + (size_t)o * MWp;
        for (int p = prog.oword_off[o]; p < prog.oword_off[o + 1]; ++p) {
            int w = prog.oword_idx[p]; par += __builtin_popcountll(om[w] & mw[w]);
        }
        orow[o >> 3] = (uint8_t)(orow[o >> 3] | (((par ^ prog.refobs[o]) & 1) << (o & 7)));
    }
}

PackedRecords sample_program(SamplerProgram& prog, int shots, uint64_t seed) {
    PackedRecords out;
    if (shots < 0) shots = 0;
    if (prog.rejected || !prog.framed) {
        // Match the monolithic run_shots_packed's reject fields: reject_gate_index always, and
        // chi == bare.chi for the feedback-plan reject (the only reject path that ran after the
        // bare state was built; the earlier rejects leave chi == 0). prog.chi carries exactly that.
        out.reject_gate_index = prog.reject_gate_index;
        out.chi = prog.chi;
        return out;
    }
    const int M = prog.M, R = prog.R, MWp = prog.MWp, D = prog.D, O = prog.O;

    out.chi = prog.chi;
    out.shots = shots;
    out.num_measurements = M;
    out.num_detectors = D;
    out.num_observables = O;
    out.num_expectations = R;
    const int MB = PackedRecords::bytes_per_shot(M);
    const int DB = PackedRecords::bytes_per_shot(D);
    const int OB = PackedRecords::bytes_per_shot(O);
    out.measurements.assign((size_t)shots * MB, 0);
    out.detectors.assign((size_t)shots * DB, 0);
    out.observables.assign((size_t)shots * OB, 0);
    out.expectations.assign((size_t)shots * R, 0.0);

    // golden-ratio odd constant (SplitMix64 mixing): derive an independent readout-flip
    // coin stream from `seed`, decorrelated from the main/noise RNG.
    std::mt19937_64 flip_rng(seed ^ 0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<double> flip_uni(0.0, 1.0);
    std::mt19937_64 rng(seed);                       // measurement uniforms, record order
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // ── THE engine (E2b): the framed circuit-level shot sampler, R >= 0 ────────────────────────
    // The engine samples the raw measurement records (tier0 TreePlan replay / pattern-memoized
    // plans / A/B error shots — distribution-validated against the retained full-state fallback,
    // Stim, and a dense statevector oracle) and, at R > 0, fills the per-shot expectation row
    // (the compile-time expectation channels — record-conditioned EXACT vs the dense law in
    // test_framed_r). The record-flip / feedback / packing tail is the shared pack_shot_records
    // helper (one source with sample_program_with_source); op.esign applies the feedback
    // expectation sign flips to the row exactly as the retired legacy call site did.
    FramedCircuitShotSampler& fcs = *prog.framed;
    fcs.begin_run(seed ^ 0x9E3779B97F4A7C15ull);   // per-call memo reset: same-seed => same bytes
    const std::function<double()> rng01 = [&rng, &uni]() { return uni(rng); };
    const bool exp_mode = R > 0 && fcs.expectations_enabled();
    std::vector<int> fout;
    std::vector<uint64_t> mw((size_t)MWp, 0);
    for (int gs = 0; gs < shots; ++gs) {
        fcs.next_shot(rng01, fout);
        for (int w = 0; w < MWp; ++w) mw[w] = 0;
        for (int j = 0; j < M; ++j)   // branchless: the outcome bits are unpredictable
            mw[j >> 6] |= (uint64_t)(fout[j] == -1) << (j & 63);
        double* exp_row = nullptr;
        if (exp_mode) {
            exp_row = out.expectations.data() + (size_t)gs * R;
            const std::vector<double>& ex = fcs.last_expectations();
            for (int r = 0; r < R; ++r) exp_row[r] = ex[r];
        }
        // op.esign is expectation-only; exp_row is null on the R == 0 path.
        pack_shot_records(prog, mw.data(), gs, out, flip_rng, flip_uni, exp_row);
    }
    out.rejected = false;
    return out;
}


// Generic seams (see sampler.hpp): the framed packing tail over a caller-supplied shot source,
// and the compiled program's framed engine. The per-shot record-flip / feedback / detector /
// observable tail is the shared pack_shot_records helper (same as the framed branch of
// sample_program above) — only the shot source differs.
FramedCircuitShotSampler* sampler_program_framed(SamplerProgram& prog) {
    return prog.framed.get();
}

PackedRecords sample_program_with_source(
    SamplerProgram& prog, int shots, uint64_t seed,
    const std::function<void(const std::function<double()>&, std::vector<int>&)>& shot_source) {
    PackedRecords out;
    if (prog.rejected || !prog.framed || shots < 0) return out;
    const int M = prog.M, MWp = prog.MWp, D = prog.D, O = prog.O;
    const int MB = PackedRecords::bytes_per_shot(M);
    const int DB = PackedRecords::bytes_per_shot(D);
    const int OB = PackedRecords::bytes_per_shot(O);
    out.chi = prog.chi;
    out.shots = shots;
    out.num_measurements = M;
    out.num_detectors = D;
    out.num_observables = O;
    out.num_expectations = 0;
    out.measurements.assign((size_t)shots * MB, 0);
    out.detectors.assign((size_t)shots * DB, 0);
    out.observables.assign((size_t)shots * OB, 0);

    std::mt19937_64 flip_rng(seed ^ 0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<double> flip_uni(0.0, 1.0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::function<double()> rng01 = [&rng, &uni]() { return uni(rng); };

    std::vector<int> fout;
    std::vector<uint64_t> mw((size_t)MWp, 0);
    for (int gs = 0; gs < shots; ++gs) {
        shot_source(rng01, fout);
        for (int w = 0; w < MWp; ++w) mw[w] = 0;
        for (int j = 0; j < M; ++j)   // branchless: the outcome bits are unpredictable
            mw[j >> 6] |= (uint64_t)(fout[j] == -1) << (j & 63);
        pack_shot_records(prog, mw.data(), gs, out, flip_rng, flip_uni, nullptr);
    }
    out.rejected = false;
    return out;
}

// Thin byte-identical wrapper: compile once, sample once. The split is purely structural, so this
// produces exactly the same PackedRecords a monolithic loop would. The stateful CompiledProgram
// binding (compile_sampler_program + sample_program) is the path that skips per-call recompile.
PackedRecords run_shots_packed(const Circuit& user_circuit, int shots, uint64_t seed,
                               const std::vector<std::vector<int>>& detectors,
                               const std::vector<std::vector<int>>& observables,
                               const FramedSuperposition* reference) {
    std::unique_ptr<SamplerProgram, void (*)(SamplerProgram*)> prog(
        compile_sampler_program(user_circuit, detectors, observables, reference),
        &delete_sampler_program);
    return sample_program(*prog, shots, seed);
}

// ── Forced faults: exact branch enumeration (see sampler.hpp) ────────────────────────────────
const Circuit& sampler_program_deferred(const SamplerProgram& prog) { return prog.deferred; }
bool sampler_program_rejected(const SamplerProgram& prog) { return prog.rejected || !prog.framed; }
int sampler_program_reject_gate_index(const SamplerProgram& prog) { return prog.reject_gate_index; }

namespace {

constexpr double kBranchEps = 1e-12;   // the engine's coin threshold (framed_reference_read)

struct ExactBranchWalk {
    const SamplerProgram& prog;
    const std::vector<Pauli>& reads;       // E^dag M_k E, record order
    const std::vector<Pauli>& exps;        // E^dag P_r E, requested columns
    const std::vector<uint64_t>& forced;   // forced record-flip words (MWp)
    const std::vector<std::vector<int>>& decisions;
    const std::vector<int>& exp_cols;      // requested column ids (for the frame / esign tail)
    long max_branches;
    ExactBranchResult& out;
    std::vector<int> bits;                 // raw read outcome bits (1 = -1), current path

    // the deterministic record tail of one leaf (pack_shot_records with every coin forced)
    void leaf(const FramedSuperposition& L, double p) {
        const int M = prog.M, MWp = prog.MWp, R = (int)exp_cols.size();
        if ((long)out.prob.size() >= max_branches)
            throw std::length_error("exact_branches: more than max_branches branches");
        std::vector<uint64_t> mw((size_t)MWp, 0);
        for (int j = 0; j < M; ++j) mw[(size_t)(j >> 6)] |= (uint64_t)bits[(size_t)j] << (j & 63);
        std::vector<double> ex((size_t)R, 0.0);
        for (int r = 0; r < R; ++r) ex[(size_t)r] = L.expectation(exps[(size_t)r]);
        if (prog.has_rf)
            for (int j = 0; j < M; ++j)
                if (prog.rec_invert[(size_t)j]) mw[(size_t)(j >> 6)] ^= 1ull << (j & 63);
        for (int w = 0; w < MWp; ++w) mw[(size_t)w] ^= forced[(size_t)w];   // the forced M(p) coins
        if (prog.has_fb) {
            for (const FeedbackOp& op : prog.fb.ops) {
                const int c = op.control_record;
                if (!((mw[(size_t)(c >> 6)] >> (c & 63)) & 1)) continue;
                for (int w = 0; w < prog.fb.MW; ++w) mw[(size_t)w] ^= op.mflips[(size_t)w];
                if (op.esign)
                    for (int r = 0; r < R; ++r)
                        if ((op.esign >> exp_cols[(size_t)r]) & 1) ex[(size_t)r] = -ex[(size_t)r];
            }
        }
        if (!prog.emask.empty())
            for (int r = 0; r < R; ++r) {
                int par = 0;
                const uint64_t* em = prog.emask.data() + (size_t)exp_cols[(size_t)r] * MWp;
                for (int w = 0; w < MWp; ++w) par ^= __builtin_popcountll(em[w] & mw[(size_t)w]) & 1;
                if (par) ex[(size_t)r] = -ex[(size_t)r];
            }
        out.prob.push_back(p);
        for (int j = 0; j < M; ++j) out.meas.push_back((uint8_t)((mw[(size_t)(j >> 6)] >> (j & 63)) & 1));
        auto parity = [&](const uint64_t* m) {
            int par = 0;
            for (int w = 0; w < MWp; ++w) par += __builtin_popcountll(m[w] & mw[(size_t)w]);
            return par & 1;
        };
        for (int d = 0; d < prog.D; ++d)
            out.det.push_back((uint8_t)(parity(prog.dmask.data() + (size_t)d * MWp) ^ prog.refdet[(size_t)d]));
        for (int o = 0; o < prog.O; ++o)
            out.obs.push_back((uint8_t)(parity(prog.omask.data() + (size_t)o * MWp) ^ prog.refobs[(size_t)o]));
        for (const auto& dl : decisions) {
            int par = 0;
            for (int j : dl) par ^= (int)((mw[(size_t)(j >> 6)] >> (j & 63)) & 1);
            out.dec.push_back((uint8_t)par);
        }
        for (double v : ex) out.exps.push_back(v);
    }

    // resolve read k onward on L (owned by this frame; mutated)
    void walk(FramedSuperposition& L, int k, double p) {
        const int M = prog.M;
        while (k < M) {
            const Pauli& Q = reads[(size_t)k];
            const double pp = L.born_p1(Q);
            const double pm = 1.0 - pp;
            if (pp > kBranchEps && pm > kBranchEps) {
                FramedSuperposition Lm = L;                       // the -1 branch
                L.measure_pauli(Q, 0.0);                           // u = 0 < pp  => +1
                bits[(size_t)k] = 0;
                walk(L, k + 1, p * pp);
                Lm.measure_pauli(Q, std::nextafter(1.0, 0.0));     // u >= pp     => -1
                bits[(size_t)k] = 1;
                walk(Lm, k + 1, p * pm);
                bits[(size_t)k] = 0;
                return;
            }
            const int o = pp > kBranchEps ? +1 : -1;               // deterministic read
            L.measure_pauli(Q, o == +1 ? 0.0 : std::nextafter(1.0, 0.0));
            bits[(size_t)k] = o == -1 ? 1 : 0;
            ++k;
        }
        leaf(L, p);
    }
};

}  // namespace

ExactBranchResult sample_program_exact_branches(const SamplerProgram& prog,
                                                std::vector<FiredPauli> fired,
                                                const std::vector<int>& record_flips,
                                                const std::vector<std::vector<int>>& decisions,
                                                const std::vector<int>& exp_cols_in,
                                                long max_branches) {
    ExactBranchResult out;
    if (prog.rejected || !prog.framed) { out.error = "rejected program"; return out; }
    const int M = prog.M, n = prog.n;
    std::vector<int> exp_cols = exp_cols_in;
    if (exp_cols.empty()) for (int r = 0; r < prog.R; ++r) exp_cols.push_back(r);
    for (int r : exp_cols)
        if (r < 0 || r >= prog.R) { out.error = "expectation column out of range"; return out; }
    for (const auto& dl : decisions)
        for (int j : dl)
            if (j < 0 || j >= M) { out.error = "decision record out of range"; return out; }
    std::vector<uint64_t> forced((size_t)prog.MWp, 0);
    for (int j : record_flips) {
        if (j < 0 || j >= M) { out.error = "record flip out of range"; return out; }
        forced[(size_t)(j >> 6)] ^= 1ull << (j & 63);
    }
    // compose_fired's order: stream order, X atom then Z atom per factor (stable within a location)
    std::stable_sort(fired.begin(), fired.end(), [](const FiredPauli& a, const FiredPauli& b) {
        return a.location_index < b.location_index;
    });
    std::vector<const PropResult*> atoms;
    try {
        for (const FiredPauli& f : fired) {
            const auto [loc, qi] = prog.table.find_slot(f.location_index, f.qubit, "exact_branches");
            if (f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) atoms.push_back(&loc->x_atom[(size_t)qi]);
            if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) atoms.push_back(&loc->z_atom[(size_t)qi]);
        }
    } catch (const std::exception& e) { out.error = e.what(); return out; }
    for (const PropResult* a : atoms)
        if (a->rejected) { out.error = "a fault's propagated error left the simulable class"; return out; }
    // Conjugate every read / expectation Pauli through the fired atoms, LAST-fired first:
    // M <- E_first^dag ... E_last^dag M E_last ... E_first = E^dag M E (process_shot_general_'s
    // convention). A diagonal atom D = X^v·∏CZ·diag(S^a) is applied by the canonical
    // diag_conjugate rules (X^v innermost, as conj_by_diag_altside_); a CH-class atom by its
    // stored general tableau (dual_image).
    struct DiagAtom {
        const CliffordTableau* gen = nullptr;
        std::vector<int> v;
        std::vector<std::pair<int, int>> a;      // (qubit, S-power)
        std::vector<std::pair<int, int>> cz;     // (qa < qb)
    };
    std::vector<DiagAtom> da(atoms.size());
    for (size_t i = 0; i < atoms.size(); ++i) {
        const PropResult& at = *atoms[i];
        if (at.general) { da[i].gen = at.general.get(); continue; }
        const DiagPauliClifford& D = at.c_prop;
        for (int q = 0; q < D.n; ++q) {
            if (D.v[(size_t)q]) da[i].v.push_back(q);
            if (D.a[(size_t)q] & 3) da[i].a.push_back({q, D.a[(size_t)q] & 3});
        }
        if (!D.B.zero_unmaterialized())
            for (int q = 0; q < D.n; ++q) {
                const uint64_t* row = D.B.row(q);
                for (int w = 0; w < D.B.words(); ++w) {
                    uint64_t bits = row[w];
                    while (bits) {
                        const int r = (w << 6) + __builtin_ctzll(bits);
                        bits &= bits - 1;
                        if (r > q) da[i].cz.push_back({q, r});
                    }
                }
            }
    }
    std::vector<uint64_t> zmask;
    auto conj = [&](Pauli& P) {
        for (size_t e = da.size(); e-- > 0;) {
            const DiagAtom& d = da[e];
            if (d.gen) { P = d.gen->dual_image(P); continue; }
            int dphase = 0;
            for (int q : d.v) if (P.zbit(q)) dphase += 2;
            if (d.a.empty() && d.cz.empty()) { P.phase = (P.phase + dphase) & 3; continue; }
            zmask.assign(P.z.size(), 0);
            for (const auto& qa : d.a) diag_spow_conjugate(P, qa.first, qa.second, zmask.data(), dphase);
            for (const auto& pr : d.cz) diag_cz_conjugate(P, pr.first, pr.second, zmask.data(), dphase);
            for (size_t w = 0; w < P.z.size(); ++w) P.z[w] ^= zmask[w];
            P.phase = (P.phase + (dphase & 3)) & 3;
        }
    };
    std::vector<Pauli> reads, exps;
    reads.reserve((size_t)M);
    for (int k = 0; k < M; ++k) reads.push_back(single_pauli(prog.plan.ms[(size_t)k].first, prog.plan.ms[(size_t)k].second, n));
    for (int r : exp_cols) exps.push_back(prog.plan.obsP[(size_t)r]);
    if (!da.empty()) {
        for (Pauli& P : reads) conj(P);
        for (Pauli& P : exps) {
            conj(P);
            if (((P.phase - P.xz_overlap()) & 1) != 0) { out.error = "internal: non-Hermitian conjugated expectation"; return out; }
        }
    }
    out.M = M; out.D = prog.D; out.O = prog.O; out.K = (int)decisions.size(); out.R = (int)exp_cols.size();
    ExactBranchWalk w{prog, reads, exps, forced, decisions, exp_cols, max_branches, out,
                      std::vector<int>((size_t)M, 0)};
    try {
        FramedSuperposition L = prog.bare;
        w.walk(L, 0, 1.0);
    } catch (const std::length_error& e) {
        ExactBranchResult bad; bad.error = e.what(); return bad;
    }
    double tot = 0.0;
    for (double p : out.prob) tot += p;
    if (std::fabs(tot - 1.0) > 1e-9) {
        ExactBranchResult bad;
        bad.error = "internal: branch probabilities sum to " + std::to_string(tot);
        return bad;
    }
    return out;
}

SampleResult run_shots(const Circuit& user_circuit, int shots, uint64_t seed,
                       const FramedSuperposition* reference) {
    // Per-shot view over the packed-native sampler (kept for tests and small interactive runs;
    // the record contract is PackedRecords).
    PackedRecords pk = run_shots_packed(user_circuit, shots, seed, {}, {}, reference);
    SampleResult out;
    out.rejected = pk.rejected;
    out.reject_gate_index = pk.reject_gate_index;
    out.chi = pk.chi;
    if (pk.rejected) return out;
    out.shots.resize(pk.shots);
    for (int s = 0; s < pk.shots; ++s) {
        ShotRecord& rec = out.shots[s];
        rec.bits.resize(pk.num_measurements);
        for (int j = 0; j < pk.num_measurements; ++j) rec.bits[j] = (uint8_t)pk.meas_bit(s, j);
        rec.expectations.assign(pk.expectations.begin() + (size_t)s * pk.num_expectations,
                                pk.expectations.begin() + (size_t)(s + 1) * pk.num_expectations);
    }
    return out;
}

}  // namespace qeccore
