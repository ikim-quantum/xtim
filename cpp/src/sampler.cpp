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
#include "qeccore/dem_pe_per_shot.hpp"
#include <limits>
#include <mutex>
#include <unordered_map>
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
            const PropAtom& ax = loc->atom(qi, PauliBasis::X, "compose_fired");
            assert(!ax.rejected);
            c = c.then(ax.c_prop);
        }
        if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) {
            const PropAtom& az = loc->atom(qi, PauliBasis::Z, "compose_fired");
            assert(!az.rejected);
            c = c.then(az.c_prop);
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

SamplerPrologue build_sampler_prologue(const Circuit& user_circuit, AtomSet atoms) {
    SamplerPrologue p;
    NormalizePolicy npol;
    npol.feedback = NormalizePolicy::Feedback::KeepCoherent;
    npol.defer = true;
    npol.want_map = false;
    p.nr = normalize(user_circuit, npol);
    p.has_fb = circuit_has_feedback(p.nr.coherent);
    p.deferred = std::move(p.nr.normalized);
    p.table = build_propagation_table(p.deferred, /*ppr_retry=*/true, atoms);
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
                                        const FramedSuperposition* reference,
                                        AtomSet atoms) {
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
    SamplerPrologue prologue = build_sampler_prologue(user_circuit, atoms);
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
    return sample_program_streamed(prog, shots, seed, nullptr);
}

PackedRecords sample_program_streamed(SamplerProgram& prog, int shots, uint64_t seed,
                                      const ShotHook* hook) {
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
        if (hook && !(*hook)(gs, out)) {   // early stop: keep shots [0, gs] (a prefix)
            const int ran = gs + 1;
            out.shots = ran;
            out.measurements.resize((size_t)ran * MB);
            out.detectors.resize((size_t)ran * DB);
            out.observables.resize((size_t)ran * OB);
            out.expectations.resize((size_t)ran * R);
            break;
        }
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
std::string sampler_program_fault_refusal(const SamplerProgram& prog, const FiredPauli& f) {
    const auto [loc, qi] = prog.table.find_slot(f.location_index, f.qubit, "forced fault");
    for (PauliBasis b : {PauliBasis::X, PauliBasis::Z}) {
        if (f.pauli != b && f.pauli != PauliBasis::Y) continue;
        const PropAtom& a = b == PauliBasis::X ? loc->x_atom[(size_t)qi] : loc->z_atom[(size_t)qi];
        if (a.present) continue;
        return std::string("has probability zero in its channel") +
               (a.rejected ? " and its propagated error leaves the simulable class (deferred-stream gate index " +
                                 std::to_string(a.reject_gate_index) + "), so it cannot be forced here"
                           : ", so no atom is built for it");
    }
    return "";
}

// The world of one fixed fired set: every terminal read and requested expectation Pauli
// conjugated through the fired atoms (E^dag P E), plus the forced record-flip words. Returns ""
// or the refusal.
static std::string forced_world(const SamplerProgram& prog, std::vector<FiredPauli> fired,
                                const std::vector<int>& record_flips,
                                const std::vector<std::vector<int>>& decisions,
                                std::vector<int>& exp_cols, std::vector<Pauli>& reads,
                                std::vector<Pauli>& exps, std::vector<uint64_t>& forced,
                                bool all_if_empty) {
    if (prog.rejected || !prog.framed) return "rejected program";
    const int M = prog.M, n = prog.n;
    if (exp_cols.empty() && all_if_empty) for (int r = 0; r < prog.R; ++r) exp_cols.push_back(r);
    for (int r : exp_cols)
        if (r < 0 || r >= prog.R) return "expectation column out of range";
    for (const auto& dl : decisions)
        for (int j : dl)
            if (j < 0 || j >= M) return "decision record out of range";
    forced.assign((size_t)prog.MWp, 0);
    for (int j : record_flips) {
        if (j < 0 || j >= M) return "record flip out of range";
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
            if (f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) atoms.push_back(&loc->atom(qi, PauliBasis::X, "exact_branches"));
            if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) atoms.push_back(&loc->atom(qi, PauliBasis::Z, "exact_branches"));
        }
    } catch (const std::exception& e) { return e.what(); }
    for (const PropResult* a : atoms)
        if (a->rejected) return "a fault's propagated error left the simulable class";
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
    reads.clear();
    exps.clear();
    reads.reserve((size_t)M);
    for (int k = 0; k < M; ++k) reads.push_back(single_pauli(prog.plan.ms[(size_t)k].first, prog.plan.ms[(size_t)k].second, n));
    for (int r : exp_cols) exps.push_back(prog.plan.obsP[(size_t)r]);
    if (!da.empty()) {
        for (Pauli& P : reads) conj(P);
        for (Pauli& P : exps) {
            conj(P);
            if (((P.phase - P.xz_overlap()) & 1) != 0) return "internal: non-Hermitian conjugated expectation";
        }
    }
    return "";
}

// ── Forced faults: the exact factored law (see sampler.hpp) ──────────────────────────────────
struct ExactLawCache {
    std::mutex mu;
    bool built = false;
    int MW = 0, T = 0, R = 0;
    std::vector<std::vector<int>> decisions;   // the decision lists the forms were built for
    std::vector<uint64_t> S;                   // T x MW: target t = ref_t + S_t . r0
    std::vector<uint8_t> ref;                  // T
    std::vector<uint64_t> U;                   // R x MW: column r's sign exponent = U_r . r0
    std::unique_ptr<PeShotAnalyzer> pe;
};
ExactLawCache* new_exact_law_cache(const SamplerProgram&) { return new ExactLawCache(); }
void delete_exact_law_cache(ExactLawCache* c) { delete c; }

namespace {
inline bool lb_bit(const uint64_t* v, int i) { return (v[i >> 6] >> (i & 63)) & 1; }
inline void lb_xor(uint64_t* a, const uint64_t* b, int W) { for (int w = 0; w < W; ++w) a[w] ^= b[w]; }
inline bool lb_any(const uint64_t* v, int W) { for (int w = 0; w < W; ++w) if (v[w]) return true; return false; }
inline int lb_low(const uint64_t* v, int W) {
    for (int w = 0; w < W; ++w) if (v[w]) return w * 64 + __builtin_ctzll(v[w]);
    return -1;
}
inline int lb_par(const uint64_t* a, const uint64_t* b, int W) {
    int p = 0;
    for (int w = 0; w < W; ++w) p ^= __builtin_popcountll(a[w] & b[w]) & 1;
    return p;
}

// The program's record tail as LINEAR forms in r0 = raw reads XOR `!` inverts XOR forced flips:
// the feedback relabel XORs a constant flip mask when a control record (itself linear in r0) is
// 1, so every recorded bit, target and expectation sign stays linear in r0 (built once).
void build_law_forms(const SamplerProgram& prog, const std::vector<std::vector<int>>& decisions,
                     ExactLawCache& c) {
    const int M = prog.M, MW = prog.MWp;
    c.MW = MW;
    c.R = prog.R;
    c.decisions = decisions;
    const int K = (int)decisions.size();
    c.T = prog.D + prog.O + K;
    std::vector<uint64_t> rec((size_t)M * MW, 0);
    for (int j = 0; j < M; ++j) rec[(size_t)j * MW + (j >> 6)] |= 1ull << (j & 63);
    c.U.assign((size_t)c.R * MW, 0);
    std::vector<uint64_t> ctrl((size_t)MW);
    if (prog.has_fb) {
        for (const FeedbackOp& op : prog.fb.ops) {
            const int cr = op.control_record;
            std::copy(rec.begin() + (size_t)cr * MW, rec.begin() + (size_t)(cr + 1) * MW, ctrl.begin());
            for (int w = 0; w < prog.fb.MW && w < MW; ++w) {
                uint64_t bits = op.mflips[(size_t)w];
                while (bits) {
                    const int j = w * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    if (j < M) lb_xor(rec.data() + (size_t)j * MW, ctrl.data(), MW);
                }
            }
            if (op.esign)
                for (int r = 0; r < c.R && r < 64; ++r)
                    if ((op.esign >> r) & 1) lb_xor(c.U.data() + (size_t)r * MW, ctrl.data(), MW);
        }
    }
    auto fold = [&](const uint64_t* mask, uint64_t* dst) {
        for (int w = 0; w < MW; ++w) {
            uint64_t bits = mask[w];
            while (bits) {
                const int j = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                lb_xor(dst, rec.data() + (size_t)j * MW, MW);
            }
        }
    };
    if (!prog.emask.empty())
        for (int r = 0; r < c.R; ++r) fold(prog.emask.data() + (size_t)r * MW, c.U.data() + (size_t)r * MW);
    c.S.assign((size_t)c.T * MW, 0);
    c.ref.assign((size_t)c.T, 0);
    for (int d = 0; d < prog.D; ++d) {
        fold(prog.dmask.data() + (size_t)d * MW, c.S.data() + (size_t)d * MW);
        c.ref[(size_t)d] = prog.refdet[(size_t)d];
    }
    for (int o = 0; o < prog.O; ++o) {
        fold(prog.omask.data() + (size_t)o * MW, c.S.data() + (size_t)(prog.D + o) * MW);
        c.ref[(size_t)(prog.D + o)] = prog.refobs[(size_t)o];
    }
    for (int k = 0; k < K; ++k)
        for (int j : decisions[(size_t)k])
            lb_xor(c.S.data() + (size_t)(prog.D + prog.O + k) * MW, rec.data() + (size_t)j * MW, MW);
    c.pe = std::make_unique<PeShotAnalyzer>(prog.bare, prog.plan.ms);
    c.built = true;
}
}  // namespace

ExactLawResult sample_program_exact_law(const SamplerProgram& prog, ExactLawCache& cache,
                                        std::vector<FiredPauli> fired,
                                        const std::vector<int>& record_flips,
                                        const std::vector<std::vector<int>>& decisions,
                                        const std::vector<int>& exp_cols_in, long max_rows,
                                        double max_work, bool no_columns) {
    ExactLawResult out;
    std::vector<int> exp_cols = exp_cols_in;
    std::vector<Pauli> A, X;
    std::vector<uint64_t> forced;
    out.error = forced_world(prog, std::move(fired), record_flips, decisions, exp_cols, A, X, forced,
                             !no_columns);
    if (!out.error.empty()) return out;
    if (max_rows < 1) { out.error = "max_rows must be >= 1"; return out; }
    if (!(max_work > 0)) { out.error = "max_work must be > 0"; return out; }
    std::lock_guard<std::mutex> lock(cache.mu);
    if (!cache.built || cache.decisions != decisions) build_law_forms(prog, decisions, cache);
    PeShotAnalyzer& pe = *cache.pe;
    const FramedSuperposition& bare = pe.bare();
    const int M = prog.M, MW = cache.MW, T = cache.T, n = prog.n;
    const int K = (int)decisions.size(), R = (int)exp_cols.size();
    const int NW = pe.nonfree_words(), FW = pe.free_words();
    const int TW = std::max(1, (T + 63) / 64);
    out.D = prog.D; out.O = prog.O; out.K = K; out.R = R;

    // Query constants: r0 = raw reads + `!` inverts + forced flips.
    std::vector<uint64_t> r0 = forced;
    r0.resize((size_t)MW, 0);
    if (prog.has_rf)
        for (int j = 0; j < M; ++j)
            if (prog.rec_invert[(size_t)j]) r0[(size_t)(j >> 6)] ^= 1ull << (j & 63);
    const uint64_t* S = cache.S.data();
    auto Srow = [&](int t) { return S + (size_t)t * MW; };
    std::vector<uint8_t> c((size_t)T);
    for (int t = 0; t < T; ++t) c[(size_t)t] = (uint8_t)(cache.ref[(size_t)t] ^ lb_par(Srow(t), r0.data(), MW));

    // Frame coordinates of every world read.
    std::vector<uint64_t> xr((size_t)M * NW), phr((size_t)M * FW);
    {
        std::vector<uint64_t> xb, pb;
        for (int k = 0; k < M; ++k) {
            pe.frame_parts(A[(size_t)k], xb, pb);
            std::copy(xb.begin(), xb.end(), xr.begin() + (size_t)k * NW);
            std::copy(pb.begin(), pb.end(), phr.begin() + (size_t)k * FW);
        }
    }
    auto fold_into = [&](const uint64_t* set, const std::vector<uint64_t>& per, int W, uint64_t* dst) {
        for (int w = 0; w < MW; ++w) {
            uint64_t bits = set[w];
            while (bits) {
                const int k = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                lb_xor(dst, per.data() + (size_t)k * W, W);
            }
        }
    };
    auto product = [&](const uint64_t* set) {
        Pauli P(n);
        for (int w = 0; w < MW; ++w) {
            uint64_t bits = set[w];
            while (bits) {
                const int k = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                pmul_into(P, A[(size_t)k]);
            }
        }
        return P;
    };
    // xs_t: the non-free X-part of target t's read set; the columns of G = S X (target space).
    std::vector<uint64_t> xs((size_t)T * NW, 0);
    for (int t = 0; t < T; ++t) fold_into(Srow(t), xr, NW, xs.data() + (size_t)t * NW);
    const int NP = NW * 64;
    std::vector<uint64_t> colv((size_t)NP * TW, 0);
    std::vector<uint8_t> colnz((size_t)NP, 0);
    for (int t = 0; t < T; ++t) {
        const uint64_t* v = xs.data() + (size_t)t * NW;
        for (int w = 0; w < NW; ++w) {
            uint64_t bits = v[w];
            while (bits) {
                const int p = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                colv[(size_t)p * TW + (t >> 6)] |= 1ull << (t & 63);
                colnz[(size_t)p] = 1;
            }
        }
    }
    // Uniform directions: the reduced column echelon of G (each g_j is 1 at its pivot target and
    // 0 at every other pivot). Every target direction in this span is an independent fair coin.
    std::vector<std::vector<uint64_t>> dirs;
    std::vector<int> dpiv;
    for (int p = 0; p < NP; ++p) {
        if (!colnz[(size_t)p]) continue;
        std::vector<uint64_t> v(colv.begin() + (size_t)p * TW, colv.begin() + (size_t)(p + 1) * TW);
        for (size_t j = 0; j < dirs.size(); ++j)
            if (lb_bit(v.data(), dpiv[j])) lb_xor(v.data(), dirs[j].data(), TW);
        const int pv = lb_low(v.data(), TW);
        if (pv < 0) continue;
        for (size_t j = 0; j < dirs.size(); ++j)
            if (lb_bit(dirs[j].data(), pv)) lb_xor(dirs[j].data(), v.data(), TW);
        dirs.push_back(std::move(v));
        dpiv.push_back(pv);
    }
    const int u = (int)dirs.size();
    std::vector<uint8_t> is_piv((size_t)T, 0);
    for (int pv : dpiv) is_piv[(size_t)pv] = 1;

    // Structured part: for each non-pivot target t the coset-representative functional
    // rep_t = c'_t + m . R_t (R_t = S_t + sum_{j: g_j[t]} S_{piv_j}; its non-free X-part vanishes),
    // decomposed over the magic-sector parities y_i = m . Tset_i as rep_t = c'_t + delta_t +
    // coef_t . y (delta_t: the deterministic +-1 of the phi-free remainder on the bare state).
    struct PB { std::vector<uint64_t> phi, comb; int piv; };
    std::vector<PB> basis;
    std::vector<uint8_t> cconst((size_t)T, 0);
    std::vector<uint64_t> coef((size_t)T, 0);
    std::vector<uint64_t> Rt((size_t)MW), ph((size_t)FW);
    const double kDet = 1e-9;
    for (int t = 0; t < T; ++t) {
        if (is_piv[(size_t)t]) continue;
        std::copy(Srow(t), Srow(t) + MW, Rt.begin());
        uint8_t ct = c[(size_t)t];
        for (int j = 0; j < u; ++j)
            if (lb_bit(dirs[(size_t)j].data(), t)) { lb_xor(Rt.data(), Srow(dpiv[(size_t)j]), MW); ct ^= c[(size_t)dpiv[(size_t)j]]; }
        if (!lb_any(Rt.data(), MW)) { cconst[(size_t)t] = ct; continue; }
        std::fill(ph.begin(), ph.end(), 0);
        fold_into(Rt.data(), phr, FW, ph.data());
        uint64_t used = 0;
        for (size_t i = 0; i < basis.size(); ++i)
            if (lb_bit(ph.data(), basis[i].piv)) {
                lb_xor(ph.data(), basis[i].phi.data(), FW);
                lb_xor(Rt.data(), basis[i].comb.data(), MW);
                used |= 1ull << i;
            }
        const int pv = lb_low(ph.data(), FW);
        if (pv >= 0) {
            if (basis.size() >= 63) {
                out.error = "the targets depend on more than 63 magic-sector read parities";
                return out;
            }
            used |= 1ull << basis.size();
            basis.push_back({ph, Rt, pv});
        } else if (lb_any(Rt.data(), MW)) {
            const double v = bare.expectation(product(Rt.data()));
            if (std::fabs(v - 1.0) < kDet) {}
            else if (std::fabs(v + 1.0) < kDet) ct ^= 1;
            else {
                out.error = "internal: a twirl-deterministic read parity has <P> = " + std::to_string(v);
                return out;
            }
        }
        cconst[(size_t)t] = ct;
        coef[(size_t)t] = used;
    }
    out.d = (int)basis.size();
    out.u = u;

    // Expectation columns: zero if anticommuting with a read or outside the reads' X-span;
    // else X'_r = X_r A_{w0} with w0 = U_r + sum_{alpha_j} S_{piv_j}, alpha the pivot-row
    // solution of xn(X_r) + xn(A_{U_r}) = sum_j alpha_j xs_{piv_j}; direction j flips column r iff
    // alpha_j; the row value is (-1)^{e_r + sum alpha_j c_{piv_j}} <X'_r | magic parities>.
    const int UW = std::max(1, (u + 63) / 64);
    struct ERow { std::vector<uint64_t> v, comb; int piv; };
    std::vector<ERow> erows;
    bool erows_built = false;
    auto build_erows = [&]() {
        for (int j = 0; j < u; ++j) {
            ERow e;
            e.v.assign(xs.begin() + (size_t)dpiv[(size_t)j] * NW, xs.begin() + (size_t)(dpiv[(size_t)j] + 1) * NW);
            e.comb.assign((size_t)UW, 0);
            e.comb[(size_t)(j >> 6)] |= 1ull << (j & 63);
            for (const ERow& f : erows)
                if (lb_bit(e.v.data(), f.piv)) { lb_xor(e.v.data(), f.v.data(), NW); lb_xor(e.comb.data(), f.comb.data(), UW); }
            e.piv = lb_low(e.v.data(), NW);
            if (e.piv < 0) return false;
            erows.push_back(std::move(e));
        }
        return true;
    };
    std::vector<int> live;                     // column index -> V index, or -1 (zero column)
    std::vector<uint8_t> esgn((size_t)R, 0);
    std::vector<Pauli> V;
    out.usign.assign((size_t)u * R, 0);
    {
        std::vector<uint64_t> xi, phx, w0((size_t)MW);
        for (int r = 0; r < R; ++r) {
            const Pauli& Xr = X[(size_t)r];
            bool anti = false;
            for (int k = 0; k < M && !anti; ++k) anti = Pauli::anticommute_bit(Xr, A[(size_t)k]);
            if (anti) { live.push_back(-1); continue; }
            const uint64_t* Ur = cache.U.data() + (size_t)exp_cols[(size_t)r] * MW;
            pe.frame_parts(Xr, xi, phx);
            fold_into(Ur, xr, NW, xi.data());
            std::vector<uint64_t> alpha((size_t)UW, 0);
            if (lb_any(xi.data(), NW)) {
                if (!erows_built) {
                    if (!build_erows()) { out.error = "internal: dependent direction pivot rows"; return out; }
                    erows_built = true;
                }
                for (const ERow& e : erows)
                    if (lb_bit(xi.data(), e.piv)) { lb_xor(xi.data(), e.v.data(), NW); lb_xor(alpha.data(), e.comb.data(), UW); }
                if (lb_any(xi.data(), NW)) { live.push_back(-1); continue; }
            }
            std::copy(Ur, Ur + MW, w0.begin());
            uint8_t s0 = (uint8_t)lb_par(Ur, r0.data(), MW);
            for (int j = 0; j < u; ++j)
                if (lb_bit(alpha.data(), j)) {
                    lb_xor(w0.data(), Srow(dpiv[(size_t)j]), MW);
                    s0 ^= c[(size_t)dpiv[(size_t)j]];
                    out.usign[(size_t)j * R + r] = 1;
                }
            Pauli Xp = Xr;
            pmul_into(Xp, product(w0.data()));
            esgn[(size_t)r] = s0;
            live.push_back((int)V.size());
            V.push_back(std::move(Xp));
        }
    }
    // The magic-sector parities' exact joint law on its support (+ the conditional columns).
    std::vector<Pauli> B;
    for (const PB& b : basis) B.push_back(product(b.comb.data()));
    std::vector<PeShotAnalyzer::JointLeaf> leaves;
    std::string why;
    if (B.empty()) {                           // no magic parity: one row, columns on the bare state
        PeShotAnalyzer::JointLeaf jl{0, 1.0, {}};
        jl.v.reserve(V.size());
        for (const Pauli& P : V) jl.v.push_back(bare.expectation(P));
        leaves.push_back(std::move(jl));
    } else {
        int kind = 0;
        if (!pe.joint_law(B, V, leaves, why, max_rows, max_work, &kind, true)) {
            const std::string rank = std::to_string(B.size()) + " magic-sector parities (magic rank " +
                                     std::to_string(bare.k()) + ", " + std::to_string(bare.chi()) +
                                     " amplitudes)";
            out.error = kind == 2
                ? "the structured law's exact walk over " + rank + " exceeds max_work = " +
                      std::to_string((long long)max_work) + " work units (deterministic count; raise max_work)"
                : "the structured law over " + rank + " has more than max_rows = " +
                      std::to_string(max_rows) + " rows";
            return out;
        }
    }
    if ((long)leaves.size() > max_rows) {
        out.error = "the structured law has " + std::to_string(leaves.size()) + " rows (> max_rows = " +
                    std::to_string(max_rows) + ")";
        return out;
    }
    // Rows (merged by identical targets, conditional expectations probability-weighted).
    std::unordered_map<std::string, size_t> seen;
    std::string key((size_t)T, '\0');
    std::vector<double> num;
    double tot = 0.0;
    for (const auto& lf : leaves) {
        for (int t = 0; t < T; ++t)
            key[(size_t)t] = is_piv[(size_t)t] ? 0
                : (char)(cconst[(size_t)t] ^ (__builtin_popcountll(coef[(size_t)t] & lf.f) & 1));
        tot += lf.pr;
        auto [it, fresh] = seen.emplace(key, out.prob.size());
        if (fresh) {
            out.prob.push_back(0.0);
            num.resize(num.size() + (size_t)R, 0.0);
            for (int t = 0; t < prog.D; ++t) out.det.push_back((uint8_t)key[(size_t)t]);
            for (int t = 0; t < prog.O; ++t) out.obs.push_back((uint8_t)key[(size_t)(prog.D + t)]);
            for (int t = 0; t < K; ++t) out.dec.push_back((uint8_t)key[(size_t)(prog.D + prog.O + t)]);
        }
        const size_t row = it->second;
        out.prob[row] += lf.pr;
        for (int r = 0; r < R; ++r)
            if (live[(size_t)r] >= 0) {
                const double v = lf.v[(size_t)live[(size_t)r]];
                num[row * R + r] += lf.pr * (esgn[(size_t)r] ? -v : v);
            }
    }
    if (std::fabs(tot - 1.0) > 1e-9) {
        ExactLawResult bad;
        bad.error = "internal: law probabilities sum to " + std::to_string(tot);
        return bad;
    }
    out.exps.assign(out.prob.size() * (size_t)R, 0.0);
    for (size_t i = 0; i < out.prob.size(); ++i)
        for (int r = 0; r < R; ++r)
            out.exps[i * R + r] = out.prob[i] > 0 ? num[i * R + r] / out.prob[i] : 0.0;
    for (int j = 0; j < u; ++j) {
        const uint64_t* g = dirs[(size_t)j].data();
        for (int t = 0; t < prog.D; ++t) out.udet.push_back((uint8_t)lb_bit(g, t));
        for (int t = 0; t < prog.O; ++t) out.uobs.push_back((uint8_t)lb_bit(g, prog.D + t));
        for (int t = 0; t < K; ++t) out.udec.push_back((uint8_t)lb_bit(g, prog.D + prog.O + t));
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
