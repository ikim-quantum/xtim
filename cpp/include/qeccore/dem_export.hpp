#pragma once
#include <string>
#include <utility>
#include <vector>
#include "qeccore/stim_parse.hpp"

namespace qeccore {

// Stim-compatible detector error model export (production roadmap R3).
//
// CONTRACT (2026-06-19, magic-state-prep completeness + reject region — supersedes the
// per-read/per-column "REFUSE" outcomes described in the algebra notes below; the algebra
// itself is unchanged, only what we DO with a non-Pauli effect):
//   * A non-Pauli (odd-S / CZ) effect on a MEASURED DETECTOR read is no longer a refusal: by
//     the measurement-twirl theorem (docs/twirl_pauli_condition.tex) it is a fair-coin Z, i.e.
//     a probability-1/2 record flip, expanded into ordinary Pauli mechanisms (see dem_classify.hpp).
//   * A fault is classified by the MAGNITUDE of the logical observables (the PAULI_EXPECTATION
//     columns): if every |<P_r>| is preserved it is Pauli-correctable (an ordinary DEM edge, with a
//     logical-observable flip on each sign-changed column); if some |<P_r>| changes it is
//     detectable-but-not-Pauli-correctable -> emitted NOWHERE in the DEM and FLAGGED in
//     `postselect_faults` (its detector signature + probability), so the user chooses to post-select
//     it or budget it. `reject_detectors` is the derived union convenience (the blunt policy), NOT a
//     forced reject — we flag the not-correctable faults, we do not force post-selection.
//   * COMPLETENESS GATE: a fault that changes a magnitude but fires no detector (or whose change
//     survives the accepted branch) is an undetectable logical error -> the export REFUSES loudly
//     (res.error), because the circuit under-declares its post-selection syndrome.
// See cpp/include/qeccore/dem_classify.hpp (the classifier) and the design/plan under
// docs/superpowers/{specs,plans}/2026-06-19-dem-exporter-* for the full rationale.
//
// Pipeline: parse -> defer_measurements -> build_propagation_table -> per noise channel,
// enumerate the alternatives (the same per-qubit / per-pair channel granularity as the
// sampler) and compute each alternative's DETERMINISTIC record-flip mask from its
// end-of-circuit propagated error D = gamma * X^v * Delta(a, B):
//   * Z-read on wire q:   flip = v[q]          (a and B commute with Z -- always exact);
//   * X-read on wire q:   flip = (a[q] == 2);  REFUSED if a[q] is odd (the error rotates
//                         the read X<->Y: the deviated outcome is a different observable's
//                         coin, NOT a record flip) or if B has any CZ incident on q (the
//                         read operator gains Z_partner: an outcome-conditional flip);
//   * Y-read on wire q:   flip = (a[q] == 2) XOR v[q], same refusal conditions.
// The refusals are the DEM determinism gate: a DEM `error(p) D... L...` line asserts a fixed
// flip set, so any alternative whose effect is outcome-dependent makes the circuit
// NON-EXPRESSIBLE as a DEM and the export fails loudly (never silently approximates).
// The gate is per-read and state-free, hence SOUND but CONSERVATIVE: it can refuse cases
// that are in fact expressible (e.g. a propagated CZ incident on an X read whose partner
// wire is deterministically |0> -- the conditional flip is then a deterministic no-op; or a
// rotated read inside a multi-read parity that stays deterministic jointly). When every
// alternative is accepted, C^dag R_j C = +-R_j exactly per read, so the deviated JOINT read
// distribution is the reference distribution shifted by the flip mask -- exact, including
// detector correlations (verified against exact statevector enumeration on T/CCZ circuits).
//
// Circuit-level gate (Stim parity): every detector/observable parity must be DETERMINISTIC
// under noiseless execution -- checked exactly via the Born probability of the parity Pauli
// (product of the read Paulis over the channel's record set) on the deduced bare state.
//
// Probability semantics (pinned against real Stim, see tests/test_dem_stim.py):
//   * within a channel, alternatives with identical (D, L) signatures merge by probability
//     ADDITION (mutually exclusive outcomes of one categorical channel);
//   * a channel with >= 2 distinct signatures is converted EXACTLY to independent error
//     mechanisms by the Walsh-Hadamard solve over the signature group (this is what Stim
//     does for DEPOLARIZE1/2 and exactly-convertible PAULI_CHANNEL_1/2: e.g. symmetric
//     depolarize1 with 3 distinct signatures gives q = (1 - sqrt(1 - 4p/3))/2 per
//     mechanism). If the solve fails (a Fourier coefficient <= 0, i.e. over-mixing, or a
//     negative mechanism probability, i.e. the channel is not a product of independent
//     flips -- Stim's `approximate_disjoint_errors` territory), the export REFUSES;
//   * across channels, mechanisms with identical signatures compose by independent XOR:
//     p (+) q = p + q - 2pq;
//   * empty-signature mechanisms are dropped (Stim drops undetectable errors); detectors /
//     declared observables not covered by any error get bare `detector Dk` /
//     `logical_observable Lk` lines (Stim's convention; coordinates are omitted --
//     parseability, not cosmetics).
//
// PAULI_EXPECTATION L-columns (xtim M2). Each PAULI_EXPECTATION declaration becomes one
// additional L-column, indexed AFTER all OBSERVABLE_INCLUDE observables, both in
// declaration order (THE ordering contract: with O observables and R declarations,
// expectation r is column L(O+r)). The per-mechanism bit asserts "this mechanism flips
// the SIGN of declared Pauli r" — sound because a kept mechanism then satisfies
// E†PE = ±P exactly, so a decoder's predicted flip is exactly a sign on the raw
// per-shot expectation value (no re-simulation).
//
// The ±1-expressibility condition (derived from the conjugation algebra; the same gate
// as the per-read rules above, generalized to a multi-qubit Pauli): write the
// mechanism's end-of-circuit propagated error E = gamma · X^v · Delta(a, B) and the
// declared Pauli P = i^{#Y} X^x Z^z. Since Delta is diagonal,
//     Delta† P Delta = P · (Delta_x† Delta),  Delta_x = X^x Delta X^x,
// and the diagonal correction Delta_x† Delta factorizes per term of Delta's exponent:
//   * S-power a_q on q in supp(x):  factor i^{-a_q} · Z_q^{a_q mod 2}
//       - a_q odd  -> a leftover Z_q (and an i phase): E†PE rotates P to a DIFFERENT
//         observable (the X<->Y rotation), NOT a sign — non-expressible;
//       - a_q == 2 -> scalar -1: a clean sign flip;
//   * CZ-arm B_{ij}: leftover Z_j if exactly one endpoint lies in supp(x), and
//     -Z_i Z_j if both do — either way a leftover Z-dressing on the operator,
//     non-expressible (the flip would be outcome/state-conditional).
// Hence E†PE = ±P  iff  no odd S-power and no CZ-arm of the propagated Delta touches
// supp(x), and then the sign is
//     (-1)^( |{q : v_q ∧ z_q}|  +  |{q in supp(x) : a_q == 2}| )
// (X^v anticommuting with P's Z-content, plus the even-S-power Z-layer on P's
// X-support). Any alternative violating the condition REFUSES the export, naming the
// column and the mechanism (same honest-gate philosophy: never approximate). As with
// the read gate, the condition is per-alternative and state-free — sound, possibly
// conservative.
//
// Back-compat surface (the chosen opt-out): include_expectations defaults ON — a DEM
// asked for on a circuit with PAULI_EXPECTATION declarations speaks for ALL of them or
// refuses loudly. The explicit opt-out (include_expectations = false) reproduces the
// pre-M2 export (expectation channels ignored) byte-for-byte: a consumer that only
// decodes detectors/observables (the classic Stim workflow) is asking a strictly
// weaker, still-honest question, so the opt-out changes the question asked — never the
// fidelity of the answer. A flag (rather than refusal-only) was chosen because the
// non-expressibility of one declared Pauli says nothing about the validity of the
// detector/observable model, which remains exactly as exportable as before.
struct DemExportResult {
    bool ok = false;
    std::string dem;                       // valid iff ok (Stim DEM text)
    std::vector<ParseError> parse_errors;  // non-empty => parse-level failure
    bool rejected = false;                 // pipeline reject (propagation table / bare state)
    int  reject_gate_index = -1;           // valid when rejected
    std::string error;                     // non-expressible diagnostic (!ok, parse ok, !rejected)
    std::vector<int> dropped_observables;  // observable indices removed (gauge observables with drop_gauge_observables=true)
    std::vector<int> reject_detectors;     // detectors to post-select on (not-Pauli-correctable faults)
    // Not-Pauli-correctable faults to FLAG (the user chooses to post-select these or budget them).
    // Each: (sorted detector signature it flips/randomizes, probability). reject_detectors above is
    // the derived union convenience only.
    std::vector<std::pair<std::vector<int>, double>> postselect_faults;
};

// trusted_detectors / trusted_observables: ids whose determinism has been certified EXTERNALLY
// against the TRUE bare state (e.g. a from_state-compiled sampler's channel_report, whose
// composed state includes a carried INPUT_QUBITS port the text alone cannot express).  For those
// ids the text-only determinism gate is skipped; every other id is still gated.  The error
// mechanisms themselves are state-free Clifford flip propagation, so a trusted id changes ONLY
// the gate, never the emitted DEM columns.  Callers MUST pass only ids certified by an exact
// bare-state check — a wrongly trusted gauge detector yields a silently wrong DEM.
DemExportResult export_dem(const std::string& stim_text,
                           bool include_expectations = true,
                           bool decompose_errors = false,
                           bool ignore_decomposition_failures = false,
                           bool drop_gauge_observables = false,
                           const std::vector<int>* trusted_detectors = nullptr,
                           const std::vector<int>* trusted_observables = nullptr);

// Exact per-detector / per-observable determinism partition of the NOISELESS circuit.
//
// A detector (resp. a declared OBSERVABLE_INCLUDE observable) is DETERMINISTIC iff its parity
// Pauli — the product of the read Paulis over its record set — is a ±1 eigen-operator (a
// stabilizer) of the noiseless bare reference state, i.e. framed_expectation gives |<P>| == 1.
// Otherwise it is GAUGE (the parity is a fair coin noiselessly).
//
// This is the SAME algebra as export_dem's circuit-level determinism gate (see the header notes
// above), factored out so the verdict can be queried WITHOUT requiring the whole DEM to be
// expressible: determinism depends only on the bare state and the parity Pauli, never on the
// noise channels. So it answers even for circuits whose full DEM refuses (a strong noise channel,
// a probabilistic observable). Exact and shot-free — no sampling, no dependence on how well a
// circuit's noise can be stripped. This is the source-of-truth determinism check; the sampling
// test in xtim.diagnose is a fallback that must AGREE with it.
struct DeterminismResult {
    bool ok = false;                        // valid iff parse ok, not rejected, no setup error
    std::vector<ParseError> parse_errors;   // non-empty => parse-level failure
    bool rejected = false;                  // propagation-class / bare-state reject
    int  reject_gate_index = -1;            // valid when rejected
    std::string error;                      // setup diagnostic (!ok, parse ok, !rejected)
    int num_detectors = 0;
    int num_observables = 0;                // max OBSERVABLE_INCLUDE index + 1 (0 if none)
    std::vector<int> deterministic_detectors;   // sorted ascending
    std::vector<int> gauge_detectors;            // sorted ascending
    std::vector<int> deterministic_observables;  // sorted ascending (declared indices only)
    std::vector<int> gauge_observables;          // sorted ascending (declared indices only)
    // Signed parity expectation <P> = pp - pm per detector (indexed by detector id) and per
    // declared observable (indexed by observable id; NaN for an undeclared/gap index). A
    // deterministic channel has |<P>| == 1; a gauge channel has |<P>| < 1 (0 for a fair coin).
    std::vector<double> detector_parity;
    std::vector<double> observable_parity;
};

DeterminismResult detector_determinism(const std::string& stim_text);

// Exact verify of each declared PAULI_EXPECTATION byproduct-frame (Task 4).
//
// A declared frame (the `rec[-k]` tail of a PAULI_EXPECTATION line) asserts that the per-shot
// SIGN of <P_r> becomes deterministic once the outcomes of the declared records are XOR-folded
// in. This verifier decides, EXACTLY and sampling-free, whether the declared record set is
// SUFFICIENT to determine that sign.
//
// METHOD (a property of the bare state's stabilizer frame, NOT forward propagation): the sign of
// <P_r> is flipped by exactly the FIXED stabilizer generators g_a = U Z_a U† that expP[r]
// anticommutes with. With avec(P)[a] = symplectic(P, g_a) over the fixed (non-`free`) generators
// only, a declared frame S RESOLVES the sign iff every such generator is covered by some declared
// read — i.e. iff  avec(expP[r]) ∈ span{ avec(R_j) : j ∈ S }  over GF(2).
//
// SPAN VERDICT (column_ok): the declared frame is ACCEPTED iff avec(expP[r]) lies in the span of
// the DECLARED reads' avecs (a zero residual after echelon-reducing avec(expP[r]) against the
// declared-read basis). A determinizing SUPERSET spans and thus ACCEPTS; a frame MISSING a
// sign-controlling record leaves an UNRESOLVED generator (nonzero residual) and REJECTS.
//
// DIAGNOSTICS (solvable, required): `solvable` is the GLOBAL span — whether SOME record set can
// determinize the sign (avec(expP) ∈ span of ALL reads); `required` is that global system's
// canonical-minimal solution (deterministic reads whose avec is 0 are never pivots, so excluded).
// These are informational: the diagnose layer refuses a column that is `solvable` (a determinizing
// frame EXISTS) yet not `column_ok` (the DECLARED frame does not span) — a missing/mis-declared
// frame. A determinizing superset has column_ok=1 while required may be a strict subset of it.
//
// CRITICAL: avec EXCLUDES the `free`/logical tableau rows. Magic (e.g. T|+>) makes expP
// anticommute with a `free` generator — that is the 1/√2 MAGNITUDE, not a sign-frame fact;
// including free rows makes the solve spuriously unsolvable.
//
// `all_ok` is false iff any column's declared frame does not span; `refusal` is a loud diagnostic
// naming each bad column and its unresolved generators (empty when all_ok). Feedback is out of
// scope (refused via `error`).
struct FrameCheckResult {
    bool ok = false;                        // valid iff parse ok, not rejected, no setup error
    std::vector<ParseError> parse_errors;   // non-empty => parse-level failure
    bool rejected = false;                  // propagation-class / bare-state reject
    int  reject_gate_index = -1;            // valid when rejected
    std::string error;                      // setup diagnostic (!ok, parse ok, !rejected)
    int num_columns = 0;                    // number of PAULI_EXPECTATION columns (R)
    std::vector<std::vector<int>> required; // per column: global canonical-minimal determining set
    std::vector<std::vector<int>> declared; // per column: sorted declared frame (obs_frame)
    std::vector<uint8_t> solvable;          // per column: SOME record set determinizes (global span)
    std::vector<uint8_t> column_ok;         // per column: avec(expP) ∈ span(declared reads)
    bool all_ok = false;                    // every column's declared frame spans
    std::string refusal;                    // loud refusal naming unresolved columns (empty if all_ok)
};

FrameCheckResult expectation_frame_check(const std::string& stim_text);

// Layout of the extended DEM produced by export_dem for a module circuit
// (one that declares OUTPUT_QUBITS and/or DECISION).  Column indices below are
// L-indices as emitted in the DEM text (L0 = first observable column).
//
// Column ordering (mandatory contract, tested by mutation):
//   L0 .. L{O-1}               — OBSERVABLE_INCLUDE columns (standard)
//   L{O} .. L{O+R-1}           — PAULI_EXPECTATION columns (R columns)
//   L{n_obs+0}, L{n_obs+1}     — frameX(q0), frameZ(q0)  per OUTPUT_QUBITS qubit q0
//   ...
//   L{n_obs+2*NOUT}..           — DECISION logical-flip columns (one per DECISION index)
//
// n_obs = O + R (total existing observable L-columns before the new module columns).
// frame[i]         = (original_qubit, colX_Lindex, colZ_Lindex)
// logical_flip[j]  = (decision_k, col_Lindex)
struct DemColumnLayout {
    bool ok = false;
    std::vector<ParseError> parse_errors;
    std::string error;
    bool rejected = false;
    int reject_gate_index = -1;
    int n_obs = 0;                                       // O + R: existing L-columns
    std::vector<std::tuple<int,int,int>> frame;          // (orig_qubit, colX, colZ)
    std::vector<std::pair<int,int>> logical_flip;        // (decision_k, col)
};

DemColumnLayout dem_column_layout(const std::string& stim_text);

}  // namespace qeccore
