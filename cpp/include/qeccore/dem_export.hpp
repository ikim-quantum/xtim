#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "qeccore/stim_parse.hpp"
#include "qeccore/dem_pe_per_shot.hpp"
#include "qeccore/dem_decompose.hpp"

namespace qeccore {

struct FramedSuperposition;
struct Pauli;

// ── The exact correlated-read law's coset weights, factored through the free register ──────────
// nu = WHT(mu), mu[c] = <A_{S_c}> on the bare state L (frame coordinates), A_{S_c} the Gray-code
// product of the W-basis Paulis N (no non-free X part). Bit-identical to one
// framed_expectation_frame per subset (the pre-factoring walk, fd4f31c): every <A_S> is the SAME Born kernel on
// A_S's free-register projection (phase, X/Z bits on `free`, one non-free eps sign bit),
// evaluated once per distinct projection (memo: <= 8 * 4^{r_free} Born sums per export). See
// dem_export.cpp, correlated_law_coset_weights.
struct FreeRegisterBornMemo {
    std::unordered_map<uint64_t, double> packed;      // r_free <= 30: (phase, sign, x_free, z_free) packed
    std::unordered_map<std::string, double> wide;     // r_free > 30
    long born_sums = 0;                               // Born sums actually evaluated (memo misses)
};
void correlated_law_coset_weights(const FramedSuperposition& L, const std::vector<Pauli>& N,
                                  std::vector<double>& nu, FreeRegisterBornMemo& memo);
// The same walk without the transform: mu[c] = <A_{S_c}> (Gray-code products of N).
void correlated_law_mu(const FramedSuperposition& L, const std::vector<Pauli>& N,
                       std::vector<double>& mu, FreeRegisterBornMemo& memo);
// One <Q> (Q with no non-free X part, or any Q: then unprojected), through the same memo slot the
// walk uses — the EXP-fix coset fast path evaluates only the wdim basis products N_i.
double correlated_law_born(const FramedSuperposition& L, const Pauli& Q, FreeRegisterBornMemo& memo);

// The classified correlated-read law of one fault's randomized targets T_R (EXP-fix; see dem_export.cpp):
//   COSET   uniform on y0 + span(yb) in the Fourier-dual coordinates (times every z): the sound fast
//           path (every Fourier basis product <N_i> = +-1: wdim Born sums) or re-detected after the walk
//           (every |mu| in {0, 1});
//   GENERAL nu = WHT(mu) on its y-support ys (filter 1e-12 relative to the peak).
// The law's support dimension is (m - |nullv|) + |yb|.
struct CorrelatedLaw {
    std::vector<std::vector<uint64_t>> nullv;   // basis of the Fourier support W (subsets of T_R)
    bool coset = true;
    bool redetected = false;
    uint32_t y0 = 0;                            // a support point (the coset centre)
    std::vector<uint32_t> yb;                   // direction basis of the y-support
    std::vector<uint32_t> ys;                   // GENERAL: the y-support
    std::vector<double> nu;                     // GENERAL: WHT(mu)
};
// The Fourier support basis of the A_t (frame coordinates Qf; no Born sum): subsets whose frame X-part
// lies on the free generators. m - size is the fault's support dimension before any Born sum (v0).
std::vector<std::vector<uint64_t>> correlated_law_fourier_support(const FramedSuperposition& L,
                                                                  const std::vector<Pauli>& Qf);
// Classify the law (law.nullv set; the caller has applied the v0 and wdim <= 20 caps): wdim Born sums on
// the fast path, else the 2^wdim walk and the R1 re-detection.
void classify_correlated_law(const FramedSuperposition& L, const std::vector<Pauli>& Qf, CorrelatedLaw& law,
                             FreeRegisterBornMemo& memo);
// The fault's DEM law over the target ids rt (ascending), dressed by the deterministic signature base.
FaultLaw correlated_fault_law(const CorrelatedLaw& law, int m, const std::vector<int>& rt, const Sig& base,
                              double prob);

// EXP-fix (owner): the dimension cap on one fault's DEM support span. A fault whose law spans more
// than 2^kDemSupportCap signatures is refused by name ("the Clifford error this fault leaves
// triggers too many detectors").
constexpr int kDemSupportCap = 12;   // owner 2026-10-22: 10 -> 12

// Test-API census of the per-fault DEM laws (G-DEM-* gates; nullptr in production). One record per
// channel instance that reached the solve, in stream order.
struct DemLawCensus {
    struct Comp {
        char kind;      // 'P' Pauli, 'C' coset (fast path), 'R' coset re-detected, 'G' general, 'L' list
        double p;
        int v0;         // m - wdim of the targets' Fourier support (randomized / list), else 0
        int wdim;
        int v;          // the support dimension the cap is applied to (coset v, general / list d)
    };
    struct Inst {
        int k;          // deferred stream index
        std::string channel;
        std::string route;    // pauli_old | coset_closed | joint_fwht
        std::string status;   // ok | overmix | negative | rank | approx_disjoint
        int rank;             // joint signature rank (joint_fwht / pauli routing), else -1
        std::vector<Comp> comps;
    };
    std::vector<Inst> inst;
    long born_sums = 0;
};

// Stim-compatible detector error model export (production roadmap R3).
//
// CONTRACT (2026-06-19, magic-state-prep completeness + reject region — supersedes the
// per-read/per-column "REFUSE" outcomes described in the algebra notes below; the algebra
// itself is unchanged, only what we DO with a non-Pauli effect):
//   * A non-Pauli (odd-S / CZ) effect on a MEASURED DETECTOR read is no longer a refusal. Since
//     3.1.10 the targets fed by such randomized reads get their EXACT joint law (until 3.1.9: each
//     read an independent fair coin — wrong whenever the flips correlate): with A_t = s_t E^dag D_t E
//     the flip pattern's Fourier transform is <prod_{t in S} A_t> on the bare state, supported on
//     the subsets whose frame X-part lies on free generators. This is the stabilizer-twirled
//     Pauli channel of the residual (each stabilizer coset weighted |sum_g c_{Pg} chi(g)|^2, with
//     its deterministic signature). EXP-fix (2026-10-21, owner design, docs/twirl_pauli_condition.tex
//     "Noise model"): each such fault's law is classified — a uniform coset c' + W (the closed form
//     of solve_coset_fault, O(2^(v+1)) output; found by the +-1 character test on the Fourier basis,
//     or re-detected when every |mu| is 0 or 1) or an explicit support — and its SUPPORT DIMENSION v
//     is capped at kDemSupportCap = 12: past it the fault is refused by name ("the Clifford error
//     this fault leaves triggers too many detectors"). A channel instance with several non-Pauli
//     alternatives is solved jointly (one FWHT over the 2^m group its supports generate; exact mode
//     refuses m > 20 by name, approximate_disjoint_errors emits the disjoint law). Then stim's rules
//     (below).
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
//     mechanisms by the Walsh-Hadamard solve over the signature group (all-Pauli instances of
//     rank <= 4: solve_channel's naive transform, byte-identical; otherwise the closed form or one
//     fast WHT, EXP-fix — probabilities within 1e-6 of the naive solve, owner ruling) (this is what Stim
//     does for DEPOLARIZE1/2 and exactly-convertible PAULI_CHANNEL_1/2: e.g. symmetric
//     depolarize1 with 3 distinct signatures gives q = (1 - sqrt(1 - 4p/3))/2 per
//     mechanism). If the solve fails (a Fourier coefficient <= 0, i.e. over-mixing, or a
//     negative mechanism probability, i.e. the channel is not a product of independent
//     flips -- Stim's `approximate_disjoint_errors` territory), the export REFUSES unless
//     `approximate_disjoint_errors` > 0 (stim's semantics: True = 1; a float t approximates only
//     if every probability argument of the channel is <= t, else refuses naming the argument);
//     the approximation makes each merged signature an independent mechanism with its summed
//     disjoint probability (exact to first order). xtim converts in SIGNATURE space, so it is
//     exact on some channels stim (Pauli space) approximates — e.g. PAULI_CHANNEL_1(0.1,0.1,0)
//     before a Z read is the single mechanism error(0.2);
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
// Declared byproduct frame (3.1.10). A column declared with frame records rec[..] reports
// (-1)^{frame parity}·<P>; in deferred space each frame record is the terminal read R_j of its own
// wire, so the column is the expectation of the frame-folded operator P' = P·Π_{j∈frame} R_j and
// every rule above (the ±1 sign, the magnitude classifier, the completeness gate, beta0) is applied
// to P', not to P. A gate fault that reaches the column only through a frame record (Z before the
// MX of a Bell half) is thereby an ordinary mechanism on L(O+r), exactly like its readout twin
// MX(p); a non-Pauli fault on a frame read (a T-residual S on it) is classified/refused like one on
// P itself. 3.1.4-3.1.9 applied the rules to P alone and dropped such mechanisms.
//
// Meaning of an expectation L-flip (3.1.10, explicit): the mechanism's propagated action
// multiplies P' by -1 (E^dag P' E = -P'), i.e. it multiplies the reported <P> by -1, with the
// records relabelled by the mechanism's own record flips. This is an operator statement, defined
// for ANY noiseless value: 0, 1/sqrt2 (logical Paulis of a magic state) or branch-dependent (a
// column without a frame whose value varies with the records). Such columns are exported, never
// refused. A mechanism whose action on P' is not +-1 (a Clifford residual: odd S-power or CZ arm
// on P''s X-support) is emitted as a sign only when the column's noiseless value is nonzero and
// the magnitude classifier certifies it; on a zero-valued column it is flagged for post-selection
// (it fires a detector) or the export refuses (it fires none).
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
    // PAULI_EXPECTATION columns with NO DEM concept (3.1.12 owner rule): (declaration index r,
    // reason). Their noiseless per-shot value varies shot by shot; L(O+r) stays reserved but no
    // line, mechanism, flag or refusal is emitted for it.
    // With include_expectations = false the same field also REPORTS, per (column, fault
    // location), the findings that refuse the export with include_expectations = true (the
    // completeness gate "non-fault-tolerant circuit (<loc>): ..." and the two "accepted branch
    // was not verified" refusals), each suffixed "(reported, not refused:
    // include_expectations=False)". In that mode no expectation column was requested and the
    // detector/observable DEM carries EVERY fault on its det/obs signature: nothing is refused,
    // flagged (postselect_faults / reject_detectors are empty) or dropped on column grounds
    // (owner decision 2026-10-12, coordinator ruling).
    std::vector<std::pair<int, std::string>> dropped_expectation_columns;
};

// Deterministic work cap of the per-shot expectation-column analysis (dem_pe_per_shot.hpp): work
// units per branch walk (a count, never wall-clock). The single source of the default is
// PeShotAnalyzer::kWorkBudget (the 3.1.12 constant; exported to Python as
// _xtim.PE_MAX_WORK_DEFAULT). A refusal past it names the argument (pe_max_work). The leaf cap
// (PeShotAnalyzer::kLeafBudget, 4096 support leaves per walk) stays fixed: it is not the binding
// limit on the known cases (a column over d magic-correlated reads has at most 2^d leaves, and the
// work cap binds first).
constexpr double kPeMaxWorkDefault = PeShotAnalyzer::kWorkBudget;

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
                           const std::vector<int>* trusted_observables = nullptr,
                           double approximate_disjoint_errors = 0.0,
                           double pe_max_work = kPeMaxWorkDefault,
                           DemLawCensus* census = nullptr);

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
// circuit's noise can be stripped. This is the source-of-truth determinism check (ler's keep mask,
// adaptq; the diagnose report that also consumed it was removed at EXP-fix).
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
// These are informational (xtim.extract_frame reads them; the diagnose report that refused a column
// that is `solvable` yet not `column_ok` was removed at EXP-fix): a column that is `solvable` (a
// determinizing frame EXISTS) yet not `column_ok` (the DECLARED frame does not span) has a
// missing/mis-declared frame. A determinizing superset has column_ok=1 while required may be a strict subset of it.
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
