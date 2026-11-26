#pragma once
// Stored reference decompositions (.ref): a CanonicalStabSum serialized as per-branch
// Clifford PREP CIRCUITS + complex coefficients (production-roadmap spec, components 1+2).
//
// Why prep circuits are the convention-free format: CanonicalStabSum::from_rays consumes
// phase-EXACT AffineStates, and AffineState gate application is phase-exact — executing the
// same prep on |0...0> always reproduces the same global phase. So a branch is fully pinned
// by (prep circuit, coefficient), with the coefficient carrying the branch amplitude
// INCLUDING its relative phase in the prep's phase convention. No tableau-sign or gauge
// convention has to be standardized across writers/readers.
//
// Text format (line-based; '#' lines and blank lines are skipped anywhere):
//   ref-format 2
//   source <benchmark filename, single token>
//   n <deferred qubit count>
//   chi <branch count>
//   # provenance: <free text>
//   anchor
//   <GATE q> | <GATE a b>          tokens: H S SDG X Y Z (1q), CX CZ (2q, a != b)
//   ...
//   endanchor
//   branch <re %.17g> <im %.17g>
//   <GATE q> | <GATE a b>          (suffix D_sigma prep for this branch)
//   ...
//   endbranch
//   ... (exactly chi branch blocks)
//
// v2 semantics: the loader replays anchor_prep once into a base AffineState, then clones
// it per branch and replays Branch.prep (the suffix). With empty anchor_prep the state is
// byte-identical to v1 behavior. Version 1 is rejected (regenerate with current ref_compile).
//
// v3 (structure-preserving, the only written format; v2 still loads via the back-compat reader):
//   ref-format 3
//   source <benchmark filename, single token>
//   n <deferred qubit count>
//   chi <branch count>
//   r <free generator count>
//   # provenance: <free text>
//   anchor
//   <GATE q> | <GATE a b>
//   ...
//   endanchor
//   frame
//   Z <phase> <xw0_hex> [<xw1_hex> ...] <zw0_hex> [<zw1_hex> ...]    (n Zrow lines)
//   ...
//   X <phase> <xw0_hex> [...] <zw0_hex> [...]                         (n Xrow lines)
//   ...
//   endframe
//   free <a0> <a1> ... <a_{r-1}>   (indices into U; one line, space-separated)
//   eps <bit0> <bit1> ... <bit_{n-1}>  (n bits, space-separated, 0/1)
//   branch <re %.17g> <im %.17g> <sigma_hex>   (sigma as hex bitmask, bit d = sigma[d])
//   ...  (exactly chi branch lines, NO endbranch)
//
// v3 semantics: loader replays anchor_prep → anchor, sets U.Zrow/U.Xrow DIRECTLY from
// the stored frame rows (no stabilizer_generators, no from_generators, no from_rays).
// eps/free/sigma are set directly. The sigma bitmask encodes the r-bit sigma vector.
//
// v4 (the production write format — written FROM the framed state, anchor block retired):
//   identical to v3 EXCEPT (a) the header says "ref-format 4", (b) there is NO anchor /
//   endanchor block (the frame block follows the provenance comment directly), and (c) the
//   branch coefficients are the framed state's coefficients AS-IS. v3 synthesized an
//   anchor-prep circuit and folded its arbitrary synthesis phase into every coefficient
//   equally, purely so the reloaded affine (CSS) state bit-matched the constructed one; no
//   consumer needs that (every overlap consumer takes |overlap|), so v4 drops the machinery.
//   Frame rows / free / eps / sigma encoding: byte-identical to v3. v4 files are loaded
//   FRAMED-ONLY (framed_from_ref); ref_to_state rejects them fail-loud. v2/v3 readers stay.
//
// Parsing is FAIL-LOUD: every reject carries a line-numbered message; the loader
// additionally rejects physics-level breakage (duplicate/parallel rays via the
// CanonicalStabSum invariants, non-unit norm).
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/stab_affine.hpp"

namespace qeccore {

struct RefFile {
    int version = 2;
    std::string source;                 // source benchmark filename
    int n = 0;                          // DEFERRED qubit count
    int chi = 0;
    int r = 0;                          // v3: free generator count (0 in v2 files)
    std::string provenance;             // free-text provenance (written as a '#' comment)
    // op: 0:H 1:S 2:SDG 3:X 4:Y 5:Z 6:CX 7:CZ ; b = -1 for 1q gates.
    struct PrepGate { int op; int a; int b; };
    struct Branch {
        std::complex<double> coeff;     // amplitude incl. relative phase (prep convention)
        std::vector<PrepGate> prep;     // v2 only: suffix D_sigma prep from anchor state
        uint64_t sigma_mask = 0;        // v3 only: bit d = sigma[d] (r <= 63 guaranteed by chi <= 2^63)
    };
    std::vector<Branch, StateAllocator<Branch>> branches;   // size == chi (state-budgeted)
    std::vector<PrepGate> anchor_prep;  // v2/v3: shared prep for |anchor>
    // v3-only fields:
    // Frame tableau: Zrow[0..n-1] then Xrow[0..n-1] (size == 2n when version==3, 0 for v2).
    // Each Pauli encoded as {phase, x words (ceil(n/64)), z words (ceil(n/64))}.
    struct StoredPauli {
        int phase = 0;
        std::vector<uint64_t> x;
        std::vector<uint64_t> z;
    };
    std::vector<StoredPauli> frame_rows;  // Zrow[0..n-1] then Xrow[0..n-1] (size 2n)
    std::vector<int> free_gens;           // free generator indices (size == r)
    std::vector<uint8_t> eps;             // anchor signs, size == n (0/1)
};

// Apply one prep gate to a (phase-exact) AffineState.
void apply_prep_gate(AffineState& s, const RefFile::PrepGate& g);

// ── writer ────────────────────────────────────────────────────────────────────────────
std::string ref_to_text(const RefFile& rf);
// Plain write (no tmp/rename — atomicity is the caller's policy, see ref_compile).
bool write_ref_file(const RefFile& rf, const std::string& path, std::string* err);

// v4 writer — the PRODUCTION write path (Phase B3): serialize a FramedSuperposition directly.
// Frame rows (U.Zrow then U.Xrow), free, eps and per-branch (sigma bitmask, coefficient) are
// read verbatim from the framed state; there is NO anchor block and NO phase folding — the
// stored coefficients are the framed state's coefficients AS-IS, so framed_from_ref of the
// written file reproduces the input FIELD-FOR-FIELD (byte-lossless round trip).
RefFile ref_from_state_v4(const FramedSuperposition& state, const std::string& source,
                          const std::string& provenance);

// ── parser / loader ───────────────────────────────────────────────────────────────────
struct RefParseResult {
    bool ok = false;
    RefFile file;
    std::string error;                  // "line N: ..." on failure
};
RefParseResult parse_ref_text(const std::string& text);
RefParseResult parse_ref_file(const std::string& path);

struct LoadedRef {
    bool ok = false;
    std::string error;
    CanonicalStabSum state;
    LoadedRef() : state(0) {}
};
// Reconstruct a CanonicalStabSum from a RefFile. v3 sets U.Zrow/Xrow, eps, free and
// per-branch sigma directly from the frame rows — NO from_rays. v2 (back-compat) executes
// each branch prep on an AffineState and assembles via from_rays, rejecting (fail-loud) on
// invariant violations (duplicate/parallel rays) and non-unit norm. v4 files carry NO anchor
// block, so there is no CSS (affine-anchored) reconstruction: ref_to_state REJECTS version 4
// fail-loud — load v4 framed via framed_from_ref / framed_from_ref_validated.
LoadedRef ref_to_state(const RefFile& rf);

// Lean bare-state sourcing: build a FramedSuperposition directly from a RefFile, SKIPPING the affine anchor
// (the lean rep ignores the anchor). For a v3 RefFile this performs the SAME field copies as
// ref_to_state's v3 loader (frame_rows -> U.Zrow/Xrow, eps -> eps, free_gens -> free, per-branch
// sigma_mask -> branches[i].sigma over rf.r bits, coeff -> branches[i].c, then U.invalidate_dual())
// but into a FramedSuperposition. It is BYTE-IDENTICAL to FramedSuperposition::from_css(ref_to_state(rf).state).
// v4 takes the SAME direct leg (the v3/v4 frame/free/eps/branch encodings are identical; the
// anchor block v3 additionally carries is ignored here anyway). For a v2 RefFile —
// frame_rows/free_gens empty, r==0 — it falls back to that exact from_css expression.
FramedSuperposition framed_from_ref(const RefFile& rf);

// Validated framed load — the shared consumer entry point (run_stim_main --ref, ref_compile
// --verify, xtim's CompiledProgram / ref_verify_cheap). v3/v4: structural-size + sanity-cap +
// unit-norm gates (the validation ref_to_state's v3 leg carried), then framed_from_ref.
// v2: the fully-validating CSS loader + FramedSuperposition::from_css (byte-identical to the
// historical sourcing). Fail-loud: ok=false + message, never a silently-wrong state.
struct LoadedFramedRef {
    bool ok = false;
    std::string error;
    FramedSuperposition state;
    LoadedFramedRef() : state(0) {}
};
LoadedFramedRef framed_from_ref_validated(const RefFile& rf);

// Reconstruct the affine anchor |phi0> of a framed state from the frame ALONE: the unique
// (up to an irrelevant global phase) state with U.Zrow[a]|phi0> = (-1)^{eps[a]}|phi0> for all a.
// Route: signed generators (eps folded into the stored Pauli phase, (-1) = i^2) through the
// existing disentangle_from_generators / invert / prep-replay machinery — no new phase math. ALWAYS-ON verification: every generator sign is re-checked EXACTLY
// (ExactPhase integer compare) on the reconstructed state; a violation throws
// std::runtime_error — that is a REAL phase bug, never loosen the check. Compile/verify-time
// code (O(n^3) synthesis dominates).
AffineState anchor_from_frame(const FramedSuperposition& L);

// materialize_rays overload for the framed rep: anchor via anchor_from_frame, then
// U.Xrow[free[a]] applied per set sigma bit of each amplitude entry (the framed rep stores sigma
// in its Amplitudes entries). The shared global phase of the ray list is the anchor's
// reconstruction phase (arbitrary).
std::vector<std::unique_ptr<AffineState>> materialize_rays(const FramedSuperposition& L);

// Exact <A|B> between two FramedSuperpositions, up to a GLOBAL phase (each side's anchor is
// reconstructed up to phase; every consumer takes std::abs). Same fast monomial fingerprint
// path + dense fallback, feeding the unchanged AffineState inner_product kernel on the
// framed-materialized rays.
std::complex<double> exact_sum_overlap(const FramedSuperposition& A, const FramedSuperposition& B);

}  // namespace qeccore
