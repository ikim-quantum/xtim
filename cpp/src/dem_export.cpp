#include "qeccore/exact_scope.hpp"   // J-b1 P13: the exact-construction counter
#include "qeccore/diag_conjugate.hpp"
#include <cstddef>
#include "qeccore/dem_export.hpp"
#include "qeccore/dem_preamble.hpp"
#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_lean.hpp"
#include "qeccore/xdem_writer.hpp"
#include "qeccore/bare_state_errors.hpp"   // BareStateCapacityError (the lean refusal's reference sign)
#include "qeccore/sampler.hpp"             // noiseless_reference_records

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/dem_classify.hpp"
#include "qeccore/dem_pe_per_shot.hpp"
#include "qeccore/dem_decompose.hpp"
#include "qeccore/fault_record_law.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/record_general.hpp"     // dem-ch: build_record_table_pullbacks (the CH-class fault law)
#include "qeccore/clifford_tableau.hpp"

namespace qeccore {

// The xdem atom census (census only, test API xdem_atom_lines): a deferred alternative `f` of noise instruction `ins`
// as the exact engine's
// forced-fault address (run_stim.cpp compile_exact_branch_program's mapping, inverted): the user position of each
// factor is frame_prov->pos (identity without provenance) and its user letter c the one whose relabelled letter
// frame_prov->letter[3 j + c - 1] is the deferred one. Target / word follow _exact_forced.py: the qubit position
// (1-qubit channels), the pair position with two letters over IXYZ (2-qubit channels), 0 with one letter per site
// qubit (an E / ELSE chain).
static DemLawCensus::Atom census_atom(const Instr& ins, const std::vector<std::pair<int, PauliBasis>>& f,
                                      double p, const std::vector<int32_t>& sig) {
    DemLawCensus::Atom a;
    a.p = p;
    a.sig = sig;
    if (ins.origin == Instr::Origin::ReadoutFold) { a.record = ins.origin_index; return a; }
    if (ins.origin != Instr::Origin::UserSite) { a.site = -2; return a; }
    a.site = ins.origin_index;
    const NoiseFrameProvenance* fp = ins.frame_prov.get();
    const bool two = ins.channel == NoiseChannel::DEPOLARIZE2 || ins.channel == NoiseChannel::PAULI_CHANNEL_2;
    const bool chain = ins.channel == NoiseChannel::CORRELATED;
    std::string word(chain ? ins.qubits.size() : two ? 2 : 1, 'I');
    for (const auto& [qi, pb] : f) {
        const int cd = pb == PauliBasis::X ? 1 : pb == PauliBasis::Y ? 2 : 3;
        const int pos = fp ? fp->pos[(size_t)qi] : qi;
        int cu = cd;
        if (fp) {
            cu = 0;
            for (int c = 1; c <= 3; ++c)
                if (fp->letter[3 * (size_t)qi + (size_t)c - 1] == cd) cu = c;
        }
        const char L = "?XYZ"[cu];
        if (chain) { a.target = 0; word[(size_t)pos] = L; }
        else if (two) { a.target = pos / 2; word[(size_t)(pos & 1)] = L; }
        else { a.target = pos; word[0] = L; }
    }
    a.word = word;
    return a;
}

// --- Observable-magnitude classifier (declared in dem_classify.hpp) ---------------------------
// Q = Dc† P Dc for the diagonal+Pauli Dc = gamma·X^v·Delta(a,B). Q's X-support is P's (Delta is
// diagonal); each S-power / CZ-arm / X^v contributes a leftover Z and/or a sign, folded into Q's
// phase. We track the raw operator i^phi X^x Z^z and let Q.phase carry the result; <Q> on the
// bare state then differs from <P> only by the accumulated ±1 (framed_expectation reads the sign
// as phase deviating from #Y mod 4). Mirrors the ±1-expressibility conjugation algebra in
// dem_export.hpp (the "PAULI_EXPECTATION L-columns" section).
Pauli conjugate_through(const DiagPauliClifford& Dc, const Pauli& P) {
    Pauli Q(P.n);
    Q.x = P.x;            // Delta fixes P's X-support
    Q.z = P.z;            // start from P's Z-content; leftover Z's XOR in below
    int phi = P.phase;    // raw exponent of i in i^phi X^x Z^z (accumulates ±1 -> +2 mod 4)
    // X^v anticommuting with P's Z-content: a -1 per qubit where v_q ∧ z_q (the Pauli part of Dc,
    // not the diagonal Delta — handled here; the Delta rules below are the shared ones). Each loop walks only the
    // set bits of P's Z- (X^v) or X-support (Delta: a no-op elsewhere); bits >= n are zero and the phase adds are
    // mod 4, so the visiting order is immaterial.
    for (size_t w = 0; w < P.z.size(); ++w)
        for (uint64_t bits = P.z[w]; bits; bits &= bits - 1) {
            const int q = (int)(w * 64) + __builtin_ctzll(bits);
            if (q >= P.n) break;
            if (Dc.v[q]) phi += 2;
        }
    // Delta = diag(S^a)·∏CZ conjugation via the shared diag_conjugate rules. S-power and CZ are
    // independent Z-bit XORs + phase adds, so splitting the original interleaved loop is
    // byte-exact; each CZ pair is visited once (i<j) where the helper carries the both-endpoint
    // sign that used to be hand-copied here.
    for (size_t w = 0; w < P.x.size(); ++w)
        for (uint64_t bits = P.x[w]; bits; bits &= bits - 1) {
            const int q = (int)(w * 64) + __builtin_ctzll(bits);
            if (q >= P.n) break;
            diag_spow_conjugate(P, q, Dc.a[q], Q.z.data(), phi);
        }
    // A CZ pair contributes only when an endpoint carries X (diag_cz_conjugate is a no-op
    // otherwise), so walk the CZ rows of P's X-support instead of all n^2/2 pairs. Each pair is
    // still visited exactly once (as (min, max)); Z-bit XORs and mod-4 phase adds commute, so the
    // result is identical to the all-pairs loop (3.1.10, DEM-export speed).
    const int BW = Dc.B.words();
    for (size_t xw = 0; BW > 0 && xw < P.x.size(); ++xw)
        for (uint64_t xbits = P.x[xw]; xbits; xbits &= xbits - 1) {
            const int i = (int)(xw * 64) + __builtin_ctzll(xbits);
            if (i >= P.n) break;
            const uint64_t* row = Dc.B.row(i);
            for (int w = 0; w < BW; ++w) {
                uint64_t bits = row[w];
                while (bits) {
                    const int j = w * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    if (j == i || j >= P.n) continue;
                    if (j < i && P.xbit(j)) continue;      // visited from j's side
                    diag_cz_conjugate(P, std::min(i, j), std::max(i, j), Q.z.data(), phi);
                }
            }
        }
    Q.phase = ((phi % 4) + 4) & 3;
    return Q;
}

LogicalAction classify_logical_action(const FramedSuperposition& bare,
                                      const std::vector<Pauli>& expP,
                                      const std::vector<double>& beta0,
                                      const DiagPauliClifford& Dc) {
    LogicalAction out;
    out.sign_flip.assign(expP.size(), 0);
    for (size_t r = 0; r < expP.size(); ++r) {
        Pauli Q = conjugate_through(Dc, expP[r]);
        auto [pp, pm] = framed_expectation(bare, Q);
        double beta = pp - pm;
        if (std::abs(std::abs(beta) - std::abs(beta0[r])) > 1e-6) out.magnitude_preserved = false;
        if ((beta < 0) != (beta0[r] < 0)) out.sign_flip[r] = 1;
    }
    return out;
}

namespace {

// The sampler's rec_invert (sampler.cpp), per terminal read in record order: the user's `!` read off the coherent
// pre-strip circuit (deferral drops it) XOR the sign eliminate_hadamards stamps on the deferred Measure. Both
// streams hold exactly M Measures (deferral never adds, drops or reorders Measures — the sampler asserts the same);
// a mismatch is an engine bug, so it throws std::logic_error instead of silently misaligning the signs.
std::vector<uint8_t> record_inverts(const Circuit& coherent, const Circuit& deferred, int M) {
    std::vector<uint8_t> rinv((size_t)M, 0);
    for (const Circuit* c : {&coherent, &deferred}) {
        int j = 0;
        for (const Instr& ins : c->stream)
            if (ins.kind == Instr::Kind::Measure) {
                if (j >= M)
                    throw std::logic_error("record_inverts: more Measures than terminal reads (engine bug)");
                if (ins.invert) rinv[(size_t)j] ^= 1;
                ++j;
            }
        if (j != M) throw std::logic_error("record_inverts: fewer Measures than terminal reads (engine bug)");
    }
    return rinv;
}

}  // namespace

namespace {

// Signature = the sorted list of flipped DEM targets: detector d -> id d, observable o ->
// id D + o. Lexicographic vector order == Stim's emission order (D ascending then L).
// (Sig and Mech are now provided by qeccore/dem_decompose.hpp.)

const char* channel_name(const Instr& ins) {
    const char* ch = "?";
    switch (ins.channel) {
        case NoiseChannel::X_ERROR: ch = "X_ERROR"; break;
        case NoiseChannel::Y_ERROR: ch = "Y_ERROR"; break;
        case NoiseChannel::Z_ERROR: ch = "Z_ERROR"; break;
        case NoiseChannel::DEPOLARIZE1: ch = "DEPOLARIZE1"; break;
        case NoiseChannel::DEPOLARIZE2: ch = "DEPOLARIZE2"; break;
        case NoiseChannel::PAULI_CHANNEL_1: ch = "PAULI_CHANNEL_1"; break;
        case NoiseChannel::PAULI_CHANNEL_2: ch = "PAULI_CHANNEL_2"; break;
        case NoiseChannel::CORRELATED: ch = "CORRELATED_ERROR"; break;
    }
    return ch;
}

std::string loc_name(const Instr& ins, int stream_index) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s at deferred stream index %d", channel_name(ins),
                  stream_index);
    return buf;
}

// Stim's disjoint-channel refusal, same wording/shape (stim 1.16 `approximate_disjoint_errors`
// message), naming the instruction; xtim's exact-solve detail follows it.
std::string approximate_disjoint_refusal(const Instr& ins, int stream_index,
                                         const std::string& detail) {
    return std::string("Encountered the operation ") + channel_name(ins) +
           " during error analysis, but this operation requires the "
           "`approximate_disjoint_errors` option to be enabled.\n"
           "If you're calling from python, using xtim.Circuit.detector_error_model, you need "
           "to add the argument approximate_disjoint_errors=True.\n"
           "(xtim: " + loc_name(ins, stream_index) + ": " + detail + ")";
}

// Does Dc act non-Pauli on P: an odd S-power or a CZ-arm on P's X-support (walked by its set bits)?
bool diag_touches_x_support(const DiagPauliClifford& Dc, const Pauli& P) {
    const int BW = Dc.B.words();
    for (size_t xw = 0; xw < P.x.size(); ++xw)
        for (uint64_t bits = P.x[xw]; bits; bits &= bits - 1) {
            const int q = (int)(xw * 64) + __builtin_ctzll(bits);
            if (q >= P.n) break;
            if (Dc.a[q] & 1) return true;                 // odd S-power rotates the column
            for (int w = 0; w < BW; ++w)
                if (Dc.B.row(q)[w]) return true;          // CZ-arm dresses the column with Z
        }
    return false;
}

// Does Dc act NON-Pauli on any declared expectation column? True iff some expP[r] has an odd
// S-power or any CZ-arm of Dc touching its X-support (the same condition the expectation-column
// gate used to refuse on; see the ±1-expressibility condition in dem_export.hpp). When FALSE,
// Dc's action on every declared
// Pauli is a clean ±1 sign (a Pauli frame update) — today's emit-with-sign-flips path is exact
// and no magnitude can change, so the magnitude classifier need not be consulted.
bool touches_expectation_nonpauli(const DiagPauliClifford& Dc, const std::vector<Pauli>& expP) {
    for (const Pauli& P : expP)
        if (diag_touches_x_support(Dc, P)) return true;
    return false;
}

}  // namespace

namespace {

// ── Diagonal-class fallback normalization (DEM export only) ─────────────────────────────────
// DEM export propagates every fault in the DIAGONAL class (Pauli x diagonal Clifford; no PPR
// retry) first. A live-wire MX/MY deferred as
// `H q; CX q a; H q` puts a Hadamard ON the data wire q; a T-gate fault's diagonal residual on q
// (S-power / CZ arm) then meets it mid-stream and leaves the class only because of how the
// deferred stream is written. The fallback rewrites it exactly by the diagonal-copy deferral
// (deferral.hpp), consulted ONLY when the default normalization leaves the class, so every
// circuit the default accepts is byte-identical. (A user-written `H q; <Pauli noise on q>; M q`
// needs no DEM-only rewrite: eliminate_hadamards commutes the pending H THROUGH the noise —
// conjugating the channel — and folds the read into MX on EVERY path, sampler and DEM alike.)
// A circuit still outside the class after the fallback gets the CH-class table (dem-ch, below): the
// samplers' PPR retry, whose general residuals are exported through their pullbacks E†·M·E.

// DemNormalized / dem_normalize / dem_preamble: dem_preamble.hpp (shared with the xdem writer).

// dem-ch: one alternative's end-of-circuit error. D is the diagonal+Pauli form (every alternative of a circuit the
// diagonal table accepts: gen == nullptr, every rule below is the pre-dem-ch one, byte for byte); gen is the composed
// CliffordTableau of an alternative with a CH-class atom (D is then the identity and unread). pull(M) = E† M E in
// either form; fixes(M, s): E† M E = ±M (s = 1 iff −M), else the fault rotates M to another operator.
// D is the export's scratch (AltScratch below): valid until the next compose (its residual_normal_form cache too).
// supp: when non-null, a superset of the wires where D's a / v / CZ rows are nonzero, else D is unrestricted.
struct AltRes {
    const DiagPauliClifford& D;
    std::shared_ptr<const CliffordTableau> gen;
    const std::vector<int>* supp = nullptr;
    Pauli pull(const Pauli& M) const { return gen ? gen->dual_image(M) : conjugate_through(D, M); }
    bool fixes(const Pauli& M, int& sign) const {
        const Pauli Q = pull(M);
        if (Q.x != M.x || Q.z != M.z) return false;
        sign = (((Q.phase - M.phase) & 3) == 2) ? 1 : 0;
        return true;
    }
};

// The per-alternative composition identity.then(atom_1).then(atom_2)... (compose_fired's order) without the O(n)
// (and, with a CZ layer, O(n^2/64)) work per step. When every fired atom is in the sparse CZ + a/v form (the two
// diagonal tables), each `then` is DiagPauliClifford::then applied in place on one n-wide scratch, touching only
// the atom's wires, with the same Z4 arithmetic and gamma in the same order (mul, then add_z8 of the same value):
//   a_j += a2_j (1 - 2 v1_j), and + 2 v1_l per atom CZ pair (j, l) (2·(B2·v1)_j, mod 4); v ^= v2; B ^= B2;
//   cst = sum a2_j v1_j + 2·#{atom CZ pairs inside supp(v1)}  (v1 = the scratch's v BEFORE the step).
// B1 enters only B. An alternative none of whose atoms has a CZ layer (cz_words() == 0: every atom of a Clifford
// circuit, and of a magic one after its last level-3 gate) keeps B the lazy zero; the first atom with one swaps in
// the scratch's materialized rows (B ^= B2 by flip_sym on the atom's pairs) and sets D.cache (active_a, b_pairs:
// the XOR of the atoms' pair lists), so residual_normal_form skips its O(n^2/64) row scan. The scratch is reset on
// the wires (and the pairs) the last alternative touched. A dense-form atom takes the pre-T7 dense chain, byte for
// byte (also the test API's oracle, set_dem_compose_mode_for_test).
class AltScratch {
   public:
    explicit AltScratch(int n)
        : sp_(DiagPauliClifford::identity(n)), dn_(DiagPauliClifford::identity(n)), b_rows_(sym_zeros(n)),
          cache_(std::make_shared<DiagPauliClifford::Cache>()) {}
    static bool sparse_ok(const PropResult& at) { return at.av_sparse() && at.cz_sparse(); }
    // Reset the sparse scratch to the identity (D for a dem-ch alternative too: "the identity and unread").
    const DiagPauliClifford& identity() {
        for (int q : supp_) { sp_.a[(size_t)q] = 0; sp_.v[(size_t)q] = 0; }
        supp_.clear();
        if (!sp_.B.zero_unmaterialized()) {                    // clear the last pairs, park the rows: B lazy zero
            for (const auto& [j, l] : toggles_) sp_.B.flip_sym(j, l);
            std::swap(sp_.B, b_rows_);
            sp_.cache.reset();
        }
        toggles_.clear();
        sp_.gamma = ExactPhase::one();
        return sp_;
    }
    // One sparse step: sp_ = sp_.then(atom) (sparse_ok(atom)).
    void then_sparse(const PropResult& at) {
        long long cst = 0;                                     // Σ a2_j·v1_j + 2 per atom CZ pair inside supp(v1)
        for (const auto& [q, a2] : at.sparse_a()) {
            const int v1 = sp_.v[(size_t)q];
            if (v1) cst += (long long)a2;
            const long long aj = (long long)sp_.a[(size_t)q] + (long long)a2 * (1 - 2 * v1);
            sp_.a[(size_t)q] = (uint8_t)(((aj % 4) + 4) % 4);
            supp_.push_back(q);
        }
        if (at.cz_words() != 0) {
            if (sp_.B.zero_unmaterialized()) {                 // the alternative's first CZ layer
                std::swap(sp_.B, b_rows_);
                if (sp_.B.zero_unmaterialized()) sp_.B.materialize_zero();   // once per export
                sp_.cache = cache_;
            }
            for (const auto& [j, l] : at.cz_pairs()) {
                const int vj = sp_.v[(size_t)j], vl = sp_.v[(size_t)l];
                if (vl) sp_.a[(size_t)j] = (uint8_t)((sp_.a[(size_t)j] + 2) & 3);
                if (vj) sp_.a[(size_t)l] = (uint8_t)((sp_.a[(size_t)l] + 2) & 3);
                if (vj && vl) cst += 2;
                sp_.B.flip_sym(j, l);
                toggles_.push_back({j, l});
                supp_.push_back(j);
                supp_.push_back(l);
            }
        }
        for (int q : at.sparse_v()) { sp_.v[(size_t)q] ^= 1; supp_.push_back(q); }
        sp_.gamma = sp_.gamma.mul(at.prop_gamma());
        sp_.gamma.add_z8((int)(((2 * cst) % 8 + 8) % 8));
    }
    const std::vector<int>* finish_sparse() {
        std::sort(supp_.begin(), supp_.end());
        supp_.erase(std::unique(supp_.begin(), supp_.end()), supp_.end());
        if (sp_.cache) {                                       // finalize()'s cache, from the lists
            DiagPauliClifford::Cache& c = *cache_;
            c.active_a.clear();
            for (int q : supp_) if (sp_.a[(size_t)q]) c.active_a.push_back(q);
            c.b_pairs.clear();
            std::sort(toggles_.begin(), toggles_.end());       // B = the pairs toggled an odd number of times
            for (size_t i = 0; i < toggles_.size();) {
                size_t e = i;
                while (e < toggles_.size() && toggles_[e] == toggles_[i]) ++e;
                if ((e - i) & 1) c.b_pairs.push_back(toggles_[i]);
                i = e;
            }
        }
        return &supp_;
    }
    // The dense chain (the pre-T7 rule): dn_ = identity, then dn_ = dn_.then(atom) per atom.
    void dense_begin() { dn_ = DiagPauliClifford::identity(sp_.n); }
    void then_dense(const PropResult& at) {
        dn_ = (at.cz_sparse() || at.av_sparse()) ? dn_.then(dense_c_prop(at)) : dn_.then(at.c_prop());
    }
    const DiagPauliClifford& dense() const { return dn_; }
    // Test API mode 2: does the sparse result sp_ (with the supp finish_sparse returned) equal the dense chain's dn_
    // field for field? n, gamma, a, v, B by content (the lazy zero reads as zeros), the residual_normal_form cache
    // when set (active_a = the a != 0 wires ascending, b_pairs = finalize()'s pair list), and supp a superset of the
    // wires where a, v or a B row is nonzero.
    bool sparse_equals_dense() const {
        const int n = sp_.n;
        if (dn_.n != n || !sp_.gamma.exact_eq(dn_.gamma) || sp_.gamma.is_zero != dn_.gamma.is_zero ||
            sp_.a != dn_.a || sp_.v != dn_.v)
            return false;
        auto row_word = [](const SymPackedMat& B, int i, int w) -> uint64_t { return w < B.words() ? B.row(i)[w] : 0; };
        const int W = std::max(sp_.B.words(), dn_.B.words());
        std::vector<uint8_t> in_supp((size_t)n, 0);
        for (int q : supp_) in_supp[(size_t)q] = 1;
        for (int i = 0; i < n; ++i) {
            bool row_nz = false;
            for (int w = 0; w < W; ++w) {
                if (row_word(sp_.B, i, w) != row_word(dn_.B, i, w)) return false;
                row_nz |= row_word(dn_.B, i, w) != 0;
            }
            if (!in_supp[(size_t)i] && (dn_.a[(size_t)i] || dn_.v[(size_t)i] || row_nz)) return false;
        }
        if (sp_.cache) {
            DiagPauliClifford f = dn_;
            f.finalize();
            if (sp_.cache->active_a != f.cache->active_a || sp_.cache->b_pairs != f.cache->b_pairs) return false;
        }
        return true;
    }
    bool has_cz() const { return sp_.B.words() != 0; }

   private:
    DiagPauliClifford sp_, dn_;
    std::vector<int> supp_;                                    // the wires sp_ may differ from the identity on
    SymPackedMat b_rows_;                                      // sp_'s CZ rows while sp_.B is the lazy zero
    std::vector<std::pair<int, int>> toggles_;                 // the CZ pairs the alternative flipped (j < l)
    std::shared_ptr<DiagPauliClifford::Cache> cache_;          // sp_.cache while sp_ has a CZ layer
};

// The test API's composition mode and mode 2's counts (set_dem_compose_mode_for_test).
std::atomic<int> g_compose_mode{0};
std::atomic<long long> g_compose_compared{0}, g_compose_mismatched{0}, g_compose_with_cz{0};
// LM-1: certified no-rotation exports, their lazy bare builds, and the test lever that turns the certificate's gate off.
std::atomic<long long> g_cert_exports{0}, g_cert_lazy_builds{0};
std::atomic<bool> g_cert_gate_off{false};

}  // namespace

DemLazyBareCounts dem_lazy_bare_counts_for_test() {
    DemLazyBareCounts c;
    c.certified = g_cert_exports.load(std::memory_order_relaxed);
    c.lazy_builds = g_cert_lazy_builds.load(std::memory_order_relaxed);
    return c;
}
void set_dem_cert_gate_off_for_test(bool on) { g_cert_gate_off.store(on, std::memory_order_relaxed); }

void set_dem_compose_mode_for_test(int mode) {
    if (mode == 2) {
        g_compose_compared.store(0, std::memory_order_relaxed);
        g_compose_mismatched.store(0, std::memory_order_relaxed);
        g_compose_with_cz.store(0, std::memory_order_relaxed);
    }
    g_compose_mode.store(mode, std::memory_order_relaxed);
}
DemComposeCheckCounts dem_compose_check_counts_for_test() {
    DemComposeCheckCounts c;
    c.compared = g_compose_compared.load(std::memory_order_relaxed);
    c.mismatched = g_compose_mismatched.load(std::memory_order_relaxed);
    c.with_cz = g_compose_with_cz.load(std::memory_order_relaxed);
    return c;
}

DemNormalized dem_normalize(const Circuit& user, const NormalizePolicy& policy_in) {
    DemNormalized d;
    // ChannelSupport consumer: a zero-probability noise instruction builds no atom (its factors
    // never fire), so it is transparent to a pending Clifford frame in eliminate_hadamards (the
    // instruction stays in place). Same frame as the twirl samplers' preset.
    NormalizePolicy policy = policy_in;
    policy.zero_noise_transparent = true;
    d.nr = normalize(user, policy);
    d.deferred = std::move(d.nr.normalized);
    // T7: the two diagonal builds in the sparse CZ + a/v form (the twirl sampler's; the Clifford suffix by the sweep):
    // the export reads their atoms only in compose_alt (AltScratch: the lists, or dense_c_prop), so the dense per-atom
    // a / v / support vectors were pure cost. The CH-class build (t3) stays dense (compose_error_tableau reads c_prop).
    d.table = build_propagation_table(d.deferred, /*ppr_retry=*/false, AtomSet::ChannelSupport, /*build_general=*/true,
                                      /*sparse_cz=*/true, /*sparse_av=*/true);   // ppr_retry OFF: the diagonal class first (CH-class: the third build below)
    if (d.table.all_in_class) return d;
    NormalizePolicy alt = policy;
    alt.diagonal_copy = true;
    NormalizeResult nr2 = normalize(user, alt);
    Circuit d2 = std::move(nr2.normalized);
    PropagationTable t2 = build_propagation_table(d2, /*ppr_retry=*/false, AtomSet::ChannelSupport, /*build_general=*/true,
                                                  /*sparse_cz=*/true, /*sparse_av=*/true);
    if (t2.all_in_class) {
        d.nr = std::move(nr2);
        d.deferred = std::move(d2);
        d.table = std::move(t2);
        d.diagonal_copy = true;
        return d;
    }
    // dem-ch: the CH-class fallback — the samplers' class (PPR retry) on the DEFAULT normalization. Reached only when
    // BOTH diagonal builds reject, so every circuit accepted above is byte-identical. An atom a CH leaves non-diagonal
    // carries its end-of-circuit error as a PprResidual; build_general is OFF (no dense n×n tableau per atom): the
    // export materializes one alternative's composed tableau only when it fires such an atom (compose_error_tableau),
    // pulls every record operator back through it (AltRes::pull) and builds its record law in the null form
    // (build_record_table_pullbacks). A circuit still outside the class keeps the DEFAULT normalization's reject.
    PropagationTable t3 = build_propagation_table(d.deferred, /*ppr_retry=*/true, AtomSet::ChannelSupport,
                                                  /*build_general=*/false);
    if (t3.all_in_class) {
        d.table = std::move(t3);
        d.general = true;
    }
    return d;
}

// The default normalization alone, no propagation table: the xdem lean route (LEAN-EXPORT-PLAN Task 3). Exact
// because the caller then requires WriterModel::h_meets_location == false, which implies the default table's
// all_in_class, so dem_normalize would have returned this same normalization with general == false.
DemNormalized dem_normalize_default_only(const Circuit& user, const NormalizePolicy& policy_in) {
    DemNormalized d;
    NormalizePolicy policy = policy_in;
    policy.zero_noise_transparent = true;
    d.nr = normalize(user, policy);
    d.deferred = std::move(d.nr.normalized);
    return d;
}

bool dem_build_bare(DemPreamble& pre, DemExportResult& res) {
    pre.bare = build_bare_state_framed(pre.dn.deferred);
    FramedBareState& bare = pre.bare;
    if (bare.rejected) {
        res.rejected = true;
        res.reject_gate_index = bare.reject_gate_index;
        return false;
    }
    // QW-B: the build leaves the frame's dual stale (lazy dual). The per-detector framed_expectation reads the cached
    // inverse rows when the dual is valid (dual_image, O(n²) per read, otherwise): build it once, here. S2-T4b: on a
    // compact frame nothing is built here (the DEM is serial; its queries build what they read; never an expansion).
    if (bare.state.U.expanded()) bare.state.U.prepare_queries();   // compact: the DEM is serial, its queries build lazily
    return true;
}

bool dem_preamble(const std::string& stim_text, bool include_expectations, DemLawCensus* census,
                  DemPreamble& pre, DemExportResult& res, bool build_bare, bool class_table) {
    pre.p = parse_stim_circuit(stim_text);
    ParsedStim& p = pre.p;
    if (!p.ok()) { res.parse_errors = std::move(p.errors); return false; }
    {   // PROVENANCE (metadata only, as run_stim.cpp): each user noise instruction's site index, so a
        // refusal can name the user site its deferred part came from (support_cap_refusal).
        int site = 0;
        for (Instr& ins : p.circuit.stream)
            if (ins.kind == Instr::Kind::Noise) { ins.origin = Instr::Origin::UserSite; ins.origin_index = site++; }
    }
    // DEM normalization preset (KeepCoherent + defer + eliminate-Hadamards), the sampler's relabel preset.
    // Classically-controlled Pauli feedback whose byproduct crosses a non-Clifford gate is coherentized
    // (as in sampler.cpp); every other (residual) feedback is stripped from the consumer circuit and folded
    // back EXACTLY below (demfb option OC): the reported records are a unitriangular GF(2) image L of the
    // stripped circuit's records, so every record-defined column reads a relabelled record set (L^T). The
    // per-fault map L_f equals L (the Q correction is the identity; checked per alternative, see q_ops).
    NormalizePolicy dem_policy;
    dem_policy.feedback = NormalizePolicy::Feedback::KeepCoherent;
    dem_policy.defer = true;
    pre.dn = class_table ? dem_normalize(p.circuit, dem_policy) : dem_normalize_default_only(p.circuit, dem_policy);
    DemNormalized& dn = pre.dn;
    NormalizeResult& nr = dn.nr;

    // Default normalization, or the diagonal-class fallback when the default leaves the class
    // (dem_normalize above; byte-identical for every circuit the default accepts).
    Circuit& deferred = dn.deferred;
    PropagationTable& table = dn.table;
    if (!table.all_in_class) {
        res.rejected = true;
        res.reject_gate_index = table.rejects.empty() ? -1 : table.rejects[0].reject_gate_index;
        return false;
    }
    if (build_bare && !dem_build_bare(pre, res)) return false;
    // dem-ch: residual classical feedback's per-fault Q correction (q_ops) is derived for diagonal residuals only; a
    // CH-class circuit with feedback is refused by name (it was a class reject before dem-ch — no accepted circuit is lost).
    if (dn.general && circuit_has_feedback(nr.coherent)) {
        res.error = "the circuit has CH-class fault residuals (a fault crossing a controlled-Hadamard) AND classical "
                    "feedback (CX/CY/CZ rec[-k] q); the DEM export supports each alone (refusing)";
        return false;
    }

    // Terminal reads, record order: (pauli idx 0:X 1:Y 2:Z, qubit).
    std::vector<std::pair<int, int>>& reads = pre.reads;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure)
            reads.push_back({ins.basis == PauliBasis::X ? 0
                             : ins.basis == PauliBasis::Y ? 1 : 2, ins.qubits[0]});
    pre.M = (int)reads.size();
    const int M = pre.M;
    {   // deferral places every terminal read on its own wire; the parity-Pauli determinism
        // check below relies on it (non-commuting same-wire reads have no joint Born form)
        std::vector<int> seen;
        for (auto& [b, q] : reads) seen.push_back(q);
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            res.error = "two terminal reads share a wire; the DEM determinism analysis "
                        "does not cover this (refusing)";
            return false;
        }
    }

    // ── Residual classical feedback (demfb option OC). The stripped circuit's raw records y map to the reported
    // records y' = L y (FeedbackFold: ops in record order, y'[flips(P_j)] ^= y'[c_j]). Every record-defined column
    // reads a reported record set S, i.e. the raw set L^T S: detectors, observables, DECISIONs, and each
    // PAULI_EXPECTATION's declared frame (plus the raw functional of its feedback sign, esign).
    // Everything below then runs on the stripped circuit unchanged; a feedback-free circuit skips this block.
    FeedbackPlan& fb = pre.fb;
    FeedbackFold& fbf = pre.fbf;
    if (circuit_has_feedback(nr.coherent)) {
        int nexp = 0;
        for (const Instr& ins : deferred.stream)
            if (ins.kind == Instr::Kind::Observable) ++nexp;
        FeedbackPlanOptions fopt;
        fopt.diagonal_copy = dn.diagonal_copy;
        fb = build_feedback_plan(nr.coherent, M, nexp, fopt);
        if (!fb.ok) {
            // Reachable (the fixplan's audit said otherwise): coherentize decides "crossing" on the USER stream, where
            // a feedback is not a gate, so a residual op whose Pauli reaches the control of a COHERENTIZED feedback is
            // carried by that coherent gate into a non-Clifford gate (e.g. `MX 4; CY rec[-1] 0; ...; M(p) 0;
            // CY rec[-1] 3; T 3`). detector_determinism refuses the same circuit (the twirl sampler, which
            // coherentizes every feedback, samples it); DEM export refuses it by name, pointing at the user's
            // feedback instruction (it refused every feedback circuit before).
            std::string instr = "?";
            int seen = 0;
            for (const Instr& ins : nr.coherent.stream) {
                if (ins.kind != Instr::Kind::ControlledPauli) continue;
                if (seen++ != fb.reject_op) continue;
                const char b = ins.basis == PauliBasis::X ? 'X' : ins.basis == PauliBasis::Y ? 'Y' : 'Z';
                instr = std::string("C") + b + " rec[-" + std::to_string(ins.control_rec_offset) + "] " +
                        std::to_string(ins.qubits.empty() ? -1 : ins.qubits[0]);
                break;
            }
            const std::string which = "the feedback `" + instr + "` (feedback instruction #" +
                                      std::to_string(fb.reject_op) + " in circuit order, REPEAT blocks unrolled)";
            if (fb.reject_kind == 'A')
                res.error = which + " flips a measurement record at or before its own control record (acausal "
                            "feedback); not DEM-expressible (refusing)";
            else
                res.error = which + " is not coherentized (its own qubits meet no non-Clifford gate), but its Pauli "
                            "reaches the control of a feedback that IS coherentized (it crosses a non-Clifford gate), "
                            "and that coherent gate carries it into a non-Clifford gate: it is no longer a Pauli "
                            "correction, so the record relabel cannot represent it (refusing)";
            return false;
        }
        fbf.build(fb, M);
        if (census) {                                    // G2's selection: the targets the fold changes
            auto norm = [](std::vector<int> v) {
                std::sort(v.begin(), v.end());
                std::vector<int> o;
                for (size_t a = 0; a < v.size();) {
                    size_t b = a;
                    while (b < v.size() && v[b] == v[a]) ++b;
                    if ((b - a) & 1) o.push_back(v[a]);
                    a = b;
                }
                return o;
            };
            int D0 = (int)p.detectors.size(), mo = -1;
            for (auto& kv : p.observables) mo = std::max(mo, kv.first);
            for (int d = 0; d < D0; ++d)
                if (fbf.fold(p.detectors[(size_t)d]) != norm(p.detectors[(size_t)d])) census->fb_touched.push_back(d);
            for (auto& kv : p.observables)
                if (fbf.fold(kv.second) != norm(kv.second)) census->fb_touched.push_back(D0 + kv.first);
            int md = -1;
            for (auto& kv : p.decisions) md = std::max(md, kv.first);
            const int base_l = D0 + mo + 1 + (include_expectations ? nexp : 0);
            for (auto& kv : p.decisions)
                if (fbf.fold(kv.second) != norm(kv.second)) census->fb_touched.push_back(base_l + kv.first);
            (void)md;
        }
#ifndef XTIM_MUTATE_DEMFB_NOFOLD
        for (auto& d : p.detectors) d = fbf.fold(d);
        for (auto& kv : p.observables) kv.second = fbf.fold(kv.second);
        for (auto& kv : p.decisions) kv.second = fbf.fold(kv.second);
        int r = 0;
        for (Instr& ins : deferred.stream)
            if (ins.kind == Instr::Kind::Observable) {
                bool oor = false;
                for (int idx : ins.obs_frame) if (idx < 0 || idx >= M) oor = true;
                if (!oor) {                              // out of range: left as is (the frame check refuses below)
                    std::vector<int> f = fbf.fold(ins.obs_frame);
                    const std::vector<int> u = fbf.ctrl_sum(fb.ops.size(), [&](size_t i) {
                        return ((fb.ops[i].esign[(size_t)(r >> 6)] >> (r & 63)) & 1) != 0;
                    });
                    std::vector<int> x;
                    std::set_symmetric_difference(f.begin(), f.end(), u.begin(), u.end(), std::back_inserter(x));
                    if (census) {
                        std::vector<int> o = ins.obs_frame;
                        std::sort(o.begin(), o.end());
                        if (o != x && include_expectations) {
                            int mo = -1;
                            for (auto& kv : p.observables) mo = std::max(mo, kv.first);
                            census->fb_touched.push_back((int)p.detectors.size() + mo + 1 + r);
                        }
                    }
                    ins.obs_frame = std::move(x);
                }
                ++r;
            }
#endif
        if (census) census->fb_ops = (long)fb.ops.size();
    }
    return true;
}

namespace {

// The determinism refusal's text (check_deterministic; the lean route writes the same text from the writer model's
// exact expectation, column_expectation): kind 'D' or 'L', the id, and the noiseless parity expectation <P>.
std::string nondeterministic_message(char kind, int id, double ev, bool sign_unknown = false) {
    // A value that prints as zero prints unsigned: the exact engine's <P> = 0 can come out as -1e-17 ("-0.000000"),
    // which the lean route's exact value cannot reproduce. No in-repo refusal printed "-0.000000" (checked against
    // the oracle hashes); 5 of the 543 in the fast, beta and boundary fuzz families did.
    char buf[160];
    if (sign_unknown) {                      // the sign needs the noiseless reference, above the capacity limit
        std::snprintf(buf, sizeof buf, "non-deterministic %s %c%d (noiseless parity expectation +-%.6f, not "
                      "+-1); not DEM-expressible", kind == 'D' ? "detector" : "observable", kind, id, std::abs(ev));
        return buf;
    }
    if (std::abs(ev) < 5e-7) ev = 0.0;
    std::snprintf(buf, sizeof buf,
                  "non-deterministic %s %c%d (noiseless parity expectation %.6f, not "
                  "+-1); not DEM-expressible",
                  kind == 'D' ? "detector" : "observable", kind, id, ev);
    return buf;
}

// What the xdem export tells its caller (export_dem's ruling-B5 wrapper): whether the text is extended, and how many
// PAULI_EXPECTATION columns the circuit declares (the xdem export runs with include_expectations=false).
struct XdemCore {
    bool extended = false;
    int num_expectations = 0;
    bool lean_refusal = false;   // the lean route's own determinism refusal: the oracle's text, not delegated
    bool det_certified = false;  // LM-1: the lean core certified every detector / observable deterministic
};

// LM-1: a certified export's lazy bare-state build was rejected (never seen: F5); the caller reruns the eager export.
struct LazyBareRejected {};

// Some user gate is non-Clifford (T, T_DAG, CS, CS_DAG, CCZ, CH). A parse error: false (the Stim export names it).
bool has_nonclifford_gate(const std::string& stim_text) {
    const ParsedStim p = parse_stim_circuit(stim_text);
    if (!p.ok()) return false;
    for (const Instr& ins : p.circuit.stream)
        if (ins.kind == Instr::Kind::Gate && (ins.gate == GateKind::T || ins.gate == GateKind::CS ||
                                              ins.gate == GateKind::CCZ || ins.gate == GateKind::CH))
            return true;
    return false;
}

DemExportResult export_dem_impl(const std::string& stim_text,
                                bool include_expectations,
                                bool decompose_errors,
                                bool ignore_decomposition_failures,
                                bool drop_gauge_observables,
                                const std::vector<int>* trusted_detectors,
                                const std::vector<int>* trusted_observables,
                                double approximate_disjoint_errors,
                                int clifford_cutoff,
                                DemLawCensus* census,
                                bool erasure,
                                DemFormat format,
                                bool require_canonical,
                                XdemCore* xc,
                                bool lean,
                                bool cert = false);

}  // namespace

DemExportResult export_dem(const std::string& stim_text, bool include_expectations, bool decompose_errors,
                           bool ignore_decomposition_failures, bool drop_gauge_observables,
                           const std::vector<int>* trusted_detectors, const std::vector<int>* trusted_observables,
                           double approximate_disjoint_errors, int clifford_cutoff, DemLawCensus* census,
                           bool erasure, DemFormat format, bool require_canonical) {
    return export_dem_route(stim_text, include_expectations, decompose_errors, ignore_decomposition_failures,
                            drop_gauge_observables, trusted_detectors, trusted_observables,
                            approximate_disjoint_errors, clifford_cutoff, census, erasure, format, require_canonical,
                            XdemRoute::Lean, nullptr);
}

DemExportResult export_dem_route(const std::string& stim_text, bool include_expectations, bool decompose_errors,
                                 bool ignore_decomposition_failures, bool drop_gauge_observables,
                                 const std::vector<int>* trusted_detectors,
                                 const std::vector<int>* trusted_observables, double approximate_disjoint_errors,
                                 int clifford_cutoff, DemLawCensus* census, bool erasure, DemFormat format,
                                 bool require_canonical, XdemRoute route, bool* lean_used,
                                 std::string* lean_exception) {
    if (lean_used) *lean_used = false;
    if (lean_exception) lean_exception->clear();
    auto ordinary = [&](DemFormat f, bool rc, DemLawCensus* cen, bool cert = false) {
        return export_dem_impl(stim_text, include_expectations, decompose_errors, ignore_decomposition_failures,
                               drop_gauge_observables, trusted_detectors, trusted_observables,
                               approximate_disjoint_errors, clifford_cutoff, cen, erasure, f, rc, nullptr,
                               /*lean=*/false, cert);
    };
    if (format != DemFormat::Xdem) return ordinary(format, require_canonical, census);
    // format="xdem" (controller ruling B5): no non-Clifford gate -> the ordinary export with the caller's options
    // (SPEC §3.4); otherwise the xdem export with the default options, then: extended -> refuse a set option by name,
    // else the ordinary export with the caller's options (require_canonical is never forwarded).
    if (!has_nonclifford_gate(stim_text)) return ordinary(DemFormat::Stim, false, census);
    auto core = [&](bool lean, XdemCore& c) {
        return export_dem_impl(stim_text, /*include_expectations=*/false, false, false, false, nullptr, nullptr,
                               approximate_disjoint_errors, kDemCliffordCutoffDefault, census, false,
                               DemFormat::Xdem, require_canonical, &c, lean);
    };
    XdemCore xc;
    DemExportResult x;
    bool lean_ok = false;
    if (route == XdemRoute::Lean && census == nullptr) {          // OD4: the census (test API) is oracle-only
        try {
            x = core(true, xc);
            lean_ok = x.ok || xc.lean_refusal;                   // the lean determinism refusal is final
        } catch (const std::exception& e) {                       // OD2: a throw is a non-success too (the oracle
            if (lean_exception) *lean_exception = e.what();        // reproduces the refusal or exception itself);
        } catch (...) {                                           // its message is kept for the test API (I2)
            if (lean_exception) *lean_exception = "unknown exception";
        }
        if (lean_ok) { if (lean_used) *lean_used = true; }
        else { xc = XdemCore{}; x = core(false, xc); }             // OD2: the oracle answers every non-success
    } else {
        x = core(false, xc);
    }
    if (!x.ok) return x;
    if (!xc.extended) {
        // LM-1: no rotating line. A lean result carries the exact determinism certificate (every detector and
        // observable deterministic): the ordinary export then skips its bare-state gate and builds the bare state only
        // on a first real read (same code writes the same text). Owner ruling 2026-11-23: such a circuit whose bare
        // state would be over capacity exports. A lazy build the frame rejects (never seen) reruns the eager export.
        if (lean_ok && xc.det_certified) {
            try {
                return ordinary(DemFormat::Stim, false, nullptr, /*cert=*/true);
            } catch (const LazyBareRejected&) {
            }
        }
        return ordinary(DemFormat::Stim, false, nullptr);
    }
    const char* bad = decompose_errors ? "decompose_errors=True (an xdem line is a fault, not a graphlike edge)"
                    : ignore_decomposition_failures ? "ignore_decomposition_failures=True (an xdem line is a fault, "
                                                      "not a graphlike edge)"
                    : erasure ? "erasure=True (the xdem format carries Clifford-residue laws itself)"
                    : drop_gauge_observables ? "drop_gauge_observables=True (every declared observable is a column "
                                               "of the xdem law)"
                    : (trusted_detectors || trusted_observables) ? "trusted_detectors / trusted_observables (the "
                                               "xdem export runs its own determinism gate)"
                    : clifford_cutoff != kDemCliffordCutoffDefault ? "clifford_cutoff (nothing is cut in the xdem "
                                               "format; leave it unset)"
                    : nullptr;
    DemExportResult r;
    if (bad) {
        r.error = std::string("format=\"xdem\" does not take ") + bad + " (refusing)";
        return r;
    }
    if (include_expectations && xc.num_expectations > 0) {
        r.error = "format=\"xdem\" with include_expectations=True: the circuit declares " +
                  std::to_string(xc.num_expectations) + " PAULI_EXPECTATION column(s) and has rotating faults; "
                  "an extended xdem covers detectors and observables only (pass include_expectations=False to "
                  "export without them; refusing)";
        return r;
    }
    return x;
}

namespace {

DemExportResult export_dem_impl(const std::string& stim_text,
                                bool include_expectations,
                                bool decompose_errors,
                                bool ignore_decomposition_failures,
                                bool drop_gauge_observables,
                                const std::vector<int>* trusted_detectors,
                                const std::vector<int>* trusted_observables,
                                double approximate_disjoint_errors,
                                int clifford_cutoff,
                                DemLawCensus* census,
                                bool erasure,
                                DemFormat format,
                                bool require_canonical,
                                XdemCore* xc,
                                bool lean_in,
                                bool cert_in) {
    note_exact_construction();                   // J-b1 P13
    DemExportResult res;
    const bool is_xdem = format == DemFormat::Xdem;  // the skeleton route (the xdem writer model)
    const bool lean = lean_in && is_xdem;           // LEAN-EXPORT-PLAN: the lean route
    // LM-1: the ordinary export of a lean no-rotation result, every detector / observable certified deterministic by
    // the lean core (export_dem_route): no eager bare state; cert_gate skips the bare-state determinism gate (the test
    // lever turns it off, forcing a lazy build).
    const bool cert = cert_in && !is_xdem;
    const bool cert_gate = cert && !g_cert_gate_off.load(std::memory_order_relaxed);
    if (cert) g_cert_exports.fetch_add(1, std::memory_order_relaxed);
    if (require_canonical && !is_xdem) {
        res.error = "require_canonical=True is a format=\"xdem\" option (refusing)";
        return res;
    }
    std::unique_ptr<ParsedStim> src;                // xdem: the source-tracked parse (canonical check, t_c, Gl)
    if (clifford_cutoff < kDemCliffordCutoffExact) {
        res.error = "clifford_cutoff must be >= 0 (or -1 = exact), got " + std::to_string(clifford_cutoff);
        return res;
    }
    const int tcut = clifford_cutoff;            // < 0: exact (every pattern)
    if (erasure && decompose_errors) {           // dem-erasure-out: an erasure line is a law, not a mechanism
        res.error = "erasure output cannot be combined with decompose_errors (an erasure line is the fault's law, "
                    "not a mechanism to decompose)";
        return res;
    }
    DemPreamble pre;
    if (!dem_preamble(stim_text, include_expectations, census, pre, res, /*build_bare=*/!is_xdem && !cert,
                      /*class_table=*/!lean))
        return res;   // lean: the default normalization only, no propagation table (Task 3; exact by the gate below)
    // xdem (ruling R9): the writer model and its diagonality check BEFORE the bare-state build (a non-diagonal residue
    // is refused by name; the bare-state build would refuse it first, generically), then, on the oracle route only,
    // the bare state for the determinism gate below (the lean route decides determinism algebraically:
    // nondeterministic_columns).
    xdem::WriterModel wm;
    if (is_xdem) {
        try {
            wm = xdem::build_writer_model(pre);
        } catch (const xdem::WriterError& e) {
            res.error = e.what();
            return res;
        }
        if (!wm.diag_violations.empty()) {
            res.error = "xdem writer: a non-Clifford gate's residue does not stay diagonal: " + wm.diag_violations[0] +
                        (wm.diag_violations.size() > 1
                             ? " (and " + std::to_string(wm.diag_violations.size() - 1) + " more such pair(s))" : "") +
                        "; the format covers circuits whose location Z's never reach a later location as X/Y "
                        "(refusing)";
            return res;
        }
        // F2 / OD4 (Task 3): with no H meeting a location's forwarded Z, the default table is in class, so the
        // oracle's dem_normalize picks this same normalization (general == false); otherwise delegate to the oracle.
        if (lean && wm.h_meets_location) { res.error = xdem::kXdemLeanDelegate; return res; }   // F2 / OD4
        if (wm.header.groups.empty()) {   // nothing to rotate: the caller returns the ordinary export
            res.ok = true;
            return res;
        }
        if (!lean && !dem_build_bare(pre, res)) return res;   // lean: no bare state (Task 1)
    }
    ParsedStim& p = pre.p;
    DemNormalized& dn = pre.dn;
    NormalizeResult& nr = dn.nr;
    Circuit& deferred = dn.deferred;
    PropagationTable& table = dn.table;
    // The bare state's frame and superposition. Built by now except on a certified export (LM-1), where the first
    // read builds it (counted; a frame rejection reruns the eager export, LazyBareRejected).
    bool bare_ready = !cert;
    auto bare_st = [&]() -> decltype(pre.bare.state)& {
        if (!bare_ready) {
            g_cert_lazy_builds.fetch_add(1, std::memory_order_relaxed);
            DemExportResult scratch;
            if (!dem_build_bare(pre, scratch)) throw LazyBareRejected{};
            bare_ready = true;               // only after a successful build (a throw leaves it unbuilt: retried)
        }
        return pre.bare.state;
    };
    std::vector<std::pair<int, int>>& reads = pre.reads;
    const int M = pre.M;
    FeedbackPlan& fb = pre.fb;

    const int D = (int)p.detectors.size();
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    const int O = max_obs + 1;
    std::vector<std::vector<int>> obs((size_t)O);
    for (auto& kv : p.observables) obs[kv.first] = kv.second;

    // PAULI_EXPECTATION L-columns: the declared Paulis on the deferred wires, in
    // declaration order (the Observable instructions survive deferral in stream order —
    // the exact list the sampler's expectation channel evaluates). Column r is DEM
    // target D+O+r, emitted as L(O+r): observables first, then expectations (THE
    // ordering contract, see the header).
    // Build expP UNCONDITIONALLY (the classifier consumes it even when include_expectations is
    // false); include_expectations gates only the later EMISSION of expectation columns.
    std::vector<Pauli> expP;
    std::vector<std::vector<int>> col_frame;   // each column's declared frame records (parity, sorted)
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Observable) {
            Pauli P(deferred.n);
            for (const PauliTerm& t : ins.obs) {
                if (t.p == PauliBasis::X) { P.setx(t.qubit); }
                else if (t.p == PauliBasis::Y) { P.setx(t.qubit); P.setz(t.qubit); P.phase += 1; }
                else { P.setz(t.qubit); }
            }
            P.phase &= 3;                    // Hermitian Pauli: i^{#Y} X^x Z^z (#Y mod 4)
            // Frame fold (3.1.10). The column the sampler reports is (-1)^{parity of the declared
            // frame records} <P>. In deferred space each frame record j is the terminal read of
            // the Pauli R_j on its own wire, so the reported value is the expectation of the
            // frame-folded operator P' = P · Π_{j in frame} R_j (duplicates cancel). Every DEM rule
            // below (the ±1 sign flip, the magnitude classifier, the completeness gate, beta0) is
            // a statement about the REPORTED column, hence about P'. Before 3.1.10 the rules saw P
            // alone: a gate fault acting on the column only through a frame record (Z before MX of
            // a Bell half) was dropped, and a magic fault that rotates a frame read was neither
            // emitted nor refused. (The readout path below already folded the frame via emask.)
            std::vector<int> fpar;
            for (int idx : ins.obs_frame) {
                if (idx < 0 || idx >= M) {
                    res.error = "a PAULI_EXPECTATION declared frame references an out-of-range "
                                "record";
                    return res;
                }
                auto it = std::find(fpar.begin(), fpar.end(), idx);
                if (it == fpar.end()) fpar.push_back(idx); else fpar.erase(it);
            }
            std::sort(fpar.begin(), fpar.end());
            for (int j : fpar) {
                const auto& [b, q] = reads[(size_t)j];
                Pauli Rj(deferred.n);
                if (b == 0) { Rj.setx(q); }
                else if (b == 1) { Rj.setx(q); Rj.setz(q); Rj.phase = 1; }
                else { Rj.setz(q); }
                P = Pauli::multiply(P, Rj);
            }
            if (!fpar.empty()) {
                int ny = 0;
                for (int q = 0; q < deferred.n; ++q) ny += (P.xbit(q) && P.zbit(q)) ? 1 : 0;
                if (((P.phase - ny) & 1) != 0) {   // anti-Hermitian: a frame read anticommutes with P
                    res.error = "a PAULI_EXPECTATION declared frame record anticommutes with the "
                                "declared Pauli in deferred space; the frame-folded column has no "
                                "DEM form (refusing)";
                    return res;
                }
            }
            expP.push_back(std::move(P));
            col_frame.push_back(fpar);
        }
    const int R = include_expectations ? (int)expP.size() : 0;   // 0 -> no expectation columns emitted
    if (xc) xc->num_expectations = (int)expP.size();
    if (lean && !expP.empty()) { res.error = xdem::kXdemLeanDelegate; return res; }   // OD4: PAULI_EXPECTATION
    // Baseline logical-observable expectations <P_r> on the noiseless bare state.
    std::vector<double> beta0(expP.size(), 0.0);
    for (size_t r = 0; r < expP.size(); ++r) {
        auto [pp, pm] = framed_expectation(bare_st(), expP[r]);
        beta0[r] = pp - pm;
    }

    // Columns with no DEM concept (noiseless per-shot value varies; filled before the channel walk).
    std::vector<uint8_t> col_dropped(expP.size(), 0);

    // ── Module extension: DECISION logical-flip columns (OUTPUT_QUBITS adds no DEM column) ────
    // Decisions: indexed by decision number 0..NDEC-1.
    int max_dec_idx = -1;
    for (auto& kv : p.decisions) if (kv.first > max_dec_idx) max_dec_idx = kv.first;
    const int NDEC = max_dec_idx + 1;
    if (is_xdem && NDEC > 0 && !wm.header.groups.empty()) {
        res.error = "format=\"xdem\": the circuit declares DECISION columns and has non-Clifford gates; xdem gives a "
                    "law to detectors and observables only (DECISION columns are written only for Clifford circuits, "
                    "as plain flips; refusing)";
        return res;
    }
    std::vector<std::vector<int>> dec_recs((size_t)(NDEC > 0 ? NDEC : 1));
    for (auto& kv : p.decisions) dec_recs[(size_t)kv.first] = kv.second;

    // Per-channel (detector then observable) read-inclusion masks; duplicates cancel (parity).
    auto inclusion = [&](const std::vector<int>& recs, std::vector<uint8_t>& inc) -> bool {
        inc.assign((size_t)M, 0);
        for (int idx : recs) {
            if (idx < 0 || idx >= M) return false;
            inc[idx] ^= 1;
        }
        return true;
    };

    // Circuit-level determinism gate (Stim parity): each detector/observable parity Pauli
    // must have a +-1 expectation on the noiseless bare state.
    // The refusal's printed value: the column's noiseless average AS THE SAMPLER REPORTS IT (detector / observable
    // bits relative to the noiseless reference record R, Stim's reference_sample convention). With the recorded bits
    // r = L u, u_k = m_k ^ inv_k (m = the deferred read outcome, inv = `!` ^ the eliminate_hadamards stamp, L the
    // unitriangular feedback relabel, folded into the column's record set S'), and R = L u_ref with
    // u_ref_k = b_k ^ inv_k (b = the reference's deferred read outcomes), the sampled average is
    // E (-1)^{xor_S (r ^ R)} = (-1)^{xor_{S'} b_k} <prod_{S'} P_k>: inv cancels, only b's parity remains. ev is
    // <prod_{S'} P_k> on the deferred circuit; 0 needs no sign. b needs the bare state: the lean route builds it here
    // (the refusal path only). When the bare state (lean) or the reference run (either route) is over the capacity
    // limit or the memory budget (BareStateCapacityError), the sign is unknown (ref_state -1: printed +-|ev|).
    std::vector<int> ref_b;
    int ref_state = 0;                       // 0 not tried, 1 ok, -1 unavailable (capacity)
    auto sampled_value = [&](const std::vector<uint8_t>& incm, double ev) -> double {
        if (std::abs(ev) < 5e-7) return 0.0;
        if (ref_state == 0) {
            ref_state = -1;
            bool have = !lean;
            if (lean) {
                DemExportResult scratch;
                try { have = dem_build_bare(pre, scratch); } catch (const BareStateCapacityError&) { have = false; }
            }
            if (have) {
                const std::vector<uint8_t> rinv = record_inverts(nr.coherent, deferred, M);   // the sampler's rec_invert
                try {                                     // the reference's kernel scratch is budget-charged
                    (void)noiseless_reference_records(bare_st(), deferred, reads, rinv,
                                                      fb.has_feedback ? &fb : nullptr, &ref_b);
                    ref_state = (int)ref_b.size() == M ? 1 : -1;
                } catch (const BareStateCapacityError&) {   // capacity / memory budget only: the sign is unknown
                    ref_b.clear();
                    ref_state = -1;
                }
            }
        }
        if (ref_state < 0) return ev;        // the caller prints +-|ev| (ref_state < 0)
        int par = 0;
        for (int j = 0; j < M; ++j) if (incm[(size_t)j] && (ref_b[(size_t)j] & 1)) par ^= 1;
        return par ? -ev : ev;
    };
    auto in_range = [&](const std::vector<int>& recs) {   // inclusion's range check alone (no O(M) mask)
        for (int idx : recs)
            if (idx < 0 || idx >= M) return false;
        return true;
    };
    std::vector<uint8_t> inc;
    std::vector<SparsePauli> Qpre;           // S2-T4b: the prefetched U†PU per target (filled below)
    bool last_nondet = false;                // the last check_deterministic failure was the determinism refusal
    auto check_deterministic = [&](const std::vector<int>& recs, char kind, int id,
                                   std::string& err) -> bool {
        last_nondet = false;
        if (!in_range(recs)) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "%c%d references an out-of-range measurement",
                          kind, id);
            err = buf;
            return false;
        }
        if (lean) return true;               // decided algebraically after the fault columns (nondeterministic_columns)
        if (cert_gate) return true;          // LM-1: certified by the lean core (every D / L deterministic)
        // QW-E D2: the prefetched U†PU needs no O(M) inclusion mask and no dense P; the mask is built only when refusing.
        // An empty prefetched slot is the target's empty record parity (seed_of: in range, odd set empty; observable
        // ids are in [0, O), as obs[] indexes them).
        const size_t t = (size_t)(kind == 'D' ? id : D + id);
        std::pair<double, double> pr;
        bool have_inc = false;
        if (t < Qpre.size() && Qpre[t].n > 0) {
            pr = framed_expectation_frame(bare_st(), Qpre[t]);   // the prefetched U†PU (sparse: QW-E D1)
        } else if (t < Qpre.size()) {
            return true;                     // constant parity
        } else {
            (void)inclusion(recs, inc);
            have_inc = true;
            Pauli P(deferred.n);
            int ny = 0, any = 0;
            for (int j = 0; j < M; ++j) {
                if (!inc[j]) continue;
                any = 1;
                const auto& [b, q] = reads[j];
                if (b == 0) { P.setx(q); }
                else if (b == 1) { P.setx(q); P.setz(q); ++ny; }
                else { P.setz(q); }
            }
            if (!any) return true;           // constant parity
            P.phase = ny & 3;
            pr = framed_expectation(bare_st(), P);
        }
        const auto [pp, pm] = pr;
        if (std::max(pp, pm) > 1.0 - 1e-9) return true;
        if (!have_inc) (void)inclusion(recs, inc);
        const double sv = sampled_value(inc, pp - pm);
        err = nondeterministic_message(kind, id, sv, ref_state < 0 && std::abs(sv) >= 5e-7);
        last_nondet = true;
        return false;
    };
    // Trusted ids (see header): determinism certified externally against the TRUE bare state
    // (from_state channel_report); the text-only gate below is skipped for exactly those ids.
    auto is_trusted = [](const std::vector<int>* ids, int id) {
        return ids && std::find(ids->begin(), ids->end(), id) != ids->end();
    };
    // S2-T4b: on a compact frame (or one with pending gates) the gate's U†PU of every detector / observable parity
    // comes from ONE backward sweep (Frame::sweep, bit-sliced) instead of a scan of the frame's gate list per target —
    // the same unique operators (I3), kept sparse until read. An expanded frame keeps the per-target framed_expectation.
    // Index: detector d at d, observable id at D + id; an empty entry (n == 0) is not prefetched.
    if (!lean && !cert_gate && !(bare_st().U.expanded() && bare_st().U.gates().empty())) {
        std::vector<SparsePauli> seeds((size_t)(D + O));
        std::vector<uint8_t> want((size_t)(D + O), 0);
        std::vector<int> odd;
        auto seed_of = [&](const std::vector<int>& recs, SparsePauli& s) {
            odd = recs;                                   // the parity of the record set (inclusion's), O(|recs| log)
            std::sort(odd.begin(), odd.end());
            size_t w = 0;
            for (size_t i = 0; i < odd.size();) {
                if (odd[i] < 0 || odd[i] >= M) return false;
                size_t j = i;
                while (j < odd.size() && odd[j] == odd[i]) ++j;
                if ((j - i) & 1) odd[w++] = odd[i];
                i = j;
            }
            odd.resize(w);
            s.n = deferred.n;
            int ny = 0;
            for (int j : odd) {
                const auto& [b, q] = reads[(size_t)j];
                if (b != 2) s.x.push_back(q);
                if (b != 0) s.z.push_back(q);
                if (b == 1) ++ny;
            }
            if (s.x.empty() && s.z.empty()) return false;
            std::sort(s.x.begin(), s.x.end());
            std::sort(s.z.begin(), s.z.end());
            s.phase = ny & 3;
            return true;
        };
        for (int d = 0; d < D; ++d)
            if (!is_trusted(trusted_detectors, d)) want[(size_t)d] = seed_of(p.detectors[d], seeds[(size_t)d]) ? 1 : 0;
        for (auto& kv : p.observables)
            if (kv.first >= 0 && kv.first < O && !is_trusted(trusted_observables, kv.first))
                want[(size_t)(D + kv.first)] = seed_of(kv.second, seeds[(size_t)(D + kv.first)]) ? 1 : 0;
        std::vector<int> at;
        std::vector<SparsePauli> packed, out;
        for (int t = 0; t < D + O; ++t)
            if (want[(size_t)t]) { at.push_back(t); packed.push_back(std::move(seeds[(size_t)t])); }
        std::vector<SparsePauli>().swap(seeds);
        bare_st().U.sweep(packed, false, out);
        Qpre.assign((size_t)(D + O), SparsePauli{});
        for (size_t i = 0; i < at.size(); ++i) Qpre[(size_t)at[i]] = std::move(out[i]);
    }
    // Detectors must be DEM-expressible (a gauge detector is fatal — no DEM can represent it).
    bool nd_refusal = false;                 // a determinism refusal is set in res.error
    std::string nd_scratch;
    for (int d = 0; d < D; ++d) {
        if (is_trusted(trusted_detectors, d)) {
            if (nd_refusal) continue;
            // Still validate the record references (out-of-range is a structural error the
            // trust cannot waive); only the bare-state expectation gate is skipped.
            if (!in_range(p.detectors[d])) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "D%d references an out-of-range measurement", d);
                res.error = buf; return res;
            }
            continue;
        }
        if (nd_refusal) {                    // refusing already: list the rest (no other error is reported)
            if (!check_deterministic(p.detectors[d], 'D', d, nd_scratch) && last_nondet)
                res.nondeterministic_targets.push_back("D" + std::to_string(d));
            continue;
        }
        if (!check_deterministic(p.detectors[d], 'D', d, res.error)) {
            if (!last_nondet) return res;    // out of range: as before
            nd_refusal = true;               // error names this first one; collect every other (the Python hint)
            res.nondeterministic_targets.push_back("D" + std::to_string(d));
        }
    }
    // Observables: classify each. A non-deterministic (gauge) observable can't be a DEM column;
    // with drop_gauge_observables we DROP it (record + don't emit), else refuse (Stim parity).
    std::vector<uint8_t> obs_dropped((size_t)(O ? O : 1), 0);
    for (auto& kv : p.observables) {
        if (is_trusted(trusted_observables, kv.first)) {
            if (nd_refusal) continue;
            if (!in_range(kv.second)) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "L%d references an out-of-range measurement",
                              kv.first);
                res.error = buf; return res;
            }
            continue;                                                        // trusted -> keep
        }
        std::string oerr;
        if (check_deterministic(kv.second, 'L', kv.first, oerr)) continue;   // deterministic -> keep
        if (nd_refusal) {                    // refusing already: list a refused (not a dropped) observable
            if (last_nondet && !drop_gauge_observables)
                res.nondeterministic_targets.push_back("L" + std::to_string(kv.first));
            continue;
        }
        if (!drop_gauge_observables) {
            res.error = oerr;
            if (!last_nondet) return res;    // out of range: as before
            nd_refusal = true;
            res.nondeterministic_targets.push_back("L" + std::to_string(kv.first));
            continue;
        }
        obs_dropped[kv.first] = 1;
        res.dropped_observables.push_back(kv.first);
    }
    if (nd_refusal) return res;              // the determinism refusal (res.error: the first one, as before)
    // Which terminal reads feed an EMITTED target (detector or non-dropped observable)? Used by the
    // alt_signature expressibility check (next task): a non-Pauli effect on a read feeding only
    // excluded columns (expectations under include_expectations=false, or a dropped gauge
    // observable) is irrelevant. Expectation columns are computed from expP, not from reads[].
    std::vector<uint8_t> read_feeds_emitted((size_t)(M ? M : 1), 0);
    for (int d = 0; d < D; ++d)
        for (int idx : p.detectors[d]) read_feeds_emitted[idx] = 1;
    for (auto& kv : p.observables)
        if (!obs_dropped[kv.first])
            for (int idx : kv.second) read_feeds_emitted[idx] = 1;

    // Per-target record rows (QW-E D4; formerly the dense T x MW tmask / NDEC x MW dmask): each target's record set
    // reduced by parity (a read listed an even number of times cancels, as the masks' XOR did), ascending, in CSR form,
    // plus the transposed read -> targets index. O(sum |recs|) memory. Targets indexed 0..D-1 (detectors) then
    // D..D+O-1 (observables); decisions 0..NDEC-1.
    const int MW = (M + 63) / 64;
    const int T = D + O;
    struct RecRows {
        std::vector<size_t> off{0};
        std::vector<int> idx;
        const int* begin(int t) const { return idx.data() + off[(size_t)t]; }
        const int* end(int t) const { return idx.data() + off[(size_t)t + 1]; }
    };
    std::vector<int> odd_scratch;
    auto add_row = [&](RecRows& rows, const std::vector<int>* recs) {   // nullptr: an empty row
        if (recs) {
            odd_scratch.clear();
            for (int idx : *recs) if (idx >= 0 && idx < M) odd_scratch.push_back(idx);
            std::sort(odd_scratch.begin(), odd_scratch.end());
            for (size_t i = 0; i < odd_scratch.size();) {
                size_t e = i;
                while (e < odd_scratch.size() && odd_scratch[e] == odd_scratch[i]) ++e;
                if ((e - i) & 1) rows.idx.push_back(odd_scratch[i]);
                i = e;
            }
        }
        rows.off.push_back(rows.idx.size());
    };
    auto transpose = [&](const RecRows& rows, int nrows) {   // read j -> the rows reading it, ascending
        std::vector<std::vector<int>> by_read((size_t)(M ? M : 1));
        for (int t = 0; t < nrows; ++t)
            for (const int* it = rows.begin(t); it != rows.end(t); ++it) by_read[(size_t)*it].push_back(t);
        return by_read;
    };

    // ── Module extensions (part 2): decision rows + column-layout constants ──────────────────
    // (Part 1 — dec_recs — is above, before the check_deterministic
    // gate; MW and T needed here were not yet available at that point.)
    RecRows drows;
    for (int k = 0; k < NDEC; ++k) add_row(drows, &dec_recs[(size_t)k]);   // out-of-range reads skipped, as before
    // S2-CLEAN (QW-E review m2): no DECISION column (the common case) -> no transpose (M empty vectors); the index is
    // empty and its two readers skip it.
    const std::vector<std::vector<int>> decisions_fed_by_read =
        NDEC > 0 ? transpose(drows, NDEC) : std::vector<std::vector<int>>();
    // Target-index layout for module columns (beyond the existing D+O+R space):
    //   LOGFL_START  = T + R            — first logical-flip column
    //   T_TOTAL      = LOGFL_START + NDEC
    const int LOGFL_START = T + R;
    const int T_TOTAL     = LOGFL_START + NDEC;

    RecRows trows;                                       // the gate range-checked every emitted target's records
    for (int d = 0; d < D; ++d) add_row(trows, &p.detectors[d]);
    for (int o = 0; o < O; ++o)
        add_row(trows, obs_dropped[o] ? nullptr : &obs[o]);   // dropped -> empty row -> never emitted

    if (census) std::sort(census->fb_touched.begin(), census->fb_touched.end());

    // One alternative's record-flip mask + the per-alternative determinism gate.
    std::vector<uint64_t> mflip((size_t)(MW ? MW : 1), 0);
    // Reads whose propagated effect on a terminal X/Y measurement is a TWIRLED fair coin (an
    // odd S-power rotates the read X<->Y, or a propagated CZ entangles it). By the
    // measurement-twirl theorem (an internal derivation note), such a read's outcome is an
    // independent fair-coin Z (the read flips with probability 1/2): an ordinary Pauli
    // mechanism, not a refusal. We record the read index here (with a CLEAN flip=0 in mflip)
    // and the channel walk below expands over the 2^k fair-coin subsets. (Cleared per
    // alternative.)
    std::vector<int> randomized_reads;
    // Detector/observable part of the signature + the twirled-fair-coin read list. The
    // expectation L-columns are appended by add_alt's router (the Pauli path: clean ±1 sign
    // flips; the magnitude path: la.sign_flip), NOT here — alt_signature is now action-agnostic.
    // T7: the alternative's read flips are found from its wires (AltRes::supp, through read_of_wire: reads are on
    // distinct wires, checked in dem_preamble) and its signature by parity-toggling the targets each flipped read
    // feeds (targets_fed_by_read: the target rows' transpose, so a read listed twice in a target cancels) — the same
    // sorted sig as the record-mask x mflip parity scan, without the O(M + T·MW) work per alternative.
    std::vector<int> read_of_wire((size_t)(deferred.n ? deferred.n : 1), -1);
    for (int j = 0; j < M; ++j) read_of_wire[(size_t)reads[(size_t)j].second] = j;
    const std::vector<std::vector<int>> targets_fed_by_read = transpose(trows, T);
    std::vector<int> flipped_reads;                     // the reads set in mflip (cleared per alternative)
    std::vector<int> sig_toggles;
    auto alt_signature = [&](const AltRes& A, Sig& sig) {
        for (int j : flipped_reads) mflip[(size_t)(j >> 6)] = 0;
        flipped_reads.clear();
        randomized_reads.clear();
        const DiagPauliClifford& Dp = A.D;
        // One read's verdict: -1 randomized (twirled fair coin), else its deterministic flip.
        auto read_rule = [&](int j) -> int {
            const auto& [b, q] = reads[j];
            int flip;
            if (A.gen) {
                // dem-ch: E† R_j E = ±R_j is a deterministic flip (the sign); any other operator is a changed read
                // whose law the record table gives (the diagonal rule below is this one in closed form).
                if (!A.fixes(single_pauli(b, q, deferred.n), flip)) return -1;
            } else if (b == 2) {
                flip = Dp.v[q] ? 1 : 0;
            } else {
                bool randomized = false;
                if (Dp.a[q] & 1) randomized = true;    // odd S-power rotates the X/Y read X<->Y
                if (!randomized)
                    for (int w = 0; w < Dp.B.words(); ++w)
                        if (Dp.B.row(q)[w]) { randomized = true; break; }   // CZ entangles the read
                // Twirled fair coin: this terminal read flips independently w.p. 1/2. Keep its deterministic flip
                // at 0 (mflip untouched) and record it; the channel walk expands the signature over the 2^k
                // subsets of randomized reads.
                if (randomized) return -1;
                flip = (Dp.a[q] == 2) ? 1 : 0;
                if (b == 1 && Dp.v[q]) flip ^= 1;
            }
            return flip;
        };
        auto take = [&](int j) {
            const int f = read_rule(j);
            if (f < 0) randomized_reads.push_back(j);
            else if (f) { mflip[(size_t)(j >> 6)] ^= 1ull << (j & 63); flipped_reads.push_back(j); }
        };
        if (A.supp) {                                   // a read off supp sees a = v = 0 and no CZ: no flip
            for (int q : *A.supp) { const int j = read_of_wire[(size_t)q]; if (j >= 0) take(j); }
            std::sort(randomized_reads.begin(), randomized_reads.end());
        } else {
            for (int j = 0; j < M; ++j) take(j);
        }
        sig.clear();
        sig_toggles.clear();
        for (int j : flipped_reads)
            for (int t : targets_fed_by_read[(size_t)j]) sig_toggles.push_back(t);
        std::sort(sig_toggles.begin(), sig_toggles.end());
        for (size_t i = 0; i < sig_toggles.size();) {   // odd multiplicity <=> odd record-row parity
            size_t e = i;
            while (e < sig_toggles.size() && sig_toggles[e] == sig_toggles[i]) ++e;
            if ((e - i) & 1) sig.push_back(sig_toggles[i]);
            i = e;
        }
    };
    // The clean ±1 expectation-column sign flips for a Pauli-action Dc (no odd-S / CZ on any
    // expP X-support — touches_expectation_nonpauli == false). Targets T+r appended ascending,
    // so sig stays sorted. Only used when R > 0 (include_expectations); mirrors the header
    // derivation: (-1)^(|{q: v_q ∧ z_q}| + |{q∈supp(x): a_q==2}|) per column.
    // SEMANTICS (3.1.10, owner decision): an L-flip on expectation column r means "the fault's
    // propagated action multiplies the frame-folded operator P'_r by -1" (E^dag P' E = -P'), i.e.
    // it multiplies the reported <P> by -1 (with the records relabelled by the fault's record
    // flips). It is a statement about the operator, well defined for ANY noiseless value — 0,
    // 1/sqrt2, or branch-dependent (no frame) — and independent of the state.
    auto pauli_expectation_flip = [&](const AltRes& A, int r) -> int {
        const Pauli& P = expP[(size_t)r];
        int flip = 0;
        if (A.gen) { A.fixes(P, flip); return flip; }   // dem-ch: called only when E† P E = ±P (column_nonpauli false)
        const DiagPauliClifford& Dp = A.D;
        auto at = [&](int q) {
            if (Dp.v[q] && P.zbit(q)) flip ^= 1;       // X^v vs P's Z-content
            if (P.xbit(q) && Dp.a[q] == 2) flip ^= 1;  // even-S Z-layer on P's X-support
        };
        if (A.supp) for (int q : *A.supp) at(q);       // a = v = 0 off supp
        else for (int q = 0; q < deferred.n; ++q) at(q);
        return flip;
    };
    auto append_pauli_expectation_flips = [&](const AltRes& A, Sig& sig) {
        for (int r = 0; r < R; ++r)
            if (!col_dropped[(size_t)r] && pauli_expectation_flip(A, r)) sig.push_back(T + r);
    };
    // Per column: does Dc act non-Pauli (not +-1) on P'_r? (odd S-power or a CZ arm on its X-support)
    auto column_nonpauli = [&](const AltRes& A, int r) -> bool {
        const Pauli& P = expP[(size_t)r];
        if (A.gen) { int s = 0; return !A.fixes(P, s); }   // dem-ch
        return diag_touches_x_support(A.D, P);
    };

    // ── The per-fault Q correction (demfb T3) is the identity: an invariant, checked per alternative. For a fault F
    // AFTER residual op j (a fault before it is fixed before P_j acts: P_j just flips what it anticommutes with),
    // pushing F's end residual C past P_j^{r_cj} leaves Q_j^{r_cj}, Q_j = the group commutator P_j C P_j^dag C^dag
    // (Z-type: {q in X(P_j) : a_q odd} XOR B . X(P_j), up to phase). The commutator of two operators is conjugated
    // by every later gate, so Q_j(end) = U Q_j(t_F) U^dag, and at the fault's creation time t_F both are Paulis
    // (P_j's cone meets no non-Clifford gate - else it was coherentized - so P_j stays a Pauli): Q_j = +-I.
    // q_ops evaluates Q_j at the end of the circuit (the probe's end Pauli against the consumer's residual): a
    // non-identity is a frame / layout disagreement between the two (an engine bug), never a correction.
    // The probe's end Paulis are in the probe deferral's frame; this circuit's eliminate_hadamards folded the last
    // H / SDG;H of a read wire into its read basis, so on a read wire the two end frames differ by the rotation
    // taking the read basis to Z: P_here = G_here^dag G_probe P_probe G_probe^dag G_here (G_X = H: X<->Z;
    // G_Y = H S^dag: X->Y->Z->X). fb_xwires: each op's X-support in THIS frame.
    std::vector<std::vector<int>> fb_xwires;
    std::vector<std::vector<int>> fb_ops_on_wire;       // deferred wire -> ops with X support there (ascending)
    std::vector<int> fb_cp_before;                      // deferred noise ordinal -> #residual ops before it
    if (fb.has_feedback) {
        if ((int)fb.reads.size() != M)
            throw std::logic_error("export_dem: residual feedback: probe read count mismatch (engine bug)");
        std::vector<int> rot((size_t)deferred.n, -1);   // wire -> (probe basis, here basis) code, -1: same frame
        std::vector<int> pbasis((size_t)deferred.n, 2), hbasis((size_t)deferred.n, 2);
        for (int j = 0; j < M; ++j) {
            if (fb.reads[(size_t)j].second != reads[(size_t)j].second)
                throw std::logic_error("export_dem: residual feedback: probe and consumer read wires differ (engine bug)");
            const int q = reads[(size_t)j].second;
            pbasis[(size_t)q] = fb.reads[(size_t)j].first;
            hbasis[(size_t)q] = reads[(size_t)j].first;
            if (pbasis[(size_t)q] != hbasis[(size_t)q]) rot[(size_t)q] = 1;
        }
        auto to_z = [](int b, int& x, int& z) {         // G_b: the read basis b -> Z
            if (b == 0) std::swap(x, z);
            else if (b == 1) { const int x0 = x; x = x ^ z; z = x0; }
        };
        auto from_z = [](int b, int& x, int& z) {       // G_b^-1
            if (b == 0) std::swap(x, z);
            else if (b == 1) { const int x0 = x; x = z; z = x0 ^ z; }
        };
        fb_xwires.assign(fb.ops.size(), {});
        for (size_t i = 0; i < fb.ops.size(); ++i) {
            const FeedbackOp& op = fb.ops[i];
            std::vector<int>& xw = fb_xwires[i];
            for (int q : op.xwires)
                if (rot[(size_t)q] < 0) xw.push_back(q);
            std::vector<int> rw;                         // rotated wires touched by P_j
            for (int q : op.xwires) if (rot[(size_t)q] >= 0) rw.push_back(q);
            for (int q : op.zwires) if (rot[(size_t)q] >= 0) rw.push_back(q);
            std::sort(rw.begin(), rw.end());
            rw.erase(std::unique(rw.begin(), rw.end()), rw.end());
            for (int q : rw) {
                int x = std::binary_search(op.xwires.begin(), op.xwires.end(), q) ? 1 : 0;
                int z = std::binary_search(op.zwires.begin(), op.zwires.end(), q) ? 1 : 0;
                to_z(pbasis[(size_t)q], x, z);
                from_z(hbasis[(size_t)q], x, z);
                if (x) xw.push_back(q);
            }
            std::sort(xw.begin(), xw.end());
        }
        fb_ops_on_wire.assign((size_t)deferred.n, {});
        for (size_t i = 0; i < fb.ops.size(); ++i)
            for (int q : fb_xwires[i])
                if (q >= 0 && q < deferred.n) fb_ops_on_wire[(size_t)q].push_back((int)i);
        // A deferred noise instruction's position relative to the residual ops, by its PROVENANCE tag (user site, or
        // the readout fold of a record): normalization may split one noise instruction into several (a pending
        // Hadamard frame on part of its qubits), so positions do not line up, tags do.
        int ncp = 0;
        std::map<std::pair<int, int>, int> cp_at;           // (origin, origin_index) -> #residual ops before it
        for (const Instr& ins : nr.coherent.stream) {
            if (ins.kind == Instr::Kind::ControlledPauli) { ++ncp; continue; }
            if (ins.kind != Instr::Kind::Noise) continue;
            if (ins.origin == Instr::Origin::None)
                throw std::logic_error("export_dem: residual feedback: an untagged noise instruction (engine bug)");
            const auto key = std::make_pair((int)ins.origin, ins.origin_index);
            auto [it, fresh] = cp_at.emplace(key, ncp);
            if (!fresh && it->second != ncp)
                throw std::logic_error("export_dem: residual feedback: one noise provenance tag on both sides of a "
                                       "feedback (engine bug)");
        }
        if (ncp != (int)fb.ops.size())
            throw std::logic_error("export_dem: residual feedback: op count mismatch (engine bug)");
        for (const Instr& ins : deferred.stream) {
            if (ins.kind != Instr::Kind::Noise) continue;
            auto it = cp_at.find(std::make_pair((int)ins.origin, ins.origin_index));
            if (ins.origin == Instr::Origin::None || it == cp_at.end())
                throw std::logic_error("export_dem: residual feedback: a deferred noise instruction without a known "
                                       "provenance tag (engine bug)");
            fb_cp_before.push_back(it->second);
        }
    }
    struct QOp { int op; std::vector<int> qz; };       // op index, Z-support wires of Q_j (ascending)
    // The ops with Q_j != I for a fault of residual Dp at deferred noise ordinal li.
    auto q_ops = [&](const DiagPauliClifford& Dp, int li) {
        std::vector<QOp> out;
        if (!fb.has_feedback) return out;
#ifndef XTIM_MUTATE_DEMFB_QBLIND
        const int before = fb_cp_before[(size_t)li];
#else
        const int before = (int)fb.ops.size(); (void)li;   // mutation: the fixplan's ordering-blind Q
#endif
        if (before == 0) return out;
        std::vector<int> act;                              // wires where C is non-Pauli
        const int BW = Dp.B.words();
        for (int q = 0; q < Dp.n; ++q) {
            bool a = (Dp.a[(size_t)q] & 1) != 0;
            if (!a && BW > 0) {
                const uint64_t* br = Dp.B.row(q);
                for (int w = 0; w < BW && !a; ++w) a = br[w] != 0;
            }
            if (a) act.push_back(q);
        }
        if (act.empty()) return out;
        std::vector<int> cand;
        for (int q : act)
            for (int i : fb_ops_on_wire[(size_t)q]) {
                if (i >= before) break;
                cand.push_back(i);
            }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        const int NW = (Dp.n + 63) / 64;
        std::vector<uint64_t> z((size_t)NW);
        for (int i : cand) {
            std::fill(z.begin(), z.end(), 0);
            for (int q : fb_xwires[(size_t)i]) {
                if (Dp.a[(size_t)q] & 1) z[(size_t)(q >> 6)] ^= 1ULL << (q & 63);
                if (BW > 0) {
                    const uint64_t* br = Dp.B.row(q);
                    for (int w = 0; w < BW && w < NW; ++w) z[(size_t)w] ^= br[w];
                }
            }
            QOp qo{i, {}};
            for (int w = 0; w < NW; ++w)
                for (uint64_t b = z[(size_t)w]; b; b &= b - 1) {
                    const int q = w * 64 + __builtin_ctzll(b);
                    if (q < Dp.n) qo.qz.push_back(q);
                }
            if (!qo.qz.empty()) out.push_back(std::move(qo));
        }
        return out;
    };

    // Logical-flip flips for DECISION columns (targets LOGFL_START .. LOGFL_START+NDEC-1).
    // Decision k flips iff the mechanism's mflip has odd overlap with decision k's record mask.
    // This is computed exactly like the det/obs parity from mflip + the record rows above.
    auto append_logical_flip_flips = [&](Sig& sig) {
        for (int k = 0; k < NDEC; ++k) {
            int par = 0;
            for (const int* it = drows.begin(k); it != drows.end(k); ++it)
                par ^= (int)((mflip[(size_t)(*it >> 6)] >> (*it & 63)) & 1);
            if (par) sig.push_back(LOGFL_START + k);
        }
    };
    // The targets (ascending), then the decision columns (ascending), that read one of the randomized reads, with their
    // record rows: the rows with a hit against the randomized-read mask, without a scan of every row.
    std::vector<int> rt_scratch;
    auto randomized_targets = [&](std::vector<int>& rt, std::vector<std::pair<const int*, const int*>>& rt_rows) {
        rt.clear();
        rt_rows.clear();
        for (int pass = 0; pass < (NDEC > 0 ? 2 : 1); ++pass) {
            const std::vector<std::vector<int>>& fed = pass ? decisions_fed_by_read : targets_fed_by_read;
            rt_scratch.clear();
            for (int j : randomized_reads) rt_scratch.insert(rt_scratch.end(), fed[(size_t)j].begin(), fed[(size_t)j].end());
            std::sort(rt_scratch.begin(), rt_scratch.end());
            rt_scratch.erase(std::unique(rt_scratch.begin(), rt_scratch.end()), rt_scratch.end());
            for (int t : rt_scratch) {
                const RecRows& rows = pass ? drows : trows;
                rt.push_back(pass ? LOGFL_START + t : t);
                rt_rows.push_back({rows.begin(t), rows.end(t)});
            }
        }
    };

    // Detectors fed by terminal read j (the detector targets whose record set includes
    // read j). The completeness gate uses this to map a randomized read to the detectors it can
    // fire on the accepted branch. (Observable/expectation targets are excluded — the reject set
    // is a set of DETECTORS to post-select on.)
    std::vector<std::vector<int>> detectors_fed_by_read((size_t)(M ? M : 1));
    for (int j = 0; j < M; ++j)                         // targets_fed_by_read is ascending: its t < D prefix
        for (int t : targets_fed_by_read[(size_t)j]) {
            if (t >= D) break;
            detectors_fed_by_read[(size_t)j].push_back(t);
        }

    // Detector d's parity Pauli on the deferred wires (product of its reads; duplicates cancel).
    auto detector_pauli = [&](int d) {
        std::vector<uint8_t> dinc((size_t)(M ? M : 1), 0);
        for (int idx : p.detectors[(size_t)d]) dinc[(size_t)idx] ^= 1;
        Pauli P(deferred.n);
        int ny = 0;
        for (int j = 0; j < M; ++j) {
            if (!dinc[(size_t)j]) continue;
            const auto& [b, q] = reads[(size_t)j];
            if (b == 0) { P.setx(q); }
            else if (b == 1) { P.setx(q); P.setz(q); ++ny; }
            else { P.setz(q); }
        }
        P.phase = ny & 3;
        return P;
    };

    // Detectors to post-select on: a single fault that CHANGES a logical-observable magnitude is
    // not Pauli-correctable; it must be rejected (post-selected away) on the detectors it fires.
    std::set<int> reject_set;
    // The not-Pauli-correctable faults THEMSELVES (the primary output): keyed by the sorted
    // detector signature F∪R the fault flips/randomizes, summing the alternative's probability.
    // reject_set above is the derived union convenience only. std::map iterates by sorted key.
    std::map<std::vector<int>, double> postselect_map;

    // ── Per-shot expectation-column law (3.1.12; dem_pe_per_shot.hpp) ─────────────────────────
    // Column kinds for one fault alternative:
    //   PK_OP   the operator rule (Pauli action, the column never sees a changed read): exact;
    //   PK_BIT  a deterministic per-shot flip (|value| unchanged on every branch, one sign);
    //   PK_RAND |value| unchanged on every branch, the sign varies with the reads (a fair or a
    //           detector-correlated sign coin): emitted through the exact joint law;
    //   PK_MAG  the per-shot magnitude changes on some branch: not Pauli-correctable (post-select).
    enum { PK_OP = 0, PK_BIT = 1, PK_RAND = 2, PK_MAG = 3 };
    // A column that anticommutes with a terminal read is 0 on every shot (the read projects it out),
    // whatever its bare-state <P'>; it stays 0 under every fault (conjugation keeps commutation).
    std::vector<uint8_t> col_anticomm(expP.size(), 0);
    for (size_t r = 0; r < expP.size(); ++r)
        for (int j = 0; j < M && !col_anticomm[r]; ++j) {
            const auto& [b, q] = reads[(size_t)j];
            const bool px = expP[r].xbit(q), pz = expP[r].zbit(q);
            if (b == 0 ? pz : b == 2 ? px : (px != pz)) col_anticomm[r] = 1;
        }
    std::unique_ptr<PeShotAnalyzer> pe;                  // built on first need (never for Pauli-only circuits)
    auto pe_get = [&]() -> PeShotAnalyzer& {
        if (!pe) pe = std::make_unique<PeShotAnalyzer>(bare_st(), reads);
        return *pe;
    };
    const std::function<Pauli(int)> noiseless_read = [&](int j) {
        return single_pauli(reads[(size_t)j].first, reads[(size_t)j].second, deferred.n);
    };
    // Noiseless per-shot class per column: 0 unknown, 1 CONST (value col_c0 on every branch),
    // 2 VAR (the per-shot value varies across branches), 3 undecided (exceeds the caps).
    std::vector<int> col_cls(expP.size(), 0);
    std::vector<double> col_c0(expP.size(), 0.0);
    auto rec_names = [&](const std::vector<int>& S) {
        std::string b;
        for (int j : S) { if (!b.empty()) b += " "; b += "rec[-" + std::to_string(M - j) + "]"; }
        return b;
    };
    // Is |h| one constant and sign(h(y)) an affine function b + y.t on the (complete) walked
    // support? Exact GF(2) Gauss-Jordan over the unknowns t_1..t_d (bits 0..d-1) and b (bit 63).
    auto sign_affine = [&](const PeShotAnalyzer::Law& Lw, uint64_t& t_out) -> bool {
        if (!Lw.walked || Lw.leaves.empty() || Lw.d > 62) return false;
        const double a0 = std::abs(Lw.leaves[0].h);
        if (a0 < 1e-9) return false;
        const uint64_t BB = 1ull << 63;
        std::vector<std::pair<uint64_t, int>> red;            // (coefficients, rhs), pivot = lowest bit
        for (const auto& lf : Lw.leaves) {
            if (std::abs(std::abs(lf.h) - a0) > 1e-9) return false;
            uint64_t c = lf.y | BB;
            int rhs = lf.h < 0 ? 1 : 0;
            for (const auto& [rc, rr] : red)
                if ((c >> __builtin_ctzll(rc)) & 1) { c ^= rc; rhs ^= rr; }
            if (!c) { if (rhs) return false; continue; }
            const int pv = __builtin_ctzll(c);
            for (auto& [rc, rr] : red)
                if ((rc >> pv) & 1) { rc ^= c; rr ^= rhs; }
            red.push_back({c, rhs});
        }
        uint64_t u = 0;                                       // free unknowns 0; pivots = rhs
        for (const auto& [rc, rr] : red) if (rr) u |= 1ull << __builtin_ctzll(rc);
        for (const auto& lf : Lw.leaves) {                    // verify on every leaf
            const int s = __builtin_popcountll((lf.y | BB) & u) & 1;
            if ((s != 0) != (lf.h < 0)) return false;
        }
        t_out = u & ~BB;
        return true;
    };
    // Owner rule (3.1.12): a PAULI_EXPECTATION column whose NOISELESS per-shot value varies from
    // shot to shot (it depends on reads that are not declared as its frame) has NO DEM concept. It
    // is dropped: no L-column line, no mechanism, no flag, no refusal on its account; its L index
    // stays reserved (the column layout is unchanged), and it is reported in
    // dropped_expectation_columns with the reason. Every other column keeps its rules. The
    // classification costs no Born sum for a column whose value cannot depend on any read (no coin
    // parity and no magic-correlated read parity: the GF(2) structure alone decides it), and an
    // early-exit branch walk otherwise. With include_expectations=false a column that cannot be
    // decided within the caps is reported instead of refused (never refuse on column grounds there).
    auto drop_column = [&](int r, std::string why) {
        if (col_dropped[(size_t)r]) return;
        col_dropped[(size_t)r] = 1;
        res.dropped_expectation_columns.push_back({r, std::move(why)});
    };
    // include_expectations = false only (owner decision 2026-10-12): a fault-tolerance finding on
    // a column that was NOT requested is reported in the same field, once per (column, reason);
    // the column itself stays judged for every later alternative (it is not dropped).
    std::set<std::pair<int, std::string>> reported_findings;
    auto report_finding = [&](int r, const std::string& why) {
        if (reported_findings.insert({r, why}).second)
            res.dropped_expectation_columns.push_back({r, why});
    };
    std::unique_ptr<PeShotAnalyzer::Alt> alt0;   // the noiseless world's column-independent structure (first need)
    for (int r = 0; r < (int)expP.size(); ++r) {
        const std::string col = "L" + std::to_string(O + r);
        if (col_anticomm[(size_t)r]) { col_cls[(size_t)r] = 1; col_c0[(size_t)r] = 0.0; continue; }
        if (std::abs(beta0[(size_t)r]) > 1.0 - 1e-12) {                 // +-1 stabilizer: constant
            col_cls[(size_t)r] = 1; col_c0[(size_t)r] = beta0[(size_t)r]; continue;
        }
        if (!alt0) alt0 = std::make_unique<PeShotAnalyzer::Alt>(pe_get().alternative({}, noiseless_read));
        PeShotAnalyzer::Law L0 = pe_get().column(*alt0, expP[(size_t)r]);
        if (L0.too_large.empty() && !L0.zero && (L0.coin || L0.d > 0)) {
            const bool coin = L0.coin;
            const double b0 = beta0[(size_t)r];
            // A CONST column's per-shot value IS its average beta0 (and 0 with a coin), so the first
            // leaf that differs from it proves VAR: usually one leaf decides.
            pe_get().walk_h(L0, [&](const std::vector<PeShotAnalyzer::Leaf>& lv) {
                if (coin) return std::abs(lv.back().h) > 1e-9;   // a nonzero value with a fair sign
                return std::abs(lv.back().h - b0) > 1e-9;
            });
        }
        if (!L0.too_large.empty()) {
            col_cls[(size_t)r] = 3;
            if (R > 0) {
                res.error = "PAULI_EXPECTATION column " + col + ": " + L0.too_large +
                            "; whether its noiseless per-shot value is deterministic was not "
                            "decided (refusing)";
                return res;
            }
            drop_column(r, col + ": " + L0.too_large + "; not judged");
            continue;
        }
        bool var = false;
        if (L0.zero) { col_c0[(size_t)r] = 0.0; }
        else if (!L0.coin && L0.d == 0) { col_c0[(size_t)r] = beta0[(size_t)r]; }   // no Born sum
        else if (L0.coin) { var = !L0.walked; col_c0[(size_t)r] = 0.0; }   // stopped: some |h| > 0
        else {
            var = !L0.walked;                                  // stopped: two values differ
            if (!var) {
                const double h0 = L0.leaves.front().h;
                col_c0[(size_t)r] = std::abs(beta0[(size_t)r] - h0) <= 1e-9 ? beta0[(size_t)r] : h0;
            }
        }
        if (!var) { col_cls[(size_t)r] = 1; continue; }
        col_cls[(size_t)r] = 2;
        // The reason names a SUFFICIENT frame when one exists (complete walk within the budget):
        // v(m) = (-1)^{m.S0} |c| (-1)^{b + y.t} => the declared frame XOR S0 XOR (sum t_i T_i).
        std::string why = col + ": its noiseless per-shot value varies with the measurement outcomes";
        PeShotAnalyzer::Law Lf = L0;
        uint64_t t = 0;
        // two leaves of different |value| already rule out every frame (no full walk then)
        const bool mag_varies = L0.leaves.size() >= 2 &&
            std::abs(std::abs(L0.leaves.back().h) - std::abs(L0.leaves.front().h)) > 1e-9;
        if (!mag_varies && pe_get().walk_h(Lf) && sign_affine(Lf, t)) {
            std::vector<uint8_t> par((size_t)(M ? M : 1), 0);
            for (int j : Lf.S0) par[(size_t)j] ^= 1;
            for (int i = 0; i < Lf.d; ++i)
                if ((t >> i) & 1) for (int j : Lf.T[(size_t)i]) par[(size_t)j] ^= 1;
            for (int j : col_frame[(size_t)r]) par[(size_t)j] ^= 1;
            std::vector<int> want;
            for (int j = 0; j < M; ++j) if (par[(size_t)j]) want.push_back(j);
            why += "; its magnitude is fixed and its sign is a record parity: declaring its frame as " +
                   (want.empty() ? std::string("no records") : rec_names(want)) +
                   (col_frame[(size_t)r].empty() ? std::string("")
                        : " (instead of " + rec_names(col_frame[(size_t)r]) + ")") +
                   " makes it deterministic per shot";
        } else {
            why += Lf.too_large.empty()
                ? "; no frame makes it deterministic per shot (its magnitude, or its sign beyond a "
                  "record parity, depends on the outcomes)"
                : "; " + Lf.too_large;
        }
        drop_column(r, std::move(why));
    }
    auto classify_per_shot = [&](const Instr& ins, int k, const AltRes& Dc,
                                 std::vector<int>& pk, std::vector<int>& pbit,
                                 std::vector<PeShotAnalyzer::Law>& plaw, bool& any_mag,
                                 bool& any_rand) -> bool {
        const int RR = (int)expP.size();
        pk.assign((size_t)RR, PK_OP);
        pbit.assign((size_t)RR, 0);
        plaw.assign((size_t)RR, PeShotAnalyzer::Law{});
        any_mag = any_rand = false;
        // Review F1: the faulted reads Dc† R_j Dc are column-independent — conjugated once per alternative (lazily) and
        // shared by every column() call (was: re-conjugated per column x read, the 511-column hot spot).
        std::vector<Pauli> fr_memo((size_t)(M ? M : 1));
        std::vector<uint8_t> fr_have((size_t)(M ? M : 1), 0);
        const std::function<Pauli(int)> faulted_read = [&](int j) {
            if (!fr_have[(size_t)j]) { fr_memo[(size_t)j] = Dc.pull(noiseless_read(j)); fr_have[(size_t)j] = 1; }
            return fr_memo[(size_t)j];
        };
        std::unique_ptr<PeShotAnalyzer::Alt> alt;   // this alternative's column-independent structure (first need)
        for (int r = 0; r < RR; ++r) {
            if (col_dropped[(size_t)r]) continue;                          // no DEM concept: PK_OP, bit 0
            const bool np = column_nonpauli(Dc, r);
            const int opf = np ? 0 : pauli_expectation_flip(Dc, r);
            pbit[(size_t)r] = opf;
            if (!np && randomized_reads.empty()) continue;                 // PK_OP (exact)
            if (col_anticomm[(size_t)r]) { pk[(size_t)r] = PK_BIT; continue; }   // 0 per shot, both worlds
            const double c0 = col_c0[(size_t)r];
            if (!np && std::abs(c0) > 1.0 - 1e-12) continue;   // ±1 stabilizer column: PK_OP (exact)
            const bool zero_col = std::abs(c0) < 1e-9;
            const Pauli Q = Dc.pull(expP[(size_t)r]);
            if (!alt) alt = std::make_unique<PeShotAnalyzer::Alt>(pe_get().alternative(randomized_reads, faulted_read));
            PeShotAnalyzer::Law Lf = pe_get().column(*alt, Q);
            auto is_mag = [&](double v) {
                return zero_col ? std::abs(v) > 1e-6 : std::abs(std::abs(v) - std::abs(c0)) > 1e-6;
            };
            if (Lf.too_large.empty() && !Lf.zero)
                pe_get().walk_h(Lf, [&](const std::vector<PeShotAnalyzer::Leaf>& lv) {
                    return is_mag(lv.back().h);                  // a magnitude change decides it
                });
            if (!Lf.too_large.empty()) {
                if (R > 0) {
                    res.error = loc_name(ins, k) + ": " + Lf.too_large + " (refusing)";
                    return false;
                }
                drop_column(r, "L" + std::to_string(O + r) + ": the per-shot law of the fault at " +
                               loc_name(ins, k) + " on it: " + Lf.too_large + "; not judged");
                pbit[(size_t)r] = 0;
                continue;
            }
            bool mag = false, vary = false;
            int s0 = -1;
            if (Lf.zero) mag = !zero_col;
            else if (!Lf.walked) mag = true;                     // stopped on a magnitude change
            else if (!zero_col)
                for (const auto& lf : Lf.leaves) {
                    const int s = (lf.h < 0) != (c0 < 0) ? 1 : 0;
                    if (s0 < 0) s0 = s; else if (s != s0) vary = true;
                }
            if (mag) { pk[(size_t)r] = PK_MAG; any_mag = true; }
            else if (zero_col) { pk[(size_t)r] = PK_BIT; }                 // 0 stays 0: opf / 0
            else if (Lf.coin || vary) { pk[(size_t)r] = PK_RAND; any_rand = true; }
            else { pk[(size_t)r] = PK_BIT; pbit[(size_t)r] = s0; }
            plaw[(size_t)r] = std::move(Lf);
        }
        return true;
    };
    // Channel walk (same per-qubit / per-pair granularity as the sampler's fast plan).
    std::map<Sig, double> acc;               // cross-channel XOR-composed mechanisms
    // Noiseless value <D_t> of each randomized target, per target id (state-only: computed once).
    std::map<int, double> target_value_cache;
    // ── twirl-routes T1: the ONE stabilizer-twirl routine (qeccore/fault_record_law.hpp) ──────────────────────
    // A fault whose propagated residual randomizes reads has, on the targets rt those reads feed (record columns
    // D_t, local bit i = rt[i]), the record law of Theorem 1 — the sampler's own record table:
    //     f = c ⊕ Σ_k u_k W_k ⊕ Σ_j z_j B_j,   u fair coins, z ~ P(z) on the table's κ Born directions,
    //     c = β ⊕ (the noiseless bits) ⊕ ⟨P, D_t⟩   (Lemma 2: the prefix P's flips; the table itself is P-free).
    // The coin part is GF(2) only (the stabilizer twirl over the certified group); Born sums are needed only on the
    // κ ≤ k kernel-logical directions (born_law, 2^κ ≤ χ leaves). One law per distinct (canonical diagonal residual
    // mod G, rt), P-free, so every alternative sharing a residual shares it. Built lazily: a Pauli-only circuit pays
    // nothing. Stage 1 (the table: ρ, κ, rows) precedes the owner's cap; stage 2 (the Born quotient: a uniform coset
    // — κ = 0, or a uniform quotient law — or a Born mixture) runs only after it.
    struct ErasureLocal {
        ErasureSplit split;
        std::vector<std::vector<uint64_t>> rref;   // canonical RREF rows of W (lowest-bit pivots; merge key), local frame
    };
    struct DemLaw {
        int m = 0, rho = 0, kappa = 0, vsup = 0, UW = 1;
        char kind = 'C';                           // 'C' uniform coset c0 + span(W); 'M' Born mixture
        bool classified = false;                   // stage 2 done
        std::vector<uint64_t> c0;                  // P-free centre: β ⊕ the noiseless bits (⊕ a uniform quotient's offset)
        std::vector<std::vector<uint64_t>> W;      // the coin rows (+ a uniform quotient's directions)
        std::vector<std::vector<uint64_t>> B;      // stage 1: the κ Born rows
        std::vector<FreeDir> lam;                  // stage 1: the κ Born directions
        std::vector<std::pair<std::vector<uint64_t>, double>> cls;   // 'M': (offset class mod W, weight)
        std::vector<Pauli> M;                      // the columns' record operators (Lemma 2's prefix flips)
        ErasureLocal el;                           // 'C' (v >= 1): the weight split of W and its canonical RREF
    };
    struct TwirlCtx {
        CertifiedGroupPlanes G;
        std::unique_ptr<RecordColumns> rc;
        std::unordered_map<int, RecColumn> col;                       // per target id (order set per law)
        std::unordered_map<std::string, std::vector<double>> born;     // P(z) per Born-direction content
    };
    std::unique_ptr<TwirlCtx> tw;
    auto tw_get = [&]() -> TwirlCtx& {
        if (!tw) {
            tw = std::make_unique<TwirlCtx>();
            tw->G = CertifiedGroupPlanes::build(bare_st().certified_stabilizers_sparse(), bare_st().n());
            tw->rc = std::make_unique<RecordColumns>(tw->G);
        }
        return *tw;
    };
    std::unordered_map<std::string, DemLaw> law_cache;   // node-stable: records keep pointers into it
    long born_evals = 0;                                 // Born-law node evaluations (census born_sums)
    BornLawStats born_stats;                             // explaw: the DEM laws routed to the coset law
    struct CensusBorn {                                  // test API: the Born count (the DEM laws' and the per-shot
        DemLawCensus* c;                                 // column walks'), on every exit path
        const long* n;
        const BornLawStats* st;
        const std::unique_ptr<PeShotAnalyzer>* pe;
        ~CensusBorn() {
            if (!c) return;
            c->born_sums = *n + (*pe ? (*pe)->born_evals() : 0);
            c->coset_laws = st->coset_laws + (*pe ? (*pe)->coset_walks() : 0);
            c->coset_work = st->coset_work + (*pe ? (*pe)->coset_work() : 0.0);
        }
    } census_born{census, &born_evals, &born_stats, &pe};
    auto wl_lead = [](const std::vector<uint64_t>& x) {
        for (size_t w = 0; w < x.size(); ++w) if (x[w]) return (int)(w * 64) + __builtin_ctzll(x[w]);
        return -1;
    };
    auto wl_bit = [](const std::vector<uint64_t>& x, int b) { return (x[(size_t)(b >> 6)] >> (b & 63)) & 1ULL; };
    auto wl_xor = [](std::vector<uint64_t>& x, const std::vector<uint64_t>& y) {
        for (size_t w = 0; w < x.size(); ++w) x[w] ^= y[w];
    };
    // Canonical RREF (lowest-bit pivots, fully reduced, sorted by pivot) of a set of local rows; dependent rows dropped.
    auto canonical_rref = [&](const std::vector<std::vector<uint64_t>>& rows) {
        std::vector<std::vector<uint64_t>> rr;
        for (const auto& w0 : rows) {
            std::vector<uint64_t> x = w0;
            for (const auto& r : rr) if (wl_bit(x, wl_lead(r))) wl_xor(x, r);
            const int pv = wl_lead(x);
            if (pv < 0) continue;
            for (auto& r : rr) if (wl_bit(r, pv)) wl_xor(r, x);
            rr.push_back(std::move(x));
        }
        std::sort(rr.begin(), rr.end(), [&](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
            return wl_lead(a) < wl_lead(b);
        });
        return rr;
    };
    // Stage 1: the record table of Dc on rt (Dts[i] = rt[i]'s record operator, its noiseless value checked by the
    // caller). nf receives the canonical residual (its prefix carries Lemma 2's flips).
    // `raw` (T2): further record columns after the targets — products of terminal reads whose RAW outcome bit (no
    // noiseless fold) the per-shot expectation-column law reads (a column's coin parity S0 and y parities T_i).
    auto fault_law = [&](const AltRes& A, const std::vector<int>& rt, const std::vector<Pauli>& Dts,
                         DiagNormalForm& nf, const std::vector<Pauli>& raw = {}) -> DemLaw& {
        TwirlCtx& X = tw_get();
        std::string key;
        auto put_len = [&](size_t L) { key.append(reinterpret_cast<const char*>(&L), sizeof L); };
        auto put_pauli = [&](const Pauli& Q) {
            key.push_back((char)Q.phase);
            put_len(Q.x.size());
            key.append(reinterpret_cast<const char*>(Q.x.data()), Q.x.size() * 8);
            key.append(reinterpret_cast<const char*>(Q.z.data()), Q.z.size() * 8);
        };
        std::vector<Pauli> pulled;                 // dem-ch: the general residual's pullbacks (targets, then raw)
        if (A.gen) {
            // dem-ch: a general residual has no diagonal normal form. Its law is a function of the pulled-back record
            // operators Q_c = E† M_c E alone (every sign included: the identity prefix below adds no Lemma-2 flip), so
            // the cache key is that list (+ rt).
            nf = DiagNormalForm{};
            nf.prefix = Pauli(deferred.n);
            nf.a.assign((size_t)deferred.n, 0);
            nf.diagonal_class = true;
            key.push_back('\x01');
            for (const Pauli& Mi : Dts) pulled.push_back(A.pull(Mi));
            for (const Pauli& Mi : raw) pulled.push_back(A.pull(Mi));
            put_len(pulled.size());
            for (const Pauli& Q : pulled) put_pauli(Q);
            put_len(rt.size());
            key.append(reinterpret_cast<const char*>(rt.data()), rt.size() * sizeof(int));
        } else {
            const DiagPauliClifford& Dc = A.D;
            key.push_back('\x00');                // branch tag: keeps the key injective across both branches
        nf = residual_normal_form(Dc);
        canonicalize_mod_stabilizers_inplace(nf, X.G);
        {   // review F5: every variable-length field is length-prefixed (an injective encoding)
            std::vector<uint64_t> aw((size_t)((nf.a.size() + 63) / 64), 0);
            for (size_t q = 0; q < nf.a.size(); ++q) if (nf.a[q] & 1) aw[q >> 6] |= 1ULL << (q & 63);
            put_len(aw.size());
            key.append(reinterpret_cast<const char*>(aw.data()), aw.size() * 8);
            put_len(nf.cz.size());
            for (const auto& pr : nf.cz) key.append(reinterpret_cast<const char*>(&pr), sizeof pr);
            put_len(rt.size());
            key.append(reinterpret_cast<const char*>(rt.data()), rt.size() * sizeof(int));
            put_len(raw.size());
            for (const Pauli& Q : raw) put_pauli(Q);
        }
        }
        auto it = law_cache.find(key);
        if (it != law_cache.end()) return it->second;
        const int mt = (int)rt.size();
        const int m = mt + (int)raw.size();
        DemLaw L;
        L.m = m;
        L.UW = std::max(1, (m + 63) / 64);
        L.c0.assign((size_t)L.UW, 0);
        std::vector<RecColumn> cols;
        cols.reserve((size_t)m);
        for (int i = 0; i < mt; ++i) {
            auto ci = X.col.find(rt[(size_t)i]);
            if (ci == X.col.end()) ci = X.col.emplace(rt[(size_t)i], X.rc->column(Dts[(size_t)i], 0)).first;
            cols.push_back(ci->second);
            cols.back().order = i;
            L.M.push_back(Dts[(size_t)i]);
            if (target_value_cache.at(rt[(size_t)i]) < 0) L.c0[(size_t)(i >> 6)] ^= 1ULL << (i & 63);
        }
        for (int i = mt; i < m; ++i) {
            cols.push_back(X.rc->column(raw[(size_t)(i - mt)], i));
            L.M.push_back(raw[(size_t)(i - mt)]);
        }
        if (m > 0) {
            RecTable T;
            if (A.gen) {                               // dem-ch: the null-form table over the pullbacks
                std::vector<int> order((size_t)m);
                for (int i = 0; i < m; ++i) order[(size_t)i] = i;
                build_record_table_pullbacks(X.G, bare_st(), pulled, order, T);
            } else if (!fault_record_law(X.G, bare_st(), nf, cols, T))
                throw std::logic_error("export_dem: a diagonal residual's record table is a fallback (engine bug)");
            L.rho = T.rho;
            L.kappa = T.kappa;
            auto local = [&](const std::vector<uint64_t>& row) {
                std::vector<uint64_t> x((size_t)L.UW, 0);
                for (size_t w = 0; w < row.size() && w < x.size(); ++w) x[w] = row[w];
                return x;
            };
            wl_xor(L.c0, local(T.base));
            for (const auto& r : T.coin_row) L.W.push_back(local(r));
            for (const auto& r : T.born_row) L.B.push_back(local(r));
            L.lam = T.lam;
        }
        return law_cache.emplace(std::move(key), std::move(L)).first->second;
    };
    // P(z) on Born directions, memoised per direction content (one Born law per distinct direction set per export).
    auto born_of = [&](const std::vector<FreeDir>& lam) -> const std::vector<double>& {
        std::string bk;
        for (const FreeDir& d : lam) {
            bk.append(reinterpret_cast<const char*>(d.a.data()), d.a.size());
            bk.append(reinterpret_cast<const char*>(d.b.data()), d.b.size());
            bk.push_back((char)d.phase); bk.push_back((char)d.par); bk.push_back('|');
        }
        TwirlCtx& X = tw_get();
        auto bi = X.born.find(bk);
        if (bi == X.born.end()) bi = X.born.emplace(std::move(bk), born_law(bare_st(), lam, &born_evals, &born_stats)).first;
        return bi->second;
    };
    // Stage 2: the Born quotient (after the caller's cap): P(z) on the κ directions (memoised per direction content),
    // the classes of the offsets Σ z_j B_j mod span(W) with their weights (support filter 1e-12 relative to the peak);
    // one class or a uniform affine set of classes is a uniform coset (W extended by the class directions), anything
    // else a Born mixture. Then the coset's weight split (the erasure records' precompute).
    auto classify_law = [&](DemLaw& L, const std::vector<int>& rt) {
        if (L.classified) return;
        L.classified = true;
        if (L.kappa > 0) {
            BornQuotient q = born_quotient(L.W, L.B, born_of(L.lam), L.UW);
            if (q.coset) {
                wl_xor(L.c0, q.r0);
                for (auto& d : q.dirs) L.W.push_back(std::move(d));
            } else {
                L.kind = 'M';
                L.cls = std::move(q.cls);
                L.vsup = L.rho + (int)q.dirs.size();
                return;
            }
        }
        L.vsup = (int)L.W.size();
        if (L.vsup == 0) return;
        std::vector<uint64_t> dml((size_t)L.UW, 0);
        for (int i = 0; i < L.m; ++i) if (rt[(size_t)i] < D) dml[(size_t)(i >> 6)] |= 1ULL << (i & 63);
        L.el.split = erasure_split(L.W, dml);
        L.el.rref = canonical_rref(L.W);
    };
    // A local pattern (bit i = rt[i]) united with the fixed part `base`, as a sorted Sig.
    auto local_sig = [&](const std::vector<uint64_t>& x, const std::vector<int>& rt, const Sig& base) {
        Sig fl, out;
        for (size_t w = 0; w < x.size(); ++w)
            for (uint64_t y = x[w]; y; y &= y - 1) fl.push_back(rt[w * 64 + (size_t)__builtin_ctzll(y)]);
        std::set_union(base.begin(), base.end(), fl.begin(), fl.end(), std::back_inserter(out));
        return out;
    };
    // The fault's P-dependent centre: c0 ⊕ ⟨P, M_i⟩ (Lemma 2).
    auto fault_centre = [&](const DemLaw& L, const DiagNormalForm& nf) {
        std::vector<uint64_t> c = L.c0;
        for (int i = 0; i < L.m; ++i)
            if (Pauli::anticommute_bit(nf.prefix, L.M[(size_t)i])) c[(size_t)(i >> 6)] ^= 1ULL << (i & 63);
        return c;
    };
    // ── twirl-routes T2: the per-shot expectation-column laws on the ONE routine (no coin branching) ─────────────
    // A judged column's per-shot value is v = (−1)^{S0 parity} h(y) (dem_pe_per_shot.hpp): its coin parity S0 and its
    // y parities T_i are products of terminal reads — RAW record columns of the fault's record table next to the
    // targets — so the joint law of (targets, S0, y) is the table's: coins are its GF(2) rows (never branched on),
    // the y bits are functions of its κ Born bits z (born_of: 2^κ ≤ χ leaves), h(y) is the column's own Born-walk
    // value (walk_h's leaves; value_at off them).
    auto read_product = [&](const std::vector<int>& S) {
        Pauli A(deferred.n);
        for (int j : S) A = Pauli::multiply(A, noiseless_read(j));
        return A;
    };
    struct ColBits { int r = -1, s0 = -1, t0 = 0, d = 0; std::unordered_map<uint64_t, double> h; };
    // Append column r's raw columns (S0 if coin, then T_1..T_d) to `raw` (positions offset by `base_m`).
    auto column_bits = [&](int r, PeShotAnalyzer::Law& Lw, int base_m, std::vector<Pauli>& raw) {
        ColBits cb;
        cb.r = r;
        if (Lw.zero) return cb;
        if (Lw.coin) { cb.s0 = base_m + (int)raw.size(); raw.push_back(read_product(Lw.S0)); }
        cb.t0 = base_m + (int)raw.size();
        cb.d = Lw.d;
        for (int i = 0; i < Lw.d; ++i) raw.push_back(read_product(Lw.T[(size_t)i]));
        for (const auto& lf : Lw.leaves) cb.h.emplace(lf.y, lf.h);
        return cb;
    };
    // Review F6: a column's y parities are Born-only (column() builds T_i from the world nullspace: no non-free X
    // part), so no coin row of the table may touch a y column; col_value reads y from the Born centre alone.
    auto assert_y_coin_free = [&](const DemLaw& J, const std::vector<ColBits>& cbs) {
        for (const auto& w : J.W)
            for (const ColBits& cb : cbs)
                for (int i = 0; i < cb.d; ++i)
                    if (wl_bit(w, cb.t0 + i))
                        throw std::logic_error("export_dem: a coin row reads an expectation column's y parity (engine bug)");
    };
    auto col_value = [&](ColBits& cb, PeShotAnalyzer::Law& Lw, const std::vector<uint64_t>& x) -> double {
        if (Lw.zero) return 0.0;
        uint64_t y = 0;
        for (int i = 0; i < cb.d; ++i) if (wl_bit(x, cb.t0 + i)) y |= 1ULL << i;
        auto it = cb.h.find(y);
        if (it == cb.h.end()) it = cb.h.emplace(y, pe_get().value_at(Lw, y)).first;
        return it->second;
    };
    // Per-shot accepted-branch check: is the fault undetectable on the ACCEPTED branch (every randomized detector in
    // `aff` reads its noiseless value)? A violation is an accepted outcome of positive probability on which some judged
    // column's per-shot value differs from its noiseless value. With the table: on Born leaf z the accepted coins
    // solve A u = x(z) on the detector columns (GF(2)); a column's coin bit is then fixed or uniform on the solutions.
    // 0 ok / 1 violation (bad_col: the column, −1 when not identified).
    auto per_shot_accepted_violation = [&](const AltRes& Dc, const std::set<int>& aff,
                                           const std::vector<int>& pk, const std::vector<int>& pbit,
                                           std::vector<PeShotAnalyzer::Law>& plaw, int& bad_col) -> int {
        bad_col = -1;
        std::vector<int> dids(aff.begin(), aff.end());
        const int kd = (int)dids.size();
        std::vector<Pauli> Dts;
        for (int d : dids) {
            Pauli Dd = detector_pauli(d);
            auto [pp, pm] = framed_expectation(bare_st(), Dd);
            const double s = pp - pm;
            if (std::abs(std::abs(s) - 1.0) > 1e-9) return 1;   // not a deterministic detector: refuse
            target_value_cache.emplace(d, s);
            Dts.push_back(std::move(Dd));
        }
        bool static_violation = false;   // a column differing on EVERY branch (if accepted is reachable)
        std::vector<Pauli> raw;
        std::vector<ColBits> cbs;
        for (size_t r = 0; r < expP.size(); ++r) {
            if (pk[r] == PK_RAND || pk[r] == PK_MAG) {
                cbs.push_back(column_bits((int)r, plaw[r], kd, raw));
            } else if (pbit[r]) {
                bool sv = false;
                if (pk[r] == PK_BIT) { if (std::abs(col_c0[r]) >= 1e-9) sv = true; }
                else if (!(col_cls[r] == 1 && std::abs(col_c0[r]) < 1e-9))   // PK_OP flip
                    sv = true;
                if (sv && !static_violation) { static_violation = true; bad_col = (int)r; }
            }
        }
        DiagNormalForm nf;
        DemLaw& J = fault_law(Dc, dids, Dts, nf, raw);
        const std::vector<uint64_t> c = fault_centre(J, nf);
        assert_y_coin_free(J, cbs);
        // coin rows split: detector part (eliminated) and the columns' S0 bits (tracked; bit q = cbs[q], CW words)
        const size_t CW = std::max<size_t>(1, (cbs.size() + 63) / 64);
        struct CRow { std::vector<uint64_t> det, s0; int piv; };
        std::vector<CRow> ech;
        std::vector<uint64_t> freecols(CW, 0);               // columns whose coin bit is uniform on the accepted set
        auto det_part = [&](const std::vector<uint64_t>& x) {
            std::vector<uint64_t> d0((size_t)J.UW, 0);
            for (int i = 0; i < kd; ++i) if (wl_bit(x, i)) d0[(size_t)(i >> 6)] |= 1ULL << (i & 63);
            return d0;
        };
        auto s0_part = [&](const std::vector<uint64_t>& x) {
            std::vector<uint64_t> b(CW, 0);
            for (size_t q = 0; q < cbs.size(); ++q)
                if (cbs[q].s0 >= 0 && wl_bit(x, cbs[q].s0)) b[q >> 6] |= 1ULL << (q & 63);
            return b;
        };
        for (const auto& w : J.W) {
            CRow rw{det_part(w), s0_part(w), -1};
            for (const auto& e : ech) if (wl_bit(rw.det, e.piv)) { wl_xor(rw.det, e.det); wl_xor(rw.s0, e.s0); }
            rw.piv = wl_lead(rw.det);
            if (rw.piv < 0) { for (size_t i = 0; i < CW; ++i) freecols[i] |= rw.s0[i]; continue; }
            for (auto& e : ech) if (wl_bit(e.det, rw.piv)) { wl_xor(e.det, rw.det); wl_xor(e.s0, rw.s0); }
            ech.push_back(std::move(rw));
        }
        const std::vector<double>& pz = born_of(J.lam);
        double pmax = 0.0;
        for (double x : pz) pmax = std::max(pmax, x);
        double pacc = 0.0;
        std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> acc_leaves;   // (x(z), the columns' fixed coin bits)
        for (size_t z = 0; z < pz.size(); ++z) {
            if (!(pz[z] > 1e-12 * pmax)) continue;
            std::vector<uint64_t> x = c;
            for (int j = 0; j < J.kappa; ++j) if ((z >> j) & 1) wl_xor(x, J.B[(size_t)j]);
            std::vector<uint64_t> t = det_part(x);
            std::vector<uint64_t> s0 = s0_part(x);
            for (const auto& e : ech) if (wl_bit(t, e.piv)) { wl_xor(t, e.det); wl_xor(s0, e.s0); }
            if (wl_lead(t) >= 0) continue;                   // no coin assignment accepts this leaf
            pacc += pz[z] * std::ldexp(1.0, -(int)ech.size());
            acc_leaves.push_back({std::move(x), std::move(s0)});
        }
        if (pacc < 1e-12) return 0;                          // accepted branch unreachable
        if (static_violation) return 1;
        bad_col = -1;
        for (auto& [x, s0] : acc_leaves)
            for (size_t q = 0; q < cbs.size(); ++q) {
                const int r = cbs[q].r;
                const double h = col_value(cbs[q], plaw[(size_t)r], x);
                const double c0 = col_c0[(size_t)r];
                const bool coin = cbs[q].s0 >= 0;
                const bool free_b = coin && wl_bit(freecols, (int)q);
                const double v = (coin && wl_bit(s0, (int)q)) ? -h : h;
                if (std::abs(v - c0) > 1e-6 || (free_b && std::abs(-v - c0) > 1e-6)) { bad_col = r; return 1; }
            }
        return 0;
    };
    // The owner's cap (kDemSupportCap): refuse by name, naming the fault, v and the detector count.
    auto support_cap_refusal = [&](const Instr& ins, int k,
                                   const std::vector<std::pair<int, PauliBasis>>& fac,
                                   const std::vector<int>& rt, int v, double candidates = -1.0,
                                   int cutoff_hint = 0) {
        int nd = 0;
        for (int t : rt) if (t < D) ++nd;
        std::string alt, qs;
        for (size_t i = 0; i < fac.size(); ++i) {
            const PauliBasis pb = fac[i].second;
            alt += std::string(i ? "\u2297" : "") + (pb == PauliBasis::X ? "X" : pb == PauliBasis::Y ? "Y" : "Z");
            qs += (i ? "," : "") + std::to_string(ins.qubits[(size_t)fac[i].first]);
        }
        std::string site;
        if (ins.origin == Instr::Origin::UserSite) site = "; user noise site " + std::to_string(ins.origin_index);
        char hb[32];
        std::snprintf(hb, sizeof hb, "%.0f", std::ldexp(1.0, v) - 1.0);
        const std::string where = loc_name(ins, k) + " (alternative " + alt + " on deferred qubit" +
                                  (fac.size() > 1 ? "s " : " ") + qs + site + ")";
        const std::string vs = std::to_string(v), cap = std::to_string(kDemSupportCap);
        if (cutoff_hint) {
            // The exact-output cap (owner, fix round 1; dem-t2: reachable only on an explicit exact request, named as the cause):
            // the reason and the finite-cutoff options. hint 2 = a uniform
            // coset law (a finite cutoff lifts the cap within 2^12 candidates); 1 = before classification (v >= v0).
            res.error = where + ": this Clifford fault randomizes " + std::to_string(nd) + " detector(s) (and " +
                        std::to_string((int)rt.size() - nd) + " other DEM target(s)) over v " + (cutoff_hint == 1 ? ">= " : "= ") +
                        vs + " signature directions; its exact DEM expansion needs " + (cutoff_hint == 1 ? "at least " : "") +
                        "2^" + vs + " - 1 = " + hb + " mechanisms, above the exact-output cap (v <= " + cap + "). " +
                        "Cause: the EXACT expansion was requested (clifford_cutoff=inf); the cap applies only to it. " +
                        "Options: the default clifford_cutoff=2 (matching-friendly) or any finite clifford_cutoff=t "
                        "emits only patterns flipping <= t detectors (larger t keeps more of the fault)" +
                        (cutoff_hint == 1 ? " - this applies when the fault's law is a uniform coset, as Clifford faults "
                                            "usually are" : "") +
                        ". A finite t drops heavier patterns: the export reports the dropped Clifford mass, and dropped "
                        "patterns can lower the decoder's error-suppression order.";
            return;
        }
        res.error = where + ": the Clifford error this fault leaves triggers too many detectors: it "
                    "randomizes " + std::to_string(nd) + " detector(s) (and " + std::to_string((int)rt.size() - nd) +
                    " other DEM target(s)) over a support of dimension v = " + vs + " > " + cap +
                    "; its exact DEM needs 2^v - 1 = " + hb + " hyperedges (refusing)";
        if (candidates >= 0.0) {
            char cb[96];
            std::snprintf(cb, sizeof cb, " (cutoff t = %d, %.0f candidate patterns)", tcut, candidates);
            res.error += cb;
        }
    };
    auto fold = [&](std::vector<Mech>&& mechs) {
        for (Mech& mech : mechs) {
            double& a = acc[mech.sig];       // default 0
            a = a + mech.p - 2.0 * a * mech.p;
        }
    };
    // ── dem-erasure records (design note signed 2026-10-30) ─────────────────────────────────────
    // One record per twirled Clifford fault (identical laws of one channel instance merged, p1 + p2),
    // never folded into acc. A coset record's lines are the patterns of c + W of detector weight
    // <= tcut at the closed-form q; an explicit-support record's are its own FWHT mechanisms, filtered
    // by the same weight.
    struct CliffordRecord {
        bool support = false;
        bool first_order = false;            // support: the disjoint law (no independent split)
        double p = 0.0;
        int v = 0;
        bool offset = false;
        double qW = 0.0;                     // offset: the omitted (negative) W \ {0} correction
        Sig supp;                            // every target the law can flip
        std::string loc, alt;
        long long kept = 0;
        double dropped = 0.0;
        std::vector<Mech> lines;
        bool erasure_line = false;           // dem-erasure-out: a coset written as ONE line (eoff + span(egen), at p)
        Sig eoff;
        std::vector<Sig> egen;
    };
    std::vector<CliffordRecord> records;
    // One Clifford alternative of the current instance, pending its flush (el: its law's weight split, computed once
    // per law in the targets' local frame, bit i = rt[i]; c: this fault's centre in that frame).
    struct PendingErasure {
        const ErasureLocal* el;
        std::vector<int> rt;
        Sig base;
        int base_w;
        bool pure;
        double p;
        std::string alt;
        std::vector<uint64_t> c;
    };
    std::vector<PendingErasure> erasures;
    std::string cur_alt;                     // the current alternative, as the record header names it
    auto alt_string = [&](const Instr& ins, const std::vector<std::pair<int, PauliBasis>>& fac) {
        std::string a;
        for (size_t i = 0; i < fac.size(); ++i) {
            const PauliBasis pb = fac[i].second;
            a += std::string(i ? "*" : "") + (pb == PauliBasis::X ? "X" : pb == PauliBasis::Y ? "Y" : "Z") +
                 std::to_string(ins.qubits[(size_t)fac[i].first]);
        }
        return a;
    };
    // Per-instance scratch, hoisted so its capacity is reused (cleared by every flush).
    std::vector<FaultLaw> comps;             // explicit-support (G / L) laws: one record each
    std::vector<std::string> comp_alt;
    std::vector<DemLawCensus::Comp> comp_meta;
    // Every alternative of a noise instance, in the channel walk's order: fn(prob, factors).
    auto for_each_alternative = [&](const Instr& ins, auto&& fn) {
        switch (ins.channel) {
            case NoiseChannel::X_ERROR:
            case NoiseChannel::Y_ERROR:
            case NoiseChannel::Z_ERROR: {
                const PauliBasis pb = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                                    : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y : PauliBasis::Z;
                for (int qi = 0; qi < (int)ins.qubits.size(); ++qi) fn(ins.probs[0], {{qi, pb}});
                break;
            }
            case NoiseChannel::DEPOLARIZE1:
            case NoiseChannel::PAULI_CHANNEL_1: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
                for (int qi = 0; qi < (int)ins.qubits.size(); ++qi)
                    for (int i = 0; i < 3; ++i) fn(dep ? ins.probs[0] / 3.0 : ins.probs[(size_t)i], {{qi, pc1_slot_basis(i)}});
                break;
            }
            case NoiseChannel::DEPOLARIZE2:
            case NoiseChannel::PAULI_CHANNEL_2:
                for (int qi = 0; qi + 1 < (int)ins.qubits.size(); qi += 2)
                    for (int idx = 0; idx < 15; ++idx) {
                        std::vector<std::pair<int, PauliBasis>> f;
                        if (const int c = pc2_first_code(idx))  f.push_back({qi, pauli_code_basis(c)});
                        if (const int c = pc2_second_code(idx)) f.push_back({qi + 1, pauli_code_basis(c)});
                        fn(ins.channel == NoiseChannel::DEPOLARIZE2 ? ins.probs[0] / 15.0 : ins.probs[(size_t)idx], f);
                    }
                break;
            case NoiseChannel::CORRELATED: {
                const std::vector<double> q = corr_outcome_probs(ins);
                for (int t = 0; t < corr_terms(ins); ++t) {
                    std::vector<std::pair<int, PauliBasis>> f;
                    for (size_t qi = 0; qi < ins.qubits.size(); ++qi)
                        if (const int c = corr_code(ins, t, (int)qi)) f.push_back({(int)qi, pauli_code_basis(c)});
                    if (!f.empty()) fn(q[(size_t)t], f);
                }
                break;
            }
        }
    };
    // alternative = composition of (qubit-index, factor) atoms in compose_fired's order. dem-ch: an alternative with a
    // PPR-retry atom composes the atoms' error tableaux instead (compose_error_tableau); D stays the identity.
    AltScratch alt_scratch(deferred.n);                // the per-alternative composition scratch (AltRes::D)
    const int compose_mode = g_compose_mode.load(std::memory_order_relaxed);   // test API; 0 in production
    auto compose_alt = [&](const LocationEntry& loc, const std::vector<std::pair<int, PauliBasis>>& factors) {
        const auto has_x = [](PauliBasis pb) { return pb == PauliBasis::X || pb == PauliBasis::Y; };
        const auto has_z = [](PauliBasis pb) { return pb == PauliBasis::Z || pb == PauliBasis::Y; };
        bool gen = false;                                  // a diagonal table has no PPR atom: never set there
        if (dn.general)
            for (auto& [qi, pb] : factors)
                if ((has_x(pb) && loc.x_atom[qi].ppr) || (has_z(pb) && loc.z_atom[qi].ppr)) { gen = true; break; }
        if (!gen) {                                        // the pre-dem-ch chain, byte for byte (AltScratch)
            bool sparse = compose_mode != 1;               // test API mode 1: the dense chain (the oracle)
            for (auto& [qi, pb] : factors) {
                if (has_x(pb) && !AltScratch::sparse_ok(loc.x_atom[qi])) sparse = false;
                if (has_z(pb) && !AltScratch::sparse_ok(loc.z_atom[qi])) sparse = false;
            }
            if (sparse) {
                const DiagPauliClifford& D0 = alt_scratch.identity();
                for (auto& [qi, pb] : factors) {
                    if (has_x(pb)) alt_scratch.then_sparse(loc.x_atom[qi]);
                    if (has_z(pb)) alt_scratch.then_sparse(loc.z_atom[qi]);
                }
                const std::vector<int>* supp = alt_scratch.finish_sparse();
                if (compose_mode == 2) {                   // test API: the dense chain beside it, compared
                    alt_scratch.dense_begin();
                    for (auto& [qi, pb] : factors) {
                        if (has_x(pb)) alt_scratch.then_dense(loc.x_atom[qi]);
                        if (has_z(pb)) alt_scratch.then_dense(loc.z_atom[qi]);
                    }
                    g_compose_compared.fetch_add(1, std::memory_order_relaxed);
                    if (alt_scratch.has_cz()) g_compose_with_cz.fetch_add(1, std::memory_order_relaxed);
                    if (!alt_scratch.sparse_equals_dense()) g_compose_mismatched.fetch_add(1, std::memory_order_relaxed);
                }
                return AltRes{D0, nullptr, supp};
            }
            alt_scratch.dense_begin();
            for (auto& [qi, pb] : factors) {
                if (has_x(pb)) alt_scratch.then_dense(loc.x_atom[qi]);
                if (has_z(pb)) alt_scratch.then_dense(loc.z_atom[qi]);
            }
            return AltRes{alt_scratch.dense(), nullptr, nullptr};
        }
        std::vector<const PropResult*> atoms;
        for (auto& [qi, pb] : factors) {
            if (has_x(pb)) atoms.push_back(&loc.x_atom[qi]);
            if (has_z(pb)) atoms.push_back(&loc.z_atom[qi]);
        }
        return AltRes{alt_scratch.identity(), compose_error_tableau(atoms, deferred.n), nullptr};
    };
    // demfb census (test API): every alternative's Q correction, counted before the walk (a refusal later in the
    // walk does not truncate it).
    if (census && fb.has_feedback) {
        int l2 = 0;
        for (const Instr& ins : deferred.stream) {
            if (ins.kind != Instr::Kind::Noise) continue;
            const LocationEntry& loc = table.locations[(size_t)l2];
            for_each_alternative(ins, [&](double prob, const std::vector<std::pair<int, PauliBasis>>& f) {
                if (prob <= 0.0) return;
                const std::vector<QOp> qs = q_ops(compose_alt(loc, f).D, l2);
                if (!qs.empty()) { ++census->fb_q_corrections; census->fb_q_pairs += (long)qs.size(); }
            });
            ++l2;
        }
    }
    // xdem: every alternative's extended signature (flips over the columns, then x ids) from one backward sweep of
    // the skeleton; the walk below merges, converts and folds them exactly as Pauli alternatives (no residual
    // propagation: a non-Clifford gate is identity in S).
    xdem::FaultColumns fault_cols;
    if (is_xdem) {
        std::vector<std::vector<int>> col_recs;
        for (int d = 0; d < D; ++d) col_recs.push_back(p.detectors[(size_t)d]);
        for (int o = 0; o < O; ++o) col_recs.push_back(obs[(size_t)o]);
        for (int k2 = 0; k2 < NDEC; ++k2) col_recs.push_back(dec_recs[(size_t)k2]);
        if ((int)col_recs.size() != T_TOTAL)
            throw std::logic_error("export_dem: xdem column layout mismatch (engine bug)");
        try {
            fault_cols = xdem::build_fault_columns(*wm.sk, deferred, col_recs);
            if (lean) {
                // The oracle's determinism refusal, answered here (no delegation): the verdict is exact (the oracle
                // refuses a column iff nondeterministic_columns lists it), and the text is the oracle's, from the
                // exact <P> of the first one (column_expectation). On a circuit above the bare-state capacity the
                // oracle would raise XtimCapacityError before its determinism check; this names the real cause.
                const std::vector<int> nd = xdem::nondeterministic_columns(wm, fault_cols);
                if (!nd.empty()) {
                    const int t0 = nd[0];
                    std::vector<uint8_t> inc0((size_t)M, 0);
                    for (int j : wm.sk->targets[(size_t)t0]) inc0[(size_t)j] = 1;   // parity-reduced
                    const double ev = sampled_value(inc0, xdem::column_expectation(wm, t0));
                    const bool unk = ref_state < 0 && std::abs(ev) >= 5e-7;
                    res.error = nondeterministic_message(t0 < D ? 'D' : 'L', t0 < D ? t0 : t0 - D, ev, unk);
                    for (int t : nd)
                        res.nondeterministic_targets.push_back(t < D ? "D" + std::to_string(t)
                                                                     : "L" + std::to_string(t - D));
                    if (xc) xc->lean_refusal = true;
                    return res;
                }
                if (xc) xc->det_certified = true;   // LM-1: every detector / observable deterministic (exact)
            }
            fault_cols.release_x_start();   // final review M6: not kept through the walk and merge (LM-3: either form)
        } catch (const xdem::WriterError& e) {
            res.error = e.what();
            return res;
        }
    }
    // xdem (SPEC §2): strict tau columns, the canonical axes, and every mechanism's time. A mechanism's converter key
    // is sig = flips (< T_TOTAL), x ids (T_TOTAL + i, fault_cols.xid), tau ids (T_TOTAL + nx + a): linear in the fault,
    // so the per-instance conversion (flush) is unchanged; each converted mechanism becomes a line key (xdem_key).
    xdem::TauColumns xdem_tau;
    xdem::Canon xdem_C;
    std::vector<xdem::Axis> xdem_axes;
    std::vector<int> xdem_li_step, xdem_rec_step;    // user step per deferred noise ordinal / per record
    int xdem_tc_step = -1, xdem_Gl = 0;
    const int xdem_nx = (int)fault_cols.xid.size();
    xdem::LineAccumulator acc_xdem;
    xdem::LeanLineAccumulator acc_lean;               // the lean route's merge (LEAN-EXPORT-PLAN Task 4)
    xdem::LeanKeyBuilder lean_key;                    // the lean route's sparse xdem_key (Task 6)
    int xdem_seq_li = -1, xdem_seq = 0;              // mechanism index within one deferred noise instruction: the walk
                                                     // flushes it once per qubit (pair), so the index runs across them
    int xdem_flush = 0;                              // converter call id (atoms resolve within their own call)
    if (is_xdem) {
        if (T_TOTAL != D + O) throw std::logic_error("export_dem: xdem column layout mismatch (engine bug)");
        try {
            xdem_tau = xdem::build_tau_columns(*wm.sk, deferred);
            xdem_C = xdem::canonical_coordinates(wm);
        } catch (const xdem::WriterError& e) {
            res.error = e.what();
            return res;
        }
        for (size_t a = 0; a < wm.header.axes.size(); ++a)
            xdem_axes.push_back({wm.header.axes[a].img, xdem_C.kappa[a], xdem_C.hidden[a],
                                 (int)xdem_C.bhat[a]});
        acc_xdem.set_axes(xdem_axes);                  // the merge guard reads the readout images
        acc_lean.set_axes(xdem_axes);
        // Steps on the source-tracked parse (also the canonical check's, below):
        // user noise site s = the s-th Noise instruction (dem_preamble's stamp), record j = the j-th Measure.
        if (!src) src = std::make_unique<ParsedStim>(parse_stim_circuit(stim_text, /*track_source=*/true));
        const ParsedStim& cp = *src;
        std::vector<int> site_step;
        int nnc = 0, nsites = 0;
        if (cp.ok())
            for (size_t i = 0; i < cp.circuit.stream.size(); ++i) {
                const Instr& ins = cp.circuit.stream[i];
                if (ins.kind == Instr::Kind::Noise) site_step.push_back(cp.stream_step[i]);
                if (ins.kind == Instr::Kind::Measure) xdem_rec_step.push_back(cp.stream_step[i]);
                if (ins.kind == Instr::Kind::Gate && (ins.gate == GateKind::T || ins.gate == GateKind::CS ||
                                                      ins.gate == GateKind::CCZ))
                    ++nnc;
                else if (ins.kind == Instr::Kind::Gate && ins.gate == GateKind::CH)
                    nnc += 2;   // a CH is two writer groups (its two pi/8 rotations, build_writer_model)
            }
        for (const Instr& ins : p.circuit.stream) nsites += ins.kind == Instr::Kind::Noise;
        const int G = (int)wm.header.groups.size();
        if (!cp.ok() || nnc != G || (int)site_step.size() != nsites || (int)xdem_rec_step.size() != M) {
            res.error = "xdem writer: internal pipeline mismatch: the source-tracked parse has " +
                        std::to_string(nnc) + " non-Clifford gates, " + std::to_string(site_step.size()) +
                        " noise sites and " + std::to_string(xdem_rec_step.size()) + " records; the writer " +
                        std::to_string(G) + " groups, " + std::to_string(nsites) + " noise sites and " +
                        std::to_string(M) + " records";
            return res;
        }
        xdem_tc_step = xdem::last_detector_point(cp).step;
        xdem_Gl = xdem::late_group_count(cp);
        if (lean) lean_key.init(wm.header, xdem_axes, fault_cols.xid, T_TOTAL, xdem_Gl);   // xdem_axes final here
        for (const Instr& ins : deferred.stream) {
            if (ins.kind != Instr::Kind::Noise) continue;
            const bool user = ins.origin == Instr::Origin::UserSite && ins.origin_index >= 0 &&
                              ins.origin_index < (int)site_step.size();
            const bool rfold = ins.origin == Instr::Origin::ReadoutFold && ins.origin_index >= 0 &&
                               ins.origin_index < M;
            if (!user && !rfold) {
                res.error = "xdem writer: internal pipeline mismatch: a deferred noise instruction has no user noise "
                            "site or record";
                return res;
            }
            xdem_li_step.push_back(user ? site_step[(size_t)ins.origin_index]
                                        : xdem_rec_step[(size_t)ins.origin_index]);
        }
    }
    // The x-patterns of a converter key: (group, pattern), by group (x ids are group-major).
    auto xdem_x = [&](const Sig& sg) {
        std::vector<std::pair<int, uint32_t>> x;
        for (int32_t id : sg)
            if (id >= T_TOTAL && id < T_TOTAL + xdem_nx) {
                const auto& [g, bit] = fault_cols.xid[(size_t)(id - T_TOTAL)];
                if (x.empty() || x.back().first != g) x.push_back({g, 0u});
                x.back().second |= 1u << bit;
            }
        return x;
    };
    // The xdem line key of a converter key at a user step (SPEC §2: flips, strict tau, E^< / E^> split at Gl, each
    // field folded, the side of t_c).
    auto xdem_key = [&](const Sig& sg, int step) {
        xdem::LineKey K;
        K.side = step > xdem_tc_step ? 1 : 0;
        xdem::Bits f;
        f.w.assign(xdem::words_for((long long)D + O), 0);
        const std::vector<std::pair<int, uint32_t>> x = xdem_x(sg);
        for (int32_t id : sg) {
            if (id < T_TOTAL) f.w[(size_t)id >> 6] ^= 1ull << (id & 63);
            else if (id >= T_TOTAL + xdem_nx) K.tau.push_back(id - T_TOTAL - xdem_nx);
        }
        const size_t A = xdem_axes.size();
        std::vector<uint8_t> Elt(A, 0), Egt(A, 0);
        xdem::fault_rotations(wm.header, x, xdem_Gl, Elt, Egt);
        xdem::fold_field(xdem_axes, Elt, f);
        xdem::fold_field(xdem_axes, Egt, f);
        for (size_t a = 0; a < A; ++a) {
            if (Elt[a]) K.lt.push_back({(int32_t)a, Elt[a]});
            if (Egt[a]) K.gt.push_back({(int32_t)a, Egt[a]});
        }
        for (size_t w = 0; w < f.w.size(); ++w)
            for (uint64_t b = f.w[w]; b; b &= b - 1) K.f.push_back((int32_t)(w * 64 + (size_t)__builtin_ctzll(b)));
        return K;
    };
    // The lean route's line key (Task 6): the same key, touching only the axes / flip words the mechanism hits
    // (xdem_lean.hpp, LeanKeyBuilder). Test only: compared with xdem_key while the key check is on.
    auto lean_line_key = [&](const Sig& sg, int step) {
        xdem::LineKey K = lean_key.build(sg, step > xdem_tc_step ? 1 : 0);
        if (xdem::lean_key_check_on()) {
            const xdem::LineKey Kd = xdem_key(sg, step);
            xdem::lean_key_check_record(!(K < Kd) && !(Kd < K));
        }
        return K;
    };
    int li = 0;
    for (int k = 0; k < (int)deferred.stream.size(); ++k) {
        const Instr& ins = deferred.stream[k];
        if (ins.kind != Instr::Kind::Noise) continue;
        const int cur_li = li;
        const LocationEntry* locp = lean ? nullptr : &table.locations[(size_t)li];   // lean: no table (Task 3)
        ++li;
        auto alt_D = [&](const std::vector<std::pair<int, PauliBasis>>& factors) { return compose_alt(*locp, factors); };
        // The alternatives of this channel instance (ONE fault, disjoint alternatives): the Pauli ones
        // merged by signature (the pre-EXP-fix construction, so an all-Pauli instance pays nothing new),
        // the others as per-fault laws; solved jointly by flush. comp_meta mirrors every non-identity
        // alternative (kind, v0, wdim, v) for the refusal texts and the test-API census.
        std::map<Sig, double> merged;
        const std::vector<std::pair<int, PauliBasis>>* cur_factors = nullptr;
        auto meta_fac = [&]() {                  // census only: the alternative's factors on the last comp_meta entry
            for (auto& [qi, pb] : *cur_factors)
                comp_meta.back().fac.push_back({qi, pb == PauliBasis::X ? 0 : pb == PauliBasis::Y ? 1 : 2});
        };
        // The disjoint probability arguments of this channel instance (stim's threshold test).
        std::vector<double> chan_args;
        Sig sig;
        // A Pauli-correctable alternative with a per-shot RANDOM column sign (PK_RAND, R > 0): the
        // exact joint law of (the targets its randomized reads feed, every PK_RAND column's coin / y
        // bits) — the 3.1.10 correlated-read law extended by the column bits — expanded into
        // signatures, each PK_RAND column flipped on exactly the patterns where its per-shot value
        // is -c0. (No PK_MAG column here: classify_per_shot found every |value| unchanged.)
        auto add_random_sign_alt = [&](const Instr& ins, int k, double prob, const AltRes& Dc,
                                       const std::vector<int>& pk, const std::vector<int>& pbit,
                                       std::vector<PeShotAnalyzer::Law>& plaw) -> bool {
            if (NDEC > 0) append_logical_flip_flips(sig);
            std::vector<int> rt;
            std::vector<std::pair<const int*, const int*>> rt_rows;
            randomized_targets(rt, rt_rows);
            const int m = (int)rt.size();
            std::vector<Pauli> Dts;
            for (int i = 0; i < m; ++i) {
                Pauli Dt(deferred.n);
                int ny = 0;
                for (const int* it = rt_rows[(size_t)i].first; it != rt_rows[(size_t)i].second; ++it) {
                    const auto& [b, q] = reads[(size_t)*it];
                    if (b == 0) { Dt.setx(q); }
                    else if (b == 1) { Dt.setx(q); Dt.setz(q); ++ny; }
                    else { Dt.setz(q); }
                }
                Dt.phase = ny & 3;
                double s0;
                {
                    auto it = target_value_cache.find(rt[(size_t)i]);
                    if (it != target_value_cache.end()) s0 = it->second;
                    else {
                        auto [pp, pm] = framed_expectation(bare_st(), Dt);
                        s0 = pp - pm;
                        target_value_cache.emplace(rt[(size_t)i], s0);
                    }
                }
                if (std::abs(std::abs(s0) - 1.0) > 1e-9) {
                    res.error = loc_name(ins, k) + ": a single fault randomizes a read that feeds a "
                                "target whose noiseless value is not deterministic (" +
                                std::string(rt[(size_t)i] >= LOGFL_START ? "a DECISION column" : "a column") +
                                "); its flip is not defined (refusing)";
                    return false;
                }
                Dts.push_back(std::move(Dt));
            }
            // Owner cap BEFORE the joint walk: the targets' coin rank rho (the support dimension before any Born
            // law; the ONE routine's table, GF(2) only).
            DiagNormalForm nf_l;
            const DemLaw& TL = fault_law(Dc, rt, Dts, nf_l);
            if (TL.rho > kDemSupportCap) { support_cap_refusal(ins, k, *cur_factors, rt, TL.rho); return false; }
            // The joint law of (targets, every PK_RAND column's coin / y parities): the ONE routine's table with the
            // columns' raw read parities next to the targets (twirl-routes T2: no coin branching).
            std::vector<Pauli> raw;
            std::vector<ColBits> cbs;
            for (int r = 0; r < R; ++r)
                if (pk[(size_t)r] == PK_RAND) cbs.push_back(column_bits(r, plaw[(size_t)r], m, raw));
            DiagNormalForm nf_j;
            DemLaw& J = fault_law(Dc, rt, Dts, nf_j, raw);
            std::vector<uint64_t> cj = fault_centre(J, nf_j);
            assert_y_coin_free(J, cbs);
            Sig base;                                           // deterministic part minus the randomized targets
            {
                Sig b0, ecols;
                std::set_difference(sig.begin(), sig.end(), rt.begin(), rt.end(), std::back_inserter(b0));
                for (int r = 0; r < R; ++r)
                    if (pk[(size_t)r] != PK_RAND && pbit[(size_t)r]) ecols.push_back(T + r);
                std::set_union(b0.begin(), b0.end(), ecols.begin(), ecols.end(), std::back_inserter(base));
            }
            // a table pattern x -> its target-space pattern: rt bits, then each PK_RAND column's flip (coin ⊕ sign)
            auto to_targets = [&](const std::vector<uint64_t>& x, const std::vector<int>* g) {
                Sig out;
                for (int i = 0; i < m; ++i) if (wl_bit(x, i)) out.push_back(rt[(size_t)i]);
                for (size_t q = 0; q < cbs.size(); ++q) {
                    int fl = cbs[q].s0 >= 0 && wl_bit(x, cbs[q].s0) ? 1 : 0;
                    if (g) fl ^= (*g)[q];
                    if (fl) out.push_back(T + cbs[q].r);
                }
                std::sort(out.begin(), out.end());
                return out;
            };
            FaultLaw F;
            F.kind = FaultLaw::Kind::Mixture;
            F.p = prob;
            for (const auto& w : J.W) F.W.push_back(to_targets(w, nullptr));
            const std::vector<double>& pz = born_of(J.lam);
            double pmax = 0.0;
            for (double x : pz) pmax = std::max(pmax, x);
            for (size_t z = 0; z < pz.size(); ++z) {
                if (!(pz[z] > 1e-12 * pmax)) continue;          // the support filter (relative 1e-12)
                std::vector<uint64_t> x = cj;
                for (int j = 0; j < J.kappa; ++j) if ((z >> j) & 1) wl_xor(x, J.B[(size_t)j]);
                std::vector<int> g(cbs.size(), 0);              // sign(h(y)) against the noiseless sign
                for (size_t q = 0; q < cbs.size(); ++q) {
                    const int r = cbs[q].r;
                    const double h = col_value(cbs[q], plaw[(size_t)r], x);
                    if (!(std::abs(h) > 1e-9))
                        throw std::logic_error("export_dem: a random-sign column with a zero per-shot value (engine bug)");
                    g[q] = (h < 0) != (col_c0[(size_t)r] < 0) ? 1 : 0;
                }
                const Sig t = to_targets(x, &g);
                Sig sg;
                std::set_union(base.begin(), base.end(), t.begin(), t.end(), std::back_inserter(sg));
                F.sup.push_back({std::move(sg), pz[z]});
            }
            {   // the owner cap on the list law's support: the direction rank of its affine hull
                std::vector<Sig> pts;
                for (const auto& e : F.sup) pts.push_back(e.first);
                for (const Sig& w : F.W) pts.push_back(sig_xor(F.sup.front().first, w));
                const int d = sig_affine_rank(pts);
                if (d > kDemSupportCap) { support_cap_refusal(ins, k, *cur_factors, rt, d); return false; }
                comp_meta.push_back({'L', prob, TL.kappa, TL.rho, d, {}, {}, {}, {}});
                if (census) meta_fac();
                if (census) { comp_meta.back().W = F.W; comp_meta.back().sup = F.sup; }
            }
            comps.push_back(std::move(F));
            comp_alt.push_back(cur_alt);
            return true;
        };
        // The per-shot routing of one alternative that acts non-Pauli on a column or randomizes a
        // read (kept out of line: the Pauli-only hot path above does not pay for it).
        auto per_shot_route = [&](const Instr& ins, int k, double prob,
                                  const AltRes& Dc) __attribute__((noinline)) -> int {
            std::vector<int> pk, pbit;
            std::vector<PeShotAnalyzer::Law> plaw;
            bool any_mag = false, any_rand = false;
            if (!classify_per_shot(ins, k, Dc, pk, pbit, plaw, any_mag, any_rand)) return 2;
            if (!any_mag) {
                // Pauli-correctable per shot. A column whose per-shot sign is random (a fair or
                // detector-correlated sign coin, |value| unchanged) joins the exact joint law of
                // the randomized targets; otherwise every column flip is deterministic.
                if (any_rand && R > 0) return add_random_sign_alt(ins, k, prob, Dc, pk, pbit, plaw) ? 1 : 2;
                for (int r = 0; r < R; ++r)
                    if (pbit[(size_t)r]) sig.push_back(T + r);
            } else {
                // NOT Pauli-correctable: on some branch this single fault changes a column's
                // per-shot magnitude. It must be post-selected away on the detectors it fires;
                // emit NOTHING to the DEM for this alternative.
                std::set<int> aff;                     // detectors this fault fires (deterministic + randomized)
                for (int t : sig) if (t < D) aff.insert(t);              // F = deterministic det flips
                bool F_empty = aff.empty();
                for (int j : randomized_reads)
                    for (int d : detectors_fed_by_read[(size_t)j]) aff.insert(d);
                if (R == 0) {
                    // include_expectations = false (owner decision 2026-10-12; coordinator ruling):
                    // no expectation column was requested, and the detector/observable DEM must
                    // carry EVERY fault on its det/obs signature — the post-select / accepted-branch
                    // machinery exists only for the columns, so here it never flags, drops or
                    // refuses. Its verdict is REPORTED in dropped_expectation_columns (location +
                    // column as the refusal names them) for the cases that refuse with R > 0:
                    // the completeness gate (both forms) and the two "not verified" refusals.
                    // A fault that merely would have been flagged (detected on every branch where
                    // the column changes) is an ordinary mechanism here: nothing to report.
                    auto report_gate = [&](int only_col, const std::string& text) {
                        for (int r = 0; r < (int)expP.size(); ++r) {
                            if (only_col >= 0 ? r != only_col
                                              : !(pk[(size_t)r] != PK_OP || pbit[(size_t)r])) continue;
                            std::string why = text;
                            const std::string tag = "%COL%";
                            for (size_t at; (at = why.find(tag)) != std::string::npos;)
                                why.replace(at, tag.size(), "logical expectation column L" + std::to_string(O + r));
                            report_finding(r, why + " (reported, not refused: include_expectations=False)");
                        }
                    };
                    if (aff.empty()) {
                        for (int r = 0; r < (int)expP.size(); ++r)
                            if (pk[(size_t)r] == PK_MAG)
                                report_gate(r, "non-fault-tolerant circuit (" + loc_name(ins, k) +
                                               "): a single fault changes the magnitude of %COL% but "
                                               "fires no detector; declare the missing post-selection "
                                               "syndrome as detectors");
                    } else if (F_empty) {
                        int bad_col = -1;
                        const int v = per_shot_accepted_violation(Dc, aff, pk, pbit, plaw, bad_col);
                        if (v == 1)
                            report_gate(bad_col, "non-fault-tolerant circuit (" + loc_name(ins, k) +
                                                 "): a single fault changes %COL% (its magnitude, or its "
                                                 "sign on the accepted branch) yet is undetectable on the "
                                                 "accepted branch (all post-selection reads trivial); "
                                                 "declare the missing post-selection syndrome as detectors");
                    }
                    return 0;                        // emitted on its det/obs signature
                }
                if (aff.empty()) {                      // gate: a magnitude change that fires no detector
                    res.error = "non-fault-tolerant circuit (" + loc_name(ins, k) +
                                "): a single fault changes the magnitude of a logical "
                                "expectation but fires no detector; declare the missing "
                                "post-selection syndrome as detectors.";
                    return 2;
                }
                if (F_empty) {                          // randomize-only: is it undetectable on the accepted branch?
                    int bad_col = -1;
                    const int v = per_shot_accepted_violation(Dc, aff, pk, pbit, plaw, bad_col);
                    if (v == 1) {
                        res.error = "non-fault-tolerant circuit (" + loc_name(ins, k) +
                                    "): a single fault changes a logical expectation "
                                    "(its magnitude, or its sign on the accepted branch) "
                                    "yet is undetectable on the accepted branch "
                                    "(all post-selection reads trivial); declare the missing "
                                    "post-selection syndrome as detectors.";
                        return 2;
                    }
                }
                reject_set.insert(aff.begin(), aff.end());
                std::vector<int> sorted_aff(aff.begin(), aff.end());  // std::set is already sorted
                postselect_map[sorted_aff] += prob;     // FLAG this fault: signature + probability
                return 1;                            // emit nothing for this alternative
            }
            return 0;
        };
        auto add_alt = [&](double prob,
                           const std::vector<std::pair<int, PauliBasis>>& factors) -> bool {
            if (prob <= 0.0) return true;
            chan_args.push_back(prob);
            cur_factors = &factors;
            if (!lean) cur_alt = alt_string(ins, factors);   // lean: messages come from the oracle (OD2)
            if (is_xdem) {                                    // a Pauli fault through the skeleton: (flips, x),
                fault_cols.signature(cur_li, factors, sig);
                {                                             // then the tau ids (the fault's strict tau)
                    const size_t s0 = sig.size();
                    const auto& slot = xdem_tau.at[(size_t)cur_li];
                    for (const auto& [qi, pb] : factors) {
                        const auto& e = slot[(size_t)qi];
                        if (pb == PauliBasis::X || pb == PauliBasis::Y)
                            for (int32_t a : e[0]) sig.push_back(T_TOTAL + xdem_nx + a);
                        if (pb == PauliBasis::Z || pb == PauliBasis::Y)
                            for (int32_t a : e[1]) sig.push_back(T_TOTAL + xdem_nx + a);
                    }
                    xdem::parity_reduce_sorted(sig, s0);     // the tau ids (two factors may share one)
                }
                if (census) {
                    census->atoms.push_back(census_atom(ins, factors, prob, sig));
                    census->atoms.back().step = xdem_li_step[(size_t)cur_li];
                    census->atoms.back().x = xdem_x(sig);
                    for (int32_t id : sig)
                        if (id < T_TOTAL) census->atoms.back().flips.push_back(id);
                    acc_xdem.note_atom(xdem_key(sig, xdem_li_step[(size_t)cur_li]), xdem_flush);
                }
                if (!sig.empty()) {
                    merged[sig] += prob;
                    comp_meta.push_back({'P', prob, 0, 0, 0, {}, {}, {}, {}});
                    if (census) { meta_fac(); comp_meta.back().off = sig; }
                }
                return true;
            }
            const AltRes Dc = alt_D(factors);
            alt_signature(Dc, sig);                        // det/obs sig + randomized_reads
            // demfb T3: Q_j = I for every residual op (q_ops' invariant; see there).
            if (fb.has_feedback) {
                const std::vector<QOp> qs = q_ops(Dc.D, cur_li);
                if (!qs.empty()) {
                    const QOp& q0 = qs.front();
                    const FeedbackOp& op = fb.ops[(size_t)q0.op];
                    std::string d = " [" + loc_name(ins, k) + " alt " + cur_alt + "; noise ordinal " +
                                    std::to_string(cur_li) + ", ops before it " + std::to_string(fb_cp_before[(size_t)cur_li]) +
                                    "; op " + std::to_string(q0.op) + " (control rec " + std::to_string(op.control_record) +
                                    ") X support";
                    const std::vector<int>& xw = fb_xwires[(size_t)q0.op];
                    for (size_t i = 0; i < xw.size() && i < 8; ++i) d += " " + std::to_string(xw[i]);
                    d += "; Q on";
                    for (size_t i = 0; i < q0.qz.size() && i < 8; ++i)
                        d += " " + std::to_string(q0.qz[i]) + "(a" + std::to_string((int)Dc.D.a[(size_t)q0.qz[i]]) + ")";
                    d += "]";
                    throw std::logic_error("export_dem: a residual classical feedback does not commute with a later "
                                           "fault's residual (Q != I): the probe and consumer frames disagree (engine bug)" + d);
                }
            }
            // ── Per-shot routing (3.1.12). The det/obs signature is fixed; decide what the fault does
            // to each declared expectation column PER SHOT — the sampler's semantics: <P'> on the
            // state conditioned on EVERY measurement outcome (declared frame records only fix the
            // sign convention). An alternative that acts as a Pauli on every column and randomizes no
            // read multiplies each per-shot value by its operator sign with the records relabelled
            // (exact; the pre-3.1.12 path, untouched). Every other alternative is judged column by
            // column from the column's exact per-shot law (classify_per_shot, dem_pe_per_shot.hpp),
            // never from the read-averaged <D^dag P' D> on the bare state (which disagreed with the
            // sampler whenever a read that is neither a detector nor a frame record is correlated
            // with the column: silent wrong exports, and false refusals / post-select flags).
            bool nonpauli_col = false;                     // dem-ch: a general residual's columns are judged by pull
            if (!expP.empty()) {
                if (Dc.gen) { for (int r = 0; r < (int)expP.size() && !nonpauli_col; ++r) nonpauli_col = column_nonpauli(Dc, r); }
                else nonpauli_col = touches_expectation_nonpauli(Dc.D, expP);
            }
            if (expP.empty() || (!nonpauli_col && randomized_reads.empty())) {
                // Pauli logical action: every declared Pauli sees a clean ±1 sign — a Pauli frame
                // update. Emit normally, including the expectation-column sign flips (R>0).
                append_pauli_expectation_flips(Dc, sig);
            } else {
                // returns 0 continue / 1 return true / 2 return false (refused)
                const int pr = per_shot_route(ins, k, prob, Dc);
                if (pr == 2) return false;
                if (pr == 1) return true;
            }
            // Logical-flip base: parity of deterministic mflip against decision record masks.
            if (NDEC > 0) append_logical_flip_flips(sig);
            const int rk = (int)randomized_reads.size();
            if (rk == 0) {                                // Pauli-only path: byte-identical to before
                if (!sig.empty()) {                       // empty signature folds into identity
                    merged[sig] += prob;
                    comp_meta.push_back({'P', prob, 0, 0, 0, {}, {}, {}, {}});
                    if (census) meta_fac();
                    if (census) comp_meta.back().off = sig;
                }
                return true;
            }
            // ── Randomized reads (3.1.10): the EXACT joint law of the targets they feed. ──────────
            // Until 3.1.9 every randomized read was an INDEPENDENT fair coin (2^rk patterns, equal
            // weight). The read flips of one fault are correlated in general (a CZ between two read
            // wires fixes their parity; two S-residuals can cancel on the state), so that law was
            // wrong: refused as "negative probability" or, when the factoring happened to succeed,
            // silently wrong. The exact law of the targets T_R fed by a randomized read follows from
            // the Clifford algebra on the noiseless bare state: with A_t = s_t . E^dag D_t E (D_t the
            // target's read-parity Pauli, s_t its noiseless sign; the A_t commute), the flip pattern
            // f in GF(2)^{T_R} has Fourier transform  mu^(S) = <prod_{t in S} A_t>,  nonzero only on
            // W = {S : the frame X-part of A_S lies on free generators}. Equivalently: the stabilizer
            // twirl of the Clifford residual E = sum_P c_P P is the Pauli channel whose weights are
            // |sum_{g in G} c_{Pg} chi(g)|^2 per stabilizer coset — each coset with its deterministic
            // stim-style signature. Support size <= 2^rk.
            std::vector<int> rt;                         // sig ids of the randomized targets
            std::vector<std::pair<const int*, const int*>> rt_rows;   // their record rows
            randomized_targets(rt, rt_rows);
            const int m = (int)rt.size();
            std::vector<Pauli> Dts;
            Dts.reserve((size_t)m);
            for (int i = 0; i < m; ++i) {
                Pauli Dt(deferred.n);
                int ny = 0;
                for (const int* it = rt_rows[(size_t)i].first; it != rt_rows[(size_t)i].second; ++it) {
                    const auto& [b, q] = reads[(size_t)*it];
                    if (b == 0) { Dt.setx(q); }
                    else if (b == 1) { Dt.setx(q); Dt.setz(q); ++ny; }
                    else { Dt.setz(q); }
                }
                Dt.phase = ny & 3;
                double s0;
                {
                    auto it = target_value_cache.find(rt[(size_t)i]);
                    if (it != target_value_cache.end()) s0 = it->second;
                    else {
                        auto [pp, pm] = framed_expectation(bare_st(), Dt);
                        s0 = pp - pm;
                        target_value_cache.emplace(rt[(size_t)i], s0);
                    }
                }
                if (std::abs(std::abs(s0) - 1.0) > 1e-9) {
                    // A randomized read feeds a target whose noiseless value is itself random (a
                    // gauge DECISION column): its "flip" has no reference value. Refuse by name.
                    res.error = loc_name(ins, k) + ": a single fault randomizes a read that feeds a "
                                "target whose noiseless value is not deterministic (" +
                                std::string(rt[(size_t)i] >= LOGFL_START ? "a DECISION column" : "a column") +
                                "); its flip is not defined (refusing)";
                    return false;
                }
                Dts.push_back(std::move(Dt));
            }
            // The law of the targets (twirl-routes T1: the ONE routine, cached per (canonical residual, rt)). Owner
            // cap at rho (the coin rank, <= the law's support dimension) BEFORE any Born sum. At a finite
            // clifford_cutoff it is lifted (a coset law is capped by its candidate count below).
            DiagNormalForm nf;
            DemLaw& law = fault_law(Dc, rt, Dts, nf);
            if (!law.classified && tcut < 0 && !erasure && law.rho > kDemSupportCap) {   // erasure: coset lines need no cap
                support_cap_refusal(ins, k, *cur_factors, rt, law.rho, -1.0, law.rho == m ? 2 : 1);   // rho = m: a coset, v = rho
                return false;
            }
            classify_law(law, rt);
            const int vsup = law.vsup;                          // the law's support dimension
            if (law.kind == 'M' && vsup > kDemSupportCap) { support_cap_refusal(ins, k, *cur_factors, rt, vsup); return false; }
            Sig base;                                           // deterministic part minus T_R
            std::set_difference(sig.begin(), sig.end(), rt.begin(), rt.end(), std::back_inserter(base));
            comp_meta.push_back({law.kind, prob, law.kappa, law.rho, vsup, {}, {}, {}, {}});
            if (census) meta_fac();
            std::vector<uint64_t> c = fault_centre(law, nf);
            if (law.kind == 'M') {                              // a Born mixture: its own record (solved by classes)
                FaultLaw F;
                F.kind = FaultLaw::Kind::Mixture;
                F.p = prob;
                for (const auto& w : law.W) F.W.push_back(local_sig(w, rt, Sig{}));
                for (const auto& [off, w] : law.cls) {
                    std::vector<uint64_t> x = c;
                    wl_xor(x, off);
                    F.sup.push_back({local_sig(x, rt, base), w});
                }
                if (census) { comp_meta.back().W = F.W; comp_meta.back().sup = F.sup; }
                comps.push_back(std::move(F));
                comp_alt.push_back(cur_alt);
                return true;
            }
            if (vsup == 0) {                                    // a point law: a Pauli fault
                const Sig sc = local_sig(c, rt, base);
                if (census) comp_meta.back().off = sc;
                if (!sc.empty()) merged[sc] += prob;
                return true;
            }
            if (census) {
                comp_meta.back().off = local_sig(c, rt, base);
                for (const auto& w : law.W) comp_meta.back().W.push_back(local_sig(w, rt, Sig{}));
            }
            // A coset law: an unheralded erasure (its weight split precomputed once per law).
            const ErasureLocal& el = law.el;
            const double cand = erasure ? 0.0 : erasure_candidates(el.split, tcut);   // erasure: no enumeration
            if (cand > std::ldexp(1.0, kDemSupportCap)) {
                if (tcut < 0) support_cap_refusal(ins, k, *cur_factors, rt, vsup, -1.0, 2);
                else support_cap_refusal(ins, k, *cur_factors, rt, vsup, cand);
                return false;
            }
            int base_w = 0;
            for (int32_t t2 : base) if (t2 < D) ++base_w;
            const bool pure = base.empty() && erasure_pure(el.split, c);
            erasures.push_back({&el, rt, std::move(base), base_w, pure, prob, cur_alt, c});
            return true;
        };
        // One channel instance (dem-erasure).
        //   * its Pauli alternatives (merged by signature): the FWHT solve over their group (twirl-routes T3:
        //     one solver; a single signature is exact at p); a failure refuses by
        //     name, or gives the disjoint law under approximate_disjoint_errors (stim's rules); folded
        //     into acc, where identical signatures across instances compose;
        //   * every Clifford alternative: its own record (identical coset laws of the instance merged,
        //     p1 + p2: disjoint alternatives with one conditional law), never folded into acc.
        auto flush = [&]() -> bool {
            const int xdem_fid = xdem_flush++;           // this call's alternatives were noted under xdem_fid
            static const char* kOverMix = "over-mixing categorical channel (a Fourier coefficient of the "
                                          "signature distribution is <= 0); not exactly DEM-convertible";
            static const char* kNegative = "channel is not a product of independent record-flip mechanisms; "
                                           "exact DEM conversion has a negative probability; refusing to approximate";
            std::vector<Mech> mechs;
            DemLawCensus::Inst rec;
            if (census) { rec.k = k; rec.li = cur_li;
                          rec.site = ins.origin == Instr::Origin::UserSite ? ins.origin_index : -1; rec.channel = channel_name(ins); rec.rank = -1; rec.comps = comp_meta; }
            auto done = [&]() {
                if (census) census->inst.push_back(std::move(rec));
                chan_args.clear();
                merged.clear();
                comps.clear();
                comp_alt.clear();
                comp_meta.clear();
                erasures.clear();
            };
            if (comps.empty() && merged.empty() && erasures.empty()) { chan_args.clear(); comp_meta.clear(); return true; }
            const bool has_records = !comps.empty() || !erasures.empty();
            // stim's threshold rule for an instance that needs approximate_disjoint_errors: every
            // probability argument of the channel must be <= the threshold.
            auto approx_ok = [&]() -> bool {
                for (double a : chan_args)
                    if (a > approximate_disjoint_errors) {
                        char buf[192];
                        std::snprintf(buf, sizeof buf, ": the channel has a probability argument "
                                      "(%.17g) larger than the `approximate_disjoint_errors` "
                                      "threshold (%.17g)", a, approximate_disjoint_errors);
                        res.error = loc_name(ins, k) + buf;
                        rec.status = "approx_threshold";
                        return false;
                    }
                return true;
            };
            // RankTooLarge cannot reach here: a Pauli remnant has <= 15 signatures, a support record rank <= v + 1 <= 13
            // (both solved with max_rank 20).
            auto refuse_status = [&](SolveStatus st) {
                assert(st == SolveStatus::Negative || st == SolveStatus::OverMixing);
                if (st == SolveStatus::Negative) res.error = approximate_disjoint_refusal(ins, k, kNegative);
                else res.error = loc_name(ins, k) + ": " + kOverMix;
                rec.status = st == SolveStatus::Negative ? "negative" : "overmix";
            };
            rec.status = "ok";
            // ── The Pauli alternatives.
            if (!merged.empty()) {
                // twirl-routes T3: ONE solver — the exact FWHT over the signature group (stim's categorical ->
                // independent split; a single signature is its own mechanism at p, exact for any p). The census
                // keeps its route names by rank (pauli_old: rank <= 4, the old naive body's domain).
                int rank = -1;
                std::vector<FaultLaw> pc;
                for (const auto& [sg, pr] : merged) {   // one law per signature
                    FaultLaw F;
                    F.p = pr;
                    F.c = sg;
                    pc.push_back(std::move(F));
                }
                SolveStatus st = SolveStatus::Ok;
                if (pc.size() == 1) { rank = 1; if (pc[0].p > 0.0) mechs.push_back({pc[0].c, pc[0].p}); }
                else st = solve_mixture_fwht(pc, 20, mechs, &rank);
                rec.route = rank <= 4 ? "pauli_old" : "joint_fwht";
                rec.rank = rank;
                if (st == SolveStatus::Negative && has_records) {
                    // The Pauli remnant of a MIXED instance (its Clifford alternatives are records): the
                    // instance is first-order by construction (owner ruling Q2), and the remnant alone need
                    // not factor (e.g. DEPOLARIZE1 with Y a record: {X, Z} exclusive). Its disjoint law
                    // (first-order exact), never a refusal the joint instance did not have.
                    mechs.clear();
                    for (const auto& [sg, pr] : merged)
                        if (pr > 0.0 && !sg.empty()) mechs.push_back({sg, pr});
                    rec.status = "first_order";
                } else if (st != SolveStatus::Ok) {
                    // Stim's rule: no exact conversion -> refuse unless approximate_disjoint_errors; then
                    // each distinct signature becomes an independent mechanism with its disjoint probability.
                    if (!(approximate_disjoint_errors > 0.0)) { refuse_status(st); done(); return false; }
                    if (!approx_ok()) { done(); return false; }
                    mechs.clear();
                    for (const auto& [sg, pr] : merged)
                        if (pr > 0.0 && !sg.empty()) mechs.push_back({sg, pr});
                    rec.status = "approx_disjoint";
                }
                if (is_xdem) {                                // xdem: each mechanism to its line (SPEC §2)
                    const int step = xdem_li_step[(size_t)cur_li];
                    if (cur_li != xdem_seq_li) { xdem_seq_li = cur_li; xdem_seq = 0; }
                    for (size_t mi = 0; mi < mechs.size(); ++mi)
                        if (lean) acc_lean.add(lean_line_key(mechs[mi].sig, step), mechs[mi].p, {step, cur_li, xdem_seq++});
                        else acc_xdem.add(xdem_key(mechs[mi].sig, step), mechs[mi].p, {step, cur_li, xdem_seq++}, xdem_fid);
                } else {
                    fold(std::move(mechs));
                }
            }
            if (!has_records) { done(); return true; }
            rec.route = "erasure";
            const std::string where = loc_name(ins, k);
            std::string site;
            if (ins.origin == Instr::Origin::UserSite) site = std::to_string(ins.origin_index);
            auto finish = [&](CliffordRecord&& r) {
                std::sort(r.lines.begin(), r.lines.end(), [](const Mech& x, const Mech& y) { return x.sig < y.sig; });
                r.kept = r.erasure_line ? 1 : (long long)r.lines.size();
                if (census) { rec.kept += r.kept; rec.dropped += r.dropped; }
                r.loc = std::string(channel_name(ins)) + "@" + std::to_string(k) + (site.empty() ? "" : "/site" + site);
                records.push_back(std::move(r));
            };
            // ── Coset records: merge identical laws (canonical RREF of W + c reduced), then enumerate.
            std::vector<int> owner(erasures.size(), -1);        // the first alternative with the same law
            if (erasures.size() > 1) {
                std::vector<std::pair<Sig, std::vector<Sig>>> keys(erasures.size());
                for (size_t i = 0; i < erasures.size(); ++i) {
                    const PendingErasure& e = erasures[i];
                    std::vector<uint64_t> c = e.c;
                    for (const auto& r : e.el->rref) if (wl_bit(c, wl_lead(r))) wl_xor(c, r);
                    keys[i].first = local_sig(c, e.rt, e.base);
                    for (const auto& r : e.el->rref) keys[i].second.push_back(local_sig(r, e.rt, Sig{}));
                }
                for (size_t i = 0; i < erasures.size(); ++i)
                    for (size_t j = 0; j < i && owner[i] < 0; ++j)
                        if (owner[j] < 0 && keys[j] == keys[i]) owner[i] = (int)j;
            }
            for (size_t i = 0; i < erasures.size(); ++i) {
                if (owner[i] >= 0) continue;
                const PendingErasure& e = erasures[i];
                CliffordRecord r;
                r.p = e.p;
                r.alt = e.alt;
                for (size_t j = i + 1; j < erasures.size(); ++j)
                    if (owner[j] == (int)i) { r.p += erasures[j].p; r.alt += "|" + erasures[j].alt; }
                const ErasureSplit& sp = e.el->split;
                r.v = sp.v();
                r.offset = !e.pure;
                if (erasure) {
                    // dem-erasure-out: the law itself at the raw p, one line: offset (c reduced on the split's
                    // pivots, with the fixed part) + the v split rows. No q, no cutoff, no enumeration.
                    std::vector<uint64_t> cr = e.c;
                    cr.resize((size_t)sp.UW, 0);
                    for (size_t g = 0; g < sp.g.size(); ++g)
                        if ((cr[(size_t)sp.piv[g] >> 6] >> (sp.piv[g] & 63)) & 1)
                            for (size_t w = 0; w < cr.size(); ++w) cr[w] ^= sp.g[g][w];
                    for (size_t h = 0; h < sp.h.size(); ++h)
                        if ((cr[(size_t)sp.hpiv[h] >> 6] >> (sp.hpiv[h] & 63)) & 1)
                            for (size_t w = 0; w < cr.size(); ++w) cr[w] ^= sp.h[h][w];
                    r.erasure_line = true;
                    r.eoff = local_sig(cr, e.rt, e.base);
                    for (const auto& g : sp.g) r.egen.push_back(local_sig(g, e.rt, Sig{}));
                    for (const auto& h : sp.h) r.egen.push_back(local_sig(h, e.rt, Sig{}));
                    std::vector<uint64_t> span_bits = cr;
                    for (const auto& g : sp.g) for (size_t w = 0; w < span_bits.size(); ++w) span_bits[w] |= g[w];
                    for (const auto& h : sp.h) for (size_t w = 0; w < span_bits.size(); ++w) span_bits[w] |= h[w];
                    r.supp = local_sig(span_bits, e.rt, e.base);
                    finish(std::move(r));
                    continue;
                }
                double q = 0.0;
                SolveStatus st = erasure_q(r.p, r.v, r.offset, q, &r.qW);
                if (st != SolveStatus::Ok) {
                    // Over-mixing (p >= 1/2 with an offset, p ~ 1 pure): no independent split. Under
                    // approximate_disjoint_errors every pattern at its disjoint mass p / 2^v (as before).
                    if (!(approximate_disjoint_errors > 0.0)) { refuse_status(st); done(); return false; }
                    if (!approx_ok()) { done(); return false; }
                    q = std::ldexp(r.p, -r.v);
                    rec.status = "approx_disjoint";
                }
                // supp: every target the law can flip
                std::vector<uint64_t> span_bits = e.c;
                for (const auto& g : sp.g) for (size_t w = 0; w < span_bits.size(); ++w) span_bits[w] |= g[w];
                for (const auto& h : sp.h) for (size_t w = 0; w < span_bits.size(); ++w) span_bits[w] |= h[w];
                r.supp = local_sig(span_bits, e.rt, e.base);
                const long long n = enumerate_erasure_split(sp, e.c, e.base_w, e.pure, tcut,
                    [&](const std::vector<uint64_t>& x) {
                        if (q > 0.0) r.lines.push_back({local_sig(x, e.rt, e.base), q});
                    });
                const double total = std::ldexp(1.0, r.v) - (e.pure ? 1.0 : 0.0);
                r.dropped = std::ldexp(r.p, -r.v) * (total - (double)n);
                finish(std::move(r));
            }
            // ── Explicit-support (L) and Born-mixture (M) records: each its own exact split by classes (twirl-routes
            // T1b: solve_mixture_classes, 2^dim(H / span W) work — the coins' span is a closed form).
            for (size_t i = 0; i < comps.size(); ++i) {
                const FaultLaw& F = comps[i];
                CliffordRecord r;
                r.support = true;
                r.p = F.p;
                r.alt = comp_alt[i];
                const SolveStatus st = solve_mixture_classes(F, D, tcut, r.lines, r.dropped, r.v, r.supp);
                if (st != SolveStatus::Ok) {
                    // Negative (no independent split, e.g. a uniform two-point law with an offset): the fault's
                    // disjoint law, first-order exact like an offset coset (owner ruling Q1), never a refusal
                    // (the pre-erasure joint instance solve may have been exact). Otherwise stim's rule.
                    const bool first_order = st == SolveStatus::Negative;
                    if (!first_order && !(approximate_disjoint_errors > 0.0)) { refuse_status(st); done(); return false; }
                    if (!first_order && !approx_ok()) { done(); return false; }
                    rec.status = first_order ? "first_order" : "approx_disjoint";
                    r.first_order = true;
                }
                finish(std::move(r));
            }
            done();
            return true;
        };
        switch (ins.channel) {
            case NoiseChannel::X_ERROR:
            case NoiseChannel::Y_ERROR:
            case NoiseChannel::Z_ERROR: {
                PauliBasis pb = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                              : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y
                                                                     : PauliBasis::Z;
                for (int qi = 0; qi < (int)ins.qubits.size(); ++qi) {
                    if (!add_alt(ins.probs[0], {{qi, pb}})) return res;
                    if (!flush()) return res;
                }
                break;
            }
            case NoiseChannel::DEPOLARIZE1:
            case NoiseChannel::PAULI_CHANNEL_1: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
                for (int qi = 0; qi < (int)ins.qubits.size(); ++qi) {
                    for (int i = 0; i < 3; ++i)
                        if (!add_alt(dep ? ins.probs[0] / 3.0 : ins.probs[i], {{qi, pc1_slot_basis(i)}}))
                            return res;
                    if (!flush()) return res;
                }
                break;
            }
            case NoiseChannel::DEPOLARIZE2:
            case NoiseChannel::PAULI_CHANNEL_2:
                for (int qi = 0; qi + 1 < (int)ins.qubits.size(); qi += 2) {
                    for (int idx = 0; idx < 15; ++idx) {
                        const double prob = ins.channel == NoiseChannel::DEPOLARIZE2
                                                ? ins.probs[0] / 15.0 : ins.probs[idx];
                        std::vector<std::pair<int, PauliBasis>> factors;
                        // Stim slot order: the shared pc2_* definition (circuit_ir.hpp).
                        if (const int c = pc2_first_code(idx))  factors.push_back({qi, pauli_code_basis(c)});
                        if (const int c = pc2_second_code(idx)) factors.push_back({qi + 1, pauli_code_basis(c)});
                        if (!add_alt(prob, factors)) return res;
                    }
                    if (!flush()) return res;
                }
                break;
            case NoiseChannel::CORRELATED: {
                // Stim's rules for an E/ELSE chain, reproduced: a chain with an ELSE term requires
                // `approximate_disjoint_errors` (its refusal, naming ELSE_CORRELATED_ERROR), every
                // categorical component q_t = p_t prod_{j<t}(1-p_j) must be <= the threshold, and
                // then EVERY term is its own independent mechanism at q_t (no exact categorical ->
                // independent solve, no within-chain merging — two terms with one signature compose
                // as p (+) q, exactly stim's `error(0.172) D0` for E(0.1) X0 / ELSE(0.1) Y0). A
                // single-term chain `E(p) P` is exact (one mechanism at p, no option needed), like
                // X_ERROR. Each term is routed through add_alt + flush like an X_ERROR alternative,
                // so the per-shot expectation routing / post-select flags / randomized-read law of
                // a magic-bearing circuit apply per term.
                const std::vector<double> q = corr_outcome_probs(ins);
                if (corr_terms(ins) > 1) {
                    if (!(approximate_disjoint_errors > 0.0)) {
                        Instr named = ins;              // stim names the ELSE instruction
                        res.error = std::string("Encountered the operation ELSE_CORRELATED_ERROR during "
                                    "error analysis, but this operation requires the "
                                    "`approximate_disjoint_errors` option to be enabled.\n"
                                    "If you're calling from python, using xtim.Circuit.detector_error_model, "
                                    "you need to add the argument approximate_disjoint_errors=True.\n"
                                    "(xtim: ") + loc_name(named, k) + ": an E/ELSE chain of " +
                                    std::to_string(corr_terms(ins)) + " terms; each term becomes an "
                                    "independent mechanism at its categorical probability, as in stim)";
                        return res;
                    }
                    for (double a : q)
                        if (a > approximate_disjoint_errors) {
                            char buf[192];
                            std::snprintf(buf, sizeof buf, ": the chain has a component probability "
                                          "(%.17g) larger than the `approximate_disjoint_errors` "
                                          "threshold (%.17g)", a, approximate_disjoint_errors);
                            res.error = loc_name(ins, k) + buf;
                            return res;
                        }
                }
                for (int t = 0; t < corr_terms(ins); ++t) {
                    std::vector<std::pair<int, PauliBasis>> factors;
                    for (size_t qi = 0; qi < ins.qubits.size(); ++qi)
                        if (const int c = corr_code(ins, t, (int)qi)) factors.push_back({(int)qi, pauli_code_basis(c)});
                    if (factors.empty()) continue;      // identity term: no mechanism
                    if (!add_alt(q[(size_t)t], factors)) return res;
                    if (!flush()) return res;
                }
                break;
            }
        }
    }

    // ── Readout flips: the (p) argument of the M-family (`M(p) q`, `MX(p)`, `MY(p)`, `MR(p)`,
    // `MRX(p)`, `MRY(p)`, `MPP(p)`, `MXX(p)`...) — Stim's measurement flip probability.
    //
    // Deferral rebuilds the terminal Measures from a (qubit, basis) list and DROPS the user's
    // readout_flip_p, so it is read off the pre-deferral coherent circuit exactly as the
    // sampler reads its record-flip metadata (sampler.cpp): coherentize copies Measures
    // verbatim and in order, deferral never reorders Measures among themselves, hence
    // nr.coherent's Measure order == terminal-read (== reads[]) order.
    //
    // A readout flip is a CLASSICAL flip of record j alone: no state action, so nothing to
    // propagate, and no magnitude question (never a post-select flag).
    // Its signature is every target whose record set reads j, by parity:
    //   * detectors / observables     — targets_fed_by_read (a dropped gauge observable's row is empty);
    //   * PAULI_EXPECTATION columns   — the declared byproduct frame (obs_frame): the sampler
    //                                   applies rec_flip BEFORE the emask fold, so a flipped
    //                                   frame record flips the reported sign of that column;
    //   * DECISION logical-flip cols  — decisions_fed_by_read.
    // Exactly Stim's model: one independent `error(p) <targets reading j>` per record, folded
    // into the same cross-mechanism XOR composition as every other mechanism (an unread
    // record's empty signature is dropped like any undetectable error).
    {
        std::vector<std::vector<uint64_t>> emask;          // per expectation column, MW words
        if (R > 0) {
            emask.assign((size_t)R, std::vector<uint64_t>((size_t)(MW ? MW : 1), 0));
            int r = 0;
            for (const Instr& ins : deferred.stream)
                if (ins.kind == Instr::Kind::Observable) {
                    if (r < R)
                        for (int idx : ins.obs_frame)
                            if (idx >= 0 && idx < M)
                                emask[(size_t)r][(size_t)(idx >> 6)] ^= 1ull << (idx & 63);
                    ++r;
                }
        }
        auto read_bit = [&](const std::vector<uint64_t>& m, size_t row_off, int j) -> bool {
            return (m[row_off + (size_t)(j >> 6)] >> (j & 63)) & 1;
        };
        int j = 0;
        for (const Instr& ins : nr.coherent.stream) {
            if (ins.kind != Instr::Kind::Measure) continue;
            assert(j < M && "readout-flip metadata: more user Measures than terminal reads");
            const double pflip = ins.readout_flip_p;
            if (pflip > 0.0) {
                Sig sig(targets_fed_by_read[(size_t)j].begin(),   // QW-E D5: the targets reading j, ascending (no T scan)
                        targets_fed_by_read[(size_t)j].end());
                for (int r = 0; r < R; ++r)
                    if (!col_dropped[(size_t)r] && read_bit(emask[(size_t)r], 0, j)) sig.push_back(T + r);
                if (NDEC > 0)
                    for (int k2 : decisions_fed_by_read[(size_t)j]) sig.push_back(LOGFL_START + k2);
                if (census && is_xdem) {                     // the xdem atom census: the record flip's address
                    DemLawCensus::Atom a;
                    a.record = j;
                    a.p = pflip;
                    a.sig = sig;
                    a.flips.assign(sig.begin(), sig.end());   // a readout flip's key is its flips alone
                    a.step = xdem_rec_step[(size_t)j];
                    acc_xdem.note_atom(xdem_key(sig, a.step), xdem_flush);
                    census->atoms.push_back(std::move(a));
                }
                if (!sig.empty() && is_xdem) {               // xdem: f = its flips, tau = E = empty
                    const int step = xdem_rec_step[(size_t)j];
                    if (lean) acc_lean.add(lean_line_key(sig, step), pflip, {step, li + j, 0});
                    else acc_xdem.add(xdem_key(sig, step), pflip, {step, li + j, 0}, xdem_flush);
                } else if (!sig.empty()) {
                    std::vector<Mech> one;
                    one.push_back({std::move(sig), pflip});
                    fold(std::move(one));
                }
                if (is_xdem) ++xdem_flush;                    // one converter call per readout flip
            }
            ++j;
        }
        assert(j == M && "readout-flip metadata: Measure count mismatch with the terminal reads");
    }

    // The flagged not-Pauli-correctable faults (the primary output): each {sorted signature, prob}.
    // std::map iterates by sorted key, so postselect_faults is sorted by signature.
    res.postselect_faults.assign(postselect_map.begin(), postselect_map.end());
    // The post-selection syndrome: detectors fired by any not-Pauli-correctable fault. std::set
    // iterates ascending, so reject_detectors is sorted. (Derived union convenience.)
    res.reject_detectors.assign(reject_set.begin(), reject_set.end());

    // xdem emission (SPEC §2, §3): no rotating line -> the caller returns the ordinary export (SPEC §2 last row).
    if (is_xdem) {
        const bool rot = lean ? acc_lean.any_rotation() : acc_xdem.any_rotation();
        if (census) {
            const std::vector<int> al = acc_xdem.atom_lines();
            if (al.size() != census->atoms.size())
                throw std::logic_error("export_dem: xdem atom map size mismatch (engine bug)");
            for (size_t i = 0; i < al.size(); ++i) census->atoms[i].line = rot ? al[i] : -1;
        }
        if (!rot) {
            res.ok = true;
            return res;
        }
        if (require_canonical) {
            const std::string why = xdem::canonical_violation(*src);
            if (!why.empty()) {
                res.error = "format=\"xdem\" with require_canonical=True: the circuit is not in canonical single-stage "
                            "form: " + why + " (refusing)";
                return res;
            }
        }
        try {
            const xdem::File F = lean ? acc_lean.to_file(D, O, xdem_C, xdem_axes)
                                      : acc_xdem.to_file(D, O, xdem_C, xdem_axes);
            res.dem = lean ? xdem::emit_fast(F) : xdem::emit(F);   // Task 5: byte-identical emitters
        } catch (const xdem::WriterError& e) {
            res.error = e.what();
            return res;
        }
        if (xc) xc->extended = true;
        res.ok = true;
        return res;
    }

    // Emit. acc is sorted by signature (lexicographic target lists == Stim's order).
    std::string out;
    std::vector<uint8_t> covered((size_t)(T_TOTAL > 0 ? T_TOTAL : 1), 0);
    char num[64];
    std::vector<DecomposedMech> decomposed;
    // dem-erasure: the Clifford records' 1-2-detector lines join acc's edge model (decompose_errors).
    std::map<Sig, double> record_edges;
    DemEdgeModel edge_model;
    if (decompose_errors) {
        for (const CliffordRecord& r : records)
            for (const Mech& mm : r.lines) {
                double& a = record_edges[mm.sig];
                a = a + mm.p - 2.0 * a * mm.p;
            }
        std::string derr;
        if (!decompose_acc(acc, D, ignore_decomposition_failures, decomposed, derr, &record_edges)) {
            res.error = derr;
            return res;
        }
        if (!records.empty()) edge_model = dem_edge_model(acc, D, &record_edges);
    } else {
        for (const auto& kv : acc) decomposed.push_back({{kv.first}, kv.second});  // pass-through
    }
    auto emit_line = [&](const DecomposedMech& m, const char* tag) {
        if (m.p <= 0.0) return;
        std::snprintf(num, sizeof num, "%.17g", m.p);
        out += "error";
        out += tag;
        out += "(";
        out += num;
        out += ")";
        for (size_t ci = 0; ci < m.components.size(); ++ci) {
            if (ci) out += " ^";
            for (int32_t t : m.components[ci]) {
                covered[t] = 1;
                std::snprintf(num, sizeof num, t < D ? " D%d" : " L%d", t < D ? t : t - D);
                out += num;
            }
        }
        out += "\n";
    };
    auto target_name = [&](int32_t t) {
        return (t < D ? "D" + std::to_string(t) : "L" + std::to_string(t - D));
    };
    if (!records.empty()) {                  // the loud summary (design note: dropped Clifford mass)
        double dm = 0.0;
        for (const CliffordRecord& r : records) dm += r.dropped;
        res.clifford_faults = (int)records.size();
        res.clifford_dropped_mass = dm;
        char hb[160];
        std::snprintf(hb, sizeof hb, "# clifford faults=%d cutoff=%s dropped_mass=%.17g%s\n", (int)records.size(),
                      tcut < 0 ? "inf" : std::to_string(tcut).c_str(), dm, erasure ? " mode=erasure" : "");
        out += hb;
    }
    for (const auto& m : decomposed) emit_line(m, "");
    for (size_t id = 0; id < records.size(); ++id) {
        const CliffordRecord& r = records[id];
        char hb[256];
        std::snprintf(hb, sizeof hb, "# clifford f%zu: p=%.17g v=%d ", id, r.p, r.v);
        out += hb;
        if (r.support) out += r.first_order ? "law=support first_order=1" : "law=support";
        else {
            out += r.offset ? "offset=1" : "offset=0";
            if (r.offset && !r.erasure_line) { std::snprintf(hb, sizeof hb, " qW=%.17g", r.qW); out += hb; }
        }
        out += " support=";
        for (size_t i = 0; i < r.supp.size(); ++i) { if (i) out += ","; out += target_name(r.supp[i]); }
        out += " loc=" + r.loc + " alt=" + r.alt;
        std::snprintf(hb, sizeof hb, " kept=%lld dropped=%.17g", r.kept, r.dropped);
        out += hb;
        out += erasure && r.support ? " (not a coset; patterns)\n" : "\n";
        const std::string tag = "[clifford:f" + std::to_string(id) + "]";
        if (r.erasure_line) {                // dem-erasure-out: the coset law as one line, at the raw p
            std::snprintf(num, sizeof num, "%.17g", r.p);
            out += "erasure" + tag + "(" + num + ")";
            auto put = [&](const Sig& sg) {
                for (int32_t t : sg) {
                    covered[t] = 1;
                    std::snprintf(num, sizeof num, t < D ? " D%d" : " L%d", t < D ? t : t - D);
                    out += num;
                }
            };
            put(r.eoff);
            for (const Sig& g : r.egen) { out += " |"; put(g); }
            out += "\n";
            continue;
        }
        std::vector<DecomposedMech> dl;
        for (const Mech& mm : r.lines) {
            dl.clear();
            if (decompose_errors) {
                std::string derr;
                if (!decompose_one(edge_model, mm.sig, mm.p, D, ignore_decomposition_failures, dl, derr)) {
                    res.error = derr;
                    return res;
                }
            } else {
                dl.push_back({{mm.sig}, mm.p});
            }
            for (const auto& m : dl) emit_line(m, tag.c_str());
        }
    }
    // M-B: detector coordinate lines. Stim ALWAYS emits a `detector(coords) D#` line when the
    // DETECTOR carries coordinates (even if the detector is also covered by an error mechanism);
    // a coordinate-free detector emits a bare `detector D#` only when otherwise uncovered (the
    // pre-M-B behavior). Coords are shift-resolved at parse time, so a flat `detector(c..) D#`
    // line reproduces Stim's get_detector_coordinates() exactly. Number format %.19g byte-matches
    // Stim's coordinate emission (e.g. 0.1 -> 0.1000000000000000056, 2.0 -> 2).
    static const std::vector<double> kNoCoords;
    for (int d = 0; d < D; ++d) {
        const std::vector<double>& dc =
            d < (int)p.detector_coords.size() ? p.detector_coords[d] : kNoCoords;
        if (!dc.empty()) {
            out += "detector(";
            for (size_t i = 0; i < dc.size(); ++i) {
                if (i) out += ", ";
                std::snprintf(num, sizeof num, "%.19g", dc[i]);
                out += num;
            }
            std::snprintf(num, sizeof num, ") D%d\n", d);
            out += num;
        } else if (!covered[d]) {
            std::snprintf(num, sizeof num, "detector D%d\n", d);
            out += num;
        }
    }
    for (auto& kv : p.observables)
        if (!obs_dropped[kv.first] && !covered[D + kv.first]) {   // dropped gauge obs -> no L# at all
            std::snprintf(num, sizeof num, "logical_observable L%d\n", kv.first);
            out += num;
        }
    for (int r = 0; r < R; ++r)                 // uncovered expectation columns: bare
        if (!covered[T + r] && !col_dropped[(size_t)r]) {   // dropped: no L line at all                  // lines, same parseability convention
            std::snprintf(num, sizeof num, "logical_observable L%d\n", O + r);
            out += num;
        }
    // Logical-flip columns (DECISION): one L# per decision index k.
    for (int k = 0; k < NDEC; ++k) {
        if (!covered[(size_t)(LOGFL_START + k)])
            { std::snprintf(num, sizeof num, "logical_observable L%d\n", O + R + k); out += num; }
    }
    res.dem = std::move(out);
    res.ok = true;
    return res;
}

}  // namespace

// Exact per-detector / per-observable determinism partition — the standalone twin of
// export_dem's circuit-level determinism gate (same setup, same parity-Pauli Born check),
// but WITHOUT the noise-channel DEM machinery, so it answers even when the full DEM refuses.
// The shared algebra is guarded against drift by test_dem_determinism_matches_export.
DeterminismResult detector_determinism(const std::string& stim_text) {
    // twirl-routes T3 (coordinator 2026-10-30): no exact-construction mark — this is a bare-state Born check
    // (the noiseless frame + parity-Pauli expectations), not the exact sampler.
    DeterminismResult res;
    ParsedStim p = parse_stim_circuit(stim_text);
    if (!p.ok()) { res.parse_errors = std::move(p.errors); return res; }

    // Sampler-preset normalization: coherentize feedback that crosses a non-Clifford gate into
    // coherent gates, keep the pre-strip `coherent` circuit for the feedback plan, and strip every
    // residual (non-crossing) ControlledPauli from the consumer circuit so it can defer. Unlike the
    // old Reject preset this ACCEPTS classical Pauli feedback — the sampler handles it, and so do we
    // below via the exact record relabel. For a feedback-free circuit this is identical to before.
    NormalizePolicy pol;
    pol.feedback = NormalizePolicy::Feedback::KeepCoherent;
    pol.defer = true;
    pol.want_map = false;
    NormalizeResult nr = normalize(p.circuit, pol);
    Circuit deferred = std::move(nr.normalized);   // feedback-free, deferred, Hadamard-free

    PropagationTable table = build_propagation_table(deferred, /*ppr_retry=*/true);  // the SAMPLER'S class: this is a bare-state diagnostic (diagnose serves every circuit the sampler serves), not a DEM export
    if (!table.all_in_class) {
        res.rejected = true;
        res.reject_gate_index = table.rejects.empty() ? -1 : table.rejects[0].reject_gate_index;
        return res;
    }
    FramedBareState bare = build_bare_state_framed(deferred);
    if (bare.rejected) {
        res.rejected = true;
        res.reject_gate_index = bare.reject_gate_index;
        return res;
    }
    if (bare.state.U.expanded()) bare.state.U.prepare_queries();   // QW-B: the per-detector reads use it (as dem_build_bare)

    // Terminal reads, record order (pauli 0:X 1:Y 2:Z, qubit) — identical to export_dem.
    std::vector<std::pair<int, int>> reads;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure)
            reads.push_back({ins.basis == PauliBasis::X ? 0
                             : ins.basis == PauliBasis::Y ? 1 : 2, ins.qubits[0]});
    const int M = (int)reads.size();
    {   // deferral places every terminal read on its own wire; the parity-Pauli Born form
        // requires it (matches export_dem's identical guard).
        std::vector<int> seen;
        for (auto& [b, q] : reads) seen.push_back(q);
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            res.error = "two terminal reads share a wire; the determinism analysis does not "
                        "cover this (matches DEM export)";
            return res;
        }
    }

    // Observable index space (needed before the feedback plan).
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    const int O = max_obs + 1;

    // Classical Pauli feedback (`CX rec[-k] q` not crossing a non-Clifford gate) is stripped from
    // the bare pipeline above; its effect is folded back in EXACTLY here — the same triangular
    // record relabel the sampler applies to its noiseless reference (sampler.cpp), but applied to
    // each channel's record SET before the parity Pauli is formed. A read the feedback flips is, in
    // the feedback-free circuit, that read with the control record XORed in; so a channel's parity
    // gains the control record iff its record set overlaps the op's flip mask oddly.
    FeedbackPlan fb = build_feedback_plan(nr.coherent, M, O);
    if (!fb.ok) {                          // a controlled-Pauli left the Pauli class / is acausal
        res.rejected = true;
        res.reject_gate_index = fb.reject_gate_index;
        return res;
    }
    // The sampler applies its forward triangular pass to reference VALUES (refmw); a channel's record SET
    // reads the transpose of that map: the shared helper FeedbackFold (feedback.hpp, demfb T1; export_dem
    // folds with the same helper).
    FeedbackFold fbf;
    if (fb.has_feedback) fbf.build(fb, M);
    auto relabel = [&](const std::vector<int>& recs) -> std::vector<int> {
        if (!fb.has_feedback) return recs;
        return fbf.fold(recs);
    };

    // The recorded-bit sign (dem_export.hpp, the DeterminismResult convention). The sampler records
    // r = L u, u_k = m_k ^ inv_k (m = the deferred read outcome, inv_k = the user's `!` ^ the sign
    // eliminate_hadamards stamps on the deferred read — the sampler's rec_invert, sampler.cpp; L = the
    // unitriangular feedback relabel). So XOR_S r = XOR_{S'} m ^ XOR_{S'} inv over the relabeled set S',
    // and E (-1)^{XOR_S r} = (-1)^{XOR_{S'} inv} <prod_{S'} P_k>. inv is assembled exactly as the sampler
    // does (and as export_dem's refusal value does): (a) off the coherent pre-strip circuit (deferral drops
    // the user's `!`), (b) off the deferred circuit (the stamp). The noiseless M(p) coin is not included.
    const std::vector<uint8_t> rinv = record_inverts(nr.coherent, deferred, M);

    // Parity Pauli of a record set on the bare state -> signed <P> = pp - pm, times the recorded-bit
    // sign (-1)^{XOR inv}; sets `deterministic` (|<P>| == 1) and `oor` (an out-of-range record index).
    // The Pauli is byte-for-byte the same construction as export_dem's check_deterministic lambda.
    std::vector<uint8_t> inc((size_t)M, 0);
    auto parity_expectation = [&](const std::vector<int>& recs, bool& oor,
                                  bool& deterministic) -> double {
        oor = false;
        std::fill(inc.begin(), inc.end(), (uint8_t)0);
        for (int idx : recs) {
            if (idx < 0 || idx >= M) { oor = true; deterministic = false; return 0.0; }
            inc[idx] ^= 1;
        }
        Pauli P(deferred.n);
        int ny = 0, any = 0, sgn = 0;
        for (int j = 0; j < M; ++j) {
            if (!inc[j]) continue;
            any = 1;
            sgn ^= rinv[(size_t)j];
            const auto& [b, q] = reads[j];
            if (b == 0) { P.setx(q); }
            else if (b == 1) { P.setx(q); P.setz(q); ++ny; }
            else { P.setz(q); }
        }
        if (!any) { deterministic = true; return 1.0; }   // empty parity is the constant +1
        P.phase = ny & 3;
        auto [pp, pm] = framed_expectation(bare.state, P);
        deterministic = (std::max(pp, pm) > 1.0 - 1e-9);
        return sgn ? pm - pp : pp - pm;
    };

    // Detectors (indices 0..D-1).
    const int D = (int)p.detectors.size();
    res.num_detectors = D;
    res.detector_parity.assign((size_t)D, 0.0);
    for (int d = 0; d < D; ++d) {
        const std::vector<int>& recs = p.detectors[d];
        for (int idx : recs)                       // OOR guard on the ORIGINAL record set
            if (idx < 0 || idx >= M) {
                char buf[96];
                std::snprintf(buf, sizeof buf,
                              "detector D%d references an out-of-range measurement", d);
                res.error = buf;
                return res;
            }
        bool oor = false, det = false;
        double v = parity_expectation(relabel(recs), oor, det);   // relabeled recs are in-range
        res.detector_parity[d] = v;
        (det ? res.deterministic_detectors : res.gauge_detectors).push_back(d);
    }

    // Declared observables (OBSERVABLE_INCLUDE indices; the space may have gaps).
    res.num_observables = O;
    if (O > 0)
        res.observable_parity.assign((size_t)O, std::numeric_limits<double>::quiet_NaN());
    std::vector<int> obs_ids;
    for (auto& kv : p.observables) obs_ids.push_back(kv.first);
    std::sort(obs_ids.begin(), obs_ids.end());   // stable ascending order
    for (int id : obs_ids) {
        const std::vector<int>& recs = p.observables.at(id);
        for (int idx : recs)
            if (idx < 0 || idx >= M) {
                char buf[96];
                std::snprintf(buf, sizeof buf,
                              "observable L%d references an out-of-range measurement", id);
                res.error = buf;
                return res;
            }
        bool oor = false, det = false;
        double v = parity_expectation(relabel(recs), oor, det);
        res.observable_parity[id] = v;
        (det ? res.deterministic_observables : res.gauge_observables).push_back(id);
    }

    res.ok = true;
    return res;
}

// Exact verify of each declared PAULI_EXPECTATION byproduct-frame (Task 4). Reuses
// detector_determinism's setup (deferred circuit + build_bare_state_framed + terminal reads),
// then adds expP + the fixed-generator anticommutation solve. See dem_export.hpp for the method.
FrameCheckResult expectation_frame_check(const std::string& stim_text) {
    FrameCheckResult res;
    ParsedStim p = parse_stim_circuit(stim_text);
    if (!p.ok()) { res.parse_errors = std::move(p.errors); return res; }

    // Sampler-preset normalization (feedback-free deferred, Hadamard-free) — identical to
    // detector_determinism so the bare state / read order match by construction.
    NormalizePolicy pol;
    pol.feedback = NormalizePolicy::Feedback::KeepCoherent;
    pol.defer = true;
    pol.want_map = false;
    NormalizeResult nr = normalize(p.circuit, pol);
    Circuit deferred = std::move(nr.normalized);

    // Classically-controlled Pauli feedback would relabel the record space the declared frame
    // lives in; that fold is out of scope for the frame verifier. Refuse cleanly if present.
    if (circuit_has_feedback(nr.coherent)) {
        res.error = "classically-controlled Pauli feedback is not handled by the "
                    "PAULI_EXPECTATION frame verifier (declared-frame record relabel deferred)";
        return res;
    }

    PropagationTable table = build_propagation_table(deferred, /*ppr_retry=*/true);  // the SAMPLER'S class: this is a bare-state diagnostic (diagnose serves every circuit the sampler serves), not a DEM export
    if (!table.all_in_class) {
        res.rejected = true;
        res.reject_gate_index = table.rejects.empty() ? -1 : table.rejects[0].reject_gate_index;
        return res;
    }
    FramedBareState bare = build_bare_state_framed(deferred);
    if (bare.rejected) {
        res.rejected = true;
        res.reject_gate_index = bare.reject_gate_index;
        return res;
    }

    // Terminal reads, record order (pauli 0:X 1:Y 2:Z, qubit) — identical to detector_determinism.
    std::vector<std::pair<int, int>> reads;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure)
            reads.push_back({ins.basis == PauliBasis::X ? 0
                             : ins.basis == PauliBasis::Y ? 1 : 2, ins.qubits[0]});
    const int M = (int)reads.size();
    {   // deferral places every terminal read on its own wire (matches determinism's guard).
        std::vector<int> seen;
        for (auto& [b, q] : reads) seen.push_back(q);
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            res.error = "two terminal reads share a wire; the frame analysis does not cover this";
            return res;
        }
    }

    const FramedSuperposition& st = bare.state;
    const int n = st.n();

    // Fixed stabilizer generators: g_a = U Z_a U† = st.U.Zrow[a] for a NOT in `free`. The `free`
    // (logical) rows are EXCLUDED — magic makes expP anticommute with a free generator (the 1/√2
    // magnitude), and including it would make the solve spuriously unsolvable.
    std::vector<uint8_t> is_free((size_t)(n ? n : 1), 0);
    for (int f : st.free) if (f >= 0 && f < n) is_free[(size_t)f] = 1;
    std::vector<int> fixed_rows;
    for (int a = 0; a < n; ++a) if (!is_free[(size_t)a]) fixed_rows.push_back(a);
    const int nf = (int)fixed_rows.size();
    const int W = (nf + 63) / 64;                 // words of the avec (over fixed generators)
    const int PW = (M + 63) / 64;                 // words of the provenance (over M reads)

    // avec(P): bit r set iff P anticommutes with the r-th fixed generator. S2-T4b: the generators are read as the
    // frame's sparse certified rows (Frame::certified_rows, no sign: commutation only) — the frame is not expanded.
    const std::vector<SparsePauli> fixed_gens = st.U.certified_rows(st.free);
    auto avec = [&](const Pauli& P) {
        std::vector<uint64_t> v((size_t)(W ? W : 1), 0);
        for (int r = 0; r < nf; ++r)
            if (anticommute_bit(P, fixed_gens[(size_t)r]))
                v[(size_t)(r >> 6)] |= 1ull << (r & 63);
        return v;
    };
    auto read_pauli = [&](int j) {
        Pauli P(n);
        const auto& [b, q] = reads[(size_t)j];
        if (b == 0) { P.setx(q); }
        else if (b == 1) { P.setx(q); P.setz(q); }
        else { P.setz(q); }
        return P;                                 // phase irrelevant for anticommutation
    };

    // GF(2) echelon basis over the read columns, with per-basis-vector provenance (the read set
    // whose avec-XOR equals the vector). Free/dependent columns (incl. avec==0 deterministic
    // reads) never become pivots ⇒ excluded from every solution (canonical minimal solution).
    struct BasisVec { std::vector<uint64_t> vec, prov; int pivot; };
    std::vector<BasisVec> basis;
    std::vector<int> which((size_t)(nf ? nf : 1), -1);   // fixed-row -> index into basis (or -1)
    auto reduce = [&](std::vector<uint64_t>& v, std::vector<uint64_t>& prov) {
        for (int r = 0; r < nf; ++r) {           // low->high: each pivot vec has no bit below its pivot
            if (!((v[(size_t)(r >> 6)] >> (r & 63)) & 1)) continue;
            if (which[(size_t)r] < 0) continue;  // no basis vec here yet -> candidate new pivot
            const BasisVec& bv = basis[(size_t)which[(size_t)r]];
            for (int w = 0; w < W; ++w) v[(size_t)w] ^= bv.vec[(size_t)w];
            for (int w = 0; w < PW; ++w) prov[(size_t)w] ^= bv.prov[(size_t)w];
        }
    };
    for (int j = 0; j < M; ++j) {
        std::vector<uint64_t> v = avec(read_pauli(j));
        std::vector<uint64_t> prov((size_t)(PW ? PW : 1), 0);
        prov[(size_t)(j >> 6)] |= 1ull << (j & 63);
        reduce(v, prov);
        int piv = -1;
        for (int w = 0; w < W && piv < 0; ++w)
            if (v[(size_t)w]) piv = w * 64 + __builtin_ctzll(v[(size_t)w]);
        if (piv >= 0) { which[(size_t)piv] = (int)basis.size(); basis.push_back({std::move(v), std::move(prov), piv}); }
        // else: dependent column (a deterministic read has v==0) -> not a pivot -> excluded.
    }

    // Declared columns: expP + obs_frame, in declaration order (deferred Observable stream order).
    std::vector<Pauli> expP;
    std::vector<std::vector<int>> declared_frames;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Observable) {
            Pauli P(deferred.n);
            for (const PauliTerm& t : ins.obs) {
                if (t.p == PauliBasis::X) { P.setx(t.qubit); }
                else if (t.p == PauliBasis::Y) { P.setx(t.qubit); P.setz(t.qubit); P.phase += 1; }
                else { P.setz(t.qubit); }
            }
            P.phase &= 3;
            expP.push_back(std::move(P));
            declared_frames.push_back(ins.obs_frame);
        }
    const int R = (int)expP.size();
    res.num_columns = R;
    res.required.resize((size_t)R);
    res.declared.resize((size_t)R);
    res.solvable.assign((size_t)R, 0);
    res.column_ok.assign((size_t)R, 0);

    std::string refusal;
    bool all_ok = true;
    for (int r = 0; r < R; ++r) {
        // Declared set (sorted, dedup; guard out-of-range).
        std::set<int> dset;
        for (int idx : declared_frames[(size_t)r]) {
            if (idx < 0 || idx >= M) {
                res.error = "a PAULI_EXPECTATION declared frame references an out-of-range record";
                return res;
            }
            dset.insert(idx);
        }
        res.declared[(size_t)r].assign(dset.begin(), dset.end());

        // DIAGNOSTICS (`solvable`, `required`): does SOME record set determinize the sign — the
        // GLOBAL span, avec(expP) ∈ span(ALL reads) — and its canonical-minimal solution. These
        // are kept as diagnostics; the diagnose layer refuses a column that is `solvable` (a
        // determinizing frame EXISTS) but whose DECLARED frame does not span (see column_ok),
        // i.e. a missing / mis-declared byproduct frame. `required` names the minimal determining
        // set (informational; a determinizing SUPERSET need not equal it — see column_ok).
        std::vector<uint64_t> tv = avec(expP[(size_t)r]);
        std::vector<uint64_t> tprov((size_t)(PW ? PW : 1), 0);
        reduce(tv, tprov);
        bool solvable = true;
        for (int w = 0; w < W; ++w) if (tv[(size_t)w]) { solvable = false; break; }
        res.solvable[(size_t)r] = solvable ? 1 : 0;
        std::vector<int> rset;
        if (solvable)
            for (int j = 0; j < M; ++j)
                if ((tprov[(size_t)(j >> 6)] >> (j & 63)) & 1) rset.push_back(j);
        res.required[(size_t)r] = rset;   // sorted by construction (ascending j)

        // SPAN VERDICT (the accept test): the declared frame is ACCEPTED iff avec(expP[r]) lies in
        // the SPAN of {avec(R_j) : j ∈ the DECLARED records}. Build a GF(2) echelon basis from the
        // DECLARED reads' avecs only (pivot = lowest set bit, mirroring the global `reduce`), then
        // reduce avec(expP[r]) against it. A ZERO residual ⇒ the declared records resolve every
        // fixed generator that flips P's sign ⇒ accept (a determinizing SUPERSET spans and thus
        // accepts). A NONZERO residual bit names an UNRESOLVED generator: a sign-controlling
        // stabilizer that NO declared read covers, so the sign stays random ⇒ reject (a frame
        // missing a sign-controlling record rejects). Free rows are already excluded from avec.
        std::vector<std::vector<uint64_t>> dbasis;
        std::vector<int> dwhich((size_t)(nf ? nf : 1), -1);   // fixed-row -> index into dbasis (or -1)
        auto dreduce = [&](std::vector<uint64_t>& v) {
            for (int g = 0; g < nf; ++g) {           // low->high: each pivot vec has no bit below its pivot
                if (!((v[(size_t)(g >> 6)] >> (g & 63)) & 1)) continue;
                if (dwhich[(size_t)g] < 0) continue;
                const std::vector<uint64_t>& bv = dbasis[(size_t)dwhich[(size_t)g]];
                for (int w = 0; w < W; ++w) v[(size_t)w] ^= bv[(size_t)w];
            }
        };
        for (int idx : dset) {                       // dset iterates ascending; order is immaterial
            std::vector<uint64_t> v = avec(read_pauli(idx));
            dreduce(v);
            int piv = -1;
            for (int w = 0; w < W && piv < 0; ++w)
                if (v[(size_t)w]) piv = w * 64 + __builtin_ctzll(v[(size_t)w]);
            if (piv >= 0) { dwhich[(size_t)piv] = (int)dbasis.size(); dbasis.push_back(std::move(v)); }
            // else: this declared read has avec == 0 (deterministic / non-sign-controlling) or is a
            // dependent combination — it contributes nothing to the span (harmless in a superset).
        }
        std::vector<uint64_t> resid = avec(expP[(size_t)r]);
        dreduce(resid);
        std::vector<int> unresolved;                 // fixed generators the declared frame cannot resolve
        for (int g = 0; g < nf; ++g)
            if ((resid[(size_t)(g >> 6)] >> (g & 63)) & 1) unresolved.push_back(g);
        bool col_ok = unresolved.empty();
        res.column_ok[(size_t)r] = col_ok ? 1 : 0;
        if (!col_ok) {
            all_ok = false;
            char buf[192];
            std::snprintf(buf, sizeof buf,
                          "  column L%d: declared frame does not determine <P>'s sign "
                          "\xE2\x80\x94 unresolved generator(s)", r);
            refusal += buf;
            for (int g : unresolved) { std::snprintf(buf, sizeof buf, " g#%d", g); refusal += buf; }
            refusal += "\n";
        }
    }
    res.all_ok = all_ok;
    if (!all_ok)
        res.refusal = "declared PAULI_EXPECTATION frame(s) do not determine <P>'s sign "
                      "(unresolved sign-controlling generator):\n" + refusal;
    res.ok = true;
    return res;
}

// dem_column_layout — lightweight structural query: parse, normalize, count columns.
// Does NOT run the full DEM export (no propagation table, no bare state).
// Returns DemColumnLayout with n_obs=O+R and logical_flip entries (decision_k, col). Column indices are
// DEM L-column numbers (0-based, so L0 = first observable in the DEM text). OUTPUT_QUBITS adds no column.
DemColumnLayout dem_column_layout(const std::string& stim_text) {
    DemColumnLayout res;
    ParsedStim p = parse_stim_circuit(stim_text);
    if (!p.ok()) { res.parse_errors = std::move(p.errors); return res; }

    NormalizePolicy pol;
    pol.feedback = NormalizePolicy::Feedback::KeepCoherent;   // export_dem's preset (residual feedback folds)
    pol.defer = true;
    DemNormalized dn = dem_normalize(p.circuit, pol);

    // Check propagation class (same normalization + fallback + rejection path as export_dem).
    Circuit& deferred = dn.deferred;
    PropagationTable& table = dn.table;
    if (!table.all_in_class) {
        res.rejected = true;
        res.reject_gate_index = table.rejects.empty() ? -1 : table.rejects[0].reject_gate_index;
        return res;
    }

    // Count O, R.
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    const int O = max_obs + 1;
    int R_count = 0;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Observable) ++R_count;
    // PAULI_EXPECTATION columns start at L(O); logical-flip at L(O+R).
    const int R = R_count;
    res.n_obs = O + R;

    // Decisions -> logical-flip column indices.
    int max_dec = -1;
    for (auto& kv : p.decisions) if (kv.first > max_dec) max_dec = kv.first;
    const int NDEC = max_dec + 1;
    for (int k = 0; k < NDEC; ++k)
        res.logical_flip.emplace_back(k, O + R + k);

    res.ok = true;
    return res;
}

}  // namespace qeccore
