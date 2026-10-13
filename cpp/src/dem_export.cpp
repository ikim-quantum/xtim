#include "qeccore/diag_conjugate.hpp"
#include <cstddef>
#include "qeccore/dem_export.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/dem_classify.hpp"
#include "qeccore/dem_pe_per_shot.hpp"
#include "qeccore/dem_decompose.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/propagation_table.hpp"

namespace qeccore {

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
    // not the diagonal Delta — handled here; the Delta rules below are the shared ones).
    for (int q = 0; q < P.n; ++q)
        if (Dc.v[q] && P.zbit(q)) phi += 2;
    // Delta = diag(S^a)·∏CZ conjugation via the shared diag_conjugate rules. S-power and CZ are
    // independent Z-bit XORs + phase adds, so splitting the original interleaved loop is
    // byte-exact; each CZ pair is visited once (i<j) where the helper carries the both-endpoint
    // sign that used to be hand-copied here.
    for (int q = 0; q < P.n; ++q) diag_spow_conjugate(P, q, Dc.a[q], Q.z.data(), phi);
    // A CZ pair contributes only when an endpoint carries X (diag_cz_conjugate is a no-op
    // otherwise), so walk the CZ rows of P's X-support instead of all n^2/2 pairs. Each pair is
    // still visited exactly once (as (min, max)); Z-bit XORs and mod-4 phase adds commute, so the
    // result is identical to the all-pairs loop (3.1.10, DEM-export speed).
    const int BW = Dc.B.words();
    for (int i = 0; i < P.n; ++i) {
        if (!P.xbit(i)) continue;
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

// Does Dc act NON-Pauli on any declared expectation column? True iff some expP[r] has an odd
// S-power or any CZ-arm of Dc touching its X-support (the same condition the expectation-column
// gate used to refuse on; see the ±1-expressibility condition in dem_export.hpp). When FALSE,
// Dc's action on every declared
// Pauli is a clean ±1 sign (a Pauli frame update) — today's emit-with-sign-flips path is exact
// and no magnitude can change, so the magnitude classifier need not be consulted.
bool touches_expectation_nonpauli(const DiagPauliClifford& Dc, const std::vector<Pauli>& expP) {
    for (const Pauli& P : expP) {
        for (int q = 0; q < P.n; ++q) {
            if (!P.xbit(q)) continue;
            if (Dc.a[q] & 1) return true;                 // odd S-power rotates the column
            for (int w = 0; w < Dc.B.words(); ++w)
                if (Dc.B.row(q)[w]) return true;          // CZ-arm dresses the column with Z
        }
    }
    return false;
}

// Accepted-branch check for the completeness gate's F-empty (randomize-only) branch: a fault
// whose action on some declared column is not a clean sign, firing no detector deterministically
// but randomizing the detectors in `adets`. Is the fault undetectable on the ACCEPTED branch (every
// detector reads its noiseless value)? Exact, from the Clifford algebra: with A_d = s_d·Dc†D_d Dc
// (D_d the detector's parity Pauli, s_d its noiseless sign; all A_d commute) the accepted-branch
// projector is Π_d (1+A_d)/2, so
//     Pr[acc]          = 2^-k Σ_{S⊆adets} <A_S>,
//     E[col_r ; acc]   = 2^-k Σ_{S⊆adets} <Q_r A_S>,   Q_r = Dc† P'_r Dc,
// each term a Pauli expectation on the noiseless bare state (A_S, Q_r A_S are products of
// commuting Hermitian Paulis). Returns 0 when the accepted branch is unreachable or every column's
// accepted-branch value equals its noiseless value beta0 (magnitude AND sign), 1 when some column
// differs there (an undetectable logical error), 2 when k > 16 (not verified; the caller refuses).
//
// Replaces the 3.1.9 projection (collapse each randomized read's qubit to Z=+1, re-classify):
// that projected the read wires in the Z basis although randomized reads are X/Y reads, and it
// also projected frame reads, which are not post-selection reads and are part of the frame-folded
// column P' (projecting one of them zeroes <P'>, so a fault that is harmless on the accepted branch
// was reported as non-fault-tolerant — example_miniature_oracle's magic-wire DEPOLARIZE1).
[[maybe_unused]] int accepted_branch_violation(const FramedSuperposition& bare, const std::vector<Pauli>& expP,
                              const std::vector<double>& beta0, const DiagPauliClifford& Dc,
                              const std::vector<Pauli>& adets) {
    const int k = (int)adets.size();
    if (k > 16) return 2;
    std::vector<Pauli> A;
    A.reserve((size_t)k);
    for (const Pauli& Dd : adets) {
        auto [pp, pm] = framed_expectation(bare, Dd);
        const double s = pp - pm;
        if (std::abs(std::abs(s) - 1.0) > 1e-9) return 1;   // not a deterministic detector: refuse
        Pauli a = conjugate_through(Dc, Dd);
        if (s < 0) a.phase = (a.phase + 2) & 3;
        A.push_back(std::move(a));
    }
    std::vector<Pauli> Q;
    Q.reserve(expP.size());
    for (const Pauli& P : expP) Q.push_back(conjugate_through(Dc, P));
    const size_t R = expP.size();
    std::vector<double> E(R, 0.0);
    double pr = 0.0;
    Pauli AS(bare.n());                                     // A_∅ = I
    std::vector<Pauli> QAS(Q);                              // Q_r A_∅ = Q_r
    for (uint32_t g = 0;; ++g) {                            // Gray-code walk over subsets S
        {
            auto [pp, pm] = framed_expectation(bare, AS);
            pr += pp - pm;
        }
        for (size_t r = 0; r < R; ++r) {
            auto [pp, pm] = framed_expectation(bare, QAS[r]);
            E[r] += pp - pm;
        }
        if (g + 1 == (1u << k)) break;
        const int i = __builtin_ctz(g + 1);                 // the element toggled between g and g+1
        AS = Pauli::multiply(AS, A[(size_t)i]);
        for (size_t r = 0; r < R; ++r) QAS[r] = Pauli::multiply(QAS[r], A[(size_t)i]);
    }
    const double scale = std::ldexp(1.0, -k);
    pr *= scale;
    if (pr < 1e-12) return 0;                               // accepted branch unreachable
    for (size_t r = 0; r < R; ++r)
        if (std::abs(E[r] * scale / pr - beta0[r]) > 1e-6) return 1;
    return 0;
}

// solve_channel — the exact categorical-channel -> independent-mechanism Walsh-Hadamard solve —
// lives in dem_solve_channel.cpp; declared in qeccore/dem_decompose.hpp.

}  // namespace

namespace {

// ── Diagonal-class fallback normalization (DEM export only) ─────────────────────────────────
// DEM export propagates every fault in the DIAGONAL class (Pauli x diagonal Clifford; no PPR
// retry — a general Clifford residual has no linear-DEM form). Two rewrites of the SAME circuit
// can take a residual out of that class only because of how the deferred stream is written, not
// because of the physics:
//   (a) a live-wire MX/MY deferred as `H q; CX q a; H q` puts a Hadamard ON the data wire q;
//       a T-gate fault's diagonal residual on q (S-power / CZ arm) then meets it mid-stream;
//   (b) a user-written `H q; <single-qubit Pauli noise on q>; M q` on an abandoned wire: the
//       noise blocks eliminate_hadamards' H;M -> MX fold, so the H survives mid-stream.
// The fallback rewrites both exactly — (a) by the diagonal-copy deferral (deferral.hpp), (b) by
// dem_fold_noisy_h_reads below — and is consulted ONLY when the default normalization leaves the
// class, so every circuit the default accepts is byte-identical. Both rewrites are exact
// identities of the circuit's measurement statistics (and of its fault channels), never
// approximations; a circuit still outside the class after them is rejected as before.

// Single-qubit Pauli noise on q conjugated by H (X<->Z, Y->Y up to sign). Returns false when the
// channel cannot be conjugated in place without regrouping (multi-qubit Pauli-specific channels).
bool dem_conjugate_noise_by_h(Instr& nz, int q) {
    switch (nz.channel) {
        case NoiseChannel::DEPOLARIZE1: return true;                 // H-invariant (any list)
        case NoiseChannel::X_ERROR:
        case NoiseChannel::Z_ERROR:
        case NoiseChannel::Y_ERROR:
        case NoiseChannel::PAULI_CHANNEL_1:
            if (nz.qubits.size() != 1 || nz.qubits[0] != q) return false;
            if (nz.channel == NoiseChannel::X_ERROR) nz.channel = NoiseChannel::Z_ERROR;
            else if (nz.channel == NoiseChannel::Z_ERROR) nz.channel = NoiseChannel::X_ERROR;
            else if (nz.channel == NoiseChannel::PAULI_CHANNEL_1) {
                if (nz.probs.size() != 3) return false;
                std::swap(nz.probs[0], nz.probs[2]);                  // (px,py,pz) -> (pz,py,px)
            }
            return true;
        default:
            return false;
    }
}

bool dem_instr_touches(const Instr& x, int q) {
    switch (x.kind) {
        case Instr::Kind::Gate:
            for (int t : x.targets) if (t == q) return true;
            return false;
        case Instr::Kind::Observable:
            for (const PauliTerm& t : x.obs) if (t.qubit == q) return true;
            return false;
        default:
            for (int t : x.qubits) if (t == q) return true;
            return false;
    }
}

// (b): on a deferred, Hadamard-eliminated stream, an `H q` whose every later touch of q is a
// conjugable single-qubit Pauli noise on q, ending in q's TERMINAL Z-basis Measure, is dropped:
// the noise is conjugated by H and the read becomes X. Exact: M_Z . N . H = M_X . (H N H) on the
// record, and nothing else ever touches q (instructions between act on other wires and commute).
// The Measure keeps its stream position (record order unchanged). Every other instruction is
// copied verbatim.
Circuit dem_fold_noisy_h_reads(const Circuit& in) {
    Circuit out = in;
    std::vector<uint8_t> drop(in.stream.size(), 0);
    for (size_t i = 0; i < in.stream.size(); ++i) {
        const Instr& h = in.stream[i];
        if (h.kind != Instr::Kind::Gate || h.gate != GateKind::H || h.targets.size() != 1) continue;
        const int q = h.targets[0];
        std::vector<size_t> noises;
        size_t meas = in.stream.size();
        bool ok = true;
        for (size_t j = i + 1; j < in.stream.size() && ok; ++j) {
            const Instr& x = in.stream[j];
            if (!dem_instr_touches(x, q)) continue;
            if (meas != in.stream.size()) { ok = false; break; }      // a touch after the read
            if (x.kind == Instr::Kind::Noise) { noises.push_back(j); continue; }
            if (x.kind == Instr::Kind::Measure && x.qubits.size() == 1
                && x.basis == PauliBasis::Z) { meas = j; continue; }
            ok = false;
        }
        if (!ok || meas == in.stream.size()) continue;
        std::vector<Instr> conj;
        for (size_t j : noises) {
            Instr nz = out.stream[j];
            if (!dem_conjugate_noise_by_h(nz, q)) { ok = false; break; }
            conj.push_back(std::move(nz));
        }
        if (!ok) continue;
        for (size_t t = 0; t < noises.size(); ++t) out.stream[noises[t]] = std::move(conj[t]);
        out.stream[meas].basis = PauliBasis::X;
        drop[i] = 1;
    }
    Circuit res = out;
    res.stream.clear();
    for (size_t i = 0; i < out.stream.size(); ++i)
        if (!drop[i]) res.stream.push_back(std::move(out.stream[i]));
    return res;
}

// The DEM path's normalization + propagation class, with the diagonal-class fallback. On a
// fallback that still rejects, the DEFAULT normalization's result (and reject index) is kept.
struct DemNormalized {
    NormalizeResult nr;
    Circuit deferred;
    PropagationTable table;
};
DemNormalized dem_normalize(const Circuit& user, const NormalizePolicy& policy_in) {
    DemNormalized d;
    // ChannelSupport consumer: a zero-probability noise instruction builds no atom (its factors
    // never fire), so it is transparent to a pending Clifford frame in eliminate_hadamards (the
    // instruction stays in place). Same frame as the twirl samplers' preset.
    NormalizePolicy policy = policy_in;
    policy.zero_noise_transparent = true;
    d.nr = normalize(user, policy);
    if (d.nr.feedback_rejected) return d;
    d.deferred = std::move(d.nr.normalized);
    d.table = build_propagation_table(d.deferred);   // ppr_retry OFF: CH/general residuals have no linear-DEM form — reject is correct here
    if (d.table.all_in_class) return d;
    NormalizePolicy alt = policy;
    alt.diagonal_copy = true;
    NormalizeResult nr2 = normalize(user, alt);
    if (nr2.feedback_rejected) return d;
    Circuit d2 = dem_fold_noisy_h_reads(nr2.normalized);
    PropagationTable t2 = build_propagation_table(d2);
    if (!t2.all_in_class) return d;
    d.nr = std::move(nr2);
    d.deferred = std::move(d2);
    d.table = std::move(t2);
    return d;
}

}  // namespace

// ── Factored evaluation of the exact correlated-read law (perf/dem-factored-law) ────────────────
// mu[c] = <A_{S_c}> for the Gray-code products A_S of the W-basis Paulis N (frame coordinates).
// In frame coordinates the bare state is |chi>_free (x) |eps>_rest, and the Born kernel
// (framed_born_from_conjugated) reads a frame-coordinate Pauli Q only through (i) Q.x — on the free
// generators for Q in W, (ii) Q.z on the free generators, and (iii) the parity of Q.z against eps on
// the NON-free generators, which enters the kernel as a sign (case A: a factor of s0; case B: the
// Q.z & sj parity, sj = eps ^ sigma_j on the free positions). Hence <Q> == <Q'> BIT FOR BIT for
//     Q' = (Q.phase, Q.x, (Q.z & free) + [parity(Q.z & eps & ~free)] * Z_{a*}),
// a* one fixed non-free generator with eps[a*] = 1: the free-register projection plus one
// representative sign bit. (The sign is NOT folded into the phase: case A reads an odd phase as +1
// while still applying the eps sign, so a phase fold is exact only for Hermitian Q; the sign bit
// is exact for every Q.) Each distinct projection is evaluated ONCE per export, by the same
// framed_expectation_frame call the 3.1.12 walk made on Q itself, so nu is reproduced to the byte:
// <= 8 * 4^{r_free} Born sums instead of 2^wdim per law, and the walk costs O(words) per subset
// (Pauli::multiply's exact phase bookkeeping, in place). A product with a non-free X bit (never one
// of the engine's W bases) is evaluated directly, unprojected.
// The projection is NOT shared with PeShotAnalyzer::frame_parts / the forced-fault factored law on
// purpose: those project ORIGINAL-coordinate Paulis through the Zrow tables, while here the A_S are
// already in frame coordinates (bit extraction only), and the W-basis elimination above fixes the
// nullv order the signature expansion and law_cache key depend on — re-deriving it would risk the
// byte contract for no speed.
void correlated_law_coset_weights(const FramedSuperposition& L, const std::vector<Pauli>& N,
                                  std::vector<double>& nu, FreeRegisterBornMemo& memo) {
    const int n = L.n();
    const int W = (n + 63) / 64;
    const int r = L.k();
    const int wdim = (int)N.size();
    const uint32_t WG = 1u << wdim;
    nu.assign(WG, 0.0);
    std::vector<uint64_t> free_mask((size_t)(W ? W : 1), 0), eps_nonfree((size_t)(W ? W : 1), 0);
    for (int d = 0; d < r; ++d) free_mask[(size_t)(L.free[(size_t)d] >> 6)] |= 1ull << (L.free[(size_t)d] & 63);
    for (int a = 0; a < n; ++a) if (L.eps[(size_t)a]) eps_nonfree[(size_t)(a >> 6)] |= 1ull << (a & 63);
    for (int w = 0; w < W; ++w) eps_nonfree[(size_t)w] &= ~free_mask[(size_t)w];
    int a_star = -1;                                          // representative non-free eps=1 position
    for (int w = 0; w < W && a_star < 0; ++w)
        if (eps_nonfree[(size_t)w]) a_star = w * 64 + __builtin_ctzll(eps_nonfree[(size_t)w]);
    const bool packed = r <= 30;
    const int FW = (r + 63) / 64 > 0 ? (r + 63) / 64 : 1;
    std::vector<uint64_t> xf((size_t)FW), zf((size_t)FW);
    std::string skey;
    Pauli AS(n), Qp(n);
    for (uint32_t g = 0;; ++g) {
        const uint32_t c = g ^ (g >> 1);                      // Gray code: subset index
        int par = 0;
        bool x_nonfree = false;
        for (int w = 0; w < W; ++w) {
            par += __builtin_popcountll(AS.z[(size_t)w] & eps_nonfree[(size_t)w]);
            if (AS.x[(size_t)w] & ~free_mask[(size_t)w]) x_nonfree = true;
        }
        const int sgn = par & 1;                              // the non-free eps sign of AS
        const int phase = AS.phase & 3;
        double val;
        if (x_nonfree) {
            auto [pp, pm] = framed_expectation_frame(L, AS);
            val = pp - pm;
            ++memo.born_sums;
        } else {
            std::fill(xf.begin(), xf.end(), 0ull);
            std::fill(zf.begin(), zf.end(), 0ull);
            for (int d = 0; d < r; ++d) {
                if (AS.xbit(L.free[(size_t)d])) xf[(size_t)(d >> 6)] |= 1ull << (d & 63);
                if (AS.zbit(L.free[(size_t)d])) zf[(size_t)(d >> 6)] |= 1ull << (d & 63);
            }
            double* slot = nullptr;
            bool hit = false;
            if (packed) {
                const uint64_t key = (uint64_t)(phase | (sgn << 2)) | (xf[0] << 3) | (zf[0] << (3 + r));
                auto ins = memo.packed.try_emplace(key, 0.0);
                slot = &ins.first->second; hit = !ins.second;
            } else {
                skey.assign(1, (char)(phase | (sgn << 2)));
                skey.append(reinterpret_cast<const char*>(xf.data()), xf.size() * 8);
                skey.append(reinterpret_cast<const char*>(zf.data()), zf.size() * 8);
                auto ins = memo.wide.try_emplace(skey, 0.0);
                slot = &ins.first->second; hit = !ins.second;
            }
            if (!hit) {
                Qp.phase = phase;
                for (int w = 0; w < W; ++w) {
                    Qp.x[(size_t)w] = AS.x[(size_t)w];
                    Qp.z[(size_t)w] = AS.z[(size_t)w] & free_mask[(size_t)w];
                }
                if (sgn) Qp.setz(a_star);                     // par odd => a_star exists
                auto [pp, pm] = framed_expectation_frame(L, Qp);
                *slot = pp - pm;
                ++memo.born_sums;
            }
            val = *slot;
        }
        nu[c] = val;
        if (g + 1 == WG) break;
        const int i = __builtin_ctz(g + 1);
        const Pauli& B = N[(size_t)i];                        // AS = Pauli::multiply(AS, B), in place
        int sign = 0;
        for (int w = 0; w < W; ++w) sign += __builtin_popcountll(AS.z[(size_t)w] & B.x[(size_t)w]);
        for (int w = 0; w < W; ++w) { AS.x[(size_t)w] ^= B.x[(size_t)w]; AS.z[(size_t)w] ^= B.z[(size_t)w]; }
        AS.phase = ((AS.phase + B.phase + 2 * (sign & 1)) % 4 + 4) % 4;
    }
    for (int h = 1; h < (int)WG; h <<= 1)                      // in-place WHT: nu(y) = sum_c (-1)^{y.c} mu[c]
        for (uint32_t i = 0; i < WG; i += 2 * h)
            for (uint32_t j = i; j < i + (uint32_t)h; ++j) {
                const double x = nu[j], y = nu[j + h];
                nu[j] = x + y; nu[j + h] = x - y;
            }
}

DemExportResult export_dem(const std::string& stim_text,
                           bool include_expectations,
                           bool decompose_errors,
                           bool ignore_decomposition_failures,
                           bool drop_gauge_observables,
                           const std::vector<int>* trusted_detectors,
                           const std::vector<int>* trusted_observables,
                           double approximate_disjoint_errors,
                           double pe_max_work) {
    DemExportResult res;
    ParsedStim p = parse_stim_circuit(stim_text);
    if (!p.ok()) { res.parse_errors = std::move(p.errors); return res; }
    // DEM normalization preset (Reject + defer + eliminate-Hadamards). Coherentize magic-crossing
    // classically-controlled Pauli feedback into coherent gates first, mirroring the sampler
    // (sampler.cpp). Any feedback that crossed a non-Clifford gate is rewritten as an
    // ancilla-controlled coherent entangler; only feedback that could NOT be safely coherentized is
    // left verbatim, and that residue trips the Reject policy below.
    //
    // Feedback (CX/CY/CZ rec[-k] q) is handled by the SAMPLER as an exact record/expectation relabel
    // (Task 2), but DEM export — the error->detector-flip map — would need the relabel folded into
    // every detector's record set (a feedback that flips a record changes which error mechanisms
    // flip a detector). That fold is deferred; DEM export rejects residual feedback cleanly for now
    // rather than emit a model that ignores the feedback's effect on detectors.
    NormalizePolicy dem_policy;
    dem_policy.feedback = NormalizePolicy::Feedback::Reject;
    dem_policy.defer = true;
    dem_policy.want_map = !p.circuit.outputs.empty();   // need final_wire for frame columns
    DemNormalized dn = dem_normalize(p.circuit, dem_policy);
    NormalizeResult& nr = dn.nr;
    if (nr.feedback_rejected) {
        res.parse_errors.push_back({0, "classically-controlled Pauli feedback (CX/CY/CZ rec[-k] q) "
                                       "is not yet handled by DEM export (deferred; the sampler "
                                       "handles feedback)"});
        return res;
    }

    // Default normalization, or the diagonal-class fallback when the default leaves the class
    // (dem_normalize above; byte-identical for every circuit the default accepts).
    Circuit& deferred = dn.deferred;
    PropagationTable& table = dn.table;
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

    // Terminal reads, record order: (pauli idx 0:X 1:Y 2:Z, qubit).
    std::vector<std::pair<int, int>> reads;
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure)
            reads.push_back({ins.basis == PauliBasis::X ? 0
                             : ins.basis == PauliBasis::Y ? 1 : 2, ins.qubits[0]});
    const int M = (int)reads.size();
    {   // deferral places every terminal read on its own wire; the parity-Pauli determinism
        // check below relies on it (non-commuting same-wire reads have no joint Born form)
        std::vector<int> seen;
        for (auto& [b, q] : reads) seen.push_back(q);
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            res.error = "two terminal reads share a wire; the DEM determinism analysis "
                        "does not cover this (refusing)";
            return res;
        }
    }

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
    // Baseline logical-observable expectations <P_r> on the noiseless bare state.
    std::vector<double> beta0(expP.size(), 0.0);
    for (size_t r = 0; r < expP.size(); ++r) {
        auto [pp, pm] = framed_expectation(bare.state, expP[r]);
        beta0[r] = pp - pm;
    }

    // Columns with no DEM concept (noiseless per-shot value varies; filled before the channel walk).
    std::vector<uint8_t> col_dropped(expP.size(), 0);

    // ── Module extensions: OUTPUT_QUBITS frame columns + DECISION logical-flip columns ────
    // Get deferred wire indices for each declared OUTPUT_QUBITS qubit.  nr.map.final_wire[q]
    // gives the deferred-space qubit for original wire q (only valid because want_map was set
    // true above when p.circuit.outputs is non-empty).
    std::vector<int> output_wires;
    for (const OutputPort& op : p.circuit.outputs)
        for (int q : op.qubits) {
            if (q < 0 || q >= (int)nr.map.final_wire.size()) {
                res.error = "OUTPUT_QUBITS qubit out of range in deferred circuit"; return res;
            }
            output_wires.push_back(nr.map.final_wire[q]);
        }
    const int NOUT = (int)output_wires.size();

    // Decisions: indexed by decision number 0..NDEC-1.
    int max_dec_idx = -1;
    for (auto& kv : p.decisions) if (kv.first > max_dec_idx) max_dec_idx = kv.first;
    const int NDEC = max_dec_idx + 1;
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
    std::vector<uint8_t> inc;
    auto check_deterministic = [&](const std::vector<int>& recs, char kind, int id,
                                   std::string& err) -> bool {
        if (!inclusion(recs, inc)) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "%c%d references an out-of-range measurement",
                          kind, id);
            err = buf;
            return false;
        }
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
        if (!any) return true;               // constant parity
        P.phase = ny & 3;
        auto [pp, pm] = framed_expectation(bare.state, P);
        if (std::max(pp, pm) > 1.0 - 1e-9) return true;
        char buf[128];
        std::snprintf(buf, sizeof buf,
                      "non-deterministic %s %c%d (noiseless parity expectation %.6f, not "
                      "+-1); not DEM-expressible",
                      kind == 'D' ? "detector" : "observable", kind, id, pp - pm);
        err = buf;
        return false;
    };
    // Trusted ids (see header): determinism certified externally against the TRUE bare state
    // (from_state channel_report); the text-only gate below is skipped for exactly those ids.
    auto is_trusted = [](const std::vector<int>* ids, int id) {
        return ids && std::find(ids->begin(), ids->end(), id) != ids->end();
    };
    // Detectors must be DEM-expressible (a gauge detector is fatal — no DEM can represent it).
    for (int d = 0; d < D; ++d) {
        if (is_trusted(trusted_detectors, d)) {
            // Still validate the record references (out-of-range is a structural error the
            // trust cannot waive); only the bare-state expectation gate is skipped.
            if (!inclusion(p.detectors[d], inc)) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "D%d references an out-of-range measurement", d);
                res.error = buf; return res;
            }
            continue;
        }
        if (!check_deterministic(p.detectors[d], 'D', d, res.error)) return res;
    }
    // Observables: classify each. A non-deterministic (gauge) observable can't be a DEM column;
    // with drop_gauge_observables we DROP it (record + don't emit), else refuse (Stim parity).
    std::vector<uint8_t> obs_dropped((size_t)(O ? O : 1), 0);
    for (auto& kv : p.observables) {
        if (is_trusted(trusted_observables, kv.first)) {
            if (!inclusion(kv.second, inc)) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "L%d references an out-of-range measurement",
                              kv.first);
                res.error = buf; return res;
            }
            continue;                                                        // trusted -> keep
        }
        std::string oerr;
        if (check_deterministic(kv.second, 'L', kv.first, oerr)) continue;   // deterministic -> keep
        if (!drop_gauge_observables) { res.error = oerr; return res; }
        obs_dropped[kv.first] = 1;
        res.dropped_observables.push_back(kv.first);
    }
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

    // dmask/omask: per-target parity masks over the measurement words (same layout as the
    // sampler's). Targets indexed 0..D-1 (detectors) then D..D+O-1 (observables).
    const int MW = (M + 63) / 64;
    const int T = D + O;

    // ── Module extensions (part 2): dmask + column-layout constants ──────────────────
    // (Part 1 — output_wires, NOUT, dec_recs — is above, before the check_deterministic
    // gate; MW and T needed here were not yet available at that point.)
    std::vector<uint64_t> dmask((size_t)(NDEC > 0 ? NDEC : 1) * (size_t)(MW ? MW : 1), 0);
    for (int k = 0; k < NDEC; ++k)
        for (int idx : dec_recs[(size_t)k])
            if (idx >= 0 && idx < M)
                dmask[(size_t)k * (size_t)MW + (size_t)(idx >> 6)] ^= 1ull << (idx & 63);
    // Target-index layout for module columns (beyond the existing D+O+R space):
    //   FRAME_START  = T + R            — first frame column (frameX of output qubit 0)
    //   LOGFL_START  = T + R + 2*NOUT  — first logical-flip column
    //   T_TOTAL      = LOGFL_START + NDEC
    const int FRAME_START = T + R;
    const int LOGFL_START = FRAME_START + 2 * NOUT;
    const int T_TOTAL     = LOGFL_START + NDEC;

    std::vector<uint64_t> tmask((size_t)T * (MW ? MW : 1), 0);
    auto fill_mask = [&](const std::vector<int>& recs, int t) {
        for (int idx : recs) tmask[(size_t)t * MW + (idx >> 6)] ^= 1ull << (idx & 63);
    };
    if (MW) {
        for (int d = 0; d < D; ++d) fill_mask(p.detectors[d], d);
        for (int o = 0; o < O; ++o)
            if (!obs_dropped[o]) fill_mask(obs[o], D + o);   // dropped -> tmask row stays 0 -> never emitted
    }

    // One alternative's record-flip mask + the per-alternative determinism gate.
    std::vector<uint64_t> mflip((size_t)(MW ? MW : 1), 0);
    // Reads whose propagated effect on a terminal X/Y measurement is a TWIRLED fair coin (an
    // odd S-power rotates the read X<->Y, or a propagated CZ entangles it). By the
    // measurement-twirl theorem (docs/twirl_pauli_condition.tex), such a read's outcome is an
    // independent fair-coin Z (the read flips with probability 1/2): an ordinary Pauli
    // mechanism, not a refusal. We record the read index here (with a CLEAN flip=0 in mflip)
    // and the channel walk below expands over the 2^k fair-coin subsets. (Cleared per
    // alternative.)
    std::vector<int> randomized_reads;
    // Detector/observable part of the signature + the twirled-fair-coin read list. The
    // expectation L-columns are appended by add_alt's router (the Pauli path: clean ±1 sign
    // flips; the magnitude path: la.sign_flip), NOT here — alt_signature is now action-agnostic.
    auto alt_signature = [&](const DiagPauliClifford& Dp, Sig& sig) {
        for (int w = 0; w < MW; ++w) mflip[w] = 0;
        randomized_reads.clear();
        for (int j = 0; j < M; ++j) {
            const auto& [b, q] = reads[j];
            int flip;
            if (b == 2) {
                flip = Dp.v[q] ? 1 : 0;
            } else {
                bool randomized = false;
                if (Dp.a[q] & 1) randomized = true;    // odd S-power rotates the X/Y read X<->Y
                if (!randomized)
                    for (int w = 0; w < Dp.B.words(); ++w)
                        if (Dp.B.row(q)[w]) { randomized = true; break; }   // CZ entangles the read
                if (randomized) {
                    // Twirled fair coin: this terminal read flips independently w.p. 1/2. Keep
                    // its deterministic flip at 0 (mflip untouched) and record it; the channel
                    // walk expands the signature over the 2^k subsets of randomized reads.
                    randomized_reads.push_back(j);
                    continue;
                }
                flip = (Dp.a[q] == 2) ? 1 : 0;
                if (b == 1 && Dp.v[q]) flip ^= 1;
            }
            if (flip) mflip[j >> 6] ^= 1ull << (j & 63);
        }
        sig.clear();
        for (int t = 0; t < T; ++t) {
            int par = 0;
            const uint64_t* tm = tmask.data() + (size_t)t * MW;
            for (int w = 0; w < MW; ++w) par ^= __builtin_popcountll(tm[w] & mflip[w]) & 1;
            if (par) sig.push_back(t);
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
    auto pauli_expectation_flip = [&](const DiagPauliClifford& Dp, int r) -> int {
        const Pauli& P = expP[(size_t)r];
        int flip = 0;
        for (int q = 0; q < deferred.n; ++q) {
            if (Dp.v[q] && P.zbit(q)) flip ^= 1;       // X^v vs P's Z-content
            if (P.xbit(q) && Dp.a[q] == 2) flip ^= 1;  // even-S Z-layer on P's X-support
        }
        return flip;
    };
    auto append_pauli_expectation_flips = [&](const DiagPauliClifford& Dp, Sig& sig) {
        for (int r = 0; r < R; ++r)
            if (!col_dropped[(size_t)r] && pauli_expectation_flip(Dp, r)) sig.push_back(T + r);
    };
    // Per column: does Dc act non-Pauli (not +-1) on P'_r? (odd S-power or a CZ arm on its X-support)
    auto column_nonpauli = [&](const DiagPauliClifford& Dp, int r) -> bool {
        const Pauli& P = expP[(size_t)r];
        for (int q = 0; q < P.n; ++q) {
            if (!P.xbit(q)) continue;
            if (Dp.a[q] & 1) return true;
            for (int w = 0; w < Dp.B.words(); ++w)
                if (Dp.B.row(q)[w]) return true;
        }
        return false;
    };

    // Frame flips for OUTPUT_QUBITS survivors (targets FRAME_START .. FRAME_START+2*NOUT-1).
    //
    // frameX(i) at target FRAME_START+2*i:
    //   Flips iff the propagated error has X-support on wire w = output_wires[i].
    //   Same rule as a Z-measurement at w: flip = Dp.v[w].
    //
    // frameZ(i) at target FRAME_START+2*i+1:
    //   Flips iff the propagated error has Z-support on wire w.  Z-support sources:
    //     * S^2 = Z on w directly: (Dp.a[w] & 2) != 0.
    //     * Y-like (odd S + X on w): (Dp.a[w] & 1) && Dp.v[w] contributes Z.
    //     * CZ arm B.get(w,j) with X on j (Dp.v[j]): Z_w appears via the CZ coupling.
    //   All three are XOR-combined (each source toggles Z-support once; duplicates cancel).
    auto append_frame_flips = [&](const DiagPauliClifford& Dp, Sig& sig) {
        for (int i = 0; i < NOUT; ++i) {
            const int w = output_wires[(size_t)i];
            // frameX: X-support on w.
            if (Dp.v[(size_t)w]) sig.push_back(FRAME_START + 2 * i);
            // frameZ: Z-support on w (XOR of all sources; no duplicates in sig).
            int z_flip = 0;
            if (Dp.a[(size_t)w] & 2) z_flip ^= 1;                          // S^2 = Z
            if ((Dp.a[(size_t)w] & 1) && Dp.v[(size_t)w]) z_flip ^= 1;     // odd-S + X = Y
            for (int j = 0; j < Dp.n; ++j)
                if (j != w && Dp.B.get(w, j) && Dp.v[(size_t)j]) z_flip ^= 1;  // CZ arm
            if (z_flip) sig.push_back(FRAME_START + 2 * i + 1);
        }
    };

    // Logical-flip flips for DECISION columns (targets LOGFL_START .. LOGFL_START+NDEC-1).
    // Decision k flips iff the mechanism's mflip has odd overlap with decision k's record mask.
    // This is computed exactly like the det/obs parity from mflip + tmask above.
    auto append_logical_flip_flips = [&](Sig& sig) {
        for (int k = 0; k < NDEC; ++k) {
            int par = 0;
            const uint64_t* dm = dmask.data() + (size_t)k * (size_t)MW;
            for (int w = 0; w < MW; ++w)
                par ^= __builtin_popcountll(dm[(size_t)w] & mflip[(size_t)w]) & 1;
            if (par) sig.push_back(LOGFL_START + k);
        }
    };

    // Detectors fed by terminal read j (the detector targets whose record set / tmask includes
    // read j). The completeness gate uses this to map a randomized read to the detectors it can
    // fire on the accepted branch. (Observable/expectation targets are excluded — the reject set
    // is a set of DETECTORS to post-select on.)
    std::vector<std::vector<int>> detectors_fed_by_read((size_t)(M ? M : 1));
    for (int d = 0; d < D; ++d) {
        const uint64_t* tm = tmask.data() + (size_t)d * MW;
        for (int j = 0; j < M; ++j)
            if ((tm[j >> 6] >> (j & 63)) & 1) detectors_fed_by_read[(size_t)j].push_back(d);
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

    // ── Per-shot expectation-column law (3.1.13; dem_pe_per_shot.hpp) ─────────────────────────
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
        if (!pe) pe = std::make_unique<PeShotAnalyzer>(bare.state, reads, PeShotAnalyzer::kLeafBudget, pe_max_work);
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
    // Owner rule (3.1.13): a PAULI_EXPECTATION column whose NOISELESS per-shot value varies from
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
    for (int r = 0; r < (int)expP.size(); ++r) {
        const std::string col = "L" + std::to_string(O + r);
        if (col_anticomm[(size_t)r]) { col_cls[(size_t)r] = 1; col_c0[(size_t)r] = 0.0; continue; }
        if (std::abs(beta0[(size_t)r]) > 1.0 - 1e-12) {                 // +-1 stabilizer: constant
            col_cls[(size_t)r] = 1; col_c0[(size_t)r] = beta0[(size_t)r]; continue;
        }
        PeShotAnalyzer::Law L0 = pe_get().structure({}, noiseless_read, expP[(size_t)r]);
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
    auto classify_per_shot = [&](const Instr& ins, int k, const DiagPauliClifford& Dc,
                                 std::vector<int>& pk, std::vector<int>& pbit,
                                 std::vector<PeShotAnalyzer::Law>& plaw, bool& any_mag,
                                 bool& any_rand) -> bool {
        const int RR = (int)expP.size();
        pk.assign((size_t)RR, PK_OP);
        pbit.assign((size_t)RR, 0);
        plaw.assign((size_t)RR, PeShotAnalyzer::Law{});
        any_mag = any_rand = false;
        const std::function<Pauli(int)> faulted_read = [&](int j) {
            return conjugate_through(Dc, noiseless_read(j));
        };
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
            const Pauli Q = conjugate_through(Dc, expP[(size_t)r]);
            PeShotAnalyzer::Law Lf = pe_get().structure(randomized_reads, faulted_read, Q);
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
    // The parities a judged column's per-shot value depends on (its coin, then its y bits); a joint
    // walk that conditions on them reads the column's per-shot value as its conditional expectation.
    auto column_bits = [&](const PeShotAnalyzer::Law& Lw, std::vector<Pauli>& B) {
        if (Lw.zero) return;
        if (Lw.coin) B.push_back(Lw.coinP);
        for (const Pauli& Y : Lw.yP) B.push_back(Y);
    };
    // Per-shot accepted-branch check (replaces the read-averaged accepted_branch_violation): is the
    // fault undetectable on the ACCEPTED branch (every randomized detector reads its noiseless
    // value)? Exact joint law of (the detectors' flip bits, every judged column's coin/y bits); a
    // violation is an accepted pattern of positive probability on which some column's per-shot value
    // differs from its noiseless value (magnitude or sign). 0 ok / 1 violation / 2 > 16 detectors /
    // 3 the joint law exceeds the walk budget (`why`). On 1, bad_col is the violating column's
    // declaration index when one column is identified (-1 otherwise).
    auto per_shot_accepted_violation = [&](const DiagPauliClifford& Dc, const std::vector<Pauli>& adets,
                                           const std::vector<int>& pk, const std::vector<int>& pbit,
                                           const std::vector<PeShotAnalyzer::Law>& plaw,
                                           std::string& why, int& bad_col) -> int {
        bad_col = -1;
        const int kd = (int)adets.size();
        if (kd > 16) return 2;
        std::vector<Pauli> B;
        for (const Pauli& Dd : adets) {
            auto [pp, pm] = framed_expectation(bare.state, Dd);
            const double s = pp - pm;
            if (std::abs(std::abs(s) - 1.0) > 1e-9) return 1;   // not a deterministic detector: refuse
            Pauli a = conjugate_through(Dc, Dd);
            if (s < 0) a.phase = (a.phase + 2) & 3;
            B.push_back(std::move(a));
        }
        bool static_violation = false;   // a column differing on EVERY branch (if accepted is reachable)
        std::vector<Pauli> V;
        std::vector<int> vcol;
        for (size_t r = 0; r < expP.size(); ++r) {
            if (pk[r] == PK_RAND || pk[r] == PK_MAG) {
                column_bits(plaw[r], B);
                V.push_back(plaw[r].Xw);
                vcol.push_back((int)r);
            } else if (pbit[r]) {
                bool sv = false;
                if (pk[r] == PK_BIT) { if (std::abs(col_c0[r]) >= 1e-9) sv = true; }
                else if (!(col_cls[r] == 1 && std::abs(col_c0[r]) < 1e-9))   // PK_OP flip
                    sv = true;
                if (sv && !static_violation) { static_violation = true; bad_col = (int)r; }
            }
        }
        std::vector<PeShotAnalyzer::JointLeaf> pat;
        if (!pe_get().joint_law(B, V, pat, why)) return 3;
        const uint64_t dmask_lo = kd ? ((1ull << kd) - 1ull) : 0ull;
        double pacc = 0.0;
        for (const auto& lf : pat) if (!(lf.f & dmask_lo)) pacc += lf.pr;
        if (pacc < 1e-12) return 0;                          // accepted branch unreachable
        if (static_violation) return 1;
        bad_col = -1;
        for (const auto& lf : pat) {
            if (lf.f & dmask_lo) continue;
            for (size_t i = 0; i < vcol.size(); ++i)
                if (std::abs(lf.v[i] - col_c0[(size_t)vcol[i]]) > 1e-6) { bad_col = vcol[i]; return 1; }
        }
        return 0;
    };
    // Channel walk (same per-qubit / per-pair granularity as the sampler's fast plan).
    std::map<Sig, double> acc;               // cross-channel XOR-composed mechanisms
    // Noiseless value <D_t> of each randomized target, per target id (state-only: computed once).
    std::map<int, double> target_value_cache;
    // Exact correlated-read law per distinct (A_t in frame coordinates): (W basis, coset weights).
    std::unordered_map<std::string, std::pair<std::vector<std::vector<uint64_t>>, std::vector<double>>> law_cache;
    // Born sums of the laws' A_S, memoised by free-register projection (state-only: one per export).
    FreeRegisterBornMemo born_memo;
    auto fold = [&](std::vector<Mech>&& mechs) {
        for (Mech& mech : mechs) {
            double& a = acc[mech.sig];       // default 0
            a = a + mech.p - 2.0 * a * mech.p;
        }
    };
    int li = 0;
    for (int k = 0; k < (int)deferred.stream.size(); ++k) {
        const Instr& ins = deferred.stream[k];
        if (ins.kind != Instr::Kind::Noise) continue;
        const LocationEntry& loc = table.locations[li++];
        // alternative = composition of (qubit-index, factor) atoms in compose_fired's order
        auto alt_D = [&](const std::vector<std::pair<int, PauliBasis>>& factors) {
            DiagPauliClifford Dc = DiagPauliClifford::identity(deferred.n);
            for (auto& [qi, pb] : factors) {
                if (pb == PauliBasis::X || pb == PauliBasis::Y)
                    Dc = Dc.then(loc.x_atom[qi].c_prop);
                if (pb == PauliBasis::Z || pb == PauliBasis::Y)
                    Dc = Dc.then(loc.z_atom[qi].c_prop);
            }
            return Dc;
        };
        std::map<Sig, double> merged;
        // The disjoint probability arguments of this channel instance (stim's threshold test).
        std::vector<double> chan_args;
        Sig sig;
        // A Pauli-correctable alternative with a per-shot RANDOM column sign (PK_RAND, R > 0): the
        // exact joint law of (the targets its randomized reads feed, every PK_RAND column's coin / y
        // bits) — the 3.1.10 correlated-read law extended by the column bits — expanded into
        // signatures, each PK_RAND column flipped on exactly the patterns where its per-shot value
        // is -c0. (No PK_MAG column here: classify_per_shot found every |value| unchanged.)
        auto add_random_sign_alt = [&](const Instr& ins, int k, double prob, const DiagPauliClifford& Dc,
                                       const std::vector<int>& pk, const std::vector<int>& pbit,
                                       const std::vector<PeShotAnalyzer::Law>& plaw) -> bool {
            if (NOUT > 0) append_frame_flips(Dc, sig);
            if (NDEC > 0) append_logical_flip_flips(sig);
            std::vector<uint64_t> rmask((size_t)(MW ? MW : 1), 0);
            for (int j : randomized_reads) rmask[(size_t)(j >> 6)] |= 1ull << (j & 63);
            auto row_hits = [&](const uint64_t* row) {
                for (int w = 0; w < MW; ++w) if (row[w] & rmask[(size_t)w]) return true;
                return false;
            };
            std::vector<int> rt;
            std::vector<const uint64_t*> rt_rows;
            for (int t = 0; t < T; ++t)
                if (row_hits(tmask.data() + (size_t)t * MW)) { rt.push_back(t); rt_rows.push_back(tmask.data() + (size_t)t * MW); }
            for (int k2 = 0; k2 < NDEC; ++k2)
                if (row_hits(dmask.data() + (size_t)k2 * MW)) { rt.push_back(LOGFL_START + k2); rt_rows.push_back(dmask.data() + (size_t)k2 * MW); }
            const int m = (int)rt.size();
            std::vector<Pauli> B;
            for (int i = 0; i < m; ++i) {
                Pauli Dt(deferred.n);
                int ny = 0;
                for (int j = 0; j < M; ++j) {
                    if (!((rt_rows[(size_t)i][j >> 6] >> (j & 63)) & 1)) continue;
                    const auto& [b, q] = reads[(size_t)j];
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
                        auto [pp, pm] = framed_expectation(bare.state, Dt);
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
                Pauli Ai = conjugate_through(Dc, Dt);
                if (s0 < 0) Ai.phase = (Ai.phase + 2) & 3;
                B.push_back(std::move(Ai));
            }
            std::vector<Pauli> V;
            std::vector<int> vidx((size_t)R, -1);
            for (int r = 0; r < R; ++r)
                if (pk[(size_t)r] == PK_RAND) {
                    column_bits(plaw[(size_t)r], B);
                    vidx[(size_t)r] = (int)V.size();
                    V.push_back(plaw[(size_t)r].Xw);
                }
            std::vector<PeShotAnalyzer::JointLeaf> pat;
            std::string why;
            if (!pe_get().joint_law(B, V, pat, why)) {
                res.error = loc_name(ins, k) + ": " + why + " (refusing)";
                return false;
            }
            Sig base;                                           // deterministic part minus the randomized targets
            std::set_difference(sig.begin(), sig.end(), rt.begin(), rt.end(), std::back_inserter(base));
            for (const auto& lf : pat) {
                Sig flipped, ecols, sig1, sig_f;
                for (int i = 0; i < m; ++i) if ((lf.f >> i) & 1) flipped.push_back(rt[(size_t)i]);
                for (int r = 0; r < R; ++r) {
                    int fl = pbit[(size_t)r];
                    if (pk[(size_t)r] == PK_RAND) {
                        const double v = lf.v[(size_t)vidx[(size_t)r]];
                        fl = (v < 0) != (col_c0[(size_t)r] < 0) ? 1 : 0;
                    }
                    if (fl) ecols.push_back(T + r);
                }
                std::set_union(base.begin(), base.end(), flipped.begin(), flipped.end(),
                               std::back_inserter(sig1));
                std::set_union(sig1.begin(), sig1.end(), ecols.begin(), ecols.end(),
                               std::back_inserter(sig_f));
                if (!sig_f.empty()) merged[sig_f] += prob * lf.pr;
            }
            return true;
        };
        // The per-shot routing of one alternative that acts non-Pauli on a column or randomizes a
        // read (kept out of line: the Pauli-only hot path above does not pay for it).
        auto per_shot_route = [&](const Instr& ins, int k, double prob,
                                  const DiagPauliClifford& Dc) __attribute__((noinline)) -> int {
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
                        std::vector<Pauli> adets;
                        for (int d : aff) adets.push_back(detector_pauli(d));
                        std::string why;
                        int bad_col = -1;
                        const int v = per_shot_accepted_violation(Dc, adets, pk, pbit, plaw, why, bad_col);
                        if (v == 3)
                            report_gate(-1, loc_name(ins, k) + ": a single fault acting non-Pauli on %COL% "
                                            "randomizes " + std::to_string(adets.size()) +
                                            " detector(s), and " + why + "; the accepted branch was not "
                                            "verified");
                        else if (v == 2)
                            report_gate(-1, loc_name(ins, k) + ": a single fault randomizes " +
                                            std::to_string(adets.size()) + " > 16 detectors while acting "
                                            "non-Pauli on %COL%; the accepted branch was not verified");
                        else if (v == 1)
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
                    std::vector<Pauli> adets;
                    for (int d : aff) adets.push_back(detector_pauli(d));
                    std::string why;
                    int bad_col = -1;
                    const int v = per_shot_accepted_violation(Dc, adets, pk, pbit, plaw, why, bad_col);
                    if (v == 3) {
                        res.error = loc_name(ins, k) + ": a single fault acting non-Pauli on a "
                                    "logical expectation randomizes " + std::to_string(adets.size()) +
                                    " detector(s), and " + why + "; the accepted branch was not "
                                    "verified (refusing)";
                        return 2;
                    }
                    if (v == 2) {
                        res.error = loc_name(ins, k) + (": a single fault randomizes " +
                                    std::to_string(adets.size()) + " > 16 detectors while "
                                    "acting non-Pauli on a logical expectation; the accepted "
                                    "branch was not verified (refusing)");
                        return 2;
                    }
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
            DiagPauliClifford Dc = alt_D(factors);
            alt_signature(Dc, sig);                        // det/obs sig + randomized_reads
            // ── Per-shot routing (3.1.13). The det/obs signature is fixed; decide what the fault does
            // to each declared expectation column PER SHOT — the sampler's semantics: <P'> on the
            // state conditioned on EVERY measurement outcome (declared frame records only fix the
            // sign convention). An alternative that acts as a Pauli on every column and randomizes no
            // read multiplies each per-shot value by its operator sign with the records relabelled
            // (exact; the pre-3.1.13 path, untouched). Every other alternative is judged column by
            // column from the column's exact per-shot law (classify_per_shot, dem_pe_per_shot.hpp),
            // never from the read-averaged <D^dag P' D> on the bare state (which disagreed with the
            // sampler whenever a read that is neither a detector nor a frame record is correlated
            // with the column: silent wrong exports, and false refusals / post-select flags).
            if (expP.empty() ||
                (!touches_expectation_nonpauli(Dc, expP) && randomized_reads.empty())) {
                // Pauli logical action: every declared Pauli sees a clean ±1 sign — a Pauli frame
                // update. Emit normally, including the expectation-column sign flips (R>0).
                append_pauli_expectation_flips(Dc, sig);
            } else {
                // returns 0 continue / 1 return true / 2 return false (refused)
                const int pr = per_shot_route(ins, k, prob, Dc);
                if (pr == 2) return false;
                if (pr == 1) return true;
            }
            // Frame columns: determined by Dc alone (independent of randomized reads).
            if (NOUT > 0) append_frame_flips(Dc, sig);
            // Logical-flip base: parity of deterministic mflip against decision record masks.
            if (NDEC > 0) append_logical_flip_flips(sig);
            const int rk = (int)randomized_reads.size();
            if (rk == 0) {                                // Pauli-only path: byte-identical to before
                if (!sig.empty()) merged[sig] += prob;   // empty signature folds into identity
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
            std::vector<uint64_t> rmask((size_t)(MW ? MW : 1), 0);
            for (int j : randomized_reads) rmask[(size_t)(j >> 6)] |= 1ull << (j & 63);
            auto row_hits = [&](const uint64_t* row) {
                for (int w = 0; w < MW; ++w) if (row[w] & rmask[(size_t)w]) return true;
                return false;
            };
            std::vector<int> rt;                         // sig ids of the randomized targets
            std::vector<const uint64_t*> rt_rows;        // their record masks
            for (int t = 0; t < T; ++t)
                if (row_hits(tmask.data() + (size_t)t * MW)) { rt.push_back(t); rt_rows.push_back(tmask.data() + (size_t)t * MW); }
            for (int k2 = 0; k2 < NDEC; ++k2)
                if (row_hits(dmask.data() + (size_t)k2 * MW)) { rt.push_back(LOGFL_START + k2); rt_rows.push_back(dmask.data() + (size_t)k2 * MW); }
            const int m = (int)rt.size();
            std::vector<Pauli> A;
            A.reserve((size_t)m);
            for (int i = 0; i < m; ++i) {
                Pauli Dt(deferred.n);
                int ny = 0;
                for (int j = 0; j < M; ++j) {
                    if (!((rt_rows[(size_t)i][j >> 6] >> (j & 63)) & 1)) continue;
                    const auto& [b, q] = reads[(size_t)j];
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
                        auto [pp, pm] = framed_expectation(bare.state, Dt);
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
                Pauli Ai = conjugate_through(Dc, Dt);
                if (s0 < 0) Ai.phase = (Ai.phase + 2) & 3;
                A.push_back(std::move(Ai));
            }
            // W: subsets S with a possibly nonzero <A_S> — the frame X-part of A_S must lie on the
            // free generators. Gaussian elimination on the non-free X-parts, tracking combinations.
            const auto& U = bare.state.U;
            std::vector<uint8_t> is_free_gen((size_t)deferred.n, 0);
            for (int f : bare.state.free) is_free_gen[(size_t)f] = 1;
            const int NW = (deferred.n + 63) / 64, CW = (m + 63) / 64;
            struct Row { std::vector<uint64_t> x, comb; };
            std::vector<Pauli> Qf;                       // the A_t in frame coordinates (U^dag A_t U)
            Qf.reserve((size_t)m);
            for (int i = 0; i < m; ++i)
                Qf.push_back(U.dual_valid() ? U.conjugate(A[(size_t)i]) : U.dual_image(A[(size_t)i]));
            // The Fourier support W and the coset weights depend only on the A_t (in frame
            // coordinates): memoised per export by their exact content (many alternatives share
            // their randomized residual). A hit returns the identical computation's result.
            std::string law_key;
            for (const Pauli& Q : Qf) {
                law_key.push_back((char)Q.phase);
                law_key.append(reinterpret_cast<const char*>(Q.x.data()), Q.x.size() * 8);
                law_key.append(reinterpret_cast<const char*>(Q.z.data()), Q.z.size() * 8);
                law_key.push_back('|');
            }
            auto law_it = law_cache.find(law_key);
            if (law_it == law_cache.end()) {
            std::vector<Row> rows((size_t)m);
            for (int i = 0; i < m; ++i) {
                const Pauli& Q = Qf[(size_t)i];
                rows[(size_t)i].x.assign((size_t)NW, 0);
                for (int a2 = 0; a2 < deferred.n; ++a2)
                    if (!is_free_gen[(size_t)a2] && Q.xbit(a2)) rows[(size_t)i].x[(size_t)(a2 >> 6)] |= 1ull << (a2 & 63);
                rows[(size_t)i].comb.assign((size_t)(CW ? CW : 1), 0);
                rows[(size_t)i].comb[(size_t)(i >> 6)] |= 1ull << (i & 63);
            }
            std::vector<std::vector<uint64_t>> nullv;    // basis of W (as subsets of the targets)
            {
                std::vector<int> piv_of_row;
                std::vector<Row> red;
                for (int i = 0; i < m; ++i) {
                    Row r = rows[(size_t)i];
                    for (size_t k3 = 0; k3 < red.size(); ++k3) {
                        const int pv = piv_of_row[k3];
                        if ((r.x[(size_t)(pv >> 6)] >> (pv & 63)) & 1) {
                            for (int w = 0; w < NW; ++w) r.x[(size_t)w] ^= red[k3].x[(size_t)w];
                            for (int w = 0; w < CW; ++w) r.comb[(size_t)w] ^= red[k3].comb[(size_t)w];
                        }
                    }
                    int pv = -1;
                    for (int w = 0; w < NW && pv < 0; ++w)
                        if (r.x[(size_t)w]) pv = w * 64 + __builtin_ctzll(r.x[(size_t)w]);
                    if (pv < 0) nullv.push_back(r.comb);
                    else { piv_of_row.push_back(pv); red.push_back(std::move(r)); }
                }
            }
            const int wdim = (int)nullv.size();
            if (wdim > 20 || m - wdim > 16) {
                res.error = loc_name(ins, k) + (": a single fault randomizes " + std::to_string(m) +
                            " targets with a support of dimension " + std::to_string(m - wdim) +
                            " (> 16) or a Fourier support of dimension " + std::to_string(wdim) +
                            " (> 20); the exact joint law is too large to expand (refusing)");
                return false;
            }
            // mu[c] = <A_{S_c}>, S_c = XOR of the nullspace basis vectors selected by c (Gray code).
            // Products are formed in FRAME coordinates (U^dag (A B) U = (U^dag A U)(U^dag B U)
            // exactly, phase included), so each <A_S> is one Born sum with no conjugation.
            std::vector<Pauli> N((size_t)wdim, Pauli(deferred.n));
            for (int c2 = 0; c2 < wdim; ++c2)
                for (int i = 0; i < m; ++i)
                    if ((nullv[(size_t)c2][(size_t)(i >> 6)] >> (i & 63)) & 1)
                        N[(size_t)c2] = Pauli::multiply(N[(size_t)c2], Qf[(size_t)i]);
            // nu = WHT(mu): one Born sum per DISTINCT free-register projection of the A_S (memoised
            // across every law of this export), bit-identical to one Born sum per subset.
            std::vector<double> nu;
            correlated_law_coset_weights(bare.state, N, nu, born_memo);
            law_it = law_cache.emplace(law_key, std::make_pair(std::move(nullv), std::move(nu))).first;
            }
            const std::vector<std::vector<uint64_t>>& nullv = law_it->second.first;
            const std::vector<double>& nu = law_it->second.second;
            const int wdim = (int)nullv.size();
            const uint32_t WG = 1u << wdim;
            // f with f.nullv[c] = y_c: row-reduce the nullspace basis (as an m-bit system).
            std::vector<std::vector<uint64_t>> nb = nullv;      // reduced rows
            std::vector<uint32_t> nb_y(nb.size());              // which y-combination each row is
            std::vector<int> pivc;
            for (size_t r = 0; r < nb.size(); ++r) nb_y[r] = 1u << r;
            for (size_t r = 0; r < nb.size(); ++r) {
                int pv = -1;
                for (int w = 0; w < CW && pv < 0; ++w)
                    if (nb[r][(size_t)w]) pv = w * 64 + __builtin_ctzll(nb[r][(size_t)w]);
                pivc.push_back(pv);                             // pv >= 0: basis vectors independent
                for (size_t r2 = 0; r2 < nb.size(); ++r2)
                    if (r2 != r && ((nb[r2][(size_t)(pv >> 6)] >> (pv & 63)) & 1)) {
                        for (int w = 0; w < CW; ++w) nb[r2][(size_t)w] ^= nb[r][(size_t)w];
                        nb_y[r2] ^= nb_y[r];
                    }
            }
            std::vector<uint8_t> is_piv((size_t)m, 0);
            for (int pv : pivc) is_piv[(size_t)pv] = 1;
            std::vector<int> freec;                             // free columns: basis of W^perp
            for (int i = 0; i < m; ++i) if (!is_piv[(size_t)i]) freec.push_back(i);
            const int vdim = (int)freec.size();                 // = m - wdim
            Sig base;                                           // deterministic part minus T_R
            std::set_difference(sig.begin(), sig.end(), rt.begin(), rt.end(), std::back_inserter(base));
            const double inv = 1.0 / (double)WG / (double)(1u << vdim);
            std::vector<uint8_t> f((size_t)m, 0);
            for (uint32_t y = 0; y < WG; ++y) {
                const double wy = nu[y] * inv;                  // probability of each pattern in this coset
                if (!(wy > 1e-15)) continue;
                for (uint32_t z = 0; z < (1u << vdim); ++z) {
                    std::fill(f.begin(), f.end(), (uint8_t)0);
                    for (int i = 0; i < vdim; ++i) f[(size_t)freec[(size_t)i]] = (uint8_t)((z >> i) & 1);
                    for (size_t r = 0; r < nb.size(); ++r) {    // pivot value: y-parity minus free part
                        int v = __builtin_popcount(nb_y[r] & y) & 1;
                        for (int i = 0; i < vdim; ++i)
                            if (f[(size_t)freec[(size_t)i]] &&
                                ((nb[r][(size_t)(freec[(size_t)i] >> 6)] >> (freec[(size_t)i] & 63)) & 1)) v ^= 1;
                        f[(size_t)pivc[r]] = (uint8_t)v;
                    }
                    Sig flipped, sig_f;
                    for (int i = 0; i < m; ++i) if (f[(size_t)i]) flipped.push_back(rt[(size_t)i]);
                    std::set_union(base.begin(), base.end(), flipped.begin(), flipped.end(),
                                   std::back_inserter(sig_f));
                    if (!sig_f.empty()) merged[sig_f] += prob * wy;
                }
            }
            return true;
        };
        auto flush = [&]() -> bool {
            std::vector<Mech> mechs;
            std::string err;
            bool needs_approx = false;
            if (!solve_channel(merged, mechs, err, &needs_approx)) {
                // Stim's rule: no exact conversion -> refuse unless approximate_disjoint_errors;
                // then every argument must be <= the threshold, and each merged signature becomes
                // an independent mechanism with its (summed) disjoint probability.
                if (!(approximate_disjoint_errors > 0.0)) {
                    res.error = needs_approx ? approximate_disjoint_refusal(ins, k, err)
                                             : loc_name(ins, k) + (": " + err);
                    chan_args.clear();
                    return false;
                }
                for (double a : chan_args)
                    if (a > approximate_disjoint_errors) {
                        char buf[192];
                        std::snprintf(buf, sizeof buf, ": the channel has a probability argument "
                                      "(%.17g) larger than the `approximate_disjoint_errors` "
                                      "threshold (%.17g)", a, approximate_disjoint_errors);
                        res.error = loc_name(ins, k) + buf;
                        chan_args.clear();
                        return false;
                    }
                mechs.clear();
                for (const auto& [sg, pr] : merged)
                    if (pr > 0.0 && !sg.empty()) mechs.push_back({sg, pr});
            }
            chan_args.clear();
            fold(std::move(mechs));
            merged.clear();
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
    // propagate, no magnitude question (never a post-select flag), and no OUTPUT_QUBITS frame
    // column. Its signature is every target whose record set reads j, by parity:
    //   * detectors / observables     — tmask (dropped gauge observables have an all-zero row);
    //   * PAULI_EXPECTATION columns   — the declared byproduct frame (obs_frame): the sampler
    //                                   applies rec_flip BEFORE the emask fold, so a flipped
    //                                   frame record flips the reported sign of that column;
    //   * DECISION logical-flip cols  — dmask.
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
                Sig sig;
                for (int t = 0; t < T; ++t)
                    if (read_bit(tmask, (size_t)t * MW, j)) sig.push_back(t);
                for (int r = 0; r < R; ++r)
                    if (!col_dropped[(size_t)r] && read_bit(emask[(size_t)r], 0, j)) sig.push_back(T + r);
                for (int k2 = 0; k2 < NDEC; ++k2)
                    if (read_bit(dmask, (size_t)k2 * (size_t)MW, j)) sig.push_back(LOGFL_START + k2);
                if (!sig.empty()) {
                    std::vector<Mech> one;
                    one.push_back({std::move(sig), pflip});
                    fold(std::move(one));
                }
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

    // Emit. acc is sorted by signature (lexicographic target lists == Stim's order).
    std::string out;
    std::vector<uint8_t> covered((size_t)(T_TOTAL > 0 ? T_TOTAL : 1), 0);
    char num[64];
    std::vector<DecomposedMech> decomposed;
    if (decompose_errors) {
        std::string derr;
        if (!decompose_acc(acc, D, ignore_decomposition_failures, decomposed, derr)) {
            res.error = derr;
            return res;
        }
    } else {
        for (const auto& kv : acc) decomposed.push_back({{kv.first}, kv.second});  // pass-through
    }
    for (const auto& m : decomposed) {
        if (m.p <= 0.0) continue;
        std::snprintf(num, sizeof num, "%.17g", m.p);
        out += "error(";
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
    // Frame columns (OUTPUT_QUBITS): two L# per output qubit (X then Z).
    for (int i = 0; i < NOUT; ++i) {
        if (!covered[(size_t)(FRAME_START + 2 * i)])
            { std::snprintf(num, sizeof num, "logical_observable L%d\n", O + R + 2 * i); out += num; }
        if (!covered[(size_t)(FRAME_START + 2 * i + 1)])
            { std::snprintf(num, sizeof num, "logical_observable L%d\n", O + R + 2 * i + 1); out += num; }
    }
    // Logical-flip columns (DECISION): one L# per decision index k.
    for (int k = 0; k < NDEC; ++k) {
        if (!covered[(size_t)(LOGFL_START + k)])
            { std::snprintf(num, sizeof num, "logical_observable L%d\n", O + R + 2 * NOUT + k); out += num; }
    }
    res.dem = std::move(out);
    res.ok = true;
    return res;
}

// Exact per-detector / per-observable determinism partition — the standalone twin of
// export_dem's circuit-level determinism gate (same setup, same parity-Pauli Born check),
// but WITHOUT the noise-channel DEM machinery, so it answers even when the full DEM refuses.
// The shared algebra is guarded against drift by test_dem_determinism_matches_export.
DeterminismResult detector_determinism(const std::string& stim_text) {
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
    // The sampler applies its forward triangular pass to reference VALUES (refmw); folding into a
    // channel's record MASK is the TRANSPOSE of that map, so the ops are consumed in REVERSE order
    // (op_k first). Each op contributes `control_record ^= parity(mask ∧ mflips)`. Reverse order is
    // load-bearing when feedback chains (op_j's control record is itself flipped by an earlier op).
    auto relabel = [&](const std::vector<int>& recs) -> std::vector<int> {
        if (!fb.has_feedback) return recs;
        std::vector<uint64_t> mask((size_t)fb.MW, 0);
        for (int idx : recs)
            if (idx >= 0 && idx < M) mask[(size_t)(idx >> 6)] ^= 1ull << (idx & 63);
        for (auto it = fb.ops.rbegin(); it != fb.ops.rend(); ++it) {
            const FeedbackOp& op = *it;
            int overlap = 0;
            for (int w = 0; w < fb.MW; ++w)
                overlap += __builtin_popcountll(mask[(size_t)w] & op.mflips[(size_t)w]);
            if (overlap & 1) {
                const int c = op.control_record;
                if (c >= 0 && c < M) mask[(size_t)(c >> 6)] ^= 1ull << (c & 63);
            }
        }
        std::vector<int> out;
        for (int j = 0; j < M; ++j)
            if ((mask[(size_t)(j >> 6)] >> (j & 63)) & 1) out.push_back(j);
        return out;
    };

    // Parity Pauli of a record set on the bare state -> signed <P> = pp - pm; sets `deterministic`
    // (|<P>| == 1) and `oor` (an out-of-range record index). Byte-for-byte the same construction
    // as export_dem's check_deterministic lambda.
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
        int ny = 0, any = 0;
        for (int j = 0; j < M; ++j) {
            if (!inc[j]) continue;
            any = 1;
            const auto& [b, q] = reads[j];
            if (b == 0) { P.setx(q); }
            else if (b == 1) { P.setx(q); P.setz(q); ++ny; }
            else { P.setz(q); }
        }
        if (!any) { deterministic = true; return 1.0; }   // empty parity is the constant +1
        P.phase = ny & 3;
        auto [pp, pm] = framed_expectation(bare.state, P);
        deterministic = (std::max(pp, pm) > 1.0 - 1e-9);
        return pp - pm;
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

    // avec(P): bit r set iff P anticommutes with the r-th fixed generator.
    auto avec = [&](const Pauli& P) {
        std::vector<uint64_t> v((size_t)(W ? W : 1), 0);
        for (int r = 0; r < nf; ++r)
            if (Pauli::anticommute_bit(P, st.U.Zrow[(size_t)fixed_rows[r]]))
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
// Returns DemColumnLayout with n_obs=O+R, frame entries (orig_qubit, colX, colZ), and
// logical_flip entries (decision_k, col).  Column indices are DEM L-column numbers
// (0-based, so L0 = first observable in the DEM text).
DemColumnLayout dem_column_layout(const std::string& stim_text) {
    DemColumnLayout res;
    ParsedStim p = parse_stim_circuit(stim_text);
    if (!p.ok()) { res.parse_errors = std::move(p.errors); return res; }

    NormalizePolicy pol;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.defer = true;
    pol.want_map = !p.circuit.outputs.empty();
    DemNormalized dn = dem_normalize(p.circuit, pol);
    NormalizeResult& nr = dn.nr;
    if (nr.feedback_rejected) {
        res.error = "classically-controlled Pauli feedback not handled by DEM export";
        return res;
    }

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
    // PAULI_EXPECTATION columns start at L(O); frame at L(O+R); logical-flip at L(O+R+2*NOUT).
    const int R = R_count;
    res.n_obs = O + R;

    // Output qubits -> deferred wire -> frame column indices.
    int col_x = O + R;   // first frame X column (L-index)
    for (const OutputPort& op : p.circuit.outputs) {
        for (int q : op.qubits) {
            if (q < 0 || q >= (int)nr.map.final_wire.size()) {
                res.error = "OUTPUT_QUBITS qubit out of range in deferred circuit";
                return res;
            }
            int colX = col_x;
            int colZ = col_x + 1;
            res.frame.emplace_back(q, colX, colZ);
            col_x += 2;
        }
    }
    const int NOUT = (int)res.frame.size();

    // Decisions -> logical-flip column indices.
    int max_dec = -1;
    for (auto& kv : p.decisions) if (kv.first > max_dec) max_dec = kv.first;
    const int NDEC = max_dec + 1;
    for (int k = 0; k < NDEC; ++k)
        res.logical_flip.emplace_back(k, O + R + 2 * NOUT + k);

    res.ok = true;
    return res;
}

}  // namespace qeccore
