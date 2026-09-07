// FramedSuperposition (implementation): the measurement/conjugation kernels first, the framed
// method surface (expectation / apply_clifford / measure_pauli*) after — one TU.
//
// Production extraction of the validated spike (was cpp/tests/test_lean_anticommuting.cpp). See
// qeccore/framed_superposition.hpp for the API contract and docs/superpowers/specs/2026-06-27-lean-tableau-
// sampling-rep-design.md §3.4 (+§2 primitive) for the algorithm. The code below is moved verbatim
// from the spike (which was validated exact vs statevector chi=1..4 + statistical on real d5).

#include "qeccore/framed_superposition.hpp"
#include <cstddef>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/pauli_kernels.hpp"   // shared single_pauli/pmul_into/conj_cx_inplace/dual_image_rows_scan/partner_hash_chi_min

namespace qeccore {

// pmul_into / conj_cx_inplace / reset_pauli_inplace / dual_image_rows_scan / single_pauli now live in
// qeccore/pauli_kernels.hpp (verified byte-identical to the former file-local copies; shared with the
// CanonicalStabSum oracle). mk_single's callers use single_pauli directly.

// ───────────────────────────── batched diagonal-Clifford conjugation ─────────────────────────
// One-pass conjugation of every forward row M = i^ph X^x Z^z by the diagonal Clifford
//   D = X^v · (∏ CZ_{c,t}) · (∏_q S^{a_q})           (applied to the state in that left-to-right
// order; conjugation M -> D M D† is gate-by-gate D_k…D_1 M D_1†…D_k†, i.e. the gates are applied
// in apply order S-powers, then CZ, then X^v — matching active_block's apply sequence).
//
// A diagonal Clifford fixes the X-content of M (S/Z/CZ/X all leave X bits invariant under
// conjugation), so:
//   * the S-power on q contributes (per gate-by-gate equivalence, derived in clifford_tableau.cpp):
//       if x_q:  z_q ^= (a_q & 1)   and   phase += a_q          [a_q=1:S a_q=2:Z a_q=3:S† all match]
//   * each CZ_{c,t} contributes (reads the FIXED x bits):
//       if x_c:  z_t ^= 1 ;  if x_t:  z_c ^= 1 ;  if (x_c & x_t):  phase += 2
//   * X^v on q contributes (reads the FINAL z bit, X conjugation: X Z X = -Z):
//       if z_q (after all the above):  phase += 2
// S-powers and CZ both only READ x (fixed) and WRITE z/phase, so they accumulate independently and
// the S-then-CZ order is irrelevant to the result; X^v reads the final z, so it is folded last.
// Each step's per-row arithmetic is the closed form the matching left_* gate applies to one row,
// so the accumulated result is bit-identical to the sequential left_* replay.
void conjugate_by_diag_clifford(CliffordTableau& U, const std::vector<uint8_t>& a,
                                const std::vector<std::pair<int, int>>& cz,
                                const std::vector<uint8_t>& v) {
    const int n = U.n;
    if (n == 0) return;
    // Pack the X^v mask once (read against each row's FINAL z; X^v word-parity = phase delta).
    const int W = (n + 63) / 64;
    static thread_local std::vector<uint64_t> vmask; vmask.assign(W, 0);
    bool any_v = false;
    if (!v.empty())
        for (int q = 0; q < n; ++q) if (v[q]) { vmask[q >> 6] |= 1ull << (q & 63); any_v = true; }

    // Collect the NONZERO S-power columns once (typically sparse). Per row we iterate only these and
    // test the (fixed) x bit — instead of scanning all n columns for EVERY row (the dominant cost).
    static thread_local std::vector<std::pair<int, uint8_t>> acols; acols.clear();
    if (!a.empty())
        for (int q = 0; q < n; ++q) { const uint8_t aq = a[q] & 3; if (aq) acols.push_back({q, aq}); }

    // ── CZ network as a SYMMETRIC adjacency over GF(2), applied WORD-PARALLEL per row. The frame
    //    rows are extremely x-SPARSE (~3 set x-bits each on d5) while |cz| is large (~108), so the
    //    old per-row scan of ALL cz pairs (108 iters/row) was ~36× wasteful. Instead build per-column
    //    bit-packed neighbour masks adj[c] = {t : edge(c,t)} once, then per row iterate ONLY its set
    //    x-bits c and XOR adj[c] into z (z ^= A·x, the exact CZ z-delta). XOR-accumulate adj so a
    //    repeated pair toggles (parity), matching the old loop's per-edge toggle.
    //    Phase: each edge with BOTH endpoints x-set adds 2. Σ_{c set} popcount(adj[c] & xmask) counts
    //    every fully-set edge TWICE (once per endpoint) → it is even and (Σ>>1)&1 is the edge parity.
    //    This is accumulated in the SAME set-bit loop, so no second pass over cz. Bit-identical to the
    //    sequential per-edge form (same z toggles, same phase parity).
    const bool any_cz = !cz.empty();
    static thread_local std::vector<std::vector<uint64_t>> adj;   // adj[q] = neighbour mask (W words)
    if (any_cz) {
        if ((int)adj.size() < n) adj.resize(n);
        for (int q = 0; q < n; ++q) { if ((int)adj[q].size() != W) adj[q].assign(W, 0); else std::fill(adj[q].begin(), adj[q].end(), 0); }
        for (const auto& pr : cz) {
            const int c = pr.first, t = pr.second;
            adj[c][t >> 6] ^= 1ull << (t & 63);
            adj[t][c >> 6] ^= 1ull << (c & 63);
        }
    }

    auto conj_row = [&](Pauli& M) {
        // S-powers: read x (fixed), toggle z, accumulate phase. Iterate the nonzero-a columns only.
        for (const auto& qa : acols) {
            const int q = qa.first; const uint8_t aq = qa.second;
            if (M.xbit(q)) {
                if (aq & 1) M.flipz(q);                     // S / S†: toggle z_q
                M.phase = (M.phase + aq) & 3;               // S:+1 Z:+2 S†:+3
            }
        }
        // CZ network (word-parallel): iterate set x-bits c, XOR neighbour mask into z, sum endpoints.
        if (any_cz) {
            int tsum = 0;
            for (int w = 0; w < W; ++w) {
                uint64_t bits = M.x[w];
                if (!bits) continue;
                const int base = w << 6;
                do {
                    const int c = base + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    const uint64_t* ac = adj[c].data();
                    for (int u = 0; u < W; ++u) { M.z[u] ^= ac[u]; tsum += __builtin_popcountll(ac[u] & M.x[u]); }
                } while (bits);
            }
            if ((tsum >> 1) & 1) M.phase = (M.phase + 2) & 3;
        }
        // X^v: read the FINAL z bits (X Z X = -Z): phase += 2·popcount(v & z).
        if (any_v) {
            int s = 0;
            for (int w = 0; w < W; ++w) s += __builtin_popcountll(vmask[w] & M.z[w]);
            if (s & 1) M.phase = (M.phase + 2) & 3;
        }
    };
    for (auto& M : U.Xrow) conj_row(M);
    for (auto& M : U.Zrow) conj_row(M);
    U.invalidate_dual();   // the lean reduction reads forward rows only; dual rebuilt lazily if needed
}

// ───────────────────────────── from_css (the compile→sampling conversion point) ──────────────
FramedSuperposition FramedSuperposition::from_css(const CanonicalStabSum& s) {
    s.ensure_frame_current();
    FramedSuperposition L(s.n());
    L.U = s.U;
    L.eps = s.eps;
    L.free = s.free;
    auto& eb = L.entries();
    eb.resize(s.chi());
    for (int i = 0; i < s.chi(); ++i) { eb[i].first = s.branches[i].sigma; eb[i].second = s.branches[i].c; }
    L.sync_alpha_k();
    return L;
}

// ───────────────────────────── lean canonicalise (frame coords; tableau-only) ────────────────
void framed_canonicalise(FramedSuperposition& L) {
    int r = (int)L.free.size();
    int x = L.chi();
    if (r == 0 || x == 0) return;
    auto& brs = L.entries();
    const int xw = (x + 63) / 64;
    static thread_local std::vector<std::vector<uint64_t>> col;
    col.assign(r, std::vector<uint64_t>(xw, 0));
    for (int i = 1; i < x; ++i)
        for (int d = 0; d < r; ++d)
            if (brs[i].first[d] ^ brs[0].first[d]) col[d][i >> 6] |= 1ull << (i & 63);

    static thread_local std::vector<int> pivot_cols; pivot_cols.clear();
    static thread_local std::vector<std::vector<uint64_t>> red; red.clear();
    static thread_local std::vector<std::vector<uint8_t>> comp; comp.clear();
    static thread_local std::vector<uint64_t> resid;
    static thread_local std::vector<uint8_t> expand;
    bool rows_edited = false;
    auto lead_bit = [&](const std::vector<uint64_t>& v) {
        for (int w = 0; w < xw; ++w) if (v[w]) return (w << 6) + __builtin_ctzll(v[w]);
        return -1;
    };
    for (int c = 0; c < r; ++c) {
        bool varying = false;
        for (int w = 0; w < xw; ++w) if (col[c][w]) { varying = true; break; }
        if (!varying) continue;
        resid = col[c];
        expand.assign(pivot_cols.size(), 0);
        for (size_t k = 0; k < pivot_cols.size(); ++k) {
            const int lb = lead_bit(red[k]);
            if (lb >= 0 && ((resid[lb >> 6] >> (lb & 63)) & 1)) {
                for (int w = 0; w < xw; ++w) resid[w] ^= red[k][w];
                for (size_t t = 0; t < comp[k].size(); ++t) expand[t] = (uint8_t)(expand[t] ^ comp[k][t]);
            }
        }
        bool zero = true;
        for (int w = 0; w < xw; ++w) if (resid[w]) { zero = false; break; }
        if (!zero) {
            std::vector<uint8_t> cc(pivot_cols.size() + 1, 0);
            for (size_t t = 0; t < expand.size(); ++t) cc[t] = expand[t];
            cc[pivot_cols.size()] = 1;
            for (auto& v : comp) v.push_back(0);
            pivot_cols.push_back(c);
            red.push_back(resid);
            comp.push_back(std::move(cc));
        } else {
            rows_edited = true;
            const Pauli Xc = L.U.Xrow[L.free[c]];
            for (size_t k = 0; k < expand.size(); ++k) {
                if (!expand[k]) continue;
                const int p = pivot_cols[k];
                L.U.Zrow[L.free[c]] = Pauli::multiply(L.U.Zrow[L.free[c]], L.U.Zrow[L.free[p]]);
                L.U.Xrow[L.free[p]] = Pauli::multiply(L.U.Xrow[L.free[p]], Xc);
                L.eps[L.free[c]] = (uint8_t)(L.eps[L.free[c]] ^ L.eps[L.free[p]]);
                for (auto& br : brs) br.first[c] = (uint8_t)(br.first[c] ^ br.first[p]);
            }
        }
    }
    if (rows_edited) L.U.invalidate_dual();

    int rp = (int)pivot_cols.size();
    static thread_local std::vector<uint8_t> is_pivot; is_pivot.assign(r, 0);
    for (int c2 : pivot_cols) is_pivot[c2] = 1;
    for (int col2 = 0; col2 < r; ++col2) {
        if (is_pivot[col2]) continue;
        if (brs[0].first[col2]) L.eps[L.free[col2]] ^= 1;
    }
    for (int d = 0; d < rp; ++d) L.free[d] = L.free[pivot_cols[d]];
    L.free.resize(rp);
    for (auto& br : brs) {
        for (int d = 0; d < rp; ++d) br.first[d] = br.first[pivot_cols[d]];
        br.first.resize(rp);
    }
    L.sync_alpha_k();
}

// ───────────────────────────── LEAN PAULI EXPECTATION ────────────────────────────────────────
// Partner-lookup crossover chi for the Born pair-sums comes from the shared partner_hash_chi_min()
// in qeccore/pauli_kernels.hpp (same default 64 + QEC_PARTNER_HASH_MIN override) so the chi-gated
// scan/hash branch matches the oracle bit-for-bit.
namespace {

// MIRRORS CanonicalStabSum::born_from_conjugated EXACTLY on the lean fields (eps/free/branches/chi).
// Same Case A / Case B split, same packed synd build, same chi-gated scan-vs-hash partner pairing,
// same i-ascending `expt` accumulation, same ipow table — so the float result is bit-identical.
std::pair<double, double> framed_born_from_conjugated(const FramedSuperposition& L, const Pauli& Q) {
    const int N = L.n();
    const int W = (N + 63) / 64;
    bool anyx = false;
    for (int w = 0; w < W; ++w) if (Q.x[w]) { anyx = true; break; }

    if (!anyx) {
        // Case A. Q = i^{Q.phase} Z^{q_z}; s0 = i^{Q.phase}·∏_{a:q_z[a]}(-1)^{eps[a]} ∈ {±1}.
        int s0 = ((Q.phase & 3) == 2) ? -1 : 1;
        for (int a = 0; a < N; ++a) if (Q.zbit(a) && L.eps[a]) s0 = -s0;
        // Stabiliser-observable sub-case: Q.z has NO support on any FREE generator. Then lam_br = s0
        // for EVERY branch, so ⟨P⟩ = s0·Σnorm(c_br) = s0 EXACTLY (±1). Return the exact result with
        // no float accumulation (the branch coeffs carry ~1e-15 collapse drift) so the downstream
        // forced-conditional gate reads an exact hard bit. (Support on NON-free generators only
        // flips s0's sign via eps, already folded above.)
        bool free_support = false;
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (Q.zbit(L.free[d])) { free_support = true; break; }
        if (!free_support)
            return (s0 == +1) ? std::pair<double, double>{1.0, 0.0}
                              : std::pair<double, double>{0.0, 1.0};
        double pp = 0.0;
        for (const auto& br : L.entries()) {
            int dot = 0;
            for (int d = 0; d < (int)L.free.size(); ++d)
                if (Q.zbit(L.free[d]) && br.first[d]) dot ^= 1;
            int lam = dot ? -s0 : s0;
            if (lam == +1) pp += std::norm(br.second);
        }
        if (pp < 0) pp = 0; if (pp > 1) pp = 1;
        return {pp, 1.0 - pp};
    }

    // Case B: build the packed per-branch sign vectors (bit a of row i = synd[i][a]).
    const int x = L.chi();
    const auto& brs = L.entries();
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    static thread_local std::vector<uint64_t> synd_pk; synd_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = synd_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) row[w] = eps_pk[w];
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (brs[i].first[d]) row[L.free[d] >> 6] ^= 1ull << (L.free[d] & 63);
    }
    static const std::complex<double> Iunit(0, 1);
    static const std::complex<double> ipow[4] = {
        std::pow(Iunit, 0.0), std::pow(Iunit, 1.0), std::pow(Iunit, 2.0), std::pow(Iunit, 3.0)};
    static thread_local std::vector<uint64_t> tgt; tgt.resize(W);
    std::complex<double> expt(0, 0);
    if (x < partner_hash_chi_min()) {
        for (int i = 0; i < x; ++i) {
            const uint64_t* si = synd_pk.data() + (size_t)i * W;
            for (int w = 0; w < W; ++w) tgt[w] = si[w] ^ Q.x[w];      // partner signature of i
            for (int j = 0; j < x; ++j) {
                const uint64_t* sj = synd_pk.data() + (size_t)j * W;
                bool match = true;
                for (int w = 0; w < W; ++w) if (sj[w] != tgt[w]) { match = false; break; }
                if (!match) continue;
                int par = 0;
                for (int w = 0; w < W; ++w) par += __builtin_popcountll(Q.z[w] & sj[w]);
                expt += std::conj(brs[i].second) * brs[j].second
                        * ipow[(Q.phase + 2 * (par & 1)) & 3];
            }
        }
    } else {
        static thread_local std::unordered_map<std::string, int> sig_idx;
        sig_idx.clear();
        sig_idx.reserve((size_t)x * 2);
        for (int j = 0; j < x; ++j) {
            const uint64_t* sj = synd_pk.data() + (size_t)j * W;
            sig_idx.emplace(
                std::string(reinterpret_cast<const char*>(sj), (size_t)W * 8), j);
        }
        for (int i = 0; i < x; ++i) {
            const uint64_t* si = synd_pk.data() + (size_t)i * W;
            for (int w = 0; w < W; ++w) tgt[w] = si[w] ^ Q.x[w];      // partner signature of i
            auto it = sig_idx.find(
                std::string(reinterpret_cast<const char*>(tgt.data()), (size_t)W * 8));
            if (it == sig_idx.end()) continue;                        // no partner: term vanishes
            const int j = it->second;
            const uint64_t* sj = synd_pk.data() + (size_t)j * W;
            int par = 0;
            for (int w = 0; w < W; ++w) par += __builtin_popcountll(Q.z[w] & sj[w]);
            expt += std::conj(brs[i].second) * brs[j].second
                    * ipow[(Q.phase + 2 * (par & 1)) & 3];
        }
    }
    double pp = 0.5 * (1.0 + std::real(expt));
    if (pp < 0) pp = 0; if (pp > 1) pp = 1;
    return {pp, 1.0 - pp};
}
}  // namespace

std::pair<double, double> framed_expectation(const FramedSuperposition& L, const Pauli& P) {
    // Q = U†PU: cached inverse rows when the dual is valid (conjugate, O(weight)); else the forward
    // rows (dual_image, O(n²)) — never force a dual rebuild. Both give the same unique Q, so the
    // result matches born_probabilities bit-for-bit (same Q feeds the identical Born math below).
    Pauli Q = L.U.dual_valid() ? L.U.conjugate(P) : L.U.dual_image(P);
    return framed_born_from_conjugated(L, Q);
}

// ───────────────────── Compile-time bare-state certification API (Task 1) ─────────────────────
std::vector<Pauli> FramedSuperposition::certified_stabilizers() const {
    const int N = n();
    std::vector<uint8_t> is_free((size_t)(N > 0 ? N : 1), 0);
    for (int f : free)
        if (f >= 0 && f < N) is_free[(size_t)f] = 1;
    std::vector<Pauli> gens;
    gens.reserve((size_t)(N - k()));
    for (int a = 0; a < N; ++a) {
        if (is_free[(size_t)a]) continue;
        // g_a = U Z_a U† = Zrow[a]; the +1 stabiliser of |psi> is (-1)^{eps[a]}·g_a (eps carries the
        // reference sign g_a|ref> = (-1)^{eps[a]}|ref>, and g_a passes through the magic superposition
        // because a∉free ⇒ g_a commutes with every Xrow[free[d]]). Fold (-1)^{eps} into the phase.
        Pauli g = U.Zrow[a];
        if (eps[a]) g.phase = (g.phase + 2) & 3;   // ×(-1) = ×i²
        gens.push_back(std::move(g));
    }
    return gens;
}

std::complex<double> FramedSuperposition::pauli_expectation(const Pauli& P) const {
    // Hermitian canonical H(x,z) = i^{x·z} X^x Z^z; <H> = p+ − p− is exact (real) via the reused
    // framed_expectation engine. P = i^{(P.phase − x·z) mod 4}·H ⇒ <P> = i^{(P.phase − x·z) mod 4}·<H>.
    const int d = P.xz_overlap() & 3;
    Pauli H = P;
    H.phase = d;
    const std::pair<double, double> pr = framed_expectation(*this, H);
    const double h = pr.first - pr.second;                  // <H> ∈ [-1,1]
    static const std::complex<double> I4[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const int ph = (((P.phase - d) % 4) + 4) % 4;
    return I4[ph] * h;
}

// ───────────────────────────── THE LEAN ANTICOMMUTING MEASUREMENT ────────────────────────────
int framed_measure_anticommuting(FramedSuperposition& L, int pauli, int q, const Pauli& P,
                               const std::vector<int>& A, double u) {
    // Single-qubit lab read: conjugate via the cached inverse (O(n)) and delegate. labP = P (built
    // here identically to the old in-body labP from pauli/q), Q = U† P U.
    Pauli Q = L.U.conjugate_single(pauli, q);
    return framed_measure_anticommuting_general(L, P, Q, A, u);
}

int framed_measure_anticommuting_general(FramedSuperposition& L, const Pauli& P, const Pauli& Q,
                                       const std::vector<int>& A, double u) {
    const int N = L.n();
    const int x = L.chi();
    const auto& brs = L.entries();
    const int W = (N + 63) / 64;

    static thread_local std::vector<int> free_pos; free_pos.assign(N, -1);
    for (int d = 0; d < (int)L.free.size(); ++d) free_pos[L.free[d]] = d;
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    static thread_local std::vector<uint64_t> synd_pk; synd_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = synd_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) row[w] = eps_pk[w];
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (brs[i].first[d]) row[L.free[d] >> 6] ^= 1ull << (L.free[d] & 63);
    }
    auto synd_bit = [&](int i, int a) -> uint64_t {
        return (synd_pk[(size_t)i * W + (a >> 6)] >> (a & 63)) & 1ull;
    };

    static thread_local std::vector<std::complex<double>> c0; c0.resize(x);
    for (int i = 0; i < x; ++i) c0[i] = brs[i].second;

    static thread_local std::vector<Pauli> oW; oW.resize(x);
    for (int i = 0; i < x; ++i) {
        reset_pauli_inplace(oW[i], N);
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (brs[i].first[d]) pmul_into(oW[i], L.U.Xrow[L.free[d]]);
    }

    int p = -1;
    for (int a : A) if (free_pos[a] >= 0) {
        bool vary = false;
        for (int i = 1; i < x; ++i) if (synd_bit(i, a) != synd_bit(0, a)) { vary = true; break; }
        if (vary) { p = a; break; }
    }
    if (p < 0) for (int a : A) if (free_pos[a] >= 0) { p = a; break; }
    if (p < 0) p = A[0];

    Pauli s_lab = L.U.Zrow[p];
    if (L.eps[p]) s_lab.phase = (s_lab.phase + 2) & 3;

    Pauli Qr = Q;
    {
        static thread_local std::vector<uint64_t> amask; amask.assign(W, 0);
        for (int a : A) if (a != p) amask[a >> 6] |= 1ull << (a & 63);
        for (int i = 0; i < x; ++i) {
            uint64_t* row = synd_pk.data() + (size_t)i * W;
            if ((row[p >> 6] >> (p & 63)) & 1ull)
                for (int w = 0; w < W; ++w) row[w] ^= amask[w];
        }
    }
    for (int a : A) {
        if (a == p) continue;
        pmul_into(L.U.Zrow[a], L.U.Zrow[p]);
        pmul_into(L.U.Xrow[p], L.U.Xrow[a]);
        conj_cx_inplace(Qr, p, a);
    }

    static const std::complex<double> Iunit(0, 1);
    std::complex<double> Iph = std::pow(Iunit, (double)(Qr.phase & 3));
    auto qz_par = [&](int j) {
        const uint64_t* row = synd_pk.data() + (size_t)j * W;
        int s = 0; for (int w = 0; w < W; ++w) s += __builtin_popcountll(Qr.z[w] & row[w]);
        return s & 1;
    };
    auto mu = [&](int j) -> std::complex<double> { return Iph * (double)(qz_par(j) ? -1 : 1); };

    static thread_local std::vector<uint64_t> post_pk; post_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        const uint64_t* src = synd_pk.data() + (size_t)i * W;
        uint64_t* dst = post_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) dst[w] = src[w];
        dst[p >> 6] &= ~(1ull << (p & 63));
    }
    auto post_eq = [&](int i, int j) {
        const uint64_t* a = post_pk.data() + (size_t)i * W;
        const uint64_t* b = post_pk.data() + (size_t)j * W;
        for (int w = 0; w < W; ++w) if (a[w] != b[w]) return false;
        return true;
    };

    // Born pairing: branch i's partner is the UNIQUE j with the same post-key (pivot column
    // zeroed) and the OPPOSITE pivot bit — the framed rep keeps the full packed syndrome rows
    // distinct (distinct branch sigma keys map injectively onto synd rows; the amask rotation
    // above is a bit-p-conditioned XOR by a p-free mask, so it preserves distinctness), hence at
    // most one j per i. CHI-GATED (see partner_hash_chi_min, same crossover as the Born hash in
    // framed_born_from_conjugated and the oracle's measure Step-2 hash): small chi keeps the
    // O(chi^2 W) scan; large chi uses the O(chi W) (post-key, pivot-bit)->index hash. BOTH keep
    // the outer i ascending with exactly one partner term each, so `expt` accumulates in the
    // identical FP order — bit-for-bit identical either side of the crossover.
    std::complex<double> expt(0, 0);
    if (x < partner_hash_chi_min()) {
        for (int i = 0; i < x; ++i)
            for (int j = 0; j < x; ++j) {
                if (!post_eq(i, j)) continue;
                if (synd_bit(i, p) == synd_bit(j, p)) continue;
                expt += std::conj(brs[i].second) * brs[j].second * mu(j);
            }
    } else {
        static thread_local std::unordered_map<std::string, int> post_idx;
        post_idx.clear();
        post_idx.reserve((size_t)x * 2);
        for (int j = 0; j < x; ++j) {
            std::string key(reinterpret_cast<const char*>(post_pk.data() + (size_t)j * W),
                            (size_t)W * 8);
            key.push_back((char)synd_bit(j, p));           // distinguish the two pivot-bit members
            // INVARIANT (defended): each (post-key, pivot-bit) keys at most one branch, so the
            // hash sums exactly the pairs the scan keeps (a duplicate would drop one).
            auto ins = post_idx.emplace(std::move(key), j);
            assert(ins.second && "collapse pairing hash: duplicate (post-key,pivot) (non-canonical state)");
            (void)ins;
        }
        for (int i = 0; i < x; ++i) {
            std::string key(reinterpret_cast<const char*>(post_pk.data() + (size_t)i * W),
                            (size_t)W * 8);
            key.push_back((char)(1 - synd_bit(i, p)));     // the partner has the opposite pivot bit
            auto it = post_idx.find(key);
            if (it == post_idx.end()) continue;            // no partner: branch i is unpaired here
            const int j = it->second;
            expt += std::conj(brs[i].second) * brs[j].second * mu(j);
        }
    }
    double pp = 0.5 * (1.0 + std::real(expt));
    if (pp < 0) pp = 0;
    if (pp > 1) pp = 1;
    int m = (u < pp) ? +1 : -1;
    double pm = (m == +1) ? pp : (1.0 - pp);
    double inv = (pm > 0) ? 1.0 / std::sqrt(pm) : 0.0;

    static thread_local std::vector<int> order; order.resize(x);
    for (int i = 0; i < x; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        const uint64_t* a = post_pk.data() + (size_t)i * W;
        const uint64_t* b = post_pk.data() + (size_t)j * W;
        for (int w = 0; w < W; ++w) if (a[w] != b[w]) {
            const int t = __builtin_ctzll(a[w] ^ b[w]);
            return ((a[w] >> t) & 1ull) == 0;
        }
        return false;
    });
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    struct Surv { std::complex<double> coeff; int rep; };
    static thread_local std::vector<Surv> surv; surv.clear();
    for (int beg = 0; beg < x; ) {
        int end = beg + 1;
        while (end < x && post_eq(order[beg], order[end])) ++end;
        int w0 = -1, w1 = -1;
        for (int t = beg; t < end; ++t) { const int i = order[t]; if (synd_bit(i, p)) w1 = i; else w0 = i; }
        beg = end;
        if (w0 >= 0 && w1 >= 0) {
            std::complex<double> nu = inv_sqrt2 * (c0[w0] + (double)m * std::conj(mu(w0)) * c0[w1]) * inv;
            if (std::abs(nu) < 1e-12) continue;
            surv.push_back({nu, w0});
        } else {
            int w = (w0 >= 0) ? w0 : w1;
            std::complex<double> nu = inv_sqrt2 * c0[w] * inv;
            if (std::abs(nu) < 1e-12) continue;
            surv.push_back({nu, w});
        }
    }

    Pauli newpiv = P;
    if (m == -1) newpiv.phase = (newpiv.phase + 2) & 3;
    Pauli new_dp = L.U.Zrow[p];
    L.U.Zrow[p] = newpiv;
    {
        const uint64_t* npx = newpiv.x.data();
        const uint64_t* npz = newpiv.z.data();
        for (int a = 0; a < N; ++a) {
            if (a == p) continue;
            const Pauli& Xa = L.U.Xrow[a];
            int ac = 0;
            for (int w = 0; w < W; ++w) ac += __builtin_popcountll(Xa.x[w] & npz[w]) + __builtin_popcountll(Xa.z[w] & npx[w]);
            if (ac & 1) pmul_into(L.U.Xrow[a], new_dp);
        }
    }
    L.U.Xrow[p] = new_dp;
    L.U.invalidate_dual();

    const int xs = (int)surv.size();
    if (xs == 0) {
        L.entries().clear(); L.free.clear(); L.sync_alpha_k(); L.eps.assign(N, 0);
        return m;
    }
    static thread_local std::vector<uint64_t> fsynd; fsynd.assign((size_t)xs * W, 0);
    for (int g = 0; g < xs; ++g) {
        const uint64_t* src = synd_pk.data() + (size_t)surv[g].rep * W;
        uint64_t* dst = fsynd.data() + (size_t)g * W;
        for (int w = 0; w < W; ++w) dst[w] = src[w];
        dst[p >> 6] &= ~(1ull << (p & 63));
    }
    auto fbit = [&](int g, int a) -> uint64_t {
        return (fsynd[(size_t)g * W + (a >> 6)] >> (a & 63)) & 1ull;
    };

    {
        const int gw = (xs + 63) >> 6;
        static thread_local std::vector<uint64_t> fsyndT;
        fsyndT.assign((size_t)N * gw, 0);
        for (int g = 0; g < xs; ++g)
            for (int a = 0; a < N; ++a)
                if (fbit(g, a)) fsyndT[(size_t)a * gw + (g >> 6)] |= 1ull << (g & 63);
        static thread_local std::vector<uint64_t> onesg; onesg.assign(gw, ~0ull);
        if (xs & 63) onesg[gw - 1] = (1ull << (xs & 63)) - 1;
        static thread_local std::vector<int> basis; basis.clear();
        static thread_local std::vector<std::vector<uint64_t>> redcol; redcol.clear();
        static thread_local std::vector<std::vector<uint8_t>> comp; comp.clear();
        size_t nb = 0;
        static thread_local std::vector<uint64_t> ecol, resid;
        ecol.resize(gw); resid.resize(gw);
        static thread_local std::vector<uint8_t> expand;
        for (int a = 0; a < N; ++a) {
            const uint64_t* ta = fsyndT.data() + (size_t)a * gw;
            const uint64_t b0 = ta[0] & 1ull;
            for (int w = 0; w < gw; ++w) ecol[w] = ta[w] ^ (b0 ? onesg[w] : 0ull);
            bool allzero = true; for (int w = 0; w < gw; ++w) if (ecol[w]) { allzero = false; break; }
            if (allzero) continue;
            resid = ecol;
            expand.assign(nb, 0);
            for (size_t bi = 0; bi < nb; ++bi) {
                const std::vector<uint64_t>& rc = redcol[bi];
                int pr = -1;
                for (int w = 0; w < gw; ++w) if (rc[w]) { pr = (w << 6) + __builtin_ctzll(rc[w]); break; }
                if (pr >= 0 && ((resid[pr >> 6] >> (pr & 63)) & 1ull)) {
                    for (int w = 0; w < gw; ++w) resid[w] ^= rc[w];
                    for (size_t k = 0; k < comp[bi].size(); ++k) expand[k] ^= comp[bi][k];
                }
            }
            bool rz = true; for (int w = 0; w < gw; ++w) if (resid[w]) { rz = false; break; }
            if (rz) {
                Pauli dofa = L.U.Xrow[a];
                for (size_t k = 0; k < expand.size(); ++k) if (expand[k]) {
                    int abase = basis[k];
                    pmul_into(L.U.Zrow[a], L.U.Zrow[abase]);
                    pmul_into(L.U.Xrow[abase], dofa);
                    for (int g = 0; g < xs; ++g) {
                        uint64_t* row = fsynd.data() + (size_t)g * W;
                        row[a >> 6] ^= ((row[abase >> 6] >> (abase & 63)) & 1ull) << (a & 63);
                    }
                }
            } else {
                for (size_t bi = 0; bi < nb; ++bi) comp[bi].push_back(0);
                if (nb >= redcol.size()) { redcol.emplace_back(); comp.emplace_back(); }
                redcol[nb] = resid;
                std::vector<uint8_t>& cc = comp[nb]; cc.assign(nb + 1, 0); cc[nb] = 1;
                for (size_t k = 0; k < expand.size(); ++k) cc[k] = expand[k];
                basis.push_back(a);
                ++nb;
            }
        }
    }

    L.eps.assign(N, 0);
    for (int a = 0; a < N; ++a) L.eps[a] = (uint8_t)fbit(0, a);
    static thread_local std::vector<uint64_t> varies; varies.assign(W, 0);
    for (int g = 1; g < xs; ++g) {
        const uint64_t* rg = fsynd.data() + (size_t)g * W;
        for (int w = 0; w < W; ++w) varies[w] |= rg[w] ^ fsynd[w];
    }
    static thread_local std::vector<int> nfree; nfree.clear();
    for (int w = 0; w < W; ++w) {
        uint64_t bits = varies[w]; const int base = w << 6;
        while (bits) { nfree.push_back(base + __builtin_ctzll(bits)); bits &= bits - 1; }
    }
    L.free = nfree;
    L.sync_alpha_k();

    static thread_local std::vector<uint64_t> epsn_pk; epsn_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) epsn_pk[a >> 6] |= 1ull << (a & 63);
    const Pauli& labP = P;     // the standard lab Pauli of the read (single-qubit or general)
    static const std::complex<double> ipw[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    static thread_local std::vector<Pauli> Wgs; Wgs.resize(xs);
    static thread_local std::vector<const Pauli*> Wtg; Wtg.resize(xs);
    for (int g = 0; g < xs; ++g) {
        Wgs[g] = oW[surv[g].rep];
        int ac = 0;
        for (int w = 0; w < W; ++w)
            ac += __builtin_popcountll(Wgs[g].x[w] & labP.z[w]) + __builtin_popcountll(Wgs[g].z[w] & labP.x[w]);
        if (ac & 1) pmul_into(Wgs[g], s_lab);
        Wtg[g] = &Wgs[g];
    }
    static thread_local std::vector<Pauli> Qfs; Qfs.resize(xs);
    dual_image_rows_scan(L.U.Xrow, L.U.Zrow, N, Wtg.data(), xs, Qfs.data());
    static thread_local std::vector<FramedSuperposition::Entry> nb; nb.clear();
    for (int g = 0; g < xs; ++g) {
        const Pauli& Qf = Qfs[g];
        FramedSuperposition::Entry br;
        br.first.resize(nfree.size());
        for (int d = 0; d < (int)nfree.size(); ++d) br.first[d] = (uint8_t)Qf.xbit(nfree[d]);
        int zs = 0; for (int w = 0; w < W; ++w) zs += __builtin_popcountll(Qf.z[w] & epsn_pk[w]);
        std::complex<double> gph = ipw[Qf.phase & 3];
        if (zs & 1) gph = -gph;
        br.second = surv[g].coeff * gph;
        nb.push_back(std::move(br));
    }
    L.entries().swap(nb);
    framed_canonicalise(L);   // restore Minimality (independent free set) for the post-measurement rep
    return m;
}

// ───────────────────────────── Case-A commuting split (shared by batch_measure + ref read) ────
// For a COMMUTING read with conjugate Qf (= U†PU, all X-bits zero), classify each branch by the
// read eigenvalue bit (0:+1, 1:-1) and return the +1/-1 branch weights in p0/p1. The chosen bit is
// returned as a convenience (b argument for callers that want the deterministic outcome). No state
// mutation. This is the exact eigenvalue column the affine measure_single Case A computes (s0 ·
// (-1)^{q_z·σ̃_i}), re-expressed on the lean fields.
static int framed_caseA_eig(const FramedSuperposition& L, const Pauli& Qf, double& p0, double& p1) {
    const int N = L.n();
    const auto& brs = L.entries();
    int zpar = 0; for (int a = 0; a < N; ++a) if (Qf.zbit(a) && L.eps[a]) zpar ^= 1;
    int base = (((Qf.phase & 3) == 2) ? 1 : 0) ^ zpar;
    p0 = 0; p1 = 0;
    int firstbit = 0;
    for (int i = 0; i < L.chi(); ++i) {
        int sp = 0;
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (brs[i].first[d] && Qf.zbit(L.free[d])) sp ^= 1;
        int b = base ^ sp;
        if (i == 0) firstbit = b;
        (b ? p1 : p0) += std::norm(brs[i].second);
    }
    return firstbit;
}

// Collapse a commuting Case-A read to the eigenvalue bit `bsel` (0:+1, 1:-1): keep the branches with
// that eigenvalue, renormalise, re-canonicalise. Shares the exact keep/renorm/canonicalise block
// the affine measure_single Case A (and batch_measure's inline split) use. Mutates L in place.
static void framed_split_to_outcome(FramedSuperposition& L, const Pauli& Qf, int bsel) {
    const int N = L.n();
    const auto& brs = L.entries();
    int zpar = 0; for (int a = 0; a < N; ++a) if (Qf.zbit(a) && L.eps[a]) zpar ^= 1;
    int base = (((Qf.phase & 3) == 2) ? 1 : 0) ^ zpar;
    std::vector<FramedSuperposition::Entry> nb;
    for (int i = 0; i < L.chi(); ++i) {
        int sp = 0;
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (brs[i].first[d] && Qf.zbit(L.free[d])) sp ^= 1;
        if ((base ^ sp) == bsel) nb.push_back(brs[i]);
    }
    double nrm = 0; for (auto& br : nb) nrm += std::norm(br.second);
    double s = nrm > 0 ? 1.0 / std::sqrt(nrm) : 1.0;
    for (auto& br : nb) br.second *= s;
    L.entries().swap(nb);
    framed_canonicalise(L);
}

// ───────────────────────────── LEAN REFERENCE-COLLAPSE PRIMITIVE ──────────────────────────────
// Classify + collapse a single-qubit read at the u=0 reference, mirroring the affine loop body's
// born_probabilities + is_commuting_single + measure_single(...,0.0). MUTATES L to the reference
// post-state. Reference convention (matches CanonicalStabSum::measure_single with u=0):
//   out = (pp>1e-12) ? +1 : -1, i.e. m = (0 < pp) ? +1 : -1.
FramedRefRead framed_reference_read(FramedSuperposition& L, int pauli, int q) {
    const int N = L.n();
    FramedRefRead r;
    r.chi_pre = L.chi();
    Pauli P = single_pauli(pauli, q, N);
    // (pp,pm): Born probs P(+1),P(-1) BEFORE collapse — bit-identical to born_probabilities.
    auto pr = framed_expectation(L, P);
    r.pp = pr.first; r.pm = pr.second;
    r.is_coin = (r.pp > 1e-12 && r.pm > 1e-12);
    r.out = (r.pp > 1e-12) ? +1 : -1;
    // Qf = U† P U: Case B iff it has any X-bit (the is_commuting_single test).
    Pauli Qf = L.U.dual_valid() ? L.U.conjugate(P) : L.U.dual_image(P);
    bool caseB = false;
    const int W = (N + 63) / 64;
    for (int w = 0; w < W; ++w) if (Qf.x[w]) { caseB = true; break; }
    r.commuting = !caseB;
    if (caseB) {
        // Anticommuting collapse. Pass u=0 so m = (0<pp)?+1:-1 matches the reference out exactly
        // (the same u=0 the affine measure_single uses on the reference path).
        std::vector<int> A;
        for (int a = 0; a < N; ++a) if (Qf.xbit(a)) A.push_back(a);
        int m = framed_measure_anticommuting_general(L, P, Qf, A, 0.0);
        (void)m;   // m == r.out by construction (u=0)
    } else {
        // Commuting read. Deterministic (pp or pm ≈ 0): no state change. Coin (χ-discriminating):
        // keep the branches consistent with the reference outcome (bit 0:+1 / 1:-1).
        if (r.is_coin) {
            const int bsel = (r.out == +1) ? 0 : 1;
            framed_split_to_outcome(L, Qf, bsel);
        }
    }
    return r;
}

// ───────────────────────────── BATCH measurement (commuting reads, no post-state) ────────────
// ── Batch-conjugate single-qubit reads: out[k] = U† P_{read k} U, all in ONE forward-row pass.
//    The reads are single-qubit, so this is dual_image_rows_scan over the SAME 2N forward rows for
//    every target — amortizing the row iteration that a per-read conjugate_single (O(n²) each on a
//    stale dual) re-pays |reads| times. Bit-identical to conjugate_single(reads[k]...) per read (same
//    forward-row decomposition + phase convention as dual_image). `Qs` is resized to reads.size().
static void batch_conjugate_reads(const FramedSuperposition& L,
                                  const std::vector<std::pair<int, int>>& reads,
                                  std::vector<Pauli>& Qs) {
    const int N = L.n();
    const size_t M = reads.size();
    Qs.resize(M);
    if (M == 0) return;
    static thread_local std::vector<Pauli> Tg;   // single-qubit lab Paulis (targets)
    static thread_local std::vector<const Pauli*> Tp;
    Tg.resize(M); Tp.resize(M);
    for (size_t k = 0; k < M; ++k) {
        reset_pauli_inplace(Tg[k], N);
        const int pauli = reads[k].first, q = reads[k].second;
        if (pauli == 0) Tg[k].setx(q);
        else if (pauli == 1) { Tg[k].setx(q); Tg[k].setz(q); Tg[k].phase = 1; }
        else Tg[k].setz(q);
        Tp[k] = &Tg[k];
    }
    dual_image_rows_scan(L.U.Xrow, L.U.Zrow, N, Tp.data(), (int)M, Qs.data());
}

// chi==1 batch, conjugates already computed (Qpre[k] = U† P_{read k} U). Shared by batch_chi1 and
// batch_measure's pass-2 (which precomputes conjugates in one forward pass). Behaviour-identical to
// the per-read conjugate_single form; only the conjugation source differs.
static void batch_chi1_pre(const FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                           const std::vector<Pauli>& Qpre, std::function<double()> rng,
                           std::vector<int>& out) {
    const int N = L.n(), W = (N + 63) / 64;
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    static thread_local std::vector<MeasCoin> coins; coins.clear();
    out.assign(reads.size(), 0);
    for (size_t k = 0; k < reads.size(); ++k) {
        Pauli acc = Qpre[k];
        int sign = 0;
        for (auto& c : coins)
            if ((acc.x[c.pivot >> 6] >> (c.pivot & 63)) & 1ull) { pmul_into(acc, c.Qc); sign ^= c.o; }
        int xnz = -1;
        for (int w = 0; w < W; ++w) if (acc.x[w]) { xnz = (w << 6) + __builtin_ctzll(acc.x[w]); break; }
        if (xnz < 0) {
            int zpar = 0; for (int w = 0; w < W; ++w) zpar += __builtin_popcountll(acc.z[w] & eps_pk[w]);
            int ob = (((acc.phase & 3) == 2) ? 1 : 0) ^ (zpar & 1) ^ sign;
            out[k] = ob;
        } else {
            int o = (rng() < 0.5) ? 0 : 1;
            out[k] = o ^ sign;
            coins.push_back({std::move(acc), xnz, o});
        }
    }
}

void batch_chi1(const FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                std::function<double()> rng, std::vector<int>& out) {
    static thread_local std::vector<Pauli> Qpre;
    batch_conjugate_reads(L, reads, Qpre);
    batch_chi1_pre(L, reads, Qpre, rng, out);
}

// COLUMN-form couple test — no conjugation needed. For a single-qubit read P_q, the conjugate
// Qf = U† P_q U has, on forward index b, an X-bit iff ⟨P_q, Zrow[b]⟩ and a Z-bit iff ⟨P_q, Xrow[b]⟩
//   (dual_image's α_b/β_b). For a single-qubit P_q the symplectic product ⟨P_q,R⟩ reads ONLY
//   column q of R: Z-read → R.xbit(q); X-read → R.zbit(q); Y-read → R.xbit(q)^R.zbit(q). So Qf
//   couples a free generator fa iff that column bit is set in Zrow[fa] OR Xrow[fa] — an O(|free|)
//   column lookup, NOT an O(n²) dual_image. Only the ONE picked read is then fully conjugated.
// Shared with the TreePlan extractor (framed_sampler.cpp) so its forced-collapse order matches
// batch_measure's Pass-1 by construction.
bool framed_read_couples_free(const FramedSuperposition& L, int pauli, int q) {
    auto sp_col = [](const Pauli& R, int pauli, int q) -> int {
        if (pauli == 2) return (int)R.xbit(q);                       // ⟨Z_q,R⟩ = R.x[q]
        if (pauli == 0) return (int)R.zbit(q);                       // ⟨X_q,R⟩ = R.z[q]
        return (int)R.xbit(q) ^ (int)R.zbit(q);                      // ⟨Y_q,R⟩ = R.x[q]^R.z[q]
    };
    for (int fa : L.free)
        if (sp_col(L.U.Zrow[fa], pauli, q) || sp_col(L.U.Xrow[fa], pauli, q)) return true;
    return false;
}

void batch_measure(FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                   std::function<double()> rng, std::vector<int>& out) {
    const int N = L.n(), W = (N + 63) / 64;
    out.assign(reads.size(), -1);
    while (L.chi() > 1) {
        int pick = -1; Pauli Qf;
        for (size_t k = 0; k < reads.size(); ++k) if (out[k] == -1) {
            if (framed_read_couples_free(L, reads[k].first, reads[k].second)) { pick = (int)k; break; }
        }
        if (pick >= 0) Qf = L.U.conjugate_single(reads[pick].first, reads[pick].second);
        if (pick < 0) break;
        bool caseB = false; for (int w = 0; w < W; ++w) if (Qf.x[w]) { caseB = true; break; }
        if (!caseB) {
            double p0, p1;
            int b = framed_caseA_eig(L, Qf, p0, p1);
            double tot = p0 + p1, pr0 = tot > 0 ? p0 / tot : 0.5;
            int bsel = (rng() < pr0) ? 0 : 1;
            (void)b;
            framed_split_to_outcome(L, Qf, bsel);
            out[pick] = bsel;
        } else {
            static thread_local std::vector<int> A; A.clear();
            for (int a = 0; a < N; ++a) if (Qf.xbit(a)) A.push_back(a);
            Pauli P = single_pauli(reads[pick].first, reads[pick].second, N);
            int m = framed_measure_anticommuting(L, reads[pick].first, reads[pick].second, P, A, rng());
            out[pick] = (m == -1) ? 1 : 0;
        }
    }
    // Pass-2 (chi==1): the branch-independent remainder. batch_chi1 conjugates the remaining reads in
    // ONE batched forward pass (dual_image_rows_scan), not |rem| × O(n²) per-read conjugate_single.
    static thread_local std::vector<std::pair<int, int>> rem; rem.clear();
    static thread_local std::vector<int> remidx; remidx.clear();
    for (size_t k = 0; k < reads.size(); ++k) if (out[k] == -1) { rem.push_back(reads[k]); remidx.push_back((int)k); }
    static thread_local std::vector<int> ro; batch_chi1(L, rem, rng, ro);
    for (size_t j = 0; j < remidx.size(); ++j) out[remidx[j]] = ro[j];
}


// ═════════════════════════ the framed method surface (former framed_superposition.cpp TU) ═══

// [from_css lives in the kernels half of this TU — the compile→sampling conversion point: frame made current,
//  (U, eps, free) copied, branch (σ, c) pairs copied VERBATIM (order preserved, no dedup/prune).
//  The pre-merge FramedSuperposition::from_css routed through make_dense, which additionally
//  merged duplicate σ keys and dropped |c|<1e-15 entries; CanonicalStabSum branches are distinct,
//  and zero-coefficient entries contribute 0 to every container op, so the two agreed on all
//  production states. The verbatim copy is the byte-oracle-covered production semantics.]

// ── expectation ─────────────────────────────────────────────────────────────────────────────
// <P> = p₊ − p₋, expressed through the amplitude container ops. Derivation (matches the oracle
// CanonicalStabSum::born_probabilities → born_from_conjugated exactly):
//
//   Q = U† P U. Each branch i carries the generator-sign syndrome synd[i][a] = eps[a] ⊕ σ_i over
//   the free positions (synd[i][free[d]] ^= σ_i[d]). The oracle Born sum pairs branches i,j with
//   synd[j] = synd[i] ⊕ Q.x and accumulates conj(c_i) c_j · i^{Q.phase + 2·(Q.z · synd[j])}.
//
//   Q.x can only be supported on free positions (else no partner exists → <P> = 0): Xa = { b :
//   Q.xbit(b) && b∉free } non-empty ⇒ return 0. On the free positions define
//       y[d]   = Q.xbit(free[d])      (partner XOR: σ_j = σ_i ⊕ y),
//       sgn[d] = Q.zbit(free[d]).
//   Split the parity Q.z·synd[j] = P0 ⊕ Σ_d sgn[d]·σ_j[d], with the branch-independent part
//       P0 = Σ_a Q.zbit(a)·eps[a]  (mod 2).
//   The container product β = shift_y( diag_{(-1)^sgn}( clone α ) ) — diagonal FIRST, then shift, so
//   β_x = (-1)^{Σ sgn·(x⊕y)} α_{x⊕y}; inner(α, β) then already carries BOTH the σ_j·sgn branch sign
//   and the shift, so the reindexing needs no extra Q.xz_overlap correction. Folding i^{Q.phase} and
//   the branch-independent (-1)^{P0}:
//       c = i^{Q.phase} · (-1)^{P0},
//       <P> = Re( c · inner(α, β) ) / ||α||².
//   (Dividing by ||α||² makes it robust to un-normalised coefficients; the oracle assumes ‖·‖=1.)
//   [NB: an earlier draft carried a spurious (-1)^{Q.xz_overlap()} term from a diag-AFTER-shift
//    ordering; the diag-then-shift order used below is correct — verified against the oracle.]
double FramedSuperposition::expectation(const Pauli& P) const {
    const int kk = k();
    // Q = U† P U. dual_image reads the forward rows only (needs no valid inverse tableau).
    const Pauli Q = U.dual_valid() ? U.conjugate(P) : U.dual_image(P);

    // Active-vs-consumed X support: any X-support off the free set kills the expectation.
    static thread_local std::vector<uint8_t> is_free;
    is_free.assign(n(), 0);
    for (int d = 0; d < kk; ++d) is_free[free[d]] = 1;
    for (int b = 0; b < n(); ++b)
        if (Q.xbit(b) && !is_free[b]) return 0.0;

    // Decode y (partner shift) and sgn (diagonal signs) over the free set.
    std::vector<uint8_t> y(kk, 0);
    std::vector<cd> g(kk, cd(1.0, 0.0));
    for (int d = 0; d < kk; ++d) {
        y[d] = Q.xbit(free[d]) ? 1 : 0;
        if (Q.zbit(free[d])) g[d] = cd(-1.0, 0.0);   // (-1)^{sgn[d]}
    }

    // Global phase c = i^{Q.phase} · (-1)^{P0},  P0 = Σ_a Q.zbit(a)·eps[a]  (mod 2). The container
    // inner(α, shift_y∘diag_g(α)) already carries the σ_j·sgn branch sign AND the shift, so no
    // extra Q.xz_overlap correction is needed (verified against the oracle on the Y-on-free case).
    int par = 0;
    for (int a = 0; a < n(); ++a)
        if (Q.zbit(a) && eps[a]) par ^= 1;               // P0
    static const cd ipow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    cd c = ipow[Q.phase & 3];
    if (par & 1) c = -c;

    // β = shift_y( diag_g( clone α ) );  <P> = Re(c · inner(α, β)) / ||α||².
    std::unique_ptr<Amplitudes> beta = alpha->clone();
    beta->apply_diagonal(cd(1.0, 0.0), g);
    beta->shift(y);
    const cd num = c * alpha->inner(*beta);
    const double nrm2 = alpha->norm2();
    if (nrm2 < 1e-30) return 0.0;
    return num.real() / nrm2;
}

// ── apply_clifford ───────────────────────────────────────────────────────────────────────────
// Design §3.2: Clifford gates update only the frame (U, eps unchanged as the reference state sign
// is absorbed into the tableau) — α is left UNTOUCHED. Dispatch 8-way on `kind`:
//   0:H  1:S  2:Sdg  3:X  4:Y  5:Z  (single-qubit on a, b ignored)
//   6:CX  7:CZ  (two-qubit: control=a, target/other=b)
void FramedSuperposition::apply_clifford(uint8_t kind, int a, int b) {
    switch (kind) {
        case 0: U.left_h(a);      break;
        case 1: U.left_s(a);      break;
        case 2: U.left_sdg(a);    break;
        case 3: U.left_x(a);      break;
        case 4: U.left_y(a);      break;
        case 5: U.left_z(a);      break;
        case 6: U.left_cx(a, b);  break;
        case 7: U.left_cz(a, b);  break;
        default: throw std::runtime_error("FramedSuperposition::apply_clifford: unknown kind");
    }
}
// ── born_p1 — the +1 Born probability measure_pauli would use, WITHOUT collapsing ─────────────
// Decoder-feedback perf: the memoized Born-decision coin (fold_p1 pattern). This replicates the
// EXACT pp arithmetic of measure_pauli below — the anticommuting-active branch returns 0.5 (⟨P⟩=0
// exactly) and the commuting branch computes the identical exp_p = Re(c·⟨α,β⟩)/‖α‖² with the same
// clamp and pp = 0.5·(1+exp_p). Because the operation sequence and operands are identical to
// measure_pauli on the SAME (unmutated) state, the double is bit-for-bit equal to measure_pauli's
// internal pp — so `u < born_p1(P)` reproduces measure_pauli(P,u)'s ε=+1 (bit 0) decision exactly,
// with NO ULP window. const: reads the dual cache / amplitude container, mutates nothing (β is a
// fresh clone). Off the hot path (called ~once per distinct record on a memo miss).
double FramedSuperposition::born_p1(const Pauli& P) const {
    const int kk = k();
    const Pauli Q = U.dual_valid() ? U.conjugate(P) : U.dual_image(P);

    // Xa: X-support off the free set. Non-empty ⇒ anticommuting-active (measure_pauli case a) ⇒
    // ⟨P⟩=0 EXACTLY ⇒ pp = ½. Same pivot search measure_pauli uses (first active X = pivot).
    std::vector<uint8_t> is_free(n() > 0 ? (size_t)n() : 1, 0);
    for (int d = 0; d < kk; ++d) is_free[(size_t)free[d]] = 1;
    for (int b = 0; b < n(); ++b)
        if (Q.xbit(b) && !is_free[(size_t)b]) return 0.5;

    // Commuting (measure_pauli case b): decode (y, g, c) and exp_p — IDENTICAL to measure_pauli.
    std::vector<uint8_t> y(kk, 0);
    std::vector<cd> g(kk, cd(1.0, 0.0));
    for (int d = 0; d < kk; ++d) {
        y[d] = Q.xbit(free[d]) ? 1 : 0;
        if (Q.zbit(free[d])) g[d] = cd(-1.0, 0.0);
    }
    int par = 0;
    for (int a = 0; a < n(); ++a)
        if (Q.zbit(a) && eps[a]) par ^= 1;               // P0
    static const cd ipow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    cd c = ipow[Q.phase & 3];
    if (par & 1) c = -c;

    std::unique_ptr<Amplitudes> beta = alpha->clone();
    beta->apply_diagonal(cd(1.0, 0.0), g);
    beta->shift(y);
    const double nrm2 = alpha->norm2();
    double exp_p = 0.0;
    if (nrm2 >= 1e-30) exp_p = (c * alpha->inner(*beta)).real() / nrm2;
    if (exp_p < -1.0) exp_p = -1.0;
    if (exp_p > 1.0) exp_p = 1.0;
    return 0.5 * (1.0 + exp_p);
}

// ── measure_pauli — case (b): read COMMUTES with the active stabilizers ───────────────────────
// Design §3.2 case (b): Q = U†PU has NO X-support on the active (non-free) indices (Xa empty), so P
// is a container operator — the tableau (U, eps) is UNTOUCHED. In the amplitude representation
// P|φ⟩ = c · β with β = shift_y(diag_g(α)) and (c, y, g) decoded exactly as in `expectation`
// (Task 2): y[d]=Q.xbit(free[d]), g[d]=(-1)^{Q.zbit(free[d])}, c = i^{Q.phase}·(-1)^{P0}. The
// projector for outcome ε is Π_ε = (I + ε·P)/2, so α ← (α + ε·c·β)/2; the 1/2 (and the 1/√p_ε
// Born rescale) are absorbed by normalize(). Outcome convention matches
// CanonicalStabSum::measure_single: ε=+1 iff u < p_+ = (1+⟨P⟩)/2.
//
// The projection can make a free container index constant across the surviving support (the χ-drop):
// drop it and erase the matching `free` entry, keeping the free↔container-index correspondence.
int FramedSuperposition::measure_pauli(const Pauli& P, double u) {
    const int kk = k();
    const Pauli Q = U.dual_valid() ? U.conjugate(P) : U.dual_image(P);

    // Xa: X-support off the free set. Non-empty ⇒ case (a) (anticommuting active) — Task 5.
    static thread_local std::vector<uint8_t> is_free;
    is_free.assign(n(), 0);
    for (int d = 0; d < kk; ++d) is_free[free[d]] = 1;
    int pivot = -1;
    for (int b = 0; b < n(); ++b)
        if (Q.xbit(b) && !is_free[b]) { pivot = b; break; }   // first active anticommuter = pivot j

    // ── measure_pauli — case (a): read ANTICOMMUTES with an ACTIVE stabilizer (Xa ≠ ∅) ──────────
    // Design §3.2 case (a). A = { a : Q.xbit(a) } is the set of generators g_a = Zrow[a] that P
    // anticommutes with (Q = U†PU carries an X on a ⟺ P anticommutes with g_a). Since the pivot j
    // is ACTIVE (j∉free), every branch shares the same pivot generator sign (there is no free
    // coordinate on j) so ⟨P⟩ = 0 EXACTLY — the outcome is a FAIR COIN and α is left UNTOUCHED (no
    // branch merge; that is case (b)). This is the standard Aaronson–Gottesman anticommuting-
    // measurement collapse on the SIGNED tableau, done in DIRECT form (no CX pivot rotation).
    //
    // Sign bookkeeping. The reference state |ψ⟩ is fixed by the SIGNED stabilizers S_a =
    // (-1)^{eps[a]}·Zrow[a]. Destabilizers D_a = Xrow[a] carry their sign wholly in the Pauli phase
    // (no eps). We fold eps into the Pauli phase for every stabilizer we touch and re-emit eps=0 on
    // it (equivalent: (-1)^{eps}Zrow with eps=0). Steps (matches measure_single_anticommuting's
    // Step-4 patch, direct form):
    //   1. Let S_j = signed pivot stabilizer, D_j = Xrow[j] (destabiliser).
    //   2. For EVERY OTHER generator P anticommutes with — both stabilizers Zrow[b] (b∈A, b≠j) and
    //      destabilizers Xrow[b] anticommuting with P (incl. the distinguished d_i=Xrow[free[a]]:
    //      the Lemma cleanup d_i ← d_i·s_j) — multiply it by S_j. Afterwards ONLY S_j anticommutes
    //      with P.  (Multiplying by S_j = old g_j keeps the symplectic frame: S_j commutes with every
    //      other generator/destabiliser, so the product's commutation with the rest is preserved.)
    //   3. Fair coin: ε=+1 iff u<0.5 (⟨P⟩=0). Draw exactly as the oracle's anticommuting branch
    //      (pp=½ ⇒ m = (u<pp)?+1:-1) with the same u.
    //   4. New destabiliser D_j ← old S_j; new stabiliser S_j ← ε·P (eps[j]=0, sign in the phase).
    //   5. α untouched; free/k unchanged; U.invalidate_dual().
    if (pivot >= 0) {
        const int N = n();
        const int j = pivot;

        // S_j = signed pivot stabiliser (eps folded into the Pauli phase). new_dj becomes the new
        // destabiliser D_j after the collapse (= old signed generator g_j, which anticommutes with P).
        Pauli S_j = U.Zrow[j];
        if (eps[j]) S_j.phase = (S_j.phase + 2) & 3;
        Pauli new_dj = S_j;

        // Step 2 — clean every OTHER generator/destabiliser anticommuting with P by folding in S_j.
        // Stabilisers: b∈A (Q.xbit(b)) with b≠j. Fold eps[b] in, multiply, re-emit eps[b]=0.
        for (int b = 0; b < N; ++b) {
            if (b == j) continue;
            if (!Q.xbit(b)) continue;                 // Zrow[b] commutes with P
            Pauli s_b = U.Zrow[b];
            if (eps[b]) s_b.phase = (s_b.phase + 2) & 3;
            pmul_into(s_b, S_j);                       // s_b ← s_b · S_j (phase-exact)
            U.Zrow[b] = s_b;
            eps[b] = 0;                                // sign now carried in the phase
        }
        // Destabilisers Xrow[b] that anticommute with P (includes the distinguished d_{free[a]}).
        for (int b = 0; b < N; ++b) {
            if (b == j) continue;                      // Xrow[j] is overwritten below
            if (Pauli::anticommute_bit(U.Xrow[b], P) == 0) continue;
            pmul_into(U.Xrow[b], S_j);                 // d_b ← d_b · S_j (destabiliser, no eps)
        }

        // Step 3 — fair coin (⟨P⟩=0 ⇒ pp=½): ε=+1 iff u<0.5 (oracle's m=(u<pp)?+1:-1 with pp=½).
        const int eps_out = (u < 0.5) ? +1 : -1;

        // Step 4 — install the new pivot rows. New generator g_j = ε·P (sign in the phase, eps[j]=0);
        // new destabiliser d_j = old signed g_j.
        Pauli newpiv = P;
        if (eps_out == -1) newpiv.phase = (newpiv.phase + 2) & 3;
        U.Zrow[j] = newpiv;
        U.Xrow[j] = new_dj;
        eps[j] = 0;

        // α untouched; free/k unchanged. Forward rows were edited in place ⇒ dual is stale.
        U.invalidate_dual();
        return eps_out;
    }

    // Decode y (partner shift), g (diagonal signs) over the free set, and the global phase c —
    // identical to `expectation` (Task 2).
    std::vector<uint8_t> y(kk, 0);
    std::vector<cd> g(kk, cd(1.0, 0.0));
    for (int d = 0; d < kk; ++d) {
        y[d] = Q.xbit(free[d]) ? 1 : 0;
        if (Q.zbit(free[d])) g[d] = cd(-1.0, 0.0);
    }
    int par = 0;
    for (int a = 0; a < n(); ++a)
        if (Q.zbit(a) && eps[a]) par ^= 1;               // P0
    static const cd ipow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    cd c = ipow[Q.phase & 3];
    if (par & 1) c = -c;

    // ⟨P⟩ = Re(c · inner(α, β)) / ||α||²,  β = shift_y(diag_g(α)).
    std::unique_ptr<Amplitudes> beta = alpha->clone();
    beta->apply_diagonal(cd(1.0, 0.0), g);
    beta->shift(y);
    const double nrm2 = alpha->norm2();
    double exp_p = 0.0;
    if (nrm2 >= 1e-30) exp_p = (c * alpha->inner(*beta)).real() / nrm2;
    if (exp_p < -1.0) exp_p = -1.0;
    if (exp_p > 1.0) exp_p = 1.0;

    // Born: p_+ = (1+⟨P⟩)/2. ε=+1 iff u < p_+ (matches measure_single's `u < pp`).
    const double pp = 0.5 * (1.0 + exp_p);
    const int eps_out = (u < pp) ? +1 : -1;

    // Project: α ← α + ε·c·β, then normalize (folds the (I+εP)/2 factor and the 1/√p_ε).
    alpha->axpy(cd((double)eps_out, 0.0) * c, *beta);
    alpha->normalize();

    // χ-drop: any container index now constant across the surviving support folds out (its free
    // generator became a fixed sign, absorbed into the reference). Walk high→low so drop_index's
    // positional erase and the matching `free` erase stay aligned.
    for (int d = k() - 1; d >= 0; --d) {
        if (alpha->support() == 0) break;
        // Constancy test: compare bit d of the first key against the rest.
        const auto* dd = dynamic_cast<const DenseAmplitudes*>(alpha.get());
        if (!dd) throw std::runtime_error("measure_pauli: χ-drop needs DenseAmplitudes");
        if (dd->b_.empty()) break;
        const uint8_t ref_bit = dd->b_[0].first[d];
        bool constant = true;
        for (const auto& e : dd->b_) if (e.first[d] != ref_bit) { constant = false; break; }
        if (constant) {
            // A constant-1 index means the destabiliser d_{free[d]} multiplies EVERY branch: factor
            // it out. Since FramedSuperposition carries no explicit anchor, absorb it into the frame
            // reference by flipping eps[free[d]] (d_a anticommutes only with g_a, so the reference
            // d_a|ψ⟩ has that single sign flipped) — mirrors canonicalise's constant-1 fold. Then
            // drop the (now redundant) index. Constant-0 just drops.
            if (ref_bit) eps[free[d]] ^= 1;
            alpha->drop_index(d);
            free.erase(free.begin() + d);
        }
    }
    return eps_out;
}

// ── measure_pauli_batch: batched terminal measurement (backend-agnostic) ──────────────────────
// Pass 1 collapses the reads that couple the magic `free` directions via measure_pauli; Pass 2
// batch-conjugates the branch-agreeing remainder in ONE forward-row pass and reads it off with GF(2)
// coin bookkeeping (χ=1 reads — no container/amplitude work). Avoids the per-read O(n²) re-conjugation
// of measure_pauli for the (usually large) branch-agreeing majority. NOTE: this does NOT yet reduce χ
// via a merging collapse (measure_pauli case-a leaves χ untouched) nor apply the A/B active-block
// reduction the production sampler uses — so on codes whose magic operator couples MANY reads (e.g.
// cultivation d5) Pass 1 still collapses per-read. Matching the sampler fully needs those two pieces
// (the batch_measure merging collapse + framed_active_block A/B), i.e. the sampler migration.
void FramedSuperposition::measure_pauli_batch(const std::vector<std::pair<int, int>>& reads,
                                              const std::function<double()>& rng,
                                              std::vector<int>& out) {
    const int N = n();
    const int W = (N + 63) / 64;
    out.assign(reads.size(), +1);
    std::vector<char> done(reads.size(), 0);

    // Pass 1: collapse every read that couples the magic `free` directions (a read anticommuting a
    // free destabiliser interferes with the magic even when its marginal is ½, so it cannot be
    // deferred; the column test is the shared framed_read_couples_free). Re-scan after each
    // collapse (`free`/frame change); stop when the remainder is all branch-agreeing.
    bool progress = true;
    while (!free.empty() && progress) {
        progress = false;
        for (size_t kk = 0; kk < reads.size(); ++kk) {
            if (done[kk] || !framed_read_couples_free(*this, reads[kk].first, reads[kk].second)) continue;
            out[kk] = measure_pauli(single_pauli(reads[kk].first, reads[kk].second, N), rng());
            done[kk] = 1;
            progress = true;
            break;
        }
    }

    // Pass 2: the remaining reads are branch-agreeing → their outcome depends only on the frame
    // reference |ψ⟩ (α is irrelevant). Batch-conjugate them all in ONE forward-row pass, then read
    // off with GF(2) coin bookkeeping (the χ=1 batch — mirrors batch_chi1_pre above).
    std::vector<int> rem_idx;
    std::vector<Pauli> tg;
    for (size_t kk = 0; kk < reads.size(); ++kk)
        if (!done[kk]) { rem_idx.push_back((int)kk); tg.push_back(single_pauli(reads[kk].first, reads[kk].second, N)); }
    if (rem_idx.empty()) return;

    std::vector<const Pauli*> tp(rem_idx.size());
    for (size_t j = 0; j < tg.size(); ++j) tp[j] = &tg[j];
    std::vector<Pauli> Qc(rem_idx.size());
    dual_image_rows_scan(U.Xrow, U.Zrow, N, tp.data(), (int)rem_idx.size(), Qc.data());

    std::vector<uint64_t> eps_pk(W, 0);
    for (int a = 0; a < N; ++a) if (eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);

    struct Coin { Pauli Qcoin; int pivot; int o; };
    std::vector<Coin> coins;
    for (size_t j = 0; j < rem_idx.size(); ++j) {
        Pauli acc = Qc[j];
        int sign = 0;
        for (const auto& c : coins)
            if ((acc.x[c.pivot >> 6] >> (c.pivot & 63)) & 1ull) { pmul_into(acc, c.Qcoin); sign ^= c.o; }
        int xnz = -1;
        for (int w = 0; w < W; ++w) if (acc.x[w]) { xnz = (w << 6) + __builtin_ctzll(acc.x[w]); break; }
        int ob;
        if (xnz < 0) {   // determined by the reference stabilizers
            int zpar = 0;
            for (int w = 0; w < W; ++w) zpar += __builtin_popcountll(acc.z[w] & eps_pk[w]);
            ob = (((acc.phase & 3) == 2) ? 1 : 0) ^ (zpar & 1) ^ sign;
        } else {         // fresh fair coin
            int o = (rng() < 0.5) ? 0 : 1;
            ob = o ^ sign;
            coins.push_back({std::move(acc), xnz, o});
        }
        out[rem_idx[j]] = ob ? -1 : +1;
    }
}

}  // namespace qeccore
