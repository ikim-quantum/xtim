// frame_cosets.hpp — the Born SECTOR table of an S-layer plan on a branch-frames reference,
// compiled.  Since EXP-fix #5 (2026-10-21): the same sectors, order and structural fields as
// xtim.frames.plan_frame_cosets_numpy, with `weight` within 1e-12 relative (owner Q4) — the exact
// Gauss-sum coset sum replaced the 2^|supp a| walk (see plan_frame_cosets below).  Until then:
// bit-for-bit the same table as xtim.frames.plan_frame_cosets_numpy (the pure
// Python production path of 3.1.2, itself pinned field-by-field to the reference oracle
// xtim.frames.plan_frame_cosets): same sectors, same first-encounter order, same fields,
// same float `weight` (docs/frame_cosets_cpp.md).
//
//     S^a |T> ∝ Σ_{v ⊆ supp a} (−i)^{|v|} Z^v |T>,   |T> = Σ_i c_i D_i |r>,
//     Z^v D_i |r> = (−1)^{<Z^v,D_i>} λ_{v,c} D_i Z^{v_c} |r>
//
// A_{c,i} = c_i Σ_{v∈c} (−i)^{|v|} (−1)^{<Z^v,D_i>} λ_{v,c}; sector weight Σ_i |A_{c,i}|²;
// frame-copy test over the 2^{2k} logical images D^x G^z of the free rows.
//
// FLOAT DISCIPLINE (why `weight` can be pinned bit-for-bit against the Python bodies): every
// term is c_i times a unit in {±1, ±i} — exact — so A is a fixed-order sum of exact values;
// the weight is Σ_i (re_i·re_i + im_i·im_i) accumulated left to right, the normalisation a
// left-to-right sum and one division — IEEE basic operations in a fixed order, the SAME
// statements the reference and numpy bodies execute in Python (which never contracts), and
// no library kernel (no abs/hypot/sqrt, no pairwise or compensated sum) on the pinned
// value.  The complex DIVISION (Smith's method, as numpy) and the proportionality residual
// feed a tolerance test only; everything here is compiled without FMA contraction so the
// compare is the same arithmetic on every build.
//
// Errors: std::invalid_argument = the Python ValueError of lambda_functional (a null-space
// basis read that is not ±1); FrameNotFull = the AssertionError of ref_expectation (P
// commutes with every generator but is not their product).  Both mean "the export is not
// a full frame"; neither is reachable from a branch_frames() export of the engine.
#pragma once
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#ifdef _MSC_VER
#pragma fp_contract(off)   // MSVC ignores the GCC pragmas; same intent (re·re + im·im unfused)
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/gf2_gauss.hpp"
#include "qeccore/exact_phase.hpp"
#include "qeccore/state_budget.hpp"

namespace qeccore {
namespace frame_cosets {

using cd = std::complex<double>;

struct FrameNotFull : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The branch_frames() export, bit-packed: rows of W = ceil(n/64) words.
struct FrameRef {
    int n = 0, k = 0, chi = 0, W = 0;
    std::vector<uint64_t> sx, sz, dx, dz;   // n rows × W words (generators / destabilizers)
    std::vector<int8_t> sp;                 // stab_phase[a] (mod 4)
    std::vector<int> free_rows;             // k
    std::vector<uint8_t> sigma;             // chi × k
    std::vector<cd> coeff;                  // chi
    std::vector<uint64_t> bx, bz;           // chi rows × W (branch Paulis D_i)

    static int words(int n) { return (n + 63) / 64; }
    const uint64_t* row(const std::vector<uint64_t>& m, int r) const { return m.data() + (size_t)r * (size_t)W; }
    uint64_t* row(std::vector<uint64_t>& m, int r) { return m.data() + (size_t)r * (size_t)W; }
    static bool bit(const uint64_t* r, int q) { return (r[q >> 6] >> (q & 63)) & 1ULL; }
    static void set(uint64_t* r, int q) { r[q >> 6] |= 1ULL << (q & 63); }

    // Build from unpacked uint8 bit rows (row-major n×n / chi×n) — the numpy layout.
    static FrameRef from_bits(int n, const uint8_t* stab_x, const uint8_t* stab_z, const int8_t* stab_phase,
                              const uint8_t* destab_x, const uint8_t* destab_z,
                              const int* free_rows, int k, const uint8_t* sigma, const cd* coeff, int chi,
                              const uint8_t* branch_x, const uint8_t* branch_z) {
        FrameRef f;
        f.n = n; f.k = k; f.chi = chi; f.W = words(n);
        auto pack = [&](std::vector<uint64_t>& dst, const uint8_t* src, int rows) {
            dst.assign((size_t)rows * (size_t)f.W, 0ULL);
            for (int r = 0; r < rows; ++r)
                for (int q = 0; q < n; ++q)
                    if (src[(size_t)r * (size_t)n + (size_t)q] & 1) set(f.row(dst, r), q);
        };
        pack(f.sx, stab_x, n); pack(f.sz, stab_z, n); pack(f.dx, destab_x, n); pack(f.dz, destab_z, n);
        pack(f.bx, branch_x, chi); pack(f.bz, branch_z, chi);
        f.sp.assign(stab_phase, stab_phase + n);
        f.free_rows.assign(free_rows, free_rows + k);
        f.sigma.assign(sigma, sigma + (size_t)chi * (size_t)k);
        f.coeff.assign(coeff, coeff + chi);
        return f;
    }
};

struct Sector {
    std::vector<uint8_t> vz, fx, fz;        // n bits each (unpacked, the numpy layout)
    bool frame_copy = false;
    bool has_flip = false;                  // logical_flip is None when false
    std::vector<int> xb, zb;                // logical_flip = (xb, zb) over the free rows
    double weight = 0.0;
};

namespace detail {

inline int parity_and(const uint64_t* a, const uint64_t* b, int W) {
    uint64_t acc = 0;
    for (int w = 0; w < W; ++w) acc ^= a[w] & b[w];
    return __builtin_parityll(acc);
}
inline int anticommute(const uint64_t* ax, const uint64_t* az, const uint64_t* bx, const uint64_t* bz, int W) {
    return parity_and(ax, bz, W) ^ parity_and(az, bx, W);
}

// <r| i^pp X^px Z^pz |r> as m in {0,1,2,3} (value i^m) or -1 for 0 — xtim.frames.ref_expectation.
inline int ref_expectation_m(const FrameRef& f, const uint64_t* px, const uint64_t* pz, int pp) {
    const int W = f.W;
    std::vector<uint64_t> rx(W, 0ULL), rz(W, 0ULL);
    int rp = 0;
    for (int a = 0; a < f.n; ++a) {
        if (anticommute(px, pz, f.row(f.sx, a), f.row(f.sz, a), W)) return -1;
        if (anticommute(px, pz, f.row(f.dx, a), f.row(f.dz, a), W)) {
            // R ← R · G_a : phase += G.phase + 2·|R.z ∧ G.x|
            const uint64_t* gx = f.row(f.sx, a); const uint64_t* gz = f.row(f.sz, a);
            int sgn = parity_and(rz.data(), gx, W);
            rp = (rp + (int)f.sp[a] + 2 * sgn) & 3;
            for (int w = 0; w < W; ++w) { rx[w] ^= gx[w]; rz[w] ^= gz[w]; }
        }
    }
    for (int w = 0; w < W; ++w)
        if (rx[w] != px[w] || rz[w] != pz[w])
            throw FrameNotFull("branch_frames: P commutes with every generator but is not their "
                               "product — the export is not a full frame");
    return (pp - rp) & 3;
}

// The pinned sector weight Σ_i |A_i|² as the Python bodies compute it: per branch
// re·re + im·im (two rounded products, one rounded add — never fused: this header is
// compiled with fp-contract=off), accumulated left to right from 0.0.
inline double sector_weight(const cd* A, int chi) {
    double w = 0.0;
    for (int i = 0; i < chi; ++i) {
        const double re = A[i].real(), im = A[i].imag();
        w += re * re + im * im;
    }
    return w;
}

// numpy's complex division (Smith's method, npy_math_internal / loops).
inline cd np_cdiv(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    } else {
        const double rat = in2r / in2i;
        const double scl = 1.0 / (in2i + in2r * rat);
        return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
    }
}
inline cd np_cmul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
// Complex modulus for the TOLERANCE gates only (1e-12 / 1e-9 thresholds; inputs are ~1e-16
// or O(1) on every fixture).  Not on the pinned `weight`, so an ulp of libm/numpy
// difference is invisible.
inline double cabs(cd a) { return std::hypot(a.real(), a.imag()); }

struct KeyHash {
    size_t operator()(const std::vector<uint64_t>& v) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        for (uint64_t w : v) { h ^= w; h *= 1099511628211ULL; h ^= h >> 29; }
        return (size_t)h;
    }
};

}  // namespace detail

// xtim.frames.lambda_functional: (free_cols, lb) with <r|Z^w|r> = (−1)^{w[free_cols]·lb} on the
// null space of M = stab_x[:, supp].  `supp` in ascending wire order; m = |supp| ≤ 62.
inline void lambda_functional(const FrameRef& f, const std::vector<int>& supp,
                              std::vector<int>& free_cols, std::vector<uint8_t>& lb,
                              std::vector<uint64_t>* kernel = nullptr) {
    const int m = (int)supp.size();
    if (m > 64) throw std::invalid_argument("lambda_functional: |supp| > 64 is not supported by the 64-bit masks");
    // M rows as m-bit masks: bit t of row a = stab_x[a][supp[t]]
    std::vector<uint64_t> M((size_t)f.n, 0ULL);
    for (int a = 0; a < f.n; ++a) {
        const uint64_t* r = f.row(f.sx, a);
        uint64_t row = 0;
        for (int t = 0; t < m; ++t) if (FrameRef::bit(r, supp[t])) row |= 1ULL << t;
        M[a] = row;
    }
    std::vector<int> piv;
    int r = 0;
    for (int c = 0; c < m && r < f.n; ++c) {
        int i = -1;
        for (int a = r; a < f.n; ++a) if ((M[a] >> c) & 1ULL) { i = a; break; }
        if (i < 0) continue;
        if (i != r) std::swap(M[i], M[r]);
        for (int a = 0; a < f.n; ++a) if (a != r && ((M[a] >> c) & 1ULL)) M[a] ^= M[r];
        piv.push_back(c);
        ++r;
    }
    free_cols.clear(); lb.clear();
    if (kernel) kernel->clear();
    std::vector<uint8_t> is_piv(m, 0);
    for (int c : piv) is_piv[c] = 1;
    for (int c = 0; c < m; ++c) if (!is_piv[c]) free_cols.push_back(c);
    std::vector<uint64_t> zero(f.W, 0ULL), wz(f.W);
    for (int fc : free_cols) {
        uint64_t w = 1ULL << fc;
        for (size_t i = 0; i < piv.size(); ++i) if ((M[i] >> fc) & 1ULL) w |= 1ULL << piv[i];
        if (kernel) kernel->push_back(w);                // the null-space basis of M (one per free column)
        std::fill(wz.begin(), wz.end(), 0ULL);
        for (int t = 0; t < m; ++t) if ((w >> t) & 1ULL) FrameRef::set(wz.data(), supp[t]);
        const int mm = detail::ref_expectation_m(f, zero.data(), wz.data(), 0);
        if (mm != 0 && mm != 2)
            throw std::invalid_argument(
                "lambda_functional: <r|Z^w|r> = " + std::string(mm < 0 ? "0j" : (mm == 1 ? "1j" : "(-0-1j)")) +
                " is not ±1 for a null-space basis vector — the export is not a full frame");
        lb.push_back(mm == 2 ? 1 : 0);
    }
}

// The sector table of the S-layer plan `a_mask` (n bits) on the reference `f`.
//
// EXP-fix #5: 2^rank cosets, never 2^|supp a| masks. With col[t] the anticommutation column of
// Z_{supp[t]} (r = rank, K = ker col, d = m - r):
//   1. the cosets of K are enumerated directly: K's basis in reduced echelon form with highest-bit
//      pivots makes the integer-minimal coset member the unique one with every pivot bit clear, so
//      the representatives (= the first-encounter order of the 2^m walk) are the subsets of the m - d
//      non-pivot positions in increasing order;
//   2. A[c][i] = c_i (−i)^{|r|} (−1)^{r·bx_i} Σ_{x∈F_2^d} i^{E(x)} (u = Σ x_j k_j over K's basis), with
//      E(x) = Σ_j x_j (−|k_j| + 2 k_j·(r⊕lb⊕bx_i)) + 2 Σ_{j<l} x_j x_l |k_j∧k_l|  (mod 4):
//      the exact Z4 Gauss sum (gf2_gauss.hpp), memoised per (k_j·s)_j.
// `weight` is therefore no longer the fixed-order float sum of the 2^m-term oracle: it matches
// plan_frame_cosets_numpy to 1e-12 relative (owner Q4, 2026-10-21); every structural field is exact.
inline std::vector<Sector> plan_frame_cosets(const FrameRef& f, const uint8_t* a_mask) {
    const int n = f.n, k = f.k, chi = f.chi, W = f.W;
    std::vector<int> supp;
    for (int q = 0; q < n; ++q) if (a_mask[q] != 0) supp.push_back(q);   // numpy's nonzero()
    const int m = (int)supp.size();
    if (m > 64)
        throw std::invalid_argument("plan_frame_cosets: |supp a| = " + std::to_string(m) +
                                    " > 64 is not supported by the 64-bit coset masks");
    const uint64_t full = m == 64 ? ~0ULL : ((1ULL << m) - 1ULL);

    // K = ker(col) as m-bit vectors: lambda_functional's own null-space basis of M = stab_x[:, supp] (one
    // elimination per plan, shared with lambda), re-reduced to highest-bit pivots below.
    std::vector<uint64_t> kb;
    std::vector<int> free_cols; std::vector<uint8_t> lb;
    lambda_functional(f, supp, free_cols, lb, &kb);
    {
        // highest-bit reduced echelon form
        for (size_t i = 0; i < kb.size(); ++i) {
            size_t best = i;
            for (size_t j = i + 1; j < kb.size(); ++j) if (kb[j] > kb[best]) best = j;
            std::swap(kb[i], kb[best]);
            const int h = 63 - __builtin_clzll(kb[i]);
            for (size_t j = 0; j < kb.size(); ++j)
                if (j != i && ((kb[j] >> h) & 1ULL)) kb[j] ^= kb[i];
        }
    }
    const int d = (int)kb.size();
    uint64_t pivmask = 0;
    for (uint64_t w : kb) pivmask |= 1ULL << (63 - __builtin_clzll(w));
    const uint64_t NP = full & ~pivmask;                 // non-pivot positions
    const int r = m - d;
    if (r > 62)
        throw std::invalid_argument("plan_frame_cosets: the plan's coset group has rank " + std::to_string(r) +
                                    " > 62 (|supp a| = " + std::to_string(m) + "); 2^rank sectors are not enumerable");
    const uint64_t ncos64 = 1ULL << r;
    // Budget (refused by name before any allocation): the A table, the representatives and the Gauss-sum
    // memo (at most one entry per (coset, branch)).
    StateBudgetTicket ticket;
    {
        const long double bytes = (long double)ncos64 * ((long double)chi * 16.0L + 8.0L) +
                                  std::min((long double)ncos64 * chi, std::ldexp(1.0L, std::min(d, 62))) * 64.0L;
        const int64_t b = bytes > 4.0e18L ? (int64_t)4000000000000000000LL : (int64_t)bytes;
        if (live_state_bytes() + b > state_budget_bytes()) {     // the refusal names the plan (built only then)
            const std::string what = "plan_frame_cosets (2^" + std::to_string(r) + " sectors: rank r = " +
                                     std::to_string(r) + ", |supp a| = " + std::to_string(m) + ", chi = " +
                                     std::to_string(chi) + ")";
            ticket.charge(b, what.c_str());
        } else {
            ticket.charge(b, "plan_frame_cosets");
        }
    }
    const size_t ncos = (size_t)ncos64;
    std::vector<uint64_t> reps((size_t)ncos);
    {
        uint64_t cur = 0;
        for (size_t c = 0; c < ncos; ++c) { reps[c] = cur; cur = ((cur | ~NP) + 1ULL) & NP; }
    }

    // λ as a GF(2) functional on K; branch signs from bx restricted to supp.
    uint64_t lbmask = 0;
    for (size_t t = 0; t < free_cols.size(); ++t) if (lb[t]) lbmask |= 1ULL << free_cols[t];
    std::vector<uint64_t> bxs(chi, 0ULL);
    for (int i = 0; i < chi; ++i)
        for (int t = 0; t < m; ++t) if (FrameRef::bit(f.row(f.bx, i), supp[t])) bxs[i] |= 1ULL << t;
    auto tvec = [&](uint64_t s) {                        // (k_j·s)_j as a d-bit mask
        uint64_t t = 0;
        for (int j = 0; j < d; ++j) if (__builtin_parityll(kb[(size_t)j] & s)) t |= 1ULL << j;
        return t;
    };
    std::vector<int> wk((size_t)d);
    std::vector<std::vector<uint8_t>> Kq((size_t)d, std::vector<uint8_t>((size_t)d, 0));
    for (int j = 0; j < d; ++j) {
        wk[(size_t)j] = __builtin_popcountll(kb[(size_t)j]);
        for (int l = 0; l < d; ++l)
            if (l != j) Kq[(size_t)j][(size_t)l] = (uint8_t)(__builtin_popcountll(kb[(size_t)j] & kb[(size_t)l]) & 1);
    }
    // G(t) = sum_x i^{Q(x)} (-1)^{t.x}, Q(x) = sum_j x_j (-|k_j|) + 2 sum_{j<l} K_jl x_j x_l (mod 4): the
    // Walsh-Hadamard transform of i^{Q(x)}. While 2^d is small (the u2-class plans) the WHOLE table is one
    // exact integer FWHT (2^d d operations, no allocation per entry); otherwise a per-t gauss_sum, memoised.
    const bool dense = d <= 12;
    std::vector<cd> gdense;
    if (dense) {
        const size_t G = (size_t)1 << d;
        std::vector<int64_t> re(G, 0), im(G, 0);
        std::vector<uint64_t> krow((size_t)d, 0);
        for (int j = 0; j < d; ++j)
            for (int l = 0; l < d; ++l) if (Kq[(size_t)j][(size_t)l]) krow[(size_t)j] |= 1ULL << l;
        static const int RE4[4] = {1, 0, -1, 0}, IM4[4] = {0, 1, 0, -1};   // i^q
        uint64_t x = 0;
        int q = 0;
        re[0] = 1;
        for (size_t g = 1; g < G; ++g) {                 // Gray walk: toggle x_j, update Q incrementally
            const int j = __builtin_ctzll(g);
            const int s2 = 2 * (__builtin_popcountll(krow[(size_t)j] & x) & 1);
            q = ((x >> j) & 1ULL) ? (q + wk[(size_t)j] + s2) & 3 : (q - wk[(size_t)j] + s2 + 4) & 3;
            x ^= 1ULL << j;
            re[(size_t)x] = RE4[q];
            im[(size_t)x] = IM4[q];
        }
        for (size_t h = 1; h < G; h <<= 1)
            for (size_t i = 0; i < G; i += 2 * h)
                for (size_t jj = i; jj < i + h; ++jj) {
                    const int64_t a = re[jj], b = re[jj + h], c = im[jj], e = im[jj + h];
                    re[jj] = a + b; re[jj + h] = a - b; im[jj] = c + e; im[jj + h] = c - e;
                }
        gdense.resize(G);
        for (size_t t = 0; t < G; ++t) gdense[t] = cd((double)re[t], (double)im[t]);
    }
    std::unordered_map<uint64_t, cd> gmemo;
    auto gauss = [&](uint64_t t) -> cd {                 // Σ_x i^{E(x)}, exact (a Gaussian integer)
        if (dense) return gdense[(size_t)t];
        auto it = gmemo.find(t);
        if (it != gmemo.end()) return it->second;
        std::vector<int> L((size_t)d);
        for (int j = 0; j < d; ++j) L[(size_t)j] = ((-wk[(size_t)j] + 2 * (int)((t >> j) & 1ULL)) % 4 + 4) % 4;
        const ExactPhase g = gauss_sum(d, 0, std::move(L), Kq);
        cd v(0.0, 0.0);
        if (!g.is_zero) {
            const int z = g.z8;
            if ((g.scale & 1) == 0) {                    // 2^{s/2} i^{z/2}
                const double mag = std::ldexp(1.0, g.scale / 2);
                static const cd I4[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
                v = I4[(z / 2) & 3] * mag;
            } else {                                     // 2^{(s-1)/2} (1+i) i^{(z-1)/2}
                const double mag = std::ldexp(1.0, (g.scale - 1) / 2);
                static const cd J4[4] = {cd(1, 1), cd(-1, 1), cd(-1, -1), cd(1, -1)};
                v = J4[((z - 1) / 2) & 3] * mag;
            }
        }
        gmemo.emplace(t, v);
        return v;
    };
    const uint64_t t_lb = tvec(lbmask);
    std::vector<uint64_t> t_bx((size_t)chi);
    for (int i = 0; i < chi; ++i) t_bx[(size_t)i] = tvec(bxs[(size_t)i]);

    // 3. A[c][i] in closed form.
    std::vector<cd> A(ncos * (size_t)chi, cd(0.0, 0.0));
    static const cd AMP[4] = {cd(1, 0), cd(0, -1), cd(-1, 0), cd(0, 1)};   // (−i)^p
    for (size_t c = 0; c < ncos; ++c) {
        const uint64_t rc = reps[c];
        const cd al = AMP[__builtin_popcountll(rc) & 3];
        const uint64_t t_rc = tvec(rc) ^ t_lb;
        cd* Ac = A.data() + (size_t)c * chi;
        for (int i = 0; i < chi; ++i) {
            const cd g = gauss(t_rc ^ t_bx[(size_t)i]);
            const double sgn = (__builtin_parityll(rc & bxs[i]) ? -1.0 : 1.0);
            Ac[i] = detail::np_cmul(al * g * sgn, f.coeff[i]);
        }
    }

    // 4. per sector: weight, frame-copy search over D^x G^z (the reference's loop, verbatim).
    std::unordered_map<uint64_t, int> idx_of;        // sigma_i (k-bit mask) -> i
    std::vector<uint64_t> sig(chi, 0ULL);
    for (int i = 0; i < chi; ++i) {
        uint64_t s = 0;
        for (int t = 0; t < k; ++t) if (f.sigma[(size_t)i * k + t]) s |= 1ULL << t;
        sig[i] = s; idx_of[s] = i;
    }
    std::vector<Sector> out;
    std::vector<cd> B(chi), cprime(chi);
    std::vector<uint64_t> vzw(W), gx(W), gz(W);
    for (size_t c = 0; c < ncos; ++c) {
        const cd* Ac = A.data() + c * (size_t)chi;
        const double w = detail::sector_weight(Ac, chi);
        if (w < 1e-12) continue;
        Sector s;
        s.vz.assign(n, 0); s.fx.assign(n, 0);
        std::fill(vzw.begin(), vzw.end(), 0ULL);
        for (int t = 0; t < m; ++t) if ((reps[c] >> t) & 1ULL) { s.vz[supp[t]] = 1; FrameRef::set(vzw.data(), supp[t]); }
        for (int i = 0; i < chi; ++i) {
            const double sg = detail::parity_and(vzw.data(), f.row(f.bx, i), W) ? -1.0 : 1.0;
            B[i] = Ac[i] * sg;
        }
        s.fz = s.vz;
        s.weight = w;
        const uint64_t ncode = 1ULL << (2 * k);
        for (uint64_t code = 0; code < ncode; ++code) {
            const uint64_t xb = code & ((1ULL << k) - 1ULL);
            const uint64_t zb = code >> k;
            std::fill(cprime.begin(), cprime.end(), cd(0.0, 0.0));
            bool ok = true;
            for (int i = 0; i < chi; ++i) {
                auto it = idx_of.find(sig[i] ^ xb);
                if (it == idx_of.end()) { ok = false; break; }
                const double sg = (__builtin_parityll(sig[i] & zb) ? -1.0 : 1.0);
                const cd t = f.coeff[i] * sg;
                cd& cp = cprime[it->second];
                cp = cd(cp.real() + t.real(), cp.imag() + t.imag());
            }
            if (!ok) continue;
            int nz0 = -1; bool bad = false;
            for (int i = 0; i < chi; ++i) {
                const double a = detail::cabs(cprime[i]);
                if (a > 1e-12) { if (nz0 < 0) nz0 = i; }
                else if (detail::cabs(B[i]) > 1e-9) { bad = true; }
            }
            if (nz0 < 0 || bad) continue;
            const cd mu = detail::np_cdiv(B[nz0], cprime[nz0]);
            if (detail::cabs(mu) < 1e-12) continue;
            double mx = 0.0;
            for (int i = 0; i < chi; ++i) {
                const cd d = B[i] - detail::np_cmul(mu, cprime[i]);
                mx = std::max(mx, detail::cabs(d));
            }
            if (mx > 1e-9) continue;
            s.frame_copy = true; s.has_flip = true;
            s.xb.resize(k); s.zb.resize(k);
            std::fill(gx.begin(), gx.end(), 0ULL); std::fill(gz.begin(), gz.end(), 0ULL);
            for (int t = 0; t < k; ++t) {
                const int fr = f.free_rows[t];
                s.xb[t] = (int)((xb >> t) & 1ULL); s.zb[t] = (int)((zb >> t) & 1ULL);
                if (s.xb[t]) for (int w2 = 0; w2 < W; ++w2) { gx[w2] ^= f.row(f.dx, fr)[w2]; gz[w2] ^= f.row(f.dz, fr)[w2]; }
                if (s.zb[t]) for (int w2 = 0; w2 < W; ++w2) { gx[w2] ^= f.row(f.sx, fr)[w2]; gz[w2] ^= f.row(f.sz, fr)[w2]; }
            }
            for (int q = 0; q < n; ++q) {
                s.fx[q] = FrameRef::bit(gx.data(), q) ? 1 : 0;
                s.fz[q] = s.vz[q] ^ (FrameRef::bit(gz.data(), q) ? 1 : 0);
            }
            break;
        }
        out.push_back(std::move(s));
    }
    // normalisation: an explicit left-to-right sum from 0.0 (the Python bodies' loop, NOT the
    // builtin sum(), whose float algorithm changed between CPython 3.11 and 3.12), then one
    // division per sector.
    double tot = 0.0;
    for (const Sector& s : out) tot += s.weight;
    for (Sector& s : out) s.weight /= tot;
    return out;
}

}  // namespace frame_cosets
}  // namespace qeccore

#pragma GCC pop_options
