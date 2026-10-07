// frame_cosets.hpp — the Born SECTOR table of an S-layer plan on a branch-frames reference,
// compiled.  Bit-for-bit the same table as xtim.frames.plan_frame_cosets_numpy (the pure
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
                              std::vector<int>& free_cols, std::vector<uint8_t>& lb) {
    const int m = (int)supp.size();
    if (m > 62) throw std::invalid_argument("lambda_functional: |supp| > 62 is not enumerable");
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
    std::vector<uint8_t> is_piv(m, 0);
    for (int c : piv) is_piv[c] = 1;
    for (int c = 0; c < m; ++c) if (!is_piv[c]) free_cols.push_back(c);
    std::vector<uint64_t> zero(f.W, 0ULL), wz(f.W);
    for (int fc : free_cols) {
        uint64_t w = 1ULL << fc;
        for (size_t i = 0; i < piv.size(); ++i) if ((M[i] >> fc) & 1ULL) w |= 1ULL << piv[i];
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
inline std::vector<Sector> plan_frame_cosets(const FrameRef& f, const uint8_t* a_mask) {
    const int n = f.n, k = f.k, chi = f.chi, W = f.W;
    std::vector<int> supp;
    for (int q = 0; q < n; ++q) if (a_mask[q] != 0) supp.push_back(q);   // numpy's nonzero()
    const int m = (int)supp.size();
    if (m > 30) throw std::invalid_argument("plan_frame_cosets: |supp a| > 30 is not enumerable");
    const uint64_t nv = 1ULL << m;

    // 1. coset keys, first-encounter order (ascending v) — the anticommutation pattern of Z^v with
    //    the n generators: key(v) = XOR_{t∈v} col[t], col[t] bit a = stab_x[a][supp[t]].
    std::vector<uint64_t> col((size_t)m * (size_t)W, 0ULL);
    for (int t = 0; t < m; ++t)
        for (int a = 0; a < n; ++a)
            if (FrameRef::bit(f.row(f.sx, a), supp[t])) col[(size_t)t * W + (a >> 6)] |= 1ULL << (a & 63);
    std::vector<uint64_t> keys((size_t)nv * (size_t)W, 0ULL);
    std::vector<int> cid(nv, 0);
    std::vector<uint64_t> reps;
    std::unordered_map<std::vector<uint64_t>, int, detail::KeyHash> seen;
    std::vector<uint64_t> kbuf(W);
    for (uint64_t v = 0; v < nv; ++v) {
        uint64_t* kv = keys.data() + (size_t)v * W;
        if (v) {
            const int t = __builtin_ctzll(v);
            const uint64_t* prev = keys.data() + (size_t)(v & (v - 1)) * W;
            for (int w = 0; w < W; ++w) kv[w] = prev[w] ^ col[(size_t)t * W + w];
        }
        kbuf.assign(kv, kv + W);
        auto it = seen.find(kbuf);
        if (it == seen.end()) {
            it = seen.emplace(kbuf, (int)reps.size()).first;
            reps.push_back(v);
        }
        cid[v] = it->second;
    }
    const int ncos = (int)reps.size();

    // 2. λ as a GF(2) functional on w = v ⊕ v_c; branch signs from bx restricted to supp.
    std::vector<int> free_cols; std::vector<uint8_t> lb;
    lambda_functional(f, supp, free_cols, lb);
    uint64_t lbmask = 0;
    for (size_t t = 0; t < free_cols.size(); ++t) if (lb[t]) lbmask |= 1ULL << free_cols[t];
    std::vector<uint64_t> bxs(chi, 0ULL);
    for (int i = 0; i < chi; ++i)
        for (int t = 0; t < m; ++t) if (FrameRef::bit(f.row(f.bx, i), supp[t])) bxs[i] |= 1ULL << t;

    // 3. A[c][i] accumulated in ascending v (numpy's add.at order); every term exact.
    std::vector<cd> A((size_t)ncos * (size_t)chi, cd(0.0, 0.0));
    static const cd AMP[4] = {cd(1, 0), cd(0, -1), cd(-1, 0), cd(0, 1)};   // (−i)^p
    for (uint64_t v = 0; v < nv; ++v) {
        const uint64_t w = v ^ reps[cid[v]];
        const double lam = (__builtin_parityll(w & lbmask) ? -1.0 : 1.0);
        const cd al = AMP[__builtin_popcountll(v) & 3] * lam;
        cd* Ac = A.data() + (size_t)cid[v] * chi;
        for (int i = 0; i < chi; ++i) {
            const double sgn = (__builtin_parityll(v & bxs[i]) ? -1.0 : 1.0);
            const cd t = detail::np_cmul(al * sgn, f.coeff[i]);
            Ac[i] = cd(Ac[i].real() + t.real(), Ac[i].imag() + t.imag());
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
    for (int c = 0; c < ncos; ++c) {
        const cd* Ac = A.data() + (size_t)c * chi;
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
