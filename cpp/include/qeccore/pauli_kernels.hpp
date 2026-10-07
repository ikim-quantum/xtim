#pragma once
// Shared stateless GF(2)/Pauli leaf kernels for the stabilizer-superposition engines.
//
// These pure, allocation-conscious helpers were copy-pasted between the CanonicalStabSum oracle
// (canonical_stab_sum.cpp) and the FramedSuperposition fast path (framed_superposition.cpp) — verified byte-for-byte
// identical and lifted here so both call ONE source. They are stateless leaves operating on Pauli
// rows; the measurement BODIES and the rep structs (CanonicalStabSum vs FramedSuperposition) stay separate to
// preserve the deliberate oracle/fast-path duality.
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "qeccore/pauli.hpp"

namespace qeccore {

// In-place GF(2) row XOR: dst ^= src, word-packed (ceil(n/64) uint64_t words). dst and src must
// be the same length.
inline void xor_into(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src) {
    for (size_t w = 0; w < dst.size(); ++w) dst[w] ^= src[w];
}

// Single-qubit Pauli in the i^phase·X^x·Z^z convention (axis: 0→X, 1→Y(=i·XZ), 2→Z).
inline Pauli single_pauli(int pauli, int q, int n) {
    Pauli P(n);
    if (pauli == 0) { P.setx(q); }
    else if (pauli == 1) { P.setx(q); P.setz(q); P.phase = 1; }
    else if (pauli == 2) { P.setz(q); }
    else throw std::logic_error("pauli must be 0/1/2");
    return P;
}

// In-place acc ← acc · b (left operand = acc), bit-identical to acc = Pauli::multiply(acc, b) but
// reusing acc's storage. acc.phase, b.phase ∈ [0,3] and the crossing term is ≤2, so the sum is
// non-negative ⇒ `& 3` == the `((·%4)+4)%4` of multiply().
inline void pmul_into(Pauli& acc, const Pauli& b) {
    int sign = 0;
    for (size_t i = 0; i < acc.z.size(); ++i) sign += __builtin_popcountll(acc.z[i] & b.x[i]);
    acc.phase = (acc.phase + b.phase + 2 * (sign & 1)) & 3;
    for (size_t i = 0; i < acc.x.size(); ++i) { acc.x[i] ^= b.x[i]; acc.z[i] ^= b.z[i]; }
}

// Conjugate a Pauli M ← CX(c,t) M CX(c,t) in place (CX is self-inverse). CX rule:
// X_c→X_cX_t, Z_t→Z_cZ_t (others fixed); phase tracked exactly via the pmul_into chain.
inline void conj_cx_inplace(Pauli& M, int c, int t) {
    bool xc = M.xbit(c), zc = M.zbit(c), xt = M.xbit(t), zt = M.zbit(t);
    if (xc) M.flipx(c);
    if (zc) M.flipz(c);
    if (xt) M.flipx(t);
    if (zt) M.flipz(t);
    // Build img = ∏ correction factors and M ← M·img via pmul_into on reusable scratch (no per-factor
    // Pauli allocation). Same factor order ⇒ byte-identical to the Pauli::multiply chain.
    thread_local Pauli img, p;
    const int w = (M.n + 63) / 64;
    auto reset = [&](Pauli& q) {
        if ((int)q.x.size() != w) { q = Pauli(M.n); }
        else { std::fill(q.x.begin(), q.x.end(), 0); std::fill(q.z.begin(), q.z.end(), 0); q.phase = 0; q.n = M.n; }
    };
    reset(img);
    if (xc) { reset(p); p.setx(c); p.setx(t); pmul_into(img, p); }   // X_c X_t
    if (zc) { reset(p); p.setz(c);            pmul_into(img, p); }   // Z_c
    if (xt) { reset(p); p.setx(t);            pmul_into(img, p); }   // X_t
    if (zt) { reset(p); p.setz(c); p.setz(t); pmul_into(img, p); }   // Z_c Z_t
    pmul_into(M, img);
}

// Reset `p` to a fresh n-qubit identity Pauli (n=N, phase=0, x=z=W zero words) WITHOUT reallocating
// when `p` already owns the right-sized word vectors. Byte-identical to `p = Pauli(N)` downstream.
inline void reset_pauli_inplace(Pauli& p, int N) {
    const int W = (N + 63) / 64;
    if ((int)p.x.size() == W && (int)p.z.size() == W) {
        std::fill(p.x.begin(), p.x.end(), 0ull);
        std::fill(p.z.begin(), p.z.end(), 0ull);
        p.n = N; p.phase = 0;
    } else {
        p = Pauli(N);
    }
}

// BATCHED dual-image U† T U for `cnt` targets against the SAME 2N frame rows (Xrow[a]=UX_aU†,
// Zrow[a]=UZ_aU†) in ONE pass, row-scan flavour. Per target the factor visit order is b ascending,
// Zrow pass then Xrow pass, with the same Z-words+phase running-product accumulation. The image's
// x/z bits come from the anticommute pattern; the running product is needed only for its phase.
inline void dual_image_rows_scan(const std::vector<Pauli>& Xrow, const std::vector<Pauli>& Zrow,
                                 int N, const Pauli* const* T, int cnt, Pauli* out) {
    const int W = (N + 63) / 64;
    thread_local std::vector<uint64_t> az;        // cnt running products' Z-words, flat cnt×W (the
    az.assign((size_t)cnt * W, 0);                // crossing term reads only acc.z; acc.x is unused)
    thread_local std::vector<int> acc_phase;
    acc_phase.assign(cnt, 0);
    for (int t = 0; t < cnt; ++t) reset_pauli_inplace(out[t], N);
    for (int b = 0; b < N; ++b) {
        const Pauli& R = Zrow[b];
        const Pauli& P = Xrow[b];
        for (int t = 0; t < cnt; ++t) {
            const uint64_t* Tx = T[t]->x.data();
            const uint64_t* Tz = T[t]->z.data();
            int ac = 0;
            for (int w = 0; w < W; ++w) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
            if (ac & 1) {
                uint64_t* azt = az.data() + (size_t)t * W;
                int sign = 0; for (int w = 0; w < W; ++w) sign += __builtin_popcountll(azt[w] & P.x[w]);
                acc_phase[t] = (acc_phase[t] + P.phase + 2 * (sign & 1)) & 3;
                for (int w = 0; w < W; ++w) azt[w] ^= P.z[w];
                out[t].setx(b);
            }
        }
    }
    for (int b = 0; b < N; ++b) {
        const Pauli& R = Xrow[b];
        const Pauli& P = Zrow[b];
        for (int t = 0; t < cnt; ++t) {
            const uint64_t* Tx = T[t]->x.data();
            const uint64_t* Tz = T[t]->z.data();
            int ac = 0;
            for (int w = 0; w < W; ++w) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
            if (ac & 1) {
                uint64_t* azt = az.data() + (size_t)t * W;
                int sign = 0; for (int w = 0; w < W; ++w) sign += __builtin_popcountll(azt[w] & P.x[w]);
                acc_phase[t] = (acc_phase[t] + P.phase + 2 * (sign & 1)) & 3;
                for (int w = 0; w < W; ++w) azt[w] ^= P.z[w];
                out[t].setz(b);
            }
        }
    }
    for (int t = 0; t < cnt; ++t)
        out[t].phase = ((T[t]->phase - acc_phase[t]) % 4 + 4) % 4;
}

// Partner-lookup crossover χ for the Born pair-sums. Below it the O(χ²) scan wins (per-shot small-χ
// path); above it the O(χ) hash. Read ONCE on first use (function-local static — NO getenv in the
// per-shot hot loop); QEC_PARTNER_HASH_MIN overrides for A/B benchmarks.
inline int partner_hash_chi_min() {
    static const int v = [] {
        const char* e = std::getenv("QEC_PARTNER_HASH_MIN");
        return e ? std::atoi(e) : 64;
    }();
    return v;
}

}  // namespace qeccore
