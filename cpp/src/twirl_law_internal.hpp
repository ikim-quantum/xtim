#pragma once
// Internal header of twirl_law.cpp (pattern: stab_affine_internal.hpp): the core entry (build_shot_law_core) for
// cpp/src/record_space_b.cpp and twirl_kernel_sampler.cpp. Hidden visibility: not exported from _xtim. (J-b2 J7b:
// the σ kernel section and its test seams are deleted.)
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qeccore/pauli_kernels.hpp"
#include "qeccore/twirl_kernel.hpp"

namespace qeccore {
namespace twirl_law_detail {

#if defined(__GNUC__) || defined(__clang__)
#define QECCORE_TWIRL_LAW_INTERNAL __attribute__((visibility("hidden")))
#else
#define QECCORE_TWIRL_LAW_INTERNAL
#endif

// J-a N6 (record-space B, switched OFF in production; cpp/src/record_space_b.cpp is the only caller):
// build_shot_law's own body run WITHOUT the κ>0 σ-fold stage (qrows, W = C_det^⊥, the ng-wide preimage
// solves, kernel_dual_basis and its asserts) and without kernel_base. The returned law carries tier1, r,
// κ, det_signs, coin_masks and kernel_logicals exactly as build_shot_law's; kernel_masks / kernel_pre /
// kernel_base / kernel_foldable are EMPTY. `core` receives the record-space inputs of the same build:
struct ShotLawCore {
    int n = 0, ng = 0;
    std::vector<int> aidx;                         // active generator indices, ascending
    std::vector<std::vector<uint64_t>> v;          // per active generator: its dressing v_i (n-bit)
    std::vector<uint8_t> phi;                      // per active generator: the phase of h_i (h-free form)
    std::vector<std::vector<uint64_t>> D;          // a basis of D = V ∩ G_z (n-bit)
    std::vector<std::vector<uint64_t>> U;          // the kernel directions u_j (n-bit); L_j = law.kernel_logicals[j]
};
// `teeth`: the caller's self-check window — the bounded exact re-verifications (Tier-1 planes vs tier1_signs,
// sign(h_i) == formula on touched-inactive generators) run only when set; outputs are identical either way.
QECCORE_TWIRL_LAW_INTERNAL ShotLaw build_shot_law_core(const CertifiedGroupPlanes& G, const DiagNormalForm& nf,
                                                       ShotLawCore& core, bool teeth);

// Support-directed reduction against the group's pure-Z RREF (2026-07-16; build_shot_law's reader of the G_z cache,
// inline here so cpp/tests/test_planes_sparse.cpp holds it against the pre-S2 dense-row reader): scan v's set bits
// ascending and jump via the column→row map instead of testing every rref row. Full reduced echelon: a pivot column
// occurs only in its own row, so no earlier pivot re-appears; XOR-only reduction => identical result to the row-walk
// overload. lean L4: `sign` (optional) receives the XOR of the signs of the rows applied — Z^{v} = (−1)^{sign} · Z^{v'} ·
// (group element) for the residual v'. S2-T10a: the rows are the cache's CSR (their set bits XORed one at a time).
inline void reduce_vec_gz_inplace(std::vector<uint64_t>& v, const CertifiedGroupPlanes::GroupZCache& GZ,
                                  uint8_t* sign = nullptr) {
    const int NW = (int)v.size();
    for (int wq = 0; wq < NW; ++wq) {
        uint64_t word = v[(size_t)wq];
        while (word) {
            const int b = __builtin_ctzll(word);
            const int q = wq * 64 + b;
            const int r = q < (int)GZ.gz_col_to_row.size() ? GZ.gz_col_to_row[(size_t)q] : -1;
            if (r >= 0) {
                for (int k = GZ.gz_off[(size_t)r]; k < GZ.gz_off[(size_t)r + 1]; ++k) {
                    const int32_t c = GZ.gz_idx[(size_t)k];
                    v[(size_t)(c >> 6)] ^= 1ULL << (c & 63);
                }
                if (sign) *sign ^= GZ.gz_sign[(size_t)r];
                word = (b == 63) ? 0 : (v[(size_t)wq] & (~0ULL << (b + 1)));
            } else {
                word &= word - 1;
            }
        }
    }
}

// S2-T10a fix 1: the G_z cache's elimination (CertifiedGroupPlanes::group_z_cache; the proof of its result is at
// build_group_z_cache, twirl_law.cpp). One forward elimination of the generators in order on lowest symplectic
// columns (X-block, then Z-block) — the same eliminations, products and pivots on either row form:
//   * sparse rows (SparsePauli; the product pmul_sparse) while the echelon stays sparse;
//   * packed words (x words then z words, 2·NW per row, and the phase; the product pmul_into's rule) once its rows
//     fill in: the switch (pure performance, identical rows) is taken between two generators when the echelon's
//     dense bytes are at most kGzDenseRowsRatio times its sparse entries' (16·NW·rows <= ratio·4·entries), and never
//     back. A forward echelon fills in where the planes' back-substituted RREF does not (deep random Clifford+T:
//     n' = 3003 took 2.6 s sparse, ~16 ms in words); the surface / chain families stay sparse (rows of weight ~3).
// switch_at: -1 = the automatic switch (production); k >= 0 = words from generator k on (0: words throughout,
// >= n_gens: sparse throughout) — the test's lever for both sides and a flip part-way (cpp/tests/test_planes_sparse.cpp).
inline constexpr long kGzDenseRowsRatio = 32;
inline void group_z_cache_build(const CertifiedGroupPlanes& P, CertifiedGroupPlanes::GroupZCache& out, long switch_at) {
    const int n = P.n_qubits, NW = (n + 63) / 64, W2 = 2 * NW;
    const size_t ng = P.gens.size();
    std::vector<int> piv_row((size_t)2 * (size_t)n, -1);         // symplectic column -> echelon row
    std::vector<SparsePauli> es;                                  // the echelon rows, sparse (lowest column = pivot)
    std::vector<uint64_t> ed;                                     // ... or packed: row r at ed[r*W2 .. (r+1)*W2)
    std::vector<int> eph;                                         //     with its phase eph[r]
    size_t rows = 0;
    long entries = 0;                                             // the sparse echelon's entries
    bool dense = false;
    std::vector<int> zpiv;                                        // p_k in order
    auto bad_herm = []() {
        throw std::logic_error("build_group_z_cache: a pure-Z combination is not a Hermitian Z-type element (engine bug)");
    };
    auto minus_i = []() { throw std::logic_error("build_group_z_cache: −I in the certified group (engine bug)"); };
    auto pack = [&](const SparsePauli& s, uint64_t* d) {
        std::fill(d, d + W2, 0ULL);
        for (int32_t q : s.x) d[q >> 6] |= 1ULL << (q & 63);
        for (int32_t q : s.z) d[NW + (q >> 6)] |= 1ULL << (q & 63);
    };
    auto to_words = [&]() {                                       // the switch: the sparse echelon packed, in order
        ed.assign(rows * (size_t)W2, 0ULL);
        eph.resize(rows);
        for (size_t r = 0; r < rows; ++r) { pack(es[r], ed.data() + r * (size_t)W2); eph[r] = es[r].phase; }
        std::vector<SparsePauli>().swap(es);
        dense = true;
    };
    SparsePauli w, t;
    std::vector<uint64_t> wd((size_t)W2);
    for (size_t i = 0; i < ng; ++i) {
        const SparsePauli& g = P.gens[i];
        if (!dense && (switch_at >= 0 ? (long)i >= switch_at
                                      : rows > 0 && 16L * NW * (long)rows <= kGzDenseRowsRatio * 4L * entries))
            to_words();
        if (!dense) {
            w = g;
            bool zstage = false;
            for (;;) {
                if (w.x.empty() && !zstage) { zstage = true; if (w.phase & 1) bad_herm(); }
                if (w.x.empty() && w.z.empty()) { if (w.phase) minus_i(); break; }
                const int c = w.x.empty() ? n + w.z[0] : w.x[0];
                const int r = piv_row[(size_t)c];
                if (r < 0) {
                    piv_row[(size_t)c] = (int)rows++;
                    if (c >= n) zpiv.push_back(c - n);
                    entries += (long)(w.x.size() + w.z.size());
                    es.push_back(std::move(w));
                    w = SparsePauli();
                    break;
                }
                pmul_sparse(w, es[(size_t)r], t);
                std::swap(w, t);
            }
            continue;
        }
        // packed words: the lowest set word from `wi` on (the eliminations only clear the lowest bit and set higher
        // ones: the pivot row's lowest column is the eliminated one)
        pack(g, wd.data());
        int phase = g.phase, wi = 0;
        bool zstage = false;
        for (;;) {
            while (wi < W2 && !wd[(size_t)wi]) ++wi;
            if (wi >= NW && !zstage) { zstage = true; if (phase & 1) bad_herm(); }
            if (wi >= W2) { if (phase) minus_i(); break; }
            const int b = __builtin_ctzll(wd[(size_t)wi]);
            const int c = wi < NW ? wi * 64 + b : n + (wi - NW) * 64 + b;
            const int r = piv_row[(size_t)c];
            if (r < 0) {
                piv_row[(size_t)c] = (int)rows++;
                if (c >= n) zpiv.push_back(c - n);
                ed.insert(ed.end(), wd.begin(), wd.end());
                eph.push_back(phase);
                break;
            }
            // w <- w · p (pmul_into's rule: the crossing term |w.z ∩ p.x|, then the XORs); p is zero below word wi
            const uint64_t* p = ed.data() + (size_t)r * (size_t)W2;
            int sign = 0;
            for (int k = wi; k < NW; ++k) sign ^= __builtin_popcountll(wd[(size_t)(NW + k)] & p[k]) & 1;
            phase = (phase + eph[(size_t)r] + 2 * sign) & 3;
            for (int k = wi; k < W2; ++k) wd[(size_t)k] ^= p[k];
        }
    }
    std::vector<SparsePauli>().swap(es);
    std::vector<uint64_t>().swap(ed);
    std::vector<int>().swap(piv_row);
    const size_t rk = zpiv.size();
    out.gz_off.assign(1, 0);
    out.gz_idx.clear();
    out.gz_sign.assign(rk, 0);
    for (size_t k = 0; k < rk; ++k) {
        const int r = P.col_to_row[(size_t)n + (size_t)zpiv[k]];
        if (r < 0 || !P.rref[(size_t)r].x.empty() || (P.rref[(size_t)r].phase & 1))
            throw std::logic_error("build_group_z_cache: the planes' RREF has no Hermitian Z row at a G_z pivot (engine bug)");
        const SparsePauli& R = P.rref[(size_t)r];
        out.gz_idx.insert(out.gz_idx.end(), R.z.begin(), R.z.end());
        out.gz_off.push_back((int)out.gz_idx.size());
        out.gz_sign[k] = (uint8_t)((R.phase >> 1) & 1);
    }
    out.gz_piv = std::move(zpiv);
    out.gz_col_to_row.assign((size_t)n, -1);
    for (size_t k = 0; k < rk; ++k) out.gz_col_to_row[(size_t)out.gz_piv[k]] = (int)k;
}

}  // namespace twirl_law_detail
}  // namespace qeccore
