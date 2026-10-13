#include "qeccore/ref_io.hpp"
#include <cstddef>

#include <algorithm>
#include <complex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qeccore {

namespace {

using cd = std::complex<double>;

}  // namespace

// ── exact sum-vs-sum overlap (library twin of protocol_reference's sum_overlap) ─────────

// Dense core: sum over the full chi_A x chi_B ray-overlap matrix. Takes already-materialized rays
// so the fast path can fall back without re-materializing.
static std::complex<double> dense_core(const CanonicalStabSum& A, const CanonicalStabSum& B,
                                       const std::vector<std::unique_ptr<AffineState>>& ra,
                                       const std::vector<std::unique_ptr<AffineState>>& rb) {
    cd tot(0, 0);
    for (size_t i = 0; i < ra.size(); ++i)
        for (size_t j = 0; j < rb.size(); ++j) {
            ExactPhase ov = ra[i]->inner_product(*rb[j]);
            if (!ov.is_zero)
                tot += std::conj(A.branches[i].c) * B.branches[j].c * ov.to_complex();
        }
    return tot;
}

std::complex<double> exact_sum_overlap_dense(const CanonicalStabSum& A, const CanonicalStabSum& B) {
    return dense_core(A, B, materialize_rays(A), materialize_rays(B));
}


// Canonical affine-support key of a stabilizer state's ray: the RREF basis of R's column space
// plus the offset b reduced modulo it. Equal affine supports (cosets) => identical key. O(n*k^2/64)
// bit-packed — replaces the O(n^3) signed-stabilizer-group fingerprint. NOTE: a support match is
// only a CANDIDATE (same support can carry different phase); exact_sum_overlap verifies each match
// with |ov|==1 and falls back to dense otherwise, so this key need not encode phase.
static std::string ray_support_fingerprint(const AffineState& ray) {
    const int n = ray.n_, k = ray.k_;
    const int W = (n + 63) / 64;
    // Extract columns of R as packed n-bit vectors.
    std::vector<std::vector<uint64_t>> col(k, std::vector<uint64_t>(W, 0));
    for (int j = 0; j < k; ++j)
        for (int q = 0; q < n; ++q)
            if (ray.R.get(q, j)) col[j][q >> 6] |= (1ull << (q & 63));
    // Extract offset b as a packed n-bit vector.
    std::vector<uint64_t> off(W, 0);
    for (int q = 0; q < n; ++q)
        if (ray.b[q]) off[q >> 6] |= (1ull << (q & 63));
    // Gauss-Jordan the k columns into a canonical RREF basis.
    // basis[p] = the basis vector with pivot at row pivots[p], reduced so only pivots[p] bit is set
    // among all pivot positions.
    std::vector<int> pivots;                                 // pivot row index, in order found
    std::vector<std::vector<uint64_t>> basis;                // RREF basis columns, same order as pivots
    pivots.reserve(k);
    basis.reserve(k);
    int rank = 0;
    for (int j = 0; j < k; ++j) {
        // Reduce column j by existing basis vectors (clear existing pivot bits).
        for (int p = 0; p < rank; ++p)
            if ((col[j][pivots[p] >> 6] >> (pivots[p] & 63)) & 1ull)
                for (int w = 0; w < W; ++w) col[j][w] ^= basis[p][w];
        // Find lowest set bit — the new pivot row.
        int piv = -1;
        for (int w = 0; w < W && piv < 0; ++w)
            if (col[j][w]) piv = (w << 6) + __builtin_ctzll(col[j][w]);
        if (piv < 0) continue;   // linearly dependent: drop
        // Back-eliminate this new pivot from all existing basis vectors and from off.
        for (int p = 0; p < rank; ++p)
            if ((basis[p][piv >> 6] >> (piv & 63)) & 1ull)
                for (int w = 0; w < W; ++w) basis[p][w] ^= col[j][w];
        if ((off[piv >> 6] >> (piv & 63)) & 1ull)
            for (int w = 0; w < W; ++w) off[w] ^= col[j][w];
        pivots.push_back(piv);
        basis.push_back(col[j]);
        ++rank;
    }
    // Serialize: rank, then each (pivot_row, basis_vector) pair in pivot order, then reduced off.
    // Sorting by pivot value ensures determinism regardless of column input order.
    // (RREF guarantees each basis vector already has a unique lowest bit = its pivot.)
    // Sort pivots+basis by ascending pivot index for canonical ordering.
    std::vector<int> order(rank);
    for (int p = 0; p < rank; ++p) order[p] = p;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return pivots[a] < pivots[b]; });
    std::string key;
    key.reserve(sizeof(int) + (size_t)rank * (sizeof(int) + (size_t)W * sizeof(uint64_t))
                             + (size_t)W * sizeof(uint64_t));
    auto push_u64 = [&](uint64_t v) { key.append(reinterpret_cast<const char*>(&v), 8); };
    auto push_i32 = [&](int v)      { key.append(reinterpret_cast<const char*>(&v), 4); };
    push_i32(rank);
    for (int p : order) {
        push_i32(pivots[p]);
        for (int w = 0; w < W; ++w) push_u64(basis[p][w]);
    }
    for (int w = 0; w < W; ++w) push_u64(off[w]);
    return key;
}

std::complex<double> exact_sum_overlap(const CanonicalStabSum& A, const CanonicalStabSum& B) {
    auto ra = materialize_rays(A);
    auto rb = materialize_rays(B);
    if (std::getenv("OVERLAP_FORCE_DENSE")) return dense_core(A, B, ra, rb);
    // Fast monomial path: match each A-ray to its unique B-ray by affine-support fingerprint.
    // Exact whenever it returns (matched rays must have |overlap|==1 verified below; equal A-ray
    // => all other B-overlaps vanish => the matched sum equals the full dense row sum); otherwise
    // falls back to dense. NOTE: support fingerprint is a CANDIDATE match only; the |ov|==1
    // check below is the safety guard for same-support-different-phase pairs.
    if (!ra.empty() && ra.size() == rb.size()) {
        std::unordered_map<std::string, int> idx;
        idx.reserve(rb.size() * 2);
        bool ok = true;
        for (int j = 0; j < (int)rb.size(); ++j)
            if (!idx.emplace(ray_support_fingerprint(*rb[j]), j).second) { ok = false; break; }  // dup support in B
        if (ok) {
            cd tot(0, 0);
            std::vector<char> usedB(rb.size(), 0);
            for (int i = 0; i < (int)ra.size(); ++i) {
                auto it = idx.find(ray_support_fingerprint(*ra[i]));
                if (it == idx.end() || usedB[it->second]) { ok = false; break; }
                const int j = it->second;
                usedB[j] = 1;
                ExactPhase ov = ra[i]->inner_product(*rb[j]);
                if (ov.is_zero || ov.scale != 0) { ok = false; break; }   // require |ov| == 1, else dense
                tot += std::conj(A.branches[i].c) * B.branches[j].c * ov.to_complex();
            }
            if (ok) return tot;
        }
    }
    return dense_core(A, B, ra, rb);
}

// Overlap A against an explicit ray-list Σ rb_coeffs[j]·rb[j] (the raw branch rays of an
// independent re-derivation) WITHOUT canonicalizing the latter into a CanonicalStabSum. The
// verification oracle only needs the physical superposition, so it skips from_rays' O(chi·n^3)
// compute_fsynd. Same fast fingerprint-match as exact_sum_overlap; dense double-sum fallback.
std::complex<double> exact_sum_overlap_rays(
        const CanonicalStabSum& A,
        const std::vector<std::unique_ptr<AffineState>>& rb,
        const std::vector<std::complex<double>>& rb_coeffs) {
    auto ra = materialize_rays(A);
    if (!std::getenv("OVERLAP_FORCE_DENSE") && !ra.empty() && ra.size() == rb.size()) {
        std::unordered_map<std::string, int> idx;
        idx.reserve(rb.size() * 2);
        bool ok = true;
        for (int j = 0; j < (int)rb.size(); ++j)
            if (!idx.emplace(ray_support_fingerprint(*rb[j]), j).second) { ok = false; break; }
        if (ok) {
            cd tot(0, 0);
            std::vector<char> usedB(rb.size(), 0);
            for (int i = 0; i < (int)ra.size(); ++i) {
                auto it = idx.find(ray_support_fingerprint(*ra[i]));
                if (it == idx.end() || usedB[it->second]) { ok = false; break; }
                const int j = it->second;
                usedB[j] = 1;
                ExactPhase ov = ra[i]->inner_product(*rb[j]);
                if (ov.is_zero || ov.scale != 0) { ok = false; break; }   // require |ov| == 1, else dense
                tot += std::conj(A.branches[i].c) * rb_coeffs[j] * ov.to_complex();
            }
            if (ok) return tot;
        }
    }
    cd tot(0, 0);                                          // dense fallback (O(chi^2))
    for (size_t i = 0; i < ra.size(); ++i)
        for (size_t j = 0; j < rb.size(); ++j) {
            ExactPhase ov = ra[i]->inner_product(*rb[j]);
            if (!ov.is_zero) tot += std::conj(A.branches[i].c) * rb_coeffs[j] * ov.to_complex();
        }
    return tot;
}

// ── framed-state overlap (Phase B1) ─────────────────────────────────────────────────────
//
// exact_sum_overlap for two FramedSuperpositions: materialize each side's rays via the
// frame-reconstructed anchor (materialize_rays(FramedSuperposition), ref_io.cpp) and feed
// the UNCHANGED AffineState::inner_product kernel — the same fast monomial fingerprint
// path + dense fallback as the CanonicalStabSum overload above, re-expressed over explicit
// (ray, coefficient) lists. The result carries an arbitrary GLOBAL phase (each anchor is
// reconstructed up to phase); consumers take std::abs.

// Dense core over explicit ray/coefficient lists (the framed twin of dense_core).
static std::complex<double> dense_core_lists(const std::vector<std::unique_ptr<AffineState>>& ra,
                                             const std::vector<cd>& ca,
                                             const std::vector<std::unique_ptr<AffineState>>& rb,
                                             const std::vector<cd>& cb) {
    cd tot(0, 0);
    for (size_t i = 0; i < ra.size(); ++i)
        for (size_t j = 0; j < rb.size(); ++j) {
            ExactPhase ov = ra[i]->inner_product(*rb[j]);
            if (!ov.is_zero) tot += std::conj(ca[i]) * cb[j] * ov.to_complex();
        }
    return tot;
}

std::complex<double> exact_sum_overlap(const FramedSuperposition& A, const FramedSuperposition& B) {
    auto ra = materialize_rays(A);
    auto rb = materialize_rays(B);
    std::vector<cd> ca, cb;
    ca.reserve(ra.size());
    cb.reserve(rb.size());
    for (const auto& e : A.entries()) ca.push_back(e.second);
    for (const auto& e : B.entries()) cb.push_back(e.second);
    if (!std::getenv("OVERLAP_FORCE_DENSE") && !ra.empty() && ra.size() == rb.size()) {
        // Fast monomial path (same contract as the CanonicalStabSum overload): support
        // fingerprints are CANDIDATE matches only; the |ov|==1 check guards same-support-
        // different-phase pairs, any miss falls back to the exact dense double sum.
        std::unordered_map<std::string, int> idx;
        idx.reserve(rb.size() * 2);
        bool ok = true;
        for (int j = 0; j < (int)rb.size(); ++j)
            if (!idx.emplace(ray_support_fingerprint(*rb[j]), j).second) { ok = false; break; }
        if (ok) {
            cd tot(0, 0);
            std::vector<char> usedB(rb.size(), 0);
            for (int i = 0; i < (int)ra.size(); ++i) {
                auto it = idx.find(ray_support_fingerprint(*ra[i]));
                if (it == idx.end() || usedB[it->second]) { ok = false; break; }
                const int j = it->second;
                usedB[j] = 1;
                ExactPhase ov = ra[i]->inner_product(*rb[j]);
                if (ov.is_zero || ov.scale != 0) { ok = false; break; }   // require |ov| == 1, else dense
                tot += std::conj(ca[i]) * cb[j] * ov.to_complex();
            }
            if (ok) return tot;
        }
    }
    return dense_core_lists(ra, ca, rb, cb);
}

}  // namespace qeccore
