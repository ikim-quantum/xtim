// plan_class — the K2' kernel elimination (qeccore/plan_class.hpp).
#include "qeccore/plan_class.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace qeccore {

namespace {

inline bool bit_(const uint64_t* w, size_t i) { return (w[i >> 6] >> (i & 63)) & 1ULL; }
inline void set_(uint64_t* w, size_t i) { w[i >> 6] |= 1ULL << (i & 63); }

// The bits of src in [src_off, src_off + nbits) XORed into dst at bit dst_off.
void xor_bits_from_(uint64_t* dst, size_t dst_off, const uint64_t* src, size_t src_off, size_t nbits) {
    size_t k = 0;
    while (k < nbits) {
        const size_t s = src_off + k;
        const size_t take = std::min<size_t>({64 - (dst_off + k) % 64, 64 - (s & 63), nbits - k});
        uint64_t v = src[s >> 6] >> (s & 63);
        if (take < 64) v &= (1ULL << take) - 1ULL;
        dst[(dst_off + k) >> 6] ^= v << ((dst_off + k) & 63);
        k += take;
    }
}

struct SignedZ {                                   // the signed pure-Z subgroup G_z, reduced echelon
    int n = 0;
    std::vector<int> piv_row;                      // wire → its pivot row (-1: a free coordinate)
    std::vector<int> freeq;                        // the free coordinates (ascending)
    std::vector<int> free_ix;                      // wire → index in freeq (-1: a pivot)
    std::vector<const Pauli*> rows;                // the Z-pivot rref rows
    std::vector<uint8_t> sign;                     // per row: the ± bit ((phase >> 1) & 1)
};

SignedZ signed_z_(const CertifiedGroupPlanes& G, int n) {
    SignedZ Z;
    Z.n = n;
    Z.piv_row.assign((size_t)n, -1);
    for (int r = 0; r < G.rank; ++r) {
        if (G.pivcol[(size_t)r] < n) continue;      // X-support: not pure Z
        const Pauli& row = G.rref[(size_t)r];
        if (row.phase & 1)
            throw std::logic_error("plan class: a pure-Z certified element with an imaginary phase (engine bug)");
        for (uint64_t x : row.x)
            if (x) throw std::logic_error("plan class: a Z-pivot rref row with X-support (engine bug)");
        Z.piv_row[(size_t)(G.pivcol[(size_t)r] - n)] = (int)Z.rows.size();
        Z.rows.push_back(&row);
        Z.sign.push_back((uint8_t)((row.phase >> 1) & 1));
    }
    Z.free_ix.assign((size_t)n, -1);
    for (int q = 0; q < n; ++q)
        if (Z.piv_row[(size_t)q] < 0) { Z.free_ix[(size_t)q] = (int)Z.freeq.size(); Z.freeq.push_back(q); }
    return Z;
}

// off_c = Σ_{i<c} (m − 1 − i): the start of record c's (ii) row (pairs (c, c'), c' > c)
inline size_t ii_off_(int c, int m) { return (size_t)c * (size_t)m - (size_t)c * ((size_t)c + 1) / 2; }

}  // namespace

PlanClassifier compile_plan_classifier(const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                                       const std::vector<std::pair<int, int>>& upairs,
                                       const std::vector<uint64_t>& upair_keep, const std::vector<uint64_t>& prodmask,
                                       int n, bool audit, bool identity_only) {
    PlanClassifier pc;
    pc.n = n;
    pc.NW = (n + 63) / 64;
    pc.PW = (int)((upairs.size() + 63) / 64);
    pc.KW = pc.NW + pc.PW;
    pc.pairs = upairs.size();
    const int m = (int)cols.size();
    // U: wires with X-support in any record column
    pc.umask.assign((size_t)pc.NW, 0);
    for (const RecColumn& c : cols)
        for (int w = 0; w < pc.NW && w < (int)c.M.x.size(); ++w) pc.umask[(size_t)w] |= c.M.x[(size_t)w];
    for (int w = 0; w < pc.NW && w < (int)prodmask.size(); ++w) pc.umask[(size_t)w] &= ~prodmask[(size_t)w];
    std::vector<int> kept;                         // kept pair indices (ascending)
    pc.pkeep.assign((size_t)pc.PW, 0);
    for (size_t i = 0; i < upairs.size(); ++i)
        if ((i >> 6) < upair_keep.size() && bit_(upair_keep.data(), i)) { kept.push_back((int)i); pc.pkeep[i >> 6] |= 1ULL << (i & 63); }
    pc.kept_pairs = kept.size();
    const SignedZ Z = signed_z_(G, n);
    const size_t nfree = Z.freeq.size();
    const size_t Wc = nfree + 1;                   // per record: the reduced z-part, then the sign bit
#ifndef XTIM_MUTATE_K2P_DROP_II
    pc.columns = (size_t)m * Wc + (size_t)m * (size_t)(m - 1) / 2;
#else
    pc.columns = (size_t)m * Wc;                   // mutation build only (MUT-K2P-DROP-II): condition (ii) dropped
#endif
    const size_t cw = (pc.columns + 63) / 64;
    pc.work = (uint64_t)kept.size() * (uint64_t)kept.size() * (uint64_t)cw;
    if (identity_only) pc.cap = "identity";
    else if (kept.size() > kPlanClassMaxPairs) pc.cap = "pairs";
    else if (pc.work > kPlanClassMaxWork) pc.cap = "work";
    if (!pc.cap.empty() || kept.empty()) { pc.pi_identity = true; return pc; }
    if (m == 0) {                                  // no record column: every kept pair is in the kernel
        pc.pi_identity = false;
        std::vector<uint64_t> row((size_t)pc.PW, 0);
        for (int p : kept) {
            std::fill(row.begin(), row.end(), 0ULL);
            set_(row.data(), (size_t)p);
            pc.pivots.push_back(p);
            pc.basis.insert(pc.basis.end(), row.begin(), row.end());
        }
        return pc;
    }
    // Per wire w: red_w = e_w reduced mod G_z over the free coordinates (packed, nfree bits) and sgn_w (the rows used).
    const size_t FW = (nfree + 63) / 64 + 1;       // +1: the sign bit may spill into one more word
    std::vector<uint64_t> red((size_t)n * FW, 0);
    std::vector<uint8_t> sgn((size_t)n, 0);
    for (int w = 0; w < n; ++w) {
        uint64_t* r = red.data() + (size_t)w * FW;
        const int pr = Z.piv_row[(size_t)w];
        if (pr < 0) { set_(r, (size_t)Z.free_ix[(size_t)w]); continue; }
        const Pauli& row = *Z.rows[(size_t)pr];       // e_w ⊕ row.z: the pivot clears, the rest are free coordinates
        sgn[(size_t)w] = Z.sign[(size_t)pr];
        for (size_t ww = 0; ww < row.z.size(); ++ww)
            for (uint64_t word = row.z[ww]; word; word &= word - 1) {
                const int q = (int)(ww * 64 + (size_t)__builtin_ctzll(word));
                if (q == w) continue;
                const int f = Z.free_ix[(size_t)q];
                if (f < 0) throw std::logic_error("plan class: a reduced row touches a pivot coordinate (engine bug)");
                r[(size_t)f >> 6] ^= 1ULL << (f & 63);
            }
    }
    // Per wire w: the m-bit column (x_c(w))_c (the (ii) rows)
    const size_t MW = ((size_t)m + 63) / 64 + 1;
    std::vector<uint64_t> XT((size_t)n * MW, 0);
    for (int c = 0; c < m; ++c)
        for (size_t ww = 0; ww < cols[(size_t)c].M.x.size(); ++ww)
            for (uint64_t word = cols[(size_t)c].M.x[ww]; word; word &= word - 1) {
                const int q = (int)(ww * 64 + (size_t)__builtin_ctzll(word));
                if (q < n) set_(XT.data() + (size_t)q * MW, (size_t)c);
            }
    auto xc = [&](int c, int w) { return bit_(XT.data() + (size_t)w * MW, (size_t)c); };
    // The kernel basis K: rows × PW words over the pair universe; initially e_p for every kept pair.
    const size_t PW = (size_t)pc.PW;
    size_t k = kept.size();
    std::vector<uint64_t> K(k * PW, 0);
    for (size_t i = 0; i < k; ++i) set_(K.data() + i * PW, (size_t)kept[i]);
    const size_t ii_base = (size_t)m * Wc;
    std::vector<uint64_t> Lblk, img, tmp(FW + 1, 0), rowv(MW + 1, 0);
    for (size_t b0 = 0; b0 < pc.columns && k > 0; b0 += kPlanClassBlockBits) {
        const size_t b1 = std::min(pc.columns, b0 + kPlanClassBlockBits);
        const size_t BW = (b1 - b0 + 63) / 64;
        Lblk.assign(kept.size() * BW, 0);
        for (size_t i = 0; i < kept.size(); ++i) {
            uint64_t* L = Lblk.data() + i * BW;
            const int j = upairs[(size_t)kept[i]].first, l = upairs[(size_t)kept[i]].second;
            // the record columns of this block
            if (b0 < ii_base) {
                const int c0 = (int)(b0 / Wc), c1 = (int)std::min<size_t>((size_t)m, (std::min(b1, ii_base) + Wc - 1) / Wc);
                for (int c = c0; c < c1; ++c) {
                    const bool xj = xc(c, j), xl = xc(c, l);
                    if (!xj && !xl) continue;
                    std::fill(tmp.begin(), tmp.end(), 0ULL);
                    if (xj) for (size_t w = 0; w < FW; ++w) tmp[w] ^= red[(size_t)l * FW + w];
                    if (xl) for (size_t w = 0; w < FW; ++w) tmp[w] ^= red[(size_t)j * FW + w];
                    const uint8_t s = (uint8_t)((xj ? sgn[(size_t)l] : 0) ^ (xl ? sgn[(size_t)j] : 0) ^ (xj && xl ? 1 : 0));
                    if (s) set_(tmp.data(), nfree);
                    // the column's bits [cs, ce) within the block
                    const size_t cs = std::max(b0, (size_t)c * Wc), ce = std::min(b1, (size_t)(c + 1) * Wc);
                    xor_bits_from_(L, cs - b0, tmp.data(), cs - (size_t)c * Wc, ce - cs);
                }
            }
            // the (ii) rows of this block: row c holds (c, c') for c' > c at off_c + (c' − c − 1)
            if (b1 > ii_base) {
                const size_t r0 = std::max(b0, ii_base) - ii_base, r1 = b1 - ii_base;
                // the first row c with off_{c+1} > r0
                int c = 0;
                while (c + 1 < m && ii_off_(c + 1, m) <= r0) ++c;
                for (; c + 1 < m && ii_off_(c, m) < r1; ++c) {
                    const bool aj = xc(c, j), bl = xc(c, l);
                    if (!aj && !bl) continue;
                    std::fill(rowv.begin(), rowv.end(), 0ULL);
                    if (aj) for (size_t w = 0; w < MW; ++w) rowv[w] ^= XT[(size_t)l * MW + w];   // a_c · B
                    if (bl) for (size_t w = 0; w < MW; ++w) rowv[w] ^= XT[(size_t)j * MW + w];   // b_c · A
                    const size_t off = ii_off_(c, m), len = (size_t)(m - c - 1);
                    const size_t cs = std::max(r0, off), ce = std::min(r1, off + len);
                    if (cs >= ce) continue;
                    // bits c' = c + 1 + (cs − off) .. of rowv
                    xor_bits_from_(L, ii_base + cs - b0, rowv.data(), (size_t)(c + 1) + (cs - off), ce - cs);
                }
            }
        }
        // the images of the basis rows under this block
        img.assign(k * BW, 0);
        for (size_t r = 0; r < k; ++r) {
            const uint64_t* v = K.data() + r * PW;
            uint64_t* I = img.data() + r * BW;
            for (size_t i = 0; i < kept.size(); ++i)
                if (bit_(v, (size_t)kept[i])) {
                    const uint64_t* L = Lblk.data() + i * BW;
                    for (size_t w = 0; w < BW; ++w) I[w] ^= L[w];
                }
        }
        // eliminate: rows whose image is zero after the reduction span the kernel of this block's columns
        size_t rank = 0;
        for (size_t w = 0; w < BW && rank < k; ++w)
            for (int b = 0; b < 64 && rank < k; ++b) {
                const uint64_t mask = 1ULL << b;
                size_t piv = rank;
                while (piv < k && !(img[piv * BW + w] & mask)) ++piv;
                if (piv == k) continue;
                if (piv != rank) {
                    std::swap_ranges(img.begin() + (long)(piv * BW), img.begin() + (long)((piv + 1) * BW), img.begin() + (long)(rank * BW));
                    std::swap_ranges(K.begin() + (long)(piv * PW), K.begin() + (long)((piv + 1) * PW), K.begin() + (long)(rank * PW));
                }
                const uint64_t* pi = img.data() + rank * BW;
                const uint64_t* pk = K.data() + rank * PW;
                for (size_t r = 0; r < k; ++r) {
                    if (r == rank || !(img[r * BW + w] & mask)) continue;
                    uint64_t* ri = img.data() + r * BW;
                    for (size_t ww = w; ww < BW; ++ww) ri[ww] ^= pi[ww];
                    uint64_t* rk = K.data() + r * PW;
                    for (size_t ww = 0; ww < PW; ++ww) rk[ww] ^= pk[ww];
                }
                ++rank;
            }
        // keep the rows [rank, k)
        if (rank) {
            std::move(K.begin() + (long)(rank * PW), K.begin() + (long)(k * PW), K.begin());
            k -= rank;
            K.resize(k * PW);
        }
    }
    // reduced echelon over the pair coordinates (ascending pair index): the pivots and the canonical basis
    {
        size_t rank = 0;
        std::vector<int> piv;
        for (size_t p = 0; p < upairs.size() && rank < k; ++p) {
            size_t r = rank;
            while (r < k && !bit_(K.data() + r * PW, p)) ++r;
            if (r == k) continue;
            if (r != rank)
                std::swap_ranges(K.begin() + (long)(r * PW), K.begin() + (long)((r + 1) * PW), K.begin() + (long)(rank * PW));
            const uint64_t* pk = K.data() + rank * PW;
            for (size_t rr = 0; rr < k; ++rr) {
                if (rr == rank || !bit_(K.data() + rr * PW, p)) continue;
                uint64_t* rk = K.data() + rr * PW;
                for (size_t ww = 0; ww < PW; ++ww) rk[ww] ^= pk[ww];
            }
            piv.push_back((int)p);
            ++rank;
        }
        if (rank != k) throw std::logic_error("plan class: dependent kernel basis rows (engine bug)");
        pc.pivots = std::move(piv);
        pc.basis = std::move(K);
    }
    pc.pi_identity = pc.pivots.empty();
    if (audit) audit_plan_classifier(pc, G, cols, upairs);
    return pc;
}

bool plan_class_kernel_member(const uint64_t* dB, const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                              const std::vector<std::pair<int, int>>& upairs) {
    const int n = G.n_qubits;
    const int m = (int)cols.size();
    std::vector<std::pair<int, int>> P;
    for (size_t i = 0; i < upairs.size(); ++i) if (bit_(dB, i)) P.push_back(upairs[i]);
    auto xq = [&](int c, int q) { return q < cols[(size_t)c].M.n && cols[(size_t)c].M.xbit(q); };
    for (int c = 0; c < m; ++c) {
        Pauli g(n);
        int q = 0;
        for (const auto& pr : P) {
            const bool xj = xq(c, pr.first), xl = xq(c, pr.second);
            if (xj) g.flipz(pr.second);
            if (xl) g.flipz(pr.first);
            if (xj && xl) q ^= 1;
        }
        g.phase = 2 * q;
        const Membership mm = G.reduce(g);
        if (mm.verdict != Membership::IN_GROUP || mm.sign != +1) return false;
#ifndef XTIM_MUTATE_K2P_DROP_II
        for (int c2 = c + 1; c2 < m; ++c2) {
            int beta = 0;
            for (const auto& pr : P)
                beta ^= (int)(xq(c, pr.first) & xq(c2, pr.second)) ^ (int)(xq(c, pr.second) & xq(c2, pr.first));
            if (beta) return false;
        }
#endif                                              // mutation build only (MUT-K2P-DROP-II): (ii) not checked
    }
    return true;
}

void audit_plan_classifier(const PlanClassifier& pc, const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                           const std::vector<std::pair<int, int>>& upairs) {
    const_cast<PlanClassifier&>(pc).audited = 0;
    if (pc.pi_identity) return;
    for (size_t i = 0; i < pc.dim(); ++i) {
        const uint64_t* v = pc.basis.data() + i * (size_t)pc.PW;
        if (!plan_class_kernel_member(v, G, cols, upairs))
            throw std::logic_error("plan class: kernel basis vector " + std::to_string(i) +
                                   " violates the exact Pauli kernel (i)/(ii) (engine bug)");
        ++const_cast<PlanClassifier&>(pc).audited;
    }
}

}  // namespace qeccore
