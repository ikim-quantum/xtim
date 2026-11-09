// J-a N5 (switched OFF): the definite-sign rule of a retained record state (record_signs.hpp).
#include "qeccore/record_signs.hpp"

#include <stdexcept>
#include <string>

namespace qeccore {

namespace {
inline bool getb(const std::vector<uint64_t>& w, int i) { return (w[(size_t)(i >> 6)] >> (i & 63)) & 1ULL; }
inline void setb(std::vector<uint64_t>& w, int i) { w[(size_t)(i >> 6)] |= 1ULL << (i & 63); }
inline void xorw(std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i];
}
inline bool zero(const std::vector<uint64_t>& a) {
    for (uint64_t w : a) if (w) return false;
    return true;
}
inline int lowbit(const std::vector<uint64_t>& a) {
    for (size_t i = 0; i < a.size(); ++i) if (a[i]) return (int)(i * 64 + (size_t)__builtin_ctzll(a[i]));
    return -1;
}
std::vector<uint64_t> symp(const Pauli& p, int NW) {
    std::vector<uint64_t> v((size_t)(2 * NW), 0);
    for (int w = 0; w < NW; ++w) { v[(size_t)w] = p.x[(size_t)w]; v[(size_t)(NW + w)] = p.z[(size_t)w]; }
    return v;
}
}  // namespace

SignRuleBuilder::SignRuleBuilder(int n, const std::vector<Pauli>& Q, const std::vector<uint8_t>& use,
                                 const std::vector<Pauli>& G)
    : n_(n), m_((int)Q.size()), NW_((n + 63) / 64), Q_(Q) {
    if (use.size() != Q.size()) throw std::invalid_argument("SignRuleBuilder: use / Q size mismatch");
    // The commuting subgroup {s ∈ ⟨G⟩ : [s, Q_c] = 0 for every used c}: the left null space of the
    // anticommutation matrix A[i][c] (rows = generators), each null vector's generator product (exact).
    const int r = (int)G.size(), AW = (m_ + 63) / 64 + 1, CW = (r + 63) / 64 + 1;
    std::vector<std::vector<uint64_t>> arow, crow;      // reduced rows (A part, combination part)
    std::vector<int> apiv;
    for (int i = 0; i < r; ++i) {
        std::vector<uint64_t> a((size_t)AW, 0), cmb((size_t)CW, 0);
        for (int c = 0; c < m_; ++c)
            if (use[(size_t)c] && Pauli::anticommute_bit(G[(size_t)i], Q[(size_t)c])) setb(a, c);
        setb(cmb, i);
        for (size_t k = 0; k < arow.size(); ++k)
            if (getb(a, apiv[k])) { xorw(a, arow[k]); xorw(cmb, crow[k]); }
        if (zero(a)) {
            Pauli s(n);
            for (int j = 0; j < r; ++j) if (getb(cmb, j)) s = Pauli::multiply(s, G[(size_t)j]);
            S_.push_back(std::move(s));
        } else {
            apiv.push_back(lowbit(a));
            arow.push_back(std::move(a));
            crow.push_back(std::move(cmb));
        }
    }
    // H's generators, records FIRST (storage column order), then the commuting subgroup's basis.
    TW_ = (m_ + (int)S_.size() + 63) / 64 + 1;
    for (int c = 0; c < m_; ++c) {
        if (!use[(size_t)c]) continue;
        std::vector<uint64_t> t((size_t)TW_, 0);
        setb(t, c);
        add_(symp(Q[(size_t)c], NW_), std::move(t));
    }
    for (size_t k = 0; k < S_.size(); ++k) {
        std::vector<uint64_t> t((size_t)TW_, 0);
        setb(t, m_ + (int)k);
        add_(symp(S_[k], NW_), std::move(t));
    }
}

void SignRuleBuilder::add_(std::vector<uint64_t> v, std::vector<uint64_t> t) {
    for (size_t k = 0; k < vec_.size(); ++k)
        if (getb(v, piv_[k])) { xorw(v, vec_[k]); xorw(t, tag_[k]); }
    if (zero(v)) return;                                 // dependent: the earlier generators represent it
    piv_.push_back(lowbit(v));
    vec_.push_back(std::move(v));
    tag_.push_back(std::move(t));
}

SignRule SignRuleBuilder::rule(const Pauli& gp) const {
    SignRule R;
    R.U.assign((size_t)((m_ + 63) / 64 + 1), 0);
    std::vector<uint64_t> v = symp(gp, NW_), t((size_t)TW_, 0);
    for (size_t k = 0; k < vec_.size(); ++k)
        if (getb(v, piv_[k])) { xorw(v, vec_[k]); xorw(t, tag_[k]); }
    if (!zero(v)) return R;                              // ±C0†gC0 ∉ H: indefinite (reported 0)
    Pauli prod(n_);
    for (int c = 0; c < m_; ++c)
        if (getb(t, c)) { prod = Pauli::multiply(prod, Q_[(size_t)c]); setb(R.U, c); }
    for (size_t k = 0; k < S_.size(); ++k)
        if (getb(t, m_ + (int)k)) prod = Pauli::multiply(prod, S_[k]);
    for (int w = 0; w < NW_; ++w)
        if (prod.x[(size_t)w] != gp.x[(size_t)w] || prod.z[(size_t)w] != gp.z[(size_t)w])
            throw std::logic_error("signs: the H representation does not reproduce C0^dag g C0 (engine bug)");
    const int d = ((gp.phase - prod.phase) % 4 + 4) % 4;
    if (d & 1)
        throw std::logic_error("signs: C0^dag g C0 is i times a member of H (a non-Hermitian g or record; engine bug)");
    R.definite = true;
    R.eps = (uint8_t)(d >> 1);
    return R;
}

}  // namespace qeccore
