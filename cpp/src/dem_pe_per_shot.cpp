// Per-shot law of a PAULI_EXPECTATION column for DEM export (3.1.13). See dem_pe_per_shot.hpp
// for the derivation; every quantity here is exact (GF(2) linear algebra + Born sums on the bare
// state through the engine's own framed_expectation).
#include "qeccore/dem_pe_per_shot.hpp"

#include <algorithm>
#include <cmath>

#include "qeccore/pauli_kernels.hpp"

namespace qeccore {

namespace {
inline bool pe_bit(const std::vector<uint64_t>& v, int i) { return (v[(size_t)(i >> 6)] >> (i & 63)) & 1; }
inline void pe_flip(std::vector<uint64_t>& v, int i) { v[(size_t)(i >> 6)] ^= 1ull << (i & 63); }
inline void pe_xor(std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (size_t w = 0; w < a.size(); ++w) a[w] ^= b[w];
}
inline bool pe_any(const std::vector<uint64_t>& v) {
    for (uint64_t w : v) if (w) return true;
    return false;
}
inline int pe_lowbit(const std::vector<uint64_t>& v) {
    for (size_t w = 0; w < v.size(); ++w) if (v[w]) return (int)(w * 64) + __builtin_ctzll(v[w]);
    return -1;
}
}  // namespace

PeShotAnalyzer::PeShotAnalyzer(const FramedSuperposition& bare,
                               const std::vector<std::pair<int, int>>& reads,
                               long leaf_budget, double work_budget)
    : L_(bare), reads_(reads), leaf_budget_(leaf_budget), work_budget_(work_budget) {
    n_ = bare.n();
    M_ = (int)reads.size();
    k_ = (int)bare.free.size();
    NW_ = (n_ + 63) / 64;
    MW_ = std::max(1, (M_ + 63) / 64);
    FW_ = std::max(1, (2 * k_ + 63) / 64);
    nfm_.assign((size_t)NW_, 0);
    for (int a = 0; a < n_; ++a) pe_flip(nfm_, a);
    for (int f : bare.free) if (f >= 0 && f < n_ && pe_bit(nfm_, f)) pe_flip(nfm_, f);
    zt_x_.assign((size_t)n_, std::vector<uint64_t>((size_t)NW_, 0));
    zt_z_.assign((size_t)n_, std::vector<uint64_t>((size_t)NW_, 0));
    for (int a = 0; a < n_; ++a) {
        const Pauli& g = bare.U.Zrow[(size_t)a];
        for (int w = 0; w < NW_; ++w) {
            uint64_t bx = g.x[(size_t)w], bz = g.z[(size_t)w];
            while (bx) { const int q = w * 64 + __builtin_ctzll(bx); bx &= bx - 1; if (q < n_) pe_flip(zt_x_[(size_t)q], a); }
            while (bz) { const int q = w * 64 + __builtin_ctzll(bz); bz &= bz - 1; if (q < n_) pe_flip(zt_z_[(size_t)q], a); }
        }
    }
    // Noiseless elimination of the reads' non-free X-parts (record order).
    std::vector<uint64_t> x((size_t)NW_);
    for (int j = 0; j < M_; ++j) {
        const auto& [b, q] = reads_[(size_t)j];
        Pauli R = single_pauli(b, q, n_);
        xv(R, x);
        Row r;
        r.v.assign((size_t)NW_, 0);
        for (int w = 0; w < NW_; ++w) r.v[(size_t)w] = x[(size_t)w] & nfm_[(size_t)w];
        r.comb.assign((size_t)MW_, 0);
        pe_flip(r.comb, j);
        phi_of(R, x, r.phi);
        reduce(r.v, r.comb, r.phi);
        r.piv = pe_lowbit(r.v);
        if (r.piv >= 0) ech_.push_back(std::move(r));
        else null_.push_back({std::move(r.comb), std::move(r.phi)});
    }
}

// X-part of P in frame coordinates over ALL frame positions a: [P, Zrow[a]].
void PeShotAnalyzer::xv(const Pauli& P, std::vector<uint64_t>& out) const {
    out.assign((size_t)NW_, 0);
    for (int w = 0; w < NW_; ++w) {
        uint64_t bx = P.x[(size_t)w], bz = P.z[(size_t)w];
        while (bx) { const int q = w * 64 + __builtin_ctzll(bx); bx &= bx - 1; pe_xor(out, zt_z_[(size_t)q]); }
        while (bz) { const int q = w * 64 + __builtin_ctzll(bz); bz &= bz - 1; pe_xor(out, zt_x_[(size_t)q]); }
    }
}

// Free-sector part of P in frame coordinates: bit 2d = X on free[d] ([P, Zrow[free[d]]]),
// bit 2d+1 = Z on free[d] ([P, Xrow[free[d]]]).
void PeShotAnalyzer::phi_of(const Pauli& P, const std::vector<uint64_t>& xvP,
                            std::vector<uint64_t>& out) const {
    out.assign((size_t)FW_, 0);
    for (int d = 0; d < k_; ++d) {
        const int f = L_.free[(size_t)d];
        if (pe_bit(xvP, f)) pe_flip(out, 2 * d);
        if (Pauli::anticommute_bit(P, L_.U.Xrow[(size_t)f])) pe_flip(out, 2 * d + 1);
    }
}

void PeShotAnalyzer::reduce(std::vector<uint64_t>& v, std::vector<uint64_t>& comb,
                            std::vector<uint64_t>& phi) const {
    for (const Row& r : ech_)
        if (pe_bit(v, r.piv)) { pe_xor(v, r.v); pe_xor(comb, r.comb); pe_xor(phi, r.phi); }
}

PeShotAnalyzer::Law PeShotAnalyzer::structure(const std::vector<int>& J,
                                               const std::function<Pauli(int)>& read_pauli,
                                               const Pauli& X) {
    Law law;
    law.Xw = X;
    const int nJ = (int)J.size();
    if (nJ > 63) {
        law.too_large = "a single fault changes " + std::to_string(nJ) +
                        " (> 63) terminal reads correlated with an expectation column";
        return law;
    }
    auto jbits = [&](const std::vector<uint64_t>& comb) {
        uint64_t p = 0;
        for (int i = 0; i < nJ; ++i) if (pe_bit(comb, J[(size_t)i])) p |= 1ull << i;
        return p;
    };
    // ── Projection of the noiseless nullspace W onto J: pivots + the kernel W_J. ──
    struct Piv { uint64_t proj; std::vector<uint64_t> comb, phi; };
    std::vector<Piv> piv;
    struct Kern { int e; uint64_t used; std::vector<uint64_t> phi; };
    std::vector<Kern> kern;
    auto reduce_proj = [&](uint64_t& p, uint64_t& used) {
        for (size_t i = 0; i < piv.size(); ++i)
            if ((p >> __builtin_ctzll(piv[i].proj)) & 1) { p ^= piv[i].proj; used ^= 1ull << i; }
    };
    for (int e = 0; e < (int)null_.size(); ++e) {
        uint64_t p = nJ ? jbits(null_[(size_t)e].comb) : 0, used = 0;
        if (p) reduce_proj(p, used);
        std::vector<uint64_t> ph = null_[(size_t)e].phi;
        for (size_t i = 0; i < piv.size(); ++i) if ((used >> i) & 1) pe_xor(ph, piv[i].phi);
        if (p) {
            std::vector<uint64_t> cb = null_[(size_t)e].comb;
            for (size_t i = 0; i < piv.size(); ++i) if ((used >> i) & 1) pe_xor(cb, piv[i].comb);
            piv.push_back({p, std::move(cb), std::move(ph)});
        } else if (k_ > 0) {
            kern.push_back({e, used, std::move(ph)});
        }
    }
    auto w_comb = [&](uint64_t used, std::vector<uint64_t>& cb) {
        for (size_t i = 0; i < piv.size(); ++i) if ((used >> i) & 1) pe_xor(cb, piv[i].comb);
    };
    auto w_phi = [&](uint64_t used, std::vector<uint64_t>& ph) {
        for (size_t i = 0; i < piv.size(); ++i) if ((used >> i) & 1) pe_xor(ph, piv[i].phi);
    };
    // ── The changed reads: reduce each world row against the noiseless echelon. ──
    struct JRow { std::vector<uint64_t> r, C, phiC, phiA; uint64_t pi, used; };
    std::vector<JRow> jr((size_t)nJ);
    std::vector<uint64_t> x;
    for (int i = 0; i < nJ; ++i) {
        Pauli A = read_pauli(J[(size_t)i]);
        xv(A, x);
        JRow& row = jr[(size_t)i];
        phi_of(A, x, row.phiA);
        row.r.assign((size_t)NW_, 0);
        for (int w = 0; w < NW_; ++w) row.r[(size_t)w] = x[(size_t)w] & nfm_[(size_t)w];
        row.C.assign((size_t)MW_, 0);
        row.phiC.assign((size_t)FW_, 0);
        reduce(row.r, row.C, row.phiC);
        row.pi = jbits(row.C);
        row.used = 0;
        reduce_proj(row.pi, row.used);
    }
    // Elimination of the J system rows (r_i | pi_i), tracking the u-combination.
    struct ERow { std::vector<uint64_t> r; uint64_t pi, u; int pr_piv, pi_piv; };
    std::vector<ERow> er;
    std::vector<uint64_t> nullu;   // homogeneous solutions u
    auto reduce_e = [&](std::vector<uint64_t>& r, uint64_t& pi, uint64_t& u) {
        for (const ERow& e : er) {
            const bool hit = e.pr_piv >= 0 ? pe_bit(r, e.pr_piv) : ((pi >> e.pi_piv) & 1);
            if (hit) { pe_xor(r, e.r); pi ^= e.pi; u ^= e.u; }
        }
    };
    for (int i = 0; i < nJ; ++i) {
        std::vector<uint64_t> r = jr[(size_t)i].r;
        uint64_t pi = jr[(size_t)i].pi, u = 1ull << i;
        reduce_e(r, pi, u);
        const int pv = pe_lowbit(r);
        if (pv >= 0) er.push_back({std::move(r), pi, u, pv, -1});
        else if (pi) er.push_back({std::move(r), pi, u, -1, __builtin_ctzll(pi)});
        else nullu.push_back(u);
    }
    // Read set / phi of the world-nullspace element S_u (u a combination of J rows) plus an
    // optional particular right-hand side (C0, used0).
    auto s_of = [&](uint64_t u, const std::vector<uint64_t>* C0, uint64_t used0,
                    std::vector<uint64_t>& cb, std::vector<uint64_t>* ph) {
        cb.assign((size_t)MW_, 0);
        if (C0) pe_xor(cb, *C0);
        uint64_t used = used0;
        if (ph) ph->assign((size_t)FW_, 0);
        for (int i = 0; i < nJ; ++i)
            if ((u >> i) & 1) {
                pe_flip(cb, J[(size_t)i]);
                pe_xor(cb, jr[(size_t)i].C);
                used ^= jr[(size_t)i].used;
                if (ph) { pe_xor(*ph, jr[(size_t)i].phiA); pe_xor(*ph, jr[(size_t)i].phiC); }
            }
        w_comb(used, cb);
        if (ph) w_phi(used, *ph);
    };
    auto to_list = [&](const std::vector<uint64_t>& cb) {
        std::vector<int> s;
        for (int j = 0; j < M_; ++j) if (pe_bit(cb, j)) s.push_back(j);
        return s;
    };
    // ── Coin: is xn(X) in the world span? ──
    std::vector<uint64_t> xX;
    xv(X, xX);
    for (int w = 0; w < NW_; ++w) xX[(size_t)w] &= nfm_[(size_t)w];
    std::vector<uint64_t> S0cb((size_t)MW_, 0);
    if (pe_any(xX)) {
        std::vector<uint64_t> CX((size_t)MW_, 0), phiCX((size_t)FW_, 0);
        reduce(xX, CX, phiCX);          // xX <- residual r_X
        uint64_t piX = jbits(CX), usedX = 0;
        reduce_proj(piX, usedX);
        uint64_t u0 = 0;
        reduce_e(xX, piX, u0);
        if (pe_any(xX) || piX) { law.zero = true; return law; }   // not in the span: v = 0 per shot
        law.coin = true;
        s_of(u0, &CX, usedX, S0cb, nullptr);
        law.S0 = to_list(S0cb);
    }
    // ── W1: the free-sector image of the world nullspace W' = W_J (+) span{S_u}. ──
    struct BRow { std::vector<uint64_t> phi, comb; int piv; };
    std::vector<BRow> basis;
    std::vector<uint8_t> usedb;
    auto add_gen = [&](std::vector<uint64_t> ph, const std::function<void(std::vector<uint64_t>&)>& comb_of) {
        usedb.assign(basis.size(), 0);
        for (size_t i = 0; i < basis.size(); ++i)
            if (pe_bit(ph, basis[i].piv)) { pe_xor(ph, basis[i].phi); usedb[i] = 1; }
        const int pv = pe_lowbit(ph);
        if (pv < 0) return;
        std::vector<uint64_t> cb;
        comb_of(cb);
        for (size_t i = 0; i < usedb.size(); ++i) if (usedb[i]) pe_xor(cb, basis[i].comb);
        basis.push_back({std::move(ph), std::move(cb), pv});
    };
    if (k_ > 0) {
        for (const Kern& kg : kern) {
            if ((int)basis.size() >= 2 * k_) break;
            add_gen(kg.phi, [&](std::vector<uint64_t>& cb) {
                cb = null_[(size_t)kg.e].comb;
                w_comb(kg.used, cb);
            });
        }
        for (uint64_t u : nullu) {
            std::vector<uint64_t> cb, ph;
            s_of(u, nullptr, 0, cb, &ph);
            add_gen(ph, [&](std::vector<uint64_t>& c2) { c2 = cb; });
        }
    }
    law.d = (int)basis.size();
    if (law.d > 63) {
        law.too_large = "an expectation column's per-shot value is conditioned on " +
                        std::to_string(law.d) + " (> 63) magic-correlated read parities";
        return law;
    }
    auto product = [&](const std::vector<int>& S) {
        Pauli A(n_);
        for (int j : S) pmul_into(A, read_pauli(j));
        return A;
    };
    if (law.coin) {
        law.coinP = product(law.S0);
        law.X0 = Pauli::multiply(X, law.coinP);
    } else {
        law.coinP = Pauli(n_);
        law.X0 = X;
    }
    for (const BRow& b : basis) {
        law.T.push_back(to_list(b.comb));
        law.yP.push_back(product(law.T.back()));
    }
    return law;
}

namespace {
constexpr double kBranchEps = 1e-12;   // the engine's coin threshold (framed_reference_read)
// The work budget in a message: an integer count, or "inf" when the caller lifted the bound
// (pe_max_work=float('inf'); a (long) cast of infinity is undefined).
std::string fmt_work(double w) {
    return std::isfinite(w) ? std::to_string((long)w) : std::string("inf");
}

// Exact branch walk over commuting Hermitian Paulis B (in order) on a framed state: every outcome
// pattern of positive probability becomes a leaf; leaf(state, pattern, prob) is called on each and
// returns false to stop. Returns false when more than `budget` leaves would be produced.
// Work is a deterministic cost proxy: each born_p1 / measure_pauli / leaf read is charged
// (chi + 1) * n * (k + 1) (an anticommuting collapse touches every amplitude entry against the
// frame rows and the free set; calibrated ~3-4e7 units/s on the k=11-13 T-ladder, so the default
// budget PeShotAnalyzer::kWorkBudget is ~0.5 s). Past the work budget the walk stops after bounded
// work and the caller refuses / reports by name.
struct BranchWalk {
    const std::vector<Pauli>& B;
    long budget;
    const std::function<bool(const FramedSuperposition&, uint64_t, double)>& leaf;
    long leaves = 0;
    double work = 0.0;
    double work_budget = PeShotAnalyzer::kWorkBudget;
    bool over = false, stopped = false, over_work = false;
    bool chi2 = false;   // forced-law charge: also (chi+1)^2 per step (measured: a collapse on a
                         // dense k-T state costs ~1.3 ns x chi^2 on this host)
    void charge(const FramedSuperposition& L, double times) {
        const double c = (double)(L.chi() + 1);
        double unit = c * (double)L.n() * (double)(L.k() + 1);
        if (chi2) unit = std::max(unit, c * c);
        work += times * unit;
        if (work > work_budget) over = over_work = true;
    }
    void walk(FramedSuperposition& L, int i, uint64_t f, double p) {
        while (i < (int)B.size()) {
            charge(L, 2.0);
            if (over || stopped) return;
            const double pp = L.born_p1(B[(size_t)i]);
            const double pm = 1.0 - pp;
            if (pp > kBranchEps && pm > kBranchEps) {
                FramedSuperposition Lm = L;
                L.measure_pauli(B[(size_t)i], 0.0);
                walk(L, i + 1, f, p * pp);
                if (over || stopped) return;
                charge(Lm, 1.0);
                if (over) return;
                Lm.measure_pauli(B[(size_t)i], std::nextafter(1.0, 0.0));
                walk(Lm, i + 1, f | (1ull << i), p * pm);
                return;
            }
            const bool minus = !(pp > kBranchEps);
            L.measure_pauli(B[(size_t)i], minus ? std::nextafter(1.0, 0.0) : 0.0);
            if (minus) f |= 1ull << i;
            ++i;
        }
        charge(L, 1.0);
        if (over || stopped) return;
        if (++leaves > budget) { over = true; return; }
        if (!leaf(L, f, p)) stopped = true;
    }
};
}  // namespace

bool PeShotAnalyzer::walk_h(Law& law, const std::function<bool(const std::vector<Leaf>&)>& stop) {
    law.leaves.clear();
    law.walked = false;
    if (law.zero) { law.walked = true; return true; }
    if (law.yP.empty()) {                       // no y bit: one leaf, no state copy
        law.leaves.push_back({0, 1.0, L_.expectation(law.X0)});
        law.walked = !(stop && stop(law.leaves));       // same contract as the walk
        return true;
    }
    const std::function<bool(const FramedSuperposition&, uint64_t, double)> leaf =
        [&](const FramedSuperposition& S, uint64_t f, double p) {
            law.leaves.push_back({f, p, S.expectation(law.X0)});
            return !(stop && stop(law.leaves));
        };
    BranchWalk w{law.yP, leaf_budget_, leaf};
    w.work_budget = work_budget_;
    FramedSuperposition L = L_;
    w.walk(L, 0, 0, 1.0);
    if (w.over) {
        // Names the DEM-export argument (pe_max_work) when the work bound binds: raising it is the
        // documented way to let the exact analysis run further. The branch cap is fixed.
        law.too_large = "an expectation column's per-shot law exceeds the exact-analysis budget (" +
                        std::to_string(leaf_budget_) + " measurement branches / " +
                        fmt_work(work_budget_) + " work units = pe_max_work; " +
                        std::to_string(law.d) + " magic-correlated read parities, " +
                        std::to_string(L_.chi()) + " amplitudes)" +
                        (w.over_work ? "; raise pe_max_work on detector_error_model() to continue "
                                       "the exact analysis"
                                     : "; the branch cap is fixed");
        return false;
    }
    law.walked = !w.stopped;
    return true;
}

void PeShotAnalyzer::frame_parts(const Pauli& P, std::vector<uint64_t>& xn,
                                 std::vector<uint64_t>& phi) const {
    xv(P, xn);
    phi_of(P, xn, phi);
    for (int w = 0; w < NW_; ++w) xn[(size_t)w] &= nfm_[(size_t)w];
}

bool PeShotAnalyzer::joint_law(const std::vector<Pauli>& B, const std::vector<Pauli>& V,
                               std::vector<JointLeaf>& out, std::string& why) {
    int over_kind = 0;
    if (joint_law(B, V, out, why, leaf_budget_, work_budget_, &over_kind, false)) return true;
    // DEM-export caps: name the argument when it is the one that binds (the forced law's caller
    // names its own); the branch cap is fixed.
    if (over_kind == 2)
        why += "; raise pe_max_work (currently " + fmt_work(work_budget_) +
               ") on detector_error_model() to continue the exact analysis";
    else if (over_kind == 1)
        why += "; the branch cap is fixed";
    return false;
}

bool PeShotAnalyzer::joint_law(const std::vector<Pauli>& B, const std::vector<Pauli>& V,
                               std::vector<JointLeaf>& out, std::string& why, long leaf_budget,
                               double work_budget, int* over_kind, bool chi2_cost) {
    out.clear();
    if (over_kind) *over_kind = 0;
    if (B.size() > 63) { why = "a single fault correlates more than 63 parities"; return false; }
    // Memo by the exact content of (B, V): many alternatives share their correlated parities.
    std::string key;
    auto put = [&](const Pauli& P) {
        key.push_back((char)P.phase);
        key.append(reinterpret_cast<const char*>(P.x.data()), P.x.size() * 8);
        key.append(reinterpret_cast<const char*>(P.z.data()), P.z.size() * 8);
    };
    for (const Pauli& P : B) put(P);
    key.push_back('|');
    for (const Pauli& P : V) put(P);
    key.push_back(chi2_cost ? 'q' : 'l');
    if (auto it = joint_memo_.find(key); it != joint_memo_.end()) {
        const bool over_rows = (long)it->second.leaves.size() > leaf_budget;
        const bool over_work = it->second.work > work_budget;
        if (!over_rows && !over_work) { out = it->second.leaves; return true; }
        if (over_kind) *over_kind = over_work ? 2 : 1;   // what a fresh walk would have hit
        why = "the exact joint law of the fault's " + std::to_string(B.size()) +
              " correlated parities exceeds the exact-analysis budget (" +
              std::to_string(leaf_budget) + " measurement branches" +
              (std::isfinite(work_budget) ? " / " + std::to_string((long)work_budget) + " work units)"
                                          : ")");
        return false;
    }
    const std::function<bool(const FramedSuperposition&, uint64_t, double)> leaf =
        [&](const FramedSuperposition& S, uint64_t f, double p) {
            JointLeaf jl{f, p, {}};
            jl.v.reserve(V.size());
            for (const Pauli& X : V) jl.v.push_back(S.expectation(X));
            out.push_back(std::move(jl));
            return true;
        };
    BranchWalk w{B, leaf_budget, leaf};
    w.work_budget = work_budget;
    w.chi2 = chi2_cost;
    FramedSuperposition L = L_;
    w.walk(L, 0, 0, 1.0);
    if (w.over) {
        if (over_kind) *over_kind = w.over_work ? 2 : 1;
        why = "the exact joint law of the fault's " + std::to_string(B.size()) +
              " correlated parities exceeds the exact-analysis budget (" +
              std::to_string(leaf_budget) + " measurement branches" +
              (std::isfinite(work_budget) ? " / " + std::to_string((long)work_budget) + " work units)"
                                          : ")");
        out.clear();
        return false;
    }
    // Memory bound: at most 4096 entries AND kMemoCells stored values in total (leaves x (|V| +
    // 2)); a result that would exceed it is simply not cached (deterministic, never evicts).
    const size_t cells = out.size() * (V.size() + 2);
    if (joint_memo_.size() < 4096 && memo_cells_ + cells <= kMemoCells) {
        memo_cells_ += cells;
        joint_memo_.emplace(std::move(key), MemoEntry{out, w.work});
    }
    return true;
}

}  // namespace qeccore
