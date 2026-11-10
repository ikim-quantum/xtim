// Per-shot law of a PAULI_EXPECTATION column for DEM export (3.1.12). See dem_pe_per_shot.hpp
// for the derivation; every quantity here is exact (GF(2) linear algebra + Born sums on the bare
// state through the engine's own framed_expectation).
#include "qeccore/dem_pe_per_shot.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "qeccore/twirl_kernel.hpp"

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
                               const std::vector<std::pair<int, int>>& reads)
    : L_(bare), reads_(reads) {
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
    // Bit rows over the changed reads J (and over the pivots, at most nJ of them): JW words each, kept in flat
    // row-major arrays (js_: no per-row or per-call allocation); pivots are lowest bits, as with one word.
    const int JW = std::max(1, (nJ + 63) / 64);
    auto bit = [](const uint64_t* v, int i) { return ((v[i >> 6] >> (i & 63)) & 1) != 0; };
    auto flip = [](uint64_t* v, int i) { v[i >> 6] ^= 1ull << (i & 63); };
    auto xorw = [JW](uint64_t* a, const uint64_t* b) { for (int w = 0; w < JW; ++w) a[w] ^= b[w]; };
    auto anyw = [JW](const uint64_t* v) { for (int w = 0; w < JW; ++w) if (v[w]) return true; return false; };
    auto loww = [JW](const uint64_t* v) {
        for (int w = 0; w < JW; ++w) if (v[w]) return w * 64 + __builtin_ctzll(v[w]);
        return -1;
    };
    auto row = [JW](std::vector<uint64_t>& m, size_t i) { return m.data() + i * (size_t)JW; };
    auto append = [](std::vector<uint64_t>& m, const std::vector<uint64_t>& v) { m.insert(m.end(), v.begin(), v.end()); };
    auto jbits = [&](const std::vector<uint64_t>& comb, uint64_t* p) {
        std::fill(p, p + JW, 0);
        for (int i = 0; i < nJ; ++i) if (pe_bit(comb, J[(size_t)i])) flip(p, i);
    };
    // ── Projection of the noiseless nullspace W onto J: pivots + the kernel W_J. ──
    struct Piv { int pv; std::vector<uint64_t> comb, phi; };
    std::vector<Piv> piv;
    std::vector<uint64_t>& piv_proj = js_.piv_proj;    // row i: pivot i's projection onto J
    piv_proj.clear();
    struct Kern { int e; std::vector<uint64_t> phi; };
    std::vector<Kern> kern;
    std::vector<uint64_t>& kern_used = js_.kern_used;  // row i: kernel element i's pivot combination
    kern_used.clear();
    auto reduce_proj = [&](uint64_t* p, uint64_t* used) {
        for (size_t i = 0; i < piv.size(); ++i)
            if (bit(p, piv[i].pv)) { xorw(p, row(piv_proj, i)); flip(used, (int)i); }
    };
    std::vector<uint64_t>& p = js_.p;
    std::vector<uint64_t>& used = js_.used;
    p.assign((size_t)JW, 0);
    used.assign((size_t)JW, 0);
    for (int e = 0; e < (int)null_.size(); ++e) {
        jbits(null_[(size_t)e].comb, p.data());
        std::fill(used.begin(), used.end(), 0);
        const bool hits_j = anyw(p.data());
        if (hits_j) reduce_proj(p.data(), used.data());
        std::vector<uint64_t> ph = null_[(size_t)e].phi;
        if (hits_j)
            for (size_t i = 0; i < piv.size(); ++i) if (bit(used.data(), (int)i)) pe_xor(ph, piv[i].phi);
        if (hits_j && anyw(p.data())) {
            std::vector<uint64_t> cb = null_[(size_t)e].comb;
            for (size_t i = 0; i < piv.size(); ++i) if (bit(used.data(), (int)i)) pe_xor(cb, piv[i].comb);
            piv.push_back({loww(p.data()), std::move(cb), std::move(ph)});
            append(piv_proj, p);
        } else if (k_ > 0) {
            kern.push_back({e, std::move(ph)});
            append(kern_used, used);
        }
    }
    auto w_comb = [&](const uint64_t* u, std::vector<uint64_t>& cb) {
        for (size_t i = 0; i < piv.size(); ++i) if (bit(u, (int)i)) pe_xor(cb, piv[i].comb);
    };
    auto w_phi = [&](const uint64_t* u, std::vector<uint64_t>& ph) {
        for (size_t i = 0; i < piv.size(); ++i) if (bit(u, (int)i)) pe_xor(ph, piv[i].phi);
    };
    // ── The changed reads: reduce each world row against the noiseless echelon. ──
    struct JRow { std::vector<uint64_t> r, C, phiC, phiA; };
    std::vector<JRow> jr((size_t)nJ);
    std::vector<uint64_t>& jr_pi = js_.jr_pi;
    std::vector<uint64_t>& jr_used = js_.jr_used;
    jr_pi.assign((size_t)nJ * JW, 0);
    jr_used.assign((size_t)nJ * JW, 0);
    std::vector<uint64_t> x;
    for (int i = 0; i < nJ; ++i) {
        Pauli A = read_pauli(J[(size_t)i]);
        xv(A, x);
        JRow& rw = jr[(size_t)i];
        phi_of(A, x, rw.phiA);
        rw.r.assign((size_t)NW_, 0);
        for (int w = 0; w < NW_; ++w) rw.r[(size_t)w] = x[(size_t)w] & nfm_[(size_t)w];
        rw.C.assign((size_t)MW_, 0);
        rw.phiC.assign((size_t)FW_, 0);
        reduce(rw.r, rw.C, rw.phiC);
        jbits(rw.C, row(jr_pi, (size_t)i));
        reduce_proj(row(jr_pi, (size_t)i), row(jr_used, (size_t)i));
    }
    // Elimination of the J system rows (r_i | pi_i), tracking the u-combination.
    struct ERow { std::vector<uint64_t> r; int pr_piv, pi_piv; };
    std::vector<ERow> er;
    std::vector<uint64_t>& er_pi = js_.er_pi;          // row k: ERow k's pi / u
    std::vector<uint64_t>& er_u = js_.er_u;
    std::vector<uint64_t>& nullu = js_.nullu;          // homogeneous solutions u (rows)
    er_pi.clear();
    er_u.clear();
    nullu.clear();
    auto reduce_e = [&](std::vector<uint64_t>& r, uint64_t* pi, uint64_t* u) {
        for (size_t k = 0; k < er.size(); ++k) {
            const ERow& e = er[k];
            const bool hit = e.pr_piv >= 0 ? pe_bit(r, e.pr_piv) : bit(pi, e.pi_piv);
            if (hit) { pe_xor(r, e.r); xorw(pi, row(er_pi, k)); xorw(u, row(er_u, k)); }
        }
    };
    std::vector<uint64_t>& pi = js_.pi;
    std::vector<uint64_t>& u = js_.u;
    pi.assign((size_t)JW, 0);
    u.assign((size_t)JW, 0);
    for (int i = 0; i < nJ; ++i) {
        std::vector<uint64_t> r = jr[(size_t)i].r;
        std::copy(row(jr_pi, (size_t)i), row(jr_pi, (size_t)i) + JW, pi.begin());
        std::fill(u.begin(), u.end(), 0);
        flip(u.data(), i);
        reduce_e(r, pi.data(), u.data());
        const int pv = pe_lowbit(r);
        if (pv >= 0 || anyw(pi.data())) {
            er.push_back({std::move(r), pv, pv >= 0 ? -1 : loww(pi.data())});
            append(er_pi, pi);
            append(er_u, u);
        } else {
            append(nullu, u);
        }
    }
    // Read set / phi of the world-nullspace element S_u (u a combination of J rows) plus an
    // optional particular right-hand side (C0, used0).
    std::vector<uint64_t>& s_used = js_.s_used;
    s_used.assign((size_t)JW, 0);
    auto s_of = [&](const uint64_t* uu, const std::vector<uint64_t>* C0, const uint64_t* used0,
                    std::vector<uint64_t>& cb, std::vector<uint64_t>* ph) {
        cb.assign((size_t)MW_, 0);
        if (C0) pe_xor(cb, *C0);
        if (used0) std::copy(used0, used0 + JW, s_used.begin());
        else std::fill(s_used.begin(), s_used.end(), 0);
        if (ph) ph->assign((size_t)FW_, 0);
        for (int i = 0; i < nJ; ++i)
            if (bit(uu, i)) {
                pe_flip(cb, J[(size_t)i]);
                pe_xor(cb, jr[(size_t)i].C);
                xorw(s_used.data(), row(jr_used, (size_t)i));
                if (ph) { pe_xor(*ph, jr[(size_t)i].phiA); pe_xor(*ph, jr[(size_t)i].phiC); }
            }
        w_comb(s_used.data(), cb);
        if (ph) w_phi(s_used.data(), *ph);
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
        jbits(CX, pi.data());           // pi / used / u reused as piX / usedX / u0
        std::fill(used.begin(), used.end(), 0);
        reduce_proj(pi.data(), used.data());
        std::fill(u.begin(), u.end(), 0);
        reduce_e(xX, pi.data(), u.data());
        if (pe_any(xX) || anyw(pi.data())) { law.zero = true; return law; }   // not in the span: v = 0 per shot
        law.coin = true;
        s_of(u.data(), &CX, used.data(), S0cb, nullptr);
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
        for (size_t ki = 0; ki < kern.size(); ++ki) {
            if ((int)basis.size() >= 2 * k_) break;
            add_gen(kern[ki].phi, [&](std::vector<uint64_t>& cb) {
                cb = null_[(size_t)kern[ki].e].comb;
                w_comb(row(kern_used, ki), cb);
            });
        }
        for (size_t ni = 0; ni * (size_t)JW < nullu.size(); ++ni) {
            std::vector<uint64_t> cb, ph;
            s_of(row(nullu, ni), nullptr, nullptr, cb, &ph);
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

// EXACT ENGINE ONLY (the forced-fault law's joint_law): an exact branch walk over commuting Hermitian Paulis B (in
// order) on a framed state: every outcome pattern of positive probability becomes a leaf; leaf(state, pattern, prob)
// is called on each and returns false to stop. Returns false when more than `budget` leaves would be produced.
// Work is a deterministic cost proxy: each born_p1 / measure_pauli / leaf read is charged
// (chi + 1) * n * (k + 1). Past the caller's work budget the walk stops after bounded work.
struct BranchWalk {
    const std::vector<Pauli>& B;
    long budget;
    const std::function<bool(const FramedSuperposition&, uint64_t, double)>& leaf;
    long leaves = 0;
    double work = 0.0;
    double work_budget = std::numeric_limits<double>::infinity();
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
    if (law.yP.empty()) {                       // no y bit: one leaf, no projection
        law.leaves.push_back({0, 1.0, L_.expectation(law.X0)});
        law.walked = !(stop && stop(law.leaves));       // same contract as the walk
        return true;
    }
    // twirl-routes T2: the y directions and the column on the bare FREE register (no non-free X part), the Born walk
    // there (the ONE routine: BornTrie projections or, when its cost model wins, the coset law — explaw), + outcome
    // first, a conditional within kBranchEps of 0 / 1 taken as deterministic (the engine's coin threshold, as the
    // full-state walk this replaced). A law copied from an already-walked one keeps its table (replay only).
    const int d = (int)law.yP.size();
    std::vector<FreeDir> lam;
    lam.reserve((size_t)d);
    for (const Pauli& Y : law.yP) lam.push_back(free_dir_of(L_, Y));
    const FreeDir Xd = free_dir_of(L_, law.X0);
    bool stopped = false;
    const bool had_table = (bool)law.table;
    born_law_enumerate(L_, lam, &Xd, kBranchEps, [&](uint64_t f, double p, double h) {
        law.leaves.push_back({f, p, h});
        if (stop && stop(law.leaves)) { stopped = true; return false; }
        return true;
    }, &born_evals_, &law.table);
    if (law.table && !had_table) { ++coset_walks_; coset_work_ += law.table->work; }
    law.walked = !stopped;
    return true;
}

double PeShotAnalyzer::value_at(const Law& law, uint64_t y) {
    if (law.zero) return 0.0;
    if (law.yP.empty()) return L_.expectation(law.X0);
    if (law.table) return coset_law_value(*law.table, y);   // explaw: the walked table's h(y)
    const int d = (int)law.yP.size();
    std::vector<FreeDir> lam;
    for (const Pauli& Y : law.yP) lam.push_back(free_dir_of(L_, Y));
    std::vector<uint8_t> z((size_t)d);
    for (int i = 0; i < d; ++i) z[(size_t)i] = (uint8_t)((y >> i) & 1);
    BornTrie bt(L_);
    ++born_evals_;
    return bt.value_from_scratch(lam, z.data(), d, free_dir_of(L_, law.X0));
}

void PeShotAnalyzer::frame_parts(const Pauli& P, std::vector<uint64_t>& xn,
                                 std::vector<uint64_t>& phi) const {
    xv(P, xn);
    phi_of(P, xn, phi);
    for (int w = 0; w < NW_; ++w) xn[(size_t)w] &= nfm_[(size_t)w];
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
