// born_law_coset — the coset-of-X-span Born law (see qeccore/born_law_coset.hpp for the derivation).
#include "qeccore/born_law_coset.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "qeccore/amplitudes.hpp"

namespace qeccore {

namespace {

// ── double-double (the per-coset sums' precision margin) ────────────────────────────────────────────────────────
struct dd { double hi = 0.0, lo = 0.0; };
inline dd two_sum(double a, double b) {
    const double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
inline dd quick_two_sum(double a, double b) { const double s = a + b; return {s, b - (s - a)}; }
inline dd two_prod(double a, double b) { const double p = a * b; return {p, std::fma(a, b, -p)}; }
inline dd dd_add(dd x, dd y) {
    dd s = two_sum(x.hi, y.hi);
    const dd t = two_sum(x.lo, y.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}
inline dd dd_sub(dd x, dd y) { return dd_add(x, dd{-y.hi, -y.lo}); }
inline dd dd_mul(dd x, dd y) {
    dd p = two_prod(x.hi, y.hi);
    p.lo += x.hi * y.lo + x.lo * y.hi;
    return quick_two_sum(p.hi, p.lo);
}
inline double to_d(dd x) { return x.hi + x.lo; }
struct cdd { dd re, im; };

// In-place unnormalised Walsh–Hadamard transform (Σ_t (−1)^{z·t} v_t).
template <class T, class Add, class Sub>
void wht(std::vector<T>& v, Add add, Sub sub) {
    const size_t n = v.size();
    for (size_t h = 1; h < n; h <<= 1)
        for (size_t i = 0; i < n; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) {
                const T x = v[j], y = v[j + h];
                v[j] = add(x, y);
                v[j + h] = sub(x, y);
            }
}
inline cdd cadd(const cdd& x, const cdd& y) { return {dd_add(x.re, y.re), dd_add(x.im, y.im)}; }
inline cdd csub(const cdd& x, const cdd& y) { return {dd_sub(x.re, y.re), dd_sub(x.im, y.im)}; }

inline int par(uint64_t x) { return __builtin_parityll(x); }
inline uint64_t packbits(const std::vector<uint8_t>& v, int k) {
    uint64_t m = 0;
    for (int i = 0; i < k && i < 64; ++i) if (v[(size_t)i] & 1) m |= 1ull << i;
    return m;
}
// i^e · (re + i·im) as an exact double pair.
inline cdd rot(double re, double im, int e) {
    switch (e & 3) {
        case 0: return {dd{re, 0.0}, dd{im, 0.0}};
        case 1: return {dd{-im, 0.0}, dd{re, 0.0}};
        case 2: return {dd{-re, 0.0}, dd{-im, 0.0}};
        default: return {dd{im, 0.0}, dd{-re, 0.0}};
    }
}
// Re conj(x)·y for exact double complex x, y, in double-double.
inline dd re_conj_prod(double xr, double xi, double yr, double yi) { return dd_add(two_prod(xr, yr), two_prod(xi, yi)); }
inline dd re_conj_prod(const cdd& x, const cdd& y) { return dd_add(dd_mul(x.re, y.re), dd_mul(x.im, y.im)); }
// conj(x)·y in double-double (exact double inputs or dd inputs).
inline cdd conj_prod(double xr, double xi, double yr, double yi) {
    return {dd_add(two_prod(xr, yr), two_prod(xi, yi)), dd_sub(two_prod(xr, yi), two_prod(xi, yr))};
}
inline cdd conj_prod(const cdd& x, const cdd& y) {
    return {dd_add(dd_mul(x.re, y.re), dd_mul(x.im, y.im)), dd_sub(dd_mul(x.re, y.im), dd_mul(x.im, y.re))};
}
// Re(w · i^e).
inline dd rot_re(const cdd& w, int e) {
    switch (e & 3) {
        case 0: return w.re;
        case 1: return dd{-w.im.hi, -w.im.lo};
        case 2: return dd{-w.re.hi, -w.re.lo};
        default: return w.im;
    }
}

// The route's cost model (model nanoseconds; calibrated on the merged tip, see STATUS): one BornTrie entry-pass of
// the model Σ_nodes χ_node, one coordinate read, one double-double butterfly / pair product.
constexpr double kEnumNs = 150.0;
constexpr double kPassNs = 25.0;
constexpr double kDdNs = 12.0;

}  // namespace

// ── steps A–B ─────────────────────────────────────────────────────────────────────────────────────────────────────

CosetLaw::Dir CosetLaw::dir_of(const FreeDir& d, int k) {
    if ((int)d.a.size() != k || (int)d.b.size() != k)
        throw std::logic_error("born_law_coset: a direction's width differs from the free register (engine bug)");
    Dir o;
    o.a = packbits(d.a, k);
    o.b = packbits(d.b, k);
    o.q = (d.phase + 2 * (d.par & 1)) & 3;
    if (((o.q - __builtin_popcountll(o.a & o.b)) & 1) != 0)
        throw std::logic_error("born_law_coset: a non-Hermitian direction (engine bug)");
    return o;
}

CosetLaw::Dir CosetLaw::product(uint64_t comb) const {
    Dir p;
    for (int j = 0; j < d_; ++j)
        if ((comb >> j) & 1) {
            const Dir& e = dirs_[(size_t)j];
            p.q = (p.q + e.q + 2 * par(p.b & e.a)) & 3;
            p.a ^= e.a;
            p.b ^= e.b;
        }
    return p;
}

uint64_t CosetLaw::coords(uint64_t x, uint64_t& t) const {
    t = 0;
    for (int i = 0; i < dX_; ++i)
        if ((x >> piv_[(size_t)i]) & 1) { x ^= alpha_[(size_t)i]; t |= 1ull << i; }
    return x;
}

uint64_t CosetLaw::class_of(uint64_t x0) const {
    uint64_t e = 0;
    for (int j = 0; j < dZ_; ++j) {
        const Dir& z = pureZ_[(size_t)j];
#ifndef XTIM_MUTATE_EXPLAW_EPSSIGN
        const int bit = ((z.q >> 1) ^ par(z.b & x0)) & 1;
#else
        const int bit = (z.q >> 1) & 1;                          // mutation build only: the coset sign dropped
#endif
        e |= (uint64_t)bit << j;
    }
    return e;
}

uint64_t CosetLaw::sigma_of(uint64_t x0) const {
    uint64_t s = 0;
    for (int i = 0; i < dX_; ++i) {
        const Dir& m = mixed_[(size_t)i];
        const int bit = ((((m.q - B(i, i)) & 3) >> 1) ^ par(m.b & x0)) & 1;
        s |= (uint64_t)bit << i;
    }
    return s;
}

CosetLaw::CosetLaw(const FramedSuperposition& bare, const std::vector<FreeDir>& lam) : bare_(bare) {
    k_ = bare.k();
    d_ = (int)lam.size();
    if (d_ > 63) throw std::logic_error("born_law_coset: more than 63 directions (engine bug)");
    if (k_ > 64) throw std::logic_error("born_law_coset: a free register wider than one key word (engine bug)");
    da_ = dynamic_cast<const DenseAmplitudes*>(bare.alpha.get());
    if (!da_) throw std::logic_error("born_law_coset: the bare register is not a DenseAmplitudes (engine bug)");
    dirs_.reserve((size_t)d_);
    for (const FreeDir& d : lam) dirs_.push_back(dir_of(d, k_));
    for (int i = 0; i < d_; ++i)
        for (int j = 0; j < i; ++j)
            if (par(dirs_[(size_t)i].a & dirs_[(size_t)j].b) ^ par(dirs_[(size_t)j].a & dirs_[(size_t)i].b))
                throw std::logic_error("born_law_coset: two directions anticommute (engine bug)");
    // A. elimination of the X-parts (lowest-bit pivots, fully reduced), combinations tracked.
    std::vector<uint64_t> mixed_comb, pure_comb;
    for (int j = 0; j < d_; ++j) {
        uint64_t v = dirs_[(size_t)j].a, cm = 1ull << j;
        for (size_t i = 0; i < alpha_.size(); ++i)
            if ((v >> piv_[i]) & 1) { v ^= alpha_[i]; cm ^= mixed_comb[i]; }
        if (!v) { pure_comb.push_back(cm); continue; }
        const int pv = __builtin_ctzll(v);
        for (size_t i = 0; i < alpha_.size(); ++i)
            if ((alpha_[i] >> pv) & 1) { alpha_[i] ^= v; mixed_comb[i] ^= cm; }
        alpha_.push_back(v);
        piv_.push_back(pv);
        mixed_comb.push_back(cm);
    }
    dX_ = (int)alpha_.size();
    dZ_ = d_ - dX_;
    for (uint64_t cm : mixed_comb) { comb_.push_back(cm); mixed_.push_back(product(cm)); }
    for (uint64_t cm : pure_comb) { comb_.push_back(cm); pureZ_.push_back(product(cm)); }
    for (int i = 0; i < dX_; ++i)
        if (mixed_[(size_t)i].a != alpha_[(size_t)i])
            throw std::logic_error("born_law_coset: a mixed product's X-part is not its pivot row (engine bug)");
    B_.assign((size_t)dX_ * (size_t)dX_, 0);
    for (int i = 0; i < dX_; ++i)
        for (int l = 0; l < dX_; ++l) B_[(size_t)i * (size_t)dX_ + (size_t)l] = (uint8_t)par(mixed_[(size_t)i].b & alpha_[(size_t)l]);
    for (int i = 0; i < dX_; ++i)
        for (int l = 0; l < i; ++l)
            if (B(i, l) != B(l, i)) throw std::logic_error("born_law_coset: the coset form is not symmetric (engine bug)");
    // B. the coset census of the support.
    const uint64_t low = key_low_mask(k_);
    ent_.reserve(da_->b_.size());
    for (size_t e = 0; e < da_->b_.size(); ++e) {
        uint64_t t;
        const uint64_t x0 = coords(da_->b_[e].first & low, t);
        ent_.push_back({x0, t, (uint32_t)e});
    }
    std::sort(ent_.begin(), ent_.end(), [](const Ent& a, const Ent& b) {
        return a.x0 != b.x0 ? a.x0 < b.x0 : a.t < b.t;
    });
    for (size_t b = 0; b < ent_.size();) {
        size_t e = b;
        while (e < ent_.size() && ent_[e].x0 == ent_[b].x0) ++e;
        cos_.push_back({ent_[b].x0, b, e});
        b = e;
    }
    // The route estimate.
    const double s = (double)ent_.size(), n = std::ldexp(1.0, dX_), cosets = (double)cos_.size();
    std::unordered_map<uint64_t, int> cls;
    double inner = 0.0;
    for (const Cos& C : cos_) {
        const double m = (double)(C.end - C.begin);
        inner += std::min(m * m, (double)dX_ * n);
        cls.emplace(class_of(C.x0), 0);
    }
    inner += (double)cls.size() * (double)dX_ * n + 2.0 * (double)d_ * std::ldexp(1.0, d_);
    route_.d = d_;
    route_.dX = dX_;
    route_.s = (long)ent_.size();
    route_.cosets = (long)cos_.size();
    route_.classes = (long)cls.size();
    route_.cost_enum = kEnumNs * ((double)dZ_ * s + (n - 1.0) * cosets * n);
    route_.cost_coset = kPassNs * s * (double)d_ + kDdNs * inner;
    route_.coset = d_ >= 3 && route_.cost_coset < route_.cost_enum;
}

// ── steps C–D ─────────────────────────────────────────────────────────────────────────────────────────────────────

void CosetLaw::build(const FreeDir* X, CosetLawTable& out) const {
    const size_t n = (size_t)1 << dX_, total = (size_t)1 << d_;
    out.d = d_; out.dX = dX_; out.dZ = dZ_;
    out.support = (long)ent_.size();
    out.cosets = (long)cos_.size();
    out.classes = out.dense_pairs = out.sparse_pairs = 0;
    out.work = (double)ent_.size() * (double)d_;
    out.column = X != nullptr;
    // The column: a_0 = red ⊕ T(t_a) (red = its coset representative), r_i = ⟨red, β_i⟩, κ = c_0 · conj(D_{t_a t_a}).
    uint64_t red = 0, ta = 0, r = 0, b0 = 0;
    int qk = 0;                                               // κ = i^qk · (−1)^{⟨b_0, x0'⟩}: real iff a_0 ∈ A
    if (X) {
        const Dir x0d = dir_of(*X, k_);
        for (const Dir& e : dirs_)
            if (par(x0d.a & e.b) ^ par(e.a & x0d.b))
                throw std::logic_error("born_law_coset: the column anticommutes with a direction (engine bug)");
        red = coords(x0d.a, ta);
        b0 = x0d.b;
        for (int i = 0; i < dX_; ++i) r |= (uint64_t)par(red & mixed_[(size_t)i].b) << i;
        int ep = 0;                                           // e'(t_a) = 2 Σ_{i<l} B_il t_i t_l + Σ_i B_ii t_i
        for (int i = 0; i < dX_; ++i)
            if ((ta >> i) & 1) {
                ep += B(i, i);
                for (int l = i + 1; l < dX_; ++l) if ((ta >> l) & 1) ep += 2 * B(i, l);
            }
        qk = (x0d.q - ep) & 3;
        if (!red && (qk & 1))
            throw std::logic_error("born_law_coset: a diagonal column's coset action is not Hermitian (engine bug)");
    }
    // D†'s diagonal phase i^{ph[t]}, ph = 2 Σ_{i<l} B_il t_i t_l + 3 Σ_i B_ii t_i (incremental over the lowest set bit).
    std::vector<uint8_t> ph(n, 0);
    for (size_t t = 1; t < n; ++t) {
        const int i0 = __builtin_ctzll(t);
        int e = ph[t & (t - 1)] + 3 * B(i0, i0);
        for (uint64_t rest = t & (t - 1); rest; rest &= rest - 1) e += 2 * B(i0, __builtin_ctzll(rest));
        ph[t] = (uint8_t)(e & 3);
    }
    std::vector<double> Pp(total, 0.0), Np(X ? total : 0, 0.0);
    // Sparse accumulators per class: E_ε(S) (P) and F_ε(S) (N).
    std::unordered_map<uint64_t, size_t> cls_index;
    std::vector<std::vector<dd>> E, F;
    std::vector<uint64_t> cls_eps;
    auto cls_acc = [&](uint64_t eps) {
        auto it = cls_index.find(eps);
        if (it != cls_index.end()) return it->second;
        const size_t id = E.size();
        cls_index.emplace(eps, id);
        cls_eps.push_back(eps);
        E.emplace_back(n, dd{});
        if (X) F.emplace_back(n, dd{});
        return id;
    };
    std::vector<cdd> bufA, bufB;
    auto fill = [&](const Cos& C, std::vector<cdd>& buf) {
        buf.assign(n, cdd{});
        for (size_t e = C.begin; e < C.end; ++e) {
            const cd a = da_->b_[ent_[e].idx].second;
            buf[ent_[e].t] = rot(a.real(), a.imag(), ph[ent_[e].t]);
        }
    };
    auto find_coset = [&](uint64_t x0) -> long {
        const auto it = std::lower_bound(cos_.begin(), cos_.end(), x0,
                                         [](const Cos& c, uint64_t v) { return c.x0 < v; });
        if (it == cos_.end() || it->x0 != x0) return -1;
        return (long)(it - cos_.begin());
    };
    // ω_C(z) = i^qk (−1)^{⟨b_0, x0'⟩ ⊕ ⟨t_a, z⟩}, x0' the partner's representative: wsign is its sign bit.
    auto wsign = [&](uint64_t x0p) {
#ifndef XTIM_MUTATE_EXPLAW_SWAPPHASE
        return par(b0 & x0p);
#else
        return red ? 0 : par(b0 & x0p);                       // mutation build only: the swap's sign dropped
#endif
    };
    const double dX_n = (double)dX_ * (double)n;
    std::vector<uint8_t> done(cos_.size(), 0);
    for (size_t ci = 0; ci < cos_.size(); ++ci) {
        if (done[ci]) continue;
        const Cos& C = cos_[ci];
        const uint64_t eps = class_of(C.x0), sC = sigma_of(C.x0);
        const uint64_t base = eps << dX_;
        long pj = (long)ci;
        if (X && red) pj = find_coset(C.x0 ^ red);
        const size_t m = C.end - C.begin;
        if (pj == (long)ci || pj < 0) {
            // One coset: a_0 ∈ A (the column is diagonal on it) or the partner coset carries no amplitude.
            const bool selfN = X && pj == (long)ci;
            const int w = selfN ? wsign(C.x0) : 0;
            const double sparse_cost = 0.5 * (double)m * (double)(m + 1) + (selfN ? (double)m * (double)m : 0.0);
            const double dense_cost = ((double)dX_ + 2.0) * (double)n;
            if (sparse_cost < dense_cost) {
                ++out.sparse_pairs;
                out.work += sparse_cost;
                const size_t id = cls_acc(eps);
                for (size_t e = C.begin; e < C.end; ++e) {
                    const cdd x = rot(da_->b_[ent_[e].idx].second.real(), da_->b_[ent_[e].idx].second.imag(), ph[ent_[e].t]);
                    const uint64_t t = ent_[e].t;
                    E[id][0] = dd_add(E[id][0], re_conj_prod(x.re.hi, x.im.hi, x.re.hi, x.im.hi));
                    for (size_t f = e + 1; f < C.end; ++f) {
                        const cdd y = rot(da_->b_[ent_[f].idx].second.real(), da_->b_[ent_[f].idx].second.imag(), ph[ent_[f].t]);
                        const uint64_t S = t ^ ent_[f].t;
                        dd p = re_conj_prod(x.re.hi, x.im.hi, y.re.hi, y.im.hi);
                        p = dd_add(p, p);
                        if (par(sC & S)) p = dd{-p.hi, -p.lo};
                        E[id][S] = dd_add(E[id][S], p);
                    }
                    if (selfN)
                        for (size_t f = C.begin; f < C.end; ++f) {
                            const cdd y = rot(da_->b_[ent_[f].idx].second.real(), da_->b_[ent_[f].idx].second.imag(), ph[ent_[f].t]);
                            const uint64_t S = t ^ ent_[f].t ^ ta;
                            dd p = rot_re(conj_prod(x.re.hi, x.im.hi, y.re.hi, y.im.hi), qk);
                            if ((w ^ par(sC & S)) & 1) p = dd{-p.hi, -p.lo};
                            F[id][S] = dd_add(F[id][S], p);
                        }
                }
            } else {
                ++out.dense_pairs;
                out.work += dense_cost;
                fill(C, bufA);
                wht(bufA, cadd, csub);
                for (size_t z = 0; z < n; ++z) {
                    const double pc = to_d(re_conj_prod(bufA[z], bufA[z]));
                    const size_t at = base | (z ^ sC);
                    Pp[at] += pc;
                    if (selfN) Np[at] += ((w ^ par(ta & z) ^ (qk >> 1)) & 1) ? -pc : pc;
                }
            }
            done[ci] = 1;
            continue;
        }
        // A coset pair C ↔ C' = C ⊕ a_0 (a_0 ∉ A), both carrying amplitude: the same class, σ(C') = σ(C) ⊕ r.
        const Cos& Cp = cos_[(size_t)pj];
        const uint64_t sCp = sigma_of(Cp.x0);
        if (class_of(Cp.x0) != eps || sCp != (sC ^ r))
            throw std::logic_error("born_law_coset: the column's coset swap changes the class or the sign vector (engine bug)");
        const size_t mp = Cp.end - Cp.begin;
        const int wC = wsign(Cp.x0), wCp = wsign(C.x0);
        const double sparse_cost = 0.5 * (double)m * (double)(m + 1) + 0.5 * (double)mp * (double)(mp + 1) +
                                   2.0 * (double)m * (double)mp;
        const double dense_cost = 2.0 * ((double)dX_ + 2.0) * (double)n;
        if (sparse_cost < dense_cost) {
            ++out.sparse_pairs;
            out.work += sparse_cost;
            const size_t id = cls_acc(eps);
            auto self_pairs = [&](const Cos& Q, uint64_t sQ) {
                for (size_t e = Q.begin; e < Q.end; ++e) {
                    const cdd x = rot(da_->b_[ent_[e].idx].second.real(), da_->b_[ent_[e].idx].second.imag(), ph[ent_[e].t]);
                    E[id][0] = dd_add(E[id][0], re_conj_prod(x.re.hi, x.im.hi, x.re.hi, x.im.hi));
                    for (size_t f = e + 1; f < Q.end; ++f) {
                        const cdd y = rot(da_->b_[ent_[f].idx].second.real(), da_->b_[ent_[f].idx].second.imag(), ph[ent_[f].t]);
                        const uint64_t S = ent_[e].t ^ ent_[f].t;
                        dd p = re_conj_prod(x.re.hi, x.im.hi, y.re.hi, y.im.hi);
                        p = dd_add(p, p);
                        if (par(sQ & S)) p = dd{-p.hi, -p.lo};
                        E[id][S] = dd_add(E[id][S], p);
                    }
                }
            };
            // F_ε(S) += w_Q (−1)^{σ_Q·S} (−1)^{r·u} Re conj(φ'_{Q,t}) φ'_{Q',u},  S = t ⊕ u ⊕ t_a  (Q receives from Q').
            auto cross_pairs = [&](const Cos& Q, uint64_t sQ, int wQ, const Cos& Qp) {
                for (size_t e = Q.begin; e < Q.end; ++e) {
                    const cdd x = rot(da_->b_[ent_[e].idx].second.real(), da_->b_[ent_[e].idx].second.imag(), ph[ent_[e].t]);
                    for (size_t f = Qp.begin; f < Qp.end; ++f) {
                        const cdd y = rot(da_->b_[ent_[f].idx].second.real(), da_->b_[ent_[f].idx].second.imag(), ph[ent_[f].t]);
                        const uint64_t u = ent_[f].t, S = ent_[e].t ^ u ^ ta;
                        dd p = rot_re(conj_prod(x.re.hi, x.im.hi, y.re.hi, y.im.hi), qk);
                        if ((wQ ^ par(sQ & S) ^ par(r & u)) & 1) p = dd{-p.hi, -p.lo};
                        F[id][S] = dd_add(F[id][S], p);
                    }
                }
            };
            self_pairs(C, sC);
            self_pairs(Cp, sCp);
            cross_pairs(C, sC, wC, Cp);
            cross_pairs(Cp, sCp, wCp, C);
        } else {
            ++out.dense_pairs;
            out.work += dense_cost;
            fill(C, bufA);
            fill(Cp, bufB);
            wht(bufA, cadd, csub);
            wht(bufB, cadd, csub);
            for (size_t z = 0; z < n; ++z) {
                Pp[base | (z ^ sC)] += to_d(re_conj_prod(bufA[z], bufA[z]));
                Pp[base | (z ^ sCp)] += to_d(re_conj_prod(bufB[z], bufB[z]));
                const int tz = par(ta & z);
                const double n1 = to_d(rot_re(conj_prod(bufA[z], bufB[z ^ r]), qk));   // C receives from C'
                const double n2 = to_d(rot_re(conj_prod(bufB[z], bufA[z ^ r]), qk));   // C' receives from C
                Np[base | (z ^ sC)] += ((wC ^ tz) & 1) ? -n1 : n1;
                Np[base | (z ^ sCp)] += ((wCp ^ tz) & 1) ? -n2 : n2;
            }
        }
        done[ci] = done[(size_t)pj] = 1;
    }
    // The classes' WHTs (sparse route), then the 1/n scale and the re-index to the caller's direction order.
    out.work += (double)E.size() * (double)dX_ * (double)n * (X ? 2.0 : 1.0) + (double)d_ * (double)total;
    for (size_t id = 0; id < E.size(); ++id) {
        const size_t base = cls_eps[id] << dX_;
        wht(E[id], dd_add, dd_sub);
        for (size_t z = 0; z < n; ++z) Pp[base | z] += to_d(E[id][z]);
        if (X) {
            wht(F[id], dd_add, dd_sub);
            for (size_t z = 0; z < n; ++z) Np[base | z] += to_d(F[id][z]);
        }
    }
    {
        std::unordered_map<uint64_t, int> seen;
        for (const Cos& C : cos_) seen.emplace(class_of(C.x0), 0);
        out.classes = (long)seen.size();
    }
    const double inv = 1.0 / (double)n;
    out.P.assign(total, 0.0);
    if (X) out.N.assign(total, 0.0); else out.N.clear();
    for (size_t z = 0; z < total; ++z) {
        uint64_t zp = 0;
        for (int i = 0; i < d_; ++i) zp |= (uint64_t)par(comb_[(size_t)i] & z) << i;
        out.P[z] = Pp[zp] * inv;
        if (X) out.N[z] = Np[zp] * inv;
    }
}

// ── step E ────────────────────────────────────────────────────────────────────────────────────────────────────────

double coset_law_value(const CosetLawTable& T, uint64_t z) {
    if (!T.column) return 0.0;
    const double p = T.P[(size_t)z];
    if (!(p > 0.0)) return 0.0;
    double h = T.N[(size_t)z] / p;
    if (h < -1.0) h = -1.0;
    if (h > 1.0) h = 1.0;
    return h;
}

void coset_law_leaves(const CosetLawTable& T, double eps, const std::function<bool(uint64_t, double, double)>& leaf,
                      long* evals) {
    const int d = T.d;
    const std::vector<double>& P = T.P;
#ifdef XTIM_MUTATE_EXPLAW_REPLAY
    eps = 0.0;                                                // mutation build only: the deterministic rule dropped
#endif
    // The mass of the prefix (bits < j of f fixed) with bit j fixed to `bit`: a direct sum over the suffix.
    auto side = [&](int j, uint64_t f, int bit) {
        const uint64_t pre = (f & ((1ull << j) - 1)) | ((uint64_t)bit << j);
        double s = 0.0;
        const size_t step = (size_t)1 << (j + 1);
        for (size_t z = pre; z < P.size(); z += step) s += P[z];
        return s;
    };
    bool stop = false;
    long ev = 0;
    std::function<void(int, uint64_t, double)> rec = [&](int j, uint64_t f, double p) {
        while (j < d) {
            const double plus = side(j, f, 0), minus = side(j, f, 1), tot = plus + minus;
            double pp = tot > 0.0 ? plus / tot : 0.5;
            if (pp < 0.0) pp = 0.0;
            if (pp > 1.0) pp = 1.0;
            const double pm = 1.0 - pp;
            if (pp > eps && pm > eps) {
                rec(j + 1, f, p * pp);
                if (stop) return;
                rec(j + 1, f | (1ull << j), p * pm);
                return;
            }
            if (!(pp > eps)) f |= 1ull << j;
            ++j;
        }
        ++ev;
        if (!leaf(f, p, coset_law_value(T, f))) stop = true;
    };
    rec(0, 0, 1.0);
    if (evals) *evals += ev;
}

void born_law_enumerate(const FramedSuperposition& bare, const std::vector<FreeDir>& lam, const FreeDir* X, double eps,
                        const std::function<bool(uint64_t, double, double)>& leaf, long* evals,
                        std::shared_ptr<const CosetLawTable>* table_out, BornRoute* route_out) {
    if (table_out && *table_out) {                           // a built table of this content: replay only
        const std::shared_ptr<const CosetLawTable>& T = *table_out;
        if (T->d != (int)lam.size() || T->column != (X != nullptr))
            throw std::logic_error("born_law_enumerate: a passed-in table does not fit the directions (engine bug)");
        if (route_out) { *route_out = BornRoute{}; route_out->coset = true; route_out->d = T->d; route_out->dX = T->dX; route_out->s = T->support; route_out->cosets = T->cosets; route_out->classes = T->classes; }
        long ev = 0;
        coset_law_leaves(*T, eps, leaf, &ev);
        if (evals) *evals += ev + T->cosets;
        return;
    }
    if (lam.size() >= 3) {
        const CosetLaw cl(bare, lam);
        if (route_out) *route_out = cl.route();
#ifdef XTIM_MUTATE_EXPLAW_FORCE_ENUM
        if (false) {                                          // mutation build only: the coset law never taken
#else
        if (cl.route().coset) {
#endif
            auto T = std::make_shared<CosetLawTable>();
            cl.build(X, *T);
            long ev = 0;
            coset_law_leaves(*T, eps, leaf, &ev);
            if (evals) *evals += ev + T->cosets;
            if (table_out) *table_out = std::move(T);
            return;
        }
    } else if (route_out) {
        *route_out = BornRoute{};
        route_out->d = (int)lam.size();
        route_out->s = bare.chi();
    }
    BornTrie bt(bare);
    bt.enumerate(lam, X, eps, leaf, evals);
}

}  // namespace qeccore
