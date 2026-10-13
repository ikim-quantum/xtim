#include "qeccore/twirl_ppr.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <stdexcept>

// build_shot_law_ppr — FAITHFUL C++ port of scripts/twirl_ppr_reference.py::build_ppr_law
// (50/50 vs twirl_oracle.analytic_law at 1e-12), structured as the axis-space transport of
// build_shot_law (twirl_law.cpp). The GF(2) helper block is duplicated from twirl_law.cpp's
// anonymous namespace on purpose (byte-identity protocol: the diagonal builder is not touched;
// T3 productization may unify).

namespace qeccore {
namespace {

// ────────────────────────────── GF(2) bit-vector helpers ──────────────────────────────
using BV = std::vector<uint64_t>;
inline int nwords(int nbits) { return (nbits + 63) / 64; }
inline bool bget(const BV& v, int i) { return (v[i >> 6] >> (i & 63)) & 1ULL; }
inline void bset(BV& v, int i) { v[i >> 6] |= (1ULL << (i & 63)); }
inline void bxor(BV& a, const BV& b) { for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i]; }
inline bool bany(const BV& v) { for (uint64_t w : v) if (w) return true; return false; }
inline int bfirst(const BV& v) {
    for (size_t w = 0; w < v.size(); ++w) if (v[w]) return (int)w * 64 + __builtin_ctzll(v[w]);
    return -1;
}

struct Rref { std::vector<BV> R; std::vector<int> piv; };
Rref rref(const std::vector<BV>& rows) {
    Rref out;
    for (const BV& r : rows) {
        BV v = r;
        for (size_t k = 0; k < out.R.size(); ++k) if (bget(v, out.piv[k])) bxor(v, out.R[k]);
        int pc = bfirst(v);
        if (pc < 0) continue;
        for (size_t k = 0; k < out.R.size(); ++k) if (bget(out.R[k], pc)) bxor(out.R[k], v);
        out.R.push_back(std::move(v));
        out.piv.push_back(pc);
    }
    return out;
}

BV reduce_vec(BV v, const Rref& b) {
    for (size_t k = 0; k < b.R.size(); ++k) if (bget(v, b.piv[k])) bxor(v, b.R[k]);
    return v;
}
bool in_span(const BV& v, const Rref& b) { return !bany(reduce_vec(v, b)); }

std::vector<BV> nullspace(const std::vector<BV>& rows, int n) {
    Rref b = rref(rows);
    std::vector<char> is_piv(n, 0);
    for (int pc : b.piv) is_piv[pc] = 1;
    std::vector<BV> basis;
    for (int f = 0; f < n; ++f) {
        if (is_piv[f]) continue;
        BV x(nwords(n), 0);
        bset(x, f);
        for (size_t k = 0; k < b.R.size(); ++k) if (bget(b.R[k], f)) bset(x, b.piv[k]);
        basis.push_back(std::move(x));
    }
    return basis;
}

std::vector<BV> left_deps(const std::vector<BV>& rows, int /*n*/) {
    const int m = (int)rows.size();
    struct Ent { BV v; BV c; int pc; };
    std::vector<Ent> red;
    std::vector<BV> deps;
    for (int i = 0; i < m; ++i) {
        BV v = rows[i];
        BV c(nwords(m), 0);
        bset(c, i);
        for (const Ent& e : red) if (bget(v, e.pc)) { bxor(v, e.v); bxor(c, e.c); }
        int pc = bfirst(v);
        if (pc < 0) deps.push_back(std::move(c));
        else        red.push_back(Ent{std::move(v), std::move(c), pc});
    }
    return deps;
}

BV solve_gf2(const std::vector<BV>& A_rows, const std::vector<int>& b, int ncols,
             bool* feasible = nullptr) {
    if (feasible) *feasible = true;
    struct Ent { BV v; int rhs; int pc; };
    std::vector<Ent> red;
    for (size_t i = 0; i < A_rows.size(); ++i) {
        BV v = A_rows[i];
        int rr = b[i] & 1;
        for (const Ent& e : red) if (bget(v, e.pc)) { bxor(v, e.v); rr ^= e.rhs; }
        int pc = bfirst(v);
        if (pc < 0) {
            if (rr != 0) {
                if (feasible) { *feasible = false; return BV(nwords(ncols), 0); }
                throw std::logic_error("build_shot_law_ppr: inconsistent linear system");
            }
            continue;
        }
        for (Ent& e : red) if (bget(e.v, pc)) { bxor(e.v, v); e.rhs ^= rr; }
        red.push_back(Ent{std::move(v), rr, pc});
    }
    BV x(nwords(ncols), 0);
    for (const Ent& e : red) if (e.rhs) bset(x, e.pc);
    return x;
}

inline int bvdot(const BV& a, const BV& b) {
    int acc = 0;
    const size_t w = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < w; ++i) acc += __builtin_popcountll(a[i] & b[i]);
    return acc & 1;
}

// ─────────────────────────────── axis algebra helpers ──────────────────────────────────
// Canonicalize a rotation (e, A) to the canonical Hermitian axis (phase ∈ {0,1}); a −1 sign
// folds into the exponent. Non-Hermitian axis (odd i-power vs the canonical) is invalid.
bool canon_axis(int e, const Pauli& A, int n, std::pair<Pauli, int>& out) {
    Pauli H(n);
    H.x = A.x;
    H.z = A.z;
    const int p0 = A.xz_overlap() & 1;                    // canonical Hermitian phase
    H.phase = p0;
    const int d = ((A.phase - p0) % 4 + 4) % 4;
    if (d & 1) return false;                              // non-Hermitian axis
    out.first = std::move(H);
    out.second = ((d == 2 ? -e : e) % 8 + 8) % 8;
    return true;
}

inline bool axis_lt(const std::pair<Pauli, int>& a, const std::pair<Pauli, int>& b) {
    if (a.first.x != b.first.x) return a.first.x < b.first.x;
    return a.first.z < b.first.z;
}
inline bool axis_eq_xz(const Pauli& a, const Pauli& b) { return a.x == b.x && a.z == b.z; }

// i^{±1}·A as an exact Pauli (for the conjugation rule h ← h·(±i A)).
inline Pauli i_pow_axis(const Pauli& A, int ipow) {
    Pauli r = A;
    r.phase = (A.phase + ipow) & 3;
    return r;
}

}  // namespace

// ─────────────────────────────── composition ───────────────────────────────────────────
PprNormalForm compose_ppr(int n, const std::vector<PprAtom>& atoms) {
    PprNormalForm nf(n);
    std::vector<std::pair<Pauli, int>> rots;              // accumulated, canonical axes
    std::pair<Pauli, int> ca;
    for (const PprAtom& at : atoms) {
        const Pauli& P = *at.prefix;
        // move P left through the accumulated rotation factors: e flips where ⟨P,A⟩ = 1
        for (auto& ea : rots)
            if (Pauli::anticommute_bit(P, ea.first)) ea.second = (8 - ea.second) & 7;
        nf.prefix = Pauli::multiply(nf.prefix, P);
        for (const auto& ea : *at.rots) {
            if (!canon_axis(ea.second, ea.first, n, ca)) { nf.commuting_class = false; return nf; }
            rots.push_back(ca);
        }
    }
    // merge same axes (exponents add mod 8); linear scan — t is lightcone-sized
    std::vector<std::pair<Pauli, int>> merged;
    for (const auto& ea : rots) {
        bool found = false;
        for (auto& m : merged)
            if (axis_eq_xz(m.first, ea.first)) { m.second = (m.second + ea.second) & 7; found = true; break; }
        if (!found) merged.push_back(ea);
    }
    // commuting-class check on the distinct axes (BEFORE dropping e=0: an axis that cancels
    // mod 8 commutes with everything trivially, so checking after the drop is equivalent —
    // but the reference checks the distinct raw set; e=0 axes are distinct entries there too)
    for (size_t i = 0; i < merged.size(); ++i)
        for (size_t j = i + 1; j < merged.size(); ++j)
            if (Pauli::anticommute_bit(merged[i].first, merged[j].first)) {
                nf.commuting_class = false;
                return nf;
            }
    // even totals fold into the prefix; odd survive
    for (auto& m : merged) {
        const int e = m.second;
        if (e == 0) continue;
        if ((e & 1) == 0) {
            if (e == 2)      nf.prefix = Pauli::multiply(nf.prefix, i_pow_axis(m.first, 1));
            else if (e == 6) nf.prefix = Pauli::multiply(nf.prefix, i_pow_axis(m.first, 3));
            // e == 4: global −1, unobservable
        } else {
            nf.rots.push_back(std::move(m));
        }
    }
    std::sort(nf.rots.begin(), nf.rots.end(), axis_lt);
    return nf;
}

void ppr_rots_from_diag(const DiagNormalForm& nf, int n,
                        std::vector<std::pair<Pauli, int>>& rots_out) {
    for (int q = 0; q < n; ++q) {
        if (!(q < (int)nf.a.size() && nf.a[q])) continue;
        Pauli Zq(n);
        Zq.setz(q);
        rots_out.emplace_back(std::move(Zq), 7);          // S → rot(7, Z_q)  (≡ 3 mod 4)
    }
    for (const auto& e : nf.cz) {                         // CZ → rot(7,Zj)·rot(7,Zl)·rot(1,ZjZl)
        Pauli Zj(n), Zl(n), Zjl(n);
        Zj.setz(e.first);
        Zl.setz(e.second);
        Zjl.setz(e.first);
        Zjl.setz(e.second);
        rots_out.emplace_back(std::move(Zj), 7);
        rots_out.emplace_back(std::move(Zl), 7);
        rots_out.emplace_back(std::move(Zjl), 1);
    }
}

// ─────────────────────────────── the axis-space law ────────────────────────────────────
ShotLaw build_shot_law_ppr(const CertifiedGroupPlanes& G, const PprNormalForm& pnf) {
    ShotLaw law;
    const int n = G.n_qubits;
    const int ng = G.n_gens;
    const int GW = nwords(ng);

    if (!pnf.commuting_class) { law.fallback = true; return law; }
    const int t = (int)pnf.rots.size();
    if (t > 60) {                                          // t-bit dirs live in one word here
        std::fprintf(stderr, "[build_shot_law_ppr] t=%d > 60 -> fallback\n", t);
        law.fallback = true;
        return law;
    }
    const int TW = nwords(t > 0 ? t : 1);                  // == 1

    // pattern rows L_j (ng-bit) + tier1 (active = OR_j L_j; sign = prefix plane, valid on
    // inactive gens exactly: col_i = 0 ⇒ h_i = ±g_i with the ⟨P,g_i⟩ sign, nothing else).
    std::vector<BV> Lrows(t);
    BV active(GW, 0);
    for (int j = 0; j < t; ++j) {
        Lrows[j] = prefix_plane1(G, pnf.rots[j].first);
        for (int w = 0; w < GW; ++w) active[w] |= Lrows[j][w];
    }
    law.tier1.sign = prefix_plane1(G, pnf.prefix);
    law.tier1.active = active;

    // touched (== active for PPR) generator list + t-bit dressing cols
    std::vector<int> aidx;
    for (int w = 0; w < GW; ++w) { uint64_t bits = active[w];
        while (bits) { aidx.push_back(w * 64 + __builtin_ctzll(bits)); bits &= bits - 1; } }
    const int A = (int)aidx.size();
    std::vector<BV> cols(A, BV(TW, 0));                    // col_i ∈ F2^t per active gen
    std::vector<int> pos_of(ng, -1);
    for (int a2 = 0; a2 < A; ++a2) {
        pos_of[aidx[a2]] = a2;
        for (int j = 0; j < t; ++j)
            if (bget(Lrows[j], aidx[a2])) bset(cols[a2], j);
    }

    // h_i exact (active gens only): prefix conj sign + per-anticommuting-axis (±i A_j) factor
    //   rot(e,A)† g rot(e,A) = g·(+iA) for e ≡ 1 (mod 4), g·(−iA) for e ≡ 3 (mod 4).
    std::vector<Pauli> h(A);
    for (int a2 = 0; a2 < A; ++a2) {
        const Pauli& g = G.gens[aidx[a2]];
        Pauli hi = g;
        hi.phase = (g.phase + 2 * Pauli::anticommute_bit(pnf.prefix, g)) & 3;
        for (int j = 0; j < t; ++j) {
            if (!bget(cols[a2], j)) continue;
            const int e = pnf.rots[j].second;
            hi = Pauli::multiply(hi, i_pow_axis(pnf.rots[j].first, (e & 3) == 1 ? 1 : 3));
        }
        h[a2] = std::move(hi);
    }

    // ── lattice classification in F2^t ──
    std::vector<BV> Vrows;                                 // nonzero cols (all active have ≥1 bit)
    for (int a2 = 0; a2 < A; ++a2) if (bany(cols[a2])) Vrows.push_back(cols[a2]);
    Rref Vb = rref(Vrows);

    // Q = {w : A_w ∈ ±G}: the reduced symplectic support of A_w is LINEAR in w (reduction
    // against the fixed group RREF), so Q = left-null of the axes' reduced (x‖z) rows.
    std::vector<BV> redrows(t);
    for (int j = 0; j < t; ++j) {
        Membership m = G.reduce(pnf.rots[j].first);
        BV row(nwords(2 * n), 0);
        for (int q = 0; q < n; ++q) {
            if (m.rep.xbit(q)) bset(row, q);
            if (m.rep.zbit(q)) bset(row, n + q);
        }
        redrows[j] = std::move(row);
    }
    std::vector<BV> Qb_members = left_deps(redrows, 2 * n);   // t-bit combos: A_w support ∈ G
    Rref Qb = rref(Qb_members);

    // Kz = V ∩ ker L via the Π-row map over the V basis (S(w) = Σ_j w_j L_j).
    std::vector<BV> Kz;
    {
        std::vector<BV> Srows(Vb.R.size());
        for (size_t k = 0; k < Vb.R.size(); ++k) {
            BV s(GW, 0);
            for (int j = 0; j < t; ++j) if (bget(Vb.R[k], j)) bxor(s, Lrows[j]);
            Srows[k] = std::move(s);
        }
        for (const BV& c : left_deps(Srows, ng)) {
            BV u(TW, 0);
            for (size_t k = 0; k < Vb.R.size(); ++k) if (bget(c, k)) bxor(u, Vb.R[k]);
            Kz.push_back(std::move(u));
        }
    }
    // D = Kz ∩ Q (structural — no basis-dependent classification; the reference's D-first fix)
    std::vector<BV> D;
    {
        std::vector<BV> Dres(Kz.size());
        for (size_t k = 0; k < Kz.size(); ++k) Dres[k] = reduce_vec(Kz[k], Qb);
        for (const BV& c : left_deps(Dres, t > 0 ? t : 1)) {
            BV d(TW, 0);
            for (size_t k = 0; k < Kz.size(); ++k) if (bget(c, k)) bxor(d, Kz[k]);
            D.push_back(std::move(d));
        }
    }
    law.r = (int)Vb.R.size() - (int)Kz.size();

    // ── coin masks: Π row per active gen = S(col_i); rank(Π) must equal r ──
    std::vector<BV> Pi;
    for (int a2 = 0; a2 < A; ++a2) {
        BV row(GW, 0);
        for (int j = 0; j < t; ++j) if (bget(cols[a2], j)) bxor(row, Lrows[j]);
        if (bany(row)) Pi.push_back(std::move(row));
    }
    Rref coin = rref(Pi);
    if ((int)coin.R.size() != law.r)
        throw std::logic_error("build_shot_law_ppr: coin rank != r (rank(Pi) assert)");
    law.coin_masks = coin.R;

    // ── C_det = {c : col_c ∈ Q} (active combos; inactive singletons pinned to plane1) ──
    std::vector<BV> BredA(A);
    for (int a2 = 0; a2 < A; ++a2) BredA[a2] = reduce_vec(cols[a2], Qb);
    std::vector<BV> Cdet;
    for (const BV& c : left_deps(BredA, t > 0 ? t : 1)) {
        BV cs(GW, 0);
        for (int a2 = 0; a2 < A; ++a2) if (bget(c, a2)) bset(cs, aidx[a2]);
        Cdet.push_back(std::move(cs));
    }

    // ── kernel dirs: FOLDABLE = extend D → Kz; UNFOLDABLE = extend (Q ∪ Kz) → ker L ──
    std::vector<BV> fold_dirs;
    {
        std::vector<BV> acc = D;
        Rref det = rref(acc);
        for (const BV& u : Kz)
            if (!in_span(u, det)) { fold_dirs.push_back(u); acc.push_back(u); det = rref(acc); }
    }
    std::vector<BV> unfold_dirs;
    {
        std::vector<BV> kerL;
        for (const BV& c : left_deps(Lrows, ng)) kerL.push_back(c);   // t-bit combos
        std::vector<BV> acc = Qb_members;
        for (const BV& u : Kz) acc.push_back(u);
        Rref cur = rref(acc);
        for (const BV& u : kerL)
            if (!in_span(u, cur)) { unfold_dirs.push_back(u); acc.push_back(u); cur = rref(acc); }
    }
    // exact axis-product reps (reduce mod group, exact phase); unfoldable dirs must be
    // LOGICAL by construction (an in-group dir would lie in span(Q)).
    auto axis_product = [&](const BV& w) {
        Pauli Aw(n);
        for (int j = 0; j < t; ++j)
            if (bget(w, j)) Aw = Pauli::multiply(Aw, pnf.rots[j].first);
        return Aw;
    };
    std::vector<BV> kernel_dirs = fold_dirs;               // foldable first (mask-parallel order)
    for (const BV& u : fold_dirs) {
        Membership m = G.reduce(axis_product(u));
        if (m.verdict != Membership::LOGICAL)
            throw std::logic_error("build_shot_law_ppr: foldable kernel dir not LOGICAL");
        law.kernel_logicals.push_back(m.rep);
    }
    for (const BV& u : unfold_dirs) {
        Membership m = G.reduce(axis_product(u));
        if (m.verdict != Membership::LOGICAL)
            throw std::logic_error("build_shot_law_ppr: unfoldable kernel dir not LOGICAL");
        law.kernel_logicals.push_back(m.rep);
    }
    const int Kf = (int)fold_dirs.size();
    const int Ku = (int)unfold_dirs.size();
    law.kappa = Kf + Ku;

    // scope guards (reference parity): ≥2 foldable logicals, or foldable logical + foldable
    // in-group dir (D≠∅) — out of the independent-fold scope; loud fallback, never wrong σ.
    if (Kf >= 2 || (Kf >= 1 && !D.empty())) {
        std::fprintf(stderr, "[build_shot_law_ppr] kappa_fold=%d |D|=%zu -> fallback (scope)\n",
                     Kf, D.size());
        law.fallback = true;
        return law;
    }

    // ── foldable kernel masks/preimages (V1 machinery verbatim; qubit rows → L rows) ──
    std::vector<BV> kernel_pre;
    if (Kf > 0) {
        std::vector<BV> Cdet_full = Cdet;
        for (int i = 0; i < ng; ++i)
            if (!bget(active, i)) { BV e(GW, 0); bset(e, i); Cdet_full.push_back(std::move(e)); }
        std::vector<BV> Wbasis = nullspace(Cdet_full, ng);
        for (const BV& u : fold_dirs) {
            // legacy probe: [L rows ‖ C_det] c = [u ‖ 0]  (row k of L: bit i = col_i[k])
            std::vector<BV> mA;
            mA.reserve((size_t)t + Cdet_full.size());
            for (int j = 0; j < t; ++j) mA.push_back(Lrows[j]);
            for (const BV& d : Cdet_full) mA.push_back(d);
            std::vector<int> mb(mA.size(), 0);
            for (int j = 0; j < t; ++j) mb[j] = bget(u, j) ? 1 : 0;
            bool legacy_ok = false;
            BV c_legacy = solve_gf2(mA, mb, ng, &legacy_ok);
            BV cj, delta;
            if (legacy_ok && !in_span(c_legacy, coin)) {
                cj = c_legacy;
                delta = c_legacy;
            } else {
                std::vector<int> pb(t, 0);
                for (int j = 0; j < t; ++j) pb[j] = bget(u, j) ? 1 : 0;
                bool pre_ok = false;
                cj = solve_gf2(Lrows, pb, ng, &pre_ok);
                if (!pre_ok)
                    throw std::logic_error("build_shot_law_ppr: preimage infeasible (u not in V)");
                for (const BV& cm : law.coin_masks) if (bvdot(cj, cm)) {
                    std::fprintf(stderr, "[build_shot_law_ppr] preimage not ⊥ coins -> fallback\n");
                    law.fallback = true;
                    break;
                }
                if (law.fallback) return law;
                for (const BV& w : Wbasis) {
                    BV wr = reduce_vec(w, coin);
                    if (bany(wr) && bvdot(cj, wr)) { delta = std::move(wr); break; }
                }
                if (delta.empty()) {
                    std::fprintf(stderr,
                                 "[build_shot_law_ppr] no fold dir in C_det^perp \\ coins -> fallback\n");
                    law.fallback = true;
                    return law;
                }
            }
            kernel_pre.push_back(std::move(cj));
            law.kernel_masks.push_back(std::move(delta));
        }
        // pairing: c_j·δ_k = δ_jk (Kf ≤ 1 here, but keep the loud guard)
        for (int j = 0; j < Kf && !law.fallback; ++j)
            for (int k = 0; k < Kf; ++k)
                if (bvdot(kernel_pre[j], law.kernel_masks[k]) != ((j == k) ? 1 : 0)) {
                    std::fprintf(stderr, "[build_shot_law_ppr] pairing != delta_jk -> fallback\n");
                    law.fallback = true;
                    break;
                }
        if (law.fallback) return law;
    }
    for (int j = 0; j < Ku; ++j) law.kernel_masks.emplace_back();   // unfoldable: empty mask
    law.kernel_foldable.assign((size_t)Kf, 1);
    law.kernel_foldable.resize((size_t)(Kf + Ku), 0);

    // joint independence: coins ∪ foldable masks span rank r + Kf
    {
        std::vector<BV> allm = law.coin_masks;
        for (int j = 0; j < Kf; ++j) allm.push_back(law.kernel_masks[j]);
        if ((int)rref(allm).R.size() != law.r + Kf)
            throw std::logic_error("build_shot_law_ppr: coin ∪ fold masks not rank r+kappa_fold");
    }

    // ── base point: b_c from exact Π h_i reduction; solve; pin inactive to plane1 ──
    std::vector<int> bvals(Cdet.size(), 0);
    for (size_t ci = 0; ci < Cdet.size(); ++ci) {
        Pauli Sc(n);
        const BV& cc = Cdet[ci];
        for (int cw = 0; cw < GW; ++cw) {
            uint64_t bits = cc[cw];
            while (bits) {
                const int i = cw * 64 + __builtin_ctzll(bits); bits &= bits - 1;
                Sc = Pauli::multiply(Sc, h[pos_of[i]]);
            }
        }
        Membership m = G.reduce(Sc);
        if (m.verdict != Membership::IN_GROUP)
            throw std::logic_error("build_shot_law_ppr: deterministic combo not in ±G");
        if (m.rep.phase & 1)
            throw std::logic_error("build_shot_law_ppr: non-Hermitian deterministic combo");
        bvals[ci] = (m.rep.phase >> 1) & 1;
    }
    BV base = solve_gf2(Cdet, bvals, ng);
    for (int w = 0; w < GW; ++w) base[w] |= law.tier1.sign[w] & ~active[w];
    law.det_signs = std::move(base);

    // ── kernel base-sign bits (foldable dirs): s_j vs the stored logical rep, then ⊕ c_j·det ──
    if (Kf > 0) {
        law.kernel_base.assign((size_t)Kf, 0);
        for (int j = 0; j < Kf; ++j) {
            Pauli Hj(n);
            for (int i = 0; i < ng; ++i)
                if (bget(kernel_pre[j], i)) Hj = Pauli::multiply(Hj, h[pos_of[i]]);
            Membership m = G.reduce(Hj);
            const Pauli& L = law.kernel_logicals[j];
            if (m.rep.x != L.x || m.rep.z != L.z)
                throw std::logic_error("build_shot_law_ppr: kernel base support != kernel logical");
            const int s_j = ((m.rep.phase ^ L.phase) >> 1) & 1;
            law.kernel_base[j] = (uint8_t)(s_j ^ bvdot(kernel_pre[j], law.det_signs));
        }
    }
    law.kernel_base.resize((size_t)(Kf + Ku), 0);
    return law;
}

// ── V3: Born-weighted observable channel for a PPR plan (see the header contract). ──────
ObsChannel classify_observable_ppr(const CertifiedGroupPlanes& G, const PprNormalForm& pnf,
                                   const Pauli& W) {
    ObsChannel out;
    const int ng = G.n_gens;
    const int GW = nwords(ng);
    const int t = (int)pnf.rots.size();
    if (!pnf.commuting_class || t > 60) { out.guard = true; return out; }
    const int TW = nwords(t > 0 ? t : 1);

    // axis pattern rows (ng-bit) and per-generator t-bit columns — as in build_shot_law_ppr
    std::vector<BV> Lrows(t);
    for (int j = 0; j < t; ++j) Lrows[j] = prefix_plane1(G, pnf.rots[j].first);
    // u_W: axes anticommuting with W; W' = E†WE (identity prefix): per axis ±i·A mul
    BV uW(TW, 0);
    Pauli Wp = W;
    for (int j = 0; j < t; ++j) {
        if (Pauli::anticommute_bit(pnf.rots[j].first, W)) {
            bset(uW, j);
            const int e = pnf.rots[j].second;
            Wp = Pauli::multiply(Wp, i_pow_axis(pnf.rots[j].first, (e & 3) == 1 ? 1 : 3));
        }
    }
    // kerL basis (axis-space normalizer frame) + reduction frame
    std::vector<BV> kerL;
    for (const BV& c : left_deps(Lrows, ng)) kerL.push_back(c);
    Rref KR = rref(kerL);
    // generator columns reduced mod kerL, eliminate with ng-bit combo tracking
    struct Ent { BV v; BV c; int pc; };
    std::vector<Ent> red;
    for (int i = 0; i < ng; ++i) {
        BV col(TW, 0);
        for (int j = 0; j < t; ++j) if (bget(Lrows[j], i)) bset(col, j);
        if (!bany(col)) continue;
        col = reduce_vec(std::move(col), KR);
        BV cc(GW, 0);
        bset(cc, i);
        for (const Ent& e : red) if (bget(col, e.pc)) { bxor(col, e.v); bxor(cc, e.c); }
        int pc = bfirst(col);
        if (pc >= 0) red.push_back(Ent{std::move(col), std::move(cc), pc});
    }
    BV target = reduce_vec(uW, KR);
    BV cW(GW, 0);
    for (const Ent& e : red) if (bget(target, e.pc)) { bxor(target, e.v); bxor(cW, e.c); }
    if (bany(target)) { out.reachable = false; return out; }   // conditional ≡ 0
    out.reachable = true;
    out.mask = cW;
    // Λ_W = reduce(W'·Π_{i∈c_W} h_i), h_i assembled exactly (identity prefix)
    Pauli R = Wp;
    for (int cw2 = 0; cw2 < GW; ++cw2) {
        uint64_t bits = cW[cw2];
        while (bits) {
            const int i = cw2 * 64 + __builtin_ctzll(bits); bits &= bits - 1;
            Pauli hi = G.gens[i];
            for (int j = 0; j < t; ++j) {
                if (!bget(Lrows[j], i)) continue;
                const int e = pnf.rots[j].second;
                hi = Pauli::multiply(hi, i_pow_axis(pnf.rots[j].first, (e & 3) == 1 ? 1 : 3));
            }
            R = Pauli::multiply(R, hi);
        }
    }
    Membership m = G.reduce(R);
    if (m.verdict == Membership::ANTI) { out.guard = true; return out; }
    out.lam = m.rep;
    return out;
}

// ─────────────────────────────── memo ──────────────────────────────────────────────────
void ppr_key_words(const PprNormalForm& pnf, int n, std::vector<uint64_t>& out) {
    const int NW = nwords(n);
    out.clear();
    out.reserve(pnf.rots.size() * (2 * (size_t)NW + 1));
    for (const auto& ea : pnf.rots) {
        for (int w = 0; w < NW; ++w) out.push_back(ea.first.x[w]);
        for (int w = 0; w < NW; ++w) out.push_back(ea.first.z[w]);
        out.push_back((uint64_t)ea.second);
    }
}

const PprCachedPlan& PprPlanCache::get_or_build(const CertifiedGroupPlanes& G,
                                                const PprNormalForm& pnf) {
    std::vector<uint64_t> key;
    ppr_key_words(pnf, G.n_qubits, key);
    uint64_t hsh = 0xcbf29ce484222325ull ^ G.identity_token();
    for (uint64_t w : key) { hsh ^= w; hsh *= 0x100000001B3ull; }
    auto& bucket = store_[hsh];
    for (const auto& p : bucket)
        if (p->key_words == key) { ++hits_; return *p; }
    ++misses_;
    // build with IDENTITY prefix (prefix-free base0; the caller XORs prefix_plane1 per shot)
    PprNormalForm id = pnf;
    id.prefix = Pauli(G.n_qubits);
    ShotLaw law = build_shot_law_ppr(G, id);
    auto plan = std::make_unique<PprCachedPlan>();
    plan->r = law.r;
    plan->kappa = law.kappa;
    plan->fallback = law.fallback;
    plan->base0 = std::move(law.det_signs);
    plan->coin_masks = std::move(law.coin_masks);
    plan->kernel_masks = std::move(law.kernel_masks);
    plan->kernel_base = std::move(law.kernel_base);
    plan->kernel_foldable = std::move(law.kernel_foldable);
    plan->kernel_logicals = std::move(law.kernel_logicals);
    plan->active = std::move(law.tier1.active);
    plan->key_words = std::move(key);
    bucket.push_back(std::move(plan));
    ++n_plans_;
    return *bucket.back();
}

}  // namespace qeccore
