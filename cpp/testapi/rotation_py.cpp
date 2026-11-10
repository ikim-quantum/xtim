// rotation_py.cpp — lean D1-S1 test API: the production rotation model / records as numpy views, and an INDEPENDENT
// ORACLE of both (the P2/D1 probe's construction — cpp/testapi/probe_p2_py.cpp on branch probe/pbc — ported on
// vectors: per-pair Pauli anticommutation for every ρ_c and generator row, its own RREFs, free_dir_of on its own Γ_n
// basis; the records from the record operators' own transpose ⟨P_f, M_c⟩, never the sampler's prefix rows).
// tests/test_lean_d1s1.py holds the production model and records equal to the oracle's on the public corpus.
//   rot_model(fam) / rot_records(fam) / rot_check(fam)   — the production objects (and the decode-vs-recompute check)
//   rot_oracle_model(fam) / rot_oracle_records(fam)       — the oracle, in the same dict layout
// `fam` is a FamilySampler (compile_family_sampler): `_engine()` arms the term view for the oracle.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "qeccore/noise_chain.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/term_residual.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"
#include "qeccore/twirl_rotation.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "twirl_test_access.hpp"

namespace py = pybind11;
using namespace qeccore;

namespace {

TwirlRecordSampler& engine_of(const py::object& fam) {
    py::capsule c = fam.attr("_engine")();
    if (std::strcmp(c.name(), "xtim.TwirlRecordSampler") != 0)
        throw std::invalid_argument("rotation test API: expected a FamilySampler (compile_family_sampler)");
    return *static_cast<TwirlRecordSampler*>(c.get_pointer());
}

template <class T>
py::array_t<T> np1(const std::vector<T>& v) {
    py::array_t<T> a((py::ssize_t)v.size());
    if (!v.empty()) std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}
template <class T>
py::array_t<T> np2(const std::vector<T>& v, py::ssize_t rows, py::ssize_t cols) {
    if ((py::ssize_t)v.size() != rows * cols) throw std::logic_error("rotation test API: a 2-d view with the wrong size");
    py::array_t<T> a({rows, cols});
    if (!v.empty()) std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}
py::list fds(const std::vector<FreeDir>& v, const std::vector<uint8_t>& valid) {
    py::list out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (!valid[i]) { out.append(py::none()); continue; }
        out.append(py::make_tuple(py::bytes(reinterpret_cast<const char*>(v[i].a.data()), v[i].a.size()),
                                  py::bytes(reinterpret_cast<const char*>(v[i].b.data()), v[i].b.size()),
                                  (int)v[i].phase, (int)v[i].par));
    }
    return out;
}

// the dict layout shared by the production view and the oracle (gn rows by ascending pivot in both)
py::dict model_dict(const RotationModel& M) {
    py::dict d;
    d["built"] = M.built; d["kind2"] = M.kind2;
    d["t"] = M.t; d["TW"] = M.TW; d["k"] = M.k; d["KW"] = M.KW; d["m"] = M.m; d["nch"] = M.nch; d["ng"] = M.ng;
    d["rho"] = np2(M.rho, M.m, M.TW);
    d["scan"] = np1(M.scan); d["rank_of"] = np1(M.rank_of);
    d["cols_off"] = np1(M.cols_off); d["cols"] = np1(M.cols);
    d["eta"] = np1(M.eta); d["live"] = np1(M.live);
    d["cone_gen"] = np1(M.cone_gen); d["cone_row"] = np2(M.cone_row, (py::ssize_t)M.cone_gen.size(), M.TW);
    d["phi_rank"] = M.phi_rank;
    d["gn_rows"] = np2(M.gn_rows, (py::ssize_t)M.gn_piv.size(), M.TW); d["gn_piv"] = np1(M.gn_piv);
    d["gn_col_to_row"] = np1(M.gn_col_to_row);
    d["gn_a"] = np2(M.gn_a, (py::ssize_t)M.gn_piv.size(), M.KW); d["gn_b"] = np2(M.gn_b, (py::ssize_t)M.gn_piv.size(), M.KW);
    d["gn_phase"] = np1(M.gn_phase); d["gn_par"] = np1(M.gn_par);
    d["gz_dim"] = M.gz_dim;
    d["nx"] = M.nx; d["nvirt"] = M.nvirt; d["TXW"] = M.TXW;
    d["x_T0"] = np2(M.x_T0, M.nx, M.TXW); d["x_logical"] = np1(M.x_logical);
    d["x_fd"] = fds(M.x_fd, M.x_logical);
    d["anti_extras"] = M.anti_extras;
    return d;
}

// ═══════════════ the oracle: the probe's model_of (probe_p2_py.cpp, lean P2/D1) on vectors ═══════════════
using Word = std::vector<uint64_t>;
inline bool wget(const Word& w, int i) { return (w[(size_t)(i >> 6)] >> (i & 63)) & 1ULL; }
inline void wset(Word& w, int i) { w[(size_t)(i >> 6)] |= 1ULL << (i & 63); }
inline void wxor(Word& a, const Word& b) { for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i]; }
inline bool wany(const Word& a) { for (uint64_t x : a) if (x) return true; return false; }
inline int wlead(const Word& a) { for (size_t i = 0; i < a.size(); ++i) if (a[i]) return (int)i * 64 + __builtin_ctzll(a[i]); return -1; }
inline int wpop_and(const Word& a, const Word& b) { int c = 0; for (size_t i = 0; i < a.size(); ++i) c += __builtin_popcountll(a[i] & b[i]); return c; }

// plain RREF (lowest-bit pivots, fully reduced; rows in insertion order — the probe's rref_words)
void rref_words(std::vector<Word>& rows, std::vector<int>& piv, int W) {
    std::vector<Word> R;
    piv.clear();
    std::vector<int> c2r((size_t)W * 64, -1);
    for (Word v : rows) {
        for (int w = 0; w < W; ++w) {
            uint64_t word = v[(size_t)w];
            while (word) {
                const int b = __builtin_ctzll(word);
                const int r = c2r[(size_t)(w * 64 + b)];
                if (r >= 0) { wxor(v, R[(size_t)r]); word = (b == 63) ? 0 : (v[(size_t)w] & (~0ULL << (b + 1))); }
                else word &= word - 1;
            }
        }
        const int pc = wlead(v);
        if (pc < 0) continue;
        for (size_t r = 0; r < R.size(); ++r) if (wget(R[r], pc)) wxor(R[r], v);
        R.push_back(v);
        piv.push_back(pc);
        c2r[(size_t)pc] = (int)R.size() - 1;
    }
    rows = std::move(R);
}

RotationModel oracle_model(TwirlRecordSampler& s) {
    TermResidual& tr = TwirlSamplerTestAccess::family_terms(s);   // armed on demand
    RotationModel M;
    const int n = s.deferred_wires(), NW = (n + 63) / 64;
    M.t = tr.num_terms(); M.TW = (M.t + 63) / 64;
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
    const FramedSuperposition& bare = s.bare_state();
    M.k = bare.k(); M.KW = std::max(1, (M.k + 63) / 64);
    M.ng = G.n_gens;
    const std::vector<Pauli> ops = s.record_operators();
    const std::vector<int> order = s.record_column_order();
    M.m = (int)ops.size(); M.nch = s.num_channels();
    const int t = M.t, TW = M.TW, m = M.m, ng = M.ng, W = std::max(TW, 1);
    M.scan.resize((size_t)m);
    for (int c = 0; c < m; ++c) M.scan[(size_t)c] = c;
    std::sort(M.scan.begin(), M.scan.end(), [&](int x, int y) { return order[(size_t)x] != order[(size_t)y] ? order[(size_t)x] < order[(size_t)y] : x < y; });
    M.rank_of.assign((size_t)m, 0);
    for (int i = 0; i < m; ++i) M.rank_of[(size_t)M.scan[(size_t)i]] = i;
    // axes (any commuting Hermitian Pauli; lean D1-S3): B_j = (−1)^{η_j} B̂_j, B̂_j the positive Hermitian form — here
    // found as the sign s with (s·B)² = +I among {B, −B} whose X^x Z^z phase is i^{#Y}: per qubit Y = i·X·Z
    M.eta.assign((size_t)TW, 0); M.live.assign((size_t)TW, 0);
    std::vector<Pauli> Bh;
    for (int j = 0; j < t; ++j) {
        const Pauli& B = tr.axis(j);
        int ny = 0;
        for (int q = 0; q < n; ++q) if (B.xbit(q) && B.zbit(q)) ++ny;
        if (((B.phase - ny) & 1) != 0) throw std::logic_error("rot_oracle_model: a non-Hermitian axis");
        if (((B.phase - ny) & 3) == 2) wset(M.eta, j);
        for (int q = 0; q < n && !M.kind2; ++q) M.kind2 = B.xbit(q);   // an axis with X content (information only)
        Pauli h = B; h.phase = ny & 3;
        Bh.push_back(h);
    }
    // per column rows (pairwise anticommutation), cols_of, live
    std::vector<Word> rho((size_t)m, Word((size_t)W, 0));
    std::vector<std::vector<int>> cols_of((size_t)std::max(t, 0));
    for (int c = 0; c < m; ++c)
        for (int j = 0; j < t; ++j)
            if (Pauli::anticommute_bit(ops[(size_t)c], tr.axis(j))) { wset(rho[(size_t)c], j); cols_of[(size_t)j].push_back(c); wset(M.live, j); }
    M.rho.clear();
    for (int c = 0; c < m; ++c) M.rho.insert(M.rho.end(), rho[(size_t)c].begin(), rho[(size_t)c].begin() + TW);
    M.cols_off.assign((size_t)t + 1, 0);
    for (int j = 0; j < t; ++j) { M.cols.insert(M.cols.end(), cols_of[(size_t)j].begin(), cols_of[(size_t)j].end()); M.cols_off[(size_t)j + 1] = (int)M.cols.size(); }
    // generator rows and Γ_n
    std::vector<Word> gen_rows((size_t)ng, Word((size_t)W, 0));
    for (int i = 0; i < ng; ++i) {
        for (int j = 0; j < t; ++j) if (Pauli::anticommute_bit(G.gens[(size_t)i], tr.axis(j))) wset(gen_rows[(size_t)i], j);
        if (wany(gen_rows[(size_t)i])) { M.cone_gen.push_back(i); M.cone_row.insert(M.cone_row.end(), gen_rows[(size_t)i].begin(), gen_rows[(size_t)i].begin() + TW); }
    }
    std::vector<Word> gn_rows; std::vector<int> gn_piv; std::vector<int> gn_c2r((size_t)W * 64, -1);
    {
        std::vector<Word> R = gen_rows; std::vector<int> piv;
        rref_words(R, piv, W);
        M.phi_rank = (int)R.size();
        std::vector<uint8_t> is_piv((size_t)std::max(t, 1), 0);
        for (int p : piv) is_piv[(size_t)p] = 1;
        std::vector<Word> ker;
        for (int f = 0; f < t; ++f) {
            if (is_piv[(size_t)f]) continue;
            Word v((size_t)W, 0); wset(v, f);
            for (size_t r = 0; r < R.size(); ++r) if (wget(R[r], f)) wset(v, piv[r]);
            ker.push_back(v);
        }
        std::vector<int> kpiv;
        rref_words(ker, kpiv, W);
        // by ascending pivot (the shared layout)
        std::vector<int> ord((size_t)ker.size());
        for (size_t r = 0; r < ker.size(); ++r) ord[r] = (int)r;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return kpiv[(size_t)a] < kpiv[(size_t)b]; });
        for (int r : ord) { gn_rows.push_back(ker[(size_t)r]); gn_piv.push_back(kpiv[(size_t)r]); }
        for (size_t r = 0; r < gn_piv.size(); ++r) gn_c2r[(size_t)gn_piv[r]] = (int)r;
        for (const Word& v : gn_rows)
            for (int i = 0; i < ng; ++i)
                if (wpop_and(v, gen_rows[(size_t)i]) & 1) throw std::logic_error("rot_oracle_model: a Γ_n row not in ker Φ");
        M.gn_col_to_row.assign((size_t)std::max(t, 1), -1);
        std::vector<Word> brows;
        for (size_t r = 0; r < gn_rows.size(); ++r) {
            Pauli Zw(n);
            for (int j = t - 1; j >= 0; --j)            // descending: the product is order-free (commuting axes)
                if (wget(gn_rows[r], j)) Zw = Pauli::multiply(Bh[(size_t)j], Zw);
            const Membership mb = G.reduce(Zw);
            if (mb.verdict == Membership::ANTI) throw std::logic_error("rot_oracle_model: a Γ_n element anticommutes with the group");
            const FreeDir fd = free_dir_of(bare, Zw);
            if (!fd_hermitian(fd)) throw std::logic_error("rot_oracle_model: a non-Hermitian Γ_n direction");
            Word ab((size_t)2 * M.KW, 0);
            for (int d = 0; d < M.k; ++d) { if (fd.a[(size_t)d]) wset(ab, d); if (fd.b[(size_t)d]) wset(ab, M.KW * 64 + d); }
            if (mb.verdict == Membership::IN_GROUP && wany(ab)) throw std::logic_error("rot_oracle_model: IN_GROUP with free content");
            if (mb.verdict == Membership::LOGICAL && !wany(ab)) throw std::logic_error("rot_oracle_model: LOGICAL without free content");
            M.gn_rows.insert(M.gn_rows.end(), gn_rows[r].begin(), gn_rows[r].begin() + TW);
            M.gn_piv.push_back(gn_piv[r]);
            M.gn_col_to_row[(size_t)gn_piv[r]] = (int)r;
            M.gn_a.insert(M.gn_a.end(), ab.begin(), ab.begin() + M.KW);
            M.gn_b.insert(M.gn_b.end(), ab.begin() + M.KW, ab.end());
            M.gn_phase.push_back((uint8_t)(fd.phase & 3));
            M.gn_par.push_back((uint8_t)(fd.par & 1));
            brows.push_back(ab);
        }
        std::vector<int> bpiv;
        rref_words(brows, bpiv, 2 * M.KW);
        M.gz_dim = (int)gn_rows.size() - (int)brows.size();
    }
    // extras: φ(M_e) reduced against RREF{Φ(e_j)} (with the t-bit combination) → T0_e + residual; the residuals'
    // own RREF → virtual coordinates
    M.nx = m - M.nch;
    {
        const int GW = std::max(1, (ng + 63) / 64);
        std::vector<Word> R; std::vector<Word> Rc; std::vector<int> c2r((size_t)GW * 64, -1);
        auto lead = [&](const Word& v) { return wlead(v); };
        auto reduce = [&](Word& v, Word& comb) {
            for (int w = 0; w < GW; ++w) { uint64_t word = v[(size_t)w]; while (word) { const int b = __builtin_ctzll(word); const int r = c2r[(size_t)(w * 64 + b)];
                if (r >= 0) { wxor(v, R[(size_t)r]); wxor(comb, Rc[(size_t)r]); word = (b == 63) ? 0 : (v[(size_t)w] & (~0ULL << (b + 1))); } else word &= word - 1; } } };
        for (int j = 0; j < t; ++j) {
            Word v((size_t)GW, 0);
            for (int i = 0; i < ng; ++i) if (wget(gen_rows[(size_t)i], j)) wset(v, i);
            Word comb((size_t)W, 0); wset(comb, j);
            reduce(v, comb);
            const int ld = lead(v);
            if (ld < 0) continue;
            for (size_t r = 0; r < R.size(); ++r) if (wget(R[r], ld)) { wxor(R[r], v); wxor(Rc[r], comb); }
            R.push_back(v); Rc.push_back(comb); c2r[(size_t)ld] = (int)R.size() - 1;
        }
        const int VW = std::max(1, (M.nx + 63) / 64);
        std::vector<Word> VR; std::vector<Word> vcomb; std::vector<int> vc2r((size_t)GW * 64, -1);
        std::vector<Word> T0r((size_t)M.nx), vb((size_t)M.nx);
        M.x_logical.assign((size_t)M.nx, 0);
        for (int e = 0; e < M.nx; ++e) {
            std::vector<uint64_t> ph;
            phi_of(G, ops[(size_t)(M.nch + e)], ph);
            Word v(ph.begin(), ph.end()); v.resize((size_t)GW, 0);
            Word T0((size_t)W, 0);
            reduce(v, T0);
            T0r[(size_t)e] = T0;
            const bool logical = lead(v) < 0;
            M.x_logical[(size_t)e] = logical ? 1 : 0;
            if (!logical) ++M.anti_extras;
            Word vbe((size_t)VW, 0);
            for (int w = 0; w < GW; ++w) { uint64_t word = v[(size_t)w]; while (word) { const int b = __builtin_ctzll(word); const int r = vc2r[(size_t)(w * 64 + b)];
                if (r >= 0) { wxor(v, VR[(size_t)r]); wxor(vbe, vcomb[(size_t)r]); word = (b == 63) ? 0 : (v[(size_t)w] & (~0ULL << (b + 1))); } else word &= word - 1; } }
            const int ld = lead(v);
            if (ld >= 0) {   // S1 review H1: the ROW carries used ⊕ new; the extra's own word is the new coordinate alone
                Word rowc = vbe; wset(rowc, M.nvirt);
                for (size_t r = 0; r < VR.size(); ++r) if (wget(VR[r], ld)) { wxor(VR[r], v); wxor(vcomb[r], rowc); }
                VR.push_back(v); vcomb.push_back(rowc); vc2r[(size_t)ld] = (int)VR.size() - 1;
                std::fill(vbe.begin(), vbe.end(), 0); wset(vbe, M.nvirt); ++M.nvirt;
            }
            vb[(size_t)e] = vbe;
        }
        M.TXW = std::max(1, (t + M.nvirt + 63) / 64);
        M.x_T0.assign((size_t)M.nx * (size_t)M.TXW, 0);
        for (int e = 0; e < M.nx; ++e) {
            for (int w = 0; w < TW; ++w) M.x_T0[(size_t)e * M.TXW + (size_t)w] = T0r[(size_t)e][(size_t)w];
            for (int vc = 0; vc < M.nvirt; ++vc) if (wget(vb[(size_t)e], vc)) M.x_T0[(size_t)e * M.TXW + (size_t)((t + vc) >> 6)] |= 1ULL << ((t + vc) & 63);
            FreeDir fd;
            if (M.x_logical[(size_t)e]) {
                Pauli Q = ops[(size_t)(M.nch + e)];
                for (int j = 0; j < t; ++j) if (wget(T0r[(size_t)e], j)) { Pauli iB = tr.axis(j); iB.phase = (uint8_t)((iB.phase + 1) & 3); Q = Pauli::multiply(Q, iB); }
                fd = free_dir_of(bare, Q);
            }
            M.x_fd.push_back(std::move(fd));
        }
    }
    M.built = true;
    return M;
}

// the oracle's records: ⟨P_f, M_c⟩ over every column from the record operators' own per-wire transpose
RotationRecords oracle_records(TwirlRecordSampler& s) {
    TermResidual& tr = TwirlSamplerTestAccess::family_terms(s);
    const int n = s.deferred_wires(), NW = (n + 63) / 64;
    const std::vector<Pauli> ops = s.record_operators();
    const int m = (int)ops.size(), nch = s.num_channels(), nx = m - nch;
    RotationRecords R;
    R.TW = tr.term_words(); R.CW = s.channel_words(); R.XR = nx > 0 ? (nx + 63) / 64 : 0;
    std::vector<std::vector<int>> zcols((size_t)n), xcols((size_t)n);   // columns with Z on q / X on q
    for (int c = 0; c < m; ++c)
        for (int q = 0; q < n; ++q) { if (ops[(size_t)c].zbit(q)) zcols[(size_t)q].push_back(c); if (ops[(size_t)c].xbit(q)) xcols[(size_t)q].push_back(c); }
    const int A = tr.num_alts();
    R.off.assign((size_t)A + 1, 0);
    std::vector<uint8_t> tbits((size_t)m);
    auto put64 = [&](uint64_t v) { R.rec.push_back((uint32_t)v); R.rec.push_back((uint32_t)(v >> 32)); };
    for (int a = 0; a < A; ++a) {
        std::fill(tbits.begin(), tbits.end(), 0);
        const uint64_t* px = tr.alt_px(a); const uint64_t* pz = tr.alt_pz(a);
        for (int w = 0; w < NW; ++w) {
            for (uint64_t word = px[w]; word; word &= word - 1) for (int c : zcols[(size_t)(w * 64 + __builtin_ctzll(word))]) tbits[(size_t)c] ^= 1;
            for (uint64_t word = pz[w]; word; word &= word - 1) for (int c : xcols[(size_t)(w * 64 + __builtin_ctzll(word))]) tbits[(size_t)c] ^= 1;
        }
        R.off[(size_t)a] = (uint32_t)R.rec.size();
        const uint64_t* sv = tr.alt_s(a);
        for (int w = 0; w < R.TW; ++w) put64(sv[w]);
        put64(plan_key_hash(sv, R.TW) ^ 0xcbf29ce484222325ull);
        for (int w = 0; w < R.XR; ++w) { uint64_t v = 0; for (int b = 0; b < 64 && w * 64 + b < nx; ++b) if (tbits[(size_t)(nch + w * 64 + b)]) v |= 1ULL << b; put64(v); }
        const size_t nzp = R.rec.size(); R.rec.push_back(0); uint32_t nz = 0;
        for (int w = 0; w < R.CW; ++w) { uint64_t v = 0; for (int b = 0; b < 64 && w * 64 + b < nch; ++b) if (tbits[(size_t)(w * 64 + b)]) v |= 1ULL << b; if (v) { R.rec.push_back((uint32_t)w); put64(v); ++nz; } }
        R.rec[nzp] = nz;
    }
    R.off[(size_t)A] = (uint32_t)R.rec.size();
    return R;
}

// S1 review H1, the INDEPENDENT check of the extras' virtual coordinates (shares nothing with the echelon construction):
// for a subset S of extras, the XOR of their virtual bits is 0 iff Q_S = Π_{e∈S} M_e · Π_{j ∈ ⊕T0_e (real)} (iB_j) commutes
// with the certified group (G.reduce's verdict is not ANTI): φ(Q_S) = ⊕ r_e, the residual planes, whose coordinates the
// words are. Every subset when nx ≤ 8, else `nsub` random subsets (seeded); returns (subsets checked, violations).
py::tuple virtual_check(TwirlRecordSampler& s, long nsub, uint64_t seed) {
    const RotationModel& M = TwirlSamplerTestAccess::family_rotation_model(s);
    if (!M.built || M.nx == 0) return py::make_tuple(0L, 0L);
    TermResidual& tr = TwirlSamplerTestAccess::family_terms(s);
    const int n = s.deferred_wires();
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
    const std::vector<Pauli> ops = s.record_operators();
    const int t = M.t, nx = M.nx, TXW = M.TXW;
    std::vector<uint64_t> subsets;
    if (nx <= 8) { for (uint64_t m = 1; m < (1ull << nx); ++m) subsets.push_back(m); }
    else {
        uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
        for (long i = 0; i < nsub; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            uint64_t m = 0;
            for (int e = 0; e < nx; ++e) if ((x >> (e & 63)) & 1) m |= 1ull << (e & 63);   // nx ≤ 64 here
            if (m) subsets.push_back(m);
        }
    }
    long checked = 0, bad = 0;
    std::vector<uint64_t> w((size_t)TXW);
    for (uint64_t m : subsets) {
        std::fill(w.begin(), w.end(), 0);
        Pauli Q(n);
        for (int e = 0; e < nx && e < 64; ++e)
            if ((m >> e) & 1) {
                for (int u = 0; u < TXW; ++u) w[(size_t)u] ^= M.x_T0[(size_t)e * TXW + (size_t)u];
                Q = Pauli::multiply(Q, ops[(size_t)(M.nch + e)]);
            }
        for (int j = 0; j < t; ++j)
            if ((w[(size_t)(j >> 6)] >> (j & 63)) & 1) { Pauli iB = tr.axis(j); iB.phase = (uint8_t)((iB.phase + 1) & 3); Q = Pauli::multiply(Q, iB); }
        bool vzero = true;
        for (int v = 0; v < M.nvirt; ++v) if ((w[(size_t)((t + v) >> 6)] >> ((t + v) & 63)) & 1) vzero = false;
        const bool commutes = G.reduce(Q).verdict != Membership::ANTI;
        ++checked;
        if (vzero != commutes) ++bad;
    }
    return py::make_tuple(checked, bad);
}

py::dict records_dict(const RotationRecords& R) {
    py::dict d;
    d["TW"] = R.TW; d["XR"] = R.XR; d["CW"] = R.CW; d["alts"] = R.num_alts();
    d["rec"] = np1(R.rec); d["off"] = np1(R.off);
    return d;
}

}  // namespace

void register_rotation(py::module_& m) {
    m.def("rot_model", [](const py::object& fam) { return model_dict(TwirlSamplerTestAccess::family_rotation_model(engine_of(fam))); }, py::arg("fam"),
          "lean D1-S1 TEST API: the sampler's rotation model (compiled at construction) as arrays.");
    m.def("rot_records", [](const py::object& fam) { return records_dict(TwirlSamplerTestAccess::family_rotation_records(engine_of(fam))); }, py::arg("fam"),
          "lean D1-S1 TEST API: the sampler's compact per-alternative rotation records (u32 words) and offsets.");
    m.def("rot_check", [](const py::object& fam) {
        const auto r = TwirlSamplerTestAccess::family_rotation_check(engine_of(fam));
        return py::make_tuple(r.first, r.second);
    }, py::arg("fam"), "lean D1-S1 TEST API: (records checked, mismatches) of the decode-vs-recompute check.");
    m.def("rot_oracle_model", [](const py::object& fam) { return model_dict(oracle_model(engine_of(fam))); }, py::arg("fam"),
          "lean D1-S1 TEST API: the ORACLE rotation model (the P2/D1 probe's construction), same layout as rot_model.");
    m.def("rot_oracle_records", [](const py::object& fam) { return records_dict(oracle_records(engine_of(fam))); }, py::arg("fam"),
          "lean D1-S1 TEST API: the ORACLE records (<P_f, M_c> from the record operators' own transpose), same layout.");
    // ── lean D1-S2 ──
    // ── P3 S-b ──
    m.def("shot_block", [](const py::object& fam, bool on) { TwirlSamplerTestAccess::family_set_shot_block(engine_of(fam), on); },
          py::arg("fam"), py::arg("on"),
          "P3 S-b TEST API: on = False sends this sampler's serial runs through the event list (the oracle route).");
    m.def("shot_block_shots", [](const py::object& fam) { return TwirlSamplerTestAccess::family_shot_block_shots(engine_of(fam)); },
          py::arg("fam"), "P3 S-b TEST API: the shots served from shot blocks so far.");
    // ── lean N1 ──
    m.def("noise_table", [](const py::object& fam, bool full) {
        const NoiseModel& N = TwirlSamplerTestAccess::family_noise_model(engine_of(fam));
        const NoiseTable& T = full ? N.full() : N.pruned();
        py::dict d;
        d["tag"] = T.tag;
        d["mo0"] = T.mo0; d["oalt"] = T.oalt; d["ow"] = T.ow; d["oprob"] = T.oprob; d["oalias"] = T.oalias;
        d["mch"] = T.mch; d["mP"] = T.mP; d["gm"] = T.gm; d["events"] = T.events;
        py::list g, r0, r;
        for (const auto& G : T.groups) { g.append(py::make_tuple(G.P, G.sc, G.m0, G.m1)); r0.append(G.r0); r.append(G.r); }
        d["groups"] = g;
        d["r0"] = r0; d["r"] = r; d["R"] = T.R; d["blast"] = T.blast;
        return d;
    }, py::arg("fam"), py::arg("full"), "lean N1 TEST API: the pruned (full = False) or full mechanism table.");
    m.def("noise_alts", [](const py::object& fam) {
        const NoiseModel& N = TwirlSamplerTestAccess::family_noise_model(engine_of(fam));
        std::vector<double> p((size_t)N.alt_total());
        std::vector<int32_t> k((size_t)N.alt_total());
        for (int a = 0; a < N.alt_total(); ++a) { p[(size_t)a] = N.alt_p(a); k[(size_t)a] = N.channel_of(a); }
        return py::make_tuple(p, k);
    }, py::arg("fam"), "lean N1 TEST API: per flat alternative (its probability, its channel).");
    m.def("noise_block", [](const py::object& fam, bool full, uint64_t seed, uint64_t b) {
        const NoiseModel& N = TwirlSamplerTestAccess::family_noise_model(engine_of(fam));
        std::vector<uint64_t> ev;
        N.block_events(full ? N.full() : N.pruned(), seed, b, ev);
        return ev;
    }, py::arg("fam"), py::arg("full"), py::arg("seed"), py::arg("block"),
          "lean N1 TEST API: block b's events (alt << 32 | shot in block), chain order.");
    m.def("noise_chain_pos", [](const py::object& fam, bool full, uint64_t seed, size_t g, uint64_t b) {
        const NoiseModel& N = TwirlSamplerTestAccess::family_noise_model(engine_of(fam));
        const NoiseTable& T = full ? N.full() : N.pruned();
        if (g >= T.groups.size() || b > T.blast) throw std::invalid_argument("noise_chain_pos: no such chain");
        return NoiseModel::chain_pos(T, krng::stream_key(seed, T.tag), g, b);
    }, py::arg("fam"), py::arg("full"), py::arg("seed"), py::arg("group"), py::arg("block"),
          "lean N2 TEST API: the position the walk starts chain (group, block) at (draw c = mix(pos + (c + 1) * gamma)).");
    m.def("noise_full", [](const py::object& fam) { return TwirlSamplerTestAccess::family_noise_full(engine_of(fam)); },
          py::arg("fam"), "lean N1 TEST API: a run of this sampler draws the full channel table now.");
    m.def("noise_force_full", [](const py::object& fam, bool on) { TwirlSamplerTestAccess::family_set_noise_full(engine_of(fam), on); },
          py::arg("fam"), py::arg("on"), "lean N1 TEST API: on = True makes every run of this sampler draw the full table.");
    m.def("rot_on", [](const py::object& fam) { return TwirlSamplerTestAccess::family_rot_on(engine_of(fam)); }, py::arg("fam"),
          "lean D1-S2: the sampler serves its shots on the rotation loop (a diagonal-class circuit with rotations).");
    m.def("rot_store_stats", [](const py::object& fam) {
        TwirlRecordSampler& s = engine_of(fam);
        const RotPlanStore* st = TwirlSamplerTestAccess::family_rot_store(s);
        py::dict d;
        d["on"] = st != nullptr;
        if (!st) return d;
        d["plans"] = (long)st->plans(); d["tuples"] = (long)st->tuples();
        d["misses_k0"] = (long)st->misses_k0(); d["misses_kpos"] = (long)st->misses_kpos();
        d["plan_bytes"] = (long)st->plan_bytes(); d["trie_bytes"] = (long)st->trie_bytes();
        long nodes = 0;
        for (size_t i = 0; i < st->tuples(); ++i) for (const BornNode& nd : st->tuple(i).trie) nodes += std::isnan(nd.p0) ? 0 : 1;
        d["trie_nodes"] = nodes;
        d["window_shots"] = TwirlSamplerTestAccess::family_rot_window_shots(s);
        std::vector<long> kh;                          // lean D1-S3: plans by κ (the store's own plans, by ordinal)
        for (size_t i = 0; i < st->plans(); ++i) {
            const size_t o = st->offset_of_ordinal(i);
            if (!o) continue;
            const int k = st->kappa(st->plan(o));
            if ((size_t)k >= kh.size()) kh.resize((size_t)k + 1, 0);
            ++kh[(size_t)k];
        }
        d["kappa_hist"] = kh;
        d["anti_products"] = TwirlSamplerTestAccess::family_rot_anti_products(s);
        return d;
    }, py::arg("fam"));
    // the gauge-map oracle of the alternatives `alts` (time order): the resident plan (stored = True: built into the store
    // when absent) or a fresh build vs (D) under the computed gauge and vs (G) on the raw s; uz: the walked outcomes
    m.def("rot_gauge_of", [](const py::object& fam, const std::vector<int32_t>& alts, bool stored, py::object uz_obj) {
        TwirlRecordSampler& s = engine_of(fam);
        std::vector<uint8_t> uz;
        if (!uz_obj.is_none()) uz = uz_obj.cast<std::vector<uint8_t>>();
        RotGaugeCheck c;
        TwirlSamplerTestAccess::family_rot_gauge(s, alts, stored, uz_obj.is_none() ? nullptr : &uz, c);
        py::dict d;
        d["ok"] = c.ok; d["why"] = c.why;
        d["s"] = np1(c.s); d["rho"] = c.rho; d["kappa"] = c.kappa;
        d["piv"] = np1(c.piv); d["intro"] = np1(c.intro);
        d["g"] = np1(c.g); d["delta"] = np1(c.delta);
        d["Px"] = np1(c.Px); d["Pz"] = np1(c.Pz); d["PDx"] = np1(c.PDx); d["PDz"] = np1(c.PDz);
        d["AW"] = c.AW; d["BW"] = c.BW;
        d["A"] = np1(c.A); d["B"] = np1(c.B); d["flip"] = np1(c.flip);
        std::vector<uint8_t> valid(c.dirs.size(), 1);
        d["dirs"] = fds(c.dirs, valid);
        d["columns"] = c.columns; d["nodes"] = c.nodes; d["g_table_equal"] = c.g_table_equal;
        return d;
    }, py::arg("fam"), py::arg("alts"), py::arg("stored") = true, py::arg("uz") = py::none());
    // per storage column its record kind (0 detector, 1 observable, 2 decision) and index (lean D1-S3, S2 review L3: no
    // packed field in the test API; the engine's own record order is RecColumn::order)
    m.def("rot_columns", [](const py::object& fam) {
        const std::vector<int> ord = engine_of(fam).record_column_order();
        std::vector<int> kind(ord.size()), index(ord.size());
        for (size_t c = 0; c < ord.size(); ++c) { kind[c] = ord[c] >> 24; index[c] = ord[c] & ((1 << 24) - 1); }
        py::dict d;
        d["kind"] = np1(kind);
        d["index"] = np1(index);
        return d;
    }, py::arg("fam"));
    m.def("rot_virtual_check", [](const py::object& fam, long nsub, uint64_t seed) { return virtual_check(engine_of(fam), nsub, seed); },
          py::arg("fam"), py::arg("nsub") = 256, py::arg("seed") = 1,
          "lean D1-S1 TEST API (review H1): (subsets, violations) of 'virtual XOR = 0 <=> the extras' product commutes with the group'.");
}
