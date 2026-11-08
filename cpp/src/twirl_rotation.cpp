// twirl_rotation — lean D1-S1: the rotation model, compiled once from the term view (qeccore/twirl_rotation.hpp).
// Every quantity here is a pure function of (the term view's axes, the certified group planes, the bare state, the
// record columns); the test API's oracle (cpp/testapi/rotation_py.cpp: the P2/D1 probe's own construction) recomputes
// all of it independently and tests/test_lean_d1s1.py holds the two equal on the public corpus.
#include "qeccore/twirl_rotation.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "qeccore/twirl_kernel.hpp"
#include "qeccore/branch_scaffold.hpp"
#include <cstring>

namespace qeccore {

namespace {

inline bool getb(const uint64_t* w, int i) { return (w[i >> 6] >> (i & 63)) & 1ULL; }
inline void setb(uint64_t* w, int i) { w[i >> 6] |= 1ULL << (i & 63); }
inline void flipb(uint64_t* w, int i) { w[i >> 6] ^= 1ULL << (i & 63); }
inline int lead_of(const uint64_t* v, int W) {
    for (int w = 0; w < W; ++w) if (v[w]) return w * 64 + __builtin_ctzll(v[w]);
    return -1;
}
inline bool any_of(const uint64_t* v, int W) { for (int w = 0; w < W; ++w) if (v[w]) return true; return false; }
inline void xor_into(uint64_t* a, const uint64_t* b, int W) { for (int w = 0; w < W; ++w) a[w] ^= b[w]; }

using Echelon = RotEchelon;   // lean D1-S2: the one echelon (declared in the header for the builder)

}  // namespace

size_t RotationModel::bytes() const {
    auto vb = [](const auto& v) -> size_t { return v.capacity() * sizeof(v[0]); };
    size_t b = vb(rho) + vb(scan) + vb(rank_of) + vb(cols_off) + vb(cols) + vb(eta) + vb(live) + vb(cone_gen) +
               vb(cone_row) + vb(gn_rows) + vb(gn_piv) + vb(gn_col_to_row) + vb(gn_a) + vb(gn_b) + vb(gn_phase) + vb(gn_par) + vb(x_T0) +
               vb(x_logical) + vb(x_fd);
    for (const FreeDir& d : x_fd) b += vb(d.a) + vb(d.b);
    return b;
}

void build_rotation_model(const TermResidual& tr, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                          const std::vector<RecColumn>& cols, int nch, const std::vector<uint64_t>& colx_on_wire,
                          const std::vector<uint64_t>& colz_on_wire, int QCW, RotationModel& M) {
    M = RotationModel();
    const int n = tr.n(), NW = (n + 63) / 64;
    M.t = tr.num_terms();
    M.TW = tr.term_words();
    M.k = bare.k();
    M.KW = std::max(1, (M.k + 63) / 64);
    M.m = (int)cols.size();
    M.nch = nch;
    M.ng = G.n_gens;
    const int t = M.t, TW = M.TW, m = M.m, ng = M.ng, GW = G.words;
    if ((size_t)QCW * 64 < (size_t)m) throw std::logic_error("twirl_rotation: column rows narrower than the columns (engine bug)");
    // record order
    M.scan.resize((size_t)m);
    for (int c = 0; c < m; ++c) M.scan[(size_t)c] = c;
    std::sort(M.scan.begin(), M.scan.end(), [&](int x, int y) {
        return cols[(size_t)x].order != cols[(size_t)y].order ? cols[(size_t)x].order < cols[(size_t)y].order : x < y; });
    M.rank_of.assign((size_t)m, 0);
    for (int i = 0; i < m; ++i) M.rank_of[(size_t)M.scan[(size_t)i]] = i;
    // axes (lean D1-S3: any commuting Hermitian Pauli, not only Z-strings): B_j = (−1)^{η_j} B̂_j with B̂_j = i^{|x ∧ z|}
    // X^x Z^z the positive Hermitian form (a Z-string: B̂ = Z^z, η = phase / 2 — the S1 convention, unchanged)
    M.eta.assign((size_t)TW, 0);
    M.live.assign((size_t)TW, 0);
    std::vector<Pauli> Bh((size_t)t);
    for (int j = 0; j < t; ++j) {
        const Pauli& B = tr.axis(j);
        int y = 0;
        for (int w = 0; w < NW; ++w) y += __builtin_popcountll(B.x[(size_t)w] & B.z[(size_t)w]);
        const int d = (B.phase - y) & 3;
        if (d & 1) throw std::logic_error("twirl_rotation: a non-Hermitian rotation axis (engine bug)");
        if (d == 2) setb(M.eta.data(), j);
        Bh[(size_t)j] = B;
        Bh[(size_t)j].phase = (uint8_t)(y & 3);
    }
    // ρ_c and cols_of: row_j over the columns = ⊕_{q ∈ z(B_j)} colx_on_wire[q] ⊕ ⊕_{q ∈ x(B_j)} colz_on_wire[q] (the columns
    // B_j anticommutes with); its set bits are cols_of[j]
    M.rho.assign((size_t)m * (size_t)TW, 0);
    M.cols_off.assign((size_t)t + 1, 0);
    {
        std::vector<uint64_t> row((size_t)QCW, 0);
        for (int j = 0; j < t; ++j) {
            std::fill(row.begin(), row.end(), 0);
            const Pauli& B = tr.axis(j);
            for (int w = 0; w < NW; ++w) {
                for (uint64_t word = B.z[(size_t)w]; word; word &= word - 1) {
                    const int q = w * 64 + __builtin_ctzll(word);
                    xor_into(row.data(), &colx_on_wire[(size_t)q * QCW], QCW);
                }
                for (uint64_t word = B.x[(size_t)w]; word; word &= word - 1) {
                    const int q = w * 64 + __builtin_ctzll(word);
                    xor_into(row.data(), &colz_on_wire[(size_t)q * QCW], QCW);
                }
            }
            for (int w = 0; w < QCW; ++w)
                for (uint64_t word = row[(size_t)w]; word; word &= word - 1) {
                    const int c = w * 64 + __builtin_ctzll(word);
                    if (c >= m) throw std::logic_error("twirl_rotation: a column row bit beyond the columns (engine bug)");
                    setb(&M.rho[(size_t)c * TW], j);
                    M.cols.push_back(c);
                }
            M.cols_off[(size_t)j + 1] = (int)M.cols.size();
            if (M.cols_off[(size_t)j + 1] != M.cols_off[(size_t)j]) setb(M.live.data(), j);   // some column sees j
        }
    }
    // the generators' rows Φ(e_j)_i = ⟨g_i, B_j⟩ (symplectic: X of g_i on z(B_j), Z of g_i on x(B_j))
    std::vector<uint64_t> gen_rows((size_t)ng * (size_t)TW, 0);
    {
        std::vector<uint64_t> col((size_t)GW, 0);
        for (int j = 0; j < t; ++j) {
            std::fill(col.begin(), col.end(), 0);
            const Pauli& B = tr.axis(j);
            for (int w = 0; w < NW; ++w) {
                for (uint64_t word = B.z[(size_t)w]; word; word &= word - 1)
                    xor_into(col.data(), G.col(w * 64 + __builtin_ctzll(word), /*xblock=*/true), GW);
                for (uint64_t word = B.x[(size_t)w]; word; word &= word - 1)
                    xor_into(col.data(), G.col(w * 64 + __builtin_ctzll(word), /*xblock=*/false), GW);
            }
            for (int w = 0; w < GW; ++w)
                for (uint64_t word = col[(size_t)w]; word; word &= word - 1) {
                    const int i = w * 64 + __builtin_ctzll(word);
                    if (i >= ng) throw std::logic_error("twirl_rotation: a plane bit beyond the generators (engine bug)");
                    setb(&gen_rows[(size_t)i * TW], j);
                }
        }
    }
    for (int i = 0; i < ng; ++i)
        if (TW && any_of(&gen_rows[(size_t)i * TW], TW)) {
            M.cone_gen.push_back(i);
            M.cone_row.insert(M.cone_row.end(), &gen_rows[(size_t)i * TW], &gen_rows[(size_t)i * TW] + TW);
        }
    // Γ_n = ker Φ: RREF of the generator rows (t-bit), the kernel from its non-pivot columns, then its own RREF
    {
        Echelon E; E.reset(std::max(TW, 1), 0);
        std::vector<uint64_t> v((size_t)std::max(TW, 1));
        for (int i : M.cone_gen) {
            std::copy(&gen_rows[(size_t)i * TW], &gen_rows[(size_t)i * TW] + TW, v.begin());
            E.reduce(v.data(), nullptr);
            if (any_of(v.data(), TW)) E.insert(v.data(), nullptr);
        }
        M.phi_rank = E.size();
        Echelon K; K.reset(std::max(TW, 1), 0);
        std::vector<uint8_t> is_piv((size_t)std::max(t, 1), 0);
        for (int p : E.piv) is_piv[(size_t)p] = 1;
        for (int f = 0; f < t; ++f) {
            if (is_piv[(size_t)f]) continue;
            std::fill(v.begin(), v.end(), 0);
            setb(v.data(), f);
            for (int r = 0; r < E.size(); ++r) if (getb(&E.rows[(size_t)r * E.W], f)) setb(v.data(), E.piv[(size_t)r]);
            K.reduce(v.data(), nullptr);
            if (any_of(v.data(), TW)) K.insert(v.data(), nullptr);
        }
        // rows by ascending pivot (the RREF is unique; the order is a convention)
        std::vector<int> ord((size_t)K.size());
        for (int r = 0; r < K.size(); ++r) ord[(size_t)r] = r;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return K.piv[(size_t)a] < K.piv[(size_t)b]; });
        M.gn_col_to_row.assign((size_t)std::max(t, 1), -1);
        for (int r : ord) {
            const uint64_t* row = &K.rows[(size_t)r * K.W];
            // every kernel row is annihilated by every generator row
            for (int i : M.cone_gen) {
                int par = 0;
                for (int w = 0; w < TW; ++w) par ^= __builtin_popcountll(row[w] & gen_rows[(size_t)i * TW + (size_t)w]) & 1;
                if (par) throw std::logic_error("twirl_rotation: a Γ_n row outside ker Φ (engine bug)");
            }
            // Π_{j ∈ row} B̂_j (the literal product, exact phase: commuting, order-free) → free_dir_of: (a, b) linear on Γ_n
            // and the effective sign. fd_of_ZT_ adds |T| + 2η(T): Z_T = Π (iB_j) = i^{|T|} (−1)^{η(T)} Π B̂_j.
            Pauli Zw(n);
            for (int w = 0; w < TW; ++w)
                for (uint64_t word = row[w]; word; word &= word - 1)
                    Zw = Pauli::multiply(Zw, Bh[(size_t)(w * 64 + __builtin_ctzll(word))]);
            const Membership mb = G.reduce(Zw);
            if (mb.verdict == Membership::ANTI) throw std::logic_error("twirl_rotation: a Γ_n element anticommutes with the group (engine bug)");
            const FreeDir fd = free_dir_of(bare, Zw);
            // the compact form of the row's Pauli (the literal product of the B̂_j; X content allowed since D1-S3) in the group's normalizer: X^a Z^b on the bare frame's free register (the
            // frame can carry X content: the P2 probe's pure-Z assumption holds on the canaries only), Hermitian
            if (!fd_hermitian(fd)) throw std::logic_error("twirl_rotation: a non-Hermitian Γ_n direction (engine bug)");
            std::vector<uint64_t> ab((size_t)2 * M.KW, 0);
            for (int d = 0; d < M.k; ++d) {
                if (fd.a[(size_t)d]) setb(ab.data(), d);
                if (fd.b[(size_t)d]) setb(ab.data() + M.KW, d);
            }
            const bool any = any_of(ab.data(), 2 * M.KW);
            if (mb.verdict == Membership::IN_GROUP && any) throw std::logic_error("twirl_rotation: IN_GROUP with free content (engine bug)");
            if (mb.verdict == Membership::LOGICAL && !any) throw std::logic_error("twirl_rotation: LOGICAL without free content (engine bug)");
            const int pr = (int)M.gn_piv.size();
            M.gn_rows.insert(M.gn_rows.end(), row, row + TW);
            M.gn_piv.push_back(K.piv[(size_t)r]);
            M.gn_col_to_row[(size_t)K.piv[(size_t)r]] = pr;
            M.gn_a.insert(M.gn_a.end(), ab.begin(), ab.begin() + M.KW);
            M.gn_b.insert(M.gn_b.end(), ab.begin() + M.KW, ab.end());
            M.gn_phase.push_back((uint8_t)(fd.phase & 3));
            M.gn_par.push_back((uint8_t)(fd.par & 1));
        }
        // dim Γ_z = dim Γ_n − rank of the (a | b) directions
        Echelon D; D.reset(2 * M.KW, 0);
        std::vector<uint64_t> bv((size_t)2 * M.KW);
        for (size_t r = 0; r < M.gn_piv.size(); ++r) {
            std::copy(&M.gn_a[r * M.KW], &M.gn_a[r * M.KW] + M.KW, bv.begin());
            std::copy(&M.gn_b[r * M.KW], &M.gn_b[r * M.KW] + M.KW, bv.begin() + M.KW);
            D.reduce(bv.data(), nullptr);
            if (any_of(bv.data(), 2 * M.KW)) D.insert(bv.data(), nullptr);
        }
        M.gz_dim = (int)M.gn_piv.size() - D.size();
    }
    // extras: φ(M_e) reduced against the RREF of the rotation planes Φ(e_j) (ng-bit rows, t-bit combinations) gives
    // T0_e (real bits) and a residual; the residuals' own RREF assigns the virtual coordinates
    M.nx = m - nch;
    {
        const int GWn = std::max(GW, 1), TWn = std::max(TW, 1);
        Echelon P; P.reset(GWn, TWn);
        std::vector<uint64_t> v((size_t)GWn), comb((size_t)TWn);
        for (int j = 0; j < t; ++j) {
            std::fill(v.begin(), v.end(), 0);
            for (int i : M.cone_gen) if (getb(&gen_rows[(size_t)i * TW], j)) setb(v.data(), i);
            std::fill(comb.begin(), comb.end(), 0);
            setb(comb.data(), j);
            P.reduce(v.data(), comb.data());
            if (any_of(v.data(), GW)) P.insert(v.data(), comb.data());
        }
        // pass 1: residuals and real T0 per extra; the residuals' echelon with a virtual-coordinate word per row
        const int VW = std::max(1, (M.nx + 63) / 64);
        Echelon V; V.reset(GWn, VW);
        std::vector<std::vector<uint64_t>> T0((size_t)M.nx), vbits((size_t)M.nx);
        std::vector<std::vector<uint64_t>> resid((size_t)M.nx);
        M.x_logical.assign((size_t)M.nx, 0);
        std::vector<uint64_t> vb((size_t)VW);
        for (int e = 0; e < M.nx; ++e) {
            std::vector<uint64_t> ph;
            phi_of(G, cols[(size_t)(nch + e)].M, ph);
            ph.resize((size_t)GWn, 0);
            T0[(size_t)e].assign((size_t)TWn, 0);
            P.reduce(ph.data(), T0[(size_t)e].data());
            const bool logical = !any_of(ph.data(), GW);
            M.x_logical[(size_t)e] = logical ? 1 : 0;
            if (!logical) ++M.anti_extras;
            resid[(size_t)e] = ph;
            // the extra's word = the coordinates of ITS residual plane in the basis {the planes that opened a coordinate}:
            // reduced fully against the rows → the rows' coordinates; a residual left opens coordinate nvirt and the
            // extra's word is THAT coordinate alone (its plane is the new basis vector); the echelon ROW (the reduced
            // residual = plane ⊕ Σ used rows) carries new ⊕ used. (S1 review H1: the two were conflated.)
            std::fill(vb.begin(), vb.end(), 0);
            V.reduce(ph.data(), vb.data());
            if (any_of(ph.data(), GW)) {
                std::vector<uint64_t> rowc = vb;
                setb(rowc.data(), M.nvirt);
                V.insert(ph.data(), rowc.data());
                std::fill(vb.begin(), vb.end(), 0);
                setb(vb.data(), M.nvirt);
                ++M.nvirt;
            }
            vbits[(size_t)e] = vb;
        }
        M.TXW = std::max(1, (t + M.nvirt + 63) / 64);
        M.x_T0.assign((size_t)M.nx * (size_t)M.TXW, 0);
        for (int e = 0; e < M.nx; ++e) {
            uint64_t* dst = &M.x_T0[(size_t)e * M.TXW];
            for (int w = 0; w < TW; ++w) dst[w] = T0[(size_t)e][(size_t)w];
            for (int vcoord = 0; vcoord < M.nvirt; ++vcoord) if (getb(vbits[(size_t)e].data(), vcoord)) setb(dst, t + vcoord);
            FreeDir fd;
            if (M.x_logical[(size_t)e]) {
                // Q_e0 = M_e · Π_{j ∈ T0 (real)} (i B_j), multiplied on the right
                Pauli Q = cols[(size_t)(nch + e)].M;
                for (int w = 0; w < TW; ++w)
                    for (uint64_t word = T0[(size_t)e][(size_t)w]; word; word &= word - 1) {
                        Pauli iB = tr.axis(w * 64 + __builtin_ctzll(word));
                        iB.phase = (uint8_t)((iB.phase + 1) & 3);
                        Q = Pauli::multiply(Q, iB);
                    }
                fd = free_dir_of(bare, Q);
            }
            M.x_fd.push_back(std::move(fd));
        }
    }
    M.built = true;
}


// ══════════════════════════════ lean D1-S2: the echelon, the builder and the plan store ══════════════════════════════
void RotEchelon::reset(int w, int cw) {
    W = w; CW = cw; rows.clear(); comb.clear(); piv.clear(); col_to_row.assign((size_t)W * 64, -1);
}
void RotEchelon::reduce(uint64_t* v, uint64_t* c) const {
    for (int w = 0; w < W; ++w) {
        uint64_t word = v[w];
        while (word) {
            const int b = __builtin_ctzll(word);
            const int r = col_to_row[(size_t)(w * 64 + b)];
            if (r >= 0) {
                xor_into(v, &rows[(size_t)r * W], W);
                if (c && CW) xor_into(c, &comb[(size_t)r * CW], CW);
                word = (b == 63) ? 0 : (v[w] & (~0ULL << (b + 1)));
            } else {
                word &= word - 1;
            }
        }
    }
}
int RotEchelon::insert(const uint64_t* v, const uint64_t* c) {
    const int ld = lead_of(v, W);
    if (ld < 0) throw std::logic_error("twirl_rotation: echelon insert of a zero row (engine bug)");
    const size_t R = piv.size();
    for (size_t r = 0; r < R; ++r)
        if (getb(&rows[r * W], ld)) {
            xor_into(&rows[r * W], v, W);
            if (c && CW) xor_into(&comb[r * CW], c, CW);
        }
    rows.insert(rows.end(), v, v + W);
    if (CW) comb.insert(comb.end(), c, c + CW);
    piv.push_back(ld);
    col_to_row[(size_t)ld] = (int)R;
    return ld;
}

namespace {
inline int pop_words(const uint64_t* a, int W) { int c = 0; for (int w = 0; w < W; ++w) c += __builtin_popcountll(a[w]); return c; }
inline int pop_and(const uint64_t* a, const uint64_t* b, int W) { int c = 0; for (int w = 0; w < W; ++w) c += __builtin_popcountll(a[w] & b[w]); return c; }
}  // namespace

RotBuilder::RotBuilder(const RotationModel& M, const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                       const std::vector<RecColumn>& cols, const std::vector<Pauli>& axes)
    : M_(M), G_(G), bare_(bare), cols_(cols), axes_(axes) {
    rmask_.assign((size_t)std::max(M.TW, 1), 0);
    for (int j = 0; j < M.t; ++j) setb(rmask_.data(), j);
}

void RotBuilder::fd_of_ZT_(const uint64_t* T, FreeDir& out) const {
    const RotationModel& M = M_;
    const int TW = M.TW, KW = M.KW, k = M.k;
    out = fd_identity(k);
    FreeDir row;
    row.a.assign((size_t)k, 0); row.b.assign((size_t)k, 0); row.hermitian = 1;
    std::vector<uint64_t> acc((size_t)std::max(TW, 1), 0);
    for (size_t r = 0; r < M.gn_piv.size(); ++r) {
        if (!getb(T, M.gn_piv[r])) continue;
        for (int d = 0; d < k; ++d) {
            row.a[(size_t)d] = (uint8_t)getb(&M.gn_a[r * (size_t)KW], d);
            row.b[(size_t)d] = (uint8_t)getb(&M.gn_b[r * (size_t)KW], d);
        }
        row.phase = M.gn_phase[r];
        row.par = M.gn_par[r];
        out = fd_mul(out, row);
        xor_into(acc.data(), &M.gn_rows[r * (size_t)TW], TW);
    }
    for (int w = 0; w < TW; ++w)
        if (acc[(size_t)w] != T[w]) throw std::logic_error("twirl_rotation: a non-pivot word outside Γ_n (engine bug)");
    const int pop = pop_words(T, TW), eta = pop_and(T, M.eta.data(), TW) & 1;
    out.phase = (uint8_t)((out.phase + pop + 2 * eta) & 3);
    out.hermitian = fd_hermitian(out) ? 1 : 0;
}

const FreeDir& RotBuilder::product_fd_(int c, const std::vector<int>& pivs) {
    key_.assign(1, c);
    key_.insert(key_.end(), pivs.begin(), pivs.end());
    std::string key(reinterpret_cast<const char*>(key_.data()), key_.size() * sizeof(int));
    auto it = prod_memo_.find(key);
    if (it != prod_memo_.end()) return it->second;
    const RotationModel& M = M_;
    const int TW = M.TW, nch = M.nch, t = M.t;
    // the base operator Q_x0 = M_x · Π_{j ∈ T0_x (real)} (i B_j), multiplied on the right in ascending j (the S1 x_fd convention)
    auto base_op = [&](int col) {
        Pauli Q = cols_[(size_t)col].M;
        if (col >= nch) {
            const uint64_t* T0 = &M.x_T0[(size_t)(col - nch) * (size_t)M.TXW];
            for (int w = 0; w < TW; ++w)
                for (uint64_t word = T0[w]; word; word &= word - 1) {
                    const int j = w * 64 + __builtin_ctzll(word);
                    if (j >= t) continue;                       // a virtual coordinate: not a rotation
                    Pauli iB = axes_[(size_t)j];
                    iB.phase = (uint8_t)((iB.phase + 1) & 3);
                    Q = Pauli::multiply(Q, iB);
                }
        }
        return Q;
    };
    Pauli Q = base_op(c);
    for (int p : pivs) Q = Pauli::multiply(Q, base_op(p));
    ++anti_products_;
    return prod_memo_.emplace(std::move(key), free_dir_of(bare_, Q)).first->second;
}

void RotBuilder::build(const uint64_t* s, RotBuild& out) {
    const RotationModel& M = M_;
    const int TW = M.TW, TXW = M.TXW, nch = M.nch, m = M.m, t = M.t, k = M.k, KW = M.KW;
    out.TW = TW; out.TXW = TXW;
    out.cols.clear(); out.pivot.clear(); out.beta.clear(); out.flip.clear(); out.A.clear(); out.B.clear();
    out.pivot_cols.clear(); out.pivot_T.clear(); out.pivot_sg.clear(); out.dirs.clear(); out.intro_cols.clear();
    out.ncoin = out.ndir = 0;
    // the touched channels (the inverted index over the set bits of s, deduplicated by a stamp) and every extra
    if ((int)out.stamp.size() != m) { out.stamp.assign((size_t)m, -1); out.epoch = 0; }
    ++out.epoch;
    out.tcols.clear();
    for (int w = 0; w < TW; ++w)
        for (uint64_t word = s[w]; word; word &= word - 1) {
            const int j = w * 64 + __builtin_ctzll(word);
            if (j >= t) throw std::logic_error("twirl_rotation: a key bit beyond the rotations (engine bug)");
            for (int i = M.cols_off[(size_t)j]; i < M.cols_off[(size_t)j + 1]; ++i) {
                const int c = M.cols[(size_t)i];
                if (c >= nch || out.stamp[(size_t)c] == out.epoch) continue;
                out.stamp[(size_t)c] = out.epoch;
                out.tcols.push_back(c);
            }
        }
    for (int c = nch; c < m; ++c) out.tcols.push_back(c);
    std::sort(out.tcols.begin(), out.tcols.end(), [&](int x, int y) { return M.rank_of[(size_t)x] < M.rank_of[(size_t)y]; });
    const int nrow = (int)out.tcols.size();
    const int AW = std::max(1, (nrow + 63) / 64), BW = AW;      // ρ, κ ≤ rows
    out.AW = AW; out.BW = BW;
    out.pech.reset(TXW, AW);
    out.dech.reset(2 * KW, BW);
    out.Tc.assign((size_t)TXW, 0); out.Tp.assign((size_t)TXW, 0); out.r.assign((size_t)TXW, 0); out.TK.assign((size_t)TXW, 0);
    out.comb.assign((size_t)AW, 0); out.cb.assign((size_t)AW, 0); out.dc.assign((size_t)BW, 0); out.r2.assign((size_t)2 * KW, 0);
    std::vector<int> pivs;
    for (int c : out.tcols) {
        uint64_t* Tc = out.Tc.data();
        uint64_t* Tp = out.Tp.data();
        for (int w = 0; w < TXW; ++w) { Tc[w] = 0; Tp[w] = 0; }
        for (int w = 0; w < TW; ++w) Tc[w] = s[w] & M.rho[(size_t)c * (size_t)TW + (size_t)w];
        const bool extra = c >= nch;
        int sg = 0;
        for (int w = 0; w < TXW; ++w) Tp[w] = Tc[w];
        if (extra) {
            const uint64_t* T0 = &M.x_T0[(size_t)(c - nch) * (size_t)TXW];
            // |T0| and |T0 ∧ T_c| over the REAL bits only (a virtual coordinate is bookkeeping, not a rotation)
            int p0 = 0, p1 = 0;
            for (int w = 0; w < TW; ++w) {
                const uint64_t rm = rmask_[(size_t)w];
                p0 += __builtin_popcountll(T0[w] & rm);
                p1 += __builtin_popcountll(T0[w] & Tc[w] & rm);
            }
            sg = (p0 + p1) & 1;
            for (int w = 0; w < TXW; ++w) Tp[w] ^= T0[w];
        }
        // the plane: T'_c mod Γ_n (real bits), then against the earlier pivots (virtual bits ride along)
        uint64_t* r = out.r.data();
        for (int w = 0; w < TXW; ++w) r[w] = Tp[w];
        for (int w = 0; w < TW; ++w) {
            uint64_t word = r[w] & rmask_[(size_t)w];
            while (word) {
                const int b = __builtin_ctzll(word);
                const int row = M.gn_col_to_row[(size_t)(w * 64 + b)];
                if (row >= 0) {
                    xor_into(r, &M.gn_rows[(size_t)row * (size_t)TW], TW);
                    word = (b == 63) ? 0 : (r[w] & rmask_[(size_t)w] & (~0ULL << (b + 1)));
                } else {
                    word &= word - 1;
                }
            }
        }
        uint64_t* comb = out.comb.data();
        for (int w = 0; w < AW; ++w) comb[w] = 0;
        out.pech.reduce(r, comb);
        out.cols.push_back(c);
        if (any_of(r, TXW)) {                          // a PIVOT: coin slot = its own value
            const int slot = out.ncoin++;
            uint64_t* cb = out.cb.data();
            for (int w = 0; w < AW; ++w) cb[w] = comb[w];
            flipb(cb, slot);
            out.pech.insert(r, cb);
            out.pivot_cols.push_back(c);
            out.pivot_T.insert(out.pivot_T.end(), Tp, Tp + TXW);
            out.pivot_sg.push_back((uint8_t)sg);
            // a pivot's RAW base is 0 (Lemma 4: y_p = u_k): its INTERNAL flip is the channel's ref (raw = internal ⊕ ref)
            out.pivot.push_back(1); out.beta.push_back(0); out.flip.push_back((uint8_t)(c < nch ? (cols_[(size_t)c].ref & 1) : 0));
            for (int w = 0; w < AW; ++w) out.A.push_back(w == slot / 64 ? (1ULL << (slot & 63)) : 0ULL);
            for (int w = 0; w < BW; ++w) out.B.push_back(0);
            continue;
        }
        // a non-pivot: K = Q_c · Π_{p ∈ comb (ascending slot)} Q_p (the engine's literal order) = (−1)^sign (Q_c0 Π Q_p0) Z_{T_K}
        uint64_t* TK = out.TK.data();
        for (int w = 0; w < TXW; ++w) TK[w] = Tp[w];
        int sign = sg;
        bool anti = extra && !M.x_logical[(size_t)(c - nch)];
        pivs.clear();
        for (int p = 0; p < out.ncoin; ++p) if (getb(comb, p)) pivs.push_back(p);
        for (int p : pivs) {
            const int pc = out.pivot_cols[(size_t)p];
            const uint64_t* Tb = &out.pivot_T[(size_t)p * (size_t)TXW];
            const uint64_t* rho_b = &M.rho[(size_t)pc * (size_t)TW];
            int mv = 0;                                 // Z_{T_acc} past M_p: |T_acc ∧ ρ_p|; Z_{T_acc} Z_{T'_p}: |T_acc ∩ T'_p| (real bits)
            for (int w = 0; w < TW; ++w) {
                const uint64_t rm = rmask_[(size_t)w];
                mv += __builtin_popcountll(TK[w] & rho_b[w] & rm) + __builtin_popcountll(TK[w] & Tb[w] & rm);
            }
            sign ^= mv & 1;
            if (pc >= nch) { sign ^= out.pivot_sg[(size_t)p]; anti = anti || !M.x_logical[(size_t)(pc - nch)]; }
            for (int w = 0; w < TXW; ++w) TK[w] ^= Tb[w];
        }
        for (int j = t; j < t + M.nvirt; ++j)
            if (getb(TK, j)) throw std::logic_error("twirl_rotation: a non-pivot column with a virtual plane residue (engine bug)");
        FreeDir& fdM = out.fdM;
        if (!anti) {
            auto base_fd = [&](int col, FreeDir& f) {
                if (col < nch) { f = fd_identity(k); f.phase = (uint8_t)(2 * (cols_[(size_t)col].ref & 1)); }
                else f = M.x_fd[(size_t)(col - nch)];
            };
            base_fd(c, fdM);
            FreeDir fp;
            for (int p : pivs) { base_fd(out.pivot_cols[(size_t)p], fp); fdM = fd_mul(fdM, fp); }
        } else {
            fdM = product_fd_(c, [&] { std::vector<int> v; for (int p : pivs) v.push_back(out.pivot_cols[(size_t)p]); return v; }());
            ++out.anti_products;
        }
        fdM.phase = (uint8_t)((fdM.phase + 2 * sign) & 3);
        fd_of_ZT_(TK, out.fdZ);
        FreeDir& Kd = out.Kd;
        Kd = fd_mul(fdM, out.fdZ);
        fd_require_hermitian(Kd, "rotation K");
        uint64_t* r2 = out.r2.data();
        for (int w = 0; w < 2 * KW; ++w) r2[w] = 0;
        for (int d = 0; d < k; ++d) { if (Kd.a[(size_t)d]) setb(r2, d); if (Kd.b[(size_t)d]) setb(r2 + KW, d); }
        uint64_t* dc = out.dc.data();
        for (int w = 0; w < BW; ++w) dc[w] = 0;
        out.dech.reduce(r2, dc);
        out.pivot.push_back(0);
        for (int w = 0; w < AW; ++w) out.A.push_back(comb[w]);
        if (any_of(r2, 2 * KW)) {                       // a new direction
            const int j = out.ndir++;
            FreeDir L = Kd; L.hermitian = 1;
            out.dirs.push_back(std::move(L));
            out.intro_cols.push_back(c);
            uint64_t* cb = out.cb.data();
            for (int w = 0; w < BW; ++w) cb[w] = dc[w];
            flipb(cb, j);
            out.dech.insert(r2, cb);
            for (int w = 0; w < BW; ++w) out.B.push_back(w == j / 64 ? (1ULL << (j & 63)) : 0ULL);
            out.beta.push_back(0);                      // an introducer's RAW base is 0 (Lemma 4): the flip is the ref
            out.flip.push_back((uint8_t)(c < nch ? (cols_[(size_t)c].ref & 1) : 0));
        } else {
            FreeDir& prod = out.prod;
            prod = fd_identity(k);
            for (int j = 0; j < out.ndir; ++j) if (getb(dc, j)) prod = fd_mul(prod, out.dirs[(size_t)j]);
            const int dph = (fd_eff_phase(Kd) - fd_eff_phase(prod)) & 3;
            if (dph & 1) throw std::logic_error("twirl_rotation: a non-Hermitian ratio of a column to its directions (engine bug)");
            for (int w = 0; w < BW; ++w) out.B.push_back(dc[w]);
            const uint8_t beta = (uint8_t)(dph == 2 ? 1 : 0);
            out.beta.push_back(beta);
            out.flip.push_back((uint8_t)(beta ^ (c < nch ? (cols_[(size_t)c].ref & 1) : 0)));
        }
    }
}

// ── the store ──
void RotPlanStore::init(int TW) {
    TW_ = TW;
    arena_.clear();
    slots_.assign((size_t)1 << 16, Slot{0, 0});
    mask_ = slots_.size() - 1;
    used_ = plans_ = misses_k0_ = misses_kpos_ = evictions_ = rebuilds_ = 0;
    tuples_.clear();
    tuple_ix_.clear();
    plan_off_.clear();
}
void RotPlanStore::clear() { init(TW_); }
void RotPlanStore::grow_() {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(old.size() * 2, Slot{0, 0});
    mask_ = slots_.size() - 1;
    for (const Slot& sl : old)
        if (sl.o) { size_t i = (size_t)sl.h & mask_; while (slots_[i].o) i = (i + 1) & mask_; slots_[i] = sl; }
}
void RotPlanStore::index_insert_(uint64_t h, size_t o) {
    if ((used_ + 1) * 2 > slots_.size()) grow_();
    size_t i = (size_t)h & mask_;
    while (slots_[i].o) i = (i + 1) & mask_;
    slots_[i] = Slot{h, o};
    ++used_;
}
std::string RotPlanStore::tuple_key(const std::vector<FreeDir>& dirs) {
    std::string k;
    const uint32_t n = (uint32_t)dirs.size();
    k.append(reinterpret_cast<const char*>(&n), sizeof n);
    for (const FreeDir& d : dirs) {
        k.append(reinterpret_cast<const char*>(d.a.data()), d.a.size());
        k.push_back('\xff');
        k.append(reinterpret_cast<const char*>(d.b.data()), d.b.size());
        k.push_back((char)(d.phase & 3));
        k.push_back((char)(d.par & 1));
    }
    return k;
}
uint64_t RotPlanStore::tuple_intern_(const std::string& key, std::vector<FreeDir>&& dirs) {
    auto it = tuple_ix_.find(key);
    if (it != tuple_ix_.end()) return it->second;
    RotTuple T;
    T.key = key;
    T.dirs = std::move(dirs);
    tuples_.push_back(std::move(T));
    tuple_ix_.emplace(key, (uint64_t)(tuples_.size() - 1));
    return (uint64_t)(tuples_.size() - 1);
}
size_t RotPlanStore::insert(const uint64_t* s, uint64_t h, const RotBuild& b, uint64_t ordinal, const RotPlanStore* home) {
    const int rho = b.ncoin, kap = b.ndir;
    const int AW = (rho + 63) / 64, BW = (kap + 63) / 64;
    uint64_t tref = kNoTuple;
    if (kap) {
        const std::string key = tuple_key(b.dirs);
        const uint64_t ix = home ? home->tuple_index(key) : kAbsent;
        if (ix != kAbsent) tref = ix | kHomeTuple;
        else tref = tuple_intern_(key, std::vector<FreeDir>(b.dirs));
    }
    int ne = 0;
    for (size_t i = 0; i < b.cols.size(); ++i) {
        bool any = b.flip[i] != 0;
        for (int w = 0; w < AW && !any; ++w) any = b.A[i * (size_t)b.AW + (size_t)w] != 0;
        for (int w = 0; w < BW && !any; ++w) any = b.B[i * (size_t)b.BW + (size_t)w] != 0;
        if (any) ++ne;
    }
    const size_t o = arena_.size() + 1;
    for (int w = 0; w < TW_; ++w) arena_.push_back(s[w]);
    arena_.push_back((uint64_t)ne);
    arena_.push_back((uint64_t)(uint32_t)rho | ((uint64_t)(uint32_t)kap << 32));
    arena_.push_back((uint64_t)(uint32_t)AW | ((uint64_t)(uint32_t)BW << 32));
    arena_.push_back(tref);
    arena_.push_back(ordinal);
    for (int i = 0; i < rho; i += 2)
        arena_.push_back((uint64_t)(uint32_t)b.pivot_cols[(size_t)i] | (i + 1 < rho ? (uint64_t)(uint32_t)b.pivot_cols[(size_t)i + 1] << 32 : 0ULL));
    for (int i = 0; i < kap; i += 2)
        arena_.push_back((uint64_t)(uint32_t)b.intro_cols[(size_t)i] | (i + 1 < kap ? (uint64_t)(uint32_t)b.intro_cols[(size_t)i + 1] << 32 : 0ULL));
    for (size_t i = 0; i < b.cols.size(); ++i) {
        bool nondet = false;
        for (int w = 0; w < AW && !nondet; ++w) nondet = b.A[i * (size_t)b.AW + (size_t)w] != 0;
        for (int w = 0; w < BW && !nondet; ++w) nondet = b.B[i * (size_t)b.BW + (size_t)w] != 0;
        if (!nondet && !b.flip[i]) continue;
        arena_.push_back((uint64_t)(uint32_t)b.cols[i] | ((uint64_t)(b.flip[i] & 1) << 32) | ((uint64_t)(nondet ? 1 : 0) << 33));
        for (int w = 0; w < AW; ++w) arena_.push_back(b.A[i * (size_t)b.AW + (size_t)w]);
        for (int w = 0; w < BW; ++w) arena_.push_back(b.B[i * (size_t)b.BW + (size_t)w]);
    }
    index_insert_(h, o);
    ++plans_;
    plan_off_.push_back(o);
    if (kap) ++misses_kpos_; else ++misses_k0_;
    return o;
}
size_t RotPlanStore::plan_bytes() const {
    size_t b = arena_.size() * sizeof(uint64_t) + slots_.size() * sizeof(Slot);
    for (const RotTuple& T : tuples_) b += T.key.size() + 32 + T.dirs.size() * (sizeof(FreeDir) + 8) + (T.dirs.empty() ? 0 : 2 * T.dirs[0].a.size() * T.dirs.size());
    return b;
}
size_t RotPlanStore::trie_bytes() const {
    size_t b = 0;
    for (const RotTuple& T : tuples_) b += T.trie_bytes();
    return b;
}
void RotPlanStore::clear_tries() {
    for (RotTuple& T : tuples_) {
        if (T.trie.empty()) continue;
        T.trie.clear(); T.trie.shrink_to_fit();
        T.evicted = true;
        ++evictions_;
    }
}
void RotPlanStore::adopt_from(RotPlanStore& L) {
    if (L.TW_ != TW_) throw std::logic_error("twirl_rotation: adopting a store of another key width (engine bug)");
    const std::vector<uint64_t>& la = L.arena_;
    size_t pos = 0;
    while (pos < la.size()) {
        const uint64_t* p = &la[pos];
        const size_t words = L.words_of(p);
        const uint64_t h = plan_key_hash(p, TW_);
        const uint64_t tref = L.tref(p);
        const int kap = L.kappa(p);
        auto home_tuple = [&]() -> uint64_t {          // the home's tuple of this lane plan (moved or merged by content)
            if (tref == kNoTuple) return kNoTuple;
            if (tref & kHomeTuple) return tref & ~kHomeTuple;
            RotTuple& lt = L.tuples_[(size_t)tref];
            uint64_t ix = tuple_index(lt.key);
            if (ix == kAbsent) {
                ix = tuple_intern_(lt.key, std::move(lt.dirs));
                tuples_[ix].trie = std::move(lt.trie);
            } else {
                born_trie_merge(tuples_[ix].trie, lt.trie);
                note_grown(ix);
            }
            lt.trie.clear();
            return ix;
        };
        if (find(p, h) == 0) {
            const size_t o = arena_.size() + 1;
            arena_.insert(arena_.end(), p, p + words);
            arena_[o - 1 + (size_t)TW_ + 3] = home_tuple();
            arena_[o - 1 + (size_t)TW_ + 4] = plans_;
            index_insert_(h, o);
            ++plans_;
            plan_off_.push_back(o);
            if (kap) ++misses_kpos_; else ++misses_k0_;   // a miss = a plan NEW to the home (T-invariant)
        } else {
            (void)home_tuple();                        // a duplicate: its tuple's trie merged, the plan dropped
        }
        pos += words;
    }
    L.clear();                                        // (a duplicate's lane miss is not a home miss: dropped)
}

}  // namespace qeccore
