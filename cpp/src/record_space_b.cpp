// record_space_b — J-a N6: the record table of a diagonal plan with record-space B. See
// qeccore/record_space_b.hpp for the construction and its correctness argument.
#include "qeccore/record_space_b.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "record_table_internal.hpp"   // add_extra, the echelon, the Bits helpers (shared with (D) and (G))
#include "twirl_law_internal.hpp"      // build_shot_law_core

namespace qeccore {

namespace {

// One echelon row over n-bit dressings: the residual v (leading bit `lead`), its U-coefficients cu and its
// pivot combination cp (the vectors it is the sum of). Rows are reduced in insertion order (forward elimination:
// every row was reduced by all earlier rows, so one pass clears every earlier lead).
struct DRow {
    Bits v, cu, cp;
    int lead;
};

// perf-engine §2.2: the echelon's rows are a pooled prefix [0, n) of a vector that keeps every row's capacity
// across builds (a cleared vector<DRow> freed its rows' buffers: ~3 allocations per echelon row per build).
struct DRows {
    std::vector<DRow> rows;
    size_t n = 0;
    void clear() { n = 0; }
    size_t size() const { return n; }
    const DRow& operator[](size_t i) const { return rows[i]; }
    // Append a row by COPY into pooled storage (the row's Bits keep their capacity).
    void push(const Bits& v, const Bits& cu, const Bits& cp, int lead) {
        if (n == rows.size()) rows.emplace_back();
        DRow& r = rows[n++];
        r.v.assign(v.begin(), v.end());
        r.cu.assign(cu.begin(), cu.end());
        r.cp.assign(cp.begin(), cp.end());
        r.lead = lead;
    }
};

inline void reduce_by(const DRows& E, Bits& v, Bits& cu, Bits& cp, long& ops) {
    for (size_t i = 0; i < E.size(); ++i) {
        const DRow& r = E[i];
        if (bit_get(v, r.lead)) {
            bits_xor(v, r.v);
            bits_xor(cu, r.cu);
            bits_xor(cp, r.cp);
            ++ops;
        }
    }
}

int det_parity(const std::vector<uint64_t>& det, const std::vector<uint64_t>& mask) {
    return and_parity(det, mask);
}

// J-b2 J7b (fixplan-J7b-recb-codegen, ruling (B)): the table's per-column row vectors are released OUT OF LINE.
// When ShotLaw lost its three vector<vector<uint64_t>> fields (the σ deletion), GCC started inlining
// vector<vector<uint64_t>>::~vector into build_record_table_recB_from_core, which lengthened its hot loop; this
// noinline release keeps the destructor's body out of the builder.
__attribute__((noinline)) void release_rows_(std::vector<Bits>& rows) { std::vector<Bits>().swap(rows); }

// S8-2 hygiene: the builder's per-call vectors live in ONE per-thread scratch (one TLS access per call; capacity kept
// across plans). Every vector is cleared / re-assigned before use, exactly as the per-call locals were constructed.
struct RecBScratch {
    std::vector<uint8_t> kb, piv_bd, bd;
    std::vector<int> piv_col;
    DRows red, E;
    Bits vv, cu, cp;                     // perf-engine §2.2: the per-column working rows (capacity kept)
    // lean L4: the per-channel active-generator lists (CSR over the columns, gathered through the record-column
    // index), supp(a) and the cz adjacency (CSR over the wires) of the plan — the pullback's sparse inputs
    std::vector<int> coff, clist, alist, adj_off, adj;
    std::vector<uint64_t> active;        // teeth / MUT-B-PAD only: the dense active mask
};
__attribute__((noinline)) RecBScratch& recb_scratch_() {   // out of line: one TLS access per call
    static thread_local RecBScratch sc;
    return sc;
}

}  // namespace

RecColIndex::RecColIndex(const std::vector<RecColumn>& cols, int ng) : m((int)cols.size()), n_gens(ng) {
    order.resize((size_t)m);
    ref.assign((size_t)m, 0);
    for (int c = 0; c < m; ++c) {
        order[(size_t)c] = cols[(size_t)c].order;
        if (cols[(size_t)c].channel) { ch.push_back(c); ref[(size_t)c] = (uint8_t)(cols[(size_t)c].ref & 1); }
        else ex.push_back(c);
    }
    auto by_order = [&](int x, int y) { return order[(size_t)x] < order[(size_t)y]; };
    if (!std::is_sorted(ch.begin(), ch.end(), by_order)) std::sort(ch.begin(), ch.end(), by_order);
    if (!std::is_sorted(ex.begin(), ex.end(), by_order)) std::sort(ex.begin(), ex.end(), by_order);
    goff.assign((size_t)ng + 1, 0);
    for (int c : ch)
        for (size_t w = 0; w < cols[(size_t)c].mask.size(); ++w)
            for (uint64_t x = cols[(size_t)c].mask[w]; x; x &= x - 1) {
                const int i = (int)(w * 64) + __builtin_ctzll(x);
                if (i < ng) ++goff[(size_t)i + 1];
            }
    for (int i = 0; i < ng; ++i) goff[(size_t)i + 1] += goff[(size_t)i];
    gcol.assign((size_t)goff[(size_t)ng], 0);
    std::vector<int> pos(goff.begin(), goff.end() - 1);
    for (int c : ch)                                  // channels visited in record order: not ascending; sorted below
        for (size_t w = 0; w < cols[(size_t)c].mask.size(); ++w)
            for (uint64_t x = cols[(size_t)c].mask[w]; x; x &= x - 1) {
                const int i = (int)(w * 64) + __builtin_ctzll(x);
                if (i < ng) gcol[(size_t)pos[(size_t)i]++] = c;
            }
    for (int i = 0; i < ng; ++i) std::sort(gcol.begin() + goff[(size_t)i], gcol.begin() + goff[(size_t)i + 1]);
}

bool build_record_table_recB(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                             const DiagNormalForm& nf, const std::vector<RecColumn>& cols, RecTable& out,
                             RecBOps* ops) {
    if (!nf.diagonal_class) throw std::logic_error("build_record_table_recB: a non-diagonal plan (engine bug)");
    const int n = G.n_qubits;
    DiagNormalForm nf0 = nf;                          // the table is P-free (Lemma 2): identity prefix
    nf0.prefix = Pauli(n);
    twirl_law_detail::ShotLawCore core;
    const ShotLaw law = twirl_law_detail::build_shot_law_core(G, nf0, core, /*teeth=*/true);
    const RecColIndex idx(cols, G.n_gens);
    const bool ok = build_record_table_recB_from_core(G, bare, nf0, law, core, cols, idx, out, /*teeth=*/true, ops);
    if (ops) ops->law_reduces = law.reduces;
    return ok;
}

// J-b1 (fixplan-coldstart (A)): the B stage on a core law built elsewhere (the FAM plan payload's ONE law build,
// handed over through the plan cache's one-slot core park). The table is P-free (Lemma 2): nf's prefix is never read
// here (the pullback reads a / cz only).
bool build_record_table_recB_from_core(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                       const DiagNormalForm& nf, const ShotLaw& law,
                                       const twirl_law_detail::ShotLawCore& core,
                                       const std::vector<RecColumn>& cols, const RecColIndex& idx, RecTable& out,
                                       bool teeth, RecBOps* ops) {
    if (!nf.diagonal_class) throw std::logic_error("build_record_table_recB: a non-diagonal plan (engine bug)");
    const int n = G.n_qubits;
    const int m = (int)cols.size();
    if (idx.m != m || idx.n_gens != G.n_gens)
        throw std::logic_error("build_record_table_recB: the record-column index is of another family (engine bug)");
    out = RecTable();
    if (law.fallback) return false;
    RecBOps o;
    const int A = (int)core.aidx.size();
    const int K = (int)core.U.size();
    if (K != (int)law.kernel_logicals.size())
        throw std::logic_error("build_record_table_recB: kernel directions != kernel logicals (engine bug)");
    const std::vector<uint64_t>& det = law.det_signs;

    // The κ_σ directions, as (D) stores them (build_rec_): the free-register form of each kernel logical.
    std::vector<FreeDir> lam;
    lam.reserve((size_t)K);
    for (const Pauli& L : law.kernel_logicals) lam.push_back(free_dir_of(bare, L));

    // 1-2. Restricted preimages c_j (over the active generators) and kb_j = s_j ⊕ c_j·det.
    RecBScratch& sc = recb_scratch_();
    std::vector<uint8_t>& kb = sc.kb;
    kb.assign((size_t)K, 0);
    Bits& vv = sc.vv;
    Bits& cu = sc.cu;
    Bits& cp = sc.cp;
    if (K > 0) {
        DRows& red = sc.red;
        red.clear();
        for (int a = 0; a < A; ++a) {
            vv.assign(core.v[(size_t)a].begin(), core.v[(size_t)a].end());
            cu.clear();
            cp.clear();
            bit_flip(cp, a);                          // cp: the active-position combination
            reduce_by(red, vv, cu, cp, o.pre_rows);
            const int ld = bits_lead(vv);
            if (ld >= 0) red.push(vv, cu, cp, ld);
        }
        for (int j = 0; j < K; ++j) {
            Bits& v = vv;
            Bits& cj = cp;
            v.assign(core.U[(size_t)j].begin(), core.U[(size_t)j].end());
            cu.clear();
            cj.clear();
            reduce_by(red, v, cu, cj, o.pre_rows);
            if (bits_any(v))
                throw std::logic_error("build_record_table_recB: a kernel direction outside V (engine bug)");
            // s_j: Π_{a ∈ c_j} h_a = (−1)^{s_j}·(group)·L_j (the h_a commute: conjugates of commuting g's)
            Pauli H(n);
            int cdet = 0;
            for (int a = 0; a < A; ++a) {
                if (!bit_get(cj, a)) continue;
                const int i = core.aidx[(size_t)a];
                Pauli h(n);
                h.x = G.gens[(size_t)i].x;
                h.z = G.gens[(size_t)i].z;
                for (size_t w = 0; w < h.z.size(); ++w) h.z[w] ^= core.v[(size_t)a][w];
                h.phase = core.phi[(size_t)a];
                H = Pauli::multiply(H, h);
                cdet ^= bit_get(det, i) ? 1 : 0;
            }
            const Membership m = G.reduce(H);
            const Pauli& L = law.kernel_logicals[(size_t)j];
            if (m.rep.x != L.x || m.rep.z != L.z)
                throw std::logic_error("build_record_table_recB: preimage product support != kernel logical (engine bug)");
            kb[(size_t)j] = (uint8_t)((((m.rep.phase ^ L.phase) >> 1) & 1) ^ cdet);
        }
    }

    // 3. The channels in record order, Theorem 1 in record space.
    // lean L4: the per-channel inputs are GATHERED from the plan's lightcone through the record-column index instead
    // of scanning every mask over every generator word: bd_c = ref_c ⊕ par(det & mask_c) by walking det's set bits
    // (each flips the channels holding that generator), and each channel's list of the active generators its mask
    // meets by walking the active generators (ascending, so each list ascends: the mask & active order). Same bits,
    // same XORs; `touch` / gen_visits are the list's emptiness / length.
    const std::vector<int>& order = idx.order;
    const std::vector<int>& ch = idx.ch;
    std::vector<uint8_t>& bd = sc.bd;
    bd.assign((size_t)m, 0);
    for (int c : ch) bd[(size_t)c] = idx.ref[(size_t)c];
    {
        const size_t nw = std::min(det.size(), (size_t)((G.n_gens + 63) / 64));
        for (size_t w = 0; w < nw; ++w)
            for (uint64_t x = det[w]; x; x &= x - 1) {
                const int i = (int)(w * 64) + __builtin_ctzll(x);
                if (i >= G.n_gens) continue;
                for (int p = idx.goff[(size_t)i]; p < idx.goff[(size_t)i + 1]; ++p) bd[(size_t)idx.gcol[(size_t)p]] ^= 1;
            }
    }
    std::vector<int>& coff = sc.coff;                 // column c → [coff[c], coff[c+1]) into clist
    std::vector<int>& clist = sc.clist;               // the active positions a whose generator meets column c
    coff.assign((size_t)m + 1, 0);
    for (int a = 0; a < A; ++a) {
        const int i = core.aidx[(size_t)a];
        for (int p = idx.goff[(size_t)i]; p < idx.goff[(size_t)i + 1]; ++p) ++coff[(size_t)idx.gcol[(size_t)p] + 1];
    }
    for (int c = 0; c < m; ++c) coff[(size_t)c + 1] += coff[(size_t)c];
    clist.resize((size_t)coff[(size_t)m]);
    {
        std::vector<int>& fill = sc.alist;            // (scratch: the fill cursors; alist is rebuilt below)
        fill.assign(coff.begin(), coff.end() - 1);
        for (int a = 0; a < A; ++a) {
            const int i = core.aidx[(size_t)a];
            for (int p = idx.goff[(size_t)i]; p < idx.goff[(size_t)i + 1]; ++p)
                clist[(size_t)fill[(size_t)idx.gcol[(size_t)p]]++] = a;
        }
    }
#if defined(XTIM_MUTATE_B_PAD)
    const bool dense_walk = true;                     // mutation build only (MUT-B-PAD): every generator of every mask
#else
    const bool dense_walk = false;
#endif
    std::vector<uint64_t>& active = sc.active;
    if (teeth || dense_walk) {
        active.assign((size_t)G.words, 0);
        for (int i : core.aidx) active[(size_t)(i >> 6)] |= 1ULL << (i & 63);
    }
    if (teeth) {                                      // the gathered per-channel data against the dense scan
        for (int c : ch) {
            const RecColumn& C = cols[(size_t)c];
            if (bd[(size_t)c] != (uint8_t)((det_parity(det, C.mask) ^ (C.ref & 1)) & 1))
                throw std::logic_error("build_record_table_recB: the index-gathered det parity disagrees with the "
                                       "mask scan (engine bug)");
            int p = coff[(size_t)c];
            const size_t nw = std::min(C.mask.size(), active.size());
            for (size_t w = 0; w < nw; ++w)
                for (uint64_t x = C.mask[w] & active[w]; x; x &= x - 1) {
                    const int i = (int)(w * 64) + __builtin_ctzll(x);
                    if (p >= coff[(size_t)c + 1] || core.aidx[(size_t)clist[(size_t)p]] != i)
                        throw std::logic_error("build_record_table_recB: the index-gathered active list disagrees "
                                               "with the mask scan (engine bug)");
                    ++p;
                }
            if (p != coff[(size_t)c + 1])
                throw std::logic_error("build_record_table_recB: the index-gathered active list disagrees with the "
                                       "mask scan (engine bug)");
        }
    }
    AffineTable W;
    W.m = m;
    W.ncoin = 0;
    W.dirs = lam;
    W.A.assign((size_t)m, {});
    W.B.assign((size_t)m, {});
    W.beta.assign((size_t)m, 0);
    DRows& E = sc.E;                                  // D ∪ U, then the pivots' dressings
    E.clear();
    for (const Bits& d : core.D) {
        vv.assign(d.begin(), d.end());
        cu.clear();
        cp.clear();
        reduce_by(E, vv, cu, cp, o.elim_rows);
        const int ld = bits_lead(vv);
        if (ld < 0) throw std::logic_error("build_record_table_recB: dependent D basis (engine bug)");
        E.push(vv, cu, cp, ld);
    }
    for (int j = 0; j < K; ++j) {
        vv.assign(core.U[(size_t)j].begin(), core.U[(size_t)j].end());
        cu.clear();
        cp.clear();
        bit_flip(cu, j);
        reduce_by(E, vv, cu, cp, o.elim_rows);
        const int ld = bits_lead(vv);
        if (ld < 0) throw std::logic_error("build_record_table_recB: kernel direction in span(D) (engine bug)");
        E.push(vv, cu, cp, ld);
    }
    std::vector<int>& piv_col = sc.piv_col;           // pivot slot -> storage column
    std::vector<uint8_t>& piv_bd = sc.piv_bd;         // pivot slot -> par(det & mask) ⊕ ref
    piv_col.clear();
    piv_bd.clear();
#ifdef XTIM_MUTATE_B_ROW_SKIP
    bool mut_row_skipped = false;
#endif
    for (int c : ch) {
        const uint8_t bdc = bd[(size_t)c];
        vv.clear();
        bool touch = false;
        if (dense_walk) {
            // mutation build only (MUT-B-PAD): every generator of the mask is visited, the inactive ones too
            const RecColumn& C = cols[(size_t)c];
            const size_t nw = std::min(C.mask.size(), active.size());
            for (size_t w = 0; w < C.mask.size(); ++w) {
                ++o.mask_words;
                for (uint64_t x = C.mask[w]; x; x &= x - 1) {
                    const int i = (int)(w * 64) + __builtin_ctzll(x);
                    ++o.gen_visits;
                    if (w < nw && ((active[w] >> (i & 63)) & 1ULL)) {
                        int a = 0;
                        while (core.aidx[(size_t)a] != i) ++a;
                        bits_xor(vv, core.v[(size_t)a]);
                        touch = true;
                    }
                }
            }
        } else {
            for (int p = coff[(size_t)c]; p < coff[(size_t)c + 1]; ++p) {
                bits_xor(vv, core.v[(size_t)clist[(size_t)p]]);
                ++o.gen_visits;
                touch = true;
            }
        }
        if (!touch) { W.beta[(size_t)c] = bdc; continue; }   // a constant column: A = B = 0
        ++o.touched;
        cu.clear();
        cp.clear();
        reduce_by(E, vv, cu, cp, o.elim_rows);
        const int ld = bits_lead(vv);
        if (ld >= 0) {                                // a PIVOT: its own fair coin
            const int slot = W.ncoin++;
            bit_flip(cp, slot);
            E.push(vv, cu, cp, ld);
            bit_flip(W.A[(size_t)c], slot);
            piv_col.push_back(c);
            piv_bd.push_back(bdc);
            ++o.pivots;
            continue;
        }
#ifdef XTIM_MUTATE_B_ROW_SKIP
        // mutation build only (MUT-B-ROW): the first channel that reads a direction gets no B row
        if (bits_any(cu) && !mut_row_skipped) { cu.clear(); mut_row_skipped = true; }
#endif
        uint8_t beta = bdc;
        for (int s = 0; s < (int)piv_bd.size(); ++s) if (bit_get(cp, s)) beta ^= piv_bd[(size_t)s];
#ifndef XTIM_MUTATE_B_KB_DROP
        for (int j = 0; j < K; ++j) if (bit_get(cu, j)) beta ^= kb[(size_t)j];
#endif   // mutation build only (MUT-B-KB): the kernel base signs dropped from β
        W.A[(size_t)c] = cp;
        W.B[(size_t)c] = cu;
        W.beta[(size_t)c] = beta;
    }

    if (ops) {                                        // coverage counter (test callers only)
        Bits read;                                    // directions some channel reads
        for (int c : ch) {
            const Bits& b = W.B[(size_t)c];
            if (read.size() < b.size()) read.resize(b.size(), 0);
            for (size_t w = 0; w < b.size(); ++w) read[w] |= b[w];
        }
        for (int j = 0; j < K; ++j) o.kb_read += (kb[(size_t)j] && bit_get(read, j)) ? 1 : 0;
    }
    // 4. The extras, in record order, against the channel table (the (D) step, shared add_extra).
    // lean L4: the pullback C0† M C0 (diag_pullback, record_table.cpp) driven from supp(a) and the cz adjacency of
    // M's X-support instead of every wire and every pair — the same Z flips (a flipz per a-wire with X in M, per pair
    // endpoint whose partner has X in M), the same lin / quad counts (a both-X pair is met from both endpoints).
    const std::vector<int>& ex = idx.ex;
    ExtraCtx X{G, bare, W, {}, {}, {}};
    if (!ex.empty()) {
        std::vector<int>& alist = sc.alist;
        alist.clear();
        const int na = std::min(n, (int)nf.a.size());
        for (int q = 0; q < na; ++q) if (nf.a[(size_t)q] & 1) alist.push_back(q);
        std::vector<int>& adj_off = sc.adj_off;
        std::vector<int>& adj = sc.adj;
        adj_off.assign((size_t)n + 1, 0);
        for (const auto& e : nf.cz) { ++adj_off[(size_t)e.first + 1]; ++adj_off[(size_t)e.second + 1]; }
        for (int q = 0; q < n; ++q) adj_off[(size_t)q + 1] += adj_off[(size_t)q];
        adj.resize((size_t)adj_off[(size_t)n]);
        {
            std::vector<int>& fill = sc.coff;         // (scratch: the fill cursors; coff is spent)
            fill.assign(adj_off.begin(), adj_off.end() - 1);
            for (const auto& e : nf.cz) {
                adj[(size_t)fill[(size_t)e.first]++] = e.second;
                adj[(size_t)fill[(size_t)e.second]++] = e.first;
            }
        }
        auto pullback = [&](const Pauli& M) {
            Pauli h = M;
            int lin = 0;
            for (int q : alist) if (M.xbit(q)) { h.flipz(q); ++lin; }
            int quad2 = 0;                            // both-X pairs, counted from each endpoint
            const int NWm = std::min((int)M.x.size(), (n + 63) / 64);
            for (int w = 0; w < NWm; ++w)
                for (uint64_t x = M.x[(size_t)w]; x; x &= x - 1) {
                    const int q = w * 64 + __builtin_ctzll(x);
                    if (q >= n) continue;
                    for (int p = adj_off[(size_t)q]; p < adj_off[(size_t)q + 1]; ++p) {
                        const int t = adj[(size_t)p];
                        h.flipz(t);
                        if (M.xbit(t)) ++quad2;
                    }
                }
            const int quad = quad2 / 2;
            h.phase = (M.phase + ((-lin - 2 * quad) % 4 + 4) % 4) & 3;
            if (teeth) {
                const Pauli ref = diag_pullback(nf, M);
                if (ref.x != h.x || ref.z != h.z || ref.phase != h.phase)
                    throw std::logic_error("build_record_table_recB: the sparse pullback disagrees with diag_pullback "
                                           "(engine bug)");
            }
            return h;
        };
        for (int c : piv_col) {
            const Pauli Q = pullback(cols[(size_t)c].M);
            std::vector<uint64_t> ph;
            phi_of(G, Q, ph);
            Bits v(ph.begin(), ph.end());
            Bits comb = X.pech.reduce(v);
            const int slot = (int)X.Qpiv.size();
            if (!bits_any(v))
                throw std::logic_error("build_record_table_recB: a pivot record's plane is dependent (record-space "
                                       "pivots and planes disagree) (engine bug)");
            bit_flip(comb, slot);
            X.pech.insert(v, comb);
            X.Qpiv.push_back(Q);
        }
        for (int j = 0; j < K; ++j) {
            Bits v = fd_coords(W.dirs[(size_t)j]);
            Bits comb = X.dech.reduce(v);
            if (!bits_any(v))
                throw std::logic_error("build_record_table_recB: dependent kernel directions (engine bug)");
            bit_flip(comb, j);
            X.dech.insert(v, comb);
        }
        for (int c : ex) add_extra(X, pullback(cols[(size_t)c].M), c);
    }

    // 5. The canonicalization over every column in global record order — in the one-word form when the table fits
    // (<= 64 coins and directions: (D)'s r7.2 form, the same canonicalizer on words; lean L4: the extras' columns
    // are rows like any other, so they no longer force the Bits form).
    if (W.ncoin <= 64 && (int)W.dirs.size() <= 64) {
        W.small = true;
        W.A64.assign((size_t)m, 0);
        W.B64.assign((size_t)m, 0);
        for (int c = 0; c < m; ++c) {
            if (!W.A[(size_t)c].empty()) W.A64[(size_t)c] = W.A[(size_t)c][0];
            if (!W.B[(size_t)c].empty()) W.B64[(size_t)c] = W.B[(size_t)c][0];
        }
        W.A.clear();
        W.B.clear();
    }
    canonicalize_record_order(W, order, out);
    out.extras = (long)ex.size();
    out.ops_plane = X.ops_plane;
    out.ops_pivot_reduce = X.ops_pivot;
    out.ops_greduce = X.ops_greduce;
    out.ops_echelon = X.ops_ech;
    if (ops) *ops = o;
    release_rows_(W.A);
    release_rows_(W.B);
    return true;
}

}  // namespace qeccore
