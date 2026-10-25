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

inline void reduce_by(const std::vector<DRow>& E, Bits& v, Bits& cu, Bits& cp, long& ops) {
    for (const DRow& r : E)
        if (bit_get(v, r.lead)) {
            bits_xor(v, r.v);
            bits_xor(cu, r.cu);
            bits_xor(cp, r.cp);
            ++ops;
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
    std::vector<uint8_t> kb, piv_bd;
    std::vector<int> order, ch, ex, piv_col;
    std::vector<uint64_t> active;
    std::vector<DRow> red, E;
};
__attribute__((noinline)) RecBScratch& recb_scratch_() {   // out of line: one TLS access per call
    static thread_local RecBScratch sc;
    return sc;
}

}  // namespace

bool build_record_table_recB(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                             const DiagNormalForm& nf, const std::vector<RecColumn>& cols, RecTable& out,
                             RecBOps* ops) {
    if (!nf.diagonal_class) throw std::logic_error("build_record_table_recB: a non-diagonal plan (engine bug)");
    const int n = G.n_qubits;
    DiagNormalForm nf0 = nf;                          // the table is P-free (Lemma 2): identity prefix
    nf0.prefix = Pauli(n);
    twirl_law_detail::ShotLawCore core;
    const ShotLaw law = twirl_law_detail::build_shot_law_core(G, nf0, core);
    return build_record_table_recB_from_core(G, bare, nf0, law, core, cols, out, ops);
}

// J-b1 (fixplan-coldstart (A)): the B stage on a core law built elsewhere (the FAM plan payload's ONE law build,
// handed over through the plan cache's one-slot core park).
bool build_record_table_recB_from_core(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                       const DiagNormalForm& nf, const ShotLaw& law,
                                       const twirl_law_detail::ShotLawCore& core,
                                       const std::vector<RecColumn>& cols, RecTable& out, RecBOps* ops) {
    if (!nf.diagonal_class) throw std::logic_error("build_record_table_recB: a non-diagonal plan (engine bug)");
    const int n = G.n_qubits;
    DiagNormalForm nf0 = nf;                          // the table is P-free (Lemma 2): identity prefix
    nf0.prefix = Pauli(n);
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
    if (K > 0) {
        std::vector<DRow>& red = sc.red;
        red.clear();
        Bits none;
        for (int a = 0; a < A; ++a) {
            Bits v = core.v[(size_t)a], cu, cp;
            bit_flip(cp, a);                          // cp: the active-position combination
            reduce_by(red, v, cu, cp, o.pre_rows);
            const int ld = bits_lead(v);
            if (ld >= 0) red.push_back(DRow{std::move(v), std::move(cu), std::move(cp), ld});
        }
        for (int j = 0; j < K; ++j) {
            Bits v = core.U[(size_t)j], cu, cj;
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
    const int m = (int)cols.size();
    std::vector<int>& order = sc.order;
    order.resize((size_t)m);
    for (int c = 0; c < m; ++c) order[(size_t)c] = cols[(size_t)c].order;
    std::vector<int>& ch = sc.ch;
    ch.clear();
    for (int c = 0; c < m; ++c) if (cols[(size_t)c].channel) ch.push_back(c);
    auto by_order = [&](int x, int y) { return order[(size_t)x] < order[(size_t)y]; };
    if (!std::is_sorted(ch.begin(), ch.end(), by_order))   // storage order is usually the record order already
        std::sort(ch.begin(), ch.end(), by_order);
    std::vector<uint64_t>& active = sc.active;
    active.assign((size_t)G.words, 0);
    for (int i : core.aidx) active[(size_t)(i >> 6)] |= 1ULL << (i & 63);
    // active generator index -> active position (A entries; aidx is ascending)
    auto pos_of = [&](int i) {
        return (int)(std::lower_bound(core.aidx.begin(), core.aidx.end(), i) - core.aidx.begin());
    };
    AffineTable W;
    W.m = m;
    W.ncoin = 0;
    W.dirs = lam;
    W.A.assign((size_t)m, {});
    W.B.assign((size_t)m, {});
    W.beta.assign((size_t)m, 0);
    std::vector<DRow>& E = sc.E;                      // D ∪ U, then the pivots' dressings
    E.clear();
    for (const Bits& d : core.D) {
        Bits v = d, cu, cp;
        reduce_by(E, v, cu, cp, o.elim_rows);
        const int ld = bits_lead(v);
        if (ld < 0) throw std::logic_error("build_record_table_recB: dependent D basis (engine bug)");
        E.push_back(DRow{std::move(v), std::move(cu), std::move(cp), ld});
    }
    for (int j = 0; j < K; ++j) {
        Bits v = core.U[(size_t)j], cu, cp;
        bit_flip(cu, j);
        reduce_by(E, v, cu, cp, o.elim_rows);
        const int ld = bits_lead(v);
        if (ld < 0) throw std::logic_error("build_record_table_recB: kernel direction in span(D) (engine bug)");
        E.push_back(DRow{std::move(v), std::move(cu), std::move(cp), ld});
    }
    std::vector<int>& piv_col = sc.piv_col;           // pivot slot -> storage column
    std::vector<uint8_t>& piv_bd = sc.piv_bd;         // pivot slot -> par(det & mask) ⊕ ref
    piv_col.clear();
    piv_bd.clear();
#ifdef XTIM_MUTATE_B_ROW_SKIP
    bool mut_row_skipped = false;
#endif
    for (int c : ch) {
        const RecColumn& C = cols[(size_t)c];
        const uint8_t bd = (uint8_t)((det_parity(det, C.mask) ^ (C.ref & 1)) & 1);
        Bits vv;
        bool touch = false;
        const size_t nw = std::min(C.mask.size(), active.size());
#ifdef XTIM_MUTATE_B_PAD
        // mutation build only (MUT-B-PAD): every generator of the mask is visited, the inactive ones too
        for (size_t w = 0; w < C.mask.size(); ++w)
            for (uint64_t x = C.mask[w]; x; x &= x - 1) {
                const int i = (int)(w * 64) + __builtin_ctzll(x);
                ++o.gen_visits;
                if (w < nw && ((active[w] >> (i & 63)) & 1ULL)) { bits_xor(vv, core.v[(size_t)pos_of(i)]); touch = true; }
            }
#else
        for (size_t w = 0; w < nw; ++w)
            for (uint64_t x = C.mask[w] & active[w]; x; x &= x - 1) {
                const int i = (int)(w * 64) + __builtin_ctzll(x);
                bits_xor(vv, core.v[(size_t)pos_of(i)]);
                ++o.gen_visits;
                touch = true;
            }
#endif
        if (!touch) { W.beta[(size_t)c] = bd; continue; }   // a constant column: A = B = 0
        ++o.touched;
        Bits cu, cp;
        reduce_by(E, vv, cu, cp, o.elim_rows);
        const int ld = bits_lead(vv);
        if (ld >= 0) {                                // a PIVOT: its own fair coin
            const int slot = W.ncoin++;
            bit_flip(cp, slot);
            E.push_back(DRow{std::move(vv), Bits(), std::move(cp), ld});
            E.back().cu = std::move(cu);
            bit_flip(W.A[(size_t)c], slot);
            piv_col.push_back(c);
            piv_bd.push_back(bd);
            ++o.pivots;
            continue;
        }
#ifdef XTIM_MUTATE_B_ROW_SKIP
        // mutation build only (MUT-B-ROW): the first channel that reads a direction gets no B row
        if (bits_any(cu) && !mut_row_skipped) { cu.clear(); mut_row_skipped = true; }
#endif
        uint8_t beta = bd;
        for (int s = 0; s < (int)piv_bd.size(); ++s) if (bit_get(cp, s)) beta ^= piv_bd[(size_t)s];
#ifndef XTIM_MUTATE_B_KB_DROP
        for (int j = 0; j < K; ++j) if (bit_get(cu, j)) beta ^= kb[(size_t)j];
#endif   // mutation build only (MUT-B-KB): the kernel base signs dropped from β
        W.A[(size_t)c] = std::move(cp);
        W.B[(size_t)c] = std::move(cu);
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
    std::vector<int>& ex = sc.ex;
    ex.clear();
    for (int c = 0; c < m; ++c) if (!cols[(size_t)c].channel) ex.push_back(c);
    if (!std::is_sorted(ex.begin(), ex.end(), by_order)) std::sort(ex.begin(), ex.end(), by_order);
    ExtraCtx X{G, bare, W, {}, {}, {}};
    if (!ex.empty()) {
        for (int c : piv_col) {
            const Pauli Q = diag_pullback(nf0, cols[(size_t)c].M);
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
        for (int c : ex) add_extra(X, diag_pullback(nf0, cols[(size_t)c].M), c);
    }

    // 5. The canonicalization over every column in global record order — in the one-word form when the table fits
    // (no extras, <= 64 coins and directions: (D)'s r7.2 form, the same canonicalizer on words).
    if (ex.empty() && W.ncoin <= 64 && K <= 64) {
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
