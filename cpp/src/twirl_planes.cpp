#include "qeccore/twirl_planes.hpp"
#include "qeccore/bare_state_errors.hpp"   // BareStateCapacityError (the chunk table's budget refusal)
#include "qeccore/pauli_kernels.hpp"   // pmul_into

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <utility>
#if defined(__BMI2__)
#include <immintrin.h>                 // _pext_u64 (the chunked walk's pivot-bit gather)
#endif

namespace qeccore {

// symplectic-column bit: c in [0,n) -> x_c, [n,2n) -> z_{c-n}
static inline bool sbit(const Pauli& P, int c, int n) { return c < n ? P.xbit(c) : P.zbit(c - n); }
static inline bool has_idx(const std::vector<int32_t>& v, int q) { return std::binary_search(v.begin(), v.end(), q); }
static inline bool sbit(const SparsePauli& P, int c, int n) { return c < n ? has_idx(P.x, c) : has_idx(P.z, c - n); }

namespace {

// The oracle lever XTIM_PLANES_LAYOUT=sparse|dense (header: build), a TEST lever: it forces the planes' two storage
// switches (kDenseColsRatio, kDenseRowsRatio — the rows' fill-in switch included: sparse = never, dense = packed words
// from the start) to one side in build() — and, read separately in twirl_sampler.cpp
// (S2-T7), the per-wire rows' and the readout-flip columns' density switches; unset or any other value = the automatic
// switches. Pure
// performance (every layout gives identical planes and outputs: the byte oracle runs under both values). Read once per
// process (a function-local static): tests that toggle it need a subprocess.
CertifiedGroupPlanes::Layout env_layout_() {
    static const CertifiedGroupPlanes::Layout L = [] {
        const char* e = std::getenv("XTIM_PLANES_LAYOUT");
        if (e && !std::strcmp(e, "sparse")) return CertifiedGroupPlanes::Layout::Sparse;
        if (e && !std::strcmp(e, "dense")) return CertifiedGroupPlanes::Layout::Dense;
        return CertifiedGroupPlanes::Layout::Auto;
    }();
    return L;
}

// The elimination's working rows, dense words (T6's form). eliminate(i, feed, on_new) multiplies row i by the current
// pivot (pmul_into's product: the crossing term popcount(r.z & piv.x), then the XORs) and, when feed, reports every
// symplectic column the product newly sets.
struct DenseWork {
    int n, NWq;
    uint64_t lastmask;                      // the last word's qubits
    std::vector<Pauli> rows;
    const Pauli* piv = nullptr;
    std::vector<int> pw;                    // the pivot row's nonzero words
    bool dense_piv = false;
    // S2-T6 fix 2: the dense-row groups record nothing for the masks here (their masks come from a ProvenanceRREF
    // built on the first mask, after this build: CertifiedGroupPlanes::mask_of_); the sparse form records the forward
    // rows and steps (SparseWork below).
    static constexpr bool kForwardForm = false;
    DenseWork(const std::vector<SparsePauli>& g, int n_) : n(n_), NWq((n_ + 63) / 64) {
        lastmask = (n & 63) ? (1ULL << (n & 63)) - 1 : ~0ULL;
        rows.reserve(g.size());
        for (const SparsePauli& s : g) rows.push_back(to_dense(s));
    }
    // S2-FINALFIX F1: the fill-in switch — the sparse working rows packed in place of them, each freed once packed
    // (the same rows, phases and positions: the elimination continues on them).
    DenseWork(std::vector<SparsePauli>&& g, int n_) : n(n_), NWq((n_ + 63) / 64) {
        lastmask = (n & 63) ? (1ULL << (n & 63)) - 1 : ~0ULL;
        rows.reserve(g.size());
        for (SparsePauli& s : g) {
            rows.push_back(to_dense(s));
            std::vector<int32_t>().swap(s.x);
            std::vector<int32_t>().swap(s.z);
        }
        std::vector<SparsePauli>().swap(g);
    }
    bool bit(int i, int c) const { return sbit(rows[(size_t)i], c, n); }
    void set_pivot(int sel) {
        piv = &rows[(size_t)sel];           // never mutated while it is the pivot (sel is skipped)
        // The pivot's nonzero words: pmul_into restricted to them is pmul_into (a zero word of piv adds no
        // crossing term and XORs nothing).
        pw.clear();
        for (size_t w = 0; w < piv->x.size(); ++w) if (piv->x[w] | piv->z[w]) pw.push_back((int)w);
        dense_piv = 2 * pw.size() > piv->x.size();   // most words nonzero: the plain word loop
    }
    template <class F> void eliminate(int i, bool feed, F&& on_new) {
        Pauli& r = rows[(size_t)i];
        const Pauli& p = *piv;
        if (!feed && dense_piv) {            // nothing to list: pmul_into itself (the same product)
            pmul_into(r, p);
            return;
        }
        int sign = 0;
        for (int w : pw) {
            const uint64_t px = p.x[(size_t)w], pz = p.z[(size_t)w], rx = r.x[(size_t)w], rz = r.z[(size_t)w];
            if (feed) {
                const uint64_t m = w == NWq - 1 ? lastmask : ~0ULL;
                for (uint64_t b = px & ~rx & m; b; b &= b - 1) on_new(w * 64 + __builtin_ctzll(b));
                for (uint64_t b = pz & ~rz & m; b; b &= b - 1) on_new(n + w * 64 + __builtin_ctzll(b));
            }
            sign += __builtin_popcountll(rz & px);
            r.x[(size_t)w] = rx ^ px;
            r.z[(size_t)w] = rz ^ pz;
        }
        r.phase = (r.phase + p.phase + 2 * (sign & 1)) & 3;   // pmul_into's phase rule
    }
    SparsePauli row(int i) const { return from_dense(rows[(size_t)i]); }
    void combine(int) {}                                       // unused (no forward form)
    void snapshot(std::vector<int32_t>&) const {}
    void set_position(int) {}
    size_t bytes = 0;
    long steps = 0;
    void drop() {}
};

// The same rows sparse (sorted supports): the product is the sorted symmetric difference of the supports, the crossing
// term |r.z ∩ piv.x| (pmul_sparse's rule, = pmul_into's on the dense words); the newly set columns are the pivot's
// support entries absent from the row.
struct SparseWork {
    int n;
    std::vector<SparsePauli> rows;
    int sel = -1;
    std::vector<int32_t> tx, tz;            // product scratch (swapped into the row)
    static constexpr bool kForwardForm = true;   // S2-T6 fix 1/2: the forward rows E and their forward steps
    // S2-T6 fix 2: per row, the pivots (by pivot position) it was multiplied by in the forward steps, in order — one
    // entry per forward multiply (the elimination's own work), never a merged combination: E_r = g_i ⊕ Σ_{s ∈ fwd} E_s.
    // Fix 4: the positions ascend (a row meets each column's pivot once, and the positions grow with the columns), so a
    // row stores the deltas from its previous step (`last`) as LEB128 varints.
    std::vector<std::vector<uint8_t>> fwd;
    std::vector<int32_t> last;              // per row: its last step's position (-1: none yet)
    int pos = -1;                           // the current pivot's position
    long steps = 0;                         // Σ |steps|
    size_t bytes;                           // fwd + last on the heap, by capacity with the row headers (fix 4)
    bool recording = true;                  // false once the budget dropped the forward form
    long entries = 0;                       // S2-FINALFIX F1: Σ the rows' support sizes now (the fill-in switch reads it)
    SparseWork(const std::vector<SparsePauli>& g, int n_)
        : n(n_), rows(g), fwd(g.size()), last(g.size(), -1),
          bytes(g.size() * (sizeof(std::vector<uint8_t>) + sizeof(int32_t))) {
        for (const SparsePauli& s : g) entries += (long)(s.x.size() + s.z.size());
    }
    bool bit(int i, int c) const { return sbit(rows[(size_t)i], c, n); }
    void set_pivot(int s) { sel = s; }
    template <class F>
    static void symdiff_(const std::vector<int32_t>& u, const std::vector<int32_t>& v, std::vector<int32_t>& o,
                         bool feed, int base, F& on_new) {
        o.clear();
        o.reserve(u.size() + v.size());
        size_t i = 0, j = 0;
        while (i < u.size() && j < v.size()) {
            if (u[i] < v[j]) o.push_back(u[i++]);
            else if (v[j] < u[i]) { if (feed) on_new(base + v[j]); o.push_back(v[j++]); }
            else { ++i; ++j; }
        }
        for (; i < u.size(); ++i) o.push_back(u[i]);
        for (; j < v.size(); ++j) { if (feed) on_new(base + v[j]); o.push_back(v[j]); }
    }
    template <class F> void eliminate(int i, bool feed, F&& on_new) {
        SparsePauli& r = rows[(size_t)i];
        const SparsePauli& p = rows[(size_t)sel];
        int sign = 0;
        for (size_t a = 0, b = 0; a < r.z.size() && b < p.x.size();) {
            if (r.z[a] < p.x[b]) ++a;
            else if (p.x[b] < r.z[a]) ++b;
            else { sign ^= 1; ++a; ++b; }
        }
        // The dense form reports the new columns word by word, X before Z within a word; the order of the reports
        // only orders entries of different columns' lists, so the lists (and everything read from them) are the same.
        symdiff_(r.x, p.x, tx, feed, 0, on_new);
        symdiff_(r.z, p.z, tz, feed, n, on_new);
        entries += (long)(tx.size() + tz.size()) - (long)(r.x.size() + r.z.size());        std::swap(r.x, tx);
        std::swap(r.z, tz);
        r.phase = (r.phase + p.phase + 2 * sign) & 3;
    }
    SparsePauli row(int i) { return std::move(rows[(size_t)i]); }   // once per row, after the elimination: moved out
    void set_position(int p) { pos = p; }
    void combine(int i) {                   // row i was multiplied by the pivot at position pos
        if (!recording) return;
        std::vector<uint8_t>& v = fwd[(size_t)i];
        int32_t& l = last[(size_t)i];
        if (pos <= l) throw std::logic_error("CertifiedGroupPlanes: forward steps not ascending (engine bug)");
        uint32_t d = (uint32_t)(pos - l);
        l = pos;
        const size_t cap0 = v.capacity();
        for (; d >= 0x80; d >>= 7) v.push_back((uint8_t)(d | 0x80));
        v.push_back((uint8_t)d);
        bytes += v.capacity() - cap0;
        ++steps;
    }
    void drop() {                           // the budget is spent: the forward form is abandoned (prref instead)
        recording = false;
        std::vector<std::vector<uint8_t>>().swap(fwd);
        std::vector<int32_t>().swap(last);
        bytes = 0;
        steps = 0;
    }
    void snapshot(std::vector<int32_t>& out) const {   // the pivot's symplectic support, ascending
        const SparsePauli& r = rows[(size_t)sel];
        out.insert(out.end(), r.x.begin(), r.x.end());
        for (int32_t q : r.z) out.push_back(n + q);
    }
};

// The elimination's state, kept across a change of the working-row form (S2-FINALFIX F1: the fill-in switch).
struct ElimState {
    std::vector<int> posof, idat;           // the dense build's positions (posof: row -> position; idat: inverse)
    std::vector<uint8_t> is_piv;
    int c = 0, used = 0;                    // the next column, the pivots so far
    long lists = 0, scans = 0;              // the scan switch's counts (below)
    bool scan = false, feed = true;
    bool dropped = false;                   // the forward form abandoned (budget, or the switch to packed words)
    size_t cap = 0;                         // the forward form's storage bound
    explicit ElimState(CertifiedGroupPlanes& G) : posof((size_t)G.n_gens), idat((size_t)G.n_gens), is_piv((size_t)G.n_gens, 0) {
        for (int i = 0; i < G.n_gens; ++i) posof[(size_t)i] = idat[(size_t)i] = i;
        G.ech_off.assign(1, 0);
        // S2-T6 fix 4, the STORAGE bound (header: THE BOUND): the forward form's heap bytes — the step lists by
        // capacity with their row headers and last positions (work.bytes), E and its offsets by capacity — may use at
        // most kFwdBytesPct % of the prref route's heap bytes; the moment they would exceed it, the form is dropped on
        // the spot and the group takes the prref route (a ProvenanceRREF on the first mask, as eb55e1e did).
        cap = G.prref_route_bytes() / 100 * (size_t)kFwdBytesPct;
        G.col_to_row.assign((size_t)(2 * G.n_qubits), -1);
    }
};

// The forward form abandoned: its storage freed, the group takes the prref route.
template <class Work>
void drop_forward_(Work& work, CertifiedGroupPlanes& G, ElimState& st) {
    st.dropped = true;
    work.drop();
    std::vector<int32_t>().swap(G.ech_col);
    std::vector<int>().swap(G.ech_off);
}

// T6's elimination (build_sparse's header comment), on either working-row form, from column st.c on. stop(work) is
// asked before each column: true returns at once (st holds the elimination there, the column not begun) — the fill-in
// switch to packed words; returns whether it stopped.
template <class Work, class Stop>
bool rref_eliminate_(Work& work, CertifiedGroupPlanes& G, std::vector<std::vector<int>>& colrows, long entry_cost,
                     ElimState& st, Stop&& stop) {
    const bool adaptive = entry_cost > 0;
    const int n_qubits = G.n_qubits, two_n = 2 * n_qubits;
    std::vector<int>& posof = st.posof;
    std::vector<int>& idat = st.idat;
    std::vector<uint8_t>& is_piv = st.is_piv;
    auto form_bytes = [&]() {
        return work.bytes + (G.ech_col.capacity() + G.ech_off.capacity()) * sizeof(int32_t);
    };
    // transient: the bytes of a buffer the last push reallocated away from (held with its successor during the copy;
    // S2-CLEAN, fix-4 review n3: counted, so the peak is bounded, not only what stays)
    auto over_budget = [&](size_t transient = 0) {
        if (st.dropped || form_bytes() + transient <= st.cap) return;
        drop_forward_(work, G, st);
    };
    int& used = st.used;
    // The scan switch (pure performance, same eliminations): `lists` counts the row-list entries read and written,
    // `scans` the row visits the dense build's scans would have made so far (its pivot search from `used`, its
    // elimination over every row). Once the lists cost more (small groups whose eliminations fill up), the remaining
    // columns are done by those scans, and the lists are no longer kept. The bound is checked at each column and after
    // each elimination's new entries (`feed` off: the lists stop growing, the column finishes on its own complete list,
    // and the next column scans), so the lists never overrun it by more than one row's fill.
    long& lists = st.lists;
    long& scans = st.scans;
    bool& scan = st.scan;
    bool& feed = st.feed;                    // eliminations still list their new bits
    std::vector<int> no_rows;               // `rows` once the lists are dropped (never read then)
    for (int& c = st.c; c < two_n && used < G.n_gens; ++c) {
        if (stop(work)) return true;
        scans += 2L * G.n_gens - used;
        if (!scan && adaptive && entry_cost * (lists + (long)colrows[(size_t)c].size()) > scans) {
            scan = true;
            feed = false;
            std::vector<std::vector<int>>().swap(colrows);
        }
        std::vector<int>& rows = scan ? no_rows : colrows[(size_t)c];
        int sel = -1;
        if (scan) {
            for (int p = used; p < G.n_gens; ++p)
                if (work.bit(idat[(size_t)p], c)) { sel = idat[(size_t)p]; break; }
        } else {
            lists += (long)rows.size();
            for (int i : rows)
                if (!is_piv[(size_t)i] && work.bit(i, c) && (sel < 0 || posof[(size_t)i] < posof[(size_t)sel]))
                    sel = i;
        }
        if (sel < 0) { if (!scan) std::vector<int>().swap(rows); continue; }
        {   // the dense build's swap of positions `used` and posof[sel]
            const int p = posof[(size_t)sel], other = idat[(size_t)used];
            idat[(size_t)used] = sel; idat[(size_t)p] = other;
            posof[(size_t)sel] = used; posof[(size_t)other] = p;
        }
        is_piv[(size_t)sel] = 1;
        // S2-T6 fix 1: the pivot row as the FORWARD elimination left it (E_r: reduced by the earlier pivots only, its
        // first column c), before any back-substitution changes it.
        work.set_pivot(sel);
        work.set_position(used);
        if (Work::kForwardForm && !st.dropped) {
            const size_t col0 = G.ech_col.capacity(), off0 = G.ech_off.capacity();
            work.snapshot(G.ech_col);
            G.ech_off.push_back((int)G.ech_col.size());
            over_budget(((G.ech_col.capacity() != col0 ? col0 : 0) + (G.ech_off.capacity() != off0 ? off0 : 0)) *
                        sizeof(int32_t));
        }
        auto eliminate = [&](int i) {
            // List row i under every later column the product newly sets (earlier columns are never read again).
            work.eliminate(i, feed, [&](int cc) { if (cc > c) { colrows[(size_t)cc].push_back(i); ++lists; } });
            ++G.build_multiplies;
            // S2-T6 fix 1: combinations through the forward steps only (rows not yet pivots). A pivot row multiplied
            // here is back-substitution: R_r changes, E_r and its combination do not (tracking those made chains
            // cubic).
            if (Work::kForwardForm && !st.dropped && !is_piv[(size_t)i]) {
                work.combine(i);
                over_budget();
            }
            if (feed && adaptive && entry_cost * lists > scans) feed = false;
        };
        if (scan) {
            for (int i = 0; i < G.n_gens; ++i)
                if (i != sel && work.bit(i, c)) eliminate(i);
        } else {
            for (int i : rows)
                if (i != sel && work.bit(i, c)) eliminate(i);   // else a stale or duplicate entry
            std::vector<int>().swap(rows);
            if (!feed) {                     // the lists are incomplete from here on: the remaining columns scan
                scan = true;
                std::vector<std::vector<int>>().swap(colrows);
            }
        }
        G.col_to_row[c] = used;
        G.pivcol.push_back(c);
        ++used;
    }
    return false;
}

// The elimination's end: the rref rows, and the forward form when it was kept.
template <class Work>
void rref_finish_(Work& work, CertifiedGroupPlanes& G, ElimState& st) {
    const int used = st.used;
    G.rank = used;
    // Copied into fresh storage in pivot order, as the oracle's assign does (the readers walk them in that order).
    G.rref.reserve((size_t)used);
    for (int r = 0; r < used; ++r) G.rref.push_back(work.row(st.idat[(size_t)r]));
    bool forward = false;
    if constexpr (Work::kForwardForm) forward = !st.dropped;
    if (!forward) {
        std::vector<int>().swap(G.ech_off);           // the prref form: a ProvenanceRREF on the first mask
        std::vector<int32_t>().swap(G.ech_col);
        return;
    }
    if constexpr (Work::kForwardForm) {
        // S2-T6: the forward rows' steps stay where the build wrote them (per generator, read as
        // prov_steps[prov_row[r]]: no copy, so no transient beside them); a dependent row's are freed.
        G.prov_row.resize((size_t)used);
        for (int r = 0; r < used; ++r) G.prov_row[(size_t)r] = st.idat[(size_t)r];
        for (int i = 0; i < G.n_gens; ++i)
            if (!st.is_piv[(size_t)i]) std::vector<uint8_t>().swap(work.fwd[(size_t)i]);
        G.prov_steps = std::move(work.fwd);
        G.fwd_steps = work.steps;
        std::vector<int32_t>().swap(work.last);
        G.ech_col.shrink_to_fit();
        G.fwd_bytes = (G.ech_col.capacity() + G.ech_off.capacity()) * sizeof(int32_t);   // E and its offsets
        for (const std::vector<uint8_t>& v : G.prov_steps) G.fwd_bytes += v.capacity();
        G.fwd_bytes += G.prov_steps.capacity() * sizeof(std::vector<uint8_t>) + G.prov_row.capacity() * sizeof(int32_t);
    }
}

}  // namespace

CertifiedGroupPlanes CertifiedGroupPlanes::build(std::vector<SparsePauli> certified, int n_qubits) {
    const Layout L = env_layout_();
    return build_sparse(std::move(certified), n_qubits, kListEntryCost, L, L);
}

CertifiedGroupPlanes CertifiedGroupPlanes::build(const std::vector<Pauli>& certified, int n_qubits) {
    std::vector<SparsePauli> s;
    s.reserve(certified.size());
    for (const Pauli& P : certified) s.push_back(from_dense(P));
    return build(std::move(s), n_qubits);
}

// T6: build_sparse, the same build on set bits. The column planes are scattered from each generator's set bits. The
// RREF runs the dense build's exact elimination (columns ascending; the pivot = the row at the LOWEST position >= used
// with the column set, positions moving by the dense build's swap; every other row with the column set multiplied by
// the pivot, pmul_into's products in the same order per row) but finds the rows of a column from a per-column row
// list instead of a scan of all rows: a row is listed under every column it has set initially, and again whenever an
// elimination sets a new bit at a later column (lazily deleted: a listed row is re-checked before use). Rows keep their
// identity (work is indexed by the input index; posof/idat carry the dense build's positions), so rref, pivcol,
// col_to_row and every phase are the dense build's. Cost: O(total weight + fill-in + eliminations x row support)
// instead of O(n_gens x n). S2-T5: the planes are the CSR column index (plus the dense planes when small against it:
// kDenseColsRatio), the working rows sparse or dense words (kDenseRowsRatio; header: build).
CertifiedGroupPlanes CertifiedGroupPlanes::build_sparse(std::vector<SparsePauli> certified_in, int n_qubits,
                                                        long entry_cost, Layout cols, Layout rows, long switch_at) {
    CertifiedGroupPlanes G;
    G.gens     = std::move(certified_in);            // the one copy (QW-A); read below through `certified`
    const std::vector<SparsePauli>& certified = G.gens;
    G.n_qubits = n_qubits;
    G.n_gens   = (int)certified.size();
    G.words    = (G.n_gens + 63) / 64;
    const int NWq = (n_qubits + 63) / 64;
    const int two_n = 2 * n_qubits;

    // ── The column index (CSR per block) from the set bits; the generators' entries ascending per wire. ──
    long nnz = 0;
    for (int blk = 0; blk < 2; ++blk) {
        ColIndex& X = blk ? G.zidx : G.xidx;
        X.off.assign((size_t)n_qubits + 1, 0);
        for (const SparsePauli& g : certified)
            for (int32_t q : blk ? g.z : g.x) ++X.off[(size_t)q + 1];
        for (int q = 0; q < n_qubits; ++q) X.off[(size_t)q + 1] += X.off[(size_t)q];
        X.gen.resize((size_t)X.off[(size_t)n_qubits]);
        nnz += (long)X.gen.size();
        std::vector<int> cur(X.off.begin(), X.off.end() - 1);
        for (int i = 0; i < G.n_gens; ++i)
            for (int32_t q : blk ? certified[(size_t)i].z : certified[(size_t)i].x) X.gen[(size_t)cur[(size_t)q]++] = i;
    }
    G.index_check_ = std::make_shared<std::once_flag>();   // S2-T6 (review m1): check_index, once per group
    // ── The dense column planes, when small against the index (header: kDenseColsRatio), or forced. ──
    G.dense_cols = cols == Layout::Dense || (cols == Layout::Auto && 4L * n_qubits * G.words <= kDenseColsRatio * nnz);
    if (G.dense_cols) {
        G.xcol.assign((size_t)n_qubits * G.words, 0);
        G.zcol.assign((size_t)n_qubits * G.words, 0);
        for (int i = 0; i < G.n_gens; ++i) {
            const uint64_t bit = 1ULL << (i & 63);
            const int wo = i >> 6;
            for (int32_t q : certified[(size_t)i].x) G.xcol[(size_t)q * G.words + wo] |= bit;
            for (int32_t q : certified[(size_t)i].z) G.zcol[(size_t)q * G.words + wo] |= bit;
        }
    }
    // ── The per-column row lists start as the column index (a row under every column it has set). ──
    std::vector<std::vector<int>> colrows((size_t)two_n);
    for (int c = 0; c < two_n; ++c) {
        const ColIndex& X = c < n_qubits ? G.xidx : G.zidx;
        const int q = c < n_qubits ? c : c - n_qubits;
        colrows[(size_t)c].assign(X.gen.begin() + X.off[(size_t)q], X.gen.begin() + X.off[(size_t)q + 1]);
    }

    // ── GF(2) RREF of the (x|z) symplectic rows, exact phase carried per row (the oracle's elimination). ──
    // The working rows: packed words from the start when small against the group (kDenseRowsRatio on the starting
    // weight), else sparse rows that switch to packed words once the elimination's fill-in makes them so (S2-FINALFIX
    // F1: the same ratio on the rows' current weight, checked before each column, and never back; the forward form is
    // dropped there, as a group built on packed words has none). switch_at >= 0 (tests): sparse rows, packed from
    // pivot position switch_at on.
    const long dense_units = 4L * G.n_gens * NWq;
    const bool dense_rows = rows == Layout::Dense ||
                            (rows == Layout::Auto && switch_at < 0 && dense_units <= kDenseRowsRatio * nnz);
    ElimState st(G);
    auto never = [](const auto&) { return false; };
    if (dense_rows) {
        DenseWork work(certified, n_qubits);
        G.rows_words_from = 0;
        rref_eliminate_(work, G, colrows, entry_cost, st, never);
        rref_finish_(work, G, st);
    } else {
        SparseWork work(certified, n_qubits);
        auto fill = [&](const SparseWork& w) {
            if (rows != Layout::Auto) return false;
            if (switch_at >= 0) return (long)st.used >= switch_at;
            return dense_units <= kDenseRowsRatio * w.entries;
        };
        if (!rref_eliminate_(work, G, colrows, entry_cost, st, fill)) {
            rref_finish_(work, G, st);
        } else {
            drop_forward_(work, G, st);
            G.rows_words_from = st.used;
            DenseWork words(std::move(work.rows), n_qubits);
            rref_eliminate_(words, G, colrows, entry_cost, st, never);
            rref_finish_(words, G, st);
        }
    }
    return G;
}

// S2-T6 (review m1): the index held to the generator rows — re-walking the supports in generator order, every entry
// (i, q) must be the next entry of column q, and every column must end there: the index is exactly the transpose of
// the supports (the build_shot_law teeth read the columns through it). O(nnz), an engine-bug throw. check_index_once:
// the first teeth call of the group (Clifford compiles never run teeth, so they never pay it), thread-safe.
void CertifiedGroupPlanes::check_index() const {
    for (int blk = 0; blk < 2; ++blk) {
        const ColIndex& X = blk ? zidx : xidx;
        bool ok = X.off.size() == (size_t)n_qubits + 1;
        std::vector<int> cur;
        if (ok) cur.assign(X.off.begin(), X.off.end() - 1);
        for (int i = 0; i < n_gens && ok; ++i)
            for (int32_t q : blk ? gens[(size_t)i].z : gens[(size_t)i].x) {
                const int p = cur[(size_t)q]++;
                if (p >= X.off[(size_t)q + 1] || X.gen[(size_t)p] != i) { ok = false; break; }
            }
        for (int q = 0; q < n_qubits && ok; ++q) ok = cur[(size_t)q] == X.off[(size_t)q + 1];
        if (!ok)
            throw std::logic_error("CertifiedGroupPlanes: the column index is not the transpose of the generators "
                                   "(engine bug)");
    }
}

void CertifiedGroupPlanes::check_index_once() const {
    if (index_check_) std::call_once(*index_check_, [this] { check_index(); });
    else check_index();
}

bool CertifiedGroupPlanes::col_bit(int q, bool xblock, int i) const {
    if (dense_cols) return ((xblock ? xcol : zcol)[(size_t)q * words + (size_t)(i >> 6)] >> (i & 63)) & 1ULL;
    return std::binary_search(col_begin(q, xblock), col_end(q, xblock), i);
}

// S2-T6 fix 1: the mask of W ∈ span — W reduced against the forward rows E in ascending pivot order (E_r is zero on
// every column before pivcol[r] and holds it, so applying it clears row r's pivot bit and can only set later ones),
// the XOR of the applied rows' combinations. The applied set is the unique E-decomposition of W, so with independent
// generators the mask is the unique Π_{mask} g = ±W. Fix 4: the pending pivots are a bitmap over the rows, scanned
// ascending word by word (a word is re-read after each row it applies: that row's flips set only later bits), in
// place of a min-heap — the same rows in the same order, without the heap's log factor and branches; the scan costs
// one unit per word of the rows' range. Cost: the applied rows' weights plus the range's words.
void CertifiedGroupPlanes::forward_mask_(const SparsePauli& W, std::vector<int32_t>& mask, ProvScratch& sc) const {
    if (sc.bits.size() < (size_t)(2 * n_qubits)) sc.bits.assign((size_t)(2 * n_qubits), 0);   // all zero between calls
    const size_t RW = (size_t)(rank + 63) / 64;
    if (sc.pend.size() < RW) sc.pend.assign(RW, 0);                                         // all zero between calls
    if (sc.mk.size() < RW) sc.mk.assign(RW, 0);
    const long w0 = sc.work;
    sc.touched.clear();
    sc.rows.clear();
    int lo = rank, hi = -1;                  // the pending rows' range
    auto flip = [&](int32_t col) {
        sc.bits[(size_t)col] ^= 1;
        sc.touched.push_back(col);
        const int r = col_to_row[(size_t)col];
        if (r >= 0) {
            sc.pend[(size_t)r >> 6] ^= 1ULL << (r & 63);
            lo = std::min(lo, r);
            hi = std::max(hi, r);
        }
    };
    for (int32_t q : W.x) flip(q);
    for (int32_t q : W.z) flip(n_qubits + q);
    for (int w = lo >> 6; hi >= 0 && w <= (hi >> 6); ++w) {   // hi grows as rows apply
        ++sc.work;
        for (uint64_t word; (word = sc.pend[(size_t)w]) != 0;) {
            const int r = w * 64 + __builtin_ctzll(word);
            sc.rows.push_back(r);
            for (int p = ech_off[(size_t)r]; p < ech_off[(size_t)r + 1]; ++p) flip(ech_col[(size_t)p]);   // clears r
            if ((sc.pend[(size_t)w] >> (r & 63)) & 1ULL)
                throw std::logic_error("CertifiedGroupPlanes: a forward row misses its pivot (engine bug)");
        }
    }
    bool zero = true;
    for (int32_t col : sc.touched) { zero = zero && !sc.bits[(size_t)col]; sc.bits[(size_t)col] = 0; }
    if (!zero) throw std::logic_error("CertifiedGroupPlanes: forward residual nonzero on IN_GROUP (engine bug)");
    sc.work += kFwdFlipUnits * (long)sc.touched.size() + (long)sc.rows.size();   // the reduction's flips and rows
    // fix 4: the prref route's decompose of W applies a row per pivot of W's decomposition (as many as E's, taken as
    // its estimate) at 2·ceil(n/64) + words words each, plus its clears
    mask_credit_ += ((long)sc.rows.size() + 1) * (2L * ((n_qubits + 63) / 64) + words);
    // S2-T6 fix 2: W = Σ_{r ∈ applied} E_r and E_r = g_{prov_row[r]} ⊕ Σ_{s ∈ steps(r)} E_s (s < r): expanded by
    // descending positions with a parity mark per row (a row reached an even number of times cancels), so each step
    // list is read at most once per query and only for rows of odd parity. Fix 4: the marks are a bitmap over the rows,
    // scanned descending word by word (re-read after each row: its steps toggle only earlier bits), in place of a
    // max-heap; one unit per word of the range.
    const size_t GWn = (size_t)words;
    if (sc.gb.size() < GWn) sc.gb.assign(GWn, 0);                                         // all zero between calls
    int lo2 = rank, hi2 = -1;
    int32_t glo = n_gens, ghi = -1;
    for (int r : sc.rows) {                       // distinct (each applied once)
        sc.mk[(size_t)r >> 6] ^= 1ULL << (r & 63);
        lo2 = std::min(lo2, r);
        hi2 = std::max(hi2, r);
    }
    mask.clear();
    for (int w = hi2 >> 6; hi2 >= 0 && w >= (lo2 >> 6); --w) {   // lo2 falls as steps reach down
        ++sc.work;
        for (uint64_t word; (word = sc.mk[(size_t)w]) != 0;) {
            const int b = 63 - __builtin_clzll(word);
            const int r = w * 64 + b;
            sc.mk[(size_t)w] = word & ~(1ULL << b);
            const int32_t g = prov_row[(size_t)r];
            sc.gb[(size_t)g >> 6] |= 1ULL << (g & 63);  // the mask's generators, read ascending below
            glo = std::min(glo, g);
            ghi = std::max(ghi, g);
            ++sc.work;
            const std::vector<uint8_t>& st = prov_steps[(size_t)g];   // varint deltas, ascending positions < r
            int32_t t = -1, t0 = -1;
            for (size_t p = 0; p < st.size();) {
                uint32_t d = 0;
                for (int sh = 0;; sh += 7) {
                    const uint8_t c = st[p++];
                    d |= (uint32_t)(c & 0x7f) << sh;
                    if (!(c & 0x80)) break;
                }
                t += (int32_t)d;
                if (t0 < 0) t0 = t;
                ++sc.work;
                sc.mk[(size_t)t >> 6] ^= 1ULL << (t & 63);
            }
            if (t >= r) throw std::logic_error("CertifiedGroupPlanes: a forward step is not earlier (engine bug)");
            if (t0 >= 0) lo2 = std::min(lo2, t0);
        }
    }
    for (int w = glo >> 6; ghi >= 0 && w <= (ghi >> 6); ++w) {   // ascending, left zeroed (a unit per word and entry)
        ++sc.work;
        for (uint64_t word = sc.gb[(size_t)w]; word; word &= word - 1) {
            mask.push_back((int32_t)(w * 64 + __builtin_ctzll(word)));
            ++sc.work;
        }
        sc.gb[(size_t)w] = 0;
    }
    mask_work_ += sc.work - w0;
}

// The mask of W ∈ span (sc.rows = the walk's rows on entry): dense provenance — the XOR of the walk's rows' reduced
// combinations; forward provenance — W's reduction against E (forward_mask_).
void CertifiedGroupPlanes::release_provenance() {
    std::vector<std::vector<uint8_t>>().swap(prov_steps);
    std::vector<int32_t>().swap(prov_row);
    dense_prov_.reset();
    std::vector<int>().swap(ech_off);
    std::vector<int32_t>().swap(ech_col);
    prov_released_ = true;
}

void CertifiedGroupPlanes::mask_of_(const SparsePauli& W, std::vector<int32_t>& mask, ProvScratch& sc) const {
    if (prov_released_) throw std::logic_error("CertifiedGroupPlanes: a mask after release_provenance (engine bug)");
    // S2-T6 fix 4, the TIME bound (header: THE BOUND): before each query, the forward queries' units so far, priced at
    // kFwdUnitWords word ops, are held to kMaskSpendPct % of the prref build's estimated word ops plus (100 +
    // kMaskSpendPct) % of what the prref route's decomposes would have cost on the same queries (mask_credit_); once
    // they pass it, this query and the remaining ones take the prref route (the crossing query was the last forward
    // one). The test lever fwd_query_limit_ replaces the budget by a query count.
    ++mask_queries_;
    if (!ech_off.empty() && !fwd_spent_) {
        if (fwd_query_limit_ >= 0) fwd_spent_ = mask_queries_ > fwd_query_limit_;
        else fwd_spent_ = 100 * kFwdUnitWords * mask_work_ >
                          kMaskSpendPct * prref_build_words() + (100 + kMaskSpendPct) * mask_credit_;
        if (fwd_spent_) {
            // S2-CLEAN (fix-4 review n3): no later query reads the forward form — freed before the prref route is
            // built below, so it is never held beside it
            std::vector<std::vector<uint8_t>>().swap(prov_steps);
            std::vector<int32_t>().swap(prov_row);
            std::vector<int32_t>().swap(ech_col);
            std::vector<int>().swap(ech_off);
        }
    }
    if (ech_off.empty() || fwd_spent_) {
        // The dense-row form: the pre-S2-T6 provenance RREF (dense rows and combinations in insertion order, built
        // once, on the first mask — after the planes' working rows are gone, as the sampler built it before S2-T6).
        if (!dense_prov_) dense_prov_ = std::make_shared<const ProvenanceRREF>(ProvenanceRREF::build(gens, n_qubits));
        ProvenanceRREF::Scratch ps;
        ps.sv.swap(sc.psv);
        ps.heap.swap(sc.pheap);
        const bool ok = dense_prov_->decompose(to_dense(W), sc.pmask, ps);
        ps.sv.swap(sc.psv);
        ps.heap.swap(sc.pheap);
        if (!ok) throw std::logic_error("CertifiedGroupPlanes: provenance residual nonzero on IN_GROUP (engine bug)");
        mask.clear();
        for (size_t w = 0; w < sc.pmask.size(); ++w)
            for (uint64_t b = sc.pmask[w]; b; b &= b - 1) mask.push_back((int32_t)(w * 64 + __builtin_ctzll(b)));
        return;
    }
    forward_mask_(W, mask, sc);
}

Membership::Verdict CertifiedGroupPlanes::reduce_sparse(const SparsePauli& W, int& phase_out, std::vector<int32_t>* mask,
                                                        ProvScratch& sc) const {
    // The walk's rows: {r : W has pivcol[r] set} (header of walk_), applied in ascending row (= pivot column) order.
    sc.rows.clear();
    for (int32_t q : W.x) { const int r = col_to_row[(size_t)q]; if (r >= 0) sc.rows.push_back(r); }
    for (int32_t q : W.z) { const int r = col_to_row[(size_t)(n_qubits + q)]; if (r >= 0) sc.rows.push_back(r); }
    std::sort(sc.rows.begin(), sc.rows.end());
    if ((long)(W.x.size() + W.z.size()) > 2L * ((n_qubits + 63) / 64)) {
        // A probe heavier than the dense words (a long observable): the dense walk (chunked) is the cheaper of the two
        // — a pure-performance switch, the same rows and the same exact products either way.
        Pauli d = to_dense(W);
        const Membership::Verdict v = reduce_cheap(d, phase_out);
        if (v == Membership::IN_GROUP && mask) mask_of_(W, *mask, sc);
        return v;
    }
    sc.w = W;
    // S2-FINALFIX F1: the residual fills in as rows apply (a filled-in RREF: random parity checks), and a sparse
    // product costs the residual's whole weight. Once it is heavier than the dense words, the rest of the walk runs on
    // a dense accumulator (each sparse row: its crossing term |w.z ∩ row.x| read off the words, then its bits
    // toggled; pmul_sparse's product and phase rule), and the residual returns to the sparse form at the end. Pure
    // performance: the same rows, in the same order, the same products.
    const long dense_at = 2L * ((n_qubits + 63) / 64);
    size_t k = 0;
    for (; k < sc.rows.size() && (long)(sc.w.x.size() + sc.w.z.size()) <= dense_at; ++k) {
        pmul_sparse(sc.w, rref[(size_t)sc.rows[k]], sc.t);
        std::swap(sc.w, sc.t);
    }
    if (k < sc.rows.size()) {
        Pauli& d = sc.wd;
        reset_pauli_inplace(d, n_qubits);
        for (int32_t q : sc.w.x) d.x[(size_t)q >> 6] |= 1ULL << (q & 63);
        for (int32_t q : sc.w.z) d.z[(size_t)q >> 6] |= 1ULL << (q & 63);
        int phase = sc.w.phase;
        for (; k < sc.rows.size(); ++k) {
            const SparsePauli& R = rref[(size_t)sc.rows[k]];
            int sign = 0;
            for (int32_t q : R.x) sign ^= (int)((d.z[(size_t)q >> 6] >> (q & 63)) & 1ULL);
            for (int32_t q : R.x) d.x[(size_t)q >> 6] ^= 1ULL << (q & 63);
            for (int32_t q : R.z) d.z[(size_t)q >> 6] ^= 1ULL << (q & 63);
            phase = ((phase + R.phase + 2 * sign) % 4 + 4) % 4;
        }
        d.phase = phase;
        sc.w = from_dense(d);
    }
    if (sc.w.x.empty() && sc.w.z.empty()) {
        phase_out = sc.w.phase;
        if (mask) mask_of_(W, *mask, sc);
        return Membership::IN_GROUP;
    }
#ifdef XTIM_MUTATE_F3_ANTI_LOGICAL
    return Membership::LOGICAL;                               // mutation build only (MUT-F3-ANTI): never ANTI
#else
    sc.phi.assign((size_t)words, 0);                          // φ(w), as anti_ (the same bits)
    for (int32_t q : sc.w.x) xor_col(sc.phi.data(), q, false);
    for (int32_t q : sc.w.z) xor_col(sc.phi.data(), q, true);
    for (uint64_t v : sc.phi) if (v) return Membership::ANTI;
    return Membership::LOGICAL;
#endif
}

size_t CertifiedGroupPlanes::storage_bytes() const {
    auto vb = [](const auto& v) -> size_t { return v.capacity() * sizeof(v[0]); };
    auto spb = [&](const std::vector<SparsePauli>& v) { size_t b = vb(v); for (const SparsePauli& s : v) b += vb(s.x) + vb(s.z); return b; };
    size_t steps = vb(prov_steps);
    for (const std::vector<uint8_t>& v : prov_steps) steps += vb(v);
    return vb(xcol) + vb(zcol) + vb(xidx.off) + vb(xidx.gen) + vb(zidx.off) + vb(zidx.gen) + spb(rref) + vb(pivcol) +
           vb(col_to_row) + steps + vb(prov_row) + vb(ech_off) + vb(ech_col) + prref_bytes_();
}

// The prref route's heap bytes on this group (header): n_gens rows (independent generators: every one is kept), each
// a DenseRow header and 2·ceil(n/64) + words words, plus row_of_col (2·64·ceil(n/64) ints) and pivmask.
size_t CertifiedGroupPlanes::prref_route_bytes() const {
    const size_t NW = (size_t)((n_qubits + 63) / 64);
    return (size_t)n_gens * (sizeof(ProvenanceRREF::DenseRow) + 8 * (2 * NW + (size_t)words)) + 128 * NW * sizeof(int) +
           16 * NW;
}

// Its build's estimated word ops (header): it reduces every generator through about as many row XORs as the forward
// elimination took steps (both are the forward elimination of the same rows), 2·ceil(n/64) + words words each, and
// writes n_gens rows.
long CertifiedGroupPlanes::prref_build_words() const {
    return ((long)n_gens + fwd_steps) * (2L * ((n_qubits + 63) / 64) + words);
}

size_t CertifiedGroupPlanes::prref_bytes_() const {   // the prref route's provenance, while held (until released)
    if (!dense_prov_) return 0;
    size_t b = dense_prov_->rows.capacity() * sizeof(ProvenanceRREF::DenseRow);
    for (const ProvenanceRREF::DenseRow& r : dense_prov_->rows) b += (r.s.capacity() + r.c.capacity()) * 8;
    return b + dense_prov_->row_of_col.capacity() * sizeof(int) + dense_prov_->pivmask.capacity() * 8;
}

Membership CertifiedGroupPlanes::reduce(const Pauli& p) const {
    // w starts as p and is reduced against the RREF (w = p · ∏ used pivot rows), then MOVED into the
    // returned Membership's rep (TLC: no scratch — the one copy of p is the result's storage).
    // termroute-a F3: the support-directed walk of reduce_cheap (the same rows in the same ascending
    // pivot-column order: byte-identical residual and phase) and the ANTI verdict by the column planes.
    Pauli w = p;
    walk_(w);

    // Residual support?
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }

    if (zero) {
        // p's support is in the group. Residual w = p * G = i^{w.phase} I relates p to the group's
        // canonical element G of that support. The +-1 sign is the phase-2 component, EXACTLY the
        // reference convention `(residual.phase >> 1) & 1` in twirl_diag_reference.py
        // (::_reduce_to_logical / base-point read-off). Real group => w.phase is 0 or 2 for a real
        // probe; the >>1 also extracts the +-component for an off-Hermitian probe (== `(red.p//2)&1`).
        int sign = ((w.phase >> 1) & 1) ? -1 : +1;
        return Membership::in_group(sign, std::move(w));
    }

    // Non-zero residual: in the normaliser (LOGICAL) unless it anticommutes with a generator.
    if (anti_(w)) return Membership::anti(std::move(w));
    return Membership::logical(std::move(w));
}

// The pivot walk (shared by reduce and reduce_cheap). The rows applied are exactly {r : w has pivcol[r] set}: in
// the FULL reduced echelon each pivot column occurs only in its own row, so a pmul never sets or clears another
// pivot bit, and the support-directed walk this replaces (2026-07-16; termroute-a F3) applied them in ascending
// pivot-column (= row) order. perf-engine §2.1(b): the same rows, the same order, multiplied in 8-row CHUNKS from
// the lazily built product table — w · R_a · R_b · … = w · (R_a · R_b · …) by associativity, every partial
// product in exact normal form (pmul_into's phase rule, inlined on the raw words) ⇒ identical (x, z, phase).
// Cost: one parallel bit extract per word of w (the pivot bits in ascending column order) + ≤ ceil(rank/8) row
// multiplies, instead of one row multiply per selected pivot.
namespace {
inline uint64_t pext_bits_(uint64_t v, uint64_t m) {
#if defined(__BMI2__)
    return _pext_u64(v, m);
#else
    uint64_t out = 0;
    int k = 0;
    while (m) { const int b = __builtin_ctzll(m); m &= m - 1; out |= ((v >> b) & 1ULL) << k; ++k; }
    return out;
#endif
}
}  // namespace

const CertifiedGroupPlanes::ChunkTable& CertifiedGroupPlanes::chunk_table() const {
    if (chunk_ready_) return chunk_;
    ChunkTable& C = chunk_;
    chunk_ready_ = true;
    C.usable = false;
    const int NW = (n_qubits + 63) / 64;
    C.NW = NW;
    C.nchunks = (rank + 7) / 8;
    const int64_t bytes = (int64_t)C.nchunks * 256 * ((int64_t)2 * NW * 8 + 1) + (int64_t)2 * NW * 8;
    if (chunk_off_ || bytes > kChunkTableCapBytes) return chunk_;
    try {
        C.ticket = std::make_shared<StateBudgetTicket>(bytes, "certified-group chunk table");
    } catch (const BareStateCapacityError&) {
        return chunk_;                                        // over budget: the row walk, nothing allocated
    }
    C.pivmask.assign((size_t)2 * NW, 0);
    for (int r = 0; r < rank; ++r) {
        const int c = pivcol[r];
        const int blk = c >= n_qubits ? 1 : 0;
        const int q = c - blk * n_qubits;
        C.pivmask[(size_t)blk * NW + (q >> 6)] |= 1ULL << (q & 63);
    }
    const size_t E = (size_t)2 * NW;                          // words per entry
    C.xz.assign((size_t)C.nchunks * 256 * E, 0);
    C.ph.assign((size_t)C.nchunks * 256, 0);
    for (int j = 0; j < C.nchunks; ++j) {
        const int cnt = std::min(8, rank - 8 * j);
        for (unsigned byte = 1; byte < (1u << cnt); ++byte) {
            const int t = 31 - __builtin_clz(byte);           // the highest selected row: appended LAST
            const unsigned prev = byte ^ (1u << t);
            const uint64_t* pe = C.xz.data() + ((size_t)j * 256 + prev) * E;
            uint64_t* e = C.xz.data() + ((size_t)j * 256 + byte) * E;
            const SparsePauli& R = rref[(size_t)(8 * j + t)];
            int sign = 0;                                     // popcount(prev.z & R.x), on R's X support
            for (int32_t q : R.x) sign += (int)((pe[NW + (q >> 6)] >> (q & 63)) & 1ULL);
            C.ph[(size_t)j * 256 + byte] = (uint8_t)((C.ph[(size_t)j * 256 + prev] + R.phase + 2 * (sign & 1)) & 3);
            std::memcpy(e, pe, E * sizeof(uint64_t));
            for (int32_t q : R.x) e[q >> 6] ^= 1ULL << (q & 63);
            for (int32_t q : R.z) e[NW + (q >> 6)] ^= 1ULL << (q & 63);
        }
    }
    C.usable = true;
    return chunk_;
}

// The row-by-row walk (the pre-§2.1(b) form; the fallback when no chunk table is built): scan w's set symplectic
// bits ascending and multiply by the pivot row of each pivot column met. Same rows, same order as the chunked walk.
void CertifiedGroupPlanes::walk_rows_(Pauli& w) const {
    const int NWq = (n_qubits + 63) / 64;
    for (int blk = 0; blk < 2; ++blk) {
        std::vector<uint64_t>& vec = blk ? w.z : w.x;
        for (int wq = 0; wq < NWq; ++wq) {
            uint64_t word = vec[(size_t)wq];
            while (word) {
                const int b = __builtin_ctzll(word);
                const int q = wq * 64 + b;
                const int c = blk ? n_qubits + q : q;
                const int r = col_to_row[(size_t)c];
                if (r >= 0) {
                    pmul_into(w, rref[(size_t)r]);                    // clears bit c (pivot-exclusive; sparse row)
                    word = (b == 63) ? 0 : (vec[(size_t)wq] & (~0ULL << (b + 1)));
                } else {
                    word &= word - 1;                                 // free column: skip
                }
            }
        }
    }
}

void CertifiedGroupPlanes::walk_(Pauli& w) const {
    const ChunkTable& C = chunk_table();
    if (!C.usable) { walk_rows_(w); return; }
    const int NW = C.NW;
    uint64_t* xw = w.x.data();
    uint64_t* zw = w.z.data();
    // Gather the pivot bits (ascending column = ascending row) into a bit stream, 8 per chunk.
    uint64_t acc = 0;
    int nacc = 0, j = 0;
    auto chunk = [&](unsigned byte) {
        if (byte) {
            const uint64_t* cx = C.xz.data() + ((size_t)j * 256 + byte) * ((size_t)2 * NW);
            const uint64_t* cz = cx + NW;
            int sign = 0;
            for (int i = 0; i < NW; ++i) sign += __builtin_popcountll(zw[i] & cx[i]);
            w.phase = (w.phase + C.ph[(size_t)j * 256 + byte] + 2 * (sign & 1)) & 3;
            for (int i = 0; i < NW; ++i) { xw[i] ^= cx[i]; zw[i] ^= cz[i]; }
        }
        ++j;
    };
    for (int t = 0; t < 2 * NW; ++t) {
        const uint64_t m = C.pivmask[(size_t)t];
        if (!m) continue;
        // The pivot bits of a word are read before any later chunk product touches w; a chunk product changes
        // only its own rows' pivot bits (already consumed) and non-pivot bits (never read), so the stream is w's
        // original pivot bits in ascending column order.
        uint64_t ex = pext_bits_(t < NW ? xw[t] : zw[t - NW], m);
        int rem = __builtin_popcountll(m);
        if (nacc) {                                           // complete the pending byte first
            const int take = std::min(8 - nacc, rem);
            acc |= (ex & ((1ULL << take) - 1)) << nacc;
            nacc += take; ex >>= take; rem -= take;
            if (nacc == 8) { chunk((unsigned)acc); acc = 0; nacc = 0; }
        }
        while (rem >= 8) { chunk((unsigned)(ex & 255)); ex >>= 8; rem -= 8; }
        if (rem) { acc = ex & ((1ULL << rem) - 1); nacc = rem; }
    }
    if (nacc > 0) chunk((unsigned)acc);
}

// termroute-a F3: w anticommutes with the group ⟺ φ(w) ≠ 0, φ_i = ⟨w, gens[i]⟩ read off the column planes
// (⟨X_q, g⟩ = g.z_q, ⟨Z_q, g⟩ = g.x_q). The gens and the rref rows generate the same group, so this is the old
// verdict (some rref row anticommutes with w) exactly. Cost O(words · |supp(w)|), no allocation.
bool CertifiedGroupPlanes::anti_(const Pauli& w) const {
#ifdef XTIM_MUTATE_F3_ANTI_LOGICAL
    (void)w;
    return false;                                             // mutation build only (MUT-F3-ANTI): never ANTI
#else
    const int NWq = (n_qubits + 63) / 64;
    if (dense_cols) {
        for (int t = 0; t < words; ++t) {
            uint64_t acc = 0;
            for (int wq = 0; wq < NWq; ++wq) {
                for (uint64_t xs = w.x[(size_t)wq]; xs; xs &= xs - 1)
                    acc ^= zcol[(size_t)(wq * 64 + __builtin_ctzll(xs)) * (size_t)words + (size_t)t];
                for (uint64_t zs = w.z[(size_t)wq]; zs; zs &= zs - 1)
                    acc ^= xcol[(size_t)(wq * 64 + __builtin_ctzll(zs)) * (size_t)words + (size_t)t];
            }
            if (acc) return true;
        }
        return false;
    }
    // S2-T5: φ(w) by the column index, toggled into a words-long plane (the same bits as the word XORs); on the stack
    // for groups up to 1024 generators (no allocation on the plan builds' small groups).
    uint64_t small[16];
    std::vector<uint64_t> big;
    uint64_t* phi = small;
    if (words <= 16) std::fill(small, small + words, 0ULL);
    else { big.assign((size_t)words, 0); phi = big.data(); }
    for (int wq = 0; wq < NWq; ++wq) {
        for (uint64_t xs = w.x[(size_t)wq]; xs; xs &= xs - 1) xor_col(phi, wq * 64 + __builtin_ctzll(xs), false);
        for (uint64_t zs = w.z[(size_t)wq]; zs; zs &= zs - 1) xor_col(phi, wq * 64 + __builtin_ctzll(zs), true);
    }
    for (int t = 0; t < words; ++t) if (phi[t]) return true;
    return false;
#endif
}

Membership::Verdict CertifiedGroupPlanes::reduce_cheap(Pauli& w, int& phase_out) const {
    // Same reduction as reduce(), IN PLACE on the caller's Pauli (TLC: no scratch; the callers pass
    // their own pooled product, rebuilt before every call).
    walk_(w);
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }
    if (zero) {
        phase_out = w.phase;
        return Membership::IN_GROUP;
    }
    return anti_(w) ? Membership::ANTI : Membership::LOGICAL;
}

// Product-wire signs (Task 11.5): scan the rref rows whose symplectic pivot lies in the Z-block
// (pure-Z stabilisers) for WEIGHT-1 rows — those are the product wires ±Z_q. Cached lazily.
const CertifiedGroupPlanes::ZAxisRREF& CertifiedGroupPlanes::z_axis_rref() const {
    if (zaxis_ready_) return zaxis_;
    zaxis_.prod_sign.assign(n_qubits, 0);
    for (int r = 0; r < rank; ++r) {
        if (pivcol[r] < n_qubits) continue;        // has X-support ⇒ not pure-Z
        const SparsePauli& row = rref[r];
        const int piv = pivcol[r] - n_qubits;      // z-column pivot
        const int pop = (int)row.z.size(), only = pop ? (int)row.z.back() : -1;
        // ±Z_q is a weight-1 stabiliser (row support == {q}) ⇒ product wire; carry its ±1 sign.
        if (pop == 1 && only == piv)
            zaxis_.prod_sign[piv] = ((row.phase >> 1) & 1) ? (int8_t)-1 : (int8_t)+1;
    }
    zaxis_ready_ = true;
    return zaxis_;
}

// FNV-1a over the group's own content (n_qubits, n_gens, each gen's x/z words + phase), memoised on
// the object (Task-11 warm-cost lever). Byte-for-byte identical to the free group_token() this
// replaced — the plan-cache key is unchanged, existing caches/refs stay valid. The stream is each generator's
// dense words (ceil(g.n/64) per block, as the dense Pauli stored them), little-endian bytes. QW-E C1: computed
// from the sparse supports, O(weight + log n) per generator: a zero byte only multiplies (FNV-1a's XOR with 0 is the
// identity), so a run of k zero words is h *= P^(8k) mod 2^64 (binary powers of P^8). The densifying form is kept as
// the oracle in cpp/tests/test_planes_sparse.cpp (token_densified).
uint64_t CertifiedGroupPlanes::identity_token() const {
    if (token_ready_) return token_cache_;
    constexpr uint64_t P = 1099511628211ull;
    static const std::array<uint64_t, 64> P8 = [] {   // P8[i] = P^(8 * 2^i) mod 2^64
        std::array<uint64_t, 64> t{};
        uint64_t p = 1;
        for (int b = 0; b < 8; ++b) p *= P;
        for (int i = 0; i < 64; ++i) { t[(size_t)i] = p; p *= p; }
        return t;
    }();
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        for (int b = 0; b < 8; ++b) { h ^= (uint8_t)(v >> (8 * b)); h *= P; }
    };
    auto zero_words = [&](uint64_t k) {               // mix(0) k times
        for (int i = 0; k; ++i, k >>= 1)
            if (k & 1) h *= P8[(size_t)i];
    };
    auto mix_support = [&](const std::vector<int32_t>& s, size_t W) {   // ascending support -> W dense words
        size_t next = 0;
        for (size_t i = 0; i < s.size();) {
            const size_t w = (size_t)s[i] >> 6;
            uint64_t word = 0;
            for (; i < s.size() && ((size_t)s[i] >> 6) == w; ++i) word |= 1ULL << (s[i] & 63);
            zero_words(w - next);
            mix(word);
            next = w + 1;
        }
        zero_words(W - next);
    };
    mix((uint64_t)n_qubits);
    mix((uint64_t)n_gens);
    for (const SparsePauli& g : gens) {
        const size_t W = (size_t)(g.n + 63) / 64;
        mix_support(g.x, W);
        mix_support(g.z, W);
        mix((uint64_t)g.phase);
    }
    token_cache_ = h;
    token_ready_ = true;
    return token_cache_;
}

// ── T6: ProvenanceRREF (header comment). ──
// The heap walk, with the same pure-performance switch as the planes build (entry_cost_ scan tests per heap entry;
// 0: never switch): a reduction whose starting pivot bits already cost more than a scan of all rows scans from the
// start; otherwise, once the heap has taken more entries than the rows left to the scan (fill), the rest of the
// reduction tests the remaining rows' pivot bits in insertion order — the pre-T6 scan from that row on, the same rows
// in the same order. Both decisions read counts only.
void ProvenanceRREF::reduce_(std::vector<uint64_t>& sv, std::vector<uint64_t>& cv, std::vector<int>& heap) const {
    auto later = [](int a, int b) { return a > b; };          // min-heap of row indices
    const int R = (int)rows.size();
    long taken = 0;                                           // heap entries taken so far
    auto apply = [&](const DenseRow& e, bool feed) {
        for (size_t w = 0; w < sv.size(); ++w) {
            // Newly set pivot bits: later rows (row k is zero on the pivots of earlier rows).
            if (feed)
                for (uint64_t b = e.s[w] & ~sv[w] & pivmask[w]; b; b &= b - 1) {
                    heap.push_back(row_of_col[w * 64 + (size_t)__builtin_ctzll(b)]);
                    std::push_heap(heap.begin(), heap.end(), later);
                    ++taken;
                }
            sv[w] ^= e.s[w];
        }
        for (int w = 0; w < GW; ++w) cv[(size_t)w] ^= e.c[(size_t)w];
    };
    int last = -1;
    heap.clear();
    long start = 0;                                           // the starting pivot bits
    for (size_t w = 0; w < sv.size(); ++w) start += __builtin_popcountll(sv[w] & pivmask[w]);
    if (entry_cost_ <= 0 || entry_cost_ * start <= R) {
        for (size_t w = 0; w < sv.size(); ++w)
            for (uint64_t b = sv[w] & pivmask[w]; b; b &= b - 1)
                heap.push_back(row_of_col[w * 64 + (size_t)__builtin_ctzll(b)]);
        std::make_heap(heap.begin(), heap.end(), later);
        taken = start;
        while (!heap.empty()) {
            if (entry_cost_ > 0 && entry_cost_ * taken > R - last - 1) break;   // the scan from row last + 1
            std::pop_heap(heap.begin(), heap.end(), later);
            const int k = heap.back();
            heap.pop_back();
            if (k == last) continue;                          // a duplicate entry
            const DenseRow& e = rows[(size_t)k];
            if (!((sv[(size_t)(e.pc >> 6)] >> (e.pc & 63)) & 1)) continue;   // cleared since it was listed
            last = k;
            apply(e, true);
        }
        if (heap.empty()) return;
    }
    for (int k = last + 1; k < R; ++k) {
        const DenseRow& e = rows[(size_t)k];
        if ((sv[(size_t)(e.pc >> 6)] >> (e.pc & 63)) & 1) apply(e, false);
    }
}

ProvenanceRREF ProvenanceRREF::build(const std::vector<Pauli>& gens, int n, long entry_cost) {
    std::vector<SparsePauli> s;
    s.reserve(gens.size());
    for (const Pauli& P : gens) s.push_back(from_dense(P));
    return build(s, n, entry_cost);
}

ProvenanceRREF ProvenanceRREF::build(const std::vector<SparsePauli>& gens, int n, long entry_cost) {
    ProvenanceRREF P;
    P.entry_cost_ = entry_cost;
    P.n = n;
    P.NW = (n + 63) / 64;
    const int ng = (int)gens.size();
    P.GW = (ng + 63) / 64;
    P.row_of_col.assign((size_t)2 * 64 * P.NW, -1);
    P.pivmask.assign((size_t)2 * P.NW, 0);
    std::vector<int> heap;
    for (int i = 0; i < ng; ++i) {
        DenseRow r;
        r.s.assign((size_t)2 * P.NW, 0);
        for (int32_t q : gens[(size_t)i].x) r.s[(size_t)(q >> 6)] |= 1ULL << (q & 63);
        for (int32_t q : gens[(size_t)i].z) r.s[(size_t)(P.NW + (q >> 6))] |= 1ULL << (q & 63);
        r.c.assign((size_t)P.GW, 0);
        r.c[(size_t)(i >> 6)] |= 1ULL << (i & 63);
        P.reduce_(r.s, r.c, heap);
        r.pc = -1;
        for (size_t w = 0; w < r.s.size(); ++w)
            if (r.s[w]) { r.pc = (int)w * 64 + __builtin_ctzll(r.s[w]); break; }
        if (r.pc < 0) continue;                               // dependent: dropped
        P.row_of_col[(size_t)r.pc] = (int)P.rows.size();
        P.pivmask[(size_t)(r.pc >> 6)] |= 1ULL << (r.pc & 63);
        P.rows.push_back(std::move(r));
    }
    return P;
}

bool ProvenanceRREF::decompose(const Pauli& W, std::vector<uint64_t>& mask) const {
    Scratch sc;
    return decompose(W, mask, sc);
}

bool ProvenanceRREF::decompose(const Pauli& W, std::vector<uint64_t>& mask, Scratch& sc) const {
    std::vector<uint64_t>& sv = sc.sv;
    sv.resize((size_t)2 * NW);
    for (int w = 0; w < NW; ++w) { sv[(size_t)w] = W.x[(size_t)w]; sv[(size_t)(NW + w)] = W.z[(size_t)w]; }
    mask.assign((size_t)GW, 0);
    reduce_(sv, mask, sc.heap);
    for (uint64_t w : sv) if (w) return false;
    return true;
}

}  // namespace qeccore
