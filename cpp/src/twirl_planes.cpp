#include "qeccore/twirl_planes.hpp"
#include "qeccore/bare_state_errors.hpp"   // BareStateCapacityError (the chunk table's budget refusal)
#include "qeccore/pauli_kernels.hpp"   // pmul_into

#include <algorithm>
#include <utility>
#if defined(__BMI2__)
#include <immintrin.h>                 // _pext_u64 (the chunked walk's pivot-bit gather)
#endif

namespace qeccore {

// symplectic-column bit: c in [0,n) -> x_c, [n,2n) -> z_{c-n}
static inline bool sbit(const Pauli& P, int c, int n) { return c < n ? P.xbit(c) : P.zbit(c - n); }

CertifiedGroupPlanes CertifiedGroupPlanes::build(const std::vector<Pauli>& certified, int n_qubits) {
    CertifiedGroupPlanes G;
    G.n_qubits = n_qubits;
    G.n_gens   = (int)certified.size();
    G.words    = (G.n_gens + 63) / 64;
    G.gens     = certified;

    // ── Columnar bit-planes: for each qubit q, pack bit i = gens[i].x/z(q). ──
    // Build is O(n_gens · n) here (row-major scatter into columns); reduce() is the hot path,
    // build is not. n up to ~298, n_gens up to ~297 ⇒ trivially cheap.
    G.xcol.assign((size_t)n_qubits * G.words, 0);
    G.zcol.assign((size_t)n_qubits * G.words, 0);
    for (int i = 0; i < G.n_gens; ++i) {
        const Pauli& g = certified[i];
        const uint64_t bit = 1ULL << (i & 63);
        const int wo = i >> 6;
        for (int q = 0; q < n_qubits; ++q) {
            if (g.xbit(q)) G.xcol[(size_t)q * G.words + wo] |= bit;
            if (g.zbit(q)) G.zcol[(size_t)q * G.words + wo] |= bit;
        }
    }

    // ── GF(2) RREF of the (x|z) symplectic rows, exact phase carried per row. ──
    // Columns 0..n-1 = X-block, n..2n-1 = Z-block. Full reduced echelon (each pivot column has a
    // single 1, in its own row) so reduce() is a single pivot-local pass in ascending column order.
    const int two_n = 2 * n_qubits;
    std::vector<Pauli> work = certified;    // mutated in place (exact phases via pmul_into)
    G.col_to_row.assign(two_n, -1);
    int used = 0;
    for (int c = 0; c < two_n; ++c) {
        int sel = -1;
        for (int i = used; i < G.n_gens; ++i) if (sbit(work[i], c, n_qubits)) { sel = i; break; }
        if (sel < 0) continue;
        if (sel != used) std::swap(work[used], work[sel]);
        const Pauli piv = work[used];       // copy: pmul_into below mutates other rows
        for (int i = 0; i < G.n_gens; ++i) {
            if (i == used) continue;
            if (sbit(work[i], c, n_qubits)) pmul_into(work[i], piv);
        }
        G.col_to_row[c] = used;
        G.pivcol.push_back(c);
        ++used;
    }
    G.rank = used;
    G.rref.assign(work.begin(), work.begin() + used);
    return G;
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
            const Pauli& R = rref[(size_t)(8 * j + t)];
            int sign = 0;
            for (int i = 0; i < NW; ++i) sign += __builtin_popcountll(pe[NW + i] & R.x[(size_t)i]);
            C.ph[(size_t)j * 256 + byte] = (uint8_t)((C.ph[(size_t)j * 256 + prev] + R.phase + 2 * (sign & 1)) & 3);
            for (int i = 0; i < NW; ++i) { e[i] = pe[i] ^ R.x[(size_t)i]; e[NW + i] = pe[NW + i] ^ R.z[(size_t)i]; }
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
                    pmul_into(w, rref[(size_t)r]);                    // clears bit c (pivot-exclusive)
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

const CertifiedGroupPlanes::XColIndex& CertifiedGroupPlanes::xcol_index() const {
    if (xidx_ready_) return xidx_;
    XColIndex& X = xidx_;
    X.off.assign((size_t)n_qubits + 1, 0);
    X.gen.clear();
    for (int q = 0; q < n_qubits; ++q) {
        const uint64_t* c = col(q, true);
        for (int w = 0; w < words; ++w)
            for (uint64_t x = c[w]; x; x &= x - 1) X.gen.push_back(w * 64 + __builtin_ctzll(x));
        X.off[(size_t)q + 1] = (int)X.gen.size();
    }
    xidx_ready_ = true;
    return xidx_;
}

// Product-wire signs (Task 11.5): scan the rref rows whose symplectic pivot lies in the Z-block
// (pure-Z stabilisers) for WEIGHT-1 rows — those are the product wires ±Z_q. Cached lazily.
const CertifiedGroupPlanes::ZAxisRREF& CertifiedGroupPlanes::z_axis_rref() const {
    if (zaxis_ready_) return zaxis_;
    zaxis_.prod_sign.assign(n_qubits, 0);
    for (int r = 0; r < rank; ++r) {
        if (pivcol[r] < n_qubits) continue;        // has X-support ⇒ not pure-Z
        const Pauli& row = rref[r];
        const int piv = pivcol[r] - n_qubits;      // z-column pivot
        int pop = 0, only = -1;
        for (size_t w = 0; w < row.z.size(); ++w) {
            uint64_t word = row.z[w];
            while (word) { only = (int)(w * 64 + __builtin_ctzll(word)); word &= word - 1; ++pop; }
        }
        // ±Z_q is a weight-1 stabiliser (row support == {q}) ⇒ product wire; carry its ±1 sign.
        if (pop == 1 && only == piv)
            zaxis_.prod_sign[piv] = ((row.phase >> 1) & 1) ? (int8_t)-1 : (int8_t)+1;
    }
    zaxis_ready_ = true;
    return zaxis_;
}

// FNV-1a over the group's own content (n_qubits, n_gens, each gen's x/z words + phase), memoised on
// the object (Task-11 warm-cost lever). Byte-for-byte identical to the free group_token() this
// replaced — the plan-cache key is unchanged, existing caches/refs stay valid.
uint64_t CertifiedGroupPlanes::identity_token() const {
    if (token_ready_) return token_cache_;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        for (int b = 0; b < 8; ++b) { h ^= (uint8_t)(v >> (8 * b)); h *= 1099511628211ull; }
    };
    mix((uint64_t)n_qubits);
    mix((uint64_t)n_gens);
    for (const Pauli& g : gens) {
        for (uint64_t w : g.x) mix(w);
        for (uint64_t w : g.z) mix(w);
        mix((uint64_t)g.phase);
    }
    token_cache_ = h;
    token_ready_ = true;
    return token_cache_;
}

}  // namespace qeccore
