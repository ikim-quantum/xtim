#include "qeccore/twirl_kernel.hpp"

namespace qeccore {

// Tier-1 sign planes + active mask (spec §5 step 2). All columnar / bit-sliced over the
// n_gens-bit planes: one pass touches every generator at once. See twirl_kernel.hpp for the
// derivation and the (proven) inactive-only sign contract.
Tier1Result tier1_signs(const CertifiedGroupPlanes& G, const DiagNormalForm& nf) {
    const int W = G.words;
    const int n = G.n_qubits;
    Tier1Result r;
    r.sign.assign(W, 0);
    r.active.assign(W, 0);
    if (W == 0) return r;

    // ── Plane 1: prefix anticommutation ⟨P, g_i⟩ = Σ_q x_P(q)·z_g(q) + z_P(q)·x_g(q). ──
    // supp_X(P): XOR the Z-column; supp_Z(P): XOR the X-column. Word-scan the prefix support
    // (2026-07-16: the per-qubit xbit/zbit walk paid n branchy tests even for the memo's
    // IDENTITY prefix — the common case).
    {
        const int NWq = (n + 63) / 64;
        for (int wq = 0; wq < NWq; ++wq) {
            uint64_t word = nf.prefix.x[wq];
            while (word) { const int q = wq * 64 + __builtin_ctzll(word); word &= word - 1;
                const uint64_t* c = G.col(q, false); for (int w = 0; w < W; ++w) r.sign[w] ^= c[w]; }
            word = nf.prefix.z[wq];
            while (word) { const int q = wq * 64 + __builtin_ctzll(word); word &= word - 1;
                const uint64_t* c = G.col(q, true);  for (int w = 0; w < W; ++w) r.sign[w] ^= c[w]; }
        }
    }

    // ── Plane 2: linear-a. Carry-save popcount (mod 4) of {xcol(q): a_q=1}. ──
    // ones = bit0 (mod-2), twos = bit1 (the sign bit). On ker M (inactive gens) ones==0 and twos
    // is the exact (1/2)Σ a_q x_{i,q} parity; on active gens the value is overridden by Task 8.
    {
        std::vector<uint64_t> ones(W, 0), twos(W, 0);
        for (int q = 0; q < n; ++q) {
            if (!nf.a[q]) continue;
            const uint64_t* c = G.col(q, true);
            for (int w = 0; w < W; ++w) {
                uint64_t carry = ones[w] & c[w];   // two 1s combine -> carry into the 2s plane
                ones[w] ^= c[w];
                twos[w] ^= carry;                  // 4s carry (twos & carry) is dropped: mod-4
            }
        }
        for (int w = 0; w < W; ++w) r.sign[w] ^= twos[w];
    }

    // ── Plane 3: CZ. For each pair (q,q'): sign ^= xcol(q) & xcol(q'). ──
    for (const auto& e : nf.cz) {
        const uint64_t* cq  = G.col(e.first,  true);
        const uint64_t* cqp = G.col(e.second, true);
        for (int w = 0; w < W; ++w) r.sign[w] ^= (cq[w] & cqp[w]);
    }

    // ── Active mask: active_i = (M x_i ≠ 0) = OR over M-rows j of (M x_i)_j. ──
    // M = diag(a) + cz-adjacency. Row j support = {j if a_j} ∪ {q : {j,q}∈cz}.
    // (M x_i)_j = pattern_j[i] = XOR of xcol over supp(row j); active |= pattern_j (per plane).
    // CSR adjacency for the cz graph (allocation-light: 3 flat arrays, no hashing / node churn).
    // (2026-07-16: pooled — these four vectors were fresh heap allocations per build.)
    struct SignScratch { std::vector<int> off, adj, cur; std::vector<uint64_t> pat; };
    thread_local SignScratch ss;
    ss.off.assign((size_t)n + 1, 0);
    std::vector<int>& off = ss.off;
    for (const auto& e : nf.cz) { ++off[e.first + 1]; ++off[e.second + 1]; }
    for (int j = 0; j < n; ++j) off[j + 1] += off[j];
    ss.adj.resize((size_t)off[n]);
    std::vector<int>& adj = ss.adj;
    { ss.cur.assign(off.begin(), off.end() - 1);
      std::vector<int>& cur = ss.cur;
      for (const auto& e : nf.cz) { adj[cur[e.first]++] = e.second; adj[cur[e.second]++] = e.first; } }

    ss.pat.resize((size_t)W);
    std::vector<uint64_t>& pat = ss.pat;
    for (int j = 0; j < n; ++j) {
        const bool has_a  = nf.a[j] != 0;
        const bool has_cz = off[j + 1] > off[j];
        if (!has_a && !has_cz) continue;                 // zero M-row: contributes nothing
        for (int w = 0; w < W; ++w) pat[w] = 0;
        if (has_a) { const uint64_t* c = G.col(j, true); for (int w = 0; w < W; ++w) pat[w] ^= c[w]; }
        for (int e = off[j]; e < off[j + 1]; ++e) {
            const uint64_t* c = G.col(adj[e], true); for (int w = 0; w < W; ++w) pat[w] ^= c[w];
        }
        for (int w = 0; w < W; ++w) r.active[w] |= pat[w];
    }

    return r;
}

}  // namespace qeccore
