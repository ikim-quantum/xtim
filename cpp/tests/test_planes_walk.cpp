// test_planes_walk.cpp — perf-engine §2.1(b): the chunked pivot walk of CertifiedGroupPlanes (reduce /
// reduce_cheap) against the row-by-row reference walk, bit for bit.
//
// Reference (this test's own, over the planes' PUBLIC rref / col_to_row / pivcol): scan the probe's set
// symplectic bits ascending; at a pivot column multiply by that pivot row (pmul_into), i.e. the
// support-directed walk the engine ran before the chunk table. Theorem the engine relies on: in the full
// reduced echelon a pivot column occurs in its own row only, so the rows applied are exactly {r : probe has
// pivcol[r]} in ascending order, and w·R_a·R_b·… = w·(R_a·R_b·…) — identical (x, z, phase) when every partial
// product is taken in exact normal form. Checked here on: random commuting (stabilizer-like) groups, random
// NON-commuting row sets (the walk's identity needs no commutation), dense and sparse probes, n from 1 to 330
// (word boundaries: 64, 128, 192, 256), ranks from 0 to 2n, probes with every pivot set (byte 0xFF chunks)
// and with pivots only in the last partial chunk. Also: reduce()'s verdict / sign / rep and reduce_cheap()'s
// verdict / phase against the reference, and the chunk table's content against a direct ascending product.
#include "check.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "qeccore/pauli_kernels.hpp"
#include "qeccore/twirl_planes.hpp"

using namespace qeccore;

namespace {


// The reference walk (the pre-§2.1(b) engine walk, verbatim in structure).
void walk_ref(const CertifiedGroupPlanes& G, Pauli& w) {
    const int n = G.n_qubits;
    const int NWq = (n + 63) / 64;
    for (int blk = 0; blk < 2; ++blk) {
        std::vector<uint64_t>& vec = blk ? w.z : w.x;
        for (int wq = 0; wq < NWq; ++wq) {
            uint64_t word = vec[(size_t)wq];
            while (word) {
                const int b = __builtin_ctzll(word);
                const int q = wq * 64 + b;
                const int c = blk ? n + q : q;
                const int r = G.col_to_row[(size_t)c];
                if (r >= 0) {
                    pmul_into(w, G.rref[(size_t)r]);
                    word = (b == 63) ? 0 : (vec[(size_t)wq] & (~0ULL << (b + 1)));
                } else {
                    word &= word - 1;
                }
            }
        }
    }
}

// reduce() by the reference walk + the engine's (public-contract) verdict rule.
struct Ref { Membership::Verdict v; int sign; Pauli rep; };
Ref reduce_ref(const CertifiedGroupPlanes& G, const Pauli& p) {
    Pauli w = p;
    walk_ref(G, w);
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }
    if (zero) return {Membership::IN_GROUP, ((w.phase >> 1) & 1) ? -1 : +1, w};
    bool anti = false;
    for (const Pauli& g : G.gens) if (Pauli::anticommute_bit(w, g)) { anti = true; break; }
    return {anti ? Membership::ANTI : Membership::LOGICAL, 0, w};
}

Pauli random_pauli(std::mt19937_64& rng, int n, double dens) {
    Pauli P(n);
    std::bernoulli_distribution b(dens);
    for (int q = 0; q < n; ++q) { if (b(rng)) P.setx(q); if (b(rng)) P.setz(q); }
    P.phase = (int)(rng() & 3);
    return P;
}

// A random commuting set: products of a random symplectic-orthogonal basis (Z-type rows plus X-type rows on
// disjoint supports commute trivially; mixing them with random products keeps commutation).
std::vector<Pauli> random_commuting(std::mt19937_64& rng, int n, int m) {
    std::vector<Pauli> base;
    for (int q = 0; q < n; ++q) {
        Pauli P(n);
        if (rng() & 1) P.setz(q); else P.setx(q);
        base.push_back(P);
    }
    std::vector<Pauli> out;
    for (int i = 0; i < m; ++i) {
        Pauli P(n);
        for (int q = 0; q < n; ++q) if (rng() & 1) pmul_into(P, base[(size_t)q]);
        P.phase = (int)(rng() & 3);
        out.push_back(P);
    }
    return out;
}

std::vector<Pauli> random_any(std::mt19937_64& rng, int n, int m, double dens) {
    std::vector<Pauli> out;
    for (int i = 0; i < m; ++i) out.push_back(random_pauli(rng, n, dens));
    return out;
}

long checked = 0, in_group = 0, anti = 0, logical = 0;

// The same probe through the row-by-row fallback (the lever forces walk_rows_, the path a group above the chunk-table
// cap or a refused budget charge takes): identical to the chunked walk and the reference.
void check_probe_rows(const CertifiedGroupPlanes& G, const Pauli& p, const Ref& r) {
    const Membership m = G.reduce(p);
    CHECK(m.verdict == r.v);
    CHECK(m.sign == r.sign);
    CHECK(m.rep.x == r.rep.x);
    CHECK(m.rep.z == r.rep.z);
    CHECK(m.rep.phase == r.rep.phase);
    Pauli w = p;
    int ph = -1;
    const Membership::Verdict v = G.reduce_cheap(w, ph);
    CHECK(v == r.v);
    CHECK(w.x == r.rep.x);
    CHECK(w.z == r.rep.z);
    CHECK(w.phase == r.rep.phase);
}

const CertifiedGroupPlanes* g_rows = nullptr;   // the current group with the row-walk lever set

void check_probe(const CertifiedGroupPlanes& G, const Pauli& p) {
    const Ref r = reduce_ref(G, p);
    if (g_rows) check_probe_rows(*g_rows, p, r);
    const Membership m = G.reduce(p);
    CHECK(m.verdict == r.v);
    CHECK(m.sign == r.sign);
    CHECK(m.rep.x == r.rep.x);
    CHECK(m.rep.z == r.rep.z);
    CHECK(m.rep.phase == r.rep.phase);
    Pauli w = p;
    int ph = -1;
    const Membership::Verdict v = G.reduce_cheap(w, ph);
    CHECK(v == r.v);
    CHECK(w.x == r.rep.x);
    CHECK(w.z == r.rep.z);
    CHECK(w.phase == r.rep.phase);
    if (v == Membership::IN_GROUP) CHECK(ph == r.rep.phase);
    ++checked;
    if (r.v == Membership::IN_GROUP) ++in_group; else if (r.v == Membership::ANTI) ++anti; else ++logical;
}

// Every chunk entry == the ascending product of its selected rows (independent of the table's own recurrence).
void check_table(const CertifiedGroupPlanes& G) {
    const CertifiedGroupPlanes::ChunkTable& C = G.chunk_table();
    const int NW = (G.n_qubits + 63) / 64;
    CHECK(C.NW == NW);
    CHECK(C.nchunks == (G.rank + 7) / 8);
    for (int j = 0; j < C.nchunks; ++j) {
        const int cnt = std::min(8, G.rank - 8 * j);
        for (unsigned byte = 0; byte < (1u << cnt); ++byte) {
            Pauli prod(G.n_qubits);
            for (int t = 0; t < cnt; ++t) if ((byte >> t) & 1u) pmul_into(prod, G.rref[(size_t)(8 * j + t)]);
            const uint64_t* e = C.xz.data() + ((size_t)j * 256 + byte) * (size_t)(2 * NW);
            for (int i = 0; i < NW; ++i) { CHECK(e[i] == prod.x[(size_t)i]); CHECK(e[NW + i] == prod.z[(size_t)i]); }
            CHECK((int)C.ph[(size_t)j * 256 + byte] == prod.phase);
        }
    }
    // pivmask: exactly the pivot columns.
    for (int c = 0; c < 2 * G.n_qubits; ++c) {
        const int blk = c >= G.n_qubits ? 1 : 0, q = c - blk * G.n_qubits;
        const bool piv = G.col_to_row[(size_t)c] >= 0;
        CHECK((((C.pivmask[(size_t)blk * NW + (q >> 6)] >> (q & 63)) & 1ULL) != 0) == piv);
    }
}

void test_random_groups() {
    std::mt19937_64 rng(20261030);
    const int ns[] = {1, 2, 3, 7, 31, 63, 64, 65, 100, 127, 128, 129, 191, 192, 193, 255, 256, 257, 298, 330};
    for (int n : ns) {
        for (int kind = 0; kind < 2; ++kind) {
            const int ms[] = {0, 1, n / 2, n, 2 * n, 2 * n + 5};
            for (int m : ms) {
                std::vector<Pauli> rows = kind == 0 ? random_commuting(rng, n, m) : random_any(rng, n, m, 0.3);
                const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(rows, n);
                CertifiedGroupPlanes Grows = CertifiedGroupPlanes::build(rows, n);
                Grows.chunk_off_ = true;                 // the fallback walk on the same group
                g_rows = &Grows;
                check_table(G);
                CHECK(G.chunk_table().usable);
                CHECK(!Grows.chunk_table().usable);
                // probes: random dense / sparse / group products / identity / all-pivots / last-chunk-only
                for (int t = 0; t < 12; ++t) check_probe(G, random_pauli(rng, n, t < 6 ? 0.5 : 0.03));
                check_probe(G, Pauli(n));
                {
                    Pauli P(n);
                    for (const Pauli& g : rows) if (rng() & 1) pmul_into(P, g);
                    check_probe(G, P);
                }
                {
                    Pauli P(n);
                    for (int c : G.pivcol) { if (c < n) P.setx(c); else P.setz(c - n); }
                    check_probe(G, P);
                }
                if (G.rank > 8) {
                    Pauli P(n);
                    for (int r = 8 * ((G.rank - 1) / 8); r < G.rank; ++r) {
                        const int c = G.pivcol[(size_t)r];
                        if (c < n) P.setx(c); else P.setz(c - n);
                    }
                    P.phase = 2;
                    check_probe(G, P);
                }
                g_rows = nullptr;
            }
        }
    }
    std::fprintf(stderr, "[planes_walk] probes %ld (in-group %ld, anti %ld, logical %ld)\n", checked, in_group, anti, logical);
    CHECK(in_group > 0);
    CHECK(anti > 0);
    CHECK(logical > 0);
}

}  // namespace

int main() {
    RUN(test_random_groups);
    REPORT();
}
