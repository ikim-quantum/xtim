// test_planes_build.cpp — T6: CertifiedGroupPlanes' set-bit build (build_sparse at forced switch cost ratios; build()
// uses kListEntryCost) and ProvenanceRREF's set-pivot reduction (likewise) against the pre-T6 dense code kept as their
// oracle (DenseGroupPlanes::build_dense; ProvenanceRREF's dense_provenance / dense_decompose below), bit for bit.
// S2-T5: the sparse planes against the dense oracle (planes_oracle.hpp), both working-row forms at every cost ratio.
//
// Planes: xcol, zcol, gens, rref (x, z, phase), pivcol, col_to_row, rank. Provenance: every row's pivot, symplectic
// bits and generator combination, and decompose()'s verdict and mask on probes. Checked on random commuting and
// NON-commuting row sets (the elimination order is the oracle's, so even non-commuting products agree), sparse and
// dense, dependent sets (m up to 2n + 5), n across word boundaries; and on the certified generators of every corpus
// circuit (tests/data, examples, the private corpus when present) with each detector / observable record product as
// a probe.
//
// Must end main() with REPORT() (check.hpp).
#include "check.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_planes.hpp"
#include "planes_oracle.hpp"

using namespace qeccore;

namespace {

// The switches' cost ratio, forced (pure performance: every value must give the oracle's bits): 0 = never switch
// (lists / heap throughout), 1 … 64 = switch at different points, 2^30 = switch at once (scan from the start; above
// every test's entry counts, and 2^30 times any count below 2^32 stays inside a long).
const long kCosts[] = {0, 1, 2, 3, 8, 32, 64, 1L << 30};

using planes_oracle::same_pauli;
using planes_oracle::same_planes;
using planes_oracle::sparse_rows;

// ProvenanceRREF's oracle: the pre-T6 code (twirl_sampler.cpp's provenance lambdas), verbatim in structure (rows only).
ProvenanceRREF dense_provenance(const std::vector<Pauli>& gens, int n) {
    ProvenanceRREF P;
    P.n = n;
    P.NW = (n + 63) / 64;
    const int NW2 = P.NW;
    const int ng = (int)gens.size();
    const int GW = (ng + 63) / 64;
    P.GW = GW;
    std::vector<ProvenanceRREF::DenseRow>& prref = P.rows;
    auto sympl = [&](const Pauli& Q) {
        std::vector<uint64_t> s(2 * (size_t)NW2, 0);
        for (int w = 0; w < NW2; ++w) { s[w] = Q.x[w]; s[NW2 + w] = Q.z[w]; }
        return s;
    };
    auto sfirst = [&](const std::vector<uint64_t>& s) {
        for (size_t w = 0; w < s.size(); ++w)
            if (s[w]) return (int)w * 64 + __builtin_ctzll(s[w]);
        return -1;
    };
    for (int i = 0; i < ng; ++i) {
        ProvenanceRREF::DenseRow r; r.s = sympl(gens[i]); r.c.assign(GW, 0); r.c[i >> 6] |= 1ULL << (i & 63);
        for (const ProvenanceRREF::DenseRow& e : prref)
            if ((r.s[e.pc >> 6] >> (e.pc & 63)) & 1) {
                for (size_t w = 0; w < r.s.size(); ++w) r.s[w] ^= e.s[w];
                for (int w = 0; w < GW; ++w) r.c[w] ^= e.c[w];
            }
        r.pc = sfirst(r.s);
        if (r.pc >= 0) prref.push_back(std::move(r));
    }
    return P;
}

// Its decompose: test every row's pivot bit in insertion order.
bool dense_decompose(const ProvenanceRREF& P, const Pauli& W, std::vector<uint64_t>& mask) {
    std::vector<uint64_t> s(2 * (size_t)P.NW, 0);
    for (int w = 0; w < P.NW; ++w) { s[w] = W.x[w]; s[P.NW + w] = W.z[w]; }
    mask.assign(P.GW, 0);
    for (const ProvenanceRREF::DenseRow& e : P.rows)
        if ((s[e.pc >> 6] >> (e.pc & 63)) & 1) {
            for (size_t w = 0; w < s.size(); ++w) s[w] ^= e.s[w];
            for (int w = 0; w < P.GW; ++w) mask[w] ^= e.c[w];
        }
    for (uint64_t w : s) if (w) return false;
    return true;
}

long planes_sets = 0;

void check_planes(const std::vector<Pauli>& rows, int n) {
    const DenseGroupPlanes B = DenseGroupPlanes::build_dense(rows, n);
    same_planes(CertifiedGroupPlanes::build(rows, n), B);                             // the production switch
    CertifiedGroupPlanes::build(rows, n).check_index();                              // S2-T6 m1: the index check passes
    using L = CertifiedGroupPlanes::Layout;
    const std::vector<SparsePauli> s = sparse_rows(rows);
    for (long cost : kCosts)                                                         // S2-T5: both working-row forms
        for (L rl : {L::Sparse, L::Dense}) same_planes(CertifiedGroupPlanes::build_sparse(s, n, cost, L::Auto, rl), B);
    for (long k : {1L, (long)rows.size() / 2})                                        // S2-FINALFIX: switched part-way
        same_planes(CertifiedGroupPlanes::build_sparse(s, n, kListEntryCost, L::Auto, L::Auto, k), B);
    ++planes_sets;
}

long probes = 0, probes_in = 0;

// S2-T6: the planes' record-column masks (reduce_sparse's mask: the forward-row reduction) against the oracle's
// decomposition mb of a W in the span: the same generator list when the rows are independent (the decomposition is unique), and in every
// case a combination whose product has W's symplectic bits. On every layout and switch point of the planes build.
long prov_checked = 0, prov_unique = 0;
void check_planes_provenance(const std::vector<CertifiedGroupPlanes>& Gs, const std::vector<Pauli>& rows, int n,
                             const Pauli& W, const std::vector<uint64_t>& mb, bool indep) {
    std::vector<int32_t> want;
    for (size_t w = 0; w < mb.size(); ++w)
        for (uint64_t b = mb[w]; b; b &= b - 1) want.push_back((int32_t)(w * 64 + __builtin_ctzll(b)));
    for (const CertifiedGroupPlanes& G : Gs) {
        CertifiedGroupPlanes::ProvScratch psc;
        std::vector<int32_t> got;
        const SparsePauli Ws = from_dense(W);
        int ph = 0;
        CHECK(G.reduce_sparse(Ws, ph, &got, psc) == Membership::IN_GROUP);
        CHECK(G.reduce_sparse(Ws, ph, &got, psc) == Membership::IN_GROUP);   // the scratch is left zeroed: the same
        CHECK(std::is_sorted(got.begin(), got.end()) && std::adjacent_find(got.begin(), got.end()) == got.end());
        if (indep) CHECK(got == want);
        Pauli prod(n);
        for (int32_t i : got) for (size_t w = 0; w < prod.x.size(); ++w) {
            prod.x[w] ^= rows[(size_t)i].x[w];
            prod.z[w] ^= rows[(size_t)i].z[w];
        }
        CHECK(prod.x == W.x && prod.z == W.z);
        ++prov_checked;
        if (indep) ++prov_unique;
    }
}

// S2-T6: reduce_sparse (the classify of the record columns) against reduce_cheap on the dense probe: the verdict, the
// IN_GROUP phase, and (in the span, independent rows) the oracle's mask; on every layout. Both sides of its dense
// switch are reached: the corpus' record products and the group products are light, the random dens-0.5 probes heavy.
long rs_checked = 0;
void check_reduce_sparse(const std::vector<CertifiedGroupPlanes>& Gs, const Pauli& W, const std::vector<uint64_t>* mb,
                         bool indep) {
    const SparsePauli Ws = from_dense(W);
    for (const CertifiedGroupPlanes& G : Gs) {
        Pauli d = W;
        int pd = -1, ps = -1;
        const Membership::Verdict vd = G.reduce_cheap(d, pd);
        CertifiedGroupPlanes::ProvScratch sc;
        std::vector<int32_t> m;
        const Membership::Verdict vs = G.reduce_sparse(Ws, ps, &m, sc);
        CHECK(vd == vs);
        if (vd == Membership::IN_GROUP) {
            CHECK(pd == ps);
            if (mb && indep) {
                std::vector<int32_t> want;
                for (size_t w = 0; w < mb->size(); ++w)
                    for (uint64_t b = (*mb)[w]; b; b &= b - 1) want.push_back((int32_t)(w * 64 + __builtin_ctzll(b)));
                CHECK(m == want);
            }
        }
        ++rs_checked;
    }
}

void check_provenance(const std::vector<Pauli>& rows, int n, const std::vector<Pauli>& probe_set) {
    const ProvenanceRREF P = ProvenanceRREF::build(rows, n);
    const ProvenanceRREF D = dense_provenance(rows, n);
    using L = CertifiedGroupPlanes::Layout;
    std::vector<CertifiedGroupPlanes> Gs;
    Gs.push_back(CertifiedGroupPlanes::build(rows, n));
    for (long cost : {0L, 8L, 1L << 30})
        for (L rl : {L::Sparse, L::Dense}) Gs.push_back(CertifiedGroupPlanes::build_sparse(sparse_rows(rows), n, cost, L::Auto, rl));
    // S2-FINALFIX: the working rows switched to packed words part-way (the forward form dropped there: the prref route)
    for (long k : {1L, (long)rows.size() / 2})
        Gs.push_back(CertifiedGroupPlanes::build_sparse(sparse_rows(rows), n, kListEntryCost, L::Auto, L::Auto, k));
    // S2-T6 fix 4: the forward form switched to the prref route at query 0, 1 and 3 (the time budget's lever), and
    // never — every mask on either route and across a switch.
    for (long lim : {0L, 1L, 3L, 1L << 40}) {
        Gs.push_back(CertifiedGroupPlanes::build_sparse(sparse_rows(rows), n, 0, L::Auto, L::Sparse));
        Gs.back().fwd_query_limit_ = lim;
    }
    const bool indep = D.rows.size() == rows.size();
    std::vector<ProvenanceRREF> forced;
    for (long cost : kCosts) forced.push_back(ProvenanceRREF::build(rows, n, cost));
    for (const ProvenanceRREF& H : forced) {
        CHECK(H.rows.size() == D.rows.size());
        for (size_t r = 0; r < D.rows.size() && r < H.rows.size(); ++r)
            CHECK(H.rows[r].pc == D.rows[r].pc && H.rows[r].s == D.rows[r].s && H.rows[r].c == D.rows[r].c);
    }
    ProvenanceRREF::Scratch sc;                       // reused across probes, as the sampler's classify does
    const int GW = ((int)rows.size() + 63) / 64;
    CHECK(P.GW == GW && D.GW == GW);
    CHECK(P.rows.size() == D.rows.size());
    for (size_t r = 0; r < D.rows.size() && r < P.rows.size(); ++r) {
        CHECK(P.rows[r].pc == D.rows[r].pc);
        CHECK(P.rows[r].s == D.rows[r].s);
        CHECK(P.rows[r].c == D.rows[r].c);
    }
    for (const Pauli& W : probe_set) {
        std::vector<uint64_t> ma, mb;
        const bool a = P.decompose(W, ma);
        const bool b = dense_decompose(D, W, mb);
        CHECK(a == b);
        CHECK(ma == mb);
        for (const ProvenanceRREF& H : forced) {
            std::vector<uint64_t> mh;
            const bool hh = H.decompose(W, mh, sc);
            CHECK(hh == b);
            CHECK(mh == mb);
        }
        ++probes;
        if (a) ++probes_in;
        if (a) check_planes_provenance(Gs, rows, n, W, mb, indep);
        check_reduce_sparse(Gs, W, a ? &mb : nullptr, indep);
    }
}

Pauli random_pauli(std::mt19937_64& rng, int n, double dens) {
    Pauli P(n);
    std::bernoulli_distribution b(dens);
    for (int q = 0; q < n; ++q) { if (b(rng)) P.setx(q); if (b(rng)) P.setz(q); }
    P.phase = (int)(rng() & 3);
    return P;
}

// Commuting rows: random products of a random single-qubit X/Z basis (weight ~ dens · n).
std::vector<Pauli> random_commuting(std::mt19937_64& rng, int n, int m, double dens) {
    std::vector<Pauli> base;
    for (int q = 0; q < n; ++q) {
        Pauli P(n);
        if (rng() & 1) P.setz(q); else P.setx(q);
        base.push_back(P);
    }
    std::bernoulli_distribution b(dens);
    std::vector<Pauli> out;
    for (int i = 0; i < m; ++i) {
        Pauli P(n);
        for (int q = 0; q < n; ++q) if (b(rng)) pmul_into(P, base[(size_t)q]);
        P.phase = (int)(rng() & 3) & 2;
        out.push_back(P);
    }
    return out;
}

std::vector<Pauli> group_probes(std::mt19937_64& rng, const std::vector<Pauli>& rows, int n) {
    std::vector<Pauli> out;
    for (int t = 0; t < 6; ++t) {
        Pauli P(n);
        for (const Pauli& g : rows) if (rng() & 1) pmul_into(P, g);
        out.push_back(P);
    }
    for (int t = 0; t < 4; ++t) out.push_back(random_pauli(rng, n, t < 2 ? 0.5 : 0.02));
    out.push_back(Pauli(n));
    return out;
}

void test_random() {
    std::mt19937_64 rng(20261121);
    const int ns[] = {1, 2, 3, 7, 31, 63, 64, 65, 100, 127, 128, 129, 191, 192, 193, 257, 330};
    int sets = 0;
    for (int n : ns) {
        const int ms[] = {0, 1, n / 2, n, 2 * n, 2 * n + 5};
        for (int m : ms)
            for (int kind = 0; kind < 4; ++kind) {
                std::vector<Pauli> rows;
                if (kind == 0) rows = random_commuting(rng, n, m, 0.5);
                else if (kind == 1) rows = random_commuting(rng, n, m, 2.0 / std::max(n, 1));
                else if (kind == 2) for (int i = 0; i < m; ++i) rows.push_back(random_pauli(rng, n, 0.3));
                else for (int i = 0; i < m; ++i) rows.push_back(random_pauli(rng, n, 1.5 / std::max(n, 1)));
                if (m > 2 && kind % 2 == 1) {                 // force duplicates / dependent rows
                    rows[(size_t)m - 1] = rows[0];
                    Pauli P = rows[1]; pmul_into(P, rows[2]); rows[(size_t)m - 2] = P;
                }
                check_planes(rows, n);
                check_provenance(rows, n, group_probes(rng, rows, n));
                ++sets;
            }
    }
    std::fprintf(stderr, "[planes_build] random: %d row sets\n", sets);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One corpus circuit: the bindings' twirl front end (test_threads' compile), then the certified generators.
bool corpus_one(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) return false;
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    pol.sparse_class_table = true;
    NormalizeResult nr;
    try { nr = normalize(ps.circuit, pol); } catch (const std::exception&) { return false; }
    FramedBareState bs;
    try { bs = build_bare_state_framed(nr.normalized); } catch (const std::exception&) { return false; }
    if (bs.rejected) return false;
    const int n = bs.state.n();
    const std::vector<Pauli> gens = bs.state.certified_stabilizers();
    std::vector<std::pair<int, int>> reads;
    for (const auto& r : nr.map.terminal_reads) reads.push_back({(int)r.second, r.first});
    auto product = [&](const std::vector<int>& recs) {
        Pauli W(n);
        for (int k : recs) {
            if (k < 0 || k >= (int)reads.size()) continue;
            Pauli P(n);
            const int b2 = reads[(size_t)k].first, q = reads[(size_t)k].second;
            if (b2 == 0) P.setx(q);
            else if (b2 == 2) P.setz(q);
            else { P.setx(q); P.setz(q); P.phase = 1; }
            W = Pauli::multiply(W, P);
        }
        return W;
    };
    std::vector<Pauli> probe_set;
    for (const auto& d : ps.detectors) probe_set.push_back(product(d));
    for (const auto& kv : ps.observables) probe_set.push_back(product(kv.second));
    std::mt19937_64 rng(7);
    for (const Pauli& P : group_probes(rng, gens, n)) probe_set.push_back(P);
    check_planes(gens, n);
    check_provenance(gens, n, probe_set);
    return true;
}

// S2-T6 fix 1/2: chain-structured Clifford families (from the reviews): GHZ / cat chains (CX i i+1, forward or
// reversed) — their fully reduced rows Z_i·Z_{n-1} have combinations {i .. n-2}, so tracking combinations through the
// back-substitution was cubic (fix 1) —, a chain then a fan-out from qubit 0 (fan rows ride the whole chain; merging
// their forward combinations was near-cubic: fix 2), the same fan-out first, and a GHZ chain measured twice. The
// oracle on small sizes (corpus_one), and a WORK tripwire on n = 4000 / 8000 that does not depend on the machine's
// load: the forward rows' steps are the elimination's own multiplies (one entry each), the masks' forward reductions
// apply no more rows than the elimination multiplied (plus O(1) per detector), and the expansions through the steps
// cost a small multiple of the steps and rows they read.
std::string family_text(const std::string& kind, int n) {
    auto all = [&]() { std::string t; for (int q = 0; q < n; ++q) t += " " + std::to_string(q); return t; };
    auto num = [](int v) { return std::to_string(v); };
    std::string t = "R" + all() + "\n";
    const int m = n / 2;
    if (kind == "ghz") { t += "H 0\n"; for (int i = 0; i + 1 < n; ++i) t += "CX " + num(i) + " " + num(i + 1) + "\n"; }
    else if (kind == "ghzrev") { t += "H " + num(n - 1) + "\n"; for (int i = n - 1; i > 0; --i) t += "CX " + num(i) + " " + num(i - 1) + "\n"; }
    else if (kind == "chainfan" || kind == "chainfanrev") {
        t += "H 0\n";
        std::string chain, fan;
        for (int i = 0; i + 1 < m; ++i) chain += "CX " + num(i) + " " + num(i + 1) + "\n";
        for (int j = m; j < n; ++j) fan += "CX 0 " + num(j) + "\n";
        t += kind == "chainfan" ? chain + fan : fan + chain;
    } else if (kind == "redund" || kind.rfind("redk", 0) == 0) {
        t += "H 0\n";
        for (int i = 0; i + 1 < n; ++i) t += "CX " + num(i) + " " + num(i + 1) + "\n";
    } else if (kind == "star") { t += "H 0\n"; for (int i = 1; i < n; ++i) t += "CX 0 " + num(i) + "\n"; }
    else if (kind.rfind("cfx", 0) == 0 || kind.rfind("qfx3", 0) == 0) {   // chainfan + K·n fan detectors (fix 3/4)
        t += "H 0\n";
        for (int i = 0; i + 1 < m; ++i) t += "CX " + num(i) + " " + num(i + 1) + "\n";
        for (int j = m; j < n; ++j) t += "CX 0 " + num(j) + "\n";
    } else if (kind.rfind("ghzl", 0) == 0 || kind.rfind("rkf", 0) == 0) {   // fix 4: the GHZ chain
        t += "H 0\n";
        for (int i = 0; i + 1 < n; ++i) t += "CX " + num(i) + " " + num(i + 1) + "\n";
    }
    t += "X_ERROR(0.01)" + all() + "\n";
    if (kind.rfind("rkf", 0) == 0) {
        // fix 4 (rev4's storage-cap edge): the GHZ chain measured twice, plus a third measurement of its first pct %
        const int f = n * std::stoi(kind.substr(3)) / 100, T = 2 * n + f;
        t += "M" + all() + "\nM" + all() + "\n";
        if (f > 0) { t += "M"; for (int q = 0; q < f; ++q) t += " " + num(q); t += "\n"; }
        for (int i = 0; i < n; ++i) t += "DETECTOR rec[-" + num(T - i) + "] rec[-" + num(T - (n + i)) + "]\n";
        for (int i = 0; i < f; ++i) t += "DETECTOR rec[-" + num(T - (n + i)) + "] rec[-" + num(T - (2 * n + i)) + "]\n";
        for (int i = 0; i + 1 < n; ++i)
            t += "DETECTOR rec[-" + num(T - (n + i)) + "] rec[-" + num(T - (n + i + 1)) + "]\n";
        return t;
    }
    const int K = kind == "redund" ? 2 : kind.rfind("redk", 0) == 0 ? std::stoi(kind.substr(4)) : 1;
    for (int k = 0; k < K; ++k) t += "M" + all() + "\n";          // the GHZ chain measured K times (redk)
    for (int r = 1; r < K; ++r)
        for (int i = 0; i < n; ++i)
            t += "DETECTOR rec[-" + num((K - r + 1) * n - i) + "] rec[-" + num((K - r) * n - i) + "]\n";
    for (int i = 0; i + 1 < n; ++i) t += "DETECTOR rec[-" + num(n - i) + "] rec[-" + num(n - i - 1) + "]\n";
    if (kind.rfind("qfx3", 0) == 0) {                // fix 4 (rev4 F1): K·n weight-4 parities of distinct fan qubits
        std::mt19937_64 rr(13);
        const long extra = std::stol(kind.substr(4)) * n;
        for (long e = 0; e < extra && n - m > 3; ++e) {
            int q[4];
            for (int& v : q) v = m + (int)(rr() % (uint64_t)(n - m));
            if (q[0] == q[1] || q[0] == q[2] || q[0] == q[3] || q[1] == q[2] || q[1] == q[3] || q[2] == q[3]) continue;
            t += "DETECTOR";
            for (int v : q) t += " rec[-" + num(n - v) + "]";
            t += "\n";
        }
    }
    if (kind.rfind("ghzl", 0) == 0) {                // fix 4 (rev4 F2): K·n far pairs (first tenth x last tenth)
        std::mt19937_64 rr(17);
        const long extra = std::stol(kind.substr(4)) * n;
        const int tt = std::max(1, n / 10);
        for (long e = 0; e < extra; ++e) {
            const int a = (int)(rr() % (uint64_t)tt), b = n - 1 - (int)(rr() % (uint64_t)tt);
            if (a != b) t += "DETECTOR rec[-" + num(n - a) + "] rec[-" + num(n - b) + "]\n";
        }
    }
    if (kind.rfind("cfx", 0) == 0) {
        std::mt19937_64 rr(11);
        const long extra = std::stol(kind.substr(3)) * n;
        for (long e = 0; e < extra && n - m > 1; ++e) {
            const int a = m + (int)(rr() % (uint64_t)(n - m)), b = m + (int)(rr() % (uint64_t)(n - m));
            if (a != b) t += "DETECTOR rec[-" + num(n - a) + "] rec[-" + num(n - b) + "]\n";
        }
    }
    return t;
}

// expect: the route the masks must take — "kept" (the forward form serves every query), "spent" (the time budget
// passes the rest to the prref route), "dropped" (the storage bound), or "" (any). Returns nothing; CHECKs the bounds.
void family_tripwire(const std::string& kind, int n, const std::string& expect = "") {
    ParsedStim ps = parse_stim_circuit(family_text(kind, n));
    CHECK(ps.ok());
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    pol.sparse_class_table = true;
    const NormalizeResult nr = normalize(ps.circuit, pol);
    const FramedBareState bs = build_bare_state_framed(nr.normalized);
    const int nq = bs.state.n();
    std::vector<SparsePauli> gens = bs.state.certified_stabilizers_sparse();
    long wg = 0;
    for (const SparsePauli& g : gens) wg += (long)(g.x.size() + g.z.size());
    const int ng = (int)gens.size();
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(std::move(gens), nq);
    CHECK(G.rank == ng);
    G.check_index();                                 // the column index is the transpose (throws otherwise)
    const long steps = G.fwd_steps, wech = (long)G.ech_col.size();
    const bool forward = !G.ech_off.empty();
    const size_t fwd_bytes = G.fwd_bytes, prref_bytes = G.prref_route_bytes();
    std::vector<std::pair<int, int>> reads;
    for (const auto& r : nr.map.terminal_reads) reads.push_back({(int)r.second, r.first});
    CertifiedGroupPlanes::ProvScratch sc;
    long applied = 0, maskw = 0, in = 0, max_query = 0;
    size_t mask_bytes = 0, list_bytes = 0, base_mask_bytes = 0;
    for (const auto& d : ps.detectors) {
        Pauli W(nq);
        for (int k : d) {
            Pauli P(nq);
            if (reads[(size_t)k].first == 2) P.setz(reads[(size_t)k].second); else P.setx(reads[(size_t)k].second);
            W = Pauli::multiply(W, P);
        }
        std::vector<int32_t> m;
        int ph = 0;
        const long w0 = G.mask_work_;
        if (G.reduce_sparse(from_dense(W), ph, &m, sc) != Membership::IN_GROUP) continue;
        max_query = std::max(max_query, G.mask_work_ - w0);
        ++in;
        applied += (long)sc.rows.size();             // the rows the mask's reduction walk took
        maskw += (long)m.size();
        list_bytes += 4 * m.size();
        base_mask_bytes += 8 * (size_t)G.words;      // eb55e1e's mask: GW words
        mask_bytes += GenMask(std::move(m)).heap_bytes();
    }
    const long D = (long)ps.detectors.size();
    std::fprintf(stderr, "[planes_build] %s n=%d: gens=%d sum|gens|=%ld multiplies=%ld steps=%ld sum|E|=%ld; %ld/%ld "
                 "detectors in group, %ld rows walked, expansion work %ld, %ld mask entries\n", kind.c_str(), n, ng, wg,
                 G.build_multiplies, steps, wech, in, D, applied, sc.work, maskw);
    CHECK(in == D);
    // S2-T6 fix 4, the two bounds stated in twirl_planes (by construction; checked here on the counts, load-free):
    // storage — the forward form's heap bytes (capacity, headers) never exceed kFwdBytesPct % of the prref route's
    // bytes (else the group took the prref route); time — the forward units, priced at kFwdUnitWords word ops, never
    // exceed kMaskSpendPct % of the prref build's estimated word ops plus (100 + kMaskSpendPct) % of its decomposes'
    // on the forward queries, plus the one query that crossed it.
    const long budget = kMaskSpendPct * G.prref_build_words() + (100 + kMaskSpendPct) * G.mask_credit_;
    CHECK(!forward || 100 * fwd_bytes <= (size_t)kFwdBytesPct * prref_bytes);
    CHECK(steps <= G.build_multiplies);              // one entry per forward multiply, never a merged combination
    CHECK(100 * kFwdUnitWords * (G.mask_work_ - max_query) <= budget);
    // fix 4 (rev4 F2): a mask costs the cheaper of its list and its bitset — never more than eb55e1e's GW words
    CHECK(mask_bytes <= base_mask_bytes && mask_bytes <= list_bytes);
    const char* route = !forward ? "dropped" : G.fwd_spent_ ? "spent" : "kept";
    if (!expect.empty()) CHECK(expect == route);
    // S2-CLEAN (fix-4 review n3): a spent group freed its forward form at the switch (not held beside the prref route)
    if (G.fwd_spent_)
        CHECK(G.prov_steps.capacity() == 0 && G.prov_row.capacity() == 0 && G.ech_col.capacity() == 0 &&
              G.ech_off.capacity() == 0);
    std::fprintf(stderr, "[planes_build]   %s n=%d: forward form %zu B against prref %zu B, route %s (expected %s), "
                 "mask units %ld (max %ld a query) over %ld queries, budget %ld word ops; masks %zu B (lists %zu B, "
                 "eb55e1e %zu B)\n", kind.c_str(), n, fwd_bytes, prref_bytes, route, expect.empty() ? "any" : expect.c_str(),
                 G.mask_work_, max_query, G.mask_queries_, budget / 100,
                 mask_bytes, list_bytes, base_mask_bytes);
}

// fix 4 (rev4 F2/F3): the stored bytes against eb55e1e's, by capacity — the forward form at most kFwdBytesPct % of the
// prref route's bytes, the masks at most mask_frac of eb55e1e's GW words (1: long masks, ghzl, where the lists were up
// to 32x larger), and the forward form plus the masks never above the prref route plus eb55e1e's masks.
void family_bytes(const std::string& kind, int n, double mask_frac) {
    ParsedStim ps = parse_stim_circuit(family_text(kind, n));
    CHECK(ps.ok());
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    pol.sparse_class_table = true;
    const NormalizeResult nr = normalize(ps.circuit, pol);
    const FramedBareState bs = build_bare_state_framed(nr.normalized);
    const int nq = bs.state.n();
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(bs.state.certified_stabilizers_sparse(), nq);
    std::vector<std::pair<int, int>> reads;
    for (const auto& r : nr.map.terminal_reads) reads.push_back({(int)r.second, r.first});
    CertifiedGroupPlanes::ProvScratch sc;
    size_t mask_bytes = 0, base_mask_bytes = 0;
    for (const auto& d : ps.detectors) {
        SparsePauli W;
        W.n = nq;
        std::vector<int32_t> xs, zs;
        for (int k : d) (reads[(size_t)k].first == 2 ? zs : xs).push_back(reads[(size_t)k].second);
        auto parity = [](std::vector<int32_t>& v) {
            std::sort(v.begin(), v.end());
            std::vector<int32_t> o;
            for (size_t i = 0; i < v.size();) {
                size_t j = i;
                while (j < v.size() && v[j] == v[i]) ++j;
                if ((j - i) & 1) o.push_back(v[i]);
                i = j;
            }
            return o;
        };
        W.x = parity(xs);
        W.z = parity(zs);
        std::vector<int32_t> m;
        int ph = 0;
        CHECK(G.reduce_sparse(W, ph, &m, sc) == Membership::IN_GROUP);
        base_mask_bytes += 8 * (size_t)G.words;
        mask_bytes += GenMask(std::move(m)).heap_bytes();
    }
    const bool forward = !G.ech_off.empty();
    const size_t fwd = forward ? G.fwd_bytes : 0, prref = G.prref_route_bytes();
    std::fprintf(stderr, "[planes_build] bytes %s n=%d: forward form %zu B (%s) against prref %zu B; masks %zu B against "
                 "eb55e1e %zu B\n", kind.c_str(), n, fwd, forward ? "kept" : "dropped", prref, mask_bytes, base_mask_bytes);
    CHECK(100 * fwd <= (size_t)kFwdBytesPct * prref);
    CHECK((double)mask_bytes <= mask_frac * (double)base_mask_bytes);
    CHECK(fwd + mask_bytes <= prref + base_mask_bytes);
}

void test_families() {
    for (const char* k : {"ghz", "ghzrev", "chainfan", "chainfanrev", "redund", "star", "cfx20", "redk3", "redk4",
                          "redk6", "qfx32", "qfx35", "ghzl20", "rkf40"})
        for (int n : {3, 64, 65, 200}) CHECK(corpus_one(family_text(k, n)));   // the oracle
    for (const char* k : {"ghz", "ghzrev", "chainfan", "chainfanrev", "redund", "star"}) family_tripwire(k, 4000);
    for (const char* k : {"ghz", "chainfan"}) family_tripwire(k, 8000);
    // fix 3: the detector count (n, 20n, 100n on the fan pairs) and the repeat count K = 2..6
    for (const char* k : {"cfx0", "cfx20", "cfx100"}) family_tripwire(k, 2000);
    for (const char* k : {"redk2", "redk3", "redk4", "redk5", "redk6"}) family_tripwire(k, 2000);
    // fix 4 (rev4 F1): the time decision on chain + fan with weight-4 fan parities and on the GHZ chain measured twice.
    // With the fix-4 bitmap scans both forward routes cost a fraction of prref's (the report: 0.3x and 0.02x), so both
    // keep the forward form, within the budget; at n = 1000 with 5n parities the priced forward units pass the budget
    // before the end (the measured forward / prref time there is 0.77, the price kFwdUnitWords errs high), so the rest
    // take the prref route — the other side of the decision.
    family_tripwire("qfx32", 2000, "kept");
    family_tripwire("redund", 8000, "kept");
    family_tripwire("qfx35", 1000, "spent");
    // fix 4 (rev4 F2 / F3): the stored bytes by capacity
    family_bytes("ghzl20", 2000, 1.0);
    family_bytes("rkf40", 4000, 0.25);
}

void test_corpus() {
    namespace fs = std::filesystem;
    const std::string root = XTIM_TESTS_DIR;
    int files = 0, checked = 0;
    for (const std::string& dir : {root + "/data", root + "/../examples", root + "/private"}) {
        if (!fs::exists(dir)) continue;
        std::vector<std::string> paths;
        for (const auto& e : fs::recursive_directory_iterator(dir))
            if (e.is_regular_file() && e.path().extension() == ".stim") paths.push_back(e.path().string());
        std::sort(paths.begin(), paths.end());
        for (const std::string& p : paths) {
            ++files;
            if (corpus_one(slurp(p))) ++checked;
        }
    }
    std::string chain;
    for (int q = 0; q < 300; ++q)
        chain += "R " + std::to_string(q) + "\nX_ERROR(0.01) " + std::to_string(q) + "\nM " + std::to_string(q) +
                 "\nDETECTOR rec[-1]\n";
    CHECK(corpus_one(chain));
    ++checked;
    CHECK(checked >= 20);
    std::fprintf(stderr, "[planes_build] corpus: %d of %d files + chain-300 (%ld probes, %ld in group; %ld sets overall; "
                 "%ld planes provenance checks, %ld unique; %ld reduce_sparse checks)\n",
                 checked - 1, files, probes, probes_in, planes_sets, prov_checked, prov_unique, rs_checked);
    CHECK(prov_unique > 1000);
}

}  // namespace

int main() {
    RUN(test_random);
    RUN(test_corpus);
    RUN(test_families);
    REPORT();
}
