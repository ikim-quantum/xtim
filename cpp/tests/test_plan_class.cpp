// test_plan_class — the K2' kernel compiler (qeccore/plan_class.hpp; plancache-design/plan.md §0-§1, §6 O3) against
// an INDEPENDENT exact oracle on random instances.
//
// Oracle: for a pair set ΔB, the pullback Q_c = C0† M_c C0 of every record column by C0 = CZ^ΔB is taken from
// diag_pullback (record_table.hpp: the readers' own exact-phase pullback), g_c = M_c · Q_c, and T3's conditions are
//   (i)  G.reduce(g_c) is IN_GROUP with sign +1 (g_c a +1-signed certified element),
//   (ii) g_c commutes with every record column M_c' (⟨Z^{ΔB x_c}, M_c'⟩ = x_c^T ΔB x_c': the β form of plan §0).
// The compiler's kernel (the span of its basis) must equal the oracle's kernel SET, enumerated over every ΔB of the
// kept pairs; Π must be linear, idempotent, land in the kernel coset and clear every pivot bit. Instances where (ii)
// binds (ANTI-class records) are counted for non-vacuity: a build with XTIM_MUTATE_K2P_DROP_II (condition (ii)
// dropped from the elimination) is RED here.
// Must end main() with REPORT().
#include "check.hpp"

#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "qeccore/pauli.hpp"
#include "qeccore/plan_class.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_planes.hpp"

using namespace qeccore;

namespace {

struct Instance {
    int n = 0;
    CertifiedGroupPlanes G;
    std::vector<RecColumn> cols;
    std::vector<std::pair<int, int>> upairs;
    std::vector<uint64_t> keep, prodmask;
};

int xz_overlap(const Pauli& P) {
    int o = 0;
    for (size_t w = 0; w < P.x.size(); ++w) o += __builtin_popcountll(P.x[w] & P.z[w]);
    return o;
}

Pauli random_pauli(std::mt19937_64& rng, int n, bool pure_z) {
    Pauli P(n);
    for (int q = 0; q < n; ++q) {
        const int t = (int)(rng() % 4);
        if (pure_z) { if (t & 1) P.setz(q); continue; }
        if (t & 1) P.setx(q);
        if (t & 2) P.setz(q);
    }
    P.phase = (xz_overlap(P) & 1) + 2 * (int)(rng() % 2);   // Hermitian: phase − |x∧z| even
    return P;
}

Instance make_instance(std::mt19937_64& rng, int n, int m) {
    Instance I;
    I.n = n;
    // INDEPENDENT commuting generators (a dependent set with inconsistent signs would hold −I: not a stabilizer
    // group): a candidate joins only when it raises the rank
    std::vector<Pauli> gens;
    auto try_add = [&](Pauli g) {
        for (const Pauli& h : gens) if (Pauli::anticommute_bit(g, h)) return;
        std::vector<Pauli> t = gens;
        t.push_back(g);
        if (CertifiedGroupPlanes::build(t, n).rank != (int)t.size()) return;
        gens = std::move(t);
    };
    const int nz = 1 + (int)(rng() % n);
    for (int i = 0; i < nz; ++i) try_add(random_pauli(rng, n, true));
    for (int t = 0; t < 2 * n && gens.size() < (size_t)n; ++t) {   // a few X-type generators (pure X commute)
        Pauli g(n);
        for (int q = 0; q < n; ++q) if (rng() % 2) g.setx(q);
        g.phase = 2 * (int)(rng() % 2);
        try_add(g);
    }
    if (gens.empty()) { Pauli g(n); g.setz(0); gens.push_back(g); }
    I.G = CertifiedGroupPlanes::build(gens, n);
    for (int c = 0; c < m; ++c) {
        RecColumn col;
        col.M = random_pauli(rng, n, false);
        col.order = c;
        I.cols.push_back(col);
    }
    for (int j = 0; j < n; ++j)
        for (int l = j + 1; l < n; ++l) I.upairs.emplace_back(j, l);
    const int NW = (n + 63) / 64, PW = (int)((I.upairs.size() + 63) / 64);
    I.prodmask.assign((size_t)NW, 0);
    const CertifiedGroupPlanes::ZAxisRREF& Z = I.G.z_axis_rref();
    for (int q = 0; q < n; ++q) if (Z.prod_sign[(size_t)q]) I.prodmask[(size_t)(q >> 6)] |= 1ULL << (q & 63);
    I.keep.assign((size_t)PW, 0);
    for (size_t i = 0; i < I.upairs.size(); ++i)
        if (!Z.prod_sign[(size_t)I.upairs[i].first] && !Z.prod_sign[(size_t)I.upairs[i].second])
            I.keep[i >> 6] |= 1ULL << (i & 63);
    return I;
}

// The oracle: (i) ∧ (ii) for ΔB; `ii_only_fail` reports an instance where (i) holds and (ii) fails.
bool oracle_member(const Instance& I, const std::vector<uint64_t>& dB, bool* ii_binds) {
    DiagNormalForm nf;
    nf.prefix = Pauli(I.n);
    nf.a.assign((size_t)I.n, 0);
    nf.diagonal_class = true;
    for (size_t i = 0; i < I.upairs.size(); ++i)
        if ((dB[i >> 6] >> (i & 63)) & 1ULL) nf.cz.push_back(I.upairs[i]);
    bool i_ok = true, ii_ok = true;
    for (const RecColumn& c : I.cols) {
        const Pauli Q = diag_pullback(nf, c.M);
        const Pauli g = Pauli::multiply(c.M, Q);        // M_c² = I: g_c = M_c · (M_c g_c)
        const Membership mm = I.G.reduce(g);
        if (mm.verdict != Membership::IN_GROUP || mm.sign != +1) i_ok = false;
        for (const RecColumn& c2 : I.cols)
            if (Pauli::anticommute_bit(g, c2.M)) ii_ok = false;
    }
    if (ii_binds) *ii_binds = i_ok && !ii_ok;
    return i_ok && ii_ok;
}

std::string key_of(const std::vector<uint64_t>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(uint64_t));
}

void test_random_instances() {
    std::mt19937_64 rng(20261101);
    long instances = 0, ii_binding_instances = 0, nontrivial = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const int n = 2 + (int)(rng() % 5);                // 2..6 wires: ≤ 15 pairs, ≤ 2^15 pair sets
        const int m = 1 + (int)(rng() % 4);
        Instance I = make_instance(rng, n, m);
        const int NW = (n + 63) / 64, PW = (int)((I.upairs.size() + 63) / 64);
        const PlanClassifier pc = compile_plan_classifier(I.G, I.cols, I.upairs, I.keep, I.prodmask, n, /*audit=*/true);
        CHECK(pc.cap.empty());
        CHECK(pc.NW == NW && pc.PW == PW);
        // the oracle's kernel set over every subset of the kept pairs
        std::vector<int> kept;
        for (size_t i = 0; i < I.upairs.size(); ++i) if ((I.keep[i >> 6] >> (i & 63)) & 1ULL) kept.push_back((int)i);
        std::set<std::string> oracle;
        bool any_ii = false;
        for (uint64_t s = 0; s < (1ULL << kept.size()); ++s) {
            std::vector<uint64_t> dB((size_t)PW, 0);
            for (size_t i = 0; i < kept.size(); ++i)
                if ((s >> i) & 1ULL) dB[(size_t)(kept[i] >> 6)] |= 1ULL << (kept[i] & 63);
            bool ii_binds = false;
            if (oracle_member(I, dB, &ii_binds)) oracle.insert(key_of(dB));
            any_ii = any_ii || ii_binds;
        }
        // the compiler's kernel: the span of its basis
        std::set<std::string> span;
        for (uint64_t s = 0; s < (1ULL << pc.dim()); ++s) {
            std::vector<uint64_t> v((size_t)PW, 0);
            for (size_t i = 0; i < pc.dim(); ++i)
                if ((s >> i) & 1ULL)
                    for (int w = 0; w < PW; ++w) v[(size_t)w] ^= pc.basis[i * (size_t)PW + (size_t)w];
            span.insert(key_of(v));
        }
        if (span != oracle) {
            std::fprintf(stderr, "trial %d: n=%d m=%d kept=%zu dim=%zu span=%zu oracle=%zu\n", trial, n, m,
                         kept.size(), pc.dim(), span.size(), oracle.size());
            CHECK(span == oracle);
        }
        CHECK(pc.audited == pc.dim());
        // Π: linear, idempotent, into the kernel coset, pivots cleared; the a-part masked by U ∧ ¬prod
        for (int t = 0; t < 16; ++t) {
            std::vector<uint64_t> a((size_t)pc.KW, 0), b((size_t)pc.KW, 0), ab((size_t)pc.KW, 0), pa, pb, pab;
            for (int w = 0; w < pc.KW; ++w) { a[(size_t)w] = rng(); b[(size_t)w] = rng(); }
            for (size_t i = 0; i < I.upairs.size(); ++i)
                if (!((I.keep[i >> 6] >> (i & 63)) & 1ULL)) {
                    a[(size_t)NW + (i >> 6)] &= ~(1ULL << (i & 63));
                    b[(size_t)NW + (i >> 6)] &= ~(1ULL << (i & 63));
                }
            for (int w = NW; w < pc.KW; ++w) {
                const size_t bits = I.upairs.size() - (size_t)(w - NW) * 64;
                if (bits < 64) { a[(size_t)w] &= (1ULL << bits) - 1; b[(size_t)w] &= (1ULL << bits) - 1; }
            }
            if (n < 64) { a[0] &= (1ULL << n) - 1; b[0] &= (1ULL << n) - 1; }
            for (int w = 0; w < pc.KW; ++w) ab[(size_t)w] = a[(size_t)w] ^ b[(size_t)w];
            pa = a; pb = b; pab = ab;
            pc.project(pa.data()); pc.project(pb.data()); pc.project(pab.data());
            std::vector<uint64_t> ppa = pa;
            pc.project(ppa.data());
            CHECK(ppa == pa);
            for (int w = 0; w < pc.KW; ++w) CHECK(pab[(size_t)w] == (pa[(size_t)w] ^ pb[(size_t)w]));
            CHECK(pc.is_representative(pa.data()));
            for (int w = 0; w < NW; ++w) CHECK((pa[(size_t)w] & ~pc.umask[(size_t)w]) == 0);
            std::vector<uint64_t> d((size_t)PW, 0);
            for (int w = 0; w < PW; ++w) d[(size_t)w] = a[(size_t)NW + w] ^ pa[(size_t)NW + w];
            CHECK(oracle.count(key_of(d)) == 1);           // a ⊕ Π(a) ∈ K2'
        }
        ++instances;
        if (any_ii) ++ii_binding_instances;
        if (pc.dim() > 0) ++nontrivial;
    }
    std::fprintf(stderr, "instances %ld, with (ii) binding %ld, nontrivial kernels %ld\n", instances,
                 ii_binding_instances, nontrivial);
    CHECK(ii_binding_instances > 0);                       // non-vacuity of (ii) (the DROP-II mutation is RED above)
    CHECK(nontrivial > 0);
}

void test_degenerate_and_cap() {
    std::mt19937_64 rng(7);
    Instance I = make_instance(rng, 4, 0);                 // no record column: every kept pair is in the kernel
    const PlanClassifier pc = compile_plan_classifier(I.G, I.cols, I.upairs, I.keep, I.prodmask, 4, true);
    size_t kept = 0;
    for (size_t i = 0; i < I.upairs.size(); ++i) if ((I.keep[i >> 6] >> (i & 63)) & 1ULL) ++kept;
    if (kept) CHECK(pc.dim() == kept && !pc.pi_identity);   // every kept pair is in the kernel
    else CHECK(pc.pi_identity && pc.dim() == 0);              // (no kept pair: nothing to project)
    const PlanClassifier id = compile_plan_classifier(I.G, I.cols, I.upairs, I.keep, I.prodmask, 4, true, true);
    CHECK(id.pi_identity && id.cap == "identity" && id.dim() == 0);
    Instance J = make_instance(rng, 5, 3);
    const PlanClassifier pj = compile_plan_classifier(J.G, J.cols, J.upairs, J.keep, J.prodmask, 5, true);
    CHECK(pj.columns > 0 && pj.kept_pairs > 0);
    CHECK(pj.work == (uint64_t)pj.kept_pairs * pj.kept_pairs * ((pj.columns + 63) / 64));
}

}  // namespace

int main() {
    RUN(test_random_instances);
    RUN(test_degenerate_and_cap);
    REPORT();
}
