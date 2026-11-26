// fault_record_law — the ONE stabilizer-twirl record law (see qeccore/fault_record_law.hpp).
#include "qeccore/fault_record_law.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "qeccore/record_space_b.hpp"

namespace qeccore {

bool fault_record_law(const CertifiedGroupPlanes& G, const FramedSuperposition& bare, const DiagNormalForm& nf,
                      const std::vector<RecColumn>& cols, RecTable& out) {
    return build_record_table_recB(G, bare, nf, cols, out);
}

std::vector<double> born_law(const FramedSuperposition& bare, const std::vector<FreeDir>& lam, long* evals,
                             BornLawStats* stats) {
    const int kappa = (int)lam.size();
    if (kappa > 62) throw std::logic_error("born_law: more Born directions than free coordinates (engine bug)");
    std::vector<double> p((size_t)1 << kappa, 0.0);
    if (kappa == 0) { p[0] = 1.0; return p; }
    std::shared_ptr<const CosetLawTable> T;
    born_law_enumerate(bare, lam, nullptr, 0.0, [&](uint64_t z, double w, double) { p[(size_t)z] += w; return true; },
                       evals, &T);
    if (stats && T) { ++stats->coset_laws; stats->coset_work += T->work; }
    return p;
}

BornQuotient born_quotient(const std::vector<std::vector<uint64_t>>& W, const std::vector<std::vector<uint64_t>>& B,
                           const std::vector<double>& pz, int UW) {
    auto lead = [](const std::vector<uint64_t>& x) {
        for (size_t w = 0; w < x.size(); ++w) if (x[w]) return (int)(w * 64) + __builtin_ctzll(x[w]);
        return -1;
    };
    auto bit = [](const std::vector<uint64_t>& x, int b) { return (x[(size_t)(b >> 6)] >> (b & 63)) & 1ULL; };
    auto xr = [](std::vector<uint64_t>& x, const std::vector<uint64_t>& y) { for (size_t w = 0; w < x.size(); ++w) x[w] ^= y[w]; };
    std::vector<std::vector<uint64_t>> ech;              // span(W), fully reduced (lowest-bit pivots)
    auto reduce = [&](std::vector<uint64_t>& x) { for (const auto& r : ech) if (bit(x, lead(r))) xr(x, r); };
    auto insert = [&](std::vector<uint64_t> x) {
        reduce(x);
        const int pv = lead(x);
        if (pv < 0) return false;
        for (auto& r : ech) if (bit(r, pv)) xr(r, x);
        ech.push_back(std::move(x));
        return true;
    };
    for (const auto& w : W) insert(w);
    double pmax = 0.0;
    for (double x : pz) pmax = std::max(pmax, x);
    std::map<std::vector<uint64_t>, double> cls;
    const int kappa = (int)B.size();
    for (size_t z = 0; z < pz.size(); ++z) {
        if (!(pz[z] > 1e-12 * pmax)) continue;
        std::vector<uint64_t> off((size_t)UW, 0);
        for (int j = 0; j < kappa; ++j) if ((z >> j) & 1) xr(off, B[(size_t)j]);
        reduce(off);
        cls[off] += pz[z];
    }
    BornQuotient q;
    q.r0 = cls.begin()->first;
    for (const auto& [r, w] : cls) {                     // the class directions (independent mod W)
        std::vector<uint64_t> d = r;
        xr(d, q.r0);
        std::vector<uint64_t> d0 = d;
        if (insert(std::move(d))) q.dirs.push_back(std::move(d0));
    }
    double wmax = 0.0, wmin = 2.0;
    for (const auto& [r, w] : cls) { wmax = std::max(wmax, w); wmin = std::min(wmin, w); }
    q.coset = ((size_t)1 << q.dirs.size()) == cls.size() && wmax - wmin <= 1e-12 * wmax;
    if (!q.coset) for (const auto& [r, w] : cls) q.cls.push_back({r, w});
    return q;
}

RecordColumns::RecordColumns(const CertifiedGroupPlanes& G) : G_(G) {
    // F3 (S2-T6): the masks come from the planes' own provenance, as the sampler's circuit channels
    // (twirl_sampler.cpp) — no second dense provenance RREF (n'·(2·NW + GW) words, O(ng²) to build) any more.
    if (G.rank != G.n_gens) throw std::logic_error("RecordColumns: dependent certified generators (engine bug)");
}

RecColumn RecordColumns::column(const Pauli& M, int order) const {
    RecColumn C;
    C.M = from_dense(M);
    C.order = order;
    int phase = 0;
    std::vector<int32_t> mask;
    if (G_.reduce_sparse(C.M, phase, &mask, sc_) != Membership::IN_GROUP) return C;   // an extra (LOGICAL / ANTI)
    if (phase & 1) throw std::logic_error("RecordColumns: non-Hermitian IN_GROUP reduction (engine bug)");
    C.mask = GenMask(std::move(mask));                   // fix 4: the cheaper of the list and the bitset
    C.channel = true;
    C.ref = (uint8_t)((phase >> 1) & 1);
    return C;
}

}  // namespace qeccore
