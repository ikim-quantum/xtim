// test_prepared_affine.cpp — byte-identity of the sparse PreparedAffine and the set-bit
// stabilizer_generators / Pauli::apply_to.
//
// The production PreparedAffine::prepare (stab_overlap.cpp) runs the RREF over R's k columns and
// keeps each row's row-op combination in k-bit SLOT space (G_θ on the pivot qubits, H_θ rows as
// e_{hq[i]} ⊕ slot combination). The pre-sparse dense RREF of [R | I_n] is kept here verbatim as
// the ORACLE (plus the row-origin permutation it applies, to name its pivot order), together with
// the pre-sparse stabilizer_generators (per-qubit loops, n-wide G/H, b-diff rhs, full scratch
// restore, a0 through amplitude_at_bits). On random AffineStates of varied n, k and density
// (n straddling word boundaries, k from 0 to n, nonzero b/D/J/omega, Paulis with phases) the
// expanded G_θ, H_θ, c_θ, the pivot order and H row order, and every generator (x, z, phase) must
// be EXACTLY the oracle's; a rank-deficient R must throw the same message.
#include "check.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "qeccore/stab_affine.hpp"
#include "qeccore/stab_generators.hpp"
#include "qeccore/stab_overlap.hpp"
#include "stab_generators_oracle.hpp"

using namespace qeccore;

namespace {

// ---- ORACLE: the pre-sparse dense prepare, verbatim (+ the origin permutation `orig`) ----------
struct DensePrepared {
    int n_ = 0, kth_ = 0, m_ = 0, mw_ = 0;
    std::vector<uint64_t> Hth, cth, Gth;
    std::vector<int> orig;        // row position -> qubit it started as (oracle-only bookkeeping)
    const uint64_t* Hrow(int i) const { return Hth.data() + (size_t)i * mw_; }
    const uint64_t* Grow(int i) const { return Gth.data() + (size_t)i * mw_; }
};

DensePrepared dense_prepare(const AffineState& theta) {
    DensePrepared pa;
    const int n = theta.n_;
    const int k = theta.k_;
    pa.n_ = n;
    pa.kth_ = k;
    pa.m_ = n - k;
    const int mw = (n + 63) / 64;
    pa.mw_ = mw;
    const int ncols = k + n;
    const int W = (ncols + 63) / 64;
    std::vector<uint64_t> A((size_t)n * W, 0);
    auto Arow = [&](int i) -> uint64_t* { return A.data() + (size_t)i * W; };
    const int kw = (k + 63) / 64;
    pa.orig.resize(n);
    for (int i = 0; i < n; ++i) {
        uint64_t* r = Arow(i);
        const uint64_t* rr = theta.R.row(i);
        for (int t = 0; t < kw && t < theta.R.words_; ++t) r[t] = rr[t];
        int c = k + i;
        r[c >> 6] |= (1ULL << (c & 63));
        pa.orig[i] = i;
    }
    int prow = 0;
    for (int col = 0; col < k && prow < n; ++col) {
        const int cw = col >> 6;
        const uint64_t cbit = 1ULL << (col & 63);
        int sel = -1;
        for (int i = prow; i < n; ++i)
            if (Arow(i)[cw] & cbit) { sel = i; break; }
        if (sel < 0) continue;
        if (sel != prow) {
            uint64_t* a = Arow(prow);
            uint64_t* b = Arow(sel);
            for (int w = 0; w < W; ++w) std::swap(a[w], b[w]);
            std::swap(pa.orig[prow], pa.orig[sel]);
        }
        const uint64_t* pr = Arow(prow);
        for (int i = 0; i < n; ++i) {
            if (i == prow) continue;
            uint64_t* ri = Arow(i);
            if (ri[cw] & cbit)
                for (int w = 0; w < W; ++w) ri[w] ^= pr[w];
        }
        ++prow;
    }
    if (prow != k)
        throw std::logic_error(
            "PreparedAffine::prepare: R is rank-deficient (rank " + std::to_string(prow) +
            " < k = " + std::to_string(k) + ") — state is not in canonical form");
    pa.Gth.assign((size_t)k * mw, 0);
    pa.Hth.assign((size_t)pa.m_ * mw, 0);
    auto copy_Iblock = [&](int srcrow, uint64_t* dst) {
        const uint64_t* r = Arow(srcrow);
        const int off = k >> 6, sh = k & 63;
        for (int w = 0; w < mw; ++w) {
            const uint64_t lo = r[off + w] >> sh;
            const uint64_t hi = (sh && off + w + 1 < W) ? (r[off + w + 1] << (64 - sh)) : 0ull;
            dst[w] = lo | hi;
        }
    };
    for (int i = 0; i < k; ++i) copy_Iblock(i, pa.Gth.data() + (size_t)i * mw);
    for (int i = 0; i < pa.m_; ++i) copy_Iblock(k + i, pa.Hth.data() + (size_t)i * mw);
    std::vector<uint64_t> bpk(mw, 0);
    for (int i = 0; i < n; ++i) if (theta.b[i]) bpk[i >> 6] |= (1ULL << (i & 63));
    pa.cth.assign(mw, 0);
    for (int i = 0; i < pa.m_; ++i) {
        const uint64_t* hr = pa.Hrow(i);
        int par = 0;
        for (int w = 0; w < mw; ++w) par ^= __builtin_popcountll(hr[w] & bpk[w]);
        if (par & 1) pa.cth[i >> 6] |= (1ULL << (i & 63));
    }
    return pa;
}

// ---- ORACLE: the pre-set-bit Pauli::apply_to and stabilizer_generators, verbatim ---------------
void old_apply_to(const Pauli& p, StabState& s) {
    for (int q = 0; q < p.n; ++q) if (p.zbit(q)) s.apply_z(q);
    for (int q = 0; q < p.n; ++q) if (p.xbit(q)) s.apply_x(q);
    if (p.phase % 4 != 0) {
        auto* a = dynamic_cast<AffineState*>(&s);
        a->omega = a->omega.mul(ExactPhase::zeta8((2 * p.phase) % 8));
    }
}

struct OldPin {
    const AffineState& psi;
    const DensePrepared& pa;
    AffineState scratch;
    ExactPhase a0;
    std::vector<uint64_t> xw, yw;
    std::vector<int> ysupp, xwords;
    OldPin(const AffineState& s, const DensePrepared& p)
        : psi(s), pa(p), scratch(s), a0(s.amplitude_at_bits(s.b)),
          xw((size_t)p.mw_, 0), yw(((size_t)s.k_ + 63) / 64, 0) {}
    void pin(Pauli& g) {
        scratch.b = psi.b;
        scratch.D = psi.D;
        scratch.omega = psi.omega;
        old_apply_to(g, scratch);
        std::fill(xw.begin(), xw.end(), 0);
        for (int q = 0; q < psi.n_; ++q)
            if (psi.b[q] ^ scratch.b[q]) xw[q >> 6] |= 1ull << (q & 63);
        xwords.clear();
        for (int t = 0; t < pa.mw_; ++t) if (xw[(size_t)t]) xwords.push_back(t);
        for (int i = 0; i < pa.m_; ++i) {
            const uint64_t* hr = pa.Hrow(i);
            uint64_t acc = 0;
            for (int t : xwords) acc ^= hr[t] & xw[(size_t)t];
            if (__builtin_parityll(acc))
                throw std::logic_error("stabilizer_generators: candidate has wrong X/Z support "
                                       "(rhs off the state's affine support): n=" + std::to_string(psi.n_) +
                                       " k=" + std::to_string(psi.k_) + " check_row=" + std::to_string(i));
        }
        std::fill(yw.begin(), yw.end(), 0);
        ysupp.clear();
        for (int j = 0; j < psi.k_; ++j) {
            const uint64_t* gr = pa.Grow(j);
            uint64_t acc = 0;
            for (int t : xwords) acc ^= gr[t] & xw[(size_t)t];
            if (__builtin_parityll(acc)) { yw[j >> 6] |= 1ull << (j & 63); ysupp.push_back(j); }
        }
        long long q4 = 0;
        for (int j : ysupp) q4 += scratch.D[j];
        const int tw = std::min(psi.J.words(), (int)yw.size());
        for (int a : ysupp) {
            const uint64_t* jr = psi.J.row(a);
            for (int t = 0; t < tw; ++t) q4 += __builtin_popcountll(jr[t] & yw[t]);
        }
        const int c0 = (int)(((q4 % 4) + 4) % 4);
        ExactPhase a1 = scratch.omega;
        a1.add_z8((2 * c0) % 8);
        a1.scale -= psi.k_;
        if (a0.is_zero || a1.is_zero || a0.scale != a1.scale)
            throw std::logic_error("stabilizer_generators: candidate has wrong X/Z support (|<psi|g|psi>| != 1)");
        const int dz = ((a1.z8 - a0.z8) % 8 + 8) % 8;
        if (dz & 1) throw std::logic_error("stabilizer_generators: candidate eigenphase is not a 4th root");
        g.phase = (4 - dz / 2) % 4;
    }
};

std::vector<Pauli> old_generators(const AffineState& s) {
    const int n = s.n_, k = s.k_;
    DensePrepared pa = dense_prepare(s);
    OldPin pin(s, pa);
    std::vector<Pauli> gens;
    for (int i = 0; i < pa.m_; ++i) {
        Pauli g(n);
        const uint64_t* hr = pa.Hrow(i);
        for (int q = 0; q < n; ++q)
            if ((hr[q >> 6] >> (q & 63)) & 1ULL) g.setz(q);
        pin.pin(g);
        gens.push_back(std::move(g));
    }
    auto xor_grow_into = [&](Pauli& g, int j) {
        const uint64_t* gr = pa.Grow(j);
        for (int q = 0; q < n; ++q)
            if ((gr[q >> 6] >> (q & 63)) & 1ULL) g.flipz(q);
    };
    for (int l = 0; l < k; ++l) {
        Pauli g(n);
        for (int q = 0; q < n; ++q)
            if (s.R.get(q, l)) g.setx(q);
        for (int j = 0; j < k; ++j)
            if (s.J.get(l, j)) xor_grow_into(g, j);
        if (s.D[l] & 1) xor_grow_into(g, l);
        pin.pin(g);
        gens.push_back(std::move(g));
    }
    return gens;
}

// ---- helpers ----------------------------------------------------------------------------------
// Expand a slot-space row (k bits, bit s ↔ qubit piv[s]) plus an optional extra qubit to n bits.
std::vector<uint64_t> expand(const PreparedAffine& pa, const uint64_t* r, int extra) {
    std::vector<uint64_t> d((size_t)pa.mw_, 0);
    for (int s = 0; s < pa.kth_; ++s)
        if ((r[s >> 6] >> (s & 63)) & 1ULL) d[pa.piv[s] >> 6] ^= 1ULL << (pa.piv[s] & 63);
    if (extra >= 0) d[extra >> 6] ^= 1ULL << (extra & 63);
    return d;
}

// Random state: |0^n> then a random Clifford word; hfrac steers k, ncx the density of R.
AffineState random_state(std::mt19937_64& rng, int n, int ngates, double hfrac, double cxfrac) {
    AffineState s(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < ngates; ++i) {
        const int a = (int)(rng() % n);
        int b = (int)(rng() % n);
        if (n > 1) while (b == a) b = (int)(rng() % n);
        const double r = u(rng);
        if (r < hfrac) { s.apply_h(a); continue; }
        if (r < hfrac + cxfrac) { if (n > 1) s.apply_cx(a, b); continue; }
        switch ((int)(rng() % 6)) {
            case 0: s.apply_s(a); break;
            case 1: s.apply_sdg(a); break;
            case 2: s.apply_x(a); break;
            case 3: s.apply_y(a); break;
            case 4: s.apply_z(a); break;
            default: if (n > 1) s.apply_cz(a, b); break;
        }
    }
    if (rng() % 3 == 0) s.omega.add_z8((int)(rng() % 8));
    return s;
}

bool same_pauli(const Pauli& a, const Pauli& b) {
    return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z;
}

void fuzz_prepare_and_generators() {
    std::mt19937_64 rng(0x7e3ULL);
    const int ns[] = {1, 2, 3, 7, 63, 64, 65, 100, 127, 128, 129, 200, 300};
    long cases = 0, gens_checked = 0, sparse_checked = 0, k0 = 0, kfull = 0, multiword_k = 0, odd_phase = 0;
    for (int n : ns) {
        for (int rep = 0; rep < 16; ++rep) {
            const double hfrac = (rep % 4 == 0) ? 0.9 : (rep % 4 == 1 ? 0.4 : (rep % 4 == 2 ? 0.1 : 0.0));
            const double cxfrac = (rep % 3 == 0) ? 0.05 : (rep % 3 == 1 ? 0.3 : 0.6);
            const int ngates = (int)(n * (1 + rep % 5)) + 2;
            AffineState s = random_state(rng, n, ngates, hfrac, cxfrac);
            const int k = s.k_;
            DensePrepared o = dense_prepare(s);
            PreparedAffine p = PreparedAffine::prepare(s);
            CHECK_EQ(p.n_, n);
            CHECK_EQ(p.kth_, k);
            CHECK_EQ(p.m_, o.m_);
            CHECK_EQ(p.mw_, o.mw_);
            // pivot order and H row order = the dense RREF's row permutation
            for (int j = 0; j < k; ++j) CHECK_EQ(p.piv[j], o.orig[j]);
            for (int i = 0; i < p.m_; ++i) CHECK_EQ(p.hq[i], o.orig[k + i]);
            for (int j = 0; j < k; ++j) CHECK(expand(p, p.Grow(j), -1) ==
                std::vector<uint64_t>(o.Grow(j), o.Grow(j) + o.mw_));
            for (int i = 0; i < p.m_; ++i) CHECK(expand(p, p.Hrow(i), p.hq[i]) ==
                std::vector<uint64_t>(o.Hrow(i), o.Hrow(i) + o.mw_));
            CHECK(p.cth == o.cth);
            for (int q = 0; q < n; ++q) {
                const int sl = p.slot_of[q];
                CHECK(sl < 0 || p.piv[sl] == q);
            }
            // generators, pinned phases included
            auto gn = stabilizer_generators(s);
            auto go = old_generators(s);
            CHECK_EQ(gn.size(), go.size());
            if (gn.size() == go.size())
                for (size_t i = 0; i < gn.size(); ++i) {
                    CHECK(same_pauli(gn[i], go[i]));
                    odd_phase += gn[i].phase != 0;
                }
            // QW-F: the support-recording emitter (dense and sparse output) == the pre-QW-F full-width emitter.
            {
                const auto gq = qwf_oracle::stabilizer_generators(s);
                const auto gs = stabilizer_generators_sparse(s);
                const auto gqs = qwf_oracle::stabilizer_generators_sparse(s);
                CHECK_EQ(gq.size(), gn.size());
                CHECK_EQ(gs.size(), gn.size());
                CHECK_EQ(gqs.size(), gn.size());
                for (size_t i = 0; i < gn.size() && i < gq.size(); ++i) CHECK(same_pauli(gn[i], gq[i]));
                for (size_t i = 0; i < gs.size() && i < gqs.size() && i < gq.size(); ++i) {
                    CHECK(gs[i].n == gqs[i].n && gs[i].phase == gqs[i].phase && gs[i].x == gqs[i].x && gs[i].z == gqs[i].z);
                    CHECK(same_pauli(to_dense(gs[i]), gq[i]));
                }
                sparse_checked += (long)gs.size();
            }
            gens_checked += (long)gn.size();
            ++cases;
            k0 += k == 0;
            kfull += k == n;
            multiword_k += k > 64;
        }
    }
    CHECK(k0 > 3);
    CHECK(kfull > 3);
    CHECK(multiword_k > 5);
    CHECK(odd_phase > 100);
    CHECK(sparse_checked == gens_checked);
    std::fprintf(stderr, "  %ld states, %ld generators (%ld nonzero phases, %ld sparse vs the pre-QW-F emitter), "
                 "k=0 %ld, k=n %ld, k>64 %ld\n", cases, gens_checked, odd_phase, sparse_checked, k0, kfull, multiword_k);
}

// Pauli::apply_to on set bits = the per-qubit scans: same b, D, J, omega on random states/Paulis.
void fuzz_apply_to() {
    std::mt19937_64 rng(0xa991ULL);
    long cases = 0;
    for (int n : {1, 5, 64, 65, 130}) {
        for (int rep = 0; rep < 40; ++rep) {
            AffineState s = random_state(rng, n, 3 * n, 0.4, 0.3);
            Pauli p(n);
            const double dens = (rep % 4 == 0) ? 0.02 : (rep % 4 == 1 ? 0.2 : 0.6);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            for (int q = 0; q < n; ++q) { if (u(rng) < dens) p.setx(q); if (u(rng) < dens) p.setz(q); }
            p.phase = (int)(rng() % 4);
            AffineState a = s, b = s;
            p.apply_to(a);
            old_apply_to(p, b);
            CHECK(a.b == b.b);
            CHECK(a.D == b.D);
            CHECK(a.omega.exact_eq(b.omega));
            ++cases;
        }
    }
    std::fprintf(stderr, "  %ld apply_to cases\n", cases);
}

// A rank-deficient R (written directly) throws the oracle's message.
void rank_deficient_throws() {
    AffineState s(5);
    s.R.reserve_words(1);
    s.R.cols_ = 3;
    s.k_ = 3;
    s.D.assign(3, 0);
    for (int i = 0; i < 3; ++i) s.J.append_zero();
    s.R.set(0, 0, 1); s.R.set(1, 1, 1); s.R.set(0, 2, 1); s.R.set(1, 2, 1);   // col2 = col0 + col1
    std::string mo, mn;
    try { dense_prepare(s); } catch (const std::logic_error& e) { mo = e.what(); }
    try { PreparedAffine::prepare(s); } catch (const std::logic_error& e) { mn = e.what(); }
    CHECK(!mo.empty());
    CHECK(mo == mn);
}

}  // namespace

int main() {
    RUN(fuzz_prepare_and_generators);
    RUN(fuzz_apply_to);
    RUN(rank_deficient_throws);
    REPORT();
}
